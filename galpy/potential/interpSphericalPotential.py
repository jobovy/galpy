###################3###################3###################3##################
# interpSphericalPotential.py: build spherical potential through interpolation
###################3###################3###################3##################
import numpy
from scipy import interpolate

from ..backend import asarray_on_device, device_of, get_namespace, match_input_dtype
from ..util._optional_deps import _JAX_LOADED
from ..util.conversion import get_physical, physical_compatible
from .Potential import _evaluatePotentials, _evaluateRforces
from .SphericalPotential import SphericalPotential

if _JAX_LOADED:
    import jax.numpy as jnp


def _spline_to_ppoly_data(spl):
    """Convert a FITPACK spline to de-duplicated piecewise-power coefficients.

    Returns ``(x, c)`` with ``x`` the distinct breakpoints (shape ``(m+1,)``)
    and ``c`` the power-basis coefficients (shape ``(k+1, m)``) such that on
    ``x[i] <= r < x[i+1]`` the spline is ``sum_j c[j, i] * (r - x[i])**(k-j)``.
    This is the exact piecewise-polynomial representation of the spline (scipy's
    ``PPoly.from_spline``), with the zero-width intervals coming from FITPACK's
    repeated boundary knots dropped so that interval lookup by ``searchsorted``
    is unambiguous. Called once at setup (init-time numpy/scipy is fine); the
    coefficients then feed the backend-agnostic ``_ppoly_eval`` below.
    """
    ppoly = interpolate.PPoly.from_spline(spl._eval_args)
    keep = numpy.diff(ppoly.x) > 0.0
    return numpy.append(ppoly.x[:-1][keep], ppoly.x[-1]), ppoly.c[:, keep]


def _ppoly_eval(xp, x, c, r):
    """Evaluate a piecewise polynomial in the power basis at ``r``.

    ``(x, c)`` are as returned by ``_spline_to_ppoly_data``; the evaluation
    (interval lookup by ``xp.searchsorted`` + Horner) uses only namespace
    operations, so the spline value is computed natively -- and is exactly
    autodifferentiable -- under jax/torch. Mathematically this is the same
    piecewise cubic as the scipy spline (agreement at the ~1 ulp level); the
    numpy code paths keep calling the scipy splines directly and never come
    through here. ``r`` outside ``[x[0], x[-1]]`` evaluates the edge polynomial
    (finite extrapolation), which keeps the dead side of the callers'
    ``xp.where`` branch selections NaN-free under autodiff.
    """
    # knots/coefficients stay float64 (precision is the point; the callers
    # exit-cast) but must live on the input's device (CUDA support)
    dev = device_of(r)
    xb = asarray_on_device(xp, x, dev)
    cb = asarray_on_device(xp, c, dev)
    idx = xp.clip(xp.searchsorted(xb, r, side="right") - 1, 0, cb.shape[1] - 1)
    dr = r - xb[idx]
    out = cb[0, idx]
    for j in range(1, cb.shape[0]):
        out = out * dr + cb[j, idx]
    return out


class interpSphericalPotential(SphericalPotential):
    """__init__(self,rforce=None,rgrid=numpy.geomspace(0.01,20,101),Phi0=None,ro=None,vo=None)

    Class that interpolates a spherical potential on a grid"""

    def __init__(
        self,
        rforce=None,
        rgrid=numpy.geomspace(0.01, 20, 101),
        Phi0=None,
        ro=None,
        vo=None,
    ):
        """
        Initialize an interpolated, spherical potential.

        Parameters
        ----------
        rforce : function or galpy Potential instance or a combined potential formed using addition (pot1+pot2+…), optional
            Either a function that gives the radial force (in internal units) as a function of r (in internal units) or a galpy Potential instance or a combined potential formed using addition (pot1+pot2+…). The default is None.
        rgrid : numpy.ndarray, optional
            Radial grid in internal units on which to evaluate the potential for interpolation (note that beyond rgrid[-1], the potential is extrapolated as -GM(<rgrid[-1])/r). The default is numpy.geomspace(0.01,20,101).
        Phi0 : float, optional
            Value of the potential at rgrid[0] in internal units (only necessary when rforce is a function, for galpy potentials automatically determined). The default is None.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2020-07-13 - Written - Bovy (UofT)

        """
        SphericalPotential.__init__(self, amp=1.0, ro=ro, vo=vo)
        self._rgrid = rgrid
        self._rforce_jax_rgrid = (
            rgrid
            if len(rgrid) > 10000
            else numpy.geomspace(
                1e-3 if rgrid[0] == 0.0 else rgrid[0], rgrid[-1], 10001
            )
        )
        # Determine whether rforce is a galpy Potential or a combined potential formed using addition (pot1+pot2+…)
        try:
            _evaluateRforces(rforce, 1.0, 0.0)
        except:
            _rforce = rforce
            Phi0 = 0.0 if Phi0 is None else Phi0
        else:
            _rforce = lambda r: _evaluateRforces(rforce, r, 0.0)
            # Determine Phi0
            Phi0 = _evaluatePotentials(rforce, rgrid[0], 0.0)
            # Also check that unit systems are compatible
            if not physical_compatible(self, rforce):
                raise RuntimeError(
                    "Unit conversion factors ro and vo incompatible between Potential to be interpolated and the factors given to interpSphericalPotential"
                )
            # If set for the parent, set for the interpolated
            phys = get_physical(rforce, include_set=True)
            if phys["roSet"]:
                self.turn_physical_on(ro=phys["ro"])
            if phys["voSet"]:
                self.turn_physical_on(vo=phys["vo"])
        self._rforce_grid = numpy.array([_rforce(r) for r in rgrid])
        self._force_spline = interpolate.InterpolatedUnivariateSpline(
            self._rgrid, self._rforce_grid, k=3, ext=0
        )
        self._rforce_jax_grid = numpy.array(
            [self._force_spline(r) for r in self._rforce_jax_rgrid]
        )
        # Get potential and r2deriv as splines for the integral and derivative
        self._pot_spline = self._force_spline.antiderivative()
        self._Phi0 = Phi0 + self._pot_spline(self._rgrid[0])
        self._r2deriv_spline = self._force_spline.derivative()
        # Piecewise-power (PPoly) representation of the three splines for the
        # non-numpy backends (see _ppoly_eval). The antiderivative/derivative
        # splines share the force spline's knots, so a single breakpoint array
        # serves all three coefficient sets.
        self._ppoly_x, self._force_ppoly_c = _spline_to_ppoly_data(self._force_spline)
        _, self._pot_ppoly_c = _spline_to_ppoly_data(self._pot_spline)
        _, self._r2deriv_ppoly_c = _spline_to_ppoly_data(self._r2deriv_spline)
        # Extrapolate as mass within rgrid[-1]
        self._rmin = rgrid[0]
        self._rmax = rgrid[-1]
        self._total_mass = -(self._rmax**2.0) * self._force_spline(self._rmax)
        self._Phimax = (
            -self._pot_spline(self._rmax) + self._Phi0 + self._total_mass / self._rmax
        )
        self.hasC = True
        self.hasC_dxdv = True
        self.hasC_dens = True
        return None

    def _revaluate(self, r, t=0.0):
        xp = get_namespace(r)
        if xp is numpy:
            out = numpy.empty_like(r)
            out[r >= self._rmax] = -self._total_mass / r[r >= self._rmax] + self._Phimax
            out[r < self._rmax] = -self._pot_spline(r[r < self._rmax]) + self._Phi0
            return out
        # Backend (jax/torch) path: same piecewise definition through xp.where.
        # The spline piece extrapolates finitely beyond rmax (the dead side of
        # the where), while the Kepler piece guards its dead-side r=0 (r >= rmax
        # implies r > 0 on the live side), so autodiff stays NaN-free.
        r = xp.asarray(r)
        inside = -_ppoly_eval(xp, self._ppoly_x, self._pot_ppoly_c, r) + float(
            self._Phi0
        )
        rsafe = xp.where(r >= self._rmax, r, 1.0)
        outside = -float(self._total_mass) / rsafe + float(self._Phimax)
        # the spline knots/coefficients are deliberately float64 (precision);
        # cast the result to the input dtype at exit (no-op for float64 input;
        # the numpy path above already follows the input dtype via empty_like)
        return match_input_dtype(xp.where(r >= self._rmax, outside, inside), r)

    def _rforce(self, r, t=0.0):
        xp = get_namespace(r)
        if xp is numpy:
            out = numpy.empty_like(r)
            out[r >= self._rmax] = -self._total_mass / r[r >= self._rmax] ** 2.0
            out[r < self._rmax] = self._force_spline(r[r < self._rmax])
            return out
        r = xp.asarray(r)
        inside = _ppoly_eval(xp, self._ppoly_x, self._force_ppoly_c, r)
        rsafe = xp.where(r >= self._rmax, r, 1.0)
        outside = -float(self._total_mass) / rsafe**2.0
        # float64 spline interior, input-dtype exit cast (see _revaluate)
        return match_input_dtype(xp.where(r >= self._rmax, outside, inside), r)

    def _rforce_jax(self, r):
        if not _JAX_LOADED:  # pragma: no cover
            raise ImportError(
                "Making use of _rforce_jax function requires the google/jax library"
            )
        return jnp.interp(r, self._rforce_jax_rgrid, self._rforce_jax_grid)

    def _r2deriv(self, r, t=0.0):
        xp = get_namespace(r)
        if xp is numpy:
            out = numpy.empty_like(r)
            out[r >= self._rmax] = -2.0 * self._total_mass / r[r >= self._rmax] ** 3.0
            out[r < self._rmax] = -self._r2deriv_spline(r[r < self._rmax])
            return out
        r = xp.asarray(r)
        inside = -_ppoly_eval(xp, self._ppoly_x, self._r2deriv_ppoly_c, r)
        rsafe = xp.where(r >= self._rmax, r, 1.0)
        outside = -2.0 * float(self._total_mass) / rsafe**3.0
        # float64 spline interior, input-dtype exit cast (see _revaluate)
        return match_input_dtype(xp.where(r >= self._rmax, outside, inside), r)

    def _rdens(self, r, t=0.0):
        xp = get_namespace(r)
        if xp is numpy:
            out = numpy.empty_like(r)
            out[r >= self._rmax] = 0.0
            # Fall back onto Poisson eqn., implemented in SphericalPotential
            out[r < self._rmax] = SphericalPotential._rdens(self, r[r < self._rmax])
            return out
        # Poisson-eqn density via the backend _r2deriv/_rforce above; their
        # finite extrapolation keeps the dead (r >= rmax) side of the where
        # NaN-free (r >= rmax > 0, so the 1/r factors are safe there too).
        r = xp.asarray(r)
        inside = SphericalPotential._rdens(self, r, t=t)
        # float64 spline interior, input-dtype exit cast (see _revaluate)
        return match_input_dtype(xp.where(r >= self._rmax, 0.0, inside), r)
