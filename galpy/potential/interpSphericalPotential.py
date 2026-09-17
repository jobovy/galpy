###################3###################3###################3##################
# interpSphericalPotential.py: build spherical potential through interpolation
###################3###################3###################3##################
import numpy
from scipy import interpolate

from ..backend import (
    as_numpy,
    coerce_coords,
    get_namespace,
    grad_namespace,
    match_input_dtype,
)
from ..backend._namespaces import differentiating
from ..backend.interpolate import cubic_spline_coeffs
from ..backend.interpolate import eval_ppoly as _ppoly_eval
from ..backend.interpolate import ppoly_antiderivative, ppoly_derivative
from ..backend.interpolate import spline_to_ppoly as _spline_to_ppoly_data
from ..util.conversion import get_physical, physical_compatible
from .Potential import _evaluatePotentials, _evaluateRforces
from .SphericalPotential import SphericalPotential


def _scal(v):
    """``float(v)``, except for a TRACED ``v`` which passes through.

    The derived scalars below (Phi0, total_mass, Phimax) are concretized on the
    numpy path so the backend branches mix plain floats; under a trace that cast
    would both raise and cut the gradient to the construction parameters.
    """
    return v if differentiating(v) else float(v)


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
        _fgrid = [_rforce(r) for r in rgrid]
        # TRACED forces (a parameter of the potential being interpolated is being
        # differentiated) must stay on the backend: numpy.array() of tracers both
        # raises and would sever d/d(parameter).
        self._traced = differentiating(*_fgrid, Phi0)
        if self._traced:
            xp = grad_namespace(*_fgrid, Phi0)
            self._rforce_grid = xp.stack(list(coerce_coords(xp, *_fgrid)))
            # In-backend spline fit, so the coefficients -- and everything derived
            # from them below -- carry the gradient. 'not-a-knot' is exactly the
            # end condition InterpolatedUnivariateSpline(k=3) uses, so this is the
            # SAME spline the numpy branch fits (agreeing to ~1e-15), not an
            # approximation of it.
            self._force_spline = self._pot_spline = self._r2deriv_spline = None
        else:
            self._rforce_grid = numpy.array(_fgrid)
            self._force_spline = interpolate.InterpolatedUnivariateSpline(
                self._rgrid, self._rforce_grid, k=3, ext=0
            )
            # Get potential and r2deriv as splines for the integral and derivative
            self._pot_spline = self._force_spline.antiderivative()
        # Freeze Phi0 on the numpy side: every other derived scalar here comes
        # from a scipy spline and is numpy, and _revaluate's numpy branch mixes
        # them directly. Built under a forced backend, _evaluatePotentials returns
        # a backend scalar, which used to make _Phi0/_Phimax backend arrays while
        # _total_mass/_rmax stayed numpy -- numpy expression + Tensor on line ~105.
        # Nothing is lost: the splines are scipy, so parameter gradients are
        # already unavailable, and the backend branch coerces with xp.asarray.
        if not self._traced:
            self._Phi0 = as_numpy(Phi0) + self._pot_spline(self._rgrid[0])
            self._r2deriv_spline = self._force_spline.derivative()
        # Piecewise-power (PPoly) representation of the three splines for the
        # non-numpy backends (see _ppoly_eval). The antiderivative/derivative
        # splines share the force spline's knots, so a single breakpoint array
        # serves all three coefficient sets.
        if self._traced:
            self._ppoly_x = self._rgrid
            self._force_ppoly_c = cubic_spline_coeffs(
                xp, self._ppoly_x, self._rforce_grid, bc="not-a-knot"
            )
            self._pot_ppoly_c = ppoly_antiderivative(
                xp, self._ppoly_x, self._force_ppoly_c
            )
            self._r2deriv_ppoly_c = ppoly_derivative(xp, self._force_ppoly_c)
            self._Phi0 = Phi0 + _ppoly_eval(
                xp,
                self._ppoly_x,
                self._pot_ppoly_c,
                coerce_coords(xp, self._rgrid[0])[0],
            )
        else:
            self._ppoly_x, self._force_ppoly_c = _spline_to_ppoly_data(
                self._force_spline
            )
            _, self._pot_ppoly_c = _spline_to_ppoly_data(self._pot_spline)
            _, self._r2deriv_ppoly_c = _spline_to_ppoly_data(self._r2deriv_spline)
        # Extrapolate as mass within rgrid[-1]
        self._rmin = rgrid[0]
        self._rmax = rgrid[-1]
        if self._traced:
            (_rmaxb,) = coerce_coords(xp, self._rmax)
            self._total_mass = -(_rmaxb**2.0) * _ppoly_eval(
                xp, self._ppoly_x, self._force_ppoly_c, _rmaxb
            )
            self._Phimax = (
                -_ppoly_eval(xp, self._ppoly_x, self._pot_ppoly_c, _rmaxb)
                + self._Phi0
                + self._total_mass / _rmaxb
            )
        else:
            self._total_mass = -(self._rmax**2.0) * self._force_spline(self._rmax)
            self._Phimax = (
                -self._pot_spline(self._rmax)
                + self._Phi0
                + self._total_mass / self._rmax
            )
        self.hasC = True
        self._backend_compatible = True
        self.hasC_dxdv = True
        self.hasC_dxdv3d = True  # full 3D Hessian (R2deriv/z2deriv/Rzderiv) in C
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
        inside = -_ppoly_eval(xp, self._ppoly_x, self._pot_ppoly_c, r) + _scal(
            self._Phi0
        )
        rsafe = xp.where(r >= self._rmax, r, 1.0)
        outside = -_scal(self._total_mass) / rsafe + _scal(self._Phimax)
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
        outside = -_scal(self._total_mass) / rsafe**2.0
        # float64 spline interior, input-dtype exit cast (see _revaluate)
        return match_input_dtype(xp.where(r >= self._rmax, outside, inside), r)

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
        outside = -2.0 * _scal(self._total_mass) / rsafe**3.0
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
