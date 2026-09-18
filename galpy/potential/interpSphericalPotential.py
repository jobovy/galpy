###################3###################3###################3##################
# interpSphericalPotential.py: build spherical potential through interpolation
###################3###################3###################3##################
import numpy
from scipy import interpolate

from ..backend import (
    as_numpy,
    coerce_coords,
    get_namespace,
    is_backend_array,
    match_input_dtype,
    resolve_namespace,
)
from ..backend._namespaces import requires_backend_grad, under_trace
from ..backend.interpolate import Spline1D
from ..util.conversion import get_physical, physical_compatible
from .Potential import _evaluatePotentials, _evaluateRforces
from .SphericalPotential import SphericalPotential


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
        # Only a DIFFERENTIATED force grid stays on the backend: numpy.array()
        # of tracers raises and would sever d/d(parameter). Backend-ness alone is
        # not the test -- under a forced backend every value is a backend array
        # while nothing is being differentiated, and fitting in-backend there
        # would abandon the scipy fit the numpy queries want.
        if any(under_trace(f) or requires_backend_grad(f) for f in _fgrid):
            xp = resolve_namespace(*_fgrid)
            self._rforce_grid = xp.stack(list(coerce_coords(xp, *_fgrid)))

            def _q(v):  # a query point on the spline's own namespace
                return coerce_coords(xp, v)[0]

        else:
            self._rforce_grid = numpy.array(_fgrid)

            def _q(v):
                return v

        # Spline1D picks its own mode: a numpy grid fits the scipy
        # InterpolatedUnivariateSpline (numpy queries byte-identical, backend
        # queries through its frozen PPoly), a backend grid is fitted IN-backend
        # so the coefficients carry d/d(parameter). 'not-a-knot' is exactly the
        # InterpolatedUnivariateSpline(k=3) end condition, so both modes are the
        # same spline (agreeing to ~1e-15).
        self._force_spline = Spline1D(
            self._rgrid, self._rforce_grid, k=3, ext=0, bc="not-a-knot"
        )
        # Phi and d2Phi/dr2 come from the SAME spline: its antiderivative, and
        # its nu=1 derivative at evaluation time (bitwise equal to scipy's
        # .derivative()(r), so no third spline is needed).
        self._pot_spline = self._force_spline.antiderivative()
        # Freeze Phi0 on the numpy side unless the grid itself is on the backend:
        # every other derived scalar here comes from the spline and is numpy, and
        # _revaluate's numpy branch mixes them directly.
        self._Phi0 = (
            Phi0
            if (under_trace(Phi0) or requires_backend_grad(Phi0))
            else as_numpy(Phi0)
        ) + self._pot_spline(_q(self._rgrid[0]))
        # Extrapolate as mass within rgrid[-1]
        self._rmin = rgrid[0]
        self._rmax = rgrid[-1]
        self._total_mass = -(self._rmax**2.0) * self._force_spline(_q(self._rmax))
        self._Phimax = (
            -self._pot_spline(_q(self._rmax))
            + self._Phi0
            + self._total_mass / self._rmax
        )
        # Concretize the derived scalars on the numpy side, once, so the backend
        # branches below can mix them as plain floats. A backend grid leaves them
        # on the namespace, keeping d/d(parameter).
        for _attr in ("_Phi0", "_total_mass", "_Phimax"):
            _val = getattr(self, _attr)
            if not is_backend_array(_val):
                setattr(self, _attr, float(_val))
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
        inside = -self._pot_spline(r) + self._Phi0
        rsafe = xp.where(r >= self._rmax, r, 1.0)
        outside = -self._total_mass / rsafe + self._Phimax
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
        inside = self._force_spline(r)
        rsafe = xp.where(r >= self._rmax, r, 1.0)
        outside = -self._total_mass / rsafe**2.0
        # float64 spline interior, input-dtype exit cast (see _revaluate)
        return match_input_dtype(xp.where(r >= self._rmax, outside, inside), r)

    def _r2deriv(self, r, t=0.0):
        xp = get_namespace(r)
        if xp is numpy:
            out = numpy.empty_like(r)
            out[r >= self._rmax] = -2.0 * self._total_mass / r[r >= self._rmax] ** 3.0
            out[r < self._rmax] = -self._force_spline(r[r < self._rmax], nu=1)
            return out
        r = xp.asarray(r)
        inside = -self._force_spline(r, nu=1)
        rsafe = xp.where(r >= self._rmax, r, 1.0)
        outside = -2.0 * self._total_mass / rsafe**3.0
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
