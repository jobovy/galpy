###################3###################3###################3##################
# interpSphericalPotential.py: build spherical potential through interpolation
###################3###################3###################3##################
import numpy
from scipy import interpolate

from ..util._optional_deps import _JAX_LOADED
from ..util.conversion import get_physical, physical_compatible
from .Potential import _evaluatePotentials, _evaluateRforces
from .SphericalPotential import SphericalPotential

if _JAX_LOADED:
    import jax.numpy as jnp


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
        rforce : function or galpy Potential instance or list thereof, optional
            Either a function that gives the radial force (in internal units) as a function of r (in internal units) or a galpy Potential instance or list thereof. The default is None.
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
        # Determine whether rforce is a galpy Potential or list thereof
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
        out = numpy.empty_like(r)
        out[r >= self._rmax] = -self._total_mass / r[r >= self._rmax] + self._Phimax
        out[r < self._rmax] = -self._pot_spline(r[r < self._rmax]) + self._Phi0
        return out

    def _rforce(self, r, t=0.0):
        out = numpy.empty_like(r)
        out[r >= self._rmax] = -self._total_mass / r[r >= self._rmax] ** 2.0
        out[r < self._rmax] = self._force_spline(r[r < self._rmax])
        return out

    def _rforce_jax(self, r):
        if not _JAX_LOADED:  # pragma: no cover
            raise ImportError(
                "Making use of _rforce_jax function requires the google/jax library"
            )
        return jnp.interp(r, self._rforce_jax_rgrid, self._rforce_jax_grid)

    def _r2deriv(self, r, t=0.0):
        out = numpy.empty_like(r)
        out[r >= self._rmax] = -2.0 * self._total_mass / r[r >= self._rmax] ** 3.0
        out[r < self._rmax] = -self._r2deriv_spline(r[r < self._rmax])
        return out

    def _rdens(self, r, t=0.0):
        out = numpy.empty_like(r)
        out[r >= self._rmax] = 0.0
        # Fall back onto Poisson eqn., implemented in SphericalPotential
        out[r < self._rmax] = SphericalPotential._rdens(self, r[r < self._rmax])
        return out
