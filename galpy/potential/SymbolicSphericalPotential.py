###############################################################################
#   SymbolicSphericalPotential.py: base class for symbolic potentials
#   corresponding to spherical density profiles
###############################################################################
import numpy

from ..util._optional_deps import _SYMPY_LOADED
from .Potential import SphericalPotential

if _SYMPY_LOADED:
    import sympy


class SymbolicSphericalPotential(SphericalPotential):
    """
    Base class for symbolic spherical potentials.

    Implement a specific spherical density distribution by specifying the (symbolic) density function.
    P.S. Only supports time-independent potential at the moment.
    """

    def __init__(self, dens=None, amp=1.0, ro=None, vo=None, amp_units=None):
        """
        Initialize a spherical potential.

        Parameters
        ----------
        amp : float, optional
            Amplitude to be applied to the potential (default: 1).
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).
        amp_units : str, optional
            Type of units that amp should have if it has units ('mass', 'velocity2', 'density').

        Notes
        -----
        - 2025-08-10 - Written - Yuzhe Zhang (Uni Mainz)

        """
        Potential.__init__(self, amp=amp, ro=ro, vo=vo, amp_units=amp_units)

        self.dens, self.r = dens()
        # Compute enclosed mass symbolically
        integrand = 4 * sympy.pi * self.r**2 * self.dens
        M_r = sympy.integrate(integrand, (self.r, 0, self.r))
        self.rawMass = M_r.simplify()
        del integrand

        # Compute the potential symbolically
        Phi_outside_r = (-4 * sympy.pi) * sympy.integrate(
            self.dens * self.r, (self.r, self.r, sympy.oo)
        ).simplify()
        self.Phi = -self.rawMass / self.r + Phi_outside_r

        # second derivative determined by sympy.diff
        self.d2Phidr2 = sympy.diff(self.Phi, self.r, 2)

        return None

    def _revaluate(self, r: float, t: float = 0.0):
        """Returns the potential at a given radius r and time t"""
        return float(self.Phi.evalf(subs={self.r: r}))

    def _rforce(self, r: float, t: float = 0.0):
        """Returns the radial force at a given radius r and time t"""
        return float((-self.rawMass / self.r**2).evalf(subs={self.r: r}))

    def _r2deriv(self, r: float, t: float = 0.0):
        """Returns the second radial derivative of the potential at a given radius r and time t"""
        # use the d2Phidr2 obtained by sympy.diff
        return float(self.d2Phidr2.evalf(subs={self.r: r}))

    def _rdens(self, r: float, t: float = 0.0):
        """Returns the density at a given radius r and time t"""
        return float(self.dens.evalf(subs={self.r: r}))

    def _rmass(self, r: float, t: float = 0.0):
        """Returns the density at a given radius r and time t"""
        return float(self.rawMass.evalf(subs={self.r: r}))

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        """Find potential at (R, z, phi, t)"""
        r = numpy.sqrt(R**2.0 + z**2.0)
        return self._revaluate(r, t=t)

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        """Return the radial force component in cylindrical coordinates (R, z, phi) at time t."""
        r = numpy.sqrt(R**2.0 + z**2.0)
        return self._rforce(r, t=t) * R / r

    def _zforce(self, R, z, phi=0.0, t=0.0):
        """Return the vertical (z) force component in cylindrical coordinates (R, z, phi) at time t."""
        r = numpy.sqrt(R**2.0 + z**2.0)
        return self._rforce(r, t=t) * z / r

    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        r = numpy.sqrt(R**2.0 + z**2.0)
        return (
            self._r2deriv(r, t=t) * R**2.0 / r**2.0
            - self._rforce(r, t=t) * z**2.0 / r**3.0
        )

    def _z2deriv(self, R, z, phi=0.0, t=0.0):
        r = numpy.sqrt(R**2.0 + z**2.0)
        return (
            self._r2deriv(r, t=t) * z**2.0 / r**2.0
            - self._rforce(r, t=t) * R**2.0 / r**3.0
        )

    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        r = numpy.sqrt(R**2.0 + z**2.0)
        return (
            self._r2deriv(r, t=t) * R * z / r**2.0
            + self._rforce(r, t=t) * R * z / r**3.0
        )

    def _dens(self, R, z, phi=0.0, t=0.0):
        r = numpy.sqrt(R**2.0 + z**2.0)
        return self._rdens(r, t=t)

    def _mass(self, R, z, phi=0.0, t=0.0):
        R, z = numpy.float64(R), numpy.float64(z)  # Avoid indexing issues
        r = numpy.sqrt(R**2.0 + z**2.0)
        return self._rmass(r=r)
