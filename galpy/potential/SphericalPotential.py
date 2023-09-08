###############################################################################
#   SphericalPotential.py: base class for potentials corresponding to
#                          spherical density profiles
###############################################################################
import numpy
from scipy import integrate

from .Potential import Potential


class SphericalPotential(Potential):
    """Base class for spherical potentials.

    Implement a specific spherical density distribution with this form by inheriting from this class and defining functions:

    * ``_revaluate(self,r,t=0.)``: the potential as a function of ``r`` and time;

    * ``_rforce(self,r,t=0.)``: the radial force as a function of ``r`` and time;

    * ``_r2deriv(self,r,t=0.)``: the second radial derivative of the potential as a function of ``r`` and time;

    * ``_rdens(self,r,t=0.)``: the density as a function of ``r`` and time (if *not* implemented, calculated using the Poisson equation).
    """

    def __init__(self, amp=1.0, ro=None, vo=None, amp_units=None):
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
        - 2020-03-30 - Written - Bovy (UofT)

        """
        Potential.__init__(self, amp=amp, ro=ro, vo=vo, amp_units=amp_units)
        return None

    def _rdens(self, r, t=0.0):
        """Implement using the Poisson equation in case this isn't implemented"""
        return (self._r2deriv(r, t=t) - 2.0 * self._rforce(r, t=t) / r) / 4.0 / numpy.pi

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        r = numpy.sqrt(R**2.0 + z**2.0)
        return self._revaluate(r, t=t)

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        r = numpy.sqrt(R**2.0 + z**2.0)
        return self._rforce(r, t=t) * R / r

    def _zforce(self, R, z, phi=0.0, t=0.0):
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

    def _mass(self, R, z=None, t=0.0):
        if z is not None:
            raise AttributeError  # use general implementation
        R = numpy.float64(R)  # Avoid indexing issues
        return -(R**2.0) * self._rforce(R, t=t)
