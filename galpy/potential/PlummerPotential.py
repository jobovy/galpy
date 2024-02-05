###############################################################################
#   PlummerPotential.py: class that implements the Plummer potential
#                                                           GM
#                              phi(R,z) = -  ---------------------------------
#                                                    \sqrt(R^2+z^2+b^2)
###############################################################################
import numpy

from ..util import conversion
from .Potential import Potential, kms_to_kpcGyrDecorator


class PlummerPotential(Potential):
    """Class that implements the Plummer potential

    .. math::

        \\Phi(R,z) = -\\frac{\\mathrm{amp}}{\\sqrt{R^2+z^2+b^2}}

    with :math:`\\mathrm{amp} = GM` the total mass.
    """

    def __init__(self, amp=1.0, b=0.8, normalize=False, ro=None, vo=None):
        """
        Initialize a Plummer potential.

        Parameters
        ----------
        amp : float or Quantity, optional
            Amplitude to be applied to the potential, the total mass. Default is 1. Can be a Quantity with units of mass or Gxmass.
        b : float or Quantity, optional
            Scale parameter. Can be a Quantity.
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1. Default is False.
        ro : float, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2015-06-15 - Written - Bovy (IAS)
        """
        Potential.__init__(self, amp=amp, ro=ro, vo=vo, amp_units="mass")
        self._b = conversion.parse_length(b, ro=self._ro)
        self._scale = self._b
        self._b2 = self._b**2.0
        if normalize or (
            isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
        ):
            self.normalize(normalize)
        self.hasC = True
        self.hasC_dxdv = True
        self.hasC_dens = True
        self._nemo_accname = "Plummer"

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        return -1.0 / numpy.sqrt(R**2.0 + z**2.0 + self._b2)

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        dPhidrr = -((R**2.0 + z**2.0 + self._b2) ** -1.5)
        return dPhidrr * R

    def _zforce(self, R, z, phi=0.0, t=0.0):
        dPhidrr = -((R**2.0 + z**2.0 + self._b2) ** -1.5)
        return dPhidrr * z

    def _rforce_jax(self, r):
        # No need for actual JAX!
        return -self._amp * r * (r**2.0 + self._b2) ** -1.5

    def _dens(self, R, z, phi=0.0, t=0.0):
        return 3.0 / 4.0 / numpy.pi * self._b2 * (R**2.0 + z**2.0 + self._b2) ** -2.5

    def _surfdens(self, R, z, phi=0.0, t=0.0):
        Rb = R**2.0 + self._b2
        return (
            self._b2
            * z
            * (3.0 * Rb + 2.0 * z**2.0)
            / Rb**2.0
            * (Rb + z**2.0) ** -1.5
            / 2.0
            / numpy.pi
        )

    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        return (self._b2 - 2.0 * R**2.0 + z**2.0) * (R**2.0 + z**2.0 + self._b2) ** -2.5

    def _z2deriv(self, R, z, phi=0.0, t=0.0):
        return (self._b2 + R**2.0 - 2.0 * z**2.0) * (R**2.0 + z**2.0 + self._b2) ** -2.5

    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        return -3.0 * R * z * (R**2.0 + z**2.0 + self._b2) ** -2.5

    def _ddensdr(self, r, t=0.0):
        return (
            self._amp
            * (-15.0)
            / 4.0
            / numpy.pi
            * self._b2
            * r
            * (r**2 + self._b2) ** -3.5
        )

    def _d2densdr2(self, r, t=0.0):
        return (
            self._amp
            * (-15.0)
            / 4.0
            / numpy.pi
            * self._b2
            * ((r**2.0 + self._b2) ** -3.5 - 7.0 * r**2.0 * (r**2 + self._b2) ** -4.5)
        )

    def _ddenstwobetadr(self, r, beta=0):
        """
        Evaluate the radial density derivative x r^(2beta) for this potential.

        Parameters
        ----------
        r : float
            Spherical radius.
        beta : int, optional
            Power of r in the density derivative. Default is 0.

        Returns
        -------
        float
            The derivative of the density times r^(2beta).

        Notes
        -----
        - 2021-03-15 - Written - Lane (UofT)

        """
        return (
            self._amp
            * 3.0
            / 4.0
            / numpy.pi
            * self._b2
            * r ** (2.0 * beta - 1.0)
            * (
                2.0 * beta * (r**2.0 + self._b2) ** -2.5
                - 5.0 * r**2.0 * (r**2.0 + self._b2) ** -3.5
            )
        )

    def _mass(self, R, z=None, t=0.0):
        if z is not None:
            raise AttributeError  # use general implementation
        r2 = R**2.0
        return (1.0 + self._b2 / r2) ** -1.5  # written so it works for r=numpy.inf

    @kms_to_kpcGyrDecorator
    def _nemo_accpars(self, vo, ro):
        ampl = self._amp * vo**2.0 * ro
        return f"0,{ampl},{self._b*ro}"
