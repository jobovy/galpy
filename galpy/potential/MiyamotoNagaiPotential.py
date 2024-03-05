###############################################################################
#   MiyamotoNagaiPotential.py: class that implements the Miyamoto-Nagai
#                              potential
#                                                           GM
#                              phi(R,z) = -  ---------------------------------
#                                             \sqrt(R^2+(a+\sqrt(z^2+b^2))^2)
###############################################################################
import numpy

from ..util import conversion
from .Potential import Potential, kms_to_kpcGyrDecorator


class MiyamotoNagaiPotential(Potential):
    """Class that implements the Miyamoto-Nagai potential [1]_

    .. math::

        \\Phi(R,z) = -\\frac{\\mathrm{amp}}{\\sqrt{R^2+(a+\\sqrt{z^2+b^2})^2}}

    with :math:`\\mathrm{amp} = GM` the total mass.
    """

    def __init__(self, amp=1.0, a=1.0, b=0.1, normalize=False, ro=None, vo=None):
        """
        Initialize a Miyamoto-Nagai potential.

        Parameters
        ----------
        amp : float or Quantity, optional
            Amplitude to be applied to the potential, the total mass (default: 1); can be a Quantity with units of mass or Gxmass.
        a : float or Quantity, optional
            Scale length.
        b : float or Quantity, optional
            Scale height.
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.
        ro : float, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2010-07-09 - Started - Bovy (NYU)

        References
        ----------
        .. [1] Miyamoto, M., & Nagai, R. (1975). Three-dimensional models for the distribution of mass in galaxies. Publications of the Astronomical Society of Japan, 27(4), 533-543. ADS: https://ui.adsabs.harvard.edu/abs/1975PASJ...27..533M/abstract
        """
        Potential.__init__(self, amp=amp, ro=ro, vo=vo, amp_units="mass")
        a = conversion.parse_length(a, ro=self._ro)
        b = conversion.parse_length(b, ro=self._ro)
        self._a = a
        self._scale = self._a
        self._b = b
        self._b2 = self._b**2.0
        if normalize or (
            isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
        ):
            self.normalize(normalize)
        self.hasC = True
        self.hasC_dxdv = True
        self.hasC_dens = True
        self._nemo_accname = "MiyamotoNagai"

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        return -1.0 / numpy.sqrt(
            R**2.0 + (self._a + numpy.sqrt(z**2.0 + self._b2)) ** 2.0
        )

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        return -R / (R**2.0 + (self._a + numpy.sqrt(z**2.0 + self._b2)) ** 2.0) ** (
            3.0 / 2.0
        )

    def _zforce(self, R, z, phi=0.0, t=0.0):
        sqrtbz = numpy.sqrt(self._b2 + z**2.0)
        asqrtbz = self._a + sqrtbz
        if isinstance(R, float) and sqrtbz == asqrtbz:
            return -z / (R**2.0 + (self._a + numpy.sqrt(z**2.0 + self._b2)) ** 2.0) ** (
                3.0 / 2.0
            )
        else:
            return (
                -z
                * asqrtbz
                / sqrtbz
                / (R**2.0 + (self._a + numpy.sqrt(z**2.0 + self._b2)) ** 2.0)
                ** (3.0 / 2.0)
            )

    def _dens(self, R, z, phi=0.0, t=0.0):
        sqrtbz = numpy.sqrt(self._b2 + z**2.0)
        asqrtbz = self._a + sqrtbz
        if isinstance(R, float) and sqrtbz == asqrtbz:
            return 3.0 / (R**2.0 + sqrtbz**2.0) ** 2.5 / 4.0 / numpy.pi * self._b2
        else:
            return (
                (self._a * R**2.0 + (self._a + 3.0 * sqrtbz) * asqrtbz**2.0)
                / (R**2.0 + asqrtbz**2.0) ** 2.5
                / sqrtbz**3.0
                / 4.0
                / numpy.pi
                * self._b2
            )

    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        return (
            1.0 / (R**2.0 + (self._a + numpy.sqrt(z**2.0 + self._b2)) ** 2.0) ** 1.5
            - 3.0
            * R**2.0
            / (R**2.0 + (self._a + numpy.sqrt(z**2.0 + self._b2)) ** 2.0) ** 2.5
        )

    def _z2deriv(self, R, z, phi=0.0, t=0.0):
        sqrtbz = numpy.sqrt(self._b2 + z**2.0)
        asqrtbz = self._a + sqrtbz
        if isinstance(R, float) and sqrtbz == asqrtbz:
            return (self._b2 + R**2.0 - 2.0 * z**2.0) * (
                self._b2 + R**2.0 + z**2.0
            ) ** -2.5
        else:
            return (
                self._a**3.0 * self._b2
                + self._a**2.0
                * (3.0 * self._b2 - 2.0 * z**2.0)
                * numpy.sqrt(self._b2 + z**2.0)
                + (self._b2 + R**2.0 - 2.0 * z**2.0) * (self._b2 + z**2.0) ** 1.5
                + self._a
                * (3.0 * self._b2**2.0 - 4.0 * z**4.0 + self._b2 * (R**2.0 - z**2.0))
            ) / ((self._b2 + z**2.0) ** 1.5 * (R**2.0 + asqrtbz**2.0) ** 2.5)

    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        sqrtbz = numpy.sqrt(self._b2 + z**2.0)
        asqrtbz = self._a + sqrtbz
        if isinstance(R, float) and sqrtbz == asqrtbz:
            return -(3.0 * R * z / (R**2.0 + asqrtbz**2.0) ** 2.5)
        else:
            return -(3.0 * R * z * asqrtbz / sqrtbz / (R**2.0 + asqrtbz**2.0) ** 2.5)

    @kms_to_kpcGyrDecorator
    def _nemo_accpars(self, vo, ro):
        ampl = self._amp * vo**2.0 * ro
        return f"0,{ampl},{self._a*ro},{self._b*ro}"
