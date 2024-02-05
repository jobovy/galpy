###############################################################################
#   PseudoIsothermalPotential.py: class that implements the pseudo-isothermal
#                                 halo potential
###############################################################################
import numpy

from ..util import conversion
from .Potential import Potential


class PseudoIsothermalPotential(Potential):
    """Class that implements the pseudo-isothermal potential

    .. math::

        \\rho(r) = \\frac{\\mathrm{amp}}{4\\,\\pi\\, a^3}\\,\\frac{1}{1+(r/a)^2}

    """

    def __init__(self, amp=1.0, a=1.0, normalize=False, ro=None, vo=None):
        """
        Initialize a pseudo-isothermal potential.

        Parameters
        ----------
        amp : float, optional
            Amplitude to be applied to the potential.
        a : float or Quantity, optional
            Core radius.
        normalize : bool, int, or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2015-12-04 - Started - Bovy (UofT)
        """
        Potential.__init__(self, amp=amp, ro=ro, vo=vo, amp_units="mass")
        a = conversion.parse_length(a, ro=self._ro)
        self.hasC = True
        self.hasC_dxdv = True
        self.hasC_dens = True
        self._a = a
        self._a2 = a**2.0
        self._a3 = a**3.0
        if normalize or (
            isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
        ):  # pragma: no cover
            self.normalize(normalize)
        return None

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        r2 = R**2.0 + z**2.0
        r = numpy.sqrt(r2)
        out = (
            0.5 * numpy.log(1 + r2 / self._a2) + self._a / r * numpy.arctan(r / self._a)
        ) / self._a
        if isinstance(r, (float, int)):
            if r == 0:
                return 1.0 / self._a
            else:
                return out
        else:
            out[r == 0] = 1.0 / self._a
            return out

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        r2 = R**2.0 + z**2.0
        r = numpy.sqrt(r2)
        return -(1.0 / r - self._a / r2 * numpy.arctan(r / self._a)) / self._a * R / r

    def _zforce(self, R, z, phi=0.0, t=0.0):
        r2 = R**2.0 + z**2.0
        r = numpy.sqrt(r2)
        return -(1.0 / r - self._a / r2 * numpy.arctan(r / self._a)) / self._a * z / r

    def _dens(self, R, z, phi=0.0, t=0.0):
        return 1.0 / (1.0 + (R**2.0 + z**2.0) / self._a2) / 4.0 / numpy.pi / self._a3

    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        r2 = R**2.0 + z**2.0
        r = numpy.sqrt(r2)
        return (
            1.0
            / r2
            * (1.0 - R**2.0 / r2 * (3.0 * self._a2 + 2.0 * r2) / (self._a2 + r2))
            + self._a / r2 / r * (3.0 * R**2.0 / r2 - 1.0) * numpy.arctan(r / self._a)
        ) / self._a

    def _z2deriv(self, R, z, phi=0.0, t=0.0):
        r2 = R**2.0 + z**2.0
        r = numpy.sqrt(r2)
        return (
            1.0
            / r2
            * (1.0 - z**2.0 / r2 * (3.0 * self._a2 + 2.0 * r2) / (self._a2 + r2))
            + self._a / r2 / r * (3.0 * z**2.0 / r2 - 1.0) * numpy.arctan(r / self._a)
        ) / self._a

    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        r2 = R**2.0 + z**2.0
        r = numpy.sqrt(r2)
        return (
            (
                3.0 * self._a / r2 / r2 * numpy.arctan(r / self._a)
                - 1.0 / r2 / r * ((3.0 * self._a2 + 2.0 * r2) / (r2 + self._a2))
            )
            * R
            * z
            / r
            / self._a
        )
