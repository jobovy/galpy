###############################################################################
#   FlattenedPowerPotential.py: Power-law potential that is flattened in the
#                               potential (NOT the density)
#
#                                     amp
#                          phi(R,z)= --------- ; m^2 = R^2 + z^2/q^2
#                                   m^\alpha
###############################################################################
import numpy

from ..util import conversion
from .Potential import Potential

_CORE = 10**-8


class FlattenedPowerPotential(Potential):
    """Class that implements a power-law potential that is flattened in the potential (NOT the density)

    .. math::

        \\Phi(R,z) = -\\frac{\\mathrm{amp}\\,r_1^\\alpha}{\\alpha\\,\\left(R^2+(z/q)^2+\\mathrm{core}^2\\right)^{\\alpha/2}}

    and the same as LogarithmicHaloPotential for :math:`\\alpha=0`

    See Figure 1 in `Evans (1994) <http://adsabs.harvard.edu/abs/1994MNRAS.267..333E>`_ for combinations of alpha and q that correspond to positive densities

    """

    def __init__(
        self,
        amp=1.0,
        alpha=0.5,
        q=0.9,
        core=_CORE,
        normalize=False,
        r1=1.0,
        ro=None,
        vo=None,
    ):
        """
        Initialize a flattened power-law potential.

        Parameters
        ----------
        amp : float or Quantity, optional
            Amplitude to be applied to the potential. Can be a Quantity with units of velocity squared.
        alpha : float, optional
            Power-law exponent.
        q : float, optional
            Flattening parameter.
        core : float or Quantity, optional
            Core radius.
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.
        r1 : float or Quantity, optional
            Reference radius for amplitude.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2013-01-09 - Written - Bovy (IAS)
        """
        Potential.__init__(self, amp=amp, ro=ro, vo=vo, amp_units="velocity2")
        core = conversion.parse_length(core, ro=self._ro)
        r1 = conversion.parse_length(r1, ro=self._ro)
        self.alpha = alpha
        self.q2 = q**2.0
        self.core2 = core**2.0
        # Back to old definition
        self._amp *= r1**self.alpha
        if normalize or (
            isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
        ):  # pragma: no cover
            self.normalize(normalize)
        self.hasC = True
        self.hasC_dxdv = True
        self.hasC_dens = True

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        if self.alpha == 0.0:
            return 1.0 / 2.0 * numpy.log(R**2.0 + z**2.0 / self.q2 + self.core2)
        else:
            m2 = self.core2 + R**2.0 + z**2.0 / self.q2
            return -(m2 ** (-self.alpha / 2.0)) / self.alpha

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        if self.alpha == 0.0:
            return -R / (R**2.0 + z**2.0 / self.q2 + self.core2)
        else:
            m2 = self.core2 + R**2.0 + z**2.0 / self.q2
            return -(m2 ** (-self.alpha / 2.0 - 1.0)) * R

    def _zforce(self, R, z, phi=0.0, t=0.0):
        if self.alpha == 0.0:
            return -z / self.q2 / (R**2.0 + z**2.0 / self.q2 + self.core2)
        else:
            m2 = self.core2 + R**2.0 + z**2.0 / self.q2
            return -(m2 ** (-self.alpha / 2.0 - 1.0)) * z / self.q2

    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        if self.alpha == 0.0:
            denom = 1.0 / (R**2.0 + z**2.0 / self.q2 + self.core2)
            return denom - 2.0 * R**2.0 * denom**2.0
        else:
            m2 = self.core2 + R**2.0 + z**2.0 / self.q2
            return -(m2 ** (-self.alpha / 2.0 - 1.0)) * (
                (self.alpha + 2) * R**2.0 / m2 - 1.0
            )

    def _z2deriv(self, R, z, phi=0.0, t=0.0):
        if self.alpha == 0.0:
            denom = 1.0 / (R**2.0 + z**2.0 / self.q2 + self.core2)
            return denom / self.q2 - 2.0 * z**2.0 * denom**2.0 / self.q2**2.0
        else:
            m2 = self.core2 + R**2.0 + z**2.0 / self.q2
            return (
                -1.0
                / self.q2
                * m2 ** (-self.alpha / 2.0 - 1.0)
                * ((self.alpha + 2) * z**2.0 / m2 / self.q2 - 1.0)
            )

    def _dens(self, R, z, phi=0.0, t=0.0):
        if self.alpha == 0.0:
            return (
                1.0
                / 4.0
                / numpy.pi
                / self.q2
                * (
                    (2.0 * self.q2 + 1.0) * self.core2
                    + R**2.0
                    + (2.0 - 1.0 / self.q2) * z**2.0
                )
                / (R**2.0 + z**2.0 / self.q2 + self.core2) ** 2.0
            )
        else:
            m2 = self.core2 + R**2.0 + z**2.0 / self.q2
            return (
                1.0
                / self.q2
                * (
                    self.core2 * (1.0 + 2.0 * self.q2)
                    + R**2.0 * (1.0 - self.alpha * self.q2)
                    + z**2.0 * (2.0 - (1.0 + self.alpha) / self.q2)
                )
                * m2 ** (-self.alpha / 2.0 - 2.0)
                / 4.0
                / numpy.pi
            )
