###############################################################################
#   IsothermalDiskPotential.py: class that implements the one-dimensional
#                               self-gravitating isothermal disk
###############################################################################
import numpy

from ..util import conversion
from .linearPotential import linearPotential


class IsothermalDiskPotential(linearPotential):
    """Class representing the one-dimensional self-gravitating isothermal disk

    .. math::

        \\rho(x) = \\mathrm{amp}\\,\\mathrm{sech}^2\\left(\\frac{x}{2H}\\right)

    where the scale height :math:`H^2 = \\sigma^2/[8\\pi G \\,\\mathrm{amp}]`. The parameter to setup the disk is the velocity dispersion :math:`\\sigma`.

    """

    def __init__(self, amp=1.0, sigma=0.1, ro=None, vo=None):
        """
        Initialize an IsothermalDiskPotential.

        Parameters
        ----------
        amp : float, optional
            An overall amplitude.
        sigma : float or Quantity, optional
            Velocity dispersion.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2018-04-11 - Written - Bovy (UofT)

        """
        linearPotential.__init__(self, amp=amp, ro=ro, vo=vo)
        sigma = conversion.parse_velocity(sigma, vo=self._vo)
        self._sigma2 = sigma**2.0
        self._H = sigma / numpy.sqrt(8.0 * numpy.pi * self._amp)
        self._amp = 1.0  # Need to manually set to 1, because amp is now contained in the combination of H and sigma^2
        self.hasC = True

    def _evaluate(self, x, t=0.0):
        return 2.0 * self._sigma2 * numpy.log(numpy.cosh(0.5 * x / self._H))

    def _force(self, x, t=0.0):
        return -self._sigma2 * numpy.tanh(0.5 * x / self._H) / self._H
