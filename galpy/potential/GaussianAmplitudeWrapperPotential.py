###############################################################################
#   GaussianAmplitudeWrapperPotential.py: Wrapper to modulate the amplitude
#                                         of a potential with a Gaussian
###############################################################################
import numpy

from ..util import conversion
from .WrapperPotential import parentWrapperPotential


class GaussianAmplitudeWrapperPotential(parentWrapperPotential):
    """Potential wrapper class that allows the amplitude of a Potential object to be modulated as a Gaussian. The amplitude A applied to a potential wrapped by an instance of this class is changed as

    .. math::

        A(t) = amp\\,\\exp\\left(-\\frac{[t-t_0]^2}{2\\,\\sigma^2}\\right)
    """

    def __init__(self, amp=1.0, pot=None, to=0.0, sigma=1.0, ro=None, vo=None):
        """
        Initialize a GaussianAmplitudeWrapper Potential.

        Parameters
        ----------
        amp : float, optional
            Amplitude to be applied to the potential. Default is 1.0.
        pot : Potential instance or list thereof, optional
            This potential is made to rotate around the z axis by the wrapper.
        to : float or Quantity, optional
            Time at which the Gaussian peaks. Default is 0.0.
        sigma : float or Quantity, optional
            Standard deviation of the Gaussian. Default is 1.0.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2018-02-21 - Started - Bovy (UofT)
        """
        to = conversion.parse_time(to, ro=self._ro, vo=self._vo)
        sigma = conversion.parse_time(sigma, ro=self._ro, vo=self._vo)
        self._to = to
        self._sigma2 = sigma**2.0
        self.hasC = True
        self.hasC_dxdv = True

    def _smooth(self, t):
        return numpy.exp(-0.5 * (t - self._to) ** 2.0 / self._sigma2)

    def _wrap(self, attribute, *args, **kwargs):
        return self._smooth(kwargs.get("t", 0.0)) * self._wrap_pot_func(attribute)(
            self._pot, *args, **kwargs
        )
