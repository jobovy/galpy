###############################################################################
#   GaussianAmplitudeWrapperPotential.py: Wrapper to modulate the amplitude
#                                         of a potential with a Gaussian
###############################################################################
from ..backend import get_namespace
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
        pot : Potential instance or a combined potential formed using addition (pot1+pot2+…), optional
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
        # Advertise the 3D variational capability unconditionally, as for
        # hasC/hasC_dxdv: _check_c recurses into the wrapped potential's own
        # hasC_dxdv3d (the wrapper's C 3D Hessian is modulation x
        # calc<deriv>(wrapped), so it is complete iff the wrapped one is).
        self.hasC_dxdv3d = True

    def _smooth(self, t):
        # The namespace follows t itself: a concrete (Python/numpy) t keeps the
        # numpy path byte-identical; a traced t (in-backend integrator, autodiff
        # wrt time) uses that backend's exp, so it is differentiable.
        xp = get_namespace(t)
        return xp.exp(-0.5 * (t - self._to) ** 2.0 / self._sigma2)

    def _wrap(self, attribute, *args, **kwargs):
        return self._smooth(kwargs.get("t", 0.0)) * self._wrap_pot_func(attribute)(
            self._pot, *args, **kwargs
        )
