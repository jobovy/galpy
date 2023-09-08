###############################################################################
#   TransientLogSpiralPotential: a transient spiral potential
###############################################################################
import numpy

from ..util import conversion
from .planarPotential import planarPotential

_degtorad = numpy.pi / 180.0


class TransientLogSpiralPotential(planarPotential):
    """Class that implements a steady-state spiral potential

    .. math::

        \\Phi(R,\\phi) = \\frac{\\mathrm{amp}(t)}{\\alpha}\\,\\cos\\left(\\alpha\\,\\ln R - m\\,(\\phi-\\Omega_s\\,t-\\gamma)\\right)

    where

    .. math::

        \\mathrm{amp}(t) = \\mathrm{amp}\\,\\times A\\,\\exp\\left(-\\frac{[t-t_0]^2}{2\\,\\sigma^2}\\right)

    """

    def __init__(
        self,
        amp=1.0,
        omegas=0.65,
        A=-0.035,
        alpha=-7.0,
        m=2,
        gamma=numpy.pi / 4.0,
        p=None,
        sigma=1.0,
        to=0.0,
        ro=None,
        vo=None,
    ):
        """
        Initialize a transient logarithmic spiral potential localized around to

        Parameters
        ----------
        amp : float, optional
            Amplitude to be applied to the potential (default: 1).
        omegas : float or Quantity, optional
            Pattern speed (default: 0.65).
        A : float or Quantity, optional
            Amplitude (alpha*potential-amplitude; default: -0.035).
        alpha : float, optional
            Alpha parameter (default: -7.).
        m : int, optional
            Number of arms (default: 2).
        gamma : float or Quantity, optional
            Angle between sun-GC line and the line connecting the peak of the spiral pattern at the Solar radius (in rad; default: 45 degree).
        p : float or Quantity, optional
            Pitch angle.
        sigma : float or Quantity, optional
            "Spiral duration" (sigma in Gaussian amplitude; default: 1.).
        to : float or Quantity, optional
            Time at which the spiral peaks (default: 0.).

        Notes
        -----
        - Either provide:
            * alpha
            * p
        - 2011-03-27 - Started - Bovy (NYU)
        """
        planarPotential.__init__(self, amp=amp, ro=ro, vo=vo)
        gamma = conversion.parse_angle(gamma)
        p = conversion.parse_angle(p)
        A = conversion.parse_energy(A, vo=self._vo)
        omegas = conversion.parse_frequency(omegas, ro=self._ro, vo=self._vo)
        to = conversion.parse_time(to, ro=self._ro, vo=self._vo)
        sigma = conversion.parse_time(sigma, ro=self._ro, vo=self._vo)
        self._omegas = omegas
        self._A = A
        self._m = m
        self._gamma = gamma
        self._to = to
        self._sigma2 = sigma**2.0
        if not p is None:
            self._alpha = self._m / numpy.tan(p)
        else:
            self._alpha = alpha
        self.hasC = True

    def _evaluate(self, R, phi=0.0, t=0.0):
        return (
            self._A
            * numpy.exp(-((t - self._to) ** 2.0) / 2.0 / self._sigma2)
            / self._alpha
            * numpy.cos(
                self._alpha * numpy.log(R)
                - self._m * (phi - self._omegas * t - self._gamma)
            )
        )

    def _Rforce(self, R, phi=0.0, t=0.0):
        return (
            self._A
            * numpy.exp(-((t - self._to) ** 2.0) / 2.0 / self._sigma2)
            / R
            * numpy.sin(
                self._alpha * numpy.log(R)
                - self._m * (phi - self._omegas * t - self._gamma)
            )
        )

    def _phitorque(self, R, phi=0.0, t=0.0):
        return (
            -self._A
            * numpy.exp(-((t - self._to) ** 2.0) / 2.0 / self._sigma2)
            / self._alpha
            * self._m
            * numpy.sin(
                self._alpha * numpy.log(R)
                - self._m * (phi - self._omegas * t - self._gamma)
            )
        )

    def OmegaP(self):
        return self._omegas
