###############################################################################
#   HenonHeilesPotential: the Henon-Heiles (1964) potential
###############################################################################
import numpy

from .planarPotential import planarPotential


class HenonHeilesPotential(planarPotential):
    """Class that implements a the `Henon & Heiles (1964) <http://adsabs.harvard.edu/abs/1964AJ.....69...73H>`__ potential

    .. math::

        \\Phi(R,\\phi) = \\frac{\\mathrm{amp}}{2}\\,\\left[R^2 + \\frac{2\\,R^3}{3}\\,\\sin\\left(3\\,\\phi\\right)\\right]

    """

    def __init__(self, amp=1.0, ro=None, vo=None):
        """
        Initialize a Henon-Heiles potential

        Parameters
        ----------
        amp : float, optional
            Amplitude to be applied to the potential (default: 1.)
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2017-10-16 - Written - Bovy (UofT)
        """
        planarPotential.__init__(self, amp=amp, ro=ro, vo=vo)
        self.hasC = True
        self.hasC_dxdv = True

    def _evaluate(self, R, phi=0.0, t=0.0):
        return 0.5 * R * R * (1.0 + 2.0 / 3.0 * R * numpy.sin(3.0 * phi))

    def _Rforce(self, R, phi=0.0, t=0.0):
        return -R * (1.0 + R * numpy.sin(3.0 * phi))

    def _phitorque(self, R, phi=0.0, t=0.0):
        return -(R**3.0) * numpy.cos(3.0 * phi)

    def _R2deriv(self, R, phi=0.0, t=0.0):
        return 1.0 + 2.0 * R * numpy.sin(3.0 * phi)

    def _phi2deriv(self, R, phi=0.0, t=0.0):
        return -3.0 * R**3.0 * numpy.sin(3.0 * phi)

    def _Rphideriv(self, R, phi=0.0, t=0.0):
        return 3.0 * R**2.0 * numpy.cos(3.0 * phi)
