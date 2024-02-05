import numpy

from ..util import conversion
from ..util._optional_deps import _APY_LOADED
from .linearPotential import linearPotential

if _APY_LOADED:
    from astropy import units


class KGPotential(linearPotential):
    """Class representing the Kuijken & Gilmore (1989) potential

    .. math::

        \\Phi(x) = \\mathrm{amp}\\,\\left(K\\,\\left(\\sqrt{x^2+D^2}-D\\right)+F\\,x^2\\right)

    """

    def __init__(self, amp=1.0, K=1.15, F=0.03, D=1.8, ro=None, vo=None):
        """
        Initialize a KGPotential

        Parameters
        ----------
        amp : float or Quantity, optional
            An overall amplitude, by default 1.0
        K : float or Quantity, optional
            K parameter (= 2πΣ_disk; specify either as surface density or directly as force [i.e., including 2πG]), by default 1.15
        F : float or Quantity, optional
            F parameter (= 4πρ_halo; specify either as density or directly as second potential derivative [i.e., including 4πG]), by default 0.03
        D : float or Quantity, optional
            D parameter (natural units or Quantity length units), by default 1.8
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2010-07-12 - Written - Bovy (NYU)

        """
        linearPotential.__init__(self, amp=amp, ro=ro, vo=vo)
        D = conversion.parse_length(D, ro=self._ro)
        K = conversion.parse_force(K, ro=self._ro, vo=self._vo)
        if _APY_LOADED and isinstance(F, units.Quantity):
            try:
                F = (
                    F.to(units.Msun / units.pc**3).value
                    / conversion.dens_in_msolpc3(self._vo, self._ro)
                    * 4.0
                    * numpy.pi
                )
            except units.UnitConversionError:
                pass
        if _APY_LOADED and isinstance(F, units.Quantity):
            try:
                F = F.to(units.km**2 / units.s**2 / units.kpc**2).value * ro**2 / vo**2
            except units.UnitConversionError:
                raise units.UnitConversionError(
                    "Units for F not understood; should be density"
                )
        self._K = K
        self._F = F
        self._D = D
        self._D2 = self._D**2.0
        self.hasC = True

    def _evaluate(self, x, t=0.0):
        return self._K * (numpy.sqrt(x**2.0 + self._D2) - self._D) + self._F * x**2.0

    def _force(self, x, t=0.0):
        return -x * (self._K / numpy.sqrt(x**2 + self._D2) + 2.0 * self._F)
