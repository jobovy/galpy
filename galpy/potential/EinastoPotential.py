###############################################################################
#   BurkertPotential.py: Potential with a Burkert density
###############################################################################
import numpy
from scipy import special

from ..util import conversion
from .SphericalPotential import SphericalPotential


class EinastoPotential(SphericalPotential):
    """EinastoPotential.py: Potential with an Einasto density

    .. math::
        \\rho(r) = \\mathrm{amp}\\,\\exp\\left(-\\left(\\frac{r}{h}\\right)^\\frac{1}{n}\\right)

    """

    def __init__(self, amp=1.0, h=2.0, n=1, normalize=False, ro=None, vo=None):
        """
        Initialize a Einasto-density potential [1]_.

        Parameters
        ----------
        amp : float or Quantity
            Amplitude to be applied to the potential. Can be a Quantity with units of mass density or Gxmass density.
        h : float or Quantity
            Scale length.
        n : float
            The Einasto index. A shape parameter defining the steepness of the power law
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1. Default is False.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2025-09-12 - Written - John Weatherall

        References
        ----------
        .. [1] Einasto (1965), Trudy Inst. Astroz. Alma-Ata, No. 17, 1 ADS: https://ui.adsabs.harvard.edu/abs/1965TrAlm...5...87E.
        """
        SphericalPotential.__init__(self, amp=amp, ro=ro, vo=vo, amp_units="density")
        h = conversion.parse_length(h, ro=self._ro, vo=self._vo)
        self.h = h
        self.n = n
        self._scale = self.h
        if normalize or (
            isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
        ):  # pragma: no cover
            self.normalize(normalize)
        self.hasC = True
        self.hasC_dxdv = True
        self.hasC_dens = True
        return None

    def _revaluate(self, r, t=0.0):
        """Potential as a function of r and time"""

        s = r / self.h
        gamma_3n = special.gamma(3 * self.n)
        gamma_2n = special.gamma(2 * self.n)
        gamma_upper_3n = special.gammaincc(3 * self.n, (s ** (1 / self.n)))
        gamma_upper_2n = special.gammaincc(2 * self.n, (s ** (1 / self.n)))

        # written to handle s = numpy.inf
        out = -(4 * numpy.pi * (self.h**2) * self.n * gamma_3n) * (
            (1 - gamma_upper_3n) / s + gamma_upper_2n * (gamma_2n / gamma_3n)
        )

        core = -(4 * numpy.pi * (self.h**2) * self.n) * special.gamma(2 * self.n)

        if isinstance(r, (float, int)):
            if r == 0:
                return core
            else:
                return out
        else:
            out[r == 0] = core
            return out

    def _rforce(self, r, t=0.0):
        s = r / self.h
        gamma_3n = special.gamma(3 * self.n)
        gamma_upper_3n = special.gammaincc(3 * self.n, (s ** (1 / self.n)))

        return (
            (4 * numpy.pi * self.h * self.n * gamma_3n) * (s**-2) * (gamma_upper_3n - 1)
        )

    def _r2deriv(self, r, t=0.0):
        s = r / self.h
        gamma_3n = special.gamma(3 * self.n)
        gamma_upper_3n = special.gammaincc(3 * self.n, (s ** (1 / self.n)))
        # (self.h**2)
        return -(4 * numpy.pi * self.n * gamma_3n) * (
            (-2 * (s**-3)) * (gamma_upper_3n - 1)
            - ((1 / self.n) * (numpy.e ** -(s ** (1 / self.n))) / gamma_3n)
        )

    def _rdens(self, r, t=0.0):
        return numpy.e ** -((r / self.h) ** (1 / self.n))
