###############################################################################
#   BurkertPotential.py: Potential with a Burkert density
###############################################################################
import numpy
from scipy import special
from scipy.optimize import fsolve

from ..util import conversion
from .SphericalPotential import SphericalPotential


class EinastoPotential(SphericalPotential):
    """Potential with an Einasto [1]_ density. Class implements the following interchangeable conventions:

    .. math::
        \\rho(r) = \\mathrm{amp}\\,\\exp\\left(-d_n\\left[\\left(\\frac{r}{r_s}\\right)^\\frac{1}{n}-1\\right]\\right)

    or

    .. math::

        \\rho(r) = \\mathrm{amp}\\,\\exp\\left(-2n\\left[\\left(\\frac{r}{r_{-2}}\\right)^\\frac{1}{n}-1\\right]\\right)

    or

    .. math::

        \\rho(r) = \\mathrm{amp}\\,\\exp\\left(-\\left(\\frac{r}{h}\\right)^\\frac{1}{n}\\right)

    With conventions taken from [2]_.

    """

    def __init__(
        self, amp=1.0, h=2.0, n=1, rs=None, rm2=None, normalize=False, ro=None, vo=None
    ):
        """
        Initialize a Einasto-density potential [1]_.

        Parameters
        ----------
        amp : float or Quantity
            Amplitude to be applied to the potential. Can be a Quantity with units of mass density or Gxmass density.
        h : float or Quantity
            Scale length.
        rs : float or Quantity
            Radius of the sphere that contains half of the total mass.
        rm2 : float or Quantity
            Radius at which rho(r) ∝ r^-2.
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
        - Either specify h or rs or rm2.
        - 2025-09-12 - Written - John Weatherall

        References
        ----------
        .. [1] Einasto (1965), Trudy Inst. Astroz. Alma-Ata, No. 17, 1 ADS: https://ui.adsabs.harvard.edu/abs/1965TrAlm...5...87E.
        .. [2] Retana-Montenegro, E., Van Hese, E., Gentile, G., Baes, M., & Frutos-Alfaro, F. 2012, A&A, 540, A70 ADS: https://ui.adsabs.harvard.edu/abs/2012A&A...540A..70R.
        """
        SphericalPotential.__init__(self, amp=amp, ro=ro, vo=vo, amp_units="density")
        if rs is not None:
            rs = conversion.parse_length(rs, ro=self._ro, vo=self._vo)
            # convert to h
            dn = self._estimate_dn(n)
            dn = self._calculate_dn(n, dn)
            self.amp = amp * numpy.e**dn
            h = rs / (dn**n)
        elif rm2 is not None:
            rm2 = conversion.parse_length(rm2, ro=self._ro, vo=self._vo)
            # convert to h
            self.amp = amp * numpy.e ** (2 * n)
            h = rm2 / ((2 * n) ** n)
        else:
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

    def _estimate_dn(self, n):
        # see [2]
        return (
            3 * n
            - 1 / 3
            + (8 / (1215 * n))
            + (184 / (229635 * n**2))
            + (1048 / (31000725 * n**3))
            - (17557576 / (1242974068875 * n**4))
        )

    def _calculate_dn(self, n, est_dn):
        # use numerical solver
        def func(x):
            gamma_3n = special.gamma(3 * n)
            gamma_3n_upper = special.gammaincc(3 * n, x) * gamma_3n
            return 2 * gamma_3n_upper - gamma_3n

        return fsolve(func, est_dn)[0]
