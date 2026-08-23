###############################################################################
#   EinastoPotential.py: Potential with an Einasto density
###############################################################################
import numpy
from scipy import special
from scipy.optimize import fsolve

from ..backend import coerce_coords, get_namespace, is_backend_array
from ..backend.optimize import brentq
from ..backend.special import gamma as _gamma
from ..backend.special import gammaincc as _gammaincc
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
        # Under a forced backend the params still arrive as plain Python floats,
        # so the d_n solve below would fall through to scipy and the potential
        # would never be built ON the backend. coerce_coords lifts a float to
        # the backend's float64 and is a strict pass-through when xp is numpy,
        # so the numpy path stays byte-identical. Must precede the solve.
        (n,) = coerce_coords(get_namespace(n), n)
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
        self._backend_compatible = True
        if normalize or (
            isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
        ):  # pragma: no cover
            self.normalize(normalize)
        self.hasC = True
        self.hasC_dxdv = True
        self.hasC_dxdv3d = True  # full 3D Hessian (R2deriv/z2deriv/Rzderiv) in C
        self.hasC_dens = True
        return None

    def _revaluate(self, r, t=0.0):
        """Potential as a function of r and time"""
        xp = get_namespace(r)
        s = r / self.h
        # r == 0 is handled by the separate `core` branch below; eager backends
        # evaluate BOTH xp.where branches, so the generic branch must stay
        # NaN-free there: its (1-Q)/s term is 0/0 at s == 0 and, for n > 1,
        # d(s**(1/n))/ds is infinite at s == 0 (which would NaN-poison reverse-
        # mode autodiff). Evaluate the dead branch at the safe s == 1 instead.
        ssafe = xp.where(r == 0, 1.0, s)
        gamma_3n = _gamma(3 * self.n)
        gamma_2n = _gamma(2 * self.n)
        gamma_upper_3n = _gammaincc(3 * self.n, (ssafe ** (1 / self.n)))
        gamma_upper_2n = _gammaincc(2 * self.n, (ssafe ** (1 / self.n)))
        # written to handle s = numpy.inf
        out = -(4 * numpy.pi * (self.h**2) * self.n * gamma_3n) * (
            (1 - gamma_upper_3n) / ssafe + gamma_upper_2n * (gamma_2n / gamma_3n)
        )
        core = -(4 * numpy.pi * (self.h**2) * self.n) * _gamma(2 * self.n)
        return xp.where(r == 0, core, out)

    def _rforce(self, r, t=0.0):
        s = r / self.h
        gamma_3n = _gamma(3 * self.n)
        gamma_upper_3n = _gammaincc(3 * self.n, (s ** (1 / self.n)))
        return (
            (4 * numpy.pi * self.h * self.n * gamma_3n) * (s**-2) * (gamma_upper_3n - 1)
        )

    def _r2deriv(self, r, t=0.0):
        s = r / self.h
        gamma_3n = _gamma(3 * self.n)
        gamma_upper_3n = _gammaincc(3 * self.n, (s ** (1 / self.n)))
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
        if not is_backend_array(n):
            # numpy: unchanged, so this stays byte-identical. The original uses
            # scipy's fsolve from the series GUESS; routing it through galpy's
            # bracketing brentq instead would move the last bits, so the numpy
            # path is deliberately left alone rather than unified.
            def func(x):
                gamma_3n = special.gamma(3 * n)
                gamma_3n_upper = special.gammaincc(3 * n, x) * gamma_3n
                return 2 * gamma_3n_upper - gamma_3n

            return fsolve(func, est_dn)[0]

        # Backend: same root, differentiable via the implicit function theorem
        # (galpy's brentq). n is passed through ``args`` rather than closed over
        # because brentq follows the DATA -- a closed-over backend value would
        # dispatch to scipy and be silently non-differentiable.
        #
        # The residual is the REGULARIZED form 2*Q(3n,x) - 1. It has the same
        # root as the numpy form Gamma(3n)*(2Q-1), and the same implicit-diff
        # gradient (dn'= -(dF/dn)/(dF/dx); the Gamma factor is common to both
        # partials at the root and cancels), while avoiding the Gamma(3n)
        # overflow the numpy form hits for n >~ 57.
        #
        # Bracket: d_n is the MEDIAN of Gamma(3n), so 0 < d_n < 3n because a
        # gamma median is below its mean. That holds for EVERY n, unlike a
        # bracket around the series estimate, which is 14x off at n = 0.1.
        def froot(x, nn):
            return 2.0 * _gammaincc(3.0 * nn, x) - 1.0

        return brentq(froot, 1e-12, 3.0 * n, args=(n,))
