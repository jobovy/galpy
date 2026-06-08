###############################################################################
#   DehnenBarPotential: Dehnen (2000)'s bar potential
###############################################################################
import numpy

from ..backend import get_namespace
from ..util import conversion
from .Potential import Potential

_degtorad = numpy.pi / 180.0


class DehnenBarPotential(Potential):
    """Class that implements the Dehnen bar potential (`Dehnen 2000 <http://adsabs.harvard.edu/abs/2000AJ....119..800D>`__; [1]_), generalized to 3D following `Monari et al. (2016) <http://adsabs.harvard.edu/abs/2016MNRAS.461.3835M>`__ [2]_

    .. math::

        \\Phi(R,z,\\phi) = A_b(t)\\,\\cos\\left(2\\,(\\phi-\\Omega_b\\,t)\\right))\\,\\left(\\frac{R}{r}\\right)^2\\,\\times \\begin{cases}
        -(R_b/r)^3\\,, & \\text{for}\\ r \\geq R_b\\\\
        (r/R_b)^3-2\\,, & \\text{for}\\ r\\leq R_b.
        \\end{cases}

    where :math:`r^2 = R^2+z^2` is the spherical radius and

    .. math::

        A_b(t) = A_f\\,\\left(\\frac{3}{16}\\xi^5-\\frac{5}{8}\\xi^3+\\frac{15}{16}\\xi+\\frac{1}{2}\\right)\\,, \\xi = 2\\frac{t/T_b-t_\\mathrm{form}}{T_\\mathrm{steady}}-1\\,,\\ \\mathrm{if}\\ t_\\mathrm{form} \\leq \\frac{t}{T_b} \\leq t_\\mathrm{form}+T_\\mathrm{steady}

    and

    .. math::

        A_b(t) = \\begin{cases}
        0\\,, & \\frac{t}{T_b} < t_\\mathrm{form}\\\\
        A_f\\,, & \\frac{t}{T_b} > t_\\mathrm{form}+T_\\mathrm{steady}
        \\end{cases}

    where

    .. math::

       T_b = \\frac{2\\pi}{\\Omega_b}

    is the bar period and the strength can also be specified using :math:`\\alpha`

    .. math::

       \\alpha = 3\\,\\frac{A_f}{v_0^2}\\,\\left(\\frac{R_b}{r_0}\\right)^3\\,.

    If the bar's pattern speed is zero, :math:`t_\\mathrm{form}` and :math:`t_\\mathrm{steady}` are straight times, not times divided by the bar period.

    """

    normalize = property()  # turn off normalize

    def __init__(
        self,
        amp=1.0,
        omegab=None,
        rb=None,
        chi=0.8,
        rolr=0.9,
        barphi=25.0 * _degtorad,
        tform=-4.0,
        tsteady=None,
        beta=0.0,
        alpha=0.01,
        Af=None,
        ro=None,
        vo=None,
    ):
        """
        Initialize a Dehnen bar potential.

        Parameters
        ----------
        amp : float, optional
            Amplitude to be applied to the potential (default: 1., see alpha or Ab below).
        omegab : float or Quantity, optional
            Rotation speed of the bar (can be Quantity).
        rb : float or Quantity, optional
            Bar radius (can be Quantity).
        Af : float or Quantity, optional
            Bar strength (can be Quantity).
        chi : float, optional
            Fraction R_bar / R_CR (corotation radius of bar).
        rolr : float or Quantity, optional
            Radius of the Outer Lindblad Resonance for a circular orbit (can be Quantity).
        barphi : float or Quantity, optional
            Angle between sun-GC line and the bar's major axis (in rad; default=25 degree; or can be Quantity).
        beta : float, optional
            Power law index of rotation curve (to calculate OLR, etc.).
        alpha : float or Quantity, optional
            Relative bar strength (default: 0.01).
        tform : float, optional
            Start of bar growth / bar period (default: -4).
        tsteady : float, optional
            Time from tform at which the bar is fully grown / bar period (default: -tform/2, so the perturbation is fully grown at tform/2).
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - Either provide (omegab, rb, Af) or (chi, rolr, alpha, beta).
        - 2010-11-24 - Started - Bovy (NYU)
        - 2017-06-23 - Converted to 3D following Monari et al. (2016) - Bovy (UofT/CCA)

        References
        ----------
        .. [1] Dehnen (1999). The Astrophysical Journal, 524, L35. ADS: https://ui.adsabs.harvard.edu/abs/1999ApJ...524L..35D/abstract
        .. [2] Monari, G., Famaey, B., Siebert, A., et al. (2016). Monthly Notices of the Royal Astronomical Society, 462(2), 2333-2346. ADS: https://ui.adsabs.harvard.edu/abs/2016MNRAS.462.2333M/abstract
        """
        Potential.__init__(self, amp=amp, ro=ro, vo=vo)
        barphi = conversion.parse_angle(barphi)
        rolr = conversion.parse_length(rolr, ro=self._ro)
        rb = conversion.parse_length(rb, ro=self._ro)
        omegab = conversion.parse_frequency(omegab, ro=self._ro, vo=self._vo)
        Af = conversion.parse_energy(Af, vo=self._vo)
        self.hasC = True
        self.hasC_dxdv = True
        self.hasC_dxdv3d = True
        self.isNonAxi = True
        self._barphi = barphi
        if omegab is None:
            self._rolr = rolr
            self._chi = chi
            self._beta = beta
            # Calculate omegab and rb
            self._omegab = 1.0 / (
                (self._rolr ** (1.0 - self._beta))
                / (1.0 + numpy.sqrt((1.0 + self._beta) / 2.0))
            )
            self._rb = self._chi * self._omegab ** (1.0 / (self._beta - 1.0))
            self._alpha = alpha
            self._af = self._alpha / 3.0 / self._rb**3.0
        else:
            self._omegab = omegab
            self._rb = rb
            self._af = Af
        self._tb = 2.0 * numpy.pi / self._omegab if self._omegab != 0.0 else 1.0
        self._tform = tform * self._tb
        if tsteady is None:
            self._tsteady = self._tform / 2.0
        else:
            self._tsteady = self._tform + tsteady * self._tb

    def _smooth(self, t):
        # Growth factor (0 before tform, smoothly to 1 at tsteady). The namespace
        # follows t itself: a concrete (Python/numpy) t -> numpy.where (a plain
        # coefficient, byte-identical to the original scalar/array branches, that
        # broadcasts into any backend's spatial arrays); a traced t (the
        # in-backend diffrax/torchdiffeq integrator, or autodiff wrt time) -> that
        # backend's where, so it is differentiable. Branch-free so a tracer works.
        xp = get_namespace(t)
        deltat = t - self._tform
        xi = 2.0 * deltat / (self._tsteady - self._tform) - 1.0
        growth = 3.0 / 16.0 * xi**5.0 - 5.0 / 8 * xi**3.0 + 15.0 / 16.0 * xi + 0.5
        return xp.where(t < self._tform, 0.0, xp.where(t < self._tsteady, growth, 1.0))

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        # Calculate relevant time
        smooth = self._smooth(t)
        xp = get_namespace(R, z, phi, t)
        r2 = R**2.0 + z**2.0
        r = xp.sqrt(r2)
        bad = r2 == 0.0
        # Safe r/r2 so the dead branch's 1/0 cannot poison reverse-mode gradients
        rsafe = xp.where(bad, xp.ones_like(r * 1.0), r)
        r2safe = xp.where(bad, xp.ones_like(r2 * 1.0), r2)
        # R^2/r^2 (== 1/(1+z^2/R^2)); at r=0, define it as 1 (matches the r==0
        # limit of the inner branch, which then gives the -2 special case). As
        # R -> inf with z fixed the naive R^2/r2 would be inf/inf = NaN, while
        # the true limit z^2/R^2 -> 0 gives 1, so substitute 1 there (using a
        # safe r2 to keep the dead branch finite under tracing).
        Rinf = xp.isinf(R)
        R2_over_r2 = xp.where(bad | Rinf, xp.ones_like(r2 * 1.0), R**2.0 / r2safe)
        factor = xp.where(
            r <= self._rb,
            (r / self._rb) ** 3.0 - 2.0,
            -((self._rb / rsafe) ** 3.0),
        )
        return (
            self._af
            * smooth
            * xp.cos(2.0 * (phi - self._omegab * t - self._barphi))
            * factor
            * R2_over_r2
        )

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        # Calculate relevant time
        smooth = self._smooth(t)
        xp = get_namespace(R, z, phi, t)
        r2 = R**2.0 + z**2.0
        r = xp.sqrt(r2)
        bad = r2 == 0.0
        rsafe = xp.where(bad, xp.ones_like(r * 1.0), r)
        r4safe = xp.where(bad, xp.ones_like(r2 * 1.0), r**4.0)
        inner = (
            -(
                (r / self._rb) ** 3.0 * R * (3.0 * R**2.0 + 2.0 * z**2.0)
                - 4.0 * R * z**2.0
            )
            / r4safe
        )
        outer = (
            -((self._rb / rsafe) ** 3.0) * R / r4safe * (3.0 * R**2.0 - 2.0 * z**2.0)
        )
        return (
            self._af
            * smooth
            * xp.cos(2.0 * (phi - self._omegab * t - self._barphi))
            * xp.where(r <= self._rb, inner, outer)
        )

    def _phitorque(self, R, z, phi=0.0, t=0.0):
        # Calculate relevant time
        smooth = self._smooth(t)
        xp = get_namespace(R, z, phi, t)
        r2 = R**2.0 + z**2.0
        r = xp.sqrt(r2)
        bad = r2 == 0.0
        rsafe = xp.where(bad, xp.ones_like(r * 1.0), r)
        r2safe = xp.where(bad, xp.ones_like(r2 * 1.0), r2)
        R2_over_r2 = R**2.0 / r2safe
        factor = xp.where(
            r <= self._rb,
            (r / self._rb) ** 3.0 - 2.0,
            -((self._rb / rsafe) ** 3.0),
        )
        return (
            2.0
            * self._af
            * smooth
            * xp.sin(2.0 * (phi - self._omegab * t - self._barphi))
            * factor
            * R2_over_r2
        )

    def _zforce(self, R, z, phi=0.0, t=0.0):
        # Calculate relevant time
        smooth = self._smooth(t)
        xp = get_namespace(R, z, phi, t)
        r2 = R**2.0 + z**2.0
        r = xp.sqrt(r2)
        bad = r2 == 0.0
        rsafe = xp.where(bad, xp.ones_like(r * 1.0), r)
        r4safe = xp.where(bad, xp.ones_like(r2 * 1.0), r**4.0)
        inner = -((r / self._rb) ** 3.0 + 4.0) * R**2.0 * z / r4safe
        outer = -5.0 * (self._rb / rsafe) ** 3.0 * R**2.0 * z / r4safe
        return (
            self._af
            * smooth
            * xp.cos(2.0 * (phi - self._omegab * t - self._barphi))
            * xp.where(r <= self._rb, inner, outer)
        )

    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        # Calculate relevant time
        smooth = self._smooth(t)
        xp = get_namespace(R, z, phi, t)
        r2 = R**2.0 + z**2.0
        r = xp.sqrt(r2)
        bad = r2 == 0.0
        rsafe = xp.where(bad, xp.ones_like(r * 1.0), r)
        r4safe = xp.where(bad, xp.ones_like(r2 * 1.0), r**4.0)
        r6safe = xp.where(bad, xp.ones_like(r2 * 1.0), r**6.0)
        inner = (r / self._rb) ** 3.0 * (
            (9.0 * R**2.0 + 2.0 * z**2.0) / r4safe
            - R**2.0 / r6safe * (3.0 * R**2.0 + 2.0 * z**2.0)
        ) + 4.0 * z**2.0 / r6safe * (4.0 * R**2.0 - r2)
        outer = (
            (self._rb / rsafe) ** 3.0
            / r6safe
            * ((r2 - 7.0 * R**2.0) * (3.0 * R**2.0 - 2.0 * z**2.0) + 6.0 * R**2.0 * r2)
        )
        return (
            self._af
            * smooth
            * xp.cos(2.0 * (phi - self._omegab * t - self._barphi))
            * xp.where(r <= self._rb, inner, outer)
        )

    def _phi2deriv(self, R, z, phi=0.0, t=0.0):
        # Calculate relevant time
        smooth = self._smooth(t)
        xp = get_namespace(R, z, phi, t)
        r2 = R**2.0 + z**2.0
        r = xp.sqrt(r2)
        bad = r2 == 0.0
        rsafe = xp.where(bad, xp.ones_like(r * 1.0), r)
        r2safe = xp.where(bad, xp.ones_like(r2 * 1.0), r2)
        R2_over_r2 = R**2.0 / r2safe
        factor = xp.where(
            r <= self._rb,
            -((r / self._rb) ** 3.0 - 2.0),
            (self._rb / rsafe) ** 3.0,
        )
        return (
            4.0
            * self._af
            * smooth
            * xp.cos(2.0 * (phi - self._omegab * t - self._barphi))
            * factor
            * R2_over_r2
        )

    def _Rphideriv(self, R, z, phi=0.0, t=0.0):
        # Calculate relevant time
        smooth = self._smooth(t)
        xp = get_namespace(R, z, phi, t)
        r2 = R**2.0 + z**2.0
        r = xp.sqrt(r2)
        bad = r2 == 0.0
        rsafe = xp.where(bad, xp.ones_like(r * 1.0), r)
        r4safe = xp.where(bad, xp.ones_like(r2 * 1.0), r**4.0)
        inner = (
            (r / self._rb) ** 3.0 * R * (3.0 * R**2.0 + 2.0 * z**2.0) - 4.0 * R * z**2.0
        ) / r4safe
        outer = (self._rb / rsafe) ** 3.0 * R / r4safe * (3.0 * R**2.0 - 2.0 * z**2.0)
        return (
            -2.0
            * self._af
            * smooth
            * xp.sin(2.0 * (phi - self._omegab * t - self._barphi))
            * xp.where(r <= self._rb, inner, outer)
        )

    def _z2deriv(self, R, z, phi=0.0, t=0.0):
        # Calculate relevant time
        smooth = self._smooth(t)
        xp = get_namespace(R, z, phi, t)
        r2 = R**2.0 + z**2.0
        r = xp.sqrt(r2)
        bad = r2 == 0.0
        rsafe = xp.where(bad, xp.ones_like(r * 1.0), r)
        r6safe = xp.where(bad, xp.ones_like(r2 * 1.0), r**6.0)
        inner = (
            R**2.0
            / r6safe
            * ((r / self._rb) ** 3.0 * (r2 - z**2.0) + 4.0 * (r2 - 4.0 * z**2.0))
        )
        outer = 5.0 * (self._rb / rsafe) ** 3.0 * R**2.0 / r6safe * (r2 - 7.0 * z**2.0)
        return (
            self._af
            * smooth
            * xp.cos(2.0 * (phi - self._omegab * t - self._barphi))
            * xp.where(r <= self._rb, inner, outer)
        )

    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        # Calculate relevant time
        smooth = self._smooth(t)
        xp = get_namespace(R, z, phi, t)
        r2 = R**2.0 + z**2.0
        r = xp.sqrt(r2)
        bad = r2 == 0.0
        rsafe = xp.where(bad, xp.ones_like(r * 1.0), r)
        r6safe = xp.where(bad, xp.ones_like(r2 * 1.0), r**6.0)
        inner = (
            R
            * z
            / r6safe
            * ((r / self._rb) ** 3.0 * (2.0 * r2 - R**2.0) + 8.0 * (r2 - 2.0 * R**2.0))
        )
        outer = (
            5.0 * (self._rb / rsafe) ** 3.0 * R * z / r6safe * (2.0 * r2 - 7.0 * R**2.0)
        )
        return (
            self._af
            * smooth
            * xp.cos(2.0 * (phi - self._omegab * t - self._barphi))
            * xp.where(r <= self._rb, inner, outer)
        )

    def _phizderiv(self, R, z, phi=0.0, t=0.0):
        # Calculate relevant time
        smooth = self._smooth(t)
        xp = get_namespace(R, z, phi, t)
        r2 = R**2.0 + z**2.0
        r = xp.sqrt(r2)
        bad = r2 == 0.0
        rsafe = xp.where(bad, xp.ones_like(r * 1.0), r)
        r4safe = xp.where(bad, xp.ones_like(r2 * 1.0), r**4.0)
        inner = -((r / self._rb) ** 3.0 + 4.0) * R**2.0 * z / r4safe
        outer = -5.0 * (self._rb / rsafe) ** 3.0 * R**2.0 * z / r4safe
        return (
            2.0
            * self._af
            * smooth
            * xp.sin(2.0 * (phi - self._omegab * t - self._barphi))
            * xp.where(r <= self._rb, inner, outer)
        )

    def tform(self):  # pragma: no cover
        """
        Return formation time of the bar.

        Returns
        -------
        tform : float
            Formation time of the bar in normalized units.

        Other Parameters
        ----------------
        none

        Notes
        -----
        - 2011-03-08 - Written - Bovy (NYU)

        """
        return self._tform

    def OmegaP(self):
        """
        Return the pattern speed.

        Returns
        -------
        float
            The pattern speed.

        Notes
        -----
        - 2011-10-10 - Written - Bovy (IAS)

        """
        return self._omegab
