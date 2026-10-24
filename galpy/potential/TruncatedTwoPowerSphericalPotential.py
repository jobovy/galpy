###############################################################################
#   TruncatedTwoPowerSphericalPotential.py: NFW and Hernquist potentials with
#                                           the Baltz, Marshall & Oguri (2009)
#                                           truncation, optionally with a
#                                           time-dependent truncation radius
###############################################################################
import functools
import math

import numpy
from scipy import interpolate as _scipy_interpolate

from ..util import conversion
from .SphericalPotential import SphericalPotential

# x = r/a below _SMALL_X min(1, rc/a): Taylor series (the closed form cancels);
# x above _LARGE_X max(1, rc/a): asymptotic series for the outer integral
_SMALL_X = 0.1
_LARGE_X = 10.0
# orders kept in both series: < 1e-15 relative at the switches
_NSERIES = 20


def _binom_neg(m, i):
    """binomial(-m, i), the coefficient of s^i in (1+s)^(-m)"""
    return (-1) ** i * math.comb(m + i - 1, i)


def _pf_coeffs(k, n, which, tau):
    """Partial-fraction coefficients of the enclosed-mass (``which='F'``,
    s / [(1+s)^k (1+s^2/tau^2)^n]) or outer-potential (``which='H'``,
    1 / [(1+s)^k (1+s^2/tau^2)^n]) integrand, for k = 2 (NFW) or 3 (Hernquist)
    and any integer n >= 1:

        sum_{j=1}^{k} A_j / (1+s)^j + sum_{j=1}^{n} (B_j s + C_j) / (s^2+tau^2)^j.

    Every coefficient is tau^(2n) P(tau^2) / (1+tau^2)^p with integer P;
    written with m = n-j+1 for the (s^2+tau^2)^(-j) terms. A_1 = -B_1, so the
    two logarithms combine into one that stays bounded at large s."""
    T = tau * tau
    opT = 1.0 + T
    rn = (T / opT) ** n

    def c(P, p):  # tau^(2n) P / (1+T)^p, without forming tau^(2n)
        return rn * P * opT ** (n - p)

    A, B, C = [], [], []
    if k == 2 and which == "F":
        A = [c(T - (2 * n - 1), n + 1), c(-1.0, n)]
        for m in range(n, 0, -1):
            B.append(c((2 * m - 1) - T, m + 1))
            C.append(c(2.0 * T - 2 * (m - 1), m + 1))
    elif k == 2:
        A = [c(2.0 * n, n + 1), c(1.0, n)]
        for m in range(n, 0, -1):
            B.append(c(-2.0 * m, m + 1))
            C.append(c((2 * m - 1) - T, m + 1))
    elif which == "F":
        A = [
            c(3 * n * T - n * (2 * n - 1), n + 2),
            c(T - (2 * n - 1), n + 1),
            c(-1.0, n),
        ]
        for m in range(n, 0, -1):
            B.append(c(m * (2 * m - 1) - 3 * m * T, m + 2))
            C.append(c(-T * T + (5 * m - 2) * T - (m - 1) * (2 * m - 1), m + 2))
    else:
        A = [c(n * (2 * n + 1) - n * T, n + 2), c(2.0 * n, n + 1), c(1.0, n)]
        for m in range(n, 0, -1):
            B.append(c(m * T - m * (2 * m + 1), m + 2))
            C.append(c(m * (2 * m - 1) - 3 * m * T, m + 2))
    return A, B, C


def _log_ratio(x, tau):
    """log[(1+x) / sqrt(1+x^2/tau^2)]: directly for x <= max(1, tau), and as
    log(tau) + log1p(1/x) - log1p(tau^2/x^2)/2 beyond, where the two logarithms
    would each grow (and overflow at x ~ 1e154) while their difference does not"""
    nu = numpy.clip(tau, 1.0, None)
    far = x > nu
    xn = numpy.where(far, numpy.zeros_like(x), x)
    xf = numpy.where(far, x, 2.0 * nu)
    near = numpy.log1p(xn) - 0.5 * numpy.log1p((xn / tau) ** 2)
    v = 1.0 / xf
    return numpy.where(
        far, numpy.log(tau) + numpy.log1p(v) - 0.5 * numpy.log1p((tau * v) ** 2), near
    )


def _F_closed(x, tau, k, n):
    """int_0^x s / [(1+s)^k (1+s^2/tau^2)^n] ds, closed form"""
    A, B, C = _pf_coeffs(k, n, "F", tau)
    # the terms below are 0 (or their limits) to double precision long before
    # x^2 overflows; clamping keeps the warnings and infs out
    xq = numpy.clip(x, None, 1e100)
    r = xq / tau
    L = numpy.log1p(r * r)
    lx = numpy.log1p(x)
    out = A[0] * _log_ratio(x, tau)
    for j in range(2, k + 1):  # A_j int (1+s)^-j = A_j [1 - (1+x)^(1-j)]/(j-1)
        out = out - A[j - 1] / (j - 1) * numpy.expm1(-(j - 1) * lx)
    for j in range(2, n + 1):  # B_j int s (s^2+tau^2)^-j
        out = out - B[j - 1] / (2 * (j - 1) * tau ** (2 * (j - 1))) * numpy.expm1(
            -(j - 1) * L
        )
    # C_j int_0^x (s^2+tau^2)^-j ds by the recursion
    # J_{j+1} = x / [2j tau^2 (x^2+tau^2)^j] + (2j-1) / (2j tau^2) J_j
    J = numpy.arctan(r) / tau
    out = out + C[0] * J
    q = xq * xq + tau * tau
    for j in range(1, n):
        J = (
            xq * (1.0 / q) ** j / (2 * j * tau * tau)
            + (2 * j - 1) / (2 * j * tau * tau) * J
        )
        out = out + C[j] * J
    return out


def _F_total(tau, k, n):
    """_F_closed at x = infinity: the total mass in units of amp"""
    A, B, C = _pf_coeffs(k, n, "F", tau)
    out = A[0] * numpy.log(tau)
    for j in range(2, k + 1):
        out = out + A[j - 1] / (j - 1)
    for j in range(2, n + 1):
        out = out + B[j - 1] / (2 * (j - 1) * tau ** (2 * (j - 1)))
    J = 0.5 * numpy.pi / tau
    out = out + C[0] * J
    for j in range(1, n):
        J = (2 * j - 1) / (2 * j * tau * tau) * J
        out = out + C[j] * J
    return out


def _H_closed(x, tau, k, n):
    """int_x^inf 1 / [(1+s)^k (1+s^2/tau^2)^n] ds, closed form"""
    A, B, C = _pf_coeffs(k, n, "H", tau)
    out = A[0] * (numpy.log(tau) - _log_ratio(x, tau))
    for j in range(2, k + 1):
        out = out + A[j - 1] / ((j - 1) * (1.0 + x) ** (j - 1))
    q = x * x + tau * tau
    for j in range(2, n + 1):
        out = out + B[j - 1] / (2 * (j - 1) * q ** (j - 1))
    # C_j int_x^inf (s^2+tau^2)^-j ds by the recursion
    # K_{j+1} = (2j-1) / (2j tau^2) K_j - x / [2j tau^2 (x^2+tau^2)^j]
    K = numpy.arctan2(tau, x) / tau
    out = out + C[0] * K
    for j in range(1, n):
        K = (2 * j - 1) / (2 * j * tau * tau) * K - x / (2 * j * tau * tau * q**j)
        out = out + C[j] * K
    return out


def _powers(x, m):
    """1, x, ..., x^m along a new last axis, by repeated products"""
    p = [numpy.ones_like(x)]
    for _ in range(m):
        p.append(p[-1] * x)
    return numpy.stack(p, axis=-1)


def _poly2(T, x, s):
    """sum_ij T[i, j] x^i s^j for a static table T, as one matrix product"""
    T = numpy.asarray(T, dtype=x.dtype)
    X = _powers(x, T.shape[0] - 1)
    return numpy.sum((X @ T) * _powers(s, T.shape[1] - 1), axis=-1)


@functools.lru_cache(maxsize=None)
def _series_tables(k, n):
    """Coefficients of the series, as tables T[i, j] of x^i s^j, i + 2j <= _NSERIES:
    the enclosed-mass integrand s (1+s)^-k (1+s^2/tau^2)^-n expanded about 0
    gives F(x) / x^2 with s = (x/tau)^2 (F) and its r2deriv sum_{m>=1} d_m
    m/(m+2) x^(m-1), m = i+2j (R1 for i >= 1 on x^(i-1); R0 for i = 0 on
    s^(j-1), times x/tau^2); the outer-potential integrand expanded in v = 1/x
    gives H / [v^(k-1) w^n] with w = (tau v)^2 (H). In the series regimes x, v
    <= 0.1 and s, w <= 0.01 for any tau, so every term is bounded"""
    N = _NSERIES
    F = numpy.zeros((N + 1, N // 2 + 1))
    R1 = numpy.zeros((N, N // 2 + 1))
    R0 = numpy.zeros((1, N // 2))
    H = numpy.zeros((N + 1, N // 2 + 1))
    for i in range(N + 1):
        for j in range((N - i) // 2 + 1):
            m = i + 2 * j
            c = _binom_neg(k, i) * _binom_neg(n, j)
            F[i, j] = c / (m + 2)
            if i > 0:
                R1[i - 1, j] = c * m / (m + 2)
            elif j > 0:
                R0[0, j - 1] = c * m / (m + 2)
            H[i, j] = c / (k + 2 * n - 1 + m)
    return F, R1, R0, H


def _H_series(x, tau, k, n):
    """Asymptotic series of _H_closed for x > max(1, tau): the integrand
    tau^(2n) s^(-k-2n) (1+1/s)^-k (1+tau^2/s^2)^-n expanded in 1/s"""
    v = 1.0 / x
    w2 = (tau * v) ** 2
    return v ** (k - 1) * w2**n * _poly2(_series_tables(k, n)[3], v, w2)


def _F_series(x, tau, k, n, p):
    """F(x) / x^p from the Taylor series, p = 0, 1, 2 (no division at x=0)"""
    P = _poly2(_series_tables(k, n)[0], x, (x / tau) ** 2)
    return x ** (2 - p) * P if p < 2 else P


def _r2deriv_series(x, tau, k, n):
    """a^3 d^2Phi/dr^2 / amp from the Taylor series: the 1/x terms of 4 pi rho
    and -2 M/r^3 cancel analytically"""
    _, R1, R0, _ = _series_tables(k, n)
    s = (x / tau) ** 2
    return _poly2(R1, x, s) + x / (tau * tau) * _poly2(R0, x, s)


class TruncatedTwoPowerSphericalPotential(SphericalPotential):
    r"""Base class for the NFW and Hernquist profiles with the closed-form
    truncation of Baltz, Marshall & Oguri (2009),

    .. math::

        \rho(r) = \frac{\mathrm{amp}}{4\,\pi\,a^3}\,\frac{1}{(r/a)\,(1+r/a)^{k}}\,\left(\frac{r_c^2}{r_c^2+r^2}\right)^n\,,

    with :math:`k = 2` (NFW) or :math:`k = 3` (Hernquist). Use
    :class:`TruncatedNFWPotential` or :class:`TruncatedHernquistPotential`.

    The closed forms come from the partial-fraction decomposition of the
    (rational) enclosed-mass and outer-potential integrands, whose coefficients
    are written out in ``_pf_coeffs`` for these two inner profiles and any
    integer :math:`n`. Other integer two-power profiles (e.g., Jaffe) follow
    from the same decomposition and only need their own coefficients.
    """

    _k = None  # (1+r/a) exponent of the inner profile; set by the subclasses

    def __init__(
        self,
        amp=1.0,
        a=1.0,
        rc=10.0,
        n=1,
        tgrid=None,
        normalize=False,
        ro=None,
        vo=None,
    ):
        SphericalPotential.__init__(self, amp=amp, ro=ro, vo=vo, amp_units="mass")
        if int(n) != n or n < 1:
            raise ValueError(f"n= must be a positive integer, got {n!r}")
        self._n = int(n)
        a = conversion.parse_length(a, ro=self._ro)
        self._tdep = tgrid is not None
        if self._tdep:
            if isinstance(tgrid, (list, tuple)):
                tgrid = numpy.asarray(tgrid, dtype=float)
            if isinstance(rc, (list, tuple)):
                rc = numpy.asarray(rc, dtype=float)
            tgrid = numpy.asarray(
                conversion.parse_time(tgrid, ro=self._ro, vo=self._vo)
            )
            if tgrid.ndim != 1 or len(tgrid) < 2 or numpy.any(numpy.diff(tgrid) <= 0.0):
                raise ValueError(
                    "tgrid= must be a strictly increasing 1D array of >= 2 times"
                )
            rc = numpy.asarray(conversion.parse_length(rc, ro=self._ro), dtype=float)
            if rc.shape != tgrid.shape:
                raise ValueError("rc= must have one value per tgrid time")
        else:
            rc = conversion.parse_length(rc, ro=self._ro)
        if numpy.any(numpy.asarray(rc) <= 0.0):
            raise ValueError("rc= must be positive")
        self.a = a
        self._scale = self.a
        if self._tdep:
            self._tgrid = tgrid
            self._rc_grid = rc
            # log rc(t) is the not-a-knot spline through log rc, so rc stays
            # positive; only its power-basis coefficients are kept (scipy
            # splines do not pickle), evaluated with the same Horner arithmetic
            # as the C implementation
            self._rc_c = _scipy_interpolate.CubicSpline(
                tgrid, numpy.log(rc), bc_type="not-a-knot"
            ).c
            self.rc = None
        else:
            self.rc = rc
        if normalize or (
            isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
        ):
            self.normalize(normalize)
        self.hasC = True
        self.hasC_dxdv = True
        self.hasC_dxdv3d = True
        self.hasC_dens = True
        return None

    def _rc(self, t):
        """The truncation radius at time t (held constant outside tgrid)"""
        if not self._tdep:
            return self.rc
        t = numpy.clip(numpy.asarray(t, dtype=float), self._tgrid[0], self._tgrid[-1])
        i = numpy.clip(
            numpy.searchsorted(self._tgrid, t, side="right") - 1,
            0,
            self._rc_c.shape[1] - 1,
        )
        dt = t - self._tgrid[i]
        out = self._rc_c[0, i]
        for j in range(1, self._rc_c.shape[0]):
            out = out * dt + self._rc_c[j, i]
        return numpy.exp(out)

    def _x_tau(self, r, t):
        """(x, tau, mu, nu): x = r/a, tau = rc(t)/a on x's shape, mu = min(1, tau)
        and nu = max(1, tau)"""
        r = numpy.asarray(r) * 1.0
        tau = (numpy.zeros_like(r) + self._rc(t)) / self.a
        return r / self.a, tau, numpy.clip(tau, None, 1.0), numpy.clip(tau, 1.0, None)

    def _Fx(self, x, tau, mu, p):
        """F(x) / x^p, F the enclosed mass in units of amp: the Taylor series
        below the switch, the closed form above it; each evaluated only where
        it is used (the other at a safe point, so neither overflows or divides
        by zero)"""
        k, n = self._k, self._n
        small = x < _SMALL_X * mu
        xs = numpy.where(small, x, numpy.zeros_like(x))
        xc = numpy.where(small, _SMALL_X * mu + numpy.zeros_like(x), x)
        series = _F_series(xs, tau, k, n, p)
        closed = _F_closed(xc, tau, k, n)
        if p > 0:
            closed = closed / xc**p
        return numpy.where(small, series, closed)

    def _H(self, x, tau, nu):
        """int_x^inf 4 pi r rho dr in units of amp/a"""
        k, n = self._k, self._n
        large = x > _LARGE_X * nu
        xl = numpy.where(large, x, 2.0 * _LARGE_X * nu + numpy.zeros_like(x))
        xc = numpy.where(large, _LARGE_X * nu + numpy.zeros_like(x), x)
        return numpy.where(large, _H_series(xl, tau, k, n), _H_closed(xc, tau, k, n))

    def _rdens(self, r, t=0.0):
        x, tau, _, _ = self._x_tau(r, t)
        a = self.a
        return (
            (1.0 + (x / tau) ** 2) ** (-self._n)
            / (x * (1.0 + x) ** self._k)
            / (4.0 * numpy.pi * a**3)
        )

    def _revaluate(self, r, t=0.0):
        x, tau, mu, nu = self._x_tau(r, t)
        a = self.a
        return -(self._Fx(x, tau, mu, 1) + self._H(x, tau, nu)) / a

    def _rforce(self, r, t=0.0):
        x, tau, mu, _ = self._x_tau(r, t)
        a = self.a
        return -self._Fx(x, tau, mu, 2) / (a * a)

    def _r2deriv(self, r, t=0.0):
        x, tau, mu, _ = self._x_tau(r, t)
        k, n = self._k, self._n
        a = self.a
        small = x < _SMALL_X * mu
        xs = numpy.where(small, x, numpy.zeros_like(x))
        xc = numpy.where(small, _SMALL_X * mu + numpy.zeros_like(x), x)
        series = _r2deriv_series(xs, tau, k, n)
        # 4 pi rho - 2 M / r^3
        closed = (1.0 + (xc / tau) ** 2) ** (-n) / (
            xc * (1.0 + xc) ** k
        ) - 2.0 * _F_closed(xc, tau, k, n) / xc**3
        return numpy.where(small, series, closed) / a**3

    def _mass(self, R, z=None, t=0.0):
        if z is not None:
            raise AttributeError  # use the general implementation
        x, tau, mu, _ = self._x_tau(R, t)
        fin = numpy.isfinite(x)
        xf = numpy.where(fin, x, numpy.zeros_like(x))
        return numpy.where(
            fin,
            self._Fx(xf, tau, mu, 0),
            _F_total(tau, self._k, self._n) + numpy.zeros_like(xf),
        )

    def _c_args(self):
        """The arguments of the C implementation: amp, a, k, n, Nt, then rc
        (static) or tgrid and the coefficients of the log rc spline"""
        args = [self._amp, self.a, self._k, self._n]
        if not self._tdep:
            return args + [0, self.rc]
        return (
            args + [len(self._tgrid)] + list(self._tgrid) + list(self._rc_c.T.flatten())
        )


class TruncatedNFWPotential(TruncatedTwoPowerSphericalPotential):
    r"""Class that implements the NFW potential with the truncation of Baltz,
    Marshall & Oguri (2009)

    .. math::

        \rho(r) = \frac{\mathrm{amp}}{4\,\pi\,a^3}\,\frac{1}{(r/a)\,(1+r/a)^{2}}\,\left(\frac{r_c^2}{r_c^2+r^2}\right)^n\,,

    the :ref:`NFW <nfw>` profile with an outer fall-off steeper by
    :math:`r^{-2n}` beyond the truncation radius :math:`r_c`, which can depend
    on time. The enclosed mass and the potential are closed-form (logarithms,
    arctangents and rational functions) for any integer :math:`n \geq 1`,
    with series expansions where the closed forms lose precision (small
    :math:`r`, and the potential's outer integral at large :math:`r`). Has a C
    implementation, enabling fast orbit integration and the full 3D variational
    equations (``Orbit.integrate_dxdv``).

    """

    _k = 2

    def __init__(
        self,
        amp=1.0,
        a=1.0,
        rc=10.0,
        n=1,
        tgrid=None,
        normalize=False,
        ro=None,
        vo=None,
    ):
        """
        Initialize a truncated NFW potential.

        Parameters
        ----------
        amp : float or Quantity, optional
            Amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass or Gxmass. This is the NFW mass-scale :math:`4\\pi\\,\\rho_s\\,a^3`, matching the ``amp`` convention of :ref:`NFWPotential <nfw>` (as :math:`r_c \\to \\infty` the two potentials coincide for equal ``amp`` and ``a``).
        a : float or Quantity, optional
            Scale radius.
        rc : float, Quantity, or array, optional
            Truncation radius; with ``tgrid``, its values at the ``tgrid`` times.
        n : int, optional
            Exponent of the truncation (a positive integer; default: 1).
        tgrid : numpy.ndarray or Quantity, optional
            Times at which ``rc`` is given, for a time-dependent truncation radius. ``log rc(t)`` is the not-a-knot cubic spline through the logarithms of the given values, so that ``rc`` stays positive (a spline through ``rc`` itself can dip below zero); like any spline it can overshoot between the given values (e.g., a radius that drops quickly and then levels off can rebound by a few percent before the last value). ``rc`` is held at its end values outside ``tgrid``; ``amp`` and ``a`` do not change, so the inner profile stays fixed while the mass beyond :math:`r_c` changes.
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1. (at t=0).
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - Baltz, Marshall & Oguri (2009), JCAP 01, 015 (arXiv:0705.0682).
        - 2026-10-23 - Written - Bovy (UofT) & Claude Code

        """
        TruncatedTwoPowerSphericalPotential.__init__(
            self,
            amp=amp,
            a=a,
            rc=rc,
            n=n,
            tgrid=tgrid,
            normalize=normalize,
            ro=ro,
            vo=vo,
        )


class TruncatedHernquistPotential(TruncatedTwoPowerSphericalPotential):
    r"""Class that implements the Hernquist potential with the truncation of
    Baltz, Marshall & Oguri (2009)

    .. math::

        \rho(r) = \frac{\mathrm{amp}}{4\,\pi\,a^3}\,\frac{1}{(r/a)\,(1+r/a)^{3}}\,\left(\frac{r_c^2}{r_c^2+r^2}\right)^n\,,

    the :ref:`Hernquist <hernquist>` profile with an outer fall-off steeper by
    :math:`r^{-2n}` beyond the truncation radius :math:`r_c`, which can depend
    on time. The enclosed mass and the potential are closed-form (logarithms,
    arctangents and rational functions) for any integer :math:`n \geq 1`,
    with series expansions where the closed forms lose precision (small
    :math:`r`, and the potential's outer integral at large :math:`r`). Has a C
    implementation, enabling fast orbit integration and the full 3D variational
    equations (``Orbit.integrate_dxdv``).

    """

    _k = 3

    def __init__(
        self,
        amp=1.0,
        a=1.0,
        rc=10.0,
        n=1,
        tgrid=None,
        normalize=False,
        ro=None,
        vo=None,
    ):
        """
        Initialize a truncated Hernquist potential.

        Parameters
        ----------
        amp : float or Quantity, optional
            Amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass or Gxmass. This matches the ``amp`` convention of :ref:`HernquistPotential <hernquist>` (as :math:`r_c \\to \\infty` the two potentials coincide for equal ``amp`` and ``a``; the untruncated total mass is amp/2).
        a : float or Quantity, optional
            Scale radius.
        rc : float, Quantity, or array, optional
            Truncation radius; with ``tgrid``, its values at the ``tgrid`` times.
        n : int, optional
            Exponent of the truncation (a positive integer; default: 1).
        tgrid : numpy.ndarray or Quantity, optional
            Times at which ``rc`` is given, for a time-dependent truncation radius. ``log rc(t)`` is the not-a-knot cubic spline through the logarithms of the given values, so that ``rc`` stays positive (a spline through ``rc`` itself can dip below zero); like any spline it can overshoot between the given values (e.g., a radius that drops quickly and then levels off can rebound by a few percent before the last value). ``rc`` is held at its end values outside ``tgrid``; ``amp`` and ``a`` do not change, so the inner profile stays fixed while the mass beyond :math:`r_c` changes.
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1. (at t=0).
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - Baltz, Marshall & Oguri (2009), JCAP 01, 015 (arXiv:0705.0682).
        - 2026-10-23 - Written - Bovy (UofT) & Claude Code

        """
        TruncatedTwoPowerSphericalPotential.__init__(
            self,
            amp=amp,
            a=a,
            rc=rc,
            n=n,
            tgrid=tgrid,
            normalize=normalize,
            ro=ro,
            vo=vo,
        )
