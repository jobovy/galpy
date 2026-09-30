###############################################################################
#   special.py: special functions galpy implements itself (spherical
#               harmonics, incomplete beta)
###############################################################################
import functools

import numpy
import scipy
from packaging.version import parse as parse_version
from scipy import special
from scipy.special import gammaln

_SCIPY_VERSION = parse_version(scipy.__version__)
if _SCIPY_VERSION < parse_version("1.15"):  # pragma: no cover
    from scipy.special import lpmn
else:
    from scipy.special import assoc_legendre_p_all


def compute_legendre(costheta, L, M, deriv=False):
    """
    Compute associated Legendre polynomials P_l^m(cos(theta)).

    Parameters
    ----------
    costheta : float
        Cosine of the polar angle.
    L : int
        Maximum degree + 1 (compute for 0 <= l < L).
    M : int
        Maximum order + 1 (compute for 0 <= m < M).
    deriv : bool or int, optional
        If False, only return P. If True or 1, also return dP/dx. If 2, also return d²P/dx².

    Returns
    -------
    PP : numpy.ndarray
        Associated Legendre polynomials, shape (L, M).
    dPP : numpy.ndarray
        Derivative with respect to costheta, shape (L, M). Only returned if deriv >= 1.
    d2PP : numpy.ndarray
        Second derivative with respect to costheta, shape (L, M). Only returned if deriv == 2.

    Notes
    -----
    - 2026-02-11 - Written - Bovy (UofT)
    - 2026-02-13 - Moved to galpy.util.special - Bovy (UofT)
    - 2026-02-18 - Added deriv=2 support - Bovy (UofT)
    """
    if _SCIPY_VERSION < parse_version("1.15"):  # pragma: no cover
        if deriv:
            PP, dPP = lpmn(M - 1, L - 1, costheta)
            PP = PP.T
            dPP = dPP.T
            if deriv == 2:
                d2PP = _compute_legendre_2nd_deriv(PP, dPP, costheta, L, M)
                return PP, dPP, d2PP
            return PP, dPP
        return lpmn(M - 1, L - 1, costheta)[0].T
    if deriv == 2:
        result = assoc_legendre_p_all(L - 1, M - 1, costheta, branch_cut=2, diff_n=2)
        PP = numpy.swapaxes(result[0][:, :M], 0, 1).T
        dPP = numpy.swapaxes(result[1][:, :M], 0, 1).T
        d2PP = numpy.swapaxes(result[2][:, :M], 0, 1).T
        return PP, dPP, d2PP
    if deriv:
        PP, dPP = assoc_legendre_p_all(L - 1, M - 1, costheta, branch_cut=2, diff_n=1)
        return (
            numpy.swapaxes(PP[:, :M], 0, 1).T,
            numpy.swapaxes(dPP[:, :M], 0, 1).T,
        )
    return numpy.swapaxes(
        assoc_legendre_p_all(L - 1, M - 1, costheta, branch_cut=2)[0, :, :M],
        0,
        1,
    ).T


def _compute_legendre_2nd_deriv(PP, dPP, x, L, M):  # pragma: no cover
    """
    Compute d²P_l^m/dx² from the Legendre differential equation for scipy < 1.15.

    Uses: (1-x²) d²P/dx² - 2x dP/dx + [l(l+1) - m²/(1-x²)] P = 0
    => d²P/dx² = [2x dP/dx - l(l+1) P + m²/(1-x²) P] / (1-x²)

    At the poles (x = ±1), d²P/dx² diverges for m > 0. To avoid numerical
    issues, we clamp x slightly away from ±1 and recompute P and dP there.
    This is safe because the divergent d²P/dx² is always multiplied by sin²θ
    in the physical second derivative d²P/dθ², making the product finite.

    Parameters
    ----------
    PP : numpy.ndarray
        Associated Legendre polynomials, shape (L, M).
    dPP : numpy.ndarray
        First derivatives, shape (L, M).
    x : float
        costheta value.
    L : int
        Maximum degree + 1.
    M : int
        Maximum order + 1.

    Returns
    -------
    d2PP : numpy.ndarray
        Second derivatives, shape (L, M).
    """
    if abs(1.0 - x * x) < 1e-14:
        x = numpy.sign(x) * (1.0 - 1e-7) if x != 0.0 else x
        PP, dPP = lpmn(M - 1, L - 1, x)
        PP = PP.T
        dPP = dPP.T
    d2PP = numpy.zeros((L, M))
    one_minus_x2 = 1.0 - x * x
    for l in range(L):
        for m in range(min(l + 1, M)):
            d2PP[l, m] = (
                2.0 * x * dPP[l, m]
                - l * (l + 1) * PP[l, m]
                + m * m / one_minus_x2 * PP[l, m]
            ) / one_minus_x2
    return d2PP


def sph_harm_normalization(L, M):
    """
    Compute the spherical harmonics normalization factor N_lm.

    Returns beta_lm = sqrt((2l+1)/(4*pi) * (l-m)!/(l+m)!) * (2 - delta_{m,0})
    for 0 <= l < L and 0 <= m < M.

    Parameters
    ----------
    L : int
        Maximum degree + 1 (compute for 0 <= l < L).
    M : int
        Maximum order + 1 (compute for 0 <= m < M).

    Returns
    -------
    numpy.ndarray
        Normalization factors, shape (L, M). Entries where m > l are zero.

    Notes
    -----
    - 2016-05-16 - Written as _Nroot - Aladdin Seaifan (UofT)
    - 2026-02-13 - Moved to galpy.util.special - Bovy (UofT)
    """
    NN = numpy.zeros((L, M), float)
    l = numpy.arange(0, L)[:, numpy.newaxis]
    m = numpy.arange(0, M)[numpy.newaxis, :]
    nLn = gammaln(l - m + 1) - gammaln(l + m + 1)
    NN[:, :] = ((2 * l + 1.0) / (4.0 * numpy.pi) * numpy.e**nLn) ** 0.5 * 2
    NN[:, 0] /= 2.0
    NN = numpy.tril(NN)
    return NN


###############################################################################
#   Incomplete beta B_z(p, q) = int_0^z u^(p-1) (1-u)^(q-1) du for p > 0, q > -1
#   (scipy's betainc needs q > 0 and loses digits as q -> 0), accurate to
#   round-off including q -> 0, q -> -1 and large p + q; the C version is
#   galpy/util/incomplete_beta.c. Used by TwoPowerSphericalPotential.
###############################################################################
def incomplete_beta_split(p, q):
    """The split point c of incomplete_beta: the integrand's mass centre
    (p+1)/(p+q+2), at most 0.9"""
    return min((p + 1.0) / (p + q + 2.0), 0.9)


_IBETA_QSMALL = 0.05


def incomplete_beta_k_series(p, q, s):
    """K(s) = ((1-s)^p 2F1(1, p+q; q+1; s) - 1)/q, or its q = 0 limit.

    Summed as (1-s)^p sum_k (p)_k/k! s^k D_k, D_k = (R_k - 1)/q with R_k the
    Pochhammer ratio (p+q)_k k! / ((p)_k (1+q)_k), by the recurrence
    D_k = D_{k-1} f_k + (1-p)/((p+k-1)(k+q)), f_k = R_k/R_{k-1}: the O(q)
    difference from 1 is never formed by subtraction, and a negative p + q
    (alpha > beta) needs no logarithm."""
    s = numpy.asarray(s, dtype=float)
    t = numpy.ones_like(s)
    D = 0.0
    out = numpy.zeros_like(s)
    k = 0
    while True:
        k += 1
        t = t * (p + k - 1.0) / k * s
        D = D * (1.0 + q / (p + k - 1.0)) / (1.0 + q / k) + (1.0 - p) / (
            (p + k - 1.0) * (k + q)
        )
        term = t * D
        out = out + term
        if (
            k > 5
            and (p + k) / (k + 1.0) * numpy.amax(s) < 1.0
            and numpy.all(numpy.fabs(term) <= 1e-17 * numpy.fabs(out))
        ):
            return (1.0 - s) ** p * out


def incomplete_beta(p, q, z, s):
    """B_z(p, q) for p > 0, q > -1, 0 <= z < 1, given s = 1 - z (exact).

    Split at the integrand's mass centre c = (p+1)/(p+q+2) (at most 0.9):
    below it z^p s^q / p 2F1(1, p+q; p+1; z) (positive terms); above it
    B_c(p, q) plus the reflected int_{1-z}^{1-c} v^(q-1) (1-v)^(p-1) dv, which
    holds the integrand's mass. That reflected piece is B_{1-c}(q, p) -
    B_{1-z}(q, p); both are ~1/q, so for |q| < _IBETA_QSMALL it is summed through
    incomplete_beta_k_series instead."""
    c = incomplete_beta_split(p, q)
    if numpy.ndim(z) == 0:
        return (
            incomplete_beta_lo(p, q, z, s) if z <= c else incomplete_beta_hi(p, q, s, c)
        )
    z = numpy.asarray(z, dtype=float)
    s = numpy.asarray(s, dtype=float)
    lo = z <= c
    out = numpy.empty(z.shape)
    if numpy.any(lo):
        out[lo] = incomplete_beta_lo(p, q, z[lo], s[lo])
    if not numpy.all(lo):
        out[~lo] = incomplete_beta_hi(p, q, s[~lo], c)
    return out


def hyp2f1_1(b, c, z):
    """2F1(1, b; c; z) for b > -1, c > 0, 0 <= z < 1.

    scipy's hyp2f1 loses ~(b - c) 1e-15 for b - c > 4 (6e-13 at beta = 180);
    there (the below-c series, with z < c ~ 4/beta) it is summed directly, as
    in the C implementation."""
    if b - c <= 4.0:
        return special.hyp2f1(1.0, b, c, z)
    scalar = numpy.ndim(z) == 0  # summed in plain floats: ~20x faster
    z = float(z) if scalar else numpy.asarray(z, dtype=float)
    t, out, k = 1.0, 1.0, 0
    while True:
        t = t * (b + k) / (c + k) * z
        out = out + t
        if k > 5 and (
            abs(t) <= 1e-17 * abs(out)
            if scalar
            else numpy.all(numpy.fabs(t) <= 1e-17 * numpy.fabs(out))
        ):
            return out
        k += 1


def incomplete_beta_lo(p, q, z, s):
    return z**p * s**q / p * hyp2f1_1(p + q, p + 1.0, z)


@functools.lru_cache(maxsize=256)  # alpha, beta vary freely (fits): bounded
def incomplete_beta_at_split(p, q, c):
    """B_c(p, q) at the split c"""
    return float(incomplete_beta_lo(p, q, c, 1.0 - c))


def incomplete_beta_hi(p, q, s1, c):
    s2 = 1.0 - c
    ibc = incomplete_beta_at_split(p, q, c)
    if abs(q) < _IBETA_QSMALL:
        return _incomplete_beta_reflected_smallq(p, q, s1, s2, ibc)
    if abs(q + 1.0) < _IBETA_QSMALL:
        # beta -> 2: B(v) below has a 1/(q+1) that cancels in B(s2) - B(s1);
        # integrate by parts to the reflected integral with q -> q + 1 instead
        def B0(v):
            return v**q * (1.0 - v) ** p / q

        return (
            ibc
            + B0(s2)
            - B0(s1)
            + (p + q) / q * _incomplete_beta_reflected_smallq(p, q + 1.0, s1, s2, 0.0)
        )

    def B(v):
        return v**q * (1.0 - v) ** p / q * special.hyp2f1(1.0, p + q, q + 1.0, v)

    return ibc + B(s2) - B(s1)


def _incomplete_beta_reflected_smallq(p, q, s1, s2, base):
    """base + int_{s1}^{s2} v^(q-1) (1-v)^(p-1) dv for |q| < _IBETA_QSMALL"""
    K2 = incomplete_beta_k_series(p, q, s2)
    lg = numpy.log(s2 / s1)
    if q == 0.0:
        first = lg
    else:
        # (s2^q - s1^q)/q; the expm1 form only where it cannot overflow
        qlg = q * lg
        first = numpy.where(
            numpy.fabs(qlg) < 1.0,
            s1**q * numpy.expm1(numpy.where(numpy.fabs(qlg) < 1.0, qlg, 0.0)) / q,
            (s2**q - s1**q) / q,
        )
    return (
        base
        + first * (1.0 + q * K2)
        + s1**q * (K2 - incomplete_beta_k_series(p, q, s1))
    )
