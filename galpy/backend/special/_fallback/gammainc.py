###############################################################################
#   Fallbacks for the regularized incomplete gamma functions P(a,x) and
#   Q(a,x) = 1 - P(a,x). Needed on torch, which has ``torch.special.gammainc``
#   but (a) no derivative w.r.t. the ORDER a -- ``NotImplementedError: the
#   derivative for 'igamma: input' is not implemented`` -- and (b) a discrete
#   algorithm switch at a ~ 20 that costs ~6 digits above it (|dQ| ~ 5e-10 for
#   a >= 21 vs ~1e-16 below). Both are cured at once by evaluating in backend
#   ops: autodiff then supplies d/da and d/dx alike. jax's native versions are
#   accurate and differentiable in both arguments, so the router does not reach
#   this fallback for jax; numpy uses scipy.
#
#   Standard split, each branch converging where it is used:
#     - x <  a+1: the ascending series  P = prefix * sum_n x^n / (a...(a+n))
#     - x >= a+1: the modified-Lentz continued fraction for Q.
#
#   The delicate part is the shared prefix  x^a e^-x / Gamma(a),  NOT the
#   series. Writing it as exp(-x + a ln x - lnGamma(a)) sums three terms that
#   are each O(a ln a) and cancel to O(1) -- at a = x = 80 that is
#   -80.0000 + 350.5621 - 269.2911 = 1.2710. Each term carries |term|*eps, so
#   exp() returns ~1e-13 relative error however many series terms are taken;
#   more iterations cannot fix it. Instead use the Stirling form, in which the
#   large parts cancel ANALYTICALLY:
#
#       x^a e^-x / Gamma(a) = e^{-a*(lam - 1 - ln lam)} * sqrt(a/2pi) * e^{-w(a)}
#
#   with lam = x/a and w the Stirling remainder. Now the exponent is O(1) by
#   construction. That form needs a >~ 14 for w's asymptotic series; below it
#   the naive exponent is itself accurate (lnGamma(a) is O(1) there, so there is
#   no large cancellation to lose), so the two are used exactly where the other
#   fails. The crossover was measured, not guessed: worst error over an
#   a-in-[0.5,200] grid is 5.4e-07 at a split of 2, 7.3e-15 at 14, 1.1e-14 at 20.
#
#   Verified against mpmath at 60 digits over a in [0.5,200], x/a in [0.3,4]:
#   worst 7.3e-15 (P) and 2.4e-14 (Q), against scipy's own 1.0e-13 and 1.1e-13
#   on the same grid -- i.e. at least as accurate as the numpy path everywhere,
#   and up to 20x better in the large-a mid-domain.
#
#   Iteration counts are the measured minima at that accuracy (N_SERIES=120:
#   80 gives only 7.6e-08; N_CF=50: 30 gives 2.7e-11, and MORE than 50 is
#   slightly worse from accumulation).
###############################################################################
import numpy

from ..._namespaces import _backend_dtype
from .._router import gammaln

_A_STIRLING = 14.0  # Stirling prefix at/above this order, naive exponent below
_N_SERIES = 120  # ascending-series terms (x < a+1)
_N_CF = 50  # modified-Lentz iterations (x >= a+1)
_FPMIN = 1e-300  # Lentz zero-denominator floor
_TINY_U = 1e-4  # |lam-1| below which u - log1p(u) cancels; use its series


def _u_minus_log1p(xp, u):
    """``u - log1p(u)`` = lam - 1 - ln(lam), accurately for all u > -1.

    The direct form loses only ~log10(2/|u|) digits (about 1 at |u| = 0.1), so
    it is used everywhere except |u| tiny, where the cancellation is total and
    the Maclaurin series u^2/2 - u^3/3 + ... takes over. Getting this threshold
    backwards (series for |u| < 0.25) costs 3.4e-07 at a=40, x=35: the first
    omitted series term is u^8/8.
    """
    tiny = xp.abs(u) < _TINY_U
    ut = xp.where(tiny, u, 0.0)  # dead branch -> 0, series is then exact
    series = ut * ut * (1.0 / 2 - ut * (1.0 / 3 - ut * (1.0 / 4 - ut / 5)))
    ud = xp.where(tiny, 1.0, u)  # dead branch -> 1, keeps log1p off its pole
    return xp.where(tiny, series, ud - xp.log1p(ud))


def _stirling_remainder(xp, a):
    """lnGamma(a) - [(a-1/2)ln a - a + ln(2pi)/2], via its asymptotic series."""
    ia = 1.0 / a
    ia2 = ia * ia
    return ia * (
        1.0 / 12
        - ia2 * (1.0 / 360 - ia2 * (1.0 / 1260 - ia2 * (1.0 / 1680 - ia2 / 1188)))
    )


def _prefix(xp, a, x):
    """x^a e^-x / Gamma(a), without the O(a ln a) cancellation. See module docs."""
    big = a >= _A_STIRLING
    ab = xp.where(big, a, _A_STIRLING)  # dead branch -> a valid Stirling order
    lam = x / ab
    stirling = (
        xp.exp(-ab * _u_minus_log1p(xp, lam - 1.0))
        * xp.sqrt(ab / (2.0 * numpy.pi))
        * xp.exp(-_stirling_remainder(xp, ab))
    )
    asm = xp.where(big, 1.0, a)  # dead branch -> a=1, gammaln(1)=0, no overflow
    naive = xp.exp(-x + asm * xp.log(x) - gammaln(asm))
    return xp.where(big, stirling, naive)


def _series_P(xp, a, x, pref):
    """Ascending series for P(a,x), valid (and used) for x < a+1."""
    ap = a
    term = 1.0 / a
    total = term
    for _ in range(_N_SERIES):
        ap = ap + 1.0
        term = term * (x / ap)
        total = total + term
    return total * pref


def _cf_Q(xp, a, x, pref):
    """Modified-Lentz continued fraction for Q(a,x), valid for x >= a+1."""
    b = x + 1.0 - a
    c = xp.full_like(b, 1.0 / _FPMIN)
    d = 1.0 / xp.where(xp.abs(b) < _FPMIN, _FPMIN, b)
    h = d
    for i in range(1, _N_CF + 1):
        an = -i * (i - a)
        b = b + 2.0
        d = an * d + b
        d = xp.where(xp.abs(d) < _FPMIN, _FPMIN, d)
        c = b + an / xp.where(xp.abs(c) < _FPMIN, _FPMIN, c)
        d = 1.0 / d
        h = h * (d * c)
    return pref * h


def _both(xp, a, x):
    """Return (P, Q), each computed by whichever branch is valid there."""
    f64 = _backend_dtype(xp, numpy.float64)
    a = xp.astype(xp.asarray(a), f64)
    x = xp.astype(xp.asarray(x), f64)
    a, x = xp.broadcast_arrays(a, x)
    use_series = x < a + 1.0
    # Clamp each branch's argument into its own convergent region wherever the
    # OTHER branch is selected: the series diverges for x >> a and the CF's
    # b = x+1-a passes through zero for x < a, so an unclamped dead branch
    # would overflow or NaN-poison the reverse-mode gradient.
    x_ser = xp.where(use_series, x, a)  # x=a is inside the series' region
    x_cf = xp.where(use_series, a + 1.0, x)  # a+1 is the CF's boundary
    p = _series_P(xp, a, x_ser, _prefix(xp, a, x_ser))
    q = _cf_Q(xp, a, x_cf, _prefix(xp, a, x_cf))
    return (
        xp.where(use_series, p, 1.0 - q),
        xp.where(use_series, 1.0 - p, q),
    )


def gammainc_fallback(xp, a, x):
    """Regularized lower incomplete gamma P(a,x), differentiable in a AND x."""
    return _both(xp, a, x)[0]


def gammaincc_fallback(xp, a, x):
    """Regularized upper incomplete gamma Q(a,x), differentiable in a AND x."""
    return _both(xp, a, x)[1]
