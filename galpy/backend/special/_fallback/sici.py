###############################################################################
#   Fallback for the sine and cosine integrals Si(x), Ci(x) on real x > 0.
#   Needed on torch (torch.special has no sici); jax has a native sici and numpy
#   uses scipy, so the router only routes torch here.
#
#   ``sici_fallback`` uses two regimes, each ~1e-14 vs scipy and AD-friendly:
#     - x <= x0: the convergent power series
#         Si(x) = sum_{k>=0} a_k/(2k+1),  a_0 = x, a_k = a_{k-1}*(-x^2/((2k)(2k+1)))
#         Ci(x) = gamma + ln(x) + sum_{k>=1} b_k/(2k),
#                 b_0 = 1, b_k = b_{k-1}*(-x^2/((2k-1)(2k))).
#       Terms are built iteratively (no factorials -> no overflow).
#     - x  > x0: the auxiliary functions f, g via Gauss-Laguerre quadrature of
#         f(x) = int_0^inf e^{-x t}/(1+t^2) dt,
#         g(x) = int_0^inf t e^{-x t}/(1+t^2) dt,
#       with the substitution t = u/x (u the Laguerre variable for the weight
#       e^{-u} on [0, inf)):
#         f(x) = (1/x)  * sum_i w_i / (1 + (u_i/x)^2),
#         g(x) = (1/x^2)* sum_i w_i u_i / (1 + (u_i/x)^2),
#       then  Si(x) = pi/2 - f*cos(x) - g*sin(x),  Ci(x) = f*sin(x) - g*cos(x).
#       The integrands are smooth and decay like e^{-x t}, so a moderate-order
#       Gauss-Laguerre rule is accurate for all x > x0.
#   Each branch's argument is clamped into its valid region wherever the OTHER
#   branch is selected, so the unused branch cannot overflow (1/x at x->0 in the
#   asymptotic branch, large x^2 in the series branch) or NaN-poison reverse-mode
#   gradients.
###############################################################################
import numpy

from ..._namespaces import asarray_on_device, device_of, under_trace

_GAMMA = 0.5772156649015328606  # Euler-Mascheroni
_X0 = 6.0  # regime split
_NSERIES = 40  # power-series terms (x <= x0); ~1e-14 at x0
_NLAG = 64  # Gauss-Laguerre order (x > x0)
# Gauss-Laguerre nodes/weights for int_0^inf e^{-u} h(u) du, kept float64.
_LAG_U, _LAG_W = numpy.polynomial.laguerre.laggauss(_NLAG)
# Series tables. Both series are a running product of per-step RATIOS, so a
# cumulative product reproduces the recurrence in ONE pass instead of _NSERIES
# eager ops (the same rewrite as exp1/bessel_k). k runs 1.._NSERIES-1.
_K = numpy.arange(1, _NSERIES).astype(float)
_SI_DEN = (2.0 * _K) * (2.0 * _K + 1.0)  # a_k ratio denominators
_SI_COEFF = 1.0 / (2.0 * _K + 1.0)  # the 1/(2k+1) weight on a_k
_CI_DEN = (2.0 * _K - 1.0) * (2.0 * _K)  # b_k ratio denominators
_CI_COEFF = 1.0 / (2.0 * _K)  # the 1/(2k) weight on b_k

_TABLE_CACHE = {}


def _tables(xp, dev):
    """The series/quadrature tables as backend arrays, once per (namespace, device)."""
    key = (id(xp), repr(dev))
    got = _TABLE_CACHE.get(key)
    if got is None:
        got = tuple(
            asarray_on_device(xp, t, dev)
            for t in (_SI_DEN, _SI_COEFF, _CI_DEN, _CI_COEFF, _LAG_U, _LAG_W)
        )
        # A conversion made inside a trace is a TRACER, and caching one leaks it
        # out of its trace (gh#1464). `under_trace`, not `under_jax_trace`:
        # sici_fallback is reached ONLY on torch (numpy uses scipy, jax has a
        # native sici), so a jax-only predicate here guards the one case that
        # cannot happen and misses the one that can -- torch.compile does not
        # raise on float(x), so only `torch.compiler.is_compiling()` sees it.
        if not under_trace(*got):
            _TABLE_CACHE[key] = got
    return got


def sici_fallback(xp, x):
    """Return (Si(x), Ci(x)) for real x > 0, ~1e-14 vs scipy, AD-friendly."""
    x = xp.asarray(x) * 1.0
    small = x <= _X0
    # Clamp the dead region of each branch into its valid domain.
    xs = xp.where(small, x, xp.ones_like(x))  # series branch (x <= x0)
    xa = xp.where(small, _X0 * xp.ones_like(x), x)  # asymptotic branch (x > x0)

    dev = device_of(x)
    si_den, si_coeff, ci_den, ci_coeff, u, w = _tables(xp, dev)

    # --- convergent power series (x <= x0), one cumulative product each ---
    # Si: a_0 = x, a_k = a_{k-1} * (-x^2/((2k)(2k+1))), summed with weight
    # 1/(2k+1); Ci: b_0 = 1, b_k = b_{k-1} * (-x^2/((2k-1)(2k))), weight 1/(2k).
    # `flip` sums the smallest terms first, which the sequential loop this
    # replaces could not do: Ci's worst relative error over x in (0, x0] drops
    # 4.5e-13 -> 5.6e-14 (the near-zero of Ci at x ~ 0.616 sets it).
    nx2 = -(xs * xs)
    a = xp.cumulative_prod(nx2[..., None] / si_den, axis=-1)  # a_k / x
    Sis = xs * (1.0 + xp.sum(xp.flip(a * si_coeff, axis=-1), axis=-1))
    b = xp.cumulative_prod(nx2[..., None] / ci_den, axis=-1)  # b_k
    Cis = _GAMMA + xp.log(xs) + xp.sum(xp.flip(b * ci_coeff, axis=-1), axis=-1)

    # --- Gauss-Laguerre auxiliary functions (x > x0) ---
    tau = u / xa[..., None]  # u_i / x, shape (..., N)
    denom = 1.0 + tau * tau
    f = xp.sum(w / denom, axis=-1) / xa
    g = xp.sum(w * tau / denom, axis=-1) / xa
    cosx = xp.cos(xa)
    sinx = xp.sin(xa)
    half_pi = numpy.pi / 2.0
    Sia = half_pi - f * cosx - g * sinx
    Cia = f * sinx - g * cosx

    return xp.where(small, Sis, Sia), xp.where(small, Cis, Cia)
