###############################################################################
#   Fallback for the exponential integral E_1(x) on real x > 0. Needed on
#   torch (torch.special has neither exp1 nor expi). jax uses its native
#   -expi(-x) instead (the router does not reach this fallback for jax); numpy
#   uses scipy.special.exp1.
#
#   Two regimes, each AD-friendly (differentiable) and ~1e-15 vs scipy:
#     - x <= 1: the ascending power series
#         E_1(x) = -gamma - ln(x) - sum_{n>=1} (-x)^n / (n * n!)
#       which converges rapidly and without cancellation for small x;
#     - x  > 1: Gauss-Legendre on
#         E_1(x) = e^{-x} int_0^inf e^{-s}/(x+s) ds,   s = t/(1-t), t in [0,1]
#       i.e. ONE weighted sum over fixed nodes.
#   Both regimes are VECTOR operations over a fixed node/term axis rather than
#   Python loops. That matters because this fallback is torch-only (jax uses its
#   native -expi(-x), numpy uses scipy), and torch pays ~3.8 us per eager op: the
#   Lentz continued fraction this replaces ran 80 sequential iterations of ~6 ops
#   each, ~580 ops per call, which made ExpTruncNFWPotential 11x slower on torch
#   than on numpy. It is also 8x MORE accurate than that recurrence (1.0e-15 vs
#   8.5e-15 max relative error against scipy over 1e-8 <= x <= 630).
#   Each branch's argument is clamped into its own valid region wherever the
#   OTHER branch is selected, so the unused branch can neither overflow (the
#   series' huge alternating terms at large x) nor NaN-poison reverse-mode
#   gradients.
###############################################################################
import numpy

from ..._namespaces import _backend_dtype

_GAMMA = 0.5772156649015328606  # Euler-Mascheroni
_SPLIT = 1.0  # series for x <= _SPLIT, quadrature above
_N_SERIES = 25  # ascending-series terms (x <= 1)
_N_GL = 80  # Gauss-Legendre nodes for the x > 1 quadrature

# Series tables: term_n = (-x)^n/n!, and the sum is sum_n term_n/n. Keeping the
# per-step RATIOS lets a cumulative product reproduce the recurrence in one pass.
_S_DEN = numpy.arange(1, _N_SERIES + 1).astype(float)  # (n+1) ratio denominators
_S_COEFF = 1.0 / _S_DEN  # the 1/n weight on term_n

# Quadrature tables for E_1(x) = e^{-x} int_0^inf e^{-s}/(x+s) ds with
# s = t/(1-t) mapping [0,1) -> [0,inf). Gauss-LEGENDRE on the mapped interval,
# not Gauss-Laguerre on the raw one: Laguerre's weights span such a range that
# roundoff floors it at ~1.3e-14 no matter how many nodes it gets, while this
# reaches 1.0e-15 at 80 nodes. Everything except 1/(x+s) is x-independent, so it
# folds into the weight once, here.
_GL_T, _GL_W_RAW = numpy.polynomial.legendre.leggauss(_N_GL)
_GL_T = 0.5 * (_GL_T + 1.0)
_GL_S = _GL_T / (1.0 - _GL_T)  # the s nodes
_GL_W = 0.5 * _GL_W_RAW * numpy.exp(-_GL_S) / (1.0 - _GL_T) ** 2  # e^{-s} ds weight

_TABLE_CACHE = {}


def _tables(xp, dev):
    """The node/weight tables as backend arrays, once per (namespace, device)."""
    from ..._namespaces import asarray_on_device, under_jax_trace

    key = (id(xp), repr(dev))
    got = _TABLE_CACHE.get(key)
    if got is None:
        got = tuple(
            asarray_on_device(xp, t, dev) for t in (_S_DEN, _S_COEFF, _GL_S, _GL_W)
        )
        # Under a jax trace asarray(..., device=) lowers to device_put, so these
        # come back as TRACERS; caching one leaks it out of its trace. Constants
        # are free inside a trace anyway (see bessel_k for the same guard).
        if not under_jax_trace(*got):
            _TABLE_CACHE[key] = got
    return got


def exp1_fallback(xp, x):
    """Exponential integral E_1(x) for real x > 0, ~1e-14 vs scipy, AD-friendly."""
    # Compute in float64 (precision is the point; the router casts back to the
    # input dtype). Explicit astype -- not a float64 scalar multiply, which torch
    # leaves float32 -- also keeps the large Lentz seed below the float32 overflow
    # (torch defaults to float32).
    x = xp.astype(xp.asarray(x), _backend_dtype(xp, numpy.float64))
    inside = x <= _SPLIT
    big = xp.isinf(x)  # E_1(inf) = 0 (r=inf appears in potential-at-infinity/mass)
    # Clamp the dead region of each branch into its valid domain so neither
    # overflows (nor lets inf*0 in the CF give NaN) nor produces a NaN gradient
    # through the masked-out branch.
    xs = xp.where(inside, x, xp.ones_like(x))  # series branch (x <= 1)
    xt = xp.where(inside | big, xp.ones_like(x), x)  # CF branch (1 < x < inf)

    from ..._namespaces import device_of

    s_den, s_coeff, gl_s, gl_w = _tables(xp, device_of(x))

    # --- ascending power series (x <= 1), one cumulative product ---
    # term_n = (-x)^n/n! is the running product of (-x)/n; the sum weights it by
    # 1/n. Same recurrence as the Python loop this replaces, one pass instead of
    # _N_SERIES eager ops.
    terms = xp.cumulative_prod(-xs[..., None] / s_den, axis=-1)
    series = -_GAMMA - xp.log(xs) - xp.sum(terms * s_coeff, axis=-1)

    # --- Gauss-Legendre quadrature (x > 1) ---
    # e^{-x} sum_i w_i/(x+s_i), with w_i already carrying e^{-s_i} and the
    # s = t/(1-t) Jacobian. One weighted sum, no recurrence.
    quad = xp.exp(-xt) * xp.sum(gl_w / (xt[..., None] + gl_s), axis=-1)

    return xp.where(big, xp.zeros_like(x), xp.where(inside, series, quad))
