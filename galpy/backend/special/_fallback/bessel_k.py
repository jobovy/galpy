###############################################################################
#   Fallbacks for the modified Bessel functions of the second kind K0, K1, Kn
#   on real x > 0. Needed on BOTH jax and torch:
#     - jax.scipy.special has no k0/k1/kn at all;
#     - torch.special has modified_bessel_k0/k1 but they are NOT differentiable
#       (no autograd backward) and lack kn entirely, so we use the fallback there
#       too (the router sees no torch.special.k0 attribute -> treats it missing).
#
#   K0, K1 (``_k01``) use two regimes, each ~1e-15 vs scipy and AD-friendly:
#     - x <= 2: the Abramowitz & Stegun ascending series (9.6.13/9.6.11), built
#       on the native i0/i1 (Tier 1) plus elementary terms;
#     - x  > 2: the trapezoidal rule on K_nu(x) = int_0^inf e^{-x cosh t}
#       cosh(nu t) dt. The integrand is double-exponentially decaying, so the
#       trapezoidal rule converges geometrically; its e^{-x(cosh t-1)} peak has
#       width ~1/sqrt(x), so the nodes are scaled by 1/sqrt(x) to resolve it
#       uniformly for all large x.
#   Each branch's argument is clamped into its valid region wherever the OTHER
#   branch is selected, so the unused branch cannot overflow (i0 at large x) or
#   NaN-poison reverse-mode gradients.
#
#   Kn (``kn_fallback``) uses the upward recurrence K_{m+1}=K_{m-1}+(2m/x)K_m
#   from K0, K1 -- the stable direction for K.
###############################################################################
import numpy

from ..._namespaces import asarray_on_device, device_of

_GAMMA = 0.5772156649015328606  # Euler-Mascheroni
_NSERIES = 30  # ascending-series terms (x <= 2)
_TRAP_H = 0.25  # trapezoidal step (in the 1/sqrt(x)-scaled variable)
_TRAP_N = 64  # trapezoidal nodes
# node positions i*h and weights (h/2 at the endpoint i=0), as numpy constants
_TRAP_NODES = numpy.arange(_TRAP_N + 1) * _TRAP_H
_TRAP_W = numpy.full(_TRAP_N + 1, _TRAP_H)
_TRAP_W[0] = _TRAP_H / 2.0

# Ascending-series tables. The two series are sums of coeff_k * t_k, where t_k is
# a running product; keeping the per-step RATIOS lets a cumulative product
# reproduce the recurrence while evaluating all _NSERIES terms in one vectorized
# pass instead of one eager op per term. That mattered: the K0/K1 series were the
# single largest cost in a jax RazorThinExponentialDiskPotential orbit -- 2218
# calls x ~270 scalar ops each.
_H = numpy.concatenate(([0.0], numpy.cumsum(1.0 / numpy.arange(1, _NSERIES + 1))))
# K0: t_k = prod_{j=1..k} 1/j^2, coeff_k = H_k, for k = 1.._NSERIES-1
_K0_RATIO_DEN = (numpy.arange(1, _NSERIES) ** 2).astype(float)
_K0_COEFF = _H[1:_NSERIES]
# K1: t_0 = 1 and t_k = t_{k-1} / (k(k+1)); coeff_k = (H_k + H_{k+1})/2 - gamma
_K1_RATIO_DEN = numpy.concatenate(
    ([1.0], (numpy.arange(1, _NSERIES) * numpy.arange(2, _NSERIES + 1)).astype(float))
)
_K1_COEFF = (_H[0:_NSERIES] + _H[1 : _NSERIES + 1]) / 2.0 - _GAMMA
_K1_NUM_POW = numpy.concatenate(([0.0], numpy.ones(_NSERIES - 1)))  # x2^0 then x2^k


_SERIES_CACHE = {}


def _series_tables(xp, dev):
    """Series tables as backend arrays, materialized once per (namespace, device).

    Rebuilding them on every call put five host->device conversions in the hot
    path, and under jax each fresh array is another primitive for the eager
    compiler to compile.
    """
    key = (id(xp), repr(dev))
    got = _SERIES_CACHE.get(key)
    if got is None:
        got = tuple(
            asarray_on_device(xp, t, dev)
            for t in (_K0_RATIO_DEN, _K0_COEFF, _K1_RATIO_DEN, _K1_COEFF, _K1_NUM_POW)
        )
        _SERIES_CACHE[key] = got
    return got


def _k01(xp, x):
    """Return (K0(x), K1(x)) for real x > 0, ~1e-15 vs scipy, AD-friendly."""
    x = xp.asarray(x) * 1.0
    inside = x <= 2.0
    # Clamp the dead region of each branch into its valid domain.
    xs = xp.where(inside, x, xp.ones_like(x))  # series branch (x<=2)
    xt = xp.where(inside, 2.0 * xp.ones_like(x), x)  # trapezoid branch (x>2)

    # --- ascending series (x <= 2), via native i0/i1 ---
    from .._router import i0, i1

    x2 = xs * xs / 4.0
    dev0 = device_of(x)
    tabs = _series_tables(xp, dev0)
    # Both series run as ONE cumulative product over the term axis rather than a
    # Python loop of _NSERIES eager ops. `cumprod` of the per-step ratios is the
    # same recurrence the loop ran; only the reduction is vectorized.
    k0_den, k0_coeff, k1_den, k1_coeff, k1_pow = tabs
    k0_terms = xp.cumulative_prod(x2[..., None] / k0_den, axis=-1)
    K0s = -(xp.log(xs / 2.0) + _GAMMA) * i0(xs) + xp.sum(k0_terms * k0_coeff, axis=-1)

    # first ratio is 1 (t_0 = 1), the rest are x2/(k(k+1))
    k1_terms = xp.cumulative_prod((x2[..., None] ** k1_pow) / k1_den, axis=-1)
    s1 = xp.sum(k1_terms * k1_coeff, axis=-1)
    K1s = 1.0 / xs + xp.log(xs / 2.0) * i1(xs) - (xs / 2.0) * s1

    # --- peak-resolving scaled trapezoidal (x > 2) ---
    # node/weight tables stay float64 (precision is the point; the router
    # exit-casts) but must live on the input's device (CUDA support)
    dev = device_of(x)
    nodes = asarray_on_device(xp, _TRAP_NODES, dev)
    weights = asarray_on_device(xp, _TRAP_W, dev)
    sc = 1.0 / xp.sqrt(xt)
    t = sc[..., None] * nodes  # (..., N+1)
    cosh_t = xp.cosh(t)
    e = xp.exp(-xt[..., None] * cosh_t) * weights
    K0t = xp.sum(e, axis=-1) * sc
    K1t = xp.sum(e * cosh_t, axis=-1) * sc

    return xp.where(inside, K0s, K0t), xp.where(inside, K1s, K1t)


def k0_fallback(xp, x):
    """Modified Bessel function of the second kind, order 0."""
    return _k01(xp, x)[0]


def k1_fallback(xp, x):
    """Modified Bessel function of the second kind, order 1."""
    return _k01(xp, x)[1]


def kn_fallback(xp, n, x):
    """Integer-order K_n(x) via the stable upward recurrence from K0, K1."""
    n = int(n)
    km1, k = _k01(xp, x)  # K0, K1
    if n == 0:
        return km1
    if n == 1:
        return k
    x = xp.asarray(x) * 1.0
    for m in range(1, n):
        km1, k = k, km1 + (2.0 * m / x) * k
    return k
