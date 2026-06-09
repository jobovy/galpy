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

_GAMMA = 0.5772156649015328606  # Euler-Mascheroni
_NSERIES = 30  # ascending-series terms (x <= 2)
_TRAP_H = 0.25  # trapezoidal step (in the 1/sqrt(x)-scaled variable)
_TRAP_N = 64  # trapezoidal nodes
# node positions i*h and weights (h/2 at the endpoint i=0), as numpy constants
_TRAP_NODES = numpy.arange(_TRAP_N + 1) * _TRAP_H
_TRAP_W = numpy.full(_TRAP_N + 1, _TRAP_H)
_TRAP_W[0] = _TRAP_H / 2.0


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
    K0s = -(xp.log(xs / 2.0) + _GAMMA) * i0(xs)
    term = xp.ones_like(xs)
    harm = 0.0
    for k in range(1, _NSERIES):
        harm += 1.0 / k
        term = term * x2 / (k * k)
        K0s = K0s + term * harm
    s1 = xp.zeros_like(xs)
    term = xp.ones_like(xs)
    hk = 0.0
    for k in range(0, _NSERIES):
        hk1 = hk + 1.0 / (k + 1)
        s1 = s1 + term * ((hk + hk1) / 2.0 - _GAMMA)
        term = term * x2 / ((k + 1) * (k + 2))
        hk = hk1
    K1s = 1.0 / xs + xp.log(xs / 2.0) * i1(xs) - (xs / 2.0) * s1

    # --- peak-resolving scaled trapezoidal (x > 2) ---
    nodes = xp.asarray(_TRAP_NODES)
    weights = xp.asarray(_TRAP_W)
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
