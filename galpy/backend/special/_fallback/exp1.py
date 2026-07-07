###############################################################################
#   Fallback for the exponential integral E_1(x) on real x > 0. Needed on
#   torch (torch.special has neither exp1 nor expi). jax uses its native
#   -expi(-x) instead (the router does not reach this fallback for jax); numpy
#   uses scipy.special.exp1.
#
#   Two regimes, each AD-friendly (differentiable) and ~1e-14 vs scipy:
#     - x <= 1: the ascending power series
#         E_1(x) = -gamma - ln(x) - sum_{n>=1} (-x)^n / (n * n!)
#       which converges rapidly and without cancellation for small x;
#     - x  > 1: the Lentz continued fraction (Numerical Recipes ``expint``, n=1)
#         E_1(x) = e^{-x} / (x + 1 - 1^2/(x + 3 - 2^2/(x + 5 - ...)))
#       evaluated by the modified-Lentz recurrence, which converges geometrically
#       for x >~ 1 (faster for larger x).
#   Each branch's argument is clamped into its own valid region wherever the
#   OTHER branch is selected, so the unused branch can neither overflow (the
#   series' huge alternating terms at large x) nor NaN-poison reverse-mode
#   gradients.
###############################################################################
import numpy

from ..._namespaces import _backend_dtype

_GAMMA = 0.5772156649015328606  # Euler-Mascheroni
_SPLIT = 1.0  # series for x <= _SPLIT, continued fraction above
_N_SERIES = 25  # ascending-series terms (x <= 1)
_N_CF = 80  # modified-Lentz iterations (x > 1)


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

    # --- ascending power series (x <= 1) ---
    s = xp.zeros_like(xs)
    term = -xs  # (-x)^1 / 1!
    for n in range(1, _N_SERIES + 1):
        s = s + term / n
        term = term * (-xs) / (n + 1)  # (-x)^{n+1} / (n+1)!
    series = -_GAMMA - xp.log(xs) - s

    # --- modified-Lentz continued fraction (x > 1) ---
    # h converges to 1/(x+1 - 1^2/(x+3 - 2^2/(x+5 - ...))); c is seeded to a
    # large value (the standard tiny-FPMIN reciprocal) that washes out after the
    # first iteration. All denominators stay >~ 2 for xt >= 1, so no guard beyond
    # the branch clamp is needed.
    b = xt + 1.0
    c = xp.full_like(xt, 1.0e300)
    d = 1.0 / b
    h = d
    for i in range(1, _N_CF + 1):
        an = -(i * i)
        b = b + 2.0
        d = 1.0 / (an * d + b)
        c = b + an / c
        h = h * (c * d)
    cf = h * xp.exp(-xt)

    return xp.where(big, xp.zeros_like(x), xp.where(inside, series, cf))
