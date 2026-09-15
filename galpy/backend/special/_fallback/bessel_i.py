###############################################################################
#   Fallback for the modified Bessel function of the first kind of integer
#   order, I_n(x), on real x. Needed on BOTH jax and torch: jax.scipy.special
#   and torch.special expose only i0/i1, no general-order iv.
#
#   n=0,1 use the native i0/i1 directly. For n>=2 (galpy uses n=2 only,
#   RazorThinExponentialDiskPotential._R2deriv) two regimes keep it accurate and
#   AD-safe for ALL x, including x=0:
#     - |x| <= 2: the ascending series I_n(x) = sum_k (x/2)^(2k+n)/(k!(n+k)!)
#       -- no 1/x, exact 0 at x=0, machine-accurate by x=2.
#     - |x| >  2: the upward recurrence I_{m+1} = I_{m-1} - (2m/x) I_m from
#       native i0, i1 (the naive recurrence catastrophically cancels as x->0 and
#       divides by 0 at x=0, so it is confined to |x|>2).
#   Each branch's dead side is clamped into its safe region (series operand -> 1,
#   recurrence operand -> 2) so the eager xp.where evaluates neither to inf/NaN
#   and reverse-mode gradients stay finite. numpy keeps the exact scipy.special.iv.
###############################################################################
import math

_NSERIES = 25  # ascending-series terms; machine precision for |x| <= 2


def iv_fallback(xp, n, x):
    """Integer-order I_n(x) via series (|x|<=2) and upward recurrence (|x|>2)."""
    from .._router import i0, i1

    n = int(n)
    x = xp.asarray(x) * 1.0
    if n == 0:
        return i0(x)
    if n == 1:
        return i1(x)
    small = xp.abs(x) <= 2.0
    xs = xp.where(small, x, xp.ones_like(x))  # series branch (dead side -> 1)
    xl = xp.where(small, 2.0 * xp.ones_like(x), x)  # recurrence branch (dead -> 2)
    # ascending series on xs
    half2 = 0.25 * xs * xs
    term = (0.5 * xs) ** n / math.factorial(n)  # k=0
    In = term
    for k in range(1, _NSERIES):
        term = term * half2 / (k * (k + n))
        In = In + term
    # upward recurrence on xl from native i0, i1
    im1, i = i0(xl), i1(xl)
    for m in range(1, n):
        im1, i = i, im1 - (2.0 * m / xl) * i
    return xp.where(small, In, i)
