###############################################################################
#   Fallback for the modified Bessel function of the first kind of general
#   integer order, I_n(x), on real x > 0. Needed on BOTH jax and torch:
#   jax.scipy.special and torch.special expose only i0/i1, no general-order iv.
#
#   I_n is built from the native i0, i1 (Tier 1, present on every backend) via
#   the upward recurrence I_{m+1}(x) = I_{m-1}(x) - (2m/x) I_m(x). Only small
#   orders are needed in galpy (n=2, RazorThinExponentialDiskPotential._R2deriv),
#   for which the upward recurrence from accurate I_0, I_1 is fine; numpy keeps
#   the exact scipy.special.iv (this fallback is the jax/torch path only).
###############################################################################


def iv_fallback(xp, n, x):
    """Integer-order I_n(x) via the upward recurrence from native I_0, I_1."""
    from .._router import i0, i1

    n = int(n)
    # Coerce x onto the active backend first so the native i0/i1 (and the
    # recurrence arithmetic) see a backend array, not a raw numpy/scalar.
    x = xp.asarray(x) * 1.0
    im1 = i0(x)  # I_0
    if n == 0:
        return im1
    i = i1(x)  # I_1
    if n == 1:
        return i
    for m in range(1, n):
        im1, i = i, im1 - (2.0 * m / x) * i
    return i
