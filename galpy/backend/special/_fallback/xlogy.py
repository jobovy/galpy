###############################################################################
#   Fallback for xlogy(x, y) = x * log(y), with the scipy convention that
#   x == 0 yields 0 even when y == 0 (so 0 * log(0) = 0). Native on every
#   backend, so the router never reaches this in practice; it is a valid,
#   AD-friendly pure-backend implementation kept (and unit-tested) so the
#   capability story is uniform with the other special functions.
###############################################################################


def xlogy_fallback(xp, x, y):
    x = xp.asarray(x) * 1.0
    y = xp.asarray(y) * 1.0
    # guard the dead branch's log(0) so it cannot NaN-poison gradients at x==0
    safe_y = xp.where(x == 0, xp.ones_like(y), y)
    return xp.where(x == 0, xp.zeros_like(x), x * xp.log(safe_y))
