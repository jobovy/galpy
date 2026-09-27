###############################################################################
#   galpy.potential._smallr: helpers for closed-form radial quantities at the
#   radial edges (r = 0, r = inf) and at small r, where the closed forms
#   cancel.
###############################################################################
import numpy


def radial_limits(r, fn, at0=None, atinf=None):
    """``fn(r)``, with ``r == 0`` / ``r == inf`` replaced by known limits.

    For closed forms that are NaN at an edge (0 * inf, inf - inf, 0/0). Only
    the edge entries change; with no edge present this is ``fn(r)`` itself.
    ``at0`` / ``atinf``: the limit, or None to leave that edge alone.
    """
    ra = numpy.asarray(r, dtype=float)
    edges = [
        (m, v)
        for m, v in ((ra == 0.0, at0), (numpy.isinf(ra), atinf))
        if v is not None and numpy.any(m)
    ]
    if not edges:
        return fn(r)
    bad = numpy.zeros(ra.shape, dtype=bool)
    for m, _ in edges:
        bad |= m
    out = numpy.asarray(fn(numpy.where(bad, 1.0, ra)), dtype=float)
    for m, v in edges:
        out = numpy.where(m, v, out)
    return out[()]


def small_r_select(r, rmax, small_fn, generic_fn, rsafe):
    """``small_fn(r)`` where ``r < rmax``, else ``generic_fn(r)``.

    Each branch only runs if some entry needs it, so a call without small
    radii is exactly the generic formula. ``small_fn`` is evaluated at
    ``rsafe`` (< rmax) on the other entries, keeping it finite there.
    """
    ra = numpy.asarray(r, dtype=float)
    small = ra < rmax
    if not numpy.any(small):
        return generic_fn(r)
    if numpy.all(small):
        return small_fn(r)
    out = numpy.where(small, small_fn(numpy.where(small, ra, rsafe)), generic_fn(ra))
    return out[()]


def power_series(x, coeffs, first):
    """``sum_i coeffs[i] x^(first + i)``, by Horner."""
    out = coeffs[-1]
    for c in coeffs[-2::-1]:
        out = out * x + c
    return out * x**first
