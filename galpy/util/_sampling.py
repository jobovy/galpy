###############################################################################
#   galpy.util._sampling: inverse-CDF sampling of tabulated distributions
###############################################################################
import numpy


def ensure_strictly_increasing(cdf_grid, floor=1e-12):
    """Project ``cdf_grid`` onto a strictly increasing grid (steps >= ``floor``).

    A closed-form CDF sampled to the far tails saturates to float ``1.0`` (and
    can dip a rounding-ulp below 0 at the bottom), leaving zero steps that make
    the inversion knots non-strictly increasing. This floors every step to
    ``floor`` and reintegrates by cumulative sum, so the result is strictly
    increasing with the same first value; only the saturated tail steps, which
    carry negligible probability, change.
    """
    diff = cdf_grid[1:] - cdf_grid[:-1]
    steps = numpy.maximum(diff, diff * 0.0 + floor)
    return numpy.concatenate([cdf_grid[:1], cdf_grid[:1] + numpy.cumsum(steps, axis=0)])


def linear_inverse_cdf_sample(omega_grid, cdf_grid, u):
    """Sample by piecewise-linear inversion of a tabulated CDF.

    ``omega_grid`` is the sample axis (strictly increasing), ``cdf_grid =
    F(omega_grid)`` (strictly increasing in [0, 1]; see
    ``ensure_strictly_increasing``), and ``u`` uniform draws; returns
    ``F^{-1}(u)``, with ``u`` outside ``[cdf_grid[0], cdf_grid[-1]]`` clamped to
    that range.
    """
    x = cdf_grid * 1.0
    y = omega_grid * 1.0
    r = numpy.clip(u, x[0], x[-1])
    idx = numpy.clip(numpy.searchsorted(x, r, side="right") - 1, 0, x.shape[0] - 2)
    x0 = x[idx]
    x1 = x[idx + 1]
    y0 = y[idx]
    y1 = y[idx + 1]
    slope = (y1 - y0) / (x1 - x0)
    return y0 + (r - x0) * slope
