###############################################################################
#   galpy.backend.sampling: backend-agnostic, differentiable inverse-CDF sampling.
#
#   The single home for galpy's spline inverse-CDF sampler, so the same drawing
#   code runs on numpy, jax, and torch and is differentiable (jax.grad /
#   torch.autograd) and jit/GPU-able (no Python-level rejection loop, static
#   shapes). It replaces per-DF adaptive-rejection sampling (ars) with a
#   reparameterized transform of uniforms u ~ U[0,1]: the sample is a pure,
#   differentiable function of u and the (distribution-parameter-dependent) CDF
#   grid, which is exactly what a reparameterization / common-random-numbers
#   pipeline needs.
#
#   WHY A DEDICATED SPLINE BUILDER (not interpolate.cubic_spline_coeffs)
#   ------------------------------------------------------------------
#   ``interpolate.cubic_spline_coeffs`` was built for the "fixed grid geometry,
#   differentiable y-VALUES" case: it converts its ``x`` to numpy
#   (``numpy.asarray(x)``) to assemble the tridiagonal matrix, so ``x`` is
#   treated as a CONSTANT and a TRACED/backend ``x`` (as under jax.grad / jax.jit)
#   raises a TracerArrayConversionError. Inverse-CDF sampling is the opposite
#   case: the knots ``x = cdf_grid`` are the parameter-dependent, differentiable
#   quantity (``y = omega_grid`` is the sample axis). ``_natknot_coeffs`` below
#   therefore assembles the tridiagonal system ENTIRELY in the namespace (a single
#   scatter into a zero matrix with STATIC index arrays, then ``xp.linalg.solve``),
#   so the coefficients -- and the sampled value -- are differentiable w.r.t. BOTH
#   ``cdf_grid`` and ``omega_grid`` and run under jit. Evaluation reuses
#   ``interpolate.eval_ppoly`` (already namespace-native and jit/grad-safe).
###############################################################################
import numpy

from .interpolate import eval_ppoly

__all__ = ["spline_inverse_cdf_sample", "ensure_strictly_increasing"]


def _scatter_matrix(xp, n, rows, cols, vals):
    """Return the (n, n) matrix with ``A[rows[k], cols[k]] = vals[k]`` (else 0).

    ``rows``/``cols`` are STATIC numpy integer index arrays (they depend only on
    the grid size, not on the traced values); ``vals`` is a namespace vector that
    carries the gradient. jax uses the functional ``.at[].set`` (jit/grad-safe);
    numpy/torch use in-place advanced-index assignment (torch's ``index_put`` is
    differentiable w.r.t. ``vals``).
    """
    if "jax" in xp.__name__:
        import jax.numpy as jnp

        return jnp.zeros((n, n), dtype=vals.dtype).at[rows, cols].set(vals)
    A = xp.zeros((n, n), dtype=vals.dtype)
    if "torch" in xp.__name__:
        import torch

        A[torch.as_tensor(rows), torch.as_tensor(cols)] = vals
    else:
        A[rows, cols] = vals
    return A


def _natknot_coeffs(xp, x, y):
    """Not-a-knot cubic-spline power-basis coefficients from ``(x, y)`` in ``xp``.

    Same ``(4, n-1)`` layout as ``interpolate.cubic_spline_coeffs`` /
    ``spline_to_ppoly`` -- on ``x[i] <= r < x[i+1]`` the spline is
    ``sum_j c[j, i] (r - x[i])**(3-j)`` -- so it feeds straight into
    ``eval_ppoly``. Unlike ``cubic_spline_coeffs`` this keeps ``x`` in the
    namespace (no ``numpy.asarray``), so the result is differentiable w.r.t. ``x``
    AND ``y`` and traces under jit. ``x`` must be strictly increasing.
    """
    n = x.shape[0]
    if n < 4:
        raise ValueError("_natknot_coeffs requires at least 4 points")
    h = x[1:] - x[:-1]  # (n-1,) traced step sizes
    dslope = (y[1:] - y[:-1]) / h  # (n-1,) secant slopes
    # Static (row, col) index arrays for A's nonzeros (structure depends on n
    # only). Interior rows i=1..n-2 are tridiagonal; the not-a-knot end rows each
    # carry three entries (row 0: cols 0,1,2; row n-1: cols n-3,n-2,n-1).
    i = numpy.arange(1, n - 1)
    rows = numpy.concatenate([i, i, i, [0, 0, 0], [n - 1, n - 1, n - 1]])
    cols = numpy.concatenate([i - 1, i, i + 1, [0, 1, 2], [n - 3, n - 2, n - 1]])
    concat = getattr(xp, "concat", None) or xp.concatenate
    # Values aligned with (rows, cols); all traced functions of h.
    vals = concat(
        [
            h[:-1],  # A[i, i-1] = h[i-1]
            2.0 * (h[:-1] + h[1:]),  # A[i, i]   = 2(h[i-1]+h[i])
            h[1:],  # A[i, i+1] = h[i]
            xp.stack([h[1], -(h[0] + h[1]), h[0]]),  # row 0 (not-a-knot)
            xp.stack([h[-1], -(h[-2] + h[-1]), h[-2]]),  # row n-1 (not-a-knot)
        ]
    )
    A = _scatter_matrix(xp, n, rows, cols, vals)
    # rhs (length n): interior 6(dslope[i]-dslope[i-1]); homogeneous end rows.
    zero = y[:1] * 0.0
    rhs = concat([zero, 6.0 * (dslope[1:] - dslope[:-1]), zero])
    M = xp.linalg.solve(A, rhs)  # second derivatives at the knots
    a3 = (M[1:] - M[:-1]) / (6.0 * h)
    a2 = M[:-1] / 2.0
    a1 = dslope - h * (2.0 * M[:-1] + M[1:]) / 6.0
    a0 = y[:-1]
    return xp.stack([a3, a2, a1, a0], axis=0)  # (4, n-1)


def ensure_strictly_increasing(xp, cdf_grid, floor=1e-12):
    """Project ``cdf_grid`` onto a strictly increasing grid (steps >= ``floor``).

    A closed-form CDF sampled to the far tails saturates to float ``1.0`` (and can
    dip a rounding-ulp below 0 at the bottom), leaving ZERO steps that make the
    spline knots non-strictly-increasing (a singular tridiagonal solve). This
    floors every step to ``floor`` and reintegrates by cumulative sum, so the
    result is strictly increasing with the SAME first value. It is a no-op in the
    bulk (real steps are orders of magnitude larger than ``floor``); only the
    saturated tail steps -- which carry negligible probability -- are nudged, so
    the sampled distribution is preserved. Differentiable (a.e.) and jit-safe:
    ``xp.maximum``/``cumsum``/``concat`` are namespace ops with STATIC shapes.
    """
    concat = getattr(xp, "concat", None) or xp.concatenate
    diff = cdf_grid[1:] - cdf_grid[:-1]
    # a same-dtype/device floor tensor (torch's maximum rejects a python float)
    steps = xp.maximum(diff, diff * 0.0 + floor)
    return concat([cdf_grid[:1], cdf_grid[:1] + xp.cumsum(steps, axis=0)])


def spline_inverse_cdf_sample(xp, omega_grid, cdf_grid, u):
    """Sample a scalar random variable by cubic-spline inversion of its CDF.

    Given a monotone CDF tabulated on a grid -- ``omega_grid`` (the sample axis,
    strictly increasing) and ``cdf_grid = F(omega_grid)`` (values in [0, 1],
    strictly increasing) -- and uniforms ``u`` in [0, 1], build the INVERSE spline
    (knots ``x = cdf_grid``, values ``y = omega_grid``) and return the sampled
    ``omega = spline(u) ~ F^{-1}(u)``.

    The whole thing is a pure, differentiable function of ``u``, ``omega_grid``,
    and ``cdf_grid``: the spline coefficients are built in the namespace
    (``_natknot_coeffs``) so ``domega/dcdf_grid`` and ``domega/domega_grid`` flow
    (jax.grad / torch.autograd), and there is no rejection loop, so it runs under
    ``jax.jit`` at a static output shape. ``cdf_grid`` typically depends on the
    distribution parameters, so this is the seam a differentiable sampler
    differentiates the parameters through.

    Parameters
    ----------
    xp : module
        The array namespace (numpy / jax.numpy / array-api-compat torch).
    omega_grid : array (n,)
        Strictly increasing sample-axis grid.
    cdf_grid : array (n,)
        ``F(omega_grid)``, strictly increasing in [0, 1]. Pass through
        :func:`ensure_strictly_increasing` first if the tabulated CDF can
        saturate (repeated float values) at the tails.
    u : array
        Uniform(0, 1) draws; the output has the same shape.

    Returns
    -------
    array
        ``F^{-1}(u)``, same shape as ``u``. ``u`` outside ``[cdf_grid[0],
        cdf_grid[-1]]`` is clamped to that range (returns the edge ``omega``),
        so the sample never leaves ``[omega_grid[0], omega_grid[-1]]``.
    """
    # The cubic-spline tridiagonal solve is singular in float32 (the
    # near-zero tail steps fall below float32 eps), which returns SILENT NaN
    # under jax's default (no jax_enable_x64) rather than failing loudly. galpy's
    # backends run in float64; fail clearly instead of poisoning the samples.
    if xp.finfo(cdf_grid.dtype).bits < 64:
        raise ValueError(
            "spline_inverse_cdf_sample requires float64 grids (the CDF-inversion "
            "spline solve is singular in float32 and would return silent NaN); "
            "for jax set jax_enable_x64=True."
        )
    coeffs = _natknot_coeffs(xp, cdf_grid, omega_grid)
    return eval_ppoly(xp, cdf_grid, coeffs, u, extrapolate="clip")
