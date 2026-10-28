###############################################################################
#   galpy.backend.sampling: backend-agnostic, differentiable inverse-CDF sampling.
#
#   The single home for galpy's inverse-CDF sampler, so the same drawing code
#   runs on numpy, jax, and torch and is differentiable (jax.grad /
#   torch.autograd) and jit/GPU-able (no Python-level rejection loop, static
#   shapes). It replaces per-DF adaptive-rejection sampling (ars) with a
#   reparameterized transform of uniforms u ~ U[0,1]: the sample is a pure,
#   differentiable function of u and the (distribution-parameter-dependent) CDF
#   grid, which is exactly what a reparameterization / common-random-numbers
#   pipeline needs. Inversion is piecewise-LINEAR (``interpolate.interp_linear``,
#   the monotone-robust choice sphericaldf adopted in #1181): the knots
#   ``x = cdf_grid`` are the parameter-dependent, differentiable quantity
#   (``y = omega_grid`` is the sample axis), so the sampled value carries the
#   gradient w.r.t. BOTH grids and traces under jit.
###############################################################################
from ._namespaces import stop_gradient
from .interpolate import interp_linear

__all__ = [
    "linear_inverse_cdf_sample",
    "batched_inverse_cdf_sample",
    "ensure_strictly_increasing",
    "trapezoid_sum",
    "trapezoid_cdf",
]


def _segments(p, grid, axis):
    # the trapezoid segments 0.5 (p_k + p_k+1) dx of a UNIFORM grid along axis
    d = grid[1] - grid[0]
    lo = [slice(None)] * p.ndim
    lo[axis] = slice(0, -1)
    hi = [slice(None)] * p.ndim
    hi[axis] = slice(1, None)
    return 0.5 * (p[tuple(lo)] + p[tuple(hi)]) * d


def trapezoid_sum(xp, p, grid, axis):
    """Trapezoid integral of ``p`` along ``axis`` over the uniform ``grid``."""
    return xp.sum(_segments(p, grid, axis), axis=axis)


def trapezoid_cdf(xp, p, grid, axis):
    """Normalized cumulative-trapezoid CDF of the density ``p`` along ``axis``.

    ``p`` is tabulated on the uniform ``grid`` (same length as ``p`` along
    ``axis``); the result has the same shape as ``p``, starts at 0 and ends at
    1 along ``axis``, and is differentiable in ``p``. A slice with no
    probability at all (``p == 0`` throughout) would normalize 0/0 -> NaN, and
    a conditional sampler that blends neighbouring slices would turn a whole
    draw into NaN; such a slice gets a uniform CDF instead -- it carries no
    weight in the marginal, so the choice is immaterial, but every CDF stays
    finite and non-decreasing. The division goes through a gradient-free copy
    of the total first: ``c/tot``'s derivative has a ``1/tot**2`` that
    overflows for a far-tail slice (``tot ~ 1e-177``) and NaN-poisons the
    sample's gradient, while the value is bit-identical (``tot/s == 1.0``).
    """
    c = xp.cumulative_sum(_segments(p, grid, axis), axis=axis)
    zsh = list(p.shape)
    zsh[axis] = 1
    c = xp.concatenate([xp.zeros(tuple(zsh), dtype=c.dtype), c], axis=axis)
    last = [slice(None)] * p.ndim
    last[axis] = slice(c.shape[axis] - 1, c.shape[axis])
    tot = c[tuple(last)]
    ramp = xp.reshape(
        xp.linspace(0.0, 1.0, c.shape[axis]),
        tuple(-1 if k == axis else 1 for k in range(p.ndim)),
    )
    good = tot > 0.0
    tot = xp.where(good, tot, xp.ones_like(tot))
    s = stop_gradient(tot)
    return xp.where(good, (c / s) / (tot / s), ramp)


def ensure_strictly_increasing(xp, cdf_grid, floor=1e-12):
    """Project ``cdf_grid`` onto a strictly increasing grid (steps >= ``floor``).

    A closed-form CDF sampled to the far tails saturates to float ``1.0`` (and can
    dip a rounding-ulp below 0 at the bottom), leaving ZERO steps that make the
    inversion knots non-strictly-increasing (an ambiguous ``searchsorted``). This
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


def linear_inverse_cdf_sample(xp, omega_grid, cdf_grid, u):
    """Sample a scalar random variable by PIECEWISE-LINEAR inversion of its CDF.

    Given a monotone CDF tabulated on a grid -- ``omega_grid`` (the sample axis,
    strictly increasing) and ``cdf_grid = F(omega_grid)`` (values in [0, 1],
    strictly increasing) -- and uniforms ``u`` in [0, 1], build the INVERSE
    interpolant (knots ``x = cdf_grid``, values ``y = omega_grid``) with a linear
    (:func:`~galpy.backend.interpolate.interp_linear`) fit and return the sampled
    ``omega ~ F^{-1}(u)``. Linear inversion has no tridiagonal solve, so it is
    monotone-robust and cannot overshoot between knots -- the behaviour
    sphericaldf adopted in #1181 for low-anisotropy grids.

    The whole thing is a pure, differentiable function of ``u``, ``omega_grid``,
    and ``cdf_grid`` (jax.grad / torch.autograd) and jit/GPU-able (searchsorted +
    lerp, static shapes, no rejection loop). ``cdf_grid`` typically depends on the
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
    return interp_linear(xp, cdf_grid, omega_grid, u, extrapolate="clip")


def batched_inverse_cdf_sample(xp, omega_grid, cdf_rows, u):
    """Per-sample piecewise-linear CDF inversion on a SHARED sample axis.

    Like :func:`linear_inverse_cdf_sample`, but each draw carries its OWN
    tabulated CDF (a row of ``cdf_rows``) while all rows share the one
    ``omega_grid`` sample axis. This is what a CONDITIONAL inverse-CDF sampler
    needs: e.g. sampling ``vR | vT`` where the vR-CDF has been interpolated to
    each draw's sampled ``vT`` (so every draw has a different CDF over the same
    vR grid). :func:`~galpy.backend.interpolate.interp_linear` cannot do this --
    its ``searchsorted`` takes a single 1-D knot vector -- so the interval search
    is done here by counting, ``sum(cdf_rows < u)``, which vectorises over rows.

    Parameters
    ----------
    xp : module
        Array namespace (numpy / jax.numpy / array-api-compat torch).
    omega_grid : array (n,)
        Strictly increasing sample-axis grid, shared by every row.
    cdf_rows : array (N, n)
        Per-draw CDFs, each strictly increasing in [0, 1] along the last axis.
    u : array (N,)
        Uniform(0, 1) draws, one per row.

    Returns
    -------
    array (N,)
        ``F_i^{-1}(u_i)`` for row ``i``, clamped to ``[omega_grid[0],
        omega_grid[-1]]``.
    """
    n = omega_grid.shape[0]
    # interval index per row: how many CDF knots lie below u, minus one
    idx = xp.clip(
        xp.astype(xp.sum(xp.astype(cdf_rows < u[:, None], xp.int64), axis=1), xp.int64)
        - 1,
        0,
        n - 2,
    )
    ar = xp.arange(u.shape[0])
    f0 = cdf_rows[ar, idx]
    f1 = cdf_rows[ar, idx + 1]
    # Guard the flat parts of the CDF with a double where, not an epsilon: a
    # zero-width step makes (u-f0)/1e-300 differentiate to ~1e300, which poisons
    # the whole gradient with NaN even though the VALUE is fine. Dividing by a
    # substituted 1.0 on those entries keeps both the value and the gradient
    # finite, and the entries carry no probability so the choice is immaterial.
    span = f1 - f0
    ok = span > 0.0
    frac = xp.where(ok, (u - f0) / xp.where(ok, span, xp.ones_like(span)), 0.0)
    return omega_grid[idx] + frac * (omega_grid[idx + 1] - omega_grid[idx])
