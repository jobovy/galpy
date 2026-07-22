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
from .interpolate import interp_linear

__all__ = [
    "linear_inverse_cdf_sample",
    "ensure_strictly_increasing",
]


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
