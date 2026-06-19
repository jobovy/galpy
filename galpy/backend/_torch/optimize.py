###############################################################################
#   galpy.backend._torch.optimize: torch half of galpy.backend.optimize.brentq.
#
#   Vectorised sign-preserving bisection (shared bisect_root) followed by the
#   one-Newton-step reparameterisation that gives exact implicit-function-theorem
#   gradients w.r.t. the parameters f closes over. See galpy.backend.optimize for
#   the math. The jax counterpart is galpy.backend._jax.optimize.
###############################################################################


def brentq_backend(f, a, b, xp, *, xtol, maxiter):
    """torch bracketed root of ``f`` on ``[a, b]``, differentiable in f's params.

    ``f`` is the single-argument closure ``x -> func(x, *args)`` in the
    array-api-compat torch namespace. Bisection localises the root (its value is
    piecewise-constant, so it is detached); the Newton reparameterisation

        x* = x0 - f(x0) / f'(x0),   x0 = detach(bisection root),

    keeps the forward value at the bisection root (f(x0) ~ 0) while propagating
    the exact implicit-function gradient dx*/dtheta = -(df/dtheta)/(df/dx). f'(x0)
    is computed with ``torch.autograd.grad(..., create_graph=True)`` against an
    x-slot leaf, so f'(x0) itself stays differentiable w.r.t. f's parameters and
    ``.backward()`` through x* recovers the implicit gradient. No internal jit.
    """
    import torch

    from ..optimize import bisect_root

    # Bisection root, detached: its branchy comparisons carry no useful gradient,
    # so x0 is a constant w.r.t. the parameters; the Newton step restores the
    # parameter sensitivity via the implicit-function theorem.
    with torch.no_grad():
        x0 = bisect_root(f, a, b, xp, xtol=xtol, maxiter=maxiter)
    x0 = x0.detach()
    # x-slot leaf for the elementwise df/dx (a fresh copy that requires grad in
    # the x argument only; the parameters keep their own grad tracking through f).
    xr = x0.clone().requires_grad_(True)
    fxr = f(xr)
    (dfdx,) = torch.autograd.grad(
        fxr,
        xr,
        grad_outputs=torch.ones_like(fxr),
        create_graph=True,  # keep df/dx differentiable w.r.t. f's parameters
    )
    # fxr carries the parameter dependence of f(x0) (xr is detached from params);
    # dfdx carries df/dx and its parameter dependence. x0 is constant. So the
    # Newton step is differentiable w.r.t. every parameter f closes over.
    return x0 - fxr / _nonzero(dfdx, xp)


def _nonzero(d, xp):
    """Replace (near-)zero slopes by a tiny same-sign value so 1/d stays finite.

    A vanishing df/dx is a genuinely ill-posed root (tangential crossing); rather
    than emit inf/NaN that would poison reverse-mode AD, floor |d| at a tiny
    epsilon keeping its sign. Dead-branch guarded: both ``xp.where`` sides are
    evaluated eagerly, and the flooring keeps every branch finite.
    """
    ones = xp.ones_like(d)
    sign = xp.where(d >= 0, ones, -ones)
    return sign * xp.maximum(xp.abs(d), 1e-300 * ones)
