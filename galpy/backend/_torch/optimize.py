###############################################################################
#   galpy.backend._torch.optimize: torch half of galpy.backend.optimize.brentq.
#
#   Vectorised sign-preserving bisection (shared bisect_root) followed by the
#   one-Newton-step reparameterisation that gives exact implicit-function-theorem
#   gradients w.r.t. the parameters f closes over. See galpy.backend.optimize for
#   the math. The jax counterpart is galpy.backend._jax.optimize.
###############################################################################


def brentq_backend(
    f, a, b, xp, *, xtol, maxiter, width=None, guess=None, newton_steps=10
):
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

    from ..optimize import bisect_root, newton_polish

    # Bisection root, detached: its branchy comparisons carry no useful gradient,
    # so x0 is a constant w.r.t. the parameters; the Newton step restores the
    # parameter sensitivity via the implicit-function theorem.
    if guess is None:
        with torch.no_grad():
            x0 = bisect_root(f, a, b, xp, xtol=xtol, maxiter=maxiter, width=width)
    else:
        x0 = _newton_root(f, a, b, guess, xp, newton_steps)
    x0 = x0.detach()
    # f(x0) carries f's PARAMETER grad-dependence (x0 is a constant w.r.t. them).
    # When no parameter requires grad -- the plain forward, e.g. the existing
    # numpy-input test suite run under a forced backend -- there is nothing to
    # differentiate, so return the detached bisection root directly. This keeps
    # the forward output free of an autograd graph (a grad-requiring tensor would
    # break callers that do plain ``.numpy()``), while still entering the
    # implicit-function reparameterisation below whenever a parameter needs grad.
    fx0 = f(x0)

    def _needs_grad(t):
        return torch.is_tensor(t) and t.requires_grad

    # Enter the implicit-function reparam whenever differentiation is intended:
    # grad enabled AND some input requires grad -- either one of f's parameters
    # (carried by fx0) OR a bracket endpoint (d(root)/d(endpoint) is 0, but the
    # caller may still .backward() through it). Otherwise -- the plain forward,
    # e.g. the numpy-input suite under a forced backend -- return the detached
    # bisection root so callers can .numpy() it.
    if not (
        torch.is_grad_enabled()
        and (fx0.requires_grad or _needs_grad(a) or _needs_grad(b))
    ):
        # Still take the Newton step, for the VALUE. Bisection alone stops at
        # the final bracket half-width (~1e-14 here), and callers that go on to
        # form sqrt(f(x0)) amplify that residual -- actionAngleVertical's Omega
        # divides by sqrt(2(E-Phi(xmax))) and turned a 4e-14 residual into a
        # 6e-10 frequency error, while jax (which always polishes) stayed at
        # 1e-13. df/dx needs autograd, so enable it locally and detach: the
        # returned tensor still carries no graph, which is what this branch is
        # for (callers here go straight to .numpy()).
        with torch.enable_grad():
            xr = x0.clone().requires_grad_(True)
            fxr = f(xr)
            (dfdx,) = torch.autograd.grad(fxr, xr, grad_outputs=torch.ones_like(fxr))
        return newton_polish(x0, fx0, dfdx.detach(), xp).detach()
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
    return newton_polish(x0, fxr, dfdx, xp)


def _newton_root(f, a, b, guess, xp, steps):
    """Safeguarded Newton from ``guess`` (see optimize.newton_step_bracketed),
    df/dx by autograd on a detached x (the iterate carries no graph)."""
    import torch

    from .._namespaces import asarray_on_device, device_of
    from ..optimize import newton_step_bracketed

    def fs(x):
        with torch.enable_grad():
            xr = x.detach().requires_grad_(True)
            fx = f(xr)
            (dfdx,) = torch.autograd.grad(fx, xr, grad_outputs=torch.ones_like(fx))
        return fx.detach(), dfdx.detach()

    # torch.as_tensor would copy a CUDA guess/bracket to a CPU default device
    dev = device_of(guess, a, b)
    x = torch.clamp(asarray_on_device(xp, guess, dev) * 1.0, min=a, max=b).detach()
    lo = (asarray_on_device(xp, a, dev) + 0.0 * x).detach()
    hi = (asarray_on_device(xp, b, dev) + 0.0 * x).detach()
    for _ in range(steps):
        x, lo, hi = newton_step_bracketed(fs, x, lo, hi, xp)
    return x


def fixed_point_backend(func, x0, args, xp, *, xtol, maxiter):
    """torch half of optimize.fixed_point: del2 on detached args, then the
    implicit-function step when a parameter needs grad."""
    import torch

    from ..optimize import del2_iterate, nonzero_slope

    dargs = tuple(a.detach() if torch.is_tensor(a) else a for a in args)
    f = lambda x: func(x, *dargs)
    with torch.no_grad():
        p = del2_iterate(f, x0.detach() * 1.0, xp, xtol, maxiter)
    g = func(p, *args)
    if not (torch.is_grad_enabled() and g.requires_grad):
        return p
    with torch.enable_grad():
        xr = p.clone().requires_grad_(True)
        fx = f(xr)
        (slope,) = torch.autograd.grad(fx, xr, grad_outputs=torch.ones_like(fx))
    return p + (g - g.detach()) / nonzero_slope(1.0 - slope, xp)
