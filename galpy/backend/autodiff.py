###############################################################################
#   galpy.backend.autodiff: functional autodiff dispatch (jax / torch).
#
#   Returns the (grad, vmap) pair for the active array namespace so that
#   backend-agnostic code (e.g. constantbetadf's fE inversion) can build a
#   nested-derivative closure once and differentiate it under either engine.
###############################################################################


def autodiff_ops(xp):
    """Return ``(grad, vmap)`` functional-autodiff operators for namespace ``xp``.

    jax -> ``(jax.grad, jax.vmap)``; torch -> ``(torch.func.grad,
    torch.func.vmap)``. numpy has no autodiff and raises (the caller picks
    jax/torch itself). ``torch.func.grad`` is scalar-output only, which suits
    the fE chain: it differentiates a scalar-per-radius function, then vmaps
    over the radius axis.
    """
    name = getattr(xp, "__name__", "")
    if name in ("jax", "jax.numpy"):
        from jax import grad, vmap

        return grad, vmap
    if "torch" in name:
        import torch

        return torch.func.grad, torch.func.vmap
    raise ValueError(
        "autodiff_ops requires a jax or torch namespace (numpy has no autodiff)"
    )


def graft_derivative(x, value, deriv, higher):
    """Backend array equal to ``value`` whose derivatives w.r.t. the scalar ``x``
    are ``deriv`` (first order) and those of ``higher(x)`` (second order and up).

    For a quantity computed in numpy (e.g. a scipy ODE solve) whose first
    derivative is known in closed form (its forward sensitivities) but whose
    backend twin ``higher`` -- the same computation in jax/torch, equal to
    ``value`` up to its own accuracy -- is too slow to run for every gradient.
    ``higher`` runs only when a derivative of the derivative is taken. ``x``
    selects the backend (jax: ``jax.custom_jvp``; torch: an autograd.Function).
    """
    if "torch" in type(x).__module__:
        from ._torch.graft import graft_derivative as _graft
    else:
        from ._jax.graft import graft_derivative as _graft
    return _graft(x, value, deriv, higher)
