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
