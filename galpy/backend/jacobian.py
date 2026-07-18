###############################################################################
#   galpy.backend.jacobian: backend-agnostic functional Jacobian (jax / torch).
#
#   ``jacobian(f, x)`` returns the dense Jacobian ``df/dx`` of a vector->vector
#   map by the active backend's automatic differentiation -- exact (no
#   finite-difference truncation) and itself differentiable, so d(Jacobian)/d
#   (any parameter ``f`` closes over) flows (higher-order AD). Used by streamdf's
#   ``calcaAJac`` to replace its finite-difference d(J,Omega,theta)/d(x,v) on the
#   backend path. There is NO numpy branch here (numpy keeps its own FD code); a
#   numpy namespace raises, since numpy has no autodiff.
#
#   No internal jit/compile (galpy is jit-COMPATIBLE, not jit-ing): the returned
#   Jacobian composes with the user's own jax.jit / torch.compile / grad.
###############################################################################
from ._namespaces import name_of_namespace
from ._resolver import get_namespace


def jacobian(f, x, xp=None):
    """Jacobian ``df/dx`` of a vector->vector map ``f`` by backend autodiff.

    Parameters
    ----------
    f : callable
        ``f(x) -> backend array`` written in the array namespace (so it is
        differentiable). ``x`` is a 1-D backend array; ``f(x)`` is a 1-D backend
        array. The returned Jacobian has shape ``(len(f(x)), len(x))``.
    x : backend array
        The (1-D) point at which to evaluate the Jacobian. Its namespace selects
        the backend.
    xp : module, optional
        The array namespace; resolved from ``x`` when omitted.

    Returns
    -------
    backend array
        The dense ``(m, n)`` Jacobian, itself differentiable w.r.t. any parameter
        ``f`` closes over (jax: composable jacrev; torch: ``create_graph=True``).

    Notes
    -----
    - jax uses ``jax.jacrev`` (reverse mode): galpy's C-STM orbit integrator is a
      ``custom_vjp``, so forward-mode ``jacfwd`` is unavailable.
    - torch uses ``torch.autograd.functional.jacobian(..., vectorize=False)``:
      the C-STM custom autograd Function has no vmap batching rule, so the
      ``vectorize=True`` fast path is unavailable.
    """
    if xp is None:
        xp = get_namespace(x)
    name = name_of_namespace(xp)
    if name == "jax":
        from ._jax.jacobian import jacobian_backend
    elif name == "torch":
        from ._torch.jacobian import jacobian_backend
    else:
        raise ValueError(
            "backend jacobian requires a jax or torch namespace (numpy has no autodiff)"
        )
    return jacobian_backend(f, x)
