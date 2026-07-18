###############################################################################
#   galpy.backend._torch.jacobian: torch half of galpy.backend.jacobian.
#
#   torch.autograd.functional.jacobian with create_graph=True (so d(Jacobian)/d
#   (params f closes over) flows -- higher-order AD, the point of streamdf's
#   backend Jacobian) and vectorize=False (the C-STM custom autograd Function has
#   no vmap batching rule, so the vectorize=True fast path is unavailable).
###############################################################################


def jacobian_backend(f, x):
    """``torch.autograd.functional.jacobian(f, x, create_graph=True)`` (dense)."""
    import torch

    return torch.autograd.functional.jacobian(f, x, create_graph=True, vectorize=False)
