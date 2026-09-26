###############################################################################
#   galpy.backend._torch.graft: torch half of galpy.backend.autodiff.graft_derivative.
###############################################################################


def graft_derivative(x, value, deriv, higher):
    """``value`` with d/dx = ``deriv``; higher orders from ``higher(x)``."""
    import torch

    class _Graft(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return torch.as_tensor(value, dtype=x.dtype, device=x.device)

        @staticmethod
        def backward(ctx, g):
            (x,) = ctx.saved_tensors
            if not torch.is_grad_enabled():  # first order: the given derivative
                d = torch.as_tensor(deriv, dtype=g.dtype, device=g.device)
                return (g * d).sum().reshape(x.shape)
            # create_graph: a backward that is itself differentiable in x
            with torch.enable_grad():
                xx = x if x.requires_grad else x.detach().requires_grad_()
                (v,) = torch.autograd.grad(higher(xx), xx, g, create_graph=True)
            return v

    return _Graft.apply(x)
