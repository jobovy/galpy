###############################################################################
#   galpy.backend._torch.no_derivative: a host-evaluated value with a NaN
#   derivative.
###############################################################################
import torch


class _NoDerivative(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, out):
        return out.clone()

    @staticmethod
    def backward(ctx, g):
        return g * float("nan"), None


def no_derivative(a, out):
    """``out``, computed from ``a`` off the graph, with a NaN derivative w.r.t.
    ``a``: it has none, and a silent zero would be a wrong gradient."""
    return _NoDerivative.apply(a, out)
