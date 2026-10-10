###############################################################################
#   galpy.backend._torch.linalg: torch half of galpy.backend.linalg.real_eig.
###############################################################################
import torch

from ..linalg import degenerate_safe_gap_inverse


class _SafeEigh(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(a):
        return tuple(torch.linalg.eigh(a))

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*output)

    @staticmethod
    def backward(ctx, glam, gv):
        # differentiable ops of the saved OUTPUTS -> double backward works
        lam, v = ctx.saved_tensors
        vt = v.transpose(-1, -2)
        g = torch.diag_embed(glam) + degenerate_safe_gap_inverse(lam) * (vt @ gv)
        abar = v @ g @ vt
        return 0.5 * (abar + abar.transpose(-1, -2))


def safe_eigh(a):
    """``eigh`` whose eigenvector derivative is zero inside degenerate clusters."""
    return _SafeEigh.apply(a)
