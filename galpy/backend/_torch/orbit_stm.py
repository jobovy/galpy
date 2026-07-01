###############################################################################
#   galpy.backend._torch.orbit_stm
#
#   torch.autograd.Function wrapping galpy's compiled C variational integrator
#   for differentiable fast orbit integration. Forward runs the C integrator and
#   saves the state-transition matrix M(t)=d x(t)/d x(0); backward returns the
#   IC gradient as sum_t M(t)^T grad_out[t]. See galpy.backend._reference
#   .inbackend_stm for the convention (Orbit order, IC-only gradients).
###############################################################################
import numpy
import torch

from .._reference.inbackend_stm import c_stm_forward


class _CSTMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vxvv, pot, ts, method, rtol, atol):
        # The C integrator is CPU/float64; move off-device + to numpy for the call.
        vnp = vxvv.detach().to("cpu", torch.float64).numpy()
        tnp = (
            ts.detach().to("cpu", torch.float64).numpy()
            if torch.is_tensor(ts)
            else numpy.asarray(ts, dtype=numpy.float64)
        )
        xt, M = c_stm_forward(pot, vnp, tnp, method, rtol, atol)
        ctx.save_for_backward(torch.as_tensor(M, dtype=vxvv.dtype, device=vxvv.device))
        ctx.single = vxvv.ndim == 1
        return torch.as_tensor(xt, dtype=vxvv.dtype, device=vxvv.device)

    @staticmethod
    def backward(ctx, grad_out):
        (M,) = ctx.saved_tensors
        if ctx.single:  # M (nt,6,6), grad_out (nt,6) -> (6,)
            vbar = torch.einsum("tab,ta->b", M, grad_out)
        else:  # M (N,nt,6,6), grad_out (N,nt,6) -> (N,6)
            vbar = torch.einsum("ntab,nta->nb", M, grad_out)
        # gradients: vxvv only; (pot, ts, method, rtol, atol) are non-differentiable
        return vbar, None, None, None, None, None


def integrate(pot, vxvv, ts, *, method="dop853_c", rtol=1e-10, atol=1e-10):
    """Differentiable C-STM orbit integration under torch.

    vxvv: torch tensor (6,) or (N,6), Orbit order. Returns (nt,6) or (N,nt,6).
    Differentiable w.r.t. vxvv.
    """
    return _CSTMFunction.apply(vxvv, pot, ts, method, rtol, atol)
