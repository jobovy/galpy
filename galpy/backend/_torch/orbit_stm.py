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


class _NoSecondOrder(torch.autograd.Function):
    """A zero whose backward raises: ties the C-STM's IC gradient back to the IC,
    so differentiating that gradient again (d/d(IC) of a Jacobian) fails loudly
    instead of silently dropping dM/d(IC)."""

    @staticmethod
    def forward(ctx, x):
        return x.new_zeros(())

    @staticmethod
    def backward(ctx, g):
        raise RuntimeError(
            "second derivatives through the C state-transition matrix are not "
            "supported (it is first-order only); integrate with "
            "method='torchdiffeq' or 'torchode' (for an actionAngle object: "
            "integrate_method='torchdiffeq')"
        )


class _CSTMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vxvv, pot, ts, method, rtol, atol):
        if ctx.needs_input_grad[2]:
            raise ValueError(
                "the C-STM carries d/d(initial condition) only, not d/d(times); "
                "integrate with method='torchdiffeq' to differentiate w.r.t. t"
            )
        # The C integrator is CPU/float64; move off-device + to numpy for the call.
        vnp = vxvv.detach().to("cpu", torch.float64).numpy()
        tnp = (
            ts.detach().to("cpu", torch.float64).numpy()
            if torch.is_tensor(ts)
            else numpy.asarray(ts, dtype=numpy.float64)
        )
        xt, M = c_stm_forward(pot, vnp, tnp, method, rtol, atol)
        ctx.save_for_backward(
            torch.as_tensor(M, dtype=vxvv.dtype, device=vxvv.device), vxvv
        )
        ctx.single = vxvv.ndim == 1
        return torch.as_tensor(xt, dtype=vxvv.dtype, device=vxvv.device)

    @staticmethod
    def backward(ctx, grad_out):
        M, vxvv = ctx.saved_tensors
        if ctx.single:  # M (nt,d,d), grad_out (nt,d) -> (d,)
            vbar = torch.einsum("tab,ta->b", M, grad_out)
        else:  # M (N,nt,d,d), grad_out (N,nt,d) -> (N,d)
            vbar = torch.einsum("ntab,nta->nb", M, grad_out)
        if torch.is_grad_enabled():  # create_graph: vbar is linear in grad_out
            vbar = vbar + _NoSecondOrder.apply(vxvv)
        # gradients: vxvv only; (pot, ts, method, rtol, atol) are non-differentiable
        return vbar, None, None, None, None, None


def integrate(pot, vxvv, ts, *, method="dop853_c", rtol=1e-10, atol=1e-10):
    """Differentiable C-STM orbit integration under torch.

    vxvv: torch tensor (d,) or (N,d), d in {6,4,2} (3D/planar/1D), Orbit order.
    Returns (nt,d) or (N,nt,d). Differentiable w.r.t. vxvv.
    """
    return _CSTMFunction.apply(vxvv, pot, ts, method, rtol, atol)
