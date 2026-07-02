###############################################################################
#   galpy.backend._torch.staeckel_c
#
#   torch wrappers for the compiled Staeckel C actions. actions_with_jac is a
#   torch.autograd.Function whose forward runs the C entry returning (jr, jz)
#   AND the full 2x5 Jacobian d(jr,jz)/d(R,vR,vT,z,vz) (assembled natively in
#   C), saving the Jacobian; backward is a matvec of that Jacobian. Registered
#   exactly like the C-STM autograd.Function (orbit_stm). First-order only.
###############################################################################
import numpy
import torch


def c_value(host, coords, nout):
    """Forward a numpy-in/numpy-out C `host` under torch (value only, no grad).

    host : callable taking len(coords) numpy (N,) arrays, returning `nout` numpy
        (N,) arrays. coords : tuple of torch (N,) tensors. Used for the
        frequency/angle values, which are not differentiated here (their
        gradients are second-order objects). Detached in, so no graph is built.
    """
    dev, dt = coords[0].device, coords[0].dtype
    cs = [t.detach().to("cpu", torch.float64).numpy() for t in coords]
    out = host(*cs)
    return tuple(
        torch.as_tensor(numpy.asarray(o, dtype=numpy.float64), dtype=dt, device=dev)
        for o in out
    )


class _ActionsJacFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, host_jac, R, vR, vT, z, vz):
        # The C entry is CPU/float64; move off-device + to numpy for the call.
        cs = [t.detach().to("cpu", torch.float64).numpy() for t in (R, vR, vT, z, vz)]
        jr, jz, jac = host_jac(*cs)
        dev, dt = R.device, R.dtype
        ctx.save_for_backward(torch.as_tensor(jac, dtype=dt, device=dev))  # (N,2,5)
        return (
            torch.as_tensor(jr, dtype=dt, device=dev),
            torch.as_tensor(jz, dtype=dt, device=dev),
        )

    @staticmethod
    def backward(ctx, g_jr, g_jz):
        (jac,) = ctx.saved_tensors  # (N,2,5)
        # grad_k = g_jr * dJr/dx_k + g_jz * dJz/dx_k
        g = g_jr[:, None] * jac[:, 0, :] + g_jz[:, None] * jac[:, 1, :]  # (N,5)
        # gradients: host_jac (non-diff) + the 5 coords
        return (None,) + tuple(g[:, k] for k in range(5))


def actions_with_jac(host_jac, R, vR, vT, z, vz):
    """Differentiable (jr, jz) with the native-C Jacobian as the backward matvec.

    host_jac : callable taking 5 numpy (N,) coord arrays, returning
        (jr, jz, jac) with jr,jz (N,) and jac (N,2,5).
    """
    return _ActionsJacFunction.apply(host_jac, R, vR, vT, z, vz)
