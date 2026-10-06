###############################################################################
#   galpy.backend._torch.adiabatic_c
#
#   torch wrapper for the compiled Adiabatic C actions. actions_with_jac is a
#   torch.autograd.Function whose forward runs the C entry returning (jr, jz)
#   AND the fused 2x5 Jacobian d(jr,jz)/d(R,vR,vT,z,vz) (assembled natively in
#   C), saving the Jacobian; backward is a matvec of that Jacobian. Mirrors the
#   Staeckel staeckel_c.actions_with_jac (#131 Adiabatic PR-2a). First-order.
###############################################################################
import torch


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
        g = g_jr[:, None] * jac[:, 0, :] + g_jz[:, None] * jac[:, 1, :]  # (N,5)
        return (None,) + tuple(g[:, k] for k in range(5))


def actions_with_jac(host_jac, R, vR, vT, z, vz):
    """Differentiable (jr, jz) with the native-C Jacobian as the backward matvec.

    host_jac : callable taking 5 numpy (N,) coord arrays, returning
        (jr, jz, jac) with jr,jz (N,) and jac (N,2,5).
    """
    return _ActionsJacFunction.apply(host_jac, R, vR, vT, z, vz)


class _EccZmaxJacFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, host_jac, R, vR, vT, z, vz):
        cs = [t.detach().to("cpu", torch.float64).numpy() for t in (R, vR, vT, z, vz)]
        ecc, zmax, rperi, rap, jac = host_jac(*cs)
        dev, dt = R.device, R.dtype
        ctx.save_for_backward(torch.as_tensor(jac, dtype=dt, device=dev))  # (N,4,5)
        return tuple(
            torch.as_tensor(x, dtype=dt, device=dev) for x in (ecc, zmax, rperi, rap)
        )

    @staticmethod
    def backward(ctx, g_ecc, g_zmax, g_rperi, g_rap):
        (jac,) = ctx.saved_tensors  # (N,4,5)
        gs = (g_ecc, g_zmax, g_rperi, g_rap)
        g = sum(gs[o][:, None] * jac[:, o, :] for o in range(4))  # (N,5)
        return (None,) + tuple(g[:, k] for k in range(5))


def ecczmax_with_jac(host_jac, R, vR, vT, z, vz):
    """Differentiable (ecc, zmax, rperi, rap) with the native-C (N,4,5) Jacobian
    as the backward matvec. Mirrors actions_with_jac (#131 Adiabatic PR-2c).

    host_jac : callable taking 5 numpy (N,) coord arrays, returning
        (ecc, zmax, rperi, rap, jac) with the 4 values (N,) and jac (N,4,5).
    """
    return _EccZmaxJacFunction.apply(host_jac, R, vR, vT, z, vz)
