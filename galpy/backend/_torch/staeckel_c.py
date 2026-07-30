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


class _ActionsFreqsJacFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, host_jac, R, vR, vT, z, vz):
        cs = [t.detach().to("cpu", torch.float64).numpy() for t in (R, vR, vT, z, vz)]
        jr, jz, Or, Op, Oz, jac = host_jac(*cs)
        dev, dt = R.device, R.dtype
        ctx.save_for_backward(torch.as_tensor(jac, dtype=dt, device=dev))  # (N,5,5)
        return tuple(
            torch.as_tensor(numpy.asarray(x, dtype=numpy.float64), dtype=dt, device=dev)
            for x in (jr, jz, Or, Op, Oz)
        )

    @staticmethod
    def backward(ctx, g_jr, g_jz, g_Or, g_Op, g_Oz):
        (jac,) = ctx.saved_tensors  # (N,5,5)
        gs = (g_jr, g_jz, g_Or, g_Op, g_Oz)
        g = sum(gs[o][:, None] * jac[:, o, :] for o in range(5))  # (N,5)
        return (None,) + tuple(g[:, k] for k in range(5))


def actionsfreqs_with_jac(host_jac, R, vR, vT, z, vz):
    """Differentiable (jr,jz,Omegar,Omegaphi,Omegaz) with the native-C fused (5,5)
    Jacobian as the backward matvec (#131). host_jac returns (jr,jz,Or,Op,Oz,jac)
    with the values (N,) and jac (N,5,5)."""
    return _ActionsFreqsJacFunction.apply(host_jac, R, vR, vT, z, vz)


class _ActionsFreqsAnglesJacFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, host_jac, R, vR, vT, z, vz):
        cs = [t.detach().to("cpu", torch.float64).numpy() for t in (R, vR, vT, z, vz)]
        out = host_jac(*cs)  # 8 values + jac (N,8,5)
        vals, jac = out[:8], out[8]
        dev, dt = R.device, R.dtype
        ctx.save_for_backward(torch.as_tensor(jac, dtype=dt, device=dev))  # (N,8,5)
        return tuple(
            torch.as_tensor(numpy.asarray(v, dtype=numpy.float64), dtype=dt, device=dev)
            for v in vals
        )

    @staticmethod
    def backward(ctx, *cts):  # 8 cotangents (jr,jz,Or,Op,Oz,angler,anglephi,anglez)
        (jac,) = ctx.saved_tensors  # (N,8,5)
        g = sum(cts[o][:, None] * jac[:, o, :] for o in range(8))  # (N,5)
        return (None,) + tuple(g[:, k] for k in range(5))


def actionsfreqsangles_with_jac(host_jac, R, vR, vT, z, vz, phi):
    """Differentiable (jr,jz,Omegar,Omegaphi,Omegaz,angler,anglephi,anglez) with the
    native-C stacked (8,5) Jacobian as the backward matvec (#131 PR-B). phi enters
    analytically via the plain remainder below (d anglephi/dphi==1). host_jac returns
    (jr,jz,Or,Op,Oz,angler,anglephi_raw,anglez,jac) with the 8 values (N,) --
    angler/anglez wrapped, anglephi WITHOUT phi -- and jac (N,8,5)."""
    raw = _ActionsFreqsAnglesJacFunction.apply(host_jac, R, vR, vT, z, vz)
    # C flags an unbound orbit by returning 9999.99 in every output, and that
    # sentinel must survive the azimuth wrap -- folding it gives 3.44, which reads
    # as an ordinary angle. The numpy C wrapper guards this the same way
    # (actionAngleStaeckel_c.py, `badAngle = Anglephi != 9999.99`); the literal is
    # the C ABI's, spelled out here as it is in every other consumer rather than
    # importing it from galpy.actionAngle, which this module is imported BY.
    # where() on the data, not a branch, so it still traces; both sides are finite,
    # so the dead one cannot poison the gradient.
    anglephi = torch.where(
        raw[6] == 9999.99, raw[6], torch.remainder(raw[6] + phi, 2.0 * torch.pi)
    )
    return raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], anglephi, raw[7]


class _EccZmaxJacFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, host_jac, R, vR, vT, z, vz):
        cs = [t.detach().to("cpu", torch.float64).numpy() for t in (R, vR, vT, z, vz)]
        e, zm, rp, ra, jac = host_jac(*cs)  # 4 values + jac (N,4,5)
        dev, dt = R.device, R.dtype
        ctx.save_for_backward(torch.as_tensor(jac, dtype=dt, device=dev))  # (N,4,5)
        return tuple(
            torch.as_tensor(numpy.asarray(x, dtype=numpy.float64), dtype=dt, device=dev)
            for x in (e, zm, rp, ra)
        )

    @staticmethod
    def backward(ctx, g_e, g_zm, g_rp, g_ra):
        (jac,) = ctx.saved_tensors  # (N,4,5)
        gs = (g_e, g_zm, g_rp, g_ra)
        g = sum(gs[o][:, None] * jac[:, o, :] for o in range(4))  # (N,5)
        return (None,) + tuple(g[:, k] for k in range(5))


def ecczmax_with_jac(host_jac, R, vR, vT, z, vz):
    """Differentiable (e,zmax,rperi,rap) with the native-C (4,5) Jacobian as the
    backward matvec (#131). first-order; partial at fixed delta. host_jac returns
    (e,zmax,rperi,rap,jac) with the 4 values (N,) and jac (N,4,5)."""
    return _EccZmaxJacFunction.apply(host_jac, R, vR, vT, z, vz)
