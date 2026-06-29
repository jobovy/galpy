###############################################################################
#   galpy.backend._reference.inbackend_stm
#
#   Differentiable FAST-C orbit integration via the state-transition matrix
#   (STM). The forward solve is galpy's compiled C variational integrator
#   (integrate_dxdv); the gradient w.r.t. the initial conditions is the STM
#   M(t) = d x(t) / d x(0) that the same C integration already produces, applied
#   as M(t)^T to the output cotangent (forward-sensitivity adjoint -- a single
#   matrix-transpose-vector product, no backward C call).
#
#   This is the fast first-order-differentiable orbit path for the jax/torch
#   backends; the in-backend ODE path (inbackend_ode.py) is the independent,
#   higher-order / parameter-gradient cross-check.
#
#   Convention: phase-space vectors use galpy's Orbit ordering
#   ``[R, vR, vT, z, vz, phi]``. We assemble M directly in this frame by
#   propagating the 6 canonical basis deviation vectors with
#   ``integrate_dxdv(..., rectIn=False, rectOut=False)`` -- so x(t)=M(t) x(0)
#   holds in Orbit order with no cylindrical<->rectangular Jacobian folding
#   (verified vs finite-difference of the flow). Gradients are w.r.t. the
#   initial conditions only; potential-parameter gradients use the in-backend
#   ODE path (the C integrator carries no d x / d theta sensitivity).
###############################################################################
import numpy

from .. import get_namespace

# dxdv-capable C integrators (the variational RHS is wired for these).
_C_DXDV_METHODS = ("rk4_c", "rk6_c", "dopr54_c", "dop853_c")


def c_stm_forward(pot, vxvv, ts, method, rtol, atol):
    """Run the C variational integrator and return (x(t), M(t)).

    Pure numpy; no autodiff. This is the host-side call wrapped by the jax/torch
    autodiff rules. Uses the AUGMENTED 42-state integrator -- the base orbit plus
    all six STM columns in ONE solve (the force + 3D Hessian are evaluated once per
    step, not six times), ~4-6x faster than re-integrating the base per column. For
    the adaptive methods (dop853_c/dopr54_c) the joint 42-state error norm accepts a
    slightly different step sequence than six 12-state solves, so M matches the
    per-column reference (``_c_stm_forward_loop``) to ~1e-10, not bit-for-bit; the
    gradient is unchanged. See ``_c_stm_forward_loop`` for the reference path.

    Parameters
    ----------
    pot : Potential (or list)
    vxvv : numpy.ndarray, shape (6,) or (N, 6), Orbit order [R,vR,vT,z,vz,phi].
    ts : numpy.ndarray, shape (nt,), output times (ts[0] is the initial time).
    method : str, one of ``_C_DXDV_METHODS``.
    rtol, atol : float

    Returns
    -------
    xt : numpy.ndarray, (nt, 6) or (N, nt, 6) -- the orbit, Orbit order.
    M : numpy.ndarray, (nt, 6, 6) or (N, nt, 6, 6) -- STM d x(t)/d x(0),
        M[...,0,:,:] = identity.
    """
    from ...orbit.integrateFullOrbit import integrateFullOrbit_stm_c
    from ...util import coords

    vxvv = numpy.asarray(vxvv, dtype=numpy.float64)
    single = vxvv.ndim == 1
    ics = vxvv[None] if single else vxvv
    ts = numpy.asarray(ts, dtype=numpy.float64)
    nt = len(ts)
    eye6 = numpy.eye(6)
    int_method = method.lower()
    xts, Ms = [], []
    for ic in ics:
        R, vR, vT, z, vz, phi = ic
        # cyl -> rect base, and the six cyl basis deviations folded to rect =
        # the columns of the cyl->rect Jacobian (basis=I), packed as the 36-block.
        X, Y, Z = coords.cyl_to_rect(R, phi, z, xp=numpy)
        vX, vY, vZ = coords.cyl_to_rect_vec(vR, vT, vz, phi)
        yo_rect = numpy.array([X, Y, Z, vX, vY, vZ])
        jac0 = coords.cyl_to_rect_jac(R, vR, vT, z, vz, phi)  # (6,6)
        dyo_block = jac0.T.reshape(-1)  # column k at offset 6k
        out, _err = integrateFullOrbit_stm_c(
            pot, yo_rect, dyo_block, ts, int_method, rtol=rtol, atol=atol
        )  # (nt, 42)
        # base rect -> cyl (Orbit order); copy Z/vz before any reuse (views).
        Rout, phiout, Zout = coords.rect_to_cyl(
            out[:, 0], out[:, 1], out[:, 2], xp=numpy
        )
        vRout, vTout, vzout = coords.rect_to_cyl_vec(
            out[:, 3], out[:, 4], out[:, 5], out[:, 0], out[:, 1], out[:, 2], xp=numpy
        )
        Zout = numpy.copy(Zout)
        vzout = numpy.copy(vzout)
        base = numpy.empty((nt, 6))
        base[:, 0], base[:, 1], base[:, 2] = Rout, vRout, vTout
        base[:, 3], base[:, 4], base[:, 5] = Zout, vzout, phiout
        # STM: fold each rect deviation column back to cyl,
        # M_cyl[t] = jac_t^{-1} . rect_cols[t]  (column b = cyl deviation of e_b).
        dev = out[:, 6:].reshape(nt, 6, 6)  # [t, k, :] = rect dev column k
        M = numpy.empty((nt, 6, 6))
        for it in range(nt):
            jac_t = coords.cyl_to_rect_jac(
                Rout[it], vRout[it], vTout[it], Zout[it], vzout[it], phiout[it]
            )
            M[it] = numpy.linalg.solve(jac_t, dev[it].T)  # (6,6), column b = cyl dev
        M[0] = eye6  # exact identity at ts[0] (matches the per-column reference)
        xts.append(base)
        Ms.append(M)
    xt = numpy.asarray(xts)
    M = numpy.asarray(Ms)
    if single:
        return xt[0], M[0]
    return xt, M


def _c_stm_forward_loop(pot, vxvv, ts, method, rtol, atol):
    """Reference STM forward: propagate the six canonical basis deviations with
    SIX separate ``integrate_dxdv`` solves (re-integrating the base each time), in
    cyl Orbit order (``rectIn=False, rectOut=False`` -- no cyl<->rect folding here).
    Kept as the bit-identity reference for the augmented ``c_stm_forward``. Same
    signature and return contract.
    """
    from ...orbit import Orbit

    vxvv = numpy.asarray(vxvv, dtype=numpy.float64)
    single = vxvv.ndim == 1
    ics = vxvv[None] if single else vxvv
    basis = numpy.eye(6)
    xts, Ms = [], []
    for ic in ics:
        cols = []
        base = None
        for i in range(6):
            o = Orbit(list(ic))
            o.integrate_dxdv(
                basis[i],
                ts,
                pot,
                method=method,
                rectIn=False,
                rectOut=False,
                rtol=rtol,
                atol=atol,
            )
            cols.append(o.getOrbit_dxdv())  # (nt,6): column i of M
            if base is None:
                base = o.getOrbit()  # (nt,6): the base orbit, Orbit order
        xts.append(base)
        Ms.append(numpy.asarray(cols).transpose(1, 2, 0))  # (nt,6,6)
    xt = numpy.asarray(xts)
    M = numpy.asarray(Ms)
    if single:
        return xt[0], M[0]
    return xt, M


def integrate_stm(pot, vxvv, ts, *, method="dop853_c", rtol=1e-10, atol=1e-10):
    """Differentiable fast-C orbit integration; dispatches on ``vxvv``'s backend.

    Parameters
    ----------
    pot : Potential (or list).
    vxvv : backend array, (6,) or (N, 6), Orbit order [R,vR,vT,z,vz,phi]. Its
        namespace (jax/torch) selects the autodiff wrapper.
    ts : array of output times (numpy or backend; ts[0] is the initial time).
    method : one of rk4_c / rk6_c / dopr54_c / dop853_c.
    rtol, atol : C integrator tolerances.

    Returns
    -------
    backend array, (nt, 6) or (N, nt, 6), the orbit in Orbit order, differentiable
    w.r.t. ``vxvv``.
    """
    if method not in _C_DXDV_METHODS:
        raise ValueError(
            f"integrate_stm needs a dxdv-capable C integrator {_C_DXDV_METHODS}, got {method!r}"
        )
    xp = get_namespace(vxvv)
    name = getattr(xp, "__name__", "")
    if "jax" in name:
        from .._jax.orbit_stm import integrate as _integrate_jax

        return _integrate_jax(pot, vxvv, ts, method=method, rtol=rtol, atol=atol)
    if "torch" in name:
        from .._torch.orbit_stm import integrate as _integrate_torch

        return _integrate_torch(pot, vxvv, ts, method=method, rtol=rtol, atol=atol)
    raise NotImplementedError(
        "C-STM autodiff requires a jax or torch input array; for numpy use "
        "Orbit.integrate (the same C integrator, non-differentiable)."
    )
