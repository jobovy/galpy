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

# dxdv-capable C integrators (the variational RHS is wired for these). The
# Runge-Kutta methods additionally support the fast AUGMENTED 3D STM (all six
# columns in one 42-state solve); the symplectic methods carry the deviation
# through the closed-form drift/kick tangent maps and are integrated per column.
_C_RK_METHODS = ("rk4_c", "rk6_c", "dopr54_c", "dop853_c")
_C_SYMPLEC_METHODS = ("leapfrog_c", "symplec4_c", "symplec6_c")
_C_DXDV_METHODS = _C_RK_METHODS + _C_SYMPLEC_METHODS


def _snap_equispaced(ts, method):
    """Symplectic integrators require EXACTLY equispaced times. An equispaced grid
    that passed through a lower-precision (e.g. float32) backend op arrives slightly
    non-equispaced -- ``integrate_dxdv`` then rejects it, though the numpy path
    (float64) never sees the drift. Snap an approximately-equispaced grid back to
    its intended ``linspace`` (a no-op for an exact grid, and only applied to the
    symplectic methods -- the Runge-Kutta ones accept arbitrary times). Genuinely
    non-equispaced input deviates far more than float rounding and is returned
    unchanged, so ``integrate_dxdv`` still rejects it exactly as numpy does.
    """
    if method.lower() not in _C_SYMPLEC_METHODS or ts.ndim != 1 or len(ts) < 3:
        return ts
    ts_lin = numpy.linspace(ts[0], ts[-1], len(ts))
    tol = 1e-6 * max(abs(float(ts[0])), abs(float(ts[-1])), 1.0)
    if numpy.max(numpy.abs(ts - ts_lin)) < tol:
        return ts_lin
    return ts


def c_stm_forward(pot, vxvv, ts, method, rtol, atol):
    """Run the C variational integrator and return (x(t), M(t)).

    Pure numpy; no autodiff. This is the host-side call wrapped by the jax/torch
    autodiff rules. Handles 3D (6D phase space), planar (4D) and 1D (2D) orbits and
    both the Runge-Kutta and symplectic dxdv-capable C integrators. A 3D orbit with
    a Runge-Kutta method uses the AUGMENTED integrator (``_c_stm_forward_augmented``
    -- base orbit + all six STM columns in ONE 42-state solve, ~4-6x faster than
    per-column); every other case (symplectic, planar, or 1D -- for which there is
    no augmented C STM) uses the per-column reference (``_c_stm_forward_loop``, d
    separate ``integrate_dxdv`` solves).

    Parameters
    ----------
    pot : Potential (or list)
    vxvv : numpy.ndarray, shape (d,) or (N, d) with d in {6, 4, 2}: Orbit order
        [R,vR,vT,z,vz,phi] (3D), [R,vR,vT,phi] (planar), or [x,vx] (1D).
    ts : numpy.ndarray, shape (nt,), output times (ts[0] is the initial time).
    method : str, one of ``_C_DXDV_METHODS``.
    rtol, atol : float

    Returns
    -------
    xt : numpy.ndarray, (nt, d) or (N, nt, d) -- the orbit, Orbit order.
    M : numpy.ndarray, (nt, d, d) or (N, nt, d, d) -- STM d x(t)/d x(0),
        M[...,0,:,:] = identity.
    """
    d = numpy.asarray(vxvv).shape[-1]
    if d == 6 and method.lower() in _C_RK_METHODS:
        return _c_stm_forward_augmented(pot, vxvv, ts, method, rtol, atol)
    # Symplectic 3D, or any planar/1D orbit: no augmented C STM -> per-column loop.
    return _c_stm_forward_loop(pot, vxvv, ts, method, rtol, atol)


def _c_stm_forward_augmented(pot, vxvv, ts, method, rtol, atol):
    """Augmented 3D STM forward: the base orbit plus all six STM columns in ONE
    42-state C solve (the force + 3D Hessian are evaluated once per step, not six
    times), ~4-6x faster than re-integrating the base per column. For the adaptive
    methods (dop853_c/dopr54_c) the joint 42-state error norm accepts a slightly
    different step sequence than six 12-state solves, so M matches the per-column
    reference (``_c_stm_forward_loop``) to ~1e-10, not bit-for-bit; the gradient is
    unchanged. 6D orbits + Runge-Kutta methods only. Same signature/return as
    ``c_stm_forward``.
    """
    from ...orbit.integrateFullOrbit import integrateFullOrbit_stm_c
    from ...util import coords

    vxvv = numpy.asarray(vxvv, dtype=numpy.float64)
    single = vxvv.ndim == 1
    ics = vxvv[None] if single else vxvv
    ts = numpy.asarray(ts, dtype=numpy.float64)
    # ts is either a shared (nt,) grid or a per-orbit (N, nt) grid.
    per_orbit_ts = ts.ndim == 2
    nt = ts.shape[-1]
    eye6 = numpy.eye(6)
    int_method = method.lower()
    xts, Ms = [], []
    for oi, ic in enumerate(ics):
        ts_o = ts[oi] if per_orbit_ts else ts
        R, vR, vT, z, vz, phi = ic
        # cyl -> rect base, and the six cyl basis deviations folded to rect =
        # the columns of the cyl->rect Jacobian (basis=I), packed as the 36-block.
        X, Y, Z = coords.cyl_to_rect(R, phi, z, xp=numpy)
        vX, vY, vZ = coords.cyl_to_rect_vec(vR, vT, vz, phi)
        yo_rect = numpy.array([X, Y, Z, vX, vY, vZ])
        jac0 = coords.cyl_to_rect_jac(R, vR, vT, z, vz, phi)  # (6,6)
        dyo_block = jac0.T.reshape(-1)  # column k at offset 6k
        out, _err = integrateFullOrbit_stm_c(
            pot, yo_rect, dyo_block, ts_o, int_method, rtol=rtol, atol=atol
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
    """Per-column STM forward for ANY phase-space dim (d in {6, 4, 2}) and any
    dxdv-capable C method (Runge-Kutta OR symplectic): propagate the d canonical
    basis deviations with d separate ``integrate_dxdv`` solves (re-integrating the
    base each time), in cyl/native Orbit order (``rectIn=False, rectOut=False`` --
    no cyl<->rect folding; the flags are ignored for 1D). This is the bit-identity
    reference for the augmented 3D path AND the actual forward for the symplectic,
    planar, and 1D cases (which have no augmented C STM). Same signature/return as
    ``c_stm_forward``. Note the STM is returned in the (non-canonical) Orbit frame,
    so ``det M != 1`` for d in {4, 6}; it is nonetheless the exact sensitivity
    ``d x(t)/d x(0)`` used for the IC gradient (validated by finite-difference of
    the flow, not by Liouville in the cyl frame).
    """
    from ...orbit import Orbit

    vxvv = numpy.asarray(vxvv, dtype=numpy.float64)
    single = vxvv.ndim == 1
    ics = vxvv[None] if single else vxvv
    ts = numpy.asarray(ts, dtype=numpy.float64)
    per_orbit_ts = ts.ndim == 2  # shared (nt,) vs per-orbit (N, nt) grid
    d = ics.shape[-1]
    basis = numpy.eye(d)
    xts, Ms = [], []
    for oi, ic in enumerate(ics):
        ts_o = _snap_equispaced(ts[oi] if per_orbit_ts else ts, method)
        cols = []
        base = None
        for i in range(d):
            o = Orbit(list(ic))
            o.integrate_dxdv(
                basis[i],
                ts_o,
                pot,
                method=method,
                rectIn=False,
                rectOut=False,
                rtol=rtol,
                atol=atol,
            )
            cols.append(o.getOrbit_dxdv())  # (nt,d): column i of M
            if base is None:
                base = numpy.array(o.getOrbit())  # (nt,d): base orbit, Orbit order
                if d in (4, 6):
                    # Orbit.integrate wraps phi (the last coord) to (-pi, pi];
                    # planar integrate_dxdv returns it in [0, 2pi) -> wrap to match
                    # the numpy path (M's delta-phi is a branch-independent deriv).
                    base[:, -1] = (
                        numpy.mod(base[:, -1] + numpy.pi, 2.0 * numpy.pi) - numpy.pi
                    )
        xts.append(base)
        Ms.append(numpy.asarray(cols).transpose(1, 2, 0))  # (nt,d,d)
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
    vxvv : backend array, (d,) or (N, d) with d in {6, 4, 2}: Orbit order
        [R,vR,vT,z,vz,phi] (3D), [R,vR,vT,phi] (planar), or [x,vx] (1D). Its
        namespace (jax/torch) selects the autodiff wrapper.
    ts : array of output times (numpy or backend; ts[0] is the initial time).
    method : one of the dxdv-capable C integrators -- Runge-Kutta
        (rk4_c / rk6_c / dopr54_c / dop853_c) or symplectic
        (leapfrog_c / symplec4_c / symplec6_c).
    rtol, atol : C integrator tolerances.

    Returns
    -------
    backend array, (nt, d) or (N, nt, d), the orbit in Orbit order, differentiable
    w.r.t. ``vxvv``.
    """
    if method.lower() not in _C_DXDV_METHODS:
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
