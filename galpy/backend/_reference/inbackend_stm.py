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
    autodiff rules.

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
