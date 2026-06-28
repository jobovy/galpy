###############################################################################
#   galpy.backend._reference.inbackend_ode
#
#   Differentiable orbit integration *inside* the backend: galpy's equations of
#   motion are evaluated through the backend-agnostic force layer
#   (_evaluateRforces / _evaluatephitorques / _evaluatezforces -- the
#   underscored, decorator-free evaluators, as the Python integrators use) and
#   integrated by the backend's own ODE solver (diffrax for jax, torchdiffeq for
#   torch), so gradients of the orbit w.r.t. initial conditions AND potential
#   parameters fall out of autodiff -- no hand-coded variational equations.
#
#   This is the reference, higher-order-differentiable orbit path for the
#   jax/torch backends, and the independent cross-check for the fast C
#   state-transition-matrix path (Track B/D of the backend plan).
#
#   Convention: phase-space vectors use galpy's Orbit ordering
#   ``[R, vR, vT, z, vz, phi]``. Internally the EOM is integrated in
#   *rectangular* variables ``[x, vx, y, vy, z, vz]`` -- matching galpy's C
#   integrator (integrateFullOrbit.c) rather than the cylindrical Python _EOM --
#   which avoids the 1/R centrifugal term and the coordinate singularity at the
#   axis, and is naturally well-behaved for autodiff. The public input/output are
#   transformed to/from ``Orbit`` order so they match ``Orbit``.
###############################################################################
from .. import get_namespace


def _eom_rhs(y, pot, t, xp, dim=6):
    """Backend-agnostic rectangular EOM. ``dim`` selects the state layout:
    ``[x, vx]`` (1D linear, dim==2) or ``[x, vx, y, vy, z, vz]`` (3D, dim==6;
    planar orbits run as 3D with z=vz=0). Recovers (R, phi) from the Cartesian
    position, evaluates the (decorator-free) force layer -- which dispatches on the
    array type of its arguments so this runs and differentiates under any backend
    -- and rotates the planar force back to Cartesian. Returns the matching-length
    tuple of time derivatives.

    Components are indexed on the TRAILING axis (``y[..., k]``) so the same RHS
    runs for a single state ``(dim,)`` and for a batch ``(N, dim)`` (N orbits in
    one solve); ``dim`` is passed explicitly because ``len(y)`` is the batch size,
    not the phase-space dimension, once a leading axis is present.
    """
    # Imported lazily so importing this module never forces the potential import
    # graph at galpy import time.
    if dim == 2:  # 1D linear potential: state [x, vx]
        from ...potential.linearPotential import _evaluatelinearForces

        x, vx = y[..., 0], y[..., 1]
        return vx, _evaluatelinearForces(pot, x, t=t)

    from ...potential.Potential import (
        _evaluatephitorques,
        _evaluateRforces,
        _evaluatezforces,
    )

    x, vx, yy, vy, z, vz = (y[..., i] for i in range(6))
    R = xp.sqrt(x**2 + yy**2)
    phi = xp.arctan2(yy, x)
    cosphi, sinphi = x / R, yy / R
    # v=[vR, vT, vz]; only used by velocity-dependent/dissipative forces.
    vR = vx * cosphi + vy * sinphi
    vT = -vx * sinphi + vy * cosphi
    v = [vR, vT, vz]
    Rforce = _evaluateRforces(pot, R, z, phi=phi, t=t, v=v)
    phitorque = _evaluatephitorques(pot, R, z, phi=phi, t=t, v=v)
    ax = cosphi * Rforce - sinphi / R * phitorque
    ay = sinphi * Rforce + cosphi / R * phitorque
    az = _evaluatezforces(pot, R, z, phi=phi, t=t, v=v)
    return vx, ax, vy, ay, vz, az


def _to_eom(xp, vxvv):
    """Orbit-order vxvv -> rectangular EOM state, dispatched on the phase-space
    dimension ``vxvv.shape[-1]``: 2 -> [x,vx] (1D); 3 [R,vR,vT] / 4 [R,vR,vT,phi]
    (planar) and 5 [R,vR,vT,z,vz] / 6 [R,vR,vT,z,vz,phi] (3D) -> [x,vx,y,vy,z,vz],
    padding the absent components (z=vz=0 planar, phi=0 axisymmetric) with zero.

    Components are read off the TRAILING axis (``vxvv[..., i]``) and stacked on the
    trailing axis, so a single IC ``(phasedim,)`` -> ``(dim,)`` and a batch
    ``(N, phasedim)`` -> ``(N, dim)`` flow through the same code (the scalar unpack
    is the no-leading-axis special case)."""
    pd = vxvv.shape[-1]
    if pd == 2:  # 1D linear: x is already the Cartesian coordinate
        return xp.stack([vxvv[..., 0], vxvv[..., 1]], axis=-1)
    zero = vxvv[..., 0] * 0.0  # backend/dtype-matched zero for padded components
    if pd == 6:
        R, vR, vT, z, vz, phi = (vxvv[..., i] for i in range(6))
    elif pd == 5:  # 3D axisymmetric (no phi)
        R, vR, vT, z, vz = (vxvv[..., i] for i in range(5))
        phi = zero
    elif pd == 4:  # 2D planar (with phi)
        R, vR, vT, phi = (vxvv[..., i] for i in range(4))
        z = vz = zero
    elif pd == 3:  # 2D axisymmetric planar (no phi)
        R, vR, vT = (vxvv[..., i] for i in range(3))
        z = vz = phi = zero
    else:
        raise ValueError(f"unsupported phase-space dimension {pd}")
    cosphi, sinphi = xp.cos(phi), xp.sin(phi)
    return xp.stack(
        [
            R * cosphi,
            vR * cosphi - vT * sinphi,
            R * sinphi,
            vR * sinphi + vT * cosphi,
            z,
            vz,
        ],
        axis=-1,
    )


def _from_eom(xp, ys, phasedim):
    """Rectangular EOM output -> Orbit-order, keeping only the tracked components for
    the requested phasedim (drops z,vz for planar and phi for axisymmetric)."""
    if phasedim == 2:  # 1D: [x, vx]
        return xp.stack([ys[..., 0], ys[..., 1]], axis=-1)
    x, vx, yy, vy, z, vz = (ys[..., i] for i in range(6))
    R = xp.sqrt(x**2 + yy**2)
    phi = xp.arctan2(yy, x)  # in [-pi, pi], matching Orbit's wrapped convention
    cosphi, sinphi = x / R, yy / R
    vR = vx * cosphi + vy * sinphi
    vT = -vx * sinphi + vy * cosphi
    cols = {
        6: [R, vR, vT, z, vz, phi],
        5: [R, vR, vT, z, vz],
        4: [R, vR, vT, phi],
        3: [R, vR, vT],
    }[phasedim]
    return xp.stack(cols, axis=-1)


def integrate_orbit(pot, vxvv, ts, *, rtol=1e-12, atol=1e-12, max_steps=100000):
    """Differentiably integrate an orbit with the backend's ODE solver.

    Parameters
    ----------
    pot : Potential (or list) -- must be backend-migrated for the chosen backend;
        a 3D Potential for planar/3D orbits, a linearPotential for 1D.
    vxvv : backend array of shape (phasedim,) for one orbit, or (N, phasedim) for
        a batch of N orbits integrated in ONE solve, Orbit order -- phasedim one of
        [x,vx] (1D), [R,vR,vT] / [R,vR,vT,phi] (2D), [R,vR,vT,z,vz] /
        [R,vR,vT,z,vz,phi] (3D). Its namespace selects the integrator:
        jax -> diffrax, torch -> torchdiffeq.
    ts : backend array of output times (monotonic; ts[0] is the initial time).
        Shape ``(nt,)`` is ONE shared grid -> all N orbits in a single solve (one
        shared adaptive controller and ``max_steps`` budget, so a much stiffer
        orbit shrinks the step for all and can exhaust ``max_steps`` where it would
        not singly -- raise ``max_steps`` or split it off). Shape ``(N, nt)`` is a
        PER-ORBIT grid (each orbit its own times, same length -- the C integrators'
        indiv_t, used by streamspraydf): each orbit gets its own saveat/span and an
        independent controller (jax.vmap / a torch per-orbit loop).
    rtol, atol : solver tolerances (default 1e-12, matching galpy's C integrators).
    max_steps : diffrax step cap (jax only).

    Returns
    -------
    backend array, shape ``(nt, phasedim)`` for one orbit or ``(nt, N, phasedim)``
    for a batch (``nt`` is ``ts.shape[-1]``), the orbit(s) in Orbit order.

    Notes
    -----
    Differentiable w.r.t. ``vxvv`` (initial conditions) and, when the parameter is
    supplied as a backend array, w.r.t. the potential's parameters. ``phi`` is
    recovered from the Cartesian position, so it is wrapped to [-pi, pi] as in
    ``Orbit``. Planar orbits are integrated as 3D with z=vz=0 (z-force vanishes in
    the plane of a symmetric potential) and the z,vz outputs are dropped.
    """
    xp = get_namespace(vxvv)
    name = xp.__name__
    # phase-space dim on the trailing axis (vxvv is (phasedim,) for one orbit or
    # (N, phasedim) for a batch); dim is the rectangular EOM state size (2 or 6).
    phasedim = vxvv.shape[-1]
    dim = 2 if phasedim == 2 else 6
    y0 = _to_eom(xp, vxvv)
    # backend-specific integrators live in galpy.backend._jax / ._torch
    if "jax" in name:
        from .._jax.orbit_ode import integrate as _integrate_jax

        ys = _integrate_jax(
            pot, y0, ts, dim=dim, rtol=rtol, atol=atol, max_steps=max_steps
        )
    elif "torch" in name:
        from .._torch.orbit_ode import integrate as _integrate_torch

        ys = _integrate_torch(pot, y0, ts, dim=dim, rtol=rtol, atol=atol)
    else:  # numpy path uses galpy's C/scipy integrators instead
        raise NotImplementedError(
            "in-backend ODE integration requires a jax or torch input array; "
            "for numpy use Orbit.integrate (C / scipy integrators)"
        )
    return _from_eom(xp, ys, phasedim)
