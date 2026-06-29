###############################################################################
#   galpy.backend._jax.orbit_ode: jax (diffrax) in-backend orbit integration.
#
#   The jax-specific half of galpy.backend._reference.integrate_orbit. Integrates
#   the shared backend-agnostic EOM (_eom_rhs) with diffrax. The torch counterpart
#   is galpy.backend._torch.orbit_ode.
###############################################################################


def _resolve_solver(diffrax, solver):
    """Map a solver name (or pass a diffrax solver object through) to a diffrax
    solver instance. ``None`` -> Dopri8 (the default 8th-order adaptive solver)."""
    if solver is None:
        return diffrax.Dopri8()
    if not isinstance(solver, str):
        return solver  # already a diffrax solver instance
    cls = {"dopri8": "Dopri8", "dopri5": "Dopri5", "tsit5": "Tsit5"}.get(
        solver.lower(), solver
    )
    try:
        return getattr(diffrax, cls)()
    except AttributeError:
        raise ValueError(f"unknown diffrax solver {solver!r}")


def _resolve_adjoint(diffrax, adjoint):
    """Map an adjoint name (or pass a diffrax adjoint object through) to a diffrax
    adjoint, or ``None`` to use diffrax's own default. 'recursive' ->
    RecursiveCheckpointAdjoint (diffrax's default; reverse-mode, FIRST order only).
    'direct' -> DirectAdjoint, which differentiates through the solver's internal
    operations and so supports forward-mode + higher-order autodiff (jax.hessian /
    nested jacrev) -- needed for SECOND derivatives, at the cost of a max_steps-long
    scan (keep max_steps small)."""
    if adjoint is None:
        return None  # let diffrax use its default (RecursiveCheckpointAdjoint)
    if not isinstance(adjoint, str):
        return adjoint  # already a diffrax adjoint instance
    key = adjoint.lower()
    if key in ("recursive", "recursivecheckpoint", "default"):
        return diffrax.RecursiveCheckpointAdjoint()
    if key == "direct":
        return diffrax.DirectAdjoint()
    raise ValueError(
        f"unknown diffrax adjoint {adjoint!r}; use 'recursive' (default, first "
        "order) or 'direct' (higher-order / hessian-capable)"
    )


def integrate(pot, y0, ts, *, dim, rtol, atol, max_steps, solver=None, adjoint=None):
    """Integrate the EOM with diffrax (Dopri8, adaptive). y0/ys in rectangular
    EOM variables [x, vx, y, vy, z, vz], shape (dim,) for one orbit or (N, dim) for
    a batch.

    ``ts`` shape (nt,): one shared output grid -> all N orbits in ONE solve (one
    shared adaptive controller). ``ts`` shape (N, nt): a PER-ORBIT grid -> jax.vmap
    over (y0, ts) so each orbit gets its own saveat/span and its own (independent)
    controller. Either way returns ys (nt, N, dim) for a batch ((nt, dim) single).

    ``solver`` selects the diffrax solver (name or instance; default Dopri8).
    ``adjoint`` selects the diffrax adjoint (name or instance; ``None`` -> diffrax's
    default RecursiveCheckpointAdjoint, reverse-mode FIRST order, so forward-mode
    jacfwd is unavailable -- use jacrev). Pass adjoint='direct' for SECOND
    derivatives (jax.hessian / nested jacrev); it scans ``max_steps`` steps, so keep
    ``max_steps`` modest."""
    import diffrax
    import jax
    import jax.numpy as jnp

    from .._reference.inbackend_ode import _eom_rhs

    # stack on the trailing (component) axis so a batch (N, dim) state maps to an
    # (N, dim) derivative; for a single (dim,) state axis=-1 == axis=0.
    term = diffrax.ODETerm(
        lambda t, y, args: jnp.stack(_eom_rhs(y, pot, t, jnp, dim), axis=-1)
    )
    # None -> 100000 (the integrator's effective default; diffrax's own default of
    # 4096 is too low for galpy's long ~10-Tdyn integrations).
    if max_steps is None:
        max_steps = 100000
    _solver = _resolve_solver(diffrax, solver)
    _adjoint = _resolve_adjoint(diffrax, adjoint)
    # only pass adjoint when explicitly chosen, so the default call is byte-for-byte
    # the prior diffeqsolve (diffrax's own RecursiveCheckpointAdjoint default).
    _extra = {} if _adjoint is None else {"adjoint": _adjoint}

    def _solve(y0i, tsi):
        return diffrax.diffeqsolve(
            term,
            _solver,
            t0=tsi[0],
            t1=tsi[-1],
            dt0=None,
            y0=y0i,
            saveat=diffrax.SaveAt(ts=tsi),
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            max_steps=max_steps,
            **_extra,
        ).ys

    if ts.ndim > 1:  # per-orbit grids (N, nt): map each (y0, ts) to its own solve
        return jax.vmap(_solve, in_axes=(0, 0), out_axes=1)(y0, ts)
    return _solve(y0, ts)  # shared (nt,) grid: native batch (or a single orbit)
