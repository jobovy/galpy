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


def _carries_jvp_tangent(*xs):
    """True if one of ``xs`` carries a forward-mode (jax.jvp/jacfwd) tangent.

    Walks each tracer's chain (``.primal`` / ``.val``) by class name -- private
    API, so it can only ever MISS a tangent (e.g. jvp of a jitted function,
    whose inner trace has none), never invent one. Off when reverse mode also
    runs on JVP tracers (``jax_use_direct_linearize`` off, older jax), so a
    jax.grad never switches adjoints."""
    import jax

    if not getattr(jax.config, "jax_use_direct_linearize", False):
        return False
    for x in xs:
        while isinstance(x, jax.core.Tracer):
            if type(x).__name__ == "JVPTracer":
                return True
            x = getattr(x, "primal", getattr(x, "val", None))
    return False


def integrate(
    pot, y0, ts, *, dim, rtol, atol, max_steps, solver=None, adjoint=None, nsteps=None
):
    """Integrate the EOM with diffrax (Dopri8, adaptive). y0/ys in rectangular
    EOM variables [x, vx, y, vy, z, vz], shape (dim,) for one orbit or (N, dim) for
    a batch.

    ``ts`` shape (nt,): one shared output grid -> all N orbits in ONE solve (one
    shared adaptive controller). ``ts`` shape (N, nt): a PER-ORBIT grid -> jax.vmap
    over (y0, ts) so each orbit gets its own saveat/span and its own (independent)
    controller. Either way returns ys (nt, N, dim) for a batch ((nt, dim) single).

    ``solver`` selects the diffrax solver (name or instance; default Dopri8).
    ``adjoint`` selects the diffrax adjoint (name or instance; ``None`` -> diffrax's
    default RecursiveCheckpointAdjoint, reverse-mode FIRST order; but
    DirectAdjoint when a forward-mode tangent reaches the solve, i.e. under
    jax.jvp/jacfwd/hessian -- except jvp OF a jax.jit-ed function, whose tangent
    is invisible here: pass adjoint='direct' there). Pass adjoint='direct' for
    SECOND derivatives (jax.hessian / nested jacrev); it scans ``max_steps``
    steps, so keep ``max_steps`` modest."""
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
    if nsteps is not None:
        # constant steps take exactly nsteps; a larger cap only lengthens the
        # (DirectAdjoint) bounded loop (~30% of a vmapped second-order solve)
        max_steps = min(max_steps, nsteps + 1)
    _solver = _resolve_solver(diffrax, solver)
    _adjoint = _resolve_adjoint(diffrax, adjoint)
    # the default adjoint is a custom_vjp, which forward mode cannot go through;
    # a tangent can enter via y0, ts or a potential parameter, so probe the EOM
    # at the initial state (unused: dead code under jit)
    if _adjoint is None and _carries_jvp_tangent(
        y0, ts, term.vf(jnp.reshape(ts, (-1,))[0], y0, None)
    ):
        _adjoint = diffrax.DirectAdjoint()
    # only pass adjoint when explicitly chosen, so the default call is byte-for-byte
    # the prior diffeqsolve (diffrax's own RecursiveCheckpointAdjoint default).
    _extra = {} if _adjoint is None else {"adjoint": _adjoint}

    def _solve(y0i, tsi):
        try:
            return _diffeqsolve(y0i, tsi)
        except TypeError as e:  # a tangent the probe above could not see
            if "custom_vjp" in str(e):
                raise TypeError(
                    f"{e} Forward-mode differentiation (jax.jvp/jacfwd) of an "
                    "orbit integration needs adjoint='direct': pass "
                    "inbackend_kwargs={'adjoint': 'direct'} to Orbit.integrate "
                    "(or integrate_kwargs={'adjoint': 'direct'} to a stream DF)."
                ) from e
            raise

    def _diffeqsolve(y0i, tsi):
        return diffrax.diffeqsolve(
            term,
            _solver,
            t0=tsi[0],
            t1=tsi[-1],
            dt0=None if nsteps is None else (tsi[-1] - tsi[0]) / nsteps,
            y0=y0i,
            saveat=diffrax.SaveAt(ts=tsi),
            stepsize_controller=(
                diffrax.PIDController(rtol=rtol, atol=atol)
                if nsteps is None
                else diffrax.ConstantStepSize()
            ),
            max_steps=max_steps,
            **_extra,
        ).ys

    if ts.ndim > 1:  # per-orbit grids (N, nt): map each (y0, ts) to its own solve
        return jax.vmap(_solve, in_axes=(0, 0), out_axes=1)(y0, ts)
    return _solve(y0, ts)  # shared (nt,) grid: native batch (or a single orbit)
