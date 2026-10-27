###############################################################################
#   galpy.backend._torch.orbit_ode: torch (torchdiffeq / torchode) in-backend
#   orbit integration.
#
#   The torch-specific half of galpy.backend._reference.integrate_orbit. Integrates
#   the shared backend-agnostic EOM (_eom_rhs) with torchdiffeq (default) or
#   torchode (torch.compile-able). The jax
#   counterpart is galpy.backend._jax.orbit_ode.
###############################################################################


def integrate(pot, y0, ts, *, dim, rtol, atol, max_steps=None, solver=None):
    """Integrate the EOM with torchdiffeq. y0/ys in rectangular EOM variables
    [x, vx, y, vy, z, vz], shape (dim,) for one orbit or (N, dim) for a batch.

    ``ts`` shape (nt,): one shared output grid -> all N orbits in ONE solve.
    ``ts`` shape (N, nt): a PER-ORBIT grid -> integrate each orbit on its own grid
    (torchdiffeq has no vmap, so a per-orbit loop) and stack on the orbit axis.
    Either way returns ys (nt, N, dim) for a batch ((nt, dim) single).

    ``solver`` selects the torchdiffeq method (default ``dopri5``); ``max_steps``,
    if given, caps the adaptive solver's step count (torchdiffeq ``max_num_steps``).

    Uses ``dopri5``, NOT ``dopri8``: torchdiffeq's ``dopri8`` *backward* pass is
    noticeably less accurate (~1e-5 relative gradient error vs ~1e-8 for
    dopri5/rk4), because its adaptive step controller is detached on the backward.
    ``dopri5`` gives accurate gradients (matching jax/diffrax Dopri8) while still
    matching the C integrator forward to ~1e-7. (diffrax's Dopri8 backward is fine;
    this is torchdiffeq-specific.) torchdiffeq's plain ``odeint`` retains the graph,
    so SECOND derivatives (torch double-backward / hessian) work without an adjoint."""
    import torch
    from torchdiffeq import odeint

    from .._reference.inbackend_ode import _eom_rhs

    method = "dopri5" if solver is None else solver
    options = None if max_steps is None else {"max_num_steps": max_steps}

    def field(t, y):
        # stack on the trailing (component) axis so a batch (N, dim) state maps to
        # an (N, dim) derivative; for a single (dim,) state axis=-1 == axis=0.
        return torch.stack(_eom_rhs(y, pot, t, torch, dim), axis=-1)

    if ts.ndim > 1:  # per-orbit grids (N, nt): one solve per orbit, stack -> (nt,N,dim)
        return torch.stack(
            [
                odeint(
                    field,
                    y0[i],
                    ts[i],
                    method=method,
                    rtol=rtol,
                    atol=atol,
                    options=options,
                )
                for i in range(y0.shape[0])
            ],
            dim=1,
        )
    return odeint(field, y0, ts, method=method, rtol=rtol, atol=atol, options=options)


# torchode step methods by name (solver=None -> dopri5)
_TORCHODE_SOLVERS = ("dopri5", "tsit5")


def torchode_solve(adjoint, problem):
    """``adjoint.solve(problem)``, compilable on CUDA too.

    On CUDA torchode's solve loop records a ``torch.cuda.Event`` every step
    (torchode/adjoints.py, ``continue_iterating_done.record``) after mutating
    tensors that are graph inputs once its data-dependent branches have graph-
    broken (``y_eval[eval_at_start, 0] = ...``, ``stats_n_steps.add_``); dynamo
    refuses that with a RuntimeError. So on CUDA the solve FRAME runs eagerly
    (non-recursive disable) while the step, controller and RHS it calls are still
    compiled. CPU records no event: unchanged."""
    import torch

    solve = adjoint.solve
    if problem.y0.device.type == "cuda":
        solve = torch.compiler.disable(solve, recursive=False)
    return solve(problem)


def integrate_torchode(pot, y0, ts, *, dim, rtol, atol, max_steps=None, solver=None):
    """Integrate the EOM with torchode: same contract as :func:`integrate`.

    Unlike torchdiffeq, torchode is torch.compile-able (inductor) and steps every
    orbit of a batch with its own controller, so a shared (nt,) grid and a
    per-orbit (N, nt) grid are one batched solve alike. ``solver`` is 'dopri5'
    (default) or 'tsit5'; ``max_steps`` caps the adaptive step count. On CUDA,
    torch.compile compiles the steps but not torchode's solve loop (see
    :func:`torchode_solve`)."""
    import torch
    import torchode as to

    from .._reference.inbackend_ode import _eom_rhs

    method = "dopri5" if solver is None else solver.lower()
    if method not in _TORCHODE_SOLVERS:
        raise ValueError(
            f"torchode solver must be one of {_TORCHODE_SOLVERS}, not '{solver}'"
        )
    single = y0.ndim == 1
    yb = y0[None] if single else y0
    tb = ts.expand(yb.shape[0], ts.shape[-1]) if ts.ndim == 1 else ts
    # torchode asserts t_eval is on y0's device (torchdiffeq moves it itself):
    # numpy times reach here as CPU tensors even for a CUDA IC
    tb = tb.to(device=yb.device, dtype=yb.dtype)
    term = to.ODETerm(lambda t, y: torch.stack(_eom_rhs(y, pot, t, torch, dim), -1))
    step = (to.Dopri5 if method == "dopri5" else to.Tsit5)(term=term)
    controller = to.IntegralController(atol=atol, rtol=rtol, term=term)
    # torchode builds atol with torch.tensor(): float32 at torch's default dtype
    controller.atol = torch.tensor(atol, dtype=yb.dtype, device=yb.device)
    # torchode uses its 0-d-tensor rtol as torch.add(..., alpha=rtol), whose value
    # torch.compile bakes in UNGUARDED (and the FX graph cache serves across
    # processes): a later compile at another rtol would silently reuse the old
    # one. As a python float it is guarded and part of the cache key.
    del controller._buffers["rtol"]
    controller.rtol = float(rtol)
    sol = torchode_solve(
        to.AutoDiffAdjoint(step, controller, max_steps=max_steps),
        to.InitialValueProblem(y0=yb, t_eval=tb),
    )
    if bool((sol.status != to.Status.SUCCESS.value).any()):
        raise RuntimeError(
            "torchode integration failed (status "
            f"{sol.status.tolist()}); raise max_steps or loosen rtol/atol"
        )
    ys = sol.ys.transpose(0, 1)  # (N, nt, dim) -> (nt, N, dim)
    return ys[:, 0] if single else ys
