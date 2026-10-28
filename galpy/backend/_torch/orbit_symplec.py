###############################################################################
#   galpy.backend._torch.orbit_symplec: torch constant-step fourth- and
#   sixth-order symplectic integrators (galpy's symplec4/symplec6) for forces
#   that derive from a potential.
#
#   The torch-specific half of galpy.backend._reference.integrate_orbit_symplectic
#   for order 4 and 6 (order 2 is orbit_leapfrog; the jax counterpart is
#   galpy.backend._jax.orbit_symplec). A step is the drift/kick composition of
#   _reference.symplectic_coeffs, with three (order 4) or seven (order 6) force
#   evaluations. When autograd records, every block of steps runs under
#   torch.utils.checkpoint, and so does every force evaluation, so backward
#   stores the state at the start of every block plus the states of one block's
#   steps, not every step's intermediates (not under torch.func transforms,
#   which do not support checkpointing).
###############################################################################
from .._reference.symplectic_coeffs import COEFFICIENTS


def integrate(pot, y0, ts, *, dim, nsteps, nblock, nint, order):
    """Integrate the EOM with the symplectic integrator of ``order`` (4 or 6)
    in ``nsteps`` constant steps from ``ts[0]`` to ``ts[-1]``. y0/ys in
    rectangular EOM variables [x, vx, y, vy, z, vz] ([x, vx] for dim=2), shape
    (dim,) for one orbit or (N, dim) for a batch. Returns the states at the
    ``nint + 1`` equally-spaced output times (ts[0] and every ``nsteps // nint``
    steps): (nint + 1, N, dim) for a batch ((nint + 1, dim) single).

    ``nblock`` (a divisor of ``nsteps`` commensurate with the ``nint`` output
    intervals, checked by the caller) is the number of checkpointed blocks."""
    import contextlib

    import torch
    from torch.utils.checkpoint import checkpoint

    from .._reference.inbackend_ode import _eom_rhs

    cs, ds = COEFFICIENTS[order]
    ts = torch.as_tensor(ts, dtype=y0.dtype, device=y0.device)
    t0 = ts[0]
    dt = (ts[-1] - ts[0]) / nsteps
    # torch.func transforms reject checkpoint's saved-tensor hooks: there every
    # step's intermediates are stored
    record = (
        torch.is_grad_enabled() and torch._C._functorch.peek_interpreter_stack() is None
    )
    # backward recomputes outside the caller's scope, but must see the same
    # default dtype (galpy's public methods set float64 for their internals)
    default = torch.get_default_dtype()

    @contextlib.contextmanager
    def forward_default_dtype():
        prev = torch.get_default_dtype()
        torch.set_default_dtype(default)
        try:
            yield
        finally:
            torch.set_default_dtype(prev)

    def remat(f, *args):  # recomputed in backward instead of stored
        if not record:
            return f(*args)
        return checkpoint(
            f,
            *args,
            use_reentrant=False,
            context_fn=lambda: (contextlib.nullcontext(), forward_default_dtype()),
        )

    def join(q, p):  # positions, velocities -> interleaved EOM state
        return torch.reshape(torch.stack([q, p], dim=-1), q.shape[:-1] + (dim,))

    def force(q, p, t):
        a = torch.stack(_eom_rhs(join(q, p), pot, t, torch, dim)[1::2], dim=-1)
        return a.to(q.dtype)  # a float64 parameter must not promote the state

    def accel(q, p, t):
        return remat(force, q, p, t)

    def block(q, p, k0):  # nper steps from step k0, and the output states
        out = []
        for k in range(k0, k0 + nper):
            tk = t0 + k * dt
            tau = 0.0  # the fraction of the step drifted so far: the kicks' time
            for c, d in zip(cs, ds):
                q = q + c * dt * p
                tau = tau + c
                p = p + d * dt * accel(q, p, tk + tau * dt)
            q = q + cs[-1] * dt * p
            if (k + 1) % nout == 0:
                out.append(join(q, p))
        return q, p, *out

    q, p = y0[..., 0::2], y0[..., 1::2]
    nper = nsteps // nblock  # steps per block
    nout = nsteps // nint  # steps per output interval
    ys = [y0]
    for k0 in range(0, nsteps, nper):
        q, p, *out = remat(block, q, p, k0)
        ys.extend(out)
    return torch.stack(ys, dim=0)
