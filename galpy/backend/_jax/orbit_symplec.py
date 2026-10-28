###############################################################################
#   galpy.backend._jax.orbit_symplec: jax constant-step fourth- and sixth-order
#   symplectic integrators (galpy's symplec4/symplec6) for forces that derive
#   from a potential.
#
#   The jax-specific half of galpy.backend._reference.integrate_orbit_symplectic
#   for order 4 and 6 (order 2 is orbit_leapfrog). A step is the drift/kick
#   composition of _reference.symplectic_coeffs, with three (order 4) or seven
#   (order 6) force evaluations. The steps run as jax.lax.scan's over blocks
#   that are each under jax.checkpoint, and every force evaluation is under
#   jax.checkpoint too, so reverse mode stores the state at the start of every
#   block plus the states of one block's steps, not every step's intermediates
#   (the same scheme as the leapfrog).
###############################################################################
from functools import partial

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
    import jax
    import jax.numpy as jnp

    from .._reference.inbackend_ode import _eom_rhs

    cs, ds = COEFFICIENTS[order]
    t0 = ts[0]
    dt = (ts[-1] - ts[0]) / nsteps

    def join(q, p):  # positions, velocities -> interleaved EOM state
        return jnp.reshape(jnp.stack([q, p], axis=-1), q.shape[:-1] + (dim,))

    # rematerialized in reverse mode, so a step stores only its state, not the
    # forces' intermediates
    @partial(jax.checkpoint, prevent_cse=False)
    def accel(q, p, t):
        a = jnp.stack(_eom_rhs(join(q, p), pot, t, jnp, dim)[1::2], axis=-1)
        return a.astype(q.dtype)  # a float64 parameter must not promote the state

    def step(carry, _):
        q, p, k = carry
        tk = t0 + k * dt
        tau = 0.0  # the fraction of the step drifted so far: the kicks' time
        for c, d in zip(cs, ds):
            q = q + c * dt * p
            tau = tau + c
            p = p + d * dt * accel(q, p, tk + tau * dt)
        q = q + cs[-1] * dt * p
        return (q, p, k + 1), None

    def run(carry, n):
        return jax.lax.scan(step, carry, None, length=n)[0]

    def save(carry):
        return join(carry[0], carry[1])

    q, p = y0[..., 0::2], y0[..., 1::2]
    carry = (q, p, jnp.asarray(0, dtype=jnp.int32))
    nper = nsteps // nblock  # steps per block
    nout = nsteps // nint  # steps per output interval
    if nper % nout == 0:  # whole output intervals in each block

        def block(carry):
            def interval(carry, _):
                carry = run(carry, nout)
                return carry, save(carry)

            return jax.lax.scan(interval, carry, None, length=nper // nout)

        block = jax.checkpoint(block, prevent_cse=False)
        _, ys = jax.lax.scan(lambda c, _: block(c), carry, None, length=nblock)
        ys = jnp.reshape(ys, (nint,) + ys.shape[2:])
    else:  # whole blocks in each output interval
        block = jax.checkpoint(lambda c: run(c, nper), prevent_cse=False)

        def interval(carry, _):
            carry = jax.lax.scan(
                lambda c, _: (block(c), None), carry, None, length=nout // nper
            )[0]
            return carry, save(carry)

        _, ys = jax.lax.scan(interval, carry, None, length=nint)
    return jnp.concatenate([y0[None], ys], axis=0)
