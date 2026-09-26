###############################################################################
#   galpy.backend._jax.king_ode: the scale-free King model's Poisson equation,
#   solved with diffrax so that W0 can be TRACED (jax.jit).
#
#   Mirrors galpy.df.kingdf._scalefreekingdf.solve step for step: segment 1 in
#   r from 0 to rbreak (saved on a linspace), segment 2 in Psi from W(rbreak)
#   down to 0 (saved on a linspace in Psi). rbreak, the second segment's start
#   and both output grids depend on W0; diffrax takes traced endpoints and save
#   points, and AD through the solve gives d/dW0 of the discretized solution.
###############################################################################
import math

_FOURPI = 4.0 * math.pi


def solve(dens_W, W0, npt, rtol=1e-10, atol=1e-12, max_steps=4096):
    """(rho0, r0, r, W, dWdr) of the scale-free King model, as jax arrays.

    ``dens_W`` is the King density as a function of W (backend-agnostic).
    DirectAdjoint (bounded ``max_steps``): diffrax's default adjoint is
    reverse-mode first order only, and d2/dW0^2 is wanted too."""
    import diffrax
    import jax.numpy as jnp

    rho0 = dens_W(W0)
    r0 = jnp.sqrt(9.0 / 4.0 / math.pi / rho0)
    rbreak = jnp.where(W0 < 2.0, r0 / 100.0, r0)
    n1 = npt // 2
    controller = diffrax.PIDController(rtol=rtol, atol=atol)
    adjoint = diffrax.DirectAdjoint()

    def rhs1(t, y, args):
        # r^2 W'' = -4 pi rho r^2 - 2 r W': the 2 W'/r term is 0 at r=0 (W'(0)=0);
        # a benign radius there keeps its backward finite
        tsafe = jnp.where(t > 0.0, t, 1.0)
        return jnp.stack(
            [
                y[1],
                -_FOURPI * dens_W(y[0]) - jnp.where(t > 0.0, 2.0 * y[1] / tsafe, 0.0),
            ]
        )

    r1 = jnp.linspace(0.0, 1.0, n1) * rbreak
    s1 = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs1),
        diffrax.Dopri8(),
        t0=0.0,
        t1=rbreak,
        dt0=None,
        y0=jnp.stack([W0 * 1.0, 0.0 * W0]),
        saveat=diffrax.SaveAt(ts=r1),
        stepsize_controller=controller,
        max_steps=max_steps,
        adjoint=adjoint,
    ).ys  # (n1, 2)

    def rhs2(t, y, args):  # in Psi: y = (r, v = dW/dr)
        return jnp.stack(
            [1.0 / y[1], -1.0 / y[1] * (_FOURPI * dens_W(t) + 2.0 * y[1] / y[0])]
        )

    Wb, vb = s1[-1, 0], s1[-1, 1]
    W2 = jnp.linspace(1.0, 0.0, npt - n1 + 1) * Wb
    s2 = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs2),
        diffrax.Dopri8(),
        t0=Wb,
        t1=0.0 * Wb,
        dt0=None,
        y0=jnp.stack([rbreak, vb]),
        saveat=diffrax.SaveAt(ts=W2),
        stepsize_controller=controller,
        max_steps=max_steps,
        adjoint=adjoint,
    ).ys  # (npt - n1 + 1, 2)
    r = jnp.concatenate([r1[:-1], s2[:, 0]])
    W = jnp.concatenate([s1[:-1, 0], W2])
    dWdr = jnp.concatenate([s1[:-1, 1], s2[:, 1]])
    return rho0, r0, r, W, dWdr
