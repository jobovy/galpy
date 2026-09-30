###############################################################################
#   galpy.backend._torch.king_ode: the scale-free King model's Poisson equation,
#   solved with torchode so that W0 is differentiable to any order.
#
#   The torch twin of galpy.backend._jax.king_ode (same two segments: in r from
#   0 to rbreak on a linspace, then in Psi from W(rbreak) down to 0).
###############################################################################
import math

_FOURPI = 4.0 * math.pi


def _solve_segment(f, y0, t_eval, rtol, atol, max_steps):
    import torchode as to

    term = to.ODETerm(f)
    ctl = to.IntegralController(atol=atol, rtol=rtol, term=term)
    # a 0-d-tensor rtol is baked into torch.compile unguarded (see orbit_ode)
    del ctl._buffers["rtol"]
    ctl.rtol = float(rtol)
    # step sizes detached: see solve()
    sol = to.AutoDiffAdjoint(
        to.Dopri5(term=term),
        ctl,
        max_steps=max_steps,
        backprop_through_step_size_control=False,
    ).solve(to.InitialValueProblem(y0=y0[None], t_eval=t_eval[None]))
    if bool((sol.status != to.Status.SUCCESS.value).any()):
        raise RuntimeError(f"King ODE solve failed (torchode status {sol.status})")
    return sol.ys[0]


def solve(dens_W, W0, npt, rtol=1e-10, atol=1e-12, max_steps=100000):
    """(rho0, r0, r, W, dWdr) of the scale-free King model, as torch tensors.

    ``dens_W`` is the King density as a function of W (backend-agnostic)."""
    import torch

    rho0 = dens_W(W0)
    r0 = torch.sqrt(9.0 / 4.0 / math.pi / rho0)
    rbreak = torch.where(W0 < 2.0, r0 / 100.0, r0)
    n1 = npt // 2

    # Both segments run in a W0-independent time, s in [0, 1] and u in [1, 0]
    # (r = s rbreak, Psi = u Wb), with detached step sizes: no gradient reaches
    # torchode's step times, whose backward (a scatter-add over the batch)
    # inductor miscompiles on CPU.
    def rhs1(s, y):
        # 2 W'/r is 0 at r=0 (W'(0)=0); a benign radius keeps its backward finite
        r = s[:, None] * rbreak
        live = r > 0.0
        rsafe = torch.where(live, r, torch.ones_like(r))
        drag = torch.where(live, 2.0 * y[:, 1:2] / rsafe, torch.zeros_like(r))
        return rbreak * torch.cat([y[:, 1:2], -_FOURPI * dens_W(y[:, 0:1]) - drag], -1)

    def rhs2(u, y):  # in Psi: y = (r, v = dW/dr)
        r, v = y[:, 0:1], y[:, 1:2]
        return Wb * torch.cat(
            [1.0 / v, -(_FOURPI * dens_W(u[:, None] * Wb) + 2.0 * v / r) / v], -1
        )

    s1grid = torch.linspace(0.0, 1.0, n1, dtype=W0.dtype, device=W0.device)
    s1 = _solve_segment(
        rhs1, torch.stack([W0 * 1.0, 0.0 * W0]), s1grid, rtol, atol, max_steps
    )
    Wb, vb = s1[-1, 0], s1[-1, 1]
    u2 = torch.linspace(1.0, 0.0, npt - n1 + 1, dtype=W0.dtype, device=W0.device)
    s2 = _solve_segment(rhs2, torch.stack([rbreak, vb]), u2, rtol, atol, max_steps)
    r = torch.cat([s1grid[:-1] * rbreak, s2[:, 0]])
    W = torch.cat([s1[:-1, 0], u2 * Wb])
    dWdr = torch.cat([s1[:-1, 1], s2[:, 1]])
    return rho0, r0, r, W, dWdr
