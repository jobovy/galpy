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
    sol = to.AutoDiffAdjoint(to.Dopri5(term=term), ctl, max_steps=max_steps).solve(
        to.InitialValueProblem(y0=y0[None], t_eval=t_eval[None])
    )
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

    def rhs1(t, y):
        # 2 W'/r is 0 at r=0 (W'(0)=0); a benign radius keeps its backward finite
        t = t[:, None]
        live = t > 0.0
        tsafe = torch.where(live, t, torch.ones_like(t))
        drag = torch.where(live, 2.0 * y[:, 1:2] / tsafe, torch.zeros_like(t))
        return torch.cat([y[:, 1:2], -_FOURPI * dens_W(y[:, 0:1]) - drag], -1)

    def rhs2(t, y):  # in Psi: y = (r, v = dW/dr)
        r, v = y[:, 0:1], y[:, 1:2]
        return torch.cat(
            [1.0 / v, -(_FOURPI * dens_W(t[:, None]) + 2.0 * v / r) / v], -1
        )

    r1 = torch.linspace(0.0, 1.0, n1, dtype=W0.dtype) * rbreak
    s1 = _solve_segment(
        rhs1, torch.stack([W0 * 1.0, 0.0 * W0]), r1, rtol, atol, max_steps
    )
    Wb, vb = s1[-1, 0], s1[-1, 1]
    W2 = torch.linspace(1.0, 0.0, npt - n1 + 1, dtype=W0.dtype) * Wb
    s2 = _solve_segment(rhs2, torch.stack([rbreak, vb]), W2, rtol, atol, max_steps)
    r = torch.cat([r1[:-1], s2[:, 0]])
    W = torch.cat([s1[:-1, 0], W2])
    dWdr = torch.cat([s1[:-1, 1], s2[:, 1]])
    return rho0, r0, r, W, dWdr
