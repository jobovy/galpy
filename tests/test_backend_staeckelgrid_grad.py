###############################################################################
# test_backend_staeckelgrid_grad.py: accurate actionAngleStaeckelGrid ACTION
# gradients under jax/torch AD. The forward value stays the grid-INTERPOLATED
# one (numpy path byte-identical, backend no-grad forward unchanged), but AD
# through the interpolating splines differentiates the interpolation error too
# (~1e-2 relative off the true gradient), so the gradient is grafted from the
# direct t^2-substituted Staeckel donor (_staeckel_prep + _staeckel_t2_action
# with the grid's pot/delta) -- the same donor as actionAngleStaeckel's graft.
# Hence gradients are validated against FD of the DIRECT (non-grid) action,
# NOT FD of the grid value. First-order only.
###############################################################################
import numpy
import pytest

pytestmark = pytest.mark.backend_managed

BACKENDS = []
try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    BACKENDS.append("jax")
except ImportError:  # pragma: no cover
    jax = None
try:
    import torch

    torch.set_default_dtype(torch.float64)

    BACKENDS.append("torch")
except ImportError:  # pragma: no cover
    torch = None

from galpy.actionAngle import actionAngleStaeckel, actionAngleStaeckelGrid
from galpy.potential import MWPotential2014

_DELTA = 0.45
# ONE module-level grid: construction is the slow part (same small-grid params
# as tests/test_backend_actionAngle.py); ICs sit well inside the grid.
_aASG = actionAngleStaeckelGrid(
    pot=MWPotential2014, delta=_DELTA, nE=15, npsi=15, nLz=18
)
_aAS = actionAngleStaeckel(pot=MWPotential2014, delta=_DELTA, c=False)

# (R, vR, vT, z, vz): generic on-grid orbits plus the vz=0 / z=0 edge orbits
# that historically hide turning-point/AD bugs.
_ORBITS = {
    "generic": (1.1, 0.15, 0.9, 0.12, 0.13),
    "generic2": (0.8, -0.2, 0.6, -0.15, 0.1),
    "edge_vz0": (1.0, 0.2, 0.95, 0.1, 0.0),
    "edge_z0": (1.2, -0.1, 1.05, 0.0, 0.12),
}


def _pair(aa, coords):
    out = aa(*[numpy.array([c]) for c in coords])
    return float(out[0][0]), float(out[2][0])


_FD_CACHE = {}


def _fd_direct(orbit, eps=1e-5):
    # gold: central FD of the DIRECT (non-grid) c=False Staeckel actions.
    if orbit in _FD_CACHE:
        return _FD_CACHE[orbit]
    gjr, gjz = [], []
    for i in range(5):
        up, dn = list(orbit), list(orbit)
        up[i] += eps
        dn[i] -= eps
        jru, jzu = _pair(_aAS, up)
        jrd, jzd = _pair(_aAS, dn)
        gjr.append((jru - jrd) / (2 * eps))
        gjz.append((jzu - jzd) / (2 * eps))
    _FD_CACHE[orbit] = (gjr, gjz)
    return gjr, gjz


def _backend_grads(backend, orbit):
    if backend == "jax":
        args = [jnp.asarray([c]) for c in orbit]
        gjr = jax.grad(lambda *a: jnp.sum(_aASG(*a)[0]), argnums=(0, 1, 2, 3, 4))(*args)
        gjz = jax.grad(lambda *a: jnp.sum(_aASG(*a)[2]), argnums=(0, 1, 2, 3, 4))(*args)
        return [float(g[0]) for g in gjr], [float(g[0]) for g in gjz]
    args = [torch.tensor([c], requires_grad=True) for c in orbit]
    out = _aASG(*args)
    gjr = torch.autograd.grad(out[0].sum(), args, retain_graph=True)
    gjz = torch.autograd.grad(out[2].sum(), args)
    return [float(g[0]) for g in gjr], [float(g[0]) for g in gjz]


@pytest.mark.parametrize("backend", BACKENDS)
def test_staeckelgrid_grad_between_nodes_resolution_invariant(backend):
    # Between-grid-points guarantee: at a generic (off-node) query point, two
    # different grid resolutions give visibly DIFFERENT interpolated values
    # (proving the point sits between nodes, interpolation error present) but
    # IDENTICAL grafted gradients matching FD-of-the-direct-action -- the donor
    # computes the true action at the exact query point and never reads the grid.
    from galpy.actionAngle import actionAngleStaeckelGrid

    coarse = actionAngleStaeckelGrid(
        pot=MWPotential2014, delta=_DELTA, nE=11, npsi=11, nLz=12
    )
    coords = _ORBITS["generic"]
    fjr, _ = _fd_direct(coords)

    def val_and_djrdR(grid):
        if backend == "jax":
            args = [jnp.asarray([c]) for c in coords]
            val = float(grid(*args)[0][0])
            g = jax.grad(lambda R: jnp.sum(grid(R, *args[1:])[0]))(args[0])
            return val, float(g[0])
        args = [torch.tensor([c], requires_grad=True) for c in coords]
        out = grid(*args)[0]
        (g,) = torch.autograd.grad(out.sum(), (args[0],))
        return float(out.detach()[0]), float(g[0])

    v_fine, g_fine = val_and_djrdR(_aASG)
    v_coarse, g_coarse = val_and_djrdR(coarse)
    assert v_fine > 0.0 and v_coarse > 0.0  # unclamped (interp value positive)
    assert abs(v_fine - v_coarse) / v_fine > 1e-3  # values differ -> off-node
    numpy.testing.assert_allclose(g_coarse, g_fine, rtol=1e-12)  # grid-free grad
    numpy.testing.assert_allclose(g_fine, fjr[0], rtol=2e-3)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("orbit", list(_ORBITS))
def test_staeckelgrid_action_grad_vs_fd_direct(backend, orbit):
    # d(jr,jz)/d(R,vR,vT,z,vz) via backend AD through the grid vs numpy central
    # FD of the direct action: the graft replaces the spline-AD gradient (which
    # tracks the interpolation error, ~1e-2 off) by the exact-action donor's.
    coords = _ORBITS[orbit]
    fjr, fjz = _fd_direct(coords)
    gjr, gjz = _backend_grads(backend, coords)
    numpy.testing.assert_allclose(gjr, fjr, rtol=2e-3, atol=2e-6)
    numpy.testing.assert_allclose(gjz, fjz, rtol=2e-3, atol=2e-6)
    assert numpy.all(numpy.isfinite(gjr)) and numpy.all(numpy.isfinite(gjz))


@pytest.mark.parametrize("backend", BACKENDS)
def test_staeckelgrid_action_value_unchanged(backend):
    # the graft must not change the forward value: it stays the INTERPOLATED
    # grid value (== the numpy grid output at machine-precision parity), both
    # with and without an AD trace -- NOT the direct action the donor integrates.
    coords = _ORBITS["generic"]
    jr_np, jz_np = _pair(_aASG, coords)
    if backend == "jax":
        args = [jnp.asarray([c]) for c in coords]
        out = _aASG(*args)
        jr_fwd, jz_fwd = float(out[0][0]), float(out[2][0])
        jr_traced = float(
            jax.value_and_grad(lambda R: jnp.sum(_aASG(R, *args[1:])[0]))(args[0])[0]
        )
        jz_traced = float(
            jax.value_and_grad(lambda R: jnp.sum(_aASG(R, *args[1:])[2]))(args[0])[0]
        )
    else:
        args = [torch.tensor([c]) for c in coords]
        out = _aASG(*args)
        jr_fwd, jz_fwd = float(out[0][0]), float(out[2][0])
        gargs = [torch.tensor([c], requires_grad=True) for c in coords]
        gout = _aASG(*gargs)
        jr_traced, jz_traced = float(gout[0].detach()[0]), float(gout[2].detach()[0])
    numpy.testing.assert_allclose(jr_fwd, jr_np, rtol=1e-10, atol=1e-12)
    numpy.testing.assert_allclose(jz_fwd, jz_np, rtol=1e-10, atol=1e-12)
    # the donor terms cancel exactly in floating point
    numpy.testing.assert_allclose(jr_traced, jr_fwd, rtol=1e-14)
    numpy.testing.assert_allclose(jz_traced, jz_fwd, rtol=1e-14)
    # ... and the interpolated value is NOT the direct action (else the graft
    # would be vacuous here)
    jr_dir, jz_dir = _pair(_aAS, coords)
    assert (jr_fwd != jr_dir) or (jz_fwd != jz_dir)
