###############################################################################
# test_backend_adiabatic_actions_grad.py: c=True C-native Adiabatic action
# gradients (#131 PR-2a). For a jax/torch input, a c=True actionAngleAdiabatic
# object returns differentiable (jr, jz) via the fused C-native (2,5) Jacobian
# d(jr,jz)/d(R,vR,vT,z,vz): analytic Leibniz derivative integrals for
# dJr/dE_radial, dJr/dLz, dJz/dEz, dJz/dR (theta-substituted GL), chained
# through the elementary d(E_radial,Lz,Ez)/dcoord blocks -- with the vertical
# action injected into the radial Lz (Lz -> |R vT| + gamma*Jz, gamma=1 default),
# so jr's (z,vz) grads are nonzero. First-order only. numpy path byte-identical.
# Mirrors test_backend_staeckel_grad.py. The grad-vs-FD floor is the order-10
# value quadrature (value plain-GL vs derivative theta-sub); the C Jacobian
# itself is accurate to ~2.5e-5 (verified vs high-order self-FD).
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

from galpy.actionAngle import actionAngleAdiabatic
from galpy.potential import MiyamotoNagaiPotential

_MP = MiyamotoNagaiPotential(normalize=1.0, a=0.5, b=0.1)
_AA = actionAngleAdiabatic(pot=_MP, gamma=1.0, c=True)  # C-native path
_AA0 = actionAngleAdiabatic(pot=_MP, gamma=0.0, c=True)  # decoupled (no gamma*Jz)

# interior orbits for grad-vs-FD.
_ORBITS = {
    "generic": (1.0, 0.2, 1.1, 0.1, 0.15),
    "eccentric": (1.2, 0.35, 0.85, 0.25, -0.2),
    "generic2": (0.8, -0.1, 1.2, 0.05, 0.1),
}
# edge orbits: finiteness only (planar z=vz=0 -> Jz=0; radially circular vR=0).
_EDGE_ORBITS = {
    "planar": (1.0, 0.2, 1.1, 0.0, 0.0),
    "edge_vR0": (1.0, 0.0, 1.1, 0.1, 0.15),
    "edge_vz0": (1.0, 0.2, 1.1, 0.1, 0.0),
    # radially circular (vR=0, vT=vc=1 for normalize=1, planar) -> rperi==rap and
    # zmax==0, exercising the C circular (dJr row) + planar (dJz row) zero-guards.
    "circular": (1.0, 0.0, 1.0, 0.0, 0.0),
}
# unbound: turning-point solve fails (sentinel) -> C zeroes the Jacobian rows.
_DEGEN_ORBIT = (1.0, 0.3, 1.1, 0.2, 1.8)


def _np_actions(aA, orbit):
    jr, lz, jz = aA(*[numpy.array([c]) for c in orbit])  # (Jr, Lz, Jz)
    return float(jr[0]), float(jz[0])


_FD_CACHE = {}


def _fd_grad(aA, orbit, eps=1e-6):
    key = (id(aA), orbit)
    if key in _FD_CACHE:
        return _FD_CACHE[key]
    g = [[], []]
    for i in range(5):
        up, dn = list(orbit), list(orbit)
        up[i] += eps
        dn[i] -= eps
        fu = _np_actions(aA, tuple(up))
        fd = _np_actions(aA, tuple(dn))
        for o in range(2):
            g[o].append((fu[o] - fd[o]) / (2.0 * eps))
    _FD_CACHE[key] = g
    return g


def _backend_grads(aA, backend, orbit):
    # rows: [jr, jz] (outputs 0 and 2 of the (Jr,Lz,Jz) tuple)
    if backend == "jax":
        args = [jnp.asarray([x]) for x in orbit]
        return [
            [
                float(g[0])
                for g in jax.grad(
                    lambda *a: jnp.sum(aA(*a)[o]), argnums=(0, 1, 2, 3, 4)
                )(*args)
            ]
            for o in (0, 2)
        ]
    args = [torch.tensor([x], requires_grad=True) for x in orbit]
    out = aA(*args)
    return [
        [
            float(g[0])
            for g in torch.autograd.grad(out[o].sum(), args, retain_graph=True)
        ]
        for o in (0, 2)
    ]


@pytest.mark.parametrize("backend", BACKENDS)
def test_adiabatic_actions_value_parity(backend):
    # c=True backend (jr,jz) VALUES equal numpy c=True (same C path).
    arr = jnp.asarray if backend == "jax" else (lambda x: torch.tensor(x))
    for orbit in list(_ORBITS.values()) + list(_EDGE_ORBITS.values()):
        ref = _np_actions(_AA, orbit)
        out = _AA(*[arr([x]) for x in orbit])
        got = (float(numpy.asarray(out[0][0])), float(numpy.asarray(out[2][0])))
        numpy.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("orbit", list(_ORBITS))
def test_adiabatic_actions_grad_vs_fd(backend, orbit):
    # d(jr,jz)/d(R,vR,vT,z,vz) via c=True C-native backend AD vs numpy central FD.
    # rtol 3e-3: the FD gold uses the order-10 value quadrature (plain-GL) while
    # the analytic C Jacobian uses the theta-substituted derivative quadrature;
    # they agree to the ~6e-4 order-10 truncation floor (the C Jacobian itself is
    # ~2.5e-5-accurate vs a high-order reference).
    coords = _ORBITS[orbit]
    fd = _fd_grad(_AA, coords)
    g = _backend_grads(_AA, backend, coords)
    for o in range(2):
        numpy.testing.assert_allclose(g[o], fd[o], rtol=3e-3, atol=2e-6)
        assert numpy.all(numpy.isfinite(g[o]))


@pytest.mark.parametrize("backend", BACKENDS)
def test_adiabatic_actions_gamma0_decoupled(backend):
    # gamma=0: the vertical action is NOT injected into Lz, so jr is independent
    # of (z,vz) -> the jr-row z/vz columns are exactly 0, and the grad matches FD.
    coords = _ORBITS["generic"]
    fd = _fd_grad(_AA0, coords)
    g = _backend_grads(_AA0, backend, coords)
    assert g[0][3] == 0.0 and g[0][4] == 0.0  # djr/dz, djr/dvz == 0
    for o in range(2):
        numpy.testing.assert_allclose(g[o], fd[o], rtol=3e-3, atol=2e-6)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("orbit", list(_EDGE_ORBITS))
def test_adiabatic_actions_grad_finite_edges(backend, orbit):
    # planar / near-circular edge orbits: grad must be FINITE (guarded, not
    # NaN/inf) -- the C zeros the degenerate rows (planar zmax~0 / circular).
    g = _backend_grads(_AA, backend, _EDGE_ORBITS[orbit])
    for o in range(2):
        assert numpy.all(numpy.isfinite(g[o])), f"non-finite grad row {o}"


@pytest.mark.parametrize("backend", BACKENDS)
def test_adiabatic_actions_grad_degenerate(backend):
    # unbound orbit -> turning-point solve fails -> C degenerate guard zeroes the
    # Jacobian rows. The grad must be FINITE (not NaN/inf).
    g = _backend_grads(_AA, backend, _DEGEN_ORBIT)
    for o in range(2):
        assert numpy.all(numpy.isfinite(g[o])), f"non-finite grad row {o}"


def test_adiabatic_actions_grad_jit():
    # jax jit survival of the C-native custom_vjp action gradient.
    if "jax" not in BACKENDS:  # pragma: no cover
        pytest.skip("jax not installed")
    coords = _ORBITS["generic"]
    fn = lambda *a: jnp.sum(_AA(*a)[0])  # noqa: E731
    f, g = jax.jit(fn), jax.jit(jax.grad(fn, argnums=0))
    args = [jnp.asarray([x]) for x in coords]
    # "jit survival" means jit must not CHANGE the value, so compare against the
    # eager result -- a jitted gradient that is wrong but finite passed before.
    numpy.testing.assert_allclose(
        float(f(*args)), float(fn(*args)), rtol=1e-12, atol=1e-14
    )
    numpy.testing.assert_allclose(
        numpy.asarray(g(*args)),
        numpy.asarray(jax.grad(fn, argnums=0)(*args)),
        rtol=1e-10,
        atol=1e-12,
    )
