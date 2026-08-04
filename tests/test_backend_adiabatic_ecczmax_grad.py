###############################################################################
# test_backend_adiabatic_ecczmax_grad.py: c=True C-native Adiabatic
# EccZmaxRperiRap gradients (#131 PR-2c). For a jax/torch input, a c=True
# actionAngleAdiabatic object returns differentiable (e,zmax,rperi,rap) via the
# fused C-native (4,5) Jacobian d(e,zmax,rperi,rap)/d(R,vR,vT,z,vz): implicit
# differentiation of the 1D radial (rperi,Rap) and vertical (zmax) turning-point
# roots (d(tp)/dcoord = -F_coord/F_r), chained through rap=sqrt(Rap^2+zmax^2) and
# ecc=(rap-rperi)/(rap+rperi). The gamma*Jz coupling into the radial Lz reuses the
# vertical action derivative (Lz = |R vT| + gamma*Jz, gamma=1 default), so the
# radial outputs' (z,vz) grads are nonzero for gamma!=0. First-order only. numpy
# path byte-identical. Mirrors test_backend_staeckel_ecczmax_grad.py.
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

from backend_jit_helpers import assert_jit_matches_eager

from galpy.actionAngle import actionAngleAdiabatic
from galpy.potential import MiyamotoNagaiPotential

_MP = MiyamotoNagaiPotential(normalize=1.0, a=0.5, b=0.1)
_AA = actionAngleAdiabatic(pot=_MP, gamma=1.0, c=True)  # C-native path
_AA0 = actionAngleAdiabatic(pot=_MP, gamma=0.0, c=True)  # decoupled (no gamma*Jz)

# interior orbits for grad-vs-FD (retrograde vT<0 exercises the s=-1 sign of the
# Lz=|R vT|+gamma*Jz derivative).
_ORBITS = {
    "generic": (1.0, 0.2, 1.1, 0.1, 0.15),
    "eccentric": (1.2, 0.35, 0.85, 0.25, -0.2),
    "generic2": (0.8, -0.1, 1.2, 0.05, 0.1),
    "retrograde": (1.1, 0.2, -1.0, 0.1, 0.15),
}
# edge orbits: finiteness only (planar z=vz=0 -> zmax=0 vertical guard;
# radially circular vR=0,vT=vc -> rperi==Rap radial guard).
_EDGE_ORBITS = {
    "edge_vR0": (1.0, 0.0, 1.1, 0.1, 0.15),
    "edge_vz0": (1.0, 0.2, 1.1, 0.1, 0.0),
    "edge_z0": (1.0, 0.2, 1.1, 0.0, 0.15),  # z=0 but vz!=0 -> zmax>0
}
# unbound: turning-point solve fails (sentinel) -> C zeroes the Jacobian rows.
_DEGEN_ORBIT = (1.0, 3.0, 0.05, 0.0, 3.0)


def _np_ecczmax(aA, orbit):
    out = aA.EccZmaxRperiRap(*[numpy.array([c]) for c in orbit])
    return [float(numpy.asarray(out[o][0])) for o in range(4)]


_FD_CACHE = {}


def _fd_grad(aA, orbit, eps=1e-6):
    # central FD of the c=True numpy (e,zmax,rperi,rap) over the 5 coords.
    key = (id(aA), orbit)
    if key in _FD_CACHE:
        return _FD_CACHE[key]
    g = [[], [], [], []]
    for i in range(5):
        up, dn = list(orbit), list(orbit)
        up[i] += eps
        dn[i] -= eps
        fu = _np_ecczmax(aA, tuple(up))
        fd = _np_ecczmax(aA, tuple(dn))
        for o in range(4):
            g[o].append((fu[o] - fd[o]) / (2.0 * eps))
    _FD_CACHE[key] = g
    return g


def _backend_grads(aA, backend, orbit):
    if backend == "jax":
        args = [jnp.asarray([x]) for x in orbit]
        return [
            [
                float(g[0])
                for g in jax.grad(
                    lambda *a: jnp.sum(aA.EccZmaxRperiRap(*a)[o]),
                    argnums=(0, 1, 2, 3, 4),
                )(*args)
            ]
            for o in range(4)
        ]
    args = [torch.tensor([x], requires_grad=True) for x in orbit]
    out = aA.EccZmaxRperiRap(*args)
    return [
        [
            float(g[0])
            for g in torch.autograd.grad(out[o].sum(), args, retain_graph=True)
        ]
        for o in range(4)
    ]


@pytest.mark.parametrize("backend", BACKENDS)
def test_adiabatic_ecczmax_value_parity(backend):
    # c=True backend (e,zmax,rperi,rap) VALUES equal numpy c=True (same C path).
    arr = jnp.asarray if backend == "jax" else (lambda x: torch.tensor(x))
    for orbit in list(_ORBITS.values()) + list(_EDGE_ORBITS.values()):
        ref = _np_ecczmax(_AA, orbit)
        out = _AA.EccZmaxRperiRap(*[arr([x]) for x in orbit])
        got = [float(numpy.asarray(out[o][0])) for o in range(4)]
        numpy.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("orbit", list(_ORBITS))
def test_adiabatic_ecczmax_grad_vs_fd(backend, orbit):
    # d(e,zmax,rperi,rap)/d(R,vR,vT,z,vz) via c=True C-native backend AD vs numpy
    # central FD, on interior orbits. rtol 3e-3: the (R,vR,vT) columns and the
    # whole zmax row are exact implicit-diff, but the (z,vz) columns of the radial
    # outputs flow through the gamma*Jz coupling, whose dJz/dcoord is the order-10
    # calcdJzAdiabatic derivative integral -- it agrees with the order-10 forward
    # Jz used by the FD gold only to that quadrature-truncation floor (~1e-4). The
    # gamma=0 decoupled test below (no dJz) holds at 1e-4. (Same floor as the
    # sibling action gradients; see test_backend_adiabatic_actions_grad.py.)
    coords = _ORBITS[orbit]
    fd = _fd_grad(_AA, coords)
    g = _backend_grads(_AA, backend, coords)
    for o in range(4):
        numpy.testing.assert_allclose(g[o], fd[o], rtol=3e-3, atol=2e-6)
        assert numpy.all(numpy.isfinite(g[o]))


@pytest.mark.parametrize("backend", BACKENDS)
def test_adiabatic_ecczmax_gamma0_decoupled(backend):
    # gamma=0: the vertical action is NOT injected into Lz, so the radial rperi
    # (index 2) is independent of (z,vz) -> its z/vz columns are exactly 0, and
    # all four grads still match FD.
    coords = _ORBITS["generic"]
    fd = _fd_grad(_AA0, coords)
    g = _backend_grads(_AA0, backend, coords)
    assert g[2][3] == 0.0 and g[2][4] == 0.0  # drperi/dz, drperi/dvz == 0
    for o in range(4):
        numpy.testing.assert_allclose(g[o], fd[o], rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("orbit", list(_EDGE_ORBITS))
def test_adiabatic_ecczmax_grad_finite_edges(backend, orbit):
    # planar / near-turning-point edge orbits: the grad must be FINITE (guarded,
    # not NaN/inf) -- the C zeros the degenerate radial/vertical rows.
    g = _backend_grads(_AA, backend, _EDGE_ORBITS[orbit])
    for o in range(4):
        assert numpy.all(numpy.isfinite(g[o])), f"non-finite grad row {o}"


@pytest.mark.parametrize("backend", BACKENDS)
def test_adiabatic_ecczmax_grad_degenerate(backend):
    # unbound orbit -> turning-point solve fails (-9999.99 sentinel) -> C zeroes
    # the Jacobian rows (grads EXACTLY 0), AND the forward VALUES still byte-match
    # the numpy c=True path (computed from the raw sentinels: rap->14142, ecc->5.83).
    arr = jnp.asarray if backend == "jax" else (lambda x: torch.tensor(x))
    ref = _np_ecczmax(_AA, _DEGEN_ORBIT)
    out = _AA.EccZmaxRperiRap(*[arr([x]) for x in _DEGEN_ORBIT])
    got = [float(numpy.asarray(out[o][0])) for o in range(4)]
    numpy.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)  # value parity
    g = _backend_grads(_AA, backend, _DEGEN_ORBIT)
    for o in range(4):
        numpy.testing.assert_array_equal(g[o], [0.0] * 5)  # sentinel -> zeroed rows


@pytest.mark.parametrize("backend", BACKENDS)
def test_adiabatic_ecczmax_radial_grad_finite(backend):
    # a radially plunging orbit (vT=0 -> Lz=0 -> rperi==0 EXACTLY, a bound orbit,
    # NOT the -9999.99 sentinel) exercises the C plunging guard: the drperi
    # implicit-diff would be 0/0 without it. Grad must be FINITE (not NaN).
    radial = (1.0, 0.3, 0.0, 0.0, 0.0)
    g = _backend_grads(_AA, backend, radial)
    for o in range(4):
        assert numpy.all(numpy.isfinite(g[o])), f"non-finite grad row {o}"


@pytest.mark.parametrize("backend", BACKENDS)
def test_adiabatic_ecczmax_planar_grad_finite(backend):
    # a truly planar orbit (z=vz=0 -> zmax=0) exercises the C Jacobian's planar
    # guard (dzmax row zeroed, zforce(R,0)=0); grad must stay finite.
    planar = (1.0, 0.2, 1.1, 0.0, 0.0)
    g = _backend_grads(_AA, backend, planar)
    for o in range(4):
        assert numpy.all(numpy.isfinite(g[o])), f"non-finite grad row {o}"


@pytest.mark.parametrize("backend", BACKENDS)
def test_adiabatic_ecczmax_circular_grad_finite(backend):
    # a radially circular orbit (vR=0, vT=vc -> rperi==Rap) exercises the C
    # Jacobian's circular guard (drperi/dRap rows zeroed); grad must stay finite.
    from galpy.potential import vcirc

    vc = float(vcirc(_MP, 1.0, use_physical=False))
    circular = (1.0, 0.0, vc, 0.0, 0.0)
    g = _backend_grads(_AA, backend, circular)
    for o in range(4):
        assert numpy.all(numpy.isfinite(g[o])), f"non-finite grad row {o}"


def test_adiabatic_ecczmax_grad_jit():
    # jax jit survival of the C-native custom_vjp EccZmax gradient.
    if "jax" not in BACKENDS:  # pragma: no cover
        pytest.skip("jax not installed")
    coords = _ORBITS["generic"]
    fn = lambda *a: jnp.sum(_AA.EccZmaxRperiRap(*a)[0])  # noqa: E731
    args = [jnp.asarray([x]) for x in coords]
    # "jit survival" means jit must not CHANGE the value, so compare against the
    # eager result -- a jitted gradient that is wrong but finite passed before.
    assert_jit_matches_eager(fn, *args, rtol=1e-12, atol=1e-14)
    assert_jit_matches_eager(jax.grad(fn, argnums=0), *args, rtol=1e-10, atol=1e-12)
