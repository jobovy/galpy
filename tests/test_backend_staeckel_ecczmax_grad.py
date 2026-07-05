###############################################################################
# test_backend_staeckel_ecczmax_grad.py: c=True C-native Staeckel
# EccZmaxRperiRap gradients (#131). For a jax/torch input, a c=True Staeckel
# object now returns differentiable (e,zmax,rperi,rap) via the fused C-native
# (4,5) Jacobian: implicit-diff of the turning points umin/umax/vmin
# (A_tp = -S_P/S_u, reusing the action-Hessian dS helpers) chained through
# uv_to_Rz at fixed delta. First-order only. numpy path byte-identical. Mirrors
# test_backend_staeckel_grad.py (the action gradients).
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

from galpy.actionAngle import actionAngleStaeckel
from galpy.potential import MiyamotoNagaiPotential

_MP = MiyamotoNagaiPotential(normalize=1.0, a=0.5, b=0.0375)
_DELTA = 0.45
_AAS = actionAngleStaeckel(pot=_MP, delta=_DELTA, c=True)  # C-native path

# interior orbits for grad-vs-FD; the vR=0/vz=0/z=0 edges (near a confocal
# turning point where S(tp)->0 makes A_tp large) are finiteness-only (there the
# C guards the row to 0 per the turning-point-edge convention).
_ORBITS = {
    "generic": (1.0, 0.2, 1.1, 0.1, 0.15),
    "eccentric": (1.2, 0.35, 0.85, 0.25, -0.2),
    "generic2": (0.8, -0.1, 1.2, 0.05, 0.1),
}
_EDGE_ORBITS = {
    "edge_vR0": (1.0, 0.0, 1.1, 0.1, 0.15),
    "edge_vz0": (1.0, 0.2, 1.1, 0.1, 0.0),
    "edge_z0": (1.0, 0.2, 1.1, 0.0, 0.15),  # planar (vmin->pi/2)
}
# unbound: the turning-point solve fails (umin/umax=-9999.99), so the C zeroes
# the Jacobian rows (degenerate guard). Grad must be FINITE (not NaN/inf).
_DEGEN_ORBIT = (1.0, 3.0, 0.05, 0.0, 3.0)


def _np_ecczmax(orbit):
    out = _AAS.EccZmaxRperiRap(*[numpy.array([c]) for c in orbit])
    return [float(numpy.asarray(out[o][0])) for o in range(4)]


_FD_CACHE = {}


def _fd_grad(orbit, eps=1e-6):
    # central FD of the c=True numpy (e,zmax,rperi,rap) over the 5 coords.
    if orbit in _FD_CACHE:
        return _FD_CACHE[orbit]
    g = [[], [], [], []]
    for i in range(5):
        up = list(orbit)
        dn = list(orbit)
        up[i] += eps
        dn[i] -= eps
        fu = _np_ecczmax(tuple(up))
        fd = _np_ecczmax(tuple(dn))
        for o in range(4):
            g[o].append((fu[o] - fd[o]) / (2.0 * eps))
    _FD_CACHE[orbit] = g
    return g


def _backend_grads(backend, orbit):
    if backend == "jax":
        args = [jnp.asarray([x]) for x in orbit]
        return [
            [
                float(g[0])
                for g in jax.grad(
                    lambda *a: jnp.sum(_AAS.EccZmaxRperiRap(*a)[o]),
                    argnums=(0, 1, 2, 3, 4),
                )(*args)
            ]
            for o in range(4)
        ]
    args = [torch.tensor([x], requires_grad=True) for x in orbit]
    out = _AAS.EccZmaxRperiRap(*args)
    return [
        [
            float(g[0])
            for g in torch.autograd.grad(out[o].sum(), args, retain_graph=True)
        ]
        for o in range(4)
    ]


@pytest.mark.parametrize("backend", BACKENDS)
def test_staeckel_ecczmax_value_parity(backend):
    # c=True backend (e,zmax,rperi,rap) VALUES equal numpy c=True (same C path).
    arr = jnp.asarray if backend == "jax" else (lambda x: torch.tensor(x))
    for orbit in list(_ORBITS.values()) + list(_EDGE_ORBITS.values()):
        ref = _np_ecczmax(orbit)
        out = _AAS.EccZmaxRperiRap(*[arr([x]) for x in orbit])
        got = [float(numpy.asarray(out[o][0])) for o in range(4)]
        numpy.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("orbit", list(_ORBITS))
def test_staeckel_ecczmax_grad_vs_fd(backend, orbit):
    # d(e,zmax,rperi,rap)/d(R,vR,vT,z,vz) via c=True C-native backend AD vs numpy
    # central FD, on interior orbits. The C Jacobian is analytic implicit-diff.
    coords = _ORBITS[orbit]
    fd = _fd_grad(coords)
    g = _backend_grads(backend, coords)
    for o in range(4):
        numpy.testing.assert_allclose(g[o], fd[o], rtol=1e-4, atol=1e-6)
        assert numpy.all(numpy.isfinite(g[o]))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("orbit", list(_EDGE_ORBITS))
def test_staeckel_ecczmax_grad_finite_edges(backend, orbit):
    # near-turning-point edge orbits: the grad must be FINITE (guarded, not
    # NaN/inf) -- the C zeros the row where S(tp)->0 (circular/planar).
    g = _backend_grads(backend, _EDGE_ORBITS[orbit])
    for o in range(4):
        assert numpy.all(numpy.isfinite(g[o])), f"non-finite grad row {o}"


@pytest.mark.parametrize("backend", BACKENDS)
def test_staeckel_ecczmax_grad_degenerate(backend):
    # unbound orbit -> turning-point solve fails -> C degenerate guard zeroes the
    # Jacobian rows. The grad must be FINITE (not NaN/inf).
    g = _backend_grads(backend, _DEGEN_ORBIT)
    for o in range(4):
        assert numpy.all(numpy.isfinite(g[o])), f"non-finite grad row {o}"
