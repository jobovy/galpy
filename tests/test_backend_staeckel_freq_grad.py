###############################################################################
# test_backend_staeckel_freq_grad.py: c=True C-native Staeckel FREQUENCY
# gradients (#131). actionsFreqs with a jax/torch input now returns
# differentiable Omega{r,phi,z} via the fused (5x5) C Jacobian: the action rows
# are #1051's actionsJac, the Omega rows compose the analytic action Hessians
# (calcd2JR/Jz, theta-map) through the quotient-rule partials of
# calcFreqsFromDerivsStaeckel and the dP/dcoord chain. First-order only. numpy
# path byte-identical. Mirrors test_backend_staeckel_grad.py (the action grads).
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
_DELTA, _ORDER = 0.45, 10
_AAS = actionAngleStaeckel(pot=_MP, delta=_DELTA, c=True)
_ORBITS = {
    "generic": (1.0, 0.2, 1.1, 0.1, 0.15),
    "eccentric": (1.2, 0.35, 0.85, 0.25, -0.2),
    "edge_vR0": (1.0, 0.0, 1.1, 0.1, 0.15),
    "edge_vz0": (1.0, 0.2, 1.1, 0.1, 0.0),
    "edge_z0": (1.0, 0.2, 1.1, 0.0, 0.15),
}


def _np_freqs(*orbit):
    out = _AAS.actionsFreqs(*[numpy.array([c]) for c in orbit])
    return float(out[3][0]), float(out[4][0]), float(out[5][0])


_FD_CACHE = {}


def _fd_freq_grad(orbit, eps=1e-5):
    # central FD of the c=True numpy frequencies -- the gold reference the
    # c=True backend AD (same C Jacobian) must reproduce.
    if orbit in _FD_CACHE:
        return _FD_CACHE[orbit]
    g = [[], [], []]  # dOr, dOp, dOz over the 5 coords
    for i in range(5):
        up = list(orbit)
        dn = list(orbit)
        up[i] += eps
        dn[i] -= eps
        fu = _np_freqs(*up)
        fd = _np_freqs(*dn)
        for k in range(3):
            g[k].append((fu[k] - fd[k]) / (2.0 * eps))
    _FD_CACHE[orbit] = g
    return g


def _backend_freq_grads(backend, orbit, useu0=False):
    aAS = actionAngleStaeckel(pot=_MP, delta=_DELTA, c=True, useu0=useu0)
    if backend == "jax":
        args = [jnp.asarray([x]) for x in orbit]
        return [
            [
                float(g[0])
                for g in jax.grad(
                    lambda *a: jnp.sum(aAS.actionsFreqs(*a)[3 + k]),
                    argnums=(0, 1, 2, 3, 4),
                )(*args)
            ]
            for k in range(3)
        ]
    args = [torch.tensor([x], requires_grad=True) for x in orbit]
    out = aAS.actionsFreqs(*args)
    return [
        [
            float(g[0])
            for g in torch.autograd.grad(out[3 + k].sum(), args, retain_graph=True)
        ]
        for k in range(3)
    ]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("orbit", list(_ORBITS))
def test_staeckel_freq_grad_vs_fd(backend, orbit):
    # d(Omega{r,phi,z})/d(R,vR,vT,z,vz) via c=True backend AD vs numpy central FD.
    # The residual is the theta-GL-vs-fixed_quad quadrature offset; the pre-#131
    # state was NO gradient (Omega was stop_gradient'd). dOmega_r/dR ~ -1.93.
    coords = _ORBITS[orbit]
    fd = _fd_freq_grad(coords)
    g = _backend_freq_grads(backend, coords)
    for k in range(3):
        numpy.testing.assert_allclose(g[k], fd[k], rtol=3e-3, atol=3e-6)
        assert numpy.all(numpy.isfinite(g[k]))


@pytest.mark.parametrize("backend", BACKENDS)
def test_staeckel_freq_value_c_true_matches_numpy(backend):
    # the c=True backend Omega VALUES equal the numpy c=True values (same C path)
    arr = jnp.asarray if backend == "jax" else (lambda x: torch.tensor(x))
    for orbit in _ORBITS.values():
        ref = _np_freqs(*orbit)
        out = _AAS.actionsFreqs(*[arr([x]) for x in orbit])
        got = [float(numpy.asarray(out[3 + k][0])) for k in range(3)]
        numpy.testing.assert_allclose(got, ref, rtol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_staeckel_freq_grad_dOmegar_dR_target(backend):
    # anchor value: dOmega_r/dR at the generic MiyamotoNagai probe ~ -1.9327
    g = _backend_freq_grads(backend, _ORBITS["generic"])
    numpy.testing.assert_allclose(g[0][0], -1.9327, rtol=1e-3)


_AASU = actionAngleStaeckel(pot=_MP, delta=_DELTA, c=True, useu0=True)


@pytest.mark.parametrize("backend", BACKENDS)
def test_staeckel_freq_grad_useu0(backend):
    # useu0=True: the reference u0=calcu0(E,Lz) is coordinate-dependent, so the C
    # Jacobian adds the dOmega/du0 * du0/dx term (mode 2). The FD reference must
    # use the SAME useu0 numpy path -- Omega is gauge-invariant only up to the
    # u0-reference quadrature offset (~3e-6), so a mode-0 FD is inconsistent.
    coords = _ORBITS["generic"]
    eps = 1e-5
    fd = [[], [], []]
    for i in range(5):
        up = list(coords)
        dn = list(coords)
        up[i] += eps
        dn[i] -= eps
        fu = _AASU.actionsFreqs(*[numpy.array([c]) for c in up])
        fdn = _AASU.actionsFreqs(*[numpy.array([c]) for c in dn])
        for k in range(3):
            fd[k].append((float(fu[3 + k][0]) - float(fdn[3 + k][0])) / (2.0 * eps))
    g = _backend_freq_grads(backend, coords, useu0=True)
    for k in range(3):
        numpy.testing.assert_allclose(g[k], fd[k], rtol=3e-3, atol=3e-6)
