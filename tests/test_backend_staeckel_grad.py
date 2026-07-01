###############################################################################
# test_backend_staeckel_grad.py: accurate Staeckel ACTION gradients under
# jax/torch AD. The plain-GL action value is kept (C parity; numpy path
# byte-identical) while the gradient is grafted from the t^2-substituted donor
# quadrature (_staeckel_t2_action), which is turning-point-regular where naive
# d(sqrt S) is singular and carries the full (E, Lz, I3, u0/v0u geometry)
# dependence that the dJ/d(E,Lz,I3) chain alone misses. First-order only.
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
from galpy.actionAngle.actionAngleStaeckel import _staeckel_actions
from galpy.backend import get_namespace
from galpy.potential import MiyamotoNagaiPotential

_MP = MiyamotoNagaiPotential(normalize=1.0, a=0.5, b=0.0375)
_DELTA, _ORDER = 0.45, 10

# (R, vR, vT, z, vz): generic / eccentric-inclined / near-circular, plus the
# turning-point edge orbits (vR=0 at the u turning point, vz=0 at the v one,
# z=0 in the plane) that historically hide bracketing/AD bugs.
_ORBITS = {
    "generic": (1.0, 0.2, 1.1, 0.1, 0.15),
    "eccentric": (1.2, 0.35, 0.85, 0.25, -0.2),
    "nearcirc": (1.0, 0.02, 1.0, 0.02, 0.02),
    "edge_vR0": (1.0, 0.0, 1.1, 0.1, 0.15),
    "edge_vz0": (1.0, 0.2, 1.1, 0.1, 0.0),
    "edge_z0": (1.0, 0.2, 1.1, 0.0, 0.15),
}
_COORDS = ("R", "vR", "vT", "z", "vz")


def _np_actions(*orbit):
    out = _staeckel_actions(
        numpy, *[numpy.array([c]) for c in orbit], _MP, _DELTA, _ORDER
    )
    return float(out[0][0]), float(out[2][0])


_FD_CACHE = {}


def _fd_grad(orbit, eps=1e-5):
    # central finite differences of the numpy plain-GL actions -- the gold
    # reference the backend AD gradients must reproduce.
    if orbit in _FD_CACHE:
        return _FD_CACHE[orbit]
    gjr, gjz = [], []
    for i in range(5):
        up = list(orbit)
        dn = list(orbit)
        up[i] += eps
        dn[i] -= eps
        jru, jzu = _np_actions(*up)
        jrd, jzd = _np_actions(*dn)
        gjr.append((jru - jrd) / (2.0 * eps))
        gjz.append((jzu - jzd) / (2.0 * eps))
    _FD_CACHE[orbit] = (gjr, gjz)
    return gjr, gjz


def _backend_grads(backend, orbit):
    if backend == "jax":

        def f(i, *coords):
            return jnp.sum(_staeckel_actions(jnp, *coords, _MP, _DELTA, _ORDER)[i])

        args = [jnp.asarray([c]) for c in orbit]
        gjr = jax.grad(lambda *a: f(0, *a), argnums=(0, 1, 2, 3, 4))(*args)
        gjz = jax.grad(lambda *a: f(2, *a), argnums=(0, 1, 2, 3, 4))(*args)
        return [float(g[0]) for g in gjr], [float(g[0]) for g in gjz]
    xt = get_namespace(torch.tensor(0.0))
    args = [torch.tensor([c], requires_grad=True) for c in orbit]
    out = _staeckel_actions(xt, *args, _MP, _DELTA, _ORDER)
    gjr = torch.autograd.grad(out[0].sum(), args, retain_graph=True)
    gjz = torch.autograd.grad(out[2].sum(), args)
    return [float(g[0]) for g in gjr], [float(g[0]) for g in gjz]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("orbit", list(_ORBITS))
def test_staeckel_action_grad_vs_fd(backend, orbit):
    # d(jr,jz)/d(R,vR,vT,z,vz) via backend AD vs numpy central FD. The residual
    # is the plain-GL-vs-donor quadrature offset (~5e-4 relative at order 10);
    # the pre-fix failure modes were ~6e2 (naive d(sqrt S)) and ~6e0 (the
    # (E,Lz,I3)-only chain missing the u0/v0u geometry) times larger.
    coords = _ORBITS[orbit]
    fjr, fjz = _fd_grad(coords)
    gjr, gjz = _backend_grads(backend, coords)
    numpy.testing.assert_allclose(gjr, fjr, rtol=2e-3, atol=2e-6)
    numpy.testing.assert_allclose(gjz, fjz, rtol=2e-3, atol=2e-6)
    assert numpy.all(numpy.isfinite(gjr)) and numpy.all(numpy.isfinite(gjz))


@pytest.mark.parametrize("backend", BACKENDS)
def test_staeckel_action_value_unchanged(backend):
    # the graft must not change the forward action value: no-grad backend
    # forward == numpy forward, and the value under an AD trace == the no-grad
    # forward (the donor terms cancel exactly).
    coords = _ORBITS["generic"]
    jr_np, jz_np = _np_actions(*coords)
    if backend == "jax":
        args = [jnp.asarray([c]) for c in coords]
        out = _staeckel_actions(jnp, *args, _MP, _DELTA, _ORDER)
        jr_fwd, jz_fwd = float(out[0][0]), float(out[2][0])
        val, _ = jax.value_and_grad(
            lambda R: jnp.sum(
                _staeckel_actions(jnp, R, *args[1:], _MP, _DELTA, _ORDER)[0]
            )
        )(args[0])
        jr_traced = float(val)
    else:
        xt = get_namespace(torch.tensor(0.0))
        args = [torch.tensor([c]) for c in coords]
        out = _staeckel_actions(xt, *args, _MP, _DELTA, _ORDER)
        jr_fwd, jz_fwd = float(out[0][0]), float(out[2][0])
        gargs = [torch.tensor([c], requires_grad=True) for c in coords]
        jr_traced = float(
            _staeckel_actions(xt, *gargs, _MP, _DELTA, _ORDER)[0].detach()[0]
        )
    numpy.testing.assert_allclose(jr_fwd, jr_np, rtol=1e-14)
    numpy.testing.assert_allclose(jz_fwd, jz_np, rtol=1e-14)
    numpy.testing.assert_allclose(jr_traced, jr_fwd, rtol=1e-14)


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b == "jax"])
def test_staeckel_action_grad_jit(backend):
    # the grafted gradient must survive the user's jit unchanged.
    coords = _ORBITS["generic"]

    def djr_dR(R):
        args = [R] + [jnp.asarray([c]) for c in coords[1:]]
        return jnp.sum(_staeckel_actions(jnp, *args, _MP, _DELTA, _ORDER)[0])

    R0 = jnp.asarray([coords[0]])
    eager = jax.grad(djr_dR)(R0)
    jitted = jax.jit(jax.grad(djr_dR))(R0)
    numpy.testing.assert_allclose(float(jitted[0]), float(eager[0]), rtol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_staeckel_action_grad_public_api(backend):
    # same gradients through the public actionAngleStaeckel(...) call.
    coords = _ORBITS["generic"]
    aAS = actionAngleStaeckel(pot=_MP, delta=_DELTA, c=False)
    fjr, _ = _fd_grad(coords)
    if backend == "jax":
        g = jax.grad(
            lambda R: jnp.sum(aAS(R, *[jnp.asarray([c]) for c in coords[1:]])[0])
        )(jnp.asarray([coords[0]]))
        djr_dR = float(g[0])
    else:
        args = [torch.tensor([c], requires_grad=True) for c in coords]
        aAS(*args)[0].sum().backward()
        djr_dR = float(args[0].grad[0])
    numpy.testing.assert_allclose(djr_dR, fjr[0], rtol=2e-3, atol=2e-6)


def test_graft_gradient_numpy_identity():
    # numpy path: stop_gradient is the identity and the graft is value-neutral.
    from galpy.backend._namespaces import graft_gradient, stop_gradient

    x = numpy.array([1.5])
    assert stop_gradient(x) is x
    numpy.testing.assert_array_equal(
        graft_gradient(numpy.array([2.0]), numpy.array([3.0])), numpy.array([2.0])
    )
