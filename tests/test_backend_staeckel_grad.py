###############################################################################
# test_backend_staeckel_grad.py: accurate Staeckel ACTION gradients under
# jax/torch AD. The plain-GL action value is kept (C parity; numpy path
# byte-identical) while the gradient is grafted from the t^2-substituted donor
# quadrature (_staeckel_t2_action), which is turning-point-regular where naive
# d(sqrt S) is singular and carries the full (E, Lz, I3, u0/v0u geometry)
# dependence that the dJ/d(E,Lz,I3) chain alone misses. First-order only.
#
# c=True: the forward VALUE comes from the compiled C wrapper (eager numpy
# round-trip; jax.pure_callback under a trace) and the same donor supplies the
# ACTION gradients; frequency/angle values pass through ungrafted (Phase 2).
# The numpy plain-GL FD gold matches C to ~4e-16, so one gold serves both.
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
from galpy.actionAngle.actionAngleStaeckel_c import _ext_loaded
from galpy.backend import get_namespace
from galpy.potential import MiyamotoNagaiPotential

_MP = MiyamotoNagaiPotential(normalize=1.0, a=0.5, b=0.0375)
_DELTA, _ORDER = 0.45, 10
# c=False exercises the in-backend vectorised path directly; c=True the
# C-value + donor-graft dispatch (needs the compiled extension).
_CS = [False] + ([True] if _ext_loaded else [])
_AAS_C = actionAngleStaeckel(pot=_MP, delta=_DELTA, c=True) if _ext_loaded else None

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


def _eval_actions(backend, args, c):
    """(jr, Lz, jz): the internal vectorised path for c=False (unit-level donor
    coverage) or the public c=True API (C value + graft dispatch)."""
    if c:
        return _AAS_C(*args)
    if backend == "jax":
        return _staeckel_actions(jnp, *args, _MP, _DELTA, _ORDER)
    xt = get_namespace(torch.tensor(0.0))
    return _staeckel_actions(xt, *args, _MP, _DELTA, _ORDER)


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


def _backend_grads(backend, orbit, c):
    if backend == "jax":

        def f(i, *coords):
            return jnp.sum(_eval_actions("jax", coords, c)[i])

        args = [jnp.asarray([x]) for x in orbit]
        gjr = jax.grad(lambda *a: f(0, *a), argnums=(0, 1, 2, 3, 4))(*args)
        gjz = jax.grad(lambda *a: f(2, *a), argnums=(0, 1, 2, 3, 4))(*args)
        return [float(g[0]) for g in gjr], [float(g[0]) for g in gjz]
    args = [torch.tensor([x], requires_grad=True) for x in orbit]
    out = _eval_actions("torch", args, c)
    gjr = torch.autograd.grad(out[0].sum(), args, retain_graph=True)
    gjz = torch.autograd.grad(out[2].sum(), args)
    return [float(g[0]) for g in gjr], [float(g[0]) for g in gjz]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("orbit", list(_ORBITS))
@pytest.mark.parametrize("c", _CS)
def test_staeckel_action_grad_vs_fd(backend, orbit, c):
    # d(jr,jz)/d(R,vR,vT,z,vz) via backend AD vs numpy central FD. The residual
    # is the plain-GL-vs-donor quadrature offset (~5e-4 relative at order 10);
    # the pre-fix failure modes were ~6e2 (naive d(sqrt S)) and ~6e0 (the
    # (E,Lz,I3)-only chain missing the u0/v0u geometry) times larger.
    coords = _ORBITS[orbit]
    fjr, fjz = _fd_grad(coords)
    gjr, gjz = _backend_grads(backend, coords, c)
    numpy.testing.assert_allclose(gjr, fjr, rtol=2e-3, atol=2e-6)
    numpy.testing.assert_allclose(gjz, fjz, rtol=2e-3, atol=2e-6)
    assert numpy.all(numpy.isfinite(gjr)) and numpy.all(numpy.isfinite(gjz))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("c", _CS)
def test_staeckel_action_value_unchanged(backend, c):
    # the graft must not change the forward action value: no-grad backend
    # forward == numpy forward, and the value under an AD trace == the no-grad
    # forward (the donor terms cancel exactly).
    coords = _ORBITS["generic"]
    if c:
        out_np = _AAS_C(*[numpy.array([x]) for x in coords])
        jr_np, jz_np = float(out_np[0][0]), float(out_np[2][0])
    else:
        jr_np, jz_np = _np_actions(*coords)
    if backend == "jax":
        args = [jnp.asarray([x]) for x in coords]
        out = _eval_actions("jax", args, c)
        jr_fwd, jz_fwd = float(out[0][0]), float(out[2][0])
        val, _ = jax.value_and_grad(
            lambda R: jnp.sum(_eval_actions("jax", (R, *args[1:]), c)[0])
        )(args[0])
        jr_traced = float(val)
    else:
        args = [torch.tensor([x]) for x in coords]
        out = _eval_actions("torch", args, c)
        jr_fwd, jz_fwd = float(out[0][0]), float(out[2][0])
        gargs = [torch.tensor([x], requires_grad=True) for x in coords]
        jr_traced = float(_eval_actions("torch", gargs, c)[0].detach()[0])
    numpy.testing.assert_allclose(jr_fwd, jr_np, rtol=1e-14)
    numpy.testing.assert_allclose(jz_fwd, jz_np, rtol=1e-14)
    numpy.testing.assert_allclose(jr_traced, jr_fwd, rtol=1e-14)


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b == "jax"])
@pytest.mark.parametrize("c", _CS)
def test_staeckel_action_grad_jit(backend, c):
    # the grafted gradient must survive the user's jit unchanged; for c=True
    # both value and grad flow through the jax.pure_callback under jit.
    coords = _ORBITS["generic"]

    def jr_sum(R):
        args = [R] + [jnp.asarray([x]) for x in coords[1:]]
        return jnp.sum(_eval_actions("jax", args, c)[0])

    R0 = jnp.asarray([coords[0]])
    eager_val = jr_sum(R0)
    jit_val = jax.jit(jr_sum)(R0)
    numpy.testing.assert_allclose(float(jit_val), float(eager_val), rtol=1e-15)
    eager = jax.grad(jr_sum)(R0)
    jitted = jax.jit(jax.grad(jr_sum))(R0)
    numpy.testing.assert_allclose(float(jitted[0]), float(eager[0]), rtol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("c", _CS)
def test_staeckel_action_grad_public_api(backend, c):
    # same gradients through the public actionAngleStaeckel(...) call.
    coords = _ORBITS["generic"]
    aAS = actionAngleStaeckel(pot=_MP, delta=_DELTA, c=c)
    fjr, _ = _fd_grad(coords)
    if backend == "jax":
        g = jax.grad(
            lambda R: jnp.sum(aAS(R, *[jnp.asarray([x]) for x in coords[1:]])[0])
        )(jnp.asarray([coords[0]]))
        djr_dR = float(g[0])
    else:
        args = [torch.tensor([x], requires_grad=True) for x in coords]
        aAS(*args)[0].sum().backward()
        djr_dR = float(args[0].grad[0])
    numpy.testing.assert_allclose(djr_dR, fjr[0], rtol=2e-3, atol=2e-6)


@pytest.mark.skipif(not _ext_loaded, reason="C extension not available")
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("orbit", list(_ORBITS))
def test_staeckel_c_backend_forward_matches_numpy(backend, orbit):
    # c=True backend forward == numpy c=True forward: the value is the SAME C
    # computation on the same float64s either way (round-trip is exact).
    coords = _ORBITS[orbit]
    np_out = _AAS_C(*[numpy.array([x]) for x in coords])
    assert isinstance(np_out[0], numpy.ndarray)  # numpy in -> numpy out
    if backend == "jax":
        args = [jnp.asarray([x]) for x in coords]
    else:
        args = [torch.tensor([x]) for x in coords]
    out = _AAS_C(*args)
    for a, b in zip(out, np_out):
        numpy.testing.assert_allclose(float(a[0]), float(b[0]), rtol=1e-15)


@pytest.mark.skipif(not _ext_loaded, reason="C extension not available")
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("orbit", ["generic", "circular"])
def test_staeckel_c_backend_freqs_angles_forward(backend, orbit):
    # c=True actionsFreqs / actionsFreqsAngles with backend inputs: all values
    # (actions grafted, freqs/angles pass-through) match the numpy C outputs.
    # The exactly-circular orbit exercises the NaN-frequency -> epifreq/omegac/
    # verticalfreq substitution inside the numpy host.
    coords = (1.0, 0.0, 1.0, 0.0, 0.0) if orbit == "circular" else _ORBITS["generic"]
    phi0 = 0.3
    np6 = _AAS_C.actionsFreqs(*[numpy.array([x]) for x in coords])
    assert numpy.all(numpy.isfinite(numpy.array(np6)))
    np9 = _AAS_C.actionsFreqsAngles(
        *([numpy.array([x]) for x in coords] + [numpy.array([phi0])])
    )
    if backend == "jax":
        args = [jnp.asarray([x]) for x in coords]
        phi = jnp.asarray([phi0])
    else:
        args = [torch.tensor([x]) for x in coords]
        phi = torch.tensor([phi0])
    b6 = _AAS_C.actionsFreqs(*args)
    b9 = _AAS_C.actionsFreqsAngles(*(args + [phi]))
    for a, b in zip(b6, np6):
        numpy.testing.assert_allclose(float(a[0]), float(b[0]), rtol=1e-15)
    for a, b in zip(b9, np9):
        numpy.testing.assert_allclose(float(a[0]), float(b[0]), rtol=1e-15)


@pytest.mark.skipif(not _ext_loaded, reason="C extension not available")
@pytest.mark.parametrize("backend", BACKENDS)
def test_staeckel_c_backend_useu0_u0_delta(backend):
    # useu0=True (u0 computed on the numpy host), an explicit backend-array u0
    # kwarg, and a backend-array delta override all match the numpy C outputs.
    coords = _ORBITS["generic"]
    aAS = actionAngleStaeckel(pot=_MP, delta=_DELTA, c=True, useu0=True)
    np_args = [numpy.array([x]) for x in coords]
    if backend == "jax":
        args = [jnp.asarray([x]) for x in coords]
        u0, delta = jnp.asarray([1.1]), jnp.asarray([0.4])
    else:
        args = [torch.tensor([x]) for x in coords]
        u0, delta = torch.tensor([1.1]), torch.tensor([0.4])
    for kw_np, kw_b in (
        ({}, {}),
        ({"u0": numpy.array([1.1])}, {"u0": u0}),
        ({"u0": numpy.array([1.1]), "delta": numpy.array([0.4])},
         {"u0": u0, "delta": delta}),
    ):  # fmt: skip
        np_out = aAS(*np_args, **kw_np)
        out = aAS(*args, **kw_b)
        for a, b in zip(out, np_out):
            numpy.testing.assert_allclose(float(a[0]), float(b[0]), rtol=1e-15)


@pytest.mark.skipif(not _ext_loaded, reason="C extension not available")
@pytest.mark.parametrize("backend", BACKENDS)
def test_staeckel_c_backend_freqs_action_grad(backend):
    # the ACTION outputs of c=True actionsFreqs carry the grafted gradient too.
    coords = _ORBITS["generic"]
    fjr, _ = _fd_grad(coords)
    if backend == "jax":
        g = jax.grad(
            lambda R: jnp.sum(
                _AAS_C.actionsFreqs(R, *[jnp.asarray([x]) for x in coords[1:]])[0]
            )
        )(jnp.asarray([coords[0]]))
        djr_dR = float(g[0])
    else:
        args = [torch.tensor([x], requires_grad=True) for x in coords]
        _AAS_C.actionsFreqs(*args)[0].sum().backward()
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


def test_under_torch_grad_without_torch(monkeypatch):
    # the sys.modules early-out (torch never imported -> False): the test env
    # always has torch imported, so simulate its absence.
    import sys

    from galpy.backend._namespaces import under_torch_grad

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert under_torch_grad(numpy.array([1.0])) is False
