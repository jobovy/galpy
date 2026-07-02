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


def _backend_grads(backend, orbit, c=False):
    # c=False exercises the internal vectorised donor-graft path; c=True routes
    # through the public API to the C-native Jacobian vjp / autograd.Function.
    if c:
        aAS = actionAngleStaeckel(pot=_MP, delta=_DELTA, c=True)
        if backend == "jax":
            args = [jnp.asarray([x]) for x in orbit]
            gjr = jax.grad(lambda *a: jnp.sum(aAS(*a)[0]), argnums=(0, 1, 2, 3, 4))(
                *args
            )
            gjz = jax.grad(lambda *a: jnp.sum(aAS(*a)[2]), argnums=(0, 1, 2, 3, 4))(
                *args
            )
            return [float(g[0]) for g in gjr], [float(g[0]) for g in gjz]
        args = [torch.tensor([x], requires_grad=True) for x in orbit]
        out = aAS(*args)
        gjr = torch.autograd.grad(out[0].sum(), args, retain_graph=True)
        gjz = torch.autograd.grad(out[2].sum(), args)
        return [float(g[0]) for g in gjr], [float(g[0]) for g in gjz]
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
@pytest.mark.parametrize("c", [False, True])
def test_staeckel_action_grad_vs_fd(backend, orbit, c):
    # d(jr,jz)/d(R,vR,vT,z,vz) via backend AD vs numpy central FD, for both the
    # c=False internal donor-graft path and the c=True C-native Jacobian. The
    # residual is the plain-GL-vs-donor quadrature offset (~5e-4 relative at
    # order 10); the pre-fix failure modes were ~6e2 (naive d(sqrt S)) and ~6e0
    # (the (E,Lz,I3)-only chain missing the u0/v0u geometry) times larger.
    coords = _ORBITS[orbit]
    fjr, fjz = _fd_grad(coords)
    gjr, gjz = _backend_grads(backend, coords, c=c)
    numpy.testing.assert_allclose(gjr, fjr, rtol=2e-3, atol=2e-6)
    numpy.testing.assert_allclose(gjz, fjz, rtol=2e-3, atol=2e-6)
    assert numpy.all(numpy.isfinite(gjr)) and numpy.all(numpy.isfinite(gjz))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("orbit", list(_ORBITS))
def test_staeckel_c_native_matches_donor_grad(backend, orbit):
    # DECISIVE cross-check: the C-native Jacobian (c=True) and the internal
    # donor-graft gradient (c=False) use the same t^2 derivative integrals, so
    # they must agree to ~machine precision (not merely the FD floor).
    coords = _ORBITS[orbit]
    gjr_c, gjz_c = _backend_grads(backend, coords, c=True)
    gjr_d, gjz_d = _backend_grads(backend, coords, c=False)
    numpy.testing.assert_allclose(gjr_c, gjr_d, rtol=1e-6, atol=1e-9)
    numpy.testing.assert_allclose(gjz_c, gjz_d, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("backend", BACKENDS)
def test_staeckel_action_value_c_true_matches_numpy(backend):
    # c=True backend forward (no-grad and under an AD trace) == numpy c=True
    # action values EXACTLY (the vjp forward is the plain round-trip C action).
    aAS = actionAngleStaeckel(pot=_MP, delta=_DELTA, c=True)
    for orbit in _ORBITS.values():
        jr_np, _, jz_np = (float(x[0]) for x in aAS(*[numpy.array([c]) for c in orbit]))
        if backend == "jax":
            out = aAS(*[jnp.asarray([c]) for c in orbit])
            jr_b, jz_b = float(out[0][0]), float(out[2][0])
        else:
            out = aAS(*[torch.tensor([c]) for c in orbit])
            jr_b, jz_b = float(out[0].detach()[0]), float(out[2].detach()[0])
        numpy.testing.assert_allclose(jr_b, jr_np, rtol=1e-15)
        numpy.testing.assert_allclose(jz_b, jz_np, rtol=1e-15)


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
@pytest.mark.parametrize("c", [False, True])
def test_staeckel_action_grad_jit(backend, c):
    # the gradient must survive the user's jit unchanged -- for c=False (donor
    # graft) and c=True (the C-native Jacobian through jax.pure_callback).
    coords = _ORBITS["generic"]
    if c:
        aAS = actionAngleStaeckel(pot=_MP, delta=_DELTA, c=True)

        def djr_dR(R):
            args = [R] + [jnp.asarray([x]) for x in coords[1:]]
            return jnp.sum(aAS(*args)[0])

    else:

        def djr_dR(R):
            args = [R] + [jnp.asarray([x]) for x in coords[1:]]
            return jnp.sum(_staeckel_actions(jnp, *args, _MP, _DELTA, _ORDER)[0])

    R0 = jnp.asarray([coords[0]])
    eager = jax.grad(djr_dR)(R0)
    jitted = jax.jit(jax.grad(djr_dR))(R0)
    numpy.testing.assert_allclose(float(jitted[0]), float(eager[0]), rtol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("c", [False, True])
def test_staeckel_action_grad_public_api(backend, c):
    # same gradients through the public actionAngleStaeckel(...) call, c in
    # (False, True).
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


@pytest.mark.parametrize("backend", BACKENDS)
def test_staeckel_actionsfreqs_c_true_backend(backend):
    # c=True on backend arrays through actionsFreqs / actionsFreqsAngles: the
    # freq/angle VALUES are byte-identical to numpy c=True (ungrafted, Phase-2),
    # and the ACTIONS stay differentiable (djr matches the FD gold).
    aAS = actionAngleStaeckel(pot=_MP, delta=_DELTA, c=True)
    mk = (
        (lambda c: jnp.asarray([c]))
        if backend == "jax"
        else (lambda c: torch.tensor([c]))
    )
    det = (
        (lambda x: numpy.asarray(x))
        if backend == "jax"
        else (lambda x: numpy.asarray(x.detach()))
    )
    # freq/angle VALUES byte-identical to numpy c=True over all orbits (incl. the
    # near-circular one, which drives the NaN->epifreq/omegac substitution).
    for orbit in _ORBITS.values():
        ref_f = [
            numpy.asarray(x)
            for x in aAS.actionsFreqs(*[numpy.array([c]) for c in orbit])
        ]
        ref_a = [
            numpy.asarray(x)
            for x in aAS.actionsFreqsAngles(*[numpy.array([c]) for c in (*orbit, 0.7)])
        ]
        gotf = [det(x) for x in aAS.actionsFreqs(*[mk(c) for c in orbit])]
        gota = [det(x) for x in aAS.actionsFreqsAngles(*[mk(c) for c in (*orbit, 0.7)])]
        for g, r in zip(gotf, ref_f):
            numpy.testing.assert_allclose(g, r, rtol=1e-15, atol=1e-15, equal_nan=True)
        for g, r in zip(gota, ref_a):
            numpy.testing.assert_allclose(g, r, rtol=1e-15, atol=1e-15, equal_nan=True)
    # the ACTIONS stay differentiable through actionsFreqs (djr matches FD gold).
    orbit = _ORBITS["generic"]
    fjr, _ = _fd_grad(orbit)
    if backend == "jax":
        af = [jnp.asarray([c]) for c in orbit]
        g = jax.grad(lambda R: jnp.sum(aAS.actionsFreqs(R, *af[1:])[0]))(af[0])
        djr_dR = float(g[0])
    else:
        gargs = [torch.tensor([c], requires_grad=True) for c in orbit]
        aAS.actionsFreqs(*gargs)[0].sum().backward()
        djr_dR = float(gargs[0].grad[0])
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
