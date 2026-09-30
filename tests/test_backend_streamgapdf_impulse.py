###############################################################################
# test_backend_streamgapdf_impulse.py: backend (jax/torch) coverage for the
# GENERAL (numerical, spherical-potential) impulse-approximation kernels of
# streamgapdf that PR #1090 left numpy-only:
#   * impulse_deltav_general             -- the scipy.quad quadrature path,
#     migrated to a single vectorised backend Gauss-Legendre pass (fixed_quad),
#   * impulse_deltav_general_orbitintegration -- the orbit-integration path,
#     migrated to ONE batched in-backend differentiable ODE solve (diffrax for
#     jax, torchdiffeq for torch) + a differentiable composite-Simpson.
# The numeric file test_streamgapdf_impulse.py exercises the numpy path (it is
# byte-identical); this file drives the resolved-namespace backend branch that
# a numpy-only run never touches:
#   (a) value parity numpy<->jax<->torch (feeding the same inputs as backend
#       arrays), asserting the result is a backend array,
#   (b) grad-vs-FD (jax.grad / torch.autograd vs central finite differences of
#       the SAME backend function) of sum(deltav**2) w.r.t. the impact parameter
#       b, the perturber velocity w, and a stream velocity component, with an
#       h-convergence check.
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

from galpy.backend import as_numpy, is_backend_array
from galpy.df.streamgapdf import (
    impulse_deltav_general,
    impulse_deltav_general_orbitintegration,
)
from galpy.potential import (
    HernquistPotential,
    LogarithmicHaloPotential,
    PlummerPotential,
)


def _to_backend(backend, x):
    return jnp.asarray(x) if backend == "jax" else torch.asarray(x)


# ---------------------------------------------------------------------------
# impulse_deltav_general (quadrature path)
# ---------------------------------------------------------------------------
def _general_config():
    numpy.random.seed(20240719)
    v = numpy.zeros((25, 3))
    v[:, 0] = 3.4
    v[:, 1] = 0.05 * numpy.random.normal(size=25)
    y = numpy.random.normal(size=25)
    w = numpy.array([0.1, numpy.pi / 2.0, 0.05])
    b = 3.0
    return v, y, b, w


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "pot",
    [
        PlummerPotential(amp=numpy.pi, b=numpy.exp(1.0)),
        HernquistPotential(amp=2.3, a=3.1),
    ],
)
def test_general_parity(backend, pot):
    v, y, b, w = _general_config()
    ref = numpy.asarray(impulse_deltav_general(v, y, b, w, pot))
    got = impulse_deltav_general(
        _to_backend(backend, v),
        _to_backend(backend, y),
        b,
        _to_backend(backend, w),
        pot,
    )
    assert is_backend_array(got), (
        f"{backend}: impulse_deltav_general did not return a backend array"
    )
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("backend", BACKENDS)
def test_general_single_vector_parity(backend):
    # 1-D v input (single star) -> (1, 3) output, hits the reshape branch
    pot = PlummerPotential(amp=1.5, b=4.0)
    v = numpy.array([3.4, 0.0, 0.0])
    y = numpy.array([4.0])
    w = numpy.array([0.0, numpy.pi / 2.0, 0.0])
    ref = numpy.asarray(impulse_deltav_general(v, y, 3.0, w, pot))
    got = impulse_deltav_general(
        _to_backend(backend, v),
        _to_backend(backend, y),
        3.0,
        _to_backend(backend, w),
        pot,
    )
    assert is_backend_array(got)
    assert as_numpy(got).shape == (1, 3)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-8, atol=1e-10)


def _grad_and_fd(backend, loss_backend, base, key, direction, hs=(1e-4, 1e-6)):
    """Return (ad_directional_deriv, {h: central_fd}) for loss along ``direction``
    in the argument ``key`` (a scalar or vector), FD taken on the SAME backend
    function (so any quadrature/ODE discretization is identical to the AD path)."""
    # AD gradient
    if backend == "jax":
        args = {k: jnp.asarray(numpy.asarray(v, dtype=float)) for k, v in base.items()}
        g = jax.grad(lambda p: loss_backend({**args, key: p}))(args[key])
        g = numpy.asarray(g)
    else:
        args = {
            k: torch.tensor(numpy.asarray(v, dtype=float), requires_grad=(k == key))
            for k, v in base.items()
        }
        loss_backend(args).backward()
        g = args[key].grad.detach().cpu().numpy()
    ad_dir = float(numpy.sum(g * direction))
    # central FD along ``direction`` on the same backend function
    fds = {}
    for h in hs:
        hi = dict(base)
        hi[key] = numpy.asarray(base[key], dtype=float) + h * direction
        lo = dict(base)
        lo[key] = numpy.asarray(base[key], dtype=float) - h * direction
        lhi = float(
            as_numpy(loss_backend({k: _to_backend(backend, v) for k, v in hi.items()}))
        )
        llo = float(
            as_numpy(loss_backend({k: _to_backend(backend, v) for k, v in lo.items()}))
        )
        fds[h] = (lhi - llo) / (2.0 * h)
    return ad_dir, fds


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("key", ["b", "w", "v"])
def test_general_grad_vs_fd(backend, key):
    v, y, b, w = _general_config()
    # smaller star set keeps the FD sweeps cheap
    v = v[:6].copy()
    y = y[:6].copy()
    pot = PlummerPotential(amp=1.5, b=4.0)

    def loss(args):
        kick = impulse_deltav_general(args["v"], args["y"], args["b"], args["w"], pot)
        return (kick**2).sum()

    base = {"v": v, "y": y, "b": b, "w": w}
    rng = numpy.random.RandomState(7)
    direction = {
        "b": numpy.array(1.0),
        "w": rng.normal(size=3),
        "v": rng.normal(size=v.shape),
    }[key]
    ad, fds = _grad_and_fd(backend, loss, base, key, direction)
    hfine = min(fds)
    numpy.testing.assert_allclose(fds[hfine], ad, rtol=1e-5, atol=1e-8)
    # h-convergence: the finer central-FD step is at least as close to AD
    assert abs(fds[hfine] - ad) <= abs(fds[max(fds)] - ad) + 1e-9


# ---------------------------------------------------------------------------
# impulse_deltav_general_orbitintegration (orbit-integration path)
# ---------------------------------------------------------------------------
_OI_TMAX = float(numpy.pi)


def _oi_config():
    # fast, close encounter (kick localised near closest approach); single small
    # orbit + short tmax keep the in-backend ODE solve cheap in the coverage shard
    x0 = numpy.array([1.5, 0.0, 0.0])
    v0 = numpy.array([0.0, 1.0, 0.0])
    w = numpy.array([0.0, 0.0, 100.0])
    return v0, x0, 3.0, w, x0, v0


@pytest.mark.parametrize("backend", BACKENDS)
def test_orbitintegration_parity(backend):
    v, x, b, w, x0, v0 = _oi_config()
    lp = LogarithmicHaloPotential(normalize=1.0)
    pp = PlummerPotential(amp=1.5, b=4.0)
    ref = numpy.asarray(
        impulse_deltav_general_orbitintegration(
            v, x, b, w, x0, v0, pp, _OI_TMAX, lp, nsamp=200
        )
    )
    got = impulse_deltav_general_orbitintegration(
        _to_backend(backend, v),
        _to_backend(backend, x),
        b,
        _to_backend(backend, w),
        _to_backend(backend, x0),
        _to_backend(backend, v0),
        pp,
        _OI_TMAX,
        lp,
        nsamp=200,
    )
    assert is_backend_array(got), (
        f"{backend}: orbitintegration did not return a backend array"
    )
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-7, atol=1e-11)


@pytest.mark.parametrize("backend", BACKENDS)
def test_orbitintegration_grad_vs_fd(backend):
    v, x, b, w, x0, v0 = _oi_config()
    lp = LogarithmicHaloPotential(normalize=1.0)
    pp = PlummerPotential(amp=1.5, b=4.0)
    nsamp = 150

    def loss_b(bval):
        kick = impulse_deltav_general_orbitintegration(
            _to_backend(backend, v),
            _to_backend(backend, x),
            bval,
            _to_backend(backend, w),
            _to_backend(backend, x0),
            _to_backend(backend, v0),
            pp,
            _OI_TMAX,
            lp,
            nsamp=nsamp,
        )
        return (kick**2).sum()

    if backend == "jax":
        ad = float(jax.grad(lambda bb: loss_b(bb))(jnp.asarray(b)))
    else:
        bt = torch.tensor(float(b), requires_grad=True)
        loss_b(bt).backward()
        ad = float(bt.grad)
    fds = {}
    for h in (1e-4, 1e-6):
        fds[h] = (float(as_numpy(loss_b(b + h))) - float(as_numpy(loss_b(b - h)))) / (
            2.0 * h
        )
    hfine = min(fds)
    numpy.testing.assert_allclose(fds[hfine], ad, rtol=1e-5, atol=1e-9)
    assert abs(fds[hfine] - ad) <= abs(fds[max(fds)] - ad) + 1e-9
