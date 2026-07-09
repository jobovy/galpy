###############################################################################
# test_backend_streamgapdf.py: backend (jax/torch) coverage for the analytic
# impulse-approximation kernels of streamgapdf (Plummer / Hernquist, straight &
# curved-stream, HernquistX, _rotation_vy). The numpy path is byte-identical
# (test_streamgapdf_impulse unchanged); this exercises the resolved-namespace
# dispatch:
#   (a) value parity numpy<->jax<->torch of every kernel (incl. the wperp->0
#       degenerate perpendicular-impact branch and all three HernquistX
#       regimes), reusing the test_streamgapdf_impulse configs with FIXED seeds,
#   (b) grad-vs-FD of ||plummer_curvedstream||^2 w.r.t. b/GM/rs/w and of
#       HernquistX across regimes (jax.grad / torch.autograd vs central FD).
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
    HernquistX,
    _rotation_vy,
    impulse_deltav_hernquist,
    impulse_deltav_hernquist_curvedstream,
    impulse_deltav_plummer,
    impulse_deltav_plummer_curvedstream,
)


def _to_backend(backend, x):
    return jnp.asarray(x) if backend == "jax" else torch.asarray(x)


# ------- Fixed-seed input configs (mirror test_streamgapdf_impulse) -------
def _make_cases():
    numpy.random.seed(12345)
    xpos = numpy.random.normal(size=20)
    vb = numpy.zeros((20, 3))
    vb[:, 0] = 3.4
    xposc = numpy.array([xpos, numpy.zeros(20), numpy.zeros(20)]).T
    wperp_nonzero = numpy.array([0.0, numpy.pi / 2.0, 0.0])
    # s spanning all three HernquistX regimes incl. near s=1
    sarr = numpy.concatenate(
        [
            numpy.linspace(1e-6, 0.999999, 30),
            numpy.array([1.0 - 1e-11, 1.0, 1.0 + 1e-11]),
            numpy.linspace(1.000001, numpy.sqrt(2.0), 30),
        ]
    )
    return {
        "plummer_bunch": (
            impulse_deltav_plummer,
            dict(v=vb.copy(), y=xpos.copy(), b=3.0, w=wperp_nonzero, GM=1.5, rs=4.0),
        ),
        # perpendicular impact -> wperp==0 degenerate (guarded) branch
        "plummer_perp": (
            impulse_deltav_plummer,
            dict(
                v=numpy.array([[0.0, numpy.pi, 0.0]]),
                y=numpy.array([0.0]),
                b=3.0,
                w=wperp_nonzero,
                GM=1.5,
                rs=4.0,
            ),
        ),
        "plummer_curved_bunch": (
            impulse_deltav_plummer_curvedstream,
            dict(
                v=vb.copy(),
                x=xposc.copy(),
                b=3.0,
                w=wperp_nonzero,
                x0=numpy.array([0.0, 0.0, 0.0]),
                v0=numpy.array([3.4, 0.0, 0.0]),
                GM=numpy.pi,
                rs=numpy.exp(1.0),
            ),
        ),
        "plummer_curved_single": (
            impulse_deltav_plummer_curvedstream,
            dict(
                v=numpy.array([[3.4, 0.1, 0.2]]),
                x=numpy.array([[4.0, 0.1, 0.0]]),
                b=3.0,
                w=numpy.array([0.2, 1.1, 0.3]),
                x0=numpy.array([0.0, 0.0, 0.0]),
                v0=numpy.array([3.4, 0.1, 0.2]),
                GM=1.5,
                rs=4.0,
            ),
        ),
        "hernquist_bunch": (
            impulse_deltav_hernquist,
            dict(
                v=vb.copy(), y=xpos.copy(), b=3.0, w=wperp_nonzero, GM=numpy.pi, rs=2.0
            ),
        ),
        # perpendicular impact -> wperp==0 degenerate (guarded) branch
        "hernquist_perp": (
            impulse_deltav_hernquist,
            dict(
                v=numpy.array([[0.0, numpy.pi, 0.0]]),
                y=numpy.array([2.0]),
                b=3.0,
                w=wperp_nonzero,
                GM=1.5,
                rs=4.0,
            ),
        ),
        "hernquist_curved_bunch": (
            impulse_deltav_hernquist_curvedstream,
            dict(
                v=vb.copy(),
                x=xposc.copy(),
                b=3.0,
                w=wperp_nonzero,
                x0=numpy.array([0.0, 0.0, 0.0]),
                v0=numpy.array([3.4, 0.0, 0.0]),
                GM=numpy.pi,
                rs=numpy.exp(1.0),
            ),
        ),
        "hernquist_curved_single": (
            impulse_deltav_hernquist_curvedstream,
            dict(
                v=numpy.array([[3.4, 0.1, 0.2]]),
                x=numpy.array([[4.0, 0.1, 0.0]]),
                b=3.0,
                w=numpy.array([0.2, 1.1, 0.3]),
                x0=numpy.array([0.0, 0.0, 0.0]),
                v0=numpy.array([3.4, 0.1, 0.2]),
                GM=1.5,
                rs=4.0,
            ),
        ),
        "hernquistX": (HernquistX, dict(s=sarr)),
        "rotation_vy_fwd": (_rotation_vy, dict(v=vb.copy(), inv=False)),
        "rotation_vy_inv": (_rotation_vy, dict(v=vb.copy(), inv=True)),
    }


CASES = _make_cases()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", list(CASES))
def test_kernel_parity(backend, name):
    fn, kwargs = CASES[name]
    ref = numpy.asarray(fn(**kwargs))
    bkwargs = {
        k: (_to_backend(backend, v) if isinstance(v, numpy.ndarray) else v)
        for k, v in kwargs.items()
    }
    got = fn(**bkwargs)
    assert is_backend_array(got), f"{name} on {backend} did not return a backend array"
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-10, atol=1e-12)


# ------- grad-vs-FD: ||plummer_curvedstream||^2 w.r.t. b/GM/rs/w -------
_GRAD_CFG = dict(
    v=numpy.array([[3.4, 0.1, 0.2], [3.3, -0.1, 0.15]]),
    x=numpy.array([[4.0, 0.1, 0.0], [3.5, -0.2, 0.1]]),
    b=3.0,
    w=numpy.array([0.2, 1.1, 0.3]),
    x0=numpy.array([0.0, 0.0, 0.0]),
    v0=numpy.array([3.4, 0.1, 0.2]),
    GM=1.5,
    rs=4.0,
)


def _loss_np(b, GM, rs, w):
    c = _GRAD_CFG
    kick = impulse_deltav_plummer_curvedstream(
        c["v"], c["x"], b, w, c["x0"], c["v0"], GM, rs
    )
    return float(numpy.sum(kick**2))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("param", ["b", "GM", "rs"])
def test_plummer_curved_grad_scalar_vs_fd(backend, param):
    c = _GRAD_CFG
    base = dict(b=c["b"], GM=c["GM"], rs=c["rs"], w=c["w"])
    h = 1e-6
    lo = dict(base)
    lo[param] = base[param] - h
    hi = dict(base)
    hi[param] = base[param] + h
    gfd = (_loss_np(**hi) - _loss_np(**lo)) / (2.0 * h)

    def loss_backend(bval, GMval, rsval, wval):
        kick = impulse_deltav_plummer_curvedstream(
            _to_backend(backend, c["v"]),
            _to_backend(backend, c["x"]),
            bval,
            wval,
            _to_backend(backend, c["x0"]),
            _to_backend(backend, c["v0"]),
            GMval,
            rsval,
        )
        return (kick**2).sum()

    if backend == "jax":
        args = dict(
            bval=jnp.asarray(c["b"]),
            GMval=jnp.asarray(c["GM"]),
            rsval=jnp.asarray(c["rs"]),
            wval=jnp.asarray(c["w"]),
        )
        key = {"b": "bval", "GM": "GMval", "rs": "rsval"}[param]
        g = float(jax.grad(lambda p: loss_backend(**{**args, key: p}))(args[key]))
    else:
        vals = {
            "bval": torch.tensor(c["b"], requires_grad=(param == "b")),
            "GMval": torch.tensor(c["GM"], requires_grad=(param == "GM")),
            "rsval": torch.tensor(c["rs"], requires_grad=(param == "rs")),
            "wval": torch.tensor(c["w"]),
        }
        key = {"b": "bval", "GM": "GMval", "rs": "rsval"}[param]
        loss_backend(**vals).backward()
        g = float(vals[key].grad)
    numpy.testing.assert_allclose(g, gfd, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("backend", BACKENDS)
def test_plummer_curved_grad_w_vs_fd(backend):
    c = _GRAD_CFG
    h = 1e-6
    gfd = numpy.empty(3)
    for i in range(3):
        wl = c["w"].copy()
        wl[i] -= h
        wh = c["w"].copy()
        wh[i] += h
        gfd[i] = (
            _loss_np(c["b"], c["GM"], c["rs"], wh)
            - _loss_np(c["b"], c["GM"], c["rs"], wl)
        ) / (2.0 * h)

    def loss_backend(wval):
        kick = impulse_deltav_plummer_curvedstream(
            _to_backend(backend, c["v"]),
            _to_backend(backend, c["x"]),
            c["b"],
            wval,
            _to_backend(backend, c["x0"]),
            _to_backend(backend, c["v0"]),
            c["GM"],
            c["rs"],
        )
        return (kick**2).sum()

    if backend == "jax":
        g = numpy.asarray(jax.grad(loss_backend)(jnp.asarray(c["w"])))
    else:
        wt = torch.tensor(c["w"], requires_grad=True)
        loss_backend(wt).backward()
        g = wt.grad.detach().cpu().numpy()
    numpy.testing.assert_allclose(g, gfd, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("s0", [0.3, 0.7, 0.95, 1.05, 1.3])
def test_hernquistX_grad_vs_fd(backend, s0):
    h = 1e-7
    gfd = (float(HernquistX(s0 + h)) - float(HernquistX(s0 - h))) / (2.0 * h)
    if backend == "jax":
        g = float(jax.grad(lambda s: HernquistX(s))(jnp.asarray(s0)))
    else:
        st = torch.tensor(s0, requires_grad=True)
        HernquistX(st).backward()
        g = float(st.grad)
    numpy.testing.assert_allclose(g, gfd, rtol=1e-5, atol=1e-7)
