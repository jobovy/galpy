###############################################################################
# test_backend_random.py: the backend-agnostic RNG shim (galpy.backend.random).
#
# Acceptance bar:
#   1. numpy byte-identity: the shim's numpy path (key=None) is tobytes-equal to
#      the raw numpy.random.<fn> call it replaces (so seeded outputs are byte-
#      for-byte unchanged), and the streamspraydf pilot's numpy draws / sample()
#      are byte-identical / deterministic.
#   2. key reproducibility (jax + torch): same key => identical draws; split
#      gives independent (statistically distinct) sub-keys.
#   3. fixed-noise derivative: a reparameterized example mu(θ)+σ(θ)*normal(key)
#      (and a uniform inverse-CDF example) has jax.grad / torch.func.grad w.r.t.
#      θ matching finite differences with the key held fixed; two calls with the
#      same key+θ are identical, different θ differ smoothly.
#   4. jit: the jax example is jax.jit-able and gives the same result.
###############################################################################
import numpy
import pytest

from galpy.backend import as_numpy as _np
from galpy.backend import is_backend_array
from galpy.backend import random as gr

pytestmark = pytest.mark.backend_managed

BACKENDS = ["numpy"]
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

AD_BACKENDS = [b for b in BACKENDS if b != "numpy"]

# A singular covariance (chen24spraydf's default: two all-zero rows/cols) to
# exercise the singular-safe multivariate_normal path on every backend.
_SINGULAR_COV = numpy.array(
    [
        [0.1225, 0, 0, 0, -0.085521, 0],
        [0, 0.161143, 0, 0, 0, 0],
        [0, 0, 0.043865, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [-0.085521, 0, 0, 0, 0.121847, 0],
        [0, 0, 0, 0, 0, 0.147435],
    ]
)
_MEAN6 = numpy.array([1.6, -0.523599, 0.0, 1.0, 0.349066, 0.0])


# ----------------------------------------------------------------------------
# 1. numpy byte-identity: the shim's key=None path == the raw numpy.random call
# ----------------------------------------------------------------------------
def _bytes_equal(a, b):
    return numpy.asarray(a).tobytes() == numpy.asarray(b).tobytes()


def test_numpy_uniform_byte_identical():
    numpy.random.seed(7)
    a = gr.uniform(None, (13,))
    numpy.random.seed(7)
    b = numpy.random.uniform(size=13)
    assert _bytes_equal(a, b)
    # explicit low/high must reproduce the historical scaled draw too
    numpy.random.seed(7)
    a = gr.uniform(None, (5,), low=-2.0, high=3.0)
    numpy.random.seed(7)
    b = numpy.random.uniform(-2.0, 3.0, size=5)
    assert _bytes_equal(a, b)


def test_numpy_normal_byte_identical():
    numpy.random.seed(3)
    a = gr.normal(None, (9,))
    numpy.random.seed(3)
    b = numpy.random.normal(size=9)
    assert _bytes_equal(a, b)
    # shape=None reproduces a scalar numpy.random.normal() (python float, not 0-d)
    numpy.random.seed(3)
    a = gr.normal(None, None)
    numpy.random.seed(3)
    b = numpy.random.normal()
    assert a == b and isinstance(a, float)


def test_numpy_random_byte_identical():
    # random() must stay a DISTINCT numpy.random.random call (not aliased to
    # uniform) so the historical draw sequence is reproduced bit-for-bit.
    numpy.random.seed(11)
    a = gr.random(None, 8)
    numpy.random.seed(11)
    b = numpy.random.random(size=8)
    assert _bytes_equal(a, b)
    numpy.random.seed(11)
    a = gr.random(None)  # scalar
    numpy.random.seed(11)
    b = numpy.random.random()
    assert a == b


def test_numpy_randint_byte_identical():
    numpy.random.seed(1)
    a = gr.randint(None, (6,), 0, 100)
    numpy.random.seed(1)
    b = numpy.random.randint(0, 100, size=6)
    assert _bytes_equal(a, b)
    numpy.random.seed(1)
    a = gr.randint(None, None, 0, 0xFFFFFF)  # scalar (Orbits.py color hash)
    numpy.random.seed(1)
    b = numpy.random.randint(0, 0xFFFFFF)
    assert a == b


def test_numpy_choice_byte_identical():
    numpy.random.seed(2)
    a = gr.choice(None, [1.0, -1.0], shape=17)
    numpy.random.seed(2)
    b = numpy.random.choice([1.0, -1.0], size=17)
    assert _bytes_equal(a, b)


def test_numpy_multivariate_normal_byte_identical():
    numpy.random.seed(4)
    a = gr.multivariate_normal(None, _MEAN6, _SINGULAR_COV, shape=25)
    numpy.random.seed(4)
    b = numpy.random.multivariate_normal(_MEAN6, _SINGULAR_COV, size=25)
    assert _bytes_equal(a, b)


def test_numpy_key_seeds_global_returns_none():
    # key(seed) on the numpy default seeds the global generator and returns the
    # None sentinel; the subsequent shim draw matches the raw seeded call.
    k = gr.key(123)
    assert k is None
    a = gr.uniform(None, (4,))
    numpy.random.seed(123)
    b = numpy.random.uniform(size=4)
    assert _bytes_equal(a, b)


def test_numpy_split_is_none_tuple():
    assert gr.split(None, 3) == (None, None, None)


# ----------------------------------------------------------------------------
# 2. key reproducibility (jax + torch): same key => identical draws
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_key_reproducible(backend):
    k = gr.key(0, backend)
    draws = [
        lambda kk: gr.uniform(kk, (7,)),
        lambda kk: gr.normal(kk, (7,)),
        lambda kk: gr.random(kk, (7,)),
        lambda kk: gr.randint(kk, (7,), 0, 50),
        lambda kk: gr.multivariate_normal(kk, _MEAN6, _SINGULAR_COV, shape=7),
    ]
    for fn in draws:
        first = _np(fn(k))
        second = _np(fn(k))
        assert is_backend_array(fn(k))
        numpy.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_multivariate_normal_singular_cov(backend):
    # The all-zero row/col (dim 3) must have exactly zero variance on every
    # backend (singular-safe factorization), matching numpy's SVD method.
    k = gr.key(1, backend)
    s = _np(gr.multivariate_normal(k, _MEAN6, _SINGULAR_COV, shape=2000))
    assert numpy.max(numpy.abs(s[:, 3] - _MEAN6[3])) < 1e-9
    assert numpy.std(s[:, 0]) > 0.1  # a non-degenerate dim actually varies


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_split_independent_and_reproducible(backend):
    k = gr.key(2, backend)
    ka, kb = gr.split(k, 2)
    a = _np(gr.uniform(ka, (20000,)))
    b = _np(gr.uniform(kb, (20000,)))
    # distinct streams: not equal, and essentially uncorrelated
    assert not numpy.allclose(a, b)
    assert abs(numpy.corrcoef(a, b)[0, 1]) < 0.05
    # split itself is reproducible (same parent key => same sub-keys)
    ka2, kb2 = gr.split(k, 2)
    numpy.testing.assert_array_equal(a, _np(gr.uniform(ka2, (20000,))))
    numpy.testing.assert_array_equal(b, _np(gr.uniform(kb2, (20000,))))


# ----------------------------------------------------------------------------
# 3. fixed-noise derivative (the headline) + 4. jit
# ----------------------------------------------------------------------------
@pytest.mark.skipif(jax is None, reason="jax not installed")
def test_jax_fixed_noise_grad_normal():
    # x(θ) = μ(θ) + σ(θ) * normal(key): grad w.r.t θ with the key FIXED must
    # match a central finite difference using the SAME key on both sides.
    key = gr.key(7, "jax")

    def f(theta):
        mu = theta**2
        sigma = jnp.exp(0.5 * theta)
        return jnp.sum(mu + sigma * gr.normal(key, (4,)))

    th = jnp.asarray(0.3)
    g = float(jax.grad(f)(th))
    eps = 1e-6
    fd = float((f(th + eps) - f(th - eps)) / (2 * eps))
    assert abs(g - fd) < 1e-5
    # same key + θ => bit-identical; different θ differ smoothly
    assert float(f(th)) == float(f(th))
    assert float(f(th)) != float(f(th + 0.1))


@pytest.mark.skipif(jax is None, reason="jax not installed")
def test_jax_fixed_noise_grad_uniform_inverse_cdf():
    # Inverse-CDF reparam: exponential x = -log(1-u)/rate, rate=θ.
    key = gr.key(9, "jax")

    def f(theta):
        u = gr.uniform(key, (5,))
        x = -jnp.log(1.0 - u) / theta
        return jnp.sum(x)

    th = jnp.asarray(1.7)
    g = float(jax.grad(f)(th))
    eps = 1e-6
    fd = float((f(th + eps) - f(th - eps)) / (2 * eps))
    assert abs(g - fd) < 1e-5


@pytest.mark.skipif(jax is None, reason="jax not installed")
def test_jax_jit():
    key = gr.key(3, "jax")

    def f(theta):
        return jnp.sum(theta + jnp.exp(theta) * gr.normal(key, (5,)))

    th = jnp.asarray(0.7)
    assert abs(float(jax.jit(f)(th)) - float(f(th))) < 1e-12
    g = jax.grad(f)
    assert abs(float(jax.jit(g)(th)) - float(g(th))) < 1e-10


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_torch_fixed_noise_grad_normal():
    # torch analog: draw the noise ONCE (a constant tensor) and differentiate
    # the reparameterized transform through it with torch.func.grad.
    key = gr.key(7, "torch")
    noise = gr.normal(key, (4,))  # fixed drawn noise (no grad)

    def f(theta):
        mu = theta**2
        sigma = torch.exp(0.5 * theta)
        return torch.sum(mu + sigma * noise)

    th = torch.tensor(0.3)
    g = float(torch.func.grad(f)(th))
    eps = 1e-6
    fd = float((f(th + eps) - f(th - eps)) / (2 * eps))
    assert abs(g - fd) < 1e-5
    assert float(f(th)) == float(f(th))
    assert float(f(th)) != float(f(th + 0.1))


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_torch_fixed_noise_grad_uniform_inverse_cdf():
    key = gr.key(9, "torch")
    u = gr.uniform(key, (5,))  # fixed drawn uniforms (no grad)

    def f(theta):
        x = -torch.log(1.0 - u) / theta
        return torch.sum(x)

    th = torch.tensor(1.7)
    g = float(torch.func.grad(f)(th))
    eps = 1e-6
    fd = float((f(th + eps) - f(th - eps)) / (2 * eps))
    assert abs(g - fd) < 1e-5


# ----------------------------------------------------------------------------
# streamspraydf pilot: numpy byte-identity + key-threaded backend-array draws
# ----------------------------------------------------------------------------
def _make_spray():
    from galpy.df import fardal15spraydf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import conversion

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    ro, vo = 8.0, 220.0
    return fardal15spraydf(
        2 * 10.0**4.0 / conversion.mass_in_msol(vo, ro),
        progenitor=obs,
        pot=lp,
        tdisrupt=4.5 / conversion.time_in_Gyr(vo, ro),
    )


def test_streamspray_numpy_draws_byte_identical():
    sp = _make_spray()
    # The single changed line: _draw_stripping_dt(None) == the historical draw.
    numpy.random.seed(321)
    got = sp._draw_stripping_dt(20, key=None)
    numpy.random.seed(321)
    ref = numpy.random.uniform(size=20) * sp._tdisrupt
    assert _bytes_equal(got, ref)
    assert not is_backend_array(got)


def test_streamspray_sample_deterministic():
    sp = _make_spray()
    numpy.random.seed(1)
    a = sp.sample(12, return_orbit=False, integrate=False)
    numpy.random.seed(1)
    b = sp.sample(12, return_orbit=False, integrate=False)
    for x, y in zip(a, b):
        numpy.testing.assert_array_equal(numpy.asarray(x), numpy.asarray(y))


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_streamspray_key_draws_reproducible(backend):
    sp = _make_spray()
    k = gr.key(4, backend)
    dt1 = sp._draw_stripping_dt(15, key=k)
    dt2 = sp._draw_stripping_dt(15, key=k)
    assert is_backend_array(dt1)  # reproducible BACKEND array from the key
    numpy.testing.assert_array_equal(_np(dt1), _np(dt2))
    dt3 = sp._draw_stripping_dt(15, key=gr.key(5, backend))
    assert not numpy.allclose(_np(dt1), _np(dt3))  # different key => different draws
