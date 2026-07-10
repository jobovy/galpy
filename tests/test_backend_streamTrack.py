###############################################################################
# test_backend_streamTrack.py: backend-agnostic primitives of the stream-track
# assembly (galpy.df.streamTrack). PR-3 of the streamspray/streamTrack migration:
# _bin_by_tp (segment mean/cov of per-particle offsets, differentiable in the
# VALUES; the bin assignment is a numpy structural index) and
# _closest_point_on_curve (a non-differentiable cKDTree index/time assignment
# that accepts backend-array inputs). The smoother (_smooth_series) is a later PR.
###############################################################################
import numpy
import pytest

from galpy.backend import as_numpy, is_backend_array
from galpy.df.streamTrack import _bin_by_tp, _closest_point_on_curve

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


def _arr(backend, x):
    return jnp.asarray(x) if backend == "jax" else torch.tensor(x)


def _case(seed=3, n=60, d=6, m=8):
    rng = numpy.random.RandomState(seed)
    tp_nodes = numpy.linspace(-5.0, 5.0, m)
    tp_assign = rng.uniform(-5.0, 5.0, n)
    values = rng.randn(n, d)
    return tp_assign, values, tp_nodes


@pytest.mark.parametrize("backend", BACKENDS)
def test_bin_by_tp_backend_parity(backend):
    # the segment mean/cov match the numpy loop to machine precision, the bin
    # counts are identical, and the mean/cov are backend arrays (differentiable).
    tp_assign, values, tp_nodes = _case()
    m_np, c_np, cnt_np = _bin_by_tp(tp_assign, values, tp_nodes)
    m_b, c_b, cnt_b = _bin_by_tp(tp_assign, _arr(backend, values), tp_nodes)
    assert is_backend_array(m_b) and is_backend_array(c_b)
    numpy.testing.assert_array_equal(as_numpy(cnt_b), cnt_np)
    fin = numpy.isfinite(m_np)  # k>=2 bins
    numpy.testing.assert_allclose(as_numpy(m_b)[fin], m_np[fin], rtol=1e-12, atol=1e-13)
    numpy.testing.assert_allclose(
        as_numpy(c_b), numpy.nan_to_num(c_np), rtol=1e-12, atol=1e-13
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_bin_by_tp_backend_grad(backend):
    # the binned mean is differentiable w.r.t. the input particle values.
    tp_assign, values, tp_nodes = _case(n=40)

    def f_np(v):
        m, _, _ = _bin_by_tp(tp_assign, v, tp_nodes)
        return numpy.nan_to_num(m).sum()

    eps = 1e-6
    j = 5
    gfd = (
        f_np(values + eps * numpy.eye(values.size)[j].reshape(values.shape))
        - f_np(values - eps * numpy.eye(values.size)[j].reshape(values.shape))
    ) / (2 * eps)
    if backend == "jax":

        def f(v):
            m, _, _ = _bin_by_tp(tp_assign, v, tp_nodes)
            return jnp.nan_to_num(m).sum()

        g = as_numpy(jax.grad(f)(jnp.asarray(values))).reshape(-1)[j]
    else:
        v = torch.tensor(values, requires_grad=True)
        m, _, _ = _bin_by_tp(tp_assign, v, tp_nodes)
        torch.nan_to_num(m).sum().backward()
        g = as_numpy(v.grad).reshape(-1)[j]
    numpy.testing.assert_allclose(g, gfd, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_closest_point_backend_inputs(backend):
    # cKDTree assignment accepts backend-array points/curve and returns the numpy
    # (non-differentiable) time assignment, identical to the numpy inputs.
    rng = numpy.random.RandomState(4)
    points = rng.randn(30, 6)
    curve = rng.randn(8, 6)
    curve_t = numpy.linspace(-5.0, 5.0, 8)
    ref = _closest_point_on_curve(points, curve, curve_t)
    got = _closest_point_on_curve(_arr(backend, points), _arr(backend, curve), curve_t)
    assert isinstance(got, numpy.ndarray)
    numpy.testing.assert_array_equal(got, ref)
    # velocity_weight path (D==6) also accepts backend inputs
    got_vw = _closest_point_on_curve(
        _arr(backend, points), _arr(backend, curve), curve_t, velocity_weight=2.0
    )
    ref_vw = _closest_point_on_curve(points, curve, curve_t, velocity_weight=2.0)
    numpy.testing.assert_array_equal(got_vw, ref_vw)
