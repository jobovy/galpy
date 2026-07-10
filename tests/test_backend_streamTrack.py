###############################################################################
# test_backend_streamTrack.py: backend-agnostic primitives of the stream-track
# assembly (galpy.df.streamTrack).
#
# PR-3: _bin_by_tp (segment mean/cov of per-particle offsets, differentiable in
# the VALUES; the bin assignment is a numpy structural index) and
# _closest_point_on_curve (a non-differentiable cKDTree index/time assignment
# that accepts backend-array inputs).
#
# PR-4: _smooth_series (the offset smoother) differentiable for backend y. The
# fit matches scipy exactly (make_smoothing_spline GCV path, UnivariateSpline
# FITPACK reuse paths, and the <5-point linear fallback), expressed as a frozen
# linear operator y -> fit; the smoothing structure (lambda / knots / p /
# weights) is a stop-gradient hyperparameter, so the gradient is d(fit)/d(y) at
# fixed smoothing.
###############################################################################
import numpy
import pytest

from galpy.backend import as_numpy, is_backend_array
from galpy.df.streamTrack import (
    _bin_by_tp,
    _closest_point_on_curve,
    _DiffSpline,
    _smooth_series,
)
from galpy.util import galpyWarning

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


# ---------------- PR-4: differentiable _smooth_series ----------------
def _smoother_case(seed=11, n=40):
    # small internal-unit scale (~1e-5) like galpy's binned offset series, which
    # is the regime that stresses make_smoothing_spline's GCV (yscale rescaling).
    rng = numpy.random.RandomState(seed)
    x = numpy.unique(numpy.sort(rng.uniform(-5.0, 5.0, n)))
    m = len(x)
    y = numpy.sin(x) * 1e-5 + 0.2e-5 * rng.randn(m)
    sigma = (0.2 + 0.1 * rng.rand(m)) * 1e-5
    return x, y, sigma


# (name -> kwargs); "s_user" is resolved per-case from the GCV effective_s so it
# exercises the round-trip-reuse path.
_SMOOTH_CFGS = {
    "gcv": {},
    "fitpack_factor": {"smoothing_factor": 2.0},
    "s_user": {"s_user": "gcv"},
}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("cfgname", list(_SMOOTH_CFGS))
def test_smooth_series_backend_parity(backend, cfgname):
    # the backend fit reproduces the numpy _smooth_series (scipy) fit at the
    # query grid -- including EXTRAPOLATION beyond the data range -- and the
    # returned effective_s matches; the fit is a backend array.
    x, y, sigma = _smoother_case()
    cfg = dict(_SMOOTH_CFGS[cfgname])
    if cfg.get("s_user") == "gcv":
        cfg = {"s_user": _smooth_series(x, y, sigma)[1]}
    ref_spl, ref_es = _smooth_series(x, y, sigma, **cfg)
    grid = numpy.linspace(-5.3, 5.3, 41)  # extends past the data -> extrapolation
    ref = ref_spl(grid)
    spl, es = _smooth_series(x, _arr(backend, y), sigma, **cfg)
    got = spl(grid)
    assert is_backend_array(got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-9, atol=1e-16)
    numpy.testing.assert_allclose(float(as_numpy(es)), ref_es, rtol=1e-9, atol=1e-14)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("nval", [1, 3, 4])
def test_smooth_series_backend_linear_fallback(backend, nval):
    # fewer than 5 valid points -> linear-interp/constant fallback, backend fit
    # matches the numpy interp1d(extrapolate) result.
    x = numpy.linspace(-2.0, 2.0, nval)
    y = numpy.array([0.1, -0.2, 0.05, 0.3])[:nval] if nval > 1 else numpy.array([0.42])
    sigma = numpy.full(nval, 0.1)
    grid = numpy.linspace(-3.0, 3.0, 20)
    ref = _smooth_series(x, y, sigma)[0](grid)
    got = _smooth_series(x, _arr(backend, y), sigma)[0](grid)
    assert is_backend_array(got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("cfgname", ["gcv", "fitpack_factor"])
def test_smooth_series_backend_grad(backend, cfgname):
    # AD (jax.grad / torch autograd) differentiates the frozen smoothing operator
    # -- d(sum fit)/d(y) equals the column sum of that operator. (A naive FD that
    # re-fits the smoother would instead pick up the frozen hyperparameter's
    # response, d(lambda)/d(y).) The operator itself is validated against scipy by
    # test_smooth_series_backend_parity, so this pins the AD path.
    x, y, sigma = _smoother_case()
    cfg = dict(_SMOOTH_CFGS[cfgname])
    grid = numpy.linspace(-4.5, 4.5, 33)
    op = _DiffSpline(
        y, x, sigma, cfg.get("s_user"), cfg.get("smoothing_factor", 1.0), "numpy"
    )
    expected = op._build(grid)(y).sum(axis=0)
    if backend == "jax":

        def f(yv):
            spl, _ = _smooth_series(x, yv, sigma, **cfg)
            return jnp.sum(spl(grid))

        g = as_numpy(jax.grad(f)(jnp.asarray(y)))
    else:
        yt = torch.tensor(y, requires_grad=True)
        spl, _ = _smooth_series(x, yt, sigma, **cfg)
        torch.sum(spl(grid)).backward()
        g = as_numpy(yt.grad)
    numpy.testing.assert_allclose(g, expected, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("backend", BACKENDS)
def test_smooth_series_backend_edges(backend):
    # degenerate regimes still reproduce the numpy fit: all-invalid sigma
    # (sig_med fallback), very large smoothing (0 interior knots -> FITPACK
    # weighted-LSQ branch), and a rough fit with many interior knots (the
    # augmented-QR reconstruction stays exact where the normal equations would
    # be ill-conditioned). A smoothing_factor far below ~0.1 drives FITPACK into
    # its non-converging near-interpolation regime and is out of scope.
    rng = numpy.random.RandomState(5)
    x = numpy.unique(numpy.sort(rng.uniform(-5.0, 5.0, 30)))
    m = len(x)
    grid = numpy.linspace(-5.2, 5.2, 37)
    ybase = numpy.sin(x) * 1e-5 + 0.2e-5 * rng.randn(m)
    sig = (0.2 + 0.1 * rng.rand(m)) * 1e-5
    cases = [
        ("bad_sigma", ybase, numpy.zeros(m), {}),
        ("very_smooth", ybase, sig, {"smoothing_factor": 500.0}),
        ("many_knots", ybase, sig, {"smoothing_factor": 0.1}),
    ]
    for name, y, s, cfg in cases:
        ref = _smooth_series(x, y, s, **cfg)[0](grid)
        got = _smooth_series(x, _arr(backend, y), s, **cfg)[0](grid)
        scale = max(1e-30, float(numpy.max(numpy.abs(ref))))
        numpy.testing.assert_allclose(
            as_numpy(got),
            ref,
            rtol=1e-7,
            atol=1e-8 * scale,
            err_msg=f"{name}/{backend}",
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_smooth_series_backend_constant_y(backend):
    # exactly-constant y triggers the yscale->1 guard; the numpy GCV path returns
    # a constant spline (O(1) scale, moderate sigma) and the backend must match.
    # A backend-only crash here would be a regression against the numpy path.
    rng = numpy.random.RandomState(2)
    x = numpy.unique(numpy.sort(rng.uniform(0.0, 10.0, 14)))
    m = len(x)
    y = numpy.full(m, 3.0)
    sigma = numpy.full(m, 0.05)
    grid = numpy.linspace(-1.0, 11.0, 25)
    ref = _smooth_series(x, y, sigma)[0](grid)
    got = _smooth_series(x, _arr(backend, y), sigma)[0](grid)
    assert is_backend_array(got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-7, atol=1e-8)


@pytest.mark.skipif("jax" not in BACKENDS, reason="jax not installed")
@pytest.mark.parametrize("cfgname", ["gcv", "fitpack_factor"])
def test_smooth_series_backend_jit(cfgname):
    # jax.jit(forward) and jit(grad) both work: the frozen operator's fixed
    # (n_query x n_data) shape hides the variable internal knot count from the
    # trace, so the scipy structure runs in a pure_callback under jit.
    x, y, sigma = _smoother_case()
    cfg = dict(_SMOOTH_CFGS[cfgname])
    grid = numpy.linspace(-4.5, 4.5, 25)
    ref = _smooth_series(x, y, sigma, **cfg)[0](grid)
    jfwd = jax.jit(lambda yv: _smooth_series(x, yv, sigma, **cfg)[0](grid))
    numpy.testing.assert_allclose(
        as_numpy(jfwd(jnp.asarray(y))), ref, rtol=1e-9, atol=1e-16
    )
    op = _DiffSpline(
        y, x, sigma, cfg.get("s_user"), cfg.get("smoothing_factor", 1.0), "numpy"
    )
    expected = op._build(grid)(y).sum(axis=0)
    jgrad = jax.jit(
        jax.grad(lambda yv: jnp.sum(_smooth_series(x, yv, sigma, **cfg)[0](grid)))
    )
    numpy.testing.assert_allclose(
        as_numpy(jgrad(jnp.asarray(y))), expected, rtol=1e-8, atol=1e-10
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_smooth_series_backend_near_interp_warns(backend, monkeypatch):
    # When FITPACK's fit corresponds to no single penalized-p form (the
    # near-interpolation regime), the backend cannot reconstruct it and WARNS
    # instead of silently returning an imprecise fit. Which real inputs trigger
    # this depends on FITPACK's (scipy-version-dependent) knot selection, so the
    # condition is forced deterministically: perturb FITPACK's coefficients away
    # from any penalized solution and confirm the warning fires. The FITPACK
    # reconstruction now lives in galpy.backend.interpolate, and the warn fires
    # when the differentiable operator is built (i.e. on the first spline
    # evaluation), so patch the backend scipy handle and evaluate the fit.
    from galpy.backend import interpolate as backend_interpolate

    real_us = backend_interpolate._scipy_interpolate.UnivariateSpline

    class _PerturbedSpline(real_us):
        def get_coeffs(self):
            c = numpy.array(real_us.get_coeffs(self), dtype=float)
            if c.size > 2:
                c[c.size // 2] += 0.5 * numpy.max(numpy.abs(c)) + 1e-3
            return c

    monkeypatch.setattr(
        backend_interpolate._scipy_interpolate, "UnivariateSpline", _PerturbedSpline
    )
    x, y, sigma = _smoother_case(n=20)
    with pytest.warns(galpyWarning, match="near-interpolation"):
        spl, _ = _smooth_series(x, _arr(backend, y), sigma, s_user=1.0)
        spl(numpy.linspace(-5.0, 5.0, 12))
