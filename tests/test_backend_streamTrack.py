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
from scipy import interpolate

from galpy.backend import as_numpy, is_backend_array
from galpy.df.streamTrack import (
    _bin_by_tp,
    _closest_point_on_curve,
    _DiffSpline,
    _fit_one_pass_backend,
    _fit_track_from_particles,
    _smooth_series,
)
from galpy.util import coords, galpyWarning

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


# ---------------- PR-5: end-to-end backend stream track ----------------
# _fit_track_from_particles produces a BACKEND (jax/torch) track from BACKEND
# particles, differentiable in the particle values, with the numpy path
# byte-identical. The STRUCTURE (cKDTree closest-point tp_assign, the
# velocity-weight probe, the trim/percentile tp_grid/tp_nodes) is a numpy
# constant; the differentiable path is offsets -> _bin_by_tp -> _smooth_series ->
# track (+ psd_project'd covariance). Because that structure needs concrete
# values, the FORWARD backend track and the TORCH (eager, .detach()) end-to-end
# gradient both work through the full function; a jax.grad, being symbolic,
# cannot pull the cKDTree assignment / scipy-smoother weights to numpy, so the
# rigorous grad checks (FD-vs-AD, jax==torch) exercise the genuine differentiable
# unit -- the offset->bin->smooth->track path with the structure and the
# smoothing weights frozen (the PR-3/PR-4 building blocks composed as
# _fit_one_pass does), which differentiates identically under jax and torch.


def _track_case(seed=7, N=200, M=400, noise=0.02):
    """A well-posed synthetic fit: a gently-curving progenitor arc and particles
    = progenitor-at-random-tp + small gaussian offsets. Returns
    ``(xv (6, N) cyl particles, prog_cart (M, 6), track_t_grid (M,))``."""
    rng = numpy.random.RandomState(seed)
    tg = numpy.linspace(-2.0, 2.0, M)
    theta = 0.35 * tg
    R0 = 1.2
    prog_cart = numpy.column_stack(
        [
            R0 * numpy.cos(theta),
            R0 * numpy.sin(theta),
            0.12 * tg,
            -R0 * 0.35 * numpy.sin(theta),
            R0 * 0.35 * numpy.cos(theta),
            numpy.full(M, 0.12),
        ]
    )
    ps = [
        interpolate.InterpolatedUnivariateSpline(tg, prog_cart[:, i], k=3)
        for i in range(6)
    ]
    tps = rng.uniform(0.1, 1.6, N)  # leading arm (arm_sign=+1)
    part = numpy.column_stack([s(tps) for s in ps]) + noise * rng.randn(N, 6)
    x, y, zc, vx, vy, vzc = part.T
    R, phi, z = coords.rect_to_cyl(x, y, zc)
    vR, vT, vz = coords.rect_to_cyl_vec(vx, vy, vzc, R, phi, z, cyl=True)
    return numpy.array([R, vR, vT, z, vz, phi]), prog_cart, tg


_TRACK_KW = dict(arm_sign=1, ninterp=101, ntp=21, order=2, niter=0, velocity_weight=1.0)


def _relerr(a, b, floor=1e-12):
    a, b = numpy.asarray(a), numpy.asarray(b)
    return float(numpy.max(numpy.abs(a - b) / numpy.maximum(numpy.abs(b), floor)))


def _frozen_track_path(xv, prog_cart, tg):
    """Replicate _fit_track_from_particles' structural setup to expose the frozen
    (numpy) tp_assign/tp_nodes/tp_grid/prog_at and the smoothing weights, and
    return a differentiable ``track(particles_cart, xp)`` closure that reproduces
    _fit_one_pass' mean path (the differentiable unit) with the structure AND the
    smoothing hyperparameters frozen."""
    pc_np = coords.galcencyl_to_galcenrect(*xv)
    ps = [
        interpolate.InterpolatedUnivariateSpline(tg, prog_cart[:, i], k=3)
        for i in range(6)
    ]

    def prog_at(tp):
        tp = numpy.atleast_1d(tp)
        return numpy.column_stack([s(tp) for s in ps])

    sign_mask = tg >= 0
    mask = numpy.broadcast_to(sign_mask[None, :], (pc_np.shape[0], sign_mask.size))
    ta = _closest_point_on_curve(pc_np, prog_cart, tg, mask=mask, velocity_weight=1.0)
    interior = numpy.abs(ta - tg[-1]) > 1e-3 * abs(tg[-1] - tg[0])
    ta, pc_np = ta[interior], pc_np[interior]
    tp_hi = float(numpy.percentile(ta, 99.0))
    tp_grid = numpy.linspace(0.0, tp_hi, 101)
    tp_nodes = numpy.linspace(0.0, tp_hi, 21)
    # sigma weights + frozen effective-s (freeze GCV lambda so FD is a clean
    # frozen-structure gradient, not d(lambda)/d(y))
    off = pc_np - prog_at(ta)
    _, cov0, cnt0 = _bin_by_tp(ta, off, tp_nodes)
    with numpy.errstate(invalid="ignore"):
        per = numpy.sqrt(numpy.diagonal(cov0, axis1=1, axis2=2))
        sig = per / numpy.sqrt(numpy.maximum(cnt0[:, None], 1))
        sig = numpy.where(cnt0[:, None] > 1, sig, numpy.nan)
    means0, _, _ = _bin_by_tp(ta, off, tp_nodes)
    s_user = [
        float(_smooth_series(tp_nodes, means0[:, i], sig[:, i])[1]) for i in range(6)
    ]
    prog_grid = prog_at(tp_grid)

    def track(pc, xp):
        offsets = pc - xp.asarray(prog_at(ta))
        means, _, _ = _bin_by_tp(ta, offsets, tp_nodes)
        splines = [
            _smooth_series(tp_nodes, means[:, i], sig[:, i], s_user=s_user[i])[0]
            for i in range(6)
        ]
        offset_fine = xp.stack([spl(tp_grid) for spl in splines], axis=-1)
        return xp.asarray(prog_grid) + offset_fine

    return pc_np, track


@pytest.mark.parametrize("backend", BACKENDS)
def test_fit_track_backend_parity(backend):
    # a backend (6, N) xv flows through _fit_track_from_particles to a BACKEND
    # track (+ covariance) matching the numpy fit; the structural tp_grid is
    # identical (computed on the numpy view of the particles).
    xv, prog_cart, tg = _track_case()
    fit_np = _fit_track_from_particles(xv, prog_cart, tg, **_TRACK_KW)
    fit_b = _fit_track_from_particles(_arr(backend, xv), prog_cart, tg, **_TRACK_KW)
    assert is_backend_array(fit_b["track_xyz"])
    assert is_backend_array(fit_b["track_vxvyvz"])
    assert is_backend_array(fit_b["cov_xyz"])
    numpy.testing.assert_array_equal(fit_b["tp_grid"], fit_np["tp_grid"])
    numpy.testing.assert_allclose(
        as_numpy(fit_b["track_xyz"]), fit_np["track_xyz"], rtol=1e-6, atol=1e-9
    )
    numpy.testing.assert_allclose(
        as_numpy(fit_b["track_vxvyvz"]), fit_np["track_vxvyvz"], rtol=1e-6, atol=1e-8
    )
    # cov entries span down to ~0 (off-diagonals): compare with an atol scaled to
    # the covariance magnitude rather than a pure rtol.
    cov_scale = numpy.max(numpy.abs(fit_np["cov_xyz"]))
    numpy.testing.assert_allclose(
        as_numpy(fit_b["cov_xyz"]), fit_np["cov_xyz"], rtol=1e-6, atol=1e-6 * cov_scale
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_fit_track_backend_auto_niter(backend):
    # velocity_weight='auto' (probe pass + inner-half sigma resolution) and
    # niter>0 (per-iteration reassignment to the current track) both run on the
    # numpy view of a backend particles_cart and still yield a BACKEND track
    # matching the numpy fit.
    xv, prog_cart, tg = _track_case()
    kw = dict(arm_sign=1, ninterp=101, ntp=21, order=2, niter=1, velocity_weight="auto")
    fit_np = _fit_track_from_particles(xv, prog_cart, tg, **kw)
    fit_b = _fit_track_from_particles(_arr(backend, xv), prog_cart, tg, **kw)
    assert is_backend_array(fit_b["track_xyz"]) and is_backend_array(fit_b["cov_xyz"])
    numpy.testing.assert_array_equal(fit_b["tp_grid"], fit_np["tp_grid"])
    numpy.testing.assert_allclose(
        as_numpy(fit_b["track_xyz"]), fit_np["track_xyz"], rtol=1e-6, atol=1e-9
    )
    cov_scale = numpy.max(numpy.abs(fit_np["cov_xyz"]))
    numpy.testing.assert_allclose(
        as_numpy(fit_b["cov_xyz"]), fit_np["cov_xyz"], rtol=1e-6, atol=1e-6 * cov_scale
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_fit_track_backend_grad(backend):
    # the track is differentiable w.r.t. the particle values: AD of sum(track_xyz)
    # w.r.t. particles_cart (frozen structure + frozen smoothing) is finite &
    # non-zero and matches a finite-difference through the numpy path.
    xv, prog_cart, tg = _track_case()
    pc_np, track = _frozen_track_path(xv, prog_cart, tg)

    def loss(pc, xp):
        return xp.sum(track(pc, xp)[:, 0:3])

    if backend == "jax":
        g = as_numpy(jax.grad(lambda pc: loss(pc, jnp))(jnp.asarray(pc_np)))
    else:
        pt = torch.tensor(pc_np, requires_grad=True)
        loss(pt, torch).backward()
        g = as_numpy(pt.grad)
    assert numpy.isfinite(g).all()
    assert numpy.max(numpy.abs(g)) > 0
    eps = 1e-6
    for i, j in [(3, 0), (30, 1), (70, 0), (120, 2)]:
        a = pc_np.copy()
        a[i, j] += eps
        b = pc_np.copy()
        b[i, j] -= eps
        fd = (
            float(numpy.sum(track(a, numpy)[:, 0:3]))
            - float(numpy.sum(track(b, numpy)[:, 0:3]))
        ) / (2 * eps)
        if abs(g[i, j]) > 1e-3:  # skip components the track is insensitive to
            numpy.testing.assert_allclose(g[i, j], fd, rtol=1e-3, atol=1e-7)


@pytest.mark.skipif(
    "jax" not in BACKENDS or "torch" not in BACKENDS, reason="need both backends"
)
def test_fit_track_backend_grad_jax_torch_agree():
    # jax and torch differentiate the frozen-structure track path identically.
    xv, prog_cart, tg = _track_case()
    pc_np, track = _frozen_track_path(xv, prog_cart, tg)
    gj = as_numpy(
        jax.grad(lambda pc: jnp.sum(track(pc, jnp)[:, 0:3]))(jnp.asarray(pc_np))
    )
    pt = torch.tensor(pc_np, requires_grad=True)
    torch.sum(track(pt, torch)[:, 0:3]).backward()
    gt = as_numpy(pt.grad)
    numpy.testing.assert_allclose(gj, gt, rtol=1e-6, atol=1e-9)


@pytest.mark.skipif("torch" not in BACKENDS, reason="torch not installed")
def test_fit_track_endtoend_torch_grad():
    # End-to-end deliverable: torch (eager) autodiff of sum(track_xyz) w.r.t. the
    # raw cylindrical xv_particles straight through _fit_track_from_particles --
    # the cKDTree structure is frozen (.detach()) and the differentiable track,
    # incl. the psd_project'd covariance, flows through _fit_one_pass_backend.
    # (A symbolic jax.grad cannot trace the cKDTree assignment; hence torch here.)
    xv, prog_cart, tg = _track_case()
    xvt = torch.tensor(xv, requires_grad=True)
    fit_t = _fit_track_from_particles(xvt, prog_cart, tg, **_TRACK_KW)
    assert is_backend_array(fit_t["track_xyz"])
    (torch.sum(fit_t["track_xyz"]) + torch.sum(fit_t["cov_xyz"])).backward()
    g = as_numpy(xvt.grad)
    assert numpy.isfinite(g).all()
    assert numpy.max(numpy.abs(g)) > 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_fit_one_pass_backend_forward(backend):
    # _fit_one_pass_backend directly: a backend particles_cart with a frozen numpy
    # structure reproduces the numpy _fit_one_pass one-pass fit (mean, velocity,
    # and covariance) and returns backend arrays.
    from galpy.df.streamTrack import _fit_one_pass

    xv, prog_cart, tg = _track_case()
    pc_np, _ = _frozen_track_path(xv, prog_cart, tg)
    ps = [
        interpolate.InterpolatedUnivariateSpline(tg, prog_cart[:, i], k=3)
        for i in range(6)
    ]

    def prog_at(tp):
        tp = numpy.atleast_1d(tp)
        return numpy.column_stack([s(tp) for s in ps])

    sign_mask = tg >= 0
    mask = numpy.broadcast_to(sign_mask[None, :], (pc_np.shape[0], sign_mask.size))
    ta = _closest_point_on_curve(pc_np, prog_cart, tg, mask=mask, velocity_weight=1.0)
    tp_hi = float(numpy.percentile(ta, 99.0))
    tp_grid = numpy.linspace(0.0, tp_hi, 101)
    tp_nodes = numpy.linspace(0.0, tp_hi, 21)
    s_um, s_uc = [None] * 6, [None] * 21
    xyz_n, vel_n, cov_n, s_n = _fit_one_pass(
        pc_np, ta, tp_nodes, tp_grid, prog_at, 2, s_um, s_uc
    )
    xyz_b, vel_b, cov_b, s_b = _fit_one_pass_backend(
        _arr(backend, pc_np), ta, tp_nodes, tp_grid, prog_at, 2, s_um, s_uc
    )
    assert is_backend_array(xyz_b) and is_backend_array(cov_b)
    numpy.testing.assert_allclose(as_numpy(xyz_b), xyz_n, rtol=1e-6, atol=1e-9)
    numpy.testing.assert_allclose(as_numpy(vel_b), vel_n, rtol=1e-6, atol=1e-8)
    cov_scale = numpy.max(numpy.abs(cov_n))
    numpy.testing.assert_allclose(
        as_numpy(cov_b), cov_n, rtol=1e-6, atol=1e-6 * cov_scale
    )
