###############################################################################
# test_backend_interpolate.py: galpy.backend.interpolate -- backend-agnostic
# interpolation. Asserts (1) numpy path byte-identical to scipy; (2) jax/torch
# value parity vs scipy to ~1e-9; (3) autodiff in the eval point AND -- the key
# new capability -- in the table y-VALUES (so gradients flow to the parameters
# that build a table, e.g. a dynamical-friction sigma_r(r)).
###############################################################################
import numpy
import pytest
import scipy.interpolate as si
import scipy.ndimage as sndi

from galpy.backend.interpolate import (
    MapCoordinates,
    Spline1D,
    Spline2D,
    cubic_spline_coeffs,
    eval_cubic,
    interp_linear,
    map_coordinates,
    spline_filter,
)

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
    import array_api_compat.torch as txp

    BACKENDS.append("torch")
except ImportError:  # pragma: no cover
    torch = None

AD_BACKENDS = [b for b in BACKENDS if b != "numpy"]


def _xp(backend):
    return {
        "numpy": numpy,
        "jax": jnp if jax else None,
        "torch": txp if torch else None,
    }[backend]


def _asarray(backend, x, requires_grad=False):
    if backend == "numpy":
        return numpy.asarray(x, dtype=float)
    if backend == "jax":
        return jnp.asarray(x, dtype=jnp.float64)
    return torch.tensor(x, dtype=torch.float64, requires_grad=requires_grad)


def _tonumpy(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().numpy()
    return numpy.asarray(x)


_XG = numpy.linspace(0.3, 6.0, 25)
_YG = numpy.sin(_XG) + 0.2 * _XG
_RQ = numpy.array([0.3, 1.1, 2.7, 4.5, 6.0])  # in-range incl. endpoints


@pytest.mark.parametrize("backend", BACKENDS)
def test_spline1d_frozen_parity(backend):
    # mode 1 (frozen scipy spline, the interpRZ-like usage): build ONCE from
    # numpy, then evaluate under each backend -> numpy byte-identical, jax/torch
    # match the frozen scipy spline to ~1e-9.
    s = Spline1D(_XG, _YG, k=3)  # numpy y -> mode 1
    ref = si.InterpolatedUnivariateSpline(_XG, _YG, k=3)(_RQ)
    got = _tonumpy(s(_asarray(backend, _RQ)))
    rtol = 0.0 if backend == "numpy" else 1e-9  # numpy byte-identical
    numpy.testing.assert_allclose(got, ref, rtol=rtol, atol=1e-12)


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_spline1d_mode2_self_consistent(backend):
    # mode 2 (in-backend cubic, the differentiable usage): the SAME instance must
    # give the same values whether queried with numpy or a backend array, and
    # match scipy CubicSpline(natural).
    y_b = _asarray(backend, _YG)
    s = Spline1D(_XG, y_b, k=3, bc="natural")  # backend y -> mode 2
    ref = si.CubicSpline(_XG, _YG, bc_type="natural")(_RQ)
    got = _tonumpy(s(_asarray(backend, _RQ)))
    numpy.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)
    # numpy query of the same mode-2 instance agrees with the backend query
    numpy.testing.assert_allclose(_tonumpy(s(_RQ)), got, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_cubic_and_linear_parity(backend):
    xp = _xp(backend)
    x, y, r = _asarray(backend, _XG), _asarray(backend, _YG), _asarray(backend, _RQ)
    c = cubic_spline_coeffs(xp, x, y, bc="natural")
    cub = _tonumpy(eval_cubic(xp, x, c, r))
    refc = si.CubicSpline(_XG, _YG, bc_type="natural")(_RQ)
    numpy.testing.assert_allclose(cub, refc, rtol=1e-9, atol=1e-12)
    lin = _tonumpy(interp_linear(xp, x, y, r))
    numpy.testing.assert_allclose(
        lin, numpy.interp(_RQ, _XG, _YG), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_spline2d_value_parity(backend):
    xg = numpy.linspace(0.0, 3.0, 12)
    yg = numpy.linspace(-1.0, 2.0, 10)
    zz = numpy.outer(numpy.sin(xg), numpy.cos(yg)) + 0.1 * xg[:, None]
    spl = si.RectBivariateSpline(xg, yg, zz)
    X = numpy.array([0.2, 1.5, 2.8])
    Y = numpy.array([-0.5, 0.7, 1.9])
    ref = spl.ev(X, Y)
    s = Spline2D(
        x=_asarray(backend, xg), y=_asarray(backend, yg), z=_asarray(backend, zz)
    )
    got = _tonumpy(s(_asarray(backend, X), _asarray(backend, Y)))
    rtol = 0.0 if backend == "numpy" else 1e-9
    numpy.testing.assert_allclose(got, ref, rtol=rtol, atol=1e-12)


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_grad_in_eval_point(backend):
    r0 = 2.7
    ref = si.CubicSpline(_XG, _YG, bc_type="natural").derivative()(r0)
    xp = _xp(backend)
    if backend == "jax":
        ad = float(
            jax.grad(lambda r: Spline1D(jnp.asarray(_XG), jnp.asarray(_YG), k=3)(r))(
                jnp.asarray(r0)
            )
        )
    else:
        rt = torch.tensor(r0, requires_grad=True)
        Spline1D(txp.asarray(_XG), txp.asarray(_YG), k=3)(rt).backward()
        ad = float(rt.grad)
    numpy.testing.assert_allclose(ad, ref, rtol=1e-6)


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_grad_in_table_values(backend):
    # THE key capability: d(spline value)/d(y) -- lets gradients flow to the
    # parameters that built a table (e.g. dynamical-friction sigma_r(r)).
    r0 = 2.7
    fd = numpy.empty_like(_YG)
    for i in range(len(_XG)):
        yp = _YG.copy()
        yp[i] += 1e-6
        ym = _YG.copy()
        ym[i] -= 1e-6
        fd[i] = (
            si.CubicSpline(_XG, yp, bc_type="natural")(r0)
            - si.CubicSpline(_XG, ym, bc_type="natural")(r0)
        ) / 2e-6
    if backend == "jax":
        g = numpy.asarray(
            jax.grad(lambda y: Spline1D(jnp.asarray(_XG), y, k=3)(jnp.asarray(r0)))(
                jnp.asarray(_YG)
            )
        )
    else:
        yt = torch.tensor(_YG, requires_grad=True)
        Spline1D(txp.asarray(_XG), yt, k=3)(txp.asarray(r0)).backward()
        g = yt.grad.numpy()
    numpy.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-8)


# out-of-range query points (below x[0], above x[-1], and one in-range)
_ROUT = numpy.array([-1.0, 0.1, 2.7, 7.0, 10.0])


@pytest.mark.parametrize("ext", ["clip", "const", 3])
@pytest.mark.parametrize("backend", BACKENDS)
def test_eval_ppoly_clamp_modes(backend, ext):
    # 'clip'/'const'/3 all clamp the eval point -> edge VALUE outside the range,
    # which is byte-identical (numpy) / ~1e-9 (jax/torch) to scipy ext=3.
    from galpy.backend.interpolate import eval_ppoly, spline_to_ppoly

    spl0 = si.InterpolatedUnivariateSpline(_XG, _YG, k=3, ext=0)
    x, c = spline_to_ppoly(spl0)
    ref = si.InterpolatedUnivariateSpline(_XG, _YG, k=3, ext=3)(_ROUT)
    xp = _xp(backend)
    got = _tonumpy(
        eval_ppoly(
            xp,
            _asarray(backend, x),
            _asarray(backend, c),
            _asarray(backend, _ROUT),
            extrapolate=ext,
        )
    )
    rtol = 0.0 if backend == "numpy" else 1e-9
    numpy.testing.assert_allclose(got, ref, rtol=rtol, atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_eval_ppoly_extrapolate_true(backend):
    # extrapolate=True (default) extends the edge polynomial (scipy ext=0).
    from galpy.backend.interpolate import eval_ppoly, spline_to_ppoly

    spl0 = si.InterpolatedUnivariateSpline(_XG, _YG, k=3, ext=0)
    x, c = spline_to_ppoly(spl0)
    ref = spl0(_ROUT)
    xp = _xp(backend)
    got = _tonumpy(
        eval_ppoly(
            xp, _asarray(backend, x), _asarray(backend, c), _asarray(backend, _ROUT)
        )
    )
    rtol = 0.0 if backend == "numpy" else 1e-9
    numpy.testing.assert_allclose(got, ref, rtol=rtol, atol=1e-12)


def test_eval_ppoly_bad_extrapolate():
    from galpy.backend.interpolate import eval_ppoly, spline_to_ppoly

    spl0 = si.InterpolatedUnivariateSpline(_XG, _YG, k=3)
    x, c = spline_to_ppoly(spl0)
    with pytest.raises(ValueError):
        eval_ppoly(numpy, x, c, _RQ, extrapolate="nope")


@pytest.mark.parametrize("backend", BACKENDS)
def test_cubic_not_a_knot(backend):
    # bc='not-a-knot' matches scipy CubicSpline's DEFAULT (byte-identical numpy).
    xp = _xp(backend)
    x, y, r = _asarray(backend, _XG), _asarray(backend, _YG), _asarray(backend, _RQ)
    c = cubic_spline_coeffs(xp, x, y, bc="not-a-knot")
    got = _tonumpy(eval_cubic(xp, x, c, r))
    ref = si.CubicSpline(_XG, _YG)(_RQ)  # default bc_type = 'not-a-knot'
    rtol = 1e-12 if backend == "numpy" else 1e-9
    numpy.testing.assert_allclose(got, ref, rtol=rtol, atol=1e-12)


def test_cubic_spline_coeffs_errors():
    with pytest.raises(ValueError):
        cubic_spline_coeffs(numpy, _XG[:2], _YG[:2])  # n < 3
    with pytest.raises(ValueError):
        cubic_spline_coeffs(numpy, _XG, _YG, bc="bogus")


@pytest.mark.parametrize("ext", ["clip", "const", 3])
@pytest.mark.parametrize("backend", BACKENDS)
def test_interp_linear_clamp_modes(backend, ext):
    # 'clip'/'const'/3 clamp the eval point -> edge value beyond the ends.
    xp = _xp(backend)
    x, y, r = _asarray(backend, _XG), _asarray(backend, _YG), _asarray(backend, _ROUT)
    got = _tonumpy(interp_linear(xp, x, y, r, extrapolate=ext))
    rclamp = numpy.clip(_ROUT, _XG[0], _XG[-1])
    ref = numpy.interp(rclamp, _XG, _YG)
    rtol = 1e-12 if backend == "numpy" else 1e-12
    numpy.testing.assert_allclose(got, ref, rtol=rtol, atol=1e-12)


def test_interp_linear_bad_extrapolate():
    with pytest.raises(ValueError):
        interp_linear(numpy, _XG, _YG, _RQ, extrapolate="nope")


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_spline1d_mode2_linear(backend):
    # mode 2 with k=1: in-backend piecewise-linear; numpy AND backend queries of
    # the same instance both agree with numpy.interp.
    y_b = _asarray(backend, _YG)
    s = Spline1D(_XG, y_b, k=1)  # backend y, k=1 -> mode-2 linear (no scipy spline)
    ref = numpy.interp(_RQ, _XG, _YG)
    got = _tonumpy(s(_asarray(backend, _RQ)))
    numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)
    # numpy query of the same mode-2 k=1 instance (interp_linear numpy branch)
    numpy.testing.assert_allclose(_tonumpy(s(_RQ)), ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_spline1d_mode2_bad_k(backend):
    y_b = _asarray(backend, _YG)
    with pytest.raises(ValueError):
        Spline1D(_XG, y_b, k=2)  # mode-2 supports only k=1 or k=3


@pytest.mark.parametrize("backend", BACKENDS)
def test_spline1d_ext3_const(backend):
    # ext=3 maps to the 'const' clamp: numpy byte-identical to scipy ext=3, and
    # jax/torch return the edge value beyond the ends.
    s = Spline1D(_XG, _YG, k=3, ext=3)
    ref = si.InterpolatedUnivariateSpline(_XG, _YG, k=3, ext=3)(_ROUT)
    got = _tonumpy(s(_asarray(backend, _ROUT)))
    rtol = 0.0 if backend == "numpy" else 1e-9
    numpy.testing.assert_allclose(got, ref, rtol=rtol, atol=1e-12)


def _grid_spline():
    xg = numpy.linspace(0.0, 3.0, 12)
    yg = numpy.linspace(-1.0, 2.0, 10)
    zz = numpy.outer(numpy.sin(xg), numpy.cos(yg)) + 0.1 * xg[:, None]
    return xg, yg, zz


@pytest.mark.parametrize("ext", ["clip", "const", 3])
@pytest.mark.parametrize("backend", BACKENDS)
def test_eval_rect_ppoly_clamp_modes(backend, ext):
    # 2D 'clip'/'const'/3 clamp (X,Y) to the grid -> edge value == scipy .ev at the
    # clamped point.
    from galpy.backend.interpolate import eval_rect_ppoly, rect_bivariate_to_ppoly

    xg, yg, zz = _grid_spline()
    spl = si.RectBivariateSpline(xg, yg, zz)
    xbr, ybr, c = rect_bivariate_to_ppoly(spl)
    X = numpy.array([-1.0, 1.5, 5.0])
    Y = numpy.array([-3.0, 0.7, 4.0])
    ref = spl.ev(numpy.clip(X, xg[0], xg[-1]), numpy.clip(Y, yg[0], yg[-1]))
    xp = _xp(backend)
    got = _tonumpy(
        eval_rect_ppoly(
            xp,
            _asarray(backend, xbr),
            _asarray(backend, ybr),
            _asarray(backend, c),
            _asarray(backend, X),
            _asarray(backend, Y),
            extrapolate=ext,
        )
    )
    rtol = 0.0 if backend == "numpy" else 1e-9
    numpy.testing.assert_allclose(got, ref, rtol=rtol, atol=1e-12)


def test_eval_rect_ppoly_bad_extrapolate():
    from galpy.backend.interpolate import eval_rect_ppoly, rect_bivariate_to_ppoly

    xg, yg, zz = _grid_spline()
    spl = si.RectBivariateSpline(xg, yg, zz)
    xbr, ybr, c = rect_bivariate_to_ppoly(spl)
    with pytest.raises(ValueError):
        eval_rect_ppoly(
            numpy,
            xbr,
            ybr,
            c,
            numpy.array([1.0]),
            numpy.array([0.0]),
            extrapolate="nope",
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_spline2d_from_prefitted_spl(backend):
    # Spline2D(spl=...) reuses a pre-fitted RectBivariateSpline instead of
    # re-fitting; numpy path byte-identical to .ev.
    xg, yg, zz = _grid_spline()
    spl = si.RectBivariateSpline(xg, yg, zz)
    X = numpy.array([0.2, 1.5, 2.8])
    Y = numpy.array([-0.5, 0.7, 1.9])
    ref = spl.ev(X, Y)
    s = Spline2D(spl=spl)
    got = _tonumpy(s(_asarray(backend, X), _asarray(backend, Y)))
    rtol = 0.0 if backend == "numpy" else 1e-9
    numpy.testing.assert_allclose(got, ref, rtol=rtol, atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_spline2d_ext3_const(backend):
    # Spline2D ext=3 clamps (X,Y) to the grid (edge value beyond it).
    xg, yg, zz = _grid_spline()
    spl = si.RectBivariateSpline(xg, yg, zz)
    s = Spline2D(x=xg, y=yg, z=zz, ext=3)
    X = numpy.array([-1.0, 1.5, 5.0])
    Y = numpy.array([-3.0, 0.7, 4.0])
    ref = spl.ev(numpy.clip(X, xg[0], xg[-1]), numpy.clip(Y, yg[0], yg[-1]))
    got = _tonumpy(s(_asarray(backend, X), _asarray(backend, Y)))
    rtol = 0.0 if backend == "numpy" else 1e-9
    numpy.testing.assert_allclose(got, ref, rtol=rtol, atol=1e-12)


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_spline2d_mixed_backend_args(backend):
    # X a plain scalar (NOT a backend array), Y a backend array -> the backend
    # path selects Y as the namespace reference (the `else Y` ref-pick branch)
    # and still matches scipy .ev.
    xg, yg, zz = _grid_spline()
    spl = si.RectBivariateSpline(xg, yg, zz)
    s = Spline2D(x=xg, y=yg, z=zz)
    ref = spl.ev([1.5], [0.7])
    got = _tonumpy(s(1.5, _asarray(backend, [0.7])))
    numpy.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)


###############################################################################
# Action-angle-grid prerequisite primitives: the EXACT consumer contract for
# migrating actionAngleStaeckelGrid + actionAngleAdiabaticGrid. These assert the
# backend eval matches scipy to MACHINE precision (~1e-12, in practice ~1e-14)
# on RANDOM NON-UNIFORM 1D grids and REGULAR 2D grids, plus the ndimage cubic
# map_coordinates, plus grad-vs-FD of an interpolated value w.r.t. a query
# coordinate -- the three primitives the grids need.
###############################################################################
# Random NON-UNIFORM 1D grid (the InterpolatedUnivariateSpline contract; the
# grids build splines on non-uniform Lzs / RL abscissae).
_rng = numpy.random.RandomState(20240607)
_X1D = numpy.sort(_rng.uniform(0.05, 9.5, 24))
_X1D[0], _X1D[-1] = 0.05, 9.5  # pin the ends for clean in-range queries
_Y1D = numpy.sin(1.3 * _X1D) + 0.2 * _X1D - 0.05 * _X1D**2
_Q1D = numpy.array([0.05, 0.9, 2.4, 4.7, 6.6, 9.5])  # in-range incl. endpoints


@pytest.mark.parametrize("backend", BACKENDS)
def test_iuspline_value_and_deriv_vs_scipy(backend):
    # 1D cubic InterpolatedUnivariateSpline: VALUE and 1st DERIVATIVE (nu=1) on a
    # random non-uniform grid. numpy byte-identical; jax/torch ~1e-12 vs scipy.
    ref = si.InterpolatedUnivariateSpline(_X1D, _Y1D, k=3)
    s = Spline1D(_X1D, _Y1D, k=3)
    q = _asarray(backend, _Q1D)
    rtol = 0.0 if backend == "numpy" else 1e-12
    val = _tonumpy(s(q))
    numpy.testing.assert_allclose(val, ref(_Q1D), rtol=rtol, atol=1e-13)
    dval = _tonumpy(s(q, nu=1))
    numpy.testing.assert_allclose(dval, ref(_Q1D, nu=1), rtol=rtol, atol=1e-13)


def test_iuspline_numpy_byte_identical():
    # numpy path is a LITERAL scipy passthrough: value AND derivative are
    # bit-identical to the bare scipy spline.
    ref = si.InterpolatedUnivariateSpline(_X1D, _Y1D, k=3)
    s = Spline1D(_X1D, _Y1D, k=3)
    numpy.testing.assert_array_equal(s(_Q1D), ref(_Q1D))
    numpy.testing.assert_array_equal(s(_Q1D, nu=1), ref(_Q1D, nu=1))


# Regular 2D grid (the RectBivariateSpline contract: logu0 / jz / jr tables are
# built on regular linspace grids).
def _rect2d():
    xg = numpy.linspace(0.05, 9.5, 18)
    yg = numpy.linspace(0.0, 1.0, 15)
    Z = numpy.log(
        1.0
        + numpy.outer(numpy.cos(0.7 * xg), numpy.sin(1.1 * yg)) ** 2
        + 0.3 * xg[:, None]
    )
    return xg, yg, Z


_QX2D = numpy.array([0.2, 1.5, 4.3, 7.8, 9.4])
_QY2D = numpy.array([0.02, 0.31, 0.55, 0.77, 0.98])


@pytest.mark.parametrize("backend", BACKENDS)
def test_rectbivariate_pointeval_vs_scipy(backend):
    # 2D cubic RectBivariateSpline point-eval (grid=False / .ev) on a regular
    # grid. numpy byte-identical; jax/torch ~1e-12 vs scipy.ev.
    xg, yg, Z = _rect2d()
    spl = si.RectBivariateSpline(xg, yg, Z, kx=3, ky=3, s=0.0)
    s = Spline2D(x=xg, y=yg, z=Z)
    got = _tonumpy(s(_asarray(backend, _QX2D), _asarray(backend, _QY2D), grid=False))
    rtol = 0.0 if backend == "numpy" else 1e-12
    numpy.testing.assert_allclose(got, spl.ev(_QX2D, _QY2D), rtol=rtol, atol=1e-13)


def test_rectbivariate_numpy_byte_identical():
    # numpy grid=False path is a literal scipy passthrough (bit-identical to .ev).
    xg, yg, Z = _rect2d()
    spl = si.RectBivariateSpline(xg, yg, Z, kx=3, ky=3, s=0.0)
    s = Spline2D(x=xg, y=yg, z=Z)
    numpy.testing.assert_array_equal(
        s(_QX2D, _QY2D, grid=False), spl(_QX2D, _QY2D, grid=False)
    )


# 3D coefficient grid for the ndimage cubic map_coordinates (the StaeckelGrid
# jr/jz/ecc/zmax/rperi/rap evaluator works on (nLz, nE, npsi) grids).
_MC_SHAPE = (7, 8, 6)
_MGRID = _rng.uniform(0.1, 2.0, _MC_SHAPE)
_MCOORDS = numpy.vstack([_rng.uniform(0.0, _MC_SHAPE[d] - 1.0, 25) for d in range(3)])


@pytest.mark.parametrize("backend", BACKENDS)
def test_map_coordinates_vs_scipy(backend):
    # ndimage cubic map_coordinates: setup-time scipy spline_filter prefilter,
    # then backend interpolation off the coefficients. numpy byte-identical;
    # jax/torch ~1e-12 vs scipy.ndimage.map_coordinates.
    filt = spline_filter(_MGRID, order=3)
    ref = sndi.map_coordinates(filt, _MCOORDS, order=3, prefilter=False, mode="nearest")
    got = _tonumpy(map_coordinates(filt, _asarray(backend, _MCOORDS)))
    rtol = 0.0 if backend == "numpy" else 1e-12
    numpy.testing.assert_allclose(got, ref, rtol=rtol, atol=1e-13)


def test_map_coordinates_numpy_byte_identical():
    # numpy path is a literal scipy.ndimage.map_coordinates passthrough.
    filt = spline_filter(_MGRID, order=3)
    ref = sndi.map_coordinates(filt, _MCOORDS, order=3, prefilter=False, mode="nearest")
    numpy.testing.assert_array_equal(map_coordinates(filt, _MCOORDS), ref)


@pytest.mark.parametrize("backend", BACKENDS)
def test_mapcoordinates_class_matches_function(backend):
    # The MapCoordinates convenience class prefilters at setup and reproduces the
    # bare scipy result (filtered grid is byte-identical to scipy.spline_filter).
    mc = MapCoordinates(_MGRID, order=3)
    numpy.testing.assert_array_equal(mc.filtered, spline_filter(_MGRID, order=3))
    ref = sndi.map_coordinates(
        mc.filtered, _MCOORDS, order=3, prefilter=False, mode="nearest"
    )
    got = _tonumpy(mc(_asarray(backend, _MCOORDS)))
    rtol = 0.0 if backend == "numpy" else 1e-12
    numpy.testing.assert_allclose(got, ref, rtol=rtol, atol=1e-13)


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_grad_map_coordinates_vs_fd(backend):
    # grad of an interpolated map_coordinates value w.r.t. the query coordinate,
    # vs central finite differences (jax + torch).
    mc = MapCoordinates(_MGRID, order=3)
    c0 = numpy.array([2.3, 3.1, 1.7])

    def _val(c):
        return float(_tonumpy(mc(c.reshape(3, 1)))[0])

    fd = numpy.empty(3)
    for d in range(3):
        cp, cm = c0.copy(), c0.copy()
        cp[d] += 1e-6
        cm[d] -= 1e-6
        fd[d] = (_val(numpy.asarray(cp)) - _val(numpy.asarray(cm))) / 2e-6
    if backend == "jax":
        g = numpy.asarray(jax.grad(lambda c: mc(c.reshape(3, 1))[0])(jnp.asarray(c0)))
    else:
        ct = torch.tensor(c0, requires_grad=True)
        mc(ct.reshape(3, 1))[0].backward()
        g = ct.grad.numpy()
    numpy.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_grad_iuspline_deriv_vs_fd(backend):
    # grad of an interpolated 1D spline VALUE w.r.t. the query point matches the
    # analytic nu=1 derivative (which we also test == scipy), tying autodiff and
    # the explicit derivative together (jax + torch).
    r0 = 4.7
    ana = si.InterpolatedUnivariateSpline(_X1D, _Y1D, k=3)(r0, nu=1)
    s = Spline1D(_X1D, _Y1D, k=3)
    if backend == "jax":
        ad = float(jax.grad(lambda r: s(r))(jnp.asarray(r0)))
    else:
        rt = torch.tensor(r0, requires_grad=True)
        s(rt).backward()
        ad = float(rt.grad)
    numpy.testing.assert_allclose(ad, ana, rtol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_rectbivariate_grid_true_vs_scipy(backend):
    # grid=True evaluates on the outer (tensor) product of X and Y, matching
    # scipy's RectBivariateSpline.__call__(grid=True). numpy byte-identical;
    # jax/torch ~1e-12.
    xg, yg, Z = _rect2d()
    spl = si.RectBivariateSpline(xg, yg, Z, kx=3, ky=3, s=0.0)
    s = Spline2D(x=xg, y=yg, z=Z)
    ref = spl(_QX2D, _QY2D, grid=True)
    got = _tonumpy(s(_asarray(backend, _QX2D), _asarray(backend, _QY2D), grid=True))
    rtol = 0.0 if backend == "numpy" else 1e-12
    numpy.testing.assert_allclose(got, ref, rtol=rtol, atol=1e-13)


@pytest.mark.parametrize("backend", BACKENDS)
def test_interp_linear_nu_branches(backend):
    # interp_linear VALUE (nu=0) == numpy.interp; nu=1 == the per-interval secant
    # slope; nu=2 (past the linear degree) == 0. Exercises all three branches on
    # every backend (the function is backend-agnostic, numpy included).
    xp = _xp(backend)
    x, y, q = _asarray(backend, _X1D), _asarray(backend, _Y1D), _asarray(backend, _Q1D)
    val = _tonumpy(interp_linear(xp, x, y, q))
    numpy.testing.assert_allclose(val, numpy.interp(_Q1D, _X1D, _Y1D), atol=1e-12)
    dval = _tonumpy(interp_linear(xp, x, y, q, nu=1))
    idx = numpy.clip(numpy.searchsorted(_X1D, _Q1D, side="right") - 1, 0, len(_X1D) - 2)
    slope = (_Y1D[idx + 1] - _Y1D[idx]) / (_X1D[idx + 1] - _X1D[idx])
    numpy.testing.assert_allclose(dval, slope, atol=1e-12)
    numpy.testing.assert_allclose(
        _tonumpy(interp_linear(xp, x, y, q, nu=2)), 0.0, atol=1e-12
    )


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_spline1d_mode2_k1_deriv(backend):
    # mode-2 k=1 (backend y) Spline1D: nu=1 returns the per-interval secant slope
    # via the interp_linear path on the backend array.
    s = Spline1D(_X1D, _asarray(backend, _Y1D), k=1)
    dval = _tonumpy(s(_asarray(backend, _Q1D), nu=1))
    idx = numpy.clip(numpy.searchsorted(_X1D, _Q1D, side="right") - 1, 0, len(_X1D) - 2)
    slope = (_Y1D[idx + 1] - _Y1D[idx]) / (_X1D[idx + 1] - _X1D[idx])
    numpy.testing.assert_allclose(dval, slope, atol=1e-12)


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_eval_ppoly_nu_past_degree_zero(backend):
    # The backend eval_ppoly returns zeros for a derivative order past the
    # polynomial degree (nu>k): a degree-3 polynomial's 4th derivative is 0. (The
    # frozen scipy spline itself REJECTS nu>k, so this is a backend-only guard
    # the grids never trigger -- they use only nu in {0, 1}.)
    from galpy.backend.interpolate import eval_ppoly, spline_to_ppoly

    spl0 = si.InterpolatedUnivariateSpline(_X1D, _Y1D, k=3, ext=0)
    x, c = spline_to_ppoly(spl0)
    xp = _xp(backend)
    got = _tonumpy(
        eval_ppoly(
            xp,
            _asarray(backend, x),
            _asarray(backend, c),
            _asarray(backend, _Q1D),
            nu=4,
        )
    )
    numpy.testing.assert_allclose(got, 0.0, atol=1e-13)
