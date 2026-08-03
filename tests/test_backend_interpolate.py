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

from galpy.backend import as_numpy
from galpy.backend.interpolate import (
    MapCoordinates,
    Spline1D,
    Spline2D,
    cubic_spline_coeffs,
    eval_cubic,
    eval_rect_ppoly,
    interp_linear,
    make_smoothing_spline,
    map_coordinates,
    native_rect_cubic_coeffs,
    smoothing_spline,
    spline_filter,
)
from galpy.util import galpyWarning

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


def _is_backend(backend, x):
    from galpy.backend import is_backend_array

    return is_backend_array(x) if backend != "numpy" else not is_backend_array(x)


def _asarray(backend, x, requires_grad=False):
    if backend == "numpy":
        return numpy.asarray(x, dtype=float)
    if backend == "jax":
        return jnp.asarray(x, dtype=jnp.float64)
    return torch.tensor(x, dtype=torch.float64, requires_grad=requires_grad)


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
    got = as_numpy(s(_asarray(backend, _RQ)))
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
    got = as_numpy(s(_asarray(backend, _RQ)))
    numpy.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)
    # numpy query of the same mode-2 instance agrees with the backend query
    numpy.testing.assert_allclose(as_numpy(s(_RQ)), got, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_cubic_and_linear_parity(backend):
    xp = _xp(backend)
    x, y, r = _asarray(backend, _XG), _asarray(backend, _YG), _asarray(backend, _RQ)
    c = cubic_spline_coeffs(xp, x, y, bc="natural")
    cub = as_numpy(eval_cubic(xp, x, c, r))
    refc = si.CubicSpline(_XG, _YG, bc_type="natural")(_RQ)
    numpy.testing.assert_allclose(cub, refc, rtol=1e-9, atol=1e-12)
    lin = as_numpy(interp_linear(xp, x, y, r))
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
    got = as_numpy(s(_asarray(backend, X), _asarray(backend, Y)))
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
    got = as_numpy(
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
    got = as_numpy(
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
    got = as_numpy(eval_cubic(xp, x, c, r))
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
    got = as_numpy(interp_linear(xp, x, y, r, extrapolate=ext))
    rclamp = numpy.clip(_ROUT, _XG[0], _XG[-1])
    ref = numpy.interp(rclamp, _XG, _YG)
    rtol = 1e-12 if backend == "numpy" else 1e-12
    numpy.testing.assert_allclose(got, ref, rtol=rtol, atol=1e-12)


def test_interp_linear_bad_extrapolate():
    with pytest.raises(ValueError):
        interp_linear(numpy, _XG, _YG, _RQ, extrapolate="nope")


# --- interp_bilinear: degree-1 tensor-product interp (the spherical-DF pvr) ----
# scipy RectBivariateSpline(kx=1, ky=1, s=0) IS exact bilinear interpolation, and
# CONSTANT-extrapolates beyond the grid (edge value) -> matched by clip.
_BX = numpy.linspace(-3.0, 3.0, 20)  # log10(r/a)-like abscissa
_BY = numpy.linspace(0.0, 1.0, 15)  # uniform-CDF-like abscissa
numpy.random.seed(4)
_BZ_RAND = numpy.random.uniform(0.0, 1.0, (_BX.size, _BY.size))
# a monotone-in-CDF, smooth-in-r "realistic" v/vesc-like grid
_BZ_REAL = (
    (numpy.tanh(_BX)[:, None] * 0.0 + 1.0)
    * numpy.sqrt(_BY)[None, :]
    * (1.0 - 0.3 / (1.0 + numpy.exp(-_BX))[:, None])
)


@pytest.mark.parametrize("Z", [_BZ_RAND, _BZ_REAL])
@pytest.mark.parametrize("backend", BACKENDS)
def test_interp_bilinear_parity(backend, Z):
    # native bilinear must reproduce RectBivariateSpline(kx=1, ky=1).ev to ~1e-13
    # (in-range and clamped out-of-range), on a random and a realistic grid.
    from galpy.backend.interpolate import interp_bilinear

    spl = si.RectBivariateSpline(_BX, _BY, Z, kx=1, ky=1)
    rng = numpy.random.default_rng(1)
    X = rng.uniform(-4.0, 4.0, 60)  # includes out-of-range in X
    Y = rng.uniform(0.0, 1.0, 60)  # in-range in Y (the CDF axis)
    ref = spl.ev(X, Y)
    xp = _xp(backend)
    got = as_numpy(
        interp_bilinear(
            xp,
            _asarray(backend, _BX),
            _asarray(backend, _BY),
            _asarray(backend, Z),
            _asarray(backend, X),
            _asarray(backend, Y),
            extrapolate="clip",
        )
    )
    numpy.testing.assert_allclose(got, ref, rtol=0.0, atol=1e-13)


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_interp_bilinear_grad_vs_fd(backend):
    # d(value)/d(X), d/d(Y), and d/d(Z-values): random directional AD must
    # h-converge to a central FD of the numpy bilinear (matches scipy).
    from galpy.backend.interpolate import interp_bilinear

    rng = numpy.random.default_rng(2)
    X0 = rng.uniform(-2.5, 2.5, 12)
    Y0 = rng.uniform(0.05, 0.95, 12)
    Z0 = _BZ_RAND

    def val_np(x, y, z):
        return interp_bilinear(numpy, _BX, _BY, z, x, y, extrapolate="clip")

    def loss_np(x, y, z):
        return numpy.sum(val_np(x, y, z))

    dX = rng.standard_normal(X0.shape)
    dX /= numpy.linalg.norm(dX)
    dY = rng.standard_normal(Y0.shape)
    dY /= numpy.linalg.norm(dY)
    dZ = rng.standard_normal(Z0.shape)
    dZ /= numpy.linalg.norm(dZ)
    if backend == "jax":

        def loss(x, y, z):
            return jnp.sum(interp_bilinear(jnp, _BX, _BY, z, x, y, extrapolate="clip"))

        gx, gy, gz = jax.grad(loss, argnums=(0, 1, 2))(
            jnp.asarray(X0), jnp.asarray(Y0), jnp.asarray(Z0)
        )
        gx, gy, gz = numpy.asarray(gx), numpy.asarray(gy), numpy.asarray(gz)
    else:
        xt = torch.tensor(X0, requires_grad=True)
        yt = torch.tensor(Y0, requires_grad=True)
        zt = torch.tensor(Z0, requires_grad=True)
        interp_bilinear(
            _xp("torch"), _BX, _BY, zt, xt, yt, extrapolate="clip"
        ).sum().backward()
        gx, gy, gz = xt.grad.numpy(), yt.grad.numpy(), zt.grad.numpy()
    adX, adY, adZ = (
        float(numpy.dot(gx, dX)),
        float(numpy.dot(gy, dY)),
        float(numpy.sum(gz * dZ)),
    )
    for ad, fn in (
        (adX, lambda h: loss_np(X0 + h * dX, Y0, Z0)),
        (adY, lambda h: loss_np(X0, Y0 + h * dY, Z0)),
        (adZ, lambda h: loss_np(X0, Y0, Z0 + h * dZ)),
    ):
        best = min(abs(ad - (fn(h) - fn(-h)) / (2 * h)) for h in (1e-4, 1e-5, 1e-6))
        assert best < 1e-5 * abs(ad) + 1e-8, f"{backend} bilinear grad best={best:.2e}"


@pytest.mark.parametrize("backend", ["jax"] if "jax" in BACKENDS else [])
def test_interp_bilinear_jit_equals_eager(backend):
    # jit(interp_bilinear) == eager (pure namespace ops, jit-safe).
    from galpy.backend.interpolate import interp_bilinear

    X = jnp.asarray(numpy.linspace(-3.5, 3.5, 30))
    Y = jnp.asarray(numpy.linspace(0.0, 1.0, 30))
    bx, by, bz = jnp.asarray(_BX), jnp.asarray(_BY), jnp.asarray(_BZ_RAND)

    def f(x, y):
        return interp_bilinear(jnp, bx, by, bz, x, y, extrapolate="clip")

    eager = numpy.asarray(f(X, Y))
    jitted = numpy.asarray(jax.jit(f)(X, Y))
    numpy.testing.assert_allclose(jitted, eager, rtol=0.0, atol=1e-14)


def test_interp_bilinear_bad_extrapolate():
    from galpy.backend.interpolate import interp_bilinear

    with pytest.raises(ValueError):
        interp_bilinear(numpy, _BX, _BY, _BZ_RAND, _BX[:3], _BY[:3], extrapolate="nope")


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_spline1d_mode2_linear(backend):
    # mode 2 with k=1: in-backend piecewise-linear; numpy AND backend queries of
    # the same instance both agree with numpy.interp.
    y_b = _asarray(backend, _YG)
    s = Spline1D(_XG, y_b, k=1)  # backend y, k=1 -> mode-2 linear (no scipy spline)
    ref = numpy.interp(_RQ, _XG, _YG)
    got = as_numpy(s(_asarray(backend, _RQ)))
    numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)
    # numpy query of the same mode-2 k=1 instance (interp_linear numpy branch)
    numpy.testing.assert_allclose(as_numpy(s(_RQ)), ref, rtol=1e-12, atol=1e-12)


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
    got = as_numpy(s(_asarray(backend, _ROUT)))
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
    got = as_numpy(
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
    got = as_numpy(s(_asarray(backend, X), _asarray(backend, Y)))
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
    got = as_numpy(s(_asarray(backend, X), _asarray(backend, Y)))
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
    got = as_numpy(s(1.5, _asarray(backend, [0.7])))
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
    val = as_numpy(s(q))
    numpy.testing.assert_allclose(val, ref(_Q1D), rtol=rtol, atol=1e-13)
    dval = as_numpy(s(q, nu=1))
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
    got = as_numpy(s(_asarray(backend, _QX2D), _asarray(backend, _QY2D), grid=False))
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
    got = as_numpy(map_coordinates(filt, _asarray(backend, _MCOORDS)))
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
    got = as_numpy(mc(_asarray(backend, _MCOORDS)))
    rtol = 0.0 if backend == "numpy" else 1e-12
    numpy.testing.assert_allclose(got, ref, rtol=rtol, atol=1e-13)


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_grad_map_coordinates_vs_fd(backend):
    # grad of an interpolated map_coordinates value w.r.t. the query coordinate,
    # vs central finite differences (jax + torch).
    mc = MapCoordinates(_MGRID, order=3)
    c0 = numpy.array([2.3, 3.1, 1.7])

    def _val(c):
        return float(as_numpy(mc(c.reshape(3, 1)))[0])

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
    got = as_numpy(s(_asarray(backend, _QX2D), _asarray(backend, _QY2D), grid=True))
    rtol = 0.0 if backend == "numpy" else 1e-12
    numpy.testing.assert_allclose(got, ref, rtol=rtol, atol=1e-13)


@pytest.mark.parametrize("backend", BACKENDS)
def test_interp_linear_nu_branches(backend):
    # interp_linear VALUE (nu=0) == numpy.interp; nu=1 == the per-interval secant
    # slope; nu=2 (past the linear degree) == 0. Exercises all three branches on
    # every backend (the function is backend-agnostic, numpy included).
    xp = _xp(backend)
    x, y, q = _asarray(backend, _X1D), _asarray(backend, _Y1D), _asarray(backend, _Q1D)
    val = as_numpy(interp_linear(xp, x, y, q))
    numpy.testing.assert_allclose(val, numpy.interp(_Q1D, _X1D, _Y1D), atol=1e-12)
    dval = as_numpy(interp_linear(xp, x, y, q, nu=1))
    idx = numpy.clip(numpy.searchsorted(_X1D, _Q1D, side="right") - 1, 0, len(_X1D) - 2)
    slope = (_Y1D[idx + 1] - _Y1D[idx]) / (_X1D[idx + 1] - _X1D[idx])
    numpy.testing.assert_allclose(dval, slope, atol=1e-12)
    numpy.testing.assert_allclose(
        as_numpy(interp_linear(xp, x, y, q, nu=2)), 0.0, atol=1e-12
    )


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_spline1d_mode2_k1_deriv(backend):
    # mode-2 k=1 (backend y) Spline1D: nu=1 returns the per-interval secant slope
    # via the interp_linear path on the backend array.
    s = Spline1D(_X1D, _asarray(backend, _Y1D), k=1)
    dval = as_numpy(s(_asarray(backend, _Q1D), nu=1))
    idx = numpy.clip(numpy.searchsorted(_X1D, _Q1D, side="right") - 1, 0, len(_X1D) - 2)
    slope = (_Y1D[idx + 1] - _Y1D[idx]) / (_X1D[idx + 1] - _X1D[idx])
    numpy.testing.assert_allclose(dval, slope, atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_spline1d_derivative_method(backend):
    # Spline1D.derivative()(r) == self(r, nu=1), mirroring scipy IUS.derivative().
    # mode-1 (numpy y) delegates to the fitted scipy spline's own .derivative()
    # (byte-identical to scipy); mode-2 (backend y) returns a callable evaluating
    # the analytic derivative of the same power-basis cubic.
    ref = si.InterpolatedUnivariateSpline(_XG, _YG, k=3).derivative()(_RQ)
    # mode 1: numpy-fitted spline, evaluated under each backend
    s1 = Spline1D(_XG, _YG, k=3)
    got1 = as_numpy(s1.derivative()(_asarray(backend, _RQ)))
    numpy.testing.assert_allclose(got1, ref, rtol=1e-9, atol=1e-12)
    if backend == "numpy":
        # mode-1 numpy path is scipy's own derivative spline (byte-identical)
        numpy.testing.assert_array_equal(s1.derivative()(_RQ), ref)
        return
    # mode 2: in-backend cubic (natural bc) -> match scipy CubicSpline derivative
    ref2 = si.CubicSpline(_XG, _YG, bc_type="natural").derivative()(_RQ)
    s2 = Spline1D(_XG, _asarray(backend, _YG), k=3, bc="natural")
    got2 = as_numpy(s2.derivative()(_asarray(backend, _RQ)))
    numpy.testing.assert_allclose(got2, ref2, rtol=1e-9, atol=1e-12)
    # derivative() agrees with the nu=1 call on the same instance
    numpy.testing.assert_allclose(
        got2, as_numpy(s2(_asarray(backend, _RQ), nu=1)), rtol=1e-12, atol=1e-12
    )


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
    got = as_numpy(
        eval_ppoly(
            xp,
            _asarray(backend, x),
            _asarray(backend, c),
            _asarray(backend, _Q1D),
            nu=4,
        )
    )
    numpy.testing.assert_allclose(got, 0.0, atol=1e-13)


###############################################################################
# make_smoothing_spline / smoothing_spline: backend-aware differentiable
# counterparts of scipy.interpolate.make_smoothing_spline (GCV) and
# UnivariateSpline(k=3). numpy y -> scipy passthrough (byte-identical); backend
# y -> a frozen linear operator y->fit differentiable in y. Asserts (a) value
# parity vs scipy on clean data incl. extrapolation; (b) grad-of-fit == the
# frozen-operator column sum; (c) the near-interpolation galpyWarning.
###############################################################################
_smrng = numpy.random.RandomState(7)
_SMX = numpy.unique(numpy.sort(_smrng.uniform(-4.0, 4.0, 30)))
_SMY = numpy.sin(_SMX) + 0.05 * _smrng.randn(len(_SMX))
_SMSIG = numpy.full(len(_SMX), 0.05)
_SMW = 1.0 / _SMSIG**2  # make_smoothing_spline weight = 1/variance
_SMG = numpy.linspace(-4.6, 4.6, 41)  # query grid, extends past the data


def test_make_smoothing_spline_numpy_passthrough():
    # numpy y is a LITERAL scipy passthrough (bit-identical to make_smoothing_spline).
    ref = si.make_smoothing_spline(_SMX, _SMY, w=_SMW)(_SMG)
    numpy.testing.assert_array_equal(
        make_smoothing_spline(_SMX, _SMY, w=_SMW)(_SMG), ref
    )


def test_smoothing_spline_numpy_passthrough():
    # numpy y is a LITERAL UnivariateSpline(k=3) passthrough.
    ref = si.UnivariateSpline(_SMX, _SMY, w=1.0 / _SMSIG, s=5.0, k=3)(_SMG)
    numpy.testing.assert_array_equal(
        smoothing_spline(_SMX, _SMY, w=1.0 / _SMSIG, s=5.0)(_SMG), ref
    )


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_make_smoothing_spline_parity(backend):
    # backend y reconstructs make_smoothing_spline's GCV fit at the query grid
    # (including extrapolation beyond the data) to ~1e-9.
    ref = si.make_smoothing_spline(_SMX, _SMY, w=_SMW)(_SMG)
    spl = make_smoothing_spline(_SMX, _asarray(backend, _SMY), w=_SMW)
    got = as_numpy(spl(_SMG))
    numpy.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_smoothing_spline_parity(backend):
    # backend y reconstructs UnivariateSpline(w, s) at the query grid to ~1e-9.
    ref = si.UnivariateSpline(_SMX, _SMY, w=1.0 / _SMSIG, s=5.0, k=3)(_SMG)
    spl = smoothing_spline(_SMX, _asarray(backend, _SMY), w=1.0 / _SMSIG, s=5.0)
    got = as_numpy(spl(_SMG))
    numpy.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("backend", AD_BACKENDS)
@pytest.mark.parametrize("kind", ["gcv", "fitpack"])
def test_smoothing_spline_grad(backend, kind):
    # AD (jax.grad / torch autograd) differentiates the frozen smoothing operator
    # -- d(sum fit)/d(y) equals the column sum of that operator (the operator is
    # validated against scipy by the parity tests, so this pins the AD path).
    if kind == "gcv":
        maker = lambda y: make_smoothing_spline(_SMX, y, w=_SMW)  # noqa: E731
    else:
        maker = lambda y: smoothing_spline(_SMX, y, w=1.0 / _SMSIG, s=5.0)  # noqa: E731
    op = maker(_asarray(backend, _SMY))  # a _DiffSmoothingSpline
    expected = op._build(_SMG)(_SMY).sum(axis=0)
    if backend == "jax":
        g = numpy.asarray(
            jax.grad(lambda y: jnp.sum(maker(y)(_SMG)))(jnp.asarray(_SMY))
        )
    else:
        yt = torch.tensor(_SMY, requires_grad=True)
        torch.sum(maker(yt)(_SMG)).backward()
        g = yt.grad.numpy()
    numpy.testing.assert_allclose(g, expected, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_smoothing_spline_near_interp_warns(backend, monkeypatch):
    # When FITPACK's fit corresponds to no single penalized-p form (the
    # near-interpolation regime), the backend reconstruction cannot reproduce it
    # and WARNS. Forced deterministically by perturbing FITPACK's coefficients
    # away from any penalized solution (same trick as test_backend_streamTrack).
    import galpy.backend.interpolate as BI

    real_us = BI._scipy_interpolate.UnivariateSpline

    class _PerturbedSpline(real_us):
        def get_coeffs(self):
            c = numpy.array(real_us.get_coeffs(self), dtype=float)
            if c.size > 2:
                c[c.size // 2] += 0.5 * numpy.max(numpy.abs(c)) + 1e-3
            return c

    monkeypatch.setattr(BI._scipy_interpolate, "UnivariateSpline", _PerturbedSpline)
    spl = smoothing_spline(_SMX, _asarray(backend, _SMY), w=1.0 / _SMSIG, s=1.0)
    with pytest.warns(galpyWarning, match="near-interpolation"):
        spl(_SMG)


@pytest.mark.parametrize("backend", AD_BACKENDS)
@pytest.mark.parametrize("nval", [1, 3])
def test_make_smoothing_spline_linear_fallback(backend, nval):
    # fewer than 5 finite points -> linear-interp / constant fallback on the
    # backend path (shared by make_smoothing_spline and smoothing_spline).
    x = numpy.linspace(-2.0, 2.0, nval)
    y = numpy.array([0.1, -0.2, 0.05])[:nval] if nval > 1 else numpy.array([0.42])
    grid = numpy.linspace(-3.0, 3.0, 15)
    got = as_numpy(
        make_smoothing_spline(x, _asarray(backend, y), w=numpy.ones(nval))(grid)
    )
    if nval == 1:
        ref = numpy.full_like(grid, y[0])  # constant fit
    else:
        ref = si.interp1d(
            x, y, kind="linear", fill_value="extrapolate", assume_sorted=True
        )(grid)
    numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_smoothing_spline_default_weight(backend):
    # w=None -> unit weights (matches scipy's default) for both public makers.
    refm = si.make_smoothing_spline(_SMX, _SMY)(_SMG)
    gotm = as_numpy(make_smoothing_spline(_SMX, _asarray(backend, _SMY))(_SMG))
    numpy.testing.assert_allclose(gotm, refm, rtol=1e-9, atol=1e-12)
    refu = si.UnivariateSpline(_SMX, _SMY, s=5.0, k=3)(_SMG)
    gotu = as_numpy(smoothing_spline(_SMX, _asarray(backend, _SMY), s=5.0)(_SMG))
    numpy.testing.assert_allclose(gotu, refu, rtol=1e-9, atol=1e-12)


###############################################################################
# NATIVE FITS (backend-native, GPU-resident, differentiable) -- the dual-path
# spline_filter (backend grid -> native recursive prefilter) and the tensor-product
# not-a-knot cubic fit native_rect_cubic_coeffs (backend z -> RectBivariateSpline
# s=0 match). These back the actionAngleStaeckelGrid/AdiabaticGrid setup on the
# backend so the frozen tables stay backend arrays and the build differentiates
# through to the query. All run under the numpy-default shard (jax+torch
# installed) so numpy.op(<tensor>) deprecation traps surface.
###############################################################################
_NF_RNG = numpy.random.default_rng(20240722)
_NF_G1D = _NF_RNG.standard_normal(21)
_NF_G2D = _NF_RNG.standard_normal((15, 18))
_NF_G3D = _NF_RNG.standard_normal((7, 8, 6))
# a realistic (strictly-positive, log-transformed) StaeckelGrid-like table
_NF_GRID_REAL = numpy.log(_NF_RNG.uniform(0.05, 3.0, (9, 11, 7)))
_MC3 = numpy.vstack(
    [_NF_RNG.uniform(0.0, _NF_GRID_REAL.shape[d] - 1.0, 25) for d in range(3)]
)


def test_spline_filter_numpy_byte_identical():
    # numpy grid -> literal scipy.ndimage.spline_filter passthrough.
    for g in (_NF_G1D, _NF_G2D, _NF_G3D):
        numpy.testing.assert_array_equal(
            spline_filter(g), sndi.spline_filter(g, order=3)
        )


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_spline_filter_native_vs_scipy(backend):
    # a backend grid takes the NATIVE recursive prefilter and matches scipy to ~1e-13,
    # while staying a backend array (GPU-resident, no numpy island).
    for g in (_NF_G1D, _NF_G2D, _NF_G3D, _NF_GRID_REAL):
        ref = sndi.spline_filter(g, order=3)
        out = spline_filter(_asarray(backend, g))
        assert _is_backend(backend, out)
        numpy.testing.assert_allclose(as_numpy(out), ref, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_mapcoordinates_native_filtered_vs_scipy(backend):
    # MapCoordinates built from a BACKEND grid prefilters natively (backend-array
    # coefficients) yet reproduces the full scipy pipeline at ~1e-12.
    mc = MapCoordinates(_asarray(backend, _NF_GRID_REAL), order=3)
    assert _is_backend(backend, mc.filtered)
    ref = sndi.map_coordinates(
        sndi.spline_filter(_NF_GRID_REAL, order=3),
        _MC3,
        order=3,
        prefilter=False,
        mode="nearest",
    )
    got = as_numpy(mc(_asarray(backend, _MC3)))
    numpy.testing.assert_allclose(got, ref, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_spline_filter_grad_vs_fd(backend):
    # d(0.5||spline_filter(g)||^2)/dg through the native prefilter, directional
    # (random unit direction), central-FD-converged. This is the differentiability
    # the grid-table build needs (scipy's prefilter severs it).
    g0 = _NF_GRID_REAL
    V = _NF_RNG.standard_normal(g0.shape)
    V /= numpy.linalg.norm(V)

    def loss(xp, arr):
        return 0.5 * xp.sum(spline_filter(arr) ** 2)

    if backend == "jax":
        gb = jnp.asarray(g0)
        grad = jax.grad(lambda a: loss(jnp, a))(gb)
        dd = float(jnp.sum(grad * jnp.asarray(V)))

        def _l(a):
            return float(loss(jnp, jnp.asarray(a)))

        # jit-safe: jax.jit == eager
        numpy.testing.assert_allclose(
            float(jax.jit(lambda a: loss(jnp, a))(gb)), _l(g0), rtol=1e-12, atol=1e-12
        )
    else:
        gt = torch.tensor(g0, requires_grad=True)
        loss(txp, gt).backward()
        dd = float(torch.sum(gt.grad * torch.as_tensor(V)))

        def _l(a):
            return float(loss(txp, torch.as_tensor(a)))

    fd = (_l(g0 + 1e-4 * V) - _l(g0 - 1e-4 * V)) / 2e-4
    numpy.testing.assert_allclose(dd, fd, rtol=1e-6, atol=1e-9)


# --- native tensor-product not-a-knot cubic fit (RectBivariateSpline s=0) ---
_NF_X = numpy.linspace(0.0, 3.0, 16)
_NF_Y = numpy.linspace(-1.0, 2.0, 19)
_NFXX, _NFYY = numpy.meshgrid(_NF_X, _NF_Y, indexing="ij")
_NF_Z_SMOOTH = numpy.sin(1.3 * _NFXX) * numpy.cos(0.7 * _NFYY) + 0.2 * _NFXX * _NFYY
_NF_Z_RAND = _NF_RNG.standard_normal((16, 19))
_NF_XQ = _NF_RNG.uniform(0.05, 2.95, 400)
_NF_YQ = _NF_RNG.uniform(-0.95, 1.95, 400)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("z", [_NF_Z_SMOOTH, _NF_Z_RAND])
def test_native_rect_cubic_coeffs_vs_scipy(backend, z):
    # native tensor-product not-a-knot cubic == RectBivariateSpline(s=0).ev to
    # ~2e-13 on smooth + random data; the coeffs stay a backend array.
    xp = _xp(backend)
    spl = si.RectBivariateSpline(_NF_X, _NF_Y, z, kx=3, ky=3, s=0.0)
    ref = spl.ev(_NF_XQ, _NF_YQ)
    xbr, ybr, c = native_rect_cubic_coeffs(xp, _NF_X, _NF_Y, _asarray(backend, z))
    assert _is_backend(backend, c)
    got = as_numpy(
        eval_rect_ppoly(
            xp, xbr, ybr, c, _asarray(backend, _NF_XQ), _asarray(backend, _NF_YQ)
        )
    )
    numpy.testing.assert_allclose(got, ref, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_spline2d_mode2_native_vs_scipy(backend):
    # Spline2D with a BACKEND z builds the native fit (mode 2) and matches scipy's
    # RectBivariateSpline.ev; the block stays a backend array.
    spl = si.RectBivariateSpline(_NF_X, _NF_Y, _NF_Z_SMOOTH, kx=3, ky=3, s=0.0)
    ref = spl.ev(_NF_XQ, _NF_YQ)
    s2 = Spline2D(_NF_X, _NF_Y, _asarray(backend, _NF_Z_SMOOTH))
    assert s2._spl is None and _is_backend(backend, s2._c)
    got = as_numpy(s2(_asarray(backend, _NF_XQ), _asarray(backend, _NF_YQ)))
    numpy.testing.assert_allclose(got, ref, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_native_rect_cubic_grad_vs_fd(backend):
    # d(0.5||fit(z).ev(Xq,Yq)||^2)/dz through the native 2D fit, directional,
    # central-FD-converged + jit-safe (jax) -- the y-value differentiability the
    # frozen scipy fit cannot provide.
    z0 = _NF_Z_RAND
    V = _NF_RNG.standard_normal(z0.shape)
    V /= numpy.linalg.norm(V)

    def loss(xp, cast, zz):
        xbr, ybr, c = native_rect_cubic_coeffs(xp, _NF_X, _NF_Y, zz)
        return 0.5 * xp.sum(
            eval_rect_ppoly(xp, xbr, ybr, c, cast(_NF_XQ), cast(_NF_YQ)) ** 2
        )

    if backend == "jax":
        zb = jnp.asarray(z0)
        grad = jax.grad(lambda zz: loss(jnp, jnp.asarray, zz))(zb)
        dd = float(jnp.sum(grad * jnp.asarray(V)))
        numpy.testing.assert_allclose(
            float(jax.jit(lambda zz: loss(jnp, jnp.asarray, zz))(zb)),
            float(loss(jnp, jnp.asarray, zb)),
            rtol=1e-12,
            atol=1e-12,
        )

        def _l(a):
            return float(loss(jnp, jnp.asarray, jnp.asarray(a)))
    else:
        zt = torch.tensor(z0, requires_grad=True)
        loss(txp, torch.as_tensor, zt).backward()
        dd = float(torch.sum(zt.grad * torch.as_tensor(V)))

        def _l(a):
            return float(loss(txp, torch.as_tensor, torch.as_tensor(a)))

    fd = (_l(z0 + 1e-4 * V) - _l(z0 - 1e-4 * V)) / 2e-4
    numpy.testing.assert_allclose(dd, fd, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_spline_filter_native_degenerate_axis(backend):
    # A length<=1 axis takes the _spline_filter1d_native early-return (nothing to
    # prefilter along it); the native prefilter still matches scipy (which leaves
    # such an axis unchanged). Covers the L<=1 short-circuit.
    for g in (numpy.array([3.7]), numpy.arange(6.0).reshape(1, 6)):
        ref = sndi.spline_filter(g, order=3)
        out = spline_filter(_asarray(backend, g))
        assert _is_backend(backend, out)
        numpy.testing.assert_allclose(as_numpy(out), ref, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_spline2d_mode2_bad_k(backend):
    # A backend z selects mode 2 (native fit); mode 2 only implements bicubic, so a
    # non-cubic degree raises. Covers the kx/ky!=3 guard.
    with pytest.raises(ValueError):
        Spline2D(_NF_X, _NF_Y, _asarray(backend, _NF_Z_SMOOTH), kx=2)
    with pytest.raises(ValueError):
        Spline2D(_NF_X, _NF_Y, _asarray(backend, _NF_Z_SMOOTH), ky=2)


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_spline2d_mode2_numpy_query(backend):
    # A mode-2 (backend-native) Spline2D queried with NUMPY points materialises the
    # native coeffs off the backend and evaluates through numpy -- both point-eval
    # (grid=False) and outer-product (grid=True) -- matching scipy .ev / __call__.
    spl = si.RectBivariateSpline(_NF_X, _NF_Y, _NF_Z_SMOOTH, kx=3, ky=3, s=0.0)
    from galpy.backend import is_backend_array

    s2 = Spline2D(_NF_X, _NF_Y, _asarray(backend, _NF_Z_SMOOTH))
    assert s2._spl is None  # mode 2 (no scipy spline stored)
    # grid=False: numpy points paired elementwise (like .ev), returns a numpy array
    got = s2(_NF_XQ, _NF_YQ)
    assert not is_backend_array(got)
    numpy.testing.assert_allclose(got, spl.ev(_NF_XQ, _NF_YQ), rtol=1e-11, atol=1e-12)
    # grid=True: outer product of numpy X,Y (like scipy __call__)
    Xg = numpy.array([0.4, 1.1, 2.6])
    Yg = numpy.array([-0.5, 0.3, 1.2, 1.8])
    got_g = s2(Xg, Yg, grid=True)
    numpy.testing.assert_allclose(got_g, spl(Xg, Yg, grid=True), rtol=1e-11, atol=1e-12)


def test_eval_ppoly_survives_vmap_of_grad_torch():
    # eval_ppoly reads its coefficients at a COMPUTED index (from searchsorted).
    # Plain `a[idx]` is fine under torch's vmap alone and under grad alone, but
    # raises inside the composition vmap(grad(...)) -- which is exactly what
    # galpy/backend/autodiff.py builds for the fE chain. Hence the take()-based
    # _take0. Guard the composition itself, not either layer.
    torch = pytest.importorskip("torch")
    import array_api_compat.torch as xp

    from galpy.backend.interpolate import cubic_spline_coeffs, eval_ppoly

    x = torch.linspace(0.5, 4.0, 24, dtype=torch.float64)
    y = torch.sin(x) + 0.3 * x**2
    c = cubic_spline_coeffs(xp, x, y)

    def f(r):
        return eval_ppoly(xp, x, c, r.reshape(())).reshape(())

    rs = torch.tensor([0.8, 1.7, 2.9, 3.6], dtype=torch.float64)
    got = torch.vmap(torch.func.grad(f))(rs)
    # Compare against the analytic derivative of the same spline (nu=1), which
    # eval_ppoly computes on a separate code path -- so this is a real value
    # check, not just "it did not raise".
    ref = eval_ppoly(xp, x, c, rs, nu=1)
    numpy.testing.assert_allclose(as_numpy(got), as_numpy(ref), rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize("nu", [0, 1, 2, 3, 5])
def test_eval_ppoly_derivative_orders_match_scipy_numpy(nu):
    # Covers eval_ppoly's three coefficient-read paths on NUMPY: the nu==0
    # Horner loop, the nu>k identically-zero shortcut (k==3 here, so nu==5
    # exercises it), and the analytic falling-factorial branch for 0<nu<=k.
    # scipy's PPoly.derivative is the independent reference -- these are value
    # checks, not "did not raise".
    from galpy.backend.interpolate import eval_ppoly

    x = numpy.linspace(0.5, 4.0, 24)
    y = numpy.sin(x) + 0.3 * x**2
    spl = si.CubicSpline(x, y, bc_type="natural")
    pp = si.PPoly(spl.c, spl.x)
    c = cubic_spline_coeffs(numpy, x, y)
    r = numpy.array([0.7, 1.3, 2.2, 3.1, 3.9])
    got = eval_ppoly(numpy, x, c, r, nu=nu)
    ref = numpy.zeros_like(r) if nu > 3 else pp.derivative(nu)(r) if nu else pp(r)
    numpy.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)
