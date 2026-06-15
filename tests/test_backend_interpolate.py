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

from galpy.backend.interpolate import (
    Spline1D,
    Spline2D,
    cubic_spline_coeffs,
    eval_cubic,
    interp_linear,
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
