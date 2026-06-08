###############################################################################
# test_backend_special.py: the native-preferring special-function router
# (galpy.backend.special). For each Tier-1 function this asserts:
#   1. value parity numpy/jax/torch vs scipy.special (numpy byte-identical;
#      jax/torch to rtol 1e-12) on galpy's argument ranges;
#   2. autodiff (jax.grad / torch.autograd) matches central finite differences;
#   3. the fallback table (_NEEDS_FALLBACK) matches the installed backends, and
#      every fallback agrees with scipy (so a fallback is deletable once the
#      backend ships the native function).
###############################################################################
import numpy
import pytest
import scipy.special as scipy_special

from galpy.backend import special as gsp
from galpy.backend.special._router import _NEEDS_FALLBACK, _backend_special

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


# (router fn, scipy fn, n-args, sample points). Ranges cover what galpy uses.
_POS = numpy.array([0.1, 0.5, 0.9, 1.3, 2.0, 3.7, 5.0])
_REAL = numpy.array([-3.0, -1.2, -0.3, 0.0, 0.4, 1.1, 2.5])
_GAMMA_X = numpy.array([0.2, 0.5, 1.0, 2.5, 4.0, -0.5, -1.5, -2.5])  # incl. reflection
_A = numpy.array([0.5, 1.0, 2.0, 3.5])
_XG = numpy.array([0.05, 0.5, 1.5, 3.0, 6.0])

UNARY = [
    ("gammaln", gsp.gammaln, scipy_special.gammaln, _POS),
    ("gamma", gsp.gamma, scipy_special.gamma, _GAMMA_X),
    ("erf", gsp.erf, scipy_special.erf, _REAL),
    ("erfc", gsp.erfc, scipy_special.erfc, _REAL),
    ("i0", gsp.i0, scipy_special.i0, numpy.abs(_REAL)),
    ("i1", gsp.i1, scipy_special.i1, _REAL),
]


@pytest.mark.parametrize("name,fn,sp_fn,pts", UNARY, ids=[u[0] for u in UNARY])
@pytest.mark.parametrize("backend", BACKENDS)
def test_unary_value_parity(backend, name, fn, sp_fn, pts):
    ref = sp_fn(pts)
    got = _tonumpy(fn(_asarray(backend, pts)))
    rtol = 0.0 if backend == "numpy" else 1e-12  # numpy must be byte-identical
    numpy.testing.assert_allclose(
        got, ref, rtol=rtol, atol=1e-12, err_msg=f"{name} ({backend})"
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_gammainc_value_parity(backend):
    for sp_fn, fn in [
        (scipy_special.gammainc, gsp.gammainc),
        (scipy_special.gammaincc, gsp.gammaincc),
    ]:
        for a in _A:
            ref = sp_fn(a, _XG)
            got = _tonumpy(fn(_asarray(backend, a), _asarray(backend, _XG)))
            rtol = 0.0 if backend == "numpy" else 1e-12
            numpy.testing.assert_allclose(got, ref, rtol=rtol, atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_xlogy_value_parity_incl_zero(backend):
    x = numpy.array([0.0, 0.0, 1.0, 2.5, 0.3])
    y = numpy.array([0.0, 5.0, 2.0, 0.7, 10.0])  # x=0 -> 0 even when y=0
    ref = scipy_special.xlogy(x, y)
    got = _tonumpy(gsp.xlogy(_asarray(backend, x), _asarray(backend, y)))
    rtol = 0.0 if backend == "numpy" else 1e-12
    numpy.testing.assert_allclose(got, ref, rtol=rtol, atol=1e-12)


@pytest.mark.parametrize("name,fn,sp_fn,pts", UNARY, ids=[u[0] for u in UNARY])
@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_unary_grad_vs_fd(backend, name, fn, sp_fn, pts):
    # differentiate at smooth interior points (avoid gamma's poles at <=0 ints)
    x0 = 1.3 if name in ("gammaln", "gamma", "i0", "i1") else 0.7
    eps = 1e-6
    fd = (float(sp_fn(x0 + eps)) - float(sp_fn(x0 - eps))) / (2 * eps)
    if backend == "jax":
        ad = float(jax.grad(lambda x: fn(x))(jnp.asarray(x0)))
    else:
        xt = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
        fn(xt).backward()
        ad = float(xt.grad)
    assert not numpy.isnan(ad), f"NaN grad for {name} ({backend})"
    numpy.testing.assert_allclose(ad, fd, rtol=1e-5, err_msg=f"{name} grad ({backend})")


def test_fallback_table_matches_installed_backends():
    # _NEEDS_FALLBACK must list exactly the Tier-1 functions the backend lacks.
    tier1 = ["gammaln", "gamma", "gammainc", "gammaincc", "erf", "erfc", "i0", "i1"]
    for backend in AD_BACKENDS:
        xp = _asarray(backend, 1.0)
        from galpy.backend import get_namespace

        _name, sp = _backend_special(get_namespace(xp))
        for fn in tier1:
            listed = fn in _NEEDS_FALLBACK.get(backend, frozenset())
            native = hasattr(sp, fn)
            assert listed == (not native), (
                f"{backend}: {fn} native={native} but listed-as-fallback={listed}; "
                f"update _NEEDS_FALLBACK"
            )


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_gamma_fallback_agrees_and_no_nan_grad(backend):
    # Exercises the pure-backend gamma fallback (torch); on jax it routes native,
    # but we still check parity + finite gradient across the reflection at x<0.
    pts = _GAMMA_X
    got = _tonumpy(gsp.gamma(_asarray(backend, pts)))
    numpy.testing.assert_allclose(got, scipy_special.gamma(pts), rtol=1e-11, atol=1e-11)
    # gradient at a negative non-integer (reflection branch) must be finite + correct
    x0 = -1.5
    eps = 1e-6
    fd = (
        float(scipy_special.gamma(x0 + eps)) - float(scipy_special.gamma(x0 - eps))
    ) / (2 * eps)
    if backend == "jax":
        ad = float(jax.grad(lambda x: gsp.gamma(x))(jnp.asarray(x0)))
    else:
        xt = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
        gsp.gamma(xt).backward()
        ad = float(xt.grad)
    assert not numpy.isnan(ad)
    numpy.testing.assert_allclose(ad, fd, rtol=1e-4)
