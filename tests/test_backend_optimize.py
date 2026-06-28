###############################################################################
# test_backend_optimize.py: galpy.backend.optimize -- a backend-agnostic 1-D
# bracketed root finder. Asserts:
#   * numpy input is BYTE-IDENTICAL to scipy.optimize.brentq,
#   * jax/torch input returns a backend array at the right root,
#   * dx*/dtheta matches the implicit-function-theorem value and a finite
#     difference (jax AND torch), with an adversarial check of the gradient
#     sign and scale (a wrong sign or a missing 1/f'(x*) factor would be caught),
#   * eager jax/torch return a backend array (not detached numpy).
###############################################################################
import numpy
import pytest
from scipy import optimize as sopt

from galpy.backend.optimize import brentq

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


def _to_np(x):
    # torch roots carry a grad_fn (the Newton step always builds a graph), so
    # detach before going to numpy; jax/numpy pass through asarray.
    if torch is not None and torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return numpy.asarray(x)


# --- numpy path is byte-identical to scipy ----------------------------------
def test_numpy_equals_scipy_simple():
    # cos(x) - 0.3 on [0, pi/2]: root arccos(0.3). brentq must equal scipy's
    # brentq exactly (it IS scipy on the numpy path).
    f = lambda x: numpy.cos(x) - 0.3
    got = brentq(f, 0.0, numpy.pi / 2.0)
    ref = sopt.brentq(f, 0.0, numpy.pi / 2.0)
    assert got == ref  # byte-identical
    numpy.testing.assert_allclose(got, numpy.arccos(0.3), rtol=1e-12)


def test_numpy_equals_scipy_with_args_and_xtol():
    # cos(x) - theta with theta passed via args, and a custom xtol/rtol: still
    # exactly scipy's value (same kwargs forwarded).
    f = lambda x, theta: numpy.cos(x) - theta
    for theta in (0.1, 0.5, 0.9):
        got = brentq(f, 0.0, numpy.pi / 2.0, args=(theta,), xtol=1e-10, rtol=1e-12)
        ref = sopt.brentq(f, 0.0, numpy.pi / 2.0, args=(theta,), xtol=1e-10, rtol=1e-12)
        assert got == ref


def test_numpy_returns_python_float():
    out = brentq(lambda x: x - 1.5, 0.0, 3.0)
    assert isinstance(out, float)
    assert out == 1.5


# --- backend path: value at the root ----------------------------------------
@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_backend_root_value(backend):
    xp = _xp(backend)
    # root of cos(x) - 0.3 on [0, pi/2] is arccos(0.3)
    theta = xp.asarray(0.3)
    a = xp.asarray(0.0)
    b = xp.asarray(numpy.pi / 2.0)
    root = brentq(lambda x, t: xp.cos(x) - t, a, b, args=(theta,))
    numpy.testing.assert_allclose(
        float(_to_np(root)), float(numpy.arccos(0.3)), rtol=1e-9
    )


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_backend_returns_backend_array(backend):
    # eager jax/torch must return a BACKEND array, not detached numpy. A bare-
    # numpy compute path would silently return numpy here -- the discriminating
    # check.
    xp = _xp(backend)
    a = xp.asarray(0.0)
    b = xp.asarray(2.0)
    root = brentq(lambda x: xp.cos(x) - 0.3, a, b)
    assert backend in type(root).__module__


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_backend_bracket_endpoint_is_array(backend):
    # Only a bracket endpoint (not args) is a backend array -> still routes to
    # the backend path (data-first dispatch on a/b).
    xp = _xp(backend)
    a = xp.asarray(0.0)
    root = brentq(lambda x: xp.cos(x) - 0.3, a, numpy.pi / 2.0)
    assert backend in type(root).__module__
    numpy.testing.assert_allclose(
        float(_to_np(root)), float(numpy.arccos(0.3)), rtol=1e-9
    )


# --- the discriminating tests: gradient via implicit function theorem -------
# For f(x; theta) = cos(x) - theta, the root is x*(theta) = arccos(theta), and
#   dx*/dtheta = -1 / sin(arccos(theta)) = -1 / sqrt(1 - theta^2).
# Implicit theorem: dx*/dtheta = -(df/dtheta)/(df/dx) = -(-1)/(-sin x*) =
#   -1/sin(x*). A wrong sign or a missing 1/f'(x*) factor changes both the sign
#   and the (theta-dependent) magnitude, so this is adversarial.
def _exact_dxstar_dtheta(theta):
    return -1.0 / numpy.sqrt(1.0 - theta**2)


@pytest.mark.parametrize("backend", AD_BACKENDS)
@pytest.mark.parametrize("theta0", [0.2, 0.5, 0.8])
def test_grad_dxstar_dtheta_vs_exact_and_fd(backend, theta0):
    xp = _xp(backend)
    a = xp.asarray(0.0)
    b = xp.asarray(numpy.pi / 2.0)

    def root_of(theta):
        return brentq(lambda x, t: xp.cos(x) - t, a, b, args=(theta,))

    if backend == "jax":
        g = float(jax.grad(lambda t: root_of(t))(jnp.asarray(theta0)))
    else:
        tt = torch.tensor(theta0, requires_grad=True)
        root_of(tt).backward()
        g = float(tt.grad)
    exact = _exact_dxstar_dtheta(theta0)
    # finite difference on the (detached) root value
    eps = 1e-6

    def root_val(t):
        return float(_to_np(root_of(xp.asarray(t))))

    fd = (root_val(theta0 + eps) - root_val(theta0 - eps)) / (2 * eps)
    numpy.testing.assert_allclose(g, exact, rtol=1e-5)
    numpy.testing.assert_allclose(g, fd, rtol=1e-4)
    # adversarial: sign must be negative and magnitude > 1 (since |sin x*| < 1),
    # so a sign flip or a dropped 1/f' factor is rejected.
    assert g < 0.0
    assert abs(g) > 1.0


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_grad_scale_with_slope(backend):
    # Adversarial scale check on a DIFFERENT f where the slope f'(x*) is not +-1,
    # so a missing 1/f'(x*) would mis-scale the gradient.
    # f(x; theta) = x^2 - theta on [0, 5], root x* = sqrt(theta),
    #   dx*/dtheta = 1/(2 sqrt(theta));  df/dx = 2x = 2 sqrt(theta).
    xp = _xp(backend)
    a = xp.asarray(0.0)
    b = xp.asarray(5.0)
    theta0 = 2.0

    def root_of(theta):
        return brentq(lambda x, t: x * x - t, a, b, args=(theta,))

    if backend == "jax":
        g = float(jax.grad(lambda t: root_of(t))(jnp.asarray(theta0)))
    else:
        tt = torch.tensor(theta0, requires_grad=True)
        root_of(tt).backward()
        g = float(tt.grad)
    exact = 1.0 / (2.0 * numpy.sqrt(theta0))
    numpy.testing.assert_allclose(g, exact, rtol=1e-5)
    assert g > 0.0  # increasing theta raises the root


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_grad_wrt_bracket_endpoint_is_zero(backend):
    # The root does not depend on the bracket endpoints (any valid bracket gives
    # the same root), so dx*/da == 0 -- a sanity check that the Newton-step
    # reparameterisation does not leak a spurious bracket gradient.
    xp = _xp(backend)
    b = xp.asarray(numpy.pi / 2.0)
    if backend == "jax":
        g = float(
            jax.grad(lambda a: brentq(lambda x: jnp.cos(x) - 0.3, a, b))(
                jnp.asarray(0.0)
            )
        )
    else:
        at = torch.tensor(0.0, requires_grad=True)
        brentq(lambda x: txp.cos(x) - 0.3, at, b).backward()
        g = 0.0 if at.grad is None else float(at.grad)
    numpy.testing.assert_allclose(g, 0.0, atol=1e-8)


# --- vectorised over an array bracket ---------------------------------------
@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_vectorized_array_theta(backend):
    # cos(x) - theta for a vector of thetas in one call: vectorised bisection.
    xp = _xp(backend)
    thetas = xp.asarray([0.2, 0.5, 0.8])
    a = xp.asarray([0.0, 0.0, 0.0])
    b = xp.asarray([numpy.pi / 2.0] * 3)
    roots = brentq(lambda x, t: xp.cos(x) - t, a, b, args=(thetas,))
    numpy.testing.assert_allclose(
        _to_np(roots), numpy.arccos([0.2, 0.5, 0.8]), rtol=1e-9
    )


@pytest.mark.skipif("jax" not in BACKENDS, reason="jax not installed")
def test_jax_jit_rolls_bisection():
    # Under a trace (jax.jit/grad) the bisection halving loop is rolled into
    # lax.fori_loop instead of unrolling ~n copies of f into the jaxpr: same root,
    # but a small jaxpr with a loop primitive (the eager path keeps the Python
    # loop -- see test_backend_optimize's eager tests). Covers the traced branch
    # of galpy.backend._jax.optimize._bisect_root.
    f = lambda x: x * x - 2.0  # root sqrt(2) on [0, 5]
    a, b = jnp.asarray(0.0), jnp.asarray(5.0)
    # eager value (Python-loop branch)
    r_eager = float(brentq(f, a, b))
    numpy.testing.assert_allclose(r_eager, numpy.sqrt(2.0), rtol=1e-9)
    # jit value (fori_loop branch): a correct root
    r_jit = float(jax.jit(lambda a, b: brentq(f, a, b))(a, b))
    numpy.testing.assert_allclose(r_jit, numpy.sqrt(2.0), rtol=1e-9)
    # the jaxpr is ROLLED: a loop primitive + far fewer eqns than an unrolled
    # ~100-step bisection (which would be 700+ lines).
    txt = str(jax.make_jaxpr(lambda a, b: brentq(f, a, b))(a, b))
    assert ("while" in txt) or ("scan" in txt)
    assert txt.count("\n") < 300

    # jit gradient still exact: d sqrt(c)/dc = 1/(2 sqrt(c)) via implicit theorem
    def root_of_c(c):
        return brentq(lambda x: x * x - c, a, b)

    g = float(jax.jit(jax.grad(root_of_c))(jnp.asarray(2.0)))
    numpy.testing.assert_allclose(g, 1.0 / (2.0 * numpy.sqrt(2.0)), rtol=1e-6)


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_vectorized_grad_jacobian(backend):
    # Per-element gradient of the vector root w.r.t. the vector theta: a diagonal
    # Jacobian with entries -1/sqrt(1-theta^2).
    xp = _xp(backend)
    a = xp.asarray([0.0, 0.0, 0.0])
    b = xp.asarray([numpy.pi / 2.0] * 3)
    theta0 = numpy.array([0.2, 0.5, 0.8])

    def roots(theta):
        return brentq(lambda x, t: xp.cos(x) - t, a, b, args=(theta,))

    if backend == "jax":
        J = numpy.asarray(jax.jacobian(roots)(jnp.asarray(theta0)))
        diag = numpy.diag(J)
    else:
        tt = torch.tensor(theta0, requires_grad=True)
        out = roots(tt)
        diag = numpy.empty(3)
        for i in range(3):
            gi = torch.autograd.grad(out[i], tt, retain_graph=True)[0]
            diag[i] = float(gi[i])
    numpy.testing.assert_allclose(
        diag, [_exact_dxstar_dtheta(t) for t in theta0], rtol=1e-5
    )
