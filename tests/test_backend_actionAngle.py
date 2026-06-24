###############################################################################
# test_backend_actionAngle.py: the closed-form pure-Python action-angle methods
# (actionAngleHarmonic, actionAngleIsochrone) under the jax/torch backends.
# These classes are algebraic (no quadrature/root-find), so the namespace-swap
# makes them evaluate AND differentiate under every backend. Validates:
#   * numpy<->jax<->torch value parity for actions, frequencies, angles, and
#     EccZmaxRperiRap (Isochrone) / actions+freqs+angle (Harmonic);
#   * outputs are backend arrays (device/dtype follow the inputs);
#   * grad-vs-finite-difference of an action / frequency / angle w.r.t. an
#     initial condition (jax.grad and torch.autograd) -- the differentiability
#     that is the point of the backend port.
# The numpy path is byte-identical (get_namespace passes numpy through), so the
# existing test_actionAngle.py suite is unchanged; this only adds backend cover.
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

from galpy.actionAngle import actionAngleHarmonic, actionAngleIsochrone

# A small batch of bound phase-space points (R,vR,vT,z,vz,phi); moderate
# velocities so the isochrone orbits are bound (E<0) and away from the
# turning-point / non-inclined kinks where the angle map is non-smooth.
_R = numpy.array([1.1, 0.8, 1.3])
_vR = numpy.array([0.2, -0.1, 0.05])
_vT = numpy.array([0.9, 0.6, 1.0])
_z = numpy.array([0.15, -0.2, 0.1])
_vz = numpy.array([0.1, 0.05, -0.1])
_phi = numpy.array([1.3, 0.4, 2.1])
_ISO = (_R, _vR, _vT, _z, _vz, _phi)

_HX = numpy.array([0.5, -1.0, 0.3])
_HVX = numpy.array([0.2, 0.4, -0.6])


def _arr(backend, x):
    return jnp.asarray(x) if backend == "jax" else torch.tensor(x)


def _np(x):
    if torch is not None and torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return numpy.asarray(x)


def _is_backend_array(backend, x):
    if backend == "jax":
        return isinstance(x, jax.Array)
    return torch.is_tensor(x)


# ----------------------------------------------------------------- value parity
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("b", [0.0, 0.8, 1.5])
def test_isochrone_parity(backend, b):
    aAI = actionAngleIsochrone(b=b)
    bargs = [_arr(backend, v) for v in _ISO]
    # _evaluate (actions), _actionsFreqs (+freqs), _actionsFreqsAngles (+angles)
    for ref, got in (
        (aAI._evaluate(*_ISO[:5]), aAI._evaluate(*bargs[:5])),
        (aAI._actionsFreqs(*_ISO[:5]), aAI._actionsFreqs(*bargs[:5])),
        (aAI._actionsFreqsAngles(*_ISO), aAI._actionsFreqsAngles(*bargs)),
    ):
        for r, g in zip(ref, got):
            assert _is_backend_array(backend, g)
            numpy.testing.assert_allclose(
                _np(g), numpy.asarray(r), rtol=1e-12, atol=1e-12
            )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("b", [0.8, 1.5])  # b=0 EccZmaxRperiRap only approximate
def test_isochrone_ecczmaxrperirap_parity(backend, b):
    aAI = actionAngleIsochrone(b=b)
    ref = aAI._EccZmaxRperiRap(*_ISO[:5])
    got = aAI._EccZmaxRperiRap(*[_arr(backend, v) for v in _ISO[:5]])
    for r, g in zip(ref, got):
        assert _is_backend_array(backend, g)
        numpy.testing.assert_allclose(_np(g), numpy.asarray(r), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("omega", [0.5, 1.7])
def test_harmonic_parity(backend, omega):
    aAH = actionAngleHarmonic(omega=omega)
    x, vx = _arr(backend, _HX), _arr(backend, _HVX)
    for ref, got in (
        (aAH._evaluate(_HX, _HVX), aAH._evaluate(x, vx)),
        (aAH._actionsFreqs(_HX, _HVX), aAH._actionsFreqs(x, vx)),
        (aAH._actionsFreqsAngles(_HX, _HVX), aAH._actionsFreqsAngles(x, vx)),
    ):
        ref = ref if isinstance(ref, tuple) else (ref,)
        got = got if isinstance(got, tuple) else (got,)
        for r, g in zip(ref, got):
            numpy.testing.assert_allclose(
                _np(g), numpy.asarray(r), rtol=1e-12, atol=1e-12
            )


# ------------------------------------------------------------ grad vs finite-diff
# Scalar single-point ICs for clean per-component derivatives.
_S = (1.1, 0.2, 0.9, 0.15, 0.1, 1.3)


def _fd(f, x0, eps=1e-6):
    return (f(x0 + eps) - f(x0 - eps)) / (2.0 * eps)


def _grad(backend, f, x0):
    if backend == "jax":
        return float(jax.grad(lambda t: f(t))(jnp.asarray(x0)))
    t = torch.tensor(x0, requires_grad=True)
    out = f(t)
    out.backward()
    return float(t.grad)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("which,idx", [("jr", 0), ("omegar", 3), ("angler", 6)])
def test_isochrone_grad_vs_fd_wrt_vR(backend, which, idx):
    # d(action/freq/angle)[component idx] / d vR at a single bound point.
    aAI = actionAngleIsochrone(b=1.0)
    R, _, vT, z, vz, phi = _S

    def call(vR_val, xp_arr):
        # build args in the namespace of vR_val (numpy float for FD, backend
        # scalar for AD) so get_namespace follows vR for the differentiated path
        args = (xp_arr(R), vR_val, xp_arr(vT), xp_arr(z), xp_arr(vz), xp_arr(phi))
        return aAI._actionsFreqsAngles(*args)[idx]

    def f_np(vR_val):
        return numpy.asarray(call(vR_val, lambda v: numpy.asarray(v)))

    fd = _fd(f_np, _S[1])

    def f_be(vR_t):
        return call(vR_t, lambda v: _arr(backend, numpy.asarray(v, dtype=float)))

    g = _grad(backend, f_be, _S[1])
    assert numpy.isfinite(g)
    numpy.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("backend", BACKENDS)
def test_harmonic_grad_vs_fd(backend):
    # dJ/dx and dOmega-angle/dx for the harmonic oscillator.
    aAH = actionAngleHarmonic(omega=1.3)
    vx0 = -0.3

    def jact(x_val, xp_arr):
        return aAH._evaluate(x_val, xp_arr(vx0))

    fd = _fd(lambda xv: numpy.asarray(jact(xv, lambda v: numpy.asarray(v))), 0.7)
    g = _grad(
        backend,
        lambda xt: jact(xt, lambda v: _arr(backend, numpy.asarray(v, float))),
        0.7,
    )
    numpy.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-7)
