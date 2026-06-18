###############################################################################
# test_backend_orbit_accessors.py: the time-evaluation accessors (o.R(t),
# o.x(t), o.vr(t), ...) made backend-aware for in-backend (diffrax/torchdiffeq)
# orbits, so a jax/torch-integrated orbit can be queried AND differentiated
# through its phase-space accessors (the deferred half of the in-backend wiring).
#
# Proves, for an Orbit integrated with method='diffrax'/'torchdiffeq':
#   1. every phase-space accessor returns a value ON the orbit's backend
#      (jax.Array / torch.Tensor) that matches galpy's C trajectory, at the
#      integration grid AND interpolated between grid points (incl. backward),
#   2. gradients flow through the accessors and the off-grid interpolation
#      (d accessor(t)/d IC and d/d query-time vs FD; torch-grad == jax-grad),
#   3. off-grid interpolation matches the true orbit + the numpy spline path,
#      and a (concrete) out-of-range time raises ValueError as numpy does,
#   4. the numpy path is untouched (a numpy orbit's accessors are unchanged).
#
# Self-skips unless the runtime ODE extra (diffrax / torchdiffeq) is installed.
###############################################################################
import numpy
import pytest

from galpy.orbit import Orbit
from galpy.potential import PlummerPotential

pytestmark = pytest.mark.backend_managed

HAVE_JAX = False
HAVE_TORCH = False
try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import diffrax  # noqa: F401
    import jax.numpy as jnp

    HAVE_JAX = True
except ImportError:  # pragma: no cover
    pass
try:
    import torch

    torch.set_default_dtype(torch.float64)
    import torchdiffeq  # noqa: F401

    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    pass

_POT = PlummerPotential(amp=1.0, b=0.6)
_IC = [1.0, 0.1, 0.9, 0.2, 0.05, 0.3]  # R, vR, vT, z, vz, phi
_TS = numpy.linspace(0.0, 6.0, 60)
# every 3D phase-space accessor (those that don't need actionAngle)
_ACCESSORS = [
    "R",
    "vR",
    "vT",
    "z",
    "vz",
    "phi",
    "x",
    "y",
    "vx",
    "vy",
    "r",
    "vr",
    "vtheta",
    "theta",
    "vphi",
]
_WRAP = {"phi", "theta"}  # compare modulo 2pi


def _c_ref(acc):
    o = Orbit(list(_IC))
    o.integrate(_TS, _POT, method="dop853_c")
    return numpy.asarray(getattr(o, acc)(_TS, use_physical=False))


def _wrap(a):
    a = numpy.asarray(a, dtype=float)
    return (a + numpy.pi) % (2 * numpy.pi) - numpy.pi


@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
@pytest.mark.parametrize("acc", _ACCESSORS)
def test_accessor_diffrax_matches_c_and_stays_jax(acc):
    o = Orbit(jnp.asarray(_IC))
    o.integrate(jnp.asarray(_TS), _POT, method="diffrax")
    val = getattr(o, acc)(jnp.asarray(_TS), use_physical=False)
    assert isinstance(val, jax.Array), f"{acc} left the jax backend"
    got, ref = numpy.asarray(val), _c_ref(acc)
    if acc in _WRAP:
        got, ref = _wrap(got), _wrap(ref)
    numpy.testing.assert_allclose(got, ref, rtol=1e-8, atol=1e-8)


@pytest.mark.skipif(not HAVE_TORCH, reason="torch/torchdiffeq not installed")
@pytest.mark.parametrize("acc", _ACCESSORS)
def test_accessor_torchdiffeq_matches_c_and_stays_torch(acc):
    o = Orbit(torch.as_tensor(_IC))
    o.integrate(torch.as_tensor(_TS), _POT, method="torchdiffeq")
    val = getattr(o, acc)(torch.as_tensor(_TS), use_physical=False)
    assert isinstance(val, torch.Tensor), f"{acc} left the torch backend"
    got, ref = val.detach().cpu().numpy(), _c_ref(acc)
    if acc in _WRAP:
        got, ref = _wrap(got), _wrap(ref)
    numpy.testing.assert_allclose(got, ref, rtol=1e-8, atol=1e-8)


@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
@pytest.mark.parametrize("acc", ["x", "r", "vr"])
def test_accessor_diffrax_grad_vs_fd(acc):
    # gradient of a derived accessor's final value w.r.t. an IC component
    def final(vR0):
        o = Orbit(jnp.array([1.0, vR0, 0.9, 0.2, 0.05, 0.3]))
        o.integrate(jnp.asarray(_TS), _POT, method="diffrax")
        return getattr(o, acc)(jnp.asarray(_TS), use_physical=False)[-1]

    g = float(jax.grad(final)(0.1))
    eps = 1e-6
    fd = float((final(0.1 + eps) - final(0.1 - eps)) / (2 * eps))
    numpy.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-7)


@pytest.mark.skipif(not (HAVE_JAX and HAVE_TORCH), reason="needs both jax and torch")
def test_accessor_torch_grad_matches_jax():
    # torch.autograd of o.x(t_final) w.r.t. vR0 == jax.grad (FD-free cross-check)
    def final_x_jax(vR0):
        o = Orbit(jnp.array([1.0, vR0, 0.9, 0.2, 0.05, 0.3]))
        o.integrate(jnp.asarray(_TS), _POT, method="diffrax")
        return o.x(jnp.asarray(_TS), use_physical=False)[-1]

    g_jax = float(jax.grad(final_x_jax)(0.1))
    v = torch.tensor(_IC, requires_grad=True)
    o = Orbit(v)
    o.integrate(torch.as_tensor(_TS), _POT, method="torchdiffeq")
    o.x(torch.as_tensor(_TS), use_physical=False)[-1].backward()
    numpy.testing.assert_allclose(float(v.grad[1]), g_jax, rtol=1e-6, atol=1e-8)


# ------------------------------------------- off-grid (interpolated) evaluation
_TQ = numpy.array([0.37, 1.84, 2.5, 3.91, 5.23])  # times between the grid points


def _c_true(acc):
    # the true trajectory: a C orbit integrated to land exactly on the query times
    o = Orbit(list(_IC))
    o.integrate(numpy.sort(numpy.concatenate([[0.0], _TQ])), _POT, method="dop853_c")
    return numpy.asarray(getattr(o, acc)(_TQ, use_physical=False))


def _np_interp(acc):
    # the existing numpy spline-interpolation path off-grid
    o = Orbit(list(_IC))
    o.integrate(_TS, _POT, method="dop853_c")
    return numpy.asarray(getattr(o, acc)(_TQ, use_physical=False))


@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
@pytest.mark.parametrize("acc", _ACCESSORS)
def test_accessor_diffrax_offgrid_matches_truth_and_numpy(acc):
    # off-grid times use the in-backend differentiable cubic spline (no scipy):
    # the interpolated value matches both the TRUE orbit and the numpy spline path
    o = Orbit(jnp.asarray(_IC))
    o.integrate(jnp.asarray(_TS), _POT, method="diffrax")
    val = getattr(o, acc)(jnp.asarray(_TQ), use_physical=False)
    assert isinstance(val, jax.Array), f"{acc} off-grid left the jax backend"
    got = numpy.asarray(val)
    truth, npv = _c_true(acc), _np_interp(acc)
    if acc in _WRAP:
        got, truth, npv = _wrap(got), _wrap(truth), _wrap(npv)
    # accurate vs the true orbit -- this is the genuine cubic-interpolation error
    # between grid points (dt~0.1); measured max over all accessors ~2e-7, and the
    # numpy/scipy spline path has the identical error (see below), so it is not a
    # backend deficiency. Tolerance kept ~50x above that floor.
    numpy.testing.assert_allclose(got, truth, rtol=1e-5, atol=1e-5)
    # and tracks the existing numpy spline path to ~machine: the backend Spline1D
    # reproduces galpy's orbit interpolant (same knots/BC), measured ~3e-11.
    numpy.testing.assert_allclose(got, npv, rtol=1e-9, atol=1e-9)


@pytest.mark.skipif(not HAVE_TORCH, reason="torch/torchdiffeq not installed")
def test_accessor_torchdiffeq_offgrid_matches_truth():
    o = Orbit(torch.as_tensor(_IC))
    o.integrate(torch.as_tensor(_TS), _POT, method="torchdiffeq")
    val = o.R(torch.as_tensor(_TQ), use_physical=False)
    assert isinstance(val, torch.Tensor)
    numpy.testing.assert_allclose(
        val.detach().cpu().numpy(), _c_true("R"), rtol=1e-5, atol=1e-5
    )


@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
def test_accessor_diffrax_offgrid_grad_ic():
    # gradient of an off-grid value w.r.t. an IC flows through the in-backend
    # spline (eps=1e-4: the adaptive solver makes a 1e-6 FD pure roundoff noise)
    def fR(vR0):
        o = Orbit(jnp.array([1.0, vR0, 0.9, 0.2, 0.05, 0.3]))
        o.integrate(jnp.asarray(_TS), _POT, method="diffrax")
        return o.R(jnp.asarray([2.345]), use_physical=False)[0]

    g = float(jax.grad(fR)(0.1))
    eps = 1e-4
    fd = float((fR(0.1 + eps) - fR(0.1 - eps)) / (2 * eps))
    numpy.testing.assert_allclose(g, fd, rtol=1e-4, atol=1e-6)


@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
def test_accessor_diffrax_offgrid_grad_query_time():
    # differentiable w.r.t. the evaluation time itself: dR/dt of the interpolant
    o = Orbit(jnp.asarray(_IC))
    o.integrate(jnp.asarray(_TS), _POT, method="diffrax")

    def Rt(tq):
        return o.R(tq.reshape(1), use_physical=False)[0]

    g = float(jax.grad(Rt)(jnp.asarray(2.345)))
    e = 1e-5
    fd = float(
        (
            o.R(jnp.asarray([2.345 + e]), use_physical=False)[0]
            - o.R(jnp.asarray([2.345 - e]), use_physical=False)[0]
        )
        / (2 * e)
    )
    numpy.testing.assert_allclose(g, fd, rtol=1e-4, atol=1e-6)


@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
def test_accessor_diffrax_offgrid_scalar_query():
    # a scalar off-grid time returns a scalar (matching the numpy path)
    o = Orbit(jnp.asarray(_IC))
    o.integrate(jnp.asarray(_TS), _POT, method="diffrax")
    val = o.R(2.345, use_physical=False)
    assert numpy.asarray(val).shape == ()
    o_np = Orbit(list(_IC))
    o_np.integrate(_TS, _POT, method="dop853_c")
    numpy.testing.assert_allclose(
        float(val), float(o_np.R(2.345, use_physical=False)), rtol=1e-9, atol=1e-9
    )


@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
def test_accessor_diffrax_scalar_backend_query_and_grad():
    # a 0-d (scalar) jax/torch query time is supported directly: o.R(jnp.asarray(t))
    # returns a scalar and is differentiable w.r.t. t. (A 0-d backend array has
    # __len__ but len() raises, which used to crash before reaching the backend
    # interpolator -- guarded by the numpy.ndim check in _call_internal.)
    o = Orbit(jnp.asarray(_IC))
    o.integrate(jnp.asarray(_TS), _POT, method="diffrax")
    val = o.R(jnp.asarray(2.345), use_physical=False)
    assert isinstance(val, jax.Array)
    assert numpy.asarray(val).shape == ()
    g = float(jax.grad(lambda t: o.R(t, use_physical=False))(jnp.asarray(2.345)))
    e = 1e-5
    fd = float(
        (
            o.R(jnp.asarray(2.345 + e), use_physical=False)
            - o.R(jnp.asarray(2.345 - e), use_physical=False)
        )
        / (2 * e)
    )
    numpy.testing.assert_allclose(g, fd, rtol=1e-4, atol=1e-6)


@pytest.mark.skipif(not HAVE_TORCH, reason="torch/torchdiffeq not installed")
def test_accessor_torchdiffeq_scalar_backend_query():
    o = Orbit(torch.as_tensor(_IC))
    o.integrate(torch.as_tensor(_TS), _POT, method="torchdiffeq")
    val = o.R(torch.as_tensor(2.345), use_physical=False)
    assert isinstance(val, torch.Tensor)
    assert val.ndim == 0


@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
def test_accessor_diffrax_offgrid_out_of_range_raises():
    # a (concrete) query outside the integration window still raises, as numpy does
    o = Orbit(jnp.asarray(_IC))
    o.integrate(jnp.asarray(_TS), _POT, method="diffrax")
    with pytest.raises(ValueError):
        o.R(jnp.asarray([99.0]), use_physical=False)


@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
@pytest.mark.parametrize("acc", ["R", "phi", "x", "z", "vr"])
def test_accessor_diffrax_offgrid_backward(acc):
    # BACKWARD integration gives a decreasing self.t; the spline needs an
    # increasing grid, so the trajectory is flipped to ascending before fitting.
    # (g14's HVS example integrates backward, so this path matters.)
    tsb = numpy.linspace(0.0, -6.0, 60)
    tqb = numpy.array([-1.3, -2.7, -4.1])
    o = Orbit(jnp.asarray(_IC))
    o.integrate(jnp.asarray(tsb), _POT, method="diffrax")
    got = numpy.asarray(getattr(o, acc)(jnp.asarray(tqb), use_physical=False))
    oc = Orbit(list(_IC))
    oc.integrate(
        numpy.sort(numpy.concatenate([[0.0], tqb]))[::-1], _POT, method="dop853_c"
    )
    ref = numpy.asarray(getattr(oc, acc)(tqb, use_physical=False))
    if acc in _WRAP:
        got, ref = _wrap(got), _wrap(ref)
    numpy.testing.assert_allclose(got, ref, rtol=1e-4, atol=5e-4)


@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
def test_accessor_diffrax_offgrid_short_grid_linear():
    # a grid too short for a cubic (<=3 points) falls back to linear interpolation
    ts = numpy.array([0.0, 1.0, 2.0])
    o = Orbit(jnp.asarray(_IC))
    o.integrate(jnp.asarray(ts), _POT, method="diffrax")
    val = o.R(jnp.asarray([0.5, 1.5]), use_physical=False)
    assert isinstance(val, jax.Array)
    assert numpy.asarray(val).shape == (2,)


@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
def test_accessor_planar_and_1d_backend():
    # planar (phasedim 4) and 1D (phasedim 2) in-backend orbits expose their
    # supported accessors on the backend
    from galpy.potential import IsothermalDiskPotential

    op = Orbit(jnp.asarray([1.0, 0.1, 0.9, 0.3]))
    op.integrate(jnp.asarray(_TS), _POT, method="diffrax")
    assert isinstance(op.R(jnp.asarray(_TS), use_physical=False), jax.Array)
    assert isinstance(op.x(jnp.asarray(_TS), use_physical=False), jax.Array)
    o1 = Orbit(jnp.asarray([0.1, 0.05]))
    o1.integrate(
        jnp.asarray(_TS), IsothermalDiskPotential(amp=1.0, sigma=0.5), method="diffrax"
    )
    assert isinstance(o1.x(jnp.asarray(_TS), use_physical=False), jax.Array)


def test_accessor_numpy_path_unchanged():
    # a numpy orbit's accessors are unaffected by the backend dispatch
    o = Orbit(list(_IC))
    o.integrate(_TS, _POT, method="dop853_c")
    for acc in _ACCESSORS:
        val = getattr(o, acc)(_TS, use_physical=False)
        assert isinstance(val, numpy.ndarray)
        numpy.testing.assert_allclose(numpy.asarray(val), _c_ref(acc), rtol=1e-12)
