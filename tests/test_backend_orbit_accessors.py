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

from galpy.backend import as_numpy, is_backend_array, use
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


# ------------------------------------------------- angular momentum L() (3-vec)
# o.L(t) cross-multiplies x,y,z,vx,vy,vz; under a forced backend it must resolve
# a single namespace and coerce all six onto it before the cross-product (the
# numpy*Tensor branch in Orbits.L, phasedim==6). These exercise that branch.
def _c_ref_L():
    o = Orbit(list(_IC))
    o.integrate(_TS, _POT, method="dop853_c")
    return numpy.asarray(o.L(_TS, use_physical=False))


@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
def test_L_diffrax_matches_c_and_stays_jax():
    o = Orbit(jnp.asarray(_IC))
    o.integrate(jnp.asarray(_TS), _POT, method="diffrax")
    val = o.L(jnp.asarray(_TS), use_physical=False)
    assert isinstance(val, jax.Array), "L left the jax backend"
    got = numpy.asarray(val)
    assert got.shape == (*_TS.shape, 3)
    numpy.testing.assert_allclose(got, _c_ref_L(), rtol=1e-8, atol=1e-8)


@pytest.mark.skipif(not HAVE_TORCH, reason="torch/torchdiffeq not installed")
def test_L_torchdiffeq_matches_c_and_stays_torch():
    o = Orbit(torch.as_tensor(_IC))
    o.integrate(torch.as_tensor(_TS), _POT, method="torchdiffeq")
    val = o.L(torch.as_tensor(_TS), use_physical=False)
    assert isinstance(val, torch.Tensor), "L left the torch backend"
    got = val.detach().cpu().numpy()
    assert got.shape == (*_TS.shape, 3)
    numpy.testing.assert_allclose(got, _c_ref_L(), rtol=1e-8, atol=1e-8)


def test_L_numpy_path_unchanged():
    # a numpy orbit's L() is unaffected by the backend dispatch (3-vector, ...,3)
    o = Orbit(list(_IC))
    o.integrate(_TS, _POT, method="dop853_c")
    val = o.L(_TS, use_physical=False)
    assert isinstance(val, numpy.ndarray)
    assert val.shape == (*_TS.shape, 3)
    numpy.testing.assert_allclose(numpy.asarray(val), _c_ref_L(), rtol=1e-12)


@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
def test_accessor_diffrax_scalar_ongrid_query():
    # a scalar time landing EXACTLY on the integration grid, on a backend orbit ->
    # the on-grid scalar branch of _call_internal (slice + permute_dims + clone)
    o = Orbit(jnp.asarray(_IC))
    o.integrate(jnp.asarray(_TS), _POT, method="diffrax")
    tg = float(_TS[10])
    val = o.R(tg, use_physical=False)
    assert isinstance(val, jax.Array), "on-grid scalar query left the jax backend"
    o_np = Orbit(list(_IC))
    o_np.integrate(_TS, _POT, method="dop853_c")
    numpy.testing.assert_allclose(
        float(val), float(o_np.R(tg, use_physical=False)), rtol=1e-8, atol=1e-8
    )


@pytest.mark.skipif(not HAVE_TORCH, reason="torch/torchdiffeq not installed")
def test_accessor_torchdiffeq_scalar_ongrid_query():
    o = Orbit(torch.as_tensor(_IC))
    o.integrate(torch.as_tensor(_TS), _POT, method="torchdiffeq")
    tg = float(_TS[10])
    val = o.R(tg, use_physical=False)
    assert isinstance(val, torch.Tensor)
    o_np = Orbit(list(_IC))
    o_np.integrate(_TS, _POT, method="dop853_c")
    numpy.testing.assert_allclose(
        float(val.detach().cpu()),
        float(o_np.R(tg, use_physical=False)),
        rtol=1e-8,
        atol=1e-8,
    )


@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
@pytest.mark.parametrize("acc", ["R", "vR", "z", "vz", "r"])
def test_accessor_diffrax_offgrid_axisymmetric(acc):
    # an axisymmetric (phasedim=5, no phi) backend orbit queried off-grid -> the
    # non-(4,6) interpolation branch (no x,y wrap-dodge needed without phi)
    ic5 = [1.0, 0.1, 0.9, 0.2, 0.05]  # R, vR, vT, z, vz
    o = Orbit(jnp.asarray(ic5))
    o.integrate(jnp.asarray(_TS), _POT, method="diffrax")
    val = getattr(o, acc)(jnp.asarray(_TQ), use_physical=False)
    assert isinstance(val, jax.Array), f"{acc} off-grid (no phi) left the jax backend"
    o_np = Orbit(list(ic5))
    o_np.integrate(_TS, _POT, method="dop853_c")
    numpy.testing.assert_allclose(
        numpy.asarray(val),
        numpy.asarray(getattr(o_np, acc)(_TQ, use_physical=False)),
        rtol=1e-5,
        atol=1e-5,
    )


# ----------------------------------- numeric orbit characteristics (reductions)
# e(), rap(), rperi(), zmax() reduce over the integrated trajectory
# (numpy.amax/amin); on a backend-integrated orbit self.r(t)/self.z(t) is a
# backend array, so the reduction must run in the orbit's namespace -- else
# numpy.amax(tensor) crashes (rap/rperi) or silently coerces back to numpy and
# drops the backend/grad (zmax, via numpy.fabs).
_CHARS = ["e", "rap", "rperi", "zmax"]


def _c_ref_char(name):
    o = Orbit(list(_IC))
    o.integrate(_TS, _POT, method="dop853_c")
    return float(numpy.asarray(getattr(o, name)(use_physical=False)))


@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
@pytest.mark.parametrize("name", _CHARS)
def test_orbit_characteristic_diffrax_matches_c_and_stays_jax(name):
    o = Orbit(jnp.asarray(_IC))
    o.integrate(jnp.asarray(_TS), _POT, method="diffrax")
    val = getattr(o, name)(use_physical=False)
    assert isinstance(val, jax.Array), f"{name} left the jax backend"
    numpy.testing.assert_allclose(
        float(numpy.asarray(val)), _c_ref_char(name), rtol=1e-6, atol=1e-6
    )


@pytest.mark.skipif(not HAVE_TORCH, reason="torch/torchdiffeq not installed")
@pytest.mark.parametrize("name", _CHARS)
def test_orbit_characteristic_torchdiffeq_matches_c_and_stays_torch(name):
    o = Orbit(torch.as_tensor(_IC))
    o.integrate(torch.as_tensor(_TS), _POT, method="torchdiffeq")
    val = getattr(o, name)(use_physical=False)
    assert isinstance(val, torch.Tensor), f"{name} left the torch backend"
    numpy.testing.assert_allclose(
        float(val.detach().cpu()), _c_ref_char(name), rtol=1e-6, atol=1e-6
    )


def test_orbit_characteristic_numpy_path_unchanged():
    # a numpy orbit's e/rap/rperi/zmax are unaffected by the namespace dispatch
    o = Orbit(list(_IC))
    o.integrate(_TS, _POT, method="dop853_c")
    for name in _CHARS:
        val = getattr(o, name)(use_physical=False)
        assert isinstance(val, (float, numpy.floating, numpy.ndarray))
        numpy.testing.assert_allclose(
            float(numpy.asarray(val)), _c_ref_char(name), rtol=1e-12
        )


# --------------------- forced-backend fixes on a NON-integrated orbit ----------
# Under `use(<backend>, force=True)` two fixes land:
#   1. derived accessors whose 1-D result is reversed with a raw `.T`
#      (x/y/vx/vy/r/vr/vtheta/theta) -> `_backend_T` (removes a torch
#      warn-once UserWarning; no value or return-type change), and
#   2. o.rperi/rap/zmax(analytic=True), whose numpy/C Staeckel machinery crashed
#      on a backend delta/Einf/energy mask (numpy.all/isnan/sum-on-a-tensor) ->
#      `as_numpy` at those boundaries, so the analytic path stays numpy and
#      returns a numpy scalar.
# The full E/ER/Ez/Jacobi energy-accessor backend migration is a separate
# follow-up PR; those accessors are UNCHANGED here (they behave as on the base).
from galpy import backend as _galpy_backend  # noqa: E402
from galpy.potential import LogarithmicHaloPotential  # noqa: E402

_LP = LogarithmicHaloPotential(normalize=1.0, q=0.9)
_IC_E = [1.0, 0.2, 0.9, 0.1, 0.05, 0.0]  # R, vR, vT, z, vz, phi (bound, z!=0)
_ANALYTIC = ["rperi", "rap", "zmax"]
# derived accessors whose 1-D result used a raw `.T` (torch warn-once deprecation)
_TSIB = ["x", "y", "vx", "vy", "r", "vr", "vtheta", "theta"]


def _analytic_np_ref(name):
    return float(
        numpy.asarray(
            getattr(Orbit(list(_IC_E)), name)(
                analytic=True, pot=_LP, use_physical=False
            )
        )
    )


# The analytic path calls o.E internally; under a forced backend that (still
# unmigrated) energy assembly emits the numpy-2 __array_wrap__ DeprecationWarning
# -- a deferred, separate issue, NOT introduced by the analytic fix -- so tolerate
# it here so the -W error coverage shard stays green.
@pytest.mark.filterwarnings("ignore:.*__array_wrap__.*:DeprecationWarning")
@pytest.mark.skipif(not HAVE_JAX, reason="jax not installed")
@pytest.mark.parametrize("name", _ANALYTIC)
def test_analytic_char_forced_jax_matches_numpy(name):
    # the analytic Staeckel path is a numpy/C computation -> returns a numpy scalar
    with _galpy_backend.use("jax", force=True):
        val = getattr(Orbit(list(_IC_E)), name)(
            analytic=True, pot=_LP, use_physical=False
        )
    assert isinstance(val, (float, numpy.floating, numpy.ndarray))
    numpy.testing.assert_allclose(
        float(numpy.asarray(val)), _analytic_np_ref(name), rtol=1e-10, atol=1e-12
    )


@pytest.mark.filterwarnings("ignore:.*__array_wrap__.*:DeprecationWarning")
@pytest.mark.skipif(not HAVE_TORCH, reason="torch not installed")
@pytest.mark.parametrize("name", _ANALYTIC)
def test_analytic_char_forced_torch_matches_numpy(name):
    with _galpy_backend.use("torch", force=True):
        val = getattr(Orbit(list(_IC_E)), name)(
            analytic=True, pot=_LP, use_physical=False
        )
    assert isinstance(val, (float, numpy.floating, numpy.ndarray))
    numpy.testing.assert_allclose(
        float(numpy.asarray(val)), _analytic_np_ref(name), rtol=1e-10, atol=1e-12
    )


@pytest.mark.skipif(not HAVE_JAX, reason="jax not installed")
@pytest.mark.parametrize("acc", _TSIB)
def test_sibling_accessor_forced_jax_matches_numpy(acc):
    # derived accessors whose 1-D result is reversed with `.T` -> a backend array
    # on the same (correct) values as numpy (the `.T`-on-1D fix).
    ref = numpy.asarray(getattr(Orbit(list(_IC_E)), acc)(use_physical=False))
    with _galpy_backend.use("jax", force=True):
        val = getattr(Orbit(list(_IC_E)), acc)(use_physical=False)
    assert isinstance(val, jax.Array), f"{acc} left the jax backend"
    got = numpy.asarray(val)
    if acc in _WRAP:
        got, ref = _wrap(got), _wrap(ref)
    numpy.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not HAVE_TORCH, reason="torch not installed")
@pytest.mark.parametrize("acc", _TSIB)
def test_sibling_accessor_forced_torch_matches_numpy(acc):
    ref = numpy.asarray(getattr(Orbit(list(_IC_E)), acc)(use_physical=False))
    with _galpy_backend.use("torch", force=True):
        val = getattr(Orbit(list(_IC_E)), acc)(use_physical=False)
    assert isinstance(val, torch.Tensor), f"{acc} left the torch backend"
    got = val.detach().cpu().numpy()
    if acc in _WRAP:
        got, ref = _wrap(got), _wrap(ref)
    numpy.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)


def test_analytic_char_numpy_path_unchanged():
    # a numpy orbit's rperi/rap/zmax(analytic=True) are unaffected by the dispatch
    for name in _ANALYTIC:
        val = getattr(Orbit(list(_IC_E)), name)(
            analytic=True, pot=_LP, use_physical=False
        )
        assert isinstance(val, (float, numpy.floating, numpy.ndarray))
        assert numpy.isfinite(float(numpy.asarray(val)))


# --- numerical-reduction accessors under a FORCED backend on a numpy orbit ------
# zmax/rap/rperi/e reduce the integrated trajectory with xp.max/min/abs. Under a
# forced backend get_namespace(numpy_zs) resolves to that backend, so
# torch.abs(numpy)/torch.max(numpy) raised -- the exact crash that blocked the
# streamdf progenitor setup (zmax on a numpy orbit) under forced torch. The
# data-guard (is_backend_array(data) ? backend : numpy) keeps a numpy orbit numpy.
_NUM_ACCESSORS = ["zmax", "rap", "rperi", "e"]
_FORCE_BACKENDS = [b for b, h in [("jax", HAVE_JAX), ("torch", HAVE_TORCH)] if h]


@pytest.mark.parametrize("acc", _NUM_ACCESSORS)
@pytest.mark.parametrize("backend_name", _FORCE_BACKENDS)
def test_numerical_accessor_forced_backend_on_numpy_orbit(acc, backend_name):
    o = Orbit(list(_IC))
    o.turn_physical_off()
    o.integrate(_TS, _POT, method="dop853_c")  # numpy integration
    ref = float(getattr(o, acc)())
    with use(backend_name, force=True):
        raw = getattr(o, acc)()
        # the point of the forced backend is that the accessor RUNS on the
        # backend (coerces the numpy trajectory onto it), not that it silently
        # falls back to numpy -- lock that in.
        assert is_backend_array(raw), f"{acc} forced-{backend_name} not on backend"
        got = float(as_numpy(raw))
    numpy.testing.assert_allclose(
        got,
        ref,
        rtol=1e-12,
        atol=1e-14,
        err_msg=f"{acc} forced-{backend_name} numpy orbit",
    )


# --- no-time accessors and o() keep a backend orbit on its backend -----------
# o.R() (no time argument) returned the numpy self.vxvv bookkeeping even for a
# jax/torch orbit, so it disagreed with o.R(t) / o.R(0.0) / o.E() for the SAME
# orbit, and o() laundered a backend orbit into a numpy one. streamdf inherits
# that: `self._progenitor = progenitor()` is the first line of its setup, so the
# whole track determination silently ran on numpy for a backend progenitor.
_IC_IDX = {"R": 0, "vR": 1, "vT": 2, "z": 3, "vz": 4, "phi": 5}


@pytest.mark.skipif(not HAVE_JAX, reason="jax not installed")
@pytest.mark.parametrize("acc", list(_IC_IDX))
@pytest.mark.parametrize("integrated", [False, True])
def test_noarg_accessor_stays_on_backend(acc, integrated):
    o = Orbit(jnp.asarray(_IC))
    if integrated:
        o.integrate(jnp.asarray(_TS), _POT, method="diffrax")
    val = getattr(o, acc)(use_physical=False)
    assert is_backend_array(val), f"{acc}() left the backend"
    # value is the IC component, and agrees with the timed accessor at t=0
    numpy.testing.assert_allclose(
        numpy.asarray(as_numpy(val)).reshape(()), _IC[_IC_IDX[acc]], rtol=1e-14
    )
    if integrated:
        at0 = getattr(o, acc)(jnp.asarray(_TS)[0], use_physical=False)
        numpy.testing.assert_allclose(
            numpy.asarray(as_numpy(val)).reshape(()),
            numpy.asarray(as_numpy(at0)).reshape(()),
            rtol=1e-12,
        )


@pytest.mark.skipif(not HAVE_JAX, reason="jax not installed")
def test_call_preserves_backend_ic():
    o = Orbit(jnp.asarray(_IC))
    oc = o()
    assert is_backend_array(oc._ic_backend), "o() laundered the backend IC to numpy"
    numpy.testing.assert_allclose(
        numpy.asarray(as_numpy(oc._ic_backend)).reshape(-1), _IC, rtol=1e-14
    )


@pytest.mark.skipif(not HAVE_JAX, reason="jax not installed")
def test_backend_ic_vxvv_bookkeeping_is_writable():
    # self.vxvv is numpy bookkeeping; numpy.asarray on a jax array is a READ-ONLY
    # view, which made in-place numpy math on it (streamdf's calcaAJac finite
    # differences: xv[ii] += dxv[ii]) raise "assignment destination is read-only".
    o = Orbit(jnp.asarray(_IC))
    assert o.vxvv.flags.writeable
    o.vxvv[0] += 1.0  # must not raise


def test_noarg_accessor_numpy_path_unchanged():
    o = Orbit(numpy.array(_IC))
    for acc, i in _IC_IDX.items():
        v = getattr(o, acc)(use_physical=False)
        assert not is_backend_array(v)
        numpy.testing.assert_allclose(numpy.asarray(v).reshape(()), _IC[i], rtol=1e-14)


@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_accessor_on_a_traced_integration_grid(direction):
    # streamdf integrates its track orbit on theta-DEPENDENT times, so the
    # orbit's own self.t is a tracer. The accessor then cannot compare the grid
    # to the query, order it by a concrete `t[1] < t[0]`, or hand the spline
    # numpy knots -- it orders with argsort and keeps the knots on the backend.
    # Both grid directions go through that branch (backward -> argsort reverses
    # the grid AND the trajectory rows).
    ibk = {"nsteps": 40, "adjoint": "direct", "max_steps": 60}

    def f(scale):
        with use("jax", force=True):
            o = Orbit(jnp.asarray([1.0, 0.1, 1.1, 0.05, -0.02, 0.3]))
            ts = direction * scale * jnp.linspace(0.0, 1.0, 21)
            o.integrate(ts, _POT, method="diffrax", inbackend_kwargs=ibk)
            return jnp.sum(o.R(ts * 0.5) ** 2)  # OFF-grid query on a traced grid

    ad = float(jax.grad(f)(2.0))
    prev = None
    for h in (1e-3, 1e-4):
        fd = (float(f(2.0 + h)) - float(f(2.0 - h))) / (2.0 * h)
        rel = abs(ad - fd) / abs(fd)
        if prev is not None:
            assert rel < prev, "the AD-vs-FD gap must shrink as h does"
        prev = rel
    assert rel < 1e-8, f"d/d(scale) through a traced grid is wrong (rel {rel:.2e})"


# ---------------------------------------------------------------------------
# d/d(POTENTIAL PARAMETER) through the analytic (un-integrated) accessors
#
# These reach the potential three different ways, and each had its own numpy
# cut before this was tested:
#   rguiding/rE/LcE -> Potential.rl/rE, whose backend dispatch keyed on the
#                      root-finder's ARGUMENT (lz, E), missing a traced
#                      potential; then numpy.array([...]) could not collect the
#                      backend scalars the fixed rl/rE return.
#   rperi/rap/zmax/e -> actionAngleStaeckel with an AUTOMAGIC delta, which
#                      reads the potential's second derivatives and so carries
#                      the gradient; it was cut with as_numpy/atleast_1d.
# c=False throughout: the C path cannot carry potential derivatives by design.
# ---------------------------------------------------------------------------
_PGRAD_B0 = 0.6
_PGRAD_IC = [1.0, 0.12, 1.08, 0.06, 0.09, 0.3]
_PGRAD_ACCESSORS = ["rguiding", "rE", "LcE", "rperi", "rap", "zmax", "e"]


def _accessor_of_b(name, b, cast):
    from galpy.potential import PlummerPotential as _PP

    pot = _PP(amp=1.0, b=b)
    o = Orbit(cast(_PGRAD_IC))
    kw = dict(pot=pot, use_physical=False)
    if name in ("rperi", "rap", "zmax", "e"):
        kw.update(analytic=True, c=False)
    out = getattr(o, name)(**kw)
    arr = out if is_backend_array(out) else numpy.atleast_1d(out)
    return arr.reshape(()) if getattr(arr, "shape", ()) == (1,) else arr


@pytest.mark.filterwarnings("ignore:.*requires_grad.*:UserWarning")
@pytest.mark.parametrize("name", _PGRAD_ACCESSORS)
def test_accessor_grad_wrt_potential_parameter(name):
    h = 1e-6 * _PGRAD_B0
    fd = (
        float(numpy.asarray(_accessor_of_b(name, _PGRAD_B0 + h, numpy.array)))
        - float(numpy.asarray(_accessor_of_b(name, _PGRAD_B0 - h, numpy.array)))
    ) / (2.0 * h)
    assert abs(fd) > 1e-6, "finite difference is ~0: the test would prove nothing"
    if HAVE_JAX:
        with use("jax", force=True):
            ad = float(
                jax.grad(lambda t: _accessor_of_b(name, t, jnp.asarray))(
                    jnp.asarray(_PGRAD_B0)
                )
            )
        assert numpy.isfinite(ad), (name, ad)
        assert ad != 0.0, (name, "gradient is identically zero (detached?)")
        numpy.testing.assert_allclose(ad, fd, rtol=2e-4, atol=1e-8)
    if HAVE_TORCH:
        with use("torch", force=True):
            t = torch.tensor(_PGRAD_B0, dtype=torch.float64, requires_grad=True)
            _accessor_of_b(
                name, t, lambda v: torch.as_tensor(numpy.asarray(v, dtype=float))
            ).backward()
            ad_t = float(t.grad)
        assert numpy.isfinite(ad_t), (name, ad_t)
        numpy.testing.assert_allclose(ad_t, fd, rtol=2e-4, atol=1e-8)
