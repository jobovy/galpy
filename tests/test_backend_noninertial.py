###############################################################################
# test_backend_noninertial.py: multi-backend tests for NonInertialFrameForce.
#
# NonInertialFrameForce._force built its fictitious-force vector with bare
# numpy (numpy.zeros / numpy.array / numpy.dot / numpy.linalg.norm) plus an
# md5 input cache keyed on numpy.array([...]). On a jax array that both
# DETACHED the result back to a numpy scalar (eager) and broke under a trace
# (the md5 numpy.array on a tracer raised TracerArrayConversionError), so
# jax.jit / jax.jacfwd of evaluateRforces were impossible; torch grad tensors
# hit "Can't call numpy() on Tensor that requires grad".
#
# The compute path is now resolved through galpy.backend (get_namespace on the
# coordinate + velocity inputs), with the md5 cache confined to the numpy path.
# For every supported configuration (scalar/vector Omega, constant or with
# Omegadot, Omega-as-function-of-time, and the a0/x0/v0 translation terms) this
# proves:
#   1. numpy / jax / torch value parity for Rforce / phitorque / zforce;
#   2. eager jax returns a jax array and eager torch a torch tensor (a bare
#      numpy path would silently detach jax);
#   3. jax.jit AND jax.jacfwd of evaluateRforces survive and return finite
#      (the exact gap: velocity-dependent force, so v= is passed through);
#   4. the force gradient matches a central finite difference (random
#      directional derivative over R,z,phi,t,v), on jax and torch.
#
# Backends that are not installed self-skip, so this is green on numpy alone.
###############################################################################
import numpy
import pytest

from galpy.backend import as_numpy, use
from galpy.potential import (
    NonInertialFrameForce,
    evaluatephitorques,
    evaluateRforces,
    evaluatezforces,
)

# This module manages backends explicitly, so it is exempt from the global
# --backend force fixture.
pytestmark = pytest.mark.backend_managed

# Discover available backends
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


def _scalar(backend_name, x):
    if backend_name == "numpy":
        return numpy.asarray(x, dtype=float)
    if backend_name == "jax":
        return jnp.asarray(x, dtype=jnp.float64)
    return torch.tensor(x, dtype=torch.float64)


def _module_of(x):
    return type(x).__module__


# A fresh potential per case so the numpy input cache never leaks between
# parametrizations. Every configuration exercises a distinct _force branch.
def _build(name):
    if name == "scalar_const":
        return NonInertialFrameForce(cinterp=False, Omega=1.3)
    if name == "scalar_odot":
        return NonInertialFrameForce(cinterp=False, Omega=1.3, Omegadot=0.2)
    if name == "vec_const":
        return NonInertialFrameForce(cinterp=False, Omega=numpy.array([0.1, 0.2, 1.3]))
    if name == "vec_odot":
        return NonInertialFrameForce(
            cinterp=False,
            Omega=numpy.array([0.1, 0.2, 1.3]),
            Omegadot=numpy.array([0.01, 0.02, 0.03]),
        )
    if name == "scalarfunc":
        return NonInertialFrameForce(
            cinterp=False, Omega=lambda t: 1.3 + 0.1 * t, Omegadot=lambda t: 0.1
        )
    if name == "vecfunc":
        return NonInertialFrameForce(
            cinterp=False,
            Omega=[lambda t: 0.1 + 0.01 * t, lambda t: 0.2, lambda t: 1.3 - 0.02 * t],
            Omegadot=[lambda t: 0.01, lambda t: 0.0, lambda t: -0.02],
        )
    if name == "a0_x0v0":
        return NonInertialFrameForce(
            cinterp=False,
            Omega=1.3,
            a0=[0.1, 0.2, -0.1],
            x0=[lambda t: 0.05 * t, lambda t: 0.02, lambda t: 0.01 * t],
            v0=[lambda t: 0.05, lambda t: 0.0, lambda t: 0.01],
        )
    raise ValueError(name)  # pragma: no cover


_CONFIGS = [
    "scalar_const",
    "scalar_odot",
    "vec_const",
    "vec_odot",
    "scalarfunc",
    "vecfunc",
    "a0_x0v0",
]

# A generic non-degenerate evaluation point plus the three cylindrical force
# methods; R,z,phi,t and the cylindrical velocity v are all non-trivial.
_R0, _Z0, _PHI0, _T0 = 1.1, 0.2, 0.4, 0.3
_V0 = [0.1, 1.0, 0.05]
_W = numpy.array([0.7, -1.3, 0.9])  # loss weights over (Rforce, phitorque, zforce)


def _forces_backend(backend_name, pot, R, z, phi, t, v):
    Rb = _scalar(backend_name, R)
    zb = _scalar(backend_name, z)
    phib = _scalar(backend_name, phi)
    tb = _scalar(backend_name, t)
    vb = [_scalar(backend_name, vv) for vv in v]
    fr = evaluateRforces(pot, Rb, zb, phi=phib, t=tb, v=vb)
    fp = evaluatephitorques(pot, Rb, zb, phi=phib, t=tb, v=vb)
    fz = evaluatezforces(pot, Rb, zb, phi=phib, t=tb, v=vb)
    return fr, fp, fz


def _forces_numpy(pot, R, z, phi, t, v):
    fr = evaluateRforces(pot, R, z, phi=phi, t=t, v=list(v))
    fp = evaluatephitorques(pot, R, z, phi=phi, t=t, v=list(v))
    fz = evaluatezforces(pot, R, z, phi=phi, t=t, v=list(v))
    return float(fr), float(fp), float(fz)


@pytest.mark.parametrize("name", _CONFIGS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_force_value_parity(backend_name, name):
    pot = _build(name)
    ref = _forces_numpy(pot, _R0, _Z0, _PHI0, _T0, _V0)
    pot_b = _build(name)
    got = [
        float(as_numpy(f))
        for f in _forces_backend(backend_name, pot_b, _R0, _Z0, _PHI0, _T0, _V0)
    ]
    numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14, err_msg=name)


@pytest.mark.parametrize("name", _CONFIGS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_force_returns_backend_array(backend_name, name):
    # A bare numpy._force would detach jax to a numpy scalar; the swept path
    # must return the native backend array for all three force components.
    pot = _build(name)
    for f in _forces_backend(backend_name, pot, _R0, _Z0, _PHI0, _T0, _V0):
        assert backend_name in _module_of(f), (
            f"{name}: force left the {backend_name} namespace ({_module_of(f)})"
        )


@pytest.mark.parametrize("name", _CONFIGS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_force_numpy_scalar_under_forced_backend(backend_name, name):
    # Regression (scalar-under-forced-backend): the python orbit integrator
    # feeds _force numpy-scalar coords/velocities under a FORCED backend -- the
    # DissipativeForce RHS bypasses the @physical_input coercion gate. get_namespace
    # then resolves to the backend while the coords stay numpy, so a strict-typed
    # call raised: for the scalar-function config, tOmega = Omega(t) is a numpy
    # scalar and torch.linalg.norm rejected it. Assert no raise, a backend array,
    # and numpy value parity (a fresh pot each time so the md5 cache never leaks).
    ref = numpy.asarray(_build(name)._force(_R0, _Z0, _PHI0, _T0, list(_V0)))
    pot = _build(name)
    R = numpy.float64(_R0)
    z = numpy.float64(_Z0)
    phi = numpy.float64(_PHI0)
    t = numpy.float64(_T0)
    v = [numpy.float64(vv) for vv in _V0]
    with use(backend_name, force=True):
        f = pot._force(R, z, phi, t, v)
    assert backend_name in _module_of(f), (
        f"{name}: forced-{backend_name} _force is {_module_of(f)}, expected a backend array"
    )
    got = numpy.asarray([float(as_numpy(fi)) for fi in f])
    numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14, err_msg=name)


@pytest.mark.parametrize("name", _CONFIGS)
def test_force_jit_and_jacfwd_finite(name):
    # The exact gap: jax.jit / jax.jacfwd of a velocity-dependent evaluateRforces
    # (bare-numpy md5 cache used to raise TracerArrayConversionError under trace).
    if jax is None:  # pragma: no cover
        pytest.skip("jax not installed")
    pot = _build(name)
    zb = jnp.asarray(_Z0)
    phib = jnp.asarray(_PHI0)
    tb = jnp.asarray(_T0)
    vb = [jnp.asarray(vv) for vv in _V0]

    def f(R):
        return evaluateRforces(pot, R, zb, phi=phib, t=tb, v=vb)

    R = jnp.asarray(_R0)
    assert numpy.isfinite(float(jax.jit(f)(R))), name
    assert numpy.isfinite(float(jax.jacfwd(f)(R))), name
    # eager reference so the traced value is the right one
    numpy.testing.assert_allclose(float(jax.jit(f)(R)), float(f(R)), rtol=1e-12)


@pytest.mark.parametrize("name", _CONFIGS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_force_grad_vs_finite_difference(backend_name, name):
    # Directional derivative of the weighted force loss over (R,z,phi,t,v)
    # against a central finite difference of the numpy path.
    pot = _build(name)
    x0 = numpy.array([_R0, _Z0, _PHI0, _T0, *_V0], dtype=float)
    rng = numpy.random.default_rng(7)
    u = rng.standard_normal(7)
    u /= numpy.linalg.norm(u)

    def loss_np(x):
        fr, fp, fz = _forces_numpy(pot, x[0], x[1], x[2], x[3], x[4:7])
        return _W[0] * fr + _W[1] * fp + _W[2] * fz

    eps = 1e-6
    fd = (loss_np(x0 + eps * u) - loss_np(x0 - eps * u)) / (2 * eps)

    if backend_name == "jax":

        def loss_j(x):
            fr = evaluateRforces(
                pot, x[0], x[1], phi=x[2], t=x[3], v=[x[4], x[5], x[6]]
            )
            fp = evaluatephitorques(
                pot, x[0], x[1], phi=x[2], t=x[3], v=[x[4], x[5], x[6]]
            )
            fz = evaluatezforces(
                pot, x[0], x[1], phi=x[2], t=x[3], v=[x[4], x[5], x[6]]
            )
            return _W[0] * fr + _W[1] * fp + _W[2] * fz

        g = jax.grad(loss_j)(jnp.asarray(x0))
        ad = float(jnp.dot(g, jnp.asarray(u)))
    else:
        xt = torch.tensor(x0, requires_grad=True)
        fr = evaluateRforces(
            pot, xt[0], xt[1], phi=xt[2], t=xt[3], v=[xt[4], xt[5], xt[6]]
        )
        fp = evaluatephitorques(
            pot, xt[0], xt[1], phi=xt[2], t=xt[3], v=[xt[4], xt[5], xt[6]]
        )
        fz = evaluatezforces(
            pot, xt[0], xt[1], phi=xt[2], t=xt[3], v=[xt[4], xt[5], xt[6]]
        )
        loss = _W[0] * fr + _W[1] * fp + _W[2] * fz
        (gt,) = torch.autograd.grad(loss, xt)
        ad = float(numpy.dot(gt.numpy(), u))

    assert numpy.isfinite(ad), f"{backend_name} {name}: grad not finite"
    numpy.testing.assert_allclose(
        ad, fd, rtol=1e-5, atol=1e-8, err_msg=f"{backend_name} {name}"
    )
