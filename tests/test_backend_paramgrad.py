###############################################################################
# test_backend_paramgrad.py: autodiff w.r.t. potential *parameters* (d/dtheta).
#
# Coordinate-gradient autodiff (d Phi/dR == -Rforce) is covered in test_backend.py.
# This module proves the complementary capability: differentiating a potential
# w.r.t. its *constructor parameters* (amp, scale lengths, ...). The enabler is
# that galpy's unit-parsing layer (conversion.parse_*) now passes backend arrays
# (jax/torch, including traced ones) through unscaled, so a parameter supplied as
# a tracer survives construction and the gradient flows through _evaluate/_Rforce.
#
# Usage contract exercised here: the *coordinates* are supplied as backend arrays
# too, so the namespace resolver follows the data into jax/torch (a numpy-float
# coordinate would pin the namespace to numpy and choke on the traced parameter).
#
# Backends that are not installed self-skip, so this is green on numpy alone.
###############################################################################
import numpy
import pytest

from galpy import backend
from galpy.potential import (
    IsochronePotential,
    PlummerPotential,
    evaluatePotentials,
    evaluateRforces,
    evaluatezforces,
)

# This module manages backends explicitly, so it is exempt from the global
# --backend force fixture.
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

# (constructor, fixed kwargs, parameter name, parameter value) -- the named
# parameter is the one differentiated; the rest are held fixed.
PARAM_SPECS = [
    (PlummerPotential, {"amp": 1.0}, "b", 0.7),
    (PlummerPotential, {"b": 0.7}, "amp", 1.3),
    (IsochronePotential, {"amp": 1.0}, "b", 1.1),
    (IsochronePotential, {"b": 1.1}, "amp", 2.0),
]
SPEC_IDS = [f"{c.__name__}-d{p}" for c, _, p, _ in PARAM_SPECS]

# the per-potential scalar quantity differentiated. These are the *public*
# evaluators (evaluatePotentials/Rforces/zforces) -- crucially they apply the
# amplitude (self._amp), so gradients w.r.t. amp flow; the private _evaluate etc.
# omit amp and would give a zero/disconnected gradient for the amp parameter.
METHODS = {
    "Phi": evaluatePotentials,
    "Rforce": evaluateRforces,
    "zforce": evaluatezforces,
}
METHOD_IDS = list(METHODS)

_R0, _Z0 = 1.2, 0.3
# Step for the Richardson finite-difference reference below. A plain central
# difference at eps=1e-6 has a relative truncation+rounding error of ~2e-10 here,
# which would force a loose AD-vs-FD tolerance. Richardson-extrapolating two
# central differences (at _EPS and _EPS/2) cancels the leading O(eps^2) term and
# drops the reference error to ~5e-12, letting us assert at rtol=1e-9 (~190x
# margin) while staying robust to cross-platform float rounding.
_EPS = 1e-4


def _value(ctor, fixed, pname, theta, method, xp_R, xp_z):
    pot = ctor(**{**fixed, pname: theta})
    return METHODS[method](pot, xp_R, xp_z)


def _fd_reference(ctor, fixed, pname, th0, method):
    # Richardson-extrapolated central finite difference (O(eps^4) accurate),
    # computed entirely on the pure-numpy path.
    def fnp(theta):
        return float(_value(ctor, fixed, pname, theta, method, _R0, _Z0))

    d1 = (fnp(th0 + _EPS) - fnp(th0 - _EPS)) / (2 * _EPS)
    d2 = (fnp(th0 + _EPS / 2) - fnp(th0 - _EPS / 2)) / _EPS
    return (4 * d2 - d1) / 3


def _ad_grad(backend_name, ctor, fixed, pname, th0, method):
    if backend_name == "jax":
        R, z = jnp.asarray(_R0), jnp.asarray(_Z0)
        return float(
            jax.grad(lambda th: _value(ctor, fixed, pname, th, method, R, z))(
                jnp.asarray(th0)
            )
        )
    R, z = torch.as_tensor(_R0), torch.as_tensor(_Z0)
    th = torch.tensor(th0, requires_grad=True)
    _value(ctor, fixed, pname, th, method, R, z).backward()
    return float(th.grad)


@pytest.mark.parametrize("method", METHOD_IDS)
@pytest.mark.parametrize("spec", PARAM_SPECS, ids=SPEC_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_param_grad_vs_finite_difference(backend_name, spec, method):
    ctor, fixed, pname, th0 = spec
    fd = _fd_reference(ctor, fixed, pname, th0, method)
    ad = _ad_grad(backend_name, ctor, fixed, pname, th0, method)
    # The AD value is exact; the limiting error is the finite-difference
    # reference (~5e-12 relative, ~2.5e-12 absolute here). rtol=1e-9 dominates
    # (gradient magnitudes are all >~0.02) and keeps a ~190x margin; atol=1e-11
    # is small enough not to mask a wrong gradient for the O(1) cases.
    numpy.testing.assert_allclose(ad, fd, rtol=1e-9, atol=1e-11)


@pytest.mark.skipif(
    "jax" not in BACKENDS or "torch" not in BACKENDS,
    reason="needs both jax and torch",
)
@pytest.mark.parametrize("method", METHOD_IDS)
@pytest.mark.parametrize("spec", PARAM_SPECS, ids=SPEC_IDS)
def test_param_grad_jax_vs_torch(spec, method):
    # The sharpest, finite-difference-independent check: both backends compute
    # the *exact* derivative by autodiff, so they must agree to ~machine
    # precision (~1e-16 measured). A subtly-wrong gradient in either backend's
    # parse/evaluate path shows up here far more sensitively than against the
    # FD reference. rtol=1e-10 leaves a huge (~1e6) margin over the observed
    # agreement while still being orders of magnitude tighter than AD-vs-FD.
    ctor, fixed, pname, th0 = spec
    g_jax = _ad_grad("jax", ctor, fixed, pname, th0, method)
    g_torch = _ad_grad("torch", ctor, fixed, pname, th0, method)
    numpy.testing.assert_allclose(g_jax, g_torch, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("spec", PARAM_SPECS, ids=SPEC_IDS)
def test_numpy_construction_unaffected(spec):
    # The parse-path change must not perturb the plain-float (numpy) path: a
    # potential built with Python-float parameters and evaluated on float
    # coordinates returns the exact same value as before (byte-identical).
    ctor, fixed, pname, th0 = spec
    pot = ctor(**{**fixed, pname: th0})
    v_float = pot._evaluate(_R0, _Z0)
    v_array = numpy.asarray(pot._evaluate(numpy.asarray([_R0]), numpy.asarray([_Z0])))
    numpy.testing.assert_array_equal(
        numpy.asarray(float(v_float)), numpy.asarray(v_array[0])
    )


def test_is_backend_array_detection():
    # numpy / scalars / None are never backend arrays (so the numpy path is
    # untouched); genuine jax/torch arrays are.
    assert not backend.is_backend_array(1.0)
    assert not backend.is_backend_array(None)
    assert not backend.is_backend_array(numpy.ones(3))
    assert not backend.is_backend_array(numpy.float64(2.0))
    if "jax" in BACKENDS:
        assert backend.is_backend_array(jnp.asarray(1.0))
        assert backend.is_backend_array(jnp.ones(3))
    if "torch" in BACKENDS:
        assert backend.is_backend_array(torch.as_tensor(1.0))
        assert backend.is_backend_array(torch.ones(3, requires_grad=True))


@pytest.mark.skipif("jax" not in BACKENDS, reason="jax not installed")
def test_param_grad_under_jit_and_vmap():
    # The parameter gradient survives jit, and vmaps over a batch of parameter
    # values (the shape a gradient-descent parameter fit would use).
    R, z = jnp.asarray(_R0), jnp.asarray(_Z0)

    def phi_of_b(b):
        return PlummerPotential(amp=1.0, b=b)._evaluate(R, z)

    g = jax.jit(jax.grad(phi_of_b))
    bs = jnp.asarray([0.5, 0.7, 1.0])
    grads = numpy.asarray(jax.vmap(g)(bs))
    fd = numpy.array(
        [
            (
                float(PlummerPotential(amp=1.0, b=float(b) + _EPS)._evaluate(_R0, _Z0))
                - float(
                    PlummerPotential(amp=1.0, b=float(b) - _EPS)._evaluate(_R0, _Z0)
                )
            )
            / (2 * _EPS)
            for b in bs
        ]
    )
    numpy.testing.assert_allclose(grads, fd, rtol=1e-5, atol=1e-8)


@pytest.mark.skipif("jax" not in BACKENDS, reason="needs jax")
@pytest.mark.parametrize("method", ["zforce", "dens", "z2deriv", "Rzderiv"])
def test_miyamotonagai_differentiates_in_a_under_grad_and_jit(method):
    # These four branch on `if self._a == 0.0`, which has no concrete value when
    # `a` is the parameter being fitted. Under plain grad that used to work by
    # accident (grad linearizes with concrete primals); inside a compiled region
    # -- an ODE solve -- it raised TracerBoolConversionError, so a differentiable
    # orbit fit in MiyamotoNagai w.r.t. `a` was impossible. MWPotential2014 uses
    # this potential, so that is the common case.
    from galpy.potential import MiyamotoNagaiPotential

    R, Z = jnp.asarray(1.1), jnp.asarray(0.2)

    def f(a):
        p = MiyamotoNagaiPotential(amp=1.0, a=a, b=0.1)
        return jnp.asarray(getattr(p, method)(R, Z))[()]

    ad = float(jax.grad(f)(0.5))
    h = 1e-6
    fd = (float(f(0.5 + h)) - float(f(0.5 - h))) / (2.0 * h)
    assert abs(ad - fd) / abs(fd) < 1e-6, f"d/d(a) {method} wrong (AD {ad}, FD {fd})"
    # and the compiled path must agree with the eager one, not just run
    assert abs(float(jax.jit(jax.grad(f))(0.5)) - ad) < 1e-12, (
        f"jit(grad) must match grad for {method}"
    )
    # the a==0 SPECIAL case is still taken when a is concrete
    p0 = MiyamotoNagaiPotential(amp=1.0, a=0.0, b=0.1)
    assert numpy.isfinite(float(numpy.asarray(getattr(p0, method)(1.1, 0.2))))


@pytest.mark.skipif("jax" not in BACKENDS, reason="needs jax")
def test_miyamotonagai_orbit_integration_differentiates_in_a():
    # the point of the above: an ODE solve in MiyamotoNagai, differentiated
    # w.r.t. the scale length
    from galpy.orbit import Orbit
    from galpy.potential import MiyamotoNagaiPotential

    ts = jnp.asarray(numpy.linspace(0.0, 1.0, 21))

    def f(a):
        o = Orbit(jnp.asarray([1.0, 0.1, 1.1, 0.05, -0.02, 0.3]))
        o.integrate(ts, MiyamotoNagaiPotential(amp=1.0, a=a, b=0.1), method="diffrax")
        return jnp.sum(o.R(ts) ** 2)

    ad = float(jax.grad(f)(0.5))
    h = 1e-5
    fd = (float(f(0.5 + h)) - float(f(0.5 - h))) / (2.0 * h)
    assert abs(ad - fd) / abs(fd) < 1e-5, f"d/d(a) of the orbit wrong ({ad} vs {fd})"


@pytest.mark.skipif("jax" not in BACKENDS, reason="needs jax")
def test_mn3_exponentialdisk_differentiates_in_hz():
    # _brd is derived from hz/hr and was range-CHECKED with Python comparisons
    # (a raise and a warning). Those are validity checks, not model branches, so
    # they are skipped for a traced parameter rather than made traceable.
    from galpy.potential import MN3ExponentialDiskPotential

    R, Z = jnp.asarray(1.1), jnp.asarray(0.2)

    def f(hz):
        return jnp.asarray(
            MN3ExponentialDiskPotential(amp=1.0, hr=1.0, hz=hz).Rforce(R, Z)
        )[()]

    ad = float(jax.grad(f)(0.1))
    h = 1e-7
    fd = (float(f(0.1 + h)) - float(f(0.1 - h))) / (2.0 * h)
    assert abs(ad - fd) / abs(fd) < 1e-6
    assert abs(float(jax.jit(jax.grad(f))(0.1)) - ad) < 1e-12


@pytest.mark.skipif("jax" not in BACKENDS, reason="needs jax")
def test_chandrasekhar_differentiates_in_rhm():
    # the rhm==0 arm is the GMvs>=rhm branch of the where with the 1/rhm removed
    from galpy.potential import (
        ChandrasekharDynamicalFrictionForce,
        HernquistPotential,
    )

    hp = HernquistPotential(amp=1.0, a=2.0)
    v = jnp.asarray([0.1, 1.0, 0.05])

    def f(rhm):
        cdf = ChandrasekharDynamicalFrictionForce(amp=1.0, GMs=0.01, rhm=rhm, dens=hp)
        return jnp.asarray(cdf.Rforce(jnp.asarray(1.1), jnp.asarray(0.2), v=v))[()]

    ad = float(jax.grad(f)(0.05))
    h = 1e-8
    fd = (float(f(0.05 + h)) - float(f(0.05 - h))) / (2.0 * h)
    assert abs(ad - fd) / abs(fd) < 1e-6, f"d/d(rhm) wrong (AD {ad}, FD {fd})"
