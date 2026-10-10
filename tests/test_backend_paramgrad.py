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
# The coordinates may be backend arrays, Python floats (the @backend_input
# boundary then follows the differentiated PARAMETER onto its backend), or
# Python floats under a forced backend.
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
    assert numpy.isfinite(float(getattr(p0, method)(1.1, 0.2)))


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


# --------------------------------------------------------------------------
# Orbit's ANALYTIC estimators w.r.t. a potential parameter.
#
# Orbit.rperi/rap/zmax/e(analytic=True) and the analytic actions/frequencies/
# angles used to concretize the energy to build the bound/unbound mask
# (as_numpy(Einf)), which raises on a tracer and, on torch, silently DETACHED --
# rap w.r.t. MiyamotoNagai a came back 0 against a finite difference of 6.77.
# The mask is a discrete selection carrying no gradient, so a differentiated
# energy now skips it and every orbit is evaluated.
#
# A Python actionAngle is required: the C implementations cannot carry
# d/d(potential parameter) by construction.
# --------------------------------------------------------------------------
_ORB_IC = [1.0, 0.1, 0.5, 0.05, 0.03, 0.0]  # bound in the Hernquist below
_ORB_A0 = 1.3


def _orb_analytic(a, method, cast):
    from galpy.orbit import Orbit
    from galpy.potential import HernquistPotential

    pot = HernquistPotential(amp=2.0, a=a)
    o = Orbit(cast(_ORB_IC))
    out = getattr(o, method)(
        analytic=True, pot=pot, type="spherical", use_physical=False
    )
    return out.reshape(-1)[0] if numpy.ndim(out) else out


@pytest.mark.parametrize("method", ["rperi", "rap", "zmax", "e", "jr", "Or", "wr"])
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_orbit_analytic_grad_wrt_potential_parameter(backend_name, method):
    def value(a, bk, cast):
        with backend.use(bk, force=True):
            return _orb_analytic(a, method, cast)

    h = 1e-5 * _ORB_A0
    npcast = numpy.array
    fd = (
        float(value(_ORB_A0 + h, "numpy", npcast))
        - float(value(_ORB_A0 - h, "numpy", npcast))
    ) / (2.0 * h)
    if backend_name == "jax":
        ad = float(
            jax.grad(lambda t: value(t, "jax", jnp.asarray))(jnp.asarray(_ORB_A0))
        )
    else:
        t = torch.tensor(_ORB_A0, dtype=torch.float64, requires_grad=True)
        out = value(
            t, "torch", lambda v: torch.as_tensor(numpy.asarray(v, dtype=float))
        )
        out.backward()
        ad = float(t.grad)
    assert numpy.isfinite(ad), f"{method}: gradient must not be nan/inf"
    # a DETACHED gradient is the failure this guards: it returns a finite 0
    assert abs(ad) > 0.0, f"{method}: gradient is identically zero (detached?)"
    numpy.testing.assert_allclose(ad, fd, rtol=1e-6, atol=1e-10)


# ---------------------------------------------------------------------------
# Python-float coordinates with a differentiated parameter (no use() block):
# the @backend_input boundary lifts the coordinates onto the parameter's
# backend. Before, `jax.grad(lambda a: MiyamotoNagaiPotential(a=a)(1.0, 0.1))`
# raised (numpy.sqrt of a tracer).
# ---------------------------------------------------------------------------
def _mn(**kw):
    from galpy.potential import MiyamotoNagaiPotential

    return MiyamotoNagaiPotential(**kw)


def _nfw(**kw):
    from galpy.potential import NFWPotential

    return NFWPotential(**kw)


def _loghalo(**kw):
    from galpy.potential import LogarithmicHaloPotential

    return LogarithmicHaloPotential(**kw)


def _smooth_mn(a):
    # a wrapper: the differentiated parameter sits one level in
    from galpy.potential import DehnenSmoothWrapperPotential

    return DehnenSmoothWrapperPotential(
        pot=_mn(amp=1.0, a=a, b=0.3), tform=-1.0, tsteady=0.5
    )


def _composite_mn(a):
    # a composite: the parameter sits in one of its components
    return _mn(amp=1.0, a=a, b=0.3) + PlummerPotential(amp=0.5, b=0.7)


FLOAT_SPECS = [
    ("MN-a", lambda th: _mn(amp=1.0, a=th, b=0.3), 0.5),
    ("MN-b", lambda th: _mn(amp=1.0, a=0.5, b=th), 0.3),
    ("Plummer-b", lambda th: PlummerPotential(amp=1.0, b=th), 0.7),
    ("Isochrone-amp", lambda th: IsochronePotential(amp=th, b=1.1), 2.0),
    ("NFW-a", lambda th: _nfw(amp=1.0, a=th), 1.5),
    ("LogHalo-q", lambda th: _loghalo(amp=1.0, core=0.2, q=th), 0.8),
    ("DehnenSmooth(MN)-a", _smooth_mn, 0.5),
    ("MN+Plummer-a", _composite_mn, 0.5),
]
FLOAT_IDS = [s[0] for s in FLOAT_SPECS]


def _float_value(build, theta, method, R=_R0, z=_Z0):
    return METHODS[method](build(theta), R, z)


def _float_fd(build, th0, method):
    def fnp(theta):
        return float(_float_value(build, theta, method))

    d1 = (fnp(th0 + _EPS) - fnp(th0 - _EPS)) / (2 * _EPS)
    d2 = (fnp(th0 + _EPS / 2) - fnp(th0 - _EPS / 2)) / _EPS
    return (4 * d2 - d1) / 3


def _forced_ctx(backend_name, mode):
    import contextlib

    return (
        backend.use(backend_name, force=True)
        if mode == "forced"
        else (contextlib.nullcontext())
    )


@pytest.mark.parametrize("mode", ["float", "forced"])
@pytest.mark.parametrize("method", METHOD_IDS)
@pytest.mark.parametrize("spec", FLOAT_SPECS, ids=FLOAT_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_param_grad_at_float_coordinates(backend_name, spec, method, mode):
    _, build, th0 = spec
    fd = _float_fd(build, th0, method)
    if backend_name == "jax":

        def f(th):
            with _forced_ctx("jax", mode):
                out = _float_value(build, th, method)
            assert backend.is_backend_array(out)
            return out

        grads = [jax.grad(f)(jnp.asarray(th0)), jax.jit(jax.grad(f))(th0)]
        # a batch of parameter values under vmap(grad) (eager and jitted)
        thetas = jnp.asarray([th0, 1.1 * th0])
        vg = jax.vmap(jax.grad(f))(thetas)
        numpy.testing.assert_allclose(
            numpy.asarray(jax.jit(jax.vmap(jax.grad(f)))(thetas)), vg, rtol=1e-13
        )
        numpy.testing.assert_allclose(
            float(vg[1]), _float_fd(build, 1.1 * th0, method), rtol=1e-9, atol=1e-11
        )
        grads.append(vg[0])
    else:
        th = torch.tensor(th0, requires_grad=True)
        with _forced_ctx("torch", mode):
            out = _float_value(build, th, method)
        assert backend.is_backend_array(out)
        out.backward()
        grads = [th.grad]
    for g in grads:
        numpy.testing.assert_allclose(float(g), fd, rtol=1e-9, atol=1e-11)


@pytest.mark.skipif(torch is None, reason="needs torch")
def test_forced_numpy_beats_a_parameter_backend():
    # precedence: a forced backend beats the data, parameters included -- under
    # use("numpy", force=True) the coordinates stay numpy, so numpy meets the
    # grad tensor and refuses it
    pot = PlummerPotential(amp=1.0, b=torch.tensor(0.7, requires_grad=True))
    assert backend.is_backend_array(pot(_R0, _Z0))
    with backend.use("numpy", force=True):
        with pytest.raises(RuntimeError, match="requires grad"):
            pot(_R0, _Z0)


@pytest.mark.skipif(torch is None, reason="needs torch")
def test_parameter_cache_follows_amp_and_attributes():
    # the "no differentiated parameter" answer is cached on the object; a later
    # differentiated amp (normalize, a reassignment) or a new attribute
    # invalidates it
    from galpy.backend._input import PARAM_CACHE_ATTR

    pot = PlummerPotential(amp=1.0, b=0.7)
    v0 = pot(_R0, _Z0)
    assert not backend.is_backend_array(v0)
    assert PARAM_CACHE_ATTR in pot.__dict__
    assert pot(_R0, _Z0) == v0  # cached negative: still numpy
    amp = torch.tensor(1.0, requires_grad=True)
    pot._amp = amp
    out = pot(_R0, _Z0)
    assert backend.is_backend_array(out)
    out.backward()
    numpy.testing.assert_allclose(float(amp.grad), v0, rtol=1e-15)
    # a new attribute holding the parameter
    pot = PlummerPotential(amp=1.0, b=0.7)
    pot(_R0, _Z0)
    pot._extra = torch.tensor(1.0, requires_grad=True)
    assert backend.is_backend_array(pot(_R0, _Z0))


def test_parameter_cache_is_not_part_of_the_jit_key():
    # the cache is derived from the attributes the jit static key already holds:
    # setting it must not change the key (that would retrace once per object)
    from galpy.backend._jit import _static_key

    pot = PlummerPotential(amp=1.0, b=0.7)
    k0 = _static_key(pot)
    pot(_R0, _Z0)
    assert _static_key(pot) == k0


def test_parameter_cache_survives_pickling():
    import pickle

    pot = PlummerPotential(amp=1.0, b=0.7)
    v0 = pot(_R0, _Z0)
    pot2 = pickle.loads(pickle.dumps(pot))
    assert pot2(_R0, _Z0) == v0
