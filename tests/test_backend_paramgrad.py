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
_EPS = 1e-6


def _value(ctor, fixed, pname, theta, method, xp_R, xp_z):
    pot = ctor(**{**fixed, pname: theta})
    return METHODS[method](pot, xp_R, xp_z)


@pytest.mark.parametrize("method", METHOD_IDS)
@pytest.mark.parametrize("spec", PARAM_SPECS, ids=SPEC_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_param_grad_vs_finite_difference(backend_name, spec, method):
    ctor, fixed, pname, th0 = spec

    # finite-difference reference, computed on the pure-numpy path
    def fnp(theta):
        return float(_value(ctor, fixed, pname, theta, method, _R0, _Z0))

    fd = (fnp(th0 + _EPS) - fnp(th0 - _EPS)) / (2 * _EPS)

    if backend_name == "jax":
        R, z = jnp.asarray(_R0), jnp.asarray(_Z0)
        ad = float(
            jax.grad(lambda th: _value(ctor, fixed, pname, th, method, R, z))(
                jnp.asarray(th0)
            )
        )
    else:
        R, z = torch.as_tensor(_R0), torch.as_tensor(_Z0)
        th = torch.tensor(th0, requires_grad=True)
        _value(ctor, fixed, pname, th, method, R, z).backward()
        ad = float(th.grad)
    numpy.testing.assert_allclose(ad, fd, rtol=1e-5, atol=1e-8)


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
