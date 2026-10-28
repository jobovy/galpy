###############################################################################
# test_backend_batched_adaptive_warning.py: differentiating a BATCH of orbits
# through the adaptive in-backend (diffrax) solve warns, because the adaptive
# steps of the orbits depend on each other and the derivative can be wrong by
# several percent (the fix is a fixed number of steps); nothing else warns.
###############################################################################
import warnings

import numpy
import pytest

from galpy.orbit import Orbit
from galpy.potential import PlummerPotential
from galpy.util import galpyWarning

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

pytestmark = pytest.mark.backend_managed

_TS = jnp.linspace(0.0, 10.0, 11)
_ICS = jnp.asarray(
    [
        [1.0, 0.1, 1.1, 0.05, 0.03, 0.0],
        [0.8, -0.1, 0.9, -0.05, 0.05, 1.0],
        [1.2, 0.0, 1.0, 0.1, 0.0, 2.0],
    ]
)
_MATCH = "batch of orbits through an adaptive"


def _final(ic, b, **kw):
    o = Orbit(ic)
    o.integrate(_TS, PlummerPotential(amp=1.0, b=b), method="diffrax", **kw)
    return jnp.sum(o.R(_TS[-1]) ** 2)


def _no_warning(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("error", galpyWarning)
        return fn()


def test_batch_grad_through_adaptive_steps_warns():
    with pytest.warns(galpyWarning, match=_MATCH):
        jax.grad(lambda ic: _final(ic, 0.6))(_ICS)
    # w.r.t. a potential parameter too
    with pytest.warns(galpyWarning, match=_MATCH):
        jax.grad(lambda b: _final(_ICS, b))(jnp.asarray(0.6))


def test_batch_jvp_through_adaptive_steps_warns():
    with pytest.warns(galpyWarning, match=_MATCH):
        jax.jvp(
            lambda b: _final(_ICS, b, inbackend_kwargs={"adjoint": "direct"}),
            (jnp.asarray(0.6),),
            (jnp.asarray(1.0),),
        )


def test_vmapped_grad_through_adaptive_steps_warns():
    with pytest.warns(galpyWarning, match=_MATCH):
        jax.vmap(jax.grad(lambda ic: _final(ic, 0.6)))(_ICS)


def test_batch_grad_with_constant_steps_does_not_warn():
    g = _no_warning(
        lambda: jax.grad(lambda ic: _final(ic, 0.6, inbackend_kwargs={"nsteps": 1000}))(
            _ICS
        )
    )
    assert numpy.all(numpy.isfinite(numpy.asarray(g)))


def test_single_orbit_grad_does_not_warn():
    g = _no_warning(lambda: jax.grad(lambda ic: _final(ic, 0.6))(_ICS[0]))
    assert numpy.all(numpy.isfinite(numpy.asarray(g)))


def test_batch_value_and_jit_do_not_warn():
    v = _no_warning(lambda: _final(_ICS, 0.6))
    vj = _no_warning(lambda: jax.jit(lambda ic: _final(ic, 0.6))(_ICS))
    numpy.testing.assert_allclose(float(vj), float(v), rtol=1e-12)
    vm = _no_warning(lambda: jax.vmap(lambda ic: _final(ic, 0.6))(_ICS))
    assert vm.shape == (3,)
