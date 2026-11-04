###############################################################################
# test_backend_batched_adaptive_grad.py: the derivative of a BATCH of orbits
# through the adaptive in-backend (diffrax) solve is the same whether the batch
# is vmapped over single-orbit solves (independent step-size controllers), is
# integrated natively on one shared time grid (one controller for the batch), or
# is a plain loop of single-orbit solves; and it matches constant steps and a
# converging finite difference. An earlier version warned here and recommended
# constant steps, after a defect that turned out to be in the direct adjoint's
# checkpointed loop (fixed in diffrax_adjoint.py), not in the batching.
###############################################################################
import numpy
import pytest

from galpy.orbit import Orbit
from galpy.potential import PlummerPotential

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


def _final(ic, b, **kw):
    o = Orbit(ic)
    o.integrate(_TS, PlummerPotential(amp=1.0, b=b), method="diffrax", **kw)
    return jnp.sum(o.R(_TS[-1]) ** 2)


def _loop(b, **kw):  # the reference: one solve per orbit, no batching
    return sum(_final(_ICS[i], b, **kw) for i in range(_ICS.shape[0]))


def _vmapped(b, **kw):  # independent solves, each with its own controller
    return jnp.sum(jax.vmap(lambda ic: _final(ic, b, **kw))(_ICS))


def _native(b, **kw):  # one solve, one controller for the whole batch
    return _final(_ICS, b, **kw)


def test_batch_grad_through_adaptive_steps_matches_loop():
    # reverse mode w.r.t. a potential parameter: vmapped and native batches
    # equal the loop of single solves to roundoff
    g_loop = float(jax.grad(_loop)(0.6))
    g_vmap = float(jax.grad(_vmapped)(0.6))
    g_native = float(jax.grad(_native)(0.6))
    assert abs(g_vmap - g_loop) < 1e-8 * abs(g_loop)
    assert abs(g_native - g_loop) < 1e-8 * abs(g_loop)
    # and the same with constant steps (a different step sequence)
    g_const = float(
        jax.grad(lambda b: _vmapped(b, inbackend_kwargs={"nsteps": 2000}))(0.6)
    )
    assert abs(g_const - g_loop) < 1e-7 * abs(g_loop)


def test_batch_grad_through_adaptive_steps_matches_finite_difference():
    # central differences converge onto the vmapped derivative as h^2
    g = float(jax.grad(_vmapped)(0.6))
    errs = []
    for h in (1e-2, 1e-3):
        fd = (float(_vmapped(0.6 + h)) - float(_vmapped(0.6 - h))) / (2.0 * h)
        errs.append(abs(fd - g) / abs(g))
    assert errs[1] < 1e-4
    assert errs[1] < errs[0] / 30.0  # h^2: a factor 100 per decade, with margin


def test_batch_jacfwd_through_adaptive_steps_matches_loop():
    # forward mode (the direct adjoint) w.r.t. the initial conditions, vmapped
    def end_point(ic, **kw):
        o = Orbit(ic)
        o.integrate(_TS, PlummerPotential(amp=1.0, b=0.6), method="diffrax", **kw)
        return jnp.stack([o.R(_TS[-1]), o.z(_TS[-1])])

    J_vmap = numpy.asarray(jax.vmap(jax.jacfwd(end_point))(_ICS))
    J_loop = numpy.stack(
        [numpy.asarray(jax.jacfwd(end_point)(_ICS[i])) for i in range(3)]
    )
    assert numpy.max(numpy.fabs(J_vmap - J_loop)) < 1e-8 * numpy.max(numpy.fabs(J_loop))
