###############################################################################
# test_backend_actionAngleHarmonicInverse.py: the frequency joins the backend
# (gap audit PR 28: a plain-float Omega was a numpy island under a forced
# backend, and not a tensor in torch data mode)
###############################################################################
import numpy
import pytest

from galpy import backend
from galpy.actionAngle import actionAngleHarmonicInverse
from galpy.backend import as_numpy, is_backend_array

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

    BACKENDS.append("torch")
except ImportError:  # pragma: no cover
    torch = None


@pytest.mark.parametrize("mode", ["data", "forced"])
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_frequency_on_the_backend(backend_name, mode):
    if backend_name == "torch":
        torch.set_default_dtype(torch.float64)
    aAI = actionAngleHarmonicInverse(omega=1.3)
    ref = aAI.xvFreqs(0.2, numpy.array([0.7, 1.2]))
    xp = jnp if backend_name == "jax" else torch
    if mode == "forced":
        with backend.use(backend_name, force=True):
            out = aAI.xvFreqs(0.2, numpy.array([0.7, 1.2]))
            O = aAI.Freqs(0.2)
    else:
        out = aAI.xvFreqs(xp.asarray(0.2), xp.asarray([0.7, 1.2]))
        O = aAI.Freqs(xp.asarray(0.2))
    for got, r in zip(out, ref):
        assert is_backend_array(got)
        numpy.testing.assert_array_equal(as_numpy(got), r)
    assert is_backend_array(O) and float(O) == 1.3


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_frequency_gradient_wrt_omega(backend_name):
    # O = omega: dO/domega = 1; x = sqrt(2J/omega) sin(a): the analytic derivative
    J, a, om = 0.2, 0.7, 1.3
    dx = -0.5 * numpy.sqrt(2.0 * J / om) / om * numpy.sin(a)
    if backend_name == "jax":
        f = lambda o: actionAngleHarmonicInverse(omega=o).xvFreqs(  # noqa: E731
            jnp.asarray(J), jnp.asarray(a)
        )
        gx = jax.jit(jax.grad(lambda o: f(o)[0]))(jnp.asarray(om))
        gO = jax.jit(jax.grad(lambda o: f(o)[2]))(jnp.asarray(om))
    else:
        o = torch.tensor(om, dtype=torch.float64, requires_grad=True)
        x, _, O = actionAngleHarmonicInverse(omega=o).xvFreqs(
            torch.tensor(J, dtype=torch.float64), torch.tensor(a, dtype=torch.float64)
        )
        (gx,) = torch.autograd.grad(x, o, retain_graph=True)
        (gO,) = torch.autograd.grad(O, o)
    numpy.testing.assert_allclose(float(gx), dx, rtol=1e-14)
    assert float(gO) == 1.0
