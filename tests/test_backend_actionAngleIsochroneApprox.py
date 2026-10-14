###############################################################################
# test_backend_actionAngleIsochroneApprox.py: estimateBIsochrone follows a forced
# backend and a differentiated potential parameter (gap audit PR 28: at plain
# coordinates it returned a float, and scipy's brentq read -- and detached --
# the objective, so d b/d(potential parameter) came back 0 under jax.jit)
###############################################################################
import pytest
from backend_param_grad import MODES, assert_param_grad

from galpy import backend
from galpy.actionAngle import estimateBIsochrone
from galpy.backend import is_backend_array
from galpy.potential import MiyamotoNagaiPotential

pytestmark = pytest.mark.backend_managed

BACKENDS = []
try:
    import jax

    jax.config.update("jax_enable_x64", True)
    BACKENDS.append("jax")
except ImportError:  # pragma: no cover
    jax = None
try:
    import torch

    BACKENDS.append("torch")
except ImportError:  # pragma: no cover
    torch = None


class _EstB:
    def __init__(self, a, b):
        self._pot = MiyamotoNagaiPotential(amp=1.0, a=a, b=b)

    def estb(self, R, z):
        return estimateBIsochrone(self._pot, R, z)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", ["a", "b"])
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_estimateBIsochrone_parameter_gradient(backend_name, mode, name):
    assert_param_grad(
        backend_name,
        mode,
        _EstB,
        {"a": 0.5, "b": 0.1},
        name,
        "estb",
        [1.0, 0.2],
        rtol=1e-8,
    )


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_estimateBIsochrone_forced_returns_backend(backend_name):
    pot = MiyamotoNagaiPotential(amp=1.0, a=0.5, b=0.1)
    ref = estimateBIsochrone(pot, 1.0, 0.2)
    with backend.use(backend_name, force=True):
        got = estimateBIsochrone(pot, 1.0, 0.2)
    assert is_backend_array(got)
    assert abs(float(got) / ref - 1.0) < 1e-10
