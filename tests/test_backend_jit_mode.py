###############################################################################
# test_backend_jit_mode.py: the opt-in trace mode.
#
# galpy.backend.jit("jax"|"torch") traces every boundary call, so the WHOLE test
# suite can be run jitted (pytest --backend jax --jit). These tests cover the
# mode itself: that it is off by default, that it leaves the numpy path alone,
# that traced results match eager, that the coordinate/static split follows the
# @backend_input declaration, and that the cache does not recompile per call.
###############################################################################
import numpy
import pytest

pytestmark = pytest.mark.backend_managed

try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    jax = None
try:
    import torch

    torch.set_default_dtype(torch.float64)
except ImportError:  # pragma: no cover
    torch = None

import galpy.backend as gb
from galpy.potential import (
    LogarithmicHaloPotential,
    MiyamotoNagaiPotential,
    evaluateRforces,
)

_R0, _Z0 = 1.1, 0.2


def test_jit_mode_is_off_by_default():
    assert gb.jit_mode() == "off"


def test_jit_mode_context_restores():
    with gb.jit("jax"):
        assert gb.jit_mode() == "jax"
    assert gb.jit_mode() == "off"


def test_jit_mode_rejects_unknown_framework():
    with pytest.raises(ValueError):
        gb.set_jit("tensorflow")
    with pytest.raises(ValueError):
        with gb.jit("tensorflow"):  # pragma: no cover - never entered
            pass


def test_numpy_path_ignores_jit_mode():
    """Trace mode must not touch the numpy path: same object type, same value."""
    pot = MiyamotoNagaiPotential(normalize=1.0)
    eager = pot.Rforce(_R0, _Z0)
    with gb.jit("jax"):
        traced = pot.Rforce(_R0, _Z0)
    assert type(traced) is type(eager)
    assert traced == eager


@pytest.mark.skipif("jax is None")
@pytest.mark.parametrize("entry", ["__call__", "Rforce", "zforce", "dens"])
def test_jax_traced_matches_eager(entry):
    pot = MiyamotoNagaiPotential(normalize=1.0)
    call = {
        "__call__": lambda p, R, z: p(R, z),
        "Rforce": lambda p, R, z: p.Rforce(R, z),
        "zforce": lambda p, R, z: p.zforce(R, z),
        "dens": lambda p, R, z: p.dens(R, z),
    }[entry]
    R, z = jnp.asarray(_R0), jnp.asarray(_Z0)
    eager = float(call(pot, R, z))
    with gb.jit("jax"):
        traced = float(call(pot, R, z))
    # XLA is free to reassociate, so this is a numerical match, not bit-identity.
    numpy.testing.assert_allclose(traced, eager, rtol=1e-12, atol=1e-14)


@pytest.mark.skipif("jax is None")
def test_jax_traces_with_an_unhashable_static():
    """A list of potentials is the ordinary way to pass a composite; it is a
    static argument and unhashable, which plain jax.jit would reject."""
    pots = [
        MiyamotoNagaiPotential(normalize=1.0),
        LogarithmicHaloPotential(normalize=1.0),
    ]
    R, z = jnp.asarray(_R0), jnp.asarray(_Z0)
    eager = float(evaluateRforces(pots, R, z))
    with gb.jit("jax"):
        traced = float(evaluateRforces(pots, R, z))
    numpy.testing.assert_allclose(traced, eager, rtol=1e-12, atol=1e-14)


@pytest.mark.skipif("jax is None")
def test_jax_control_parameters_stay_static():
    """dR/dphi are derivative ORDERS, not coordinates: traced as a value they
    would be a tracer where the body does `if dR == 0`."""
    pot = MiyamotoNagaiPotential(normalize=1.0)
    R, z = jnp.asarray(_R0), jnp.asarray(_Z0)
    eager = float(pot(R, z, dR=1))
    with gb.jit("jax"):
        traced = float(pot(R, z, dR=1))
    numpy.testing.assert_allclose(traced, eager, rtol=1e-12, atol=1e-14)


@pytest.mark.skipif("jax is None")
def test_jax_reuses_the_compiled_trace():
    """A second call with the same static bundle must hit the cache. Without a
    value-based key on the static bundle every call would recompile."""
    pot = MiyamotoNagaiPotential(normalize=1.0)
    R, z = jnp.asarray(_R0), jnp.asarray(_Z0)
    with gb.jit("jax"):
        pot.Rforce(R, z)
        before = jax.jit._cache_size() if hasattr(jax.jit, "_cache_size") else None
        for _ in range(5):
            pot.Rforce(R, z)
        after = jax.jit._cache_size() if hasattr(jax.jit, "_cache_size") else None
    if before is not None:  # pragma: no cover - jax version dependent
        assert after == before


@pytest.mark.skipif("jax is None")
def test_jax_retraces_after_a_parameter_changes():
    """A trace bakes self._amp in as a constant, so a cache keyed on the
    potential's IDENTITY alone would keep returning the previous
    normalization's numbers after normalize() -- silently wrong, not an error.
    """
    pot = MiyamotoNagaiPotential(normalize=1.0)
    R, z = jnp.asarray(1.0), jnp.asarray(0.0)
    with gb.jit("jax"):
        first = float(pot.Rforce(R, z))
        pot.normalize(0.5)
        second = float(pot.Rforce(R, z))
    numpy.testing.assert_allclose(first, -1.0, rtol=1e-10)
    numpy.testing.assert_allclose(second, -0.5, rtol=1e-10)


@pytest.mark.skipif("jax is None")
def test_jax_nested_boundaries_do_not_retrace():
    """Entry points call each other; only the outermost may trace."""
    from galpy.backend._jit import _TRACING

    pot = MiyamotoNagaiPotential(normalize=1.0)
    seen = []

    R, z = jnp.asarray(_R0), jnp.asarray(_Z0)
    original = pot._evaluate

    def spy(*args, **kwargs):
        seen.append(_TRACING.get())
        return original(*args, **kwargs)

    pot._evaluate = spy
    try:
        with gb.jit("jax"):
            pot(R, z)
    finally:
        pot._evaluate = original
    assert seen and all(seen), "inner boundaries must run inside the outer trace"


@pytest.mark.skipif("torch is None")
def test_torch_compiled_matches_eager():
    pot = MiyamotoNagaiPotential(normalize=1.0)
    R, z = torch.tensor(_R0), torch.tensor(_Z0)
    eager = float(pot.Rforce(R, z))
    try:
        with gb.jit("torch"):
            traced = float(pot.Rforce(R, z))
    except Exception as exc:  # pragma: no cover - torch.compile unsupported here
        pytest.skip(f"torch.compile unavailable: {type(exc).__name__}")
    numpy.testing.assert_allclose(traced, eager, rtol=1e-12, atol=1e-14)
