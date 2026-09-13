###############################################################################
# test_backend_trace.py: coverage for the two trace-awareness helpers in
# galpy.backend._namespaces that make galpy COMPATIBLE with torch.compile
# (galpy never compiles anything itself).
#
# ``under_trace``      -- "am I being traced rather than concretely evaluated?".
#   galpy has several dispatch points that pick an out-of-backend (scipy/numpy)
#   computation when they hold a concrete value. Under jax those probe by
#   ``float(x)``, which a tracer answers by RAISING. torch.compile does not:
#   dynamo turns ``float(x)`` into a symbolic scalar, so the probe wrongly takes
#   the concrete branch and drags the scipy code into the graph.
#
# ``untraceable_setup`` -- marks a lazy numpy/scipy SETUP builder so a tracer
#   runs it eagerly instead of tracing it. The table-backed potentials build
#   their constant backend tables lazily; when the FIRST call lands inside a
#   compiled region dynamo tries to trace scipy and dies.
#
# The backend-tests CI job uploads no coverage, so both branches are exercised
# here explicitly. Backends that are not installed self-skip.
###############################################################################
import numpy
import pytest
from backend_jit_helpers import no_torch_compile_deprecations

from galpy.backend._namespaces import under_trace, untraceable_setup

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


def test_under_trace_false_on_numpy():
    # Never true off a trace: plain numpy/python values are always concrete.
    assert not under_trace(numpy.float64(1.1), numpy.zeros(3), 2.0, None)


@pytest.mark.skipif(jax is None, reason="jax not installed")
def test_under_trace_jax():
    # Concrete jax arrays are NOT traced; a jit tracer is.
    assert not under_trace(jnp.asarray(1.1), jnp.zeros(3))
    seen = []

    def probe(x):
        seen.append(under_trace(x))
        return x * 2.0

    jax.jit(probe)(jnp.asarray(1.1))
    assert seen == [True]


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_under_trace_torch():
    # A concrete tensor -- even one carrying grad -- is not traced; only a
    # torch.compile region is. This is THE case float() cannot detect.
    assert not under_trace(torch.tensor(1.1, dtype=torch.float64))
    assert not under_trace(torch.tensor(1.1, dtype=torch.float64, requires_grad=True))
    seen = []

    def probe(x):
        seen.append(under_trace(x))
        return x * 2.0

    torch._dynamo.reset()
    with no_torch_compile_deprecations():
        torch.compile(probe, fullgraph=False, dynamic=False, backend="eager")(
            torch.tensor(1.1, dtype=torch.float64)
        )
    assert seen == [True]


def _scipy_only_builder():
    # Stands in for the real lazy table builders: pure scipy, untraceable by
    # dynamo (scipy.special.comb assigns into a numpy scalar under a trace).
    from scipy.special import comb

    return float(comb(5, 2))


@untraceable_setup
def _decorated_builder():
    return _scipy_only_builder()


def test_untraceable_setup_is_transparent_off_a_trace():
    # Plain call: same value, no behaviour change (and no torch import needed).
    assert _decorated_builder() == _scipy_only_builder() == 10.0


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_untraceable_setup_runs_scipy_setup_under_compile():
    # Without the decorator dynamo traces scipy.special.comb and raises
    # ("'numpy.float64' object does not support item assignment"); with it the
    # builder runs eagerly and its constant feeds back into the graph.
    def use_builder(x):
        return x * _decorated_builder()

    torch._dynamo.reset()
    with no_torch_compile_deprecations():
        got = torch.compile(use_builder, fullgraph=False, dynamic=False)(
            torch.tensor(2.0, dtype=torch.float64)
        )
    assert float(got) == 20.0


def test_helpers_take_the_fallback_when_torch_is_not_imported():
    # Both helpers guard on `"torch" not in sys.modules` so that a numpy-only
    # run never pays the torch import -- that guard IS the contract, not an
    # optimisation. The coverage shard always has torch imported, so these two
    # fallbacks are only reachable by hiding it.
    import sys
    from unittest import mock

    built = []

    @untraceable_setup
    def build(x):
        built.append(x)
        return 2 * x

    with mock.patch.dict(sys.modules):  # copies; restored on exit
        sys.modules.pop("torch", None)
        # under_trace: nothing is being traced, and torch is not consulted
        assert under_trace(numpy.array([1.0, 2.0])) is False
        # untraceable_setup: calls straight through, undecorated semantics
        assert build(21) == 42
        # the point of the guard: neither helper imported torch to answer
        assert "torch" not in sys.modules, "the numpy-only path must not import torch"
    assert built == [21], "the wrapped builder must run exactly once"
