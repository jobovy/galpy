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


def _no_torch_compile_deprecations():
    """Suppress torch's own import-time DeprecationWarnings during a compile.

    ``torch.compile`` lazily imports ``torch._inductor``, whose mkldnn module
    warns on ``torch.jit.script_method`` at class-definition time. A per-test
    ``filterwarnings`` mark cannot suppress it (a module-level
    ``error::DeprecationWarning`` pytestmark is applied last and wins), so
    filter it here, around the call itself.
    """
    import contextlib
    import warnings

    @contextlib.contextmanager
    def _ctx():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            yield

    return _ctx()


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
    with _no_torch_compile_deprecations():
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
    with _no_torch_compile_deprecations():
        got = torch.compile(use_builder, fullgraph=False, dynamic=False)(
            torch.tensor(2.0, dtype=torch.float64)
        )
    assert float(got) == 20.0
