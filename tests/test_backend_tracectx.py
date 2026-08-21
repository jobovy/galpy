###############################################################################
# test_backend_tracectx.py: coverage for galpy.backend._tracectx.
#
# galpy resolves its backend, and its jit mode, through context variables. A
# plain ``ContextVar.get()`` is opaque to torch dynamo: it severs the graph at
# every read, and under ``fullgraph=True`` it refuses to compile at all. Since
# get_namespace() reads one on every dispatch, that single call made the whole
# of galpy uncompilable -- not because the potentials were untraceable, but
# because of the lookup in front of them.
#
# ``TracedContextVar`` keeps the ContextVar (that isolation is the point) and
# adds a thread-local mirror consulted ONLY while dynamo is tracing, where the
# value is fixed by construction.
#
# The backend-tests CI job uploads no coverage, so the compiling branch is
# exercised here explicitly. Backends that are not installed self-skip.
###############################################################################
import contextvars
import sys
import threading
from unittest import mock

import pytest
from backend_jit_helpers import no_torch_compile_deprecations

from galpy.backend import _tracectx
from galpy.backend._tracectx import TracedContextVar, _is_compiling

pytestmark = pytest.mark.backend_managed

try:
    import torch

    torch.set_default_dtype(torch.float64)
except ImportError:  # pragma: no cover
    torch = None


@pytest.mark.parametrize("default", [None, "off", 0, False])
def test_matches_a_contextvar_through_nested_set_and_reset(default):
    # The eager contract is "indistinguishable from contextvars.ContextVar",
    # so assert that against a real one step for step rather than spot-checking
    # a single set/get.
    ref = contextvars.ContextVar(f"ref_{default}", default=default)
    got = TracedContextVar(f"got_{default}", default=default)
    assert got.get() == ref.get() == default
    ref_a, got_a = ref.set("a"), got.set("a")
    assert got.get() == ref.get() == "a"
    ref_b, got_b = ref.set("b"), got.set("b")
    assert got.get() == ref.get() == "b"
    ref.reset(ref_b), got.reset(got_b)
    assert got.get() == ref.get() == "a"
    ref.reset(ref_a), got.reset(got_a)
    assert got.get() == ref.get() == default


def test_get_reads_the_mirror_only_while_compiling():
    # Drive the two sources of truth APART -- set the underlying ContextVar
    # without going through .set(), so the mirror keeps the older value -- and
    # check each branch returns its own. Setting both to the same value would
    # pass no matter which branch ran.
    var = TracedContextVar("which", default="default")
    var.set("through-set")
    var._var.set("contextvar-only")
    assert var.get() == "contextvar-only"
    with mock.patch.object(_tracectx, "_is_compiling", return_value=True):
        assert var.get() == "through-set"


def test_threads_do_not_share_the_mirror():
    # A module global would make two threads forcing different backends observe
    # each other; that isolation is the whole reason these are ContextVars, so
    # the mirror has to be thread-local. Assert BOTH directions.
    var = TracedContextVar("iso", default="unset")
    var.set("main")
    seen = {}

    def worker():
        with mock.patch.object(_tracectx, "_is_compiling", return_value=True):
            seen["before"] = var.get()
        var.set("worker")
        with mock.patch.object(_tracectx, "_is_compiling", return_value=True):
            seen["after"] = var.get()

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert seen["before"] == "unset"  # the main thread's value is not visible
    assert seen["after"] == "worker"
    with mock.patch.object(_tracectx, "_is_compiling", return_value=True):
        assert var.get() == "main"  # and the worker's did not leak back


def test_is_compiling_is_false_when_torch_is_not_imported():
    # The guard reads sys.modules rather than importing torch, so a numpy-only
    # run never pays the import. That guard IS the contract; the coverage shard
    # always has torch imported, so it is only reachable by hiding it.
    with mock.patch.dict(sys.modules, {"torch": None}):
        assert _is_compiling() is False


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_is_compiling_is_false_outside_a_trace():
    assert _is_compiling() is False


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_forced_backend_survives_a_fullgraph_compile():
    # The regression test. Before TracedContextVar this raised
    #   Unsupported: Dynamo does not know how to trace method `get` of ContextVar
    # for every potential and every method. fullgraph=True errors on ANY graph
    # break, so reaching the assert at all is the no-break assertion; the
    # comparison then pins that tracing did not change the value.
    from galpy.backend import use
    from galpy.potential import MiyamotoNagaiPotential

    mp = MiyamotoNagaiPotential(normalize=1.0, a=0.5, b=0.05)
    R = torch.tensor([1.0, 1.2, 0.8], dtype=torch.float64)
    z = torch.tensor([0.1, 0.05, 0.2], dtype=torch.float64)
    with use("torch", force=True):
        eager = mp.Rforce(R, z)
        torch._dynamo.reset()
        with no_torch_compile_deprecations():
            compiled = torch.compile(
                lambda a, b: mp.Rforce(a, b), fullgraph=True, dynamic=False
            )(R, z)
    assert torch.equal(compiled, eager)


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_jit_mode_contextvars_are_readable_inside_a_trace():
    # _JIT_CTX/_TRACING are read by traced_call, i.e. inside the region
    # torch.compile is walking -- the second break site, on the same mechanism.
    from galpy.backend._jit import _JIT_CTX, _TRACING

    seen = {}

    def probe(x):
        seen["mode"] = _JIT_CTX.get()
        seen["tracing"] = _TRACING.get()
        return x * 2.0

    token = _JIT_CTX.set("torch")
    try:
        torch._dynamo.reset()
        with no_torch_compile_deprecations():
            torch.compile(probe, fullgraph=True, dynamic=False)(
                torch.tensor(1.5, dtype=torch.float64)
            )
    finally:
        _JIT_CTX.reset(token)
    assert seen == {"mode": "torch", "tracing": False}


def test_a_set_while_tracing_touches_the_mirror_only():
    # `ContextVar.set` is opaque to dynamo exactly as `.get` is, and galpy sets
    # one INSIDE the compiled region (_jit.py's tracing flag), so `.set` has to
    # be traceable too. It stays off the ContextVar while tracing: a trace is a
    # compile-time artifact and its writes must not outlive it. Assert the
    # ContextVar is genuinely untouched, not merely that the call succeeded.
    var = TracedContextVar("scoped", default="default")
    var.set("eager")
    with mock.patch.object(_tracectx, "_is_compiling", return_value=True):
        token = var.set("traced")
        assert var.get() == "traced"  # reads during the trace see it
        assert var._var.get() == "eager"  # the ContextVar does not
        var.reset(token)
        assert var.get() == "eager"
    assert var.get() == "eager"
    assert var._var.get() == "eager"


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_set_and_reset_survive_a_fullgraph_compile():
    # The regression test for the `.set` half. `_jit.py`'s compiled `call` does
    # `_TRACING.set(True) ... _TRACING.reset(token)` around the method, so a
    # non-traceable `.set` would break the graph inside galpy's own jit path.
    var = TracedContextVar("in_trace", default=False)

    def f(x):
        token = var.set(True)
        try:
            return x * 2.0 if var.get() else x
        finally:
            var.reset(token)

    torch._dynamo.reset()
    with no_torch_compile_deprecations():
        got = torch.compile(f, fullgraph=True, dynamic=False)(
            torch.tensor(1.5, dtype=torch.float64)
        )
    assert float(got) == 3.0  # the True branch ran, i.e. .get() saw the .set()
    assert var.get() is False  # and the trace left no residue behind
