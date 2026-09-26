###############################################################################
# test_backend_namespaces.py: focused coverage for galpy.backend.as_numpy, the
# GPU-safe backend->numpy converter that the test suite shares for value
# assertions (it replaced ~18 duplicated per-file _tonumpy/_np/_to_numpy
# helpers). The torch branch (.detach().cpu().numpy()) is production code, so it
# is exercised here explicitly since the backend-tests CI job uploads no
# coverage. The numpy path is unaffected (as_numpy is the identity on numpy
# arrays and python scalars).
###############################################################################
import numpy
import pytest

from galpy.backend import as_numpy, set_at

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

    torch.set_default_dtype(torch.float64)

    BACKENDS.append("torch")
except ImportError:  # pragma: no cover
    torch = None

_SRC = [1.0, 2.5, -3.0, 4.25]


def test_as_numpy_passthrough_numpy_and_scalars():
    # A numpy input is returned unchanged (identity, not a copy), so the numpy
    # path stays byte-identical; python scalars pass through unchanged too.
    a = numpy.asarray(_SRC)
    assert as_numpy(a) is a
    assert as_numpy(3.5) == 3.5
    assert as_numpy(7) == 7


@pytest.mark.parametrize("backend", BACKENDS)
def test_as_numpy_roundtrip(backend):
    src = numpy.asarray(_SRC)
    if backend == "jax":
        x = jnp.asarray(_SRC)
    else:  # torch: a grad-tracking tensor exercises the .detach() branch
        x = torch.tensor(_SRC, requires_grad=True)
    out = as_numpy(x)
    assert isinstance(out, numpy.ndarray)
    numpy.testing.assert_allclose(out, src)


# ---------------------------------------------------------------------------
# set_at: the backend-agnostic scatter. jax arrays are immutable and need
# .at[].set(); torch tensors are mutable but assigning into one that carries a
# graph raises, so both go out of place. Production code (the AdiabaticGrid
# off-grid fallback), and the backend-tests CI job uploads no coverage, so it
# is exercised explicitly here.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", BACKENDS)
def test_set_at_replaces_masked_entries_out_of_place(backend):
    src = numpy.asarray([1.0, 2.0, 3.0, 4.0])
    mask_np = numpy.asarray([False, True, False, True])
    if backend == "jax":
        xp, arr, mask = jnp, jnp.asarray(src), jnp.asarray(mask_np)
        vals = jnp.asarray([20.0, 40.0])
    else:
        xp, arr, mask = torch, torch.tensor(src), torch.tensor(mask_np)
        vals = torch.tensor([20.0, 40.0])
    out = set_at(xp, arr, mask, vals)
    numpy.testing.assert_allclose(as_numpy(out), [1.0, 20.0, 3.0, 40.0])
    # out of place: the input is untouched, which is what lets callers keep the
    # original around (and is REQUIRED for jax, whose arrays are immutable).
    numpy.testing.assert_allclose(as_numpy(arr), src)


@pytest.mark.parametrize("backend", BACKENDS)
def test_set_at_leaves_a_grad_tracking_input_intact(backend):
    # torch raises on in-place assignment into a graph-carrying tensor, so the
    # clone is load-bearing rather than defensive. jax is checked for symmetry.
    if backend == "jax":
        arr = jnp.asarray([1.0, 2.0, 3.0])
        out = set_at(jnp, arr, jnp.asarray([False, True, False]), jnp.asarray([9.0]))
    else:
        arr = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        out = set_at(
            torch, arr, torch.tensor([False, True, False]), torch.tensor([9.0])
        )
        assert arr.requires_grad
    numpy.testing.assert_allclose(as_numpy(out), [1.0, 9.0, 3.0])
    numpy.testing.assert_allclose(as_numpy(arr), [1.0, 2.0, 3.0])


def _mk(backend, x):
    return jnp.asarray(x) if backend == "jax" else torch.tensor(x)


def test_namespace_from_arrays_follows_the_data_not_a_forced_context():
    # get_namespace resolves a FORCED default ahead of the data -- "forced
    # default beats the data", in its own source. That is right for "which
    # namespace should this computation use" and WRONG for "which namespace
    # does this VALUE live in", which is what lifting another operand onto it
    # needs. Done with get_namespace inside use("numpy", force=True) the lift
    # yields an ndarray, which then raises against a grad tensor and otherwise
    # silently loses the namespace/device.
    import galpy.backend as gb
    from galpy.backend import get_namespace
    from galpy.backend._namespaces import namespace_from_arrays

    for backend in BACKENDS:
        x = _mk(backend, [1.0, 2.0])
        assert namespace_from_arrays((x,)) is get_namespace(x)  # agree, no context
        with gb.use("numpy", force=True):
            assert get_namespace(x) is numpy, "get_namespace should follow force"
            assert namespace_from_arrays((x,)) is not numpy, (
                "the data-side resolver must follow the DATA, not the forced default"
            )
    # nothing array-like -> None, so callers keep their own fallback
    assert namespace_from_arrays((1.0,)) is None


@pytest.mark.parametrize("backend", BACKENDS)
def test_lift_helpers_resolve_the_data_under_a_forced_numpy_context(backend):
    # The consequence at a real call site: constantbetadf's construction runs
    # inside use("numpy", force=True), and these leaves lift the numpy grid
    # onto the (backend) scale. Resolved through get_namespace the lift
    # produced an ndarray and the multiply that follows met a tensor.
    import galpy.backend as gb
    from galpy.backend import is_backend_array
    from galpy.potential.SCFPotential import _RToxi, _xiToR

    a = _mk(backend, 1.3)
    with gb.use("numpy", force=True):
        r_out = _xiToR(numpy.array([-0.5, 0.0, 0.5]), a=a)
        xi_out = _RToxi(numpy.array([0.5, 1.0, 2.0]), a=a)
    assert is_backend_array(r_out), "_xiToR lifted onto the forced namespace"
    assert is_backend_array(xi_out), "_RToxi lifted onto the forced namespace"


def test_as_numpy_constant_reads_the_primal_but_not_under_jit():
    # For a value used only as a numerical constant (a calibration, bracket or
    # integration limit). as_numpy itself refuses a jax tracer -- a guard kept on
    # purpose -- while the primal IS concrete under eager jax.grad.
    from galpy.backend import as_numpy_constant

    assert as_numpy_constant(1.5) == 1.5  # numpy / scalars pass through
    if "jax" in BACKENDS:
        seen = []

        def f(x):
            seen.append(as_numpy_constant(x * 2.0))
            return x * 3.0

        assert float(jax.grad(f)(1.5)) == 3.0  # gradient unaffected
        assert isinstance(seen[0], numpy.ndarray) and seen[0] == 3.0
        with pytest.raises(Exception):  # no value exists under jit
            jax.jit(lambda x: as_numpy_constant(x) + 0.0)(1.5)
    if "torch" in BACKENDS:
        t = torch.tensor(1.5, requires_grad=True)
        c = as_numpy_constant(t * 2.0)
        assert isinstance(c, numpy.ndarray) and c == 3.0


def test_compilable_singledispatchmethod():
    # dispatches exactly like functools.singledispatchmethod (Orbit.integrate)
    from galpy.backend._namespaces import compilable_singledispatchmethod

    class Base:
        pass

    class Sub(Base):
        pass

    class A:
        @compilable_singledispatchmethod
        def f(self, t, k=1.0):
            """default"""
            return ("default", t, k)

        @f.register(Base)
        @f.register(list)
        def _(self, t, k=1.0):
            return ("registered", type(t).__name__, k)

    a = A()
    assert a.f(1.5, k=2.0) == ("default", 1.5, 2.0)
    assert a.f([1.0], 3.0) == ("registered", "list", 3.0)
    assert a.f(Sub()) == ("registered", "Sub", 1.0)  # resolved through the MRO
    assert A.f.__doc__ == "default"
    with pytest.raises(TypeError, match="requires at least 1 positional argument"):
        a.f()


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_compilable_singledispatchmethod_under_torch_compile():
    # a graph break inside the dispatched method: torch 2.14 recursed forever
    # through functools.singledispatchmethod's own dispatch
    from galpy.backend._namespaces import compilable_singledispatchmethod

    class A:
        @compilable_singledispatchmethod
        def f(self, t, k=1.0):
            torch._dynamo.graph_break()
            return t * k

    a = A()
    torch._dynamo.reset()
    out = torch.compile(lambda x: a.f(x, 2.0) + 1.0, backend="eager")(torch.ones(2))
    assert out.tolist() == [3.0, 3.0]
