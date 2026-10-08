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

from galpy.backend import as_numpy, cummax, get_namespace, on_host, set_at, to_host

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


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_get_namespace_all_torch_fast_path():
    # all-torch inputs resolve to the array-api-compat torch namespace without
    # array_api_compat.array_namespace (untraceable: it hashes modules into a
    # set, a torch.compile graph break at every get_namespace call)
    import array_api_compat
    import array_api_compat.torch as txp

    from galpy.backend import get_namespace

    x = torch.ones(3)
    p = torch.nn.Parameter(torch.ones(2))
    assert get_namespace(x) is txp
    assert get_namespace(x, p, 2.0) is txp
    assert get_namespace(x) is array_api_compat.array_namespace(x)

    from galpy.potential import HernquistPotential

    hp = HernquistPotential(normalize=1.0)

    # two+ arrays (a potential's R, z) used to break the graph: one array alone
    # never did, so it cannot measure the fast path
    def f(y):
        return get_namespace(y, 2.0 * y).sin(y) + hp(y, 0.1 * y)

    torch._dynamo.reset()
    assert torch._dynamo.explain(f)(x).graph_break_count == 0
    # a non-torch array in the mix still goes through array_namespace
    with pytest.raises(TypeError):
        get_namespace(x, [1.0, 2.0])


# cummax: numpy.maximum.accumulate on every namespace (kingdf's traced solve)
@pytest.mark.parametrize("backend", ["numpy", *BACKENDS])
def test_cummax_matches_numpy(backend):
    src = numpy.asarray([1.0, 3.0, 2.0, 2.5, 5.0, 4.0])
    xp = {"numpy": numpy, "jax": jnp, "torch": torch}[backend]
    out = cummax(xp, xp.asarray(src))
    numpy.testing.assert_array_equal(as_numpy(out), numpy.maximum.accumulate(src))


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_to_host_maps_nested_sequences_and_keeps_autograd():
    t = torch.tensor([1.0, 2.0])
    out = to_host([t, (3.0, t), numpy.ones(2)])
    assert type(out) is list and type(out[1]) is tuple
    assert out[0].device.type == "cpu" and out[1][0] == 3.0
    assert isinstance(out[2], numpy.ndarray)
    numpy.testing.assert_array_equal(
        numpy.array(out[:1] + [out[1][1]]), [[1.0, 2.0], [1.0, 2.0]]
    )
    # a grad-tracking tensor stays grad-tracking, so numpy still refuses it
    g = torch.tensor(1.0, requires_grad=True)
    assert to_host([g])[0].requires_grad
    with pytest.raises(RuntimeError):
        numpy.array(to_host([g]))
    # non-sequence, non-tensor inputs pass through as the same object
    arr = numpy.ones(3)
    assert to_host(arr) is arr


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_on_host_hosts_the_result():
    f = on_host(lambda x, k=1.0: (torch.tensor([x, k]), x))
    t, x = f(2.0, k=3.0)
    assert t.device.type == "cpu" and x == 2.0
    numpy.testing.assert_array_equal(numpy.asarray(t), [2.0, 3.0])


@pytest.mark.parametrize("backend", BACKENDS)
def test_as_backend_constant_with_a_numpy_ref(backend):
    # a numpy ref's dtype is translated to the backend's (torch.asarray rejects
    # a numpy dtype)
    from galpy.backend import as_backend_constant
    from galpy.backend._namespaces import namespace_for_name

    xp = namespace_for_name(backend)
    out = as_backend_constant(xp, numpy.array([1.5, 2.0]), numpy.ones(2))
    assert get_namespace(out) is xp
    numpy.testing.assert_array_equal(as_numpy(out), [1.5, 2.0])
    assert str(out.dtype).endswith("float64")


def test_set_at_casts_values_to_the_destination_dtype(torch_default_float32):
    # numpy casts assigned values to the destination; torch raised on a float32
    # source (its default dtype) into a float64 tensor
    import torch

    arr = torch.zeros(4, dtype=torch.float64)
    out = set_at(torch, arr, arr == 0.0, torch.tensor([1.5, 2.5, 3.5, 4.5]))
    assert out.dtype == torch.float64
    numpy.testing.assert_array_equal(as_numpy(out), [1.5, 2.5, 3.5, 4.5])


def test_bucket_size():
    from galpy.backend import bucket_size

    assert [bucket_size(n) for n in (0, 1, 16, 17, 64, 65, 4096, 4097, 16000)] == [
        0,
        16,
        16,
        64,
        64,
        256,
        4096,
        16384,
        16384,
    ]
    assert bucket_size(3, minimum=4) == 4


def test_scalar_like_anchors_a_numpy_scalar_on_torch():
    from galpy.backend import scalar_like

    a = numpy.float64(1.25)
    assert scalar_like(numpy.ones(3), a) is a  # numpy: object-identical
    assert scalar_like(2.0, a) is a
    if jax is not None:  # jax takes numpy scalars natively: untouched
        assert scalar_like(jnp.ones(3), a) is a
    if torch is None:
        return
    t = torch.ones(3, dtype=torch.float64)
    out = scalar_like(t, a)
    assert isinstance(out, torch.Tensor) and out.ndim == 0 and float(out) == 1.25
    assert out.dtype == t.dtype and out.device == t.device
    arr = numpy.ones(3)
    assert scalar_like(t, arr) is arr  # only scalars
    assert scalar_like(t, 1.5) == 1.5
    # anchored on ref's dtype, as eager's weak numpy scalar: a 0-d float32
    # tensor stays float32 (like() would make a 0-d float64 tensor: upcast)
    t32 = torch.ones((), dtype=torch.float32)
    assert (scalar_like(t32, a) * t32).dtype == (a * t32).dtype == torch.float32


@pytest.mark.skipif("torch" not in BACKENDS, reason="torch not installed")
def test_torch_asarray_requires_grad_warning_silenced():
    # torch >= 2.12 warns when array-api-compat's asarray (galpy's torch xp.asarray)
    # gets a grad-tracking tensor; galpy wants exactly the new behaviour (keep the
    # graph), so that one message from that wrapper is filtered. A fresh process:
    # the warnings registry would hide a repeat in this one.
    import subprocess
    import sys

    code = (
        "import warnings, torch\n"
        "torch.set_default_dtype(torch.float64)\n"
        "from galpy.df import kingdf\n"
        "with warnings.catch_warnings(record=True) as rec:\n"
        "    W = torch.tensor(3.0, requires_grad=True)\n"
        "    kingdf(W0=W, M=2.3, rt=1.4).dens(torch.tensor(0.4)).backward()\n"
        "    import array_api_compat.torch as txp\n"
        "    txp.asarray(W)  # the wrapper itself, directly\n"
        "print(sum('torch.asarray' in str(r.message) for r in rec))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip().splitlines()[-1] == "0", out.stdout + out.stderr
