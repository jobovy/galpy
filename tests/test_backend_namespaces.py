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
