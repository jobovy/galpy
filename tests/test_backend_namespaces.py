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

from galpy.backend import as_numpy

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
