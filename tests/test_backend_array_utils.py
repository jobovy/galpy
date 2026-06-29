###############################################################################
# test_backend_array_utils.py: multi-backend tests for the small backend-agnostic
# array helpers in galpy.backend (atleast_1d, median) and the numpy_island
# decorator. These exercise the backend-only branches (the array-api-compat
# torch median workaround, numpy_island's backend-array passthrough) that the
# numpy-only matrix can never hit. Backends that are not installed self-skip.
###############################################################################
import numpy
import pytest

from galpy.backend import (
    atleast_1d,
    get_namespace,
    is_backend_array,
    median,
    numpy_island,
    use,
)

# This module manages backends explicitly; exempt from the global --backend
# force fixture.
pytestmark = pytest.mark.backend_managed

try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    _HAS_JAX = True
except ImportError:  # pragma: no cover
    jax = None
    _HAS_JAX = False
try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    _HAS_TORCH = False


def _backends():
    out = ["numpy"]
    if _HAS_JAX:
        out.append("jax")
    if _HAS_TORCH:
        out.append("torch")
    return out


def _asarray(backend_name, x):
    # Build an array in the requested backend's namespace.
    if backend_name == "jax":
        return jnp.asarray(x, dtype=float)
    if backend_name == "torch":
        return torch.as_tensor(numpy.asarray(x, dtype=float))
    return numpy.asarray(x, dtype=float)


@pytest.mark.parametrize("backend_name", _backends())
def test_atleast_1d(backend_name):
    with use(backend_name, force=True):
        # scalar -> 1d, in the active backend's namespace
        out = atleast_1d(1.3)
        assert out.shape == (1,)
        assert float(numpy.asarray(out)[0]) == pytest.approx(1.3)
        # an existing backend array is returned in its own namespace untouched
        a = _asarray(backend_name, [1.0, 2.0, 3.0])
        out2 = atleast_1d(a)
        if backend_name != "numpy":
            assert is_backend_array(out2)
        numpy.testing.assert_allclose(numpy.asarray(out2), [1.0, 2.0, 3.0])


@pytest.mark.parametrize("backend_name", _backends())
def test_median_matches_numpy(backend_name):
    xp = get_namespace(_asarray(backend_name, [0.0]))
    # 1d, NaN-skipping -> numpy.nanmedian (even count: mean of the two central)
    a = _asarray(backend_name, [1.0, float("nan"), 3.0, 2.0])
    numpy.testing.assert_allclose(
        float(numpy.asarray(median(xp, a, skipnan=True))),
        numpy.nanmedian([1.0, 3.0, 2.0]),
    )
    # along axis=1 -> numpy.median (the torch lower-median workaround matters for
    # even counts: [0,1,2,3] -> 1.5, not torch.median's 1.0)
    b = _asarray(backend_name, numpy.arange(12.0).reshape(3, 4))
    numpy.testing.assert_allclose(
        numpy.asarray(median(xp, b, axis=1)),
        numpy.median(numpy.arange(12.0).reshape(3, 4), axis=1),
    )
    # axis=None, no skipnan
    numpy.testing.assert_allclose(
        float(numpy.asarray(median(xp, a[~xp.isnan(a)]))), numpy.median([1.0, 3.0, 2.0])
    )


@pytest.mark.parametrize("backend_name", _backends())
def test_numpy_island_both_branches(backend_name):
    @numpy_island
    def f(x):
        # report the namespace the body actually ran under
        return get_namespace(x).__name__

    with use(backend_name, force=True):
        # numpy/scalar input -> forced numpy regardless of the active backend
        assert f(1.0) == "numpy"
        # a real backend-array input is left on the backend-native path
        a = _asarray(backend_name, [1.0, 2.0])
        ns = f(a)
        if backend_name == "numpy":
            assert ns == "numpy"
        else:
            assert backend_name in ns
