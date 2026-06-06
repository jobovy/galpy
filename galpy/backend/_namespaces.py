###############################################################################
#   galpy.backend._namespaces: helpers mapping backend names to array
#   namespaces and small namespace-agnostic utilities.
###############################################################################
import numpy

from ..util._optional_deps import (
    _ARRAY_API_COMPAT_LOADED,
    _JAX_LOADED,
    _TORCH_LOADED,
)

# Canonical backend names accepted throughout galpy.backend
_NUMPY_NAMES = frozenset(("numpy", "np"))
_JAX_NAMES = frozenset(("jax", "jnp", "jax.numpy"))
_TORCH_NAMES = frozenset(("torch", "pytorch"))


def _is_python_scalar(x):
    """True for plain Python scalars (and None), which carry no backend info."""
    return x is None or isinstance(x, (bool, int, float, complex))


def is_backend_array(x):
    """True if ``x`` is a non-numpy backend array (a jax or torch array/tensor).

    Plain Python scalars, ``None``, numpy arrays/scalars, astropy Quantities, and
    anything the backend layer does not recognise return ``False`` -- so the
    numpy/Quantity code paths stay byte-identical and only genuine backend arrays
    (including traced ones, so autodiff w.r.t. parameters works) take any
    pass-through branch keyed on this. Detection is by direct ``isinstance``
    against the public ``jax.Array`` / ``torch.Tensor`` base classes, gated on the
    optional-dependency flags so a numpy-only install never imports jax/torch.
    """
    if _is_python_scalar(x) or isinstance(x, (numpy.ndarray, numpy.generic)):
        return False
    if _JAX_LOADED:
        import jax

        if isinstance(x, jax.Array):
            return True
    if _TORCH_LOADED:
        import torch

        if isinstance(x, torch.Tensor):
            return True
    return False


def namespace_for_name(name):
    """Map a backend name ('numpy'|'jax'|'torch') to its array namespace module.

    numpy resolves to the *plain* numpy module (so the numpy code path is
    byte-identical to today); jax/torch resolve to their array-API namespaces.
    """
    if not isinstance(name, str):
        # Already a namespace module; pass through.
        return name
    lname = name.lower()
    if lname in _NUMPY_NAMES:
        return numpy
    if lname in _JAX_NAMES:
        if not _JAX_LOADED:  # pragma: no cover - defensive: needs jax absent
            raise ImportError("galpy backend 'jax' requested but jax is not installed")
        import jax.numpy as jnp

        return jnp
    if lname in _TORCH_NAMES:
        if not _TORCH_LOADED:  # pragma: no cover - defensive: needs torch absent
            raise ImportError(
                "galpy backend 'torch' requested but torch is not installed"
            )
        import array_api_compat.torch as txp

        return txp
    raise ValueError(f"unknown galpy backend '{name}'")


def namespace_from_arrays(arrays):
    """Infer the array namespace from the (non-scalar) array arguments.

    Returns the plain numpy module when every array-like argument is a numpy
    array (byte-identical numpy path), the appropriate jax/torch namespace when
    a tracked array is present, or None when there is nothing array-like to
    dispatch on (so the caller can fall through to the context/global default).
    """
    arrs = [a for a in arrays if not _is_python_scalar(a)]
    if not arrs:
        return None
    if all(isinstance(a, (numpy.ndarray, numpy.generic)) for a in arrs):
        return numpy
    if not _ARRAY_API_COMPAT_LOADED:  # pragma: no cover - backend extra installs it
        raise ImportError(
            "galpy's non-numpy backends require array-api-compat "
            "(pip install array-api-compat, or galpy[jax]/galpy[torch])"
        )
    import array_api_compat

    # Non-numpy arrays only reach here (numpy is handled by the fast path above),
    # so this returns the jax / array-api-compat-torch namespace.
    return array_api_compat.array_namespace(*arrs)
