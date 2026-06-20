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


def _is_floating_dtype(dtype):
    """True for real floating-point dtypes of any backend.

    numpy and jax expose numpy dtypes (checked via ``numpy.issubdtype``);
    torch dtypes expose an ``is_floating_point`` attribute (which for torch is
    False for complex dtypes, matching ``numpy.floating``).
    """
    is_fp = getattr(dtype, "is_floating_point", None)
    if is_fp is not None:  # torch.dtype
        return bool(is_fp)
    try:
        return numpy.issubdtype(dtype, numpy.floating)
    except TypeError:  # pragma: no cover - defensive: not a dtype-like
        return False


def match_input_dtype(out, *coords):
    """Cast ``out`` to the common (result) dtype of the coordinate inputs.

    Potentials whose interiors deliberately work in float64 (expansion-
    coefficient tables, Ogata quadrature nodes/weights, spline coefficients --
    SCF, DoubleExponentialDisk, interpSpherical, MultipoleExpansion) call this
    at compute-method exit so that float32 coordinates give a float32 result
    computed at float64 quality (the tables are *not* anchored to the input
    dtype). The function is a strict no-op -- returning the ``out`` object
    itself -- when no coordinate carries a floating dtype (plain Python
    scalars), when ``out`` has no real floating dtype, or when the dtypes
    already match; in particular the float64 numpy path returns its result
    object unchanged (bit-identical). Mixed floating input dtypes resolve via
    the namespace's ``result_type``. When a cast is needed it uses the
    namespace's ``astype`` (differentiable under jax/torch, so autodiff flows
    through it).
    """
    out_dtype = getattr(out, "dtype", None)
    if (out_dtype is None and not isinstance(out, float)) or (
        out_dtype is not None and not _is_floating_dtype(out_dtype)
    ):
        return out
    dtypes = [
        dtype
        for dtype in (getattr(coord, "dtype", None) for coord in coords)
        if dtype is not None and _is_floating_dtype(dtype)
    ]
    if not dtypes:
        return out
    if out_dtype is None:
        # Plain Python float output (float64 by construction; e.g. the scalar
        # _dens path of MultipoleExpansion): cast only when the coordinates
        # all carry a NARROWER floating dtype, so that float64 and plain-
        # scalar inputs keep the plain-float return type bit-identically
        target = dtypes[0] if all(d == dtypes[0] for d in dtypes) else None
        if target is not None and target != numpy.float64:
            return numpy.asarray(out, dtype=target)[()]
        return out
    if all(dtype == dtypes[0] for dtype in dtypes):
        target = dtypes[0]
    else:
        target = namespace_from_arrays((out,)).result_type(*dtypes)
    if target == out_dtype:
        return out
    if isinstance(out, (numpy.ndarray, numpy.generic)):
        # plain numpy: ndarray.astype works on every supported numpy version
        return out.astype(target)
    return namespace_from_arrays((out,)).astype(out, target)


def device_of(*coords):
    """Return the device of the first backend (jax/torch) array in ``coords``.

    The table-backed potentials anchor their stored numpy constant tables
    (expansion coefficients, quadrature nodes/weights, spline coefficients) on
    the device of the coordinate inputs through this helper: a plain
    ``xp.asarray(table)`` materializes on the CPU, and torch raises a
    mixed-device error when CUDA coordinates meet a CPU table. Plain Python
    scalars, numpy arrays, and traced values (jax tracers expose no concrete
    ``device``) yield None, which makes ``asarray_on_device`` omit the
    ``device`` keyword entirely -- so the numpy path keeps issuing the exact
    same ``asarray`` call as before (byte-identical, and safe on numpy
    versions without an ``asarray`` device keyword), and device placement
    under jit tracing is left to the tracer.
    """
    for coord in coords:
        if is_backend_array(coord):
            device = getattr(coord, "device", None)
            if device is not None:
                return device
    return None


def asarray_on_device(xp, a, device, dtype=None):
    """``xp.asarray(a, dtype=dtype)`` placed on ``device`` when one is given.

    ``device`` is the result of ``device_of`` on the coordinate inputs; when
    it is None (numpy arrays, plain scalars, traced values) the keyword is
    omitted so the call reduces to today's plain ``xp.asarray`` (and
    ``dtype=None`` is the default pass-through on every backend).
    """
    if device is None:
        return xp.asarray(a, dtype=dtype)
    return xp.asarray(a, dtype=dtype, device=device)


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
