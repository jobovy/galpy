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


def under_jax_trace(*xs):
    """True iff jax is imported AND one of ``xs`` is a jax tracer (jit/grad/vmap).

    The predicate that gates the eager-loop-vs-``lax.fori_loop`` choice wherever
    galpy rolls a fixed-schedule loop (bracket expansion, bisection, ...): the
    eager Python loop stays byte-identical and ~9x faster outside a trace, while
    under a jax trace the same body is rolled into a ``fori_loop`` so its ``n``
    embedded copies of the physics closure do not unroll into the user's jaxpr.

    Cheap on numpy/torch and on plain (untraced) jax arrays: if ``jax`` is not
    even imported we short-circuit to ``False`` (via ``sys.modules``, so the
    numpy/torch eager paths never import jax). This is deliberately gated on
    ``sys.modules`` rather than the ``_JAX_LOADED`` install flag so a jax-
    installed-but-unused run (pure numpy/torch) keeps the eager hot path from
    importing jax at all.
    """
    import sys

    if "jax" not in sys.modules:
        return False
    import jax

    return any(isinstance(x, jax.core.Tracer) for x in xs)


def under_torch_grad(*xs):
    """True iff torch is imported, grad is enabled, and some input is a grad tensor."""
    import sys

    if "torch" not in sys.modules:
        return False
    import torch

    return torch.is_grad_enabled() and any(
        isinstance(x, torch.Tensor) and x.requires_grad for x in xs
    )


def stop_gradient(x):
    """Backend stop-gradient: identity (numpy), ``jax.lax.stop_gradient`` / ``.detach``."""
    import sys

    if "jax" in sys.modules:
        import jax

        if isinstance(x, jax.Array):
            return jax.lax.stop_gradient(x)
    if "torch" in sys.modules:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach()
    return x


def graft_gradient(value, donor):
    """Forward value of ``value`` with the first derivative of ``donor``.

    The ``bisect_root`` stop-gradient reparameterisation:
    ``sg(value) + donor - sg(donor)`` equals ``value`` exactly (the donor terms
    cancel in floating point) while AD sees only ``donor``. First-order only.
    """
    return stop_gradient(value) + donor - stop_gradient(donor)


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


def _backend_dtype(xp, dtype):
    """Map a numpy dtype to the active backend's dtype.

    Some callers hand ``asarray_on_device`` a *numpy* dtype taken off a
    coordinate (``dtype=getattr(R, "dtype", None)``); ``torch.asarray`` rejects
    a numpy dtype (``torch.asarray(x, dtype=numpy.float64)`` raises), so it is
    translated to ``xp``'s own same-named dtype (``torch.float64``). The numpy
    path is a strict pass-through (``xp is numpy`` -> dtype unchanged), as is
    ``None``; jax accepts numpy dtypes natively, so only a numpy dtype handed to
    a backend that does not expose it as-is gets translated (``getattr`` falls
    back to the original dtype when the backend has no same-named attribute).
    """
    if dtype is None or xp is numpy:
        return dtype
    try:
        is_numpy_dtype = numpy.issubdtype(dtype, numpy.generic)
    except TypeError:  # not a numpy dtype (already a torch.dtype): leave it
        return dtype
    if not is_numpy_dtype:
        return dtype
    return getattr(xp, numpy.dtype(dtype).name, dtype)


def asarray_on_device(xp, a, device, dtype=None):
    """``xp.asarray(a, dtype=dtype)`` placed on ``device`` when one is given.

    ``device`` is the result of ``device_of`` on the coordinate inputs; when
    it is None (numpy arrays, plain scalars, traced values) the keyword is
    omitted so the call reduces to today's plain ``xp.asarray`` (and
    ``dtype=None`` is the default pass-through on every backend). A numpy
    ``dtype`` argument is translated to the backend's own dtype first so
    ``torch.asarray(x, dtype=numpy.float64)`` (which raises) works.
    """
    dtype = _backend_dtype(xp, dtype)
    if device is None:
        return xp.asarray(a, dtype=dtype)
    try:
        return xp.asarray(a, dtype=dtype, device=device)
    except (TypeError, ValueError):
        # The namespace rejects this device value/kwarg (array-api jax exposes
        # .device as the string 'cpu', and jnp.asarray(device='cpu') raises
        # ValueError; a namespace without a device= kwarg raises TypeError):
        # fall back to a device-less asarray. A genuine dtype error re-raises
        # from the fallback (same dtype, no device), so it is not masked.
        return xp.asarray(a, dtype=dtype)


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
