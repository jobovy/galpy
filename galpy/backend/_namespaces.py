###############################################################################
#   galpy.backend._namespaces: helpers mapping backend names to array
#   namespaces and small namespace-agnostic utilities.
###############################################################################
from functools import wraps

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


def has_concrete_truth_value(x):
    """True if a Python ``bool`` can be read from the 0-d/scalar ``x`` right now.

    False exactly when ``x`` is a traced value (jax/torch): under a trace a
    comparison is a tracer with no truth value, so `if` on it raises. That lets a
    routine keep its eager error path -- raise on an out-of-domain input, where
    the check is a real check -- and degrade to an elementwise select while
    tracing, instead of simply failing to trace.

    ``x`` must already be reduced to a scalar/0-d (e.g. via ``xp.all``): a
    multi-element numpy array has no truth value either, and would be reported
    here as non-concrete.
    """
    try:
        bool(x)
    except Exception:
        return False
    return True


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


def under_trace(*xs):
    """True iff one of ``xs`` is being TRACED rather than concretely evaluated.

    The backend-agnostic generalisation of ``under_jax_trace``, for the "do I
    have a concrete value?" probes that pick an out-of-backend (scipy/numpy)
    computation when they do. A jax tracer answers by raising from ``float(x)``;
    a torch tensor under ``torch.compile`` does NOT -- dynamo turns ``float(x)``
    into a symbolic scalar, so the probe wrongly takes the concrete branch and
    drags the scipy/numpy code into the graph (where it dies on a data-dependent
    output shape). ``torch.compiler.is_compiling()`` is dynamo's own answer.
    False on numpy, and on plain (untraced) jax/torch arrays.
    """
    import sys

    if under_jax_trace(*xs):
        return True
    if "torch" not in sys.modules:
        return False
    import torch

    from ._tracectx import is_compiling

    return is_compiling() and any(isinstance(x, torch.Tensor) for x in xs)


def untraceable_setup(method):
    """Mark a lazy SETUP/table builder so a tracer never traces it.

    The table-backed potentials build their constant backend tables lazily and
    memoize them on the instance, so numpy-only users never pay for the build.
    That build is pure numpy/scipy (FITPACK splines, PPoly basis changes,
    ``scipy.special.comb``, ...) and is not traceable: when the very FIRST call
    happens to be inside a ``torch.compile`` region, dynamo tries to trace the
    scipy code and dies (``scipy.special.comb`` -> "'numpy.float64' object does
    not support item assignment"). It is also pointless to trace -- the result
    is a dict of constants, identical on every call.

    ``torch.compiler.disable`` tells dynamo to run the builder eagerly (as an
    opaque call) and feed its constants back into the graph. torch is imported
    only if it is ALREADY imported, so numpy-only runs never touch it, and the
    disabled view is built once and reused. jax needs nothing here: a lazy build
    under ``jax.jit`` sees concrete (untraced) values because the tables depend
    on nothing traced.
    """
    disabled = []

    @wraps(method)
    def wrapper(*args, **kwargs):
        import sys

        if "torch" not in sys.modules:
            return method(*args, **kwargs)
        if not disabled:
            import torch

            disabled.append(torch.compiler.disable(method))
        return disabled[0](*args, **kwargs)

    return wrapper


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


def as_numpy(x):
    """Pull a backend array back to numpy (a consumption/sampling boundary)."""
    if not is_backend_array(x):
        return x
    if hasattr(x, "detach"):  # torch (possibly CUDA)
        return x.detach().cpu().numpy()
    return numpy.asarray(x)


def exit_cast(value, *inputs):
    """Cast a public physical/consumption output back to numpy unless a caller
    input is a backend array (autodiff keeps backend arrays). Under a forced
    backend a scalar/numpy call still computes on the backend, but astropy
    can't hold a backend-array Quantity, so consumption outputs stay numpy."""
    if any(is_backend_array(a) for a in inputs):
        return value
    return as_numpy(value)


def resolve_namespace(*args):
    """Namespace from the backend-array args, else the context/forced default
    (list/None/scalar inputs stay resolver-safe). A follow-the-data variant of
    ``get_namespace`` for leaves consumed by both numpy and backend callers."""
    from ._resolver import get_namespace

    return get_namespace(*(a for a in args if is_backend_array(a)))


def prefer_backend_namespace(*args):
    """Namespace of the backend args, falling back to probing ALL of them.

    For entry points whose inputs may legitimately MIX backend arrays with
    numpy/python ones -- e.g. backend coordinates with a numpy ``Xsun``.
    Probing the mix raises "Multiple namespaces"; the numpy values are weak and
    get coerced across, so the backend ones decide.

    Distinct from :func:`resolve_namespace`, which falls back to the
    context/forced namespace: that is right for a leaf shared by numpy and
    backend callers, but wrong here, because it would reroute an all-numpy call
    onto a forced backend rather than following the arguments it was given.
    """
    from ._resolver import get_namespace

    on_backend = [a for a in args if is_backend_array(a)]
    return get_namespace(*(on_backend or args))


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
    xp = namespace_from_arrays((out,))
    # Normalise every coord dtype to ``out``'s namespace: an unmigrated potential
    # (or the Python integrator) can hand mixed numpy+backend coords, so ``dtypes``
    # carries a numpy dtype object alongside a torch dtype -- torch's ``result_type``
    # (and ``astype``) reject a numpy dtype. ``_backend_dtype`` is a strict
    # pass-through when ``xp is numpy`` (numpy path byte-identical) and maps a numpy
    # dtype to the backend's same-named dtype otherwise.
    dtypes = [_backend_dtype(xp, dtype) for dtype in dtypes]
    if all(dtype == dtypes[0] for dtype in dtypes):
        target = dtypes[0]
    else:
        target = xp.result_type(*dtypes)
    if target == out_dtype:
        return out
    if isinstance(out, (numpy.ndarray, numpy.generic)):
        # plain numpy: ndarray.astype works on every supported numpy version
        return out.astype(target)
    return xp.astype(out, target)


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


_DEFAULT_DEVICE = {}


def default_device(xp):
    """Device ``xp`` places a new array on when none is requested.

    Asking an explicit device of ``asarray`` is not free -- it commits the array
    rather than letting the backend place it, measured at 397.6 us/call against
    134.6 us/call for ``device=None`` on jax CPU. The coordinate boundary is
    crossed in a tight loop during an orbit integration, so paying that on every
    coercion turned a 1.7 s dxdv test into a 300 s timeout (galpy #1300). Callers
    compare against this to skip the device keyword when it cannot change
    placement anyway. Cached because asking costs an allocation.
    """
    key = getattr(xp, "__name__", None) or repr(xp)
    if key not in _DEFAULT_DEVICE:
        try:
            _DEFAULT_DEVICE[key] = getattr(xp.asarray(0.0), "device", None)
        except Exception:  # pragma: no cover - namespace without asarray/device
            _DEFAULT_DEVICE[key] = None
    return _DEFAULT_DEVICE[key]


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

    A backend dtype (``torch.float64``) is recognised structurally rather than
    by letting ``numpy`` raise on it: ``torch.compile`` turns that ``TypeError``
    into an ``InternalTorchDynamoError`` instead of letting the ``except`` below
    catch it, which would make every traced call through here fail.
    """
    if dtype is None or xp is numpy:
        return dtype
    if not isinstance(dtype, (numpy.dtype, str, type)):
        return dtype  # a backend dtype already: nothing to map
    try:
        name = numpy.dtype(dtype).name
    except TypeError:  # not a dtype numpy understands: leave it
        return dtype
    return getattr(xp, name, dtype)


def effective_device(xp, device):
    """``device``, or None when it is the backend's DEFAULT device.

    Naming the default device COMMITS the placement instead of letting the
    backend choose, which costs ~3x (see default_device). Every caller that
    hands a device to ``asarray`` wants this, so the policy lives here once:
    when it was open-coded per call site, ``as_backend_constant`` was simply
    missed and paid the cost on all 144 of its call sites (galpy #1300 was the
    same bug in ``coerce_coords``).
    """
    if device is not None and device == default_device(xp):
        return None
    return device


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
    except (TypeError, ValueError, AttributeError):
        # The namespace rejects this device value/kwarg (array-api jax exposes
        # .device as the string 'cpu', and jnp.asarray(device='cpu') raises
        # ValueError; a namespace without a device= kwarg raises TypeError; a
        # jax vmap tracer exposes .device as a SingleDeviceSharding, which
        # jnp.asarray(device=...) rejects with AttributeError under tracing):
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


def name_of_namespace(xp):
    """Map a resolved array namespace module to its canonical backend name.

    The inverse of ``namespace_for_name``: the plain ``numpy`` module -> "numpy",
    the ``jax.numpy`` namespace -> "jax", the array-api-compat torch namespace ->
    "torch"; an unrecognized namespace defaults to "numpy" (defensive).
    """
    if xp is numpy:
        return "numpy"
    name = getattr(xp, "__name__", "")
    if "jax" in name:
        return "jax"
    if "torch" in name:
        return "torch"
    return "numpy"


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


def restrict_to_single_thread():
    """Cap the array backends at one compute thread, for a forked child.

    ``galpy.util.multi.parallel_map`` forks (spawn cannot pickle the mapped
    closures, #457). torch's intra-op thread pool does not survive ``fork``: the
    child inherits pool state it cannot use and deadlocks on its first parallel
    region, hanging the parent in ``proc.join()`` forever. Calling this first
    thing in the child stops the pool from being re-entered -- and one thread per
    child is the right split anyway, since one process per core already
    saturates the machine.

    No-op when torch is not loaded. jax has no equivalent knob (it warns at
    every ``os.fork`` that a deadlock is likely, and the only cure available
    there is not forking at all), so jax is deliberately not handled here.
    """
    import sys

    torch = sys.modules.get("torch")
    if torch is not None:
        torch.set_num_threads(1)


def fork_deadlocks_backend():
    """Whether ``os.fork`` risks deadlocking the array backend in play.

    Companion to `restrict_to_single_thread`, which cures the torch case by
    capping the forked child to one thread. jax has no equivalent knob: it warns
    at every ``os.fork`` that a deadlock is likely, and the only cure is not to
    fork. Callers that fork (`galpy.util.multi.parallel_map`) must run serially
    when this is True.

    Keyed on the active backend rather than on whether jax is merely imported,
    so a numpy run that happens to have jax loaded still forks. A numpy-default
    run that feeds jax arrays to the mapped function is not covered -- deciding
    that would mean inspecting the sequence.
    """
    from ._resolver import backend

    return backend() == "jax"
