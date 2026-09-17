###############################################################################
#   galpy.backend._coerce: backend DATA-coercion helpers.
###############################################################################
"""Backend data-coercion helpers: the single home for bringing numpy/Python
data onto the active jax/torch backend.

PURPOSE
-------
This module does two related jobs:

  * it brings numpy/Python *coordinate* data onto the active backend's array
    type (so e.g. ``torch.sqrt`` -- which rejects ``numpy.float64`` -- and
    ``Tensor`` arithmetic see real backend arrays), and
  * it anchors *stored numpy constants* (rotation matrices, lookup tables, a
    zero reference coordinate) onto the dtype/device of an input array, so the
    constant joins the computation as a same-dtype/same-device backend array.

It is the single home for data-coercion. Namespace *resolution* (which backend
a call dispatches to) and the dtype/device *primitives* it builds on live in
``_namespaces.py`` (``is_backend_array``, ``device_of``, ``asarray_on_device``,
``match_input_dtype``); this module only consumes those primitives.

THE CORE INVARIANT
------------------
Every function here is a STRICT PASS-THROUGH when ``xp is numpy``: it returns
its inputs OBJECT-IDENTICALLY (no asarray, no copy, no dtype touch). This is
what keeps the numpy code path BYTE-IDENTICAL to galpy's historical behaviour.
Any new coercion helper added to this module MUST preserve this invariant --
guard the work behind ``if xp is numpy: return <inputs unchanged>`` first.

WHEN TO USE EACH
----------------
  * ``coerce_coords(xp, *coords)`` -- at the PUBLIC INPUT BOUNDARY (the
    ``@potential_physical_input`` decorator) to bring coordinate arguments onto
    the backend: plain Python/int scalars become float64 (galpy's interior
    precision), float arrays keep their dtype (so the float32 exit-cast policy
    still applies), and ``None`` passes through.
  * ``promote_scalars(xp, *vals)`` -- INSIDE coordinate transforms to promote
    plain Python scalars sitting alongside array arguments, anchored on the
    dtype/device of the first array, so mixed scalar/array inputs work on a
    backend whose functions require arrays.
  * ``as_backend_constant(xp, value, ref)`` -- to anchor a single STORED numpy
    constant (a rotation matrix, an offset, a table) on a backend ``ref`` array
    derived from the coordinate inputs.
  * ``zeros_like_backend(xp, R)`` -- for a backend ZERO reference coordinate
    (e.g. the ``z = 0`` plane a spherical-in-disguise wrapper feeds its wrapped
    potential).

WHY float64-INTERIOR / DEVICE-ANCHORING
---------------------------------------
galpy computes in float64 internally: a bare ``asarray`` of a Python float
yields torch float32 and silently misses galpy's tolerances, so plain scalars
are lifted to ``xp.float64`` while genuine float arrays keep their own dtype.
Anchoring constants and promoted scalars on an input array's dtype/device keeps
the whole computation on one device and at one precision, which is required for
torch (cross-device / mixed-dtype ops raise) and correct for jax.
"""

import numpy

from ._namespaces import (
    _is_floating_dtype,
    asarray_on_device,
    device_of,
    differentiating,
    effective_device,
    is_backend_array,
    under_trace,
)
from ._resolver import get_namespace


def coerce_coords(xp, *coords, device=None):
    """Bring coordinate inputs onto the active backend's array type.

    The dominant non-numpy failure mode is "the namespace resolved to a backend
    (forced harness, or a user mixing a backend tensor with a numpy/python arg)
    but a coordinate is still numpy/python", which torch rejects strictly
    (``torch.sqrt(numpy.float64)`` raises; ``numpy.ndarray * Tensor`` raises).
    Coercing the coordinates to backend arrays once, at the public input
    boundary, fixes it for every potential at once.

    Rules (applied only when the backend is NOT numpy):
      * ``None`` is passed through (axisymmetric ``phi=None`` etc.).
      * a coordinate that already carries a *floating* dtype (a numpy/backend
        float32/float64 array or scalar) is moved onto the backend with its
        dtype PRESERVED, so the float32/exit-cast policy (``match_input_dtype``)
        still applies.
      * a plain Python scalar (``1.0``/``1``) or an integer array is brought to
        the backend's float64 -- galpy's interior precision; a bare ``asarray``
        of a Python float would give torch float32 and miss the tolerances.

    ``device`` overrides the device the coerced coordinates land on. Without it
    the anchor is derived from ``coords`` alone, which is only right when the
    caller passes every coordinate of a call at once: coercing them one at a
    time gives each its own anchor, so a numpy coordinate anchors to None (the
    backend default device, i.e. CPU for torch) while its CUDA siblings stay on
    the GPU, and the evaluator is handed a split-device coordinate set. Callers
    that coerce coordinate-by-coordinate pass the shared anchor explicitly.

    The numpy backend is a strict pass-through (``coords`` returned object-
    identical) -> the numpy path stays byte-identical.
    """
    if xp is numpy:
        return coords
    # Only a NON-default anchor (a CUDA sibling on a CPU-default run) is worth
    # naming; effective_device drops the rest.
    dev = effective_device(xp, device_of(*coords) if device is None else device)
    out = []
    for c in coords:
        if c is None:
            out.append(c)
            continue
        dt = getattr(c, "dtype", None)
        if dt is not None and _is_floating_dtype(dt):
            out.append(asarray_on_device(xp, c, dev))  # preserve float dtype
        else:
            out.append(asarray_on_device(xp, c, dev, dtype=xp.float64))
    return tuple(out)


def promote_scalars(xp, *vals):
    """Promote plain Python scalars among ``vals`` to the active non-numpy
    namespace, anchored on the dtype/device of the first array argument, so
    that e.g. torch functions -- which require Tensors -- accept the mixed
    scalar/array inputs that the numpy path has always supported. The numpy
    path passes everything through untouched (byte-identical)."""
    if xp is numpy:
        return vals
    # "Leave it" only for genuine backend (jax/torch) arrays: a numpy.float64
    # (or numpy.ndarray) HAS .ndim but torch rejects it, so it must be PROMOTED.
    ref = next((v for v in vals if is_backend_array(v)), None)
    if ref is None:
        # No backend array to anchor on, but xp is non-numpy (a forced default,
        # or an array-API call). torch's functions REJECT numpy.float64/python
        # floats, so coerce the operands onto the backend (python/int -> float64,
        # numpy float arrays dtype-preserved) instead of passing through -- jax
        # tolerates raw scalars but the coerced values are identical under x64.
        return coerce_coords(xp, *vals)
    dtype = getattr(ref, "dtype", None)
    device = getattr(ref, "device", None)

    def _promote(v):
        if is_backend_array(v):
            return v
        # Delegate to the same helper coerce_coords uses: it places the value on
        # the device and translates a numpy dtype to the backend dtype.
        return asarray_on_device(xp, v, device, dtype=dtype)

    return tuple(_promote(v) for v in vals)


def as_backend_constant(xp, value, ref):
    """Bring a stored numpy constant (rotation matrix / offset) into the active
    namespace, anchored on the dtype/device of ``ref`` (a backend array derived
    from the coordinate inputs). The numpy path passes the stored array through
    untouched (byte-identical)."""
    if xp is numpy:
        return value
    dtype = getattr(ref, "dtype", None)
    device = effective_device(xp, getattr(ref, "device", None))
    try:
        return xp.asarray(value, dtype=dtype, device=device)
    except TypeError:  # pragma: no cover - namespace without device= kwarg
        return xp.asarray(value, dtype=dtype)


def zeros_like_backend(xp, R):
    """The numpy path passes the plain scalar through untouched
    (byte-identical); on a non-numpy backend the z = 0 reference
    coordinate is anchored on the inputs so the wrapped potential sees a
    backend array (torch functions require Tensors) on the right
    device/dtype."""
    return 0.0 if xp is numpy else xp.zeros_like(R)


def ns_unary(name, x):
    """``numpy.<name>(x)``, but on ``x``'s own namespace when it is TRACED.

    ``numpy.sqrt``/``numpy.log``/... of a tracer raises. They work EAGERLY on a
    concrete backend array (via ``__array__``), which is why this only bites
    once a stored parameter is being differentiated -- typically in a DF or
    potential constructor, where a scale factor is derived from a fit parameter.

    numpy/python input keeps the numpy call, so those paths stay byte-identical.
    A backend array takes the namespace route whether or not it is being
    differentiated: ``numpy.sqrt`` of a plain tensor does return the right value,
    but it converts out of the backend on the way and emits a numpy-2
    DeprecationWarning that CI escalates to an error.
    """
    if not (is_backend_array(x) or differentiating(x)):
        return getattr(numpy, name)(x)
    xp = get_namespace(x)
    (xv,) = coerce_coords(xp, x)
    return getattr(xp, name)(xv)


def ns_mul(a, b):
    """``a * b`` where either side may be a grad-carrying backend array.

    A grad-tracking torch tensor multiplied by a numpy array raises in BOTH
    operand orders -- each side tries to convert the other, and the conversion
    is what fails -- so the operands are brought onto a common namespace first.
    Concrete operands keep the plain product, so numpy stays byte-identical.

    This is the shape almost every "scale factor x tabulated grid" line in a DF
    or potential constructor takes once the scale factor becomes a fit
    parameter.
    """
    if not (is_backend_array(a) or is_backend_array(b) or differentiating(a, b)):
        return a * b
    av, bv = coerce_coords(grad_namespace(a, b), a, b)
    return av * bv


def grad_namespace(*xs):
    """The namespace of whichever of ``xs`` carries the gradient.

    ``get_namespace(numpy_array, tracer)`` RAISES ("Multiple namespaces for array
    inputs") -- and mixing the two is exactly the situation a gradient creates,
    where one operand is a differentiated parameter and the other a plain
    tabulated grid. Resolving from the differentiated operand alone gives the
    namespace the result has to live in; the concrete operands are then coerced
    onto it.

    Falls back to the ordinary ambient resolution when nothing is being
    differentiated.
    """
    for x in xs:
        if differentiating(x):
            return get_namespace(x)
    for x in xs:
        if is_backend_array(x):
            return get_namespace(x)
    return get_namespace(*xs)
