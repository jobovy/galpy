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
    is_backend_array,
)


def coerce_coords(xp, *coords):
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

    The numpy backend is a strict pass-through (``coords`` returned object-
    identical) -> the numpy path stays byte-identical.
    """
    if xp is numpy:
        return coords
    dev = device_of(*coords)
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
    device = getattr(ref, "device", None)
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
