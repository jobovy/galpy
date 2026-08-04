"""Helpers for pickling objects that hold scipy piecewise polynomials.

scipy 1.18 caches the array namespace -- a module -- on ``PPoly``/``BPoly``
instances, so anything holding one raises "cannot pickle 'module' object". The
coefficients themselves pickle fine, so drop the scipy object on the way out and
rebuild it on the way back in.

Rebuilding is done at the **base** class (``PPoly``/``BPoly``), not at
``type(obj)``, which matters for subclasses whose constructor takes a different
signature: ``CubicSpline(x, y)`` takes samples, not ``(c, x)`` coefficients, so
``type(obj)(obj.c, obj.x)`` raises "`x` must be 1-dimensional". Rebuilding a
``CubicSpline`` as a ``PPoly`` reproduces its values exactly (bit-for-bit) and
keeps the ``.c``/``.x``/evaluation API that callers use; only the subclass
identity is lost, and nothing in galpy branches on it. ``PPoly.construct_fast``
would preserve the subclass but returns an object that is itself still
unpicklable, so it is not an option.
"""

import copy

from scipy.interpolate import BPoly, PPoly

_SPLINE_SURROGATE = "__galpy_spline__"


def pack_splines(obj):
    """Replace scipy piecewise polynomials in a nested list with plain tuples.

    Returns ``obj`` unchanged when it holds nothing to pack, so it is safe to
    apply to any attribute.
    """
    if isinstance(obj, (PPoly, BPoly)):
        base = BPoly if isinstance(obj, BPoly) else PPoly
        return (_SPLINE_SURROGATE, base, obj.c, obj.x, obj.extrapolate, obj.axis)
    if isinstance(obj, list):
        return [pack_splines(v) for v in obj]
    return obj


def unpack_splines(obj):
    """Inverse of :func:`pack_splines`."""
    if isinstance(obj, tuple) and len(obj) == 6 and obj[0] == _SPLINE_SURROGATE:
        _, base, c, x, extrapolate, axis = obj
        return base(c, x, extrapolate=extrapolate, axis=axis)
    if isinstance(obj, list):
        return [unpack_splines(v) for v in obj]
    return obj


class SplinePickleMixin:
    """Pickle support for classes holding scipy piecewise polynomials.

    Subclasses list the attributes to pack in ``_PICKLE_SPLINE_ATTRS``. Names
    that are absent on a given instance are skipped, so an attribute created
    only by some constructor branch (a time-dependent expansion, say) needs no
    special casing.
    """

    _PICKLE_SPLINE_ATTRS = ()

    def __getstate__(self):
        pdict = copy.copy(self.__dict__)
        for name in self._PICKLE_SPLINE_ATTRS:
            if name in pdict:
                pdict[name] = pack_splines(pdict[name])
        return pdict

    def __setstate__(self, pdict):
        self.__dict__ = pdict
        for name in self._PICKLE_SPLINE_ATTRS:
            if name in self.__dict__:
                setattr(self, name, unpack_splines(self.__dict__[name]))
