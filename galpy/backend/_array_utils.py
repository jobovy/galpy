###############################################################################
#   galpy.backend._array_utils: small backend-agnostic array helpers.
#
#   Namespace-dispatched array operations whose array-api / numpy spelling
#   differs across backends (or that need the array-api-compat torch median
#   workaround), factored out of the individual compute modules so there is one
#   implementation instead of a per-module copy. The numpy path is byte-identical
#   to the plain numpy spelling.
###############################################################################
from ._namespaces import is_backend_array
from ._resolver import get_namespace


def apply_amp(amp, result):
    """Multiply a potential ``result`` by its amplitude ``amp``, coercing a
    stored backend-array amp onto the result's namespace (or vice versa).

    A potential constructed under a forced backend stores a backend-array
    ``_amp`` (e.g. from ``normalize()``); evaluating it under a different active
    namespace -- a forced-numpy action-angle island, say -- then mixes a backend
    amp with a numpy result. Coerce so both sides share a namespace. The
    pure-numpy path (a numpy/Python-scalar amp AND a numpy/scalar result) is left
    untouched, so it is byte-identical to the plain ``amp * result``.
    """
    if is_backend_array(amp) or is_backend_array(result):
        amp = get_namespace(result).asarray(amp)
    return amp * result


def atleast_1d(x):
    """Backend-agnostic ``numpy.atleast_1d``.

    Resolves the namespace from ``x`` and promotes a numpy/Python scalar onto
    the active backend first (under a forced backend the per-object scalar path
    otherwise mixes numpy and backend values; ``asarray`` unifies them). numpy
    path stays byte-identical.
    """
    xp = get_namespace(x)
    return xp.atleast_1d(xp.asarray(x))


def median(xp, a, axis=None, skipnan=False):
    """Backend-agnostic ``numpy.median`` (mean of the two central order
    statistics for even counts).

    ``array-api-compat`` torch's ``median``/``nanmedian`` return the *lower* of
    the two central values for even counts, so torch is routed through
    ``quantile(.,0.5)`` (no axis) or an explicit sort (along an axis) to match
    numpy's convention. ``skipnan=True`` drops NaNs first (flattening, so use
    with ``axis=None``), matching ``numpy.nanmedian``. The median is a
    non-differentiable selection (eager-only). numpy/jax is byte-identical to
    ``numpy.nanmedian`` / ``numpy.median``.
    """
    is_torch = "torch" in getattr(xp, "__name__", "")
    if skipnan:
        a = a[~xp.isnan(a)]
        return xp.quantile(a, 0.5) if is_torch else xp.nanmedian(a)
    if axis is None:
        return xp.quantile(a, 0.5) if is_torch else xp.median(a)
    # Along an axis: the lower-median torch behaviour is wrong for even counts,
    # so take the mean of the two central order statistics from a sort. This is
    # numpy.median's own definition, hence byte-identical on numpy/jax too.
    s = xp.sort(a, axis=axis)
    n = a.shape[axis]
    lo = [slice(None)] * a.ndim
    hi = [slice(None)] * a.ndim
    lo[axis] = (n - 1) // 2
    hi[axis] = n // 2
    return 0.5 * (s[tuple(lo)] + s[tuple(hi)])
