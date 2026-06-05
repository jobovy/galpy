###############################################################################
#   galpy.backend._resolver: the single backend-resolution mechanism.
#
#   Precedence (the "follow the data" rule of the design):
#     1. an explicit xp=/backend= override (escape hatch),
#     2. a *forced* default backend (testing / "run everything in backend X"),
#     3. the namespace of the array arguments (so autodiff "just works"),
#     4. the (non-forced) context-manager / global default,
#     5. numpy.
#
#   The forced default (set via use(..., force=True) / set_default_backend(...,
#   force=True)) overrides the data so the whole test suite can be pinned to a
#   single backend; the ordinary default (force=False) only applies when there is
#   no array to dispatch on, so production autodiff (data-first) is unaffected.
###############################################################################
import contextlib
import contextvars

import numpy

from ._namespaces import namespace_for_name, namespace_from_arrays

# Thread-/async-safe default backend; stores (name, force) or None.
_BACKEND_CTX = contextvars.ContextVar("galpy_backend", default=None)


def get_namespace(*arrays, xp=None):
    """Resolve the array namespace to use for a computation.

    Parameters
    ----------
    *arrays
        The array arguments of the computation. Their type selects the backend
        (the primary mechanism: pass a jax/torch array and galpy uses jax/torch).
    xp : module or str, optional
        Explicit override. A namespace module, or one of 'numpy'|'jax'|'torch'.

    Returns
    -------
    module
        The array namespace (``numpy`` / ``jax.numpy`` / array-api-compat torch).
    """
    if xp is not None:
        return namespace_for_name(xp)
    ctx = _BACKEND_CTX.get()
    if ctx is not None and ctx[1]:  # forced default beats the data
        return namespace_for_name(ctx[0])
    ns = namespace_from_arrays(arrays)
    if ns is not None:
        return ns
    if ctx is not None:
        return namespace_for_name(ctx[0])
    return numpy


def backend():
    """Return the name of the current default backend ('numpy' if unset)."""
    ctx = _BACKEND_CTX.get()
    return ctx[0] if ctx is not None else "numpy"


@contextlib.contextmanager
def use(name, force=False):
    """Context manager selecting the default backend for enclosed code.

    With ``force=False`` (default), data-first dispatch still wins inside the
    block: passing a jax/torch array overrides this default; the default only
    applies when there is no array to dispatch on. With ``force=True`` the backend
    is used regardless of the input array type (used by the test harness to pin a
    whole run to one backend).

    Examples
    --------
    >>> with galpy.backend.use("jax"):
    ...     ...
    """
    namespace_for_name(name)  # validate eagerly
    token = _BACKEND_CTX.set((name, force))
    try:
        yield namespace_for_name(name)
    finally:
        _BACKEND_CTX.reset(token)


def set_default_backend(name, force=False):
    """Set the process-wide default backend (no automatic restoration).

    Mirrors ``use`` but without a context manager; intended for scripts, the
    config-file default, and the test harness. ``name`` is validated immediately.
    """
    namespace_for_name(name)  # validate
    _BACKEND_CTX.set((name, force))


def _seed_from_config():
    """Seed the default backend from the [backend] config section at import."""
    try:
        from ..util.config import __config__

        name = __config__.get("backend", "default", fallback="numpy")
    except Exception:  # pragma: no cover
        name = "numpy"
    if name and name.lower() != "numpy":  # pragma: no cover - non-default config only
        try:
            set_default_backend(name)
        except (ImportError, ValueError):
            # configured backend not importable / invalid: stay on numpy
            pass
