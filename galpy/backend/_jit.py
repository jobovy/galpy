###############################################################################
#   galpy.backend._jit: opt-in trace mode for the boundary.
#
#   galpy never jits itself (see the no-internal-jit rule): the library is
#   written to be jit-COMPATIBLE and the user decides when to trace. This module
#   is how a user -- or the test suite -- says "trace it", once, globally:
#
#       galpy.backend.set_jit("jax")      # jax.jit every boundary call
#       with galpy.backend.jit("torch"):  # torch.compile every boundary call
#           ...
#
#   Every public entry point already carries ``@backend_input(*coords)``, which
#   names its COORDINATES explicitly. That declaration is exactly the split
#   ``jax.jit`` needs: coordinates are traced, and everything else (derivative
#   orders, ``forceint``, integrator names, the potential itself) is static. So
#   trace mode is a few lines at the same boundary rather than a jit annotation
#   scattered over the library, and turning it off restores the eager path
#   exactly.
#
#   OFF by default and a hard no-op on the numpy path, so nothing about the
#   numpy behaviour changes.
###############################################################################
from contextlib import contextmanager
from contextvars import ContextVar

# "off" | "jax" | "torch". Process-wide, like the backend selection itself.
_JIT_CTX = ContextVar("galpy_jit_mode", default="off")
# Set while a traced call is on the stack. Entry points call each other
# (Potential.__call__ -> evaluatePotentials -> ...), and re-tracing at every
# nested boundary multiplies compile time for no benefit -- the outermost trace
# already inlines the inner calls.
_TRACING = ContextVar("galpy_jit_tracing", default=False)

_VALID = ("off", "jax", "torch")


def set_jit(mode):
    """Trace every boundary call under the named framework ("jax"/"torch"/"off").

    Parameters
    ----------
    mode : str
        ``"jax"`` wraps each entry point in ``jax.jit``, ``"torch"`` in
        ``torch.compile``, ``"off"`` (default) leaves galpy eager.

    Notes
    -----
    - 2026-07-25 - Written - Bovy (UofT)
    """
    if mode not in _VALID:
        raise ValueError(f"jit mode must be one of {_VALID}, not {mode!r}")
    _JIT_CTX.set(mode)


def jit_mode():
    """Return the active trace mode ("off" when galpy is running eager)."""
    return _JIT_CTX.get()


@contextmanager
def jit(mode):
    """Context manager form of :func:`set_jit`."""
    if mode not in _VALID:
        raise ValueError(f"jit mode must be one of {_VALID}, not {mode!r}")
    token = _JIT_CTX.set(mode)
    try:
        yield
    finally:
        _JIT_CTX.reset(token)


def _static_key(val):
    """A hashable key standing in for a static argument.

    Unhashable static arguments are ordinary here -- a list of potentials is the
    standard
    way to pass a composite potential -- so fall back to identity for those. jax
    keeps its static arguments alive in the trace cache, so an id kept as a key
    cannot be recycled onto a different object while the entry is live.
    """
    if isinstance(val, (list, tuple)):
        return (tuple, tuple(_static_key(v) for v in val))
    if isinstance(val, dict):
        return (dict, tuple((k, _static_key(v)) for k, v in sorted(val.items())))
    try:
        hash(val)
    except TypeError:
        return (id, id(val))
    return (type(val), val)


class _StaticArgs:
    """Carries the non-coordinate arguments through ``jax.jit`` as one static."""

    __slots__ = ("value", "_key")

    def __init__(self, value):
        self.value = value
        self._key = _static_key(value)

    def __hash__(self):
        return hash(self._key)

    def __eq__(self, other):
        return isinstance(other, _StaticArgs) and self._key == other._key


_JITTED = {}


def _jax_shim(method):
    """``jax.jit``-ed shim taking (static bundle, *coords); built once per method."""
    import jax

    def shim(_static_args, *coord_vals):
        args, kwargs, order = _static_args.value
        args = list(args)
        kwargs = dict(kwargs)
        for (positional, key), val in zip(order, coord_vals):
            if positional:
                args[key] = val
            else:
                kwargs[key] = val
        token = _TRACING.set(True)
        try:
            return method(*args, **kwargs)
        finally:
            _TRACING.reset(token)

    return jax.jit(shim, static_argnums=(0,))


def _torch_compiled(method):
    """``torch.compile``-ed method. dynamo derives its own guards, so unlike jax
    it needs no static/traced split -- the declaration is only used by jax."""
    import torch

    def call(*args, **kwargs):
        token = _TRACING.set(True)
        try:
            return method(*args, **kwargs)
        finally:
            _TRACING.reset(token)

    return torch.compile(call, fullgraph=False, dynamic=False)


def traced_call(method, args, kwargs, slots, nargs):
    """Run ``method`` under the active trace mode, or return ``NOT_TRACED``.

    ``slots`` is ``@backend_input``'s precomputed (name, positional index) list:
    those arguments are the traced coordinates and every other argument is
    static.
    """
    mode = _JIT_CTX.get()
    if mode == "off" or _TRACING.get():
        return NOT_TRACED
    if mode == "torch":
        compiled = _JITTED.get((method, "torch"))
        if compiled is None:
            compiled = _JITTED[(method, "torch")] = _torch_compiled(method)
        return compiled(*args, **kwargs)
    # jax: split the call into traced coordinates and one static bundle.
    args = list(args)
    kwargs = dict(kwargs)
    order, coord_vals = [], []
    for name, index in slots:
        if index is not None and index < nargs:
            order.append((True, index))
            coord_vals.append(args[index])
            args[index] = None  # placeholder; the shim writes the tracer back
        elif name in kwargs:
            order.append((False, name))
            coord_vals.append(kwargs.pop(name))
    jitted = _JITTED.get((method, "jax"))
    if jitted is None:
        jitted = _JITTED[(method, "jax")] = _jax_shim(method)
    return jitted(_StaticArgs((tuple(args), kwargs, tuple(order))), *coord_vals)


class _NotTraced:
    """Sentinel: trace mode is off (or we are already inside a trace)."""

    __bool__ = staticmethod(lambda: False)
    __repr__ = staticmethod(lambda: "NOT_TRACED")


NOT_TRACED = _NotTraced()
