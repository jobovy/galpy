###############################################################################
#   galpy.backend._tracectx: context variables that survive a torch.compile
#   trace.
###############################################################################
import contextvars
import sys
import threading


def _is_compiling():
    """True while dynamo is tracing (cheap, and never imports torch itself)."""
    torch = sys.modules.get("torch")
    return torch is not None and torch.compiler.is_compiling()


class TracedContextVar:
    """A ``ContextVar`` that is also readable from inside a ``torch.compile`` trace.

    ``ContextVar.get()`` is opaque to dynamo: every read severs the graph, so a
    single one on a hot dispatch path breaks the whole compiled region, and
    ``fullgraph=True`` refuses to compile at all. Inside a trace the value is
    fixed by construction -- the backend and jit mode cannot change while dynamo
    walks the bytecode -- so a mirror of the value is read instead.

    The mirror is a ``threading.local``, NOT a module global: two threads that
    force different backends must not observe each other's value, and that
    isolation is the whole reason these are ContextVars rather than globals. A
    plain thread starts with an empty context, so ``ContextVar`` and the mirror
    agree there by construction.

    Known limit, and it is narrow: they can disagree when a context is carried
    somewhere the ``set`` did not happen -- ``copy_context().run(...)`` or an
    asyncio task -- because the mirror follows the thread rather than the
    context. Only a read taken WHILE COMPILING consults the mirror, so eager
    galpy is unaffected, and galpy compiles on the thread that entered ``use``.
    """

    __slots__ = ("_var", "_mirror", "_default")

    def __init__(self, name, default=None):
        self._var = contextvars.ContextVar(name, default=default)
        self._mirror = threading.local()
        self._default = default

    def get(self):
        if _is_compiling():
            return getattr(self._mirror, "value", self._default)
        return self._var.get()

    def set(self, value):
        previous = getattr(self._mirror, "value", self._default)
        self._mirror.value = value
        return (self._var.set(value), previous)

    def reset(self, token):
        var_token, previous = token
        self._var.reset(var_token)
        self._mirror.value = previous
