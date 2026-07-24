###############################################################################
#   galpy.backend._kernel: the @backend_kernel decorator.
#
#   Lifts the inline
#
#       xp = get_namespace(R, z)
#       R, z = coerce_coords(xp, R, z)
#
#   boilerplate -- copy-pasted into every migrated method only because the
#   backend migration was incremental -- into one decorator. The decorated
#   function becomes the PURE kernel: it receives the coerced coordinate
#   arguments plus an injected ``xp`` keyword and does nothing but the math.
#
#   This also defines the jit seam. jax.jit / torch.compile want exactly this
#   split: outer Python glue (namespace resolution, coercion) separated from the
#   inner traceable kernel. galpy itself never jits (see the no-internal-jit
#   policy); the test suite's --backend jax-jit / torch-compile dimension wraps
#   the decorated kernels.
#
#   The coordinate arguments are declared EXPLICITLY -- @backend_kernel("R","z")
#   -- carrying exactly the information the old ``coerce_coords(xp, R, z)`` call
#   already named, so there is no name-magic and no per-method whitelist to keep
#   correct (a velocity list ``v`` must not be coerced like a scalar coord, for
#   instance). The numpy path is a strict pass-through: ``get_namespace`` returns
#   numpy and ``coerce_coords`` is object-identity, so the decorated method is
#   byte-identical to the inline version.
###############################################################################
import functools
import inspect

from ._coerce import coerce_coords
from ._resolver import get_namespace

# Test-only jit seam. None in normal galpy operation (galpy never jits itself --
# the no-internal-jit policy); the test suite's --backend jax-jit dimension sets
# this so every decorated kernel is traced through jax.jit with the declared
# (coerced) coords as the traced inputs. (torch-compile is a deferred follow-up.)
_JIT_MODE = None  # None | "jax"


def set_jit_mode(mode):
    """Enable jit wrapping of decorated kernels (tests/conftest only).

    mode in {None, "jax"}. galpy itself never calls this.
    """
    global _JIT_MODE
    _JIT_MODE = mode


def backend_kernel(*coord_names):
    """Resolve the array namespace from the named coordinate arguments, coerce
    them onto the active backend, and inject the namespace as an ``xp`` keyword.

    Usage::

        @backend_kernel("R", "z")
        def _evaluate(self, R, z, phi=0.0, t=0.0, *, xp):
            r2 = R**2.0 + z**2.0
            return xp.log(r2) / 2.0

    ``coord_names`` are the parameters holding coordinate arrays -- the same ones
    the method used to pass to ``coerce_coords`` -- given by name. The decorated
    kernel must accept a keyword-only ``xp`` (default ``None`` so it is bindable
    without the decorator, e.g. under introspection); the wrapper always supplies
    it. The numpy path is byte-identical: ``get_namespace`` -> numpy and
    ``coerce_coords`` returns its inputs object-identically.
    """

    def deco(fn):
        params = list(inspect.signature(fn).parameters.values())
        names = [p.name for p in params]
        if "xp" not in names:
            raise TypeError(
                f"@backend_kernel kernel {fn.__qualname__!r} must accept a "
                "keyword-only 'xp' parameter"
            )
        # Precompute, once, each coordinate's positional index and default so the
        # per-call path does no signature binding.
        coord_info = []
        for cn in coord_names:
            try:
                idx = names.index(cn)
            except ValueError:
                raise TypeError(
                    f"@backend_kernel({cn!r}) names no parameter of {fn.__qualname__!r}"
                )
            coord_info.append((cn, idx, params[idx].default))

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # Gather each coordinate value from wherever it was passed
            # (positional / keyword / default), remembering where so the coerced
            # value can be written back in place.
            vals = []
            locs = []
            for cn, idx, default in coord_info:
                if idx < len(args):
                    vals.append(args[idx])
                    locs.append((0, idx))  # positional
                elif cn in kwargs:
                    vals.append(kwargs[cn])
                    locs.append((1, cn))  # keyword
                else:
                    vals.append(default)
                    locs.append((2, cn))  # used its default
            xp = get_namespace(*vals)
            coerced = coerce_coords(xp, *vals)
            args = list(args)
            for (kind, key), cv in zip(locs, coerced):
                if kind == 0:
                    args[key] = cv
                else:  # keyword, or a defaulted coord we now pass explicitly
                    kwargs[key] = cv
            kwargs["xp"] = xp
            if _JIT_MODE is None:
                return fn(*args, **kwargs)
            # Test jit dimension: trace the kernel through jax.jit over the declared
            # coords that are actual backend arrays (now coerced, so traceable),
            # closing over everything else -- self, xp, and any None/scalar declared
            # coord (e.g. phi=None axisymmetric) -- as static. Fresh per call: this
            # tests traceability, not speed. All coerced coords (incl. the static
            # ones) are already placed in args/kwargs above. (torch.compile is
            # deferred: its dynamo cannot trace the more complex kernels.)
            import jax

            from ._namespaces import is_backend_array

            frozen_args, frozen_kwargs = list(args), dict(kwargs)
            trace_slots = [sl for sl, cv in zip(locs, coerced) if is_backend_array(cv)]
            trace_vals = [cv for cv in coerced if is_backend_array(cv)]

            def _call_with(coord_vals):
                a2, k2 = list(frozen_args), dict(frozen_kwargs)
                for (kind, key), cv in zip(trace_slots, coord_vals):
                    if kind == 0:
                        a2[key] = cv
                    else:
                        k2[key] = cv
                return fn(*a2, **k2)

            return jax.jit(_call_with)(trace_vals)

        return wrapper

    return deco
