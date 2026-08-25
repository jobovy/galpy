###############################################################################
#   galpy.backend._input: the @backend_input boundary decorator.
#
#   Coerces a potential/df evaluator's DECLARED coordinate inputs onto the
#   active array backend, so torch's strict scalar handling
#   (torch.sqrt(numpy.float64) etc.) does not reject them. This is the backend
#   counterpart to the legacy ``potential_physical_input`` unit decorator --
#   kept as its OWN decorator in galpy.backend rather than bolted onto the unit
#   decorator, so the backend concern is cleanly separated from the pre-backend
#   galpy machinery.
#
#   The coordinates are named EXPLICITLY, never guessed. These entry points mix
#   coordinates with control/option parameters (``dR``/``dphi`` derivative
#   orders, ``forceint``, ``M``, ``nsigma``, ``integrate_method``, grid objects,
#   ...), and only the coordinates may be coerced -- turning a derivative order
#   into a float tensor, or an option string into an array, is either silently
#   wrong or an outright error. Declaring them per site also documents at each
#   entry point exactly what crosses onto the backend.
#
#   Stack it just INSIDE ``@potential_physical_input`` (which parses units to
#   plain floats first) and outside ``@physical_conversion``:
#
#       @potential_physical_input        # units -> floats (outer)
#       @backend_input("R", "z", "phi", "t")   # floats -> active backend (this)
#       @physical_conversion("energy", pop=True)
#       def __call__(self, R, z, phi=0.0, t=0.0, dR=0, dphi=0): ...
#
#   Declared names are checked against the signature at DECORATION time, so a
#   typo raises at import instead of silently skipping a coordinate.
#
#   No-op on the numpy path (xp is numpy), so the numpy/Quantity path stays
#   byte-identical, and -- for now -- on targets that are not backend-ready
#   (see ``_backend_ready``, a temporary guard for the few unmigrated potentials
#   that break when their coordinates arrive as jax/torch arrays).
###############################################################################
import inspect
from functools import wraps

import numpy

from ._coerce import coerce_coords
from ._compat import is_backend_compatible
from ._jit import NOT_TRACED, traced_call
from ._namespaces import device_of, is_backend_array, prefer_backend_namespace
from ._resolver import get_namespace

_EMPTY = inspect.Parameter.empty


def _backend_ready(target):
    """True if ``target``'s compute methods can take backend arrays.

    Thin alias of ``is_backend_compatible`` (the general galpy-object check),
    named for what the decorator uses it for and kept as the place to document
    WHY the decorator asks at all.

    The question is a TEMPORARY compatibility guard, not part of the decorator's
    design: an entry point that carries ``@backend_input`` should simply coerce.
    It is still asked because a handful of potentials are not migrated and break
    when their coordinates arrive as jax/torch arrays -- measured, not assumed:

      * ``interpRZPotential`` (scipy interpolation) diverges from the numpy path
        at the grid edge -- at (R,z)=(0.5,0.0) the force is off by 3.3% and the
        second derivative by 32%, in float64;
      * ``MovingObjectPotential`` interpolates the perturber's trajectory with
        ``self._orb.R(t)``, so a coerced ``t`` reaches a numpy lookup as a jax
        tracer under jit (``TracerArrayConversionError``).

    Drop this guard -- and this function -- once those are backend-native; the
    lists in tests/test_backend_input.py track what is left.
    """
    return is_backend_compatible(target)


def _coerce_one(xp, val, device=None):
    """Coerce a single declared coordinate, preserving a sequence as a sequence.

    A coordinate may be a sequence -- the ``[vR,vT,vz]`` velocity of the
    dissipative forces, whose elements can be grad-carrying tensors. Coercing it
    element-wise and rebuilding the container keeps each element's autograd
    graph; stacking it into one array with a single ``asarray`` detaches them.
    """
    # A Quantity is not a coordinate value yet -- it still carries units. Some
    # entry points strip units INSIDE the body (sphericaldf.sigmar does
    # `r = conversion.parse_length(r, ro=self._ro)`) rather than through an outer
    # units decorator, so the boundary can be handed one; asarray() of a Quantity
    # yields garbage that the later parse turns into NaN. Pass it through and let
    # the body parse it. Detected by duck-typing so this stays astropy-free.
    if hasattr(val, "unit"):
        return val
    if isinstance(val, (list, tuple)):
        return type(val)(coerce_coords(xp, *val, device=device))
    (out,) = coerce_coords(xp, val, device=device)
    return out


def backend_input(*coords):
    """Coerce the named coordinate inputs of an entry point onto the active backend.

    Parameters
    ----------
    *coords : str
        Names of the parameters that are coordinates, e.g.
        ``@backend_input("R", "z", "phi", "t")``. Only these are coerced,
        whether passed positionally or by keyword; every other parameter
        (derivative orders, flags, masses, grids, option strings) is left
        untouched. Names must exist in the decorated signature.

    Notes
    -----
    See the module docstring for placement in the decorator stack. The numpy
    path is a strict pass-through (byte-identical).
    """
    if not coords or callable(coords[0]):
        raise TypeError(
            "@backend_input must be called with the coordinate names to coerce, "
            'e.g. @backend_input("R", "z")'
        )

    def decorator(method):
        params = list(inspect.signature(method).parameters.values())
        index_of = {
            p.name: ii
            for ii, p in enumerate(params)
            if p.kind is not inspect.Parameter.VAR_KEYWORD
        }
        # A coordinate may be reachable only through **kwargs (Force.rforce takes
        # (R, z, **kwargs) and forwards phi/t): declarable, but keyword-only.
        takes_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params)
        unknown = [c for c in coords if c not in index_of]
        if unknown and not takes_var_kw:
            raise ValueError(
                f"@backend_input on {method.__qualname__}: declared coordinate(s) "
                f"{unknown} are not parameters of the decorated function"
            )
        # Precomputed at decoration time so the per-call path does no signature
        # introspection: (name, positional index) for each declared coordinate,
        # plus the defaults worth injecting when the caller omits them.
        # index None => the coordinate can only arrive by keyword (via **kwargs).
        slots = tuple((c, index_of.get(c)) for c in coords)
        # A coordinate left at a non-None signature default (phi=0.0, t=0.0) never
        # appears in args or kwargs, so it would reach the backend as a raw Python
        # float (torch.sin(0.0) raises); inject it coerced instead. None defaults
        # (v=None, z=None) are passed through by coerce_coords, so skip them.
        defaults = tuple(
            (c, ii, params[ii].default)
            for c, ii in slots
            if ii is not None
            and params[ii].default is not _EMPTY
            and params[ii].default is not None
        )

        @wraps(method)
        def wrapper(*args, **kwargs):
            nargs = len(args)
            probe = []
            for c, ii in slots:
                if ii is not None and ii < nargs:
                    val = args[ii]
                elif c in kwargs:
                    val = kwargs[c]
                else:
                    continue
                # A coordinate may be a sequence -- the [vR,vT,vz] velocity of
                # the dissipative forces. get_namespace dispatches on arrays and
                # rejects a bare list, so probe the components instead.
                if isinstance(val, (list, tuple)):
                    probe.extend(val)
                else:
                    probe.append(val)
            # Namespace follows the coordinates only. When some coordinates are
            # backend arrays and others are numpy (Orbits.E passes a numpy t=
            # alongside torch R/z), resolve from the backend ones -- the numpy
            # coordinates are weak and coerce_coords brings them across below,
            # whereas probing the mix raises "Multiple namespaces". Everything
            # below is skipped on the numpy path, so numpy pays just this probe.
            xp = prefer_backend_namespace(*probe)
            if xp is not numpy and _backend_ready(args[0]):
                # ONE device anchor for the whole call. Coercing each coordinate
                # on its own would let each derive its own: a numpy/python
                # coordinate anchors to None -> the backend default device (CPU
                # for torch) while its CUDA siblings stay on the GPU, and the
                # evaluator gets a split-device coordinate set -- a mixed-device
                # error in some potentials, a silent GPU->CPU transfer in others.
                dev = device_of(*probe)
                newargs = None
                for c, ii in slots:
                    if ii is not None and ii < nargs:
                        if newargs is None:
                            newargs = list(args)
                        newargs[ii] = _coerce_one(xp, newargs[ii], dev)
                    elif c in kwargs:
                        kwargs[c] = _coerce_one(xp, kwargs[c], dev)
                if newargs is not None:
                    args = tuple(newargs)
                for c, ii, dflt in defaults:
                    if ii >= nargs and c not in kwargs:
                        kwargs[c] = _coerce_one(xp, dflt, dev)
                # Under an opt-in trace mode this boundary is also where the
                # jit/compile happens: the declared coordinates are the traced
                # arguments and everything else is static. Off by default.
                out = traced_call(method, args, kwargs, slots, nargs, xp)
                if out is not NOT_TRACED:
                    return out
            return method(*args, **kwargs)

        return wrapper

    return decorator
