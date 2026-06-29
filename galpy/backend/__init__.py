###############################################################################
#   galpy.backend: multi-backend (numpy / jax / torch / array-API) dispatch.
#
#   The whole of galpy's pure-Python compute layer resolves its array namespace
#   through ``get_namespace`` so that the same code runs and differentiates under
#   numpy, JAX, and PyTorch. Backend selection follows the data first (the type
#   of the array arguments), with an explicit ``xp=`` override and a
#   context-manager/global default as fallbacks. See ``_resolver`` for details.
#
#   See ``_coerce`` for the data-coercion helpers (bringing numpy/Python data
#   onto the active backend, anchoring stored constants) and ``_namespaces``
#   for the namespace-resolution and dtype/device primitives they build on.
###############################################################################
import functools as _functools

from ._coerce import (
    as_backend_constant,
    coerce_coords,
    promote_scalars,
    zeros_like_backend,
)
from ._namespaces import (
    asarray_on_device,
    device_of,
    is_backend_array,
    match_input_dtype,
)
from ._resolver import (
    _seed_from_config,
    backend,
    get_namespace,
    set_default_backend,
    use,
)

# Seed the default backend from the [backend] section of the config file.
_seed_from_config()


def numpy_island(func):
    """Decorator for methods/functions that compute against inherently-numpy
    structures (scipy splines, in-place table normalization, the vectorised
    numpy core) by calling backend-promoting potential evaluators internally.

    Under a forced jax/torch backend with numpy/scalar (non-backend-array)
    inputs, the potential funnels promote to the active backend, which would
    feed backend arrays into the numpy core and mix dtypes. Force numpy for the
    whole call in that case so the numpy path stays byte-identical and returns
    numpy. A real backend-array input (in args or kwargs) is left untouched,
    running the backend-native path. No-op under an unforced numpy run.

    Works for both methods (the leading ``self`` is never a backend array) and
    plain functions.
    """

    @_functools.wraps(func)
    def wrapper(*args, **kwargs):
        if any(is_backend_array(_a) for _a in args) or any(
            is_backend_array(_a) for _a in kwargs.values()
        ):
            return func(*args, **kwargs)
        with use("numpy", force=True):
            return func(*args, **kwargs)

    return wrapper


__all__ = [
    "get_namespace",
    "backend",
    "use",
    "numpy_island",
    "set_default_backend",
    "is_backend_array",
    "match_input_dtype",
    "device_of",
    "asarray_on_device",
    "as_backend_constant",
    "coerce_coords",
    "promote_scalars",
    "zeros_like_backend",
]
