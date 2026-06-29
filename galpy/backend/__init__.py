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
from ._array_utils import apply_amp, atleast_1d, median
from ._coerce import (
    as_backend_constant,
    coerce_coords,
    promote_scalars,
    zeros_like_backend,
)
from ._decorators import numpy_island
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
    "apply_amp",
    "atleast_1d",
    "median",
]
