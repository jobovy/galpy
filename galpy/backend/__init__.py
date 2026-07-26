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
from ._coerce import (
    as_backend_constant,
    coerce_coords,
    promote_scalars,
    zeros_like_backend,
)
from ._compat import is_backend_compatible
from ._input import backend_input
from ._jit import jit, jit_mode, set_jit
from ._namespaces import (
    as_numpy,
    asarray_on_device,
    device_of,
    exit_cast,
    is_backend_array,
    match_input_dtype,
    name_of_namespace,
    resolve_namespace,
)
from ._resolver import (
    _seed_from_config,
    backend,
    get_namespace,
    set_default_backend,
    use,
)
from .autodiff import autodiff_ops

# Seed the default backend from the [backend] section of the config file.
_seed_from_config()

__all__ = [
    "autodiff_ops",
    "backend_input",
    "jit",
    "jit_mode",
    "set_jit",
    "get_namespace",
    "backend",
    "use",
    "set_default_backend",
    "is_backend_array",
    "is_backend_compatible",
    "match_input_dtype",
    "name_of_namespace",
    "device_of",
    "asarray_on_device",
    "as_numpy",
    "exit_cast",
    "resolve_namespace",
    "as_backend_constant",
    "coerce_coords",
    "promote_scalars",
    "zeros_like_backend",
]
