###############################################################################
#   galpy.backend: multi-backend (numpy / jax / torch / array-API) dispatch.
#
#   The whole of galpy's pure-Python compute layer resolves its array namespace
#   through ``get_namespace`` so that the same code runs and differentiates under
#   numpy, JAX, and PyTorch. Backend selection follows the data first (the type
#   of the array arguments), with an explicit ``xp=`` override and a
#   context-manager/global default as fallbacks. See ``_resolver`` for details.
###############################################################################
from ._namespaces import is_backend_array
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
    "set_default_backend",
    "is_backend_array",
]
