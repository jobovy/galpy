###############################################################################
#   galpy.backend._decorators: decorators for the multi-backend dispatch layer.
###############################################################################
import functools

from ._namespaces import is_backend_array
from ._resolver import use


def numpy_island(func):
    """Decorator that runs a method or function under numpy when none of its
    inputs are backend arrays.

    Many compute paths build against inherently-numpy structures (scipy splines,
    in-place table normalization, the vectorised numpy cores, or a closed-form
    body that calls backend-promoting potential evaluators internally). Under a
    forced jax/torch backend with numpy/scalar inputs those evaluators promote to
    the active backend and feed backend arrays into the numpy code, mixing dtypes.

    Force numpy for the whole call in that case so the numpy path stays
    byte-identical and returns numpy; a real backend-array input (in args or
    kwargs) is left untouched and runs the backend-native path. No-op under an
    unforced numpy run. Works on both methods (the leading ``self`` is never a
    backend array) and plain functions.

    Examples
    --------
    >>> @numpy_island
    ... def _evaluate(self, R, vR, vT, z, vz):
    ...     ...   # numpy core; a torch/jax ARRAY input runs the backend path
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if any(is_backend_array(_a) for _a in args) or any(
            is_backend_array(_a) for _a in kwargs.values()
        ):
            return func(*args, **kwargs)
        with use("numpy", force=True):
            return func(*args, **kwargs)

    return wrapper
