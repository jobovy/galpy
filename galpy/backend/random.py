###############################################################################
#   galpy.backend.random: a backend-agnostic random-number shim.
#
#   The single home for galpy's random draws so the same sampling code runs on
#   numpy, jax, and torch. It exists to unblock differentiable/jit-able sampling
#   (the reparameterization / common-random-numbers trick) while keeping the
#   numpy path byte-identical to galpy's historical behaviour.
#
#   DISPATCH is on the KEY, not on get_namespace:
#     * key is None            -> the GLOBAL ``numpy.random.*`` (stateful,
#                                 byte-identical: the shim calls the exact same
#                                 ``numpy.random.<fn>`` the code called before).
#     * key is a jax key       -> ``jax.random.*`` (a key is explicit data; the
#                                 draw is a pure deterministic function of it).
#     * key is a ``_TorchKey`` -> a ``torch.Generator`` seeded from the key, so
#                                 a draw is reproducible given the key and the
#                                 drawn noise tensor is a constant that a
#                                 reparameterized transform differentiates through.
#
#   The numpy path is DELIBERATELY stateful/global: numpy is never
#   differentiated or jit'd here, so it keeps its current behaviour exactly.
#   jax's stateless key model gives fixed-noise derivatives natively; torch does
#   it via draw-once-fixed-noise.
###############################################################################
import numpy

from ..util._optional_deps import _JAX_LOADED
from ._namespaces import namespace_for_name
from ._resolver import get_namespace

__all__ = [
    "key",
    "split",
    "uniform",
    "normal",
    "random",
    "randint",
    "choice",
    "multivariate_normal",
]


class _TorchKey:
    """A stateless torch key: a 64-bit seed a ``torch.Generator`` is built from.

    torch has no native stateless key model, so a key is represented as a
    Python int seed. ``split`` derives independent child seeds deterministically
    from it (via a Generator seeded by this seed), and every draw builds a fresh
    ``torch.Generator().manual_seed(seed)`` -- so a draw is a pure function of
    the key (reproducible), and the drawn tensor is a constant (no grad) that a
    reparameterized transform differentiates through.
    """

    __slots__ = ("seed",)

    def __init__(self, seed):
        self.seed = int(seed) & 0xFFFFFFFFFFFFFFFF

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"_TorchKey(seed={self.seed})"


# --- backend/name/shape helpers ----------------------------------------------
def _name_of_namespace(xp):
    """Map a resolved namespace module to a backend name ('numpy'|'jax'|'torch')."""
    if xp is numpy:
        return "numpy"
    name = getattr(xp, "__name__", "")
    if "jax" in name:
        return "jax"
    if "torch" in name:
        return "torch"
    return "numpy"  # pragma: no cover - defensive


def _resolve_backend_name(backend):
    """Backend name for ``key(...)``: explicit ``backend=`` or the active default."""
    if backend is None:
        return _name_of_namespace(get_namespace())
    return _name_of_namespace(namespace_for_name(backend))


def _backend_of_key(key):
    """Dispatch: which backend a draw for ``key`` uses (numpy for ``key is None``)."""
    if key is None:
        return "numpy"
    if isinstance(key, _TorchKey):
        return "torch"
    if _JAX_LOADED:
        import jax

        if isinstance(key, jax.Array):
            return "jax"
    raise TypeError(
        "galpy.backend.random: unrecognized key; pass None (numpy), a jax "
        "key from key(seed, 'jax'), or a _TorchKey from key(seed, 'torch')"
    )


def _tuple_shape(shape):
    """Normalize a shape to a tuple for jax/torch (``None`` -> scalar ``()``)."""
    if shape is None:
        return ()
    if isinstance(shape, (int, numpy.integer)):
        return (int(shape),)
    return tuple(shape)


def _torch_generator(key):
    import torch

    g = torch.Generator()
    g.manual_seed(key.seed)
    return g


# --- key management ----------------------------------------------------------
def key(seed, backend=None):
    """Create a random key for the resolved backend.

    Parameters
    ----------
    seed : int
        The seed.
    backend : {None, 'numpy', 'jax', 'torch'} or namespace module, optional
        Which backend to make a key for. ``None`` (default) uses the active
        galpy backend (``galpy.backend.get_namespace`` context/global default),
        so ``key(s)`` under a numpy default seeds ``numpy.random`` and returns
        ``None``, under a jax default returns a ``jax.random`` key, etc.

    Returns
    -------
    key
        ``None`` for numpy (having seeded the global ``numpy.random``), a
        ``jax.random`` key for jax, or a ``_TorchKey`` for torch.

    Notes
    -----
    numpy is stateful by design: ``key(seed)`` seeds the global generator and
    returns ``None`` (the "use the global ``numpy.random``" sentinel), which is
    exactly what keeps the numpy path byte-identical. jax/torch keys are
    explicit stateless data threaded through ``split`` and the draw functions.
    """
    name = _resolve_backend_name(backend)
    if name == "numpy":
        numpy.random.seed(seed)
        return None
    if name == "jax":
        import jax

        return jax.random.key(int(seed))
    return _TorchKey(seed)  # torch


def split(key, num=2):
    """Split ``key`` into ``num`` independent sub-keys.

    Returns a tuple of ``num`` keys. For numpy this is ``(None,) * num`` -- the
    global generator produces independent sequential draws with no substreams
    needed. For jax it is ``jax.random.split``; for torch the child seeds are
    drawn deterministically from ``key`` (so the split is reproducible).
    """
    name = _backend_of_key(key)
    if name == "numpy":
        return (None,) * num
    if name == "jax":
        import jax

        return tuple(jax.random.split(key, num))
    # torch: derive num child seeds deterministically from the parent key
    import torch

    g = _torch_generator(key)
    child = torch.randint(0, 2**62, (num,), generator=g, dtype=torch.int64)
    return tuple(_TorchKey(int(c)) for c in child)


# --- draw functions ----------------------------------------------------------
def uniform(key, shape, low=0.0, high=1.0):
    """Draw from Uniform[low, high).

    numpy path (``key is None``) is ``numpy.random.uniform(low, high, size=shape)``
    (byte-identical to the historical call). jax/torch draw a backend array that
    is a deterministic function of ``key``.
    """
    name = _backend_of_key(key)
    if name == "numpy":
        return numpy.random.uniform(low, high, size=shape)
    if name == "jax":
        from jax import random as jrandom

        return jrandom.uniform(key, shape=_tuple_shape(shape), minval=low, maxval=high)
    import torch

    g = _torch_generator(key)
    out = torch.rand(_tuple_shape(shape), generator=g, dtype=torch.get_default_dtype())
    return low + (high - low) * out


def normal(key, shape, loc=0.0, scale=1.0):
    """Draw from Normal(loc, scale).

    numpy path is ``numpy.random.normal(loc, scale, size=shape)`` (byte-
    identical). ``shape=None`` gives a scalar on numpy (matching a bare
    ``numpy.random.normal()``) and a 0-d array on jax/torch.
    """
    name = _backend_of_key(key)
    if name == "numpy":
        return numpy.random.normal(loc, scale, size=shape)
    if name == "jax":
        from jax import random as jrandom

        return loc + scale * jrandom.normal(key, shape=_tuple_shape(shape))
    import torch

    g = _torch_generator(key)
    z = torch.randn(_tuple_shape(shape), generator=g, dtype=torch.get_default_dtype())
    return loc + scale * z


def random(key, shape=None):
    """Draw from Uniform[0, 1).

    numpy path is ``numpy.random.random(size=shape)`` (byte-identical; kept as a
    distinct call from ``uniform`` so the exact historical draw sequence is
    reproduced). jax/torch return a backend array function of ``key``.
    """
    name = _backend_of_key(key)
    if name == "numpy":
        return numpy.random.random(size=shape)
    if name == "jax":
        from jax import random as jrandom

        return jrandom.uniform(key, shape=_tuple_shape(shape))
    import torch

    g = _torch_generator(key)
    return torch.rand(_tuple_shape(shape), generator=g, dtype=torch.get_default_dtype())


def randint(key, shape, low, high):
    """Draw integers from [low, high) (``high`` exclusive on every backend).

    numpy path is ``numpy.random.randint(low, high, size=shape)`` (byte-
    identical). ``shape=None`` gives a scalar on numpy.
    """
    name = _backend_of_key(key)
    if name == "numpy":
        return numpy.random.randint(low, high, size=shape)
    if name == "jax":
        from jax import random as jrandom

        return jrandom.randint(key, shape=_tuple_shape(shape), minval=low, maxval=high)
    import torch

    g = _torch_generator(key)
    return torch.randint(
        int(low), int(high), _tuple_shape(shape), generator=g, dtype=torch.int64
    )


def choice(key, a, shape=None, p=None):
    """Draw from ``a`` (an array/list, or an int meaning ``arange(a)``) with
    replacement, optionally weighted by ``p``.

    numpy path is ``numpy.random.choice(a, size=shape, p=p)`` (byte-identical).
    jax uses ``jax.random.choice``; torch maps to ``randint`` (unweighted) or
    ``multinomial`` (weighted) over the elements of ``a``.
    """
    name = _backend_of_key(key)
    if name == "numpy":
        return numpy.random.choice(a, size=shape, p=p)
    if name == "jax":
        from jax import random as jrandom

        return jrandom.choice(key, a, shape=_tuple_shape(shape), replace=True, p=p)
    import torch

    g = _torch_generator(key)
    a_t = (
        torch.arange(int(a))
        if isinstance(a, (int, numpy.integer))
        else torch.as_tensor(a)
    )
    n = a_t.shape[0]
    tshape = _tuple_shape(shape)
    count = 1
    for s in tshape:
        count *= s
    if p is None:
        idx = torch.randint(0, n, (count,), generator=g, dtype=torch.int64)
    else:
        p_t = torch.as_tensor(p, dtype=torch.get_default_dtype())
        idx = torch.multinomial(p_t, count, replacement=True, generator=g)
    return a_t[idx].reshape(tshape)


def multivariate_normal(key, mean, cov, shape=None):
    """Draw from a multivariate normal N(mean, cov).

    numpy path is ``numpy.random.multivariate_normal(mean, cov, size=shape)``
    (byte-identical). jax uses ``jax.random.multivariate_normal`` with the SVD
    method (robust to the singular covariances galpy's spray DFs use); torch
    reparameterizes an eigendecomposition of ``cov`` (also singular-safe).
    """
    name = _backend_of_key(key)
    if name == "numpy":
        return numpy.random.multivariate_normal(mean, cov, size=shape)
    if name == "jax":
        from jax import random as jrandom

        return jrandom.multivariate_normal(
            key, mean, cov, shape=_tuple_shape(shape), method="svd"
        )
    import torch

    g = _torch_generator(key)
    mean_t = torch.as_tensor(mean, dtype=torch.get_default_dtype())
    cov_t = torch.as_tensor(cov, dtype=torch.get_default_dtype())
    d = mean_t.shape[0]
    # Singular-safe factor via symmetric eigendecomposition: cov = V diag(w) V^T,
    # L = V sqrt(max(w, 0)); a standard-normal z gives mean + z @ L^T ~ N(mean, cov).
    w, V = torch.linalg.eigh(cov_t)
    L = V * torch.sqrt(torch.clamp(w, min=0.0))
    tshape = _tuple_shape(shape)
    z = torch.randn(tshape + (d,), generator=g, dtype=torch.get_default_dtype())
    return mean_t + z @ L.T
