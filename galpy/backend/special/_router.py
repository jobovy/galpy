###############################################################################
#   galpy.backend.special._router: dispatch each special function to the
#   backend's native implementation, falling back to a pure-backend version.
###############################################################################
import numpy

from .._resolver import get_namespace

# Per-backend list of functions whose NATIVE implementation is missing, so the
# router must use the pure-backend fallback. Kept tiny and explicit; a capability
# test (tests/test_backend_special.py) asserts this matches reality and that each
# fallback agrees with scipy, so entries are removed as backends add the native
# version. (numpy always has the full scipy.special, so it never needs a fallback.)
_NEEDS_FALLBACK = {
    "jax": frozenset(),
    "torch": frozenset(("gamma",)),  # torch.special has gammaln but not gamma
}


def _backend_special(xp):
    """Return (backend_name, native_special_module) for a resolved namespace xp.

    numpy -> scipy.special, jax.numpy -> jax.scipy.special,
    array_api_compat.torch -> torch.special.
    """
    name = getattr(xp, "__name__", "")
    if name in ("numpy", "np"):
        import scipy.special as _sp

        return "numpy", _sp
    if name in ("jax.numpy", "jax"):
        import jax.scipy.special as _sp

        return "jax", _sp
    if "torch" in name:
        import torch.special as _sp

        return "torch", _sp
    # Unknown namespace: treat as numpy/scipy (defensive).
    import scipy.special as _sp  # pragma: no cover

    return "numpy", _sp  # pragma: no cover


def _dispatch(fnname, args, fallback):
    """Route ``fnname(*args)`` to native or fallback based on the args' namespace."""
    xp = get_namespace(*args)
    name, sp = _backend_special(xp)
    if fnname not in _NEEDS_FALLBACK.get(name, frozenset()) and hasattr(sp, fnname):
        return getattr(sp, fnname)(*args)
    return fallback(xp, *args)


def _no_fallback(fnname):
    def _fb(xp, *args):  # pragma: no cover - reached only if a backend regresses
        raise NotImplementedError(
            f"galpy.backend.special.{fnname} has no fallback and the active "
            f"backend lacks a native implementation"
        )

    return _fb


# --- Tier 1: native on every backend except where noted -----------------------
def gammaln(x):
    return _dispatch("gammaln", (x,), _no_fallback("gammaln"))


def gamma(x):
    from ._fallback.gamma import gamma_fallback

    return _dispatch("gamma", (x,), gamma_fallback)


def gammainc(a, x):
    # Regularized lower incomplete gamma P(a, x).
    return _dispatch("gammainc", (a, x), _no_fallback("gammainc"))


def gammaincc(a, x):
    # Regularized upper incomplete gamma Q(a, x) = 1 - P(a, x).
    return _dispatch("gammaincc", (a, x), _no_fallback("gammaincc"))


def erf(x):
    return _dispatch("erf", (x,), _no_fallback("erf"))


def erfc(x):
    return _dispatch("erfc", (x,), _no_fallback("erfc"))


def i0(x):
    return _dispatch("i0", (x,), _no_fallback("i0"))


def i1(x):
    return _dispatch("i1", (x,), _no_fallback("i1"))


def xlogy(x, y):
    # x * log(y), with the scipy/native convention 0 * log(0) = 0. Native
    # everywhere, but the trivial backend-agnostic form is also a valid fallback.
    def _fb(xp, x, y):
        x = xp.asarray(x)
        y = xp.asarray(y)
        safe_y = xp.where(x == 0, xp.ones_like(y), y)
        return xp.where(x == 0, xp.zeros_like(x * 1.0), x * xp.log(safe_y))

    return _dispatch("xlogy", (x, y), _fb)


# Silence "imported but unused" for numpy (kept for potential defensive use).
_ = numpy
