###############################################################################
#   galpy.backend.special._router: dispatch each special function to the
#   backend's native implementation, falling back to a pure-backend version.
###############################################################################
import numpy

from .._resolver import get_namespace

# Per-backend functions whose NATIVE implementation is simply absent, so the
# router must use the pure-backend fallback. A capability test
# (test_fallback_table_matches_installed_backends) asserts this matches reality
# (hasattr on the backend's special module), so entries are removed as backends
# add the native version. (numpy always has the full scipy.special.)
_NATIVE_MISSING = {
    "jax": frozenset(("ellipk", "ellipe", "k0", "k1", "kn")),
    # torch.special lacks all of these. (It does have modified_bessel_k0/k1, but
    # they are NOT differentiable -- no autograd backward -- and there is no kn,
    # so the k0/k1/kn fallbacks are used; the router sees no torch.special.k0.)
    "torch": frozenset(
        ("gamma", "ellipk", "ellipe", "hyp2f1", "hyp1f1", "k0", "k1", "kn")
    ),
}

# Functions whose native implementation EXISTS but is too inaccurate on galpy's
# argument domain to use -- we force the fallback anyway. Currently jax's
# hyp2f1/hyp1f1: both are catastrophically wrong for z < -1 (return +-inf with
# NaN gradients), which is exactly galpy's regime (z = -r/a, -(R/rc)**2, ...).
# A tripwire test documents the breakage so these move to native if jax fixes it.
_NATIVE_UNRELIABLE = {
    "jax": frozenset(("hyp2f1", "hyp1f1")),
    "torch": frozenset(),
}

# The router routes a function to its fallback iff it is in EITHER set.
_NEEDS_FALLBACK = {
    name: _NATIVE_MISSING.get(name, frozenset())
    | _NATIVE_UNRELIABLE.get(name, frozenset())
    for name in ("jax", "torch")
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


def _dispatch(fnname, args, fallback, ns_args=None):
    """Route ``fnname(*args)`` to native or fallback by the arrays' namespace.

    ns_args (default: all of args) selects which arguments determine the
    backend; pass only the array arguments when some of ``args`` are plain
    scalars (e.g. the (a, b, c) parameters of hyp2f1) so they don't confuse
    the resolver.
    """
    xp = get_namespace(*(args if ns_args is None else ns_args))
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


# --- Tier 2: pure-backend fallbacks (jax/torch); high force-unblock impact ----
def hyp2f1(a, b, c, z):
    # Gauss 2F1; only the array arg z carries the backend namespace.
    from ._fallback.hyp2f1 import hyp2f1_fallback

    return _dispatch("hyp2f1", (a, b, c, z), hyp2f1_fallback, ns_args=(z,))


def hyp1f1(a, b, z):
    # Confluent 1F1 (Kummer); only z carries the backend namespace.
    from ._fallback.hyp1f1 import hyp1f1_fallback

    return _dispatch("hyp1f1", (a, b, z), hyp1f1_fallback, ns_args=(z,))


def ellipk(m):
    # Complete elliptic integral K(m) (scipy parameter-m convention).
    from ._fallback.elliptic import ellipk_fallback

    return _dispatch("ellipk", (m,), ellipk_fallback)


def ellipe(m):
    # Complete elliptic integral E(m) (scipy parameter-m convention).
    from ._fallback.elliptic import ellipe_fallback

    return _dispatch("ellipe", (m,), ellipe_fallback)


# --- Tier 3: modified Bessel functions of the second kind (disk force paths) --
def k0(x):
    from ._fallback.bessel_k import k0_fallback

    return _dispatch("k0", (x,), k0_fallback)


def k1(x):
    from ._fallback.bessel_k import k1_fallback

    return _dispatch("k1", (x,), k1_fallback)


def kn(n, x):
    # Integer-order modified Bessel K_n; only the array arg x carries the namespace.
    from ._fallback.bessel_k import kn_fallback

    return _dispatch("kn", (n, x), kn_fallback, ns_args=(x,))


# --- Tier 4: associated Legendre P_l^m (SCF / MultipoleExpansion) -------------
def _scipy_assoc_legendre(L, M, x, deriv):
    """numpy path: scipy.special.assoc_legendre_p_all reshaped to (...,L,M),
    byte-identical to scipy (the convention used by util.special.compute_legendre)."""
    import scipy.special as sp

    arr = numpy.asarray(
        sp.assoc_legendre_p_all(
            L - 1, M - 1, numpy.asarray(x, dtype=float), branch_cut=2, diff_n=deriv
        )
    )  # (deriv+1, L, 2M-1, *x.shape)  -- m=0..M-1 are the first M columns
    out = numpy.moveaxis(arr[:, :, :M], (1, 2), (-2, -1))  # (deriv+1, *x.shape, L, M)
    return out[0] if deriv == 0 else tuple(out[i] for i in range(deriv + 1))


def assoc_legendre(L, M, x, deriv=0):
    """P_l^m(x) for 0<=l<L, 0<=m<M (Condon-Shortley phase), shape x.shape+(L,M).

    deriv: 0 -> P; 1 -> (P, dP/dx); 2 -> (P, dP/dx, d2P/dx2). numpy routes to
    scipy (byte-identical); jax/torch use the pure-backend Bonnet recurrence.
    """
    name, _ = _backend_special(get_namespace(x))
    if name == "numpy":
        return _scipy_assoc_legendre(L, M, x, deriv)
    from ._fallback.assoc_legendre import assoc_legendre as _fb

    return _fb(get_namespace(x), L, M, x, deriv)


def gegenbauer(N, alpha, x):
    """Gegenbauer polynomials C_n^alpha(x) for 0<=n<N, shape x.shape+(N,).

    N static int, alpha scalar, x a backend array. Uses the three-term
    recurrence on every backend (galpy's SCF radial basis never used a scipy
    Gegenbauer, so there is no native to prefer)."""
    from ._fallback.gegenbauer import gegenbauer as _fb

    return _fb(get_namespace(x), N, alpha, x)


def xlogy(x, y):
    # x * log(y), with the scipy/native convention 0 * log(0) = 0.
    from ._fallback.xlogy import xlogy_fallback

    return _dispatch("xlogy", (x, y), xlogy_fallback)


# Silence "imported but unused" for numpy (kept for potential defensive use).
_ = numpy
