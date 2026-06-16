###############################################################################
#   Fallbacks for the complete elliptic integrals K(m) and E(m) on backends
#   without a native implementation (currently both jax and torch).
#
#   Parameter convention matches scipy.special: the argument is the parameter
#   m = k**2 (NOT the modulus k), valid for m < 1; m -> 1 is the genuine
#   logarithmic singularity of K.
###############################################################################

# Fixed iteration count: the AGM converges quadratically, so this is ample for
# float64 across m in [0, 1); extra iterations are harmless (c_n -> 0).
_AGM_NITER = 16


def _agm_KE(xp, m):
    r"""Both K(m) and E(m) from a single arithmetic-geometric-mean iteration.

    K(m) = pi / (2 * AGM(1, sqrt(1-m))),
    E(m) = K(m) * (1 - sum_{n>=0} 2^{n-1} c_n^2),
    with a_0=1, b_0=sqrt(1-m), c_0=sqrt(m), and
    a_{n+1}=(a_n+b_n)/2, b_{n+1}=sqrt(a_n b_n), c_{n+1}=(a_n-b_n)/2.

    Pure arithmetic + sqrt, so it differentiates cleanly under jax/torch. At
    m -> 1 (b_0 -> 0) the AGM -> 0 and K -> +inf, the correct singularity.
    """
    m = xp.asarray(m) * 1.0
    one = xp.ones_like(m)
    # At m==1 (b0=0) the AGM reaches 0 only asymptotically; run it on a safe
    # argument and substitute the exact limits K=+inf, E=1 afterwards (AD-safe).
    on_edge = m == 1.0
    ms = xp.where(on_edge, 0.5 * one, m)
    a = one
    b = xp.sqrt(1.0 - ms)  # real for all m < 1, incl. m < 0
    # sum starts at n=0 with weight 2^{-1} and c_0^2 = ms exactly (sqrt(ms) would
    # be nan for m < 0, NaN-poisoning E; K depends only on b = sqrt(1-ms))
    p = 0.5
    s = p * ms
    for _ in range(1, _AGM_NITER):
        an = 0.5 * (a + b)
        bn = xp.sqrt(a * b)
        c = 0.5 * (a - b)
        a, b = an, bn
        p = p * 2.0
        s = s + p * c * c
    pi = xp.pi if hasattr(xp, "pi") else 3.141592653589793
    K = pi / (2.0 * a)
    E = K * (1.0 - s)
    K = xp.where(on_edge, xp.full_like(m, float("inf")), K)  # K(1) = +inf
    E = xp.where(on_edge, one, E)  # E(1) = 1
    return K, E


def ellipk_fallback(xp, m):
    """Complete elliptic integral of the first kind K(m) (scipy m-convention)."""
    return _agm_KE(xp, m)[0]


def ellipe_fallback(xp, m):
    """Complete elliptic integral of the second kind E(m) (scipy m-convention)."""
    return _agm_KE(xp, m)[1]
