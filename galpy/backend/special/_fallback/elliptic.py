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
    a = xp.ones_like(m)
    b = xp.sqrt(1.0 - m)
    c = xp.sqrt(m)
    # sum starts at n=0 with weight 2^{-1} and c_0^2 = m
    p = 0.5
    s = p * c * c
    for _ in range(1, _AGM_NITER):
        an = 0.5 * (a + b)
        bn = xp.sqrt(a * b)
        c = 0.5 * (a - b)
        a, b = an, bn
        p = p * 2.0
        s = s + p * c * c
    K = xp.pi / (2.0 * a) if hasattr(xp, "pi") else 3.141592653589793 / (2.0 * a)
    E = K * (1.0 - s)
    return K, E


def ellipk_fallback(xp, m):
    """Complete elliptic integral of the first kind K(m) (scipy m-convention)."""
    return _agm_KE(xp, m)[0]


def ellipe_fallback(xp, m):
    """Complete elliptic integral of the second kind E(m) (scipy m-convention)."""
    return _agm_KE(xp, m)[1]
