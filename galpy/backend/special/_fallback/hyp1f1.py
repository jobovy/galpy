###############################################################################
#   Fallback for the confluent hypergeometric function 1F1(a, b; z) on the
#   real, non-positive-z domain galpy uses (z = -(R/rc)**2 <= 0). Needed on
#   BOTH jax and torch: torch.special has no hyp1f1, and jax.scipy.special.
#   hyp1f1 is wildly wrong for large negative z (rel. error ~1e10 by z~-64),
#   exactly galpy's regime. See the tripwire test in test_backend_special.py.
#
#   galpy only ever calls the b = a + 1 case (PowerSphericalwCutoff._mass:
#   1F1(1.5-alpha/2, 2.5-alpha/2, -(R/rc)**2)), which has the closed form
#       1F1(a, a+1; -X) = a * X**(-a) * Gamma(a) * P(a, X)
#   with X = -z >= 0 and P the regularized lower incomplete gamma (gammainc,
#   native on every backend) -- machine precision and robust for all X. Only
#   this case is implemented; a general b would need an accurate treatment of
#   the (1-t)^{b-a-1} endpoint (cf. the 2F1 fallback), so it raises rather than
#   silently returning a low-accuracy value.
###############################################################################
import math


def hyp1f1_fallback(xp, a, b, z):
    r"""1F1(a, b; z) for real z <= 0, restricted to b = a + 1 (galpy's case).

    a, b scalars; z a backend array/scalar.
    """
    if abs(b - (a + 1.0)) >= 1e-12:
        raise NotImplementedError(
            "galpy.backend.special.hyp1f1 fallback only implements the b=a+1 case "
            f"(galpy's only use); got (a, b)=({a}, {b})"
        )
    from .._router import gammainc

    z = xp.asarray(z) * 1.0
    X = -z  # >= 0
    # Floor away from 0 so X**(-a) stays finite; a 2-term Taylor series covers
    # the small-|z| region (and supplies the correct z=0 limit, 1F1=1).
    Xs = xp.maximum(X, xp.ones_like(X) * 1e-12)
    # ``a`` as an array (torch.special.gammainc requires both args be tensors).
    a_arr = xp.ones_like(Xs) * a
    closed = a * Xs ** (-a) * math.gamma(a) * gammainc(a_arr, Xs)
    series = 1.0 + a / (a + 1.0) * z + a / (a + 2.0) * z * z / 2.0
    return xp.where(X < 1e-6, series, closed)
