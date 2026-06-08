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
#   native on every backend) -- machine precision and robust for all X. A
#   general b > a > 0 path via the Euler integral (same boundary-layer
#   quadrature as the 2F1 fallback, with e^{zt} in place of (1-zt)^{-A}) is
#   kept for completeness.
###############################################################################
import math

from ._quadrature import gauss_legendre_01

_NODES = 96


def hyp1f1_fallback(xp, a, b, z):
    r"""1F1(a, b; z) for real z <= 0. a, b scalars; z a backend array/scalar."""
    if abs(b - (a + 1.0)) < 1e-12:
        return _hyp1f1_aplus1(xp, a, z)
    return _hyp1f1_euler(xp, a, b, z)


def _hyp1f1_aplus1(xp, a, z):
    """Exact b=a+1 case via the lower incomplete gamma (gammainc)."""
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


def _hyp1f1_euler(xp, a, b, z):
    r"""General b>a>0 via Euler's integral with an exponential boundary layer.

    1F1(a,b;z) = Gamma(b)/(Gamma(a)Gamma(b-a)) int_0^1 e^{z t} t^{a-1}(1-t)^{b-a-1} dt
    for z <= 0; the map e^{z t} = 1 - x(1-e^{z}) sends the boundary layer near
    t=0 to a uniform x-grid, and x = xi^k regularizes the t^{a-1} endpoint.
    """
    z = xp.asarray(z) * 1.0
    X = -z
    q = b - a
    k = min(12.0, max(1.0, float(math.ceil(6.0 / a))))  # regularize t^{a-1} endpoint
    pref = math.exp(math.lgamma(b) - math.lgamma(a) - math.lgamma(q))
    nodes, weights = gauss_legendre_01(_NODES)
    xg = xp.asarray(nodes)
    wg = xp.asarray(weights)
    xx = xg**k
    dxx = k * xg ** (k - 1.0)

    Xs = xp.maximum(X, xp.ones_like(X) * 1e-10)
    em = -xp.expm1(-Xs)[..., None]  # 1 - e^{-X} in (0, 1]
    Xsb = Xs[..., None]
    ez = 1.0 - xx * em  # = e^{-X t}
    t = -xp.log1p(-xx * em) / Xsb
    dt = (em / Xsb) / ez
    integ = ez * t ** (a - 1.0) * (1.0 - t) ** (q - 1.0) * dt * dxx
    return pref * xp.sum(integ * wg, axis=-1)
