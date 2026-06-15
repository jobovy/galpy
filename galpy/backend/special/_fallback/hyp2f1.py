###############################################################################
#   Fallback for the Gauss hypergeometric function 2F1(a, b; c; z) on the
#   real, non-positive-z domain galpy uses (z = -r/a, -a/r, -(z/R)**2, ... so
#   z <= 0).  Needed on BOTH jax and torch:
#     - torch.special has no hyp2f1 at all;
#     - jax.scipy.special.hyp2f1 is catastrophically wrong for z < -1 (it
#       returns +-inf and NaN gradients), which is exactly galpy's regime
#       (r > a). See tests/test_backend_special.py::test_jax_native_hyp2f1_is
#       _unreliable for the tripwire that documents this.
#
#   Method: Euler's integral representation, valid for all z not in [1, inf):
#       2F1(a,b;c;z) = Gamma(c)/(Gamma(B) Gamma(c-B))
#                      * int_0^1 t^{B-1} (1-t)^{c-B-1} (1 - z t)^{-A} dt
#   where {A, B} = {a, b} is chosen so B > 0 and c - B > 0 (preferring
#   c - B >= 1 so the t=1 endpoint is non-singular). For z <= 0 the factor
#   (1 - z t) = (1 + |z| t) >= 1 is smooth, but for large |z| it forms a thin
#   boundary layer near t=0; two substitutions resolve it for fixed-order
#   Gauss-Legendre quadrature:
#     1. t = ((1+|z|)^X - 1)/|z| maps the boundary layer to a uniform X-grid;
#     2. X = xi^k (k = ceil(1/B) when B < 1) regularizes the t^{B-1} endpoint
#        singularity so plain Gauss-Legendre converges.
#   Pure arithmetic + log1p/expm1/pow, so it differentiates under jax and torch.
###############################################################################
import math

from ..._namespaces import asarray_on_device, device_of
from ._quadrature import gauss_legendre_01

# 128 nodes: ~1e-10 or better vs scipy at realistic radii (|z| = r/a <~ 50);
# accuracy degrades smoothly to ~1e-6 at the extreme |z| ~ 500 (r/a ~ 500, far
# beyond any realistic galactic radius) for awkward exponent combinations.
_NODES = 128


def _euler_labeling(a, b, c):
    """Pick (B, A) with {A,B}={a,b}, B>0 and c-B >= 1.

    Requiring c-B >= 1 keeps the (1-t)^{c-B-1} endpoint non-singular so the
    fixed-order quadrature stays accurate; galpy's 2F1 calls always satisfy
    this (c-a is 1 or 2). A configuration with c-max(a,b) < 1 would need the
    t=1 endpoint regularized too, so it raises rather than return a
    low-accuracy value.
    """
    if a > 0 and (c - a) >= 1.0:
        return a, b
    if b > 0 and (c - b) >= 1.0:
        return b, a
    raise NotImplementedError(
        f"hyp2f1 fallback requires c - max(a, b) >= 1 (galpy's regime); "
        f"got (a={a}, b={b}, c={c})"
    )


def hyp2f1_fallback(xp, a, b, c, z):
    r"""2F1(a, b; c; z) for real z <= 0 via the boundary-layer Euler integral.

    a, b, c are scalars (galpy potential parameters); z is a backend array
    (or scalar) with z <= 0.
    """
    z = xp.asarray(z) * 1.0
    w = -z  # >= 0
    B, A = _euler_labeling(a, b, c)
    q = c - B  # exponent of (1-t) is q-1
    # X = xi^k regularizes the t^{B-1} endpoint: after the boundary-layer map the
    # integrand carries X^{B-1} near X=0, which is only algebraically integrable
    # for non-integer B. Raise it to xi^{kB-1} with kB >= ~6 so plain GL is
    # spectrally accurate (k=1 already suffices once B-1 is a smooth high power).
    # Capped at 12 (covers B >= 0.5, galpy's range) so X=xi^k cannot underflow.
    k = min(12.0, max(1.0, float(math.ceil(6.0 / B))))
    pref = math.exp(math.lgamma(c) - math.lgamma(B) - math.lgamma(q))

    # node/weight tables stay float64 (precision is the point; the router
    # exit-casts) but must live on the input's device (CUDA support)
    nodes, weights = gauss_legendre_01(_NODES)
    dev = device_of(z)
    xg = asarray_on_device(xp, nodes, dev)
    wg = asarray_on_device(xp, weights, dev)
    X = xg**k
    dX = k * xg ** (k - 1.0)

    # Floor |z| away from exactly zero so the 1/|z| substitution is finite (and
    # X*log1p(|z|) cannot underflow to 0, which would make T**(B-1) blow up for
    # B<1). At |z|=0 this evaluates 2F1 at z=-1e-10, i.e. 1 to ~1e-10; the
    # subgradient blocks gradient only at the measure-zero origin point.
    w_safe = xp.maximum(w, xp.ones_like(w) * 1e-10)
    L = xp.log1p(w_safe)[..., None]  # (..., 1)
    wb = w_safe[..., None]
    XL = X * L  # (..., N)
    T = xp.expm1(XL) / wb
    dt = xp.exp(XL) * L / wb
    integ = T ** (B - 1.0) * (1.0 - T) ** (q - 1.0) * (1.0 + wb * T) ** (-A) * dt * dX
    return pref * xp.sum(integ * wg, axis=-1)
