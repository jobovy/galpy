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


# Terms for the series route below. Its argument is z/(z-1), so accuracy is set
# by how close that is to 1, i.e. by |z| -- not much by the parameters (even the
# worst conditioning, a = b, is exact at |z| <~ 5). Measured worst case over the
# both-non-positive triples galpy's DFs request plus a = b:
#     |z|      1       5       20      50      200     500
#     rel err  0       4e-14   9e-13   2e-6    7e-3    5e-2
# 512 is chosen so that at |z| = 50 this matches what the quadrature route above
# actually delivers there (also ~2e-6), i.e. up to |z| = 50 the series route is
# no less accurate than the one galpy already ships. Past that it falls off
# faster, because its convergence is algebraic in the term count rather than
# spectral -- reaching the quadrature's ~5e-6 at |z| = 500 would need >2048
# terms. galpy's own anisotropic DFs call this branch at |z| < 1 (measured), so
# the exact range covers real use with a wide margin.
_SERIES_TERMS = 512


def _in_regime(a, b, c):
    """True when the Euler integral below can be used for these parameters."""
    return (a > 0 and (c - a) >= 1.0) or (b > 0 and (c - b) >= 1.0)


def _euler_labeling(a, b, c):
    """Pick (B, A) with {A,B}={a,b}, B>0 and c-B >= 1.

    Requiring c-B >= 1 keeps the (1-t)^{c-B-1} endpoint non-singular so the
    fixed-order quadrature stays accurate. Callers check _in_regime first; the
    raise is a guard against reaching the integral with parameters it cannot
    represent, and names BOTH conditions, because the binding one is often the
    sign: with a and b both non-positive there is no admissible B at all, even
    though c - max(a, b) >= 1 may well hold.
    """
    if a > 0 and (c - a) >= 1.0:
        return a, b
    if b > 0 and (c - b) >= 1.0:
        return b, a
    raise NotImplementedError(
        "hyp2f1 Euler integral needs some P in {a, b} with P > 0 and c - P >= 1; "
        f"got (a={a}, b={b}, c={c})"
    )


def _pfaff_series(xp, a, b, c, z):
    r"""2F1(a, b; c; z) for z <= 0 by Gauss series in the Pfaff variable.

    Used when no transformation puts a parameter in the Euler integral's range,
    which happens exactly when a and b are both non-positive. The two Pfaff
    transformations are

        2F1(a,b;c;z) = (1-z)^{-a} 2F1(a, c-b; c; x),  C-A-B = b-a
                     = (1-z)^{-b} 2F1(c-a, b; c; x),  C-A-B = a-b

    with x = z/(z-1) in [0, 1) for z <= 0. Taking whichever puts max(a, b) in
    the c-. slot gives C-A-B = |a-b| > 0, so the series converges even in the
    x -> 1 (z -> -inf) limit. |a-b| is invariant under Euler's transformation,
    so this is the best conditioning available -- there is no relabeling that
    converges faster.
    """
    if b > a:
        A, B, pref_exp = a, c - b, -a
    else:
        A, B, pref_exp = c - a, b, -b
    x = z / (z - 1.0)
    term = xp.ones_like(x)
    total = xp.ones_like(x)
    for n in range(_SERIES_TERMS):
        term = term * ((A + n) * (B + n) / ((c + n) * (n + 1.0))) * x
        total = total + term
    return (1.0 - z) ** pref_exp * total


def hyp2f1_fallback(xp, a, b, c, z):
    r"""2F1(a, b; c; z) for real z <= 0.

    a, b, c are scalars (galpy potential parameters); z is a backend array
    (or scalar) with z <= 0.

    Three routes, in order of preference:

    1. the boundary-layer Euler integral, when the parameters admit it;
    2. Euler's transformation 2F1(a,b;c;z) = (1-z)^{c-a-b} 2F1(c-a,c-b;c;z),
       which leaves z alone -- so the integral's z <= 0 machinery applies
       verbatim -- and rescues the case where the only positive parameter has
       c - P < 1 (e.g. a=-3.2, b=4.4, c=5.2 becomes 8.4, 0.8, 5.2);
    3. otherwise a Gauss series, see _pfaff_series. Reached only when a and b
       are both non-positive, where no transformation lands a parameter in the
       integral's range.

    Note a Pfaff transformation is NOT usable for route 2: it maps z <= 0 to
    z/(z-1) in [0, 1), and the integral's substitutions assume z <= 0.
    """
    z = xp.asarray(z) * 1.0
    if not _in_regime(a, b, c):
        if _in_regime(c - a, c - b, c):
            return (1.0 - z) ** (c - a - b) * _euler_integral(xp, c - a, c - b, c, z)
        return _pfaff_series(xp, a, b, c, z)
    return _euler_integral(xp, a, b, c, z)


def _euler_integral(xp, a, b, c, z):
    r"""2F1(a, b; c; z) for real z <= 0 via the boundary-layer Euler integral."""
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

    # Tiny |z|: the 1/|z| substitution is singular, so feed a safe non-zero w
    # into the integral (double-where: the dead branch can't NaN-poison AD via
    # 1/w / log1p(0)) and return the exact Maclaurin limit instead -- which also
    # gives the correct gradient a*b/c at z->0 (a plain maximum-floor flattens it).
    tiny = w < 1e-10
    w_for_int = xp.where(tiny, xp.ones_like(w), w)
    L = xp.log1p(w_for_int)[..., None]  # (..., 1)
    wb = w_for_int[..., None]
    XL = X * L  # (..., N)
    # T = expm1(XL)/|z|, but X = xi^k puts the first node at XL ~ 1e-49, where
    # inductor's FUSED expm1 (its standalone one is exact) degenerates to
    # exp(x)-1 and returns 0; T**(B-1) with B < 1 is then inf and poisons the
    # quadrature at EVERY z. Factor out expm1(u)/u -> 1 and series it below the
    # crossover instead, so nothing tiny ever reaches expm1.
    small = XL < 1e-8
    u_safe = xp.where(small, xp.ones_like(XL), XL)
    ratio = xp.where(small, 1.0 + XL / 2.0 + XL**2.0 / 6.0, xp.expm1(u_safe) / u_safe)
    T = XL * ratio / wb
    dt = xp.exp(XL) * L / wb
    integ = T ** (B - 1.0) * (1.0 - T) ** (q - 1.0) * (1.0 + wb * T) ** (-A) * dt * dX
    val_int = pref * xp.sum(integ * wg, axis=-1)
    val_series = 1.0 + (a * b / c) * z  # 2F1 = 1 + (ab/c) z + O(z^2)
    return xp.where(tiny, val_series, val_int)
