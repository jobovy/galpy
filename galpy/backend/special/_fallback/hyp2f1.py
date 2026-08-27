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

import numpy

from ..._namespaces import (
    asarray_on_device,
    device_of,
    has_concrete_truth_value,
    is_backend_array,
)
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


def _params_are_backend(*ps):
    """True when the (a, b, c) parameters must be handled ON the backend.

    Two distinct reasons, and BOTH matter:

    * a tracer -- ``float()`` / ``math.lgamma`` raise, so the code cannot run;
    * an ordinary (untraced) jax array or torch tensor -- ``math.lgamma`` DOES
      run, via ``__float__``, and silently DETACHES that factor from the
      autograd graph. The result still has ``requires_grad=True`` and a
      ``grad_fn``, so nothing short of grad-vs-finite-difference notices: the
      measured torch ``d/da 2F1(a, 2, 3; -5)`` was -8.98e-03 against a true
      -8.56e-02, and ``dPhi/dalpha`` for TwoPowerTriaxial 0.834 against 0.548.
    """
    return any(is_backend_array(p) or not has_concrete_truth_value(p > 0.0) for p in ps)


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
    if has_concrete_truth_value(b > a):
        A, B, pref_exp = a, c - b, -a
    else:  # TRACED parameters: same choice, selected rather than branched
        swap = b > a
        A = xp.where(swap, a, c - a)
        B = xp.where(swap, c - b, b)
        pref_exp = xp.where(swap, -a, -b)
    x = z / (z - 1.0)
    term = xp.ones_like(x)
    total = xp.ones_like(x)
    for n in range(_SERIES_TERMS):
        term = term * ((A + n) * (B + n) / ((c + n) * (n + 1.0))) * x
        total = total + term
    return (1.0 - z) ** pref_exp * total


def hyp2f1_fallback(xp, a, b, c, z):
    r"""2F1(a, b; c; z) for real z < 1.

    a, b, c are scalars (galpy potential parameters); z is a backend array
    (or scalar).

    **0 < z < 1 is handled by Pfaff, entering at the top.** The routes below all
    assume z <= 0, and before this they were simply applied to positive z as
    well, which returned a silently wrong answer -- measured against scipy,
    5.1e-03 at z=0.1 rising to 6.4e-01 at z=0.95 (the Gauss series is truncated
    far too early there). galpy itself only ever passes z <= 0
    (`TwoPowerSphericalPotential` spans -15.8 .. -0.06), so nothing in-tree was
    affected, but the wrong answer was silent rather than an error.

    Pfaff maps 0 < z < 1 to z/(z-1) in (-inf, 0], i.e. exactly onto the domain
    the routes below are built for:

        2F1(a,b;c;z) = (1-z)^{-a} 2F1(a, c-b; c; z/(z-1))

    Verified over 36 (a,b,c,z) combinations with 0.05 <= z <= 0.999: worst
    relative error 9.6e-07, and <= 1.4e-14 for five of the six parameter sets.

    Three routes for z <= 0, in order of preference:

    1. the boundary-layer Euler integral, when the parameters admit it;
    2. Euler's transformation 2F1(a,b;c;z) = (1-z)^{c-a-b} 2F1(c-a,c-b;c;z),
       which leaves z alone -- so the integral's z <= 0 machinery applies
       verbatim -- and rescues the case where the only positive parameter has
       c - P < 1 (e.g. a=-3.2, b=4.4, c=5.2 becomes 8.4, 0.8, 5.2);
    3. otherwise a Gauss series, see _pfaff_series. Reached only when a and b
       are both non-positive, where no transformation lands a parameter in the
       integral's range.

    Note a Pfaff transformation is NOT usable for route 2: it maps z <= 0 to
    z/(z-1) in [0, 1), and the integral's substitutions assume z <= 0. (That is
    the same identity used for entry above, in the opposite direction -- there it
    moves positive z ONTO this domain, which is precisely what is wanted.)
    """
    z = xp.asarray(z) * 1.0
    pos = z > 0.0
    # Each branch is evaluated at a z that is valid FOR THAT BRANCH -- never the
    # raw z -- so the dead side cannot produce a nan that a gradient would carry
    # back (xp.where evaluates both sides eagerly).
    safe = -xp.ones_like(z)
    direct = _hyp2f1_nonpositive(xp, a, b, c, xp.where(pos, safe, z))
    pfaff = _hyp2f1_nonpositive(xp, a, c - b, c, xp.where(pos, z / (z - 1.0), safe))
    return xp.where(pos, (1.0 - z) ** (-a) * pfaff, direct)


def _in_regime_mask(xp, a, b, c):
    """_in_regime as an elementwise mask, for traced (a, b, c)."""
    return ((a > 0) & ((c - a) >= 1.0)) | ((b > 0) & ((c - b) >= 1.0))


def _nonpositive_traced(xp, a, b, c, z):
    """The same three routes for z <= 0, selected rather than branched.

    Reached when a, b or c is a TRACER -- differentiating a potential w.r.t. an
    exponent, say -- where `if a > 0` has no truth value. Both surviving
    formulas are evaluated and one is selected, so this costs an Euler
    quadrature plus a 512-term series per call where the concrete route pays
    for one of them; that is the price of not knowing which route applies until
    the values exist.

    The Euler evaluation is fed a SAFE (B, A, c) = (1, 0, 2) wherever no
    labeling is admissible, rather than the inadmissible one: xp.where evaluates
    both sides, and T**(B-1) with B <= 0 is inf at the t -> 0 node, which would
    NaN-poison the gradient of the series result that actually wins there.
    """
    ok_ab = _in_regime_mask(xp, a, b, c)
    ok_tr = _in_regime_mask(xp, c - a, c - b, c)
    use_tr = (~ok_ab) & ok_tr
    use_euler = ok_ab | ok_tr
    ea = xp.where(use_tr, c - a, a)
    eb = xp.where(use_tr, c - b, b)
    oka = (ea > 0) & ((c - ea) >= 1.0)
    B = xp.where(use_euler, xp.where(oka, ea, eb), 1.0)
    A = xp.where(use_euler, xp.where(oka, eb, ea), 0.0)
    cc = xp.where(use_euler, c, 2.0)
    euler = _euler_quad(xp, A, B, cc, z)
    # Euler's transformation carries a (1-z)^{c-a-b} prefactor; z <= 0 here so
    # 1-z >= 1 and the unused side is finite.
    euler = xp.where(use_tr, (1.0 - z) ** (c - a - b), 1.0) * euler
    return xp.where(use_euler, euler, _pfaff_series(xp, a, b, c, z))


def _hyp2f1_nonpositive(xp, a, b, c, z):
    """2F1(a, b; c; z) for real z <= 0 -- the three routes named above."""
    if not all(has_concrete_truth_value(p > 0.0) for p in (a, b, c)):
        return _nonpositive_traced(xp, a, b, c, z)
    if not _in_regime(a, b, c):
        if _in_regime(c - a, c - b, c):
            return (1.0 - z) ** (c - a - b) * _euler_integral(xp, c - a, c - b, c, z)
        return _pfaff_series(xp, a, b, c, z)
    return _euler_integral(xp, a, b, c, z)


def _euler_integral(xp, a, b, c, z):
    r"""2F1(a, b; c; z) for real z <= 0 via the boundary-layer Euler integral."""
    B, A = _euler_labeling(a, b, c)
    return _euler_quad(xp, A, B, c, z)


# Below this B the xi^k substitution in _euler_quad cannot regularize the
# t^{B-1} endpoint: it needs k >= ~6/B, and k is capped at 12 because X = xi^k
# underflows above that. MEASURED (a=-1.010, c=B+3, z=-0.6), shipping rule vs
# an uncapped k vs tanh-sinh:
#
#     B      k wanted   cap=12     cap=100    tanh-sinh
#     0.500  12         7.11e-15   7.11e-15   2.22e-16
#     0.200  30         1.76e-11   6.66e-15   4.44e-16
#     0.100  60         1.00e-06   5.00e-15   4.44e-16
#     0.050  120        1.04e-03   nan        8.88e-16
#     0.020  300        7.21e-02   nan        6.80e-07
#
# So raising the cap helps only to B ~ 0.1 before underflowing, while tanh-sinh
# is better at EVERY B and needs no substitution. 0.25 is where the shipping
# rule's error first exceeds ~1e-11; above it the GL path is left untouched, so
# this cannot regress the cases that already work.
_TS_B_MAX = 0.25
_TS_NODES = 200
_TS_HALFWIDTH = 7.5


def _tanh_sinh_quad(xp, A, B, c, z):
    """2F1 via the Euler integral on a tanh-sinh (double-exponential) rule.

    Same contract as _euler_quad: (A, B) already labelled, B > 0, c - B >= 1.

    t = sigmoid(2v) with v = (pi/2) sinh(u), so dt/du = pi t (1-t) cosh u. The
    endpoint singularity is cancelled ANALYTICALLY against that weight:

        t^{B-1} (1-t)^{c-B-1} (1-zt)^{-A} * pi t (1-t) cosh u
      = pi t^{B} (1-t)^{c-B} (1-zt)^{-A} cosh u

    Both exponents are then positive, so no node blows up and none has to be
    discarded -- forming f and w separately gives inf*0 = nan at small B and
    silently drops real contributions.

    Nodes are FIXED (independent of A, B, c), which is why this stays traceable
    and differentiable in the parameters; a Gauss-Jacobi rule would move its
    nodes with the exponents and need a differentiable eigenproblem per call.

    Accuracy floor: below B ~ 0.01 this degrades too (2.9e-02 at B = 0.005) --
    the shipping rule is worse there (5.2e-01), so this is still the better of
    the two, but neither is accurate and the limitation is real. Measured; the
    mechanism of that floor is NOT characterized -- it is insensitive to both
    the node count and the half-width, so it is not ordinary quadrature error.
    """
    dev = device_of(z)
    if not _params_are_backend(A, B, c):
        pref = math.exp(math.lgamma(c) - math.lgamma(B) - math.lgamma(c - B))
    else:
        from .._router import gammaln

        Bx, cx = (asarray_on_device(xp, v, dev) for v in (B, c))
        pref = xp.exp(gammaln(cx) - gammaln(Bx) - gammaln(cx - Bx))
    h = 2.0 * _TS_HALFWIDTH / _TS_NODES
    u = asarray_on_device(
        xp, numpy.linspace(-_TS_HALFWIDTH, _TS_HALFWIDTH, _TS_NODES + 1), dev
    )
    v = (numpy.pi / 2.0) * xp.sinh(u)
    t = 1.0 / (1.0 + xp.exp(-2.0 * v))
    omt = 1.0 / (1.0 + xp.exp(2.0 * v))
    fw = (
        numpy.pi * t**B * omt ** (c - B) * (1.0 - z[..., None] * t) ** (-A) * xp.cosh(u)
    )
    return pref * h * xp.sum(fw, axis=-1)


def _euler_quad(xp, A, B, c, z):
    """The quadrature itself, on an ALREADY-LABELLED (A, B).

    Split out from _euler_integral so the traced-parameter route below can
    choose (A, B) with xp.where instead of a Python branch and still reuse this
    verbatim.

    Small B goes to tanh-sinh: the xi^k substitution below needs k >= ~6/B and k
    is capped at 12 (X = xi^k underflows above that), so for B < _TS_B_MAX the
    t^{B-1} endpoint is simply not regularized and plain GL loses badly -- 7.2e-02
    at B = 0.02, which is a real galpy request (constantbetaHernquistdf beta=-1.5
    asks for a=-1.010, b=0.020, c=3.010). See _tanh_sinh_quad for the table.
    """
    # Concreteness, not a data guard: with traced parameters B has no truth
    # value, and every galpy caller passes CONCRETE potential/DF parameters --
    # only z is ever traced. A traced B keeps the historical GL path rather than
    # silently changing rule mid-trace.
    if has_concrete_truth_value(B) and bool(B < _TS_B_MAX):
        return _tanh_sinh_quad(xp, A, B, c, z)
    w = -z  # >= 0
    q = c - B  # exponent of (1-t) is q-1
    dev = device_of(z)
    # X = xi^k regularizes the t^{B-1} endpoint: after the boundary-layer map the
    # integrand carries X^{B-1} near X=0, which is only algebraically integrable
    # for non-integer B. Raise it to xi^{kB-1} with kB >= ~6 so plain GL is
    # spectrally accurate (k=1 already suffices once B-1 is a smooth high power).
    # Capped at 12 (covers B >= 0.5, galpy's range) so X=xi^k cannot underflow.
    if not _params_are_backend(A, B, c):
        k = min(12.0, max(1.0, float(math.ceil(6.0 / B))))
        pref = math.exp(math.lgamma(c) - math.lgamma(B) - math.lgamma(q))
    else:
        # Backend parameters: the same expressions, evaluated ON the backend, so
        # they neither concretize a tracer nor detach a tensor. ceil is a step,
        # so its zero gradient is the correct one -- k only shapes the
        # substitution; the integral's value does not depend on it.
        from .._router import gammaln

        # every operand on the backend and on z's device: the router's gammaln
        # dispatches on its ARGUMENT, and torch.special.gammaln rejects a plain
        # float outright.
        Bx, cx, qx = (asarray_on_device(xp, v, dev) for v in (B, c, q))
        k = xp.clip(xp.ceil(6.0 / Bx), 1.0, 12.0)
        pref = xp.exp(gammaln(cx) - gammaln(Bx) - gammaln(qx))

    # node/weight tables stay float64 (precision is the point; the router
    # exit-casts) but must live on the input's device (CUDA support)
    nodes, weights = gauss_legendre_01(_NODES)
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
    # A*B is a*b: {A, B} is just a relabelling of {a, b}.
    val_series = 1.0 + (A * B / c) * z  # 2F1 = 1 + (ab/c) z + O(z^2)
    return xp.where(tiny, val_series, val_int)
