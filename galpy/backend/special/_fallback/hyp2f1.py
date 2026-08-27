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
#   where {A, B} = {a, b} is chosen so B > 0 and c - B > 0 -- just what the Beta
#   integral needs to converge. The quadrature is a tanh-sinh (double-
#   exponential) rule that cancels BOTH endpoint singularities analytically
#   against its own weight, so neither exponent has to be regularized by a
#   substitution and neither endpoint has to be preferred over the other.
#   Pure arithmetic + log1p/pow, so it differentiates under jax and torch.
###############################################################################
import math

import numpy

from ..._namespaces import (
    asarray_on_device,
    device_of,
    has_concrete_truth_value,
    is_backend_array,
)

# 128 nodes: ~1e-10 or better vs scipy at realistic radii (|z| = r/a <~ 50);
# accuracy degrades smoothly to ~1e-6 at the extreme |z| ~ 500 (r/a ~ 500, far
# beyond any realistic galactic radius) for awkward exponent combinations.


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
    return (a > 0 and (c - a) > 0.0) or (b > 0 and (c - b) > 0.0)


def _euler_labeling(a, b, c):
    """Pick (B, A) with {A,B}={a,b}, B > 0 and c - B > 0.

    Both conditions are just what the Beta integral needs to converge. The old
    requirement was the stronger ``c - B >= 1``, which existed ONLY to keep
    (1-t)^{c-B-1} non-singular for the fixed-order Gauss-Legendre rule this
    module used to run; the tanh-sinh rule cancels that endpoint analytically
    and does not care. Measured across c - B in (0, 1] -- the whole block the
    old bound excluded -- the rule is accurate to 2.3e-15.

    Dropping it also removes a float64 knife-edge: at (a=5.0, b=0.001, c=1.001)
    the old test asked ``1.001 - 0.001 >= 1.0``, which is False by one ulp, so
    the Euler route was refused and a much less accurate one ran instead
    (measured 4.6e-03 there).

    The raise still names BOTH conditions, because the binding one is often the
    sign: with a and b both non-positive there is no admissible B at all.
    """
    if a > 0 and (c - a) > 0.0:
        return a, b
    if b > 0 and (c - b) > 0.0:
        return b, a
    raise NotImplementedError(
        "hyp2f1 Euler integral needs some P in {a, b} with P > 0 and c - P > 0; "
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
    return ((a > 0) & ((c - a) > 0.0)) | ((b > 0) & ((c - b) > 0.0))


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
    # Same symmetry as the concrete path: _in_regime_mask(c-a, c-b, c) is
    # _in_regime_mask(a, b, c), so the transform selector (~ok) & ok was
    # identically False and its (1-z)^{c-a-b} factor identically 1 -- dead
    # computation in every traced call. Both are gone.
    use_euler = _in_regime_mask(xp, a, b, c)
    ea, eb = a, b
    # Must match _in_regime_mask / _euler_labeling EXACTLY: c - P > 0, not the
    # old >= 1. When it did not, a first label with 0 < c-a < 1 was rejected
    # here while the concrete path accepted it, so the traced route fell to the
    # SECOND label even when that one was non-positive -- T**(B-1) with B <= 0
    # is inf at the t->0 node, and the traced result came back inf or nan while
    # eager was correct.
    oka = (ea > 0) & ((c - ea) > 0.0)
    B = xp.where(use_euler, xp.where(oka, ea, eb), 1.0)
    A = xp.where(use_euler, xp.where(oka, eb, ea), 0.0)
    cc = xp.where(use_euler, c, 2.0)
    euler = _euler_quad(xp, A, B, cc, z)
    return xp.where(use_euler, euler, _pfaff_series(xp, a, b, c, z))


def _hyp2f1_nonpositive(xp, a, b, c, z):
    """2F1(a, b; c; z) for real z <= 0 -- the three routes named above."""
    if not all(has_concrete_truth_value(p > 0.0) for p in (a, b, c)):
        return _nonpositive_traced(xp, a, b, c, z)
    # No Euler-TRANSFORM branch: under `c - P > 0` it is unreachable. The regime
    # test is symmetric under (a, b) -> (c-a, c-b), because
    #   _in_regime(c-a, c-b, c) = (c-a > 0 and a > 0) or (c-b > 0 and b > 0)
    # is literally _in_regime(a, b, c) with the conjuncts swapped. So the
    # transform can never rescue a parameter set the direct route rejects.
    # Under the OLD bound `c - P >= 1` the two were NOT symmetric and the branch
    # was live (2031 hits over a -3..5 grid; 0 after the relaxation), which is
    # why it existed. The relaxation SUBSUMED it.
    if not _in_regime(a, b, c):
        return _pfaff_series(xp, a, b, c, z)
    return _euler_integral(xp, a, b, c, z)


def _euler_integral(xp, a, b, c, z):
    r"""2F1(a, b; c; z) for real z <= 0 via the boundary-layer Euler integral."""
    B, A = _euler_labeling(a, b, c)
    return _euler_quad(xp, A, B, c, z)


# ONE rule, one grid. The integrand falls off as exp(-B pi sinh|u|) at t=0 and
# exp(-(c-B) pi sinh u) at t=1, and the rule is SYMMETRIC in u, so a half-width
# that resolves one endpoint resolves the other: 12.45 covers EITHER exponent
# down to 1e-3 (exp(-40) ~ 4e-18, below eps against an O(1) integrand). That
# symmetry is what makes the relaxed labelling safe -- under c - B > 0 the t=1
# exponent can now be the small one, which it never could when the bound was
# c - B >= 1. Measured with c - B down to 1e-4: 2.3e-15. The step 0.075 sets the
# accuracy and the node count follows from the two.
#
# Sizing the grid PER CALL from B was measured and dropped: over 720 cases
# spanning B = 0.001..5.0 the fixed grid is as accurate as the adaptive one
# (worst 2.67e-15 vs 2.89e-15, nothing above 1e-13), and being independent of
# the parameters is what lets a TRACED B use this rule at all -- sizing from B
# needs bool(B < ...), which has no answer under jit.
_TS_HALFWIDTH = 12.45
_TS_STEP = 0.075
_TS_NODES = int(math.ceil(2.0 * _TS_HALFWIDTH / _TS_STEP))


def _log_sigmoid(xp, x):
    """``log(1/(1+exp(-x)))``, i.e. ``-softplus(-x)``, without overflow."""
    # min(x, 0) - log1p(exp(-|x|)); spelled with where because torch.maximum /
    # torch.minimum reject a Python scalar for their second operand.
    return xp.where(x < 0.0, x, xp.zeros_like(x)) - xp.log1p(xp.exp(-xp.abs(x)))


def _euler_quad(xp, A, B, c, z):
    """2F1 via the Euler integral on a tanh-sinh (double-exponential) rule.

    Operates on an ALREADY-LABELLED (A, B): B > 0 and c - B > 0, guaranteed by
    _euler_labeling. Split out from _euler_integral so the traced-parameter
    route can choose (A, B) with xp.where instead of a Python branch and still
    reuse this verbatim.

    t = sigmoid(2v) with v = (pi/2) sinh(u), so dt/du = pi t (1-t) cosh u. The
    endpoint singularity is cancelled ANALYTICALLY against that weight:

        t^{B-1} (1-t)^{c-B-1} (1-zt)^{-A} * pi t (1-t) cosh u
      = pi t^{B} (1-t)^{c-B} (1-zt)^{-A} cosh u

    Both exponents are then positive, so no node blows up and none has to be
    discarded -- forming f and w separately gives inf*0 = nan at small B and
    silently drops real contributions.

    Nodes are FIXED -- independent of A, B, c AND of z -- which is what keeps
    this traceable and differentiable in every argument.

    This REPLACED a fixed-order Gauss-Legendre rule that regularized the
    t^{B-1} endpoint with an X = xi^k substitution. That rule needed k >= ~6/B
    and capped k at 12 (X = xi^k underflows above it), so it was accurate only
    for B >= ~0.5 and catastrophic below. Measured against mpmath over 768
    (A, B, c, z) cases with B ABOVE the old 0.25 routing threshold -- i.e. the
    regime the GL rule was kept for -- its worst relative error was 1.5e-04
    against 2.9e-15 here, with NO case where tanh-sinh was worse. The threshold
    was only protecting the less accurate rule, so both it and the substitution
    are gone.

    Small |z| needs no special case: (1 - z t)^{-A} is simply 1 at z = 0, where
    the GL rule's 1/|z| substitution was singular and needed its own series
    branch.
    """
    dev = device_of(z)
    if not _params_are_backend(A, B, c):
        pref = math.exp(math.lgamma(c) - math.lgamma(B) - math.lgamma(c - B))
    else:
        from .._router import gammaln

        # every operand on the backend and on z's device: the router's gammaln
        # dispatches on its ARGUMENT, and torch.special.gammaln rejects a plain
        # float outright.
        Bx, cx = (asarray_on_device(xp, v, dev) for v in (B, c))
        pref = xp.exp(gammaln(cx) - gammaln(Bx) - gammaln(cx - Bx))
    h = 2.0 * _TS_HALFWIDTH / _TS_NODES
    u = asarray_on_device(
        xp, numpy.linspace(-_TS_HALFWIDTH, _TS_HALFWIDTH, _TS_NODES + 1), dev
    )
    v = (numpy.pi / 2.0) * xp.sinh(u)
    # t**B and (1-t)**(c-B) are formed in LOG space. Forming t first loses the
    # t=0 tail outright: t**B is still 1e-3 where t ~ 1e-300, so at B = 0.01 the
    # integrand matters until t ~ exp(-3684) -- thousands of orders below what a
    # double can hold. t underflows to 0, t**B with it, and the tail is silently
    # discarded. log t = -softplus(-2v) stays finite there (it is ~ 2v), so the
    # product B*log(t) is exactly what it should be.
    log_t = _log_sigmoid(xp, 2.0 * v)
    log_omt = _log_sigmoid(xp, -2.0 * v)
    t = xp.exp(log_t)  # only for (1-zt)^-A, which is 1 wherever t underflows
    fw = (
        numpy.pi
        * xp.exp(B * log_t + (c - B) * log_omt)
        * (1.0 - z[..., None] * t) ** (-A)
        * xp.cosh(u)
    )
    return pref * h * xp.sum(fw, axis=-1)
