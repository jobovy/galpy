###############################################################################
#   Fallbacks for the regularized incomplete gamma functions P(a,x) and
#   Q(a,x) = 1 - P(a,x). Needed on torch, which has ``torch.special.gammainc``
#   but (a) no derivative w.r.t. the ORDER a -- ``NotImplementedError: the
#   derivative for 'igamma: input' is not implemented`` -- and (b) a discrete
#   algorithm switch at a ~ 20 that costs ~6 digits above it (|dQ| ~ 5e-10 for
#   a >= 21 vs ~1e-16 below). Both are cured at once by evaluating in backend
#   ops: autodiff then supplies d/da and d/dx alike. jax's native versions are
#   accurate but iterate per element (~100x slower on CPU), so jax uses this
#   vectorized form too (forward and d/da; d/dx in closed form); numpy uses
#   scipy.
#
#   Standard split, each branch converging where it is used:
#     - x <  a+1: the ascending series  P = prefix * sum_n x^n / (a...(a+n))
#     - x >= a+1: the modified-Lentz continued fraction for Q.
#
#   The delicate part is the shared prefix  x^a e^-x / Gamma(a),  NOT the
#   series. Writing it as exp(-x + a ln x - lnGamma(a)) sums three terms that
#   are each O(a ln a) and cancel to O(1) -- at a = x = 80 that is
#   -80.0000 + 350.5621 - 269.2911 = 1.2710. Each term carries |term|*eps, so
#   exp() returns ~1e-13 relative error however many series terms are taken;
#   more iterations cannot fix it. Instead use the Stirling form, in which the
#   large parts cancel ANALYTICALLY:
#
#       x^a e^-x / Gamma(a) = e^{-a*(lam - 1 - ln lam)} * sqrt(a/2pi) * e^{-w(a)}
#
#   with lam = x/a and w the Stirling remainder. Now the exponent is O(1) by
#   construction. That form needs a >~ 14 for w's asymptotic series; below it
#   the naive exponent is itself accurate (lnGamma(a) is O(1) there, so there is
#   no large cancellation to lose), so the two are used exactly where the other
#   fails. The crossover was measured, not guessed: worst error over an
#   a-in-[0.5,200] grid is 5.4e-07 at a split of 2, 7.3e-15 at 14, 1.1e-14 at 20.
#
#   Verified against mpmath at 60 digits over a in [0.5,200], x/a in [0.3,4]:
#   worst 7.3e-15 (P) and 2.4e-14 (Q), against scipy's own 1.0e-13 and 1.1e-13
#   on the same grid -- i.e. at least as accurate as the numpy path everywhere,
#   and up to 20x better in the large-a mid-domain.
#
#   Iteration counts are the measured minima at that accuracy (N_SERIES=120:
#   80 gives only 7.6e-08; N_CF=50: 30 gives 2.7e-11, and MORE than 50 is
#   slightly worse from accumulation).
###############################################################################
import numpy
from scipy.special import zeta as _zeta

from ..._namespaces import _backend_dtype, name_of_namespace
from .._router import gammaln

_A_STIRLING = 14.0  # Stirling prefix at/above this order, naive exponent below
_N_SERIES = 120  # ascending-series terms (x < a+1)
_N_CF = 50  # modified-Lentz iterations (x >= a+1)
_FPMIN = 1e-300  # Lentz zero-denominator floor
_TINY_U = 1e-4  # |lam-1| below which u - log1p(u) cancels; use its series
# Small order AND argument (a < _A_SMALL, 0 < x < _X_SMALL): Q directly from its
# own series (cephes igamc_series), not as 1 - P (cancels when P ~ 1) nor from
# the CF (unconverged near x ~ a+1 ~ 1): those cost up to 8e-12 at a=0.005,
# x=1.01, vs scipy's 3e-15.
_A_SMALL = 0.5
_X_SMALL = 1.8  # measured crossover: series and CF both ~3e-15..8e-15 there
_N_QSERIES = 40  # sum (-x)^n / (n! (a+n)): x^n/n! < 1e-37 at n=40, x=1.8
# On a GPU the loops are staged ROLLED (lax.fori_loop, partially unrolled): the
# unrolled series/CF put ~600 ops into the graph per call, and a program that
# calls this at every quadrature node (the Staeckel transformation of
# MWPotential2014) then takes minutes to compile for the GPU. XLA fuses less
# across a rolled loop, which costs at large N on the CPU, where the loops stay
# unrolled. The two forms perform the same operations in the same order.
# A GPU pays per loop iteration, so the unroll factor trades compile time for
# run time; on that Staeckel transformation on an H100, 10 runs 6x faster than
# jax's native gammainc for 2.5x its compile time (5: 2x slower; 30: 4x longer
# compile). 10 divides every trip count (120, 50, 40): jax unrolls a
# remainder in full.
_GPU_UNROLL = 10
# lgamma(1+a) = -gamma*a + sum_{k>=2} (-1)^k zeta(k)/k a^k, |a| <= 1/2
_LGAM1P_COEFS = tuple((-1.0) ** k * float(_zeta(k, 1.0)) / k for k in range(2, 62))


def _lam_minus_1_minus_log(xp, lam):
    """``lam - 1 - ln(lam)`` for all lam > 0, with the absolute error (what the
    Stirling exponent multiplies by a) at its eps*|ln lam| floor.

    Not ``u - log1p(u)`` with u = lam - 1: forming 1 + u far below lam = 1/2
    discards lam's digits (P(15, 1e-12) came out 1e-2 off), and XLA's CPU
    log1p is itself ~9e-15 off near u = -0.4 (7e-13 in P(150, 90)). Near
    lam = 1 the difference cancels totally, so |u| < _TINY_U takes the
    Maclaurin series u^2/2 - u^3/3 + ... (its first omitted term is u^6/6).
    """
    u = lam - 1.0
    tiny = xp.abs(u) < _TINY_U
    ut = xp.where(tiny, u, 0.0)  # dead branch -> 0, series is then exact
    series = ut * ut * (1.0 / 2 - ut * (1.0 / 3 - ut * (1.0 / 4 - ut / 5)))
    return xp.where(tiny, series, u - xp.log(lam))


def _stirling_remainder(xp, a):
    """lnGamma(a) - [(a-1/2)ln a - a + ln(2pi)/2], via its asymptotic series."""
    ia = 1.0 / a
    ia2 = ia * ia
    return ia * (
        1.0 / 12
        - ia2 * (1.0 / 360 - ia2 * (1.0 / 1260 - ia2 * (1.0 / 1680 - ia2 / 1188)))
    )


def _prefix(xp, a, x):
    """x^a e^-x / Gamma(a), without the O(a ln a) cancellation. See module docs."""
    big = a >= _A_STIRLING
    ab = xp.where(big, a, _A_STIRLING)  # dead branch -> a valid Stirling order
    lam = x / ab
    stirling = (
        xp.exp(-ab * _lam_minus_1_minus_log(xp, lam))
        * xp.sqrt(ab / (2.0 * numpy.pi))
        * xp.exp(-_stirling_remainder(xp, ab))
    )
    asm = xp.where(big, 1.0, a)  # dead branch -> a=1, gammaln(1)=0, no overflow
    naive = xp.exp(-x + asm * xp.log(x) - gammaln(asm))
    return xp.where(big, stirling, naive)


def _series_P(xp, a, x, pref, rolled=False):
    """Ascending series for P(a,x), valid (and used) for x < a+1."""

    def step(_, carry):
        ap, term, total = carry
        ap = ap + 1.0
        term = term * (x / ap)
        return ap, term, total + term

    carry = (a, 1.0 / a, 1.0 / a)
    if rolled:
        import jax

        carry = jax.lax.fori_loop(0, _N_SERIES, step, carry, unroll=_GPU_UNROLL)
    else:
        for i in range(_N_SERIES):
            carry = step(i, carry)
    return carry[2] * pref


def _cf_Q(xp, a, x, pref, rolled=False):
    """Modified-Lentz continued fraction for Q(a,x), valid for x >= a+1."""

    def step(i, carry):
        b, c, d, h = carry
        an = -i * (i - a)
        b = b + 2.0
        d = an * d + b
        d = xp.where(xp.abs(d) < _FPMIN, _FPMIN, d)
        c = b + an / xp.where(xp.abs(c) < _FPMIN, _FPMIN, c)
        d = 1.0 / d
        return b, c, d, h * (d * c)

    b = x + 1.0 - a
    c = xp.full_like(b, 1.0 / _FPMIN)
    d = 1.0 / xp.where(xp.abs(b) < _FPMIN, _FPMIN, b)
    carry = (b, c, d, d)
    if rolled:
        import jax

        carry = jax.lax.fori_loop(
            1,
            _N_CF + 1,
            lambda i, cr: step(xp.astype(i, b.dtype), cr),
            carry,
            unroll=_GPU_UNROLL,
        )
    else:
        for i in range(1, _N_CF + 1):
            carry = step(float(i), carry)
    return pref * carry[3]


def _lgam1p_small(xp, a):
    """lgamma(1+a) to full RELATIVE accuracy for |a| <= 1/2 (gammaln(1+a)
    loses it as a -> 0, where the value ~ -0.577 a)."""
    s = 0.0
    for c in reversed(_LGAM1P_COEFS):
        s = s * a + c
    return a * (numpy.euler_gamma * -1.0 + a * s)


def _small_Q(xp, a, x, rolled=False):
    """Q(a,x) for small a and x (cephes igamc_series): no 1 - P cancellation."""

    def step(n, carry):
        fac, total = carry
        fac = fac * (-x / n)
        return fac, total + fac / (a + n)

    carry = (xp.ones_like(x), xp.zeros_like(x))
    if rolled:
        import jax

        carry = jax.lax.fori_loop(
            1,
            _N_QSERIES + 1,
            lambda n, cr: step(xp.astype(n, x.dtype), cr),
            carry,
            unroll=_GPU_UNROLL,
        )
    else:
        for n in range(1, _N_QSERIES + 1):
            carry = step(float(n), carry)
    total = carry[1]
    e = a * xp.log(x) - _lgam1p_small(xp, a)
    # x^a / Gamma(a) = a exp(e), as Gamma(a) = Gamma(1+a) / a
    return -xp.expm1(e) - a * xp.exp(e) * total


def _both(xp, a, x, rolled=False):
    """Return (P, Q), each computed by whichever branch is valid there;
    ``rolled``: the loops as lax.fori_loop (jax on a GPU, see _GPU_UNROLL)."""
    f64 = _backend_dtype(xp, numpy.float64)
    a = xp.astype(xp.asarray(a), f64)
    x = xp.astype(xp.asarray(x), f64)
    a, x = xp.broadcast_arrays(a, x)
    use_series = x < a + 1.0
    # x = inf is a real argument here -- the potential at r = inf and the total
    # mass both reach it (the exp1 fallback documents the same hazard). Lentz
    # would give b = inf -> d = 1/inf = 0 and then h *= d*c = 0*inf = NaN, so
    # infinity is clamped out of the CF and the exact limits are applied after.
    at_inf = xp.isinf(x)
    # Clamp each branch's argument into its own convergent region wherever the
    # OTHER branch is selected: the series diverges for x >> a and the CF's
    # b = x+1-a passes through zero for x < a, so an unclamped dead branch
    # would overflow or NaN-poison the reverse-mode gradient.
    x_ser = xp.where(use_series, x, a)  # x=a is inside the series' region
    x_cf = xp.where(
        xp.logical_or(use_series, at_inf), a + 1.0, x
    )  # a+1 is the CF's boundary
    p = _series_P(xp, a, x_ser, _prefix(xp, a, x_ser), rolled)
    q = _cf_Q(xp, a, x_cf, _prefix(xp, a, x_cf), rolled)
    p_out = xp.where(use_series, p, 1.0 - q)
    q_out = xp.where(use_series, 1.0 - p, q)
    small = (a < _A_SMALL) & (x > 0.0) & (x < _X_SMALL)
    # dead branch -> a point inside the region (log(0) / the lgamma series' radius)
    qs = _small_Q(xp, xp.where(small, a, 0.25), xp.where(small, x, 1.0), rolled)
    q_out = xp.where(small, qs, q_out)
    p_out = xp.where(small & ~use_series, 1.0 - qs, p_out)
    return (
        xp.where(at_inf, xp.ones_like(p_out), p_out),  # P(a, inf) = 1
        xp.where(at_inf, xp.zeros_like(q_out), q_out),  # Q(a, inf) = 0
    )


def _torch_autograd(upper):
    """Build a torch.autograd.Function: native forward, our backward.

    The series/CF above is ~485x slower than ``torch.special.gammainc`` on a
    scalar (measured), and ``PowerSphericalPotentialwCutoff`` -- a component of
    ``MWPotential2014`` -- calls this on nearly every evaluation, with scalars.
    So the loop must not run on the forward pass. It does not have to:

    * ``dP/dx = x^(a-1) e^-x / Gamma(a) = prefix(a, x) / x`` in CLOSED FORM, and
      ``prefix`` is exactly the (cheap, loop-free) helper above;
    * ``dP/da`` has no closed form and does need the series/CF -- but only when
      the order actually requires grad, which no hot path does.

    Forward is therefore the native call, and the loop is paid only by callers
    that differentiate with respect to the order.
    """
    import torch

    native = torch.special.gammaincc if upper else torch.special.gammainc
    sign = -1.0 if upper else 1.0

    class _IncGamma(torch.autograd.Function):
        # functorch needs both of these: generate_vmap_rule so vmap can batch
        # the op, and the split forward/setup_context form (the modern API): the combined
        # forward(ctx, ...) form raises under functorch transforms
        # ("must override the setup_context staticmethod"), and the spherical
        # DFs reach this through torch.func vmap/grad.
        generate_vmap_rule = True

        @staticmethod
        def forward(a, x):
            return native(a, x)

        @staticmethod
        def setup_context(ctx, inputs, output):
            ctx.save_for_backward(*inputs)

        @staticmethod
        def backward(ctx, grad_out):
            a, x = ctx.saved_tensors
            need_a, need_x = ctx.needs_input_grad[:2]
            grad_a = grad_x = None
            if need_x:
                # closed form, no series/CF (x = 0 limit taken explicitly)
                import galpy.backend as _gb

                xp = _gb.get_namespace(x)
                grad_x = grad_out * _dx_closed_form(xp, a, x, sign)
            if need_a:
                # only here does the loop run
                import galpy.backend as _gb

                xp = _gb.get_namespace(x)
                with torch.enable_grad():
                    ad = a.detach().requires_grad_(True)
                    out = _both(xp, ad, x.detach())[1 if upper else 0]
                    (grad_a,) = torch.autograd.grad(out, ad, grad_out)
            return grad_a, grad_x

    return _IncGamma


def _dx_closed_form(xp, a, x, sign):
    """sign * dP/dx = sign * x^(a-1) e^-x / Gamma(a) = prefix(a,x)/x, with the
    x = 0 limit taken explicitly (prefix/x is 0/0 there):
    a < 1 -> +inf, a = 1 -> 1, a > 1 -> 0; and 0 at x = inf (exp(-inf + inf)).
    The limits are built in the result dtype: an INTEGER order (Einasto) would
    make torch.full_like(a, inf) raise."""
    pos = x > 0
    fin = xp.isfinite(x)
    x_safe = xp.where(pos & fin, x, xp.ones_like(x))  # keep 0, inf out
    dens = _prefix(xp, a, x_safe) / x_safe
    at_zero = xp.where(
        a < 1.0,
        xp.full_like(dens, float("inf")),
        xp.where(a == 1.0, xp.ones_like(dens), xp.zeros_like(dens)),
    )
    return sign * xp.where(pos, xp.where(fin, dens, xp.zeros_like(dens)), at_zero)


_JAX_FNS = {}
# Below this many elements jax uses its native kernel even when staged ON THE
# CPU: the series/CF unrolls ~270 iterations into the graph (~1 s more per
# compile), which only pays off for large arrays there (native ~2 us/element,
# the series/CF ~20 ns). On a GPU the native kernel's per-element while_loop is
# slow at ANY size (it set a 0.7 s floor on the compiled Staeckel transformation
# of MWPotential2014 below 4096 points), so a staged call on an accelerator
# always takes the series/CF.
_JAX_MIN_SIZE = 4096


def _jax_on_accelerator():
    """Whether jax stages programs for a GPU/TPU: a tracer carries no device,
    so the default backend stands in for the device of a staged call."""
    import jax

    return jax.default_backend() != "cpu"


def _jax_incgamma(upper):
    """jax P or Q. STAGED arrays (jit / vmap) take the vectorized series/CF
    when large (>= _JAX_MIN_SIZE) or when jax runs on an accelerator -- jax's
    native gammainc iterates per element in a while_loop, ~100x slower on CPU
    for large arrays (measured) and slow on a GPU at every size. Otherwise
    (eager values, incl. eager grad, or small arrays on the CPU) the native
    kernel: one dispatch, small graph. Either way a closed-form d/dx (with the
    x = 0 / inf limits native lacks) and d/da by forward mode, paid only when
    the order is differentiated."""
    import jax
    import jax.numpy as jnp
    import jax.scipy.special as jss
    from jax.custom_derivatives import SymbolicZero

    idx = 1 if upper else 0
    sign = -1.0 if upper else 1.0
    native = jss.gammaincc if upper else jss.gammainc
    from ..._namespaces import has_concrete_truth_value

    def _is_concrete(v):
        # jax.core.is_concrete (deprecated in jax 0.10) through public API: a
        # non-tracer, or a tracer with a concrete value (eager grad's JVPTracer)
        return not isinstance(v, jax.core.Tracer) or has_concrete_truth_value(
            jnp.ravel(v)[0]
        )

    def value(a, x):
        if _is_concrete(a) and _is_concrete(x):
            return native(a, x)
        if _jax_on_accelerator():
            return _both(jnp, a, x, rolled=True)[idx]
        if x.size < _JAX_MIN_SIZE:
            return native(a, x)
        return _both(jnp, a, x)[idx]

    @jax.custom_jvp
    def f(a, x):
        return value(a, x)

    def f_jvp(primals, tangents):
        a, x = primals
        ta, tx = tangents
        out = f(a, x)
        tout = jnp.zeros_like(out)
        if not isinstance(tx, SymbolicZero):
            tout = tout + _dx_closed_form(jnp, a, x, sign) * tx
        if not isinstance(ta, SymbolicZero):
            tout = tout + jax.jvp(lambda aa: value(aa, x), (a,), (ta,))[1]
        return out, tout

    f.defjvp(f_jvp, symbolic_zeros=True)
    return f


_TORCH_FNS = {}


def _dispatch_one(xp, a, x, upper):
    # numpy uses scipy and never comes here
    if name_of_namespace(xp) == "jax":
        import jax.numpy as jnp

        if upper not in _JAX_FNS:
            _JAX_FNS[upper] = _jax_incgamma(upper)
        f64 = _backend_dtype(xp, numpy.float64)
        a, x = jnp.broadcast_arrays(jnp.asarray(a, f64), jnp.asarray(x, f64))
        return _JAX_FNS[upper](a, x)
    import torch

    # a Python/numpy order lands on the argument tensor's device (torch's default
    # device can differ: CUDA x with a float order raised in torch.igamma)
    ref = x if torch.is_tensor(x) else a
    dev = ref.device if torch.is_tensor(ref) else None
    a = torch.as_tensor(a, device=dev)
    x = torch.as_tensor(x, device=dev)
    native = torch.special.gammaincc if upper else torch.special.gammainc
    if not (a.requires_grad or x.requires_grad):
        # Nothing to differentiate: hand straight to the native kernel and skip
        # the autograd.Function entirely. This is the hot path -- MWPotential2014's
        # PowerSphericalPotentialwCutoff lands here on every evaluation -- and the
        # wrapper alone costs ~2x on a scalar.
        return native(a, x)
    if upper not in _TORCH_FNS:
        _TORCH_FNS[upper] = _torch_autograd(upper)
    a, x = torch.broadcast_tensors(a, x)
    return _TORCH_FNS[upper].apply(a, x)


def gammainc_fallback(xp, a, x):
    """Regularized lower incomplete gamma P(a,x), differentiable in a AND x."""
    return _dispatch_one(xp, a, x, False)


def gammaincc_fallback(xp, a, x):
    """Regularized upper incomplete gamma Q(a,x), differentiable in a AND x."""
    return _dispatch_one(xp, a, x, True)
