###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleStaeckel
#
#             Use Binney (2012; MNRAS 426, 1324)'s Staeckel approximation for
#             calculating the actions
#
#      methods:
#             __call__: returns (jr,lz,jz)
#
###############################################################################
import warnings
from functools import lru_cache

import numpy

from ..backend import (
    asarray_on_device,
    bucket_size,
    coerce_coords,
    concretely_true,
    device_of,
    get_namespace,
    host_eval,
    is_backend_array,
    name_of_namespace,
    promote_scalars,
    to_host,
    use,
)
from ..backend._coerce import lazy_cond
from ..backend._namespaces import (
    as_numpy,
    has_concrete_truth_value,
    pad_edge,
    stop_gradient,
    under_jax_trace,
)
from ..backend.optimize import (
    bisect_root,
    fixed_point_implicit_step,
    iterate_bracket,
    iterate_ladder,
    iterate_to_fixed_point,
)
from ..backend.optimize import brentq as _backend_brentq
from ..potential import (
    CompositePotential,
    DiskSCFPotential,
    MWPotential,
    SCFPotential,
    epifreq,
    evaluateR2derivs,
    evaluateRzderivs,
    evaluatez2derivs,
    omegac,
    verticalfreq,
)
from ..potential.Potential import (
    Potential,
    PotentialError,
    _accepts_node_array,
    _check_c,
    _check_potential_list_and_deprecate,
    _evaluatePotentials,
    _evaluateRforces,
    _evaluatezforces,
    _isNonAxi,
    _pot_data_namespace,
)
from ..util import (  # for prolate confocal transforms
    conversion,
    coords,
    galpyWarning,
)
from ..util.conversion import physical_conversion, potential_physical_input
from . import actionAngleStaeckel_c
from .actionAngle import UnboundError, _c_cannot_carry, actionAngle
from .actionAngleStaeckel_c import _ext_loaded as ext_loaded

# the fixed point of the Staeckel model of a torus of the inverse through a
# point: tolerance on the step of the reference curve, and cap on the cycles
_MODEL_TOL = 1e-9
_MODEL_MAXITER = 10


def _fixed_point_error(x0, x1, x2):
    """The distance left to the fixed point of a linearly converging
    iteration after its iterates x0, x1, x2: that of x2 itself (the
    geometric series of the remaining steps), which bounds that of Aitken's
    extrapolation of the three"""
    xp = get_namespace(x0, x1, x2)
    d1, d2 = xp.abs(x1 - x0), xp.abs(x2 - x1)
    rate = xp.clip(d2 / xp.where(d1 > 0.0, d1, 1.0), 0.0, 0.9)
    return xp.where(xp.isfinite(d2), d2 * rate / (1.0 - rate), 0.0)


# the same fixed point by bracketing, for the points that the iteration does
# not converge for: the grid of sinh u0, the halvings, and the largest step
# of the rule across the closed bracket that is not a jump
_MODEL_SCAN = numpy.arcsinh(numpy.geomspace(1e-3, 1e2, 21))
_MODEL_NBISECT = 35
_MODEL_JUMP = 1e-6
# (on jax/torch, the halvings in sections of this many equal parts at once)
_MODEL_NSECT = 32
# (and the number of grid points over the range of a rule for the focal length)
_MODEL_NDSCAN = 17
# the range of the third integral, relative to the energy, below which an
# orbit is circular for a rule for the focal length
_MODEL_CIRC = 1e-10


def _bracket_fixed_point(rule, n, scan=_MODEL_SCAN, undefined=False):
    """The fixed points u0 = rule(idx, u0) of n points by bracketing: the
    change of the reference curve under the rule decreases through zero at
    the fixed point, which is bracketed on a grid of u0 and bisected. A
    bracket that closes on a jump of the rule (where the inner turning
    point passes from one lobe of p_u^2 to another) holds no fixed point:
    nan, as for a point without a bracket (no torus of the rule passes
    through it). The same for a rule for the focal length, on the grid scan
    of focal lengths; with undefined, a bracket may end where the rule is
    not defined (nan), and holds no fixed point if it closes there"""
    out = numpy.full(n, numpy.nan)
    idx = numpy.repeat(numpy.arange(n), len(scan))
    u = numpy.tile(scan, n)
    g = (rule(idx, u) - u).reshape((n, len(scan)))
    bracket = (g[:, :-1] > 0.0) & (~(g[:, 1:] > 0.0) if undefined else (g[:, 1:] < 0.0))
    idx = numpy.flatnonzero(numpy.any(bracket, axis=1))
    if len(idx) == 0:
        return out
    k = numpy.argmax(bracket[idx], axis=1)
    lo, hi = scan[k], scan[k + 1]
    glo, ghi = g[idx, k], g[idx, k + 1]
    for _ in range(_MODEL_NBISECT):
        mid = 0.5 * (lo + hi)
        gmid = rule(idx, mid) - mid
        up = gmid > 0.0
        lo, glo = numpy.where(up, mid, lo), numpy.where(up, gmid, glo)
        hi, ghi = numpy.where(up, hi, mid), numpy.where(up, ghi, gmid)
    out[idx] = numpy.where(glo - ghi < _MODEL_JUMP, 0.5 * (lo + hi), numpy.nan)
    return out


def _bk_bracket_fixed_point(xp, g, need, x, scan=_MODEL_SCAN, undefined=False):
    """_bracket_fixed_point on jax/torch at a fixed shape, for the points of
    the (n,) mask need: g(u, *x) is the change under the rule at the points
    x (a tuple of (k,) arrays: the n points, or each of them repeated), each
    evaluated without derivatives (the caller's implicit-function step
    supplies them). The bracket on the grid scan is narrowed by sections of
    _MODEL_NSECT equal parts at once, to the width of _MODEL_NBISECT
    halvings (each section's points are those of the halvings, evaluated
    together: one evaluation of the rule per section, not per halving, which
    is what an eager backend pays for). Returns the fixed points, nan where
    no bracket holds one; only when a point is in need (lazy_cond; eagerly on
    torch, on the points in need alone, padded to a bucket_size: each point's
    bracket is its own; not on eager jax, which compiles every op anew for
    the shapes that this makes)"""
    n = need.shape[0]
    if "jax" not in name_of_namespace(xp) and has_concrete_truth_value(xp.any(need)):
        sel = numpy.flatnonzero(numpy.asarray(as_numpy(need)))
        m = bucket_size(len(sel))
        if len(sel) == 0:
            return numpy.nan + 0.0 * x[0]
        if m < n:
            dev = device_of(x[0])
            take = asarray_on_device(
                xp, numpy.concatenate([sel, numpy.full(m - len(sel), sel[0])]), dev
            )
            sub = _bk_bracket_fixed_point(
                xp,
                g,
                take == take,
                tuple(xp.take(stop_gradient(t), take, axis=0) for t in x),
                scan,
                undefined,
            )
            pos = numpy.clip(numpy.cumsum(numpy.asarray(as_numpy(need))) - 1, 0, None)
            return xp.where(
                need, xp.take(sub, asarray_on_device(xp, pos, dev), axis=0), numpy.nan
            )

    def rep(k):  # each point repeated k times, point-major
        return [
            xp.reshape(
                stop_gradient(t)[:, None]
                + xp.zeros((1, k), dtype=t.dtype, device=device_of(t)),
                (-1,),
            )
            for t in x
        ]

    def evaluate(u):  # g at u of shape (n, k)
        k = u.shape[1]
        return xp.reshape(stop_gradient(g(xp.reshape(u, (-1,)), *rep(k))), (n, k))

    def first_crossing(nodes, vals):
        # the first pair of neighbouring nodes with the change above zero at
        # the first and not above it at the second (nan: undefined, or not
        # a bracket of a fixed point)
        cross = (vals[:, :-1] > 0.0) & (
            ~(vals[:, 1:] > 0.0) if undefined else (vals[:, 1:] < 0.0)
        )
        has = xp.any(cross, axis=1)
        k = xp.argmax(xp.astype(cross, vals.dtype), axis=1)
        sel = (
            asarray_on_device(xp, numpy.arange(nodes.shape[1] - 1), device_of(k))[
                None, :
            ]
            == k[:, None]
        )

        def pick(t):
            return xp.sum(xp.where(sel, t, 0.0), axis=1)

        return (
            has,
            pick(nodes[:, :-1]),
            pick(nodes[:, 1:]),
            pick(vals[:, :-1]),
            pick(vals[:, 1:]),
        )

    def run():
        x0 = stop_gradient(x[0])
        dev = device_of(x0)
        sc = asarray_on_device(xp, numpy.asarray(scan, dtype="float"), dev)
        nodes = sc[None, :] + 0.0 * x0[:, None]
        has, lo, hi, glo, ghi = first_crossing(nodes, evaluate(nodes))
        lo = xp.where(has, lo, sc[0])
        hi = xp.where(has, hi, sc[1])
        frac = asarray_on_device(xp, numpy.arange(1, _MODEL_NSECT) / _MODEL_NSECT, dev)[
            None, :
        ]

        def step(st):
            lo, hi, glo, ghi = st[0], st[1], st[2], st[3]
            inner = lo[:, None] + (hi - lo)[:, None] * frac
            nodes = xp.concat([lo[:, None], inner, hi[:, None]], axis=1)
            vals = xp.concat([glo[:, None], evaluate(inner), ghi[:, None]], axis=1)
            ok, *new = first_crossing(nodes, vals)
            return xp.stack([xp.where(ok, t, o) for t, o in zip(new, st)])

        nsect = -(-_MODEL_NBISECT // int(numpy.log2(_MODEL_NSECT)))

        def sections():
            st = iterate_bracket(step, xp.stack([lo, hi, glo, ghi]), nsect)
            return xp.where(
                has & (st[2] - st[3] < _MODEL_JUMP), 0.5 * (st[0] + st[1]), numpy.nan
            )

        # (no sections when no point in need has a bracket on the grid)
        return lazy_cond(xp, xp.any(has & need), sections, lambda: numpy.nan + 0.0 * x0)

    return lazy_cond(xp, xp.any(need), run, lambda: numpy.nan + 0.0 * x[0])


def _focal_length_of(pot):
    """The focal length that a potential supplies through its _delta
    attribute, or that the components of a composite potential all supply,
    if they supply the same one; None otherwise"""
    if isinstance(pot, CompositePotential):
        deltas = [getattr(p, "_delta", None) for p in pot]
        if any(d is None for d in deltas) or any(
            numpy.fabs(d - deltas[0]) > 1e-10 * (1.0 + numpy.fabs(deltas[0]))
            for d in deltas
        ):
            return None
        return deltas[0]
    return getattr(pot, "_delta", None)


def _coerce_delta_arraylike(delta):
    """Coerce a plain Python sequence delta (allowed by the public API for
    individual-delta inputs) to an ndarray: the backend-agnostic coords
    transforms resolve their namespace from the data, and plain sequences
    are not backend-resolvable. Scalars/arrays pass through untouched."""
    return numpy.array(delta) if isinstance(delta, (list, tuple)) else delta


def _nanmedian(xp, a):
    """NaN-skipping median matching numpy.median over the non-NaN values.

    numpy/jax: ``xp.nanmedian`` is byte-identical to the original
    ``numpy.median(a[~isnan])``. array-api-compat torch's median/nanmedian
    return the lower of the two central order statistics for even counts
    (not their average), so torch is handled via ``quantile(.,0.5)`` on the
    NaN-filtered values to match numpy's convention (eager-only; the median
    is a non-differentiable selection)."""
    if name_of_namespace(xp) == "torch":
        return xp.quantile(a[~xp.isnan(a)], 0.5)
    return xp.nanmedian(a)


class _PointwisePotential(Potential):
    """A potential with a component whose methods take scalar coordinates only
    (e.g., DoubleExponentialDiskPotential), evaluated point by point for the
    vectorized core (numpy only)"""

    def __init__(self, pot):
        Potential.__init__(self, amp=1.0)
        self._pot = pot

    def _pointwise(self, func, R, z):
        R, z = numpy.broadcast_arrays(R, z)
        return numpy.reshape(
            numpy.array(
                [func(self._pot, r, zz) for r, zz in zip(R.ravel(), z.ravel())]
            ),
            R.shape,
        )

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        return self._pointwise(_evaluatePotentials, R, z)

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        return self._pointwise(_evaluateRforces, R, z)

    def _zforce(self, R, z, phi=0.0, t=0.0):
        return self._pointwise(_evaluatezforces, R, z)


def _staeckel_pot(xp, pot):
    """pot, or, on numpy, its point-by-point evaluation if a component's
    potential or forces take scalar coordinates only"""
    comps = list(pot) if isinstance(pot, CompositePotential) else [pot]
    if xp is not numpy or all(
        _accepts_node_array(p, "_evaluate", "_Rforce", "_zforce") for p in comps
    ):
        return pot
    return _PointwisePotential(pot)


# ----------------------------------------------------------------------------
# Vectorised, backend-agnostic Staeckel action core (numpy / jax / torch). One
# unified path that replaced the former per-object scipy loop:
# elementwise setup + turning points via the shared backend.optimize.bisect_root
# (fixed-iteration expanding bracket) + the action/frequency/angle integrals via
# the chi-anomaly quadrature of _staeckel_chi_quads (q = qmin + D sin^2(chi/2)
# makes sqrt(S) analytic at the turning points) on C's mesh (_staeckel_nchi).
# v0=pi/2 for the u (J_R) integral, u0 for the v (J_z) integral.


def _staeckel_setup(xp, R, vR, vT, z, vz, pot, delta, u0=None):
    """Vectorised setup quantities (the former per-object __init__ did these one
    orbit at a time); with u0, for the Staeckel model of (delta, u0): its
    reference curve and the energy in it, Phi = [U(u) - V(v)] / (sinh^2 u +
    sin^2 v) in place of the potential itself"""
    ux, vx = coords.Rz_to_uv(R, z, delta=delta)
    sinvx, cosvx = xp.sin(vx), xp.cos(vx)
    coshux, sinhux = xp.cosh(ux), xp.sinh(ux)
    pux = delta * (vR * coshux * sinvx + vz * sinhux * cosvx)
    pvx = delta * (vR * sinhux * cosvx - vz * coshux * sinvx)
    E, Lz = calcELStaeckel(R, vR, vT, z, vz, pot)
    if u0 is None:
        u0 = ux  # u0 does not matter for a single action evaluation
    else:
        E = (
            (
                coshux**2.0 * potentialStaeckel(ux, numpy.pi / 2.0, pot, delta)
                - xp.cosh(u0) ** 2.0 * potentialStaeckel(u0, numpy.pi / 2.0, pot, delta)
                + (xp.sinh(u0) ** 2.0 + sinvx**2.0)
                * potentialStaeckel(u0, vx, pot, delta)
            )
            / (sinhux**2.0 + sinvx**2.0)
            + vR**2.0 / 2.0
            + vT**2.0 / 2.0
            + vz**2.0 / 2.0
        )
    sinh2u0 = xp.sinh(u0) ** 2.0
    v0u = numpy.pi / 2.0
    sin2v0u = numpy.sin(v0u) ** 2.0
    potu0v0 = potentialStaeckel(u0, v0u, pot, delta)
    I3U = (
        E * sinhux**2.0
        - pux**2.0 / 2.0 / delta**2.0
        - Lz**2.0 / 2.0 / delta**2.0 / sinhux**2.0
        - (sinhux**2.0 + sin2v0u) * potentialStaeckel(ux, v0u, pot, delta)
        + (sinh2u0 + sin2v0u) * potu0v0
    )
    cosh2u0v = xp.cosh(u0) ** 2.0
    sinh2u0v = sinh2u0
    potupi2 = potentialStaeckel(u0, numpy.pi / 2.0, pot, delta)
    dV = cosh2u0v * potupi2 - (sinh2u0v + sinvx**2.0) * potentialStaeckel(
        u0, vx, pot, delta
    )
    I3V = (
        -E * sinvx**2.0
        + pvx**2.0 / 2.0 / delta**2.0
        + Lz**2.0 / 2.0 / delta**2.0 / sinvx**2.0
        - dV
    )
    return {
        "ux": ux, "vx": vx, "pux": pux, "pvx": pvx, "E": E, "Lz": Lz,
        "u0": u0, "sinh2u0": sinh2u0, "v0u": v0u, "sin2v0u": sin2v0u,
        "potu0v0": potu0v0, "I3U": I3U, "cosh2u0v": cosh2u0v,
        "sinh2u0v": sinh2u0v, "potupi2": potupi2, "I3V": I3V,
    }  # fmt: skip


def _refine_tp(xp, f, df, u, skip):
    """Differentiable turning point: one Newton step off the frozen bisection
    root, grafted so the VALUE is byte-identical (numpy: no refinement) but AD carries the TRUE implicit derivative du/dtheta =
    -f_theta(u)/f_u(u). The bisection roots (backend.optimize.bisect_root) build
    u from piecewise-constant xp.where comparisons, so their AD gradient is a
    meaningless bracket artifact; without this the Staeckel FREQUENCY gradients
    (dJ/d(E,Lz,I3) panels) and the action HESSIAN are finite-but-wrong (the
    turning-point boundary term that cancels the S^-3/2 integral divergence is
    missing). ``skip`` masks entries that are NOT simple roots (axis-reaching
    umin=0, circular umin=umax) where f_u~0."""
    if xp is numpy:
        return u  # numpy path has no autodiff: keep the root EXACTLY. (Grafting
        # here is only ~machine-eps identity, which the 1/sqrt(S) frequency panels
        # amplify to ~1e-13 -- a byte-identity break we must not introduce.)
    u0 = stop_gradient(u)
    fu0 = f(u0)
    # analytic slope: a finite difference's ~1e-9 relative error carries into
    # du/dtheta, which the chi profiles amplify by 1/y(1-y) at their edge nodes
    fp = df(u0)
    # f(u0) can be NaN/inf AT the root (S dips <0 / divides by 0 there) while the
    # slope is finite -> skip the Newton step there. Mask the
    # numerator to NaN-free BEFORE dividing (dead-branch guard: eager xp.where
    # evaluates both sides, so raw fu0/fp would still poison the value/grad).
    good = (xp.abs(fp) > 1e-10) & xp.isfinite(fu0) & xp.isfinite(fp) & ~skip
    fp_safe = xp.where(good, fp, xp.ones_like(fp))
    fu0_safe = xp.where(good, fu0, xp.zeros_like(fu0))
    donor = u0 - fu0_safe / fp_safe  # finite everywhere; == u0 where ~good
    # Value-exact graft: donor - stop_gradient(donor) == 0 EXACTLY (finite donor),
    # so the value is byte-exactly the bisection root (no 1/sqrt(S) amplification of
    # a sg(u)+donor-sg(donor) ~1e-16 wobble) while AD carries du/dtheta = grad(donor).
    return u0 + (donor - stop_gradient(donor))


def _staeckel_jr_args(s, pot, delta):
    """The arguments of the u-libration integrands after u"""
    return (s["E"], s["Lz"], s["I3U"], delta, s["u0"], s["sinh2u0"],
            s["v0u"], s["sin2v0u"], s["potu0v0"], pot)  # fmt: skip


def _stackable(xp, func, args, ref):
    """func(x, *args) at x of ref's shape, or at points stacked along a new
    leading axis (iterate_ladder, bisect_root(stacked=True)): there flat, with
    the arguments of more than one point repeated, as at ref's shape"""

    def f(x):
        if numpy.shape(x) == numpy.shape(ref):
            return func(x, *args)
        flat = [
            xp.reshape(a + 0.0 * x, (-1,)) if len(getattr(a, "shape", ())) else a
            for a in args
        ]
        return xp.reshape(func(xp.reshape(x, (-1,)), *flat), x.shape)

    return f


def _staeckel_uminumax(xp, s, pot, delta):
    """Vectorised (umin, umax): bracket-and-bisect roots of the J_R integrand^2."""
    args = _staeckel_jr_args(s, pot, delta)
    f = _stackable(xp, _JRStaeckelIntegrandSquared, args, s["ux"])
    ux, eps = s["ux"], 1e-8
    at_turn = (xp.abs(s["pux"]) < 1e-7) | (xp.abs(f(ux)) < 1e-10)
    peps, meps = f(ux + eps), f(ux - eps)
    at_umin = at_turn & (peps > 0.0) & (meps < 0.0)
    at_umax = at_turn & (peps < 0.0) & (meps > 0.0)
    circular = at_turn & ~at_umin & ~at_umax
    # lower bracket: step below ux by factors of 0.9 until f < 0, as the C
    # code does: the turning point is the first below ux also when p_u^2 has
    # a second lobe below it (220 steps reach ~1e-10, so even a near-axis
    # turning point of a nearly radial orbit is straddled); traced, the steps
    # stop once every point is bracketed (a while_loop, so no derivative:
    # _refine_tp below supplies it)
    lo = iterate_ladder(lambda l: (f(l) >= 0.0) & (l > 1e-10), ux * 0.9, 0.9, 220)
    # f still >0 at the floor -> no lower J_R turning point: the orbit reaches the
    # symmetry axis (Lz~0, purely-radial), so umin=0 (mirrors C / Single rstart==0).
    reaches_axis = f(lo) >= 0.0
    # expanding bracket above ux until f<0 (stop at u=100)
    hi = iterate_ladder(lambda h: (f(h) >= 0.0) & (h < 100.0), ux * 1.1, 1.1, 80)
    # No upper turning point below u=100 (f(100)>=0 -> u=100 still in the allowed
    # region) -> unbound, mirroring the per-object _uminUmaxFindStart
    # `utry > 100 -> UnboundError`.
    unbound = (f(100.0 * xp.ones_like(ux)) >= 0.0) & ~(at_umax | circular)
    # When the orbit sits exactly AT a turning point (pux~0), f(ux)~0 to round-off
    # (sign indeterminate), so the OTHER turning point must be bracketed from
    # strictly INSIDE the allowed region (ux+/-eps, where f>0) -- else both ends
    # of the bracket are <0 and a narrow interior root is missed (the bisection
    # returns the outer endpoint). Mirrors the Single calcUminUmax ux+/-eps
    # brackets. The snapped side's bisection result is discarded by the where below.
    u_lo_umax = xp.where(at_umin, ux + eps, ux)  # umax: bracket above (from inside)
    u_hi_umin = xp.where(at_umax, ux - eps, ux)  # umin: bracket below (from inside)
    # (the last step of each bracket, where it took more than one)
    u_lo_umax = xp.where(hi > 1.15 * ux, hi / 1.1, u_lo_umax)
    u_hi_umin = xp.where(lo < 0.85 * ux, lo / 0.9, u_hi_umin)
    umin = bisect_root(f, lo, u_hi_umin, xp, xtol=1e-13, maxiter=200, stacked=True)
    umax = bisect_root(f, u_lo_umax, hi, xp, xtol=1e-13, maxiter=200, stacked=True)
    umin = xp.where(at_umin | circular, ux, umin)
    umax = xp.where(at_umax | circular, ux, umax)
    # A circular orbit is a DOUBLE root of f at ux, so f(ux +- eps) ~ -k eps^2 is
    # the size of f's rounding noise and the at_umin/at_umax/circular split above
    # is decided by the last bits (CPU-dependent vector kernels): a circular orbit
    # classified as at a turning point gets its partner root bisected out of the
    # noise, ~sqrt(noise/k) ~ 1e-8 away (e ~ 1e-8 on some machines, 1e-10 on
    # others). A turning-point pair closer than the frequencies' own circular
    # threshold (below) cannot be resolved from that noise: snap it to circular.
    unresolved = at_turn & ((umax - umin) < 1e-6 * umax)
    umin = xp.where(unresolved, ux, umin)
    umax = xp.where(unresolved, ux, umax)
    circular = circular | unresolved
    umin = xp.where(reaches_axis, xp.zeros_like(umin), umin)  # axis-reaching -> 0
    # differentiable turning points (value byte-identical; injects du/dtheta)
    df = lambda u: _dJRStaeckelIntegrandSquareddu(u, *args)
    umin = _refine_tp(xp, f, df, umin, skip=reaches_axis | circular)
    umax = _refine_tp(xp, f, df, umax, skip=circular)
    return umin, umax, unbound


def _staeckel_vmin(xp, s, pot, delta):
    """Vectorised vmin: bracket-and-bisect root of the J_z integrand^2 below vx."""
    args = (s["E"], s["Lz"], s["I3V"], delta, s["u0"], s["cosh2u0v"],
            s["sinh2u0v"], s["potupi2"], pot)  # fmt: skip
    # (the integrand is symmetric about the midplane: a point below it is
    # searched from its mirror image, which the lower turning point is below)
    vx, eps = xp.minimum(s["vx"], numpy.pi - s["vx"]), 1e-8
    f = _stackable(xp, _JzStaeckelIntegrandSquared, args, vx)
    at_turn = (xp.abs(s["pvx"]) < 1e-7) | (xp.abs(f(vx)) < 1e-10)
    at_vmin = at_turn & (f(vx + eps) > 0.0) & (f(vx - eps) < 0.0)
    # step below vx by factors of 0.9 until f < 0 (220 steps reach ~1e-10, so
    # the turning point of an orbit with very little angular momentum, which
    # comes close to the symmetry axis, is straddled) and bisect the last step;
    # traced, the steps stop once every point is bracketed (a while_loop, so no
    # derivative: _refine_tp below supplies it)
    vlo = iterate_ladder(lambda v: (f(v) >= 0.0) & (v > 1e-10), vx * 0.9, 0.9, 220)
    vmin = bisect_root(f, vlo, vlo / 0.9, xp, xtol=1e-13, maxiter=200, stacked=True)
    # f still >= 0 at the floor: the orbit reaches the symmetry axis
    axis = f(vlo) >= 0.0
    vmin = xp.where(at_vmin, vx, xp.where(axis, vlo, vmin))
    df = lambda v: _dJzStaeckelIntegrandSquareddv(v, *args)
    return _refine_tp(xp, f, df, vmin, skip=axis & ~at_vmin)


def _staeckel_trig(xp, name, x):
    """``xp.sin`` etc., but numpy for a plain Python scalar.

    The u0/v0 reference-geometry arguments may arrive as scalars, and torch's
    trig rejects a float. numpy for non-backend values also keeps the numpy path
    byte-identical."""
    return getattr(xp if is_backend_array(x) else numpy, name)(x)


def _staeckel_dS_flat(xp, dSsq, q, args):
    """dS/dq on a (N, nnodes) grid, evaluated FLAT.

    The dS path goes through _evaluateRforces, and some potentials
    (KuzminKutuzovStaeckel) only accept 1-D coordinates there, while the S path
    (evaluatePotentials) is fine with 2-D. Broadcast the per-orbit args against
    the node axis, ravel, evaluate, and restore the shape."""
    shp = q.shape
    fargs = tuple(
        xp.reshape(xp.broadcast_to(a, shp), (-1,)) if getattr(a, "ndim", 0) >= 1 else a
        for a in args
    )
    return xp.reshape(dSsq(xp.reshape(q, (-1,)), *fargs), shp)


def _staeckel_chi_mesh_edges(e):
    """10-node GL rule on each panel of the edge list `e`, ravelled."""
    mid = 0.5 * (e[:-1] + e[1:])
    half = 0.5 * (e[1:] - e[:-1])
    t = (mid[:, None] + _CHIQUAD_GLX[None, :] * half[:, None]).ravel()
    w = (half[:, None] * _CHIQUAD_GLW[None, :]).ravel()
    return t, w


@lru_cache(maxsize=None)
def _staeckel_chi_mesh(nchi):
    """The composite rule on the NORMALIZED anomaly t in [0,1]: `nchi` uniform
    panels of the 10-node GL rule. Kept normalized so ``chimax`` may be per-orbit
    (the angles need an INCOMPLETE integral, whose upper anomaly differs per
    orbit)."""
    return _staeckel_chi_mesh_edges(numpy.linspace(0.0, 1.0, nchi + 1))


@lru_cache(maxsize=None)
def _staeckel_chi_mesh_nearaxis(nchi):
    """The J_R action meshes of a batch with near-axis orbits, at ONE node count
    so the choice is a per-orbit xp.where: (uniform, graded). The graded one is
    C's glfixed_graded on the low half t in [0, 1/2] -- panels halving toward the
    cusp at the axis -- plus the uniform high half; the uniform one has the same
    number of panels in all."""
    nhigh = nchi // 2
    lo = [0.0] + [0.5 * 2.0 ** (k + 1 - _STAECKEL_NEARAXIS_NPANELS)
                  for k in range(_STAECKEL_NEARAXIS_NPANELS)]  # fmt: skip
    hi = list(numpy.linspace(0.5, 1.0, nhigh + 1)[1:])
    graded = _staeckel_chi_mesh_edges(numpy.array(lo + hi))
    uniform = _staeckel_chi_mesh(_STAECKEL_NEARAXIS_NPANELS + nhigh)
    return uniform, graded


def _staeckel_nchi(order, vside):
    """Panels of the composite 10-node chi rule. C parity: C integrates each half
    of the u anomaly with one `order`-point GL rule, so nchi = 2*order/10 gives
    C's nodes at the default order=10 and its node count above it. C's v side
    ends in a t^2 panel based at the midplane, which resolves the disk structure
    there; a uniform chi mesh needs twice the panels to match it (MWPotential2014
    disk orbits: jz 1e-8 -> 2e-10, the turning-point floor)."""
    n = max(2, -(-int(order) // 5))
    return 2 * n if vside else n


def _staeckel_chi_quads(
    xp, Ssq, dSsq, args, qmin, D, order, chimax=numpy.pi, vside=False, nearaxis=None
):
    """Chi-anomaly quadratures of sqrt(S) over the turning-point interval.

    S vanishes LINEARLY at each turning point, so sqrt(S) has a square-root
    branch point there and plain GL converges only algebraically (4.7e-5 at
    order 10 on MWPotential2014). In the anomaly q = qmin + D sin^2(chi/2), with
    y = sin^2(chi/2) and Q = S/[y(1-y)] smooth and nonzero,

        int sqrt(S) dq = (D/4) int sqrt(Q) sin^2(chi) dchi,
        int f/sqrt(S) dq = D int f/sqrt(Q) dchi,

    both analytic in chi. This is the rule main's pure-Python path uses
    (gh#1357) and the one C is regularized to, so c=True and c=False agree to
    machine precision.

    Near a turning point the DIRECT evaluation of S is a difference of O(1)
    potential terms, and dividing it by y(1-y) -> 0 amplifies that cancellation;
    there Q is rebuilt from the analytic derivative as
    S ~ (q - q0)[S'(q0) + S'(q)]/2. Without this the actions stall at ~4e-14.

    ``chimax`` is the anomaly of the upper limit -- pi for a complete
    turning-point-to-turning-point integral, pi/2 for the z action (whose anomaly
    spans the whole v loop [vmin, pi - vmin]: the midplane is a symmetry point of
    S_z, not a turning point), or a per-orbit 2 arcsin(sqrt((q-qmin)/D)) for the
    incomplete integrals the angles need.

    ``nearaxis`` (the J_R action only, as in C) is the per-orbit mask of orbits
    reaching the axis, umin < 0.2: there S picks up the potential's inner cusp,
    S ~ A - B u^0.2, i.e. chi^0.4 in the anomaly, against which a uniform rule
    converges only algebraically (4.5e-5 at two panels) and the S' edge model
    below is wrong (the model's linearization is what the cusp breaks). Those
    orbits get C's graded mesh and the DIRECT Q; the rest of the batch keeps the
    uniform rule at the same node count (jr matches C to 4e-12 either way).

    Returns (action, sqrt(Q), q, wts) so every quadrature on this mesh -- the
    action and the 1/p profiles -- comes from ONE evaluation of S.
    """
    nchi = _staeckel_nchi(order, vside)
    dev = device_of(qmin)
    if nearaxis is None:
        t, w = _staeckel_chi_mesh(nchi)
        t = asarray_on_device(xp, t, dev)
        w = asarray_on_device(xp, w, dev)
    else:
        (tu, wu), (tg, wg) = _staeckel_chi_mesh_nearaxis(nchi)
        na = nearaxis[..., None]
        t = xp.where(na, asarray_on_device(xp, tg, dev), asarray_on_device(xp, tu, dev))
        w = xp.where(na, asarray_on_device(xp, wg, dev), asarray_on_device(xp, wu, dev))
    chi = chimax * t
    wts = chimax * w
    y = xp.sin(chi / 2.0) ** 2.0
    y1my = y * (1.0 - y)
    # masks, not control flow: xp.where evaluates BOTH branches, so each unused
    # denominator is kept finite or reverse-mode AD is poisoned by its NaN
    ones = xp.ones_like(y1my)
    is_edge = y1my <= _STAECKEL_CHI_EDGE
    if nearaxis is not None:
        is_edge = is_edge & ~na
    is_lo = y < 0.5
    y1my_safe = xp.where(is_edge, ones, y1my)
    omy_safe = xp.where(is_lo, 1.0 - y, ones)
    y_safe = xp.where(is_lo, ones, y)

    a2 = tuple(x[..., None] if getattr(x, "ndim", 0) >= 1 else x for x in args)
    q = qmin[..., None] + D[..., None] * y
    Q = Ssq(q, *a2) / y1my_safe
    dS_n = _staeckel_dS_flat(xp, dSsq, q, a2)
    dS_lo = _staeckel_dS_flat(xp, dSsq, qmin[..., None], a2)
    dS_hi = _staeckel_dS_flat(xp, dSsq, (qmin + D)[..., None], a2)
    Q_edge = xp.where(
        is_lo,
        D[..., None] * (dS_lo + dS_n) / 2.0 / omy_safe,
        D[..., None] * (-dS_hi - dS_n) / 2.0 / y_safe,
    )
    Q = xp.where(is_edge, Q_edge, Q)
    tiny = numpy.finfo(float).tiny
    Q = xp.where(Q > tiny, Q, tiny * ones)
    sqQ = xp.sqrt(Q)
    action = (D / 4.0) * xp.sum(wts * sqQ * xp.sin(chi) ** 2.0, axis=-1)
    return action, sqQ, q, wts


def _staeckel_chi_action(
    xp, Ssq, dSsq, args, qmin, D, order, chimax=numpy.pi, vside=False, nearaxis=None
):
    """The action integral alone -- see :func:`_staeckel_chi_quads`."""
    return _staeckel_chi_quads(
        xp, Ssq, dSsq, args, qmin, D, order, chimax, vside, nearaxis
    )[0]


def _staeckel_prep(xp, R, vR, vT, z, vz, pot, delta, flag_unbound=False, u0=None):
    """Setup quantities + turning points (+ unbound check), shared by the
    vectorised actions and frequencies. Returns (setup, umin, umax, vmin, delta).
    ``flag_unbound`` (c=True routed here) follows C instead of raising: unbound
    orbits are computed on a bound stand-in at the same (R,z), so the dead branch
    stays finite under AD, and _staeckel_flag_unbound sets their outputs to
    9999.99."""
    if is_backend_array(R) and not is_backend_array(delta):
        # match R's namespace AND device: a bare xp.asarray(delta) lands on the
        # CPU, so an array-valued delta (e.g. EccZmax's atleast_1d) then collides
        # with CUDA R/z in coords.Rz_to_uv. No-op on numpy (device_of -> None).
        delta = asarray_on_device(xp, delta, device_of(R))
    s = _staeckel_setup(xp, R, vR, vT, z, vz, pot, delta, u0=u0)
    umin, umax, unbound = _staeckel_uminumax(xp, s, pot, delta)
    # Unbound orbits raise, mirroring the Single class. Eager jax/torch have a
    # concrete truth value here and raise identically to numpy; only under a jax
    # trace is `unbound` a tracer, and there the orbit falls through to NaN for
    # the caller to check.
    if flag_unbound:
        any_unbound = xp.any(unbound)
        if not has_concrete_truth_value(any_unbound) or bool(any_unbound):
            vc = xp.sqrt(-R * _evaluateRforces(pot, R, xp.zeros_like(R)))
            vR, vT, vz = (
                xp.where(unbound, f * vc, v)
                for f, v in ((0.1, vR), (0.9, vT), (0.1, vz))
            )
            Lz = s["Lz"]
            s = _staeckel_setup(xp, R, vR, vT, z, vz, pot, delta, u0=u0)
            umin, umax, _ = _staeckel_uminumax(xp, s, pot, delta)
            s["Lz_out"] = Lz
        s["unbound"] = unbound
    elif concretely_true(xp.any(unbound)):
        raise UnboundError("Orbit seems to be unbound")
    vmin = _staeckel_vmin(xp, s, pot, delta)
    # Planar orbit (jz=0): snap vmin to exactly pi/2 (the bisection lands ~1e-8
    # off). Shared by actions (jz->0), freqs (zero-width J_z panels -> det(A)=0
    # exactly, deterministic NaN/Inf across backends, matching C) and EccZmax
    # (zmax=0 exactly).
    vmin = xp.where(
        (numpy.pi / 2.0 - vmin) < 1e-7, numpy.pi / 2.0 * xp.ones_like(vmin), vmin
    )
    return s, umin, umax, vmin, delta


def _staeckel_flag_unbound(xp, s, outs):
    """(Lz, outs) with the outputs of flagged unbound orbits set to C's 9999.99."""
    if "unbound" not in s:
        return s["Lz"], outs
    return s.get("Lz_out", s["Lz"]), tuple(
        xp.where(s["unbound"], 9999.99, o) for o in outs
    )


def _staeckel_jr_jz(xp, s, umin, umax, vmin, pot, delta, order):
    """(jr, jz) action integrals from prepared setup + turning points."""
    sqrt2 = numpy.sqrt(2.0)
    jr_args = (s["E"], s["Lz"], s["I3U"], delta, s["u0"], s["sinh2u0"],
               s["v0u"], s["sin2v0u"], s["potu0v0"], pot)  # fmt: skip
    jr = (
        _staeckel_chi_action(
            xp,
            _JRStaeckelIntegrandSquared,
            _dJRStaeckelIntegrandSquareddu,
            jr_args,
            umin,
            umax - umin,
            order,
            nearaxis=umin < _STAECKEL_NEARAXIS,
        )
        * sqrt2
        * delta
        / numpy.pi
    )
    jz_args = (s["E"], s["Lz"], s["I3V"], delta, s["u0"], s["cosh2u0v"],
               s["sinh2u0v"], s["potupi2"], pot)  # fmt: skip
    jz = (
        _staeckel_chi_action(
            xp,
            _JzStaeckelIntegrandSquared,
            _dJzStaeckelIntegrandSquareddv,
            jz_args,
            vmin,
            numpy.pi - 2.0 * vmin,  # the FULL v loop; midplane is chi = pi/2
            order,
            chimax=numpy.pi / 2.0,
            vside=True,
        )
        * 2.0
        * sqrt2
        * delta
        / numpy.pi
    )
    jr = xp.where((umax - umin) / umax < 1e-6, xp.zeros_like(jr), jr)
    jz = xp.where((numpy.pi / 2.0 - vmin) < 1e-7, xp.zeros_like(jz), jz)
    return jr, jz


def _staeckel_actions(
    xp, R, vR, vT, z, vz, pot, delta, order, flag_unbound=False, u0=None
):
    """Unified vectorised (jr, Lz, jz) for numpy and jax/torch backends."""
    s, umin, umax, vmin, delta = _staeckel_prep(
        xp, R, vR, vT, z, vz, pot, delta, flag_unbound, u0=u0
    )
    Lz, (jr, jz) = _staeckel_flag_unbound(
        xp, s, _staeckel_jr_jz(xp, s, umin, umax, vmin, pot, delta, order)
    )
    return jr, Lz, jz


# --------------------------------------------------------------- frequencies
# The frequency derivative integrals dJ/d(E,Lz,I3) need the t^2-substitution
# (their integrands are (factor)/sqrt(S), SINGULAR at the turning points, unlike
# the action integrand sqrt(S) which vanishes there). Mirror the C
# dJ?d?{Low,High}StaeckelIntegrand split: low panel u=lo+t^2, high panel
# u=hi-t^2, both over t in [0, sqrt(0.5(hi-lo))]. xp.where guards the dead S<=0
# branch (the orbit can sit arbitrarily close to a turning point).


def _staeckel_chi_profiles(
    xp, Ssq, dSsq, args, qmin, D, order, weight_fns, chimax=numpy.pi, vside=False
):
    """The 1/p profile integrals int f/sqrt(S) dq for every f in `weight_fns`,
    on the SAME chi mesh as the action: int f/sqrt(S) dq = D int f/sqrt(Q) dchi.

    One mesh serves all of them (main's `_chiQuadsU`/`_chiQuadsV` do the same),
    so the six Leibniz derivatives cost two S evaluations rather than twelve."""
    _, sqQ, q, wts = _staeckel_chi_quads(
        xp, Ssq, dSsq, args, qmin, D, order, chimax, vside
    )
    return [D * xp.sum(wts * f(xp, q) / sqQ, axis=-1) for f in weight_fns]


def _staeckel_jacobian(xp, s, umin, umax, vmin, pot, delta, order):
    """The six full-range Leibniz derivatives (djrdE,djrdLz,djrdI3,djzdE,djzdLz,
    djzdI3) -- the t^2-substituted dJ/d(E,Lz,I3) integrals with their prefactors.
    Shared by the frequencies and the angles."""
    sqrt2 = numpy.sqrt(2.0)
    Lz = s["Lz"]
    prefr = delta / numpy.pi / sqrt2
    prefz = sqrt2 * delta / numpy.pi  # NB: djz prefactors are 2x djr's, +I3
    jr_args = (s["E"], s["Lz"], s["I3U"], delta, s["u0"], s["sinh2u0"],
               s["v0u"], s["sin2v0u"], s["potu0v0"], pot)  # fmt: skip
    jz_args = (s["E"], s["Lz"], s["I3V"], delta, s["u0"], s["cosh2u0v"],
               s["sinh2u0v"], s["potupi2"], pot)  # fmt: skip
    # One chi mesh per coordinate, regularized exactly as the actions are: the
    # 1/sqrt(S) integrands are SINGULAR at the turning points, and the t^2 panels
    # this replaces converged only to ~6e-10 -- 20x main's bar for the
    # frequencies (gh#1357's convergence test, output 3).
    duE, duI3, duLz = _staeckel_chi_profiles(
        xp,
        _JRStaeckelIntegrandSquared,
        _dJRStaeckelIntegrandSquareddu,
        jr_args,
        umin,
        umax - umin,
        order,
        (
            lambda xp, u: xp.sinh(u) ** 2.0,
            lambda xp, u: xp.ones_like(u),
            lambda xp, u: 1.0 / xp.sinh(u) ** 2.0,
        ),
    )
    dvE, dvI3, dvLz = _staeckel_chi_profiles(
        xp,
        _JzStaeckelIntegrandSquared,
        _dJzStaeckelIntegrandSquareddv,
        jz_args,
        vmin,
        numpy.pi - 2.0 * vmin,  # the FULL v loop; the midplane is chi = pi/2
        order,
        (
            lambda xp, v: xp.sin(v) ** 2.0,
            lambda xp, v: xp.ones_like(v),
            lambda xp, v: 1.0 / xp.sin(v) ** 2.0,
        ),
        chimax=numpy.pi / 2.0,
        vside=True,
    )
    djrdE = duE * prefr
    djrdLz = duLz * (-Lz / numpy.pi / sqrt2 / delta)
    djrdI3 = duI3 * (-prefr)
    djzdE = dvE * prefz
    djzdLz = dvLz * (-Lz * sqrt2 / numpy.pi / delta)
    djzdI3 = dvI3 * prefz
    return djrdE, djrdLz, djrdI3, djzdE, djzdLz, djzdI3


def _staeckel_freqs(xp, s, umin, umax, vmin, pot, delta, order, jac):
    """Vectorised (Omegar, Omegaphi, Omegaz); NaN for circular (caller substitutes
    epifreq/omegac/verticalfreq, mirroring the C 0/0=NaN -> close-to-circular path).
    `jac` = the _staeckel_jacobian 6-tuple, computed once by the caller and shared
    with the angles."""
    djrdE, djrdLz, djrdI3, djzdE, djzdLz, djzdI3 = jac
    detA = djrdE * djzdI3 - djzdE * djrdI3
    circ = (umax - umin) / umax < 1e-6  # circular in R: det(A)=0 (J_R panels ->0)
    planar = (numpy.pi / 2.0 - vmin) < 1e-7  # planar (J_z=0): det(A)=0 (J_z panels ->0)
    degen = circ | planar
    detsafe = xp.where(degen, xp.ones_like(detA), detA)  # avoid the 0/0 division
    nan = numpy.nan * xp.ones_like(detA)
    inf = numpy.inf * xp.ones_like(detA)
    # circular -> all NaN (caller substitutes epifreq/omegac/verticalfreq, since
    # jr,jz<1e-3). planar-but-radially-eccentric -> Omegar,Omegaphi=NaN and
    # Omegaz=Inf (NOT NaN): this reproduces the C 0/0 & x/0 IEEE result and, crucially,
    # keeps the Omegaz<1e-3-substitution from firing -- so Omegar stays NaN for the
    # genuinely eccentric radial motion rather than being wrongly set to epifreq.
    Omegar = xp.where(degen, nan, djzdI3 / detsafe)
    Omegaz = xp.where(circ, nan, xp.where(planar, inf, -djrdI3 / detsafe))
    Omegaphi = xp.where(degen, nan, (djrdI3 * djzdLz - djzdI3 * djrdLz) / detsafe)
    return Omegar, Omegaphi, Omegaz


def _staeckel_actions_freqs(
    xp,
    R,
    vR,
    vT,
    z,
    vz,
    pot,
    delta,
    order,
    flag_unbound=False,
    u0=None,
    torus=False,
    ddelta=None,
    du0=None,
):
    """Unified vectorised (jr, Lz, jz, Omegar, Omegaphi, Omegaz); the frequencies
    are NaN for circular orbits (the caller substitutes epifreq/omegac/verticalfreq).
    Setup + turning points are computed once and shared between actions and freqs.
    With torus, those of the family whose reference curve follows its tori
    (_staeckel_angles)."""
    s, umin, umax, vmin, delta = _staeckel_prep(
        xp, R, vR, vT, z, vz, pot, delta, flag_unbound, u0=u0
    )
    jr, jz = _staeckel_jr_jz(xp, s, umin, umax, vmin, pot, delta, order)
    jac = _staeckel_jacobian(xp, s, umin, umax, vmin, pot, delta, order)
    Omegar, Omegaphi, Omegaz = _staeckel_freqs(
        xp, s, umin, umax, vmin, pot, delta, order, jac
    )
    if torus:
        dOm = _staeckel_angles(
            xp, s, umin, umax, vmin, pot, delta, order, jac, torus, ddelta, du0
        )[3]
        Omegar, Omegaphi, Omegaz = Omegar + dOm[0], Omegaphi + dOm[1], Omegaz + dOm[2]
    Lz, (jr, jz, Omegar, Omegaphi, Omegaz) = _staeckel_flag_unbound(
        xp, s, (jr, jz, Omegar, Omegaphi, Omegaz)
    )
    return jr, Lz, jz, Omegar, Omegaphi, Omegaz


# ------------------------------------------------------------------- angles
# The angles need PARTIAL Leibniz integrals (from a turning point to the current
# u/v), unlike the freqs' full turning-point-to-turning-point integrals. The
# vectorised quadrant tree mirrors the per-object calcAnglesStaeckel: the panel
# (Low from umin/vmin, High from umax/pi-2) is chosen by which turning point the
# position is closer to, and a reflection constant K and sign s -- functions only
# of the momentum sign x position quadrant -- map the partial integral onto the
# full angle (4 leaves in u, 8 in v). All branches are computed and xp.where-
# selected (with the turning-point dead-branch guard) for vectorisation.


def _dVdu0(v, u0, delta, pot):
    """The derivative of V(v) = cosh^2 u0 Phi(u0,pi/2) - (sinh^2 u0 + sin^2 v)
    Phi(u0,v) with respect to the reference u0 along which it is built"""
    xp = get_namespace(v, u0)
    sh, ch = xp.sinh(u0), xp.cosh(u0)
    R, z = coords.uv_to_Rz(u0, v, delta=delta)
    R0 = delta * sh
    return (
        2.0
        * sh
        * ch
        * (_evaluatePotentials(pot, R0, 0.0 * R0) - _evaluatePotentials(pot, R, z))
        - ch**3.0 * delta * _evaluateRforces(pot, R0, 0.0 * R0)
        + (sh**2.0 + xp.sin(v) ** 2.0)
        * delta
        * (
            _evaluateRforces(pot, R, z) * ch * xp.sin(v)
            + _evaluatezforces(pot, R, z) * sh * xp.cos(v)
        )
    )


def _dUddelta(u, delta, pot):
    """The derivative of U(u) = cosh^2 u Phi(u,pi/2) with respect to the
    focal length at fixed u"""
    xp = get_namespace(u, delta)
    R = delta * xp.sinh(u)
    return -(xp.cosh(u) ** 2.0) * xp.sinh(u) * _evaluateRforces(pot, R, 0.0 * R)


def _dVddelta(v, u0, delta, pot):
    """The derivative of V(v) = cosh^2 u0 Phi(u0,pi/2) - (sinh^2 u0 + sin^2 v)
    Phi(u0,v) with respect to the focal length at fixed u0 and v"""
    xp = get_namespace(v, u0, delta)
    sh, ch = xp.sinh(u0), xp.cosh(u0)
    R, z = coords.uv_to_Rz(u0, v, delta=delta)
    R0 = delta * sh
    return -(ch**2.0) * sh * _evaluateRforces(pot, R0, 0.0 * R0) + (
        sh**2.0 + xp.sin(v) ** 2.0
    ) * (
        _evaluateRforces(pot, R, z) * sh * xp.sin(v)
        + _evaluatezforces(pot, R, z) * ch * xp.cos(v)
    )


def _u0torus_partials(xp, s, umin, umax, pot, delta):
    """The partials with respect to (E, I_3, L_z, delta) of the reference
    curve that follows a torus's own turning points, delta sinh(u0) = 2 /
    (1/R_- + 1/R_+) with R = delta sinh(u) at the two turning points of the
    u libration, which move with the integrals and the focal length by the
    level-set rule"""
    args = _staeckel_jr_args(s, pot, delta)
    Lz = s["Lz"]
    Rm, Rp = delta * xp.sinh(umin), delta * xp.sinh(umax)
    out = 0.0
    for u, dH in ((umin, 2.0 * Rp**2.0), (umax, 2.0 * Rm**2.0)):
        sh2 = xp.sinh(u) ** 2.0
        dS = _dJRStaeckelIntegrandSquareddu(u, *args)
        chain = dH / (Rm + Rp) ** 2.0 * xp.cosh(u) / xp.cosh(s["u0"]) / dS
        out = out + chain * xp.stack(
            [
                -sh2,
                xp.ones_like(sh2),
                Lz / delta**2.0 / sh2,
                _dUddelta(u, delta, pot) - Lz**2.0 / delta**3.0 / sh2,
            ]
        )
    return out


def _staeckel_planar(xp, s, pot, delta):
    """The turning points of the u libration of the planar orbit of each
    point's energy and angular momentum in its model (that of the setup s):
    those at the third integral at which p_v vanishes in the midplane"""
    sp = dict(s)
    sp["I3U"] = (
        s["Lz"] ** 2.0 / 2.0 / delta**2.0
        - s["E"]
        + (s["sinh2u0"] + s["sin2v0u"]) * s["potu0v0"]
    )
    sp["pux"] = xp.sqrt(
        xp.maximum(
            s["pux"] ** 2.0 + 2.0 * delta**2.0 * (s["I3U"] - sp["I3U"]),
            xp.zeros_like(s["pux"]),
        )
    )
    return _staeckel_uminumax(xp, sp, pot, delta)[:2]


def _u0planar_partials(xp, s, pot, delta):
    """The partials with respect to (E, I_3, L_z, delta) of the reference
    curve that follows the planar orbit of a torus's energy and angular
    momentum (u0='planar' of actionAngleStaeckelInverse): those of
    _u0torus_partials at the planar orbit's turning points, along its third
    integral L_z^2 / (2 delta^2) - E"""
    g = _u0torus_partials(xp, s, *_staeckel_planar(xp, s, pot, delta), pot, delta)
    Lz = s["Lz"]
    return xp.stack(
        [
            g[0] - g[1],
            xp.zeros_like(g[1]),
            g[2] + g[1] * Lz / delta**2.0,
            g[3] - g[1] * Lz**2.0 / delta**3.0,
        ]
    )


def _u0shell(xp, s, umin, umax, pot, delta):
    """The u of the shell orbit of each point's energy and angular momentum
    (u0='shell' of actionAngleStaeckelInverse): where p_u^2, which the third
    integral only shifts, is largest, between the point's turning points (on
    jax/torch, the backend brentq, whose final Newton step carries the
    derivative)"""
    args = _staeckel_jr_args(s, pot, delta)
    if xp is numpy:
        return bisect_root(
            lambda u: _dJRStaeckelIntegrandSquareddu(u, *args),
            umin,
            umax,
            numpy,
            xtol=1e-14,
            maxiter=200,
        )
    return _backend_brentq(
        lambda u: _dJRStaeckelIntegrandSquareddu(u, *args), umin, umax, xtol=1e-14
    )


def _staeckel_fraction(xp, s, ush, pot, delta, partials=False):
    """The fraction f of the way of each point's third integral from the
    planar orbit's (in the gauge V(pi/2) = 0: L_z^2 / (2 delta^2) - E) to
    the shell orbit's (above the point's by p_u^2 / (2 delta^2) at the shell
    orbit's u, ush) of its energy and angular momentum, in its Staeckel
    model; a circular orbit has no fraction (1/2). With partials, also the
    partials of f with respect to (E, I_3, L_z, delta): the range of the
    third integral, E cosh^2 u - U(u) - L_z^2 coth^2 u / (2 delta^2) at
    ush, is stationary in ush"""
    E, Lz = s["E"], s["Lz"]
    below = s["I3V"] - (Lz**2.0 / 2.0 / delta**2.0 - E)
    tot = below + _JRStaeckelIntegrandSquared(ush, *_staeckel_jr_args(s, pot, delta))
    # (a circular orbit: a range of the third integral at the level of the
    # rounding)
    ok = tot > _MODEL_CIRC * xp.abs(E)
    tot = xp.where(ok, tot, 1.0)
    f = xp.clip(xp.where(ok, below / tot, 0.5), 0.0, 1.0)
    if not partials:
        return f
    coth2 = 1.0 / xp.tanh(ush) ** 2.0
    dbelow = (1.0, 1.0, -Lz / delta**2.0, Lz**2.0 / delta**3.0)
    dtot = (
        xp.cosh(ush) ** 2.0,
        0.0,
        -Lz / delta**2.0 * coth2,
        Lz**2.0 / delta**3.0 * coth2 - _dUddelta(ush, delta, pot),
    )
    return f, xp.stack(
        [xp.where(ok, (b - f * t) / tot, 0.0) + 0.0 * f for b, t in zip(dbelow, dtot)]
    )


def _u0shell_partials(xp, s, ush, pot, delta, R2deriv):
    """The partials with respect to (E, I_3, L_z, delta) of the shell u, by
    the level-set rule on d S_u / du = 0, S_u = E sinh^2 u - U(u) - L_z^2 /
    (2 delta^2 sinh^2 u) with U(u) = cosh^2 u Phi(delta sinh u, 0); R2deriv
    is the potential's second radial derivative in the midplane"""
    E, Lz = s["E"], s["Lz"]
    sh, ch = xp.sinh(ush), xp.cosh(ush)
    R = delta * sh
    Phi = _evaluatePotentials(pot, R, 0.0 * R)
    PhiR = -_evaluateRforces(pot, R, 0.0 * R)
    Suu = (
        2.0 * xp.cosh(2.0 * ush) * (E - Phi)
        - 5.0 * delta * sh * ch**2.0 * PhiR
        - delta**2.0 * ch**4.0 * R2deriv
        + Lz**2.0 / delta**2.0 * (sh**2.0 - 3.0 * ch**2.0) / sh**4.0
    )
    SuD = (
        -(2.0 * sh**2.0 * ch + ch**3.0) * PhiR
        - delta * ch**3.0 * sh * R2deriv
        - 2.0 * Lz**2.0 / delta**3.0 * ch / sh**3.0
    )
    return xp.stack(
        [
            -xp.sinh(2.0 * ush) / Suu,
            xp.zeros_like(Suu),
            -2.0 * Lz / delta**2.0 * ch / sh**3.0 / Suu,
            -SuD / Suu,
        ]
    )


def _bk_host_linearized(host_fn, q, partials=False):
    """A function that jax/torch cannot run (a user's callable, or a family's
    rule), evaluated on the host (on numpy, whatever backend is forced) at
    the derivative-free values of the traced arrays q = (q_1, ..., q_k):
    host_fn(Q) returns its value (n,), gradient (k, n) and second
    derivatives (k, k, n) at Q (k, n). Returns its value linearized in q
    with the gradient or, with partials, its gradient linearized with the
    second derivatives, which carries the derivatives of both"""
    xp = get_namespace(*q)
    k = len(q)
    cols = [stop_gradient(t) for t in q]
    # (host_eval returns the shape it is given)
    packed = xp.stack(
        [t + 0.0 * cols[0] for t in cols] + [0.0 * cols[0]] * (1 + k * k), axis=-1
    )

    def host(a):
        a = numpy.asarray(a, dtype="float")
        with use("numpy", force=True):
            val, grad, hess = host_fn(a[:, :k].T.copy())
        return numpy.concatenate(
            [val[:, None], grad.T, hess.reshape(k * k, -1).T], axis=1
        )

    out = host_eval(host, packed)
    grad = xp.permute_dims(out[:, 1 : 1 + k], (1, 0))
    dq = [t - stop_gradient(t) for t in q]
    if partials:
        hess = xp.reshape(xp.permute_dims(out[:, 1 + k :], (1, 0)), (k, k, -1))
        return xp.stack(
            [grad[i] + sum(hess[i, j] * dq[j] for j in range(k)) for i in range(k)]
        )
    return out[:, 0] + sum(grad[i] * dq[i] for i in range(k))


def _u0callable(rule, nargs, E, Lz, I3, partials=False):
    """The reference curve of a callable rule u0(E, |L_z|[, I_3]) of
    actionAngleStaeckelInverse for each torus and, with partials, its
    partials with respect to (E, I_3, L_z, delta), by central differences of
    the callable (FINITE DIFFERENCES, of the user's callable alone). On
    jax/torch the callable runs on the host: its value is linearized in (E,
    I_3, L_z) with those partials, and the partials with its second
    derivatives (also by central differences), which carries the derivatives
    of both"""
    # (by the data: the host evaluation below is numpy whatever is forced)
    if any(is_backend_array(t) for t in (E, Lz, I3)):

        def host_fn(Q):
            E, I3, Lz = Q
            der = _u0callable(rule, nargs, E, Lz, I3, partials=True)
            hes = _u0callable_hessian(rule, nargs, E, Lz, I3)
            return _u0callable(rule, nargs, E, Lz, I3), der[:3], hes

        out = _bk_host_linearized(host_fn, (E, I3, Lz), partials)
        if partials:
            # (a reference curve's rule does not depend on the focal length)
            return get_namespace(out).concat([out, 0.0 * out[:1]])
        return out

    def f(E, ell, I3):
        return numpy.array(
            [float(rule(*a[:nargs])) for a in zip(*numpy.broadcast_arrays(E, ell, I3))]
        )

    ell = numpy.fabs(Lz)
    if not partials:
        return f(E, ell, I3)
    hE, hL, hI = (1e-6 * (1.0 + numpy.fabs(a)) for a in (E, ell, I3))
    return numpy.array(
        [
            (f(E + hE, ell, I3) - f(E - hE, ell, I3)) / (2.0 * hE),
            (f(E, ell, I3 + hI) - f(E, ell, I3 - hI)) / (2.0 * hI),
            numpy.sign(Lz) * (f(E, ell + hL, I3) - f(E, ell - hL, I3)) / (2.0 * hL),
            numpy.zeros_like(hE),
        ]
    )


def _u0callable_hessian(rule, nargs, E, Lz, I3):
    """The second derivatives of a callable rule u0(E, |L_z|[, I_3]) with
    respect to (E, I_3, L_z), (3, 3, n), by central differences of the
    callable (FINITE DIFFERENCES, of the user's callable alone)"""

    def f(x):
        E, I3, ell = x
        return numpy.array(
            [float(rule(*a[:nargs])) for a in zip(*numpy.broadcast_arrays(E, ell, I3))]
        )

    x = [E, I3, numpy.fabs(Lz)]
    h = [1e-4 * (1.0 + numpy.fabs(a)) for a in x]
    sgn = numpy.array([1.0, 1.0, 1.0])[:, None] * numpy.ones_like(E)
    sgn[2] = numpy.sign(Lz)

    def at(**d):
        return f([x[k] + d.get(f"d{k}", 0.0) * h[k] for k in range(3)])

    f0 = at()
    out = numpy.empty((3, 3) + numpy.shape(E))
    for k in range(3):
        out[k, k] = (at(**{f"d{k}": 1}) - 2.0 * f0 + at(**{f"d{k}": -1})) / h[k] ** 2
        for j in range(k):
            out[k, j] = out[j, k] = (
                at(**{f"d{k}": 1, f"d{j}": 1})
                - at(**{f"d{k}": 1, f"d{j}": -1})
                - at(**{f"d{k}": -1, f"d{j}": 1})
                + at(**{f"d{k}": -1, f"d{j}": -1})
            ) / (4.0 * h[k] * h[j])
    # (L_z enters as |L_z|)
    return out * sgn[:, None] * sgn[None, :]


def _staeckel_angles(
    xp, s, umin, umax, vmin, pot, delta, order, jac, torus=False, ddelta=None, du0=None
):
    """Vectorised (angler, anglephi_raw, anglez); the caller folds the azimuth phi
    into anglephi. angler/anglez are in [0, 2pi); circular orbits -> all 0.
    `jac` = the _staeckel_jacobian 6-tuple, computed once by the caller and shared
    with the freqs. With torus, for a reference curve that follows the tori
    (torus 2: u0='torus' of actionAngleStaeckelInverse; 3: its u0='planar'):
    the angles of that family and
    the difference of its frequencies from the model's, in the order (R, phi,
    z). The family's generating function is the model's at the reference
    curve u0(J) of each torus, so that its angles are the model's plus K
    du0/dJ, with K = dS/du0 at fixed position and actions, and its
    frequencies dE/dJ are the model's plus (dE/du0 at fixed actions) du0/dJ.
    Everything but the integral of dV/du0 over p_v, which enters as the
    integral of the third integral's weight does, is what the model's own
    angles are made of. With ddelta, the derivatives of each torus's focal
    length with respect to (J_R, L_z, J_z), also for a focal length that
    follows the tori: the same with K_delta = dS/ddelta at fixed position
    and actions, which is the dilation of the point transformation, -(R p_R
    + z p_z)/delta, plus the integrals of the derivatives of p_u and p_v
    with respect to delta along the orbit less the angles times their
    complete versions."""
    sqrt2 = numpy.sqrt(2.0)
    pi = numpy.pi
    Lz = s["Lz"]
    ux, vx, pux, pvx = s["ux"], s["vx"], s["pux"], s["pvx"]
    djrdE, djrdLz, djrdI3, djzdE, djzdLz, djzdI3 = jac
    detA = djrdE * djzdI3 - djzdE * djrdI3
    circ = (umax - umin) / umax < 1e-6
    planar = (pi / 2.0 - vmin) < 1e-7
    detsafe = xp.where(circ | planar, xp.ones_like(detA), detA)
    Omegar = djzdI3 / detsafe
    Omegaz = -djrdI3 / detsafe
    Omegaphi = (djrdI3 * djzdLz - djzdI3 * djrdLz) / detsafe
    dI3dJR = -djzdE / detsafe
    dI3dJz = djrdE / detsafe
    dI3dLz = -(djrdE * djzdLz - djzdE * djrdLz) / detsafe
    jr_args = (s["E"], s["Lz"], s["I3U"], delta, s["u0"], s["sinh2u0"],
               s["v0u"], s["sin2v0u"], s["potu0v0"], pot)  # fmt: skip
    jz_args = (s["E"], s["Lz"], s["I3V"], delta, s["u0"], s["cosh2u0v"],
               s["sinh2u0v"], s["potupi2"], pot)  # fmt: skip
    JRsq, JZsq = _JRStaeckelIntegrandSquared, _JzStaeckelIntegrandSquared
    # ---- u-branch: chi-anomaly partials (same rule as the actions/frequencies).
    # chi(u) = 2 arcsin(sqrt((u-umin)/Du)) is the anomaly of the current u, so the
    # partial from the LOW turning point is just chimax = chi(ux); the partial
    # from the HIGH one is (complete - that). Replaces the t^2 panels, which
    # converged only to ~1.6e-9 -- 50x main's bar for the angles.
    Du = umax - umin
    Du_safe = xp.where(Du > 0.0, Du, xp.ones_like(Du))
    chi_ux = 2.0 * xp.asin(xp.sqrt(xp.clip((ux - umin) / Du_safe, 0.0, 1.0)))
    high_u = ux > umin + 0.5 * Du
    uweights = (
        lambda xp, u: xp.sinh(u) ** 2.0,
        lambda xp, u: xp.ones_like(u),
        lambda xp, u: 1.0 / xp.sinh(u) ** 2.0,
    )
    if ddelta is not None:
        # sqrt(2) d p/d delta at fixed (E, I_3), over the 1/sqrt(S) of the
        # profile integrals: 2 S + L_z^2 / (delta^2 sinh^2 u) - delta dU/ddelta
        # and 2 S + L_z^2 / (delta^2 sin^2 v) + delta dV/ddelta
        col = lambda a: a[..., None] if getattr(a, "ndim", 0) >= 1 else a  # noqa: E731
        dcol, Lcol = col(delta), col(Lz)
        ucols = tuple(col(a) for a in jr_args)
        vcols = tuple(col(a) for a in jz_args)
        uweights += (
            lambda xp, q: (
                2.0 * _staeckel_dS_flat(xp, JRsq, q, ucols)
                + Lcol**2.0 / dcol**2.0 / xp.sinh(q) ** 2.0
                - dcol
                * _staeckel_dS_flat(
                    xp, lambda qq, dd: _dUddelta(qq, dd, pot), q, (dcol,)
                )
            ),
        )
    part_u = _staeckel_chi_profiles(
        xp, JRsq, _dJRStaeckelIntegrandSquareddu, jr_args, umin, Du, order,
        uweights, chimax=chi_ux[..., None],
    )  # fmt: skip
    full_u = _staeckel_chi_profiles(
        xp, JRsq, _dJRStaeckelIntegrandSquareddu, jr_args, umin, Du, order,
        uweights,
    )  # fmt: skip
    PE, PI, PL, *PW = (xp.where(high_u, fu - pu, pu) for pu, fu in zip(part_u, full_u))
    pos_u = pux > 0.0
    K_u = xp.where(high_u, pi, xp.where(pos_u, 0.0, 2.0 * pi)) * xp.ones_like(ux)
    s_u = xp.where(high_u, xp.where(pos_u, -1.0, 1.0), xp.where(pos_u, 1.0, -1.0))
    Or1 = K_u * djrdE + s_u * (delta / sqrt2) * PE
    I3r1 = K_u * djrdI3 - s_u * (delta / sqrt2) * PI  # u-branch I3 has a leading minus
    aphi_u = K_u * djrdLz - s_u * (Lz / delta / sqrt2) * PL
    # ---- v-branch: the v anomaly spans the FULL loop [vmin, pi-vmin], and
    # chi(pi-v) = pi - chi(v), so the eight t^2 leaves collapse to one partial
    # plus the to-midplane integral (chi = pi/2).
    Dv = numpy.pi - 2.0 * vmin
    Dv_safe = xp.where(Dv > 0.0, Dv, xp.ones_like(Dv))
    chi_vx = 2.0 * xp.asin(xp.sqrt(xp.clip((vx - vmin) / Dv_safe, 0.0, 1.0)))
    mid_v_pt = vmin + 0.5 * (pi / 2.0 - vmin)
    low_v = (vx < mid_v_pt) | (vx > (pi - mid_v_pt))
    above = vx > pi / 2.0
    # low_v panels integrate from the vmin turning point; by the pi/2 symmetry the
    # `above` ones are the mirrored partial, i.e. anomaly pi - chi(vx)
    chimax_v = xp.where(low_v & above, numpy.pi - chi_vx, chi_vx)
    vweights = (
        lambda xp, v: xp.sin(v) ** 2.0,
        lambda xp, v: xp.ones_like(v),
        lambda xp, v: 1.0 / xp.sin(v) ** 2.0,
    )
    if torus:
        cols = tuple(
            a[..., None] if getattr(a, "ndim", 0) else a for a in (s["u0"], delta)
        )
        vweights += (
            lambda xp, v: _staeckel_dS_flat(
                xp, lambda q, u0, d: _dVdu0(q, u0, d, pot), v, cols
            ),
        )
    if ddelta is not None:
        vweights += (
            lambda xp, q: (
                2.0 * _staeckel_dS_flat(xp, JZsq, q, vcols)
                + Lcol**2.0 / dcol**2.0 / xp.sin(q) ** 2.0
                + dcol
                * _staeckel_dS_flat(
                    xp, lambda qq, u0, d: _dVddelta(qq, u0, d, pot), q, cols
                )
            ),
        )
    part_v = _staeckel_chi_profiles(
        xp, JZsq, _dJzStaeckelIntegrandSquareddv, jz_args, vmin, Dv, order,
        vweights, chimax=chimax_v[..., None], vside=True,
    )  # fmt: skip
    half_v = _staeckel_chi_profiles(
        xp, JZsq, _dJzStaeckelIntegrandSquareddv, jz_args, vmin, Dv, order,
        vweights, chimax=numpy.pi / 2.0, vside=True,
    )  # fmt: skip
    QE, QI, QL, *QV = (
        xp.where(low_v, pv, xp.where(above, pv - hv, hv - pv))
        for pv, hv in zip(part_v, half_v)
    )
    pos_v = pvx > 0.0
    K_v = xp.where(
        low_v,
        xp.where(pos_v, xp.where(above, pi, 0.0), xp.where(above, pi, 2.0 * pi)),
        xp.where(pos_v, pi / 2.0, 1.5 * pi),
    ) * xp.ones_like(vx)
    s_v = xp.where(
        low_v,
        xp.where(pos_v, xp.where(above, -1.0, 1.0), xp.where(above, 1.0, -1.0)),
        xp.where(pos_v, xp.where(above, 1.0, -1.0), xp.where(above, -1.0, 1.0)),
    )
    Or2 = K_v * djzdE + s_v * (delta / sqrt2) * QE
    I3r2 = K_v * djzdI3 + s_v * (delta / sqrt2) * QI  # v-branch I3: NO leading minus
    phitmp = K_v * djzdLz - s_v * (Lz / delta / sqrt2) * QL
    # ---- assembly (calcAnglesStaeckel)
    Or_sum = Or1 + Or2
    I3_sum = I3r1 + I3r2
    angler = Omegar * Or_sum + dI3dJR * I3_sum
    anglez = Omegaz * Or_sum + dI3dJz * I3_sum + pi / 2.0
    anglephi = aphi_u + phitmp + Omegaphi * Or_sum + dI3dLz * I3_sum
    if torus:
        djzdu0 = half_v[3] * sqrt2 * delta / pi
        # K = dS/du0 at fixed position and actions, from the same origin
        # as the (unwrapped) vertical angle
        K = (
            K_v * djzdu0
            + s_v * (delta / sqrt2) * QV[0]
            - (Omegaz * Or_sum + dI3dJz * I3_sum) * djzdu0
        )
        # (E, I_3) move with u0 at fixed actions as -(dE/dJz, dI3/dJz) djzdu0
        if du0 is not None:
            # (a rule that the caller evaluates: its partials are handed in)
            rE, rI, rL, rD = du0
        elif torus == 3:
            rE, rI, rL, rD = _u0planar_partials(xp, s, pot, delta)
        else:
            rE, rI, rL, rD = _u0torus_partials(xp, s, umin, umax, pot, delta)
        dEdu0 = -Omegaz * djzdu0
        norm = 1.0 - rE * dEdu0 + rI * dI3dJz * djzdu0
        du0 = xp.stack(
            [
                rE * Omegar + rI * dI3dJR,
                rE * Omegaphi + rI * dI3dLz + rL,
                rE * Omegaz + rI * dI3dJz,
            ]
        )
        dOm = 0.0
        if ddelta is not None:
            djrdD = full_u[3] / pi / sqrt2
            djzdD = half_v[4] * sqrt2 / pi
            thr = Omegar * Or_sum + dI3dJR * I3_sum
            thz = Omegaz * Or_sum + dI3dJz * I3_sum
            sinhux, sinvx = xp.sinh(ux), xp.sin(vx)
            KD = (
                -(sinhux * xp.cosh(ux) * pux - sinvx * xp.cos(vx) * pvx)
                / (sinhux**2.0 + sinvx**2.0)
                / delta
                + K_u * djrdD
                + s_u * PW[0] / sqrt2
                + K_v * djzdD
                + s_v * QV[1] / sqrt2
                - thr * djrdD
                - thz * djzdD
            )
            dEdD = -(Omegar * djrdD + Omegaz * djzdD)
            dI3dD = -(dI3dJR * djrdD + dI3dJz * djzdD)
            feed = rE * dEdD + rI * dI3dD + rD
            if len(ddelta) == 4:
                # the partials of the focal length's rule with respect to
                # (E, I_3, L_z, delta) instead: its derivatives in the
                # actions are those of the fixed point of the two rules
                # along the family, on which the energy and the third
                # integral move with the actions and with both parameters
                # of the model (a linear system in the derivatives of the
                # focal length and the reference curve)
                qE, qI, qL, qD = ddelta
                mdu = qE * dEdu0 - qI * dI3dJz * djzdu0
                ddelta = (
                    norm
                    * xp.stack(
                        [
                            qE * Omegar + qI * dI3dJR,
                            qE * Omegaphi + qI * dI3dLz + qL,
                            qE * Omegaz + qI * dI3dJz,
                        ]
                    )
                    + mdu * du0
                ) / (norm * (1.0 - qE * dEdD - qI * dI3dD - qD) - feed * mdu)
            du0 = du0 + feed * ddelta
            dOm = dEdD * ddelta
            angler, anglephi, anglez = (
                a + KD * d for a, d in zip((angler, anglephi, anglez), ddelta)
            )
        du0 = du0 / norm
        dOm = dOm + dEdu0 * du0
        angler, anglephi, anglez = (
            a + K * d for a, d in zip((angler, anglephi, anglez), du0)
        )
    angler = xp.remainder(angler, 2.0 * pi)  # fmod + non-negative wrap == remainder
    anglez = xp.remainder(anglez, 2.0 * pi)
    zeros = xp.zeros_like(angler)
    circ_full = circ | planar  # both degeneracies -> C calcAngles returns 0
    angler = xp.where(circ_full, zeros, angler)
    anglez = xp.where(circ_full, zeros, anglez)
    anglephi = xp.where(circ_full, zeros, anglephi)
    if torus:
        return angler, anglephi, anglez, dOm
    return angler, anglephi, anglez


def _staeckel_actions_freqs_angles(
    xp,
    R,
    vR,
    vT,
    z,
    vz,
    phi,
    pot,
    delta,
    order,
    flag_unbound=False,
    u0=None,
    torus=False,
    ddelta=None,
    du0=None,
):
    """Unified vectorised (jr,Lz,jz,Omegar,Omegaphi,Omegaz,angler,anglephi,anglez);
    setup + turning points computed once and shared. anglephi includes the azimuth."""
    s, umin, umax, vmin, delta = _staeckel_prep(
        xp, R, vR, vT, z, vz, pot, delta, flag_unbound, u0=u0
    )
    jr, jz = _staeckel_jr_jz(xp, s, umin, umax, vmin, pot, delta, order)
    # The six Leibniz derivative panels are shared by the frequencies and the
    # angles -- compute once and thread into both (else each recomputes all six).
    jac = _staeckel_jacobian(xp, s, umin, umax, vmin, pot, delta, order)
    Omegar, Omegaphi, Omegaz = _staeckel_freqs(
        xp, s, umin, umax, vmin, pot, delta, order, jac=jac
    )
    angler, anglephi, anglez, *dOm = _staeckel_angles(
        xp, s, umin, umax, vmin, pot, delta, order, jac, torus, ddelta, du0
    )
    if torus:
        Omegar, Omegaphi, Omegaz = (
            Omegar + dOm[0][0],
            Omegaphi + dOm[0][1],
            Omegaz + dOm[0][2],
        )
    anglephi = xp.remainder(anglephi + phi, 2.0 * numpy.pi)  # fold in the azimuth
    outs = (jr, jz, Omegar, Omegaphi, Omegaz, angler, anglephi, anglez)
    Lz, outs = _staeckel_flag_unbound(xp, s, outs)
    return (outs[0], Lz) + outs[1:]


def _staeckel_circular_freqs(xp, pot, R, jr, jz, Omegar, Omegaphi, Omegaz):
    """The frequencies with those of close-to-circular orbits, NaN (det(A)=0),
    replaced by epifreq/omegac/verticalfreq (as the C wrapper does). On numpy
    only those orbits are evaluated, one by one, so that a potential whose
    methods take scalars only works"""
    indx = (xp.isnan(Omegar) & (jr < 1e-3)) | (xp.isnan(Omegaz) & (jz < 1e-3))
    if xp is numpy:
        if numpy.any(indx):
            Omegar[indx] = [epifreq(pot, r, use_physical=False) for r in R[indx]]
            Omegaphi[indx] = [omegac(pot, r, use_physical=False) for r in R[indx]]
            Omegaz[indx] = [verticalfreq(pot, r, use_physical=False) for r in R[indx]]
        return Omegar, Omegaphi, Omegaz
    return (
        xp.where(indx, epifreq(pot, R, use_physical=False), Omegar),
        xp.where(indx, omegac(pot, R, use_physical=False), Omegaphi),
        xp.where(indx, verticalfreq(pot, R, use_physical=False), Omegaz),
    )


def _staeckel_c_grad_actions(pot, delta, R, vR, vT, z, vz, u0, order, useu0=False):
    """Differentiable (jr, jz) via the C-native Staeckel action Jacobian.

    For jax/torch inputs, wraps the compiled 2x5 d(jr,jz)/d(R,vR,vT,z,vz) C entry
    (actionAngleStaeckel_actionsJac_c) in the backend custom_vjp / autograd.Function
    (galpy.backend._{jax,torch}.staeckel_c): the forward is the plain round-trip C
    action value; the backward is a matvec of the C-computed Jacobian. numpy inputs
    never reach here. delta is a fixed reference (no gradient). u0 is a fixed
    reference too when a user kwarg (useu0=False); when it is the calcu0(E,Lz)
    reference (useu0=True) the C Jacobian adds the exact dJ/du0*du0/dx term.
    First-order only."""
    from ..orbit.integrateFullOrbit import _parse_pot

    _parse_pot(
        pot, potforactions=True
    )  # eager: surface unsupported-pot NotImplementedError outside the jax pure_callback (matches the numpy path)
    delta_np = numpy.atleast_1d(
        numpy.asarray(to_host(stop_gradient(delta)), dtype=numpy.float64)
    )
    u0_np = (
        None
        if u0 is None
        else numpy.asarray(to_host(stop_gradient(u0)), dtype=numpy.float64)
    )

    def host_jac(Rn, vRn, vTn, zn, vzn):
        jr, jz, jac, err = actionAngleStaeckel_c.actionAngleStaeckel_actionsJac_c(
            pot, delta_np, Rn, vRn, vTn, zn, vzn, u0=u0_np, order=order, useu0=useu0
        )
        return jr, jz, jac

    def host_val(Rn, vRn, vTn, zn, vzn):  # the same values, without the Jacobian
        return actionAngleStaeckel_c.actionAngleStaeckel_c(
            pot, delta_np, Rn, vRn, vTn, zn, vzn, u0=u0_np, order=order
        )[:2]

    name = name_of_namespace(get_namespace(R, vR, vT, z, vz))
    if name == "jax":
        from ..backend._jax.staeckel_c import actions_with_jac

        return actions_with_jac(host_jac, host_val, (R, vR, vT, z, vz))
    if name == "torch":
        from ..backend._torch.staeckel_c import actions_with_jac

        return actions_with_jac(host_jac, host_val, R, vR, vT, z, vz)
    raise NotImplementedError(  # pragma: no cover
        "C-native Staeckel action gradients require a jax or torch input array."
    )


def _staeckel_c_grad_ecczmax(pot, delta, R, vR, vT, z, vz, u0, useu0=False):
    """Differentiable (e,zmax,rperi,rap) via the C-native Staeckel EccZmax
    Jacobian.

    For jax/torch inputs, wraps the compiled 4x5 d(e,zmax,rperi,rap)/d(R,vR,vT,z,vz)
    C entry (actionAngleStaeckel_EccZmaxRperiRapJac_c) in the backend custom_vjp /
    autograd.Function: the forward is the round-trip C value, the backward a matvec
    of the C-computed Jacobian. numpy inputs never reach here. delta is a fixed
    reference (a differentiated one runs the c=False path, see _use_c).
    First-order only."""
    from ..orbit.integrateFullOrbit import _parse_pot

    _parse_pot(
        pot, potforactions=True
    )  # eager: surface unsupported-pot NotImplementedError outside the jax pure_callback (matches the numpy path)
    delta_np = numpy.atleast_1d(
        numpy.asarray(to_host(stop_gradient(delta)), dtype=numpy.float64)
    )
    u0_np = (
        None
        if u0 is None
        else numpy.asarray(to_host(stop_gradient(u0)), dtype=numpy.float64)
    )

    def host_jac(Rn, vRn, vTn, zn, vzn):
        e, zm, rp, ra, jac, err = (
            actionAngleStaeckel_c.actionAngleStaeckel_EccZmaxRperiRapJac_c(
                pot, delta_np, Rn, vRn, vTn, zn, vzn, u0=u0_np, useu0=useu0
            )
        )
        return e, zm, rp, ra, jac

    name = name_of_namespace(get_namespace(R, vR, vT, z, vz))
    if name == "jax":
        from ..backend._jax.staeckel_c import ecczmax_with_jac

        return ecczmax_with_jac(host_jac, (R, vR, vT, z, vz))
    if name == "torch":
        from ..backend._torch.staeckel_c import ecczmax_with_jac

        return ecczmax_with_jac(host_jac, R, vR, vT, z, vz)
    raise NotImplementedError(  # pragma: no cover
        "C-native Staeckel EccZmax gradients require a jax or torch input array."
    )


def _staeckel_c_grad_actionsfreqs(pot, delta, R, vR, vT, z, vz, u0, order, useu0=False):
    """Differentiable (jr,jz,Omegar,Omegaphi,Omegaz) via the fused C-native (5x5)
    Staeckel Jacobian (actionsFreqsJac_c): the jr,jz rows are the #1051 action
    Jacobian; the Omega rows are the analytic action-Hessian composition (#131).
    One C pass (setup + turning points + derivative integrals shared). numpy never
    reaches here. First-order only; close-to-circular/planar frequency VALUES get
    the epifreq/omegac/verticalfreq substitution (host) with their Jacobian rows
    zeroed in C."""
    from ..orbit.integrateFullOrbit import _parse_pot

    _parse_pot(
        pot, potforactions=True
    )  # eager: surface unsupported-pot NotImplementedError outside the jax pure_callback (matches the numpy path)
    delta_np = numpy.atleast_1d(
        numpy.asarray(to_host(stop_gradient(delta)), dtype=numpy.float64)
    )
    u0_np = (
        None
        if u0 is None
        else numpy.asarray(to_host(stop_gradient(u0)), dtype=numpy.float64)
    )

    def host_jac(Rn, vRn, vTn, zn, vzn):
        jr, jz, Or, Op, Oz, jac, err = (
            actionAngleStaeckel_c.actionAngleStaeckel_actionsFreqsJac_c(
                pot, delta_np, Rn, vRn, vTn, zn, vzn, u0=u0_np, order=order, useu0=useu0
            )
        )
        Or, Op, Oz = _staeckel_c_freq_circ_fix(pot, Rn, jr, jz, Or, Op, Oz)
        return jr, jz, Or, Op, Oz, jac

    def host_val(Rn, vRn, vTn, zn, vzn):  # the same values, without the Jacobian
        jr, jz, Or, Op, Oz, err = actionAngleStaeckel_c.actionAngleFreqStaeckel_c(
            pot, delta_np, Rn, vRn, vTn, zn, vzn, u0=u0_np, order=order
        )
        return jr, jz, *_staeckel_c_freq_circ_fix(pot, Rn, jr, jz, Or, Op, Oz)

    name = name_of_namespace(get_namespace(R, vR, vT, z, vz))
    if name == "jax":
        from ..backend._jax.staeckel_c import actionsfreqs_with_jac

        return actionsfreqs_with_jac(host_jac, host_val, (R, vR, vT, z, vz))
    if name == "torch":
        from ..backend._torch.staeckel_c import actionsfreqs_with_jac

        return actionsfreqs_with_jac(host_jac, host_val, R, vR, vT, z, vz)
    raise NotImplementedError(  # pragma: no cover
        "C-native Staeckel freq gradients require a jax or torch input array."
    )


def _staeckel_c_grad_actionsfreqsangles(
    pot, delta, R, vR, vT, z, vz, phi, u0, order, useu0=False
):
    """Differentiable (jr,jz,Omegar,Omegaphi,Omegaz,angler,anglephi,anglez) via the
    fused C-native Staeckel Jacobian (actionsFreqsAnglesJac_c): the jr,jz+Omega rows
    are #1051/#131-PR-A; the angle rows compose the action Hessians through the SAME
    dP/dcoord chain PLUS the current-position boundary term (#131 PR-B). phi enters
    analytically (d anglephi/dphi==1) via the backend tie. numpy never reaches here.
    First-order only; near-turning-point angle-Jacobian rows are zeroed in C (the AA
    turning-point-edge convention)."""
    from ..orbit.integrateFullOrbit import _parse_pot

    _parse_pot(
        pot, potforactions=True
    )  # eager: surface unsupported-pot NotImplementedError outside the jax pure_callback (matches the numpy path)
    delta_np = numpy.atleast_1d(
        numpy.asarray(to_host(stop_gradient(delta)), dtype=numpy.float64)
    )
    u0_np = (
        None
        if u0 is None
        else numpy.asarray(to_host(stop_gradient(u0)), dtype=numpy.float64)
    )

    def host_jac(Rn, vRn, vTn, zn, vzn):
        (jr, jz, Or, Op, Oz, angler, anglephi, anglez, ojac, ajac, err) = (
            actionAngleStaeckel_c.actionAngleStaeckel_actionsFreqsAnglesJac_c(
                pot, delta_np, Rn, vRn, vTn, zn, vzn, u0=u0_np, order=order, useu0=useu0
            )
        )
        Or, Op, Oz = _staeckel_c_freq_circ_fix(pot, Rn, jr, jz, Or, Op, Oz)
        jac = numpy.concatenate((ojac, ajac), axis=1)  # (N,8,5): ojac(5,5) + ajac(3,5)
        return jr, jz, Or, Op, Oz, angler, anglephi, anglez, jac

    name = name_of_namespace(get_namespace(R, vR, vT, z, vz))
    if name == "jax":
        from ..backend._jax.staeckel_c import actionsfreqsangles_with_jac

        return actionsfreqsangles_with_jac(host_jac, (R, vR, vT, z, vz), phi)
    if name == "torch":
        from ..backend._torch.staeckel_c import actionsfreqsangles_with_jac

        return actionsfreqsangles_with_jac(host_jac, R, vR, vT, z, vz, phi)
    raise NotImplementedError(  # pragma: no cover
        "C-native Staeckel angle gradients require a jax or torch input array."
    )


def _staeckel_c_backend_refu0(pot, delta, R, vR, vT, z, vz, useu0, u0_kwarg):
    """Reference u0 (numpy) for the C-native backend path, plus a flag marking
    whether it is the coordinate-dependent calcu0(E,Lz) reference (useu0=True,
    no kwarg) -> the C Jacobian then adds the exact dJ/du0*du0/dx term. An
    explicit u0-kwarg is a fixed reference (du0/dx=0); if neither, returns
    (None, False) and the C uses ux (du0/dx=dux/dx)."""
    if u0_kwarg is not None:
        return numpy.asarray(
            to_host(stop_gradient(u0_kwarg)), dtype=numpy.float64
        ), False
    if not useu0:
        return None, False
    Rn, vRn, vTn, zn, vzn = (
        numpy.atleast_1d(numpy.asarray(to_host(stop_gradient(c)), dtype=numpy.float64))
        for c in (R, vR, vT, z, vz)
    )
    E = numpy.array(
        to_host(
            [
                _evaluatePotentials(pot, Rn[ii], zn[ii])
                + vRn[ii] ** 2.0 / 2.0
                + vzn[ii] ** 2.0 / 2.0
                + vTn[ii] ** 2.0 / 2.0
                for ii in range(len(Rn))
            ]
        )
    )
    return (
        actionAngleStaeckel_c.actionAngleStaeckel_calcu0(
            E, Rn * vTn, pot, numpy.asarray(to_host(stop_gradient(delta)))
        )[0],
        True,
    )


def _staeckel_c_freq_circ_fix(pot, Rn, jrn, jzn, Or, Op, Oz):
    """Close-to-circular/planar frequency substitution (numpy host mirror of the
    C-wrapper adjustment): NaN freqs at small jr/jz -> epifreq/omegac/verticalfreq."""
    indx = numpy.isnan(Or) * (jrn < 1e-3) + numpy.isnan(Oz) * (jzn < 1e-3)
    if numpy.sum(indx) > 0:
        Or[indx] = to_host([epifreq(pot, r, use_physical=False) for r in Rn[indx]])
        Op[indx] = to_host([omegac(pot, r, use_physical=False) for r in Rn[indx]])
        Oz[indx] = to_host([verticalfreq(pot, r, use_physical=False) for r in Rn[indx]])
    return Or, Op, Oz


class actionAngleStaeckel(actionAngle):
    """Action-angle formalism for axisymmetric potentials using Binney (2012)'s Staeckel approximation"""

    def __init__(self, *args, **kwargs):
        """
        Initialize an actionAngleStaeckel object.

        Parameters
        ----------
        pot : potential or a combined potential formed using addition (pot1+pot2+…) (3D)
            The potential or a combined potential formed using addition (pot1+pot2+…).
        delta : float or Quantity
            The focal length of the prolate spheroidal coordinate system; may
            be left out for a potential that supplies its own (an exact
            Staeckel potential such as KuzminKutuzovStaeckelPotential, or an
            OblateStaeckelWrapperPotential), whose focal length is then used.
        useu0 : bool, optional
            Use u0 to calculate dV (not recommended). Default is False.
        c : bool, optional
            If True, always use C for calculations. Default is False. With jax/torch inputs, derivatives w.r.t. the phase-space coordinates are C-native; a derivative C cannot carry -- w.r.t. a differentiated potential parameter or delta -- falls back to the backend c=False implementation (as a C orbit integration falls back to the in-backend ODE), which agrees with C to ~1e-8.
        order : int, optional
            Number of points to use in the Gauss-Legendre numerical integration of the relevant action, frequency, and angle integrals. Default is 10.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - Use actionAngleStaeckel.from_inverse to set up the transformation that an actionAngleStaeckelInverse instance inverts.
        - 2012-11-27 - Started - Bovy (IAS).
        """
        self._inverse = kwargs.get("_inverse", None)
        # whether the energy is that of the Staeckel model of (delta, u0)
        self._model = False
        self._polish = kwargs.get("polish", False)
        if self._polish and not (self._inverse is not None and self._inverse._interp):
            raise ValueError("polish= requires a family (setup_interp=True)")
        if self._inverse is not None:
            kwargs = self._setup_inverse(kwargs)
        actionAngle.__init__(self, ro=kwargs.get("ro", None), vo=kwargs.get("vo", None))
        if not "pot" in kwargs:  # pragma: no cover
            raise OSError("Must specify pot= for actionAngleStaeckel")
        self._pot = _check_potential_list_and_deprecate(kwargs["pot"])
        if self._pot == MWPotential:
            warnings.warn(
                "Use of MWPotential as a Milky-Way-like potential is deprecated; galpy.potential.MWPotential2014, a potential fit to a large variety of dynamical constraints (see Bovy 2015), is the preferred Milky-Way-like potential in galpy",
                galpyWarning,
            )
        # a potential that supplies its own focal length (an exact Staeckel
        # potential such as KuzminKutuzovStaeckelPotential, or an
        # OblateStaeckelWrapperPotential) does not need delta=
        delta = kwargs.pop("delta", _focal_length_of(self._pot))
        if delta is None:
            raise OSError(
                "Must specify delta= for actionAngleStaeckel, unless the "
                "potential supplies its own focal length (an exact "
                "Staeckel potential such as KuzminKutuzovStaeckelPotential, "
                "or an OblateStaeckelWrapperPotential)"
            )
        if ext_loaded and (("c" in kwargs and kwargs["c"]) or not "c" in kwargs):
            self._c = _check_c(self._pot)
            if "c" in kwargs and kwargs["c"] and not self._c:
                warnings.warn(
                    "C module not used because potential does not have a C implementation",
                    galpyWarning,
                )  # pragma: no cover
        else:
            self._c = False
        self._useu0 = kwargs.get("useu0", False)
        self._order = kwargs.get("order", 10)
        self._delta = _coerce_delta_arraylike(
            conversion.parse_length(delta, ro=self._ro)
        )
        # Check the units
        self._check_consistent_units()
        return None

    @classmethod
    def from_inverse(cls, aAI, polish=False, **kwargs):
        """
        Set up the transformation (x,v) -> (J,theta) that an actionAngleStaeckelInverse instance inverts.

        Parameters
        ----------
        aAI : actionAngleStaeckelInverse instance
            The inverse transformation, which supplies the potential and the focal length. Each phase-space point is transformed in the Staeckel model of the torus of the instance that passes through it: the instance's single model or, for a reference curve that follows the tori (u0='torus'), the model whose reference curve is that of the point's own torus and, for a focal length that follows the tori (delta='fit' or a callable), whose focal length is the family's at that torus's actions. For a family (setup_interp=True), these are the exact actions of the family's tori, to which the family's interpolated map agrees to its interpolation error. Points beyond the family's grid are transformed in the same way: the rules for the reference curve do not depend on the grid, and a fitted focal length (delta='fit') is held at its value at the edge of the grid's range of energy and angular momentum, with a warning. Explicit tori whose focal lengths follow a rule (delta= a callable or a family's delta method) are transformed with that rule, like a family's. Points that no torus passes through return nan with a warning: unbound points and, close to the symmetry axis, points in the gaps that a reference curve or focal length that follows the tori leaves between them. With jax/torch inputs, the transformation runs in the backend (c=False) implementation, with derivatives through each point's model.
        polish : bool, optional
            For a family (setup_interp=True): make the transformation the exact inverse of the family's interpolated map, by Newton's method on that map from the actions and angles of the family's tori; points that no torus of the family passes through return nan with a warning. Default is False: the actions and angles of the family's tori, to which the interpolated map agrees to its interpolation error.
        **kwargs
            Other keywords of actionAngleStaeckel (c=, order=, ro=, vo=); pot= and delta= cannot be given.

        Returns
        -------
        actionAngleStaeckel instance
            Such that aAI(J,theta) returns (x,v) for its (J,theta) at (x,v).

        Notes
        -----
        - 2026-11-24 - Written as a class method (was the inverse= keyword) - Bovy (UofT)

        """
        return cls(_inverse=aAI, polish=polish, **kwargs)

    def _setup_inverse(self, kwargs):
        """The set-up that inverts the actionAngleStaeckelInverse instance
        inverse: its Staeckel model or, for tori that each have their own
        model, the potential that those models approximate, with the
        energy of every point evaluated in its own model"""
        aAI = self._inverse
        if "pot" in kwargs or "delta" in kwargs:
            raise TypeError(
                "pot= and delta= cannot be given to from_inverse: the inverse supplies them"
            )
        # the rule of the reference curve, as the C code numbers them: 2,
        # u0='torus', 3, 'planar', and 5, 'shell', which it evaluates
        # itself; 4, a callable, evaluated here
        rules = {"torus": 2, "planar": 3, "shell": 5}
        self._model = aAI._adaptive * (
            rules.get(aAI._u0_rule, 4) if isinstance(aAI._u0_rule, str) else 4
        )
        kwargs = dict(kwargs)
        kwargs["pot"] = aAI._chart_pot if aAI._adaptive else aAI._staeckelwrap
        kwargs["delta"] = aAI._delta
        if aAI._delta_rule is not None:
            # (a placeholder: every point gets the focal length of its torus)
            if not aAI._interp and aAI._delta_callable is None:
                raise RuntimeError(
                    "The callable delta= of the explicit tori was not "
                    "pickled with them: they can only be inverted by the "
                    "instance that was set up with the callable"
                )
            kwargs["delta"] = float(
                numpy.median(aAI._A_tab[..., 3] if aAI._interp else aAI._deltas)
            )
        if aAI._roSet:
            kwargs.setdefault("ro", aAI._ro)
        if aAI._voSet:
            kwargs.setdefault("vo", aAI._vo)
        return kwargs

    def _polished(self, *args, **kwargs):
        """(J_R, L_z, J_z, Omega_R, Omega_phi, Omega_z, theta_R, theta_phi,
        theta_z) such that the family inverse maps the actions and angles
        to the points exactly: Newton's method on the family's interpolated
        map, from those of its tori, with the family's frequencies"""
        aAI = self._inverse
        if len(args) == 5 or len(args) == 6:
            R, vR, vT, z, vz = args[:5]
            phi = args[5] if len(args) == 6 else 0.0
        else:
            self._eval_phi = 0.0
            self._parse_eval_args(*args)
            R, vR, vT = self._eval_R, self._eval_vR, self._eval_vT
            z, vz, phi = self._eval_z, self._eval_vz, self._eval_phi
        xp = get_namespace(R, vR, vT, z, vz, phi)
        if xp is not numpy:
            R, vR, vT, z, vz, phi = xp.broadcast_arrays(
                *(
                    xp.reshape(a * 1.0, (-1,))
                    for a in promote_scalars(xp, R, vR, vT, z, vz, phi)
                )
            )
            return self._bk_polished(xp, R, vR, vT, z, vz, phi, **kwargs)
        R, vR, vT, z, vz, phi = (
            a.copy()
            for a in numpy.broadcast_arrays(
                *(numpy.atleast_1d(a) * 1.0 for a in (R, vR, vT, z, vz, phi))
            )
        )
        out = numpy.array(
            self._actionsFreqsAngles(R, vR, vT, z, vz, phi, _direct=True, **kwargs)
        )
        out[1] = R * vT
        ok = numpy.all(numpy.isfinite(out[[0, 2]]), axis=0)
        sol = numpy.full((5, len(R)), numpy.nan)
        if numpy.any(ok):
            sol[:, ok] = aAI._invert_family(
                R[ok], vR[ok], vT[ok], z[ok], vz[ok], phi[ok], *out[[0, 2]][:, ok]
            )
        out[[0, 2, 6, 7, 8]] = sol
        bad = numpy.isnan(sol[0])
        if numpy.any(bad):
            warnings.warn(
                f"{numpy.sum(bad)} points are not on a torus of the inverse: their "
                "actions, frequencies and angles are nan",
                galpyWarning,
            )
        good = ~bad
        out[3:6, bad] = numpy.nan
        if numpy.any(good):
            out[3:6, good] = aAI._Freqs(*out[:3][:, good])
        return tuple(out)

    def _bk_polished(self, xp, R, vR, vT, z, vz, phi, **kwargs):
        """_polished on jax/torch: the family's _bk_invert_family from the
        actions of the transformation without the polish, with the family's
        frequencies at the polished actions; nan (with the same warning) for
        the points that no torus of the family passes through"""
        aAI = self._inverse
        n = R.shape[0]
        if n < 16 and not under_jax_trace(R, vR, vT, z, vz, phi):
            if "jax" in name_of_namespace(xp):
                # eager jax compiles every op per new number of points: a few
                # points are padded to 16 (copies of the last), dropped below
                R, vR, vT, z, vz, phi = pad_edge(
                    xp, bucket_size(n), R, vR, vT, z, vz, phi
                )
        # (the start carries no derivative)
        out = self._actionsFreqsAngles(
            *(stop_gradient(a) for a in (R, vR, vT, z, vz, phi)),
            _direct=True,
            _npts=n,
            **kwargs,
        )
        Lz = R * vT
        sol = aAI._bk_invert_family(xp, R, vR, vT, z, vz, phi, out[0], out[2])
        bad = xp.isnan(sol[0])
        nbad = xp.sum(xp.astype(bad[:n], xp.int64))
        concrete = has_concrete_truth_value(nbad)
        if concrete and int(nbad) > 0:
            warnings.warn(
                f"{int(nbad)} points are not on a torus of the inverse: their "
                "actions, frequencies and angles are nan",
                galpyWarning,
            )
        # the family's frequencies at the polished actions (a solved point's
        # stand-in for the others; the padding copies the last point)
        if concrete and int(nbad) == n:
            Om = (sol[0],) * 3
        else:
            jr, jz, L = sol[0], sol[1], Lz
            if concrete and int(nbad) > 0:
                i = int(numpy.flatnonzero(~as_numpy(bad))[0])
                jr, jz, L = (xp.where(bad, t[i], t) for t in (jr, jz, L))
            Om = tuple(xp.where(bad, numpy.nan, o) for o in aAI._Freqs(jr, L, jz))
        out = (sol[0], Lz, sol[1], *Om, sol[2], sol[3], sol[4])
        return tuple(o[:n] for o in out)

    def _inverse_model(
        self, R, vR, vT, z, vz, c=True, deriv=True, flag_unbound=False, npts=None
    ):
        """The focal length and reference curve of the Staeckel model of
        the torus of the inverse through each phase-space point. A reference
        curve that follows the tori depends on the torus's energy and third
        integral in its own model, which depend on the reference curve: a
        fixed point, found by iterating the rule on the turning points in
        the current model with Aitken's acceleration and, where that does
        not converge, by bracketing: by the C code itself (which is then
        handed nan to start from the point's own u) or, with c False, here;
        on jax/torch by _bk_inverse_model (where flag_unbound models unbound
        points as C does, on a bound stand-in, and only the first npts points
        count in its warning). A point without a fixed point is on no torus:
        its u0 is nan"""
        aAI = self._inverse
        delta = aAI._delta
        xp = get_namespace(R, vR, vT, z, vz)
        if not self._model:
            if xp is not numpy:
                return delta, aAI._staeckelwrap._u0 + 0.0 * R, None
            return delta, numpy.full(numpy.shape(R), aAI._staeckelwrap._u0), None
        if xp is not numpy:
            return self._bk_inverse_model(
                xp, R, vR, vT, z, vz, delta, deriv, flag_unbound, npts
            )
        R, vR, vT, z, vz = (
            a.copy()
            for a in numpy.broadcast_arrays(
                *(numpy.atleast_1d(a) * 1.0 for a in (R, vR, vT, z, vz))
            )
        )
        pot = self._pot if c else _staeckel_pot(numpy, self._pot)

        def rule(idx, u0, delta=delta):
            x = (R[idx], vR[idx], vT[idx], z[idx], vz[idx])
            if self._model >= 4:
                return self._inverse_rule(x, delta, u0)
            if self._model == 3:
                um, up = _staeckel_planar(
                    numpy, _staeckel_setup(numpy, *x, pot, delta, u0=u0), pot, delta
                )
            else:
                # (a point that is unbound in this model has no turning points)
                um, up, unbound = _staeckel_uminumax(
                    numpy, _staeckel_setup(numpy, *x, pot, delta, u0=u0), pot, delta
                )
                up = numpy.where(unbound, numpy.nan, up)
            sm, sp = numpy.sinh(um), numpy.sinh(up)
            return numpy.arcsinh(2.0 * sm * sp / (sm + sp))

        def solve(sel, delta=delta, u0=None):
            # the fixed point of the rule at these focal lengths, from u0
            # where that is given and finite, else from the points' own u
            d = numpy.broadcast_to(delta, sel.shape) * 1.0
            u0 = numpy.arcsinh(R[sel] / d) if u0 is None else u0 * 1.0
            start = ~numpy.isfinite(u0)
            u0[start] = numpy.arcsinh(R[sel] / d)[start]
            u0 = rule(sel, u0, d)
            act = numpy.arange(len(sel))
            for _ in range(_MODEL_MAXITER):
                u1 = rule(sel[act], u0[act], d[act])
                u2 = rule(sel[act], u1, d[act])
                den = u2 - 2.0 * u1 + u0[act]
                ok = den != 0.0
                acc = u2 - (u2 - u1) ** 2 / numpy.where(ok, den, 1.0)
                left = _fixed_point_error(u0[act], u1, u2)
                u0[act] = numpy.where(ok & numpy.isfinite(acc), acc, u2)
                act = act[left > _MODEL_TOL]
                if len(act) == 0:
                    break
            # the points that this does not converge for (a first model in
            # which the point is unbound, or a cycle between two models): by
            # bracketing
            u0[act] = numpy.nan
            act = numpy.flatnonzero(~numpy.isfinite(u0))
            if len(act) > 0:
                u0[act] = _bracket_fixed_point(
                    lambda k, u: rule(sel[act[k]], u, d[act[k]]), len(act)
                )
            return u0

        idx = numpy.arange(len(R))
        if aAI._delta_rule is not None:
            return self._inverse_model_delta(solve, R, vR, vT, z, vz, pot, c, deriv)
        if c and self._model != 4:
            return delta, numpy.full(len(R), numpy.nan), None
        return delta, solve(idx), None

    def _inverse_missing(self, delta, u0, *out):
        """out with nan for the points without a model: those that no torus
        of the inverse passes through (on jax/torch, the warning for concrete
        points only)"""
        xp = get_namespace(u0, *out)
        missing = ~xp.isfinite(delta + u0)
        if xp is not numpy:
            nmiss = xp.sum(xp.astype(missing, xp.int64))
            if has_concrete_truth_value(nmiss) and int(nmiss) > 0:
                warnings.warn(
                    f"{int(nmiss)} of the {missing.shape[0]} points are not on "
                    "a torus of the inverse (a reference curve or focal length "
                    "that follows the tori leaves gaps between them close to "
                    "the symmetry axis, and unbound points are on none): their "
                    "actions, frequencies, and angles are nan",
                    galpyWarning,
                )
            return tuple(xp.where(missing, numpy.nan, o) for o in out)
        if numpy.any(missing):
            warnings.warn(
                f"{numpy.sum(missing)} of the {len(missing)} points are not on "
                "a torus of the inverse (a reference curve or focal length that "
                "follows the tori leaves gaps between them close to the "
                "symmetry axis, and unbound points are on none): their "
                "actions, frequencies, and angles are nan",
                galpyWarning,
            )
        return tuple(numpy.where(missing, numpy.nan, o) for o in out)

    def _inverse_rule(self, x, delta, u0, partials=False):
        """One application, in the models (delta, u0) of the points x = (R,
        vR, vT, z, vz), of a rule for the reference curve that is evaluated
        here: u0='shell' of the inverse, the shell orbit of each point's energy
        and angular momentum, or its callable u0(E, |L_z|[, I_3]). With
        partials, the rule's partials with respect to (E, I_3, L_z, delta)
        instead"""
        aAI = self._inverse
        xp = get_namespace(*x)
        pot = _staeckel_pot(xp, self._pot)
        s = _staeckel_setup(xp, *x, pot, delta, u0=u0)
        if callable(aAI._u0_rule):
            return _u0callable(
                aAI._u0_rule, aAI._u0_nargs, s["E"], s["Lz"], s["I3V"], partials
            )
        ush = _u0shell(xp, s, *_staeckel_uminumax(xp, s, pot, delta)[:2], pot, delta)
        if not partials:
            return ush
        Rsh = delta * xp.sinh(ush)
        if xp is numpy:
            R2 = numpy.array(
                [evaluateR2derivs(self._pot, r, 0.0, use_physical=False) for r in Rsh]
            )
        else:
            R2 = evaluateR2derivs(self._pot, Rsh, 0.0 * Rsh, use_physical=False)
        return _u0shell_partials(xp, s, ush, pot, delta, R2)

    def _inverse_du0(self, R, vR, vT, z, vz, delta, u0, c=False):
        """The partials of the reference curve of each point's torus for a
        rule that is evaluated here (_inverse_rule: a callable, and
        u0='shell' unless for the C code, which has that rule), else None"""
        if self._model < 4 or (c and self._model == 5):
            return None
        xp = get_namespace(R, vR, vT, z, vz)
        if xp is numpy:
            x = numpy.broadcast_arrays(
                *(numpy.atleast_1d(a) * 1.0 for a in (R, vR, vT, z, vz))
            )
        else:
            x = xp.broadcast_arrays(R, vR, vT, z, vz)
        return self._inverse_rule(x, delta, u0, partials=True)

    def _inverse_model_delta(self, solve, R, vR, vT, z, vz, pot, c, deriv):
        """_inverse_model for a family whose focal length follows its tori:
        the focal length of a torus is the family's rule at the torus's
        energy, angular momentum, and fraction f of the way of its third
        integral from the planar to the shell orbit's, all in its own
        model: the fixed point in the focal length, by the same iteration,
        with the reference curve at its own fixed point (solve) at each
        focal length. For a family that does not have
        its rule (a pickled family with a callable delta=), the focal
        length is read from the family's table at the torus's actions
        instead. Also returns, for the frequencies and angles, the partials
        of the focal length's rule with respect to (E, I_3, L_z, delta) at
        each point's torus (the rule's own with respect to the energy,
        the angular momentum, and f, and those of f in the model), or, for
        a family without its rule, the derivatives of the focal length
        with respect to (J_R, L_z, J_z) from the family's table (nan for
        points whose torus the family does not hold). Explicit tori have
        their rule (that of the family whose delta method they were given,
        or their callable: _rule_family), as a family does"""
        aAI = self._inverse._rule_family
        Lz = R * vT
        pypot = _staeckel_pot(numpy, self._pot)
        edges = aAI._delta_edges(Lz)
        direct = aAI._delta_of(R[:0], R[:0], R[:0], edges[:0]) is not None

        def step(idx, u0, delta, partials=False):
            x = (R[idx], vR[idx], vT[idx], z[idx], vz[idx])
            if direct:
                if turn is not None:
                    # (at the fixed point that the C code found, with its
                    # turning points)
                    um, up = turn[0][idx], turn[1][idx]
                    s = _staeckel_setup(numpy, *x, pypot, delta, u0=u0)
                elif c and self._model != 4:
                    # (the C code moves u0 to the fixed point at this delta)
                    u0 = numpy.array(u0, dtype="float64", order="C")
                    um, up = actionAngleStaeckel_c.actionAngleUminUmaxVminStaeckel_c(
                        pot, delta, *x, u0=u0, model=self._model
                    )[:2]
                    s = _staeckel_setup(numpy, *x, pypot, delta, u0=u0)
                else:
                    u0 = solve(idx, delta, u0)
                    s = _staeckel_setup(numpy, *x, pypot, delta, u0=u0)
                    um, up = _staeckel_uminumax(numpy, s, pypot, delta)[:2]
                # the third integral above the planar orbit's (in the gauge
                # V(pi/2) = 0) and below the shell orbit's (p_u^2 at the
                # shell orbit's u); a circular orbit has no fraction
                ush = _u0shell(numpy, s, um, up, pypot, delta)
                if partials:
                    f, df = _staeckel_fraction(numpy, s, ush, pypot, delta, True)
                    _, (dE, dL, dF) = aAI._delta_of(
                        s["E"], Lz[idx], f, edges[idx], deriv=True
                    )
                    return numpy.array(
                        [
                            dE + dF * df[0],
                            dF * df[1],
                            numpy.sign(Lz[idx]) * dL + dF * df[2],
                            dF * df[3],
                        ]
                    )
                f = _staeckel_fraction(numpy, s, ush, pypot, delta)
                return u0, aAI._delta_of(s["E"], Lz[idx], f, edges[idx]), None
            if c:
                # (the C code moves u0 to the fixed point at this delta,
                # unless the rule is evaluated here)
                u0 = numpy.array(
                    solve(idx, delta, u0) if self._model == 4 else u0,
                    dtype="float64",
                    order="C",
                )
                jr, jz = actionAngleStaeckel_c.actionAngleStaeckel_c(
                    pot, delta, *x, u0=u0, order=self._order, model=self._model
                )[:2]
            else:
                u0 = solve(idx, delta, u0)
                jr, _, jz = _staeckel_actions(numpy, *x, pot, delta, self._order, u0=u0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", galpyWarning)
                # (the labels are not defined on the faces of vanishing action)
                lab = aAI._label_tori(
                    numpy.maximum(jr, 1e-10), numpy.maximum(jz, 1e-10), Lz[idx]
                )
            new = numpy.where(lab[-1], lab[6], numpy.nan)
            return u0, new, numpy.where(lab[-1][:, None], lab[7], numpy.nan)

        idx = numpy.arange(len(R))
        delta = numpy.full(len(R), self._delta * 1.0)
        ddelta = numpy.full((len(R), 3), numpy.nan)
        turn = None
        if c and self._model != 4 and aAI._delta_fit is not None:
            # the fitted rule, with a reference curve that the C code
            # evaluates: the whole fixed point in C, which is handed the
            # rule (the fit's degrees, limits, and coefficients, the grid's
            # range in ln L_z, and each point's two energy edges)
            coef, b0, b1, m = aAI._delta_fit
            L0, L1 = aAI._Lzgrid[0], aAI._Lzgrid[-1]
            dfit = numpy.concatenate(
                [
                    aAI._delta_deg,
                    [b0, b1, m, numpy.log(L0), numpy.log(L1 / L0)],
                    coef,
                    edges[:, 0],
                    edges[:, 1],
                ]
            ).astype("float64")
            u0 = numpy.full(len(R), numpy.nan)
            turn = actionAngleStaeckel_c.actionAngleUminUmaxVminStaeckel_c(
                pot, delta, R, vR, vT, z, vz, u0=u0, model=self._model, dfit=dfit
            )[:2]
            idx = idx[:0]
        else:
            u0, delta, _ = step(idx, numpy.arcsinh(R / delta), delta)
            idx = idx[numpy.isfinite(delta)]
        found = numpy.flatnonzero(numpy.isfinite(delta))
        for _ in range(_MODEL_MAXITER):
            if len(idx) == 0:
                break
            u1, d1, _ = step(idx, u0[idx], delta[idx])
            d1 = numpy.where(numpy.isfinite(d1), d1, delta[idx])
            u2, d2, dd = step(idx, u1, d1)
            left = numpy.maximum(
                _fixed_point_error(u0[idx], u1, u2),
                _fixed_point_error(delta[idx], d1, d2) / d1,
            )
            for old, x1, x2 in ((u0, u1, u2), (delta, d1, d2)):
                den = x2 - 2.0 * x1 + old[idx]
                ok = den != 0.0
                acc = x2 - (x2 - x1) ** 2 / numpy.where(ok, den, 1.0)
                old[idx] = numpy.where(ok & numpy.isfinite(acc), acc, x2)
            if not direct:
                ddelta[idx] = dd
            idx = idx[left > _MODEL_TOL]
        if turn is None and aAI._delta_fit is not None:
            # the points that this does not converge for, or that it takes
            # to a focal length where the rule is not defined: by bracketing
            # over the rule's whole range, as the C code does
            delta[idx] = numpy.nan
            act = numpy.flatnonzero(~numpy.isfinite(delta))
            nan = lambda k: numpy.full(len(k), numpy.nan)
            if len(act) > 0:
                delta[act] = _bracket_fixed_point(
                    lambda k, d: step(act[k], nan(k), d)[1],
                    len(act),
                    numpy.exp(numpy.linspace(*aAI._delta_fit[1:3], _MODEL_NDSCAN)),
                    True,
                )
                act = act[numpy.isfinite(delta[act])]
            if len(act) > 0:
                u0[act] = step(act, nan(act), delta[act])[0]
            found = numpy.flatnonzero(numpy.isfinite(delta))
        if turn is None and (not c or self._model == 4) and len(found) > 0:
            # the reference curve at its fixed point at the focal length
            # itself (the iteration's is that at the last iterate, which
            # Aitken's extrapolation of the focal length moves on from), as
            # the C code solves for it when it is handed the focal length
            u0[found] = solve(found, delta[found], u0[found])
        # (a point without a focal length has no model at all)
        u0 = numpy.where(numpy.isfinite(delta), u0, numpy.nan)
        if aAI._delta_fit is not None:
            # points beyond the tori that the rule was fitted to
            E = _staeckel_setup(numpy, R, vR, vT, z, vz, pypot, delta, u0=u0)["E"]
            off = numpy.fabs(Lz) != numpy.clip(
                numpy.fabs(Lz), aAI._Lzgrid[0], aAI._Lzgrid[-1]
            )
            for edge, sign in zip(edges.T[:2], (-1.0, 1.0)):
                off |= sign * (E - _evaluatePotentials(pypot, edge, 0.0 * edge)) > 0.0
            if numpy.any(off):
                warnings.warn(
                    f"{numpy.sum(off)} of the {len(R)} points lie outside the "
                    "range of energy or angular momentum of the grid of "
                    "the inverse: their focal lengths are those of the family's "
                    "rule at the edge of that range",
                    galpyWarning,
                )
        if direct:
            ddelta = numpy.full((4, len(R)), numpy.nan)
            if deriv and len(found) > 0:
                ddelta[:, found] = step(found, u0[found], delta[found], True)
            return delta, u0, ddelta
        # (the family's table is in (J_R, J_z, J_phi))
        return delta, u0, ddelta[:, [0, 2, 1]].T

    def _bk_inverse_model(
        self, xp, R, vR, vT, z, vz, delta, deriv=True, flag_unbound=False, npts=None
    ):
        """_inverse_model's fixed point on jax/torch: the same Aitken schedule
        at fixed shape on derivative-free inputs (a point is frozen once its
        step is below _MODEL_TOL), then the implicit-function step, so that
        the reference curve carries its derivative in the phase-space point"""
        pot = _staeckel_pot(xp, self._pot)
        x = tuple(xp.broadcast_arrays(R, vR, vT, z, vz))

        def rule(u0, R, vR, vT, z, vz, delta=delta):
            if self._model >= 4:
                return self._inverse_rule((R, vR, vT, z, vz), delta, u0)
            if self._model == 3:
                um, up = _staeckel_planar(
                    xp, _staeckel_setup(xp, R, vR, vT, z, vz, pot, delta, u0=u0),
                    pot, delta,
                )  # fmt: skip
            else:
                # (a point that is unbound in this model has no turning points)
                um, up, unbound = _staeckel_uminumax(
                    xp, _staeckel_setup(xp, R, vR, vT, z, vz, pot, delta, u0=u0),
                    pot, delta,
                )  # fmt: skip
                up = xp.where(unbound, numpy.nan, up)
            sm, sp = xp.sinh(um), xp.sinh(up)
            return xp.asinh(2.0 * sm * sp / (sm + sp))

        xs = tuple(stop_gradient(a) for a in x)
        if self._inverse._delta_rule is not None:
            return self._bk_inverse_model_delta(xp, x, xs, rule, pot, deriv, npts)
        u0 = self._bk_solve_u0(xp, rule, xs, delta)
        return delta, fixed_point_implicit_step(rule, stop_gradient(u0), x), None

    @staticmethod
    def _bk_solve_u0(xp, rule, xs, delta, u0=None):
        """solve of _inverse_model on jax/torch, at fixed shape on the
        derivative-free points xs: the fixed point of the rule for the
        reference curve at the focal lengths delta, from u0 where that is
        given and finite, else from the points' own u, by the Aitken
        schedule (a point is frozen once its step is below _MODEL_TOL) and,
        for the points that this does not converge for, by bracketing; nan
        for a point that no torus of the rule passes through"""
        own = xp.asinh(xs[0] / delta)
        u0 = own if u0 is None else xp.where(xp.isfinite(u0), u0, own)
        u0 = stop_gradient(rule(u0, *xs, delta=delta))

        def aitken(s):
            # (s: u0 and the mask of the points that are still iterated)
            u0, live = s[0], s[1] > 0.0
            u1 = rule(u0, *xs, delta=delta)
            u2 = rule(u1, *xs, delta=delta)
            den = u2 - 2.0 * u1 + u0
            ok = den != 0.0
            acc = u2 - (u2 - u1) ** 2 / xp.where(ok, den, 1.0)
            left = _fixed_point_error(u0, u1, u2)
            u0 = xp.where(live, xp.where(ok & xp.isfinite(acc), acc, u2), u0)
            live = live & (left > _MODEL_TOL)
            return stop_gradient(xp.stack([u0, xp.astype(live, u0.dtype)]))

        s = iterate_to_fixed_point(
            aitken,
            xp.stack([u0, 1.0 + 0.0 * u0]),
            _MODEL_MAXITER,
            done=lambda s: ~xp.any(s[1] > 0.0),
        )
        u0, live = s[0], s[1] > 0.0
        # the points that this does not converge for (a first model in which
        # the point is unbound, or a cycle between two models): by bracketing
        # (not those without a focal length, as main's, which solves only
        # those that have one)
        u0 = xp.where(live, numpy.nan, u0)
        need = ~xp.isfinite(u0) & xp.isfinite(delta + 0.0 * xs[0])
        return xp.where(
            need,
            _bk_bracket_fixed_point(
                xp,
                lambda u, *p: rule(u, *p[:5], delta=p[5]) - u,
                need,
                (*xs, delta + 0.0 * xs[0]),
            ),
            u0,
        )

    def _bk_inverse_model_delta(self, xp, x, xs, rule, pot, deriv, npts=None):
        """_inverse_model_delta on jax/torch: main's iteration in the focal
        length and the reference curve together, at fixed shape on the
        derivative-free inputs xs, then the implicit-function step of the
        two-dimensional fixed point. The focal length of a point's torus is
        the family's rule at its (E, L_z, f) (_staeckel_fraction): a fitted
        rule in the backend (_bk_delta_of), a callable on the host, linearized
        in (E, L_z, f) (_bk_host_linearized), or, for a family without its
        rule, read from its table through the
        vectorized label inversion (_bk_focal_lengths). With deriv, the
        partials of the focal length (as _inverse_model_delta), with their
        own derivatives. nan for points without a model. Explicit tori have
        their rule (_rule_family), as a family does"""
        aAI = self._inverse._rule_family
        # (a rule of its own: fitted, or a callable that was kept)
        direct = aAI._delta_fit is not None or aAI._delta_callable is not None

        def rule_delta(Q):
            # the rule at (E, L_z, f), its gradient, and that gradient's
            # central differences (FINITE DIFFERENCES of the closed form)
            def grad_at(Q):
                E, Lz, f = Q
                val, (dE, dL, dF) = aAI._delta_of(
                    E, Lz, f, aAI._delta_edges(Lz), deriv=True
                )
                return val, numpy.array([dE, numpy.sign(Lz) * dL, dF])

            val, grad = grad_at(Q)
            hess = numpy.empty((3, 3, Q.shape[1]))
            for k in range(3):
                h = numpy.zeros_like(Q)
                h[k] = 1e-5 * (1.0 + numpy.fabs(Q[k]))
                hess[:, k] = (grad_at(Q + h)[1] - grad_at(Q - h)[1]) / (2.0 * h[k])
            return val, grad, hess

        def rule_bk(E, q, f, edges, partials=False):
            if aAI._delta_fit is None:  # a callable, on the host
                return _bk_host_linearized(rule_delta, (E, q, f), partials)
            if not partials:
                return aAI._bk_delta_of(E, q, f, edges)
            _, (dE, dL, dF) = aAI._bk_delta_of(E, q, f, edges, deriv=True)
            return xp.stack([dE, xp.sign(q) * dL, dF])

        def step(u0, delta, R, vR, vT, z, vz, partials=False, edges=None, solve=True):
            # (the reference curve at its own fixed point at this focal length,
            # as main's; one step of the rule for the implicit-function step)
            u0 = (
                self._bk_solve_u0(xp, rule, (R, vR, vT, z, vz), delta, u0)
                if solve
                else rule(u0, R, vR, vT, z, vz, delta=delta)
            )
            if not direct:
                jr, _, jz = _staeckel_actions(
                    xp, R, vR, vT, z, vz, pot, delta, self._order, u0=u0
                )
                d, dd, bad = aAI._bk_focal_lengths(xp, jr, jz, R * vT)
                if bad is not None:
                    held = asarray_on_device(xp, ~bad, device_of(d))
                    d = xp.where(held, d, numpy.nan)
                    dd = xp.where(held[:, None], dd, numpy.nan)
                return u0, d, dd
            s = _staeckel_setup(xp, R, vR, vT, z, vz, pot, delta, u0=u0)
            um, up = _staeckel_uminumax(xp, s, pot, delta)[:2]
            ush = _u0shell(xp, s, um, up, pot, delta)
            q = s["Lz"]
            if partials:
                f, df = _staeckel_fraction(xp, s, ush, pot, delta, True)
                dE, dL, dF = rule_bk(s["E"], q, f, edges, True)
                return xp.stack(
                    [dE + dF * df[0], dF * df[1], dL + dF * df[2], dF * df[3]]
                )
            f = _staeckel_fraction(xp, s, ush, pot, delta)
            return u0, rule_bk(s["E"], q, f, edges), None

        # (the fitted rule's edges at the points' L_z, for every iteration)
        e0 = None
        if direct and aAI._delta_fit is not None:
            e0 = aAI._bk_delta_edges(xs[0] * xs[2])
        delta = self._delta * 1.0 + 0.0 * xs[0]
        u0, delta, _ = step(xp.asinh(xs[0] / delta), delta, *xs, edges=e0)

        def aitken(s):
            # (s: u0, delta and the mask of the points that are still iterated)
            u0, delta, live = s[0], s[1], s[2] > 0.0
            u1, d1, _ = step(u0, delta, *xs, edges=e0)
            d1 = xp.where(xp.isfinite(d1), d1, delta)
            u2, d2, _ = step(u1, d1, *xs, edges=e0)
            left = xp.maximum(
                _fixed_point_error(u0, u1, u2), _fixed_point_error(delta, d1, d2) / d1
            )
            new = []
            for old, x1, x2 in ((u0, u1, u2), (delta, d1, d2)):
                den = x2 - 2.0 * x1 + old
                ok = den != 0.0
                acc = x2 - (x2 - x1) ** 2 / xp.where(ok, den, 1.0)
                new.append(
                    xp.where(live, xp.where(ok & xp.isfinite(acc), acc, x2), old)
                )
            live = live & (left > _MODEL_TOL)
            return stop_gradient(xp.stack([*new, xp.astype(live, u0.dtype)]))

        s = iterate_to_fixed_point(
            aitken,
            stop_gradient(
                xp.stack([u0, delta, xp.astype(xp.isfinite(delta), delta.dtype)])
            ),
            _MODEL_MAXITER,
            done=lambda s: ~xp.any(s[2] > 0.0),
        )
        u0, delta, live = s[0], s[1], s[2] > 0.0
        if direct and aAI._delta_fit is not None:
            # the points that this does not converge for, or that it takes to
            # a focal length where the rule is not defined: by bracketing over
            # the rule's whole range (at each focal length, the reference
            # curve at its own fixed point), as main's _inverse_model_delta
            delta = xp.where(live, numpy.nan, delta)
            need = ~xp.isfinite(delta)

            def g(d, *p):
                return step(d + numpy.nan, d, *p)[1] - d

            lo, hi = (float(as_numpy(stop_gradient(t))) for t in aAI._delta_fit[1:3])
            delta = xp.where(
                need,
                _bk_bracket_fixed_point(
                    xp,
                    g,
                    need,
                    xs,
                    numpy.exp(numpy.linspace(lo, hi, _MODEL_NDSCAN)),
                    True,
                ),
                delta,
            )
            u0 = xp.where(need, self._bk_solve_u0(xp, rule, xs, delta), u0)
        # the reference curve at its fixed point at the focal length itself
        # (the iteration's is that at the last iterate), as main's
        u0 = self._bk_solve_u0(xp, rule, xs, delta, u0)

        def G(q, R, vR, vT, z, vz):
            u, d, _ = step(q[0], q[1], R, vR, vT, z, vz, solve=False)
            return xp.stack([u, d])

        q = fixed_point_implicit_step(
            G, stop_gradient(xp.stack([u0, delta])), x, vector=True
        )
        u0, delta = q[0], q[1]
        # (a point without a focal length has no model at all)
        u0 = xp.where(xp.isfinite(delta), u0, numpy.nan)
        if (
            direct
            and aAI._delta_fit is not None
            and has_concrete_truth_value(xp.all(u0 == u0))
        ):
            # points beyond the tori that the rule was fitted to: beyond the
            # grid's range of |L_z|, or of energy at the edges' apocentres
            E = _staeckel_setup(
                xp, *xs, pot, stop_gradient(delta), u0=stop_gradient(u0)
            )["E"]
            ell = xp.abs(xs[0] * xs[2])
            off = (
                (ell < float(aAI._Lzgrid[0]))
                | (ell > float(aAI._Lzgrid[-1]))
                | (E < e0["Phic"])
                | (E > e0["Phim"])
            )
            off = numpy.asarray(as_numpy(off))[:npts]  # (not _bk_polished's padding)
            if numpy.any(off):
                warnings.warn(
                    f"{numpy.sum(off)} of the {len(off)} points lie outside the "
                    "range of energy or angular momentum of the grid of "
                    "the inverse: their focal lengths are those of the family's "
                    "rule at the edge of that range",
                    galpyWarning,
                )
        if not deriv:
            return delta, u0, None
        dd = step(u0, delta, *x, partials=direct, solve=False)
        if direct:
            return delta, u0, dd
        # (the family's table is in (J_R, J_z, J_phi))
        dd = dd[2]
        return delta, u0, xp.stack([dd[:, 0], dd[:, 2], dd[:, 1]])

    def _c_requested(self, kwargs):
        return (
            (self._c and not ("c" in kwargs and not kwargs["c"]))
            or (ext_loaded and ("c" in kwargs and kwargs["c"]))
        ) and _check_c(self._pot)

    def _use_c(self, kwargs, delta):
        """Whether this call runs the C code: requested and available, and no
        derivative C cannot carry -- w.r.t. a potential parameter or delta. As
        Orbit.integrate falls back from a C method to the in-backend ODE, such a
        call runs the backend c=False implementation instead (equal to C to
        ~1e-8; unbound orbits flagged as C does); coordinate derivatives stay
        C-native."""
        return self._c_requested(kwargs) and not _c_cannot_carry(self._pot, delta)

    def _evaluate(self, *args, **kwargs):
        """
        Evaluate the actions (jr,lz,jz).

        Parameters
        ----------
        *args : tuple
            Either:
            a) R,vR,vT,z,vz[,phi]:
                1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)
                2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
            b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
        delta: bool, optional
            can be used to override the object-wide focal length; can also be an array with length N to allow different delta for different phase-space points
        u0: float, optional
            if object-wide option useu0 is set, u0 to use (if useu0 and useu0 is None, a good value will be computed).
        c: bool, optional
            True/False to override the object-wide setting for whether or not to use the C implementation. With jax/torch inputs, derivatives w.r.t. the phase-space coordinates are C-native; a derivative C cannot carry -- w.r.t. a differentiated potential parameter or delta -- falls back to the backend c=False implementation (as a C orbit integration falls back to the in-backend ODE), which agrees with C to ~1e-8.
        order: int, optional
            number of points to use in the Gauss-Legendre numerical integration of the relevant action integrals.

        Returns
        -------
        tuple
            (jr,lz,jz)

        Notes
        -----
        - 2012-11-27 - Written - Bovy (IAS)
        - 2017-12-27 - Allowed individual delta for each point - Bovy (UofT)
        """
        if self._polish and not kwargs.pop("_direct", False):
            return self._polished(*args, **kwargs)[:3]
        loop_c = bool(self._c and kwargs.get("c", True))
        ddelta = None
        delta = kwargs.pop("delta", self._delta)
        order = kwargs.get("order", self._order)
        if len(args) == 5:  # R,vR.vT, z, vz
            R, vR, vT, z, vz = args
        elif len(args) == 6:  # R,vR.vT, z, vz, phi
            R, vR, vT, z, vz, phi = args
        else:
            self._parse_eval_args(*args)
            R = self._eval_R
            vR = self._eval_vR
            vT = self._eval_vT
            z = self._eval_z
            vz = self._eval_vz
        if isinstance(R, float):
            R = numpy.array([R])
            vR = numpy.array([to_host(vR)])
            vT = numpy.array([to_host(vT)])
            z = numpy.array([to_host(z)])
            vz = numpy.array([to_host(vz)])
        if self._use_c(kwargs, delta):
            Lz = R * vT
            # Resolve namespace first so a forced backend (numpy inputs) also routes
            # to the C-native path; numpy stays on the plain C path below.
            xp = get_namespace(R, vR, vT, z, vz)
            if xp is not numpy:
                if self._inverse is not None:
                    # each point in its own model: the backend c=False code
                    return self._evaluate(
                        R,
                        vR,
                        vT,
                        z,
                        vz,
                        delta=delta,
                        c=False,
                        _direct=True,
                        _routed=True,
                    )
                # jax/torch: differentiable actions via the C-native 2x5 Jacobian
                # (vjp/autograd.Function).
                R, vR, vT, z, vz = promote_scalars(xp, R, vR, vT, z, vz)
                Lz = R * vT
                u0, refu0_calc = _staeckel_c_backend_refu0(
                    self._pot,
                    delta,
                    R,
                    vR,
                    vT,
                    z,
                    vz,
                    self._useu0,
                    kwargs.pop("u0", None),
                )
                jr, jz = _staeckel_c_grad_actions(
                    self._pot, delta, R, vR, vT, z, vz, u0, order, useu0=refu0_calc
                )
                return (jr, Lz, jz)
            if self._useu0:
                # First calculate u0
                if "u0" in kwargs:
                    u0 = numpy.asarray(kwargs["u0"])
                else:
                    E = numpy.array(
                        [
                            _evaluatePotentials(self._pot, R[ii], z[ii])
                            + vR[ii] ** 2.0 / 2.0
                            + vz[ii] ** 2.0 / 2.0
                            + vT[ii] ** 2.0 / 2.0
                            for ii in range(len(R))
                        ]
                    )
                    u0 = actionAngleStaeckel_c.actionAngleStaeckel_calcu0(
                        E, Lz, self._pot, delta
                    )[0]
                kwargs.pop("u0", None)
            elif self._inverse is not None:
                delta, u0, ddelta = self._inverse_model(R, vR, vT, z, vz, deriv=False)
            else:
                u0 = None
            jr, jz, err = actionAngleStaeckel_c.actionAngleStaeckel_c(
                self._pot,
                delta,
                R,
                vR,
                vT,
                z,
                vz,
                u0=u0,
                order=order,
                model=self._model,
            )
            if self._inverse is not None:
                jr, jz = self._inverse_missing(delta, u0, jr, jz)
            if err == 0:
                return (jr, Lz, jz)
            else:  # pragma: no cover
                raise RuntimeError(
                    "C-code for calculation actions failed; try with c=False"
                )
        else:
            if "c" in kwargs and kwargs["c"] and not self._c:  # pragma: no cover
                warnings.warn(
                    "C module not used because potential does not have a C implementation",
                    galpyWarning,
                )
            # C requested, rerouted here (or the inverse= reroute below)
            routed = kwargs.pop("_routed", False) or self._c_requested(kwargs)
            kwargs.pop("c", None)
            # Unified vectorised, backend-agnostic path (numpy + jax/torch),
            # replacing the former per-object scipy loop. Uses the chi-anomaly
            # quadrature on C's mesh (_staeckel_chi_quads), so the standalone
            # c=False actions are consistent with both c=True and
            # _actionsFreqsAngles (was ~1e-5 off via adaptive quad).
            # Resolve from the active namespace (honours use(backend, force=True))
            # and promote numpy inputs, so the existing tests run the vectorised
            # backend path for real; numpy stays byte-identical (xp is numpy).
            xp = get_namespace(R, vR, vT, z, vz)
            if xp is not numpy:
                R, vR, vT, z, vz = promote_scalars(xp, R, vR, vT, z, vz)
            u0 = ddelta = None
            if self._inverse is not None:
                delta, u0, ddelta = self._inverse_model(
                    R, vR, vT, z, vz, c=loop_c, deriv=False, flag_unbound=routed
                )
            jr, Lz, jz = _staeckel_actions(
                xp,
                R,
                vR,
                vT,
                z,
                vz,
                _staeckel_pot(xp, self._pot),
                _coerce_delta_arraylike(delta),
                order,
                flag_unbound=routed,
                u0=u0,
            )
            if self._inverse is not None:
                jr, jz = self._inverse_missing(delta, u0, jr, jz)
            if xp is not numpy:
                return (jr, Lz, jz)
            return (numpy.atleast_1d(jr), numpy.atleast_1d(Lz), numpy.atleast_1d(jz))

    def _actionsFreqs(self, *args, **kwargs):
        """
        Evaluate the actions and frequencies (jr,lz,jz,Omegar,Omegaphi,Omegaz).

        Parameters
        ----------
        *args : tuple
            Either:
            a) R,vR,vT,z,vz[,phi]:
                1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)
                2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
            b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
        delta: bool, optional
            can be used to override the object-wide focal length; can also be an array with length N to allow different delta for different phase-space points
        u0: float, optional
            if object-wide option useu0 is set, u0 to use (if useu0 and useu0 is None, a good value will be computed).
        c: bool, optional
            True/False to override the object-wide setting for whether or not to use the C implementation. With jax/torch inputs, derivatives w.r.t. the phase-space coordinates are C-native; a derivative C cannot carry -- w.r.t. a differentiated potential parameter or delta -- falls back to the backend c=False implementation (as a C orbit integration falls back to the in-backend ODE), which agrees with C to ~1e-8.
        order: int, optional
            number of points to use in the Gauss-Legendre numerical integration of the relevant action and frequency integrals.

        Returns
        -------
        tuple
            (jr,lz,jz,Omegar,Omegaphi,Omegaz)

        Notes
        -----
        - 2013-08-28 - Written - Bovy (IAS)
        """
        if self._polish and not kwargs.pop("_direct", False):
            return self._polished(*args, **kwargs)[:6]
        loop_c = bool(self._c and kwargs.get("c", True))
        ddelta = None
        delta = kwargs.pop("delta", self._delta)
        order = kwargs.get("order", self._order)
        if self._use_c(kwargs, delta):
            if len(args) == 5:  # R,vR.vT, z, vz
                R, vR, vT, z, vz = args
            elif len(args) == 6:  # R,vR.vT, z, vz, phi
                R, vR, vT, z, vz, phi = args
            else:
                self._parse_eval_args(*args)
                R = self._eval_R
                vR = self._eval_vR
                vT = self._eval_vT
                z = self._eval_z
                vz = self._eval_vz
            if isinstance(R, float):
                R = numpy.array([R])
                vR = numpy.array([to_host(vR)])
                vT = numpy.array([to_host(vT)])
                z = numpy.array([to_host(z)])
                vz = numpy.array([to_host(vz)])
            Lz = R * vT
            # Resolve namespace first so a forced backend (numpy inputs) also routes
            # to the C-native path; numpy stays on the plain C path below.
            xp = get_namespace(R, vR, vT, z, vz)
            if xp is not numpy:
                if self._inverse is not None:
                    # each point in its own model: the backend c=False code
                    return self._actionsFreqs(
                        R,
                        vR,
                        vT,
                        z,
                        vz,
                        delta=delta,
                        c=False,
                        _direct=True,
                        _routed=True,
                    )
                # jax/torch: differentiable (jr,jz,Omega) via the fused C-native
                # (5x5) Jacobian -- actions rows (#1051) + freq rows (#131), in one
                # C pass. First-order.
                R, vR, vT, z, vz = promote_scalars(xp, R, vR, vT, z, vz)
                Lz = R * vT
                u0, refu0_calc = _staeckel_c_backend_refu0(
                    self._pot,
                    delta,
                    R,
                    vR,
                    vT,
                    z,
                    vz,
                    self._useu0,
                    kwargs.pop("u0", None),
                )
                jr, jz, Omegar, Omegaphi, Omegaz = _staeckel_c_grad_actionsfreqs(
                    self._pot, delta, R, vR, vT, z, vz, u0, order, useu0=refu0_calc
                )
                return (jr, Lz, jz, Omegar, Omegaphi, Omegaz)
            if self._useu0:
                # First calculate u0
                if "u0" in kwargs:
                    u0 = numpy.asarray(kwargs["u0"])
                else:
                    E = numpy.array(
                        [
                            _evaluatePotentials(self._pot, R[ii], z[ii])
                            + vR[ii] ** 2.0 / 2.0
                            + vz[ii] ** 2.0 / 2.0
                            + vT[ii] ** 2.0 / 2.0
                            for ii in range(len(R))
                        ]
                    )
                    u0 = actionAngleStaeckel_c.actionAngleStaeckel_calcu0(
                        E, Lz, self._pot, delta
                    )[0]
                kwargs.pop("u0", None)
            elif self._inverse is not None:
                delta, u0, ddelta = self._inverse_model(R, vR, vT, z, vz)
            else:
                u0 = None
            if self._model:
                # the frequencies of tori whose reference curve follows them
                # are computed with their angles
                jr, jz, Omegar, Omegaphi, Omegaz, _, _, _, err = (
                    actionAngleStaeckel_c.actionAngleFreqAngleStaeckel_c(
                        self._pot, delta, R, vR, vT, z, vz, 0.0 * R,
                        u0=u0, order=order, model=self._model, ddelta=ddelta,
                        du0=self._inverse_du0(R, vR, vT, z, vz, delta, u0, c=True),
                    )
                )  # fmt: skip
            else:
                jr, jz, Omegar, Omegaphi, Omegaz, err = (
                    actionAngleStaeckel_c.actionAngleFreqStaeckel_c(
                        self._pot, delta, R, vR, vT, z, vz, u0=u0, order=order
                    )
                )
            if self._inverse is not None:
                jr, jz, Omegar, Omegaphi, Omegaz = self._inverse_missing(
                    delta, u0, jr, jz, Omegar, Omegaphi, Omegaz
                )
            # Adjustments for close-to-circular orbits
            indx = numpy.isnan(Omegar) * (jr < 10.0**-3.0) + numpy.isnan(Omegaz) * (
                jz < 10.0**-3.0
            )  # Close-to-circular and close-to-the-plane orbits
            if numpy.sum(indx) > 0:
                Omegar[indx] = [
                    epifreq(self._pot, r, use_physical=False) for r in R[indx]
                ]
                Omegaphi[indx] = [
                    omegac(self._pot, r, use_physical=False) for r in R[indx]
                ]
                Omegaz[indx] = [
                    verticalfreq(self._pot, r, use_physical=False) for r in R[indx]
                ]
            if err == 0:
                return (jr, Lz, jz, Omegar, Omegaphi, Omegaz)
            else:  # pragma: no cover
                raise RuntimeError(
                    "C-code for calculation actions failed; try with c=False"
                )
        else:
            if "c" in kwargs and kwargs["c"] and not self._c:  # pragma: no cover
                warnings.warn(
                    "C module not used because potential does not have a C implementation",
                    galpyWarning,
                )
            if len(args) == 5:  # R,vR.vT, z, vz
                R, vR, vT, z, vz = args
            elif len(args) == 6:  # R,vR.vT, z, vz, phi
                R, vR, vT, z, vz, phi = args
            else:
                self._parse_eval_args(*args)
                R = self._eval_R
                vR = self._eval_vR
                vT = self._eval_vT
                z = self._eval_z
                vz = self._eval_vz
            if isinstance(R, float):
                R = numpy.array([R])
                vR = numpy.array([to_host(vR)])
                vT = numpy.array([to_host(vT)])
                z = numpy.array([to_host(z)])
                vz = numpy.array([to_host(vz)])
            # C requested, rerouted here (or the inverse= reroute below)
            routed = kwargs.pop("_routed", False) or self._c_requested(kwargs)
            kwargs.pop("c", None)
            kwargs.pop("u0", None)
            # Unified vectorised, backend-agnostic path (the useu0 reference is
            # action/frequency-invariant, so it is not needed here).
            xp = get_namespace(R, vR, vT, z, vz)
            if xp is not numpy:
                R, vR, vT, z, vz = promote_scalars(xp, R, vR, vT, z, vz)
            u0 = ddelta = None
            if self._inverse is not None:
                delta, u0, ddelta = self._inverse_model(
                    R, vR, vT, z, vz, c=loop_c, flag_unbound=routed
                )
            jr, Lz, jz, Omegar, Omegaphi, Omegaz = _staeckel_actions_freqs(
                xp,
                R,
                vR,
                vT,
                z,
                vz,
                _staeckel_pot(xp, self._pot),
                _coerce_delta_arraylike(delta),
                order,
                flag_unbound=routed,
                u0=u0,
                torus=self._model,
                ddelta=ddelta,
                du0=self._inverse_du0(R, vR, vT, z, vz, delta, u0),
            )
            if self._inverse is not None:
                jr, jz, Omegar, Omegaphi, Omegaz = self._inverse_missing(
                    delta, u0, jr, jz, Omegar, Omegaphi, Omegaz
                )
            # Close-to-circular orbits: the freqs are NaN (det(A)=0); substitute
            # epifreq/omegac/verticalfreq (vectorised mirror of the C wrapper).
            Omegar, Omegaphi, Omegaz = _staeckel_circular_freqs(
                xp, self._pot, R, jr, jz, Omegar, Omegaphi, Omegaz
            )
            return (jr, Lz, jz, Omegar, Omegaphi, Omegaz)

    def _actionsFreqsAngles(self, *args, **kwargs):
        """
        Evaluate the actions, frequencies, and angles (jr,lz,jz,Omegar,Omegaphi,Omegaz,angler,anglephi,anglez).

        Parameters
        ----------
        *args : tuple
            Either:
            a) R,vR,vT,z,vz[,phi]:
                1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)
                2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
            b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
        delta: bool, optional
            can be used to override the object-wide focal length; can also be an array with length N to allow different delta for different phase-space points
        u0: float, optional
            if object-wide option useu0 is set, u0 to use (if useu0 and useu0 is None, a good value will be computed).
        c: bool, optional
            True/False to override the object-wide setting for whether or not to use the C implementation. With jax/torch inputs, derivatives w.r.t. the phase-space coordinates are C-native; a derivative C cannot carry -- w.r.t. a differentiated potential parameter or delta -- falls back to the backend c=False implementation (as a C orbit integration falls back to the in-backend ODE), which agrees with C to ~1e-8.
        order: int, optional
            number of points to use in the Gauss-Legendre numerical integration of the relevant action, frequency, and angle integrals.

        Returns
        -------
        tuple
            (jr,lz,jz,Omegar,Omegaphi,Omegaz,angler,anglephi,anglez)

        Notes
        -----
        - 2013-08-28 - Written - Bovy (IAS)
        """
        if self._polish and not kwargs.pop("_direct", False):
            return self._polished(*args, **kwargs)
        # (the number of points that are not _bk_polished's padding)
        npts = kwargs.pop("_npts", None)
        loop_c = bool(self._c and kwargs.get("c", True))
        ddelta = None
        delta = kwargs.pop("delta", self._delta)
        order = kwargs.get("order", self._order)
        if self._use_c(kwargs, delta):
            if len(args) == 5:  # R,vR.vT, z, vz pragma: no cover
                raise OSError("Must specify phi")
            elif len(args) == 6:  # R,vR.vT, z, vz, phi
                R, vR, vT, z, vz, phi = args
            else:
                self._parse_eval_args(*args)
                R = self._eval_R
                vR = self._eval_vR
                vT = self._eval_vT
                z = self._eval_z
                vz = self._eval_vz
                phi = self._eval_phi
            if isinstance(R, float):
                R = numpy.array([R])
                vR = numpy.array([to_host(vR)])
                vT = numpy.array([to_host(vT)])
                z = numpy.array([to_host(z)])
                vz = numpy.array([to_host(vz)])
                phi = numpy.array([to_host(phi)])
            Lz = R * vT
            # Resolve namespace first so a forced backend (numpy inputs) also routes
            # to the C-native path; numpy stays on the plain C path below.
            xp = get_namespace(R, vR, vT, z, vz)
            if xp is not numpy:
                if self._inverse is not None:
                    # each point in its own model: the backend c=False code
                    return self._actionsFreqsAngles(
                        R,
                        vR,
                        vT,
                        z,
                        vz,
                        phi,
                        delta=delta,
                        c=False,
                        _direct=True,
                        _routed=True,
                        _npts=npts,
                    )
                # jax/torch: differentiable actions, frequencies AND angles via the
                # fused C-native Jacobian (#131 PR-B); phi enters analytically
                # (d anglephi/dphi==1). Values byte-identical to the c=True numpy path.
                R, vR, vT, z, vz, phi = promote_scalars(xp, R, vR, vT, z, vz, phi)
                Lz = R * vT
                u0, refu0_calc = _staeckel_c_backend_refu0(
                    self._pot,
                    delta,
                    R,
                    vR,
                    vT,
                    z,
                    vz,
                    self._useu0,
                    kwargs.pop("u0", None),
                )
                jr, jz, Omegar, Omegaphi, Omegaz, angler, anglephi, anglez = (
                    _staeckel_c_grad_actionsfreqsangles(
                        self._pot,
                        delta,
                        R,
                        vR,
                        vT,
                        z,
                        vz,
                        phi,
                        u0,
                        order,
                        useu0=refu0_calc,
                    )
                )
                return (jr, Lz, jz, Omegar, Omegaphi, Omegaz, angler, anglephi, anglez)
            if self._useu0:
                # First calculate u0
                if "u0" in kwargs:
                    u0 = numpy.asarray(kwargs["u0"])
                else:
                    E = numpy.array(
                        [
                            _evaluatePotentials(self._pot, R[ii], z[ii])
                            + vR[ii] ** 2.0 / 2.0
                            + vz[ii] ** 2.0 / 2.0
                            + vT[ii] ** 2.0 / 2.0
                            for ii in range(len(R))
                        ]
                    )
                    u0 = actionAngleStaeckel_c.actionAngleStaeckel_calcu0(
                        E, Lz, self._pot, delta
                    )[0]
                kwargs.pop("u0", None)
            elif self._inverse is not None:
                delta, u0, ddelta = self._inverse_model(R, vR, vT, z, vz)
            else:
                u0 = None
            (
                jr,
                jz,
                Omegar,
                Omegaphi,
                Omegaz,
                angler,
                anglephi,
                anglez,
                err,
            ) = actionAngleStaeckel_c.actionAngleFreqAngleStaeckel_c(
                self._pot,
                delta,
                R,
                vR,
                vT,
                z,
                vz,
                phi,
                u0=u0,
                order=order,
                model=self._model,
                ddelta=ddelta,
                du0=self._inverse_du0(R, vR, vT, z, vz, delta, u0, c=True),
            )
            if self._inverse is not None:
                jr, jz, Omegar, Omegaphi, Omegaz, angler, anglephi, anglez = (
                    self._inverse_missing(
                        delta,
                        u0,
                        jr,
                        jz,
                        Omegar,
                        Omegaphi,
                        Omegaz,
                        angler,
                        anglephi,
                        anglez,
                    )
                )
            # Adjustments for close-to-circular orbits
            indx = numpy.isnan(Omegar) * (jr < 10.0**-3.0) + numpy.isnan(Omegaz) * (
                jz < 10.0**-3.0
            )  # Close-to-circular and close-to-the-plane orbits
            if numpy.sum(indx) > 0:
                Omegar[indx] = [
                    epifreq(self._pot, r, use_physical=False) for r in R[indx]
                ]
                Omegaphi[indx] = [
                    omegac(self._pot, r, use_physical=False) for r in R[indx]
                ]
                Omegaz[indx] = [
                    verticalfreq(self._pot, r, use_physical=False) for r in R[indx]
                ]
            if err == 0:
                return (jr, Lz, jz, Omegar, Omegaphi, Omegaz, angler, anglephi, anglez)
            else:
                raise RuntimeError(
                    "C-code for calculation actions failed; try with c=False"
                )  # pragma: no cover
        else:
            if "c" in kwargs and kwargs["c"] and not self._c:  # pragma: no cover
                warnings.warn(
                    "C module not used because potential does not have a C implementation",
                    galpyWarning,
                )
            if len(args) == 5:  # R,vR.vT, z, vz pragma: no cover
                raise OSError("Must specify phi")
            elif len(args) == 6:  # R,vR.vT, z, vz, phi
                R, vR, vT, z, vz, phi = args
            else:
                self._parse_eval_args(*args)
                R = self._eval_R
                vR = self._eval_vR
                vT = self._eval_vT
                z = self._eval_z
                vz = self._eval_vz
                phi = self._eval_phi
            if isinstance(R, float):
                R = numpy.array([R])
                vR = numpy.array([to_host(vR)])
                vT = numpy.array([to_host(vT)])
                z = numpy.array([to_host(z)])
                vz = numpy.array([to_host(vz)])
                phi = numpy.array([to_host(phi)])
            # C requested, rerouted here (or the inverse= reroute below)
            routed = kwargs.pop("_routed", False) or self._c_requested(kwargs)
            kwargs.pop("c", None)
            kwargs.pop("u0", None)
            # Unified vectorised, backend-agnostic path (the useu0 reference is
            # action/frequency/angle-invariant, so it is not needed here).
            xp = get_namespace(R, vR, vT, z, vz)
            if xp is not numpy:
                R, vR, vT, z, vz = promote_scalars(xp, R, vR, vT, z, vz)
                # fold the azimuth in R's namespace AND device (a bare xp.asarray
                # lands on the CPU and would collide with a CUDA anglephi).
                phi = asarray_on_device(xp, phi, device_of(R))
            u0 = ddelta = None
            if self._inverse is not None:
                delta, u0, ddelta = self._inverse_model(
                    R, vR, vT, z, vz, c=loop_c, flag_unbound=routed, npts=npts
                )
            (
                jr,
                Lz,
                jz,
                Omegar,
                Omegaphi,
                Omegaz,
                angler,
                anglephi,
                anglez,
            ) = _staeckel_actions_freqs_angles(
                xp,
                R,
                vR,
                vT,
                z,
                vz,
                phi,
                _staeckel_pot(xp, self._pot),
                _coerce_delta_arraylike(delta),
                order,
                flag_unbound=routed,
                u0=u0,
                torus=self._model,
                ddelta=ddelta,
                du0=self._inverse_du0(R, vR, vT, z, vz, delta, u0),
            )
            if self._inverse is not None:
                jr, jz, Omegar, Omegaphi, Omegaz, angler, anglephi, anglez = (
                    self._inverse_missing(
                        delta, u0, jr, jz, Omegar, Omegaphi, Omegaz,
                        angler, anglephi, anglez,
                    )
                )  # fmt: skip
            # Close-to-circular orbits: substitute epifreq/omegac/verticalfreq for
            # the NaN frequencies (vectorised mirror of the C wrapper; the angles
            # are already 0 there, as in the C calcAnglesStaeckel).
            Omegar, Omegaphi, Omegaz = _staeckel_circular_freqs(
                xp, self._pot, R, jr, jz, Omegar, Omegaphi, Omegaz
            )
            return (jr, Lz, jz, Omegar, Omegaphi, Omegaz, angler, anglephi, anglez)

    def _EccZmaxRperiRap(self, *args, **kwargs):
        """
        Evaluate the eccentricity, maximum height above the plane, peri- and apocenter in the Staeckel approximation.

        Parameters
        ----------
        *args : tuple
            Either:
            a) R,vR,vT,z,vz[,phi]:
                1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)
                2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
            b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
        delta: bool, optional
            can be used to override the object-wide focal length; can also be an array with length N to allow different delta for different phase-space points
        u0: float, optional
            if object-wide option useu0 is set, u0 to use (if useu0 and useu0 is None, a good value will be computed).
        c: bool, optional
            True/False to override the object-wide setting for whether or not to use the C implementation. With jax/torch inputs, derivatives w.r.t. the phase-space coordinates are C-native; a derivative C cannot carry -- w.r.t. a differentiated potential parameter or delta -- falls back to the backend c=False implementation (as a C orbit integration falls back to the in-backend ODE), which agrees with C to ~1e-8.

        Returns
        -------
        tuple
            (e,zmax,rperi,rap)

        Notes
        -----
        - 2017-12-12 - Written - Bovy (UofT)
        """
        delta = _coerce_delta_arraylike(kwargs.get("delta", self._delta))
        # Parse args to (R,vR,vT,z,vz) for the c=True backend gate.
        if len(args) == 5:
            R, vR, vT, z, vz = args
        elif len(args) == 6:
            R, vR, vT, z, vz, phi = args
        else:
            self._parse_eval_args(*args)
            R = self._eval_R
            vR = self._eval_vR
            vT = self._eval_vT
            z = self._eval_z
            vz = self._eval_vz
        if isinstance(R, float):
            R = numpy.array([R])
            vR = numpy.array([to_host(vR)])
            vT = numpy.array([to_host(vT)])
            z = numpy.array([to_host(z)])
            vz = numpy.array([to_host(vz)])
        # Resolve namespace first so a forced backend (numpy inputs) also routes to
        # the C-native path; numpy stays on the turning-point path below.
        xp = get_namespace(R, vR, vT, z, vz)
        if self._use_c(kwargs, delta) and xp is not numpy and self._inverse is not None:
            # each point in its own model: the backend c=False code
            kwargs["c"] = False
        elif self._use_c(kwargs, delta) and xp is not numpy:
            # jax/torch: differentiable (e,zmax,rperi,rap) via the C-native 4x5
            # Jacobian (custom_vjp/autograd.Function); numpy stays on the path
            # below (the ctypes turning-point wrapper cannot take backend arrays).
            R, vR, vT, z, vz = promote_scalars(xp, R, vR, vT, z, vz)
            u0, refu0_calc = _staeckel_c_backend_refu0(
                self._pot, delta, R, vR, vT, z, vz, self._useu0, kwargs.pop("u0", None)
            )
            return _staeckel_c_grad_ecczmax(
                self._pot, delta, R, vR, vT, z, vz, u0, useu0=refu0_calc
            )
        umin, umax, vmin = self._uminumaxvmin(*args, **kwargs)
        xp = get_namespace(umin) if is_backend_array(umin) else numpy
        if isinstance(delta, numpy.ndarray):  # uv_to_Rz resolves one namespace
            (delta,) = coerce_coords(xp, delta, device=device_of(umin))
        rperi = coords.uv_to_Rz(umin, numpy.pi / 2.0, delta=delta)[0]
        rap_tmp, zmax = coords.uv_to_Rz(umax, vmin, delta=delta)
        rap = xp.sqrt(rap_tmp**2.0 + zmax**2.0)
        e = (rap - rperi) / (rap + rperi)
        return (e, zmax, rperi, rap)

    def _uminumaxvmin(self, *args, **kwargs):
        """
        Evaluate u_min, u_max, and v_min in the Staeckel approximation.

        Parameters
        ----------
        *args : tuple
            Either:
            a) R,vR,vT,z,vz[,phi]:
                1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)
                2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
            b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
        delta: bool, optional
            can be used to override the object-wide focal length; can also be an array with length N to allow different delta for different phase-space points
        u0: float, optional
            if object-wide option useu0 is set, u0 to use (if useu0 and useu0 is None, a good value will be computed).
        c: bool, optional
            True/False to override the object-wide setting for whether or not to use the C implementation. With jax/torch inputs, derivatives w.r.t. the phase-space coordinates are C-native; a derivative C cannot carry -- w.r.t. a differentiated potential parameter or delta -- falls back to the backend c=False implementation (as a C orbit integration falls back to the in-backend ODE), which agrees with C to ~1e-8.

        Returns
        -------
        tuple
            (u_min, u_max, v_min)

        Notes
        -----
        - 2017-12-12 - Written - Bovy (UofT)
        """
        loop_c = bool(self._c and kwargs.get("c", True))
        _delta = kwargs.pop("delta", self._delta)
        # a backend delta (e.g. estimated from a differentiated potential) may
        # be traced; numpy.atleast_1d refuses it. numpy input keeps the numpy
        # call.
        delta = (
            get_namespace(_delta).atleast_1d(_delta)
            if is_backend_array(_delta)
            else numpy.atleast_1d(_delta)
        )
        if len(args) == 5:  # R,vR.vT, z, vz
            R, vR, vT, z, vz = args
        elif len(args) == 6:  # R,vR.vT, z, vz, phi
            R, vR, vT, z, vz, phi = args
        else:
            self._parse_eval_args(*args)
            R = self._eval_R
            vR = self._eval_vR
            vT = self._eval_vT
            z = self._eval_z
            vz = self._eval_vz
        if isinstance(R, float):
            R = numpy.array([R])
            vR = numpy.array([to_host(vR)])
            vT = numpy.array([to_host(vT)])
            z = numpy.array([to_host(z)])
            vz = numpy.array([to_host(vz)])
        if self._use_c(kwargs, delta):
            Lz = R * vT
            if self._useu0:
                # First calculate u0
                if "u0" in kwargs:
                    u0 = numpy.asarray(kwargs["u0"])
                else:
                    E = numpy.array(
                        [
                            _evaluatePotentials(self._pot, R[ii], z[ii])
                            + vR[ii] ** 2.0 / 2.0
                            + vz[ii] ** 2.0 / 2.0
                            + vT[ii] ** 2.0 / 2.0
                            for ii in range(len(R))
                        ]
                    )
                    u0 = actionAngleStaeckel_c.actionAngleStaeckel_calcu0(
                        E, Lz, self._pot, delta
                    )[0]
                kwargs.pop("u0", None)
            elif self._inverse is not None:
                delta, u0, ddelta = self._inverse_model(R, vR, vT, z, vz, deriv=False)
            else:
                u0 = None
            (
                umin,
                umax,
                vmin,
                err,
            ) = actionAngleStaeckel_c.actionAngleUminUmaxVminStaeckel_c(
                self._pot, delta, R, vR, vT, z, vz, u0=u0, model=self._model
            )
            if err == 0:
                return (umin, umax, vmin)
            else:  # pragma: no cover
                raise RuntimeError(
                    "C-code for calculation actions failed; try with c=False"
                )
        else:
            if "c" in kwargs and kwargs["c"] and not self._c:  # pragma: no cover
                warnings.warn(
                    "C module not used because potential does not have a C implementation",
                    galpyWarning,
                )
            routed = self._c_requested(kwargs)  # C requested, rerouted here
            kwargs.pop("c", None)
            # Unified vectorised, backend-agnostic turning points (shared with the
            # actions/freqs via _staeckel_prep); feeds _EccZmaxRperiRap.
            xp = get_namespace(R, vR, vT, z, vz)
            if xp is not numpy:
                R, vR, vT, z, vz = promote_scalars(xp, R, vR, vT, z, vz)
            # _staeckel_prep already snaps vmin to pi/2 for planar orbits.
            u0 = ddelta = None
            if self._inverse is not None:
                delta, u0, ddelta = self._inverse_model(
                    R, vR, vT, z, vz, c=loop_c, deriv=False, flag_unbound=routed
                )
            s, umin, umax, vmin, _ = _staeckel_prep(
                xp,
                R,
                vR,
                vT,
                z,
                vz,
                _staeckel_pot(xp, self._pot),
                delta,
                flag_unbound=routed,
                u0=u0,
            )
            if routed:  # C's unbound turning points: e=nan, rperi=-inf, rap=inf
                umin = xp.where(s["unbound"], -numpy.inf, umin)
                umax = xp.where(s["unbound"], numpy.inf, umax)
            return (umin, umax, vmin)


def calcELStaeckel(R, vR, vT, z, vz, pot, vc=1.0, ro=1.0):
    """
    Calculate the energy and angular momentum.

    Parameters
    ----------
    R : float
        Galactocentric radius (/ro).
    vR : float
        Radial part of the velocity (/vc).
    vT : float
        Azimuthal part of the velocity (/vc).
    z : float
        Vertical height (/ro).
    vz : float
        Vertical velocity (/vc).
    pot : Potential object
        galpy Potential object or a combined potential formed using addition (pot1+pot2+…).
    vc : float, optional
        Circular velocity at ro (km/s). Default: 1.0.
    ro : float, optional
        Distance to the Galactic center (kpc). Default: 1.0.

    Returns
    -------
    tuple
        Tuple containing energy and angular momentum.

    Notes
    -----
    - 2012-11-30 - Written - Bovy (IAS)

    """
    return (
        _evaluatePotentials(pot, R, z) + vR**2.0 / 2.0 + vT**2.0 / 2.0 + vz**2.0 / 2.0,
        R * vT,
    )


def potentialStaeckel(u, v, pot, delta):
    """
    Return the potential.

    Parameters
    ----------
    u : float
        Confocal u.
    v : float
        Confocal v.
    pot : Potential object
        Potential.
    delta : float
        Focus.

    Returns
    -------
    float
        Potential at (u, v).

    Notes
    -----
    - 2012-11-29 - Written - Bovy (IAS)
    """
    R, z = coords.uv_to_Rz(u, v, delta=delta)
    return _evaluatePotentials(pot, R, z)


def FRStaeckel(u, v, pot, delta):  # pragma: no cover because unused
    """
    Return the radial force.

    Parameters
    ----------
    u : float
        Confocal u.
    v : float
        Confocal v.
    pot : Potential object
        Potential.
    delta : float
        Focus.

    Returns
    -------
    float
        Radial force.

    Notes
    -----
    - 2012-11-30 - Written - Bovy (IAS)

    """
    R, z = coords.uv_to_Rz(u, v, delta=delta)
    return _evaluateRforces(pot, R, z)


def FZStaeckel(u, v, pot, delta):  # pragma: no cover because unused
    """
    Return the vertical force.

    Parameters
    ----------
    u : float
        Confocal u.
    v : float
        Confocal v.
    pot : Potential object
        Potential.
    delta : float
        Focus.

    Returns
    -------
    Ffloat
        Vertical force.

    Notes
    -----
    - 2012-11-30 - Written - Bovy (IAS)
    """
    R, z = coords.uv_to_Rz(u, v, delta=delta)
    return _evaluatezforces(pot, R, z)


def _JRStaeckelIntegrandSquared(
    u, E, Lz, I3U, delta, u0, sinh2u0, v0, sin2v0, potu0v0, pot
):
    # potu0v0= potentialStaeckel(u0,v0,pot,delta)
    """The J_R integrand: p^2_u(u)/2/delta^2"""
    xp = get_namespace(u) if is_backend_array(u) else numpy
    sinh2u = xp.sinh(u) ** 2.0
    dU = (sinh2u + sin2v0) * potentialStaeckel(u, v0, pot, delta) - (
        sinh2u0 + sin2v0
    ) * potu0v0
    return E * sinh2u - I3U - dU - Lz**2.0 / 2.0 / delta**2.0 / sinh2u


def _JzStaeckelIntegrandSquared(
    v, E, Lz, I3V, delta, u0, cosh2u0, sinh2u0, potu0pi2, pot
):
    # potu0pi2= potentialStaeckel(u0,numpy.pi/2.,pot,delta)
    """The J_z integrand: p_v(v)/2/delta^2"""
    xp = get_namespace(v) if is_backend_array(v) else numpy
    sin2v = xp.sin(v) ** 2.0
    dV = cosh2u0 * potu0pi2 - (sinh2u0 + sin2v) * potentialStaeckel(u0, v, pot, delta)
    return E * sin2v + I3V + dV - Lz**2.0 / 2.0 / delta**2.0 / sin2v


# Derivatives of the under-radical functions S_R/S_z with respect to the
# integration coordinate (analytic, via the forces); these supply the finite
# turning-point limits of the chi-anomaly quadratures below
def _dJRStaeckelIntegrandSquareddu(
    u, E, Lz, I3U, delta, u0, sinh2u0, v0, sin2v0, potu0v0, pot
):
    # xp, not numpy: the chi-anomaly edge reconstruction calls this on the
    # BACKEND path too, where numpy.cosh(Tensor) breaks autograd via
    # __array__. get_namespace(u) is numpy for every numpy caller, so the
    # numpy path stays byte-identical.
    xp = get_namespace(u)
    R, z = coords.uv_to_Rz(u, v0, delta=delta)
    dPhidu = -delta * (
        _evaluateRforces(pot, R, z) * xp.cosh(u) * _staeckel_trig(xp, "sin", v0)
        + _evaluatezforces(pot, R, z) * xp.sinh(u) * _staeckel_trig(xp, "cos", v0)
    )
    return (
        E * xp.sinh(2.0 * u)
        - xp.sinh(2.0 * u) * potentialStaeckel(u, v0, pot, delta)
        - (xp.sinh(u) ** 2.0 + sin2v0) * dPhidu
        + Lz**2.0 / delta**2.0 * xp.cosh(u) / xp.sinh(u) ** 3.0
    )


def _dJzStaeckelIntegrandSquareddv(
    v, E, Lz, I3V, delta, u0, cosh2u0, sinh2u0, potu0pi2, pot
):
    # see _dJRStaeckelIntegrandSquareddu on why this resolves a namespace
    xp = get_namespace(v)
    R, z = coords.uv_to_Rz(u0, v, delta=delta)
    dPhidv = -delta * (
        _evaluateRforces(pot, R, z) * _staeckel_trig(xp, "sinh", u0) * xp.cos(v)
        - _evaluatezforces(pot, R, z) * _staeckel_trig(xp, "cosh", u0) * xp.sin(v)
    )
    return (
        E * xp.sin(2.0 * v)
        - xp.sin(2.0 * v) * potentialStaeckel(u0, v, pot, delta)
        - (sinh2u0 + xp.sin(v) ** 2.0) * dPhidv
        + Lz**2.0 / delta**2.0 * xp.cos(v) / xp.sin(v) ** 3.0
    )


# Nodes/weights of the composite 10-point Gauss-Legendre rule used by the
# chi-anomaly quadratures: applied per interval of an nchi-panel mesh, the
# error is O((chimax/nchi)^20), so the integrals are machine-converged already
# at the default mesh (_staeckel_nchi)
_CHIQUAD_GLX, _CHIQUAD_GLW = numpy.polynomial.legendre.leggauss(10)
# below this y(1-y) the direct S is a cancelling difference; Q is rebuilt from
# S' instead (C's STAECKEL_CHI_EDGE)
_STAECKEL_CHI_EDGE = 1e-6
# umin below which the J_R action uses the graded near-axis mesh (C's
# STAECKEL_NEARAXIS); its low-half panel count. C grades 24 panels deep; 8
# already reproduce C's jr to 4e-12 (6: 1e-10, 4: 2e-7) on the near-axis grid
_STAECKEL_NEARAXIS = 0.2
_STAECKEL_NEARAXIS_NPANELS = 8


@potential_physical_input
@physical_conversion("position", pop=True)
def estimateDeltaStaeckel(pot, R, z, no_median=False, delta0=1e-6):
    """
    Estimate a good value for delta using eqn. (9) in Sanders (2012)

    Parameters
    ----------
    pot : Potential instance or a combined potential formed using addition (pot1+pot2+…)
    R : float or numpy.ndarray
        coordinates
    z : float or numpy.ndarray
        coordinates
    no_median : bool, optional
        if True, and input is array, return all calculated values of delta (useful for quickly estimating delta for many phase space points)
    delta0 : float, optional
        value to return when delta<delta0 (because actionAngleStaeckel does not work with delta=0 exactly)

    Returns
    -------
    float or numpy.ndarray
        estimate of delta

    Notes
    -----
    - 2013-08-28 - Written - Bovy (IAS)
    - 2016-02-20 - Changed input order to allow physical conversions - Bovy (UofT)
    - 2022-09-14 - Deal with numerical issues with SCF/DiskSCFPotentials - Bovy (UofT)
    - 2022-09-15 - Add delta0 - Bovy (UofT)
    """

    pot = _check_potential_list_and_deprecate(pot)
    if _isNonAxi(pot):
        raise PotentialError(
            "Calling estimateDeltaStaeckel with non-axisymmetric potentials is not supported"
        )
    # We'll special-case delta<0 when the potential includes SCF/DiskSCF components
    # because their numerical second derivatives can lead to slightly negative delta2
    pot_includes_scf = (
        numpy.any(
            [
                isinstance(p, SCFPotential) or isinstance(p, DiskSCFPotential)
                for p in pot
            ]
        )
        if isinstance(pot, CompositePotential)
        else isinstance(pot, SCFPotential) or isinstance(pot, DiskSCFPotential)
    )
    # a differentiated potential parameter counts as data at float (R, z)
    xp = _pot_data_namespace((pot,), R, z)
    if xp is not numpy:
        # Resolved-namespace path (runs on jax/torch): under a forced backend the
        # numpy inputs are promoted UP to the backend so the whole estimate runs
        # on the backend instead of falling through to a numpy island. xp.where /
        # _nanmedian reproduce the numpy in-place writes / masked median.
        R, z = promote_scalars(xp, R, z)
        z = xp.where(z == 0.0, 1e-4, z)

        def _delta2(Ri, zi):
            # eqn. (9) has a sign error (hence z^2 - R^2)
            return (
                zi**2.0
                - Ri**2.0
                + (
                    3.0 * Ri * _evaluatezforces(pot, Ri, zi)
                    - 3.0 * zi * _evaluateRforces(pot, Ri, zi)
                    + Ri
                    * zi
                    * (
                        evaluateR2derivs(pot, Ri, zi, use_physical=False)
                        - evaluatez2derivs(pot, Ri, zi, use_physical=False)
                    )
                )
                / evaluateRzderivs(pot, Ri, zi, use_physical=False)
            )

        try:
            # array-capable potentials evaluate the whole array at once
            delta2 = _delta2(R, z)
        except (TypeError, RuntimeError):
            # potentials whose evaluators reject a whole-array call are done
            # element-by-element; each scalar is a backend scalar so the migrated
            # scalar path still runs on the backend. No SHIPPED potential lands
            # here: DoubleExponentialDisk does not (its scalar-only
            # decorator sits on the public methods, and the calls above go
            # through the internal _evaluateRforces/_evaluatezforces, which
            # bypass it), and AnyAxisymmetricRazorThinDisk -- which did, via a
            # float() on an array -- now opts in to backend arrays. The branch
            # is kept for third-party potentials and is covered by a synthetic
            # in test_backend_actionAngle.py.
            delta2 = xp.stack([_delta2(R[ii], z[ii]) for ii in range(len(R))])
        indx = (delta2 < delta0**2.0) & (
            (delta2 > -(10.0**-10.0)) | bool(pot_includes_scf)
        )
        delta2 = xp.where(indx, delta0**2.0, delta2)
        if not no_median and getattr(delta2, "ndim", 0) > 0:
            delta2 = _nanmedian(xp, delta2)
        return xp.sqrt(delta2)
    # numpy path: byte-identical to the original (per-element evaluation, so
    # potentials whose methods only accept scalars keep working).
    if numpy.any(z == 0.0):
        if isinstance(z, numpy.ndarray):
            z[z == 0.0] = 1e-4
        else:
            z = 1e-4
    if isinstance(R, numpy.ndarray):
        delta2 = numpy.array(
            [
                (
                    z[ii] ** 2.0
                    - R[ii] ** 2.0  # eqn. (9) has a sign error
                    + (
                        3.0 * R[ii] * _evaluatezforces(pot, R[ii], z[ii])
                        - 3.0 * z[ii] * _evaluateRforces(pot, R[ii], z[ii])
                        + R[ii]
                        * z[ii]
                        * (
                            evaluateR2derivs(pot, R[ii], z[ii], use_physical=False)
                            - evaluatez2derivs(pot, R[ii], z[ii], use_physical=False)
                        )
                    )
                    / evaluateRzderivs(pot, R[ii], z[ii], use_physical=False)
                )
                for ii in range(len(R))
            ]
        )
        indx = (delta2 < delta0**2.0) * ((delta2 > -(10.0**-10.0)) + pot_includes_scf)
        delta2[indx] = delta0**2.0
        if not no_median:
            delta2 = numpy.median(delta2[True ^ numpy.isnan(delta2)])
    else:
        delta2 = (
            z**2.0
            - R**2.0  # eqn. (9) has a sign error
            + (
                3.0 * R * _evaluatezforces(pot, R, z)
                - 3.0 * z * _evaluateRforces(pot, R, z)
                + R
                * z
                * (
                    evaluateR2derivs(pot, R, z, use_physical=False)
                    - evaluatez2derivs(pot, R, z, use_physical=False)
                )
            )
            / evaluateRzderivs(pot, R, z, use_physical=False)
        )
        if delta2 < delta0**2.0 and (delta2 > -(10.0**-10.0) or pot_includes_scf):
            delta2 = delta0**2.0
    return numpy.sqrt(delta2)
