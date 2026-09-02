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
from scipy import integrate, optimize

from ..backend import (
    asarray_on_device,
    device_of,
    get_namespace,
    is_backend_array,
    name_of_namespace,
    promote_scalars,
)
from ..backend._namespaces import (
    graft_gradient,
    stop_gradient,
    under_jax_trace,
    under_torch_grad,
)
from ..backend.optimize import bisect_root, iterate_bracket
from ..backend.quadrature import fixed_quad as _backend_fixed_quad
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
    PotentialError,
    _check_c,
    _check_potential_list_and_deprecate,
    _evaluatePotentials,
    _evaluateRforces,
    _evaluatezforces,
    _isNonAxi,
)
from ..util import coords  # for prolate confocal transforms
from ..util import (
    conversion,
    galpyWarning,
)
from ..util.conversion import physical_conversion, potential_physical_input
from . import actionAngleStaeckel_c
from .actionAngle import UnboundError, actionAngle
from .actionAngleStaeckel_c import _ext_loaded as ext_loaded


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


# ----------------------------------------------------------------------------
# Vectorised, backend-agnostic Staeckel action core (numpy / jax / torch). One
# unified path that replaced the former per-object scipy loop:
# elementwise setup + turning points via the shared backend.optimize.bisect_root
# (fixed-iteration expanding bracket) + the action integrals via
# backend.quadrature.fixed_quad. Matches the C gsl_glfixed exactly: plain GL of
# `order` points over [umin,umax]/[vmin,pi/2] (the J integrands VANISH at the
# turning points, so no t^2-substitution is needed and grads don't flow through
# the limits). v0=pi/2 for the u (J_R) integral, u0 for the v (J_z) integral.


def _staeckel_setup(xp, R, vR, vT, z, vz, pot, delta):
    """Vectorised setup quantities (the former per-object __init__ did these one
    orbit at a time)."""
    ux, vx = coords.Rz_to_uv(R, z, delta=delta)
    sinvx, cosvx = xp.sin(vx), xp.cos(vx)
    coshux, sinhux = xp.cosh(ux), xp.sinh(ux)
    pux = delta * (vR * coshux * sinvx + vz * sinhux * cosvx)
    pvx = delta * (vR * sinhux * cosvx - vz * coshux * sinvx)
    E, Lz = calcELStaeckel(R, vR, vT, z, vz, pot)
    u0 = ux  # u0 does not matter for a single action evaluation
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


def _refine_tp(xp, f, u, skip):
    """Differentiable turning point: one Newton step off the frozen bisection
    root, grafted so the VALUE is byte-identical (numpy: graft_gradient is the
    identity) but AD carries the TRUE implicit derivative du/dtheta =
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
    h = 1e-6
    fu0 = f(u0)
    fp = (f(u0 + h) - f(u0 - h)) / (2.0 * h)
    # f(u0) can be NaN/inf AT the root (S dips <0 / divides by 0 there) while the
    # FD slope from f(u0+/-h) is finite -> skip the Newton step there. Mask the
    # numerator to NaN-free BEFORE dividing (dead-branch guard: eager xp.where
    # evaluates both sides, so raw fu0/fp would still poison the value/grad).
    good = (xp.abs(fp) > 1e-10) & xp.isfinite(fu0) & xp.isfinite(fp) & ~skip
    fp_safe = xp.where(good, fp, xp.ones_like(fp))
    fu0_safe = xp.where(good, fu0, xp.zeros_like(fu0))
    donor = u0 - fu0_safe / fp_safe  # finite everywhere; == u0 where ~good
    # Value-exact graft: donor - stop_gradient(donor) == 0 EXACTLY (finite donor),
    # so the value is byte-exactly the bisection root (no 1/sqrt(S) amplification of
    # a graft_gradient ~1e-16 wobble) while AD carries du/dtheta = grad(donor).
    return u0 + (donor - stop_gradient(donor))


def _staeckel_uminumax(xp, s, pot, delta):
    """Vectorised (umin, umax): bracket-and-bisect roots of the J_R integrand^2."""
    args = (s["E"], s["Lz"], s["I3U"], delta, s["u0"], s["sinh2u0"],
            s["v0u"], s["sin2v0u"], s["potu0v0"], pot)  # fmt: skip
    f = lambda u: _JRStaeckelIntegrandSquared(u, *args)
    ux, eps = s["ux"], 1e-8
    at_turn = (xp.abs(s["pux"]) < 1e-7) | (xp.abs(f(ux)) < 1e-10)
    peps, meps = f(ux + eps), f(ux - eps)
    at_umin = at_turn & (peps > 0.0) & (meps < 0.0)
    at_umax = at_turn & (peps < 0.0) & (meps > 0.0)
    circular = at_turn & ~at_umin & ~at_umax
    # Lower bracket: HALVE below ux until f<0 (60 halvings reach ~1e-18, so even a
    # near-axis turning point at u~1e-4 -- low-Lz, nearly-radial orbits -- is
    # straddled; *0.9 only reached ~3.8e-4*ux in 80 steps and collapsed umin to ux).
    lo = iterate_bracket(
        lambda l: xp.where((f(l) >= 0.0) & (l > 1e-10), l * 0.5, l), ux * 0.5, 60
    )
    # f still >0 at the floor -> no lower J_R turning point: the orbit reaches the
    # symmetry axis (Lz~0, purely-radial), so umin=0 (mirrors C / Single rstart==0).
    reaches_axis = f(lo) >= 0.0
    # expanding bracket above ux until f<0 (stop at u=100)
    hi = iterate_bracket(
        lambda h: xp.where((f(h) >= 0.0) & (h < 100.0), h * 1.1, h), ux * 1.1, 80
    )
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
    umin = bisect_root(f, lo, u_hi_umin, xp, xtol=1e-13, maxiter=200)
    umax = bisect_root(f, u_lo_umax, hi, xp, xtol=1e-13, maxiter=200)
    umin = xp.where(at_umin | circular, ux, umin)
    umax = xp.where(at_umax | circular, ux, umax)
    umin = xp.where(reaches_axis, xp.zeros_like(umin), umin)  # axis-reaching -> 0
    # differentiable turning points (value byte-identical; injects du/dtheta)
    umin = _refine_tp(xp, f, umin, skip=reaches_axis | circular)
    umax = _refine_tp(xp, f, umax, skip=circular)
    return umin, umax, unbound


def _staeckel_vmin(xp, s, pot, delta):
    """Vectorised vmin: bracket-and-bisect root of the J_z integrand^2 below vx."""
    args = (s["E"], s["Lz"], s["I3V"], delta, s["u0"], s["cosh2u0v"],
            s["sinh2u0v"], s["potupi2"], pot)  # fmt: skip
    f = lambda v: _JzStaeckelIntegrandSquared(v, *args)
    vx, eps = s["vx"], 1e-8
    at_turn = (xp.abs(s["pvx"]) < 1e-7) | (xp.abs(f(vx)) < 1e-10)
    at_vmin = at_turn & (f(vx + eps) > 0.0) & (f(vx - eps) < 0.0)
    vlo = iterate_bracket(
        lambda v: xp.where((f(v) >= 0.0) & (v > 1e-9), v * 0.9, v), vx * 0.9, 80
    )
    vmin = bisect_root(f, vlo, vx, xp, xtol=1e-13, maxiter=200)
    vmin = xp.where(at_vmin, vx, vmin)
    return _refine_tp(xp, f, vmin, skip=(xp.zeros_like(vmin) > 0.5))


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


@lru_cache(maxsize=None)
def _staeckel_chi_mesh(nchi):
    """The composite rule on the NORMALIZED anomaly t in [0,1]: `nchi` panels of
    the 10-node GL rule. Kept normalized so ``chimax`` may be per-orbit (the
    angles need an INCOMPLETE integral, whose upper anomaly differs per orbit)."""
    e = numpy.linspace(0.0, 1.0, nchi + 1)
    mid = 0.5 * (e[:-1] + e[1:])
    half = 0.5 * (e[1:] - e[:-1])
    t = (mid[:, None] + _CHIQUAD_GLX[None, :] * half[:, None]).ravel()
    w = (half[:, None] * _CHIQUAD_GLW[None, :]).ravel()
    return t, w


def _staeckel_chi_quads(xp, Ssq, dSsq, args, qmin, D, order, chimax=numpy.pi):
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

    Returns (action, sqrt(Q), q, wts) so every quadrature on this mesh -- the
    action and the 1/p profiles -- comes from ONE evaluation of S.
    """
    t, w = _staeckel_chi_mesh(max(2 * int(order), 20))
    dev = device_of(qmin)
    t = asarray_on_device(xp, t, dev)
    w = asarray_on_device(xp, w, dev)
    chi = chimax * t
    wts = chimax * w
    y = xp.sin(chi / 2.0) ** 2.0
    y1my = y * (1.0 - y)
    # masks, not control flow: xp.where evaluates BOTH branches, so each unused
    # denominator is kept finite or reverse-mode AD is poisoned by its NaN
    ones = xp.ones_like(y1my)
    is_edge = y1my <= 1e-6
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


def _staeckel_chi_action(xp, Ssq, dSsq, args, qmin, D, order, chimax=numpy.pi):
    """The action integral alone -- see :func:`_staeckel_chi_quads`."""
    return _staeckel_chi_quads(xp, Ssq, dSsq, args, qmin, D, order, chimax)[0]


def _staeckel_t2_action(xp, sqfunc, args, lo, hi, order):
    """Low(lo)+High(hi) t^2-substituted panels of sqrt(sqfunc) over [lo, hi] --
    the AD-regular gradient DONOR for the plain-GL action value: d(sqrt S) is
    turning-point-singular under plain GL, but after u = lo + t^2 the du = 2t dt
    Jacobian cancels the 1/sqrt(S) ~ 1/t, and AD through it carries the FULL
    dependence (E, Lz, I3, the u0/v0u reference geometry, potential parameters)."""
    a2 = tuple(x[..., None] if getattr(x, "ndim", 0) >= 1 else x for x in args)
    # Turning-point limits held fixed: the Leibniz boundary terms vanish exactly
    # (S = 0 there). The limits now arrive from _refine_tp carrying their TRUE
    # implicit derivatives (needed for the action Hessian / frequency gradients);
    # the direct boundary term still vanishes (S=0), so the action's first
    # gradient is unchanged, while the second derivative gets its missing
    # turning-point-motion term.
    span = hi - lo
    ok = span > 0.0  # degenerate (circular/planar) panel: 0 with 0 gradient
    mid = xp.sqrt(0.5 * xp.where(ok, span, xp.ones_like(span)))

    def panel(base, sign):
        def integ(s):  # s: (n,) -> (N, n); u = base + sign*t^2, t = mid*s
            t = mid[..., None] * s
            u = base[..., None] + sign * t**2.0
            S = sqfunc(u, *a2)
            Ssafe = xp.where(S > 0.0, S, xp.ones_like(S))  # dead-branch guard
            g = xp.where(S > 0.0, xp.sqrt(Ssafe), xp.zeros_like(S))
            return 2.0 * t * g * mid[..., None]

        return _backend_fixed_quad(xp, integ, 0.0, 1.0, n=order, device=device_of(mid))

    return xp.where(ok, panel(lo, 1.0) + panel(hi, -1.0), xp.zeros_like(span))


def _staeckel_prep(xp, R, vR, vT, z, vz, pot, delta):
    """Setup quantities + turning points (+ unbound check), shared by the
    vectorised actions and frequencies. Returns (setup, umin, umax, vmin, delta)."""
    if is_backend_array(R) and not is_backend_array(delta):
        # match R's namespace AND device: a bare xp.asarray(delta) lands on the
        # CPU, so an array-valued delta (e.g. EccZmax's atleast_1d) then collides
        # with CUDA R/z in coords.Rz_to_uv. No-op on numpy (device_of -> None).
        delta = asarray_on_device(xp, delta, device_of(R))
    s = _staeckel_setup(xp, R, vR, vT, z, vz, pot, delta)
    umin, umax, unbound = _staeckel_uminumax(xp, s, pot, delta)
    # Unbound orbits raise eagerly on the numpy path (mirrors the Single class);
    # under a backend they must stay jit-traceable, so we cannot branch on the
    # traced `unbound` -- let unbound orbits fall through to NaN instead (the
    # caller jits/AD-traces and checks). numpy stays byte-identical (still raises).
    if not is_backend_array(R) and bool(numpy.any(unbound)):
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
        )
        * sqrt2
        * delta
        / numpy.pi
    )
    jz_args = (s["E"], s["Lz"], s["I3V"], delta, s["u0"], s["cosh2u0v"],
               s["sinh2u0v"], s["potupi2"], pot)  # fmt: skip
    pi2 = numpy.pi / 2.0 * xp.ones_like(vmin)
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
        )
        * 2.0
        * sqrt2
        * delta
        / numpy.pi
    )
    # Backend AD: graft the t^2-substituted donor's gradient onto the plain-GL
    # value (naive d(sqrt S) is turning-point-singular, and the dJ/d(E,Lz,I3)
    # chain misses the u0/v0u geometry's direct R,z dependence). Value unchanged
    # (C parity), first-order only, no cost on plain (untraced/no-grad) forwards.
    if is_backend_array(jr) and (under_jax_trace(jr) or under_torch_grad(jr)):
        jr = graft_gradient(
            jr,
            _staeckel_t2_action(
                xp, _JRStaeckelIntegrandSquared, jr_args, umin, umax, order
            )
            * sqrt2
            * delta
            / numpy.pi,
        )
        jz = graft_gradient(
            jz,
            _staeckel_t2_action(
                xp, _JzStaeckelIntegrandSquared, jz_args, vmin, pi2, order
            )
            * 2.0
            * sqrt2
            * delta
            / numpy.pi,
        )
    jr = xp.where((umax - umin) / umax < 1e-6, xp.zeros_like(jr), jr)
    jz = xp.where((numpy.pi / 2.0 - vmin) < 1e-7, xp.zeros_like(jz), jz)
    return jr, jz


def _staeckel_actions(xp, R, vR, vT, z, vz, pot, delta, order):
    """Unified vectorised (jr, Lz, jz) for numpy and jax/torch backends."""
    s, umin, umax, vmin, delta = _staeckel_prep(xp, R, vR, vT, z, vz, pot, delta)
    jr, jz = _staeckel_jr_jz(xp, s, umin, umax, vmin, pot, delta, order)
    return jr, s["Lz"], jz


# --------------------------------------------------------------- frequencies
# The frequency derivative integrals dJ/d(E,Lz,I3) need the t^2-substitution
# (their integrands are (factor)/sqrt(S), SINGULAR at the turning points, unlike
# the action integrand sqrt(S) which vanishes there). Mirror the C
# dJ?d?{Low,High}StaeckelIntegrand split: low panel u=lo+t^2, high panel
# u=hi-t^2, both over t in [0, sqrt(0.5(hi-lo))]. xp.where guards the dead S<=0
# branch (the orbit can sit arbitrarily close to a turning point).


def _staeckel_chi_profiles(
    xp, Ssq, dSsq, args, qmin, D, order, weight_fns, chimax=numpy.pi
):
    """The 1/p profile integrals int f/sqrt(S) dq for every f in `weight_fns`,
    on the SAME chi mesh as the action: int f/sqrt(S) dq = D int f/sqrt(Q) dchi.

    One mesh serves all of them (main's `_chiQuadsU`/`_chiQuadsV` do the same),
    so the six Leibniz derivatives cost two S evaluations rather than twelve."""
    _, sqQ, q, wts = _staeckel_chi_quads(xp, Ssq, dSsq, args, qmin, D, order, chimax)
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


def _staeckel_actions_freqs(xp, R, vR, vT, z, vz, pot, delta, order):
    """Unified vectorised (jr, Lz, jz, Omegar, Omegaphi, Omegaz); the frequencies
    are NaN for circular orbits (the caller substitutes epifreq/omegac/verticalfreq).
    Setup + turning points are computed once and shared between actions and freqs."""
    s, umin, umax, vmin, delta = _staeckel_prep(xp, R, vR, vT, z, vz, pot, delta)
    jr, jz = _staeckel_jr_jz(xp, s, umin, umax, vmin, pot, delta, order)
    jac = _staeckel_jacobian(xp, s, umin, umax, vmin, pot, delta, order)
    Omegar, Omegaphi, Omegaz = _staeckel_freqs(
        xp, s, umin, umax, vmin, pot, delta, order, jac
    )
    return jr, s["Lz"], jz, Omegar, Omegaphi, Omegaz


# ------------------------------------------------------------------- angles
# The angles need PARTIAL Leibniz integrals (from a turning point to the current
# u/v), unlike the freqs' full turning-point-to-turning-point integrals. The
# vectorised quadrant tree mirrors the per-object calcAnglesStaeckel: the panel
# (Low from umin/vmin, High from umax/pi-2) is chosen by which turning point the
# position is closer to, and a reflection constant K and sign s -- functions only
# of the momentum sign x position quadrant -- map the partial integral onto the
# full angle (4 leaves in u, 8 in v). All branches are computed and xp.where-
# selected (with the turning-point dead-branch guard) for vectorisation.


def _staeckel_angles(xp, s, umin, umax, vmin, pot, delta, order, jac):
    """Vectorised (angler, anglephi_raw, anglez); the caller folds the azimuth phi
    into anglephi. angler/anglez are in [0, 2pi); circular orbits -> all 0.
    `jac` = the _staeckel_jacobian 6-tuple, computed once by the caller and shared
    with the freqs."""
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
    part_u = _staeckel_chi_profiles(
        xp, JRsq, _dJRStaeckelIntegrandSquareddu, jr_args, umin, Du, order,
        (
            lambda xp, u: xp.sinh(u) ** 2.0,
            lambda xp, u: xp.ones_like(u),
            lambda xp, u: 1.0 / xp.sinh(u) ** 2.0,
        ), chimax=chi_ux[..., None],
    )  # fmt: skip
    full_u = _staeckel_chi_profiles(
        xp, JRsq, _dJRStaeckelIntegrandSquareddu, jr_args, umin, Du, order,
        (
            lambda xp, u: xp.sinh(u) ** 2.0,
            lambda xp, u: xp.ones_like(u),
            lambda xp, u: 1.0 / xp.sinh(u) ** 2.0,
        ),
    )  # fmt: skip
    PE, PI, PL = (xp.where(high_u, fu - pu, pu) for pu, fu in zip(part_u, full_u))
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
    part_v = _staeckel_chi_profiles(
        xp, JZsq, _dJzStaeckelIntegrandSquareddv, jz_args, vmin, Dv, order,
        (
            lambda xp, v: xp.sin(v) ** 2.0,
            lambda xp, v: xp.ones_like(v),
            lambda xp, v: 1.0 / xp.sin(v) ** 2.0,
        ), chimax=chimax_v[..., None],
    )  # fmt: skip
    half_v = _staeckel_chi_profiles(
        xp, JZsq, _dJzStaeckelIntegrandSquareddv, jz_args, vmin, Dv, order,
        (
            lambda xp, v: xp.sin(v) ** 2.0,
            lambda xp, v: xp.ones_like(v),
            lambda xp, v: 1.0 / xp.sin(v) ** 2.0,
        ), chimax=numpy.pi / 2.0,
    )  # fmt: skip
    QE, QI, QL = (
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
    angler = xp.remainder(angler, 2.0 * pi)  # fmod + non-negative wrap == remainder
    anglez = xp.remainder(anglez, 2.0 * pi)
    zeros = xp.zeros_like(angler)
    circ_full = circ | planar  # both degeneracies -> C calcAngles returns 0
    angler = xp.where(circ_full, zeros, angler)
    anglez = xp.where(circ_full, zeros, anglez)
    anglephi = xp.where(circ_full, zeros, anglephi)
    return angler, anglephi, anglez


def _staeckel_actions_freqs_angles(xp, R, vR, vT, z, vz, phi, pot, delta, order):
    """Unified vectorised (jr,Lz,jz,Omegar,Omegaphi,Omegaz,angler,anglephi,anglez);
    setup + turning points computed once and shared. anglephi includes the azimuth."""
    s, umin, umax, vmin, delta = _staeckel_prep(xp, R, vR, vT, z, vz, pot, delta)
    jr, jz = _staeckel_jr_jz(xp, s, umin, umax, vmin, pot, delta, order)
    # The six Leibniz derivative panels are shared by the frequencies and the
    # angles -- compute once and thread into both (else each recomputes all six).
    jac = _staeckel_jacobian(xp, s, umin, umax, vmin, pot, delta, order)
    Omegar, Omegaphi, Omegaz = _staeckel_freqs(
        xp, s, umin, umax, vmin, pot, delta, order, jac=jac
    )
    angler, anglephi, anglez = _staeckel_angles(
        xp, s, umin, umax, vmin, pot, delta, order, jac=jac
    )
    anglephi = xp.remainder(anglephi + phi, 2.0 * numpy.pi)  # fold in the azimuth
    return jr, s["Lz"], jz, Omegar, Omegaphi, Omegaz, angler, anglephi, anglez


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
        numpy.asarray(stop_gradient(delta), dtype=numpy.float64)
    )
    u0_np = (
        None if u0 is None else numpy.asarray(stop_gradient(u0), dtype=numpy.float64)
    )

    def host_jac(Rn, vRn, vTn, zn, vzn):
        jr, jz, jac, err = actionAngleStaeckel_c.actionAngleStaeckel_actionsJac_c(
            pot, delta_np, Rn, vRn, vTn, zn, vzn, u0=u0_np, order=order, useu0=useu0
        )
        return jr, jz, jac

    name = name_of_namespace(get_namespace(R, vR, vT, z, vz))
    if name == "jax":
        from ..backend._jax.staeckel_c import actions_with_jac

        return actions_with_jac(host_jac, (R, vR, vT, z, vz))
    if name == "torch":
        from ..backend._torch.staeckel_c import actions_with_jac

        return actions_with_jac(host_jac, R, vR, vT, z, vz)
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
    reference (no gradient), so the Jacobian is the partial at fixed delta.
    First-order only."""
    from ..orbit.integrateFullOrbit import _parse_pot

    _parse_pot(
        pot, potforactions=True
    )  # eager: surface unsupported-pot NotImplementedError outside the jax pure_callback (matches the numpy path)
    delta_np = numpy.atleast_1d(
        numpy.asarray(stop_gradient(delta), dtype=numpy.float64)
    )
    u0_np = (
        None if u0 is None else numpy.asarray(stop_gradient(u0), dtype=numpy.float64)
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
        numpy.asarray(stop_gradient(delta), dtype=numpy.float64)
    )
    u0_np = (
        None if u0 is None else numpy.asarray(stop_gradient(u0), dtype=numpy.float64)
    )

    def host_jac(Rn, vRn, vTn, zn, vzn):
        jr, jz, Or, Op, Oz, jac, err = (
            actionAngleStaeckel_c.actionAngleStaeckel_actionsFreqsJac_c(
                pot, delta_np, Rn, vRn, vTn, zn, vzn, u0=u0_np, order=order, useu0=useu0
            )
        )
        Or, Op, Oz = _staeckel_c_freq_circ_fix(pot, Rn, jr, jz, Or, Op, Oz)
        return jr, jz, Or, Op, Oz, jac

    name = name_of_namespace(get_namespace(R, vR, vT, z, vz))
    if name == "jax":
        from ..backend._jax.staeckel_c import actionsfreqs_with_jac

        return actionsfreqs_with_jac(host_jac, (R, vR, vT, z, vz))
    if name == "torch":
        from ..backend._torch.staeckel_c import actionsfreqs_with_jac

        return actionsfreqs_with_jac(host_jac, R, vR, vT, z, vz)
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
        numpy.asarray(stop_gradient(delta), dtype=numpy.float64)
    )
    u0_np = (
        None if u0 is None else numpy.asarray(stop_gradient(u0), dtype=numpy.float64)
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
        return numpy.asarray(stop_gradient(u0_kwarg), dtype=numpy.float64), False
    if not useu0:
        return None, False
    Rn, vRn, vTn, zn, vzn = (
        numpy.atleast_1d(numpy.asarray(stop_gradient(c), dtype=numpy.float64))
        for c in (R, vR, vT, z, vz)
    )
    E = numpy.array(
        [
            _evaluatePotentials(pot, Rn[ii], zn[ii])
            + vRn[ii] ** 2.0 / 2.0
            + vzn[ii] ** 2.0 / 2.0
            + vTn[ii] ** 2.0 / 2.0
            for ii in range(len(Rn))
        ]
    )
    return (
        actionAngleStaeckel_c.actionAngleStaeckel_calcu0(E, Rn * vTn, pot, delta)[0],
        True,
    )


def _staeckel_c_freq_circ_fix(pot, Rn, jrn, jzn, Or, Op, Oz):
    """Close-to-circular/planar frequency substitution (numpy host mirror of the
    C-wrapper adjustment): NaN freqs at small jr/jz -> epifreq/omegac/verticalfreq."""
    indx = numpy.isnan(Or) * (jrn < 1e-3) + numpy.isnan(Oz) * (jzn < 1e-3)
    if numpy.sum(indx) > 0:
        Or[indx] = [epifreq(pot, r, use_physical=False) for r in Rn[indx]]
        Op[indx] = [omegac(pot, r, use_physical=False) for r in Rn[indx]]
        Oz[indx] = [verticalfreq(pot, r, use_physical=False) for r in Rn[indx]]
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
            The focus.
        useu0 : bool, optional
            Use u0 to calculate dV (not recommended). Default is False.
        c : bool, optional
            If True, always use C for calculations. Default is False.
        order : int, optional
            Number of points to use in the Gauss-Legendre numerical integration of the relevant action, frequency, and angle integrals (C path). On the pure-Python path this instead scales the number of panels of the composite chi-anomaly quadrature (nchi = max(2 x order, 20)), which is machine-converged at the default, so increasing it there has no practical effect. Default is 10.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2012-11-27 - Started - Bovy (IAS).
        """
        actionAngle.__init__(self, ro=kwargs.get("ro", None), vo=kwargs.get("vo", None))
        if not "pot" in kwargs:  # pragma: no cover
            raise OSError("Must specify pot= for actionAngleStaeckel")
        self._pot = _check_potential_list_and_deprecate(kwargs["pot"])
        if self._pot == MWPotential:
            warnings.warn(
                "Use of MWPotential as a Milky-Way-like potential is deprecated; galpy.potential.MWPotential2014, a potential fit to a large variety of dynamical constraints (see Bovy 2015), is the preferred Milky-Way-like potential in galpy",
                galpyWarning,
            )
        if not "delta" in kwargs:  # pragma: no cover
            raise OSError("Must specify delta= for actionAngleStaeckel")
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
        self._delta = kwargs["delta"]
        self._order = kwargs.get("order", 10)
        self._delta = _coerce_delta_arraylike(
            conversion.parse_length(self._delta, ro=self._ro)
        )
        # Check the units
        self._check_consistent_units()
        return None

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
            True/False to override the object-wide setting for whether or not to use the C implementation.
        order: int, optional
            number of points to use in the Gauss-Legendre numerical integration of the relevant action integrals.
        fixed_quad: bool, optional
            if True, use Gaussian quadrature (scipy.integrate.fixed_quad instead of scipy.integrate.quad).
        **kwargs: dict, optional
            scipy.integrate.fixed_quad or .quad keywords when not using C

        Returns
        -------
        tuple
            (jr,lz,jz)

        Notes
        -----
        - 2012-11-27 - Written - Bovy (IAS)
        - 2017-12-27 - Allowed individual delta for each point - Bovy (UofT)
        """
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
            vR = numpy.array([vR])
            vT = numpy.array([vT])
            z = numpy.array([z])
            vz = numpy.array([vz])
        if (
            (self._c and not ("c" in kwargs and not kwargs["c"]))
            or (ext_loaded and ("c" in kwargs and kwargs["c"]))
        ) and _check_c(self._pot):
            Lz = R * vT
            # Resolve namespace first so a forced backend (numpy inputs) also routes
            # to the C-native path; numpy stays on the plain C path below.
            xp = get_namespace(R, vR, vT, z, vz)
            if xp is not numpy:
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
            else:
                u0 = None
            jr, jz, err = actionAngleStaeckel_c.actionAngleStaeckel_c(
                self._pot, delta, R, vR, vT, z, vz, u0=u0, order=order
            )
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
            kwargs.pop("c", None)
            # Unified vectorised, backend-agnostic path (numpy + jax/torch),
            # replacing the former per-object scipy loop. Uses
            # plain GL order-`order` to match the C path (the default GL order);
            # the standalone-actions c=False result is thus now consistent with
            # both c=True and _actionsFreqsAngles (was ~1e-5 off via adaptive quad).
            # Resolve from the active namespace (honours use(backend, force=True))
            # and promote numpy inputs, so the existing tests run the vectorised
            # backend path for real; numpy stays byte-identical (xp is numpy).
            xp = get_namespace(R, vR, vT, z, vz)
            if xp is not numpy:
                R, vR, vT, z, vz = promote_scalars(xp, R, vR, vT, z, vz)
            jr, Lz, jz = _staeckel_actions(
                xp, R, vR, vT, z, vz, self._pot, _coerce_delta_arraylike(delta), order
            )
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
            True/False to override the object-wide setting for whether or not to use the C implementation.
        order: int, optional
            number of points to use in the Gauss-Legendre numerical integration of the relevant action integrals.
        fixed_quad: bool, optional
            if True, use Gaussian quadrature (scipy.integrate.fixed_quad instead of scipy.integrate.quad).
        **kwargs: dict, optional
            scipy.integrate.fixed_quad or .quad keywords when not using C

        Returns
        -------
        tuple
            (jr,lz,jz,Omegar,Omegaphi,Omegaz)

        Notes
        -----
        - 2013-08-28 - Written - Bovy (IAS)
        """
        delta = kwargs.pop("delta", self._delta)
        order = kwargs.get("order", self._order)
        if (
            (self._c and not ("c" in kwargs and not kwargs["c"]))
            or (ext_loaded and ("c" in kwargs and kwargs["c"]))
        ) and _check_c(self._pot):
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
                vR = numpy.array([vR])
                vT = numpy.array([vT])
                z = numpy.array([z])
                vz = numpy.array([vz])
            Lz = R * vT
            # Resolve namespace first so a forced backend (numpy inputs) also routes
            # to the C-native path; numpy stays on the plain C path below.
            xp = get_namespace(R, vR, vT, z, vz)
            if xp is not numpy:
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
            else:
                u0 = None
            (
                jr,
                jz,
                Omegar,
                Omegaphi,
                Omegaz,
                err,
            ) = actionAngleStaeckel_c.actionAngleFreqStaeckel_c(
                self._pot, delta, R, vR, vT, z, vz, u0=u0, order=order
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
                vR = numpy.array([vR])
                vT = numpy.array([vT])
                z = numpy.array([z])
                vz = numpy.array([vz])
            kwargs.pop("c", None)
            kwargs.pop("u0", None)
            # Unified vectorised, backend-agnostic path (the useu0 reference is
            # action/frequency-invariant, so it is not needed here).
            xp = get_namespace(R, vR, vT, z, vz)
            if xp is not numpy:
                R, vR, vT, z, vz = promote_scalars(xp, R, vR, vT, z, vz)
            jr, Lz, jz, Omegar, Omegaphi, Omegaz = _staeckel_actions_freqs(
                xp, R, vR, vT, z, vz, self._pot, _coerce_delta_arraylike(delta), order
            )
            # Close-to-circular orbits: the freqs are NaN (det(A)=0); substitute
            # epifreq/omegac/verticalfreq (vectorised mirror of the C wrapper).
            indx = (xp.isnan(Omegar) & (jr < 1e-3)) | (xp.isnan(Omegaz) & (jz < 1e-3))
            Omegar = xp.where(indx, epifreq(self._pot, R, use_physical=False), Omegar)
            Omegaphi = xp.where(
                indx, omegac(self._pot, R, use_physical=False), Omegaphi
            )
            Omegaz = xp.where(
                indx, verticalfreq(self._pot, R, use_physical=False), Omegaz
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
            True/False to override the object-wide setting for whether or not to use the C implementation.
        order: int, optional
            number of points to use in the Gauss-Legendre numerical integration of the relevant action integrals.
        fixed_quad: bool, optional
            if True, use Gaussian quadrature (scipy.integrate.fixed_quad instead of scipy.integrate.quad).
        **kwargs: dict, optional
            scipy.integrate.fixed_quad or .quad keywords when not using C

        Returns
        -------
        tuple
            (jr,lz,jz,Omegar,Omegaphi,Omegaz,angler,anglephi,anglez)

        Notes
        -----
        - 2013-08-28 - Written - Bovy (IAS)
        """
        delta = kwargs.pop("delta", self._delta)
        order = kwargs.get("order", self._order)
        if (
            (self._c and not ("c" in kwargs and not kwargs["c"]))
            or (ext_loaded and ("c" in kwargs and kwargs["c"]))
        ) and _check_c(self._pot):
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
                vR = numpy.array([vR])
                vT = numpy.array([vT])
                z = numpy.array([z])
                vz = numpy.array([vz])
                phi = numpy.array([phi])
            Lz = R * vT
            # Resolve namespace first so a forced backend (numpy inputs) also routes
            # to the C-native path; numpy stays on the plain C path below.
            xp = get_namespace(R, vR, vT, z, vz)
            if xp is not numpy:
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
                self._pot, delta, R, vR, vT, z, vz, phi, u0=u0, order=order
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
                vR = numpy.array([vR])
                vT = numpy.array([vT])
                z = numpy.array([z])
                vz = numpy.array([vz])
                phi = numpy.array([phi])
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
                self._pot,
                _coerce_delta_arraylike(delta),
                order,
            )
            # Close-to-circular orbits: substitute epifreq/omegac/verticalfreq for
            # the NaN frequencies (vectorised mirror of the C wrapper; the angles
            # are already 0 there, as in the C calcAnglesStaeckel).
            indx = (xp.isnan(Omegar) & (jr < 1e-3)) | (xp.isnan(Omegaz) & (jz < 1e-3))
            Omegar = xp.where(indx, epifreq(self._pot, R, use_physical=False), Omegar)
            Omegaphi = xp.where(
                indx, omegac(self._pot, R, use_physical=False), Omegaphi
            )
            Omegaz = xp.where(
                indx, verticalfreq(self._pot, R, use_physical=False), Omegaz
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
            True/False to override the object-wide setting for whether or not to use the C implementation.

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
            vR = numpy.array([vR])
            vT = numpy.array([vT])
            z = numpy.array([z])
            vz = numpy.array([vz])
        # Resolve namespace first so a forced backend (numpy inputs) also routes to
        # the C-native path; numpy stays on the turning-point path below.
        xp = get_namespace(R, vR, vT, z, vz)
        if (
            (
                (self._c and not ("c" in kwargs and not kwargs["c"]))
                or (ext_loaded and ("c" in kwargs and kwargs["c"]))
            )
            and _check_c(self._pot)
            and xp is not numpy
        ):
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
            True/False to override the object-wide setting for whether or not to use the C implementation.

        Returns
        -------
        tuple
            (u_min, u_max, v_min)

        Notes
        -----
        - 2017-12-12 - Written - Bovy (UofT)
        """
        delta = numpy.atleast_1d(kwargs.pop("delta", self._delta))
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
            vR = numpy.array([vR])
            vT = numpy.array([vT])
            z = numpy.array([z])
            vz = numpy.array([vz])
        if (
            (self._c and not ("c" in kwargs and not kwargs["c"]))
            or (ext_loaded and ("c" in kwargs and kwargs["c"]))
        ) and _check_c(self._pot):
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
            else:
                u0 = None
            (
                umin,
                umax,
                vmin,
                err,
            ) = actionAngleStaeckel_c.actionAngleUminUmaxVminStaeckel_c(
                self._pot, delta, R, vR, vT, z, vz, u0=u0
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
            kwargs.pop("c", None)
            # Unified vectorised, backend-agnostic turning points (shared with the
            # actions/freqs via _staeckel_prep); feeds _EccZmaxRperiRap.
            xp = get_namespace(R, vR, vT, z, vz)
            if xp is not numpy:
                R, vR, vT, z, vz = promote_scalars(xp, R, vR, vT, z, vz)
            # _staeckel_prep already snaps vmin to pi/2 for planar orbits.
            _, umin, umax, vmin, _ = _staeckel_prep(
                xp, R, vR, vT, z, vz, self._pot, delta
            )
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
# error is O((chimax/nchi)^20), so the integrals are machine-converged for
# modest nchi
_CHIQUAD_GLX, _CHIQUAD_GLW = numpy.polynomial.legendre.leggauss(10)


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
    xp = get_namespace(R, z)
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
            # scalar path still runs on the backend. Measured 2026-08-16, the
            # potential that actually lands here is
            # AnyAxisymmetricRazorThinDiskPotential -- NOT DoubleExponentialDisk,
            # which this comment used to name: its scalar-only decorator sits on
            # the public methods, and the calls above go through the internal
            # _evaluateRforces/_evaluatezforces, which bypass it.
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
