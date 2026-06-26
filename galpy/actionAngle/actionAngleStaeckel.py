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

import numpy
from scipy import integrate, optimize

from ..backend import get_namespace, is_backend_array
from ..backend.optimize import bisect_root
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


# ----------------------------------------------------------------------------
# Vectorised, backend-agnostic Staeckel action core (numpy / jax / torch). One
# unified path replacing the per-object actionAngleStaeckelSingle scipy loop:
# elementwise setup + turning points via the shared backend.optimize.bisect_root
# (fixed-iteration expanding bracket) + the action integrals via
# backend.quadrature.fixed_quad. Matches the C gsl_glfixed exactly: plain GL of
# `order` points over [umin,umax]/[vmin,pi/2] (the J integrands VANISH at the
# turning points, so no t^2-substitution is needed and grads don't flow through
# the limits). v0=pi/2 for the u (J_R) integral, u0 for the v (J_z) integral.


def _staeckel_setup(xp, R, vR, vT, z, vz, pot, delta):
    """Vectorised mirror of actionAngleStaeckelSingle.__init__ setup quantities."""
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
    lo = ux * 0.5
    for _ in range(60):
        lo = xp.where((f(lo) >= 0.0) & (lo > 1e-10), lo * 0.5, lo)
    # f still >0 at the floor -> no lower J_R turning point: the orbit reaches the
    # symmetry axis (Lz~0, purely-radial), so umin=0 (mirrors C / Single rstart==0).
    reaches_axis = f(lo) >= 0.0
    hi = ux * 1.1
    for _ in range(80):  # expanding bracket above ux until f<0 (stop at u=100)
        hi = xp.where((f(hi) >= 0.0) & (hi < 100.0), hi * 1.1, hi)
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
    return umin, umax, unbound


def _staeckel_vmin(xp, s, pot, delta):
    """Vectorised vmin: bracket-and-bisect root of the J_z integrand^2 below vx."""
    args = (s["E"], s["Lz"], s["I3V"], delta, s["u0"], s["cosh2u0v"],
            s["sinh2u0v"], s["potupi2"], pot)  # fmt: skip
    f = lambda v: _JzStaeckelIntegrandSquared(v, *args)
    vx, eps = s["vx"], 1e-8
    at_turn = (xp.abs(s["pvx"]) < 1e-7) | (xp.abs(f(vx)) < 1e-10)
    at_vmin = at_turn & (f(vx + eps) > 0.0) & (f(vx - eps) < 0.0)
    vlo = vx * 0.9
    for _ in range(80):
        vlo = xp.where((f(vlo) >= 0.0) & (vlo > 1e-9), vlo * 0.9, vlo)
    vmin = bisect_root(f, vlo, vx, xp, xtol=1e-13, maxiter=200)
    return xp.where(at_vmin, vx, vmin)


def _staeckel_gl_action(xp, sqfunc, args, lo, hi, order):
    """Plain GL order-`order` of sqrt(sqfunc) over [lo, hi] (C gsl_glfixed parity)."""
    span = hi - lo
    a2 = tuple(
        x[..., None] if getattr(x, "ndim", 0) >= 1 else x for x in args
    )  # [N]->[N,1] to broadcast against the [N,n] node grid

    def integrand(t):  # t: (n,) -> (N, n); u = lo + span*t
        u = lo[..., None] + span[..., None] * t
        sq = sqfunc(u, *a2)
        sq = xp.where(sq > 0.0, sq, xp.zeros_like(sq))  # clip (AD/round-off guard)
        return xp.sqrt(sq) * span[..., None]

    return _backend_fixed_quad(xp, integrand, 0.0, 1.0, n=order)


def _staeckel_prep(xp, R, vR, vT, z, vz, pot, delta):
    """Setup quantities + turning points (+ unbound check), shared by the
    vectorised actions and frequencies. Returns (setup, umin, umax, vmin, delta)."""
    if is_backend_array(R) and not is_backend_array(delta):
        delta = xp.asarray(delta)  # per-object numpy delta -> match R's namespace
    s = _staeckel_setup(xp, R, vR, vT, z, vz, pot, delta)
    umin, umax, unbound = _staeckel_uminumax(xp, s, pot, delta)
    if bool(xp.any(unbound)):  # eager (no internal jit); mirrors the Single
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
        _staeckel_gl_action(xp, _JRStaeckelIntegrandSquared, jr_args, umin, umax, order)
        * sqrt2
        * delta
        / numpy.pi
    )
    jz_args = (s["E"], s["Lz"], s["I3V"], delta, s["u0"], s["cosh2u0v"],
               s["sinh2u0v"], s["potupi2"], pot)  # fmt: skip
    pi2 = numpy.pi / 2.0 * xp.ones_like(vmin)
    jz = (
        _staeckel_gl_action(xp, _JzStaeckelIntegrandSquared, jz_args, vmin, pi2, order)
        * 2.0
        * sqrt2
        * delta
        / numpy.pi
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


def _staeckel_deriv_panels(xp, Sfunc, sq_args, factor_fn, lo, hi, order):
    """Low(lo)+High(hi) t^2-substituted panels of factor_fn(u)/sqrt(Sfunc(u))."""
    a2 = tuple(x[..., None] if getattr(x, "ndim", 0) >= 1 else x for x in sq_args)
    mid = xp.sqrt(0.5 * (hi - lo))

    def panel(base, sign):
        def integ(s):  # s: (n,) -> (N, n); u = base + sign*t^2, t = mid*s
            t = mid[..., None] * s
            u = base[..., None] + sign * t**2.0
            S = Sfunc(u, *a2)
            Ssafe = xp.where(S > 0.0, S, xp.ones_like(S))  # dead-branch guard
            g = xp.where(S > 0.0, factor_fn(xp, u) / xp.sqrt(Ssafe), xp.zeros_like(S))
            return 2.0 * t * g * mid[..., None]  # du = 2t dt, dt = mid ds

        return _backend_fixed_quad(xp, integ, 0.0, 1.0, n=order)

    return panel(lo, 1.0) + panel(hi, -1.0)


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
    pi2 = numpy.pi / 2.0 * xp.ones_like(vmin)
    dP, JRsq, JZsq = (
        _staeckel_deriv_panels,
        _JRStaeckelIntegrandSquared,
        _JzStaeckelIntegrandSquared,
    )  # noqa: E501
    djrdE = (
        dP(xp, JRsq, jr_args, lambda xp, u: xp.sinh(u) ** 2.0, umin, umax, order)
        * prefr
    )
    djrdLz = dP(
        xp, JRsq, jr_args, lambda xp, u: 1.0 / xp.sinh(u) ** 2.0, umin, umax, order
    ) * (-Lz / numpy.pi / sqrt2 / delta)  # noqa: E501
    djrdI3 = dP(xp, JRsq, jr_args, lambda xp, u: xp.ones_like(u), umin, umax, order) * (
        -prefr
    )
    djzdE = (
        dP(xp, JZsq, jz_args, lambda xp, v: xp.sin(v) ** 2.0, vmin, pi2, order) * prefz
    )
    djzdLz = dP(
        xp, JZsq, jz_args, lambda xp, v: 1.0 / xp.sin(v) ** 2.0, vmin, pi2, order
    ) * (-Lz * sqrt2 / numpy.pi / delta)  # noqa: E501
    djzdI3 = (
        dP(xp, JZsq, jz_args, lambda xp, v: xp.ones_like(v), vmin, pi2, order) * prefz
    )
    return djrdE, djrdLz, djrdI3, djzdE, djzdLz, djzdI3


def _staeckel_freqs(xp, s, umin, umax, vmin, pot, delta, order):
    """Vectorised (Omegar, Omegaphi, Omegaz); NaN for circular (caller substitutes
    epifreq/omegac/verticalfreq, mirroring the C 0/0=NaN -> close-to-circular path)."""
    djrdE, djrdLz, djrdI3, djzdE, djzdLz, djzdI3 = _staeckel_jacobian(
        xp, s, umin, umax, vmin, pot, delta, order
    )
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
    Omegar, Omegaphi, Omegaz = _staeckel_freqs(
        xp, s, umin, umax, vmin, pot, delta, order
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


def _staeckel_angle_partial(xp, Sfunc, sq_args, factor_fn, base, sign, mid, order):
    """t^2-substituted partial integral of factor_fn(.)/sqrt(S) from the turning
    point `base` to base+sign*mid^2 (sign=+1 Low from umin/vmin, -1 High from
    umax/pi-2); `mid` and `sign` are per-orbit. Guarded at the turning point."""
    a2 = tuple(x[..., None] if getattr(x, "ndim", 0) >= 1 else x for x in sq_args)

    def integ(t01):  # t01 in [0,1] -> t = mid*t01; u = base + sign*t^2
        t = mid[..., None] * t01
        u = base[..., None] + sign[..., None] * t**2.0
        S = Sfunc(u, *a2)
        Ssafe = xp.where(S > 0.0, S, xp.ones_like(S))  # dead-branch guard
        g = xp.where(S > 0.0, factor_fn(xp, u) / xp.sqrt(Ssafe), xp.zeros_like(S))
        return 2.0 * t * g * mid[..., None]

    return _backend_fixed_quad(xp, integ, 0.0, 1.0, n=order)


def _staeckel_angles(xp, s, umin, umax, vmin, pot, delta, order):
    """Vectorised (angler, anglephi_raw, anglez); the caller folds the azimuth phi
    into anglephi. angler/anglez are in [0, 2pi); circular orbits -> all 0."""
    sqrt2 = numpy.sqrt(2.0)
    pi = numpy.pi
    Lz = s["Lz"]
    ux, vx, pux, pvx = s["ux"], s["vx"], s["pux"], s["pvx"]
    djrdE, djrdLz, djrdI3, djzdE, djzdLz, djzdI3 = _staeckel_jacobian(
        xp, s, umin, umax, vmin, pot, delta, order
    )
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
    JRsq, JZsq, AP = (
        _JRStaeckelIntegrandSquared,
        _JzStaeckelIntegrandSquared,
        _staeckel_angle_partial,
    )
    # ---- u-branch (4 leaves): panel by ux vs midpoint, K/s by (pux sign x panel)
    high_u = ux > umin + 0.5 * (umax - umin)
    base_u = xp.where(high_u, umax, umin)
    sign_u = xp.where(high_u, -xp.ones_like(ux), xp.ones_like(ux))
    mid_u = xp.sqrt(xp.where(high_u, umax - ux, ux - umin))
    PE = AP(
        xp, JRsq, jr_args, lambda xp, u: xp.sinh(u) ** 2.0, base_u, sign_u, mid_u, order
    )
    PI = AP(
        xp, JRsq, jr_args, lambda xp, u: xp.ones_like(u), base_u, sign_u, mid_u, order
    )
    PL = AP(
        xp,
        JRsq,
        jr_args,
        lambda xp, u: 1.0 / xp.sinh(u) ** 2.0,
        base_u,
        sign_u,
        mid_u,
        order,
    )
    pos_u = pux > 0.0
    K_u = xp.where(high_u, pi, xp.where(pos_u, 0.0, 2.0 * pi)) * xp.ones_like(ux)
    s_u = xp.where(high_u, xp.where(pos_u, -1.0, 1.0), xp.where(pos_u, 1.0, -1.0))
    Or1 = K_u * djrdE + s_u * (delta / sqrt2) * PE
    I3r1 = K_u * djrdI3 - s_u * (delta / sqrt2) * PI  # u-branch I3 has a leading minus
    aphi_u = K_u * djrdLz - s_u * (Lz / delta / sqrt2) * PL
    # ---- v-branch (8 leaves): panel by vx vs midpoints, K/s by (pvx x panel x vx</>pi/2)
    mid_v_pt = vmin + 0.5 * (pi / 2.0 - vmin)
    low_v = (vx < mid_v_pt) | (vx > (pi - mid_v_pt))
    above = vx > pi / 2.0
    base_v = xp.where(low_v, vmin, (pi / 2.0) * xp.ones_like(vx))
    sign_v = xp.where(low_v, xp.ones_like(vx), -xp.ones_like(vx))
    mid_v = xp.where(
        low_v,
        xp.where(above, xp.sqrt(xp.abs(pi - vx - vmin)), xp.sqrt(xp.abs(vx - vmin))),
        xp.sqrt(xp.abs(pi / 2.0 - vx)),
    )
    QE = AP(
        xp, JZsq, jz_args, lambda xp, v: xp.sin(v) ** 2.0, base_v, sign_v, mid_v, order
    )
    QI = AP(
        xp, JZsq, jz_args, lambda xp, v: xp.ones_like(v), base_v, sign_v, mid_v, order
    )
    QL = AP(
        xp,
        JZsq,
        jz_args,
        lambda xp, v: 1.0 / xp.sin(v) ** 2.0,
        base_v,
        sign_v,
        mid_v,
        order,
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
    Omegar, Omegaphi, Omegaz = _staeckel_freqs(
        xp, s, umin, umax, vmin, pot, delta, order
    )
    angler, anglephi, anglez = _staeckel_angles(
        xp, s, umin, umax, vmin, pot, delta, order
    )
    anglephi = xp.remainder(anglephi + phi, 2.0 * numpy.pi)  # fold in the azimuth
    return jr, s["Lz"], jz, Omegar, Omegaphi, Omegaz, angler, anglephi, anglez


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
            Number of points to use in the Gauss-Legendre numerical integration of the relevant action, frequency, and angle integrals. Default is 10.
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
            # replacing the per-object actionAngleStaeckelSingle scipy loop. Uses
            # plain GL order-`order` to match the C path (the default GL order);
            # the standalone-actions c=False result is thus now consistent with
            # both c=True and _actionsFreqsAngles (was ~1e-5 off via adaptive quad).
            xp = get_namespace(R) if is_backend_array(R) else numpy
            jr, Lz, jz = _staeckel_actions(
                xp, R, vR, vT, z, vz, self._pot, _coerce_delta_arraylike(delta), order
            )
            if is_backend_array(R):
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
            xp = get_namespace(R) if is_backend_array(R) else numpy
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
            xp = get_namespace(R) if is_backend_array(R) else numpy
            if is_backend_array(R) and not is_backend_array(phi):
                phi = xp.asarray(phi)  # fold the azimuth in R's namespace
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
            xp = get_namespace(R) if is_backend_array(R) else numpy
            # _staeckel_prep already snaps vmin to pi/2 for planar orbits.
            _, umin, umax, vmin, _ = _staeckel_prep(
                xp, R, vR, vT, z, vz, self._pot, delta
            )
            return (umin, umax, vmin)


class actionAngleStaeckelSingle(actionAngle):
    """Action-angle formalism for axisymmetric potentials using Binney (2012)'s Staeckel approximation"""

    def __init__(self, *args, **kwargs):
        """
        Initialize an actionAngleStaeckelSingle object

        Parameters
        ----------
        *args : tuple
            Either:
            a) R,vR,vT,z,vz[,phi]:
                1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)
                2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
            b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
        pot: Potential or a combined potential formed using addition (pot1+pot2+…)
            Potential to use
        delta: float, optional
            focal length of confocal coordinate system

        Notes
        -----
        - 2012-11-27 - Written - Bovy (IAS)
        """
        self._parse_eval_args(*args, _noOrbUnitsCheck=True, **kwargs)
        self._R = self._eval_R
        self._vR = self._eval_vR
        self._vT = self._eval_vT
        self._z = self._eval_z
        self._vz = self._eval_vz
        if not "pot" in kwargs:  # pragma: no cover
            raise OSError("Must specify pot= for actionAngleStaeckelSingle")
        self._pot = kwargs["pot"]
        if not "delta" in kwargs:  # pragma: no cover
            raise OSError("Must specify delta= for actionAngleStaeckel")
        self._delta = _coerce_delta_arraylike(kwargs["delta"])
        # Pre-calculate everything
        self._ux, self._vx = coords.Rz_to_uv(self._R, self._z, delta=self._delta)
        self._sinvx = numpy.sin(self._vx)
        self._cosvx = numpy.cos(self._vx)
        self._coshux = numpy.cosh(self._ux)
        self._sinhux = numpy.sinh(self._ux)
        self._pux = self._delta * (
            self._vR * self._coshux * self._sinvx
            + self._vz * self._sinhux * self._cosvx
        )
        self._pvx = self._delta * (
            self._vR * self._sinhux * self._cosvx
            - self._vz * self._coshux * self._sinvx
        )
        EL = self.calcEL()
        self._E = EL[0]
        self._Lz = EL[1]
        # Determine umin and umax
        self._u0 = kwargs.pop(
            "u0", self._ux
        )  # u0 as defined by Binney does not matter for a
        # single action evaluation, so we don't determine it here
        self._sinhu0 = numpy.sinh(self._u0)
        # All Staeckel integrals (actions, frequencies, angles) use v0=pi/2 for
        # the u (J_R) integral and u0 for the v (J_z) integral, matching the C
        # implementation. (_v0u is still overridable.)
        self._v0u = kwargs.pop("_v0u", numpy.pi / 2.0)
        self._sinv0u = numpy.sin(self._v0u)
        self._potu0v0 = potentialStaeckel(self._u0, self._v0u, self._pot, self._delta)
        # I3U with the dU reference at (u0, v0u); robust to u0!=ux (useu0=True),
        # reduces to the bare I3 when u0=ux.
        self._I3U = (
            self._E * self._sinhux**2.0
            - self._pux**2.0 / 2.0 / self._delta**2.0
            - self._Lz**2.0 / 2.0 / self._delta**2.0 / self._sinhux**2.0
            - (self._sinhux**2.0 + self._sinv0u**2.0)
            * potentialStaeckel(self._ux, self._v0u, self._pot, self._delta)
            + (self._sinhu0**2.0 + self._sinv0u**2.0) * self._potu0v0
        )
        self._u0v = self._u0
        self._coshu0v = numpy.cosh(self._u0v)
        self._sinhu0v = numpy.sinh(self._u0v)
        self._potupi2 = potentialStaeckel(
            self._u0v, numpy.pi / 2.0, self._pot, self._delta
        )
        dV = self._coshu0v**2.0 * self._potupi2 - (
            self._sinhu0v**2.0 + self._sinvx**2.0
        ) * potentialStaeckel(self._u0v, self._vx, self._pot, self._delta)
        self._I3V = (
            -self._E * self._sinvx**2.0
            + self._pvx**2.0 / 2.0 / self._delta**2.0
            + self._Lz**2.0 / 2.0 / self._delta**2.0 / self._sinvx**2.0
            - dV
        )
        self.calcUminUmax()
        self.calcVmin()
        return None

    def angleR(self, **kwargs):
        raise NotImplementedError(
            "'angleR' not yet implemented for Staeckel approximation"
        )

    def TR(self, **kwargs):
        raise NotImplementedError("'TR' not implemented yet for Staeckel approximation")

    def Tphi(self, **kwargs):
        raise NotImplementedError(
            "'Tphi' not implemented yet for Staeckel approxximation"
        )

    def I(self, **kwargs):
        raise NotImplementedError("'I' not implemented yet for Staeckel approxximation")

    def Jphi(self):  # pragma: no cover
        return self._R * self._vT

    def JR(self, **kwargs):
        """
        Calculate the radial action

        Parameters
        ----------
        fixed_quad : bool, optional
            If True, use n=10 fixed_quad. Default is False.
        **kwargs
            scipy.integrate.quad keywords

        Returns
        -------
        float
            J_R(R,vT,vT)/ro/vc + estimate of the error (nan for fixed_quad)

        Notes
        -----
        - 2012-11-27 - Written - Bovy (IAS)

        """
        if hasattr(self, "_JR"):  # pragma: no cover
            return self._JR
        umin, umax = self.calcUminUmax()
        # print self._ux, self._pux, (umax-umin)/umax
        if (umax - umin) / umax < 10.0**-6:
            return numpy.array([0.0])
        order = kwargs.pop("order", 10)
        if kwargs.pop("fixed_quad", False):
            # factor in next line bc integrand=/2delta^2
            self._JR = (
                1.0
                / numpy.pi
                * numpy.sqrt(2.0)
                * self._delta
                * integrate.fixed_quad(
                    _JRStaeckelIntegrand,
                    umin,
                    umax,
                    args=(
                        self._E,
                        self._Lz,
                        self._I3U,
                        self._delta,
                        self._u0,
                        self._sinhu0**2.0,
                        self._v0u,
                        self._sinv0u**2.0,
                        self._potu0v0,
                        self._pot,
                    ),
                    n=order,
                    **kwargs,
                )[0]
            )
        else:
            self._JR = (
                1.0
                / numpy.pi
                * numpy.sqrt(2.0)
                * self._delta
                * integrate.quad(
                    _JRStaeckelIntegrand,
                    umin,
                    umax,
                    args=(
                        self._E,
                        self._Lz,
                        self._I3U,
                        self._delta,
                        self._u0,
                        self._sinhu0**2.0,
                        self._v0u,
                        self._sinv0u**2.0,
                        self._potu0v0,
                        self._pot,
                    ),
                    **kwargs,
                )[0]
            )
        return self._JR

    def Jz(self, **kwargs):
        """
        Calculate the vertical action

        Parameters
        ----------
        fixed_quad : bool, optional
            If True, use n=10 fixed_quad. Default is False.
        **kwargs
            scipy.integrate.quad keywords

        Returns
        -------
        float
            J_z(R,vT,vT)/ro/vc + estimate of the error

        Notes
        -----
        - 2012-11-27 - Written - Bovy (IAS)
        """
        if hasattr(self, "_JZ"):  # pragma: no cover
            return self._JZ
        vmin = self.calcVmin()
        if (numpy.pi / 2.0 - vmin) < 10.0**-7:
            return numpy.array([0.0])
        order = kwargs.pop("order", 10)
        if kwargs.pop("fixed_quad", False):
            # factor in next line bc integrand=/2delta^2
            self._JZ = (
                2.0
                / numpy.pi
                * numpy.sqrt(2.0)
                * self._delta
                * integrate.fixed_quad(
                    _JzStaeckelIntegrand,
                    vmin,
                    numpy.pi / 2,
                    args=(
                        self._E,
                        self._Lz,
                        self._I3V,
                        self._delta,
                        self._u0v,
                        self._coshu0v**2.0,
                        self._sinhu0v**2.0,
                        self._potupi2,
                        self._pot,
                    ),
                    n=order,
                    **kwargs,
                )[0]
            )
        else:
            # factor in next line bc integrand=/2delta^2
            self._JZ = (
                2.0
                / numpy.pi
                * numpy.sqrt(2.0)
                * self._delta
                * integrate.quad(
                    _JzStaeckelIntegrand,
                    vmin,
                    numpy.pi / 2,
                    args=(
                        self._E,
                        self._Lz,
                        self._I3V,
                        self._delta,
                        self._u0v,
                        self._coshu0v**2.0,
                        self._sinhu0v**2.0,
                        self._potupi2,
                        self._pot,
                    ),
                    **kwargs,
                )[0]
            )
        return self._JZ

    def calcEL(self, **kwargs):
        """
        Calculate the energy and angular momentum.

        Parameters
        ----------
        **kwargs : dict
            scipy.integrate.quadrature keywords

        Returns
        -------
        tuple
            A tuple containing the energy and angular momentum.

        Notes
        -----
        - 2012-11-27 - Written - Bovy (IAS)
        """
        E, L = calcELStaeckel(self._R, self._vR, self._vT, self._z, self._vz, self._pot)
        return (E, L)

    def calcUminUmax(self, **kwargs):
        """
        Calculate the u 'apocenter' and 'pericenter'

        Returns
        -------
        tuple
            (umin,umax)

        Notes
        -----
        - 2012-11-27 - Written - Bovy (IAS)
        """
        if hasattr(self, "_uminumax"):  # pragma: no cover
            return self._uminumax
        E, L = self._E, self._Lz
        # Calculate value of the integrand at current point, to check whether
        # we are at a turning point
        current_val = _JRStaeckelIntegrandSquared(
            self._ux,
            E,
            L,
            self._I3U,
            self._delta,
            self._u0,
            self._sinhu0**2.0,
            self._v0u,
            self._sinv0u**2.0,
            self._potu0v0,
            self._pot,
        )
        if (
            numpy.fabs(self._pux) < 1e-7 or numpy.fabs(current_val) < 1e-10
        ):  # We are at umin or umax
            eps = 10.0**-8.0
            peps = _JRStaeckelIntegrandSquared(
                self._ux + eps,
                E,
                L,
                self._I3U,
                self._delta,
                self._u0,
                self._sinhu0**2.0,
                self._v0u,
                self._sinv0u**2.0,
                self._potu0v0,
                self._pot,
            )
            meps = _JRStaeckelIntegrandSquared(
                self._ux - eps,
                E,
                L,
                self._I3U,
                self._delta,
                self._u0,
                self._sinhu0**2.0,
                self._v0u,
                self._sinv0u**2.0,
                self._potu0v0,
                self._pot,
            )
            if peps < 0.0 and meps > 0.0:  # we are at umax
                umax = self._ux
                rstart, prevr = _uminUmaxFindStart(
                    self._ux,
                    E,
                    L,
                    self._I3U,
                    self._delta,
                    self._u0,
                    self._sinhu0**2.0,
                    self._v0u,
                    self._sinv0u**2.0,
                    self._potu0v0,
                    self._pot,
                )
                if rstart == 0.0:
                    umin = 0.0
                else:
                    try:
                        umin = optimize.brentq(
                            _JRStaeckelIntegrandSquared,
                            numpy.atleast_1d(rstart)[0],
                            numpy.atleast_1d(self._ux)[0] - eps,
                            (
                                E,
                                L,
                                self._I3U,
                                self._delta,
                                self._u0,
                                self._sinhu0**2.0,
                                self._v0u,
                                self._sinv0u**2.0,
                                self._potu0v0,
                                self._pot,
                            ),
                            maxiter=200,
                        )
                    except RuntimeError:  # pragma: no cover
                        raise UnboundError("Orbit seems to be unbound")
            elif peps > 0.0 and meps < 0.0:  # we are at umin
                umin = self._ux
                rend, prevr = _uminUmaxFindStart(
                    self._ux,
                    E,
                    L,
                    self._I3U,
                    self._delta,
                    self._u0,
                    self._sinhu0**2.0,
                    self._v0u,
                    self._sinv0u**2.0,
                    self._potu0v0,
                    self._pot,
                    umax=True,
                )
                umax = optimize.brentq(
                    _JRStaeckelIntegrandSquared,
                    numpy.atleast_1d(self._ux)[0] + eps,
                    numpy.atleast_1d(rend)[0],
                    (
                        E,
                        L,
                        self._I3U,
                        self._delta,
                        self._u0,
                        self._sinhu0**2.0,
                        self._v0u,
                        self._sinv0u**2.0,
                        self._potu0v0,
                        self._pot,
                    ),
                    maxiter=200,
                )
            else:  # circular orbit
                umin = self._ux
                umax = self._ux
        else:
            rstart, prevr = _uminUmaxFindStart(
                self._ux,
                E,
                L,
                self._I3U,
                self._delta,
                self._u0,
                self._sinhu0**2.0,
                self._v0u,
                self._sinv0u**2.0,
                self._potu0v0,
                self._pot,
            )
            if rstart == 0.0:  # pragma: no cover (plunge to u=0; bound orbits don't)
                umin = 0.0
            else:
                if numpy.fabs(prevr - self._ux) < 10.0**-2.0:
                    rup = self._ux
                else:
                    rup = prevr
                try:
                    umin = optimize.brentq(
                        _JRStaeckelIntegrandSquared,
                        rstart,
                        rup,
                        (
                            E,
                            L,
                            self._I3U,
                            self._delta,
                            self._u0,
                            self._sinhu0**2.0,
                            self._v0u,
                            self._sinv0u**2.0,
                            self._potu0v0,
                            self._pot,
                        ),
                        maxiter=200,
                    )
                except RuntimeError:  # pragma: no cover
                    raise UnboundError("Orbit seems to be unbound")
            rend, prevr = _uminUmaxFindStart(
                self._ux,
                E,
                L,
                self._I3U,
                self._delta,
                self._u0,
                self._sinhu0**2.0,
                self._v0u,
                self._sinv0u**2.0,
                self._potu0v0,
                self._pot,
                umax=True,
            )
            umax = optimize.brentq(
                _JRStaeckelIntegrandSquared,
                prevr,
                rend,
                (
                    E,
                    L,
                    self._I3U,
                    self._delta,
                    self._u0,
                    self._sinhu0**2.0,
                    self._v0u,
                    self._sinv0u**2.0,
                    self._potu0v0,
                    self._pot,
                ),
                maxiter=200,
            )
        self._uminumax = (umin, umax)
        return self._uminumax

    def calcVmin(self, **kwargs):
        """
        Calculate the v 'pericenter'

        Returns
        -------
        float
            v_min(R,vT,vT)/vc + estimate of the error

        Notes
        -----
        - 2012-11-28 - Written - Bovy (IAS)
        """
        if hasattr(self, "_vmin"):  # pragma: no cover
            return self._vmin
        E, L = self._E, self._Lz
        if numpy.fabs(self._pvx) < 10.0**-7.0:  # We are at vmin or vmax
            eps = 10.0**-8.0
            peps = _JzStaeckelIntegrandSquared(
                self._vx + eps,
                E,
                L,
                self._I3V,
                self._delta,
                self._u0v,
                self._coshu0v**2.0,
                self._sinhu0v**2.0,
                self._potupi2,
                self._pot,
            )
            meps = _JzStaeckelIntegrandSquared(
                self._vx - eps,
                E,
                L,
                self._I3V,
                self._delta,
                self._u0v,
                self._coshu0v**2.0,
                self._sinhu0v**2.0,
                self._potupi2,
                self._pot,
            )
            if peps < 0.0 and meps > 0.0:  # pragma: no cover
                # we are at vmax, which cannot happen
                raise RuntimeError(
                    "Orbit is at the vmax turning point in v, which mathematically cannot happen; something is very wrong!!"
                )
            elif peps > 0.0 and meps < 0.0:  # we are at vmin
                vmin = self._vx
            else:  # planar orbit
                vmin = self._vx
        else:
            rstart = _vminFindStart(
                self._vx,
                E,
                L,
                self._I3V,
                self._delta,
                self._u0v,
                self._coshu0v**2.0,
                self._sinhu0v**2.0,
                self._potupi2,
                self._pot,
            )
            if rstart == 0.0:  # pragma: no cover (reach v=0 pole; bound orbits don't)
                vmin = 0.0
            else:
                try:
                    vmin = optimize.brentq(
                        _JzStaeckelIntegrandSquared,
                        rstart,
                        rstart / 0.9,
                        (
                            E,
                            L,
                            self._I3V,
                            self._delta,
                            self._u0v,
                            self._coshu0v**2.0,
                            self._sinhu0v**2.0,
                            self._potupi2,
                            self._pot,
                        ),
                        maxiter=200,
                    )
                except RuntimeError:  # pragma: no cover
                    raise UnboundError("Orbit seems to be unbound")
        self._vmin = vmin
        return self._vmin


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


def _JRStaeckelIntegrand(u, E, Lz, I3U, delta, u0, sinh2u0, v0, sin2v0, potu0v0, pot):
    return numpy.sqrt(
        _JRStaeckelIntegrandSquared(
            u, E, Lz, I3U, delta, u0, sinh2u0, v0, sin2v0, potu0v0, pot
        )
    )


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


def _JzStaeckelIntegrand(v, E, Lz, I3V, delta, u0, cosh2u0, sinh2u0, potu0pi2, pot):
    return numpy.sqrt(
        _JzStaeckelIntegrandSquared(
            v, E, Lz, I3V, delta, u0, cosh2u0, sinh2u0, potu0pi2, pot
        )
    )


def _JzStaeckelIntegrandSquared(
    v, E, Lz, I3V, delta, u0, cosh2u0, sinh2u0, potu0pi2, pot
):
    # potu0pi2= potentialStaeckel(u0,numpy.pi/2.,pot,delta)
    """The J_z integrand: p_v(v)/2/delta^2"""
    xp = get_namespace(v) if is_backend_array(v) else numpy
    sin2v = xp.sin(v) ** 2.0
    dV = cosh2u0 * potu0pi2 - (sinh2u0 + sin2v) * potentialStaeckel(u0, v, pot, delta)
    return E * sin2v + I3V + dV - Lz**2.0 / 2.0 / delta**2.0 / sin2v


def _uminUmaxFindStart(
    u, E, Lz, I3U, delta, u0, sinh2u0, v0, sin2v0, potu0v0, pot, umax=False
):
    """
    Find adequate start or end points to solve for umin and umax

    Parameters
    ----------
    u : float
        Current value of the coordinate to solve for (either umin or umax)
    E : float
        Energy
    Lz : float
        Angular momentum along z
    I3U : float
        Third isolating integral of motion
    delta : float
        Focus parameter of the confocal coordinate system
    u0 : float
        u coordinate of the center of the coordinate system
    sinh2u0 : float
        Hyperbolic sine of twice the u coordinate of the center of the coordinate system
    v0 : float
        v coordinate of the center of the coordinate system
    sin2v0 : float
        Sine of twice the v coordinate of the center of the coordinate system
    potu0v0 : float
        Potential at the center of the coordinate system
    pot : Potential object
        Instance of a galpy Potential object
    umax : bool, optional
        If True, solve for umax instead of umin (default is False)

    Returns
    -------
    float
        Adequate start or end point to solve for umin or umax

    Notes
    -----
    - 2012-11-30 - Written - Bovy (IAS)
    """
    if umax:
        utry = u * 1.1
    else:
        utry = u * 0.9
    prevu = u
    while (
        _JRStaeckelIntegrandSquared(
            utry, E, Lz, I3U, delta, u0, sinh2u0, v0, sin2v0, potu0v0, pot
        )
        >= 0.0
        and utry > 0.000000001
    ):
        prevu = utry
        if umax:
            if utry > 100.0:
                raise UnboundError("Orbit seems to be unbound")
            utry *= 1.1
        else:
            utry *= 0.9
    if utry < 0.000000001:
        return (0.0, prevu)
    return (utry, prevu)


def _vminFindStart(v, E, Lz, I3V, delta, u0, cosh2u0, sinh2u0, potu0pi2, pot):
    """
    Find adequate start point to solve for vmin

    Parameters
    ----------
    v : float
        Velocity
    E : float
        Energy
    Lz : float
        Angular momentum along z-axis
    I3V : float
        Third isolating integral
    delta : float
        Staeckel delta parameter
    u0 : float
        Staeckel energy
    cosh2u0 : float
        Hyperbolic cosine squared of u0
    sinh2u0 : float
        Hyperbolic sine squared of u0
    potu0pi2 : float
        Potential at u0 times pi/2
    pot : Potential object
        galpy Potential object

    Returns
    -------
    float
        Adequate start point to solve for vmin

    Notes
    -----
    - 2012-11-28 - Written - Bovy (IAS)
    """
    vtry = 0.9 * v
    while (
        _JzStaeckelIntegrandSquared(
            vtry, E, Lz, I3V, delta, u0, cosh2u0, sinh2u0, potu0pi2, pot
        )
        >= 0.0
        and vtry > 0.000000001
    ):
        vtry *= 0.9
    if (
        vtry < 0.000000001
    ):  # pragma: no cover (degenerate v=0 start; bound orbits don't)
        return 0.0
    return vtry if vtry >= 0.000000001 else 0.0


def _u0Equation(u, E, Lz22delta, delta, pot):
    """Port of the C u0Equation: the quantity minimized to obtain u0."""
    sinh2u = numpy.sinh(u) ** 2.0
    cosh2u = numpy.cosh(u) ** 2.0
    dU = cosh2u * potentialStaeckel(u, numpy.pi / 2.0, pot, delta)
    return -(E * sinh2u - dU - Lz22delta / sinh2u)


def calcu0(E, Lz, pot, delta):
    """
    Calculate u0 in the Staeckel approximation (pure-Python port of the C calcu0).

    Parameters
    ----------
    E : numpy.ndarray
        Energy.
    Lz : numpy.ndarray
        Angular momentum along z.
    pot : Potential object
        galpy Potential object or a combined potential.
    delta : float or numpy.ndarray
        Focus.

    Returns
    -------
    numpy.ndarray
        u0 for each point.

    Notes
    -----
    - Port of the C calcu0; minimizes _u0Equation over u in [0.001,100].
    """
    E = numpy.atleast_1d(E)
    Lz = numpy.atleast_1d(Lz)
    delta = numpy.atleast_1d(delta)
    delta_stride = 0 if len(delta) == 1 else 1
    out = numpy.empty(len(E))
    for ii in range(len(E)):
        tdelta = delta[ii * delta_stride]
        Lz22delta = 0.5 * Lz[ii] ** 2.0 / tdelta**2.0
        res = optimize.minimize_scalar(
            _u0Equation,
            bounds=(0.001, 100.0),
            args=(E[ii], Lz22delta, tdelta, pot),
            method="bounded",
            options={"xatol": 1e-12},
        )
        out[ii] = res.x
    return out


# coerce_backend=False: numpy-only body (in-place z masking, numpy.array/median)
@potential_physical_input(coerce_backend=False)
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
