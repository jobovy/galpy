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
import math
import warnings
from functools import lru_cache

import numpy

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
)
from ..util import conversion, coords, galpyWarning  # for prolate confocal transforms
from ..util.conversion import physical_conversion, potential_physical_input
from . import actionAngleStaeckel_c
from .actionAngle import UnboundError, actionAngle
from .actionAngleStaeckel_c import _ext_loaded as ext_loaded


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
    """A plain Python sequence delta (individual deltas) as an ndarray, so it
    broadcasts against the phase-space arrays; scalars and arrays pass through"""
    return numpy.array(delta) if isinstance(delta, (list, tuple)) else delta


class _PointwisePotential(Potential):
    """A potential with a component whose methods take scalar coordinates only
    (e.g., DoubleExponentialDiskPotential), evaluated point by point for the
    vectorized core"""

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


def _staeckel_pot(pot):
    """pot, or its point-by-point evaluation if a component's potential or
    forces take scalar coordinates only"""
    comps = list(pot) if isinstance(pot, CompositePotential) else [pot]
    if all(_accepts_node_array(p, "_evaluate", "_Rforce", "_zforce") for p in comps):
        return pot
    return _PointwisePotential(pot)


# ----------------------------------------------------------------------------
# The pure-Python (c=False) Staeckel core, vectorized over all phase-space
# points (as on galpy's v2 branch): elementwise setup, the turning points by a
# fixed-schedule bracket-and-bisect, and the action, frequency, and angle
# integrals by the chi-anomaly quadrature of _staeckel_chi_quads
# (q = qmin + D sin^2(chi/2) makes sqrt(S) analytic at the turning points) on
# C's mesh (_staeckel_nchi). v0=pi/2 for the u (J_R) integral, u0=ux for the
# v (J_z) integral (u0 does not matter for the actions, frequencies, and
# angles).

# cap on the bisection halvings (a degenerate, zero-width bracket gets it)
_BISECT_MAXITER = 100
# the fixed point of the Staeckel model of a torus of inverse= through a
# point: tolerance on the step of the reference curve, and cap on the cycles
_MODEL_TOL = 1e-9
_MODEL_MAXITER = 10


def _fixed_point_error(x0, x1, x2):
    """The distance left to the fixed point of a linearly converging
    iteration after its iterates x0, x1, x2: that of x2 itself (the
    geometric series of the remaining steps), which bounds that of Aitken's
    extrapolation of the three"""
    d1, d2 = numpy.fabs(x1 - x0), numpy.fabs(x2 - x1)
    rate = numpy.clip(d2 / numpy.where(d1 > 0.0, d1, 1.0), 0.0, 0.9)
    return numpy.where(numpy.isfinite(d2), d2 * rate / (1.0 - rate), 0.0)


def _iterate_bracket(step, x0, n):
    """Run x = step(x) up to n times, stopping at a fixed point"""
    for _ in range(n):
        x1 = step(x0)
        if numpy.all(x1 == x0):
            return x1
        x0 = x1
    return x0


def _bisect_root(f, a, b, xtol, maxiter):
    """Root of f on [a, b] (arrays, f(a) and f(b) of opposite sign) by a fixed
    number of halvings: enough to shrink the widest bracket below xtol"""
    lo = numpy.asarray(a) * 1.0
    hi = numpy.asarray(b) * 1.0
    width = float(numpy.max(numpy.abs(numpy.asarray(b) - numpy.asarray(a))))
    n = (
        max(1, min(int(math.ceil(math.log2(width / xtol))) + 1, maxiter))
        if width > 0.0
        else min(maxiter, _BISECT_MAXITER)
    )
    slo = numpy.sign(f(lo))
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        same = numpy.sign(f(mid)) == slo
        lo, hi = numpy.where(same, mid, lo), numpy.where(same, hi, mid)
    return 0.5 * (lo + hi)


def _staeckel_setup(R, vR, vT, z, vz, pot, delta, u0=None):
    """The setup quantities of every phase-space point; with u0, for the
    Staeckel model of (delta, u0): its reference curve and the energy in
    it, Phi = [U(u) - V(v)] / (sinh^2 u + sin^2 v) in place of the
    potential itself"""
    ux, vx = coords.Rz_to_uv(R, z, delta=delta)
    sinvx, cosvx = numpy.sin(vx), numpy.cos(vx)
    coshux, sinhux = numpy.cosh(ux), numpy.sinh(ux)
    pux = delta * (vR * coshux * sinvx + vz * sinhux * cosvx)
    pvx = delta * (vR * sinhux * cosvx - vz * coshux * sinvx)
    E, Lz = calcELStaeckel(R, vR, vT, z, vz, pot)
    if u0 is None:
        u0 = ux  # u0 does not matter for a single action evaluation
    else:
        E = (
            (
                coshux**2.0 * potentialStaeckel(ux, numpy.pi / 2.0, pot, delta)
                - numpy.cosh(u0) ** 2.0
                * potentialStaeckel(u0, numpy.pi / 2.0, pot, delta)
                + (numpy.sinh(u0) ** 2.0 + sinvx**2.0)
                * potentialStaeckel(u0, vx, pot, delta)
            )
            / (sinhux**2.0 + sinvx**2.0)
            + vR**2.0 / 2.0
            + vT**2.0 / 2.0
            + vz**2.0 / 2.0
        )
    sinh2u0 = numpy.sinh(u0) ** 2.0
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
    cosh2u0v = numpy.cosh(u0) ** 2.0
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


def _staeckel_jr_args(s, pot, delta):
    return (s["E"], s["Lz"], s["I3U"], delta, s["u0"], s["sinh2u0"],
            s["v0u"], s["sin2v0u"], s["potu0v0"], pot)  # fmt: skip


def _staeckel_jz_args(s, pot, delta):
    return (s["E"], s["Lz"], s["I3V"], delta, s["u0"], s["cosh2u0v"],
            s["sinh2u0v"], s["potupi2"], pot)  # fmt: skip


def _staeckel_uminumax(s, pot, delta):
    """(umin, umax, unbound): bracket-and-bisect roots of the J_R integrand^2"""
    args = _staeckel_jr_args(s, pot, delta)
    f = lambda u: _JRStaeckelIntegrandSquared(u, *args)
    ux, eps = s["ux"], 1e-8
    at_turn = (numpy.abs(s["pux"]) < 1e-7) | (numpy.abs(f(ux)) < 1e-10)
    peps, meps = f(ux + eps), f(ux - eps)
    at_umin = at_turn & (peps > 0.0) & (meps < 0.0)
    at_umax = at_turn & (peps < 0.0) & (meps > 0.0)
    circular = at_turn & ~at_umin & ~at_umax
    # lower bracket: halve below ux until f < 0 (60 halvings reach ~1e-18, so
    # even a near-axis turning point of a nearly radial orbit is straddled)
    lo = _iterate_bracket(
        lambda l: numpy.where((f(l) >= 0.0) & (l > 1e-10), l * 0.5, l), ux * 0.5, 60
    )
    # f still >= 0 at the floor: the orbit reaches the symmetry axis, umin=0
    reaches_axis = f(lo) >= 0.0
    # upper bracket: expand above ux until f < 0 (up to u=100)
    hi = _iterate_bracket(
        lambda h: numpy.where((f(h) >= 0.0) & (h < 100.0), h * 1.1, h), ux * 1.1, 80
    )
    # no upper turning point below u=100: unbound
    unbound = (f(100.0 * numpy.ones_like(ux)) >= 0.0) & ~(at_umax | circular)
    # an orbit exactly at a turning point has f(ux) ~ 0 of either sign, so the
    # other turning point is bracketed from strictly inside the allowed region
    u_lo_umax = numpy.where(at_umin, ux + eps, ux)
    u_hi_umin = numpy.where(at_umax, ux - eps, ux)
    umin = _bisect_root(f, lo, u_hi_umin, xtol=1e-13, maxiter=200)
    umax = _bisect_root(f, u_lo_umax, hi, xtol=1e-13, maxiter=200)
    umin = numpy.where(at_umin | circular, ux, umin)
    umax = numpy.where(at_umax | circular, ux, umax)
    # a circular orbit is a double root of f at ux, so f(ux +- eps) is at the
    # level of f's rounding noise and its classification as at a turning point
    # is decided by the last bits: a turning-point pair closer than the
    # frequencies' circular threshold cannot be resolved, so it is circular
    unresolved = at_turn & ((umax - umin) < 1e-6 * umax)
    umin = numpy.where(unresolved, ux, umin)
    umax = numpy.where(unresolved, ux, umax)
    umin = numpy.where(reaches_axis, numpy.zeros_like(umin), umin)
    return umin, umax, unbound


def _staeckel_vmin(s, pot, delta):
    """vmin: bracket-and-bisect root of the J_z integrand^2 below vx"""
    args = _staeckel_jz_args(s, pot, delta)
    f = lambda v: _JzStaeckelIntegrandSquared(v, *args)
    vx, eps = s["vx"], 1e-8
    at_turn = (numpy.abs(s["pvx"]) < 1e-7) | (numpy.abs(f(vx)) < 1e-10)
    at_vmin = at_turn & (f(vx + eps) > 0.0) & (f(vx - eps) < 0.0)
    vlo = _iterate_bracket(
        lambda v: numpy.where((f(v) >= 0.0) & (v > 1e-9), v * 0.9, v), vx * 0.9, 80
    )
    vmin = _bisect_root(f, vlo, vx, xtol=1e-13, maxiter=200)
    return numpy.where(at_vmin, vx, vmin)


def _staeckel_eval_flat(func, q, args):
    """func(q, *args) on an (N, nnodes) grid, evaluated on the raveled grid
    (the potentials' functions take 1D coordinates) and reshaped"""
    shp = q.shape
    fargs = tuple(
        numpy.reshape(numpy.broadcast_to(a, shp), (-1,))
        if getattr(a, "ndim", 0) >= 1
        else a
        for a in args
    )
    return numpy.reshape(func(numpy.reshape(q, (-1,)), *fargs), shp)


def _staeckel_chi_mesh_edges(e):
    """10-node GL rule on each panel of the edge list e, raveled"""
    mid = 0.5 * (e[:-1] + e[1:])
    half = 0.5 * (e[1:] - e[:-1])
    t = (mid[:, None] + _CHIQUAD_GLX[None, :] * half[:, None]).ravel()
    w = (half[:, None] * _CHIQUAD_GLW[None, :]).ravel()
    return t, w


@lru_cache(maxsize=None)
def _staeckel_chi_mesh(nchi):
    """The composite rule on the normalized anomaly t in [0,1]: nchi uniform
    panels of the 10-node GL rule (normalized, so that the upper anomaly may
    differ per orbit, as for the angles' incomplete integrals)"""
    return _staeckel_chi_mesh_edges(numpy.linspace(0.0, 1.0, nchi + 1))


@lru_cache(maxsize=None)
def _staeckel_chi_mesh_nearaxis(nchi):
    """The J_R action meshes of a batch with near-axis orbits, at one node
    count so that the choice is per orbit: (uniform, graded). The graded one
    is C's glfixed_graded on the low half t in [0, 1/2] -- panels halving
    toward the cusp at the axis -- plus the uniform high half"""
    nhigh = nchi // 2
    lo = [0.0] + [0.5 * 2.0 ** (k + 1 - _STAECKEL_NEARAXIS_NPANELS)
                  for k in range(_STAECKEL_NEARAXIS_NPANELS)]  # fmt: skip
    hi = list(numpy.linspace(0.5, 1.0, nhigh + 1)[1:])
    graded = _staeckel_chi_mesh_edges(numpy.array(lo + hi))
    uniform = _staeckel_chi_mesh(_STAECKEL_NEARAXIS_NPANELS + nhigh)
    return uniform, graded


def _staeckel_chi_quads(
    Ssq, dSsq, args, qmin, D, order, chimax=numpy.pi, vside=False, nearaxis=None
):
    """Chi-anomaly quadratures of sqrt(S) over the turning-point interval.

    S vanishes linearly at each turning point, so sqrt(S) has a square-root
    branch point there. In the anomaly q = qmin + D sin^2(chi/2), with
    y = sin^2(chi/2) and Q = S/[y(1-y)] smooth and nonzero,

        int sqrt(S) dq = (D/4) int sqrt(Q) sin^2(chi) dchi,
        int f/sqrt(S) dq = D int f/sqrt(Q) dchi,

    both analytic in chi (the rule C is regularized to, so c=True and c=False
    agree to machine precision). Near a turning point the direct S is a
    difference of O(1) potential terms, which dividing by y(1-y) amplifies;
    there Q is rebuilt from the analytic derivative as
    S ~ (q - q0)[S'(q0) + S'(q)]/2.

    chimax is the anomaly of the upper limit: pi for a complete integral, pi/2
    for the z action (whose anomaly spans the whole v loop [vmin, pi - vmin];
    the midplane is a symmetry point of S_z), or a per-orbit
    2 arcsin(sqrt((q-qmin)/D)) for the angles' incomplete integrals.

    nearaxis (the J_R action only, as in C) masks the orbits reaching the axis,
    umin < _STAECKEL_NEARAXIS: there S picks up the potential's inner cusp,
    S ~ A - B u^0.2, against which a uniform rule converges only algebraically
    and the S' edge model is wrong; those orbits get C's graded mesh and the
    direct Q, the rest of the batch the uniform rule at the same node count.

    Returns (action, sqrt(Q), q, wts), so that every quadrature on this mesh
    comes from one evaluation of S.
    """
    nchi = _staeckel_nchi(order, vside)
    if nearaxis is None:
        t, w = _staeckel_chi_mesh(nchi)
    else:
        (tu, wu), (tg, wg) = _staeckel_chi_mesh_nearaxis(nchi)
        na = nearaxis[..., None]
        t = numpy.where(na, tg, tu)
        w = numpy.where(na, wg, wu)
    chi = chimax * t
    wts = chimax * w
    y = numpy.sin(chi / 2.0) ** 2.0
    y1my = y * (1.0 - y)
    # masks: the denominators of the unused branches are kept finite
    ones = numpy.ones_like(y1my)
    is_edge = y1my <= _STAECKEL_CHI_EDGE
    if nearaxis is not None:
        is_edge = is_edge & ~na
    is_lo = y < 0.5
    y1my_safe = numpy.where(is_edge, ones, y1my)
    omy_safe = numpy.where(is_lo, 1.0 - y, ones)
    y_safe = numpy.where(is_lo, ones, y)
    a2 = tuple(x[..., None] if getattr(x, "ndim", 0) >= 1 else x for x in args)
    q = qmin[..., None] + D[..., None] * y
    Q = _staeckel_eval_flat(Ssq, q, a2) / y1my_safe
    dS_n = _staeckel_eval_flat(dSsq, q, a2)
    dS_lo = _staeckel_eval_flat(dSsq, qmin[..., None], a2)
    dS_hi = _staeckel_eval_flat(dSsq, (qmin + D)[..., None], a2)
    Q_edge = numpy.where(
        is_lo,
        D[..., None] * (dS_lo + dS_n) / 2.0 / omy_safe,
        D[..., None] * (-dS_hi - dS_n) / 2.0 / y_safe,
    )
    Q = numpy.where(is_edge, Q_edge, Q)
    tiny = numpy.finfo(float).tiny
    Q = numpy.where(Q > tiny, Q, tiny * ones)
    sqQ = numpy.sqrt(Q)
    action = (D / 4.0) * numpy.sum(wts * sqQ * numpy.sin(chi) ** 2.0, axis=-1)
    return action, sqQ, q, wts


def _staeckel_prep(R, vR, vT, z, vz, pot, delta, u0=None):
    """Setup quantities and turning points, shared by the actions,
    frequencies, and angles: (setup, umin, umax, vmin)"""
    s = _staeckel_setup(R, vR, vT, z, vz, pot, delta, u0=u0)
    umin, umax, unbound = _staeckel_uminumax(s, pot, delta)
    if numpy.any(unbound):
        raise UnboundError("Orbit seems to be unbound")
    vmin = _staeckel_vmin(s, pot, delta)
    # planar orbit (jz=0): snap vmin to exactly pi/2 (the bisection lands
    # ~1e-8 off), so that jz=0, the J_z panels vanish, and zmax=0 exactly
    vmin = numpy.where(
        (numpy.pi / 2.0 - vmin) < 1e-7, numpy.pi / 2.0 * numpy.ones_like(vmin), vmin
    )
    return s, umin, umax, vmin


def _staeckel_jr_jz(s, umin, umax, vmin, pot, delta, order):
    """(jr, jz) from the setup quantities and turning points"""
    sqrt2 = numpy.sqrt(2.0)
    jr = (
        _staeckel_chi_quads(
            _JRStaeckelIntegrandSquared,
            _dJRStaeckelIntegrandSquareddu,
            _staeckel_jr_args(s, pot, delta),
            umin,
            umax - umin,
            order,
            nearaxis=umin < _STAECKEL_NEARAXIS,
        )[0]
        * sqrt2
        * delta
        / numpy.pi
    )
    jz = (
        _staeckel_chi_quads(
            _JzStaeckelIntegrandSquared,
            _dJzStaeckelIntegrandSquareddv,
            _staeckel_jz_args(s, pot, delta),
            vmin,
            numpy.pi - 2.0 * vmin,  # the full v loop; the midplane is chi = pi/2
            order,
            chimax=numpy.pi / 2.0,
            vside=True,
        )[0]
        * 2.0
        * sqrt2
        * delta
        / numpy.pi
    )
    jr = numpy.where((umax - umin) / umax < 1e-6, numpy.zeros_like(jr), jr)
    jz = numpy.where((numpy.pi / 2.0 - vmin) < 1e-7, numpy.zeros_like(jz), jz)
    return jr, jz


def _staeckel_actions(R, vR, vT, z, vz, pot, delta, order, u0=None):
    """(jr, Lz, jz)"""
    s, umin, umax, vmin = _staeckel_prep(R, vR, vT, z, vz, pot, delta, u0=u0)
    jr, jz = _staeckel_jr_jz(s, umin, umax, vmin, pot, delta, order)
    return jr, s["Lz"], jz


# Weight functions of the 1/p_u and 1/p_v profile integrals, in the order
# (dE, dI3, dLz)
def _sinh2(u):
    return numpy.sinh(u) ** 2.0


def _sin2(v):
    return numpy.sin(v) ** 2.0


def _one_over_sinh2(u):
    return 1.0 / numpy.sinh(u) ** 2.0


def _one_over_sin2(v):
    return 1.0 / numpy.sin(v) ** 2.0


_CHIQUAD_UWEIGHTS = (_sinh2, numpy.ones_like, _one_over_sinh2)
_CHIQUAD_VWEIGHTS = (_sin2, numpy.ones_like, _one_over_sin2)


def _staeckel_chi_profiles(
    Ssq, dSsq, args, qmin, D, order, weight_fns, chimax=numpy.pi, vside=False
):
    """The 1/p profile integrals int f/sqrt(S) dq for every f in weight_fns,
    on the same chi mesh as the action: int f/sqrt(S) dq = D int f/sqrt(Q) dchi
    (one evaluation of S serves all of them)"""
    _, sqQ, q, wts = _staeckel_chi_quads(Ssq, dSsq, args, qmin, D, order, chimax, vside)
    return [D * numpy.sum(wts * f(q) / sqQ, axis=-1) for f in weight_fns]


def _staeckel_jacobian(s, umin, umax, vmin, pot, delta, order):
    """The six Leibniz derivatives (djrdE, djrdLz, djrdI3, djzdE, djzdLz,
    djzdI3), shared by the frequencies and the angles"""
    sqrt2 = numpy.sqrt(2.0)
    Lz = s["Lz"]
    prefr = delta / numpy.pi / sqrt2
    prefz = sqrt2 * delta / numpy.pi
    duE, duI3, duLz = _staeckel_chi_profiles(
        _JRStaeckelIntegrandSquared,
        _dJRStaeckelIntegrandSquareddu,
        _staeckel_jr_args(s, pot, delta),
        umin,
        umax - umin,
        order,
        _CHIQUAD_UWEIGHTS,
    )
    dvE, dvI3, dvLz = _staeckel_chi_profiles(
        _JzStaeckelIntegrandSquared,
        _dJzStaeckelIntegrandSquareddv,
        _staeckel_jz_args(s, pot, delta),
        vmin,
        numpy.pi - 2.0 * vmin,  # the full v loop; the midplane is chi = pi/2
        order,
        _CHIQUAD_VWEIGHTS,
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


def _staeckel_freqs(umin, umax, vmin, jac):
    """(Omegar, Omegaphi, Omegaz) from the Leibniz derivatives; NaN for
    circular orbits (the caller substitutes epifreq/omegac/verticalfreq, as
    the C wrapper does) and, for planar eccentric orbits, NaN Omegar and
    Omegaphi with an infinite Omegaz (C's 0/0 and x/0)"""
    djrdE, djrdLz, djrdI3, djzdE, djzdLz, djzdI3 = jac
    detA = djrdE * djzdI3 - djzdE * djrdI3
    circ = (umax - umin) / umax < 1e-6  # circular in R: the J_R panels vanish
    planar = (numpy.pi / 2.0 - vmin) < 1e-7  # planar: the J_z panels vanish
    degen = circ | planar
    detsafe = numpy.where(degen, numpy.ones_like(detA), detA)
    nan = numpy.nan * numpy.ones_like(detA)
    inf = numpy.inf * numpy.ones_like(detA)
    Omegar = numpy.where(degen, nan, djzdI3 / detsafe)
    Omegaz = numpy.where(circ, nan, numpy.where(planar, inf, -djrdI3 / detsafe))
    Omegaphi = numpy.where(degen, nan, (djrdI3 * djzdLz - djzdI3 * djrdLz) / detsafe)
    return Omegar, Omegaphi, Omegaz


def _staeckel_actions_freqs(
    R, vR, vT, z, vz, pot, delta, order, u0=None, torus=False, ddelta=None, du0=None
):
    """(jr, Lz, jz, Omegar, Omegaphi, Omegaz); the frequencies are NaN for
    circular orbits (see _staeckel_freqs); with torus, those of the family
    whose reference curve follows its tori (_staeckel_angles)"""
    s, umin, umax, vmin = _staeckel_prep(R, vR, vT, z, vz, pot, delta, u0=u0)
    jr, jz = _staeckel_jr_jz(s, umin, umax, vmin, pot, delta, order)
    jac = _staeckel_jacobian(s, umin, umax, vmin, pot, delta, order)
    Omegar, Omegaphi, Omegaz = _staeckel_freqs(umin, umax, vmin, jac)
    if torus:
        dOm = _staeckel_angles(
            s, umin, umax, vmin, pot, delta, order, jac, torus, ddelta, du0
        )[3]
        Omegar, Omegaphi, Omegaz = Omegar + dOm[0], Omegaphi + dOm[1], Omegaz + dOm[2]
    return jr, s["Lz"], jz, Omegar, Omegaphi, Omegaz


def _dVdu0(v, u0, delta, pot):
    """The derivative of V(v) = cosh^2 u0 Phi(u0,pi/2) - (sinh^2 u0 + sin^2 v)
    Phi(u0,v) with respect to the reference u0 along which it is built"""
    sh, ch = numpy.sinh(u0), numpy.cosh(u0)
    R, z = coords.uv_to_Rz(u0, v, delta=delta)
    R0 = delta * sh
    return (
        2.0
        * sh
        * ch
        * (_evaluatePotentials(pot, R0, 0.0 * R0) - _evaluatePotentials(pot, R, z))
        - ch**3.0 * delta * _evaluateRforces(pot, R0, 0.0 * R0)
        + (sh**2.0 + numpy.sin(v) ** 2.0)
        * delta
        * (
            _evaluateRforces(pot, R, z) * ch * numpy.sin(v)
            + _evaluatezforces(pot, R, z) * sh * numpy.cos(v)
        )
    )


def _dUddelta(u, delta, pot):
    """The derivative of U(u) = cosh^2 u Phi(u,pi/2) with respect to the
    focal length at fixed u"""
    R = delta * numpy.sinh(u)
    return -(numpy.cosh(u) ** 2.0) * numpy.sinh(u) * _evaluateRforces(pot, R, 0.0 * R)


def _dVddelta(v, u0, delta, pot):
    """The derivative of V(v) = cosh^2 u0 Phi(u0,pi/2) - (sinh^2 u0 + sin^2 v)
    Phi(u0,v) with respect to the focal length at fixed u0 and v"""
    sh, ch = numpy.sinh(u0), numpy.cosh(u0)
    R, z = coords.uv_to_Rz(u0, v, delta=delta)
    R0 = delta * sh
    return -(ch**2.0) * sh * _evaluateRforces(pot, R0, 0.0 * R0) + (
        sh**2.0 + numpy.sin(v) ** 2.0
    ) * (
        _evaluateRforces(pot, R, z) * sh * numpy.sin(v)
        + _evaluatezforces(pot, R, z) * ch * numpy.cos(v)
    )


def _u0torus_partials(s, umin, umax, pot, delta):
    """The partials with respect to (E, I_3, L_z, delta) of the reference
    curve that follows a torus's own turning points, delta sinh(u0) = 2 /
    (1/R_- + 1/R_+) with R = delta sinh(u) at the two turning points of the
    u libration, which move with the integrals and the focal length by the
    level-set rule"""
    args = _staeckel_jr_args(s, pot, delta)
    Lz = s["Lz"]
    Rm, Rp = delta * numpy.sinh(umin), delta * numpy.sinh(umax)
    out = numpy.zeros((4,) + numpy.shape(umin))
    for u, dH in ((umin, 2.0 * Rp**2.0), (umax, 2.0 * Rm**2.0)):
        sh2 = numpy.sinh(u) ** 2.0
        dS = _dJRStaeckelIntegrandSquareddu(u, *args)
        chain = dH / (Rm + Rp) ** 2.0 * numpy.cosh(u) / numpy.cosh(s["u0"]) / dS
        out += chain * numpy.array(
            [
                -sh2,
                numpy.ones_like(sh2),
                Lz / delta**2.0 / sh2,
                _dUddelta(u, delta, pot) - Lz**2.0 / delta**3.0 / sh2,
            ]
        )
    return out


def _staeckel_planar(s, pot, delta):
    """The turning points of the u libration of the planar orbit of each
    point's energy and angular momentum in its model (that of the setup s):
    those at the third integral at which p_v vanishes in the midplane"""
    sp = dict(s)
    sp["I3U"] = (
        s["Lz"] ** 2.0 / 2.0 / delta**2.0
        - s["E"]
        + (s["sinh2u0"] + s["sin2v0u"]) * s["potu0v0"]
    )
    sp["pux"] = numpy.sqrt(
        numpy.maximum(s["pux"] ** 2.0 + 2.0 * delta**2.0 * (s["I3U"] - sp["I3U"]), 0.0)
    )
    return _staeckel_uminumax(sp, pot, delta)[:2]


def _u0planar_partials(s, pot, delta):
    """The partials with respect to (E, I_3, L_z, delta) of the reference
    curve that follows the planar orbit of a torus's energy and angular
    momentum (u0='planar' of actionAngleStaeckelInverse): those of
    _u0torus_partials at the planar orbit's turning points, along its third
    integral L_z^2 / (2 delta^2) - E"""
    g = _u0torus_partials(s, *_staeckel_planar(s, pot, delta), pot, delta)
    Lz = s["Lz"]
    return numpy.array(
        [
            g[0] - g[1],
            numpy.zeros_like(g[1]),
            g[2] + g[1] * Lz / delta**2.0,
            g[3] - g[1] * Lz**2.0 / delta**3.0,
        ]
    )


def _u0shell(s, umin, umax, pot, delta):
    """The u of the shell orbit of each point's energy and angular momentum
    (u0='shell' of actionAngleStaeckelInverse): where p_u^2, which the third
    integral only shifts, is largest, between the point's turning points"""
    args = _staeckel_jr_args(s, pot, delta)
    return _bisect_root(
        lambda u: _dJRStaeckelIntegrandSquareddu(u, *args),
        umin,
        umax,
        xtol=1e-14,
        maxiter=200,
    )


def _staeckel_fraction(s, ush, pot, delta, partials=False):
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
    ok = tot > 0.0
    tot = numpy.where(ok, tot, 1.0)
    f = numpy.clip(numpy.where(ok, below / tot, 0.5), 0.0, 1.0)
    if not partials:
        return f
    coth2 = 1.0 / numpy.tanh(ush) ** 2.0
    dbelow = (1.0, 1.0, -Lz / delta**2.0, Lz**2.0 / delta**3.0)
    dtot = (
        numpy.cosh(ush) ** 2.0,
        0.0,
        -Lz / delta**2.0 * coth2,
        Lz**2.0 / delta**3.0 * coth2 - _dUddelta(ush, delta, pot),
    )
    return f, numpy.array(
        [numpy.where(ok, (b - f * t) / tot, 0.0) for b, t in zip(dbelow, dtot)]
    )


def _u0shell_partials(s, ush, pot, delta, R2deriv):
    """The partials with respect to (E, I_3, L_z, delta) of the shell u, by
    the level-set rule on d S_u / du = 0, S_u = E sinh^2 u - U(u) - L_z^2 /
    (2 delta^2 sinh^2 u) with U(u) = cosh^2 u Phi(delta sinh u, 0); R2deriv
    is the potential's second radial derivative in the midplane"""
    E, Lz = s["E"], s["Lz"]
    sh, ch = numpy.sinh(ush), numpy.cosh(ush)
    R = delta * sh
    Phi = _evaluatePotentials(pot, R, 0.0 * R)
    PhiR = -_evaluateRforces(pot, R, 0.0 * R)
    Suu = (
        2.0 * numpy.cosh(2.0 * ush) * (E - Phi)
        - 5.0 * delta * sh * ch**2.0 * PhiR
        - delta**2.0 * ch**4.0 * R2deriv
        + Lz**2.0 / delta**2.0 * (sh**2.0 - 3.0 * ch**2.0) / sh**4.0
    )
    SuD = (
        -(2.0 * sh**2.0 * ch + ch**3.0) * PhiR
        - delta * ch**3.0 * sh * R2deriv
        - 2.0 * Lz**2.0 / delta**3.0 * ch / sh**3.0
    )
    return numpy.array(
        [
            -numpy.sinh(2.0 * ush) / Suu,
            numpy.zeros_like(Suu),
            -2.0 * Lz / delta**2.0 * ch / sh**3.0 / Suu,
            -SuD / Suu,
        ]
    )


def _u0callable(rule, nargs, E, Lz, I3, partials=False):
    """The reference curve of a callable rule u0(E, |L_z|[, I_3]) of
    actionAngleStaeckelInverse for each torus and, with partials, its
    partials with respect to (E, I_3, L_z, delta), by central differences of
    the callable (FINITE DIFFERENCES, of the user's callable alone)"""

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


def _staeckel_angles(
    s, umin, umax, vmin, pot, delta, order, jac, torus=False, ddelta=None, du0=None
):
    """(angler, anglephi without the azimuth, anglez); angler and anglez in
    [0, 2pi), all three zero for circular or planar orbits. With torus, for
    a reference curve that follows the tori (torus 2: u0='torus' of
    actionAngleStaeckelInverse; 3: its u0='planar'): the angles of that family and the
    difference of its frequencies from the model's, in the order
    (R, phi, z). The family's generating function is the model's at the
    reference curve u0(J) of each torus, so that its angles are the model's
    plus K du0/dJ, with K = dS/du0 at fixed position and actions, and its
    frequencies dE/dJ are the model's plus (dE/du0 at fixed actions)
    du0/dJ. Everything but the integral of dV/du0 over p_v, which enters
    as the integral of the third integral's weight does, is what the
    model's own angles are made of. With ddelta, the derivatives of each
    torus's focal length with respect to (J_R, L_z, J_z), also for a focal
    length that follows the tori: the same with K_delta = dS/ddelta at
    fixed position and actions, which is the dilation of the point
    transformation, -(R p_R + z p_z)/delta, plus the integrals of the
    derivatives of p_u and p_v with respect to delta along the orbit less
    the angles times their complete versions.

    The angles need partial Leibniz integrals, from a turning point to the
    current u/v. The u partial from the low turning point has upper anomaly
    chi(ux) = 2 arcsin(sqrt((ux-umin)/Du)), that from the high one is the
    complete integral minus it; the v anomaly spans the full loop
    [vmin, pi-vmin], and chi(pi-v) = pi - chi(v), so the v partials are one
    partial and the integral to the midplane. A reflection constant K and a
    sign, functions of the momentum's sign and the position's quadrant, map
    each partial onto the full angle (C's calcAnglesStaeckel)."""
    sqrt2 = numpy.sqrt(2.0)
    pi = numpy.pi
    Lz = s["Lz"]
    ux, vx, pux, pvx = s["ux"], s["vx"], s["pux"], s["pvx"]
    djrdE, djrdLz, djrdI3, djzdE, djzdLz, djzdI3 = jac
    detA = djrdE * djzdI3 - djzdE * djrdI3
    circ = (umax - umin) / umax < 1e-6
    planar = (pi / 2.0 - vmin) < 1e-7
    detsafe = numpy.where(circ | planar, numpy.ones_like(detA), detA)
    Omegar = djzdI3 / detsafe
    Omegaz = -djrdI3 / detsafe
    Omegaphi = (djrdI3 * djzdLz - djzdI3 * djrdLz) / detsafe
    dI3dJR = -djzdE / detsafe
    dI3dJz = djrdE / detsafe
    dI3dLz = -(djrdE * djzdLz - djzdE * djrdLz) / detsafe
    jr_args = _staeckel_jr_args(s, pot, delta)
    jz_args = _staeckel_jz_args(s, pot, delta)
    JRsq, JZsq = _JRStaeckelIntegrandSquared, _JzStaeckelIntegrandSquared
    dJRsq, dJZsq = _dJRStaeckelIntegrandSquareddu, _dJzStaeckelIntegrandSquareddv
    # u
    Du = umax - umin
    Du_safe = numpy.where(Du > 0.0, Du, numpy.ones_like(Du))
    chi_ux = 2.0 * numpy.arcsin(numpy.sqrt(numpy.clip((ux - umin) / Du_safe, 0.0, 1.0)))
    high_u = ux > umin + 0.5 * Du
    uweights = _CHIQUAD_UWEIGHTS
    if ddelta is not None:
        # sqrt(2) d p/d delta at fixed (E, I_3), over the 1/sqrt(S) of the
        # profile integrals: 2 S + L_z^2 / (delta^2 sinh^2 u) - delta dU/ddelta
        # and 2 S + L_z^2 / (delta^2 sin^2 v) + delta dV/ddelta
        col = lambda a: a[..., None] if getattr(a, "ndim", 0) >= 1 else a
        dcol, Lcol = col(delta), col(Lz)
        ucols = tuple(col(a) for a in jr_args)
        vcols = tuple(col(a) for a in jz_args)
        uweights += (
            lambda q: (
                2.0 * _staeckel_eval_flat(JRsq, q, ucols)
                + Lcol**2.0 / dcol**2.0 / numpy.sinh(q) ** 2.0
                - dcol * _staeckel_eval_flat(_dUddelta, q, (dcol, pot))
            ),
        )
    part_u = _staeckel_chi_profiles(
        JRsq, dJRsq, jr_args, umin, Du, order, uweights,
        chimax=chi_ux[..., None],
    )  # fmt: skip
    full_u = _staeckel_chi_profiles(JRsq, dJRsq, jr_args, umin, Du, order, uweights)
    PE, PI, PL, *PW = (
        numpy.where(high_u, fu - pu, pu) for pu, fu in zip(part_u, full_u)
    )
    pos_u = pux > 0.0
    K_u = numpy.where(high_u, pi, numpy.where(pos_u, 0.0, 2.0 * pi)) * numpy.ones_like(
        ux
    )
    s_u = numpy.where(
        high_u, numpy.where(pos_u, -1.0, 1.0), numpy.where(pos_u, 1.0, -1.0)
    )
    Or1 = K_u * djrdE + s_u * (delta / sqrt2) * PE
    I3r1 = K_u * djrdI3 - s_u * (delta / sqrt2) * PI  # leading minus in u
    aphi_u = K_u * djrdLz - s_u * (Lz / delta / sqrt2) * PL
    # v
    Dv = numpy.pi - 2.0 * vmin
    Dv_safe = numpy.where(Dv > 0.0, Dv, numpy.ones_like(Dv))
    chi_vx = 2.0 * numpy.arcsin(numpy.sqrt(numpy.clip((vx - vmin) / Dv_safe, 0.0, 1.0)))
    mid_v_pt = vmin + 0.5 * (pi / 2.0 - vmin)
    low_v = (vx < mid_v_pt) | (vx > (pi - mid_v_pt))
    above = vx > pi / 2.0
    # low_v panels integrate from the vmin turning point; by the pi/2 symmetry
    # the ones above the midplane are the mirrored partial, anomaly pi - chi(vx)
    chimax_v = numpy.where(low_v & above, numpy.pi - chi_vx, chi_vx)
    vweights = _CHIQUAD_VWEIGHTS
    if torus:
        cols = tuple(a[..., None] if numpy.ndim(a) else a for a in (s["u0"], delta))
        vweights += (lambda q: _staeckel_eval_flat(_dVdu0, q, cols + (pot,)),)
    if ddelta is not None:
        vweights += (
            lambda q: (
                2.0 * _staeckel_eval_flat(JZsq, q, vcols)
                + Lcol**2.0 / dcol**2.0 / numpy.sin(q) ** 2.0
                + dcol * _staeckel_eval_flat(_dVddelta, q, cols + (pot,))
            ),
        )
    part_v = _staeckel_chi_profiles(
        JZsq, dJZsq, jz_args, vmin, Dv, order, vweights,
        chimax=chimax_v[..., None], vside=True,
    )  # fmt: skip
    half_v = _staeckel_chi_profiles(
        JZsq, dJZsq, jz_args, vmin, Dv, order, vweights,
        chimax=numpy.pi / 2.0, vside=True,
    )  # fmt: skip
    QE, QI, QL, *QV = (
        numpy.where(low_v, pv, numpy.where(above, pv - hv, hv - pv))
        for pv, hv in zip(part_v, half_v)
    )
    pos_v = pvx > 0.0
    K_v = numpy.where(
        low_v,
        numpy.where(
            pos_v, numpy.where(above, pi, 0.0), numpy.where(above, pi, 2.0 * pi)
        ),
        numpy.where(pos_v, pi / 2.0, 1.5 * pi),
    ) * numpy.ones_like(vx)
    s_v = numpy.where(
        low_v,
        numpy.where(
            pos_v, numpy.where(above, -1.0, 1.0), numpy.where(above, 1.0, -1.0)
        ),
        numpy.where(
            pos_v, numpy.where(above, 1.0, -1.0), numpy.where(above, -1.0, 1.0)
        ),
    )
    Or2 = K_v * djzdE + s_v * (delta / sqrt2) * QE
    I3r2 = K_v * djzdI3 + s_v * (delta / sqrt2) * QI  # no leading minus in v
    phitmp = K_v * djzdLz - s_v * (Lz / delta / sqrt2) * QL
    # assembly (C's calcAnglesStaeckel)
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
            rE, rI, rL, rD = _u0planar_partials(s, pot, delta)
        else:
            rE, rI, rL, rD = _u0torus_partials(s, umin, umax, pot, delta)
        dEdu0 = -Omegaz * djzdu0
        norm = 1.0 - rE * dEdu0 + rI * dI3dJz * djzdu0
        du0 = numpy.array(
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
            sinhux, sinvx = numpy.sinh(ux), numpy.sin(vx)
            KD = (
                -(sinhux * numpy.cosh(ux) * pux - sinvx * numpy.cos(vx) * pvx)
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
                    * numpy.array(
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
    angler = numpy.remainder(angler, 2.0 * pi)
    anglez = numpy.remainder(anglez, 2.0 * pi)
    zeros = numpy.zeros_like(angler)
    degen = circ | planar  # both degeneracies: C's calcAngles returns 0
    angler = numpy.where(degen, zeros, angler)
    anglez = numpy.where(degen, zeros, anglez)
    anglephi = numpy.where(degen, zeros, anglephi)
    if torus:
        return angler, anglephi, anglez, dOm
    return angler, anglephi, anglez


def _staeckel_actions_freqs_angles(
    R,
    vR,
    vT,
    z,
    vz,
    phi,
    pot,
    delta,
    order,
    u0=None,
    torus=False,
    ddelta=None,
    du0=None,
):
    """(jr, Lz, jz, Omegar, Omegaphi, Omegaz, angler, anglephi, anglez), the
    azimuth folded into anglephi"""
    s, umin, umax, vmin = _staeckel_prep(R, vR, vT, z, vz, pot, delta, u0=u0)
    jr, jz = _staeckel_jr_jz(s, umin, umax, vmin, pot, delta, order)
    jac = _staeckel_jacobian(s, umin, umax, vmin, pot, delta, order)
    Omegar, Omegaphi, Omegaz = _staeckel_freqs(umin, umax, vmin, jac)
    angler, anglephi, anglez, *dOm = _staeckel_angles(
        s, umin, umax, vmin, pot, delta, order, jac, torus, ddelta, du0
    )
    if torus:
        Omegar, Omegaphi, Omegaz = (
            Omegar + dOm[0][0],
            Omegaphi + dOm[0][1],
            Omegaz + dOm[0][2],
        )
    anglephi = numpy.remainder(anglephi + phi, 2.0 * numpy.pi)
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
            The focal length of the prolate spheroidal coordinate system; may
            be left out for a potential that supplies its own (an exact
            Staeckel potential such as KuzminKutuzovStaeckelPotential, or an
            OblateStaeckelWrapperPotential), whose focal length is then used.
        inverse : actionAngleStaeckelInverse, optional
            Set up the transformation that inverts this instance, in place of pot= and delta=: (x,v) -> (J,theta) such that inverse(J,theta) returns (x,v). Each phase-space point is transformed in the Staeckel model of the torus of the instance that passes through it: the instance's single model or, for a reference curve that follows the tori (u0='torus'), the model whose reference curve is that of the point's own torus and, for a focal length that follows the tori (delta='fit' or a callable), whose focal length is the family's at that torus's actions. For a family (setup_interp=True), these are the exact actions of the family's tori, to which the family's interpolated map agrees to its interpolation error.
        polish : bool, optional
            With inverse= a family (setup_interp=True): make the transformation the exact inverse of the family's interpolated map, by Newton's method on that map from the actions and angles of the family's tori; points that no torus of the family passes through return nan with a warning. Default is False: the actions and angles of the family's tori, to which the interpolated map agrees to its interpolation error.
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
        self._inverse = kwargs.get("inverse", None)
        # whether the energy is that of the Staeckel model of (delta, u0)
        self._model = False
        self._polish = kwargs.get("polish", False)
        if self._polish and not (self._inverse is not None and self._inverse._interp):
            raise ValueError(
                "polish= requires inverse= to be a family (setup_interp=True)"
            )
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

    def _setup_inverse(self, kwargs):
        """The set-up that inverts the actionAngleStaeckelInverse instance
        inverse=: its Staeckel model or, for tori that each have their own
        model, the potential that those models approximate, with the
        energy of every point evaluated in its own model"""
        aAI = self._inverse
        if "pot" in kwargs or "delta" in kwargs:
            raise TypeError(
                "pot= and delta= cannot be given with inverse=, which supplies them"
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
            kwargs["delta"] = float(numpy.median(aAI._A_tab[..., 3]))
        if aAI._roSet:
            kwargs.setdefault("ro", aAI._ro)
        if aAI._voSet:
            kwargs.setdefault("vo", aAI._vo)
        return kwargs

    def _polished(self, *args, **kwargs):
        """(J_R, L_z, J_z, Omega_R, Omega_phi, Omega_z, theta_R, theta_phi,
        theta_z) such that the family inverse= maps the actions and angles
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
        # a point without one of the two librations has no angle of it (nor,
        # in C, of the other): the starting angles of a point slightly off it
        flat = numpy.all(numpy.isfinite(out[[0, 2]]), axis=0) & ~numpy.all(
            numpy.isfinite(out[[6, 8]]), axis=0
        )
        if numpy.any(flat):
            kick = 1e-3 * numpy.sqrt(vR**2 + vT**2 + vz**2)[flat]
            out[6:, flat] = self._actionsFreqsAngles(
                R[flat],
                vR[flat] + kick,
                vT[flat],
                z[flat],
                vz[flat] + kick,
                phi[flat],
                _direct=True,
                **kwargs,
            )[6:]
        ok = numpy.all(numpy.isfinite(out[[0, 2, 6, 8]]), axis=0)
        sol = numpy.full((5, len(R)), numpy.nan)
        if numpy.any(ok):
            sol[:, ok] = aAI._invert(
                R[ok], vR[ok], vT[ok], z[ok], vz[ok], phi[ok], *out[[0, 2, 6, 8]][:, ok]
            )
        out[[0, 2, 6, 7, 8]] = sol
        bad = numpy.isnan(sol[0])
        if numpy.any(bad):
            warnings.warn(
                f"{numpy.sum(bad)} points are not on a torus of inverse=: their "
                "actions, frequencies and angles are nan",
                galpyWarning,
            )
        good = ~bad
        out[3:6, bad] = numpy.nan
        if numpy.any(good):
            out[3:6, good] = aAI._Freqs(*out[:3][:, good])
        return tuple(out)

    def _inverse_model(self, R, vR, vT, z, vz, c=True, deriv=True):
        """The focal length and reference curve of the Staeckel model of
        the torus of inverse= through each phase-space point. A reference
        curve that follows the tori depends on the torus's energy and third
        integral in its own model, which depend on the reference curve: a
        fixed point, found by iterating the rule on the turning points in
        the current model with Aitken's acceleration: by the C code itself
        (which is then handed nan to start from the point's own u) or, with
        c False, here"""
        aAI = self._inverse
        delta = aAI._delta
        if not self._model:
            return delta, numpy.full(numpy.shape(R), aAI._staeckelwrap._u0), None
        R, vR, vT, z, vz = (
            a.copy()
            for a in numpy.broadcast_arrays(
                *(numpy.atleast_1d(a) * 1.0 for a in (R, vR, vT, z, vz))
            )
        )
        pot = self._pot if c else _staeckel_pot(self._pot)

        def rule(idx, u0, delta=delta):
            x = (R[idx], vR[idx], vT[idx], z[idx], vz[idx])
            if self._model >= 4:
                return self._inverse_rule(x, delta, u0)
            if self._model == 3:
                um, up = _staeckel_planar(
                    _staeckel_setup(*x, pot, delta, u0=u0), pot, delta
                )
            else:
                um, up = _staeckel_prep(*x, pot, delta, u0=u0)[1:3]
            sm, sp = numpy.sinh(um), numpy.sinh(up)
            return numpy.arcsinh(2.0 * sm * sp / (sm + sp))

        idx = numpy.arange(len(R))
        if aAI._delta_rule is not None:
            return self._inverse_model_delta(rule, R, vR, vT, z, vz, pot, c, deriv)
        if c and self._model != 4:
            return delta, numpy.full(len(R), numpy.nan), None
        u0 = rule(idx, numpy.arcsinh(R / delta))
        for _ in range(_MODEL_MAXITER):
            u1 = rule(idx, u0[idx])
            u2 = rule(idx, u1)
            den = u2 - 2.0 * u1 + u0[idx]
            ok = den != 0.0
            acc = u2 - (u2 - u1) ** 2 / numpy.where(ok, den, 1.0)
            left = _fixed_point_error(u0[idx], u1, u2)
            u0[idx] = numpy.where(ok & numpy.isfinite(acc), acc, u2)
            idx = idx[left > _MODEL_TOL]
            if len(idx) == 0:
                break
        return delta, u0, None

    def _inverse_rule(self, x, delta, u0, partials=False):
        """One application, in the models (delta, u0) of the points x = (R,
        vR, vT, z, vz), of a rule for the reference curve that is evaluated
        here: u0='shell' of inverse=, the shell orbit of each point's energy
        and angular momentum, or its callable u0(E, |L_z|[, I_3]). With
        partials, the rule's partials with respect to (E, I_3, L_z, delta)
        instead"""
        aAI = self._inverse
        pot = _staeckel_pot(self._pot)
        s = _staeckel_setup(*x, pot, delta, u0=u0)
        if callable(aAI._u0_rule):
            return _u0callable(
                aAI._u0_rule, aAI._u0_nargs, s["E"], s["Lz"], s["I3V"], partials
            )
        ush = _u0shell(s, *_staeckel_uminumax(s, pot, delta)[:2], pot, delta)
        if not partials:
            return ush
        Rsh = delta * numpy.sinh(ush)
        R2 = numpy.array(
            [evaluateR2derivs(self._pot, r, 0.0, use_physical=False) for r in Rsh]
        )
        return _u0shell_partials(s, ush, pot, delta, R2)

    def _inverse_du0(self, R, vR, vT, z, vz, delta, u0, c=False):
        """The partials of the reference curve of each point's torus for a
        rule that is evaluated here (_inverse_rule: a callable, and
        u0='shell' unless for the C code, which has that rule), else None"""
        if self._model < 4 or (c and self._model == 5):
            return None
        x = numpy.broadcast_arrays(
            *(numpy.atleast_1d(a) * 1.0 for a in (R, vR, vT, z, vz))
        )
        return self._inverse_rule(x, delta, u0, partials=True)

    def _inverse_model_delta(self, rule, R, vR, vT, z, vz, pot, c, deriv):
        """_inverse_model for a family whose focal length follows its tori:
        the focal length of a torus is the family's rule at the torus's
        energy, angular momentum, and fraction f of the way of its third
        integral from the planar to the shell orbit's, all in its own
        model: the fixed point in the focal length and the reference curve
        together, by the same iteration. For a family that does not have
        its rule (a pickled family with a callable delta=), the focal
        length is read from the family's table at the torus's actions
        instead. Also returns, for the frequencies and angles, the partials
        of the focal length's rule with respect to (E, I_3, L_z, delta) at
        each point's torus (the rule's own with respect to the energy,
        the angular momentum, and f, and those of f in the model), or, for
        a family without its rule, the derivatives of the focal length
        with respect to (J_R, L_z, J_z) from the family's table (nan for
        points whose torus the family does not hold)"""
        aAI = self._inverse
        Lz = R * vT
        pypot = _staeckel_pot(self._pot)
        edges = aAI._delta_edges(Lz)
        direct = aAI._delta_of(R[:0], R[:0], R[:0], edges[:0]) is not None

        def step(idx, u0, delta, partials=False):
            x = (R[idx], vR[idx], vT[idx], z[idx], vz[idx])
            if direct:
                if turn is not None:
                    # (at the fixed point that the C code found, with its
                    # turning points)
                    um, up = turn[0][idx], turn[1][idx]
                    s = _staeckel_setup(*x, pypot, delta, u0=u0)
                elif c and self._model != 4:
                    # (the C code moves u0 to the fixed point at this delta)
                    u0 = numpy.array(u0, dtype="float64", order="C")
                    um, up = actionAngleStaeckel_c.actionAngleUminUmaxVminStaeckel_c(
                        pot, delta, *x, u0=u0, model=self._model
                    )[:2]
                    s = _staeckel_setup(*x, pypot, delta, u0=u0)
                else:
                    u0 = rule(idx, u0, delta)
                    s = _staeckel_setup(*x, pypot, delta, u0=u0)
                    um, up = _staeckel_uminumax(s, pypot, delta)[:2]
                # the third integral above the planar orbit's (in the gauge
                # V(pi/2) = 0) and below the shell orbit's (p_u^2 at the
                # shell orbit's u); a circular orbit has no fraction
                ush = _u0shell(s, um, up, pypot, delta)
                if partials:
                    f, df = _staeckel_fraction(s, ush, pypot, delta, True)
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
                f = _staeckel_fraction(s, ush, pypot, delta)
                return u0, aAI._delta_of(s["E"], Lz[idx], f, edges[idx]), None
            if c:
                # (the C code moves u0 to the fixed point at this delta,
                # unless the rule is evaluated here)
                u0 = numpy.array(
                    rule(idx, u0, delta) if self._model == 4 else u0,
                    dtype="float64",
                    order="C",
                )
                jr, jz = actionAngleStaeckel_c.actionAngleStaeckel_c(
                    pot, delta, *x, u0=u0, order=self._order, model=self._model
                )[:2]
            else:
                u0 = rule(idx, u0, delta)
                jr, _, jz = _staeckel_actions(*x, pot, delta, self._order, u0=u0)
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
        # (a point without a focal length has no model at all)
        u0 = numpy.where(numpy.isfinite(delta), u0, numpy.nan)
        if direct:
            ddelta = numpy.full((4, len(R)), numpy.nan)
            if deriv and len(found) > 0:
                ddelta[:, found] = step(found, u0[found], delta[found], True)
            return delta, u0, ddelta
        # (the family's table is in (J_R, J_z, J_phi))
        return delta, u0, ddelta[:, [0, 2, 1]].T

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
                # (a point without a model has no actions)
                jr, jz = jr + 0.0 * delta, jz + 0.0 * delta
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
            u0 = ddelta = None
            if self._inverse is not None:
                delta, u0, ddelta = self._inverse_model(
                    R, vR, vT, z, vz, c=loop_c, deriv=False
                )
            jr, Lz, jz = _staeckel_actions(
                R,
                vR,
                vT,
                z,
                vz,
                _staeckel_pot(self._pot),
                _coerce_delta_arraylike(delta),
                order,
                u0=u0,
            )
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
                # (a point without a model has no actions)
                jr, jz = jr + 0.0 * delta, jz + 0.0 * delta
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
            # the u0 reference of useu0 does not change the actions and
            # frequencies, so it is not needed here
            u0 = ddelta = None
            if self._inverse is not None:
                delta, u0, ddelta = self._inverse_model(R, vR, vT, z, vz, c=loop_c)
            jr, Lz, jz, Omegar, Omegaphi, Omegaz = _staeckel_actions_freqs(
                R,
                vR,
                vT,
                z,
                vz,
                _staeckel_pot(self._pot),
                _coerce_delta_arraylike(delta),
                order,
                u0=u0,
                torus=self._model,
                ddelta=ddelta,
                du0=self._inverse_du0(R, vR, vT, z, vz, delta, u0),
            )
            # Close-to-circular orbits: the frequencies are NaN (det(A)=0);
            # substitute epifreq/omegac/verticalfreq (as the C wrapper does)
            indx = (numpy.isnan(Omegar) & (jr < 10.0**-3.0)) | (
                numpy.isnan(Omegaz) & (jz < 10.0**-3.0)
            )
            if numpy.any(indx):
                Omegar[indx] = [
                    epifreq(self._pot, r, use_physical=False) for r in R[indx]
                ]
                Omegaphi[indx] = [
                    omegac(self._pot, r, use_physical=False) for r in R[indx]
                ]
                Omegaz[indx] = [
                    verticalfreq(self._pot, r, use_physical=False) for r in R[indx]
                ]
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
        loop_c = bool(self._c and kwargs.get("c", True))
        ddelta = None
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
                # (a point without a model has no actions)
                jr, jz = jr + 0.0 * delta, jz + 0.0 * delta
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
            # the u0 reference of useu0 does not change the actions,
            # frequencies, and angles, so it is not needed here
            u0 = ddelta = None
            if self._inverse is not None:
                delta, u0, ddelta = self._inverse_model(R, vR, vT, z, vz, c=loop_c)
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
                R,
                vR,
                vT,
                z,
                vz,
                phi,
                _staeckel_pot(self._pot),
                _coerce_delta_arraylike(delta),
                order,
                u0=u0,
                torus=self._model,
                ddelta=ddelta,
                du0=self._inverse_du0(R, vR, vT, z, vz, delta, u0),
            )
            # Close-to-circular orbits: the frequencies are NaN (det(A)=0);
            # substitute epifreq/omegac/verticalfreq (as the C wrapper does; the
            # angles are already zero there)
            indx = (numpy.isnan(Omegar) & (jr < 10.0**-3.0)) | (
                numpy.isnan(Omegaz) & (jz < 10.0**-3.0)
            )
            if numpy.any(indx):
                Omegar[indx] = [
                    epifreq(self._pot, r, use_physical=False) for r in R[indx]
                ]
                Omegaphi[indx] = [
                    omegac(self._pot, r, use_physical=False) for r in R[indx]
                ]
                Omegaz[indx] = [
                    verticalfreq(self._pot, r, use_physical=False) for r in R[indx]
                ]
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
        delta = kwargs.get("delta", self._delta)
        umin, umax, vmin = self._uminumaxvmin(*args, **kwargs)
        rperi = coords.uv_to_Rz(umin, numpy.pi / 2.0, delta=delta)[0]
        rap_tmp, zmax = coords.uv_to_Rz(umax, vmin, delta=delta)
        rap = numpy.sqrt(rap_tmp**2.0 + zmax**2.0)
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
        loop_c = bool(self._c and kwargs.get("c", True))
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
            kwargs.pop("c", None)
            u0 = ddelta = None
            if self._inverse is not None:
                delta, u0, ddelta = self._inverse_model(
                    R, vR, vT, z, vz, c=loop_c, deriv=False
                )
            _, umin, umax, vmin = _staeckel_prep(
                R, vR, vT, z, vz, _staeckel_pot(self._pot), delta, u0=u0
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
    sinh2u = numpy.sinh(u) ** 2.0
    dU = (sinh2u + sin2v0) * potentialStaeckel(u, v0, pot, delta) - (
        sinh2u0 + sin2v0
    ) * potu0v0
    return E * sinh2u - I3U - dU - Lz**2.0 / 2.0 / delta**2.0 / sinh2u


def _JzStaeckelIntegrandSquared(
    v, E, Lz, I3V, delta, u0, cosh2u0, sinh2u0, potu0pi2, pot
):
    # potu0pi2= potentialStaeckel(u0,numpy.pi/2.,pot,delta)
    """The J_z integrand: p_v(v)/2/delta^2"""
    sin2v = numpy.sin(v) ** 2.0
    dV = cosh2u0 * potu0pi2 - (sinh2u0 + sin2v) * potentialStaeckel(u0, v, pot, delta)
    return E * sin2v + I3V + dV - Lz**2.0 / 2.0 / delta**2.0 / sin2v


# Derivatives of the under-radical functions S_R/S_z with respect to the
# integration coordinate (analytic, via the forces); these supply the finite
# turning-point limits of the chi-anomaly quadratures below
def _dJRStaeckelIntegrandSquareddu(
    u, E, Lz, I3U, delta, u0, sinh2u0, v0, sin2v0, potu0v0, pot
):
    R, z = coords.uv_to_Rz(u, v0, delta=delta)
    dPhidu = -delta * (
        _evaluateRforces(pot, R, z) * numpy.cosh(u) * numpy.sin(v0)
        + _evaluatezforces(pot, R, z) * numpy.sinh(u) * numpy.cos(v0)
    )
    return (
        E * numpy.sinh(2.0 * u)
        - numpy.sinh(2.0 * u) * potentialStaeckel(u, v0, pot, delta)
        - (numpy.sinh(u) ** 2.0 + sin2v0) * dPhidu
        + Lz**2.0 / delta**2.0 * numpy.cosh(u) / numpy.sinh(u) ** 3.0
    )


def _dJzStaeckelIntegrandSquareddv(
    v, E, Lz, I3V, delta, u0, cosh2u0, sinh2u0, potu0pi2, pot
):
    R, z = coords.uv_to_Rz(u0, v, delta=delta)
    dPhidv = -delta * (
        _evaluateRforces(pot, R, z) * numpy.sinh(u0) * numpy.cos(v)
        - _evaluatezforces(pot, R, z) * numpy.cosh(u0) * numpy.sin(v)
    )
    return (
        E * numpy.sin(2.0 * v)
        - numpy.sin(2.0 * v) * potentialStaeckel(u0, v, pot, delta)
        - (sinh2u0 + numpy.sin(v) ** 2.0) * dPhidv
        + Lz**2.0 / delta**2.0 * numpy.cos(v) / numpy.sin(v) ** 3.0
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
# already reproduce C's jr to ~1e-11 (6: 1e-10, 4: 2e-7) on a near-axis grid
_STAECKEL_NEARAXIS = 0.2
_STAECKEL_NEARAXIS_NPANELS = 8


def _staeckel_nchi(order, vside):
    """Panels of the composite 10-node chi rule. C parity: C integrates each half
    of the u anomaly with one `order`-point GL rule, so nchi = 2*order/10 gives
    C's nodes at the default order=10 and its node count above it. C's v side
    ends in a t^2 panel based at the midplane, which resolves the disk structure
    there; a uniform chi mesh needs twice the panels to match it (MWPotential2014
    disk orbits: jz 1e-8 -> 2e-10, the turning-point floor)."""
    n = max(2, -(-int(order) // 5))
    return 2 * n if vside else n


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
