###############################################################################
# actionAngleStaeckelInverse.py: inverse action-angle transformation for
#   axisymmetric Staeckel potentials through a momentum-matched canonical
#   map: each torus is the product of its two librations, in the prolate
#   spheroidal coordinates u and v, sampled exactly in their anomalies and
#   mapped onto the torus of equal (J_R, L = J_z + |L_z|, L_z) of a frozen
#   isochrone auxiliary by two point transformations, one per libration --
#   u onto the auxiliary's radius, v onto its polar angle -- each the one
#   that matches cumulative actions (a sine series in the anomaly) and each
#   cotangent-lifted, with the compensation of the three angles for the
#   moving turning points of both librations in closed form. The assembled
#   (J, theta) -> (x, v) map is exactly symplectic for ANY stored tables:
#   every derivative is the stored interpolant's own. (The Staeckel section
#   of canonical.tex in the fast-orbits repository.)
###############################################################################
import inspect
import warnings
from contextlib import contextmanager

import numpy
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from ..potential import (
    IsochronePotential,
    OblateStaeckelWrapperPotential,
    epifreq,
    evaluatePotentials,
    evaluateR2derivs,
    evaluateRforces,
    evaluatez2derivs,
    evaluatezforces,
    rl,
    vcirc,
    verticalfreq,
)
from ..potential.Potential import _check_potential_list_and_deprecate
from ..util import conversion, coords, galpyWarning
from ..util._hermite import HermiteFamily3D, slope_at_zero
from ..util._optional_deps import _TQDM_LOADED
from ..util.conversion import actionAngleInverse_physical_input
from .actionAngleInverse import actionAngleInverse
from .actionAngleInverseUtil import (
    _BOUND_MARGIN,
    _chunked,
    _fit_isochrone_vc2,
    _harmonics,
    _harmonics_kept,
    _harmonics_needed,
    _offset_spec_coeffs,
    _progress,
    _sine_galerkin,
    _sine_project,
    _sine_series,
    _spaced_grid,
    _spec_eval,
)
from .actionAngleIsochroneInverse import actionAngleIsochroneInverse
from .actionAngleStaeckel import _focal_length_of

# The points of an array evaluation on a family are grouped by the numbers of
# harmonics of their cells, and every group is one vectorized pass with a
# fixed cost (that of some tens of points): groups of fewer points than this
# are evaluated together, with the largest numbers among them
_EVAL_GROUP_MIN = 128


def _cumulative(f):
    """The integral from tau = 0 of a periodic function sampled on the
    regular offset grid: its mean (the secular rate), the periodic part on
    the grid, that part's spectral coefficients (trimmed to the harmonics
    that carry it, the rest being round-off, so that evaluating the series
    elsewhere costs no more than it must), and its value at zero"""
    N = len(f)
    k = numpy.fft.fftfreq(N, d=1.0 / N)
    m = numpy.mean(f)
    fh = numpy.fft.fft(f - m)
    ah = numpy.zeros_like(fh)
    ah[1:] = fh[1:] / (1j * k[1:])
    q = numpy.real(numpy.fft.ifft(ah))
    cq = _offset_spec_coeffs(q)
    keep = numpy.flatnonzero(numpy.abs(cq) > 1e-16 * numpy.max(numpy.abs(cq)))
    # one vanishing coefficient beyond the last kept one, because the series
    # gives the last coefficient the Nyquist weight
    cq = numpy.append(cq[: keep[-1] + 1], 0.0) if len(keep) > 0 else cq
    return m, q, cq, _spec_eval(cq, 0.0)[0]


def _solve_increasing(f, fp, lo, hi):
    """The root of an increasing function f with derivative fp, for arrays
    of brackets [lo, hi] with f(lo) <= 0 <= f(hi): bisection until the
    brackets are narrow, then Newton steps kept inside the brackets, to
    round-off"""
    for _ in range(8):
        mid = 0.5 * (lo + hi)
        neg = f(mid) < 0.0
        lo, hi = numpy.where(neg, mid, lo), numpy.where(neg, hi, mid)
    x = 0.5 * (lo + hi)
    for _ in range(12):
        fx = f(x)
        neg = fx < 0.0
        lo, hi = numpy.where(neg, x, lo), numpy.where(neg, hi, x)
        xn = x - fx / fp(x)
        # a step out of the bracket is replaced by its midpoint; a converged
        # point sits on the bracket's edge and must not be thrown back
        xn = numpy.where((xn < lo) | (xn > hi), 0.5 * (lo + hi), xn)
        done = numpy.all(numpy.fabs(xn - x) <= 3e-15 * numpy.fabs(x))
        x = xn
        if done:
            break
    return x


def _hermite_table(x, y, yp):
    """A function tabulated with its slope at the uniform nodes x, as the
    coefficients of its cubic Hermite interpolant on every interval: (first
    node, spacing, coefficients (4, intervals) in the fraction of the
    interval)"""
    h = (x[-1] - x[0]) / (len(x) - 1)
    y0, y1, d0, d1 = y[:-1], y[1:], h * yp[:-1], h * yp[1:]
    return (
        x[0],
        h,
        numpy.array(
            [y0, d0, 3.0 * (y1 - y0) - 2.0 * d0 - d1, 2.0 * (y0 - y1) + d0 + d1]
        ),
    )


def _hermite_uniform(tab, x, deriv=False):
    """The value at x of the cubic Hermite interpolant of _hermite_table,
    and with deriv also its slope; x is held inside the table, and a NaN
    stays one"""
    x0, h, coef = tab
    s = numpy.clip((x - x0) / h, 0.0, coef.shape[1] - 1e-6)
    i = numpy.where(s == s, s, 0.0).astype("int")
    t = s - i
    c0, c1, c2, c3 = coef[:, i]
    val = c0 + t * (c1 + t * (c2 + t * c3))
    if not deriv:
        return val
    return val, (c1 + t * (2.0 * c2 + 3.0 * t * c3)) / h


def _multisect(f, lo, hi, bits):
    """The first root of f in each of the brackets [lo, hi] (arrays; f is
    negative at lo and takes a point axis after the brackets'), located to
    2^-bits of the bracket by rounds of K-section, each one evaluation of f
    at the K - 1 interior points of every bracket. K is a power of two set
    by the number of brackets alone (bisection for many, 64 sections for a
    few, where the number of calls and not that of evaluations is the
    cost), so that the schedule is fixed."""
    logK = int(numpy.clip(numpy.log2(2048.0 / max(len(lo), 1)), 1, 6))
    w = numpy.arange(1, 2**logK) / 2**logK
    for _ in range(-(-bits // logK)):
        d = (hi - lo) / 2**logK
        with numpy.errstate(invalid="ignore"):
            neg = f(lo[:, None] + (hi - lo)[:, None] * w) < 0.0
        lo = lo + numpy.sum(numpy.cumprod(neg, axis=1), axis=1) * d
        hi = lo + d
    return 0.5 * (lo + hi)


def _contract(smt, D):
    """sum_m D[m] smt[m] for the harmonics smt, (M, npoints), and the
    coefficients D of one torus, (M,), or of every point's own torus,
    (M, npoints)"""
    return D @ smt if D.ndim == 1 else numpy.einsum("mp,mp->p", smt, D)


def _parse_target(target):
    """The phase-space rows (R, vR, vT, z, vz) of a target: an Orbit
    (possibly an array of orbits, or one evaluated at an array of times) or
    array rows (R, vR, vT, z, vz[, phi]) in internal units; phi, when
    present, is ignored because the potential is axisymmetric"""
    from ..orbit import Orbit

    if isinstance(target, Orbit):
        out = numpy.array(
            [
                numpy.atleast_1d(getattr(target, q)(use_physical=False)).ravel()
                for q in ("R", "vR", "vT", "z", "vz")
            ]
        )
    else:
        arr = numpy.atleast_2d(numpy.asarray(target, dtype="float"))
        if arr.ndim != 2 or arr.shape[1] not in (5, 6):
            raise ValueError(
                "target= rows must be (R, vR, vT, z, vz) or (R, vR, vT, z, vz, phi); "
                f"got shape {arr.shape}"
            )
        out = arr[:, :5].T.copy()
    if numpy.any(out[0] <= 0.0) or numpy.any(out[0] * numpy.fabs(out[2]) <= 0.0):
        raise ValueError(
            "target= needs R > 0 and L_z = R vT != 0 for every point: the grid is "
            "a box in |L_z| > 0"
        )
    return out


def _bisect(fn, lo, hi, n=40):
    """The roots of fn between the arrays lo, where it is negative, and hi,
    where it is positive, by n bisections of all of them at once"""
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        neg = fn(mid) < 0.0
        lo, hi = numpy.where(neg, mid, lo), numpy.where(neg, hi, mid)
    return 0.5 * (lo + hi)


def _golden(fn, lo, hi, n=40):
    """The minima of fn between the arrays lo and hi, where it has one
    each, by n golden-section steps of all of them at once"""
    g = 0.5 * (numpy.sqrt(5.0) - 1.0)
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = fn(a), fn(b)
    for _ in range(n):
        left = fa < fb
        lo, hi = numpy.where(left, lo, a), numpy.where(left, b, hi)
        a, b = (
            numpy.where(left, hi - g * (hi - lo), b),
            numpy.where(left, a, lo + g * (hi - lo)),
        )
        fnew = fn(numpy.where(left, a, b))
        fa, fb = numpy.where(left, fnew, fb), numpy.where(left, fa, fnew)
    return 0.5 * (lo + hi)


def _focal_tori(pot, E, Lz, f, delta, Rm, Rp, rule):
    """The tori (E, L_z, f) of the Staeckel approximations of pot with the
    focal lengths delta, f the fraction of the way of I_3 from the planar
    to the shell orbit's and all of them arrays of one shape, from the
    potential alone: the potential and the I_3 at which p_u^2 vanishes at
    R = delta sinh(u) in the plane, as functions; I_3; the radii in the
    plane of the two u turning points; and a function that returns V(v),
    p_v^2(v) and the v turning point for a reference radius R_0 = delta
    sinh(u0) and a third integral. Rm and Rp are the pericentre and the
    apocentre of the planar orbits (E, L_z), which do not depend on the
    focal length; the reference radius of the rule for u0 is the last
    entry"""

    def Phi(R, z=None):
        z = 0.0 * R if z is None else z
        return evaluatePotentials(
            pot, R.ravel(), z.ravel(), use_physical=False
        ).reshape(R.shape)

    def col(a, like):
        return a.reshape(a.shape + (1,) * (like.ndim - a.ndim))

    d2 = delta**2

    def w(R):
        P = Phi(R)
        return (R**2 * (col(E, R) - P) - col(d2, R) * P) / col(d2, R) - col(
            Lz, R
        ) ** 2 / (2.0 * R**2)

    Rsh = _golden(lambda R: -w(R), Rm, Rp, n=30)
    Ipl, Ish = Lz**2 / (2.0 * d2) - E, w(Rsh)
    I3 = Ipl + f * (Ish - Ipl)
    R1 = _bisect(lambda R: w(R) - I3, Rm, Rsh)
    R2 = _bisect(lambda R: I3 - w(R), Rsh, Rp)

    def vlib(R0, I):
        sh0 = R0 / delta
        ch0 = numpy.sqrt(1.0 + sh0**2)
        Phi0 = ch0**2 * Phi(R0)

        def V(v):
            sn = numpy.sin(v)
            return col(Phi0, v) - (col(sh0, v) ** 2 + sn**2) * Phi(
                col(delta * sh0, v) * sn, col(delta * ch0, v) * numpy.cos(v)
            )

        def Wv(v):
            sn2 = numpy.sin(v) ** 2
            return (
                2.0 * col(d2, v) * (col(E, v) * sn2 + V(v) + col(I, v))
                - col(Lz, v) ** 2 / sn2
            )

        return V, Wv, _bisect(Wv, 1e-8 + 0.0 * E, 0.5 * numpy.pi + 0.0 * E)

    R0 = {
        "torus": 2.0 * R1 * R2 / (R1 + R2),
        "planar": 2.0 * Rm * Rp / (Rm + Rp),
        "shell": Rsh,
    }[rule]
    return Phi, w, I3, Ish, R1, R2, vlib, R0


def _focal_spread(pot, E, Lz, f, delta, Rm, Rp, rule="torus", n=16):
    """The rms spread of the true Hamiltonian over the tori (E, L_z, f) of
    the Staeckel approximations of pot with the focal lengths delta
    (_focal_tori). H - E is the error potential Phi - Phi_S, averaged over
    the model torus with the time spent, (sinh^2 u + sin^2 v) du dv / (p_u
    p_v), on n x n/2 anomalies"""
    Phi, w, I3, _, R1, R2, vlib, R0 = _focal_tori(pot, E, Lz, f, delta, Rm, Rp, rule)
    V, Wv, vmin = vlib(R0, I3)
    u1, u2 = numpy.arcsinh(R1 / delta), numpy.arcsinh(R2 / delta)
    wu, wv = 0.5 * (u2 - u1), 0.5 * numpy.pi - vmin
    tu = numpy.pi * (numpy.arange(n) + 0.5) / n
    tv = 0.5 * numpy.pi * (numpy.arange(n // 2) + 0.5) / (n // 2)
    u = (0.5 * (u1 + u2))[..., None] - wu[..., None] * numpy.cos(tu)
    v = 0.5 * numpy.pi - wv[..., None] * numpy.cos(tv)
    Ru = delta[..., None] * numpy.sinh(u)
    gu = (
        wu[..., None]
        * numpy.sin(tu)
        / numpy.sqrt(
            numpy.maximum(2.0 * delta[..., None] ** 2 * (w(Ru) - I3[..., None]), 1e-300)
        )
    )
    gv = wv[..., None] * numpy.sin(tv) / numpy.sqrt(numpy.maximum(Wv(v), 1e-300))
    U = numpy.cosh(u) ** 2 * Phi(Ru)
    sh2, sn2 = numpy.sinh(u)[..., :, None] ** 2, numpy.sin(v)[..., None, :] ** 2
    dd = delta[..., None, None]
    dH = Phi(
        dd * numpy.sinh(u)[..., :, None] * numpy.sin(v)[..., None, :],
        dd * numpy.cosh(u)[..., :, None] * numpy.cos(v)[..., None, :],
    ) - (U[..., :, None] - V(v)[..., None, :]) / (sh2 + sn2)
    wt = (sh2 + sn2) * gu[..., :, None] * gv[..., None, :]
    norm = numpy.sum(wt, axis=(-2, -1))
    mean = numpy.sum(wt * dH, axis=(-2, -1)) / norm
    return numpy.sqrt(
        numpy.sum(wt * (dH - mean[..., None, None]) ** 2, axis=(-2, -1)) / norm
    )


def _focal_y(pot, E, Lz, f, delta, Rm, Rp, rule, axis_weight):
    """The third-integral variable y of a family's grid (_y_of_I3) for the
    tori of _focal_tori, on the arrays of all of them at once: the fraction
    f blended with the axis reach, from the turning points alone"""
    _, _, I3, Ish, R1, R2, vlib, R0 = _focal_tori(pot, E, Lz, f, delta, Rm, Rp, rule)
    # the v turning points are those of the model of the planar orbit
    Rpl = R0 if rule == "shell" else 2.0 * Rm * Rp / (Rm + Rp)

    def Qu(Ra, Rb):
        th = lambda R: R / numpy.sqrt(R**2 + delta**2)  # noqa: E731
        return 1.0 - numpy.sqrt(1.0 + 4.0 * numpy.log(th(Rb) / th(Ra)) ** 2)

    def Qv(I):
        return -2.0 * numpy.log(numpy.sin(vlib(Rpl, I)[2]))

    lpl, lsh = Qu(Rm, Rp), Qv(Ish)
    return (f + axis_weight * (Qu(R1, R2) + Qv(I3) - lpl)) / (
        1.0 + axis_weight * (lsh - lpl)
    )


def _focal_fit(pot, E, Lz, f, Rm, Rp, rule="torus", nscan=12):
    """The focal lengths that minimize _focal_spread for the tori (E, L_z,
    f): the best of nscan between 0.05 of the harmonic mean of the planar
    orbit's pericentre and apocentre and 2.5 of its apocentre, uniform in
    the logarithm, refined between its neighbours"""

    def spread(lnD):
        args = numpy.broadcast_arrays(E, Lz, f, numpy.exp(lnD), Rm, Rp)
        with numpy.errstate(all="ignore"):
            s = _focal_spread(pot, *args, rule=rule)
        return numpy.where(numpy.isfinite(s), s, numpy.inf)

    lo, hi = numpy.log(0.1 * Rm * Rp / (Rm + Rp)), numpy.log(2.5 * Rp)
    grid = lo + (hi - lo) * numpy.linspace(0.0, 1.0, nscan)[:, None]
    k = numpy.clip(numpy.argmin(spread(grid), axis=0), 1, nscan - 2)
    pick = numpy.arange(len(E))
    return numpy.exp(_golden(spread, grid[k - 1, pick], grid[k + 1, pick], n=20))


class actionAngleStaeckelInverse(actionAngleInverse):
    """Inverse action-angle transformation for axisymmetric Staeckel
    potentials through a momentum-matched canonical map onto an isochrone
    auxiliary (the ``auxiliary`` attribute); a torus is labelled by
    (E, L_z, I_3), the third integral in the convention
    p_u^2 = 2 delta^2 [E sinh^2 u - U(u) - I_3] - L_z^2 / sinh^2 u with the
    gauge V(pi/2) = 0"""

    # the format of an interpolated family's tables, stored with the family:
    # 2 since the half-widths are stored as the variables of _halfwidth
    _FAMILY_FORMAT = 2

    # the number of trial points that a pass of the label inversion's Newton
    # iteration evaluates at most for a torus (_newton_labels)
    _LABEL_STEPS = 32

    # the rule for a focal length that follows the tori (None: a single
    # one), and what of a Staeckel model goes with its focal length
    _delta_rule = None
    # the coefficients of the fitted focal length (delta='fit')
    _delta_fit = None
    _MODEL_STATE = "_delta _ush_cache _reach_cache _u_turning_memo _reach_v".split()

    def __setstate__(self, state):
        """Restore a pickled instance; an interpolated family whose tables
        were stored in another format (or without a format, as before the
        formats were numbered) would be decoded wrongly, and is refused;
        a family is pickled with its interpolants, so that it loads without
        rebuilding them"""
        if state.get("_interp", False) and state.get(
            "_family_format", 1
        ) != self._FAMILY_FORMAT + (state.get("_delta_rule") is not None):
            raise RuntimeError(
                "This interpolated actionAngleStaeckelInverse family was pickled "
                f"with the table format {state.get('_family_format', 1)}, while "
                f"this version of galpy reads the format {self._FAMILY_FORMAT}: "
                "its tables cannot be read; rebuild the family"
            )
        self.__dict__.update(state)

    def __init__(
        self,
        pot=None,
        Es=[-0.5],
        Lzs=[0.5],
        I3s=[0.1],
        jrs=None,
        jzs=None,
        setup_interp=False,
        Rmin=0.5,
        Rmax=2.0,
        Rinf=10.0,
        nE=9,
        nLz=9,
        Lzspacing="log",
        nI3=9,
        axis_weight=0.25,
        mm_npt=32,
        mm_nta=None,
        auxiliary=None,
        maxiter=100,
        angle_tol=1e-12,
        progressbar=True,
        delta=None,
        u0=None,
        target=None,
        target_pad=1.5,
        target_minwidth=0.02,
        delta_deg=(3, 3, 3),
        **kwargs,
    ):
        """
        Initialize an actionAngleStaeckelInverse object.

        Parameters
        ----------
        pot : Potential or a combined potential formed using addition (pot1+pot2+…)
            An axisymmetric potential of exact Staeckel form that supplies
            its focal length (an OblateStaeckelWrapperPotential or an exact
            Staeckel potential such as KuzminKutuzovStaeckelPotential); a
            general axisymmetric potential is accepted together with
            delta=, which wraps it in an OblateStaeckelWrapperPotential.
        Es : array-like or Quantity
            Energies of the tori to set up when setup_interp is False
            (paired with Lzs and I3s).
        Lzs : array-like or Quantity
            z-components of the angular momentum of the tori (non-zero).
        I3s : array-like or Quantity
            Third integrals of the tori, in the convention p_u^2 =
            2 delta^2 [E sinh^2 u - U(u) - I_3] - L_z^2 / sinh^2 u with the
            gauge V(pi/2) = 0 (an energy).
        jrs : array-like or Quantity, optional
            Radial actions of the tori to set up when setup_interp is
            False, in place of their integrals: with jzs and Lzs, the tori
            with the actions (J_R, L_z, J_z) are set up, their (E, I_3)
            solved for (in the Staeckel model that a rule for the reference
            curve assigns to each torus, when u0= is a rule). A vanishing
            action is the shell orbit (J_R = 0), the planar orbit (J_z =
            0) or the circular orbit (both).
        jzs : array-like or Quantity, optional
            Vertical actions of the tori to set up from their actions
            (paired with jrs and Lzs).
        setup_interp : bool, optional
            If True, set up a grid of tori in (L_z, E, I_3) spanning the
            circular angular momenta of [Rmin, Rmax], energies from the
            circular orbit's up to that of the planar orbit with apocentre
            Rinf, and third integrals from the planar orbit's to the shell
            orbit's, and interpolate canonically between them.
        Rmin : float or Quantity, optional
            Radius of the circular orbit with the smallest angular momentum
            of the interpolation grid.
        Rmax : float or Quantity, optional
            Radius of the circular orbit with the largest angular momentum
            of the interpolation grid.
        Rinf : float or Quantity, optional
            Apocentre, in the plane, of the orbit with the interpolation
            grid's largest energy at each angular momentum. The grid's
            energies are uniform in the logarithm of the apocentre of the
            planar orbit without angular momentum between each circular
            orbit's energy and this one (uniform in the energy for a
            logarithmic potential, and compressed toward the top in a
            potential with an escape energy, where the apocentre diverges),
            spaced quadratically at the circular face.
        nE : int, optional
            Number of energies of the interpolation grid.
        nLz : int, optional
            Number of angular momenta of the interpolation grid.
        Lzspacing : str, optional
            Spacing of the angular-momentum grid: 'log' (default; uniform
            in ln L_z), 'sqrt' (uniform in sqrt(L_z), finer at small L_z)
            or 'linear' (uniform in L_z). The default is the logarithm
            because every cell then spans the same factor in L_z: a linear
            grid's innermost cell spans a large factor in L_z when Rmin is
            much smaller than Rmax, and its interpolation errors then
            concentrate there, while its outer cells are finer than the
            logarithmic grid's; the two differ little for a grid that
            spans a factor of a few in L_z.
        nI3 : int, optional
            Number of third integrals of the interpolation grid.
        axis_weight : float, optional
            Weight of the tori's approach to the axis in the variable along
            which the grid's third integrals are spaced: that variable is
            the fraction of the way from the planar to the shell orbit's
            third integral plus this weight times a measure of how closely
            the two librations come to the axis, logarithmic in the
            distance of each libration's inner turning point from it. At
            small |L_z| the centrifugal barrier holds either the u or the v
            libration just off the axis, on either side of the third
            integral of the L_z = 0 orbits through the foci, and every
            table of a torus changes around that third integral over a
            range proportional to |L_z|, which nodes spaced in the third
            integral itself do not resolve however many they are; in the
            blended variable the tables are smooth uniformly in |L_z|, and
            the nodes that a family needs along it grow only with the
            logarithm of its smallest |L_z|. The tori that stay away from
            the axis (those of a disc) are spaced as with a weight of
            zero, which spaces every torus in the third integral alone.
            The default does not normally need to be changed: the measure
            is dimensionless and acts only on the tori that come close to
            the axis, whose accuracy depends weakly on the weight (a few
            tens of percent between weights of 0.1 and 0.5).
        mm_npt : int, optional
            Highest harmonic of the two momentum-matched anomaly maps (the
            v map's odd harmonics vanish by the symmetry of its libration
            about the midplane, so it stores the even ones up to this);
            the reconstruction converges spectrally in this, and a warning
            is raised when it does not suffice for a torus. The u-map needs
            a number of harmonics that grows with a torus's ratio of outer
            to inner radius in the plane, to at most about six times the
            square root of that ratio, and the v-map one that grows with
            L / |L_z| as the torus approaches a polar orbit; on a grid the
            most demanding tori are the ones at the smallest angular
            momentum (set by Rmin) and the highest energy (set by Rinf),
            and the set-up raises when the harmonics cannot hold them.
            With setup_interp=True this is the largest number that a torus
            may use: every torus keeps the coefficients of each map up to
            the last that exceeds 1e-11 and as many more as the next three
            decades take at the rate at which they fall between 1e-8 and
            1e-11 (down to their round-off floor, so that a torus of the
            grid is as accurate as with all mm_npt, while the number does
            not depend on round-off), and every point is evaluated with
            the harmonics that the eight tori around it keep (the highest
            of them in each map, rounded up to 16, 32, 64, ... or mm_npt),
            so that harmonics that only the most demanding tori need cost
            nothing elsewhere.
        mm_nta : int, optional
            Number of anomaly samples per libration (even), used to sample
            the torus, to fit the maps, and for the quadratures of its
            actions and frequencies; must exceed 4 * mm_npt for the samples
            to resolve the maps' highest harmonic, which is 2 * mm_npt. The
            default (None) is 8 * mm_npt.
        auxiliary : IsochronePotential, optional
            The isochrone auxiliary onto which every torus is lifted. By
            default it is fitted to the potential's rotation curve over the
            sampled radial range; either way it is available afterwards as
            the ``auxiliary`` attribute.
        maxiter : int, optional
            Maximum Newton iterations of the angle solve.
        angle_tol : float, optional
            Convergence tolerance of the angle solve.
        progressbar : bool, optional
            If True, display tqdm progress bars over the tori while they
            are lifted and their tables computed (requires tqdm to be
            installed). Default is True.
        delta : float, Quantity, 'fit' or callable, optional
            Focal length of the prolate spheroidal coordinate system, to
            wrap a general axisymmetric potential in the Staeckel
            approximation. A potential that supplies its own focal length
            accepts a delta= that agrees with it (to 1e-14) and rejects one
            that does not. A callable delta(E, Lz, y) lets the focal length
            follow the tori of an interpolated family: each torus is one of
            the Staeckel approximation in its own prolate spheroidal
            coordinates. It is called with the arrays of the energies,
            angular momenta (> 0) and third-integral variables of all the
            grid's nodes at once (internal units; y is the grid's variable,
            from 0 for the planar to 1 for the shell orbit of (E, L_z), in
            the model of the rule's focal length at y = 1/2), returns their
            focal lengths, and has to be smooth, also slightly outside the
            nodes' ranges (its partials are central differences). The focal
            length is then a table of the family, the derivative tables
            carry the variation of the model along it, the angles
            compensate the dilation of the coordinates, so that the map
            stays exactly canonical, and the frequencies are dE/dJ along
            the family. 'fit' is the built-in rule: on a
            sub-grid of the family's tori, the focal length that minimizes
            the spread of the true Hamiltonian over each torus of the
            Staeckel approximation, and a smooth polynomial fit of it
            (delta_deg); the family is that of the fitted polynomial, and
            its derivatives are those at the fit, held fixed. The delta
            method of a family returns its focal lengths and is itself such
            a rule (delta=family.delta gives another family of the same
            potential the same focal lengths without fitting them again,
            for instance the localized families of many targets). A rule
            requires setup_interp=True and one of the built-in rules for
            the reference curve (u0='torus' is the default with it).
        u0 : float, callable, 'torus', 'planar' or 'shell', optional
            Reference u of the wrapping (see OblateStaeckelWrapperPotential),
            whose curve R_0 = delta sinh(u0) is where the Staeckel
            approximation of a general potential is most accurate. A number
            (one, shared by all tori) requires delta= and cannot be given
            with a potential that supplies its own focal length. A rule gives every torus, of an
            interpolated family or set up explicitly, its own reference
            curve, so that each torus is a torus of its own Staeckel model
            (without u0=, the explicit tori of a general potential share
            the wrapper's default curve, which crosses the midplane at R =
            1 wherever the torus lies: pass u0='torus'). 'torus', the rule to
            prefer, places the curve at the harmonic mean of the radii at
            which the torus's own inner and outer turning spheroids cross
            the midplane: it weights the inner turning point, where a
            disc's vertical structure is sharpest, and follows the torus as
            it tilts out of the plane. 'planar' is that rule on the planar
            orbit of the torus's (E, L_z) (the harmonic mean of its
            pericentre and apocentre): the same as 'torus' near the plane,
            and far inside an inclined torus. 'shell' is that rule on the
            shell orbit of (E, L_z). A callable u0(E, Lz) or u0(E, Lz, I3)
            (internal units, with L_z > 0; it is given I_3 when it has a
            third positional parameter without a default) is the user's own
            rule, which has to be smooth and, with three arguments, the
            same for every I_3 at the circular orbit's energy.
            The variation of the Staeckel model from torus to torus is
            part of the derivative tables, so that a family converges
            between its nodes, at the interpolation's order, to the tori
            of the models that the rule assigns, which are the tori that
            the same rule sets up explicitly; the frequencies and the
            angles are those of that family of models (the frequencies are
            the derivatives of the energy with respect to the actions along
            it), which differ from those of each torus's own Staeckel model
            by a term of the order of the potential's departure from
            Staeckel form (none for a Staeckel potential; pass the rule's
            u0 as a number to set up a torus in its own model alone).
            The reference curve is construction-only (the map's
            (R, z) <-> (u, v) uses the focal length alone), so a rule costs
            nothing at evaluation.
        target : Orbit or array_like, optional
            Localize the interpolation grid on the tori of a set of
            phase-space points: an Orbit (possibly an array of orbits, or
            one evaluated at an array of times) or rows (R, vR, vT, z, vz[,
            phi]) in internal units. The grid's box in its own coordinates
            (L_z, the energy variable w_E, the third-integral variable
            w_I) is then the points' own range, padded, instead of the
            default box between the circular orbits of Rmin and Rmax and
            from the circular to the Rinf energies; the same number of
            nodes then resolves a target orbit or stream far better. The
            labelling does a few root-finds per point, so subsample a very
            large set. Requires setup_interp=True.
        target_pad : float, optional
            Padding of the target's box: each axis is extended by this
            factor times the target's spread on both ends. Generous
            padding is load-bearing, because a box tight around the target
            puts it in the boundary layer of the interpolation stencil, a
            failure mode that is silent.
        target_minwidth : float, optional
            Minimum width of the target's box on each axis, as a fraction
            of that axis's default span: what keeps the box of a single
            orbit, whose spread is zero on every axis, a grid rather than a
            point.
        delta_deg : tuple of int, optional
            The degrees of the polynomial that delta='fit' fits to the
            logarithm of the focal length, in the energy variable, the
            third-integral variable and ln L_z; the fit uses 2 deg + 1
            tori along each.

        Notes
        -----
        - Angle conventions match those of the forward actionAngleStaeckel:
          theta_R = 0 at the inner u turning point and theta_z = 0 at the
          upward midplane crossing at pericentre.
        - 2026-09-24 - Started - Bovy (UofT)
        """
        actionAngleInverse.__init__(self, *[], **kwargs)
        if pot is None:
            raise OSError("Must specify pot= for actionAngleStaeckelInverse")
        self._delta_deg = delta_deg
        self._setup_staeckel(_check_potential_list_and_deprecate(pot), delta, u0)
        self._target = None if target is None else _parse_target(target)
        self._target_pad, self._target_minwidth = target_pad, target_minwidth
        if self._delta_rule is not None and not setup_interp:
            raise TypeError(
                "a focal length that follows the tori (delta='fit' or a callable) "
                "builds an interpolated family and requires setup_interp=True"
            )
        if self._target is not None and not setup_interp:
            raise TypeError(
                "target= localizes the interpolation grid and requires setup_interp=True"
            )
        if mm_nta is None:
            mm_nta = 8 * mm_npt
        if mm_nta % 2 == 1:
            raise ValueError("mm_nta has to be even")
        if mm_nta <= 4 * mm_npt:
            # the maps' highest harmonic is 2 mm_npt, which mm_nta uniform
            # samples only resolve below their Nyquist harmonic mm_nta / 2
            raise ValueError(
                "mm_nta must exceed 4 * mm_npt for the anomaly samples to resolve the maps' harmonics"
            )
        self._ntau = mm_nta
        self._npt = mm_npt
        if mm_npt < 2:
            raise ValueError(
                "mm_npt must be at least 2: the v anomaly map has even harmonics only and needs one"
            )
        self._nforDm = numpy.arange(1, mm_npt + 1)
        # the v map's odd harmonics vanish: the libration is symmetric about
        # the midplane, and with it the matching, eta(pi - tau) = pi - eta(tau)
        self._nforDmv = numpy.arange(2, mm_npt + 1, 2)
        self._nptv = len(self._nforDmv)
        self._maxiter = maxiter
        self._angle_tol = angle_tol
        self._progressbar = progressbar and _TQDM_LOADED
        self._interp = setup_interp
        self._circ_cache = {}
        self._ush_cache = {}
        self._ubar_cache = {}
        self._reach_cache = {}
        self._midplane_tab, self._Vref_tab = None, None
        self._axisw = axis_weight
        if not setup_interp:
            self._Lzs = conversion._parse_grid_quantity(
                Lzs, conversion.parse_angmom, ro=self._ro, vo=self._vo
            ).astype("float")
            if numpy.any(self._Lzs == 0.0):
                raise ValueError(
                    "L_z = 0 is not supported: the v libration of a polar orbit reaches the axis"
                )
            if jrs is None and jzs is None:
                self._Es = conversion._parse_grid_quantity(
                    Es, conversion.parse_energy, vo=self._vo
                ).astype("float")
                self._I3s = conversion._parse_grid_quantity(
                    I3s, conversion.parse_energy, vo=self._vo
                ).astype("float")
                if not len(self._Es) == len(self._Lzs) == len(self._I3s):
                    raise ValueError("Es, Lzs, and I3s have to have the same length")
            else:
                if jrs is None or jzs is None:
                    raise ValueError(
                        "tori are set up from their actions with both jrs= and "
                        "jzs= (and Lzs=)"
                    )
                jrs, jzs = (
                    conversion._parse_grid_quantity(
                        j, conversion.parse_angmom, ro=self._ro, vo=self._vo
                    ).astype("float")
                    for j in (jrs, jzs)
                )
                if not len(jrs) == len(self._Lzs) == len(jzs):
                    raise ValueError("jrs, Lzs, and jzs have to have the same length")
                if numpy.any(jrs < 0.0) or numpy.any(jzs < 0.0):
                    raise ValueError("J_R and J_z have to be non-negative")
                self._Es, self._I3s, found = self._integrals_many(
                    jrs, numpy.fabs(self._Lzs), jzs
                )
                if not numpy.all(found):
                    first = numpy.argmin(found)
                    raise ValueError(
                        "No torus with the actions (J_R, |L_z|, J_z) of "
                        f"{numpy.sum(~found)} of the {len(found)} tori was found "
                        "in this Staeckel model; the first is "
                        f"({jrs[first]}, {numpy.fabs(self._Lzs[first])}, "
                        f"{jzs[first]}). Either the model has no such torus "
                        "(close to the axis the tori of a Staeckel approximation "
                        "are not one continuous family, and the actions jump "
                        "where two of them merge) or the torus is unbound"
                    )
        else:
            self._setup_grid(
                conversion.parse_length(Rmin, ro=self._ro),
                conversion.parse_length(Rmax, ro=self._ro),
                conversion.parse_length(Rinf, ro=self._ro),
                nE,
                nLz,
                nI3,
                Lzspacing=Lzspacing,
            )
        # find every torus's turning points, fix the frozen auxiliary over
        # the radii that they span, then, torus by torus, sample the torus
        # (exact placement), lift it onto the auxiliary, libration by
        # libration (all torus-dependence beyond the auxiliary lives in the
        # two momentum-matched maps, whose compensation is closed-form), and
        # compute its tables: the samples and the lifts of a torus, which
        # are far larger than its tables, are dropped once those are computed
        self._setup_turning_points()
        self._setup_auxiliary(auxiliary)
        self._setup_tori()
        # the turning points found and remembered during the set-up are
        # not kept, except those of explicit tori, whose second-order
        # tables are computed when they are first asked for
        if self._interp:
            del self._tps
        self._u_turning_memo, self._memo_keep, self._reach_v = {}, False, (None, None)
        # (nor are the tables of the solve of tori from their actions)
        self._midplane_tab, self._Vref_tab = None, None
        self._check_consistent_units()
        if self._delta_rule is not None:
            # the family is its tables: neither the rule nor the nodes'
            # models are kept (and the family pickles whatever the callable)
            self._delta_rule, self._models, self._chart_cache = True, {}, {}
            del self._delta_tab, self._ddelta_tab, self._delta_col
        return None

    def _setup_staeckel(self, pot, delta, u0):
        """The Staeckel model the tori live in: the potential itself when it
        supplies its focal length, or its Staeckel approximation at the
        given focal length; with an adaptive reference curve, that model
        is the reference one, and every (E, L_z) of the family gets its
        own (_chart)"""
        if isinstance(u0, str) and u0 not in ("torus", "planar", "shell"):
            raise ValueError(
                "the string values u0= accepts are 'torus', 'planar' and 'shell'"
            )
        self._models = {}
        if isinstance(delta, str) and delta != "fit":
            raise ValueError("the only string value delta= accepts is 'fit'")
        if callable(delta) or isinstance(delta, str):
            # (the wrapper set up here only serves the midplane)
            if not (u0 is None or isinstance(u0, str)):
                raise TypeError(
                    "a focal length that follows the tori (delta='fit' or a callable) "
                    "needs a built-in rule for the reference curve: u0='torus' "
                    "(the default with it), 'planar' or 'shell'"
                )
            self._delta_rule, delta, u0 = delta, 1.0, "torus" if u0 is None else u0
        self._adaptive = callable(u0) or isinstance(u0, str)
        if numpy.ndim(u0) != 0:
            raise TypeError(
                "a numeric u0= is one reference curve shared by all tori; set "
                "up tori with different ones in separate instances, or use a "
                "rule (u0='torus', 'planar', 'shell' or a callable)"
            )
        # the rule: one of the built-in ones, on the turning points of a
        # torus, or a callable of (E, L_z) or of (E, L_z, I_3)
        self._u0_rule = u0 if self._adaptive else None
        self._u0_nargs = 2
        if callable(u0):
            # a callable takes I_3 when it has a third positional parameter
            # without a default (or *args)
            pars = list(inspect.signature(u0).parameters.values())
            if (
                any(q.kind == q.VAR_POSITIONAL for q in pars)
                or sum(
                    q.kind in (q.POSITIONAL_ONLY, q.POSITIONAL_OR_KEYWORD)
                    and q.default is q.empty
                    for q in pars
                )
                >= 3
            ):
                self._u0_nargs = 3
        # whether the reference curve varies along the third integral
        self._u0_third = u0 == "torus" or self._u0_nargs == 3
        self._chart_cache = {}
        self._circRE_cache = {}
        self._u_turning_memo, self._memo_keep, self._reach_v = {}, True, (None, None)
        if self._adaptive:
            # the reference wrapper keeps the default reference curve
            u0 = None
        own = None if self._delta_rule is not None else _focal_length_of(pot)
        if delta is not None:
            delta = conversion.parse_length(delta, ro=self._ro)
            if own is not None:
                # the potential's own focal length wins; a delta= that agrees
                # with it is harmless, one that does not is a contradiction
                if u0 is not None:
                    raise TypeError(
                        "u0= cannot be given with a potential that already "
                        "supplies its focal length"
                    )
                if not numpy.fabs(delta - own) <= 1e-14 * numpy.fabs(own):
                    raise TypeError(
                        "delta= {:g} disagrees with the focal length {:g} that the "
                        "potential supplies; pass the potential without delta= to "
                        "use its own".format(delta, own)
                    )
                delta = None
        if delta is not None:
            self._staeckelwrap = (
                OblateStaeckelWrapperPotential(pot=pot, delta=delta)
                if u0 is None
                else OblateStaeckelWrapperPotential(pot=pot, delta=delta, u0=u0)
            )
        elif u0 is not None:
            raise TypeError("u0= requires delta=")
        elif isinstance(pot, OblateStaeckelWrapperPotential):
            self._staeckelwrap = pot
        elif own is not None:
            self._staeckelwrap = OblateStaeckelWrapperPotential(pot=pot, delta=own)
        else:
            raise OSError(
                "actionAngleStaeckelInverse requires a potential of Staeckel "
                "form that supplies its focal length (an "
                "OblateStaeckelWrapperPotential or an exact Staeckel potential "
                "such as KuzminKutuzovStaeckelPotential), or delta= to wrap a "
                "general axisymmetric potential in the Staeckel approximation"
            )
        self._pot = pot
        self._delta = self._staeckelwrap._delta
        # the potential the per-(E, L_z) charts of an adaptive family wrap:
        # always the raw one (wrapping a wrapper would approximate the
        # approximation)
        self._chart_pot = self._staeckelwrap._pot
        return None

    def _chart(self, E, Lz, I3=None):
        """The Staeckel model of the (E, L_z, I_3) torus: the family's single
        wrapper or, with a rule for the reference curve, the wrapper at this
        torus's own u0, cached. Without a third integral, that of the planar
        orbit of (E, L_z) stands in (for what does not depend on the
        reference curve: U(u) and everything of the u libration)"""
        if not self._adaptive:
            return self._staeckelwrap
        E, ell = float(E), float(numpy.fabs(Lz))
        if not self._u0_third:
            I3 = None
        elif I3 is None:
            I3 = self._I3_planar(E, ell)
        key = (E, ell, I3 if I3 is None else float(I3), self._delta)
        if key not in self._chart_cache:
            self._chart_cache[key] = OblateStaeckelWrapperPotential(
                pot=self._chart_pot,
                delta=self._delta,
                u0=float(self._u0_of(E, ell, I3)),
            )
        return self._chart_cache[key]

    @contextmanager
    def _in_chart(self, E, Lz, I3=None, delta=None):
        """Run the chart-dependent construction of the (E, L_z, I_3) torus
        (its sampling, turning points and edge relations) in its own
        Staeckel model: that of the current focal length or, for a torus
        of a family whose focal length follows its tori, of the focal length
        delta, with what is remembered of it (I3 False: for U(u) alone)"""
        keys = ["_staeckelwrap"] + self._MODEL_STATE[: 1 + 4 * (delta is not None)]
        save = {k: self.__dict__[k] for k in keys}
        if delta is not None:
            fresh = {"_delta": delta, "_reach_v": (None, None)}
            fresh.update({k: {} for k in self._MODEL_STATE[1:4]})
            self.__dict__.update(self._models.setdefault(delta, fresh))
            # (U(u) is that of this focal length)
            if delta not in self._chart_cache:
                self._chart_cache[delta] = OblateStaeckelWrapperPotential(
                    pot=self._chart_pot, delta=delta
                )
            self._staeckelwrap = self._chart_cache[delta]
        if I3 is not False:
            self._staeckelwrap = self._chart(E, Lz, I3)
        try:
            yield
        finally:
            if delta is not None:
                self._models[delta] = {k: self.__dict__[k] for k in self._MODEL_STATE}
            self.__dict__.update(save)

    def _u0_of(self, E, ell, I3):
        """The reference u0 of the (E, |L_z|, I_3) torus under the rule (I_3
        is None for a rule that does not use it)"""
        rule = self._u0_rule
        if callable(rule):
            return rule(E, ell, I3) if self._u0_nargs == 3 else rule(E, ell)
        if rule == "shell":
            if self._on_circular(E, ell):
                return self._circular_RE(ell)[2]
            return self._ushell(E, ell)
        return self._u0_torus(
            E, ell, self._I3_planar(E, ell) if rule == "planar" else I3
        )[0]

    def _u0_torus(self, E, ell, I3, deriv=False):
        """The reference curve that follows a torus's own turning points: the
        harmonic mean of the radii R_- and R_+ at which the torus's inner and
        outer turning spheroids u_- and u_+ cross the midplane, delta
        sinh(u0) = 2 / (1/R_- + 1/R_+) with R = delta sinh(u). It weights the
        inner turning point, where a disc's vertical structure is sharpest;
        it is the harmonic mean of the pericentre and apocentre for the
        planar orbit (u0='planar' is this rule at the planar orbit's I_3), the
        shell orbit's own spheroid for the shell orbit (u0='shell'), and the
        circular radius for the circular orbit; and it is even in the u
        libration's half-width, hence smooth in the variables in which the
        family's actions are smooth. The turning points do not depend on the
        reference curve (U(u) does not), so the rule is explicit. Returns u0
        and, with deriv, its partials in (E, I_3, L_z): the level-set rule
        on the two turning points or, for the shell orbit, where that
        degenerates, the limit through the momentum function's expansion
        about the shell u to third order."""
        umin, umax, ush, udeg = self._u_turning(E, ell, I3)
        d = self._delta
        Rm, Rp = d * numpy.sinh(umin), d * numpy.sinh(umax)
        H = 2.0 * Rm * Rp / (Rm + Rp)
        u0 = numpy.arcsinh(H / d)
        if not deriv:
            return u0, None
        if not udeg:
            dH = (
                2.0
                * (
                    Rp**2 * d * numpy.cosh(umin) * self._tp_derivs_u(umin, E, ell)
                    + Rm**2 * d * numpy.cosh(umax) * self._tp_derivs_u(umax, E, ell)
                )
                / (Rm + Rp) ** 2
            )
        else:
            # about the shell u, p_u^2 = W_0 + W_2 s^2/2 + W_3 s^3/6 with W_0
            # = 2 delta^2 (I_sh - I_3): the half-width squared is a^2 = -2
            # W_0/W_2, the midpoint sits at s = -W_3 a^2/(6 W_2), and the
            # harmonic mean of f = delta sinh(u) at the two turning points
            # is f + f' s + (f''/2 - f'^2/f) a^2
            hstep = 1e-4
            W2 = float(self._d2Wu(ush, E, ell))
            W3 = float(
                self._d2Wu(ush + hstep, E, ell) - self._d2Wu(ush - hstep, E, ell)
            ) / (2.0 * hstep)
            f, fp = d * numpy.sinh(ush), d * numpy.cosh(ush)
            K = -fp * W3 / (6.0 * W2) + 0.5 * f - fp**2 / f
            dush = self._dushell(ush, E, ell)
            sh2 = numpy.sinh(ush) ** 2
            dHdI = 4.0 * d**2 * K / W2
            dH = numpy.array(
                [
                    fp * dush[0] - dHdI * sh2,
                    dHdI,
                    fp * dush[1] + dHdI * ell / (d**2 * sh2),
                    *(
                        fp * numpy.array(dush[2:])
                        - dHdI * self._faces_ddelta(ush, ell)[1]
                    ),
                ]
            )
        return u0, dH / (d * numpy.cosh(u0))

    def _u0_grad(self, E, ell, I3):
        """The partials of the reference u0 of the (E, |L_z|, I_3) torus (not
        the circular orbit) in (E, I_3, L_z), which carry the variation of
        the Staeckel model from torus to torus into the family's derivative
        tables: closed forms for the built-in rules, and central differences
        of a callable in (E, y, L_z), with y the position of I_3 between the
        planar and the shell orbit's, in which the tori of neighbouring (E,
        L_z) exist up to both faces"""
        rule = self._u0_rule
        if rule == "torus":
            return self._u0_torus(E, ell, I3, deriv=True)[1]
        if rule == "planar":
            g = self._u0_torus(E, ell, self._I3_planar(E, ell), deriv=True)[1]
            return numpy.array(
                [
                    g[0] - g[1],
                    0.0,
                    g[2] + g[1] * ell / self._delta**2,
                    *(g[3:] - g[1] * ell**2 / self._delta**3),
                ]
            )
        if rule == "shell":
            dush = self._dushell(self._ushell(E, ell), E, ell)
            return numpy.array([dush[0], 0.0, *dush[1:]])

        def edges(EE, LL):
            Ipl = self._I3_planar(EE, LL)
            return Ipl, self._I3_shell(EE, LL, self._ushell(EE, LL))

        def g(EE, LL, yy):
            Ipl, Ish = edges(EE, LL)
            return self._u0_of(EE, LL, Ipl + yy * (Ish - Ipl))

        Ec, Omc = self._circular_RE(ell)[1], ell / self._circular_RE(ell)[0] ** 2
        Ipl, Ish = edges(E, ell)
        y = min(max((I3 - Ipl) / (Ish - Ipl), 0.0), 1.0)
        hE = 1e-4 * (E - Ec)
        hL = min(1e-5 * ell, 1e-3 * (E - Ec) / Omc)
        gE = (g(E + hE, ell, y) - g(E - hE, ell, y)) / (2.0 * hE)
        gL = (g(E, ell + hL, y) - g(E, ell - hL, y)) / (2.0 * hL)
        gI = 0.0
        if self._u0_nargs == 3:
            hy = 1e-4
            if y <= 1.0 - hy:
                ylo = max(y - hy, 0.0)
                gy = (g(E, ell, y + hy) - g(E, ell, ylo)) / (y + hy - ylo)
            else:
                # one-sided, to second order, at the shell face (the planar
                # face's nodes have no v libration, and need no derivative)
                gy = (
                    3.0 * g(E, ell, y)
                    - 4.0 * g(E, ell, y - hy)
                    + g(E, ell, y - 2.0 * hy)
                ) / (2.0 * hy)
            gI = gy / (Ish - Ipl)
        # from fixed y to fixed I_3: I_3 = I_pl + y (I_sh - I_pl), whose
        # partials at fixed y are those of the grid (_setup_tori_interp)
        sh2 = numpy.sinh(self._ushell(E, ell)) ** 2
        dI_dE = y * (1.0 + sh2) - 1.0
        dI_dL = ell / self._delta**2 * (1.0 - y * (1.0 + 1.0 / sh2))
        return numpy.array([gE - gI * dI_dE, gI, gL - gI * dI_dL])

    # ---------- the separated Hamilton-Jacobi equation
    def _U(self, u):
        u = numpy.asarray(u, dtype="float")
        return self._staeckelwrap._U(u.ravel()).reshape(u.shape)

    def _dU(self, u):
        u = numpy.asarray(u, dtype="float")
        return self._staeckelwrap._dUdu(u.ravel()).reshape(u.shape)

    def _d2U(self, u):
        u = numpy.asarray(u, dtype="float")
        return self._staeckelwrap._d2Udu2(u.ravel()).reshape(u.shape)

    def _V(self, v):
        v = numpy.asarray(v, dtype="float")
        return self._staeckelwrap._V(v.ravel()).reshape(v.shape)

    def _dV(self, v):
        v = numpy.asarray(v, dtype="float")
        return self._staeckelwrap._dVdv(v.ravel()).reshape(v.shape)

    def _d2V(self, v):
        v = numpy.asarray(v, dtype="float")
        return self._staeckelwrap._d2Vdv2(v.ravel()).reshape(v.shape)

    def _Wu(self, u, E, Lz, I3):
        """p_u^2 on the (E, L_z, I_3) torus"""
        sh2 = numpy.sinh(u) ** 2
        return 2.0 * self._delta**2 * (E * sh2 - self._U(u) - I3) - Lz**2 / sh2

    def _Wv(self, v, E, Lz, I3):
        """p_v^2 on the (E, L_z, I_3) torus"""
        sn2 = numpy.sin(v) ** 2
        return 2.0 * self._delta**2 * (E * sn2 + self._V(v) + I3) - Lz**2 / sn2

    def _dWu(self, u, E, Lz):
        """d p_u^2 / du (independent of I_3)"""
        return (
            2.0 * self._delta**2 * (E * numpy.sinh(2.0 * u) - self._dU(u))
            + 2.0 * Lz**2 * numpy.cosh(u) / numpy.sinh(u) ** 3
        )

    def _dWv(self, v, E, Lz):
        """d p_v^2 / dv (independent of I_3)"""
        return (
            2.0 * self._delta**2 * (E * numpy.sin(2.0 * v) + self._dV(v))
            + 2.0 * Lz**2 * numpy.cos(v) / numpy.sin(v) ** 3
        )

    def _d2Wu(self, u, E, Lz):
        """d^2 p_u^2 / du^2 (independent of I_3)"""
        sh2 = numpy.sinh(u) ** 2
        return (
            2.0 * self._delta**2 * (2.0 * E * numpy.cosh(2.0 * u) - self._d2U(u))
            - Lz**2 * (6.0 + 4.0 * sh2) / sh2**2
        )

    def _d2Wv(self, v, E, Lz):
        """d^2 p_v^2 / dv^2 (independent of I_3)"""
        sn2 = numpy.sin(v) ** 2
        return (
            2.0 * self._delta**2 * (2.0 * E * numpy.cos(2.0 * v) + self._d2V(v))
            - 2.0 * Lz**2 * (3.0 - 2.0 * sn2) / sn2**2
        )

    def _dW_ddelta(self, q, Lz, W=0.0, lib="u"):
        """The derivative of p_u^2 (for lib 'v', of p_v^2) with respect to
        the focal length at fixed coordinate q, (E, I_3, L_z) and reference
        u0, where the momentum squared is W (zero at a turning point): 2 (W
        + L_z^2/sinh^2 u)/delta - 2 delta^2 dU/ddelta, and 2 (W + L_z^2/sin^2
        v)/delta + 2 delta^2 dV/ddelta, with dU/ddelta = cosh^2 u sinh u
        dPhi/dR(delta sinh u, 0) and dV/ddelta = cosh^2 u0 sinh u0
        dPhi/dR(R_0, 0) - (sinh^2 u0 + sin^2 v) [sinh u0 sin v dPhi/dR +
        cosh u0 cos v dPhi/dz] on the reference spheroid. A list of one
        element when the focal length follows the tori (the fourth direction
        of all that is differentiated along the integrals), else empty"""
        if self._delta_rule is None:
            return []
        d, pot = self._delta, self._chart_pot
        if lib == "u":
            sh = numpy.sinh(q)
            s2 = sh**2
            dUV = (
                numpy.cosh(q) ** 2
                * sh
                * evaluateRforces(pot, d * sh, 0.0, use_physical=False)
            )
        else:
            u0 = self._staeckelwrap._u0
            sh, ch, sn, cs = numpy.sinh(u0), numpy.cosh(u0), numpy.sin(q), numpy.cos(q)
            s2 = sn**2
            R, z = d * sh * sn, d * ch * cs
            dUV = (sh**2 + s2) * (
                sh * sn * evaluateRforces(pot, R, z, use_physical=False)
                + ch * cs * evaluatezforces(pot, R, z, use_physical=False)
            ) - ch**2 * sh * evaluateRforces(pot, d * sh, 0.0, use_physical=False)
        return [2.0 * (W + Lz**2 / s2) / d + 2.0 * d**2 * dUV]

    def _faces_ddelta(self, ush, Lz):
        """The derivatives with respect to the focal length of the planar
        and of the shell orbit's I_3 at fixed (E, L_z), as arrays of one
        element or, for a single focal length, of none (_dW_ddelta)"""
        d = self._delta
        return (
            numpy.array([-(Lz**2) / d**3] * len(self._dW_ddelta(ush, Lz))),
            numpy.array(self._dW_ddelta(ush, Lz)) / (2.0 * d**2),
        )

    def _tp_derivs_u(self, u, E, Lz):
        """Closed-form derivatives of a u turning point (a root of p_u^2)
        with respect to (E, I_3, L_z), by the level-set rule"""
        sh2 = numpy.sinh(u) ** 2
        d2 = self._delta**2
        return -numpy.array(
            [2.0 * d2 * sh2, -2.0 * d2, -2.0 * Lz / sh2, *self._dW_ddelta(u, Lz)]
        ) / float(self._dWu(u, E, Lz))

    def _tp_derivs_v(self, v, E, Lz):
        """Closed-form derivatives of the v turning point with respect to
        (E, I_3, L_z), by the level-set rule"""
        sn2 = numpy.sin(v) ** 2
        d2 = self._delta**2
        return -numpy.array(
            [
                2.0 * d2 * sn2,
                2.0 * d2,
                -2.0 * Lz / sn2,
                *self._dW_ddelta(v, Lz, lib="v"),
            ]
        ) / float(self._dWv(v, E, Lz))

    # ---------- the circular, shell, and planar edges
    def _circular(self, Lz):
        """The circular orbit of angular momentum |L_z|: its radius and
        energy, its u, and its epicycle, vertical, and circular
        frequencies, cached"""
        ell = numpy.fabs(Lz)
        if ell not in self._circ_cache:
            Rc, Ec, uc = self._circular_RE(ell)
            wrap = self._chart(Ec, ell)
            self._circ_cache[ell] = {
                "Rc": Rc,
                "Ec": Ec,
                "uc": uc,
                "kappa": epifreq(wrap, Rc, use_physical=False),
                "nu": verticalfreq(wrap, Rc, use_physical=False),
                "Omc": ell / Rc**2,
            }
        circ = self._circ_cache[ell]
        if self._delta_rule is not None:
            # (its u in the current model, one of many)
            circ = {**circ, "uc": numpy.arcsinh(circ["Rc"] / self._delta)}
        return circ

    def _circular_RE(self, ell):
        """The radius, energy and u of the circular orbit of angular
        momentum ell > 0, from the potential's midplane, which every chart
        reproduces; cached"""
        if ell not in self._circRE_cache:
            Rc = rl(self._staeckelwrap, ell, use_physical=False)
            Ec = evaluatePotentials(
                self._staeckelwrap, Rc, 0.0, use_physical=False
            ) + ell**2 / (2.0 * Rc**2)
            self._circRE_cache[ell] = (Rc, Ec)
        Rc, Ec = self._circRE_cache[ell]
        return Rc, Ec, numpy.arcsinh(Rc / self._delta)

    def _on_circular(self, E, ell):
        """Whether E is the circular orbit's energy at angular momentum ell,
        to the tolerance of the torus construction"""
        Ec = self._circular_RE(ell)[1]
        return E < Ec + 1e-12 * (1.0 + numpy.fabs(Ec))

    def _ushell(self, E, Lz):
        """The u of the shell orbit of (E, L_z), where p_u^2 attains its
        maximum over u (independent of I_3), cached"""
        key = (E, Lz)
        if key not in self._ush_cache:
            us = numpy.geomspace(1e-4, 20.0, 400)
            with numpy.errstate(invalid="ignore", over="ignore"):
                # the potential may not evaluate at the scan's extreme radii
                W = self._Wu(us, E, Lz, 0.0)
            finite = numpy.isfinite(W)
            us, W = us[finite], W[finite]
            tops = numpy.flatnonzero((W[1:-1] > W[:-2]) & (W[1:-1] >= W[2:])) + 1
            if (
                self._interp
                and numpy.sum(W[tops] > 2.0 * self._delta**2 * self._I3_planar(E, Lz))
                > 1
            ):
                # at small |L_z| in a cusped potential p_u^2 has a second
                # maximum close to the centre: tori there share their
                # integrals with tori further out and merge with them where
                # I_3 lifts the barrier between the two above zero, across
                # which the tori are not a continuous family
                raise ValueError(
                    "The momentum function of the u libration of the tori at "
                    f"(E, L_z) = ({E}, {Lz}) has more than one maximum: tori "
                    "close to the centre share their integrals with tori further "
                    "out (as they do at small |L_z| in a cusped potential), the "
                    "tori at these integrals are not a continuous family, and an "
                    "interpolation grid cannot hold them. Keep the grid away "
                    "from the axis (a larger Rmin) or change delta"
                )
            j = int(numpy.argmax(W))
            if j == 0 or j == len(us) - 1:
                raise ValueError(
                    f"No bound u libration exists at (E, L_z) = ({E}, {Lz}): "
                    "the torus is unbound"
                )
            dWu = lambda u: float(self._dWu(u, E, Lz))
            ush = brentq(dWu, us[j - 1], us[j + 1], xtol=1e-14)
            # the barriers of p_u^2 next to the shell u on either side, where
            # it has more than one maximum: (u, p_u^2 at I_3 = 0) of the
            # nearest local minimum (refined where the scan brackets it),
            # or None
            dips = numpy.flatnonzero((W[1:-1] < W[:-2]) & (W[1:-1] <= W[2:])) + 1
            bars = []
            for side in (dips[dips < j][-1:], dips[dips > j][:1]):
                bars.append(None)
                for k in side:
                    bracketed = dWu(us[k - 1]) < 0.0 < dWu(us[k + 1])
                    ub = brentq(dWu, us[k - 1], us[k + 1]) if bracketed else us[k]
                    bars[-1] = (ub, float(self._Wu(ub, E, Lz, 0.0)))
            self._ush_cache[key], self._ubar_cache[key] = ush, bars
        return self._ush_cache[key]

    def _dushell(self, ush, E, Lz):
        """Derivatives of the shell u with respect to E and to L_z, by the
        level-set rule on d p_u^2 / du = 0"""
        d2W = float(self._d2Wu(ush, E, Lz))
        out = (
            -2.0 * self._delta**2 * numpy.sinh(2.0 * ush) / d2W,
            -4.0 * Lz * numpy.cosh(ush) / numpy.sinh(ush) ** 3 / d2W,
        )
        if self._delta_rule is None:
            return out
        # and with respect to the focal length: R = delta sinh(u) of the
        # shell orbit is where 2 R (E - Phi) - (delta^2 + R^2) Phi' + L_z^2
        # delta^2 / R^3 vanishes, with R-derivative p_u^2'' / (2 delta^2 cosh^2 u)
        d, sh, ch = self._delta, numpy.sinh(ush), numpy.cosh(ush)
        R = d * sh
        dR = -4.0 * d**3 * ch**2 * (Lz**2 / R**2 - self._RdPhi(R)) / R / d2W
        return (*out, (dR - sh) / (d * ch))

    def _I3_planar(self, E, Lz):
        """I_3 of the planar orbit (J_z = 0) of (E, L_z): closed form in the
        gauge V(pi/2) = 0, where p_v^2(pi/2) = 2 delta^2 [E + I_3] - L_z^2"""
        return Lz**2 / (2.0 * self._delta**2) - E

    def _I3_shell(self, E, Lz, ush):
        """I_3 of the shell orbit (J_R = 0) of (E, L_z): the value at which
        p_u^2 vanishes at its maximum"""
        sh2 = numpy.sinh(ush) ** 2
        return E * sh2 - float(self._U(ush)) - Lz**2 / (2.0 * self._delta**2 * sh2)

    # ---------- the grid's third-integral variable
    def _axis_reach(self, E, Lz, I3):
        """How closely the two librations of the (E, L_z, I_3) torus
        approach the axis, as one number that increases with I_3, and its
        partials with respect to (E, I_3, L_z): Lambda = Q_u + Q_v with
        Q_v = -2 ln sin v_min, which grows as the v libration reaches for
        the axis, and Q_u = 1 - sqrt(1 + rho^2), rho = 2 ln[tanh u_+ /
        tanh u_-], which falls as the inner u turning point does. At small
        |L_z| either the u libration is stopped just off the axis by the
        centrifugal barrier (I_3 below -U(0): sinh u_- ~ |L_z|, v_min
        finite) or the v libration is (I_3 above -V(0)), and between the
        two, around the L_z = 0 orbits through the foci, the turning
        points and with them every table of a torus change over a range of
        I_3 proportional to |L_z|; there Lambda is linear in I_3 / |L_z|
        and on either side logarithmic in the distance to it, so the
        tables are smooth functions of Lambda, uniformly in |L_z|. Both
        parts are smooth through the faces on which their libration is
        degenerate (Q_u is a function of rho^2, which is linear in I_3 at
        the shell face; Q_v is linear in I_3 at the planar face), where the
        partials are the harmonic limits, and both vanish toward the
        circular orbit. The v turning point is that of the Staeckel model
        of the planar orbit of (E, L_z) (the torus's own model, unless its
        reference curve follows the torus along I_3): within one model I_3
        only shifts the momentum function, so that Lambda increases with
        I_3 whatever the rule. Not for the circular orbit."""
        d2 = self._delta**2
        umin, umax, ush, udeg = self._u_turning(E, Lz, I3)
        self._reach_u = (umin, umax, ush, udeg, 0.0, 0.0)
        if udeg:
            sh2 = numpy.sinh(ush) ** 2
            K = (
                64.0
                * d2
                / (-0.5 * float(self._d2Wu(ush, E, Lz)) * numpy.sinh(2.0 * ush) ** 2)
            )
            dIshD = self._faces_ddelta(ush, Lz)[1]
            Qu, dQu = 0.0, K * numpy.array([-sh2, 1.0, Lz / (d2 * sh2), *(-dIshD)])
        else:
            rho = 2.0 * numpy.log(numpy.tanh(umax) / numpy.tanh(umin))
            dumin, dumax = (self._tp_derivs_u(u, E, Lz) for u in (umin, umax))
            drho = 4.0 * (
                dumax / numpy.sinh(2.0 * umax) - dumin / numpy.sinh(2.0 * umin)
            )
            self._reach_u = (umin, umax, ush, udeg, dumin[1], dumax[1])
            root = numpy.sqrt(1.0 + rho**2)
            Qu, dQu = -(rho**2) / (1.0 + root), -rho / root * drho
        with self._in_chart(E, Lz):
            Ipl, Ish = self._I3_planar(E, Lz), self._I3_shell(E, Lz, ush)
            if I3 <= Ipl + 1e-12 * (1.0 + numpy.fabs(Ipl) + numpy.fabs(Ish)):
                K = 2.0 * d2 / (-0.5 * float(self._d2Wv(0.5 * numpy.pi, E, Lz)))
                dIplD = self._faces_ddelta(ush, Lz)[0]
                Qv, dQv = 0.0, K * numpy.array([1.0, 1.0, -Lz / d2, *(-dIplD)])
            else:
                # while a column's nodes are placed, from its table and two
                # chord steps on p_v^2 (the second one negligible), else by
                # the bracketed solve
                vmin, dW, step = 0.0, 1.0, 1.0
                if self._reach_v[0] == (E, Lz):
                    vmin = float(self._reach_v[1](numpy.sqrt(max(I3 - Ipl, 0.0))))
                    dW = float(self._dWv(vmin, E, Lz))
                    for _ in range(2):
                        step = float(self._Wv(vmin, E, Lz, I3)) / dW
                        vmin -= step
                if not (0.0 < vmin < 0.5 * numpy.pi and numpy.fabs(step) < 1e-10):
                    vmin = self._v_turning(E, Lz, I3)
                    dW = float(self._dWv(vmin, E, Lz))
                Qv = -2.0 * numpy.log(numpy.sin(vmin))
                # the level-set rule on the turning point (_tp_derivs_v)
                sn2 = numpy.sin(vmin) ** 2
                dWD = self._dW_ddelta(vmin, Lz, lib="v")
                dvmin = -numpy.array([2.0 * d2 * sn2, 2.0 * d2, -2.0 * Lz / sn2, *dWD])
                dvmin = dvmin / dW
                if self._adaptive:
                    # the model of the planar orbit changes with (E, L_z)
                    # along its reference curve, and the turning point with it
                    g = self._u0_grad(E, numpy.fabs(Lz), Ipl)
                    dvmin = dvmin - 2.0 * d2 * float(
                        self._staeckelwrap._dVdu0(vmin)
                    ) / dW * numpy.array(
                        [
                            g[0] - g[1],
                            0.0,
                            (g[2] + g[1] * numpy.fabs(Lz) / d2) * numpy.sign(Lz),
                            *(g[3:] + g[1] * self._faces_ddelta(ush, Lz)[0]),
                        ]
                    )
                dQv = -2.0 / numpy.tan(vmin) * dvmin
        return Qu + Qv, dQu + dQv

    def _axis_reach_faces(self, E, Lz):
        """The I_3 of the planar and shell orbits of (E, L_z), with their
        derivatives with respect to (E, L_z), and the axis reach on each
        face with its derivatives along the face; cached"""
        key = (E, Lz)
        if key not in self._reach_cache:
            d2 = self._delta**2
            ush = self._ushell(E, Lz)
            sh2 = numpy.sinh(ush) ** 2
            Ipl, Ish = self._I3_planar(E, Lz), self._I3_shell(E, Lz, ush)
            dIplD, dIshD = self._faces_ddelta(ush, Lz)
            dIpl = numpy.array([-1.0, Lz / d2, *dIplD])
            dIsh = numpy.array([sh2, -Lz / (d2 * sh2), *dIshD])
            faces = []
            for I, dI in ((Ipl, dIpl), (Ish, dIsh)):
                lam, dlam = self._axis_reach(E, Lz, I)
                faces += [lam, numpy.delete(dlam, 1) + dlam[1] * dI]
            self._reach_cache[key] = (Ipl, dIpl, Ish, dIsh, *faces)
        return self._reach_cache[key]

    def _y_of_I3(self, E, Lz, I3):
        """The grid's third-integral variable y of the (E, L_z, I_3) torus
        and its partials with respect to (E, I_3, L_z): the fraction f =
        (I_3 - I_pl) / (I_sh - I_pl) of the way from the planar to the
        shell orbit blended with the axis reach (_axis_reach),

            y = [f + a (Lambda - Lambda_pl)] / [1 + a (Lambda_sh - Lambda_pl)]

        with the weight a = axis_weight: zero on the planar face, one on
        the shell face, increasing with I_3 and linear in it at both
        faces. For the tori that stay away from the axis (those of a
        disc) y is close to f; at small |L_z| it spends its nodes evenly
        in the logarithm of the turning points' distances to the axis,
        which is what resolves the tables there."""
        Ipl, dIpl, Ish, dIsh, lpl, dlpl, lsh, dlsh = self._axis_reach_faces(E, Lz)
        a, DI = self._axisw, Ish - Ipl
        f = (I3 - Ipl) / DI
        # (with a fourth partial, along the focal length, for a family
        # whose focal length follows its tori: _dW_ddelta)
        df = numpy.insert(-(dIpl + f * (dIsh - dIpl)), 1, 1.0) / DI
        lam, dlam = self._axis_reach(E, Lz, I3)
        D = 1.0 + a * (lsh - lpl)
        y = (f + a * (lam - lpl)) / D
        dN = df + a * (dlam - numpy.insert(dlpl, 1, 0.0))
        dD = a * numpy.insert(dlsh - dlpl, 1, 0.0)
        return y, (dN - y * dD) / D

    def _I3_of_y(self, E, Lz, y, start=None, node=False):
        """The I_3 of the torus at the grid's third-integral variable y at
        (E, L_z), with (dI_3/dE, dI_3/dy, dI_3/dL_z) at fixed grid
        coordinates: the inverse of _y_of_I3, by Newton steps kept inside
        a bracket, from a first guess of I_3 when one is given; a residual
        below 1e-4 is removed by its Newton step without another
        evaluation (what is left is of second order, below 1e-7 of the
        variable's range, far within the family's interpolation error). For a node of the
        grid, the u turning points at the returned I_3 are remembered for
        the torus's sampling: those of the last evaluation, moved to the
        returned I_3 along their slopes (and by a Newton step on the
        momentum function, unless the move is negligible)."""
        Ipl, _, Ish, _ = self._axis_reach_faces(E, Lz)[:4]
        lo, hi = 0.0, 1.0
        f = min(max(y if start is None else (start - Ipl) / (Ish - Ipl), lo), hi)
        if y <= 0.0 or y >= 1.0:
            # a face is the face exactly
            f = min(max(y, 0.0), 1.0)
        for _ in range(100):
            fe = f
            yf, dy = self._y_of_I3(E, Lz, Ipl + f * (Ish - Ipl))
            if numpy.fabs(yf - y) < 1e-13 or hi - lo < 4e-16:
                break
            lo, hi = (f, hi) if yf < y else (lo, f)
            fn = f - (yf - y) / (dy[1] * (Ish - Ipl))
            f = fn if lo < fn < hi else 0.5 * (lo + hi)
            if f == fn and numpy.fabs(yf - y) < 1e-4:
                break
        I3 = Ipl + f * (Ish - Ipl)
        umin, umax, ush, udeg, dumin, dumax = self._reach_u
        if node and f != fe and not udeg:
            # along the turning points' own slopes, and by a Newton step
            # where the residual was not already negligible
            umin, umax = (
                umin + dumin * (f - fe) * (Ish - Ipl),
                umax + dumax * (f - fe) * (Ish - Ipl),
            )
            for _ in range(int(numpy.fabs(yf - y) > 1e-9)):
                umin -= float(self._Wu(umin, E, Lz, I3) / self._dWu(umin, E, Lz))
                umax -= float(self._Wu(umax, E, Lz, I3) / self._dWu(umax, E, Lz))
            self._u_turning_memo[(E, Lz, I3)] = (umin, umax, ush, udeg)
        return I3, numpy.insert(-numpy.delete(dy, 1), 1, 1.0) / dy[1]

    def _I3_starts(self, E, Lz, ys):
        """First guesses of the I_3 at the third-integral variable's values
        ys at (E, L_z), from the two momentum functions tabulated once: I_3
        only shifts them, so the u at which p_u^2 vanishes on the way in
        from the shell u gives I_3 in closed form, and the outer u turning
        point and the v turning point at that I_3 are read from the other
        two branches, each interpolated in the square root of the distance
        to the face on which it degenerates (in the Staeckel model that is
        current: that of the planar orbit of (E, L_z), as for _axis_reach)"""
        Ipl, _, Ish, _, lpl, _, lsh, _ = self._axis_reach_faces(E, Lz)
        ush, d2 = self._ushell(E, Lz), 2.0 * self._delta**2
        umin, umax, _, _ = self._u_turning(E, Lz, Ipl)
        vsh = self._v_turning(E, Lz, Ish)

        def increasing(x):
            # strictly, whatever the round-off of the potential
            x = numpy.maximum.accumulate(x)
            return x + 1e-15 * (1.0 + x[-1]) * numpy.arange(len(x))

        n = 64
        ui = numpy.geomspace(umin, ush, n)[1:-1]
        I3 = increasing(self._Wu(ui, E, Lz, 0.0) / d2)
        uo = numpy.geomspace(ush, umax, n)
        qo = numpy.sqrt(numpy.clip(Ish - self._Wu(uo, E, Lz, 0.0) / d2, 0.0, None))
        # (each read inside its table, whatever the round-off)
        up = CubicSpline(increasing(qo), uo)(
            numpy.clip(numpy.sqrt(numpy.clip(Ish - I3, 0.0, None)), 0.0, qo[-1])
        )
        vs = numpy.geomspace(0.5 * numpy.pi, vsh, n)
        pv = numpy.sqrt(numpy.clip(-self._Wv(vs, E, Lz, 0.0) / d2 - Ipl, 0.0, None))
        # (the v branch is kept for the exact evaluations of this column:
        # _axis_reach)
        self._reach_v = ((E, Lz), CubicSpline(increasing(pv), vs))
        vmin = self._reach_v[1](
            numpy.clip(numpy.sqrt(numpy.clip(I3 - Ipl, 0.0, None)), 0.0, pv[-1])
        )
        rho = 2.0 * numpy.log(numpy.tanh(up) / numpy.tanh(ui))
        lam = 1.0 - numpy.sqrt(1.0 + rho**2) - 2.0 * numpy.log(numpy.sin(vmin))
        yk = ((I3 - Ipl) / (Ish - Ipl) + self._axisw * (lam - lpl)) / (
            1.0 + self._axisw * (lsh - lpl)
        )
        return CubicSpline(
            increasing(numpy.concatenate(([0.0], yk, [1.0]))),
            numpy.concatenate(([Ipl], I3, [Ish])),
        )(ys)

    def _u_turning(self, E, Lz, I3):
        """The inner and outer turning points of the u libration of the
        (E, L_z, I_3) torus, the shell u, and whether the libration is
        degenerate; none of them depends on the reference curve, because
        U(u) does not. Raises for labels that do not define a torus. They
        are remembered, for every torus while an instance is set up and for
        the last one afterwards: a rule for the reference curve, the
        turning points, the sampling and the rule's derivatives ask for
        the same torus's in turn"""
        key = (E, Lz, I3)
        if key not in self._u_turning_memo:
            if not self._memo_keep:
                self._u_turning_memo.clear()
            self._u_turning_memo[key] = self._u_turning_solve(E, Lz, I3)
        return self._u_turning_memo[key]

    def _u_turning_solve(self, E, Lz, I3):
        ell = numpy.fabs(Lz)
        _, Ec, uc = self._circular_RE(ell)
        tolE = 1e-12 * (1.0 + numpy.fabs(Ec))
        if E < Ec - tolE:
            raise ValueError(
                f"No orbit exists at E = {E}, L_z = {Lz}: the energy lies below "
                "the circular orbit's"
            )
        if E < Ec + tolE:
            # the circular orbit
            return uc, uc, uc, True
        ush = self._ushell(E, Lz)
        Ipl, Ish = self._I3_planar(E, Lz), self._I3_shell(E, Lz, ush)
        tolI = 1e-12 * (1.0 + numpy.fabs(Ipl) + numpy.fabs(Ish))
        if I3 > Ish + tolI:
            raise ValueError(
                f"No orbit exists at (E, L_z, I_3) = ({E}, {Lz}, {I3}): I_3 "
                f"lies above the shell orbit's, {Ish}"
            )
        if I3 < Ipl - tolI:
            raise ValueError(
                f"No orbit exists at (E, L_z, I_3) = ({E}, {Lz}, {I3}): I_3 "
                f"lies below the planar orbit's, {Ipl}"
            )
        if I3 >= Ish - tolI:
            return ush, ush, ush, True
        Wu = lambda u: float(self._Wu(u, E, Lz, I3))
        # the libration around the shell u ends, on either side, at a
        # barrier of p_u^2 that this I_3 leaves below zero (the tori beyond
        # it are tori of their own), and else beyond the barrier, where the
        # two have merged
        ends = []
        for bar, fac in zip(self._ubar_cache[(E, Lz)], (0.7, 1.3)):
            end = ush if bar is None else bar[0]
            while Wu(end) > 0.0:
                end *= fac
            ends.append(brentq(Wu, *sorted((end, ush)), xtol=1e-14))
        return (ends[0], ends[1], ush, False)

    def _v_turning(self, E, Lz, I3):
        """The turning point of the v libration about the midplane of the
        (E, L_z, I_3) torus (not a planar orbit) in the current Staeckel
        model"""
        Wv = lambda v: float(self._Wv(v, E, Lz, I3))
        # the libration about the midplane ends at the first root below
        # pi/2, bracketed by the first change of sign on a scan down
        # from the midplane. In a Staeckel potential p_v^2 falls
        # monotonically from the midplane to that root; in a Staeckel
        # approximation whose foci attract (V(0) > U(0), as near the
        # centre of a cusped potential) it can rise again toward the
        # axis, where a second torus shares the integrals, until I_3
        # lifts the barrier between the two above zero and they merge
        # into one, across which the tori are not a continuous family:
        # no grid can hold them
        vs = 0.5 * numpy.pi * (1.0 - numpy.linspace(0.0, 1.0, 49)[:-1])
        Ws = self._Wv(vs, E, Lz, I3)
        neg = numpy.flatnonzero(Ws <= 0.0)
        npos = neg[0] if len(neg) > 0 else len(vs)
        if self._interp and numpy.any(numpy.diff(Ws[:npos]) > 1e-10 * Ws[0]):
            raise ValueError(
                "The momentum function of the v libration of the (E, L_z, I_3) "
                f"= ({E}, {Lz}, {I3}) torus does not fall monotonically from "
                "the midplane to its turning point: in this Staeckel "
                "approximation the tori at these integrals do not continue "
                "those about the midplane (near its foci, which attract as "
                "they do at the centre of a cusped potential, the "
                "approximation holds tori of its own), and an interpolation "
                "grid cannot hold them. Keep the grid away from the axis (a "
                "larger Rmin) or change delta or u0"
            )
        if len(neg) > 0:
            vlo, vhi = vs[npos], vs[npos - 1]
        else:
            vlo = vhi = vs[-1]
            while Wv(vlo) > 0.0:
                vlo *= 0.7
        return brentq(Wv, vlo, vhi, xtol=1e-14)

    def _turning_points(self, E, Lz, I3):
        """The turning points of the (E, L_z, I_3) torus as the midpoint and
        half-width of the u libration and the half-width of the v libration
        (about the midplane), with the shell u and whether either libration
        is degenerate; raises for labels that do not define a torus"""
        umin, umax, ush, udeg = self._u_turning(E, Lz, I3)
        if self._on_circular(E, numpy.fabs(Lz)):
            # the circular orbit: both librations degenerate
            return umin, 0.0, 0.0, umin, True, True
        Ipl = self._I3_planar(E, Lz)
        Ish = self._I3_shell(E, Lz, ush)
        tolI = 1e-12 * (1.0 + numpy.fabs(Ipl) + numpy.fabs(Ish))
        vdeg = I3 <= Ipl + tolI
        if vdeg:
            vmin = 0.5 * numpy.pi
        else:
            vmin = self._v_turning(E, Lz, I3)
        return (
            0.5 * (umin + umax),
            0.5 * (umax - umin),
            0.5 * numpy.pi - vmin,
            ush,
            udeg,
            vdeg,
        )

    # ---------- node construction: exact tori by quadrature
    def _sample_torus(self, E, Lz, I3, tp=None):
        """Exact phase-space samples along both librations of the
        (E, L_z, I_3) torus, between its turning points (tp, as
        _turning_points returns them, when they are known), each
        parametrized by its tau anomaly on the offset grid tau_k =
        2 pi (k + 1/2)/N, which keeps every sample off the turning points;
        placement is exact by construction (the momenta from the separated
        Hamilton-Jacobi equation, not from a fit). A degenerate libration
        (a shell orbit's in u, a planar orbit's in v) has no samples."""
        uc, wu, wv, ush, udeg, vdeg = (
            self._turning_points(E, Lz, I3) if tp is None else tp
        )
        tau = 2.0 * numpy.pi * (numpy.arange(self._ntau) + 0.5) / self._ntau
        sgn = numpy.where(tau < numpy.pi, 1.0, -1.0)
        smp = {
            "tau": tau,
            "E": E,
            "Lz": Lz,
            "I3": I3,
            "uc": uc,
            "wu": wu,
            "wv": wv,
            "ush": ush,
            "u": None,
            "pu": None,
            "jr": 0.0,
            "v": None,
            "pv": None,
            "jz": 0.0,
        }
        if not udeg:
            u = uc - wu * numpy.cos(tau)
            pu = sgn * numpy.sqrt(numpy.clip(self._Wu(u, E, Lz, I3), 0.0, None))
            smp["u"], smp["pu"] = u, pu
            smp["jr"] = float(numpy.mean(pu * wu * numpy.sin(tau)))
        if not vdeg:
            v = 0.5 * numpy.pi - wv * numpy.cos(tau)
            pv = sgn * numpy.sqrt(numpy.clip(self._Wv(v, E, Lz, I3), 0.0, None))
            smp["v"], smp["pv"] = v, pv
            smp["jz"] = float(numpy.mean(pv * wv * numpy.sin(tau)))
        return smp

    # ---------- many tori at once: the momentum functions from tables, the
    # librations, and the tori of given actions
    def _midplane(self):
        """The potential in the midplane, on which U(u) rests, tabulated
        once with its logarithmic derivative on a uniform grid of ln R (a
        cubic Hermite interpolant with exact slopes, accurate to about
        1e-12 of the potential's scale) over the radii at which it
        evaluates; and the scan of ln R on which the extrema of p_u^2 are
        first located, with the table's values and slopes on it"""
        if self._midplane_tab is None:
            d = self._delta
            lnR = numpy.linspace(
                numpy.log(1e-8 * d), numpy.log(d * numpy.sinh(20.0)), 12001
            )
            R = numpy.exp(lnR)
            with numpy.errstate(all="ignore"):
                Phi = evaluatePotentials(
                    self._chart_pot, R, 0.0 * R, use_physical=False
                )
                dPhi = -R * evaluateRforces(
                    self._chart_pot, R, 0.0 * R, use_physical=False
                )
            # up to the first radius at which the potential does not evaluate
            n = int(numpy.argmin(numpy.append(numpy.isfinite(Phi + dPhi), False)))
            tab = _hermite_table(lnR[:n], Phi[:n], dPhi[:n])
            scan = numpy.linspace(numpy.log(1e-6 * d), lnR[n - 1], 1000)
            self._midplane_tab = (tab, (scan, *_hermite_uniform(tab, scan, deriv=True)))
        return self._midplane_tab

    def _W0(self, lnR, E, ell, deriv=False):
        """p_u^2 + 2 delta^2 I_3 for tori of energies E and angular momenta
        ell where the u spheroid crosses the midplane at ln R (R = delta
        sinh u), from the tabulated midplane potential: 2 E R^2 - 2
        (delta^2 + R^2) Phi(R) - L_z^2 delta^2 / R^2; with deriv, its
        derivative with respect to ln R instead"""
        d2, R2 = self._delta**2, numpy.exp(2.0 * lnR)
        if not deriv:
            return (
                2.0 * R2 * E
                - 2.0 * (d2 + R2) * _hermite_uniform(self._midplane()[0], lnR)
                - ell**2 * d2 / R2
            )
        Phi, dPhi = _hermite_uniform(self._midplane()[0], lnR, deriv=True)
        return 4.0 * R2 * (E - Phi) - 2.0 * (d2 + R2) * dPhi + 2.0 * ell**2 * d2 / R2

    def _wells(self, E, ell):
        """The extrema of p_u^2 for tori of energies E and angular momenta
        ell (I_3 only shifts it): the ln R (R = delta sinh u) of its
        outermost maximum, of the barrier next to that on the inside and of
        the maximum inside the barrier (NaN where p_u^2 has a single
        maximum), and p_u^2 + 2 delta^2 I_3 at the three. In the Staeckel
        approximation of a cusped potential the three exist at small
        |L_z|: below the barrier's level one libration spans both wells,
        above it the libration about the outer maximum and that about the
        inner one coexist at the same integrals. Each extremum is located
        on a fixed scan and refined by sectioning."""
        xs, Phi, dPhi = self._midplane()[1]
        d2, R2 = self._delta**2, numpy.exp(2.0 * xs)
        pos = (
            4.0 * R2 * (E[:, None] - Phi)
            - 2.0 * (d2 + R2) * dPhi
            + 2.0 * ell[:, None] ** 2 * d2 / R2
        ) > 0.0
        down, up = pos[:, :-1] & ~pos[:, 1:], ~pos[:, :-1] & pos[:, 1:]
        k = numpy.arange(len(xs) - 1)
        ko = numpy.amax(numpy.where(down, k, -1), axis=1)
        kb = numpy.amax(numpy.where(up & (k < ko[:, None]), k, -1), axis=1)
        ki = numpy.amax(numpy.where(down & (k < kb[:, None]), k, -1), axis=1)
        # (a barrier counts with the maximum inside it)
        kb = numpy.where(ki >= 0, kb, -1)
        # (the three in one pass)
        kk = numpy.concatenate([ko, kb, ki])
        sgn = numpy.repeat([-1.0, 1.0, -1.0], len(E))[:, None]
        E3, l3 = numpy.tile(E, 3), numpy.tile(ell, 3)
        x = _multisect(
            lambda x: sgn * self._W0(x, E3[:, None], l3[:, None], deriv=True),
            xs[numpy.maximum(kk, 0)],
            xs[numpy.maximum(kk, 0) + 1],
            36,
        )
        x = numpy.where(kk >= 0, x, numpy.nan)
        with numpy.errstate(invalid="ignore"):
            return (*x.reshape(3, -1), *self._W0(x, E3, l3).reshape(3, -1))

    def _circular_many(self, ell):
        """The energies and the epicycle and vertical frequencies of the
        circular orbits of many angular momenta: the radius bracketed on
        the tabulated midplane potential and solved on the potential
        itself"""
        lnR, _, dPhi = self._midplane()[1]
        pot = self._chart_pot

        def dphi(R):
            return -evaluateRforces(
                pot, R.ravel(), 0.0 * R.ravel(), use_physical=False
            ).reshape(R.shape)

        L2 = numpy.maximum.accumulate(numpy.exp(2.0 * lnR) * dPhi)
        k = numpy.clip(numpy.searchsorted(L2, ell**2), 1, len(L2) - 1)
        # (a point of the scan further on either side: the table's and the
        # potential's own slopes differ in the last digits)
        Rc = numpy.exp(
            _multisect(
                lambda x: numpy.exp(3.0 * x) * dphi(numpy.exp(x)) - ell[:, None] ** 2,
                lnR[numpy.maximum(k - 2, 0)],
                lnR[numpy.minimum(k + 1, len(L2) - 1)],
                44,
            )
        )
        kappa = numpy.sqrt(
            evaluateR2derivs(pot, Rc, 0.0 * Rc, use_physical=False)
            + 3.0 * dphi(Rc) / Rc
        )
        with numpy.errstate(invalid="ignore"):
            nu = numpy.sqrt(evaluatez2derivs(pot, Rc, 0.0 * Rc, use_physical=False))
        return (
            evaluatePotentials(pot, Rc, 0.0 * Rc, use_physical=False)
            + ell**2 / (2.0 * Rc**2),
            kappa,
            numpy.where(numpy.isfinite(nu), nu, kappa),
        )

    def _V_many(self, v, u0, deriv=False):
        """V(v) for many tori, each on the reference curve u0 that its rule
        gives it (from the potential on that spheroid; with deriv, also
        dV/du0), or all on the family's single curve (u0 is None: V is
        tabulated once on [0, pi/2], about which it is even)"""
        if u0 is None:
            if self._Vref_tab is None:
                vs = numpy.linspace(1e-10, 0.5 * numpy.pi, 4001)
                self._Vref_tab = _hermite_table(vs, self._V(vs), self._dV(vs))
            return _hermite_uniform(self._Vref_tab, v)
        d = self._delta
        sh, ch = numpy.sinh(u0), numpy.cosh(u0)
        sn, cs = numpy.sin(v), numpy.cos(v)
        R, z, pre = d * sh * sn, d * ch * cs, sh**2 + sn**2
        # (the midplane radius of each torus's spheroid: once per torus)
        R0 = d * sh

        def at(func, R, z):
            # (the potentials take flat arrays)
            return func(
                self._chart_pot, R.ravel(), z.ravel(), use_physical=False
            ).reshape(R.shape)

        Phi, Phi0 = at(evaluatePotentials, R, z), at(evaluatePotentials, R0, 0.0 * R0)
        V = ch**2 * Phi0 - pre * Phi
        if not deriv:
            return V
        return V, (
            2.0 * sh * ch * (Phi0 - Phi)
            - ch**3 * d * at(evaluateRforces, R0, 0.0 * R0)
            + pre
            * d
            * (
                at(evaluateRforces, R, z) * ch * sn
                + at(evaluatezforces, R, z) * sh * cs
            )
        )

    def _u0_many(self, E, ell, I3, um, up, uref, udeg, valid, wells):
        """The reference u0 that the rule gives each of many tori, with its
        partials in (E, I_3) (None and zeros without a rule). The built-in
        rules are vectorized: 'torus' is the closed form of _u0_torus on
        the torus's own turning points, 'planar' the same on those of the
        planar orbit of (E, L_z), 'shell' the u of the highest maximum of
        p_u^2; a callable, which need not take arrays, is evaluated torus by
        torus, with one-sided differences"""
        if not self._adaptive:
            return None, 0.0, 0.0
        d, d2 = self._delta, self._delta**2

        def dWdu(u):
            return self._W0(numpy.log(d * numpy.sinh(u)), E, ell, deriv=True) / (
                numpy.tanh(u)
            )

        def dushell(u):
            # the level-set rule on the maximum of p_u^2
            h = 1e-5 * u
            return -4.0 * d2 * numpy.sinh(2.0 * u) * h / (dWdu(u + h) - dWdu(u - h))

        def mean(um, up, udeg, uref):
            # the harmonic mean of the radii of the two turning points, and
            # the level-set rule on them; on the shell face the curve is the
            # shell orbit's own spheroid
            Rm, Rp = d * numpy.sinh(um), d * numpy.sinh(up)
            u0 = numpy.arcsinh(2.0 * Rm * Rp / (Rm + Rp) / d)
            wm = 2.0 * d * Rp**2 * numpy.cosh(um) / (Rm + Rp) ** 2 / dWdu(um)
            wp = 2.0 * d * Rm**2 * numpy.cosh(up) / (Rm + Rp) ** 2 / dWdu(up)
            scale = d * numpy.cosh(u0)
            return (
                u0,
                numpy.where(
                    udeg,
                    dushell(uref),
                    -2.0
                    * d2
                    * (numpy.sinh(um) ** 2 * wm + numpy.sinh(up) ** 2 * wp)
                    / scale,
                ),
                numpy.where(udeg, 0.0, 2.0 * d2 * (wm + wp) / scale),
            )

        with numpy.errstate(divide="ignore", invalid="ignore"):
            if self._u0_rule == "torus":
                return mean(um, up, udeg, uref)
            if self._u0_rule == "planar":
                pm, pp, _, _, _, ok = self._u_ends(
                    E, ell, ell**2 / (2.0 * d2) - E, wells, False
                )
                u0, gE, gI = mean(pm, pp, ~ok, uref)
                return numpy.where(ok, u0, numpy.nan), gE - gI, 0.0 * gI
            if self._u0_rule == "shell":
                inner = numpy.isfinite(wells[5]) & (wells[5] > wells[3])
                u0 = numpy.arcsinh(
                    numpy.exp(numpy.where(inner, wells[2], wells[0])) / d
                )
                # (refined on the potential itself, on which the torus is
                # set up)
                u0 = u0 - self._dWu(u0, E, ell) / self._d2Wu(u0, E, ell)
                return u0, dushell(u0), 0.0 * u0
        out = numpy.full((3, len(E)), numpy.nan)
        for ii in numpy.flatnonzero(valid):
            Ei, li, Ii = E[ii], ell[ii], I3[ii]
            hE, hI = 1e-6 * (1.0 + numpy.fabs(Ei)), 1e-6 * (1.0 + numpy.fabs(Ii))
            out[0, ii] = self._u0_of(Ei, li, Ii)
            out[1, ii] = (self._u0_of(Ei + hE, li, Ii) - out[0, ii]) / hE
            out[2, ii] = (self._u0_of(Ei, li, Ii + hI) - out[0, ii]) / hI
        return tuple(out)

    def _u_ends(self, E, ell, I3, wells, inner):
        """The u librations of many tori from their integrals: the turning
        points of the libration about the outermost maximum of p_u^2 (it
        ends at a barrier that I_3 leaves below zero, and spans both wells
        where I_3 lifts the barrier above zero) or, where inner is set and
        I_3 leaves the barrier below zero, of the libration about the
        maximum inside the barrier; the u of the libration's maximum,
        whether the libration is degenerate, the side of the barrier's
        level on which the torus lies (1 above, -1 below, 0 where there is
        no barrier), and whether the integrals are those of a torus"""
        d2 = self._delta**2
        xo, xb, xi, Wo, Wb, Wi = wells
        c = 2.0 * d2 * I3
        split = numpy.isfinite(xb)
        with numpy.errstate(invalid="ignore"):
            above = split & (c > Wb)
            own = above & inner
            xref, Wref = numpy.where(own, xi, xo), numpy.where(own, Wi, Wo)
            Ipl, Ish = ell**2 / (2.0 * d2) - E, Wref / (2.0 * d2)
            tolI = 1e-12 * (1.0 + numpy.fabs(Ipl) + numpy.fabs(Ish))
            udeg = I3 >= Ish - tolI
            valid = (I3 <= Ish + tolI) & (I3 >= Ipl - tolI)
        # the inner and the outer turning point in one pass
        tab, n = self._midplane()[0], len(E)
        lo = numpy.concatenate(
            [numpy.where(above & ~own, xb, tab[0] + tab[1]), numpy.where(own, xi, xo)]
        )
        hi = numpy.concatenate(
            [
                numpy.where(split & ~(above & ~own), xi, xo),
                numpy.where(own, xb, tab[0] + tab[1] * (tab[2].shape[1] - 1)),
            ]
        )
        sgn = numpy.repeat([1.0, -1.0], n)
        E2, l2, c2, deg = (numpy.tile(q, 2) for q in (E, ell, c, udeg))
        lo = numpy.where(numpy.tile(valid, 2), lo, 0.0)
        hi = numpy.where(numpy.tile(valid, 2), hi, 1.0)
        ok = deg | (
            (sgn * (self._W0(lo, E2, l2) - c2) < 0.0)
            & (sgn * (self._W0(hi, E2, l2) - c2) >= 0.0)
        )
        x = _multisect(
            lambda x: (
                sgn[:, None] * (self._W0(x, E2[:, None], l2[:, None]) - c2[:, None])
            ),
            lo,
            hi,
            48,
        )
        um, up = numpy.arcsinh(
            numpy.exp(numpy.where(deg, numpy.tile(xref, 2), x)) / self._delta
        ).reshape(2, -1)
        return (
            um,
            up,
            numpy.arcsinh(numpy.exp(xref) / self._delta),
            udeg,
            numpy.where(split, numpy.where(above, 1, -1), 0),
            valid & ok[:n] & ok[n:],
        )

    def _tori_many(self, E, ell, I3, wells, inner, ntau):
        """Many tori from their integrals, without a loop over them: the
        turning points of both librations, and the actions and their
        derivatives with respect to (E, I_3) by the quadratures of
        _sample_torus on ntau anomalies (on the half and the quarter of
        them that the symmetries of the librations leave), with the
        variation of the Staeckel model along a rule for the reference
        curve. Returns the
        actions (ntori, 2), their derivatives (ntori, 2, 2), and the rows
        of: the u turning points, the u of the libration's maximum,
        whether the u and the v libration are degenerate, the side of a
        barrier's level on which the torus lies (_u_ends), the v turning
        point, and whether the integrals are those of a torus"""
        d2 = self._delta**2
        um, up, uref, udeg, side, valid = self._u_ends(E, ell, I3, wells, inner)
        c = 2.0 * d2 * I3
        with numpy.errstate(invalid="ignore"):
            vdeg = I3 <= ell**2 / (2.0 * d2) - E + 1e-12 * (
                1.0 + numpy.fabs(ell**2 / (2.0 * d2) - E) + numpy.fabs(c / d2)
            )

        u0, gE, gI = self._u0_many(E, ell, I3, um, up, uref, udeg, valid, wells)
        if u0 is not None:
            valid &= numpy.isfinite(u0)
            u0 = numpy.where(valid, u0, 1.0)[:, None]

        def Wv(v, deriv=False):
            sn2 = numpy.sin(v) ** 2
            V = self._V_many(v, u0, deriv=deriv)
            V, dV = V if deriv else (V, 0.0)
            return (
                2.0 * d2 * (E[:, None] * sn2 + V + I3[:, None])
                - ell[:, None] ** 2 / sn2,
                dV,
            )

        # the v libration about the midplane ends at the first root below
        # it (_v_turning)
        vs = 0.5 * numpy.pi * (1.0 - numpy.linspace(0.0, 1.0, 49)[:-1])
        with numpy.errstate(invalid="ignore"):
            neg = ~(Wv(vs[None, :] + 0.0 * E[:, None])[0] > 0.0)
        first = numpy.where(numpy.any(neg, axis=1), numpy.argmax(neg, axis=1), len(vs))
        vmin = numpy.exp(
            _multisect(
                lambda x: Wv(numpy.exp(x))[0],
                numpy.log(numpy.append(vs, 1e-9)[first]),
                numpy.log(vs[numpy.maximum(first - 1, 0)]),
                48,
            )
        )
        vmin = numpy.where(vdeg, 0.5 * numpy.pi, vmin)
        # the actions and their derivatives: the mean over the anomalies of
        # a libration is that over the first half of them (a libration is
        # even about its turning points) or the first quarter (the v
        # libration also about the midplane)
        tau = 2.0 * numpy.pi * (numpy.arange(ntau) + 0.5) / ntau
        J, M = numpy.zeros((len(E), 2)), numpy.zeros((len(E), 2, 2))
        with numpy.errstate(divide="ignore", invalid="ignore"):
            t = tau[: ntau // 2]
            wu = 0.5 * (up - um)[:, None]
            u = 0.5 * (up + um)[:, None] - wu * numpy.cos(t)
            pu = numpy.sqrt(
                numpy.clip(
                    self._W0(
                        numpy.log(self._delta * numpy.sinh(u)), E[:, None], ell[:, None]
                    )
                    - c[:, None],
                    0.0,
                    None,
                )
            )
            ratio = wu * numpy.sin(t) / pu
            J[:, 0] = numpy.mean(pu * wu * numpy.sin(t), axis=1)
            M[:, 0, 0] = numpy.mean(d2 * numpy.sinh(u) ** 2 * ratio, axis=1)
            M[:, 0, 1] = -d2 * numpy.mean(ratio, axis=1)
            t = tau[: ntau // 4] if ntau % 4 == 0 else t
            wv = (0.5 * numpy.pi - vmin)[:, None]
            v = 0.5 * numpy.pi - wv * numpy.cos(t)
            W, dV = Wv(v, deriv=u0 is not None)
            pv = numpy.sqrt(numpy.clip(W, 0.0, None))
            ratio = wv * numpy.sin(t) / pv
            J[:, 1] = numpy.mean(pv * wv * numpy.sin(t), axis=1)
            M[:, 1, 0] = numpy.mean(d2 * numpy.sin(v) ** 2 * ratio, axis=1)
            M[:, 1, 1] = d2 * numpy.mean(ratio, axis=1)
            # the model follows the torus along its rule
            Ju0 = d2 * numpy.mean(dV * ratio, axis=1)
            M[:, 1, 0] += Ju0 * gE
            M[:, 1, 1] += Ju0 * gI
        return J, M, numpy.array([um, up, uref, udeg, vdeg, side, vmin, valid])

    def _action_residual(self, E, f, ell, jr, jz, inner, ntau=256, side=None):
        """The residual of the solve for the integrals of many tori from
        their actions, apart from its iteration: at the energies E and the
        fractions f of the way from the planar orbit's I_3 to that of the
        shell orbit (the outermost maximum of p_u^2 or, where inner is
        set, the maximum inside a barrier), the actions of the tori minus
        (J_R, J_z), relative to J_R + J_z; with the derivatives of the
        actions with respect to (E, f), the I_3, the f and the rows of
        _tori_many. With side (1 or -1), f is instead the fraction of the
        way across the tori on that side of a barrier's level: from it to
        the shell orbit, or from the planar orbit to it."""
        d2 = 2.0 * self._delta**2
        wells = self._wells(E, ell)
        own = inner & numpy.isfinite(wells[1])
        Ipl = ell**2 / d2 - E
        with numpy.errstate(invalid="ignore"):
            Ish = numpy.where(own, wells[5], wells[3]) / d2
            if side is not None:
                Ibar = numpy.clip(wells[4] / d2, Ipl, Ish)
                lo = numpy.where(numpy.isfinite(wells[1]) & (side == 1), Ibar, Ipl)
                hi = numpy.where(numpy.isfinite(wells[1]) & (side == -1), Ibar, Ish)
                f = (lo + f * (hi - lo) - Ipl) / (Ish - Ipl)
            I3 = Ipl + f * (Ish - Ipl)
            J, M, rows = self._tori_many(E, ell, I3, wells, inner, ntau)
            ch2 = 1.0 + 2.0 * numpy.exp(2.0 * numpy.where(own, wells[2], wells[0])) / d2
            D = numpy.stack(
                [
                    M[:, :, 0] + M[:, :, 1] * (f * ch2 - 1.0)[:, None],
                    M[:, :, 1] * (Ish - Ipl)[:, None],
                ],
                axis=2,
            )
            res = (J - numpy.stack([jr, jz], axis=1)) / (jr + jz)[:, None]
            rows[-1] = rows[-1].astype("bool") & (Ish > Ipl)
        return res, D, I3, f, rows

    def _newton_many(self, E, f, ell, jr, jz, Ec, hold, inner, side, ntau, steps):
        """Damped Newton iteration on _action_residual for many tori from
        the starting points (E, f), on a fixed schedule: every torus has
        steps trial points, each either accepted (it is a torus, on the
        side of a barrier's level that side asks for when that is not
        zero, and it reduces the residual: the torus moves there and takes
        a new Newton step from it) or rejected (the torus stays and halves
        its step), and a torus is frozen once its residual is below the
        tolerance or its step has been halved below 1e-8 of a Newton step;
        the tori that are frozen are not evaluated any more. hold is 1 for
        the tori on the shell face and 2 for those on the planar face,
        whose f is held and whose energy is solved from the other action.
        Returns the energies, the fractions, the I_3, the rows of
        _tori_many at the last accepted points, and which tori
        converged."""
        n = len(E)
        E, f, Et, ft = E.copy(), f.copy(), E.copy(), f.copy()
        I3, norm, t = numpy.full(n, numpy.nan), numpy.full(n, numpy.inf), numpy.ones(n)
        tol, dE, df = numpy.full(n, 1e-11), numpy.zeros(n), numpy.zeros(n)
        rows = numpy.full((8, n), numpy.nan)
        a = numpy.arange(n)
        for _ in range(steps):
            res, D, I3t, _, rowst = self._action_residual(
                Et[a], ft[a], ell[a], jr[a], jz[a], inner[a], ntau=ntau
            )
            h = hold[a]
            # a torus held on a face has that face's libration degenerate;
            # any other torus has neither
            valid = (
                (rowst[-1] == 1.0)
                & (rowst[3] == (h == 1))
                & (rowst[4] == (h == 2))
                & ((side[a] == 0) | (rowst[5] == 0) | (rowst[5] == side[a]))
            )
            with numpy.errstate(divide="ignore", invalid="ignore"):
                r0, r1, tot = res[:, 0], res[:, 1], (jr + jz)[a]
                new = numpy.where(
                    h == 0,
                    numpy.maximum(numpy.fabs(r0), numpy.fabs(r1)),
                    numpy.where(h == 1, numpy.fabs(r1), numpy.fabs(r0)),
                )
                det = D[:, 0, 0] * D[:, 1, 1] - D[:, 0, 1] * D[:, 1, 0]
                sE = tot * numpy.where(
                    h == 0,
                    -(D[:, 1, 1] * r0 - D[:, 0, 1] * r1) / det,
                    numpy.where(h == 1, -r1 / D[:, 1, 0], -r0 / D[:, 0, 0]),
                )
                sf = tot * numpy.where(
                    h == 0, (D[:, 1, 0] * r0 - D[:, 0, 0] * r1) / det, 0.0
                )
                acc = valid & (new < norm[a]) & numpy.isfinite(sE) & numpy.isfinite(sf)
            g, r = a[acc], a[~acc]
            E[g], f[g], I3[g], norm[g] = Et[g], ft[g], I3t[acc], new[acc]
            dE[g], df[g], t[g], rows[:, g] = sE[acc], sf[acc], 1.0, rowst[:, acc]
            # (close to the circular orbit the energy above it, and with it
            # the actions, are only known to the round-off of the energy)
            tol[g] = numpy.maximum(
                1e-11, 1.5e-14 * (1.0 + numpy.fabs(E[g])) / (E[g] - Ec[g])
            )
            t[r] *= 0.5
            a = a[(norm[a] >= tol[a]) & (t[a] >= 1e-8)]
            if a.size == 0:
                break
            Et[a] = numpy.maximum(E[a] + t[a] * dE[a], Ec[a] + 0.1 * (E[a] - Ec[a]))
            ft[a] = numpy.where(
                hold[a] == 0, numpy.clip(f[a] + t[a] * df[a], 1e-13, 1.0 - 1e-13), f[a]
            )
        return E, f, I3, rows, norm < tol

    def _integrals_many(self, jr, ell, jz):
        """(E, I_3) of the tori with actions (J_R, |L_z|, J_z), arrays, in
        the Staeckel model that the reference curve gives each, and which
        of the actions are those of a torus (the integrals of the
        others are NaN): a Newton iteration in the energy and the fraction
        f of the way from the planar to the shell orbit's I_3 (f = 0 or 1
        held on a face, where the energy alone is solved for), vectorized
        over the tori (_newton_many on _action_residual), first on a
        coarse anomaly grid and then on the tori's own. Where p_u^2 has
        two maxima the tori are not one continuous family (_wells) and the
        actions jump across the barrier's level, so the tori that the
        iteration from the epicycle energy leaves are looked for in turn,
        from the best point of a scan of energies and fractions, about the
        outermost maximum on either side of the barrier's level and then
        about the inner maximum (a torus that only the libration about the
        inner maximum has is a torus of the model). The u turning points
        of the tori are remembered for their set-up (_u_turning): where two
        librations share the integrals they tell which is meant."""
        n = len(jr)
        E, I3 = numpy.full(n, numpy.nan), numpy.full(n, numpy.nan)
        found = numpy.zeros(n, dtype="bool")
        tot = jr + jz
        circ = tot <= 1e-10 * ell
        for ii in numpy.flatnonzero(circ):
            # not told from the circular orbit
            E[ii] = self._circular(ell[ii])["Ec"]
            I3[ii], found[ii] = self._I3_planar(E[ii], ell[ii]), True
        chunk = max(2000000 // self._ntau, 16)
        todo = numpy.flatnonzero(~circ)
        for k in range(0, len(todo), chunk):
            sel = todo[k : k + chunk]
            E[sel], I3[sel], found[sel] = self._integrals_chunk(
                jr[sel], ell[sel], jz[sel]
            )
        return E, I3, found

    def _integrals_chunk(self, jr, ell, jz):
        """_integrals_many for tori that are evaluated together, none of
        them the circular orbit"""
        n, tot = len(jr), jr + jz
        Ec, kappa, nu = self._circular_many(ell)
        # an action below 1e-10 of their sum is that of the face
        hold = numpy.where(jr <= 1e-10 * tot, 1, numpy.where(jz <= 1e-10 * tot, 2, 0))
        f0 = numpy.where(hold == 0, numpy.clip(jz / tot, 0.05, 0.95), 1.0 * (hold == 1))
        # the first iterations on a coarse anomaly grid, where the tori are
        # many enough for the size of the grid to set the cost (for a few,
        # the number of evaluations does, and they are on their own grid)
        coarse = min(self._ntau, 256) if n * self._ntau > 2**15 else self._ntau
        inner = numpy.zeros(n, dtype="bool")
        side = numpy.zeros(n, dtype="int")

        def torus(rows, hold, side):
            return (
                (rows[-1] == 1.0)
                & (rows[3] == (hold == 1))
                & (rows[4] == (hold == 2))
                & ((side == 0) | (rows[5] == 0) | (rows[5] == side))
            )

        # from the epicycle energy at the vertical fraction of the actions,
        # lowered until a torus exists there: the highest of a ladder of
        # energies at which one does, on a few anomalies, the rungs taken
        # in blocks (all at once for a few tori, one by one for many) and
        # only for the tori without a torus on the rungs above
        lad = Ec[:, None] + (kappa * jr + nu * jz)[:, None] * 2.0 ** -numpy.arange(16.0)
        E, a, B = lad[:, -1].copy(), numpy.arange(n), int(min(max(1024 // n, 1), 16))
        for k0 in range(0, 16, B):
            ks = numpy.arange(k0, min(k0 + B, 16))
            rep, kk = numpy.repeat(a, len(ks)), numpy.tile(ks, len(a))
            rws = self._action_residual(
                lad[rep, kk], f0[rep], ell[rep], jr[rep], jz[rep], inner[rep], ntau=16
            )[4]
            good = torus(rws, hold[rep], 0).reshape(len(a), -1)
            hit = numpy.any(good, axis=1)
            E[a[hit]] = lad[a[hit], ks[numpy.argmax(good[hit], axis=1)]]
            a = a[~hit]
            if a.size == 0:
                break
        E, f, I3, rows, ok = self._newton_many(
            E, f0, ell, jr, jz, Ec, hold, inner, side, coarse, 40
        )
        Es = Ec[:, None] + (kappa * jr + nu * jz)[:, None] * 2.0 ** numpy.arange(
            -8.0, 14.0
        )
        gs = numpy.linspace(0.04, 0.96, 9)
        for inn, sd in ((False, 0), (False, 1), (False, -1), (True, 1)):
            left = numpy.flatnonzero(~ok)
            if len(left) == 0:
                break
            # the best of a scan of energies and fractions among the tori
            # on the side asked for
            shape = (len(left), Es.shape[1], len(gs))
            rep = numpy.repeat(left, shape[1] * shape[2])
            Eg = numpy.broadcast_to(Es[left][:, :, None], shape).ravel()
            gg = numpy.where(
                hold[rep] == 0, numpy.broadcast_to(gs, shape).ravel(), f0[rep]
            )
            res, _, _, fg, rws = self._action_residual(
                Eg,
                gg,
                ell[rep],
                jr[rep],
                jz[rep],
                numpy.full(len(rep), inn),
                ntau=min(coarse, 64),
                side=numpy.full(len(rep), sd),
            )
            miss = numpy.where(
                torus(rws, hold[rep], sd), numpy.sum(numpy.fabs(res), axis=1), numpy.inf
            ).reshape(shape[0], -1)
            best = numpy.argmin(miss, axis=1) + numpy.arange(shape[0]) * (
                shape[1] * shape[2]
            )
            inner[left], side[left] = inn, sd
            E[left], f[left], I3[left], rows[:, left], ok[left] = self._newton_many(
                Eg[best],
                fg[best],
                ell[left],
                jr[left],
                jz[left],
                Ec[left],
                hold[left],
                inner[left],
                side[left],
                coarse,
                40,
            )
        if self._ntau > coarse:
            # on the tori's own anomaly grid, each torus kept on its side
            done = numpy.flatnonzero(ok)
            side[done] = rows[5, done].astype("int")
            E[done], f[done], I3[done], rows[:, done], ok[done] = self._newton_many(
                E[done],
                f[done],
                ell[done],
                jr[done],
                jz[done],
                Ec[done],
                hold[done],
                inner[done],
                side[done],
                self._ntau,
                8,
            )
        for ii in numpy.flatnonzero(ok):
            for Lz in (ell[ii], -ell[ii]):
                self._u_turning_memo[(E[ii], Lz, I3[ii])] = (
                    rows[0, ii],
                    rows[1, ii],
                    rows[2, ii],
                    bool(rows[3, ii]),
                )
        return numpy.where(ok, E, numpy.nan), numpy.where(ok, I3, numpy.nan), ok

    def _setup_turning_points(self):
        """The turning points of every torus, each in its own Staeckel
        model: what the auxiliary's fit and the samples of the tori need;
        kept only during the set-up by a family (the second-order tables of
        explicit tori, computed when first asked for, sample the tori again)"""
        self._tps = []
        for ii, (E, Lz, I3) in enumerate(zip(self._Es, self._Lzs, self._I3s)):
            with self._in_chart(E, Lz, I3, self._node_delta(ii)):
                self._tps.append(self._turning_points(E, Lz, I3))
        return None

    # ---------- the auxiliary: one isochrone for the whole family
    @staticmethod
    def _aux_vflux(L, c, zeta):
        """The auxiliary's polar action flux p^A_theta d theta^A / d zeta at
        the polar anomaly zeta, in closed form: the auxiliary's polar
        libration is cos(theta^A) = sqrt(c) cos(zeta) with c = sin^2 i =
        1 - L_z^2 / L^2, in which its angle relations are affine"""
        s2 = numpy.sin(zeta) ** 2
        return L * c * s2 / (1.0 - c * numpy.cos(zeta) ** 2)

    @staticmethod
    def _aux_vcumulative(L, c, zeta):
        """The auxiliary's cumulative polar action from the northern turning
        point, A^A_theta(zeta) = int_0^zeta f^A_theta, and its partials with
        respect to (L, c) at fixed zeta, all closed forms: the flux
        L c sin^2 zeta/(1 - c cos^2 zeta) is L [1 - (1 - c)/(1 - c cos^2
        zeta)], whose integral is L [zeta - s Theta(zeta)] with s = sqrt(1 -
        c) and Theta = arctan(tan(zeta)/s) continued through its branches by
        the winding number of zeta about pi; the secular rate over a
        libration is L - |L_z|, the auxiliary's polar action. As for the
        radial cumulative action (_cumulative), the linear parts
        are gathered so that no term exceeds the O(c) result: Theta = zeta +
        psi with psi = arctan(tan(zeta)/s) - arctan(tan(zeta)) = arctan[c
        tan(zeta)/((1 + s)(s + tan^2 zeta))] (no branch to continue), and
        1 - s = c/(1 + s)."""
        zeta = numpy.asarray(zeta, dtype="float")
        s = numpy.sqrt(1.0 - c)
        t = numpy.tan(zeta)
        psi = numpy.arctan(c * t / ((1.0 + s) * (s + t**2)))
        A = L * (c * zeta / (1.0 + s) - s * psi)
        return A, A / L, L * (0.5 * (zeta + psi) / s - 0.5 * t / (s**2 + t**2))

    def _aux_vflux_derivs(self, L, c, zeta):
        """The auxiliary's polar action flux (_aux_vflux) with its partials
        in zeta, L and c, closed forms: with D = 1 - c cos^2 zeta, f = L c
        sin^2 zeta / D, df/dzeta = 2 L c (1 - c) sin zeta cos zeta / D^2,
        df/dL = f / L and df/dc = L sin^2 zeta / D^2"""
        sz, cz = numpy.sin(zeta), numpy.cos(zeta)
        D = 1.0 - c * cz**2
        f = L * c * sz**2 / D
        return f, 2.0 * L * c * (1.0 - c) * sz * cz / D**2, f / L, L * sz**2 / D**2

    def _aux_vcumulative2(self, L, c, zeta):
        """The auxiliary's cumulative polar action (_aux_vcumulative) with
        its first and second partials in (L, c) at fixed zeta: A is linear
        in L, so A_L = A/L, A_LL = 0 and A_Lc = A_c/L; and with Theta = zeta
        + varpi, s = sqrt(1 - c) and t = tan zeta, from dTheta/dc = t/[2 s
        (s^2 + t^2)] and ds/dc = -1/(2s), A_c = L [Theta/(2s) - t/(2 (s^2 +
        t^2))] and A_cc = L [Theta/(4 s^3) + t/(4 s^2 (s^2 + t^2)) - t/(2
        (s^2 + t^2)^2)]. Verified against a forty-digit evaluation of the
        plain form to 1e-13."""
        A, A_L, A_c = self._aux_vcumulative(L, c, zeta)
        s = numpy.sqrt(1.0 - c)
        t = numpy.tan(zeta)
        Theta = zeta + numpy.arctan(c * t / ((1.0 + s) * (s + t**2)))
        q = s**2 + t**2
        A_cc = L * (Theta / (4.0 * s**3) + t / (4.0 * s**2 * q) - 0.5 * t / q**2)
        return A, A_L, A_c, 0.0 * A, A_c / L, A_cc

    def _setup_auxiliary(self, auxiliary):
        """Fix the frozen isochrone auxiliary of the whole family.

        The momentum-matched maps lift every torus onto its equal-action
        auxiliary torus (_lift), so the auxiliary's only job is to be a
        well-conditioned global surrogate of the target. Unless one is
        given, its rotation curve is fitted to the target's over the
        radial range of the tori."""
        if auxiliary is not None:
            if not isinstance(auxiliary, IsochronePotential):
                raise TypeError("auxiliary= has to be an IsochronePotential")
            self._GM, self._b = auxiliary._amp, auxiliary.b
        else:
            self._GM, self._b = self._fit_auxiliary()
        self.auxiliary = (
            auxiliary
            if auxiliary is not None
            else IsochronePotential(amp=self._GM, b=self._b)
        )
        # the isochrone inverse supplies the closed forms of the auxiliary's
        # tori that the lifts, the tables and the evaluation use
        self._aAIinv = actionAngleIsochroneInverse(ip=self.auxiliary)
        return None

    def _lift(self, smp):
        """The lifts of a sampled torus onto the auxiliary, libration by
        libration (None for a degenerate libration). The truncated u-lift
        is required to clear escape by a fraction of its own auxiliary
        torus's binding energy, which fails only when the stored anomaly
        map is under-resolved."""
        L = smp["jz"] + numpy.fabs(smp["Lz"])
        lift_u = lift_v = None
        if smp["u"] is not None:
            lift_u = self._lift_u(smp, smp["jr"], L)
            rA, pA = lift_u["rA"], lift_u["pA"]
            EAs = numpy.max(
                0.5 * (pA**2 + L**2 / rA**2)
                - self._GM / (self._b + numpy.sqrt(self._b**2 + rA**2))
            )
            if EAs >= _BOUND_MARGIN * self._aAIinv._torus_E(smp["jr"], L):
                Rinner = self._delta * numpy.sinh(smp["uc"] - smp["wu"])
                Router = self._delta * numpy.sinh(smp["uc"] + smp["wu"])
                raise RuntimeError(
                    "The momentum-matched lift of the (E, L_z, I_3) = "
                    f"({smp['E']:g}, {smp['Lz']:g}, {smp['I3']:g}) torus is "
                    "not bound in the fitted auxiliary: the u anomaly map is "
                    "under-resolved for it. Its u libration spans the "
                    f"radii {Rinner:g} to {Router:g} in the plane, a ratio of "
                    f"{Router / Rinner:.0f}, and the number of harmonics "
                    "that the map needs grows with that ratio, to at most "
                    "about six times its square root (about "
                    f"{6.0 * numpy.sqrt(Router / Rinner):.0f}), against the "
                    f"mm_npt = {self._npt} given; the most eccentric torus "
                    "of a grid is the one at its smallest angular momentum "
                    "(Rmin) and highest energy (Rinf). Raise mm_npt (and, "
                    "with it, mm_nta), or raise Rmin or lower Rinf"
                )
        if smp["v"] is not None:
            lift_v = self._lift_v(
                smp, L, smp["jz"] * (smp["jz"] + 2.0 * numpy.fabs(smp["Lz"])) / L**2
            )
        return lift_u, lift_v

    def _fit_auxiliary(self):
        """Fit the isochrone's (GM, b) to the potential's rotation curve
        over the radial range of the tori (zero-point free, log residuals
        on geometrically spaced radii)"""
        ds = [self._node_delta(k) or self._delta for k in range(len(self._tps))]
        rlo = min(d * numpy.sinh(tp[0] - tp[1]) for d, tp in zip(ds, self._tps))
        rhi = max(
            d * numpy.sqrt(numpy.sinh(tp[0] + tp[1]) ** 2 + numpy.sin(tp[2]) ** 2)
            for d, tp in zip(ds, self._tps)
        )
        rlo = max(rlo, 1e-3 * rhi)
        rf = numpy.geomspace(rlo, rhi, 25)
        return _fit_isochrone_vc2(
            rf, numpy.log(vcirc(self._staeckelwrap, rf, use_physical=False) ** 2)
        )

    # ---------- the momentum-matched maps and their lifts
    @staticmethod
    def _map(tau, Dm, ms, full=False):
        """The anomaly map eta(tau) = tau + sum_m D_m sin(m tau) over the
        harmonics ms, its derivative, and the sine matrix it is built from
        (and, with full, the cosine matrix);
        the coefficients are those of one torus, (M,), or of every point's
        own torus, (M, npoints)"""
        smt, cmt = _harmonics(tau, ms)
        out = (
            tau + _contract(smt, Dm),
            1.0 + _contract(cmt, ms[:, None] * Dm if Dm.ndim == 2 else ms * Dm),
            smt,
        )
        return (*out, cmt) if full else out

    def _match_anomaly(self, tau, ft, flux, cumulative, ms):
        """The momentum-matched map of one libration onto the auxiliary's:
        match cumulative actions from the inner turning point, eta(tau) =
        A_A^{-1}(A_t(tau)), for the target's flux ft = p dq/dtau sampled on
        the offset grid and the auxiliary's flux and cumulative action, both
        closed forms in its anomaly. Both cumulatives share the linear part
        J (the equal-action choice of the auxiliary torus), so eta - tau is
        periodic and, by time-reversal parity, a pure sine series; the map
        is stored truncated to the harmonics ms, and returned together with
        the truncated map's values and derivative on the grid."""
        mt, qt, _, qt0 = _cumulative(ft)
        At = mt * tau + qt - qt0
        # the auxiliary's secular rate is the action to round-off; the ratio
        # of the two rates makes the linear parts coincide exactly
        scale = mt / (cumulative(2.0 * numpy.pi) / (2.0 * numpy.pi))
        # pointwise Newton on the closed-form cumulative action, with the
        # closed-form flux as its derivative, started from the inverse of
        # the auxiliary's cumulative action interpolated between its grid
        # values (both cumulatives are monotone); the matching is monotone,
        # and a residual left by the iteration shows in the truncation
        # diagnostic of _node_tables
        eta = numpy.interp(At, scale * cumulative(tau), tau)
        for _ in range(200):
            fres = scale * cumulative(eta) - At
            fp = numpy.maximum(scale * flux(eta), 1e-10 * mt)
            eta += numpy.clip(-fres / fp, -0.5, 0.5)
            if numpy.max(numpy.fabs(fres)) < 1e-13 * max(mt, 1e-10):
                break
        Dm = _sine_project(eta - tau, ms)
        shift, dshift = _sine_series(self._ntau, Dm, ms)
        return Dm, tau + shift, 1.0 + dshift

    def _lift_u(self, smp, jr, L):
        """The u libration lifted onto the auxiliary's radial libration of
        equal action: the map's coefficients, its values on the grid, and
        the lifted radius and radial momentum (rebuilt from the TRUNCATED
        map cotangent-consistently, p^A = pi' p_u, so the reconstruction is
        the one the stored tables define)"""
        tau = smp["tau"]
        a, e = self._aAIinv._orbital_params(jr, L)
        ft = smp["pu"] * smp["wu"] * numpy.sin(tau)
        Dm, eta, deta = self._match_anomaly(
            tau,
            ft,
            lambda x: self._aAIinv._flux_derivs(a, e, x)[0],
            lambda x: self._aAIinv._cumulative(a, e, x)[0],
            self._nforDm,
        )
        rA, _, gA = self._aAIinv._profile(a, e, eta)
        return {
            "a": a,
            "e": e,
            "Dm": Dm,
            "eta": eta,
            "rA": rA,
            "pA": ft / (gA * numpy.sin(eta) * deta),
        }

    def _lift_v(self, smp, L, c):
        """The v libration lifted onto the auxiliary's polar libration of
        equal action: the map's coefficients and values, and the lifted
        polar angle and momentum"""
        tau = smp["tau"]
        ft = smp["pv"] * smp["wv"] * numpy.sin(tau)
        Dm, zeta, dzeta = self._match_anomaly(
            tau,
            ft,
            lambda x: self._aux_vflux(L, c, x),
            lambda x: self._aux_vcumulative(L, c, x)[0],
            self._nforDmv,
        )
        sinth = numpy.sqrt(1.0 - c * numpy.cos(zeta) ** 2)
        return {
            "Dm": Dm,
            "zeta": zeta,
            "sinth": sinth,
            "pA": ft * sinth / (numpy.sqrt(c) * numpy.sin(zeta) * dzeta),
        }

    def _map_slopes_u(self, smp, jr, L, a, e, eta, dJr, dL, dsup, nharm, dLu0=None):
        """The derivatives of the u map's coefficients with respect to
        (E, I_3, L_z), on one torus from that torus alone: the variation of
        the matching condition A^A(eta(tau); a, e) = A(tau; E, L_z, I_3) at
        fixed anomaly,

            f^A(eta) sum_m dD_m/dalpha sin(m tau)
                = dA/dalpha|_tau - dA^A/da a_alpha - dA^A/de e_alpha ,

        a LINEAR problem for the dD_m/dalpha with the vanishing flux f^A
        multiplying the unknowns, solved by Galerkin projection onto the
        sine basis (_sine_galerkin). The target's flux derivative at fixed
        tau goes through the moving turning points and is regular there
        (its numerator vanishes with p_u), so it integrates spectrally like
        the flux itself; the auxiliary's cumulative action varies through
        its torus parameters' closed-form chains, with dJ^A_r/dalpha and
        dL^A/dalpha from the torus's period matrix and the closed-form
        partials of _cumulative. The problem is solved in the map's first
        nharm harmonics, and the slopes of the others are zero. With dLu0,
        the derivative of L = J_z + |L_z| with respect to the reference u0
        of a rule at fixed integrals, the result has a fourth entry, the
        map's derivative with respect to u0: the u libration does not
        depend on u0, but the auxiliary torus onto which it is lifted does,
        through L."""
        tau, u, pu, wu = smp["tau"], smp["u"], smp["pu"], smp["wu"]
        E, Lz = smp["E"], smp["Lz"]
        d2 = self._delta**2
        st, ct = numpy.sin(tau), numpy.cos(tau)
        dudtau = wu * st
        dWdu = self._dWu(u, E, Lz)
        sh2 = numpy.sinh(u) ** 2
        dWda = (
            2.0 * d2 * sh2,
            -2.0 * d2 * numpy.ones_like(u),
            -2.0 * Lz / sh2,
            *self._dW_ddelta(u, Lz, pu**2),
        )
        ndir = len(dWda)
        _, FAa, FAe = self._aAIinv._cumulative(a, e, eta)
        fAeta = self._aAIinv._flux_derivs(a, e, eta)[0]
        rhs = numpy.empty((self._ntau, ndir))
        for k in range(ndir):
            du = dsup[0, k] - dsup[1, k] * ct
            dft = (dWda[k] + dWdu * du) / (2.0 * pu) * dudtau + pu * dsup[1, k] * st
            mt, qt, _, qt0 = _cumulative(dft)
            _, _, da, de = self._aAIinv._orbital_param_chains(jr, L, dJr[k], dL[k])
            rhs[:, k] = mt * tau + qt - qt0 - FAa * da - FAe * de
        if dLu0 is not None:
            _, _, da, de = self._aAIinv._orbital_param_chains(jr, L, 0.0, dLu0)
            rhs = numpy.hstack((rhs, (-FAa * da - FAe * de)[:, None]))
        out = numpy.zeros((self._npt, rhs.shape[1]))
        out[:nharm] = _sine_galerkin(fAeta, rhs, self._nforDm[:nharm])
        return out

    def _map_slopes_v(self, smp, L, c, zeta, dL, dwv, nharm, dVdu0=None):
        """The derivatives of the v map's coefficients with respect to
        (E, I_3, L_z), as _map_slopes_u for the u map: the auxiliary's polar
        cumulative action depends on its torus through (L, c = 1 -
        L_z^2/L^2), with closed-form partials (_aux_vcumulative). With
        dVdu0, the derivative of V with respect to the reference u0 on the
        samples, dL and dwv have a fourth entry and so has the result: the
        derivative with respect to u0 at fixed integrals, in which p_v^2
        varies by 2 delta^2 dV/du0 (as it does by 2 delta^2 in I_3)"""
        tau, v, pv, wv = smp["tau"], smp["v"], smp["pv"], smp["wv"]
        E, Lz = smp["E"], smp["Lz"]
        d2 = self._delta**2
        st, ct = numpy.sin(tau), numpy.cos(tau)
        dvdtau = wv * st
        dWdv = self._dWv(v, E, Lz)
        sn2 = numpy.sin(v) ** 2
        dWda = [2.0 * d2 * sn2, 2.0 * d2 * numpy.ones_like(v), -2.0 * Lz / sn2]
        dWda += self._dW_ddelta(v, Lz, pv**2, lib="v")
        dLz = [0.0, 0.0, 1.0, 0.0][: len(dWda)]
        if dVdu0 is not None:
            dWda.append(2.0 * d2 * dVdu0)
            dLz.append(0.0)
        ndir = len(dWda)
        _, FL, Fc = self._aux_vcumulative(L, c, zeta)
        fAzeta = self._aux_vflux(L, c, zeta)
        dc = 2.0 * Lz**2 / L**3 * dL - 2.0 * Lz / L**2 * numpy.array(dLz)
        rhs = numpy.empty((self._ntau, ndir))
        for k in range(ndir):
            dv = -dwv[k] * ct
            dft = (dWda[k] + dWdv * dv) / (2.0 * pv) * dvdtau + pv * dwv[k] * st
            mt, qt, _, qt0 = _cumulative(dft)
            rhs[:, k] = mt * tau + qt - qt0 - FL * dL[k] - Fc * dc[k]
        out = numpy.zeros((self._nptv, ndir))
        out[:nharm] = _sine_galerkin(fAzeta, rhs, self._nforDmv[:nharm])
        return out

    # ---------- the per-torus tables, computed (never fitted)
    def _node_tables(self, ii):
        """One torus (the ii-th sampled): its actions and its period
        matrix by regular quadrature in the anomalies (the derivatives of
        the actions with respect to the integrals, whose inverse gives the
        frequencies), the turning points with their closed-form
        derivatives, the two maps' coefficients with their slopes in
        (E, I_3, L_z) from the torus alone, and the variation of the
        auxiliary's actions along the lifted librations (the maps'
        truncation, which should be at round-off). A degenerate libration
        contributes its harmonic limit. The torus's samples and lifts, which
        the second-order tables use as well, are returned with the tables
        and not kept."""
        smp = self._sample_torus(
            self._Es[ii], self._Lzs[ii], self._I3s[ii], tp=self._tps[ii]
        )
        tau, E, Lz, I3 = smp["tau"], smp["E"], smp["Lz"], smp["I3"]
        uc, wu, wv, ush = smp["uc"], smp["wu"], smp["wv"], smp["ush"]
        d2 = self._delta**2
        st = numpy.sin(tau)
        udeg, vdeg = smp["u"] is None, smp["v"] is None
        jr, jz = smp["jr"], smp["jz"]
        # the period matrix dJ/d(E, I_3, L_z): (dq/dtau)/p is periodic and
        # finite (dq/dtau and p vanish together at the turning points), so
        # the trapezoid rule on the periodic grid is spectrally accurate; a
        # degenerate libration has the harmonic limit, J = W(q_0) / (2 sqrt
        # beta) with beta = -W''(q_0)/2 at the degenerate point
        ndir = 3 + (self._delta_rule is not None)
        M = numpy.empty((2, ndir))
        dsup = numpy.full((3, ndir), numpy.nan)
        # (half the derivatives of the momenta squared; the fourth direction,
        # of a family whose focal length follows its tori, is _dW_ddelta's)
        half = lambda dW, w=1.0: [0.5 * numpy.mean(t * w) for t in dW]  # noqa: E731
        if udeg:
            sqb = numpy.sqrt(-0.5 * float(self._d2Wu(ush, E, Lz)))
            sh2 = numpy.sinh(ush) ** 2
            dWD = half(self._dW_ddelta(ush, Lz))
            M[0] = numpy.array([d2 * sh2, -d2, -Lz / sh2, *dWD]) / sqb
            dush = self._dushell(ush, E, Lz)
            dsup[0] = [dush[0], 0.0, *dush[1:]]
        else:
            u, pu = smp["u"], smp["pu"]
            ratio = wu * st / pu
            sh2 = numpy.sinh(u) ** 2
            M[0] = [
                numpy.mean(d2 * sh2 * ratio),
                -d2 * numpy.mean(ratio),
                -Lz * numpy.mean(ratio / sh2),
                *half(self._dW_ddelta(u, Lz, pu**2), ratio),
            ]
            dinner = self._tp_derivs_u(uc - wu, E, Lz)
            douter = self._tp_derivs_u(uc + wu, E, Lz)
            dsup[0], dsup[1] = 0.5 * (dinner + douter), 0.5 * (douter - dinner)
        if vdeg:
            sqb = numpy.sqrt(-0.5 * float(self._d2Wv(0.5 * numpy.pi, E, Lz)))
            dWD = half(self._dW_ddelta(0.5 * numpy.pi, Lz, lib="v"))
            M[1] = numpy.array([d2, d2, -Lz, *dWD]) / sqb
        else:
            v, pv = smp["v"], smp["pv"]
            ratio = wv * st / pv
            sn2 = numpy.sin(v) ** 2
            M[1] = [
                numpy.mean(d2 * sn2 * ratio),
                d2 * numpy.mean(ratio),
                -Lz * numpy.mean(ratio / sn2),
                *half(self._dW_ddelta(v, Lz, pv**2, lib="v"), ratio),
            ]
            dsup[2] = -self._tp_derivs_v(0.5 * numpy.pi - wv, E, Lz)
        # d(E, I_3, L_z)/d(J_R, J_z, J_phi), from J_phi = L_z
        N = numpy.linalg.inv(numpy.vstack((M[:, :3], [0.0, 0.0, 1.0])))
        L = jz + numpy.fabs(Lz)
        c = jz * (jz + 2.0 * numpy.fabs(Lz)) / L**2
        dJr, dL = M[0], M[1] + numpy.array([0.0, 0.0, numpy.sign(Lz), 0.0][:ndir])
        lift_u, lift_v = self._lift(smp)
        dVu = Ju = None
        if self._adaptive and not vdeg:
            # along the reference u0 of a rule at fixed integrals V(v)
            # changes, and with it J_z (from the same quadrature as its
            # period matrix)
            dVu = self._staeckelwrap._dVdu0(smp["v"])
            Ju = d2 * numpy.mean(dVu * wv * st / smp["pv"])

        def kept(Dm):
            # a family's torus keeps the coefficients of each of its maps
            # down to the map's round-off floor (_harmonics_kept): those
            # beyond are set to zero, and the slopes are solved in those kept
            nh = _harmonics_kept(Dm) if self._interp else len(Dm)
            return nh, numpy.where(numpy.arange(len(Dm)) < nh, Dm, 0.0)

        node = {
            "jr": jr,
            "jz": jz,
            "M": M,
            "N": N,
            "uc": uc,
            "wu": wu,
            "wv": wv,
            "ush": ush,
            "dsup": dsup,
            "Dmu": numpy.zeros(self._npt),
            "Dmv": numpy.zeros(self._nptv),
            "dDmu": None,
            "dDmv": None,
            "perr_u": 0.0,
            "perr_v": 0.0,
            "smp": smp,
            "lifts": (lift_u, lift_v),
        }
        if not udeg:
            # the truncated map's lift is not exactly the equal-action
            # torus: the auxiliary's radial action varies along it by the
            # truncation residual
            rA, pA = lift_u["rA"], lift_u["pA"]
            EA = 0.5 * (pA**2 + L**2 / rA**2) - self._GM / (
                self._b + numpy.sqrt(self._b**2 + rA**2)
            )
            node["perr_u"] = float(
                numpy.amax(numpy.fabs(self._aAIinv._torus_Jr(EA, L) - jr)) / jr
            )
            nh, node["Dmu"] = kept(lift_u["Dm"])
            slopes = self._map_slopes_u(
                smp,
                jr,
                L,
                lift_u["a"],
                lift_u["e"],
                lift_u["eta"],
                dJr,
                dL,
                dsup,
                nh,
                dLu0=Ju,
            )
            node["dDmu"] = slopes[:, :ndir]
            if Ju is not None:
                node["du0u"] = slopes[:, ndir]
        if not vdeg:
            # likewise the auxiliary's polar action along the lifted v
            # libration, through its total angular momentum
            LA = numpy.sqrt(lift_v["pA"] ** 2 + Lz**2 / lift_v["sinth"] ** 2)
            node["perr_v"] = float(
                numpy.amax(numpy.fabs(LA - numpy.fabs(Lz) - jz)) / jz
            )
            nh, node["Dmv"] = kept(lift_v["Dm"])
            if self._adaptive:
                # the tables of the v libration along the reference u0 at
                # fixed integrals (of the u libration's, only the map
                # depends on it, through the auxiliary torus: above): the
                # action's above, the half-width's by the level-set rule,
                # the map's as a fourth direction of its slopes
                vt = 0.5 * numpy.pi - wv
                wu0 = (
                    2.0
                    * d2
                    * float(self._staeckelwrap._dVdu0(vt))
                    / float(self._dWv(vt, E, Lz))
                )
                slopes = self._map_slopes_v(
                    smp,
                    L,
                    c,
                    lift_v["zeta"],
                    numpy.append(dL, Ju),
                    numpy.append(dsup[2], wu0),
                    nh,
                    dVdu0=dVu,
                )
                node["dDmv"] = slopes[:, :ndir]
                node["du0"] = (Ju, wu0, slopes[:, ndir])
            else:
                node["dDmv"] = self._map_slopes_v(
                    smp, L, c, lift_v["zeta"], dL, dsup[2], nh
                )
        return node

    def _second_order_tables(self, ii, node):
        """The second-order tables of the ii-th explicit torus, from the
        torus alone, which make its one-node family quadratic: over the
        six pairs of (E, I_3, L_z), the second derivatives of the u
        libration's midpoint and half-width and of the v half-width (the
        level-set rule differentiated once more), of the two actions (the
        period matrix's derivatives, from the same regular quadratures as
        the matrix itself), and of the two maps' coefficients (the second
        variation of each matching condition at fixed anomaly, solved by
        the same Galerkin projection as the slopes; the u map's auxiliary
        chains run through (a, e)(J_R, L) and the v map's through (L, c),
        with L = J_z + |L_z| itself varying to second order through the
        period matrix). Every target-side integrand is regular at the
        turning points: the numerators N_a = S_a + S_q dq/da|_tau and N_ab
        vanish there to second order in tau while p vanishes to first.
        Returns None for a torus with a degenerate libration, whose tables
        stay zero. The chain to the actions is _quadratic_node."""
        smp = node["smp"]
        if smp["u"] is None or smp["v"] is None:
            return None
        lift_u, lift_v = node["lifts"]
        tau, E, Lz = smp["tau"], smp["E"], smp["Lz"]
        jr, jz, M, dsup = smp["jr"], smp["jz"], node["M"], node["dsup"]
        L = jz + numpy.fabs(Lz)
        c = jz * (jz + 2.0 * numpy.fabs(Lz)) / L**2
        d2 = self._delta**2
        st, ct = numpy.sin(tau), numpy.cos(tau)
        pairs = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))

        def tp2(S_q, S_qq, S_qa, S_LL, q1):
            # a turning point's second derivatives by the level-set rule
            # differentiated once more, S_q q_ab = -[S_qq q_a q_b + S_qa q_b
            # + S_qb q_a + S_ab], with S_ab = S_LL for (L_z, L_z) and zero
            # for every other pair
            return numpy.array(
                [
                    -(
                        S_qq * q1[i] * q1[j]
                        + S_qa[i] * q1[j]
                        + S_qa[j] * q1[i]
                        + (S_LL if i == j == 2 else 0.0)
                    )
                    / S_q
                    for i, j in pairs
                ]
            )

        def libration(q, p, w, q1, w1, q2, w2, S_q, S_qq, S_a, S_qa, S_LL):
            # at fixed anomaly: the second derivatives of the action (the
            # period-matrix row) and of the cumulative action, from the
            # coordinate's derivatives q1[a] = dq/da|_tau and q2[ab], the
            # half-width's w1, w2, and the momentum function's partials
            qtau = w * st
            N1 = [S_a[i] + S_q * q1[i] for i in range(3)]
            M2, A2 = numpy.empty(6), numpy.empty((self._ntau, 6))
            for kk, (i, j) in enumerate(pairs):
                S_ab = S_LL if i == j == 2 else 0.0 * q
                N2 = (
                    S_ab
                    + S_qa[i] * q1[j]
                    + S_qa[j] * q1[i]
                    + S_qq * q1[i] * q1[j]
                    + S_q * q2[kk]
                )
                M2[kk] = numpy.mean(
                    (S_ab + S_qa[i] * q1[j]) * qtau / (2.0 * p)
                    + S_a[i] * (w1[j] * st / (2.0 * p) - qtau * N1[j] / (4.0 * p**3))
                )
                mt, qt, _, qt0 = _cumulative(
                    (N2 * qtau + N1[i] * w1[j] * st + N1[j] * w1[i] * st) / (2.0 * p)
                    - N1[i] * N1[j] * qtau / (4.0 * p**3)
                    + p * w2[kk] * st
                )
                A2[:, kk] = mt * tau + qt - qt0
            return M2, A2

        # the u libration: both turning points, then the midpoint and the
        # half-width
        u, pu, wu, uc = smp["u"], smp["pu"], smp["wu"], smp["uc"]
        d2tp = []
        for ut, dut in ((uc - wu, dsup[0] - dsup[1]), (uc + wu, dsup[0] + dsup[1])):
            sh2 = numpy.sinh(ut) ** 2
            d2tp.append(
                tp2(
                    float(self._dWu(ut, E, Lz)),
                    float(self._d2Wu(ut, E, Lz)),
                    (
                        2.0 * d2 * numpy.sinh(2.0 * ut),
                        0.0,
                        4.0 * Lz * numpy.cosh(ut) / numpy.sinh(ut) ** 3,
                    ),
                    -2.0 / sh2,
                    dut,
                )
            )
        uc2, wu2 = 0.5 * (d2tp[0] + d2tp[1]), 0.5 * (d2tp[1] - d2tp[0])
        sh2 = numpy.sinh(u) ** 2
        M2u, A2u = libration(
            u,
            pu,
            wu,
            [dsup[0, k] - dsup[1, k] * ct for k in range(3)],
            dsup[1],
            [uc2[kk] - wu2[kk] * ct for kk in range(6)],
            wu2,
            self._dWu(u, E, Lz),
            self._d2Wu(u, E, Lz),
            (2.0 * d2 * sh2, -2.0 * d2 + 0.0 * u, -2.0 * Lz / sh2),
            (
                2.0 * d2 * numpy.sinh(2.0 * u),
                0.0 * u,
                4.0 * Lz * numpy.cosh(u) / numpy.sinh(u) ** 3,
            ),
            -2.0 / sh2,
        )
        # the v libration: its turning point, then the half-width
        v, pv, wv = smp["v"], smp["pv"], smp["wv"]
        vt = 0.5 * numpy.pi - wv
        sn2 = numpy.sin(vt) ** 2
        wv2 = -tp2(
            float(self._dWv(vt, E, Lz)),
            float(self._d2Wv(vt, E, Lz)),
            (
                2.0 * d2 * numpy.sin(2.0 * vt),
                0.0,
                4.0 * Lz * numpy.cos(vt) / numpy.sin(vt) ** 3,
            ),
            -2.0 / sn2,
            -dsup[2],
        )
        sn2 = numpy.sin(v) ** 2
        M2v, A2v = libration(
            v,
            pv,
            wv,
            [-dsup[2, k] * ct for k in range(3)],
            dsup[2],
            [-wv2[kk] * ct for kk in range(6)],
            wv2,
            self._dWv(v, E, Lz),
            self._d2Wv(v, E, Lz),
            (2.0 * d2 * sn2, 2.0 * d2 + 0.0 * v, -2.0 * Lz / sn2),
            (
                2.0 * d2 * numpy.sin(2.0 * v),
                0.0 * v,
                4.0 * Lz * numpy.cos(v) / numpy.sin(v) ** 3,
            ),
            -2.0 / sn2,
        )
        # the auxiliary torus's parameters along the integrals: J_R and L =
        # J_z + |L_z| to second order from the period matrix and its
        # derivatives, (a, e) through their closed-form partials in (J_R, L)
        J1, J2 = M[0], M2u
        L1 = M[1] + numpy.array([0.0, 0.0, numpy.sign(Lz)])
        L2 = M2v

        def chain2(d):
            dJ, dL, dJJ, dJL, dLL = d
            first = [dJ * J1[i] + dL * L1[i] for i in range(3)]
            second = [
                dJJ * J1[i] * J1[j]
                + dJL * (J1[i] * L1[j] + J1[j] * L1[i])
                + dLL * L1[i] * L1[j]
                + dJ * J2[kk]
                + dL * L2[kk]
                for kk, (i, j) in enumerate(pairs)
            ]
            return first, second

        _, _, da, de = self._aAIinv._orbital_params2(jr, L)
        a1, a2 = chain2(da)
        e1, e2 = chain2(de)
        a, e, eta = lift_u["a"], lift_u["e"], lift_u["eta"]
        _, AA_a, AA_e, AA_aa, AA_ae, AA_ee = self._aAIinv._cumulative2(a, e, eta)
        fA, fA_eta = self._aAIinv._flux_eta(a, e, eta)
        _, fA_a, fA_e = self._aAIinv._flux_derivs(a, e, eta)
        eta1 = [
            _sine_series(self._ntau, node["dDmu"][:, k], self._nforDm)[0]
            for k in range(3)
        ]
        rhs = numpy.empty((self._ntau, 6))
        for kk, (i, j) in enumerate(pairs):
            rhs[:, kk] = (
                A2u[:, kk]
                - (
                    fA_eta * eta1[i] * eta1[j]
                    + fA_a * (a1[j] * eta1[i] + a1[i] * eta1[j])
                    + fA_e * (e1[j] * eta1[i] + e1[i] * eta1[j])
                )
                - (
                    AA_aa * a1[i] * a1[j]
                    + AA_ae * (a1[i] * e1[j] + a1[j] * e1[i])
                    + AA_ee * e1[i] * e1[j]
                )
                - AA_a * a2[kk]
                - AA_e * e2[kk]
            )
        D2u = _sine_galerkin(fA, rhs, self._nforDm)
        # the v map: the auxiliary's polar libration through (L, c), with
        # c = 1 - L_z^2/L^2 to second order
        dLz = numpy.array([0.0, 0.0, 1.0])
        c1 = 2.0 * Lz**2 / L**3 * L1 - 2.0 * Lz / L**2 * dLz
        c2 = numpy.array(
            [
                2.0 * Lz**2 / L**3 * L2[kk]
                - 6.0 * Lz**2 / L**4 * L1[i] * L1[j]
                + 4.0 * Lz / L**3 * (dLz[i] * L1[j] + dLz[j] * L1[i])
                - 2.0 / L**2 * dLz[i] * dLz[j]
                for kk, (i, j) in enumerate(pairs)
            ]
        )
        zeta = lift_v["zeta"]
        fv, fv_z, fv_L, fv_c = self._aux_vflux_derivs(L, c, zeta)
        _, Av_L, Av_c, Av_LL, Av_Lc, Av_cc = self._aux_vcumulative2(L, c, zeta)
        zeta1 = [
            _sine_series(self._ntau, node["dDmv"][:, k], self._nforDmv)[0]
            for k in range(3)
        ]
        for kk, (i, j) in enumerate(pairs):
            rhs[:, kk] = (
                A2v[:, kk]
                - (
                    fv_z * zeta1[i] * zeta1[j]
                    + fv_L * (L1[j] * zeta1[i] + L1[i] * zeta1[j])
                    + fv_c * (c1[j] * zeta1[i] + c1[i] * zeta1[j])
                )
                - (
                    Av_LL * L1[i] * L1[j]
                    + Av_Lc * (L1[i] * c1[j] + L1[j] * c1[i])
                    + Av_cc * c1[i] * c1[j]
                )
                - Av_L * L2[kk]
                - Av_c * c2[kk]
            )
        D2v = _sine_galerkin(fv, rhs, self._nforDmv)
        return {
            "d2sup": numpy.vstack((uc2, wu2, wv2)),
            "d2M": numpy.vstack((M2u, M2v)),
            "d2Dmu": D2u,
            "d2Dmv": D2v,
        }

    def _second_order(self):
        """The second-order tables of the explicit tori (those of the u
        turning points, the actions and the two maps: _second_order_tables),
        which make each one-node family quadratic: an explicit torus then
        carries the true derivatives of the map in the actions at its node.
        No evaluation needs them, so they are computed when they are first
        asked for (not for tori set up with a rule for the reference curve,
        whose tables would need the model's second derivatives along the
        rule; theirs stay zero)"""
        if self._d2 is None:
            ntori = len(self._Es)
            d2 = (
                numpy.zeros((ntori, 3, 6)),
                numpy.zeros((ntori, 2, 6)),
                numpy.zeros((ntori, self._npt, 6)),
                numpy.zeros((ntori, self._nptv, 6)),
            )
            for ii in range(ntori * (not self._adaptive)):
                so = self._second_order_tables(ii, self._node_tables(ii))
                if so is not None:
                    d2[0][ii], d2[1][ii] = so["d2sup"], so["d2M"]
                    d2[2][ii], d2[3][ii] = so["d2Dmu"], so["d2Dmv"]
            self._d2 = d2
        return self._d2

    _d2sups = property(lambda self: self._second_order()[0])
    _d2Ms = property(lambda self: self._second_order()[1])
    _d2Dmus = property(lambda self: self._second_order()[2])
    _d2Dmvs = property(lambda self: self._second_order()[3])

    def _quadratic_node(self, ii):
        """The ii-th explicit torus's one-node family to second order in
        the actions (J_R, J_z, J_phi): with N = d(E, I_3, L_z)/dJ the
        inverse period matrix, the integrals' second derivatives in the
        actions follow from the inverse-function rule, alpha_{gamma,ij} =
        -sum_k N_{gamma k} [N^T J_{k,..} N]_{ij} over the two actions'
        second derivatives (the E row is the frequencies' derivatives), and
        every table X(E, I_3, L_z) chains as X_{J_i} = sum_a X_a N_{ai} and
        X_{J_i J_j} = [N^T X_{..} N]_{ij} + sum_a X_a alpha_{a,ij}; a
        degenerate libration's half-width and map have no derivatives, as
        in _explicit_tables"""
        if self._adaptive:
            raise NotImplementedError(
                "the second-order tables of a torus set up with a rule for its "
                "reference curve (u0='torus', 'planar', 'shell' or a callable) are "
                "not available: they need the second derivatives of the Staeckel "
                "model along the rule; set the torus up with the rule's u0 as a "
                "number for those of its own Staeckel model"
            )
        N = self._Ns[ii]
        idx = numpy.array([[0, 1, 2], [1, 3, 4], [2, 4, 5]])

        def sym(x6):
            # the six pairs as a symmetric matrix (along the last axis)
            return numpy.asarray(x6)[..., idx]

        JJ = numpy.zeros((3, 3, 3))
        JJ[:2] = sym(self._d2Ms[ii])
        d2alpha = -numpy.einsum("gk,ai,kab,bj->gij", N, N, JJ, N)

        def chain(x1, x6, zero=False):
            x1, x6 = numpy.asarray(x1), numpy.asarray(x6)
            if zero:
                return numpy.zeros(x1.shape), numpy.zeros(x1.shape + (3,))
            return x1 @ N, numpy.einsum(
                "ai,...ab,bj->...ij", N, sym(x6), N
            ) + numpy.einsum("...a,aij->...ij", x1, d2alpha)

        udeg, vdeg = self._wus[ii] == 0.0, self._wvs[ii] == 0.0
        dsup, d2sup = self._dsups[ii], self._d2sups[ii]
        out = {
            "jr": self._jrs[ii],
            "jz": self._jzs[ii],
            "Lz": self._Lzs[ii],
            "Om": N[0],
            "d2E": d2alpha[0],
            "d2alpha": d2alpha,
            "uc": self._ucs[ii],
            "wu": self._wus[ii],
            "wv": self._wvs[ii],
            "Dmu": self._Dmus[ii],
            "Dmv": self._Dmvs[ii],
        }
        out["duc"], out["d2uc"] = chain(dsup[0], d2sup[0])
        out["dwu"], out["d2wu"] = chain(dsup[1], d2sup[1], zero=udeg)
        out["dwv"], out["d2wv"] = chain(dsup[2], d2sup[2], zero=vdeg)
        out["dDmu"], out["d2Dmu"] = chain(self._dDmus[ii], self._d2Dmus[ii])
        out["dDmv"], out["d2Dmv"] = chain(self._dDmvs[ii], self._d2Dmvs[ii])
        return out

    def _warn_unresolved(self, perr, Es, Lzs, I3s, which):
        """Warn when a truncated map leaves the auxiliary's action varying
        along a lifted libration by more than round-off allows"""
        bad = numpy.asarray(perr) > 1e-6
        if numpy.any(bad):
            warnings.warn(
                "The momentum-matched {} anomaly map is not converged for the (E, L_z, I_3) tori: {} (maximum relative variation of the auxiliary action along a lifted libration {:.1e}); increase mm_npt and, with it, mm_nta".format(
                    which,
                    ", ".join(
                        f"({E:g}, {Lz:g}, {I3:g})"
                        for E, Lz, I3 in zip(
                            numpy.asarray(Es)[bad],
                            numpy.asarray(Lzs)[bad],
                            numpy.asarray(I3s)[bad],
                        )
                    ),
                    float(numpy.amax(perr)),
                ),
                galpyWarning,
            )
        return None

    def _setup_tori(self):
        if self._interp:
            return self._setup_tori_interp()
        ntori = len(self._Es)
        self._jrs = numpy.empty(ntori)
        self._jzs = numpy.empty(ntori)
        self._Ns = numpy.empty((ntori, 3, 3))
        self._ucs = numpy.empty(ntori)
        self._wus = numpy.empty(ntori)
        self._wvs = numpy.empty(ntori)
        self._dsups = numpy.empty((ntori, 3, 3))
        self._Dmus = numpy.empty((ntori, self._npt))
        self._Dmvs = numpy.empty((ntori, self._nptv))
        self._dDmus = numpy.zeros((ntori, self._npt, 3))
        self._dDmvs = numpy.zeros((ntori, self._nptv, 3))
        # (the second-order tables, which no evaluation needs, are computed
        # when they are first asked for: _second_order)
        self._d2 = None
        perr_u, perr_v = numpy.empty(ntori), numpy.empty(ntori)
        for ii in _progress(range(ntori), "node tables", self._progressbar):
            with self._in_chart(self._Es[ii], self._Lzs[ii], self._I3s[ii]):
                node = self._node_tables(ii)
            if "du0" in node:
                # a torus set up with a rule for its reference curve is a
                # torus of the family that the rule defines: its tables
                # carry the variation of the model along the rule, as a
                # family's nodes do
                du0 = self._u0_grad(
                    self._Es[ii], numpy.fabs(self._Lzs[ii]), self._I3s[ii]
                ) * numpy.array([1.0, 1.0, numpy.sign(self._Lzs[ii])])
                M = node["M"].copy()
                M[1] += node["du0"][0] * du0
                node["N"] = numpy.linalg.inv(numpy.vstack((M, [0.0, 0.0, 1.0])))
                node["dsup"][2] += node["du0"][1] * du0
                node["dDmv"] = node["dDmv"] + node["du0"][2][:, None] * du0[None, :]
                if "du0u" in node:
                    node["dDmu"] = node["dDmu"] + node["du0u"][:, None] * du0[None, :]
            self._jrs[ii], self._jzs[ii] = node["jr"], node["jz"]
            self._Ns[ii] = node["N"]
            self._ucs[ii], self._wus[ii], self._wvs[ii] = (
                node["uc"],
                node["wu"],
                node["wv"],
            )
            self._dsups[ii] = node["dsup"]
            self._Dmus[ii], self._Dmvs[ii] = node["Dmu"], node["Dmv"]
            # a degenerate libration has no map slopes to store: it
            # contributes nothing to the evaluation
            if node["dDmu"] is not None:
                self._dDmus[ii] = node["dDmu"]
            if node["dDmv"] is not None:
                self._dDmvs[ii] = node["dDmv"]
            perr_u[ii], perr_v[ii] = node["perr_u"], node["perr_v"]
        self._warn_unresolved(perr_u, self._Es, self._Lzs, self._I3s, "u")
        self._warn_unresolved(perr_v, self._Es, self._Lzs, self._I3s, "v")
        return None

    # ---------- the (L_z, E, I_3) interpolation grid
    def _setup_grid(self, Rmin, Rmax, Rinf, nE, nLz, nI3, Lzspacing="log"):
        """Rectangular grid in (w_E, w_I, L_z): L_z between the circular
        angular momenta of Rmin and Rmax; the energies uniform, in x =
        w_E^2 with w_E uniform in [0, 1], in the logarithm of the apocentre
        of the planar orbit without angular momentum, R_0(E) with
        Phi(R_0, 0) = E, between the circular orbit's energy E_c(L_z) and
        E_max(L_z), the energy of the planar orbit with apocentre Rinf
        (_E_of_xL); the circular orbits form the bottom face, and in the
        energy the spacing is quadratic there and compresses toward the
        top in a potential with an escape energy, where every apocentre
        diverges, so that the tables stay smooth in x up to the top row
        (for a logarithmic potential it is E = E_c + [E_max - E_c] x);
        the third integrals at y = sin^2(pi w_I / 2) of the third-integral
        variable y (_y_of_I3), with w_I uniform in [0, 1] between the
        planar orbit (J_z = 0) and the shell orbit (J_R = 0), the two faces
        on which one libration degenerates -- quadratic spacing at every
        degenerate edge, where the actions behave as the square of the
        distance to it and the half-widths as the distance itself"""
        if nE < 4 or nLz < 4 or nI3 < 4:
            raise ValueError("setup_interp=True requires nE, nLz, and nI3 >= 4")
        if Rinf <= Rmax:
            raise ValueError(
                "Rinf has to exceed Rmax: it is the apocentre in the plane of "
                "the grid's most energetic orbits, beyond every circular orbit "
                "of the grid"
            )
        wrap = self._staeckelwrap
        Lmin = Rmin * vcirc(wrap, Rmin, use_physical=False)
        Lmax = Rmax * vcirc(wrap, Rmax, use_physical=False)
        self._Rinf = Rinf
        self._Phiinf = evaluatePotentials(wrap, Rinf, 0.0, use_physical=False)
        self._R0_cache = {}
        # the grid's box in (L_z, w_E, w_I): the default spans, or the
        # target's padded range
        box = ((Lmin, Lmax), (0.0, 1.0), (0.0, 1.0))
        if self._target is not None:
            box = self._target_box(Lmin, Lmax)
        self._box = box
        self._Lzgrid = _spaced_grid(box[0][0], box[0][1], nLz, Lzspacing)
        # the energy variable's bottom apocentre is that of the planar orbit
        # without angular momentum at the circular orbit's energy, which has
        # to lie inside Rinf for every L_z of the grid: Phi(Rinf, 0) must
        # exceed the circular energy of the grid's largest |L_z|
        if self._Phiinf <= self._circular(self._Lzgrid[-1])["Ec"]:
            raise ValueError(
                "Rinf is too small: the planar orbit without angular momentum at "
                "the circular orbit's energy of the grid's largest |L_z| reaches "
                "beyond Rinf; increase Rinf"
                + (
                    ""
                    if self._target is None
                    else " or reduce target_pad: the grid's largest |L_z|, "
                    f"{self._Lzgrid[-1]:g}, is the target's, "
                    f"{numpy.amax(self._target[0] * numpy.fabs(self._target[2])):g}, "
                    f"padded by target_pad = {self._target_pad:g} times the "
                    "target's range of |L_z|"
                )
            )
        self._wEs = numpy.linspace(box[1][0], box[1][1], nE)
        self._wIs = numpy.linspace(box[2][0], box[2][1], nI3)
        self._ys = numpy.sin(0.5 * numpy.pi * self._wIs) ** 2
        self._E_tab = numpy.empty((nE, nLz))
        self._ush_tab = numpy.empty((nE, nI3, nLz))
        self._Ipl_tab = numpy.empty((nE, nI3, nLz))
        self._Ish_tab = numpy.empty((nE, nI3, nLz))
        I3s = numpy.empty((nE, nI3, nLz))
        self._dI_tab = numpy.zeros((nE, nI3, nLz, 3 + (self._delta_rule is not None)))
        varies = self._delta_rule is not None
        if varies:
            self._setup_delta_tab()
        for iL, Lz in enumerate(self._Lzgrid):
            Ec = self._circular(Lz)["Ec"]
            for iE, wE in enumerate(self._wEs):
                E, faces = self._E_of_xL(wE**2, Lz)[0], None
                # the nodes of (E, L_z) are placed in one Staeckel model
                # (up to their reference curves); with a focal length that
                # follows the tori that is the model of the rule's focal
                # length at the middle of the third-integral range, and each
                # node then moves to its own (_own_models)
                with self._in_chart(
                    E, Lz, delta=self._delta_col[iE, iL, 0] if varies else None
                ):
                    if wE == 0.0:
                        ush = self._circular(Lz)["uc"]
                        Ipl = Ish = self._I3_planar(Ec, Lz)
                        I3s[iE, :, iL] = Ipl
                    else:
                        ush = self._ushell(E, Lz)
                        Ipl, Ish = self._I3_planar(E, Lz), self._I3_shell(E, Lz, ush)
                        starts = self._I3_starts(E, Lz, self._ys)
                        for iI, y in enumerate(self._ys):
                            I3s[iE, iI, iL], self._dI_tab[iE, iI, iL] = self._I3_of_y(
                                E, Lz, y, starts[iI], node=not varies
                            )
                        faces = self._axis_reach_faces(E, Lz)[:4]
                self._E_tab[iE, iL] = E
                self._ush_tab[iE, :, iL] = ush
                self._Ipl_tab[iE, :, iL], self._Ish_tab[iE, :, iL] = Ipl, Ish
                if varies:
                    self._own_models(iE, iL, I3s, faces)
        # the tori in (iE, iI, iL) order
        self._Es = numpy.repeat(self._E_tab[:, None, :], nI3, axis=1).flatten()
        self._Lzs = numpy.tile(self._Lzgrid, nE * nI3)
        self._I3s = I3s.flatten()
        return None

    def _setup_delta_tab(self):
        """The focal lengths of the nodes of a family whose focal length
        follows its tori, from the rule delta(E, L_z, y) called with the
        arrays of all nodes at once, and their partials with respect to the
        grid coordinates (x, y, L_z), by central differences of the rule
        (FINITE DIFFERENCES IN THE SET-UP, of the user's callable alone);
        those of delta='fit' are closed forms (_setup_delta_fit)"""
        if self._delta_rule == "fit":
            return self._setup_delta_fit()
        Egrid = numpy.array(
            [[self._E_of_xL(wE**2, Lz) for Lz in self._Lzgrid] for wE in self._wEs]
        )
        E, dE_dx, dE_dL = (
            numpy.repeat(Egrid[:, None, :, k], len(self._ys), axis=1) for k in range(3)
        )
        y = self._ys[None, :, None] + 0.0 * E
        L = self._Lzgrid[None, None, :] + 0.0 * E

        def rule(*a):
            return conversion.parse_length(self._delta_rule(*a), ro=self._ro) + 0.0 * E

        self._delta_tab = rule(E, L, y)
        if not numpy.all(self._delta_tab > 0.0):
            raise ValueError("the callable delta= has to return positive focal lengths")
        hE, hy, hL = 1e-5 * (E[-1:] - E[:1]), 1e-5, 1e-6 * L

        def partials(yy):
            return (
                (rule(E + hE, L, yy) - rule(E - hE, L, yy)) / (2.0 * hE),
                (rule(E, L + hL, yy) - rule(E, L - hL, yy)) / (2.0 * hL),
            )

        DE, DL = partials(y)
        Dy = (rule(E, L, y + hy) - rule(E, L, y - hy)) / (2.0 * hy)
        self._ddelta_tab = numpy.stack([DE * dE_dx, Dy, DL + DE * dE_dL], axis=-1)
        # the placement's model of each (E, L_z) and its partials (_setup_grid)
        mid = 0.5 + 0.0 * y
        self._delta_col = numpy.stack([rule(E, L, mid), *partials(mid)], axis=-1)[:, 0]
        return None

    def _own_models(self, iE, iL, I3s, faces):
        """Move the nodes of an (E, L_z), placed in one model (_setup_grid),
        to the models of their own focal lengths: a node keeps its fraction
        f of the way of I_3 from the planar to the shell orbit's. Its I_3
        and that I_3's partials in (E, y, L_z, delta) at fixed grid
        coordinates follow from those of f (faces: the planar and shell I_3
        of the placement's model with partials; None on the circular face)"""
        E, Lz = self._E_tab[iE, iL], self._Lzgrid[iL]
        if faces is not None:
            Ipl, dIpl, Ish, dIsh = faces
            dc = self._delta_col[iE, iL, 1:]

            def along(d):
                # -> (d/dE, d/dy, d/dL_z) along the placement's delta(E, L_z)
                return numpy.array(
                    [d[0] + d[-1] * dc[0], d[1] * (len(d) == 4), d[-2] + d[-1] * dc[1]]
                )

        for iI in range(len(self._ys)):
            with self._in_chart(E, Lz, False, self._delta_tab[iE, iI, iL]):
                if faces is None:
                    ush = self._circular(Lz)["uc"]
                    Ipn = Isn = I3s[iE, iI, iL] = self._I3_planar(E, Lz)
                else:
                    f = (I3s[iE, iI, iL] - Ipl) / (Ish - Ipl)
                    df = (
                        along(self._dI_tab[iE, iI, iL])
                        - along(dIpl)
                        - f * (along(dIsh) - along(dIpl))
                    ) / (Ish - Ipl)
                    ush, d2 = self._ushell(E, Lz), self._delta**2
                    sh2 = numpy.sinh(ush) ** 2
                    Ipn, Isn = self._I3_planar(E, Lz), self._I3_shell(E, Lz, ush)
                    dIplD, dIshD = self._faces_ddelta(ush, Lz)
                    gp = numpy.array([-1.0, 0.0, Lz / d2, dIplD[0]])
                    gs = numpy.array([sh2, 0.0, -Lz / (d2 * sh2), dIshD[0]])
                    I3s[iE, iI, iL] = Ipn + f * (Isn - Ipn)
                    self._dI_tab[iE, iI, iL] = (
                        gp + f * (gs - gp) + (Isn - Ipn) * numpy.append(df, 0.0)
                    )
            self._ush_tab[iE, iI, iL] = ush
            self._Ipl_tab[iE, iI, iL], self._Ish_tab[iE, iI, iL] = Ipn, Isn
        return None

    def _delta_features(self, x, y, L):
        """The tensor monomials x^i y^j lam^k, up to the degrees delta_deg,
        of the grid coordinates of the tori (x, y, L_z), with lam the
        position of ln L_z in the grid's range, and their partials with
        respect to (x, y, L_z): the basis of the fitted focal length"""
        span = numpy.log(self._Lzgrid[-1] / self._Lzgrid[0])
        lam = numpy.log(L / self._Lzgrid[0]) / span
        P, dP = (
            [
                numpy.array(
                    [t**i if k == 0 else i * t ** max(i - 1, 0) for i in range(d + 1)]
                )
                for t, d in zip((x, y, lam), self._delta_deg)
            ]
            for k in range(2)
        )

        def tensor(a, b, c):
            return numpy.einsum("in,jn,kn->nijk", a, b, c).reshape(len(x), -1)

        return (
            tensor(*P),
            tensor(dP[0], P[1], P[2]),
            tensor(P[0], dP[1], P[2]),
            tensor(P[0], P[1], dP[2]) / (L * span)[:, None],
        )

    def _delta_fitted(self, x, y, L):
        """The fitted focal length of the tori (x, y, L_z) and its partials
        with respect to them: exp[s(P)], with P the fitted tensor
        polynomial (_delta_features) and s a soft limit to [b0, b1],
        s = P - m ln[1 + e^((P - b1) / m)] + m ln[1 + e^((b0 - P) / m)]:
        P itself (to e^(-10) of m) wherever P stays within the range of the
        fitted optima, and bent smoothly onto b0 or b1, which lie a quarter
        of that range beyond it, where P leaves it by more"""
        coef, b0, b1, m = self._delta_fit
        F = self._delta_features(x, y, L)
        P = F[0] @ coef
        D = numpy.exp(
            P
            - m * numpy.logaddexp(0.0, (P - b1) / m)
            + m * numpy.logaddexp(0.0, (b0 - P) / m)
        )
        ds = 0.5 * (numpy.tanh(0.5 * (b1 - P) / m) - numpy.tanh(0.5 * (b0 - P) / m))
        return D, [D * ds * (G @ coef) for G in F[1:]]

    def _setup_delta_fit(self):
        """The focal lengths of the nodes for delta='fit', with their
        partials in closed form. On a sub-grid of the family's box (2 deg +
        1 tori along each coordinate: from a tenth of the way up in the
        energy variable, where the spread is too small to fit, between
        0.15 and 0.7 of the third-integral range in w_I (the spread
        vanishes on the planar and on the shell orbits, and with the
        reference curve on the torus's own spheroid the optima close to
        the shell orbit are noise), and
        uniform in ln L_z), the focal length that minimizes the spread of
        the true Hamiltonian over each torus (_focal_fit: an optimization,
        on the arrays of all these tori at once); then the least-squares
        tensor polynomial of the degrees delta_deg for its logarithm in the
        grid coordinates (_delta_features), with each torus at the grid's
        third-integral variable of its own model (_focal_y), without the
        tori whose optimum lies far from the fit of the others, limited
        softly to about the range of their optima (_delta_fitted). The fit
        is then held fixed: the family's nodes and derivative tables are
        those of the limited polynomial"""
        deg, box = self._delta_deg, self._box
        w, f, L = (
            t.ravel()
            for t in numpy.meshgrid(
                box[1][0]
                + (box[1][1] - box[1][0]) * numpy.linspace(0.1, 1.0, 2 * deg[0] + 1),
                numpy.sin(
                    0.5
                    * numpy.pi
                    * (
                        box[2][0]
                        + (box[2][1] - box[2][0])
                        * numpy.linspace(0.15, 0.7, 2 * deg[1] + 1)
                    )
                )
                ** 2,
                numpy.geomspace(self._Lzgrid[0], self._Lzgrid[-1], 2 * deg[2] + 1),
                indexing="ij",
            )
        )
        E = numpy.array([self._E_of_xL(ww**2, ll)[0] for ww, ll in zip(w, L)])
        args = (self._chart_pot, E, L, f, *self._planar_radii(E, L), self._u0_rule)
        D = _focal_fit(*args)
        # (the grid's variable is that of the model in which the nodes of
        # an (E, L_z) are placed, here that of its middle torus's optimum)
        nf = 2 * deg[1] + 1
        Dc = numpy.repeat(D.reshape(-1, nf, 2 * deg[2] + 1)[:, nf // 2], nf, axis=0)
        y = _focal_y(
            *args[:4],
            Dc.reshape(-1, nf, Dc.shape[1]).swapaxes(0, 1).ravel(),
            *args[4:],
            self._axisw,
        )
        F, lnD = self._delta_features(w**2, y, L)[0], numpy.log(D)
        keep = numpy.ones(len(D), dtype="bool")
        for _ in range(3):
            # (twice without the tori whose optimum lies far from the fit:
            # the spread of an extreme torus can have two minima of the
            # same depth, far apart). The polynomial is fitted to ln delta
            # itself and limited softly (_delta_fitted) to the range of the
            # kept optima widened on each side by a quarter of it (and by
            # at least 0.05), over a tenth of that widening: the limit
            # leaves the polynomial as it is wherever it stays within the
            # optima's range, so that it adds nothing for the tables to
            # resolve, and holds it where it would run away between or
            # beyond the fitted tori
            lo, hi = numpy.amin(lnD[keep]), numpy.amax(lnD[keep])
            a = 0.25 * (hi - lo) + 0.05
            coef = numpy.linalg.lstsq(F[keep], lnD[keep], rcond=None)[0]
            self._delta_fit = (coef, lo - a, hi + a, 0.1 * a)
            res = numpy.fabs(numpy.log(self._delta_fitted(w**2, y, L)[0]) - lnD)
            keep = res < max(0.25, 5.0 * numpy.median(res))
        x, y, L = numpy.meshgrid(self._wEs**2, self._ys, self._Lzgrid, indexing="ij")
        self._delta_tab, dD = self._delta_fitted(x.ravel(), y.ravel(), L.ravel())
        self._delta_tab = self._delta_tab.reshape(x.shape)
        self._ddelta_tab = numpy.stack([d.reshape(x.shape) for d in dD], axis=-1)
        # the model in which the nodes of each (E, L_z) are placed: the fit
        # at y = 1/2, with its partials in (E, L_z) at fixed L_z and E
        xc, Lc = x[:, 0].ravel(), L[:, 0].ravel()
        Dc, (Dx, _, DL) = self._delta_fitted(xc, 0.5 + 0.0 * xc, Lc)
        dE = numpy.array([self._E_of_xL(xx, ll)[1:] for xx, ll in zip(xc, Lc)])
        self._delta_col = numpy.stack(
            [Dc, Dx / dE[:, 0], DL - Dx * dE[:, 1] / dE[:, 0]], axis=-1
        ).reshape(x.shape[0], x.shape[2], 3)
        return None

    def _planar_radii(self, E, L):
        """The pericentres and apocentres of the planar orbits (E, L_z), for
        arrays of them (inside Rinf)"""
        Rc = numpy.array([self._circular_RE(ll)[0] for ll in L])

        def above(R):
            # the planar orbit's energy above the effective potential
            return (
                E
                - evaluatePotentials(self._chart_pot, R, 0.0 * R, use_physical=False)
                - L**2 / (2.0 * R**2)
            )

        return (
            _bisect(above, 1e-4 * Rc, Rc),
            _bisect(lambda R: -above(R), Rc, 1.01 * self._Rinf + 0.0 * Rc),
        )

    def _target_deltas(self, R, vR, vT, z, vz):
        """The focal lengths in which the points of a target are labelled
        for its box, when the focal length follows the tori (else None for
        each): the rule at the middle of the third-integral range of each
        point's (E, L_z), called with the arrays of all points, or for
        delta='fit', which is fitted on the box, the optimum of the torus
        halfway between the planar and the shell orbit (the box is padded,
        so that the difference to the torus's own model is immaterial)"""
        if self._delta_rule is None:
            return [None] * len(R)
        E = 0.5 * (vR**2 + vT**2 + vz**2) + evaluatePotentials(
            self._chart_pot, R, z, use_physical=False
        )
        L, half = R * numpy.fabs(vT), 0.5 + 0.0 * R
        if self._delta_rule == "fit":
            return _focal_fit(
                self._chart_pot, E, L, half, *self._planar_radii(E, L), self._u0_rule
            )
        return (
            conversion.parse_length(self._delta_rule(E, L, half), ro=self._ro) + 0.0 * E
        )

    def _node_delta(self, ii):
        """The ii-th torus's focal length when it follows the tori, else None"""
        return None if self._delta_rule is None else float(self._delta_tab.flat[ii])

    def _target_box(self, Lmin, Lmax):
        """The padded (L_z, w_E, w_I) box that localizes the grid on the
        target's tori. Every target point is labelled through the relations
        the node lattice uses (in its own chart for an adaptive reference
        curve); the box is the labels' range, padded by target_pad times
        the spread on both ends of each axis and at least target_minwidth
        of each axis's default span wide. An end that reaches a degenerate
        face (the circular, planar or shell orbits) is the face itself,
        because the grid must reach a face exactly or not at all, whereas
        the L_z ends are anchors that the box may exceed."""
        R, vR, vT, z, vz = self._target
        Lzs = R * numpy.fabs(vT)
        deltas = self._target_deltas(R, vR, vT, z, vz)
        wEs, wIs = numpy.empty(len(Lzs)), numpy.empty(len(Lzs))
        for i, (Rp, vRp, vTp, zp, vzp, Lz) in enumerate(zip(R, vR, vT, z, vz, Lzs)):
            kin = 0.5 * (vRp**2 + vTp**2 + vzp**2)
            # the point's chart, placed with the raw potential's energy: the
            # reference curve varies smoothly, so the model error in the
            # energy is immaterial for the box
            Eraw = kin + evaluatePotentials(self._chart_pot, Rp, zp, use_physical=False)
            with self._in_chart(Eraw, Lz, False, deltas[i]):
                I3 = self._target_I3(Eraw, Lz, Rp, vRp, zp, vzp)
            with self._in_chart(Eraw, Lz, I3, deltas[i]):
                E = kin + evaluatePotentials(
                    self._staeckelwrap, Rp, zp, use_physical=False
                )
                Ec, Emax = self._circular(Lz)["Ec"], self._Emax(Lz)
                if E > Emax:
                    raise ValueError(
                        "a target point's energy lies above the energies the grid "
                        "can cover; increase Rinf"
                    )
                if E <= Ec + 1e-12 * (1.0 + numpy.fabs(Ec)):
                    # the circular orbit, where every I_3 label coincides and
                    # the w_I direction is free
                    wEs[i], wIs[i] = 0.0, 0.5
                    continue
                R0c, _, R0m, _ = self._R0_edges(Lz)
                x = numpy.log(self._R0(E, R0c, R0m) / R0c) / numpy.log(R0m / R0c)
                wEs[i] = numpy.sqrt(min(max(x, 0.0), 1.0))
                # the point's third integral, from the u equation of the
                # separated Hamilton-Jacobi relation
                u, v = coords.Rz_to_uv(Rp, zp, delta=self._delta)
                sh = numpy.sinh(u)
                pu = self._delta * (
                    vRp * numpy.cosh(u) * numpy.sin(v) + vzp * sh * numpy.cos(v)
                )
                I3 = (
                    E * sh**2
                    - float(self._U(u))
                    - (pu**2 + Lz**2 / sh**2) / (2.0 * self._delta**2)
                )
                Ipl = self._I3_planar(E, Lz)
                Ish = self._I3_shell(E, Lz, self._ushell(E, Lz))
                y = self._y_of_I3(E, Lz, min(max(I3, Ipl), Ish))[0]
                wIs[i] = (
                    2.0 / numpy.pi * numpy.arcsin(numpy.sqrt(min(max(y, 0.0), 1.0)))
                )

        def padded(vals, lo0, hi0, snap):
            lo, hi = float(numpy.min(vals)), float(numpy.max(vals))
            lo, hi = (
                lo - self._target_pad * (hi - lo),
                hi + self._target_pad * (hi - lo),
            )
            minw = self._target_minwidth * (hi0 - lo0)
            if hi - lo < minw:
                mid = 0.5 * (lo + hi)
                lo, hi = mid - 0.5 * minw, mid + 0.5 * minw
            if snap:
                # an end that reaches a face is the face
                return (lo0 if lo <= lo0 else lo, hi0 if hi >= hi0 else hi)
            # the L_z ends are anchors, not faces: the box may exceed them,
            # and only the lower end is guarded, against L_z <= 0
            return (max(lo, 0.5 * float(numpy.min(vals))), hi)

        Lzbox = padded(Lzs, Lmin, Lmax, False)
        if rl(self._staeckelwrap, Lzbox[1], use_physical=False) >= self._Rinf:
            raise ValueError(
                "the target's padded L_z range reaches circular orbits beyond "
                "Rinf, which must lie beyond every circular orbit of the grid; "
                "increase Rinf or reduce target_pad"
            )
        return (Lzbox, padded(wEs, 0.0, 1.0, True), padded(wIs, 0.0, 1.0, True))

    def _target_I3(self, E, Lz, R, vR, z, vz):
        """The third integral of a target point at energy E for placing its
        chart, from the u equation of the separated Hamilton-Jacobi
        relation, which does not involve the reference curve; clipped to
        the tori of (E, L_z). None for a rule that does not use it"""
        if not self._u0_third or self._on_circular(E, Lz):
            return None
        u, v = coords.Rz_to_uv(R, z, delta=self._delta)
        sh = numpy.sinh(u)
        pu = self._delta * (vR * numpy.cosh(u) * numpy.sin(v) + vz * sh * numpy.cos(v))
        I3 = (
            E * sh**2
            - float(self._U(u))
            - (pu**2 + Lz**2 / sh**2) / (2.0 * self._delta**2)
        )
        Ipl = self._I3_planar(E, Lz)
        Ish = self._I3_shell(E, Lz, self._ushell(E, Lz))
        return min(max(I3, Ipl), Ish)

    def _check_circular(self):
        """Raise for the circular orbit of a family whose grid does not reach
        the circular face"""
        if self._wEs[0] != 0.0:
            raise ValueError(
                "the circular orbit lies outside the family's grid, whose box does "
                "not reach the circular face"
            )
        return None

    def _Emax(self, Lz):
        """The grid's top energy at |L_z|: that of the planar orbit with
        apocentre Rinf"""
        return self._Phiinf + Lz**2 / (2.0 * self._Rinf**2)

    def _R0(self, E, Rlo, Rhi):
        """The apocentre of the planar orbit without angular momentum at
        energy E, Phi(R_0, 0) = E, bracketed by [Rlo, Rhi]"""
        return brentq(
            lambda R: (
                evaluatePotentials(self._staeckelwrap, R, 0.0, use_physical=False) - E
            ),
            Rlo,
            Rhi,
            xtol=1e-14,
        )

    def _RdPhi(self, R):
        """R dPhi/dR in the plane, the derivative of the potential with
        respect to ln R"""
        return -R * evaluateRforces(self._staeckelwrap, R, 0.0, use_physical=False)

    def _R0_edges(self, Lz):
        """The apocentres R_0 of the planar orbit without angular momentum
        at the grid's bottom and top energies at |L_z|, with R dPhi/dR at
        each (the level-set rule gives d ln R_0/dL_z = (dE/dL_z)/[R_0
        Phi'(R_0)]); cached. The bottom one lies above R_c, since E_c
        exceeds Phi(R_c, 0) by the centrifugal term, and the top one just
        beyond Rinf, by L_z^2 / (2 Rinf^2) in energy"""
        if Lz not in self._R0_cache:
            circ = self._circular(Lz)
            R0c = self._R0(circ["Ec"], circ["Rc"], self._Rinf)
            R0m = self._R0(self._Emax(Lz), self._Rinf, 10.0 * self._Rinf)
            self._R0_cache[Lz] = (R0c, self._RdPhi(R0c), R0m, self._RdPhi(R0m))
        return self._R0_cache[Lz]

    def _E_of_xL(self, x, Lz):
        """The grid's energy variable, analytic: E = Phi(R, 0) with ln R =
        (1 - x) ln R_0(E_c) + x ln R_0(E_max), uniform in the logarithm of
        the apocentre of the planar orbit without angular momentum; returns
        E, dE/dx at fixed L_z, and dE/dL_z at fixed x, the latter through
        the two edges' apocentres by the level-set rule, formed so that it
        is exactly dE_c/dL_z at the circular face and dE_max/dL_z at the
        top"""
        R0c, g0c, R0m, g0m = self._R0_edges(Lz)
        R = R0c * (R0m / R0c) ** x
        g = self._RdPhi(R)
        Omc = self._circular(Lz)["Omc"]
        return (
            evaluatePotentials(self._staeckelwrap, R, 0.0, use_physical=False),
            g * numpy.log(R0m / R0c),
            (1.0 - x) * Omc * (g / g0c) + x * (Lz / self._Rinf**2) * (g / g0m),
        )

    def _E_of_xL_array(self, x, Lz):
        """_E_of_xL for arrays of points, each with its own |L_z|: the
        circular radius and the two apocentre edges from a vectorized
        root-find on the potential's midplane, which the Staeckel wrapper
        reproduces exactly, bracketed by the grid's extreme circular radii
        and Rinf"""
        wrap = self._chart_pot

        def phi(R):
            return evaluatePotentials(wrap, R, 0.0, use_physical=False)

        def dphi(R):
            return -evaluateRforces(wrap, R, 0.0, use_physical=False)

        def d2phi(R):
            return evaluateR2derivs(wrap, R, 0.0, use_physical=False)

        Rlo = 0.9 * self._circular(self._Lzgrid[0])["Rc"] + 0.0 * Lz
        Rhi = 1.1 * self._circular(self._Lzgrid[-1])["Rc"] + 0.0 * Lz
        Rc = _solve_increasing(
            lambda R: R**3 * dphi(R) - Lz**2,
            lambda R: R**2 * (3.0 * dphi(R) + R * d2phi(R)),
            Rlo,
            Rhi,
        )
        Ec = phi(Rc) + Lz**2 / (2.0 * Rc**2)
        Emax = self._Phiinf + Lz**2 / (2.0 * self._Rinf**2)
        Rinf = self._Rinf + 0.0 * Lz
        R0c = _solve_increasing(lambda R: phi(R) - Ec, dphi, Rc, Rinf)
        R0m = _solve_increasing(lambda R: phi(R) - Emax, dphi, Rinf, 10.0 * Rinf)
        g0c, g0m = R0c * dphi(R0c), R0m * dphi(R0m)
        R = R0c * (R0m / R0c) ** x
        g = R * dphi(R)
        return (
            phi(R),
            g * numpy.log(R0m / R0c),
            (1.0 - x) * (Lz / Rc**2) * (g / g0c) + x * (Lz / self._Rinf**2) * (g / g0m),
        )

    def _setup_tori_interp(self):
        """The family's tables and their exact first partials at every
        node, in the variables in which each quantity is smooth up to the
        degenerate edges: the actions and the u midpoint in (x = w_E^2,
        y = sin^2(pi w_I/2), L_z), in which they are smooth (the actions
        vanish linearly at the edges); the u half-width and the u map in
        (w_E, s_u = cos(pi w_I/2), L_z), the v half-width and the v map in
        (w_E, s_v = sin(pi w_I/2), L_z), in which they vanish linearly at
        their edges, each half-width as the variable of _halfwidth, which
        keeps the libration's inner turning point off the axis and
        accurate relative to its distance from it. The partials come from the node's period matrix, the
        level-set rule, and the map slopes, chained through the grid's
        analytic (E, I_3)(w_E, w_I, L_z) relation; on the degenerate faces
        they are the harmonic limits."""
        nE, nI, nL = len(self._wEs), len(self._wIs), len(self._Lzgrid)
        npt, varies = self._npt, self._delta_rule is not None
        # (with the focal length as a fourth table when it follows the tori)
        A_tab, A_dx, A_dy, A_dL = (
            numpy.zeros((nE, nI, nL, 3 + varies)) for _ in range(4)
        )
        U_tab, U_dwE, U_dsu, U_dL = (
            numpy.zeros((nE, nI, nL, 1 + npt)) for _ in range(4)
        )
        V_tab, V_dwE, V_dsv, V_dL = (
            numpy.zeros((nE, nI, nL, 1 + self._nptv)) for _ in range(4)
        )
        perr_u, perr_v = numpy.zeros((nE, nI, nL)), numpy.zeros((nE, nI, nL))
        for ii in _progress(range(nE * nI * nL), "node tables", self._progressbar):
            iE, rest = divmod(ii, nI * nL)
            iI, iL = divmod(rest, nL)
            wE, wI, Lz = self._wEs[iE], self._wIs[iI], self._Lzgrid[iL]
            with self._in_chart(
                self._E_tab[iE, iL], Lz, self._I3s[ii], self._node_delta(ii)
            ):
                y, d2 = self._ys[iI], self._delta**2
                su, sv = numpy.cos(0.5 * numpy.pi * wI), numpy.sin(0.5 * numpy.pi * wI)
                if True:
                    node = self._node_tables(ii)
                    perr_u[iE, iI, iL], perr_v[iE, iI, iL] = (
                        node["perr_u"],
                        node["perr_v"],
                    )
                    A_tab[iE, iI, iL, :3] = [node["jr"], node["jz"], node["uc"]]
                    if varies:
                        dD = self._ddelta_tab[iE, iI, iL]
                        A_tab[iE, iI, iL, 3] = self._delta
                        (
                            A_dx[iE, iI, iL, 3],
                            A_dy[iE, iI, iL, 3],
                            A_dL[iE, iI, iL, 3],
                        ) = dD
                    U_tab[iE, iI, iL, 0], U_tab[iE, iI, iL, 1:] = (
                        node["wu"],
                        node["Dmu"],
                    )
                    V_tab[iE, iI, iL, 0], V_tab[iE, iI, iL, 1:] = (
                        node["wv"],
                        node["Dmv"],
                    )
                    circ = self._circular(Lz)
                    Ec = circ["Ec"]
                    E, ush = self._E_tab[iE, iL], self._ush_tab[iE, iI, iL]
                    Ipl, Ish = self._Ipl_tab[iE, iI, iL], self._Ish_tab[iE, iI, iL]
                    _, dE_dx, dE_dL = self._E_of_xL(wE**2, Lz)
                    if wE == 0.0:
                        # the circular face: both librations harmonic, with
                        # J = (E - E_c) times the fraction the third integral
                        # assigns to each, over its frequency; the
                        # half-widths grow linearly in w_E out of the edge
                        # and the maps vanish (their slopes come from the
                        # next two rows, below)
                        uc = circ["uc"]
                        sqbu = numpy.sqrt(-0.5 * float(self._d2Wu(uc, Ec, Lz)))
                        sqbv = numpy.sqrt(
                            -0.5 * float(self._d2Wv(0.5 * numpy.pi, Ec, Lz))
                        )
                        ch2 = numpy.cosh(uc) ** 2
                        A_dx[iE, iI, iL, :3] = [
                            dE_dx * (1.0 - y) * d2 * ch2 / sqbu,
                            dE_dx * y * d2 * ch2 / sqbv,
                            0.0,
                        ]
                        # the circular radius moves with L_z at the rate the
                        # epicycle sets (L_z^2 = R_c^3 Phi'(R_c) gives dR_c/dL_z
                        # = 2 L_z / [R_c^3 kappa^2])
                        A_dL[iE, iI, iL, 2] = (
                            2.0
                            * Lz
                            / (circ["Rc"] ** 3 * circ["kappa"] ** 2)
                            / (self._delta * numpy.cosh(uc))
                        )
                        if varies:
                            # the circular orbit's u changes with the focal
                            # length from node to node
                            A_dy[iE, iI, iL, 2] = -numpy.tanh(uc) / self._delta * dD[1]
                            A_dL[iE, iI, iL, 2] -= numpy.tanh(uc) / self._delta * dD[2]
                        U_dwE[iE, iI, iL, 0] = (
                            numpy.sqrt(2.0 * d2 * ch2 * dE_dx) / sqbu * su
                        )
                        V_dwE[iE, iI, iL, 0] = (
                            numpy.sqrt(2.0 * d2 * ch2 * dE_dx) / sqbv * sv
                        )
                        continue
                    dI_dE, dI_dy, dI_dL, *dI_dD = self._dI_tab[iE, iI, iL]

                    def chain(q):
                        # (dq/dE, dq/dI_3, dq/dL_z) -> (dq/dx, dq/dy, dq/dL_z)
                        # at fixed grid coordinates
                        qE, qI, qL = q[..., 0], q[..., 1], q[..., 2]
                        qx = (qE + qI * dI_dE) * dE_dx
                        out = (qx, qI * dI_dy, qL + qI * dI_dL + qx / dE_dx * dE_dL)
                        if not varies:
                            return out
                        # plus the derivative along the focal length at
                        # fixed grid coordinates times the rule's, when the
                        # focal length follows the tori
                        qD = q[..., 3] + qI * dI_dD[0]
                        return tuple(o + qD * d for o, d in zip(out, dD))

                    M, dsup = node["M"], node["dsup"]
                    udeg, vdeg = node["dDmu"] is None, node["dDmv"] is None
                    Mv, dwv, dDmv = M[1], dsup[2], node["dDmv"]
                    if "du0" in node:
                        # the Staeckel model changes from node to node with
                        # its reference curve: the node values' derivatives
                        # along the grid are those within the node's model
                        # plus the derivative along u0 at fixed integrals
                        # times the rule's own (the v libration's tables,
                        # since U(u) does not depend on u0, and the u map,
                        # which depends on it through the auxiliary torus)
                        du0 = self._u0_grad(E, Lz, self._I3s[ii])
                        Mv = Mv + node["du0"][0] * du0
                        dwv = dwv + node["du0"][1] * du0
                        dDmv = dDmv + node["du0"][2][:, None] * du0[None, :]
                        if "du0u" in node:
                            node["dDmu"] = (
                                node["dDmu"] + node["du0u"][:, None] * du0[None, :]
                            )
                    if udeg:
                        # the shell face: the u libration harmonic at the
                        # shell u, its action and half-width vanishing
                        # linearly in (1 - y) and in s_u
                        sqbu = numpy.sqrt(-0.5 * float(self._d2Wu(ush, E, Lz)))
                        A_dy[iE, iI, iL, 0] = -d2 * dI_dy / sqbu
                        U_dsu[iE, iI, iL, 0] = numpy.sqrt(2.0 * d2 * dI_dy) / sqbu
                    else:
                        (
                            A_dx[iE, iI, iL, 0],
                            A_dy[iE, iI, iL, 0],
                            A_dL[iE, iI, iL, 0],
                        ) = chain(M[0])
                        r = node["wu"] / node["uc"]
                        U_tab[iE, iI, iL, 0] = node["uc"] * numpy.arctanh(r)
                        dlam = numpy.arctanh(r) * dsup[0] + (dsup[1] - r * dsup[0]) / (
                            1.0 - r**2
                        )
                        qx, qy, qL = chain(numpy.vstack((dlam[None, :], node["dDmu"])))
                        U_dwE[iE, iI, iL] = 2.0 * wE * qx
                        U_dsu[iE, iI, iL] = -2.0 * su * qy
                        U_dL[iE, iI, iL] = qL
                    if vdeg:
                        # the planar face: the v libration harmonic about
                        # the midplane
                        sqbv = numpy.sqrt(
                            -0.5 * float(self._d2Wv(0.5 * numpy.pi, E, Lz))
                        )
                        A_dy[iE, iI, iL, 1] = d2 * dI_dy / sqbv
                        V_dsv[iE, iI, iL, 0] = numpy.sqrt(2.0 * d2 * dI_dy) / sqbv
                    else:
                        (
                            A_dx[iE, iI, iL, 1],
                            A_dy[iE, iI, iL, 1],
                            A_dL[iE, iI, iL, 1],
                        ) = chain(Mv)
                        V_tab[iE, iI, iL, 0] = numpy.arcsinh(numpy.tan(node["wv"]))
                        qx, qy, qL = chain(
                            numpy.vstack((dwv[None, :] / numpy.cos(node["wv"]), dDmv))
                        )
                        V_dwE[iE, iI, iL] = 2.0 * wE * qx
                        V_dsv[iE, iI, iL] = 2.0 * sv * qy
                        V_dL[iE, iI, iL] = qL
                    A_dx[iE, iI, iL, 2], A_dy[iE, iI, iL, 2], A_dL[iE, iI, iL, 2] = (
                        chain(dsup[0])
                    )
        # the maps' coefficients vanish at the circular edge and grow
        # linearly in w_E out of it (the first anharmonic correction of a
        # libration is linear in its amplitude); their slope there is that
        # of the quartic through zero and the next two rows, as the 1D
        # family does at its harmonic bottom; likewise the u map at the
        # shell face in s_u and the v map at the planar face in s_v. The u
        # libration's midpoint leaves the shell u quadratically in the
        # half-width (the cubic term of the momentum function skews the
        # libration), i.e. linearly in x at the circular face and in y at
        # the shell face, at rates the same quartics through the next two
        # rows give
        xs = self._wEs**2
        sus = numpy.cos(0.5 * numpy.pi * self._wIs)
        svs = numpy.sin(0.5 * numpy.pi * self._wIs)
        if self._wEs[0] == 0.0:
            A_dx[0, :, :, 2] = slope_at_zero(
                xs[1:3], A_tab[1:3, :, :, 2] - A_tab[0, :, :, 2], A_dx[1:3, :, :, 2]
            )
            U_dwE[0, :, :, 1:] = slope_at_zero(
                self._wEs[1:3], U_tab[1:3, :, :, 1:], U_dwE[1:3, :, :, 1:]
            )
            V_dwE[0, :, :, 1:] = slope_at_zero(
                self._wEs[1:3], V_tab[1:3, :, :, 1:], V_dwE[1:3, :, :, 1:]
            )
        if self._wIs[-1] == 1.0:
            A_dy[:, -1, :, 2] = slope_at_zero(
                self._ys[[-2, -3]] - 1.0,
                numpy.moveaxis(A_tab[:, [-2, -3], :, 2:3], 1, 0)[..., 0]
                - A_tab[:, -1, :, 2][None],
                numpy.moveaxis(A_dy[:, [-2, -3], :, 2:3], 1, 0)[..., 0],
            )
            U_dsu[:, -1, :, 1:] = slope_at_zero(
                sus[[-2, -3]],
                numpy.moveaxis(U_tab[:, [-2, -3], :, 1:], 1, 0),
                numpy.moveaxis(U_dsu[:, [-2, -3], :, 1:], 1, 0),
            )
        if self._wIs[0] == 0.0:
            V_dsv[:, 0, :, 1:] = slope_at_zero(
                svs[[1, 2]],
                numpy.moveaxis(V_tab[:, [1, 2], :, 1:], 1, 0),
                numpy.moveaxis(V_dsv[:, [1, 2], :, 1:], 1, 0),
            )
        self._family_format = self._FAMILY_FORMAT + varies
        self._perr_u_tab, self._perr_v_tab = perr_u, perr_v
        self._A_tab, self._A_dx, self._A_dy, self._A_dL = A_tab, A_dx, A_dy, A_dL
        self._U_tab, self._U_dwE, self._U_dsu, self._U_dL = U_tab, U_dwE, U_dsu, U_dL
        self._V_tab, self._V_dwE, self._V_dsv, self._V_dL = V_tab, V_dwE, V_dsv, V_dL
        self._warn_unresolved(perr_u.flatten(), self._Es, self._Lzs, self._I3s, "u")
        self._warn_unresolved(perr_v.flatten(), self._Es, self._Lzs, self._I3s, "v")
        # a torus whose tables are not finite (its maps so far from
        # converged, or the potential's forces so ill-behaved along it,
        # that its slopes are not) cannot be interpolated: say which
        bad = ~numpy.all(
            [
                numpy.isfinite(t).reshape(len(self._Es), -1).all(axis=1)
                for t in (A_tab, A_dx, A_dy, A_dL, U_tab, U_dwE, U_dsu, U_dL)
                + (V_tab, V_dwE, V_dsv, V_dL)
            ],
            axis=0,
        )
        if numpy.any(bad):
            raise ValueError(
                "The tables of the (E, L_z, I_3) tori {} are not finite, so that "
                "the family cannot be interpolated: their maps are far from "
                "converged (see the warnings above; increase mm_npt and, with "
                "it, mm_nta) or the potential's forces are not finite along "
                "them; or keep the grid away from them".format(
                    ", ".join(
                        f"({E:g}, {Lz:g}, {I3:g})"
                        for E, Lz, I3 in zip(
                            self._Es[bad], self._Lzs[bad], self._I3s[bad]
                        )
                    )
                )
            )
        self._rebuild_interp()
        return None

    def _rebuild_interp(self):
        """(Re)build the interpolants from the stored tables: Hermite
        interpolants, quintic along the energy variable and cubic along the
        other two, all with their first partials exact at the nodes. A node
        keeps the coefficients of each map up to the last that is not zero
        in its tables (_nharm_node_u, _nharm_node_v; the v map's are those
        of its even harmonics), and the interpolant's own mixed and second
        derivatives are set to zero beyond that number too, so that a
        cell's interpolant vanishes exactly beyond the largest number of
        its eight nodes; that number, rounded up to the next of a fixed set
        (_nharm_buckets_u: 16, 32, 64, ... below mm_npt, and mm_npt;
        _nharm_buckets_v: their halves, the even harmonics up to the same),
        is the number of coefficients with which the cell's points are
        evaluated (_nharm_cell_u, _nharm_cell_v); all of these in the order
        of the map's own interpolant"""
        wE, Lz = self._wEs, self._Lzgrid
        self._A_ip = HermiteFamily3D(
            wE**2, self._ys, Lz, self._A_tab, self._A_dx, self._A_dy, self._A_dL
        )
        # s_u = cos(pi w_I/2) decreases along the grid: the u tables are
        # reversed along w_I so that their second variable increases
        su = numpy.cos(0.5 * numpy.pi * self._wIs)[::-1]
        sv = numpy.sin(0.5 * numpy.pi * self._wIs)
        utabs = [
            t[:, ::-1] for t in (self._U_tab, self._U_dwE, self._U_dsu, self._U_dL)
        ]
        vtabs = [self._V_tab, self._V_dwE, self._V_dsv, self._V_dL]
        self._nharm_buckets_u = numpy.array(
            [2**k for k in range(4, 31) if 2**k < self._npt] + [self._npt]
        )
        self._nharm_buckets_v = numpy.unique(self._nharm_buckets_u // 2)
        out = []
        for s, tabs, buckets in (
            (su, utabs, self._nharm_buckets_u),
            (sv, vtabs, self._nharm_buckets_v),
        ):
            # the tables' first entry is the half-width, which every node keeps
            need = _harmonics_needed(*(t[..., 1:] for t in tabs))
            # where the circular face meets the face on which this map's
            # libration is degenerate, all of a node's tables vanish, yet the
            # coefficients grow out of that edge as the product of the two
            # variables: the mixed derivative there, which the interpolant
            # estimates, belongs to the harmonics of the next nodes
            need[0, 0] = numpy.maximum(
                need[0, 0], numpy.maximum(need[0, 1], need[1, 0])
            )
            cell = numpy.maximum(need[:-1], need[1:])
            cell = numpy.maximum(cell[:, :-1], cell[:, 1:])
            cell = numpy.maximum(cell[:, :, :-1], cell[:, :, 1:])
            cell = buckets[numpy.searchsorted(buckets, cell)]
            out += [HermiteFamily3D(wE, s, Lz, *tabs, nkeep=1 + need), need, cell]
        self._U_ip, self._nharm_node_u, self._nharm_cell_u = out[:3]
        self._V_ip, self._nharm_node_v, self._nharm_cell_v = out[3:]
        return None

    # ---------- evaluation: the manifest chain
    @property
    def _target_hint(self):
        """The tail of an out-of-range error for a family localized on a
        target, whose box is the target's padded range rather than the
        default spans"""
        if self._target is None:
            return ""
        return (
            " (the grid is localized on a target: increase target_pad, or build "
            "the family without target=)"
        )

    def _check_Lz(self, ell):
        """Raise for an angular momentum outside the family's grid"""
        if ell < self._Lzgrid[0] or ell > self._Lzgrid[-1]:
            raise ValueError(
                f"|L_z| = {ell} outside the interpolation grid "
                f"[{self._Lzgrid[0]}, {self._Lzgrid[-1]}]" + self._target_hint
            )
        return None

    def _interp_coords(self, jr, jz, ell):
        """Solve the implicit inverse labels: the grid coordinates (x, y)
        of the torus with actions (J_R, J_z) at |L_z|, by root-finding on
        the stored action interpolants themselves -- exact-in-the-family,
        so canonicity is untouched. A vanishing action puts the torus on
        the corresponding face exactly."""

        def A(x, y, **kw):
            return self._A_ip(x, y, ell, **kw)[0, 0]

        # the box the family covers, in x = w_E^2 and y
        xlo, xhi = self._wEs[0] ** 2, self._wEs[-1] ** 2
        ylo, yhi = self._ys[0], self._ys[-1]
        # the residual is relative to the actions, so that actions down to
        # round-off above zero resolve to the edges' neighbourhood rather
        # than to the edges themselves
        tol = 1e-12 * (jr + jz)
        for face, jother, k in ((1.0, jz, 1), (0.0, jr, 0)):
            if (jr if k == 1 else jz) == 0.0:
                if (yhi if face == 1.0 else ylo) != face:
                    raise ValueError(
                        f"(J_R, J_z) = ({jr}, {jz}) lies on a degenerate face "
                        "that the family's grid does not reach"
                    )
                top, bottom = A(xhi, face)[k], A(xlo, face)[k]
                if jother > top + tol or jother < bottom - tol:
                    raise ValueError(
                        f"(J_R, J_z) = ({jr}, {jz}) lies outside the interpolated "
                        f"family at |L_z| = {ell}: "
                        + ("above" if jother > top else "below")
                        + f" the covered actions [{bottom:g}, {top:g}]"
                        + (
                            " (increase Rinf)"
                            if jother > top and self._target is None
                            else ""
                        )
                        + self._target_hint
                    )
                jother = min(max(jother, bottom), top)
                # on a face, the root is found in w_E = sqrt(x), in which a
                # tiny action resolves to a small w_E rather than to zero
                wE = brentq(
                    lambda ww: A(ww * ww, face)[k] - jother,
                    self._wEs[0],
                    self._wEs[-1],
                    xtol=1e-15,
                )
                return wE * wE, face
        x, y, bad = self._interp_coords_array(
            numpy.array([jr]), numpy.array([jz]), numpy.array([ell])
        )
        if not bad[0]:
            return x[0], y[0]
        top = max(numpy.sum(A(xhi, yy)[:2]) for yy in numpy.linspace(ylo, yhi, 33))
        raise ValueError(
            f"(J_R, J_z) = ({jr}, {jz}) could not be matched inside the "
            f"interpolated family at |L_z| = {ell}"
            + (
                f": its total action lies above the covered {top:g}"
                + (" (increase Rinf)" if self._target is None else "")
                if jr + jz > top
                else ""
            )
            + self._target_hint
        )

    def _interp_tables(self, jr, jz, Lz):
        """The family's tables of the (J_R, J_z, L_z) torus with their
        derivatives along the three action chains, all the interpolants'
        own: the implicit inverse labels, the frequencies from the action
        interpolants' partials, and the turning points and the maps chained
        from their own variables. A vanishing action leaves its libration
        degenerate, with no chain."""
        ell, sgn = numpy.fabs(Lz), numpy.sign(Lz)
        self._check_Lz(ell)
        x, y = self._interp_coords(jr, jz, ell)
        A = self._A_ip(x, y, ell)[0, 0]
        Ax = self._A_ip(x, y, ell, dx=1)[0, 0]
        Ay = self._A_ip(x, y, ell, dy=1)[0, 0]
        AL = self._A_ip(x, y, ell, dz=1)[0, 0]
        # d(x, y, |L_z|)/d(J_R, J_z, J_phi), from the action interpolants'
        # partials (J_phi = L_z moves |L_z| with the sign of L_z)
        det = Ax[0] * Ay[1] - Ay[0] * Ax[1]
        N = numpy.zeros((3, 3))
        N[0, 0], N[0, 1] = Ay[1] / det, -Ay[0] / det
        N[1, 0], N[1, 1] = -Ax[1] / det, Ax[0] / det
        N[0, 2] = -(N[0, 0] * AL[0] + N[0, 1] * AL[1]) * sgn
        N[1, 2] = -(N[1, 0] * AL[0] + N[1, 1] * AL[1]) * sgn
        N[2, 2] = sgn
        circ = self._circular(ell)
        _, dE_dx, dE_dL = self._E_of_xL(x, ell)
        tab = {
            "Om": dE_dx * N[0] + dE_dL * N[2],
            "uc": A[2],
            "duc": Ax[2] * N[0] + Ay[2] * N[1] + AL[2] * N[2],
            "udeg": jr == 0.0,
            "vdeg": jz == 0.0,
        }
        if self._delta_rule is not None:
            # the torus's own focal length and its chain
            tab["delta"] = A[3]
            tab["ddelta"] = Ax[3] * N[0] + Ay[3] * N[1] + AL[3] * N[2]
        wE = numpy.sqrt(x)
        for deg, ip, s, fs, key in (
            ("udeg", self._U_ip, numpy.sqrt(1.0 - y), -1.0, "u"),
            ("vdeg", self._V_ip, numpy.sqrt(y), 1.0, "v"),
        ):
            if tab[deg]:
                npt = self._npt if key == "u" else self._nptv
                tab["w" + key], tab["dw" + key] = 0.0, numpy.zeros(3)
                tab["Dm" + key] = numpy.zeros(npt)
                tab["dDm" + key] = numpy.zeros((npt, 3))
                continue
            T = ip(wE, s, ell)[0, 0]
            TwE = ip(wE, s, ell, dx=1)[0, 0]
            Ts = ip(wE, s, ell, dy=1)[0, 0]
            TL = ip(wE, s, ell, dz=1)[0, 0]
            # chains through the tables' own variables, w_E = sqrt(x) and
            # s = sqrt(1 - y) or sqrt(y), whose derivatives diverge only at
            # the faces on which this libration is degenerate
            dT = (
                TwE[:, None] * N[0][None, :] / (2.0 * wE)
                + Ts[:, None] * N[1][None, :] * fs / (2.0 * s)
                + TL[:, None] * N[2][None, :]
            )
            tab["w" + key], tab["dw" + key] = self._halfwidth(
                key, T[0], dT[0], tab["uc"], tab["duc"]
            )
            tab["Dm" + key], tab["dDm" + key] = T[1:], dT[1:]
        return tab

    @staticmethod
    def _halfwidth(key, lam, dlam, uc, duc):
        """The half-width of a libration and its chain from the variable in
        which the family stores it, and that variable's chain. The u
        libration: lambda_u = u_c artanh(w_u / u_c) = (u_c / 2) ln(u_+ /
        u_-), so that w_u = u_c tanh(lambda_u / u_c) and the inner turning
        point u_c - w_u is positive whatever the interpolation error, and
        accurate relative to itself where the centrifugal barrier holds it
        just off the axis. The v libration: lambda_v = arcsinh(tan w_v),
        the inverse Gudermannian, so that w_v = arcsin(tanh lambda_v) stays
        below pi / 2 and sin v_min is accurate relative to itself for a
        nearly polar torus. Both variables equal the half-width to first
        order where it vanishes. For one torus, or for many with a leading
        torus axis."""
        if key == "v":
            return (
                numpy.arcsin(numpy.tanh(lam)),
                dlam / numpy.cosh(numpy.asarray(lam)[..., None]),
            )
        q = numpy.asarray(lam / uc)[..., None]
        th = numpy.tanh(q)
        return uc * th[..., 0], th * duc + (1.0 - th**2) * (dlam - q * duc)

    def _interp_coords_array(self, jr, jz, ell):
        """The grid coordinates (x, y) of many tori at once (none of them on
        a degenerate face), and which of them could not be matched inside
        the family: a Newton iteration vectorized over the tori on the
        batched action interpolant (_newton_labels) from the middle of the
        energies at the vertical fraction of the actions and, for the tori
        that this leaves, from the best point of a coarse scan of the box"""
        xlo, xhi = self._wEs[0] ** 2, self._wEs[-1] ** 2
        ylo, yhi = self._ys[0], self._ys[-1]
        frac = jz / (jr + jz)
        x = numpy.full(len(jr), 0.5 * (xlo + xhi))
        y = numpy.clip(frac, ylo + 0.02 * (yhi - ylo), yhi - 0.02 * (yhi - ylo))
        bad = self._newton_labels(jr, jz, ell, x, y)
        left = numpy.flatnonzero(bad)
        if len(left) > 0:
            xs, ys = (
                t.ravel()
                for t in numpy.meshgrid(
                    numpy.linspace(xlo + 0.05 * (xhi - xlo), xhi, 12),
                    numpy.linspace(ylo, yhi, 12),
                )
            )
            A = self._A_ip.batch_all(
                numpy.tile(xs, len(left)),
                numpy.tile(ys, len(left)),
                numpy.repeat(ell[left], len(xs)),
            )[0].reshape(len(left), len(xs), -1)
            k = numpy.argmin(
                numpy.fabs(A[:, :, 0] - jr[left, None])
                + numpy.fabs(A[:, :, 1] - jz[left, None]),
                axis=1,
            )
            xl, yl = xs[k], ys[k]
            bad[left] = self._newton_labels(jr[left], jz[left], ell[left], xl, yl)
            x[left], y[left] = xl, yl
        return x, y, bad

    def _newton_labels(self, jr, jz, ell, x, y):
        """Newton iteration for the grid coordinates of many tori from the
        starting points (x, y), which it overwrites; returns which tori did
        not converge. The unknowns are solved from the total action and
        the vertical fraction of it, both of which the grid coordinates
        control at every scale: near the circular edge both actions vanish
        with x and their fraction alone fixes y, so a Newton on the actions
        themselves would have a degenerate direction there. A step is at
        most 0.3 long, is kept inside the box, and is halved until it
        reduces the residual: full steps can cycle between a face and the
        interior where an action rises steeply out of the face. A torus
        outside the family ends against a face of the box, where no step
        reduces the residual. The schedule is fixed: every torus has
        _LABEL_STEPS trial points, each either accepted (the torus moves
        there and takes a new Newton step from it) or rejected (it stays
        and halves its step), and is frozen once its residual is below
        1e-12 or its step has been halved below 1e-6 of a Newton step; the
        tori that are frozen are simply not evaluated any more, which gives
        each torus what _LABEL_STEPS masked iterations over all of them
        would. A torus that has not converged when the trial points are
        used up counts as not converged, like one that is stuck."""
        xlo, xhi = self._wEs[0] ** 2, self._wEs[-1] ** 2
        ylo, yhi = self._ys[0], self._ys[-1]
        jtot, frac = jr + jz, jz / (jr + jz)
        n = len(jr)
        # the trial points; (x, y) are the accepted iterates, with their
        # residuals, Newton directions and step lengths
        xt, yt = x.copy(), y.copy()
        norm, t = numpy.full(n, numpy.inf), numpy.ones(n)
        dx, dy = numpy.zeros(n), numpy.zeros(n)
        active = numpy.arange(n)
        for _ in range(self._LABEL_STEPS):
            v, vx, vy, _ = self._A_ip.batch_all(xt[active], yt[active], ell[active])
            tot = v[:, 0] + v[:, 1]
            with numpy.errstate(divide="ignore", invalid="ignore"):
                G0, G1 = tot / jtot[active] - 1.0, v[:, 1] / tot - frac[active]
                Gx0, Gx1 = (
                    (vx[:, 0] + vx[:, 1]) / jtot[active],
                    (vx[:, 1] * tot - v[:, 1] * (vx[:, 0] + vx[:, 1])) / tot**2,
                )
                Gy0, Gy1 = (
                    (vy[:, 0] + vy[:, 1]) / jtot[active],
                    (vy[:, 1] * tot - v[:, 1] * (vy[:, 0] + vy[:, 1])) / tot**2,
                )
                det = Gx0 * Gy1 - Gy0 * Gx1
                ndx = -(Gy1 * G0 - Gy0 * G1) / det
                ndy = -(-Gx1 * G0 + Gx0 * G1) / det
            # an interpolant that leaves the physical range is no improvement
            new = numpy.where(
                tot > 0.0, numpy.maximum(numpy.fabs(G0), numpy.fabs(G1)), numpy.inf
            )
            acc = new < norm[active]
            a, r = active[acc], active[~acc]
            x[a], y[a], norm[a] = xt[a], yt[a], new[acc]
            dx[a], dy[a] = ndx[acc], ndy[acc]
            t[a] = numpy.minimum(
                1.0,
                0.3
                / numpy.maximum(
                    numpy.maximum(numpy.fabs(dx[a]), numpy.fabs(dy[a])), 1e-300
                ),
            )
            t[r] *= 0.5
            active = active[(norm[active] >= 1e-12) & (t[active] >= 1e-6)]
            if active.size == 0:
                break
            xt[active] = numpy.clip(x[active] + t[active] * dx[active], xlo, xhi)
            yt[active] = numpy.clip(y[active] + t[active] * dy[active], ylo, yhi)
        return norm >= 1e-12

    def _label_tori(self, jr, jz, Lz):
        """What the evaluation of many (J_R, J_z, L_z) tori of a family,
        none on a degenerate face, needs that does not involve the maps'
        harmonics, with a leading torus axis: the grid coordinates (x, y)
        from the label inversion, the frequencies and the u midpoint with
        its chain, as _interp_tables returns them for one torus, the
        derivatives d(x, y, |L_z|)/d(J_R, J_z, J_phi), and the group of the
        torus's cell, which is the pair of the cell's numbers of
        coefficients of the u and of the v map (_nharm_cell_u,
        _nharm_cell_v) as one integer; for a family whose focal length
        follows its tori, each torus's focal length and its chain come
        after the derivatives"""
        sgn = numpy.sign(Lz)
        x, y, ell, inside = self._locate_tori(jr, jz, Lz)
        A, Ax, Ay, AL = self._A_ip.batch_all(x, y, ell)
        det = Ax[:, 0] * Ay[:, 1] - Ay[:, 0] * Ax[:, 1]
        N = numpy.zeros((len(jr), 3, 3))
        N[:, 0, 0], N[:, 0, 1] = Ay[:, 1] / det, -Ay[:, 0] / det
        N[:, 1, 0], N[:, 1, 1] = -Ax[:, 1] / det, Ax[:, 0] / det
        N[:, 0, 2] = -(N[:, 0, 0] * AL[:, 0] + N[:, 0, 1] * AL[:, 1]) * sgn
        N[:, 1, 2] = -(N[:, 1, 0] * AL[:, 0] + N[:, 1, 1] * AL[:, 1]) * sgn
        N[:, 2, 2] = sgn
        _, dE_dx, dE_dL = self._E_of_xL_array(x, ell)
        wE = numpy.sqrt(x)
        dil = (
            [A[:, 3], sum(G[:, 3, None] * N[:, k] for k, G in enumerate((Ax, Ay, AL)))]
            if self._delta_rule is not None
            else []
        )
        return (
            x,
            y,
            dE_dx[:, None] * N[:, 0] + dE_dL[:, None] * N[:, 2],
            A[:, 2],
            Ax[:, 2, None] * N[:, 0]
            + Ay[:, 2, None] * N[:, 1]
            + AL[:, 2, None] * N[:, 2],
            N,
            *dil,
            self._nharm_cell_u[self._U_ip.cells(wE, numpy.sqrt(1.0 - y), ell)]
            * (self._nptv + 1)
            + self._nharm_cell_v[self._V_ip.cells(wE, numpy.sqrt(y), ell)],
            inside,
        )

    def _locate_tori(self, jr, jz, Lz):
        """The grid coordinates (x, y, |L_z|) of many tori, none on a
        degenerate face, and which of them lie inside the family: those with
        |L_z| inside the grid whose actions the label inversion matches. A
        torus outside gets the middle of the box at the nearest |L_z| of the
        grid, a placeholder at which the tables evaluate."""
        ell = numpy.fabs(Lz)
        inside = (ell >= self._Lzgrid[0]) & (ell <= self._Lzgrid[-1])
        ell = numpy.clip(ell, self._Lzgrid[0], self._Lzgrid[-1])
        x = numpy.full(len(jr), 0.5 * (self._wEs[0] ** 2 + self._wEs[-1] ** 2))
        y = numpy.full(len(jr), 0.5 * (self._ys[0] + self._ys[-1]))
        xi, yi, bad = self._interp_coords_array(jr[inside], jz[inside], ell[inside])
        ok = numpy.flatnonzero(inside)[~bad]
        x[ok], y[ok] = xi[~bad], yi[~bad]
        inside[inside] = ~bad
        return x, y, ell, inside

    def _raise_outside(self, tori, inside):
        """Raise for the tori, rows of (J_R, J_phi, J_z), of an array
        evaluation that lie outside the family: all of them are counted, the
        first is named"""
        if numpy.all(inside):
            return None
        ell = numpy.fabs(tori[:, 1])
        offgrid = (ell < self._Lzgrid[0]) | (ell > self._Lzgrid[-1])
        first = tori[numpy.argmin(inside)]
        raise ValueError(
            f"{numpy.sum(~inside)} of the {len(tori)} tori lie outside the "
            f"interpolated family: {numpy.sum(offgrid)} with |L_z| outside the "
            f"interpolation grid [{self._Lzgrid[0]}, {self._Lzgrid[-1]}] and "
            f"{numpy.sum(~inside & ~offgrid)} whose (J_R, J_z) could not be matched "
            "inside the family at their |L_z|; the first is (J_R, L_z, J_z) = "
            f"({first[0]}, {first[1]}, {first[2]}); in_domain(jr, jphi, jz) returns "
            "which tori lie inside" + self._target_hint
        )

    def _interp_tables_array(self, Lz, x, y, Om, uc, duc, N, *rest):
        """The family's tables of many tori at once, none on a degenerate
        face, from their labels (_label_tori), as _interp_tables returns
        them for one torus but with a leading point axis and with the first
        nharm = (u, v) coefficients of the two maps (the maps' coefficients
        as (M, npoints), their chains as (M, npoints, 3))"""
        tab = {"Om": Om, "uc": uc, "duc": duc, "udeg": False, "vdeg": False}
        nharm = rest[-1]
        if len(rest) > 1:
            # (the labels of a family whose focal length follows its tori)
            tab["delta"], tab["ddelta"] = rest[:2]
        wE, ell = numpy.sqrt(x), numpy.fabs(Lz)
        for ip, sv, fs, key, nh in (
            (self._U_ip, numpy.sqrt(1.0 - y), -1.0, "u", nharm[0]),
            (self._V_ip, numpy.sqrt(y), 1.0, "v", nharm[1]),
        ):
            T, TwE, Ts, TL = ip.batch_all(wE, sv, ell, ntab=1 + nh)
            dT = (
                TwE[:, :, None] * N[:, None, 0, :] / (2.0 * wE)[:, None, None]
                + Ts[:, :, None] * N[:, None, 1, :] * fs / (2.0 * sv)[:, None, None]
                + TL[:, :, None] * N[:, None, 2, :]
            )
            tab["w" + key], tab["dw" + key] = self._halfwidth(
                key, T[:, 0], dT[:, 0], uc, duc
            )
            tab["Dm" + key], tab["dDm" + key] = (
                T[:, 1:].T,
                numpy.moveaxis(dT[:, 1:], 0, 1),
            )
        return tab

    def _explicit_tables(self, ii):
        """An explicit torus is a one-node family: its own values and exact
        slopes stand in for the interpolants', the slopes in (E, I_3, L_z)
        chained to the actions through the torus's period matrix"""
        N = self._Ns[ii]
        udeg, vdeg = self._wus[ii] == 0.0, self._wvs[ii] == 0.0
        dsup = self._dsups[ii]
        return {
            "Om": N[0],
            "uc": self._ucs[ii],
            "duc": dsup[0] @ N,
            "udeg": udeg,
            "vdeg": vdeg,
            "wu": self._wus[ii],
            "dwu": numpy.zeros(3) if udeg else dsup[1] @ N,
            "Dmu": self._Dmus[ii],
            "dDmu": self._dDmus[ii] @ N,
            "wv": self._wvs[ii],
            "dwv": numpy.zeros(3) if vdeg else dsup[2] @ N,
            "Dmv": self._Dmvs[ii],
            "dDmv": self._dDmvs[ii] @ N,
        }

    def _kernel_u(self, tau, a, e, L, Dm, uc, wu, chains, jac=False):
        """Everything the evaluation needs from the u libration at anomaly
        tau, in one pass and from the tables alone: the auxiliary anomaly
        eta(tau), the auxiliary's radial angle (Kepler's equation, closed
        form) and its derivative, the function Lambda(eta) of its vertical
        angle relation and its derivative (the branch keyed to the winding
        number of eta, so that both are continuous for any tau), the
        target's u and p_u (the flux identity p_u du/dtau = p^A dr^A/dtau,
        with sin(eta)/sin(tau) grouped so that every factor is regular at
        the turning points), and, for each chain (da, de, duc, dwu, dDm),
        the compensation p^A dr^A/dJ|_tau - p_u du/dJ|_tau. The map has as
        many harmonics as coefficients are given. With jac, also the
        derivatives of p_u and of the compensations with respect to tau,
        in closed form."""
        ms = self._nforDm[: len(Dm)]
        eta, deta, smt, *cmt = self._map(tau, Dm, ms, jac)
        se, ce = numpy.sin(eta), numpy.cos(eta)
        rA, pA, gA = self._aAIinv._profile(a, e, eta)
        kap = a * e / (a + self._b)
        thetaA = eta - kap * se
        dthetaA = (1.0 - kap * ce) * deta
        sq = numpy.sqrt(L**2 + 4.0 * self._GM * self._b)
        c1 = numpy.sqrt((1.0 + e) / (1.0 - e))
        c2 = numpy.sqrt(
            (a * (1.0 + e) + 2.0 * self._b) / (a * (1.0 - e) + 2.0 * self._b)
        )
        t2 = numpy.tan(0.5 * eta)
        nwind = numpy.round(eta / (2.0 * numpy.pi))
        Lam = (
            numpy.arctan(c1 * t2)
            + L / sq * numpy.arctan(c2 * t2)
            + numpy.pi * nwind * (1.0 + L / sq)
        )
        ch2, sh2 = numpy.cos(0.5 * eta) ** 2, numpy.sin(0.5 * eta) ** 2
        dLam = (
            0.5 * c1 / (ch2 + c1**2 * sh2) + 0.5 * L / sq * c2 / (ch2 + c2**2 * sh2)
        ) * deta
        st, ct = numpy.sin(tau), numpy.cos(tau)
        u = uc - wu * ct
        # sin(eta)/sin(tau) -> eta'(tau) at the turning points
        sratio = numpy.where(
            numpy.fabs(st) > 1e-12, se / numpy.where(st == 0.0, 1.0, st), deta
        )
        pu = pA * gA * sratio * deta / wu
        drA_da, drA_de = self._aAIinv._radius_partials(a, e, eta)
        comps = [
            pA * (drA_da * da + drA_de * de + gA * se * _contract(smt, dDm))
            - pu * (duc - dwu * ct)
            for da, de, duc, dwu, dDm in chains
        ]
        out = (eta, thetaA, dthetaA, Lam, dLam, u, pu, comps)
        if not jac:
            return out
        ms = ms[:, None] if Dm.ndim == 2 else ms
        d2eta = -_contract(smt, ms**2 * Dm)
        y = 1.0 - e * ce
        sA, dy, yb = self._b + a * y, e * se, y + self._b / a
        # with p^A = P sin(eta): P, and the derivatives with respect to eta
        # of ln P, of g^A and of the radius partials
        P = numpy.sqrt(self._GM / (a + self._b)) * a * e / rA
        dlnP = -gA * se / rA
        dgA = gA * dy * (1.0 / yb - yb * a**2 / rA**2)
        dXa = (dy * sA + y * a * dy) / rA - y * sA * gA * se / rA**2
        dXe = -a * ((a * dy * ce - sA * se) / rA - sA * ce * gA * se / rA**2)
        # every factor regular at the turning points: sin(eta) times the
        # derivative of sin(eta)/sin(tau) is that ratio times [cos(eta)
        # eta' - (sin(eta)/sin(tau)) cos(tau)]
        dpu = (
            P
            * sratio
            / wu
            * (
                ((dlnP * gA + dgA) * se + 2.0 * gA * ce) * deta**2
                - gA * sratio * ct * deta
                + gA * se * d2eta
            )
        )
        dpA = P * (dlnP * se + ce) * deta
        dcomps = []
        for da, de, duc, dwu, dDm in chains:
            S = _contract(smt, dDm)
            dcomps.append(
                dpA * (drA_da * da + drA_de * de + gA * se * S)
                + pA
                * (
                    (dXa * da + dXe * de + (dgA * se + gA * ce) * S) * deta
                    + gA * se * _contract(cmt[0], ms * dDm)
                )
                - dpu * (duc - dwu * ct)
                - pu * dwu * st
            )
        return (*out, dpu, dcomps)

    def _kernel_v(self, tau, L, c, Dm, wv, chains, jac=False):
        """Everything the evaluation needs from the v libration at anomaly
        tau: the auxiliary's polar anomaly zeta(tau) and its derivative,
        the target's v and p_v (the flux identity, regular at the turning
        points), and, for each chain (dc, dwv, dDm), the compensation
        p^A_theta d theta^A/dJ|_tau - p_v dv/dJ|_tau, whose auxiliary part
        is grouped as -L sin(zeta) cos(zeta) / [2 sin^2 theta^A] dc, regular
        down to the planar edge, plus the flux times the map's shift. With
        jac, also the derivatives of p_v and of the compensations with
        respect to tau, in closed form."""
        ms = self._nforDmv[: len(Dm)]
        zeta, dzeta, smt, *cmt = self._map(tau, Dm, ms, jac)
        sz, cz = numpy.sin(zeta), numpy.cos(zeta)
        sinth2 = 1.0 - c * cz**2
        fluxA = L * c * sz**2 / sinth2
        st, ct = numpy.sin(tau), numpy.cos(tau)
        v = 0.5 * numpy.pi - wv * ct
        sratio = numpy.where(
            numpy.fabs(st) > 1e-12, sz / numpy.where(st == 0.0, 1.0, st), dzeta
        )
        pv = L * c * sz * sratio * dzeta / (sinth2 * wv)
        comps = [
            -0.5 * L * sz * cz / sinth2 * dc
            + fluxA * _contract(smt, dDm)
            + pv * dwv * ct
            for dc, dwv, dDm in chains
        ]
        out = (zeta, dzeta, v, pv, comps)
        if not jac:
            return out
        ms = ms[:, None] if Dm.ndim == 2 else ms
        d2zeta = -_contract(smt, ms**2 * Dm)
        ds2 = 2.0 * c * cz * sz * dzeta
        dpv = (
            L
            * c
            * sratio
            / wv
            * (
                ((2.0 * cz * dzeta - sratio * ct) * dzeta + sz * d2zeta) / sinth2
                - sz * dzeta * ds2 / sinth2**2
            )
        )
        dA = ((cz**2 - sz**2) * dzeta * sinth2 - sz * cz * ds2) / sinth2**2
        dflux = L * c * (2.0 * sz * cz * dzeta * sinth2 - sz**2 * ds2) / sinth2**2
        dcomps = [
            -0.5 * L * dA * dc
            + dflux * _contract(smt, dDm)
            + fluxA * _contract(cmt[0], ms * dDm)
            + dpv * dwv * ct
            - pv * dwv * st
            for dc, dwv, dDm in chains
        ]
        return (*out, dpv, dcomps)

    def _torus_at(self, tu, tv, aux, chains_u, chains_v, jac=False):
        """The torus at the anomaly pair (tau_u, tau_v): the auxiliary's
        radial and vertical angles with their derivatives, the point
        (u, p_u, v, p_v), the auxiliary's polar anomaly, and the total
        compensation of each chain (the u chains and v chains paired, and
        each libration's part on its own)"""
        n = len(tu)
        ju = jv = (0.0, [0.0, 0.0])
        if aux["udeg"]:
            # a shell orbit: the auxiliary's radial libration is degenerate
            # too (e = 0), so its radial angle is the anomaly and Lambda is
            # linear in it
            thetaAr, dthetaAr = tu, numpy.ones(n)
            Lam, dLam = aux["ratio"] * tu, aux["ratio"] * numpy.ones(n)
            u, pu = aux["uc"] * numpy.ones(n), numpy.zeros(n)
            comps_u = [numpy.zeros(n)] * len(chains_u)
        else:
            _, thetaAr, dthetaAr, Lam, dLam, u, pu, comps_u, *ju = self._kernel_u(
                tu,
                aux["a"],
                aux["e"],
                aux["L"],
                aux["Dmu"],
                aux["uc"],
                aux["wu"],
                chains_u,
                jac,
            )
        if aux["vdeg"]:
            zeta, dzeta = tv, numpy.ones(n)
            v, pv = 0.5 * numpy.pi * numpy.ones(n), numpy.zeros(n)
            comps_v = [numpy.zeros(n)] * len(chains_v)
        else:
            zeta, dzeta, v, pv, comps_v, *jv = self._kernel_v(
                tv, aux["L"], aux["c"], aux["Dmv"], aux["wv"], chains_v, jac
            )
        # the auxiliary's vertical angle: affine in its polar anomaly (the
        # in-plane phase psi = zeta + pi/2), plus its radial angle relation
        T = {
            "thetaAr": thetaAr,
            "dthetaAr": dthetaAr,
            "thetaAz": zeta + 0.5 * numpy.pi + aux["ratio"] * thetaAr - Lam,
            "dthetaAz_du": aux["ratio"] * dthetaAr - dLam,
            "dthetaAz_dv": dzeta,
            "zeta": zeta,
            "Lam": Lam,
            "u": u,
            "pu": pu,
            "v": v,
            "pv": pv,
            "comps": [cu + cv for cu, cv in zip(comps_u, comps_v)],
            "comps_u": comps_u,
            "comps_v": comps_v,
        }
        if jac:
            T["dpu"], T["dpv"] = ju[0], jv[0]
            T["jac"] = (
                T["dthetaAr"] + ju[1][0],
                jv[1][0],
                T["dthetaAz_du"] + ju[1][1],
                T["dthetaAz_dv"] + jv[1][1],
            )
        if "dlnD" in aux:
            self._dilate(T, tu, tv, aux, ju, jv, jac)
        return T

    def _dilate(self, T, tu, tv, aux, ju, jv, jac):
        """The compensation of a focal length that follows the tori, added
        to those of the torus T in place: the point transformation (u, v)
        -> (R, z) = delta (sinh u sin v, cosh u cos v) dilates with the
        torus's focal length, which adds -(R p_R + z p_z) dln(delta)/dJ to
        the angle of each chain, with R p_R + z p_z = [sinh u cosh u p_u -
        sin v cos v p_v] / (sinh^2 u + sin^2 v); it couples the two
        librations. With jac, also the Jacobian of the two angle relations,
        d(theta_R, theta_z)/d(tau_u, tau_v) row by row, in closed form from
        the kernels' derivatives (ju, jv)"""
        u, pu, v, pv = T["u"], T["pu"], T["v"], T["pv"]
        shc, snc = 0.5 * numpy.sinh(2.0 * u), 0.5 * numpy.sin(2.0 * v)
        den = numpy.sinh(u) ** 2 + numpy.sin(v) ** 2
        xp = (shc * pu - snc * pv) / den
        dl = aux["dlnD"]
        T["comps"] = [c - xp * d for c, d in zip(T["comps"], dl)]
        if jac:
            du, dv = aux["wu"] * numpy.sin(tu), aux["wv"] * numpy.sin(tv)
            xu = (
                numpy.cosh(2.0 * u) * du * pu
                + shc * ju[0]
                - xp * numpy.sinh(2.0 * u) * du
            ) / den
            xv = (
                -(
                    numpy.cos(2.0 * v) * dv * pv
                    + snc * jv[0]
                    + xp * numpy.sin(2.0 * v) * dv
                )
                / den
            )
            T["jac"] = (
                T["dthetaAr"] + ju[1][0] - xu * dl[0],
                jv[1][0] - xv * dl[0],
                T["dthetaAz_du"] + ju[1][1] - xu * dl[1],
                T["dthetaAz_dv"] + jv[1][1] - xv * dl[1],
            )
        return None

    def _newton_angles_dilated(self, thR, thz, aux, chains_u, chains_v):
        """_newton_angles for a family whose focal length follows its
        tori, whose compensation couples the two librations: Newton steps
        with the Jacobian of the two angle relations in closed form, on a
        fixed schedule (maxiter steps, a pair that has converged stays, a
        step is at most 0.5 long in each anomaly and a vanishing
        determinant is replaced by one, so that every step is finite)"""

        def residuals(x, y):
            T = self._torus_at(x, y, aux, chains_u, chains_v, jac=True)
            FR = T["thetaAr"] + T["comps"][0] - thR
            Fz = T["thetaAz"] + T["comps"][1] - thz
            FR, Fz = ((F + numpy.pi) % (2.0 * numpy.pi) - numpy.pi for F in (FR, Fz))
            return FR, Fz, numpy.maximum(numpy.fabs(FR), numpy.fabs(Fz)), T["jac"]

        x = numpy.array(thR, dtype="float")
        T0 = self._torus_at(x, 0.0 * x, aux, [], [])
        y = thz - 0.5 * numpy.pi - aux["ratio"] * T0["thetaAr"] + T0["Lam"]
        for _ in range(self._maxiter):
            FR, Fz, norm, (a, b, c, d) = residuals(x, y)
            go = norm >= self._angle_tol
            if not numpy.any(go):
                # (all the remaining steps would be masked)
                break
            det = a * d - b * c
            det = numpy.where(numpy.fabs(det) > 1e-10, det, 1.0)
            x = x + go * numpy.clip(-(d * FR - b * Fz) / det, -0.5, 0.5)
            y = y + go * numpy.clip(-(a * Fz - c * FR) / det, -0.5, 0.5)
        else:
            go = residuals(x, y)[2] >= self._angle_tol
        return x, y, numpy.flatnonzero(go)

    def _newton_angles(self, thR, thz, aux, chains_u, chains_v):
        """Newton solve of the two angle relations theta_R(tau_u, tau_v) and
        theta_z(tau_u, tau_v) -- the auxiliary's angles plus the
        compensations along the J_R and J_z chains -- for the anomaly pair
        of each requested angle pair, vectorized over the angles. The
        Jacobian is that of the auxiliary's angles, in closed form, plus
        that of the compensations, which are small and each the sum of a
        function of tau_u and one of tau_v, so that their four slopes are
        secants between successive iterates and the convergence is
        superlinear; returns the anomalies and the indices of the pairs
        that did not converge within maxiter iterations"""
        if "dlnD" in aux:
            return self._newton_angles_dilated(thR, thz, aux, chains_u, chains_v)

        def wrap(f):
            return (f + numpy.pi) % (2.0 * numpy.pi) - numpy.pi

        def residuals(x, y):
            T = self._torus_at(x, y, aux, chains_u, chains_v)
            return (
                wrap(T["thetaAr"] + T["comps"][0] - thR),
                wrap(T["thetaAz"] + T["comps"][1] - thz),
                T,
            )

        x = numpy.array(thR, dtype="float")
        # the vertical anomaly that the auxiliary's angle relation gives at
        # tau_u = theta_R with the map and the compensation neglected
        T0 = self._torus_at(x, 0.0 * x, aux, [], [])
        y = thz - 0.5 * numpy.pi - aux["ratio"] * T0["thetaAr"] + T0["Lam"]
        prev, slope = None, numpy.zeros((4, len(x)))
        for _ in range(self._maxiter):
            FR, Fz, T = residuals(x, y)
            if (
                max(numpy.max(numpy.fabs(FR)), numpy.max(numpy.fabs(Fz)))
                < self._angle_tol
            ):
                return x, y, numpy.array([], dtype="int")
            # the compensations of the (J_R, J_z) chains from the u
            # libration, then from the v libration, and where each was taken
            comps = numpy.array([*T["comps_u"], *T["comps_v"]])
            at = numpy.array([x, x, y, y])
            if prev is not None:
                step = at - prev[0]
                moved = numpy.fabs(step) > 1e-9
                slope = numpy.where(
                    moved, (comps - prev[1]) / numpy.where(moved, step, 1.0), slope
                )
            prev = (at, comps)
            # a secant that would take the determinant far from the
            # auxiliary's own is not trusted
            det0 = T["dthetaAr"] * T["dthetaAz_dv"]
            trusted = (T["dthetaAr"] + slope[0]) * (
                T["dthetaAz_dv"] + slope[3]
            ) - slope[2] * (T["dthetaAz_du"] + slope[1]) > 0.5 * det0
            sl = numpy.where(trusted, slope, 0.0)
            a, b = T["dthetaAr"] + sl[0], sl[2]
            c, d = T["dthetaAz_du"] + sl[1], T["dthetaAz_dv"] + sl[3]
            dx = numpy.clip(-(d * FR - b * Fz) / (a * d - b * c), -0.5, 0.5)
            dy = numpy.clip(-(Fz + c * dx) / d, -0.5, 0.5)
            x += dx
            y += dy
        FR, Fz, _ = residuals(x, y)
        return (
            x,
            y,
            numpy.flatnonzero(
                numpy.maximum(numpy.fabs(FR), numpy.fabs(Fz)) >= self._angle_tol
            ),
        )

    def _solve_angles(self, thR, thz, aux, chains_u, chains_v):
        """The anomaly pair of each requested angle pair on one torus: the
        Newton solve, with a bracketed scalar fallback for any pair that
        does not converge"""
        x, y, bad = self._newton_angles(thR, thz, aux, chains_u, chains_v)
        if len(bad) > 0:
            # theta_R - tau_u and theta_z - tau_v are periodic and bounded,
            # so [theta - s, theta + s] with s beyond their extremes are
            # guaranteed brackets: the vertical relation is solved at each
            # trial tau_u inside the solve of the radial one. The extremes
            # are estimated on a grid, which can miss them, but by less than
            # 2 pi, by which the periodic part of an increasing angle
            # relation can vary at most
            g = numpy.linspace(0.0, 2.0 * numpy.pi, 48, endpoint=False)
            GU, GV = (q.flatten() for q in numpy.meshgrid(g, g))
            Ts = self._torus_at(GU, GV, aux, chains_u, chains_v)
            sR = numpy.max(numpy.fabs(Ts["thetaAr"] + Ts["comps"][0] - GU)) + 6.3
            sz = numpy.max(numpy.fabs(Ts["thetaAz"] + Ts["comps"][1] - GV)) + 6.3
            for ii in bad:
                thri, thzi = thR[ii], thz[ii]

                def _tv(xx):
                    def _fz(yy):
                        T = self._torus_at(
                            numpy.array([xx]),
                            numpy.array([yy]),
                            aux,
                            chains_u,
                            chains_v,
                        )
                        return T["thetaAz"][0] + T["comps"][1][0] - thzi

                    return brentq(_fz, thzi - sz, thzi + sz, xtol=1e-15, maxiter=200)

                def _fR(xx):
                    T = self._torus_at(
                        numpy.array([xx]),
                        numpy.array([_tv(xx)]),
                        aux,
                        chains_u,
                        chains_v,
                    )
                    return T["thetaAr"][0] + T["comps"][0][0] - thri

                x[ii] = brentq(_fR, thri - sR, thri + sR, xtol=1e-15, maxiter=200)
                y[ii] = _tv(x[ii])
        return x, y

    def _match_node(self, jr, jz, Lz):
        """Locate the explicit torus with actions (J_R, J_z, L_z)"""
        dev = (
            numpy.fabs(self._jrs - jr)
            + numpy.fabs(self._jzs - jz)
            + numpy.fabs(self._Lzs - Lz)
        )
        ii = numpy.argmin(dev)
        if dev[ii] > 1e-8 * (1.0 + numpy.fabs(jr) + numpy.fabs(jz) + numpy.fabs(Lz)):
            raise ValueError(
                f"(J_R, J_z, L_z) = ({jr}, {jz}, {Lz}) is not one of the set-up "
                "tori; discrete mode evaluates the stored tori only (use "
                "setup_interp=True to interpolate)"
            )
        return ii

    # ---------- the public inverse map
    def _evaluate(self, jr, jphi, jz, angler, anglephi, anglez, **kwargs):
        return self._xvFreqs(jr, jphi, jz, angler, anglephi, anglez, **kwargs)[:6]

    def _actions_from_integrals(self, E, Lz, I3):
        """(J_R, J_z) of the torus with integrals (E, L_z, I_3)"""
        E = conversion.parse_energy(E, vo=self._vo)
        Lz = conversion.parse_angmom(Lz, ro=self._ro, vo=self._vo)
        I3 = conversion.parse_energy(I3, vo=self._vo)
        if not self._interp:
            dev = (
                numpy.fabs(self._Es - E)
                + numpy.fabs(self._Lzs - Lz)
                + numpy.fabs(self._I3s - I3)
            )
            ii = numpy.argmin(dev)
            if dev[ii] > 1e-10 * (
                1.0 + numpy.fabs(E) + numpy.fabs(Lz) + numpy.fabs(I3)
            ):
                raise ValueError(
                    f"(E, L_z, I_3) = ({E}, {Lz}, {I3}) is not one of the set-up "
                    "tori; to use interpolation, initialize with setup_interp=True"
                )
            return self._jrs[ii], self._jzs[ii]
        if self._delta_rule is not None:
            raise NotImplementedError(
                "(E, L_z, I_3) do not label the tori of a family whose focal "
                "length follows them: I_3 is that of each torus's own model"
            )
        ell = numpy.fabs(Lz)
        self._check_Lz(ell)
        circ = self._circular(ell)
        Ec, Emax = circ["Ec"], self._Emax(ell)
        tol = 1e-12 * (Emax - Ec)
        if E < Ec - tol or E > Emax + tol:
            raise ValueError(
                f"E = {E} outside the interpolation grid at |L_z| = {ell}: [{Ec}, {Emax}]"
            )
        xlo, xhi = self._wEs[0] ** 2, self._wEs[-1] ** 2
        ylo, yhi = self._ys[0], self._ys[-1]
        if E < Ec + tol:
            # the circular orbit, to the tolerance of the torus construction
            self._check_circular()
            return 0.0, 0.0
        # the normalized energy from the apocentre of the planar orbit
        # without angular momentum at E
        R0c, _, R0m, _ = self._R0_edges(ell)
        R0 = self._R0(min(E, Emax), R0c, R0m)
        x = min(numpy.log(R0 / R0c) / numpy.log(R0m / R0c), 1.0)
        if x < xlo - 1e-10 or x > xhi + 1e-10:
            raise ValueError(
                f"E = {E} outside the interpolation grid at |L_z| = {ell}: its energy "
                f"variable {x:g} lies outside [{xlo:g}, {xhi:g}]" + self._target_hint
            )
        with self._in_chart(E, ell):
            ush = self._ushell(E, ell)
            Ipl, Ish = self._I3_planar(E, ell), self._I3_shell(E, ell, ush)
            tolI = 1e-10 * (Ish - Ipl)
            y = (
                self._y_of_I3(E, ell, min(max(I3, Ipl), Ish))[0]
                if Ipl - tolI <= I3 <= Ish + tolI
                else numpy.inf
            )
            if y < ylo - 1e-10 or y > yhi + 1e-10:
                raise ValueError(
                    f"I_3 = {I3} outside the interpolation grid at (E, |L_z|) = "
                    f"({E}, {ell}): [{self._I3_of_y(E, ell, ylo)[0]}, "
                    f"{self._I3_of_y(E, ell, yhi)[0]}]" + self._target_hint
                )
        A = self._A_ip(min(max(x, xlo), xhi), min(max(y, ylo), yhi), ell)[0, 0]
        return A[0], A[1]

    @conversion.physical_conversion("position", pop=True)
    def delta(self, E, Lz, y, **kwargs):
        """
        Return the focal length of the Staeckel model of the torus (E, L_z, y).

        Parameters
        ----------
        E : float, numpy.ndarray or Quantity
            Energy of the torus.
        Lz : float, numpy.ndarray or Quantity
            z-component of the angular momentum of the torus.
        y : float or numpy.ndarray
            Third-integral variable of the torus: the family's grid
            variable, from 0 for the planar orbit to 1 for the shell orbit
            of (E, L_z).

        Returns
        -------
        float, numpy.ndarray or Quantity
            Focal length of the prolate spheroidal coordinates in which the
            torus is a torus of the Staeckel approximation.

        Notes
        -----
        - The single focal length of a family or of explicit tori; for a
          family whose focal length follows its tori, the fitted polynomial
          (delta='fit', also beyond the grid, where it extends smoothly,
          limited softly to a little beyond the range of the fitted
          optima) or the family's own table of it (a callable delta=,
          within the grid).
        - The method is itself a rule: delta=family.delta gives another
          family of the same potential (with the same axis_weight) the
          same focal lengths without fitting them again.
        - 2026-11-17 - Written - Bovy (UofT)
        """
        E = conversion.parse_energy(E, vo=self._vo)
        Lz = conversion.parse_angmom(Lz, ro=self._ro, vo=self._vo)
        E, ell, y = numpy.broadcast_arrays(
            *(numpy.asarray(t, dtype="float") for t in (E, numpy.fabs(Lz), y))
        )
        if self._delta_rule is None:
            return (self._delta + 0.0 * E)[()]
        x = numpy.empty(E.size)
        for k, (e, l) in enumerate(zip(E.ravel(), ell.ravel())):
            R0c, _, R0m, _ = self._R0_edges(l)
            x[k] = numpy.log(self._R0(e, 0.5 * R0c, 2.0 * R0m) / R0c) / numpy.log(
                R0m / R0c
            )
        if self._delta_fit is not None:
            out = self._delta_fitted(x, y.ravel(), ell.ravel())[0]
        else:
            out = self._A_ip.batch_all(
                numpy.clip(x, self._wEs[0] ** 2, self._wEs[-1] ** 2),
                numpy.clip(y.ravel(), self._ys[0], self._ys[-1]),
                numpy.clip(ell.ravel(), self._Lzgrid[0], self._Lzgrid[-1]),
            )[0][:, 3]
        return out.reshape(E.shape)[()]

    @conversion.physical_conversion("action", pop=True)
    def JR(self, E, Lz, I3, **kwargs):
        """
        Return the radial action of the torus with integrals (E, L_z, I_3).

        Parameters
        ----------
        E : float or Quantity
            Energy of the torus.
        Lz : float or Quantity
            z-component of the angular momentum of the torus.
        I3 : float or Quantity
            Third integral of the torus (see the class documentation for
            its convention).

        Returns
        -------
        float or Quantity
            Radial action J_R of the torus.

        Notes
        -----
        - For an interpolating instance (setup_interp=True), any (E, L_z,
          I_3) within the grid, read from the family's own action
          interpolant (exact at the nodes); for an instance set up with
          explicit tori, (E, L_z, I_3) must be one of them.
        - 2026-09-24 - Written - Bovy (UofT)
        """
        return self._actions_from_integrals(E, Lz, I3)[0]

    @conversion.physical_conversion("action", pop=True)
    def Jz(self, E, Lz, I3, **kwargs):
        """
        Return the vertical action of the torus with integrals (E, L_z, I_3).

        Parameters
        ----------
        E : float or Quantity
            Energy of the torus.
        Lz : float or Quantity
            z-component of the angular momentum of the torus.
        I3 : float or Quantity
            Third integral of the torus (see the class documentation for
            its convention).

        Returns
        -------
        float or Quantity
            Vertical action J_z of the torus.

        Notes
        -----
        - As JR.
        - 2026-09-24 - Written - Bovy (UofT)
        """
        return self._actions_from_integrals(E, Lz, I3)[1]

    @actionAngleInverse_physical_input
    def in_domain(self, jr, jphi, jz, **kwargs):
        """
        Whether tori lie inside the domain on which this instance can evaluate them.

        Parameters
        ----------
        jr : float, numpy.ndarray or Quantity
            Radial action.
        jphi : float, numpy.ndarray or Quantity
            Azimuthal action.
        jz : float, numpy.ndarray or Quantity
            Vertical action.

        Returns
        -------
        bool or numpy.ndarray
            True for the tori that the instance evaluates, False for those for which an evaluation raises, in the broadcast shape of the actions (a bool for a single torus).

        Notes
        -----
        - For an interpolating instance (setup_interp=True), the tori with |L_z| inside the angular-momentum grid whose (J_R, J_z) the family covers at that |L_z|; for an instance set up with explicit tori, those tori. An array evaluation raises when any of its tori lies outside, so use this to select the tori of a sample beforehand.
        - 2026-11-16 - Written - Bovy (UofT)
        """
        jr, Lz, jz = numpy.broadcast_arrays(
            *(numpy.asarray(t, dtype="float") for t in (jr, jphi, jz))
        )
        shape = jr.shape
        jr, Lz, jz = jr.ravel(), Lz.ravel(), jz.ravel()
        out = numpy.zeros(len(jr), dtype="bool")
        valid = (jr >= 0.0) & (jz >= 0.0) & (Lz != 0.0)
        # the tori with both librations of a family are located in one
        # vectorized pass, the others one by one
        many = valid & (jr > 0.0) & (jz > 0.0) & self._interp
        if numpy.any(many):
            out[many] = self._locate_tori(jr[many], jz[many], Lz[many])[3]
        for ii in numpy.flatnonzero(valid & ~many):
            try:
                if jr[ii] == 0.0 and jz[ii] == 0.0 and self._interp:
                    self._check_Lz(numpy.fabs(Lz[ii]))
                    self._check_circular()
                else:
                    self._tables(jr[ii], jz[ii], Lz[ii])
                out[ii] = True
            except ValueError:
                pass
        return out.reshape(shape) if shape else bool(out[0])

    def _tables(self, jr, jz, Lz):
        """The tables of the requested torus, from the family or from the
        explicit tori"""
        if self._interp:
            return self._interp_tables(jr, jz, Lz)
        return self._explicit_tables(self._match_node(jr, jz, Lz))

    def _xvFreqs(self, jr, jphi, jz, angler, anglephi, anglez, **kwargs):
        """(J, theta) -> (x, v) and the frequencies, for actions and angles
        that broadcast against each other by numpy's rules (scalar actions
        with arrays of angles: one torus at those angles; arrays of actions
        with scalar angles: that angle on every torus; equal shapes: paired
        element by element, each star on its own torus at its own angle;
        actions of shape (n,1) with angles of shape (m,): every angle on
        every torus as (n,m) arrays); a single torus returns its
        frequencies as floats"""
        scalar = numpy.ndim(jr) == 0 and numpy.ndim(jphi) == 0 and numpy.ndim(jz) == 0
        jr, jphi, jz, angler, anglephi, anglez = numpy.broadcast_arrays(
            *(
                numpy.atleast_1d(numpy.asarray(t, dtype="float"))
                for t in (jr, jphi, jz, angler, anglephi, anglez)
            )
        )
        if numpy.any(jr < 0.0) or numpy.any(jz < 0.0):
            raise ValueError("J_R and J_z have to be non-negative")
        if numpy.any(jphi == 0.0):
            raise ValueError("L_z = 0 is not supported")
        shape = jr.shape
        jr, jphi, jz, thR, anglephi, thz = (
            t.ravel()
            for t in (
                jr,
                jphi,
                jz,
                angler % (2.0 * numpy.pi),
                anglephi,
                anglez % (2.0 * numpy.pi),
            )
        )
        if scalar:
            out = self._xvFreqs_torus(jr[0], jphi[0], jz[0], thR, anglephi, thz)
            return (*(o.reshape(shape) for o in out[:6]), *out[6:])
        out = self._xvFreqs_points(jr, jphi, jz, thR, anglephi, thz)
        return tuple(o.reshape(shape) for o in out)

    def _xvFreqs_points(self, jr, jphi, jz, thR, anglephi, thz):
        """The evaluation at flat arrays of actions and angles of one length:
        the tables are read once per distinct torus; a family's tori with
        both librations are labelled once and evaluated in vectorized
        passes with one row of tables per point, the points grouped by
        their cells' numbers of coefficients of the two maps, which are all
        that a pass reads and sums (the groups of few points gathered into
        one with the largest numbers among them); the others (explicit
        tori, tori on a degenerate face, a single distinct torus, and any
        point whose angle solve does not converge) go through the
        one-torus path"""
        out = numpy.empty((9, len(jr)))

        def one_torus(actions, pts):
            for o, q in zip(
                out, self._xvFreqs_torus(*actions, thR[pts], anglephi[pts], thz[pts])
            ):
                o[pts] = q

        tori, inv = numpy.unique(
            numpy.stack([jr, jphi, jz], axis=1), axis=0, return_inverse=True
        )
        inv = inv.ravel()
        single = (tori[:, 0] == 0.0) | (tori[:, 2] == 0.0)
        if not self._interp or len(tori) == 1:
            single[:] = True
        order = numpy.argsort(inv, kind="stable")
        starts = numpy.searchsorted(inv[order], numpy.arange(len(tori) + 1))
        for k in numpy.flatnonzero(single):
            one_torus(tori[k], order[starts[k] : starts[k + 1]])
        if not numpy.all(single):
            pts = numpy.flatnonzero(~single[inv])
            # each point's row among the tori of the vectorized pass
            rows = (numpy.cumsum(~single) - 1)[inv[pts]]
            tsel = tori[~single]
            *labels, group, inside = _chunked(
                self._label_tori, len(tsel), tsel[:, 0], tsel[:, 2], tsel[:, 1]
            )
            self._raise_outside(tsel, inside)
            group = group[rows]
            _, ginv, counts = numpy.unique(
                group, return_inverse=True, return_counts=True
            )
            small = (counts < _EVAL_GROUP_MIN)[ginv]
            nhu, nhv = divmod(group[small], self._nptv + 1)
            group[small] = numpy.amax(nhu, initial=0) * (self._nptv + 1) + numpy.amax(
                nhv, initial=0
            )
            for g in numpy.unique(group):
                idx, nharm = pts[group == g], divmod(g, self._nptv + 1)
                *out[:, idx], bad = _chunked(
                    lambda *args, nharm=nharm: self._xvFreqs_rows(
                        tsel, labels, *args, nharm
                    ),
                    len(idx),
                    rows[group == g],
                    thR[idx],
                    anglephi[idx],
                    thz[idx],
                    nharm=nharm[0],
                )
                for ii in idx[bad]:
                    one_torus(tori[inv[ii]], numpy.array([ii]))
        return tuple(out)

    def _xvFreqs_torus(self, jr, jphi, jz, thR, anglephi, thz):
        """One torus at flat arrays of angles: solve the two angle relations
        for the anomaly pair of each requested angle pair, reconstruct the
        point from the two librations (u, p_u, v, p_v) -> (R, v_R, z, v_z),
        and take the azimuth from the auxiliary's angle relations at the
        compensated azimuthal angle; the frequencies are floats"""
        Lz = jphi
        sgn = numpy.sign(Lz)
        if jr == 0.0 and jz == 0.0:
            # the circular orbit: the point sits at the circular radius in
            # the plane with no radial or vertical motion, at the requested
            # azimuthal angle (the auxiliary's angle relations give phi =
            # theta_phi there), with the epicycle, circular, and vertical
            # frequencies
            if self._interp:
                self._check_Lz(numpy.fabs(Lz))
                self._check_circular()
            else:
                self._match_node(jr, jz, Lz)
            circ = self._circular(Lz)
            Rc = circ["Rc"]
            return (
                Rc + 0.0 * thR,
                0.0 * thR,
                Lz / Rc + 0.0 * thR,
                0.0 * thR,
                0.0 * thR,
                anglephi % (2.0 * numpy.pi),
                circ["kappa"],
                sgn * circ["Omc"],
                circ["nu"],
            )
        tab = self._tables(jr, jz, Lz)
        L = jz + numpy.fabs(Lz)
        a, e = self._aAIinv._orbital_params(jr, L)
        # c = sin^2 i = 1 - L_z^2/L^2 in the product form, exact for a thin
        # vertical libration where the difference would cancel
        c = jz * (jz + 2.0 * numpy.fabs(Lz)) / L**2
        sq = numpy.sqrt(L**2 + 4.0 * self._GM * self._b)
        aux = {
            "a": a,
            "e": e,
            "L": L,
            "c": c,
            "ratio": 0.5 * (1.0 + L / sq),
            "uc": tab["uc"],
            "wu": tab["wu"],
            "Dmu": tab["Dmu"],
            "wv": tab["wv"],
            "Dmv": tab["Dmv"],
            "udeg": tab["udeg"],
            "vdeg": tab["vdeg"],
        }
        if "delta" in tab:
            aux["delta"], aux["dlnD"] = tab["delta"], tab["ddelta"] / tab["delta"]
        # the three action chains, (J_R, J_z, J_phi): through the auxiliary
        # torus's parameters -- (a, e) from (J_R, L) and c = sin^2 i from
        # (L, L_z), with L = J_z + |L_z| -- the turning points, and the
        # maps' coefficients
        dL = numpy.array([0.0, 1.0, sgn])
        dc = 2.0 * Lz**2 / L**3 * dL - 2.0 * Lz / L**2 * numpy.array([0.0, 0.0, 1.0])
        chains_u, chains_v = [], []
        for i in range(3):
            # the auxiliary's radial libration is degenerate with the
            # target's (J^A_r = J_R), and then has no chains to speak of
            da, de = 0.0, 0.0
            if not tab["udeg"]:
                _, _, da, de = self._aAIinv._orbital_param_chains(
                    jr, L, 1.0 if i == 0 else 0.0, dL[i]
                )
            chains_u.append((da, de, tab["duc"][i], tab["dwu"][i], tab["dDmu"][:, i]))
            chains_v.append((dc[i], tab["dwv"][i], tab["dDmv"][:, i]))
        tu, tv = self._solve_angles(thR, thz, aux, chains_u[:2], chains_v[:2])
        T = self._torus_at(tu, tv, aux, chains_u, chains_v)
        Om = tab["Om"]
        return (
            *self._point(T, Lz, L, anglephi, aux.get("delta")),
            Om[0],
            Om[2],
            Om[1],
        )

    def _point(self, T, Lz, L, anglephi, delta=None):
        """(R, v_R, v_T, z, v_z, phi) from the two librations of _torus_at:
        the azimuth from the auxiliary's angle relations, phi = theta^A_phi
        - sgn(L_z) theta^A_z + arctan2(cos i sin psi, cos psi) with the
        in-plane phase psi = zeta + pi/2 and cos i = L_z / L, at the
        compensated azimuthal angle; in the coordinates of the focal
        length delta (of each point's own torus when it follows the tori,
        else the family's)"""
        delta = self._delta if delta is None else delta
        zeta = T["zeta"]
        phi = (
            anglephi
            - T["comps"][2]
            - numpy.sign(Lz) * T["thetaAz"]
            + numpy.arctan2(Lz / L * numpy.cos(zeta), -numpy.sin(zeta))
        )
        u, pu, v, pv = T["u"], T["pu"], T["v"], T["pv"]
        sh, ch = numpy.sinh(u), numpy.cosh(u)
        sn, cs = numpy.sin(v), numpy.cos(v)
        den = delta * (sh**2 + sn**2)
        R = delta * sh * sn
        return (
            R,
            (pu * ch * sn + pv * sh * cs) / den,
            Lz / R,
            delta * ch * cs,
            (pu * sh * cs - pv * ch * sn) / den,
            phi % (2.0 * numpy.pi),
        )

    def _xvFreqs_rows(self, tori, labels, rows, thR, anglephi, thz, nharm):
        """A vectorized pass over a family's tori with both librations,
        given as rows of (J_R, J_phi, J_z) with their labels (_label_tori),
        at points that each sit on the torus rows[point]: the tables of the
        points' tori with the first nharm = (u, v) coefficients of the
        maps, one row of them per point, the kernels with per-point
        coefficients, and the Newton solve of the angle relations; returns
        the nine outputs and whether each point's solve failed to converge"""
        used, rows = numpy.unique(rows, return_inverse=True)
        jr, Lz, jz = tori[used].T
        tab, aux, chains_u, chains_v = self._rows_tables(
            jr, Lz, jz, [t[used] for t in labels], nharm, rows
        )
        L = (jz + numpy.fabs(Lz))[rows]
        tu, tv, bad = self._newton_angles(thR, thz, aux, chains_u[:2], chains_v[:2])
        T = self._torus_at(tu, tv, aux, chains_u, chains_v)
        Om = tab["Om"][rows]
        return (
            *self._point(T, Lz[rows], L, anglephi, aux.get("delta")),
            *Om[:, [0, 2, 1]].T,
            numpy.isin(numpy.arange(len(rows)), bad),
        )

    def _rows_tables(self, jr, Lz, jz, labels, nharm, rows):
        """The tables of a family's tori with both librations, from their
        labels (_label_tori) with the first nharm = (u, v) coefficients of
        the maps, and what the kernels need at points that each sit on the
        torus rows[point]: the auxiliary torus's parameters and the maps'
        tables, and the three action chains of each libration"""
        sgn, ell = numpy.sign(Lz), numpy.fabs(Lz)
        tab = self._interp_tables_array(Lz, *labels, nharm)
        L = jz + ell
        a, e = self._aAIinv._orbital_params(jr, L)
        c = jz * (jz + 2.0 * ell) / L**2
        sq = numpy.sqrt(L**2 + 4.0 * self._GM * self._b)
        aux = {
            "a": a[rows],
            "e": e[rows],
            "L": L[rows],
            "c": c[rows],
            "ratio": 0.5 * (1.0 + L / sq)[rows],
            "uc": tab["uc"][rows],
            "wu": tab["wu"][rows],
            "Dmu": tab["Dmu"][:, rows],
            "wv": tab["wv"][rows],
            "Dmv": tab["Dmv"][:, rows],
            "udeg": False,
            "vdeg": False,
        }
        if "delta" in tab:
            aux["delta"] = tab["delta"][rows]
            aux["dlnD"] = (tab["ddelta"] / tab["delta"][:, None])[rows].T
        zero, one = numpy.zeros(len(jr)), numpy.ones(len(jr))
        dL = numpy.stack([zero, one, sgn], axis=1)
        dc = (2.0 * Lz**2 / L**3)[:, None] * dL - (2.0 * Lz / L**2)[
            :, None
        ] * numpy.stack([zero, zero, one], axis=1)
        chains_u, chains_v = [], []
        for i in range(3):
            _, _, da, de = self._aAIinv._orbital_param_chains(
                jr, L, 1.0 if i == 0 else 0.0, dL[:, i]
            )
            chains_u.append(
                (
                    da[rows],
                    de[rows],
                    tab["duc"][rows, i],
                    tab["dwu"][rows, i],
                    tab["dDmu"][:, rows, i],
                )
            )
            chains_v.append((dc[rows, i], tab["dwv"][rows, i], tab["dDmv"][:, rows, i]))
        return tab, aux, chains_u, chains_v

    # ---------- the inverse of the map
    def _invert(self, R, vR, vT, z, vz, phi, jr, jz, angler, anglez, **kwargs):
        """actionAngleInverse._invert, with a family's points first solved
        in the family's own variables (_invert_family); what that leaves
        (tori without one of the librations, points near the edge of the
        domain) goes through the general solve"""
        args = (R, vR, vT, z, vz, phi, jr, jz, angler, anglez)
        if kwargs.get("fixed", False) or not self._interp:
            return super()._invert(*args, **kwargs)
        out = self._invert_family(R, vR, vT, z, vz, phi, jr, jz)
        left = numpy.isnan(out[0])
        if numpy.any(left):
            out[:, left] = super()._invert(*(a[left] for a in args), **kwargs)
        return tuple(out)

    def _invert_family(self, R, vR, vT, z, vz, phi, jr, jz, tol=1e-12, maxiter=8):
        """The actions and angles that a family maps to the points (R, v_R,
        v_T, z, v_z, phi), by Newton's method in the actions and the two
        anomalies (J_R, J_z, tau_u, tau_v), in which the point of a torus
        is explicit (no solve of the angle relations) and the Jacobian is
        in closed form (_point_jacobian): each iteration reads the tables
        of each point's current torus once. The anomalies start where the
        guess's torus has the point's (u, v), on the branch of the signs of
        (p_u, p_v). Returns the array (J_R, J_z, theta_R, theta_phi,
        theta_z), nan for the points that are left to the general solve:
        those that do not converge, or whose iterates leave the tori with
        both librations inside the domain"""
        n, Lz = len(R), R * vT
        target = numpy.array([z, R, vz, vR])
        jr, jz = numpy.array(jr, dtype="float"), numpy.array(jz, dtype="float")
        tu, tv = numpy.zeros(n), numpy.zeros(n)
        out = numpy.full((5, n), numpy.nan)
        active = numpy.isfinite(jr + jz) & (jr > 0.0) & (jz > 0.0)
        start = True
        for _ in range(maxiter):
            idx = numpy.flatnonzero(active)
            if len(idx) == 0:
                break
            *labels, group, inside = _chunked(
                self._label_tori, len(idx), jr[idx], jz[idx], Lz[idx]
            )
            active[idx[~inside]] = False
            idx, group = idx[inside], group[inside]
            labels = [t[inside] for t in labels]
            for g in numpy.unique(group):
                sel = numpy.flatnonzero(group == g)
                pts = idx[sel]
                tab, aux, chains_u, chains_v = self._rows_tables(
                    jr[pts],
                    Lz[pts],
                    jz[pts],
                    [t[sel] for t in labels],
                    divmod(g, self._nptv + 1),
                    numpy.arange(len(pts)),
                )
                delta = aux.get("delta", self._delta)
                if start:
                    u, v = coords.Rz_to_uv(R[pts], z[pts], delta=delta)
                    sh, ch, sn, cs = (
                        numpy.sinh(u),
                        numpy.cosh(u),
                        numpy.sin(v),
                        numpy.cos(v),
                    )
                    for t, lib, q, w, p in (
                        (
                            tu,
                            aux["uc"],
                            u,
                            aux["wu"],
                            vR[pts] * ch * sn + vz[pts] * sh * cs,
                        ),
                        (
                            tv,
                            0.5 * numpy.pi,
                            v,
                            aux["wv"],
                            vR[pts] * sh * cs - vz[pts] * ch * sn,
                        ),
                    ):
                        t0 = numpy.arccos(numpy.clip((lib - q) / w, -1.0, 1.0))
                        t[pts] = numpy.where(p >= 0.0, t0, 2.0 * numpy.pi - t0)
                T = self._torus_at(tu[pts], tv[pts], aux, chains_u, chains_v, jac=True)
                x = self._point(T, Lz[pts], aux["L"], 0.0, aux.get("delta"))
                res = target[:, pts] - numpy.array([x[3], x[0], x[4], x[1]])
                done = numpy.amax(numpy.fabs(res), axis=0) < tol
                fin = pts[done]
                out[:, fin] = (
                    jr[fin],
                    jz[fin],
                    (T["thetaAr"] + T["comps"][0])[done] % (2.0 * numpy.pi),
                    (phi[fin] - x[5][done]) % (2.0 * numpy.pi),
                    (T["thetaAz"] + T["comps"][1])[done] % (2.0 * numpy.pi),
                )
                active[fin] = False
                jac = self._point_jacobian(
                    T, aux, chains_u, chains_v, tu[pts], tv[pts], x, delta
                )
                step = numpy.linalg.solve(jac[~done], res.T[~done][:, :, None])[:, :, 0]
                go = pts[~done]
                jr[go] += step[:, 0]
                jz[go] += step[:, 1]
                tu[go] += numpy.clip(step[:, 2], -0.5, 0.5)
                tv[go] += numpy.clip(step[:, 3], -0.5, 0.5)
            active &= (jr > 0.0) & (jz > 0.0)
            start = False
        return out

    def _point_jacobian(self, T, aux, chains_u, chains_v, tu, tv, x, delta):
        """d(z, R, v_z, v_R)/d(J_R, J_z, tau_u, tau_v) of the points x =
        _point(T) of tori with both librations, as (npoints, 4, 4), in
        closed form. With zeta = z + i R = delta cosh(u + i v) and pi =
        v_z - i v_R = (p_u - i p_v) / [delta sinh(u + i v)], the
        derivatives with respect to the anomalies follow from those of
        (u, p_u) and (v, p_v), and those of the position with respect to
        the actions from the chains of the turning points and of the focal
        length. Those of the velocity with respect to the actions at fixed
        anomalies come from the canonical structure, dp/dJ at a fixed
        position being the transpose of dtheta/dx: dp/dJ = (dtheta/dtau
        [dx/dtau]^-1)^T + dp/dtau [dx/dtau]^-1 dx/dJ, with the vanishing of
        dx/dtau at a turning point (where the sum is regular) kept from
        zero"""
        npts = len(tu)
        zeta, pi = x[3] + 1j * x[0], x[4] - 1j * x[1]
        om = T["u"] + 1j * T["v"]
        dsh, coth = delta * numpy.sinh(om), 1.0 / numpy.tanh(om)
        su, sv = aux["wu"] * numpy.sin(tu), aux["wv"] * numpy.sin(tv)

        def cols(*zs, conj=False):
            # complex columns as real (npoints, 2, len(zs)) matrices
            sign = -1.0 if conj else 1.0
            return numpy.stack(
                [numpy.stack([q.real, sign * q.imag], axis=1) for q in zs], axis=2
            )

        Xt = cols(dsh * su, 1j * dsh * sv)
        Pt = cols(
            T["dpu"] / dsh - pi * coth * su,
            -1j * T["dpv"] / dsh - 1j * pi * coth * sv,
            conj=True,
        )
        dlnD = aux.get("dlnD", numpy.zeros((2, npts)))
        XJ = cols(
            *(
                dsh
                * (
                    chains_u[i][2]
                    - chains_u[i][3] * numpy.cos(tu)
                    - 1j * chains_v[i][1] * numpy.cos(tv)
                )
                + zeta * dlnD[i]
                for i in range(2)
            )
        )
        floor = 1e-7 * numpy.stack([aux["wu"], aux["wv"]], axis=1)
        s = numpy.stack([su, sv], axis=1)
        s = numpy.where(numpy.fabs(s) > floor, s, numpy.where(s >= 0.0, floor, -floor))
        Xti = numpy.linalg.inv(cols(dsh, 1j * dsh)) / s[:, :, None]
        dth = numpy.stack(T["jac"], axis=1).reshape(npts, 2, 2)
        jac = numpy.empty((npts, 4, 4))
        jac[:, :2, :2], jac[:, :2, 2:] = XJ, Xt
        jac[:, 2:, :2] = numpy.swapaxes(dth @ Xti, 1, 2) + Pt @ Xti @ XJ
        jac[:, 2:, 2:] = Pt
        return jac

    def _Freqs(self, jr, jphi, jz, **kwargs):
        """Frequencies of the (J_R, J_z, L_z) tori, for actions that
        broadcast against each other: in interpolation mode the stored
        action interpolants' own partials through the label chain (the
        integrator contract); for explicit tori the tori's own period
        matrices; floats for a single torus"""
        scalar = numpy.ndim(jr) == 0 and numpy.ndim(jphi) == 0 and numpy.ndim(jz) == 0
        jr, jphi, jz = numpy.broadcast_arrays(
            *(numpy.atleast_1d(numpy.asarray(t, dtype="float")) for t in (jr, jphi, jz))
        )
        if scalar:
            return self._Freqs_torus(jr[0], jphi[0], jz[0])
        tori, inv = numpy.unique(
            numpy.stack([t.ravel() for t in (jr, jphi, jz)], axis=1),
            axis=0,
            return_inverse=True,
        )
        single = (tori[:, 0] == 0.0) | (tori[:, 2] == 0.0)
        if not self._interp or len(tori) == 1:
            single[:] = True
        Om = numpy.empty((3, len(tori)))
        for k in numpy.flatnonzero(single):
            Om[:, k] = self._Freqs_torus(*tori[k])
        if not numpy.all(single):
            labels = self._label_tori(
                tori[~single, 0], tori[~single, 2], tori[~single, 1]
            )
            self._raise_outside(tori[~single], labels[-1])
            Om[:, ~single] = labels[2][:, [0, 2, 1]].T
        return tuple(o.reshape(jr.shape) for o in Om[:, inv.ravel()])

    def _Freqs_torus(self, jr, Lz, jz):
        """Frequencies of one torus, as floats"""
        if jr == 0.0 and jz == 0.0:
            if self._interp:
                self._check_Lz(numpy.fabs(Lz))
                self._check_circular()
            else:
                self._match_node(jr, jz, Lz)
            circ = self._circular(Lz)
            return (circ["kappa"], numpy.sign(Lz) * circ["Omc"], circ["nu"])
        Om = self._tables(jr, jz, Lz)["Om"]
        return (Om[0], Om[2], Om[1])
