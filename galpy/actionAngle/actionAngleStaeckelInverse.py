###############################################################################
#   actionAngleStaeckelInverse.py: inverse action-angle transformation for
#   axisymmetric Staeckel potentials, computed directly from the separable
#   structure: each angle is a u-profile plus a v-profile, both regular
#   quadratures in the chi anomaly, and (J,theta) -> (x,v) is an additively
#   separable 2x2 Newton solve. No auxiliary torus, generating function, or
#   Fourier lattice; placement on the torus is exact by construction.
###############################################################################
import warnings

import numpy
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.ndimage import spline_filter1d
from scipy.optimize import brentq, minimize, minimize_scalar

from ..potential import (
    IsochronePotential,
    OblateStaeckelWrapperPotential,
    evaluatePotentials,
    rl,
    vcirc,
)
from ..util import conversion, coords, galpyWarning
from .actionAngleInverse import actionAngleInverse
from .actionAngleIsochrone import actionAngleIsochrone
from .actionAngleIsochroneInverse import actionAngleIsochroneInverse

# Nodes/weights for composite 10-point Gauss-Legendre quadrature: applied
# per interval of the nchi-point chi mesh, the error per panel is
# O((pi/nchi)^20), so every stored integral is at machine precision for any
# reasonable nchi; there is no accuracy knob to tune
_GLX, _GLW = numpy.polynomial.legendre.leggauss(10)


def _bspline_weights(t):
    """Cubic B-spline weights for the four-point stencil at offset t"""
    t2 = t * t
    t3 = t2 * t
    return numpy.array(
        [
            (1.0 - 3.0 * t + 3.0 * t2 - t3) / 6.0,
            (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0,
            (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t3) / 6.0,
            t3 / 6.0,
        ]
    )


def _bspline_dweights(t):
    """Derivative of the cubic B-spline weights with respect to the offset"""
    t2 = t * t
    return numpy.array(
        [
            -0.5 * (1.0 - t) ** 2,
            -2.0 * t + 1.5 * t2,
            0.5 + t - 1.5 * t2,
            0.5 * t2,
        ]
    )


def _pad_axis(arr, axis, pad):
    """Extend arr beyond both ends of axis by quadratic extrapolation from the
    three nearest slices, vectorized over everything else"""
    n = arr.shape[axis]
    out = numpy.moveaxis(arr, axis, 0)
    flat = out.reshape((n, -1))
    deg = 2 if n > 2 else 1
    # one polynomial fit for all columns at once, from the edge slices only:
    # a global fit would be dominated by the far side of the grid
    left = numpy.polyfit(numpy.arange(deg + 1), flat[: deg + 1], deg)
    right = numpy.polyfit(numpy.arange(deg + 1), flat[n - deg - 1 :], deg)
    lo = numpy.array(
        [numpy.polyval(left, xx) for xx in numpy.arange(-pad, 0, dtype=float)]
    )
    hi = numpy.array(
        [
            numpy.polyval(right, xx)
            for xx in numpy.arange(deg + 1, deg + 1 + pad, dtype=float)
        ]
    )
    ext = numpy.vstack((lo, flat, hi)).reshape((n + 2 * pad,) + out.shape[1:])
    return numpy.moveaxis(ext, 0, axis)


def _prefilter_padded(arr, axes, pad):
    """Pad arr along axes and return its cubic-spline prefiltered form, ready
    for four-point-stencil evaluation. Only the given axes are
    prefiltered: the leading axis indexes distinct quantities, and filtering
    across it would mix them."""
    out = arr
    for ax in axes:
        out = _pad_axis(out, ax, pad)
    for ax in axes:
        out = spline_filter1d(out, order=3, axis=ax, mode="nearest")
    return out


class actionAngleStaeckelInverse(actionAngleInverse):
    """Inverse action-angle formalism for axisymmetric Staeckel potentials

    The angles of a separable potential are additively separable in the
    per-degree-of-freedom phases: theta_i = dW/dJ_i with
    W = int p_u du + int p_v dv + L_z phi, so each angle is the sum of a
    u-profile and a v-profile, both regular quadratures in the chi anomaly
    (u = u_- + [u_+ - u_-] sin^2 chi_u/2 and similarly for the full v loop).
    Evaluating (J,theta) -> (x,v) requires only inverting the additively
    separable 2x2 angle system, followed by closed-form coordinate
    transformations. Placement on the torus is exact by construction.
    """

    # A Staeckel torus is labelled by (E, L_z, I3) as well as by its actions;
    # I3 has the dimensions of an energy in the convention used here
    _integral_labels = (("E", "energy"), ("Lz", "angmom"), ("I3", "energy"))

    def __init__(
        self,
        pot=None,
        Es=[0.5],
        Lzs=[0.5],
        I3s=[0.1],
        setup_interp=False,
        Rmin=0.5,
        Rmax=2.0,
        Rinf=10.0,
        nLz=9,
        nE=9,
        nI3=9,
        grid_pad=0.02,
        nchi=2001,
        canonical=False,
        ncanon=128,
        npt=32,
        isochrone_ab=None,
        maxiter=60,
        angle_tol=1e-13,
        Lzlim=None,
        wElim=None,
        wIlim=None,
        target=None,
        target_pad=1.5,
        target_minwidth=0.02,
        **kwargs,
    ):
        """
        Initialize an actionAngleStaeckelInverse object.

        Parameters
        ----------
        pot : Potential or list thereof
            The potential; must be of exact Staeckel form and supply its own
            focal distance: either an OblateStaeckelWrapperPotential (also
            the route for the Staeckel approximation of a general
            axisymmetric potential) or a potential with a _delta attribute
            such as KuzminKutuzovStaeckelPotential.
        Es : list of float
            Energies of the tori to set up.
        Lzs : list of float
            z-components of the angular momentum of the tori.
        I3s : list of float
            Third integrals of the tori (in the convention
            p_u^2 = 2 delta^2 [E sinh^2 u - U(u) - I_3] - L_z^2/sinh^2 u
            with the gauge V(pi/2) = 0). Ignored when setup_interp is True.
        setup_interp : bool, optional
            If True, set up a grid of tori spanning the bound phase space and
            accept arbitrary actions within it, rather than only the tori
            listed in Es/Lzs/I3s. The grid locates a torus from its actions by
            interpolation alone, with no root finding, and the torus itself is
            then constructed exactly, so the interpolation enters only through
            the integrals it supplies.
        Rmin, Rmax : float, optional
            Radii of the circular orbits that anchor the range of L_z spanned
            by the grid (only used when setup_interp is True). As in the
            spherical grid, the range is anchored on radii rather than on L_z
            directly.
        Rinf : float, optional
            Radius setting the outer energy of the grid, through the energy of
            the planar orbit with this apocenter (only used when setup_interp
            is True).
        nLz, nE, nI3 : int, optional
            Number of grid points in each direction (only used when
            setup_interp is True).
        grid_pad : float, optional
            Fractional inset of the grid from the circular, planar, and shell
            edges, where the tori become degenerate (only used when
            setup_interp is True).
        nchi : int, optional
            Number of grid points in the chi anomaly used when constructing
            a torus.
        canonical : bool, optional
            If True, additionally build the canonical (momentum-matched)
            construction for the discrete tori: each torus is lifted onto
            its equal-action isochrone torus by per-degree momentum-matched
            point transformations, and evaluation runs through the analytic
            isochrone inverse (STAECKEL_CANONICAL_MATH.md section 10).
        ncanon : int, optional
            Number of anomaly samples (even) of the canonical
            correspondence tables per degree of freedom.
        npt : int, optional
            Number of sine modes of the stored momentum-matching anomaly
            maps (at most ncanon/2 - 1).
        isochrone_ab : (float, float), optional
            Frozen (GM, b) of the auxiliary isochrone to use instead of
            fitting one; the adaptive build hands every node the same
            auxiliary through this, because the compensation's closed-form
            auxiliary chains assume a single isochrone across the family.
        maxiter : int, optional
            Maximum number of Newton iterations in the angle inversion.
        angle_tol : float, optional
            Convergence tolerance of the angle inversion.
        Lzlim, wElim, wIlim : (float or None, float or None), optional
            Explicit (lower, upper) limits of the interpolation grid's box
            in each of its axes (L_z, w_E, w_I): a box narrower than the
            default localizes the grid -- on a stream, say -- and is worth
            as much as proportionally more nodes, because the interpolation
            error goes as the spacing. None on either end means that axis's
            own default edge; this matters because the w_I edges are the
            planar and shell degeneracies, which the grid must reach
            exactly or not at all (only used when setup_interp is True).
        target : Orbit or array_like, optional
            Localize the grid on a target instead of setting the box
            explicitly: phase-space points -- an Orbit (possibly an array
            of orbits, or one evaluated at an array of times) or rows
            (R, vR, vT, z, vz[, phi]) in internal units -- whose tori the
            grid should cover. The box is the target's own (L_z, w_E, w_I)
            range padded by target_pad, computed through the same label
            relations the node lattice uses (chart-local for an adaptive
            family), and is recorded in the _targetbox attribute. The
            labelling does a few root-finds per point, so subsample very
            large debris sets.
        target_pad : float, optional
            Padding of the target's box: each axis is extended by this
            factor times the target's spread on BOTH ends. Generous
            padding is load-bearing, because a box tight around the target
            puts it in the boundary layer of the interpolation stencil,
            and that failure mode is silent (the median error never
            notices; the worst case does).
        target_minwidth : float, optional
            Minimum width of the target box on each axis, as a fraction of
            that axis's default span -- what keeps the box of a single
            orbit, whose spread is zero on every axis, a grid rather than
            a point.

        Notes
        -----
        - Angle conventions match those of the forward actionAngleStaeckel.
        - When set up with setup_interp, the evaluation methods accept
          the canonical family's tables (STAECKEL_CANONICAL_MATH.md
          section 10).
        - 2026-08-19 - Started - Bovy (UofT)
        """
        delta = kwargs.pop("delta", None)
        u0 = kwargs.pop("u0", None)
        self._fit_nsub = kwargs.pop("fit_nsub", 6)
        self._delta_fit = isinstance(delta, str)
        self._u0_fit = isinstance(u0, str)
        if self._delta_fit and delta != "fit":
            raise ValueError("the only string value delta= accepts is 'fit'")
        if self._u0_fit and u0 != "fit":
            raise ValueError("the only string value u0= accepts is 'fit'")
        self._delta_func = delta if callable(delta) else None
        self._u0_func = u0 if callable(u0) else None
        if self._delta_fit and not setup_interp:
            raise TypeError(
                "delta='fit' builds an adaptive interpolated family and "
                "therefore requires setup_interp=True"
            )
        if self._delta_func is not None and not setup_interp:
            raise TypeError(
                "a callable delta(E, Lz) builds an adaptive interpolated "
                "family and therefore requires setup_interp=True"
            )
        if (self._u0_func is not None or self._u0_fit) and not setup_interp:
            raise TypeError(
                "an adaptive u0 (callable or 'fit') builds an adaptive "
                "interpolated family and therefore requires setup_interp=True"
            )
        actionAngleInverse.__init__(self, **kwargs)
        if pot is None:
            raise OSError("Must specify pot= for actionAngleStaeckelInverse")
        self._pot = pot
        if self._delta_fit:
            # the fitted surfaces: survey the |dPhi|-minimizing focal length
            # over the grid's own (L_z, E) domain, fit a smooth quadratic,
            # and use it exactly as a user-supplied callable
            dfun, u0fun, self._deltafit_info = _fit_staeckel_surface(
                pot,
                conversion.parse_length(Rmin, ro=self._ro),
                conversion.parse_length(Rmax, ro=self._ro),
                conversion.parse_length(Rinf, ro=self._ro),
                nsub=self._fit_nsub,
            )
            self._delta_func = dfun
            if self._u0_func is None:
                # fitted u0 unless the caller supplied their own callable;
                # u0='fit' and u0=None mean the same thing here, since the
                # fitted delta wants its matching reference curve
                self._u0_func = u0fun
        if self._delta_func is not None:
            # the reference chart, used only by the pre-grid helpers; every
            # node of the grid gets its own wrapper, and the evaluation reads
            # the focal length from the stored table row
            # probe the callables at a physically sensible reference point:
            # the circular orbit at the middle of the radial range, so that a
            # delta(E, L_z) or u0(E, L_z) built on rl/vcirc never sees the
            # degenerate (0, 0) it may not be defined at
            Rref = 0.5 * (
                conversion.parse_length(Rmin, ro=self._ro)
                + conversion.parse_length(Rmax, ro=self._ro)
            )
            Lzref = Rref * vcirc(pot, Rref, use_physical=False)
            Eref = (
                evaluatePotentials(pot, Rref, 0.0, use_physical=False)
                + Lzref**2.0 / 2.0 / Rref**2.0
            )
            dref = float(self._delta_func(Eref, Lzref))
            u0ref = None if self._u0_func is None else float(self._u0_func(Eref, Lzref))
            self._staeckelwrap = (
                OblateStaeckelWrapperPotential(pot=pot, delta=dref)
                if u0ref is None
                else OblateStaeckelWrapperPotential(pot=pot, delta=dref, u0=u0ref)
            )
        elif delta is not None or (
            u0 is not None and not callable(u0) and not self._u0_fit
        ):
            # the convenience spelling for a RAW potential; a potential that
            # already supplies its focal distance makes a scalar delta= (or
            # u0=) ambiguous, and stays an error as before -- an ADAPTIVE u0
            # (callable or 'fit') is not the convenience spelling, so it
            # falls through to the potential-supplied focal length below
            if isinstance(pot, OblateStaeckelWrapperPotential) or (
                getattr(pot, "_delta", None) is not None
            ):
                raise TypeError(
                    "delta= and u0= conflict with a potential that already "
                    "supplies its focal distance; pass the raw potential, "
                    "or a callable delta(E, Lz) for an adaptive family"
                )
            if delta is None:
                raise TypeError("u0= requires delta= as well")
            # an adaptive u0 leaves the reference wrapper's u0 at its
            # default; every node of the grid gets its own
            u0_scalar = None if callable(u0) or self._u0_fit else u0
            self._staeckelwrap = (
                OblateStaeckelWrapperPotential(pot=pot, delta=float(delta))
                if u0_scalar is None
                else OblateStaeckelWrapperPotential(
                    pot=pot, delta=float(delta), u0=float(u0_scalar)
                )
            )
        elif isinstance(pot, OblateStaeckelWrapperPotential):
            self._staeckelwrap = pot
        else:
            delta = getattr(pot, "_delta", None)
            if delta is None:
                raise OSError(
                    "actionAngleStaeckelInverse requires a potential of "
                    "Staeckel form that supplies its focal distance (e.g., "
                    "KuzminKutuzovStaeckelPotential); wrap a general "
                    "axisymmetric potential in OblateStaeckelWrapperPotential"
                )
            self._staeckelwrap = OblateStaeckelWrapperPotential(pot=pot, delta=delta)
        self._delta = self._staeckelwrap._delta
        # the potential the per-node charts wrap: a chart is always built on
        # the RAW potential -- rewrapping an OblateStaeckelWrapperPotential
        # would Staeckelize the Staeckelization, a different (and much more
        # expensive) model than intended
        self._chart_pot = (
            self._pot._pot
            if isinstance(self._pot, OblateStaeckelWrapperPotential)
            else self._pot
        )
        if self._u0_fit and self._u0_func is None:
            # u0='fit' with a FIXED focal length: the reference curve at the
            # zero-velocity R-midpoint of each (E, L_z) -- the same rule the
            # delta='fit' companion uses -- with no |dPhi| survey at all,
            # because the focal length is not being optimized.  Measured on
            # MWPotential2014: this placement carries most of the adaptive
            # win (the optimal delta is nearly universal there, while u0
            # placement matters by factors of a few to ~30).
            self._u0_func = _u0_midpoint_fun(
                self._chart_pot,
                conversion.parse_length(Rmin, ro=self._ro),
                conversion.parse_length(Rmax, ro=self._ro),
                lambda E, Lz, _d=float(self._delta): _d,
            )
        # a chart that varies across the family in ANY parameter builds
        # per-node wrappers and runs its label relations chart-locally;
        # only a varying delta needs the compensation term, because u0 is
        # construction-only gauge (the map's (R, z) <-> (u, v) uses delta
        # alone), so the stored delta row of a u0-only family is constant
        # and its chain contribution vanishes identically
        self._adaptive_chart = self._delta_func is not None or self._u0_func is not None
        # I3 has the dimensions of an energy in the convention used here
        self._Es = conversion._parse_grid_quantity(
            Es, conversion.parse_energy, vo=self._vo
        )
        self._Lzs = conversion._parse_grid_quantity(
            Lzs, conversion.parse_angmom, ro=self._ro, vo=self._vo
        )
        self._I3s = conversion._parse_grid_quantity(
            I3s, conversion.parse_energy, vo=self._vo
        )
        self._ntori = len(self._Es)
        self._nchi = nchi
        self._maxiter = maxiter
        self._angle_tol = angle_tol
        self._chi = numpy.linspace(0.0, numpy.pi, nchi)
        # theta_z = 0 at the upward midplane crossing, matching the forward
        # actionAngleStaeckel (the natural profile zero point is the northern
        # turning point, a quarter v-period earlier)
        self._anglez0 = numpy.pi / 2.0
        self._interp = setup_interp
        self._canonical = canonical
        self._isochrone_ab = isochrone_ab
        if ncanon % 2 == 1:
            raise ValueError("ncanon has to be even")
        if npt > ncanon // 2 - 1:
            raise ValueError("npt has to be at most ncanon/2 - 1")
        self._ncanon = ncanon
        self._npt = npt
        self._nforDm = numpy.arange(1, npt + 1)
        self._targetbox = None
        self._target_pad = target_pad
        self._target_minwidth = target_minwidth
        if target is not None:
            if not setup_interp:
                raise TypeError(
                    "target= localizes the interpolation grid and therefore "
                    "requires setup_interp=True"
                )
            if Lzlim is not None or wElim is not None or wIlim is not None:
                raise TypeError(
                    "target= computes Lzlim/wElim/wIlim itself and cannot "
                    "be combined with setting them explicitly"
                )
            self._target = _parse_target(target)
        else:
            self._target = None
        if setup_interp:
            # the interpolated family is the canonical construction (the
            # contingent interpolated-direct path was removed once the
            # canonical family matched its accuracy per the fast-orbits #23 review)
            self._canonical = True
            Rmin = conversion.parse_length(Rmin, ro=self._ro)
            Rmax = conversion.parse_length(Rmax, ro=self._ro)
            Rinf = conversion.parse_length(Rinf, ro=self._ro)
            self._setup_canonical_grid(
                Rmin, Rmax, Rinf, nLz, nE, nI3, grid_pad, Lzlim, wElim, wIlim
            )
            return
        # Setup in three logical stages
        self._find_turning_points()
        self._compute_actions_frequencies_profiles()
        self._build_angle_profile_splines()
        if canonical:
            self._setup_canonical()

    ############################ MOMENTA ######################################
    def _Wu(self, u, E, Lz, I3):
        """p_u^2 on the torus (E, Lz, I3); u may be an array"""
        return (
            2.0
            * self._delta**2.0
            * (E * numpy.sinh(u) ** 2.0 - self._staeckelwrap._U(u) - I3)
            - Lz**2.0 / numpy.sinh(u) ** 2.0
        )

    def _Wv(self, v, E, Lz, I3):
        """p_v^2 on the torus (E, Lz, I3); v may be an array"""
        return (
            2.0
            * self._delta**2.0
            * (E * numpy.sin(v) ** 2.0 + self._staeckelwrap._V(v) + I3)
            - Lz**2.0 / numpy.sin(v) ** 2.0
        )

    ############################ SETUP: TURNING POINTS ########################
    def _find_turning_points(self):
        """Bracket and refine the turning points of all tori. A single
        vectorized scan over a shared mesh (one potential evaluation for all
        tori, because U and V are shared) seeds the brackets; neither the
        range nor the resolution of the scan is a hard limit, because the
        brackets are extended by geometric stepping beyond the mesh (like
        the rperi/rap search in actionAngleSpherical) and an oscillation
        narrower than the mesh spacing is recovered by refining the maximum
        of W_u before declaring it absent. Roots are polished with brentq
        at machine tolerance."""
        d2 = self._delta**2.0
        us = numpy.linspace(1e-3, 40.0, 16000)
        Uu = self._staeckelwrap._U(us)
        shu2 = numpy.sinh(us) ** 2.0
        Wu_all = (
            2.0 * d2 * (self._Es[:, None] * shu2 - Uu - self._I3s[:, None])
            - self._Lzs[:, None] ** 2.0 / shu2
        )
        vs = numpy.linspace(1e-3, numpy.pi / 2.0, 4000)
        Vv = self._staeckelwrap._V(vs)
        snv2 = numpy.sin(vs) ** 2.0
        Wv_all = (
            2.0 * d2 * (self._Es[:, None] * snv2 + Vv + self._I3s[:, None])
            - self._Lzs[:, None] ** 2.0 / snv2
        )
        self._umins = numpy.empty(self._ntori)
        self._umaxs = numpy.empty(self._ntori)
        self._vmins = numpy.empty(self._ntori)
        for ii in range(self._ntori):
            E, Lz, I3 = self._Es[ii], self._Lzs[ii], self._I3s[ii]
            Wu = lambda u: self._Wu(u, E, Lz, I3)
            pos = numpy.where(Wu_all[ii] > 0.0)[0]
            if len(pos) > 0:
                ulo = us[pos[0] - 1] if pos[0] > 0 else 1e-8
                uin, uout = us[pos[0]], us[pos[-1]]
                uhi = us[pos[-1] + 1] if pos[-1] < len(us) - 1 else None
            else:
                # No positive sample: the oscillation may simply be narrower
                # than the mesh spacing (a near-shell torus); refine the
                # maximum of W_u around the largest sample before declaring
                # that there is no bound oscillation (guarding against NaNs
                # from potential evaluations at extreme coordinates)
                jmax = numpy.nanargmax(Wu_all[ii])
                res = minimize_scalar(
                    lambda u: -self._Wu(u, E, Lz, I3),
                    bounds=(us[max(jmax - 1, 0)], us[min(jmax + 1, len(us) - 1)]),
                    method="bounded",
                    options={"xatol": 1e-14},
                )
                if -res.fun <= 0.0:
                    raise ValueError(
                        f"No bound u oscillation found for torus {ii} "
                        f"(E={E}, Lz={Lz}, I3={I3})"
                    )
                uin = uout = res.x
                ulo, uhi = uin / 2.0, 2.0 * uout
            if uhi is None:
                # The oscillation extends beyond the scan: step out
                # geometrically until W_u turns properly negative (the
                # not-< comparison keeps stepping through NaNs from
                # potential evaluations at extreme coordinates)
                uhi = 2.0 * uout
                while not Wu(uhi) < 0.0:
                    if uhi > 200.0:
                        raise ValueError(
                            f"u oscillation of torus {ii} extends beyond "
                            f"u=200 (E={E}, Lz={Lz}, I3={I3}); the torus is "
                            "likely unbound"
                        )
                    uout = uhi
                    uhi *= 2.0
            self._umins[ii] = brentq(Wu, ulo, uin, xtol=1e-15, rtol=8.9e-16)
            self._umaxs[ii] = brentq(Wu, uout, uhi, xtol=1e-15, rtol=8.9e-16)
            if (self._umaxs[ii] - self._umins[ii]) < 1e-12 * (1.0 + self._umaxs[ii]):
                # Separated by a few ulp: nothing built on this
                # oscillation would be better than noise, so fail loudly
                # rather than return a plausible-looking zero
                raise ValueError(
                    f"u oscillation of torus {ii} is unresolvable in double "
                    f"precision (u_max - u_min = "
                    f"{self._umaxs[ii] - self._umins[ii]:e} for "
                    f"E={E}, Lz={Lz}, I3={I3}); the torus is a shell orbit "
                    "to within machine precision"
                )
            if Wv_all[ii, -1] <= 0.0:
                raise ValueError(
                    f"Midplane not reached for torus {ii}; no valid torus "
                    f"for (E={E}, Lz={Lz}, I3={I3})"
                )
            Wv = lambda v: self._Wv(v, E, Lz, I3)
            neg = numpy.where(Wv_all[ii] < 0.0)[0]
            self._vmins[ii] = (
                brentq(
                    Wv,
                    vs[neg[-1]],
                    numpy.pi / 2.0,
                    xtol=1e-15,
                    rtol=8.9e-16,
                )
                if len(neg) > 0
                else 1e-8
            )

    ############################ SETUP: QUADRATURES ###########################
    def _dWu(self, u, E, Lz):
        """dW_u/du (independent of I3); u may be an array of any shape (the
        wrapper's derivative only accepts 1D, so it is evaluated flat)"""
        u = numpy.asarray(u)
        dUdu = self._staeckelwrap._dUdu(u.ravel()).reshape(u.shape)
        return (
            2.0 * self._delta**2.0 * (E * numpy.sinh(2.0 * u) - dUdu)
            + 2.0 * Lz**2.0 * numpy.cosh(u) / numpy.sinh(u) ** 3.0
        )

    def _dWv(self, v, E, Lz):
        """dW_v/dv (independent of I3); v may be an array of any shape"""
        v = numpy.asarray(v)
        dVdv = self._staeckelwrap._dVdv(v.ravel()).reshape(v.shape)
        return (
            2.0 * self._delta**2.0 * (E * numpy.sin(2.0 * v) + dVdv)
            + 2.0 * Lz**2.0 * numpy.cos(v) / numpy.sin(v) ** 3.0
        )

    def _compute_actions_frequencies_profiles(self):
        """Actions, the 3x3 period matrices, frequencies, and the cumulative
        angle profiles of all tori, all from two vectorized potential
        evaluations (one per degree of freedom). All integrands are written
        in terms of the regular ratio Q = W/[y(1-y)] (y = sin^2 chi/2), so
        that the action integrand is (D/4) sqrt(Q) sin^2(chi) and the 1/p
        profile integrands are f(q) D/sqrt(Q); near the turning points,
        where the direct evaluation of W is dominated by cancellation
        error, Q is replaced by its finite limits |W'(q_-+)| D computed
        from the analytic derivative of the momentum (the same masking as
        in the 1D and spherical quadratures)"""
        d2 = self._delta**2.0
        mid = 0.5 * (self._chi[:-1] + self._chi[1:])
        half = 0.5 * (self._chi[1:] - self._chi[:-1])
        nodes = (mid[:, None] + half[:, None] * _GLX[None, :]).ravel()
        sin_nodes = numpy.sin(nodes)
        y_nodes = numpy.sin(nodes / 2.0) ** 2.0
        y1my = y_nodes * (1.0 - y_nodes)
        npan, ngl = len(half), len(_GLX)

        def profiles(qmin, qmax, W_of_q, dW_of_q, fEs, fIs, fLs):
            D = qmax - qmin
            # q(chi) at the nodes for all tori: (ntori, nnodes)
            q = qmin[:, None] + D[:, None] * y_nodes[None, :]
            W = W_of_q(q)
            with numpy.errstate(invalid="ignore", divide="ignore"):
                Q = W / y1my[None, :]
            # Near the turning points W is a difference of O(1) terms and
            # is cancellation-dominated, so reconstruct Q there from the
            # analytic derivative, W ~ (q - q0) [W'(q0) + W'(q)]/2. The
            # switch is relative to the size of W on the torus, not a fixed
            # cut in the anomaly: the signal near a turning point scales as
            # W' D y while the noise does not, so for a thin oscillation a
            # fixed cut leaves nodes in the noise, where 1/sqrt(Q) blows up
            dWq = dW_of_q(q)
            Qtp = numpy.where(
                y_nodes[None, :] < 0.5,
                D[:, None]
                * (dW_of_q(qmin)[:, None] + dWq)
                / 2.0
                / (1.0 - y_nodes[None, :]),
                D[:, None]
                * (-dW_of_q(qmax)[:, None] - dWq)
                / 2.0
                / numpy.maximum(y_nodes[None, :], 1e-300),
            )
            reliable = (y1my[None, :] > 1e-6) & (
                W > 1e-6 * numpy.nanmax(W, axis=1, keepdims=True)
            )
            Q = numpy.where(reliable, Q, Qtp)
            Q[~numpy.isfinite(Q) | (Q < numpy.finfo(float).tiny)] = numpy.finfo(
                float
            ).tiny
            sqQ = numpy.sqrt(Q)
            # action: (D/4) int sqrt(Q) sin^2(chi) dchi
            act_vals = sqQ * (D[:, None] / 4.0) * sin_nodes[None, :] ** 2.0
            action = (
                (half[None, :, None] * _GLW[None, None, :])
                * act_vals.reshape(self._ntori, npan, ngl)
            ).sum(axis=(-1, -2))
            # cumulative 1/p profiles: int f(q) D/sqrt(Q) dchi
            base = D[:, None] / sqQ
            cums = []
            for f in (fEs, fIs, fLs):
                vals = f(q) * base
                pans = (
                    (half[None, :, None] * _GLW[None, None, :])
                    * vals.reshape(self._ntori, npan, ngl)
                ).sum(axis=-1)
                cums.append(
                    numpy.hstack(
                        (
                            numpy.zeros((self._ntori, 1)),
                            numpy.cumsum(pans, axis=1),
                        )
                    )
                )
            return action, cums

        Es, Lzs, I3s = self._Es, self._Lzs, self._I3s
        actu, (PEu, PIu, PLu) = profiles(
            self._umins,
            self._umaxs,
            lambda q: self._Wu(q, Es[:, None], Lzs[:, None], I3s[:, None]),
            lambda q: self._dWu(
                q,
                Es.reshape((-1,) + (1,) * (numpy.ndim(q) - 1)),
                Lzs.reshape((-1,) + (1,) * (numpy.ndim(q) - 1)),
            ),
            lambda q: d2 * numpy.sinh(q) ** 2.0,
            lambda q: d2 * numpy.ones_like(q),
            lambda q: Lzs[:, None] / numpy.sinh(q) ** 2.0,
        )
        actv, (PEv, PIv, PLv) = profiles(
            self._vmins,
            numpy.pi - self._vmins,
            lambda q: self._Wv(q, Es[:, None], Lzs[:, None], I3s[:, None]),
            lambda q: self._dWv(
                q,
                Es.reshape((-1,) + (1,) * (numpy.ndim(q) - 1)),
                Lzs.reshape((-1,) + (1,) * (numpy.ndim(q) - 1)),
            ),
            lambda q: d2 * numpy.sin(q) ** 2.0,
            lambda q: d2 * numpy.ones_like(q),
            lambda q: Lzs[:, None] / numpy.sin(q) ** 2.0,
        )
        self._jr = actu / numpy.pi
        self._jz = actv / numpy.pi
        # pi dJ_R = PEu[-1] dE - PIu[-1] dI3 - PLu[-1] dLz, and likewise for
        # J_z. J_phi = L_z makes the third row (0,0,1), so M is
        # block-triangular and its inverse follows from the 2x2 (E,I3) block
        # -- the structure the forward actionAngleStaeckel also uses
        a11 = PEu[:, -1] / numpy.pi
        a12 = -PIu[:, -1] / numpy.pi
        a13 = -PLu[:, -1] / numpy.pi
        a21 = PEv[:, -1] / numpy.pi
        a22 = PIv[:, -1] / numpy.pi
        a23 = -PLv[:, -1] / numpy.pi
        det = a11 * a22 - a12 * a21
        self._dEI3Lz_dJ = numpy.zeros((self._ntori, 3, 3))
        # d(E,I3)/d(J_R,J_z) from the inverse of the 2x2 block
        self._dEI3Lz_dJ[:, 0, 0] = a22 / det
        self._dEI3Lz_dJ[:, 0, 1] = -a12 / det
        self._dEI3Lz_dJ[:, 1, 0] = -a21 / det
        self._dEI3Lz_dJ[:, 1, 1] = a11 / det
        # d(E,I3)/dJ_phi = -[2x2 inverse] . (dJ_R/dLz, dJ_z/dLz)
        self._dEI3Lz_dJ[:, 0, 2] = -(a22 * a13 - a12 * a23) / det
        self._dEI3Lz_dJ[:, 1, 2] = -(-a21 * a13 + a11 * a23) / det
        self._dEI3Lz_dJ[:, 2, 2] = 1.0
        self._OmegaR = self._dEI3Lz_dJ[:, 0, 0]
        self._Omegaz = self._dEI3Lz_dJ[:, 0, 1]
        self._Omegaphi = self._dEI3Lz_dJ[:, 0, 2]
        self._Pu = (PEu, PIu, PLu)
        self._Pv = (PEv, PIv, PLv)

    def _build_angle_profile_splines(self):
        """The six per-torus angle-profile splines A_i(chi_u), B_i(chi_v):
        theta_i = sum_X dX/dJ_i dW/dX with dW/dE = PEu + PEv,
        dW/dI3 = -PIu + PIv, and dW/dLz = -PLu - PLv (+ phi)"""
        PEu, PIu, PLu = self._Pu
        PEv, PIv, PLv = self._Pv
        self._Aprof, self._Bprof, self._dAprof, self._dBprof = [], [], [], []
        for ii in range(self._ntori):
            Minv = self._dEI3Lz_dJ[ii]
            A = [
                InterpolatedUnivariateSpline(
                    self._chi,
                    Minv[0, i] * PEu[ii] - Minv[1, i] * PIu[ii] - Minv[2, i] * PLu[ii],
                    k=5,
                )
                for i in range(3)
            ]
            B = [
                InterpolatedUnivariateSpline(
                    self._chi,
                    Minv[0, i] * PEv[ii] + Minv[1, i] * PIv[ii] - Minv[2, i] * PLv[ii],
                    k=5,
                )
                for i in range(3)
            ]
            self._Aprof.append(A)
            self._Bprof.append(B)
            self._dAprof.append([a.derivative() for a in A])
            self._dBprof.append([b.derivative() for b in B])

    ############################ SETUP: TORUS GRID ############################
    def _circular_orbit(self, Lz):
        """Radius and energy of the circular orbit with this L_z"""
        Rc = rl(self._pot, Lz, use_physical=False)
        return Rc, evaluatePotentials(
            self._pot, Rc, 0.0, use_physical=False
        ) + Lz**2.0 / 2.0 / Rc**2.0

    def _I3_planar(self, E, Lz):
        """I3 of the J_z = 0 edge: closed form in the V(pi/2) = 0 gauge,
        because W_v(pi/2) = 2 delta^2 [E + I3] - L_z^2 vanishes there"""
        return Lz**2.0 / 2.0 / self._delta**2.0 - E

    def _I3_shell(self, E, Lz, return_u=False):
        """I3 of the J_R = 0 edge, where W_u acquires a double root; with
        return_u, also the u of the shell (the double root)"""

        def argmaxWu(I3):
            return minimize_scalar(
                lambda u: -self._Wu(u, E, Lz, I3),
                bounds=(1e-3, 20.0),
                method="bounded",
                options={"xatol": 1e-13},
            )

        def maxWu(I3):
            return -argmaxWu(I3).fun

        lo = self._I3_planar(E, Lz)
        hi = lo + 1.0
        while maxWu(hi) > 0.0:
            hi += 1.0
        Ish = brentq(maxWu, lo, hi, xtol=1e-14)
        if return_u:
            return Ish, argmaxWu(Ish).x
        return Ish

    def _torus_index(self, jr, jphi, jz):
        indx = numpy.nanargmin(
            numpy.fabs(jr - self._jr)
            + numpy.fabs(jphi - self._Lzs)
            + numpy.fabs(jz - self._jz)
        )
        if (
            numpy.fabs(jr - self._jr[indx]) > 1e-8
            or numpy.fabs(jphi - self._Lzs[indx]) > 1e-8
            or numpy.fabs(jz - self._jz[indx]) > 1e-8
        ):
            raise ValueError(
                "Given actions not found among the actions of the tori set up "
                "in this actionAngleStaeckelInverse instance"
            )
        return indx

    def _fold(self, P, w):
        """P on [0,pi] extended over the full loop, w in [0,2pi)."""
        w = numpy.mod(w, 2.0 * numpy.pi)
        return numpy.where(
            w <= numpy.pi, P(w), 2.0 * P(numpy.pi) - P(2.0 * numpy.pi - w)
        )

    def _dfold(self, dP, w):
        w = numpy.mod(w, 2.0 * numpy.pi)
        return numpy.where(w <= numpy.pi, dP(w), dP(2.0 * numpy.pi - w))

    def _evaluate(self, jr, jphi, jz, angler, anglephi, anglez, **kwargs):
        return self._xvFreqs(jr, jphi, jz, angler, anglephi, anglez, **kwargs)[:6]

    def _xvFreqs(self, jr, jphi, jz, angler, anglephi, anglez, **kwargs):
        if kwargs.get("integrals", False) and not self._interp:
            raise ValueError(
                "integrals=True requires an actionAngleStaeckelInverse "
                "set up with setup_interp=True"
            )
        if self._interp:
            if kwargs.get("integrals", False):
                x = self._canon_coords_integrals(jr, jphi, jz)
                v = self._canon_table_eval(numpy.atleast_2d(x))[:, 0]
                return self._xvFreqs_canonical_interp(
                    v[0], jphi, v[1], angler, anglephi, anglez, x=x
                )
            return self._xvFreqs_canonical_interp(
                jr, jphi, jz, angler, anglephi, anglez
            )
        ii = self._torus_index(jr, jphi, jz)
        if self._canonical:
            return self._xvFreqs_canonical(ii, angler, anglephi, anglez)
        return self._solve_and_map(
            self._Aprof[ii],
            self._Bprof[ii],
            self._dAprof[ii],
            self._dBprof[ii],
            self._Es[ii],
            self._Lzs[ii],
            self._I3s[ii],
            self._umins[ii],
            self._umaxs[ii],
            self._vmins[ii],
            self._OmegaR[ii],
            self._Omegaphi[ii],
            self._Omegaz[ii],
            angler,
            anglephi,
            anglez,
        )

    def _solve_and_map(
        self,
        A,
        B,
        dA,
        dB,
        E,
        Lz,
        I3,
        umin,
        umaxx,
        vmin,
        OmR,
        Omphi,
        Omz,
        angler,
        anglephi,
        anglez,
    ):
        """Invert the angle system on one torus and map the result to
        (R, vR, vT, z, vz, phi); shared by the direct and interpolated paths.
        A and B are the per-torus angle-profile splines of the direct
        construction."""
        thR = numpy.atleast_1d(numpy.array(angler, dtype="float"))
        thz = numpy.atleast_1d(numpy.array(anglez, dtype="float"))
        thphi = numpy.atleast_1d(numpy.array(anglephi, dtype="float"))
        wu, wv = numpy.copy(thR), numpy.copy(thz)
        unconv = numpy.ones(wu.shape, dtype="bool")
        for _ in range(self._maxiter):
            twu, twv = wu[unconv], wv[unconv]
            f0 = self._fold(A[0], twu) + self._fold(B[0], twv) - thR[unconv]
            f1 = (
                self._fold(A[1], twu)
                + self._fold(B[1], twv)
                + self._anglez0
                - thz[unconv]
            )
            J00, J01 = self._dfold(dA[0], twu), self._dfold(dB[0], twv)
            J10, J11 = self._dfold(dA[1], twu), self._dfold(dB[1], twv)
            f0 = (f0 + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
            f1 = (f1 + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
            det = J00 * J11 - J01 * J10
            dwu = (J11 * f0 - J01 * f1) / det
            dwv = (-J10 * f0 + J00 * f1) / det
            step = numpy.maximum(numpy.fabs(dwu), numpy.fabs(dwv))
            lim = numpy.minimum(1.0, 0.5 / numpy.maximum(step, 1e-30))
            wu[unconv] -= dwu * lim
            wv[unconv] -= dwv * lim
            unconv[unconv] = step >= self._angle_tol
            if not numpy.any(unconv):
                break
        # phases -> (u, v, p_u, p_v)
        Du, Dv = umaxx - umin, numpy.pi - 2.0 * vmin
        wum, wvm = numpy.mod(wu, 2.0 * numpy.pi), numpy.mod(wv, 2.0 * numpy.pi)
        chiu = numpy.where(wum <= numpy.pi, wum, 2.0 * numpy.pi - wum)
        chiv = numpy.where(wvm <= numpy.pi, wvm, 2.0 * numpy.pi - wvm)
        su = numpy.where(wum <= numpy.pi, 1.0, -1.0)
        sv = numpy.where(wvm <= numpy.pi, 1.0, -1.0)
        u = umin + Du * numpy.sin(chiu / 2.0) ** 2.0
        v = vmin + Dv * numpy.sin(chiv / 2.0) ** 2.0
        # A degenerate oscillation carries no momentum, and saying so
        # explicitly matters: p = sqrt(W) turns the ~1e-16 residual of
        # W_v(pi/2) into a vertical velocity of ~1e-8 in a planar orbit
        pu = (
            numpy.zeros_like(u)
            if Du <= 1e-12
            else su * numpy.sqrt(numpy.clip(self._Wu(u, E, Lz, I3), 0.0, None))
        )
        pv = (
            numpy.zeros_like(v)
            if Dv <= 1e-12
            else sv * numpy.sqrt(numpy.clip(self._Wv(v, E, Lz, I3), 0.0, None))
        )
        phi = thphi - self._fold(A[2], wu) - self._fold(B[2], wv)
        # (u, v, p_u, p_v) -> (R, z, vR, vz)
        sh, ch = numpy.sinh(u), numpy.cosh(u)
        sn, cs = numpy.sin(v), numpy.cos(v)
        R, z = coords.uv_to_Rz(u, v, delta=self._delta)
        den = self._delta * (sh**2.0 + sn**2.0)
        vR = (pu * ch * sn + pv * sh * cs) / den
        vz = (pu * sh * cs - pv * ch * sn) / den
        vT = Lz / R
        return (
            R,
            vR,
            vT,
            z,
            vz,
            phi % (2.0 * numpy.pi),
            OmR,
            Omphi,
            Omz,
        )

    def _Freqs(self, jr, jphi, jz, **kwargs):
        if kwargs.get("integrals", False) and not self._interp:
            raise ValueError(
                "integrals=True requires an actionAngleStaeckelInverse "
                "set up with setup_interp=True"
            )
        if self._interp:
            if kwargs.get("integrals", False):
                x = self._canon_coords_integrals(jr, jphi, jz)
            else:
                x = self._canon_coords(float(jr), float(jphi), float(jz))
            _, dq = self._canon_family_chains(x)
            return (dq[2, 0], dq[2, 1], dq[2, 2])
        ii = self._torus_index(jr, jphi, jz)
        return (self._OmegaR[ii], self._Omegaphi[ii], self._Omegaz[ii])

    ################## CANONICAL (momentum-matched) CONSTRUCTION ##############
    # The Stage-3 canonical path (STAECKEL_CANONICAL_MATH.md section 10): the
    # torus is lifted onto its equal-action isochrone torus by per-degree
    # momentum-matched point transformations (cumulative radial/vertical
    # actions matched; the anomaly maps eta(tau) - tau are pure sine series),
    # cotangent-lifted, so the lift is exactly canonical for any stored maps
    # and the 2-D generating-function content collapses to the stored maps'
    # truncation residual.
    @staticmethod
    def _spec_coeffs1(f):
        """Fourier coefficients on the offset grid tau_j = 2 pi (j+1/2)/N"""
        N = len(f)
        k = numpy.arange(N // 2 + 1)
        return numpy.fft.rfft(f) / N * numpy.exp(-1j * k * numpy.pi / N)

    @staticmethod
    def _spec_eval1(c, tau, deriv=False):
        k = numpy.arange(len(c))
        w = numpy.ones(len(c))
        w[1:-1] = 2.0
        cc = c * (1j * k) if deriv else c
        ph = numpy.exp(1j * numpy.atleast_1d(tau)[:, None] * k[None, :])
        return numpy.real(ph @ (w * cc))

    @staticmethod
    def _spec_coeffs2(f):
        """2-D Fourier coefficients on the offset product grid"""
        N = f.shape[0]
        c = numpy.fft.fft2(f) / N**2
        k = numpy.fft.fftfreq(N, d=1.0 / N)
        return c * numpy.exp(-1j * numpy.pi / N * (k[:, None] + k[None, :]))

    @staticmethod
    def _spec_eval2(c, tu, tv):
        """Evaluate the 2-D series at arbitrary (tau_u, tau_v) pairs"""
        N = c.shape[0]
        k = numpy.fft.fftfreq(N, d=1.0 / N)
        phu = numpy.exp(1j * numpy.atleast_1d(tu)[:, None] * k[None, :])
        phv = numpy.exp(1j * numpy.atleast_1d(tv)[:, None] * k[None, :])
        return numpy.real(numpy.einsum("pk,kl,pl->p", phu, c, phv))

    def _iso_E_of_Jr(self, Jr, L):
        """Energy of the toy torus with radial action Jr: closed form"""
        CA = 0.5 * (L + numpy.sqrt(L**2 + 4.0 * self._GMc * self._bc))
        return -(self._GMc**2) / (2.0 * (Jr + CA) ** 2)

    def _toy_ae(self, JAr, LA):
        """Semi-major axis and eccentricity of the toy torus (J^A_r, L^A)"""
        H = self._iso_E_of_Jr(JAr, LA)
        a = -self._GMc / 2.0 / H - self._bc
        return a, numpy.sqrt(numpy.clip(1.0 + LA**2 / (2.0 * H * a**2), 0.0, None))

    def _toy_r_profile(self, a, e, eta):
        """Toy radial loop: radius, momentum, dr/deta at anomaly eta"""
        b = self._bc
        y = 1.0 - e * numpy.cos(eta)
        rA = a * numpy.sqrt(y * (y + 2.0 * b / a))
        pA = numpy.sqrt(self._GMc / (a + b)) * a * e * numpy.sin(eta) / rA
        drAdeta = (
            a * e * numpy.sin(eta) * (y + b / a) / numpy.sqrt(y * (y + 2.0 * b / a))
        )
        return rA, pA, drAdeta

    @staticmethod
    def _toy_th_profile(LA, Lz, thmin, eta):
        """Toy vertical loop: polar angle, momentum, dtheta/deta at eta"""
        th = 0.5 * numpy.pi - (0.5 * numpy.pi - thmin) * numpy.cos(eta)
        pth2 = numpy.clip(LA**2 - Lz**2 / numpy.sin(th) ** 2, 0.0, None)
        pth = numpy.sign(numpy.sin(eta)) * numpy.sqrt(pth2)
        dthdeta = (0.5 * numpy.pi - thmin) * numpy.sin(eta)
        return th, pth, dthdeta

    def _cum_match(self, ft, fA, tau):
        """The momentum-matching anomaly map eta(tau): match the cumulative
        actions A_t(tau) = A_A(eta) spectrally (both integrands sampled on
        the same offset grid; equal actions make the linear parts agree, so
        eta - tau is periodic and, by parity, a pure sine series)"""
        N = len(tau)
        k = numpy.fft.fftfreq(N, d=1.0 / N)

        def _antider(f):
            fh = numpy.fft.fft(f - numpy.mean(f))
            ah = numpy.zeros_like(fh)
            ah[1:] = fh[1:] / (1j * k[1:])
            return numpy.real(numpy.fft.ifft(ah))

        mt, mA = numpy.mean(ft), numpy.mean(fA)
        scale = mt / mA
        qt = _antider(ft)
        At = mt * tau + qt - self._spec_eval1(self._spec_coeffs1(qt), 0.0)[0]
        qA = _antider(fA)
        qA0 = self._spec_eval1(self._spec_coeffs1(qA), 0.0)[0]
        cqA = self._spec_coeffs1(qA)
        cfA = self._spec_coeffs1(fA)
        eta = numpy.array(tau)
        for _ in range(self._maxiter):
            fres = scale * (mA * eta + self._spec_eval1(cqA, eta) - qA0) - At
            fp = numpy.maximum(scale * self._spec_eval1(cfA, eta), 1e-10 * mt)
            de = numpy.clip(-fres / fp, -0.5, 0.5)
            eta += de
            if numpy.max(numpy.fabs(fres)) < 1e-13 * max(mt, 1e-10):
                break
        else:
            raise RuntimeError(
                "Newton's method for the momentum-matching anomaly map did not converge"
            )
        smat = numpy.sin(tau[:, None] * self._nforDm[None, :])
        Dm = 2.0 * numpy.mean((eta - tau)[:, None] * smat, axis=0)
        return Dm

    def _tau_of_eta(self, eta, Dm, where=""):
        """Invert the stored anomaly map (monotone) for tau"""
        ms = self._nforDm
        # start from the one-term inversion rather than from eta itself: the
        # map is tau + sum_m D_m sin(m tau), so eta - sum_m D_m sin(m eta) is
        # already correct to second order in the (small) coefficients, and
        # Newton then needs a handful of steps instead of a few dozen
        eta = numpy.asarray(eta, dtype="float")
        x = eta - numpy.sin(eta[:, None] * ms[None, :]) @ Dm
        for _ in range(self._maxiter):
            f = x + numpy.sin(x[:, None] * ms[None, :]) @ Dm - eta
            fp = 1.0 + numpy.cos(x[:, None] * ms[None, :]) @ (ms * Dm)
            dx = numpy.clip(-f / fp, -0.5, 0.5)
            x += dx
            worst = numpy.max(numpy.fabs(f))
            if worst < self._angle_tol:
                break
        else:
            # The stored map is monotone whenever sum_m m |D_m| < 1, which
            # the construction guarantees, so eta(tau) can always be
            # bracketed on [0, 2 pi] even where Newton has not converged.
            # Newton is the fast path and reaches its residual floor of
            # ~1e-16 in three or four steps here; this is the guarantee, not
            # the expectation, and it exists because a failure of this
            # inversion used to raise and take the whole evaluation with it.
            badDm = int(numpy.sum(~numpy.isfinite(Dm)))
            badeta = int(numpy.sum(~numpy.isfinite(eta)))
            if badDm or badeta:
                # Bracketing compares against NaN, and every such comparison
                # is False, so it would converge on an endpoint and return it
                # as a root.  Fail loudly instead of silently, and say WHICH
                # of the two is bad: the coefficients come from the stored
                # family and the anomaly from the request, so they go wrong
                # for different reasons and want different fixes.
                raise RuntimeError(
                    "Newton's method for the map anomaly did not converge, "
                    "and the map anomaly or its coefficients are not finite: "
                    f"{badDm} of {Dm.size} coefficients and {badeta} of "
                    f"{eta.size} anomalies are non-finite"
                    + (f" ({where})" if where else "")
                )
            if numpy.sum(ms * numpy.fabs(Dm)) >= 1.0:
                # sum_m m |D_m| >= 1 admits d eta / d tau <= 0: the map may
                # fold, the root need not be unique, and bracketing would
                # return a root without saying which.  That is a broken
                # stored map rather than a slow solve, so it still raises.
                raise RuntimeError(
                    "Newton's method for the map anomaly did not converge, "
                    "and the stored map is not monotone"
                )
            f = x + numpy.sin(x[:, None] * ms[None, :]) @ Dm - eta
            bad = numpy.fabs(f) >= self._angle_tol
            for ii in numpy.arange(len(x))[bad]:
                lo, hi = 0.0, 2.0 * numpy.pi
                for _ in range(200):
                    mid = 0.5 * (lo + hi)
                    if mid + numpy.sum(numpy.sin(mid * ms) * Dm) - eta[ii] < 0.0:
                        lo = mid
                    else:
                        hi = mid
                x[ii] = 0.5 * (lo + hi)
        return x

    def _setup_canonical(self):
        """The frozen toy (rotation-curve fit over the sampled radial range)
        and, per torus: the equal-action closure, the two momentum-matched
        anomaly maps, the truncated-map-consistent lift, the product-grid
        correspondence through the analytic isochrone, the zero-mode labels
        (Stokes-checked against the direct quadrature actions), and the
        2-D correspondence tables the canonical evaluation reads"""
        N = self._ncanon
        tau = 2.0 * numpy.pi * (numpy.arange(N) + 0.5) / N
        # the sampled radial range, in closed prolate forms
        rlos = self._delta * numpy.sinh(self._umins)
        rhis = self._delta * numpy.sqrt(
            numpy.sinh(self._umaxs) ** 2 + numpy.cos(self._vmins) ** 2
        )
        rlo = max(numpy.min(rlos), 1e-3 * numpy.max(rhis))
        rhi = numpy.max(rhis)
        rf = numpy.geomspace(rlo, rhi, 25)
        lnvc2 = numpy.log(vcirc(self._pot, rf, use_physical=False) ** 2)

        def _vc2cost(x):
            GMf, bf = numpy.exp(x)
            sf = numpy.sqrt(bf**2 + rf**2)
            return numpy.sum(
                (numpy.log(GMf * rf**2 / (sf * (bf + sf) ** 2)) - lnvc2) ** 2
            )

        if self._isochrone_ab is not None:
            # The compensation's closed-form auxiliary chains assume ONE
            # frozen isochrone across the whole family, so an adaptive build
            # fits it on a single representative node and hands it to every
            # other node through this override.
            self._GMc, self._bc = self._isochrone_ab
        else:
            res = minimize(
                _vc2cost,
                numpy.log(
                    [
                        rhi * vcirc(self._pot, rhi, use_physical=False) ** 2,
                        0.1 * numpy.sqrt(rlo * rhi),
                    ]
                ),
                method="Nelder-Mead",
            )
            self._GMc, self._bc = numpy.exp(res.x)
        ipc = IsochronePotential(amp=self._GMc, b=self._bc)
        self._aAIc = actionAngleIsochrone(ip=ipc)
        self._aAIinvc = actionAngleIsochroneInverse(ip=ipc)
        ntori = self._ntori
        self._can_LA = numpy.empty(ntori)
        self._can_a = numpy.empty(ntori)
        self._can_e = numpy.empty(ntori)
        self._can_thmin = numpy.empty(ntori)
        self._can_Dmu = numpy.empty((ntori, self._npt))
        self._can_Dmv = numpy.empty((ntori, self._npt))
        self._can_labels = numpy.empty((ntori, 2))
        nk = N
        self._can_cJr = numpy.empty((ntori, nk, nk), dtype=complex)
        self._can_cJz = numpy.empty((ntori, nk, nk), dtype=complex)
        self._can_cDR = numpy.empty((ntori, nk, nk), dtype=complex)
        self._can_cDz = numpy.empty((ntori, nk, nk), dtype=complex)
        self._can_cDphi = numpy.empty((ntori, nk, nk), dtype=complex)
        self._can_maxdev = 0.0
        self._can_stokes = 0.0
        for ii in range(ntori):
            self._canonical_torus_tables(ii)
        return None

    def _canonical_torus_tables(self, ii):
        """The canonical tables of one torus: the equal-action closure, the
        two momentum-matched anomaly maps, the truncated-map-consistent
        lift, the product-grid correspondence, the zero-mode labels, and
        the 2-D correspondence tables"""
        N = self._ncanon
        tau = 2.0 * numpy.pi * (numpy.arange(N) + 0.5) / N
        E, Lz, I3 = self._Es[ii], self._Lzs[ii], self._I3s[ii]
        Du = self._umaxs[ii] - self._umins[ii]
        Dv = numpy.pi - 2.0 * self._vmins[ii]
        u = self._umins[ii] + Du * numpy.sin(tau / 2.0) ** 2
        v = self._vmins[ii] + Dv * numpy.sin(tau / 2.0) ** 2
        pu = numpy.where(tau < numpy.pi, 1.0, -1.0) * numpy.sqrt(
            numpy.clip(self._Wu(u, E, Lz, I3), 0.0, None)
        )
        pv = numpy.where(tau < numpy.pi, 1.0, -1.0) * numpy.sqrt(
            numpy.clip(self._Wv(v, E, Lz, I3), 0.0, None)
        )
        dudtau = 0.5 * Du * numpy.sin(tau)
        dvdtau = 0.5 * Dv * numpy.sin(tau)
        # equal-action closure, all closed forms
        LA = self._jz[ii] + numpy.fabs(Lz)
        EA = self._iso_E_of_Jr(self._jr[ii], LA)
        a = -self._GMc / (2.0 * EA) - self._bc
        e = numpy.sqrt(1.0 + LA**2 / (2.0 * EA * a**2))
        thmin = numpy.arcsin(numpy.clip(numpy.fabs(Lz) / LA, 0.0, 1.0))
        # the two anomaly maps and the truncated-map-consistent lifts
        _, pA_eta, drA_eta = self._toy_r_profile(a, e, tau)
        Dmu = self._cum_match(pu * dudtau, pA_eta * drA_eta, tau)
        _, pth_eta, dth_eta = self._toy_th_profile(LA, Lz, thmin, tau)
        Dmv = self._cum_match(pv * dvdtau, pth_eta * dth_eta, tau)
        smat = numpy.sin(tau[:, None] * self._nforDm[None, :])
        cmat = numpy.cos(tau[:, None] * self._nforDm[None, :])
        etau = tau + smat @ Dmu
        detau = 1.0 + cmat @ (self._nforDm * Dmu)
        rA, _, drA_t = self._toy_r_profile(a, e, etau)
        pAr = pu * dudtau / (drA_t * detau)
        etav = tau + smat @ Dmv
        detav = 1.0 + cmat @ (self._nforDm * Dmv)
        thetaA, _, dth_t = self._toy_th_profile(LA, Lz, thmin, etav)
        pAth = pv * dvdtau / (dth_t * detav)
        # product-grid correspondence through the analytic isochrone
        R2 = rA[:, None] * numpy.sin(thetaA)[None, :]
        z2 = rA[:, None] * numpy.cos(thetaA)[None, :]
        vr2 = pAr[:, None] * numpy.ones(N)[None, :]
        vth2 = numpy.ones(N)[:, None] * pAth[None, :] / rA[:, None]
        sn2 = numpy.ones(N)[:, None] * numpy.sin(thetaA)[None, :]
        cs2 = numpy.ones(N)[:, None] * numpy.cos(thetaA)[None, :]
        vR2 = vr2 * sn2 + vth2 * cs2
        vz2 = vr2 * cs2 - vth2 * sn2
        vT2 = Lz / R2
        with numpy.errstate(invalid="ignore", divide="ignore"):
            o = self._aAIc.actionsFreqsAngles(
                R2.ravel(),
                vR2.ravel(),
                vT2.ravel(),
                z2.ravel(),
                vz2.ravel(),
                numpy.zeros(N * N),
            )
        JAr = numpy.atleast_1d(o[0]).reshape(N, N)
        JAz = numpy.atleast_1d(o[2]).reshape(N, N)
        thetaAr = numpy.atleast_1d(o[6]).reshape(N, N)
        thAphi = numpy.atleast_1d(o[7]).reshape(N, N)
        thAz = numpy.atleast_1d(o[8]).reshape(N, N)
        if numpy.any(~numpy.isfinite(JAr + JAz)):
            raise RuntimeError(
                "The toy correspondence failed for the (E, Lz, I3) = "
                f"({E}, {Lz}, {I3}) torus (unbound lifted samples)"
            )
        # target angles at the product samples, from the direct engine's
        # own profiles (exact convention match by construction)
        A, B = self._Aprof[ii], self._Bprof[ii]
        thR_t = self._fold(A[0], tau)[:, None] + self._fold(B[0], tau)[None, :]
        thz_t = (
            self._fold(A[1], tau)[:, None]
            + self._fold(B[1], tau)[None, :]
            + self._anglez0
        )
        thphi_t = self._fold(A[2], tau)[:, None] + self._fold(B[2], tau)[None, :]

        # correspondence-difference fields: periodic in both anomalies
        def _wrap2(f):
            f = numpy.unwrap(f, axis=0)
            return numpy.unwrap(f, axis=1)

        DR = _wrap2(thetaAr - thR_t)
        Dz = _wrap2(thAz - thz_t)
        Dphi = _wrap2(thAphi - thphi_t)
        # zero-mode labels with the 2-D angle-measure weight
        k1 = numpy.fft.fftfreq(N, d=1.0 / N)

        def _ddtau(f, axis):
            return numpy.real(
                numpy.fft.ifft(
                    1j
                    * (k1.reshape(-1, 1) if axis == 0 else k1.reshape(1, -1))
                    * numpy.fft.fft(f, axis=axis),
                    axis=axis,
                )
            )

        duu = 1.0 + _ddtau(DR + thR_t - tau[:, None], 0)
        duv = _ddtau(DR + thR_t, 1)
        dvu = _ddtau(Dz + thz_t, 0)
        dvv = 1.0 + _ddtau(Dz + thz_t - tau[None, :], 1)
        det = duu * dvv - duv * dvu
        labr = numpy.mean(JAr * det)
        labz = numpy.mean(JAz * det)
        self._can_labels[ii] = [labr, labz]
        self._can_LA[ii] = LA
        self._can_a[ii] = a
        self._can_e[ii] = e
        self._can_thmin[ii] = thmin
        self._can_Dmu[ii] = Dmu
        self._can_Dmv[ii] = Dmv
        self._can_cJr[ii] = self._spec_coeffs2(JAr - labr)
        self._can_cJz[ii] = self._spec_coeffs2(JAz - labz)
        self._can_cDR[ii] = self._spec_coeffs2(DR)
        self._can_cDz[ii] = self._spec_coeffs2(Dz)
        self._can_cDphi[ii] = self._spec_coeffs2(Dphi)
        self._can_maxdev = max(
            self._can_maxdev,
            float(numpy.max(numpy.fabs(JAr - labr))),
            float(numpy.max(numpy.fabs(JAz - labz))),
        )
        self._can_stokes = max(
            self._can_stokes,
            float(numpy.fabs(labr - self._jr[ii])),
            float(numpy.fabs(labz - self._jz[ii])),
        )
        return None

    def _xvFreqs_canonical(self, ii, angler, anglephi, anglez):
        """Canonical discrete evaluation: solve the target angle system for
        the two anomalies (the direct engine's own 2-D Newton), read the
        toy angles and actions from the correspondence tables, delegate the
        full 3-D reconstruction to the analytic isochrone inverse, and
        un-lift per degree through the stored anomaly maps"""
        thR = numpy.atleast_1d(numpy.array(angler, dtype="float"))
        thphi = numpy.atleast_1d(numpy.array(anglephi, dtype="float"))
        thz = numpy.atleast_1d(numpy.array(anglez, dtype="float"))
        thR, thphi, thz = numpy.broadcast_arrays(thR, thphi, thz)
        A, B = self._Aprof[ii], self._Bprof[ii]
        dA, dB = self._dAprof[ii], self._dBprof[ii]
        wu, wv = numpy.copy(thR), numpy.copy(thz)
        for _ in range(self._maxiter):
            f0 = self._fold(A[0], wu) + self._fold(B[0], wv) - thR
            f1 = self._fold(A[1], wu) + self._fold(B[1], wv) + self._anglez0 - thz
            f0 = (f0 + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
            f1 = (f1 + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
            J00, J01 = self._dfold(dA[0], wu), self._dfold(dB[0], wv)
            J10, J11 = self._dfold(dA[1], wu), self._dfold(dB[1], wv)
            det = J00 * J11 - J01 * J10
            dwu = (J11 * f0 - J01 * f1) / det
            dwv = (-J10 * f0 + J00 * f1) / det
            step = numpy.maximum(numpy.fabs(dwu), numpy.fabs(dwv))
            lim = numpy.minimum(1.0, 0.5 / numpy.maximum(step, 1e-30))
            wu -= dwu * lim
            wv -= dwv * lim
            if numpy.max(step) < self._angle_tol:
                break
        else:
            raise RuntimeError("Newton's method for the target angles did not converge")
        tu = numpy.mod(wu, 2.0 * numpy.pi)
        tv = numpy.mod(wv, 2.0 * numpy.pi)
        JAr = self._can_labels[ii, 0] + self._spec_eval2(self._can_cJr[ii], tu, tv)
        JAz = self._can_labels[ii, 1] + self._spec_eval2(self._can_cJz[ii], tu, tv)
        thetaAr = thR + self._spec_eval2(self._can_cDR[ii], tu, tv)
        thAz = thz + self._spec_eval2(self._can_cDz[ii], tu, tv)
        thAphi = thphi + self._spec_eval2(self._can_cDphi[ii], tu, tv)
        Lz = self._Lzs[ii]
        out = numpy.empty((6, len(tu)))
        for jj in range(len(tu)):
            oo = self._aAIinvc._xvFreqs(
                JAr[jj], Lz, JAz[jj], thetaAr[jj], thAphi[jj], thAz[jj]
            )
            for kk in range(6):
                out[kk, jj] = oo[kk][0]
        Rt, vRt, vTt, zt, vzt, phit = self._canon_unlift(
            out,
            self._can_a[ii],
            self._can_e[ii],
            self._can_LA[ii],
            self._can_thmin[ii],
            self._can_Dmu[ii],
            self._can_Dmv[ii],
            self._umins[ii],
            self._umaxs[ii],
            self._vmins[ii],
            Lz,
        )
        return (
            Rt,
            vRt,
            vTt,
            zt,
            vzt,
            phit,
            self._OmegaR[ii],
            self._Omegaphi[ii],
            self._Omegaz[ii],
        )

    def _canon_unlift(
        self, out, a, e, LA, thmin, Dmu, Dmv, umin, umax, vmin, Lz, delta=None
    ):
        """Map the toy-chart reconstruction to the target chart: per-degree
        anomaly inversions through the stored maps (closed-form radius and
        linear polar-angle inversions), momenta through the per-degree
        action-flux groups (regular at all turning points), then prolate ->
        cylindrical exactly as the direct path"""
        R, vR, vT, z, vz, phi = out
        rA = numpy.sqrt(R**2 + z**2)
        vrA = (R * vR + z * vz) / rA
        pAth = vR * z - vz * R
        Du = umax - umin
        Dv = numpy.pi - 2.0 * vmin
        if Du < 1e-10:
            # shell: the u-oscillation is degenerate; the u-support is the
            # analytic shell spheroid and it carries no momentum
            u = numpy.full_like(rA, 0.5 * (umin + umax))
            pu = numpy.zeros_like(rA)
        if Dv < 1e-10 or (0.5 * numpy.pi - thmin) < 1e-8:
            # planar: the v-oscillation is degenerate
            v = numpy.full_like(rA, 0.5 * numpy.pi)
            pv = numpy.zeros_like(rA)
        # u-degree: eta from the closed-form radius inversion, tau from the
        # stored map, u from the cosine anomaly; p_u through the flux group
        y = (numpy.sqrt(self._bc**2 + rA**2) - self._bc) / a
        # A circular auxiliary has e = 0 and no radial anomaly, so (1 - y)/e
        # is 0/0 there and clip cannot rescue it, clip(nan) being nan.  That
        # is reached whenever J_R = 0 exactly, and only when the rounding
        # lands e on zero rather than on a tiny positive residue -- which is
        # why it appeared on one platform and intermittently.  The
        # u-oscillation is degenerate in that case and is handled above, so
        # the anomaly is arbitrary; it only has to be finite, since a nan
        # propagates into the map inversion and takes the evaluation down.
        esafe = numpy.where(numpy.asarray(e) > 1e-12, e, 1.0)
        coseta = numpy.clip(
            numpy.where(numpy.asarray(e) > 1e-12, (1.0 - y) / esafe, 1.0),
            -1.0,
            1.0,
        )
        sineta = numpy.sign(vrA) * numpy.sqrt(numpy.clip(1.0 - coseta**2, 0.0, None))
        etau = numpy.arctan2(sineta, coseta) % (2.0 * numpy.pi)
        if Du >= 1e-10:
            tuu = self._tau_of_eta(etau, Dmu, where="etau")
            u = umin + Du * numpy.sin(tuu / 2.0) ** 2
            gA = a * e * (y + self._bc / a) / numpy.sqrt(y * (y + 2.0 * self._bc / a))
            ms = self._nforDm
            detau = 1.0 + numpy.cos(tuu[:, None] * ms[None, :]) @ (ms * Dmu)
            sintu = numpy.sin(tuu)
            sru = numpy.where(
                numpy.fabs(sintu) > 1e-12,
                sineta / numpy.maximum(numpy.fabs(sintu), 1e-12) * numpy.sign(sintu),
                1.0,
            )
            pu = vrA * gA * sru * detau / (0.5 * Du)
        if Dv >= 1e-10 and (0.5 * numpy.pi - thmin) >= 1e-8:
            # v-degree: theta^A from the position, eta from the linear
            # cosine inversion, tau from the stored map; p_v through the
            # flux group
            ms = self._nforDm
            thetaA = numpy.arccos(numpy.clip(z / rA, -1.0, 1.0))
            cosetav = numpy.clip(
                (0.5 * numpy.pi - thetaA) / (0.5 * numpy.pi - thmin), -1.0, 1.0
            )
            sinetav = numpy.sign(pAth) * numpy.sqrt(
                numpy.clip(1.0 - cosetav**2, 0.0, None)
            )
            etav = numpy.arctan2(sinetav, cosetav) % (2.0 * numpy.pi)
            tvv = self._tau_of_eta(etav, Dmv, where="etav")
            v = vmin + Dv * numpy.sin(tvv / 2.0) ** 2
            detav = 1.0 + numpy.cos(tvv[:, None] * ms[None, :]) @ (ms * Dmv)
            sintv = numpy.sin(tvv)
            srv = numpy.where(
                numpy.fabs(sintv) > 1e-12,
                sinetav / numpy.maximum(numpy.fabs(sintv), 1e-12) * numpy.sign(sintv),
                1.0,
            )
            gth = 0.5 * numpy.pi - thmin
            pv = pAth * gth * srv * detav / (0.5 * Dv)
        # prolate -> cylindrical, exactly as the direct path
        sh, ch = numpy.sinh(u), numpy.cosh(u)
        sn, cs = numpy.sin(v), numpy.cos(v)
        dl = self._delta if delta is None else float(delta)
        Rt, zt = coords.uv_to_Rz(u, v, delta=dl)
        den = dl * (sh**2 + sn**2)
        vRt = (pu * ch * sn + pv * sh * cs) / den
        vzt = (pu * sh * cs - pv * ch * sn) / den
        return Rt, vRt, Lz / Rt, zt, vzt, phi % (2.0 * numpy.pi)

    ################## CANONICAL FAMILY (T2) ##################################
    def _canon_table_eval(self, x, deriv=None):
        """Evaluate the stacked, prefiltered 3-D canonical tables (and their
        own derivatives) at fractional grid coordinates x = (xL, xE, xI),
        vectorized over points; deriv is None or the axis (0, 1, 2) along
        which to take the tables' own first derivative, in grid-index units"""
        x = numpy.atleast_2d(x)
        npts = x.shape[0]
        vals = numpy.empty((self._canon_tab.shape[0], npts))
        idx = numpy.floor(x).astype(int)
        t = x - idx
        for ax in range(3):
            idx[:, ax] = numpy.clip(idx[:, ax], 0, self._canon_shape[ax] - 2)
        t = x - idx
        wts = []
        for ax in range(3):
            w = (
                _bspline_dweights(t[:, ax])
                if deriv == ax
                else _bspline_weights(t[:, ax])
            )
            wts.append(w.T)  # (npts, 4)
        # gather the 4x4x4 neighborhoods: pad offset is 2
        i0 = idx[:, 0][:, None] + numpy.arange(4)[None, :] + 1
        i1 = idx[:, 1][:, None] + numpy.arange(4)[None, :] + 1
        i2 = idx[:, 2][:, None] + numpy.arange(4)[None, :] + 1
        block = self._canon_tab[
            :,
            i0[:, :, None, None],
            i1[:, None, :, None],
            i2[:, None, None, :],
        ]
        vals = numpy.einsum("qpabc,pa,pb,pc->qp", block, wts[0], wts[1], wts[2])
        return vals

    def _target_box(self, Rmin, Rmax, Rinf, wpad):
        """The padded (L_z, w_E, w_I) box that localizes the grid on the
        target's tori.  Every target point is labelled through the same
        relations the node lattice uses (chart-local for an adaptive
        family); the box is the labels' range, padded by target_pad times
        the spread on both ends of each axis -- generous padding is
        load-bearing, because a box tight around the target puts it in the
        boundary layer of the interpolation stencil, and that failure mode
        is silent -- and an end that reaches its axis's default edge is
        returned as None, i.e. the edge itself, which matters because the
        w_I edges are the planar and shell degeneracies, which the grid
        must reach exactly or not at all."""
        R, vR, vT, z, vz = self._target
        Lzs = R * numpy.fabs(vT)
        wEs, wIs = numpy.empty(len(Lzs)), numpy.empty(len(Lzs))
        Phiinf = evaluatePotentials(self._pot, Rinf, 0.0, use_physical=False)
        for i, (Rp, vRp, vTp, zp, vzp, Lz) in enumerate(zip(R, vR, vT, z, vz, Lzs)):
            Ec = self._circular_orbit(Lz)[1]
            Emax = Phiinf + Lz**2.0 / 2.0 / Rinf**2.0
            kin = (vRp**2.0 + vTp**2.0 + vzp**2.0) / 2.0
            _save = None
            if self._adaptive_chart:
                # place the point's chart with the raw-potential energy: the
                # chart surfaces are smooth, so the O(model error) difference
                # from the wrapper energy is immaterial for the box
                _save = self._swap_local_chart(
                    kin + evaluatePotentials(self._pot, Rp, zp, use_physical=False),
                    Lz,
                )
            try:
                E = kin + evaluatePotentials(
                    self._staeckelwrap, Rp, zp, use_physical=False
                )
                u, v = coords.Rz_to_uv(Rp, zp, delta=self._delta)
                sh = numpy.sinh(u)
                # the point's third integral, from the u-equation of the
                # separated relation (the same convention as _Wu: p_u^2 =
                # 2 delta^2 [E sinh^2 u - U(u) - I3] - L_z^2/sinh^2 u)
                pu = self._delta * (
                    vRp * numpy.cosh(u) * numpy.sin(v) + vzp * sh * numpy.cos(v)
                )
                try:
                    I3 = (
                        E * sh**2.0
                        - self._staeckelwrap._U(u)
                        - (pu**2.0 + Lz**2.0 / sh**2.0) / 2.0 / self._delta**2.0
                    )
                    Ipl = self._I3_planar(E, Lz)
                    den = self._I3_shell(E, Lz) - Ipl
                    wIs[i] = (
                        0.5
                        if den <= 0.0
                        else 2.0
                        / numpy.pi
                        * numpy.arcsin(
                            numpy.sqrt(numpy.clip((I3 - Ipl) / den, 0.0, 1.0))
                        )
                    )
                except ValueError:
                    # too close to the circular degeneracy for the shell
                    # relation to bracket; every I3 label coincides there,
                    # so the direction is free
                    wIs[i] = 0.5
            finally:
                if _save is not None:
                    self._staeckelwrap, self._delta = _save
            if E > Emax:
                raise ValueError(
                    "a target point's energy lies above the energies the "
                    "grid can cover; increase Rinf"
                )
            wEs[i] = numpy.sqrt(numpy.clip((E - Ec) / (Emax - Ec), 0.0, 1.0))

        def _padded(vals, lo0, hi0, snap=True):
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
                # an end that reaches the default edge IS the edge: None,
                # which _edge resolves per end
                return (None if lo <= lo0 else lo, None if hi >= hi0 else hi)
            # the L_z ends are anchors rather than degeneracies, so the box
            # may exceed them; only guard the lower end against L_z <= 0
            return (max(lo, 0.5 * float(numpy.min(vals))), hi)

        box = (
            _padded(
                Lzs,
                Rmin * vcirc(self._pot, Rmin, use_physical=False),
                Rmax * vcirc(self._pot, Rmax, use_physical=False),
                snap=False,
            ),
            _padded(wEs, wpad, 1.0 - wpad),
            _padded(wIs, 0.0, 1.0),
        )
        self._targetbox = {"Lzlim": box[0], "wElim": box[1], "wIlim": box[2]}
        return box

    def _setup_canonical_grid(
        self, Rmin, Rmax, Rinf, nLz, nE, nI3, wpad, Lzlim=None, wElim=None, wIlim=None
    ):
        """The canonical family: the rectified (L_z, w_E, w_I) node lattice
        of the direct grid, all node tori built in one vectorized canonical
        construction, and the family stored as prefiltered 3-D tables whose
        own derivatives drive every evaluation chain (manifest canonicity:
        the labels are the stored action tables, inverted implicitly, and
        no derivative is ever stored separately)"""
        if self._target is not None:
            Lzlim, wElim, wIlim = self._target_box(Rmin, Rmax, Rinf, wpad)
        self._nLz, self._nE, self._nI3 = nLz, nE, nI3
        self._Lzgrid = numpy.linspace(
            *_edge(
                Lzlim,
                (
                    Rmin * vcirc(self._pot, Rmin, use_physical=False),
                    Rmax * vcirc(self._pot, Rmax, use_physical=False),
                ),
            ),
            nLz,
        )
        # The grid is a box in (L_z, w_E, w_I).  Spanning a SUB-interval of
        # each axis is what localizes it on a target -- a stream, say -- and
        # since the interpolation error goes as the spacing, a domain narrower
        # by F is worth as much as F times more nodes.  Rmin/Rmax/Rinf set only
        # the outer extent and cannot express a narrow energy box.
        # A limit of None on either end means that axis's own edge, which
        # matters because the edges are degeneracies rather than arbitrary
        # boundaries: w_E = 0 is the circular orbit, which is why the default
        # pads away from it, and w_I = 0 and 1 are the planar and shell
        # orbits, whose handling keys on the grid reaching them EXACTLY.  A
        # box meant to sit against one of those has to say so rather than
        # approach it with a number.
        self._wEgrid = numpy.linspace(*_edge(wElim, (wpad, 1.0 - wpad)), nE)
        self._wIgrid = numpy.linspace(*_edge(wIlim, (0.0, 1.0)), nI3)
        self._wIedge = 1e-4
        shape = (nLz, nE, nI3)
        self._canon_shape = shape
        self._canon_Rinf = Rinf
        self._canon_wpad = wpad
        self._canon_ushell = numpy.empty((nLz, nE))
        Es, Lzs, I3s = (numpy.empty(shape) for _ in range(3))
        if self._adaptive_chart:
            self._canon_deltas = numpy.empty((nLz, nE))
            self._canon_wraps = [[None] * nE for _ in range(nLz)]
            self._delta_ref, self._wrap_ref = self._delta, self._staeckelwrap
        wIbuild = numpy.clip(self._wIgrid, self._wIedge, 1.0 - self._wIedge)
        sinw = numpy.sin(numpy.pi * wIbuild / 2.0) ** 2.0
        for ii, Lz in enumerate(self._Lzgrid):
            Ec = self._circular_orbit(Lz)[1]
            Emax = (
                evaluatePotentials(self._pot, Rinf, 0.0, use_physical=False)
                + Lz**2.0 / 2.0 / Rinf**2.0
            )
            for jj, wE in enumerate(self._wEgrid):
                E = Ec + wE**2.0 * (Emax - Ec)
                if self._adaptive_chart:
                    # the node's own chart: swap it in for the label
                    # relations, which probe W_u in the node's wrapper
                    dnode = (
                        self._delta_ref
                        if self._delta_func is None
                        else float(self._delta_func(E, Lz))
                    )
                    self._canon_deltas[ii, jj] = dnode
                    u0node = (
                        None if self._u0_func is None else float(self._u0_func(E, Lz))
                    )
                    self._canon_wraps[ii][jj] = (
                        OblateStaeckelWrapperPotential(pot=self._chart_pot, delta=dnode)
                        if u0node is None
                        else OblateStaeckelWrapperPotential(
                            pot=self._chart_pot, delta=dnode, u0=u0node
                        )
                    )
                    self._staeckelwrap = self._canon_wraps[ii][jj]
                    self._delta = dnode
                Ipl = self._I3_planar(E, Lz)
                Ish, ushell = self._I3_shell(E, Lz, return_u=True)
                Es[ii, jj] = E
                Lzs[ii, jj] = Lz
                I3s[ii, jj] = Ipl + sinw * (Ish - Ipl)
                self._canon_ushell[ii, jj] = ushell
        if not self._adaptive_chart:
            grid = actionAngleStaeckelInverse(
                pot=self._staeckelwrap,
                Es=Es.ravel(),
                Lzs=Lzs.ravel(),
                I3s=I3s.ravel(),
                nchi=self._nchi,
                canonical=True,
                ncanon=self._ncanon,
                npt=self._npt,
            )
        else:
            # restore the reference chart; each node build below carries its
            # own wrapper explicitly
            self._delta, self._staeckelwrap = self._delta_ref, self._wrap_ref
            # one inner discrete build per (L_z, E) node, its nI3 tori all in
            # the node's own Staeckel model, sharing one frozen isochrone:
            # the mid node fits it, every other node inherits it, because
            # the compensation's closed-form auxiliary chains assume a
            # single (GM, b) across the family
            import types as _types

            mid = (nLz // 2, nE // 2)
            inners = {}
            iso = None
            order = [mid] + [
                (a, b) for a in range(nLz) for b in range(nE) if (a, b) != mid
            ]
            for a, b in order:
                inners[(a, b)] = actionAngleStaeckelInverse(
                    pot=self._canon_wraps[a][b],
                    Es=Es[a, b],
                    Lzs=Lzs[a, b],
                    I3s=I3s[a, b],
                    nchi=self._nchi,
                    canonical=True,
                    ncanon=self._ncanon,
                    npt=self._npt,
                    isochrone_ab=iso,
                )
                if iso is None:
                    iso = (inners[mid]._GMc, inners[mid]._bc)
            cat = lambda name: numpy.concatenate(
                [getattr(inners[(a, b)], name) for a in range(nLz) for b in range(nE)]
            )
            grid = _types.SimpleNamespace(
                _jr=cat("_jr"),
                _jz=cat("_jz"),
                _umins=cat("_umins"),
                _umaxs=cat("_umaxs"),
                _vmins=cat("_vmins"),
                _can_Dmu=cat("_can_Dmu"),
                _can_Dmv=cat("_can_Dmv"),
                _can_maxdev=max(inners[k]._can_maxdev for k in inners),
                _can_stokes=max(inners[k]._can_stokes for k in inners),
                _GMc=inners[mid]._GMc,
                _bc=inners[mid]._bc,
                _aAIc=inners[mid]._aAIc,
                _aAIinvc=inners[mid]._aAIinvc,
            )
        self._canon_node_maxdev = grid._can_maxdev
        self._canon_node_stokes = grid._can_stokes
        if self._canon_node_maxdev > 1e-6:
            warnings.warn(
                "The stored anomaly maps are under-resolved for the most "
                "extreme grid tori (worst node action deviation "
                f"{self._canon_node_maxdev:.2e}); evaluation stays exactly "
                "canonical, but the accuracy near those tori is limited to "
                "that scale -- raise npt and/or ncanon to resolve them",
                galpyWarning,
            )
        self._GMc, self._bc = grid._GMc, grid._bc
        self._aAIc, self._aAIinvc = grid._aAIc, grid._aAIinvc
        # the stacked tables: labels, energy, supports, and the two anomaly
        # maps' sine coefficients.  The supports are NOT stored as the
        # turning points themselves: each oscillation's half-width vanishes
        # at its degenerate edge, so umax - umin there is the difference of
        # two interpolants that agree to every digit the grid resolves, and
        # the cancellation destroys it (50% wrong one cell from the shell
        # edge).  Stored instead are the midpoint and the SQUARED half-width
        # divided by the action that drives it -- K_u = (umax-umin)^2/4J_R
        # and K_v = (pi/2-vmin)^2/J_z -- both bounded and smooth right up to
        # the edge, where the vanishing is carried entirely by the action
        # itself, which is known exactly at evaluation time.
        nq = 7 + 2 * self._npt
        tab = numpy.empty((nq,) + shape)
        tab[0] = grid._jr.reshape(shape)
        tab[1] = grid._jz.reshape(shape)
        tab[2] = Es
        tab[3] = 0.5 * (grid._umins + grid._umaxs).reshape(shape)
        tab[4] = (0.25 * (grid._umaxs - grid._umins) ** 2.0).reshape(shape) / tab[0]
        tab[5] = ((0.5 * numpy.pi - grid._vmins) ** 2.0).reshape(shape) / tab[1]
        tab[6 : 6 + self._npt] = numpy.moveaxis(
            grid._can_Dmu.reshape(shape + (self._npt,)), -1, 0
        )
        tab[6 + self._npt : 6 + 2 * self._npt] = numpy.moveaxis(
            grid._can_Dmv.reshape(shape + (self._npt,)), -1, 0
        )
        # The focal length as one more stored row, constant along I3 (delta
        # depends on (E, L_z) only) and simply constant everywhere until the
        # adaptive construction fills it.  Storing it as a table row buys its
        # action-derivative chain for free: d delta/dJ comes out of the same
        # differentiated interpolant as every other stored quantity, so the
        # compensation's new term obeys the same no-separate-derivative
        # discipline as the rest of the map.
        tab[6 + 2 * self._npt] = (
            self._delta if self._delta_func is None else self._canon_deltas[:, :, None]
        )
        # the analytic limits at the degenerate edges: the vanishing action
        # is exactly zero, the degenerate oscillation's midpoint sits at its
        # analytic point (the shell u), and its anomaly map vanishes (both
        # loops are harmonic in the thin limit, so eta = tau exactly).  The
        # half-widths need no special case at all: they are reconstructed as
        # sqrt(K J) and so collapse onto the midpoint exactly when the
        # action does, with K carrying its finite limit.
        if self._wIgrid[-1] == 1.0:
            tab[0, :, :, -1] = 0.0
            tab[3, :, :, -1] = self._canon_ushell
            tab[6 : 6 + self._npt, :, :, -1] = 0.0
        if self._wIgrid[0] == 0.0:
            tab[1, :, :, 0] = 0.0
            tab[6 + self._npt : 6 + 2 * self._npt, :, :, 0] = 0.0
        self._canon_tab_raw = tab
        self._canon_dLz = (self._Lzgrid[-1] - self._Lzgrid[0]) / (nLz - 1)
        self._rebuild_canon_interp()
        return None

    def _rebuild_canon_interp(self):
        """(Re)build the prefiltered tables from the raw ones; separate so
        that table perturbations (the noise-injection manifest test)
        re-enter through exactly this call"""
        self._canon_tab = _prefilter_padded(self._canon_tab_raw, (1, 2, 3), 2)
        return None

    def _canon_coords_vec(self, jr, Lz, jz):
        """
        Invert the stored label tables for many tori at once.

        The scalar :meth:`_canon_coords` runs a two-dimensional Newton per
        torus.  Evaluating an ensemble that way costs one Python-level call
        per orbit, and at these array sizes the call overhead dominates the
        arithmetic: the per-torus label inversion measures 3.4 ms against
        0.4 ms of actual per-point work.  This solves all of them together,
        every iterate being an array over tori, which is what makes an
        ensemble affordable.

        Parameters
        ----------
        jr, Lz, jz : numpy.ndarray
            Actions of the tori, all the same shape.

        Returns
        -------
        numpy.ndarray
            Fractional grid coordinates, shape (n, 3).

        Notes
        -----
        - 2026-08-29 - Written - Bovy (UofT)
        """
        jr = numpy.atleast_1d(jr).astype("float")
        Lz = numpy.atleast_1d(Lz).astype("float")
        jz = numpy.atleast_1d(jz).astype("float")
        if numpy.any(Lz < self._Lzgrid[0]) or numpy.any(Lz > self._Lzgrid[-1]):
            raise ValueError(
                f"L_z outside the grid [{self._Lzgrid[0]}, {self._Lzgrid[-1]}]"
            )
        xL = (
            (Lz - self._Lzgrid[0])
            / (self._Lzgrid[-1] - self._Lzgrid[0])
            * (self._nLz - 1)
        )
        xE = numpy.full_like(xL, 0.5 * (self._nE - 1))
        xI = numpy.full_like(xL, 0.5 * (self._nI3 - 1))
        tol = 1e-12 * (1.0 + numpy.fabs(jr) + numpy.fabs(jz))
        for _ in range(self._maxiter):
            x = numpy.stack((xL, xE, xI), axis=1)
            v = self._canon_table_eval(x)
            dE_ = self._canon_table_eval(x, deriv=1)
            dI_ = self._canon_table_eval(x, deriv=2)
            f0 = v[0] - jr
            f1 = v[1] - jz
            det = dE_[0] * dI_[1] - dI_[0] * dE_[1]
            dxE = (dI_[1] * f0 - dI_[0] * f1) / det
            dxI = (-dE_[1] * f0 + dE_[0] * f1) / det
            lim = numpy.minimum(
                1.0,
                1.0
                / numpy.maximum(numpy.maximum(numpy.fabs(dxE), numpy.fabs(dxI)), 1e-30),
            )
            xE = numpy.clip(xE - dxE * lim, 0.0, self._nE - 1.0)
            xI = numpy.clip(xI - dxI * lim, 0.0, self._nI3 - 1.0)
            if numpy.all(numpy.maximum(numpy.fabs(f0), numpy.fabs(f1)) < tol):
                break
        else:
            bad = numpy.maximum(numpy.fabs(f0), numpy.fabs(f1)) >= tol
            raise ValueError(
                "The label inversion did not converge for %d of %d tori; the "
                "first is (J_R, J_z) = (%g, %g)"
                % (numpy.sum(bad), len(jr), jr[bad][0], jz[bad][0])
            )
        return numpy.stack((xL, xE, xI), axis=1)

    def _canon_coords(self, jr, Lz, jz):
        """Invert the stored label tables for the fractional grid
        coordinates: the implicit-inverse labels (exact-in-the-family), by
        a 2-D Newton in (w_E, w_I) at fixed L_z"""
        if Lz < self._Lzgrid[0] or Lz > self._Lzgrid[-1]:
            raise ValueError(
                f"L_z = {Lz} lies outside the grid "
                f"[{self._Lzgrid[0]}, {self._Lzgrid[-1]}]"
            )
        xL = (
            (Lz - self._Lzgrid[0])
            / (self._Lzgrid[-1] - self._Lzgrid[0])
            * (self._nLz - 1)
        )
        xE, xI = 0.5 * (self._nE - 1), 0.5 * (self._nI3 - 1)
        for _ in range(self._maxiter):
            x = numpy.array([[xL, xE, xI]])
            v = self._canon_table_eval(x)
            dE_ = self._canon_table_eval(x, deriv=1)
            dI_ = self._canon_table_eval(x, deriv=2)
            f0 = v[0, 0] - jr
            f1 = v[1, 0] - jz
            J00, J01 = dE_[0, 0], dI_[0, 0]
            J10, J11 = dE_[1, 0], dI_[1, 0]
            det = J00 * J11 - J01 * J10
            dxE = (J11 * f0 - J01 * f1) / det
            dxI = (-J10 * f0 + J00 * f1) / det
            lim = min(1.0, 1.0 / max(abs(dxE), abs(dxI), 1e-30))
            xE = numpy.clip(xE - dxE * lim, 0.0, self._nE - 1.0)
            xI = numpy.clip(xI - dxI * lim, 0.0, self._nI3 - 1.0)
            if max(abs(f0), abs(f1)) < 1e-12 * (1.0 + abs(jr) + abs(jz)):
                break
        else:
            # say which way the torus falls outside: the rectified grid
            # means a near-circular and a too-energetic torus otherwise
            # read the same, with nothing to act on
            wIs = numpy.linspace(0.0, self._nI3 - 1.0, 33)
            lo = min(
                self._canon_table_eval(numpy.array([[xL, 0.0, w]]))[:2, 0].sum()
                for w in wIs
            )
            hi = max(
                self._canon_table_eval(numpy.array([[xL, self._nE - 1.0, w]]))[
                    :2, 0
                ].sum()
                for w in wIs
            )
            if jr + jz < lo:
                raise ValueError(
                    f"(J_R, J_z) = ({jr}, {jz}) lies outside the grid: it "
                    "falls below the covered total action J_R+J_z at "
                    f"L_z = {Lz} (the grid reaches down to "
                    f"J_R+J_z = {lo:g})"
                )
            if jr + jz > hi:
                raise ValueError(
                    f"(J_R, J_z) = ({jr}, {jz}) lies outside the grid: it "
                    "falls above the covered total action J_R+J_z at "
                    f"L_z = {Lz} (the grid reaches up to "
                    f"J_R+J_z = {hi:g}); increase Rinf"
                )
            raise ValueError(
                f"(J_R, J_z) = ({jr}, {jz}) could not be matched inside the "
                f"interpolation grid at L_z = {Lz}: the torus lies outside "
                "the interpolated family"
            )
        return numpy.array([xL, xE, xI])

    def _canon_toy_radial(self, JAr, LA, thetaAr):
        """The radial half of the analytic isochrone inverse: (J^A_r, L^A,
        theta^A_r) -> (eta, r^A, p^A_r), vectorized over points"""
        amp, bb = self._GMc, self._bc
        sq = numpy.sqrt(LA**2 + 4.0 * bb * amp)
        H = -2.0 * amp**2 / (2.0 * JAr + LA + sq) ** 2
        a = -amp / 2.0 / H - bb
        ab = a + bb
        e = numpy.sqrt(numpy.clip(1.0 + LA**2 / (2.0 * H * a**2), 0.0, None))
        ar = numpy.atleast_1d(thetaAr) % (2.0 * numpy.pi)
        aeab = a * e / ab
        x = numpy.array(ar)
        for _ in range(self._maxiter):
            f = x - aeab * numpy.sin(x) - ar
            x -= numpy.clip(f / (1.0 - aeab * numpy.cos(x)), -1.0, 1.0)
            if numpy.max(numpy.fabs(f)) < self._angle_tol:
                break
        else:
            raise RuntimeError(
                "Newton's method for the toy eccentric anomaly did not converge"
            )
        coseta = numpy.cos(x)
        rA = a * numpy.sqrt((1.0 - e * coseta) * (1.0 - e * coseta + 2.0 * bb / a))
        pA = numpy.sqrt(amp / ab) * a * e * numpy.sin(x) / rA
        return x, rA, pA

    def _canon_toy_vert(self, JAr, LA, Lz, thetaAr, thetaAz, eta):
        """The toy's vertical geometry at given toy angles: the polar angle
        and vertical momentum from the isochrone's own angle relations
        (psi = theta^A_z - (omega_z/omega_r) theta^A_r + Lambda(eta)),
        all closed forms"""
        amp, bb = self._GMc, self._bc
        sq = numpy.sqrt(LA**2 + 4.0 * bb * amp)
        H = -2.0 * amp**2 / (2.0 * JAr + LA + sq) ** 2
        a = -amp / 2.0 / H - bb
        e = numpy.sqrt(numpy.clip(1.0 + LA**2 / (2.0 * H * a**2), 0.0, None))
        # theta^A_r is used unwrapped below, so eta has to be in its branch:
        # the radial inverse returns eta on [0, 2 pi), which puts the two a
        # full period apart whenever theta^A_r is just below zero.
        eta = eta + 2.0 * numpy.pi * numpy.round(
            (numpy.atleast_1d(thetaAr) - eta) / (2.0 * numpy.pi)
        )
        taneta2 = numpy.tan(eta / 2.0)
        tan11 = numpy.arctan(numpy.sqrt((1.0 + e) / (1.0 - e)) * taneta2)
        tan12 = numpy.arctan(
            numpy.sqrt((a * (1.0 + e) + 2.0 * bb) / (a * (1.0 - e) + 2.0 * bb))
            * taneta2
        )
        # Lambda climbs by pi (1 + L^A/sq) per radial period, cancelling the
        # -1/2 (1 + L^A/sq) theta^A_r in psi, so psi is periodic.  Selecting
        # the branch by the SIGN OF THE ARCTAN gets that right inside
        # (0, 2 pi) and wrong at the ends: for eta slightly below zero the
        # arctan is negative and picks up a spurious pi, which is the same
        # value it takes at eta just below 2 pi.  Keying the branch to eta
        # instead is identical on (0, 2 pi) and continuous through zero.
        nwind = numpy.round(eta / (2.0 * numpy.pi))
        tan11 = tan11 + numpy.pi * nwind
        tan12 = tan12 + numpy.pi * nwind
        Lambdaeta = tan11 + LA / sq * tan12
        psi = thetaAz - 0.5 * (1.0 + LA / sq) * thetaAr + Lambdaeta
        sini = numpy.sqrt(numpy.clip(1.0 - Lz**2 / LA**2, 0.0, None))
        costh = numpy.sin(psi) * sini  # polar angle: cos(vartheta)
        sinth = numpy.sqrt(numpy.clip(1.0 - costh**2, 0.0, None))
        # p^A_theta = -L sin(i) cos(psi)/sin(vartheta), with magnitude
        # sqrt(L^2 - Lz^2/sin^2 vartheta)
        pAth = -LA * sini * numpy.cos(psi) / numpy.maximum(sinth, 1e-15)
        thetaA = numpy.arccos(numpy.clip(costh, -1.0, 1.0))
        return thetaA, pAth

    def _swap_local_chart(self, E, Lz):
        """Swap in an adaptive family's LOCAL chart at (E, L_z) -- cached,
        because consecutive calls cluster on few charts -- and return the
        previous (wrapper, delta) for the caller's finally block"""
        _save = (self._staeckelwrap, self._delta)
        dloc = (
            float(self._delta)
            if self._delta_func is None
            else float(self._delta_func(E, Lz))
        )
        u0loc = None if self._u0_func is None else float(self._u0_func(E, Lz))
        key = (round(dloc, 10), None if u0loc is None else round(u0loc, 10))
        if getattr(self, "_chart_cache_key", None) != key:
            self._chart_cache = (
                OblateStaeckelWrapperPotential(pot=self._chart_pot, delta=dloc)
                if u0loc is None
                else OblateStaeckelWrapperPotential(
                    pot=self._chart_pot, delta=dloc, u0=u0loc
                )
            )
            self._chart_cache_key = key
        self._staeckelwrap, self._delta = self._chart_cache, dloc
        return _save

    def _canon_coords_integrals(self, E, Lz, I3):
        """Fractional grid coordinates directly from the integrals
        (E, L_z, I3) -- the rectified coordinates are closed forms, so
        labelling a torus by its integrals needs no inversion at all"""
        if Lz < self._Lzgrid[0] or Lz > self._Lzgrid[-1]:
            raise ValueError(
                f"L_z = {Lz} lies outside the grid "
                f"[{self._Lzgrid[0]}, {self._Lzgrid[-1]}]"
            )
        xL = (
            (Lz - self._Lzgrid[0])
            / (self._Lzgrid[-1] - self._Lzgrid[0])
            * (self._nLz - 1)
        )
        Ec = self._circular_orbit(Lz)[1]
        Emax = (
            evaluatePotentials(self._pot, self._canon_Rinf, 0.0, use_physical=False)
            + Lz**2 / 2.0 / self._canon_Rinf**2
        )
        if E < Ec:
            raise ValueError(
                f"E = {E} lies outside the grid: below the circular orbit's "
                f"energy {Ec:g} at L_z = {Lz}"
            )
        wE = numpy.sqrt((E - Ec) / (Emax - Ec))
        if wE > 1.0:
            raise ValueError(
                f"E = {E} lies outside the grid: above the energies covered "
                f"at L_z = {Lz}; increase Rinf"
            )
        if self._adaptive_chart:
            # I3 is chart-defined, so the label relations must run in the
            # LOCAL chart at this (E, L_z) -- the same smooth surfaces the
            # nodes were built from, so labels and tables are consistent
            _save = self._swap_local_chart(E, Lz)
        try:
            Ipl = self._I3_planar(E, Lz)
            Ish = self._I3_shell(E, Lz)
        finally:
            if self._adaptive_chart:
                self._staeckelwrap, self._delta = _save
        sfrac = numpy.clip((I3 - Ipl) / (Ish - Ipl), 0.0, 1.0)
        wI = 2.0 / numpy.pi * numpy.arcsin(numpy.sqrt(sfrac))
        # through the grid's ACTUAL span, which is the default one unless the
        # grid was narrowed
        wE0, wE1 = self._wEgrid[0], self._wEgrid[-1]
        wI0, wI1 = self._wIgrid[0], self._wIgrid[-1]
        xE = numpy.clip((wE - wE0) / (wE1 - wE0), 0.0, 1.0) * (self._nE - 1)
        xI = (wI - wI0) / (wI1 - wI0) * (self._nI3 - 1)
        return numpy.array([xL, float(xE), float(xI)])

    def _canon_family_chains(self, x):
        """All family values and their action chains at fractional grid
        coordinates x: the stored tables' own derivatives, contracted with
        the inverse of the label-coordinate matrix (J_R, J_phi, J_z) vs
        (L_z, w_E-index, w_I-index)"""
        xx = numpy.atleast_2d(x)
        v = self._canon_table_eval(xx)[:, 0]
        dL = self._canon_table_eval(xx, deriv=0)[:, 0] / self._canon_dLz
        dE_ = self._canon_table_eval(xx, deriv=1)[:, 0]
        dI_ = self._canon_table_eval(xx, deriv=2)[:, 0]
        M = numpy.array(
            [
                [dL[0], dE_[0], dI_[0]],
                [1.0, 0.0, 0.0],
                [dL[1], dE_[1], dI_[1]],
            ]
        )
        Minv = numpy.linalg.inv(M)
        # chains of every stored quantity along (J_R, J_phi, J_z)
        dq = numpy.stack((dL, dE_, dI_), axis=1) @ Minv  # (nq, 3)
        # turn the stored midpoint-and-K combinations back into the turning
        # points, differentiating the reconstruction itself so the chains
        # remain the exact derivatives of what is evaluated.  dq[0] and
        # dq[1] are the identity rows (1, 0, 0) and (0, 0, 1) by the
        # construction of M, so J_R's and J_z's own chains enter here
        # exactly rather than through the interpolation
        uc, duc = v[3], dq[3]
        hw = numpy.sqrt(numpy.clip(v[4] * v[0], 0.0, None))
        hv = numpy.sqrt(numpy.clip(v[5] * v[1], 0.0, None))
        # below the floor the oscillation is degenerate and the compensation
        # drops the term outright, so the divergence is never evaluated
        dhw = (v[4] * dq[0] + v[0] * dq[4]) / (2.0 * numpy.maximum(hw, 1e-14))
        dhv = (v[5] * dq[1] + v[1] * dq[5]) / (2.0 * numpy.maximum(hv, 1e-14))
        v[3], v[4], v[5] = uc - hw, uc + hw, 0.5 * numpy.pi - hv
        dq[3], dq[4], dq[5] = duc - dhw, duc + dhw, -dhv
        return v, dq

    def _canon_family_chains_vec(self, x):
        """
        Family values and action chains for many tori at once.

        The scalar :meth:`_canon_family_chains` keeps only the first point of
        the stencil evaluation; this keeps all of them, so the 3x3 label
        matrix is inverted as a stack and the chains are contracted with
        ``einsum``.  See :meth:`_canon_coords_vec` for why an ensemble needs
        this.

        Parameters
        ----------
        x : numpy.ndarray
            Fractional grid coordinates, shape (n, 3).

        Returns
        -------
        tuple
            ``(v, dq)`` with ``v`` of shape (nq, n) and ``dq`` of shape
            (nq, n, 3), the turning points reconstructed exactly as in the
            scalar routine.

        Notes
        -----
        - 2026-08-29 - Written - Bovy (UofT)
        """
        xx = numpy.atleast_2d(x)
        npts = xx.shape[0]
        v = self._canon_table_eval(xx)
        dL = self._canon_table_eval(xx, deriv=0) / self._canon_dLz
        dE_ = self._canon_table_eval(xx, deriv=1)
        dI_ = self._canon_table_eval(xx, deriv=2)
        M = numpy.empty((npts, 3, 3))
        M[:, 0, 0], M[:, 0, 1], M[:, 0, 2] = dL[0], dE_[0], dI_[0]
        M[:, 1, 0], M[:, 1, 1], M[:, 1, 2] = 1.0, 0.0, 0.0
        M[:, 2, 0], M[:, 2, 1], M[:, 2, 2] = dL[1], dE_[1], dI_[1]
        Minv = numpy.linalg.inv(M)
        dq = numpy.einsum("qpk,pkj->qpj", numpy.stack((dL, dE_, dI_), axis=-1), Minv)
        # same reconstruction of the turning points from the stored
        # midpoint-and-K combinations as the scalar routine, differentiated
        # the same way so the chains stay exact derivatives of what is used
        uc, duc = v[3].copy(), dq[3].copy()
        hw = numpy.sqrt(numpy.clip(v[4] * v[0], 0.0, None))
        hv = numpy.sqrt(numpy.clip(v[5] * v[1], 0.0, None))
        dhw = (v[4][:, None] * dq[0] + v[0][:, None] * dq[4]) / (
            2.0 * numpy.maximum(hw, 1e-14)[:, None]
        )
        dhv = (v[5][:, None] * dq[1] + v[1][:, None] * dq[5]) / (
            2.0 * numpy.maximum(hv, 1e-14)[:, None]
        )
        v[3], v[4], v[5] = uc - hw, uc + hw, 0.5 * numpy.pi - hv
        dq[3], dq[4], dq[5] = duc - dhw, duc + dhw, -dhv
        return v, dq

    def _canon_comp(self, thetaAr, thetaAz, jr, LA, Lz, v, dq):
        """The two-map compensation terms along the three action chains,
        every factor grouped through the per-degree action-flux identities
        so it is closed-form and regular at all turning points; returns
        (comp_R, comp_phi, comp_z) arrays over the points"""
        npt = self._npt
        umin, umax, vmin = v[3], v[4], v[5]
        dumin, dumax, dvmin = dq[3], dq[4], dq[5]
        Dmu, Dmv = v[6 : 6 + npt], v[6 + npt : 6 + 2 * npt]
        dDmu, dDmv = dq[6 : 6 + npt], dq[6 + npt : 6 + 2 * npt]
        delc, ddel = v[6 + 2 * npt], dq[6 + 2 * npt]
        # closed-form toy-parameter chains: a, e from (J^A_r = J_R, L^A)
        GM, bb = self._GMc, self._bc
        sq = numpy.sqrt(LA**2 + 4.0 * bb * GM)
        CA = 0.5 * (LA + sq)
        EA = -(GM**2) / (2.0 * (jr + CA) ** 2)
        a = -GM / (2.0 * EA) - bb
        e = numpy.sqrt(1.0 + LA**2 / (2.0 * EA * a**2))
        thmin = numpy.arcsin(numpy.clip(numpy.fabs(Lz) / LA, 0.0, 1.0))
        # dLA/d(J_R, J_phi, J_z) and the induced (a, e, thmin) chains
        dLA = numpy.array([0.0, numpy.sign(Lz), 1.0])
        dEA = (
            GM**2
            / (jr + CA) ** 3
            * (numpy.array([1.0, 0.0, 0.0]) + 0.5 * (1.0 + LA / sq) * dLA)
        )
        da = GM / (2.0 * EA**2) * dEA
        de = (
            2.0 * LA * dLA / (2.0 * EA * a**2)
            - LA**2 * (dEA * a + 2.0 * EA * da) / (2.0 * EA**2 * a**3)
        ) / (2.0 * e)
        costhmin = numpy.cos(thmin)
        dthmin = numpy.array(
            [
                0.0,
                numpy.sign(Lz) * (numpy.sign(Lz) / LA - numpy.fabs(Lz) / LA**2),
                -numpy.fabs(Lz) / LA**2,
            ]
        ) / numpy.maximum(costhmin, 1e-12)
        dthmin[1] = (
            numpy.sign(Lz)
            * (1.0 / LA - numpy.fabs(Lz) / LA**2 * 1.0)
            / numpy.maximum(costhmin, 1e-12)
        )
        # u-degree at the current phases
        eta_u, rA, pAr = self._canon_toy_radial(jr, LA, thetaAr)
        tuu = self._tau_of_eta(eta_u, Dmu, where="eta_u")
        ms = self._nforDm
        s = numpy.sqrt(bb**2 + rA**2)
        y = (s - bb) / a
        coseta = numpy.cos(eta_u)
        sineta = numpy.sin(eta_u)
        smat_u = numpy.sin(tuu[:, None] * ms[None, :])
        drAdeta = a * e * sineta * (y + bb / a) / numpy.sqrt(y * (y + 2.0 * bb / a))
        # dr^A/dJ_i at fixed tau_u, and du/dJ_i at fixed tau_u
        gA = a * e * (y + bb / a) / numpy.sqrt(y * (y + 2.0 * bb / a))
        detau = 1.0 + numpy.cos(tuu[:, None] * ms[None, :]) @ (ms * Dmu)
        sintu = numpy.sin(tuu)
        sru = numpy.where(
            numpy.fabs(sintu) > 1e-12,
            sineta / numpy.maximum(numpy.fabs(sintu), 1e-12) * numpy.sign(sintu),
            1.0,
        )
        pu = pAr * gA * sru * detau / (0.5 * (umax - umin))
        comp = numpy.empty((3, len(thetaAr)))
        c2u = numpy.cos(tuu / 2.0) ** 2
        s2u = numpy.sin(tuu / 2.0) ** 2
        # Test degeneracy on the REQUESTED actions, not on the reconstructed
        # supports.  An oscillation is degenerate exactly when its action
        # vanishes, and the action is exact input, whereas umax - umin comes
        # back as sqrt(K J) through the label inversion and so carries that
        # inversion's residual: a torus asked for at J = 0 can reappear here
        # with a half-width of ~1e-6, clearing a threshold on the support
        # while the compensation's true sqrt(K/J)/2 divergence fires on what
        # is really a degenerate torus.  Which side of such a threshold the
        # noise lands on is platform-dependent, which is what made
        # test_actionAngleStaeckelInverse_interp_degenerate_edges fail on
        # Python 3.14 alone.  Keying on the action removes the ambiguity.
        udeg = jr <= 0.0 or (umax - umin) < 1e-10
        vdeg = (
            (LA - numpy.fabs(Lz)) <= 0.0
            or (0.5 * numpy.pi - thmin) < 1e-8
            or (numpy.pi - 2.0 * vmin) < 1e-10
        )
        # v-degree phases from the toy's own vertical geometry
        thetaAv, pAthv = self._canon_toy_vert(jr, LA, Lz, thetaAr, thetaAz, eta_u)
        cosetav = numpy.clip(
            (0.5 * numpy.pi - thetaAv) / numpy.maximum(0.5 * numpy.pi - thmin, 1e-12),
            -1.0,
            1.0,
        )
        sinetav = numpy.sign(pAthv) * numpy.sqrt(
            numpy.clip(1.0 - cosetav**2, 0.0, None)
        )
        eta_v = numpy.arctan2(sinetav, cosetav) % (2.0 * numpy.pi)
        tvv = self._tau_of_eta(eta_v, Dmv, where="eta_v")
        smat_v = numpy.sin(tvv[:, None] * ms[None, :])
        detav = 1.0 + numpy.cos(tvv[:, None] * ms[None, :]) @ (ms * Dmv)
        sintv = numpy.sin(tvv)
        srv = numpy.where(
            numpy.fabs(sintv) > 1e-12,
            sinetav / numpy.maximum(numpy.fabs(sintv), 1e-12) * numpy.sign(sintv),
            1.0,
        )
        gth = 0.5 * numpy.pi - thmin
        pv = pAthv * gth * srv * detav / (0.5 * (numpy.pi - 2.0 * vmin))
        cv = numpy.cos(tvv)
        # The focal length's own compensation (ADAPTIVE_STAECKEL_MATH.md
        # section 2): (R, z) are homogeneous of degree one in delta at fixed
        # (u, v), so the un-lift's generating function contributes
        # (p_R R + p_z z)/delta = [sinh u cosh u p_u - sin v cos v p_v] /
        # [(sinh^2 u + sin^2 v) delta] per unit d delta/dJ_i, entering with
        # the TARGET-chart sign: like the -p_u du_i and -p_v dv_i terms
        # below, a parameter that moves the target position at fixed
        # anomaly contributes with a minus, opposite to the auxiliary's
        # terms.  (Wired the other way, the symplectic defect under a 5%
        # delta variation is 0.88 -- twice the uncompensated 0.47, the
        # signature of a sign error -- and with this sign it is 1.4e-9,
        # the clean-family floor.)  Manifestly
        # regular -- delta > 0 and the numerator vanishes with the momenta
        # at the turning points -- and identically zero for a constant-delta
        # family, which is the fixed-focal special case.
        ucrd = umin * c2u + umax * s2u
        vcrd = vmin + (numpy.pi - 2.0 * vmin) * numpy.sin(tvv / 2.0) ** 2
        shc, chc = numpy.sinh(ucrd), numpy.cosh(ucrd)
        snc, csc = numpy.sin(vcrd), numpy.cos(vcrd)
        pdq = numpy.zeros(len(thetaAr))
        if not udeg:
            pdq += shc * chc * pu
        if not vdeg:
            pdq -= snc * csc * pv
        pdq /= (shc**2 + snc**2) * delc
        for i in range(3):
            if udeg:
                uterm = 0.0
            else:
                drA_i = (
                    y * s / rA * da[i]
                    - a * s * coseta / rA * de[i]
                    + drAdeta * (smat_u @ dDmu[:, i])
                )
                du_i = dumin[i] * c2u + dumax[i] * s2u
                uterm = pAr * drA_i - pu * du_i
            if vdeg:
                vterm = 0.0
            else:
                dth_i = dthmin[i] * cosetav + gth * sinetav * (smat_v @ dDmv[:, i])
                dv_i = dvmin[i] * cv
                vterm = pAthv * dth_i - pv * dv_i
            comp[i] = uterm + vterm - pdq * ddel[i]
        return comp[0], comp[1], comp[2]

    def _toy_angle_solve(self, thR, thz, jr, LA, Lz, v, dq):
        """
        Solve theta^A + c(theta^A) = theta for the auxiliary angles.

        A damped Picard iteration, not a Newton: the update has multiplier
        -c', so it contracts wherever |c'| < 1 and needs no derivative of the
        compensation.  Its one failure mode is a torus where c' reaches 1,
        which makes it cycle with period two rather than diverge -- the
        residual flips sign at constant amplitude, and the step limiter never
        engages because it only caps steps above 0.5.  Under-relaxation turns
        the multiplier into 1 - 2 omega, so halving omega on any
        non-contracting iteration breaks the cycle while leaving a
        contracting one untouched.

        Parameters
        ----------
        thR, thz : numpy.ndarray
            Requested radial and vertical angles.
        jr, LA, Lz : float
            Actions of the torus.
        v, dq : numpy.ndarray
            Family values and their action derivatives.

        Returns
        -------
        tuple
            (theta^A_r, theta^A_z, c_phi).

        Notes
        -----
        - 2026-08-30 - Written - Bovy (UofT)
        """
        thetaAr = numpy.copy(thR)
        thetaAz = numpy.copy(thz)
        omega = 1.0
        prev = numpy.inf
        for _ in range(self._maxiter):
            cR, cphi, cz = self._canon_comp(thetaAr, thetaAz, jr, LA, Lz, v, dq)
            f0 = (thetaAr + cR - thR + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
            f1 = (thetaAz + cz - thz + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
            step = numpy.maximum(numpy.fabs(f0), numpy.fabs(f1))
            mx = numpy.max(step)
            if mx >= prev:
                # Not contracting.  This iteration is a damped Picard
                # iteration, not the Newton the message once claimed, so its
                # multiplier is -c' and a torus where c' reaches 1 makes it
                # cycle with period two rather than diverge: the residual
                # then flips sign at CONSTANT amplitude and the existing
                # limiter never engages, since it only caps steps above 0.5.
                # Observed within about 5e-4 of the radial turning point.
                # Under-relaxing turns the multiplier into 1 - 2 omega, so
                # halving omega breaks the cycle and any omega < 1 converges.
                omega *= 0.5
            prev = mx
            lim = omega * numpy.minimum(1.0, 0.5 / numpy.maximum(step, 1e-30))
            thetaAr -= f0 * lim
            thetaAz -= f1 * lim
            if mx < self._angle_tol:
                break
        else:
            raise RuntimeError("Newton's method for the toy angles did not converge")
        return thetaAr, thetaAz, cphi

    def _xvFreqs_canonical_interp(self, jr, jphi, jz, angler, anglephi, anglez, x=None):
        """The canonical family evaluation: implicit-inverse labels, the
        compensated 2-D angle Newton (exact residuals, identity-dominated
        Jacobian), delegation to the analytic isochrone inverse, and the
        per-degree un-lift with the family-interpolated maps; frequencies
        are the stored energy table's own derivatives through the label
        chains (the integrator contract)"""
        jr, jphi, jz = float(jr), float(jphi), float(jz)
        Lz = jphi
        LA = jz + numpy.fabs(Lz)
        thR = numpy.atleast_1d(numpy.array(angler, dtype="float"))
        thphi = numpy.atleast_1d(numpy.array(anglephi, dtype="float"))
        thz = numpy.atleast_1d(numpy.array(anglez, dtype="float"))
        thR, thphi, thz = numpy.broadcast_arrays(thR, thphi, thz)
        if x is None:
            x = self._canon_coords(jr, Lz, jz)
        v, dq = self._canon_family_chains(x)
        thetaAr, thetaAz, cphi = self._toy_angle_solve(thR, thz, jr, LA, Lz, v, dq)
        thetaAphi = thphi - cphi
        # one vectorized delegation for all points: the analytic isochrone
        # inverse broadcasts the (constant) actions against the angle arrays,
        # and its own root find solves the whole batch at once
        oo = self._aAIinvc._xvFreqs(jr, Lz, jz, thetaAr, thetaAphi, thetaAz)
        out = numpy.array([numpy.atleast_1d(q) for q in oo[:6]], dtype="float")
        npt = self._npt
        GM, bb = self._GMc, self._bc
        sq = numpy.sqrt(LA**2 + 4.0 * bb * GM)
        EA = -(GM**2) / (2.0 * (jr + 0.5 * (LA + sq)) ** 2)
        a = -GM / (2.0 * EA) - bb
        e = numpy.sqrt(1.0 + LA**2 / (2.0 * EA * a**2))
        thmin = numpy.arcsin(numpy.clip(numpy.fabs(Lz) / LA, 0.0, 1.0))
        Rt, vRt, vTt, zt, vzt, phit = self._canon_unlift(
            out,
            a,
            e,
            LA,
            thmin,
            v[6 : 6 + npt],
            v[6 + npt : 6 + 2 * npt],
            v[3],
            v[4],
            v[5],
            Lz,
            delta=v[6 + 2 * npt],
        )
        # frequencies: the stored energy table's own derivative chains
        OmR, Omphi, Omz = dq[2]
        return (Rt, vRt, vTt, zt, vzt, phit, OmR, Omphi, Omz)

    ################## ANALYTIC d/dJ OF THE STORED QUANTITIES #################
    # STAECKEL_CANONICAL_MATH.md section 10.7. The family reconstructs angles
    # by differentiating stored map data, so those derivatives must be known
    # analytically rather than inferred from the interpolant: supports by the
    # implicit function theorem on W=0, anomaly maps by differentiating the
    # action-matching condition at fixed tau.
    def _canon_dsup_dJ(self, ii):
        """d(umin, umax, vmin)/d(J_R, J_phi, J_z), analytic"""
        E, Lz = self._Es[ii], self._Lzs[ii]
        d2 = self._delta**2.0
        out = numpy.empty((3, 3))
        for row, (q, isu) in enumerate(
            ((self._umins[ii], True), (self._umaxs[ii], True), (self._vmins[ii], False))
        ):
            if isu:
                dWdq = self._dWu(numpy.array([q]), E, Lz)[0]
                dWda = numpy.array(
                    [
                        2.0 * d2 * numpy.sinh(q) ** 2.0,
                        -2.0 * d2,
                        -2.0 * Lz / numpy.sinh(q) ** 2.0,
                    ]
                )
            else:
                dWdq = self._dWv(numpy.array([q]), E, Lz)[0]
                dWda = numpy.array(
                    [
                        2.0 * d2 * numpy.sin(q) ** 2.0,
                        +2.0 * d2,
                        -2.0 * Lz / numpy.sin(q) ** 2.0,
                    ]
                )
            out[row] = -dWda / dWdq
        # d(E,I3,Lz)/d(J_R,J_z,J_phi) -> columns reordered to (J_R,J_phi,J_z)
        return out @ self._dEI3Lz_dJ[ii][:, [0, 2, 1]], self._dEI3Lz_dJ[ii][
            :, [0, 2, 1]
        ]

    def _toy_gu_partials(self, eta, a, e):
        """The toy radial action integrand g = p^A_r dr^A/deta and its
        (a, e) partials, in closed form"""
        b = self._bc
        c, s = numpy.cos(eta), numpy.sin(eta)
        y = 1.0 - e * c
        w = y + 2.0 * b / a
        P = y + b / a
        Q = y * w
        K = numpy.sqrt(self._GMc / (a + b))
        g = K * a * e**2 * s**2 * P / Q
        # d/de at fixed eta: y_e = w_e = P_e = -c, Q_e = -c (w + y)
        Qe = -c * (w + y)
        dg_de = K * a * s**2 * (2.0 * e * P / Q + e**2 * (-c * Q - P * Qe) / Q**2)
        # d/da at fixed eta
        Ka = -0.5 * numpy.sqrt(self._GMc) * (a + b) ** -1.5
        Pa = -b / a**2
        Qa = y * (-2.0 * b / a**2)
        dg_da = (
            Ka * a * e**2 * s**2 * P / Q
            + K * e**2 * s**2 * P / Q
            + K * a * e**2 * s**2 * (Pa * Q - P * Qa) / Q**2
        )
        return g, dg_da, dg_de

    def _toy_gv_partials(self, eta, LA, Lz):
        """The toy vertical action integrand and its (L^A, L_z) partials"""
        m = numpy.clip(numpy.fabs(Lz) / LA, 0.0, 1.0 - 1e-15)
        thmin = numpy.arcsin(m)
        gth = 0.5 * numpy.pi - thmin
        c, s = numpy.cos(eta), numpy.sin(eta)
        th = 0.5 * numpy.pi - gth * c
        sth = numpy.sin(th)
        pth2 = numpy.clip(LA**2 - Lz**2 / sth**2, 1e-300, None)
        g = numpy.sqrt(pth2) * gth * numpy.fabs(s)
        rt = 1.0 / numpy.sqrt(numpy.clip(1.0 - m**2, 1e-300, None))
        dthmin_dLA = rt * (-numpy.fabs(Lz) / LA**2)
        dthmin_dLz = rt * (numpy.sign(Lz) / LA)
        out = []
        for dthmin, dpth2_expl in (
            (dthmin_dLA, 2.0 * LA),
            (dthmin_dLz, -2.0 * Lz / sth**2),
        ):
            dgth = -dthmin
            dth = -c * dgth
            dpth2 = dpth2_expl + 2.0 * Lz**2 * numpy.cos(th) / sth**3 * dth
            out.append(
                dpth2 / (2.0 * numpy.sqrt(pth2)) * gth * numpy.fabs(s)
                + numpy.sqrt(pth2) * dgth * numpy.fabs(s)
            )
        return g, out[0], out[1]

    def _canon_dDm_dJ(self, ii, dsupJ, Malpha):
        """d(D^u_m, D^v_m)/dJ by differentiating the action-matching condition
        A^A(eta) = A_t(tau) at fixed tau. The denominator p^A dq^A/deta
        vanishes at both turning points, so rather than dividing we expand
        deta/dJ|_tau = sum_m x_m sin(m tau) -- exact by construction -- and
        solve the resulting Galerkin system, which never divides."""
        N = self._ncanon
        tau = 2.0 * numpy.pi * (numpy.arange(N) + 0.5) / N
        kk = numpy.fft.fftfreq(N, d=1.0 / N)
        E, Lz, I3 = self._Es[ii], self._Lzs[ii], self._I3s[ii]
        d2 = self._delta**2.0
        ms = self._nforDm

        def antider(f):
            fh = numpy.fft.fft(f - numpy.mean(f))
            ah = numpy.zeros_like(fh)
            ah[1:] = fh[1:] / (1j * kk[1:])
            return numpy.real(numpy.fft.ifft(ah))

        def solve(gA, num):
            MM = N // 2 - 1
            mall = numpy.arange(1, MM + 1)
            S = numpy.sin(tau[:, None] * mall[None, :])
            return numpy.linalg.solve((S * gA[:, None]).T @ S / N, S.T @ num / N)[
                : self._npt
            ]

        # ---- u-degree
        umin, umax = self._umins[ii], self._umaxs[ii]
        u = umin + (umax - umin) * numpy.sin(tau / 2.0) ** 2
        dudtau = 0.5 * (umax - umin) * numpy.sin(tau)
        # p_u is SIGNED (negative on the return branch), so that the loop mean
        # of p_u du/dtau is the action; with |p_u| it would vanish identically
        sgn = numpy.where(tau < numpy.pi, 1.0, -1.0)
        pu = sgn * numpy.sqrt(numpy.clip(self._Wu(u, E, Lz, I3), 1e-300, None))
        dWdalpha = numpy.stack(
            (
                2.0 * d2 * numpy.sinh(u) ** 2.0,
                -2.0 * d2 * numpy.ones_like(u),
                -2.0 * Lz / numpy.sinh(u) ** 2.0,
            )
        )
        dW_dJ = numpy.einsum("an,ak->nk", dWdalpha, Malpha)
        du_dJ = (numpy.cos(tau / 2.0) ** 2)[:, None] * dsupJ[0] + (
            numpy.sin(tau / 2.0) ** 2
        )[:, None] * dsupJ[1]
        dgt = (self._dWu(u, E, Lz)[:, None] * du_dJ + dW_dJ) / (
            2.0 * pu[:, None]
        ) * dudtau[:, None] + pu[:, None] * (0.5 * numpy.sin(tau))[:, None] * (
            dsupJ[1] - dsupJ[0]
        )
        jr, jz = self._jr[ii], self._jz[ii]
        LA = jz + numpy.fabs(Lz)
        a, e = self._toy_ae(jr, LA)
        dLA = numpy.array([0.0, numpy.sign(Lz), 1.0])
        sq = numpy.sqrt(LA**2 + 4.0 * self._bc * self._GMc)
        EA = self._iso_E_of_Jr(jr, LA)
        dEA = (
            self._GMc**2
            / (jr + 0.5 * (LA + sq)) ** 3
            * (numpy.array([1.0, 0.0, 0.0]) + 0.5 * (1.0 + LA / sq) * dLA)
        )
        da = self._GMc / (2.0 * EA**2) * dEA
        de = (
            2.0 * LA * dLA / (2.0 * EA * a**2)
            - LA**2 * (dEA * a + 2.0 * EA * da) / (2.0 * EA**2 * a**3)
        ) / (2.0 * e)
        smat = numpy.sin(tau[:, None] * ms[None, :])
        cmat = numpy.cos(tau[:, None] * ms[None, :])
        etau = tau + smat @ self._can_Dmu[ii]
        detau = 1.0 + cmat @ (ms * self._can_Dmu[ii])
        gu, dgu_da, dgu_de = self._toy_gu_partials(etau, a, e)
        dgA = dgu_da[:, None] * da + dgu_de[:, None] * de
        num_u = numpy.stack(
            [antider(dgt[:, i]) - antider(dgA[:, i] * detau) for i in range(3)], axis=1
        )
        dDmu = solve(gu, num_u)
        # ---- v-degree
        vmin = self._vmins[ii]
        Dv = numpy.pi - 2.0 * vmin
        v = vmin + Dv * numpy.sin(tau / 2.0) ** 2
        dvdtau = 0.5 * Dv * numpy.sin(tau)
        pv = sgn * numpy.sqrt(numpy.clip(self._Wv(v, E, Lz, I3), 1e-300, None))
        dWdalpha_v = numpy.stack(
            (
                2.0 * d2 * numpy.sin(v) ** 2.0,
                +2.0 * d2 * numpy.ones_like(v),
                -2.0 * Lz / numpy.sin(v) ** 2.0,
            )
        )
        dWv_dJ = numpy.einsum("an,ak->nk", dWdalpha_v, Malpha)
        dv_dJ = numpy.cos(tau)[:, None] * dsupJ[2]
        dgtv = (self._dWv(v, E, Lz)[:, None] * dv_dJ + dWv_dJ) / (
            2.0 * pv[:, None]
        ) * dvdtau[:, None] + pv[:, None] * (-numpy.sin(tau))[:, None] * dsupJ[2]
        etav = tau + smat @ self._can_Dmv[ii]
        detav = 1.0 + cmat @ (ms * self._can_Dmv[ii])
        gv, dgv_dLA, dgv_dLz = self._toy_gv_partials(etav, LA, Lz)
        dLzv = numpy.array([0.0, 1.0, 0.0])
        dgAv = dgv_dLA[:, None] * dLA + dgv_dLz[:, None] * dLzv
        num_v = numpy.stack(
            [antider(dgtv[:, i]) - antider(dgAv[:, i] * detav) for i in range(3)],
            axis=1,
        )
        dDmv = solve(gv, num_v)
        return dDmu, dDmv

    def _canon_node_dJ(self, ii):
        """Analytic d/dJ of every per-torus quantity the family stores"""
        dsupJ, Malpha = self._canon_dsup_dJ(ii)
        dDmu, dDmv = self._canon_dDm_dJ(ii, dsupJ, Malpha)
        return dsupJ, dDmu, dDmv


def _fit_staeckel_surface(pot, Rmin, Rmax, Rinf, nsub=6, rms_warn=0.05):
    """Survey the max|Phi_S - Phi|-minimizing focal length over the grid's
    (L_z, E) domain and fit a smooth quadratic surface delta(E, L_z), with
    the matching reference curve u0(E, L_z) at the zero-velocity R-midpoint.

    The smoothness is load-bearing rather than cosmetic: the stored tables
    are chart-valued, so they interpolate well across tori only if the chart
    varies smoothly -- and near circular orbits the per-node optimum is
    ill-defined (the objective flattens as the accessible region shrinks),
    so the fit supplies the smooth extension there.  The survey therefore
    samples eccentric energies only and lets the quadratic extend inward.
    """

    def phieff(R, z, Lz):
        return (
            evaluatePotentials(pot, R, z, use_physical=False) + Lz**2.0 / 2.0 / R**2.0
        )

    Lzlo = Rmin * vcirc(pot, Rmin, use_physical=False)
    Lzhi = Rmax * vcirc(pot, Rmax, use_physical=False)
    nodes = []
    for Lz in numpy.linspace(Lzlo, Lzhi, nsub):
        Rc = rl(pot, Lz, use_physical=False)
        Ec = phieff(Rc, 0.0, Lz)
        Emax = phieff(Rinf, 0.0, Lz)
        for w in numpy.linspace(0.3, 0.95, nsub):
            E = Ec + w**2.0 * (Emax - Ec)
            try:
                Rp = brentq(lambda R: phieff(R, 0.0, Lz) - E, 0.02 * Rc, Rc)
                Ra = brentq(lambda R: phieff(R, 0.0, Lz) - E, Rc, 40.0 * Rc)
            except ValueError:
                continue
            Rmid = 0.5 * (Rp + Ra)
            pts = []
            for R in numpy.linspace(Rp, Ra, 9):
                try:
                    zm = (
                        brentq(lambda z: phieff(R, z, Lz) - E, 0.0, 3.0 * Ra)
                        if phieff(R, 3.0 * Ra, Lz) > E
                        else 3.0 * Ra
                    )
                except ValueError:
                    zm = 0.3 * Ra
                for z in numpy.linspace(0.0, 0.9 * zm, 4):
                    pts.append((R, z))
            pts = numpy.array(pts)
            Ptrue = evaluatePotentials(pot, pts[:, 0], pts[:, 1], use_physical=False)

            def _obj(d):
                try:
                    w_ = OblateStaeckelWrapperPotential(
                        pot=pot, delta=d, u0=numpy.arcsinh(Rmid / d)
                    )
                    return float(
                        numpy.max(
                            numpy.fabs(
                                evaluatePotentials(
                                    w_, pts[:, 0], pts[:, 1], use_physical=False
                                )
                                - Ptrue
                            )
                            / numpy.fabs(Ptrue)
                        )
                    )
                except Exception:
                    return 1e3

            res = minimize_scalar(
                _obj, bounds=(0.1, 3.0), method="bounded", options={"xatol": 1e-3}
            )
            nodes.append((Lz, E, float(res.x), float(res.fun), Rmid))
    if len(nodes) < 6:
        raise RuntimeError(
            "delta='fit' could not survey enough (L_z, E) nodes to fit a "
            "surface; check Rmin/Rmax/Rinf"
        )
    dat = numpy.array(nodes)
    L, E, D = dat[:, 0], dat[:, 1], dat[:, 2]
    A = numpy.vstack([numpy.ones_like(L), L, E, L * E, L**2.0, E**2.0]).T
    c = numpy.linalg.lstsq(A, D, rcond=None)[0]
    rms = float(numpy.sqrt(numpy.mean((A @ c - D) ** 2.0)))
    lo, hi = 0.8 * D.min(), 1.25 * D.max()
    if rms > rms_warn:
        warnings.warn(
            "delta='fit': the fitted focal-length surface has rms residual "
            f"{rms:.3f} against the per-node optima; the family remains "
            "exactly canonical, but the Staeckel models may fit the "
            "potential less well than per-node optimization could",
            galpyWarning,
        )

    def dfun(Ev, Lzv):
        Lzv = numpy.fabs(Lzv)
        return float(
            numpy.clip(
                c[0]
                + c[1] * Lzv
                + c[2] * Ev
                + c[3] * Lzv * Ev
                + c[4] * Lzv**2.0
                + c[5] * Ev**2.0,
                lo,
                hi,
            )
        )

    def u0fun(Ev, Lzv):
        Lzv = numpy.fabs(Lzv)
        try:
            Rc = rl(pot, Lzv, use_physical=False)
            Rp = brentq(lambda R: phieff(R, 0.0, Lzv) - Ev, 0.02 * Rc, Rc)
            Ra = brentq(lambda R: phieff(R, 0.0, Lzv) - Ev, Rc, 40.0 * Rc)
            Rm = 0.5 * (Rp + Ra)
        except Exception:
            Rm = 0.5 * (Rmin + Rmax)
        return float(numpy.arcsinh(Rm / dfun(Ev, Lzv)))

    return dfun, u0fun, {"nodes": dat, "coeffs": c, "rms": rms, "clip": (lo, hi)}


def _u0_midpoint_fun(pot, Rmin, Rmax, dfun):
    """The zero-velocity R-midpoint reference curve: u0(E, L_z) placing
    delta*sinh(u0) at the middle of the (planar) radial range of the
    (E, L_z) orbit family, with the focal length supplied by dfun"""

    def phieff(R, z, Lz):
        return (
            evaluatePotentials(pot, R, z, use_physical=False) + Lz**2.0 / 2.0 / R**2.0
        )

    def u0fun(Ev, Lzv):
        Lzv = numpy.fabs(Lzv)
        try:
            Rc = rl(pot, Lzv, use_physical=False)
            Rp = brentq(lambda R: phieff(R, 0.0, Lzv) - Ev, 0.02 * Rc, Rc)
            Ra = brentq(lambda R: phieff(R, 0.0, Lzv) - Ev, Rc, 40.0 * Rc)
            Rm = 0.5 * (Rp + Ra)
        except Exception:
            Rm = 0.5 * (Rmin + Rmax)
        return float(numpy.arcsinh(Rm / dfun(Ev, Lzv)))

    return u0fun


def _parse_target(target):
    """The phase-space rows (R, vR, vT, z, vz) of a target, from an Orbit
    (possibly an array of orbits, or one evaluated at an array of times) or
    from array rows (R, vR, vT, z, vz[, phi]) in internal units; phi, when
    present, is ignored because the potential is axisymmetric"""
    from ..orbit import Orbit

    if isinstance(target, Orbit):
        out = numpy.array(
            [
                numpy.atleast_1d(target.R(use_physical=False)).ravel(),
                numpy.atleast_1d(target.vR(use_physical=False)).ravel(),
                numpy.atleast_1d(target.vT(use_physical=False)).ravel(),
                numpy.atleast_1d(target.z(use_physical=False)).ravel(),
                numpy.atleast_1d(target.vz(use_physical=False)).ravel(),
            ]
        )
    else:
        arr = numpy.atleast_2d(numpy.asarray(target, dtype="float64"))
        if arr.ndim != 2 or arr.shape[1] not in (5, 6):
            raise ValueError(
                "target= rows must be (R, vR, vT, z, vz) or "
                f"(R, vR, vT, z, vz, phi); got shape {arr.shape}"
            )
        out = arr[:, :5].T.copy()
    if numpy.any(out[0] <= 0.0) or numpy.any(out[0] * numpy.fabs(out[2]) <= 0.0):
        raise ValueError(
            "target= needs R > 0 and L_z = R vT != 0 for every point: the "
            "grid is a box in |L_z| > 0"
        )
    return out


def _edge(lim, default):
    """Resolve a grid limit against its axis's default edges.

    None for the whole limit keeps both defaults; None for either end keeps
    that end.  The point is to let a narrow grid sit ON an edge -- circular,
    planar, shell -- exactly, since those are degeneracies whose handling
    tests for the grid reaching them, not merely approaching them.
    """
    if lim is None:
        return default
    lo, hi = lim
    return (default[0] if lo is None else lo, default[1] if hi is None else hi)
