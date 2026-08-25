###############################################################################
#   actionAngleStaeckelInverse.py: inverse action-angle transformation for
#   axisymmetric Staeckel potentials, computed directly from the separable
#   structure: each angle is a u-profile plus a v-profile, both regular
#   quadratures in the chi anomaly, and (J,theta) -> (x,v) is an additively
#   separable 2x2 Newton solve. No auxiliary torus, generating function, or
#   Fourier lattice; placement on the torus is exact by construction.
###############################################################################
import numpy
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
from scipy.ndimage import map_coordinates, spline_filter1d
from scipy.optimize import brentq, minimize_scalar

from ..potential import (
    OblateStaeckelWrapperPotential,
    evaluatePotentials,
    rl,
    vcirc,
)
from ..util import coords
from .actionAngleInverse import actionAngleInverse

# Nodes/weights for composite 10-point Gauss-Legendre quadrature: applied
# per interval of the nchi-point chi mesh, the error per panel is
# O((pi/nchi)^20), so every stored integral is at machine precision for any
# reasonable nchi; there is no accuracy knob to tune
_GLX, _GLW = numpy.polynomial.legendre.leggauss(10)
# Reference u of the internal Staeckel splitting for potentials that are not
# already wrapped; its value is irrelevant for an exactly Staeckel potential
# (any choice gives the same U, V up to the fixed gauge)
_U0INTERNAL = 1.15


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


class _ProfileSet:
    """Several angle profiles sharing one uniform chi mesh.

    The profiles are stored as cubic B-spline coefficients, padded so the
    four-point stencil is always in range, and evaluated with an explicit
    weight formula rather than through a library routine: the Newton
    iteration needs the same chi for several profiles at a time, and for a
    single angle the per-call overhead of a library interpolator dominates
    everything else.
    """

    def __init__(self, coeffs, dchi):
        self._c = numpy.concatenate(
            (coeffs[:, :1], coeffs[:, :1], coeffs, coeffs[:, -1:], coeffs[:, -1:]),
            axis=1,
        )
        self._dchi = dchi
        self._n = coeffs.shape[1]

    def block(self, which):
        """The padded coefficients of a subset of the profiles, sliced once
        so that repeated evaluations do not re-copy them"""
        return numpy.ascontiguousarray(self._c[which])

    def evaluate(self, block, chi):
        """Evaluate a block returned by block() at the angles chi"""
        x = numpy.clip(chi / self._dchi, 0.0, self._n - 1.0)
        i = x.astype(int)
        t = x - i
        j = i + 2
        t2 = t * t
        t3 = t2 * t
        return (
            (1.0 - 3.0 * t + 3.0 * t2 - t3) * block[:, j - 1]
            + (4.0 - 6.0 * t2 + 3.0 * t3) * block[:, j]
            + (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t3) * block[:, j + 1]
            + t3 * block[:, j + 2]
        ) / 6.0

    def __call__(self, chi, which):
        return self.evaluate(self.block(which), numpy.atleast_1d(chi))


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
    for map_coordinates with prefilter=False. Only the given axes are
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
        nchi_store=201,
        nchi=2001,
        maxiter=60,
        angle_tol=1e-13,
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
        nchi_store : int, optional
            Number of grid points in the chi anomaly on which the angle
            profiles of the grid tori are stored for interpolation (only used
            when setup_interp is True). The profiles are smooth functions of
            the anomaly by construction, so this can be much smaller than
            nchi.
        maxiter : int, optional
            Maximum number of Newton iterations in the angle inversion.
        angle_tol : float, optional
            Convergence tolerance of the angle inversion.

        Notes
        -----
        - Angle conventions match those of the forward actionAngleStaeckel.
        - When set up with setup_interp, the evaluation methods accept
          ``integrals=True``, which reinterprets their first three arguments
          as (E, L_z, I3) rather than (J_R, J_phi, J_z). Tori are naturally
          labelled by their integrals -- the construction takes those as
          input and delivers the actions as output -- so this route skips the
          inversion that the action route performs, and is correspondingly
          more accurate.
        - 2026-08-19 - Started - Bovy (UofT)
        """
        if "delta" in kwargs or "u0" in kwargs:
            raise TypeError(
                "actionAngleStaeckelInverse does not accept delta= or u0=: "
                "the potential itself supplies the focal distance (pass an "
                "OblateStaeckelWrapperPotential or a potential with a "
                "_delta attribute)"
            )
        actionAngleInverse.__init__(self, **kwargs)
        if pot is None:  # pragma: no cover
            raise OSError("Must specify pot= for actionAngleStaeckelInverse")
        self._pot = pot
        if isinstance(pot, OblateStaeckelWrapperPotential):
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
            self._staeckelwrap = OblateStaeckelWrapperPotential(
                pot=pot, delta=delta, u0=_U0INTERNAL
            )
        self._delta = self._staeckelwrap._delta
        self._Es = numpy.atleast_1d(numpy.array(Es, dtype="float"))
        self._Lzs = numpy.atleast_1d(numpy.array(Lzs, dtype="float"))
        self._I3s = numpy.atleast_1d(numpy.array(I3s, dtype="float"))
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
        if setup_interp:
            self._setup_grid(Rmin, Rmax, Rinf, nLz, nE, nI3, grid_pad, nchi_store)
            return
        # Setup in three logical stages
        self._find_turning_points()
        self._compute_actions_frequencies_profiles()
        self._build_angle_profile_splines()

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

    def _I3_shell(self, E, Lz):
        """I3 of the J_R = 0 edge, where W_u acquires a double root"""

        def maxWu(I3):
            return -minimize_scalar(
                lambda u: -self._Wu(u, E, Lz, I3),
                bounds=(1e-3, 20.0),
                method="bounded",
                options={"xatol": 1e-13},
            ).fun

        lo = self._I3_planar(E, Lz)
        hi = lo + 1.0
        while maxWu(hi) > 0.0:
            hi += 1.0
        return brentq(maxWu, lo, hi, xtol=1e-14)

    def _setup_grid(self, Rmin, Rmax, Rinf, nLz, nE, nI3, wpad, nchistore):
        """Grid of tori spanning the bound phase space, and the tables that
        recover a torus's integrals from its actions without root finding.

        The grid is rectified at every edge, because torus properties vary as
        the square root of the distance to each: uniform in L_z, quadratic in
        E away from the circular orbit (as in the spherical grid), and in
        s = sin^2(pi w_I/2) between the planar (J_z = 0) and shell (J_R = 0)
        edges, which rectifies both at once. Uniform spacing in (E, I3) is not
        merely slower to converge but useless, with relative errors of order
        unity in the actions.
        """
        self._nLz, self._nE, self._nI3 = nLz, nE, nI3
        self._nchistore = nchistore
        self._Lzgrid = numpy.linspace(
            Rmin * vcirc(self._pot, Rmin, use_physical=False),
            Rmax * vcirc(self._pot, Rmax, use_physical=False),
            nLz,
        )
        self._wEgrid = numpy.linspace(wpad, 1.0 - wpad, nE)
        # w_I spans the CLOSED interval so that J_R = 0 (shell) and J_z = 0
        # (planar) are grid nodes rather than lying outside the grid. The
        # construction degenerates exactly there, so those two nodes are
        # built a small step inside and then corrected to their analytic
        # limits below.
        self._wIgrid = numpy.linspace(0.0, 1.0, nI3)
        self._wIedge = 1e-4
        shape = (nLz, nE, nI3)
        Es, Lzs, I3s = (numpy.empty(shape) for _ in range(3))
        self._Ecs, self._Emaxs = numpy.empty(nLz), numpy.empty(nLz)
        self._grid_Ish = numpy.empty((nLz, nE))
        wIbuild = numpy.clip(self._wIgrid, self._wIedge, 1.0 - self._wIedge)
        sinw = numpy.sin(numpy.pi * wIbuild / 2.0) ** 2.0
        for ii, Lz in enumerate(self._Lzgrid):
            self._Ecs[ii] = self._circular_orbit(Lz)[1]
            self._Emaxs[ii] = (
                evaluatePotentials(self._pot, Rinf, 0.0, use_physical=False)
                + Lz**2.0 / 2.0 / Rinf**2.0
            )
            for jj, wE in enumerate(self._wEgrid):
                E = self._Ecs[ii] + wE**2.0 * (self._Emaxs[ii] - self._Ecs[ii])
                Ipl = self._I3_planar(E, Lz)
                self._grid_Ish[ii, jj] = self._I3_shell(E, Lz)
                Es[ii, jj] = E
                Lzs[ii, jj] = Lz
                I3s[ii, jj] = Ipl + sinw * (self._grid_Ish[ii, jj] - Ipl)
        # Build every torus of the grid in one vectorized construction
        grid = actionAngleStaeckelInverse(
            pot=self._staeckelwrap,
            Es=Es.ravel(),
            Lzs=Lzs.ravel(),
            I3s=I3s.ravel(),
            nchi=self._nchi,
        )
        self._grid_jr = grid._jr.reshape(shape).copy()
        self._grid_jz = grid._jz.reshape(shape).copy()
        # Set the vanishing action to exactly zero at each edge, so that
        # zeta reaches 0 and 1 and J_z = 0, J_R = 0 are inside the grid
        if self._wIgrid[-1] == 1.0:
            self._grid_jr[:, :, -1] = 0.0
        if self._wIgrid[0] == 0.0:
            self._grid_jz[:, :, 0] = 0.0
        # Keep what the construction produced -- turning points,
        # frequencies, and the six profiles with their derivatives -- on a
        # common, coarser chi mesh: rebuilding a torus instead costs ~45 ms
        self._chistore = numpy.linspace(0.0, numpy.pi, self._nchistore)
        nprof = self._nchistore
        prof = numpy.empty((12, grid._ntori, nprof))
        for kk in range(grid._ntori):
            for ll in range(3):
                prof[ll, kk] = grid._Aprof[kk][ll](self._chistore)
                prof[3 + ll, kk] = grid._Bprof[kk][ll](self._chistore)
                prof[6 + ll, kk] = grid._dAprof[kk][ll](self._chistore)
                prof[9 + ll, kk] = grid._dBprof[kk][ll](self._chistore)
        # Analytic limits on the degenerate oscillation of each edge: it is
        # harmonic there, so its angle is its anomaly (A_R = chi at the
        # shell, B_z = chi at the planar edge) and its cross profiles vanish
        prof = prof.reshape((12,) + shape + (nprof,))
        chis = self._chistore
        ones = numpy.ones_like(chis)
        if self._wIgrid[-1] == 1.0:  # shell edge: J_R = 0
            prof[0, :, :, -1] = chis
            prof[6, :, :, -1] = ones
            for ll in (1, 2):
                prof[ll, :, :, -1] = 0.0
                prof[6 + ll, :, :, -1] = 0.0
        if self._wIgrid[0] == 0.0:  # planar edge: J_z = 0
            prof[4, :, :, 0] = chis
            prof[10, :, :, 0] = ones
            for ll in (0, 2):
                prof[3 + ll, :, :, 0] = 0.0
                prof[9 + ll, :, :, 0] = 0.0
        prof = prof.reshape((12, grid._ntori, nprof))
        # the degenerate turning points collapse: u_min = u_max at the shell,
        # v_min = pi/2 at the planar edge
        umins = grid._umins.reshape(shape).copy()
        umaxs = grid._umaxs.reshape(shape).copy()
        vmins = grid._vmins.reshape(shape).copy()
        if self._wIgrid[-1] == 1.0:
            mid = 0.5 * (umins[:, :, -1] + umaxs[:, :, -1])
            umins[:, :, -1] = mid
            umaxs[:, :, -1] = mid
        if self._wIgrid[0] == 0.0:
            vmins[:, :, 0] = numpy.pi / 2.0
        grid._umins = umins.ravel()
        grid._umaxs = umaxs.ravel()
        grid._vmins = vmins.ravel()
        self._grid_scal = numpy.array(
            [
                grid._umins,
                grid._umaxs,
                grid._vmins,
                grid._OmegaR,
                grid._Omegaz,
                grid._Omegaphi,
                Es.ravel(),
                I3s.ravel(),
            ]
        ).reshape((8,) + shape)
        # Prefilter along chi once here: prefiltering and interpolation
        # across the grid are both linear, so they commute, and doing it now
        # keeps it off the per-torus path
        prof = spline_filter1d(prof, order=3, axis=-1, mode="nearest")
        self._grid_prof = prof.reshape((12,) + shape + (nprof,))
        # Pad the grid directions by polynomial extrapolation before
        # prefiltering: spline_filter assumes a vanishing derivative beyond
        # the edge, which otherwise dominates the error there (7.7e-4
        # unpadded versus 2.6e-6 padded)
        self._gpad = 4
        self._grid_scal_f = _prefilter_padded(self._grid_scal, (1, 2, 3), self._gpad)
        self._grid_prof_f = _prefilter_padded(self._grid_prof, (1, 2, 3), self._gpad)
        self._Ish_spl = RectBivariateSpline(self._Lzgrid, self._wEgrid, self._grid_Ish)
        self._Ec_spl = InterpolatedUnivariateSpline(self._Lzgrid, self._Ecs, k=3)
        self._Emax_spl = InterpolatedUnivariateSpline(self._Lzgrid, self._Emaxs, k=3)
        self._build_action_lookup()

    def _build_action_lookup(self):
        """Tables giving the grid coordinates (w_E, w_I) of a torus from its
        actions, by two nested one-dimensional monotone inversions rather
        than a two-dimensional root find.

        rho = sqrt(J_R + J_z) increases with w_E at fixed w_I, and
        zeta = (2/pi) arcsin sqrt[J_z/(J_R + J_z)] increases with w_I at
        fixed w_E. zeta carries the same rectification as the grid: J_z
        vanishes as w_I^2 at the planar edge (and J_R as [1-w_I]^2 at the
        shell edge), so w_I would be a square root of the bare action ratio
        there, while the arcsin makes it linear.
        """
        nLz, nE, nI3 = self._nLz, self._nE, self._nI3
        rho = numpy.sqrt(self._grid_jr + self._grid_jz)
        eta = self._grid_jz / (self._grid_jr + self._grid_jz)
        zeta = 2.0 / numpy.pi * numpy.arcsin(numpy.sqrt(numpy.clip(eta, 0.0, 1.0)))
        self._zetamesh = numpy.linspace(
            numpy.amax(zeta.min(axis=2)), numpy.amin(zeta.max(axis=2)), nI3
        )
        wI_z, rho_z = numpy.empty((nLz, nE, nI3)), numpy.empty((nLz, nE, nI3))
        for ii in range(nLz):
            for jj in range(nE):
                wI_z[ii, jj] = InterpolatedUnivariateSpline(
                    zeta[ii, jj], self._wIgrid, k=3
                )(self._zetamesh)
                rho_z[ii, jj] = InterpolatedUnivariateSpline(
                    zeta[ii, jj], rho[ii, jj], k=3
                )(self._zetamesh)
        self._rhomax = rho_z[:, -1, :]
        rhat = rho_z / self._rhomax[:, None, :]
        self._rhatmesh = numpy.linspace(numpy.amax(rhat[:, 0, :]), 1.0, nE)
        self._tab_wE, self._tab_wI = (numpy.empty((nLz, nE, nI3)) for _ in range(2))
        self._rhomax_f = None
        for ii in range(nLz):
            for kk in range(nI3):
                self._tab_wE[ii, :, kk] = InterpolatedUnivariateSpline(
                    rhat[ii, :, kk], self._wEgrid, k=3
                )(self._rhatmesh)
                self._tab_wI[ii, :, kk] = InterpolatedUnivariateSpline(
                    rhat[ii, :, kk], wI_z[ii, :, kk], k=3
                )(self._rhatmesh)
        # prefilter once: rebuilding splines on every call dominated the cost
        self._tab_f = _prefilter_padded(
            numpy.array([self._tab_wE, self._tab_wI]), (1, 2, 3), self._gpad
        )
        self._rhomax_f = _prefilter_padded(self._rhomax[None], (1, 2), self._gpad)

    def _integrals_from_actions(self, jr, jphi, jz):
        """(J_R, J_phi, J_z) -> (E, L_z, I3), by interpolation only: rho and
        zeta give the position within the rectified action space, and the
        stored tables convert that to grid coordinates"""
        Lz = jphi
        if Lz < self._Lzgrid[0] or Lz > self._Lzgrid[-1]:
            raise ValueError(
                f"J_phi = {Lz} lies outside the grid of this "
                f"actionAngleStaeckelInverse instance "
                f"([{self._Lzgrid[0]}, {self._Lzgrid[-1]}])"
            )
        rho = numpy.sqrt(jr + jz)
        zeta = (
            2.0
            / numpy.pi
            * numpy.arcsin(numpy.sqrt(numpy.clip(jz / (jr + jz), 0.0, 1.0)))
        )
        p = self._gpad
        iLz = numpy.interp(Lz, self._Lzgrid, numpy.arange(self._nLz))
        izeta = numpy.interp(zeta, self._zetamesh, numpy.arange(self._nI3))
        rhomax = map_coordinates(
            self._rhomax_f[0],
            numpy.array([[iLz + p], [izeta + p]]),
            order=3,
            prefilter=False,
            mode="nearest",
        )[0]
        rhat = rho / rhomax
        if not (
            self._rhatmesh[0] <= rhat <= self._rhatmesh[-1]
            and self._zetamesh[0] <= zeta <= self._zetamesh[-1]
        ):
            raise ValueError(
                "Given actions lie outside the grid of this "
                "actionAngleStaeckelInverse instance"
            )
        irhat = numpy.interp(rhat, self._rhatmesh, numpy.arange(self._nE))
        c = numpy.array([[iLz + p], [irhat + p], [izeta + p]])
        wE = map_coordinates(
            self._tab_f[0], c, order=3, prefilter=False, mode="nearest"
        )[0]
        wI = map_coordinates(
            self._tab_f[1], c, order=3, prefilter=False, mode="nearest"
        )[0]
        return float(wE), float(wI)

    def _coords_from_actions(self, jr, jphi, jz):
        """(J_R, J_phi, J_z) -> fractional grid index, directly. Going via
        (E, I3) and back would evaluate the shell edge and the energy limits
        twice for nothing: the torus quantities that evaluation needs,
        including E and I3 themselves, are interpolated on the grid."""
        wE, wI = self._integrals_from_actions(jr, jphi, jz)
        # A vanishing action means the oscillation is absent, which is an
        # edge of the grid: snap to it, or the table's rounding leaves a
        # sliver of oscillation behind
        if jz <= 0.0:
            wI = self._wIgrid[0]
        elif jr <= 0.0:
            wI = self._wIgrid[-1]
        return self._fractional_index(jphi, wE, wI)

    def _interp_torus(self, idx):
        """Interpolate the stored torus quantities at a fractional grid index.

        The three grid directions are contracted explicitly with the cubic
        B-spline weights: calling a library interpolator once per profile
        instead costs more than everything else in an evaluation. The result
        is cached, because the usual pattern is many angles on one torus.
        """
        key = tuple(idx)
        if getattr(self, "_torus_cache_key", None) == key:
            return self._torus_cache
        p = self._gpad
        w, base = [], []
        for aa in range(3):
            x = idx[aa] + p
            ii = int(numpy.floor(x))
            w.append(_bspline_weights(x - ii))
            base.append(ii - 1)
        sl = (
            slice(base[0], base[0] + 4),
            slice(base[1], base[1] + 4),
            slice(base[2], base[2] + 4),
        )
        Ws = w[0][:, None, None] * w[1][None, :, None] * w[2][None, None, :]
        scal = numpy.tensordot(
            self._grid_scal_f[:, sl[0], sl[1], sl[2]], Ws, axes=([1, 2, 3], [0, 1, 2])
        )
        # contract the grid axes one at a time: contracting all three at once
        # forces a copy of the (12, 4, 4, 4, nchi) slice, which costs more
        # than the arithmetic it saves
        prof = self._grid_prof_f[:, sl[0], sl[1], sl[2], :]
        prof = numpy.tensordot(prof, w[0], axes=([1], [0]))
        prof = numpy.tensordot(prof, w[1], axes=([1], [0]))
        prof = numpy.tensordot(prof, w[2], axes=([1], [0]))
        profs = _ProfileSet(prof, self._chistore[1] - self._chistore[0])
        # Polish the turning points against the interpolated integrals:
        # interpolated, they are not exact roots of W, so p = sqrt(W) is
        # clipped near them and the energy drifts by ~1e-6. A degenerate
        # oscillation is left alone.
        Lz = self._interp_Lz
        E = scal[6]
        if numpy.pi / 2.0 - scal[2] <= 1e-12:
            # planar torus: the exact condition W_v(pi/2) = 0 fixes I3 in
            # closed form, which is better than any interpolated value
            scal[7] = self._I3_planar(E, Lz)
        elif scal[1] - scal[0] <= 1e-12:
            # shell torus: a double root needs both W_u'(u*) = 0 and
            # W_u(u*) = 0, so u* and I3 are solved together; W_u depends on
            # I3 only through -2 delta^2 I3, making the second update exact
            ustar, I3s_, h = scal[0], scal[7], 1e-4
            for _ in range(3):
                # three points in one call give slope and curvature
                uu = numpy.array([ustar - h, ustar, ustar + h])
                Wq = self._Wu(uu, E, Lz, I3s_)
                d1 = (Wq[2] - Wq[0]) / 2.0 / h
                d2 = (Wq[2] - 2.0 * Wq[1] + Wq[0]) / h**2.0
                ustar -= d1 / d2
                I3s_ += (
                    self._Wu(numpy.array([ustar]), E, Lz, I3s_)[0]
                    / 2.0
                    / self._delta**2.0
                )
            scal[0] = scal[1] = ustar
            scal[7] = I3s_
        I3 = scal[7]
        # One vectorized evaluation of W per degree of freedom, differenced
        # for the slope: the analytic dW costs three potential-layer calls
        # against one, and only sets the step, not the root
        h = 1e-6
        # Two steps: the turning points arrive with the ~1e-5 error of the
        # interpolation, so one step leaves ~1e-10, which shows up in the
        # energy when angles sample close to a turning point
        for _ in range(2):
            if scal[1] - scal[0] > 1e-12:
                uu = numpy.array(
                    [
                        scal[0] - h,
                        scal[0],
                        scal[0] + h,
                        scal[1] - h,
                        scal[1],
                        scal[1] + h,
                    ]
                )
                Wu = self._Wu(uu, E, Lz, I3)
                scal[0] -= Wu[1] * 2.0 * h / (Wu[2] - Wu[0])
                scal[1] -= Wu[4] * 2.0 * h / (Wu[5] - Wu[3])
            if numpy.pi / 2.0 - scal[2] > 1e-12:
                vv = numpy.array([scal[2] - h, scal[2], scal[2] + h])
                Wv = self._Wv(vv, E, Lz, I3)
                scal[2] -= Wv[1] * 2.0 * h / (Wv[2] - Wv[0])
        self._torus_cache_key = key
        self._torus_cache = (scal, profs)
        return self._torus_cache

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

    @staticmethod
    def _fold_set(profs, w, block, half):
        """Fold several profiles of a _ProfileSet at once.

        The argument is folded BEFORE evaluation -- chi = w on the outgoing
        branch and 2pi - w on the return branch -- so each profile is
        evaluated once instead of on both branches, and the half-loop values
        needed by the reflection are passed in rather than re-evaluated.
        """
        w = numpy.mod(w, 2.0 * numpy.pi)
        out = w <= numpy.pi
        chi = numpy.where(out, w, 2.0 * numpy.pi - w)
        vals = profs.evaluate(block, chi)
        refl = 2.0 * half[:, None] - vals
        # rows flagged with a NaN half are derivatives: the reflection leaves
        # them unchanged in the folded argument
        keep = numpy.isnan(half)
        refl[keep] = vals[keep]
        return numpy.where(out, vals, refl)

    def _evaluate(self, jr, jphi, jz, angler, anglephi, anglez, **kwargs):
        return self._xvFreqs(jr, jphi, jz, angler, anglephi, anglez, **kwargs)[:6]

    def _grid_coords(self, E, Lz, I3):
        """(E, L_z, I3) -> fractional grid indices. This direction needs no
        inversion at all: w_E follows from the circular and outer energies,
        w_I from the planar and shell edges."""
        if Lz < self._Lzgrid[0] or Lz > self._Lzgrid[-1]:
            raise ValueError(
                f"L_z = {Lz} lies outside the grid of this "
                f"actionAngleStaeckelInverse instance "
                f"([{self._Lzgrid[0]}, {self._Lzgrid[-1]}])"
            )
        Ec, Emax = self._Ec_spl(Lz), self._Emax_spl(Lz)
        wE = numpy.sqrt(numpy.clip((E - Ec) / (Emax - Ec), 0.0, numpy.inf))
        Ipl = self._I3_planar(E, Lz)
        Ish = self._Ish_spl(Lz, numpy.clip(wE, self._wEgrid[0], self._wEgrid[-1]))[0, 0]
        s = numpy.clip((I3 - Ipl) / (Ish - Ipl), 0.0, 1.0)
        wI = 2.0 / numpy.pi * numpy.arcsin(numpy.sqrt(s))
        return self._fractional_index(Lz, float(wE), float(wI))

    def _fractional_index(self, Lz, wE, wI):
        """Fractional index into the (L_z, w_E, w_I) grid, checking that the
        torus is inside it"""
        out = []
        for val, grid, name in (
            (Lz, self._Lzgrid, "L_z"),
            (wE, self._wEgrid, "energy"),
            (wI, self._wIgrid, "third integral"),
        ):
            # a torus on an edge can land a rounding step outside it
            tol = 1e-6 * (grid[-1] - grid[0])
            if val < grid[0] - tol or val > grid[-1] + tol:
                raise ValueError(
                    f"Requested torus lies outside the grid of this "
                    f"actionAngleStaeckelInverse instance in the {name} "
                    "direction"
                )
            out.append(numpy.clip(val, grid[0], grid[-1]))
        return (
            numpy.interp(out[0], self._Lzgrid, numpy.arange(self._nLz)),
            numpy.interp(out[1], self._wEgrid, numpy.arange(self._nE)),
            numpy.interp(out[2], self._wIgrid, numpy.arange(self._nI3)),
        )

    def _xvFreqs(self, jr, jphi, jz, angler, anglephi, anglez, **kwargs):
        if kwargs.get("integrals", False):
            if not self._interp:
                raise ValueError(
                    "integrals=True requires an actionAngleStaeckelInverse "
                    "set up with setup_interp=True"
                )
            # (jr, jphi, jz) are really (E, L_z, I3) here: the grid
            # coordinates follow from them directly, with no inversion
            return self._xvFreqs_index(
                self._grid_coords(jr, jphi, jz), jphi, angler, anglephi, anglez
            )
        if self._interp:
            return self._xvFreqs_index(
                self._coords_from_actions(jr, jphi, jz),
                jphi,
                angler,
                anglephi,
                anglez,
            )
        ii = self._torus_index(jr, jphi, jz)
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

    def _xvFreqs_index(self, idx, Lz, angler, anglephi, anglez):
        """Evaluate on the interpolated torus at a fractional grid index"""
        self._interp_Lz = Lz
        scal, profs = self._interp_torus(idx)
        return self._solve_and_map(
            profs,
            None,
            None,
            None,
            scal[6],
            Lz,
            scal[7],
            scal[0],
            scal[1],
            scal[2],
            scal[3],
            scal[5],
            scal[4],
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
        A is either a list of profile splines (direct) or a _ProfileSet
        holding all twelve profiles of an interpolated torus."""
        thR = numpy.atleast_1d(numpy.array(angler, dtype="float"))
        thz = numpy.atleast_1d(numpy.array(anglez, dtype="float"))
        thphi = numpy.atleast_1d(numpy.array(anglephi, dtype="float"))
        batched = isinstance(A, _ProfileSet)
        if batched:
            profs = A
            # profile order: A_R,A_z,A_phi,B_R,B_z,B_phi then their derivatives
            # values and derivatives share the angle, hence the weights
            blku = profs.block(numpy.array([0, 1, 6, 7]))
            blkv = profs.block(numpy.array([3, 4, 9, 10]))
            bphiu = profs.block(numpy.array([2]))
            bphiv = profs.block(numpy.array([5]))
            allhalf = profs(numpy.array([numpy.pi]), numpy.arange(12))[:, 0]
            # NaN marks the derivative rows, which the reflection leaves
            # unchanged in the folded argument
            halfu = numpy.array([allhalf[0], allhalf[1], numpy.nan, numpy.nan])
            halfv = numpy.array([allhalf[3], allhalf[4], numpy.nan, numpy.nan])

            def fu(w):
                return self._fold_set(profs, w, blku, halfu)

            def fv(w):
                return self._fold_set(profs, w, blkv, halfv)

        wu, wv = numpy.copy(thR), numpy.copy(thz)
        unconv = numpy.ones(wu.shape, dtype="bool")
        for _ in range(self._maxiter):
            twu, twv = wu[unconv], wv[unconv]
            if batched:
                fA, fB = fu(twu), fv(twv)
                f0 = fA[0] + fB[0] - thR[unconv]
                f1 = fA[1] + fB[1] + self._anglez0 - thz[unconv]
                J00, J01 = fA[2], fB[2]
                J10, J11 = fA[3], fB[3]
            else:
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
        if batched:
            phi = (
                thphi
                - self._fold_set(profs, wu, bphiu, allhalf[[2]])[0]
                - self._fold_set(profs, wv, bphiv, allhalf[[5]])[0]
            )
        else:
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
        if kwargs.get("integrals", False):
            if not self._interp:
                raise ValueError(
                    "integrals=True requires an actionAngleStaeckelInverse "
                    "set up with setup_interp=True"
                )
            self._interp_Lz = jphi
            scal = self._interp_torus(self._grid_coords(jr, jphi, jz))[0]
            return (scal[3], scal[5], scal[4])
        if self._interp:
            self._interp_Lz = jphi
            scal = self._interp_torus(self._coords_from_actions(jr, jphi, jz))[0]
            return (scal[3], scal[5], scal[4])
        ii = self._torus_index(jr, jphi, jz)
        return (self._OmegaR[ii], self._Omegaphi[ii], self._Omegaz[ii])
