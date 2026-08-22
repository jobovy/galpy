###############################################################################
#   actionAngleStaeckelInverse.py: inverse action-angle transformation for
#   axisymmetric Staeckel potentials, computed directly from the separable
#   structure: each angle is the sum of a u-profile and a v-profile
#   (theta_i = dW/dJ_i with W = int p_u du + int p_v dv + L_z phi), with all
#   profiles evaluated as regular quadratures in the chi anomaly and the
#   (J,theta) -> (x,v) direction solved by an additively-separable 2x2
#   Newton iteration. No auxiliary torus, generating function, or Fourier
#   lattice is involved; placement on the torus is exact by construction.
###############################################################################
import numpy
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
from scipy.optimize import brentq, minimize_scalar

from ..potential import OblateStaeckelWrapperPotential, evaluatePotentials, vcirc
from ..potential.Potential import _evaluateRforces
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
            Number of grid points in the chi anomaly for the stored
            profiles.
        maxiter : int, optional
            Maximum number of Newton iterations in the angle inversion.
        angle_tol : float, optional
            Convergence tolerance of the angle inversion.

        Notes
        -----
        - Angle conventions match those of the forward actionAngleStaeckel.
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
            self._setup_grid(Rmin, Rmax, Rinf, nLz, nE, nI3, grid_pad)
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
                # The two turning points are separated by a few ulp: the
                # oscillation cannot be resolved in double precision, and
                # everything built on it (the profiles, and hence the
                # frequencies) would be noise. Fail loudly rather than
                # return a plausible-looking zero
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
            # Near the turning points the direct evaluation of W is a
            # difference of O(1) terms and is dominated by cancellation, so
            # reconstruct Q there from the analytic derivative by the
            # trapezoid W ~ (q - q0) [W'(q0) + W'(q)]/2, whose O(y^2) model
            # error is far below the switch threshold. The switch has to be
            # RELATIVE to the size of W on the torus, not a fixed threshold
            # in the anomaly: the signal near a turning point scales as
            # W' D y while the cancellation noise does not, so for a very
            # thin oscillation (a near-shell or near-planar torus) the
            # noise-dominated region is wider than any fixed anomaly cut,
            # and a single node left on the wrong side of it makes 1/sqrt(Q)
            # blow up and destroys the frequencies
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
        # period matrices: pi dJ_R = PEu[-1] dE - PIu[-1] dI3 - PLu[-1] dLz;
        #                  pi dJ_z = PEv[-1] dE + PIv[-1] dI3 - PLv[-1] dLz
        # Because J_phi = L_z identically, the third row of M is (0,0,1) and
        # M is block-triangular: its inverse follows in closed form from the
        # 2x2 (E,I3) block, with no general matrix inversion needed (the same
        # 2x2 structure the forward actionAngleStaeckel uses for its
        # frequencies)
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
        Rc = brentq(
            lambda R: Lz**2.0 / R**3.0 + _evaluateRforces(self._pot, R, 0.0),
            1e-4,
            1e4,
            xtol=1e-14,
        )
        return Rc, evaluatePotentials(self._pot, Rc, 0.0) + Lz**2.0 / 2.0 / Rc**2.0

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

    def _setup_grid(self, Rmin, Rmax, Rinf, nLz, nE, nI3, wpad):
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
        self._Lzgrid = numpy.linspace(
            Rmin * vcirc(self._pot, Rmin), Rmax * vcirc(self._pot, Rmax), nLz
        )
        self._wEgrid = numpy.linspace(wpad, 1.0 - wpad, nE)
        self._wIgrid = numpy.linspace(wpad, 1.0 - wpad, nI3)
        shape = (nLz, nE, nI3)
        Es, Lzs, I3s = (numpy.empty(shape) for _ in range(3))
        self._Ecs, self._Emaxs = numpy.empty(nLz), numpy.empty(nLz)
        sinw = numpy.sin(numpy.pi * self._wIgrid / 2.0) ** 2.0
        for ii, Lz in enumerate(self._Lzgrid):
            self._Ecs[ii] = self._circular_orbit(Lz)[1]
            self._Emaxs[ii] = (
                evaluatePotentials(self._pot, Rinf, 0.0) + Lz**2.0 / 2.0 / Rinf**2.0
            )
            for jj, wE in enumerate(self._wEgrid):
                E = self._Ecs[ii] + wE**2.0 * (self._Emaxs[ii] - self._Ecs[ii])
                Ipl = self._I3_planar(E, Lz)
                Es[ii, jj] = E
                Lzs[ii, jj] = Lz
                I3s[ii, jj] = Ipl + sinw * (self._I3_shell(E, Lz) - Ipl)
        # Build every torus of the grid in one vectorized construction and
        # keep only what the lookup needs: its actions
        grid = actionAngleStaeckelInverse(
            pot=self._staeckelwrap,
            Es=Es.ravel(),
            Lzs=Lzs.ravel(),
            I3s=I3s.ravel(),
            nchi=self._nchi,
        )
        self._grid_jr = grid._jr.reshape(shape)
        self._grid_jz = grid._jz.reshape(shape)
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
        for ii in range(nLz):
            for kk in range(nI3):
                self._tab_wE[ii, :, kk] = InterpolatedUnivariateSpline(
                    rhat[ii, :, kk], self._wEgrid, k=3
                )(self._rhatmesh)
                self._tab_wI[ii, :, kk] = InterpolatedUnivariateSpline(
                    rhat[ii, :, kk], wI_z[ii, :, kk], k=3
                )(self._rhatmesh)

    def _integrals_from_actions(self, jr, jphi, jz):
        """(J_R, J_phi, J_z) -> (E, L_z, I3), by interpolation only"""
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

        # interpolate the tables in L_z first, then read (w_E, w_I) off them
        def alongLz(tab):
            return numpy.array(
                [
                    InterpolatedUnivariateSpline(self._Lzgrid, tab[:, aa, bb], k=3)(Lz)
                    for aa in range(tab.shape[1])
                    for bb in range(tab.shape[2])
                ]
            ).reshape(tab.shape[1:])

        rhomax = InterpolatedUnivariateSpline(
            self._zetamesh,
            numpy.array(
                [
                    InterpolatedUnivariateSpline(
                        self._Lzgrid, self._rhomax[:, kk], k=3
                    )(Lz)
                    for kk in range(self._nI3)
                ]
            ),
            k=3,
        )(zeta)
        rhat = rho / rhomax
        if not (self._rhatmesh[0] <= rhat <= self._rhatmesh[-1]) or not (
            self._zetamesh[0] <= zeta <= self._zetamesh[-1]
        ):
            raise ValueError(
                "Given actions lie outside the grid of this "
                "actionAngleStaeckelInverse instance"
            )
        wE = RectBivariateSpline(self._rhatmesh, self._zetamesh, alongLz(self._tab_wE))(
            rhat, zeta
        )[0, 0]
        wI = RectBivariateSpline(self._rhatmesh, self._zetamesh, alongLz(self._tab_wI))(
            rhat, zeta
        )[0, 0]
        Ec = InterpolatedUnivariateSpline(self._Lzgrid, self._Ecs, k=3)(Lz)
        Emax = InterpolatedUnivariateSpline(self._Lzgrid, self._Emaxs, k=3)(Lz)
        E = Ec + wE**2.0 * (Emax - Ec)
        Ipl = self._I3_planar(E, Lz)
        I3 = Ipl + numpy.sin(numpy.pi * wI / 2.0) ** 2.0 * (self._I3_shell(E, Lz) - Ipl)
        return float(E), float(Lz), float(I3)

    ############################ EVALUATION ###################################
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

    def _interp_instance(self, jr, jphi, jz):
        """Single-torus instance for the given actions, from the integrals
        that the grid supplies; cached, because the usual pattern is many
        angles on one torus"""
        key = (jr, jphi, jz)
        if getattr(self, "_interp_cache_key", None) != key:
            E, Lz, I3 = self._integrals_from_actions(jr, jphi, jz)
            self._interp_cache_key = key
            self._interp_cache = actionAngleStaeckelInverse(
                pot=self._staeckelwrap,
                Es=[E],
                Lzs=[Lz],
                I3s=[I3],
                nchi=self._nchi,
                maxiter=self._maxiter,
                angle_tol=self._angle_tol,
            )
        return self._interp_cache

    def _xvFreqs(self, jr, jphi, jz, angler, anglephi, anglez, **kwargs):
        if self._interp:
            sub = self._interp_instance(jr, jphi, jz)
            return sub._xvFreqs(
                sub._jr[0], jphi, sub._jz[0], angler, anglephi, anglez, **kwargs
            )
        ii = self._torus_index(jr, jphi, jz)
        A, B = self._Aprof[ii], self._Bprof[ii]
        dA, dB = self._dAprof[ii], self._dBprof[ii]
        thR = numpy.atleast_1d(numpy.array(angler, dtype="float"))
        thz = numpy.atleast_1d(numpy.array(anglez, dtype="float"))
        thphi = numpy.atleast_1d(numpy.array(anglephi, dtype="float"))
        # Solve the additively separable 2x2 system for the extended phases,
        # iterating only on the angles that have not yet converged (as in the
        # 1D and spherical inverses): the Newton iteration is quadratically
        # convergent, so most angles are done in a few steps while a few
        # near-degenerate ones take longer
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
            f0 = (f0 + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
            f1 = (f1 + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
            J00, J01 = self._dfold(dA[0], twu), self._dfold(dB[0], twv)
            J10, J11 = self._dfold(dA[1], twu), self._dfold(dB[1], twv)
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
        E, Lz, I3 = self._Es[ii], self._Lzs[ii], self._I3s[ii]
        umin, umaxx, vmin = self._umins[ii], self._umaxs[ii], self._vmins[ii]
        Du, Dv = umaxx - umin, numpy.pi - 2.0 * vmin
        wum, wvm = numpy.mod(wu, 2.0 * numpy.pi), numpy.mod(wv, 2.0 * numpy.pi)
        chiu = numpy.where(wum <= numpy.pi, wum, 2.0 * numpy.pi - wum)
        chiv = numpy.where(wvm <= numpy.pi, wvm, 2.0 * numpy.pi - wvm)
        su = numpy.where(wum <= numpy.pi, 1.0, -1.0)
        sv = numpy.where(wvm <= numpy.pi, 1.0, -1.0)
        u = umin + Du * numpy.sin(chiu / 2.0) ** 2.0
        v = vmin + Dv * numpy.sin(chiv / 2.0) ** 2.0
        pu = su * numpy.sqrt(numpy.clip(self._Wu(u, E, Lz, I3), 0.0, None))
        pv = sv * numpy.sqrt(numpy.clip(self._Wv(v, E, Lz, I3), 0.0, None))
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
            self._OmegaR[ii],
            self._Omegaphi[ii],
            self._Omegaz[ii],
        )

    def _Freqs(self, jr, jphi, jz, **kwargs):
        if self._interp:
            sub = self._interp_instance(jr, jphi, jz)
            return (sub._OmegaR[0], sub._Omegaphi[0], sub._Omegaz[0])
        ii = self._torus_index(jr, jphi, jz)
        return (self._OmegaR[ii], self._Omegaphi[ii], self._Omegaz[ii])
