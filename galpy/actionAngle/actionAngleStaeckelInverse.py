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
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import brentq

from ..potential import evaluatePotentials
from ..util import conversion
from .actionAngleInverse import actionAngleInverse

_GLX, _GLW = numpy.polynomial.legendre.leggauss(10)


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
        delta=None,
        Es=[0.5],
        Lzs=[0.5],
        I3s=[0.1],
        nchi=2001,
        uref=1.15,
        umax=10.0,
        nscan=800,
        maxiter=60,
        angle_tol=1e-13,
        **kwargs,
    ):
        """
        Initialize an actionAngleStaeckelInverse object.

        Parameters
        ----------
        pot : Potential or list thereof
            The potential; must be of exact Staeckel form in prolate
            spheroidal coordinates with focal distance delta (e.g.,
            KuzminKutuzovStaeckelPotential).
        delta : float
            Focal distance of the prolate spheroidal coordinate system.
        Es : list of float
            Energies of the tori to set up.
        Lzs : list of float
            z-components of the angular momentum of the tori.
        I3s : list of float
            Third integrals of the tori (in the convention
            p_u^2 = 2 delta^2 [E sinh^2 u - U(u) - I_3] - L_z^2/sinh^2 u
            with the gauge V(pi/2) = 0).
        nchi : int, optional
            Number of grid points in the chi anomaly for the stored
            profiles.
        uref : float, optional
            Reference u used in the Staeckel splitting of the potential.
        umax : float, optional
            Maximum u scanned for the u turning points.
        nscan : int, optional
            Number of points in the turning-point bracketing scans.
        maxiter : int, optional
            Maximum number of Newton iterations in the angle inversion.
        angle_tol : float, optional
            Convergence tolerance of the angle inversion.

        Notes
        -----
        - Angle conventions: theta_R = 0 at u = u_- moving outward;
          theta_z = 0 at v = v_- (the northern turning point) moving toward
          the midplane; theta_phi follows from dW/dL_z.
        - 2026-08-19 - Started - Bovy (UofT)
        """
        actionAngleInverse.__init__(self, *[], **kwargs)
        if pot is None:  # pragma: no cover
            raise OSError("Must specify pot= for actionAngleStaeckelInverse")
        if delta is None:  # pragma: no cover
            raise OSError("Must specify delta= for actionAngleStaeckelInverse")
        self._pot = pot
        self._delta = conversion.parse_length(delta, ro=self._ro)
        self._Es = numpy.atleast_1d(numpy.array(Es, dtype="float"))
        self._Lzs = numpy.atleast_1d(numpy.array(Lzs, dtype="float"))
        self._I3s = numpy.atleast_1d(numpy.array(I3s, dtype="float"))
        self._ntori = len(self._Es)
        self._nchi = nchi
        self._uref = uref
        self._umax = umax
        self._nscan = nscan
        self._maxiter = maxiter
        self._angle_tol = angle_tol
        self._chi = numpy.linspace(0.0, numpy.pi, nchi)
        # Staeckel splitting with the gauge V(pi/2) = 0
        self._Vref = self._Uofu(uref)
        # Per-torus setup
        self._umins = numpy.empty(self._ntori)
        self._umaxs = numpy.empty(self._ntori)
        self._vmins = numpy.empty(self._ntori)
        self._jr = numpy.empty(self._ntori)
        self._jz = numpy.empty(self._ntori)
        self._OmegaR = numpy.empty(self._ntori)
        self._Omegaz = numpy.empty(self._ntori)
        self._Omegaphi = numpy.empty(self._ntori)
        self._dEI3Lz_dJ = numpy.empty((self._ntori, 3, 3))
        self._Aprof = []  # per torus: 3 u-profile splines A_i(chi_u)
        self._Bprof = []  # per torus: 3 v-profile splines B_i(chi_v)
        self._dAprof = []
        self._dBprof = []
        for ii in range(self._ntori):
            self._setup_torus(ii)

    ############################ STAECKEL SPLITTING ###########################
    def _phi_uv(self, u, v):
        R = self._delta * numpy.sinh(u) * numpy.sin(v)
        z = self._delta * numpy.cosh(u) * numpy.cos(v)
        return evaluatePotentials(self._pot, R, z, use_physical=False)

    def _Uofu(self, u):
        return self._phi_uv(u, numpy.pi / 2.0) * (numpy.sinh(u) ** 2.0 + 1.0)

    def _Vofv(self, v):
        return self._Vref - self._phi_uv(self._uref, v) * (
            numpy.sinh(self._uref) ** 2.0 + numpy.sin(v) ** 2.0
        )

    def _Wu(self, u, E, Lz, I3):
        return (
            2.0 * self._delta**2.0 * (E * numpy.sinh(u) ** 2.0 - self._Uofu(u) - I3)
            - Lz**2.0 / numpy.sinh(u) ** 2.0
        )

    def _Wv(self, v, E, Lz, I3):
        return (
            2.0 * self._delta**2.0 * (E * numpy.sin(v) ** 2.0 + self._Vofv(v) + I3)
            - Lz**2.0 / numpy.sin(v) ** 2.0
        )

    ############################ PER-TORUS SETUP ##############################
    def _setup_torus(self, ii):
        E, Lz, I3 = self._Es[ii], self._Lzs[ii], self._I3s[ii]
        Wu = lambda u: self._Wu(u, E, Lz, I3)
        Wv = lambda v: self._Wv(v, E, Lz, I3)
        # turning points from bracketing scans
        us = numpy.linspace(1e-3, self._umax, self._nscan)
        pos = numpy.where(Wu(us) > 0.0)[0]
        if len(pos) == 0:
            raise ValueError(
                f"No bound u oscillation found for torus {ii} "
                "(E={E}, Lz={Lz}, I3={I3})"
            )
        ulo = us[pos[0] - 1] if pos[0] > 0 else 1e-8
        uhi = us[pos[-1] + 1] if pos[-1] < self._nscan - 1 else self._umax
        umin = brentq(Wu, ulo, us[pos[0]], xtol=1e-15, rtol=8.9e-16)
        umaxx = brentq(Wu, us[pos[-1]], uhi, xtol=1e-15, rtol=8.9e-16)
        vs = numpy.linspace(1e-3, numpy.pi / 2.0, self._nscan)
        if Wv(numpy.pi / 2.0) <= 0.0:
            raise ValueError(
                f"Midplane not reached for torus {ii}; no valid torus for "
                "(E={E}, Lz={Lz}, I3={I3})"
            )
        neg = numpy.where(Wv(vs) < 0.0)[0]
        vmin = (
            brentq(Wv, vs[neg[-1]], numpy.pi / 2.0, xtol=1e-15, rtol=8.9e-16)
            if len(neg) > 0
            else 1e-8
        )
        self._umins[ii], self._umaxs[ii], self._vmins[ii] = umin, umaxx, vmin
        Du, Dv = umaxx - umin, numpy.pi - 2.0 * vmin
        u_of_chi = lambda c: umin + Du * numpy.sin(c / 2.0) ** 2.0
        v_of_chi = lambda c: vmin + Dv * numpy.sin(c / 2.0) ** 2.0
        d2 = self._delta**2.0
        # actions: J = (1/pi) int p dq over the half (u) / full-half (v) loop
        self._jr[ii] = self._action_quad(Wu, u_of_chi, Du) / numpy.pi
        self._jz[ii] = self._action_quad(Wv, v_of_chi, Dv) / numpy.pi
        # 1/p profiles: cumulative int f(q)/p dq for f = delta^2 sinh^2/sin^2,
        # delta^2, and Lz/sinh^2 / Lz/sin^2
        PEu = self._cumprof(Wu, u_of_chi, Du, lambda q: d2 * numpy.sinh(q) ** 2.0)
        PIu = self._cumprof(Wu, u_of_chi, Du, lambda q: d2 * numpy.ones_like(q))
        PLu = self._cumprof(Wu, u_of_chi, Du, lambda q: Lz / numpy.sinh(q) ** 2.0)
        PEv = self._cumprof(Wv, v_of_chi, Dv, lambda q: d2 * numpy.sin(q) ** 2.0)
        PIv = self._cumprof(Wv, v_of_chi, Dv, lambda q: d2 * numpy.ones_like(q))
        PLv = self._cumprof(Wv, v_of_chi, Dv, lambda q: Lz / numpy.sin(q) ** 2.0)
        # period matrix: pi dJ_R = totEu dE - totIu dI3 - totLu dLz;
        #                pi dJ_z = totEv dE + totIv dI3 - totLv dLz; dJ_phi = dLz
        totEu, totIu, totLu = PEu[-1], PIu[-1], PLu[-1]
        totEv, totIv, totLv = PEv[-1], PIv[-1], PLv[-1]
        M = numpy.array(
            [
                [totEu / numpy.pi, -totIu / numpy.pi, -totLu / numpy.pi],
                [totEv / numpy.pi, totIv / numpy.pi, -totLv / numpy.pi],
                [0.0, 0.0, 1.0],
            ]
        )
        Minv = numpy.linalg.inv(M)  # d(E,I3,Lz)/d(J_R,J_z,J_phi)
        self._dEI3Lz_dJ[ii] = Minv
        self._OmegaR[ii], self._Omegaz[ii], self._Omegaphi[ii] = Minv[0]
        # angle profiles: theta_i = sum_X dX/dJ_i dW/dX with
        # dW/dE = PEu + PEv, dW/dI3 = -PIu + PIv, dW/dLz = -PLu - PLv (+phi)
        A = [
            InterpolatedUnivariateSpline(
                self._chi,
                Minv[0, i] * PEu - Minv[1, i] * PIu - Minv[2, i] * PLu,
                k=5,
            )
            for i in range(3)
        ]
        B = [
            InterpolatedUnivariateSpline(
                self._chi,
                Minv[0, i] * PEv + Minv[1, i] * PIv - Minv[2, i] * PLv,
                k=5,
            )
            for i in range(3)
        ]
        self._Aprof.append(A)
        self._Bprof.append(B)
        self._dAprof.append([a.derivative() for a in A])
        self._dBprof.append([b.derivative() for b in B])

    def _action_quad(self, W, q_of_chi, D):
        """int p dq over the loop as (D/2) int sqrt(W) sin chi dchi (regular)."""
        a, b = self._chi[:-1], self._chi[1:]
        mid, half = 0.5 * (a + b), 0.5 * (b - a)
        nodes = (mid[:, None] + half[:, None] * _GLX[None, :]).ravel()
        vals = (
            numpy.sqrt(numpy.clip(W(q_of_chi(nodes)), 0.0, None))
            * 0.5
            * D
            * numpy.sin(nodes)
        )
        return float(numpy.sum((half[:, None] * _GLW[None, :]).ravel() * vals))

    def _cumprof(self, W, q_of_chi, D, f):
        """Cumulative int f(q)/p dq = int f (D/2) sin chi / sqrt(W) dchi on the
        chi mesh; the integrand is regular including at the turning points."""
        a, b = self._chi[:-1], self._chi[1:]
        mid, half = 0.5 * (a + b), 0.5 * (b - a)
        nodes = (mid[:, None] + half[:, None] * _GLX[None, :]).ravel()
        q = q_of_chi(nodes)
        w = W(q)
        vals = numpy.empty_like(nodes)
        good = w > 0.0
        vals[good] = (
            f(q[good]) * (D / 2.0) * numpy.sin(nodes[good]) / numpy.sqrt(w[good])
        )
        vals[~good] = 0.0  # only possible through rounding exactly at the ends
        panels = (half[:, None] * _GLW[None, :] * vals.reshape(len(a), -1)).sum(axis=1)
        return numpy.concatenate([[0.0], numpy.cumsum(panels)])

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

    def _xvFreqs(self, jr, jphi, jz, angler, anglephi, anglez, **kwargs):
        ii = self._torus_index(jr, jphi, jz)
        A, B = self._Aprof[ii], self._Bprof[ii]
        dA, dB = self._dAprof[ii], self._dBprof[ii]
        thR = numpy.atleast_1d(numpy.array(angler, dtype="float"))
        thz = numpy.atleast_1d(numpy.array(anglez, dtype="float"))
        thphi = numpy.atleast_1d(numpy.array(anglephi, dtype="float"))
        # solve the additively separable 2x2 system for the extended phases
        wu, wv = numpy.copy(thR), numpy.copy(thz)
        for _ in range(self._maxiter):
            f0 = self._fold(A[0], wu) + self._fold(B[0], wv) - thR
            f1 = self._fold(A[1], wu) + self._fold(B[1], wv) - thz
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
        R = self._delta * sh * sn
        z = self._delta * ch * cs
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
        ii = self._torus_index(jr, jphi, jz)
        return (self._OmegaR[ii], self._Omegaphi[ii], self._Omegaz[ii])

    ############################ FORWARD (for tests) ##########################
    def _point_to_angles(self, R, vR, vT, z, vz, phi, indx):
        """Angles of a phase-space point on torus indx (internal, used to
        anchor traversal tests); returns (theta_R, theta_phi, theta_z)."""
        ii = indx
        d = self._delta
        d1 = numpy.sqrt(R**2.0 + (z + d) ** 2.0)
        d2 = numpy.sqrt(R**2.0 + (z - d) ** 2.0)
        u = numpy.arccosh(numpy.clip((d1 + d2) / (2.0 * d), 1.0, None))
        v = numpy.arccos(numpy.clip((d1 - d2) / (2.0 * d), -1.0, 1.0))
        pu = d * (vR * numpy.cosh(u) * numpy.sin(v) + vz * numpy.sinh(u) * numpy.cos(v))
        pv = d * (vR * numpy.sinh(u) * numpy.cos(v) - vz * numpy.cosh(u) * numpy.sin(v))
        umin, umaxx, vmin = self._umins[ii], self._umaxs[ii], self._vmins[ii]
        Du, Dv = umaxx - umin, numpy.pi - 2.0 * vmin
        chiu = 2.0 * numpy.arcsin(numpy.sqrt(numpy.clip((u - umin) / Du, 0.0, 1.0)))
        chiv = 2.0 * numpy.arcsin(numpy.sqrt(numpy.clip((v - vmin) / Dv, 0.0, 1.0)))
        wu = numpy.where(pu >= 0.0, chiu, 2.0 * numpy.pi - chiu)
        wv = numpy.where(pv >= 0.0, chiv, 2.0 * numpy.pi - chiv)
        A, B = self._Aprof[ii], self._Bprof[ii]
        thR = self._fold(A[0], wu) + self._fold(B[0], wv)
        thz = self._fold(A[1], wu) + self._fold(B[1], wv)
        thphi = phi + self._fold(A[2], wu) + self._fold(B[2], wv)
        return (
            thR % (2.0 * numpy.pi),
            thphi % (2.0 * numpy.pi),
            thz % (2.0 * numpy.pi),
        )
