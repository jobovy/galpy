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

from ..potential import OblateStaeckelWrapperPotential
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
            with the gauge V(pi/2) = 0).
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
        """Bracket and refine the turning points of all tori: a single
        vectorized scan over a dense mesh (one potential evaluation for all
        tori, because U and V are shared), refined per torus with brentq"""
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
            pos = numpy.where(Wu_all[ii] > 0.0)[0]
            if len(pos) == 0:
                raise ValueError(
                    f"No bound u oscillation found for torus {ii} "
                    f"(E={E}, Lz={Lz}, I3={I3})"
                )
            if pos[-1] == len(us) - 1:
                raise ValueError(
                    f"u oscillation not enclosed for torus {ii} "
                    f"(E={E}, Lz={Lz}, I3={I3}); the torus is likely unbound"
                )
            Wu = lambda u: self._Wu(u, E, Lz, I3)
            ulo = us[pos[0] - 1] if pos[0] > 0 else 1e-8
            self._umins[ii] = brentq(Wu, ulo, us[pos[0]], xtol=1e-15, rtol=8.9e-16)
            self._umaxs[ii] = brentq(
                Wu, us[pos[-1]], us[pos[-1] + 1], xtol=1e-15, rtol=8.9e-16
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
    def _compute_actions_frequencies_profiles(self):
        """Actions, the 3x3 period matrices, frequencies, and the cumulative
        angle profiles of all tori, all from two vectorized potential
        evaluations (one per degree of freedom): the momenta at the
        composite Gauss-Legendre nodes of the shared chi mesh give both the
        action integrands sqrt(W) sin(chi) and the 1/p profile integrands
        f(q) sin(chi)/sqrt(W)"""
        d2 = self._delta**2.0
        mid = 0.5 * (self._chi[:-1] + self._chi[1:])
        half = 0.5 * (self._chi[1:] - self._chi[:-1])
        nodes = (mid[:, None] + half[:, None] * _GLX[None, :]).ravel()
        sin_nodes = numpy.sin(nodes)
        y_nodes = numpy.sin(nodes / 2.0) ** 2.0
        npan, ngl = len(half), len(_GLX)

        def profiles(qmin, D, W_of_q, fEs, fIs, fLs):
            # q(chi) at the nodes for all tori: (ntori, nnodes)
            q = qmin[:, None] + D[:, None] * y_nodes[None, :]
            W = W_of_q(q)
            W[W < numpy.finfo(float).tiny] = numpy.finfo(float).tiny
            sqW = numpy.sqrt(W)
            # action: (D/2) int sqrt(W) sin(chi) dchi
            act_vals = sqW * (D[:, None] / 2.0) * sin_nodes[None, :]
            action = (
                (half[None, :, None] * _GLW[None, None, :])
                * act_vals.reshape(self._ntori, npan, ngl)
            ).sum(axis=(-1, -2))
            # cumulative 1/p profiles: int f(q) (D/2) sin(chi)/sqrt(W) dchi
            base = (D[:, None] / 2.0) * sin_nodes[None, :] / sqW
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
            self._umaxs - self._umins,
            lambda q: self._Wu(q, Es[:, None], Lzs[:, None], I3s[:, None]),
            lambda q: d2 * numpy.sinh(q) ** 2.0,
            lambda q: d2 * numpy.ones_like(q),
            lambda q: Lzs[:, None] / numpy.sinh(q) ** 2.0,
        )
        actv, (PEv, PIv, PLv) = profiles(
            self._vmins,
            numpy.pi - 2.0 * self._vmins,
            lambda q: self._Wv(q, Es[:, None], Lzs[:, None], I3s[:, None]),
            lambda q: d2 * numpy.sin(q) ** 2.0,
            lambda q: d2 * numpy.ones_like(q),
            lambda q: Lzs[:, None] / numpy.sin(q) ** 2.0,
        )
        self._jr = actu / numpy.pi
        self._jz = actv / numpy.pi
        # period matrices: pi dJ_R = PEu[-1] dE - PIu[-1] dI3 - PLu[-1] dLz;
        #                  pi dJ_z = PEv[-1] dE + PIv[-1] dI3 - PLv[-1] dLz
        M = numpy.zeros((self._ntori, 3, 3))
        M[:, 0, 0] = PEu[:, -1] / numpy.pi
        M[:, 0, 1] = -PIu[:, -1] / numpy.pi
        M[:, 0, 2] = -PLu[:, -1] / numpy.pi
        M[:, 1, 0] = PEv[:, -1] / numpy.pi
        M[:, 1, 1] = PIv[:, -1] / numpy.pi
        M[:, 1, 2] = -PLv[:, -1] / numpy.pi
        M[:, 2, 2] = 1.0
        self._dEI3Lz_dJ = numpy.linalg.inv(M)  # d(E,I3,Lz)/d(J_R,J_z,J_phi)
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
        ii = self._torus_index(jr, jphi, jz)
        return (self._OmegaR[ii], self._Omegaphi[ii], self._Omegaz[ii])
