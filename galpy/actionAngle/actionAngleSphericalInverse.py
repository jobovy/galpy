###############################################################################
# actionAngleSphericalInverse.py: inverse action-angle transformation for
#   spherical potentials through a momentum-matched canonical map: each
#   torus is sampled exactly in its radial anomaly, mapped onto the torus of
#   equal (J_r, L) of a frozen isochrone auxiliary by the radial point
#   transformation that matches cumulative radial actions (a sine series in
#   the anomaly), cotangent-lifted, with the compensation of the radial and
#   polar angles for the moving turning points in closed form. The assembled
#   (J, theta) -> (x, v) map is exactly symplectic for ANY stored tables:
#   every derivative is the stored interpolant's own. (The spherical section
#   of canonical.tex in the fast-orbits repository.)
###############################################################################
import warnings

import numpy
from scipy.optimize import brentq, minimize

from ..potential import IsochronePotential, evaluatePotentials, rl, vcirc
from ..potential.Potential import _check_potential_list_and_deprecate
from ..util import conversion, galpyWarning
from .actionAngleInverse import actionAngleInverse
from .actionAngleIsochrone import actionAngleIsochrone
from .actionAngleIsochroneInverse import actionAngleIsochroneInverse


def _spec_coeffs(f):
    """True Fourier coefficients c_k of a real periodic function sampled on
    the regular offset grid tau_j = 2 pi (j + 1/2)/N, such that
    f(tau) = Re c_0 + sum_{k=1}^{N/2-1} 2 Re[c_k e^{i k tau}] +
    Re[c_{N/2} e^{i N tau / 2}]; the offset grid requires the phase
    correction e^{-i k pi / N} relative to the raw rfft"""
    N = len(f)
    k = numpy.arange(N // 2 + 1)
    return numpy.fft.rfft(f) / N * numpy.exp(-1j * k * numpy.pi / N)


def _spec_eval(c, tau, deriv=False):
    """Evaluate the Fourier series with coefficients c (from _spec_coeffs)
    or its derivative at arbitrary tau"""
    k = numpy.arange(len(c))
    w = numpy.ones(len(c))
    w[1:-1] = 2.0
    cc = c * (1j * k) if deriv else c
    ph = numpy.exp(1j * numpy.atleast_1d(tau)[:, None] * k[None, :])
    return numpy.real(ph @ (w * cc))


class actionAngleSphericalInverse(actionAngleInverse):
    """Inverse action-angle transformation for spherical potentials through a
    momentum-matched canonical map.

    Each torus is sampled exactly in its radial anomaly (turning points and
    momenta by direct quadrature) and mapped onto the torus of equal
    (J_r, L) of a frozen isochrone auxiliary by the radial point
    transformation that matches cumulative radial actions, a sine series in
    the anomaly, lifted to the momenta cotangent-consistently; the polar
    libration needs no map, because it does not depend on the potential.
    Evaluation reconstructs phase-space points through the analytic
    isochrone inverse and undoes the lift; the angles carry the closed-form
    compensation for the moving turning points. Canonicity is manifest: it
    holds for any stored table content, because every derivative the map
    needs is that of the stored interpolant itself.
    """

    def __init__(
        self,
        pot=None,
        Es=[0.5, 1.0],
        Ls=[0.9, 1.1],
        mm_npt=32,
        mm_nta=256,
        maxiter=100,
        angle_tol=1e-12,
        **kwargs,
    ):
        """
        Initialize an actionAngleSphericalInverse object.

        Parameters
        ----------
        pot : Potential or list thereof
            A spherical potential.
        Es, Ls : array-like
            Energies and angular momenta of the tori to set up (paired
            lists).
        mm_npt : int, optional
            Number of harmonics of the momentum-matched anomaly map; the
            reconstruction converges spectrally in this, and a warning is
            raised when it does not suffice for a torus.
        mm_nta : int, optional
            Number of anomaly samples per torus (even), used to sample the
            torus, to fit the map, and for the quadratures of its action,
            frequencies, and angles; must exceed 4 * mm_npt for the samples
            to resolve the map's highest harmonic, which is 2 * mm_npt.
        maxiter : int, optional
            Maximum Newton iterations of the angle solves.
        angle_tol : float, optional
            Convergence tolerance of the angle solves.

        Notes
        -----
        - 2026-08-25 - Started - Bovy (UofT)
        """
        actionAngleInverse.__init__(self, *[], **kwargs)
        if pot is None:
            raise OSError("Must specify pot= for actionAngleSphericalInverse")
        self._pot = _check_potential_list_and_deprecate(pot)
        if mm_nta % 2 == 1:
            raise ValueError("mm_nta has to be even")
        if mm_nta <= 4 * mm_npt:
            # the map's highest harmonic is 2 mm_npt, which mm_nta uniform
            # samples only resolve below their Nyquist harmonic mm_nta / 2
            raise ValueError(
                "mm_nta must exceed 4 * mm_npt for the anomaly samples to resolve the map's harmonics"
            )
        self._ntau = mm_nta
        self._npt = mm_npt
        self._nforDm = numpy.arange(1, mm_npt + 1)
        self._maxiter = maxiter
        self._angle_tol = angle_tol
        self._Es = conversion._parse_grid_quantity(
            Es, conversion.parse_energy, vo=self._vo
        ).astype("float")
        self._Ls = conversion._parse_grid_quantity(
            Ls, conversion.parse_angmom, ro=self._ro, vo=self._vo
        ).astype("float")
        if len(self._Es) != len(self._Ls):
            raise ValueError("Es and Ls have to have the same length")
        # sample every torus once (exact placement), then choose the frozen
        # auxiliary (all torus-dependence beyond it lives in the
        # momentum-matched map, whose compensation is closed-form), then
        # compute the tables against it
        self._sample_all()
        self._setup_toy()
        self._setup_tori()
        self._check_consistent_units()
        return None

    # ---------- node construction: exact radial tori by quadrature
    def _Phi(self, r):
        return evaluatePotentials(self._pot, r, 0.0, use_physical=False)

    def _turning_points(self, E, L):
        """Radial turning points of the (E, L) torus"""
        rc = rl(self._pot, L, use_physical=False)
        pr2 = lambda r: 2.0 * (E - self._Phi(r)) - L**2 / r**2
        if pr2(rc) < 0.0:
            raise ValueError(
                f"No orbit exists at E = {E}, L = {L}: the energy lies below "
                "the circular orbit's"
            )
        ttol = 1e-12
        rlo, rhi = rc, rc
        while pr2(rlo) > 0.0 and rlo > 1e-12:
            rlo /= 1.3
        while pr2(rhi) > 0.0 and rhi < 1e12:
            rhi *= 1.3
        rp = rc if pr2(rc * (1.0 - 1e-14)) <= 0.0 else brentq(pr2, rlo, rc, xtol=ttol)
        ra = rc if pr2(rc * (1.0 + 1e-14)) <= 0.0 else brentq(pr2, rc, rhi, xtol=ttol)
        if ra - rp < 1e-10 * rc:
            raise ValueError(
                f"The (E, L) = ({E}, {L}) torus is (numerically) circular, "
                "which the discrete torus construction does not support; "
                "the interpolation grid handles J_r -> 0 through its "
                "circular edge"
            )
        return rp, ra

    def _sample_torus(self, E, L):
        """Exact phase-space samples along the radial loop, parametrized by
        the tau anomaly; placement is exact by construction (p_r from the
        energy relation, not from a fit)"""
        rp, ra = self._turning_points(E, L)
        tau = 2.0 * numpy.pi * (numpy.arange(self._ntau) + 0.5) / self._ntau
        r = 0.5 * (ra + rp) - 0.5 * (ra - rp) * numpy.cos(tau)
        pr2 = 2.0 * (E - self._Phi(r)) - L**2 / r**2
        pr2[pr2 < 0.0] = 0.0
        pr = numpy.where(tau < numpy.pi, 1.0, -1.0) * numpy.sqrt(pr2)
        return tau, r, pr, rp, ra

    # ---------- the toy
    def _sample_all(self):
        self._samples = []
        for E, L in zip(self._Es, self._Ls):
            self._samples.append(self._sample_torus(E, L) + (E, L))
        return None

    def _setup_toy(self):
        """The frozen isochrone auxiliary of the whole family.

        The momentum-matched map lifts every torus onto its equal-action
        auxiliary torus, so the auxiliary's only job is to be a
        well-conditioned global surrogate of the target: its rotation curve
        is fitted to the target's over the sampled radial range (two
        parameters, starting from the isochrone whose circular radius and
        frequency ratio match the central torus's). Every node's truncated
        lift is then required to clear escape by a fraction of its own
        auxiliary torus's binding energy, which fails only when the stored
        anomaly map is under-resolved."""
        imid = len(self._Es) // 2
        E, L = self._Es[imid], self._Ls[imid]
        tau, r, pr, rp, ra = self._sample_torus(E, L)
        # frequency ratio of the central torus by regular quadrature in tau:
        # dt/dtau = (dr/dtau)/p_r is periodic and finite (dr/dtau and p_r
        # vanish together at the turning points)
        dtdtau = (
            0.5
            * (ra - rp)
            * numpy.fabs(numpy.sin(tau))
            / numpy.maximum(numpy.fabs(pr), 1e-300)
        )
        Ompsi_over_OmR = numpy.mean(L / r**2 * dtdtau) / numpy.mean(dtdtau)
        rho = min(max(Ompsi_over_OmR, 0.501), 0.999)
        rc = rl(self._pot, L, use_physical=False)

        def _GM_rc_pinned(b):
            # closed-form circular condition of the isochrone: the toy's
            # circular radius at L equals rc exactly
            s = numpy.sqrt(b**2 + rc**2)
            return L**2 * s * (b + s) ** 2 / rc**4

        GM = _GM_rc_pinned(1e-8)
        for _ in range(200):
            b = max(((L / (2.0 * rho - 1.0)) ** 2 - L**2) / (4.0 * GM), 1e-8)
            GMn = _GM_rc_pinned(b)
            if abs(GMn - GM) < 1e-14 * (1.0 + GM):
                break
            GM = GMn
        # fit the isochrone's rotation curve to the target's over the
        # sampled radial range (zero-point free, two parameters)
        rlo = min(smp[3] for smp in self._samples)
        rhi = max(smp[4] for smp in self._samples)
        rf = numpy.geomspace(rlo, rhi, 25)
        lnvc2 = numpy.log(vcirc(self._pot, rf, use_physical=False) ** 2)

        def _vc2cost(x):
            GMf, bf = numpy.exp(x)
            sf = numpy.sqrt(bf**2 + rf**2)
            return numpy.sum(
                (numpy.log(GMf * rf**2 / (sf * (bf + sf) ** 2)) - lnvc2) ** 2
            )

        res = minimize(
            _vc2cost,
            numpy.log([GM, max(b, 1e-3 * numpy.sqrt(rlo * rhi))]),
            method="Nelder-Mead",
        )
        GM, b = numpy.exp(res.x)
        self._GM, self._b = GM, b
        self._ip = IsochronePotential(amp=GM, b=b)
        for stau, sr, spr, srp, sra, sE, sL in self._samples:
            Jrq, a, e, _, _, _, rA, pA = self._pt_match(stau, sr, spr, srp, sra, sL)
            EAs = numpy.max(
                0.5 * (pA**2 + sL**2 / rA**2) - GM / (b + numpy.sqrt(b**2 + rA**2))
            )
            if EAs >= 0.05 * self._iso_E_of_Jr(Jrq, sL):
                raise RuntimeError(
                    "The momentum-matched lift of the (E, L) = "
                    f"({sE}, {sL}) torus is not bound in the fitted "
                    "auxiliary: the torus reaches beyond the depth of the "
                    "single isochrone fitted to the family's radial range, or "
                    "the anomaly map is under-resolved (raise mm_npt and, with "
                    "it, mm_nta)"
                )
        self._aAI = actionAngleIsochrone(ip=self._ip)
        self._aAIinv = actionAngleIsochroneInverse(ip=self._ip)
        return None

    # ---------- the momentum-matched radial map, cotangent-lifted
    def _iso_E_of_Jr(self, Jr, L):
        """Energy of the toy torus with radial action Jr: the isochrone's
        closed form"""
        CA = 0.5 * (L + numpy.sqrt(L**2 + 4.0 * self._GM * self._b))
        return -(self._GM**2) / (2.0 * (Jr + CA) ** 2)

    def _toy_params(self, Jr, L):
        """Closed-form (a, e) of the equal-action reference toy torus"""
        EA = self._iso_E_of_Jr(Jr, L)
        a = -self._GM / (2.0 * EA) - self._b
        e = numpy.sqrt(1.0 + L**2 / (2.0 * EA * a**2))
        return a, e

    def _toy_profile(self, a, e, eta):
        """The reference toy torus's radius, momentum, and dr^A/deta at
        eccentric anomaly eta -- all closed forms of the isochrone"""
        b = self._b
        y = 1.0 - e * numpy.cos(eta)
        rA = a * numpy.sqrt(y * (y + 2.0 * b / a))
        pA = numpy.sqrt(self._GM / (a + b)) * a * e * numpy.sin(eta) / rA
        drAdeta = (
            a * e * numpy.sin(eta) * (y + b / a) / numpy.sqrt(y * (y + 2.0 * b / a))
        )
        return rA, pA, drAdeta

    def _pt_match(self, tau, r, pr, rp, ra, L):
        """The momentum-matched lift of one torus: match cumulative radial
        actions from pericenter, eta(tau) = A_A^{-1}(A_t(tau)); both
        cumulatives share the linear part J_r (the equal-action choice of
        the reference toy torus), so eta - tau is periodic and, by
        time-reversal parity, a pure sine series. The lift is rebuilt from
        the TRUNCATED stored map cotangent-consistently (p^A = pi' p_r), so
        the reconstruction is the one the stored tables define; how far the
        truncation is from the exact map shows as the variation of the
        auxiliary action along the lifted torus, which _node_tables
        reports."""
        k = numpy.fft.fftfreq(self._ntau, d=1.0 / self._ntau)

        def _antider(f):
            fh = numpy.fft.fft(f - numpy.mean(f))
            ah = numpy.zeros_like(fh)
            ah[1:] = fh[1:] / (1j * k[1:])
            return numpy.real(numpy.fft.ifft(ah))

        drdtau_s = 0.5 * (ra - rp) * numpy.sin(tau)
        Jrq = float(numpy.mean(pr * drdtau_s))
        a, e = self._toy_params(Jrq, L)
        ft = pr * drdtau_s  # target dA/dtau >= 0
        _, pA_eta, drAdeta_eta = self._toy_profile(a, e, tau)
        fA = pA_eta * drAdeta_eta  # toy dA/deta >= 0, on the eta grid
        mt, mA = numpy.mean(ft), numpy.mean(fA)
        scale = mt / mA
        qt_t = _antider(ft)
        At = mt * tau + qt_t - _spec_eval(_spec_coeffs(qt_t), 0.0)[0]
        qt_A = _antider(fA)
        qA0 = _spec_eval(_spec_coeffs(qt_A), 0.0)[0]
        cqA = _spec_coeffs(qt_A)
        cfA = _spec_coeffs(fA)
        # the matching is monotone; a residual left by the iteration shows
        # in the truncation diagnostic of _node_tables
        eta_s = numpy.array(tau)
        for _ in range(200):
            fres = scale * (mA * eta_s + _spec_eval(cqA, eta_s) - qA0) - At
            fp = numpy.maximum(scale * _spec_eval(cfA, eta_s), 1e-10 * mt)
            de = numpy.clip(-fres / fp, -0.5, 0.5)
            eta_s += de
            if numpy.max(numpy.fabs(fres)) < 1e-13 * max(mt, 1e-10):
                break
        smat = numpy.sin(tau[:, None] * self._nforDm[None, :])
        Dm = 2.0 * numpy.mean((eta_s - tau)[:, None] * smat, axis=0)
        # rebuild the truncated map and its cotangent lift
        etat = tau + smat @ Dm
        detadtau = 1.0 + numpy.cos(tau[:, None] * self._nforDm[None, :]) @ (
            self._nforDm * Dm
        )
        rA, _, drAdeta_t = self._toy_profile(a, e, etat)
        pA = pr * drdtau_s / (drAdeta_t * detadtau)
        return Jrq, a, e, Dm, etat, detadtau, rA, pA

    # ---------- the generating-function tables, computed (never fitted)
    def _node_tables(self, ii):
        """One node torus (the ii-th sampled): fit the momentum-matched map and
        lift the samples onto the auxiliary, and return the torus's action
        and frequencies by regular quadrature in the anomaly, the map's
        tables, the variation of the auxiliary action along the lifted torus
        (the map's truncation, which should be at round-off), and the
        anomaly-to-angle tables that the discrete evaluation path reads"""
        tau, r, pr, rp, ra, E, L = self._samples[ii]
        k = numpy.fft.fftfreq(self._ntau, d=1.0 / self._ntau)

        def _antider(f):
            fh = numpy.fft.fft(f - numpy.mean(f))
            ah = numpy.zeros_like(fh)
            ah[1:] = fh[1:] / (1j * k[1:])
            return numpy.real(numpy.fft.ifft(ah))

        Jrq, a, e, Dm, etat, detadtau, rA, pA = self._pt_match(tau, r, pr, rp, ra, L)
        with numpy.errstate(invalid="ignore"):
            # the samples are planar (j_z = 0), so the auxiliary's
            # inclination angles divide by zero; only o[0], o[6], and o[7]
            # are read, none of which involve the inclination
            o = self._aAI.actionsFreqsAngles(
                rA,
                pA,
                L / rA,
                numpy.zeros_like(rA),
                numpy.zeros_like(rA),
                numpy.zeros_like(rA),
            )
        JA = numpy.atleast_1d(o[0])
        thetaA = numpy.atleast_1d(o[6])
        # the truncated map's lift is not exactly the equal-action torus:
        # the auxiliary action varies along it by the truncation residual
        perr = float(numpy.amax(numpy.fabs(JA - Jrq)) / Jrq)
        P = numpy.unwrap(thetaA - tau + numpy.pi) - numpy.pi
        # the target's own angles by regular quadrature in tau: dt/dtau is
        # periodic and finite (dr/dtau and p_r vanish together at the
        # turning points), so spectral antiderivatives apply
        dtdtau = (
            0.5
            * (ra - rp)
            * numpy.fabs(numpy.sin(tau))
            / numpy.maximum(numpy.fabs(pr), 1e-300)
        )
        Tr = 2.0 * numpy.pi * numpy.mean(dtdtau)
        OmR = 2.0 * numpy.pi / Tr
        gpsi = L / r**2 * dtdtau
        Ompsi = numpy.mean(gpsi) / numpy.mean(dtdtau)
        qt = _antider(dtdtau)
        spsi = _antider(gpsi)
        # fix the angle origins at pericenter (tau = 0): theta_r(0) = 0 and
        # chi(0) = 0, evaluating the periodic antiderivatives there spectrally
        qt0 = _spec_eval(_spec_coeffs(qt), 0.0)[0]
        spsi0 = _spec_eval(_spec_coeffs(spsi), 0.0)[0]
        Pt = OmR * (qt - qt0)  # theta_r(tau) = tau + Pt(tau)
        chi = Ompsi * (qt - qt0) - (spsi - spsi0)  # theta_psi at psi = 0
        # the psi-angle shift along the torus (theta^A_z = theta_z - Dpsi,
        # i.e. Dpsi = theta_psi - theta^A_psi): the samples sit at azimuth
        # zero, where the auxiliary's in-plane angle is o[7] directly
        Dpsi = numpy.unwrap(chi - numpy.atleast_1d(o[7]))
        return {
            "jr": Jrq,
            "perr": perr,
            "rp": rp,
            "ra": ra,
            "Dm": Dm,
            "OmR": OmR,
            "Ompsi": Ompsi,
            "cP": _spec_coeffs(P),
            "cPt": _spec_coeffs(Pt),
            "cD": _spec_coeffs(Dpsi),
        }

    def _warn_unresolved(self, perr, Es, Ls):
        """Warn when the truncated map leaves the auxiliary action varying
        along a lifted torus by more than round-off allows"""
        bad = numpy.asarray(perr) > 1e-6
        if numpy.any(bad):
            warnings.warn(
                "The momentum-matched anomaly map is not converged for the (E, L) tori: {} (maximum relative variation of the auxiliary action along a lifted torus {:.1e}); increase mm_npt and, with it, mm_nta".format(
                    ", ".join(
                        f"({E:g}, {L:g})"
                        for E, L in zip(numpy.asarray(Es)[bad], numpy.asarray(Ls)[bad])
                    ),
                    float(numpy.amax(perr)),
                ),
                galpyWarning,
            )
        return None

    def _setup_tori(self):
        ntori = len(self._Es)
        nk = self._ntau // 2 + 1
        self._jrs = numpy.empty(ntori)
        self._rps = numpy.empty(ntori)
        self._ras = numpy.empty(ntori)
        self._Dms = numpy.empty((ntori, self._npt))
        self._OmRs = numpy.empty(ntori)
        self._Ompsis = numpy.empty(ntori)
        self._cP = numpy.empty((ntori, nk), dtype=complex)
        self._cPt = numpy.empty((ntori, nk), dtype=complex)
        self._cD = numpy.empty((ntori, nk), dtype=complex)
        perr = numpy.empty(ntori)
        for ii in range(ntori):
            node = self._node_tables(ii)
            self._jrs[ii] = node["jr"]
            self._rps[ii] = node["rp"]
            self._ras[ii] = node["ra"]
            self._Dms[ii] = node["Dm"]
            self._OmRs[ii] = node["OmR"]
            self._Ompsis[ii] = node["Ompsi"]
            self._cP[ii] = node["cP"]
            self._cPt[ii] = node["cPt"]
            self._cD[ii] = node["cD"]
            perr[ii] = node["perr"]
        self._warn_unresolved(perr, self._Es, self._Ls)
        return None

    # ---------- the (E, L) interpolation grid
    # ---------- evaluation: the manifest chain
    def _kernel(self, tau, a, e, Dm, rp, ra):
        """Everything the evaluation needs at anomaly tau, in one pass and
        from the tables alone: the auxiliary anomaly eta(tau) and radial
        angle theta^A_r (the isochrone's mean-anomaly relation, closed
        form), and the target's radius and radial momentum (the flux
        identity p_r dr/dtau = p^A dr^A/dtau, with sin(eta)/sin(tau) grouped
        so that every factor is regular at the turning points). Also returns
        d theta^A_r / d tau for the Newton solve of the angle relation."""
        ms = self._nforDm
        mt = tau[:, None] * ms[None, :]
        smt, cmt = numpy.sin(mt), numpy.cos(mt)
        eta = tau + smt @ Dm
        deta = 1.0 + cmt @ (ms * Dm)
        b = self._b
        se, ce = numpy.sin(eta), numpy.cos(eta)
        y = 1.0 - e * ce
        sq = numpy.sqrt(y * (y + 2.0 * b / a))
        rA = a * sq
        gA = a * e * (y + b / a) / sq  # dr^A/deta / sin(eta)
        pA = numpy.sqrt(self._GM / (a + b)) * a * e * se / rA
        kap = a * e / (a + b)
        thetaA = eta - kap * se
        dthetaA = (1.0 - kap * ce) * deta
        st, ct = numpy.sin(tau), numpy.cos(tau)
        r = rp * (1.0 + ct) / 2.0 + ra * (1.0 - ct) / 2.0
        # sin(eta)/sin(tau) -> eta'(tau) at the turning points
        sratio = numpy.where(
            numpy.fabs(st) > 1e-12, se / numpy.where(st == 0.0, 1.0, st), deta
        )
        pr = pA * gA * sratio * deta / (0.5 * (ra - rp))
        return eta, thetaA, dthetaA, r, pr

    def _unlift(self, out, r, pr):
        """Undo the lift: the auxiliary-chart reconstruction (R, v_R, v_T,
        z, v_z, phi) becomes the target's by replacing the radius and the
        radial velocity with the kernel's r and p_r; the position direction,
        the plane, and the azimuth are untouched, and the tangential speed
        rescales to keep |L| = r x v exact"""
        R, vR, vT, z, vz, phi = out
        rA = numpy.sqrt(R**2 + z**2)
        vth = (vR * z - vz * R) / rA
        scale = r / rA
        vth2 = vth / scale
        return (
            R * scale,
            pr * R / rA + vth2 * z / rA,
            vT / scale,
            z * scale,
            pr * z / rA - vth2 * R / rA,
            phi,
        )

    def _tau_solve(self, ii, thr):
        """Newton solve of theta_r = tau + Pt(tau) for the anomaly tau on
        the discrete node torus ii; theta_r(tau) is monotone"""
        x = numpy.array(thr, dtype="float")
        for _ in range(self._maxiter):
            f = x + _spec_eval(self._cPt[ii], x) - thr
            fp = 1.0 + _spec_eval(self._cPt[ii], x, deriv=True)
            dx = numpy.clip(-f / fp, -0.5, 0.5)
            x += dx
            if numpy.max(numpy.fabs(f)) < self._angle_tol:
                break
        else:
            raise RuntimeError("Newton's method for the anomaly did not converge")
        return x

    def _match_node(self, jr, L):
        """Locate the discrete node torus with actions (J_r, L)"""
        dev = numpy.fabs(self._jrs - jr) + numpy.fabs(self._Ls - L)
        ii = numpy.argmin(dev)
        if dev[ii] > 1e-8 * (1.0 + numpy.fabs(jr) + numpy.fabs(L)):
            raise ValueError(f"(J_r, L) = ({jr}, {L}) is not one of the set-up tori")
        return ii

    # ---------- the public inverse map
    def _evaluate(self, jr, jphi, jz, angler, anglephi, anglez, **kwargs):
        return self._xvFreqs(jr, jphi, jz, angler, anglephi, anglez, **kwargs)[:6]

    @conversion.physical_conversion("action", pop=True)
    def Jr(self, E, L, **kwargs):
        """
        Return the radial action of the torus with energy E and angular momentum L.

        Parameters
        ----------
        E : float or Quantity
            Energy of the torus.
        L : float or Quantity
            Total angular momentum of the torus.

        Returns
        -------
        float or Quantity
            Radial action J_r of the torus.

        Notes
        -----
        - (E, L) must be one of the set-up tori.
        - 2026-09-18 - Written - Bovy (UofT)
        """
        E = conversion.parse_energy(E, vo=self._vo)
        L = conversion.parse_angmom(L, ro=self._ro, vo=self._vo)
        dev = numpy.fabs(self._Es - E) + numpy.fabs(self._Ls - L)
        ii = numpy.argmin(dev)
        if dev[ii] > 1e-10 * (1.0 + numpy.fabs(E) + numpy.fabs(L)):
            raise ValueError(f"(E, L) = ({E}, {L}) is not one of the set-up tori")
        return self._jrs[ii]

    def _xvFreqs(self, jr, jphi, jz, angler, anglephi, anglez, **kwargs):
        """(J, theta) -> (x, v): solve for the anomaly of each requested
        radial angle, shift the two companion angles by the map's
        L-compensation, delegate the full 3-D reconstruction to the analytic
        isochrone inverse (L^A = L, so the plane geometry is the auxiliary's
        own), and undo the lift"""
        jr, jphi, jz = float(jr), float(jphi), float(jz)
        L = jz + numpy.fabs(jphi)
        angler, anglephi, anglez = numpy.broadcast_arrays(
            numpy.atleast_1d(angler).astype(float),
            numpy.atleast_1d(anglephi).astype(float),
            numpy.atleast_1d(anglez).astype(float),
        )
        thr = angler % (2.0 * numpy.pi)
        ii = self._match_node(jr, L)
        OmR, OmL = self._OmRs[ii], self._Ompsis[ii]
        taus = self._tau_solve(ii, thr)
        thetaAr = taus + _spec_eval(self._cP[ii], taus)
        Delta = _spec_eval(self._cD[ii], taus)
        a, e = self._toy_params(self._jrs[ii], L)
        _, _, _, r, pr = self._kernel(
            taus, a, e, self._Dms[ii], self._rps[ii], self._ras[ii]
        )
        thetaAz = anglez - Delta
        thetaAphi = anglephi - numpy.sign(jphi) * Delta
        # the three-dimensional reconstruction on the auxiliary torus, in one
        # vectorized call (the auxiliary has the requested actions), then
        # the lift undone with the kernel's radius and radial momentum
        out = self._unlift(
            self._aAIinv._xvFreqs(jr, jphi, jz, thetaAr, thetaAphi, thetaAz)[:6],
            r,
            pr,
        )
        return (
            out[0],
            out[1],
            out[2],
            out[3],
            out[4],
            out[5],
            OmR,
            numpy.sign(jphi) * OmL,
            OmL,
        )

    def _Freqs(self, jr, jphi, jz, **kwargs):
        """Frequencies of the (J_r, L) torus: the node's own, by regular
        quadrature in the anomaly"""
        jr, jphi, jz = float(jr), float(jphi), float(jz)
        L = jz + numpy.fabs(jphi)
        ii = self._match_node(jr, L)
        OmR, OmL = self._OmRs[ii], self._Ompsis[ii]
        return (OmR, numpy.sign(jphi) * OmL, OmL)
