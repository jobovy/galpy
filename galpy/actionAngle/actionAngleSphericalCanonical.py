###############################################################################
# actionAngleSphericalCanonical.py: inverse action-angle transformation for
#   spherical potentials, built the Staeckel-canonical way: exact node tori
#   by direct radial quadrature (no auxiliary torus in the construction),
#   lifted onto a manifestly canonical Fourier transform against an
#   isochrone toy in the target's own coordinates. The assembled
#   (J, theta) -> (x, v) map is exactly symplectic for ANY stored tables:
#   every derivative is the stored interpolant's own, and every toy-
#   parameter chain is compensated in closed form.
#   (STAECKEL_CANONICAL_MATH.md section 9 in the fast-orbits repository.)
###############################################################################
import numpy
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import brentq, minimize

from ..potential import IsochronePotential, evaluatePotentials, rl, vcirc
from ..potential.Potential import _check_potential_list_and_deprecate
from ..util import conversion
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


class actionAngleSphericalCanonical(actionAngleInverse):
    """Canonical inverse action-angle transformation for spherical potentials.

    Node tori are exact (radial turning points and momenta by direct
    quadrature); the generating function to an isochrone toy is COMPUTED
    from the pointwise closed-form correspondence (the toy shares the
    target's coordinates, so L^A = L identically and the coefficient
    lattice is purely radial); evaluation reconstructs phase-space points
    through the analytic isochrone inverse. Canonicity is manifest: it
    holds for any stored table content.
    """

    def __init__(
        self,
        pot=None,
        Es=[0.5, 1.0],
        Ls=[0.9, 1.1],
        setup_interp=False,
        pt=True,
        Rmin=0.5,
        Rmax=2.0,
        Rinf=25.0,
        nE=16,
        nL=16,
        ntau=256,
        nn=16,
        npt=None,
        maxiter=100,
        angle_tol=1e-12,
        **kwargs,
    ):
        """
        Initialize an actionAngleSphericalCanonical object.

        Parameters
        ----------
        pot : Potential or list thereof
            A spherical potential.
        Es, Ls : array-like
            Energies and angular momenta of the tori to set up when
            setup_interp is False (paired lists).
        setup_interp : bool, optional
            If True, set up an (E, L) grid of tori spanning the circular
            angular momenta of [Rmin, Rmax] and energies up to the
            potential at Rinf, and interpolate canonically between them.
        pt : bool, optional
            If True (the default), align each torus with the toy through
            the momentum-matched radial point transformation: the map to
            the equal-action toy torus that matches cumulative radial
            actions, cotangent-lifted, so canonical for any stored
            tables. The generating-function lattice collapses to the
            stored map's truncation residual at every eccentricity, and
            the frozen toy's winding and boundness conditions become
            structural. If False, use the bare frozen-toy correspondence.
        Rmin, Rmax, Rinf : float, optional
            Radial anchors of the interpolation grid.
        nE, nL, ntau, nn, npt : int, optional
            Grid sizes: energies, angular momenta, torus-sampling points
            (even), the number of generating-function Fourier
            coefficients, and the number of sine modes of the stored
            point-transformation map (default: 32, capped at
            ntau/2 - 1; raise it together with ntau for very eccentric
            tori).
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
            raise OSError("Must specify pot= for actionAngleSphericalCanonical")
        self._pot = _check_potential_list_and_deprecate(pot)
        if ntau % 2 == 1:
            raise ValueError("ntau has to be even")
        if npt is None:
            npt = min(32, ntau // 2 - 1)
        elif npt > ntau // 2 - 1:
            raise ValueError("npt has to be at most ntau/2 - 1")
        self._ntau = ntau
        self._nn = nn
        self._npt = npt
        self._nforDm = numpy.arange(1, npt + 1)
        self._nforSn = numpy.arange(1, nn + 1)
        self._maxiter = maxiter
        self._angle_tol = angle_tol
        self._interp = setup_interp
        self._pt = pt
        if not setup_interp:
            self._Es = conversion._parse_grid_quantity(
                Es, conversion.parse_energy, vo=self._vo
            ).astype("float")
            self._Ls = conversion._parse_grid_quantity(
                Ls, conversion.parse_angmom, ro=self._ro, vo=self._vo
            ).astype("float")
            if len(self._Es) != len(self._Ls):
                raise ValueError("Es and Ls have to have the same length")
        else:
            self._setup_grid(
                conversion.parse_length(Rmin, ro=self._ro),
                conversion.parse_length(Rmax, ro=self._ro),
                conversion.parse_length(Rinf, ro=self._ro),
                nE,
                nL,
            )
        # sample every torus once (exact placement), then choose the frozen
        # toy (all torus-dependence beyond it lives in the support-matched
        # PT family, whose compensation is closed-form), then compute the
        # tables against it
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
        """Frozen isochrone toy for the whole set of tori. Selection at the
        central torus: the toy's circular radius at the central angular
        momentum is pinned to the target's (the winding condition: the
        correspondence tau -> theta^A only winds once per radial period when
        the toy's circular point lies inside the target loop in the
        (r, p_r) plane, and near-circular tori enclose only the immediate
        neighborhood of their own circular point), and b comes from the
        frequency-ratio match (the notes' rule; solvable for spherical
        targets). The frequency MAGNITUDE is left free -- the correspondence
        absorbs it exactly. If some sampled point is unbound in the toy,
        escalate along the family that keeps the circular radius pinned
        (b and GM together; the well depth GM/2b grows without bound), and
        verify the winding condition for every torus."""
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
        if self._pt:
            # the momentum-matched PT lifts every torus onto its
            # equal-action toy torus, so winding and boundness are
            # structural and per-torus alignment is the map's job; the
            # frozen toy's only job is to be a well-conditioned GLOBAL
            # surrogate of the target. Fit the isochrone's rotation curve
            # to the target's over the sampled radial range (zero-point
            # free, two parameters); then validate that every node's
            # truncated lift clears escape by a fraction of its own
            # reference torus's binding energy, which fails only when the
            # stored anomaly map is under-resolved.
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
                        "toy: the stored anomaly map is under-resolved for "
                        "this torus; raise npt and/or ntau"
                    )
            self._aAI = actionAngleIsochrone(ip=self._ip)
            self._aAIinv = actionAngleIsochroneInverse(ip=self._ip)
            return None
        for _ in range(60):
            ip = IsochronePotential(amp=GM, b=b)
            # boundness: every sampled point of every torus bound in the toy
            EAmax = max(
                numpy.max(
                    0.5 * (spr**2 + sL**2 / sr**2)
                    + evaluatePotentials(ip, sr, 0.0, use_physical=False)
                )
                for (stau, sr, spr, srp, sra, sE, sL) in self._samples
            )
            # winding: the toy's circular point inside every target loop
            wind_ok = all(
                srp < rl(ip, sL, use_physical=False) < sra
                for (stau, sr, spr, srp, sra, sE, sL) in self._samples
            )
            if EAmax < -1e-8 and wind_ok:
                break
            b *= 1.2
            GM = _GM_rc_pinned(b)
        else:
            raise RuntimeError(
                "Could not find a frozen isochrone toy that keeps every "
                "sampled point bound and its circular radius inside every "
                "torus's radial range (the winding condition): the "
                "requested tori span too wide a range for a single "
                "isochrone (the varying toy addresses this)"
            )
        self._GM, self._b = GM, b
        self._ip = IsochronePotential(amp=GM, b=b)
        self._aAI = actionAngleIsochrone(ip=self._ip)
        self._aAIinv = actionAngleIsochroneInverse(ip=self._ip)
        return None

    # ---------- the radial alignment PT (support-matched, cotangent-lifted)
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

    def _toy_coseta(self, rA, a, e):
        """Invert the toy radius profile: from (a y + b)^2 = b^2 + r^A2,
        y = 1 - e cos(eta) follows in closed form"""
        y = (numpy.sqrt(self._b**2 + rA**2) - self._b) / a
        return numpy.clip((1.0 - y) / e, -1.0, 1.0)

    def _pt_match(self, tau, r, pr, rp, ra, L):
        """The momentum-matched lift of one torus: match cumulative radial
        actions from pericenter, eta(tau) = A_A^{-1}(A_t(tau)); both
        cumulatives share the linear part J_r (the equal-action choice of
        the reference toy torus), so eta - tau is periodic and, by
        time-reversal parity, a pure sine series. The lift is rebuilt from
        the TRUNCATED stored map cotangent-consistently (p^A = pi' p_r),
        so the S_n mop up exactly what the stored map misses and the
        construction stays exact at nodes for any npt."""
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
        eta_s = numpy.array(tau)
        for _ in range(self._maxiter):
            fres = scale * (mA * eta_s + _spec_eval(cqA, eta_s) - qA0) - At
            fp = numpy.maximum(scale * _spec_eval(cfA, eta_s), 1e-10 * mt)
            de = numpy.clip(-fres / fp, -0.5, 0.5)
            eta_s += de
            if numpy.max(numpy.fabs(fres)) < 1e-13 * max(mt, 1e-10):
                break
        else:
            raise RuntimeError(
                "Newton's method for the momentum-matching anomaly map did not converge"
            )
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

    def _toy_radial(self, JAr, L, thetaAr):
        """The radial half of the analytic isochrone inverse: (J^A_r, L,
        theta^A_r) -> (r^A, p^A_r), vectorized over points (each target
        point sits on its own toy torus)"""
        amp, bb = self._GM, self._b
        sqrtfourbkL2 = numpy.sqrt(L**2 + 4.0 * bb * amp)
        H = -2.0 * amp**2 / (2.0 * JAr + L + sqrtfourbkL2) ** 2
        a = -amp / 2.0 / H - bb
        ab = a + bb
        e = numpy.sqrt(1.0 + L**2 / (2.0 * H * a**2))
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
        return rA, pA

    # ---------- the generating-function tables, computed (never fitted)
    def _cached_sample(self, E, L):
        for smp in self._samples:
            if smp[5] == E and smp[6] == L:
                return smp
        return self._sample_torus(E, L) + (E, L)

    def _node_tables(self, E, L):
        """One node torus: sample exactly, run the closed-form isochrone
        correspondence (L^A = L pointwise, so only J^A_r fluctuates and the
        lattice is purely radial), and return the canonical action label
        (the gauge-independent zero mode), the sine coefficients of the
        generating function from the 1-D pullback, the frequencies by
        regular quadrature, and the per-torus correspondence tables that
        the discrete evaluation path reads"""
        tau, r, pr, rp, ra, _, _ = self._cached_sample(E, L)
        k = numpy.fft.fftfreq(self._ntau, d=1.0 / self._ntau)

        def _antider(f):
            fh = numpy.fft.fft(f - numpy.mean(f))
            ah = numpy.zeros_like(fh)
            ah[1:] = fh[1:] / (1j * k[1:])
            return numpy.real(numpy.fft.ifft(ah))

        if self._pt:
            Jrq, a, e, Dm, etat, detadtau, rA, pA = self._pt_match(
                tau, r, pr, rp, ra, L
            )
        else:
            Jrq, a, e = 0.0, 0.0, 0.0
            Dm = numpy.zeros(self._npt)
            rA, pA = r, pr
        with numpy.errstate(invalid="ignore"):
            # the samples are planar (j_z = 0), so the toy's inclination
            # angles divide by zero; only o[0], o[6], and o[7] are read,
            # none of which involve the inclination
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
        if numpy.any(~numpy.isfinite(JA)):
            # the toy-selection margins prevent this in any __init__ path;
            # guards against inconsistent externally-modified state
            raise RuntimeError(
                f"The toy correspondence failed for the (E, L) = ({E}, {L}) "
                "torus (unbound lifted samples)"
            )
        P = numpy.unwrap(thetaA - tau + numpy.pi) - numpy.pi
        k = numpy.fft.fftfreq(self._ntau, d=1.0 / self._ntau)
        dthdtau = 1.0 + numpy.real(numpy.fft.ifft(1j * k * numpy.fft.fft(P)))
        jr = float(numpy.mean(JA * dthdtau))
        g = (JA - jr) * dthdtau
        gh = numpy.fft.fft(g)
        sh = numpy.zeros(self._ntau, dtype=complex)
        sh[1:] = gh[1:] / (1j * k[1:])
        sigma = numpy.real(numpy.fft.ifft(sh))
        Sn = numpy.empty(self._nn)
        maxcos = 0.0
        for q, n in enumerate(self._nforSn):
            c = numpy.mean(sigma * numpy.exp(-1j * n * thetaA) * dthdtau)
            Sn[q] = -numpy.imag(c)
            maxcos = max(maxcos, abs(numpy.real(c)))
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
        # the psi-angle shift along the torus, stored with the sign of the
        # generating-function chain (theta^A_z = theta_z - Dpsi, i.e.
        # Dpsi = theta_psi - theta^A_psi): the samples sit at azimuth zero,
        # where the toy's in-plane angle is o[7] directly
        Dpsi = numpy.unwrap(chi - numpy.atleast_1d(o[7]))
        return {
            "jr": jr,
            "Sn": Sn,
            "maxcos": maxcos,
            "rp": rp,
            "ra": ra,
            "Jrq": Jrq,
            "Dm": Dm,
            "OmR": OmR,
            "Ompsi": Ompsi,
            "cP": _spec_coeffs(P),
            "cPt": _spec_coeffs(Pt),
            "cJ": _spec_coeffs(JA - jr),
            "cD": _spec_coeffs(Dpsi),
        }

    def _setup_tori(self):
        if self._interp:
            return self._setup_tori_interp()
        ntori = len(self._Es)
        nk = self._ntau // 2 + 1
        self._jrs = numpy.empty(ntori)
        self._Sn = numpy.empty((ntori, self._nn))
        self._rps = numpy.empty(ntori)
        self._ras = numpy.empty(ntori)
        self._Jrqs = numpy.empty(ntori)
        self._Dms = numpy.empty((ntori, self._npt))
        self._OmRs = numpy.empty(ntori)
        self._Ompsis = numpy.empty(ntori)
        self._cP = numpy.empty((ntori, nk), dtype=complex)
        self._cPt = numpy.empty((ntori, nk), dtype=complex)
        self._cJ = numpy.empty((ntori, nk), dtype=complex)
        self._cD = numpy.empty((ntori, nk), dtype=complex)
        self._coserr = 0.0
        for ii, (E, L) in enumerate(zip(self._Es, self._Ls)):
            node = self._node_tables(E, L)
            self._jrs[ii] = node["jr"]
            self._Sn[ii] = node["Sn"]
            self._rps[ii] = node["rp"]
            self._ras[ii] = node["ra"]
            self._Jrqs[ii] = node["Jrq"]
            self._Dms[ii] = node["Dm"]
            self._OmRs[ii] = node["OmR"]
            self._Ompsis[ii] = node["Ompsi"]
            self._cP[ii] = node["cP"]
            self._cPt[ii] = node["cPt"]
            self._cJ[ii] = node["cJ"]
            self._cD[ii] = node["cD"]
            self._coserr = max(self._coserr, node["maxcos"])
        return None

    # ---------- the (E, L) interpolation grid
    def _setup_grid(self, Rmin, Rmax, Rinf, nE, nL):
        """Rectangular grid in (u, L): L between the circular angular
        momenta of Rmin and Rmax; E = Ec(L) + [E(Rinf) - Ec(L)] u^2 with u
        uniform in (0, 1] -- quadratic energy spacing at the circular edge
        (the phase-2 rectification lesson)"""
        if nE < 4 or nL < 4:
            raise ValueError("setup_interp=True requires nE >= 4 and nL >= 4")
        Lmin = Rmin * vcirc(self._pot, Rmin, use_physical=False)
        Lmax = Rmax * vcirc(self._pot, Rmax, use_physical=False)
        self._Lgrid = numpy.linspace(Lmin, Lmax, nL)
        self._us = (numpy.arange(nE) + 1.0) / nE
        self._Emax = self._Phi(Rinf)
        self._Ecs = numpy.empty(nL)
        for jj, L in enumerate(self._Lgrid):
            rc = rl(self._pot, L, use_physical=False)
            self._Ecs[jj] = self._Phi(rc) + L**2 / (2.0 * rc**2)
        if numpy.any(self._Ecs >= self._Emax):
            raise ValueError(
                "Rinf is too small: the grid's top energy lies below a "
                "circular orbit's; increase Rinf"
            )
        Etab = self._Ecs[None, :] + (self._Emax - self._Ecs[None, :]) * (
            self._us[:, None] ** 2
        )
        self._E_tab = Etab
        self._Es = Etab.flatten()
        self._Ls = numpy.tile(self._Lgrid, nE)
        return None

    def _setup_tori_interp(self):
        nu, nLg = len(self._us), len(self._Lgrid)
        self._jr_tab = numpy.empty((nu, nLg))
        self._Sn_tab = numpy.empty((nu, nLg, self._nn))
        self._OmR_tab = numpy.empty((nu, nLg))
        self._Ompsi_tab = numpy.empty((nu, nLg))
        self._sup_tab = numpy.empty((nu, nLg, 2))  # rp, ra
        self._Jrq_tab = numpy.empty((nu, nLg))
        self._Dm_tab = numpy.empty((nu, nLg, self._npt))
        self._coserr = 0.0
        for ii in range(nu):
            for jj in range(nLg):
                node = self._node_tables(self._E_tab[ii, jj], self._Lgrid[jj])
                self._jr_tab[ii, jj] = node["jr"]
                self._Sn_tab[ii, jj] = node["Sn"]
                self._OmR_tab[ii, jj] = node["OmR"]
                self._Ompsi_tab[ii, jj] = node["Ompsi"]
                self._sup_tab[ii, jj] = [node["rp"], node["ra"]]
                self._Jrq_tab[ii, jj] = node["Jrq"]
                self._Dm_tab[ii, jj] = node["Dm"]
                self._coserr = max(self._coserr, node["maxcos"])
        self._rebuild_interp()
        return None

    def _rebuild_interp(self):
        """(Re)build the spline interpolants from the stored tables; kept
        separate so that table perturbations (e.g. the noise-injection
        manifest test) re-enter through exactly this call"""
        u, Lg = self._us, self._Lgrid
        self._jr_ip = RectBivariateSpline(u, Lg, self._jr_tab, kx=3, ky=3, s=0.0)
        self._E_ip = RectBivariateSpline(u, Lg, self._E_tab, kx=3, ky=3, s=0.0)
        self._Sn_ip = [
            RectBivariateSpline(u, Lg, self._Sn_tab[:, :, q], kx=3, ky=3, s=0.0)
            for q in range(self._nn)
        ]
        self._sup_ip = [
            RectBivariateSpline(u, Lg, self._sup_tab[:, :, q], kx=3, ky=3, s=0.0)
            for q in range(2)
        ]
        self._Jrq_ip = RectBivariateSpline(u, Lg, self._Jrq_tab, kx=3, ky=3, s=0.0)
        self._Dm_ip = [
            RectBivariateSpline(u, Lg, self._Dm_tab[:, :, q], kx=3, ky=3, s=0.0)
            for q in range(self._npt)
        ]
        return None

    # ---------- evaluation: the manifest chain
    def _interp_tables(self, jr, L):
        """Solve the implicit inverse label (u from (J_r, L) by root-finding
        on the stored J_r interpolant -- exact-in-the-family, so canonicity
        is untouched) and return the interpolants' own values and
        derivatives, combined into the chains the evaluation needs"""
        if L < self._Lgrid[0] or L > self._Lgrid[-1]:
            raise ValueError(
                f"L = {L} outside the interpolation grid "
                f"[{self._Lgrid[0]}, {self._Lgrid[-1]}]"
            )
        jlo = self._jr_ip(self._us[0], L)[0, 0]
        jhi = self._jr_ip(self._us[-1], L)[0, 0]
        if jr < jlo or jr > jhi:
            raise ValueError(
                f"J_r = {jr} outside the interpolated family's range "
                f"[{jlo}, {jhi}] at L = {L}"
            )
        u = brentq(
            lambda uu: self._jr_ip(uu, L)[0, 0] - jr,
            self._us[0],
            self._us[-1],
            xtol=1e-14,
        )
        djr_du = self._jr_ip(u, L, dx=1)[0, 0]
        djr_dL = self._jr_ip(u, L, dy=1)[0, 0]
        Sn = numpy.array([ip(u, L)[0, 0] for ip in self._Sn_ip])
        dSn_du = numpy.array([ip(u, L, dx=1)[0, 0] for ip in self._Sn_ip])
        dSn_dL = numpy.array([ip(u, L, dy=1)[0, 0] for ip in self._Sn_ip])
        dE_du = self._E_ip(u, L, dx=1)[0, 0]
        dE_dL = self._E_ip(u, L, dy=1)[0, 0]
        # chains at fixed L resp. fixed J_r, all from the stored
        # interpolants' own derivatives
        dSdJ = dSn_du / djr_du
        dSdL = dSn_dL - dSn_du * djr_dL / djr_du
        OmR = dE_du / djr_du
        OmL = dE_dL - dE_du * djr_dL / djr_du
        if not self._pt:
            return u, Sn, dSdJ, dSdL, OmR, OmL, None
        sup = numpy.array([ip(u, L)[0, 0] for ip in self._sup_ip])
        dsup_du = numpy.array([ip(u, L, dx=1)[0, 0] for ip in self._sup_ip])
        dsup_dL = numpy.array([ip(u, L, dy=1)[0, 0] for ip in self._sup_ip])
        Jrq = self._Jrq_ip(u, L)[0, 0]
        dJrq_du = self._Jrq_ip(u, L, dx=1)[0, 0]
        dJrq_dL = self._Jrq_ip(u, L, dy=1)[0, 0]
        Dm = numpy.array([ip(u, L)[0, 0] for ip in self._Dm_ip])
        dDm_du = numpy.array([ip(u, L, dx=1)[0, 0] for ip in self._Dm_ip])
        dDm_dL = numpy.array([ip(u, L, dy=1)[0, 0] for ip in self._Dm_ip])
        ptdata = {
            "L": L,
            "sup": sup,
            "dsupJ": dsup_du / djr_du,
            "dsupL": dsup_dL - dsup_du * djr_dL / djr_du,
            "Jrq": Jrq,
            "dJrqJ": dJrq_du / djr_du,
            "dJrqL": dJrq_dL - dJrq_du * djr_dL / djr_du,
            "Dm": Dm,
            "dDmJ": dDm_du / djr_du,
            "dDmL": dDm_dL - dDm_du * djr_dL / djr_du,
        }
        return u, Sn, dSdJ, dSdL, OmR, OmL, ptdata

    def _toy_param_chains(self, Jrq, L, dJrq, dLex):
        """(a, e) of the reference toy torus and their derivatives along a
        chain with dJrq = d Jrq/d alpha and dLex = dL/d alpha (0 or 1),
        all closed forms"""
        GM, b = self._GM, self._b
        sq = numpy.sqrt(L**2 + 4.0 * GM * b)
        CA = 0.5 * (L + sq)
        EA = -(GM**2) / (2.0 * (Jrq + CA) ** 2)
        dEA = GM**2 / (Jrq + CA) ** 3 * (dJrq + 0.5 * (1.0 + L / sq) * dLex)
        a = -GM / (2.0 * EA) - b
        da = GM / (2.0 * EA**2) * dEA
        e = numpy.sqrt(1.0 + L**2 / (2.0 * EA * a**2))
        de = (
            2.0 * L * dLex / (2.0 * EA * a**2)
            - L**2 * (dEA * a + 2.0 * EA * da) / (2.0 * EA**2 * a**3)
        ) / (2.0 * e)
        return a, e, da, de

    def _tau_of_eta(self, eta, Dm):
        """Invert the stored anomaly map eta = tau + sum_m D_m sin(m tau)
        (monotone) for tau, vectorized"""
        ms = self._nforDm
        x = numpy.array(eta, dtype="float")
        for _ in range(self._maxiter):
            f = x + numpy.sin(x[:, None] * ms[None, :]) @ Dm - eta
            fp = 1.0 + numpy.cos(x[:, None] * ms[None, :]) @ (ms * Dm)
            dx = numpy.clip(-f / fp, -0.5, 0.5)
            x += dx
            if numpy.max(numpy.fabs(f)) < self._angle_tol:
                break
        else:
            raise RuntimeError("Newton's method for the map anomaly did not converge")
        return x

    def _pt_comp(self, rA, pA, ptdata, dJrq, dsup, dDm, dLex):
        """The point transformation's compensation term along one chain:
        -p_r (d pi/d alpha)|_{r^A} = p^A (d r^A/d alpha)|_tau
        - p_r (d r/d alpha)|_tau, grouped through the action-flux identity
        p_r dr/dtau = p^A dr^A/dtau so every factor is regular at the
        turning points"""
        L, Dm = ptdata["L"], ptdata["Dm"]
        rp, ra = ptdata["sup"]
        drp, dra = dsup
        a, e, da, de = self._toy_param_chains(ptdata["Jrq"], L, dJrq, dLex)
        coseta = self._toy_coseta(rA, a, e)
        sineta = numpy.sign(pA) * numpy.sqrt(numpy.clip(1.0 - coseta**2, 0.0, None))
        eta = numpy.arctan2(sineta, coseta) % (2.0 * numpy.pi)
        tauv = self._tau_of_eta(eta, Dm)
        ms = self._nforDm
        s = numpy.sqrt(self._b**2 + rA**2)
        y = (s - self._b) / a
        # d r^A/d alpha at fixed tau: through (a, e) and the stored map
        drAdeta = (
            a * e * sineta * (y + self._b / a) / numpy.sqrt(y * (y + 2.0 * self._b / a))
        )
        drA = (
            y * s / rA * da
            - a * s * coseta / rA * de
            + drAdeta * (numpy.sin(tauv[:, None] * ms[None, :]) @ dDm)
        )
        # d r/d alpha at fixed tau, and p_r through the flux identity
        costau = numpy.cos(tauv)
        dr = drp * (1.0 + costau) / 2.0 + dra * (1.0 - costau) / 2.0
        detadtau = 1.0 + numpy.cos(tauv[:, None] * ms[None, :]) @ (ms * Dm)
        sintau = numpy.sin(tauv)
        # pi' = (dr/dtau)/(dr^A/dtau); p_r = p^A/pi'; group the sine ratio
        sratio = numpy.where(
            numpy.fabs(sintau) > 1e-12,
            sineta / numpy.maximum(numpy.fabs(sintau), 1e-12) * numpy.sign(sintau),
            1.0,
        )
        gA = a * e * (y + self._b / a) / numpy.sqrt(y * (y + 2.0 * self._b / a))
        pr = pA * gA * sratio * detadtau / (0.5 * (ra - rp))
        return pA * drA - pr * dr

    def _thetaA_solve(self, thr, dSdJ, comp=None):
        """Newton solve of theta_r = theta^A + 2 sum dS_n/dJ_r sin(n theta^A)
        [+ the PT-family compensation -p_r (dpi/dJ_r)|_{r^A}] for theta^A,
        vectorized over the requested angles; the residual is exact, the
        Jacobian approximates by dropping the compensation's derivative
        (the 1D-validated approach)"""
        n = self._nforSn

        def _residual(x, target):
            snx = numpy.sin(x[:, None] * n[None, :])
            cnx = numpy.cos(x[:, None] * n[None, :])
            f = x + 2.0 * snx @ dSdJ - target
            if comp is not None:
                jr, L, Sn, ptdata = comp
                JAr = jr + 2.0 * cnx @ (n * Sn)
                rA, pA = self._toy_radial(JAr, L, x)
                f = f + self._pt_comp(
                    rA,
                    pA,
                    ptdata,
                    ptdata["dJrqJ"],
                    ptdata["dsupJ"],
                    ptdata["dDmJ"],
                    0.0,
                )
            return f, cnx

        x = numpy.array(thr, dtype="float")
        for _ in range(self._maxiter):
            f, cnx = _residual(x, thr)
            fp = 1.0 + 2.0 * cnx @ (n * dSdJ)
            dx = -f / fp
            dx = numpy.clip(dx, -0.5, 0.5)
            x += dx
            if numpy.max(numpy.fabs(f)) < self._angle_tol:
                break
        else:
            # the approximate-Jacobian Newton can cycle on hard tori even
            # where the angle map is monotone; fall back to safeguarded
            # scalar root-finding per non-converged point. The residual is
            # x + Q(x) - theta with Q periodic, so [theta - maxQ - eps,
            # theta + maxQ + eps] is a guaranteed bracket. (Where the
            # family chart FOLDS -- measured beyond u ~ 0.8 on very wide
            # grids, the frozen toy's phase mismatch growing too fast
            # along the family -- the root is not unique, root-finding
            # returns one branch, and the round trip degrades; the
            # profile-matched PT is the recorded fix.)
            f, _ = _residual(x, thr)
            bad = numpy.flatnonzero(numpy.fabs(f) >= self._angle_tol)
            xscan = numpy.linspace(0.0, 2.0 * numpy.pi, 256, endpoint=False)
            s = (
                numpy.max(numpy.fabs(_residual(xscan, numpy.zeros(256))[0] - xscan))
                + 0.1
            )
            for ii in bad:
                thri = thr[ii]

                def _fi(xx):
                    return _residual(numpy.array([xx]), numpy.array([thri]))[0][0]

                x[ii] = brentq(_fi, thri - s, thri + s, xtol=1e-15, maxiter=200)
        return x

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
            raise ValueError(
                f"(J_r, L) = ({jr}, {L}) is not one of the set-up tori; "
                "discrete mode evaluates the stored tori only (use "
                "setup_interp=True to interpolate)"
            )
        return ii

    # ---------- the public inverse map
    def _evaluate(self, jr, jphi, jz, angler, anglephi, anglez, **kwargs):
        return self._xvFreqs(jr, jphi, jz, angler, anglephi, anglez, **kwargs)[:6]

    def _xvFreqs(self, jr, jphi, jz, angler, anglephi, anglez, **kwargs):
        """(J, theta) -> (x, v): solve the 1-D Newton for theta^A_r, shift
        the two companion angles by the lattice's L-chain, and delegate the
        full 3-D reconstruction to the analytic isochrone inverse (L^A = L,
        so the plane geometry is the toy's own)"""
        jr, jphi, jz = float(jr), float(jphi), float(jz)
        L = jz + numpy.fabs(jphi)
        angler, anglephi, anglez = numpy.broadcast_arrays(
            numpy.atleast_1d(angler).astype(float),
            numpy.atleast_1d(anglephi).astype(float),
            numpy.atleast_1d(anglez).astype(float),
        )
        thr = angler % (2.0 * numpy.pi)
        n = self._nforSn
        params = None
        if self._interp:
            u, Sn, dSdJ, dSdL, OmR, OmL, ptdata = self._interp_tables(jr, L)
            comp = None if not self._pt else (jr, L, Sn, ptdata)
            thetaAr = self._thetaA_solve(thr, dSdJ, comp=comp)
            snx = numpy.sin(thetaAr[:, None] * n[None, :])
            cnx = numpy.cos(thetaAr[:, None] * n[None, :])
            JAr = jr + 2.0 * cnx @ (n * Sn)
            Delta = 2.0 * snx @ dSdL
            if self._pt:
                # the PT family's L-chain compensates the psi-angles the
                # same way the J_r-chain compensates theta_r
                rA, pA = self._toy_radial(JAr, L, thetaAr)
                Delta = Delta + self._pt_comp(
                    rA,
                    pA,
                    ptdata,
                    ptdata["dJrqL"],
                    ptdata["dsupL"],
                    ptdata["dDmL"],
                    1.0,
                )
                a, e = self._toy_params(ptdata["Jrq"], L)
                params = (a, e, ptdata["Dm"], ptdata["sup"][0], ptdata["sup"][1])
        else:
            ii = self._match_node(jr, L)
            OmR, OmL = self._OmRs[ii], self._Ompsis[ii]
            taus = self._tau_solve(ii, thr)
            thetaAr = taus + _spec_eval(self._cP[ii], taus)
            JAr = jr + _spec_eval(self._cJ[ii], taus)
            Delta = _spec_eval(self._cD[ii], taus)
            if self._pt:
                a, e = self._toy_params(self._Jrqs[ii], L)
                params = (a, e, self._Dms[ii], self._rps[ii], self._ras[ii])
        thetaAz = anglez - Delta
        thetaAphi = anglephi - numpy.sign(jphi) * Delta
        out = numpy.empty((6, len(thr)))
        for ii in range(len(thr)):
            oo = self._aAIinv._xvFreqs(
                JAr[ii], jphi, jz, thetaAr[ii], thetaAphi[ii], thetaAz[ii]
            )
            for jj in range(6):
                out[jj, ii] = oo[jj][0]
        if self._pt:
            out = self._pt_unlift(out, params)
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

    def _pt_unlift(self, out, params):
        """Map the toy-chart reconstruction to the target chart: the toy
        anomaly from the closed-form radius inversion, the stored anomaly
        map, then the target's cosine form give the radius; the radial
        momentum maps through 1/pi' (grouped through the action-flux
        identity so every factor is regular at the turning points); the
        position direction, the plane, and the azimuth are untouched, and
        the tangential speed rescales to keep |L| = r x v exact"""
        a, e, Dm, rp, ra = params
        R, vR, vT, z, vz, phi = out
        rA = numpy.sqrt(R**2 + z**2)
        vr = (R * vR + z * vz) / rA
        vth = (vR * z - vz * R) / rA
        coseta = self._toy_coseta(rA, a, e)
        sineta = numpy.sign(vr) * numpy.sqrt(numpy.clip(1.0 - coseta**2, 0.0, None))
        eta = numpy.arctan2(sineta, coseta) % (2.0 * numpy.pi)
        tauv = self._tau_of_eta(eta, Dm)
        costau = numpy.cos(tauv)
        r = rp * (1.0 + costau) / 2.0 + ra * (1.0 - costau) / 2.0
        y = (numpy.sqrt(self._b**2 + rA**2) - self._b) / a
        gA = a * e * (y + self._b / a) / numpy.sqrt(y * (y + 2.0 * self._b / a))
        ms = self._nforDm
        detadtau = 1.0 + numpy.cos(tauv[:, None] * ms[None, :]) @ (ms * Dm)
        sintau = numpy.sin(tauv)
        sratio = numpy.where(
            numpy.fabs(sintau) > 1e-12,
            sineta / numpy.maximum(numpy.fabs(sintau), 1e-12) * numpy.sign(sintau),
            1.0,
        )
        prT = vr * gA * sratio * detadtau / (0.5 * (ra - rp))
        scale = r / rA
        vth2 = vth / scale
        return numpy.array(
            [
                R * scale,
                prT * R / rA + vth2 * z / rA,
                vT / scale,
                z * scale,
                prT * z / rA - vth2 * R / rA,
                phi,
            ]
        )

    def _Freqs(self, jr, jphi, jz, **kwargs):
        """Frequencies of the (J_r, L) torus: in interpolation mode these
        are the stored energy interpolant's own derivatives through the
        label chain (the integrator contract); in discrete mode the node
        quadrature values"""
        jr, jphi, jz = float(jr), float(jphi), float(jz)
        L = jz + numpy.fabs(jphi)
        if self._interp:
            res = self._interp_tables(jr, L)
            OmR, OmL = res[4], res[5]
        else:
            ii = self._match_node(jr, L)
            OmR, OmL = self._OmRs[ii], self._Ompsis[ii]
        return (OmR, numpy.sign(jphi) * OmL, OmL)
