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
from scipy.optimize import brentq

from ..potential import IsochronePotential, evaluatePotentials, rl, vcirc
from ..potential.Potential import _check_potential_list_and_deprecate
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
            the support-matched radial point transformation
            (cotangent-lifted, so canonical for any stored support
            tables); this shrinks the generating-function lattice at all
            eccentricities and removes the frozen toy's winding
            condition. If False, use the bare frozen-toy correspondence.
        Rmin, Rmax, Rinf : float, optional
            Radial anchors of the interpolation grid.
        nE, nL, ntau, nn : int, optional
            Grid sizes: energies, angular momenta, torus-sampling points
            (even), and the number of Fourier coefficients.
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
        self._ntau = ntau
        self._nn = nn
        self._nforSn = numpy.arange(1, nn + 1)
        self._maxiter = maxiter
        self._angle_tol = angle_tol
        self._interp = setup_interp
        self._pt = pt
        if not setup_interp:
            self._Es = numpy.atleast_1d(numpy.array(Es, dtype="float"))
            self._Ls = numpy.atleast_1d(numpy.array(Ls, dtype="float"))
            if len(self._Es) != len(self._Ls):
                raise ValueError("Es and Ls have to have the same length")
        else:
            self._setup_grid(Rmin, Rmax, Rinf, nE, nL)
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
            # the support-matched PT removes the winding condition
            # structurally (the lifted loop is the toy's own support, which
            # brackets the toy's circular point), so the rc-pinning of the
            # start point is only a good initial guess. What the lift DOES
            # need is depth headroom: the equal-action toy torus of the
            # most eccentric node must sit well below escape, else the
            # affine lift's shape mismatch crosses E^A = 0. Escalate GM
            # until every lifted sample clears escape by a fixed fraction
            # of the central well depth (E^A is monotone in GM, so this
            # terminates).
            for _ in range(20):
                self._GM, self._b = GM, b
                self._ip = IsochronePotential(amp=GM, b=b)
                ok = True
                for stau, sr, spr, srp, sra, sE, sL in self._samples:
                    Jrq = numpy.mean(
                        numpy.fabs(spr)
                        * 0.5
                        * (sra - srp)
                        * numpy.fabs(numpy.sin(stau))
                    )
                    EAtorus = self._iso_E_of_Jr(Jrq, sL)
                    rpA, raA = self._iso_turning(EAtorus, sL)
                    pip = (sra - srp) / (raA - rpA)
                    rA = rpA + (sr - srp) / pip
                    pA = spr * pip
                    EAs = numpy.max(
                        0.5 * (pA**2 + sL**2 / rA**2)
                        - GM / (b + numpy.sqrt(b**2 + rA**2))
                    )
                    # node-local margin: the lifted curve must clear escape
                    # by a (small) fraction of its own toy torus's binding
                    # energy; in-domain grids pass with the initial toy,
                    # and deepening cannot repair an out-of-domain grid
                    # (measured: the overshoot ratio is GM-invariant), so
                    # fail fast rather than churn
                    if EAs >= -1e-3 * numpy.fabs(EAtorus):
                        ok = False
                        break
                if ok:
                    break
                GM *= 1.2
            else:
                raise RuntimeError(
                    "The affine support-matched lift of the most eccentric "
                    "requested torus is not bound in any isochrone toy: "
                    "beyond eccentricity ~0.97 the target's radial-momentum "
                    "profile cannot be matched onto a near-Kepler toy torus "
                    "by an affine radial map (a profile-matched point "
                    "transformation would be needed here)"
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

    def _iso_turning(self, EA, L):
        """Radial turning points of the toy torus at (E^A, L)"""
        pr2 = lambda r: (
            2.0 * (EA + self._GM / (self._b + numpy.sqrt(self._b**2 + r**2)))
            - L**2 / r**2
        )
        rcA = rl(self._ip, L, use_physical=False)
        rlo, rhi = rcA, rcA
        while pr2(rlo) > 0.0 and rlo > 1e-12:
            rlo /= 1.3
        while pr2(rhi) > 0.0 and rhi < 1e12:
            rhi *= 1.3
        return (
            brentq(pr2, rlo, rcA, xtol=1e-12),
            brentq(pr2, rcA, rhi, xtol=1e-12),
        )

    def _pt_supports(self, tau, r, pr, rp, ra, L):
        """The PT parameters of one torus: the toy support [rpA, raA] of
        the equal-action toy torus, matched onto the target's [rp, ra];
        the parametrizing action is the plain loop-action quadrature (any
        stored parametrization of the family is canonical)"""
        Jr = numpy.mean(numpy.fabs(pr) * 0.5 * (ra - rp) * numpy.fabs(numpy.sin(tau)))
        rpA, raA = self._iso_turning(self._iso_E_of_Jr(Jr, L), L)
        return rpA, raA

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
        if self._pt:
            # lift the samples to the toy chart through the support-matched
            # PT: r = pi(r^A) affine through the four support parameters,
            # p^A = pi' p_r (the cotangent lift)
            rpA, raA = self._pt_supports(tau, r, pr, rp, ra, L)
            piprime = (ra - rp) / (raA - rpA)
            rA = rpA + (r - rp) / piprime
            pA = pr * piprime
        else:
            rpA, raA = rp, ra
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

        def _antider(f):
            fh = numpy.fft.fft(f - numpy.mean(f))
            ah = numpy.zeros_like(fh)
            ah[1:] = fh[1:] / (1j * k[1:])
            return numpy.real(numpy.fft.ifft(ah))

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
            "rpA": rpA,
            "raA": raA,
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
        self._rpAs = numpy.empty(ntori)
        self._raAs = numpy.empty(ntori)
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
            self._rpAs[ii] = node["rpA"]
            self._raAs[ii] = node["raA"]
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
        self._sup_tab = numpy.empty((nu, nLg, 4))  # rp, ra, rpA, raA
        self._coserr = 0.0
        for ii in range(nu):
            for jj in range(nLg):
                node = self._node_tables(self._E_tab[ii, jj], self._Lgrid[jj])
                self._jr_tab[ii, jj] = node["jr"]
                self._Sn_tab[ii, jj] = node["Sn"]
                self._OmR_tab[ii, jj] = node["OmR"]
                self._Ompsi_tab[ii, jj] = node["Ompsi"]
                self._sup_tab[ii, jj] = [
                    node["rp"],
                    node["ra"],
                    node["rpA"],
                    node["raA"],
                ]
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
            for q in range(4)
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
            return u, Sn, dSdJ, dSdL, OmR, OmL, None, None, None
        sup = numpy.array([ip(u, L)[0, 0] for ip in self._sup_ip])
        dsup_du = numpy.array([ip(u, L, dx=1)[0, 0] for ip in self._sup_ip])
        dsup_dL = numpy.array([ip(u, L, dy=1)[0, 0] for ip in self._sup_ip])
        dsupJ = dsup_du / djr_du
        dsupL = dsup_dL - dsup_du * djr_dL / djr_du
        return u, Sn, dSdJ, dSdL, OmR, OmL, sup, dsupJ, dsupL

    @staticmethod
    def _pi_chain(rA, sup, dsup):
        """d pi/d alpha at fixed r^A of the affine support map
        pi(r^A) = rp + (r^A - rpA) c, c = (ra - rp)/(raA - rpA), given the
        four support parameters and their derivatives along alpha"""
        rp, ra, rpA, raA = sup
        drp, dra, drpA, draA = dsup
        c = (ra - rp) / (raA - rpA)
        dc = ((dra - drp) - c * (draA - drpA)) / (raA - rpA)
        return drp + (rA - rpA) * dc - c * drpA

    # the compensation's sign, adjudicated by the symplectic-defect harness
    # (defect 6e-10 at -1 vs O(1) at +1/0), matching the 1D exact PT's
    # -v (dpi/dE) E' term
    _PT_COMP_SIGN = -1.0

    def _thetaA_solve(self, thr, dSdJ, comp=None):
        """Newton solve of theta_r = theta^A + 2 sum dS_n/dJ_r sin(n theta^A)
        [+ the PT-family compensation p_r (dpi/dJ_r)|_{r^A}] for theta^A,
        vectorized over the requested angles; the residual is exact, the
        Jacobian approximates by dropping the compensation's derivative
        (the 1D-validated approach)"""
        n = self._nforSn

        def _residual(x, target):
            snx = numpy.sin(x[:, None] * n[None, :])
            cnx = numpy.cos(x[:, None] * n[None, :])
            f = x + 2.0 * snx @ dSdJ - target
            if comp is not None:
                jr, L, Sn, sup, dsupJ = comp
                JAr = jr + 2.0 * cnx @ (n * Sn)
                rA, pA = self._toy_radial(JAr, L, x)
                pr = pA / ((sup[1] - sup[0]) / (sup[3] - sup[2]))
                f = f + self._PT_COMP_SIGN * pr * self._pi_chain(rA, sup, dsupJ)
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
        sup = None
        if self._interp:
            u, Sn, dSdJ, dSdL, OmR, OmL, sup, dsupJ, dsupL = self._interp_tables(jr, L)
            comp = None if not self._pt else (jr, L, Sn, sup, dsupJ)
            thetaAr = self._thetaA_solve(thr, dSdJ, comp=comp)
            snx = numpy.sin(thetaAr[:, None] * n[None, :])
            cnx = numpy.cos(thetaAr[:, None] * n[None, :])
            JAr = jr + 2.0 * cnx @ (n * Sn)
            Delta = 2.0 * snx @ dSdL
            if self._pt:
                # the PT family's L-chain compensates the psi-angles the
                # same way the J_r-chain compensates theta_r
                rA, pA = self._toy_radial(JAr, L, thetaAr)
                pr = pA / ((sup[1] - sup[0]) / (sup[3] - sup[2]))
                Delta = Delta + self._PT_COMP_SIGN * pr * self._pi_chain(rA, sup, dsupL)
        else:
            ii = self._match_node(jr, L)
            OmR, OmL = self._OmRs[ii], self._Ompsis[ii]
            taus = self._tau_solve(ii, thr)
            thetaAr = taus + _spec_eval(self._cP[ii], taus)
            JAr = jr + _spec_eval(self._cJ[ii], taus)
            Delta = _spec_eval(self._cD[ii], taus)
            if self._pt:
                sup = numpy.array(
                    [self._rps[ii], self._ras[ii], self._rpAs[ii], self._raAs[ii]]
                )
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
            out = self._pt_unlift(out, sup)
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

    @staticmethod
    def _pt_unlift(out, sup):
        """Map the toy-chart reconstruction to the target chart: the
        radial PT rescales the spherical radius (pi) and the radial
        momentum (1/pi'), leaves the position direction, the plane, and
        the azimuth alone, and rescales the tangential speed to keep
        |L| = r x v exact"""
        rp, ra, rpA, raA = sup
        c = (ra - rp) / (raA - rpA)
        R, vR, vT, z, vz, phi = out
        rA = numpy.sqrt(R**2 + z**2)
        vr = (R * vR + z * vz) / rA
        vth = (vR * z - vz * R) / rA
        rT = rp + (rA - rpA) * c
        scale = rT / rA
        vr2 = vr / c
        vth2 = vth / scale
        return numpy.array(
            [
                R * scale,
                vr2 * R / rA + vth2 * z / rA,
                vT / scale,
                z * scale,
                vr2 * z / rA - vth2 * R / rA,
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

    def _check_consistent_units(self):
        pass
