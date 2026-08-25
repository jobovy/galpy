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
from scipy.optimize import brentq

from ..potential import IsochronePotential, evaluatePotentials, rl, vcirc
from ..potential.Potential import _check_potential_list_and_deprecate
from .actionAngleInverse import actionAngleInverse
from .actionAngleIsochrone import actionAngleIsochrone
from .actionAngleIsochroneInverse import actionAngleIsochroneInverse


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
            If True, set up an (E, L) grid of tori spanning [Rmin, Rmax]
            circular angular momenta and energies up to the potential at
            Rinf, and interpolate canonically between them.
        Rmin, Rmax, Rinf : float, optional
            Radial anchors of the interpolation grid.
        nE, nL, ntau, nn : int, optional
            Grid sizes: energies, angular momenta, torus-sampling points,
            and the number of Fourier coefficients.
        maxiter : int, optional
            Maximum Newton iterations of the angle solve.
        angle_tol : float, optional
            Convergence tolerance of the angle solve.

        Notes
        -----
        - 2026-08-25 - Started - Bovy (UofT)
        """
        actionAngleInverse.__init__(self, *[], **kwargs)
        if pot is None:  # pragma: no cover
            raise OSError("Must specify pot= for actionAngleSphericalCanonical")
        self._pot = _check_potential_list_and_deprecate(pot)
        self._ntau = ntau
        self._nn = nn
        self._nforSn = numpy.arange(1, nn + 1)
        self._maxiter = maxiter
        self._angle_tol = angle_tol
        self._interp = setup_interp
        if not setup_interp:
            self._Es = numpy.atleast_1d(numpy.array(Es, dtype="float"))
            self._Ls = numpy.atleast_1d(numpy.array(Ls, dtype="float"))
            if len(self._Es) != len(self._Ls):
                raise ValueError("Es and Ls have to have the same length")
        else:
            self._setup_grid(Rmin, Rmax, Rinf, nE, nL)
        # sample every torus once (exact placement), then choose the toy,
        # then compute the tables against it
        self._sample_all()
        # frozen toy for the whole set (S2 brings the varying, compensated
        # form together with the radial alignment PT): the notes' rule --
        # unclamped, spherical potentials sit inside the isochrone's
        # frequency-ratio range -- at the central torus, with GM escalated
        # until the toy covers every sampled point (E^A < 0 is monotone in
        # GM, so this terminates; the price is S-size at the extremes,
        # which is S2's business)
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
        """Frozen isochrone by the notes' selection rule at the central
        torus: GM from the implicit frequency rule (fixed point), b from the
        frequency-ratio match, which is solvable for spherical targets;
        GM then escalated until every sampled point of every torus is bound
        in the toy"""
        E, L = self._Es[len(self._Es) // 2], self._Ls[len(self._Ls) // 2]
        tau, r, pr, rp, ra = self._sample_torus(E, L)
        Jr = (
            numpy.trapezoid(
                numpy.where(tau < numpy.pi, pr, 0.0) * 0.5 * (ra - rp) * numpy.sin(tau),
                tau,
            )
            / numpy.pi
        )
        # frequencies by regular quadrature in tau
        dr = 0.5 * (ra - rp) * numpy.sin(tau)
        w = numpy.where((tau > 0) & (tau < numpy.pi), 1.0, 0.0)
        prs = numpy.where(pr > 0, pr, numpy.inf)
        Tr = 2.0 * numpy.trapezoid(w * dr / prs, tau)
        OmR = 2.0 * numpy.pi / Tr
        Ompsi = 2.0 * numpy.trapezoid(w * dr * L / r**2 / prs, tau) / Tr
        rho = min(max(Ompsi / OmR, 0.501), 0.999)
        GM = 1.0
        for _ in range(200):
            b = max(((L / (2.0 * rho - 1.0)) ** 2 - L**2) / (4.0 * GM), 1e-8)
            CA = 0.5 * (L + numpy.sqrt(L**2 + 4.0 * GM * b))
            GMn = numpy.sqrt(OmR * (Jr + CA) ** 3)
            if abs(GMn - GM) < 1e-14:
                break
            GM = GMn
        # coverage: every sampled point of every torus must be bound in the
        # toy; E^A decreases monotonically with GM at fixed b
        for _ in range(60):
            ip = IsochronePotential(amp=GM, b=b)
            EAmax = max(
                numpy.max(
                    0.5 * (spr**2 + sL**2 / sr**2)
                    + evaluatePotentials(ip, sr, 0.0, use_physical=False)
                )
                for (stau, sr, spr, srp, sra, sE, sL) in self._samples
            )
            if EAmax < -1e-8:
                break
            GM *= 1.5
        else:  # pragma: no cover
            raise RuntimeError(
                "Could not find an isochrone toy that covers all requested tori"
            )
        self._GM, self._b = GM, b
        self._ip = IsochronePotential(amp=GM, b=b)
        self._aAI = actionAngleIsochrone(ip=self._ip)
        self._aAIinv = actionAngleIsochroneInverse(ip=self._ip)
        return None

    # ---------- the generating-function tables, computed (never fitted)
    def _cached_sample(self, E, L):
        for smp in self._samples:
            if smp[5] == E and smp[6] == L:
                return smp
        return self._sample_torus(E, L) + (E, L)  # pragma: no cover

    def _node_tables(self, E, L):
        """One node torus: sample exactly, run the closed-form isochrone
        correspondence (L^A = L pointwise, so only J^A_r fluctuates and the
        lattice is purely radial), and return the canonical action label
        (the gauge-independent zero mode) plus the sine coefficients of the
        generating function from the 1-D pullback"""
        tau, r, pr, rp, ra, _, _ = self._cached_sample(E, L)
        o = self._aAI.actionsFreqsAngles(
            r, pr, L / r, numpy.zeros_like(r), numpy.zeros_like(r), numpy.zeros_like(r)
        )
        JA = numpy.atleast_1d(o[0])
        thetaA = numpy.atleast_1d(o[6])
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
        return jr, Sn, maxcos, rp, ra

    def _setup_tori(self):
        if self._interp:
            return self._setup_tori_interp()
        self._jrs = numpy.empty(len(self._Es))
        self._Sn = numpy.empty((len(self._Es), self._nn))
        self._rps = numpy.empty(len(self._Es))
        self._ras = numpy.empty(len(self._Es))
        self._coserr = 0.0
        for ii, (E, L) in enumerate(zip(self._Es, self._Ls)):
            jr, Sn, maxcos, rp, ra = self._node_tables(E, L)
            self._jrs[ii] = jr
            self._Sn[ii] = Sn
            self._rps[ii] = rp
            self._ras[ii] = ra
            self._coserr = max(self._coserr, maxcos)
        return None

    def _setup_grid(self, Rmin, Rmax, Rinf, nE, nL):  # pragma: no cover
        raise NotImplementedError(
            "setup_interp=True arrives with the S1 interpolation commit"
        )

    def _setup_tori_interp(self):  # pragma: no cover
        raise NotImplementedError

    def _check_consistent_units(self):
        pass
