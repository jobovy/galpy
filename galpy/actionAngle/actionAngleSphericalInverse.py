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
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize

from ..potential import (
    IsochronePotential,
    evaluatePotentials,
    evaluateRforces,
    rl,
    vcirc,
)
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


class _HermiteFamily:
    """A tensor-product Hermite interpolant on a rectangular grid, quintic
    in the first variable and cubic in the second.  The values and the
    first partials are prescribed at every node and reproduced exactly
    there, together with the first partial's derivative along the second
    variable; the second derivatives in the first variable (and their
    derivative along the second) are estimated by differentiating cubic
    splines of the prescribed first partials, which makes the interpolant's
    first derivative in that variable accurate to one order beyond a cubic
    Hermite's.  Called like a RectBivariateSpline: ip(x, y, dx=, dy=)[0, 0]."""

    # the coefficient matrices of the unit-interval Hermite polynomials:
    # quintic through (f, f', f'') at both ends, cubic through (f, f')
    _Mq = numpy.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
            [-10.0, -6.0, -1.5, 10.0, -4.0, 0.5],
            [15.0, 8.0, 1.5, -15.0, 7.0, -1.0],
            [-6.0, -3.0, -0.5, 6.0, -3.0, 0.5],
        ]
    )
    _Mc = numpy.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-3.0, 3.0, -2.0, -1.0],
            [2.0, -2.0, 1.0, 1.0],
        ]
    )

    def __init__(self, x, y, f, fx, fy):
        self._x, self._y = numpy.asarray(x), numpy.asarray(y)
        nx, ny = len(x), len(y)
        fxy = numpy.array([CubicSpline(y, fx[i])(y, 1) for i in range(nx)])
        fxx = numpy.array([CubicSpline(x, fx[:, j])(x, 1) for j in range(ny)]).T
        fxxy = numpy.array([CubicSpline(x, fxy[:, j])(x, 1) for j in range(ny)]).T
        self._c = numpy.empty((nx - 1, ny - 1, 6, 4))
        for i in range(nx - 1):
            hx = self._x[i + 1] - self._x[i]
            for j in range(ny - 1):
                hy = self._y[j + 1] - self._y[j]
                # rows: (f, hx f_x, hx^2 f_xx) at x_i then at x_{i+1};
                # columns: values at y_j, y_{j+1}, then hy times the
                # y-derivatives there
                F = numpy.empty((6, 4))
                for r, (tab, sc) in enumerate(((f, 1.0), (fx, hx), (fxx, hx * hx))):
                    for k, ii in enumerate((i, i + 1)):
                        F[r + 3 * k, 0] = tab[ii, j] * sc
                        F[r + 3 * k, 1] = tab[ii, j + 1] * sc
                for r, (tab, sc) in enumerate(
                    ((fy, hy), (fxy, hx * hy), (fxxy, hx * hx * hy))
                ):
                    for k, ii in enumerate((i, i + 1)):
                        F[r + 3 * k, 2] = tab[ii, j] * sc
                        F[r + 3 * k, 3] = tab[ii, j + 1] * sc
                self._c[i, j] = self._Mq @ F @ self._Mc.T

    def __call__(self, x, y, dx=0, dy=0):
        i = min(
            max(numpy.searchsorted(self._x, x, side="right") - 1, 0), len(self._x) - 2
        )
        j = min(
            max(numpy.searchsorted(self._y, y, side="right") - 1, 0), len(self._y) - 2
        )
        hx, hy = self._x[i + 1] - self._x[i], self._y[j + 1] - self._y[j]
        sv = (x - self._x[i]) / hx
        tv = (y - self._y[j]) / hy
        if dx == 0:
            ps = sv ** numpy.arange(6)
        else:
            ps = (
                numpy.array([0.0, 1.0, 2.0 * sv, 3.0 * sv**2, 4.0 * sv**3, 5.0 * sv**4])
                / hx
            )
        if dy == 0:
            pt = tv ** numpy.arange(4)
        else:
            pt = numpy.array([0.0, 1.0, 2.0 * tv, 3.0 * tv**2]) / hy
        return numpy.array([[ps @ self._c[i, j] @ pt]])


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
        setup_interp=False,
        Rmin=0.5,
        Rmax=2.0,
        Rinf=25.0,
        nE=16,
        nL=16,
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
            Energies and angular momenta of the tori to set up when
            setup_interp is False (paired lists).
        setup_interp : bool, optional
            If True, set up an (E, L) grid of tori spanning the circular
            angular momenta of [Rmin, Rmax] and energies up to the
            potential at Rinf, and interpolate canonically between them.
        Rmin, Rmax, Rinf : float, optional
            Radial anchors of the interpolation grid: the angular momenta of
            the grid span those of the circular orbits at Rmin and Rmax, and
            its energies reach the potential at Rinf.
        nE, nL : int, optional
            Numbers of energies and angular momenta of the interpolation
            grid.
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
        self._interp = setup_interp
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

    def _toy_flux_derivs(self, a, e, eta):
        """The auxiliary's radial action flux f^A = p^A dr^A/deta at
        eccentric anomaly eta, and its partials with respect to the torus
        parameters (a, e) at fixed eta, all closed forms of the isochrone:
        with beta = b/a, y = 1 - e cos(eta), c = sqrt(GM/(a+b)) and
        g(y) = (y + beta) / [y (y + 2 beta)], f^A = c a e^2 sin^2(eta) g"""
        b = self._b
        beta = b / a
        y = 1.0 - e * numpy.cos(eta)
        c = numpy.sqrt(self._GM / (a + b))
        s2 = numpy.sin(eta) ** 2
        g = (y + beta) / (y * (y + 2.0 * beta))
        dg_dy = -(y**2 + 2.0 * beta * y + 2.0 * beta**2) / (y * (y + 2.0 * beta)) ** 2
        dg_da = (b / a**2) / (y + 2.0 * beta) ** 2
        fA = c * a * e**2 * s2 * g
        dfA_da = e**2 * s2 * (c * (a + 2.0 * b) / (2.0 * (a + b)) * g + c * a * dg_da)
        dfA_de = c * a * s2 * (2.0 * e * g - e**2 * dg_dy * numpy.cos(eta))
        return fA, dfA_da, dfA_de

    def _map_slopes(self, tau, r, pr, rp, ra, E, L, Jrq, a, e, etat, OmR, Ompsi):
        """The derivatives of the anomaly-map coefficients with respect to E
        at fixed L and to L at fixed E, on one torus from that torus alone:
        the variation of the matching condition A^A(eta(tau); a, e) =
        A(tau; E, L) at fixed anomaly,

            f^A(eta) sum_m dD_m/dalpha sin(m tau)
                = dA/dalpha|_tau - dA^A/da a_alpha - dA^A/de e_alpha ,

        a LINEAR least-squares problem for the dD_m/dalpha with the
        vanishing flux f^A multiplying the unknowns, as in the map's own
        fit. The target's flux derivative at fixed tau goes through the
        moving turning points and is regular there (its numerator vanishes
        with p_r), so it integrates spectrally like the flux itself; the
        auxiliary's goes through its torus parameters' closed-form chains."""
        k = numpy.fft.fftfreq(self._ntau, d=1.0 / self._ntau)

        def _cum(f):
            # the integral from tau = 0 of a periodic f sampled on the
            # (half-offset) grid: its mean times tau plus the periodic part,
            # evaluated at the grid and returned with its spectral
            # coefficients for evaluation elsewhere
            m = numpy.mean(f)
            fh = numpy.fft.fft(f - m)
            ah = numpy.zeros_like(fh)
            ah[1:] = fh[1:] / (1j * k[1:])
            q = numpy.real(numpy.fft.ifft(ah))
            cq = _spec_coeffs(q)
            q0 = _spec_eval(cq, 0.0)[0]
            return m, cq, q0

        costau, sintau = numpy.cos(tau), numpy.sin(tau)
        drdtau = 0.5 * (ra - rp) * sintau
        dPhieff = -evaluateRforces(self._pot, r, 0.0, use_physical=False) - L**2 / r**3
        fA, dfA_da, dfA_de = self._toy_flux_derivs(a, e, tau)
        cfA = _spec_coeffs(fA)
        mA_a, cqa, qa0 = _cum(dfA_da)
        mA_e, cqe, qe0 = _cum(dfA_de)
        FAa = mA_a * etat + _spec_eval(cqa, etat) - qa0
        FAe = mA_e * etat + _spec_eval(cqe, etat) - qe0
        B = _spec_eval(cfA, etat)[:, None] * numpy.sin(
            tau[:, None] * self._nforDm[None, :]
        )
        out = []
        for alpha in ("E", "L"):
            drp = self._turning_point_derivs(rp, E, L)[alpha == "L"]
            dra = self._turning_point_derivs(ra, E, L)[alpha == "L"]
            dr = drp * (1.0 + costau) / 2.0 + dra * (1.0 - costau) / 2.0
            num = (1.0 if alpha == "E" else -L / r**2) - dPhieff * dr
            dft = num / pr * drdtau + pr * 0.5 * (dra - drp) * sintau
            mt, cqt, qt0 = _cum(dft)
            dA = mt * tau + _spec_eval(cqt, tau) - qt0
            if alpha == "E":
                _, _, da, de = self._toy_param_chains(Jrq, L, 1.0 / OmR, 0.0)
            else:
                _, _, da, de = self._toy_param_chains(Jrq, L, -Ompsi / OmR, 1.0)
            rhs = dA - FAa * da - FAe * de
            out.append(numpy.linalg.lstsq(B, rhs, rcond=None)[0])
        return out[0], out[1]

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
        for _ in range(100):
            f = x - aeab * numpy.sin(x) - ar
            x -= numpy.clip(f / (1.0 - aeab * numpy.cos(x)), -1.0, 1.0)
            if numpy.max(numpy.fabs(f)) < 1e-14:
                break
        coseta = numpy.cos(x)
        rA = a * numpy.sqrt((1.0 - e * coseta) * (1.0 - e * coseta + 2.0 * bb / a))
        pA = numpy.sqrt(amp / ab) * a * e * numpy.sin(x) / rA
        return rA, pA

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
        dDm_dE, dDm_dL = self._map_slopes(
            tau, r, pr, rp, ra, E, L, Jrq, a, e, etat, OmR, Ompsi
        )
        return {
            "jr": Jrq,
            "perr": perr,
            "rp": rp,
            "ra": ra,
            "Dm": Dm,
            "dDm_dE": dDm_dE,
            "dDm_dL": dDm_dL,
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
        if self._interp:
            return self._setup_tori_interp()
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
        self._Ecs = numpy.array([self._Ec(L)[0] for L in self._Lgrid])
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

    def _Ec(self, L):
        """Energy of the circular orbit of angular momentum L, and its
        derivative dE_c/dL = L / r_c^2 (the circular frequency)"""
        rc = rl(self._pot, L, use_physical=False)
        return self._Phi(rc) + L**2 / (2.0 * rc**2), L / rc**2

    def _E_of_uL(self, u, L):
        """The grid's energy variable, analytic: E = E_c(L) + [E_max -
        E_c(L)] u^2, with its partials in u and in L at fixed u"""
        Ec, dEc = self._Ec(L)
        return (
            Ec + (self._Emax - Ec) * u**2,
            2.0 * u * (self._Emax - Ec),
            dEc * (1.0 - u**2),
        )

    def _turning_point_derivs(self, r, E, L):
        """Closed-form derivatives of a radial turning point (a root of
        E - Phi(r) - L^2 / 2 r^2) with respect to E at fixed L and to L at
        fixed E, by the level-set rule"""
        dPhieff = -evaluateRforces(self._pot, r, 0.0, use_physical=False) - L**2 / r**3
        return 1.0 / dPhieff, -(L / r**2) / dPhieff

    def _setup_tori_interp(self):
        nu, nLg = len(self._us), len(self._Lgrid)
        self._jr_tab = numpy.empty((nu, nLg))
        self._OmR_tab = numpy.empty((nu, nLg))
        self._Ompsi_tab = numpy.empty((nu, nLg))
        self._sup_tab = numpy.empty((nu, nLg, 2))  # rp, ra
        self._Dm_tab = numpy.empty((nu, nLg, self._npt))
        # exact first partials at every node: of J_r from the node's
        # frequencies (dJ_r = [dE - Omega_psi dL] / Omega_r), in the
        # normalized energy x = u^2 in which J_r is smooth up to the circular
        # edge (in u its derivative would vanish there), and of the turning
        # points, which behave as u at the edge, in u itself; both from the
        # level-set rule
        self._jr_dx = numpy.empty((nu, nLg))
        self._jr_dL = numpy.empty((nu, nLg))
        self._sup_du = numpy.empty((nu, nLg, 2))
        self._sup_dL = numpy.empty((nu, nLg, 2))
        self._Dm_du = numpy.empty((nu, nLg, self._npt))
        self._Dm_dL = numpy.empty((nu, nLg, self._npt))
        perr = numpy.empty((nu, nLg))
        for ii in range(nu):
            for jj in range(nLg):
                L = self._Lgrid[jj]
                node = self._node_tables(ii * nLg + jj)
                self._jr_tab[ii, jj] = node["jr"]
                self._OmR_tab[ii, jj] = node["OmR"]
                self._Ompsi_tab[ii, jj] = node["Ompsi"]
                self._sup_tab[ii, jj] = [node["rp"], node["ra"]]
                self._Dm_tab[ii, jj] = node["Dm"]
                perr[ii, jj] = node["perr"]
                E, dE_du, dE_dL = self._E_of_uL(self._us[ii], L)
                # dE/dx at fixed L is E_max - E_c, and dE/dL at fixed x is
                # dE/dL at fixed u
                self._jr_dx[ii, jj] = (self._Emax - self._Ecs[jj]) / node["OmR"]
                self._jr_dL[ii, jj] = (dE_dL - node["Ompsi"]) / node["OmR"]
                for q, r in enumerate((node["rp"], node["ra"])):
                    dr_dE, dr_dL = self._turning_point_derivs(r, E, L)
                    self._sup_du[ii, jj, q] = dr_dE * dE_du
                    self._sup_dL[ii, jj, q] = dr_dE * dE_dL + dr_dL
                self._Dm_du[ii, jj] = node["dDm_dE"] * dE_du
                self._Dm_dL[ii, jj] = node["dDm_dE"] * dE_dL + node["dDm_dL"]
        self._warn_unresolved(
            perr.flatten(), self._E_tab.flatten(), numpy.tile(self._Lgrid, nu)
        )
        self._rebuild_interp()
        return None

    def _hermite(self, xs, f, fx, fL):
        """The family's interpolant of a table on (xs, L) with exact first
        partials at the nodes: quintic Hermite along xs, cubic along L"""
        return _HermiteFamily(xs, self._Lgrid, f, fx, fL)

    def _rebuild_interp(self):
        """(Re)build the interpolants from the stored tables: Hermite
        interpolants of J_r (in the normalized energy), of the turning
        points and of the anomaly-map coefficients (in u), all with their
        first partials exact at the nodes"""
        u, Lg = self._us, self._Lgrid
        self._jr_ip = self._hermite(u**2, self._jr_tab, self._jr_dx, self._jr_dL)
        self._sup_ip = [
            self._hermite(
                u, self._sup_tab[:, :, q], self._sup_du[:, :, q], self._sup_dL[:, :, q]
            )
            for q in range(2)
        ]
        self._Dm_ip = [
            self._hermite(
                u, self._Dm_tab[:, :, q], self._Dm_du[:, :, q], self._Dm_dL[:, :, q]
            )
            for q in range(self._npt)
        ]
        return None

    # ---------- evaluation: the manifest chain
    def _interp_tables(self, jr, L):
        """Solve the implicit inverse label (u from (J_r, L) by root-finding
        on the stored J_r interpolant -- exact-in-the-family, so canonicity
        is untouched) and return the frequencies and the map's tables with
        their derivatives, all the interpolants' own, combined into the
        chains at fixed L and at fixed J_r that the evaluation needs"""
        if L < self._Lgrid[0] or L > self._Lgrid[-1]:
            raise ValueError(
                f"L = {L} outside the interpolation grid "
                f"[{self._Lgrid[0]}, {self._Lgrid[-1]}]"
            )
        jlo = self._jr_ip(self._us[0] ** 2, L)[0, 0]
        jhi = self._jr_ip(self._us[-1] ** 2, L)[0, 0]
        tol = 1e-12 * (1.0 + numpy.fabs(jr))
        if jr < jlo - tol or jr > jhi + tol:
            raise ValueError(
                f"J_r = {jr} outside the interpolated family's range "
                f"[{jlo}, {jhi}] at L = {L}"
            )
        jr = min(max(jr, jlo), jhi)  # the grid's own nodes, to round-off
        x = brentq(
            lambda xx: self._jr_ip(xx, L)[0, 0] - jr,
            self._us[0] ** 2,
            self._us[-1] ** 2,
            xtol=1e-14,
        )
        u = numpy.sqrt(x)
        djr_du = 2.0 * u * self._jr_ip(x, L, dx=1)[0, 0]
        djr_dL = self._jr_ip(x, L, dy=1)[0, 0]
        _, dE_du, dE_dL = self._E_of_uL(u, L)
        # chains at fixed L resp. fixed J_r, all from the stored
        # interpolants' own derivatives
        OmR = dE_du / djr_du
        OmL = dE_dL - dE_du * djr_dL / djr_du
        sup = numpy.array([ip(u, L)[0, 0] for ip in self._sup_ip])
        dsup_du = numpy.array([ip(u, L, dx=1)[0, 0] for ip in self._sup_ip])
        dsup_dL = numpy.array([ip(u, L, dy=1)[0, 0] for ip in self._sup_ip])
        Dm = numpy.array([ip(u, L)[0, 0] for ip in self._Dm_ip])
        dDm_du = numpy.array([ip(u, L, dx=1)[0, 0] for ip in self._Dm_ip])
        dDm_dL = numpy.array([ip(u, L, dy=1)[0, 0] for ip in self._Dm_ip])
        # the auxiliary torus has the requested radial action itself, so
        # its parameter chain is the identity in J_r and nothing in L
        ptdata = {
            "L": L,
            "sup": sup,
            "dsupJ": dsup_du / djr_du,
            "dsupL": dsup_dL - dsup_du * djr_dL / djr_du,
            "Jrq": jr,
            "dJrqJ": 1.0,
            "dJrqL": 0.0,
            "Dm": Dm,
            "dDmJ": dDm_du / djr_du,
            "dDmL": dDm_dL - dDm_du * djr_dL / djr_du,
        }
        return u, OmR, OmL, ptdata

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
        # the map is monotone and smooth: Newton converges to round-off in a
        # handful of steps, so this has its own budget and tolerance and
        # maxiter / angle_tol govern only the angle solves
        ms = self._nforDm
        x = numpy.array(eta, dtype="float")
        for _ in range(100):
            f = x + numpy.sin(x[:, None] * ms[None, :]) @ Dm - eta
            fp = 1.0 + numpy.cos(x[:, None] * ms[None, :]) @ (ms * Dm)
            dx = numpy.clip(-f / fp, -0.5, 0.5)
            x += dx
            if numpy.max(numpy.fabs(f)) < 1e-14:
                break
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

    def _thetaA_solve(self, thr, jr, L, ptdata):
        """Newton solve of theta_r = theta^A_r - p_r (d pi / d J_r)|_{r^A}
        for the auxiliary angle theta^A_r, vectorized over the requested
        angles; the residual is exact and the Jacobian approximates by
        dropping the compensation's derivative (the 1D-validated approach),
        with a safeguarded scalar fallback"""

        def _residual(x, target):
            rA, pA = self._toy_radial(jr, L, x)
            return (
                x
                - target
                + self._pt_comp(
                    rA,
                    pA,
                    ptdata,
                    ptdata["dJrqJ"],
                    ptdata["dsupJ"],
                    ptdata["dDmJ"],
                    0.0,
                )
            )

        x = numpy.array(thr, dtype="float")
        for _ in range(self._maxiter):
            f = _residual(x, thr)
            dx = numpy.clip(-f, -0.5, 0.5)
            x += dx
            if numpy.max(numpy.fabs(f)) < self._angle_tol:
                break
        else:
            # the approximate-Jacobian Newton can cycle on hard tori even
            # where the angle map is monotone; fall back to safeguarded
            # scalar root-finding per non-converged point. The residual is
            # x + Q(x) - theta with Q periodic, so [theta - maxQ - eps,
            # theta + maxQ + eps] is a guaranteed bracket.
            f = _residual(x, thr)
            bad = numpy.flatnonzero(numpy.fabs(f) >= self._angle_tol)
            xscan = numpy.linspace(0.0, 2.0 * numpy.pi, 256, endpoint=False)
            s = numpy.max(numpy.fabs(_residual(xscan, numpy.zeros(256)) - xscan)) + 0.1
            for ii in bad:
                thri = thr[ii]

                def _fi(xx):
                    return _residual(numpy.array([xx]), numpy.array([thri]))[0]

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
        - For an interpolating instance (setup_interp=True), any (E, L) within
          the grid, read from the family's own J_r(u, L) interpolant (exact at
          the nodes); for an instance set up with explicit tori, (E, L) must be
          one of them.
        - 2026-09-18 - Written - Bovy (UofT)
        """
        E = conversion.parse_energy(E, vo=self._vo)
        L = conversion.parse_angmom(L, ro=self._ro, vo=self._vo)
        if not self._interp:
            dev = numpy.fabs(self._Es - E) + numpy.fabs(self._Ls - L)
            ii = numpy.argmin(dev)
            if dev[ii] > 1e-10 * (1.0 + numpy.fabs(E) + numpy.fabs(L)):
                raise ValueError(
                    f"(E, L) = ({E}, {L}) is not one of the set-up tori; to use "
                    "interpolation, initialize with setup_interp=True"
                )
            return self._jrs[ii]
        if L < self._Lgrid[0] or L > self._Lgrid[-1]:
            raise ValueError(
                f"L = {L} outside the interpolation grid "
                f"[{self._Lgrid[0]}, {self._Lgrid[-1]}]"
            )
        Ec, _ = self._Ec(L)
        u2 = (E - Ec) / (self._Emax - Ec)
        if u2 < 0.0 or u2 > 1.0:
            raise ValueError(
                f"E = {E} outside the interpolation grid at L = {L}: "
                f"[{Ec}, {self._Emax}]"
            )
        return self._jr_ip(u2, L)[0, 0]

    def _xvFreqs(self, jr, jphi, jz, angler, anglephi, anglez, **kwargs):
        """(J, theta) -> (x, v): solve the 1-D Newton for theta^A_r, shift
        the two companion angles by the map's L-compensation, delegate the
        full 3-D reconstruction to the analytic isochrone inverse (L^A = L,
        so the plane geometry is the auxiliary's own), and undo the lift"""
        jr, jphi, jz = float(jr), float(jphi), float(jz)
        L = jz + numpy.fabs(jphi)
        angler, anglephi, anglez = numpy.broadcast_arrays(
            numpy.atleast_1d(angler).astype(float),
            numpy.atleast_1d(anglephi).astype(float),
            numpy.atleast_1d(anglez).astype(float),
        )
        thr = angler % (2.0 * numpy.pi)
        if self._interp:
            u, OmR, OmL, ptdata = self._interp_tables(jr, L)
            thetaAr = self._thetaA_solve(thr, jr, L, ptdata)
            # the map's L-chain compensates the psi-angles the same way
            # its J_r-chain compensates theta_r
            rA, pA = self._toy_radial(jr, L, thetaAr)
            Delta = self._pt_comp(
                rA,
                pA,
                ptdata,
                ptdata["dJrqL"],
                ptdata["dsupL"],
                ptdata["dDmL"],
                1.0,
            )
            a, e = self._toy_params(jr, L)
            params = (a, e, ptdata["Dm"], ptdata["sup"][0], ptdata["sup"][1])
        else:
            ii = self._match_node(jr, L)
            OmR, OmL = self._OmRs[ii], self._Ompsis[ii]
            taus = self._tau_solve(ii, thr)
            thetaAr = taus + _spec_eval(self._cP[ii], taus)
            Delta = _spec_eval(self._cD[ii], taus)
            a, e = self._toy_params(self._jrs[ii], L)
            params = (a, e, self._Dms[ii], self._rps[ii], self._ras[ii])
        thetaAz = anglez - Delta
        thetaAphi = anglephi - numpy.sign(jphi) * Delta
        out = numpy.empty((6, len(thr)))
        for ii in range(len(thr)):
            oo = self._aAIinv._xvFreqs(
                jr, jphi, jz, thetaAr[ii], thetaAphi[ii], thetaAz[ii]
            )
            for jj in range(6):
                out[jj, ii] = oo[jj][0]
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
        """Undo the lift: from the auxiliary-chart reconstruction to the
        target chart. The auxiliary anomaly from the closed-form radius inversion, the stored anomaly
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
            _, OmR, OmL, _ = self._interp_tables(jr, L)
        else:
            ii = self._match_node(jr, L)
            OmR, OmL = self._OmRs[ii], self._Ompsis[ii]
        return (OmR, numpy.sign(jphi) * OmL, OmL)
