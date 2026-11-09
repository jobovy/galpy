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
from scipy.optimize import brentq

from ..backend import (
    as_numpy,
    asarray_on_device,
    at_least_float64,
    backend,
    concretely_true,
    device_of,
    get_namespace,
    has_concrete_truth_value,
    is_backend_array,
    lazy_cond,
    match_input_dtype,
    to_host,
)
from ..backend._namespaces import stop_gradient, under_jax_trace
from ..backend.interpolate import interp_linear
from ..backend.optimize import brentq as _backend_brentq
from ..backend.optimize import iterate_bracket, iterate_to_fixed_point
from ..potential import (
    IsochronePotential,
    epifreq,
    evaluatePotentials,
    evaluateR2derivs,
    evaluateRforces,
    rl,
    vcirc,
)
from ..potential.Potential import (
    _check_potential_list_and_deprecate,
    _evaluatePotentials,
    _evaluateRforces,
    _pot_grad_namespace,
)
from ..util import conversion, galpyWarning
from ..util._hermite import HermiteFamily2D, slope_at_zero
from ..util._optional_deps import _TQDM_LOADED
from .actionAngleInverse import actionAngleInverse
from .actionAngleInverseUtil import (
    _BOUND_MARGIN,
    _antiderivative_xp,
    _bk_matching,
    _chunked,
    _fit_isochrone_vc2,
    _ns,
    _offset_spec_coeffs,
    _offset_spec_coeffs_xp,
    _progress,
    _safeguarded_newton,
    _sdot,
    _sine_galerkin,
    _sine_galerkin_xp,
    _sine_project,
    _sine_project_xp,
    _sine_series,
    _sine_series_xp,
    _spec_eval,
    _spec_eval_xp,
)
from .actionAngleIsochroneInverse import actionAngleIsochroneInverse


def _scalar_probe(t):
    """A scalar standing in for the array t in a concreteness test (which
    needs a scalar): concrete exactly when t is"""
    return get_namespace(t).sum(t * 0.0) if t.ndim else t


def _bk_dot(xp, A, D):
    """sin(m tau) (or cos) of shape (n, M) times the coefficients D: one
    vector for all anomalies (a matrix product) or one row per anomaly"""
    return A @ D if D.ndim == 1 else xp.sum(A * D, axis=-1)


class actionAngleSphericalInverse(actionAngleInverse):
    """Inverse action-angle transformation for spherical potentials through a
    momentum-matched canonical map onto an isochrone auxiliary (the
    ``auxiliary`` attribute).

    .. note::
       The construction and the evaluation run natively under jax/torch (under
       a forced backend, or for a potential with jax/torch parameters), and
       are differentiable in the actions, the angles, and the potential's
       parameters, and traceable under ``jax.jit``. The fitted auxiliary is
       held fixed in these derivatives (any auxiliary gives a canonical map).
       An explicit torus evaluates as a quadratic one-node family, from its
       second-order tables, so that its derivatives in the actions are those
       of the family of tori at the torus.
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
        mm_nta=None,
        auxiliary=None,
        maxiter=100,
        angle_tol=1e-12,
        progressbar=True,
        **kwargs,
    ):
        """
        Initialize an actionAngleSphericalInverse object.

        Parameters
        ----------
        pot : Potential or a combined potential formed using addition (pot1+pot2+…)
            A spherical potential.
        Es : array-like or Quantity
            Energies of the tori to set up when setup_interp is False
            (paired with Ls).
        Ls : array-like or Quantity
            Total angular momenta of the tori to set up when setup_interp
            is False (paired with Es).
        setup_interp : bool, optional
            If True, set up an (E, L) grid of tori spanning the circular
            angular momenta of [Rmin, Rmax] and energies up to the
            potential at Rinf, and interpolate canonically between them.
        Rmin : float or Quantity, optional
            Radius of the circular orbit with the smallest angular momentum
            of the interpolation grid.
        Rmax : float or Quantity, optional
            Radius of the circular orbit with the largest angular momentum
            of the interpolation grid.
        Rinf : float or Quantity, optional
            Radius at which the potential equals the interpolation grid's
            largest energy. The grid's energies are uniform in the
            logarithm of the apocentre of the radial orbit between each
            circular orbit's energy and this one (uniform in the energy for
            a logarithmic halo, and compressed toward the top in a
            potential with an escape energy, where the apocentre diverges),
            spaced quadratically at the circular edge.
        nE : int, optional
            Number of energies of the interpolation grid.
        nL : int, optional
            Number of angular momenta of the interpolation grid.
        mm_npt : int, optional
            Number of harmonics of the momentum-matched anomaly map; the
            reconstruction converges spectrally in this, and a warning is
            raised when it does not suffice for a torus. The number needed
            grows with a torus's ratio of apocentre to pericentre, roughly
            in proportion to it; on a grid the most eccentric torus is the
            one at the smallest angular momentum (set by Rmin) and the
            highest energy (set by Rinf), and the set-up raises when the
            harmonics cannot hold it.
        mm_nta : int, optional
            Number of anomaly samples per torus (even), used to sample the
            torus, to fit the map, and for the quadratures of its action
            and frequencies; must exceed 4 * mm_npt for the samples to
            resolve the map's highest harmonic, which is 2 * mm_npt. The
            default (None) is 8 * mm_npt.
        auxiliary : IsochronePotential, optional
            The isochrone auxiliary onto which every torus is lifted. By
            default it is fitted to the potential's rotation curve over the
            sampled radial range; either way it is available afterwards as
            the ``auxiliary`` attribute.
        maxiter : int, optional
            Maximum Newton iterations of the angle solve.
        angle_tol : float, optional
            Convergence tolerance of the angle solve.
        progressbar : bool, optional
            If True, display tqdm progress bars over the tori while they
            are lifted and their tables computed (requires tqdm to be
            installed). Default is True.

        Notes
        -----
        - 2026-08-25 - Started - Bovy (UofT)
        """
        actionAngleInverse.__init__(self, *[], **kwargs)
        if pot is None:
            raise OSError("Must specify pot= for actionAngleSphericalInverse")
        self._pot = _check_potential_list_and_deprecate(pot)
        if mm_nta is None:
            mm_nta = 8 * mm_npt
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
        self._progressbar = progressbar and _TQDM_LOADED
        self._interp = setup_interp
        self._rc_cache = {}
        # A forced backend, or a potential with jax/torch parameters, builds
        # the family natively, differentiable in those parameters
        if backend() != "numpy":
            xp_bk = get_namespace()
        else:
            xp_bk = _pot_grad_namespace(self._pot, any_backend=True)
        self._bk = None
        if not setup_interp:
            self._Es = conversion._parse_grid_quantity(
                to_host(Es), conversion.parse_energy, vo=self._vo
            ).astype("float")
            self._Ls = conversion._parse_grid_quantity(
                to_host(Ls), conversion.parse_angmom, ro=self._ro, vo=self._vo
            ).astype("float")
            if len(self._Es) != len(self._Ls):
                raise ValueError("Es and Ls have to have the same length")
        if xp_bk is not None:
            self._init_backend(
                xp_bk,
                *(
                    float(as_numpy(conversion.parse_length(R, ro=self._ro)))
                    for R in (Rmin, Rmax, Rinf)
                ),
                nE,
                nL,
                auxiliary,
            )
            self._check_consistent_units()
            return None
        if setup_interp:
            self._setup_grid(
                conversion.parse_length(Rmin, ro=self._ro),
                conversion.parse_length(Rmax, ro=self._ro),
                conversion.parse_length(Rinf, ro=self._ro),
                nE,
                nL,
            )
        # sample every torus once (exact placement), then fix the frozen
        # auxiliary and lift every torus onto it (all torus-dependence
        # beyond the auxiliary lives in the momentum-matched map, whose
        # compensation is closed-form), then compute the tables
        self._sample_all()
        self._setup_auxiliary(auxiliary)
        self._setup_tori()
        self._check_consistent_units()
        return None

    # ---------- node construction: exact radial tori by quadrature
    def _Phi(self, r):
        return evaluatePotentials(self._pot, r, 0.0, use_physical=False)

    def _rc(self, L):
        """Radius of the circular orbit with angular momentum L: cached for
        a single L, solved for all at once for an array"""
        if numpy.ndim(L) > 0:
            if numpy.size(L) == 1:  # a single torus: through the cache
                return numpy.array([self._rc(float(numpy.ravel(L)[0]))])
            return rl(self._pot, L, use_physical=False)
        if L not in self._rc_cache:
            # a float: under a forced backend rl returns a 0-d tensor, which
            # _turning_points's in-place bracket steps would share
            self._rc_cache[L] = float(rl(self._pot, L, use_physical=False))
        return self._rc_cache[L]

    def _turning_points(self, E, L):
        """Radial turning points of the (E, L) torus, bracketed outward
        from the circular radius"""
        rc = self._rc(L)

        def pr2(r):
            return 2.0 * (E - self._Phi(r)) - L**2 / r**2

        ttol = 1e-12
        rlo, rhi = rc, rc
        while pr2(rlo) > 0.0 and rlo > 1e-12:
            rlo /= 1.3
        while pr2(rhi) > 0.0 and rhi < 1e12:
            rhi *= 1.3
        if pr2(rhi) > 0.0:
            raise ValueError(
                f"No bound radial libration exists at (E, L) = ({E:g}, {L:g}): "
                "the torus is unbound"
            )
        return brentq(pr2, rlo, rc, xtol=ttol), brentq(pr2, rc, rhi, xtol=ttol)

    def _sample_torus(self, E, L):
        """Exact phase-space samples along the radial loop, parametrized by
        the tau anomaly on the offset grid tau_k = 2 pi (k + 1/2)/N, which
        keeps every sample off the turning points; placement is exact by
        construction (p_r from the energy relation, not from a fit)"""
        rp, ra = self._turning_points(E, L)
        tau = 2.0 * numpy.pi * (numpy.arange(self._ntau) + 0.5) / self._ntau
        r = 0.5 * (ra + rp) - 0.5 * (ra - rp) * numpy.cos(tau)
        pr2 = 2.0 * (E - self._Phi(r)) - L**2 / r**2
        pr2[pr2 < 0.0] = 0.0
        pr = numpy.where(tau < numpy.pi, 1.0, -1.0) * numpy.sqrt(pr2)
        return tau, r, pr, rp, ra

    def _sample_all(self):
        # a circular torus has no libration to sample: it is stored as its
        # radius alone, and its tables are the harmonic limit's
        self._samples = []
        for E, L in zip(self._Es, self._Ls):
            if self._is_circular(E, L):
                rc = self._rc(L)
                self._samples.append((None, None, None, rc, rc, E, L))
            else:
                self._samples.append(self._sample_torus(E, L) + (E, L))
        return None

    # ---------- the auxiliary: one isochrone for the whole family
    def _setup_auxiliary(self, auxiliary):
        """Fix the frozen isochrone auxiliary of the whole family and lift
        every torus onto it.

        The momentum-matched map lifts every torus onto its equal-action
        auxiliary torus, so the auxiliary's only job is to be a
        well-conditioned global surrogate of the target. Unless one is
        given, its rotation curve is fitted to the target's over the
        sampled radial range (two parameters, by a converged least
        squares). Every torus's truncated lift is then required to clear
        escape by a fraction of its own auxiliary torus's binding energy,
        which fails only when the stored anomaly map is under-resolved."""
        librating = [smp for smp in self._samples if smp[0] is not None]
        if auxiliary is not None:
            if not isinstance(auxiliary, IsochronePotential):
                raise TypeError("auxiliary= has to be an IsochronePotential")
            self._GM, self._b = auxiliary._amp, auxiliary.b
        elif not librating:
            # only circular orbits: any auxiliary serves, since the lift is
            # undone with the target's own radius; take the isochrone with
            # the central torus's circular radius as its scale
            L = self._Ls[len(self._Ls) // 2]
            rc = self._rc(L)
            self._GM, self._b = (
                L**2 * numpy.sqrt(2.0) * (1.0 + numpy.sqrt(2.0)) ** 2 / rc,
                rc,
            )
        else:
            self._GM, self._b = self._fit_auxiliary(librating)
        if auxiliary is None:
            # Python floats, not numpy scalars: torch.compile under
            # set_default_device mis-traces numpy scalar arithmetic on tensors
            self._GM, self._b = float(self._GM), float(self._b)
        self.auxiliary = (
            auxiliary
            if auxiliary is not None
            else IsochronePotential(amp=self._GM, b=self._b)
        )
        self._aAIinv = actionAngleIsochroneInverse(ip=self.auxiliary)
        # lift every librating torus once, keeping the lift for the tables,
        # and require it to stay bound in the auxiliary
        self._lifts = {}
        for ii in _progress(
            range(len(self._samples)), "lifting tori", self._progressbar
        ):
            tau, r, pr, rp, ra, E, L = self._samples[ii]
            if tau is None:
                continue
            lift = self._momentum_matched_lift(tau, r, pr, rp, ra, L)
            self._lifts[ii] = lift
            Jrq, rA, pA = lift[0], lift[6], lift[7]
            EAs = numpy.max(
                0.5 * (pA**2 + L**2 / rA**2)
                - self._GM / (self._b + numpy.sqrt(self._b**2 + rA**2))
            )
            if EAs >= _BOUND_MARGIN * self._aAIinv._torus_E(Jrq, L):
                raise RuntimeError(
                    "The momentum-matched lift of the (E, L) = "
                    f"({E:g}, {L:g}) torus is not bound in the fitted "
                    "auxiliary: the anomaly map is under-resolved for it. "
                    f"Its turning points are {rp:g} and {ra:g}, a ratio of "
                    f"{ra / rp:.0f}, and the map needs a number of harmonics "
                    "of the order of that ratio, against the mm_npt = "
                    f"{self._npt} given; the most eccentric torus of a grid "
                    "is the one at its smallest angular momentum (Rmin) and "
                    "highest energy (Rinf). Raise mm_npt (and, with it, "
                    "mm_nta), or raise Rmin or lower Rinf"
                )
        return None

    def _fit_auxiliary(self, librating):
        """Fit the isochrone's (GM, b) to the potential's rotation curve
        over the sampled radial range (zero-point free, log residuals on
        geometrically spaced radii), by the converged least squares of
        _fit_isochrone_vc2"""
        rlo = min(smp[3] for smp in librating)
        rhi = max(smp[4] for smp in librating)
        rf = numpy.geomspace(rlo, rhi, 25)
        return _fit_isochrone_vc2(
            rf, numpy.log(vcirc(self._pot, rf, use_physical=False) ** 2)
        )

    # ---------- the momentum-matched map and its lift
    def _map(self, tau, Dm):
        """The anomaly map eta(tau) = tau + sum_m D_m sin(m tau), its
        derivative, and the sine matrix sin(m tau) it is built from"""
        mt = tau[:, None] * self._nforDm[None, :]
        smt = numpy.sin(mt)
        eta = tau + _sdot(smt, Dm)
        deta = 1.0 + _sdot(numpy.cos(mt), self._nforDm * Dm)
        return eta, deta, smt

    def _momentum_matched_lift(self, tau, r, pr, rp, ra, L):
        """The momentum-matched lift of one torus onto the auxiliary: match
        cumulative radial actions from pericentre, eta(tau) =
        A_A^{-1}(A_t(tau)); both cumulatives share the linear part J_r (the
        equal-action choice of the auxiliary torus), so eta - tau is
        periodic and, by time-reversal parity, a pure sine series. The lift
        is rebuilt from the TRUNCATED stored map cotangent-consistently
        (p^A = pi' p_r), so the reconstruction is the one the stored tables
        define; how far the truncation is from the exact map shows as the
        variation of the auxiliary action along the lifted torus, which
        _node_tables reports."""
        k = numpy.fft.fftfreq(self._ntau, d=1.0 / self._ntau)

        def _antider(f):
            fh = numpy.fft.fft(f - numpy.mean(f))
            ah = numpy.zeros_like(fh)
            ah[1:] = fh[1:] / (1j * k[1:])
            return numpy.real(numpy.fft.ifft(ah))

        drdtau_s = 0.5 * (ra - rp) * numpy.sin(tau)
        Jrq = float(numpy.mean(pr * drdtau_s))
        a, e = self._aAIinv._orbital_params(Jrq, L)
        ft = pr * drdtau_s  # target dA/dtau >= 0
        mt = numpy.mean(ft)
        qt_t = _antider(ft)
        At = mt * tau + qt_t - _spec_eval(_offset_spec_coeffs(qt_t), 0.0)[0]
        # the auxiliary's cumulative action is closed form; its secular rate
        # is J_r to round-off (the equal-action choice of the torus), and the
        # ratio of the two rates makes the linear parts coincide exactly
        AA = self._aAIinv._cumulative(a, e, tau)[0]
        scale = mt / (
            self._aAIinv._cumulative(a, e, 2.0 * numpy.pi)[0] / (2.0 * numpy.pi)
        )
        # pointwise Newton on the closed-form cumulative action, with the
        # closed-form flux as its derivative, started from the inverse of
        # the auxiliary's cumulative action interpolated between its grid
        # values (both cumulatives are monotone); the matching is monotone,
        # and a residual left by the iteration shows in the truncation
        # diagnostic of _node_tables
        eta_s = numpy.interp(At, scale * AA, tau)
        for _ in range(200):
            fres = scale * self._aAIinv._cumulative(a, e, eta_s)[0] - At
            fp = numpy.maximum(
                scale * self._aAIinv._flux_derivs(a, e, eta_s)[0], 1e-10 * mt
            )
            eta_s += numpy.clip(-fres / fp, -0.5, 0.5)
            if numpy.max(numpy.fabs(fres)) < 1e-13 * max(mt, 1e-10):
                break
        # the sine projection of the pointwise map, then the truncated map
        # and its cotangent lift
        Dm = _sine_project(eta_s - tau, self._nforDm)
        shift, dshift = _sine_series(self._ntau, Dm, self._nforDm)
        etat, detadtau = tau + shift, 1.0 + dshift
        rA, _, gA = self._aAIinv._profile(a, e, etat)
        pA = pr * drdtau_s / (gA * numpy.sin(etat) * detadtau)
        return Jrq, a, e, Dm, etat, detadtau, rA, pA

    def _map_slopes(self, tau, r, pr, rp, ra, E, L, Jrq, a, e, etat, OmR, Ompsi):
        """The derivatives of the anomaly-map coefficients with respect to E
        at fixed L and to L at fixed E, on one torus from that torus alone:
        the variation of the matching condition A^A(eta(tau); a, e) =
        A(tau; E, L) at fixed anomaly,

            f^A(eta) sum_m dD_m/dalpha sin(m tau)
                = dA/dalpha|_tau - dA^A/da a_alpha - dA^A/de e_alpha ,

        a LINEAR problem for the dD_m/dalpha with the vanishing flux f^A
        multiplying the unknowns, solved by Galerkin projection onto the
        sine basis (_sine_galerkin). The target's flux derivative at fixed
        tau goes through the moving turning points and is regular there
        (its numerator vanishes with p_r), so it integrates spectrally like
        the flux itself; the auxiliary's cumulative action varies through
        its torus parameters' closed-form chains, with the closed-form
        partials of _cumulative."""
        k = numpy.fft.fftfreq(self._ntau, d=1.0 / self._ntau)

        def _cum(f):
            # the integral from tau = 0 of a periodic f sampled on the
            # (half-offset) grid: its mean times tau plus the periodic part,
            # the latter returned on the grid together with its spectral
            # coefficients for evaluation elsewhere and its value at zero
            m = numpy.mean(f)
            fh = numpy.fft.fft(f - m)
            ah = numpy.zeros_like(fh)
            ah[1:] = fh[1:] / (1j * k[1:])
            q = numpy.real(numpy.fft.ifft(ah))
            cq = _offset_spec_coeffs(q)
            return m, q, cq, _spec_eval(cq, 0.0)[0]

        costau, sintau = numpy.cos(tau), numpy.sin(tau)
        drdtau = 0.5 * (ra - rp) * sintau
        dPhieff = -evaluateRforces(self._pot, r, 0.0, use_physical=False) - L**2 / r**3
        _, FAa, FAe = self._aAIinv._cumulative(a, e, etat)
        fAeta = self._aAIinv._flux_derivs(a, e, etat)[0]
        rhs = numpy.empty((self._ntau, 2))
        for kk, alpha in enumerate(("E", "L")):
            drp = self._turning_point_derivs(rp, E, L)[alpha == "L"]
            dra = self._turning_point_derivs(ra, E, L)[alpha == "L"]
            dr = drp * (1.0 + costau) / 2.0 + dra * (1.0 - costau) / 2.0
            num = (1.0 if alpha == "E" else -L / r**2) - dPhieff * dr
            dft = num / pr * drdtau + pr * 0.5 * (dra - drp) * sintau
            mt, qt, _, qt0 = _cum(dft)
            dA = mt * tau + qt - qt0
            if alpha == "E":
                _, _, da, de = self._aAIinv._orbital_param_chains(
                    Jrq, L, 1.0 / OmR, 0.0
                )
            else:
                _, _, da, de = self._aAIinv._orbital_param_chains(
                    Jrq, L, -Ompsi / OmR, 1.0
                )
            rhs[:, kk] = dA - FAa * da - FAe * de
        out = _sine_galerkin(fAeta, rhs, self._nforDm)
        return out[:, 0], out[:, 1]

    # ---------- the per-torus tables, computed (never fitted)
    def _node_tables(self, ii):
        """One torus (the ii-th sampled): from its lift onto the auxiliary,
        return the torus's action and frequencies by regular quadrature in
        the anomaly, the map's coefficients with their slopes in (E, L)
        from the torus alone, and the variation of the auxiliary action
        along the lifted torus (the map's truncation, which should be at
        round-off)"""
        tau, r, pr, rp, ra, E, L = self._samples[ii]
        if tau is None:
            # the circular orbit: no libration, the identity map, and the
            # epicycle and circular frequencies
            rc, kappa, Omc = self._circular(L)
            return {
                "jr": 0.0,
                "perr": 0.0,
                "rp": rc,
                "ra": rc,
                "Dm": numpy.zeros(self._npt),
                "dDm_dE": None,
                "dDm_dL": None,
                "OmR": kappa,
                "Ompsi": Omc,
            }
        Jrq, a, e, Dm, etat, detadtau, rA, pA = self._lifts[ii]
        # the truncated map's lift is not exactly the equal-action torus:
        # the auxiliary action varies along it by the truncation residual
        EA = 0.5 * (pA**2 + L**2 / rA**2) - self._GM / (
            self._b + numpy.sqrt(self._b**2 + rA**2)
        )
        perr = float(numpy.amax(numpy.fabs(self._aAIinv._torus_Jr(EA, L) - Jrq)) / Jrq)
        # the target's frequencies by regular quadrature in tau: dt/dtau is
        # periodic and finite (dr/dtau and p_r vanish together at the
        # turning points), so the trapezoid rule on the periodic grid is
        # spectrally accurate; T_r = 2 pi <dt/dtau> and the azimuth per
        # radial period is 2 pi <(L/r^2) dt/dtau>
        dtdtau = (
            0.5
            * (ra - rp)
            * numpy.fabs(numpy.sin(tau))
            / numpy.maximum(numpy.fabs(pr), 1e-300)
        )
        OmR = 1.0 / numpy.mean(dtdtau)
        Ompsi = numpy.mean(L / r**2 * dtdtau) * OmR
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
        }

    def _second_order_tables(self, ii, dDm_dE, dDm_dL, OmR, Ompsi):
        """The second-order tables of the ii-th explicit torus, from the
        torus alone, which make its one-node family quadratic: the second
        derivatives in (E, L) of the turning points (the level-set rule
        differentiated once more), of the map's coefficients (the second
        variation of the matching condition at fixed anomaly, solved by
        the same Galerkin projection as the slopes), and the derivatives
        of the frequencies (those of the period quadratures, regular in
        the anomaly like the slopes' integrands). Every target-side
        integrand is regular at the turning points: the numerators N_a =
        S_a + S_r dr/da|_tau and N_ab vanish there to second order in tau
        while p_r vanishes to first. Returns a dict in (E, L); the chain
        to (J_r, L) is _quadratic_node."""
        tau, r, pr, rp, ra, E, L = self._samples[ii]
        Jrq, a, e, Dm, etat, _, _, _ = self._lifts[ii]
        k = numpy.fft.fftfreq(self._ntau, d=1.0 / self._ntau)

        def _cum(f):
            m = numpy.mean(f)
            fh = numpy.fft.fft(f - m)
            ah = numpy.zeros_like(fh)
            ah[1:] = fh[1:] / (1j * k[1:])
            q = numpy.real(numpy.fft.ifft(ah))
            return m * tau + q - _spec_eval(_offset_spec_coeffs(q), 0.0)[0]

        pairs = ((0, 0), (0, 1), (1, 1))
        d1p = numpy.array(self._turning_point_derivs(rp, E, L))
        d1a = numpy.array(self._turning_point_derivs(ra, E, L))
        d2p = numpy.array(self._turning_point_derivs2(rp, E, L))
        d2a = numpy.array(self._turning_point_derivs2(ra, E, L))
        cost, sint = numpy.cos(tau), numpy.sin(tau)
        rtau = 0.5 * (ra - rp) * sint
        rc1, w1 = 0.5 * (d1p + d1a), 0.5 * (d1a - d1p)
        rc2, w2 = 0.5 * (d2p + d2a), 0.5 * (d2a - d2p)
        r1 = [rc1[i] - w1[i] * cost for i in range(2)]
        r2 = [rc2[kk] - w2[kk] * cost for kk in range(3)]
        P1 = -evaluateRforces(self._pot, r, 0.0, use_physical=False) - L**2 / r**3
        P2 = evaluateR2derivs(self._pot, r, 0.0, use_physical=False) + 3.0 * L**2 / r**4
        S_r, S_rr = -2.0 * P1, -2.0 * P2
        S_a = [2.0 + 0.0 * r, -2.0 * L / r**2]
        S_ra = [0.0 * r, 4.0 * L / r**3]
        S_ab = [0.0 * r, 0.0 * r, -2.0 / r**2]
        N1 = [S_a[i] + S_r * r1[i] for i in range(2)]
        A2 = []
        for kk, (i, j) in enumerate(pairs):
            N2 = (
                S_ab[kk]
                + S_ra[i] * r1[j]
                + S_ra[j] * r1[i]
                + S_rr * r1[i] * r1[j]
                + S_r * r2[kk]
            )
            A2.append(
                _cum(
                    (N2 * rtau + N1[i] * w1[j] * sint + N1[j] * w1[i] * sint)
                    / (2.0 * pr)
                    - N1[i] * N1[j] * rtau / (4.0 * pr**3)
                    + pr * w2[kk] * sint
                )
            )
        # the period quadratures P = <dt/dtau> and Q = <(L/r^2) dt/dtau> and
        # their derivatives; J_r(E, L) has J_E = P, J_L = -Q
        dtdtau = rtau / pr
        P, Q = numpy.mean(dtdtau), numpy.mean(L / r**2 * dtdtau)
        dP, dQ = [], []
        for i in range(2):
            dtd = w1[i] * sint / pr - rtau * N1[i] / (2.0 * pr**3)
            dP.append(numpy.mean(dtd))
            dQ.append(
                numpy.mean(
                    ((1.0 if i == 1 else 0.0) / r**2 - 2.0 * L * r1[i] / r**3) * dtdtau
                    + L / r**2 * dtd
                )
            )
        Jal, Lal = [P, -Q], [0.0, 1.0]
        J2 = [dP[0], dP[1], -dQ[1]]
        _, _, da, de = self._aAIinv._orbital_params2(Jrq, L)

        def chain2(d):
            dJ, dL, dJJ, dJL, dLL = d
            first = [dJ * Jal[i] + dL * Lal[i] for i in range(2)]
            second = [
                dJJ * Jal[i] * Jal[j]
                + dJL * (Jal[i] * Lal[j] + Jal[j] * Lal[i])
                + dLL * Lal[i] * Lal[j]
                + dJ * J2[kk]
                for kk, (i, j) in enumerate(pairs)
            ]
            return first, second

        a1, a2 = chain2(da)
        e1, e2 = chain2(de)
        _, AA_a, AA_e, AA_aa, AA_ae, AA_ee = self._aAIinv._cumulative2(a, e, etat)
        fA, fA_eta = self._aAIinv._flux_eta(a, e, etat)
        _, fA_a, fA_e = self._aAIinv._flux_derivs(a, e, etat)
        eta1 = [_sine_series(self._ntau, d, self._nforDm)[0] for d in (dDm_dE, dDm_dL)]
        rhs = numpy.empty((self._ntau, 3))
        for kk, (i, j) in enumerate(pairs):
            rhs[:, kk] = (
                A2[kk]
                - (
                    fA_eta * eta1[i] * eta1[j]
                    + fA_a * (a1[j] * eta1[i] + a1[i] * eta1[j])
                    + fA_e * (e1[j] * eta1[i] + e1[i] * eta1[j])
                )
                - (
                    AA_aa * a1[i] * a1[j]
                    + AA_ae * (a1[i] * e1[j] + a1[j] * e1[i])
                    + AA_ee * e1[i] * e1[j]
                )
                - AA_a * a2[kk]
                - AA_e * e2[kk]
            )
        D2 = _sine_galerkin(fA, rhs, self._nforDm)
        return {
            "d2rp": d2p,
            "d2ra": d2a,
            "d2Dm": D2,
            "dOmR": numpy.array([-dP[0] / P**2, -dP[1] / P**2]),
            "dOmpsi": numpy.array(
                [(dQ[0] * P - Q * dP[0]) / P**2, (dQ[1] * P - Q * dP[1]) / P**2]
            ),
        }

    def _quadratic_node(self, ii):
        """The ii-th explicit torus's one-node family to second order in
        the actions: the values, first and second derivatives in (J_r, L)
        of the turning points and the map's coefficients, and the
        frequencies with their derivatives (the second derivatives of E(J_r,
        L)), chained from the stored (E, L) tables through the torus's own
        frequencies, dE = Omega_r dJ_r + Omega_psi dL, and their derivatives"""
        OmR, OmL = self._OmRs[ii], self._Ompsis[ii]
        rp, ra = self._rps[ii], self._ras[ii]
        E, L = self._Es[ii], self._Ls[ii]
        # E(J_r, L): first derivatives are the frequencies, second
        # derivatives from the frequencies' (E, L) derivatives
        dOmR, dOmL = self._dOmRs[ii], self._dOmpsis[ii]
        E_JJ = dOmR[0] * OmR
        E_JL = dOmR[0] * OmL + dOmR[1]
        E_LL = dOmL[0] * OmL + dOmL[1]

        def chain(x1E, x1L, x2):
            # x(E, L) with partials (x_E, x_L) and (x_EE, x_EL, x_LL) to
            # (J_r, L) at fixed L and at fixed J_r
            xEE, xEL, xLL = x2
            return (
                (x1E * OmR, x1E * OmL + x1L),
                (
                    xEE * OmR**2 + x1E * E_JJ,
                    xEE * OmR * OmL + xEL * OmR + x1E * E_JL,
                    xEE * OmL**2 + 2.0 * xEL * OmL + xLL + x1E * E_LL,
                ),
            )

        drp = self._turning_point_derivs(rp, E, L)
        dra = self._turning_point_derivs(ra, E, L)
        out = {"jr": self._jrs[ii], "L": L, "rp": rp, "ra": ra, "Dm": self._Dms[ii]}
        out["drp"], out["d2rp"] = chain(drp[0], drp[1], self._d2rps[ii])
        out["dra"], out["d2ra"] = chain(dra[0], dra[1], self._d2ras[ii])
        dDE, dDL, D2 = self._dDm_dEs[ii], self._dDm_dLs[ii], self._d2Dms[ii]
        out["dDm"], out["d2Dm"] = chain(dDE, dDL, (D2[:, 0], D2[:, 1], D2[:, 2]))
        out["Om"], out["d2E"] = (OmR, OmL), (E_JJ, E_JL, E_LL)
        return out

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
        self._jrs = numpy.empty(ntori)
        self._rps = numpy.empty(ntori)
        self._ras = numpy.empty(ntori)
        self._Dms = numpy.empty((ntori, self._npt))
        self._dDm_dEs = numpy.zeros((ntori, self._npt))
        self._dDm_dLs = numpy.zeros((ntori, self._npt))
        self._OmRs = numpy.empty(ntori)
        self._Ompsis = numpy.empty(ntori)
        perr = numpy.empty(ntori)
        self._d2rps, self._d2ras = numpy.zeros((ntori, 3)), numpy.zeros((ntori, 3))
        self._d2Dms = numpy.zeros((ntori, self._npt, 3))
        self._dOmRs, self._dOmpsis = numpy.zeros((ntori, 2)), numpy.zeros((ntori, 2))
        for ii in _progress(range(ntori), "node tables", self._progressbar):
            node = self._node_tables(ii)
            if node["dDm_dE"] is not None:
                # the second-order tables make the one-node family
                # quadratic: an explicit torus then carries the true
                # derivatives of the map in the actions at its node
                so = self._second_order_tables(
                    ii, node["dDm_dE"], node["dDm_dL"], node["OmR"], node["Ompsi"]
                )
                self._d2rps[ii], self._d2ras[ii] = so["d2rp"], so["d2ra"]
                self._d2Dms[ii] = so["d2Dm"]
                self._dOmRs[ii], self._dOmpsis[ii] = so["dOmR"], so["dOmpsi"]
            self._jrs[ii] = node["jr"]
            self._rps[ii] = node["rp"]
            self._ras[ii] = node["ra"]
            self._Dms[ii] = node["Dm"]
            if node["dDm_dE"] is not None:
                # a circular torus has no slopes to store: it is evaluated
                # in closed form
                self._dDm_dEs[ii] = node["dDm_dE"]
                self._dDm_dLs[ii] = node["dDm_dL"]
            self._OmRs[ii] = node["OmR"]
            self._Ompsis[ii] = node["Ompsi"]
            perr[ii] = node["perr"]
        self._warn_unresolved(perr, self._Es, self._Ls)
        return None

    # ---------- the (E, L) interpolation grid
    def _setup_grid(self, Rmin, Rmax, Rinf, nE, nL):
        """Rectangular grid in (u, L): L between the circular angular
        momenta of Rmin and Rmax; the energies uniform, in x = u^2 with u
        uniform in [0, 1], in the logarithm of the apocentre of the radial
        orbit, r_0(E) with Phi(r_0) = E, between the circular orbit's energy
        and the potential at Rinf, the circular orbits themselves forming
        the bottom row: E(x, L) = Phi(r_0(E_c(L)) [Rinf/r_0(E_c(L))]^x).
        In the energy the spacing is quadratic at the circular edge, where
        the tables behave as sqrt(E - E_c), and for the logarithmic halo it
        is E = E_c + [E(Rinf) - E_c] x throughout; in a potential with an
        escape energy, where the apocentre of every torus has a pole, the
        spacing compresses toward the top so that the tables stay smooth in
        x up to it (in the energy itself the pole sits just beyond the top
        row, and no polynomial interpolant follows it from the last cell)"""
        if nE < 4 or nL < 4:
            raise ValueError("setup_interp=True requires nE >= 4 and nL >= 4")
        Lmin = Rmin * vcirc(self._pot, Rmin, use_physical=False)
        Lmax = Rmax * vcirc(self._pot, Rmax, use_physical=False)
        self._Lgrid = numpy.linspace(Lmin, Lmax, nL)
        self._us = numpy.linspace(0.0, 1.0, nE)
        self._Emax = self._Phi(Rinf)
        self._Ecs = numpy.array([self._Ec(L)[0] for L in self._Lgrid])
        if numpy.any(self._Ecs >= self._Emax):
            raise ValueError(
                "Rinf is too small: the grid's top energy lies below a "
                "circular orbit's; increase Rinf"
            )
        self._Rinf = Rinf
        self._r0c_cache = {}
        Etab = numpy.array(
            [[self._E_of_uL(u, L)[0] for L in self._Lgrid] for u in self._us]
        )
        self._E_tab = Etab
        self._Es = Etab.flatten()
        self._Ls = numpy.tile(self._Lgrid, nE)
        return None

    def _Ec(self, L):
        """Energy of the circular orbit of angular momentum L, and its
        derivative dE_c/dL = L / r_c^2 (the circular frequency)"""
        rc = self._rc(L)
        return self._Phi(rc) + L**2 / (2.0 * rc**2), L / rc**2

    def _circular(self, L):
        """The circular orbit of angular momentum L: its radius, epicycle
        frequency (the radial frequency in the limit J_r -> 0), and circular
        frequency"""
        rc = self._rc(L)
        return rc, epifreq(self._pot, rc, use_physical=False), L / rc**2

    def _is_circular(self, E, L):
        """Whether the (E, L) torus is the circular orbit, to round-off;
        raises when it lies below it"""
        Ec, _ = self._Ec(L)
        tol = 1e-12 * (1.0 + numpy.fabs(Ec))
        if E < Ec - tol:
            raise ValueError(
                f"No orbit exists at E = {E}, L = {L}: the energy lies below "
                "the circular orbit's"
            )
        return E < Ec + tol

    def _r0(self, E, rlo, rhi):
        """The apocentre of the radial orbit of energy E, Phi(r_0) = E,
        bracketed by [rlo, rhi]; for an array of energies, by bisection on
        all of them at once"""
        if numpy.ndim(E) == 0:
            return brentq(lambda r: self._Phi(r) - E, rlo, rhi, xtol=1e-14)
        lo, hi = numpy.broadcast_arrays(
            numpy.asarray(rlo, dtype="float"), numpy.asarray(rhi, dtype="float")
        )
        return _safeguarded_newton(
            lambda r: (self._Phi(r) - E, self._rdPhi(r) / r), lo, hi, 1e-14
        )

    def _r0c(self, L):
        """The apocentre of the radial orbit at the circular orbit's energy,
        r_0(E_c(L)) (above r_c, since E_c exceeds Phi(r_c) by the centrifugal
        term), and its derivative along L by the level-set rule, dr_0/dL =
        (dE_c/dL)/Phi'(r_0); cached for a single L, solved for all at once
        for an array"""
        if numpy.ndim(L) > 0:
            if numpy.size(L) == 1:  # a single torus: through the cache
                r0, g0 = self._r0c(float(numpy.ravel(L)[0]))
                return numpy.array([r0]), numpy.array([g0])
            Ec, _ = self._Ec(L)
            r0 = self._r0(Ec, self._rc(L), self._Rinf + 0.0 * L)
            return r0, self._rdPhi(r0)
        if L not in self._r0c_cache:
            Ec, dEc = self._Ec(L)
            r0 = self._r0(Ec, self._rc(L), self._Rinf)
            self._r0c_cache[L] = (r0, self._rdPhi(r0))
        return self._r0c_cache[L]

    def _rdPhi(self, r):
        """r Phi'(r), the derivative of the potential with respect to ln r"""
        return -r * evaluateRforces(self._pot, r, 0.0, use_physical=False)

    def _E_of_uL(self, u, L):
        """The grid's energy variable, analytic: E = Phi(r) with r =
        r_0(E_c(L)) [Rinf/r_0(E_c(L))]^x and x = u^2, uniform in the logarithm
        of the radial orbit's apocentre; returns E, dE/dx at fixed L, and
        dE/dL at fixed x (= at fixed u). The latter chains through r_0(E_c(L))
        by the level-set rule, d ln r_0/dL = (dE_c/dL)/[r_0 Phi'(r_0)], and
        is formed so that it is dE_c/dL exactly at the circular edge"""
        r0c, g0c = self._r0c(L)
        lnq = numpy.log(self._Rinf / r0c)
        r = r0c * numpy.exp(lnq * u**2)
        g = self._rdPhi(r)
        return (
            self._Phi(r),
            g * lnq,
            (1.0 - u**2) * self._Ec(L)[1] * (g / g0c),
        )

    def _turning_point_derivs(self, r, E, L):
        """Closed-form derivatives of a radial turning point (a root of
        E - Phi(r) - L^2 / 2 r^2) with respect to E at fixed L and to L at
        fixed E, by the level-set rule"""
        dPhieff = -evaluateRforces(self._pot, r, 0.0, use_physical=False) - L**2 / r**3
        return 1.0 / dPhieff, -(L / r**2) / dPhieff

    def _turning_point_derivs2(self, r, E, L):
        """Closed-form second derivatives (EE, EL, LL) of a radial turning
        point, the level-set rule differentiated once more: with S = 2(E -
        Phi) - L^2/r^2 vanishing identically along the family, S_r r_ab +
        S_rr r_a r_b + S_ra r_b + S_rb r_a + S_ab = 0"""
        P1 = -evaluateRforces(self._pot, r, 0.0, use_physical=False) - L**2 / r**3
        P2 = evaluateR2derivs(self._pot, r, 0.0, use_physical=False) + 3.0 * L**2 / r**4
        rE, rL = 1.0 / P1, -(L / r**2) / P1
        return (
            -P2 * rE**2 / P1,
            (-P2 * rE * rL + (2.0 * L / r**3) * rE) / P1,
            (-P2 * rL**2 + (4.0 * L / r**3) * rL - 1.0 / r**2) / P1,
        )

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
        for idx in _progress(range(nu * nLg), "node tables", self._progressbar):
            ii, jj = divmod(idx, nLg)
            if True:
                L = self._Lgrid[jj]
                node = self._node_tables(ii * nLg + jj)
                self._jr_tab[ii, jj] = node["jr"]
                self._OmR_tab[ii, jj] = node["OmR"]
                self._Ompsi_tab[ii, jj] = node["Ompsi"]
                self._sup_tab[ii, jj] = [node["rp"], node["ra"]]
                self._Dm_tab[ii, jj] = node["Dm"]
                perr[ii, jj] = node["perr"]
                E, dE_dx, dE_dL = self._E_of_uL(self._us[ii], L)
                dE_du = 2.0 * self._us[ii] * dE_dx
                self._jr_dx[ii, jj] = dE_dx / node["OmR"]
                self._jr_dL[ii, jj] = (dE_dL - node["Ompsi"]) / node["OmR"]
                if node["jr"] == 0.0:
                    # the circular edge: the turning points leave the circular
                    # radius linearly in u, at the rate the epicycle sets
                    # (E - E_c = kappa^2 w^2 / 2 with w the half-width, and
                    # E - E_c = dE/dx u^2 at the edge), and move with it
                    # along L (L^2 = r_c^3 Phi'(r_c) gives dr_c/dL = 2 L /
                    # [r_c^3 kappa^2]); the map's slopes come from the next
                    # two rows below
                    rc, kappa, _ = self._circular(L)
                    w1 = numpy.sqrt(2.0 * dE_dx) / kappa
                    drc = 2.0 * L / (rc**3 * kappa**2)
                    self._sup_du[ii, jj] = [-w1, w1]
                    self._sup_dL[ii, jj] = [drc, drc]
                    self._Dm_dL[ii, jj] = 0.0
                    continue
                for q, r in enumerate((node["rp"], node["ra"])):
                    dr_dE, dr_dL = self._turning_point_derivs(r, E, L)
                    self._sup_du[ii, jj, q] = dr_dE * dE_du
                    self._sup_dL[ii, jj, q] = dr_dE * dE_dL + dr_dL
                self._Dm_du[ii, jj] = node["dDm_dE"] * dE_du
                self._Dm_dL[ii, jj] = node["dDm_dE"] * dE_dL + node["dDm_dL"]
        # the map's coefficients vanish at the circular edge and grow
        # linearly in u out of it (the first anharmonic correction of the
        # radial libration is linear in its amplitude); their slope there is
        # that of the quartic through zero and the next two rows, as the 1D
        # family does at its harmonic bottom
        self._Dm_du[0] = slope_at_zero(
            self._us[1:3], self._Dm_tab[1:3], self._Dm_du[1:3]
        )
        self._warn_unresolved(
            perr.flatten(), self._E_tab.flatten(), numpy.tile(self._Lgrid, nu)
        )
        self._rebuild_interp()
        return None

    def _rebuild_interp(self):
        """(Re)build the interpolants from the stored tables: Hermite
        interpolants of J_r (in the normalized energy), of the turning
        points and of the anomaly-map coefficients (in u), quintic along
        the energy variable and cubic along L, all with their first
        partials exact at the nodes"""
        u, Lg = self._us, self._Lgrid
        self._jr_ip = HermiteFamily2D(u**2, Lg, self._jr_tab, self._jr_dx, self._jr_dL)
        self._sup_ip = HermiteFamily2D(u, Lg, self._sup_tab, self._sup_du, self._sup_dL)
        self._Dm_ip = HermiteFamily2D(u, Lg, self._Dm_tab, self._Dm_du, self._Dm_dL)
        return None

    # ---------- evaluation: the manifest chain
    def _check_L(self, L):
        """Raise for an angular momentum outside the family's grid (for an
        array, the first one outside it)"""
        L = numpy.atleast_1d(L)
        bad = (L < self._Lgrid[0]) | (L > self._Lgrid[-1])
        if numpy.any(bad):
            raise ValueError(
                f"L = {L[numpy.argmax(bad)]} outside the interpolation grid "
                f"[{self._Lgrid[0]}, {self._Lgrid[-1]}]"
            )
        return None

    def _interp_tables(self, jr, L):
        """Solve the implicit inverse label (u from (J_r, L) by root-finding
        on the stored J_r interpolant -- exact-in-the-family, so canonicity
        is untouched) and return the frequencies and the map's tables with
        their derivatives, all the interpolants' own, combined into the
        chains at fixed L and at fixed J_r that the evaluation needs, for
        all the (J_r, L) tori at once, with one row per torus"""
        jr, L = numpy.broadcast_arrays(
            numpy.atleast_1d(numpy.asarray(jr, dtype="float")),
            numpy.atleast_1d(numpy.asarray(L, dtype="float")),
        )
        self._check_L(L)
        ulo, uhi = self._us[0], self._us[-1]
        jlo = self._jr_ip.points(ulo**2 + 0.0 * L, L)
        jhi = self._jr_ip.points(uhi**2 + 0.0 * L, L)
        tol = 1e-12 * (1.0 + numpy.fabs(jr))
        bad = (jr < jlo - tol) | (jr > jhi + tol)
        if numpy.any(bad):
            ii = numpy.argmax(bad)
            raise ValueError(
                f"J_r = {jr[ii]} outside the interpolated family's range "
                f"[{jlo[ii]}, {jhi[ii]}] at L = {L[ii]}"
            )
        jr = numpy.clip(jr, jlo, jhi)  # the grid's own nodes, to round-off

        # J_r is monotone in u; the root is found in u rather than in the
        # interpolant's own variable x = u^2, because J_r ~ u^2 at the
        # circular edge and a tiny radial action must resolve to a small u,
        # not to zero: for a single torus by a bracketed scalar solve on the
        # interpolant (brentq returns an end point at which the residual
        # vanishes exactly, so the grid's edges need no case), for many
        # tori by safeguarded Newton steps on all of them at once, to
        # round-off, with the grid's edges exact
        if len(jr) == 1:
            u = numpy.array(
                [
                    brentq(
                        lambda uu: self._jr_ip(uu * uu, L[0])[0, 0] - jr[0],
                        ulo,
                        uhi,
                        xtol=1e-15,
                    )
                ]
            )
        else:

            def _label(uu):
                val, der = self._jr_ip.points_dx(uu * uu, L)
                return val - jr, 2.0 * uu * der

            u = _safeguarded_newton(
                _label,
                ulo + 0.0 * jr,
                uhi + 0.0 * jr,
                1e-15,
                x0=ulo + (uhi - ulo) * numpy.sqrt((jr - jlo) / (jhi - jlo)),
            )
            u = numpy.where(jr == jlo, ulo, numpy.where(jr == jhi, uhi, u))
        _, jr_dx, djr_dL = self._jr_ip.points_grad(u * u, L)
        djr_du = 2.0 * u * jr_dx
        _, dE_dx, dE_dL = self._E_of_uL(u, L)
        # chains at fixed L resp. fixed J_r, all from the stored
        # interpolants' own derivatives; the frequencies with the common
        # factor 2u of dE/du and dJ_r/du cancelled, so that they are finite
        # down to the circular edge
        OmR = dE_dx / jr_dx
        OmL = dE_dL - OmR * djr_dL
        sup, dsup_du, dsup_dL = self._sup_ip.points_grad(u, L)
        Dm, dDm_du, dDm_dL = self._Dm_ip.points_grad(u, L)
        cJ, cL = 1.0 / djr_du[:, None], (djr_dL / djr_du)[:, None]
        ptdata = {
            "sup": sup,
            "dsupJ": dsup_du * cJ,
            "dsupL": dsup_dL - dsup_du * cL,
            "Dm": Dm,
            "dDmJ": dDm_du * cJ,
            "dDmL": dDm_dL - dDm_du * cL,
        }
        return u, OmR, OmL, ptdata

    def _kernel(self, tau, a, e, Dm, rp, ra, chains):
        """Everything the evaluation needs at anomaly tau, in one pass and
        from the tables alone: the auxiliary anomaly eta(tau) and radial
        angle theta^A_r (the isochrone's mean-anomaly relation, closed
        form), the target's radius and radial momentum (the flux identity
        p_r dr/dtau = p^A dr^A/dtau, with sin(eta)/sin(tau) grouped so that
        every factor is regular at the turning points), and, for each chain
        alpha in chains (the J_r-chain at fixed L and the L-chain at fixed
        J_r, each given as (da, de, drp, dra, dDm)), the compensation
        p^A (dr^A/dalpha)|_tau - p_r (dr/dalpha)|_tau. Also returns
        d theta^A_r / d tau for the Newton solve of the angle relation.
        The torus's parameters and tables are either one set for all
        anomalies or one per anomaly (arrays along the first axis)."""
        eta, deta, smt = self._map(tau, Dm)
        se, ce = numpy.sin(eta), numpy.cos(eta)
        rA, pA, gA = self._aAIinv._profile(a, e, eta)
        kap = a * e / (a + self._b)
        thetaA = eta - kap * se
        dthetaA = (1.0 - kap * ce) * deta
        st, ct = numpy.sin(tau), numpy.cos(tau)
        r = rp * (1.0 + ct) / 2.0 + ra * (1.0 - ct) / 2.0
        # sin(eta)/sin(tau) -> eta'(tau) at the turning points
        sratio = numpy.where(
            numpy.fabs(st) > 1e-12, se / numpy.where(st == 0.0, 1.0, st), deta
        )
        pr = pA * gA * sratio * deta / (0.5 * (ra - rp))
        drA_da, drA_de = self._aAIinv._radius_partials(a, e, eta)
        comps = []
        for da, de, drp, dra, dDm in chains:
            drA = drA_da * da + drA_de * de + gA * se * _sdot(smt, dDm)
            dr = drp * (1.0 + ct) / 2.0 + dra * (1.0 - ct) / 2.0
            comps.append(pA * drA - pr * dr)
        return eta, thetaA, dthetaA, r, pr, comps

    def _tau_solve(self, thr, a, e, Dm, rp, ra, chainJ):
        """Newton solve of theta_r(tau) = theta^A_r(tau) + [compensation
        along the J_r-chain](tau) for the anomaly of each requested angle,
        on the derivative of the auxiliary angle alone (the compensation is
        a small correction), vectorized over the angles, each on its own
        torus (the parameters and tables have one row per angle, or one row
        for all), with the residual wrapped into (-pi, pi] so that an
        anomaly that crosses the pericentre converges to the equivalent
        angle, converged angles dropped from the iteration, a bracketed
        scalar fallback for any angle that does not converge within maxiter
        iterations, and a final Newton step on every angle that takes the
        tolerance to round-off"""
        tables = (a, e, Dm, rp, ra) + tuple(chainJ)
        n = len(thr)

        def _rows(idx):
            # those angles' rows of every per-angle table; a table shared by
            # all angles (one row) is kept as it is
            return tuple(
                t[idx] if numpy.ndim(t) >= 1 and numpy.shape(t)[0] == n and n > 1 else t
                for t in tables
            )

        def _f(x, tabs):
            _, thetaA, dthetaA, _, _, (compJ,) = self._kernel(
                x, *tabs[:5], chains=(tabs[5:],)
            )
            return thetaA + compJ, dthetaA

        x = numpy.array(thr, dtype="float")
        active = numpy.ones(n, dtype="bool")
        for _ in range(self._maxiter):
            idx = numpy.flatnonzero(active)
            f, fp = _f(x[idx], _rows(idx))
            f = (f - thr[idx] + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
            conv = numpy.fabs(f) < self._angle_tol
            x[idx] += numpy.clip(-f / fp, -0.5, 0.5) * ~conv
            active[idx[conv]] = False
            if not numpy.any(active):
                break
        else:
            # theta_r(tau) - tau is periodic and bounded, so [theta - s,
            # theta + s] with s beyond its extreme is a guaranteed bracket;
            # one angle at a time, on that angle's own torus
            xscan = numpy.linspace(0.0, 2.0 * numpy.pi, 256, endpoint=False)
            for ii in numpy.flatnonzero(active):
                thri = thr[ii]
                tabs = _rows(numpy.array([ii]))
                sc = numpy.max(numpy.fabs(_f(xscan, tabs)[0] - xscan)) + 0.1

                def _fi(xx):
                    return _f(numpy.array([xx]), tabs)[0][0] - thri

                x[ii] = brentq(_fi, thri - sc, thri + sc, xtol=1e-15, maxiter=200)
        # one more Newton step on every angle: the loop stops at angle_tol,
        # and the quadratic step takes that to round-off, so that the returned
        # anomaly does not depend on the tolerance
        f, fp = _f(x, tables)
        f = (f - thr + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
        return x - f / fp

    def _unlift(self, out, r, pr):
        """Undo the lift: the auxiliary-chart reconstruction (R, v_R, v_T,
        z, v_z, phi) becomes the target's by replacing the radius and the
        radial velocity with the kernel's r and p_r; the position direction,
        the plane, and the azimuth are untouched, and the tangential speed
        rescales to keep |L| = r x v exact"""
        R, vR, vT, z, vz, phi = out
        rA = _ns(R, z).sqrt(R**2 + z**2)
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

    def _match_node(self, jr, L):
        """Locate the explicit torus (tori, for arrays) with actions (J_r, L)"""
        scalar = numpy.ndim(jr) == 0 and numpy.ndim(L) == 0
        jr, L = numpy.broadcast_arrays(numpy.atleast_1d(jr), numpy.atleast_1d(L))
        dev = numpy.fabs(self._jrs[None, :] - jr[:, None]) + numpy.fabs(
            self._Ls[None, :] - L[:, None]
        )
        ii = numpy.argmin(dev, axis=1)
        bad = dev[numpy.arange(len(ii)), ii] > 1e-8 * (
            1.0 + numpy.fabs(jr) + numpy.fabs(L)
        )
        if numpy.any(bad):
            kk = numpy.argmax(bad)
            raise ValueError(
                f"(J_r, L) = ({jr[kk]}, {L[kk]}) is not one of the set-up tori; "
                "discrete mode evaluates the stored tori only (use "
                "setup_interp=True to interpolate)"
            )
        return ii[0] if scalar else ii

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
        xp = self._namespace(get_namespace(E, L))
        if xp is not numpy:
            return self._bk_Jr(xp, E, L)
        if not self._interp:
            dev = numpy.fabs(self._Es - E) + numpy.fabs(self._Ls - L)
            ii = numpy.argmin(dev)
            if dev[ii] > 1e-10 * (1.0 + numpy.fabs(E) + numpy.fabs(L)):
                raise ValueError(
                    f"(E, L) = ({E}, {L}) is not one of the set-up tori; to use "
                    "interpolation, initialize with setup_interp=True"
                )
            return self._jrs[ii]
        self._check_L(L)
        Ec, _ = self._Ec(L)
        if E < Ec - 1e-12 * (self._Emax - Ec) or E > self._Emax:
            raise ValueError(
                f"E = {E} outside the interpolation grid at L = {L}: "
                f"[{Ec}, {self._Emax}]"
            )
        # the normalized energy from the radial orbit's apocentre at E; the
        # circular orbit itself, to round-off, is the bottom of the grid
        if E <= Ec + 1e-12 * (1.0 + numpy.fabs(Ec)):
            return self._jr_ip(0.0, L)[0, 0]
        r0c, _ = self._r0c(L)
        r0 = self._r0(min(E, self._Emax), r0c, self._Rinf)
        x = numpy.log(r0 / r0c) / numpy.log(self._Rinf / r0c)
        return self._jr_ip(min(x, 1.0), L)[0, 0]

    def _xvFreqs(self, jr, jphi, jz, angler, anglephi, anglez, **kwargs):
        """(J, theta) -> (x, v) and the frequencies, for actions and angles
        that broadcast against each other by numpy's rules (scalar actions
        with arrays of angles: one torus at those angles; arrays of actions
        with scalar angles: that angle on every torus; equal shapes: paired
        element by element, each star on its own torus at its own angle;
        actions of shape (n,1) with angles of shape (m,): every angle on
        every torus as (n,m) arrays): the tables
        are read once per distinct torus; for each requested radial angle
        the anomaly is solved for, the two companion angles are shifted by
        the map's L-compensation, the full 3-D reconstruction is delegated
        to the analytic isochrone inverse at the kernel's eccentric anomaly
        (L^A = L, so the plane geometry is the auxiliary's own), and the
        lift is undone; a single torus returns its frequencies as floats"""
        xp = self._namespace(get_namespace(jr, jphi, jz, angler, anglephi, anglez))
        if xp is not numpy:
            return self._bk_xvFreqs(xp, jr, jphi, jz, angler, anglephi, anglez)
        scalar = numpy.ndim(jr) == 0 and numpy.ndim(jphi) == 0 and numpy.ndim(jz) == 0
        jr, jphi, jz, angler, anglephi, anglez = numpy.broadcast_arrays(
            *(
                numpy.atleast_1d(numpy.asarray(t, dtype="float"))
                for t in (jr, jphi, jz, angler, anglephi, anglez)
            )
        )
        shape = jr.shape
        out = numpy.stack(
            _chunked(
                self._xvFreqs_points,
                jr.size,
                *(
                    t.ravel()
                    for t in (jr, jphi, jz, angler % (2.0 * numpy.pi), anglephi, anglez)
                ),
            )
        ).reshape((9,) + shape)
        if scalar:
            return (*out[:6], *(float(o.flat[0]) for o in out[6:]))
        return tuple(out)

    def _xvFreqs_points(self, jr, jphi, jz, thr, anglephi, anglez):
        """The evaluation at flat arrays of actions and angles of one length:
        the circular orbits by closed form, the librating tori through the
        kernel with one row of tables per angle"""
        L = jz + numpy.fabs(jphi)
        if self._interp:
            self._check_L(L)
        else:
            jr = self._jrs[self._match_node(jr, L)]  # the stored tori, to round-off
        out = numpy.empty((9, len(jr)))
        circ = jr == 0.0
        if numpy.any(circ):
            # the circular orbit: the point sits at the circular radius with
            # no radial motion, and the requested radial angle is the
            # epicyclic phase, which does not enter. The plane geometry is
            # the auxiliary's at the same (L, J_phi) and in-plane angle,
            # which the isochrone inverse gives exactly at the pericentre
            # of any of its librating tori (the in-plane angle equals the
            # angle variable there); its own circular orbit is degenerate
            # for it, so a moderately eccentric auxiliary torus is used and
            # the lift undone with the circular radius
            Lc = L[circ]
            rc, OmR, OmL = self._circular(Lc)
            out[:6, circ] = self._unlift(
                self._aAIinv._xvFreqs_from_eta(
                    0.1 * Lc,
                    jphi[circ],
                    jz[circ],
                    0.0 * Lc,
                    anglephi[circ],
                    anglez[circ],
                )[:6],
                rc,
                0.0 * Lc,
            )
            out[6, circ], out[7, circ], out[8, circ] = (
                OmR,
                numpy.sign(jphi[circ]) * OmL,
                OmL,
            )
        lib = ~circ
        if numpy.any(lib):
            jrl, Ll = jr[lib], L[lib]
            # the tables once per distinct torus, then one row per point
            tori, inv = numpy.unique(
                numpy.stack([jrl, Ll], axis=1), axis=0, return_inverse=True
            )
            inv = inv.ravel()
            jru, Lu = tori[:, 0], tori[:, 1]
            if self._interp:
                _, OmRu, OmLu, pt = self._interp_tables(jru, Lu)
                Dmu, dDmJu, dDmLu = pt["Dm"], pt["dDmJ"], pt["dDmL"]
                (rpu, rau), (drpJu, draJu), (drpLu, draLu) = (
                    pt["sup"].T,
                    pt["dsupJ"].T,
                    pt["dsupL"].T,
                )
            else:
                # an explicit torus is a one-node family: its own values and
                # exact slopes stand in for the interpolants', the slopes in
                # (E, L) chained to J_r at fixed L and to L at fixed J_r
                # through the torus's frequencies (dE = Omega_r dJ_r +
                # Omega_psi dL)
                ii = self._match_node(jru, Lu)
                OmRu, OmLu = self._OmRs[ii], self._Ompsis[ii]
                Dmu, rpu, rau = self._Dms[ii], self._rps[ii], self._ras[ii]
                dDmJu = self._dDm_dEs[ii] * OmRu[:, None]
                dDmLu = self._dDm_dLs[ii] + self._dDm_dEs[ii] * OmLu[:, None]
                drpE, drpLE = self._turning_point_derivs(rpu, self._Es[ii], Lu)
                draE, draLE = self._turning_point_derivs(rau, self._Es[ii], Lu)
                drpJu, draJu = drpE * OmRu, draE * OmRu
                drpLu, draLu = drpLE + drpE * OmLu, draLE + draE * OmLu
            a, e = self._aAIinv._orbital_params(jru, Lu)
            # the J_r-chain at fixed L compensates theta_r, the L-chain at
            # fixed J_r the psi-angles, both through the auxiliary torus's
            # parameters, the turning points, and the map's coefficients
            _, _, daJ, deJ = self._aAIinv._orbital_param_chains(jru, Lu, 1.0, 0.0)
            _, _, daL, deL = self._aAIinv._orbital_param_chains(jru, Lu, 0.0, 1.0)
            # one row of tables per angle, unless all angles share one torus,
            # whose single row then broadcasts
            rows = (lambda t: t) if len(jru) == 1 else (lambda t: t[inv])
            a, e, Dm, rp, ra = (rows(t) for t in (a, e, Dmu, rpu, rau))
            chainJ = tuple(rows(t) for t in (daJ, deJ, drpJu, draJu, dDmJu))
            chainL = tuple(rows(t) for t in (daL, deL, drpLu, draLu, dDmLu))
            taus = self._tau_solve(thr[lib], a, e, Dm, rp, ra, chainJ)
            eta, _, _, r, pr, (Delta,) = self._kernel(
                taus, a, e, Dm, rp, ra, chains=(chainL,)
            )
            # the three-dimensional reconstruction on the auxiliary torus, in
            # one vectorized call at the kernel's eccentric anomaly (the
            # auxiliary has the requested actions, and its radial angle
            # follows from eta, so Kepler's equation is not solved again),
            # then the lift undone with the kernel's radius and radial
            # momentum
            out[:6, lib] = self._unlift(
                self._aAIinv._xvFreqs_from_eta(
                    jrl,
                    jphi[lib],
                    jz[lib],
                    eta,
                    anglephi[lib] - numpy.sign(jphi[lib]) * Delta,
                    anglez[lib] - Delta,
                )[:6],
                r,
                pr,
            )
            out[6, lib], out[7, lib], out[8, lib] = (
                OmRu[inv],
                numpy.sign(jphi[lib]) * OmLu[inv],
                OmLu[inv],
            )
        return tuple(out)

    def _Freqs(self, jr, jphi, jz, **kwargs):
        """Frequencies of the (J_r, L) tori, for actions that broadcast
        against each other: in interpolation mode these are the stored
        energy interpolant's own derivatives through the label chain (the
        integrator contract); for explicit tori the tori's own quadrature
        values; floats for a single torus"""
        xp = self._namespace(get_namespace(jr, jphi, jz))
        if xp is not numpy:
            return self._bk_Freqs(xp, jr, jphi, jz)
        scalar = numpy.ndim(jr) == 0 and numpy.ndim(jphi) == 0 and numpy.ndim(jz) == 0
        jr, jphi, jz = numpy.broadcast_arrays(
            *(numpy.atleast_1d(numpy.asarray(t, dtype="float")) for t in (jr, jphi, jz))
        )
        shape = jr.shape
        jr, jphi, jz = (t.ravel() for t in (jr, jphi, jz))
        L = jz + numpy.fabs(jphi)
        if self._interp:
            self._check_L(L)
        else:
            ii = self._match_node(jr, L)
            jr = self._jrs[ii]
        OmR, OmL = numpy.empty(len(jr)), numpy.empty(len(jr))
        circ = jr == 0.0
        if numpy.any(circ):
            _, OmR[circ], OmL[circ] = self._circular(L[circ])
        lib = ~circ
        if numpy.any(lib):
            if self._interp:
                _, OmR[lib], OmL[lib], _ = self._interp_tables(jr[lib], L[lib])
            else:
                OmR[lib], OmL[lib] = self._OmRs[ii[lib]], self._Ompsis[ii[lib]]
        OmR, OmL = OmR.reshape(shape), OmL.reshape(shape)
        sgn = numpy.sign(jphi).reshape(shape)
        if scalar:
            return (
                float(OmR.flat[0]),
                float(sgn.flat[0] * OmL.flat[0]),
                float(OmL.flat[0]),
            )
        return (OmR, sgn * OmL, OmL)

    # ---------- the family on a backend (jax/torch)
    # The construction above for all tori at once, in the namespace of a
    # forced backend or of the potential's jax/torch parameters, so that the
    # tables -- and through them the map -- are differentiable in those
    # parameters; and the evaluation natively in jax/torch, differentiable in
    # the actions and the angles as well. The auxiliary is fitted (on the
    # device under a trace) and frozen (any auxiliary of the torus's actions
    # gives a canonical map).
    # The root-finds go through the backend brentq, whose final Newton step
    # carries the implicit-function derivative (the level-set rule of the
    # turning points, 1/Phi' of the apocentre variable, 1/(dJ_r/du) of the
    # label). The checks and the warnings run when the values are concrete.
    def _bk_Phi(self, r):
        xp = get_namespace(r)
        flat = xp.reshape(r, (-1,))
        return xp.reshape(_evaluatePotentials(self._pot, flat, 0.0 * flat), r.shape)

    def _bk_dPhi(self, r):
        """dPhi/dr"""
        xp = get_namespace(r)
        flat = xp.reshape(r, (-1,))
        return -xp.reshape(_evaluateRforces(self._pot, flat, 0.0 * flat), r.shape)

    def _bk_rc(self, L):
        return rl(self._pot, L, use_physical=False)

    def _bk_rc_eval(self, L):
        """The circular radius at the angular momentum L (0-d) for an
        evaluation: in an interpolated family the circular condition r^3
        Phi'(r) = L^2 solved by the backend brentq within the grid's own
        range of circular radii (rl's bracket search would be most of an
        eager evaluation), by Newton steps from the radii interpolated
        between the grid's under a trace (where a bisection would take ~45
        sequential halvings); for explicit tori rl"""
        xp = get_namespace(L)
        if not self._interp:
            if L.ndim:
                return self._bk_rc(L)
            return xp.reshape(self._bk_rc(xp.reshape(L, (1,))), ())
        if self._bk is not None:
            rg, Lg = stop_gradient(self._bk["rcs"]), stop_gradient(self._Lgrid_bk)
        else:
            if getattr(self, "_rcs_grid", None) is None:
                self._rcs_grid = numpy.array([self._rc(Lg) for Lg in self._Lgrid])
            rg, Lg = self._rcs_grid, self._Lgrid
        one = xp.reshape(0.0 * L + 1.0, (-1,))
        L1 = xp.reshape(L, (-1,))
        guess = None
        if L.ndim or not has_concrete_truth_value(L):
            dev = device_of(L)
            guess = interp_linear(
                xp,
                asarray_on_device(xp, Lg, dev),
                asarray_on_device(xp, rg, dev),
                stop_gradient(L1),
                extrapolate="clip",
            )
        out = _backend_brentq(
            lambda r, LL: r**3 * self._bk_dPhi(r) - LL**2,
            one * (0.99 * rg[0]),
            one * (1.01 * rg[-1]),
            args=(L1,),
            xtol=1e-14,
            elementwise=True,
            guess=guess,
            newton_steps=5,
        )
        return out if L.ndim else out[0]

    def _init_backend(self, xp, Rmin, Rmax, Rinf, nE, nL, auxiliary):
        """The construction (as _setup_grid, _sample_all, _setup_auxiliary and
        _setup_tori do it) natively in xp, every torus at once; the circular
        tori go through it with a small libration in their place, which the
        circular values then replace"""
        # on the device of the potential's backend parameters, if any
        from ..backend._input import _grad_param

        par = _grad_param(self._pot)
        probe = self._bk_Phi(
            asarray_on_device(
                xp, numpy.ones(1), None if par is None else device_of(par)
            )
        )
        dev = device_of(probe)
        concrete = has_concrete_truth_value(probe[0])

        def f64(a):
            return asarray_on_device(xp, numpy.asarray(a, dtype="float"), dev)

        cat = getattr(xp, "concat", None) or xp.concatenate
        Phi, dPhi = self._bk_Phi, self._bk_dPhi
        ntau, npt, ms = self._ntau, self._npt, self._nforDm
        if not self._interp:
            Es, Ls = f64(self._Es), f64(self._Ls)
            rc = self._bk_rc(Ls)
            Ec = Phi(rc) + Ls**2 / (2.0 * rc**2)
            tol = 1e-12 * (1.0 + xp.abs(Ec))
            if concrete:
                below = numpy.flatnonzero(as_numpy(stop_gradient(Es < Ec - tol)))
                if len(below):
                    raise ValueError(
                        f"No orbit exists at E = {self._Es[below[0]]}, "
                        f"L = {self._Ls[below[0]]}: the energy lies below the "
                        "circular orbit's"
                    )
            circ = Es < Ec + tol
        else:
            if nE < 4 or nL < 4:
                raise ValueError("setup_interp=True requires nE >= 4 and nL >= 4")
            Lmin, Lmax = f64([Rmin, Rmax]) * vcirc(
                self._pot, f64([Rmin, Rmax]), use_physical=False
            )
            # as numpy.linspace
            Lgrid = cat(
                [
                    f64(numpy.arange(nL - 1)) * ((Lmax - Lmin) / (nL - 1)) + Lmin,
                    xp.reshape(Lmax, (1,)),
                ]
            )
            us = numpy.linspace(0.0, 1.0, nE)
            Emax = Phi(f64([Rinf]))[0]
            rcs = self._bk_rc(Lgrid)
            Ecs = Phi(rcs) + Lgrid**2 / (2.0 * rcs**2)
            if concrete and concretely_true(xp.any(Ecs >= Emax)):
                raise ValueError(
                    "Rinf is too small: the grid's top energy lies below a "
                    "circular orbit's; increase Rinf"
                )
            r0c = _backend_brentq(
                lambda r, E: Phi(r) - E,
                rcs,
                f64(numpy.full(nL, Rinf)),
                args=(Ecs,),
                xtol=1e-14,
            )
            g0c = r0c * dPhi(r0c)
            lnq = xp.log(Rinf / r0c)
            U = f64(us)[:, None]
            rgrid = r0c[None, :] * xp.exp(lnq[None, :] * U**2)
            g = rgrid * dPhi(rgrid)
            Etab = Phi(rgrid)
            dE_dx = g * lnq[None, :]
            dE_dL = (1.0 - U**2) * (Lgrid / rcs**2)[None, :] * (g / g0c[None, :])
            Es = xp.reshape(Etab, (-1,))
            Ls = xp.reshape(Lgrid[None, :] + 0.0 * Etab, (-1,))
            rc = xp.reshape(rcs[None, :] + 0.0 * Etab, (-1,))
            Ec = xp.reshape(Ecs[None, :] + 0.0 * Etab, (-1,))
            # the bottom row is the circular edge, every other node librates
            circ = f64(numpy.repeat(us == 0.0, nL)) > 0.5
        n = Es.shape[0]
        lib = ~circ
        kappa = epifreq(self._pot, rc, use_physical=False)
        # a circular torus goes through as a libration of a hundredth of its
        # radius, whose values the circular ones replace below
        Es_s = xp.where(circ, Ec + 0.5 * (0.01 * rc * kappa) ** 2, Es)

        def pr2(r, E, L):
            return 2.0 * (E - Phi(r)) - L**2 / r**2

        # the turning points: the bracket expanded from the circular radius
        # by factors of 1.3 (as _turning_points), then brentq
        rlo = iterate_bracket(
            lambda r: xp.where((pr2(r, Es_s, Ls) > 0.0) & (r > 1e-12), r / 1.3, r),
            rc,
            160,
        )
        rhi = iterate_bracket(
            lambda r: xp.where((pr2(r, Es_s, Ls) > 0.0) & (r < 1e12), r * 1.3, r),
            rc,
            160,
        )
        unbound = lib & (pr2(rhi, Es_s, Ls) > 0.0)
        if concrete:
            bad = numpy.flatnonzero(as_numpy(stop_gradient(unbound)))
            if len(bad):
                E, L = (float(as_numpy(stop_gradient(t[bad[0]]))) for t in (Es, Ls))
                raise ValueError(
                    f"No bound radial libration exists at (E, L) = ({E:g}, {L:g}): "
                    "the torus is unbound"
                )
        rhi = xp.where(unbound, 2.0 * rc, rhi)
        rp = _backend_brentq(pr2, rlo, rc, args=(Es_s, Ls), xtol=1e-12)
        ra = _backend_brentq(pr2, rc, rhi, args=(Es_s, Ls), xtol=1e-12)
        # the samples on the offset grid
        tau_np = 2.0 * numpy.pi * (numpy.arange(ntau) + 0.5) / ntau
        tau, st, ct = f64(tau_np), f64(numpy.sin(tau_np)), f64(numpy.cos(tau_np))
        sgn = f64(numpy.where(tau_np < numpy.pi, 1.0, -1.0))
        r = 0.5 * (ra + rp)[:, None] - 0.5 * (ra - rp)[:, None] * ct
        p2 = pr2(r, Es_s[:, None], Ls[:, None])
        pos = p2 > 0.0
        pr = sgn * xp.where(pos, xp.sqrt(xp.where(pos, p2, 1.0)), 0.0)
        # the target's frequencies by regular quadrature (as _node_tables)
        dtdtau = (
            0.5 * (ra - rp)[:, None] * xp.abs(st) / xp.clip(xp.abs(pr), 1e-300, None)
        )
        mdt = xp.mean(dtdtau, axis=-1)
        mpsi = xp.mean(Ls[:, None] / r**2 * dtdtau, axis=-1)
        OmR = 1.0 / mdt
        Ompsi = mpsi * OmR
        # the auxiliary (as _setup_auxiliary and _fit_auxiliary)
        if auxiliary is not None:
            if not isinstance(auxiliary, IsochronePotential):
                raise TypeError("auxiliary= has to be an IsochronePotential")
            self._GM, self._b = auxiliary._amp, auxiliary.b
        else:
            self._GM, self._b = self._bk_fit_auxiliary(
                xp, f64, concrete, lib, Ls, rc, rp, ra
            )
        self.auxiliary = (
            auxiliary
            if auxiliary is not None
            else IsochronePotential(amp=self._GM, b=self._b)
        )
        self._aAIinv = actionAngleIsochroneInverse(ip=self.auxiliary)
        # the momentum-matched lift (as _momentum_matched_lift)
        drdtau_s = 0.5 * (ra - rp)[:, None] * st
        ft = pr * drdtau_s
        Jrq = xp.mean(ft, axis=-1)
        a, e = self._aAIinv._orbital_params(Jrq, Ls)
        qt = _antiderivative_xp(xp, ft)
        At = (
            Jrq[:, None] * tau
            + qt
            - _spec_eval_xp(xp, _offset_spec_coeffs_xp(xp, qt), f64([0.0]))
        )
        ac, ec = a[:, None], e[:, None]
        scale = Jrq / (
            self._aAIinv._cumulative(a, e, f64(2.0 * numpy.pi) + 0.0 * a)[0]
            / (2.0 * numpy.pi)
        )
        sc = scale[:, None]

        def resid(x, args=(ac, ec, sc, At)):
            a_, e_, s_, A_ = args
            return s_ * self._aAIinv._cumulative(a_, e_, x)[0] - A_

        def slope(x, args=(ac, ec, sc, Jrq)):
            a_, e_, s_, J_ = args
            return xp.maximum(
                s_ * self._aAIinv._flux_derivs(a_, e_, x)[0], 1e-10 * J_[:, None]
            )

        sg = tuple(stop_gradient(t) for t in (ac, ec, sc, At, Jrq))
        eta = _bk_matching(
            xp,
            resid,
            slope,
            lambda x: resid(x, sg[:4]),
            lambda x: slope(x, (*sg[:3], sg[4])),
            stop_gradient(tau + 0.0 * At),
            sg[4],
        )
        Dm = _sine_project_xp(xp, eta - tau, ms)
        shift, dshift = _sine_series_xp(xp, ntau, Dm, ms)
        etat, detadtau = tau + shift, 1.0 + dshift
        rA, _, gA = self._aAIinv._profile(ac, ec, etat)
        pA = pr * drdtau_s / (gA * xp.sin(etat) * detadtau)
        EA = 0.5 * (pA**2 + Ls[:, None] ** 2 / rA**2) - self._GM / (
            self._b + xp.sqrt(self._b**2 + rA**2)
        )
        # the bound-lift guard (as _setup_auxiliary)
        unbound_lift = lib & (
            xp.max(EA, axis=-1) >= _BOUND_MARGIN * self._aAIinv._torus_E(Jrq, Ls)
        )
        if concrete:
            bad = numpy.flatnonzero(as_numpy(stop_gradient(unbound_lift)))
            if len(bad):
                E, L, rpb, rab = (
                    float(as_numpy(stop_gradient(t[bad[0]]))) for t in (Es, Ls, rp, ra)
                )
                raise RuntimeError(
                    "The momentum-matched lift of the (E, L) = "
                    f"({E:g}, {L:g}) torus is not bound in the fitted "
                    "auxiliary: the anomaly map is under-resolved for it. "
                    f"Its turning points are {rpb:g} and {rab:g}, a ratio of "
                    f"{rab / rpb:.0f}, and the map needs a number of harmonics "
                    "of the order of that ratio, against the mm_npt = "
                    f"{self._npt} given; the most eccentric torus of a grid "
                    "is the one at its smallest angular momentum (Rmin) and "
                    "highest energy (Rinf). Raise mm_npt (and, with it, "
                    "mm_nta), or raise Rmin or lower Rinf"
                )
        # the map's slopes in (E, L) from each torus alone (as _map_slopes)
        dPeff = dPhi(r) - Ls[:, None] ** 2 / r**3
        _, FAa, FAe = self._aAIinv._cumulative(ac, ec, etat)
        fAeta = self._aAIinv._flux_derivs(ac, ec, etat)[0]
        dtp = [self._bk_turning_point_derivs(t, Es_s, Ls) for t in (rp, ra)]
        rhs = []
        for kk in (0, 1):  # E, L
            drp, dra = dtp[0][kk][:, None], dtp[1][kk][:, None]
            dr = drp * (1.0 + ct) / 2.0 + dra * (1.0 - ct) / 2.0
            num = (1.0 if kk == 0 else -Ls[:, None] / r**2) - dPeff * dr
            dft = (
                num / pr * (0.5 * (ra - rp)[:, None] * st) + pr * 0.5 * (dra - drp) * st
            )
            qd = _antiderivative_xp(xp, dft)
            dA = (
                xp.mean(dft, axis=-1, keepdims=True) * tau
                + qd
                - _spec_eval_xp(xp, _offset_spec_coeffs_xp(xp, qd), f64([0.0]))
            )
            if kk == 0:
                _, _, da, de = self._aAIinv._orbital_param_chains(
                    Jrq, Ls, 1.0 / OmR, 0.0 * Jrq
                )
            else:
                _, _, da, de = self._aAIinv._orbital_param_chains(
                    Jrq, Ls, -Ompsi / OmR, 1.0 + 0.0 * Jrq
                )
            rhs.append(dA - FAa * da[:, None] - FAe * de[:, None])
        slopes = _sine_galerkin_xp(xp, fAeta, xp.stack(rhs, axis=-1), ms)
        # the circular tori: no libration, the identity map, and the epicycle
        # and circular frequencies (as _node_tables)
        Omc = Ls / rc**2
        cc = circ[:, None]
        nanb = xp.where(unbound | unbound_lift, xp.nan, 0.0)
        jr = xp.where(circ, 0.0, Jrq) + nanb
        rp = xp.where(circ, rc, rp)
        ra = xp.where(circ, rc, ra)
        Dm = xp.where(cc, 0.0, Dm)
        dDm_dE = xp.where(cc, 0.0, slopes[..., 0])
        dDm_dL = xp.where(cc, 0.0, slopes[..., 1])
        OmR = xp.where(circ, kappa, OmR)
        Ompsi = xp.where(circ, Omc, Ompsi)
        perr = None
        if concrete:
            # the truncated map's variation of the auxiliary action along the
            # lifted tori (as _node_tables), and the warning
            perr = as_numpy(
                stop_gradient(
                    xp.where(
                        circ,
                        0.0,
                        xp.max(
                            xp.abs(
                                self._aAIinv._torus_Jr(EA, Ls[:, None]) - Jrq[:, None]
                            ),
                            axis=-1,
                        )
                        / xp.where(circ, 1.0, Jrq),
                    )
                )
            )
        tonp = lambda t: numpy.array(as_numpy(stop_gradient(t)))  # noqa: E731
        if not self._interp:
            # the second-order tables of the quadratic one-node family (as
            # _second_order_tables), zero for a circular torus
            so = self._bk_second_order_tables(
                xp,
                tau,
                st,
                ct,
                r,
                pr,
                rp,
                ra,
                Es_s,
                Ls,
                Jrq,
                a,
                e,
                etat,
                slopes[..., 0],
                slopes[..., 1],
            )
            d2rps, d2ras, d2Dms, dOmRs, dOmpsis = (
                xp.where(xp.reshape(circ, (-1,) + (1,) * (t.ndim - 1)), 0.0, t)
                for t in so
            )
            self._bk = {
                "xp": xp,
                "dev": dev,
                "Es": Es,
                "Ls": Ls,
                "d2rps": d2rps,
                "d2ras": d2ras,
                "d2Dms": d2Dms,
                "dOmRs": dOmRs,
                "dOmpsis": dOmpsis,
                "jrs": jr,
                "rps": rp,
                "ras": ra,
                "Dms": Dm,
                "dDm_dEs": dDm_dE,
                "dDm_dLs": dDm_dL,
                "OmRs": OmR,
                "Ompsis": Ompsi,
            }
            if concrete:
                self._warn_unresolved(perr, self._Es, self._Ls)
                self._jrs, self._rps, self._ras = tonp(jr), tonp(rp), tonp(ra)
                self._Dms = tonp(Dm)
                self._dDm_dEs, self._dDm_dLs = tonp(dDm_dE), tonp(dDm_dL)
                self._OmRs, self._Ompsis = tonp(OmR), tonp(Ompsi)
                self._d2rps, self._d2ras = tonp(d2rps), tonp(d2ras)
                self._d2Dms = tonp(d2Dms)
                self._dOmRs, self._dOmpsis = tonp(dOmRs), tonp(dOmpsis)
            return None
        # the interpolation tables (as _setup_tori_interp)
        shp = (nE, nL)
        jr_tab = xp.reshape(jr, shp)
        OmR_tab, Ompsi_tab = xp.reshape(OmR, shp), xp.reshape(Ompsi, shp)
        dE_du = 2.0 * f64(us)[:, None] * dE_dx
        jr_dx = dE_dx / OmR_tab
        jr_dL = (dE_dL - Ompsi_tab) / OmR_tab
        sup_tab = xp.stack([xp.reshape(rp, shp), xp.reshape(ra, shp)], axis=-1)
        Dm_tab = xp.reshape(Dm, shp + (npt,))
        drE = xp.stack(
            [xp.reshape(dtp[q][0], shp) for q in (0, 1)], axis=-1
        )  # dr/dE of rp, ra
        drL = xp.stack([xp.reshape(dtp[q][1], shp) for q in (0, 1)], axis=-1)
        sup_du = drE * dE_du[..., None]
        sup_dL = drE * dE_dL[..., None] + drL
        dDE = xp.reshape(dDm_dE, shp + (npt,))
        Dm_du = dDE * dE_du[..., None]
        Dm_dL = dDE * dE_dL[..., None] + xp.reshape(dDm_dL, shp + (npt,))
        # the circular edge (the first row): the turning points leave the
        # circular radius linearly in u at the epicycle's rate and move with
        # it along L; the map's slope in u is the quartic's through zero and
        # the next two rows
        rc0, kap0 = rcs, xp.reshape(kappa, shp)[0]
        w1 = xp.sqrt(2.0 * dE_dx[0]) / kap0
        drc = 2.0 * Lgrid / (rc0**3 * kap0**2)
        sup_du = cat([xp.stack([-w1, w1], axis=-1)[None], sup_du[1:]])
        sup_dL = cat([xp.stack([drc, drc], axis=-1)[None], sup_dL[1:]])
        Dm_dL = cat([0.0 * Dm_dL[:1], Dm_dL[1:]])
        Dm_du = cat(
            [
                slope_at_zero(us[1:3], Dm_tab[1:3], Dm_du[1:3], xp=xp)[None],
                Dm_du[1:],
            ]
        )
        self._bk = {"xp": xp, "dev": dev, "Emax": Emax, "rcs": rcs}
        self._Rinf = Rinf
        self._us = us
        self._Lgrid_bk = Lgrid
        self._build_interp(
            us,
            Lgrid,
            jr_tab,
            jr_dx,
            jr_dL,
            sup_tab,
            sup_du,
            sup_dL,
            Dm_tab,
            Dm_du,
            Dm_dL,
        )
        if concrete:
            self._warn_unresolved(perr, tonp(Es), tonp(Ls))
            self._Lgrid, self._Emax = tonp(Lgrid), float(tonp(Emax))
            self._E_tab, self._Es, self._Ls = tonp(Etab), tonp(Es), tonp(Ls)
            self._Ecs = tonp(Ecs)
            self._r0c_cache = {}
            self._jr_tab, self._jr_dx, self._jr_dL = (
                tonp(t) for t in (jr_tab, jr_dx, jr_dL)
            )
            self._OmR_tab, self._Ompsi_tab = tonp(OmR_tab), tonp(Ompsi_tab)
            self._sup_tab, self._sup_du, self._sup_dL = (
                tonp(t) for t in (sup_tab, sup_du, sup_dL)
            )
            self._Dm_tab, self._Dm_du, self._Dm_dL = (
                tonp(t) for t in (Dm_tab, Dm_du, Dm_dL)
            )
        return None

    def _bk_turning_point_derivs(self, r, E, L):
        """_turning_point_derivs on backend arrays"""
        dPeff = self._bk_dPhi(r) - L**2 / r**3
        return 1.0 / dPeff, -(L / r**2) / dPeff

    def _bk_R2(self, r):
        """d^2Phi/dr^2"""
        xp = get_namespace(r)
        flat = xp.reshape(r, (-1,))
        return xp.reshape(
            evaluateR2derivs(self._pot, flat, 0.0 * flat, use_physical=False), r.shape
        )

    def _bk_turning_point_derivs2(self, r, E, L):
        """_turning_point_derivs2 on backend arrays"""
        P1 = self._bk_dPhi(r) - L**2 / r**3
        P2 = self._bk_R2(r) + 3.0 * L**2 / r**4
        rE, rL = 1.0 / P1, -(L / r**2) / P1
        return (
            -P2 * rE**2 / P1,
            (-P2 * rE * rL + (2.0 * L / r**3) * rE) / P1,
            (-P2 * rL**2 + (4.0 * L / r**3) * rL - 1.0 / r**2) / P1,
        )

    def _bk_second_order_tables(
        self, xp, tau, st, ct, r, pr, rp, ra, E, L, Jrq, a, e, etat, dDm_dE, dDm_dL
    ):
        """_second_order_tables for every torus at once (tori along the first
        axis, samples along the last): the (E, L) second derivatives of the
        turning points (EE, EL, LL) and of the map's coefficients, and the
        (E, L) derivatives of the two frequencies"""
        ms, N = self._nforDm, self._ntau

        def _cum(f):
            q = _antiderivative_xp(xp, f)
            return (
                xp.mean(f, axis=-1, keepdims=True) * tau
                + q
                - _spec_eval_xp(xp, _offset_spec_coeffs_xp(xp, q), 0.0 * tau[:1])
            )

        pairs = ((0, 0), (0, 1), (1, 1))
        d1p = self._bk_turning_point_derivs(rp, E, L)
        d1a = self._bk_turning_point_derivs(ra, E, L)
        d2p = self._bk_turning_point_derivs2(rp, E, L)
        d2a = self._bk_turning_point_derivs2(ra, E, L)
        Lc = L[:, None]
        rtau = 0.5 * (ra - rp)[:, None] * st
        w1 = [(0.5 * (d1a[i] - d1p[i]))[:, None] for i in range(2)]
        w2 = [(0.5 * (d2a[k] - d2p[k]))[:, None] for k in range(3)]
        r1 = [(0.5 * (d1p[i] + d1a[i]))[:, None] - w1[i] * ct for i in range(2)]
        r2 = [(0.5 * (d2p[k] + d2a[k]))[:, None] - w2[k] * ct for k in range(3)]
        P1 = self._bk_dPhi(r) - Lc**2 / r**3
        P2 = self._bk_R2(r) + 3.0 * Lc**2 / r**4
        S_r, S_rr = -2.0 * P1, -2.0 * P2
        S_a = [2.0 + 0.0 * r, -2.0 * Lc / r**2]
        S_ra = [0.0 * r, 4.0 * Lc / r**3]
        S_ab = [0.0 * r, 0.0 * r, -2.0 / r**2]
        N1 = [S_a[i] + S_r * r1[i] for i in range(2)]
        A2 = []
        for kk, (i, j) in enumerate(pairs):
            N2 = (
                S_ab[kk]
                + S_ra[i] * r1[j]
                + S_ra[j] * r1[i]
                + S_rr * r1[i] * r1[j]
                + S_r * r2[kk]
            )
            A2.append(
                _cum(
                    (N2 * rtau + N1[i] * w1[j] * st + N1[j] * w1[i] * st) / (2.0 * pr)
                    - N1[i] * N1[j] * rtau / (4.0 * pr**3)
                    + pr * w2[kk] * st
                )
            )
        dtdtau = rtau / pr
        P = xp.mean(dtdtau, axis=-1)
        Q = xp.mean(Lc / r**2 * dtdtau, axis=-1)
        dP, dQ = [], []
        for i in range(2):
            dtd = w1[i] * st / pr - rtau * N1[i] / (2.0 * pr**3)
            dP.append(xp.mean(dtd, axis=-1))
            dQ.append(
                xp.mean(
                    ((1.0 if i == 1 else 0.0) / r**2 - 2.0 * Lc * r1[i] / r**3) * dtdtau
                    + Lc / r**2 * dtd,
                    axis=-1,
                )
            )
        Jal, Lal = [P, -Q], [0.0, 1.0]
        J2 = [dP[0], dP[1], -dQ[1]]
        _, _, da, de = self._aAIinv._orbital_params2(Jrq, L)

        def chain2(d):
            dJ, dL, dJJ, dJL, dLL = d
            first = [(dJ * Jal[i] + dL * Lal[i])[:, None] for i in range(2)]
            second = [
                (
                    dJJ * Jal[i] * Jal[j]
                    + dJL * (Jal[i] * Lal[j] + Jal[j] * Lal[i])
                    + dLL * Lal[i] * Lal[j]
                    + dJ * J2[kk]
                )[:, None]
                for kk, (i, j) in enumerate(pairs)
            ]
            return first, second

        a1, a2 = chain2(da)
        e1, e2 = chain2(de)
        ac, ec = a[:, None], e[:, None]
        _, AA_a, AA_e, AA_aa, AA_ae, AA_ee = self._aAIinv._cumulative2(ac, ec, etat)
        fA, fA_eta = self._aAIinv._flux_eta(ac, ec, etat)
        _, fA_a, fA_e = self._aAIinv._flux_derivs(ac, ec, etat)
        eta1 = [_sine_series_xp(xp, N, d, ms)[0] for d in (dDm_dE, dDm_dL)]
        rhs = []
        for kk, (i, j) in enumerate(pairs):
            rhs.append(
                A2[kk]
                - (
                    fA_eta * eta1[i] * eta1[j]
                    + fA_a * (a1[j] * eta1[i] + a1[i] * eta1[j])
                    + fA_e * (e1[j] * eta1[i] + e1[i] * eta1[j])
                )
                - (
                    AA_aa * a1[i] * a1[j]
                    + AA_ae * (a1[i] * e1[j] + a1[j] * e1[i])
                    + AA_ee * e1[i] * e1[j]
                )
                - AA_a * a2[kk]
                - AA_e * e2[kk]
            )
        D2 = _sine_galerkin_xp(xp, fA, xp.stack(rhs, axis=-1), ms)
        return (
            xp.stack(d2p, axis=-1),
            xp.stack(d2a, axis=-1),
            D2,
            xp.stack([-dP[0] / P**2, -dP[1] / P**2], axis=-1),
            xp.stack(
                [(dQ[0] * P - Q * dP[0]) / P**2, (dQ[1] * P - Q * dP[1]) / P**2],
                axis=-1,
            ),
        )

    def _bk_fit_auxiliary(self, xp, f64, concrete, lib, Ls, rc, rp, ra):
        """The auxiliary as _setup_auxiliary and _fit_auxiliary choose it: the
        isochrone fitted to the rotation curve over the librating tori's
        radial range (_fit_isochrone_vc2, on the host for a concrete family
        and on the device under a trace); with only circular tori, the
        closed-form isochrone of the central torus. Frozen: no derivative
        flows through it."""
        n = Ls.shape[0]
        nlib = xp.sum(xp.astype(lib, xp.int64))
        Lc, rcc = stop_gradient(Ls[n // 2]), stop_gradient(rc[n // 2])

        def _rf_lnvc2():
            inf = f64(numpy.inf)
            rlo = xp.min(xp.where(lib, rp, inf))
            rhi = xp.max(xp.where(lib, ra, -inf))
            lnr = xp.log(rlo) + (xp.log(rhi) - xp.log(rlo)) * f64(
                numpy.linspace(0.0, 1.0, 25)
            )
            rf = (getattr(xp, "concat", None) or xp.concatenate)(
                [xp.reshape(rlo, (1,)), xp.exp(lnr[1:-1]), xp.reshape(rhi, (1,))]
            )
            lnvc2 = xp.log(vcirc(self._pot, rf, use_physical=False) ** 2)
            return stop_gradient(rf), stop_gradient(lnvc2)

        if concrete:
            if not concretely_true(nlib > 0):
                L, r0 = (float(as_numpy(t)) for t in (Lc, rcc))
                return (
                    float(L**2 * numpy.sqrt(2.0) * (1.0 + numpy.sqrt(2.0)) ** 2 / r0),
                    r0,
                )
            return _fit_isochrone_vc2(*(numpy.array(as_numpy(t)) for t in _rf_lnvc2()))
        return lazy_cond(
            xp,
            nlib > 0,
            lambda: _fit_isochrone_vc2(*_rf_lnvc2()),
            lambda: (Lc**2 * numpy.sqrt(2.0) * (1.0 + numpy.sqrt(2.0)) ** 2 / rcc, rcc),
        )

    def _build_interp(
        self,
        us,
        Lg,
        jr_tab,
        jr_dx,
        jr_dL,
        sup_tab,
        sup_du,
        sup_dL,
        Dm_tab,
        Dm_du,
        Dm_dL,
    ):
        """The interpolants of _rebuild_interp from the given tables"""
        self._jr_ip = HermiteFamily2D(us**2, Lg, jr_tab, jr_dx, jr_dL)
        self._sup_ip = HermiteFamily2D(us, Lg, sup_tab, sup_du, sup_dL)
        self._Dm_ip = HermiteFamily2D(us, Lg, Dm_tab, Dm_du, Dm_dL)
        return None

    # ---------- evaluation on a backend (jax/torch)
    def _namespace(self, xp):
        """A family built on a backend evaluates there whatever the inputs, so
        derivatives in the potential's parameters reach every output"""
        return xp if self._bk is None else self._bk["xp"]

    def _bk_node_tables(self, xp, dev):
        """The explicit tori's tables in xp: the backend-built ones, or the
        numpy ones copied over"""
        if self._bk is not None:
            return self._bk
        key = (getattr(xp, "__name__", ""), str(dev))
        cache = self.__dict__.setdefault("_bk_tables_on", {})
        if key in cache:
            return cache[key]
        out = {
            k: asarray_on_device(xp, getattr(self, "_" + k), dev)
            for k in (
                "jrs",
                "rps",
                "ras",
                "Dms",
                "dDm_dEs",
                "dDm_dLs",
                "OmRs",
                "Ompsis",
                "Es",
                "Ls",
                "d2rps",
                "d2ras",
                "d2Dms",
                "dOmRs",
                "dOmpsis",
            )
        }
        if not under_jax_trace(*out.values()):  # a trace's constants die with it
            cache[key] = out
        return out

    def _bk_E_of_uL(self, u, L):
        """_E_of_uL in jax/torch, with r_0(E_c(L)) solved afresh (as _r0c)"""
        xp = get_namespace(u, L)
        shape = L.shape  # () for one torus, (n,) for one per point
        rc = xp.reshape(self._bk_rc_eval(L), (-1,))
        Lf, uf = xp.reshape(L, (-1,)), xp.reshape(u, (-1,))
        Ec = self._bk_Phi(rc) + Lf**2 / (2.0 * rc**2)
        r0c = _backend_brentq(
            lambda r, E: self._bk_Phi(r) - E,
            rc,
            0.0 * rc + self._Rinf,
            args=(Ec,),
            xtol=1e-14,
            elementwise=True,
        )
        g0c = r0c * self._bk_dPhi(r0c)
        lnq = xp.log(self._Rinf / r0c)
        r = r0c * xp.exp(lnq * uf**2)
        g = r * self._bk_dPhi(r)
        return tuple(
            xp.reshape(t, shape)
            for t in (
                self._bk_Phi(r),
                g * lnq,
                (1.0 - uf**2) * (Lf / rc**2) * (g / g0c),
            )
        )

    def _bk_check_L(self, L):
        # stop_gradient first: inside a lax.cond branch a concrete L comes
        # out of it traced
        L = stop_gradient(L)
        if has_concrete_truth_value(_scalar_probe(L)):
            self._check_L(as_numpy(L))
        return None

    def _bk_interp_tables(self, xp, jr, L):
        """_interp_tables in jax/torch: the label u from (J_r, L) by the
        backend brentq on the J_r interpolant (its final Newton step gives
        du/dJ_r = 1/(dJ_r/du) and du/dL), then the chains"""
        self._bk_check_L(L)
        jlo = self._jr_ip.ev(0.0 * jr, L)
        jhi = self._jr_ip.ev(1.0 + 0.0 * jr, L)
        tol = 1e-12 * (1.0 + xp.abs(jr))
        bad = xp.any((jr < jlo - tol) | (jr > jhi + tol))
        if has_concrete_truth_value(bad) and concretely_true(bad):
            jrf, jlf, jhf, Lf = (
                numpy.atleast_1d(as_numpy(stop_gradient(t))) for t in (jr, jlo, jhi, L)
            )
            kk = int(
                numpy.argmax(
                    (jrf < jlf - 1e-12 * (1.0 + numpy.fabs(jrf)))
                    | (jrf > jhf + 1e-12 * (1.0 + numpy.fabs(jrf)))
                )
            )
            raise ValueError(
                f"J_r = {jrf[kk]} outside the interpolated family's range "
                f"[{jlf[kk]}, {jhf[kk]}] at L = {Lf[kk]}"
            )
        jrc = xp.where(jr < jlo, jlo, xp.where(jr > jhi, jhi, jr))
        if jr.ndim:
            return self._bk_interp_tables_points(xp, jrc, L, jlo, jhi)
        guess = None
        if not has_concrete_truth_value(jr):
            # under a trace, where a bisection would take ~50 sequential
            # halvings: Newton steps from the piecewise-linear inverse through
            # the interpolant's values at its x = u^2 nodes (to ~1e-2 in u,
            # three steps reach round-off)
            xn = asarray_on_device(xp, self._us**2, device_of(jr))
            jn = stop_gradient(self._jr_ip.ev(xn, L + 0.0 * xn))
            x0 = interp_linear(xp, jn, xn, stop_gradient(jrc), extrapolate="clip")
            guess = xp.reshape(xp.sqrt(x0), (1,))
        u = _backend_brentq(
            lambda uu, jj, LL: self._jr_ip.ev(uu * uu, LL) - jj,
            xp.reshape(0.0 * jr, (1,)),
            xp.reshape(1.0 + 0.0 * jr, (1,)),
            args=(xp.reshape(jrc, (1,)), xp.reshape(L, (1,))),
            xtol=1e-15,
            bracket_width=1.0,
            elementwise=True,
            guess=guess,
            newton_steps=5,
        )[0]
        x = u * u
        _, jr_dx, djr_dL = self._jr_ip.ev3(x, L)
        djr_du = 2.0 * u * jr_dx
        _, dE_dx, dE_dL = self._bk_E_of_uL(u, L)
        OmR = dE_dx / jr_dx
        OmL = dE_dL - OmR * djr_dL
        sup, dsup_du, dsup_dL = self._sup_ip.ev3(u, L)
        Dm, dDm_du, dDm_dL = self._Dm_ip.ev3(u, L)
        return (
            OmR,
            OmL,
            (sup[0], sup[1]),
            (dsup_du[0] / djr_du, dsup_du[1] / djr_du),
            (
                dsup_dL[0] - dsup_du[0] * djr_dL / djr_du,
                dsup_dL[1] - dsup_du[1] * djr_dL / djr_du,
            ),
            Dm,
            dDm_du / djr_du,
            dDm_dL - dDm_du * djr_dL / djr_du,
        )

    def _bk_interp_tables_points(self, xp, jrc, L, jlo, jhi):
        """_bk_interp_tables for arrays of tori: the label of each by Newton
        steps from the piecewise-linear inverse through the interpolant's
        values at its x = u^2 nodes along that torus's L, all at once, and
        the tables and chains with one row per torus"""
        dev = device_of(jrc)
        xn = asarray_on_device(xp, self._us**2, dev)
        nx = xn.shape[0]
        jn = stop_gradient(self._jr_ip.ev(xn + 0.0 * L[:, None], L[:, None] + 0.0 * xn))
        jt = stop_gradient(jrc)
        k = xp.clip(
            xp.sum(xp.astype(jn[:, 1:] <= jt[:, None], xp.int64), axis=-1), 0, nx - 2
        )
        base = xp.arange(jt.shape[0]) * nx
        flat = xp.reshape(jn, (-1,))
        J0, J1 = xp.take(flat, base + k), xp.take(flat, base + k + 1)
        x0 = xp.take(xn, k) + (jt - J0) / (J1 - J0) * (
            xp.take(xn, k + 1) - xp.take(xn, k)
        )
        x0 = xp.clip(x0, 0.0, 1.0)
        u = _backend_brentq(
            lambda uu, jj, LL: self._jr_ip.ev(uu * uu, LL) - jj,
            0.0 * jrc,
            1.0 + 0.0 * jrc,
            args=(jrc, L),
            xtol=1e-15,
            bracket_width=1.0,
            guess=xp.sqrt(x0),
            newton_steps=5,
        )
        x = u * u
        _, jr_dx, djr_dL = self._jr_ip.ev3(x, L)
        djr_du = 2.0 * u * jr_dx
        _, dE_dx, dE_dL = self._bk_E_of_uL(u, L)
        OmR = dE_dx / jr_dx
        OmL = dE_dL - OmR * djr_dL
        sup, dsup_du, dsup_dL = self._sup_ip.ev3(u, L)
        Dm, dDm_du, dDm_dL = self._Dm_ip.ev3(u, L)
        cJ, cL = 1.0 / djr_du, djr_dL / djr_du
        return (
            OmR,
            OmL,
            (sup[:, 0], sup[:, 1]),
            (dsup_du[:, 0] * cJ, dsup_du[:, 1] * cJ),
            (dsup_dL[:, 0] - dsup_du[:, 0] * cL, dsup_dL[:, 1] - dsup_du[:, 1] * cL),
            Dm,
            dDm_du * cJ[:, None],
            dDm_dL - dDm_du * cL[:, None],
        )

    def _bk_match_node(self, xp, tabs, jr, L):
        """_match_node in jax/torch: the index of the explicit torus, raising
        when the values are concrete and match none"""
        # one row per requested torus (a single row for a scalar)
        dev = xp.abs(tabs["jrs"] - jr[..., None]) + xp.abs(tabs["Ls"] - L[..., None])
        ii = xp.argmin(stop_gradient(dev), axis=-1)
        bad = xp.min(dev, axis=-1) > 1e-8 * (1.0 + xp.abs(jr) + xp.abs(L))
        if concretely_true(xp.any(bad)):
            kk = int(numpy.argmax(numpy.atleast_1d(as_numpy(stop_gradient(bad)))))
            jrf, Lf = (
                float(numpy.atleast_1d(as_numpy(stop_gradient(t)))[kk]) for t in (jr, L)
            )
            raise ValueError(
                f"(J_r, L) = ({jrf}, {Lf}) is not one of the set-up tori; "
                "discrete mode evaluates the stored tori only (use "
                "setup_interp=True to interpolate)"
            )
        return ii

    def _bk_node(self, xp, tabs, ii, jr, L):
        """An explicit torus as a quadratic one-node family in jax/torch (as
        _quadratic_node chains it): its tables, their slopes, and its
        frequencies as polynomials in the departures (dJ, dL) from the node,
        with its exact first and second derivatives in (J_r, L), so that the
        map is canonical and its derivatives in the actions are the true
        family's at the node. The values are the node's (as on numpy, where
        J_r is replaced by the stored one); the departures carry only
        derivatives (in the requested actions, and, as the node moves with a
        potential parameter, against its own)"""
        OmR, OmL = tabs["OmRs"][ii], tabs["Ompsis"][ii]
        E, L0 = tabs["Es"][ii], tabs["Ls"][ii]
        dJ, dL = jr - tabs["jrs"][ii], L - L0
        dJ, dL = dJ - stop_gradient(dJ), dL - stop_gradient(dL)
        dOmR, dOmL = tabs["dOmRs"][ii], tabs["dOmpsis"][ii]
        E_JJ = dOmR[..., 0] * OmR
        E_JL = dOmR[..., 0] * OmL + dOmR[..., 1]
        E_LL = dOmL[..., 0] * OmL + dOmL[..., 1]
        circ = tabs["jrs"][ii] == 0.0

        def family(x0, x1E, x1L, x2):
            # x(E, L) with partials (x_E, x_L) and (x_EE, x_EL, x_LL), chained
            # to (J_r, L) and expanded to second order in (dJ, dL): the value
            # and the two first derivatives, each to its own order (the
            # torus's scalars broadcast against a vector-valued table's rows)
            def c(t):
                return t[..., None] if x0.ndim > t.ndim else t

            oR, oL, eJJ, eJL, eLL, dj, dl = (
                c(t) for t in (OmR, OmL, E_JJ, E_JL, E_LL, dJ, dL)
            )
            xEE, xEL, xLL = x2
            xJ, xL = x1E * oR, x1E * oL + x1L
            xJJ = xEE * oR**2 + x1E * eJJ
            xJL = xEE * oR * oL + xEL * oR + x1E * eJL
            xLL = xEE * oL**2 + 2.0 * xEL * oL + xLL + x1E * eLL
            return (
                x0
                + xJ * dj
                + xL * dl
                + 0.5 * (xJJ * dj**2 + 2.0 * xJL * dj * dl + xLL * dl**2),
                xJ + xJJ * dj + xJL * dl,
                xL + xJL * dj + xLL * dl,
            )

        sup = []
        for key, k2 in (("rps", "d2rps"), ("ras", "d2ras")):
            r = tabs[key][ii]
            # the level-set rule needs a libration: the circular node's dead
            # branch reads a nearby radius
            drE, drL = self._bk_turning_point_derivs(xp.where(circ, 1.1 * r, r), E, L0)
            d2 = tabs[k2][ii]
            sup.append(family(r, drE, drL, (d2[..., 0], d2[..., 1], d2[..., 2])))
        D2 = tabs["d2Dms"][ii]
        Dm, dDmJ, dDmL = family(
            tabs["Dms"][ii],
            tabs["dDm_dEs"][ii],
            tabs["dDm_dLs"][ii],
            (D2[..., 0], D2[..., 1], D2[..., 2]),
        )
        return (
            tabs["jrs"][ii] + dJ,
            OmR + E_JJ * dJ + E_JL * dL,
            OmL + E_JL * dJ + E_LL * dL,
            (sup[0][0], sup[1][0]),
            (sup[0][1], sup[1][1]),
            (sup[0][2], sup[1][2]),
            Dm,
            dDmJ,
            dDmL,
        )

    def _bk_kernel(self, xp, tau, a, e, Dm, rp, ra, chains):
        """_kernel in jax/torch"""
        ms = asarray_on_device(xp, self._nforDm * 1.0, device_of(tau))
        mt = tau[:, None] * ms[None, :]
        smt = xp.sin(mt)
        eta = tau + _bk_dot(xp, smt, Dm)
        deta = 1.0 + _bk_dot(xp, xp.cos(mt), ms * Dm)
        se, ce = xp.sin(eta), xp.cos(eta)
        rA, pA, gA = self._aAIinv._profile(a, e, eta)
        kap = a * e / (a + self._b)
        thetaA = eta - kap * se
        dthetaA = (1.0 - kap * ce) * deta
        st, ct = xp.sin(tau), xp.cos(tau)
        r = rp * (1.0 + ct) / 2.0 + ra * (1.0 - ct) / 2.0
        # sin(eta)/sin(tau) -> eta'(tau) at the turning points
        big = xp.abs(st) > 1e-12
        sratio = xp.where(big, se / xp.where(big, st, 1.0), deta)
        pr = pA * gA * sratio * deta / (0.5 * (ra - rp))
        drA_da, drA_de = self._aAIinv._radius_partials(a, e, eta)
        comps = []
        for da, de, drp, dra, dDm in chains:
            drA = drA_da * da + drA_de * de + gA * se * _bk_dot(xp, smt, dDm)
            dr = drp * (1.0 + ct) / 2.0 + dra * (1.0 - ct) / 2.0
            comps.append(pA * drA - pr * dr)
        return eta, thetaA, dthetaA, r, pr, comps

    def _bk_tau_solve(self, xp, thr, a, e, Dm, rp, ra, chainJ):
        """_tau_solve in jax/torch: the clipped Newton iteration on the
        auxiliary angle's derivative, without a derivative, until every angle
        converges (or for maxiter steps under a trace), then the backend
        brentq from there in the bracket [theta - s, theta + s] with s beyond
        the extreme of theta_r(tau) - tau (as the numpy fallback), whose
        Newton steps finish any unconverged angle and whose final step
        carries the implicit-function derivative"""

        def _theta(x, args=(a, e, Dm, rp, ra, chainJ)):
            _, thetaA, dthetaA, _, _, (compJ,) = self._bk_kernel(
                xp, x, *args[:5], (args[5],)
            )
            return thetaA + compJ, dthetaA

        # the iteration on derivative-free copies of the tables (a jax
        # while_loop must carry no derivative), stopped at convergence
        sg_args = (
            *(stop_gradient(t) for t in (a, e, Dm, rp, ra)),
            tuple(stop_gradient(t) for t in chainJ),
        )
        thr_sg = stop_gradient(thr)
        if Dm.ndim == 2:
            return self._bk_tau_solve_points(xp, thr, thr_sg, _theta, sg_args)

        def step(x):
            f, fp = _theta(x, sg_args)
            f = f - thr_sg
            return xp.where(
                xp.abs(f) < self._angle_tol, x, x + xp.clip(-f / fp, -0.5, 0.5)
            )

        x = iterate_to_fixed_point(step, thr_sg, self._maxiter)
        done = concretely_true(
            xp.all(xp.abs(_theta(x, sg_args)[0] - thr_sg) < self._angle_tol)
        )
        if done:  # the root is within ~angle_tol of x
            lo, hi = x - 1e-3, x + 1e-3
        else:
            xscan = asarray_on_device(
                xp,
                numpy.linspace(0.0, 2.0 * numpy.pi, 256, endpoint=False),
                device_of(thr),
            )
            sc = xp.max(xp.abs(_theta(xscan, sg_args)[0] - xscan)) + 0.1
            lo, hi = thr_sg - sc, thr_sg + sc
        return _backend_brentq(
            lambda xx, tt: _theta(xx)[0] - tt,
            lo,
            hi,
            args=(thr,),
            guess=x,
            # under a trace the iteration has run to convergence already
            newton_steps=1 if done else (24 if has_concrete_truth_value(x[0]) else 2),
        )

    def _bk_tau_solve_points(self, xp, thr, thr_sg, _theta, sg_args):
        """_bk_tau_solve with one torus per angle: the residual wrapped into
        (-pi, pi], so that an anomaly crossing the pericentre converges to
        the equivalent angle (as numpy), then the brentq on the branch the
        iteration reached, theta + 2 pi k, in the bracket of half-width 2 pi
        around it, which always holds the root (theta_r(tau) - tau stays
        within 2 pi), in place of a scan of every torus"""

        def step(x):
            f, fp = _theta(x, sg_args)
            f = xp.remainder(f - thr_sg + numpy.pi, 2.0 * numpy.pi) - numpy.pi
            return xp.where(
                xp.abs(f) < self._angle_tol, x, x + xp.clip(-f / fp, -0.5, 0.5)
            )

        x = iterate_to_fixed_point(step, thr_sg, self._maxiter)
        th = _theta(x, sg_args)[0]
        tgt = thr + 2.0 * numpy.pi * stop_gradient(
            xp.round((th - thr_sg) / (2.0 * numpy.pi))
        )
        tgt_sg = stop_gradient(tgt)
        done = concretely_true(xp.all(xp.abs(th - tgt_sg) < self._angle_tol))
        if done:  # the root is within ~angle_tol of x
            lo, hi = x - 1e-3, x + 1e-3
        else:
            lo, hi = tgt_sg - 2.0 * numpy.pi, tgt_sg + 2.0 * numpy.pi
        return _backend_brentq(
            lambda xx, tt: _theta(xx)[0] - tt,
            lo,
            hi,
            args=(tgt,),
            guess=x,
            newton_steps=1 if done else (24 if has_concrete_truth_value(x[0]) else 2),
        )

    def _bk_inputs(self, xp, *args):
        """The inputs on the family's backend and device, and in float64"""
        if self._bk is not None:
            dev = self._bk["dev"]
        else:
            dev = device_of(*(a for a in args if is_backend_array(a)))
        ins = tuple(
            a * 1.0
            if is_backend_array(a)
            else asarray_on_device(xp, numpy.asarray(a, dtype="float"), dev)
            for a in args
        )
        return ins, at_least_float64(*ins)

    def _bk_librating(self, xp, jr, jphi, jz, L, thr, anglephi, anglez, circ):
        """The (x, v) and the frequencies of a librating torus in jax/torch
        (the numpy _xvFreqs below its circular case); a circular request that
        reaches it under a trace reads a stand-in torus"""
        if self._interp:
            jr_s = xp.where(circ, 0.5 * self._jr_ip.ev(1.0 + 0.0 * jr, L), jr)
            OmR, OmL, (rp, ra), (drpJ, draJ), (drpL, draL), Dm, dDmJ, dDmL = (
                self._bk_interp_tables(xp, jr_s, L)
            )
        else:
            tabs = self._bk_node_tables(xp, device_of(jr))
            ii = self._bk_match_node(xp, tabs, jr, L)
            (
                jr_s,
                OmR,
                OmL,
                (rp, ra),
                (drpJ, draJ),
                (drpL, draL),
                Dm,
                dDmJ,
                dDmL,
            ) = self._bk_node(xp, tabs, ii, jr, L)
            jr_s = xp.where(circ, 0.01 * L, jr_s)
            rp, ra = xp.where(circ, 0.9 * rp, rp), xp.where(circ, 1.1 * ra, ra)
        a, e = self._aAIinv._orbital_params(jr_s, L)
        one, zero = 1.0 + 0.0 * jr_s, 0.0 * jr_s
        _, _, daJ, deJ = self._aAIinv._orbital_param_chains(jr_s, L, one, zero)
        _, _, daL, deL = self._aAIinv._orbital_param_chains(jr_s, L, zero, one)
        chainJ = (daJ, deJ, drpJ, draJ, dDmJ)
        chainL = (daL, deL, drpL, draL, dDmL)
        taus = self._bk_tau_solve(xp, thr, a, e, Dm, rp, ra, chainJ)
        eta, _, _, r, pr, (Delta,) = self._bk_kernel(
            xp, taus, a, e, Dm, rp, ra, (chainL,)
        )
        out = self._unlift(
            self._aAIinv._xvFreqs_from_eta(
                jr_s,
                jphi,
                jz,
                eta,
                anglephi - xp.sign(jphi) * Delta,
                anglez - Delta,
            )[:6],
            r,
            pr,
        )
        return out, OmR, OmL

    def _bk_circular(self, xp, L, jphi, jz, thr, anglephi, anglez, explicit_jr=None):
        """The circular orbit of angular momentum L in jax/torch (as the numpy
        _xvFreqs's circular case)"""
        if self._interp:
            self._bk_check_L(L)
        elif explicit_jr is not None:
            self._bk_match_node(
                xp, self._bk_node_tables(xp, device_of(L)), explicit_jr, L
            )
        rc = self._bk_rc_eval(L)
        if L.ndim:  # one circular orbit per point
            kappa = epifreq(self._pot, rc, use_physical=False)
            eta0 = 0.0 * L
        else:
            kappa = xp.reshape(
                epifreq(self._pot, xp.reshape(rc, (1,)), use_physical=False), ()
            )
            # the eccentric anomaly zero once, broadcast against the angles
            eta0 = xp.reshape(0.0 * L, (1,))
        out = self._unlift(
            self._aAIinv._xvFreqs_from_eta(0.1 * L, jphi, jz, eta0, anglephi, anglez)[
                :6
            ],
            rc + 0.0 * thr,
            0.0 * thr,
        )
        return out, kappa, L / rc**2

    def _bk_xvFreqs(self, xp, jr, jphi, jz, angler, anglephi, anglez):
        """_xvFreqs in jax/torch: one torus, any number of angles,
        differentiable in the actions, the angles, and (for a family built on
        a backend) the potential's parameters"""
        ins, (jr, jphi, jz, angler, anglephi, anglez) = self._bk_inputs(
            xp, jr, jphi, jz, angler, anglephi, anglez
        )
        if jr.ndim or jphi.ndim or jz.ndim:
            return self._bk_xvFreqs_points(
                xp, ins, jr, jphi, jz, angler, anglephi, anglez
            )
        jr, jphi, jz = (xp.reshape(t, ()) for t in (jr, jphi, jz))
        angler, anglephi, anglez = xp.broadcast_arrays(
            *(xp.reshape(t, (-1,)) for t in (angler, anglephi, anglez))
        )
        L = jz + xp.abs(jphi)
        thr = xp.remainder(angler, 2.0 * numpy.pi)
        circ = jr == 0.0
        args = (jphi, jz, thr, anglephi, anglez)
        if has_concrete_truth_value(jr):
            # a concrete action takes one branch only
            if concretely_true(circ):
                out, OmR, OmL = self._bk_circular(
                    xp, L, *args, explicit_jr=None if self._interp else jr
                )
            else:
                out, OmR, OmL = self._bk_librating(xp, jr, jphi, jz, L, *args[2:], circ)
        else:
            if self._interp:  # before the branches, which trace L
                self._bk_check_L(L)

            # one branch under jax.jit (both, selected, under vmap or torch)
            def _circ():
                oc, kc, lc = self._bk_circular(xp, L, *args)
                return (*oc, kc, lc)

            def _lib():
                ol, kl, ll = self._bk_librating(xp, jr, jphi, jz, L, *args[2:], circ)
                return (*ol, kl, ll)

            *out, OmR, OmL = lazy_cond(xp, circ, _circ, _lib)
        res = (*out, OmR, xp.sign(jphi) * OmL, OmL)
        return tuple(match_input_dtype(t, *ins) for t in res)

    def _bk_select(self, xp, circ, circular, librating):
        """The circular and the librating outputs per point, each computed
        only when some point needs it (when the actions are concrete)"""
        anyc, allc = xp.any(circ), xp.all(circ)
        if has_concrete_truth_value(anyc):
            if not concretely_true(anyc):
                return librating()
            if concretely_true(allc):
                return circular()
        return tuple(xp.where(circ, c, li) for c, li in zip(circular(), librating()))

    def _bk_xvFreqs_points(self, xp, ins, jr, jphi, jz, angler, anglephi, anglez):
        """_bk_xvFreqs for arrays of actions, broadcast against the angles
        (by numpy's rules): every point its own torus, the tables read at
        each point's own (J_r, L), one row per point; the frequencies are
        arrays"""
        jr, jphi, jz, angler, anglephi, anglez = xp.broadcast_arrays(
            jr, jphi, jz, angler, anglephi, anglez
        )
        shape = jr.shape
        jr, jphi, jz, angler, anglephi, anglez = (
            xp.reshape(t, (-1,)) for t in (jr, jphi, jz, angler, anglephi, anglez)
        )
        L = jz + xp.abs(jphi)
        thr = xp.remainder(angler, 2.0 * numpy.pi)
        circ = jr == 0.0
        if self._interp:
            self._bk_check_L(L)

        def _circ():
            oc, kc, lc = self._bk_circular(xp, L, jphi, jz, thr, anglephi, anglez)
            return (*oc, kc, lc)

        def _lib():
            ol, kl, ll = self._bk_librating(
                xp, jr, jphi, jz, L, thr, anglephi, anglez, circ
            )
            return (*ol, kl, ll)

        *out, OmR, OmL = self._bk_select(xp, circ, _circ, _lib)
        res = (*out, OmR, xp.sign(jphi) * OmL, OmL)
        return tuple(match_input_dtype(xp.reshape(t, shape), *ins) for t in res)

    def _bk_Freqs_points(self, xp, ins, jr, jphi, jz):
        """_bk_Freqs for arrays of actions"""
        jr, jphi, jz = xp.broadcast_arrays(jr, jphi, jz)
        shape = jr.shape
        jr, jphi, jz = (xp.reshape(t, (-1,)) for t in (jr, jphi, jz))
        L = jz + xp.abs(jphi)
        circ = jr == 0.0
        if self._interp:
            self._bk_check_L(L)

        def _circ():
            rc = self._bk_rc_eval(L)
            return epifreq(self._pot, rc, use_physical=False), L / rc**2

        def _lib():
            if self._interp:
                jr_s = xp.where(circ, 0.5 * self._jr_ip.ev(1.0 + 0.0 * jr, L), jr)
                return self._bk_interp_tables(xp, jr_s, L)[:2]
            tabs = self._bk_node_tables(xp, device_of(jr))
            ii = self._bk_match_node(xp, tabs, jr, L)
            return self._bk_node(xp, tabs, ii, jr, L)[1:3]

        OmR, OmL = self._bk_select(xp, circ, _circ, _lib)
        res = (OmR, xp.sign(jphi) * OmL, OmL)
        return tuple(match_input_dtype(xp.reshape(t, shape), *ins) for t in res)

    def _bk_Freqs(self, xp, jr, jphi, jz):
        """_Freqs in jax/torch"""
        ins, (jr, jphi, jz) = self._bk_inputs(xp, jr, jphi, jz)
        if jr.ndim or jphi.ndim or jz.ndim:
            return self._bk_Freqs_points(xp, ins, jr, jphi, jz)
        jr, jphi, jz = (xp.reshape(t, ()) for t in (jr, jphi, jz))
        L = jz + xp.abs(jphi)
        circ = jr == 0.0

        def _circ():
            if self._interp:
                self._bk_check_L(L)
            rc = self._bk_rc_eval(L)
            kappa = xp.reshape(
                epifreq(self._pot, xp.reshape(rc, (1,)), use_physical=False), ()
            )
            return kappa, L / rc**2

        def _lib():
            if self._interp:
                jr_s = xp.where(circ, 0.5 * self._jr_ip.ev(1.0 + 0.0 * jr, L), jr)
                return self._bk_interp_tables(xp, jr_s, L)[:2]
            tabs = self._bk_node_tables(xp, device_of(jr))
            ii = self._bk_match_node(xp, tabs, jr, L)
            return self._bk_node(xp, tabs, ii, jr, L)[1:3]

        if has_concrete_truth_value(jr):
            if concretely_true(circ) and not self._interp:
                self._bk_match_node(xp, self._bk_node_tables(xp, device_of(jr)), jr, L)
            OmR, OmL = _circ() if concretely_true(circ) else _lib()
        else:
            if self._interp:  # before the branches, which trace L
                self._bk_check_L(L)
            OmR, OmL = lazy_cond(
                xp, circ, lambda: tuple(_circ()), lambda: tuple(_lib())
            )
        res = (OmR, xp.sign(jphi) * OmL, OmL)
        return tuple(match_input_dtype(t, *ins) for t in res)

    def _bk_Jr(self, xp, E, L):
        """Jr in jax/torch"""
        ins, (E, L) = self._bk_inputs(xp, E, L)
        E, L = xp.reshape(E, ()), xp.reshape(L, ())
        if not self._interp:
            tabs = self._bk_node_tables(xp, device_of(E))
            dev = xp.abs(tabs["Es"] - E) + xp.abs(tabs["Ls"] - L)
            ii = xp.argmin(stop_gradient(dev))
            if concretely_true(dev[ii] > 1e-10 * (1.0 + xp.abs(E) + xp.abs(L))):
                Ef, Lf = (float(as_numpy(stop_gradient(t))) for t in (E, L))
                raise ValueError(
                    f"(E, L) = ({Ef}, {Lf}) is not one of the set-up tori; to use "
                    "interpolation, initialize with setup_interp=True"
                )
            # the stored action, to second order about the node in (dE, dL):
            # J_E = P = 1/Omega_r, J_L = -Q = -Omega_psi/Omega_r, and their
            # derivatives from the frequencies'
            OmR, Ompsi = tabs["OmRs"][ii], tabs["Ompsis"][ii]
            dOmR, dOmpsi = tabs["dOmRs"][ii], tabs["dOmpsis"][ii]
            P_, Q_ = 1.0 / OmR, Ompsi / OmR
            P_E, P_L = -dOmR[0] / OmR**2, -dOmR[1] / OmR**2
            Q_L = (dOmpsi[1] * OmR - Ompsi * dOmR[1]) / OmR**2
            dE, dL = E - tabs["Es"][ii], L - tabs["Ls"][ii]
            dE, dL = dE - stop_gradient(dE), dL - stop_gradient(dL)
            out = (
                tabs["jrs"][ii]
                + P_ * dE
                - Q_ * dL
                + 0.5 * (P_E * dE**2 + 2.0 * P_L * dE * dL - Q_L * dL**2)
            )
            return match_input_dtype(out, *ins)
        self._bk_check_L(L)
        rc = self._bk_rc_eval(L)
        Ec = self._bk_Phi(xp.reshape(rc, (1,)))[0] + L**2 / (2.0 * rc**2)
        Emax = self._bk["Emax"] if self._bk is not None else self._Emax
        if has_concrete_truth_value(E) and concretely_true(
            (E < Ec - 1e-12 * (Emax - Ec)) | (E > Emax)
        ):
            Ef, Lf, Ecf, Emf = (
                float(as_numpy(stop_gradient(0.0 * E + t))) for t in (E, L, Ec, Emax)
            )
            raise ValueError(
                f"E = {Ef} outside the interpolation grid at L = {Lf}: [{Ecf}, {Emf}]"
            )
        circ = E <= Ec + 1e-12 * (1.0 + xp.abs(Ec))
        r0c = _backend_brentq(
            lambda r, EE: self._bk_Phi(r) - EE,
            xp.reshape(rc, (1,)),
            xp.reshape(0.0 * rc + self._Rinf, (1,)),
            args=(xp.reshape(Ec, (1,)),),
            xtol=1e-14,
            elementwise=True,
        )[0]
        Es = xp.where(E >= Ec, xp.where(E > Emax, Emax, E), Ec)
        r0 = _backend_brentq(
            lambda r, EE: self._bk_Phi(r) - EE,
            xp.reshape(r0c, (1,)),
            xp.reshape(0.0 * r0c + self._Rinf, (1,)),
            args=(xp.reshape(Es, (1,)),),
            xtol=1e-14,
            elementwise=True,
        )[0]
        x = xp.log(r0 / r0c) / xp.log(self._Rinf / r0c)
        # the circular orbit, to round-off, is the bottom of the grid (as on
        # numpy), with the one-sided derivative of the grid's bottom row
        x = xp.where(circ, x - stop_gradient(x), xp.where(x > 1.0, 1.0, x))
        return match_input_dtype(self._jr_ip.ev(x, L), *ins)
