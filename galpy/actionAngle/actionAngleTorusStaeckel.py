###############################################################################
#   actionAngleTorusStaeckel.py: torus mapping for general axisymmetric
#   potentials on the canonical Staeckel-inverse auxiliary: a Fourier
#   generating function removes the residual non-Staeckel part of the true
#   Hamiltonian from each torus. The auxiliary already matches the target to
#   the Staeckel-model error, so the generating coefficients are perturbative:
#   one FFT division seeds them and a few Gauss-Newton steps in coefficient
#   space (with LOCAL frequency weights, which is what makes eccentric tori
#   converge) polish them. Design and measurements: fast-orbits
#   TORUSMAPPER_MATH.md.
###############################################################################
import warnings

import numpy

from ..potential import (
    OblateStaeckelWrapperPotential,
    evaluatePotentials,
)
from ..util import galpyWarning
from .actionAngleStaeckelInverse import actionAngleStaeckelInverse


class actionAngleTorusStaeckel:
    """Torus mapping for a general axisymmetric potential: (J, theta) ->
    (x, v) with the torus lying on a single energy surface of the TRUE
    Hamiltonian, built as a Fourier generating-function layer on the
    canonical actionAngleStaeckelInverse family of the potential's Staeckel
    model. Per-torus derivatives (frequencies and the angle shift) come
    from a local quadratic model of the fitted coefficients over a J-star,
    differentiated analytically -- derivatives of a stored model, never
    separately fitted numbers -- so the composite map is exactly symplectic
    for the model it evaluates."""

    def __init__(
        self,
        pot=None,
        delta=None,
        u0=None,
        family=None,
        ngrid=24,
        maxn=8,
        polish=3,
        starfrac=0.15,
        resonance_tol=1e-3,
        **family_kwargs,
    ):
        """
        Initialize an actionAngleTorusStaeckel object.

        Parameters
        ----------
        pot : Potential or list thereof
            The TRUE potential (a general axisymmetric potential; it is
            Staeckelized internally at the given focal length for the
            auxiliary family).
        delta : float
            Focal length of the auxiliary Staeckel model (single delta,
            like the forward actionAngleStaeckel); required unless family=
            is given.
        u0 : float, callable, or 'fit', optional
            Reference u of the Staeckelization, passed to the family build.
        family : actionAngleStaeckelInverse, optional
            An already-built interpolated family for the wrapped potential;
            overrides pot/delta/u0/family_kwargs for the auxiliary. The
            family's box must cover the fitted tori's action excursions
            J +- max|dJ| ~ |Delta H|/Omega.
        ngrid : int, optional
            Number of angle-grid points per dimension of the
            (theta_R, theta_z) fitting grid.
        maxn : int, optional
            Maximum |n_R|, |n_z| of the retained Fourier lattice (must be
            < ngrid/2 to be alias-free).
        polish : int, optional
            Number of Gauss-Newton polish iterations after the first-order
            FFT seed (0 = seed only).
        starfrac : float or (float, float, float), optional
            Fractional half-widths of the J-star used for the local
            quadratic model of the coefficients, per action (J_R, L_z,
            J_z); a scalar applies to J_R and J_z with the L_z fraction
            reduced 12-fold, because L_z is O(1) where the others are
            O(0.01) and the cubic truncation of the quadratic model goes
            as the SQUARE of the step (a 0.12 L_z fraction puts ~2e-3 in
            dE/dL_z, the measured orbit-drift scale). Floored at 1e-4 in
            J units and one-sided away from the J_R, J_z >= 0 edges.
        family_kwargs : dict
            Passed to the internal actionAngleStaeckelInverse build
            (Rmin/Rmax/Rinf, grid sizes, target=, ...).

        Notes
        -----
        - 2026-09-05 - Started - Bovy (UofT)
        """
        if family is not None:
            self._fam = family
            # the TRUE potential is the family's RAW (unwrapped) potential:
            # family._pot is the OblateStaeckelWrapper (the MODEL), whose
            # Hamiltonian is trivially flat on its own tori -- flattening
            # that instead of the true H is a silent no-op
            self._pot = getattr(family, "_chart_pot", family._pot)
        else:
            if pot is None:
                raise OSError("Must specify pot= for actionAngleTorusStaeckel")
            if delta is None:
                raise OSError(
                    "Must specify delta= (the auxiliary Staeckel model's "
                    "focal length) for actionAngleTorusStaeckel"
                )
            self._pot = pot
            fkw = dict(family_kwargs)
            fkw.setdefault("setup_interp", True)
            # a scalar u0 fixes the wrapper's reference curve; an adaptive
            # u0 ('fit' or a callable) varies it across the family and is
            # passed on to the family build instead
            if u0 is not None and not (isinstance(u0, str) or callable(u0)):
                swp = OblateStaeckelWrapperPotential(
                    pot=pot, delta=float(delta), u0=float(u0)
                )
            else:
                swp = OblateStaeckelWrapperPotential(pot=pot, delta=float(delta))
                if u0 is not None:
                    fkw["u0"] = u0
            self._fam = actionAngleStaeckelInverse(pot=swp, **fkw)
        if 2 * maxn >= ngrid:
            raise ValueError("maxn must be < ngrid/2 for an alias-free Fourier lattice")
        self._ngrid = ngrid
        self._maxn = maxn
        self._polish = polish
        self._resonance_tol = resonance_tol
        self._starfrac = (
            tuple(starfrac)
            if hasattr(starfrac, "__len__")
            else (starfrac, starfrac / 12.0, starfrac)
        )
        # the fitting angle grid and the retained half-lattice: n_R > 0, or
        # n_R = 0 and n_z > 0 (the conjugate half is implied by reality)
        th = 2.0 * numpy.pi * numpy.arange(ngrid) / ngrid
        self._thr, self._thz = (
            a.ravel() for a in numpy.meshgrid(th, th, indexing="ij")
        )
        self._modes = [
            (nr, nz)
            for nr in range(0, maxn + 1)
            for nz in range(-maxn, maxn + 1)
            if nr > 0 or nz > 0
        ]
        self._torus_cache = {}

    ############################ THE PER-TORUS FIT ############################
    def _Hfield(self, jr, lz, jz, dJr, dJz):
        """The true Hamiltonian and the auxiliary's local frequencies over
        the fitting grid, at per-point actions (jr + dJr, lz, jz + dJz).

        When the action perturbation is uniform (the seed pass, dJ = 0),
        every grid point shares one J and the family evaluates the whole
        angle grid in a single vectorized call; otherwise J varies per
        point (the generating function's action shift) and the family --
        which caches per torus at a scalar J -- is called point by point."""
        npt = len(self._thr)
        if numpy.all(dJr == dJr.flat[0]) and numpy.all(dJz == dJz.flat[0]):
            out = self._fam._xvFreqs(
                jr + dJr.flat[0],
                lz,
                jz + dJz.flat[0],
                self._thr,
                numpy.zeros(npt),
                self._thz,
            )
            R, vR, vT, z, vz = (numpy.atleast_1d(q) for q in out[:5])
            H = 0.5 * (vR**2.0 + vT**2.0 + vz**2.0) + evaluatePotentials(
                self._pot, R, z, use_physical=False
            )
            Omr = numpy.broadcast_to(float(numpy.atleast_1d(out[6])[0]), (npt,)).copy()
            Omz = numpy.broadcast_to(float(numpy.atleast_1d(out[8])[0]), (npt,)).copy()
            return H, Omr, Omz
        H = numpy.empty(npt)
        Omr = numpy.empty(npt)
        Omz = numpy.empty(npt)
        for i in range(npt):
            out = self._fam._xvFreqs(
                jr + dJr[i],
                lz,
                jz + dJz[i],
                numpy.array([self._thr[i]]),
                numpy.array([0.0]),
                numpy.array([self._thz[i]]),
            )
            R, vR, vT, z, vz = (float(numpy.atleast_1d(q)[0]) for q in out[:5])
            H[i] = 0.5 * (vR**2.0 + vT**2.0 + vz**2.0) + evaluatePotentials(
                self._pot, R, z, use_physical=False
            )
            Omr[i] = float(numpy.atleast_1d(out[6])[0])
            Omz[i] = float(numpy.atleast_1d(out[8])[0])
        return H, Omr, Omz

    def _dJ_of_AB(self, A, B):
        """The action perturbation of the generating function: dJ_i =
        sum_n n_i [A_n cos(n.theta) + B_n sin(n.theta)] on the grid"""
        dJr = numpy.zeros(len(self._thr))
        dJz = numpy.zeros(len(self._thr))
        for k, (nr, nz) in enumerate(self._modes):
            ph = nr * self._thr + nz * self._thz
            c, s = numpy.cos(ph), numpy.sin(ph)
            dJr += nr * (A[k] * c + B[k] * s)
            dJz += nz * (A[k] * c + B[k] * s)
        return dJr, dJz

    def _flatten(self, jr, lz, jz):
        """Fit the generating coefficients that flatten the true Hamiltonian
        on the torus: first-order FFT division to seed, then Gauss-Newton in
        coefficient space with LOCAL frequency weights"""
        ng = self._ngrid
        A = numpy.zeros(len(self._modes))
        B = numpy.zeros(len(self._modes))
        dJr = numpy.zeros(len(self._thr))
        dJz = numpy.zeros(len(self._thr))
        H, Omr, Omz = self._Hfield(jr, lz, jz, dJr, dJz)
        flat0 = numpy.ptp(H) / numpy.fabs(numpy.mean(H))
        skipped = 0.0
        # seed: Fourier division at the central frequencies
        cO_r, cO_z = numpy.median(Omr), numpy.median(Omz)
        c = numpy.fft.fft2(H.reshape(ng, ng) - numpy.mean(H)) / ng**2
        kk = numpy.fft.fftfreq(ng, d=1.0 / ng).astype(int)
        for k, (nr, nz) in enumerate(self._modes):
            cn = c[list(kk).index(nr), list(kk).index(nz)]
            nOm = nr * cO_r + nz * cO_z
            if numpy.fabs(nOm) < self._resonance_tol * numpy.fabs(cO_z):
                skipped += 2.0 * numpy.abs(cn)
                continue
            A[k] = -2.0 * numpy.real(cn) / nOm
            B[k] = 2.0 * numpy.imag(cn) / nOm
        flat = flat0
        flat_prev = flat0
        nclip = 0
        for it in range(self._polish + 1):
            dJr, dJz = self._dJ_of_AB(A, B)
            # trust region: stay inside the physical action domain (the
            # planar and circular edges), scaling the WHOLE correction so
            # neither J_R + dJ_R nor J_z + dJ_z crosses zero
            lam = 1.0
            for Jax, dJax in ((jr, dJr), (jz, dJz)):
                if dJax.min() < -0.85 * Jax:
                    lam = min(lam, 0.85 * Jax / -dJax.min())
            if lam < 1.0:
                A *= lam
                B *= lam
                dJr *= lam
                dJz *= lam
                nclip += 1
            H, Omr, Omz = self._Hfield(jr, lz, jz, dJr, dJz)
            flat = numpy.ptp(H) / numpy.fabs(numpy.mean(H))
            if it > 0 and flat >= 0.9 * flat_prev and lam == 1.0:
                # a full polish step no longer improves the flatness by
                # more than 10%: at the lattice/auxiliary floor, stop
                break
            flat_prev = flat
            # Gauss-Newton step: residual r = H - <H>, Jacobian rows
            # dH/dA_k = (n.Omega_local) cos(n.theta), dH/dB_k = (...) sin
            r = H - numpy.mean(H)
            nm = len(self._modes)
            Jac = numpy.empty((len(r), 2 * nm))
            for k, (nr, nz) in enumerate(self._modes):
                ph = nr * self._thr + nz * self._thz
                nOm = nr * Omr + nz * Omz
                Jac[:, k] = nOm * numpy.cos(ph)
                Jac[:, nm + k] = nOm * numpy.sin(ph)
            step, *_ = numpy.linalg.lstsq(Jac, -r, rcond=None)
            A = A + step[:nm]
            B = B + step[nm:]
        if skipped > 0.0:
            warnings.warn(
                "actionAngleTorusStaeckel: near-resonant Fourier modes were "
                f"skipped on torus (J_R, L_z, J_z) = ({jr:g}, {lz:g}, {jz:g}); "
                f"their power, {skipped:g}, bounds the flattening there",
                galpyWarning,
            )
        return {
            "A": A,
            "B": B,
            "E": float(numpy.mean(H)),
            "flat0": float(flat0),
            "flat": float(flat),
            "skipped": float(skipped),
            "nclip": nclip,
        }

    ####################### THE LOCAL MODEL OVER A J-STAR #####################
    def _fit_torus(self, jr, lz, jz):
        """The star of flattening fits and its local separable-quadratic
        model: S_n(J) and E(J) as stored quadratics whose ANALYTIC gradients
        supply the angle shift and the frequencies -- derivatives of the
        stored model, so the composite map is exactly symplectic for the
        model it evaluates"""
        if jr <= 0.0 or jz <= 0.0:
            raise ValueError(
                "actionAngleTorusStaeckel needs an interior torus with "
                "J_R > 0 and J_z > 0 (the radial and vertical shell/planar "
                "edges are degenerate for the local model)"
            )
        key = (round(jr, 12), round(lz, 12), round(jz, 12))
        if key in self._torus_cache:
            return self._torus_cache[key]
        steps = []
        for J, frac, floor in (
            (jr, self._starfrac[0], 1e-4),
            (lz, self._starfrac[1], 1e-3),
            (jz, self._starfrac[2], 1e-4),
        ):
            steps.append(max(frac * J, floor))
        hr, hL, hz = steps
        # keep the star inside J_R, J_z > 0 (possible by the interior check)
        lor = min(hr, 0.9 * jr)
        loz = min(hz, 0.9 * jz)
        pts = {
            "c": (jr, lz, jz),
            "rp": (jr + hr, lz, jz),
            "rm": (jr - lor, lz, jz),
            "Lp": (jr, lz + hL, jz),
            "Lm": (jr, lz - hL, jz),
            "zp": (jr, lz, jz + hz),
            "zm": (jr, lz, jz - loz),
        }
        fits = {k: self._flatten(*p) for k, p in pts.items()}
        # per-direction quadratic through the three points of each axis:
        # f(x0 + d) = f0 + a d + b d^2 with exact 3-point coefficients for
        # the possibly one-sided spacings
        model = {"J0": (jr, lz, jz), "fits": fits}
        for name, up, lo, hup, hlo in (
            ("r", "rp", "rm", hr, lor),
            ("L", "Lp", "Lm", hL, hL),
            ("z", "zp", "zm", hz, loz),
        ):
            for q in ("A", "B", "E"):
                f0 = fits["c"][q]
                fp = fits[up][q]
                fm = fits[lo][q]
                # exact 3-point quadratic for the (possibly unequal)
                # one-sided-away-from-edge spacings hup, hlo (both > 0)
                a = (hlo**2.0 * fp - hup**2.0 * fm - (hlo**2.0 - hup**2.0) * f0) / (
                    hup * hlo * (hup + hlo)
                )
                b = (hlo * fp + hup * fm - (hup + hlo) * f0) / (hup * hlo * (hup + hlo))
                model[f"d{q}_d{name}"] = a
                model[f"d2{q}_d{name}2"] = b
        self._torus_cache[key] = model
        return model

    def _model_eval(self, model, jr, lz, jz):
        """The stored quadratic model and its exact gradient at J"""
        J0 = model["J0"]
        d = (jr - J0[0], lz - J0[1], jz - J0[2])
        out = {}
        for q in ("A", "B", "E"):
            v = model["fits"]["c"][q]
            g = []
            for name, dd in zip(("r", "L", "z"), d):
                a = (
                    model["dA_d%s" % name]
                    if q == "A"
                    else (
                        model["dB_d%s" % name] if q == "B" else model["dE_d%s" % name]
                    )
                )
                b = (
                    model["d2A_d%s2" % name]
                    if q == "A"
                    else (
                        model["d2B_d%s2" % name]
                        if q == "B"
                        else model["d2E_d%s2" % name]
                    )
                )
                v = v + a * dd + b * dd**2.0
                g.append(a + 2.0 * b * dd)
            out[q] = v
            out["d" + q] = g
        return out

    ############################### EVALUATION ################################
    def __call__(self, jr, jphi, jz, angler, anglephi, anglez, maxiter=30):
        """
        Evaluate the torus map: (J, theta) -> (R, vR, vT, z, vz, phi).

        Parameters
        ----------
        jr, jphi, jz : float
            Actions (jphi = L_z).
        angler, anglephi, anglez : numpy.ndarray
            TRUE angles on the torus.
        maxiter : int, optional
            Maximum Newton iterations of the angle-shift inversion.

        Returns
        -------
        tuple
            (R, vR, vT, z, vz, phi)
        """
        jr, lz, jz = float(jr), float(jphi), float(jz)
        angler = numpy.atleast_1d(angler).astype(float)
        anglephi = numpy.atleast_1d(anglephi).astype(float)
        anglez = numpy.atleast_1d(anglez).astype(float)
        model = self._fit_torus(jr, lz, jz)
        ev = self._model_eval(model, jr, lz, jz)
        A, B = ev["A"], ev["B"]
        dA, dB = ev["dA"], ev["dB"]
        # invert theta = theta^S + dF/dJ(theta^S) for theta^S (2D Newton in
        # (theta_r, theta_z); the theta_phi shift is then explicit)
        thr = angler.copy()
        thz = anglez.copy()
        for _ in range(maxiter):
            shift_r = numpy.zeros_like(thr)
            shift_z = numpy.zeros_like(thz)
            dsr_dr = numpy.zeros_like(thr)
            dsr_dz = numpy.zeros_like(thr)
            dsz_dr = numpy.zeros_like(thr)
            dsz_dz = numpy.zeros_like(thr)
            for k, (nr, nz) in enumerate(self._modes):
                ph = nr * thr + nz * thz
                s, cph = numpy.sin(ph), numpy.cos(ph)
                # dF/dJ_i at fixed theta^S: the model-gradient coefficients
                fr = dA[0][k] * s + dB[0][k] * (1.0 - cph)
                fz = dA[2][k] * s + dB[2][k] * (1.0 - cph)
                shift_r += fr
                shift_z += fz
                dsr_dr += (dA[0][k] * cph + dB[0][k] * s) * nr
                dsr_dz += (dA[0][k] * cph + dB[0][k] * s) * nz
                dsz_dr += (dA[2][k] * cph + dB[2][k] * s) * nr
                dsz_dz += (dA[2][k] * cph + dB[2][k] * s) * nz
            Fr = thr + shift_r - angler
            Fz = thz + shift_z - anglez
            if max(numpy.fabs(Fr).max(), numpy.fabs(Fz).max()) < 1e-12:
                break
            det = (1.0 + dsr_dr) * (1.0 + dsz_dz) - dsr_dz * dsz_dr
            thr = thr - ((1.0 + dsz_dz) * Fr - dsr_dz * Fz) / det
            thz = thz - (-dsz_dr * Fr + (1.0 + dsr_dr) * Fz) / det
        # theta_phi^S from the explicit L_z shift
        shift_p = numpy.zeros_like(thr)
        for k, (nr, nz) in enumerate(self._modes):
            ph = nr * thr + nz * thz
            shift_p += dA[1][k] * numpy.sin(ph) + dB[1][k] * (1.0 - numpy.cos(ph))
        thp = anglephi - shift_p
        # the auxiliary actions along the torus and the family evaluation
        out = [numpy.empty(len(thr)) for _ in range(6)]
        for i in range(len(thr)):
            dJr = 0.0
            dJz = 0.0
            for k, (nr, nz) in enumerate(self._modes):
                ph = nr * thr[i] + nz * thz[i]
                amp = A[k] * numpy.cos(ph) + B[k] * numpy.sin(ph)
                dJr += nr * amp
                dJz += nz * amp
            oi = self._fam(
                jr + dJr,
                lz,
                jz + dJz,
                numpy.array([thr[i]]),
                numpy.array([thp[i]]),
                numpy.array([thz[i]]),
            )
            for j in range(6):
                out[j][i] = float(numpy.atleast_1d(oi[j])[0])
        return tuple(out)

    def xvFreqs(self, jr, jphi, jz, angler, anglephi, anglez, **kwargs):
        """Evaluate the torus map and the frequencies: the gradient of the
        stored E(J) model, exactly the frequencies of the trajectories the
        map returns"""
        out = self(jr, jphi, jz, angler, anglephi, anglez, **kwargs)
        Om = self.Freqs(jr, jphi, jz)
        return out + Om

    def Freqs(self, jr, jphi, jz):
        """Frequencies (Omega_R, Omega_phi, Omega_z): the analytic gradient
        of the stored quadratic E(J) model at J"""
        jr, lz, jz = float(jr), float(jphi), float(jz)
        model = self._fit_torus(jr, lz, jz)
        ev = self._model_eval(model, jr, lz, jz)
        return (ev["dE"][0], ev["dE"][1], ev["dE"][2])
