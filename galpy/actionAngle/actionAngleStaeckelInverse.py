###############################################################################
# actionAngleStaeckelInverse.py: inverse action-angle transformation for
#   axisymmetric Staeckel potentials through a momentum-matched canonical
#   map: each torus is the product of its two librations, in the prolate
#   spheroidal coordinates u and v, sampled exactly in their anomalies and
#   mapped onto the torus of equal (J_R, L = J_z + |L_z|, L_z) of a frozen
#   isochrone auxiliary by two point transformations, one per libration --
#   u onto the auxiliary's radius, v onto its polar angle -- each the one
#   that matches cumulative actions (a sine series in the anomaly) and each
#   cotangent-lifted, with the compensation of the three angles for the
#   moving turning points of both librations in closed form. The assembled
#   (J, theta) -> (x, v) map is exactly symplectic for ANY stored tables:
#   every derivative is the stored interpolant's own. (The Staeckel section
#   of canonical.tex in the fast-orbits repository.)
###############################################################################
import warnings
from contextlib import contextmanager

import numpy
from scipy.optimize import brentq

from ..potential import (
    IsochronePotential,
    OblateStaeckelWrapperPotential,
    epifreq,
    evaluatePotentials,
    evaluateR2derivs,
    evaluateRforces,
    rl,
    vcirc,
    verticalfreq,
)
from ..potential.Potential import _check_potential_list_and_deprecate
from ..util import conversion, coords, galpyWarning
from ..util._hermite import HermiteFamily3D, slope_at_zero
from ..util._optional_deps import _TQDM_LOADED
from .actionAngleInverse import actionAngleInverse
from .actionAngleInverseUtil import (
    _BOUND_MARGIN,
    _chunked,
    _fit_isochrone_vc2,
    _harmonics,
    _offset_spec_coeffs,
    _progress,
    _sine_galerkin,
    _sine_project,
    _sine_series,
    _spaced_grid,
    _spec_eval,
)
from .actionAngleIsochroneInverse import actionAngleIsochroneInverse
from .actionAngleStaeckel import _focal_length_of


def _cumulative(f):
    """The integral from tau = 0 of a periodic function sampled on the
    regular offset grid: its mean (the secular rate), the periodic part on
    the grid, that part's spectral coefficients (trimmed to the harmonics
    that carry it, the rest being round-off, so that evaluating the series
    elsewhere costs no more than it must), and its value at zero"""
    N = len(f)
    k = numpy.fft.fftfreq(N, d=1.0 / N)
    m = numpy.mean(f)
    fh = numpy.fft.fft(f - m)
    ah = numpy.zeros_like(fh)
    ah[1:] = fh[1:] / (1j * k[1:])
    q = numpy.real(numpy.fft.ifft(ah))
    cq = _offset_spec_coeffs(q)
    keep = numpy.flatnonzero(numpy.abs(cq) > 1e-16 * numpy.max(numpy.abs(cq)))
    # one vanishing coefficient beyond the last kept one, because the series
    # gives the last coefficient the Nyquist weight
    cq = numpy.append(cq[: keep[-1] + 1], 0.0) if len(keep) > 0 else cq
    return m, q, cq, _spec_eval(cq, 0.0)[0]


def _solve_increasing(f, fp, lo, hi):
    """The root of an increasing function f with derivative fp, for arrays
    of brackets [lo, hi] with f(lo) <= 0 <= f(hi): bisection until the
    brackets are narrow, then Newton steps kept inside the brackets, to
    round-off"""
    for _ in range(8):
        mid = 0.5 * (lo + hi)
        neg = f(mid) < 0.0
        lo, hi = numpy.where(neg, mid, lo), numpy.where(neg, hi, mid)
    x = 0.5 * (lo + hi)
    for _ in range(12):
        fx = f(x)
        neg = fx < 0.0
        lo, hi = numpy.where(neg, x, lo), numpy.where(neg, hi, x)
        xn = x - fx / fp(x)
        # a step out of the bracket is replaced by its midpoint; a converged
        # point sits on the bracket's edge and must not be thrown back
        xn = numpy.where((xn < lo) | (xn > hi), 0.5 * (lo + hi), xn)
        done = numpy.all(numpy.fabs(xn - x) <= 3e-15 * numpy.fabs(x))
        x = xn
        if done:
            break
    return x


def _contract(smt, D):
    """sum_m D[m] smt[m] for the harmonics smt, (M, npoints), and the
    coefficients D of one torus, (M,), or of every point's own torus,
    (M, npoints)"""
    return D @ smt if D.ndim == 1 else numpy.einsum("mp,mp->p", smt, D)


def _parse_target(target):
    """The phase-space rows (R, vR, vT, z, vz) of a target: an Orbit
    (possibly an array of orbits, or one evaluated at an array of times) or
    array rows (R, vR, vT, z, vz[, phi]) in internal units; phi, when
    present, is ignored because the potential is axisymmetric"""
    from ..orbit import Orbit

    if isinstance(target, Orbit):
        out = numpy.array(
            [
                numpy.atleast_1d(getattr(target, q)(use_physical=False)).ravel()
                for q in ("R", "vR", "vT", "z", "vz")
            ]
        )
    else:
        arr = numpy.atleast_2d(numpy.asarray(target, dtype="float"))
        if arr.ndim != 2 or arr.shape[1] not in (5, 6):
            raise ValueError(
                "target= rows must be (R, vR, vT, z, vz) or (R, vR, vT, z, vz, phi); "
                f"got shape {arr.shape}"
            )
        out = arr[:, :5].T.copy()
    if numpy.any(out[0] <= 0.0) or numpy.any(out[0] * numpy.fabs(out[2]) <= 0.0):
        raise ValueError(
            "target= needs R > 0 and L_z = R vT != 0 for every point: the grid is "
            "a box in |L_z| > 0"
        )
    return out


class actionAngleStaeckelInverse(actionAngleInverse):
    """Inverse action-angle transformation for axisymmetric Staeckel
    potentials through a momentum-matched canonical map onto an isochrone
    auxiliary (the ``auxiliary`` attribute); a torus is labelled by
    (E, L_z, I_3), the third integral in the convention
    p_u^2 = 2 delta^2 [E sinh^2 u - U(u) - I_3] - L_z^2 / sinh^2 u with the
    gauge V(pi/2) = 0"""

    def __init__(
        self,
        pot=None,
        Es=[-0.5],
        Lzs=[0.5],
        I3s=[0.1],
        setup_interp=False,
        Rmin=0.5,
        Rmax=2.0,
        Rinf=10.0,
        nE=9,
        nLz=9,
        Lzspacing="linear",
        nI3=9,
        mm_npt=32,
        mm_nta=None,
        auxiliary=None,
        maxiter=100,
        angle_tol=1e-12,
        progressbar=True,
        delta=None,
        u0=None,
        target=None,
        target_pad=1.5,
        target_minwidth=0.02,
        **kwargs,
    ):
        """
        Initialize an actionAngleStaeckelInverse object.

        Parameters
        ----------
        pot : Potential or a combined potential formed using addition (pot1+pot2+…)
            An axisymmetric potential of exact Staeckel form that supplies
            its focal length (an OblateStaeckelWrapperPotential or an exact
            Staeckel potential such as KuzminKutuzovStaeckelPotential); a
            general axisymmetric potential is accepted together with
            delta=, which wraps it in an OblateStaeckelWrapperPotential.
        Es : array-like or Quantity
            Energies of the tori to set up when setup_interp is False
            (paired with Lzs and I3s).
        Lzs : array-like or Quantity
            z-components of the angular momentum of the tori (non-zero).
        I3s : array-like or Quantity
            Third integrals of the tori, in the convention p_u^2 =
            2 delta^2 [E sinh^2 u - U(u) - I_3] - L_z^2 / sinh^2 u with the
            gauge V(pi/2) = 0 (an energy).
        setup_interp : bool, optional
            If True, set up a grid of tori in (L_z, E, I_3) spanning the
            circular angular momenta of [Rmin, Rmax], energies from the
            circular orbit's up to that of the planar orbit with apocentre
            Rinf, and third integrals from the planar orbit's to the shell
            orbit's, and interpolate canonically between them.
        Rmin : float or Quantity, optional
            Radius of the circular orbit with the smallest angular momentum
            of the interpolation grid.
        Rmax : float or Quantity, optional
            Radius of the circular orbit with the largest angular momentum
            of the interpolation grid.
        Rinf : float or Quantity, optional
            Apocentre, in the plane, of the orbit with the interpolation
            grid's largest energy at each angular momentum. The grid's
            energies are uniform in the logarithm of the apocentre of the
            planar orbit without angular momentum between each circular
            orbit's energy and this one (uniform in the energy for a
            logarithmic potential, and compressed toward the top in a
            potential with an escape energy, where the apocentre diverges),
            spaced quadratically at the circular face.
        nE : int, optional
            Number of energies of the interpolation grid.
        nLz : int, optional
            Number of angular momenta of the interpolation grid.
        Lzspacing : str, optional
            Spacing of the angular-momentum grid: 'linear' (default), 'sqrt'
            (uniform in sqrt(L_z), finer at small L_z) or 'log' (uniform in
            ln L_z).
        nI3 : int, optional
            Number of third integrals of the interpolation grid.
        mm_npt : int, optional
            Highest harmonic of the two momentum-matched anomaly maps (the
            v map's odd harmonics vanish by the symmetry of its libration
            about the midplane, so it stores the even ones up to this);
            the reconstruction converges spectrally in this, and a warning
            is raised when it does not suffice for a torus. The u-map needs
            a number of harmonics that grows with a torus's ratio of outer
            to inner radius in the plane, and the v-map one that grows with
            L / |L_z| as the torus approaches a polar orbit; on a grid the
            most demanding tori are the ones at the smallest angular
            momentum (set by Rmin) and the highest energy (set by Rinf),
            and the set-up raises when the harmonics cannot hold them.
        mm_nta : int, optional
            Number of anomaly samples per libration (even), used to sample
            the torus, to fit the maps, and for the quadratures of its
            actions and frequencies; must exceed 4 * mm_npt for the samples
            to resolve the maps' highest harmonic, which is 2 * mm_npt. The
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
        delta : float or Quantity, optional
            Focal length of the prolate spheroidal coordinate system, to
            wrap a general axisymmetric potential in the Staeckel
            approximation. A potential that supplies its own focal length
            accepts a delta= that agrees with it (to 1e-14) and rejects one
            that does not.
        u0 : float, callable, or 'fit', optional
            Reference u of the wrapping (see OblateStaeckelWrapperPotential),
            whose curve R_0 = delta sinh(u0) is where the Staeckel
            approximation of a general potential is most accurate. A number
            requires delta= and cannot be given with a potential that
            supplies its own focal length. A callable u0(E, Lz) (internal
            units, with L_z > 0) or 'fit' gives every (E, L_z) of an interpolated family
            its own reference curve, so that each torus is a torus of its
            own best-placed Staeckel model: 'fit' places the curve at the
            harmonic mean of the pericentre and apocentre of the planar
            orbits at (E, L_z), which weights the pericentre, where a
            disc's vertical structure is sharpest.
            The reference curve is construction-only gauge (the map's
            (R, z) <-> (u, v) uses the focal length alone), so this costs
            nothing at evaluation; requires setup_interp=True.
        target : Orbit or array_like, optional
            Localize the interpolation grid on the tori of a set of
            phase-space points: an Orbit (possibly an array of orbits, or
            one evaluated at an array of times) or rows (R, vR, vT, z, vz[,
            phi]) in internal units. The grid's box in its own coordinates
            (L_z, the energy variable w_E, the third-integral variable
            w_I) is then the points' own range, padded, instead of the
            default box between the circular orbits of Rmin and Rmax and
            from the circular to the Rinf energies; the same number of
            nodes then resolves a target orbit or stream far better. The
            labelling does a few root-finds per point, so subsample a very
            large set. Requires setup_interp=True.
        target_pad : float, optional
            Padding of the target's box: each axis is extended by this
            factor times the target's spread on both ends. Generous
            padding is load-bearing, because a box tight around the target
            puts it in the boundary layer of the interpolation stencil, a
            failure mode that is silent.
        target_minwidth : float, optional
            Minimum width of the target's box on each axis, as a fraction
            of that axis's default span: what keeps the box of a single
            orbit, whose spread is zero on every axis, a grid rather than a
            point.

        Notes
        -----
        - Angle conventions match those of the forward actionAngleStaeckel:
          theta_R = 0 at the inner u turning point and theta_z = 0 at the
          upward midplane crossing at pericentre.
        - 2026-09-24 - Started - Bovy (UofT)
        """
        actionAngleInverse.__init__(self, *[], **kwargs)
        if pot is None:
            raise OSError("Must specify pot= for actionAngleStaeckelInverse")
        self._setup_staeckel(_check_potential_list_and_deprecate(pot), delta, u0)
        if self._adaptive and not setup_interp:
            raise TypeError(
                "an adaptive reference curve (a callable u0(E, Lz) or u0='fit') "
                "builds an interpolated family and requires setup_interp=True"
            )
        self._target = None if target is None else _parse_target(target)
        self._target_pad, self._target_minwidth = target_pad, target_minwidth
        if self._target is not None and not setup_interp:
            raise TypeError(
                "target= localizes the interpolation grid and requires setup_interp=True"
            )
        if mm_nta is None:
            mm_nta = 8 * mm_npt
        if mm_nta % 2 == 1:
            raise ValueError("mm_nta has to be even")
        if mm_nta <= 4 * mm_npt:
            # the maps' highest harmonic is 2 mm_npt, which mm_nta uniform
            # samples only resolve below their Nyquist harmonic mm_nta / 2
            raise ValueError(
                "mm_nta must exceed 4 * mm_npt for the anomaly samples to resolve the maps' harmonics"
            )
        self._ntau = mm_nta
        self._npt = mm_npt
        if mm_npt < 2:
            raise ValueError(
                "mm_npt must be at least 2: the v anomaly map has even harmonics only and needs one"
            )
        self._nforDm = numpy.arange(1, mm_npt + 1)
        # the v map's odd harmonics vanish: the libration is symmetric about
        # the midplane, and with it the matching, eta(pi - tau) = pi - eta(tau)
        self._nforDmv = numpy.arange(2, mm_npt + 1, 2)
        self._nptv = len(self._nforDmv)
        self._maxiter = maxiter
        self._angle_tol = angle_tol
        self._progressbar = progressbar and _TQDM_LOADED
        self._interp = setup_interp
        self._circ_cache = {}
        self._ush_cache = {}
        if not setup_interp:
            self._Es = conversion._parse_grid_quantity(
                Es, conversion.parse_energy, vo=self._vo
            ).astype("float")
            self._Lzs = conversion._parse_grid_quantity(
                Lzs, conversion.parse_angmom, ro=self._ro, vo=self._vo
            ).astype("float")
            self._I3s = conversion._parse_grid_quantity(
                I3s, conversion.parse_energy, vo=self._vo
            ).astype("float")
            if not len(self._Es) == len(self._Lzs) == len(self._I3s):
                raise ValueError("Es, Lzs, and I3s have to have the same length")
            if numpy.any(self._Lzs == 0.0):
                raise ValueError(
                    "L_z = 0 is not supported: the v libration of a polar orbit reaches the axis"
                )
        else:
            self._setup_grid(
                conversion.parse_length(Rmin, ro=self._ro),
                conversion.parse_length(Rmax, ro=self._ro),
                conversion.parse_length(Rinf, ro=self._ro),
                nE,
                nLz,
                nI3,
                Lzspacing=Lzspacing,
            )
        # sample every torus once (exact placement), then fix the frozen
        # auxiliary and lift every torus onto it, libration by libration
        # (all torus-dependence beyond the auxiliary lives in the two
        # momentum-matched maps, whose compensation is closed-form), then
        # compute the tables
        self._sample_all()
        self._setup_auxiliary(auxiliary)
        self._setup_tori()
        self._check_consistent_units()
        return None

    def _setup_staeckel(self, pot, delta, u0):
        """The Staeckel model the tori live in: the potential itself when it
        supplies its focal length, or its Staeckel approximation at the
        given focal length; with an adaptive reference curve, that model
        is the reference one, and every (E, L_z) of the family gets its
        own (_chart)"""
        if isinstance(u0, str) and u0 != "fit":
            raise ValueError("the only string value u0= accepts is 'fit'")
        self._adaptive = callable(u0) or u0 == "fit"
        self._u0_func = u0 if callable(u0) else self._u0_fit
        self._chart_cache = {}
        if self._adaptive:
            # the reference wrapper keeps the default reference curve
            u0 = None
        own = _focal_length_of(pot)
        if delta is not None:
            delta = conversion.parse_length(delta, ro=self._ro)
            if own is not None:
                # the potential's own focal length wins; a delta= that agrees
                # with it is harmless, one that does not is a contradiction
                if u0 is not None:
                    raise TypeError(
                        "u0= cannot be given with a potential that already "
                        "supplies its focal length"
                    )
                if not numpy.fabs(delta - own) <= 1e-14 * numpy.fabs(own):
                    raise TypeError(
                        "delta= {:g} disagrees with the focal length {:g} that the "
                        "potential supplies; pass the potential without delta= to "
                        "use its own".format(delta, own)
                    )
                delta = None
        if delta is not None:
            self._staeckelwrap = (
                OblateStaeckelWrapperPotential(pot=pot, delta=delta)
                if u0 is None
                else OblateStaeckelWrapperPotential(pot=pot, delta=delta, u0=u0)
            )
        elif u0 is not None:
            raise TypeError("u0= requires delta=")
        elif isinstance(pot, OblateStaeckelWrapperPotential):
            self._staeckelwrap = pot
        elif own is not None:
            self._staeckelwrap = OblateStaeckelWrapperPotential(pot=pot, delta=own)
        else:
            raise OSError(
                "actionAngleStaeckelInverse requires a potential of Staeckel "
                "form that supplies its focal length (an "
                "OblateStaeckelWrapperPotential or an exact Staeckel potential "
                "such as KuzminKutuzovStaeckelPotential), or delta= to wrap a "
                "general axisymmetric potential in the Staeckel approximation"
            )
        self._pot = pot
        self._delta = self._staeckelwrap._delta
        # the potential the per-(E, L_z) charts of an adaptive family wrap:
        # always the raw one (wrapping a wrapper would approximate the
        # approximation)
        self._chart_pot = self._staeckelwrap._pot
        return None

    def _chart(self, E, Lz):
        """The Staeckel model of the (E, L_z) tori: the family's single
        wrapper or, with an adaptive reference curve, the wrapper at this
        (E, |L_z|)'s own u0, cached"""
        if not self._adaptive:
            return self._staeckelwrap
        key = (float(E), float(numpy.fabs(Lz)))
        if key not in self._chart_cache:
            self._chart_cache[key] = OblateStaeckelWrapperPotential(
                pot=self._chart_pot,
                delta=self._delta,
                u0=float(self._u0_func(key[0], key[1])),
            )
        return self._chart_cache[key]

    @contextmanager
    def _in_chart(self, E, Lz):
        """Run the chart-dependent construction of the (E, L_z) tori (their
        sampling, turning points and edge relations) in their own Staeckel
        model"""
        save = self._staeckelwrap
        self._staeckelwrap = self._chart(E, Lz)
        try:
            yield
        finally:
            self._staeckelwrap = save

    def _u0_fit(self, E, Lz):
        """The reference curve of u0='fit': at the harmonic mean of the
        pericentre and apocentre of the planar orbits at (E, L_z), delta
        sinh(u0) = 2 / (1/R_p + 1/R_a), from the potential's midplane, which
        every wrapper reproduces; the circular radius at the circular
        energy. The harmonic mean weights the pericentre, where a disc's
        vertical structure is sharpest: measured on MWPotential2014 against
        the arithmetic midpoint (which moves the curve far out on the
        energetic rows, whose v maps then do not resolve, and is worse than
        the default curve near R = 1), the geometric mean and the
        guiding-centre radius, it resolves every map and gives the smallest
        energy spread on seven of eight test tori"""
        ell = numpy.fabs(Lz)
        pot = self._chart_pot

        def phieff(R):
            return evaluatePotentials(pot, R, 0.0, use_physical=False) + ell**2 / (
                2.0 * R**2
            )

        Rc = rl(pot, ell, use_physical=False)
        Ec = phieff(Rc)
        Emax = phieff(self._Rinf)
        if E <= Ec + 1e-12 * (1.0 + numpy.fabs(Ec)):
            return numpy.arcsinh(Rc / self._delta)
        # the centrifugal term brackets the pericentre from below; the
        # grid's top energy is that of the planar orbit with apocentre Rinf
        Rp = brentq(lambda R: phieff(R) - E, 1e-4 * Rc, Rc, xtol=1e-12)
        Ra = (
            self._Rinf
            if E >= Emax - 1e-12 * (1.0 + numpy.fabs(Emax))
            else brentq(lambda R: phieff(R) - E, Rc, self._Rinf, xtol=1e-12)
        )
        return numpy.arcsinh(2.0 / (1.0 / Rp + 1.0 / Ra) / self._delta)

    # ---------- the separated Hamilton-Jacobi equation
    def _U(self, u):
        u = numpy.asarray(u, dtype="float")
        return self._staeckelwrap._U(u.ravel()).reshape(u.shape)

    def _dU(self, u):
        u = numpy.asarray(u, dtype="float")
        return self._staeckelwrap._dUdu(u.ravel()).reshape(u.shape)

    def _d2U(self, u):
        u = numpy.asarray(u, dtype="float")
        return self._staeckelwrap._d2Udu2(u.ravel()).reshape(u.shape)

    def _V(self, v):
        v = numpy.asarray(v, dtype="float")
        return self._staeckelwrap._V(v.ravel()).reshape(v.shape)

    def _dV(self, v):
        v = numpy.asarray(v, dtype="float")
        return self._staeckelwrap._dVdv(v.ravel()).reshape(v.shape)

    def _d2V(self, v):
        v = numpy.asarray(v, dtype="float")
        return self._staeckelwrap._d2Vdv2(v.ravel()).reshape(v.shape)

    def _Wu(self, u, E, Lz, I3):
        """p_u^2 on the (E, L_z, I_3) torus"""
        sh2 = numpy.sinh(u) ** 2
        return 2.0 * self._delta**2 * (E * sh2 - self._U(u) - I3) - Lz**2 / sh2

    def _Wv(self, v, E, Lz, I3):
        """p_v^2 on the (E, L_z, I_3) torus"""
        sn2 = numpy.sin(v) ** 2
        return 2.0 * self._delta**2 * (E * sn2 + self._V(v) + I3) - Lz**2 / sn2

    def _dWu(self, u, E, Lz):
        """d p_u^2 / du (independent of I_3)"""
        return (
            2.0 * self._delta**2 * (E * numpy.sinh(2.0 * u) - self._dU(u))
            + 2.0 * Lz**2 * numpy.cosh(u) / numpy.sinh(u) ** 3
        )

    def _dWv(self, v, E, Lz):
        """d p_v^2 / dv (independent of I_3)"""
        return (
            2.0 * self._delta**2 * (E * numpy.sin(2.0 * v) + self._dV(v))
            + 2.0 * Lz**2 * numpy.cos(v) / numpy.sin(v) ** 3
        )

    def _d2Wu(self, u, E, Lz):
        """d^2 p_u^2 / du^2 (independent of I_3)"""
        sh2 = numpy.sinh(u) ** 2
        return (
            2.0 * self._delta**2 * (2.0 * E * numpy.cosh(2.0 * u) - self._d2U(u))
            - Lz**2 * (6.0 + 4.0 * sh2) / sh2**2
        )

    def _d2Wv(self, v, E, Lz):
        """d^2 p_v^2 / dv^2 (independent of I_3)"""
        sn2 = numpy.sin(v) ** 2
        return (
            2.0 * self._delta**2 * (2.0 * E * numpy.cos(2.0 * v) + self._d2V(v))
            - 2.0 * Lz**2 * (3.0 - 2.0 * sn2) / sn2**2
        )

    def _tp_derivs_u(self, u, E, Lz):
        """Closed-form derivatives of a u turning point (a root of p_u^2)
        with respect to (E, I_3, L_z), by the level-set rule"""
        sh2 = numpy.sinh(u) ** 2
        d2 = self._delta**2
        return -numpy.array([2.0 * d2 * sh2, -2.0 * d2, -2.0 * Lz / sh2]) / float(
            self._dWu(u, E, Lz)
        )

    def _tp_derivs_v(self, v, E, Lz):
        """Closed-form derivatives of the v turning point with respect to
        (E, I_3, L_z), by the level-set rule"""
        sn2 = numpy.sin(v) ** 2
        d2 = self._delta**2
        return -numpy.array([2.0 * d2 * sn2, 2.0 * d2, -2.0 * Lz / sn2]) / float(
            self._dWv(v, E, Lz)
        )

    # ---------- the circular, shell, and planar edges
    def _circular(self, Lz):
        """The circular orbit of angular momentum |L_z|: its radius and
        energy, its u, and its epicycle, vertical, and circular
        frequencies, cached"""
        ell = numpy.fabs(Lz)
        if ell not in self._circ_cache:
            Rc = rl(self._staeckelwrap, ell, use_physical=False)
            Ec = evaluatePotentials(
                self._staeckelwrap, Rc, 0.0, use_physical=False
            ) + ell**2 / (2.0 * Rc**2)
            wrap = self._chart(Ec, ell)
            self._circ_cache[ell] = {
                "Rc": Rc,
                "Ec": Ec,
                "uc": numpy.arcsinh(Rc / self._delta),
                "kappa": epifreq(wrap, Rc, use_physical=False),
                "nu": verticalfreq(wrap, Rc, use_physical=False),
                "Omc": ell / Rc**2,
            }
        return self._circ_cache[ell]

    def _ushell(self, E, Lz):
        """The u of the shell orbit of (E, L_z), where p_u^2 attains its
        maximum over u (independent of I_3), cached"""
        key = (E, Lz)
        if key not in self._ush_cache:
            us = numpy.geomspace(1e-4, 20.0, 400)
            with numpy.errstate(invalid="ignore", over="ignore"):
                # the potential may not evaluate at the scan's extreme radii
                W = self._Wu(us, E, Lz, 0.0)
            finite = numpy.isfinite(W)
            us, W = us[finite], W[finite]
            j = int(numpy.argmax(W))
            if j == 0 or j == len(us) - 1:
                raise ValueError(
                    f"No bound u libration exists at (E, L_z) = ({E}, {Lz}): "
                    "the torus is unbound"
                )
            self._ush_cache[key] = brentq(
                lambda u: float(self._dWu(u, E, Lz)), us[j - 1], us[j + 1], xtol=1e-14
            )
        return self._ush_cache[key]

    def _dushell(self, ush, E, Lz):
        """Derivatives of the shell u with respect to E and to L_z, by the
        level-set rule on d p_u^2 / du = 0"""
        d2W = float(self._d2Wu(ush, E, Lz))
        return (
            -2.0 * self._delta**2 * numpy.sinh(2.0 * ush) / d2W,
            -4.0 * Lz * numpy.cosh(ush) / numpy.sinh(ush) ** 3 / d2W,
        )

    def _I3_planar(self, E, Lz):
        """I_3 of the planar orbit (J_z = 0) of (E, L_z): closed form in the
        gauge V(pi/2) = 0, where p_v^2(pi/2) = 2 delta^2 [E + I_3] - L_z^2"""
        return Lz**2 / (2.0 * self._delta**2) - E

    def _I3_shell(self, E, Lz, ush):
        """I_3 of the shell orbit (J_R = 0) of (E, L_z): the value at which
        p_u^2 vanishes at its maximum"""
        sh2 = numpy.sinh(ush) ** 2
        return E * sh2 - float(self._U(ush)) - Lz**2 / (2.0 * self._delta**2 * sh2)

    def _turning_points(self, E, Lz, I3):
        """The turning points of the (E, L_z, I_3) torus as the midpoint and
        half-width of the u libration and the half-width of the v libration
        (about the midplane), with the shell u and whether either libration
        is degenerate; raises for labels that do not define a torus"""
        circ = self._circular(Lz)
        Ec = circ["Ec"]
        tolE = 1e-12 * (1.0 + numpy.fabs(Ec))
        if E < Ec - tolE:
            raise ValueError(
                f"No orbit exists at E = {E}, L_z = {Lz}: the energy lies below "
                "the circular orbit's"
            )
        if E < Ec + tolE:
            # the circular orbit: both librations degenerate
            return circ["uc"], 0.0, 0.0, circ["uc"], True, True
        ush = self._ushell(E, Lz)
        Ipl, Ish = self._I3_planar(E, Lz), self._I3_shell(E, Lz, ush)
        tolI = 1e-12 * (1.0 + numpy.fabs(Ipl) + numpy.fabs(Ish))
        if I3 > Ish + tolI:
            raise ValueError(
                f"No orbit exists at (E, L_z, I_3) = ({E}, {Lz}, {I3}): I_3 "
                f"lies above the shell orbit's, {Ish}"
            )
        if I3 < Ipl - tolI:
            raise ValueError(
                f"No orbit exists at (E, L_z, I_3) = ({E}, {Lz}, {I3}): I_3 "
                f"lies below the planar orbit's, {Ipl}"
            )
        udeg, vdeg = I3 >= Ish - tolI, I3 <= Ipl + tolI
        if udeg:
            umin = umax = ush
        else:
            Wu = lambda u: float(self._Wu(u, E, Lz, I3))
            ulo = ush
            while Wu(ulo) > 0.0:
                ulo *= 0.7
            uhi = ush
            while Wu(uhi) > 0.0:
                uhi *= 1.3
            umin = brentq(Wu, ulo, ush, xtol=1e-14)
            umax = brentq(Wu, ush, uhi, xtol=1e-14)
        if vdeg:
            vmin = 0.5 * numpy.pi
        else:
            Wv = lambda v: float(self._Wv(v, E, Lz, I3))
            vlo = 0.35 * numpy.pi
            while Wv(vlo) > 0.0:
                vlo *= 0.7
            vmin = brentq(Wv, vlo, 0.5 * numpy.pi, xtol=1e-14)
        return (
            0.5 * (umin + umax),
            0.5 * (umax - umin),
            0.5 * numpy.pi - vmin,
            ush,
            udeg,
            vdeg,
        )

    # ---------- node construction: exact tori by quadrature
    def _sample_torus(self, E, Lz, I3):
        """Exact phase-space samples along both librations, each
        parametrized by its tau anomaly on the offset grid tau_k =
        2 pi (k + 1/2)/N, which keeps every sample off the turning points;
        placement is exact by construction (the momenta from the separated
        Hamilton-Jacobi equation, not from a fit). A degenerate libration
        (a shell orbit's in u, a planar orbit's in v) has no samples."""
        uc, wu, wv, ush, udeg, vdeg = self._turning_points(E, Lz, I3)
        tau = 2.0 * numpy.pi * (numpy.arange(self._ntau) + 0.5) / self._ntau
        sgn = numpy.where(tau < numpy.pi, 1.0, -1.0)
        smp = {
            "tau": tau,
            "E": E,
            "Lz": Lz,
            "I3": I3,
            "uc": uc,
            "wu": wu,
            "wv": wv,
            "ush": ush,
            "u": None,
            "pu": None,
            "jr": 0.0,
            "v": None,
            "pv": None,
            "jz": 0.0,
        }
        if not udeg:
            u = uc - wu * numpy.cos(tau)
            pu = sgn * numpy.sqrt(numpy.clip(self._Wu(u, E, Lz, I3), 0.0, None))
            smp["u"], smp["pu"] = u, pu
            smp["jr"] = float(numpy.mean(pu * wu * numpy.sin(tau)))
        if not vdeg:
            v = 0.5 * numpy.pi - wv * numpy.cos(tau)
            pv = sgn * numpy.sqrt(numpy.clip(self._Wv(v, E, Lz, I3), 0.0, None))
            smp["v"], smp["pv"] = v, pv
            smp["jz"] = float(numpy.mean(pv * wv * numpy.sin(tau)))
        return smp

    def _sample_all(self):
        self._samples = []
        for E, Lz, I3 in zip(self._Es, self._Lzs, self._I3s):
            with self._in_chart(E, Lz):
                self._samples.append(self._sample_torus(E, Lz, I3))
        return None

    # ---------- the auxiliary: one isochrone for the whole family
    @staticmethod
    def _aux_vflux(L, c, zeta):
        """The auxiliary's polar action flux p^A_theta d theta^A / d zeta at
        the polar anomaly zeta, in closed form: the auxiliary's polar
        libration is cos(theta^A) = sqrt(c) cos(zeta) with c = sin^2 i =
        1 - L_z^2 / L^2, in which its angle relations are affine"""
        s2 = numpy.sin(zeta) ** 2
        return L * c * s2 / (1.0 - c * numpy.cos(zeta) ** 2)

    @staticmethod
    def _aux_vcumulative(L, c, zeta):
        """The auxiliary's cumulative polar action from the northern turning
        point, A^A_theta(zeta) = int_0^zeta f^A_theta, and its partials with
        respect to (L, c) at fixed zeta, all closed forms: the flux
        L c sin^2 zeta/(1 - c cos^2 zeta) is L [1 - (1 - c)/(1 - c cos^2
        zeta)], whose integral is L [zeta - s Theta(zeta)] with s = sqrt(1 -
        c) and Theta = arctan(tan(zeta)/s) continued through its branches by
        the winding number of zeta about pi; the secular rate over a
        libration is L - |L_z|, the auxiliary's polar action. As for the
        radial cumulative action (_cumulative), the linear parts
        are gathered so that no term exceeds the O(c) result: Theta = zeta +
        psi with psi = arctan(tan(zeta)/s) - arctan(tan(zeta)) = arctan[c
        tan(zeta)/((1 + s)(s + tan^2 zeta))] (no branch to continue), and
        1 - s = c/(1 + s)."""
        zeta = numpy.asarray(zeta, dtype="float")
        s = numpy.sqrt(1.0 - c)
        t = numpy.tan(zeta)
        psi = numpy.arctan(c * t / ((1.0 + s) * (s + t**2)))
        A = L * (c * zeta / (1.0 + s) - s * psi)
        return A, A / L, L * (0.5 * (zeta + psi) / s - 0.5 * t / (s**2 + t**2))

    def _aux_vflux_derivs(self, L, c, zeta):
        """The auxiliary's polar action flux (_aux_vflux) with its partials
        in zeta, L and c, closed forms: with D = 1 - c cos^2 zeta, f = L c
        sin^2 zeta / D, df/dzeta = 2 L c (1 - c) sin zeta cos zeta / D^2,
        df/dL = f / L and df/dc = L sin^2 zeta / D^2"""
        sz, cz = numpy.sin(zeta), numpy.cos(zeta)
        D = 1.0 - c * cz**2
        f = L * c * sz**2 / D
        return f, 2.0 * L * c * (1.0 - c) * sz * cz / D**2, f / L, L * sz**2 / D**2

    def _aux_vcumulative2(self, L, c, zeta):
        """The auxiliary's cumulative polar action (_aux_vcumulative) with
        its first and second partials in (L, c) at fixed zeta: A is linear
        in L, so A_L = A/L, A_LL = 0 and A_Lc = A_c/L; and with Theta = zeta
        + varpi, s = sqrt(1 - c) and t = tan zeta, from dTheta/dc = t/[2 s
        (s^2 + t^2)] and ds/dc = -1/(2s), A_c = L [Theta/(2s) - t/(2 (s^2 +
        t^2))] and A_cc = L [Theta/(4 s^3) + t/(4 s^2 (s^2 + t^2)) - t/(2
        (s^2 + t^2)^2)]. Verified against a forty-digit evaluation of the
        plain form to 1e-13."""
        A, A_L, A_c = self._aux_vcumulative(L, c, zeta)
        s = numpy.sqrt(1.0 - c)
        t = numpy.tan(zeta)
        Theta = zeta + numpy.arctan(c * t / ((1.0 + s) * (s + t**2)))
        q = s**2 + t**2
        A_cc = L * (Theta / (4.0 * s**3) + t / (4.0 * s**2 * q) - 0.5 * t / q**2)
        return A, A_L, A_c, 0.0 * A, A_c / L, A_cc

    def _setup_auxiliary(self, auxiliary):
        """Fix the frozen isochrone auxiliary of the whole family and lift
        every torus onto it, libration by libration.

        The momentum-matched maps lift every torus onto its equal-action
        auxiliary torus, so the auxiliary's only job is to be a
        well-conditioned global surrogate of the target. Unless one is
        given, its rotation curve is fitted to the target's over the
        sampled radial range. Every torus's truncated u-lift is then
        required to clear escape by a fraction of its own auxiliary torus's
        binding energy, which fails only when the stored anomaly map is
        under-resolved."""
        if auxiliary is not None:
            if not isinstance(auxiliary, IsochronePotential):
                raise TypeError("auxiliary= has to be an IsochronePotential")
            self._GM, self._b = auxiliary._amp, auxiliary.b
        else:
            self._GM, self._b = self._fit_auxiliary()
        self.auxiliary = (
            auxiliary
            if auxiliary is not None
            else IsochronePotential(amp=self._GM, b=self._b)
        )
        # the isochrone inverse supplies the closed forms of the auxiliary's
        # tori that the lifts, the tables and the evaluation use
        self._aAIinv = actionAngleIsochroneInverse(ip=self.auxiliary)
        self._lifts = []
        for ii in _progress(
            range(len(self._samples)), "lifting tori", self._progressbar
        ):
            smp = self._samples[ii]
            L = smp["jz"] + numpy.fabs(smp["Lz"])
            lift_u = lift_v = None
            if smp["u"] is not None:
                lift_u = self._lift_u(smp, smp["jr"], L)
                rA, pA = lift_u["rA"], lift_u["pA"]
                EAs = numpy.max(
                    0.5 * (pA**2 + L**2 / rA**2)
                    - self._GM / (self._b + numpy.sqrt(self._b**2 + rA**2))
                )
                if EAs >= _BOUND_MARGIN * self._aAIinv._torus_E(smp["jr"], L):
                    Rinner = self._delta * numpy.sinh(smp["uc"] - smp["wu"])
                    Router = self._delta * numpy.sinh(smp["uc"] + smp["wu"])
                    raise RuntimeError(
                        "The momentum-matched lift of the (E, L_z, I_3) = "
                        f"({smp['E']:g}, {smp['Lz']:g}, {smp['I3']:g}) torus is "
                        "not bound in the fitted auxiliary: the u anomaly map is "
                        "under-resolved for it. Its u libration spans the "
                        f"radii {Rinner:g} to {Router:g} in the plane, a ratio of "
                        f"{Router / Rinner:.0f}, and the map needs a number of "
                        "harmonics of the order of that ratio, against the "
                        f"mm_npt = {self._npt} given; the most eccentric torus "
                        "of a grid is the one at its smallest angular momentum "
                        "(Rmin) and highest energy (Rinf). Raise mm_npt (and, "
                        "with it, mm_nta), or raise Rmin or lower Rinf"
                    )
            if smp["v"] is not None:
                lift_v = self._lift_v(
                    smp, L, smp["jz"] * (smp["jz"] + 2.0 * numpy.fabs(smp["Lz"])) / L**2
                )
            self._lifts.append((lift_u, lift_v))
        return None

    def _fit_auxiliary(self):
        """Fit the isochrone's (GM, b) to the potential's rotation curve
        over the sampled radial range (zero-point free, log residuals on
        geometrically spaced radii)"""
        rlo = min(
            self._delta * numpy.sinh(smp["uc"] - smp["wu"]) for smp in self._samples
        )
        rhi = max(
            self._delta
            * numpy.sqrt(
                numpy.sinh(smp["uc"] + smp["wu"]) ** 2 + numpy.sin(smp["wv"]) ** 2
            )
            for smp in self._samples
        )
        rlo = max(rlo, 1e-3 * rhi)
        rf = numpy.geomspace(rlo, rhi, 25)
        return _fit_isochrone_vc2(
            rf, numpy.log(vcirc(self._staeckelwrap, rf, use_physical=False) ** 2)
        )

    # ---------- the momentum-matched maps and their lifts
    @staticmethod
    def _map(tau, Dm, ms):
        """The anomaly map eta(tau) = tau + sum_m D_m sin(m tau) over the
        harmonics ms, its derivative, and the sine matrix it is built from;
        the coefficients are those of one torus, (M,), or of every point's
        own torus, (M, npoints)"""
        smt, cmt = _harmonics(tau, ms)
        return (
            tau + _contract(smt, Dm),
            1.0 + _contract(cmt, ms[:, None] * Dm if Dm.ndim == 2 else ms * Dm),
            smt,
        )

    def _match_anomaly(self, tau, ft, flux, cumulative, ms):
        """The momentum-matched map of one libration onto the auxiliary's:
        match cumulative actions from the inner turning point, eta(tau) =
        A_A^{-1}(A_t(tau)), for the target's flux ft = p dq/dtau sampled on
        the offset grid and the auxiliary's flux and cumulative action, both
        closed forms in its anomaly. Both cumulatives share the linear part
        J (the equal-action choice of the auxiliary torus), so eta - tau is
        periodic and, by time-reversal parity, a pure sine series; the map
        is stored truncated to the harmonics ms, and returned together with
        the truncated map's values and derivative on the grid."""
        mt, qt, _, qt0 = _cumulative(ft)
        At = mt * tau + qt - qt0
        # the auxiliary's secular rate is the action to round-off; the ratio
        # of the two rates makes the linear parts coincide exactly
        scale = mt / (cumulative(2.0 * numpy.pi) / (2.0 * numpy.pi))
        # pointwise Newton on the closed-form cumulative action, with the
        # closed-form flux as its derivative, started from the inverse of
        # the auxiliary's cumulative action interpolated between its grid
        # values (both cumulatives are monotone); the matching is monotone,
        # and a residual left by the iteration shows in the truncation
        # diagnostic of _node_tables
        eta = numpy.interp(At, scale * cumulative(tau), tau)
        for _ in range(200):
            fres = scale * cumulative(eta) - At
            fp = numpy.maximum(scale * flux(eta), 1e-10 * mt)
            eta += numpy.clip(-fres / fp, -0.5, 0.5)
            if numpy.max(numpy.fabs(fres)) < 1e-13 * max(mt, 1e-10):
                break
        Dm = _sine_project(eta - tau, ms)
        shift, dshift = _sine_series(self._ntau, Dm, ms)
        return Dm, tau + shift, 1.0 + dshift

    def _lift_u(self, smp, jr, L):
        """The u libration lifted onto the auxiliary's radial libration of
        equal action: the map's coefficients, its values on the grid, and
        the lifted radius and radial momentum (rebuilt from the TRUNCATED
        map cotangent-consistently, p^A = pi' p_u, so the reconstruction is
        the one the stored tables define)"""
        tau = smp["tau"]
        a, e = self._aAIinv._orbital_params(jr, L)
        ft = smp["pu"] * smp["wu"] * numpy.sin(tau)
        Dm, eta, deta = self._match_anomaly(
            tau,
            ft,
            lambda x: self._aAIinv._flux_derivs(a, e, x)[0],
            lambda x: self._aAIinv._cumulative(a, e, x)[0],
            self._nforDm,
        )
        rA, _, gA = self._aAIinv._profile(a, e, eta)
        return {
            "a": a,
            "e": e,
            "Dm": Dm,
            "eta": eta,
            "rA": rA,
            "pA": ft / (gA * numpy.sin(eta) * deta),
        }

    def _lift_v(self, smp, L, c):
        """The v libration lifted onto the auxiliary's polar libration of
        equal action: the map's coefficients and values, and the lifted
        polar angle and momentum"""
        tau = smp["tau"]
        ft = smp["pv"] * smp["wv"] * numpy.sin(tau)
        Dm, zeta, dzeta = self._match_anomaly(
            tau,
            ft,
            lambda x: self._aux_vflux(L, c, x),
            lambda x: self._aux_vcumulative(L, c, x)[0],
            self._nforDmv,
        )
        sinth = numpy.sqrt(1.0 - c * numpy.cos(zeta) ** 2)
        return {
            "Dm": Dm,
            "zeta": zeta,
            "sinth": sinth,
            "pA": ft * sinth / (numpy.sqrt(c) * numpy.sin(zeta) * dzeta),
        }

    def _map_slopes_u(self, smp, jr, L, a, e, eta, dJr, dL, dsup):
        """The derivatives of the u map's coefficients with respect to
        (E, I_3, L_z), on one torus from that torus alone: the variation of
        the matching condition A^A(eta(tau); a, e) = A(tau; E, L_z, I_3) at
        fixed anomaly,

            f^A(eta) sum_m dD_m/dalpha sin(m tau)
                = dA/dalpha|_tau - dA^A/da a_alpha - dA^A/de e_alpha ,

        a LINEAR problem for the dD_m/dalpha with the vanishing flux f^A
        multiplying the unknowns, solved by Galerkin projection onto the
        sine basis (_sine_galerkin). The target's flux derivative at fixed
        tau goes through the moving turning points and is regular there
        (its numerator vanishes with p_u), so it integrates spectrally like
        the flux itself; the auxiliary's cumulative action varies through
        its torus parameters' closed-form chains, with dJ^A_r/dalpha and
        dL^A/dalpha from the torus's period matrix and the closed-form
        partials of _cumulative."""
        tau, u, pu, wu = smp["tau"], smp["u"], smp["pu"], smp["wu"]
        E, Lz = smp["E"], smp["Lz"]
        d2 = self._delta**2
        st, ct = numpy.sin(tau), numpy.cos(tau)
        dudtau = wu * st
        dWdu = self._dWu(u, E, Lz)
        sh2 = numpy.sinh(u) ** 2
        dWda = (2.0 * d2 * sh2, -2.0 * d2 * numpy.ones_like(u), -2.0 * Lz / sh2)
        _, FAa, FAe = self._aAIinv._cumulative(a, e, eta)
        fAeta = self._aAIinv._flux_derivs(a, e, eta)[0]
        rhs = numpy.empty((self._ntau, 3))
        for k in range(3):
            du = dsup[0, k] - dsup[1, k] * ct
            dft = (dWda[k] + dWdu * du) / (2.0 * pu) * dudtau + pu * dsup[1, k] * st
            mt, qt, _, qt0 = _cumulative(dft)
            _, _, da, de = self._aAIinv._orbital_param_chains(jr, L, dJr[k], dL[k])
            rhs[:, k] = mt * tau + qt - qt0 - FAa * da - FAe * de
        return _sine_galerkin(fAeta, rhs, self._nforDm)

    def _map_slopes_v(self, smp, L, c, zeta, dL, dwv):
        """The derivatives of the v map's coefficients with respect to
        (E, I_3, L_z), as _map_slopes_u for the u map: the auxiliary's polar
        cumulative action depends on its torus through (L, c = 1 -
        L_z^2/L^2), with closed-form partials (_aux_vcumulative)"""
        tau, v, pv, wv = smp["tau"], smp["v"], smp["pv"], smp["wv"]
        E, Lz = smp["E"], smp["Lz"]
        d2 = self._delta**2
        st, ct = numpy.sin(tau), numpy.cos(tau)
        dvdtau = wv * st
        dWdv = self._dWv(v, E, Lz)
        sn2 = numpy.sin(v) ** 2
        dWda = (2.0 * d2 * sn2, 2.0 * d2 * numpy.ones_like(v), -2.0 * Lz / sn2)
        _, FL, Fc = self._aux_vcumulative(L, c, zeta)
        fAzeta = self._aux_vflux(L, c, zeta)
        dc = 2.0 * Lz**2 / L**3 * dL - 2.0 * Lz / L**2 * numpy.array([0.0, 0.0, 1.0])
        rhs = numpy.empty((self._ntau, 3))
        for k in range(3):
            dv = -dwv[k] * ct
            dft = (dWda[k] + dWdv * dv) / (2.0 * pv) * dvdtau + pv * dwv[k] * st
            mt, qt, _, qt0 = _cumulative(dft)
            rhs[:, k] = mt * tau + qt - qt0 - FL * dL[k] - Fc * dc[k]
        return _sine_galerkin(fAzeta, rhs, self._nforDmv)

    # ---------- the per-torus tables, computed (never fitted)
    def _node_tables(self, ii):
        """One torus (the ii-th sampled): its actions and its period
        matrix by regular quadrature in the anomalies (the derivatives of
        the actions with respect to the integrals, whose inverse gives the
        frequencies), the turning points with their closed-form
        derivatives, the two maps' coefficients with their slopes in
        (E, I_3, L_z) from the torus alone, and the variation of the
        auxiliary's actions along the lifted librations (the maps'
        truncation, which should be at round-off). A degenerate libration
        contributes its harmonic limit."""
        smp = self._samples[ii]
        tau, E, Lz, I3 = smp["tau"], smp["E"], smp["Lz"], smp["I3"]
        uc, wu, wv, ush = smp["uc"], smp["wu"], smp["wv"], smp["ush"]
        d2 = self._delta**2
        st = numpy.sin(tau)
        udeg, vdeg = smp["u"] is None, smp["v"] is None
        jr, jz = smp["jr"], smp["jz"]
        # the period matrix dJ/d(E, I_3, L_z): (dq/dtau)/p is periodic and
        # finite (dq/dtau and p vanish together at the turning points), so
        # the trapezoid rule on the periodic grid is spectrally accurate; a
        # degenerate libration has the harmonic limit, J = W(q_0) / (2 sqrt
        # beta) with beta = -W''(q_0)/2 at the degenerate point
        M = numpy.empty((2, 3))
        dsup = numpy.full((3, 3), numpy.nan)
        if udeg:
            sqb = numpy.sqrt(-0.5 * float(self._d2Wu(ush, E, Lz)))
            sh2 = numpy.sinh(ush) ** 2
            M[0] = numpy.array([d2 * sh2, -d2, -Lz / sh2]) / sqb
            dE, dLz = self._dushell(ush, E, Lz)
            dsup[0] = [dE, 0.0, dLz]
        else:
            u, pu = smp["u"], smp["pu"]
            ratio = wu * st / pu
            sh2 = numpy.sinh(u) ** 2
            M[0] = [
                numpy.mean(d2 * sh2 * ratio),
                -d2 * numpy.mean(ratio),
                -Lz * numpy.mean(ratio / sh2),
            ]
            dinner = self._tp_derivs_u(uc - wu, E, Lz)
            douter = self._tp_derivs_u(uc + wu, E, Lz)
            dsup[0], dsup[1] = 0.5 * (dinner + douter), 0.5 * (douter - dinner)
        if vdeg:
            sqb = numpy.sqrt(-0.5 * float(self._d2Wv(0.5 * numpy.pi, E, Lz)))
            M[1] = numpy.array([d2, d2, -Lz]) / sqb
        else:
            v, pv = smp["v"], smp["pv"]
            ratio = wv * st / pv
            sn2 = numpy.sin(v) ** 2
            M[1] = [
                numpy.mean(d2 * sn2 * ratio),
                d2 * numpy.mean(ratio),
                -Lz * numpy.mean(ratio / sn2),
            ]
            dsup[2] = -self._tp_derivs_v(0.5 * numpy.pi - wv, E, Lz)
        # d(E, I_3, L_z)/d(J_R, J_z, J_phi), from J_phi = L_z
        N = numpy.linalg.inv(numpy.vstack((M, [0.0, 0.0, 1.0])))
        L = jz + numpy.fabs(Lz)
        c = jz * (jz + 2.0 * numpy.fabs(Lz)) / L**2
        dJr, dL = M[0], M[1] + numpy.array([0.0, 0.0, numpy.sign(Lz)])
        lift_u, lift_v = self._lifts[ii]
        node = {
            "jr": jr,
            "jz": jz,
            "M": M,
            "N": N,
            "uc": uc,
            "wu": wu,
            "wv": wv,
            "ush": ush,
            "dsup": dsup,
            "Dmu": numpy.zeros(self._npt),
            "Dmv": numpy.zeros(self._nptv),
            "dDmu": None,
            "dDmv": None,
            "perr_u": 0.0,
            "perr_v": 0.0,
        }
        if not udeg:
            # the truncated map's lift is not exactly the equal-action
            # torus: the auxiliary's radial action varies along it by the
            # truncation residual
            rA, pA = lift_u["rA"], lift_u["pA"]
            EA = 0.5 * (pA**2 + L**2 / rA**2) - self._GM / (
                self._b + numpy.sqrt(self._b**2 + rA**2)
            )
            node["perr_u"] = float(
                numpy.amax(numpy.fabs(self._aAIinv._torus_Jr(EA, L) - jr)) / jr
            )
            node["Dmu"] = lift_u["Dm"]
            node["dDmu"] = self._map_slopes_u(
                smp, jr, L, lift_u["a"], lift_u["e"], lift_u["eta"], dJr, dL, dsup
            )
        if not vdeg:
            # likewise the auxiliary's polar action along the lifted v
            # libration, through its total angular momentum
            LA = numpy.sqrt(lift_v["pA"] ** 2 + Lz**2 / lift_v["sinth"] ** 2)
            node["perr_v"] = float(
                numpy.amax(numpy.fabs(LA - numpy.fabs(Lz) - jz)) / jz
            )
            node["Dmv"] = lift_v["Dm"]
            node["dDmv"] = self._map_slopes_v(smp, L, c, lift_v["zeta"], dL, dsup[2])
        return node

    def _second_order_tables(self, ii, node):
        """The second-order tables of the ii-th explicit torus, from the
        torus alone, which make its one-node family quadratic: over the
        six pairs of (E, I_3, L_z), the second derivatives of the u
        libration's midpoint and half-width and of the v half-width (the
        level-set rule differentiated once more), of the two actions (the
        period matrix's derivatives, from the same regular quadratures as
        the matrix itself), and of the two maps' coefficients (the second
        variation of each matching condition at fixed anomaly, solved by
        the same Galerkin projection as the slopes; the u map's auxiliary
        chains run through (a, e)(J_R, L) and the v map's through (L, c),
        with L = J_z + |L_z| itself varying to second order through the
        period matrix). Every target-side integrand is regular at the
        turning points: the numerators N_a = S_a + S_q dq/da|_tau and N_ab
        vanish there to second order in tau while p vanishes to first.
        Returns None for a torus with a degenerate libration, whose tables
        stay zero. The chain to the actions is _quadratic_node."""
        smp = self._samples[ii]
        if smp["u"] is None or smp["v"] is None:
            return None
        lift_u, lift_v = self._lifts[ii]
        tau, E, Lz = smp["tau"], smp["E"], smp["Lz"]
        jr, jz, M, dsup = smp["jr"], smp["jz"], node["M"], node["dsup"]
        L = jz + numpy.fabs(Lz)
        c = jz * (jz + 2.0 * numpy.fabs(Lz)) / L**2
        d2 = self._delta**2
        st, ct = numpy.sin(tau), numpy.cos(tau)
        pairs = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))

        def tp2(S_q, S_qq, S_qa, S_LL, q1):
            # a turning point's second derivatives by the level-set rule
            # differentiated once more, S_q q_ab = -[S_qq q_a q_b + S_qa q_b
            # + S_qb q_a + S_ab], with S_ab = S_LL for (L_z, L_z) and zero
            # for every other pair
            return numpy.array(
                [
                    -(
                        S_qq * q1[i] * q1[j]
                        + S_qa[i] * q1[j]
                        + S_qa[j] * q1[i]
                        + (S_LL if i == j == 2 else 0.0)
                    )
                    / S_q
                    for i, j in pairs
                ]
            )

        def libration(q, p, w, q1, w1, q2, w2, S_q, S_qq, S_a, S_qa, S_LL):
            # at fixed anomaly: the second derivatives of the action (the
            # period-matrix row) and of the cumulative action, from the
            # coordinate's derivatives q1[a] = dq/da|_tau and q2[ab], the
            # half-width's w1, w2, and the momentum function's partials
            qtau = w * st
            N1 = [S_a[i] + S_q * q1[i] for i in range(3)]
            M2, A2 = numpy.empty(6), numpy.empty((self._ntau, 6))
            for kk, (i, j) in enumerate(pairs):
                S_ab = S_LL if i == j == 2 else 0.0 * q
                N2 = (
                    S_ab
                    + S_qa[i] * q1[j]
                    + S_qa[j] * q1[i]
                    + S_qq * q1[i] * q1[j]
                    + S_q * q2[kk]
                )
                M2[kk] = numpy.mean(
                    (S_ab + S_qa[i] * q1[j]) * qtau / (2.0 * p)
                    + S_a[i] * (w1[j] * st / (2.0 * p) - qtau * N1[j] / (4.0 * p**3))
                )
                mt, qt, _, qt0 = _cumulative(
                    (N2 * qtau + N1[i] * w1[j] * st + N1[j] * w1[i] * st) / (2.0 * p)
                    - N1[i] * N1[j] * qtau / (4.0 * p**3)
                    + p * w2[kk] * st
                )
                A2[:, kk] = mt * tau + qt - qt0
            return M2, A2

        # the u libration: both turning points, then the midpoint and the
        # half-width
        u, pu, wu, uc = smp["u"], smp["pu"], smp["wu"], smp["uc"]
        d2tp = []
        for ut, dut in ((uc - wu, dsup[0] - dsup[1]), (uc + wu, dsup[0] + dsup[1])):
            sh2 = numpy.sinh(ut) ** 2
            d2tp.append(
                tp2(
                    float(self._dWu(ut, E, Lz)),
                    float(self._d2Wu(ut, E, Lz)),
                    (
                        2.0 * d2 * numpy.sinh(2.0 * ut),
                        0.0,
                        4.0 * Lz * numpy.cosh(ut) / numpy.sinh(ut) ** 3,
                    ),
                    -2.0 / sh2,
                    dut,
                )
            )
        uc2, wu2 = 0.5 * (d2tp[0] + d2tp[1]), 0.5 * (d2tp[1] - d2tp[0])
        sh2 = numpy.sinh(u) ** 2
        M2u, A2u = libration(
            u,
            pu,
            wu,
            [dsup[0, k] - dsup[1, k] * ct for k in range(3)],
            dsup[1],
            [uc2[kk] - wu2[kk] * ct for kk in range(6)],
            wu2,
            self._dWu(u, E, Lz),
            self._d2Wu(u, E, Lz),
            (2.0 * d2 * sh2, -2.0 * d2 + 0.0 * u, -2.0 * Lz / sh2),
            (
                2.0 * d2 * numpy.sinh(2.0 * u),
                0.0 * u,
                4.0 * Lz * numpy.cosh(u) / numpy.sinh(u) ** 3,
            ),
            -2.0 / sh2,
        )
        # the v libration: its turning point, then the half-width
        v, pv, wv = smp["v"], smp["pv"], smp["wv"]
        vt = 0.5 * numpy.pi - wv
        sn2 = numpy.sin(vt) ** 2
        wv2 = -tp2(
            float(self._dWv(vt, E, Lz)),
            float(self._d2Wv(vt, E, Lz)),
            (
                2.0 * d2 * numpy.sin(2.0 * vt),
                0.0,
                4.0 * Lz * numpy.cos(vt) / numpy.sin(vt) ** 3,
            ),
            -2.0 / sn2,
            -dsup[2],
        )
        sn2 = numpy.sin(v) ** 2
        M2v, A2v = libration(
            v,
            pv,
            wv,
            [-dsup[2, k] * ct for k in range(3)],
            dsup[2],
            [-wv2[kk] * ct for kk in range(6)],
            wv2,
            self._dWv(v, E, Lz),
            self._d2Wv(v, E, Lz),
            (2.0 * d2 * sn2, 2.0 * d2 + 0.0 * v, -2.0 * Lz / sn2),
            (
                2.0 * d2 * numpy.sin(2.0 * v),
                0.0 * v,
                4.0 * Lz * numpy.cos(v) / numpy.sin(v) ** 3,
            ),
            -2.0 / sn2,
        )
        # the auxiliary torus's parameters along the integrals: J_R and L =
        # J_z + |L_z| to second order from the period matrix and its
        # derivatives, (a, e) through their closed-form partials in (J_R, L)
        J1, J2 = M[0], M2u
        L1 = M[1] + numpy.array([0.0, 0.0, numpy.sign(Lz)])
        L2 = M2v

        def chain2(d):
            dJ, dL, dJJ, dJL, dLL = d
            first = [dJ * J1[i] + dL * L1[i] for i in range(3)]
            second = [
                dJJ * J1[i] * J1[j]
                + dJL * (J1[i] * L1[j] + J1[j] * L1[i])
                + dLL * L1[i] * L1[j]
                + dJ * J2[kk]
                + dL * L2[kk]
                for kk, (i, j) in enumerate(pairs)
            ]
            return first, second

        _, _, da, de = self._aAIinv._orbital_params2(jr, L)
        a1, a2 = chain2(da)
        e1, e2 = chain2(de)
        a, e, eta = lift_u["a"], lift_u["e"], lift_u["eta"]
        _, AA_a, AA_e, AA_aa, AA_ae, AA_ee = self._aAIinv._cumulative2(a, e, eta)
        fA, fA_eta = self._aAIinv._flux_eta(a, e, eta)
        _, fA_a, fA_e = self._aAIinv._flux_derivs(a, e, eta)
        eta1 = [
            _sine_series(self._ntau, node["dDmu"][:, k], self._nforDm)[0]
            for k in range(3)
        ]
        rhs = numpy.empty((self._ntau, 6))
        for kk, (i, j) in enumerate(pairs):
            rhs[:, kk] = (
                A2u[:, kk]
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
        D2u = _sine_galerkin(fA, rhs, self._nforDm)
        # the v map: the auxiliary's polar libration through (L, c), with
        # c = 1 - L_z^2/L^2 to second order
        dLz = numpy.array([0.0, 0.0, 1.0])
        c1 = 2.0 * Lz**2 / L**3 * L1 - 2.0 * Lz / L**2 * dLz
        c2 = numpy.array(
            [
                2.0 * Lz**2 / L**3 * L2[kk]
                - 6.0 * Lz**2 / L**4 * L1[i] * L1[j]
                + 4.0 * Lz / L**3 * (dLz[i] * L1[j] + dLz[j] * L1[i])
                - 2.0 / L**2 * dLz[i] * dLz[j]
                for kk, (i, j) in enumerate(pairs)
            ]
        )
        zeta = lift_v["zeta"]
        fv, fv_z, fv_L, fv_c = self._aux_vflux_derivs(L, c, zeta)
        _, Av_L, Av_c, Av_LL, Av_Lc, Av_cc = self._aux_vcumulative2(L, c, zeta)
        zeta1 = [
            _sine_series(self._ntau, node["dDmv"][:, k], self._nforDmv)[0]
            for k in range(3)
        ]
        for kk, (i, j) in enumerate(pairs):
            rhs[:, kk] = (
                A2v[:, kk]
                - (
                    fv_z * zeta1[i] * zeta1[j]
                    + fv_L * (L1[j] * zeta1[i] + L1[i] * zeta1[j])
                    + fv_c * (c1[j] * zeta1[i] + c1[i] * zeta1[j])
                )
                - (
                    Av_LL * L1[i] * L1[j]
                    + Av_Lc * (L1[i] * c1[j] + L1[j] * c1[i])
                    + Av_cc * c1[i] * c1[j]
                )
                - Av_L * L2[kk]
                - Av_c * c2[kk]
            )
        D2v = _sine_galerkin(fv, rhs, self._nforDmv)
        return {
            "d2sup": numpy.vstack((uc2, wu2, wv2)),
            "d2M": numpy.vstack((M2u, M2v)),
            "d2Dmu": D2u,
            "d2Dmv": D2v,
        }

    def _quadratic_node(self, ii):
        """The ii-th explicit torus's one-node family to second order in
        the actions (J_R, J_z, J_phi): with N = d(E, I_3, L_z)/dJ the
        inverse period matrix, the integrals' second derivatives in the
        actions follow from the inverse-function rule, alpha_{gamma,ij} =
        -sum_k N_{gamma k} [N^T J_{k,..} N]_{ij} over the two actions'
        second derivatives (the E row is the frequencies' derivatives), and
        every table X(E, I_3, L_z) chains as X_{J_i} = sum_a X_a N_{ai} and
        X_{J_i J_j} = [N^T X_{..} N]_{ij} + sum_a X_a alpha_{a,ij}; a
        degenerate libration's half-width and map have no derivatives, as
        in _explicit_tables"""
        N = self._Ns[ii]
        idx = numpy.array([[0, 1, 2], [1, 3, 4], [2, 4, 5]])

        def sym(x6):
            # the six pairs as a symmetric matrix (along the last axis)
            return numpy.asarray(x6)[..., idx]

        JJ = numpy.zeros((3, 3, 3))
        JJ[:2] = sym(self._d2Ms[ii])
        d2alpha = -numpy.einsum("gk,ai,kab,bj->gij", N, N, JJ, N)

        def chain(x1, x6, zero=False):
            x1, x6 = numpy.asarray(x1), numpy.asarray(x6)
            if zero:
                return numpy.zeros(x1.shape), numpy.zeros(x1.shape + (3,))
            return x1 @ N, numpy.einsum(
                "ai,...ab,bj->...ij", N, sym(x6), N
            ) + numpy.einsum("...a,aij->...ij", x1, d2alpha)

        udeg, vdeg = self._wus[ii] == 0.0, self._wvs[ii] == 0.0
        dsup, d2sup = self._dsups[ii], self._d2sups[ii]
        out = {
            "jr": self._jrs[ii],
            "jz": self._jzs[ii],
            "Lz": self._Lzs[ii],
            "Om": N[0],
            "d2E": d2alpha[0],
            "d2alpha": d2alpha,
            "uc": self._ucs[ii],
            "wu": self._wus[ii],
            "wv": self._wvs[ii],
            "Dmu": self._Dmus[ii],
            "Dmv": self._Dmvs[ii],
        }
        out["duc"], out["d2uc"] = chain(dsup[0], d2sup[0])
        out["dwu"], out["d2wu"] = chain(dsup[1], d2sup[1], zero=udeg)
        out["dwv"], out["d2wv"] = chain(dsup[2], d2sup[2], zero=vdeg)
        out["dDmu"], out["d2Dmu"] = chain(self._dDmus[ii], self._d2Dmus[ii])
        out["dDmv"], out["d2Dmv"] = chain(self._dDmvs[ii], self._d2Dmvs[ii])
        return out

    def _warn_unresolved(self, perr, Es, Lzs, I3s, which):
        """Warn when a truncated map leaves the auxiliary's action varying
        along a lifted libration by more than round-off allows"""
        bad = numpy.asarray(perr) > 1e-6
        if numpy.any(bad):
            warnings.warn(
                "The momentum-matched {} anomaly map is not converged for the (E, L_z, I_3) tori: {} (maximum relative variation of the auxiliary action along a lifted libration {:.1e}); increase mm_npt and, with it, mm_nta".format(
                    which,
                    ", ".join(
                        f"({E:g}, {Lz:g}, {I3:g})"
                        for E, Lz, I3 in zip(
                            numpy.asarray(Es)[bad],
                            numpy.asarray(Lzs)[bad],
                            numpy.asarray(I3s)[bad],
                        )
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
        self._jzs = numpy.empty(ntori)
        self._Ns = numpy.empty((ntori, 3, 3))
        self._ucs = numpy.empty(ntori)
        self._wus = numpy.empty(ntori)
        self._wvs = numpy.empty(ntori)
        self._dsups = numpy.empty((ntori, 3, 3))
        self._Dmus = numpy.empty((ntori, self._npt))
        self._Dmvs = numpy.empty((ntori, self._nptv))
        self._dDmus = numpy.zeros((ntori, self._npt, 3))
        self._dDmvs = numpy.zeros((ntori, self._nptv, 3))
        self._d2sups = numpy.zeros((ntori, 3, 6))
        self._d2Ms = numpy.zeros((ntori, 2, 6))
        self._d2Dmus = numpy.zeros((ntori, self._npt, 6))
        self._d2Dmvs = numpy.zeros((ntori, self._nptv, 6))
        perr_u, perr_v = numpy.empty(ntori), numpy.empty(ntori)
        for ii in _progress(range(ntori), "node tables", self._progressbar):
            with self._in_chart(self._Es[ii], self._Lzs[ii]):
                node = self._node_tables(ii)
                # the second-order tables make the one-node family
                # quadratic: an explicit torus then carries the true
                # derivatives of the map in the actions at its node
                so = self._second_order_tables(ii, node)
            if so is not None:
                self._d2sups[ii], self._d2Ms[ii] = so["d2sup"], so["d2M"]
                self._d2Dmus[ii], self._d2Dmvs[ii] = so["d2Dmu"], so["d2Dmv"]
            self._jrs[ii], self._jzs[ii] = node["jr"], node["jz"]
            self._Ns[ii] = node["N"]
            self._ucs[ii], self._wus[ii], self._wvs[ii] = (
                node["uc"],
                node["wu"],
                node["wv"],
            )
            self._dsups[ii] = node["dsup"]
            self._Dmus[ii], self._Dmvs[ii] = node["Dmu"], node["Dmv"]
            # a degenerate libration has no map slopes to store: it
            # contributes nothing to the evaluation
            if node["dDmu"] is not None:
                self._dDmus[ii] = node["dDmu"]
            if node["dDmv"] is not None:
                self._dDmvs[ii] = node["dDmv"]
            perr_u[ii], perr_v[ii] = node["perr_u"], node["perr_v"]
        self._warn_unresolved(perr_u, self._Es, self._Lzs, self._I3s, "u")
        self._warn_unresolved(perr_v, self._Es, self._Lzs, self._I3s, "v")
        return None

    # ---------- the (L_z, E, I_3) interpolation grid
    def _setup_grid(self, Rmin, Rmax, Rinf, nE, nLz, nI3, Lzspacing="linear"):
        """Rectangular grid in (w_E, w_I, L_z): L_z between the circular
        angular momenta of Rmin and Rmax; the energies uniform, in x =
        w_E^2 with w_E uniform in [0, 1], in the logarithm of the apocentre
        of the planar orbit without angular momentum, R_0(E) with
        Phi(R_0, 0) = E, between the circular orbit's energy E_c(L_z) and
        E_max(L_z), the energy of the planar orbit with apocentre Rinf
        (_E_of_xL); the circular orbits form the bottom face, and in the
        energy the spacing is quadratic there and compresses toward the
        top in a potential with an escape energy, where every apocentre
        diverges, so that the tables stay smooth in x up to the top row
        (for a logarithmic potential it is E = E_c + [E_max - E_c] x);
        I_3 = I_pl + (I_sh - I_pl) sin^2(pi w_I / 2)
        with w_I uniform in [0, 1] between the planar orbit (J_z = 0) and
        the shell orbit (J_R = 0), the two faces on which one libration
        degenerates -- quadratic spacing at every degenerate edge, where
        the actions behave as the square of the distance to it and the
        half-widths as the distance itself"""
        if nE < 4 or nLz < 4 or nI3 < 4:
            raise ValueError("setup_interp=True requires nE, nLz, and nI3 >= 4")
        if Rinf <= Rmax:
            raise ValueError(
                "Rinf has to exceed Rmax: it is the apocentre in the plane of "
                "the grid's most energetic orbits, beyond every circular orbit "
                "of the grid"
            )
        wrap = self._staeckelwrap
        Lmin = Rmin * vcirc(wrap, Rmin, use_physical=False)
        Lmax = Rmax * vcirc(wrap, Rmax, use_physical=False)
        self._Rinf = Rinf
        self._Phiinf = evaluatePotentials(wrap, Rinf, 0.0, use_physical=False)
        self._R0_cache = {}
        # the grid's box in (L_z, w_E, w_I): the default spans, or the
        # target's padded range
        box = ((Lmin, Lmax), (0.0, 1.0), (0.0, 1.0))
        if self._target is not None:
            box = self._target_box(Lmin, Lmax)
        self._box = box
        self._Lzgrid = _spaced_grid(box[0][0], box[0][1], nLz, Lzspacing)
        # the energy variable's bottom apocentre is that of the planar orbit
        # without angular momentum at the circular orbit's energy, which has
        # to lie inside Rinf for every L_z of the grid: Phi(Rinf, 0) must
        # exceed the circular energy of the grid's largest |L_z|
        if self._Phiinf <= self._circular(self._Lzgrid[-1])["Ec"]:
            raise ValueError(
                "Rinf is too small: the planar orbit without angular momentum at "
                "the circular orbit's energy of the grid's largest |L_z| reaches "
                "beyond Rinf; increase Rinf"
            )
        self._wEs = numpy.linspace(box[1][0], box[1][1], nE)
        self._wIs = numpy.linspace(box[2][0], box[2][1], nI3)
        self._ys = numpy.sin(0.5 * numpy.pi * self._wIs) ** 2
        self._E_tab = numpy.empty((nE, nLz))
        self._ush_tab = numpy.empty((nE, nLz))
        self._Ipl_tab = numpy.empty((nE, nLz))
        self._Ish_tab = numpy.empty((nE, nLz))
        I3s = numpy.empty((nE, nI3, nLz))
        for iL, Lz in enumerate(self._Lzgrid):
            circ = self._circular(Lz)
            Ec = circ["Ec"]
            for iE, wE in enumerate(self._wEs):
                E = self._E_of_xL(wE**2, Lz)[0]
                if wE == 0.0:
                    ush = circ["uc"]
                    Ipl = Ish = self._I3_planar(Ec, Lz)
                else:
                    with self._in_chart(E, Lz):
                        ush = self._ushell(E, Lz)
                        Ipl, Ish = self._I3_planar(E, Lz), self._I3_shell(E, Lz, ush)
                self._E_tab[iE, iL] = E
                self._ush_tab[iE, iL] = ush
                self._Ipl_tab[iE, iL], self._Ish_tab[iE, iL] = Ipl, Ish
                I3s[iE, :, iL] = Ipl + self._ys * (Ish - Ipl)
        # the tori in (iE, iI, iL) order
        self._Es = numpy.repeat(self._E_tab[:, None, :], nI3, axis=1).flatten()
        self._Lzs = numpy.tile(self._Lzgrid, nE * nI3)
        self._I3s = I3s.flatten()
        return None

    def _target_box(self, Lmin, Lmax):
        """The padded (L_z, w_E, w_I) box that localizes the grid on the
        target's tori. Every target point is labelled through the relations
        the node lattice uses (in its own chart for an adaptive reference
        curve); the box is the labels' range, padded by target_pad times
        the spread on both ends of each axis and at least target_minwidth
        of each axis's default span wide. An end that reaches a degenerate
        face (the circular, planar or shell orbits) is the face itself,
        because the grid must reach a face exactly or not at all, whereas
        the L_z ends are anchors that the box may exceed."""
        R, vR, vT, z, vz = self._target
        Lzs = R * numpy.fabs(vT)
        wEs, wIs = numpy.empty(len(Lzs)), numpy.empty(len(Lzs))
        for i, (Rp, vRp, vTp, zp, vzp, Lz) in enumerate(zip(R, vR, vT, z, vz, Lzs)):
            kin = 0.5 * (vRp**2 + vTp**2 + vzp**2)
            # the point's chart, placed with the raw potential's energy: the
            # reference curve varies smoothly, so the model error in the
            # energy is immaterial for the box
            Eraw = kin + evaluatePotentials(self._chart_pot, Rp, zp, use_physical=False)
            with self._in_chart(Eraw, Lz):
                E = kin + evaluatePotentials(
                    self._staeckelwrap, Rp, zp, use_physical=False
                )
                Ec, Emax = self._circular(Lz)["Ec"], self._Emax(Lz)
                if E > Emax:
                    raise ValueError(
                        "a target point's energy lies above the energies the grid "
                        "can cover; increase Rinf"
                    )
                if E <= Ec + 1e-12 * (1.0 + numpy.fabs(Ec)):
                    # the circular orbit, where every I_3 label coincides and
                    # the w_I direction is free
                    wEs[i], wIs[i] = 0.0, 0.5
                    continue
                R0c, _, R0m, _ = self._R0_edges(Lz)
                x = numpy.log(self._R0(E, R0c, R0m) / R0c) / numpy.log(R0m / R0c)
                wEs[i] = numpy.sqrt(min(max(x, 0.0), 1.0))
                # the point's third integral, from the u equation of the
                # separated Hamilton-Jacobi relation
                u, v = coords.Rz_to_uv(Rp, zp, delta=self._delta)
                sh = numpy.sinh(u)
                pu = self._delta * (
                    vRp * numpy.cosh(u) * numpy.sin(v) + vzp * sh * numpy.cos(v)
                )
                I3 = (
                    E * sh**2
                    - float(self._U(u))
                    - (pu**2 + Lz**2 / sh**2) / (2.0 * self._delta**2)
                )
                Ipl = self._I3_planar(E, Lz)
                Ish = self._I3_shell(E, Lz, self._ushell(E, Lz))
                y = min(max((I3 - Ipl) / (Ish - Ipl), 0.0), 1.0)
                wIs[i] = 2.0 / numpy.pi * numpy.arcsin(numpy.sqrt(y))

        def padded(vals, lo0, hi0, snap):
            lo, hi = float(numpy.min(vals)), float(numpy.max(vals))
            lo, hi = (
                lo - self._target_pad * (hi - lo),
                hi + self._target_pad * (hi - lo),
            )
            minw = self._target_minwidth * (hi0 - lo0)
            if hi - lo < minw:
                mid = 0.5 * (lo + hi)
                lo, hi = mid - 0.5 * minw, mid + 0.5 * minw
            if snap:
                # an end that reaches a face is the face
                return (lo0 if lo <= lo0 else lo, hi0 if hi >= hi0 else hi)
            # the L_z ends are anchors, not faces: the box may exceed them,
            # and only the lower end is guarded, against L_z <= 0
            return (max(lo, 0.5 * float(numpy.min(vals))), hi)

        Lzbox = padded(Lzs, Lmin, Lmax, False)
        if rl(self._staeckelwrap, Lzbox[1], use_physical=False) >= self._Rinf:
            raise ValueError(
                "the target's padded L_z range reaches circular orbits beyond "
                "Rinf, which must lie beyond every circular orbit of the grid; "
                "increase Rinf or reduce target_pad"
            )
        return (Lzbox, padded(wEs, 0.0, 1.0, True), padded(wIs, 0.0, 1.0, True))

    def _check_circular(self):
        """Raise for the circular orbit of a family whose grid does not reach
        the circular face"""
        if self._wEs[0] != 0.0:
            raise ValueError(
                "the circular orbit lies outside the family's grid, whose box does "
                "not reach the circular face"
            )
        return None

    def _Emax(self, Lz):
        """The grid's top energy at |L_z|: that of the planar orbit with
        apocentre Rinf"""
        return self._Phiinf + Lz**2 / (2.0 * self._Rinf**2)

    def _R0(self, E, Rlo, Rhi):
        """The apocentre of the planar orbit without angular momentum at
        energy E, Phi(R_0, 0) = E, bracketed by [Rlo, Rhi]"""
        return brentq(
            lambda R: (
                evaluatePotentials(self._staeckelwrap, R, 0.0, use_physical=False) - E
            ),
            Rlo,
            Rhi,
            xtol=1e-14,
        )

    def _RdPhi(self, R):
        """R dPhi/dR in the plane, the derivative of the potential with
        respect to ln R"""
        return -R * evaluateRforces(self._staeckelwrap, R, 0.0, use_physical=False)

    def _R0_edges(self, Lz):
        """The apocentres R_0 of the planar orbit without angular momentum
        at the grid's bottom and top energies at |L_z|, with R dPhi/dR at
        each (the level-set rule gives d ln R_0/dL_z = (dE/dL_z)/[R_0
        Phi'(R_0)]); cached. The bottom one lies above R_c, since E_c
        exceeds Phi(R_c, 0) by the centrifugal term, and the top one just
        beyond Rinf, by L_z^2 / (2 Rinf^2) in energy"""
        if Lz not in self._R0_cache:
            circ = self._circular(Lz)
            R0c = self._R0(circ["Ec"], circ["Rc"], self._Rinf)
            R0m = self._R0(self._Emax(Lz), self._Rinf, 10.0 * self._Rinf)
            self._R0_cache[Lz] = (R0c, self._RdPhi(R0c), R0m, self._RdPhi(R0m))
        return self._R0_cache[Lz]

    def _E_of_xL(self, x, Lz):
        """The grid's energy variable, analytic: E = Phi(R, 0) with ln R =
        (1 - x) ln R_0(E_c) + x ln R_0(E_max), uniform in the logarithm of
        the apocentre of the planar orbit without angular momentum; returns
        E, dE/dx at fixed L_z, and dE/dL_z at fixed x, the latter through
        the two edges' apocentres by the level-set rule, formed so that it
        is exactly dE_c/dL_z at the circular face and dE_max/dL_z at the
        top"""
        R0c, g0c, R0m, g0m = self._R0_edges(Lz)
        R = R0c * (R0m / R0c) ** x
        g = self._RdPhi(R)
        Omc = self._circular(Lz)["Omc"]
        return (
            evaluatePotentials(self._staeckelwrap, R, 0.0, use_physical=False),
            g * numpy.log(R0m / R0c),
            (1.0 - x) * Omc * (g / g0c) + x * (Lz / self._Rinf**2) * (g / g0m),
        )

    def _E_of_xL_array(self, x, Lz):
        """_E_of_xL for arrays of points, each with its own |L_z|: the
        circular radius and the two apocentre edges from a vectorized
        root-find on the potential's midplane, which the Staeckel wrapper
        reproduces exactly, bracketed by the grid's extreme circular radii
        and Rinf"""
        wrap = self._chart_pot

        def phi(R):
            return evaluatePotentials(wrap, R, 0.0, use_physical=False)

        def dphi(R):
            return -evaluateRforces(wrap, R, 0.0, use_physical=False)

        def d2phi(R):
            return evaluateR2derivs(wrap, R, 0.0, use_physical=False)

        Rlo = 0.9 * self._circular(self._Lzgrid[0])["Rc"] + 0.0 * Lz
        Rhi = 1.1 * self._circular(self._Lzgrid[-1])["Rc"] + 0.0 * Lz
        Rc = _solve_increasing(
            lambda R: R**3 * dphi(R) - Lz**2,
            lambda R: R**2 * (3.0 * dphi(R) + R * d2phi(R)),
            Rlo,
            Rhi,
        )
        Ec = phi(Rc) + Lz**2 / (2.0 * Rc**2)
        Emax = self._Phiinf + Lz**2 / (2.0 * self._Rinf**2)
        Rinf = self._Rinf + 0.0 * Lz
        R0c = _solve_increasing(lambda R: phi(R) - Ec, dphi, Rc, Rinf)
        R0m = _solve_increasing(lambda R: phi(R) - Emax, dphi, Rinf, 10.0 * Rinf)
        g0c, g0m = R0c * dphi(R0c), R0m * dphi(R0m)
        R = R0c * (R0m / R0c) ** x
        g = R * dphi(R)
        return (
            phi(R),
            g * numpy.log(R0m / R0c),
            (1.0 - x) * (Lz / Rc**2) * (g / g0c) + x * (Lz / self._Rinf**2) * (g / g0m),
        )

    def _setup_tori_interp(self):
        """The family's tables and their exact first partials at every
        node, in the variables in which each quantity is smooth up to the
        degenerate edges: the actions and the u midpoint in (x = w_E^2,
        y = sin^2(pi w_I/2), L_z), in which they are smooth (the actions
        vanish linearly at the edges); the u half-width and the u map in
        (w_E, s_u = cos(pi w_I/2), L_z), the v half-width and the v map in
        (w_E, s_v = sin(pi w_I/2), L_z), in which they vanish linearly at
        their edges. The partials come from the node's period matrix, the
        level-set rule, and the map slopes, chained through the grid's
        analytic (E, I_3)(w_E, w_I, L_z) relation; on the degenerate faces
        they are the harmonic limits."""
        nE, nI, nL = len(self._wEs), len(self._wIs), len(self._Lzgrid)
        npt, d2 = self._npt, self._delta**2
        A_tab, A_dx, A_dy, A_dL = (numpy.zeros((nE, nI, nL, 3)) for _ in range(4))
        U_tab, U_dwE, U_dsu, U_dL = (
            numpy.zeros((nE, nI, nL, 1 + npt)) for _ in range(4)
        )
        V_tab, V_dwE, V_dsv, V_dL = (
            numpy.zeros((nE, nI, nL, 1 + self._nptv)) for _ in range(4)
        )
        perr_u, perr_v = numpy.zeros((nE, nI, nL)), numpy.zeros((nE, nI, nL))
        for ii in _progress(range(nE * nI * nL), "node tables", self._progressbar):
            iE, rest = divmod(ii, nI * nL)
            iI, iL = divmod(rest, nL)
            wE, wI, Lz = self._wEs[iE], self._wIs[iI], self._Lzgrid[iL]
            with self._in_chart(self._E_tab[iE, iL], Lz):
                y = self._ys[iI]
                su, sv = numpy.cos(0.5 * numpy.pi * wI), numpy.sin(0.5 * numpy.pi * wI)
                if True:
                    node = self._node_tables(ii)
                    perr_u[iE, iI, iL], perr_v[iE, iI, iL] = (
                        node["perr_u"],
                        node["perr_v"],
                    )
                    A_tab[iE, iI, iL] = [node["jr"], node["jz"], node["uc"]]
                    U_tab[iE, iI, iL, 0], U_tab[iE, iI, iL, 1:] = (
                        node["wu"],
                        node["Dmu"],
                    )
                    V_tab[iE, iI, iL, 0], V_tab[iE, iI, iL, 1:] = (
                        node["wv"],
                        node["Dmv"],
                    )
                    circ = self._circular(Lz)
                    Ec = circ["Ec"]
                    E, ush = self._E_tab[iE, iL], self._ush_tab[iE, iL]
                    Ipl, Ish = self._Ipl_tab[iE, iL], self._Ish_tab[iE, iL]
                    _, dE_dx, dE_dL = self._E_of_xL(wE**2, Lz)
                    if wE == 0.0:
                        # the circular face: both librations harmonic, with
                        # J = (E - E_c) times the fraction the third integral
                        # assigns to each, over its frequency; the
                        # half-widths grow linearly in w_E out of the edge
                        # and the maps vanish (their slopes come from the
                        # next two rows, below)
                        uc = circ["uc"]
                        sqbu = numpy.sqrt(-0.5 * float(self._d2Wu(uc, Ec, Lz)))
                        sqbv = numpy.sqrt(
                            -0.5 * float(self._d2Wv(0.5 * numpy.pi, Ec, Lz))
                        )
                        ch2 = numpy.cosh(uc) ** 2
                        A_dx[iE, iI, iL] = [
                            dE_dx * (1.0 - y) * d2 * ch2 / sqbu,
                            dE_dx * y * d2 * ch2 / sqbv,
                            0.0,
                        ]
                        # the circular radius moves with L_z at the rate the
                        # epicycle sets (L_z^2 = R_c^3 Phi'(R_c) gives dR_c/dL_z
                        # = 2 L_z / [R_c^3 kappa^2])
                        A_dL[iE, iI, iL, 2] = (
                            2.0
                            * Lz
                            / (circ["Rc"] ** 3 * circ["kappa"] ** 2)
                            / (self._delta * numpy.cosh(uc))
                        )
                        U_dwE[iE, iI, iL, 0] = (
                            numpy.sqrt(2.0 * d2 * ch2 * dE_dx) / sqbu * su
                        )
                        V_dwE[iE, iI, iL, 0] = (
                            numpy.sqrt(2.0 * d2 * ch2 * dE_dx) / sqbv * sv
                        )
                        continue
                    sh2 = numpy.sinh(ush) ** 2
                    dI_dE = y * (1.0 + sh2) - 1.0
                    dI_dy = Ish - Ipl
                    dI_dL = Lz / d2 * (1.0 - y * (1.0 + 1.0 / sh2))

                    def chain(q):
                        # (dq/dE, dq/dI_3, dq/dL_z) -> (dq/dx, dq/dy, dq/dL_z)
                        # at fixed grid coordinates
                        qE, qI, qL = q[..., 0], q[..., 1], q[..., 2]
                        qx = (qE + qI * dI_dE) * dE_dx
                        return qx, qI * dI_dy, qL + qI * dI_dL + qx / dE_dx * dE_dL

                    M, dsup = node["M"], node["dsup"]
                    udeg, vdeg = node["dDmu"] is None, node["dDmv"] is None
                    if udeg:
                        # the shell face: the u libration harmonic at the
                        # shell u, its action and half-width vanishing
                        # linearly in (1 - y) and in s_u
                        sqbu = numpy.sqrt(-0.5 * float(self._d2Wu(ush, E, Lz)))
                        A_dy[iE, iI, iL, 0] = -d2 * dI_dy / sqbu
                        U_dsu[iE, iI, iL, 0] = numpy.sqrt(2.0 * d2 * dI_dy) / sqbu
                    else:
                        (
                            A_dx[iE, iI, iL, 0],
                            A_dy[iE, iI, iL, 0],
                            A_dL[iE, iI, iL, 0],
                        ) = chain(M[0])
                        qx, qy, qL = chain(
                            numpy.vstack((dsup[1][None, :], node["dDmu"]))
                        )
                        U_dwE[iE, iI, iL] = 2.0 * wE * qx
                        U_dsu[iE, iI, iL] = -2.0 * su * qy
                        U_dL[iE, iI, iL] = qL
                    if vdeg:
                        # the planar face: the v libration harmonic about
                        # the midplane
                        sqbv = numpy.sqrt(
                            -0.5 * float(self._d2Wv(0.5 * numpy.pi, E, Lz))
                        )
                        A_dy[iE, iI, iL, 1] = d2 * dI_dy / sqbv
                        V_dsv[iE, iI, iL, 0] = numpy.sqrt(2.0 * d2 * dI_dy) / sqbv
                    else:
                        (
                            A_dx[iE, iI, iL, 1],
                            A_dy[iE, iI, iL, 1],
                            A_dL[iE, iI, iL, 1],
                        ) = chain(M[1])
                        qx, qy, qL = chain(
                            numpy.vstack((dsup[2][None, :], node["dDmv"]))
                        )
                        V_dwE[iE, iI, iL] = 2.0 * wE * qx
                        V_dsv[iE, iI, iL] = 2.0 * sv * qy
                        V_dL[iE, iI, iL] = qL
                    A_dx[iE, iI, iL, 2], A_dy[iE, iI, iL, 2], A_dL[iE, iI, iL, 2] = (
                        chain(dsup[0])
                    )
        # the maps' coefficients vanish at the circular edge and grow
        # linearly in w_E out of it (the first anharmonic correction of a
        # libration is linear in its amplitude); their slope there is that
        # of the quartic through zero and the next two rows, as the 1D
        # family does at its harmonic bottom; likewise the u map at the
        # shell face in s_u and the v map at the planar face in s_v. The u
        # libration's midpoint leaves the shell u quadratically in the
        # half-width (the cubic term of the momentum function skews the
        # libration), i.e. linearly in x at the circular face and in y at
        # the shell face, at rates the same quartics through the next two
        # rows give
        xs = self._wEs**2
        sus = numpy.cos(0.5 * numpy.pi * self._wIs)
        svs = numpy.sin(0.5 * numpy.pi * self._wIs)
        if self._wEs[0] == 0.0:
            A_dx[0, :, :, 2] = slope_at_zero(
                xs[1:3], A_tab[1:3, :, :, 2] - A_tab[0, :, :, 2], A_dx[1:3, :, :, 2]
            )
            U_dwE[0, :, :, 1:] = slope_at_zero(
                self._wEs[1:3], U_tab[1:3, :, :, 1:], U_dwE[1:3, :, :, 1:]
            )
            V_dwE[0, :, :, 1:] = slope_at_zero(
                self._wEs[1:3], V_tab[1:3, :, :, 1:], V_dwE[1:3, :, :, 1:]
            )
        if self._wIs[-1] == 1.0:
            A_dy[:, -1, :, 2] = slope_at_zero(
                self._ys[[-2, -3]] - 1.0,
                numpy.moveaxis(A_tab[:, [-2, -3], :, 2:3], 1, 0)[..., 0]
                - A_tab[:, -1, :, 2][None],
                numpy.moveaxis(A_dy[:, [-2, -3], :, 2:3], 1, 0)[..., 0],
            )
            U_dsu[:, -1, :, 1:] = slope_at_zero(
                sus[[-2, -3]],
                numpy.moveaxis(U_tab[:, [-2, -3], :, 1:], 1, 0),
                numpy.moveaxis(U_dsu[:, [-2, -3], :, 1:], 1, 0),
            )
        if self._wIs[0] == 0.0:
            V_dsv[:, 0, :, 1:] = slope_at_zero(
                svs[[1, 2]],
                numpy.moveaxis(V_tab[:, [1, 2], :, 1:], 1, 0),
                numpy.moveaxis(V_dsv[:, [1, 2], :, 1:], 1, 0),
            )
        self._perr_u_tab, self._perr_v_tab = perr_u, perr_v
        self._A_tab, self._A_dx, self._A_dy, self._A_dL = A_tab, A_dx, A_dy, A_dL
        self._U_tab, self._U_dwE, self._U_dsu, self._U_dL = U_tab, U_dwE, U_dsu, U_dL
        self._V_tab, self._V_dwE, self._V_dsv, self._V_dL = V_tab, V_dwE, V_dsv, V_dL
        self._warn_unresolved(perr_u.flatten(), self._Es, self._Lzs, self._I3s, "u")
        self._warn_unresolved(perr_v.flatten(), self._Es, self._Lzs, self._I3s, "v")
        self._rebuild_interp()
        return None

    def _rebuild_interp(self):
        """(Re)build the interpolants from the stored tables: Hermite
        interpolants, quintic along the energy variable and cubic along the
        other two, all with their first partials exact at the nodes"""
        wE, Lz = self._wEs, self._Lzgrid
        self._A_ip = HermiteFamily3D(
            wE**2, self._ys, Lz, self._A_tab, self._A_dx, self._A_dy, self._A_dL
        )
        # s_u = cos(pi w_I/2) decreases along the grid: the u tables are
        # reversed along w_I so that their second variable increases
        su = numpy.cos(0.5 * numpy.pi * self._wIs)[::-1]
        self._U_ip = HermiteFamily3D(
            wE,
            su,
            Lz,
            self._U_tab[:, ::-1],
            self._U_dwE[:, ::-1],
            self._U_dsu[:, ::-1],
            self._U_dL[:, ::-1],
        )
        sv = numpy.sin(0.5 * numpy.pi * self._wIs)
        self._V_ip = HermiteFamily3D(
            wE, sv, Lz, self._V_tab, self._V_dwE, self._V_dsv, self._V_dL
        )
        return None

    # ---------- evaluation: the manifest chain
    @property
    def _target_hint(self):
        """The tail of an out-of-range error for a family localized on a
        target, whose box is the target's padded range rather than the
        default spans"""
        if self._target is None:
            return ""
        return (
            " (the grid is localized on a target: increase target_pad, or build "
            "the family without target=)"
        )

    def _check_Lz(self, ell):
        """Raise for an angular momentum outside the family's grid"""
        if ell < self._Lzgrid[0] or ell > self._Lzgrid[-1]:
            raise ValueError(
                f"|L_z| = {ell} outside the interpolation grid "
                f"[{self._Lzgrid[0]}, {self._Lzgrid[-1]}]" + self._target_hint
            )
        return None

    def _interp_coords(self, jr, jz, ell):
        """Solve the implicit inverse labels: the grid coordinates (x, y)
        of the torus with actions (J_R, J_z) at |L_z|, by root-finding on
        the stored action interpolants themselves -- exact-in-the-family,
        so canonicity is untouched. A vanishing action puts the torus on
        the corresponding face exactly."""

        def A(x, y, **kw):
            return self._A_ip(x, y, ell, **kw)[0, 0]

        # the box the family covers, in x = w_E^2 and y
        xlo, xhi = self._wEs[0] ** 2, self._wEs[-1] ** 2
        ylo, yhi = self._ys[0], self._ys[-1]
        # the residual is relative to the actions, so that actions down to
        # round-off above zero resolve to the edges' neighbourhood rather
        # than to the edges themselves
        tol = 1e-12 * (jr + jz)
        for face, jother, k in ((1.0, jz, 1), (0.0, jr, 0)):
            if (jr if k == 1 else jz) == 0.0:
                if (yhi if face == 1.0 else ylo) != face:
                    raise ValueError(
                        f"(J_R, J_z) = ({jr}, {jz}) lies on a degenerate face "
                        "that the family's grid does not reach"
                    )
                top, bottom = A(xhi, face)[k], A(xlo, face)[k]
                if jother > top + tol or jother < bottom - tol:
                    raise ValueError(
                        f"(J_R, J_z) = ({jr}, {jz}) lies outside the interpolated "
                        f"family at |L_z| = {ell}: "
                        + ("above" if jother > top else "below")
                        + f" the covered actions [{bottom:g}, {top:g}]"
                        + (
                            " (increase Rinf)"
                            if jother > top and self._target is None
                            else ""
                        )
                        + self._target_hint
                    )
                jother = min(max(jother, bottom), top)
                # on a face, the root is found in w_E = sqrt(x), in which a
                # tiny action resolves to a small w_E rather than to zero
                wE = brentq(
                    lambda ww: A(ww * ww, face)[k] - jother,
                    self._wEs[0],
                    self._wEs[-1],
                    xtol=1e-15,
                )
                return wE * wE, face
        # Newton on the total action and on the vertical fraction of it,
        # both of which the grid coordinates control at every scale: near
        # the circular edge both actions vanish with x and their fraction
        # alone fixes y, so a Newton on the actions themselves would have a
        # degenerate direction there
        jtot, frac = jr + jz, jz / (jr + jz)
        x = 0.5 * (xlo + xhi)
        y = min(max(frac, ylo + 0.02 * (yhi - ylo)), yhi - 0.02 * (yhi - ylo))
        for attempt in range(2):
            for _ in range(60):
                v = A(x, y)
                tot = v[0] + v[1]
                G = numpy.array([tot / jtot - 1.0, v[1] / tot - frac])
                if numpy.max(numpy.fabs(G)) < 1e-12:
                    return x, y
                vx, vy = A(x, y, dx=1), A(x, y, dy=1)
                Gx = numpy.array(
                    [
                        (vx[0] + vx[1]) / jtot,
                        (vx[1] * tot - v[1] * (vx[0] + vx[1])) / tot**2,
                    ]
                )
                Gy = numpy.array(
                    [
                        (vy[0] + vy[1]) / jtot,
                        (vy[1] * tot - v[1] * (vy[0] + vy[1])) / tot**2,
                    ]
                )
                det = Gx[0] * Gy[1] - Gy[0] * Gx[1]
                dx = -(Gy[1] * G[0] - Gy[0] * G[1]) / det
                dy = -(-Gx[1] * G[0] + Gx[0] * G[1]) / det
                lim = min(1.0, 0.3 / max(numpy.fabs(dx), numpy.fabs(dy), 1e-300))
                x = min(max(x + lim * dx, xlo), xhi)
                y = min(max(y + lim * dy, ylo), yhi)
            if attempt == 0:
                # restart from the best point of a coarse scan
                xs, ys = numpy.meshgrid(
                    numpy.linspace(xlo + 0.05 * (xhi - xlo), xhi, 12),
                    numpy.linspace(ylo, yhi, 12),
                )
                res = [
                    numpy.sum(numpy.fabs(A(xx, yy)[:2] - [jr, jz]))
                    for xx, yy in zip(xs.flatten(), ys.flatten())
                ]
                k = int(numpy.argmin(res))
                x, y = xs.flatten()[k], ys.flatten()[k]
        top = max(numpy.sum(A(xhi, yy)[:2]) for yy in numpy.linspace(ylo, yhi, 33))
        raise ValueError(
            f"(J_R, J_z) = ({jr}, {jz}) could not be matched inside the "
            f"interpolated family at |L_z| = {ell}"
            + (
                f": its total action lies above the covered {top:g}"
                + (" (increase Rinf)" if self._target is None else "")
                if jr + jz > top
                else ""
            )
            + self._target_hint
        )

    def _interp_tables(self, jr, jz, Lz):
        """The family's tables of the (J_R, J_z, L_z) torus with their
        derivatives along the three action chains, all the interpolants'
        own: the implicit inverse labels, the frequencies from the action
        interpolants' partials, and the turning points and the maps chained
        from their own variables. A vanishing action leaves its libration
        degenerate, with no chain."""
        ell, sgn = numpy.fabs(Lz), numpy.sign(Lz)
        self._check_Lz(ell)
        x, y = self._interp_coords(jr, jz, ell)
        A = self._A_ip(x, y, ell)[0, 0]
        Ax = self._A_ip(x, y, ell, dx=1)[0, 0]
        Ay = self._A_ip(x, y, ell, dy=1)[0, 0]
        AL = self._A_ip(x, y, ell, dz=1)[0, 0]
        # d(x, y, |L_z|)/d(J_R, J_z, J_phi), from the action interpolants'
        # partials (J_phi = L_z moves |L_z| with the sign of L_z)
        det = Ax[0] * Ay[1] - Ay[0] * Ax[1]
        N = numpy.zeros((3, 3))
        N[0, 0], N[0, 1] = Ay[1] / det, -Ay[0] / det
        N[1, 0], N[1, 1] = -Ax[1] / det, Ax[0] / det
        N[0, 2] = -(N[0, 0] * AL[0] + N[0, 1] * AL[1]) * sgn
        N[1, 2] = -(N[1, 0] * AL[0] + N[1, 1] * AL[1]) * sgn
        N[2, 2] = sgn
        circ = self._circular(ell)
        _, dE_dx, dE_dL = self._E_of_xL(x, ell)
        tab = {
            "Om": dE_dx * N[0] + dE_dL * N[2],
            "uc": A[2],
            "duc": Ax[2] * N[0] + Ay[2] * N[1] + AL[2] * N[2],
            "udeg": jr == 0.0,
            "vdeg": jz == 0.0,
        }
        wE = numpy.sqrt(x)
        for deg, ip, s, fs, key in (
            ("udeg", self._U_ip, numpy.sqrt(1.0 - y), -1.0, "u"),
            ("vdeg", self._V_ip, numpy.sqrt(y), 1.0, "v"),
        ):
            if tab[deg]:
                npt = self._npt if key == "u" else self._nptv
                tab["w" + key], tab["dw" + key] = 0.0, numpy.zeros(3)
                tab["Dm" + key] = numpy.zeros(npt)
                tab["dDm" + key] = numpy.zeros((npt, 3))
                continue
            T = ip(wE, s, ell)[0, 0]
            TwE = ip(wE, s, ell, dx=1)[0, 0]
            Ts = ip(wE, s, ell, dy=1)[0, 0]
            TL = ip(wE, s, ell, dz=1)[0, 0]
            # chains through the tables' own variables, w_E = sqrt(x) and
            # s = sqrt(1 - y) or sqrt(y), whose derivatives diverge only at
            # the faces on which this libration is degenerate
            dT = (
                TwE[:, None] * N[0][None, :] / (2.0 * wE)
                + Ts[:, None] * N[1][None, :] * fs / (2.0 * s)
                + TL[:, None] * N[2][None, :]
            )
            tab["w" + key], tab["dw" + key] = T[0], dT[0]
            tab["Dm" + key], tab["dDm" + key] = T[1:], dT[1:]
        return tab

    def _interp_coords_array(self, jr, jz, ell):
        """The grid coordinates (x, y) of many tori at once (none of them on
        a degenerate face): the Newton of _interp_coords vectorized over the
        tori on the batched action interpolant, with the scalar inversion
        as the fallback for any torus that does not converge"""
        n = len(jr)
        xlo, xhi = self._wEs[0] ** 2, self._wEs[-1] ** 2
        ylo, yhi = self._ys[0], self._ys[-1]
        jtot, frac = jr + jz, jz / (jr + jz)
        x = numpy.full(n, 0.5 * (xlo + xhi))
        y = numpy.clip(frac, ylo + 0.02 * (yhi - ylo), yhi - 0.02 * (yhi - ylo))
        active = numpy.arange(n)
        for _ in range(60):
            xa, ya, ea = x[active], y[active], ell[active]
            v, vx, vy, _ = self._A_ip.batch_all(xa, ya, ea)
            tot = v[:, 0] + v[:, 1]
            G0, G1 = tot / jtot[active] - 1.0, v[:, 1] / tot - frac[active]
            done = numpy.maximum(numpy.fabs(G0), numpy.fabs(G1)) < 1e-12
            if numpy.all(done):
                active = active[~done]
                break
            Gx0, Gx1 = (
                (vx[:, 0] + vx[:, 1]) / jtot[active],
                (vx[:, 1] * tot - v[:, 1] * (vx[:, 0] + vx[:, 1])) / tot**2,
            )
            Gy0, Gy1 = (
                (vy[:, 0] + vy[:, 1]) / jtot[active],
                (vy[:, 1] * tot - v[:, 1] * (vy[:, 0] + vy[:, 1])) / tot**2,
            )
            det = Gx0 * Gy1 - Gy0 * Gx1
            dx = -(Gy1 * G0 - Gy0 * G1) / det
            dy = -(-Gx1 * G0 + Gx0 * G1) / det
            lim = numpy.minimum(
                1.0,
                0.3
                / numpy.maximum(numpy.maximum(numpy.fabs(dx), numpy.fabs(dy)), 1e-300),
            )
            # the converged tori stay where they are
            upd = active[~done]
            x[upd] = numpy.clip((xa + lim * dx)[~done], xlo, xhi)
            y[upd] = numpy.clip((ya + lim * dy)[~done], ylo, yhi)
            active = upd
        for ii in active:
            x[ii], y[ii] = self._interp_coords(jr[ii], jz[ii], ell[ii])
        return x, y

    def _interp_tables_array(self, jr, jz, Lz):
        """The family's tables of many (J_R, J_z, L_z) tori at once, none on
        a degenerate face, as _interp_tables returns them for one torus but
        with a leading point axis (the maps' coefficients as (M, npoints),
        their chains as (M, npoints, 3))"""
        ell, sgn = numpy.fabs(Lz), numpy.sign(Lz)
        outside = (ell < self._Lzgrid[0]) | (ell > self._Lzgrid[-1])
        if numpy.any(outside):
            self._check_Lz(ell[numpy.argmax(outside)])
        x, y = self._interp_coords_array(jr, jz, ell)
        A, Ax, Ay, AL = self._A_ip.batch_all(x, y, ell)
        det = Ax[:, 0] * Ay[:, 1] - Ay[:, 0] * Ax[:, 1]
        N = numpy.zeros((len(jr), 3, 3))
        N[:, 0, 0], N[:, 0, 1] = Ay[:, 1] / det, -Ay[:, 0] / det
        N[:, 1, 0], N[:, 1, 1] = -Ax[:, 1] / det, Ax[:, 0] / det
        N[:, 0, 2] = -(N[:, 0, 0] * AL[:, 0] + N[:, 0, 1] * AL[:, 1]) * sgn
        N[:, 1, 2] = -(N[:, 1, 0] * AL[:, 0] + N[:, 1, 1] * AL[:, 1]) * sgn
        N[:, 2, 2] = sgn
        _, dE_dx, dE_dL = self._E_of_xL_array(x, ell)
        tab = {
            "Om": dE_dx[:, None] * N[:, 0] + dE_dL[:, None] * N[:, 2],
            "uc": A[:, 2],
            "duc": Ax[:, 2, None] * N[:, 0]
            + Ay[:, 2, None] * N[:, 1]
            + AL[:, 2, None] * N[:, 2],
            "udeg": False,
            "vdeg": False,
        }
        wE = numpy.sqrt(x)
        for ip, sv, fs, key in (
            (self._U_ip, numpy.sqrt(1.0 - y), -1.0, "u"),
            (self._V_ip, numpy.sqrt(y), 1.0, "v"),
        ):
            T, TwE, Ts, TL = ip.batch_all(wE, sv, ell)
            dT = (
                TwE[:, :, None] * N[:, None, 0, :] / (2.0 * wE)[:, None, None]
                + Ts[:, :, None] * N[:, None, 1, :] * fs / (2.0 * sv)[:, None, None]
                + TL[:, :, None] * N[:, None, 2, :]
            )
            tab["w" + key], tab["dw" + key] = T[:, 0], dT[:, 0]
            tab["Dm" + key], tab["dDm" + key] = (
                T[:, 1:].T,
                numpy.moveaxis(dT[:, 1:], 0, 1),
            )
        return tab

    def _explicit_tables(self, ii):
        """An explicit torus is a one-node family: its own values and exact
        slopes stand in for the interpolants', the slopes in (E, I_3, L_z)
        chained to the actions through the torus's period matrix"""
        N = self._Ns[ii]
        udeg, vdeg = self._wus[ii] == 0.0, self._wvs[ii] == 0.0
        dsup = self._dsups[ii]
        return {
            "Om": N[0],
            "uc": self._ucs[ii],
            "duc": dsup[0] @ N,
            "udeg": udeg,
            "vdeg": vdeg,
            "wu": self._wus[ii],
            "dwu": numpy.zeros(3) if udeg else dsup[1] @ N,
            "Dmu": self._Dmus[ii],
            "dDmu": self._dDmus[ii] @ N,
            "wv": self._wvs[ii],
            "dwv": numpy.zeros(3) if vdeg else dsup[2] @ N,
            "Dmv": self._Dmvs[ii],
            "dDmv": self._dDmvs[ii] @ N,
        }

    def _kernel_u(self, tau, a, e, L, Dm, uc, wu, chains):
        """Everything the evaluation needs from the u libration at anomaly
        tau, in one pass and from the tables alone: the auxiliary anomaly
        eta(tau), the auxiliary's radial angle (Kepler's equation, closed
        form) and its derivative, the function Lambda(eta) of its vertical
        angle relation and its derivative (the branch keyed to the winding
        number of eta, so that both are continuous for any tau), the
        target's u and p_u (the flux identity p_u du/dtau = p^A dr^A/dtau,
        with sin(eta)/sin(tau) grouped so that every factor is regular at
        the turning points), and, for each chain (da, de, duc, dwu, dDm),
        the compensation p^A dr^A/dJ|_tau - p_u du/dJ|_tau."""
        eta, deta, smt = self._map(tau, Dm, self._nforDm)
        se, ce = numpy.sin(eta), numpy.cos(eta)
        rA, pA, gA = self._aAIinv._profile(a, e, eta)
        kap = a * e / (a + self._b)
        thetaA = eta - kap * se
        dthetaA = (1.0 - kap * ce) * deta
        sq = numpy.sqrt(L**2 + 4.0 * self._GM * self._b)
        c1 = numpy.sqrt((1.0 + e) / (1.0 - e))
        c2 = numpy.sqrt(
            (a * (1.0 + e) + 2.0 * self._b) / (a * (1.0 - e) + 2.0 * self._b)
        )
        t2 = numpy.tan(0.5 * eta)
        nwind = numpy.round(eta / (2.0 * numpy.pi))
        Lam = (
            numpy.arctan(c1 * t2)
            + L / sq * numpy.arctan(c2 * t2)
            + numpy.pi * nwind * (1.0 + L / sq)
        )
        ch2, sh2 = numpy.cos(0.5 * eta) ** 2, numpy.sin(0.5 * eta) ** 2
        dLam = (
            0.5 * c1 / (ch2 + c1**2 * sh2) + 0.5 * L / sq * c2 / (ch2 + c2**2 * sh2)
        ) * deta
        st, ct = numpy.sin(tau), numpy.cos(tau)
        u = uc - wu * ct
        # sin(eta)/sin(tau) -> eta'(tau) at the turning points
        sratio = numpy.where(
            numpy.fabs(st) > 1e-12, se / numpy.where(st == 0.0, 1.0, st), deta
        )
        pu = pA * gA * sratio * deta / wu
        drA_da, drA_de = self._aAIinv._radius_partials(a, e, eta)
        comps = [
            pA * (drA_da * da + drA_de * de + gA * se * _contract(smt, dDm))
            - pu * (duc - dwu * ct)
            for da, de, duc, dwu, dDm in chains
        ]
        return eta, thetaA, dthetaA, Lam, dLam, u, pu, comps

    def _kernel_v(self, tau, L, c, Dm, wv, chains):
        """Everything the evaluation needs from the v libration at anomaly
        tau: the auxiliary's polar anomaly zeta(tau) and its derivative,
        the target's v and p_v (the flux identity, regular at the turning
        points), and, for each chain (dc, dwv, dDm), the compensation
        p^A_theta d theta^A/dJ|_tau - p_v dv/dJ|_tau, whose auxiliary part
        is grouped as -L sin(zeta) cos(zeta) / [2 sin^2 theta^A] dc, regular
        down to the planar edge, plus the flux times the map's shift."""
        zeta, dzeta, smt = self._map(tau, Dm, self._nforDmv)
        sz, cz = numpy.sin(zeta), numpy.cos(zeta)
        sinth2 = 1.0 - c * cz**2
        fluxA = L * c * sz**2 / sinth2
        st, ct = numpy.sin(tau), numpy.cos(tau)
        v = 0.5 * numpy.pi - wv * ct
        sratio = numpy.where(
            numpy.fabs(st) > 1e-12, sz / numpy.where(st == 0.0, 1.0, st), dzeta
        )
        pv = L * c * sz * sratio * dzeta / (sinth2 * wv)
        comps = [
            -0.5 * L * sz * cz / sinth2 * dc
            + fluxA * _contract(smt, dDm)
            + pv * dwv * ct
            for dc, dwv, dDm in chains
        ]
        return zeta, dzeta, v, pv, comps

    def _torus_at(self, tu, tv, aux, chains_u, chains_v):
        """The torus at the anomaly pair (tau_u, tau_v): the auxiliary's
        radial and vertical angles with their derivatives, the point
        (u, p_u, v, p_v), the auxiliary's polar anomaly, and the total
        compensation of each chain (the u chains and v chains paired, and
        each libration's part on its own)"""
        n = len(tu)
        if aux["udeg"]:
            # a shell orbit: the auxiliary's radial libration is degenerate
            # too (e = 0), so its radial angle is the anomaly and Lambda is
            # linear in it
            thetaAr, dthetaAr = tu, numpy.ones(n)
            Lam, dLam = aux["ratio"] * tu, aux["ratio"] * numpy.ones(n)
            u, pu = aux["uc"] * numpy.ones(n), numpy.zeros(n)
            comps_u = [numpy.zeros(n)] * len(chains_u)
        else:
            _, thetaAr, dthetaAr, Lam, dLam, u, pu, comps_u = self._kernel_u(
                tu,
                aux["a"],
                aux["e"],
                aux["L"],
                aux["Dmu"],
                aux["uc"],
                aux["wu"],
                chains_u,
            )
        if aux["vdeg"]:
            zeta, dzeta = tv, numpy.ones(n)
            v, pv = 0.5 * numpy.pi * numpy.ones(n), numpy.zeros(n)
            comps_v = [numpy.zeros(n)] * len(chains_v)
        else:
            zeta, dzeta, v, pv, comps_v = self._kernel_v(
                tv, aux["L"], aux["c"], aux["Dmv"], aux["wv"], chains_v
            )
        # the auxiliary's vertical angle: affine in its polar anomaly (the
        # in-plane phase psi = zeta + pi/2), plus its radial angle relation
        return {
            "thetaAr": thetaAr,
            "dthetaAr": dthetaAr,
            "thetaAz": zeta + 0.5 * numpy.pi + aux["ratio"] * thetaAr - Lam,
            "dthetaAz_du": aux["ratio"] * dthetaAr - dLam,
            "dthetaAz_dv": dzeta,
            "zeta": zeta,
            "Lam": Lam,
            "u": u,
            "pu": pu,
            "v": v,
            "pv": pv,
            "comps": [cu + cv for cu, cv in zip(comps_u, comps_v)],
            "comps_u": comps_u,
            "comps_v": comps_v,
        }

    def _newton_angles(self, thR, thz, aux, chains_u, chains_v):
        """Newton solve of the two angle relations theta_R(tau_u, tau_v) and
        theta_z(tau_u, tau_v) -- the auxiliary's angles plus the
        compensations along the J_R and J_z chains -- for the anomaly pair
        of each requested angle pair, vectorized over the angles. The
        Jacobian is that of the auxiliary's angles, in closed form, plus
        that of the compensations, which are small and each the sum of a
        function of tau_u and one of tau_v, so that their four slopes are
        secants between successive iterates and the convergence is
        superlinear; returns the anomalies and the indices of the pairs
        that did not converge within maxiter iterations"""

        def wrap(f):
            return (f + numpy.pi) % (2.0 * numpy.pi) - numpy.pi

        def residuals(x, y):
            T = self._torus_at(x, y, aux, chains_u, chains_v)
            return (
                wrap(T["thetaAr"] + T["comps"][0] - thR),
                wrap(T["thetaAz"] + T["comps"][1] - thz),
                T,
            )

        x = numpy.array(thR, dtype="float")
        # the vertical anomaly that the auxiliary's angle relation gives at
        # tau_u = theta_R with the map and the compensation neglected
        T0 = self._torus_at(x, 0.0 * x, aux, [], [])
        y = thz - 0.5 * numpy.pi - aux["ratio"] * T0["thetaAr"] + T0["Lam"]
        prev, slope = None, numpy.zeros((4, len(x)))
        for _ in range(self._maxiter):
            FR, Fz, T = residuals(x, y)
            if (
                max(numpy.max(numpy.fabs(FR)), numpy.max(numpy.fabs(Fz)))
                < self._angle_tol
            ):
                return x, y, numpy.array([], dtype="int")
            # the compensations of the (J_R, J_z) chains from the u
            # libration, then from the v libration, and where each was taken
            comps = numpy.array([*T["comps_u"], *T["comps_v"]])
            at = numpy.array([x, x, y, y])
            if prev is not None:
                step = at - prev[0]
                moved = numpy.fabs(step) > 1e-9
                slope = numpy.where(
                    moved, (comps - prev[1]) / numpy.where(moved, step, 1.0), slope
                )
            prev = (at, comps)
            # a secant that would take the determinant far from the
            # auxiliary's own is not trusted
            det0 = T["dthetaAr"] * T["dthetaAz_dv"]
            trusted = (T["dthetaAr"] + slope[0]) * (
                T["dthetaAz_dv"] + slope[3]
            ) - slope[2] * (T["dthetaAz_du"] + slope[1]) > 0.5 * det0
            sl = numpy.where(trusted, slope, 0.0)
            a, b = T["dthetaAr"] + sl[0], sl[2]
            c, d = T["dthetaAz_du"] + sl[1], T["dthetaAz_dv"] + sl[3]
            dx = numpy.clip(-(d * FR - b * Fz) / (a * d - b * c), -0.5, 0.5)
            dy = numpy.clip(-(Fz + c * dx) / d, -0.5, 0.5)
            x += dx
            y += dy
        FR, Fz, _ = residuals(x, y)
        return (
            x,
            y,
            numpy.flatnonzero(
                numpy.maximum(numpy.fabs(FR), numpy.fabs(Fz)) >= self._angle_tol
            ),
        )

    def _solve_angles(self, thR, thz, aux, chains_u, chains_v):
        """The anomaly pair of each requested angle pair on one torus: the
        Newton solve, with a bracketed scalar fallback for any pair that
        does not converge"""
        x, y, bad = self._newton_angles(thR, thz, aux, chains_u, chains_v)
        if len(bad) > 0:
            # theta_R - tau_u and theta_z - tau_v are periodic and bounded,
            # so [theta - s, theta + s] with s beyond their extremes are
            # guaranteed brackets: the vertical relation is solved at each
            # trial tau_u inside the solve of the radial one
            g = numpy.linspace(0.0, 2.0 * numpy.pi, 48, endpoint=False)
            GU, GV = (q.flatten() for q in numpy.meshgrid(g, g))
            Ts = self._torus_at(GU, GV, aux, chains_u, chains_v)
            sR = numpy.max(numpy.fabs(Ts["thetaAr"] + Ts["comps"][0] - GU)) + 0.1
            sz = numpy.max(numpy.fabs(Ts["thetaAz"] + Ts["comps"][1] - GV)) + 0.1
            for ii in bad:
                thri, thzi = thR[ii], thz[ii]

                def _tv(xx):
                    def _fz(yy):
                        T = self._torus_at(
                            numpy.array([xx]),
                            numpy.array([yy]),
                            aux,
                            chains_u,
                            chains_v,
                        )
                        return T["thetaAz"][0] + T["comps"][1][0] - thzi

                    return brentq(_fz, thzi - sz, thzi + sz, xtol=1e-15, maxiter=200)

                def _fR(xx):
                    T = self._torus_at(
                        numpy.array([xx]),
                        numpy.array([_tv(xx)]),
                        aux,
                        chains_u,
                        chains_v,
                    )
                    return T["thetaAr"][0] + T["comps"][0][0] - thri

                x[ii] = brentq(_fR, thri - sR, thri + sR, xtol=1e-15, maxiter=200)
                y[ii] = _tv(x[ii])
        return x, y

    def _match_node(self, jr, jz, Lz):
        """Locate the explicit torus with actions (J_R, J_z, L_z)"""
        dev = (
            numpy.fabs(self._jrs - jr)
            + numpy.fabs(self._jzs - jz)
            + numpy.fabs(self._Lzs - Lz)
        )
        ii = numpy.argmin(dev)
        if dev[ii] > 1e-8 * (1.0 + numpy.fabs(jr) + numpy.fabs(jz) + numpy.fabs(Lz)):
            raise ValueError(
                f"(J_R, J_z, L_z) = ({jr}, {jz}, {Lz}) is not one of the set-up "
                "tori; discrete mode evaluates the stored tori only (use "
                "setup_interp=True to interpolate)"
            )
        return ii

    # ---------- the public inverse map
    def _evaluate(self, jr, jphi, jz, angler, anglephi, anglez, **kwargs):
        return self._xvFreqs(jr, jphi, jz, angler, anglephi, anglez, **kwargs)[:6]

    def _actions_from_integrals(self, E, Lz, I3):
        """(J_R, J_z) of the torus with integrals (E, L_z, I_3)"""
        E = conversion.parse_energy(E, vo=self._vo)
        Lz = conversion.parse_angmom(Lz, ro=self._ro, vo=self._vo)
        I3 = conversion.parse_energy(I3, vo=self._vo)
        if not self._interp:
            dev = (
                numpy.fabs(self._Es - E)
                + numpy.fabs(self._Lzs - Lz)
                + numpy.fabs(self._I3s - I3)
            )
            ii = numpy.argmin(dev)
            if dev[ii] > 1e-10 * (
                1.0 + numpy.fabs(E) + numpy.fabs(Lz) + numpy.fabs(I3)
            ):
                raise ValueError(
                    f"(E, L_z, I_3) = ({E}, {Lz}, {I3}) is not one of the set-up "
                    "tori; to use interpolation, initialize with setup_interp=True"
                )
            return self._jrs[ii], self._jzs[ii]
        ell = numpy.fabs(Lz)
        self._check_Lz(ell)
        circ = self._circular(ell)
        Ec, Emax = circ["Ec"], self._Emax(ell)
        tol = 1e-12 * (Emax - Ec)
        if E < Ec - tol or E > Emax + tol:
            raise ValueError(
                f"E = {E} outside the interpolation grid at |L_z| = {ell}: [{Ec}, {Emax}]"
            )
        xlo, xhi = self._wEs[0] ** 2, self._wEs[-1] ** 2
        ylo, yhi = self._ys[0], self._ys[-1]
        if E < Ec + tol:
            # the circular orbit, to the tolerance of the torus construction
            self._check_circular()
            return 0.0, 0.0
        # the normalized energy from the apocentre of the planar orbit
        # without angular momentum at E
        R0c, _, R0m, _ = self._R0_edges(ell)
        R0 = self._R0(min(E, Emax), R0c, R0m)
        x = min(numpy.log(R0 / R0c) / numpy.log(R0m / R0c), 1.0)
        if x < xlo - 1e-10 or x > xhi + 1e-10:
            raise ValueError(
                f"E = {E} outside the interpolation grid at |L_z| = {ell}: its energy "
                f"variable {x:g} lies outside [{xlo:g}, {xhi:g}]" + self._target_hint
            )
        with self._in_chart(E, ell):
            ush = self._ushell(E, ell)
            Ipl, Ish = self._I3_planar(E, ell), self._I3_shell(E, ell, ush)
        y = (I3 - Ipl) / (Ish - Ipl)
        if y < ylo - 1e-10 or y > yhi + 1e-10:
            raise ValueError(
                f"I_3 = {I3} outside the interpolation grid at (E, |L_z|) = "
                f"({E}, {ell}): [{Ipl + ylo * (Ish - Ipl)}, {Ipl + yhi * (Ish - Ipl)}]"
                + self._target_hint
            )
        A = self._A_ip(min(max(x, xlo), xhi), min(max(y, ylo), yhi), ell)[0, 0]
        return A[0], A[1]

    @conversion.physical_conversion("action", pop=True)
    def JR(self, E, Lz, I3, **kwargs):
        """
        Return the radial action of the torus with integrals (E, L_z, I_3).

        Parameters
        ----------
        E : float or Quantity
            Energy of the torus.
        Lz : float or Quantity
            z-component of the angular momentum of the torus.
        I3 : float or Quantity
            Third integral of the torus (see the class documentation for
            its convention).

        Returns
        -------
        float or Quantity
            Radial action J_R of the torus.

        Notes
        -----
        - For an interpolating instance (setup_interp=True), any (E, L_z,
          I_3) within the grid, read from the family's own action
          interpolant (exact at the nodes); for an instance set up with
          explicit tori, (E, L_z, I_3) must be one of them.
        - 2026-09-24 - Written - Bovy (UofT)
        """
        return self._actions_from_integrals(E, Lz, I3)[0]

    @conversion.physical_conversion("action", pop=True)
    def Jz(self, E, Lz, I3, **kwargs):
        """
        Return the vertical action of the torus with integrals (E, L_z, I_3).

        Parameters
        ----------
        E : float or Quantity
            Energy of the torus.
        Lz : float or Quantity
            z-component of the angular momentum of the torus.
        I3 : float or Quantity
            Third integral of the torus (see the class documentation for
            its convention).

        Returns
        -------
        float or Quantity
            Vertical action J_z of the torus.

        Notes
        -----
        - As JR.
        - 2026-09-24 - Written - Bovy (UofT)
        """
        return self._actions_from_integrals(E, Lz, I3)[1]

    def _tables(self, jr, jz, Lz):
        """The tables of the requested torus, from the family or from the
        explicit tori"""
        if self._interp:
            return self._interp_tables(jr, jz, Lz)
        return self._explicit_tables(self._match_node(jr, jz, Lz))

    def _xvFreqs(self, jr, jphi, jz, angler, anglephi, anglez, **kwargs):
        """(J, theta) -> (x, v) and the frequencies, for actions and angles
        that broadcast against each other by numpy's rules (scalar actions
        with arrays of angles: one torus at those angles; arrays of actions
        with scalar angles: that angle on every torus; equal shapes: paired
        element by element, each star on its own torus at its own angle;
        actions of shape (n,1) with angles of shape (m,): every angle on
        every torus as (n,m) arrays); a single torus returns its
        frequencies as floats"""
        scalar = numpy.ndim(jr) == 0 and numpy.ndim(jphi) == 0 and numpy.ndim(jz) == 0
        jr, jphi, jz, angler, anglephi, anglez = numpy.broadcast_arrays(
            *(
                numpy.atleast_1d(numpy.asarray(t, dtype="float"))
                for t in (jr, jphi, jz, angler, anglephi, anglez)
            )
        )
        if numpy.any(jr < 0.0) or numpy.any(jz < 0.0):
            raise ValueError("J_R and J_z have to be non-negative")
        if numpy.any(jphi == 0.0):
            raise ValueError("L_z = 0 is not supported")
        shape = jr.shape
        jr, jphi, jz, thR, anglephi, thz = (
            t.ravel()
            for t in (
                jr,
                jphi,
                jz,
                angler % (2.0 * numpy.pi),
                anglephi,
                anglez % (2.0 * numpy.pi),
            )
        )
        if scalar:
            out = self._xvFreqs_torus(jr[0], jphi[0], jz[0], thR, anglephi, thz)
            return (*(o.reshape(shape) for o in out[:6]), *out[6:])
        return tuple(
            numpy.stack(
                _chunked(
                    self._xvFreqs_points, len(jr), jr, jphi, jz, thR, anglephi, thz
                )
            ).reshape((9,) + shape)
        )

    def _xvFreqs_points(self, jr, jphi, jz, thR, anglephi, thz):
        """The evaluation at flat arrays of actions and angles of one length:
        the tables are read once per distinct torus; a family's tori with
        both librations are evaluated in one vectorized pass with one row
        of tables per point, and the others (explicit tori, tori on a
        degenerate face, a single distinct torus, and any point whose angle
        solve does not converge) through the one-torus path"""
        out = numpy.empty((9, len(jr)))

        def one_torus(actions, pts):
            for o, q in zip(
                out, self._xvFreqs_torus(*actions, thR[pts], anglephi[pts], thz[pts])
            ):
                o[pts] = q

        tori, inv = numpy.unique(
            numpy.stack([jr, jphi, jz], axis=1), axis=0, return_inverse=True
        )
        inv = inv.ravel()
        single = (tori[:, 0] == 0.0) | (tori[:, 2] == 0.0)
        if not self._interp or len(tori) == 1:
            single[:] = True
        order = numpy.argsort(inv, kind="stable")
        starts = numpy.searchsorted(inv[order], numpy.arange(len(tori) + 1))
        for k in numpy.flatnonzero(single):
            one_torus(tori[k], order[starts[k] : starts[k + 1]])
        if not numpy.all(single):
            pts = numpy.flatnonzero(~single[inv])
            # each point's row among the tori of the vectorized pass
            rows = (numpy.cumsum(~single) - 1)[inv[pts]]
            out[:, pts], bad = self._xvFreqs_rows(
                tori[~single], rows, thR[pts], anglephi[pts], thz[pts]
            )
            for ii in pts[bad]:
                one_torus(tori[inv[ii]], numpy.array([ii]))
        return tuple(out)

    def _xvFreqs_torus(self, jr, jphi, jz, thR, anglephi, thz):
        """One torus at flat arrays of angles: solve the two angle relations
        for the anomaly pair of each requested angle pair, reconstruct the
        point from the two librations (u, p_u, v, p_v) -> (R, v_R, z, v_z),
        and take the azimuth from the auxiliary's angle relations at the
        compensated azimuthal angle; the frequencies are floats"""
        Lz = jphi
        sgn = numpy.sign(Lz)
        if jr == 0.0 and jz == 0.0:
            # the circular orbit: the point sits at the circular radius in
            # the plane with no radial or vertical motion, at the requested
            # azimuthal angle (the auxiliary's angle relations give phi =
            # theta_phi there), with the epicycle, circular, and vertical
            # frequencies
            if self._interp:
                self._check_Lz(numpy.fabs(Lz))
                self._check_circular()
            else:
                self._match_node(jr, jz, Lz)
            circ = self._circular(Lz)
            Rc = circ["Rc"]
            return (
                Rc + 0.0 * thR,
                0.0 * thR,
                Lz / Rc + 0.0 * thR,
                0.0 * thR,
                0.0 * thR,
                anglephi % (2.0 * numpy.pi),
                circ["kappa"],
                sgn * circ["Omc"],
                circ["nu"],
            )
        tab = self._tables(jr, jz, Lz)
        L = jz + numpy.fabs(Lz)
        a, e = self._aAIinv._orbital_params(jr, L)
        # c = sin^2 i = 1 - L_z^2/L^2 in the product form, exact for a thin
        # vertical libration where the difference would cancel
        c = jz * (jz + 2.0 * numpy.fabs(Lz)) / L**2
        sq = numpy.sqrt(L**2 + 4.0 * self._GM * self._b)
        aux = {
            "a": a,
            "e": e,
            "L": L,
            "c": c,
            "ratio": 0.5 * (1.0 + L / sq),
            "uc": tab["uc"],
            "wu": tab["wu"],
            "Dmu": tab["Dmu"],
            "wv": tab["wv"],
            "Dmv": tab["Dmv"],
            "udeg": tab["udeg"],
            "vdeg": tab["vdeg"],
        }
        # the three action chains, (J_R, J_z, J_phi): through the auxiliary
        # torus's parameters -- (a, e) from (J_R, L) and c = sin^2 i from
        # (L, L_z), with L = J_z + |L_z| -- the turning points, and the
        # maps' coefficients
        dL = numpy.array([0.0, 1.0, sgn])
        dc = 2.0 * Lz**2 / L**3 * dL - 2.0 * Lz / L**2 * numpy.array([0.0, 0.0, 1.0])
        chains_u, chains_v = [], []
        for i in range(3):
            # the auxiliary's radial libration is degenerate with the
            # target's (J^A_r = J_R), and then has no chains to speak of
            da, de = 0.0, 0.0
            if not tab["udeg"]:
                _, _, da, de = self._aAIinv._orbital_param_chains(
                    jr, L, 1.0 if i == 0 else 0.0, dL[i]
                )
            chains_u.append((da, de, tab["duc"][i], tab["dwu"][i], tab["dDmu"][:, i]))
            chains_v.append((dc[i], tab["dwv"][i], tab["dDmv"][:, i]))
        tu, tv = self._solve_angles(thR, thz, aux, chains_u[:2], chains_v[:2])
        T = self._torus_at(tu, tv, aux, chains_u, chains_v)
        Om = tab["Om"]
        return (*self._point(T, Lz, L, anglephi), Om[0], Om[2], Om[1])

    def _point(self, T, Lz, L, anglephi):
        """(R, v_R, v_T, z, v_z, phi) from the two librations of _torus_at:
        the azimuth from the auxiliary's angle relations, phi = theta^A_phi
        - sgn(L_z) theta^A_z + arctan2(cos i sin psi, cos psi) with the
        in-plane phase psi = zeta + pi/2 and cos i = L_z / L, at the
        compensated azimuthal angle"""
        zeta = T["zeta"]
        phi = (
            anglephi
            - T["comps"][2]
            - numpy.sign(Lz) * T["thetaAz"]
            + numpy.arctan2(Lz / L * numpy.cos(zeta), -numpy.sin(zeta))
        )
        u, pu, v, pv = T["u"], T["pu"], T["v"], T["pv"]
        sh, ch = numpy.sinh(u), numpy.cosh(u)
        sn, cs = numpy.sin(v), numpy.cos(v)
        den = self._delta * (sh**2 + sn**2)
        R = self._delta * sh * sn
        return (
            R,
            (pu * ch * sn + pv * sh * cs) / den,
            Lz / R,
            self._delta * ch * cs,
            (pu * sh * cs - pv * ch * sn) / den,
            phi % (2.0 * numpy.pi),
        )

    def _xvFreqs_rows(self, tori, rows, thR, anglephi, thz):
        """The vectorized pass over a family's tori with both librations,
        given as rows of (J_R, J_phi, J_z), at points that each sit on the
        torus rows[point]: the batched label inversion and tables of the
        tori, one row of them per point, the kernels with per-point
        coefficients, and the Newton solve of the angle relations; returns
        the nine outputs and the points whose solve did not converge"""
        jr, Lz, jz = tori.T
        sgn, ell = numpy.sign(Lz), numpy.fabs(Lz)
        tab = self._interp_tables_array(jr, jz, Lz)
        L = jz + ell
        a, e = self._aAIinv._orbital_params(jr, L)
        c = jz * (jz + 2.0 * ell) / L**2
        sq = numpy.sqrt(L**2 + 4.0 * self._GM * self._b)
        aux = {
            "a": a[rows],
            "e": e[rows],
            "L": L[rows],
            "c": c[rows],
            "ratio": 0.5 * (1.0 + L / sq)[rows],
            "uc": tab["uc"][rows],
            "wu": tab["wu"][rows],
            "Dmu": tab["Dmu"][:, rows],
            "wv": tab["wv"][rows],
            "Dmv": tab["Dmv"][:, rows],
            "udeg": False,
            "vdeg": False,
        }
        zero, one = numpy.zeros(len(jr)), numpy.ones(len(jr))
        dL = numpy.stack([zero, one, sgn], axis=1)
        dc = (2.0 * Lz**2 / L**3)[:, None] * dL - (2.0 * Lz / L**2)[
            :, None
        ] * numpy.stack([zero, zero, one], axis=1)
        chains_u, chains_v = [], []
        for i in range(3):
            _, _, da, de = self._aAIinv._orbital_param_chains(
                jr, L, 1.0 if i == 0 else 0.0, dL[:, i]
            )
            chains_u.append(
                (
                    da[rows],
                    de[rows],
                    tab["duc"][rows, i],
                    tab["dwu"][rows, i],
                    tab["dDmu"][:, rows, i],
                )
            )
            chains_v.append((dc[rows, i], tab["dwv"][rows, i], tab["dDmv"][:, rows, i]))
        tu, tv, bad = self._newton_angles(thR, thz, aux, chains_u[:2], chains_v[:2])
        T = self._torus_at(tu, tv, aux, chains_u, chains_v)
        Om = tab["Om"][rows]
        return (
            numpy.stack(
                [*self._point(T, Lz[rows], L[rows], anglephi), *Om[:, [0, 2, 1]].T]
            ),
            bad,
        )

    def _Freqs(self, jr, jphi, jz, **kwargs):
        """Frequencies of the (J_R, J_z, L_z) tori, for actions that
        broadcast against each other: in interpolation mode the stored
        action interpolants' own partials through the label chain (the
        integrator contract); for explicit tori the tori's own period
        matrices; floats for a single torus"""
        scalar = numpy.ndim(jr) == 0 and numpy.ndim(jphi) == 0 and numpy.ndim(jz) == 0
        jr, jphi, jz = numpy.broadcast_arrays(
            *(numpy.atleast_1d(numpy.asarray(t, dtype="float")) for t in (jr, jphi, jz))
        )
        if scalar:
            return self._Freqs_torus(jr[0], jphi[0], jz[0])
        tori, inv = numpy.unique(
            numpy.stack([t.ravel() for t in (jr, jphi, jz)], axis=1),
            axis=0,
            return_inverse=True,
        )
        single = (tori[:, 0] == 0.0) | (tori[:, 2] == 0.0)
        if not self._interp or len(tori) == 1:
            single[:] = True
        Om = numpy.empty((3, len(tori)))
        for k in numpy.flatnonzero(single):
            Om[:, k] = self._Freqs_torus(*tori[k])
        if not numpy.all(single):
            Om[:, ~single] = self._interp_tables_array(
                tori[~single, 0], tori[~single, 2], tori[~single, 1]
            )["Om"][:, [0, 2, 1]].T
        return tuple(o.reshape(jr.shape) for o in Om[:, inv.ravel()])

    def _Freqs_torus(self, jr, Lz, jz):
        """Frequencies of one torus, as floats"""
        if jr == 0.0 and jz == 0.0:
            if self._interp:
                self._check_Lz(numpy.fabs(Lz))
                self._check_circular()
            else:
                self._match_node(jr, jz, Lz)
            circ = self._circular(Lz)
            return (circ["kappa"], numpy.sign(Lz) * circ["Omc"], circ["nu"])
        Om = self._tables(jr, jz, Lz)["Om"]
        return (Om[0], Om[2], Om[1])
