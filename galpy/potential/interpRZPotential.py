import copy
import ctypes
import ctypes.util
from functools import wraps

import numpy
from numpy.ctypeslib import ndpointer
from scipy import interpolate

from ..backend import get_namespace, is_backend_array, match_input_dtype
from ..backend.interpolate import (
    eval_ppoly,
    eval_rect_ppoly,
    rect_bivariate_to_ppoly,
    spline_to_ppoly,
)
from ..util import _load_extension_libs, multi
from ..util.conversion import physical_conversion
from .Potential import Potential

_DEBUG = False

_lib, ext_loaded = _load_extension_libs.load_libgalpy()


def scalarVectorDecorator(func):
    """Decorator to return scalar outputs as a set"""

    @wraps(func)
    def scalar_wrapper(*args, **kwargs):
        if is_backend_array(args[1]) or is_backend_array(args[2]):
            # backend (jax/torch) R/z: skip the numpy scalar/vector normalization;
            # the inner function's backend branch broadcasts (R,z) natively.
            return func(*args, **kwargs)
        if (
            numpy.array(args[1]).shape == () and numpy.array(args[2]).shape == ()
        ):  # only if both R and z are scalars
            scalarOut = True
            args = (args[0], numpy.array([args[1]]), numpy.array([args[2]]))
        elif (
            numpy.array(args[1]).shape == () and not numpy.array(args[2]).shape == ()
        ):  # R scalar, z vector
            scalarOut = False
            args = (args[0], args[1] * numpy.ones_like(args[2]), args[2])
        elif (
            not numpy.array(args[1]).shape == () and numpy.array(args[2]).shape == ()
        ):  # R vector, z scalar
            scalarOut = False
            args = (args[0], args[1], args[2] * numpy.ones_like(args[1]))
        else:
            scalarOut = False
        result = func(*args, **kwargs)
        if scalarOut:
            return result[0]
        else:
            return result

    return scalar_wrapper


def zsymDecorator(odd):
    """Decorator to deal with zsym=True input; set odd=True if the function is an odd function of z (like zforce)"""

    def wrapper(func):
        @wraps(func)
        def zsym_wrapper(*args, **kwargs):
            R, z = args[1], args[2]
            backend = is_backend_array(R) or is_backend_array(z)
            if args[0]._zsym:
                absz = get_namespace(R, z).abs(z) if backend else numpy.fabs(z)
                out = func(args[0], R, absz, **kwargs)
            else:
                out = func(*args, **kwargs)
            if odd and args[0]._zsym:
                # out can be a backend array even for numpy R,z under a forced
                # backend (the interpolated force resolves the forced namespace),
                # so key the sign correction off the output, not the inputs
                if is_backend_array(out):
                    xp = get_namespace(out)
                    return xp.where(xp.asarray(z) < 0.0, -1.0, 1.0) * out
                return sign(z) * out
            else:
                return out

        return zsym_wrapper

    return wrapper


def scalarDecorator(func):
    """Decorator to return scalar output for 1D functions (vcirc,etc.)"""

    @wraps(func)
    def scalar_wrapper(*args, **kwargs):
        if is_backend_array(args[1]):
            # backend (jax/torch) R: skip the numpy scalar normalization; the
            # inner function's backend branch broadcasts R natively.
            return func(*args, **kwargs)
        if numpy.array(args[1]).shape == ():
            scalarOut = True
            args = (args[0], numpy.array([args[1]]))
        else:
            scalarOut = False
        result = func(*args, **kwargs)
        if scalarOut:
            return result[0]
        else:
            return result

    return scalar_wrapper


def _spot_check_cells(nR, nz):
    """Deterministic (i, j) sample for verifying a vectorised grid.

    A 3x5 lattice over both axes -- so both R edges, both z edges, all four
    corners and the interior -- plus the two diagonals, which break the lattice
    alignment so a stride-periodic bug cannot sit entirely between samples.
    ~19 cells regardless of grid size, against 2*nz for a two-row check (502 at
    the default 251x251): measured 31x cheaper on the potentials where this
    actually costs anything.
    """
    if nR < 1 or nz < 1:
        # Degenerate grid: nR-1 would index -1. Sample nothing and let the
        # spline fitter reject it as it already did before this spot check
        # existed ("(mx>kx) failed ... mx=0"), which names the real problem.
        return []
    ii = sorted({0, nR // 2, nR - 1})
    jj = sorted({0, nz // 4, nz // 2, 3 * nz // 4, nz - 1})
    cells = {(a, b) for a in ii for b in jj}
    for k in range(4):  # both diagonals, off-lattice
        f = k / 3.0
        cells.add((int(f * (nR - 1)), int(f * (nz - 1))))
        cells.add((int(f * (nR - 1)), int((1.0 - f) * (nz - 1))))
    return sorted(cells)


def _grid_eval(evaluator, pot, rgrid, zgrid):
    """Sample ``evaluator(pot, R, z)`` on the ``(rgrid, zgrid)`` tensor grid.

    Tries a single vectorised call over the whole meshgrid and falls back to the
    cell-by-cell loop. The fall-back is the *identical* computation, so nothing
    is masked: a genuine error re-raises from the loop.

    Two things can go wrong with the vectorised call, and both are handled:

    * the potential rejects an array outright -- it raises, and we loop;
    * the potential neither raises nor broadcasts correctly. This is the
      dangerous one, because a ``try/except`` sails straight past it and the
      whole interpolation grid is then built from wrong numbers. Measured
      2026-08-10, ``AnySphericalPotential`` does exactly that: its array results
      differ from the cell-by-cell ones in 95 % of cells, silently.

    **Composites are not a special case and do not need one.** A list or a
    ``CompositePotential`` broadcasts iff its components do, because the
    evaluator just sums them. Verified over all 56 pairwise composites (both
    spellings) of 8 individually-vectorisable potentials: **none** fell back.
    Only a composite *containing* an array-unsafe component -- in practice
    ``AnySphericalPotential`` -- falls back, which is the correct outcome, since
    the sum is then as wrong as its worst term.

    So the vectorised result is accepted only after it reproduces the scalar
    path **bit for bit** on a spot-check sample (see `_spot_check_cells`).
    Bit-for-bit rather than within a tolerance is deliberate: the point of this
    is to be a pure speed-up, so anything that would move a shipped value falls
    back instead. Across a 19-potential zoo x the 7 sampled quantities, 94 of
    108 combinations are bit-identical over the entire grid, 9 raise, and 5 (all
    ``AnySphericalPotential``) differ and are caught here.

    The sample is ~19 cells rather than two whole rows. That is a deliberate
    trade: it costs 31x less on potentials where a scalar call is expensive
    (``DoubleExponentialDiskPotential``: 0.008 s vs 0.251 s), at the price of
    being probabilistic for a *sparse* disagreement. It is not probabilistic for
    the failure this actually guards against -- a broadcasting bug is a
    whole-array phenomenon, and the one real instance disagrees in 95 % of
    cells, which ~19 independent samples miss with probability 5e-25.
    """
    nR, nz = len(rgrid), len(zgrid)

    def _loop():
        out = numpy.empty((nR, nz))
        for ii in range(nR):
            for jj in range(nz):
                out[ii, jj] = evaluator(pot, rgrid[ii], zgrid[jj], use_physical=False)
        return out

    Rmesh, zmesh = numpy.meshgrid(rgrid, zgrid, indexing="ij")
    try:
        raw = evaluator(pot, Rmesh, zmesh, use_physical=False)
        grid = numpy.asarray(raw)
    except Exception:  # scalar-only potentials must be driven cell by cell
        return _loop()
    if grid.shape != (nR, nz):
        return _loop()
    # numpy compares bit for bit; jax/torch reassociate reductions differently
    # between a whole-mesh call and a scalar one, so an exact test rejects a
    # CORRECT vectorised result (measured: 2 of 9 cells, worst 1.6e-15) and
    # falls back for every cell -- 1643x on a 201x201 MWPotential build. A
    # relative tolerance still catches the failure this guards against: the
    # disagreement it exists to catch is not in the ULPs.
    #
    # 1e-14, i.e. ~6x the measured 1.6e-15 reassociation, NOT 1e-12: the margin
    # over ULP noise has to stay small enough that this is a reassociation
    # allowance and not a correctness allowance. The bound is pinned from both
    # sides by test_grid_eval_falls_back_when_the_vectorised_call_disagrees,
    # which injects a 1e-13 relative error that 1e-14 catches and 1e-12 does not.
    rtol = 0.0 if not is_backend_array(raw) else 1e-14
    for ii, jj in _spot_check_cells(nR, nz):
        ref = numpy.asarray(evaluator(pot, rgrid[ii], zgrid[jj], use_physical=False))
        got = numpy.asarray(grid[ii, jj])
        # rtol=0 atol=0 makes allclose exactly `array_equal` (verified over nan,
        # +-inf, -0.0 and denormals), so numpy keeps bit-for-bit through the same
        # single expression -- no backend-only branch to leave uncovered.
        if not numpy.allclose(got, ref, rtol=rtol, atol=0.0, equal_nan=True):
            return _loop()
    return grid


class interpRZPotential(Potential):
    """Class that interpolates a given potential on a grid for fast orbit integration"""

    def __init__(
        self,
        RZPot=None,
        rgrid=(numpy.log(0.01), numpy.log(20.0), 101),
        zgrid=(0.0, 1.0, 101),
        logR=True,
        interpPot=False,
        interpRforce=False,
        interpzforce=False,
        interpDens=False,
        interpvcirc=False,
        interpdvcircdr=False,
        interpepifreq=False,
        interpverticalfreq=False,
        ro=None,
        vo=None,
        use_c=False,
        enable_c=False,
        zsym=True,
        numcores=None,
        interpR2deriv=False,
        interpz2deriv=False,
        interpRzderiv=False,
    ):
        """
        Initialize an interpRZPotential instance.

        Parameters
        ----------
        RZPot : RZPotential or a combined potential formed using addition (pot1+pot2+…)
            RZPotential to be interpolated.
        rgrid : tuple, optional
            R grid to be given to linspace as in rs= linspace(*rgrid).
        zgrid : tuple, optional
            z grid to be given to linspace as in zs= linspace(*zgrid).
        logR : bool, optional
            If True, rgrid is in the log of R so logrs= linspace(*rgrid).
        interpPot : bool, optional
            If True, interpolate the potential.
        interpRforce : bool, optional
            If True, interpolate the radial force.
        interpzforce : bool, optional
            If True, interpolate the vertical force.
        interpDens : bool, optional
            If True, interpolate the density.
        interpvcirc : bool, optional
            If True, interpolate the circular velocity.
        interpdvcircdr : bool, optional
            If True, interpolate the derivative of the circular velocity with respect to R.
        interpepifreq : bool, optional
            If True, interpolate the epicyclic frequency.
        interpverticalfreq : bool, optional
            If True, interpolate the vertical frequency.
        ro : float, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float, optional
            Velocity scale for translation into internal units (default from configuration file).
        use_c : bool, optional
            Use C to speed up the calculation of the grid.
        enable_c : bool, optional
            Enable use of C for interpolations.
        zsym : bool, optional
            If True (default), the potential is assumed to be symmetric around z=0 (so you can use, e.g.,  zgrid=(0.,1.,101)).
        numcores : int, optional
            If set to an integer, use this many cores (only used for vcirc, dvcircdR, epifreq, and verticalfreq; NOT NECESSARILY FASTER, TIME TO MAKE SURE).
        interpR2deriv : bool, optional
            If True, interpolate the second radial derivative of the potential (grid of exact values, computed in Python regardless of use_c).
        interpz2deriv : bool, optional
            If True, interpolate the second vertical derivative of the potential (grid of exact values, computed in Python regardless of use_c).
        interpRzderiv : bool, optional
            If True, interpolate the mixed radial-vertical derivative of the potential (grid of exact values, computed in Python regardless of use_c). Together with interpR2deriv and interpz2deriv (and the potential and forces) this provides the full 3D Hessian in C (hasC_dxdv3d) for integrating phase-space volumes (integrate_dxdv) when enable_c=True.

        Notes
        -----
        - 2010-07-21 - Written - Bovy (NYU)
        - 2013-01-24 - Started with new implementation - Bovy (IAS)
        - 2026-06-09 - Added interpolated 2nd derivatives (full 3D Hessian in C for the 3D variational equations) - Bovy (UofT)

        """
        if isinstance(RZPot, interpRZPotential):
            from ..potential import PotentialError

            raise PotentialError(
                "Cannot setup interpRZPotential with another interpRZPotential"
            )
        # Propagate ro and vo
        roSet = True
        voSet = True
        if ro is None:
            if isinstance(RZPot, list):
                ro = RZPot[0]._ro
                roSet = RZPot[0]._roSet
            else:
                ro = RZPot._ro
                roSet = RZPot._roSet
        if vo is None:
            if isinstance(RZPot, list):
                vo = RZPot[0]._vo
                voSet = RZPot[0]._voSet
            else:
                vo = RZPot._vo
                voSet = RZPot._voSet
        Potential.__init__(self, amp=1.0, ro=ro, vo=vo)
        # Turn off physical if it hadn't been on
        if not roSet:
            self._roSet = False
        if not voSet:
            self._voSet = False
        self._origPot = RZPot
        self._rgrid = numpy.linspace(*rgrid)
        self._logR = logR
        if self._logR:
            self._rgrid = numpy.exp(self._rgrid)
            self._logrgrid = numpy.log(self._rgrid)
        self._zgrid = numpy.linspace(*zgrid)
        self._interpPot = interpPot
        self._interpRforce = interpRforce
        self._interpzforce = interpzforce
        self._interpDens = interpDens
        self._interpvcirc = interpvcirc
        self._interpdvcircdr = interpdvcircdr
        self._interpepifreq = interpepifreq
        self._interpverticalfreq = interpverticalfreq
        self._interpR2deriv = interpR2deriv
        self._interpz2deriv = interpz2deriv
        self._interpRzderiv = interpRzderiv
        self._enable_c = enable_c * ext_loaded
        self.hasC = self._enable_c
        # No planar version of interpRZPotential in the C integrator
        self.hasC_planar = False
        # Full 3D Hessian in C (R2deriv/z2deriv/Rzderiv interpolated from
        # their own grids of exact values; phi derivatives are identically
        # zero for this axisymmetric potential), as required by the 3D
        # variational equations (integrate_dxdv); needs the potential and
        # forces in C too (they drive the orbit part of the variational flow).
        self.hasC_dxdv3d = bool(
            self._enable_c
            * interpPot
            * interpRforce
            * interpzforce
            * interpR2deriv
            * interpz2deriv
            * interpRzderiv
        )
        self._zsym = zsym
        if interpPot:
            if use_c * ext_loaded:
                self._potGrid, err = calc_potential_c(
                    self._origPot, self._rgrid, self._zgrid
                )
            else:
                from ..potential import evaluatePotentials

                self._potGrid = _grid_eval(
                    evaluatePotentials, self._origPot, self._rgrid, self._zgrid
                )
            if self._logR:
                self._potInterp = interpolate.RectBivariateSpline(
                    self._logrgrid, self._zgrid, self._potGrid, kx=3, ky=3, s=0.0
                )
            else:
                self._potInterp = interpolate.RectBivariateSpline(
                    self._rgrid, self._zgrid, self._potGrid, kx=3, ky=3, s=0.0
                )
            if enable_c * ext_loaded:
                self._potGrid_splinecoeffs = calc_2dsplinecoeffs_c(self._potGrid)
        if interpRforce:
            if use_c * ext_loaded:
                self._rforceGrid, err = calc_potential_c(
                    self._origPot, self._rgrid, self._zgrid, rforce=True
                )
            else:
                from ..potential import evaluateRforces

                self._rforceGrid = _grid_eval(
                    evaluateRforces, self._origPot, self._rgrid, self._zgrid
                )
            if self._logR:
                self._rforceInterp = interpolate.RectBivariateSpline(
                    self._logrgrid, self._zgrid, self._rforceGrid, kx=3, ky=3, s=0.0
                )
            else:
                self._rforceInterp = interpolate.RectBivariateSpline(
                    self._rgrid, self._zgrid, self._rforceGrid, kx=3, ky=3, s=0.0
                )
            if enable_c * ext_loaded:
                self._rforceGrid_splinecoeffs = calc_2dsplinecoeffs_c(self._rforceGrid)
        if interpzforce:
            if use_c * ext_loaded:
                self._zforceGrid, err = calc_potential_c(
                    self._origPot, self._rgrid, self._zgrid, zforce=True
                )
            else:
                from ..potential import evaluatezforces

                self._zforceGrid = _grid_eval(
                    evaluatezforces, self._origPot, self._rgrid, self._zgrid
                )
            if self._logR:
                self._zforceInterp = interpolate.RectBivariateSpline(
                    self._logrgrid, self._zgrid, self._zforceGrid, kx=3, ky=3, s=0.0
                )
            else:
                self._zforceInterp = interpolate.RectBivariateSpline(
                    self._rgrid, self._zgrid, self._zforceGrid, kx=3, ky=3, s=0.0
                )
            if enable_c * ext_loaded:
                self._zforceGrid_splinecoeffs = calc_2dsplinecoeffs_c(self._zforceGrid)
        # Interpolated 2nd derivatives: like the forces, each is a grid of
        # exact (original-potential) values interpolated with a 2D cubic
        # spline. The grids are always computed in Python (not with use_c):
        # the C grid-filler aggregators are NULL-safe and would silently
        # return 0 for a potential without that 2nd derivative in C.
        # d2Phi/dR2 and d2Phi/dz2 are even in z for a zsym potential, while
        # d2Phi/dRdz is odd (zero at z=0), so all three can be sampled on the
        # z>=0 grid like the forces.
        if interpR2deriv:
            from ..potential import evaluateR2derivs

            self._r2derivGrid = _grid_eval(
                evaluateR2derivs, self._origPot, self._rgrid, self._zgrid
            )
            if self._logR:
                self._r2derivInterp = interpolate.RectBivariateSpline(
                    self._logrgrid, self._zgrid, self._r2derivGrid, kx=3, ky=3, s=0.0
                )
            else:
                self._r2derivInterp = interpolate.RectBivariateSpline(
                    self._rgrid, self._zgrid, self._r2derivGrid, kx=3, ky=3, s=0.0
                )
            if enable_c * ext_loaded:
                self._r2derivGrid_splinecoeffs = calc_2dsplinecoeffs_c(
                    self._r2derivGrid
                )
        if interpz2deriv:
            from ..potential import evaluatez2derivs

            self._z2derivGrid = _grid_eval(
                evaluatez2derivs, self._origPot, self._rgrid, self._zgrid
            )
            if self._logR:
                self._z2derivInterp = interpolate.RectBivariateSpline(
                    self._logrgrid, self._zgrid, self._z2derivGrid, kx=3, ky=3, s=0.0
                )
            else:
                self._z2derivInterp = interpolate.RectBivariateSpline(
                    self._rgrid, self._zgrid, self._z2derivGrid, kx=3, ky=3, s=0.0
                )
            if enable_c * ext_loaded:
                self._z2derivGrid_splinecoeffs = calc_2dsplinecoeffs_c(
                    self._z2derivGrid
                )
        if interpRzderiv:
            from ..potential import evaluateRzderivs

            self._rzderivGrid = _grid_eval(
                evaluateRzderivs, self._origPot, self._rgrid, self._zgrid
            )
            if self._logR:
                self._rzderivInterp = interpolate.RectBivariateSpline(
                    self._logrgrid, self._zgrid, self._rzderivGrid, kx=3, ky=3, s=0.0
                )
            else:
                self._rzderivInterp = interpolate.RectBivariateSpline(
                    self._rgrid, self._zgrid, self._rzderivGrid, kx=3, ky=3, s=0.0
                )
            if enable_c * ext_loaded:
                self._rzderivGrid_splinecoeffs = calc_2dsplinecoeffs_c(
                    self._rzderivGrid
                )
        if interpDens:
            from ..potential import evaluateDensities

            self._densGrid = _grid_eval(
                evaluateDensities, self._origPot, self._rgrid, self._zgrid
            )
            if self._logR:
                self._densInterp = interpolate.RectBivariateSpline(
                    self._logrgrid,
                    self._zgrid,
                    numpy.log(self._densGrid + 10.0**-10.0),
                    kx=3,
                    ky=3,
                    s=0.0,
                )
            else:
                self._densInterp = interpolate.RectBivariateSpline(
                    self._rgrid,
                    self._zgrid,
                    numpy.log(self._densGrid + 10.0**-10.0),
                    kx=3,
                    ky=3,
                    s=0.0,
                )
        if interpvcirc:
            from ..potential import vcirc

            if not numcores is None:
                self._vcircGrid = multi.parallel_map(
                    (
                        lambda x: vcirc(
                            self._origPot, self._rgrid[x], use_physical=False
                        )
                    ),
                    list(range(len(self._rgrid))),
                    numcores=numcores,
                )
            else:
                self._vcircGrid = numpy.array(
                    [vcirc(self._origPot, r, use_physical=False) for r in self._rgrid]
                )
            if self._logR:
                self._vcircInterp = interpolate.InterpolatedUnivariateSpline(
                    self._logrgrid, self._vcircGrid, k=3
                )
            else:
                self._vcircInterp = interpolate.InterpolatedUnivariateSpline(
                    self._rgrid, self._vcircGrid, k=3
                )
        if interpdvcircdr:
            from ..potential import dvcircdR

            if not numcores is None:
                self._dvcircdrGrid = multi.parallel_map(
                    (
                        lambda x: dvcircdR(
                            self._origPot, self._rgrid[x], use_physical=False
                        )
                    ),
                    list(range(len(self._rgrid))),
                    numcores=numcores,
                )
            else:
                self._dvcircdrGrid = numpy.array(
                    [
                        dvcircdR(self._origPot, r, use_physical=False)
                        for r in self._rgrid
                    ]
                )
            if self._logR:
                self._dvcircdrInterp = interpolate.InterpolatedUnivariateSpline(
                    self._logrgrid, self._dvcircdrGrid, k=3
                )
            else:
                self._dvcircdrInterp = interpolate.InterpolatedUnivariateSpline(
                    self._rgrid, self._dvcircdrGrid, k=3
                )
        if interpepifreq:
            from ..potential import epifreq

            if not numcores is None:
                self._epifreqGrid = numpy.array(
                    multi.parallel_map(
                        (
                            lambda x: epifreq(
                                self._origPot, self._rgrid[x], use_physical=False
                            )
                        ),
                        list(range(len(self._rgrid))),
                        numcores=numcores,
                    )
                )
            else:
                self._epifreqGrid = numpy.array(
                    [epifreq(self._origPot, r, use_physical=False) for r in self._rgrid]
                )
            indx = True ^ numpy.isnan(self._epifreqGrid)
            if numpy.sum(indx) < 4:
                if self._logR:
                    self._epifreqInterp = interpolate.InterpolatedUnivariateSpline(
                        self._logrgrid[indx], self._epifreqGrid[indx], k=1
                    )
                else:
                    self._epifreqInterp = interpolate.InterpolatedUnivariateSpline(
                        self._rgrid[indx], self._epifreqGrid[indx], k=1
                    )
            else:
                if self._logR:
                    self._epifreqInterp = interpolate.InterpolatedUnivariateSpline(
                        self._logrgrid[indx], self._epifreqGrid[indx], k=3
                    )
                else:
                    self._epifreqInterp = interpolate.InterpolatedUnivariateSpline(
                        self._rgrid[indx], self._epifreqGrid[indx], k=3
                    )
        if interpverticalfreq:
            from ..potential import verticalfreq

            if not numcores is None:
                self._verticalfreqGrid = multi.parallel_map(
                    (
                        lambda x: verticalfreq(
                            self._origPot, self._rgrid[x], use_physical=False
                        )
                    ),
                    list(range(len(self._rgrid))),
                    numcores=numcores,
                )
            else:
                self._verticalfreqGrid = numpy.array(
                    [
                        verticalfreq(self._origPot, r, use_physical=False)
                        for r in self._rgrid
                    ]
                )
            if self._logR:
                self._verticalfreqInterp = interpolate.InterpolatedUnivariateSpline(
                    self._logrgrid, self._verticalfreqGrid, k=3
                )
            else:
                self._verticalfreqInterp = interpolate.InterpolatedUnivariateSpline(
                    self._rgrid, self._verticalfreqGrid, k=3
                )
        return None

    def _grid_ppoly(self, which):
        """Lazily build & cache the backend tensor-product PPoly block for the
        interpolated 2D quantity ``which`` (``pot``/``rforce``/``zforce``/
        ``r2deriv``/``z2deriv``/``rzderiv``/``dens``). Built once, on first
        backend use, from the SAME scipy ``RectBivariateSpline`` the numpy path
        uses, so the backend interpolation reuses its knots/coefficients. numpy
        setup is untouched (no extra work for numpy-only users)."""
        attr = "_" + which + "PPoly"
        pp = getattr(self, attr, None)
        if pp is None:
            pp = rect_bivariate_to_ppoly(getattr(self, "_" + which + "Interp"))
            setattr(self, attr, pp)
        return pp

    def _eval_grid_backend(self, which, R, z, *, log_transform=False):
        """Backend (jax/torch) evaluation of an interpolated 2D quantity: the same
        frozen tensor-product spline as the numpy ``.ev`` path, evaluated through
        namespace-agnostic ``eval_rect_ppoly`` (searchsorted + 2D Horner), so the
        value is computed natively and is exactly autodifferentiable w.r.t. (R,z).
        Matches ``RectBivariateSpline.ev`` to ~1 ulp; like scipy's ``.ev`` it
        extrapolates the edge polynomial outside the grid (finite, NaN-free)."""
        xp = get_namespace(R, z)
        xbr, ybr, c = self._grid_ppoly(which)
        Rq = xp.log(R) if self._logR else R
        out = eval_rect_ppoly(xp, xbr, ybr, c, Rq, z, extrapolate=True)
        if log_transform:
            out = xp.exp(out) - 10.0**-10.0
        return match_input_dtype(out, R, z)

    def _grid_ppoly1d(self, which):
        """Lazily build & cache the backend 1D piecewise-power block for the
        interpolated 1D quantity ``which`` (``vcirc``/``dvcircdr``/``epifreq``/
        ``verticalfreq``), converting the SAME fitted scipy
        ``InterpolatedUnivariateSpline`` the numpy path uses (so the backend eval
        reuses its knots/coefficients). Built once, on first backend use; numpy
        setup is untouched."""
        attr = "_" + which + "PPoly1d"
        pp = getattr(self, attr, None)
        if pp is None:
            pp = spline_to_ppoly(getattr(self, "_" + which + "Interp"))
            setattr(self, attr, pp)
        return pp

    def _eval_grid_backend_1d(self, which, R):
        """Backend (jax/torch) evaluation of an interpolated 1D quantity: the same
        frozen spline as the numpy ``InterpolatedUnivariateSpline`` call, through
        namespace-agnostic ``eval_ppoly`` (searchsorted + Horner), so the value is
        native and exactly autodifferentiable w.r.t. R. Matches the scipy spline
        to ~1 ulp; like scipy (ext=0) it extrapolates the edge polynomial outside
        the grid (finite, NaN-free) -- the backend path is on-grid interpolation
        only (the numpy off-grid fallback to the orig potential is numpy-only)."""
        xp = get_namespace(R)
        x, c = self._grid_ppoly1d(which)
        Rq = xp.log(R) if self._logR else R
        out = eval_ppoly(xp, x, c, Rq, extrapolate=True)
        return match_input_dtype(out, R)

    @scalarVectorDecorator
    @zsymDecorator(False)
    def _evaluate(self, R, z, phi=0.0, t=0.0):
        from ..potential import evaluatePotentials

        if self._interpPot:
            if is_backend_array(R) or is_backend_array(z):
                return self._eval_grid_backend("pot", R, z)
            out = numpy.empty(R.shape)
            indx = (
                (R >= self._rgrid[0])
                * (R <= self._rgrid[-1])
                * (z <= self._zgrid[-1])
                * (z >= self._zgrid[0])
            )
            if numpy.sum(indx) > 0:
                if self._enable_c:
                    out[indx] = eval_potential_c(self, R[indx], z[indx])[0] / self._amp
                else:
                    if self._logR:
                        out[indx] = self._potInterp.ev(numpy.log(R[indx]), z[indx])
                    else:
                        out[indx] = self._potInterp.ev(R[indx], z[indx])
            if numpy.sum(True ^ indx) > 0:
                out[True ^ indx] = evaluatePotentials(
                    self._origPot, R[True ^ indx], z[True ^ indx], use_physical=False
                )
            return out
        else:
            return evaluatePotentials(self._origPot, R, z, use_physical=False)

    @scalarVectorDecorator
    @zsymDecorator(False)
    def _Rforce(self, R, z, phi=0.0, t=0.0):
        from ..potential import evaluateRforces

        if self._interpRforce:
            if is_backend_array(R) or is_backend_array(z):
                return self._eval_grid_backend("rforce", R, z)
            out = numpy.empty(R.shape)
            indx = (
                (R >= self._rgrid[0])
                * (R <= self._rgrid[-1])
                * (z <= self._zgrid[-1])
                * (z >= self._zgrid[0])
            )
            if numpy.sum(indx) > 0:
                if self._enable_c:
                    out[indx] = eval_force_c(self, R[indx], z[indx])[0] / self._amp
                else:
                    if self._logR:
                        out[indx] = self._rforceInterp.ev(numpy.log(R[indx]), z[indx])
                    else:
                        out[indx] = self._rforceInterp.ev(R[indx], z[indx])
            if numpy.sum(True ^ indx) > 0:
                out[True ^ indx] = evaluateRforces(
                    self._origPot, R[True ^ indx], z[True ^ indx], use_physical=False
                )
            return out
        else:
            return evaluateRforces(self._origPot, R, z, use_physical=False)

    @scalarVectorDecorator
    @zsymDecorator(True)
    def _zforce(self, R, z, phi=0.0, t=0.0):
        from ..potential import evaluatezforces

        if self._interpzforce:
            if is_backend_array(R) or is_backend_array(z):
                return self._eval_grid_backend("zforce", R, z)
            out = numpy.empty(R.shape)
            indx = (
                (R >= self._rgrid[0])
                * (R <= self._rgrid[-1])
                * (z <= self._zgrid[-1])
                * (z >= self._zgrid[0])
            )
            if numpy.sum(indx) > 0:
                if self._enable_c:
                    out[indx] = (
                        eval_force_c(self, R[indx], z[indx], zforce=True)[0] / self._amp
                    )
                else:
                    if self._logR:
                        out[indx] = self._zforceInterp.ev(numpy.log(R[indx]), z[indx])
                    else:
                        out[indx] = self._zforceInterp.ev(R[indx], z[indx])
            if numpy.sum(True ^ indx) > 0:
                out[True ^ indx] = evaluatezforces(
                    self._origPot, R[True ^ indx], z[True ^ indx], use_physical=False
                )
            return out
        else:
            return evaluatezforces(self._origPot, R, z, use_physical=False)

    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        if not self._interpR2deriv:
            # Not interpolated: pass through to the original potential
            from ..potential import evaluateR2derivs

            return evaluateR2derivs(self._origPot, R, z, use_physical=False)
        return self._R2deriv_interpolated(R, z)

    @scalarVectorDecorator
    @zsymDecorator(False)
    def _R2deriv_interpolated(self, R, z):
        from ..potential import evaluateR2derivs

        if is_backend_array(R) or is_backend_array(z):
            return self._eval_grid_backend("r2deriv", R, z)
        out = numpy.empty(R.shape)
        indx = (
            (R >= self._rgrid[0])
            * (R <= self._rgrid[-1])
            * (z <= self._zgrid[-1])
            * (z >= self._zgrid[0])
        )
        if numpy.sum(indx) > 0:
            if self._enable_c:
                out[indx] = (
                    eval_2ndderiv_c(self, R[indx], z[indx], deriv="r2deriv")[0]
                    / self._amp
                )
            else:
                if self._logR:
                    out[indx] = self._r2derivInterp.ev(numpy.log(R[indx]), z[indx])
                else:
                    out[indx] = self._r2derivInterp.ev(R[indx], z[indx])
        if numpy.sum(True ^ indx) > 0:
            out[True ^ indx] = evaluateR2derivs(
                self._origPot, R[True ^ indx], z[True ^ indx], use_physical=False
            )
        return out

    def _z2deriv(self, R, z, phi=0.0, t=0.0):
        if not self._interpz2deriv:
            # Not interpolated: pass through to the original potential
            from ..potential import evaluatez2derivs

            return evaluatez2derivs(self._origPot, R, z, use_physical=False)
        return self._z2deriv_interpolated(R, z)

    @scalarVectorDecorator
    @zsymDecorator(False)
    def _z2deriv_interpolated(self, R, z):
        from ..potential import evaluatez2derivs

        if is_backend_array(R) or is_backend_array(z):
            return self._eval_grid_backend("z2deriv", R, z)
        out = numpy.empty(R.shape)
        indx = (
            (R >= self._rgrid[0])
            * (R <= self._rgrid[-1])
            * (z <= self._zgrid[-1])
            * (z >= self._zgrid[0])
        )
        if numpy.sum(indx) > 0:
            if self._enable_c:
                out[indx] = (
                    eval_2ndderiv_c(self, R[indx], z[indx], deriv="z2deriv")[0]
                    / self._amp
                )
            else:
                if self._logR:
                    out[indx] = self._z2derivInterp.ev(numpy.log(R[indx]), z[indx])
                else:
                    out[indx] = self._z2derivInterp.ev(R[indx], z[indx])
        if numpy.sum(True ^ indx) > 0:
            out[True ^ indx] = evaluatez2derivs(
                self._origPot, R[True ^ indx], z[True ^ indx], use_physical=False
            )
        return out

    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        if not self._interpRzderiv:
            # Not interpolated: pass through to the original potential
            from ..potential import evaluateRzderivs

            return evaluateRzderivs(self._origPot, R, z, use_physical=False)
        return self._Rzderiv_interpolated(R, z)

    @scalarVectorDecorator
    @zsymDecorator(True)
    def _Rzderiv_interpolated(self, R, z):
        from ..potential import evaluateRzderivs

        if is_backend_array(R) or is_backend_array(z):
            return self._eval_grid_backend("rzderiv", R, z)
        out = numpy.empty(R.shape)
        indx = (
            (R >= self._rgrid[0])
            * (R <= self._rgrid[-1])
            * (z <= self._zgrid[-1])
            * (z >= self._zgrid[0])
        )
        if numpy.sum(indx) > 0:
            if self._enable_c:
                out[indx] = (
                    eval_2ndderiv_c(self, R[indx], z[indx], deriv="rzderiv")[0]
                    / self._amp
                )
            else:
                if self._logR:
                    out[indx] = self._rzderivInterp.ev(numpy.log(R[indx]), z[indx])
                else:
                    out[indx] = self._rzderivInterp.ev(R[indx], z[indx])
        if numpy.sum(True ^ indx) > 0:
            out[True ^ indx] = evaluateRzderivs(
                self._origPot, R[True ^ indx], z[True ^ indx], use_physical=False
            )
        return out

    @scalarVectorDecorator
    @zsymDecorator(False)
    def _dens(self, R, z, phi=0.0, t=0.0):
        from ..potential import evaluateDensities

        if self._interpDens:
            if is_backend_array(R) or is_backend_array(z):
                return self._eval_grid_backend("dens", R, z, log_transform=True)
            out = numpy.empty(R.shape)
            indx = (
                (R >= self._rgrid[0])
                * (R <= self._rgrid[-1])
                * (z <= self._zgrid[-1])
                * (z >= self._zgrid[0])
            )
            if numpy.sum(indx) > 0:
                if self._logR:
                    out[indx] = (
                        numpy.exp(self._densInterp.ev(numpy.log(R[indx]), z[indx]))
                        - 10.0**-10.0
                    )
                else:
                    out[indx] = (
                        numpy.exp(self._densInterp.ev(R[indx], z[indx])) - 10.0**-10.0
                    )
            if numpy.sum(True ^ indx) > 0:
                out[True ^ indx] = evaluateDensities(
                    self._origPot, R[True ^ indx], z[True ^ indx], use_physical=False
                )
            return out
        else:
            return evaluateDensities(self._origPot, R, z, use_physical=False)

    @physical_conversion("velocity", pop=True)
    @scalarDecorator
    def vcirc(self, R):
        from ..potential import vcirc

        if self._interpvcirc:
            if is_backend_array(R):
                return self._eval_grid_backend_1d("vcirc", R)
            indx = (R >= self._rgrid[0]) * (R <= self._rgrid[-1])
            out = numpy.empty(R.shape)
            if numpy.sum(indx) > 0:
                if self._logR:
                    out[indx] = self._vcircInterp(numpy.log(R[indx]))
                else:
                    out[indx] = self._vcircInterp(R[indx])
            if numpy.sum(True ^ indx) > 0:
                out[True ^ indx] = vcirc(
                    self._origPot, R[True ^ indx], use_physical=False
                )
            return out
        else:
            return vcirc(self._origPot, R, use_physical=False)

    @physical_conversion("frequency", pop=True)
    @scalarDecorator
    def dvcircdR(self, R):
        from ..potential import dvcircdR

        if self._interpdvcircdr:
            if is_backend_array(R):
                return self._eval_grid_backend_1d("dvcircdr", R)
            indx = (R >= self._rgrid[0]) * (R <= self._rgrid[-1])
            out = numpy.empty(R.shape)
            if numpy.sum(indx) > 0:
                if self._logR:
                    out[indx] = self._dvcircdrInterp(numpy.log(R[indx]))
                else:
                    out[indx] = self._dvcircdrInterp(R[indx])
            if numpy.sum(True ^ indx) > 0:
                out[True ^ indx] = dvcircdR(
                    self._origPot, R[True ^ indx], use_physical=False
                )
            return out
        else:
            return dvcircdR(self._origPot, R, use_physical=False)

    @physical_conversion("frequency", pop=True)
    @scalarDecorator
    def epifreq(self, R):
        from ..potential import epifreq

        if self._interpepifreq:
            if is_backend_array(R):
                return self._eval_grid_backend_1d("epifreq", R)
            indx = (R >= self._rgrid[0]) * (R <= self._rgrid[-1])
            out = numpy.empty(R.shape)
            if numpy.sum(indx) > 0:
                if self._logR:
                    out[indx] = self._epifreqInterp(numpy.log(R[indx]))
                else:
                    out[indx] = self._epifreqInterp(R[indx])
            if numpy.sum(True ^ indx) > 0:
                out[True ^ indx] = epifreq(
                    self._origPot, R[True ^ indx], use_physical=False
                )
            return out
        else:
            return epifreq(self._origPot, R, use_physical=False)

    @physical_conversion("frequency", pop=True)
    @scalarDecorator
    def verticalfreq(self, R):
        from ..potential import verticalfreq

        if self._interpverticalfreq:
            if is_backend_array(R):
                return self._eval_grid_backend_1d("verticalfreq", R)
            indx = (R >= self._rgrid[0]) * (R <= self._rgrid[-1])
            out = numpy.empty(R.shape)
            if numpy.sum(indx) > 0:
                if self._logR:
                    out[indx] = self._verticalfreqInterp(numpy.log(R[indx]))
                else:
                    out[indx] = self._verticalfreqInterp(R[indx])
            if numpy.sum(True ^ indx) > 0:
                out[True ^ indx] = verticalfreq(
                    self._origPot, R[True ^ indx], use_physical=False
                )
            return out
        else:
            return verticalfreq(self._origPot, R, use_physical=False)


def calc_potential_c(pot, R, z, rforce=False, zforce=False):
    """
    Calculate the potential on a grid.

    Parameters
    ----------
    pot : Potential or a combined potential formed using addition (pot1+pot2+…)
        Potential object(s) to calculate the potential from.
    R : numpy.ndarray
        Grid in R.
    z : numpy.ndarray
        Grid in z.
    rforce : bool, optional
        If True, calculate the radial force instead. Default is False.
    zforce : bool, optional
        If True, calculate the vertical force instead. Default is False.

    Returns
    -------
    numpy.ndarray
        Potential on the grid (2D array).

    Notes
    -----
    - 2013-01-24 - Written - Bovy (IAS)
    - 2013-01-29 - Added forces - Bovy (IAS)

    """
    from ..orbit.integrateFullOrbit import (  # here bc otherwise there is an infinite loop
        _parse_pot,
    )
    from ..orbit.integratePlanarOrbit import _prep_tfuncs

    # Parse the potential
    npot, pot_type, pot_args, pot_tfuncs = _parse_pot(pot)
    pot_tfuncs = _prep_tfuncs(pot_tfuncs)

    # Set up result arrays
    out = numpy.empty((len(R), len(z)))
    err = ctypes.c_int(0)

    # Set up the C code
    ndarrayFlags = ("C_CONTIGUOUS", "WRITEABLE")
    if rforce:
        interppotential_calc_potentialFunc = _lib.calc_rforce
    elif zforce:
        interppotential_calc_potentialFunc = _lib.calc_zforce
    else:
        interppotential_calc_potentialFunc = _lib.calc_potential
    interppotential_calc_potentialFunc.argtypes = [
        ctypes.c_int,
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_int,
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_int,
        ndpointer(dtype=numpy.int32, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_void_p,
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.POINTER(ctypes.c_int),
    ]

    # Array requirements, first store old order
    f_cont = [R.flags["F_CONTIGUOUS"], z.flags["F_CONTIGUOUS"]]
    R = numpy.require(R, dtype=numpy.float64, requirements=["C", "W"])
    z = numpy.require(z, dtype=numpy.float64, requirements=["C", "W"])
    out = numpy.require(out, dtype=numpy.float64, requirements=["C", "W"])

    # Run the C code
    interppotential_calc_potentialFunc(
        len(R),
        R,
        len(z),
        z,
        ctypes.c_int(npot),
        pot_type,
        pot_args,
        pot_tfuncs,
        out,
        ctypes.byref(err),
    )

    # Reset input arrays
    if f_cont[0]:
        R = numpy.asfortranarray(R)
    if f_cont[1]:
        z = numpy.asfortranarray(z)

    return (out, err.value)


def calc_2dsplinecoeffs_c(array2d):
    """
    Calculate spline coefficients for a 2D array.

    Parameters
    ----------
    array2d : numpy.ndarray
        2D array to calculate spline coefficients for.

    Returns
    -------
    ndarray
        New array with spline coefficients.

    Notes
    -----
    - 2013-01-24 - Written - Bovy (IAS)
    """
    # Set up result arrays
    out = copy.copy(array2d)
    out = numpy.require(out, dtype=numpy.float64, requirements=["C", "W"])

    # Set up the C code
    ndarrayFlags = ("C_CONTIGUOUS", "WRITEABLE")
    interppotential_calc_2dsplinecoeffs = _lib.samples_to_coefficients
    interppotential_calc_2dsplinecoeffs.argtypes = [
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_int,
        ctypes.c_int,
    ]

    # Run the C code
    interppotential_calc_2dsplinecoeffs(out, out.shape[1], out.shape[0])

    return out


def eval_potential_c(pot, R, z):
    """
    Use C to evaluate the interpolated potential.

    Parameters
    ----------
    pot : Potential or a combined potential formed using addition (pot1+pot2+…)
        The potential
    R : numpy.ndarray
        Galactocentric cylindrical radius.
    z : numpy.ndarray
        Galactocentric height.

    Returns
    -------
    numpy.ndarray
        Potential evaluated at R and z.

    Notes
    -----
    - 2013-01-24: Written - Bovy (IAS)
    """
    from ..orbit.integrateFullOrbit import (  # here bc otherwise there is an infinite loop
        _parse_pot,
    )
    from ..orbit.integratePlanarOrbit import _prep_tfuncs

    # Parse the potential
    npot, pot_type, pot_args, pot_tfuncs = _parse_pot(pot, potforactions=True)
    pot_tfuncs = _prep_tfuncs(pot_tfuncs)

    # Set up result arrays
    out = numpy.empty(len(R))
    err = ctypes.c_int(0)

    # Set up the C code
    ndarrayFlags = ("C_CONTIGUOUS", "WRITEABLE")
    interppotential_calc_potentialFunc = _lib.eval_potential
    interppotential_calc_potentialFunc.argtypes = [
        ctypes.c_int,
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_int,
        ndpointer(dtype=numpy.int32, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_void_p,
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.POINTER(ctypes.c_int),
    ]

    # Array requirements, first store old order
    f_cont = [R.flags["F_CONTIGUOUS"], z.flags["F_CONTIGUOUS"]]
    R = numpy.require(R, dtype=numpy.float64, requirements=["C", "W"])
    z = numpy.require(z, dtype=numpy.float64, requirements=["C", "W"])
    out = numpy.require(out, dtype=numpy.float64, requirements=["C", "W"])

    # Run the C code
    interppotential_calc_potentialFunc(
        len(R),
        R,
        z,
        ctypes.c_int(npot),
        pot_type,
        pot_args,
        pot_tfuncs,
        out,
        ctypes.byref(err),
    )

    # Reset input arrays
    if f_cont[0]:
        R = numpy.asfortranarray(R)
    if f_cont[1]:
        z = numpy.asfortranarray(z)

    return (out, err.value)


def eval_force_c(pot, R, z, zforce=False):
    """
    Use C to evaluate the interpolated potential's forces

    Parameters
    ----------
    pot : Potential or a combined potential formed using addition (pot1+pot2+…)
        The potential
    R : numpy.ndarray
        Galactocentric cylindrical radius.
    z : numpy.ndarray
        Galactocentric height.
    zforce : bool, optional
        If True, return the vertical force, otherwise return the radial force. Default is False.

    Returns
    -------
    numpy.ndarray
        Force evaluated at R and z.

    Notes
    -----
    - 2013-01-29: Written - Bovy (IAS)

    """
    from ..orbit.integrateFullOrbit import (  # here bc otherwise there is an infinite loop
        _parse_pot,
    )
    from ..orbit.integratePlanarOrbit import _prep_tfuncs

    # Parse the potential
    npot, pot_type, pot_args, pot_tfuncs = _parse_pot(pot)
    pot_tfuncs = _prep_tfuncs(pot_tfuncs)

    # Set up result arrays
    out = numpy.empty(len(R))
    err = ctypes.c_int(0)

    # Set up the C code
    ndarrayFlags = ("C_CONTIGUOUS", "WRITEABLE")
    if zforce:
        interppotential_calc_forceFunc = _lib.eval_zforce
    else:
        interppotential_calc_forceFunc = _lib.eval_rforce
    interppotential_calc_forceFunc.argtypes = [
        ctypes.c_int,
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_int,
        ndpointer(dtype=numpy.int32, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_void_p,
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.POINTER(ctypes.c_int),
    ]

    # Array requirements, first store old order
    f_cont = [R.flags["F_CONTIGUOUS"], z.flags["F_CONTIGUOUS"]]
    R = numpy.require(R, dtype=numpy.float64, requirements=["C", "W"])
    z = numpy.require(z, dtype=numpy.float64, requirements=["C", "W"])
    out = numpy.require(out, dtype=numpy.float64, requirements=["C", "W"])

    # Run the C code
    interppotential_calc_forceFunc(
        len(R),
        R,
        z,
        ctypes.c_int(npot),
        pot_type,
        pot_args,
        pot_tfuncs,
        out,
        ctypes.byref(err),
    )

    # Reset input arrays
    if f_cont[0]:
        R = numpy.asfortranarray(R)
    if f_cont[1]:
        z = numpy.asfortranarray(z)

    return (out, err.value)


def eval_2ndderiv_c(pot, R, z, deriv="r2deriv"):
    """
    Use C to evaluate the interpolated potential's second derivatives.

    Parameters
    ----------
    pot : Potential or a combined potential formed using addition (pot1+pot2+…)
        The potential
    R : numpy.ndarray
        Galactocentric cylindrical radius.
    z : numpy.ndarray
        Galactocentric height.
    deriv : str, optional
        Which second derivative to evaluate: 'r2deriv' (default), 'z2deriv',
        or 'rzderiv'.

    Returns
    -------
    tuple
        (Second derivative evaluated at R and z, error code).

    Notes
    -----
    - 2026-06-09: Written - Bovy (UofT)

    """
    from ..orbit.integrateFullOrbit import (  # here bc otherwise there is an infinite loop
        _parse_pot,
    )
    from ..orbit.integratePlanarOrbit import _prep_tfuncs

    # Parse the potential
    npot, pot_type, pot_args, pot_tfuncs = _parse_pot(pot)
    pot_tfuncs = _prep_tfuncs(pot_tfuncs)

    # Set up result arrays
    out = numpy.empty(len(R))
    err = ctypes.c_int(0)

    # Set up the C code
    ndarrayFlags = ("C_CONTIGUOUS", "WRITEABLE")
    if deriv.lower() == "z2deriv":
        interppotential_calc_2ndderivFunc = _lib.eval_z2deriv
    elif deriv.lower() == "rzderiv":
        interppotential_calc_2ndderivFunc = _lib.eval_rzderiv
    else:
        interppotential_calc_2ndderivFunc = _lib.eval_r2deriv
    interppotential_calc_2ndderivFunc.argtypes = [
        ctypes.c_int,
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_int,
        ndpointer(dtype=numpy.int32, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_void_p,
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.POINTER(ctypes.c_int),
    ]

    # Array requirements, first store old order
    f_cont = [R.flags["F_CONTIGUOUS"], z.flags["F_CONTIGUOUS"]]
    R = numpy.require(R, dtype=numpy.float64, requirements=["C", "W"])
    z = numpy.require(z, dtype=numpy.float64, requirements=["C", "W"])
    out = numpy.require(out, dtype=numpy.float64, requirements=["C", "W"])

    # Run the C code
    interppotential_calc_2ndderivFunc(
        len(R),
        R,
        z,
        ctypes.c_int(npot),
        pot_type,
        pot_args,
        pot_tfuncs,
        out,
        ctypes.byref(err),
    )

    # Reset input arrays
    if f_cont[0]:
        R = numpy.asfortranarray(R)
    if f_cont[1]:
        z = numpy.asfortranarray(z)

    return (out, err.value)


def sign(x):
    out = numpy.ones_like(x)
    out[(x < 0.0)] = -1.0
    return out
