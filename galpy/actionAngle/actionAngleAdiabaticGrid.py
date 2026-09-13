###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleAdiabaticGrid
#
#             build grid in integrals of motion to quickly evaluate
#             actionAngleAdiabatic
#
#      methods:
#             __call__: returns (jr,lz,jz)
#
###############################################################################
import numpy
from scipy import interpolate

from .. import potential
from ..backend import (
    as_numpy,
    asarray_on_device,
    device_of,
    get_namespace,
)
from ..backend import interpolate as backend_interpolate
from ..backend import promote_scalars, set_at, use
from ..backend._namespaces import under_trace
from ..potential.Potential import (
    _check_potential_list_and_deprecate,
    _evaluatePotentials,
)
from ..util import multi
from .actionAngle import UnboundError, actionAngle
from .actionAngleAdiabatic import actionAngleAdiabatic

_PRINTOUTSIDEGRID = False


class actionAngleAdiabaticGrid(actionAngle):
    """Action-angle formalism for axisymmetric potentials using the adiabatic approximation, grid-based interpolation

    jax/torch input is supported and differentiable, but evaluates the
    interpolation grid only (no off-grid fallback): inputs must lie within the
    grid, whereas the numpy path falls back to a per-point solve off-grid."""

    def __init__(
        self,
        pot=None,
        zmax=1.0,
        gamma=1.0,
        Rmax=5.0,
        nR=16,
        nEz=16,
        nEr=31,
        nLz=31,
        numcores=1,
        **kwargs,
    ):
        """
        Initialize an actionAngleAdiabaticGrid object

        Parameters
        ----------
        pot : Potential or a combined potential formed using addition (pot1+pot2+…)
            The potential or a combined potential formed using addition (pot1+pot2+…).
        zmax : float
            Maximum height to which to calculate Ez
        gamma : float
            Replace Lz by Lz+gamma Jz in effective potential
        Rmax : float
            Maximum radius to which to calculate Er
        nR : int
            Number of radii to use in the grid
        nEz : int
            Number of Ez values to use in the grid
        nEr : int
            Number of Er values to use in the grid
        nLz : int
            Number of Lz values to use in the grid
        numcores : int
            Number of cores to use for multi-processing
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2012-07-27 - Written - Bovy (IAS@MPIA)
        """
        actionAngle.__init__(self, ro=kwargs.get("ro", None), vo=kwargs.get("vo", None))
        if pot is None:  # pragma: no cover
            raise OSError("Must specify pot= for actionAngleAdiabaticGrid")
        self._c = kwargs.pop("c", False)
        self._gamma = gamma
        self._pot = _check_potential_list_and_deprecate(pot)
        self._zmax = zmax
        self._Rmax = Rmax
        self._Rmin = 0.01
        # Set up the actionAngleAdiabatic object that we will use to interpolate
        self._aA = actionAngleAdiabatic(pot=self._pot, gamma=self._gamma, c=self._c)
        xp = get_namespace()
        if xp is not numpy:
            # Forced/default backend: build the whole grid ON the backend so the
            # frozen tables are backend arrays (GPU-resident) fit NATIVELY, and the
            # build differentiates through to the query. numpy body below is left
            # byte-identical (scipy).
            self._build_grid_backend(xp, nR, nEz, nEr, nLz, numcores, **kwargs)
            self._check_consistent_units()
            return None
        # Build grid for Ez, first calculate Ez(zmax;R) function
        self._Rs = numpy.linspace(self._Rmin, self._Rmax, nR)
        self._EzZmaxs = _evaluatePotentials(
            self._pot, self._Rs, self._zmax * numpy.ones(nR)
        ) - _evaluatePotentials(self._pot, self._Rs, numpy.zeros(nR))
        self._EzZmaxsInterp = interpolate.InterpolatedUnivariateSpline(
            self._Rs, numpy.log(self._EzZmaxs), k=3
        )
        y = numpy.linspace(0.0, 1.0, nEz)
        jz = numpy.zeros((nR, nEz))
        jzEzzmax = numpy.zeros(nR)
        thisRs = (numpy.tile(self._Rs, (nEz, 1)).T).flatten()
        thisEzZmaxs = (numpy.tile(self._EzZmaxs, (nEz, 1)).T).flatten()
        this = (numpy.tile(y, (nR, 1))).flatten()
        if self._c:
            jz = self._aA(
                thisRs,
                numpy.zeros(len(thisRs)),
                numpy.ones(len(thisRs)),  # these two r dummies
                numpy.zeros(len(thisRs)),
                numpy.sqrt(2.0 * this * thisEzZmaxs),
                **kwargs,
            )[2]
            # the c=True vectorized aA (no _justjz) returns a backend array under a
            # forced backend; the grid precompute must be a WRITABLE numpy array
            # (it feeds scipy splines + in-place table fills like jz[ii,:]/=...).
            # as_numpy of an immutable backend array is read-only -> numpy.array
            # copies it writable. numpy path: numpy.array(numpy) is a plain copy.
            jz = numpy.array(as_numpy(jz))
            jz = numpy.reshape(jz, (nR, nEz))
            jzEzzmax[0:nR] = jz[:, nEz - 1]
        else:
            if numcores > 1:
                jz = multi.parallel_map(
                    (
                        lambda x: self._aA(
                            thisRs[x],
                            0.0,
                            1.0,  # these two r dummies
                            0.0,
                            numpy.sqrt(2.0 * this[x] * thisEzZmaxs[x]),
                            _justjz=True,
                            **kwargs,
                        )[2]
                    ),
                    range(nR * nEz),
                    numcores=numcores,
                )
                jz = numpy.reshape(jz, (nR, nEz))
                jzEzzmax[0:nR] = jz[:, nEz - 1]
            else:
                for ii in range(nR):
                    for jj in range(nEz):
                        # Calculate Jz
                        jz[ii, jj] = self._aA(
                            self._Rs[ii],
                            0.0,
                            1.0,  # these two r dummies
                            0.0,
                            numpy.sqrt(2.0 * y[jj] * self._EzZmaxs[ii]),
                            _justjz=True,
                            **kwargs,
                        )[2][0]
                        if jj == nEz - 1:
                            jzEzzmax[ii] = jz[ii, jj]
        for ii in range(nR):
            jz[ii, :] /= jzEzzmax[ii]
        # First interpolate Ez=Ezmax
        self._jzEzmaxInterp = interpolate.InterpolatedUnivariateSpline(
            self._Rs, numpy.log(jzEzzmax + 10.0**-5.0), k=3
        )
        self._jz = jz
        self._jzInterp = interpolate.RectBivariateSpline(
            self._Rs, y, jz, kx=3, ky=3, s=0.0
        )
        # JR grid
        self._Lzmin = 0.01
        self._Lzs = numpy.linspace(
            self._Lzmin, self._Rmax * potential.vcirc(self._pot, self._Rmax), nLz
        )
        self._Lzmax = self._Lzs[-1]
        # Calculate ER(vr=0,R=RL)
        self._RL = numpy.array([potential.rl(self._pot, l) for l in self._Lzs])
        self._RLInterp = interpolate.InterpolatedUnivariateSpline(
            self._Lzs, self._RL, k=3
        )
        self._ERRL = (
            _evaluatePotentials(self._pot, self._RL, numpy.zeros(nLz))
            + self._Lzs**2.0 / 2.0 / self._RL**2.0
        )
        self._ERRLmax = numpy.amax(self._ERRL) + 1.0
        self._ERRLInterp = interpolate.InterpolatedUnivariateSpline(
            self._Lzs, numpy.log(-(self._ERRL - self._ERRLmax)), k=3
        )
        self._Ramax = 99.0
        self._ERRa = (
            _evaluatePotentials(self._pot, self._Ramax, 0.0)
            + self._Lzs**2.0 / 2.0 / self._Ramax**2.0
        )
        self._ERRamax = numpy.amax(self._ERRa) + 1.0
        self._ERRaInterp = interpolate.InterpolatedUnivariateSpline(
            self._Lzs, numpy.log(-(self._ERRa - self._ERRamax)), k=3
        )
        y = numpy.linspace(0.0, 1.0, nEr)
        jr = numpy.zeros((nLz, nEr))
        jrERRa = numpy.zeros(nLz)
        thisRL = (numpy.tile(self._RL, (nEr - 1, 1)).T).flatten()
        thisLzs = (numpy.tile(self._Lzs, (nEr - 1, 1)).T).flatten()
        thisERRL = (numpy.tile(self._ERRL, (nEr - 1, 1)).T).flatten()
        thisERRa = (numpy.tile(self._ERRa, (nEr - 1, 1)).T).flatten()
        this = (numpy.tile(y[0:-1], (nLz, 1))).flatten()
        if self._c:
            mjr = self._aA(
                thisRL,
                numpy.sqrt(
                    2.0
                    * (
                        thisERRa
                        + this * (thisERRL - thisERRa)
                        - _evaluatePotentials(
                            self._pot, thisRL, numpy.zeros((nEr - 1) * nLz)
                        )
                    )
                    - thisLzs**2.0 / thisRL**2.0
                ),
                thisLzs / thisRL,
                numpy.zeros(len(thisRL)),
                numpy.zeros(len(thisRL)),
                **kwargs,
            )[0]
            mjr = numpy.array(as_numpy(mjr))  # writable numpy precompute; see above
            jr[:, 0:-1] = numpy.reshape(mjr, (nLz, nEr - 1))
            jrERRa[0:nLz] = jr[:, 0]
        else:
            if numcores > 1:
                mjr = multi.parallel_map(
                    (
                        lambda x: self._aA(
                            thisRL[x],
                            numpy.sqrt(
                                2.0
                                * (
                                    thisERRa[x]
                                    + this[x] * (thisERRL[x] - thisERRa[x])
                                    - _evaluatePotentials(self._pot, thisRL[x], 0.0)
                                )
                                - thisLzs[x] ** 2.0 / thisRL[x] ** 2.0
                            ),
                            thisLzs[x] / thisRL[x],
                            0.0,
                            0.0,
                            _justjr=True,
                            **kwargs,
                        )[0]
                    ),
                    range((nEr - 1) * nLz),
                    numcores=numcores,
                )
                jr[:, 0:-1] = numpy.reshape(mjr, (nLz, nEr - 1))
                jrERRa[0:nLz] = jr[:, 0]
            else:
                for ii in range(nLz):
                    for jj in range(nEr - 1):  # Last one is zero by construction
                        try:
                            jr[ii, jj] = self._aA(
                                self._RL[ii],
                                numpy.sqrt(
                                    2.0
                                    * (
                                        self._ERRa[ii]
                                        + y[jj] * (self._ERRL[ii] - self._ERRa[ii])
                                        - _evaluatePotentials(
                                            self._pot, self._RL[ii], 0.0
                                        )
                                    )
                                    - self._Lzs[ii] ** 2.0 / self._RL[ii] ** 2.0
                                ),
                                self._Lzs[ii] / self._RL[ii],
                                0.0,
                                0.0,
                                _justjr=True,
                                **kwargs,
                            )[0][0]
                        except UnboundError:  # pragma: no cover
                            raise
                        if jj == 0:
                            jrERRa[ii] = jr[ii, jj]
        for ii in range(nLz):
            jr[ii, :] /= jrERRa[ii]
        # First interpolate Ez=Ezmax
        self._jr = jr
        self._jrERRaInterp = interpolate.InterpolatedUnivariateSpline(
            self._Lzs, numpy.log(jrERRa + 10.0**-5.0), k=3
        )
        self._jrInterp = interpolate.RectBivariateSpline(
            self._Lzs, y, jr, kx=3, ky=3, s=0.0
        )
        # Backend-agnostic eval wrappers from the SAME fitted scipy objects
        # (numpy path is byte-identical -- the wrappers delegate to scipy).
        self._build_backend_interp(jzEzzmax, jrERRa)
        # Check the units
        self._check_consistent_units()
        return None

    def _build_backend_interp(self, jzEzzmax, jrERRa):
        """Build the jax/torch eval wrappers from the RAW frozen tables (dual-path).

        On the numpy path the tables are numpy, so the Spline1D/Spline2D wrappers
        fit scipy internally (byte-identical to the scipy interpolators the numpy
        ``_evaluate`` uses); on a forced backend the tables are backend arrays, so
        the wrappers fit NATIVELY (Spline2D not-a-knot cubic) and stay backend
        arrays, differentiable through to the query. ``bc='not-a-knot'`` matches
        FITPACK's s=0 interpolating cubic on the mode-2 path (ignored on numpy).
        """
        xp = get_namespace(self._jz)
        Rs = as_numpy(self._Rs)
        Lzs = as_numpy(self._Lzs)
        y_ez = numpy.linspace(0.0, 1.0, self._jz.shape[1])
        y_er = numpy.linspace(0.0, 1.0, self._jr.shape[1])
        # jz / jr tables are fit directly (NOT logged).
        self._jzInterp_b = backend_interpolate.Spline2D(Rs, y_ez, self._jz)
        self._jrInterp_b = backend_interpolate.Spline2D(Lzs, y_er, self._jr)
        self._EzZmaxsInterp_b = backend_interpolate.Spline1D(
            Rs, xp.log(self._EzZmaxs), k=3, bc="not-a-knot"
        )
        self._jzEzmaxInterp_b = backend_interpolate.Spline1D(
            Rs, xp.log(jzEzzmax + 10.0**-5.0), k=3, bc="not-a-knot"
        )
        self._RLInterp_b = backend_interpolate.Spline1D(
            Lzs, self._RL, k=3, bc="not-a-knot"
        )
        self._ERRLInterp_b = backend_interpolate.Spline1D(
            Lzs, xp.log(-(self._ERRL - self._ERRLmax)), k=3, bc="not-a-knot"
        )
        self._ERRaInterp_b = backend_interpolate.Spline1D(
            Lzs, xp.log(-(self._ERRa - self._ERRamax)), k=3, bc="not-a-knot"
        )
        self._jrERRaInterp_b = backend_interpolate.Spline1D(
            Lzs, xp.log(jrERRa + 10.0**-5.0), k=3, bc="not-a-knot"
        )
        return None

    def _build_grid_backend(self, xp, nR, nEz, nEr, nLz, numcores, **kwargs):
        """Backend (jax/torch) counterpart of the numpy grid build.

        Produces the SAME frozen-table attributes as the numpy path
        (``_Rs, _EzZmaxs, _jz, _Lzs, _RL, _ERRL, _ERRa, _jr`` + the derived
        maxima) as BACKEND arrays fit NATIVELY, so the tables are GPU-resident
        and the build differentiates through to the query. The non-differentiable
        per-orbit solvers (``rl``, the c=False ``actionAngleAdiabatic`` loop) run
        as on numpy; the vectorised c=True action solve returns backend arrays
        directly. Values match the numpy(scipy) grid to grid-parity tolerance.
        """
        # --- Ez grid ---
        self._Rs = asarray_on_device(
            xp, numpy.linspace(self._Rmin, self._Rmax, nR), None
        )
        self._EzZmaxs = _evaluatePotentials(
            self._pot, self._Rs, self._zmax * xp.ones(nR)
        ) - _evaluatePotentials(self._pot, self._Rs, xp.zeros(nR))
        dev = device_of(self._EzZmaxs)
        y = asarray_on_device(xp, numpy.linspace(0.0, 1.0, nEz), dev)
        thisRs = _tileT_flat(xp, self._Rs, nEz)
        thisEzZmaxs = _tileT_flat(xp, self._EzZmaxs, nEz)
        this = xp.reshape(xp.tile(y, (nR, 1)), (-1,))
        if self._c:
            jz = self._aA(
                thisRs,
                xp.zeros(nR * nEz),
                xp.ones(nR * nEz),  # these two r dummies
                xp.zeros(nR * nEz),
                xp.sqrt(2.0 * this * thisEzZmaxs),
                **kwargs,
            )[2]
            jz = xp.reshape(jz, (nR, nEz))
        else:
            # c=False actionAngleAdiabatic per-point loop: run as numpy, bring back.
            with use("numpy", force=True):
                Rs_np = as_numpy(self._Rs)
                Ez_np = as_numpy(self._EzZmaxs)
                y_np = as_numpy(y)
                jz_np = numpy.zeros((nR, nEz))
                for ii in range(nR):
                    for jj in range(nEz):
                        jz_np[ii, jj] = self._aA(
                            Rs_np[ii],
                            0.0,
                            1.0,
                            0.0,
                            numpy.sqrt(2.0 * y_np[jj] * Ez_np[ii]),
                            _justjz=True,
                            **kwargs,
                        )[2][0]
            jz = asarray_on_device(xp, jz_np, dev)
        jzEzzmax = jz[:, nEz - 1]
        jz = jz / jzEzzmax[:, None]
        self._jz = jz
        # --- JR grid ---
        self._Lzmin = 0.01
        vc = potential.vcirc(self._pot, self._Rmax)
        frac = asarray_on_device(xp, numpy.linspace(0.0, 1.0, nLz), dev)
        self._Lzs = self._Lzmin + frac * (self._Rmax * vc - self._Lzmin)
        self._Lzmax = self._Lzs[-1]
        with use("numpy", force=True):
            RL_np = numpy.array(
                [potential.rl(self._pot, as_numpy(l)) for l in self._Lzs]
            )
        self._RL = asarray_on_device(xp, RL_np, dev)
        self._ERRL = (
            _evaluatePotentials(self._pot, self._RL, xp.zeros(nLz))
            + self._Lzs**2.0 / 2.0 / self._RL**2.0
        )
        self._ERRLmax = xp.max(self._ERRL) + 1.0
        self._Ramax = 99.0
        self._ERRa = (
            _evaluatePotentials(self._pot, self._Ramax, 0.0)
            + self._Lzs**2.0 / 2.0 / self._Ramax**2.0
        )
        self._ERRamax = xp.max(self._ERRa) + 1.0
        y = asarray_on_device(xp, numpy.linspace(0.0, 1.0, nEr), dev)
        thisRL = _tileT_flat(xp, self._RL, nEr - 1)
        thisLzs = _tileT_flat(xp, self._Lzs, nEr - 1)
        thisERRL = _tileT_flat(xp, self._ERRL, nEr - 1)
        thisERRa = _tileT_flat(xp, self._ERRa, nEr - 1)
        this = xp.reshape(xp.tile(y[0:-1], (nLz, 1)), (-1,))
        if self._c:
            mjr = self._aA(
                thisRL,
                xp.sqrt(
                    2.0
                    * (
                        thisERRa
                        + this * (thisERRL - thisERRa)
                        - _evaluatePotentials(
                            self._pot, thisRL, xp.zeros((nEr - 1) * nLz)
                        )
                    )
                    - thisLzs**2.0 / thisRL**2.0
                ),
                thisLzs / thisRL,
                xp.zeros((nEr - 1) * nLz),
                xp.zeros((nEr - 1) * nLz),
                **kwargs,
            )[0]
            mjr = xp.reshape(mjr, (nLz, nEr - 1))
        else:
            with use("numpy", force=True):
                RL_np = as_numpy(self._RL)
                Lzs_np = as_numpy(self._Lzs)
                ERRL_np = as_numpy(self._ERRL)
                ERRa_np = as_numpy(self._ERRa)
                y_np = as_numpy(y)
                mjr_np = numpy.zeros((nLz, nEr - 1))
                for ii in range(nLz):
                    for jj in range(nEr - 1):
                        mjr_np[ii, jj] = self._aA(
                            RL_np[ii],
                            numpy.sqrt(
                                2.0
                                * (
                                    ERRa_np[ii]
                                    + y_np[jj] * (ERRL_np[ii] - ERRa_np[ii])
                                    - _evaluatePotentials(self._pot, RL_np[ii], 0.0)
                                )
                                - Lzs_np[ii] ** 2.0 / RL_np[ii] ** 2.0
                            ),
                            Lzs_np[ii] / RL_np[ii],
                            0.0,
                            0.0,
                            _justjr=True,
                            **kwargs,
                        )[0][0]
            mjr = asarray_on_device(xp, mjr_np, dev)
        # last Er column is zero by construction
        jr = xp.concat([mjr, xp.zeros((nLz, 1))], axis=1)
        jrERRa = jr[:, 0]
        jr = jr / jrERRa[:, None]
        self._jr = jr
        # native fits + backend eval wrappers
        self._build_backend_interp(jzEzzmax, jrERRa)
        return None

    def _evaluate(self, *args, **kwargs):
        """
        Evaluate the actions (jr,lz,jz).

        Parameters
        ----------
        *args : tuple
            Either:
            a) R,vR,vT,z,vz[,phi]:
                1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)
                2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
            b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
        **kwargs: dict, optional
            scipy.integrate.quadrature keywords (used when directly evaluating a point off the grid)

        Returns
        -------
        tuple
            (jr,lz,jz)

        Notes
        -----
        - 2012-07-27 - Written - Bovy (IAS@MPIA)
        """
        if len(args) == 5:  # R,vR.vT, z, vz
            R, vR, vT, z, vz = args
        elif len(args) == 6:  # R,vR.vT, z, vz, phi
            R, vR, vT, z, vz, phi = args
        else:
            self._parse_eval_args(*args)
            R = self._eval_R
            vR = self._eval_vR
            vT = self._eval_vT
            z = self._eval_z
            vz = self._eval_vz
        xp = get_namespace(R, vR, vT, z, vz)
        if xp is not numpy:  # jax/torch: vectorised, differentiable grid eval
            R, vR, vT, z, vz = promote_scalars(xp, R, vR, vT, z, vz)
            return self._evaluate_backend(R, vR, vT, z, vz)
        # First work on the vertical action
        Phi = _evaluatePotentials(self._pot, R, z)
        try:
            Phio = _evaluatePotentials(self._pot, R, numpy.zeros(len(R)))
        except TypeError:
            Phio = _evaluatePotentials(self._pot, R, 0.0)
        Ez = Phi - Phio + vz**2.0 / 2.0
        # Bigger than Ezzmax?
        thisEzZmax = numpy.exp(self._EzZmaxsInterp(R))
        if isinstance(R, numpy.ndarray):
            indx = R > self._Rmax
            indx += R < self._Rmin
            indx += (Ez != 0.0) * (numpy.log(Ez) > thisEzZmax)
            indxc = True ^ indx
            jz = numpy.empty(R.shape)
            if numpy.sum(indxc) > 0:
                jz[indxc] = self._jzInterp.ev(
                    R[indxc], Ez[indxc] / thisEzZmax[indxc]
                ) * (numpy.exp(self._jzEzmaxInterp(R[indxc])) - 10.0**-5.0)
            if numpy.sum(indx) > 0:
                jz[indx] = self._aA(
                    R[indx],
                    numpy.zeros(numpy.sum(indx)),
                    numpy.ones(numpy.sum(indx)),  # these two r dummies
                    numpy.zeros(numpy.sum(indx)),
                    numpy.sqrt(2.0 * Ez[indx]),
                    _justjz=True,
                    **kwargs,
                )[2]
        else:
            if (
                R > self._Rmax
                or R < self._Rmin
                or (Ez != 0 and numpy.log(Ez) > thisEzZmax)
            ):  # Outside of the grid
                if _PRINTOUTSIDEGRID:  # pragma: no cover
                    print(
                        "Outside of grid in Ez",
                        R > self._Rmax,
                        R < self._Rmin,
                        (Ez != 0 and numpy.log(Ez) > thisEzZmax),
                    )
                jz = self._aA(
                    R,
                    0.0,
                    1.0,  # these two r dummies
                    0.0,
                    numpy.sqrt(2.0 * Ez),
                    _justjz=True,
                    **kwargs,
                )[2]
            else:
                jz = (
                    self._jzInterp(R, Ez / thisEzZmax)
                    * (numpy.exp(self._jzEzmaxInterp(R)) - 10.0**-5.0)
                )[0][0]
        # Radial action
        ERLz = numpy.fabs(R * vT) + self._gamma * jz
        ER = Phio + vR**2.0 / 2.0 + ERLz**2.0 / 2.0 / R**2.0
        thisRL = self._RLInterp(ERLz)
        thisERRL = -numpy.exp(self._ERRLInterp(ERLz)) + self._ERRLmax
        thisERRa = -numpy.exp(self._ERRaInterp(ERLz)) + self._ERRamax
        if isinstance(R, numpy.ndarray):
            indx = ((ER - thisERRa) / (thisERRL - thisERRa) > 1.0) * (
                ((ER - thisERRa) / (thisERRL - thisERRa) - 1.0) < 10.0**-2.0
            )
            ER[indx] = thisERRL[indx]
            indx = ((ER - thisERRa) / (thisERRL - thisERRa) < 0.0) * (
                (ER - thisERRa) / (thisERRL - thisERRa) > -(10.0**-2.0)
            )
            ER[indx] = thisERRa[indx]
            indx = ERLz < self._Lzmin
            indx += ERLz > self._Lzmax
            indx += (ER - thisERRa) / (thisERRL - thisERRa) > 1.0
            indx += (ER - thisERRa) / (thisERRL - thisERRa) < 0.0
            indxc = True ^ indx
            jr = numpy.empty(R.shape)
            if numpy.sum(indxc) > 0:
                jr[indxc] = self._jrInterp.ev(
                    ERLz[indxc],
                    (ER[indxc] - thisERRa[indxc]) / (thisERRL[indxc] - thisERRa[indxc]),
                ) * (numpy.exp(self._jrERRaInterp(ERLz[indxc])) - 10.0**-5.0)
            if numpy.sum(indx) > 0:
                jr[indx] = self._aA(
                    thisRL[indx],
                    numpy.sqrt(
                        2.0
                        * (ER[indx] - _evaluatePotentials(self._pot, thisRL[indx], 0.0))
                        - ERLz[indx] ** 2.0 / thisRL[indx] ** 2.0
                    ),
                    ERLz[indx] / thisRL[indx],
                    numpy.zeros(len(thisRL)),
                    numpy.zeros(len(thisRL)),
                    _justjr=True,
                    **kwargs,
                )[0]
        else:
            if (ER - thisERRa) / (thisERRL - thisERRa) > 1.0 and (
                (ER - thisERRa) / (thisERRL - thisERRa) - 1.0
            ) < 10.0**-2.0:
                ER = thisERRL
            elif (ER - thisERRa) / (thisERRL - thisERRa) < 0.0 and (ER - thisERRa) / (
                thisERRL - thisERRa
            ) > -(10.0**-2.0):
                ER = thisERRa
            # Outside of grid?
            if (
                ERLz < self._Lzmin
                or ERLz > self._Lzmax
                or (ER - thisERRa) / (thisERRL - thisERRa) > 1.0
                or (ER - thisERRa) / (thisERRL - thisERRa) < 0.0
            ):
                if _PRINTOUTSIDEGRID:  # pragma: no cover
                    print(
                        "Outside of grid in ER/Lz",
                        ERLz < self._Lzmin,
                        ERLz > self._Lzmax,
                        (ER - thisERRa) / (thisERRL - thisERRa) > 1.0,
                        (ER - thisERRa) / (thisERRL - thisERRa) < 0.0,
                        ER,
                        thisERRL,
                        thisERRa,
                        (ER - thisERRa) / (thisERRL - thisERRa),
                    )
                jr = self._aA(
                    thisRL[0],
                    numpy.sqrt(
                        2.0 * (ER - _evaluatePotentials(self._pot, thisRL, 0.0))
                        - ERLz**2.0 / thisRL**2.0
                    )[0],
                    (ERLz / thisRL)[0],
                    0.0,
                    0.0,
                    _justjr=True,
                    **kwargs,
                )[0]
            else:
                jr = (
                    self._jrInterp(ERLz, (ER - thisERRa) / (thisERRL - thisERRa))
                    * (numpy.exp(self._jrERRaInterp(ERLz)) - 10.0**-5.0)
                )[0][0]
        return (jr, R * vT, jz)

    def _offgrid_fill(self, xp, base, off, exact):
        """``base`` with the off-grid entries replaced by an exact solve.

        Mirrors what the numpy path already does -- mask, ``sum(indx) > 0``
        guard, scatter-assign -- in backend ops. EAGER can skip the solve
        entirely when nothing is off-grid, which is the whole reason the cost is
        acceptable: an exact ``self._aA`` call costs ~187 ms on torch and ~1.25 s
        on jax BEFORE it touches a single point (measured), so paying it
        unconditionally would make every call 14-77x slower than the grid it
        exists to avoid.

        Under a trace none of that is available: a boolean mask has no concrete
        size, so the points cannot be gathered and the ``any`` cannot be
        branched on. Returning the extrapolated interpolant there would be
        silently wrong, so the off-grid entries come back NaN instead --
        visible, and the same convention RazorThinExponentialDisk uses when a
        domain is undecidable under trace.
        """
        if under_trace(base, off):
            return xp.where(off, xp.asarray(xp.nan, dtype=base.dtype), base)
        if not bool(xp.any(off)):
            return base
        return set_at(xp, base, off, exact(off))

    def _evaluate_backend(self, R, vR, vT, z, vz):
        """Vectorised, differentiable action eval for jax/torch inputs.

        On-grid this is the interpolants. Off-grid it falls back to the exact
        ``self._aA`` solve for just those points, matching the numpy path; under
        a trace off-grid entries come back NaN (see ``_offgrid_fill``).
        """
        xp = get_namespace(R)
        zero = xp.zeros_like(R)
        # Vertical action
        Phi = _evaluatePotentials(self._pot, R, z)
        Phio = _evaluatePotentials(self._pot, R, zero)
        Ez = Phi - Phio + vz**2.0 / 2.0
        thisEzZmax = xp.exp(self._EzZmaxsInterp_b(R))
        jz = self._jzInterp_b(R, Ez / thisEzZmax, grid=False) * (
            xp.exp(self._jzEzmaxInterp_b(R)) - 10.0**-5.0
        )
        # Off-grid in Ez, exactly the numpy `indx`. log() is guarded because it
        # is evaluated for EVERY element here, including the Ez <= 0 ones the
        # (Ez != 0) factor is there to reject -- numpy tolerates the resulting
        # nan/-inf, but feeding nan through a backend op can poison a gradient.
        offz = (
            (R > self._Rmax)
            | (R < self._Rmin)
            | (
                (Ez != 0.0)
                & (xp.log(xp.where(Ez > 0.0, Ez, xp.ones_like(Ez))) > thisEzZmax)
            )
        )
        jz = self._offgrid_fill(
            xp,
            jz,
            offz,
            lambda m: self._aA(
                R[m],
                xp.zeros_like(R[m]),
                xp.ones_like(R[m]),  # these two are dummies
                xp.zeros_like(R[m]),
                xp.sqrt(2.0 * Ez[m]),
                _justjz=True,
                # c=False so use_c is False and _evaluate takes its BACKEND
                # branch. Without it a c=True grid falls through to the C call
                # with backend arrays, which raises NotImplementedError: the C
                # extension cannot accept a jax/torch array at all.
                c=False,
            )[2],
        )
        # Radial action
        ERLz = xp.abs(R * vT) + self._gamma * jz
        ER = Phio + vR**2.0 / 2.0 + ERLz**2.0 / 2.0 / R**2.0
        thisRL = self._RLInterp_b(ERLz)
        thisERRL = -xp.exp(self._ERRLInterp_b(ERLz)) + self._ERRLmax
        thisERRa = -xp.exp(self._ERRaInterp_b(ERLz)) + self._ERRamax
        frac = (ER - thisERRa) / (thisERRL - thisERRa)
        # Snap the two near-boundary cases (mirrors the numpy ER[indx]= writes).
        ER = xp.where((frac > 1.0) & ((frac - 1.0) < 10.0**-2.0), thisERRL, ER)
        ER = xp.where((frac < 0.0) & (frac > -(10.0**-2.0)), thisERRa, ER)
        frac = (ER - thisERRa) / (thisERRL - thisERRa)
        jr = self._jrInterp_b(ERLz, frac, grid=False) * (
            xp.exp(self._jrERRaInterp_b(ERLz)) - 10.0**-5.0
        )
        # Off-grid in Lz / ER, exactly the numpy `indx`. Recomputed from the
        # SNAPPED frac, as numpy does -- the snap moves points back on-grid.
        offr = (ERLz < self._Lzmin) | (ERLz > self._Lzmax) | (frac > 1.0) | (frac < 0.0)
        jr = self._offgrid_fill(
            xp,
            jr,
            offr,
            lambda m: self._aA(
                thisRL[m],
                xp.sqrt(
                    2.0
                    * (
                        ER[m]
                        - _evaluatePotentials(
                            self._pot, thisRL[m], xp.zeros_like(thisRL[m])
                        )
                    )
                    - ERLz[m] ** 2.0 / thisRL[m] ** 2.0
                ),
                ERLz[m] / thisRL[m],
                xp.zeros_like(thisRL[m]),
                xp.zeros_like(thisRL[m]),
                _justjr=True,
                c=False,  # see the _justjz call above
            )[0],
        )
        return (jr, R * vT, jz)

    def Jz(self, *args, **kwargs):
        """
        Evaluate the action jz.

        Parameters
        ----------
        *args : tuple
            Either:
                a) R,vR,vT,z,vz
                b) Orbit instance: initial condition used if that's it, orbit(t)
                    if there is a time given as well
        **kwargs: dict
            scipy.integrate.quadrature keywords

        Returns
        -------
        float
            The action jz.

        Notes
        -----
        - 2012-07-30 - Written - Bovy (IAS@MPIA)

        """
        self._parse_eval_args(*args)
        xp = get_namespace(
            self._eval_R, self._eval_vR, self._eval_vT, self._eval_z, self._eval_vz
        )
        if xp is not numpy:  # jax/torch: on-grid backend eval (no scipy fits built)
            R, vR, vT, z, vz = promote_scalars(
                xp,
                self._eval_R,
                self._eval_vR,
                self._eval_vT,
                self._eval_z,
                self._eval_vz,
            )
            return self._evaluate_backend(R, vR, vT, z, vz)[2]
        Phi = _evaluatePotentials(self._pot, self._eval_R, self._eval_z)
        Phio = _evaluatePotentials(self._pot, self._eval_R, 0.0)
        Ez = Phi - Phio + self._eval_vz**2.0 / 2.0
        # Bigger than Ezzmax?
        thisEzZmax = numpy.exp(self._EzZmaxsInterp(self._eval_R))
        if (
            self._eval_R > self._Rmax
            or self._eval_R < self._Rmin
            or (Ez != 0.0 and numpy.log(Ez) > thisEzZmax)
        ):  # Outside of the grid
            if _PRINTOUTSIDEGRID:  # pragma: no cover
                print("Outside of grid in Ez")
            jz = self._aA(
                self._eval_R,
                0.0,
                1.0,  # these two r dummies
                0.0,
                numpy.sqrt(2.0 * Ez),
                _justjz=True,
                **kwargs,
            )[2]
        else:
            jz = (
                self._jzInterp(self._eval_R, Ez / thisEzZmax)
                * (numpy.exp(self._jzEzmaxInterp(self._eval_R)) - 10.0**-5.0)
            )[0][0]
        return jz


def _tileT_flat(xp, a, reps):
    """``(tile(a, (reps, 1)).T).flatten()`` -- the numpy grid-tiling idiom, in xp.
    ``a`` is 1-D (n,); returns (n*reps,) with ``a`` varying slowest."""
    return xp.reshape(xp.matrix_transpose(xp.tile(a, (reps, 1))), (-1,))
