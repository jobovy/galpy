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
from ..potential.Potential import _evaluatePotentials
from ..potential.Potential import flatten as flatten_potential
from ..util import multi
from .actionAngle import UnboundError, actionAngle
from .actionAngleAdiabatic import actionAngleAdiabatic

_PRINTOUTSIDEGRID = False


class actionAngleAdiabaticGrid(actionAngle):
    """Action-angle formalism for axisymmetric potentials using the adiabatic approximation, grid-based interpolation"""

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
        pot : Potential or list of Potential instances
            The potential (instance) or list of potentials (instances) that make up the potential
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
        self._pot = flatten_potential(pot)
        self._zmax = zmax
        self._Rmax = Rmax
        self._Rmin = 0.01
        # Set up the actionAngleAdiabatic object that we will use to interpolate
        self._aA = actionAngleAdiabatic(pot=self._pot, gamma=self._gamma, c=self._c)
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
        # Check the units
        self._check_consistent_units()
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
