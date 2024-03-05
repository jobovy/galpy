###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleIsochroneApprox
#
#             Calculate actions-angle coordinates for any potential by using
#             an isochrone potential as an approximate potential and using
#             a Fox & Binney (2013?) + torus machinery-like algorithm
#             (angle-fit) (Bovy 2014)
#
#      methods:
#             __call__: returns (jr,lz,jz)
#             actionsFreqs: returns (jr,lz,jz,Or,Op,Oz)
#             actionsFreqsAngles: returns (jr,lz,jz,Or,Op,Oz,ar,ap,az)
#
###############################################################################
import warnings

import numpy
from numpy import linalg
from scipy import optimize

from ..potential import IsochronePotential, MWPotential, _isNonAxi, dvcircdR, vcirc
from ..potential.Potential import flatten as flatten_potential
from ..util import conversion, galpyWarning, plot
from ..util.conversion import physical_conversion, potential_physical_input, time_in_Gyr
from .actionAngle import actionAngle
from .actionAngleIsochrone import actionAngleIsochrone

_TWOPI = 2.0 * numpy.pi
_ANGLETOL = 0.02  # tolerance for deciding whether full angle range is covered


class actionAngleIsochroneApprox(actionAngle):
    """Action-angle formalism using an isochrone potential as an approximate potential and using a Fox & Binney (2014?) like algorithm to calculate the actions using orbit integrations and a torus-machinery-like angle-fit to get the angles and frequencies (Bovy 2014)"""

    def __init__(self, *args, **kwargs):
        """
        Initialize an actionAngleIsochroneApprox object.

        Parameters
        ----------
        b : float or Quantity, optional
            Scale parameter of the isochrone parameter.
        ip : IsochronePotential, optional
            Instance of a IsochronePotential.
        aAI : actionAngleIsochrone, optional
            Instance of an actionAngleIsochrone.
        pot : Potential or list of Potentials, optional
            Potential to calculate action-angle variables for.
        tintJ : float, optional
            Time to integrate orbits for to estimate actions (can be Quantity).
        ntintJ : int, optional
            Number of time-integration points.
        integrate_method : str, optional
            Integration method to use.
        dt : float, optional
            orbit.integrate dt keyword (for fixed stepsize integration).
        maxn : int, optional
            Default value for all methods when using a grid in vec(n) up to this n (zero-based).
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2013-09-10 - Written - Bovy (IAS).

        """
        actionAngle.__init__(self, ro=kwargs.get("ro", None), vo=kwargs.get("vo", None))
        if not "pot" in kwargs:  # pragma: no cover
            raise OSError("Must specify pot= for actionAngleIsochroneApprox")
        self._pot = flatten_potential(kwargs["pot"])
        if self._pot == MWPotential:
            warnings.warn(
                "Use of MWPotential as a Milky-Way-like potential is deprecated; galpy.potential.MWPotential2014, a potential fit to a large variety of dynamical constraints (see Bovy 2015), is the preferred Milky-Way-like potential in galpy",
                galpyWarning,
            )
        if (
            not "b" in kwargs and not "ip" in kwargs and not "aAI" in kwargs
        ):  # pragma: no cover
            raise OSError(
                "Must specify b=, ip=, or aAI= for actionAngleIsochroneApprox"
            )
        if "aAI" in kwargs:
            if not isinstance(kwargs["aAI"], actionAngleIsochrone):  # pragma: no cover
                raise OSError(
                    "'Provided aAI= does not appear to be an instance of an actionAngleIsochrone"
                )
            self._aAI = kwargs["aAI"]
        elif "ip" in kwargs:
            ip = kwargs["ip"]
            if not isinstance(ip, IsochronePotential):  # pragma: no cover
                raise OSError(
                    "'Provided ip= does not appear to be an instance of an IsochronePotential"
                )
            self._aAI = actionAngleIsochrone(ip=ip)
        else:
            b = conversion.parse_length(kwargs["b"], ro=self._ro)
            self._aAI = actionAngleIsochrone(ip=IsochronePotential(b=b, normalize=1.0))
        self._tintJ = kwargs.get("tintJ", 100.0)
        self._tintJ = conversion.parse_time(self._tintJ, ro=self._ro, vo=self._vo)
        self._ntintJ = kwargs.get("ntintJ", 10000)
        self._integrate_dt = kwargs.get("dt", None)
        self._tsJ = numpy.linspace(0.0, self._tintJ, self._ntintJ)
        self._integrate_method = kwargs.get("integrate_method", "dopr54_c")
        self._maxn = kwargs.get("maxn", 3)
        self._c = False
        ext_loaded = False
        if ext_loaded and (
            ("c" in kwargs and kwargs["c"]) or not "c" in kwargs
        ):  # pragma: no cover
            self._c = True
        else:
            self._c = False
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
        cumul: bool, optional
            if True, return the cumulative average actions (to look at convergence).

        Returns
        -------
        tuple
            (jr,lz,jz)

        Notes
        -----
        - 2013-09-10 - Written - Bovy (IAS).
        """
        R, vR, vT, z, vz, phi = self._parse_args(False, False, *args)
        if self._c:  # pragma: no cover
            pass
        else:
            # Use self._aAI to calculate the actions and angles in the isochrone potential
            acfs = self._aAI._actionsFreqsAngles(
                R.flatten(),
                vR.flatten(),
                vT.flatten(),
                z.flatten(),
                vz.flatten(),
                phi.flatten(),
            )
            jrI = numpy.reshape(acfs[0], R.shape)[:, :-1]
            jzI = numpy.reshape(acfs[2], R.shape)[:, :-1]
            anglerI = numpy.reshape(acfs[6], R.shape)
            anglezI = numpy.reshape(acfs[8], R.shape)
            if numpy.any(
                (numpy.fabs(numpy.amax(anglerI, axis=1) - _TWOPI) > _ANGLETOL)
                * (numpy.fabs(numpy.amin(anglerI, axis=1)) > _ANGLETOL)
            ):  # pragma: no cover
                warnings.warn(
                    "Full radial angle range not covered for at least one object; actions are likely not reliable",
                    galpyWarning,
                )
            if numpy.any(
                (numpy.fabs(numpy.amax(anglezI, axis=1) - _TWOPI) > _ANGLETOL)
                * (numpy.fabs(numpy.amin(anglezI, axis=1)) > _ANGLETOL)
            ):  # pragma: no cover
                warnings.warn(
                    "Full vertical angle range not covered for at least one object; actions are likely not reliable",
                    galpyWarning,
                )
            danglerI = ((numpy.roll(anglerI, -1, axis=1) - anglerI) % _TWOPI)[:, :-1]
            danglezI = ((numpy.roll(anglezI, -1, axis=1) - anglezI) % _TWOPI)[:, :-1]
            if kwargs.get("cumul", False):
                sumFunc = numpy.cumsum
            else:
                sumFunc = numpy.sum
            jr = sumFunc(jrI * danglerI, axis=1) / sumFunc(danglerI, axis=1)
            jz = sumFunc(jzI * danglezI, axis=1) / sumFunc(danglezI, axis=1)
            if _isNonAxi(self._pot):
                lzI = numpy.reshape(acfs[1], R.shape)[:, :-1]
                anglephiI = numpy.reshape(acfs[7], R.shape)
                danglephiI = ((numpy.roll(anglephiI, -1, axis=1) - anglephiI) % _TWOPI)[
                    :, :-1
                ]
                if numpy.any(
                    (numpy.fabs(numpy.amax(anglephiI, axis=1) - _TWOPI) > _ANGLETOL)
                    * (numpy.fabs(numpy.amin(anglephiI, axis=1)) > _ANGLETOL)
                ):  # pragma: no cover
                    warnings.warn(
                        "Full azimuthal angle range not covered for at least one object; actions are likely not reliable",
                        galpyWarning,
                    )
                lz = sumFunc(lzI * danglephiI, axis=1) / sumFunc(danglephiI, axis=1)
            else:
                lz = R[:, 0] * vT[:, 0]
            return (jr, lz, jz)

    def _actionsFreqs(self, *args, **kwargs):
        """
        Evaluate the actions (jr,lz,jz) and frequencies (Or,Op,Oz).

        Parameters
        ----------
        *args : tuple
            Either:
            a) R,vR,vT,z,vz[,phi]:
                1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)
                2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
            b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
        maxn: int, optional
            Use a grid in vec(n) up to this n (zero-based). Default is the object-wide default.
        ts : float, optional
            if set, the phase-space points correspond to these times (IF NOT SET, WE ASSUME THAT ts IS THAT THAT IS ASSOCIATED WITH THIS OBJECT)
        _firstFlip : bool, optional
            if True and Orbits are given, the backward part of the orbit is integrated first and stored in the Orbit object

        Returns
        -------
        tuple
            (jr,lz,jz,Or,Op,Oz)

        Notes
        -----
        - 2013-09-10 - Written - Bovy (IAS).
        """
        acfs = self._actionsFreqsAngles(*args, **kwargs)
        return (acfs[0], acfs[1], acfs[2], acfs[3], acfs[4], acfs[5])

    def _actionsFreqsAngles(self, *args, **kwargs):
        """
        Evaluate the actions (jr,lz,jz), frequencies (Or,Op,Oz), and angles (angler,anglephi,anglez).

        Parameters
        ----------
        *args : tuple
            Either:
            a) R,vR,vT,z,vz[,phi]:
                1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)
                2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
            b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
        maxn: int, optional
            Use a grid in vec(n) up to this n (zero-based). Default is the object-wide default.
        ts : float, optional
            if set, the phase-space points correspond to these times (IF NOT SET, WE ASSUME THAT ts IS THAT THAT IS ASSOCIATED WITH THIS OBJECT)
        _firstFlip : bool, optional
            if True and Orbits are given, the backward part of the orbit is integrated first and stored in the Orbit object

        Returns
        -------
        tuple
            (jr,lz,jz,Or,Op,Oz,angler,anglephi,anglez)

        Notes
        -----
        - 2013-09-10 - Written - Bovy (IAS).
        """
        from ..orbit import Orbit

        _firstFlip = kwargs.get("_firstFlip", False)
        # If the orbit was already integrated, set ts to the integration times
        if (
            isinstance(args[0], Orbit)
            and hasattr(args[0], "orbit")
            and not "ts" in kwargs
        ):
            kwargs["ts"] = args[0].t
        elif (
            (isinstance(args[0], list) and isinstance(args[0][0], Orbit))
            and hasattr(args[0][0], "orbit")
            and not "ts" in kwargs
        ):
            kwargs["ts"] = args[0][0].t
        R, vR, vT, z, vz, phi = self._parse_args(True, _firstFlip, *args)
        if "ts" in kwargs and not kwargs["ts"] is None:
            ts = kwargs["ts"]
            ts = conversion.parse_time(ts, ro=self._ro, vo=self._vo)
        else:
            ts = numpy.empty(R.shape[1])
            ts[self._ntintJ - 1 :] = self._tsJ
            ts[: self._ntintJ - 1] = -self._tsJ[1:][::-1]
        maxn = kwargs.get("maxn", self._maxn)
        if self._c:  # pragma: no cover
            pass
        else:
            # Use self._aAI to calculate the actions and angles in the isochrone potential
            acfs = self._aAI._actionsFreqsAngles(
                R.flatten(),
                vR.flatten(),
                vT.flatten(),
                z.flatten(),
                vz.flatten(),
                phi.flatten(),
            )
            jrI = numpy.reshape(acfs[0], R.shape)[:, :-1]
            jzI = numpy.reshape(acfs[2], R.shape)[:, :-1]
            anglerI = numpy.reshape(acfs[6], R.shape)
            anglezI = numpy.reshape(acfs[8], R.shape)
            if numpy.any(
                (numpy.fabs(numpy.amax(anglerI, axis=1) - _TWOPI) > _ANGLETOL)
                * (numpy.fabs(numpy.amin(anglerI, axis=1)) > _ANGLETOL)
            ):  # pragma: no cover
                warnings.warn(
                    "Full radial angle range not covered for at least one object; actions are likely not reliable",
                    galpyWarning,
                )
            if numpy.any(
                (numpy.fabs(numpy.amax(anglezI, axis=1) - _TWOPI) > _ANGLETOL)
                * (numpy.fabs(numpy.amin(anglezI, axis=1)) > _ANGLETOL)
            ):  # pragma: no cover
                warnings.warn(
                    "Full vertical angle range not covered for at least one object; actions are likely not reliable",
                    galpyWarning,
                )
            danglerI = ((numpy.roll(anglerI, -1, axis=1) - anglerI) % _TWOPI)[:, :-1]
            danglezI = ((numpy.roll(anglezI, -1, axis=1) - anglezI) % _TWOPI)[:, :-1]
            jr = numpy.sum(jrI * danglerI, axis=1) / numpy.sum(danglerI, axis=1)
            jz = numpy.sum(jzI * danglezI, axis=1) / numpy.sum(danglezI, axis=1)
            if _isNonAxi(self._pot):  # pragma: no cover
                lzI = numpy.reshape(acfs[1], R.shape)[:, :-1]
                anglephiI = numpy.reshape(acfs[7], R.shape)
                if numpy.any(
                    (numpy.fabs(numpy.amax(anglephiI, axis=1) - _TWOPI) > _ANGLETOL)
                    * (numpy.fabs(numpy.amin(anglephiI, axis=1)) > _ANGLETOL)
                ):  # pragma: no cover
                    warnings.warn(
                        "Full azimuthal angle range not covered for at least one object; actions are likely not reliable",
                        galpyWarning,
                    )
                danglephiI = ((numpy.roll(anglephiI, -1, axis=1) - anglephiI) % _TWOPI)[
                    :, :-1
                ]
                lz = numpy.sum(lzI * danglephiI, axis=1) / numpy.sum(danglephiI, axis=1)
            else:
                lz = R[:, len(ts) // 2] * vT[:, len(ts) // 2]
            # Now do an 'angle-fit'
            angleRT = dePeriod(numpy.reshape(acfs[6], R.shape))
            acfs7 = numpy.reshape(acfs[7], R.shape)
            negFreqIndx = (
                numpy.median(acfs7 - numpy.roll(acfs7, 1, axis=1), axis=1) < 0.0
            )  # anglephi is decreasing
            anglephiT = numpy.empty(acfs7.shape)
            anglephiT[negFreqIndx, :] = dePeriod(_TWOPI - acfs7[negFreqIndx, :])
            negFreqPhi = numpy.zeros(R.shape[0], dtype="bool")
            negFreqPhi[negFreqIndx] = True
            anglephiT[True ^ negFreqIndx, :] = dePeriod(acfs7[True ^ negFreqIndx, :])
            angleZT = dePeriod(numpy.reshape(acfs[8], R.shape))
            # Write the angle-fit as Y=AX, build A and Y
            nt = len(ts)
            no = R.shape[0]
            # remove 0,0,0 and half-plane
            if _isNonAxi(self._pot):
                nn = (2 * maxn - 1) ** 2 * maxn - (maxn - 1) * (2 * maxn - 1) - maxn
            else:
                nn = maxn * (2 * maxn - 1) - maxn
            A = numpy.zeros((no, nt, 2 + nn))
            A[:, :, 0] = 1.0
            A[:, :, 1] = ts
            # sorting the phi and Z grids this way makes it easy to exclude the origin
            phig = list(numpy.arange(-maxn + 1, maxn, 1))
            phig.sort(key=lambda x: abs(x))
            phig = numpy.array(phig, dtype="int")
            if _isNonAxi(self._pot):
                grid = numpy.meshgrid(numpy.arange(maxn), phig, phig)
            else:
                grid = numpy.meshgrid(numpy.arange(maxn), phig)
            gridR = grid[0].T.flatten()[1:]  # remove 0,0,0
            gridZ = grid[1].T.flatten()[1:]
            mask = numpy.ones(len(gridR), dtype=bool)
            # excludes axis that is not in half-space
            if _isNonAxi(self._pot):
                gridphi = grid[2].T.flatten()[1:]
                mask = True ^ (gridR == 0) * (
                    (gridphi < 0) + ((gridphi == 0) * (gridZ < 0))
                )
            else:
                mask[: 2 * maxn - 3 : 2] = False
            gridR = gridR[mask]
            gridZ = gridZ[mask]
            tangleR = numpy.tile(angleRT.T, (nn, 1, 1)).T
            tgridR = numpy.tile(gridR, (no, nt, 1))
            tangleZ = numpy.tile(angleZT.T, (nn, 1, 1)).T
            tgridZ = numpy.tile(gridZ, (no, nt, 1))
            if _isNonAxi(self._pot):
                gridphi = gridphi[mask]
                tgridphi = numpy.tile(gridphi, (no, nt, 1))
                tanglephi = numpy.tile(anglephiT.T, (nn, 1, 1)).T
                sinnR = numpy.sin(
                    tgridR * tangleR + tgridphi * tanglephi + tgridZ * tangleZ
                )
            else:
                sinnR = numpy.sin(tgridR * tangleR + tgridZ * tangleZ)
            A[:, :, 2:] = sinnR
            # Matrix magic
            atainv = numpy.empty((no, 2 + nn, 2 + nn))
            AT = numpy.transpose(A, axes=(0, 2, 1))
            for ii in range(no):
                atainv[
                    ii,
                    :,
                    :,
                ] = linalg.inv(numpy.dot(AT[ii, :, :], A[ii, :, :]))
            ATAR = numpy.sum(
                AT
                * numpy.transpose(numpy.tile(angleRT, (2 + nn, 1, 1)), axes=(1, 0, 2)),
                axis=2,
            )
            ATAT = numpy.sum(
                AT
                * numpy.transpose(
                    numpy.tile(anglephiT, (2 + nn, 1, 1)), axes=(1, 0, 2)
                ),
                axis=2,
            )
            ATAZ = numpy.sum(
                AT
                * numpy.transpose(numpy.tile(angleZT, (2 + nn, 1, 1)), axes=(1, 0, 2)),
                axis=2,
            )
            angleR = numpy.sum(atainv[:, 0, :] * ATAR, axis=1)
            OmegaR = numpy.sum(atainv[:, 1, :] * ATAR, axis=1)
            anglephi = numpy.sum(atainv[:, 0, :] * ATAT, axis=1)
            Omegaphi = numpy.sum(atainv[:, 1, :] * ATAT, axis=1)
            angleZ = numpy.sum(atainv[:, 0, :] * ATAZ, axis=1)
            OmegaZ = numpy.sum(atainv[:, 1, :] * ATAZ, axis=1)
            Omegaphi[negFreqIndx] = -Omegaphi[negFreqIndx]
            anglephi[negFreqIndx] = _TWOPI - anglephi[negFreqIndx]
            if kwargs.get("_retacfs", False):
                return (
                    jr,
                    lz,
                    jz,
                    OmegaR,
                    Omegaphi,
                    OmegaZ,  # pragma: no cover
                    angleR % _TWOPI,
                    anglephi % _TWOPI,
                    angleZ % _TWOPI,
                    acfs,
                )
            else:
                return (
                    jr,
                    lz,
                    jz,
                    OmegaR,
                    Omegaphi,
                    OmegaZ,
                    angleR % _TWOPI,
                    anglephi % _TWOPI,
                    angleZ % _TWOPI,
                )

    def plot(self, *args, **kwargs):
        """
        Plot the angles vs. each other, to check whether the isochrone approximation is good.

        Parameters
        ----------
        *args : tuple
            Either:
            a) R,vR,vT,z,vz[,phi]:
                1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)
                2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
            b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
        type: {'araz','araphi','azaphi','jr','lz','jz'}, optional
            type of plot to make
        deperiod: bool, optional
            if True, de-period the angles.
        downsample: bool, optional
            if True, downsample what's plotted to 400 points.
        **kwargs: dict, optional
            Any other keyword argument supported by plot.bovy_plot (e.g., xrange=[xmin,xmax])

        Returns
        -------
        plot

        Notes
        -----
        - 2013-09-10 - Written - Bovy (IAS).
        """
        # Kwargs
        type = kwargs.pop("type", "araz")
        deperiod = kwargs.pop("deperiod", False)
        downsample = kwargs.pop("downsample", False)
        # Parse input
        R, vR, vT, z, vz, phi = self._parse_args("a" in type, False, *args)
        # Use self._aAI to calculate the actions and angles in the isochrone potential
        acfs = self._aAI._actionsFreqsAngles(
            R.flatten(),
            vR.flatten(),
            vT.flatten(),
            z.flatten(),
            vz.flatten(),
            phi.flatten(),
        )
        if type == "jr" or type == "lz" or type == "jz":
            jrI = numpy.reshape(acfs[0], R.shape)[:, :-1]
            jzI = numpy.reshape(acfs[2], R.shape)[:, :-1]
            anglerI = numpy.reshape(acfs[6], R.shape)
            anglezI = numpy.reshape(acfs[8], R.shape)
            danglerI = ((numpy.roll(anglerI, -1, axis=1) - anglerI) % _TWOPI)[:, :-1]
            danglezI = ((numpy.roll(anglezI, -1, axis=1) - anglezI) % _TWOPI)[:, :-1]
            if True:
                sumFunc = numpy.cumsum
            jr = sumFunc(jrI * danglerI, axis=1) / sumFunc(danglerI, axis=1)
            jz = sumFunc(jzI * danglezI, axis=1) / sumFunc(danglezI, axis=1)
            lzI = numpy.reshape(acfs[1], R.shape)[:, :-1]
            anglephiI = numpy.reshape(acfs[7], R.shape)
            danglephiI = ((numpy.roll(anglephiI, -1, axis=1) - anglephiI) % _TWOPI)[
                :, :-1
            ]
            lz = sumFunc(lzI * danglephiI, axis=1) / sumFunc(danglephiI, axis=1)
            from ..orbit import Orbit

            if isinstance(args[0], Orbit) and hasattr(args[0], "t"):
                ts = args[0].t[:-1]
            else:
                ts = self._tsJ[:-1]
            if type == "jr":
                if downsample:
                    plotx = ts[:: int(round(self._ntintJ // 400))]
                    ploty = jr[0, :: int(round(self._ntintJ // 400))] / jr[0, -1]
                    plotz = anglerI[0, : -1 : int(round(self._ntintJ // 400))]
                else:
                    plotx = ts
                    ploty = jr[0, :] / jr[0, -1]
                    plotz = anglerI[0, :-1]
                plot.plot(
                    plotx,
                    ploty,
                    c=plotz,
                    s=20.0,
                    scatter=True,
                    edgecolor="none",
                    xlabel=r"$t$",
                    ylabel=r"$J^A_R / \langle J^A_R \rangle$",
                    clabel=r"$\theta^A_R$",
                    vmin=0.0,
                    vmax=2.0 * numpy.pi,
                    crange=[0.0, 2.0 * numpy.pi],
                    colorbar=True,
                    **kwargs,
                )
            elif type == "lz":
                if downsample:
                    plotx = ts[:: int(round(self._ntintJ // 400))]
                    ploty = lz[0, :: int(round(self._ntintJ // 400))] / lz[0, -1]
                    plotz = anglephiI[0, : -1 : int(round(self._ntintJ // 400))]
                else:
                    plotx = ts
                    ploty = lz[0, :] / lz[0, -1]
                    plotz = anglephiI[0, :-1]
                plot.plot(
                    plotx,
                    ploty,
                    c=plotz,
                    s=20.0,
                    scatter=True,
                    edgecolor="none",
                    xlabel=r"$t$",
                    ylabel=r"$L^A_Z / \langle L^A_Z \rangle$",
                    clabel=r"$\theta^A_\phi$",
                    vmin=0.0,
                    vmax=2.0 * numpy.pi,
                    crange=[0.0, 2.0 * numpy.pi],
                    colorbar=True,
                    **kwargs,
                )
            elif type == "jz":
                if downsample:
                    plotx = ts[:: int(round(self._ntintJ // 400))]
                    ploty = jz[0, :: int(round(self._ntintJ // 400))] / jz[0, -1]
                    plotz = anglezI[0, : -1 : int(round(self._ntintJ // 400))]
                else:
                    plotx = ts
                    ploty = jz[0, :] / jz[0, -1]
                    plotz = anglezI[0, :-1]
                plot.plot(
                    plotx,
                    ploty,
                    c=plotz,
                    s=20.0,
                    scatter=True,
                    edgecolor="none",
                    xlabel=r"$t$",
                    ylabel=r"$J^A_Z / \langle J^A_Z \rangle$",
                    clabel=r"$\theta^A_Z$",
                    vmin=0.0,
                    vmax=2.0 * numpy.pi,
                    crange=[0.0, 2.0 * numpy.pi],
                    colorbar=True,
                    **kwargs,
                )
        else:
            if deperiod:
                if "ar" in type:
                    angleRT = dePeriod(numpy.reshape(acfs[6], R.shape))
                else:
                    angleRT = numpy.reshape(acfs[6], R.shape)
                if "aphi" in type:
                    acfs7 = numpy.reshape(acfs[7], R.shape)
                    negFreqIndx = (
                        numpy.median(acfs7 - numpy.roll(acfs7, 1, axis=1), axis=1) < 0.0
                    )  # anglephi is decreasing
                    anglephiT = numpy.empty(acfs7.shape)
                    anglephiT[negFreqIndx, :] = dePeriod(_TWOPI - acfs7[negFreqIndx, :])
                    negFreqPhi = numpy.zeros(R.shape[0], dtype="bool")
                    negFreqPhi[negFreqIndx] = True
                    anglephiT[True ^ negFreqIndx, :] = dePeriod(
                        acfs7[True ^ negFreqIndx, :]
                    )
                else:
                    anglephiT = numpy.reshape(acfs[7], R.shape)
                if "az" in type:
                    angleZT = dePeriod(numpy.reshape(acfs[8], R.shape))
                else:
                    angleZT = numpy.reshape(acfs[8], R.shape)
                xrange = None
                yrange = None
            else:
                angleRT = numpy.reshape(acfs[6], R.shape)
                anglephiT = numpy.reshape(acfs[7], R.shape)
                angleZT = numpy.reshape(acfs[8], R.shape)
                xrange = [-0.5, 2.0 * numpy.pi + 0.5]
                yrange = [-0.5, 2.0 * numpy.pi + 0.5]
            vmin, vmax = 0.0, 2.0 * numpy.pi
            crange = [vmin, vmax]
            if type == "araz":
                if downsample:
                    plotx = angleRT[0, :: int(round(self._ntintJ // 400))]
                    ploty = angleZT[0, :: int(round(self._ntintJ // 400))]
                    plotz = anglephiT[0, :: int(round(self._ntintJ // 400))]
                else:
                    plotx = angleRT[0, :]
                    ploty = angleZT[0, :]
                    plotz = anglephiT[0, :]
                plot.plot(
                    plotx,
                    ploty,
                    c=plotz,
                    s=20.0,
                    scatter=True,
                    edgecolor="none",
                    xlabel=r"$\theta^A_R$",
                    ylabel=r"$\theta^A_Z$",
                    clabel=r"$\theta^A_\phi$",
                    xrange=xrange,
                    yrange=yrange,
                    vmin=vmin,
                    vmax=vmax,
                    crange=crange,
                    colorbar=True,
                    **kwargs,
                )
            elif type == "araphi":
                if downsample:
                    plotx = angleRT[0, :: int(round(self._ntintJ // 400))]
                    ploty = anglephiT[0, :: int(round(self._ntintJ // 400))]
                    plotz = angleZT[0, :: int(round(self._ntintJ // 400))]
                else:
                    plotx = angleRT[0, :]
                    ploty = anglephiT[0, :]
                    plotz = angleZT[0, :]
                plot.plot(
                    plotx,
                    ploty,
                    c=plotz,
                    s=20.0,
                    scatter=True,
                    edgecolor="none",
                    xlabel=r"$\theta^A_R$",
                    clabel=r"$\theta^A_Z$",
                    ylabel=r"$\theta^A_\phi$",
                    xrange=xrange,
                    yrange=yrange,
                    vmin=vmin,
                    vmax=vmax,
                    crange=crange,
                    colorbar=True,
                    **kwargs,
                )
            elif type == "azaphi":
                if downsample:
                    plotx = angleZT[0, :: int(round(self._ntintJ // 400))]
                    ploty = anglephiT[0, :: int(round(self._ntintJ // 400))]
                    plotz = angleRT[0, :: int(round(self._ntintJ // 400))]
                else:
                    plotx = angleZT[0, :]
                    ploty = anglephiT[0, :]
                    plotz = angleRT[0, :]
                plot.plot(
                    plotx,
                    ploty,
                    c=plotz,
                    s=20.0,
                    scatter=True,
                    edgecolor="none",
                    clabel=r"$\theta^A_R$",
                    xlabel=r"$\theta^A_Z$",
                    ylabel=r"$\theta^A_\phi$",
                    xrange=xrange,
                    yrange=yrange,
                    vmin=vmin,
                    vmax=vmax,
                    crange=crange,
                    colorbar=True,
                    **kwargs,
                )
        return None

    def _parse_args(self, freqsAngles=True, _firstFlip=False, *args):
        """Helper function to parse the arguments to the __call__ and actionsFreqsAngles functions"""
        from ..orbit import Orbit

        RasOrbit = False
        integrated = True  # whether the orbit was already integrated when given
        if len(args) == 5 or len(args) == 3:  # pragma: no cover
            raise OSError("Must specify phi for actionAngleIsochroneApprox")
        if len(args) == 6 or len(args) == 4:
            if len(args) == 6:
                R, vR, vT, z, vz, phi = args
            else:
                R, vR, vT, phi = args
                z, vz = numpy.zeros_like(R), numpy.zeros_like(R)
            if isinstance(R, float):
                os = [Orbit([R, vR, vT, z, vz, phi])]
                RasOrbit = True
                integrated = False
            elif len(R.shape) == 1:  # not integrated yet
                os = [
                    Orbit([R[ii], vR[ii], vT[ii], z[ii], vz[ii], phi[ii]])
                    for ii in range(R.shape[0])
                ]
                RasOrbit = True
                integrated = False
        if (
            isinstance(args[0], Orbit)
            or (isinstance(args[0], list) and isinstance(args[0][0], Orbit))
            or RasOrbit
        ):
            if RasOrbit:
                pass
            elif not isinstance(args[0], list):
                os = [args[0]]
                if os[0].phasedim() == 3 or os[0].phasedim() == 5:  # pragma: no cover
                    raise OSError("Must specify phi for actionAngleIsochroneApprox")
            else:
                os = args[0]
                if os[0].phasedim() == 3 or os[0].phasedim() == 5:  # pragma: no cover
                    raise OSError("Must specify phi for actionAngleIsochroneApprox")
            self._check_consistent_units_orbitInput(os[0])
            if not hasattr(os[0], "orbit"):  # not integrated yet
                if _firstFlip:
                    for o in os:
                        o.vxvv[..., 1] = -o.vxvv[..., 1]
                        o.vxvv[..., 2] = -o.vxvv[..., 2]
                        o.vxvv[..., 4] = -o.vxvv[..., 4]
                [
                    o.integrate(
                        self._tsJ,
                        pot=self._pot,
                        method=self._integrate_method,
                        dt=self._integrate_dt,
                    )
                    for o in os
                ]
                if _firstFlip:
                    for o in os:
                        o.vxvv[..., 1] = -o.vxvv[..., 1]
                        o.vxvv[..., 2] = -o.vxvv[..., 2]
                        o.vxvv[..., 4] = -o.vxvv[..., 4]
                        o.orbit[..., 1] = -o.orbit[..., 1]
                        o.orbit[..., 2] = -o.orbit[..., 2]
                        o.orbit[..., 4] = -o.orbit[..., 4]
                integrated = False
            ntJ = os[0].getOrbit().shape[0]
            no = len(os)
            R = numpy.empty((no, ntJ))
            vR = numpy.empty((no, ntJ))
            vT = numpy.empty((no, ntJ))
            z = numpy.zeros((no, ntJ)) + 10.0**-7.0  # To avoid numpy warnings for
            vz = numpy.zeros((no, ntJ)) + 10.0**-7.0  # planarOrbits
            phi = numpy.empty((no, ntJ))
            for ii in range(len(os)):
                this_orbit = os[ii].getOrbit()
                R[ii, :] = this_orbit[:, 0]
                vR[ii, :] = this_orbit[:, 1]
                vT[ii, :] = this_orbit[:, 2]
                if this_orbit.shape[1] == 6:
                    z[ii, :] = this_orbit[:, 3]
                    vz[ii, :] = this_orbit[:, 4]
                    phi[ii, :] = this_orbit[:, 5]
                else:
                    phi[ii, :] = this_orbit[:, 3]
        if (
            freqsAngles and not integrated
        ):  # also integrate backwards in time, such that the requested point is not at the edge
            no = R.shape[0]
            nt = R.shape[1]
            oR = numpy.empty((no, 2 * nt - 1))
            ovR = numpy.empty((no, 2 * nt - 1))
            ovT = numpy.empty((no, 2 * nt - 1))
            oz = (
                numpy.zeros((no, 2 * nt - 1)) + 10.0**-7.0
            )  # To avoid numpy warnings for
            ovz = numpy.zeros((no, 2 * nt - 1)) + 10.0**-7.0  # planarOrbits
            ophi = numpy.empty((no, 2 * nt - 1))
            if _firstFlip:
                oR[:, :nt] = R[:, ::-1]
                ovR[:, :nt] = vR[:, ::-1]
                ovT[:, :nt] = vT[:, ::-1]
                oz[:, :nt] = z[:, ::-1]
                ovz[:, :nt] = vz[:, ::-1]
                ophi[:, :nt] = phi[:, ::-1]
            else:
                oR[:, nt - 1 :] = R
                ovR[:, nt - 1 :] = vR
                ovT[:, nt - 1 :] = vT
                oz[:, nt - 1 :] = z
                ovz[:, nt - 1 :] = vz
                ophi[:, nt - 1 :] = phi
            # load orbits
            if _firstFlip:
                os = [
                    Orbit(
                        [
                            R[ii, 0],
                            vR[ii, 0],
                            vT[ii, 0],
                            z[ii, 0],
                            vz[ii, 0],
                            phi[ii, 0],
                        ]
                    )
                    for ii in range(R.shape[0])
                ]
            else:
                os = [
                    Orbit(
                        [
                            R[ii, 0],
                            -vR[ii, 0],
                            -vT[ii, 0],
                            z[ii, 0],
                            -vz[ii, 0],
                            phi[ii, 0],
                        ]
                    )
                    for ii in range(R.shape[0])
                ]
            # integrate orbits
            [
                o.integrate(
                    self._tsJ,
                    pot=self._pot,
                    method=self._integrate_method,
                    dt=self._integrate_dt,
                )
                for o in os
            ]
            # extract phase-space points along the orbit
            ts = self._tsJ
            if _firstFlip:
                for ii in range(no):
                    oR[ii, nt:] = os[ii].R(ts[1:])  # drop t=0, which we have
                    ovR[ii, nt:] = os[ii].vR(ts[1:])  # already
                    ovT[ii, nt:] = os[ii].vT(ts[1:])  # reverse, such that
                    if os[ii].getOrbit().shape[1] == 6:
                        oz[ii, nt:] = os[ii].z(ts[1:])  # everything is in the
                        ovz[ii, nt:] = os[ii].vz(ts[1:])  # right order
                    ophi[ii, nt:] = os[ii].phi(ts[1:])  #!
            else:
                for ii in range(no):
                    oR[ii, : nt - 1] = os[ii].R(ts[1:])[::-1]  # drop t=0, which we have
                    ovR[ii, : nt - 1] = -os[ii].vR(ts[1:])[::-1]  # already
                    ovT[ii, : nt - 1] = -os[ii].vT(ts[1:])[::-1]  # reverse, such that
                    if os[ii].getOrbit().shape[1] == 6:
                        oz[ii, : nt - 1] = os[ii].z(ts[1:])[
                            ::-1
                        ]  # everything is in the
                        ovz[ii, : nt - 1] = -os[ii].vz(ts[1:])[::-1]  # right order
                    ophi[ii, : nt - 1] = os[ii].phi(ts[1:])[::-1]  #!
            return (oR, ovR, ovT, oz, ovz, ophi)
        else:
            return (R, vR, vT, z, vz, phi)


@potential_physical_input
@physical_conversion("position", pop=True)
def estimateBIsochrone(pot, R, z, phi=None):
    """
    Estimate a good value for the scale of the isochrone potential by matching the slope of the rotation curve

    Parameters
    ----------
    pot : Potential or list thereof
        Potential or list of potentials to estimate the scale of the isochrone potential for
    R : float or Quantity
        Galactocentric radius.
    z : float or Quantity
        Vertical height.
    phi : float or Quantity, optional
        Azimuth (optional).

    Returns
    -------
    float or tuple
        If a single (R,z,[phi]) is given, a single value b is returned. If a list of (R,z,[phi]) is given, a tuple (bmin,bmedian,bmax) is returned.

    Notes
    -----
    - 2013-09-12 - Written - Bovy (IAS)
    - 2016-02-20 - Changed input order to allow physical conversions - Bovy (UofT)
    - 2016-06-28 - Added phi= keyword for non-axisymmetric potential - Bovy (UofT)
    """
    if pot is None:  # pragma: no cover
        raise OSError("pot= needs to be set to a Potential instance or list thereof")
    if isinstance(R, numpy.ndarray):
        if phi is None:
            phi = [None for r in R]
        bs = numpy.array(
            [
                estimateBIsochrone(pot, R[ii], z[ii], phi=phi[ii], use_physical=False)
                for ii in range(len(R))
            ]
        )
        return numpy.array(
            [
                numpy.amin(bs[True ^ numpy.isnan(bs)]),
                numpy.median(bs[True ^ numpy.isnan(bs)]),
                numpy.amax(bs[True ^ numpy.isnan(bs)]),
            ]
        )
    else:
        r2 = R**2.0 + z**2
        r = numpy.sqrt(r2)
        dlvcdlr = (
            dvcircdR(pot, r, phi=phi, use_physical=False)
            / vcirc(pot, r, phi=phi, use_physical=False)
            * r
        )
        try:
            b = optimize.brentq(
                lambda x: dlvcdlr
                - (x / numpy.sqrt(r2 + x**2.0) - 0.5 * r2 / (r2 + x**2.0)),
                0.01,
                100.0,
            )
        except:  # pragma: no cover
            b = numpy.nan
        return b


def dePeriod(arr):
    """make an array of periodic angles increase linearly"""
    diff = arr - numpy.roll(arr, 1, axis=1)
    w = diff < -6.0
    addto = numpy.cumsum(w.astype(int), axis=1)
    return arr + _TWOPI * addto
