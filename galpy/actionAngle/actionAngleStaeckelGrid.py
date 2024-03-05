###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleStaeckelGrid
#
#             build grid in integrals of motion to quickly evaluate
#             actionAngleStaeckel
#
#      methods:
#             __call__: returns (jr,lz,jz)
#
###############################################################################
import numpy
from scipy import interpolate, ndimage, optimize

from .. import potential
from ..potential.Potential import _evaluatePotentials
from ..potential.Potential import flatten as flatten_potential
from ..util import conversion, coords, multi
from . import actionAngleStaeckel, actionAngleStaeckel_c
from .actionAngle import actionAngle
from .actionAngleStaeckel_c import _ext_loaded as ext_loaded

_PRINTOUTSIDEGRID = False


class actionAngleStaeckelGrid(actionAngle):
    """Action-angle formalism for axisymmetric potentials using Binney (2012)'s Staeckel approximation, grid-based interpolation"""

    def __init__(
        self,
        pot=None,
        delta=None,
        Rmax=5.0,
        nE=25,
        npsi=25,
        nLz=30,
        numcores=1,
        interpecc=False,
        **kwargs,
    ):
        """
        Initialize an actionAngleStaeckelGrid object

        Parameters
        ----------
        pot : Potential or list of Potential instances
            The potential or list of potentials to use for the actionAngleStaeckelGrid object.
        delta : float or Quantity
            The focal length of the confocal coordinate system.
        Rmax : float
            The maximum R to consider (natural units).
        nE : int
            The number of grid points in energy.
        npsi : int
            The number of grid points in psi.
        nLz : int
            The number of grid points in Lz.
        numcores : int
            The number of cores to use for multi-processing.
        interpecc : bool
            If True, also interpolate the approximate eccentricity, zmax, rperi, and rapo.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2012-11-29 - Written - Bovy (IAS)
        - 2017-12-15 - Written - Bovy (UofT)
        """
        actionAngle.__init__(self, ro=kwargs.get("ro", None), vo=kwargs.get("vo", None))
        if pot is None:
            raise OSError("Must specify pot= for actionAngleStaeckelGrid")
        self._pot = flatten_potential(pot)
        if delta is None:
            raise OSError("Must specify delta= for actionAngleStaeckelGrid")
        if ext_loaded and "c" in kwargs and kwargs["c"]:
            self._c = True
        else:
            self._c = False
        self._delta = conversion.parse_length(delta, ro=self._ro)
        self._Rmax = Rmax
        self._Rmin = 0.01
        # Set up the actionAngleStaeckel object that we will use to interpolate
        self._aA = actionAngleStaeckel.actionAngleStaeckel(
            pot=self._pot, delta=self._delta, c=self._c
        )
        # Build grid
        self._Lzmin = 0.01
        self._Lzs = numpy.linspace(
            self._Lzmin, self._Rmax * potential.vcirc(self._pot, self._Rmax), nLz
        )
        self._Lzmax = self._Lzs[-1]
        self._nLz = nLz
        # Calculate E_c(R=RL), energy of circular orbit
        self._RL = numpy.array([potential.rl(self._pot, l) for l in self._Lzs])
        self._RLInterp = interpolate.InterpolatedUnivariateSpline(
            self._Lzs, self._RL, k=3
        )
        self._ERL = (
            _evaluatePotentials(self._pot, self._RL, numpy.zeros(self._nLz))
            + self._Lzs**2.0 / 2.0 / self._RL**2.0
        )
        self._ERLmax = numpy.amax(self._ERL) + 1.0
        self._ERLInterp = interpolate.InterpolatedUnivariateSpline(
            self._Lzs, numpy.log(-(self._ERL - self._ERLmax)), k=3
        )
        self._Ramax = 200.0 / 8.0
        self._ERa = (
            _evaluatePotentials(self._pot, self._Ramax, 0.0)
            + self._Lzs**2.0 / 2.0 / self._Ramax**2.0
        )
        # self._EEsc= numpy.array([self._ERL[ii]+potential.vesc(self._pot,self._RL[ii])**2./4. for ii in range(nLz)])
        self._ERamax = numpy.amax(self._ERa) + 1.0
        self._ERaInterp = interpolate.InterpolatedUnivariateSpline(
            self._Lzs, numpy.log(-(self._ERa - self._ERamax)), k=3
        )
        y = numpy.linspace(0.0, 1.0, nE)
        self._nE = nE
        psis = numpy.linspace(0.0, 1.0, npsi) * numpy.pi / 2.0
        self._npsi = npsi
        jr = numpy.zeros((nLz, nE, npsi))
        jz = numpy.zeros((nLz, nE, npsi))
        u0 = numpy.zeros((nLz, nE))
        jrLzE = numpy.zeros(nLz)
        jzLzE = numpy.zeros(nLz)
        # First calculate u0
        thisLzs = (numpy.tile(self._Lzs, (nE, 1)).T).flatten()
        thisERL = (numpy.tile(self._ERL, (nE, 1)).T).flatten()
        thisERa = (numpy.tile(self._ERa, (nE, 1)).T).flatten()
        this = (numpy.tile(y, (nLz, 1))).flatten()
        thisE = _invEfunc(
            _Efunc(thisERa, thisERL)
            + this * (_Efunc(thisERL, thisERL) - _Efunc(thisERa, thisERL)),
            thisERL,
        )
        if isinstance(self._pot, potential.interpRZPotential) and hasattr(
            self._pot, "_origPot"
        ):
            u0pot = self._pot._origPot
        else:
            u0pot = self._pot
        if self._c:
            mu0 = actionAngleStaeckel_c.actionAngleStaeckel_calcu0(
                thisE, thisLzs, u0pot, self._delta
            )[0]
        else:
            if numcores > 1:
                mu0 = multi.parallel_map(
                    (lambda x: self.calcu0(thisE[x], thisLzs[x])),
                    range(nE * nLz),
                    numcores=numcores,
                )
            else:
                mu0 = list(
                    map((lambda x: self.calcu0(thisE[x], thisLzs[x])), range(nE * nLz))
                )
        u0 = numpy.reshape(mu0, (nLz, nE))
        thisR = self._delta * numpy.sinh(u0)
        thisv = numpy.reshape(
            self.vatu0(
                thisE.flatten(), thisLzs.flatten(), u0.flatten(), thisR.flatten()
            ),
            (nLz, nE),
        )
        self.thisv = thisv
        # reshape
        thisLzs = numpy.reshape(thisLzs, (nLz, nE))
        thispsi = numpy.tile(psis, (nLz, nE, 1)).flatten()
        thisLzs = numpy.tile(thisLzs.T, (npsi, 1, 1)).T.flatten()
        thisR = numpy.tile(thisR.T, (npsi, 1, 1)).T.flatten()
        thisv = numpy.tile(thisv.T, (npsi, 1, 1)).T.flatten()
        mjr, mlz, mjz = self._aA(
            thisR,  # R
            thisv * numpy.cos(thispsi),  # vR
            thisLzs / thisR,  # vT
            numpy.zeros(len(thisR)),  # z
            thisv * numpy.sin(thispsi),  # vz
            fixed_quad=True,
        )
        if interpecc:
            mecc, mzmax, mrperi, mrap = self._aA.EccZmaxRperiRap(
                thisR,  # R
                thisv * numpy.cos(thispsi),  # vR
                thisLzs / thisR,  # vT
                numpy.zeros(len(thisR)),  # z
                thisv * numpy.sin(thispsi),
            )  # vz
        if isinstance(self._pot, potential.interpRZPotential) and hasattr(
            self._pot, "_origPot"
        ):
            # Interpolated potentials have problems with extreme orbits
            indx = mjr == 9999.99
            indx += mjz == 9999.99
            # Re-calculate these using the original potential, hopefully not too slow
            tmpaA = actionAngleStaeckel.actionAngleStaeckel(
                pot=self._pot._origPot, delta=self._delta, c=self._c
            )
            mjr[indx], dumb, mjz[indx] = tmpaA(
                thisR[indx],  # R
                thisv[indx] * numpy.cos(thispsi[indx]),  # vR
                thisLzs[indx] / thisR[indx],  # vT
                numpy.zeros(numpy.sum(indx)),  # z
                thisv[indx] * numpy.sin(thispsi[indx]),  # vz
                fixed_quad=True,
            )
            if interpecc:
                (
                    mecc[indx],
                    mzmax[indx],
                    mrperi[indx],
                    mrap[indx],
                ) = self._aA.EccZmaxRperiRap(
                    thisR[indx],  # R
                    thisv[indx] * numpy.cos(thispsi[indx]),  # vR
                    thisLzs[indx] / thisR[indx],  # vT
                    numpy.zeros(numpy.sum(indx)),  # z
                    thisv[indx] * numpy.sin(thispsi[indx]),
                )  # vz
        jr = numpy.reshape(mjr, (nLz, nE, npsi))
        jz = numpy.reshape(mjz, (nLz, nE, npsi))
        if interpecc:
            ecc = numpy.reshape(mecc, (nLz, nE, npsi))
            zmax = numpy.reshape(mzmax, (nLz, nE, npsi))
            rperi = numpy.reshape(mrperi, (nLz, nE, npsi))
            rap = numpy.reshape(mrap, (nLz, nE, npsi))
            zmaxLzE = numpy.zeros(nLz)
            rperiLzE = numpy.zeros(nLz)
            rapLzE = numpy.zeros(nLz)
        for ii in range(nLz):
            jrLzE[ii] = numpy.nanmax(jr[ii, (jr[ii, :, :] != 9999.99)])  #:,:])
            jzLzE[ii] = numpy.nanmax(jz[ii, (jz[ii, :, :] != 9999.99)])  #:,:])
            if interpecc:
                zmaxLzE[ii] = numpy.amax(zmax[ii, numpy.isfinite(zmax[ii])])
                rperiLzE[ii] = numpy.amax(rperi[ii, numpy.isfinite(rperi[ii])])
                rapLzE[ii] = numpy.amax(rap[ii, numpy.isfinite(rap[ii])])
        jrLzE[(jrLzE == 0.0)] = numpy.nanmin(jrLzE[(jrLzE > 0.0)])
        jzLzE[(jzLzE == 0.0)] = numpy.nanmin(jzLzE[(jzLzE > 0.0)])
        if interpecc:
            zmaxLzE[(zmaxLzE == 0.0)] = numpy.nanmin(zmaxLzE[(zmaxLzE > 0.0)])
            rperiLzE[(rperiLzE == 0.0)] = numpy.nanmin(rperiLzE[(rperiLzE > 0.0)])
            rapLzE[(rapLzE == 0.0)] = numpy.nanmin(rapLzE[(rapLzE > 0.0)])
        for ii in range(nLz):
            jr[ii, :, :] /= jrLzE[ii]
            jz[ii, :, :] /= jzLzE[ii]
            if interpecc:
                zmax[ii, :, :] /= zmaxLzE[ii]
                rperi[ii, :, :] /= rperiLzE[ii]
                rap[ii, :, :] /= rapLzE[ii]
        # Deal w/ 9999.99
        jr[(jr > 1.0)] = 1.0
        jz[(jz > 1.0)] = 1.0
        # Deal w/ NaN
        jr[numpy.isnan(jr)] = 0.0
        jz[numpy.isnan(jz)] = 0.0
        if interpecc:
            ecc[(ecc < 0.0)] = 0.0
            ecc[(ecc > 1.0)] = 1.0
            ecc[numpy.isnan(ecc)] = 0.0
            ecc[numpy.isinf(ecc)] = 1.0
            zmax[(zmax > 1.0)] = 1.0
            zmax[numpy.isnan(zmax)] = 0.0
            zmax[numpy.isinf(zmax)] = 1.0
            rperi[(rperi > 1.0)] = 1.0
            rperi[numpy.isnan(rperi)] = 0.0
            rperi[numpy.isinf(rperi)] = 0.0  # typically orbits that can reach 0
            rap[(rap > 1.0)] = 1.0
            rap[numpy.isnan(rap)] = 0.0
            rap[numpy.isinf(rap)] = 1.0
        # First interpolate the maxima
        self._jr = jr
        self._jz = jz
        self._u0 = u0
        self._jrLzE = jrLzE
        self._jzLzE = jzLzE
        self._jrLzInterp = interpolate.InterpolatedUnivariateSpline(
            self._Lzs, numpy.log(jrLzE + 10.0**-5.0), k=3
        )
        self._jzLzInterp = interpolate.InterpolatedUnivariateSpline(
            self._Lzs, numpy.log(jzLzE + 10.0**-5.0), k=3
        )
        if interpecc:
            self._ecc = ecc
            self._zmax = zmax
            self._rperi = rperi
            self._rap = rap
            self._zmaxLzE = zmaxLzE
            self._rperiLzE = rperiLzE
            self._rapLzE = rapLzE
            self._zmaxLzInterp = interpolate.InterpolatedUnivariateSpline(
                self._Lzs, numpy.log(zmaxLzE + 10.0**-5.0), k=3
            )
            self._rperiLzInterp = interpolate.InterpolatedUnivariateSpline(
                self._Lzs, numpy.log(rperiLzE + 10.0**-5.0), k=3
            )
            self._rapLzInterp = interpolate.InterpolatedUnivariateSpline(
                self._Lzs, numpy.log(rapLzE + 10.0**-5.0), k=3
            )
        # Interpolate u0
        self._logu0Interp = interpolate.RectBivariateSpline(
            self._Lzs, y, numpy.log(u0), kx=3, ky=3, s=0.0
        )
        # spline filter jr and jz, such that they can be used with ndimage.map_coordinates
        self._jrFiltered = ndimage.spline_filter(
            numpy.log(self._jr + 10.0**-10.0), order=3
        )
        self._jzFiltered = ndimage.spline_filter(
            numpy.log(self._jz + 10.0**-10.0), order=3
        )
        if interpecc:
            self._eccFiltered = ndimage.spline_filter(
                numpy.log(self._ecc + 10.0**-10.0), order=3
            )
            self._zmaxFiltered = ndimage.spline_filter(
                numpy.log(self._zmax + 10.0**-10.0), order=3
            )
            self._rperiFiltered = ndimage.spline_filter(
                numpy.log(self._rperi + 10.0**-10.0), order=3
            )
            self._rapFiltered = ndimage.spline_filter(
                numpy.log(self._rap + 10.0**-10.0), order=3
            )
        # Check the units
        self._check_consistent_units()
        return None

    def _evaluate(self, *args, **kwargs):
        """
        Evaluate the actions (jr,lz,jz)

        Parameters
        ----------
        *args : tuple
            Either:
            a) R,vR,vT,z,vz[,phi]:
                1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)
                2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
            b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
        **kwargs: dict, optional
            Keywords for actionAngleStaeckel.__call__ for off-the-grid evaluations

        Returns
        -------
        tuple
            (jr,lz,jz)

        Notes
        -----
        - 2012-11-29 - Written - Bovy (IAS)
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
        Lz = R * vT
        Phi = _evaluatePotentials(self._pot, R, z)
        E = Phi + vR**2.0 / 2.0 + vT**2.0 / 2.0 + vz**2.0 / 2.0
        thisERL = -numpy.exp(self._ERLInterp(Lz)) + self._ERLmax
        thisERa = -numpy.exp(self._ERaInterp(Lz)) + self._ERamax
        if isinstance(R, numpy.ndarray):
            indx = ((E - thisERa) / (thisERL - thisERa) > 1.0) * (
                ((E - thisERa) / (thisERL - thisERa) - 1.0) < 10.0**-2.0
            )
            E[indx] = thisERL[indx]
            indx = ((E - thisERa) / (thisERL - thisERa) < 0.0) * (
                (E - thisERa) / (thisERL - thisERa) > -(10.0**-2.0)
            )
            E[indx] = thisERa[indx]
            indx = Lz < self._Lzmin
            indx += Lz > self._Lzmax
            indx += (E - thisERa) / (thisERL - thisERa) > 1.0
            indx += (E - thisERa) / (thisERL - thisERa) < 0.0
            indxc = True ^ indx
            jr = numpy.empty(R.shape)
            jz = numpy.empty(R.shape)
            if numpy.sum(indxc) > 0:
                u0 = numpy.exp(
                    self._logu0Interp.ev(
                        Lz[indxc],
                        (
                            _Efunc(E[indxc], thisERL[indxc])
                            - _Efunc(thisERa[indxc], thisERL[indxc])
                        )
                        / (
                            _Efunc(thisERL[indxc], thisERL[indxc])
                            - _Efunc(thisERa[indxc], thisERL[indxc])
                        ),
                    )
                )
                sinh2u0 = numpy.sinh(u0) ** 2.0
                thisEr = self.Er(
                    R[indxc],
                    z[indxc],
                    vR[indxc],
                    vz[indxc],
                    E[indxc],
                    Lz[indxc],
                    sinh2u0,
                    u0,
                )
                thisEz = self.Ez(
                    R[indxc],
                    z[indxc],
                    vR[indxc],
                    vz[indxc],
                    E[indxc],
                    Lz[indxc],
                    sinh2u0,
                    u0,
                )
                thisv2 = self.vatu0(
                    E[indxc], Lz[indxc], u0, self._delta * numpy.sinh(u0), retv2=True
                )
                cos2psi = 2.0 * thisEr / thisv2 / (1.0 + sinh2u0)  # latter is cosh2u0
                cos2psi[(cos2psi > 1.0) * (cos2psi < 1.0 + 10.0**-5.0)] = 1.0
                indxCos2psi = cos2psi > 1.0
                indxCos2psi += cos2psi < 0.0
                indxc[indxc] = True ^ indxCos2psi  # Handle these two cases as off-grid
                indx = True ^ indxc
                psi = numpy.arccos(numpy.sqrt(cos2psi[True ^ indxCos2psi]))
                coords = numpy.empty((3, numpy.sum(indxc)))
                coords[0, :] = (
                    (Lz[indxc] - self._Lzmin)
                    / (self._Lzmax - self._Lzmin)
                    * (self._nLz - 1.0)
                )
                y = (
                    _Efunc(E[indxc], thisERL[indxc])
                    - _Efunc(thisERa[indxc], thisERL[indxc])
                ) / (
                    _Efunc(thisERL[indxc], thisERL[indxc])
                    - _Efunc(thisERa[indxc], thisERL[indxc])
                )
                coords[1, :] = y * (self._nE - 1.0)
                coords[2, :] = psi / numpy.pi * 2.0 * (self._npsi - 1.0)
                jr[indxc] = (
                    numpy.exp(
                        ndimage.map_coordinates(
                            self._jrFiltered, coords, order=3, prefilter=False
                        )
                    )
                    - 10.0**-10.0
                ) * (numpy.exp(self._jrLzInterp(Lz[indxc])) - 10.0**-5.0)
                # Switch to Ez-calculated psi
                sin2psi = (
                    2.0
                    * thisEz[True ^ indxCos2psi]
                    / thisv2[True ^ indxCos2psi]
                    / (1.0 + sinh2u0[True ^ indxCos2psi])
                )  # latter is cosh2u0
                sin2psi[(sin2psi > 1.0) * (sin2psi < 1.0 + 10.0**-5.0)] = 1.0
                indxSin2psi = sin2psi > 1.0
                indxSin2psi += sin2psi < 0.0
                indxc[indxc] = True ^ indxSin2psi  # Handle these two cases as off-grid
                indx = True ^ indxc
                psiz = numpy.arcsin(numpy.sqrt(sin2psi[True ^ indxSin2psi]))
                newcoords = numpy.empty((3, numpy.sum(indxc)))
                newcoords[0:2, :] = coords[0:2, True ^ indxSin2psi]
                newcoords[2, :] = psiz / numpy.pi * 2.0 * (self._npsi - 1.0)
                jz[indxc] = (
                    numpy.exp(
                        ndimage.map_coordinates(
                            self._jzFiltered, newcoords, order=3, prefilter=False
                        )
                    )
                    - 10.0**-10.0
                ) * (numpy.exp(self._jzLzInterp(Lz[indxc])) - 10.0**-5.0)
            if numpy.sum(indx) > 0:
                jrindiv, lzindiv, jzindiv = self._aA(
                    R[indx], vR[indx], vT[indx], z[indx], vz[indx], **kwargs
                )
                jr[indx] = jrindiv
                jz[indx] = jzindiv
                """
                jrindiv= numpy.empty(numpy.sum(indx))
                jzindiv= numpy.empty(numpy.sum(indx))
                for ii in range(numpy.sum(indx)):
                    try:
                        thisaA= actionAngleStaeckel.actionAngleStaeckelSingle(\
                            R[indx][ii], #R
                            vR[indx][ii], #vR
                            vT[indx][ii], #vT
                            z[indx][ii], #z
                            vz[indx][ii], #vz
                            pot=self._pot,delta=self._delta)
                        jrindiv[ii]= thisaA.JR(fixed_quad=True)[0]
                        jzindiv[ii]= thisaA.Jz(fixed_quad=True)[0]
                    except (UnboundError,OverflowError):
                        jrindiv[ii]= numpy.nan
                        jzindiv[ii]= numpy.nan
                jr[indx]= jrindiv
                jz[indx]= jzindiv
                """
        else:
            jr, Lz, jz = self(
                numpy.array([R]),
                numpy.array([vR]),
                numpy.array([vT]),
                numpy.array([z]),
                numpy.array([vz]),
                **kwargs,
            )
            return (jr[0], Lz[0], jz[0])
        jr[jr < 0.0] = 0.0
        jz[jz < 0.0] = 0.0
        return (jr, R * vT, jz)

    def Jz(self, *args, **kwargs):
        """
        Evaluate the action jz

        Parameters
        ----------
        *args : tuple
            Either:
            a) R,vR,vT,z,vz[,phi]:
                1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)
                2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
            b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
        **kwargs: dict, optional
            Keywords for actionAngleStaeckel.__call__ for off-the-grid evaluations

        Returns
        -------
        float or numpy.ndarray
            jz

        Notes
        -----
        - 2012-07-30 - Written - Bovy (IAS@MPIA)
        """
        return self(*args, **kwargs)[2]

    def JR(self, *args, **kwargs):
        """
        Evaluate the action jr.

        Parameters
        ----------
        *args : tuple
            Either:
            a) R,vR,vT,z,vz[,phi]:
                1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)
                2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
            b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
        **kwargs: dict, optional
            Keywords for actionAngleStaeckel.__call__ for off-the-grid evaluations


        Returns
        -------
        float
               The action jr.

        Notes
        -----
        - 2012-07-30 - Written - Bovy (IAS@MPIA)
        """
        return self(*args, **kwargs)[0]

    def _EccZmaxRperiRap(self, *args, **kwargs):
        """
        Evaluate the eccentricity, maximum height above the plane, peri- and apocenter in the Staeckel approximation

        Parameters
        ----------
        *args : tuple
            Either:
            a) R,vR,vT,z,vz[,phi]:
                1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)
                2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
            b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument

        Returns
        -------
        tuple
            (e,zmax,rperi,rap)
        Notes
        -----
        - 2017-12-15 - Written - Bovy (UofT)
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
        Lz = R * vT
        Phi = _evaluatePotentials(self._pot, R, z)
        E = Phi + vR**2.0 / 2.0 + vT**2.0 / 2.0 + vz**2.0 / 2.0
        thisERL = -numpy.exp(self._ERLInterp(Lz)) + self._ERLmax
        thisERa = -numpy.exp(self._ERaInterp(Lz)) + self._ERamax
        if isinstance(R, numpy.ndarray):
            indx = ((E - thisERa) / (thisERL - thisERa) > 1.0) * (
                ((E - thisERa) / (thisERL - thisERa) - 1.0) < 10.0**-2.0
            )
            E[indx] = thisERL[indx]
            indx = ((E - thisERa) / (thisERL - thisERa) < 0.0) * (
                (E - thisERa) / (thisERL - thisERa) > -(10.0**-2.0)
            )
            E[indx] = thisERa[indx]
            indx = Lz < self._Lzmin
            indx += Lz > self._Lzmax
            indx += (E - thisERa) / (thisERL - thisERa) > 1.0
            indx += (E - thisERa) / (thisERL - thisERa) < 0.0
            indxc = True ^ indx
            ecc = numpy.empty(R.shape)
            zmax = numpy.empty(R.shape)
            rperi = numpy.empty(R.shape)
            rap = numpy.empty(R.shape)
            if numpy.sum(indxc) > 0:
                u0 = numpy.exp(
                    self._logu0Interp.ev(
                        Lz[indxc],
                        (
                            _Efunc(E[indxc], thisERL[indxc])
                            - _Efunc(thisERa[indxc], thisERL[indxc])
                        )
                        / (
                            _Efunc(thisERL[indxc], thisERL[indxc])
                            - _Efunc(thisERa[indxc], thisERL[indxc])
                        ),
                    )
                )
                sinh2u0 = numpy.sinh(u0) ** 2.0
                thisEr = self.Er(
                    R[indxc],
                    z[indxc],
                    vR[indxc],
                    vz[indxc],
                    E[indxc],
                    Lz[indxc],
                    sinh2u0,
                    u0,
                )
                thisEz = self.Ez(
                    R[indxc],
                    z[indxc],
                    vR[indxc],
                    vz[indxc],
                    E[indxc],
                    Lz[indxc],
                    sinh2u0,
                    u0,
                )
                thisv2 = self.vatu0(
                    E[indxc], Lz[indxc], u0, self._delta * numpy.sinh(u0), retv2=True
                )
                cos2psi = 2.0 * thisEr / thisv2 / (1.0 + sinh2u0)  # latter is cosh2u0
                cos2psi[(cos2psi > 1.0) * (cos2psi < 1.0 + 10.0**-5.0)] = 1.0
                indxCos2psi = cos2psi > 1.0
                indxCos2psi += cos2psi < 0.0
                indxc[indxc] = True ^ indxCos2psi  # Handle these two cases as off-grid
                indx = True ^ indxc
                psi = numpy.arccos(numpy.sqrt(cos2psi[True ^ indxCos2psi]))
                coords = numpy.empty((3, numpy.sum(indxc)))
                coords[0, :] = (
                    (Lz[indxc] - self._Lzmin)
                    / (self._Lzmax - self._Lzmin)
                    * (self._nLz - 1.0)
                )
                y = (
                    _Efunc(E[indxc], thisERL[indxc])
                    - _Efunc(thisERa[indxc], thisERL[indxc])
                ) / (
                    _Efunc(thisERL[indxc], thisERL[indxc])
                    - _Efunc(thisERa[indxc], thisERL[indxc])
                )
                coords[1, :] = y * (self._nE - 1.0)
                coords[2, :] = psi / numpy.pi * 2.0 * (self._npsi - 1.0)
                ecc[indxc] = (
                    numpy.exp(
                        ndimage.map_coordinates(
                            self._eccFiltered, coords, order=3, prefilter=False
                        )
                    )
                    - 10.0**-10.0
                )
                rperi[indxc] = (
                    numpy.exp(
                        ndimage.map_coordinates(
                            self._rperiFiltered, coords, order=3, prefilter=False
                        )
                    )
                    - 10.0**-10.0
                ) * (numpy.exp(self._rperiLzInterp(Lz[indxc])) - 10.0**-5.0)
                # We do rap below with zmax
                # Switch to Ez-calculated psi
                sin2psi = (
                    2.0
                    * thisEz[True ^ indxCos2psi]
                    / thisv2[True ^ indxCos2psi]
                    / (1.0 + sinh2u0[True ^ indxCos2psi])
                )  # latter is cosh2u0
                sin2psi[(sin2psi > 1.0) * (sin2psi < 1.0 + 10.0**-5.0)] = 1.0
                indxSin2psi = sin2psi > 1.0
                indxSin2psi += sin2psi < 0.0
                indxc[indxc] = True ^ indxSin2psi  # Handle these two cases as off-grid
                indx = True ^ indxc
                psiz = numpy.arcsin(numpy.sqrt(sin2psi[True ^ indxSin2psi]))
                newcoords = numpy.empty((3, numpy.sum(indxc)))
                newcoords[0:2, :] = coords[0:2, True ^ indxSin2psi]
                newcoords[2, :] = psiz / numpy.pi * 2.0 * (self._npsi - 1.0)
                zmax[indxc] = (
                    numpy.exp(
                        ndimage.map_coordinates(
                            self._zmaxFiltered, newcoords, order=3, prefilter=False
                        )
                    )
                    - 10.0**-10.0
                ) * (numpy.exp(self._zmaxLzInterp(Lz[indxc])) - 10.0**-5.0)
                rap[indxc] = (
                    numpy.exp(
                        ndimage.map_coordinates(
                            self._rapFiltered, newcoords, order=3, prefilter=False
                        )
                    )
                    - 10.0**-10.0
                ) * (numpy.exp(self._rapLzInterp(Lz[indxc])) - 10.0**-5.0)
            if numpy.sum(indx) > 0:
                eccindiv, zmaxindiv, rperiindiv, rapindiv = self._aA.EccZmaxRperiRap(
                    R[indx], vR[indx], vT[indx], z[indx], vz[indx], **kwargs
                )
                ecc[indx] = eccindiv
                zmax[indx] = zmaxindiv
                rperi[indx] = rperiindiv
                rap[indx] = rapindiv
        else:
            ecc, zmax, rperi, rap = self.EccZmaxRperiRap(
                numpy.array([R]),
                numpy.array([vR]),
                numpy.array([vT]),
                numpy.array([z]),
                numpy.array([vz]),
                **kwargs,
            )
            return (ecc[0], zmax[0], rperi[0], rap[0])
        ecc[ecc < 0.0] = 0.0
        zmax[zmax < 0.0] = 0.0
        rperi[rperi < 0.0] = 0.0
        rap[rap < 0.0] = 0.0
        return (ecc, zmax, rperi, rap)

    def vatu0(self, E, Lz, u0, R, retv2=False):
        """
        Calculate the velocity at u0.

        Parameters
        ----------
        E : float
            Energy.
        Lz : float
            Angular momentum.
        u0 : float
            u0.
        R : float
            Radius corresponding to u0, pi/2.
        retv2 : bool, optional
            If True, return v^2. Default is False.

        Returns
        -------
        float or numpy.ndarray
            Velocity or velocity squared if retv2 is True.

        Notes
        -----
        - 2012-11-29 - Written - Bovy (IAS).
        """
        v2 = (
            2.0
            * (
                E
                - actionAngleStaeckel.potentialStaeckel(
                    u0, numpy.pi / 2.0, self._pot, self._delta
                )
            )
            - Lz**2.0 / R**2.0
        )
        if retv2:
            return v2
        v2[(v2 < 0.0) * (v2 > -(10.0**-7.0))] = 0.0
        return numpy.sqrt(v2)

    def calcu0(self, E, Lz):
        """
        Calculate the minimum of the u potential.

        Parameters
        ----------
        E : float
            Energy.
        Lz : float
            Angular momentum.

        Returns
        -------
        float
            Minimum of the u potential.

        Notes
        -----
        - 2012-11-29 - Written - Bovy (IAS)
        """
        logu0 = optimize.brent(_u0Eq, args=(self._delta, self._pot, E, Lz**2.0 / 2.0))
        return numpy.exp(logu0)

    def Er(self, R, z, vR, vz, E, Lz, sinh2u0, u0):
        """
        Calculate the 'radial energy'

        Parameters
        ----------
        R : float
            Galactocentric cylindrical radius
        z : float
            vertical height
        vR : float
            Galactocentric radial velocity
        vz : float
            vertical velocity
        E : float
            energy
        Lz : float
            angular momentum
        sinh2u0 : float
            sinh^2 and u0
        u0 : float
            sinh^2 and u0

        Returns
        -------
        float
            radial energy

        Notes
        -----
        - 2012-11-29 - Written - Bovy (IAS).
        """
        u, v = coords.Rz_to_uv(R, z, self._delta)
        pu = vR * numpy.cosh(u) * numpy.sin(v) + vz * numpy.sinh(u) * numpy.cos(
            v
        )  # no delta, bc we will divide it out
        out = (
            pu**2.0 / 2.0
            + Lz**2.0
            / 2.0
            / self._delta**2.0
            * (1.0 / numpy.sinh(u) ** 2.0 - 1.0 / sinh2u0)
            - E * (numpy.sinh(u) ** 2.0 - sinh2u0)
            + (numpy.sinh(u) ** 2.0 + 1.0)
            * actionAngleStaeckel.potentialStaeckel(
                u, numpy.pi / 2.0, self._pot, self._delta
            )
            - (sinh2u0 + 1.0)
            * actionAngleStaeckel.potentialStaeckel(
                u0, numpy.pi / 2.0, self._pot, self._delta
            )
        )
        #              +(numpy.sinh(u)**2.+numpy.sin(v)**2.)*actionAngleStaeckel.potentialStaeckel(u,v,self._pot,self._delta)
        #              -(sinh2u0+numpy.sin(v)**2.)*actionAngleStaeckel.potentialStaeckel(u0,v,self._pot,self._delta))
        return out

    def Ez(self, R, z, vR, vz, E, Lz, sinh2u0, u0):
        """
        Calculate the 'vertical energy'

        Parameters
        ----------
        R : float
            Galactocentric cylindrical radius (can be Quantity)
        z : float
            height above the plane (can be Quantity)
        vR : float
            Galactocentric radial velocity (can be Quantity)
        vz : float
            Galactocentric vertical velocity (can be Quantity)
        E : float
            energy
        Lz : float
            angular momentum
        sinh2u0 : float
            sinh^2 and u0
        u0 : float
            sinh^2 and u0

        Returns
        -------
        float
            vertical energy

        Notes
        -----
        - 2012-12-23 - Written - Bovy (IAS)
        """
        u, v = coords.Rz_to_uv(R, z, self._delta)
        pv = vR * numpy.sinh(u) * numpy.cos(v) - vz * numpy.cosh(u) * numpy.sin(
            v
        )  # no delta, bc we will divide it out
        out = (
            pv**2.0 / 2.0
            + Lz**2.0 / 2.0 / self._delta**2.0 * (1.0 / numpy.sin(v) ** 2.0 - 1.0)
            - E * (numpy.sin(v) ** 2.0 - 1.0)
            - (sinh2u0 + 1.0)
            * actionAngleStaeckel.potentialStaeckel(
                u0, numpy.pi / 2.0, self._pot, self._delta
            )
            + (sinh2u0 + numpy.sin(v) ** 2.0)
            * actionAngleStaeckel.potentialStaeckel(u0, v, self._pot, self._delta)
        )
        return out


def _u0Eq(logu, delta, pot, E, Lz22):
    """The equation that needs to be minimized to find u0"""
    u = numpy.exp(logu)
    sinh2u = numpy.sinh(u) ** 2.0
    cosh2u = numpy.cosh(u) ** 2.0
    dU = cosh2u * actionAngleStaeckel.potentialStaeckel(u, numpy.pi / 2.0, pot, delta)
    return -(E * sinh2u - dU - Lz22 / delta**2.0 / sinh2u)


def _Efunc(E, *args):
    """Function to apply to the energy in building the grid (e.g., if this is a log, then the grid will be logarithmic"""
    #    return ((E-args[0]))**0.5
    return numpy.log(E - args[0] + 10.0**-10.0)


def _invEfunc(Ef, *args):
    """Inverse of Efunc"""
    #    return Ef**2.+args[0]
    return numpy.exp(Ef) + args[0] - 10.0**-10.0
