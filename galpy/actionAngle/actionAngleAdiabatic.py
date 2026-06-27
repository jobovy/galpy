###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleAdiabatic
#
#      methods:
#             __call__: returns (jr,lz,jz)
#             _EccZmaxRperiRap: return (e,zmax,rperi,rap)
#
###############################################################################
import copy
import warnings

import numpy

from ..backend import get_namespace, is_backend_array
from ..potential import MWPotential, toPlanarPotential, toVerticalPotential
from ..potential.Potential import (
    _check_c,
    _check_potential_list_and_deprecate,
    _dim,
    _evaluatePotentials,
)
from ..potential.verticalPotential import _BatchedVerticalPotential
from ..util import galpyWarning
from . import actionAngleAdiabatic_c
from .actionAngle import actionAngle
from .actionAngleAdiabatic_c import _ext_loaded as ext_loaded
from .actionAngleSpherical import actionAngleSpherical
from .actionAngleVertical import actionAngleVertical


class actionAngleAdiabatic(actionAngle):
    """Action-angle formalism for axisymmetric potentials using the adiabatic approximation"""

    def __init__(self, *args, **kwargs):
        """
        Initialize an actionAngleAdiabatic object.

        Parameters
        ----------
        pot : potential or a combined potential formed using addition (pot1+pot2+…)
            The potential or a combined potential formed using addition (pot1+pot2+…).
        gamma : float, optional
            Replace Lz by Lz+gamma Jz in effective potential. Default is 1.0.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2012-07-26 - Written - Bovy (IAS@MPIA).
        """
        actionAngle.__init__(self, ro=kwargs.get("ro", None), vo=kwargs.get("vo", None))
        if not "pot" in kwargs:  # pragma: no cover
            raise OSError("Must specify pot= for actionAngleAdiabatic")
        self._pot = _check_potential_list_and_deprecate(kwargs["pot"])
        if self._pot == MWPotential:
            warnings.warn(
                "Use of MWPotential as a Milky-Way-like potential is deprecated; galpy.potential.MWPotential2014, a potential fit to a large variety of dynamical constraints (see Bovy 2015), is the preferred Milky-Way-like potential in galpy",
                galpyWarning,
            )
        if ext_loaded and "c" in kwargs and kwargs["c"]:
            self._c = _check_c(self._pot)
            if "c" in kwargs and kwargs["c"] and not self._c:
                warnings.warn(
                    "C module not used because potential does not have a C implementation",
                    galpyWarning,
                )  # pragma: no cover
        else:
            self._c = False
        self._gamma = kwargs.get("gamma", 1.0)
        # Setup actionAngleSpherical object for calculations in Python
        # (if they become necessary)
        if _dim(self._pot) == 3:
            thispot = toPlanarPotential(self._pot)
        else:
            thispot = self._pot
            self._gamma = 0.0
        self._aAS = actionAngleSpherical(pot=thispot, _gamma=self._gamma)
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
        c: bool, optional
            True/False to override the object-wide setting for whether or not to use the C implementation
        _justjr, _justjz: bool, optional
            If True, only calculate the radial or vertical action (internal use)
        **kwargs : dict
            scipy.integrate.quadrature keywords

        Returns
        -------
        (jr,lz,jz)
            Actions (jr,lz,jz).

        Notes
        -----
        - 2012-07-26 - Written - Bovy (IAS@MPIA).
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
        if isinstance(R, float):
            R = numpy.array([R])
            vR = numpy.array([vR])
            vT = numpy.array([vT])
            z = numpy.array([z])
            vz = numpy.array([vz])
        if is_backend_array(R):
            # jax/torch inputs: vectorised, differentiable path (see backend
            # section below); processes all N objects at once. numpy stays below.
            return self._evaluate_backend(R, vR, vT, z, vz)
        if (
            (self._c and not ("c" in kwargs and not kwargs["c"]))
            or (ext_loaded and ("c" in kwargs and kwargs["c"]))
        ) and _check_c(self._pot):
            Lz = R * vT
            jr, jz, err = actionAngleAdiabatic_c.actionAngleAdiabatic_c(
                self._pot, self._gamma, R, vR, vT, z, vz
            )
            if err == 0:
                return (jr, Lz, jz)
            else:  # pragma: no cover
                raise RuntimeError(
                    "C-code for calculation actions failed; try with c=False"
                )
        else:
            if "c" in kwargs and kwargs["c"] and not self._c:
                warnings.warn(
                    "C module not used because potential does not have a C implementation",
                    galpyWarning,
                )  # pragma: no cover
            kwargs.pop("c", None)
            if len(R) > 1:
                ojr = numpy.zeros(len(R))
                olz = numpy.zeros(len(R))
                ojz = numpy.zeros(len(R))
                for ii in range(len(R)):
                    targs = (R[ii], vR[ii], vT[ii], z[ii], vz[ii])
                    tjr, tlz, tjz = self(*targs, **copy.copy(kwargs))
                    ojr[ii] = tjr[0]
                    ojz[ii] = tjz[0]
                    olz[ii] = tlz[0]
                return (ojr, olz, ojz)
            else:
                if kwargs.get("_justjr", False):
                    kwargs.pop("_justjr")
                    return (
                        self._aAS(R[0], vR[0], vT[0], 0.0, 0.0, _Jz=0.0)[0],
                        numpy.nan,
                        numpy.nan,
                    )
                # Set up the actionAngleVertical object
                if _dim(self._pot) == 3:
                    thisverticalpot = toVerticalPotential(self._pot, R[0])
                    aAV = actionAngleVertical(pot=thisverticalpot)
                    Jz = aAV(z[0], vz[0])
                else:  # 2D in-plane
                    Jz = numpy.zeros(1)
                if kwargs.get("_justjz", False):
                    kwargs.pop("_justjz")
                    return (
                        numpy.atleast_1d(numpy.nan),
                        numpy.atleast_1d(numpy.nan),
                        Jz,
                    )
                else:
                    axiJ = self._aAS(R[0], vR[0], vT[0], 0.0, 0.0, _Jz=Jz)
                    return (
                        numpy.atleast_1d(axiJ[0]),
                        numpy.atleast_1d(axiJ[1]),
                        numpy.atleast_1d(Jz),
                    )

    def _actionsFreqs(self, *args, **kwargs):
        """
        Evaluate the actions and frequencies (jr,lz,jz,Omegar,Omegaphi,Omegaz).

        Parameters
        ----------
        *args : tuple
            Either:
            a) R,vR,vT,z,vz[,phi]:
                1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)
                2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
            b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
        fixed_quad: bool, optional
            if True, use n=10 fixed_quad integration
        **kwargs: dict, optional
            scipy.integrate.quadrature or .fixed_quad keywords

        Returns
        -------
        tuple
            (jr,lz,jz,Omegar,Omegaphi,Omegaz)

        Notes
        -----
        - 2026-01-14 - Written - Bovy (UofT)
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
        if isinstance(R, float):
            R = numpy.array([R])
            vR = numpy.array([vR])
            vT = numpy.array([vT])
            z = numpy.array([z])
            vz = numpy.array([vz])
        if is_backend_array(R):
            # jax/torch inputs: vectorised, differentiable path (see below).
            return self._actionsFreqs_backend(R, vR, vT, z, vz)
        if len(R) > 1:
            ojr = numpy.zeros(len(R))
            olz = numpy.zeros(len(R))
            ojz = numpy.zeros(len(R))
            oor = numpy.zeros(len(R))
            oophi = numpy.zeros(len(R))
            ooz = numpy.zeros(len(R))
            for ii in range(len(R)):
                targs = (R[ii], vR[ii], vT[ii], z[ii], vz[ii])
                tjr, tlz, tjz, toor, toophi, tooz = self._actionsFreqs(
                    *targs, **copy.copy(kwargs)
                )
                ojr[ii] = tjr[0]
                ojz[ii] = tjz[0]
                olz[ii] = tlz[0]
                oor[ii] = toor[0]
                oophi[ii] = toophi[0]
                ooz[ii] = tooz[0]
            return (ojr, olz, ojz, oor, oophi, ooz)
        else:
            # Set up the actionAngleVertical object
            if _dim(self._pot) == 3 and not (z[0] == 0.0 and vz[0] == 0.0):
                thisverticalpot = toVerticalPotential(self._pot, R[0])
                aAV = actionAngleVertical(pot=thisverticalpot)
                Jz, Oz = aAV.actionsFreqs(z[0], vz[0])
            else:  # 2D in-plane
                Jz = numpy.zeros(1)
                Oz = numpy.ones(1) * self._pot.verticalfreq(R[0])
            axiJO = self._aAS.actionsFreqs(R[0], vR[0], vT[0], 0.0, 0.0, _Jz=Jz)
            return (
                numpy.atleast_1d(axiJO[0]),
                numpy.atleast_1d(axiJO[1]),
                numpy.atleast_1d(Jz),
                numpy.atleast_1d(axiJO[3]),
                numpy.atleast_1d(axiJO[4]),
                numpy.atleast_1d(Oz),
            )

    def _actionsFreqsAngles(self, *args, **kwargs):
        """
        Evaluate the actions, frequencies, and angles (jr,lz,jz,Omegar,Omegaphi,Omegaz,ar,aphi,az).

        Parameters
        ----------
        *args : tuple
            Either:
            a) R,vR,vT,z,vz[,phi]:
                1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)
                2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
            b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
        fixed_quad: bool, optional
            if True, use n=10 fixed_quad integration
        **kwargs: dict, optional
            scipy.integrate.quadrature or .fixed_quad keywords

        Returns
        -------
        tuple
            (jr,lz,jz,Omegar,Omegaphi,Omegaz,ar,aphi,az)

        Notes
        -----
        - 2026-01-15 - Written - Bovy (UofT)
        """
        if len(args) == 5:  # R,vR.vT, z, vz pragma: no cover
            raise OSError("You need to provide phi when calculating angles")
        elif len(args) == 6:  # R,vR.vT, z, vz, phi
            R, vR, vT, z, vz, phi = args
        else:
            self._parse_eval_args(*args)
            R = self._eval_R
            vR = self._eval_vR
            vT = self._eval_vT
            z = self._eval_z
            vz = self._eval_vz
            phi = self._eval_phi
        if isinstance(R, float):
            R = numpy.array([R])
            vR = numpy.array([vR])
            vT = numpy.array([vT])
            z = numpy.array([z])
            vz = numpy.array([vz])
            phi = numpy.array([phi])
        if is_backend_array(R):
            # jax/torch inputs: vectorised, differentiable path (see below).
            return self._actionsFreqsAngles_backend(R, vR, vT, z, vz, phi)
        if len(R) > 1:
            ojr = numpy.zeros(len(R))
            olz = numpy.zeros(len(R))
            ojz = numpy.zeros(len(R))
            oor = numpy.zeros(len(R))
            oophi = numpy.zeros(len(R))
            ooz = numpy.zeros(len(R))
            oar = numpy.zeros(len(R))
            oaphi = numpy.zeros(len(R))
            oaz = numpy.zeros(len(R))
            for ii in range(len(R)):
                targs = (R[ii], vR[ii], vT[ii], z[ii], vz[ii], phi[ii])
                tjr, tlz, tjz, toor, toophi, tooz, tar, taphi, taz = (
                    self._actionsFreqsAngles(*targs, **copy.copy(kwargs))
                )
                ojr[ii] = tjr[0]
                ojz[ii] = tjz[0]
                olz[ii] = tlz[0]
                oor[ii] = toor[0]
                oophi[ii] = toophi[0]
                ooz[ii] = tooz[0]
                oar[ii] = tar[0]
                oaphi[ii] = taphi[0]
                oaz[ii] = taz[0]
            return (ojr, olz, ojz, oor, oophi, ooz, oar, oaphi, oaz)
        else:
            # Set up the actionAngleVertical object
            if _dim(self._pot) == 3 and not (z[0] == 0.0 and vz[0] == 0.0):
                thisverticalpot = toVerticalPotential(self._pot, R[0])
                aAV = actionAngleVertical(pot=thisverticalpot)
                Jz, Oz, az = aAV.actionsFreqsAngles(z[0], vz[0])
            else:  # 2D in-plane
                Jz = numpy.zeros(1)
                Oz = numpy.ones(1) * self._pot.verticalfreq(R[0])
                az = numpy.zeros(1)
            axiJO = self._aAS.actionsFreqsAngles(
                R[0], vR[0], vT[0], 0.0, 0.0, phi[0], _Jz=Jz
            )
            return (
                numpy.atleast_1d(axiJO[0]),
                numpy.atleast_1d(axiJO[1]),
                numpy.atleast_1d(Jz),
                numpy.atleast_1d(axiJO[3]),
                numpy.atleast_1d(axiJO[4]),
                numpy.atleast_1d(Oz),
                numpy.atleast_1d(axiJO[6]),
                numpy.atleast_1d(axiJO[7]),
                numpy.atleast_1d(az),
            )

    def _EccZmaxRperiRap(self, *args, **kwargs):
        """
        Evaluate the eccentricity, maximum height above the plane, peri- and apocenter in the adiabatic approximation.

        Parameters
        ----------
        *args : tuple
            Either:
            a) R,vR,vT,z,vz[,phi]:
                1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)
                2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
            b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
        c: bool, optional
            True/False to override the object-wide setting for whether or not to use the C implementation

        Returns
        -------
        (e,zmax,rperi,rap)
            Eccentricity, maximum height above the plane, peri- and apocenter.

        Notes
        -----
        - 2017-12-21 - Written - Bovy (UofT)
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
        if isinstance(R, float):
            R = numpy.array([R])
            vR = numpy.array([vR])
            vT = numpy.array([vT])
            z = numpy.array([z])
            vz = numpy.array([vz])
        if is_backend_array(R):
            # jax/torch inputs: vectorised, differentiable path (see below).
            return self._EccZmaxRperiRap_backend(R, vR, vT, z, vz)
        if (
            (self._c and not ("c" in kwargs and not kwargs["c"]))
            or (ext_loaded and ("c" in kwargs and kwargs["c"]))
        ) and _check_c(self._pot):
            (
                rperi,
                Rap,
                zmax,
                err,
            ) = actionAngleAdiabatic_c.actionAngleRperiRapZmaxAdiabatic_c(
                self._pot, self._gamma, R, vR, vT, z, vz
            )
            if err == 0:
                rap = numpy.sqrt(Rap**2.0 + zmax**2.0)
                ecc = (rap - rperi) / (rap + rperi)
                return (ecc, zmax, rperi, rap)
            else:  # pragma: no cover
                raise RuntimeError(
                    "C-code for calculation actions failed; try with c=False"
                )
        else:
            if "c" in kwargs and kwargs["c"] and not self._c:
                warnings.warn(
                    "C module not used because potential does not have a C implementation",
                    galpyWarning,
                )  # pragma: no cover
            kwargs.pop("c", None)
            if len(R) > 1:
                oecc = numpy.zeros(len(R))
                orperi = numpy.zeros(len(R))
                orap = numpy.zeros(len(R))
                ozmax = numpy.zeros(len(R))
                for ii in range(len(R)):
                    targs = (R[ii], vR[ii], vT[ii], z[ii], vz[ii])
                    tecc, tzmax, trperi, trap = self._EccZmaxRperiRap(
                        *targs, **copy.copy(kwargs)
                    )
                    oecc[ii] = tecc[0]
                    ozmax[ii] = tzmax[0]
                    orperi[ii] = trperi[0]
                    orap[ii] = trap[0]
                return (oecc, ozmax, orperi, orap)
            else:
                if _dim(self._pot) == 3:
                    thisverticalpot = toVerticalPotential(self._pot, R[0])
                    aAV = actionAngleVertical(pot=thisverticalpot)
                    zmax = aAV.calcxmax(z[0], vz[0], **kwargs)
                    if self._gamma != 0.0:
                        Jz = aAV(z[0], vz[0])
                    else:
                        Jz = 0.0
                else:
                    zmax = 0.0
                    Jz = 0.0
                _, _, rperi, Rap = self._aAS.EccZmaxRperiRap(
                    R[0], vR[0], vT[0], 0.0, 0.0, _Jz=Jz
                )
                rap = numpy.sqrt(Rap**2.0 + zmax**2.0)
                return (
                    numpy.atleast_1d((rap - rperi) / (rap + rperi)),
                    numpy.atleast_1d(zmax),
                    numpy.atleast_1d(rperi),
                    numpy.atleast_1d(rap),
                )

    # ------------------------------------------------ backend (jax/torch) path
    # Vectorised, differentiable mirror of the per-object numpy loop above (the
    # numpy path is byte-identical and untouched; jax/torch inputs branch here).
    # All N objects are processed at once: the RADIAL part is delegated to the
    # already-backend-migrated actionAngleSpherical self._aAS (which now handles
    # _gamma + the _Jz array internally), the VERTICAL part reuses
    # actionAngleVertical's backend Gauss-Legendre / root-find machinery via a
    # _BatchedVerticalPotential carrying the per-object effective vertical
    # potential Phi(R_i, z) - Phi(R_i, 0). The planar (z==0 & vz==0) and 2D
    # (_dim==2) cases are handled with xp.where: Jz=0, Oz=verticalfreq(R), az=0.

    def _batched_aAV(self, R):
        """actionAngleVertical over the batched effective vertical potential."""
        vpot = _BatchedVerticalPotential(self._pot, R)
        return actionAngleVertical(pot=vpot)

    def _vertical_Jz_backend(self, R, z, vz):
        """Vertical action Jz for each object (0 for the planar / 2D cases)."""
        xp = get_namespace(R)
        if _dim(self._pot) == 2:  # in-plane: no vertical motion
            return xp.zeros_like(R)
        aAV = self._batched_aAV(R)
        Jz = aAV._evaluate(z, vz)
        # Planar orbits (z==0 & vz==0): Jz==0 exactly (dead-branch guard: the
        # GL/root-find above is meaningless there, but xp.where overrides it).
        planar = (z == 0.0) & (vz == 0.0)
        return xp.where(planar, xp.zeros_like(Jz), Jz)

    def _vertical_JzOz_backend(self, R, z, vz):
        """Vertical (Jz, Oz); planar/2D -> (0, verticalfreq(R))."""
        from ..potential import verticalfreq

        xp = get_namespace(R)
        vfreq = verticalfreq(self._pot, R)  # raises for a 2D pot (no Oz)
        if _dim(self._pot) == 2:  # pragma: no cover -- vfreq above raised first
            return (xp.zeros_like(R), vfreq)
        aAV = self._batched_aAV(R)
        Jz, Oz = aAV._actionsFreqs(z, vz)
        planar = (z == 0.0) & (vz == 0.0)
        Jz = xp.where(planar, xp.zeros_like(Jz), Jz)
        Oz = xp.where(planar, vfreq, Oz)
        return (Jz, Oz)

    def _vertical_JzOzaz_backend(self, R, z, vz):
        """Vertical (Jz, Oz, az); planar/2D -> (0, verticalfreq(R), 0)."""
        from ..potential import verticalfreq

        xp = get_namespace(R)
        vfreq = verticalfreq(self._pot, R)  # raises for a 2D pot (no Oz)
        if _dim(self._pot) == 2:  # pragma: no cover -- vfreq above raised first
            return (xp.zeros_like(R), vfreq, xp.zeros_like(R))
        aAV = self._batched_aAV(R)
        Jz, Oz, az = aAV._actionsFreqsAngles(z, vz)
        planar = (z == 0.0) & (vz == 0.0)
        Jz = xp.where(planar, xp.zeros_like(Jz), Jz)
        Oz = xp.where(planar, vfreq, Oz)
        az = xp.where(planar, xp.zeros_like(az), az)
        return (Jz, Oz, az)

    def _evaluate_backend(self, R, vR, vT, z, vz):
        xp = get_namespace(R)
        Jz = self._vertical_Jz_backend(R, z, vz)
        z0 = xp.zeros_like(R)
        Jr, Lz, _ = self._aAS._evaluate(R, vR, vT, z0, z0, _Jz=Jz)
        return (Jr, Lz, Jz)

    def _actionsFreqs_backend(self, R, vR, vT, z, vz):
        xp = get_namespace(R)
        Jz, Oz = self._vertical_JzOz_backend(R, z, vz)
        z0 = xp.zeros_like(R)
        Jr, Lz, _, Or, Op, _ = self._aAS._actionsFreqs(R, vR, vT, z0, z0, _Jz=Jz)
        return (Jr, Lz, Jz, Or, Op, Oz)

    def _actionsFreqsAngles_backend(self, R, vR, vT, z, vz, phi):
        xp = get_namespace(R)
        Jz, Oz, az = self._vertical_JzOzaz_backend(R, z, vz)
        z0 = xp.zeros_like(R)
        Jr, Lz, _, Or, Op, _, ar, aphi, _ = self._aAS._actionsFreqsAngles(
            R, vR, vT, z0, z0, phi, _Jz=Jz
        )
        return (Jr, Lz, Jz, Or, Op, Oz, ar, aphi, az)

    def _EccZmaxRperiRap_backend(self, R, vR, vT, z, vz):
        xp = get_namespace(R)
        # zmax + Jz from the (batched) vertical potential.
        if _dim(self._pot) == 3:
            aAV = self._batched_aAV(R)
            E = aAV._E_backend(z, vz)
            zmax = aAV._calc_xmax_backend(z, vz, E)
            if self._gamma != 0.0:
                Jz = self._vertical_Jz_backend(R, z, vz)
            else:
                Jz = xp.zeros_like(R)
        else:
            zmax = xp.zeros_like(R)
            Jz = xp.zeros_like(R)
        z0 = xp.zeros_like(R)
        _, _, rperi, Rap = self._aAS._EccZmaxRperiRap(R, vR, vT, z0, z0, _Jz=Jz)
        rap = xp.sqrt(Rap**2.0 + zmax**2.0)
        return ((rap - rperi) / (rap + rperi), zmax, rperi, rap)
