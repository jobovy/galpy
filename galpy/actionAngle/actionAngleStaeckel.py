###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleStaeckel
#
#             Use Binney (2012; MNRAS 426, 1324)'s Staeckel approximation for
#             calculating the actions
#
#      methods:
#             __call__: returns (jr,lz,jz)
#
###############################################################################
import copy
import warnings

import numpy
from scipy import integrate, optimize

from ..potential import (
    DiskSCFPotential,
    MWPotential,
    SCFPotential,
    epifreq,
    evaluateR2derivs,
    evaluateRzderivs,
    evaluatez2derivs,
    omegac,
    verticalfreq,
)
from ..potential.Potential import (
    _check_c,
    _evaluatePotentials,
    _evaluateRforces,
    _evaluatezforces,
)
from ..potential.Potential import flatten as flatten_potential
from ..util import coords  # for prolate confocal transforms
from ..util import conversion, galpyWarning
from ..util.conversion import physical_conversion, potential_physical_input
from . import actionAngleStaeckel_c
from .actionAngle import UnboundError, actionAngle
from .actionAngleStaeckel_c import _ext_loaded as ext_loaded


class actionAngleStaeckel(actionAngle):
    """Action-angle formalism for axisymmetric potentials using Binney (2012)'s Staeckel approximation"""

    def __init__(self, *args, **kwargs):
        """
        Initialize an actionAngleStaeckel object.

        Parameters
        ----------
        pot : potential or list of potentials (3D)
            The potential or list of potentials.
        delta : float or Quantity
            The focus.
        useu0 : bool, optional
            Use u0 to calculate dV (not recommended). Default is False.
        c : bool, optional
            If True, always use C for calculations. Default is False.
        order : int, optional
            Number of points to use in the Gauss-Legendre numerical integration of the relevant action, frequency, and angle integrals. Default is 10.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2012-11-27 - Started - Bovy (IAS).
        """
        actionAngle.__init__(self, ro=kwargs.get("ro", None), vo=kwargs.get("vo", None))
        if not "pot" in kwargs:  # pragma: no cover
            raise OSError("Must specify pot= for actionAngleStaeckel")
        self._pot = flatten_potential(kwargs["pot"])
        if self._pot == MWPotential:
            warnings.warn(
                "Use of MWPotential as a Milky-Way-like potential is deprecated; galpy.potential.MWPotential2014, a potential fit to a large variety of dynamical constraints (see Bovy 2015), is the preferred Milky-Way-like potential in galpy",
                galpyWarning,
            )
        if not "delta" in kwargs:  # pragma: no cover
            raise OSError("Must specify delta= for actionAngleStaeckel")
        if ext_loaded and (("c" in kwargs and kwargs["c"]) or not "c" in kwargs):
            self._c = _check_c(self._pot)
            if "c" in kwargs and kwargs["c"] and not self._c:
                warnings.warn(
                    "C module not used because potential does not have a C implementation",
                    galpyWarning,
                )  # pragma: no cover
        else:
            self._c = False
        self._useu0 = kwargs.get("useu0", False)
        self._delta = kwargs["delta"]
        self._order = kwargs.get("order", 10)
        self._delta = conversion.parse_length(self._delta, ro=self._ro)
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
        delta: bool, optional
            can be used to override the object-wide focal length; can also be an array with length N to allow different delta for different phase-space points
        u0: float, optional
            if object-wide option useu0 is set, u0 to use (if useu0 and useu0 is None, a good value will be computed).
        c: bool, optional
            True/False to override the object-wide setting for whether or not to use the C implementation.
        order: int, optional
            number of points to use in the Gauss-Legendre numerical integration of the relevant action integrals.
        fixed_quad: bool, optional
            if True, use Gaussian quadrature (scipy.integrate.fixed_quad instead of scipy.integrate.quad).
        **kwargs: dict, optional
            scipy.integrate.fixed_quad or .quad keywords when not using C

        Returns
        -------
        tuple
            (jr,lz,jz)

        Notes
        -----
        - 2012-11-27 - Written - Bovy (IAS)
        - 2017-12-27 - Allowed individual delta for each point - Bovy (UofT)
        """
        delta = kwargs.pop("delta", self._delta)
        order = kwargs.get("order", self._order)
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
        if (
            (self._c and not ("c" in kwargs and not kwargs["c"]))
            or (ext_loaded and ("c" in kwargs and kwargs["c"]))
        ) and _check_c(self._pot):
            Lz = R * vT
            if self._useu0:
                # First calculate u0
                if "u0" in kwargs:
                    u0 = numpy.asarray(kwargs["u0"])
                else:
                    E = numpy.array(
                        [
                            _evaluatePotentials(self._pot, R[ii], z[ii])
                            + vR[ii] ** 2.0 / 2.0
                            + vz[ii] ** 2.0 / 2.0
                            + vT[ii] ** 2.0 / 2.0
                            for ii in range(len(R))
                        ]
                    )
                    u0 = actionAngleStaeckel_c.actionAngleStaeckel_calcu0(
                        E, Lz, self._pot, delta
                    )[0]
                kwargs.pop("u0", None)
            else:
                u0 = None
            jr, jz, err = actionAngleStaeckel_c.actionAngleStaeckel_c(
                self._pot, delta, R, vR, vT, z, vz, u0=u0, order=order
            )
            if err == 0:
                return (jr, Lz, jz)
            else:  # pragma: no cover
                raise RuntimeError(
                    "C-code for calculation actions failed; try with c=False"
                )
        else:
            if "c" in kwargs and kwargs["c"] and not self._c:  # pragma: no cover
                warnings.warn(
                    "C module not used because potential does not have a C implementation",
                    galpyWarning,
                )
            kwargs.pop("c", None)
            if len(R) > 1:
                ojr = numpy.zeros(len(R))
                olz = numpy.zeros(len(R))
                ojz = numpy.zeros(len(R))
                for ii in range(len(R)):
                    targs = (R[ii], vR[ii], vT[ii], z[ii], vz[ii])
                    tkwargs = copy.copy(kwargs)
                    try:
                        tkwargs["delta"] = delta[ii]
                    except (TypeError, IndexError):
                        tkwargs["delta"] = delta
                    tjr, tlz, tjz = self(*targs, **tkwargs)
                    ojr[ii] = tjr[0]
                    ojz[ii] = tjz[0]
                    olz[ii] = tlz[0]
                return (ojr, olz, ojz)
            else:
                # Set up the actionAngleStaeckelSingle object
                aASingle = actionAngleStaeckelSingle(
                    R[0], vR[0], vT[0], z[0], vz[0], pot=self._pot, delta=delta
                )
                return (
                    numpy.atleast_1d(aASingle.JR(**copy.copy(kwargs))),
                    numpy.atleast_1d(aASingle._R * aASingle._vT),
                    numpy.atleast_1d(aASingle.Jz(**copy.copy(kwargs))),
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
        delta: bool, optional
            can be used to override the object-wide focal length; can also be an array with length N to allow different delta for different phase-space points
        u0: float, optional
            if object-wide option useu0 is set, u0 to use (if useu0 and useu0 is None, a good value will be computed).
        c: bool, optional
            True/False to override the object-wide setting for whether or not to use the C implementation.
        order: int, optional
            number of points to use in the Gauss-Legendre numerical integration of the relevant action integrals.
        fixed_quad: bool, optional
            if True, use Gaussian quadrature (scipy.integrate.fixed_quad instead of scipy.integrate.quad).
        **kwargs: dict, optional
            scipy.integrate.fixed_quad or .quad keywords when not using C

        Returns
        -------
        tuple
            (jr,lz,jz,Omegar,Omegaphi,Omegaz)

        Notes
        -----
        - 2013-08-28 - Written - Bovy (IAS)
        """
        delta = kwargs.pop("delta", self._delta)
        order = kwargs.get("order", self._order)
        if (
            (self._c and not ("c" in kwargs and not kwargs["c"]))
            or (ext_loaded and ("c" in kwargs and kwargs["c"]))
        ) and _check_c(self._pot):
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
            Lz = R * vT
            if self._useu0:
                # First calculate u0
                if "u0" in kwargs:
                    u0 = numpy.asarray(kwargs["u0"])
                else:
                    E = numpy.array(
                        [
                            _evaluatePotentials(self._pot, R[ii], z[ii])
                            + vR[ii] ** 2.0 / 2.0
                            + vz[ii] ** 2.0 / 2.0
                            + vT[ii] ** 2.0 / 2.0
                            for ii in range(len(R))
                        ]
                    )
                    u0 = actionAngleStaeckel_c.actionAngleStaeckel_calcu0(
                        E, Lz, self._pot, delta
                    )[0]
                kwargs.pop("u0", None)
            else:
                u0 = None
            (
                jr,
                jz,
                Omegar,
                Omegaphi,
                Omegaz,
                err,
            ) = actionAngleStaeckel_c.actionAngleFreqStaeckel_c(
                self._pot, delta, R, vR, vT, z, vz, u0=u0, order=order
            )
            # Adjustments for close-to-circular orbits
            indx = numpy.isnan(Omegar) * (jr < 10.0**-3.0) + numpy.isnan(Omegaz) * (
                jz < 10.0**-3.0
            )  # Close-to-circular and close-to-the-plane orbits
            if numpy.sum(indx) > 0:
                Omegar[indx] = [
                    epifreq(self._pot, r, use_physical=False) for r in R[indx]
                ]
                Omegaphi[indx] = [
                    omegac(self._pot, r, use_physical=False) for r in R[indx]
                ]
                Omegaz[indx] = [
                    verticalfreq(self._pot, r, use_physical=False) for r in R[indx]
                ]
            if err == 0:
                return (jr, Lz, jz, Omegar, Omegaphi, Omegaz)
            else:  # pragma: no cover
                raise RuntimeError(
                    "C-code for calculation actions failed; try with c=False"
                )
        else:
            if "c" in kwargs and kwargs["c"] and not self._c:  # pragma: no cover
                warnings.warn(
                    "C module not used because potential does not have a C implementation",
                    galpyWarning,
                )
            raise NotImplementedError(
                "actionsFreqs with c=False not implemented; maybe you meant to install the C extension?"
            )

    def _actionsFreqsAngles(self, *args, **kwargs):
        """
        Evaluate the actions, frequencies, and angles (jr,lz,jz,Omegar,Omegaphi,Omegaz,angler,anglephi,anglez).

        Parameters
        ----------
        *args : tuple
            Either:
            a) R,vR,vT,z,vz[,phi]:
                1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)
                2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
            b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
        delta: bool, optional
            can be used to override the object-wide focal length; can also be an array with length N to allow different delta for different phase-space points
        u0: float, optional
            if object-wide option useu0 is set, u0 to use (if useu0 and useu0 is None, a good value will be computed).
        c: bool, optional
            True/False to override the object-wide setting for whether or not to use the C implementation.
        order: int, optional
            number of points to use in the Gauss-Legendre numerical integration of the relevant action integrals.
        fixed_quad: bool, optional
            if True, use Gaussian quadrature (scipy.integrate.fixed_quad instead of scipy.integrate.quad).
        **kwargs: dict, optional
            scipy.integrate.fixed_quad or .quad keywords when not using C

        Returns
        -------
        tuple
            (jr,lz,jz,Omegar,Omegaphi,Omegaz,angler,anglephi,anglez)

        Notes
        -----
        - 2013-08-28 - Written - Bovy (IAS)
        """
        delta = kwargs.pop("delta", self._delta)
        order = kwargs.get("order", self._order)
        if (
            (self._c and not ("c" in kwargs and not kwargs["c"]))
            or (ext_loaded and ("c" in kwargs and kwargs["c"]))
        ) and _check_c(self._pot):
            if len(args) == 5:  # R,vR.vT, z, vz pragma: no cover
                raise OSError("Must specify phi")
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
            Lz = R * vT
            if self._useu0:
                # First calculate u0
                if "u0" in kwargs:
                    u0 = numpy.asarray(kwargs["u0"])
                else:
                    E = numpy.array(
                        [
                            _evaluatePotentials(self._pot, R[ii], z[ii])
                            + vR[ii] ** 2.0 / 2.0
                            + vz[ii] ** 2.0 / 2.0
                            + vT[ii] ** 2.0 / 2.0
                            for ii in range(len(R))
                        ]
                    )
                    u0 = actionAngleStaeckel_c.actionAngleStaeckel_calcu0(
                        E, Lz, self._pot, delta
                    )[0]
                kwargs.pop("u0", None)
            else:
                u0 = None
            (
                jr,
                jz,
                Omegar,
                Omegaphi,
                Omegaz,
                angler,
                anglephi,
                anglez,
                err,
            ) = actionAngleStaeckel_c.actionAngleFreqAngleStaeckel_c(
                self._pot, delta, R, vR, vT, z, vz, phi, u0=u0, order=order
            )
            # Adjustments for close-to-circular orbits
            indx = numpy.isnan(Omegar) * (jr < 10.0**-3.0) + numpy.isnan(Omegaz) * (
                jz < 10.0**-3.0
            )  # Close-to-circular and close-to-the-plane orbits
            if numpy.sum(indx) > 0:
                Omegar[indx] = [
                    epifreq(self._pot, r, use_physical=False) for r in R[indx]
                ]
                Omegaphi[indx] = [
                    omegac(self._pot, r, use_physical=False) for r in R[indx]
                ]
                Omegaz[indx] = [
                    verticalfreq(self._pot, r, use_physical=False) for r in R[indx]
                ]
            if err == 0:
                return (jr, Lz, jz, Omegar, Omegaphi, Omegaz, angler, anglephi, anglez)
            else:
                raise RuntimeError(
                    "C-code for calculation actions failed; try with c=False"
                )  # pragma: no cover
        else:  # pragma: no cover
            if "c" in kwargs and kwargs["c"] and not self._c:  # pragma: no cover
                warnings.warn(
                    "C module not used because potential does not have a C implementation",
                    galpyWarning,
                )
            raise NotImplementedError(
                "actionsFreqs with c=False not implemented; maybe you meant to install the C extension?"
            )

    def _EccZmaxRperiRap(self, *args, **kwargs):
        """
        Evaluate the eccentricity, maximum height above the plane, peri- and apocenter in the Staeckel approximation.

        Parameters
        ----------
        *args : tuple
            Either:
            a) R,vR,vT,z,vz[,phi]:
                1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)
                2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
            b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
        delta: bool, optional
            can be used to override the object-wide focal length; can also be an array with length N to allow different delta for different phase-space points
        u0: float, optional
            if object-wide option useu0 is set, u0 to use (if useu0 and useu0 is None, a good value will be computed).
        c: bool, optional
            True/False to override the object-wide setting for whether or not to use the C implementation.

        Returns
        -------
        tuple
            (e,zmax,rperi,rap)

        Notes
        -----
        - 2017-12-12 - Written - Bovy (UofT)
        """
        delta = kwargs.get("delta", self._delta)
        umin, umax, vmin = self._uminumaxvmin(*args, **kwargs)
        rperi = coords.uv_to_Rz(umin, numpy.pi / 2.0, delta=delta)[0]
        rap_tmp, zmax = coords.uv_to_Rz(umax, vmin, delta=delta)
        rap = numpy.sqrt(rap_tmp**2.0 + zmax**2.0)
        e = (rap - rperi) / (rap + rperi)
        return (e, zmax, rperi, rap)

    def _uminumaxvmin(self, *args, **kwargs):
        """
        Evaluate u_min, u_max, and v_min in the Staeckel approximation.

        Parameters
        ----------
        *args : tuple
            Either:
            a) R,vR,vT,z,vz[,phi]:
                1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)
                2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
            b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
        delta: bool, optional
            can be used to override the object-wide focal length; can also be an array with length N to allow different delta for different phase-space points
        u0: float, optional
            if object-wide option useu0 is set, u0 to use (if useu0 and useu0 is None, a good value will be computed).
        c: bool, optional
            True/False to override the object-wide setting for whether or not to use the C implementation.

        Returns
        -------
        tuple
            (u_min, u_max, v_min)

        Notes
        -----
        - 2017-12-12 - Written - Bovy (UofT)
        """
        delta = numpy.atleast_1d(kwargs.pop("delta", self._delta))
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
        if (
            (self._c and not ("c" in kwargs and not kwargs["c"]))
            or (ext_loaded and ("c" in kwargs and kwargs["c"]))
        ) and _check_c(self._pot):
            Lz = R * vT
            if self._useu0:
                # First calculate u0
                if "u0" in kwargs:
                    u0 = numpy.asarray(kwargs["u0"])
                else:
                    E = numpy.array(
                        [
                            _evaluatePotentials(self._pot, R[ii], z[ii])
                            + vR[ii] ** 2.0 / 2.0
                            + vz[ii] ** 2.0 / 2.0
                            + vT[ii] ** 2.0 / 2.0
                            for ii in range(len(R))
                        ]
                    )
                    u0 = actionAngleStaeckel_c.actionAngleStaeckel_calcu0(
                        E, Lz, self._pot, delta
                    )[0]
                kwargs.pop("u0", None)
            else:
                u0 = None
            (
                umin,
                umax,
                vmin,
                err,
            ) = actionAngleStaeckel_c.actionAngleUminUmaxVminStaeckel_c(
                self._pot, delta, R, vR, vT, z, vz, u0=u0
            )
            if err == 0:
                return (umin, umax, vmin)
            else:  # pragma: no cover
                raise RuntimeError(
                    "C-code for calculation actions failed; try with c=False"
                )
        else:
            if "c" in kwargs and kwargs["c"] and not self._c:  # pragma: no cover
                warnings.warn(
                    "C module not used because potential does not have a C implementation",
                    galpyWarning,
                )
            kwargs.pop("c", None)
            if len(R) > 1:
                oumin = numpy.zeros(len(R))
                oumax = numpy.zeros(len(R))
                ovmin = numpy.zeros(len(R))
                for ii in range(len(R)):
                    targs = (R[ii], vR[ii], vT[ii], z[ii], vz[ii])
                    tkwargs = copy.copy(kwargs)
                    tkwargs["delta"] = delta[ii] if len(delta) > 1 else delta[0]
                    tumin, tumax, tvmin = self._uminumaxvmin(*targs, **tkwargs)
                    oumin[ii] = tumin[0]
                    oumax[ii] = tumax[0]
                    ovmin[ii] = tvmin[0]
                return (oumin, oumax, ovmin)
            else:
                # Set up the actionAngleStaeckelSingle object
                aASingle = actionAngleStaeckelSingle(
                    R[0], vR[0], vT[0], z[0], vz[0], pot=self._pot, delta=delta[0]
                )
                umin, umax = aASingle.calcUminUmax()
                vmin = aASingle.calcVmin()
                return (
                    numpy.atleast_1d(umin),
                    numpy.atleast_1d(umax),
                    numpy.atleast_1d(vmin),
                )


class actionAngleStaeckelSingle(actionAngle):
    """Action-angle formalism for axisymmetric potentials using Binney (2012)'s Staeckel approximation"""

    def __init__(self, *args, **kwargs):
        """
        Initialize an actionAngleStaeckelSingle object

        Parameters
        ----------
        *args : tuple
            Either:
            a) R,vR,vT,z,vz[,phi]:
                1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)
                2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
            b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
        pot: Potential or list of Potentials
            Potential to use
        delta: float, optional
            focal length of confocal coordinate system

        Notes
        -----
        - 2012-11-27 - Written - Bovy (IAS)
        """
        self._parse_eval_args(*args, _noOrbUnitsCheck=True, **kwargs)
        self._R = self._eval_R
        self._vR = self._eval_vR
        self._vT = self._eval_vT
        self._z = self._eval_z
        self._vz = self._eval_vz
        if not "pot" in kwargs:  # pragma: no cover
            raise OSError("Must specify pot= for actionAngleStaeckelSingle")
        self._pot = kwargs["pot"]
        if not "delta" in kwargs:  # pragma: no cover
            raise OSError("Must specify delta= for actionAngleStaeckel")
        self._delta = kwargs["delta"]
        # Pre-calculate everything
        self._ux, self._vx = coords.Rz_to_uv(self._R, self._z, delta=self._delta)
        self._sinvx = numpy.sin(self._vx)
        self._cosvx = numpy.cos(self._vx)
        self._coshux = numpy.cosh(self._ux)
        self._sinhux = numpy.sinh(self._ux)
        self._pux = self._delta * (
            self._vR * self._coshux * self._sinvx
            + self._vz * self._sinhux * self._cosvx
        )
        self._pvx = self._delta * (
            self._vR * self._sinhux * self._cosvx
            - self._vz * self._coshux * self._sinvx
        )
        EL = self.calcEL()
        self._E = EL[0]
        self._Lz = EL[1]
        # Determine umin and umax
        self._u0 = kwargs.pop(
            "u0", self._ux
        )  # u0 as defined by Binney does not matter for a
        # single action evaluation, so we don't determine it here
        self._sinhu0 = numpy.sinh(self._u0)
        self._potu0v0 = potentialStaeckel(self._u0, self._vx, self._pot, self._delta)
        self._I3U = (
            self._E * self._sinhux**2.0
            - self._pux**2.0 / 2.0 / self._delta**2.0
            - self._Lz**2.0 / 2.0 / self._delta**2.0 / self._sinhux**2.0
        )
        self._potupi2 = potentialStaeckel(
            self._ux, numpy.pi / 2.0, self._pot, self._delta
        )
        dV = self._coshux**2.0 * self._potupi2 - (
            self._sinhux**2.0 + self._sinvx**2.0
        ) * potentialStaeckel(self._ux, self._vx, self._pot, self._delta)
        self._I3V = (
            -self._E * self._sinvx**2.0
            + self._pvx**2.0 / 2.0 / self._delta**2.0
            + self._Lz**2.0 / 2.0 / self._delta**2.0 / self._sinvx**2.0
            - dV
        )
        self.calcUminUmax()
        self.calcVmin()
        return None

    def angleR(self, **kwargs):
        raise NotImplementedError(
            "'angleR' not yet implemented for Staeckel approximation"
        )

    def TR(self, **kwargs):
        raise NotImplementedError("'TR' not implemented yet for Staeckel approximation")

    def Tphi(self, **kwargs):
        raise NotImplementedError(
            "'Tphi' not implemented yet for Staeckel approxximation"
        )

    def I(self, **kwargs):
        raise NotImplementedError("'I' not implemented yet for Staeckel approxximation")

    def Jphi(self):  # pragma: no cover
        return self._R * self._vT

    def JR(self, **kwargs):
        """
        Calculate the radial action

        Parameters
        ----------
        fixed_quad : bool, optional
            If True, use n=10 fixed_quad. Default is False.
        **kwargs
            scipy.integrate.quad keywords

        Returns
        -------
        float
            J_R(R,vT,vT)/ro/vc + estimate of the error (nan for fixed_quad)

        Notes
        -----
        - 2012-11-27 - Written - Bovy (IAS)

        """
        if hasattr(self, "_JR"):  # pragma: no cover
            return self._JR
        umin, umax = self.calcUminUmax()
        # print self._ux, self._pux, (umax-umin)/umax
        if (umax - umin) / umax < 10.0**-6:
            return numpy.array([0.0])
        order = kwargs.pop("order", 10)
        if kwargs.pop("fixed_quad", False):
            # factor in next line bc integrand=/2delta^2
            self._JR = (
                1.0
                / numpy.pi
                * numpy.sqrt(2.0)
                * self._delta
                * integrate.fixed_quad(
                    _JRStaeckelIntegrand,
                    umin,
                    umax,
                    args=(
                        self._E,
                        self._Lz,
                        self._I3U,
                        self._delta,
                        self._u0,
                        self._sinhu0**2.0,
                        self._vx,
                        self._sinvx**2.0,
                        self._potu0v0,
                        self._pot,
                    ),
                    n=order,
                    **kwargs,
                )[0]
            )
        else:
            self._JR = (
                1.0
                / numpy.pi
                * numpy.sqrt(2.0)
                * self._delta
                * integrate.quad(
                    _JRStaeckelIntegrand,
                    umin,
                    umax,
                    args=(
                        self._E,
                        self._Lz,
                        self._I3U,
                        self._delta,
                        self._u0,
                        self._sinhu0**2.0,
                        self._vx,
                        self._sinvx**2.0,
                        self._potu0v0,
                        self._pot,
                    ),
                    **kwargs,
                )[0]
            )
        return self._JR

    def Jz(self, **kwargs):
        """
        Calculate the vertical action

        Parameters
        ----------
        fixed_quad : bool, optional
            If True, use n=10 fixed_quad. Default is False.
        **kwargs
            scipy.integrate.quad keywords

        Returns
        -------
        float
            J_z(R,vT,vT)/ro/vc + estimate of the error

        Notes
        -----
        - 2012-11-27 - Written - Bovy (IAS)
        """
        if hasattr(self, "_JZ"):  # pragma: no cover
            return self._JZ
        vmin = self.calcVmin()
        if (numpy.pi / 2.0 - vmin) < 10.0**-7:
            return numpy.array([0.0])
        order = kwargs.pop("order", 10)
        if kwargs.pop("fixed_quad", False):
            # factor in next line bc integrand=/2delta^2
            self._JZ = (
                2.0
                / numpy.pi
                * numpy.sqrt(2.0)
                * self._delta
                * integrate.fixed_quad(
                    _JzStaeckelIntegrand,
                    vmin,
                    numpy.pi / 2,
                    args=(
                        self._E,
                        self._Lz,
                        self._I3V,
                        self._delta,
                        self._ux,
                        self._coshux**2.0,
                        self._sinhux**2.0,
                        self._potupi2,
                        self._pot,
                    ),
                    n=order,
                    **kwargs,
                )[0]
            )
        else:
            # factor in next line bc integrand=/2delta^2
            self._JZ = (
                2.0
                / numpy.pi
                * numpy.sqrt(2.0)
                * self._delta
                * integrate.quad(
                    _JzStaeckelIntegrand,
                    vmin,
                    numpy.pi / 2,
                    args=(
                        self._E,
                        self._Lz,
                        self._I3V,
                        self._delta,
                        self._ux,
                        self._coshux**2.0,
                        self._sinhux**2.0,
                        self._potupi2,
                        self._pot,
                    ),
                    **kwargs,
                )[0]
            )
        return self._JZ

    def calcEL(self, **kwargs):
        """
        Calculate the energy and angular momentum.

        Parameters
        ----------
        **kwargs : dict
            scipy.integrate.quadrature keywords

        Returns
        -------
        tuple
            A tuple containing the energy and angular momentum.

        Notes
        -----
        - 2012-11-27 - Written - Bovy (IAS)
        """
        E, L = calcELStaeckel(self._R, self._vR, self._vT, self._z, self._vz, self._pot)
        return (E, L)

    def calcUminUmax(self, **kwargs):
        """
        Calculate the u 'apocenter' and 'pericenter'

        Returns
        -------
        tuple
            (umin,umax)

        Notes
        -----
        - 2012-11-27 - Written - Bovy (IAS)
        """
        if hasattr(self, "_uminumax"):  # pragma: no cover
            return self._uminumax
        E, L = self._E, self._Lz
        # Calculate value of the integrand at current point, to check whether
        # we are at a turning point
        current_val = _JRStaeckelIntegrandSquared(
            self._ux,
            E,
            L,
            self._I3U,
            self._delta,
            self._u0,
            self._sinhu0**2.0,
            self._vx,
            self._sinvx**2.0,
            self._potu0v0,
            self._pot,
        )
        if (
            numpy.fabs(self._pux) < 1e-7 or numpy.fabs(current_val) < 1e-10
        ):  # We are at umin or umax
            eps = 10.0**-8.0
            peps = _JRStaeckelIntegrandSquared(
                self._ux + eps,
                E,
                L,
                self._I3U,
                self._delta,
                self._u0,
                self._sinhu0**2.0,
                self._vx,
                self._sinvx**2.0,
                self._potu0v0,
                self._pot,
            )
            meps = _JRStaeckelIntegrandSquared(
                self._ux - eps,
                E,
                L,
                self._I3U,
                self._delta,
                self._u0,
                self._sinhu0**2.0,
                self._vx,
                self._sinvx**2.0,
                self._potu0v0,
                self._pot,
            )
            if peps < 0.0 and meps > 0.0:  # we are at umax
                umax = self._ux
                rstart, prevr = _uminUmaxFindStart(
                    self._ux,
                    E,
                    L,
                    self._I3U,
                    self._delta,
                    self._u0,
                    self._sinhu0**2.0,
                    self._vx,
                    self._sinvx**2.0,
                    self._potu0v0,
                    self._pot,
                )
                if rstart == 0.0:
                    umin = 0.0
                else:
                    try:
                        umin = optimize.brentq(
                            _JRStaeckelIntegrandSquared,
                            numpy.atleast_1d(rstart)[0],
                            numpy.atleast_1d(self._ux)[0] - eps,
                            (
                                E,
                                L,
                                self._I3U,
                                self._delta,
                                self._u0,
                                self._sinhu0**2.0,
                                self._vx,
                                self._sinvx**2.0,
                                self._potu0v0,
                                self._pot,
                            ),
                            maxiter=200,
                        )
                    except RuntimeError:  # pragma: no cover
                        raise UnboundError("Orbit seems to be unbound")
            elif peps > 0.0 and meps < 0.0:  # we are at umin
                umin = self._ux
                rend, prevr = _uminUmaxFindStart(
                    self._ux,
                    E,
                    L,
                    self._I3U,
                    self._delta,
                    self._u0,
                    self._sinhu0**2.0,
                    self._vx,
                    self._sinvx**2.0,
                    self._potu0v0,
                    self._pot,
                    umax=True,
                )
                umax = optimize.brentq(
                    _JRStaeckelIntegrandSquared,
                    numpy.atleast_1d(self._ux)[0] + eps,
                    numpy.atleast_1d(rend)[0],
                    (
                        E,
                        L,
                        self._I3U,
                        self._delta,
                        self._u0,
                        self._sinhu0**2.0,
                        self._vx,
                        self._sinvx**2.0,
                        self._potu0v0,
                        self._pot,
                    ),
                    maxiter=200,
                )
            else:  # circular orbit
                umin = self._ux
                umax = self._ux
        else:
            rstart, prevr = _uminUmaxFindStart(
                self._ux,
                E,
                L,
                self._I3U,
                self._delta,
                self._u0,
                self._sinhu0**2.0,
                self._vx,
                self._sinvx**2.0,
                self._potu0v0,
                self._pot,
            )
            if rstart == 0.0:
                umin = 0.0
            else:
                if numpy.fabs(prevr - self._ux) < 10.0**-2.0:
                    rup = self._ux
                else:
                    rup = prevr
                try:
                    umin = optimize.brentq(
                        _JRStaeckelIntegrandSquared,
                        rstart,
                        rup,
                        (
                            E,
                            L,
                            self._I3U,
                            self._delta,
                            self._u0,
                            self._sinhu0**2.0,
                            self._vx,
                            self._sinvx**2.0,
                            self._potu0v0,
                            self._pot,
                        ),
                        maxiter=200,
                    )
                except RuntimeError:  # pragma: no cover
                    raise UnboundError("Orbit seems to be unbound")
            rend, prevr = _uminUmaxFindStart(
                self._ux,
                E,
                L,
                self._I3U,
                self._delta,
                self._u0,
                self._sinhu0**2.0,
                self._vx,
                self._sinvx**2.0,
                self._potu0v0,
                self._pot,
                umax=True,
            )
            umax = optimize.brentq(
                _JRStaeckelIntegrandSquared,
                prevr,
                rend,
                (
                    E,
                    L,
                    self._I3U,
                    self._delta,
                    self._u0,
                    self._sinhu0**2.0,
                    self._vx,
                    self._sinvx**2.0,
                    self._potu0v0,
                    self._pot,
                ),
                maxiter=200,
            )
        self._uminumax = (umin, umax)
        return self._uminumax

    def calcVmin(self, **kwargs):
        """
        Calculate the v 'pericenter'

        Returns
        -------
        float
            v_min(R,vT,vT)/vc + estimate of the error

        Notes
        -----
        - 2012-11-28 - Written - Bovy (IAS)
        """
        if hasattr(self, "_vmin"):  # pragma: no cover
            return self._vmin
        E, L = self._E, self._Lz
        if numpy.fabs(self._pvx) < 10.0**-7.0:  # We are at vmin or vmax
            eps = 10.0**-8.0
            peps = _JzStaeckelIntegrandSquared(
                self._vx + eps,
                E,
                L,
                self._I3V,
                self._delta,
                self._ux,
                self._coshux**2.0,
                self._sinhux**2.0,
                self._potupi2,
                self._pot,
            )
            meps = _JzStaeckelIntegrandSquared(
                self._vx - eps,
                E,
                L,
                self._I3V,
                self._delta,
                self._ux,
                self._coshux**2.0,
                self._sinhux**2.0,
                self._potupi2,
                self._pot,
            )
            if peps < 0.0 and meps > 0.0:  # pragma: no cover
                # we are at vmax, which cannot happen
                raise RuntimeError(
                    "Orbit is at the vmax turning point in v, which mathematically cannot happen; something is very wrong!!"
                )
            elif peps > 0.0 and meps < 0.0:  # we are at vmin
                vmin = self._vx
            else:  # planar orbit
                vmin = self._vx
        else:
            rstart = _vminFindStart(
                self._vx,
                E,
                L,
                self._I3V,
                self._delta,
                self._ux,
                self._coshux**2.0,
                self._sinhux**2.0,
                self._potupi2,
                self._pot,
            )
            if rstart == 0.0:
                vmin = 0.0
            else:
                try:
                    vmin = optimize.brentq(
                        _JzStaeckelIntegrandSquared,
                        rstart,
                        rstart / 0.9,
                        (
                            E,
                            L,
                            self._I3V,
                            self._delta,
                            self._ux,
                            self._coshux**2.0,
                            self._sinhux**2.0,
                            self._potupi2,
                            self._pot,
                        ),
                        maxiter=200,
                    )
                except RuntimeError:  # pragma: no cover
                    raise UnboundError("Orbit seems to be unbound")
        self._vmin = vmin
        return self._vmin


def calcELStaeckel(R, vR, vT, z, vz, pot, vc=1.0, ro=1.0):
    """
    Calculate the energy and angular momentum.

    Parameters
    ----------
    R : float
        Galactocentric radius (/ro).
    vR : float
        Radial part of the velocity (/vc).
    vT : float
        Azimuthal part of the velocity (/vc).
    z : float
        Vertical height (/ro).
    vz : float
        Vertical velocity (/vc).
    pot : Potential object
        galpy Potential object or list of such objects.
    vc : float, optional
        Circular velocity at ro (km/s). Default: 1.0.
    ro : float, optional
        Distance to the Galactic center (kpc). Default: 1.0.

    Returns
    -------
    tuple
        Tuple containing energy and angular momentum.

    Notes
    -----
    - 2012-11-30 - Written - Bovy (IAS)

    """
    return (
        _evaluatePotentials(pot, R, z) + vR**2.0 / 2.0 + vT**2.0 / 2.0 + vz**2.0 / 2.0,
        R * vT,
    )


def potentialStaeckel(u, v, pot, delta):
    """
    Return the potential.

    Parameters
    ----------
    u : float
        Confocal u.
    v : float
        Confocal v.
    pot : Potential object
        Potential.
    delta : float
        Focus.

    Returns
    -------
    float
        Potential at (u, v).

    Notes
    -----
    - 2012-11-29 - Written - Bovy (IAS)
    """
    R, z = coords.uv_to_Rz(u, v, delta=delta)
    return _evaluatePotentials(pot, R, z)


def FRStaeckel(u, v, pot, delta):  # pragma: no cover because unused
    """
    Return the radial force.

    Parameters
    ----------
    u : float
        Confocal u.
    v : float
        Confocal v.
    pot : Potential object
        Potential.
    delta : float
        Focus.

    Returns
    -------
    float
        Radial force.

    Notes
    -----
    - 2012-11-30 - Written - Bovy (IAS)

    """
    R, z = coords.uv_to_Rz(u, v, delta=delta)
    return _evaluateRforces(pot, R, z)


def FZStaeckel(u, v, pot, delta):  # pragma: no cover because unused
    """
    Return the vertical force.

    Parameters
    ----------
    u : float
        Confocal u.
    v : float
        Confocal v.
    pot : Potential object
        Potential.
    delta : float
        Focus.

    Returns
    -------
    Ffloat
        Vertical force.

    Notes
    -----
    - 2012-11-30 - Written - Bovy (IAS)
    """
    R, z = coords.uv_to_Rz(u, v, delta=delta)
    return _evaluatezforces(pot, R, z)


def _JRStaeckelIntegrand(u, E, Lz, I3U, delta, u0, sinh2u0, v0, sin2v0, potu0v0, pot):
    return numpy.sqrt(
        _JRStaeckelIntegrandSquared(
            u, E, Lz, I3U, delta, u0, sinh2u0, v0, sin2v0, potu0v0, pot
        )
    )


def _JRStaeckelIntegrandSquared(
    u, E, Lz, I3U, delta, u0, sinh2u0, v0, sin2v0, potu0v0, pot
):
    # potu0v0= potentialStaeckel(u0,v0,pot,delta)
    """The J_R integrand: p^2_u(u)/2/delta^2"""
    sinh2u = numpy.sinh(u) ** 2.0
    dU = (sinh2u + sin2v0) * potentialStaeckel(u, v0, pot, delta) - (
        sinh2u0 + sin2v0
    ) * potu0v0
    return E * sinh2u - I3U - dU - Lz**2.0 / 2.0 / delta**2.0 / sinh2u


def _JzStaeckelIntegrand(v, E, Lz, I3V, delta, u0, cosh2u0, sinh2u0, potu0pi2, pot):
    return numpy.sqrt(
        _JzStaeckelIntegrandSquared(
            v, E, Lz, I3V, delta, u0, cosh2u0, sinh2u0, potu0pi2, pot
        )
    )


def _JzStaeckelIntegrandSquared(
    v, E, Lz, I3V, delta, u0, cosh2u0, sinh2u0, potu0pi2, pot
):
    # potu0pi2= potentialStaeckel(u0,numpy.pi/2.,pot,delta)
    """The J_z integrand: p_v(v)/2/delta^2"""
    sin2v = numpy.sin(v) ** 2.0
    dV = cosh2u0 * potu0pi2 - (sinh2u0 + sin2v) * potentialStaeckel(u0, v, pot, delta)
    return E * sin2v + I3V + dV - Lz**2.0 / 2.0 / delta**2.0 / sin2v


def _uminUmaxFindStart(
    u, E, Lz, I3U, delta, u0, sinh2u0, v0, sin2v0, potu0v0, pot, umax=False
):
    """
    Find adequate start or end points to solve for umin and umax

    Parameters
    ----------
    u : float
        Current value of the coordinate to solve for (either umin or umax)
    E : float
        Energy
    Lz : float
        Angular momentum along z
    I3U : float
        Third isolating integral of motion
    delta : float
        Focus parameter of the confocal coordinate system
    u0 : float
        u coordinate of the center of the coordinate system
    sinh2u0 : float
        Hyperbolic sine of twice the u coordinate of the center of the coordinate system
    v0 : float
        v coordinate of the center of the coordinate system
    sin2v0 : float
        Sine of twice the v coordinate of the center of the coordinate system
    potu0v0 : float
        Potential at the center of the coordinate system
    pot : Potential object
        Instance of a galpy Potential object
    umax : bool, optional
        If True, solve for umax instead of umin (default is False)

    Returns
    -------
    float
        Adequate start or end point to solve for umin or umax

    Notes
    -----
    - 2012-11-30 - Written - Bovy (IAS)
    """
    if umax:
        utry = u * 1.1
    else:
        utry = u * 0.9
    prevu = u
    while (
        _JRStaeckelIntegrandSquared(
            utry, E, Lz, I3U, delta, u0, sinh2u0, v0, sin2v0, potu0v0, pot
        )
        >= 0.0
        and utry > 0.000000001
    ):
        prevu = utry
        if umax:
            if utry > 100.0:
                raise UnboundError("Orbit seems to be unbound")
            utry *= 1.1
        else:
            utry *= 0.9
    if utry < 0.000000001:
        return (0.0, prevu)
    return (utry, prevu)


def _vminFindStart(v, E, Lz, I3V, delta, u0, cosh2u0, sinh2u0, potu0pi2, pot):
    """
    Find adequate start point to solve for vmin

    Parameters
    ----------
    v : float
        Velocity
    E : float
        Energy
    Lz : float
        Angular momentum along z-axis
    I3V : float
        Third isolating integral
    delta : float
        Staeckel delta parameter
    u0 : float
        Staeckel energy
    cosh2u0 : float
        Hyperbolic cosine squared of u0
    sinh2u0 : float
        Hyperbolic sine squared of u0
    potu0pi2 : float
        Potential at u0 times pi/2
    pot : Potential object
        galpy Potential object

    Returns
    -------
    float
        Adequate start point to solve for vmin

    Notes
    -----
    - 2012-11-28 - Written - Bovy (IAS)
    """
    vtry = 0.9 * v
    while (
        _JzStaeckelIntegrandSquared(
            vtry, E, Lz, I3V, delta, u0, cosh2u0, sinh2u0, potu0pi2, pot
        )
        >= 0.0
        and vtry > 0.000000001
    ):
        vtry *= 0.9
    if vtry < 0.000000001:
        return 0.0
    return vtry if vtry >= 0.000000001 else 0.0


@potential_physical_input
@physical_conversion("position", pop=True)
def estimateDeltaStaeckel(pot, R, z, no_median=False, delta0=1e-6):
    """
    Estimate a good value for delta using eqn. (9) in Sanders (2012)

    Parameters
    ----------
    pot : Potential instance or list thereof
    R : float or numpy.ndarray
        coordinates
    z : float or numpy.ndarray
        coordinates
    no_median : bool, optional
        if True, and input is array, return all calculated values of delta (useful for quickly estimating delta for many phase space points)
    delta0 : float, optional
        value to return when delta<delta0 (because actionAngleStaeckel does not work with delta=0 exactly)

    Returns
    -------
    float or numpy.ndarray
        estimate of delta

    Notes
    -----
    - 2013-08-28 - Written - Bovy (IAS)
    - 2016-02-20 - Changed input order to allow physical conversions - Bovy (UofT)
    - 2022-09-14 - Deal with numerical issues with SCF/DiskSCFPotentials - Bovy (UofT)
    - 2022-09-15 - Add delta0 - Bovy (UofT)
    """
    pot = flatten_potential(pot)
    # We'll special-case delta<0 when the potential includes SCF/DiskSCF components
    pot_includes_scf = (
        numpy.any(
            [
                isinstance(p, SCFPotential) or isinstance(p, DiskSCFPotential)
                for p in pot
            ]
        )
        if isinstance(pot, list)
        else isinstance(pot, SCFPotential) or isinstance(pot, DiskSCFPotential)
    )
    if numpy.any(z == 0.0):
        if isinstance(z, numpy.ndarray):
            z[z == 0.0] = 1e-4
        else:
            z = 1e-4
    if isinstance(R, numpy.ndarray):
        delta2 = numpy.array(
            [
                (
                    z[ii] ** 2.0
                    - R[ii] ** 2.0  # eqn. (9) has a sign error
                    + (
                        3.0 * R[ii] * _evaluatezforces(pot, R[ii], z[ii])
                        - 3.0 * z[ii] * _evaluateRforces(pot, R[ii], z[ii])
                        + R[ii]
                        * z[ii]
                        * (
                            evaluateR2derivs(pot, R[ii], z[ii], use_physical=False)
                            - evaluatez2derivs(pot, R[ii], z[ii], use_physical=False)
                        )
                    )
                    / evaluateRzderivs(pot, R[ii], z[ii], use_physical=False)
                )
                for ii in range(len(R))
            ]
        )
        indx = (delta2 < delta0**2.0) * ((delta2 > -(10.0**-10.0)) + pot_includes_scf)
        delta2[indx] = delta0**2.0
        if not no_median:
            delta2 = numpy.median(delta2[True ^ numpy.isnan(delta2)])
    else:
        delta2 = (
            z**2.0
            - R**2.0  # eqn. (9) has a sign error
            + (
                3.0 * R * _evaluatezforces(pot, R, z)
                - 3.0 * z * _evaluateRforces(pot, R, z)
                + R
                * z
                * (
                    evaluateR2derivs(pot, R, z, use_physical=False)
                    - evaluatez2derivs(pot, R, z, use_physical=False)
                )
            )
            / evaluateRzderivs(pot, R, z, use_physical=False)
        )
        if delta2 < delta0**2.0 and (delta2 > -(10.0**-10.0) or pot_includes_scf):
            delta2 = delta0**2.0
    return numpy.sqrt(delta2)
