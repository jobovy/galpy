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
    CompositePotential,
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
    PotentialError,
    _check_c,
    _check_potential_list_and_deprecate,
    _evaluatePotentials,
    _evaluateRforces,
    _evaluatezforces,
    _isNonAxi,
)
from ..util import coords  # for prolate confocal transforms
from ..util import conversion, galpyWarning
from ..util.conversion import physical_conversion, potential_physical_input
from . import actionAngleStaeckel_c
from .actionAngle import UnboundError, actionAngle
from .actionAngleStaeckel_c import _ext_loaded as ext_loaded


def _coerce_delta_arraylike(delta):
    """Coerce a plain Python sequence delta (allowed by the public API for
    individual-delta inputs) to an ndarray: the backend-agnostic coords
    transforms resolve their namespace from the data, and plain sequences
    are not backend-resolvable. Scalars/arrays pass through untouched."""
    return numpy.array(delta) if isinstance(delta, (list, tuple)) else delta


class actionAngleStaeckel(actionAngle):
    """Action-angle formalism for axisymmetric potentials using Binney (2012)'s Staeckel approximation"""

    def __init__(self, *args, **kwargs):
        """
        Initialize an actionAngleStaeckel object.

        Parameters
        ----------
        pot : potential or a combined potential formed using addition (pot1+pot2+…) (3D)
            The potential or a combined potential formed using addition (pot1+pot2+…).
        delta : float or Quantity
            The focus.
        useu0 : bool, optional
            Use u0 to calculate dV (not recommended). Default is False.
        c : bool, optional
            If True, always use C for calculations. Default is False.
        order : int, optional
            Number of points to use in the Gauss-Legendre numerical integration of the relevant action, frequency, and angle integrals (C path). On the pure-Python path this instead scales the number of panels of the composite chi-anomaly quadrature (nchi = max(2 x order, 20)), which is machine-converged at the default, so increasing it there has no practical effect. Default is 10.
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
        self._pot = _check_potential_list_and_deprecate(kwargs["pot"])
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
        self._delta = _coerce_delta_arraylike(
            conversion.parse_length(self._delta, ro=self._ro)
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
                    R[0],
                    vR[0],
                    vT[0],
                    z[0],
                    vz[0],
                    pot=self._pot,
                    delta=delta[0] if hasattr(delta, "__len__") else delta,
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
            kwargs.pop("c", None)
            kwargs.pop("u0", None)
            Lz = R * vT
            jr = numpy.zeros(len(R))
            jz = numpy.zeros(len(R))
            Omegar = numpy.zeros(len(R))
            Omegaphi = numpy.zeros(len(R))
            Omegaz = numpy.zeros(len(R))
            for ii in range(len(R)):
                tdelta = delta[ii] if hasattr(delta, "__len__") else delta
                singlekw = {
                    "pot": self._pot,
                    "delta": tdelta,
                    "_v0u": numpy.pi / 2.0,
                }
                if self._useu0:
                    E = (
                        _evaluatePotentials(self._pot, R[ii], z[ii])
                        + vR[ii] ** 2.0 / 2.0
                        + vz[ii] ** 2.0 / 2.0
                        + vT[ii] ** 2.0 / 2.0
                    )
                    singlekw["u0"] = calcu0(E, Lz[ii], self._pot, tdelta)[0]
                aASingle = actionAngleStaeckelSingle(
                    R[ii], vR[ii], vT[ii], z[ii], vz[ii], **singlekw
                )
                jr[ii] = numpy.atleast_1d(aASingle.JR(fixed_quad=True, order=order))[0]
                jz[ii] = numpy.atleast_1d(aASingle.Jz(fixed_quad=True, order=order))[0]
                with numpy.errstate(divide="ignore", invalid="ignore"):
                    tOr, tOp, tOz, _, _, _, _ = aASingle.calcFreqs(order=order)
                Omegar[ii] = tOr
                Omegaphi[ii] = tOp
                Omegaz[ii] = tOz
            # Adjustments for close-to-circular orbits (mirror the C wrapper)
            indx = numpy.isnan(Omegar) * (jr < 10.0**-3.0) + numpy.isnan(Omegaz) * (
                jz < 10.0**-3.0
            )
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
            return (jr, Lz, jz, Omegar, Omegaphi, Omegaz)

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
        else:
            if "c" in kwargs and kwargs["c"] and not self._c:  # pragma: no cover
                warnings.warn(
                    "C module not used because potential does not have a C implementation",
                    galpyWarning,
                )
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
            kwargs.pop("c", None)
            kwargs.pop("u0", None)
            Lz = R * vT
            jr = numpy.zeros(len(R))
            jz = numpy.zeros(len(R))
            Omegar = numpy.zeros(len(R))
            Omegaphi = numpy.zeros(len(R))
            Omegaz = numpy.zeros(len(R))
            angler = numpy.zeros(len(R))
            anglephi = numpy.zeros(len(R))
            anglez = numpy.zeros(len(R))
            for ii in range(len(R)):
                tdelta = delta[ii] if hasattr(delta, "__len__") else delta
                singlekw = {
                    "pot": self._pot,
                    "delta": tdelta,
                    "_v0u": numpy.pi / 2.0,
                }
                if self._useu0:
                    E = (
                        _evaluatePotentials(self._pot, R[ii], z[ii])
                        + vR[ii] ** 2.0 / 2.0
                        + vz[ii] ** 2.0 / 2.0
                        + vT[ii] ** 2.0 / 2.0
                    )
                    singlekw["u0"] = calcu0(E, Lz[ii], self._pot, tdelta)[0]
                aASingle = actionAngleStaeckelSingle(
                    R[ii], vR[ii], vT[ii], z[ii], vz[ii], **singlekw
                )
                jr[ii] = numpy.atleast_1d(aASingle.JR(fixed_quad=True, order=order))[0]
                jz[ii] = numpy.atleast_1d(aASingle.Jz(fixed_quad=True, order=order))[0]
                with numpy.errstate(divide="ignore", invalid="ignore"):
                    tOr, tOp, tOz, tar, taphi, taz = aASingle.calcAngles(order=order)
                Omegar[ii] = tOr
                Omegaphi[ii] = tOp
                Omegaz[ii] = tOz
                angler[ii] = tar
                # Assemble Anglephi as in the C wrapper: (raw + phi%2pi)%2pi
                taphi = (taphi + phi[ii] % (2.0 * numpy.pi)) % (2.0 * numpy.pi)
                if taphi < 0.0:  # pragma: no cover (Python % is non-negative)
                    taphi += 2.0 * numpy.pi
                anglephi[ii] = taphi
                anglez[ii] = taz
            # Adjustments for close-to-circular orbits (mirror the C wrapper)
            indx = numpy.isnan(Omegar) * (jr < 10.0**-3.0) + numpy.isnan(Omegaz) * (
                jz < 10.0**-3.0
            )
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
            return (jr, Lz, jz, Omegar, Omegaphi, Omegaz, angler, anglephi, anglez)

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
        delta = _coerce_delta_arraylike(kwargs.get("delta", self._delta))
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
        pot: Potential or a combined potential formed using addition (pot1+pot2+…)
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
        self._delta = _coerce_delta_arraylike(kwargs["delta"])
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
        # All Staeckel integrals (actions, frequencies, angles) use v0=pi/2 for
        # the u (J_R) integral and u0 for the v (J_z) integral, matching the C
        # implementation. (_v0u is still overridable.)
        self._v0u = kwargs.pop("_v0u", numpy.pi / 2.0)
        self._sinv0u = numpy.sin(self._v0u)
        self._potu0v0 = potentialStaeckel(self._u0, self._v0u, self._pot, self._delta)
        # I3U with the dU reference at (u0, v0u); robust to u0!=ux (useu0=True),
        # reduces to the bare I3 when u0=ux.
        self._I3U = (
            self._E * self._sinhux**2.0
            - self._pux**2.0 / 2.0 / self._delta**2.0
            - self._Lz**2.0 / 2.0 / self._delta**2.0 / self._sinhux**2.0
            - (self._sinhux**2.0 + self._sinv0u**2.0)
            * potentialStaeckel(self._ux, self._v0u, self._pot, self._delta)
            + (self._sinhu0**2.0 + self._sinv0u**2.0) * self._potu0v0
        )
        self._u0v = self._u0
        self._coshu0v = numpy.cosh(self._u0v)
        self._sinhu0v = numpy.sinh(self._u0v)
        self._potupi2 = potentialStaeckel(
            self._u0v, numpy.pi / 2.0, self._pot, self._delta
        )
        dV = self._coshu0v**2.0 * self._potupi2 - (
            self._sinhu0v**2.0 + self._sinvx**2.0
        ) * potentialStaeckel(self._u0v, self._vx, self._pot, self._delta)
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
            If True, use the composite chi-anomaly Gauss-Legendre quadrature (machine-converged; order= scales the mesh as nchi = max(2 x order, 20)) instead of adaptive scipy.integrate.quad. Default is False.
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
        order = kwargs.pop("order", 10)
        fixed_quad = kwargs.pop("fixed_quad", False)
        if hasattr(self, "_JR") and self._JR_key == (fixed_quad, order):
            return self._JR
        umin, umax = self.calcUminUmax()
        # print self._ux, self._pux, (umax-umin)/umax
        if (umax - umin) / umax < 10.0**-6:
            return numpy.array([0.0])
        self._JR_key = (fixed_quad, order)
        if fixed_quad:
            # chi-anomaly composite quadrature: machine-converged, with the
            # sqrt turning-point behavior absorbed by the parametrization
            # factor in next line bc integrand=/2delta^2
            self._JR = (
                1.0
                / numpy.pi
                * numpy.sqrt(2.0)
                * self._delta
                * self._chiQuadsU(order=order)[0]
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
                        self._v0u,
                        self._sinv0u**2.0,
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
            If True, use the composite chi-anomaly Gauss-Legendre quadrature (machine-converged; order= scales the mesh as nchi = max(2 x order, 20)) instead of adaptive scipy.integrate.quad. Default is False.
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
        order = kwargs.pop("order", 10)
        fixed_quad = kwargs.pop("fixed_quad", False)
        if hasattr(self, "_JZ") and self._JZ_key == (fixed_quad, order):
            return self._JZ
        vmin = self.calcVmin()
        if (numpy.pi / 2.0 - vmin) < 10.0**-7:
            return numpy.array([0.0])
        self._JZ_key = (fixed_quad, order)
        if fixed_quad:
            # chi-anomaly composite quadrature: machine-converged, with the
            # sqrt turning-point behavior absorbed by the parametrization
            # factor in next line bc integrand=/2delta^2
            self._JZ = (
                2.0
                / numpy.pi
                * numpy.sqrt(2.0)
                * self._delta
                * self._chiQuadsV(order=order)[0]
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
                        self._u0v,
                        self._coshu0v**2.0,
                        self._sinhu0v**2.0,
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
            self._v0u,
            self._sinv0u**2.0,
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
                self._v0u,
                self._sinv0u**2.0,
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
                self._v0u,
                self._sinv0u**2.0,
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
                    self._v0u,
                    self._sinv0u**2.0,
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
                                self._v0u,
                                self._sinv0u**2.0,
                                self._potu0v0,
                                self._pot,
                            ),
                            maxiter=200,
                            xtol=1e-15,
                            rtol=8.9e-16,
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
                    self._v0u,
                    self._sinv0u**2.0,
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
                        self._v0u,
                        self._sinv0u**2.0,
                        self._potu0v0,
                        self._pot,
                    ),
                    maxiter=200,
                    xtol=1e-15,
                    rtol=8.9e-16,
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
                self._v0u,
                self._sinv0u**2.0,
                self._potu0v0,
                self._pot,
            )
            if rstart == 0.0:  # pragma: no cover (plunge to u=0; bound orbits don't)
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
                            self._v0u,
                            self._sinv0u**2.0,
                            self._potu0v0,
                            self._pot,
                        ),
                        maxiter=200,
                        xtol=1e-15,
                        rtol=8.9e-16,
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
                self._v0u,
                self._sinv0u**2.0,
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
                    self._v0u,
                    self._sinv0u**2.0,
                    self._potu0v0,
                    self._pot,
                ),
                maxiter=200,
                xtol=1e-15,
                rtol=8.9e-16,
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
                self._u0v,
                self._coshu0v**2.0,
                self._sinhu0v**2.0,
                self._potupi2,
                self._pot,
            )
            meps = _JzStaeckelIntegrandSquared(
                self._vx - eps,
                E,
                L,
                self._I3V,
                self._delta,
                self._u0v,
                self._coshu0v**2.0,
                self._sinhu0v**2.0,
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
                self._u0v,
                self._coshu0v**2.0,
                self._sinhu0v**2.0,
                self._potupi2,
                self._pot,
            )
            if rstart == 0.0:  # pragma: no cover (reach v=0 pole; bound orbits don't)
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
                            self._u0v,
                            self._coshu0v**2.0,
                            self._sinhu0v**2.0,
                            self._potupi2,
                            self._pot,
                        ),
                        maxiter=200,
                        xtol=1e-15,
                        rtol=8.9e-16,
                    )
                except RuntimeError:  # pragma: no cover
                    raise UnboundError("Orbit seems to be unbound")
        self._vmin = vmin
        return self._vmin

    def _uIntegrandArgs(self):
        return (
            self._E,
            self._Lz,
            self._I3U,
            self._delta,
            self._u0,
            self._sinhu0**2.0,
            self._v0u,
            self._sinv0u**2.0,
            self._potu0v0,
            self._pot,
        )

    def _vIntegrandArgs(self):
        return (
            self._E,
            self._Lz,
            self._I3V,
            self._delta,
            self._u0v,
            self._coshu0v**2.0,
            self._sinhu0v**2.0,
            self._potupi2,
            self._pot,
        )

    def _chiQuadsU(self, order=10, uupp=None):
        """All u quadratures (the action integral and the three 1/p_u
        profile integrals int f/sqrt(S_R) du for f = sinh^2 u, 1,
        1/sinh^2 u) from umin up to uupp (default: the complete integral
        to umax), from a single vectorized evaluation of the momentum on
        the chi mesh; cached"""
        nchi = max(2 * int(order), 20)
        umin, umax = self.calcUminUmax()
        chimax = (
            numpy.pi
            if uupp is None
            else 2.0
            * numpy.arcsin(
                numpy.sqrt(numpy.clip((uupp - umin) / (umax - umin), 0.0, 1.0))
            )
        )
        if not hasattr(self, "_chiQuadsUCache"):
            self._chiQuadsUCache = {}
        if (nchi, chimax) not in self._chiQuadsUCache:
            self._chiQuadsUCache[(nchi, chimax)] = _staeckelChiQuadratures(
                _JRStaeckelIntegrandSquared,
                _dJRStaeckelIntegrandSquareddu,
                self._uIntegrandArgs(),
                umin,
                umax - umin,
                _CHIQUAD_UWEIGHTS,
                nchi=nchi,
                chimax=chimax,
            )
        return self._chiQuadsUCache[(nchi, chimax)]

    def _chiQuadsV(self, order=10, vupp=None):
        """All v quadratures (the action integral and the three 1/p_v
        profile integrals int f/sqrt(S_z) dv for f = sin^2 v, 1,
        1/sin^2 v) from vmin up to vupp (default: to the midplane pi/2),
        from a single vectorized evaluation of the momentum on the chi
        mesh; cached. The anomaly always spans the full v loop
        [vmin, pi - vmin] (the midplane is a symmetry point of S_z, not a
        turning point), so integrating to the midplane is chimax = pi/2"""
        nchi = max(2 * int(order), 20)
        vmin = self.calcVmin()
        chimax = (
            numpy.pi / 2.0
            if vupp is None
            else 2.0
            * numpy.arcsin(
                numpy.sqrt(
                    numpy.clip((vupp - vmin) / (numpy.pi - 2.0 * vmin), 0.0, 1.0)
                )
            )
        )
        if not hasattr(self, "_chiQuadsVCache"):
            self._chiQuadsVCache = {}
        if (nchi, chimax) not in self._chiQuadsVCache:
            self._chiQuadsVCache[(nchi, chimax)] = _staeckelChiQuadratures(
                _JzStaeckelIntegrandSquared,
                _dJzStaeckelIntegrandSquareddv,
                self._vIntegrandArgs(),
                vmin,
                numpy.pi - 2.0 * vmin,
                _CHIQUAD_VWEIGHTS,
                nchi=nchi,
                chimax=chimax,
            )
        return self._chiQuadsVCache[(nchi, chimax)]

    def calcdJR(self, order=10):
        """
        Calculate the derivatives djr/dE, djr/dLz, djr/dI3.

        Parameters
        ----------
        order : int, optional
            Scales the number of panels of the composite chi-anomaly quadrature (nchi = max(2 x order, 20)); machine-converged at the default. Default is 10.

        Returns
        -------
        tuple
            (djrdE, djrdLz, djrdI3)

        Notes
        -----
        - Port of the C calcdJRStaeckel.
        """
        if hasattr(self, "_djrdE") and self._djrd_order == order:
            return (self._djrdE, self._djrdLz, self._djrdI3)
        self._djrd_order = order
        umin, umax = self.calcUminUmax()
        if (umax - umin) / umax < 1e-6:  # circular
            self._djrdE = 0.0
            self._djrdLz = 0.0
            self._djrdI3 = 0.0
            return (self._djrdE, self._djrdLz, self._djrdI3)
        _, (djrdE, djrdI3, djrdLz) = self._chiQuadsU(order=order)
        djrdE *= self._delta / numpy.pi / numpy.sqrt(2.0)
        djrdLz *= -self._Lz / numpy.pi / numpy.sqrt(2.0) / self._delta
        djrdI3 *= -self._delta / numpy.pi / numpy.sqrt(2.0)
        self._djrdE = djrdE
        self._djrdLz = djrdLz
        self._djrdI3 = djrdI3
        return (self._djrdE, self._djrdLz, self._djrdI3)

    def calcdJz(self, order=10):
        """
        Calculate the derivatives djz/dE, djz/dLz, djz/dI3.

        Parameters
        ----------
        order : int, optional
            Scales the number of panels of the composite chi-anomaly quadrature (nchi = max(2 x order, 20)); machine-converged at the default. Default is 10.

        Returns
        -------
        tuple
            (djzdE, djzdLz, djzdI3)

        Notes
        -----
        - Port of the C calcdJzStaeckel.
        """
        if hasattr(self, "_djzdE") and self._djzd_order == order:
            return (self._djzdE, self._djzdLz, self._djzdI3)
        self._djzd_order = order
        vmin = self.calcVmin()
        if (numpy.pi / 2.0 - vmin) / numpy.pi * 2.0 < 1e-6:  # circular
            self._djzdE = 0.0
            self._djzdLz = 0.0
            self._djzdI3 = 0.0
            return (self._djzdE, self._djzdLz, self._djzdI3)
        _, (djzdE, djzdI3, djzdLz) = self._chiQuadsV(order=order)
        djzdE *= numpy.sqrt(2.0) * self._delta / numpy.pi
        djzdLz *= -self._Lz * numpy.sqrt(2.0) / numpy.pi / self._delta
        djzdI3 *= numpy.sqrt(2.0) * self._delta / numpy.pi
        self._djzdE = djzdE
        self._djzdLz = djzdLz
        self._djzdI3 = djzdI3
        return (self._djzdE, self._djzdLz, self._djzdI3)

    def calcFreqs(self, order=10):
        """
        Calculate the frequencies Omegar, Omegaphi, Omegaz and the dI3/dJ
        derivatives in the Staeckel approximation.

        Parameters
        ----------
        order : int, optional
            Scales the number of panels of the composite chi-anomaly quadrature (nchi = max(2 x order, 20)); machine-converged at the default. Default is 10.

        Returns
        -------
        tuple
            (Omegar, Omegaphi, Omegaz, dI3dJR, dI3dJz, dI3dLz, detA)

        Notes
        -----
        - Port of the C calcFreqsFromDerivsStaeckel + calcdI3dJFromDerivsStaeckel.
        """
        djrdE, djrdLz, djrdI3 = self.calcdJR(order=order)
        djzdE, djzdLz, djzdI3 = self.calcdJz(order=order)
        detA = djrdE * djzdI3 - djzdE * djrdI3
        if detA == 0.0:
            # Exactly circular: the derivatives all vanish (circular guards in
            # calcdJR/calcdJz). The C path gets IEEE 0/0=NaN here and the caller
            # substitutes epifreq/omegac/verticalfreq; a Python scalar 0.0/0.0
            # would raise instead, so emit NaN explicitly to trigger that path.
            nan = numpy.nan
            return (nan, nan, nan, nan, nan, nan, detA)
        Omegar = djzdI3 / detA
        Omegaz = -djrdI3 / detA
        Omegaphi = (djrdI3 * djzdLz - djzdI3 * djrdLz) / detA
        dI3dJR = -djzdE / detA
        dI3dJz = djrdE / detA
        dI3dLz = -(djrdE * djzdLz - djzdE * djrdLz) / detA
        return (Omegar, Omegaphi, Omegaz, dI3dJR, dI3dJz, dI3dLz, detA)

    def calcAngles(self, order=10):
        """
        Calculate the angles angler, anglephi, anglez in the Staeckel
        approximation (port of the C calcAnglesStaeckel).

        Parameters
        ----------
        order : int, optional
            Scales the number of panels of the composite chi-anomaly quadrature (nchi = max(2 x order, 20)); machine-converged at the default. Default is 10.

        Returns
        -------
        tuple
            (Omegar, Omegaphi, Omegaz, angler, anglephi, anglez)

        Notes
        -----
        - Port of the C calcAnglesStaeckel.
        """
        umin, umax = self.calcUminUmax()
        if (umax - umin) / umax < 1e-6:  # circular
            # Angles are 0 (as in C calcAnglesStaeckel); the frequencies are
            # left to the close-to-circular fallback in the caller (they are
            # NaN/inf here, mirroring the C extension).
            Omegar, Omegaphi, Omegaz = self.calcFreqs(order=order)[:3]
            return (Omegar, Omegaphi, Omegaz, 0.0, 0.0, 0.0)
        djrdE, djrdLz, djrdI3 = self.calcdJR(order=order)
        djzdE, djzdLz, djzdI3 = self.calcdJz(order=order)
        Omegar, Omegaphi, Omegaz, dI3dJR, dI3dJz, dI3dLz, _ = self.calcFreqs(
            order=order
        )
        delta = self._delta
        Lz = self._Lz
        sqrt2 = numpy.sqrt(2.0)
        ux = self._ux
        vx = self._vx
        pux = self._pux
        pvx = self._pvx
        vmin = self._vmin

        # Partial-oscillation integrals via the cumulative chi-anomaly
        # quadratures; the (panel, mid) encoding of the C port is kept:
        # "low" integrates [qmin, qmin+mid^2], "high" [qmax-mid^2, qmax]
        def uquad(key, panel, bound, mid):
            idx = {"E": 0, "I3": 1, "Lz": 2}[key]
            if panel == "low":
                return self._chiQuadsU(order=order, uupp=umin + mid**2.0)[1][idx]
            return (
                self._chiQuadsU(order=order)[1][idx]
                - self._chiQuadsU(order=order, uupp=umax - mid**2.0)[1][idx]
            )

        def vquad(key, panel, mid):
            idx = {"E": 0, "I3": 1, "Lz": 2}[key]
            if panel == "low":
                return self._chiQuadsV(order=order, vupp=vmin + mid**2.0)[1][idx]
            return (
                self._chiQuadsV(order=order)[1][idx]
                - self._chiQuadsV(order=order, vupp=numpy.pi / 2.0 - mid**2.0)[1][idx]
            )

        # u-branch (Or1, I3r1, Anglephi-u-term); follows calcAnglesStaeckel @1308
        midpoint_u = umin + 0.5 * (umax - umin)
        if pux > 0.0:
            if ux > midpoint_u:
                mid = numpy.sqrt(umax - ux)
                Or1 = uquad("E", "high", umax, mid)
                I3r1 = -uquad("I3", "high", umax, mid)
                anglephi = (
                    numpy.pi * djrdLz
                    + Lz * uquad("Lz", "high", umax, mid) / delta / sqrt2
                )
                Or1 *= delta / sqrt2
                I3r1 *= delta / sqrt2
                Or1 = numpy.pi * djrdE - Or1
                I3r1 = numpy.pi * djrdI3 - I3r1
            else:
                mid = numpy.sqrt(ux - umin)
                Or1 = uquad("E", "low", umin, mid)
                I3r1 = -uquad("I3", "low", umin, mid)
                anglephi = -Lz * uquad("Lz", "low", umin, mid) / delta / sqrt2
                Or1 *= delta / sqrt2
                I3r1 *= delta / sqrt2
        else:
            if ux > midpoint_u:
                mid = numpy.sqrt(umax - ux)
                Or1 = uquad("E", "high", umax, mid)
                Or1 *= delta / sqrt2
                Or1 = numpy.pi * djrdE + Or1
                I3r1 = -uquad("I3", "high", umax, mid)
                I3r1 *= delta / sqrt2
                I3r1 = numpy.pi * djrdI3 + I3r1
                anglephi = (
                    numpy.pi * djrdLz
                    - Lz * uquad("Lz", "high", umax, mid) / delta / sqrt2
                )
            else:
                mid = numpy.sqrt(ux - umin)
                Or1 = uquad("E", "low", umin, mid)
                Or1 *= delta / sqrt2
                Or1 = 2.0 * numpy.pi * djrdE - Or1
                I3r1 = -uquad("I3", "low", umin, mid)
                I3r1 *= delta / sqrt2
                I3r1 = 2.0 * numpy.pi * djrdI3 - I3r1
                anglephi = (
                    2.0 * numpy.pi * djrdLz
                    + Lz * uquad("Lz", "low", umin, mid) / delta / sqrt2
                )

        # v-branch (Or2, I3r2, phitmp); follows calcAnglesStaeckel @1374
        midpoint_v = vmin + 0.5 * (0.5 * numpy.pi - vmin)
        if pvx > 0.0:
            if vx < midpoint_v or vx > (numpy.pi - midpoint_v):
                mid = (
                    numpy.sqrt(numpy.pi - vx - vmin)
                    if vx > 0.5 * numpy.pi
                    else numpy.sqrt(vx - vmin)
                )
                Or2 = vquad("E", "low", mid) * delta / sqrt2
                I3r2 = vquad("I3", "low", mid) * delta / sqrt2
                phitmp = vquad("Lz", "low", mid) * -Lz / delta / sqrt2
                if vx > 0.5 * numpy.pi:
                    Or2 = numpy.pi * djzdE - Or2
                    I3r2 = numpy.pi * djzdI3 - I3r2
                    phitmp = numpy.pi * djzdLz - phitmp
            else:
                mid = numpy.sqrt(numpy.fabs(0.5 * numpy.pi - vx))
                Or2 = vquad("E", "high", mid) * delta / sqrt2
                I3r2 = vquad("I3", "high", mid) * delta / sqrt2
                phitmp = vquad("Lz", "high", mid) * -Lz / delta / sqrt2
                if vx > 0.5 * numpy.pi:
                    Or2 = 0.5 * numpy.pi * djzdE + Or2
                    I3r2 = 0.5 * numpy.pi * djzdI3 + I3r2
                    phitmp = 0.5 * numpy.pi * djzdLz + phitmp
                else:
                    Or2 = 0.5 * numpy.pi * djzdE - Or2
                    I3r2 = 0.5 * numpy.pi * djzdI3 - I3r2
                    phitmp = 0.5 * numpy.pi * djzdLz - phitmp
        else:
            if vx < midpoint_v or vx > (numpy.pi - midpoint_v):
                mid = (
                    numpy.sqrt(numpy.pi - vx - vmin)
                    if vx > 0.5 * numpy.pi
                    else numpy.sqrt(vx - vmin)
                )
                Or2 = vquad("E", "low", mid) * delta / sqrt2
                I3r2 = vquad("I3", "low", mid) * delta / sqrt2
                phitmp = vquad("Lz", "low", mid) * -Lz / delta / sqrt2
                if vx < 0.5 * numpy.pi:
                    Or2 = 2.0 * numpy.pi * djzdE - Or2
                    I3r2 = 2.0 * numpy.pi * djzdI3 - I3r2
                    phitmp = 2.0 * numpy.pi * djzdLz - phitmp
                else:
                    Or2 = numpy.pi * djzdE + Or2
                    I3r2 = numpy.pi * djzdI3 + I3r2
                    phitmp = numpy.pi * djzdLz + phitmp
            else:
                mid = numpy.sqrt(numpy.fabs(0.5 * numpy.pi - vx))
                Or2 = vquad("E", "high", mid) * delta / sqrt2
                I3r2 = vquad("I3", "high", mid) * delta / sqrt2
                phitmp = vquad("Lz", "high", mid) * -Lz / delta / sqrt2
                if vx < 0.5 * numpy.pi:
                    Or2 = 1.5 * numpy.pi * djzdE + Or2
                    I3r2 = 1.5 * numpy.pi * djzdI3 + I3r2
                    phitmp = 1.5 * numpy.pi * djzdLz + phitmp
                else:
                    Or2 = 1.5 * numpy.pi * djzdE - Or2
                    I3r2 = 1.5 * numpy.pi * djzdI3 - I3r2
                    phitmp = 1.5 * numpy.pi * djzdLz - phitmp

        angler = Omegar * (Or1 + Or2) + dI3dJR * (I3r1 + I3r2)
        anglez = Omegaz * (Or1 + Or2) + dI3dJz * (I3r1 + I3r2) + 0.5 * numpy.pi
        anglephi += phitmp
        anglephi += Omegaphi * (Or1 + Or2) + dI3dLz * (I3r1 + I3r2)
        angler = numpy.fmod(angler, 2.0 * numpy.pi)
        anglez = numpy.fmod(anglez, 2.0 * numpy.pi)
        # Defensive [0, 2pi) normalisation: the >2pi loops are dead (fmod is
        # already < 2pi); the <0 loops only fire for a raw angle that lands just
        # below 0 (orbit/floating-point-dependent) -- exclude from coverage.
        while angler < 0.0:  # pragma: no cover
            angler += 2.0 * numpy.pi
        while anglez < 0.0:  # pragma: no cover
            anglez += 2.0 * numpy.pi
        while angler > 2.0 * numpy.pi:  # pragma: no cover (fmod is already < 2 pi)
            angler -= 2.0 * numpy.pi
        while anglez > 2.0 * numpy.pi:  # pragma: no cover (fmod is already < 2 pi)
            anglez -= 2.0 * numpy.pi
        return (Omegar, Omegaphi, Omegaz, angler, anglephi, anglez)


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
        galpy Potential object or a combined potential formed using addition (pot1+pot2+…).
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


# Derivatives of the under-radical functions S_R/S_z with respect to the
# integration coordinate (analytic, via the forces); these supply the finite
# turning-point limits of the chi-anomaly quadratures below
def _dJRStaeckelIntegrandSquareddu(
    u, E, Lz, I3U, delta, u0, sinh2u0, v0, sin2v0, potu0v0, pot
):
    R, z = coords.uv_to_Rz(u, v0, delta=delta)
    dPhidu = -delta * (
        _evaluateRforces(pot, R, z) * numpy.cosh(u) * numpy.sin(v0)
        + _evaluatezforces(pot, R, z) * numpy.sinh(u) * numpy.cos(v0)
    )
    return (
        E * numpy.sinh(2.0 * u)
        - numpy.sinh(2.0 * u) * potentialStaeckel(u, v0, pot, delta)
        - (numpy.sinh(u) ** 2.0 + sin2v0) * dPhidu
        + Lz**2.0 / delta**2.0 * numpy.cosh(u) / numpy.sinh(u) ** 3.0
    )


def _dJzStaeckelIntegrandSquareddv(
    v, E, Lz, I3V, delta, u0, cosh2u0, sinh2u0, potu0pi2, pot
):
    R, z = coords.uv_to_Rz(u0, v, delta=delta)
    dPhidv = -delta * (
        _evaluateRforces(pot, R, z) * numpy.sinh(u0) * numpy.cos(v)
        - _evaluatezforces(pot, R, z) * numpy.cosh(u0) * numpy.sin(v)
    )
    return (
        E * numpy.sin(2.0 * v)
        - numpy.sin(2.0 * v) * potentialStaeckel(u0, v, pot, delta)
        - (sinh2u0 + numpy.sin(v) ** 2.0) * dPhidv
        + Lz**2.0 / delta**2.0 * numpy.cos(v) / numpy.sin(v) ** 3.0
    )


# Nodes/weights of the composite 10-point Gauss-Legendre rule used by the
# chi-anomaly quadratures: applied per interval of an nchi-panel mesh, the
# error is O((chimax/nchi)^20), so the integrals are machine-converged for
# modest nchi
_CHIQUAD_GLX, _CHIQUAD_GLW = numpy.polynomial.legendre.leggauss(10)
# Weight functions of the 1/p_u and 1/p_v profile integrals, in the order
# (dE, dI3, dLz)
_CHIQUAD_UWEIGHTS = (
    lambda q: numpy.sinh(q) ** 2.0,
    lambda q: numpy.ones_like(q),
    lambda q: 1.0 / numpy.sinh(q) ** 2.0,
)
_CHIQUAD_VWEIGHTS = (
    lambda q: numpy.sin(q) ** 2.0,
    lambda q: numpy.ones_like(q),
    lambda q: 1.0 / numpy.sin(q) ** 2.0,
)


def _staeckelChiQuadratures(
    Ssq, dSsq, args, qmin, D, weights, nchi=20, chimax=numpy.pi
):
    """
    Composite Gauss-Legendre quadratures in the chi anomaly.

    Evaluates int sqrt(S) dq and int f/sqrt(S) dq for all weight functions
    f at once, from a single vectorized evaluation of S, in the anomaly
    parametrization q = qmin + D sin^2(chi/2) that renders every integrand
    regular: with y = sin^2(chi/2) and Q = S/[y(1-y)],
    int sqrt(S) dq = (D/4) int sqrt(Q) sin^2(chi) dchi and
    int f/sqrt(S) dq = D int f/sqrt(Q) dchi, where Q has the finite
    turning-point limits |dS/dq| D that are switched in near the endpoints,
    where the direct evaluation of S is dominated by cancellation.

    Parameters
    ----------
    Ssq : callable
        The under-radical function S(q, *args) = p^2/(2 delta^2); must
        accept an array q.
    dSsq : callable
        dS/dq(q, *args), evaluated only at the turning points.
    args : tuple
        Extra arguments of Ssq and dSsq.
    qmin : float
        Lower turning point.
    D : float
        Full oscillation range: the upper turning point is qmin + D (for
        the v oscillation this is pi - 2 vmin, even when integrating only
        to the midplane).
    weights : tuple of callables
        Weight functions f(q) of the 1/sqrt(S) integrals; must accept an
        array q.
    nchi : int, optional
        Number of panels of the composite rule.
    chimax : float, optional
        Upper integration limit in the anomaly (pi for the complete
        oscillation, pi/2 for the v integral to the midplane, or
        2 arcsin(sqrt([q - qmin]/D)) for an incomplete integral).

    Returns
    -------
    tuple
        (int sqrt(S) dq, [int f/sqrt(S) dq for f in weights])

    Notes
    -----
    - 2026-08-21 - Written - Bovy (UofT)
    """
    chi = numpy.linspace(0.0, chimax, nchi + 1)
    mid = 0.5 * (chi[:-1] + chi[1:])
    half = 0.5 * (chi[1:] - chi[:-1])
    nodes = (mid[:, None] + half[:, None] * _CHIQUAD_GLX[None, :]).ravel()
    wts = (half[:, None] * _CHIQUAD_GLW[None, :]).ravel()
    y = numpy.sin(nodes / 2.0) ** 2.0
    y1my = y * (1.0 - y)
    q = qmin + D * y
    S = Ssq(q, *args)
    with numpy.errstate(invalid="ignore"):
        Q = S / y1my
    # Near the turning points the direct evaluation of S is dominated by
    # cancellation error; there, reconstruct S from the analytic derivative
    # by the trapezoid between the turning point and the node,
    # S ~ (q - q0) [S'(q0) + S'(q)]/2, whose O(y^2) model error is far below
    # the switch threshold (a constant turning-point limit S' D would leave
    # O(y) model error at the switch, which dominates coarse meshes)
    edge = y1my <= 1e-6
    if numpy.any(edge):
        qe, ye = q[edge], y[edge]
        dSe = dSsq(qe, *args)
        Q[edge] = numpy.where(
            ye < 0.5,
            D * (dSsq(qmin, *args) + dSe) / 2.0 / (1.0 - ye),
            D * (-dSsq(qmin + D, *args) - dSe) / 2.0 / ye,
        )
    Q[Q < numpy.finfo(float).tiny] = numpy.finfo(float).tiny
    sqQ = numpy.sqrt(Q)
    action = (D / 4.0) * numpy.sum(wts * sqQ * numpy.sin(nodes) ** 2.0)
    return action, [D * numpy.sum(wts * f(q) / sqQ) for f in weights]


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
    if (
        vtry < 0.000000001
    ):  # pragma: no cover (degenerate v=0 start; bound orbits don't)
        return 0.0
    return vtry if vtry >= 0.000000001 else 0.0


def _u0Equation(u, E, Lz22delta, delta, pot):
    """Port of the C u0Equation: the quantity minimized to obtain u0."""
    sinh2u = numpy.sinh(u) ** 2.0
    cosh2u = numpy.cosh(u) ** 2.0
    dU = cosh2u * potentialStaeckel(u, numpy.pi / 2.0, pot, delta)
    return -(E * sinh2u - dU - Lz22delta / sinh2u)


def calcu0(E, Lz, pot, delta):
    """
    Calculate u0 in the Staeckel approximation (pure-Python port of the C calcu0).

    Parameters
    ----------
    E : numpy.ndarray
        Energy.
    Lz : numpy.ndarray
        Angular momentum along z.
    pot : Potential object
        galpy Potential object or a combined potential.
    delta : float or numpy.ndarray
        Focus.

    Returns
    -------
    numpy.ndarray
        u0 for each point.

    Notes
    -----
    - Port of the C calcu0; minimizes _u0Equation over u in [0.001,100].
    """
    E = numpy.atleast_1d(E)
    Lz = numpy.atleast_1d(Lz)
    delta = numpy.atleast_1d(delta)
    delta_stride = 0 if len(delta) == 1 else 1
    out = numpy.empty(len(E))
    for ii in range(len(E)):
        tdelta = delta[ii * delta_stride]
        Lz22delta = 0.5 * Lz[ii] ** 2.0 / tdelta**2.0
        res = optimize.minimize_scalar(
            _u0Equation,
            bounds=(0.001, 100.0),
            args=(E[ii], Lz22delta, tdelta, pot),
            method="bounded",
            options={"xatol": 1e-12},
        )
        out[ii] = res.x
    return out


@potential_physical_input
@physical_conversion("position", pop=True)
def estimateDeltaStaeckel(pot, R, z, no_median=False, delta0=1e-6):
    """
    Estimate a good value for delta using eqn. (9) in Sanders (2012)

    Parameters
    ----------
    pot : Potential instance or a combined potential formed using addition (pot1+pot2+…)
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

    pot = _check_potential_list_and_deprecate(pot)
    if _isNonAxi(pot):
        raise PotentialError(
            "Calling estimateDeltaStaeckel with non-axisymmetric potentials is not supported"
        )
    # We'll special-case delta<0 when the potential includes SCF/DiskSCF components
    # because their numerical second derivatives can lead to slightly negative delta2
    pot_includes_scf = (
        numpy.any(
            [
                isinstance(p, SCFPotential) or isinstance(p, DiskSCFPotential)
                for p in pot
            ]
        )
        if isinstance(pot, CompositePotential)
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
