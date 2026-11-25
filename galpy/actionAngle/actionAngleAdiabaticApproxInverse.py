###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleAdiabaticApproxInverse
#
#             Calculate (x,v) coordinates for any potential from given actions
#             and angles by adiabatically transforming the potential of an
#             inverse action-angle instance into it; the inverse of
#             actionAngleAdiabaticApprox
#
###############################################################################
import numpy

from ..backend import coerce_coords, like
from ..potential import DehnenSmoothWrapperPotential
from ..potential.Potential import flatten as flatten_potential
from ..util import conversion
from .actionAngleAdiabaticApprox import (
    _aA_from_inverse,
    _auxiliary_potential,
    _in_namespace,
    _integrate_backend,
    actionAngleAdiabaticApprox,
)
from .actionAngleInverse import actionAngleInverse
from .actionAngleIsochrone import actionAngleIsochrone


class actionAngleAdiabaticApproxInverse(actionAngleInverse):
    """Inverse action-angle formalism using an adiabatic transformation from an explicitly-integrable potential: the exact inverse of actionAngleAdiabaticApprox for the same transformation"""

    def __init__(self, *args, **kwargs):
        """
        Initialize an actionAngleAdiabaticApproxInverse object.

        Parameters
        ----------
        pot : Potential or list of Potentials
            Potential to calculate phase-space points for.
        aAI : actionAngleInverse instance
            Inverse action-angle instance (for example, actionAngleIsochroneInverse, actionAngleSphericalInverse, or actionAngleStaeckelInverse) for the potential that is adiabatically transformed into pot; it has to return points in internal units. For an actionAngleStaeckelInverse instance that was set up with delta= for a potential that is not of Staeckel form, that potential is the Staeckel potential that the instance approximates it with (a numerical delta= and u0=: instances whose Staeckel potential follows their tori are not yet supported).
        aA : actionAngle instance, optional
            Action-angle instance for the same potential as aAI (the aA= of the corresponding actionAngleAdiabaticApprox); used for the frequencies, and only necessary for those when aAI is not an actionAngleIsochroneInverse, actionAngleSphericalInverse, or actionAngleStaeckelInverse instance, whose forward transformations are set up from them.
        aAA : actionAngleAdiabaticApprox instance, optional
            Set up the inverse of this instance, in place of pot=, aA=, tintJ=, integrate_method=, and dt=, which are taken from it; aAI= is then only necessary when its aA= is not an actionAngleIsochrone instance.
        tintJ : float or Quantity, optional
            Duration of the adiabatic transformation.
        calibrate_angles : bool, optional
            If True, the angles are those of the usual convention (those of actionAngleAdiabaticApprox with calibrate_angles=True) rather than angles with an offset that is constant on each torus: the offset of each torus is measured with a round trip of the transformation and the transformation is repeated where an angle is half a turn off, which makes it three to four times as expensive. The offset is the same function of the actions as in actionAngleAdiabaticApprox, such that the two calibrated transformations are also each other's inverse. Requires the forward transformation (aA=, or an aAI= that it can be set up from). Default: False.
        integrate_method : str, optional
            Orbit-integration method. With 'diffrax' (jax) or 'torchode'/'torchdiffeq' (PyTorch), backend-array inputs are integrated in the backend, which is what derivatives with respect to parameters of the potentials and higher-order derivatives require (the C methods give first derivatives with respect to the actions and angles).
        integrate_kwargs : dict, optional
            Options for the in-backend integrator (Orbit.integrate's inbackend_kwargs).
        dt : float, optional
            Orbit.integrate's dt keyword (for fixed-stepsize integration).
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - The transformation integrates the point of aAI's torus with the given actions and angles backwards in time through the transformation of actionAngleAdiabaticApprox with the same tintJ, such that it is that transformation's inverse to the precision of the orbit integration, for any tintJ. The angles are therefore those of actionAngleAdiabaticApprox: angle coordinates on the torus that differ from those of other action-angle methods by an offset that is constant on each torus.
        - 2026-11-23 - Written - Bovy (UofT)

        """
        actionAngleInverse.__init__(self, *args, **kwargs)
        if "aAA" in kwargs:
            # The inverse of an actionAngleAdiabaticApprox instance
            aAA = kwargs["aAA"]
            kwargs = dict(kwargs)
            kwargs.update(
                pot=aAA._pot,
                aA=aAA._aA,
                tintJ=aAA._tintJ,
                integrate_method=aAA._integrate_method,
                integrate_kwargs=aAA._integrate_kwargs,
                dt=aAA._integrate_dt,
            )
            if "aAI" not in kwargs and isinstance(aAA._aA, actionAngleIsochrone):
                from .actionAngleIsochroneInverse import actionAngleIsochroneInverse

                kwargs["aAI"] = actionAngleIsochroneInverse(ip=aAA._aA._ip)
            if "aAI" not in kwargs:
                raise ValueError(
                    "aAI= is necessary with aAA= when the aA= of the "
                    "actionAngleAdiabaticApprox instance is not an "
                    "actionAngleIsochrone instance"
                )
        if "pot" not in kwargs:  # pragma: no cover
            raise OSError("Must specify pot= for actionAngleAdiabaticApproxInverse")
        if "aAI" not in kwargs:  # pragma: no cover
            raise OSError("Must specify aAI= for actionAngleAdiabaticApproxInverse")
        self._pot = flatten_potential(kwargs["pot"])
        self._aAI = kwargs["aAI"]
        self._auxiliary_pot = _auxiliary_potential(self._aAI)
        self._tintJ = conversion.parse_time(
            kwargs.get("tintJ", 100.0), ro=self._ro, vo=self._vo
        )
        self._integrate_method = kwargs.get("integrate_method", "dop853_c")
        self._integrate_kwargs = kwargs.get("integrate_kwargs", None)
        self._integrate_dt = kwargs.get("dt", None)
        # Time reversal of actionAngleAdiabaticApprox's transformation: the
        # same switch run from the auxiliary potential to pot, applied to
        # the point with its velocities reversed
        self._adiabatic_pot = DehnenSmoothWrapperPotential(
            pot=self._auxiliary_pot, tform=0.0, tsteady=self._tintJ, decay=True
        ) + DehnenSmoothWrapperPotential(
            pot=self._pot, tform=0.0, tsteady=self._tintJ, decay=False
        )
        # The forward transformation, for the frequencies: with the aA= given
        # or the one that galpy can set up from aAI (None otherwise)
        aA = kwargs["aA"] if "aA" in kwargs else _aA_from_inverse(self._aAI)
        self._aAA = (
            None
            if aA is None
            else actionAngleAdiabaticApprox(
                pot=self._pot,
                aA=aA,
                tintJ=self._tintJ,
                ntintJ=2,
                integrate_method=self._integrate_method,
                integrate_kwargs=self._integrate_kwargs,
                dt=self._integrate_dt,
            )
        )
        self._calibrate = kwargs.get("calibrate_angles", False)
        if self._calibrate and self._aAA is None:
            raise ValueError(
                "calibrate_angles=True requires aA= at initialization: the "
                "action-angle instance that aAI= inverts, which galpy cannot "
                "set up from this aAI="
            )
        if self._aAA is not None:
            # (its half turns from the tori of this aAI=)
            self._aAA._aAI = self._aAI
        self._check_consistent_units()
        return None

    @_in_namespace
    def _evaluate(self, jr, jphi, jz, angler, anglephi, anglez, **kwargs):
        """
        Evaluate the phase-space coordinates (x,v) at actions and angles that broadcast against each other.

        Parameters
        ----------
        jr : float or numpy.ndarray
            Radial action.
        jphi : float or numpy.ndarray
            Azimuthal action.
        jz : float or numpy.ndarray
            Vertical action.
        angler : float or numpy.ndarray
            Radial angle.
        anglephi : float or numpy.ndarray
            Azimuthal angle.
        anglez : float or numpy.ndarray
            Vertical angle.

        Returns
        -------
        tuple
            Phase-space coordinates (R,vR,vT,z,vz,phi), each in the broadcast shape of the actions and angles.

        Notes
        -----
        - 2026-11-23 - Written - Bovy (UofT).
        """
        args = numpy.broadcast_arrays(
            *(
                numpy.atleast_1d(numpy.asarray(t, dtype="float"))
                for t in (jr, jphi, jz, angler, anglephi, anglez)
            )
        )
        shape = args[0].shape
        acts = [t.ravel() for t in args[:3]]
        angles = numpy.array([t.ravel() for t in args[3:]])
        if self._calibrate:
            # The offset of each torus, from the round trip of its point in
            # the potential of aAI=; then the points, and again where an
            # angle is half a turn off
            angles = angles + self._aAA._offset_torus(
                acts, self._auxiliary(acts, angles)
            )
            xv = self._morph(acts, angles)
            half = self._aAA._halfturns(
                numpy.array([t.ravel() for t in args[3:]]), acts, *xv
            )
            if numpy.any(half):
                xv = self._morph(acts, angles + numpy.pi * half)
        else:
            xv = self._morph(acts, angles)
        return tuple(x.reshape(shape) for x in xv)

    def _require_aA(self):
        """The frequencies need the forward transformation"""
        if self._aAA is None:
            raise RuntimeError(
                "The frequencies of actionAngleAdiabaticApproxInverse require "
                "aA= at initialization: the action-angle instance that aAI= "
                "inverts, which galpy cannot set up from this aAI="
            )

    def _backend(self, name, xp, *args, **kwargs):
        """The method name (_evaluate, _xvFreqs, or _Freqs) in the backend
        xp, for traced or differentiated inputs: the same computation with
        the orbits integrated differentiably and the arithmetic in xp"""
        if name != "_evaluate":
            self._require_aA()
        if name == "_Freqs":
            return self._backend("_xvFreqs", xp, *args, 1.0, 0.0, 1.0)[6:]
        args = xp.broadcast_arrays(
            *(xp.atleast_1d(a) for a in coerce_coords(xp, *args))
        )
        shape = args[0].shape
        acts = [xp.reshape(a, (-1,)) for a in args[:3]]
        angles = xp.stack([xp.reshape(a, (-1,)) for a in args[3:]])
        flip = like(acts[0], numpy.array([1.0, -1.0, -1.0, 1.0, -1.0, 1.0]))

        def auxiliary(angles):
            return xp.stack(
                [
                    xp.broadcast_to(xp.reshape(x, (-1,)), acts[0].shape)
                    for x in self._aAI(*acts, *angles, use_physical=False)
                ],
                axis=-1,
            )

        def morph(angles):
            return (
                flip
                * _integrate_backend(
                    self,
                    flip * auxiliary(angles),
                    numpy.array([0.0, self._tintJ]),
                    self._adiabatic_pot,
                )[:, -1]
            )

        if self._calibrate:
            shifted = angles + self._aAA._offset_backend(xp, acts, auxiliary(angles))
            xv = morph(shifted)
            xv = morph(
                shifted + numpy.pi * self._aAA._halfturns_backend(xp, angles, acts, xv)
            )
        else:
            xv = morph(angles)
        out = tuple(xv[:, ii] for ii in range(6))
        if name == "_xvFreqs":
            out += tuple(self._aAA.actionsFreqs(*out, use_physical=False)[3:])
        return tuple(xp.reshape(o, shape) for o in out)

    def _auxiliary(self, acts, angles):
        """The points of the tori of aAI= with these actions and angles"""
        return tuple(
            numpy.broadcast_to(numpy.ravel(x), acts[0].shape)
            for x in self._aAI(*acts, *angles, use_physical=False)
        )

    def _morph(self, acts, angles):
        """The points of the tori of aAI= with these actions and angles,
        transformed to pot"""
        from ..orbit import Orbit

        R, vR, vT, z, vz, phi = self._auxiliary(acts, angles)
        orb = Orbit(numpy.array([R, -vR, -vT, z, -vz, phi]).T)
        orb.integrate(
            numpy.array([0.0, self._tintJ]),
            self._adiabatic_pot,
            method=self._integrate_method,
            dt=self._integrate_dt,
            progressbar=False,
        )
        return tuple(
            sign * getattr(orb, q)(self._tintJ, use_physical=False)
            for q, sign in (
                ("R", 1.0),
                ("vR", -1.0),
                ("vT", -1.0),
                ("z", 1.0),
                ("vz", -1.0),
                ("phi", 1.0),
            )
        )

    @_in_namespace
    def _xvFreqs(self, jr, jphi, jz, angler, anglephi, anglez, **kwargs):
        """
        Evaluate the phase-space coordinates (x,v) and the frequencies at actions and angles that broadcast against each other.

        Parameters
        ----------
        jr : float or numpy.ndarray
            Radial action.
        jphi : float or numpy.ndarray
            Azimuthal action.
        jz : float or numpy.ndarray
            Vertical action.
        angler : float or numpy.ndarray
            Radial angle.
        anglephi : float or numpy.ndarray
            Azimuthal angle.
        anglez : float or numpy.ndarray
            Vertical angle.

        Returns
        -------
        tuple
            Phase-space coordinates (R,vR,vT,z,vz,phi) and the frequencies (OmegaR,Omegaphi,Omegaz), each in the broadcast shape of the actions and angles.

        Notes
        -----
        - The frequencies are those that actionAngleAdiabaticApprox returns for each point and require aA= at initialization.
        - 2026-11-23 - Written - Bovy (UofT).
        """
        self._require_aA()
        xv = self._evaluate(jr, jphi, jz, angler, anglephi, anglez)
        shape = xv[0].shape
        freqs = self._aAA.actionsFreqs(*(x.ravel() for x in xv), use_physical=False)[3:]
        return (*xv, *(f.reshape(shape) for f in freqs))

    @_in_namespace
    def _Freqs(self, jr, jphi, jz, **kwargs):
        """
        Return the frequencies of tori.

        Parameters
        ----------
        jr : float or numpy.ndarray
            Radial action.
        jphi : float or numpy.ndarray
            Azimuthal action.
        jz : float or numpy.ndarray
            Vertical action.

        Returns
        -------
        tuple
            Frequencies (OmegaR,Omegaphi,Omegaz), each in the broadcast shape of the actions.

        Notes
        -----
        - The frequencies are those that actionAngleAdiabaticApprox returns for one point of each torus (that at the angles (1,0,1)); they converge to the frequencies of the torus as tintJ increases, and for a finite tintJ they vary over the torus by about their error, such that they differ from those of xvFreqs at other angles at that level.
        - 2026-11-23 - Written - Bovy (UofT).
        """
        return self._xvFreqs(jr, jphi, jz, 1.0, 0.0, 1.0)[6:]
