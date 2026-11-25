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

from ..potential.Potential import flatten as flatten_potential
from ..util import conversion
from .actionAngleAdiabaticApprox import (
    _aA_from_inverse,
    _auxiliary_potential,
    _deform,
    _switch,
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
        pot : Potential or a combined potential formed using addition (pot1+pot2+…)
            Potential to calculate phase-space points for.
        aAI : actionAngleInverse instance
            Inverse action-angle instance of the auxiliary potential that is adiabatically transformed into pot (for example, actionAngleIsochroneInverse, actionAngleSphericalInverse, or actionAngleStaeckelInverse; see the Notes for the latter); it has to return points in internal units.
        aA : actionAngle instance, optional
            Action-angle instance of the same auxiliary potential, for the frequencies and for calibrate_angles=True; set up automatically for the three classes above.
        tintJ : float or Quantity, optional
            Duration of the adiabatic transformation. Default: 100.
        calibrate_angles : bool, optional
            If True, the angles are those of the usual convention (those of actionAngleAdiabaticApprox with calibrate_angles=True) rather than angles with an offset that is constant on each torus; this makes the transformation three to four times as expensive. Default: False.
        integrate_method : str, optional
            Orbit-integration method. Default: 'dop853_c'.
        dt : float, optional
            Orbit.integrate's dt keyword (for fixed-stepsize integration).
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - The point of aAI's torus with the given actions and angles is integrated for tintJ while the auxiliary potential is slowly changed into pot: the time reversal of the transformation of actionAngleAdiabaticApprox, and so its exact inverse (for npoints=0 there) to the precision of the orbit integration, with the same angles. Use actionAngleAdiabaticApproxInverse.from_forward to set up the inverse of an actionAngleAdiabaticApprox instance.
        - An actionAngleStaeckelInverse instance as aAI= sets the auxiliary potential in one of three ways. Set up for an OblateStaeckelWrapperPotential (or another potential of Staeckel form), it is that potential. Set up with delta= for another potential, it is the Staeckel potential that the instance approximates that potential with. For a family of tori of pot whose Staeckel potential follows its tori (setup_interp=True and delta='fit' or u0='torus'), each torus is transformed from the Staeckel potential that the family assigns to it, such that a single family serves all orbits of a galaxy; tori are then integrated one at a time, in parallel threads.
        - 2026-11-23 - Written - Bovy (UofT)

        """
        actionAngleInverse.__init__(self, *args, **kwargs)
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
        self._integrate_dt = kwargs.get("dt", None)
        # Time reversal of actionAngleAdiabaticApprox's transformation: the
        # same switch run from the auxiliary potential to pot, applied to
        # the point with its velocities reversed
        self._adiabatic_pot = (
            None
            if self._auxiliary_pot is None
            else _switch(self._auxiliary_pot, self._pot, self._tintJ)
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

    @classmethod
    def from_forward(cls, aAA, aAI=None, **kwargs):
        """
        Set up the inverse of an actionAngleAdiabaticApprox instance.

        Parameters
        ----------
        aAA : actionAngleAdiabaticApprox instance
            The forward transformation; the potential, the duration of the transformation, the orbit-integration settings, and the aA= are taken from it.
        aAI : actionAngleInverse instance, optional
            Inverse action-angle instance of the auxiliary potential; only necessary when aAA has none (it has one when it was set up with aAI=, or with an actionAngleIsochrone or actionAngleStaeckel.from_inverse instance as aA=).
        **kwargs
            Other keywords of actionAngleAdiabaticApproxInverse (calibrate_angles=, ro=, vo=).

        Returns
        -------
        actionAngleAdiabaticApproxInverse instance
            The exact inverse of aAA when that has npoints=0 (also for calibrate_angles=True in both).

        Notes
        -----
        - 2026-11-24 - Written as a class method (was the aAA= keyword) - Bovy (UofT)

        """
        if aAI is None:
            aAI = aAA._aAI
        if aAI is None:
            raise ValueError(
                "aAI= is necessary when the actionAngleAdiabaticApprox "
                "instance does not have the inverse of its aA="
            )
        return cls(
            pot=aAA._pot,
            aAI=aAI,
            aA=aAA._aA,
            tintJ=aAA._tintJ,
            integrate_method=aAA._integrate_method,
            dt=aAA._integrate_dt,
            **kwargs,
        )

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
        if self._aAA is not None:
            # (the tori of an aAI= with a Staeckel potential for each torus
            # are kept for one evaluation)
            self._aAA._wrappers.clear()
            self._aAA._tori.clear()
        acts = [t.ravel() for t in args[:3]]
        angles = numpy.array([t.ravel() for t in args[3:]])
        if self._calibrate:
            # The offset of each torus, from the round trip of its point in
            # the potential of aAI=; then the points, and again where an
            # angle is half a turn off
            angles = angles + self._aAA._offset_torus(
                acts, self._auxiliary(acts, angles)[0]
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

    def _auxiliary(self, acts, angles):
        """The points of the tori of aAI= with these actions and angles
        and, for an aAI= with a Staeckel potential for each torus (for
        which they are the points of each torus in its own Staeckel
        potential), the rows (delta, u0) of those potentials (else None)"""
        if self._auxiliary_pot is None:
            return self._aAA._aux_points(acts, angles)
        return (
            numpy.array(
                [
                    numpy.broadcast_to(numpy.ravel(x), acts[0].shape)
                    for x in self._aAI(*acts, *angles, use_physical=False)
                ]
            ),
            None,
        )

    def _morph(self, acts, angles):
        """The points of the tori of aAI= with these actions and angles,
        transformed to pot"""
        xv, models = self._auxiliary(acts, angles)
        flip = numpy.array([1.0, -1.0, -1.0, 1.0, -1.0, 1.0])
        out = flip * _deform(
            self._pot,
            self._auxiliary_pot,
            flip * xv.T,
            models,
            numpy.array([0.0, self._tintJ]),
            self._integrate_method,
            self._integrate_dt,
            back=True,
            wrapper=None if self._aAA is None else self._aAA._wrapper,
        )
        return tuple(out.T)

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
        if self._aAA is None:
            raise RuntimeError(
                "The frequencies of actionAngleAdiabaticApproxInverse require "
                "aA= at initialization: the action-angle instance that aAI= "
                "inverts, which galpy cannot set up from this aAI="
            )
        xv = self._evaluate(jr, jphi, jz, angler, anglephi, anglez)
        shape = xv[0].shape
        # (tori that aAI= does not have are nan)
        flat = numpy.array([x.ravel() for x in xv])
        ok = numpy.all(numpy.isfinite(flat), axis=0)
        freqs = numpy.full((3, flat.shape[1]), numpy.nan)
        if numpy.any(ok):
            freqs[:, ok] = self._aAA.actionsFreqs(*flat[:, ok], use_physical=False)[3:]
        return (*xv, *(f.reshape(shape) for f in freqs))

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
