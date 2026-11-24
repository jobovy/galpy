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

from ..potential import DehnenSmoothWrapperPotential
from ..potential.Potential import flatten as flatten_potential
from ..util import conversion
from .actionAngleAdiabaticApprox import actionAngleAdiabaticApprox
from .actionAngleInverse import actionAngleInverse


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
            Inverse action-angle instance (for example, actionAngleIsochroneInverse, actionAngleSphericalInverse, or actionAngleStaeckelInverse) for the potential that is adiabatically transformed into pot; it has to return points in internal units.
        aA : actionAngle instance, optional
            Action-angle instance for the same potential as aAI (the aA= of the corresponding actionAngleAdiabaticApprox); only necessary for the frequencies.
        tintJ : float or Quantity, optional
            Duration of the adiabatic transformation.
        integrate_method : str, optional
            Orbit-integration method.
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
        if "pot" not in kwargs:  # pragma: no cover
            raise OSError("Must specify pot= for actionAngleAdiabaticApproxInverse")
        if "aAI" not in kwargs:  # pragma: no cover
            raise OSError("Must specify aAI= for actionAngleAdiabaticApproxInverse")
        self._pot = flatten_potential(kwargs["pot"])
        self._aAI = kwargs["aAI"]
        self._auxiliary_pot = self._aAI._pot
        self._tintJ = conversion.parse_time(
            kwargs.get("tintJ", 100.0), ro=self._ro, vo=self._vo
        )
        self._integrate_method = kwargs.get("integrate_method", "dop853_c")
        self._integrate_dt = kwargs.get("dt", None)
        # Time reversal of actionAngleAdiabaticApprox's transformation: the
        # same switch run from the auxiliary potential to pot, applied to
        # the point with its velocities reversed
        self._adiabatic_pot = DehnenSmoothWrapperPotential(
            pot=self._auxiliary_pot, tform=0.0, tsteady=self._tintJ, decay=True
        ) + DehnenSmoothWrapperPotential(
            pot=self._pot, tform=0.0, tsteady=self._tintJ, decay=False
        )
        self._aAA = (
            actionAngleAdiabaticApprox(
                pot=self._pot,
                aA=kwargs["aA"],
                tintJ=self._tintJ,
                ntintJ=2,
                integrate_method=self._integrate_method,
                dt=self._integrate_dt,
            )
            if "aA" in kwargs
            else None
        )
        self._check_consistent_units()
        return None

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
        from ..orbit import Orbit

        args = numpy.broadcast_arrays(
            *(
                numpy.atleast_1d(numpy.asarray(t, dtype="float"))
                for t in (jr, jphi, jz, angler, anglephi, anglez)
            )
        )
        shape = args[0].shape
        R, vR, vT, z, vz, phi = (
            numpy.broadcast_to(numpy.ravel(x), (args[0].size,))
            for x in self._aAI(*(t.ravel() for t in args), use_physical=False)
        )
        orb = Orbit(numpy.array([R, -vR, -vT, z, -vz, phi]).T)
        orb.integrate(
            numpy.array([0.0, self._tintJ]),
            self._adiabatic_pot,
            method=self._integrate_method,
            dt=self._integrate_dt,
            progressbar=False,
        )
        return tuple(
            (sign * getattr(orb, q)(self._tintJ, use_physical=False)).reshape(shape)
            for q, sign in (
                ("R", 1.0),
                ("vR", -1.0),
                ("vT", -1.0),
                ("z", 1.0),
                ("vz", -1.0),
                ("phi", 1.0),
            )
        )

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
                "aA= at initialization"
            )
        xv = self._evaluate(jr, jphi, jz, angler, anglephi, anglez)
        shape = xv[0].shape
        freqs = self._aAA.actionsFreqs(*(x.ravel() for x in xv), use_physical=False)[3:]
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
        - The frequencies are those that actionAngleAdiabaticApprox returns for one point of each torus and require aA= at initialization.
        - 2026-11-23 - Written - Bovy (UofT).
        """
        return self._xvFreqs(jr, jphi, jz, 1.0, 0.0, 1.0)[6:]
