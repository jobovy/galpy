###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleAdiabaticApprox
#
#             Calculate actions-angle coordinates for any potential by adiabatically
#             transforming the potential to an isochrone potential
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

from ..potential import DehnenSmoothWrapperPotential, IsochronePotential, MWPotential
from ..potential.Potential import flatten as flatten_potential
from ..util import conversion, galpyWarning
from .actionAngle import actionAngle
from .actionAngleIsochrone import actionAngleIsochrone


class actionAngleAdiabaticApprox(actionAngle):
    """Action-angle formalism using an adiabatic transformation to an isochrone potential"""

    def __init__(self, *args, **kwargs):
        """
        Initialize an actionAngleAdiabaticApprox object.

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
        npoints : int, optional
            Number of points along the orbit to integrate to estimate the actions, frequencies, and angles.
        npoints_dt : float, optional
            Amount of time to integrate forward and backwards for the npoints points (can be Quantity).
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2025-09-23 - Started - Bovy (UofT).

        """
        actionAngle.__init__(self, ro=kwargs.get("ro", None), vo=kwargs.get("vo", None))
        if "pot" not in kwargs:  # pragma: no cover
            raise OSError("Must specify pot= for actionAngleAdiabaticApprox")
        self._pot = flatten_potential(kwargs["pot"])
        if self._pot == MWPotential:
            warnings.warn(
                "Use of MWPotential as a Milky-Way-like potential is deprecated; galpy.potential.MWPotential2014, a potential fit to a large variety of dynamical constraints (see Bovy 2015), is the preferred Milky-Way-like potential in galpy",
                galpyWarning,
            )
        if (
            "b" not in kwargs and "ip" not in kwargs and "aAI" not in kwargs
        ):  # pragma: no cover
            raise OSError(
                "Must specify b=, ip=, or aAI= for actionAngleAdiabaticApprox"
            )
        if "aAI" in kwargs:
            if not isinstance(kwargs["aAI"], actionAngleIsochrone):  # pragma: no cover
                raise OSError(
                    "'Provided aAI= does not appear to be an instance of an actionAngleIsochrone"
                )
            self._aAI = kwargs["aAI"]
            self._ip = self._aAI._pot
        elif "ip" in kwargs:
            ip = kwargs["ip"]
            if not isinstance(ip, IsochronePotential):  # pragma: no cover
                raise OSError(
                    "'Provided ip= does not appear to be an instance of an IsochronePotential"
                )
            self._ip = ip
            self._aAI = actionAngleIsochrone(ip=self._ip)
        else:
            b = conversion.parse_length(kwargs["b"], ro=self._ro)
            self._ip = IsochronePotential(b=b, normalize=1.0)
            self._aAI = actionAngleIsochrone(ip=self._ip)
        self._tintJ = conversion.parse_time(
            kwargs.get("tintJ", 100.0), ro=self._ro, vo=self._vo
        )
        self._ntintJ = kwargs.get("ntintJ", 10_000)
        self._integrate_dt = kwargs.get("dt", None)
        self._tsJ = numpy.linspace(0.0, self._tintJ, self._ntintJ)
        self._integrate_method = kwargs.get("integrate_method", "dop853_c")
        self._npoints = int(numpy.floor(kwargs.get("npoints", 2) / 2) * 2)
        if self._npoints < 2:
            raise OSError("npoints= must be at least 2")
        self._npoints_dt = conversion.parse_time(
            kwargs.get("npoints_dt", 0.1), ro=self._ro, vo=self._vo
        )
        self._npoints_ts = numpy.linspace(0.0, self._npoints_dt, self._npoints // 2 + 1)
        self._vanderA = numpy.vander(
            numpy.linspace(-self._npoints_dt, self._npoints_dt, self._npoints + 1),
            N=2,
            increasing=True,
        )
        self._adiabatic_pot = DehnenSmoothWrapperPotential(
            pot=self._pot, tform=0.0, tsteady=self._tintJ, decay=True
        ) + DehnenSmoothWrapperPotential(
            pot=self._ip, tform=0.0, tsteady=self._tintJ, decay=False
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

        Returns
        -------
        tuple
            (jr,lz,jz)

        Notes
        -----
        - 2025-09-23 - Written - Bovy (UofT)
        """
        R, vR, vT, z, vz, phi = self._parse_args(*args)
        jr, lz, jz = self._aAI(R, vR, vT, z, vz, phi)
        return (numpy.mean(jr, axis=1), numpy.mean(lz, axis=1), numpy.mean(jz, axis=1))

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

        Returns
        -------
        tuple
            (jr,lz,jz,Or,Op,Oz)

        Notes
        -----
        - 2025-09-23 - Written - Bovy (UofT).
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

        Returns
        -------
        tuple
            (jr,lz,jz,Or,Op,Oz,angler,anglephi,anglez)

        Notes
        -----
        - 2025-09-23 - Written - Bovy (UofT).
        """
        R, vR, vT, z, vz, phi = self._parse_args(*args)
        jr, lz, jz, _, _, _, ar, ap, az = self._aAI.actionsFreqsAngles(
            R, vR, vT, z, vz, phi
        )
        jr, lz, jz = (
            numpy.mean(jr, axis=1),
            numpy.mean(lz, axis=1),
            numpy.mean(jz, axis=1),
        )
        # Fit for the frequency and angle for each dimension
        angle_out, freq_out = (), ()
        for angle in (ar, ap, az):
            angleT = numpy.unwrap(numpy.reshape(angle, R.shape), axis=1)
            out = linalg.lstsq(self._vanderA, angleT.T)[0]
            angle_out += (out[0],)
            freq_out += (out[1],)
        return (jr, lz, jz, *freq_out, *angle_out)

    def _parse_args(self, *args):
        """Helper function to parse the arguments to the __call__ and actionsFreqsAngles functions"""
        from ..orbit import Orbit

        if len(args) == 5 or len(args) == 3:  # pragma: no cover
            raise ValueError("Must specify phi for actionAngleAdiabaticApprox")
        elif len(args) == 6 or len(args) == 4:
            if len(args) == 6:
                R, vR, vT, z, vz, phi = args
            else:
                R, vR, vT, phi = args
                z, vz = numpy.zeros_like(R), numpy.zeros_like(R)
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
        # Integrate orbit forward and backward to get multiple _npoints_ts to average
        # actions and estimate frequencies and angles from fit
        o_forward = Orbit([R, vR, vT, z, vz, phi])
        o_forward.integrate(self._npoints_ts, self._pot, method=self._integrate_method)
        o_backward = o_forward()
        o_backward.integrate(
            -self._npoints_ts, self._pot, method=self._integrate_method
        )
        phase_space_input = []
        for attr in ["R", "vR", "vT", "z", "vz", "phi"]:
            this_coord = numpy.empty((len(R), self._npoints + 1))
            this_coord[:, self._npoints // 2 :] = getattr(o_forward, attr)(
                self._npoints_ts
            )
            this_coord[:, : self._npoints // 2] = getattr(o_backward, attr)(
                -self._npoints_ts[1:][::-1]
            )
            phase_space_input.append(this_coord)
        # Need to do a bit of shape wrangling to make sure the resulting Orbit is
        # also (len(R), self._npoints + 1)
        os_for_actions = Orbit(
            numpy.rollaxis(
                numpy.array(phase_space_input),
                0,
                start=3,
            )
        )
        os_for_actions.integrate(
            self._tsJ,
            self._adiabatic_pot,
            method=self._integrate_method,
            dt=self._integrate_dt,
        )
        return (
            os_for_actions.R(self._tsJ[-1]),
            os_for_actions.vR(self._tsJ[-1]),
            os_for_actions.vT(self._tsJ[-1]),
            os_for_actions.z(self._tsJ[-1]),
            os_for_actions.vz(self._tsJ[-1]),
            os_for_actions.phi(self._tsJ[-1]),
        )
