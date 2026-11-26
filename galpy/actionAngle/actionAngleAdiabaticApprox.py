###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleAdiabaticApprox
#
#             Calculate actions-angle coordinates for any potential by adiabatically
#             transforming the potential to an explicitly-integrable potential
#
#      methods:
#             __call__: returns (jr,lz,jz)
#             actionsFreqs: returns (jr,lz,jz,Or,Op,Oz)
#             actionsFreqsAngles: returns (jr,lz,jz,Or,Op,Oz,ar,ap,az)
#
###############################################################################
import functools
import warnings

import numpy
from numpy import linalg

from ..backend import as_numpy, get_namespace, is_backend_array
from ..backend._namespaces import (
    asarray_on_device,
    default_device,
    device_of,
    requires_backend_grad,
    under_trace,
)
from ..potential import DehnenSmoothWrapperPotential, MWPotential
from ..potential.Potential import flatten as flatten_potential
from ..util import conversion, galpyWarning
from .actionAngle import actionAngle
from .actionAngleAdiabatic import actionAngleAdiabatic
from .actionAngleIsochrone import actionAngleIsochrone
from .actionAngleSpherical import actionAngleSpherical
from .actionAngleStaeckel import actionAngleStaeckel


def _on_host(method):
    """Run a method of the adiabatic-transformation classes on the host:
    the transformation integrates orbits in C, so the inputs have to be
    concrete and the class's own arithmetic is numpy's (what the auxiliary
    instances, orbits, and potentials return is brought to numpy with
    _host). The output is returned on the namespace and device of
    backend-array inputs or of a forced backend"""

    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        bk = [a for a in args if is_backend_array(a)]
        xp = get_namespace(*bk)
        if xp is numpy:
            return method(self, *args, **kwargs)
        if any(under_trace(a) or requires_backend_grad(a) for a in bk):
            raise NotImplementedError(
                f"{type(self).__name__} integrates orbits in C and cannot be "
                "traced or differentiated"
            )
        dev = device_of(*bk) if bk else default_device(xp)
        out = method(
            self, *(as_numpy(a) if is_backend_array(a) else a for a in args), **kwargs
        )
        return tuple(asarray_on_device(xp, o, dev) for o in out)

    return wrapped


def _host(out):
    """What an auxiliary instance, an orbit, or a potential returns (under a
    forced backend: backend arrays) as numpy arrays"""
    if isinstance(out, tuple):
        return tuple(as_numpy(o) for o in out)
    return as_numpy(out)


class actionAngleAdiabaticApprox(actionAngle):
    """Action-angle formalism using an adiabatic transformation to an explicitly-integrable potential"""

    def __init__(self, *args, **kwargs):
        """
        Initialize an actionAngleAdiabaticApprox object.

        Parameters
        ----------
        pot : Potential or list of Potentials, optional
            Potential to calculate action-angle variables for.
        aA : actionAngleIsochrone, actionAngleSpherical, actionAngleAdiabatic, or actionAngleStaeckel instance
            actionAngle instance for the potential that is being adiabatically transformed to.
        tintJ : float, optional
            Time to integrate orbits for to estimate actions (can be Quantity).
        ntintJ : int, optional
            Number of output times of the orbit integration (only the last one is used).
        integrate_method : str, optional
            Integration method to use.
        dt : float, optional
            orbit.integrate dt keyword (for fixed stepsize integration).
        npoints : int, optional
            Number of points along the orbit, in addition to the point itself, that the actions are averaged over and that the frequencies and angles are fit to (rounded down to an even number). With npoints=0, each point is transformed on its own: the transformation is then exactly canonical and actionAngleAdiabaticApproxInverse with the same tintJ is its inverse, but no frequencies are available.
        npoints_dt : float, optional
            Amount of time to integrate forward and backwards for the npoints points (can be Quantity).
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - The angles are angle coordinates on the torus that advance at the orbital frequencies, but they differ from the angles of other action-angle methods by an offset that is constant on each torus (it depends on the actions, on tintJ, and on the potentials). Many applications do not need the offset (for example, any use of this class together with actionAngleAdiabaticApproxInverse for the same tintJ).
        - 2025-09-23 - Started - Bovy (UofT).
        - 2025-10-15 - Allowed more general target potentials - Bovy (UofT).

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
        if "aA" not in kwargs:  # pragma: no cover
            raise OSError("Must specify aA= for actionAngleAdiabaticApprox")
        if not isinstance(
            kwargs["aA"],
            (
                actionAngleIsochrone,
                actionAngleSpherical,
                actionAngleAdiabatic,
                actionAngleStaeckel,
            ),
        ):  # pragma: no cover
            raise OSError(
                "Provided aA= does not appear to be an instance of a supported actionAngle class"
            )
        self._aA = kwargs["aA"]
        self._target_pot = self._aA._pot
        self._tintJ = conversion.parse_time(
            kwargs.get("tintJ", 100.0), ro=self._ro, vo=self._vo
        )
        self._ntintJ = kwargs.get("ntintJ", 2)
        self._integrate_dt = kwargs.get("dt", None)
        self._tsJ = numpy.linspace(0.0, self._tintJ, self._ntintJ)
        self._integrate_method = kwargs.get("integrate_method", "dop853_c")
        self._npoints = int(numpy.floor(kwargs.get("npoints", 2) / 2) * 2)
        if self._npoints < 0:
            raise OSError("npoints= must be at least 0")
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
            pot=self._target_pot, tform=0.0, tsteady=self._tintJ, decay=False
        )
        # Check the units
        self._check_consistent_units()
        return None

    @_on_host
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
        Rshape = R.shape
        jr, lz, jz = _host(
            tuple(
                self._aA(
                    R.flatten(),
                    vR.flatten(),
                    vT.flatten(),
                    z.flatten(),
                    vz.flatten(),
                    phi.flatten(),
                )
            )
        )
        return (
            numpy.mean(jr.reshape(Rshape), axis=1),
            numpy.mean(lz.reshape(Rshape), axis=1),
            numpy.mean(jz.reshape(Rshape), axis=1),
        )

    @_on_host
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

    @_on_host
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
        Rshape = R.shape
        jr, lz, jz, _, _, _, ar, ap, az = _host(
            tuple(
                self._aA.actionsFreqsAngles(
                    R.flatten(),
                    vR.flatten(),
                    vT.flatten(),
                    z.flatten(),
                    vz.flatten(),
                    phi.flatten(),
                )
            )
        )
        jr, lz, jz = (
            numpy.mean(jr.reshape(Rshape), axis=1),
            numpy.mean(lz.reshape(Rshape), axis=1),
            numpy.mean(jz.reshape(Rshape), axis=1),
        )
        if self._npoints == 0:
            # Each point on its own: its angles, and no frequencies
            nofreq = numpy.full_like(jr, numpy.nan)
            return (
                jr,
                lz,
                jz,
                nofreq,
                nofreq,
                nofreq,
                *(a.reshape(Rshape)[:, 0] % (2.0 * numpy.pi) for a in (ar, ap, az)),
            )
        # Fit for the frequency and angle for each dimension
        angle_out, freq_out = (), ()
        for angle in (ar.reshape(Rshape), ap.reshape(Rshape), az.reshape(Rshape)):
            angleT = numpy.unwrap(numpy.reshape(angle, R.shape), axis=1)
            out = linalg.lstsq(self._vanderA, angleT.T)[0]
            angle_out += (out[0] % (2.0 * numpy.pi),)
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
            self._eval_phi = None
            self._parse_eval_args(*args)
            R = self._eval_R
            vR = self._eval_vR
            vT = self._eval_vT
            z = self._eval_z
            vz = self._eval_vz
            phi = self._eval_phi
            R, vR, vT, z, vz = _host((R, vR, vT, z, vz))
            phi = None if phi is None else _host(phi)
            if phi is None:
                raise ValueError("Must specify phi for actionAngleAdiabaticApprox")
        R, vR, vT, z, vz, phi = (
            numpy.atleast_1d(numpy.asarray(x, dtype="float"))
            for x in (R, vR, vT, z, vz, phi)
        )
        if self._npoints == 0:
            # Each point on its own
            os_for_actions = Orbit(
                numpy.stack([R, vR, vT, z, vz, phi], axis=-1)[:, None, :]
            )
        else:
            # Integrate orbit forward and backward to get multiple
            # _npoints_ts to average actions and estimate frequencies and
            # angles from fit
            orb = Orbit([R, vR, vT, z, vz, phi])
            orb.integrate(
                self._npoints_ts,
                self._pot,
                method=self._integrate_method,
                progressbar=False,
            )
            orb.integrate(
                -self._npoints_ts,
                self._pot,
                method=self._integrate_method,
                progressbar=False,
            )
            # Times in increasing order, the order that self._vanderA assumes
            os_for_actions = orb(
                numpy.concatenate((-self._npoints_ts[:0:-1], self._npoints_ts))
            )
        os_for_actions.integrate(
            self._tsJ,
            self._adiabatic_pot,
            method=self._integrate_method,
            dt=self._integrate_dt,
            progressbar=False,
        )
        return (
            _host(os_for_actions.R(self._tsJ[-1])),
            _host(os_for_actions.vR(self._tsJ[-1])),
            _host(os_for_actions.vT(self._tsJ[-1])),
            _host(os_for_actions.z(self._tsJ[-1])),
            _host(os_for_actions.vz(self._tsJ[-1])),
            _host(os_for_actions.phi(self._tsJ[-1])),
        )
