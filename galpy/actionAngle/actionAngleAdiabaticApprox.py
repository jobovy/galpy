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
from ..potential import (
    DehnenSmoothWrapperPotential,
    MWPotential,
    OblateStaeckelWrapperPotential,
    evaluateRforces,
    evaluatezforces,
)
from ..potential.Potential import flatten as flatten_potential
from ..util import conversion, galpyWarning
from .actionAngle import actionAngle
from .actionAngleAdiabatic import actionAngleAdiabatic
from .actionAngleIsochrone import actionAngleIsochrone
from .actionAngleSpherical import actionAngleSpherical
from .actionAngleStaeckel import _focal_length_of, actionAngleStaeckel


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


def _aA_from_inverse(aAI):
    """The action-angle instance that an inverse action-angle instance
    inverts, where galpy can set it up: None otherwise"""
    from .actionAngleIsochroneInverse import actionAngleIsochroneInverse
    from .actionAngleSphericalInverse import actionAngleSphericalInverse
    from .actionAngleStaeckelInverse import actionAngleStaeckelInverse

    if isinstance(aAI, actionAngleIsochroneInverse):
        return actionAngleIsochrone(ip=aAI._ip)
    if isinstance(aAI, actionAngleSphericalInverse):
        return actionAngleSpherical(pot=aAI._pot)
    if isinstance(aAI, actionAngleStaeckelInverse):
        return actionAngleStaeckel(inverse=aAI, c=True)
    return None


def _auxiliary_potential(aA):
    """The potential that an action-angle or inverse action-angle instance
    is exact for, which is the potential to transform to: its pot, except
    for an actionAngleStaeckelInverse instance set up with delta= for a
    potential that is not of Staeckel form (and an actionAngleStaeckel
    instance set up from one), for which it is the Staeckel potential that
    the instance approximates that potential with. An instance whose
    Staeckel potential differs from torus to torus has no single such
    potential"""
    from .actionAngleStaeckelInverse import actionAngleStaeckelInverse

    aAI = aA if isinstance(aA, actionAngleStaeckelInverse) else None
    if isinstance(aA, actionAngleStaeckel):
        aAI = aA._inverse
    if aAI is None:
        return aA._pot
    if aAI._adaptive or aAI._delta_rule is not None:
        raise NotImplementedError(
            "An actionAngleStaeckelInverse instance whose focal length or "
            "reference curve follows its tori (a callable or 'fit' delta=, or "
            "a rule for u0=) has a different Staeckel potential for each "
            "torus and is not yet supported as the potential that is "
            "transformed to: use a numerical delta= and u0="
        )
    return aAI._staeckelwrap


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
            actionAngle instance for the potential that is being adiabatically transformed to. For an actionAngleStaeckel instance for a potential that is not of Staeckel form (the Staeckel approximation of that potential, which is not the action-angle transformation of any one potential), the potential that is transformed to is the Staeckel potential with the instance's focal length that approximates that potential (OblateStaeckelWrapperPotential with its default reference curve), and the actions and angles are computed in that potential; set aA= up for an OblateStaeckelWrapperPotential directly to choose the reference curve.
        aAAI : actionAngleAdiabaticApproxInverse instance, optional
            Set up the transformation that this instance inverts, in place of pot=, tintJ=, integrate_method=, and dt=, which are taken from it; aA= is then only necessary when the instance was set up without one and its aAI= is not an actionAngleIsochroneInverse, actionAngleSphericalInverse, or actionAngleStaeckelInverse instance. Its aAI= is used as this instance's aAI= unless one is given.
        tintJ : float, optional
            Time to integrate orbits for to estimate actions (can be Quantity).
        ntintJ : int, optional
            Number of output times of the orbit integration (only the last one is used).
        integrate_method : str, optional
            Integration method to use.
        dt : float, optional
            orbit.integrate dt keyword (for fixed stepsize integration).
        npoints : int, optional
            Number of points along the orbit, in addition to the point itself, that the actions are averaged over and that the frequencies and angles are fit to (an even number). With npoints=0, each point is transformed on its own: the transformation is then exactly canonical and actionAngleAdiabaticApproxInverse with the same tintJ is its inverse, but no frequencies are available (an implementation that can differentiate through the orbit integration can return them for npoints=0 as well, as the derivative of the energy with respect to the actions).
        npoints_dt : float, optional
            Amount of time to integrate forward and backwards for the npoints points (can be Quantity); the frequencies of the potential that is transformed to have to predict the change of each angle over this time to within pi.
        aAI : actionAngleInverse instance, optional
            The inverse of aA=, used by calibrate_angles=True for the offset and the half turns of the angles (set up automatically for an actionAngleIsochrone aA=; it has to return points for the actions of the points that are transformed; without it, or where it returns nan, the offset is measured at each point's own image and a less robust rule for the half turns is used that needs aA= to be close to pot).
        calibrate_angles : bool, optional
            If True, return the angles in the usual convention (zero radial angle at pericenter, zero vertical angle when moving upward through the plane) rather than with the offset of the first Note below: the offset of each point's torus is measured with a round trip of the transformation, which triples the orbit integration. With npoints=0 and aAI=, actionAngleAdiabaticApproxInverse with calibrate_angles=True remains the exact inverse. Default: False.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - The angles are angle coordinates on the torus that advance at the orbital frequencies, but they differ from the angles of other action-angle methods by an offset that is constant on each torus (it depends on the actions, on tintJ, and on the potentials). Many applications do not need the offset (for example, any use of this class together with actionAngleAdiabaticApproxInverse for the same tintJ).
        - With calibrate_angles=True, the offset is measured for each point's torus: transforming a point of the torus of the potential of aA= back to pot and again to that of aA=, both forward in time, changes its angles by twice the offset. For a finite tintJ the offset varies slightly over the torus (at the level of the error of the angles); it is measured at one fixed point of each torus (that of aAI= at the angles (1,0,1)), such that it is a function of the actions alone and the same for this class and its inverse (without aAI=, at the point's own image). This gives the offset up to half a turn, which is fixed by the angles that aA= itself assigns to the point (and, for the radial angle, by whether the point moves outward); the half turn can come out wrong for points whose image in the potential of aA= is unbound or whose torus there is very different.
        - 2025-09-23 - Started - Bovy (UofT).
        - 2025-10-15 - Allowed more general target potentials - Bovy (UofT).

        """
        actionAngle.__init__(self, ro=kwargs.get("ro", None), vo=kwargs.get("vo", None))
        if "aAAI" in kwargs:
            # The transformation that an actionAngleAdiabaticApproxInverse
            # instance inverts
            aAAI = kwargs["aAAI"]
            kwargs = dict(kwargs)
            kwargs.update(
                pot=aAAI._pot,
                tintJ=aAAI._tintJ,
                integrate_method=aAAI._integrate_method,
                dt=aAAI._integrate_dt,
            )
            kwargs.setdefault("aAI", aAAI._aAI)
            if "aA" not in kwargs:
                kwargs["aA"] = (
                    aAAI._aAA._aA
                    if aAAI._aAA is not None
                    else _aA_from_inverse(aAAI._aAI)
                )
            if kwargs["aA"] is None:
                raise ValueError(
                    "aA= is necessary with aAAI= when the "
                    "actionAngleAdiabaticApproxInverse instance was set up "
                    "without one and from an inverse action-angle instance "
                    "that galpy cannot set up the forward transformation of"
                )
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
        if (
            isinstance(self._aA, actionAngleStaeckel)
            and self._aA._inverse is None
            and _focal_length_of(self._aA._pot) is None
        ):
            # The Staeckel approximation of a potential that is not of
            # Staeckel form: the potential that is transformed to is the
            # Staeckel potential of its focal length, with the action-angle
            # instance that is exact for that potential
            self._aA = actionAngleStaeckel(
                pot=OblateStaeckelWrapperPotential(
                    pot=self._aA._pot, delta=self._aA._delta
                ),
                delta=self._aA._delta,
                c=self._aA._c,
                order=self._aA._order,
                **{
                    k: getattr(self._aA, f"_{k}")
                    for k in ("ro", "vo")
                    if getattr(self._aA, f"_{k}Set")
                },
            )
        self._target_pot = _auxiliary_potential(self._aA)
        self._tintJ = conversion.parse_time(
            kwargs.get("tintJ", 100.0), ro=self._ro, vo=self._vo
        )
        self._ntintJ = kwargs.get("ntintJ", 2)
        self._integrate_dt = kwargs.get("dt", None)
        self._tsJ = numpy.linspace(0.0, self._tintJ, self._ntintJ)
        self._integrate_method = kwargs.get("integrate_method", "dop853_c")
        self._npoints = int(kwargs.get("npoints", 2))
        if self._npoints < 0:
            raise OSError("npoints= must be at least 0")
        if self._npoints % 2:
            raise ValueError(
                "npoints= must be even: half of the points are taken on either "
                "side of each point along its orbit"
            )
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
        self._calibrate = kwargs.get("calibrate_angles", False)
        # The inverse of aA=, for the half turns of the calibrated angles
        self._aAI = kwargs.get("aAI", None)
        if self._aAI is None and isinstance(self._aA, actionAngleIsochrone):
            from .actionAngleIsochroneInverse import actionAngleIsochroneInverse

            self._aAI = actionAngleIsochroneInverse(ip=self._aA._ip)
        # Check the units
        self._check_consistent_units()
        return None

    def _offset(self, R, vR, vT, z, vz, phi):
        """The offset (jr,lz,jz order; modulo pi) of the angles of points
        of the potential that is transformed to: half of the change of
        their angles over the transformation back to pot and again to that
        potential, both forward in time"""
        from ..orbit import Orbit

        back_pot = DehnenSmoothWrapperPotential(
            pot=self._target_pot, tform=0.0, tsteady=self._tintJ, decay=True
        ) + DehnenSmoothWrapperPotential(
            pot=self._pot, tform=0.0, tsteady=self._tintJ, decay=False
        )
        start = numpy.array(
            _host(tuple(self._aA.actionsFreqsAngles(R, vR, vT, z, vz, phi)))[6:]
        )
        orb = Orbit(numpy.stack([R, vR, vT, z, vz, phi], axis=-1))
        for pot in (back_pot, self._adiabatic_pot):
            orb.integrate(
                self._tsJ,
                pot,
                method=self._integrate_method,
                dt=self._integrate_dt,
                progressbar=False,
            )
            orb = orb(self._tsJ[-1])
        end = numpy.array(
            _host(
                tuple(
                    self._aA.actionsFreqsAngles(
                        _host(orb.R()),
                        _host(orb.vR()),
                        _host(orb.vT()),
                        _host(orb.z()),
                        _host(orb.vz()),
                        _host(orb.phi()),
                    )
                )
            )[6:]
        )
        return 0.5 * ((end - start + numpy.pi) % (2.0 * numpy.pi) - numpy.pi)

    def _offset_torus(self, actions, end):
        """The offset (jr,lz,jz order; modulo pi) of the angles of the tori
        with these actions: that of one fixed point of each torus (that of
        aAI= at the angles (1,0,1)), such that it is the same function of
        the actions here and in the inverse transformation; without aAI=,
        or for tori that it does not return, that of the points end
        (R,vR,vT,z,vz,phi) of the potential that is transformed to"""
        end = [numpy.array(x, dtype="float") for x in end]
        if self._aAI is not None:
            fixed = [
                numpy.broadcast_to(numpy.ravel(x), end[0].shape)
                for x in _host(
                    tuple(self._aAI(*actions, 1.0, 0.0, 1.0, use_physical=False))
                )
            ]
            ok = numpy.all(numpy.isfinite(fixed), axis=0)
            end = [numpy.where(ok, f, x) for f, x in zip(fixed, end)]
        return self._offset(*end)

    def _halfturns(self, angles, actions, R, vR, vT, z, vz, phi):
        """Whether each of the angles (jr,lz,jz order), known modulo pi,
        of the points (R,vR,vT,z,vz,phi) of pot with the given actions is
        half a turn off. The radial angle is in (0,pi) when the point moves
        outward and, close to a turning point (where its sine is small and
        its error could decide), zero when the radial acceleration is
        outward. The azimuthal and vertical angles are those for which
        the point of the torus of aA= with the same actions and angles lies
        in the direction of the point and moves in its direction; without
        an inverse of aA= (or where it has no such torus), those nearest
        the angles that aA= assigns to the point itself"""
        angles = numpy.array(angles)
        r = numpy.sqrt(R**2.0 + z**2.0)
        vr = (R * vR + z * vz) / r
        v2 = vR**2.0 + vT**2.0 + vz**2.0
        racc = (v2 - vr**2.0) / r + (
            R * _host(evaluateRforces(self._pot, R, z, phi=phi, use_physical=False))
            + z * _host(evaluatezforces(self._pot, R, z, phi=phi, use_physical=False))
        ) / r
        off = numpy.zeros((3, len(R)), dtype="int")
        off[0] = (
            numpy.where(
                numpy.fabs(numpy.sin(angles[0])) > 0.3,
                vr * numpy.sin(angles[0]),
                racc * numpy.cos(angles[0]),
            )
            < 0.0
        )
        todo = numpy.ones(len(R), dtype="bool")
        if self._aAI is not None:
            # (score of the four candidates: the cosines between the
            # positions and between the velocities)
            def cart(R, vR, vT, z, vz, phi):
                c, s = numpy.cos(phi), numpy.sin(phi)
                return numpy.array(
                    [R * c, R * s, z, vR * c - vT * s, vR * s + vT * c, vz]
                )

            x = cart(R, vR, vT, z, vz, phi)
            score = numpy.full((4, len(R)), -numpy.inf)
            for kk in range(4):
                cand = angles + numpy.pi * numpy.array(
                    [off[0], 0 * off[0] + kk // 2, 0 * off[0] + kk % 2]
                )
                xa = cart(
                    *(
                        numpy.broadcast_to(numpy.ravel(q), R.shape)
                        for q in _host(
                            tuple(self._aAI(*actions, *cand, use_physical=False))
                        )
                    )
                )
                score[kk] = sum(
                    numpy.sum(x[ii : ii + 3] * xa[ii : ii + 3], axis=0)
                    / numpy.sqrt(
                        numpy.sum(x[ii : ii + 3] ** 2.0, axis=0)
                        * numpy.sum(xa[ii : ii + 3] ** 2.0, axis=0)
                    )
                    for ii in (0, 3)
                )
            best = numpy.argmax(numpy.where(numpy.isfinite(score), score, -3.0), axis=0)
            todo = ~numpy.all(numpy.isfinite(score), axis=0)
            off[1], off[2] = best // 2, best % 2
        if numpy.any(todo):
            try:
                guess = numpy.array(
                    _host(tuple(self._aA.actionsFreqsAngles(R, vR, vT, z, vz, phi)))[6:]
                )
                guess[:, ~numpy.all(numpy.fabs(guess) < 100.0, axis=0)] = numpy.nan
            except Exception:
                guess = numpy.full((3, len(R)), numpy.nan)
            nguess = todo & ~numpy.all(numpy.isfinite(guess), axis=0)
            if numpy.any(nguess):
                warnings.warn(
                    f"{numpy.sum(nguess)} of the {len(R)} points have no angles "
                    "in the potential of aA= (they are unbound there): their "
                    "calibrated azimuthal and vertical angles may be half a "
                    "turn off",
                    galpyWarning,
                )
            near = numpy.round(
                (guess[1:] - angles[1:] + numpy.pi) % (2.0 * numpy.pi) / numpy.pi - 1.0
            )
            use = todo & ~nguess
            off[1:, use] = near[:, use].astype("int") % 2
        return off

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
        jr, lz, jz, Or, Op, Oz, ar, ap, az = _host(
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
                *self._calibrated(
                    [a.reshape(Rshape)[:, 0] % (2.0 * numpy.pi) for a in (ar, ap, az)],
                    (jr, lz, jz),
                    [x[:, 0] for x in (R, vR, vT, z, vz, phi)],
                ),
            )
        # Fit for the frequency and angle for each dimension
        angle_out, freq_out = (), ()
        # (the angles advance at close to the frequencies of the potential
        # that is transformed to: unwrapped about that advance, such that
        # npoints_dt does not have to resolve half a period)
        ts = self._vanderA[:, 1]
        for angle, freq in zip((ar, ap, az), (Or, Op, Oz)):
            guess = numpy.mean(freq.reshape(Rshape), axis=1)[:, None] * ts
            angleT = numpy.unwrap(angle.reshape(Rshape) - guess, axis=1) + guess
            out = linalg.lstsq(self._vanderA, angleT.T)[0]
            angle_out += (out[0] % (2.0 * numpy.pi),)
            freq_out += (out[1],)
        mid = self._npoints // 2
        return (
            jr,
            lz,
            jz,
            *freq_out,
            *self._calibrated(
                angle_out, (jr, lz, jz), [x[:, mid] for x in (R, vR, vT, z, vz, phi)]
            ),
        )

    def _calibrated(self, angles, actions, end):
        """The angles with the offset of each point's torus removed when
        calibrate_angles is set: end is the image of each point in the
        potential that is transformed to"""
        if not self._calibrate:
            return tuple(angles)
        angles = (numpy.array(angles) - self._offset_torus(actions, end)) % (
            2.0 * numpy.pi
        )
        return tuple(
            (angles + numpy.pi * self._halfturns(angles, actions, *self._start))
            % (2.0 * numpy.pi)
        )

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
        self._start = (R, vR, vT, z, vz, phi)
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
                dt=self._integrate_dt,
                progressbar=False,
            )
            orb.integrate(
                -self._npoints_ts,
                self._pot,
                method=self._integrate_method,
                dt=self._integrate_dt,
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
