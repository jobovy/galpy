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

from ..backend import (
    coerce_coords,
    get_namespace,
    is_backend_array,
    like,
)
from ..backend._namespaces import (
    has_concrete_truth_value,
    inbackend_ode_method,
    name_of_namespace,
    namespace_from_arrays,
    param_grad_array,
    requires_backend_grad,
    stop_gradient,
    under_trace,
)
from ..backend.optimize import columnwise_jvp
from ..potential import (
    DehnenSmoothWrapperPotential,
    MWPotential,
    OblateStaeckelWrapperPotential,
    evaluateRforces,
    evaluatezforces,
)
from ..potential.Potential import _pot_grad_namespace
from ..potential.Potential import flatten as flatten_potential
from ..util import conversion, galpyWarning
from .actionAngle import actionAngle
from .actionAngleAdiabatic import actionAngleAdiabatic
from .actionAngleIsochrone import actionAngleIsochrone
from .actionAngleSpherical import actionAngleSpherical
from .actionAngleStaeckel import _focal_length_of, actionAngleStaeckel


def _switch_potential(start, end, tintJ):
    """The potential that changes from start to end over tintJ"""
    return DehnenSmoothWrapperPotential(
        pot=start, tform=0.0, tsteady=tintJ, decay=True
    ) + DehnenSmoothWrapperPotential(pot=end, tform=0.0, tsteady=tintJ, decay=False)


def _integrate_backend(self, ic, ts, pot, method=None):
    """The orbits [M,nt,6] of the backend-array initial conditions ic [M,6]
    in pot at the times ts, with the instance's integrator (or method):
    differentiable with respect to the initial conditions (for a C method, to
    first order) and, for an in-backend method, the potential's parameters"""
    from ..orbit import Orbit

    orb = Orbit(ic)
    orb.integrate(
        ts,
        pot,
        method=self._integrate_method if method is None else method,
        dt=self._integrate_dt,
        inbackend_kwargs=self._integrate_kwargs,
    )
    return orb.getOrbit()


def _orbit_velocity(xp, pot, xv):
    """The time derivative [M,6] of the phase-space points xv [M,6] along
    their orbits in pot (at t=0)"""
    from ..orbit.integrateFullOrbit import _EOM

    R, vR, vT, z, vz, phi = (xv[:, ii] for ii in range(6))
    d = _EOM([R, vR, phi, vT / R, z, vz], 0.0, pot)
    return xp.stack([d[0], d[1], d[0] * d[2] + R * d[3], d[4], d[5], d[2]], axis=-1)


def _in_namespace(method):
    """Run a method of the adiabatic-transformation classes in the namespace
    of its inputs: numpy inputs, with potentials that have no backend
    parameters and without a forced backend, take the numpy path; anything
    else (backend-array inputs, a forced backend, or backend parameters of
    the potentials) the class's implementation in that backend (_backend)"""

    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        vals = args
        if len(args) <= 2 and hasattr(self, "_coords"):
            # (an Orbit: its phase-space coordinates)
            vals = self._coords(*args)
        bk = tuple(a for a in vals if is_backend_array(a))
        xp = _pot_grad_namespace(self._adiabatic_pot, any_backend=True)
        if xp is None:
            xp = namespace_from_arrays(bk) if bk else get_namespace()
        if xp is numpy:
            return method(self, *args, **kwargs)
        return self._backend(method.__name__, xp, *args, **kwargs)

    return wrapped


def _unwrap(xp, x):
    """numpy.unwrap along the second axis"""
    dd = x[:, 1:] - x[:, :-1]
    ddmod = (dd + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
    ddmod = xp.where((ddmod == -numpy.pi) & (dd > 0.0), numpy.pi, ddmod)
    correct = xp.where(xp.abs(dd) < numpy.pi, 0.0, ddmod - dd)
    return xp.concatenate([x[:, :1], x[:, 1:] + xp.cumsum(correct, axis=1)], axis=1)


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
            Integration method to use. With 'diffrax' (jax) or 'torchode'/'torchdiffeq' (PyTorch), backend-array inputs are integrated in the backend, which is what derivatives with respect to parameters of the potentials and higher-order derivatives require (the C methods give first derivatives with respect to the phase-space point). On jax, differentiated frequencies for npoints=0 integrate with 'diffrax'.
        integrate_kwargs : dict, optional
            Options for the in-backend integrator (Orbit.integrate's inbackend_kwargs).
        dt : float, optional
            orbit.integrate dt keyword (for fixed stepsize integration).
        npoints : int, optional
            Number of points along the orbit, in addition to the point itself, that the actions are averaged over and that the frequencies and angles are fit to (an even number). With npoints=0, each point is transformed on its own: the transformation is then exactly canonical and actionAngleAdiabaticApproxInverse with the same tintJ is its inverse, but on numpy no frequencies are available (they are nan). For backend inputs (or under a forced backend), the frequencies are then the derivatives of the energy with respect to the actions at fixed angles, by automatic differentiation of the transformation along each point's orbit (when they are differentiated on jax, the orbits are integrated with 'diffrax'; on PyTorch, derivatives of these frequencies with respect to the phase-space point need integrate_method='torchode' or 'torchdiffeq').
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
        self._integrate_kwargs = kwargs.get("integrate_kwargs", None)
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
        start = numpy.array(self._aA.actionsFreqsAngles(R, vR, vT, z, vz, phi)[6:])
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
            self._aA.actionsFreqsAngles(
                orb.R(),
                orb.vR(),
                orb.vT(),
                orb.z(),
                orb.vz(),
                orb.phi(),
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
                for x in self._aAI(*actions, 1.0, 0.0, 1.0, use_physical=False)
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
            R * evaluateRforces(self._pot, R, z, phi=phi, use_physical=False)
            + z * evaluatezforces(self._pot, R, z, phi=phi, use_physical=False)
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
                        for q in self._aAI(*actions, *cand, use_physical=False)
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
                    self._aA.actionsFreqsAngles(R, vR, vT, z, vz, phi)[6:]
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

    @_in_namespace
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
        jr, lz, jz = self._aA(
            R.flatten(),
            vR.flatten(),
            vT.flatten(),
            z.flatten(),
            vz.flatten(),
            phi.flatten(),
        )

        return (
            numpy.mean(jr.reshape(Rshape), axis=1),
            numpy.mean(lz.reshape(Rshape), axis=1),
            numpy.mean(jz.reshape(Rshape), axis=1),
        )

    @_in_namespace
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

    @_in_namespace
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
        jr, lz, jz, Or, Op, Oz, ar, ap, az = self._aA.actionsFreqsAngles(
            R.flatten(),
            vR.flatten(),
            vT.flatten(),
            z.flatten(),
            vz.flatten(),
            phi.flatten(),
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

    def _backend(self, name, xp, *args, **kwargs):
        """The method name (_evaluate, _actionsFreqs, or
        _actionsFreqsAngles) in the backend xp, for traced or differentiated
        inputs: the same computation with the orbits integrated
        differentiably and the arithmetic in xp"""
        if len(args) == 4:
            args = (*args[:3], 0.0 * args[0], 0.0 * args[0], args[3])
        ic = xp.stack(
            [xp.atleast_1d(x) for x in coerce_coords(xp, *self._coords(*args))], axis=-1
        )
        if self._npoints == 0 and name != "_evaluate":
            actions, freqs, angles, end = self._rates_backend(xp, ic)
            if name == "_actionsFreqs":
                return (*actions, *freqs)
            return self._calibrated_backend(xp, actions, freqs, angles, end, ic)
        flip = like(ic, numpy.array([1.0, -1.0, -1.0, 1.0, -1.0, 1.0]))
        pts = ic[:, None, :]
        if self._npoints > 0:
            # (backward in time: forward with the velocities reversed)
            fwd = _integrate_backend(self, ic, self._npoints_ts, self._pot)
            bwd = _integrate_backend(self, ic * flip, self._npoints_ts, self._pot)
            pts = xp.concatenate([xp.flip(bwd[:, 1:], axis=1) * flip, fwd], axis=1)
        shape = pts.shape[:2]
        end = _integrate_backend(
            self, xp.reshape(pts, (-1, 6)), self._tsJ, self._adiabatic_pot
        )[:, -1]
        out = [
            xp.reshape(o, shape)
            for o in self._aA.actionsFreqsAngles(
                *(end[:, ii] for ii in range(6)), use_physical=False
            )
        ]
        actions = tuple(xp.mean(o, axis=1) for o in out[:3])
        if name == "_evaluate":
            return actions
        ts, pinv = like(ic, self._vanderA[:, 1], linalg.pinv(self._vanderA).T)
        freqs, angles = (), []
        for angle, freq in zip(out[6:], out[3:6]):
            guess = xp.mean(freq, axis=1)[:, None] * ts
            fit = (_unwrap(xp, angle - guess) + guess) @ pinv
            angles.append(fit[:, 0] % (2.0 * numpy.pi))
            freqs += (fit[:, 1],)
        if name == "_actionsFreqs":
            return (*actions, *freqs)
        end = xp.reshape(end, (*shape, 6))[:, self._npoints // 2]
        return self._calibrated_backend(xp, actions, freqs, angles, end, ic)

    def _calibrated_backend(self, xp, actions, freqs, angles, end, ic):
        """The actions, frequencies, and angles, the latter calibrated when
        calibrate_angles is set (end [M,6]: the images of the points ic)"""
        if self._calibrate:
            angles = xp.stack(angles) - self._offset_backend(xp, actions, end)
            angles = (
                angles + numpy.pi * self._halfturns_backend(xp, angles, actions, ic)
            ) % (2.0 * numpy.pi)
        return (*actions, *freqs, *angles)

    def _rates_backend(self, xp, ic):
        """For npoints=0: the actions, frequencies, angles, and images [M,6]
        of the points ic [M,6]. The transformation of a single point is
        canonical, so the frequencies dH/dJ at fixed angles are the rates of
        change of the angles along the orbits: the derivative of the
        transformation in the direction of each point's orbit"""
        from ..orbit.Orbits import _INBACKEND_SYMPLECTIC

        method = self._integrate_method
        derivative = (
            under_trace(ic)
            or requires_backend_grad(ic)
            or param_grad_array(self) is not None
        )
        if (
            derivative
            and name_of_namespace(xp) == "jax"
            and method.lower() != "diffrax"
            and method.lower() not in _INBACKEND_SYMPLECTIC
        ):
            # (frequencies that are differentiated on jax: forward mode, which
            # the C integrators do not have)
            method = inbackend_ode_method(xp)

        # the transformed points and their rates of change along the orbits
        end, rate = columnwise_jvp(
            lambda p: (
                _integrate_backend(
                    self, p.T, self._tsJ, self._adiabatic_pot, method=method
                )[:, -1].T
            ),
            ic.T,
            _orbit_velocity(xp, self._pot, ic).T,
            create_graph=derivative,
            reverse=None,
        )

        def auxiliary(p):
            out = [
                xp.broadcast_to(xp.reshape(o, (-1,)), p[0].shape)
                for o in self._aA.actionsFreqsAngles(*p, use_physical=False)
            ]
            return xp.stack(out[6:]), xp.stack(out[:3])

        # the angles of aA= there and their rates of change: the frequencies
        angles, freqs, actions = columnwise_jvp(
            auxiliary, end, rate, has_aux=True, create_graph=derivative, reverse=None
        )
        return (
            tuple(actions),
            tuple(freqs),
            [a % (2.0 * numpy.pi) for a in angles],
            end.T,
        )

    def _halfturns_backend(self, xp, angles, actions, xv):
        """_halfturns in the backend xp: whether each of the angles [3,M],
        known modulo pi, of the points xv [M,6] with the given actions is
        half a turn off, as 0 or 1 (without a derivative: the rule only
        decides between two candidates)"""
        sg = stop_gradient
        angles = sg(angles) % (2.0 * numpy.pi)
        actions = tuple(sg(a) for a in actions)
        R, vR, vT, z, vz, phi = (sg(xv[:, ii]) for ii in range(6))
        r = xp.sqrt(R**2.0 + z**2.0)
        vr = (R * vR + z * vz) / r
        v2 = vR**2.0 + vT**2.0 + vz**2.0
        racc = (v2 - vr**2.0) / r + (
            R * evaluateRforces(self._pot, R, z, phi=phi, use_physical=False)
            + z * evaluatezforces(self._pot, R, z, phi=phi, use_physical=False)
        ) / r
        off0 = xp.where(
            xp.where(
                xp.abs(xp.sin(angles[0])) > 0.3,
                vr * xp.sin(angles[0]),
                racc * xp.cos(angles[0]),
            )
            < 0.0,
            1.0,
            0.0,
        )
        off1 = off2 = 0.0 * off0
        todo = off0 == off0
        if self._aAI is not None:
            # (score of the four candidates: the cosines between the
            # positions and between the velocities)
            def cart(R, vR, vT, z, vz, phi):
                c, s = xp.cos(phi), xp.sin(phi)
                return xp.stack([R * c, R * s, z, vR * c - vT * s, vR * s + vT * c, vz])

            x = cart(R, vR, vT, z, vz, phi)
            scores = []
            for kk in range(4):
                cand = angles + numpy.pi * xp.stack(
                    [off0, 0.0 * off0 + kk // 2, 0.0 * off0 + kk % 2]
                )
                xa = cart(
                    *(
                        xp.broadcast_to(xp.reshape(sg(q), (-1,)), R.shape)
                        for q in self._aAI(*actions, *cand, use_physical=False)
                    )
                )
                scores.append(
                    sum(
                        xp.sum(x[ii : ii + 3] * xa[ii : ii + 3], axis=0)
                        / xp.sqrt(
                            xp.sum(x[ii : ii + 3] ** 2.0, axis=0)
                            * xp.sum(xa[ii : ii + 3] ** 2.0, axis=0)
                        )
                        for ii in (0, 3)
                    )
                )
            score = xp.stack(scores)
            best = xp.argmax(xp.where(xp.isfinite(score), score, -3.0), axis=0)
            todo = ~xp.all(xp.isfinite(score), axis=0)
            off1 = xp.where(best // 2 > 0, 1.0, 0.0)
            off2 = xp.where(best % 2 > 0, 1.0, 0.0)
        if not (
            self._aAI is not None
            and has_concrete_truth_value(todo)
            and not bool(xp.any(todo))
        ):
            # without an inverse of aA= (or where it has no such torus): the
            # half turns nearest the angles that aA= assigns to the point
            try:
                guess = sg(
                    xp.stack(self._aA.actionsFreqsAngles(R, vR, vT, z, vz, phi)[6:])
                )
                guess = xp.where(
                    xp.all(xp.abs(guess) < 100.0, axis=0), guess, numpy.nan
                )
            except Exception:
                guess = xp.full_like(angles, numpy.nan)
            nguess = todo & ~xp.all(xp.isfinite(guess), axis=0)
            if has_concrete_truth_value(nguess) and bool(xp.any(nguess)):
                warnings.warn(
                    f"{int(xp.sum(xp.astype(nguess, xp.int64)))} of the "
                    f"{R.shape[0]} points have no angles in the potential of aA= "
                    "(they are unbound there): their calibrated azimuthal and "
                    "vertical angles may be half a turn off",
                    galpyWarning,
                )
            near = (
                xp.round(
                    (guess[1:] - angles[1:] + numpy.pi) % (2.0 * numpy.pi) / numpy.pi
                    - 1.0
                )
                % 2.0
            )
            use = todo & ~nguess
            off1 = xp.where(use, near[0], off1)
            off2 = xp.where(use, near[1], off2)
        return xp.stack([off0, off1, off2])

    def _offset_backend(self, xp, actions, end):
        """_offset_torus in the backend xp: the offset [3,M] of the tori with
        these actions, as a differentiable function of them where aAI=
        returns the tori's fixed points, else of the points end [M,6]"""
        if self._aAI is not None:
            fixed = xp.stack(
                [
                    xp.broadcast_to(xp.reshape(x, (-1,)), actions[0].shape)
                    for x in self._aAI(*actions, 1.0, 0.0, 1.0, use_physical=False)
                ],
                axis=-1,
            )
            end = xp.where(xp.all(xp.isfinite(fixed), axis=-1)[:, None], fixed, end)
        angles = []
        for xv in (
            end,
            _integrate_backend(
                self,
                _integrate_backend(
                    self,
                    end,
                    self._tsJ,
                    _switch_potential(self._target_pot, self._pot, self._tintJ),
                )[:, -1],
                self._tsJ,
                self._adiabatic_pot,
            )[:, -1],
        ):
            angles.append(
                xp.stack(
                    self._aA.actionsFreqsAngles(
                        *(xv[:, ii] for ii in range(6)), use_physical=False
                    )[6:]
                )
            )
        return 0.5 * ((angles[1] - angles[0] + numpy.pi) % (2.0 * numpy.pi) - numpy.pi)

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

    def _coords(self, *args):
        """The phase-space coordinates (R,vR,vT,z,vz,phi) of the input"""
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
            if phi is None:
                raise ValueError("Must specify phi for actionAngleAdiabaticApprox")
        return R, vR, vT, z, vz, phi

    def _parse_args(self, *args):
        """Helper function to parse the arguments to the __call__ and actionsFreqsAngles functions"""
        from ..orbit import Orbit

        R, vR, vT, z, vz, phi = (
            numpy.atleast_1d(numpy.asarray(x, dtype="float"))
            for x in self._coords(*args)
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
            os_for_actions.R(self._tsJ[-1]),
            os_for_actions.vR(self._tsJ[-1]),
            os_for_actions.vT(self._tsJ[-1]),
            os_for_actions.z(self._tsJ[-1]),
            os_for_actions.vz(self._tsJ[-1]),
            os_for_actions.phi(self._tsJ[-1]),
        )
