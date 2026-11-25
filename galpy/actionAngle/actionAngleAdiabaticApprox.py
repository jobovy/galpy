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
import warnings

import numpy
from numpy import linalg

from ..potential import (
    DehnenSmoothWrapperPotential,
    MWPotential,
    OblateStaeckelWrapperPotential,
    evaluatePotentials,
    evaluateRforces,
    evaluatezforces,
)
from ..potential.Potential import _check_c
from ..potential.Potential import flatten as flatten_potential
from ..util import conversion, galpyWarning
from .actionAngle import actionAngle
from .actionAngleAdiabatic import actionAngleAdiabatic
from .actionAngleIsochrone import actionAngleIsochrone
from .actionAngleSpherical import actionAngleSpherical
from .actionAngleStaeckel import _focal_length_of, actionAngleStaeckel


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
        return actionAngleStaeckel.from_inverse(aAI, c=True)
    return None


def _auxiliary_potential(aA):
    """The potential that an action-angle or inverse action-angle instance
    is exact for, which is the potential to transform to: its pot, except
    for an actionAngleStaeckelInverse instance set up with delta= for a
    potential that is not of Staeckel form (and an actionAngleStaeckel
    instance set up from one), for which it is the Staeckel potential that
    the instance approximates that potential with. An instance whose
    Staeckel potential differs from torus to torus has no single such
    potential: None"""
    from .actionAngleStaeckelInverse import actionAngleStaeckelInverse

    aAI = aA if isinstance(aA, actionAngleStaeckelInverse) else None
    if isinstance(aA, actionAngleStaeckel):
        aAI = aA._inverse
    if aAI is None:
        return aA._pot
    if aAI._adaptive or aAI._delta_rule is not None:
        return None
    return aAI._staeckelwrap


def _switch(start, end, tintJ):
    """The potential that changes from start to end over tintJ"""
    return DehnenSmoothWrapperPotential(
        pot=start, tform=0.0, tsteady=tintJ, decay=True
    ) + DehnenSmoothWrapperPotential(pot=end, tform=0.0, tsteady=tintJ, decay=False)


def _staeckel_models(aA, R, vR, vT, z, vz):
    """The Staeckel potential, as rows (delta, u0), of the torus through
    each point of the family that the actionAngleStaeckel instance aA was
    set up from (nan for a point that is on none of its tori)"""
    # (for a fixed focal length the C code keeps the reference curve to
    # itself)
    delta, u0 = aA._inverse_model(
        R, vR, vT, z, vz, c=aA._inverse._delta_rule is not None, deriv=False
    )[:2]
    return numpy.stack(numpy.broadcast_arrays(delta, u0), axis=-1) * 1.0


def _deform(pot, aux, vxvv, models, ts, method, dt, back=False, wrapper=None):
    """The points vxvv [M,6] integrated over the times ts while pot
    changes to the potential that is transformed to (with back, while that
    potential changes to pot): aux or, without a single such potential,
    the Staeckel potential of each point's row (delta, u0) of models (the
    first of the pair that wrapper(delta, u0) returns), in which case the
    points are integrated each in its own potential (in C, all at once;
    otherwise one Staeckel potential at a time)"""
    from ..orbit import Orbit
    from ..orbit.integrateFullOrbit import _ext_loaded as ext_loaded
    from ..orbit.integrateFullOrbit import _parse_indiv_pot, integrateFullOrbit_c

    out = numpy.full_like(vxvv, numpy.nan)

    def run(sel, aux):
        orb = Orbit(vxvv[sel])
        orb.integrate(
            ts,
            _switch(aux, pot, ts[-1]) if back else _switch(pot, aux, ts[-1]),
            method=method,
            dt=dt,
            progressbar=False,
        )
        out[sel] = numpy.stack(
            [
                getattr(orb, q)(ts[-1], use_physical=False)
                for q in ("R", "vR", "vT", "z", "vz", "phi")
            ],
            axis=-1,
        )

    if aux is not None:
        run(numpy.arange(len(vxvv)), aux)
        return out
    ok = numpy.flatnonzero(numpy.all(numpy.isfinite(models), axis=1))
    uniq, inv = numpy.unique(models[ok], axis=0, return_inverse=True)
    if len(ok) > 0 and ext_loaded and method.endswith("_c") and _check_c(pot):
        # all points at once in C, each in its own potential: one potential
        # is passed in full and the focal length, reference curve, and
        # reference value of the potential of each point's Staeckel potential
        # in its place among that potential's arguments
        delta, u0 = models[ok].T

        def switch(aux):
            return _switch(aux, pot, ts[-1]) if back else _switch(pot, aux, ts[-1])

        def reference(delta, u0):
            return (
                evaluatePotentials(
                    pot, delta * numpy.sinh(u0), 0.0 * delta, use_physical=False
                )
                * numpy.cosh(u0) ** 2.0
            )

        template = switch(
            OblateStaeckelWrapperPotential(pot=pot, delta=delta[0], u0=u0[0])
        )
        other = switch(
            OblateStaeckelWrapperPotential(
                pot=pot, delta=1.1 * delta[0], u0=1.1 * u0[0]
            )
        )
        idx = _parse_indiv_pot([template, other], t=ts)[4]
        out[ok] = integrateFullOrbit_c(
            template,
            vxvv[ok],
            ts,
            method,
            dt=dt,
            progressbar=False,
            indiv_args=(idx, numpy.stack([delta, u0, reference(delta, u0)], axis=1)),
        )[0][:, -1]
        return out
    auxs = [wrapper(delta, u0)[0] for delta, u0 in uniq]
    for ii, aux in enumerate(auxs):
        run(ok[numpy.ravel(inv) == ii], aux)
    return out


class actionAngleAdiabaticApprox(actionAngle):
    """Action-angle formalism using an adiabatic transformation to an explicitly-integrable potential"""

    def __init__(self, *args, **kwargs):
        """
        Initialize an actionAngleAdiabaticApprox object.

        Parameters
        ----------
        pot : Potential or a combined potential formed using addition (pot1+pot2+…)
            Potential to calculate action-angle variables for.
        aA : actionAngleIsochrone, actionAngleSpherical, actionAngleAdiabatic, or actionAngleStaeckel instance
            Action-angle instance of the auxiliary potential that pot is adiabatically transformed to (see the Notes for an actionAngleStaeckel instance).
        tintJ : float or Quantity, optional
            Duration of the adiabatic transformation. Default: 100.
        npoints : int, optional
            Number of points along each point's orbit, in addition to the point itself, that the actions are averaged over and that the frequencies and angles are fit to (an even number). With npoints=0, each point is transformed on its own and no frequencies are available. Default: 2.
        npoints_dt : float or Quantity, optional
            Time on either side of each point that the npoints points cover; the frequencies in the auxiliary potential have to predict the change of each angle over this time to within pi. Default: 0.1.
        calibrate_angles : bool, optional
            If True, return the angles in the usual convention (zero radial angle at pericenter, zero vertical angle when moving upward through the plane) rather than with an offset that is constant on each torus (see the Notes); this triples the orbit integration. Default: False.
        aAI : actionAngleInverse instance, optional
            The inverse of aA=, which calibrate_angles=True uses to measure the offset at one fixed point of each torus (set up automatically for an actionAngleIsochrone instance and for actionAngleStaeckel.from_inverse); without it, the offset is measured at each point's own image, which requires the auxiliary potential to be close to pot.
        niter : int, optional
            For an aA= that has a Staeckel potential for each torus (see the Notes): the number of times that each point's Staeckel potential is updated, each at the cost of another orbit integration. Default: 2.
        integrate_method : str, optional
            Orbit-integration method. Default: 'dop853_c'.
        dt : float, optional
            Orbit.integrate's dt keyword (for fixed-stepsize integration).
        ntintJ : int, optional
            Number of output times of the orbit integration (only the last one is used).
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - Each point is integrated for tintJ while pot is slowly changed into the auxiliary potential, and its actions, frequencies, and angles are those of aA= at the end. actionAngleAdiabaticApproxInverse is the inverse; use actionAngleAdiabaticApprox.from_inverse to set up the transformation that an instance of it inverts. With npoints=0 the two are exact inverses of each other, to the precision of the orbit integration.
        - The angles advance at the orbital frequencies, but they differ from those of other action-angle methods by an offset that is constant on each torus. Many applications do not need the offset (any use together with actionAngleAdiabaticApproxInverse, for example); calibrate_angles=True removes it, by measuring it for each point's torus with a round trip of the transformation.
        - An actionAngleStaeckel instance as aA= sets the auxiliary potential in one of three ways. Set up for an OblateStaeckelWrapperPotential (or another potential of Staeckel form), it is that potential. Set up with delta= for another potential, it is the Staeckel potential OblateStaeckelWrapperPotential(pot=,delta=) that approximates that potential. Set up with actionAngleStaeckel.from_inverse from a family of tori of pot whose Staeckel potential follows its tori (actionAngleStaeckelInverse with setup_interp=True and delta='fit' or u0='torus'), each point is transformed to a Staeckel potential chosen to match its own torus, such that a single family serves all orbits of a galaxy: the potential of the family's torus with the point's actions, found by starting from the family's torus through the point and updating niter times with the actions that the transformation returns. actionAngleAdiabaticApproxInverse is then the inverse to the convergence of these updates (about two digits for each, such that the default two give a round trip of 1e-7 or better also for eccentric halo orbits; the precision of the actions does not depend on them). The C orbit integrators integrate all points in parallel, each in its own potential; other integrators integrate them one at a time.
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
        self._adiabatic_pot = (
            None
            if self._target_pot is None
            else _switch(self._pot, self._target_pot, self._tintJ)
        )
        self._niter = int(kwargs.get("niter", 2))
        # (the Staeckel potentials of tori and their action-angle instances,
        # and tori of them: kept for one evaluation)
        self._wrappers, self._tori, self._end_models = {}, {}, None
        self._calibrate = kwargs.get("calibrate_angles", False)
        # The inverse of aA=, for the half turns of the calibrated angles
        self._aAI = kwargs.get("aAI", None)
        if self._target_pot is None:
            self._aAI = self._aA._inverse
        if self._aAI is None and isinstance(self._aA, actionAngleStaeckel):
            self._aAI = self._aA._inverse
        if self._aAI is None and isinstance(self._aA, actionAngleIsochrone):
            from .actionAngleIsochroneInverse import actionAngleIsochroneInverse

            self._aAI = actionAngleIsochroneInverse(ip=self._aA._ip)
        # Check the units
        self._check_consistent_units()
        return None

    @classmethod
    def from_inverse(cls, aAAI, aA=None, **kwargs):
        """
        Set up the transformation that an actionAngleAdiabaticApproxInverse instance inverts.

        Parameters
        ----------
        aAAI : actionAngleAdiabaticApproxInverse instance
            The inverse transformation; the potential, the duration of the transformation, and the orbit-integration settings are taken from it, and its aAI= is the aAI= of the new instance.
        aA : actionAngle instance, optional
            Action-angle instance of the auxiliary potential; only necessary when aAAI was set up without one and its aAI= is not an actionAngleIsochroneInverse, actionAngleSphericalInverse, or actionAngleStaeckelInverse instance.
        **kwargs
            Other keywords of actionAngleAdiabaticApprox (npoints=, npoints_dt=, calibrate_angles=, niter=, ntintJ=, ro=, vo=).

        Returns
        -------
        actionAngleAdiabaticApprox instance
            With npoints=0, the exact inverse of aAAI (also for calibrate_angles=True in both).

        Notes
        -----
        - 2026-11-24 - Written as a class method (was the aAAI= keyword) - Bovy (UofT)

        """
        if aA is None:
            aA = aAAI._aAA._aA if aAAI._aAA is not None else _aA_from_inverse(aAAI._aAI)
        if aA is None:
            raise ValueError(
                "aA= is necessary when the actionAngleAdiabaticApproxInverse "
                "instance was set up without one and from an inverse "
                "action-angle instance that galpy cannot set up the forward "
                "transformation of"
            )
        kwargs.setdefault("aAI", aAAI._aAI)
        return cls(
            pot=aAAI._pot,
            aA=aA,
            tintJ=aAAI._tintJ,
            integrate_method=aAAI._integrate_method,
            dt=aAAI._integrate_dt,
            **kwargs,
        )

    def _offset(self, R, vR, vT, z, vz, phi, models=None):
        """The offset (jr,lz,jz order; modulo pi) of the angles of points
        of the potential that is transformed to: half of the change of
        their angles over the transformation back to pot and again to that
        potential, both forward in time"""
        if models is None:
            models = self._models(R, vR, vT, z, vz)
        start = self._aux(R, vR, vT, z, vz, phi, models)[6:]
        xv = numpy.stack([R, vR, vT, z, vz, phi], axis=-1) * 1.0
        for back in (True, False):
            xv = self._deform(xv, models, back=back)
        end = self._aux(*xv.T, models)[6:]
        return 0.5 * ((end - start + numpy.pi) % (2.0 * numpy.pi) - numpy.pi)

    def _wrapper(self, delta, u0):
        """The Staeckel potential (delta, u0) of pot and the action-angle
        instance that is exact for it"""
        key = (float(delta), float(u0))
        if key not in self._wrappers:
            aux = OblateStaeckelWrapperPotential(pot=self._pot, delta=key[0], u0=key[1])
            self._wrappers[key] = (
                aux,
                actionAngleStaeckel(pot=aux, delta=key[0], c=True),
            )
        return self._wrappers[key]

    def _aux(self, R, vR, vT, z, vz, phi, models=None):
        """The actions, frequencies, and angles [9,M] of points of the
        potential that is transformed to: those of aA= or, without a single
        such potential, those in the Staeckel potential of each point's row
        (delta, u0) of models (the family's own angles are not angle
        coordinates of any one of these potentials)"""
        if models is None:
            return numpy.array(self._aA.actionsFreqsAngles(R, vR, vT, z, vz, phi))
        out = numpy.full((9, len(R)), numpy.nan)
        ok = numpy.flatnonzero(numpy.all(numpy.isfinite(models), axis=1))
        uniq, inv = numpy.unique(models[ok], axis=0, return_inverse=True)
        for ii, (delta, u0) in enumerate(uniq):
            sel = ok[numpy.ravel(inv) == ii]
            out[:, sel] = self._wrapper(delta, u0)[1].actionsFreqsAngles(
                *(x[sel] for x in (R, vR, vT, z, vz, phi))
            )
        return out

    def _torus(self, jr, lz, jz):
        """Without a single potential that is transformed to: the Staeckel
        potential (delta, u0) of the torus of the family of aA= with these
        actions and that torus as an explicit torus of that potential
        ((nan, nan), None for actions that the family has no torus for)"""
        key = (float(jr), float(lz), float(jz))
        if key not in self._tori:
            from .actionAngleStaeckelInverse import actionAngleStaeckelInverse

            model = self._models_of_tori(
                *(numpy.array([k]) for k in key), numpy.full((1, 2), numpy.nan)
            )[0]
            self._tori[key] = (
                model,
                actionAngleStaeckelInverse(
                    pot=self._wrapper(*model)[0],
                    jrs=[key[0]],
                    Lzs=[key[1]],
                    jzs=[key[2]],
                    mm_npt=256,
                    progressbar=False,
                )
                if numpy.all(numpy.isfinite(model))
                else None,
            )
        return self._tori[key]

    def _aux_points(self, actions, angles):
        """The points [6,M] of the tori of the potential that is transformed
        to with these actions and angles, from aAI= (for a family: the
        points of each torus in its own Staeckel potential), and for a
        family the rows (delta, u0) of the Staeckel potentials (else None)"""
        shape = numpy.shape(actions[0])
        if self._target_pot is not None:
            return (
                numpy.array(
                    [
                        numpy.broadcast_to(numpy.ravel(x), shape)
                        for x in self._aAI(*actions, *angles, use_physical=False)
                    ]
                ),
                None,
            )
        angles = numpy.broadcast_arrays(*angles, actions[0])[:3]
        out, models = (
            numpy.full((6, *shape), numpy.nan),
            numpy.full((*shape, 2), numpy.nan),
        )
        tori, inv = numpy.unique(
            numpy.stack(actions, axis=-1), axis=0, return_inverse=True
        )
        for ii, key in enumerate(tori):
            sel = numpy.ravel(inv) == ii
            models[sel], torus = self._torus(*key)
            if torus is not None:
                out[:, sel] = numpy.array(
                    torus(*key, *(a[sel] for a in angles), use_physical=False)
                )
        return out, models

    def _models_of_tori(self, jr, lz, jz, fallback):
        """Without a single potential that is transformed to: the Staeckel
        potential (rows (delta, u0)) of the tori of the family of aA= with
        these actions, which is that of the family's torus through its own
        point with these actions at the angles (1,0,1): a function of the
        actions alone, the same here and in the inverse transformation. The
        rows of fallback for actions that the family has no torus for"""
        out = numpy.array(fallback) * 1.0
        ok = numpy.atleast_1d(self._aAI.in_domain(jr, lz, jz))
        if numpy.any(ok):
            models = _staeckel_models(
                self._aA,
                *(
                    numpy.broadcast_to(numpy.ravel(x), (numpy.sum(ok),))
                    for x in self._aAI(
                        jr[ok], lz[ok], jz[ok], 1.0, 0.0, 1.0, use_physical=False
                    )[:5]
                ),
            )
            out[ok] = numpy.where(
                numpy.all(numpy.isfinite(models), axis=1)[:, None], models, out[ok]
            )
        return out

    def _models(self, R, vR, vT, z, vz):
        """The Staeckel potential (rows (delta, u0)) of each point's torus
        of the family of aA=, where there is no single potential that is
        transformed to: None otherwise"""
        if self._target_pot is not None:
            return None
        return _staeckel_models(self._aA, R, vR, vT, z, vz)

    def _deform(self, xv, models, back=False):
        """The points xv [M,6] at the end of the transformation to the
        potential that is transformed to (back: from it to pot)"""
        return _deform(
            self._pot,
            self._target_pot,
            xv,
            models,
            self._tsJ,
            self._integrate_method,
            self._integrate_dt,
            back=back,
            wrapper=self._wrapper,
        )

    def _offset_torus(self, actions, end):
        """The offset (jr,lz,jz order; modulo pi) of the angles of the tori
        with these actions: that of one fixed point of each torus (that of
        aAI= at the angles (1,0,1)), such that it is the same function of
        the actions here and in the inverse transformation; without aAI=,
        or for tori that it does not return, that of the points end
        (R,vR,vT,z,vz,phi) of the potential that is transformed to"""
        end = [numpy.array(x, dtype="float") for x in end]
        models = None if self._target_pot is not None else self._models(*end[:5])
        if self._aAI is not None:
            fixed, torus = self._aux_points(actions, (1.0, 0.0, 1.0))
            ok = numpy.all(numpy.isfinite(fixed), axis=0)
            end = [numpy.where(ok, f, x) for f, x in zip(fixed, end)]
            if models is not None:
                models = numpy.where(ok[:, None], torus, models)
        return self._offset(*end, models=models)

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
                xa = cart(*self._aux_points(actions, cand)[0])
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
        jr, lz, jz = self._aux(
            R.flatten(),
            vR.flatten(),
            vT.flatten(),
            z.flatten(),
            vz.flatten(),
            phi.flatten(),
            self._end_models,
        )[:3]
        return (
            numpy.mean(jr.reshape(Rshape), axis=1),
            numpy.mean(lz.reshape(Rshape), axis=1),
            numpy.mean(jz.reshape(Rshape), axis=1),
        )

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
        Rshape = R.shape
        jr, lz, jz, Or, Op, Oz, ar, ap, az = self._aux(
            R.flatten(),
            vR.flatten(),
            vT.flatten(),
            z.flatten(),
            vz.flatten(),
            phi.flatten(),
            self._end_models,
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
            if phi is None:
                raise ValueError("Must specify phi for actionAngleAdiabaticApprox")
        R, vR, vT, z, vz, phi = (
            numpy.atleast_1d(numpy.asarray(x, dtype="float"))
            for x in (R, vR, vT, z, vz, phi)
        )
        self._start = (R, vR, vT, z, vz, phi)
        self._wrappers.clear()
        self._tori.clear()
        if self._npoints == 0:
            # Each point on its own
            xv = numpy.stack([R, vR, vT, z, vz, phi], axis=-1)[:, None, :]
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
            ts = numpy.concatenate((-self._npoints_ts[:0:-1], self._npoints_ts))
            xv = numpy.stack(
                [
                    numpy.reshape(getattr(orb, q)(ts, use_physical=False), (len(R), -1))
                    for q in ("R", "vR", "vT", "z", "vz", "phi")
                ],
                axis=-1,
            )
        shape = xv.shape
        # (without a single potential to transform to: each point and the
        # points along its orbit to the Staeckel potential of the point's
        # torus, updated to that of the torus that the point ends on)
        models = self._models(R, vR, vT, z, vz)
        mid = self._npoints // 2
        for ii in range(1 + self._niter * (models is not None)):
            end = self._deform(
                xv.reshape((-1, 6)),
                None if models is None else numpy.repeat(models, shape[1], axis=0),
            ).reshape(shape)
            if models is not None and ii < self._niter:
                models = self._models_of_tori(
                    *self._aux(*end[:, mid].T, models)[:3], models
                )
        self._end_models = (
            None if models is None else numpy.repeat(models, shape[1], axis=0)
        )
        return tuple(end[..., ii] for ii in range(6))
