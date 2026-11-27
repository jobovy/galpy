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

from ..backend import (
    as_numpy,
    coerce_coords,
    like,
    use,
)
from ..backend._input import _grad_param, numpy_or_backend
from ..backend._namespaces import (
    has_concrete_truth_value,
    inbackend_ode_method,
    jax_transformations,
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
from .actionAngleStaeckel import (
    _focal_length_of,
    _staeckel_actions_freqs_angles,
    actionAngleStaeckel,
)


def _inbackend(self, ic, tangent=False):
    """Whether the orbits of the backend-array points ic [M,6] are integrated
    with the backend's own solver (diffrax, torchdiffeq) in place of the
    instance's C integrator: where the C integrator cannot carry the
    derivative (the potentials' parameters) or lacks the potential, and on jax
    where that solver is faster, on a CPU or a GPU: under jax.jit or jax.vmap,
    and for more than 1000 points (500 differentiated; for the orbits'
    derivatives in a direction, the tangents, 100). The tangents' derivatives
    with respect to the points are faster in C, also under jax.jit."""
    from ..backend._reference.inbackend_stm import _C_DXDV_METHODS

    if self._integrate_method.lower() not in _C_DXDV_METHODS:
        return False
    if param_grad_array(self._bk_params) is not None or not _check_c(
        self._bk_params, dxdv3d=True
    ):
        return True
    if name_of_namespace(namespace_from_arrays((ic,))) != "jax":
        return False
    transformations = jax_transformations(ic)
    derivative = bool(transformations & {"forward", "reverse"})
    if tangent and derivative:
        return False
    if transformations & {"jit", "vmap"}:
        return True
    return ic.shape[0] > (100 if tangent else 500 if derivative else 1000)


def _integrate_backend(self, ic, ts, pot, method=None):
    """The orbits [M,nt,6] of the backend-array initial conditions ic [M,6]
    in pot at the times ts, with the instance's integrator (or method; see
    _inbackend): differentiable with respect to the initial conditions (for a
    C method, to first order) and, for an in-backend method, the potential's
    parameters"""
    from ..orbit import Orbit

    if method is None:
        method = (
            inbackend_ode_method(namespace_from_arrays((ic,)))
            if _inbackend(self, ic)
            else self._integrate_method
        )
    orb = Orbit(ic)
    orb.integrate(
        ts,
        pot,
        method=method,
        progressbar=False,
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


def _point_coords(self, *args):
    """The phase-space coordinates among the inputs of a method of the
    adiabatic-transformation classes (an Orbit's, for an Orbit)"""
    return self._coords(*args) if len(args) <= 2 and hasattr(self, "_coords") else args


# numpy inputs, with potentials that have no backend parameters and without a
# forced backend, take the numpy path; anything else (backend-array inputs, a
# forced backend, or backend parameters of the potentials) the class's
# implementation in that backend (_backend)
_in_namespace = numpy_or_backend("_bk_params", coords=_point_coords)


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
    xp = namespace_from_arrays((R, delta, u0)) or numpy
    if xp is numpy:
        return numpy.stack(numpy.broadcast_arrays(delta, u0), axis=-1) * 1.0
    return xp.stack([delta + 0.0 * R, u0 + 0.0 * R], axis=-1)


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
            Number of points along each point's orbit, in addition to the point itself, that the actions are averaged over and that the frequencies and angles are fit to (an even number). With npoints=0, each point is transformed on its own and no frequencies are available on numpy (they are nan); for jax/torch inputs (or under a forced backend), they are then the derivatives of the energy with respect to the actions at fixed angles, by automatic differentiation of the transformation along each point's orbit (derivatives of these frequencies with respect to the phase-space point need integrate_method='diffrax' on jax and 'torchode' or 'torchdiffeq' on PyTorch, and an aA= that these backends can differentiate twice, such as one with c=False). Default: 2.
        npoints_dt : float or Quantity, optional
            Time on either side of each point that the npoints points cover; the frequencies in the auxiliary potential have to predict the change of each angle over this time to within pi. Default: 0.1.
        calibrate_angles : bool, optional
            If True, return the angles in the usual convention (zero radial angle at pericenter, zero vertical angle when moving upward through the plane) rather than with an offset that is constant on each torus (see the Notes); this triples the orbit integration. Default: False.
        aAI : actionAngleInverse instance, optional
            The inverse of aA=, which calibrate_angles=True uses to measure the offset at one fixed point of each torus (set up automatically for an actionAngleIsochrone instance and for actionAngleStaeckel.from_inverse); without it, the offset is measured at each point's own image, which requires the auxiliary potential to be close to pot.
        niter : int, optional
            For an aA= that has a Staeckel potential for each torus (see the Notes): the number of times that each point's Staeckel potential is updated, each at the cost of another orbit integration. Default: 2.
        integrate_method : str, optional
            Orbit-integration method. With 'diffrax' (jax) or 'torchode'/'torchdiffeq' (PyTorch), jax/torch inputs are integrated in the backend, which is what derivatives with respect to parameters of the potentials and higher-order derivatives require (the C methods give first derivatives with respect to the phase-space point). Default: 'dop853_c'.
        integrate_kwargs : dict, optional
            Options for the in-backend integrator (Orbit.integrate's inbackend_kwargs).
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

    @property
    def _bk_params(self):
        """The potential whose jax/torch parameters make a transformation a
        backend one: the switch, or pot for a Staeckel family as the
        auxiliary"""
        return self._pot if self._adiabatic_pot is None else self._adiabatic_pot

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
            integrate_kwargs=aAAI._integrate_kwargs,
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
            actions, freqs, angles, end, models = self._rates_backend(xp, ic)
            if name == "_actionsFreqs":
                return (*actions, *freqs)
            return self._calibrated_backend(xp, actions, freqs, angles, end, ic, models)
        flip = like(ic, numpy.array([1.0, -1.0, -1.0, 1.0, -1.0, 1.0]))
        pts = ic[:, None, :]
        if self._npoints > 0:
            # (backward in time: forward with the velocities reversed)
            fwd = _integrate_backend(self, ic, self._npoints_ts, self._pot)
            bwd = _integrate_backend(self, ic * flip, self._npoints_ts, self._pot)
            pts = xp.concatenate([xp.flip(bwd[:, 1:], axis=1) * flip, fwd], axis=1)
        shape = pts.shape[:2]
        flat = xp.reshape(pts, (-1, 6))
        # (without a single potential to transform to: each point and the
        # points along its orbit to the Staeckel potential of the point's
        # torus, updated niter times to that of the torus it ends on, as on
        # numpy; the models are held, a gauge)
        models = None if self._target_pot is not None else self._bk_models(xp, ic)
        for ii in range(1 + self._niter * (models is not None)):
            rows = None if models is None else xp.repeat(models, shape[1], axis=0)
            if models is None:
                end = _integrate_backend(self, flat, self._tsJ, self._adiabatic_pot)[
                    :, -1
                ]
            else:
                end = self._bk_family_end(xp, flat, rows)
            if models is not None and ii < self._niter:
                mid = xp.reshape(end, (*shape, 6))[:, self._npoints // 2]
                models = self._bk_models_of_tori(
                    xp, self._bk_aux(xp, mid, models)[:3], models
                )
        rows = None if models is None else xp.repeat(models, shape[1], axis=0)
        out = [xp.reshape(o, shape) for o in self._bk_aux(xp, end, rows)]
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
        return self._calibrated_backend(xp, actions, freqs, angles, end, ic, models)

    # ---------- a Staeckel family as the auxiliary, on jax/torch: each point
    # in the Staeckel potential of its own torus (held: a gauge)
    def _bk_host(self):
        """Whether the family of aA= (and aAI=) and the potentials hold no
        jax/torch arrays: the held models, and the points of concrete tori
        without a derivative, are then found on the host, as numpy"""
        fams = (getattr(self._aA, "_inverse", None), self._aAI)
        return all(getattr(f, "_bk", None) is None for f in fams) and all(
            _grad_param(p) is None for p in (self._pot, getattr(self._aA, "_pot", None))
        )

    def _bk_models(self, xp, xv):
        """The Staeckel potential (rows (delta, u0), held) of each point's
        torus of the family of aA= (_models; on the host for concrete
        points)"""
        xs = stop_gradient(xv)
        if self._bk_host() and has_concrete_truth_value(xp.all(xs == xs)):
            h = numpy.asarray(as_numpy(xs), dtype="float")
            with use("numpy", force=True):
                return like(xv, _staeckel_models(self._aA, *h.T[:5]))
        return stop_gradient(_staeckel_models(self._aA, *(xs[:, k] for k in range(5))))

    def _bk_models_of_tori(self, xp, actions, fallback):
        """_models_of_tori on jax/torch: the Staeckel potential (rows,
        held) of the family's tori with these actions, from the family's
        point of each at the angles (1,0,1); the rows of fallback for tori
        that the family does not have (outside its domain, decided on the
        host for concrete actions, where for a family set up with numpy the
        models are found too; under a trace, those whose point or model is
        not finite)"""
        jr, lz, jz = (stop_gradient(t) for t in actions)
        fallback = stop_gradient(fallback)
        concrete = has_concrete_truth_value(
            xp.all(jr == jr) & xp.all(lz == lz) & xp.all(fallback == fallback)
        )
        if concrete and self._bk_host():
            h = [numpy.asarray(as_numpy(t), dtype="float") for t in (jr, lz, jz)]
            fb = numpy.asarray(as_numpy(fallback), dtype="float")
            with use("numpy", force=True):
                return like(jr, self._models_of_tori(*h, fb))
        ok = None
        if concrete:
            ok = like(jr, numpy.atleast_1d(self._aAI.in_domain(jr, lz, jz)))
            if not bool(xp.any(ok)):
                return fallback
            # (stand-in actions, of a torus that the family has, elsewhere)
            first = int(numpy.argmax(numpy.asarray(as_numpy(ok))))
            jr, lz, jz = (xp.where(ok, t, t[first]) for t in (jr, lz, jz))
        pts = self._aAI(jr, lz, jz, 1.0, 0.0, 1.0, use_physical=False)[:5]
        models = _staeckel_models(
            self._aA, *(xp.broadcast_to(xp.reshape(x, (-1,)), jr.shape) for x in pts)
        )
        good = xp.all(xp.isfinite(models), axis=1)
        if ok is not None:
            good = good & ok
        return stop_gradient(xp.where(good[:, None], models, fallback))

    def _bk_stand_ins(self, xp, xv, models):
        """For a family as the auxiliary: the points xv [M,6] and their
        Staeckel potentials (rows of models) with a stand-in for each point
        whose potential is not defined (on no torus of the family), which
        the backends' integrators cannot integrate; and which are not
        stand-ins"""
        ok = xp.all(xp.isfinite(models), axis=1) & xp.all(xp.isfinite(xv), axis=1)
        pt, md = like(
            xv, numpy.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0]), numpy.array([0.5, 1.0])
        )
        return xp.where(ok[:, None], xv, pt), xp.where(ok[:, None], models, md), ok

    def _bk_family_c(self, xp, xv, models, tangent=False):
        """Whether the points xv of a family as the auxiliary (each in the
        Staeckel potential of its row of models) integrate in C, each in its
        own potential: for concrete models, as _inbackend decides for the
        instance's C integrator"""
        from ..backend._reference.inbackend_stm import _C_DXDV_METHODS

        return (
            self._integrate_method.lower() in _C_DXDV_METHODS
            and has_concrete_truth_value(xp.all(models == models))
            and not _inbackend(self, xv, tangent=tangent)
        )

    def _bk_family_pots(self, models, back=False):
        """The switch potential of each point's Staeckel potential (rows of
        concrete models), one potential for each point, on the host"""
        rows = numpy.asarray(as_numpy(stop_gradient(models)), dtype="float")
        with use("numpy", force=True):
            return [
                _switch(self._wrapper(*r)[0], self._pot, self._tintJ)
                if back
                else _switch(self._pot, self._wrapper(*r)[0], self._tintJ)
                for r in rows
            ]

    def _bk_family_end(self, xp, xv, models, back=False):
        """The points xv [M,6] at the end of the transformation, each in the
        Staeckel potential of its row of models (back: from it to pot):
        concrete points without a derivative as on numpy (_deform: in C all
        at once), with one in C through the state-transition matrix (one
        potential for each point), else in the backend's integrator; nan for
        the points on no torus of the family"""
        from ..backend._reference.inbackend_stm import integrate_stm

        xv_s, models_s, ok = self._bk_stand_ins(xp, xv, models)
        if self._bk_family_c(xp, xv_s, models_s):
            if self._bk_host() and not (under_trace(xv) or requires_backend_grad(xv)):
                with use("numpy", force=True):
                    end = _deform(
                        self._pot,
                        None,
                        numpy.asarray(as_numpy(xv), dtype="float"),
                        numpy.asarray(as_numpy(stop_gradient(models)), dtype="float"),
                        self._tsJ,
                        self._integrate_method,
                        self._integrate_dt,
                        back=back,
                        wrapper=self._wrapper,
                    )
                return like(xv, end)
            end = integrate_stm(
                self._bk_family_pots(models_s, back=back),
                xv_s,
                self._tsJ,
                method=self._integrate_method,
                rtol=1e-12,
                atol=1e-12,
            )[:, -1]
        else:
            end = _integrate_backend(
                self,
                xv_s,
                self._tsJ,
                self._bk_switch(models_s, back=back),
                method=inbackend_ode_method(xp),
            )[:, -1]
        return xp.where(ok[:, None], end, numpy.nan)

    def _bk_switch(self, models, back=False):
        """The potential that changes from pot to the potential that is
        transformed to (back: the reverse): the auxiliary or, for rows
        (delta, u0) of models, each point's Staeckel potential (one wrapper
        with a focal length and reference curve per point)"""
        if models is None:
            if not back:
                return self._adiabatic_pot
            return _switch(self._target_pot, self._pot, self._tintJ)
        aux = OblateStaeckelWrapperPotential(
            pot=self._pot, delta=models[:, 0], u0=models[:, 1]
        )
        return (
            _switch(aux, self._pot, self._tintJ)
            if back
            else _switch(self._pot, aux, self._tintJ)
        )

    def _bk_aux(self, xp, xv, models):
        """The actions, frequencies, and angles of the points xv [M,6] in the
        potential that is transformed to: aA='s, or with rows (delta, u0) of
        models, each in its own Staeckel potential (_aux)"""
        cols = [xv[:, k] for k in range(6)]
        if models is None:
            return tuple(self._aA.actionsFreqsAngles(*cols, use_physical=False))
        return tuple(
            _staeckel_actions_freqs_angles(
                xp, *cols, self._pot, models[:, 0], 10, u0=models[:, 1]
            )
        )

    def _bk_aux_points(self, xp, actions, angles):
        """The points [M,6] of the tori of the potential that is transformed
        to with these actions and angles, and for a family the rows (delta,
        u0) of their Staeckel potentials (else None): aAI='s, or each
        torus explicitly in its own Staeckel potential (_aux_points)"""
        shape = actions[0].shape
        if self._target_pot is not None:
            return (
                xp.stack(
                    [
                        xp.broadcast_to(xp.reshape(x, (-1,)), shape)
                        for x in self._aAI(*actions, *angles, use_physical=False)
                    ],
                    axis=-1,
                ),
                None,
            )
        from .actionAngleStaeckelInverse import actionAngleStaeckelInverse

        if self._bk_host() and not (
            under_trace(*actions, *angles) or requires_backend_grad(*actions, *angles)
        ):
            # (concrete tori without a derivative: as numpy)
            h = [numpy.asarray(as_numpy(t), dtype="float") for t in (*actions, *angles)]
            with use("numpy", force=True):
                out, models = self._aux_points(h[:3], h[3:])
            return like(actions[0], out.T), like(actions[0], models)
        nan = numpy.nan + 0.0 * actions[0]
        models = self._bk_models_of_tori(xp, actions, xp.stack([nan, nan], axis=-1))
        ok = xp.all(xp.isfinite(models), axis=1)
        # (the tori that the family does not have on stand-ins, and nan)
        okc = ok if has_concrete_truth_value(xp.any(ok)) else None
        if okc is not None and not bool(xp.any(okc)):
            return xp.stack([nan] * 6, axis=-1), models
        k = int(numpy.argmax(numpy.asarray(as_numpy(ok)))) if okc is not None else 0
        acts = [xp.where(ok, t, t[k]) for t in actions]
        mods = xp.where(ok[:, None], models, models[k])
        with use(name_of_namespace(xp), force=True):
            tori = actionAngleStaeckelInverse(
                pot=self._pot,
                jrs=acts[0],
                Lzs=acts[1],
                jzs=acts[2],
                mm_npt=256,
                progressbar=False,
                _models=(mods[:, 0], mods[:, 1]),
            )
            angles = xp.broadcast_arrays(*(a + 0.0 * acts[0] for a in angles))
            xv = xp.stack(
                [
                    xp.reshape(x, (-1,))
                    for x in tori(*acts, *angles, use_physical=False)
                ],
                axis=-1,
            )
        return xp.where(ok[:, None], xv, numpy.nan), models

    def _calibrated_backend(self, xp, actions, freqs, angles, end, ic, models=None):
        """The actions, frequencies, and angles, the latter calibrated when
        calibrate_angles is set (end [M,6]: the images of the points ic;
        models: their Staeckel potentials for a family as the auxiliary)"""
        if self._calibrate:
            angles = xp.stack(angles) - self._offset_backend(xp, actions, end, models)
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
        from ..backend._reference.inbackend_stm import _C_DXDV_METHODS

        derivative = (
            under_trace(ic)
            or requires_backend_grad(ic)
            or param_grad_array(self) is not None
        )
        velocity = _orbit_velocity(xp, self._pot, ic)
        # (a family as the auxiliary: each point's Staeckel potential, held,
        # integrated in the backend)
        models, ok = None, None
        if self._target_pot is None:
            models = self._bk_models(xp, ic)
            for _ in range(self._niter):
                end = self._bk_family_end(xp, ic, models)
                models = self._bk_models_of_tori(
                    xp, self._bk_aux(xp, end, models)[:3], models
                )
            ic, models, ok = self._bk_stand_ins(xp, ic, models)
            velocity = xp.where(ok[:, None], velocity, 0.0)
        if models is not None:
            # (each point in its own potential: in C, one for each point, or
            # in the backend's integrator)
            c = self._bk_family_c(xp, ic, models, tangent=True)
            pot = self._bk_family_pots(models) if c else self._bk_switch(models)
            method = self._integrate_method.lower() if c else inbackend_ode_method(xp)
        else:
            pot = self._bk_switch(models)
            method = (
                inbackend_ode_method(xp)
                if _inbackend(self, ic, tangent=True)
                else self._integrate_method.lower()
            )
        if method == "diffrax":
            # (jax: the orbits with their variational equations in diffrax,
            # which reverse mode goes through with checkpoints)
            from ..backend._reference import integrate_orbit

            kwargs = dict(self._integrate_kwargs or {})
            end, rate = (
                o[-1].T
                for o in integrate_orbit(
                    pot,
                    ic,
                    xp.asarray(self._tsJ),
                    rtol=kwargs.pop("rtol", 1e-12),
                    atol=kwargs.pop("atol", 1e-12),
                    tangent=velocity,
                    **kwargs,
                )
            )
        elif method in _C_DXDV_METHODS:
            # (C: the orbits with their variational equations from the
            # points' velocities, one solve each)
            from ..backend._reference.inbackend_stm import integrate_stm

            end, rate = (
                o[:, -1].T
                for o in integrate_stm(
                    pot,
                    ic,
                    self._tsJ,
                    method=method,
                    rtol=1e-12,
                    atol=1e-12,
                    tangent=velocity,
                )
            )
        else:
            # the transformed points and their rates of change along the orbits
            end, rate = columnwise_jvp(
                lambda p: (
                    _integrate_backend(self, p.T, self._tsJ, pot, method=method)[
                        :, -1
                    ].T
                ),
                ic.T,
                velocity.T,
                create_graph=derivative,
                reverse=None,
            )

        def auxiliary(p):
            out = [
                xp.broadcast_to(xp.reshape(o, (-1,)), p[0].shape)
                for o in self._bk_aux(xp, xp.stack(list(p), axis=-1), models)
            ]
            return xp.stack(out[6:]), xp.stack(out[:3])

        # the angles of aA= there and their rates of change: the frequencies
        angles, freqs, actions = columnwise_jvp(
            auxiliary, end, rate, has_aux=True, create_graph=derivative, reverse=None
        )
        if ok is not None:
            # (the points on no torus of a family: nan)
            angles, freqs, actions = (
                xp.where(ok[None, :], t, numpy.nan) for t in (angles, freqs, actions)
            )
            end = xp.where(ok[None, :], end, numpy.nan)
            models = xp.where(ok[:, None], models, numpy.nan)
        return (
            tuple(actions),
            tuple(freqs),
            [a % (2.0 * numpy.pi) for a in angles],
            end.T,
            models,
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
                pts = sg(self._bk_aux_points(xp, actions, tuple(cand))[0])
                xa = cart(*(pts[:, ii] for ii in range(6)))
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
            and has_concrete_truth_value(xp.any(todo))
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
            if has_concrete_truth_value(xp.any(nguess)) and bool(xp.any(nguess)):
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

    def _offset_backend(self, xp, actions, end, models=None):
        """_offset_torus in the backend xp: the offset [3,M] of the tori with
        these actions, as a differentiable function of them where aAI=
        returns the tori's fixed points, else of the points end [M,6]
        (models: their Staeckel potentials for a family as the auxiliary)"""
        if self._aAI is not None:
            fixed, torus = self._bk_aux_points(xp, actions, (1.0, 0.0, 1.0))
            ok = xp.all(xp.isfinite(fixed), axis=-1)
            end = xp.where(ok[:, None], fixed, end)
            if models is not None:
                models = xp.where(ok[:, None], torus, models)
        if models is None:
            trip = _integrate_backend(
                self,
                _integrate_backend(
                    self, end, self._tsJ, self._bk_switch(None, back=True)
                )[:, -1],
                self._tsJ,
                self._adiabatic_pot,
            )[:, -1]
        else:
            trip = self._bk_family_end(
                xp, self._bk_family_end(xp, end, models, back=True), models
            )
        angles = [xp.stack(self._bk_aux(xp, xv, models)[6:]) for xv in (end, trip)]
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
