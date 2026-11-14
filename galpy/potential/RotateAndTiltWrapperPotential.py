###############################################################################
#   RotateAndTiltWrapperPotential.py: Wrapper to rotate and tilt the z-axis
#   of a potential
###############################################################################
import copy

import numpy
from scipy import interpolate

from ..backend import (
    as_backend_constant,
    as_numpy,
    coerce_coords,
    concretely_true,
    get_namespace,
    is_backend_array,
    promote_scalars,
    to_host,
)
from ..backend._coerce import where
from ..backend._namespaces import (
    eager_value_memo,
    like,
    namespace_from_arrays,
    requires_backend_grad,
    under_trace,
)
from ..backend.interpolate import cubic_spline_coeffs, eval_ppoly
from ..util import _rotate_to_arbitrary_vector, _zvec_pa_rotation, conversion, coords
from .Potential import (
    _evaluatephitorques,
    _evaluatePotentials,
    _evaluateRforces,
    _evaluatezforces,
    check_potential_inputs_not_arrays,
    evaluateDensities,
    evaluatephi2derivs,
    evaluatephizderivs,
    evaluateR2derivs,
    evaluateRphiderivs,
    evaluateRzderivs,
    evaluatez2derivs,
)
from .WrapperPotential import WrapperPotential


# Only implement 3D wrapper
class RotateAndTiltWrapperPotential(WrapperPotential):
    """Potential wrapper that allows a potential to be rotated in 3D
    according to three orientation angles. These angles can either be
    specified using:

    * A rotation around the original z-axis (`galaxy_pa`) and the new direction of the z-axis (`zvec`) or

    * A rotation around the original z-axis (`galaxy_pa`), the `inclination`, and a rotation around the new z axis (`sky_pa`).

    The second option allows one to specify the inclination and sky position angle (measured from North) in the usual manner in extragalactic observations, that is, the relation between the observed coordinates :math:`(x',y',z')` and the intrinsic coordinates :math:`(x,y,z)` is given by

    .. math::

        \\begin{pmatrix} x \\\\ y \\\\ z \\end{pmatrix} = \\begin{pmatrix} \\phantom{-}\\cos(\\text{PA}_\\text{gal}) & \\sin(\\text{PA}_\\text{gal}) & 0 \\\\ -\\sin(\\text{PA}_\\text{gal}) & \\cos(\\text{PA}_\\text{gal}) & 0 \\\\ 0 & 0 & 1 \\end{pmatrix} \\begin{pmatrix} 1 & 0 & 0 \\\\ 0 & \\phantom{-}\\cos i & \\sin i \\\\ 0 & -\\sin i & \\cos i \\end{pmatrix} \\begin{pmatrix} \\sin(\\text{PA}_\\text{sky}) & -\\cos(\\text{PA}_\\text{sky}) & 0 \\\\ \\cos(\\text{PA}_\\text{sky}) & \\phantom{-}\\sin(\\text{PA}_\\text{sky}) & 0 \\\\ 0 & 0 & 1 \\end{pmatrix} \\begin{pmatrix} x' \\\\ y' \\\\ z' \\end{pmatrix}\\,,


    A final `offset` option allows one to apply a static offset in Cartesian coordinate space to be applied to the potential following the rotation and tilt.

    The potential can also move and turn in time: with a `center` (a constant position, a tabulated path, or an integrated `Orbit`) and a time-dependent `orientation` (tabulated as unit quaternions or rotation matrices), the potential is

    .. math::

        \\Phi(\\mathbf{x},t) = \\Phi_\\mathrm{pot}\\left(\\mathsf{R}(t)\\,[\\mathbf{x}-\\mathbf{c}(t)] + \\mathbf{o},t\\right)\\,,

    where :math:`\\mathsf{R}(t)` is the rotation (the constant one set by the angles above, or the tabulated one), :math:`\\mathbf{c}(t)` the center, and :math:`\\mathbf{o}` the `offset`. The rows of :math:`\\mathsf{R}(t)` are the wrapped potential's axes in the (Galactocentric) frame of :math:`\\mathbf{x}`.
    """

    # Evaluation broadcasts over a trailing node axis: the coordinate stack and
    # the Hessian stack both build at a common rank, and the rotation is a
    # matmul that carries batch axes. The scalars-only numpy contract still
    # holds -- the guard additionally requires every argument to be a backend
    # array.
    _backend_accepts_arrays = True

    def __init__(
        self,
        amp=1.0,
        inclination=None,
        galaxy_pa=None,
        sky_pa=None,
        zvec=None,
        offset=None,
        pot=None,
        ro=None,
        vo=None,
        center=None,
        orientation=None,
        cinterp_n=3000,
    ):
        """
        A potential that rotates and tilts another potential.

        Parameters
        ----------
        amp : float, optional
            Overall amplitude to apply to the potential. Default is 1.0.
        inclination : float, Quantity, or callable, optional
            Usual inclination angle (with the line-of-sight being the z axis). Like ``galaxy_pa``, ``sky_pa``, and ``zvec``, it can be a function of time (in internal units, returning the angle in radians), which makes the orientation time-dependent.
        galaxy_pa : float, Quantity, or callable, optional
            Rotation angle of the original potential around the original z axis; for example, ``galaxy_pa=lambda t: Omega * t`` rotates the potential around the z axis with pattern speed ``Omega``.
        sky_pa : float, Quantity, or callable, optional
            Rotation angle around the inclined z axis (usual sky position angle measured from North).
        zvec : numpy.ndarray or callable, optional
            3D vector specifying the direction of the rotated z axis (or a function of time returning it).
        offset : numpy.ndarray or Quantity, optional
            Static offset in Cartesian coordinates, applied in the frame of the wrapped potential (after the rotation).
        pot : Potential instance or a combined potential formed using addition (pot1+pot2+…)
            The Potential instance or a combined potential formed using addition (pot1+pot2+…) to rotate and tilt.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).
        center : numpy.ndarray, Quantity, tuple, Orbit, or callable, optional
            Position of the wrapped potential's origin in the (Galactocentric) frame of the evaluation point, before the rotation: a constant (x,y,z), a tabulated path ``(t, xyz)`` with ``t`` an increasing array of N >= 3 times and ``xyz`` an (N,3) array of positions (interpolated with a natural cubic spline, and held at its end values outside of the tabulated times), an integrated ``Orbit`` (its positions; interpolated by the Orbit itself in Python and by a natural cubic spline of its integration output in C, and held at its end positions outside of the integration times), or a function of time (in internal units) returning (x,y,z) (in internal units). Any object with an Orbit's ``t`` attribute and ``x``, ``y``, ``z``, ``dim``, and ``turn_physical_off`` methods (and, for the JAX and PyTorch backends, ``R``, ``phi``, and ``z`` that take ``_stored=True``) can stand in for the ``Orbit``.
        orientation : tuple or callable, optional
            Time-dependent rotation R(t), as a table ``(t, q)`` with ``t`` an increasing array of N >= 3 times and ``q`` either an (N,4) array of unit quaternions (w,x,y,z) or an (N,3,3) array of rotation matrices (the quaternions are interpolated with natural cubic splines and normalized after interpolation, and held at their end values outside of the tabulated times), or as a function of time (in internal units) returning a (3,3) rotation matrix or a (4,) quaternion; the rows of R(t) are the wrapped potential's axes in the frame of the evaluation point. Cannot be combined with inclination, galaxy_pa, sky_pa, or zvec.
        cinterp_n : int, optional
            For C orbit integration, a center, orientation, or angles given as functions of time are tabulated at this many times over the integration's time range and interpolated with natural cubic splines (held at their end values outside of that range), as for ``NonInertialFrameForce``'s ``cinterp``; in Python and with the JAX and PyTorch backends, the functions are evaluated at every time (with the backends, they are traced and differentiated if they are written in the array namespace, e.g., ``jax.numpy``). Default is 3000.

        Notes
        -----
        - 2021-03-29 - Started - Mackereth (UofT)
        - 2021-04-18 - Added inclination, sky_pa, galaxy_pa setup - Bovy (UofT)
        - 2022-03-14 - added offset kwarg - Mackereth (UofT)
        - 2026-11-14 - Added time-dependent center and orientation - Bovy (UofT)
        """
        WrapperPotential.__init__(self, amp=amp, pot=pot, ro=ro, vo=vo, _init=True)
        if orientation is not None and not all(
            angle is None for angle in (inclination, galaxy_pa, sky_pa, zvec)
        ):
            raise ValueError(
                "RotateAndTiltWrapperPotential: specify the orientation either as a time-dependent orientation= or through the angles inclination, galaxy_pa, sky_pa, and zvec, not both"
            )
        self._cinterp_n = cinterp_n
        self._cinterp_cache = None
        # angles that are functions of time: the orientation at time t is the
        # constant orientation of the angles at t
        self._angle_funcs = None
        if any(callable(angle) for angle in (inclination, galaxy_pa, sky_pa, zvec)):
            self._angle_funcs = tuple(
                angle if callable(angle) else conversion.parse_angle(angle)
                for angle in (inclination, galaxy_pa, sky_pa)
            ) + (zvec,)
            inclination, galaxy_pa, sky_pa, zvec = None, None, None, None
        inclination = conversion.parse_angle(inclination)
        sky_pa = conversion.parse_angle(sky_pa)
        galaxy_pa = conversion.parse_angle(galaxy_pa)
        zvec, galaxy_pa = self._parse_inclination(inclination, sky_pa, zvec, galaxy_pa)
        if isinstance(offset, list) and any(is_backend_array(o) for o in offset):
            ref = next(o for o in offset if is_backend_array(o))
            offset = namespace_from_arrays((ref,)).stack(
                promote_scalars(namespace_from_arrays((ref,)), *offset)
            )
        self._offset = conversion.parse_length(
            numpy.array(offset) if isinstance(offset, list) else offset, ro=self._ro
        )
        self._setup_zvec_pa(zvec, galaxy_pa)
        # a traced or grad-tracking rotation is kept even when it is the identity
        self._norot = not (
            under_trace(self._rot) or requires_backend_grad(self._rot)
        ) and concretely_true((self._rot == like(self._rot, numpy.eye(3))).all())
        self._setup_center(center)
        self._setup_orientation(orientation)
        self.hasC = True
        self._backend_compatible = True
        self.hasC_dxdv = True
        # Advertise the 3D variational capability unconditionally, as for
        # hasC/hasC_dxdv: _check_c recurses into the wrapped potential's own
        # hasC_dxdv3d (the wrapper's C 3D Hessian conjugates the wrapped
        # potential's calc<deriv> Hessian with the rotation matrix, so it is
        # complete iff the wrapped one is).
        self.hasC_dxdv3d = True
        self.isNonAxi = True

    def _parse_inclination(self, inclination, sky_pa, zvec, galaxy_pa):
        if inclination is None:
            return (zvec, galaxy_pa)
        if sky_pa is None:
            sky_pa = 0.0
        refs = [x for x in (inclination, sky_pa, galaxy_pa) if is_backend_array(x)]
        if refs:
            return self._parse_inclination_backend(
                refs[0], inclination, sky_pa, galaxy_pa
            )
        zvec_rot = numpy.dot(
            numpy.array(
                [
                    [numpy.sin(sky_pa), numpy.cos(sky_pa), 0.0],
                    [-numpy.cos(sky_pa), numpy.sin(sky_pa), 0.0],
                    [0.0, 0.0, 1],
                ]
            ),
            numpy.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, -numpy.cos(inclination), -numpy.sin(inclination)],
                    [0.0, -numpy.sin(inclination), numpy.cos(inclination)],
                ]
            ),
        )
        zvec = numpy.dot(zvec_rot, numpy.array([0.0, 0.0, 1.0]))
        int_rot = _rotate_to_arbitrary_vector(
            numpy.array([[0.0, 0.0, 1.0]]), zvec, inv=False
        )[0]
        pa = numpy.dot(int_rot, numpy.dot(zvec_rot, [1.0, 0.0, 0.0]))
        return (zvec, galaxy_pa + numpy.arctan2(pa[1], pa[0]))

    @staticmethod
    def _parse_inclination_backend(ref, inclination, sky_pa, galaxy_pa):
        """_parse_inclination with a backend-array angle (differentiable)"""
        xp = namespace_from_arrays((ref,))
        inc, spa = (
            v if is_backend_array(v) else as_backend_constant(xp, v, ref)
            for v in (inclination, sky_pa)
        )
        z0, one = xp.zeros_like(inc), xp.ones_like(inc)
        ss, cs, si, ci = xp.sin(spa), xp.cos(spa), xp.sin(inc), xp.cos(inc)
        zvec_rot = xp.stack(
            [xp.stack([ss, cs, z0]), xp.stack([-cs, ss, z0]), xp.stack([z0, z0, one])]
        ) @ xp.stack(
            [xp.stack([one, z0, z0]), xp.stack([z0, -ci, -si]), xp.stack([z0, -si, ci])]
        )
        zvec = zvec_rot[:, 2]
        int_rot = _rotate_to_arbitrary_vector(
            numpy.array([[0.0, 0.0, 1.0]]), zvec, inv=False
        )[0]
        pa = int_rot @ zvec_rot[:, 0]
        return (zvec, galaxy_pa + xp.atan2(pa[1], pa[0]))

    def _setup_zvec_pa(self, zvec, pa):
        self._rot = _zvec_pa_rotation(zvec, pa)
        if is_backend_array(self._rot):
            self._inv_rot = namespace_from_arrays((self._rot,)).linalg.inv(self._rot)
        else:
            self._inv_rot = numpy.linalg.inv(self._rot)
        return None

    def _setup_center(self, center):
        """Set up the (time-dependent) center: None, constant, table, Orbit,
        or function"""
        self._center_type = None
        if center is None:
            return None
        if callable(center) and not _orbit_like(center):
            self._center_type = "function"
            self._center_func = center
        elif _orbit_like(center):
            if getattr(center, "size", 1) != 1:
                raise ValueError(
                    "RotateAndTiltWrapperPotential: center= Orbit must hold a single object"
                )
            if not hasattr(center, "t"):
                raise ValueError(
                    "RotateAndTiltWrapperPotential: center= Orbit must be integrated"
                )
            self._center_type = "orbit"
            # backend arrays are shared, not copied (a non-leaf torch tensor
            # cannot be deep-copied; none is mutated here)
            memo = {id(v): v for v in vars(center).values() if is_backend_array(v)}
            self._center_orb = copy.deepcopy(center, memo)
            self._center_orb.turn_physical_off()
            # every evaluation queries the orbit: build its backend spline once
            if callable(getattr(self._center_orb, "_cache_backend_interp", None)):
                self._center_orb._cache_backend_interp()
            # its integration times' range, outside of which it is held
            tc = self._center_orb.t
            tc = tc if is_backend_array(tc) else numpy.asarray(tc, dtype=float)
            self._center_trange = (tc.min(), tc.max())
        elif isinstance(center, tuple):
            if len(center) != 2:
                raise ValueError(
                    "RotateAndTiltWrapperPotential: a tabulated center= is a tuple (t, xyz)"
                )
            tc = _parse_table_times(
                center[0], len(center[1]), "center", ro=self._ro, vo=self._vo
            )
            xyzc = conversion.parse_length(center[1], ro=self._ro)
            if not is_backend_array(xyzc):
                xyzc = numpy.array(xyzc, dtype=float)
            if tuple(xyzc.shape) != (len(tc), 3):
                raise ValueError(
                    "RotateAndTiltWrapperPotential: a tabulated center= (t, xyz) needs xyz of shape (len(t), 3)"
                )
            self._center_type = "table"
            self._center_t, self._center_xyz = tc, xyzc
            self._center_spline, self._center_c = _table_spline(tc, xyzc)
        else:
            if isinstance(center, list) and any(is_backend_array(c) for c in center):
                ref = next(c for c in center if is_backend_array(c))
                center = namespace_from_arrays((ref,)).stack(
                    promote_scalars(namespace_from_arrays((ref,)), *center)
                )
            center = conversion.parse_length(
                numpy.array(center) if isinstance(center, list) else center,
                ro=self._ro,
            )
            if not is_backend_array(center):
                center = numpy.array(center, dtype=float)
            if tuple(center.shape) != (3,):
                raise ValueError(
                    "RotateAndTiltWrapperPotential: a constant center= is a position (x, y, z)"
                )
            self._center_type = "constant"
            self._center_xyz = center
        return None

    def _c_tables(self, t):
        """The center's and the orientation's tables passed to C, (t, xyz)
        and (t, quaternions) (None when there is none): a table's, an orbit's
        integration output (increasing in time), or, for a center or an
        orientation that is a function of time, its tabulation at cinterp_n
        times over the range of the integration times t"""
        tabulate = self._center_type == "function" or self._orient_type in (
            "function",
            "angles",
        )
        if tabulate:
            if t is None:
                raise NotImplementedError(
                    "RotateAndTiltWrapperPotential: a center or orientation that is a function of time is only supported in C for orbit integration"
                )
            tn = numpy.asarray(as_numpy(t), dtype=float)
            key = (float(numpy.amin(tn)), float(numpy.amax(tn)), self._cinterp_n)
            if self._cinterp_cache is not None and self._cinterp_cache[0] == key:
                return self._cinterp_cache[1]
            tgrid = numpy.linspace(*key)
        center, orient = None, None
        if self._center_type == "table":
            center = (self._center_t, self._center_xyz)
        elif self._center_type == "orbit":
            orb = self._center_orb
            tc = numpy.array(as_numpy(orb.t), dtype=float)
            xyzc = numpy.array(
                [
                    as_numpy(orb.x(tc)),
                    as_numpy(orb.y(tc)),
                    as_numpy(orb.z(tc)) if orb.dim() == 3 else numpy.zeros_like(tc),
                ]
            ).T
            if tc[-1] < tc[0]:
                tc, xyzc = tc[::-1], xyzc[::-1]
            center = (tc, xyzc)
        elif self._center_type == "function":
            center = (
                tgrid,
                numpy.array(
                    [as_numpy(self._center_at(numpy, tt, None)) for tt in tgrid]
                ),
            )
        if self._orient_type == "table":
            orient = (self._orient_t, self._orient_q)
        elif self._orientation:
            rots = numpy.array(
                [as_numpy(self._rot_at(numpy, tt, None)) for tt in tgrid]
            )
            orient = (tgrid, _continuous_unit_quaternions(_matrix_to_quaternion(rots)))
        if tabulate:
            self._cinterp_cache = (key, (center, orient))
        return center, orient

    def _setup_orientation(self, orientation):
        """Set up the time-dependent orientation: angles that are functions of
        time, a table of quaternions or rotation matrices, or a function"""
        self._orient_type = None
        if self._angle_funcs is not None:
            self._orient_type = "angles"
        elif callable(orientation):
            self._orient_type = "function"
            self._orient_func = orientation
        elif orientation is not None:
            self._orient_type = "table"
            self._setup_orientation_table(orientation)
        self._orientation = self._orient_type is not None
        if self._orientation:
            self._norot = False
        return None

    def _setup_orientation_table(self, orientation):
        """Set up the orientation from a table of quaternions or rotation
        matrices"""
        if not isinstance(orientation, tuple) or len(orientation) != 2:
            raise ValueError(
                "RotateAndTiltWrapperPotential: orientation= is a tuple (t, quaternions or rotation matrices)"
            )
        q = orientation[1]
        if not is_backend_array(q):
            q = numpy.array(q, dtype=float)
        # numpy tables stay numpy (also under a forced backend, whose namespace
        # get_namespace would return)
        xp = get_namespace(q) if is_backend_array(q) else numpy
        if q.ndim == 3 and tuple(q.shape[1:]) == (3, 3):
            if not (under_trace(q) or requires_backend_grad(q)) and (
                numpy.amax(
                    numpy.fabs(
                        numpy.einsum("nij,nkj->nik", as_numpy(q), as_numpy(q))
                        - numpy.eye(3)
                    )
                )
                > 1e-8
                or numpy.any(numpy.linalg.det(as_numpy(q)) < 0.0)
            ):
                raise ValueError(
                    "RotateAndTiltWrapperPotential: orientation= matrices must be rotation matrices"
                )
            q = (
                _matrix_to_quaternion(q)
                if xp is numpy
                else _matrix_to_quaternion_xp(xp, q)
            )
        elif q.ndim != 2 or q.shape[1] != 4:
            raise ValueError(
                "RotateAndTiltWrapperPotential: orientation= needs an (N,4) array of quaternions or an (N,3,3) array of rotation matrices"
            )
        if xp is numpy:
            q = _continuous_unit_quaternions(q)
        else:
            q = _continuous_unit_quaternions_xp(xp, q)
        self._orient_t = _parse_table_times(
            orientation[0], q.shape[0], "orientation", ro=self._ro, vo=self._vo
        )
        self._orient_q = q
        self._orient_spline, self._orient_c = _table_spline(self._orient_t, q)
        return None

    def _center_at(self, xp, t, ref):
        """The center (3,) at time t in xp (None when there is none)"""
        if self._center_type is None:
            return None
        elif self._center_type == "constant":
            c = self._center_xyz
            return c if is_backend_array(c) or xp is numpy else like(ref, c)
        elif self._center_type == "function":
            c = _function_value(xp, self._center_func(t), ref)
            if tuple(c.shape) != (3,):
                c = xp.reshape(c, (3,))
            return c
        elif self._center_type == "orbit":
            # held at its end positions outside of the integration times, as in C
            orb = self._center_orb
            if xp is numpy and not is_backend_array(t):
                t = numpy.clip(t, *self._center_trange)
                return numpy.array(
                    [orb.x(t), orb.y(t), orb.z(t) if orb.dim() == 3 else 0.0],
                    dtype=float,
                ).flatten()
            tb = t if is_backend_array(t) else as_backend_constant(xp, t, ref)
            tb = xp.clip(tb, *self._center_trange)
            # a numpy orbit's positions are numpy (_stored): bring them on xp
            oR, ophi = (
                v if is_backend_array(v) else as_backend_constant(xp, v, ref)
                for v in (orb.R(tb, _stored=True), orb.phi(tb, _stored=True))
            )
            if orb.dim() == 3:
                oz = orb.z(tb, _stored=True)
                oz = oz if is_backend_array(oz) else as_backend_constant(xp, oz, ref)
            else:
                oz = 0.0 * oR
            return xp.reshape(
                xp.stack([oR * xp.cos(ophi), oR * xp.sin(ophi), oz * (oR * 0.0 + 1.0)]),
                (3,),
            )
        if xp is numpy and self._center_spline is not None and not is_backend_array(t):
            return self._center_spline(
                numpy.clip(t, self._center_t[0], self._center_t[-1])
            )
        tb = t if is_backend_array(t) else as_backend_constant(xp, t, ref)
        return eval_ppoly(xp, self._center_t, self._center_c, tb, extrapolate="clip")

    def _rot_at(self, xp, t, ref):
        """The rotation matrix at time t in xp (None when there is no
        rotation)"""
        if self._orient_type == "function":
            rot = _function_value(xp, self._orient_func(t), ref)
            if tuple(rot.shape) == (4,):
                rxp = numpy if not is_backend_array(rot) else get_namespace(rot)
                rot = rot / rxp.sqrt(rxp.sum(rot * rot))
                return (
                    _quaternion_to_matrix(rot)
                    if rxp is numpy
                    else _quaternion_to_matrix_xp(rxp, rot)
                )
            if tuple(rot.shape) != (3, 3):
                raise ValueError(
                    "RotateAndTiltWrapperPotential: an orientation= function must return a (3,3) rotation matrix or a (4,) quaternion (w,x,y,z)"
                )
            return rot
        elif self._orient_type == "angles":
            inclination, galaxy_pa, sky_pa, zvec = (
                angle(t) if callable(angle) else angle for angle in self._angle_funcs
            )
            zvec, galaxy_pa = self._parse_inclination(
                inclination, sky_pa, zvec, galaxy_pa
            )
            return _function_value(xp, _zvec_pa_rotation(zvec, galaxy_pa), ref)
        elif self._orientation:
            if (
                xp is numpy
                and self._orient_spline is not None
                and not is_backend_array(t)
            ):
                q = self._orient_spline(
                    numpy.clip(t, self._orient_t[0], self._orient_t[-1])
                )
                return _quaternion_to_matrix(q / numpy.sqrt(numpy.sum(q**2.0)))
            tb = t if is_backend_array(t) else as_backend_constant(xp, t, ref)
            q = eval_ppoly(xp, self._orient_t, self._orient_c, tb, extrapolate="clip")
            return _quaternion_to_matrix_xp(xp, q / xp.sqrt(xp.sum(q * q)))
        return None if self._norot else as_backend_constant(xp, self._rot, ref)

    def _eval_namespace(self, R, z, phi, t):
        """The namespace of an evaluation and its coordinates: the
        coordinates', or, when they are numpy and a center, orientation, or
        angle that is a function of time returns a backend array at t (a
        function closing over backend parameters), that backend's, with the
        coordinates brought onto it"""
        xp = get_namespace(R, z, phi, t)
        if xp is not numpy:
            return xp, R, z, phi, t
        vals = []
        if self._center_type == "function":
            vals.append(self._center_func(t))
        if self._orient_type == "function":
            vals.append(self._orient_func(t))
        elif self._orient_type == "angles":
            vals.extend(angle(t) for angle in self._angle_funcs if callable(angle))
        for v in vals:
            for c in v if isinstance(v, (list, tuple)) else (v,):
                if is_backend_array(c):
                    xp = get_namespace(c)
                    return (xp,) + tuple(coerce_coords(xp, R, z, phi, t))
        return xp, R, z, phi, t

    def __getattr__(self, attribute):
        return super().__getattr__(attribute)

    def _vertical_quad_split(self, R, phi=0.0, t=0.0):
        """z where the WRAPPED potential's mid-plane crosses the vertical line
        at (R, phi).

        ``_rect_transformed`` maps (R, phi, z) to ``rot @ (x,y,z) + offset``, so
        the wrapped z' vanishes at

            z = -(rot[2,0] x + rot[2,1] y + offset[2]) / rot[2,2] .

        Rotating or offsetting the potential moves its disk plane away from
        z = 0, and the vertical quadratures cluster their nodes at the split;
        splitting at 0 then samples a sharply-peaked density where it is flat.
        Edge-on (rot[2,2] == 0) has no crossing -- the wrapped plane contains
        the line of integration -- so keep 0 there.
        """
        if self._norot and self._offset is None and self._center_type is None:
            return 0.0
        if self._center_type is not None or self._orientation:
            # z' = 0 for z' = R(t)[2] . (x - center(t)) + offset[2]
            xp = get_namespace(R, phi)
            x, y, _ = coords.cyl_to_rect(R, phi, 0.0 * R)
            rot = self._rot_at(xp, t, x)
            r2 = (
                rot[2]
                if rot is not None
                else as_backend_constant(xp, numpy.array([0.0, 0.0, 1.0]), x)
            )
            if concretely_true(r2[2] == 0.0):
                return 0.0
            c = self._center_at(xp, t, x)
            if c is None:
                c = 0.0 * r2
            num = r2[0] * (x - c[0]) + r2[1] * (y - c[1]) - r2[2] * c[2]
            if self._offset is not None:
                num = num + as_backend_constant(xp, self._offset, R)[2]
            return -num / r2[2]
        xp = get_namespace(R, phi)
        # float(), not the numpy scalar: dividing a backend array by a
        # numpy.float64 hands the op to numpy, which emits the numpy-2.0
        # __array_wrap__ deprecation (96 of them across the traced-torch
        # surfdens tests). A plain float stays a weak scalar, so it also will
        # not upcast a float32 backend array the way as_backend_constant would.
        if self._norot:
            r22 = 1.0
        elif is_backend_array(self._rot):  # a differentiated rotation
            r22 = self._rot[2, 2]
        else:
            r22 = float(self._rot[2, 2])
        if concretely_true(r22 == 0.0):
            return 0.0
        num = 0.0
        if not self._norot:
            x, y, _ = coords.cyl_to_rect(R, phi, 0.0 * R)
            rot2 = as_backend_constant(xp, self._rot[2, :2], x)
            num = num + rot2[0] * x + rot2[1] * y
        if self._offset is not None:
            num = num + as_backend_constant(xp, self._offset, R)[2]
        return -num / r22

    def _rect_transformed(
        self, xp, R, z, phi, t=0.0, guard_inf=False, return_rot=False
    ):
        """Rectangular coordinates in the rotated/tilted (+offset) frame,
        R(t) . (x - center(t)) + offset.

        With ``guard_inf`` (the _evaluate/_dens R == inf branch), the wrapped
        potential is evaluated at rectangular (R, 0, z) instead; the finite-R
        branch's input is guarded (eager backends evaluate both ``xp.where``
        sides, and inf*cos(phi) would NaN-poison reverse-mode autodiff).
        With ``return_rot``, also returns the rotation matrix used for a
        time-dependent orientation (None otherwise)."""
        if guard_inf:
            Rinf = xp.isinf(R)
            x, y, z = coords.cyl_to_rect(where(xp, Rinf, 1.0, R), phi, z)
            x = where(xp, Rinf, R, x)
            y = where(xp, Rinf, 0.0, y)
        else:
            x, y, z = coords.cyl_to_rect(R, phi, z)
        # With a trailing node axis only z carries it, so the stack would mix
        # rank 0 and rank 1; (3,3) @ (3,N) is the same matmul as (3,3) @ (3,).
        x, y, z = xp.broadcast_arrays(x, y, z)
        xyzp = xp.stack([x, y, z])
        center = self._center_at(xp, t, xyzp)
        if center is not None:
            # (3,), with a trailing singleton per node axis, as the offset
            xyzp = xyzp - xp.reshape(center, (3,) + (1,) * (xyzp.ndim - 1))
        rot = None
        if self._orientation:
            rot = self._rot_at(xp, t, xyzp)
            xyzp = rot @ xyzp
        elif not self._norot:
            xyzp = as_backend_constant(xp, self._rot, xyzp) @ xyzp
        if self._offset is not None:
            # Offset is (3,); give it a trailing singleton per node axis so it
            # broadcasts down the (3, N) stack. No-op when xyzp is (3,).
            offset = as_backend_constant(xp, self._offset, xyzp)
            xyzp = xyzp + xp.reshape(offset, (3,) + (1,) * (xyzp.ndim - 1))
        if guard_inf:
            # at infinity along x, a component without x in it is inf * 0 =
            # nan after the rotation; it is finite
            xyzp = where(xp, Rinf & xp.isnan(xyzp), 0.0, xyzp)
        return (xyzp, rot) if return_rot else xyzp

    @check_potential_inputs_not_arrays
    def _evaluate(self, R, z, phi=0.0, t=0.0):
        xp, R, z, phi, t = self._eval_namespace(R, z, phi, t)
        xyzp = self._rect_transformed(xp, R, z, phi, t=t, guard_inf=True)
        Rp, phip, zp = coords.rect_to_cyl(xyzp[0], xyzp[1], xyzp[2])
        return _evaluatePotentials(self._pot, Rp, zp, phi=phip, t=t)

    @check_potential_inputs_not_arrays
    def _Rforce(self, R, z, phi=0.0, t=0.0):
        xp, R, z, phi, t = self._eval_namespace(R, z, phi, t)
        Fxyz = self._force_xyz(R, z, phi=phi, t=t)
        return xp.cos(phi) * Fxyz[0] + xp.sin(phi) * Fxyz[1]

    @check_potential_inputs_not_arrays
    def _phitorque(self, R, z, phi=0.0, t=0.0):
        xp, R, z, phi, t = self._eval_namespace(R, z, phi, t)
        Fxyz = self._force_xyz(R, z, phi=phi, t=t)
        return R * (-xp.sin(phi) * Fxyz[0] + xp.cos(phi) * Fxyz[1])

    @check_potential_inputs_not_arrays
    def _zforce(self, R, z, phi=0.0, t=0.0):
        return self._force_xyz(R, z, phi=phi, t=t)[2]

    def _force_xyz(self, R, z, phi=0.0, t=0.0):
        """Get the rectangular forces in the transformed frame"""
        xp, R, z, phi, t = self._eval_namespace(R, z, phi, t)
        if xp is not numpy:  # eager: the methods at one point share it
            # coerced first, so float and array calls at one point share a key
            R, z, phi, t = coerce_coords(xp, R, z, phi, t)
            return eager_value_memo(
                self,
                "force_xyz",
                (R, z, phi, t),
                lambda: self._force_xyz_eval(xp, R, z, phi, t),
            )
        return self._force_xyz_eval(xp, R, z, phi, t)

    def _force_xyz_eval(self, xp, R, z, phi, t):
        xyzp, rot = self._rect_transformed(xp, R, z, phi, t=t, return_rot=True)
        Rp, phip, zp = coords.rect_to_cyl(xyzp[0], xyzp[1], xyzp[2])
        Rforcep = _evaluateRforces(self._pot, Rp, zp, phi=phip, t=t)
        phitorquep = _evaluatephitorques(self._pot, Rp, zp, phi=phip, t=t)
        zforcep = _evaluatezforces(self._pot, Rp, zp, phi=phip, t=t)
        xforcep = xp.cos(phip) * Rforcep - xp.sin(phip) * phitorquep / Rp
        yforcep = xp.sin(phip) * Rforcep + xp.cos(phip) * phitorquep / Rp
        Fxyzp = [xforcep, yforcep, zforcep]
        if xp is numpy:  # a backend-parameter potential under a forced numpy
            Fxyzp = to_host(Fxyzp)
        Fxyzp = xp.stack(Fxyzp)
        if rot is not None:
            return rot.T @ Fxyzp
        return as_backend_constant(xp, self._inv_rot, Fxyzp) @ Fxyzp

    @check_potential_inputs_not_arrays
    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        xp, R, z, phi, t = self._eval_namespace(R, z, phi, t)
        phi2 = self._2ndderiv_xyz(R, z, phi=phi, t=t)
        return (
            xp.cos(phi) ** 2.0 * phi2[..., 0, 0]
            + xp.sin(phi) ** 2.0 * phi2[..., 1, 1]
            + 2.0 * xp.cos(phi) * xp.sin(phi) * phi2[..., 0, 1]
        )

    @check_potential_inputs_not_arrays
    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        xp, R, z, phi, t = self._eval_namespace(R, z, phi, t)
        phi2 = self._2ndderiv_xyz(R, z, phi=phi, t=t)
        return xp.cos(phi) * phi2[..., 0, 2] + xp.sin(phi) * phi2[..., 1, 2]

    @check_potential_inputs_not_arrays
    def _z2deriv(self, R, z, phi=0.0, t=0.0):
        return self._2ndderiv_xyz(R, z, phi=phi, t=t)[..., 2, 2]

    @check_potential_inputs_not_arrays
    def _phi2deriv(self, R, z, phi=0.0, t=0.0):
        xp, R, z, phi, t = self._eval_namespace(R, z, phi, t)
        Fxyz = self._force_xyz(R, z, phi=phi, t=t)
        phi2 = self._2ndderiv_xyz(R, z, phi=phi, t=t)
        return R**2.0 * (
            xp.sin(phi) ** 2.0 * phi2[..., 0, 0]
            + xp.cos(phi) ** 2.0 * phi2[..., 1, 1]
            - 2.0 * xp.cos(phi) * xp.sin(phi) * phi2[..., 0, 1]
        ) + R * (xp.cos(phi) * Fxyz[0] + xp.sin(phi) * Fxyz[1])

    @check_potential_inputs_not_arrays
    def _Rphideriv(self, R, z, phi=0.0, t=0.0):
        xp, R, z, phi, t = self._eval_namespace(R, z, phi, t)
        Fxyz = self._force_xyz(R, z, phi=phi, t=t)
        phi2 = self._2ndderiv_xyz(R, z, phi=phi, t=t)
        return (
            R * xp.cos(phi) * xp.sin(phi) * (phi2[..., 1, 1] - phi2[..., 0, 0])
            + R * xp.cos(2.0 * phi) * phi2[..., 0, 1]
            + xp.sin(phi) * Fxyz[0]
            - xp.cos(phi) * Fxyz[1]
        )

    @check_potential_inputs_not_arrays
    def _phizderiv(self, R, z, phi=0.0, t=0.0):
        xp, R, z, phi, t = self._eval_namespace(R, z, phi, t)
        phi2 = self._2ndderiv_xyz(R, z, phi=phi, t=t)
        return R * (xp.cos(phi) * phi2[..., 1, 2] - xp.sin(phi) * phi2[..., 0, 2])

    def _2ndderiv_xyz(self, R, z, phi=0.0, t=0.0):
        """Get the rectangular second derivatives in the transformed frame"""
        xp, R, z, phi, t = self._eval_namespace(R, z, phi, t)
        if xp is not numpy:  # eager: the methods at one point share it
            # coerced first, so float and array calls at one point share a key
            R, z, phi, t = coerce_coords(xp, R, z, phi, t)
            return eager_value_memo(
                self,
                "2ndderiv_xyz",
                (R, z, phi, t),
                lambda: self._2ndderiv_xyz_eval(xp, R, z, phi, t),
            )
        return self._2ndderiv_xyz_eval(xp, R, z, phi, t)

    def _2ndderiv_xyz_eval(self, xp, R, z, phi, t):
        xyzp, rot = self._rect_transformed(xp, R, z, phi, t=t, return_rot=True)
        Rp, phip, zp = coords.rect_to_cyl(xyzp[0], xyzp[1], xyzp[2])
        Rforcep = _evaluateRforces(self._pot, Rp, zp, phi=phip, t=t)
        phitorquep = _evaluatephitorques(self._pot, Rp, zp, phi=phip, t=t)
        R2derivp = evaluateR2derivs(
            self._pot, Rp, zp, phi=phip, t=t, use_physical=False
        )
        phi2derivp = evaluatephi2derivs(
            self._pot, Rp, zp, phi=phip, t=t, use_physical=False
        )
        z2derivp = evaluatez2derivs(
            self._pot, Rp, zp, phi=phip, t=t, use_physical=False
        )
        Rzderivp = evaluateRzderivs(
            self._pot, Rp, zp, phi=phip, t=t, use_physical=False
        )
        Rphiderivp = evaluateRphiderivs(
            self._pot, Rp, zp, phi=phip, t=t, use_physical=False
        )
        phizderivp = evaluatephizderivs(
            self._pot, Rp, zp, phi=phip, t=t, use_physical=False
        )
        cp, sp = xp.cos(phip), xp.sin(phip)
        cp2, sp2, cpsp = cp**2.0, sp**2.0, cp * sp
        Rp2 = Rp * Rp
        x2derivp = (
            R2derivp * cp2
            - 2.0 * Rphiderivp * cpsp / Rp
            + phi2derivp * sp2 / Rp2
            - Rforcep * sp2 / Rp
            - 2.0 * phitorquep * cpsp / Rp2
        )
        y2derivp = (
            R2derivp * sp2
            + 2.0 * Rphiderivp * cpsp / Rp
            + phi2derivp * cp2 / Rp2
            - Rforcep * cp2 / Rp
            + 2.0 * phitorquep * cpsp / Rp2
        )
        xyderivp = (
            R2derivp * cpsp
            + Rphiderivp * (cp2 - sp2) / Rp
            - phi2derivp * cpsp / Rp2
            + Rforcep * cpsp / Rp
            + phitorquep * (cp2 - sp2) / Rp2
        )
        xzderivp = Rzderivp * cp - phizderivp * sp / Rp
        yzderivp = Rzderivp * sp + phizderivp * cp / Rp
        # Build as (..., 3, 3) -- tensor axes LAST -- so any node axis is a
        # leading batch axis that matmul carries through. einsum would also
        # broadcast but picks a different contraction order and is NOT numpy
        # byte-identical; chained matmul on (3,3) executes exactly as before.
        deriv2p = xp.stack(
            [
                xp.stack([x2derivp, xyderivp, xzderivp], axis=-1),
                xp.stack([xyderivp, y2derivp, yzderivp], axis=-1),
                xp.stack([xzderivp, yzderivp, z2derivp], axis=-1),
            ],
            axis=-2,
        )
        if rot is not None:
            return rot.T @ (deriv2p @ rot)
        inv_rot = as_backend_constant(xp, self._inv_rot, deriv2p)
        inv_rot_T = as_backend_constant(xp, self._inv_rot.T, deriv2p)
        return inv_rot @ (deriv2p @ inv_rot_T)

    @check_potential_inputs_not_arrays
    def _dens(self, R, z, phi=0.0, t=0.0):
        xp, R, z, phi, t = self._eval_namespace(R, z, phi, t)
        xyzp = self._rect_transformed(xp, R, z, phi, t=t, guard_inf=True)
        Rp, phip, zp = coords.rect_to_cyl(xyzp[0], xyzp[1], xyzp[2])
        return evaluateDensities(self._pot, Rp, zp, phi=phip, t=t, use_physical=False)


def _function_value(xp, v, ref):
    """The value v of a center= or orientation= function (or of the angles'
    rotation), a numpy array, a backend array, or a list holding backend
    scalars, as an array in xp (numpy when xp is numpy)"""
    if isinstance(v, (list, tuple)) and any(is_backend_array(c) for c in v):
        r = next(c for c in v if is_backend_array(c))
        rxp = namespace_from_arrays((r,))
        v = rxp.stack(promote_scalars(rxp, *v))
    if is_backend_array(v):
        return v
    v = numpy.array(v, dtype=float)
    return v if xp is numpy else as_backend_constant(xp, v, ref)


def _continuous_unit_quaternions(q):
    """The (N,4) quaternions q normalized, with the sign (q and -q are the
    same rotation) continuous along the table, so that their interpolation
    does not pass through zero"""
    q = q / numpy.sqrt(numpy.sum(q**2.0, axis=1))[:, None]
    flips = numpy.sum(q[1:] * q[:-1], axis=1) < 0.0
    q[1:][numpy.cumsum(flips) % 2 == 1] *= -1.0
    return q


def _orbit_like(obj):
    """Whether obj is an Orbit or has the interface of one that the center= uses"""
    return all(
        callable(getattr(obj, a, None))
        for a in ("x", "y", "z", "dim", "turn_physical_off")
    )


def _parse_table_times(t, n, name, ro=None, vo=None):
    """Times of a center= or orientation= table, checked to be increasing
    (when concrete)"""
    t = conversion.parse_time(t, ro=ro, vo=vo)
    if not is_backend_array(t):
        t = numpy.array(t, dtype=float)
    if t.ndim != 1 or t.shape[0] != n:
        raise ValueError(
            f"RotateAndTiltWrapperPotential: the times of a tabulated {name}= must be a 1D array of the same length as the table"
        )
    if n < 3:
        raise ValueError(
            f"RotateAndTiltWrapperPotential: a tabulated {name}= needs at least 3 times"
        )
    if not (under_trace(t) or requires_backend_grad(t)) and numpy.any(
        numpy.diff(as_numpy(t)) <= 0.0
    ):
        raise ValueError(
            f"RotateAndTiltWrapperPotential: the times of a tabulated {name}= must be increasing"
        )
    return t


def _table_spline(t, table):
    """A natural cubic spline of a table along its first axis: scipy's (and
    its power-basis coefficients, for evaluation on a backend) for numpy
    tables; the backend's (differentiable, traceable) coefficients for backend
    tables (then no scipy spline)"""
    if not (is_backend_array(t) or is_backend_array(table)):
        spline = interpolate.CubicSpline(t, table, axis=0, bc_type="natural")
        return spline, spline.c
    ref = table if is_backend_array(table) else t
    xp = get_namespace(ref)
    if not is_backend_array(table):
        table = as_backend_constant(xp, table, ref)
    return None, cubic_spline_coeffs(xp, t, table, bc="natural")


def _quaternion_to_matrix_xp(xp, q):
    """_quaternion_to_matrix in the namespace xp (differentiable)"""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return xp.stack(
        [
            xp.stack(
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - w * z),
                    2.0 * (x * z + w * y),
                ]
            ),
            xp.stack(
                [
                    2.0 * (x * y + w * z),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - w * x),
                ]
            ),
            xp.stack(
                [
                    2.0 * (x * z - w * y),
                    2.0 * (y * z + w * x),
                    1.0 - 2.0 * (x * x + y * y),
                ]
            ),
        ]
    )


def _matrix_to_quaternion_xp(xp, rot):
    """_matrix_to_quaternion in the namespace xp: all four candidates, the
    one with the largest component selected without branching (the square
    root guarded, so that the unselected candidates stay finite under
    autodiff)"""
    m = rot
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    cands = [1.0 + trace] + [1.0 + 2.0 * m[:, ii, ii] - trace for ii in range(3)]
    s = [2.0 * xp.sqrt(where(xp, c > 0.0, c, 1.0)) for c in cands]
    a21, a02, a10 = (
        m[:, 2, 1] - m[:, 1, 2],
        m[:, 0, 2] - m[:, 2, 0],
        m[:, 1, 0] - m[:, 0, 1],
    )
    s01, s02, s12 = (
        m[:, 0, 1] + m[:, 1, 0],
        m[:, 0, 2] + m[:, 2, 0],
        m[:, 1, 2] + m[:, 2, 1],
    )
    candq = [
        [s[0] / 4.0, a21 / s[0], a02 / s[0], a10 / s[0]],
        [a21 / s[1], s[1] / 4.0, s01 / s[1], s02 / s[1]],
        [a02 / s[2], s01 / s[2], s[2] / 4.0, s12 / s[2]],
        [a10 / s[3], s02 / s[3], s12 / s[3], s[3] / 4.0],
    ]
    pick = xp.argmax(xp.stack(cands, axis=1), axis=1)
    q = [0.0 * trace for _ in range(4)]
    for kk in range(4):
        sel = where(xp, pick == kk, 1.0, 0.0)
        q = [q[jj] + sel * candq[kk][jj] for jj in range(4)]
    return xp.stack(q, axis=1)


def _continuous_unit_quaternions_xp(xp, q):
    """Normalized quaternions with the sign continuous along the table (as
    for numpy tables in _setup_orientation), in the namespace xp"""
    q = q / xp.sqrt(xp.sum(q * q, axis=1))[:, None]
    flips = where(xp, xp.sum(q[1:] * q[:-1], axis=1) < 0.0, 1.0, 0.0)
    sign = 1.0 - 2.0 * (xp.cumsum(flips, axis=0) % 2.0)
    concat = getattr(xp, "concat", None) or xp.concatenate
    return concat([q[:1], q[1:] * sign[:, None]], axis=0)


def _quaternion_to_matrix(q):
    """Rotation matrix of the unit quaternion q = (w,x,y,z) (the same formula
    as in RotateAndTiltWrapperPotential.c)"""
    w, x, y, z = q
    return numpy.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def _matrix_to_quaternion(rot):
    """Unit quaternions (w,x,y,z) of an (N,3,3) array of rotation matrices,
    from the largest of the four candidate components (Shepperd's method)"""
    trace = numpy.trace(rot, axis1=1, axis2=2)
    diag = numpy.diagonal(rot, axis1=1, axis2=2)
    # 4 w^2, 4 x^2, 4 y^2, 4 z^2
    cands = numpy.stack(
        [1.0 + trace] + [1.0 + 2.0 * diag[:, ii] - trace for ii in range(3)],
        axis=1,
    )
    pick = numpy.argmax(cands, axis=1)
    q = numpy.empty((len(rot), 4))
    for ii in range(len(rot)):
        m = rot[ii]
        s = 2.0 * numpy.sqrt(cands[ii, pick[ii]])
        if pick[ii] == 0:
            q[ii] = [
                s / 4.0,
                (m[2, 1] - m[1, 2]) / s,
                (m[0, 2] - m[2, 0]) / s,
                (m[1, 0] - m[0, 1]) / s,
            ]
        elif pick[ii] == 1:
            q[ii] = [
                (m[2, 1] - m[1, 2]) / s,
                s / 4.0,
                (m[0, 1] + m[1, 0]) / s,
                (m[0, 2] + m[2, 0]) / s,
            ]
        elif pick[ii] == 2:
            q[ii] = [
                (m[0, 2] - m[2, 0]) / s,
                (m[0, 1] + m[1, 0]) / s,
                s / 4.0,
                (m[1, 2] + m[2, 1]) / s,
            ]
        else:
            q[ii] = [
                (m[1, 0] - m[0, 1]) / s,
                (m[0, 2] + m[2, 0]) / s,
                (m[1, 2] + m[2, 1]) / s,
                s / 4.0,
            ]
    return q
