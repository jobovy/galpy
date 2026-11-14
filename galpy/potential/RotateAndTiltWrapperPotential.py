###############################################################################
#   RotateAndTiltWrapperPotential.py: Wrapper to rotate and tilt the z-axis
#   of a potential
###############################################################################
import copy

import numpy
from scipy import interpolate

from ..util import _rotate_to_arbitrary_vector, conversion, coords
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
    ):
        """
        A potential that rotates and tilts another potential.

        Parameters
        ----------
        amp : float, optional
            Overall amplitude to apply to the potential. Default is 1.0.
        inclination : float or Quantity, optional
            Usual inclination angle (with the line-of-sight being the z axis).
        galaxy_pa : float or Quantity, optional
            Rotation angle of the original potential around the original z axis.
        sky_pa : float or Quantity, optional
            Rotation angle around the inclined z axis (usual sky position angle measured from North).
        zvec : numpy.ndarray, optional
            3D vector specifying the direction of the rotated z axis.
        offset : numpy.ndarray or Quantity, optional
            Static offset in Cartesian coordinates, applied in the frame of the wrapped potential (after the rotation).
        pot : Potential instance or a combined potential formed using addition (pot1+pot2+…)
            The Potential instance or a combined potential formed using addition (pot1+pot2+…) to rotate and tilt.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).
        center : numpy.ndarray, Quantity, tuple, or Orbit, optional
            Position of the wrapped potential's origin in the (Galactocentric) frame of the evaluation point, before the rotation: a constant (x,y,z), a tabulated path ``(t, xyz)`` with ``t`` an increasing array of N >= 3 times and ``xyz`` an (N,3) array of positions (interpolated with a natural cubic spline, and held at its end values outside of the tabulated times), or an integrated ``Orbit`` (its positions; interpolated by the Orbit itself in Python and by a natural cubic spline of its integration output in C).
        orientation : tuple, optional
            Time-dependent rotation R(t), as ``(t, q)`` with ``t`` an increasing array of N >= 3 times and ``q`` either an (N,4) array of unit quaternions (w,x,y,z) or an (N,3,3) array of rotation matrices; the rows of R(t) are the wrapped potential's axes in the frame of the evaluation point. The quaternions are interpolated with natural cubic splines and normalized after interpolation (held at their end values outside of the tabulated times); see ``orientation_from_angles`` to tabulate them from angles. Cannot be combined with inclination, galaxy_pa, sky_pa, or zvec.

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
                "RotateAndTiltWrapperPotential: specify the orientation either as a time-dependent orientation= or through the constant inclination, galaxy_pa, sky_pa, and zvec, not both"
            )
        inclination = conversion.parse_angle(inclination)
        sky_pa = conversion.parse_angle(sky_pa)
        galaxy_pa = conversion.parse_angle(galaxy_pa)
        zvec, galaxy_pa = self._parse_inclination(inclination, sky_pa, zvec, galaxy_pa)
        self._offset = conversion.parse_length(
            numpy.array(offset) if isinstance(offset, list) else offset, ro=self._ro
        )
        self._setup_zvec_pa(zvec, galaxy_pa)
        self._norot = False
        if (self._rot == numpy.eye(3)).all():
            self._norot = True
        self._setup_center(center)
        self._setup_orientation(orientation)
        self.hasC = True
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

    def _setup_zvec_pa(self, zvec, pa):
        if not pa is None:
            pa_rot = numpy.array(
                [
                    [numpy.cos(pa), numpy.sin(pa), 0.0],
                    [-numpy.sin(pa), numpy.cos(pa), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        else:
            pa_rot = numpy.eye(3)
        if not zvec is None:
            # a float copy: an integer zvec cannot be normalized in place, and
            # the caller's array is not to be modified
            zvec = numpy.array(zvec, dtype=float)
            zvec /= numpy.sqrt(numpy.sum(zvec**2.0))
            zvec_rot = _rotate_to_arbitrary_vector(
                numpy.array([[0.0, 0.0, 1.0]]), zvec, inv=True
            )[0]
        else:
            zvec_rot = numpy.eye(3)
        self._rot = numpy.dot(pa_rot, zvec_rot)
        self._inv_rot = numpy.linalg.inv(self._rot)
        return None

    def _setup_center(self, center):
        """Set up the (time-dependent) center: None, constant, table, or Orbit"""
        from ..orbit import Orbit

        self._center_type = None
        if center is None:
            return None
        if isinstance(center, Orbit):
            if center.size != 1:
                raise ValueError(
                    "RotateAndTiltWrapperPotential: center= Orbit must hold a single object"
                )
            if not hasattr(center, "t"):
                raise ValueError(
                    "RotateAndTiltWrapperPotential: center= Orbit must be integrated"
                )
            self._center_type = "orbit"
            self._center_orb = copy.deepcopy(center)
            self._center_orb.turn_physical_off()
            # the tabulation passed to C: the integration output, increasing
            # in time
            tc = numpy.array(self._center_orb.t, dtype=float)
            xyzc = numpy.array(
                [
                    self._center_orb.x(tc),
                    self._center_orb.y(tc),
                    (
                        self._center_orb.z(tc)
                        if self._center_orb.dim() == 3
                        else numpy.zeros_like(tc)
                    ),
                ]
            ).T
            if tc[-1] < tc[0]:
                tc, xyzc = tc[::-1], xyzc[::-1]
            self._center_t, self._center_xyz = tc, xyzc
        elif isinstance(center, tuple):
            if len(center) != 2:
                raise ValueError(
                    "RotateAndTiltWrapperPotential: a tabulated center= is a tuple (t, xyz)"
                )
            tc = _parse_table_times(
                center[0], len(center[1]), "center", ro=self._ro, vo=self._vo
            )
            xyzc = numpy.array(
                conversion.parse_length(center[1], ro=self._ro), dtype=float
            )
            if xyzc.shape != (len(tc), 3):
                raise ValueError(
                    "RotateAndTiltWrapperPotential: a tabulated center= (t, xyz) needs xyz of shape (len(t), 3)"
                )
            self._center_type = "table"
            self._center_t, self._center_xyz = tc, xyzc
            self._center_spline = interpolate.CubicSpline(
                tc, xyzc, axis=0, bc_type="natural"
            )
        else:
            center = numpy.array(
                conversion.parse_length(
                    numpy.array(center) if isinstance(center, list) else center,
                    ro=self._ro,
                ),
                dtype=float,
            )
            if center.shape != (3,):
                raise ValueError(
                    "RotateAndTiltWrapperPotential: a constant center= is a position (x, y, z)"
                )
            self._center_type = "constant"
            self._center_xyz = center
        return None

    def _setup_orientation(self, orientation):
        """Set up the time-dependent orientation from a table of quaternions or
        rotation matrices"""
        self._orientation = orientation is not None
        if not self._orientation:
            return None
        if not isinstance(orientation, tuple) or len(orientation) != 2:
            raise ValueError(
                "RotateAndTiltWrapperPotential: orientation= is a tuple (t, quaternions or rotation matrices)"
            )
        q = numpy.array(orientation[1], dtype=float)
        if q.ndim == 3 and q.shape[1:] == (3, 3):
            if numpy.amax(
                numpy.fabs(numpy.einsum("nij,nkj->nik", q, q) - numpy.eye(3))
            ) > 1e-8 or numpy.any(numpy.linalg.det(q) < 0.0):
                raise ValueError(
                    "RotateAndTiltWrapperPotential: orientation= matrices must be rotation matrices"
                )
            q = _matrix_to_quaternion(q)
        elif q.ndim != 2 or q.shape[1] != 4:
            raise ValueError(
                "RotateAndTiltWrapperPotential: orientation= needs an (N,4) array of quaternions or an (N,3,3) array of rotation matrices"
            )
        q = q / numpy.sqrt(numpy.sum(q**2.0, axis=1))[:, None]
        # q and -q are the same rotation: pick the sign continuous along the
        # table, so that the interpolation does not pass through zero
        flips = numpy.sum(q[1:] * q[:-1], axis=1) < 0.0
        q[1:][numpy.cumsum(flips) % 2 == 1] *= -1.0
        self._orient_t = _parse_table_times(
            orientation[0], len(q), "orientation", ro=self._ro, vo=self._vo
        )
        self._orient_q = q
        self._orient_spline = interpolate.CubicSpline(
            self._orient_t, q, axis=0, bc_type="natural"
        )
        self._norot = False
        return None

    def _center_at(self, t):
        """The center at time t (None when there is none)"""
        if self._center_type is None:
            return None
        elif self._center_type == "constant":
            return self._center_xyz
        elif self._center_type == "orbit":
            orb = self._center_orb
            return numpy.array(
                [orb.x(t), orb.y(t), orb.z(t) if orb.dim() == 3 else 0.0],
                dtype=float,
            ).flatten()
        return self._center_spline(numpy.clip(t, self._center_t[0], self._center_t[-1]))

    def _rot_at(self, t):
        """The rotation matrix at time t (None when there is no rotation)"""
        if self._orientation:
            q = self._orient_spline(
                numpy.clip(t, self._orient_t[0], self._orient_t[-1])
            )
            return _quaternion_to_matrix(q / numpy.sqrt(numpy.sum(q**2.0)))
        return None if self._norot else self._rot

    def _to_body(self, x, y, z, t):
        """The evaluation point (x,y,z) at time t in the wrapped potential's
        frame, and the rotation matrix used (None when there is none)"""
        xyz = numpy.array([x, y, z])
        if self._center_type is not None:
            xyz = xyz - self._center_at(t)
        rot = self._rot_at(t)
        xyzp = xyz if rot is None else numpy.dot(rot, xyz)
        if self._offset is not None:
            xyzp += self._offset
        if numpy.isinf(x):
            # at infinity along x (see _evaluate and _dens), a component
            # without x in it is inf * 0 = nan after the rotation; it is
            # finite
            xyzp[numpy.isnan(xyzp)] = 0.0
        return xyzp, rot

    def _inv_rot_at(self, rot):
        """The inverse of the rotation rot returned by _to_body"""
        return rot.T if self._orientation else self._inv_rot

    @staticmethod
    def orientation_from_angles(
        t, inclination=None, galaxy_pa=None, sky_pa=None, zvec=None
    ):
        """
        Tabulate a time-dependent orientation from tabulated angles, for the ``orientation=`` input.

        Parameters
        ----------
        t : numpy.ndarray or Quantity
            Times of the tabulation.
        inclination : float, numpy.ndarray, or Quantity, optional
            Inclination at each time (or constant), as for the constant orientation.
        galaxy_pa : float, numpy.ndarray, or Quantity, optional
            Rotation angle around the original z axis at each time (or constant).
        sky_pa : float, numpy.ndarray, or Quantity, optional
            Sky position angle at each time (or constant).
        zvec : numpy.ndarray, optional
            Direction of the rotated z axis at each time, as an (N,3) array (or a constant (3,) array).

        Returns
        -------
        tuple
            ``(t, R)`` with ``R`` the (N,3,3) rotation matrices, to be given as ``orientation=``.

        Notes
        -----
        - 2026-11-14 - Written - Bovy (UofT)
        """
        nt = len(t)
        angles = [
            None if angle is None else conversion.parse_angle(angle)
            for angle in (inclination, galaxy_pa, sky_pa)
        ]
        zvec = None if zvec is None else numpy.array(zvec, dtype=float)

        def _at(angle, ii):
            if angle is None:
                return None
            return angle[ii] if numpy.ndim(angle) > 0 else angle

        # only the rotation set-up of the constant orientation is needed
        rtwp = object.__new__(RotateAndTiltWrapperPotential)
        rots = numpy.empty((nt, 3, 3))
        for ii in range(nt):
            zvec_ii, galaxy_pa_ii = rtwp._parse_inclination(
                _at(angles[0], ii),
                _at(angles[2], ii),
                None if zvec is None else (zvec[ii] if zvec.ndim == 2 else zvec),
                _at(angles[1], ii),
            )
            rtwp._setup_zvec_pa(zvec_ii, galaxy_pa_ii)
            rots[ii] = rtwp._rot
        return (t, rots)

    def __getattr__(self, attribute):
        return super().__getattr__(attribute)

    @check_potential_inputs_not_arrays
    def _evaluate(self, R, z, phi=0.0, t=0.0):
        x, y, z = coords.cyl_to_rect(R, phi, z) if not numpy.isinf(R) else (R, 0.0, z)
        xyzp, _ = self._to_body(x, y, z, t)
        Rp, phip, zp = coords.rect_to_cyl(xyzp[0], xyzp[1], xyzp[2])
        return _evaluatePotentials(self._pot, Rp, zp, phi=phip, t=t)

    @check_potential_inputs_not_arrays
    def _Rforce(self, R, z, phi=0.0, t=0.0):
        Fxyz = self._force_xyz(R, z, phi=phi, t=t)
        return numpy.cos(phi) * Fxyz[0] + numpy.sin(phi) * Fxyz[1]

    @check_potential_inputs_not_arrays
    def _phitorque(self, R, z, phi=0.0, t=0.0):
        Fxyz = self._force_xyz(R, z, phi=phi, t=t)
        return R * (-numpy.sin(phi) * Fxyz[0] + numpy.cos(phi) * Fxyz[1])

    @check_potential_inputs_not_arrays
    def _zforce(self, R, z, phi=0.0, t=0.0):
        return self._force_xyz(R, z, phi=phi, t=t)[2]

    def _force_xyz(self, R, z, phi=0.0, t=0.0):
        """Get the rectangular forces in the transformed frame"""
        x, y, z = coords.cyl_to_rect(R, phi, z)
        xyzp, rot = self._to_body(x, y, z, t)
        Rp, phip, zp = coords.rect_to_cyl(xyzp[0], xyzp[1], xyzp[2])
        Rforcep = _evaluateRforces(self._pot, Rp, zp, phi=phip, t=t)
        phitorquep = _evaluatephitorques(self._pot, Rp, zp, phi=phip, t=t)
        zforcep = _evaluatezforces(self._pot, Rp, zp, phi=phip, t=t)
        xforcep = numpy.cos(phip) * Rforcep - numpy.sin(phip) * phitorquep / Rp
        yforcep = numpy.sin(phip) * Rforcep + numpy.cos(phip) * phitorquep / Rp
        return numpy.dot(
            self._inv_rot_at(rot), numpy.array([xforcep, yforcep, zforcep])
        )

    @check_potential_inputs_not_arrays
    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        phi2 = self._2ndderiv_xyz(R, z, phi=phi, t=t)
        return (
            numpy.cos(phi) ** 2.0 * phi2[0, 0]
            + numpy.sin(phi) ** 2.0 * phi2[1, 1]
            + 2.0 * numpy.cos(phi) * numpy.sin(phi) * phi2[0, 1]
        )

    @check_potential_inputs_not_arrays
    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        phi2 = self._2ndderiv_xyz(R, z, phi=phi, t=t)
        return numpy.cos(phi) * phi2[0, 2] + numpy.sin(phi) * phi2[1, 2]

    @check_potential_inputs_not_arrays
    def _z2deriv(self, R, z, phi=0.0, t=0.0):
        return self._2ndderiv_xyz(R, z, phi=phi, t=t)[2, 2]

    @check_potential_inputs_not_arrays
    def _phi2deriv(self, R, z, phi=0.0, t=0.0):
        Fxyz = self._force_xyz(R, z, phi=phi, t=t)
        phi2 = self._2ndderiv_xyz(R, z, phi=phi, t=t)
        return R**2.0 * (
            numpy.sin(phi) ** 2.0 * phi2[0, 0]
            + numpy.cos(phi) ** 2.0 * phi2[1, 1]
            - 2.0 * numpy.cos(phi) * numpy.sin(phi) * phi2[0, 1]
        ) + R * (numpy.cos(phi) * Fxyz[0] + numpy.sin(phi) * Fxyz[1])

    @check_potential_inputs_not_arrays
    def _Rphideriv(self, R, z, phi=0.0, t=0.0):
        Fxyz = self._force_xyz(R, z, phi=phi, t=t)
        phi2 = self._2ndderiv_xyz(R, z, phi=phi, t=t)
        return (
            R * numpy.cos(phi) * numpy.sin(phi) * (phi2[1, 1] - phi2[0, 0])
            + R * numpy.cos(2.0 * phi) * phi2[0, 1]
            + numpy.sin(phi) * Fxyz[0]
            - numpy.cos(phi) * Fxyz[1]
        )

    @check_potential_inputs_not_arrays
    def _phizderiv(self, R, z, phi=0.0, t=0.0):
        phi2 = self._2ndderiv_xyz(R, z, phi=phi, t=t)
        return R * (numpy.cos(phi) * phi2[1, 2] - numpy.sin(phi) * phi2[0, 2])

    def _2ndderiv_xyz(self, R, z, phi=0.0, t=0.0):
        """Get the rectangular forces in the transformed frame"""
        x, y, z = coords.cyl_to_rect(R, phi, z)
        xyzp, rot = self._to_body(x, y, z, t)
        inv_rot = self._inv_rot_at(rot)
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
        cp, sp = numpy.cos(phip), numpy.sin(phip)
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
        return numpy.dot(
            inv_rot,
            numpy.dot(
                numpy.array(
                    [
                        [x2derivp, xyderivp, xzderivp],
                        [xyderivp, y2derivp, yzderivp],
                        [xzderivp, yzderivp, z2derivp],
                    ]
                ),
                inv_rot.T,
            ),
        )

    @check_potential_inputs_not_arrays
    def _dens(self, R, z, phi=0.0, t=0.0):
        x, y, z = coords.cyl_to_rect(R, phi, z) if not numpy.isinf(R) else (R, 0.0, z)
        xyzp, _ = self._to_body(x, y, z, t)
        Rp, phip, zp = coords.rect_to_cyl(xyzp[0], xyzp[1], xyzp[2])
        return evaluateDensities(self._pot, Rp, zp, phi=phip, t=t, use_physical=False)


def _parse_table_times(t, n, name, ro=None, vo=None):
    """Times of a center= or orientation= table, checked to be increasing"""
    t = numpy.array(conversion.parse_time(t, ro=ro, vo=vo), dtype=float)
    if t.ndim != 1 or len(t) != n:
        raise ValueError(
            f"RotateAndTiltWrapperPotential: the times of a tabulated {name}= must be a 1D array of the same length as the table"
        )
    if n < 3:
        raise ValueError(
            f"RotateAndTiltWrapperPotential: a tabulated {name}= needs at least 3 times"
        )
    if numpy.any(numpy.diff(t) <= 0.0):
        raise ValueError(
            f"RotateAndTiltWrapperPotential: the times of a tabulated {name}= must be increasing"
        )
    return t


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
