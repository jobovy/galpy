###############################################################################
#   NonInertialFrameForce: Class that implements the fictitious forces
#                          present when integrating orbits in a non-intertial
#                          frame
###############################################################################
import hashlib
import warnings

import numpy
import numpy.linalg

from ..backend import (
    as_backend_constant,
    coerce_coords,
    get_namespace,
    is_backend_array,
)
from ..util import conversion, coords, galpyWarning
from .DissipativeForce import DissipativeForce


def _is_freq_func(x):
    """Whether a frequency specification (Omega or Omegadot) is given as
    function(s) of time rather than constant(s): a callable, or a sequence whose
    first element is callable."""
    if callable(x):
        return True
    try:
        return callable(x[0])
    except (TypeError, IndexError, KeyError):
        return False


def _time_derivative(func, dt=1e-6):
    """Central finite-difference time derivative of a scalar time function,
    used to derive Omegadot from Omega when Omegadot is not provided (only for
    the direct-function-evaluation path; cinterp uses the spline derivative)."""
    return lambda t: (func(t + dt) - func(t - dt)) / (2.0 * dt)


class NonInertialFrameForce(DissipativeForce):
    """Class that implements the fictitious forces present when integrating
    orbits in a non-intertial frame. Coordinates in the inertial frame
    :math:`\\mathbf{x}` and in the non-inertial frame :math:`\\mathbf{r}` are
    related through rotation and linear motion as

    .. math::

        \\mathbf{x} = \\mathbf{R}\\,\\left(\\mathbf{r} + \\mathbf{x}_0\\right)

    where :math:`\\mathbf{R}` is a rotation matrix and :math:`\\mathbf{x}_0`
    is the motion of the origin. The rotation matrix has angular frequencies
    :math:`\\boldsymbol{\\Omega}` with time derivative :math:`\\dot{\\boldsymbol{\\Omega}}`;
    :math:`\\boldsymbol{\\Omega}` can be any function of time (note that the sign of :math:`\\boldsymbol{\\Omega}` is such that :math:`\\boldsymbol{\\Omega}` is the frequency of the rotating frame as seen from the inertial frame). The motion of the
    origin can also be any function of time.
    This leads to the fictitious force

    .. math::

        \\mathbf{F} = -\\mathbf{a}_0 - \\boldsymbol{\\Omega} \\times ( \\boldsymbol{\\Omega} \\times \\left[\\mathbf{r} + \\mathbf{x}_0\\right]) - \\dot{\\boldsymbol{\\Omega}} \\times \\left[\\mathbf{r}+\\mathbf{x}_0\\right] -2\\boldsymbol{\\Omega}\\times \\left[\\dot{\\mathbf{r}}+\\mathbf{v}_0\\right]

    where :math:`\\mathbf{a}_0`, :math:`\\mathbf{v}_0`, and :math:`\\mathbf{x}_0` are
    the acceleration, velocity, and position of the origin of the non-inertial frame,
    respectively, as a function of time. Note that if the non-inertial frame is not
    rotating, it is not necessary to specify :math:`\\mathbf{v}_0` and :math:`\\mathbf{x}_0`.
    In that case, the fictitious force is simply

    .. math::

        \\mathbf{F} = -\\mathbf{a}_0\\quad (\\boldsymbol{\\Omega} = 0)

    If the non-inertial frame only rotates without any motion of the origin, the
    fictitious force is the familiar combination of the centrifugal force
    and the Coriolis force (plus an additional term if :math:`\\dot{\\boldsymbol{\\Omega}}`
    is not constant)

    .. math::

        \\mathbf{F} = - \\boldsymbol{\\Omega} \\times ( \\boldsymbol{\\Omega} \\times \\mathbf{r}) - \\dot{\\boldsymbol{\\Omega}} \\times \\mathbf{r} -2\\boldsymbol{\\Omega}\\times \\dot{\\mathbf{r}}\\quad (\\mathbf{a}_0=\\mathbf{v}_0=\\mathbf{x}_0=0)

    For fast orbit integration in C, by default (``cinterp=True``) the
    time-dependent inputs are replaced by cubic-spline interpolations built over
    the integration's time range (see the ``cinterp`` keyword), so the provided
    functions only need to be evaluated ``cinterp_n`` times when setting up the
    integration and are not called from C at every step. Alternatively, with
    ``cinterp=False`` the functions are passed directly to the C code, which
    attempts to build fast ``numba`` versions of them; significant speed-ups
    then require that the provided functions can be turned into
    ``nopython=True`` ``numba`` functions (try running ``numba.njit`` on them
    and then evaluating them to check).
    """

    def __init__(
        self,
        amp=1.0,
        Omega=None,
        Omegadot=None,
        x0=None,
        v0=None,
        a0=None,
        cinterp=True,
        cinterp_n=3000,
        ro=None,
        vo=None,
    ):
        """
        Initialize a NonInertialFrameForce.

        Parameters
        ----------
        amp : float, optional
            Amplitude to be applied to the potential (default: 1).
        Omega : float or list of floats or Quantity or list of Quantities or callable or list of callables, optional
            Angular frequency of the rotation of the non-inertial frame as seen from an inertial one; can either be a function of time or a number (when the frequency is assumed to be Omega + Omegadot x t) and in each case can be a list [Omega_x,Omega_y,Omega_z] or a single value Omega_z (when not a function, can be a Quantity; when a function, need to take input time in internal units and output the frequency in internal units; see galpy.util.conversion.time_in_Gyr and galpy.util.conversion.freq_in_XXX conversion functions).
        Omegadot : float or list of floats or Quantity or list of Quantities or callable or list of callables, optional
            Time derivative of the angular frequency of the non-intertial frame's rotation. Must match the *kind* of Omega: a [list of] function[s] when Omega is one, or a number/list/Quantity when Omega is one (a mismatch raises an error). When a function, need to take input time in internal units and output the frequency derivative in internal units; see galpy.util.conversion.time_in_Gyr and galpy.util.conversion.freq_in_XXX conversion functions. If Omega is a function of time and Omegadot is omitted, Omegadot is derived as the time-derivative of Omega (a warning is issued); if Omega is constant and Omegadot is omitted, the frequency is taken to be constant in time.
        x0 : list of callables, optional
            Position vector x_0 (cartesian) of the center of mass of the non-intertial frame (see definition in the class documentation); list of functions [x_0x,x_0y,x_0z]; only necessary when considering both rotation and center-of-mass acceleration of the inertial frame (functions need to take input time in internal units and output the position in internal units; see galpy.util.conversion.time_in_Gyr and divided physical positions by the `ro` parameter in kpc).
        v0 : list of callables, optional
            Velocity vector v_0 (cartesian) of the center of mass of the non-intertial frame (see definition in the class documentation); list of functions [v_0x,v_0y,v_0z]; only necessary when considering both rotation and center-of-mass acceleration of the inertial frame (functions need to take input time in internal units and output the velocity in internal units; see galpy.util.conversion.time_in_Gyr and divided physical positions by the `vo` parameter in km/s).
        a0 : float or list of callables, optional
            Acceleration vector a_0 (cartesian) of the center of mass of the non-intertial frame (see definition in the class documentation); constant or a list of functions [a_0x,a_0y, a_0z] (functions need to take input time in internal units and output the acceleration in internal units; see galpy.util.conversion.time_in_Gyr and galpy.util.conversion.force_in_XXX conversion functions [force is actually acceleration in galpy]).
        cinterp : bool, optional
            If True (default), build cubic-spline interpolations of the time-dependent inputs (``a0``, ``x0``, ``v0``, ``Omega``, with ``Omegadot`` taken as the spline derivative of ``Omega``) over the integration's time range and evaluate them in C, instead of calling the supplied functions from C at every integration step; this is typically much faster, especially when the inputs are not ``numba``-compatible. Only affects C orbit integration (the pure-Python integration always uses the exact functions) and is not supported for surface-of-section integration (use ``cinterp=False``).
        cinterp_n : int, optional
            Number of grid points used for the C interpolation over the integration time range (default: 3000); only used when ``cinterp=True``.
        ro : float, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2022-03-02 - Started - Bovy (UofT)
        - 2022-03-26 - Generalized Omega to any function of time - Bovy (UofT)
        """
        DissipativeForce.__init__(self, amp=amp, ro=ro, vo=vo)
        # cinterp: build C-side spline interpolations of the time functions on
        # the fly at integration time (see _parse_pot); only affects C integration
        self._cinterp = cinterp
        self._cinterp_n = cinterp_n
        # Single-entry cache of the on-the-fly interpolation tables, keyed by the
        # integration time range; see _parse_noninertial_frame_force.
        self._cinterp_table_cache = None
        self._rot_acc = not Omega is None
        self._omegaz_only = len(numpy.atleast_1d(Omega)) == 1
        self._const_freq = Omegadot is None
        # Omega and Omegadot must be the same kind: both functions of time or
        # both constants. A mismatch (e.g. a function Omega with a constant
        # Omegadot) is almost always a mistake, so reject it explicitly.
        if (
            self._rot_acc
            and Omegadot is not None
            and _is_freq_func(Omegadot) != _is_freq_func(Omega)
        ):
            raise ValueError(
                "Omega and Omegadot must be the same kind: both functions of "
                "time or both constants/Quantities (Omegadot may also be omitted)"
            )
        if (self._omegaz_only and callable(Omega)) or (
            not self._omegaz_only and callable(Omega[0])
        ):
            self._Omega_as_func = True
            self._Omega = Omega
            if Omegadot is None:
                # Omega varies in time but its derivative was not provided:
                # derive it (so the Euler force term is included). cinterp uses
                # the analytic spline derivative of Omega; the direct-evaluation
                # path uses finite differences (see _time_derivative).
                warnings.warn(
                    "NonInertialFrameForce: Omega is a function of time but "
                    "Omegadot was not provided; Omegadot will be derived as the "
                    "time derivative of Omega (analytically from the spline when "
                    "cinterp=True, by finite differences otherwise)",
                    galpyWarning,
                )
                self._const_freq = False
                if self._omegaz_only:
                    self._Omegadot = _time_derivative(Omega)
                else:
                    self._Omegadot = [_time_derivative(o) for o in Omega]
            else:
                self._Omegadot = Omegadot
            # Convenient access in Python
            if not self._omegaz_only:
                self._Omega_py = lambda t: numpy.array(
                    [self._Omega[0](t), self._Omega[1](t), self._Omega[2](t)]
                )
                self._Omegadot_py = lambda t: numpy.array(
                    [self._Omegadot[0](t), self._Omegadot[1](t), self._Omegadot[2](t)]
                )
            else:
                self._Omega_py = self._Omega
                self._Omegadot_py = self._Omegadot
        else:
            self._Omega_as_func = False
            self._Omega = conversion.parse_frequency(Omega, ro=self._ro, vo=self._vo)
            self._Omegadot = conversion.parse_frequency(
                Omegadot, ro=self._ro, vo=self._vo
            )
        self._lin_acc = not (a0 is None)
        if self._lin_acc:
            if not callable(a0[0]):
                # Constant a0: wrapped as (trivial) functions of time. Tracked
                # separately so cinterp does not bother interpolating a constant
                # (and so constant-a0 SOS integration is not rejected; see
                # _parse_noninertial_frame_force).
                self._a0_as_func = False
                self._a0 = [
                    lambda t, copy=a0[0]: copy,
                    lambda t, copy=a0[1]: copy,
                    lambda t, copy=a0[2]: copy,
                ]
            else:
                self._a0_as_func = True
                self._a0 = a0
            # Convenient access in Python
            self._a0_py = lambda t: [self._a0[0](t), self._a0[1](t), self._a0[2](t)]
        if self._lin_acc and self._rot_acc:
            if x0 is None or v0 is None:
                raise ValueError(
                    "x0 and v0 (the position and velocity of the non-inertial "
                    "frame's origin) must be provided when the frame both rotates "
                    "(Omega) and accelerates (a0)"
                )
            self._x0 = x0
            self._v0 = v0
            # Convenient access in Python
            self._x0_py = lambda t: numpy.array(
                [self._x0[0](t), self._x0[1](t), self._x0[2](t)]
            )
            self._v0_py = lambda t: numpy.array(
                [self._v0[0](t), self._v0[1](t), self._v0[2](t)]
            )
        # Useful derived quantities
        self._Omega2 = (
            numpy.linalg.norm(self._Omega) ** 2.0
            if self._rot_acc and not self._Omega_as_func
            else 0.0
        )
        if not self._omegaz_only and not self._Omega_as_func:
            self._Omega_for_cross = numpy.array(
                [
                    [0.0, -self._Omega[2], self._Omega[1]],
                    [self._Omega[2], 0.0, -self._Omega[0]],
                    [-self._Omega[1], self._Omega[0], 0.0],
                ]
            )
            if not self._const_freq:
                self._Omegadot_for_cross = numpy.array(
                    [
                        [0.0, -self._Omegadot[2], self._Omegadot[1]],
                        [self._Omegadot[2], 0.0, -self._Omegadot[0]],
                        [-self._Omegadot[1], self._Omegadot[0], 0.0],
                    ]
                )
        self._force_hash = None
        self.hasC = True
        # The rectangular force Jacobian (dF/dx, dF/dv) of the frame force is
        # wired in C for the 3D variational equations (integrate_dxdv): the
        # force is linear in position and velocity, so the Jacobian is exact
        # for EVERY supported configuration (scalar/vector Omega, constant or
        # time-dependent through tfuncs or the cinterp splines, Omegadot,
        # x0/v0/a0 translation terms -- the latter contribute zero).
        self.hasC_dxdv3d = True
        return None

    def _force(self, R, z, phi, t, v):
        """Internal function that computes the fictitious forces in rectangular
        coordinates"""
        xp = get_namespace(R, z, phi, t, v[0], v[1], v[2])
        numpy_input = xp is numpy
        if numpy_input:
            # Single-entry input cache: numpy path only (a backend trace cannot
            # be md5-hashed, and caching eager backend arrays defeats the trace)
            new_hash = hashlib.md5(
                numpy.array([R, phi, z, v[0], v[1], v[2], t])
            ).hexdigest()
            if new_hash == self._force_hash:
                return self._cached_force
        x, y, z = coords.cyl_to_rect(R, phi, z)
        vx, vy, vz = coords.cyl_to_rect_vec(v[0], v[1], v[2], phi)
        # Bring the coordinate/velocity inputs onto the active backend (numpy =
        # byte-identical pass-through); stored numpy constants are anchored on a
        # coerced coordinate (ref) via as_backend_constant.
        x, y, z, vx, vy, vz = coerce_coords(xp, x, y, z, vx, vy, vz)
        ref = x

        def _anchor(c):
            # keep a backend (grad-carrying) comp; anchor a scalar/numpy const on the
            # coords' device/dtype via the shared helper (device-safe on GPU, where an
            # all-scalar _vec would otherwise land on CPU and mismatch the backend force)
            return c if is_backend_array(c) else as_backend_constant(xp, c, ref)

        def _vec(comps):
            # 1D vector from (possibly backend, grad-carrying) scalar comps;
            # stack (not asarray) preserves the autograd graph
            return (
                numpy.asarray(comps)
                if numpy_input
                else xp.stack([_anchor(c) for c in comps])
            )

        def _mat(rows):
            # 3x3 matrix from scalar comps (stack preserves autograd graph)
            return (
                numpy.asarray(rows)
                if numpy_input
                else xp.stack([xp.stack([_anchor(c) for c in r]) for r in rows])
            )

        def _mv(mat, vecarr):  # matrix (3x3) times vector (3): torch.dot is 1D-only
            return numpy.dot(mat, vecarr) if numpy_input else xp.matmul(mat, vecarr)

        force = numpy.zeros(3) if numpy_input else _vec([0.0, 0.0, 0.0])
        if self._rot_acc:
            if self._const_freq:
                tOmega = self._Omega
                tOmega2 = self._Omega2
            elif self._Omega_as_func:
                tOmega = (
                    self._Omega_py(t)
                    if self._omegaz_only
                    else _vec([self._Omega[0](t), self._Omega[1](t), self._Omega[2](t)])
                )
                tOmega2 = xp.linalg.norm(tOmega) ** 2.0
            else:
                tOmega = (
                    as_backend_constant(xp, self._Omega, ref)
                    + as_backend_constant(xp, self._Omegadot, ref) * t
                )
                tOmega2 = xp.linalg.norm(tOmega) ** 2.0
            if self._omegaz_only:
                force += -2.0 * tOmega * _vec([-vy, vx, 0.0]) + tOmega2 * _vec(
                    [x, y, 0.0]
                )
                if self._lin_acc:
                    force += -2.0 * tOmega * _vec(
                        [-self._v0[1](t), self._v0[0](t), 0.0]
                    ) + tOmega2 * _vec([self._x0[0](t), self._x0[1](t), 0.0])
                if not self._const_freq:
                    if self._Omega_as_func:
                        force -= self._Omegadot_py(t) * _vec([-y, x, 0.0])
                        if self._lin_acc:
                            force -= self._Omegadot_py(t) * _vec(
                                [-self._x0[1](t), self._x0[0](t), 0.0]
                            )
                    else:
                        force -= self._Omegadot * _vec([-y, x, 0.0])
                        if self._lin_acc:
                            force -= self._Omegadot * _vec(
                                [-self._x0[1](t), self._x0[0](t), 0.0]
                            )
            else:
                if self._Omega_as_func:
                    Omega_for_cross = _mat(
                        [
                            [0.0, -self._Omega[2](t), self._Omega[1](t)],
                            [self._Omega[2](t), 0.0, -self._Omega[0](t)],
                            [-self._Omega[1](t), self._Omega[0](t), 0.0],
                        ]
                    )
                    if not self._const_freq:
                        Omegadot_for_cross = _mat(
                            [
                                [0.0, -self._Omegadot[2](t), self._Omegadot[1](t)],
                                [self._Omegadot[2](t), 0.0, -self._Omegadot[0](t)],
                                [-self._Omegadot[1](t), self._Omegadot[0](t), 0.0],
                            ]
                        )
                else:
                    Omega_for_cross = as_backend_constant(
                        xp, self._Omega_for_cross, ref
                    )
                    if not self._const_freq:
                        Omegadot_for_cross = as_backend_constant(
                            xp, self._Omegadot_for_cross, ref
                        )
                if not numpy_input and not is_backend_array(tOmega):
                    # anchor the stored numpy constant
                    tOmega = as_backend_constant(xp, tOmega, ref)
                xyz = _vec([x, y, z])
                force += (
                    -2.0 * _mv(Omega_for_cross, _vec([vx, vy, vz]))
                    + tOmega2 * xyz
                    - tOmega * xp.dot(tOmega, xyz)
                )
                if self._lin_acc:
                    v0 = _vec([self._v0[0](t), self._v0[1](t), self._v0[2](t)])
                    x0 = _vec([self._x0[0](t), self._x0[1](t), self._x0[2](t)])
                    force += (
                        -2.0 * _mv(Omega_for_cross, v0)
                        + tOmega2 * x0
                        - tOmega * xp.dot(tOmega, x0)
                    )
                if not self._const_freq:
                    if (
                        not self._Omega_as_func
                    ):  # Already included above when Omega=func
                        force -= 2.0 * t * _mv(Omegadot_for_cross, _vec([vx, vy, vz]))
                    force -= _mv(Omegadot_for_cross, xyz)
                    if self._lin_acc:
                        if not self._Omega_as_func:
                            force -= 2.0 * t * _mv(Omegadot_for_cross, v0)
                        force -= _mv(Omegadot_for_cross, x0)
        if self._lin_acc:
            force -= _vec([self._a0[0](t), self._a0[1](t), self._a0[2](t)])
        if numpy_input:
            self._force_hash = new_hash
            self._cached_force = force
        return force

    def _Rforce(self, R, z, phi=0.0, t=0.0, v=None):
        force = self._force(R, z, phi, t, v)
        xp = get_namespace(phi, force[0])
        # phi may arrive as a bare python/numpy scalar while force is a backend
        # array (the DissipativeForce path skips the input-coercion gate); torch
        # rejects cos() on a python float, so anchor phi on the force namespace.
        phi = phi if is_backend_array(phi) else as_backend_constant(xp, phi, force[0])
        return xp.cos(phi) * force[0] + xp.sin(phi) * force[1]

    def _phitorque(self, R, z, phi=0.0, t=0.0, v=None):
        force = self._force(R, z, phi, t, v)
        xp = get_namespace(phi, force[0])
        phi = phi if is_backend_array(phi) else as_backend_constant(xp, phi, force[0])
        return R * (-xp.sin(phi) * force[0] + xp.cos(phi) * force[1])

    def _zforce(self, R, z, phi=0.0, t=0.0, v=None):
        return self._force(R, z, phi, t, v)[2]
