###############################################################################
#   NonInertialFrameForce: Class that implements the fictitious forces
#                          present when integrating orbits in a non-intertial
#                          frame
###############################################################################
import hashlib

import numpy
import numpy.linalg

from ..util import conversion, coords
from .DissipativeForce import DissipativeForce


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

    The functions of time are passed to the C code for fast orbit integration
    by attempting to build fast ``numba`` versions of them. Significant
    speed-ups can therefore be obtained by making sure that the provided
    functions can be turned into ``nopython=True`` ``numba`` functions (try
    running ``numba.njit`` on them and then evaluate them to check).
    """

    def __init__(
        self,
        amp=1.0,
        Omega=None,
        Omegadot=None,
        x0=None,
        v0=None,
        a0=None,
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
            Time derivative of the angular frequency of the non-intertial frame's rotation. Format should match Omega input ([list of] function[s] when Omega is one, number/list if Omega is a number/list; when a function, need to take input time in internal units and output the frequency derivative in internal units; see galpy.util.conversion.time_in_Gyr and galpy.util.conversion.freq_in_XXX conversion functions).
        x0 : list of callables, optional
            Position vector x_0 (cartesian) of the center of mass of the non-intertial frame (see definition in the class documentation); list of functions [x_0x,x_0y,x_0z]; only necessary when considering both rotation and center-of-mass acceleration of the inertial frame (functions need to take input time in internal units and output the position in internal units; see galpy.util.conversion.time_in_Gyr and divided physical positions by the `ro` parameter in kpc).
        v0 : list of callables, optional
            Velocity vector v_0 (cartesian) of the center of mass of the non-intertial frame (see definition in the class documentation); list of functions [v_0x,v_0y,v_0z]; only necessary when considering both rotation and center-of-mass acceleration of the inertial frame (functions need to take input time in internal units and output the velocity in internal units; see galpy.util.conversion.time_in_Gyr and divided physical positions by the `vo` parameter in km/s).
        a0 : float or list of callables, optional
            Acceleration vector a_0 (cartesian) of the center of mass of the non-intertial frame (see definition in the class documentation); constant or a list of functions [a_0x,a_0y, a_0z] (functions need to take input time in internal units and output the acceleration in internal units; see galpy.util.conversion.time_in_Gyr and galpy.util.conversion.force_in_XXX conversion functions [force is actually acceleration in galpy]).
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
        self._rot_acc = not Omega is None
        self._omegaz_only = len(numpy.atleast_1d(Omega)) == 1
        self._const_freq = Omegadot is None
        if (self._omegaz_only and callable(Omega)) or (
            not self._omegaz_only and callable(Omega[0])
        ):
            self._Omega_as_func = True
            self._Omega = Omega
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
                self._a0 = [
                    lambda t, copy=a0[0]: copy,
                    lambda t, copy=a0[1]: copy,
                    lambda t, copy=a0[2]: copy,
                ]
            else:
                self._a0 = a0
            # Convenient access in Python
            self._a0_py = lambda t: [self._a0[0](t), self._a0[1](t), self._a0[2](t)]
        if self._lin_acc and self._rot_acc:
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
        return None

    def _force(self, R, z, phi, t, v):
        """Internal function that computes the fictitious forces in rectangular
        coordinates"""
        new_hash = hashlib.md5(
            numpy.array([R, phi, z, v[0], v[1], v[2], t])
        ).hexdigest()
        if new_hash == self._force_hash:
            return self._cached_force
        x, y, z = coords.cyl_to_rect(R, phi, z)
        vx, vy, vz = coords.cyl_to_rect_vec(v[0], v[1], v[2], phi)
        force = numpy.zeros(3)
        if self._rot_acc:
            if self._const_freq:
                tOmega = self._Omega
                tOmega2 = self._Omega2
            elif self._Omega_as_func:
                tOmega = self._Omega_py(t)
                tOmega2 = numpy.linalg.norm(tOmega) ** 2.0
            else:
                tOmega = self._Omega + self._Omegadot * t
                tOmega2 = numpy.linalg.norm(tOmega) ** 2.0
            if self._omegaz_only:
                force += -2.0 * tOmega * numpy.array(
                    [-vy, vx, 0.0]
                ) + tOmega2 * numpy.array([x, y, 0.0])
                if self._lin_acc:
                    force += -2.0 * tOmega * numpy.array(
                        [-self._v0[1](t), self._v0[0](t), 0.0]
                    ) + tOmega2 * numpy.array([self._x0[0](t), self._x0[1](t), 0.0])
                if not self._const_freq:
                    if self._Omega_as_func:
                        force -= self._Omegadot_py(t) * numpy.array([-y, x, 0.0])
                        if self._lin_acc:
                            force -= self._Omegadot_py(t) * numpy.array(
                                [-self._x0[1](t), self._x0[0](t), 0.0]
                            )
                    else:
                        force -= self._Omegadot * numpy.array([-y, x, 0.0])
                        if self._lin_acc:
                            force -= self._Omegadot * numpy.array(
                                [-self._x0[1](t), self._x0[0](t), 0.0]
                            )
            else:
                if self._Omega_as_func:
                    self._Omega_for_cross = numpy.array(
                        [
                            [0.0, -self._Omega[2](t), self._Omega[1](t)],
                            [self._Omega[2](t), 0.0, -self._Omega[0](t)],
                            [-self._Omega[1](t), self._Omega[0](t), 0.0],
                        ]
                    )
                    if not self._const_freq:
                        self._Omegadot_for_cross = numpy.array(
                            [
                                [0.0, -self._Omegadot[2](t), self._Omegadot[1](t)],
                                [self._Omegadot[2](t), 0.0, -self._Omegadot[0](t)],
                                [-self._Omegadot[1](t), self._Omegadot[0](t), 0.0],
                            ]
                        )
                force += (
                    -2.0 * numpy.dot(self._Omega_for_cross, numpy.array([vx, vy, vz]))
                    + tOmega2 * numpy.array([x, y, z])
                    - tOmega * numpy.dot(tOmega, numpy.array([x, y, z]))
                )
                if self._lin_acc:
                    force += (
                        -2.0 * numpy.dot(self._Omega_for_cross, self._v0_py(t))
                        + tOmega2 * self._x0_py(t)
                        - tOmega * numpy.dot(tOmega, self._x0_py(t))
                    )
                if not self._const_freq:
                    if (
                        not self._Omega_as_func
                    ):  # Already included above when Omega=func
                        force -= (
                            2.0
                            * t
                            * numpy.dot(
                                self._Omegadot_for_cross, numpy.array([vx, vy, vz])
                            )
                        )
                    force -= numpy.dot(self._Omegadot_for_cross, numpy.array([x, y, z]))
                    if self._lin_acc:
                        if not self._Omega_as_func:
                            force -= (
                                2.0
                                * t
                                * numpy.dot(self._Omegadot_for_cross, self._v0_py(t))
                            )
                        force -= numpy.dot(self._Omegadot_for_cross, self._x0_py(t))
        if self._lin_acc:
            force -= self._a0_py(t)
        self._force_hash = new_hash
        self._cached_force = force
        return force

    def _Rforce(self, R, z, phi=0.0, t=0.0, v=None):
        force = self._force(R, z, phi, t, v)
        return numpy.cos(phi) * force[0] + numpy.sin(phi) * force[1]

    def _phitorque(self, R, z, phi=0.0, t=0.0, v=None):
        force = self._force(R, z, phi, t, v)
        return R * (-numpy.sin(phi) * force[0] + numpy.cos(phi) * force[1])

    def _zforce(self, R, z, phi=0.0, t=0.0, v=None):
        return self._force(R, z, phi, t, v)[2]
