###############################################################################
#   EllipsoidalPotential.py: base class for potentials corresponding to
#                            density profiles that are stratified on
#                            ellipsoids:
#
#                            \rho(x,y,z) ~ \rho(m)
#
#                            with m^2 = x^2+y^2/b^2+z^2/c^2
#
###############################################################################
import hashlib

import numpy
from scipy import integrate

from ..util import _rotate_to_arbitrary_vector, conversion, coords
from .Potential import Potential, check_potential_inputs_not_arrays


class EllipsoidalPotential(Potential):
    """Base class for potentials corresponding to density profiles that are stratified on ellipsoids:

    .. math::

        \\rho(x,y,z) \\equiv \\rho(m^2)

    where :math:`m^2 = x^2+y^2/b^2+z^2/c^2`. Note that :math:`b` and :math:`c` are defined to be the axis ratios (rather than using :math:`m^2 = x^2/a^2+y^2/b^2+z^2/c^2` as is common).

    Implement a specific density distribution with this form by inheriting from this class and defining functions ``_mdens(self,m)`` (the density as a function of ``m``), ``_mdens_deriv(self,m)`` (the derivative of the density as a function of ``m``), and ``_psi(self,m)``, which is:

    .. math::

        \\psi(m) = -\\int_{m^2}^\\infty d m^2 \\rho(m^2)

    See PerfectEllipsoidPotential for an example and `Merritt & Fridman (1996) <http://adsabs.harvard.edu/abs/1996ApJ...460..136M>`_ for the formalism.
    """

    def __init__(
        self,
        amp=1.0,
        b=1.0,
        c=1.0,
        zvec=None,
        pa=None,
        glorder=50,
        ro=None,
        vo=None,
        amp_units=None,
    ):
        """
        Initialize an ellipsoidal potential.

        Parameters
        ----------
        amp : float or Quantity, optional
            Amplitude to be applied to the potential (default: 1); can be a Quantity with units that depend on the specific spheroidal potential.
        b : float, optional
            y-to-x axis ratio of the density.
        c : float, optional
            z-to-x axis ratio of the density.
        zvec : numpy.ndarray, optional
            If set, a unit vector that corresponds to the z axis.
        pa : float or Quantity, optional
            If set, the position angle of the x axis (rad or Quantity).
        glorder : int, optional
            If set, compute the relevant force and potential integrals with Gaussian quadrature of this order.
        ro : float, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float, optional
            Velocity scale for translation into internal units (default from configuration file).
        amp_units : str, optional
            Type of units that amp should have if it has units (passed to Potential.__init__).

        Notes
        -----
        - 2018-08-06 - Started - Bovy (UofT)

        """
        Potential.__init__(self, amp=amp, ro=ro, vo=vo, amp_units=amp_units)
        # Setup axis ratios
        self._b = b
        self._c = c
        self._b2 = self._b**2.0
        self._c2 = self._c**2.0
        self._force_hash = None
        # Setup rotation
        self._setup_zvec_pa(zvec, pa)
        # Setup integration
        self._setup_gl(glorder)
        if not self._aligned or numpy.fabs(self._b - 1.0) > 10.0**-10.0:
            self.isNonAxi = True
        return None

    def _setup_zvec_pa(self, zvec, pa):
        if not pa is None:
            pa = conversion.parse_angle(pa)
        if zvec is None and (pa is None or numpy.fabs(pa) < 10.0**-10.0):
            self._aligned = True
        else:
            self._aligned = False
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
                if not isinstance(zvec, numpy.ndarray):
                    zvec = numpy.array(zvec)
                zvec /= numpy.sqrt(numpy.sum(zvec**2.0))
                zvec_rot = _rotate_to_arbitrary_vector(
                    numpy.array([[0.0, 0.0, 1.0]]), zvec, inv=True
                )[0]
            else:
                zvec_rot = numpy.eye(3)
            self._rot = numpy.dot(pa_rot, zvec_rot)
        return None

    def _setup_gl(self, glorder):
        self._glorder = glorder
        if self._glorder is None:
            self._glx, self._glw = None, None
        else:
            self._glx, self._glw = numpy.polynomial.legendre.leggauss(self._glorder)
            # Interval change
            self._glx = 0.5 * self._glx + 0.5
            self._glw *= 0.5
        return None

    @check_potential_inputs_not_arrays
    def _evaluate(self, R, z, phi=0.0, t=0.0):
        if not self.isNonAxi:
            phi = 0.0
        x, y, z = coords.cyl_to_rect(R, phi, z)
        if numpy.isinf(R):
            y = 0.0
        if self._aligned:
            return self._evaluate_xyz(x, y, z)
        else:
            xyzp = numpy.dot(self._rot, numpy.array([x, y, z]))
            return self._evaluate_xyz(xyzp[0], xyzp[1], xyzp[2])

    def _evaluate_xyz(self, x, y, z):
        """Evaluation of the potential as a function of (x,y,z) in the
        aligned coordinate frame"""
        return (
            2.0
            * numpy.pi
            * self._b
            * self._c
            * _potInt(
                x, y, z, self._psi, self._b2, self._c2, glx=self._glx, glw=self._glw
            )
        )

    @check_potential_inputs_not_arrays
    def _Rforce(self, R, z, phi=0.0, t=0.0):
        if not self.isNonAxi:
            phi = 0.0
        x, y, z = coords.cyl_to_rect(R, phi, z)
        # Compute all rectangular forces
        new_hash = hashlib.md5(numpy.array([x, y, z])).hexdigest()
        if new_hash == self._force_hash:
            Fx = self._cached_Fx
            Fy = self._cached_Fy
            Fz = self._cached_Fz
        else:
            if self._aligned:
                xp, yp, zp = x, y, z
            else:
                xyzp = numpy.dot(self._rot, numpy.array([x, y, z]))
                xp, yp, zp = xyzp[0], xyzp[1], xyzp[2]
            Fx = self._force_xyz(xp, yp, zp, 0)
            Fy = self._force_xyz(xp, yp, zp, 1)
            Fz = self._force_xyz(xp, yp, zp, 2)
            self._force_hash = new_hash
            self._cached_Fx = Fx
            self._cached_Fy = Fy
            self._cached_Fz = Fz
        if not self._aligned:
            Fxyz = numpy.dot(self._rot.T, numpy.array([Fx, Fy, Fz]))
            Fx, Fy = Fxyz[0], Fxyz[1]
        return numpy.cos(phi) * Fx + numpy.sin(phi) * Fy

    @check_potential_inputs_not_arrays
    def _phitorque(self, R, z, phi=0.0, t=0.0):
        if not self.isNonAxi:
            phi = 0.0
        x, y, z = coords.cyl_to_rect(R, phi, z)
        # Compute all rectangular forces
        new_hash = hashlib.md5(numpy.array([x, y, z])).hexdigest()
        if new_hash == self._force_hash:
            Fx = self._cached_Fx
            Fy = self._cached_Fy
            Fz = self._cached_Fz
        else:
            if self._aligned:
                xp, yp, zp = x, y, z
            else:
                xyzp = numpy.dot(self._rot, numpy.array([x, y, z]))
                xp, yp, zp = xyzp[0], xyzp[1], xyzp[2]
            Fx = self._force_xyz(xp, yp, zp, 0)
            Fy = self._force_xyz(xp, yp, zp, 1)
            Fz = self._force_xyz(xp, yp, zp, 2)
            self._force_hash = new_hash
            self._cached_Fx = Fx
            self._cached_Fy = Fy
            self._cached_Fz = Fz
        if not self._aligned:
            Fxyz = numpy.dot(self._rot.T, numpy.array([Fx, Fy, Fz]))
            Fx, Fy = Fxyz[0], Fxyz[1]
        return R * (-numpy.sin(phi) * Fx + numpy.cos(phi) * Fy)

    @check_potential_inputs_not_arrays
    def _zforce(self, R, z, phi=0.0, t=0.0):
        if not self.isNonAxi:
            phi = 0.0
        x, y, z = coords.cyl_to_rect(R, phi, z)
        # Compute all rectangular forces
        new_hash = hashlib.md5(numpy.array([x, y, z])).hexdigest()
        if new_hash == self._force_hash:
            Fx = self._cached_Fx
            Fy = self._cached_Fy
            Fz = self._cached_Fz
        else:
            if self._aligned:
                xp, yp, zp = x, y, z
            else:
                xyzp = numpy.dot(self._rot, numpy.array([x, y, z]))
                xp, yp, zp = xyzp[0], xyzp[1], xyzp[2]
            Fx = self._force_xyz(xp, yp, zp, 0)
            Fy = self._force_xyz(xp, yp, zp, 1)
            Fz = self._force_xyz(xp, yp, zp, 2)
            self._force_hash = new_hash
            self._cached_Fx = Fx
            self._cached_Fy = Fy
            self._cached_Fz = Fz
        if not self._aligned:
            Fxyz = numpy.dot(self._rot.T, numpy.array([Fx, Fy, Fz]))
            Fz = Fxyz[2]
        return Fz

    def _force_xyz(self, x, y, z, i):
        """Evaluation of the i-th force component as a function of (x,y,z)"""
        return (
            -4.0
            * numpy.pi
            * self._b
            * self._c
            * _forceInt(
                x,
                y,
                z,
                lambda m: self._mdens(m),
                self._b2,
                self._c2,
                i,
                glx=self._glx,
                glw=self._glw,
            )
        )

    @check_potential_inputs_not_arrays
    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        if not self.isNonAxi:
            phi = 0.0
        x, y, z = coords.cyl_to_rect(R, phi, z)
        if not self._aligned:
            raise NotImplementedError(
                "2nd potential derivatives of TwoPowerTriaxialPotential not implemented for rotated coordinated frames (non-trivial zvec and pa); use RotateAndTiltWrapperPotential for this functionality instead"
            )
        phixx = self._2ndderiv_xyz(x, y, z, 0, 0)
        phixy = self._2ndderiv_xyz(x, y, z, 0, 1)
        phiyy = self._2ndderiv_xyz(x, y, z, 1, 1)
        return (
            numpy.cos(phi) ** 2.0 * phixx
            + numpy.sin(phi) ** 2.0 * phiyy
            + 2.0 * numpy.cos(phi) * numpy.sin(phi) * phixy
        )

    @check_potential_inputs_not_arrays
    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        if not self.isNonAxi:
            phi = 0.0
        x, y, z = coords.cyl_to_rect(R, phi, z)
        if not self._aligned:
            raise NotImplementedError(
                "2nd potential derivatives of TwoPowerTriaxialPotential not implemented for rotated coordinated frames (non-trivial zvec and pa; use RotateAndTiltWrapperPotential for this functionality instead)"
            )
        phixz = self._2ndderiv_xyz(x, y, z, 0, 2)
        phiyz = self._2ndderiv_xyz(x, y, z, 1, 2)
        return numpy.cos(phi) * phixz + numpy.sin(phi) * phiyz

    @check_potential_inputs_not_arrays
    def _z2deriv(self, R, z, phi=0.0, t=0.0):
        if not self.isNonAxi:
            phi = 0.0
        x, y, z = coords.cyl_to_rect(R, phi, z)
        if not self._aligned:
            raise NotImplementedError(
                "2nd potential derivatives of TwoPowerTriaxialPotential not implemented for rotated coordinated frames (non-trivial zvec and pa; use RotateAndTiltWrapperPotential for this functionality instead)"
            )
        return self._2ndderiv_xyz(x, y, z, 2, 2)

    @check_potential_inputs_not_arrays
    def _phi2deriv(self, R, z, phi=0.0, t=0.0):
        if not self.isNonAxi:
            phi = 0.0
        x, y, z = coords.cyl_to_rect(R, phi, z)
        if not self._aligned:
            raise NotImplementedError(
                "2nd potential derivatives of TwoPowerTriaxialPotential not implemented for rotated coordinated frames (non-trivial zvec and pa; use RotateAndTiltWrapperPotential for this functionality instead)"
            )
        Fx = self._force_xyz(x, y, z, 0)
        Fy = self._force_xyz(x, y, z, 1)
        phixx = self._2ndderiv_xyz(x, y, z, 0, 0)
        phixy = self._2ndderiv_xyz(x, y, z, 0, 1)
        phiyy = self._2ndderiv_xyz(x, y, z, 1, 1)
        return R**2.0 * (
            numpy.sin(phi) ** 2.0 * phixx
            + numpy.cos(phi) ** 2.0 * phiyy
            - 2.0 * numpy.cos(phi) * numpy.sin(phi) * phixy
        ) + R * (numpy.cos(phi) * Fx + numpy.sin(phi) * Fy)

    @check_potential_inputs_not_arrays
    def _Rphideriv(self, R, z, phi=0.0, t=0.0):
        if not self.isNonAxi:
            phi = 0.0
        x, y, z = coords.cyl_to_rect(R, phi, z)
        if not self._aligned:
            raise NotImplementedError(
                "2nd potential derivatives of TwoPowerTriaxialPotential not implemented for rotated coordinated frames (non-trivial zvec and pa; use RotateAndTiltWrapperPotential for this functionality instead)"
            )
        Fx = self._force_xyz(x, y, z, 0)
        Fy = self._force_xyz(x, y, z, 1)
        phixx = self._2ndderiv_xyz(x, y, z, 0, 0)
        phixy = self._2ndderiv_xyz(x, y, z, 0, 1)
        phiyy = self._2ndderiv_xyz(x, y, z, 1, 1)
        return (
            R * numpy.cos(phi) * numpy.sin(phi) * (phiyy - phixx)
            + R * numpy.cos(2.0 * phi) * phixy
            + numpy.sin(phi) * Fx
            - numpy.cos(phi) * Fy
        )

    @check_potential_inputs_not_arrays
    def _phizderiv(self, R, z, phi=0.0, t=0.0):
        if not self.isNonAxi:
            phi = 0.0
        x, y, z = coords.cyl_to_rect(R, phi, z)
        if not self._aligned:
            raise NotImplementedError(
                "2nd potential derivatives of TwoPowerTriaxialPotential not implemented for rotated coordinated frames (non-trivial zvec and pa; use RotateAndTiltWrapperPotential for this functionality instead)"
            )
        phixz = self._2ndderiv_xyz(x, y, z, 0, 2)
        phiyz = self._2ndderiv_xyz(x, y, z, 1, 2)
        return R * (numpy.cos(phi) * phiyz - numpy.sin(phi) * phixz)

    def _2ndderiv_xyz(self, x, y, z, i, j):
        """General 2nd derivative of the potential as a function of (x,y,z)
        in the aligned coordinate frame"""
        return (
            4.0
            * numpy.pi
            * self._b
            * self._c
            * _2ndDerivInt(
                x,
                y,
                z,
                lambda m: self._mdens(m),
                lambda m: self._mdens_deriv(m),
                self._b2,
                self._c2,
                i,
                j,
                glx=self._glx,
                glw=self._glw,
            )
        )

    @check_potential_inputs_not_arrays
    def _dens(self, R, z, phi=0.0, t=0.0):
        x, y, z = coords.cyl_to_rect(R, phi, z)
        if self._aligned:
            xp, yp, zp = x, y, z
        else:
            xyzp = numpy.dot(self._rot, numpy.array([x, y, z]))
            xp, yp, zp = xyzp[0], xyzp[1], xyzp[2]
        m = numpy.sqrt(xp**2.0 + yp**2.0 / self._b2 + zp**2.0 / self._c2)
        return self._mdens(m)

    def _mass(self, R, z=None, t=0.0):
        if not z is None:
            raise AttributeError  # Hack to fall back to general
        return (
            4.0
            * numpy.pi
            * self._b
            * self._c
            * integrate.quad(lambda m: m**2.0 * self._mdens(m), 0, R)[0]
        )

    def OmegaP(self):
        return 0.0


def _potInt(x, y, z, psi, b2, c2, glx=None, glw=None):
    r"""int_0^\infty [psi(m)-psi(\infy)]/sqrt([1+tau]x[b^2+tau]x[c^2+tau])dtau"""

    def integrand(s):
        t = 1 / s**2.0 - 1.0
        return psi(
            numpy.sqrt(x**2.0 / (1.0 + t) + y**2.0 / (b2 + t) + z**2.0 / (c2 + t))
        ) / numpy.sqrt((1.0 + (b2 - 1.0) * s**2.0) * (1.0 + (c2 - 1.0) * s**2.0))

    if glx is None:
        return integrate.quad(integrand, 0.0, 1.0)[0]
    else:
        return numpy.sum(glw * integrand(glx))


def _forceInt(x, y, z, dens, b2, c2, i, glx=None, glw=None):
    """Integral that gives the force in x,y,z"""

    def integrand(s):
        t = 1 / s**2.0 - 1.0
        return (
            dens(numpy.sqrt(x**2.0 / (1.0 + t) + y**2.0 / (b2 + t) + z**2.0 / (c2 + t)))
            * (
                x / (1.0 + t) * (i == 0)
                + y / (b2 + t) * (i == 1)
                + z / (c2 + t) * (i == 2)
            )
            / numpy.sqrt((1.0 + (b2 - 1.0) * s**2.0) * (1.0 + (c2 - 1.0) * s**2.0))
        )

    if glx is None:
        return integrate.quad(integrand, 0.0, 1.0)[0]
    else:
        return numpy.sum(glw * integrand(glx))


def _2ndDerivInt(x, y, z, dens, densDeriv, b2, c2, i, j, glx=None, glw=None):
    """Integral that gives the 2nd derivative of the potential in x,y,z"""

    def integrand(s):
        t = 1 / s**2.0 - 1.0
        m = numpy.sqrt(x**2.0 / (1.0 + t) + y**2.0 / (b2 + t) + z**2.0 / (c2 + t))
        return (
            densDeriv(m)
            * (
                x / (1.0 + t) * (i == 0)
                + y / (b2 + t) * (i == 1)
                + z / (c2 + t) * (i == 2)
            )
            * (
                x / (1.0 + t) * (j == 0)
                + y / (b2 + t) * (j == 1)
                + z / (c2 + t) * (j == 2)
            )
            / m
            + dens(m)
            * (i == j)
            * (
                1.0 / (1.0 + t) * (i == 0)
                + 1.0 / (b2 + t) * (i == 1)
                + 1.0 / (c2 + t) * (i == 2)
            )
        ) / numpy.sqrt((1.0 + (b2 - 1.0) * s**2.0) * (1.0 + (c2 - 1.0) * s**2.0))

    if glx is None:
        return integrate.quad(integrand, 0.0, 1.0)[0]
    else:
        return numpy.sum(glw * integrand(glx))
