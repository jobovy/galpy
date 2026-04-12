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
from .Potential import Potential


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
        self._2ndderiv_hash = None
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

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        if not self.isNonAxi:
            phi = 0.0
        x, y, z = coords.cyl_to_rect(R, phi, z)
        if numpy.ndim(R) == 0:
            if numpy.isinf(R):
                y = 0.0
        else:
            y = numpy.where(numpy.isinf(R), 0.0, y)
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

    def _compute_forces(self, x, y, z):
        """Compute and cache all three force components in the aligned frame"""
        new_hash = hashlib.md5(numpy.array([x, y, z])).hexdigest()
        if new_hash != self._force_hash:
            if self._aligned:
                xp, yp, zp = x, y, z
            else:
                xyzp = numpy.dot(self._rot, numpy.array([x, y, z]))
                xp, yp, zp = xyzp[0], xyzp[1], xyzp[2]
            prefac = -4.0 * numpy.pi * self._b * self._c
            Fx, Fy, Fz = _forceInt_all(
                xp,
                yp,
                zp,
                lambda m: self._mdens(m),
                self._b2,
                self._c2,
                glx=self._glx,
                glw=self._glw,
            )
            self._cached_Fx = prefac * Fx
            self._cached_Fy = prefac * Fy
            self._cached_Fz = prefac * Fz
            self._force_hash = new_hash

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        if not self.isNonAxi:
            phi = 0.0
        x, y, z = coords.cyl_to_rect(R, phi, z)
        self._compute_forces(x, y, z)
        Fx = self._cached_Fx
        Fy = self._cached_Fy
        if not self._aligned:
            Fxyz = numpy.dot(self._rot.T, numpy.array([Fx, Fy, self._cached_Fz]))
            Fx, Fy = Fxyz[0], Fxyz[1]
        return numpy.cos(phi) * Fx + numpy.sin(phi) * Fy

    def _phitorque(self, R, z, phi=0.0, t=0.0):
        if not self.isNonAxi:
            phi = 0.0
        x, y, z = coords.cyl_to_rect(R, phi, z)
        self._compute_forces(x, y, z)
        Fx = self._cached_Fx
        Fy = self._cached_Fy
        if not self._aligned:
            Fxyz = numpy.dot(self._rot.T, numpy.array([Fx, Fy, self._cached_Fz]))
            Fx, Fy = Fxyz[0], Fxyz[1]
        return R * (-numpy.sin(phi) * Fx + numpy.cos(phi) * Fy)

    def _zforce(self, R, z, phi=0.0, t=0.0):
        if not self.isNonAxi:
            phi = 0.0
        x, y, z = coords.cyl_to_rect(R, phi, z)
        self._compute_forces(x, y, z)
        Fz = self._cached_Fz
        if not self._aligned:
            Fxyz = numpy.dot(
                self._rot.T,
                numpy.array([self._cached_Fx, self._cached_Fy, Fz]),
            )
            Fz = Fxyz[2]
        return Fz

    def _compute_2ndderivs(self, x, y, z):
        """Compute and cache all six unique 2nd-derivative components in the
        aligned frame"""
        new_hash = hashlib.md5(numpy.array([x, y, z])).hexdigest()
        if new_hash != self._2ndderiv_hash:
            prefac = 4.0 * numpy.pi * self._b * self._c
            xx, xy, xz, yy, yz, zz = _2ndDerivInt_all(
                x,
                y,
                z,
                lambda m: self._mdens(m),
                lambda m: self._mdens_deriv(m),
                self._b2,
                self._c2,
                glx=self._glx,
                glw=self._glw,
            )
            self._cached_2nd_xx = prefac * xx
            self._cached_2nd_xy = prefac * xy
            self._cached_2nd_xz = prefac * xz
            self._cached_2nd_yy = prefac * yy
            self._cached_2nd_yz = prefac * yz
            self._cached_2nd_zz = prefac * zz
            self._2ndderiv_hash = new_hash

    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        if not self.isNonAxi:
            phi = 0.0
        x, y, z = coords.cyl_to_rect(R, phi, z)
        if not self._aligned:
            raise NotImplementedError(
                "2nd potential derivatives of TwoPowerTriaxialPotential not implemented for rotated coordinated frames (non-trivial zvec and pa); use RotateAndTiltWrapperPotential for this functionality instead"
            )
        self._compute_2ndderivs(x, y, z)
        phixx = self._cached_2nd_xx
        phixy = self._cached_2nd_xy
        phiyy = self._cached_2nd_yy
        return (
            numpy.cos(phi) ** 2.0 * phixx
            + numpy.sin(phi) ** 2.0 * phiyy
            + 2.0 * numpy.cos(phi) * numpy.sin(phi) * phixy
        )

    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        if not self.isNonAxi:
            phi = 0.0
        x, y, z = coords.cyl_to_rect(R, phi, z)
        if not self._aligned:
            raise NotImplementedError(
                "2nd potential derivatives of TwoPowerTriaxialPotential not implemented for rotated coordinated frames (non-trivial zvec and pa); use RotateAndTiltWrapperPotential for this functionality instead"
            )
        self._compute_2ndderivs(x, y, z)
        phixz = self._cached_2nd_xz
        phiyz = self._cached_2nd_yz
        return numpy.cos(phi) * phixz + numpy.sin(phi) * phiyz

    def _z2deriv(self, R, z, phi=0.0, t=0.0):
        if not self.isNonAxi:
            phi = 0.0
        x, y, z = coords.cyl_to_rect(R, phi, z)
        if not self._aligned:
            raise NotImplementedError(
                "2nd potential derivatives of TwoPowerTriaxialPotential not implemented for rotated coordinated frames (non-trivial zvec and pa); use RotateAndTiltWrapperPotential for this functionality instead"
            )
        self._compute_2ndderivs(x, y, z)
        return self._cached_2nd_zz

    def _phi2deriv(self, R, z, phi=0.0, t=0.0):
        if not self.isNonAxi:
            phi = 0.0
        x, y, z = coords.cyl_to_rect(R, phi, z)
        if not self._aligned:
            raise NotImplementedError(
                "2nd potential derivatives of TwoPowerTriaxialPotential not implemented for rotated coordinated frames (non-trivial zvec and pa); use RotateAndTiltWrapperPotential for this functionality instead"
            )
        self._compute_forces(x, y, z)
        Fx = self._cached_Fx
        Fy = self._cached_Fy
        self._compute_2ndderivs(x, y, z)
        phixx = self._cached_2nd_xx
        phixy = self._cached_2nd_xy
        phiyy = self._cached_2nd_yy
        return R**2.0 * (
            numpy.sin(phi) ** 2.0 * phixx
            + numpy.cos(phi) ** 2.0 * phiyy
            - 2.0 * numpy.cos(phi) * numpy.sin(phi) * phixy
        ) + R * (numpy.cos(phi) * Fx + numpy.sin(phi) * Fy)

    def _Rphideriv(self, R, z, phi=0.0, t=0.0):
        if not self.isNonAxi:
            phi = 0.0
        x, y, z = coords.cyl_to_rect(R, phi, z)
        if not self._aligned:
            raise NotImplementedError(
                "2nd potential derivatives of TwoPowerTriaxialPotential not implemented for rotated coordinated frames (non-trivial zvec and pa); use RotateAndTiltWrapperPotential for this functionality instead"
            )
        self._compute_forces(x, y, z)
        Fx = self._cached_Fx
        Fy = self._cached_Fy
        self._compute_2ndderivs(x, y, z)
        phixx = self._cached_2nd_xx
        phixy = self._cached_2nd_xy
        phiyy = self._cached_2nd_yy
        return (
            R * numpy.cos(phi) * numpy.sin(phi) * (phiyy - phixx)
            + R * numpy.cos(2.0 * phi) * phixy
            + numpy.sin(phi) * Fx
            - numpy.cos(phi) * Fy
        )

    def _phizderiv(self, R, z, phi=0.0, t=0.0):
        if not self.isNonAxi:
            phi = 0.0
        x, y, z = coords.cyl_to_rect(R, phi, z)
        if not self._aligned:
            raise NotImplementedError(
                "2nd potential derivatives of TwoPowerTriaxialPotential not implemented for rotated coordinated frames (non-trivial zvec and pa); use RotateAndTiltWrapperPotential for this functionality instead"
            )
        self._compute_2ndderivs(x, y, z)
        phixz = self._cached_2nd_xz
        phiyz = self._cached_2nd_yz
        return R * (numpy.cos(phi) * phiyz - numpy.sin(phi) * phixz)

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
    elif numpy.ndim(x) > 0:
        result = numpy.zeros(len(x))
        x2 = x**2
        y2 = y**2
        z2 = z**2
        for k in range(len(glx)):
            s = glx[k]
            t = 1.0 / s**2 - 1.0
            denom = numpy.sqrt((1.0 + (b2 - 1.0) * s**2) * (1.0 + (c2 - 1.0) * s**2))
            m = numpy.sqrt(x2 / (1.0 + t) + y2 / (b2 + t) + z2 / (c2 + t))
            result += glw[k] * psi(m) / denom
        return result
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
    elif numpy.ndim(x) > 0:
        result = numpy.zeros(len(x))
        x2 = x**2
        y2 = y**2
        z2 = z**2
        coord = [x, y, z][i]
        denom_shift = [1.0, b2, c2][i]
        for k in range(len(glx)):
            s = glx[k]
            w = glw[k]
            t = 1.0 / s**2 - 1.0
            m = numpy.sqrt(x2 / (1.0 + t) + y2 / (b2 + t) + z2 / (c2 + t))
            denom = numpy.sqrt((1.0 + (b2 - 1.0) * s**2) * (1.0 + (c2 - 1.0) * s**2))
            result += w * dens(m) * coord / (denom_shift + t) / denom
        return result
    else:
        return numpy.sum(glw * integrand(glx))


def _forceInt_all(x, y, z, dens, b2, c2, glx=None, glw=None):
    """Compute all three force integral components in a single pass."""
    if glx is None:
        return (
            _forceInt(x, y, z, dens, b2, c2, 0),
            _forceInt(x, y, z, dens, b2, c2, 1),
            _forceInt(x, y, z, dens, b2, c2, 2),
        )
    if numpy.ndim(x) > 0:
        n = len(x)
        Fx = numpy.zeros(n)
        Fy = numpy.zeros(n)
        Fz = numpy.zeros(n)
        x2 = x**2
        y2 = y**2
        z2 = z**2
        for k in range(len(glx)):
            s = glx[k]
            w = glw[k]
            t = 1.0 / s**2 - 1.0
            inv1t = 1.0 / (1.0 + t)
            invb2t = 1.0 / (b2 + t)
            invc2t = 1.0 / (c2 + t)
            m = numpy.sqrt(x2 * inv1t + y2 * invb2t + z2 * invc2t)
            denom = numpy.sqrt((1.0 + (b2 - 1.0) * s**2) * (1.0 + (c2 - 1.0) * s**2))
            common = w * dens(m) / denom
            Fx += common * x * inv1t
            Fy += common * y * invb2t
            Fz += common * z * invc2t
        return Fx, Fy, Fz
    else:
        Fx = 0.0
        Fy = 0.0
        Fz = 0.0
        for k in range(len(glx)):
            s = glx[k]
            w = glw[k]
            t = 1.0 / s**2 - 1.0
            inv1t = 1.0 / (1.0 + t)
            invb2t = 1.0 / (b2 + t)
            invc2t = 1.0 / (c2 + t)
            m = numpy.sqrt(x**2 * inv1t + y**2 * invb2t + z**2 * invc2t)
            denom = numpy.sqrt((1.0 + (b2 - 1.0) * s**2) * (1.0 + (c2 - 1.0) * s**2))
            common = w * dens(m) / denom
            Fx += common * x * inv1t
            Fy += common * y * invb2t
            Fz += common * z * invc2t
        return Fx, Fy, Fz


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
    elif numpy.ndim(x) > 0:
        result = numpy.zeros(len(x))
        x2 = x**2
        y2 = y**2
        z2 = z**2
        coord_i = [x, y, z][i]
        coord_j = [x, y, z][j]
        shift_i = [1.0, b2, c2][i]
        shift_j = [1.0, b2, c2][j]
        diag = float(i == j)
        for k in range(len(glx)):
            s = glx[k]
            w = glw[k]
            t = 1.0 / s**2 - 1.0
            inv1t = 1.0 / (1.0 + t)
            invb2t = 1.0 / (b2 + t)
            invc2t = 1.0 / (c2 + t)
            m = numpy.sqrt(x2 * inv1t + y2 * invb2t + z2 * invc2t)
            denom = numpy.sqrt((1.0 + (b2 - 1.0) * s**2) * (1.0 + (c2 - 1.0) * s**2))
            inv_si = 1.0 / (shift_i + t)
            inv_sj = 1.0 / (shift_j + t)
            result += (
                w
                * (
                    densDeriv(m) * coord_i * inv_si * coord_j * inv_sj / m
                    + diag * dens(m) * inv_si
                )
                / denom
            )
        return result
    else:
        return numpy.sum(glw * integrand(glx))


def _2ndDerivInt_all(x, y, z, dens, densDeriv, b2, c2, glx=None, glw=None):
    """Compute all six unique 2nd derivative integrals in a single pass.
    Returns (xx, xy, xz, yy, yz, zz)."""
    if glx is None:
        return (
            _2ndDerivInt(x, y, z, dens, densDeriv, b2, c2, 0, 0),
            _2ndDerivInt(x, y, z, dens, densDeriv, b2, c2, 0, 1),
            _2ndDerivInt(x, y, z, dens, densDeriv, b2, c2, 0, 2),
            _2ndDerivInt(x, y, z, dens, densDeriv, b2, c2, 1, 1),
            _2ndDerivInt(x, y, z, dens, densDeriv, b2, c2, 1, 2),
            _2ndDerivInt(x, y, z, dens, densDeriv, b2, c2, 2, 2),
        )
    if numpy.ndim(x) > 0:
        n = len(x)
        xx = numpy.zeros(n)
        xy = numpy.zeros(n)
        xz = numpy.zeros(n)
        yy = numpy.zeros(n)
        yz = numpy.zeros(n)
        zz = numpy.zeros(n)
        x2 = x**2
        y2 = y**2
        z2 = z**2
        for k in range(len(glx)):
            s = glx[k]
            w = glw[k]
            t = 1.0 / s**2 - 1.0
            inv1t = 1.0 / (1.0 + t)
            invb2t = 1.0 / (b2 + t)
            invc2t = 1.0 / (c2 + t)
            m = numpy.sqrt(x2 * inv1t + y2 * invb2t + z2 * invc2t)
            denom = numpy.sqrt((1.0 + (b2 - 1.0) * s**2) * (1.0 + (c2 - 1.0) * s**2))
            w_over_denom = w / denom
            dens_val = dens(m)
            dderiv_over_m = densDeriv(m) / m
            xi = x * inv1t
            yi = y * invb2t
            zi = z * invc2t
            dd_xi = w_over_denom * dderiv_over_m * xi
            dd_yi = w_over_denom * dderiv_over_m * yi
            dd_zi = w_over_denom * dderiv_over_m * zi
            xx += dd_xi * xi + w_over_denom * dens_val * inv1t
            xy += dd_xi * yi
            xz += dd_xi * zi
            yy += dd_yi * yi + w_over_denom * dens_val * invb2t
            yz += dd_yi * zi
            zz += dd_zi * zi + w_over_denom * dens_val * invc2t
        return xx, xy, xz, yy, yz, zz
    else:
        xx = 0.0
        xy = 0.0
        xz = 0.0
        yy = 0.0
        yz = 0.0
        zz = 0.0
        for k in range(len(glx)):
            s = glx[k]
            w = glw[k]
            t = 1.0 / s**2 - 1.0
            inv1t = 1.0 / (1.0 + t)
            invb2t = 1.0 / (b2 + t)
            invc2t = 1.0 / (c2 + t)
            m = numpy.sqrt(x**2 * inv1t + y**2 * invb2t + z**2 * invc2t)
            denom = numpy.sqrt((1.0 + (b2 - 1.0) * s**2) * (1.0 + (c2 - 1.0) * s**2))
            w_over_denom = w / denom
            dens_val = dens(m)
            dderiv_over_m = densDeriv(m) / m
            xi = x * inv1t
            yi = y * invb2t
            zi = z * invc2t
            dd_xi = w_over_denom * dderiv_over_m * xi
            dd_yi = w_over_denom * dderiv_over_m * yi
            dd_zi = w_over_denom * dderiv_over_m * zi
            xx += dd_xi * xi + w_over_denom * dens_val * inv1t
            xy += dd_xi * yi
            xz += dd_xi * zi
            yy += dd_yi * yi + w_over_denom * dens_val * invb2t
            yz += dd_yi * zi
            zz += dd_zi * zi + w_over_denom * dens_val * invc2t
        return xx, xy, xz, yy, yz, zz
