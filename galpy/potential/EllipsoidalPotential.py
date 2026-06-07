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
import math

import numpy
from scipy import integrate

from ..backend import get_namespace
from ..util import _rotate_to_arbitrary_vector, conversion
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

    Note that rotated instances (non-trivial ``zvec``/``pa``) do not support C-based variational integration (``Orbit.integrate_dxdv``; ``hasC_dxdv3d=False``): wrap the aligned potential in a ``RotateAndTiltWrapperPotential`` instead, which implements identical physics with full 3D dxdv support.
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

    def _rotate_to_aligned(self, x, y, z, xp):
        """Rotate (x,y,z) into the aligned density frame using ``self._rot``.

        Written as explicit (constant) component products so that it is pure
        arithmetic and backend-agnostic (numpy values unchanged from the former
        ``numpy.dot(self._rot, numpy.array([x, y, z]))``)."""
        rot = self._rot
        xp_ = rot[0, 0] * x + rot[0, 1] * y + rot[0, 2] * z
        yp_ = rot[1, 0] * x + rot[1, 1] * y + rot[1, 2] * z
        zp_ = rot[2, 0] * x + rot[2, 1] * y + rot[2, 2] * z
        return xp_, yp_, zp_

    def _rotate_force_back(self, Fx, Fy, Fz, xp):
        """Rotate a force triple from the aligned frame back to the data frame
        using ``self._rot.T`` (pure arithmetic, backend-agnostic)."""
        rot = self._rot
        Fx_ = rot[0, 0] * Fx + rot[1, 0] * Fy + rot[2, 0] * Fz
        Fy_ = rot[0, 1] * Fx + rot[1, 1] * Fy + rot[2, 1] * Fz
        Fz_ = rot[0, 2] * Fx + rot[1, 2] * Fy + rot[2, 2] * Fz
        return Fx_, Fy_, Fz_

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        if not self.isNonAxi:
            phi = 0.0
        x, y = R * xp.cos(phi), R * xp.sin(phi)
        # When R is infinite, y = R*sin(phi) is inf/nan; force it to 0 (the
        # potential along the axis). Shape-polymorphic so it works on scalars
        # and arrays in every backend.
        y = xp.where(xp.isinf(R), 0.0, y)
        if self._aligned:
            return self._evaluate_xyz(x, y, z, xp)
        else:
            xp_, yp_, zp_ = self._rotate_to_aligned(x, y, z, xp)
            return self._evaluate_xyz(xp_, yp_, zp_, xp)

    def _evaluate_xyz(self, x, y, z, xp=None):
        """Evaluation of the potential as a function of (x,y,z) in the
        aligned coordinate frame"""
        if xp is None:
            xp = get_namespace(x, y, z)
        return (
            2.0
            * math.pi
            * self._b
            * self._c
            * _potInt(
                x, y, z, self._psi, self._b2, self._c2, xp, glx=self._glx, glw=self._glw
            )
        )

    def _compute_forces(self, x, y, z, xp):
        """Compute all three force components in the aligned frame.

        Returns ``(Fx, Fy, Fz)`` already rotated back into the data frame. The
        shared force integral is computed as a local; for the numpy backend the
        result is also stored in a per-instance cache (keyed on the input hash)
        so the three public force methods evaluated at the same point reuse a
        single quadrature, exactly as before. The traced (jax/torch) path never
        touches ``self``-state."""
        if xp is numpy:
            new_hash = hashlib.md5(numpy.array([x, y, z])).hexdigest()
            if new_hash == self._force_hash:
                return self._cached_Fx, self._cached_Fy, self._cached_Fz
        if self._aligned:
            xa, ya, za = x, y, z
        else:
            xa, ya, za = self._rotate_to_aligned(x, y, z, xp)
        prefac = -4.0 * math.pi * self._b * self._c
        Fx, Fy, Fz = _forceInt_all(
            xa,
            ya,
            za,
            lambda m: self._mdens(m),
            self._b2,
            self._c2,
            xp,
            glx=self._glx,
            glw=self._glw,
        )
        Fx = prefac * Fx
        Fy = prefac * Fy
        Fz = prefac * Fz
        if not self._aligned:
            Fx, Fy, Fz = self._rotate_force_back(Fx, Fy, Fz, xp)
        if xp is numpy:
            self._cached_Fx, self._cached_Fy, self._cached_Fz = Fx, Fy, Fz
            self._force_hash = new_hash
        return Fx, Fy, Fz

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        if not self.isNonAxi:
            phi = 0.0
        x, y = R * xp.cos(phi), R * xp.sin(phi)
        Fx, Fy, _ = self._compute_forces(x, y, z, xp)
        return xp.cos(phi) * Fx + xp.sin(phi) * Fy

    def _phitorque(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        if not self.isNonAxi:
            phi = 0.0
        x, y = R * xp.cos(phi), R * xp.sin(phi)
        Fx, Fy, _ = self._compute_forces(x, y, z, xp)
        return R * (-xp.sin(phi) * Fx + xp.cos(phi) * Fy)

    def _zforce(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        if not self.isNonAxi:
            phi = 0.0
        x, y = R * xp.cos(phi), R * xp.sin(phi)
        _, _, Fz = self._compute_forces(x, y, z, xp)
        return Fz

    def _compute_2ndderivs(self, x, y, z, xp):
        """Compute all six unique 2nd-derivative components in the aligned frame.

        Returns ``(xx, xy, xz, yy, yz, zz)``. The shared quadrature is computed
        as a local; for the numpy backend the result is also cached on the
        instance (keyed on the input hash) so methods sharing a point reuse it.
        The traced (jax/torch) path never touches ``self``-state. Only used for
        the aligned case (the public methods raise for rotated frames)."""
        if xp is numpy:
            new_hash = hashlib.md5(numpy.array([x, y, z])).hexdigest()
            if new_hash == self._2ndderiv_hash:
                return (
                    self._cached_2nd_xx,
                    self._cached_2nd_xy,
                    self._cached_2nd_xz,
                    self._cached_2nd_yy,
                    self._cached_2nd_yz,
                    self._cached_2nd_zz,
                )
        prefac = 4.0 * math.pi * self._b * self._c
        xx, xy, xz, yy, yz, zz = _2ndDerivInt_all(
            x,
            y,
            z,
            lambda m: self._mdens(m),
            lambda m: self._mdens_deriv(m),
            self._b2,
            self._c2,
            xp,
            glx=self._glx,
            glw=self._glw,
        )
        xx = prefac * xx
        xy = prefac * xy
        xz = prefac * xz
        yy = prefac * yy
        yz = prefac * yz
        zz = prefac * zz
        if xp is numpy:
            self._cached_2nd_xx = xx
            self._cached_2nd_xy = xy
            self._cached_2nd_xz = xz
            self._cached_2nd_yy = yy
            self._cached_2nd_yz = yz
            self._cached_2nd_zz = zz
            self._2ndderiv_hash = new_hash
        return xx, xy, xz, yy, yz, zz

    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        if not self.isNonAxi:
            phi = 0.0
        x, y = R * xp.cos(phi), R * xp.sin(phi)
        if not self._aligned:
            raise NotImplementedError(
                "2nd potential derivatives of TwoPowerTriaxialPotential not implemented for rotated coordinated frames (non-trivial zvec and pa); use RotateAndTiltWrapperPotential for this functionality instead"
            )
        phixx, phixy, _, phiyy, _, _ = self._compute_2ndderivs(x, y, z, xp)
        return (
            xp.cos(phi) ** 2.0 * phixx
            + xp.sin(phi) ** 2.0 * phiyy
            + 2.0 * xp.cos(phi) * xp.sin(phi) * phixy
        )

    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        if not self.isNonAxi:
            phi = 0.0
        x, y = R * xp.cos(phi), R * xp.sin(phi)
        if not self._aligned:
            raise NotImplementedError(
                "2nd potential derivatives of TwoPowerTriaxialPotential not implemented for rotated coordinated frames (non-trivial zvec and pa); use RotateAndTiltWrapperPotential for this functionality instead"
            )
        _, _, phixz, _, phiyz, _ = self._compute_2ndderivs(x, y, z, xp)
        return xp.cos(phi) * phixz + xp.sin(phi) * phiyz

    def _z2deriv(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        if not self.isNonAxi:
            phi = 0.0
        x, y = R * xp.cos(phi), R * xp.sin(phi)
        if not self._aligned:
            raise NotImplementedError(
                "2nd potential derivatives of TwoPowerTriaxialPotential not implemented for rotated coordinated frames (non-trivial zvec and pa); use RotateAndTiltWrapperPotential for this functionality instead"
            )
        _, _, _, _, _, phizz = self._compute_2ndderivs(x, y, z, xp)
        return phizz

    def _phi2deriv(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        if not self.isNonAxi:
            phi = 0.0
        x, y = R * xp.cos(phi), R * xp.sin(phi)
        if not self._aligned:
            raise NotImplementedError(
                "2nd potential derivatives of TwoPowerTriaxialPotential not implemented for rotated coordinated frames (non-trivial zvec and pa); use RotateAndTiltWrapperPotential for this functionality instead"
            )
        Fx, Fy, _ = self._compute_forces(x, y, z, xp)
        phixx, phixy, _, phiyy, _, _ = self._compute_2ndderivs(x, y, z, xp)
        return R**2.0 * (
            xp.sin(phi) ** 2.0 * phixx
            + xp.cos(phi) ** 2.0 * phiyy
            - 2.0 * xp.cos(phi) * xp.sin(phi) * phixy
        ) + R * (xp.cos(phi) * Fx + xp.sin(phi) * Fy)

    def _Rphideriv(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        if not self.isNonAxi:
            phi = 0.0
        x, y = R * xp.cos(phi), R * xp.sin(phi)
        if not self._aligned:
            raise NotImplementedError(
                "2nd potential derivatives of TwoPowerTriaxialPotential not implemented for rotated coordinated frames (non-trivial zvec and pa); use RotateAndTiltWrapperPotential for this functionality instead"
            )
        Fx, Fy, _ = self._compute_forces(x, y, z, xp)
        phixx, phixy, _, phiyy, _, _ = self._compute_2ndderivs(x, y, z, xp)
        return (
            R * xp.cos(phi) * xp.sin(phi) * (phiyy - phixx)
            + R * xp.cos(2.0 * phi) * phixy
            + xp.sin(phi) * Fx
            - xp.cos(phi) * Fy
        )

    def _phizderiv(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        if not self.isNonAxi:
            phi = 0.0
        x, y = R * xp.cos(phi), R * xp.sin(phi)
        if not self._aligned:
            raise NotImplementedError(
                "2nd potential derivatives of TwoPowerTriaxialPotential not implemented for rotated coordinated frames (non-trivial zvec and pa); use RotateAndTiltWrapperPotential for this functionality instead"
            )
        _, _, phixz, _, phiyz, _ = self._compute_2ndderivs(x, y, z, xp)
        return R * (xp.cos(phi) * phiyz - xp.sin(phi) * phixz)

    def _dens(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        x, y = R * xp.cos(phi), R * xp.sin(phi)
        if self._aligned:
            xa, ya, za = x, y, z
        else:
            xa, ya, za = self._rotate_to_aligned(x, y, z, xp)
        m = xp.sqrt(xa**2.0 + ya**2.0 / self._b2 + za**2.0 / self._c2)
        return self._mdens(m)

    def _mass(self, R, z=None, t=0.0):
        if not z is None:
            raise AttributeError  # Hack to fall back to general
        # Pspecial-blocked: the generic ellipsoidal mass uses an adaptive
        # scipy.integrate.quad over the density, which has no backend-agnostic
        # (jax/torch) replacement -> numpy only.
        return (
            4.0
            * numpy.pi
            * self._b
            * self._c
            * integrate.quad(lambda m: m**2.0 * self._mdens(m), 0, R)[0]
        )

    def OmegaP(self):
        return 0.0


def _potInt(x, y, z, psi, b2, c2, xp=numpy, glx=None, glw=None):
    r"""int_0^\infty [psi(m)-psi(\infy)]/sqrt([1+tau]x[b^2+tau]x[c^2+tau])dtau"""

    def integrand(s):
        t = 1 / s**2.0 - 1.0
        return psi(
            numpy.sqrt(x**2.0 / (1.0 + t) + y**2.0 / (b2 + t) + z**2.0 / (c2 + t))
        ) / numpy.sqrt((1.0 + (b2 - 1.0) * s**2.0) * (1.0 + (c2 - 1.0) * s**2.0))

    if glx is None:
        # scipy.integrate fallback (glorder=None): numpy-only, deferred to Pspecial
        return integrate.quad(integrand, 0.0, 1.0)[0]
    result = 0.0
    x2 = x**2
    y2 = y**2
    z2 = z**2
    for k in range(len(glx)):
        s = glx[k]
        t = 1.0 / s**2 - 1.0
        denom = numpy.sqrt((1.0 + (b2 - 1.0) * s**2) * (1.0 + (c2 - 1.0) * s**2))
        m = xp.sqrt(x2 / (1.0 + t) + y2 / (b2 + t) + z2 / (c2 + t))
        result = result + glw[k] * psi(m) / denom
    return result


def _forceInt(x, y, z, dens, b2, c2, i):
    """Force integral fallback using scipy.integrate.quad (scalar inputs only).
    Used by _forceInt_all when glorder is None."""

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

    return integrate.quad(integrand, 0.0, 1.0)[0]


def _forceInt_all(x, y, z, dens, b2, c2, xp=numpy, glx=None, glw=None):
    """Compute all three force integral components in a single pass."""
    if glx is None:
        # scipy.integrate fallback (glorder=None): numpy-only, deferred to Pspecial
        return (
            _forceInt(x, y, z, dens, b2, c2, 0),
            _forceInt(x, y, z, dens, b2, c2, 1),
            _forceInt(x, y, z, dens, b2, c2, 2),
        )
    Fx = Fy = Fz = 0.0
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
        m = xp.sqrt(x2 * inv1t + y2 * invb2t + z2 * invc2t)
        denom = numpy.sqrt((1.0 + (b2 - 1.0) * s**2) * (1.0 + (c2 - 1.0) * s**2))
        common = w * dens(m) / denom
        Fx = Fx + common * x * inv1t
        Fy = Fy + common * y * invb2t
        Fz = Fz + common * z * invc2t
    return Fx, Fy, Fz


def _2ndDerivInt(x, y, z, dens, densDeriv, b2, c2, i, j):
    """2nd-derivative integral fallback using scipy.integrate.quad (scalar inputs
    only). Used by _2ndDerivInt_all when glorder is None."""

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

    return integrate.quad(integrand, 0.0, 1.0)[0]


def _2ndDerivInt_all(x, y, z, dens, densDeriv, b2, c2, xp=numpy, glx=None, glw=None):
    """Compute all six unique 2nd derivative integrals in a single pass.
    Returns (xx, xy, xz, yy, yz, zz)."""
    if glx is None:
        # scipy.integrate fallback (glorder=None): numpy-only, deferred to Pspecial
        return (
            _2ndDerivInt(x, y, z, dens, densDeriv, b2, c2, 0, 0),
            _2ndDerivInt(x, y, z, dens, densDeriv, b2, c2, 0, 1),
            _2ndDerivInt(x, y, z, dens, densDeriv, b2, c2, 0, 2),
            _2ndDerivInt(x, y, z, dens, densDeriv, b2, c2, 1, 1),
            _2ndDerivInt(x, y, z, dens, densDeriv, b2, c2, 1, 2),
            _2ndDerivInt(x, y, z, dens, densDeriv, b2, c2, 2, 2),
        )
    xx = xy = xz = yy = yz = zz = 0.0
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
        m = xp.sqrt(x2 * inv1t + y2 * invb2t + z2 * invc2t)
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
        xx = xx + dd_xi * xi + w_over_denom * dens_val * inv1t
        xy = xy + dd_xi * yi
        xz = xz + dd_xi * zi
        yy = yy + dd_yi * yi + w_over_denom * dens_val * invb2t
        yz = yz + dd_yi * zi
        zz = zz + dd_zi * zi + w_over_denom * dens_val * invc2t
    return xx, xy, xz, yy, yz, zz
