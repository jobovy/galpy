###############################################################################
#   FerrersPotential.py: General class for triaxial Ferrers Potential
#
#       rho(r) = amp/[a^3 b c pi^1.5] Gamma(n+5/2)/Gamma(n+1) (1 - (m/a)^2)^n
#
#       with
#
#       m^2 = x^2 + y^2/b^2 + z^2/c^2
########################################################################
import hashlib
import math

import numpy
from scipy import integrate
from scipy.special import gamma

from ..backend import get_namespace, zeros_like_backend
from ..util import conversion, coords
from .Potential import Potential


class FerrersPotential(Potential):
    """Class that implements triaxial Ferrers potential for the ellipsoidal density profile with the short axis along the z-direction

    .. math::

        \\rho(x,y,z) = \\frac{\\mathrm{amp}}{\\pi^{1.5} a^3 b c} \\frac{\\Gamma(n+\\frac{5}{2})}{\\Gamma(n+1)}\\,(1-(m/a)^2)^n

    with

    .. math::

        m^2 = x'^2 + \\frac{y'^2}{b^2}+\\frac{z'^2}{c^2}

    and :math:`(x',y',z')` is a rotated frame wrt :math:`(x,y,z)`
    so that the major axis is aligned with :math:`x'`.

    Note that this potential has not yet been optimized for speed and has no C implementation, so orbit integration is currently slow.
    """

    def __init__(
        self,
        amp=1.0,
        a=1.0,
        n=2,
        b=0.35,
        c=0.2375,
        omegab=0.0,
        pa=0.0,
        normalize=False,
        ro=None,
        vo=None,
    ):
        """
        Initialize a Ferrers potential.

        Parameters
        ----------
        amp : float or Quantity, optional
            Total mass of the ellipsoid determines the amplitude of the potential.
        a : float or Quantity, optional
            Scale radius.
        n : int, optional
            Power of Ferrers density (n > 0).
        b : float, optional
            y-to-x axis ratio of the density.
        c : float, optional
            z-to-x axis ratio of the density.
        omegab : float or Quantity, optional
            Rotation speed of the ellipsoid.
        pa : float or Quantity, optional
            If set, the position angle of the x axis (rad or Quantity).
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.
        ro : float, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2011-02-23: Written - Bovy (NYU)
        """
        Potential.__init__(self, amp=amp, ro=ro, vo=vo, amp_units="mass")
        a = conversion.parse_length(a, ro=self._ro)
        omegab = conversion.parse_frequency(omegab, ro=self._ro, vo=self._vo)
        pa = conversion.parse_angle(pa)
        self.a = a
        self._scale = self.a
        if n <= 0:
            raise ValueError("FerrersPotential requires n > 0")
        self.n = n
        self._b = b
        self._c = c
        self._omegab = omegab
        self._a2 = self.a**2
        self._b2 = self._b**2.0
        self._c2 = self._c**2.0
        self._force_hash = None
        self._pa = pa
        self._backend_compatible = True
        self._rhoc_M = gamma(n + 2.5) / gamma(n + 1) / numpy.pi**1.5 / a**3 / b / c
        if normalize or (
            isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
        ):  # pragma: no cover
            self.normalize(normalize)
        if numpy.fabs(self._b - 1.0) > 10.0**-10.0:
            self.isNonAxi = True
        return None

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        if not self.isNonAxi:
            phi = zeros_like_backend(get_namespace(R, z), R)
        x, y, z = coords.cyl_to_rect(R, phi, z)
        # rotation into the aligned frame: rot(t) @ [x, y] without array stacking
        # (numpy.array([x, y]) stacking is the torch-concat backend blocker). The
        # rotation angle follows t (concrete t -> numpy coefficient; traced t ->
        # that backend), like SoftenedNeedleBarPotential.
        xpt = get_namespace(t)
        ang = self._pa + self._omegab * t
        ca, sa = xpt.cos(ang), xpt.sin(ang)
        x, y = ca * x + sa * y, -sa * x + ca * y
        return self._evaluate_xyz(x, y, z)

    def _evaluate_xyz(self, x, y, z=0.0):
        """Evaluation of the potential as a function of (x,y,z) in the
        aligned coordinate frame"""
        return (
            -math.pi
            * self._rhoc_M
            / (self.n + 1.0)
            * self.a**3
            * self._b
            * self._c
            * _potInt(
                x, y, z, self._a2, self._b2 * self._a2, self._c2 * self._a2, self.n
            )
        )

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        if not self.isNonAxi:
            phi = zeros_like_backend(xp, R)
        Fx, Fy, _ = self._cached_xyzforces(R, z, phi, t, xp)
        return xp.cos(phi) * Fx + xp.sin(phi) * Fy

    def _phitorque(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        if not self.isNonAxi:
            phi = zeros_like_backend(xp, R)
        Fx, Fy, _ = self._cached_xyzforces(R, z, phi, t, xp)
        return R * (-xp.sin(phi) * Fx + xp.cos(phi) * Fy)

    def _zforce(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        if not self.isNonAxi:
            phi = zeros_like_backend(xp, R)
        _, _, Fz = self._cached_xyzforces(R, z, phi, t, xp)
        return Fz

    def _compute_xyz(self, R, phi, z, t):
        return coords.cyl_to_rect(R, phi - self._pa - self._omegab * t, z)

    def _xyzforces(self, R, z, phi, t):
        # Pure-functional aligned-then-de-rotated rectangular forces; no
        # per-instance state, so it is safe under jax/torch tracing.
        x, y, z = self._compute_xyz(R, phi, z, t)
        Fx = self._xforce_xyz(x, y, z)
        Fy = self._yforce_xyz(x, y, z)
        Fz = self._zforce_xyz(x, y, z)
        # de-rotation angle; follows t (concrete t -> numpy coefficient that
        # broadcasts; traced t -> that backend's cos/sin), as in
        # SoftenedNeedleBarPotential.
        xpt = get_namespace(t)
        tp = self._pa + self._omegab * t
        cp, sp = xpt.cos(tp), xpt.sin(tp)
        return (cp * Fx - sp * Fy, sp * Fx + cp * Fy, Fz)

    def _cached_xyzforces(self, R, z, phi, t, xp):
        # numpy gets a per-instance hash cache (perf); jax/torch compute directly
        # so the traced path never reads/writes self-state (illegal under tracing).
        if xp is not numpy:
            return self._xyzforces(R, z, phi, t)
        new_hash = hashlib.md5(numpy.array([R, phi, z, t])).hexdigest()
        if new_hash != self._force_hash:
            self._cached_Fx, self._cached_Fy, self._cached_Fz = self._xyzforces(
                R, z, phi, t
            )
            self._force_hash = new_hash
        return self._cached_Fx, self._cached_Fy, self._cached_Fz

    def _xforce_xyz(self, x, y, z):
        """Evaluation of the x force as a function of (x,y,z) in the aligned
        coordinate frame"""
        return (
            -2.0
            * math.pi
            * self._rhoc_M
            * self.a**3
            * self._b
            * self._c
            * _forceInt(
                x, y, z, self._a2, self._b2 * self._a2, self._c2 * self._a2, self.n, 0
            )
        )

    def _yforce_xyz(self, x, y, z):
        """Evaluation of the y force as a function of (x,y,z) in the aligned
        coordinate frame"""
        return (
            -2.0
            * math.pi
            * self._rhoc_M
            * self.a**3
            * self._b
            * self._c
            * _forceInt(
                x, y, z, self._a2, self._b2 * self._a2, self._c2 * self._a2, self.n, 1
            )
        )

    def _zforce_xyz(self, x, y, z):
        """Evaluation of the z force as a function of (x,y,z) in the aligned
        coordinate frame"""
        return (
            -2.0
            * math.pi
            * self._rhoc_M
            * self.a**3
            * self._b
            * self._c
            * _forceInt(
                x, y, z, self._a2, self._b2 * self._a2, self._c2 * self._a2, self.n, 2
            )
        )

    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        if not self.isNonAxi:
            phi = zeros_like_backend(xp, R)
        x, y, z = self._compute_xyz(R, phi, z, t)
        phixxa = self._2ndderiv_xyz(x, y, z, 0, 0)
        phixya = self._2ndderiv_xyz(x, y, z, 0, 1)
        phiyya = self._2ndderiv_xyz(x, y, z, 1, 1)
        xpt = get_namespace(t)
        ang = self._omegab * t + self._pa
        c, s = xpt.cos(ang), xpt.sin(ang)
        phixx = c**2 * phixxa + 2.0 * c * s * phixya + s**2 * phiyya
        phixy = (c**2 - s**2) * phixya + c * s * (phiyya - phixxa)
        phiyy = s**2 * phixxa - 2.0 * c * s * phixya + c**2 * phiyya
        return (
            xp.cos(phi) ** 2.0 * phixx
            + xp.sin(phi) ** 2.0 * phiyy
            + 2.0 * xp.cos(phi) * xp.sin(phi) * phixy
        )

    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        if not self.isNonAxi:
            phi = zeros_like_backend(xp, R)
        x, y, z = self._compute_xyz(R, phi, z, t)
        phixza = self._2ndderiv_xyz(x, y, z, 0, 2)
        phiyza = self._2ndderiv_xyz(x, y, z, 1, 2)
        xpt = get_namespace(t)
        ang = self._omegab * t + self._pa
        c, s = xpt.cos(ang), xpt.sin(ang)
        phixz = c * phixza + s * phiyza
        phiyz = -s * phixza + c * phiyza
        return xp.cos(phi) * phixz + xp.sin(phi) * phiyz

    def _z2deriv(self, R, z, phi=0.0, t=0.0):
        if not self.isNonAxi:
            phi = zeros_like_backend(get_namespace(R, z), R)
        x, y, z = self._compute_xyz(R, phi, z, t)
        return self._2ndderiv_xyz(x, y, z, 2, 2)

    def _phi2deriv(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        if not self.isNonAxi:
            phi = zeros_like_backend(xp, R)
        x, y, z = self._compute_xyz(R, phi, z, t)
        Fx = self._xforce_xyz(x, y, z)
        Fy = self._yforce_xyz(x, y, z)
        # rot(t, transposed=True) @ [Fx, Fy] without array stacking (torch concat
        # blocker); the rotation angle follows t.
        xpt = get_namespace(t)
        ang = self._omegab * t + self._pa
        c, s = xpt.cos(ang), xpt.sin(ang)
        Fx, Fy = c * Fx - s * Fy, s * Fx + c * Fy
        phixxa = self._2ndderiv_xyz(x, y, z, 0, 0)
        phixya = self._2ndderiv_xyz(x, y, z, 0, 1)
        phiyya = self._2ndderiv_xyz(x, y, z, 1, 1)
        phixx = c**2 * phixxa + 2.0 * c * s * phixya + s**2 * phiyya
        phixy = (c**2 - s**2) * phixya + c * s * (phiyya - phixxa)
        phiyy = s**2 * phixxa - 2.0 * c * s * phixya + c**2 * phiyya
        return R**2.0 * (
            xp.sin(phi) ** 2.0 * phixx
            + xp.cos(phi) ** 2.0 * phiyy
            - 2.0 * xp.cos(phi) * xp.sin(phi) * phixy
        ) + R * (xp.cos(phi) * Fx + xp.sin(phi) * Fy)

    def _Rphideriv(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        if not self.isNonAxi:
            phi = zeros_like_backend(xp, R)
        x, y, z = self._compute_xyz(R, phi, z, t)
        Fx = self._xforce_xyz(x, y, z)
        Fy = self._yforce_xyz(x, y, z)
        # rot(t, transposed=True) @ [Fx, Fy] without array stacking (torch concat
        # blocker); the rotation angle follows t.
        xpt = get_namespace(t)
        ang = self._omegab * t + self._pa
        c, s = xpt.cos(ang), xpt.sin(ang)
        Fx, Fy = c * Fx - s * Fy, s * Fx + c * Fy
        phixxa = self._2ndderiv_xyz(x, y, z, 0, 0)
        phixya = self._2ndderiv_xyz(x, y, z, 0, 1)
        phiyya = self._2ndderiv_xyz(x, y, z, 1, 1)
        phixx = c**2 * phixxa + 2.0 * c * s * phixya + s**2 * phiyya
        phixy = (c**2 - s**2) * phixya + c * s * (phiyya - phixxa)
        phiyy = s**2 * phixxa - 2.0 * c * s * phixya + c**2 * phiyya
        return (
            R * xp.cos(phi) * xp.sin(phi) * (phiyy - phixx)
            + R * xp.cos(2.0 * (phi)) * phixy
            + xp.sin(phi) * Fx
            - xp.cos(phi) * Fy
        )

    def _phizderiv(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        if not self.isNonAxi:
            phi = zeros_like_backend(xp, R)
        x, y, z = self._compute_xyz(R, phi, z, t)
        phixza = self._2ndderiv_xyz(x, y, z, 0, 2)
        phiyza = self._2ndderiv_xyz(x, y, z, 1, 2)
        # rot(t, transposed=True) @ [phixza, phiyza] without array stacking; the
        # rotation angle follows t.
        xpt = get_namespace(t)
        ang = self._omegab * t + self._pa
        c, s = xpt.cos(ang), xpt.sin(ang)
        phixz, phiyz = c * phixza - s * phiyza, s * phixza + c * phiyza
        return R * (xp.cos(phi) * phiyz - xp.sin(phi) * phixz)

    def _2ndderiv_xyz(self, x, y, z, i, j):
        r"""General 2nd derivative of the potential as a function of (x,y,z)
        in the aligned coordinate frame, d^2\Phi/dx_i/dx_j"""
        return (
            -math.pi
            * self._rhoc_M
            * self.a**3
            * self._b
            * self._c
            * _2ndDerivInt(
                x,
                y,
                z,
                self._a2,
                self._b2 * self._a2,
                self._c2 * self._a2,
                self.n,
                i,
                j,
            )
        )

    def _dens(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        x, y, z = self._compute_xyz(R, phi, z, t)
        m2 = x**2 / self._a2 + y**2 / self._b2 + z**2 / self._c2
        if xp is numpy:
            # preserve the original numpy behavior exactly (scalar-only: a
            # multi-element array raises on `m2 < 1`, as it always has)
            if m2 < 1:
                return self._rhoc_M * (1.0 - m2 / self.a**2) ** self.n
            else:
                return 0.0
        # jax/torch (incl. 0-d traced scalars): branch-free with a guarded base
        # (1 - m2/a^2) on the m2>=1 side so a non-integer n does not NaN-poison
        # AD; the where discards it.
        base_safe = xp.where(m2 < 1, 1.0 - m2 / self.a**2, 1.0)
        return xp.where(m2 < 1, self._rhoc_M * base_safe**self.n, 0.0)

    def OmegaP(self):
        return self._omegab


def _potInt(x, y, z, a2, b2, c2, n):
    """Integral involved in the potential at (x,y,z)
    integrates 1/A B^(n+1) where
    A = sqrt((tau+a)(tau+b)(tau+c)) and B = (1-x^2/(tau+a)-y^2/(tau+b)-z^2/(tau+c))
    from lambda to infty with respect to tau.
    The lower limit lambda is given by lowerlim function.
    """

    def integrand(tau):
        return _FracInt(x, y, z, a2, b2, c2, tau, n + 1)

    return integrate.quad(integrand, lowerlim(x**2, y**2, z**2, a2, b2, c2), numpy.inf)[
        0
    ]


def _forceInt(x, y, z, a2, b2, c2, n, i):
    """Integral involved in the force at (x,y,z)
    integrates 1/A B^n (x_i/(tau+a_i)) where
    A = sqrt((tau+a)(tau+b)(tau+c)) and B = (1-x^2/(tau+a)-y^2/(tau+b)-z^2/(tau+c))
    from lambda to infty with respect to tau.
    The lower limit lambda is given by lowerlim function.
    """

    def integrand(tau):
        return (
            (x * (i == 0) + y * (i == 1) + z * (i == 2))
            / (a2 * (i == 0) + b2 * (i == 1) + c2 * (i == 2) + tau)
            * _FracInt(x, y, z, a2, b2, c2, tau, n)
        )

    return integrate.quad(
        integrand, lowerlim(x**2, y**2, z**2, a2, b2, c2), numpy.inf, epsabs=1e-12
    )[0]


def _2ndDerivInt(x, y, z, a2, b2, c2, n, i, j):
    r"""Integral involved in second derivatives d^\Phi/(dx_i dx_j)
    integrate
        1/A B^(n-1) (-2 x_i/(tau+a_i)) (-2 x_j/(tau+a_j))
    when i /= j or
        1/A [ B^(n-1) 4n x_i^2 / (a_i+t)^2 + B^n -(-2/(a_i+t)) ]
    when i == j where
    A = sqrt((tau+a)(tau+b)(tau+c)) and B = (1-x^2/(tau+a)-y^2/(tau+b)-z^2/(tau+c))
    from lambda to infty with respect to tau
    The lower limit lambda is given by lowerlim function.
    This is a second derivative of _potInt.
    """

    def integrand(tau):
        if i != j:
            return (
                _FracInt(x, y, z, a2, b2, c2, tau, n - 1)
                * n
                * (1.0 + (-1.0 - 2.0 * x / (tau + a2)) * (i == 0 or j == 0))
                * (1.0 + (-1.0 - 2.0 * y / (tau + b2)) * (i == 1 or j == 1))
                * (1.0 + (-1.0 - 2.0 * z / (tau + c2)) * (i == 2 or j == 2))
            )
        else:
            var2 = x**2 * (i == 0) + y**2 * (i == 1) + z**2 * (i == 2)
            coef2 = a2 * (i == 0) + b2 * (i == 1) + c2 * (i == 2)
            return _FracInt(x, y, z, a2, b2, c2, tau, n - 1) * n * (4.0 * var2) / (
                tau + coef2
            ) ** 2 + _FracInt(x, y, z, a2, b2, c2, tau, n) * (-2.0 / (tau + coef2))

    return integrate.quad(integrand, lowerlim(x**2, y**2, z**2, a2, b2, c2), numpy.inf)[
        0
    ]


def _FracInt(x, y, z, a, b, c, tau, n):
    """Returns
                1                     x^2       y^2       z^2
    -------------------------- (1 - ------- - ------- - -------)^n
    sqrt(tau+a)(tau+b)(tau+c))       tau+a     tau+b     tau+c
    """
    denom = numpy.sqrt((a + tau) * (b + tau) * (c + tau))
    return (1.0 - x**2 / (a + tau) - y**2 / (b + tau) - z**2 / (c + tau)) ** n / denom


def lowerlim(x, y, z, a, b, c):
    """Returns the real positive root of
      x/(a+t) + y/(b+t) + z/(c+t) = 1
    when x/a + y/b + z/c > 1 else zero
    """
    if x / a + y / b + z / c > 1:
        B = a + b + c - x - y - z
        C = a * b + a * c + b * c - a * y - a * z - b * x - b * z - c * x - c * y
        D = a * b * c - a * b * z - a * c * y - b * c * x
        r = numpy.roots([1, B, C, D])
        ll = r[~numpy.iscomplex(r) & (r > 0.0)]
        return ll[0].real
    else:
        return 0.0
