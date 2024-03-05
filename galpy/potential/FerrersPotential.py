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

import numpy
from scipy import integrate
from scipy.special import gamma

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
            phi = 0.0
        x, y, z = coords.cyl_to_rect(R, phi, z)
        xy = numpy.dot(self.rot(t), numpy.array([x, y]))
        x, y = xy[0], xy[1]
        return self._evaluate_xyz(x, y, z)

    def _evaluate_xyz(self, x, y, z=0.0):
        """Evaluation of the potential as a function of (x,y,z) in the
        aligned coordinate frame"""
        return (
            -numpy.pi
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
        if not self.isNonAxi:
            phi = 0.0
        self._compute_xyzforces(R, z, phi, t)
        return numpy.cos(phi) * self._cached_Fx + numpy.sin(phi) * self._cached_Fy

    def _phitorque(self, R, z, phi=0.0, t=0.0):
        if not self.isNonAxi:
            phi = 0.0
        self._compute_xyzforces(R, z, phi, t)
        return R * (
            -numpy.sin(phi) * self._cached_Fx + numpy.cos(phi) * self._cached_Fy
        )

    def _zforce(self, R, z, phi=0.0, t=0.0):
        if not self.isNonAxi:
            phi = 0.0
        self._compute_xyzforces(R, z, phi, t)
        return self._cached_Fz

    def _compute_xyz(self, R, phi, z, t):
        return coords.cyl_to_rect(R, phi - self._pa - self._omegab * t, z)

    def _compute_xyzforces(self, R, z, phi, t):
        # Compute all rectangular forces
        new_hash = hashlib.md5(numpy.array([R, phi, z, t])).hexdigest()
        if new_hash == self._force_hash:
            Fx = self._cached_Fx
            Fy = self._cached_Fy
            Fz = self._cached_Fz
        else:
            x, y, z = self._compute_xyz(R, phi, z, t)
            Fx = self._xforce_xyz(x, y, z)
            Fy = self._yforce_xyz(x, y, z)
            Fz = self._zforce_xyz(x, y, z)
            self._force_hash = new_hash
            tp = self._pa + self._omegab * t
            cp, sp = numpy.cos(tp), numpy.sin(tp)
            self._cached_Fx = cp * Fx - sp * Fy
            self._cached_Fy = sp * Fx + cp * Fy
            self._cached_Fz = Fz

    def _xforce_xyz(self, x, y, z):
        """Evaluation of the x force as a function of (x,y,z) in the aligned
        coordinate frame"""
        return (
            -2.0
            * numpy.pi
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
            * numpy.pi
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
            * numpy.pi
            * self._rhoc_M
            * self.a**3
            * self._b
            * self._c
            * _forceInt(
                x, y, z, self._a2, self._b2 * self._a2, self._c2 * self._a2, self.n, 2
            )
        )

    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        if not self.isNonAxi:
            phi = 0.0
        x, y, z = self._compute_xyz(R, phi, z, t)
        phixxa = self._2ndderiv_xyz(x, y, z, 0, 0)
        phixya = self._2ndderiv_xyz(x, y, z, 0, 1)
        phiyya = self._2ndderiv_xyz(x, y, z, 1, 1)
        ang = self._omegab * t + self._pa
        c, s = numpy.cos(ang), numpy.sin(ang)
        phixx = c**2 * phixxa + 2.0 * c * s * phixya + s**2 * phiyya
        phixy = (c**2 - s**2) * phixya + c * s * (phiyya - phixxa)
        phiyy = s**2 * phixxa - 2.0 * c * s * phixya + c**2 * phiyya
        return (
            numpy.cos(phi) ** 2.0 * phixx
            + numpy.sin(phi) ** 2.0 * phiyy
            + 2.0 * numpy.cos(phi) * numpy.sin(phi) * phixy
        )

    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        if not self.isNonAxi:
            phi = 0.0
        x, y, z = self._compute_xyz(R, phi, z, t)
        phixza = self._2ndderiv_xyz(x, y, z, 0, 2)
        phiyza = self._2ndderiv_xyz(x, y, z, 1, 2)
        ang = self._omegab * t + self._pa
        c, s = numpy.cos(ang), numpy.sin(ang)
        phixz = c * phixza + s * phiyza
        phiyz = -s * phixza + c * phiyza
        return numpy.cos(phi) * phixz + numpy.sin(phi) * phiyz

    def _z2deriv(self, R, z, phi=0.0, t=0.0):
        if not self.isNonAxi:
            phi = 0.0
        x, y, z = self._compute_xyz(R, phi, z, t)
        return self._2ndderiv_xyz(x, y, z, 2, 2)

    def _phi2deriv(self, R, z, phi=0.0, t=0.0):
        if not self.isNonAxi:
            phi = 0.0
        x, y, z = self._compute_xyz(R, phi, z, t)
        Fx = self._xforce_xyz(x, y, z)
        Fy = self._yforce_xyz(x, y, z)
        Fxy = numpy.dot(self.rot(t, transposed=True), numpy.array([Fx, Fy]))
        Fx, Fy = Fxy[0], Fxy[1]
        phixxa = self._2ndderiv_xyz(x, y, z, 0, 0)
        phixya = self._2ndderiv_xyz(x, y, z, 0, 1)
        phiyya = self._2ndderiv_xyz(x, y, z, 1, 1)
        ang = self._omegab * t + self._pa
        c, s = numpy.cos(ang), numpy.sin(ang)
        phixx = c**2 * phixxa + 2.0 * c * s * phixya + s**2 * phiyya
        phixy = (c**2 - s**2) * phixya + c * s * (phiyya - phixxa)
        phiyy = s**2 * phixxa - 2.0 * c * s * phixya + c**2 * phiyya
        return R**2.0 * (
            numpy.sin(phi) ** 2.0 * phixx
            + numpy.cos(phi) ** 2.0 * phiyy
            - 2.0 * numpy.cos(phi) * numpy.sin(phi) * phixy
        ) + R * (numpy.cos(phi) * Fx + numpy.sin(phi) * Fy)

    def _Rphideriv(self, R, z, phi=0.0, t=0.0):
        if not self.isNonAxi:
            phi = 0.0
        x, y, z = self._compute_xyz(R, phi, z, t)
        Fx = self._xforce_xyz(x, y, z)
        Fy = self._yforce_xyz(x, y, z)
        Fxy = numpy.dot(self.rot(t, transposed=True), numpy.array([Fx, Fy]))
        Fx, Fy = Fxy[0], Fxy[1]
        phixxa = self._2ndderiv_xyz(x, y, z, 0, 0)
        phixya = self._2ndderiv_xyz(x, y, z, 0, 1)
        phiyya = self._2ndderiv_xyz(x, y, z, 1, 1)
        ang = self._omegab * t + self._pa
        c, s = numpy.cos(ang), numpy.sin(ang)
        phixx = c**2 * phixxa + 2.0 * c * s * phixya + s**2 * phiyya
        phixy = (c**2 - s**2) * phixya + c * s * (phiyya - phixxa)
        phiyy = s**2 * phixxa - 2.0 * c * s * phixya + c**2 * phiyya
        return (
            R * numpy.cos(phi) * numpy.sin(phi) * (phiyy - phixx)
            + R * numpy.cos(2.0 * (phi)) * phixy
            + numpy.sin(phi) * Fx
            - numpy.cos(phi) * Fy
        )

    def _phizderiv(self, R, z, phi=0.0, t=0.0):
        if not self.isNonAxi:
            phi = 0.0
        x, y, z = self._compute_xyz(R, phi, z, t)
        phixza = self._2ndderiv_xyz(x, y, z, 0, 2)
        phiyza = self._2ndderiv_xyz(x, y, z, 1, 2)
        phixz, phiyz = numpy.dot(
            self.rot(t, transposed=True), numpy.array([phixza, phiyza])
        )
        return R * (numpy.cos(phi) * phiyz - numpy.sin(phi) * phixz)

    def _2ndderiv_xyz(self, x, y, z, i, j):
        r"""General 2nd derivative of the potential as a function of (x,y,z)
        in the aligned coordinate frame, d^2\Phi/dx_i/dx_j"""
        return (
            -numpy.pi
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
        x, y, z = self._compute_xyz(R, phi, z, t)
        m2 = x**2 / self._a2 + y**2 / self._b2 + z**2 / self._c2
        if m2 < 1:
            return self._rhoc_M * (1.0 - m2 / self.a**2) ** self.n
        else:
            return 0.0

    def OmegaP(self):
        return self._omegab

    def rot(self, t=0.0, transposed=False):
        """2D Rotation matrix for non-zero pa or pattern speed
        to goto the aligned coordinates
        """
        rotmat = numpy.array(
            [
                [
                    numpy.cos(self._pa + self._omegab * t),
                    numpy.sin(self._pa + self._omegab * t),
                ],
                [
                    -numpy.sin(self._pa + self._omegab * t),
                    numpy.cos(self._pa + self._omegab * t),
                ],
            ]
        )
        if transposed:
            return rotmat.T
        else:
            return rotmat


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
