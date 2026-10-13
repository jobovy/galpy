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

from ..backend import (
    coerce_coords,
    get_namespace,
    is_backend_array,
    scalar_like,
    zeros_like_backend,
)
from ..backend._namespaces import (
    eager_memo_applies,
    eager_value_memo,
    stop_gradient,
)
from ..backend.optimize import newton_polish
from ..backend.quadrature import fixed_quad_semiinfinite
from ..backend.special import gamma
from ..util import conversion, coords
from .Potential import Potential, _normalize_requested

# Gauss-Legendre order for the backend (jax/torch) branch of the ellipsoidal
# integrals; the 'recip' semi-infinite substitution makes the Ferrers integrand
# smooth, so this converges immediately (order 50 already matches scipy to
# ~1e-13). numpy inputs keep the scipy adaptive path (byte-identical).
_GLORDER = 100
# every force axis and second-derivative pair (_derivs_xyz's eager memo)
_FORCES = (0, 1, 2)
_SECONDS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


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
        # Routed gamma, on a coerced n, so rho_c is computed ON the backend under
        # a force (scipy would hand back a DETACHED tensor) and stays
        # differentiable in n. b and c are deliberately NOT coerced: the
        # numpy.fabs(self._b - 1.0) check below rejects a tensor under -W error.
        (n_,) = coerce_coords(get_namespace(n), n)
        self._rhoc_M = gamma(n_ + 2.5) / gamma(n_ + 1) / numpy.pi**1.5 / a**3 / b / c
        if _normalize_requested(normalize):  # pragma: no cover
            self.normalize(normalize)
        if numpy.fabs(self._b - 1.0) > 10.0**-10.0:
            self.isNonAxi = True
        return None

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        R, z, phi = coerce_coords(xp, R, z, phi)
        if not self.isNonAxi:
            phi = zeros_like_backend(xp, R)
        x, y, z = coords.cyl_to_rect(R, phi, z)
        # rotation into the aligned frame: rot(t) @ [x, y] without array stacking
        # (numpy.array([x, y]) stacking is the torch-concat backend blocker). The
        # rotation angle follows t: a concrete scalar t -> numpy coefficient (broadcasts
        # against backend coords under force; byte-identical for numpy), a traced backend
        # t -> that backend (differentiable), like SoftenedNeedleBarPotential.
        xpt = get_namespace(t) if is_backend_array(t) else numpy
        ang = self._pa + self._omegab * t
        ca, sa = xpt.cos(ang), xpt.sin(ang)
        x, y = ca * x + sa * y, -sa * x + ca * y
        return self._evaluate_xyz(x, y, z)

    def _evaluate_xyz(self, x, y, z=0.0):
        """Evaluation of the potential as a function of (x,y,z) in the
        aligned coordinate frame"""
        return (
            -math.pi
            * scalar_like(x, self._rhoc_M)
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
        R, z, phi = coerce_coords(xp, R, z, phi)
        if not self.isNonAxi:
            phi = zeros_like_backend(xp, R)
        Fx, Fy, _ = self._cached_xyzforces(R, z, phi, t, xp)
        return xp.cos(phi) * Fx + xp.sin(phi) * Fy

    def _phitorque(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        R, z, phi = coerce_coords(xp, R, z, phi)
        if not self.isNonAxi:
            phi = zeros_like_backend(xp, R)
        Fx, Fy, _ = self._cached_xyzforces(R, z, phi, t, xp)
        return R * (-xp.sin(phi) * Fx + xp.cos(phi) * Fy)

    def _zforce(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        R, z, phi = coerce_coords(xp, R, z, phi)
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
        (Fx, Fy, Fz), _ = self._derivs_xyz(x, y, z, forces=(0, 1, 2))
        # de-rotation angle; follows t (concrete scalar t -> numpy coefficient that
        # broadcasts against backend coords; traced backend t -> that backend's cos/sin),
        # as in SoftenedNeedleBarPotential.
        xpt = get_namespace(t) if is_backend_array(t) else numpy
        tp = self._pa + self._omegab * t
        cp, sp = xpt.cos(tp), xpt.sin(tp)
        return (cp * Fx - sp * Fy, sp * Fx + cp * Fy, Fz)

    def _cached_xyzforces(self, R, z, phi, t, xp):
        # numpy gets a per-instance hash cache (perf); eager jax/torch reuse
        # through _derivs_xyz's memo, and the traced path holds no state.
        if xp is not numpy:
            return self._xyzforces(R, z, phi, t)
        new_hash = hashlib.md5(numpy.array([R, phi, z, t])).hexdigest()
        if new_hash != self._force_hash:
            self._cached_Fx, self._cached_Fy, self._cached_Fz = self._xyzforces(
                R, z, phi, t
            )
            self._force_hash = new_hash
        return self._cached_Fx, self._cached_Fy, self._cached_Fz

    def _derivs_xyz(self, x, y, z, forces=(), seconds=()):
        """Aligned-frame forces (axes ``forces``) and second derivatives (index
        pairs ``seconds``) at (x,y,z), as ``_[xyz]force_xyz``/``_2ndderiv_xyz``.
        numpy: those, one adaptive quad each. Backend: ONE fixed-order quadrature
        over all the integrands, sharing the lower limit and the node evaluations
        (the dxdv EOM needs 19 integrals per step, at one point)."""
        if not (is_backend_array(x) or is_backend_array(y) or is_backend_array(z)):
            force_xyz = (self._xforce_xyz, self._yforce_xyz, self._zforce_xyz)
            return (
                [force_xyz[i](x, y, z) for i in forces],
                [self._2ndderiv_xyz(x, y, z, i, j) for i, j in seconds],
            )
        if eager_memo_applies(self, x, y, z):
            # eager: every derivative at the point in one quadrature, reused by
            # all the methods asked at that point (the EOM asks five)
            F, S = eager_value_memo(
                self,
                "derivs_xyz",
                (x, y, z),
                lambda: self._derivs_xyz_backend(x, y, z, _FORCES, _SECONDS),
            )
            return [F[i] for i in forces], [S[_SECONDS.index(p)] for p in seconds]
        return self._derivs_xyz_backend(x, y, z, forces, seconds)

    def _derivs_xyz_backend(self, x, y, z, forces, seconds):
        ints = _derivInts_backend(
            x,
            y,
            z,
            self._a2,
            self._b2 * self._a2,
            self._c2 * self._a2,
            self.n,
            forces,
            seconds,
            get_namespace(x, y, z),
        )
        pref = scalar_like(x, self._rhoc_M) * self.a**3 * self._b * self._c
        nf = len(forces)
        return (
            [-2.0 * math.pi * pref * ints[k] for k in range(nf)],
            [-math.pi * pref * ints[nf + k] for k in range(len(seconds))],
        )

    def _xforce_xyz(self, x, y, z):
        """Evaluation of the x force as a function of (x,y,z) in the aligned
        coordinate frame"""
        return (
            -2.0
            * math.pi
            * scalar_like(x, self._rhoc_M)
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
            * scalar_like(x, self._rhoc_M)
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
            * scalar_like(x, self._rhoc_M)
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
        _, (phixxa, phixya, phiyya) = self._derivs_xyz(
            x, y, z, seconds=((0, 0), (0, 1), (1, 1))
        )
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
        _, (phixza, phiyza) = self._derivs_xyz(x, y, z, seconds=((0, 2), (1, 2)))
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
        (Fx, Fy), (phixxa, phixya, phiyya) = self._derivs_xyz(
            x, y, z, forces=(0, 1), seconds=((0, 0), (0, 1), (1, 1))
        )
        # rot(t, transposed=True) @ [Fx, Fy] without array stacking (torch concat
        # blocker); the rotation angle follows t.
        xpt = get_namespace(t)
        ang = self._omegab * t + self._pa
        c, s = xpt.cos(ang), xpt.sin(ang)
        Fx, Fy = c * Fx - s * Fy, s * Fx + c * Fy
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
        (Fx, Fy), (phixxa, phixya, phiyya) = self._derivs_xyz(
            x, y, z, forces=(0, 1), seconds=((0, 0), (0, 1), (1, 1))
        )
        # rot(t, transposed=True) @ [Fx, Fy] without array stacking (torch concat
        # blocker); the rotation angle follows t.
        xpt = get_namespace(t)
        ang = self._omegab * t + self._pa
        c, s = xpt.cos(ang), xpt.sin(ang)
        Fx, Fy = c * Fx - s * Fy, s * Fx + c * Fy
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
        _, (phixza, phiyza) = self._derivs_xyz(x, y, z, seconds=((0, 2), (1, 2)))
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
            * scalar_like(x, self._rhoc_M)
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
        return xp.where(m2 < 1, scalar_like(m2, self._rhoc_M) * base_safe**self.n, 0.0)

    def OmegaP(self):
        return self._omegab


def _potInt(x, y, z, a2, b2, c2, n):
    """Integral involved in the potential at (x,y,z)
    integrates 1/A B^(n+1) where
    A = sqrt((tau+a)(tau+b)(tau+c)) and B = (1-x^2/(tau+a)-y^2/(tau+b)-z^2/(tau+c))
    from lambda to infty with respect to tau.
    The lower limit lambda is given by lowerlim function.
    numpy inputs use scipy.integrate.quad (byte-identical); backend inputs use a
    fixed-order Gauss-Legendre semi-infinite quadrature (jit/grad-safe).
    """
    if not (is_backend_array(x) or is_backend_array(y) or is_backend_array(z)):

        def integrand(tau):
            return _FracInt(x, y, z, a2, b2, c2, tau, n + 1, numpy)

        return integrate.quad(
            integrand, lowerlim(x**2, y**2, z**2, a2, b2, c2), numpy.inf
        )[0]
    xp = get_namespace(x, y, z)
    ll = _lowerlim_backend(x**2, y**2, z**2, a2, b2, c2, xp)
    # fixed_quad_semiinfinite appends a trailing quadrature-node axis to ll, so the
    # integrand's tau carries that axis; give the coordinates the same trailing axis
    # (no-op broadcast for a scalar point, correct for an ARRAY of points -- e.g. the
    # per-time positions Orbit.E() evaluates at once).
    xe, ye, ze = x[..., None], y[..., None], z[..., None]

    def integrand(tau):
        return _FracInt(xe, ye, ze, a2, b2, c2, tau, n + 1, xp)

    return fixed_quad_semiinfinite(xp, integrand, ll, n=_GLORDER, kind="recip")


def _forceInt(x, y, z, a2, b2, c2, n, i):
    """Integral involved in the force at (x,y,z)
    integrates 1/A B^n (x_i/(tau+a_i)) where
    A = sqrt((tau+a)(tau+b)(tau+c)) and B = (1-x^2/(tau+a)-y^2/(tau+b)-z^2/(tau+c))
    from lambda to infty with respect to tau.
    The lower limit lambda is given by lowerlim function.
    numpy only (scipy.integrate.quad): backend forces are all computed with the
    other derivatives at the point, by :func:`_derivInts_backend`.
    """

    def integrand(tau):
        return (
            (x * (i == 0) + y * (i == 1) + z * (i == 2))
            / (a2 * (i == 0) + b2 * (i == 1) + c2 * (i == 2) + tau)
            * _FracInt(x, y, z, a2, b2, c2, tau, n, numpy)
        )

    return integrate.quad(
        integrand,
        lowerlim(x**2, y**2, z**2, a2, b2, c2),
        numpy.inf,
        epsabs=1e-12,
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

    def _integrand(tau, xp, X, Y, Z):
        if i != j:
            return (
                _FracInt(X, Y, Z, a2, b2, c2, tau, n - 1, xp)
                * n
                * (1.0 + (-1.0 - 2.0 * X / (tau + a2)) * (i == 0 or j == 0))
                * (1.0 + (-1.0 - 2.0 * Y / (tau + b2)) * (i == 1 or j == 1))
                * (1.0 + (-1.0 - 2.0 * Z / (tau + c2)) * (i == 2 or j == 2))
            )
        else:
            var2 = X**2 * (i == 0) + Y**2 * (i == 1) + Z**2 * (i == 2)
            coef2 = a2 * (i == 0) + b2 * (i == 1) + c2 * (i == 2)
            return _FracInt(X, Y, Z, a2, b2, c2, tau, n - 1, xp) * n * (4.0 * var2) / (
                tau + coef2
            ) ** 2 + _FracInt(X, Y, Z, a2, b2, c2, tau, n, xp) * (-2.0 / (tau + coef2))

    if not (is_backend_array(x) or is_backend_array(y) or is_backend_array(z)):
        return integrate.quad(
            lambda tau: _integrand(tau, numpy, x, y, z),
            lowerlim(x**2, y**2, z**2, a2, b2, c2),
            numpy.inf,
        )[0]
    xp = get_namespace(x, y, z)
    ll = _lowerlim_backend(x**2, y**2, z**2, a2, b2, c2, xp)
    # trailing quadrature-node axis on the coordinates (see _potInt): scalar-safe,
    # correct for an array of evaluation points.
    xe, ye, ze = x[..., None], y[..., None], z[..., None]
    return fixed_quad_semiinfinite(
        xp, lambda tau: _integrand(tau, xp, xe, ye, ze), ll, n=_GLORDER, kind="recip"
    )


def _derivInts_backend(x, y, z, a2, b2, c2, n, forces, seconds, xp):
    """Backend :func:`_forceInt` (axes ``forces``) and :func:`_2ndDerivInt` (index
    pairs ``seconds``) at one (x,y,z) in ONE fixed-order quadrature: one lower
    limit, one set of node evaluations, the integrands stacked on a leading axis.
    Each integrand is the same arithmetic as in those functions (their factors of
    exactly 1 and terms of exactly 0 dropped)."""
    ll = _lowerlim_backend(x**2, y**2, z**2, a2, b2, c2, xp)
    X = (x[..., None], y[..., None], z[..., None])
    A = (a2, b2, c2)

    def integrand(tau):
        denom = xp.sqrt((a2 + tau) * (b2 + tau) * (c2 + tau))
        base = (
            1.0
            - X[0] ** 2 / (a2 + tau)
            - X[1] ** 2 / (b2 + tau)
            - X[2] ** 2 / (c2 + tau)
        )
        frac_n = base**n / denom
        out = [X[i] / (A[i] + tau) * frac_n for i in forces]
        if seconds:
            frac_nm1 = base ** (n - 1) / denom
        for i, j in seconds:
            if i != j:
                out.append(
                    frac_nm1
                    * n
                    * (1.0 + (-1.0 - 2.0 * X[i] / (tau + A[i])))
                    * (1.0 + (-1.0 - 2.0 * X[j] / (tau + A[j])))
                )
            else:
                out.append(
                    frac_nm1 * n * (4.0 * X[i] ** 2) / (tau + A[i]) ** 2
                    + frac_n * (-2.0 / (tau + A[i]))
                )
        return xp.stack(out, axis=0)

    return fixed_quad_semiinfinite(xp, integrand, ll, n=_GLORDER, kind="recip")


def _FracInt(x, y, z, a, b, c, tau, n, xp=numpy):
    """Returns
                1                     x^2       y^2       z^2
    -------------------------- (1 - ------- - ------- - -------)^n
    sqrt(tau+a)(tau+b)(tau+c))       tau+a     tau+b     tau+c

    ``xp`` is the array namespace of the inputs; for numpy inputs
    ``xp.sqrt`` is ``numpy.sqrt`` so this is byte-identical.
    """
    denom = xp.sqrt((a + tau) * (b + tau) * (c + tau))
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


def _lowerlim_backend(x, y, z, a, b, c, xp):
    """Backend (jax/torch) counterpart of :func:`lowerlim`.

    ``x, y, z`` are the SQUARED coordinates and ``a, b, c`` the squared axis
    parameters (as ``lowerlim`` is called). The lower limit is the largest root
    of the cubic ``lowerlim`` solves with ``numpy.roots`` when the point is
    outside (``g(0) > 0`` for ``g(t) = x/(a+t) + y/(b+t) + z/(c+t) - 1``), else 0.
    Its three roots are real (one per pole interval of ``g``), so the largest is
    the trigonometric closed form, held constant and Newton-polished once on
    ``g``: machine precision, and the implicit-function gradient. A bisection
    cost ~42 eager ``g`` evaluations per call, and the forces take ~19 calls.
    Inside points polish at ``t = 0`` (finite) and are masked to 0.
    """
    outside = (x / a + y / b + z / c) > 1.0
    B = a + b + c - x - y - z
    C = a * b + a * c + b * c - a * y - a * z - b * x - b * z - c * x - c * y
    D = a * b * c - a * b * z - a * c * y - b * c * x
    # t = s - B/3: s^3 + p s + q = 0 with p < 0 (three real roots)
    p = C - B**2 / 3.0
    q = 2.0 * B**3 / 27.0 - B * C / 3.0 + D
    m = xp.sqrt(xp.maximum(-p / 3.0, 1e-300 * xp.ones_like(p)))
    cos3 = xp.clip(-q / (2.0 * m**3), -1.0, 1.0)
    t0 = stop_gradient(2.0 * m * xp.cos(xp.acos(cos3) / 3.0) - B / 3.0)
    t0 = xp.where(outside, t0, xp.zeros_like(t0))
    fx0 = x / (a + t0) + y / (b + t0) + z / (c + t0) - 1.0
    dfx0 = -(x / (a + t0) ** 2 + y / (b + t0) ** 2 + z / (c + t0) ** 2)
    root = newton_polish(t0, fx0, dfx0, xp)
    return xp.where(outside, root, xp.zeros_like(root))
