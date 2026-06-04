###############################################################################
#   SoftenedNeedleBarPotential.py: class that implements the softened needle
#                                  bar potential from Long & Murali (1992)
###############################################################################
import hashlib
import math

import numpy

from ..backend import get_namespace
from ..util import conversion
from .Potential import Potential


class SoftenedNeedleBarPotential(Potential):
    """Class that implements the softened needle bar potential from `Long & Murali (1992) <http://adsabs.harvard.edu/abs/1992ApJ...397...44L>`__

    .. math::

        \\Phi(x,y,z) = \\frac{\\mathrm{amp}}{2a}\\,\\ln\\left(\\frac{x-a+T_-}{x+a+T_+}\\right)

    where

    .. math::

        T_{\\pm} = \\sqrt{(a\\pm x)^2 + y^2+(b+\\sqrt{z^2+c^2})^2}

    For a prolate bar, set :math:`b` to zero.

    """

    def __init__(
        self,
        amp=1.0,
        a=4.0,
        b=0.0,
        c=1.0,
        normalize=False,
        pa=0.4,
        omegab=1.8,
        ro=None,
        vo=None,
    ):
        """
        Initialize a softened-needle bar potential.

        Parameters
        ----------
        amp : float or Quantity, optional
            Amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass.
        a : float or Quantity, optional
            Bar half-length.
        b : float , optional
            Triaxial softening length (can be Quantity).
        c : float, optional
            Prolate softening length (can be Quantity).
        pa : float or Quantity, optional
            The position angle of the x axis.
        omegab : float or Quantity, optional
            Pattern speed.
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2016-11-02 - Started - Bovy (UofT)

        """
        Potential.__init__(self, amp=amp, ro=ro, vo=vo, amp_units="mass")
        a = conversion.parse_length(a, ro=self._ro)
        b = conversion.parse_length(b, ro=self._ro)
        c = conversion.parse_length(c, ro=self._ro)
        pa = conversion.parse_angle(pa)
        omegab = conversion.parse_frequency(omegab, ro=self._ro, vo=self._vo)
        self._a = a
        self._b = b
        self._c2 = c**2.0
        self._pa = pa
        self._omegab = omegab
        self._force_hash = None
        self.hasC = True
        self.hasC_dxdv = False
        if normalize or (
            isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
        ):  # pragma: no cover
            self.normalize(normalize)
        self.isNonAxi = True
        return None

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z, phi)
        x, y, z = self._compute_xyz(R, phi, z, t)
        Tp, Tm = self._compute_TpTm(x, y, z)
        return xp.log((x - self._a + Tm) / (x + self._a + Tp)) / 2.0 / self._a

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z, phi)
        Fx, Fy, _ = self._cached_xyzforces(R, z, phi, t, xp)
        return xp.cos(phi) * Fx + xp.sin(phi) * Fy

    def _phitorque(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z, phi)
        Fx, Fy, _ = self._cached_xyzforces(R, z, phi, t, xp)
        return R * (-xp.sin(phi) * Fx + xp.cos(phi) * Fy)

    def _zforce(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z, phi)
        _, _, Fz = self._cached_xyzforces(R, z, phi, t, xp)
        return Fz

    def OmegaP(self):
        return self._omegab

    def _compute_xyz(self, R, phi, z, t):
        xp = get_namespace(R, z, phi)
        ang = phi - self._pa - self._omegab * t
        return (R * xp.cos(ang), R * xp.sin(ang), z)

    def _compute_TpTm(self, x, y, z):
        xp = get_namespace(x, y, z)
        secondpart = y**2.0 + (self._b + xp.sqrt(self._c2 + z**2.0)) ** 2.0
        return (
            xp.sqrt((self._a + x) ** 2.0 + secondpart),
            xp.sqrt((self._a - x) ** 2.0 + secondpart),
        )

    def _xyzforces(self, R, z, phi, t):
        # Pure-functional rectangular (galactocentric, de-rotated) forces; no
        # per-instance state, so it is safe under jax/torch tracing.
        x, y, z = self._compute_xyz(R, phi, z, t)
        Tp, Tm = self._compute_TpTm(x, y, z)
        Fx = self._xforce_xyz(x, y, z, Tp, Tm)
        Fy = self._yforce_xyz(x, y, z, Tp, Tm)
        Fz = self._zforce_xyz(x, y, z, Tp, Tm)
        # de-rotation angle depends on the time t. For a scalar t, math.cos/sin
        # keep it a plain coefficient that broadcasts cleanly against any
        # backend's forces; for an array t (numpy), use that array's namespace
        # so the rotation broadcasts element-wise.
        tp = self._pa + self._omegab * t
        if getattr(tp, "ndim", 0) > 0:
            txp = get_namespace(tp)
            cp, sp = txp.cos(tp), txp.sin(tp)
        else:
            cp, sp = math.cos(tp), math.sin(tp)
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

    def _xforce_xyz(self, x, y, z, Tp, Tm):
        return -2.0 * x / Tp / Tm / (Tp + Tm)

    def _yforce_xyz(self, x, y, z, Tp, Tm):
        xp = get_namespace(x, y, z)
        return (
            -y
            / 2.0
            / Tp
            / Tm
            * (Tp + Tm - 4.0 * x**2.0 / (Tp + Tm))
            / (y**2.0 + (self._b + xp.sqrt(z**2.0 + self._c2)) ** 2.0)
        )

    def _zforce_xyz(self, x, y, z, Tp, Tm):
        xp = get_namespace(x, y, z)
        zc = xp.sqrt(z**2.0 + self._c2)
        return (
            -z
            / 2.0
            / Tp
            / Tm
            * (Tp + Tm - 4.0 * x**2.0 / (Tp + Tm))
            / (y**2.0 + (self._b + zc) ** 2.0)
            * (self._b + zc)
            / zc
        )

    def _dens(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z, phi)
        x, y, z = self._compute_xyz(R, phi, z, t)
        zc = xp.sqrt(z**2.0 + self._c2)
        bzc2 = (self._b + zc) ** 2.0
        bigA = self._b * y**2.0 + (self._b + 3.0 * zc) * bzc2
        bigC = y**2.0 + bzc2
        return (
            self._c2
            / 24.0
            / math.pi
            / self._a
            / bigC**2.0
            / zc**3.0
            * (
                (x + self._a)
                * (
                    3.0 * bigA * bigC
                    + (2.0 * bigA + self._b * bigC) * (x + self._a) ** 2.0
                )
                / (bigC + (x + self._a) ** 2.0) ** 1.5
                - (x - self._a)
                * (
                    3.0 * bigA * bigC
                    + (2.0 * bigA + self._b * bigC) * (x - self._a) ** 2.0
                )
                / (bigC + (x - self._a) ** 2.0) ** 1.5
            )
        )
