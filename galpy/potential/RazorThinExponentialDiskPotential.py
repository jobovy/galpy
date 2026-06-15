###############################################################################
#   RazorThinExponentialDiskPotential.py: class that implements the razor thin
#                                         exponential disk potential
#
#                                      rho(R,z) = rho_0 e^-R/h_R delta(z)
###############################################################################
import math

import numpy
from scipy import special

from ..backend import get_namespace
from ..backend import special as bspecial
from ..util import conversion
from .Potential import Potential


def _iv2(xp, y):
    # Modified Bessel I_2(y). The galpy.backend.special router exposes i0/i1 but
    # not iv, and neither jax nor torch has a native iv. On the numpy path use
    # scipy.special.iv directly (byte-identical); on jax/torch use the recurrence
    # I_2 = I_0 - (2/y) I_1, which is autodiff-friendly and agrees with scipy to
    # ~1e-12. y is strictly positive here (y = 0.5*alpha*R with R > 0 on this
    # branch), so 2/y is safe.
    if getattr(xp, "__name__", "") in ("numpy", "np"):
        return special.iv(2, y)
    return bspecial.i0(y) - 2.0 / y * bspecial.i1(y)


class RazorThinExponentialDiskPotential(Potential):
    """Class that implements the razor-thin exponential disk potential

    .. math::

        \\rho(R,z) = \\mathrm{amp}\\,\\exp\\left(-R/h_R\\right)\\,\\delta(z)

    """

    def __init__(
        self,
        amp=1.0,
        hr=1.0 / 3.0,
        normalize=False,
        ro=None,
        vo=None,
        new=True,
        glorder=100,
    ):
        """
        Class that implements a razor-thin exponential disk potential.

        Parameters
        ----------
        amp : float or Quantity, optional
            Amplitude to be applied to the potential (default: 1); can be a Quantity with units of surface-mass or Gxsurface-mass.
        hr : float or Quantity, optional
            Disk scale-length.
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).
        new : bool, optional
            If True, use a new implementation of the potential that is more accurate for small scale lengths (default: True).
        glorder : int, optional
            Gaussian quadrature order to use for numerical integration (default: 100).

        Notes
        -----
        - 2012-12-27 - Written - Bovy (IAS)
        """
        Potential.__init__(self, amp=amp, ro=ro, vo=vo, amp_units="surfacedensity")
        hr = conversion.parse_length(hr, ro=self._ro)
        self._new = new
        self._glorder = glorder
        self._hr = hr
        self._scale = self._hr
        self._alpha = 1.0 / self._hr
        self._glx, self._glw = numpy.polynomial.legendre.leggauss(self._glorder)
        self._backend_compatible = True
        if normalize or (
            isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
        ):  # pragma: no cover
            self.normalize(normalize)

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        if self._new:
            if xp.abs(z) < 10.0**-6.0:
                y = 0.5 * self._alpha * R
                return (
                    -math.pi
                    * R
                    * (
                        bspecial.i0(y) * bspecial.k1(y)
                        - bspecial.i1(y) * bspecial.k0(y)
                    )
                )
            kalphamax = 10.0
            # ks/weights are built from the float64 Gauss-Legendre nodes; move
            # them onto the active backend/device anchored on the inputs so that
            # ks + R etc. are same-namespace (a numpy ndarray + a torch tensor
            # raises). xp.asarray on the numpy path is a no-op (byte-identical).
            ks = xp.asarray(kalphamax * 0.5 * (self._glx + 1.0))
            weights = xp.asarray(kalphamax * self._glw)
            sqrtp = xp.sqrt(z**2.0 + (ks + R) ** 2.0)
            sqrtm = xp.sqrt(z**2.0 + (ks - R) ** 2.0)
            evalInt = (
                xp.arcsin(2.0 * ks / (sqrtp + sqrtm))
                * ks
                * bspecial.k0(self._alpha * ks)
            )
            return -2.0 * self._alpha * xp.sum(weights * evalInt)
        raise NotImplementedError(
            "Not new=True not implemented for RazorThinExponentialDiskPotential"
        )

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        if self._new:
            # if R > 6.: return self._kp(R,z)
            if xp.abs(z) < 10.0**-6.0:
                y = 0.5 * self._alpha * R
                return (
                    -2.0
                    * math.pi
                    * y
                    * (
                        bspecial.i0(y) * bspecial.k0(y)
                        - bspecial.i1(y) * bspecial.k1(y)
                    )
                )
            kalphamax1 = R
            ks1 = kalphamax1 * 0.5 * (self._glx + 1.0)
            weights1 = kalphamax1 * self._glw
            sqrtp = xp.sqrt(z**2.0 + (ks1 + R) ** 2.0)
            sqrtm = xp.sqrt(z**2.0 + (ks1 - R) ** 2.0)
            evalInt1 = (
                ks1**2.0
                * bspecial.k0(ks1 * self._alpha)
                * ((ks1 + R) / sqrtp - (ks1 - R) / sqrtm)
                / xp.sqrt(R**2.0 + z**2.0 - ks1**2.0 + sqrtp * sqrtm)
                / (sqrtp + sqrtm)
            )
            if R < 10.0:
                kalphamax2 = 10.0
                ks2 = (kalphamax2 - kalphamax1) * 0.5 * (self._glx + 1.0) + kalphamax1
                weights2 = (kalphamax2 - kalphamax1) * self._glw
                sqrtp = xp.sqrt(z**2.0 + (ks2 + R) ** 2.0)
                sqrtm = xp.sqrt(z**2.0 + (ks2 - R) ** 2.0)
                evalInt2 = (
                    ks2**2.0
                    * bspecial.k0(ks2 * self._alpha)
                    * ((ks2 + R) / sqrtp - (ks2 - R) / sqrtm)
                    / xp.sqrt(R**2.0 + z**2.0 - ks2**2.0 + sqrtp * sqrtm)
                    / (sqrtp + sqrtm)
                )
                return (
                    -2.0
                    * math.sqrt(2.0)
                    * self._alpha
                    * xp.sum(weights1 * evalInt1 + weights2 * evalInt2)
                )
            else:
                return -2.0 * math.sqrt(2.0) * self._alpha * xp.sum(weights1 * evalInt1)
        raise NotImplementedError(
            "Not new=True not implemented for RazorThinExponentialDiskPotential"
        )

    def _zforce(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        if self._new:
            # if R > 6.: return self._kp(R,z)
            if xp.abs(z) < 10.0**-6.0:
                return 0.0
            kalphamax1 = R
            ks1 = kalphamax1 * 0.5 * (self._glx + 1.0)
            weights1 = kalphamax1 * self._glw
            sqrtp = xp.sqrt(z**2.0 + (ks1 + R) ** 2.0)
            sqrtm = xp.sqrt(z**2.0 + (ks1 - R) ** 2.0)
            evalInt1 = (
                ks1**2.0
                * bspecial.k0(ks1 * self._alpha)
                * (1.0 / sqrtp + 1.0 / sqrtm)
                / xp.sqrt(R**2.0 + z**2.0 - ks1**2.0 + sqrtp * sqrtm)
                / (sqrtp + sqrtm)
            )
            if R < 10.0:
                kalphamax2 = 10.0
                ks2 = (kalphamax2 - kalphamax1) * 0.5 * (self._glx + 1.0) + kalphamax1
                weights2 = (kalphamax2 - kalphamax1) * self._glw
                sqrtp = xp.sqrt(z**2.0 + (ks2 + R) ** 2.0)
                sqrtm = xp.sqrt(z**2.0 + (ks2 - R) ** 2.0)
                evalInt2 = (
                    ks2**2.0
                    * bspecial.k0(ks2 * self._alpha)
                    * (1.0 / sqrtp + 1.0 / sqrtm)
                    / xp.sqrt(R**2.0 + z**2.0 - ks2**2.0 + sqrtp * sqrtm)
                    / (sqrtp + sqrtm)
                )
                return (
                    -z
                    * 2.0
                    * math.sqrt(2.0)
                    * self._alpha
                    * xp.sum(weights1 * evalInt1 + weights2 * evalInt2)
                )
            else:
                return (
                    -z
                    * 2.0
                    * math.sqrt(2.0)
                    * self._alpha
                    * xp.sum(weights1 * evalInt1)
                )
        raise NotImplementedError(
            "Not new=True not implemented for RazorThinExponentialDiskPotential"
        )

    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        if self._new:
            if xp.abs(z) < 10.0**-6.0:
                y = 0.5 * self._alpha * R
                return math.pi * self._alpha * (
                    bspecial.i0(y) * bspecial.k0(y) - bspecial.i1(y) * bspecial.k1(y)
                ) + math.pi / 4.0 * self._alpha**2.0 * R * (
                    bspecial.i1(y) * (3.0 * bspecial.k0(y) + bspecial.kn(2, y))
                    - bspecial.k1(y) * (3.0 * bspecial.i0(y) + _iv2(xp, y))
                )
            raise AttributeError(
                "'R2deriv' for RazorThinExponentialDisk not implemented for z =/= 0"
            )

    def _z2deriv(self, R, z, phi=0.0, t=0.0):  # pragma: no cover
        return math.inf

    def _surfdens(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        return xp.exp(-self._alpha * R)

    def _mass(self, R, z=None, t=0.0):
        xp = get_namespace(R)
        return (
            2.0
            * math.pi
            * (1.0 - xp.exp(-self._alpha * R) * (1.0 + self._alpha * R))
            / self._alpha**2.0
        )
