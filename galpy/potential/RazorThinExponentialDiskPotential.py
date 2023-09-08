###############################################################################
#   RazorThinExponentialDiskPotential.py: class that implements the razor thin
#                                         exponential disk potential
#
#                                      rho(R,z) = rho_0 e^-R/h_R delta(z)
###############################################################################
import numpy
from scipy import special

from ..util import conversion
from .Potential import Potential


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
        if normalize or (
            isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
        ):  # pragma: no cover
            self.normalize(normalize)

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        if self._new:
            if numpy.fabs(z) < 10.0**-6.0:
                y = 0.5 * self._alpha * R
                return (
                    -numpy.pi
                    * R
                    * (special.i0(y) * special.k1(y) - special.i1(y) * special.k0(y))
                )
            kalphamax = 10.0
            ks = kalphamax * 0.5 * (self._glx + 1.0)
            weights = kalphamax * self._glw
            sqrtp = numpy.sqrt(z**2.0 + (ks + R) ** 2.0)
            sqrtm = numpy.sqrt(z**2.0 + (ks - R) ** 2.0)
            evalInt = (
                numpy.arcsin(2.0 * ks / (sqrtp + sqrtm))
                * ks
                * special.k0(self._alpha * ks)
            )
            return -2.0 * self._alpha * numpy.sum(weights * evalInt)
        raise NotImplementedError(
            "Not new=True not implemented for RazorThinExponentialDiskPotential"
        )

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        if self._new:
            # if R > 6.: return self._kp(R,z)
            if numpy.fabs(z) < 10.0**-6.0:
                y = 0.5 * self._alpha * R
                return (
                    -2.0
                    * numpy.pi
                    * y
                    * (special.i0(y) * special.k0(y) - special.i1(y) * special.k1(y))
                )
            kalphamax1 = R
            ks1 = kalphamax1 * 0.5 * (self._glx + 1.0)
            weights1 = kalphamax1 * self._glw
            sqrtp = numpy.sqrt(z**2.0 + (ks1 + R) ** 2.0)
            sqrtm = numpy.sqrt(z**2.0 + (ks1 - R) ** 2.0)
            evalInt1 = (
                ks1**2.0
                * special.k0(ks1 * self._alpha)
                * ((ks1 + R) / sqrtp - (ks1 - R) / sqrtm)
                / numpy.sqrt(R**2.0 + z**2.0 - ks1**2.0 + sqrtp * sqrtm)
                / (sqrtp + sqrtm)
            )
            if R < 10.0:
                kalphamax2 = 10.0
                ks2 = (kalphamax2 - kalphamax1) * 0.5 * (self._glx + 1.0) + kalphamax1
                weights2 = (kalphamax2 - kalphamax1) * self._glw
                sqrtp = numpy.sqrt(z**2.0 + (ks2 + R) ** 2.0)
                sqrtm = numpy.sqrt(z**2.0 + (ks2 - R) ** 2.0)
                evalInt2 = (
                    ks2**2.0
                    * special.k0(ks2 * self._alpha)
                    * ((ks2 + R) / sqrtp - (ks2 - R) / sqrtm)
                    / numpy.sqrt(R**2.0 + z**2.0 - ks2**2.0 + sqrtp * sqrtm)
                    / (sqrtp + sqrtm)
                )
                return (
                    -2.0
                    * numpy.sqrt(2.0)
                    * self._alpha
                    * numpy.sum(weights1 * evalInt1 + weights2 * evalInt2)
                )
            else:
                return (
                    -2.0
                    * numpy.sqrt(2.0)
                    * self._alpha
                    * numpy.sum(weights1 * evalInt1)
                )
        raise NotImplementedError(
            "Not new=True not implemented for RazorThinExponentialDiskPotential"
        )

    def _zforce(self, R, z, phi=0.0, t=0.0):
        if self._new:
            # if R > 6.: return self._kp(R,z)
            if numpy.fabs(z) < 10.0**-6.0:
                return 0.0
            kalphamax1 = R
            ks1 = kalphamax1 * 0.5 * (self._glx + 1.0)
            weights1 = kalphamax1 * self._glw
            sqrtp = numpy.sqrt(z**2.0 + (ks1 + R) ** 2.0)
            sqrtm = numpy.sqrt(z**2.0 + (ks1 - R) ** 2.0)
            evalInt1 = (
                ks1**2.0
                * special.k0(ks1 * self._alpha)
                * (1.0 / sqrtp + 1.0 / sqrtm)
                / numpy.sqrt(R**2.0 + z**2.0 - ks1**2.0 + sqrtp * sqrtm)
                / (sqrtp + sqrtm)
            )
            if R < 10.0:
                kalphamax2 = 10.0
                ks2 = (kalphamax2 - kalphamax1) * 0.5 * (self._glx + 1.0) + kalphamax1
                weights2 = (kalphamax2 - kalphamax1) * self._glw
                sqrtp = numpy.sqrt(z**2.0 + (ks2 + R) ** 2.0)
                sqrtm = numpy.sqrt(z**2.0 + (ks2 - R) ** 2.0)
                evalInt2 = (
                    ks2**2.0
                    * special.k0(ks2 * self._alpha)
                    * (1.0 / sqrtp + 1.0 / sqrtm)
                    / numpy.sqrt(R**2.0 + z**2.0 - ks2**2.0 + sqrtp * sqrtm)
                    / (sqrtp + sqrtm)
                )
                return (
                    -z
                    * 2.0
                    * numpy.sqrt(2.0)
                    * self._alpha
                    * numpy.sum(weights1 * evalInt1 + weights2 * evalInt2)
                )
            else:
                return (
                    -z
                    * 2.0
                    * numpy.sqrt(2.0)
                    * self._alpha
                    * numpy.sum(weights1 * evalInt1)
                )
        raise NotImplementedError(
            "Not new=True not implemented for RazorThinExponentialDiskPotential"
        )

    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        if self._new:
            if numpy.fabs(z) < 10.0**-6.0:
                y = 0.5 * self._alpha * R
                return numpy.pi * self._alpha * (
                    special.i0(y) * special.k0(y) - special.i1(y) * special.k1(y)
                ) + numpy.pi / 4.0 * self._alpha**2.0 * R * (
                    special.i1(y) * (3.0 * special.k0(y) + special.kn(2, y))
                    - special.k1(y) * (3.0 * special.i0(y) + special.iv(2, y))
                )
            raise AttributeError(
                "'R2deriv' for RazorThinExponentialDisk not implemented for z =/= 0"
            )

    def _z2deriv(self, R, z, phi=0.0, t=0.0):  # pragma: no cover
        return numpy.infty

    def _surfdens(self, R, z, phi=0.0, t=0.0):
        return numpy.exp(-self._alpha * R)

    def _mass(self, R, z=None, t=0.0):
        return (
            2.0
            * numpy.pi
            * (1.0 - numpy.exp(-self._alpha * R) * (1.0 + self._alpha * R))
            / self._alpha**2.0
        )
