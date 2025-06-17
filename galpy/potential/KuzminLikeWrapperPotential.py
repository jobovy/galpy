###############################################################################
#   KuzminLikeWrapperPotential.py: Wrapper to convert a potential to a Kuzmin
#   like potential (phi(r) --> phi(xi) where xi = sqrt(R^2 + (a + sqrt(z^2 + b^2))^2))
###############################################################################
import numpy

from ..util import conversion
from .Potential import (
    _evaluatePotentials,
    _evaluateRforces,
    _isNonAxi,
    evaluateR2derivs,
)
from .WrapperPotential import WrapperPotential


# Only implement 3D wrapper
class KuzminLikeWrapperPotential(WrapperPotential):
    """Wrapper to convert a spherical potential to a Kuzmin-like potential

    .. math::

        \\Phi(r) \\rightarrow \\Phi(\\xi)\\,,

    where

    .. math::

        \\xi = \\sqrt{R^2 + \\left(a + \\sqrt{z^2 + b^2}\\right)^2}\\,.

    Applying this wrapper to a ``KeplerPotential`` results in the ``KuzminDiskPotential`` (for :math:`b=0`) or the ``MiyamotoNagaiPotential`` (for :math:`b \\neq 0`).
    """

    def __init__(
        self,
        amp=1.0,
        a=1.1,
        b=0.0,
        pot=None,
        ro=None,
        vo=None,
    ):
        """
        Initialize a KuzminLikeWrapperPotential

        Parameters
        ----------
        amp : float, optional
            Overall amplitude to apply to the potential. Default is 1.0.
        a : float or Quantity, optional
            Scale radius of the Kuzmin-like potential. Default is 1.1.
        b : float or Quantity, optional
            Scale height of the Kuzmin-like potential. Default is 0.0.
        pot : Potential instance or list thereof
            The potential to be wrapped.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2024-01-15 - Written - Bovy (UofT)
        """
        WrapperPotential.__init__(self, amp=amp, pot=pot, ro=ro, vo=vo, _init=True)
        self._a = conversion.parse_length(a, ro=self._ro)
        self._scale = self._a
        self._b = conversion.parse_length(b, ro=self._ro)
        self._b2 = self._b**2.0
        if _isNonAxi(self._pot):
            raise RuntimeError(
                "KuzminLikeWrapperPotential only works for spherical or axisymmetric potentials"
            )
        self.hasC = True
        self.hasC_dxdv = True
        self.isNonAxi = False

    def _xi(self, R, z):
        return numpy.sqrt(R**2.0 + (self._a + numpy.sqrt(z**2.0 + self._b2)) ** 2.0)

    def _dxidR(self, R, z):
        return R / self._xi(R, z)

    def _dxidz(self, R, z):
        return (
            (self._a + numpy.sqrt(z**2.0 + self._b2))
            * z
            / self._xi(R, z)
            / numpy.sqrt(z**2.0 + self._b2)
        )

    def _d2xidR2(self, R, z):
        return ((self._a + numpy.sqrt(z**2.0 + self._b2)) ** 2.0) / self._xi(
            R, z
        ) ** 3.0

    def _d2xidz2(self, R, z):
        return (
            (
                self._a**3.0 * self._b2
                + 3.0 * self._a**2.0 * self._b2 * numpy.sqrt(self._b2 + z**2.0)
                + self._a * self._b2 * (3.0 * self._b2 + R**2.0 + 3.0 * z**2.0)
                + (self._b2 + R**2.0) * (self._b2 + z**2.0) ** (1.5)
            )
            / (self._b2 + z**2.0) ** 1.5
            / self._xi(R, z) ** 3.0
        )

    def _d2xidRdz(self, R, z):
        return -(R * z * (self._a + numpy.sqrt(self._b2 + z**2.0))) / (
            numpy.sqrt(self._b2 + z**2.0)
            * ((self._a + numpy.sqrt(self._b2 + z**2.0)) ** 2.0 + R**2.0) ** 1.5
        )

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        return _evaluatePotentials(self._pot, self._xi(R, z), 0.0, phi=phi, t=t)

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        return _evaluateRforces(
            self._pot, self._xi(R, z), 0.0, phi=phi, t=t
        ) * self._dxidR(R, z)

    def _zforce(self, R, z, phi=0.0, t=0.0):
        return _evaluateRforces(
            self._pot, self._xi(R, z), 0.0, phi=phi, t=t
        ) * self._dxidz(R, z)

    def _phitorque(self, R, z, phi=0.0, t=0.0):
        return 0.0

    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        return evaluateR2derivs(
            self._pot, self._xi(R, z), 0.0, phi=phi, t=t
        ) * self._dxidR(R, z) ** 2.0 - _evaluateRforces(
            self._pot, self._xi(R, z), 0.0, phi=phi, t=t
        ) * self._d2xidR2(R, z)

    def _z2deriv(self, R, z, phi=0.0, t=0.0):
        return evaluateR2derivs(
            self._pot, self._xi(R, z), 0.0, phi=phi, t=t
        ) * self._dxidz(R, z) ** 2.0 - _evaluateRforces(
            self._pot, self._xi(R, z), 0.0, phi=phi, t=t
        ) * self._d2xidz2(R, z)

    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        return evaluateR2derivs(
            self._pot, self._xi(R, z), 0.0, phi=phi, t=t
        ) * self._dxidR(R, z) * self._dxidz(R, z) - _evaluateRforces(
            self._pot, self._xi(R, z), 0.0, phi=phi, t=t
        ) * self._d2xidRdz(R, z)
