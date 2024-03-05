###############################################################################
#   BurkertPotential.py: Potential with a Burkert density
###############################################################################
import numpy
from scipy import special

from ..util import conversion
from .SphericalPotential import SphericalPotential


class BurkertPotential(SphericalPotential):
    """BurkertPotential.py: Potential with a Burkert density

    .. math::

        \\rho(r) = \\frac{\\mathrm{amp}}{(1+r/a)\\,(1+[r/a]^2)}

    """

    def __init__(self, amp=1.0, a=2.0, normalize=False, ro=None, vo=None):
        """
        Initialize a Burkert-density potential [1]_.

        Parameters
        ----------
        amp : float or Quantity
            Amplitude to be applied to the potential. Can be a Quantity with units of mass density or Gxmass density.
        a : float or Quantity
            Scale radius.
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1. Default is False.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2013-04-10 - Written - Bovy (IAS)
        - 2020-03-30 - Re-implemented using SphericalPotential - Bovy (UofT)

        References
        ----------
        .. [1] Burkert (1995), Astrophysical Journal, 447, L25. ADS: https://ui.adsabs.harvard.edu/abs/1995ApJ...447L..25B.
        """
        SphericalPotential.__init__(self, amp=amp, ro=ro, vo=vo, amp_units="density")
        a = conversion.parse_length(a, ro=self._ro, vo=self._vo)
        self.a = a
        self._scale = self.a
        if normalize or (
            isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
        ):  # pragma: no cover
            self.normalize(normalize)
        self.hasC = True
        self.hasC_dxdv = True
        self.hasC_dens = True
        return None

    def _revaluate(self, r, t=0.0):
        """Potential as a function of r and time"""
        x = r / self.a
        return (
            -(self.a**2.0)
            * numpy.pi
            * (
                -numpy.pi / x
                + 2.0 * (1.0 / x + 1) * numpy.arctan(1 / x)
                + (1.0 / x + 1) * numpy.log((1.0 + 1.0 / x) ** 2.0 / (1.0 + 1 / x**2.0))
                + special.xlogy(2.0 / x, 1.0 + x**2.0)
            )
        )

    # Previous way, not stable as r -> infty
    # return -self.a**2.*numpy.pi/x*(-numpy.pi+2.*(1.+x)*numpy.arctan(1/x)
    #                                +2.*(1.+x)*numpy.log(1.+x)
    #                                +(1.-x)*numpy.log(1.+x**2.))

    def _rforce(self, r, t=0.0):
        x = r / self.a
        return (
            self.a
            * numpy.pi
            / x**2.0
            * (
                numpy.pi
                - 2.0 * numpy.arctan(1.0 / x)
                - 2.0 * numpy.log(1.0 + x)
                - numpy.log(1.0 + x**2.0)
            )
        )

    def _r2deriv(self, r, t=0.0):
        x = r / self.a
        return (
            4.0 * numpy.pi / (1.0 + x**2.0) / (1.0 + x)
            + 2.0 * self._rforce(r) / x / self.a
        )

    def _rdens(self, r, t=0.0):
        x = r / self.a
        return 1.0 / (1.0 + x) / (1.0 + x**2.0)

    def _surfdens(self, R, z, phi=0.0, t=0.0):
        r = numpy.sqrt(R**2.0 + z**2.0)
        x = r / self.a
        Rpa = numpy.sqrt(R**2.0 + self.a**2.0)
        Rma = numpy.sqrt(R**2.0 - self.a**2.0 + 0j)
        if Rma == 0:
            za = z / self.a
            return (
                self.a**2.0
                / 2.0
                * (
                    (
                        2.0
                        - 2.0 * numpy.sqrt(za**2.0 + 1)
                        + numpy.sqrt(2.0) * za * numpy.arctan(za / numpy.sqrt(2.0))
                    )
                    / z
                    + numpy.sqrt(2 * za**2.0 + 2.0)
                    * numpy.arctanh(za / numpy.sqrt(2.0 * (za**2.0 + 1)))
                    / numpy.sqrt(self.a**2.0 + z**2.0)
                )
            )
        else:
            return (
                self.a**2.0
                * (
                    numpy.arctan(z / x / Rma) / Rma
                    + numpy.arctanh(z / x / Rpa) / Rpa
                    - numpy.arctan(z / Rma) / Rma
                    + numpy.arctan(z / Rpa) / Rpa
                ).real
            )
