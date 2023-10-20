###############################################################################
#   TriaxialGaussianPotential.py: Potential of a triaxial Gaussian stratified
#                                 on similar ellipsoids
#
#                                 \rho(x,y,z) ~ exp(-m^2/[2\sigma^2])
#
#                                 with m^2 = x^2+y^2/b^2+z^2/c^2
#
###############################################################################
import numpy
from scipy import special

from ..util import conversion
from .EllipsoidalPotential import EllipsoidalPotential


class TriaxialGaussianPotential(EllipsoidalPotential):
    """Potential of a triaxial Gaussian (`Emsellem et al. 1994 <https://ui.adsabs.harvard.edu/abs/1994A%26A...285..723E/abstract>`__):

    .. math::

        \\rho(x,y,z) = \\frac{\\mathrm{amp}}{(2\\pi\\,\\sigma)^{3/2}\\,b\\,c}\\,e^{-\\frac{m^2}{2\\sigma^2}}

    where :math:`\\mathrm{amp} = GM` is the total mass and :math:`m^2 = x^2+y^2/b^2+z^2/c^2`.
    """

    def __init__(
        self,
        amp=1.0,
        sigma=5.0,
        b=1.0,
        c=1.0,
        zvec=None,
        pa=None,
        glorder=50,
        normalize=False,
        ro=None,
        vo=None,
    ):
        """
        Initialize a triaxial Gaussian potential.

        Parameters
        ----------
        amp : float or Quantity, optional
            Amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass or Gxmass.
        sigma : float or Quantity, optional
            Gaussian dispersion scale.
        b : float, optional
            y-to-x axis ratio of the density.
        c : float, optional
            z-to-x axis ratio of the density.
        zvec : numpy.ndarray, optional
            If set, a unit vector that corresponds to the z axis.
        pa : float or Quantity, optional
            If set, the position angle of the x axis.
        glorder : int, optional
            If set, compute the relevant force and potential integrals with Gaussian quadrature of this order.
        normalize : bool or float, optional
            If True, normalize the potential (default: False). If a float, normalize the potential to this value.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2020-08-18 - Started - Bovy (UofT)

        """
        EllipsoidalPotential.__init__(
            self,
            amp=amp,
            b=b,
            c=c,
            zvec=zvec,
            pa=pa,
            glorder=glorder,
            ro=ro,
            vo=vo,
            amp_units="mass",
        )
        sigma = conversion.parse_length(sigma, ro=self._ro)
        self._sigma = sigma
        self._twosigma2 = 2.0 * self._sigma**2
        self._scale = self._sigma
        # Adjust amp
        self._amp /= (2.0 * numpy.pi) ** 1.5 * self._sigma**3.0 * self._b * self._c
        if normalize or (
            isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
        ):  # pragma: no cover
            self.normalize(normalize)
        self.hasC = not self._glorder is None
        self.hasC_dxdv = False
        self.hasC_dens = self.hasC  # works if mdens is defined, necessary for hasC
        return None

    def _psi(self, m):
        """\\psi(m) = -\\int_m^\\infty d m^2 \rho(m^2)"""
        return -self._twosigma2 * numpy.exp(-(m**2.0) / self._twosigma2)

    def _mdens(self, m):
        """Density as a function of m"""
        return numpy.exp(-(m**2) / self._twosigma2)

    def _mdens_deriv(self, m):
        """Derivative of the density as a function of m"""
        return -2.0 * m * numpy.exp(-(m**2) / self._twosigma2) / self._twosigma2

    def _mass(self, R, z=None, t=0.0):
        if not z is None:
            raise AttributeError  # Hack to fall back to general
        return (
            numpy.pi
            * self._b
            * self._c
            * self._twosigma2
            * self._sigma
            * (
                numpy.sqrt(2.0 * numpy.pi)
                * special.erf(R / self._sigma / numpy.sqrt(2.0))
                - 2.0 * R / self._sigma * numpy.exp(-(R**2.0) / self._twosigma2)
            )
        )
