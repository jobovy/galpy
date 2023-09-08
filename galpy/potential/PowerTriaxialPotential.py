###############################################################################
#   PowerTriaxialPotential: Potential of a triaxial power-law
#
#                                        amp
#                          rho(x,y,z)= ---------
#                                       m^\alpha
#
#                                 with m^2 = x^2+y^2/b^2+z^2/c^2
#
###############################################################################
import numpy

from ..util import conversion
from .EllipsoidalPotential import EllipsoidalPotential


class PowerTriaxialPotential(EllipsoidalPotential):
    """Class that implements triaxial potentials that are derived from power-law density models (including an elliptical power law)

    .. math::

        \\rho(r) = \\frac{\\mathrm{amp}}{r_1^3}\\,\\left(\\frac{r_1}{m}\\right)^{\\alpha}

    where :math:`m^2 = x^2+y^2/b^2+z^2/c^2`.
    """

    def __init__(
        self,
        amp=1.0,
        alpha=1.0,
        r1=1.0,
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
        Initialize a triaxial power-law potential.

        Parameters
        ----------
        amp : float or Quantity, optional
            Amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass or Gxmass.
        alpha : float
            Power-law exponent.
        r1 : float or Quantity, optional
            Reference radius for amplitude.
        b : float
            Y-to-x axis ratio of the density.
        c : float
            Z-to-x axis ratio of the density.
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

        Notes
        -----
        - 2021-05-07 - Started - Bovy (UofT)

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
        r1 = conversion.parse_length(r1, ro=self._ro)
        self.alpha = alpha
        # Back to old definition
        if self.alpha != 3.0:
            self._amp *= r1 ** (self.alpha - 3.0) * 4.0 * numpy.pi / (3.0 - self.alpha)
        # Multiply in constants
        self._amp *= (3.0 - self.alpha) / 4.0 / numpy.pi
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
        return 2.0 / (2.0 - self.alpha) * m ** (2.0 - self.alpha)

    def _mdens(self, m):
        """Density as a function of m"""
        return m**-self.alpha

    def _mdens_deriv(self, m):
        """Derivative of the density as a function of m"""
        return -self.alpha * m ** -(1.0 + self.alpha)
