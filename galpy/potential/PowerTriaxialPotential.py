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
    def __init__(self,amp=1.,alpha=1.,r1=1.,b=1.,c=1.,
                 zvec=None,pa=None,glorder=50,
                 normalize=False,ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a triaxial power-law potential

        INPUT:

           amp - amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass or Gxmass

           alpha - power-law exponent

           r1= (1.) reference radius for amplitude (can be Quantity)

           b - y-to-x axis ratio of the density

           c - z-to-x axis ratio of the density

           zvec= (None) If set, a unit vector that corresponds to the z axis

           pa= (None) If set, the position angle of the x axis (rad or Quantity)

           glorder= (50) if set, compute the relevant force and potential integrals with Gaussian quadrature of this order

           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

        OUTPUT:

           (none)

        HISTORY:

           2021-05-07 - Started - Bovy (UofT)

        """
        EllipsoidalPotential.__init__(self,amp=amp,b=b,c=c,
                                      zvec=zvec,pa=pa,glorder=glorder,
                                      ro=ro,vo=vo,amp_units='mass')
        r1= conversion.parse_length(r1,ro=self._ro)
        self.alpha= alpha
        # Back to old definition
        if self.alpha != 3.:
            self._amp*= r1**(self.alpha-3.)*4.*numpy.pi/(3.-self.alpha)
        # Multiply in constants
        self._amp*= (3.-self.alpha)/4./numpy.pi
        if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)): #pragma: no cover
            self.normalize(normalize)
        self.hasC= not self._glorder is None
        self.hasC_dxdv= False
        self.hasC_dens= self.hasC # works if mdens is defined, necessary for hasC
        return None

    def _psi(self,m):
        """\psi(m) = -\int_m^\infty d m^2 \rho(m^2)"""
        return 2./(2.-self.alpha)*m**(2.-self.alpha)

    def _mdens(self,m):
        """Density as a function of m"""
        return m**-self.alpha

    def _mdens_deriv(self,m):
        """Derivative of the density as a function of m"""
        return -self.alpha*m**-(1.+self.alpha)
