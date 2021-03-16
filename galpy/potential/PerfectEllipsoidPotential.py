###############################################################################
#   PerfectEllipsoidPotential.py: Potential of the perfect ellipsoid 
#                                 (de Zeeuw 1985):
#
#                                 \rho(x,y,z) ~ 1/(1+m^2)^2
#
#                                 with m^2 = x^2+y^2/b^2+z^2/c^2
#
###############################################################################
import numpy
from ..util import conversion
from .EllipsoidalPotential import EllipsoidalPotential
class PerfectEllipsoidPotential(EllipsoidalPotential):
    """Potential of the perfect ellipsoid (de Zeeuw 1985):

    .. math::

        \\rho(x,y,z) = \\frac{\\mathrm{amp\,a}}{\\pi^2\\,bc}\\,\\frac{1}{(m^2+a^2)^2}

    where :math:`\\mathrm{amp} = GM` is the total mass and :math:`m^2 = x^2+y^2/b^2+z^2/c^2`.
    """
    def __init__(self,amp=1.,a=5.,b=1.,c=1.,
                 zvec=None,pa=None,glorder=50,
                 normalize=False,ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a perfect ellipsoid potential

        INPUT:

           amp - amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass or G x mass

           a - scale radius (can be Quantity)

           b - y-to-x axis ratio of the density

           c - z-to-x axis ratio of the density

           zvec= (None) If set, a unit vector that corresponds to the z axis

           pa= (None) If set, the position angle of the x axis (rad or Quantity)

           glorder= (50) if set, compute the relevant force and potential integrals with Gaussian quadrature of this order

           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

        OUTPUT:

           (none)

        HISTORY:

           2018-08-06 - Started - Bovy (UofT)

        """
        EllipsoidalPotential.__init__(self,amp=amp,b=b,c=c,
                                      zvec=zvec,pa=pa,glorder=glorder,
                                      ro=ro,vo=vo,amp_units='mass')
        a= conversion.parse_length(a,ro=self._ro)
        self.a= a
        self.a2= self.a**2
        self._scale= self.a
        # Adjust amp
        self._amp*= self.a/(numpy.pi**2*self._b*self._c)
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
        return -1./(self.a2+m**2)

    def _mdens(self,m):
        """Density as a function of m"""
        return (self.a2+m**2)**-2

    def _mdens_deriv(self,m):
        """Derivative of the density as a function of m"""
        return -4.*m*(self.a2+m**2)**-3

    def _mass(self,R,z=None,t=0.):
        """
        NAME:
           _mass
        PURPOSE:
           evaluate the mass within R (and z) for this potential; if z=None, integrate to ellipsoidal boundary
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           t - time
        OUTPUT:
           the mass enclosed
        HISTORY:
           2021-03-08 - Written - Bovy (UofT)
        """
        if not z is None: raise AttributeError # Hack to fall back to general
        return 2.*numpy.pi*self._b*self._c/self.a\
            *(numpy.arctan(R/self.a)-R*self.a/(1.+R**2.))

    
