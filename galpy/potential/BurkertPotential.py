###############################################################################
#   BurkertPotential.py: Potential with a Burkert density
###############################################################################
import numpy
from scipy import special
from .SphericalPotential import SphericalPotential
from ..util import conversion
class BurkertPotential(SphericalPotential):
    """BurkertPotential.py: Potential with a Burkert density

    .. math::

        \\rho(r) = \\frac{\\mathrm{amp}}{(1+r/a)\\,(1+[r/a]^2)}

    """
    def __init__(self,amp=1.,a=2.,normalize=False,
                 ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a Burkert-density potential

        INPUT:

           amp - amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass density or Gxmass density

           a = scale radius (can be Quantity)

           normalize - if True, normalize such that vc(1.,0.)=1., or, if 
                       given as a number, such that the force is this fraction 
                       of the force necessary to make vc(1.,0.)=1.

           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

        OUTPUT:

           (none)

        HISTORY:

           2013-04-10 - Written - Bovy (IAS)

           2020-03-30 - Re-implemented using SphericalPotential - Bovy (UofT)

        """
        SphericalPotential.__init__(self,amp=amp,ro=ro,vo=vo,
                                    amp_units='density')
        a= conversion.parse_length(a,ro=self._ro,vo=self._vo)
        self.a=a
        self._scale= self.a
        if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)): #pragma: no cover 
            self.normalize(normalize)
        self.hasC= True
        self.hasC_dxdv= True
        self.hasC_dens= True
        return None

    def _revaluate(self,r,t=0.):
        """Potential as a function of r and time"""
        x= r/self.a
        return -self.a**2.*numpy.pi*(-numpy.pi/x+2.*(1./x+1)*numpy.arctan(1/x)
                                +(1./x+1)*numpy.log((1.+1./x)**2./(1.+1/x**2.))
                                     +special.xlogy(2./x,1.+x**2.))
    #Previous way, not stable as r -> infty
    #return -self.a**2.*numpy.pi/x*(-numpy.pi+2.*(1.+x)*numpy.arctan(1/x)
    #                                +2.*(1.+x)*numpy.log(1.+x)
    #                                +(1.-x)*numpy.log(1.+x**2.))

    def _rforce(self,r,t=0.):
        x= r/self.a
        return self.a*numpy.pi/x**2.*(numpy.pi-2.*numpy.arctan(1./x)
                                      -2.*numpy.log(1.+x)-numpy.log(1.+x**2.))
    
    def _r2deriv(self,r,t=0.):
        x= r/self.a
        return 4.*numpy.pi/(1.+x**2.)/(1.+x)+2.*self._rforce(r)/x/self.a

    def _rdens(self,r,t=0.):
        x= r/self.a
        return 1./(1.+x)/(1.+x**2.)

    def _surfdens(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _surfdens
        PURPOSE:
           evaluate the surface density for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the surface density
        HISTORY:
           2018-08-19 - Written - Bovy (UofT)
        """
        r= numpy.sqrt(R**2.+z**2.)
        x= r/self.a
        Rpa= numpy.sqrt(R**2.+self.a**2.)
        Rma= numpy.sqrt(R**2.-self.a**2.+0j)
        if Rma == 0:
            za= z/self.a
            return self.a**2./2.*((2.-2.*numpy.sqrt(za**2.+1)
                                   +numpy.sqrt(2.)*za\
                                       *numpy.arctan(za/numpy.sqrt(2.)))/z
                                  +numpy.sqrt(2*za**2.+2.)\
                                   *numpy.arctanh(za/numpy.sqrt(2.*(za**2.+1)))
                                  /numpy.sqrt(self.a**2.+z**2.))
        else:
            return self.a**2.*(numpy.arctan(z/x/Rma)/Rma
                               +numpy.arctanh(z/x/Rpa)/Rpa
                               -numpy.arctan(z/Rma)/Rma
                               +numpy.arctan(z/Rpa)/Rpa).real
