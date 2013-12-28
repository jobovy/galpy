###############################################################################
#   BurkertPotential.py: Potential with a Burkert density
###############################################################################
import numpy
from scipy import special, integrate
from Potential import Potential
class BurkertPotential(Potential):
    """BurkertPotential.py: Potential with a Burkert density

                amp
    rho(r)= -------------      ; x = r/a
             (1+x)(1+x^2)
    """
    def __init__(self,amp=1.,a=1.,normalize=False):
        """
        NAME:
           __init__
        PURPOSE:
           initialize a Burkert-density potential
        INPUT:
           amp - amplitude to be applied to the potential (default: 1)
           a = scale radius
           normalize - if True, normalize such that vc(1.,0.)=1., or, if 
                       given as a number, such that the force is this fraction 
                       of the force necessary to make vc(1.,0.)=1.
        OUTPUT:
           (none)
        HISTORY:
           2013-04-10 - Written - Bovy (IAS)
        """
        Potential.__init__(self,amp=amp)
        self.a=a
        if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)):
            self.normalize(normalize)
        self.hasC= False

    def _evaluate(self,R,z,phi=0.,t=0.,dR=0,dphi=0):
        """
        NAME:
           _evaluate
        PURPOSE:
           evaluate the potential at R,z
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
           dR, dphi - return dR, dphi-th derivative (only implemented for 0 and 1)
        OUTPUT:
           Phi(R,z)
        HISTORY:
           2013-04-10 - Started - Bovy (IAS)
        """
        if dR == 0 and dphi == 0:
            x= numpy.sqrt(R**2.+z**2.)/self.a
            return -numpy.pi*self.a**2.*(2.*(1+x)/x*numpy.arctan(x)+2.*(1.+x)/x*numpy.log(1.+x)-(x-1.)/x*numpy.log(1.+x**2.))
        elif dR == 1 and dphi == 0:
            return -self._Rforce(R,z,phi=phi,t=t)
        elif dR == 0 and dphi == 1:
            return -self._phiforce(R,z,phi=phi,t=t)

    def _Rforce(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _Rforce
        PURPOSE:
           evaluate the radial force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the radial force
        HISTORY:
           2013-04-10 - Written - Bovy (IAS)
        """
        r= numpy.sqrt(R**2.+z**2.)
        x= r/self.a
        return -numpy.pi*self.a**3./r**2.*(2.*numpy.log(1.+x)+numpy.log(1.+x**2.)-2.*numpy.arctan(x))*R/r

    def _zforce(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _zforce
        PURPOSE:
           evaluate the vertical force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the vertical force
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        r= numpy.sqrt(R**2.+z**2.)
        x= r/self.a
        return -numpy.pi*self.a**3./r**2.*(2.*numpy.log(1.+x)+numpy.log(1.+x**2.)-2.*numpy.arctan(x))*z/r

    def _R2deriv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _Rderiv
        PURPOSE:
           evaluate the second radial derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the second radial derivative
        HISTORY:
           2011-10-09 - Written - Bovy (NYU)
        """
        r= numpy.sqrt(R**2.+z**2.)
        x= r/self.a
        return numpy.pi*((4.*x**3./(1.+x+x**2.+x**3.)+4.*numpy.arctan(x)-4.*numpy.log(1.+x)-2.*numpy.log(1.+x**2.))/x**3.*R**2./r**2.
                         +z**2./r**2.*self.a*(1./x**2.*(2.*numpy.log(1.+x)+numpy.log(1.+x**2.)-2.*numpy.arctan(x))))

    def _z2deriv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _z2deriv
        PURPOSE:
           evaluate the second vertical derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t- time
        OUTPUT:
           the second vertical derivative
        HISTORY:
           2012-07-26 - Written - Bovy (IAS@MPIA)
        """
        return self._R2deriv(z,R) #Spherical potential

    def _dens(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _dens
        PURPOSE:
           evaluate the density force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the density
        HISTORY:
           2013-01-09 - Written - Bovy (IAS)
        """
        r= numpy.sqrt(R**2.+z**2.)
        x= r/self.a
        return 1./(1.+x)/(1.+x**2.)

