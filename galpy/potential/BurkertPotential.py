###############################################################################
#   BurkertPotential.py: Potential with a Burkert density
###############################################################################
import numpy
from .Potential import Potential, _APY_LOADED
if _APY_LOADED:
    from astropy import units
class BurkertPotential(Potential):
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

        """
        Potential.__init__(self,amp=amp,ro=ro,vo=vo,amp_units='density')
        if _APY_LOADED and isinstance(a,units.Quantity):
            a= a.to(units.kpc).value/self._ro
        self.a=a
        self._scale= self.a
        if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)): #pragma: no cover 
            self.normalize(normalize)
        self.hasC= True
        self.hasC_dxdv= True
        return None

    def _evaluate(self,R,z,phi=0.,t=0.):
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
        OUTPUT:
           Phi(R,z)
        HISTORY:
           2013-04-10 - Started - Bovy (IAS)
        """
        x= numpy.sqrt(R**2.+z**2.)/self.a
        return -self.a**2.*numpy.pi/x*(-numpy.pi+2.*(1.+x)*numpy.arctan(1/x)+2.*(1.+x)*numpy.log(1.+x)+(1.-x)*numpy.log(1.+x**2.))

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
        return self.a*numpy.pi/x**2.*(numpy.pi-2.*numpy.arctan(1./x)-2.*numpy.log(1.+x)-numpy.log(1.+x**2.))*R/r

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
           2013-04-10 - Written - Bovy (IAS)
        """
        r= numpy.sqrt(R**2.+z**2.)
        x= r/self.a
        return self.a*numpy.pi/x**2.*(numpy.pi-2.*numpy.arctan(1./x)-2.*numpy.log(1.+x)-numpy.log(1.+x**2.))*z/r

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
           2013-04-10 - Written - Bovy (IAS)
        """
        r= numpy.sqrt(R**2.+z**2.)
        x= r/self.a
        return -numpy.pi/x**3./r**2.*(-4.*R**2.*r**3./(self.a**2.+r**2.)/(self.a+r)+(z**2.-2.*R**2.)*(numpy.pi-2.*numpy.arctan(1./x)-2.*numpy.log(1.+x)-numpy.log(1.+x**2.)))

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
           evaluate the density for this potential
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
