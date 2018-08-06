###############################################################################
#   RingPotential.py: The gravitational potential of a thin, circular ring
###############################################################################
import numpy as nu
from scipy import special
from .Potential import Potential, _APY_LOADED
if _APY_LOADED:
    from astropy import units
class RingPotential(Potential):
    """Class that implements the potential of an infinitesimally-thin, circular ring

    .. math::

        \\rho(R,z) = \\frac{\\mathrm{amp}}{2\pi\,R_0}\\,\\delta(R-R_0)\\,\\delta(z)

    with :math:`\\mathrm{amp} = GM` the mass of the ring.
    """
    def __init__(self,amp=1.,a=0.75,normalize=False,ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a circular ring potential

        INPUT:

           amp - mass of the ring (default: 1); can be a Quantity with units of mass or Gxmass

           a= (0.75) radius of the ring (can be Quantity)

           normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.; note that because the force is always positive at r < a, this does not work if a > 1

           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

        OUTPUT:

           (none)

        HISTORY:

           2018-08-04 - Written - Bovy (UofT)

        """
        Potential.__init__(self,amp=amp,ro=ro,vo=vo,amp_units='mass')
        if _APY_LOADED and isinstance(a,units.Quantity):
            a= a.to(units.kpc).value/self._ro
        self.a= a
        self.a2= self.a**2
        self._amp/= 2.*nu.pi*self.a
        if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)):
            if self.a > 1.:
                raise ValueError('RingPotential with normalize= for a > 1 is not supported (because the force is always positive at r=1)')
            self.normalize(normalize)
        self.hasC= False
        self.hasC_dxdv= False

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
           2018-08-04 - Written - Bovy (UofT)
        """
        m= 4.*R*self.a/((R+self.a)**2+z**2)
        return -4.*self.a/nu.sqrt((R+self.a)**2+z**2)*special.ellipk(m)

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
           2018-08-04 - Written - Bovy (UofT)
        """
        m= 4.*R*self.a/((R+self.a)**2+z**2)
        return -2.*self.a/R/nu.sqrt((R+self.a)**2+z**2)\
            *(m*(R**2-self.a2-z**2)/4./(1.-m)/self.a/R*special.ellipe(m)
              +special.ellipk(m))

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
           2018-08-04 - Written - Bovy (UofT)
        """
        m= 4.*R*self.a/((R+self.a)**2+z**2)
        return -4.*z*self.a/(1.-m)*((R+self.a)**2+z**2)**-1.5*special.ellipe(m)

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
           2018-08-04 - Written - Bovy (UofT)
        """
        Raz2= (R+self.a)**2+z**2
        Raz= nu.sqrt(Raz2)
        m= 4.*R*self.a/Raz2
        R2ma2mz2o4aR1m= (R**2-self.a2-z**2)/4./self.a/R/(1.-m)
        return (2*R**2+self.a2+3*R*self.a+z**2)/R/Raz2*self._Rforce(R,z)\
            +2.*self.a/R/Raz*(m*(R**2+self.a2+z**2)/4./(1.-m)/self.a/R**2\
                                  *special.ellipe(m)\
              +(R2ma2mz2o4aR1m/(1.-m)*special.ellipe(m)
                +0.5*R2ma2mz2o4aR1m*(special.ellipe(m)-special.ellipk(m))
                +0.5*(special.ellipe(m)/(1.-m)-special.ellipk(m))/m)\
                                  *4*self.a*(self.a2+z**2-R**2)/Raz2**2)
    
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
           2018-08-04 - Written - Bovy (UofT)
        """
        Raz2= (R+self.a)**2+z**2
        m= 4.*R*self.a/Raz2
        # Explicitly swapped in zforce here, so the z/z can be cancelled 
        # and z=0 is handled properly
        return -4.*(3.*z**2/Raz2-1.
                    +4.*((1.+m)/(1.-m)-special.ellipk(m)/special.ellipe(m))\
                        *self.a*R*z**2/Raz2**2/m)\
                    *self.a/(1.-m)*((R+self.a)**2+z**2)**-1.5*special.ellipe(m)
     
    def _Rzderiv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _Rzderiv
        PURPOSE:
           evaluate the mixed R,z derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           d2phi/dR/dz
        HISTORY:
           2018-08-04 - Written - Bovy (UofT)
        """
        Raz2= (R+self.a)**2+z**2
        m= 4.*R*self.a/Raz2
        return (3.*(R+self.a)/Raz2
                -2.*((1.+m)/(1.-m)-special.ellipk(m)/special.ellipe(m))\
                    *self.a*(self.a2+z**2-R**2)/Raz2**2/m)*self._zforce(R,z)
