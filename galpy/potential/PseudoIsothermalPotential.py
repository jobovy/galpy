###############################################################################
#   PseudoIsothermalPotential.py: class that implements the pseudo-isothermal
#                                 halo potential
###############################################################################
import numpy as nu
from .Potential import Potential, _APY_LOADED
if _APY_LOADED:
    from astropy import units
class PseudoIsothermalPotential(Potential):
    """Class that implements the pseudo-isothermal potential

    .. math::

        \\rho(r) = \\frac{\\mathrm{amp}}{4\\,\pi\\, a^3}\\,\\frac{1}{1+(r/a)^2}

    """
    def __init__(self,amp=1.,a=1.,normalize=False,
                 ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a pseudo-isothermal potential

        INPUT:

           amp - amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass or Gxmass

           a - core radius (can be Quantity)

           normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.

           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

        OUTPUT:

           (none)

        HISTORY:

           2015-12-04 - Started - Bovy (UofT)

        """
        Potential.__init__(self,amp=amp,ro=ro,vo=vo,amp_units='mass')
        if _APY_LOADED and isinstance(a,units.Quantity):
            a= a.to(units.kpc).value/self._ro
        self.hasC= True
        self.hasC_dxdv= True
        self._a= a
        self._a2= a**2.
        self._a3= a**3.
        if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)): #pragma: no cover 
            self.normalize(normalize)
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
           2015-12-04 - Started - Bovy (UofT)
        """
        r2= R**2.+z**2.
        r= nu.sqrt(r2)
        return (0.5*nu.log(1+r2/self._a2)\
                    +self._a/r*nu.arctan(r/self._a))/self._a

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
           2015-12-04 - Started - Bovy (UofT)
        """
        r2= R**2.+z**2.
        r= nu.sqrt(r2)
        return -(1./r-self._a/r2*nu.arctan(r/self._a))/self._a*R/r

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
           2015-12-04 - Started - Bovy (UofT)
        """
        r2= R**2.+z**2.
        r= nu.sqrt(r2)
        return -(1./r-self._a/r2*nu.arctan(r/self._a))/self._a*z/r

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
           2015-12-04 - Started - Bovy (UofT)
        """
        return 1./(1.+(R**2.+z**2.)/self._a2)/4./nu.pi/self._a3

    def _R2deriv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _R2deriv
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
           2011-10-09 - Written - Bovy (IAS)
        """
        r2= R**2.+z**2.
        r= nu.sqrt(r2)
        return (1./r2*(1.-R**2./r2*(3.*self._a2+2.*r2)/(self._a2+r2))\
                    +self._a/r2/r*(3.*R**2./r2-1.)*nu.arctan(r/self._a))\
                    /self._a

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
           t - time
        OUTPUT:
           the second vertical derivative
        HISTORY:
           2012-07-25 - Written - Bovy (IAS@MPIA)
        """
        r2= R**2.+z**2.
        r= nu.sqrt(r2)
        return (1./r2*(1.-z**2./r2*(3.*self._a2+2.*r2)/(self._a2+r2))\
                    +self._a/r2/r*(3.*z**2./r2-1.)*nu.arctan(r/self._a))\
                    /self._a

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
           d2Phi/dR/dz
        HISTORY:
           2013-08-28 - Written - Bovy (IAS)
        """
        r2= R**2.+z**2.
        r= nu.sqrt(r2)
        return (3.*self._a/r2/r2*nu.arctan(r/self._a)\
                    -1./r2/r*((3.*self._a2+2.*r2)/(r2+self._a2)))*R*z/r\
                    /self._a

