###############################################################################
#   IsochronePotential.py: The isochrone potential
#
#                                     - amp
#                          Phi(r)= ---------------------
#                                   b + sqrt{b^2+r^2}
###############################################################################
import numpy as nu
from .Potential import Potential, _APY_LOADED
if _APY_LOADED:
    from astropy import units
class IsochronePotential(Potential):
    """Class that implements the Isochrone potential

    .. math::

        \\Phi(r) = -\\frac{\\mathrm{amp}}{b+\\sqrt{b^2+r^2}}

    with :math:`\\mathrm{amp} = GM` the total mass.
    """
    def __init__(self,amp=1.,b=1.,normalize=False,
                 ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize an isochrone potential

        INPUT:

           amp= amplitude to be applied to the potential, the total mass (default: 1); can be a Quantity with units of mass density or Gxmass density

           b= scale radius of the isochrone potential (can be Quantity)

           normalize= if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.

           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

        OUTPUT:

           (none)

        HISTORY:

           2013-09-08 - Written - Bovy (IAS)

        """
        Potential.__init__(self,amp=amp,ro=ro,vo=vo,amp_units='mass')
        if _APY_LOADED and isinstance(b,units.Quantity):
            b= b.to(units.kpc).value/self._ro
        self.b= b
        self._scale= self.b
        self.b2= self.b**2.
        if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)): #pragma: no cover
            self.normalize(normalize)
        self.hasC= True
        self.hasC_dxdv= True

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
           2013-09-08 - Written - Bovy (IAS)
        """
        r2= R**2.+z**2.
        rb= nu.sqrt(r2+self.b2)
        return -1./(self.b+rb)

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
           2013-09-08 - Written - Bovy (IAS)
        """
        r2= R**2.+z**2.
        rb= nu.sqrt(r2+self.b2)
        dPhidrr= -1./rb/(self.b+rb)**2.
        return dPhidrr*R

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
           2013-09-08 - Written - Bovy (IAS)
        """
        r2= R**2.+z**2.
        rb= nu.sqrt(r2+self.b2)
        dPhidrr= -1./rb/(self.b+rb)**2.
        return dPhidrr*z

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
           2013-09-08 - Written - Bovy (IAS)
        """
        r2= R**2.+z**2.
        rb= nu.sqrt(r2+self.b2)
        return -(-self.b**3.-self.b*z**2.+(2.*R**2.-z**2.-self.b**2.)*rb)/\
            rb**3./(self.b+rb)**3.

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
           2013-09-08 - Written - Bovy (IAS)
        """
        r2= R**2.+z**2.
        rb= nu.sqrt(r2+self.b2)
        return -(-self.b**3.-self.b*R**2.-(R**2.-2.*z**2.+self.b**2.)*rb)/\
            rb**3./(self.b+rb)**3.

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
           t- time
        OUTPUT:
           d2phi/dR/dz
        HISTORY:
           2013-09-08 - Written - Bovy (IAS)
        """
        r2= R**2.+z**2.
        rb= nu.sqrt(r2+self.b2)
        return -R*z*(self.b+3.*rb)/\
            rb**3./(self.b+rb)**3.

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
           2013-09-08 - Written - Bovy (IAS)
        """
        r2= R**2.+z**2.
        rb= nu.sqrt(r2+self.b2)
        return (3.*(self.b+rb)*rb**2.-r2*(self.b+3.*rb))/\
            rb**3./(self.b+rb)**3./4./nu.pi

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
        r2= R**2.+z**2.
        rb= nu.sqrt(r2+self.b2)
        return self.b*((R*z)/r2-(self.b*R*z*(self.b**2+2.*R**2+z**2))
                       /((self.b**2+R**2)*r2*rb)
                       +nu.arctan(z/R)-nu.arctan(self.b*z/R/rb))/R**3/2./nu.pi

