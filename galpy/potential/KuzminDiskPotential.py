###############################################################################
#   KuzminDiskPotential.py: class that implements Kuzmin disk potential
#
#                                   - amp                   
#               Phi(R, z)=  ---------------------------
#                            \sqrt{R^2 + (a + |z|)^2} 
###############################################################################
import numpy as nu
from .Potential import Potential, _APY_LOADED
if _APY_LOADED:
    from astropy import units
class KuzminDiskPotential(Potential):
    """Class that implements the Kuzmin Disk potential

    .. math::

        \\Phi(R,z) = -\\frac{\\mathrm{amp}}{\\sqrt{R^2 + (a + |z|)^2}}

    with :math:`\\mathrm{amp} = GM` the total mass.
    """
    def __init__(self, amp=1., a=1. ,normalize=False, ro=None,vo=None):
        """
        NAME:

            __init__

        PURPOSE:

            initialize a Kuzmin disk Potential

        INPUT:

            amp - amplitude to be applied to the potential, the total mass (default: 1); can be a Quantity with units of mass density or Gxmass density

            a - scale length (can be Quantity)
    
            normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.

           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

        OUTPUT:

           KuzminDiskPotential object

        HISTORY:

           2016-05-09 - Written - Aladdin 

        """
        Potential.__init__(self,amp=amp,ro=ro,vo=vo,amp_units='mass')
        if _APY_LOADED and isinstance(a,units.Quantity): 
            a= a.to(units.kpc).value/self._ro 
        self._a = a ## a must be greater or equal to 0. 
        if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)): 
            self.normalize(normalize)
        self.hasC = True
        self.hasC_dxdv= True
        return None

    def _evaluate(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _evaluate
        PURPOSE:
           evaluate the potential at (R,z)
        INPUT:
           R - Cylindrical Galactocentric radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           potential at (R,z)
        HISTORY:
           2016-05-09 - Written - Aladdin 
        """
        return -self._denom(R, z)**-0.5

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
            the radial force = -dphi/dR
        HISTORY:
            2016-05-09 - Written - Aladdin 
        """
        return -self._denom(R, z)**-1.5 * R

    def _zforce(self, R, z, phi=0., t=0.):
        """
        NAME:
           _zforce
        PURPOSE:
           evaluate the vertical force  for this potential
        INPUT:
           R - Cylindrical Galactocentric radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the vertical force = -dphi/dz
        HISTORY:
           2016-05-09 - Written - Aladdin 
        """
        return -nu.sign(z) * self._denom(R,z)**-1.5 * (self._a + nu.fabs(z))
        
    def _R2deriv(self,R,z,phi=0.,t=0.):
        """
        NAME:
            _Rforce
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
            2016-05-13 - Written - Aladdin 
        """
        return self._denom(R, z)**-1.5 - 3.*R**2 * self._denom(R, z)**-2.5 
        
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
           2016-05-13 - Written - Aladdin 
        """
        a = self._a
        return self._denom(R, z)**-1.5 - 3. * (a + nu.fabs(z))**2. * self._denom(R, z)**-2.5 

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
           2016-05-13 - Written - Aladdin 
        """
        return -3 * nu.sign(z) * R * (self._a + nu.fabs(z)) *self._denom(R, z)**-2.5 
       
    def _surfdens(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _surfdens
        PURPOSE:
           evaluate the surface density
        INPUT:
           R - Cylindrical Galactocentric radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           Sigma (R,z)
        HISTORY:
           2018-08-19 - Written - Bovy (UofT)
        """
        return self._a*(R**2+self._a**2)**-1.5/2./nu.pi

    def _denom(self, R, z):
        """
        NAME:
           _denom
        PURPOSE:
           evaluate R^2 + (a + |z|)^2 which is used in the denominator
           of most equations
        INPUT:
           R - Cylindrical Galactocentric radius
           z - vertical height
        OUTPUT:
           R^2 + (a + |z|)^2
        HISTORY:
           2016-05-09 - Written - Aladdin 
        """
        return (R**2. + (self._a + nu.fabs(z))**2.)
