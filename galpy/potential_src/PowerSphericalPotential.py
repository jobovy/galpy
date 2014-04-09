###############################################################################
#   PowerSphericalPotential.py: General class for potentials derived from 
#                               densities with two power-laws
#
#                                     amp
#                          rho(r)= ---------
#                                   r^\alpha
###############################################################################
import numpy as nu
from scipy import special, integrate
from Potential import Potential
class PowerSphericalPotential(Potential):
    """Class that implements spherical potentials that are derived from 
    power-law density models

                amp
    rho(r)= ---------
             r^\alpha
    """
    def __init__(self,amp=1.,alpha=1.,normalize=False):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a power-law-density potential

        INPUT:

           amp - amplitude to be applied to the potential (default: 1)

           alpha - inner power

           normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.

        OUTPUT:

           (none)

        HISTORY:

           2010-07-10 - Written - Bovy (NYU)

        """
        Potential.__init__(self,amp=amp)
        self.alpha= alpha
        if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)):
            self.normalize(normalize)
        self.hasC= True

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
           2010-07-10 - Started - Bovy (NYU)
        """
        if dR == 0 and dphi == 0:
            if self.alpha == 2.:
                return nu.log(R**2.+z**2.)/2. 
            else:
                return -(R**2.+z**2.)**(1.-self.alpha/2.)/(self.alpha-2.)
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
           2010-07-10 - Written - Bovy (NYU)
        """
        return -R/(R**2.+z**2.)**(self.alpha/2.)

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
        return -z/(R**2.+z**2.)**(self.alpha/2.)

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
        return 1./(R**2.+z**2.)**(self.alpha/2.)\
            -self.alpha*R**2./(R**2.+z**2.)**(self.alpha/2.+1.)

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
           2013-08-28 - Written - Bovy (IAs)
        """
        return -self.alpha*R*z*(R**2.+z**2.)**(-1.-self.alpha/2.)

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
        r= nu.sqrt(R**2.+z**2.)
        return (3.-self.alpha)/4./nu.pi/r**self.alpha

class KeplerPotential(PowerSphericalPotential):
    """Class that implements the Kepler potential

                amp
    Phi(r)= ---------
                 r
    """
    def __init__(self,amp=1.,normalize=False):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a Kepler potential

        INPUT:

           amp - amplitude to be applied to the potential (default: 1)

           alpha - inner power

           normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.

        OUTPUT:

           (none)

        HISTORY:

           2010-07-10 - Written - Bovy (NYU)

        """
        PowerSphericalPotential.__init__(self,amp=amp,normalize=normalize,
                                         alpha=3.)


