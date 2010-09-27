###############################################################################
#   LogarithmicHaloPotential.py: class that implements the logarithmic halo
#                                halo potential Phi(r) = vc**2 ln(r)
###############################################################################
import math as m
from Potential import Potential
_CORE=10**-8
class LogarithmicHaloPotential(Potential):
    """Class that implements the logarithmic halo potential Phi(r)"""
    def __init__(self,amp=1.,core=_CORE,q=1.,normalize=False):
        """
        NAME:
           __init__
        PURPOSE:
           initialize a Logarithmic Halo potential
        INPUT:
           amp - amplitude to be applied to the potential (default: 1)
           core - core radius at which the logarithm is cut
           q - potential flattening (z/q)**2.
           normalize - if True, normalize such that vc(1.,0.)=1., or, if 
                       given as a number, such that the force is this fraction 
                       of the force necessary to make vc(1.,0.)=1.
        OUTPUT:
           (none)
        HISTORY:
           2010-04-02 - Started - Bovy (NYU)
        """
        Potential.__init__(self,amp=amp)
        self._core2= core**2.
        self._q= q
        if normalize:
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
           2010-04-02 - Started - Bovy (NYU)
           2010-04-30 - Adapted for R,z - Bovy (NYU)
        """
        return 1./2.*m.log(R**2.+(z/self._q)**2.+self._core2)

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
        """
        return -R/(R**2.+(z/self._q)**2.+self._core2)

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
        """
        return -z/self._q**2./(R**2.+(z/self._q)**2.+self._core2)

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
        """
        return 1./self._q**2.*((2.*self._q**2.+1.)*self._core2+R**2.\
                                   +(2.-self._q**-2.)*z**2.)/\
                                   (R**2.+(z/self._q)**2.+self._core2)**2.
