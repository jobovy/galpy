###############################################################################
#   MiyamotoNagaiPotential.py: class that implements the Miyamoto-Nagai 
#                              potential
#                                                           GM
#                              phi(R,z) = -  ---------------------------------
#                                             \sqrt(R^2+(a+\sqrt(z^2+b^2))^2)
###############################################################################
import math as m
from Potential import Potential
class MiyamotoNagaiPotential(Potential):
    """Class that implements the Miyamoto-Nagai potential
                                 amp
    phi(R,z) = -  ---------------------------------
                   \sqrt(R^2+(a+\sqrt(z^2+b^2))^2)
    """
    def __init__(self,amp=1.,a=0.,b=0.,normalize=False):
        """
        NAME:
           __init__
        PURPOSE:
           initialize a Miyamoto-Nagai potential
        INPUT:
           amp - amplitude to be applied to the potential (default: 1)
           a - "disk scale" (in terms of Ro)
           b - "disk height" (in terms of Ro)
           normalize - if True, normalize such that vc(1.,0.)=1., or, if 
                       given as a number, such that the force is this fraction 
                       of the force necessary to make vc(1.,0.)=1.
        OUTPUT:
           (none)
        HISTORY:
           2010-07-09 - Started - Bovy (NYU)
        """
        Potential.__init__(self,amp=amp)
        self._a= a
        self._b= b
        self._b2= self._b**2.
        if normalize:
            self.normalize(normalize)

    def _evaluate(self,R,z):
        """
        NAME:
           _evaluate
        PURPOSE:
           evaluate the potential at R,z
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
        OUTPUT:
           Phi(R,z)
        HISTORY:
           2010-07-09 - Started - Bovy (NYU)
        """
        return -1./m.sqrt(R**2.+(self._a+m.sqrt(z**2.+self._b2))**2.)

    def _Rforce(self,R,z):
        """
        NAME:
           _Rforce
        PURPOSE:
           evaluate the radial force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
        OUTPUT:
           the radial force
        HISTORY:
           2010-07-09 - Written - Bovy (NYU)
        """
        return -R/(R**2.+(self._a+m.sqrt(z**2.+self._b2))**2.)**(3./2.)

    def _zforce(self,R,z):
        """
        NAME:
           _zforce
        PURPOSE:
           evaluate the vertical force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
        OUTPUT:
           the vertical force
        HISTORY:
           2010-07-09 - Written - Bovy (NYU)
        """
        sqrtbz= m.sqrt(self._b2+z**2.)
        asqrtbz= self._a+sqrtbz
        return (-z*asqrtbz/sqrtbz/
                 (R**2.+(self._a+m.sqrt(z**2.+self._b2))**2.)**(3./2.))
