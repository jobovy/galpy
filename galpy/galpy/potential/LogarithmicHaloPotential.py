###############################################################################
#   LogarithmicHaloPotential.py: class that implements the logarithmic halo
#                                halo potential Phi(r) = vc**2 ln(r)
###############################################################################
import math as m
from Potential import Potential
_CORE=10**-8
class LogarithmicHaloPotential(Potential):
    """Class that implements the logarithmic halo potential Phi(r)"""
    def __init__(self,amp=1.,vc=235.,core=_CORE):
        """
        NAME:
           __init__
        PURPOSE:
           initialize a Logarithmic Halo potential
        INPUT:
           amp - amplitude to be applied to the potential (default: 1)
           vc - circular velocity
           core - core radius at which the logarithm is cut
        OUTPUT:
           (none)
        HISTORY:
           2010-04-02 - Started - Bovy (NYU)
        """
        Potential.__init__(self,amp=amp)
        self._vc2= vc**2.
        self._core2= core**2.
        return None

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
           2010-04-02 - Started - Bovy (NYU)
           2010-04-30 - Adapted for R,z - Bovy (NYU)
        """
        return self._vc2/2.*m.log(R**2.+z**2.+self._core2)

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
        """
        return -self._vc2*R/(R**2.+z**2.+self._core2)

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
        """
        return -self._vc2*z/(R**2.+z**2.+self._core2)
