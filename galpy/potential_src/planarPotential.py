import numpy as nu
import galpy.util.bovy_plot as plot
from Potential import PotentialError
class planarPotential:
    """Class representing 2D (R,\phi) potentials"""
    def __init__(self,amp=1.):
        self._amp= 1.
        self.dim= 2
        return None

    def __call__(self,*args):
        """
        NAME:
           __call__
        PURPOSE:
           evaluate the potential
        INPUT:
           Either: R or R,phi [rad]
        OUTPUT:
           Phi(R(,phi)))
        HISTORY:
           2010-07-13 - Written - Bovy (NYU)
        """
        try:
            return self._amp*self._evaluate(*args)
        except AttributeError:
            raise PotentialError("'_evaluate' function not implemented for this potential")

    def Rforce(self,*args):
        """
        NAME:
           Rforce
        PURPOSE:
           evaluate the radial force
        INPUT:
           Either: R or R,phi [rad]         
        OUTPUT:
           F_R(R,(\phi)))
        HISTORY:
           2010-07-13 - Written - Bovy (NYU)
        """
        try:
            return self._amp*self._Rforce(*args)
        except AttributeError:
            raise PotentialError("'_Rforce' function not implemented for this potential")

    def phiforce(self,*args):
        """
        NAME:
           phiforce
        PURPOSE:
           evaluate the phi force
        INPUT:
           Either: R or R,phi [rad]         
        OUTPUT:
           F_\phi(R,(\phi)))
        HISTORY:
           2010-07-13 - Written - Bovy (NYU)
        """
        try:
            return self._amp*self._phiforce(*args)
        except AttributeError:
            raise PotentialError("'_phiforce' function not implemented for this potential")


class planarAxiPotential(planarPotential):
    """Class representing axisymmetric planar potentials"""
    def __init__(self,amp=1.):
        planarPotential.__init__(self,amp=amp)
        return None
    
    def _phiforce(self,*args):
        return 0.
