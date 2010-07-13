from Potential import PotentialError
class linearPotential:
    """Class representing 1D potentials"""
    def __init__(self,amp=1.):
        self._amp= amp
        self.dim= 1
        return None

    def __call__(self,x):
        """
        NAME:
           __call__
        PURPOSE:
           evaluate the potential
        INPUT:
           x
        OUTPUT:
           Phi(x)
        HISTORY:
           2010-07-12 - Written - Bovy (NYU)
        """
        try:
            return self._amp*self._evaluate(x)
        except AttributeError:
            raise PotentialError("'_evaluate' function not implemented for this potential")

    def force(self,x):
        """
        NAME:
           force
        PURPOSE:
           evaluate the force
        INPUT:
           x
        OUTPUT:
           F(x)
        HISTORY:
           2010-07-12 - Written - Bovy (NYU)
        """
        try:
            return self._amp*self._force(x)
        except AttributeError:
            raise PotentialError("'_force' function not implemented for this potential")
