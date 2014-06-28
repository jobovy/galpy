import scipy as sc
from linearPotential import linearPotential
class KGPotential(linearPotential):
    """Class representing the Kuijken & Gilmore (1989) potential

    .. math::

        \Phi(x) = \\mathrm{amp}\\,\\left(K\\,\\left(\\sqrt{x^2+D^2}-D\\right)+F\\,x^2\\right)

    """
    def __init__(self,K=1.15,F=0.03,D=1.8,amp=1.):
        """
        NAME:

           __init__

        PURPOSE:

           Initialize a KGPotential

        INPUT:

           K= K parameter

           F= F parameter

           D= D parameter

           amp - an overall amplitude

        OUTPUT:

           instance

        HISTORY:

           2010-07-12 - Written - Bovy (NYU)

        """
        linearPotential.__init__(self,amp=amp)
        self._K= K
        self._F= F
        self._D= D
        self._D2= self._D**2.
        
    def _evaluate(self,x,t=0.):
        return self._K*(sc.sqrt(x**2.+self._D2)-self._D)+self._F*x**2.

    def _force(self,x,t=0.):
        return -x*(self._K/sc.sqrt(x**2+self._D2)+2.*self._F)
