# Class that implements isotropic spherical Plummer DF
import numpy
from ..util import conversion
from ..potential import PlummerPotential
from .sphericaldf import isotropicsphericaldf

class isotropicPlummerdf(isotropicsphericaldf):
    """Class that implements isotropic spherical Plummer DF:

    .. math::
    
        f(E) = {24\\sqrt{2} \\over 7\\pi^3}\\,{b^2\\over (GM)^5}\\,(-E)^{7/2}

    for :math:`-GM/b \leq E \leq 0` and zero otherwise. The parameter :math:`GM` is the total mass and :math:`b` the Plummer profile's scale parameter.
    """
    def __init__(self,pot=None,ro=None,vo=None):
        """
        NAME:

            __init__

        PURPOSE:

            Initialize an isotropic Plummer distribution function

        INPUT:

           pot= (None) Plummer Potential instance

           ro=, vo= galpy unit parameters

        OUTPUT:
        
            None

        HISTORY:

            2020-10-01 - Written - Bovy (UofT)

        """
        assert isinstance(pot,PlummerPotential),'pot= must be potential.PlummerPotential'
        isotropicsphericaldf.__init__(self,pot=pot,ro=ro,vo=vo)
        self._Etildemax= pot._amp/pot._b
        # /amp^4 instead of /amp^5 to make the DF that of mass density
        self._fEnorm=24.*numpy.sqrt(2.)/7./numpy.pi**3.*pot._b**2./pot._amp**4.

    def fE(self,E):
        """
        NAME:

            fE

        PURPOSE

            Calculate the energy portion of an isotropic Plummer distribution function

        INPUT:

            E - The energy (can be Quantity)

        OUTPUT:

            fE - The value of the energy portion of the DF

        HISTORY:

            2020-10-01 - Written - Bovy (UofT)
        """
        Etilde= -conversion.parse_energy(E,vo=self._vo)
        out= numpy.zeros_like(Etilde)
        indx= (Etilde > 0)*(Etilde <= self._Etildemax)
        out[indx]= self._fEnorm*(Etilde[indx])**3.5
        return out

    def _icmf(self,ms):
        '''Analytic expression for the normalized inverse cumulative mass 
        function. The argument ms is normalized mass fraction [0,1]'''
        return self._pot._b/numpy.sqrt(ms**(-2./3.)-1.)
