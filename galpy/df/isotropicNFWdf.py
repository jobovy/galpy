# Class that implements isotropic spherical NFW DF (approx form from Widrow00)
import numpy
from ..util import conversion
from ..potential import NFWPotential
from .sphericaldf import isotropicsphericaldf

class isotropicNFWdf(isotropicsphericaldf):
    """Class that implements the approximate isotropic spherical NFW DF (`Widrow 2000 <https://ui.adsabs.harvard.edu/abs/2000ApJS..131...39W/abstract>`__).
    """
    def __init__(self,pot=None,rmax=100.,ro=None,vo=None):
        """
        NAME:

            __init__

        PURPOSE:

            Initialize an isotropic distribution function

        INPUT:

           pot= (None) NFW Potential instance or list thereof

           rmax= (100.) when sampling, maximum radius to consider (can be Quantity)

           ro=, vo= galpy unit parameters

        OUTPUT:
        
            None

        HISTORY:

            2021-02-01 - Written - Bovy (UofT)

        """
        assert isinstance(pot,NFWPotential),'pot= must be potential.NFWPotential'
        isotropicsphericaldf.__init__(self,pot=pot,ro=ro,vo=vo)
        self._Etildemax= pot._amp/pot.a
        self._fEnorm= 9.1968e-2/(4.*numpy.pi)/pot.a**1.5/pot._amp**0.5
        self._rmax= conversion.parse_length(rmax,ro=self._ro)

    def fE(self,E):
        """
        NAME:

            fE

        PURPOSE

            Calculate the energy portion of an isotropic NFW distribution function

        INPUT:

            E - The energy (can be Quantity)

        OUTPUT:

            fE - The value of the energy portion of the DF

        HISTORY:

            2021-02-01 - Written - Bovy (UofT)
        """
        Etilde= -conversion.parse_energy(E,vo=self._vo)/self._Etildemax
        out= numpy.zeros_like(Etilde)
        indx= (Etilde > 0)*(Etilde <= 1.)
        out[indx]= self._fEnorm*Etilde[indx]**1.5*(1-Etilde[indx])**-2.5\
            *(-numpy.log(Etilde[indx])/(1.-Etilde[indx]))**-2.7419\
            *numpy.exp(0.3620*Etilde[indx]-0.5639*Etilde[indx]**2.
                       -0.0859*Etilde[indx]**3.-0.4912*Etilde[indx]**4.)
        return out
