# Class that implements isotropic spherical NFW DF
import numpy
from ..util import conversion
from ..potential import NFWPotential
from .sphericaldf import isotropicsphericaldf
# Coefficients of the improved analytical approximation that JB made
_COEFFS= numpy.array([-1491.8902622624896139, 11690.6687704701107577,
                      -41507.5120287257886957, 88327.4966507880599238,
                      -125579.5629730796063086, 125886.0143554724199930,
                      -91520.0457568992424058, 48889.6565313585524564,
                      -19234.8914455144113163, 5535.6739262642768153,
                      -1146.7675217480441461, 166.4047219881630326,
                      -16.2119268143120756,0.9071500840024120,
                      0.0093256420259421, 0.0926700870836157])

class isotropicNFWdf(isotropicsphericaldf):
    """Class that implements the approximate isotropic spherical NFW DF (either `Widrow 2000 <https://ui.adsabs.harvard.edu/abs/2000ApJS..131...39W/abstract>`__ or an improved fit by Bovy 2022).
    """
    def __init__(self,pot=None,widrow=False,rmax=1e4,ro=None,vo=None):
        """
        NAME:

            __init__

        PURPOSE:

            Initialize an isotropic NFW distribution function

        INPUT:

           pot= (None) NFW Potential instance

           widrow= (False) if True, use the approximate form from Widrow (2000), otherwise use improved fit that has <~1e-5 relative density errors

           rmax= (1e4) maximum radius to consider (can be Quantity); set to numpy.inf to evaluate NFW w/o cut-off

           ro=, vo= galpy unit parameters

        OUTPUT:
        
            None

        HISTORY:

            2021-02-01 - Written - Bovy (UofT)

        """
        assert isinstance(pot,NFWPotential),'pot= must be potential.NFWPotential'
        isotropicsphericaldf.__init__(self,pot=pot,rmax=rmax,ro=ro,vo=vo)
        self._Etildemax= pot._amp/pot.a
        self._fEnorm= (9.1968e-2)**widrow/\
            (4.*numpy.pi)/pot.a**1.5/pot._amp**0.5
        self._widrow= widrow
        self._Etildemin= -pot(self._rmax,0,use_physical=False)/self._Etildemax

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
        indx= (Etilde > self._Etildemin)*(Etilde <= 1.)
        if self._widrow:
            out[indx]= self._fEnorm*Etilde[indx]**1.5*(1-Etilde[indx])**-2.5\
                *(-numpy.log(Etilde[indx])/(1.-Etilde[indx]))**-2.7419\
                *numpy.exp(0.3620*Etilde[indx]-0.5639*Etilde[indx]**2.
                           -0.0859*Etilde[indx]**3.-0.4912*Etilde[indx]**4.)
        else:
            out[indx]= self._fEnorm*Etilde[indx]**1.5*(1-Etilde[indx])**-2.5\
                *(-numpy.log(Etilde[indx])/(1.-Etilde[indx]))**-2.75\
                *numpy.polyval(_COEFFS,Etilde[indx])
        return out
