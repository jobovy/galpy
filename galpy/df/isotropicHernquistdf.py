# Class that implements isotropic spherical Hernquist DF
# computed using the Eddington formula
import numpy
from ..util import conversion
from ..potential import evaluatePotentials,HernquistPotential
from .sphericaldf import isotropicsphericaldf

class isotropicHernquistdf(isotropicsphericaldf):
    """Class that implements isotropic spherical Hernquist DF computed using the Eddington formula"""
    def __init__(self,pot=None,ro=None,vo=None):
        """
        NAME:

            __init__

        PURPOSE:

            Initialize an isotropic Hernquist distribution function

        INPUT:

           pot= (None) Hernquist Potential instance


           ro=, vo= galpy unit parameters

        OUTPUT:
        
            None

        HISTORY:

            2020-08-09 - Written - Lane (UofT)

        """
        assert isinstance(pot,HernquistPotential),'pot= must be potential.HernquistPotential'
        isotropicsphericaldf.__init__(self,pot=pot,ro=ro,vo=vo)
        self._psi0= -evaluatePotentials(self._pot,0,0,use_physical=False)
        self._GMa = self._psi0*self._pot.a**2.
        # first factor = mass to make the DF that of mass density
        self._fEnorm= self._psi0*self._pot.a\
            /numpy.sqrt(2.)/(2*numpy.pi)**3/self._GMa**1.5

    def fE(self,E):
        """
        NAME:

            fE

        PURPOSE

            Calculate the energy portion of an isotropic Hernquist distribution function

        INPUT:

            E - The energy (can be Quantity)

        OUTPUT:

            fE - The value of the energy portion of the DF

        HISTORY:

            2020-08-09 - Written - James Lane (UofT)
        """
        Etilde= -conversion.parse_energy(E,vo=self._vo)/self._psi0
        # Handle E out of bounds
        Etilde_out = numpy.where(numpy.logical_or(Etilde<0,Etilde>1))[0]
        if len(Etilde_out)>0:
            # Set to dummy and 0 later, prevents functions throwing errors?
            Etilde[Etilde_out]=0.5
        sqrtEtilde= numpy.sqrt(Etilde)
        fE= self._fEnorm*sqrtEtilde/(1-Etilde)**2.\
            *((1.-2.*Etilde)*(8.*Etilde**2.-8.*Etilde-3.)\
              +((3.*numpy.arcsin(sqrtEtilde))\
                /numpy.sqrt(Etilde*(1.-Etilde))))
        # Fix out of bounds values
        if len(Etilde_out) > 0:
            fE[Etilde_out] = 0
        return fE

    def _icmf(self,ms):
        '''Analytic expression for the normalized inverse cumulative mass 
        function. The argument ms is normalized mass fraction [0,1]'''
        return self._pot.a*numpy.sqrt(ms)/(1-numpy.sqrt(ms))
