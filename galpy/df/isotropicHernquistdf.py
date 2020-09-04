# Class that implements isotropic spherical Hernquist DF
# computed using the Eddington formula
import numpy
from ..util import conversion
from ..potential import evaluatePotentials,HernquistPotential
from .Eddingtondf import Eddingtondf

class isotropicHernquistdf(Eddingtondf):
    """Class that implements isotropic spherical Hernquist DF computed using the Eddington formula"""
    def __init__(self,pot=None,ro=None,vo=None):
        assert isinstance(pot,HernquistPotential),'pot= must be potential.HernquistPotential'
        Eddingtondf.__init__(self,pot=pot,ro=ro,vo=vo)
        self._phi0= -evaluatePotentials(self._pot,0,0,use_physical=False)
        self._GMa = self._phi0*self._pot.a**2.
        self._fEnorm= 1./numpy.sqrt(2.)/(2*numpy.pi)**3/self._GMa**1.5

    def _call_internal(self,*args):
        """
        NAME:

            _call_internal

        PURPOSE

            Calculate the distribution function for an isotropic Hernquist

        INPUT:

            E,L,Lz - The energy, angular momemtum magnitude, and its z component (only E is used)

        OUTPUT:

            f(x,v) = f(E[x,v])

        HISTORY:

            2020-07 - Written - Lane (UofT)

        """
        return self.fE(args[0])

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
        Etilde= -conversion.parse_energy(E,vo=self._vo)/self._phi0
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
