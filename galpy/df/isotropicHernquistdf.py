# Class that implements isotropic spherical Hernquist DF
# computed using the Eddington formula
import numpy
import pdb
from .Eddingtondf import Eddingtondf
from ..potential import evaluatePotentials,HernquistPotential
from .df import _APY_LOADED
if _APY_LOADED:
    from astropy import units

class isotropicHernquistdf(Eddingtondf):
    """Class that implements isotropic spherical Hernquist DF computed using the Eddington formula"""
    def __init__(self,pot=None,ro=None,vo=None):
        assert isinstance(pot,HernquistPotential),'pot= must be potential.HernquistPotential'
        Eddingtondf.__init__(self,pot=pot,ro=ro,vo=vo)

    def _call_internal(self,*args):
        """
        NAME:

            _call_internal

        PURPOSE

            Calculate the distribution function for an isotropic Hernquist

        INPUT:

            E - The energy

        OUTPUT:

            fH - The distribution function

        HISTORY:

            2020-07 - Written

        """
        E = args[0]
        return self.fE(E)

    def fE(self,E):
        """
        NAME:

            fE

        PURPOSE

            Calculate the energy portion of an isotropic Hernquist distribution 
            function

        INPUT:

            E - The energy (can be Quantity)

        OUTPUT:

            fE - The value of the energy portion of the DF

        HISTORY:

            2020-08-09 - Written - James Lane (UofT)
        """
        if _APY_LOADED and isinstance(E,units.quantity.Quantity):
            E= E.to(units.km**2/units.s**2).value/vo**2.
        phi0 = -evaluatePotentials(self._pot,0,0,use_physical=False)
        Erel = -E
        Etilde = Erel/phi0
        # Handle potential E out of bounds
        Etilde_out = numpy.where(numpy.logical_or(Etilde<0,Etilde>1))[0]
        if len(Etilde_out)>0:
            # Set to dummy and 0 later, prevents functions throwing errors?
            Etilde[Etilde_out]=0.5

        _GMa = phi0*self._pot.a**2.
        fE = numpy.power((2**0.5)*((2*numpy.pi)**3)*((_GMa)**1.5),-1)\
           *(numpy.sqrt(Etilde)/numpy.power(1-Etilde,2))\
           *((1-2*Etilde)*(8*numpy.power(Etilde,2)-8*Etilde-3)\
           +((3*numpy.arcsin(numpy.sqrt(Etilde)))\
           /numpy.sqrt(Etilde*(1-Etilde))))
        # Fix out of bounds values
        if len(Etilde_out) > 0:
            fE[Etilde_out] = 0
        return fE

    def _icmf(self,ms):
        '''Analytic expression for the normalized inverse cumulative mass 
        function. The argument ms is normalized mass fraction [0,1]'''
        return self._pot.a*numpy.sqrt(ms)/(1-numpy.sqrt(ms))
