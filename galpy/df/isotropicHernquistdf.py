# Class that implements isotropic spherical Hernquist DF
# computed using the Eddington formula
from .sphericaldf import sphericaldf
from .Eddingtondf import Eddingtondf

class isotropicHernquistdf(Eddingtondf):
    """Class that implements isotropic spherical Hernquist DF computed using the Eddington formula"""
    def __init__(self,pot=None,ro=None,vo=None):
        # Initialize using sphericaldf rather than Eddingtondf, because
        # Eddingtondf will have code specific to computing the Eddington
        # integral, which is not necessary for Hernquist
        sphericaldf.__init__(self,pot=pot,ro=ro,vo=vo)

    def __call_internal__(self,*args):
        """
        NAME:

            __call_internal__

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
        if _APY_LOADED and isinstance(E,units.quantity.Quantity):
            E= E.to(units.km**2/units.s**2).value/vo**2.
        # Scale energies
        phi0 = evaluatePotentials(self._pot,0,0)
        Erel = -E
        Etilde = Erel/phi0
        # Handle potential E out of bounds
        Etilde_out = numpy.where(Etilde<0|Etilde>1)[0]
        if len(Etilde_out)>0:
            # Set to dummy and 0 later, prevents functions throwing errors?
            Etilde[Etilde_out]=0.5
        _GMa = phi0*self._pot.a**2.
        fH = numpy.power((2**0.5)*((2*numpy.pi)**3) *((_GMa)**1.5),-1)\
            *(numpy.sqrt(Etilde)/numpy.power(1-Etilde,2))\
            *((1-2*Etilde)*(8*numpy.power(Etilde,2)-8*Etilde-3)\
            +((3*numpy.arcsin(numpy.sqrt(Etilde)))\
            /numpy.sqrt(Etilde*(1-Etilde))))
        # Fix out of bounds values
        if len(Etilde_out) > 0:
            fH[Etilde_out] = 0
        return fH

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
        return self.__call_internal__(E)
