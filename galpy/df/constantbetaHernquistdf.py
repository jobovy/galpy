# Class that implements the anisotropic spherical Hernquist DF with constant
# beta parameter
import numpy
import pdb
import scipy.special
import scipy.integrate
from .constantbetadf import constantbetadf
from .df import _APY_LOADED
from ..potential import evaluatePotentials,HernquistPotential
if _APY_LOADED:
    from astropy import units

class constantbetaHernquistdf(constantbetadf):
    """Class that implements the anisotropic spherical Hernquist DF with constant beta parameter"""
    def __init__(self,pot=None,beta=0,ro=None,vo=None):
        """
        NAME:

            __init__

        PURPOSE:

            Initialize a DF with constant anisotropy

        INPUT:

            pot - Hernquist potential which determines the DF

            beta - anisotropy parameter

        OUTPUT:

            None

        HISTORY:

            2020-07-22 - Written
        """
        assert isinstance(pot,HernquistPotential),'pot= must be potential.HernquistPotential'
        constantbetadf.__init__(self,pot=pot,beta=beta,ro=ro,vo=vo)

    def __call_internal__(self,*args):
        """
        NAME:

            __call_internal

        PURPOSE:

            Evaluate the DF for a constant anisotropy Hernquist

        INPUT:

            E - The energy

            L - The angular momentum

        OUTPUT:

            fH - The value of the DF

        HISTORY:

            2020-07-22 - Written
        """
        E = args[0]
        L = args[1]
        f1 = self.f1E(E)
        return L**(-2*self.beta)*f1

    def f1E(self,E):
        # Stub for computing f_1(E)
        return None

    
