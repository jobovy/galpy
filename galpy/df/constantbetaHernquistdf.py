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
        fE = self.fE(E)
        return L**(-2*self.beta)*fE

    def fE(self,E):
        """
        NAME:

            fE

        PURPOSE

            Calculate the energy portion of a Hernquist distribution function

        INPUT:

            E - The energy (can be Quantity)

        OUTPUT:

            fE - The value of the energy portion of the DF

        HISTORY:

            2020-07-22 - Written
        """
        if _APY_LOADED and isinstance(E,units.quantity.Quantity):
            E= E.to(units.km**2/units.s**2).value/self._vo**2.
        psi0 = -evaluatePotentials(self._pot,0,0,use_physical=False)
        Erel = -E
        Etilde = Erel/psi0
        # Handle potential E outside of bounds
        Etilde_out = numpy.where(numpy.logical_or(Etilde<0,Etilde>1))[0]
        if len(Etilde_out)>0:
            # Dummy variable now and 0 later, prevents numerical issues?
            Etilde[Etilde_out]=0.5

        # Evaluate depending on beta
        _GMa = psi0*self._pot.a**2.
        if self.beta == 0.:
            f1 = numpy.power((2**0.5)*((2*numpy.pi)**3)*((_GMa)**1.5),-1)\
               *(numpy.sqrt(Etilde)/numpy.power(1-Etilde,2))\
               *((1-2*Etilde)*(8*numpy.power(Etilde,2)-8*Etilde-3)\
               +((3*numpy.arcsin(numpy.sqrt(Etilde)))\
               /numpy.sqrt(Etilde*(1-Etilde))))
        elif self.beta == 0.5:
            f1 = (3*Etilde**2)/(4*(numpy.pi**3)*_GMa)
        elif self.beta == -0.5:
            f1 = ((20*Etilde**3-20*Etilde**4+6*Etilde**5)\
               /(1-Etilde)**4)/(4*numpy.pi**3*(_GMa)**2)
        elif self.beta < 1.0 and self.beta > 0.5:
            f1 = self._f1_beta_gt05_Hernquist(Erel)
        else:
            f1 = self._f1_any_beta(Erel) # This function sits in the super class?
        if len(Etilde_out)>0:
            f1[Etilde_out] = 0
        return f1

    def _f1_beta_gt05_Hernquist(self,Erel):
        """Calculate f1 for a Hernquist model when 0.5 < beta < 1.0"""
        psi0 = evaluatePotentials(self._pot,0,0) 
        _a = self._pot.a
        _GM = psi0*_a
        Ibeta = numpy.sqrt(numpy.pi)*scipy.special.gamma(1-self.beta)\
              /scipy.special.gamma(1.5-self.beta)
        Cbeta = 2**(self.beta-0.5)/(2*numpy.pi*Ibeta)
        alpha = self.beta-0.5
        coeff = (Cbeta*_a**(2*self.beta-2))*(numpy.sin(alpha*numpy.pi))\
                /(_GM*2*numpy.pi**2)
        integral = scipy.integrate.quad(self.f1_beta_gt05_integral, 
            a=0, b=Erel, args=(Erel,psi0) )[0]
        return coeff*integral
    
    def _f1_beta_gt05_integral_Hernquist(self,psi,Erel,psi0):
        """Integral for calculating f1 for a Hernquist when 0.5 < beta < 1.0"""
        psiTilde = psi/psi0
        # Absolute value because the answer normally comes out imaginary?
        denom = numpy.abs( (Erel-psi)**(1.5-self.beta) ) 
        numer = ((4-2*self.beta)*numpy.power(1-psiTilde,2*self.beta-1)\
              *numpy.power(psiTilde,3-2*self.beta)+(1-2*self.beta)\
              *numpy.power(1-psiTilde,2*self.beta-2)\
              *numpy.power(psiTilde,4-2*self.beta))
        return numer/denom
