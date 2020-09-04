# Class that implements the anisotropic spherical Hernquist DF with constant
# beta parameter
import numpy
import scipy.special
import scipy.integrate
from ..util import conversion
from ..potential import evaluatePotentials,HernquistPotential
from .constantbetadf import constantbetadf

class constantbetaHernquistdf(constantbetadf):
    """Class that implements the anisotropic spherical Hernquist DF with constant beta parameter"""
    def __init__(self,pot=None,beta=0,use_BD02=True,ro=None,vo=None):
        """
        NAME:

            __init__

        PURPOSE:

            Initialize a DF with constant anisotropy

        INPUT:

            pot - Hernquist potential which determines the DF

            beta - anisotropy parameter, must be in range [-0.5, 1.0)

            use_BD02 - Use Baes & Dejonghe (2002) solution for f(E) when
                non-trivial algebraic solution does not exist

        OUTPUT:

            None

        HISTORY:

            2020-07-22 - Written
        """
        assert isinstance(pot,HernquistPotential),'pot= must be potential.HernquistPotential'
        assert -0.5 <= beta and beta < 1.0,'Beta must be in range [-0.5,1.0)'
        self._use_BD02 = use_BD02
        constantbetadf.__init__(self,pot=pot,beta=beta,ro=ro,vo=vo)

    def _call_internal(self,*args):
        """
        NAME:

            _call_internal

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
        return L**(-2*self._beta)*fE

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
        E= conversion.parse_energy(E,vo=self._vo)
        psi0 = -evaluatePotentials(self._pot,0,0,use_physical=False)
        Erel = -E
        Etilde = Erel/psi0
        # Handle potential E outside of bounds
        Etilde_out = numpy.where(numpy.logical_or(Etilde<0,Etilde>1))[0]
        if len(Etilde_out)>0:
            # Dummy variable now and 0 later, prevents numerical issues?
            Etilde[Etilde_out]=0.5

        # First check algebraic solutions
        _GMa = psi0*self._pot.a**2.
        if self._beta == 0.:
            fE = numpy.power((2**0.5)*((2*numpy.pi)**3)*((_GMa)**1.5),-1)\
               *(numpy.sqrt(Etilde)/numpy.power(1-Etilde,2))\
               *((1-2*Etilde)*(8*numpy.power(Etilde,2)-8*Etilde-3)\
               +((3*numpy.arcsin(numpy.sqrt(Etilde)))\
               /numpy.sqrt(Etilde*(1-Etilde))))
        elif self._beta == 0.5:
            fE = (3*Etilde**2)/(4*(numpy.pi**3)*_GMa)
        elif self._beta == -0.5:
            fE = ((20*Etilde**3-20*Etilde**4+6*Etilde**5)\
               /(1-Etilde)**4)/(4*numpy.pi**3*(_GMa)**2)
        elif self._use_BD02:
            fE = self._fE_BD02(Etilde)
        elif self._beta < 1.0 and self._beta > 0.5:
            fE = self._fE_beta_gt05(Erel)
        elif self._beta < 0.5 and self._beta > -0.5:
            fE = self._fE_beta_gtm05_lt05(Erel)
        if len(Etilde_out)>0:
            fE[Etilde_out] = 0
        return fE

    def _fE_beta_gt05(self,Erel):
        """Calculate fE for a Hernquist model when 0.5 < beta < 1.0"""
        psi0 = -1*evaluatePotentials(self._pot,0,0,use_physical=False) 
        _a = self._pot.a
        _GM = psi0*_a
        Ibeta = numpy.sqrt(numpy.pi)*scipy.special.gamma(1-self._beta)\
              /scipy.special.gamma(1.5-self._beta)
        Cbeta = 2**(self._beta-0.5)/(2*numpy.pi*Ibeta)
        alpha = self._beta-0.5
        coeff = (Cbeta*_a**(2*self._beta-2))*(numpy.sin(alpha*numpy.pi))\
                /(_GM*2*numpy.pi**2)
        if hasattr(Erel,'shape'):
            _Erel_shape = Erel.shape
            _Erel_flat = Erel.flatten()
            integral = numpy.zeros_like(_Erel_flat)
            for ii in range(len(_Erel_flat)):
                integral[ii] = scipy.integrate.quad(
                    self._fE_beta_gt05_integral, a=0, b=_Erel_flat[ii], 
                    args=(_Erel_flat[ii],psi0))[0]
        else:
            integral = scipy.integrate.quad(
                self._fE_beta_gt05_integral, a=0, b=Erel, 
                args=(Erel,psi0))[0]
        return coeff*integral
    
    def _fE_beta_gt05_integral(self,psi,Erel,psi0):
        """Integral for calculating fE for a Hernquist when 0.5 < beta < 1.0"""
        psiTilde = psi/psi0
        # Absolute value because the answer normally comes out imaginary?
        denom = numpy.abs( (Erel-psi)**(1.5-self._beta) ) 
        numer = ((4-2*self._beta)*numpy.power(1-psiTilde,2*self._beta-1)\
              *numpy.power(psiTilde,3-2*self._beta)+(1-2*self._beta)\
              *numpy.power(1-psiTilde,2*self._beta-2)\
              *numpy.power(psiTilde,4-2*self._beta))
        return numer/denom
        
    def _fE_beta_gtm05_lt05(self,Erel):
        """Calculate fE for a Hernquist model when -0.5 < beta < 0.5"""
        psi0 = -1*evaluatePotentials(self._pot,0,0,use_physical=False) 
        _a = self._pot.a
        _GM = psi0*_a
        alpha = 0.5-self._beta
        Ibeta = numpy.sqrt(numpy.pi)*scipy.special.gamma(1-self._beta)\
              /scipy.special.gamma(1.5-self._beta)
        Cbeta = 2**(self._beta-0.5)/(2*numpy.pi*Ibeta*alpha)
        coeff = (Cbeta*_a**(2*self._beta-1))*(numpy.sin(alpha*numpy.pi))\
                /((_GM**2)*2*numpy.pi**2)
        if hasattr(Erel,'shape'):
            _Erel_shape = Erel.shape
            _Erel_flat = Erel.flatten()
            integral = numpy.zeros_like(_Erel_flat)
            for ii in range(len(_Erel_flat)):
                integral[ii] = scipy.integrate.quad(
                    self._fE_beta_gt05_integral, a=0, b=_Erel_flat[ii], 
                    args=(_Erel_flat[ii],psi0))[0]
        else:
            integral = scipy.integrate.quad(
                self._fE_beta_gt05_integral, a=0, b=Erel, 
                args=(Erel,psi0))[0]
        return coeff*integral

    def _fE_beta_gtm05_lt05_integral(self,psi,Erel,psi0):
        """Integral for calculating fE for a Hernquist when -0.5 < beta < 0.5"""
        psiTilde = psi/psi0
        # Absolute value because the answer normally comes out imaginary?
        denom = numpy.abs( (Erel-psi)**(0.5-self._beta) ) 
        numer = (4-2*self._beta-3*psiTilde)\
            *numpy.power(1-psiTilde,2*self._beta-2)\
            *numpy.power(psiTilde,3-2*self._beta)
        return numer/denom

    def _fE_BD02(self,Erel):
        """Calculate fE according to the hypergeometric solution of Baes & 
        Dejonghe (2002)"""
        coeff = (2.**self._beta/(2.*numpy.pi)**2.5)*scipy.special.gamma(5.-2.*self._beta)/\
                ( scipy.special.gamma(1.-self._beta)*scipy.special.gamma(3.5-self._beta) )
        fE = coeff*numpy.power(Erel,2.5-self._beta)*\
            scipy.special.hyp2f1(5.-2.*self._beta,1.-2.*self._beta,3.5-self._beta,Erel)
        return fE
        
    def _icmf(self,ms):
        '''Analytic expression for the normalized inverse cumulative mass 
        function. The argument ms is normalized mass fraction [0,1]'''
        return self._pot.a*numpy.sqrt(ms)/(1-numpy.sqrt(ms))
