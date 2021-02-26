# Class that implements the anisotropic spherical Hernquist DF with constant
# beta parameter
import numpy
import scipy.special
import scipy.integrate
from ..util import conversion
from ..potential import evaluatePotentials,HernquistPotential
from .constantbetadf import _constantbetadf

class constantbetaHernquistdf(_constantbetadf):
    """Class that implements the anisotropic spherical Hernquist DF with constant beta parameter"""
    def __init__(self,pot=None,beta=0,ro=None,vo=None):
        """
        NAME:

            __init__

        PURPOSE:

            Initialize a Hernquist DF with constant anisotropy

        INPUT:

            pot - Hernquist potential which determines the DF

            beta - anisotropy parameter

        OUTPUT:

            None

        HISTORY:

            2020-07-22 - Written - Lane (UofT)
        """
        assert isinstance(pot,HernquistPotential),'pot= must be potential.HernquistPotential'
        _constantbetadf.__init__(self,pot=pot,beta=beta,ro=ro,vo=vo)
        self._psi0= -evaluatePotentials(self._pot,0,0,use_physical=False)
        self._potInf= 0.
        self._GMa = self._psi0*self._pot.a**2.
        # Final factor is mass to make the DF that of the mass density
        self._fEnorm= (2.**self._beta/(2.*numpy.pi)**2.5)\
            *scipy.special.gamma(5.-2.*self._beta)\
            /scipy.special.gamma(1.-self._beta)\
            /scipy.special.gamma(3.5-self._beta)\
            /self._GMa**(1.5-self._beta)\
            *self._psi0*self._pot.a
  
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
        Etilde= -conversion.parse_energy(E,vo=self._vo)/self._psi0
        # Handle potential E outside of bounds
        Etilde_out = numpy.where(numpy.logical_or(Etilde<0,Etilde>1))[0]
        if len(Etilde_out)>0:
            # Dummy variable now and 0 later, prevents numerical issues?
            Etilde[Etilde_out]=0.5
        # First check algebraic solutions, all adjusted such that DF = mass den
        if self._beta == 0.: # isotropic case
            sqrtEtilde= numpy.sqrt(Etilde)
            fE= self._psi0*self._pot.a\
                /numpy.sqrt(2.)/(2*numpy.pi)**3/self._GMa**1.5\
                *sqrtEtilde/(1-Etilde)**2.\
                *((1.-2.*Etilde)*(8.*Etilde**2.-8.*Etilde-3.)\
                  +((3.*numpy.arcsin(sqrtEtilde))\
                    /numpy.sqrt(Etilde*(1.-Etilde))))
        elif self._beta == 0.5:
            fE= (3.*Etilde**2.)/(4.*numpy.pi**3.*self._pot.a)
        elif self._beta == -0.5:
            fE= ((20.*Etilde**3.-20.*Etilde**4.+6.*Etilde**5.)\
                 /(1.-Etilde)**4)/(4.*numpy.pi**3.*self._GMa*self._pot.a)
        else:
            fE= self._fEnorm*numpy.power(Etilde,2.5-self._beta)*\
                scipy.special.hyp2f1(5.-2.*self._beta,1.-2.*self._beta,
                                     3.5-self._beta,Etilde)
        if len(Etilde_out) > 0:
            fE[Etilde_out]= 0.
        return fE

       
    def _icmf(self,ms):
        '''Analytic expression for the normalized inverse cumulative mass 
        function. The argument ms is normalized mass fraction [0,1]'''
        return self._pot.a*numpy.sqrt(ms)/(1-numpy.sqrt(ms))
