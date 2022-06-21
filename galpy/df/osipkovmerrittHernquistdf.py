# Class that implements the anisotropic spherical Hernquist DF with radially
# varying anisotropy of the Osipkov-Merritt type
import numpy
from ..util import conversion
from ..potential import evaluatePotentials, HernquistPotential
from .osipkovmerrittdf import _osipkovmerrittdf

class osipkovmerrittHernquistdf(_osipkovmerrittdf):
    """Class that implements the anisotropic spherical Hernquist DF with radially varying anisotropy of the Osipkov-Merritt type
    
    .. math::

        \\beta(r) = \\frac{1}{1+r_a^2/r^2}

    with :math:`r_a` the anistropy radius.

    """
    def __init__(self,pot=None,ra=1.4,ro=None,vo=None):
        """
        NAME:

            __init__

        PURPOSE:

            Initialize a Hernquist DF with Osipkov-Merritt anisotropy

        INPUT:

            pot - Hernquist potential which determines the DF

            ra - anisotropy radius (can be a Quantity)

           ro=, vo= galpy unit parameters

        OUTPUT:

            None

        HISTORY:

            2020-11-12 - Written - Bovy (UofT)
        """
        assert isinstance(pot,HernquistPotential), \
            'pot= must be potential.HernquistPotential'
        _osipkovmerrittdf.__init__(self,pot=pot,ra=ra,ro=ro,vo=vo)
        self._psi0= -evaluatePotentials(self._pot,0,0,use_physical=False)
        self._GMa = self._psi0*self._pot.a**2.
        self._a2overra2= self._pot.a**2./self._ra2
        # First factor is the mass to make the DF that of the mass density
        self._fQnorm= self._psi0*self._pot.a\
            /numpy.sqrt(2.)/(2*numpy.pi)**3/self._GMa**1.5
  
    def fQ(self,Q):
        """
        NAME:

            fQ

        PURPOSE

            Calculate the f(Q) portion of an Osipkov-Merritt Hernquist distribution function

        INPUT:

            Q - The Osipkov-Merritt 'energy' E-L^2/[2ra^2] (can be Quantity)

        OUTPUT:

            fQ - The value of the f(Q) portion of the DF

        HISTORY:

            2020-11-12 - Written - Bovy (UofT)

        """
        Qtilde= conversion.parse_energy(Q,vo=self._vo)/self._psi0
        # Handle potential Q outside of bounds
        Qtilde_out = numpy.where(numpy.logical_or(Qtilde<0,Qtilde>1))[0]
        if len(Qtilde_out)>0:
            # Dummy variable now and 0 later, prevents numerical issues
            Qtilde[Qtilde_out]=0.5
        sqrtQtilde= numpy.sqrt(Qtilde)
        # The 'ergodic' part
        fQ= sqrtQtilde/(1-Qtilde)**2.\
            *((1.-2.*Qtilde)*(8.*Qtilde**2.-8.*Qtilde-3.)\
              +((3.*numpy.arcsin(sqrtQtilde))\
                /numpy.sqrt(Qtilde*(1.-Qtilde))))
        # The other part
        fQ+= 8.*self._a2overra2*sqrtQtilde*(1.-2.*Qtilde)
        if len(Qtilde_out) > 0:
            fQ[Qtilde_out]= 0.
        return self._fQnorm*fQ
       
    def _icmf(self,ms):
        '''Analytic expression for the normalized inverse cumulative mass 
        function. The argument ms is normalized mass fraction [0,1]'''
        return self._pot.a*numpy.sqrt(ms)/(1-numpy.sqrt(ms))
