###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleRazorThinSlab
#
#             Calculate actions-angle coordinates for a one-dimensional, 
#             razor-thin slab (or sheet)
#
#      methods:
#             __call__: returns (j)
#             actionsFreqs: returns (j,omega)
#             actionsFreqsAngles: returns (j,omega,a)
#
###############################################################################
import numpy
from galpy.actionAngle_src.actionAngle import actionAngle
from galpy.util import bovy_conversion
_APY_LOADED= True
try:
    from astropy import units
except ImportError:
    _APY_LOADED= False
class actionAngleRazorThinSlab(actionAngle):
    """Action-angle formalism for the one-dimensional razor-thin slab with potetial Sigma |x| (Bovy 2018)"""
    def __init__(self,*args,**kwargs):
        """
        NAME:

           __init__

        PURPOSE:

           initialize an actionAngleRazorThinSlab object

        INPUT:

           Sigma= surface-density of the slab

           ro= distance from vantage point to GC (kpc; can be Quantity)

           vo= circular velocity at ro (km/s; can be Quantity)

        OUTPUT:
        
           instance

        HISTORY:

           2018-06-03 - Written - Bovy (Uoft)

        """
        actionAngle.__init__(self,
                             ro=kwargs.get('ro',None),vo=kwargs.get('vo',None))
        if not 'Sigma' in kwargs: #pragma: no cover
            raise IOError("Must specify Sigma= for actionAngleRazorThinSlab")
        Sigma= kwargs.get('Sigma')
        # BOVY: Deal wit unitful Sigmalitude
        self._Sigma= Sigma
        return None
    
    def _evaluate(self,*args,**kwargs):
        """
        NAME:
           __call__ (_evaluate)
        PURPOSE:
           evaluate the action
        INPUT:
           Either:
              a) x,vx:
                 1) floats: phase-space value for single object (each can be a Quantity)
                 2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
        OUTPUT:
           action
        HISTORY:
           2018-06-03 - Written - Bovy (UofT)
        """
        if len(args) == 2: # x,vx
            x,vx= args
            return (2/3./numpy.pi*(2*self._Sigma*numpy.fabs(x)+vx**2.)**(3./2)\
                        /self._Sigma)
        else: # pragma: no cover
            raise ValueError('actionAngleRazorThinSlab __call__ input not understood')

    def _actionsFreqs(self,*args,**kwargs):
        """
        NAME:
           actionsFreqs (_actionsFreqs)
        PURPOSE:
           evaluate the action and frequency
        INPUT:
           Either:
              a) x,vx:
                 1) floats: phase-space value for single object (each can be a Quantity)
                 2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
        OUTPUT:
           action,frequency
        HISTORY:
           2018-04-08 - Written - Bovy (UofT)
        """
        if len(args) == 2: # x,vx
            x,vx= args
            return (2/3./numpy.pi*(2*self._Sigma*numpy.fabs(x)+vx**2.)**(3./2)\
                        /self._Sigma,
                    self._Sigma*numpy.pi/numpy.sqrt(2*self._Sigma*numpy.fabs(x)
                                                  +vx**2.)/2.)
        else: # pragma: no cover
            raise ValueError('actionAngleRazorThinSlab __call__ input not understood')

    def _actionsFreqsAngles(self,*args,**kwargs):
        """
        NAME:
           actionsFreqsAngles (_actionsFreqsAngles)
        PURPOSE:
           evaluate the action, frequency, and angle
        INPUT:
           Either:
              a) x,vx:
                 1) floats: phase-space value for single object (each can be a Quantity)
                 2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
        OUTPUT:
           action,frequency,angle
        HISTORY:
           2018-04-08 - Written - Bovy (UofT)
        """
        if len(args) == 2: # x,vx
            x,vx= args
            return (2/3./numpy.pi*(2*self._Sigma*numpy.fabs(x)+vx**2.)**(3./2)\
                        /self._Sigma,
                    self._Sigma*numpy.pi/numpy.sqrt(2*self._Sigma*numpy.fabs(x)
                                                  +vx**2.)/2.,
                    numpy.pi/2.*(1.-numpy.sign(x)*vx\
                                     /numpy.sqrt((2*self._Sigma*numpy.fabs(x)
                                                  +vx**2.)))\
                        +(x<0.)*numpy.pi)
        else: # pragma: no cover
            raise ValueError('actionAngleRazorThinSlab __call__ input not understood')
