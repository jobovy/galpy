###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleHarmonic
#
#             Calculate actions-angle coordinates for the harmonic-oscillator
#
#      methods:
#             __call__: returns (j)
#             actionsFreqs: returns (j,omega)
#             actionsFreqsAngles: returns (j,omega,a)
#
###############################################################################
import numpy
from .actionAngle import actionAngle
from ..util import conversion
class actionAngleHarmonic(actionAngle):
    """Action-angle formalism for the one-dimensional harmonic oscillator"""
    def __init__(self,*args,**kwargs):
        """
        NAME:

           __init__

        PURPOSE:

           initialize an actionAngleHarmonic object

        INPUT:

           omega= frequencies (can be Quantity)

           ro= distance from vantage point to GC (kpc; can be Quantity)

           vo= circular velocity at ro (km/s; can be Quantity)

        OUTPUT:
        
           instance

        HISTORY:

           2018-04-08 - Written - Bovy (Uoft)

        """
        actionAngle.__init__(self,
                             ro=kwargs.get('ro',None),vo=kwargs.get('vo',None))
        if not 'omega' in kwargs: #pragma: no cover
            raise IOError("Must specify omega= for actionAngleHarmonic")
        self._omega= conversion.parse_frequency(kwargs.get('omega'),
                                                ro=self._ro,vo=self._vo)
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
           2018-04-08 - Written - Bovy (UofT)
        """
        if len(args) == 2: # x,vx
            x,vx= args
            return (vx**2./self._omega+self._omega*x**2.)/2.
        else: # pragma: no cover
            raise ValueError('actionAngleHarmonic __call__ input not understood')

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
            return ((vx**2./self._omega+self._omega*x**2.)/2.,
                    self._omega*numpy.ones_like(x))
        else: # pragma: no cover
            raise ValueError('actionAngleHarmonic __call__ input not understood')

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
            return ((vx**2./self._omega+self._omega*x**2.)/2.,
                    self._omega*numpy.ones_like(x),
                    numpy.arctan2(self._omega*x,vx))
        else: # pragma: no cover
            raise ValueError('actionAngleHarmonic __call__ input not understood')
