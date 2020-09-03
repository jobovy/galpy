###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleHarmonicInverse
#
#             Calculate (x,v) coordinates for the harmonic oscillator from 
#             given actions-angle coordinates
#
###############################################################################
import numpy
from .actionAngleInverse import actionAngleInverse
from ..util import conversion
class actionAngleHarmonicInverse(actionAngleInverse):
    """Inverse action-angle formalism for the one-dimensional harmonic oscillator"""
    def __init__(self,*args,**kwargs):
        """
        NAME:

           __init__

        PURPOSE:

           initialize an actionAngleHarmonicInverse object

        INPUT:

           omega= frequency (can be Quantity)

           ro= distance from vantage point to GC (kpc; can be Quantity)

           vo= circular velocity at ro (km/s; can be Quantity)

        OUTPUT:
        
           instance

        HISTORY:

           2018-04-08 - Started - Bovy (UofT)

        """
        actionAngleInverse.__init__(self,*args,**kwargs)
        if not 'omega' in kwargs: #pragma: no cover
            raise IOError("Must specify omega= for actionAngleHarmonic")
        omega= conversion.parse_frequency(kwargs.get('omega'),
                                          ro=self._ro,vo=self._vo)
        self._omega= omega
        return None
    
    def _evaluate(self,j,angle,**kwargs):
        """
        NAME:

           __call__

        PURPOSE:

           evaluate the phase-space coordinates (x,v) for a number of angles on a single torus

        INPUT:

           j - action (scalar)

           angle - angle (array [N])

        OUTPUT:

           [x,vx]

        HISTORY:

           2018-04-08 - Written - Bovy (UofT)

        """
        return self._xvFreqs(j,angle,**kwargs)[:2]
        
    def _xvFreqs(self,j,angle,**kwargs):
        """
        NAME:

           xvFreqs

        PURPOSE:

           evaluate the phase-space coordinates (x,v) for a number of angles on a single torus as well as the frequency

        INPUT:

           j - action (scalar)

           angle - angle (array [N])

        OUTPUT:

           ([x,vx],Omega)

        HISTORY:

           2018-04-08 - Written - Bovy (UofT)

        """
        amp= numpy.sqrt(2.*j/self._omega)
        x= amp*numpy.sin(angle)
        vx= amp*self._omega*numpy.cos(angle)
        return (x,vx,self._omega)
        
    def _Freqs(self,j,**kwargs):
        """
        NAME:

           Freqs

        PURPOSE:

           return the frequency corresponding to a torus

        INPUT:

           j - action (scalar)

        OUTPUT:

           (Omega)

        HISTORY:

           2018-04-08 - Written - Bovy (UofT)

        """
        return self._omega
