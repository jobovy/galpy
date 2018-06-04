###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleRazorThinSlabInverse
#
#             Calculate (x,v) coordinates for the one-dimensional, razor-thin
#             slab from given actions-angle coordinates
#
###############################################################################
import numpy
from galpy.actionAngle_src.actionAngleInverse import actionAngleInverse
from galpy.util import bovy_conversion
_APY_LOADED= True
try:
    from astropy import units
except ImportError:
    _APY_LOADED= False
class actionAngleRazorThinSlabInverse(actionAngleInverse):
    """Inverse action-angle formalism for the one-dimensional, razor-thin slab with potential Sigma |x|"""
    def __init__(self,*args,**kwargs):
        """
        NAME:

           __init__

        PURPOSE:

           initialize an actionAngleRazorThinSlabInverse object

        INPUT:

           Sigma= surface density of the slab (can be Quantity)

           ro= distance from vantage point to GC (kpc; can be Quantity)

           vo= circular velocity at ro (km/s; can be Quantity)

        OUTPUT:
        
           instance

        HISTORY:

           2018-06-03 - Started - Bovy (UofT)

        """
        actionAngleInverse.__init__(self,*args,**kwargs)
        if not 'Sigma' in kwargs: #pragma: no cover
            raise IOError("Must specify Sigma= for actionAngleRazorThinSlab")
        Sigma= kwargs.get('Sigma')
        # BOVY: Deal with unitful Sigma
        self._Sigma= Sigma
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

           2018-06-03 - Written - Bovy (UofT)

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

           2018-06-03 - Written - Bovy (UofT)

        """
        theta= numpy.atleast_1d(angle) % (2.*numpy.pi)
        theta[theta>numpy.pi]-= 2.*numpy.pi
        twoE= (3.*numpy.pi/2.*self._Sigma*j)**(2./3.)
        Omega= numpy.pi/2.*self._Sigma/numpy.sqrt(twoE)
        vx= (1.-2.*numpy.fabs(theta)/numpy.pi)*numpy.sqrt(twoE)
        x= (twoE-vx**2)/2./self._Sigma*theta/numpy.fabs(theta)
        x[vx**2.>twoE]= 0.
        return (x,vx,Omega)
        
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

           2018-06-03 - Written - Bovy (UofT)

        """
        twoE= (3.*numpy.pi/2.*self._Sigma*j)**(2./3.)
        return numpy.pi/2.*self._Sigma/numpy.sqrt(twoE)
