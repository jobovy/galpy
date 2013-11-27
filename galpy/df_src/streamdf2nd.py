#The DF of a tidal stream
import numpy
from galpy import potential
from galpy.orbit import Orbit
from galpy.util import bovy_coords
from galpy.actionAngle_src.actionAngleIsochroneApprox import actionAngleIsochroneApprox
class streamdf:
    """The DF of a tidal stream"""
    def __init__(self,sigO1,sigO2,siga,
                 covar,
                 progenitor=None,pot=None,aA=None,
                 ts=None,integrate_method='dopr54_c'):
        """
        NAME:
           __init__
        PURPOSE:
           Initialize a quasi-isothermal DF
        INPUT:
           sigO1 - largest eigenvalue of frequency covariance matrix
           sigO2 - second largest eigenvalue of frequency covariance matrix
           siga - dispersion in angle
           sigO3= smallest eigenvalue of frequency covariance matrix (default = sigO2)
           covar - covariance matrix as rotation matrix: either specified using 2 angles in a list or as a unit vector (covariance is supposed to have two equal small eigenvalues)
           progenitor= progenitor orbit as Orbit instance 
           pot= Potential instance or list thereof
           aA= actionAngle instance used to convert (x,v) to actions
           integrate_method= (default: 'dopr54_c') integration method to use
        OUTPUT:
           object
        HISTORY:
           2013-11-22 - Started - Bovy (IAS)
        """
        self._sigO1= sigO1
        self._sigO2= sigO2
        if pot is None:
            raise IOError("pot= must be set")
        self._pot= pot
        self._integrate_method= integrate_method
        if aA is None:
            raise IOError("aA= must be set")
        self._aA= aA
        self.progenitor= progenitor
        if not isinstance(self.progenitor,Orbit):
            raise IOError('progenitor= kwargs needs to be an Orbit instance')
        #Integrate progenitor
        self._integrate_method= integrate_method
        self._integrate_progenitor()
        #Setup model covariance matrix, either given as a unit vector or 2 angles

    def _integrate_progenitor(self):
        """Integrate the progenitor orbit forward and backward"""
        #load actionAngleIsochroneApprox object which has this capability
        if not isinstance(self._aA,actionAngleIsochroneApprox):
            aAI= actionAngleIsochroneApprox(b=1.,pot=self._pot,
                                            tintJ=100.,
                                            ntintJ=10000,
                                            integrate_method=self._integrate_method)
        else:
            aAI= self._aA
        R,vR,vT,z,vz,phi= aAI._parse_args(True,self.progenitor)
        self.progenitor._orb.t= numpy.empty(R.shape[1])
        self.progenitor._orb.t[aAI._ntintJ-1:]= aAI._tsJ
        self.progenitor._orb.t[:aAI._ntintJ-1]= -aAI._tsJ[1:][::-1]
        self.progenitor._orb.orbit= numpy.empty((len(self.progenitor._orb.t),6))
        self.progenitor._orb.orbit[:,0]= R
        self.progenitor._orb.orbit[:,1]= vR
        self.progenitor._orb.orbit[:,2]= vT
        self.progenitor._orb.orbit[:,3]= z
        self.progenitor._orb.orbit[:,4]= vz
        self.progenitor._orb.orbit[:,5]= phi
        return None
