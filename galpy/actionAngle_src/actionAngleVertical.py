###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleVertical
#
#      methods:
#              Jz
#              anglez
#              Tz
#              calczmax
#              calcEz
###############################################################################
import math as m
import numpy as nu
from scipy import optimize, integrate
from actionAngle import *
from galpy.potential_src.linearPotential import linearPotential, \
    evaluatelinearPotentials
class actionAngleVertical(actionAngle):
    """Action-angle formalism for vertical integral using the adiabatic approximation"""
    def __init__(self,*args,**kwargs):
        """
        NAME:
           __init__
        PURPOSE:
           initialize an actionAngleVertical object
        INPUT:
           Either:
              a) z,vz
              b) Orbit instance: initial condition used if that's it, orbit(t)
                 if there is a time given as well
              pot= potential or list of potentials (planarPotentials)
        OUTPUT:
        HISTORY:
           2012-06-01 - Written - Bovy (IAS)
        """
        actionAngle.__init__(self,*args,**kwargs)
        if not kwargs.has_key('pot'):
            raise IOError("Must specify pot= for actionAngleVertical")
        self._pot= kwargs['pot']
        return None
    
    def Jz(self,**kwargs):
        """
        NAME:
           Jz
        PURPOSE:
           Calculate the vertical action
        INPUT:
           +scipy.integrate.quad keywords
        OUTPUT:
           J_z(z,vz)/ro/vc + estimate of the error
        HISTORY:
           2012-06-01 - Written - Bovy (IAS)
        """
        if hasattr(self,'_Jz'):
            return self._Jz
        zmax= self.calczmax()
        Ez= calcEz(self._z,self._vz,self._pot)
        self._Jz= (2.*nu.array(integrate.quad(_JzIntegrand,0.,zmax,
                                              args=(Ez,self._pot),
                                              **kwargs)))/nu.pi
        return self._Jz

    def calczmax(self):
        """
        NAME:
           calczmax
        PURPOSE:
           calculate the maximum height
        INPUT:
        OUTPUT:
           zmax
        HISTORY:
           2012-06-01 - Written - Bovy (IAS)
        """
        if hasattr(self,'_zmax'):
            return self._zmax
        Ez= calcEz(self._z,self._vz,self._pot)
        if self._vz == 0.: #We are exactly at the maximum height
            zmax= nu.fabs(self._z)
        else:
            zstart= self._z
            zend= _zmaxFindStart(self._z,Ez,self._pot)
            zmax= optimize.brentq(_zmaxEq,zstart,zend,
                                  (Ez,self._pot))
        self._zmax= zmax
        return self._zmax

def _zmaxEq(z,Ez,pot):
    """The vz=0 equation that needs to be solved to find zmax"""
    return Ez-potentialVertical(z,pot)

def calcEz(z,vz,pot):
    """
    NAME:
       calcEz
    PURPOSE:
       calculate the vertical energy
    INPUT:
       z - height (/ro)
       vz - vertical part of the velocity (/vc)
       pot - potential
    OUTPUT:
       Ez
    HISTORY:
       2012-06-01 - Written - Bovy (IAS)
    """                           
    return potentialVertical(z,pot)+vz**2./2.

def potentialVertical(z,pot):
    """
    NAME:
       potentialVertical
    PURPOSE:
       return the potential
    INPUT:
       z - height (/ro)
       pot - potential
    OUTPUT:
       Phi_z(z)
    HISTORY:
       2012-06-01 - Written - Bovy (IAS)
    """
    return evaluatelinearPotentials(z,pot)

def _JzIntegrand(z,Ez,pot):
    """The J_z integrand"""
    return nu.sqrt(2.*(Ez-potentialVertical(z,pot)))

def _zmaxFindStart(z,Ez,pot):
    """
    NAME:
       _zmaxFindStart
    PURPOSE:
       Find adequate end point to solve for zmax
    INPUT:
       z - height
       Ez - vertical energy
       pot - potential
    OUTPUT:
       zend
    HISTORY:
       2012-06-01 - Written - Bovy (IAS)
    """
    ztry= 2.*z
    while (Ez-potentialVertical(ztry,pot)) > 0.:
        ztry*= 2.
    return ztry

