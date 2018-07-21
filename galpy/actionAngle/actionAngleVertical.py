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
from .actionAngle import *
from galpy.potential.linearPotential import linearPotential, \
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
        self._parse_eval_args(*args,_noOrbUnitsCheck=True,**kwargs)
        self._z= self._eval_z
        self._vz= self._eval_vz
        if not 'pot' in kwargs: #pragma: no cover
            raise IOError("Must specify pot= for actionAngleVertical")
        self._verticalpot= kwargs['pot']
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
        if zmax == -9999.99: return nu.array([9999.99,nu.nan])
        Ez= calcEz(self._z,self._vz,self._verticalpot)
        self._Jz= 2.*integrate.quad(_JzIntegrand,0.,zmax,
                                    args=(Ez,self._verticalpot),
                                    **kwargs)[0]/nu.pi
        return self._Jz

    def Tz(self,**kwargs): #pragma: no cover
        """
        NAME:
           Tz
        PURPOSE:
           Calculate the vertical period
        INPUT:
           +scipy.integrate.quad keywords
        OUTPUT:
           T_z(z,vz)*vc/ro + estimate of the error
        HISTORY:
           2012-06-01 - Written - Bovy (IAS)
        """
        if hasattr(self,'_Tz'):
            return self._Tz
        zmax= self.calczmax()
        Ez= calcEz(self._z,self._vz,self._verticalpot)
        self._Tz= 4.*integrate.quad(_TzIntegrand,0.,zmax,
                                    args=(Ez,self._verticalpot),
                                    **kwargs)[0]
        return self._Tz

    def anglez(self,**kwargs): #pragma: no cover
        """
        NAME:
           anglez
        PURPOSE:
           Calculate the vertical angle
        INPUT:
           +scipy.integrate.quad keywords
        OUTPUT:
           angle_z(z,vz)*vc/ro + estimate of the error
        HISTORY:
           2012-06-01 - Written - Bovy (IAS)
        """
        if hasattr(self,'_anglez'):
            return self._anglez
        zmax= self.calczmax()
        Ez= calcEz(self._z,self._vz,self._verticalpot)
        Tz= self.Tz(**kwargs)
        self._anglez= 2.*nu.pi*(nu.array(integrate.quad(_TzIntegrand,0.,nu.fabs(self._z),
                                                        args=(Ez,self._verticalpot),
                                                        **kwargs)))/Tz[0]
        if self._z >= 0. and self._vz >= 0.:
            pass
        elif self._z >= 0. and self._vz < 0.:
            self._anglez[0]= nu.pi-self._anglez[0]
        elif self._z < 0. and self._vz <= 0.:
            self._anglez[0]= nu.pi+self._anglez[0]
        else:
            self._anglez[0]= 2.*nu.pi-self._anglez[0]
        return self._anglez

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
        if hasattr(self,'_zmax'): #pragma: no cover
            return self._zmax
        Ez= calcEz(self._z,self._vz,self._verticalpot)
        if self._vz == 0.: #We are exactly at the maximum height
            zmax= nu.fabs(self._z)
        else:
            zstart= self._z
            try:
                zend= _zmaxFindStart(self._z,Ez,self._verticalpot)
            except OverflowError: #pragma: no cover
                zmax= -9999.99
            else:
                zmax= optimize.brentq(_zmaxEq,zstart,zend,
                                      (Ez,self._verticalpot))
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
    return evaluatelinearPotentials(pot,z,use_physical=False)

def _JzIntegrand(z,Ez,pot):
    """The J_z integrand"""
    return nu.sqrt(2.*(Ez-potentialVertical(z,pot)))

def _TzIntegrand(z,Ez,pot): #pragma: no cover
    """The T_z integrand"""
    return 1./_JzIntegrand(z,Ez,pot)

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
    if z == 0.: ztry= 0.00001
    else: ztry= 2.*nu.fabs(z)
    while (Ez-potentialVertical(ztry,pot)) > 0.:
        ztry*= 2.
        if ztry > 100.: #pragma: no cover
            raise OverflowError
    return ztry

