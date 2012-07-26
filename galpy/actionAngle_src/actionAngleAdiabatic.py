###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleAdiabatic
#
#             wrapper around actionAngleAxi (adiabatic approximation) to do
#             this for any (x,v)
#
#      methods:
#              JR
#              Jphi
#              Jz
#              angleR
#              anglez
#              TR
#              Tphi
#              Tz
#              I
#              calcRapRperi
#              calcEL
###############################################################################
import math as m
import numpy as nu
from actionAngleAxi import actionAngleAxi
from actionAngle import actionAngle
class actionAngleAdiabatic():
    """Action-angle formalism for axisymmetric potentials using the adiabatic approximation"""
    def __init__(self,*args,**kwargs):
        """
        NAME:
           __init__
        PURPOSE:
           initialize an actionAngleAdiabatic object
        INPUT:
              pot= potential or list of potentials (planarPotentials)
        OUTPUT:
        HISTORY:
            2012-07-26 - Written - Bovy (IAS@MPIA)
        """
        if not kwargs.has_key('pot'):
            raise IOError("Must specify pot= for actionAngleAxi")
        self._pot= kwargs['pot']
        return None
    
    def __call__(self,*args,**kwargs):
        """
        NAME:
           __call__
        PURPOSE:
           evaluate the actions (jr,lz,jz)
        INPUT:
           Either:
              a) R,vR,vT,z,vz
              b) Orbit instance: initial condition used if that's it, orbit(t)
                 if there is a time given as well
           scipy.integrate.quadrature keywords
        OUTPUT:
           (jr,lz,jz), where jr=[jr,jrerr], and jz=[jz,jzerr]
        HISTORY:
           2012-07-26 - Written - Bovy (IAS@MPIA)
        """
        #Set up the actionAngleAxi object
        meta= actionAngle(*args)
        if isinstance(self._pot,list):
            thispot= [p.toPlanar() for p in self._pot]
        else:
            thispot= self._pot.toPlanar()
        if isinstance(self._pot,list):
            thisverticalpot= [p.toVertical(meta._R) for p in self._pot]
        else:
            thisverticalpot= self._pot.toVertical(meta._R)
        aAAxi= actionAngleAxi(*args,pot=thispot,
                               verticalPot=thisverticalpot)
        return (aAAxi.JR(**kwargs),aAAxi._R*aAAxi._vT,aAAxi.Jz(**kwargs))
