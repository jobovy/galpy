###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleAdiabatic
#
#             wrapper around actionAngleAxi (adiabatic approximation) to do
#             this for any (x,v)
#
#      methods:
#             __call__: returns (jr,lz,jz)
#
###############################################################################
import copy
import warnings
import math as m
import numpy as nu
from galpy.util import galpyWarning
from actionAngleAxi import actionAngleAxi
from actionAngle import actionAngle
try:
    import actionAngleAdiabatic_c
except IOError:
    warnings.warn("actionAngle_c extension module not loaded",galpyWarning)
    ext_loaded= False
else:
    ext_loaded= True
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

           gamma= (default=1.) replace Lz by Lz+gamma Jz in effective potential
        OUTPUT:
        HISTORY:
            2012-07-26 - Written - Bovy (IAS@MPIA)
        """
        if not kwargs.has_key('pot'):
            raise IOError("Must specify pot= for actionAngleAxi")
        self._pot= kwargs['pot']
        if ext_loaded and kwargs.has_key('c') and kwargs['c']:
            #print "BOVY: CHECK THAT POTENTIALS HAVE C IMPLEMENTATIONS"
            self._c= True
        else:
            self._c= False
        if kwargs.has_key('gamma'):
            self._gamma= kwargs['gamma']
        else:
            self._gamma= 1.
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
        if self._c or (ext_loaded and kwargs.has_key('c') and kwargs['c']):
            #print "BOVY: CHECK THAT POTENTIALS HAVE C IMPLEMENTATIONS"
            if len(args) == 5: #R,vR.vT, z, vz
                R,vR,vT, z, vz= args
            elif len(args) == 6: #R,vR.vT, z, vz, phi
                R,vR,vT, z, vz, phi= args
            else:
                meta= actionAngle(*args)
                R= meta._R
                vR= meta._vR
                vT= meta._vT
                z= meta._z
                vz= meta._vz
            if isinstance(R,float):
                R= nu.array([R])
                vR= nu.array([vR])
                vT= nu.array([vT])
                z= nu.array([z])
                vz= nu.array([vz])
            Lz= R*vT
            jr, jz, err= actionAngleAdiabatic_c.actionAngleAdiabatic_c(\
                self._pot,self._gamma,R,vR,vT,z,vz)
            if err == 0:
                return (jr,Lz,jz)
            else:
                raise RuntimeError("C-code for calculation actions failed; try with c=False")
        else:
            if (len(args) == 5 or len(args) == 6) \
                    and isinstance(args[0],nu.ndarray):
                ojr= nu.zeros((len(args[0])))
                olz= nu.zeros((len(args[0])))
                ojz= nu.zeros((len(args[0])))
                for ii in range(len(args[0])):
                    if len(args) == 5:
                        targs= (args[0][ii],args[1][ii],args[2][ii],
                                args[3][ii],args[4][ii])
                    elif len(args) == 6:
                        targs= (args[0][ii],args[1][ii],args[2][ii],
                                args[3][ii],args[4][ii],args[5][ii])
                    tjr,tlz,tjz= self(*targs,**copy.copy(kwargs))
                    ojr[ii]= tjr
                    ojz[ii]= tjz
                    olz[ii]= tlz
                return (ojr,olz,ojz)
            else:
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
                                       verticalPot=thisverticalpot,
                                       gamma=self._gamma)
                return (aAAxi.JR(**kwargs),aAAxi._R*aAAxi._vT,aAAxi.Jz(**kwargs))

    def JR(self,*args,**kwargs):
        """
        NAME:
           JR
        PURPOSE:
           evaluate the action jr
        INPUT:
           Either:
              a) R,vR,vT,z,vz
              b) Orbit instance: initial condition used if that's it, orbit(t)
                 if there is a time given as well
           scipy.integrate.quadrature keywords
        OUTPUT:
           Jr
        HISTORY:
           2012-07-30 - Written - Bovy (IAS@MPIA)
        """
        #Set up the actionAngleAxi object
        if isinstance(self._pot,list):
            thispot= [p.toPlanar() for p in self._pot]
        else:
            thispot= self._pot.toPlanar()
        aAAxi= actionAngleAxi(*args,pot=thispot,
                               gamma=self._gamma)
        return aAAxi.JR(**kwargs)

    def Jz(self,*args,**kwargs):
        """
        NAME:
           Jz
        PURPOSE:
           evaluate the action jz
        INPUT:
           Either:
              a) R,vR,vT,z,vz
              b) Orbit instance: initial condition used if that's it, orbit(t)
                 if there is a time given as well
           scipy.integrate.quadrature keywords
        OUTPUT:
           jz,jzerr
        HISTORY:
           2012-07-27 - Written - Bovy (IAS@MPIA)
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
                               verticalPot=thisverticalpot,
                               gamma=self._gamma)
        return aAAxi.Jz(**kwargs)

    def calcRapRperi(self,*args,**kwargs):
        """
        NAME:
           calcRapRperi
        PURPOSE:
           calculate the apocenter and pericenter radii
        INPUT:
           Either:
              a) R,vR,vT,z,vz
              b) Orbit instance: initial condition used if that's it, orbit(t)
                 if there is a time given as well
        OUTPUT:
           (rperi,rap)
        HISTORY:
           2013-11-27 - Written - Bovy (IAS)
        """
        #Set up the actionAngleAxi object
        if isinstance(self._pot,list):
            thispot= [p.toPlanar() for p in self._pot]
        else:
            thispot= self._pot.toPlanar()
        aAAxi= actionAngleAxi(*args,pot=thispot,
                               gamma=self._gamma)
        return aAAxi.calcRapRperi(**kwargs)
        
    def calczmax(self,*args,**kwargs):
        """
        NAME:
           calczmax
        PURPOSE:
           calculate the maximum height
        INPUT:
           Either:
              a) R,vR,vT,z,vz
              b) Orbit instance: initial condition used if that's it, orbit(t)
                 if there is a time given as well
        OUTPUT:
           zmax
        HISTORY:
           2012-06-01 - Written - Bovy (IAS)
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
                               verticalPot=thisverticalpot,
                               gamma=self._gamma)
        return aAAxi.calczmax(**kwargs)
