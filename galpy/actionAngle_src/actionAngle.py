import math as m
class actionAngle:
    """Top-level class for actionAngle classes"""
    def __init__(self,*args,**kwargs):
        """
        NAME:
           __init__
        PURPOSE:
           initialize an actionAngle object
        INPUT:
        OUTPUT:
        HISTORY:
           2010-07-11 - Written - Bovy (NYU)
        """
        if len(args) == 3: #R,vR.vT
            R,vR,vT= args
            self._R= R
            self._vR= vR
            self._vT= vT
            self._z= 0.
            self._vz= 0.
        elif len(args) == 5: #R,vR.vT, z, vz
            R,vR,vT, z, vz= args
            self._R= R
            self._vR= vR
            self._vT= vT
            self._z= z
            self._vz= vz
        elif len(args) == 6: #R,vR.vT, z, vz, phi
            R,vR,vT, z, vz, phi= args
            self._R= R
            self._vR= vR
            self._vT= vT
            self._z= z
            self._vz= vz
            self._phi= phi
        else:
            if len(args) == 2:
                vxvv= args[0](args[1])._orb.vxvv
            else:
                try:
                    vxvv= args[0]._orb.vxvv
                except AttributeError: #if we're given an OrbitTop instance
                    vxvv= args[0].vxvv
            self._R= vxvv[0]
            self._vR= vxvv[1]
            self._vT= vxvv[2]
            if len(vxvv) > 4:
                self._z= vxvv[3]
                self._vz= vxvv[4]
                if len(vxvv) > 5:
                    self._phi= vxvv[5]
            elif len(vxvv) > 3:
                self._phi= vxvv[3]
                self._z= 0.
                self._vz= 0.
            else:
                self._z= 0.
                self._vz= 0.
        if hasattr(self,'_z'): #calculate the polar angle
            if self._z == 0.: self._theta= m.pi/2.
            else: self._theta= m.atan(self._R/self._z)
        return None

    def __call__(self,*args,**kwargs):
        """
        NAME:
           __call__
        PURPOSE:
           evaluate the actions (jr,lz,jz)
        INPUT:

           Either:

              a) R,vR,vT,z,vz[,phi]:

                 1) floats: phase-space value for single object (phi is optional)

                 2) numpy.ndarray: [N] phase-space values for N objects 

              b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
                 
        OUTPUT:
           (jr,lz,jz)
        HISTORY:
           2014-01-03 - Written for top level - Bovy (IAS)
        """
        raise NotImplementedError("'__call__' method not implemented for this actionAngle module")

    def actionsFreqs(self,*args,**kwargs):
        """
        NAME:
           actionsFreqs
        PURPOSE:
           evaluate the actions and frequencies (jr,lz,jz,Omegar,Omegaphi,Omegaz)
        INPUT:

           Either:

              a) R,vR,vT,z,vz[,phi]:

                 1) floats: phase-space value for single object (phi is optional)

                 2) numpy.ndarray: [N] phase-space values for N objects 

              b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
                 
        OUTPUT:
            (jr,lz,jz,Omegar,Omegaphi,Omegaz)
        HISTORY:
           2014-01-03 - Written for top level - Bovy (IAS)
        """
        raise NotImplementedError("'actionsFreqs' method not implemented for this actionAngle module")

    def actionsFreqsAngles(self,*args,**kwargs):
        """
        NAME:
           actionsFreqsAngles
        PURPOSE:
           evaluate the actions, frequencies, and angles 
           (jr,lz,jz,Omegar,Omegaphi,Omegaz,angler,anglephi,anglez)
        INPUT:

           Either:

              a) R,vR,vT,z,vz,phi:

                 1) floats: phase-space value for single object (phi needs to be specified)

                 2) numpy.ndarray: [N] phase-space values for N objects 

              b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
                 
        OUTPUT:
            (jr,lz,jz,Omegar,Omegaphi,Omegaz,angler,anglephi,anglez)
        HISTORY:
           2014-01-03 - Written for top level - Bovy (IAS)
        """
        raise NotImplementedError("'actionsFreqsAngles' method not implemented for this actionAngle module")


class UnboundError(Exception): #pragma: no cover
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
