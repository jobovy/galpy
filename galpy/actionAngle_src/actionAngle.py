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
                vxvv= args[0](args[1]).vxvv
            else:
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

class UnboundError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
