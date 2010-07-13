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
        else:
            if len(args) == 2:
                vxvv= args[0](args[1])
            else:
                vxvv= args[0].vxvv
            self._R= vxvv[0]
            self._vR= vxvv[1]
            self._vT= vxvv[2]
        return None
