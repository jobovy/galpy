#A 'Binney' quasi-isothermal DF
import math
import numpy
class quasiisothermaldf:
    """Class that represents a 'Binney' quasi-isothermal DF"""
    def __init__(self,hr,sr,sz,hsr,hsz):
        """
        NAME:
           __init__
        PURPOSE:
           Initialize a quasi-isothermal DF
        INPUT:
           hr - radial scale length
           sr - radial velocity dispersion at the solar radius
           sz - vertical velocity dispersion at the solar radius
           hsr - radial-velocity-dispersion scale length
           hsz - vertial-velocity-dispersion scale length
        OUTPUT:
           object
        HISTORY:
           2012-07-25 - Started - Bovy (IAS@MPIA)
        """
        self._hr= hr
        self._sr= sr
        self._sz= sz
        self._hsr= hsr
        self._hsz= hsz
        return None

    def __call__(self,jr,lz,jz,log=False):
        """
        NAME:
           __call__
        PURPOSE:
           return the DF
        INPUT:
           jr - radial action
           lz - z-component of angular momentum
           jz - vertical action
           log= if True, return the natural log
        OUTPUT:
           value of DF
        HISTORY:
           2012-07-25 - Written - Bovy (IAS@MPIA)
        """
        
