#A 'Binney' quasi-isothermal DF
import math
import numpy
from galpy.potential import vcirc
class quasiisothermaldf:
    """Class that represents a 'Binney' quasi-isothermal DF"""
    def __init__(self,hr,sr,sz,hsr,hsz,pot=None,
                 _precomputevcirc=True,_precomputevcircrmax=None,
                 _precomputevcircnr=51):
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
           pot= Potential instance or list thereof
        OTHER INPUTS:
           _precomputevcirc= if True (default), pre-compute the circular velocity curve
           _precomputevcircrmax= if set, this is the maximum R for which to pre-compute vcirc (default: 5*hr
           _precomputevcircnr if set, number of R to pre-compute vc for (default: 51)
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
        if pot is None:
            raise IOError("pot= must be set")
        if _precomputevcirc:
            if _precomputevcircrmax is None:
                _precomputevcircrmax= 5*self._hr
            self._precomputevcircrmax= _precomputevcircrmax
            self._precomputevcircnr= _precomputevcircnr
            self._precomputevcircrgrid= numpy.linspace(0.00001,self._precomputevcircrmax,self._precomputevcircnr)
            self._vcircs= numpy.array([vcirc(pot,r) for r in self._precomputevcircrgrid])
            #Spline interpolate
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
        
