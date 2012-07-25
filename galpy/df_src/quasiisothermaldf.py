#A 'Binney' quasi-isothermal DF
import math
import numpy
from scipy import optimize, interpolate
from galpy import potential
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
        self._pot= pot
        if _precomputevcirc:
            if _precomputevcircrmax is None:
                _precomputevcircrmax= 5*self._hr
            self._precomputevcircrmax= _precomputevcircrmax
            self._precomputevcircnr= _precomputevcircnr
            self._precomputevcircrgrid= numpy.linspace(0.00001,self._precomputevcircrmax,self._precomputevcircnr)
            self._vcircs= numpy.array([potential.vcirc(self._pot,r) for r in self._precomputevcircrgrid])
            #Spline interpolate
            self._vcircInterp= interpolate.InterpolatedUnivariateSpline(self._precomputevcircrgrid,self._vcircs,k=3)
        else:
            self._precomputevcircrmax= 0.
            self._vcircInterp= None
            self._vcircs= None
            self._precomputevcircnr= None
            self._precomputevcircrgrid= None
        self._precomputevcirc= _precomputevcirc
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
        #First calculate rg
        thisrg= self.rg(lz)
        #Then calculate the epicycle and vertical frequencies
        kappa, nu= self._calc_epifreq(thisrg), self._calc_verticalfreq(thisrg)
        return None

    def _calc_epifreq(self,r):
        """
        NAME:
           _calc_epifreq
        PURPOSE:
           calculate the epicycle frequency at r
        INPUT:
           r - radius
        OUTPUT:
           kappa
        HISTORY:
           2012-07-25 - Written - Bovy (IAS@MPIA)
        NOTE:
           takes about 0.1 ms for a Miyamoto-Nagai potential
        """
        return potential.epifreq(self._pot,r)

    def _calc_verticalfreq(self,r):
        """
        NAME:
           _calc_verticalfreq
        PURPOSE:
           calculate the vertical frequency at r
        INPUT:
           r - radius
        OUTPUT:
           nu
        HISTORY:
           2012-07-25 - Written - Bovy (IAS@MPIA)
        NOTE:
           takes about 0.05 ms for a Miyamoto-Nagai potential
        """
        return potential.verticalfreq(self._pot,r)

    def rg(self,lz):
        """
        NAME:
           rg
        PURPOSE:
           calculate the radius of a circular orbit of Lz
        INPUT:
           lz - Angular momentum
        OUTPUT:
           radius
        HISTORY:
           2012-07-25 - Written - Bovy (IAS@MPIA)
        NOTE:
           seems to take about ~0.5 ms for a Miyamoto-Nagai potential; 
           ~0.75 ms for a MWPotential
           about the same with or without interpolation of the rotation curve
        """
        #Find interval
        rstart= _rgFindStart(5.*self._hr,
                             self._vcircInterp,lz,self._precomputevcircrmax,
                             self._pot)
        return optimize.brentq(_rgfunc,0.0000001,rstart,
                               args=(self._vcircInterp,lz,
                                     self._precomputevcircrmax,self._pot))
        
def _rgfunc(rg,vcircInterp,lz,rmax,pot):
    """Function that gives rvc-lz"""
    if rg >= rmax:
        thisvcirc= potential.vcirc(pot,rg)
    else:
        thisvcirc= vcircInterp(rg)
    return rg*thisvcirc-lz

def _rgFindStart(rg,vcircInterp,lz,rmax,pot):
    """find a starting interval for rg"""
    rtry= 2.*rg
    while _rgfunc(rtry,vcircInterp,lz,rmax,pot) < 0.:
        rtry*= 2.
    return rtry
