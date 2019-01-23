import os
import sys
import distutils.sysconfig as sysconfig
import copy
import ctypes
import ctypes.util
import warnings
from functools import wraps
import numpy
from numpy.ctypeslib import ndpointer
from scipy import interpolate
from galpy.util import multi, galpyWarning
from .Potential import Potential
from galpy.util.bovy_conversion import physical_conversion
_DEBUG= False
#Find and load the library
_lib= None
outerr= None
PY3= sys.version > '3'
if PY3:
    _ext_suffix= sysconfig.get_config_var('EXT_SUFFIX')
else: #pragma: no cover
    _ext_suffix= '.so'
for path in sys.path:
    try:
        _lib = ctypes.CDLL(os.path.join(path,'galpy_interppotential_c%s' % _ext_suffix))
    except OSError as e:
        if os.path.exists(os.path.join(path,'galpy_interppotential_c%s' % _ext_suffix)): #pragma: no cover
            outerr= e
        _lib = None
    else:
        break
if _lib is None: #pragma: no cover
    if not outerr is None:
        warnings.warn("interppotential_c extension module not loaded, because of error '%s' " % outerr,
                      galpyWarning)
    else:
        warnings.warn("interppotential_c extension module not loaded, because galpy_interppotential_c%s image was not found" % _ext_suffix,
                      galpyWarning)
    ext_loaded= False
else:
    ext_loaded= True

def scalarVectorDecorator(func):
    """Decorator to return scalar outputs as a set"""
    @wraps(func)
    def scalar_wrapper(*args,**kwargs):
        if numpy.array(args[1]).shape == () \
                and numpy.array(args[2]).shape == (): #only if both R and z are scalars
            scalarOut= True
            args= (args[0],numpy.array([args[1]]),numpy.array([args[2]]))
        elif numpy.array(args[1]).shape == () \
                and not numpy.array(args[2]).shape == (): #R scalar, z vector
            scalarOut= False
            args= (args[0],args[1]*numpy.ones_like(args[2]),args[2])
        elif not numpy.array(args[1]).shape == () \
                and numpy.array(args[2]).shape == (): #R vector, z scalar
            scalarOut= False
            args= (args[0],args[1],args[2]*numpy.ones_like(args[1]))
        else:
            scalarOut= False
        result= func(*args,**kwargs)
        if scalarOut:
            return result[0]
        else:
            return result
    return scalar_wrapper

def zsymDecorator(odd):
    """Decorator to deal with zsym=True input; set odd=True if the function is an odd function of z (like zforce)"""
    def wrapper(func):
        @wraps(func)
        def zsym_wrapper(*args,**kwargs):
            if args[0]._zsym:
                out= func(args[0],args[1],numpy.fabs(args[2]),**kwargs)
            else:
                out= func(*args,**kwargs)
            if odd and args[0]._zsym:
                return sign(args[2])*out
            else:
                return out
        return zsym_wrapper
    return wrapper

def scalarDecorator(func):
    """Decorator to return scalar output for 1D functions (vcirc,etc.)"""
    @wraps(func)
    def scalar_wrapper(*args,**kwargs):
        if numpy.array(args[1]).shape == ():
            scalarOut= True
            args= (args[0],numpy.array([args[1]]))
        else:
            scalarOut= False
        result= func(*args,**kwargs)
        if scalarOut:
            return result[0]
        else:
            return result
    return scalar_wrapper

class interpRZPotential(Potential):
    """Class that interpolates a given potential on a grid for fast orbit integration"""
    def __init__(self,
                 RZPot=None,rgrid=(numpy.log(0.01),numpy.log(20.),101),
                 zgrid=(0.,1.,101),logR=True,
                 interpPot=False,interpRforce=False,interpzforce=False,
                 interpDens=False,
                 interpvcirc=False,
                 interpdvcircdr=False,
                 interpepifreq=False,interpverticalfreq=False,
                 ro=None,vo=None,
                 use_c=False,enable_c=False,zsym=True,
                 numcores=None):
        """
        NAME:

           __init__

        PURPOSE:

           Initialize an interpRZPotential instance

        INPUT:

           RZPot - RZPotential to be interpolated

           rgrid - R grid to be given to linspace as in rs= linspace(*rgrid)

           zgrid - z grid to be given to linspace as in zs= linspace(*zgrid)

           logR - if True, rgrid is in the log of R so logrs= linspace(*rgrid)

           interpPot, interpRforce, interpzforce, interpDens,interpvcirc, interpepifreq, interpverticalfreq, interpdvcircdr= if True, interpolate these functions

           use_c= use C to speed up the calculation of the grid

           enable_c= enable use of C for interpolations

           zsym= if True (default), the potential is assumed to be symmetric around z=0 (so you can use, e.g.,  zgrid=(0.,1.,101)).

           numcores= if set to an integer, use this many cores (only used for vcirc, dvcircdR, epifreq, and verticalfreq; NOT NECESSARILY FASTER, TIME TO MAKE SURE)

           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

        OUTPUT:

           instance

        HISTORY:

           2010-07-21 - Written - Bovy (NYU)

           2013-01-24 - Started with new implementation - Bovy (IAS)

        """
        if isinstance(RZPot,interpRZPotential):
            from galpy.potential import PotentialError
            raise PotentialError('Cannot setup interpRZPotential with another interpRZPotential')
        # Propagate ro and vo
        roSet= True
        voSet= True
        if ro is None:
            if isinstance(RZPot,list):
                ro= RZPot[0]._ro
                roSet= RZPot[0]._roSet
            else:
                ro= RZPot._ro
                roSet= RZPot._roSet
        if vo is None:
            if isinstance(RZPot,list):
                vo= RZPot[0]._vo
                voSet= RZPot[0]._voSet
            else:
                vo= RZPot._vo
                voSet= RZPot._voSet
        Potential.__init__(self,amp=1.,ro=ro,vo=vo)
        # Turn off physical if it hadn't been on
        if not roSet: self._roSet= False
        if not voSet: self._voSet= False
        self._origPot= RZPot
        self._rgrid= numpy.linspace(*rgrid)
        self._logR= logR
        if self._logR:
            self._rgrid= numpy.exp(self._rgrid)
            self._logrgrid= numpy.log(self._rgrid)
        self._zgrid= numpy.linspace(*zgrid)
        self._interpPot= interpPot
        self._interpRforce= interpRforce
        self._interpzforce= interpzforce
        self._interpDens= interpDens
        self._interpvcirc= interpvcirc
        self._interpdvcircdr= interpdvcircdr
        self._interpepifreq= interpepifreq
        self._interpverticalfreq= interpverticalfreq
        self._enable_c= enable_c*ext_loaded
        self.hasC= self._enable_c
        self._zsym= zsym
        if interpPot:
            if use_c*ext_loaded:
                self._potGrid, err= calc_potential_c(self._origPot,self._rgrid,self._zgrid)
            else:
                from galpy.potential import evaluatePotentials
                potGrid= numpy.zeros((len(self._rgrid),len(self._zgrid)))
                for ii in range(len(self._rgrid)):
                    for jj in range(len(self._zgrid)):
                        potGrid[ii,jj]= evaluatePotentials(self._origPot,self._rgrid[ii],self._zgrid[jj])
                self._potGrid= potGrid
            if self._logR:
                self._potInterp= interpolate.RectBivariateSpline(self._logrgrid,
                                                                 self._zgrid,
                                                                 self._potGrid,
                                                                 kx=3,ky=3,s=0.)
            else:
                self._potInterp= interpolate.RectBivariateSpline(self._rgrid,
                                                                 self._zgrid,
                                                                 self._potGrid,
                                                                 kx=3,ky=3,s=0.)
            if enable_c*ext_loaded:
                self._potGrid_splinecoeffs= calc_2dsplinecoeffs_c(self._potGrid)
        if interpRforce:
            if use_c*ext_loaded:
                self._rforceGrid, err= calc_potential_c(self._origPot,self._rgrid,self._zgrid,rforce=True)
            else:
                from galpy.potential import evaluateRforces
                rforceGrid= numpy.zeros((len(self._rgrid),len(self._zgrid)))
                for ii in range(len(self._rgrid)):
                    for jj in range(len(self._zgrid)):
                        rforceGrid[ii,jj]= evaluateRforces(self._origPot,self._rgrid[ii],self._zgrid[jj])
                self._rforceGrid= rforceGrid
            if self._logR:
                self._rforceInterp= interpolate.RectBivariateSpline(self._logrgrid,
                                                                    self._zgrid,
                                                                    self._rforceGrid,
                                                                    kx=3,ky=3,s=0.)
            else:
                self._rforceInterp= interpolate.RectBivariateSpline(self._rgrid,
                                                                    self._zgrid,
                                                                    self._rforceGrid,
                                                                    kx=3,ky=3,s=0.)
            if enable_c*ext_loaded:
                self._rforceGrid_splinecoeffs= calc_2dsplinecoeffs_c(self._rforceGrid)
        if interpzforce:
            if use_c*ext_loaded:
                self._zforceGrid, err= calc_potential_c(self._origPot,self._rgrid,self._zgrid,zforce=True)
            else:
                from galpy.potential import evaluatezforces
                zforceGrid= numpy.zeros((len(self._rgrid),len(self._zgrid)))
                for ii in range(len(self._rgrid)):
                    for jj in range(len(self._zgrid)):
                        zforceGrid[ii,jj]= evaluatezforces(self._origPot,self._rgrid[ii],self._zgrid[jj])
                self._zforceGrid= zforceGrid
            if self._logR:
                self._zforceInterp= interpolate.RectBivariateSpline(self._logrgrid,
                                                                    self._zgrid,
                                                                    self._zforceGrid,
                                                                    kx=3,ky=3,s=0.)
            else:
                self._zforceInterp= interpolate.RectBivariateSpline(self._rgrid,
                                                                    self._zgrid,
                                                                    self._zforceGrid,
                                                                    kx=3,ky=3,s=0.)
            if enable_c*ext_loaded:
                self._zforceGrid_splinecoeffs= calc_2dsplinecoeffs_c(self._zforceGrid)
        if interpDens:
            from galpy.potential import evaluateDensities
            densGrid= numpy.zeros((len(self._rgrid),len(self._zgrid)))
            for ii in range(len(self._rgrid)):
                for jj in range(len(self._zgrid)):
                    densGrid[ii,jj]= evaluateDensities(self._origPot,self._rgrid[ii],self._zgrid[jj])
            self._densGrid= densGrid
            if self._logR:
                self._densInterp= interpolate.RectBivariateSpline(self._logrgrid,
                                                                  self._zgrid,
                                                                  numpy.log(self._densGrid+10.**-10.),
                                                                  kx=3,ky=3,s=0.)
            else:
                self._densInterp= interpolate.RectBivariateSpline(self._rgrid,
                                                                  self._zgrid,
                                                                  numpy.log(self._densGrid+10.**-10.),
                                                                  kx=3,ky=3,s=0.)
        if interpvcirc:
            from galpy.potential import vcirc
            if not numcores is None:
                self._vcircGrid= multi.parallel_map((lambda x: vcirc(self._origPot,self._rgrid[x])),
                                                    list(range(len(self._rgrid))),numcores=numcores)
            else:
                self._vcircGrid= numpy.array([vcirc(self._origPot,r) for r in self._rgrid])
            if self._logR:
                self._vcircInterp= interpolate.InterpolatedUnivariateSpline(self._logrgrid,self._vcircGrid,k=3)
            else:
                self._vcircInterp= interpolate.InterpolatedUnivariateSpline(self._rgrid,self._vcircGrid,k=3)
        if interpdvcircdr:
            from galpy.potential import dvcircdR
            if not numcores is None:
                self._dvcircdrGrid= multi.parallel_map((lambda x: dvcircdR(self._origPot,self._rgrid[x])),
                                                       list(range(len(self._rgrid))),numcores=numcores)
            else:
                self._dvcircdrGrid= numpy.array([dvcircdR(self._origPot,r) for r in self._rgrid])
            if self._logR:
                self._dvcircdrInterp= interpolate.InterpolatedUnivariateSpline(self._logrgrid,self._dvcircdrGrid,k=3)
            else:
                self._dvcircdrInterp= interpolate.InterpolatedUnivariateSpline(self._rgrid,self._dvcircdrGrid,k=3)
        if interpepifreq:
            from galpy.potential import epifreq
            if not numcores is None:
                self._epifreqGrid= numpy.array(multi.parallel_map((lambda x: epifreq(self._origPot,self._rgrid[x])),
                                                      list(range(len(self._rgrid))),numcores=numcores))
            else:
                self._epifreqGrid= numpy.array([epifreq(self._origPot,r) for r in self._rgrid])
            indx= True^numpy.isnan(self._epifreqGrid)
            if numpy.sum(indx) < 4:
                if self._logR:
                    self._epifreqInterp= interpolate.InterpolatedUnivariateSpline(self._logrgrid[indx],self._epifreqGrid[indx],k=1)
                else:
                    self._epifreqInterp= interpolate.InterpolatedUnivariateSpline(self._rgrid[indx],self._epifreqGrid[indx],k=1)
            else:
                if self._logR:
                    self._epifreqInterp= interpolate.InterpolatedUnivariateSpline(self._logrgrid[indx],self._epifreqGrid[indx],k=3)
                else:
                    self._epifreqInterp= interpolate.InterpolatedUnivariateSpline(self._rgrid[indx],self._epifreqGrid[indx],k=3)
        if interpverticalfreq:
            from galpy.potential import verticalfreq
            if not numcores is None:
                self._verticalfreqGrid= multi.parallel_map((lambda x: verticalfreq(self._origPot,self._rgrid[x])),
                                                       list(range(len(self._rgrid))),numcores=numcores)
            else:
                self._verticalfreqGrid= numpy.array([verticalfreq(self._origPot,r) for r in self._rgrid])
            if self._logR:
                self._verticalfreqInterp= interpolate.InterpolatedUnivariateSpline(self._logrgrid,self._verticalfreqGrid,k=3)
            else:
                self._verticalfreqInterp= interpolate.InterpolatedUnivariateSpline(self._rgrid,self._verticalfreqGrid,k=3)
        return None
                                                 
    @scalarVectorDecorator
    @zsymDecorator(False)
    def _evaluate(self,R,z,phi=0.,t=0.):
        from galpy.potential import evaluatePotentials
        if self._interpPot:
            out= numpy.empty_like(R)
            indx= (R >= self._rgrid[0])*(R <= self._rgrid[-1])\
                *(z <= self._zgrid[-1])*(z >= self._zgrid[0])
            if numpy.sum(indx) > 0:
                if self._enable_c:
                    out[indx]= eval_potential_c(self,R[indx],z[indx])[0]/self._amp
                else:
                    if self._logR:
                        out[indx]= self._potInterp.ev(numpy.log(R[indx]),z[indx])
                    else:
                        out[indx]= self._potInterp.ev(R[indx],z[indx])
            if numpy.sum(True^indx) > 0:
                out[True^indx]= evaluatePotentials(self._origPot,
                                                   R[True^indx],
                                                   z[True^indx])
            return out
        else:
            return evaluatePotentials(self._origPot,R,z)

    @scalarVectorDecorator
    @zsymDecorator(False)
    def _Rforce(self,R,z,phi=0.,t=0.):
        from galpy.potential import evaluateRforces
        if self._interpRforce:
            out= numpy.empty_like(R)
            indx= (R >= self._rgrid[0])*(R <= self._rgrid[-1])\
                *(z <= self._zgrid[-1])*(z >= self._zgrid[0])
            if numpy.sum(indx) > 0:
                if self._enable_c:
                    out[indx]= eval_force_c(self,R[indx],z[indx])[0]/self._amp
                else:
                    if self._logR:
                        out[indx]= self._rforceInterp.ev(numpy.log(R[indx]),z[indx])
                    else:
                        out[indx]= self._rforceInterp.ev(R[indx],z[indx])
            if numpy.sum(True^indx) > 0:
                out[True^indx]= evaluateRforces(self._origPot,
                                                R[True^indx],
                                                z[True^indx])
            return out
        else:
            return evaluateRforces(self._origPot,R,z)

    @scalarVectorDecorator
    @zsymDecorator(True)
    def _zforce(self,R,z,phi=0.,t=0.):
        from galpy.potential import evaluatezforces
        if self._interpzforce:
            out= numpy.empty_like(R)
            indx= (R >= self._rgrid[0])*(R <= self._rgrid[-1])\
                *(z <= self._zgrid[-1])*(z >= self._zgrid[0])
            if numpy.sum(indx) > 0:
                if self._enable_c:
                    out[indx]= eval_force_c(self,R[indx],z[indx],
                                            zforce=True)[0]/self._amp
                else:
                    if self._logR:
                        out[indx]= self._zforceInterp.ev(numpy.log(R[indx]),
                                                         z[indx])
                    else:
                        out[indx]= self._zforceInterp.ev(R[indx],z[indx])
            if numpy.sum(True^indx) > 0:
                out[True^indx]= evaluatezforces(self._origPot,
                                                R[True^indx],
                                                z[True^indx])
            return out
        else:
            return evaluatezforces(self._origPot,R,z)
    
    def _Rzderiv(self,R,z,phi=0.,t=0.):
        from galpy.potential import evaluateRzderivs
        return evaluateRzderivs(self._origPot,R,z)
    
    @scalarVectorDecorator
    @zsymDecorator(False)
    def _dens(self,R,z,phi=0.,t=0.):
        from galpy.potential import evaluateDensities
        if self._interpDens:
            out= numpy.empty_like(R)
            indx= (R >= self._rgrid[0])*(R <= self._rgrid[-1])\
                *(z <= self._zgrid[-1])*(z >= self._zgrid[0])
            if numpy.sum(indx) > 0:
                if self._logR:
                    out[indx]= numpy.exp(self._densInterp.ev(numpy.log(R[indx]),z[indx]))-10.**-10.
                else:
                    out[indx]= numpy.exp(self._densInterp.ev(R[indx],z[indx]))-10.**-10.
            if numpy.sum(True^indx) > 0:
                out[True^indx]= evaluateDensities(self._origPot,
                                                  R[True^indx],
                                                  z[True^indx])
            return out
        else:
            return evaluateDensities(self._origPot,R,z)

    @physical_conversion('velocity',pop=True)
    @scalarDecorator
    def vcirc(self,R):
        from galpy.potential import vcirc
        if self._interpvcirc:
            indx= (R >= self._rgrid[0])*(R <= self._rgrid[-1])
            out= numpy.empty_like(R)
            if numpy.sum(indx) > 0:
                if self._logR:
                    out[indx]= self._vcircInterp(numpy.log(R[indx]))
                else:
                    out[indx]= self._vcircInterp(R[indx])
            if numpy.sum(True^indx) > 0:
                out[True^indx]= vcirc(self._origPot,R[True^indx])
            return out
        else:
            return vcirc(self._origPot,R)

    @physical_conversion('frequency',pop=True)
    @scalarDecorator
    def dvcircdR(self,R):
        from galpy.potential import dvcircdR
        if self._interpdvcircdr:
            indx= (R >= self._rgrid[0])*(R <= self._rgrid[-1])
            out= numpy.empty_like(R)
            if numpy.sum(indx) > 0:
                if self._logR:
                    out[indx]= self._dvcircdrInterp(numpy.log(R[indx]))
                else:
                    out[indx]= self._dvcircdrInterp(R[indx])
            if numpy.sum(True^indx) > 0:
                out[True^indx]= dvcircdR(self._origPot,R[True^indx])
            return out
        else:
            return dvcircdR(self._origPot,R)

    @physical_conversion('frequency',pop=True)
    @scalarDecorator
    def epifreq(self,R):
        from galpy.potential import epifreq
        if self._interpepifreq:
            indx= (R >= self._rgrid[0])*(R <= self._rgrid[-1])
            out= numpy.empty_like(R)
            if numpy.sum(indx) > 0:
                if self._logR:
                    out[indx]= self._epifreqInterp(numpy.log(R[indx]))
                else:
                    out[indx]= self._epifreqInterp(R[indx])
            if numpy.sum(True^indx) > 0:
                out[True^indx]= epifreq(self._origPot,R[True^indx])
            return out
        else:
            return epifreq(self._origPot,R)

    @physical_conversion('frequency',pop=True)
    @scalarDecorator
    def verticalfreq(self,R):
        from galpy.potential import verticalfreq
        if self._interpverticalfreq:
            indx= (R >= self._rgrid[0])*(R <= self._rgrid[-1])
            out= numpy.empty_like(R)
            if numpy.sum(indx) > 0:
                if self._logR:
                    out[indx]= self._verticalfreqInterp(numpy.log(R[indx]))
                else:
                    out[indx]= self._verticalfreqInterp(R[indx])
            if numpy.sum(True^indx) > 0:
                out[True^indx]= verticalfreq(self._origPot,R[True^indx])
            return out
        else:
            return verticalfreq(self._origPot,R)
    
def calc_potential_c(pot,R,z,rforce=False,zforce=False):
    """
    NAME:
       calc_potential_c
    PURPOSE:
       Use C to calculate the potential on a grid
    INPUT:
       pot - Potential or list of such instances
       R - grid in R
       z - grid in z
       rforce=, zforce= if either of these is True, calculate the radial or vertical force instead
    OUTPUT:
       potential on the grid (2D array)
    HISTORY:
       2013-01-24 - Written - Bovy (IAS)
       2013-01-29 - Added forces - Bovy (IAS)
    """
    from galpy.orbit.integrateFullOrbit import _parse_pot #here bc otherwise there is an infinite loop
    #Parse the potential
    npot, pot_type, pot_args= _parse_pot(pot)

    #Set up result arrays
    out= numpy.empty((len(R),len(z)))
    err= ctypes.c_int(0)

    #Set up the C code
    ndarrayFlags= ('C_CONTIGUOUS','WRITEABLE')
    if rforce:
        interppotential_calc_potentialFunc= _lib.calc_rforce
    elif zforce:
        interppotential_calc_potentialFunc= _lib.calc_zforce
    else:
        interppotential_calc_potentialFunc= _lib.calc_potential
    interppotential_calc_potentialFunc.argtypes= [ctypes.c_int,
                                                  ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                                                  ctypes.c_int,
                                                  ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                                                  ctypes.c_int,
                                                  ndpointer(dtype=numpy.int32,flags=ndarrayFlags),
                                                  ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                                                  ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                                                  ctypes.POINTER(ctypes.c_int)]

    #Array requirements, first store old order
    f_cont= [R.flags['F_CONTIGUOUS'],
             z.flags['F_CONTIGUOUS']]
    R= numpy.require(R,dtype=numpy.float64,requirements=['C','W'])
    z= numpy.require(z,dtype=numpy.float64,requirements=['C','W'])
    out= numpy.require(out,dtype=numpy.float64,requirements=['C','W'])

    #Run the C code
    interppotential_calc_potentialFunc(len(R),
                                       R,
                                       len(z),
                                       z,
                                       ctypes.c_int(npot),
                                       pot_type,
                                       pot_args,
                                       out,
                                       ctypes.byref(err))
    
    #Reset input arrays
    if f_cont[0]: R= numpy.asfortranarray(R)
    if f_cont[1]: z= numpy.asfortranarray(z)

    return (out,err.value)

def calc_2dsplinecoeffs_c(array2d):
    """
    NAME:
       calc_2dsplinecoeffs_c
    PURPOSE:
       Use C to calculate spline coefficients for a 2D array
    INPUT:
       array2d
    OUTPUT:
       new array with spline coeffs
    HISTORY:
       2013-01-24 - Written - Bovy (IAS)
    """
    #Set up result arrays
    out= copy.copy(array2d)
    out= numpy.require(out,dtype=numpy.float64,requirements=['C','W'])

    #Set up the C code
    ndarrayFlags= ('C_CONTIGUOUS','WRITEABLE')
    interppotential_calc_2dsplinecoeffs= _lib.samples_to_coefficients
    interppotential_calc_2dsplinecoeffs.argtypes= [ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                                                   ctypes.c_int,
                                                   ctypes.c_int]

    #Run the C code
    interppotential_calc_2dsplinecoeffs(out,out.shape[1],out.shape[0])

    return out

def eval_potential_c(pot,R,z):
    """
    NAME:
       eval_potential_c
    PURPOSE:
       Use C to evaluate the interpolated potential
    INPUT:
       pot - Potential or list of such instances
       R - array
       z - array
    OUTPUT:
       potential evaluated R and z
    HISTORY:
       2013-01-24 - Written - Bovy (IAS)
    """
    from galpy.orbit.integrateFullOrbit import _parse_pot #here bc otherwise there is an infinite loop
    #Parse the potential
    npot, pot_type, pot_args= _parse_pot(pot,potforactions=True)

    #Set up result arrays
    out= numpy.empty((len(R)))
    err= ctypes.c_int(0)

    #Set up the C code
    ndarrayFlags= ('C_CONTIGUOUS','WRITEABLE')
    interppotential_calc_potentialFunc= _lib.eval_potential
    interppotential_calc_potentialFunc.argtypes= [ctypes.c_int,
                                                  ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                                                  ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                                                  ctypes.c_int,
                                                  ndpointer(dtype=numpy.int32,flags=ndarrayFlags),
                                                  ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                                                  ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                                                  ctypes.POINTER(ctypes.c_int)]

    #Array requirements, first store old order
    f_cont= [R.flags['F_CONTIGUOUS'],
             z.flags['F_CONTIGUOUS']]
    R= numpy.require(R,dtype=numpy.float64,requirements=['C','W'])
    z= numpy.require(z,dtype=numpy.float64,requirements=['C','W'])
    out= numpy.require(out,dtype=numpy.float64,requirements=['C','W'])

    #Run the C code
    interppotential_calc_potentialFunc(len(R),
                                       R,
                                       z,
                                       ctypes.c_int(npot),
                                       pot_type,
                                       pot_args,
                                       out,
                                       ctypes.byref(err))

    #Reset input arrays
    if f_cont[0]: R= numpy.asfortranarray(R)
    if f_cont[1]: z= numpy.asfortranarray(z)

    return (out,err.value)

def eval_force_c(pot,R,z,zforce=False):
    """
    NAME:
       eval_force_c
    PURPOSE:
       Use C to evaluate the interpolated potential's forces
    INPUT:
       pot - Potential or list of such instances
       R - array
       z - array
       zforce= if True, return the vertical force, otherwise return the radial force
    OUTPUT:
       force evaluated R and z
    HISTORY:
       2013-01-29 - Written - Bovy (IAS)
    """
    from galpy.orbit.integrateFullOrbit import _parse_pot #here bc otherwise there is an infinite loop
    #Parse the potential
    npot, pot_type, pot_args= _parse_pot(pot)

    #Set up result arrays
    out= numpy.empty((len(R)))
    err= ctypes.c_int(0)

    #Set up the C code
    ndarrayFlags= ('C_CONTIGUOUS','WRITEABLE')
    if zforce:
        interppotential_calc_forceFunc= _lib.eval_zforce
    else:
        interppotential_calc_forceFunc= _lib.eval_rforce
    interppotential_calc_forceFunc.argtypes= [ctypes.c_int,
                                                  ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                                                  ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                                                  ctypes.c_int,
                                                  ndpointer(dtype=numpy.int32,flags=ndarrayFlags),
                                                  ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                                                  ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                                                  ctypes.POINTER(ctypes.c_int)]

    #Array requirements, first store old order
    f_cont= [R.flags['F_CONTIGUOUS'],
             z.flags['F_CONTIGUOUS']]
    R= numpy.require(R,dtype=numpy.float64,requirements=['C','W'])
    z= numpy.require(z,dtype=numpy.float64,requirements=['C','W'])
    out= numpy.require(out,dtype=numpy.float64,requirements=['C','W'])

    #Run the C code
    interppotential_calc_forceFunc(len(R),
                                   R,
                                   z,
                                   ctypes.c_int(npot),
                                   pot_type,
                                   pot_args,
                                   out,
                                   ctypes.byref(err))
    
    #Reset input arrays
    if f_cont[0]: R= numpy.asfortranarray(R)
    if f_cont[1]: z= numpy.asfortranarray(z)

    return (out,err.value)

def sign(x):
    out= numpy.ones_like(x)
    out[(x < 0.)]= -1.
    return out
