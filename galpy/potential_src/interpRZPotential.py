import os
import copy
import ctypes
import ctypes.util
import warnings
import numpy
from numpy.ctypeslib import ndpointer
from scipy import interpolate
from galpy.util import multi, galpyWarning
from Potential import Potential
_DEBUG= False
#Find and load the library
_lib = None
_libname = ctypes.util.find_library('galpy_interppotential_c')
if _libname:
    _lib = ctypes.CDLL(_libname)
if _lib is None:
    import sys
    for path in sys.path:
        try:
            _lib = ctypes.CDLL(os.path.join(path,'galpy_interppotential_c.so'))
        except OSError:
            _lib = None
        else:
            break
if _lib is None:
    #raise IOError('galpy interppotential_c module not found')
    warnings.warn("interppotential_c extension module not loaded",
                  galpyWarning)
    ext_loaded= False
else:
    ext_loaded= True

class interpRZPotential(Potential):
    """Class that interpolates a given potential on a grid for fast orbit integration"""
    def __init__(self,
                 RZPot=None,rgrid=(0.01,2.,101),zgrid=(0.,0.2,101),logR=False,
                 interpPot=False,interpRforce=False,interpzforce=False,
                 interpDens=False,
                 interpvcirc=False,
                 interpdvcircdr=False,
                 interpepifreq=False,interpverticalfreq=False,
                 use_c=False,enable_c=False,zsym=True,
                 numcores=None):
        """
        NAME:
           __init__
        PURPOSE:
           Initialize an interpRZPotential instance
        INPUT:
           RZPot - RZPotential to be interpolated
           rgrid - R grid to be given to linspace
           zgrid - z grid to be given to linspace
           logR - if True, rgrid is in the log of R
           interpPot, interpRfoce, interpzforce, interpDens,interpvcirc, interpeopifreq, interpverticalfreq, interpdvcircdr= if True, interpolate these functions
           use_c= use C to speed up the calculation
           enable_c= enable use of C for interpolations
           zsym= if True (default), the potential is assumed to be symmetric around z=0 (so you can use, e.g.,  zgrid=(0.,1.,101)).
           numcores= if set to an integer, use this many cores (only used for vcirc, dvcircdR, epifreq, and verticalfreq; NOT NECESSARILY FASTER, TIME TO MAKE SURE)
        OUTPUT:
           instance
        HISTORY:
           2010-07-21 - Written - Bovy (NYU)
           2013-01-24 - Started with new implementation - Bovy (IAS)
        """
        Potential.__init__(self,amp=1.)
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
        self._zsym= zsym
        if interpPot:
            if use_c*ext_loaded:
                self._potGrid, err= calc_potential_c(self._origPot,self._rgrid,self._zgrid)
            else:
                from galpy.potential import evaluatePotentials
                potGrid= numpy.zeros((len(self._rgrid),len(self._zgrid)))
                for ii in range(len(self._rgrid)):
                    for jj in range(len(self._zgrid)):
                        potGrid[ii,jj]= evaluatePotentials(self._rgrid[ii],self._zgrid[jj],self._origPot)
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
                        rforceGrid[ii,jj]= evaluateRforces(self._rgrid[ii],self._zgrid[jj],self._origPot)
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
                        zforceGrid[ii,jj]= evaluatezforces(self._rgrid[ii],self._zgrid[jj],self._origPot)
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
            if False:
                raise NotImplementedError("Using C to calculate an interpolation grid for the density is not supported currently")
                self._densGrid, err= calc_dens_c(self._origPot,self._rgrid,self._zgrid)
            else:
                from galpy.potential import evaluateDensities
                densGrid= numpy.zeros((len(self._rgrid),len(self._zgrid)))
                for ii in range(len(self._rgrid)):
                    for jj in range(len(self._zgrid)):
                        densGrid[ii,jj]= evaluateDensities(self._rgrid[ii],self._zgrid[jj],self._origPot)
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
            if False:
                self._densGrid_splinecoeffs= calc_2dsplinecoeffs_c(self._densGrid)
        if interpvcirc:
            from galpy.potential import vcirc
            if not numcores is None:
                self._vcircGrid= multi.parallel_map((lambda x: vcirc(self._origPot,self._rgrid[x])),
                                                    range(len(self._rgrid)),numcores=numcores)
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
                                                       range(len(self._rgrid)),numcores=numcores)
            else:
                self._dvcircdrGrid= numpy.array([dvcircdR(self._origPot,r) for r in self._rgrid])
            if self._logR:
                self._dvcircdrInterp= interpolate.InterpolatedUnivariateSpline(self._logrgrid,self._dvcircdrGrid,k=3)
            else:
                self._dvcircdrInterp= interpolate.InterpolatedUnivariateSpline(self._rgrid,self._dvcircdrGrid,k=3)
        if interpepifreq:
            from galpy.potential import epifreq
            if not numcores is None:
                self._epifreqGrid= multi.parallel_map((lambda x: epifreq(self._origPot,self._rgrid[x])),
                                                      range(len(self._rgrid)),numcores=numcores)
            else:
                self._epifreqGrid= numpy.array([epifreq(self._origPot,r) for r in self._rgrid])
            indx= True-numpy.isnan(self._epifreqGrid)
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
                                                       range(len(self._rgrid)),numcores=numcores)
            else:
                self._verticalfreqGrid= numpy.array([verticalfreq(self._origPot,r) for r in self._rgrid])
            if self._logR:
                self._verticalfreqInterp= interpolate.InterpolatedUnivariateSpline(self._logrgrid,self._verticalfreqGrid,k=3)
            else:
                self._verticalfreqInterp= interpolate.InterpolatedUnivariateSpline(self._rgrid,self._verticalfreqGrid,k=3)
        return None
                                                 
    def _evaluate(self,R,z,phi=0.,t=0.,dR=0,dphi=0):
        if self._interpPot and self._enable_c:
            if isinstance(R,float):
                R= numpy.array([R])
            if isinstance(z,float):
                z= numpy.array([z])
            if self._zsym:
                return eval_potential_c(self,R,numpy.fabs(z))[0]
            else:
                return eval_potential_c(self,R,z)[0]
        from galpy.potential import evaluatePotentials
        if self._interpPot:
            if isinstance(R,float):
                return self._evaluate(numpy.array([R]),numpy.array([z]))
            out= numpy.empty_like(R)
            indx= (R >= self._rgrid[0])*(R <= self._rgrid[-1])
            if numpy.sum(indx) > 0:
                if self._zsym:
                    if self._logR:
                        out[indx]= self._potInterp.ev(numpy.log(R[indx]),numpy.fabs(z[indx]))
                    else:
                        out[indx]= self._potInterp.ev(R[indx],numpy.fabs(z[indx]))
                else:
                    if self._logR:
                        out[indx]= self._potInterp.ev(numpy.log(R[indx]),z[indx])
                    else:
                        out[indx]= self._potInterp.ev(R[indx],z[indx])
            if numpy.sum(True-indx) > 0:
                out[True-indx]= evaluatePotentials(R[True-indx],
                                                   z[True-indx],
                                                   self._origPot)
            return out
        else:
            return evaluatePotentials(R,z,self._origPot)

    def _Rforce(self,R,z,phi=0.,t=0.):
        if self._interpRforce and self._enable_c:
            if isinstance(R,float):
                R= numpy.array([R])
            if isinstance(z,float):
                z= numpy.array([z])
            if self._zsym:
                return eval_force_c(self,R,numpy.fabs(z))[0]
            else:
                return eval_force_c(self,R,z)[0]
        from galpy.potential import evaluateRforces
        if self._interpRforce:
            if isinstance(R,float):
                return self._Rforce(numpy.array([R]),numpy.array([z]))
            out= numpy.empty_like(R)
            indx= (R >= self._rgrid[0])*(R <= self._rgrid[-1])
            if numpy.sum(indx) > 0:
                if self._zsym:
                    if self._logR:
                        out[indx]= self._rforceInterp.ev(numpy.log(R[indx]),numpy.fabs(z[indx]))
                    else:
                        out[indx]= self._rforceInterp.ev(R[indx],numpy.fabs(z[indx]))
                else:
                    if self._logR:
                        out[indx]= self._rforceInterp.ev(numpy.log(R[indx]),z[indx])
                    else:
                        out[indx]= self._rforceInterp.ev(R[indx],z[indx])
            if numpy.sum(True-indx) > 0:
                out[True-indx]= evaluateRforces(R[True-indx],
                                                z[True-indx],
                                                self._origPot)
            return out
        else:
            return evaluateRforces(R,z,self._origPot)

    def _zforce(self,R,z,phi=0.,t=0.):
        if self._interpzforce and self._enable_c:
            if isinstance(R,float):
                R= numpy.array([R])
            if isinstance(z,float):
                z= numpy.array([z])
            if self._zsym:
                return sign(z) * eval_force_c(self,R,numpy.fabs(z),zforce=True)[0]
            else:
                return eval_force_c(self,R,z,zforce=True)[0]
        from galpy.potential import evaluatezforces
        if self._interpzforce:
            if isinstance(R,float):
                return self._zforce(numpy.array([R]),numpy.array([z]))
            out= numpy.empty_like(R)
            indx= (R >= self._rgrid[0])*(R <= self._rgrid[-1])
            if numpy.sum(indx) > 0:
                if self._zsym:
                    if self._logR:
                        out[indx]= sign(z) * self._zforceInterp.ev(numpy.log(R[indx]),numpy.fabs(z[indx]))
                    else:
                        out[indx]= sign(z) * self._zforceInterp.ev(R[indx],numpy.fabs(z[indx]))
                else:
                    if self._logR:
                        out[indx]= self._zforceInterp.ev(numpy.log(R[indx]),z[indx])
                    else:
                        out[indx]= self._zforceInterp.ev(R[indx],z[indx])
            if numpy.sum(True-indx) > 0:
                out[True-indx]= evaluatezforces(R[True-indx],
                                                z[True-indx],
                                                self._origPot)
            return out
        else:
            return evaluatezforces(R,z,self._origPot)
    
    def _Rzderiv(self,R,z,phi=0.,t=0.):
        from galpy.potential import evaluateRzderivs
        return evaluateRzderivs(R,z,self._origPot)
    
    def _dens(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _dens
        PURPOSE:
           evaluate the density for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the density
        HISTORY:
           2013-01-29 - Written - Bovy (IAS)
        """
        if self._interpDens and False:#self._enable_c:
            if isinstance(R,float):
                R= numpy.array([R])
            if isinstance(z,float):
                z= numpy.array([z])
            if self._zsym:
                return eval_dens_c(self,R,numpy.fabs(z))[0]
            else:
                return eval_dens_c(self,R,z)[0]
        from galpy.potential import evaluateDensities
        if self._interpDens:
            if isinstance(R,float):
                return self._dens(numpy.array([R]),numpy.array([z]))
            out= numpy.empty_like(R)
            indx= (R >= self._rgrid[0])*(R <= self._rgrid[-1])
            if numpy.sum(indx) > 0:
                if self._zsym:
                    if self._logR:
                        out[indx]= numpy.exp(self._densInterp.ev(numpy.log(R[indx]),numpy.fabs(z[indx])))-10.**-10.
                    else:
                        out[indx]= numpy.exp(self._densInterp.ev(R[indx],numpy.fabs(z[indx])))-10.**-10.
                else:
                    if self._logR:
                        out[indx]= numpy.exp(self._densInterp.ev(numpy.log(R[indx]),z[indx]))-10.**-10.
                    else:
                        out[indx]= numpy.exp(self._densInterp.ev(R[indx],z[indx]))-10.**-10.
            if numpy.sum(True-indx) > 0:
                out[True-indx]= evaluateDensities(R[True-indx],
                                                  z[True-indx],
                                                  self._origPot)
            return out
        else:
            return evaluateDensities(R,z,self._origPot)

    def vcirc(self,R):
        if self._interpvcirc:
            if self._logR:
                return self._vcircInterp(numpy.log(R))
            else:
                return self._vcircInterp(R)
        else:
            from galpy.potential import vcirc
            return vcirc(self._origPot,R)

    def dvcircdR(self,R):
        if self._interpdvcircdr:
            if self._logR:
                return self._dvcircdrInterp(numpy.log(R))
            else:
                return self._dvcircdrInterp(R)
        else:
            from galpy.potential import dvcircdR
            return dvcircdR(self._origPot,R)

    def epifreq(self,R):
        if self._interpepifreq:
            if self._logR:
                return self._epifreqInterp(numpy.log(R))
            else:
                return self._epifreqInterp(R)
        else:
            from galpy.potential import epifreq
            return epifreq(self._origPot,R)

    def verticalfreq(self,R):
        if self._interpverticalfreq:
            if self._logR:
                return self._verticalfreqInterp(numpy.log(R))
            else:
                return self._verticalfreqInterp(R)
        else:
            from galpy.potential import verticalfreq
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
    from galpy.orbit_src.integrateFullOrbit import _parse_pot #here bc otherwise there is an infinite loop
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
    from galpy.orbit_src.integrateFullOrbit import _parse_pot #here bc otherwise there is an infinite loop
    #Parse the potential
    npot, pot_type, pot_args= _parse_pot(pot,potforactions=True)

    #check input
    if isinstance(z,float):
        z= numpy.ones(len(R))*z
    if isinstance(R,float):
        R= numpy.ones(len(z))*R

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
    from galpy.orbit_src.integrateFullOrbit import _parse_pot #here bc otherwise there is an infinite loop
    #Parse the potential
    npot, pot_type, pot_args= _parse_pot(pot)

    #check input
    if isinstance(z,float):
        z= numpy.ones(len(R))*z
    if isinstance(R,float):
        R= numpy.ones(len(z))*R

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
