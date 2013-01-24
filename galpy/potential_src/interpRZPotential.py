import os
import ctypes
import ctypes.util
import numpy
from numpy.ctypeslib import ndpointer
from scipy import interpolate
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
    raise IOError('galpy interppotential_c module not found')

class interpRZPotential(Potential):
    """Class that interpolates a given potential on a grid for fast orbit integration"""
    def __init__(self,
                 RZPot=None,rgrid=(0.01,2.,101),zgrid=(0.,0.2,101),logR=False,
                 interpPot=False,interpRforce=False,interpzforce=False,
                 interpz2deriv=False,interpR2deriv=False,
                 use_c=False,enable_c=False):
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
        OUTPUT:
           instance
        HISTORY:
           2010-07-21 - Written - Bovy (NYU)
           2013-01-24 - Started with new implementation - Bovy (IAS)
        """
        Potential.__init__(self,amp=1.)
        self._origPot= RZPot
        self._rgrid= numpy.linspace(*rgrid)
        if logR:
            self._rgrid= numpy.exp(self._rgrid)
        self._zgrid= numpy.linspace(*zgrid)
        if interpPot:
            if use_c:
                self._potGrid, err= calc_potential_c(self._origPot,self._rgrid,self._zgrid)
            else:
                from galpy.potential import evaluatePotentials
                potGrid= numpy.zeros((len(self._rgrid),len(self._zgrid)))
                for ii in range(len(self._rgrid)):
                    for jj in range(len(self._zgrid)):
                        potGrid[ii,jj]= evaluatePotentials(self._rgrid[ii],self._zgrid[jj],self._origPot)
                self._potGrid= potGrid
        return None
        Rforce= numpy.zeros(len(self._rgrid)*len(self._zgrid))
        zforce= numpy.zeros(len(self._rgrid)*len(self._zgrid))
        if _DEBUG:
            print "Computing forces on grid ..."
        for ii in range(len(self._rgrid)):
            for jj in range(len(self._zgrid)):
                Rforce[ii*len(self._zgrid)+jj]= RZPot.Rforce(self._rgrid[ii],
                                                             self._zgrid[jj])
                zforce[ii*len(self._zgrid)+jj]= RZPot.zforce(self._rgrid[ii],
                                                             self._zgrid[jj])
                R[ii*len(self._zgrid)+jj]= self._rgrid[ii]
                z[ii*len(self._zgrid)+jj]= self._zgrid[jj]
        if _DEBUG:
            print "Interpolating ..."
        self._interpRforce= interpolate.interp2d(R,z,Rforce,bounds_error=True)
        self._interpzforce= interpolate.interp2d(R,z,zforce,bounds_error=True)
                                                 
    def _Rforce(self,R,z,phi=0.,t=0.):
        if R < self._rgrid[0] or R > self._rgrid[-1] \
                or z < self._zgrid[0] or z > self._zgrid[-1]:
            print "Current position out of range of interpolation, consider interpolating on a larger range"
            return self._origPot.Rforce(R,z)
        else:
            return self._interpRforce(R,z)

    def _zforce(self,R,z,phi=0.,t=0.):
        if R < self._rgrid[0] or R > self._rgrid[-1] \
                or z < self._zgrid[0] or z > self._zgrid[-1]:
            print "Current position out of range of interpolation, consider interpolating on a larger range"
            return self._origPot.zforce(R,z)
        else:
            return self._interpzforce(R,z)


def calc_potential_c(pot,R,z):
    """
    NAME:
       calc_potential_c
    PURPOSE:
       Use C to calculate the potential on a grid
    INPUT:
       pot - Potential or list of such instances
       R - grid in R
       z - grid in z
    OUTPUT:
       potential on the grid (2D array)
    HISTORY:
       2013-01-24 - Written - Bovy (IAS)
    """
    from galpy.orbit_src.integrateFullOrbit import _parse_pot #here bc otherwise there is an infinite loop
    #Parse the potential
    npot, pot_type, pot_args= _parse_pot(pot)

    #Set up result arrays
    out= numpy.empty((len(R),len(z)))
    err= ctypes.c_int(0)

    #Set up the C code
    ndarrayFlags= ('C_CONTIGUOUS','WRITEABLE')
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

