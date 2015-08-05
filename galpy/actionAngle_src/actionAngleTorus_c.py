import os
import sys
import sysconfig
import warnings
import ctypes
import ctypes.util
import numpy
from numpy.ctypeslib import ndpointer
from galpy.util import galpyWarning
from galpy.orbit_src.integrateFullOrbit import _parse_pot
#Find and load the library
_lib= None
outerr= None
PY3= sys.version > '3'
if PY3: #pragma: no cover
    _ext_suffix= sysconfig.get_config_var('EXT_SUFFIX')
else:
    _ext_suffix= '.so'
for path in sys.path:
    try:
        _lib = ctypes.CDLL(os.path.join(path,'galpy_actionAngleTorus_c%s' % _ext_suffix))
    except OSError as e:
        if os.path.exists(os.path.join(path,'galpy_actionAngleTorus_c%s' % _ext_suffix)): #pragma: no cover
            outerr= e
        _lib = None
    else:
        break
if _lib is None: #pragma: no cover
    if not outerr is None:
        warnings.warn("actionAngleTorus_c extension module not loaded, because of error '%s' " % outerr,
                      galpyWarning)
    else:
        warnings.warn("actionAngleTorus_c extension module not loaded, because galpy_actionAngle_c%s image was not found" % _ext_suffix,
                      galpyWarning)
    _ext_loaded= False
else:
    _ext_loaded= True

def actionAngleTorus_xv_c(pot,jr,jphi,jz,
                          angler,anglephi,anglez):
    """
    NAME:
    PURPOSE:
    INPUT:
    OUTPUT:
    HISTORY:
    """
    #Parse the potential
    #npot, pot_type, pot_args= _parse_pot(pot,potforactions=True)

    #Set up result arrays
    R= numpy.empty(len(angler))
    vR= numpy.empty(len(angler))
    vT= numpy.empty(len(angler))
    z= numpy.empty(len(angler))
    vz= numpy.empty(len(angler))
    phi= numpy.empty(len(angler))

    #Set up the C code
    ndarrayFlags= ('C_CONTIGUOUS','WRITEABLE')
    actionAngleTorus_xvFunc= _lib.actionAngleTorus_xv
    actionAngleTorus_xvFunc.argtypes= [ctypes.c_double,
                                       ctypes.c_double,
                                       ctypes.c_double,
                                       ctypes.c_int,
                                       ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                                       ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                                       ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                                       ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                                       ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                                       ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                                       ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                                       ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                                       ndpointer(dtype=numpy.float64,flags=ndarrayFlags)]

    #Array requirements, first store old order
    f_cont= [angler.flags['F_CONTIGUOUS'],
             anglephi.flags['F_CONTIGUOUS'],
             anglez.flags['F_CONTIGUOUS']]
    angler= numpy.require(angler,dtype=numpy.float64,requirements=['C','W'])
    anglephi= numpy.require(anglephi,dtype=numpy.float64,requirements=['C','W'])
    anglez= numpy.require(anglez,dtype=numpy.float64,requirements=['C','W'])
    R= numpy.require(R,dtype=numpy.float64,requirements=['C','W'])
    vR= numpy.require(vR,dtype=numpy.float64,requirements=['C','W'])
    vT= numpy.require(vT,dtype=numpy.float64,requirements=['C','W'])
    z= numpy.require(z,dtype=numpy.float64,requirements=['C','W'])
    vz= numpy.require(vz,dtype=numpy.float64,requirements=['C','W'])
    phi= numpy.require(phi,dtype=numpy.float64,requirements=['C','W'])
    
    #Run the C code
    actionAngleTorus_xvFunc(ctypes.c_double(jr),
                            ctypes.c_double(jphi),
                            ctypes.c_double(jz),
                            ctypes.c_int(len(angler)),
                            angler,
                            anglephi,
                            anglez,
                            R,vR,vT,z,vz,phi)

    #Reset input arrays
    if f_cont[0]: angler= numpy.asfortranarray(angler)
    if f_cont[0]: anglephi= numpy.asfortranarray(anglephi)
    if f_cont[0]: anglez= numpy.asfortranarray(anglez)

    return (R,vR,vT,z,vz,phi)



