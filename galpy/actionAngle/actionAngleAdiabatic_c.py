import os
import sys
import distutils.sysconfig as sysconfig
import warnings
import ctypes
import ctypes.util
import numpy
from numpy.ctypeslib import ndpointer
from galpy.util import galpyWarning
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
        _lib = ctypes.CDLL(os.path.join(path,'galpy_actionAngle_c%s' % _ext_suffix))
    except OSError as e:
        if os.path.exists(os.path.join(path,'galpy_actionAngle_c%s' % _ext_suffix)): #pragma: no cover
            outerr= e
        _lib = None
    else:
        break
if _lib is None: #pragma: no cover
    if not outerr is None:
        warnings.warn("actionAngleAdiabatic_c extension module not loaded, because of error '%s' " % outerr,
                      galpyWarning)
    else:
        warnings.warn("actionAngleAdiabatic_c extension module not loaded, because galpy_actionAngle_c%s image was not found" % _ext_suffix,
                      galpyWarning)
    _ext_loaded= False
else:
    _ext_loaded= True

def actionAngleAdiabatic_c(pot,gamma,R,vR,vT,z,vz):
    """
    NAME:
       actionAngleAdiabatic_c
    PURPOSE:
       Use C to calculate actions using the adiabatic approximation
    INPUT:
       pot - Potential or list of such instances
       gamma - as in Lz -> Lz+\gamma * J_z
       R, vR, vT, z, vz - coordinates (arrays)
    OUTPUT:
       (jr,jz,err)
       jr,jz : array, shape (len(R))
       err - non-zero if error occured
    HISTORY:
       2012-12-10 - Written - Bovy (IAS)
    """
    #Parse the potential
    from galpy.orbit.integrateFullOrbit import _parse_pot
    npot, pot_type, pot_args= _parse_pot(pot,potforactions=True)

    #Set up result arrays
    jr= numpy.empty(len(R))
    jz= numpy.empty(len(R))
    err= ctypes.c_int(0)

    #Set up the C code
    ndarrayFlags= ('C_CONTIGUOUS','WRITEABLE')
    actionAngleAdiabatic_actionsFunc= _lib.actionAngleAdiabatic_actions
    actionAngleAdiabatic_actionsFunc.argtypes= [ctypes.c_int,
                                                ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                                                ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                                                ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                                                ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                                                ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                                                ctypes.c_int,
                                                ndpointer(dtype=numpy.int32,flags=ndarrayFlags),
                                                ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                                                ctypes.c_double,
                                                ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                                                ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                                                ctypes.POINTER(ctypes.c_int)]

    #Array requirements, first store old order
    f_cont= [R.flags['F_CONTIGUOUS'],
             vR.flags['F_CONTIGUOUS'],
             vT.flags['F_CONTIGUOUS'],
             z.flags['F_CONTIGUOUS'],
             vz.flags['F_CONTIGUOUS']]
    R= numpy.require(R,dtype=numpy.float64,requirements=['C','W'])
    vR= numpy.require(vR,dtype=numpy.float64,requirements=['C','W'])
    vT= numpy.require(vT,dtype=numpy.float64,requirements=['C','W'])
    z= numpy.require(z,dtype=numpy.float64,requirements=['C','W'])
    vz= numpy.require(vz,dtype=numpy.float64,requirements=['C','W'])
    jr= numpy.require(jr,dtype=numpy.float64,requirements=['C','W'])
    jz= numpy.require(jz,dtype=numpy.float64,requirements=['C','W'])

    #Run the C code
    actionAngleAdiabatic_actionsFunc(len(R),
                                     R,
                                     vR,
                                     vT,
                                     z,
                                     vz,
                                     ctypes.c_int(npot),
                                     pot_type,
                                     pot_args,
                                     ctypes.c_double(gamma),
                                     jr,
                                     jz,
                                     ctypes.byref(err))

    #Reset input arrays
    if f_cont[0]: R= numpy.asfortranarray(R)
    if f_cont[1]: vR= numpy.asfortranarray(vR)
    if f_cont[2]: vT= numpy.asfortranarray(vT)
    if f_cont[3]: z= numpy.asfortranarray(z)
    if f_cont[4]: vz= numpy.asfortranarray(vz)

    return (jr,jz,err.value)

def actionAngleRperiRapZmaxAdiabatic_c(pot,gamma,R,vR,vT,z,vz):
    """
    NAME:
       actionAngleRperiRapZmaxAdiabatic_c
    PURPOSE:
       Use C to calculate planar (Rperi,Rap) and the maximum height Zmax using the adiabatic approximation (rap = sqrt(Rap^2+Zmax^2))
    INPUT:
       pot - Potential or list of such instances
       gamma - as in Lz -> Lz+\gamma * J_z
       R, vR, vT, z, vz - coordinates (arrays)
    OUTPUT:
       (Rperi,Rap,Zmax,err)
       Rperi,Rap,Zmax : array, shape (len(R))
       err - non-zero if error occured
    HISTORY:
       2017-12-21 - Written - Bovy (UofT)
    """
    #Parse the potential
    from galpy.orbit.integrateFullOrbit import _parse_pot
    npot, pot_type, pot_args= _parse_pot(pot,potforactions=True)

    #Set up result arrays
    rperi= numpy.empty(len(R))
    rap= numpy.empty(len(R))
    zmax= numpy.empty(len(R))
    err= ctypes.c_int(0)

    #Set up the C code
    ndarrayFlags= ('C_CONTIGUOUS','WRITEABLE')
    actionAngleAdiabatic_actionsFunc= _lib.actionAngleAdiabatic_RperiRapZmax
    actionAngleAdiabatic_actionsFunc.argtypes= [ctypes.c_int,
                                                ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                                                ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                                                ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                                                ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                                                ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                                                ctypes.c_int,
                                                ndpointer(dtype=numpy.int32,flags=ndarrayFlags),
                                                ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                                                ctypes.c_double,
                                                ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                                                ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                                                ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                                                ctypes.POINTER(ctypes.c_int)]

    #Array requirements, first store old order
    f_cont= [R.flags['F_CONTIGUOUS'],
             vR.flags['F_CONTIGUOUS'],
             vT.flags['F_CONTIGUOUS'],
             z.flags['F_CONTIGUOUS'],
             vz.flags['F_CONTIGUOUS']]
    R= numpy.require(R,dtype=numpy.float64,requirements=['C','W'])
    vR= numpy.require(vR,dtype=numpy.float64,requirements=['C','W'])
    vT= numpy.require(vT,dtype=numpy.float64,requirements=['C','W'])
    z= numpy.require(z,dtype=numpy.float64,requirements=['C','W'])
    vz= numpy.require(vz,dtype=numpy.float64,requirements=['C','W'])
    rperi= numpy.require(rperi,dtype=numpy.float64,requirements=['C','W'])
    rap= numpy.require(rap,dtype=numpy.float64,requirements=['C','W'])
    zmax= numpy.require(zmax,dtype=numpy.float64,requirements=['C','W'])

    #Run the C code
    actionAngleAdiabatic_actionsFunc(len(R),
                                     R,
                                     vR,
                                     vT,
                                     z,
                                     vz,
                                     ctypes.c_int(npot),
                                     pot_type,
                                     pot_args,
                                     ctypes.c_double(gamma),
                                     rperi,
                                     rap,
                                     zmax,
                                     ctypes.byref(err))

    #Reset input arrays
    if f_cont[0]: R= numpy.asfortranarray(R)
    if f_cont[1]: vR= numpy.asfortranarray(vR)
    if f_cont[2]: vT= numpy.asfortranarray(vT)
    if f_cont[3]: z= numpy.asfortranarray(z)
    if f_cont[4]: vz= numpy.asfortranarray(vz)

    return (rperi,rap,zmax,err.value)
