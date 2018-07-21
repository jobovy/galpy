import os
import sys
import distutils.sysconfig as sysconfig
import warnings
import ctypes
import ctypes.util
import numpy
from numpy.ctypeslib import ndpointer
from galpy.util import galpyWarning
from galpy.util import bovy_coords
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
        warnings.warn("actionAngleStaeckel_c extension module not loaded, because of error '%s' " % outerr,
                      galpyWarning)
    else:
        warnings.warn("actionAngleStaeckel_c extension module not loaded, because galpy_actionAngle_c%s image was not found" % _ext_suffix,
                      galpyWarning)
    _ext_loaded= False
else:
    _ext_loaded= True

def actionAngleStaeckel_c(pot,delta,R,vR,vT,z,vz,u0=None,order=10):
    """
    NAME:
       actionAngleStaeckel_c
    PURPOSE:
       Use C to calculate actions using the Staeckel approximation
    INPUT:
       pot - Potential or list of such instances
       delta - focal length of prolate spheroidal coordinates
       R, vR, vT, z, vz - coordinates (arrays)
       u0= (None) if set, u0 to use
       order= (10) order of Gauss-Legendre integration of the relevant integrals
    OUTPUT:
       (jr,jz,err)
       jr,jz : array, shape (len(R))
       err - non-zero if error occured
    HISTORY:
       2012-12-01 - Written - Bovy (IAS)
    """
    if u0 is None:
        u0, dummy= bovy_coords.Rz_to_uv(R,z,delta=numpy.atleast_1d(delta))
    #Parse the potential
    from galpy.orbit.integrateFullOrbit import _parse_pot
    npot, pot_type, pot_args= _parse_pot(pot,potforactions=True)

    #Parse delta
    delta= numpy.atleast_1d(delta)
    ndelta= len(delta)

    #Set up result arrays
    jr= numpy.empty(len(R))
    jz= numpy.empty(len(R))
    err= ctypes.c_int(0)

    #Set up the C code
    ndarrayFlags= ('C_CONTIGUOUS','WRITEABLE')
    actionAngleStaeckel_actionsFunc= _lib.actionAngleStaeckel_actions
    actionAngleStaeckel_actionsFunc.argtypes= [ctypes.c_int,
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ctypes.c_int,
                               ndpointer(dtype=numpy.int32,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ctypes.c_int,
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ctypes.c_int,
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ctypes.POINTER(ctypes.c_int)]

    #Array requirements, first store old order
    f_cont= [R.flags['F_CONTIGUOUS'],
             vR.flags['F_CONTIGUOUS'],
             vT.flags['F_CONTIGUOUS'],
             z.flags['F_CONTIGUOUS'],
             vz.flags['F_CONTIGUOUS'],
             u0.flags['F_CONTIGUOUS'],
             delta.flags['F_CONTIGUOUS']]
    R= numpy.require(R,dtype=numpy.float64,requirements=['C','W'])
    vR= numpy.require(vR,dtype=numpy.float64,requirements=['C','W'])
    vT= numpy.require(vT,dtype=numpy.float64,requirements=['C','W'])
    z= numpy.require(z,dtype=numpy.float64,requirements=['C','W'])
    vz= numpy.require(vz,dtype=numpy.float64,requirements=['C','W'])
    u0= numpy.require(u0,dtype=numpy.float64,requirements=['C','W'])
    delta= numpy.require(delta,dtype=numpy.float64,requirements=['C','W'])
    jr= numpy.require(jr,dtype=numpy.float64,requirements=['C','W'])
    jz= numpy.require(jz,dtype=numpy.float64,requirements=['C','W'])

    #Run the C code
    actionAngleStaeckel_actionsFunc(len(R),
                                    R,
                                    vR,
                                    vT,
                                    z,
                                    vz,
                                    u0,
                                    ctypes.c_int(npot),
                                    pot_type,
                                    pot_args,
                                    ctypes.c_int(ndelta),
                                    delta,
                                    ctypes.c_int(order),
                                    jr,
                                    jz,
                                    ctypes.byref(err))

    #Reset input arrays
    if f_cont[0]: R= numpy.asfortranarray(R)
    if f_cont[1]: vR= numpy.asfortranarray(vR)
    if f_cont[2]: vT= numpy.asfortranarray(vT)
    if f_cont[3]: z= numpy.asfortranarray(z)
    if f_cont[4]: vz= numpy.asfortranarray(vz)
    if f_cont[5]: u0= numpy.asfortranarray(u0)
    if f_cont[6]: delta= numpy.asfortranarray(delta)

    return (jr,jz,err.value)

def actionAngleStaeckel_calcu0(E,Lz,pot,delta):
    """
    NAME:
       actionAngleStaeckel_calcu0
    PURPOSE:
       Use C to calculate u0 in the Staeckel approximation
    INPUT:
       E, Lz - energy and angular momentum
       pot - Potential or list of such instances
       delta - focal length of prolate spheroidal coordinates
    OUTPUT:
       (u0,err)
       u0 : array, shape (len(E))
       err - non-zero if error occured
    HISTORY:
       2012-12-03 - Written - Bovy (IAS)
    """
    #Parse the potential
    from galpy.orbit.integrateFullOrbit import _parse_pot
    npot, pot_type, pot_args= _parse_pot(pot,potforactions=True)

    #Set up result arrays
    u0= numpy.empty(len(E))
    err= ctypes.c_int(0)

    #Parse delta
    delta= numpy.atleast_1d(delta)
    ndelta= len(delta)

    #Set up the C code
    ndarrayFlags= ('C_CONTIGUOUS','WRITEABLE')
    actionAngleStaeckel_actionsFunc= _lib.calcu0
    actionAngleStaeckel_actionsFunc.argtypes= [ctypes.c_int,
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ctypes.c_int,
                               ndpointer(dtype=numpy.int32,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ctypes.c_int,
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ctypes.POINTER(ctypes.c_int)]

    #Array requirements, first store old order
    f_cont= [E.flags['F_CONTIGUOUS'],
             Lz.flags['F_CONTIGUOUS'],
             delta.flags['F_CONTIGUOUS']]
    E= numpy.require(E,dtype=numpy.float64,requirements=['C','W'])
    Lz= numpy.require(Lz,dtype=numpy.float64,requirements=['C','W'])
    delta= numpy.require(delta,dtype=numpy.float64,requirements=['C','W'])
    u0= numpy.require(u0,dtype=numpy.float64,requirements=['C','W'])

    #Run the C code
    actionAngleStaeckel_actionsFunc(len(E),
                                    E,
                                    Lz,
                                    ctypes.c_int(npot),
                                    pot_type,
                                    pot_args,
                                    ctypes.c_int(ndelta),
                                    delta,
                                    u0,
                                    ctypes.byref(err))

    #Reset input arrays
    if f_cont[0]: E= numpy.asfortranarray(E)
    if f_cont[1]: Lz= numpy.asfortranarray(Lz)
    if f_cont[2]: delta= numpy.asfortranarray(delta)

    return (u0,err.value)

def actionAngleFreqStaeckel_c(pot,delta,R,vR,vT,z,vz,u0=None,order=10):
    """
    NAME:
       actionAngleFreqStaeckel_c
    PURPOSE:
       Use C to calculate actions and frequencies 
       using the Staeckel approximation
    INPUT:
       pot - Potential or list of such instances
       delta - focal length of prolate spheroidal coordinates
       R, vR, vT, z, vz - coordinates (arrays)
       u0= (None) if set, u0 to use
       order= (10) order of Gauss-Legendre integration of the relevant integrals
    OUTPUT:
       (jr,jz,Omegar,Omegaphi,Omegaz,err)
       jr,jz,Omegar,Omegaphi,Omegaz : array, shape (len(R))
       err - non-zero if error occured
    HISTORY:
       2013-08-23 - Written - Bovy (IAS)
    """
    if u0 is None:
        u0, dummy= bovy_coords.Rz_to_uv(R,z,delta=numpy.atleast_1d(delta))
    #Parse the potential
    from galpy.orbit.integrateFullOrbit import _parse_pot
    npot, pot_type, pot_args= _parse_pot(pot,potforactions=True)

    #Parse delta
    delta= numpy.atleast_1d(delta)
    ndelta= len(delta)

    #Set up result arrays
    jr= numpy.empty(len(R))
    jz= numpy.empty(len(R))
    Omegar= numpy.empty(len(R))
    Omegaphi= numpy.empty(len(R))
    Omegaz= numpy.empty(len(R))
    err= ctypes.c_int(0)

    #Set up the C code
    ndarrayFlags= ('C_CONTIGUOUS','WRITEABLE')
    actionAngleStaeckel_actionsFunc= _lib.actionAngleStaeckel_actionsFreqs
    actionAngleStaeckel_actionsFunc.argtypes= [ctypes.c_int,
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ctypes.c_int,
                               ndpointer(dtype=numpy.int32,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ctypes.c_int,
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ctypes.c_int,
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ctypes.POINTER(ctypes.c_int)]

    #Array requirements, first store old order
    f_cont= [R.flags['F_CONTIGUOUS'],
             vR.flags['F_CONTIGUOUS'],
             vT.flags['F_CONTIGUOUS'],
             z.flags['F_CONTIGUOUS'],
             vz.flags['F_CONTIGUOUS'],
             u0.flags['F_CONTIGUOUS'],
             delta.flags['F_CONTIGUOUS']]
    R= numpy.require(R,dtype=numpy.float64,requirements=['C','W'])
    vR= numpy.require(vR,dtype=numpy.float64,requirements=['C','W'])
    vT= numpy.require(vT,dtype=numpy.float64,requirements=['C','W'])
    z= numpy.require(z,dtype=numpy.float64,requirements=['C','W'])
    vz= numpy.require(vz,dtype=numpy.float64,requirements=['C','W'])
    u0= numpy.require(u0,dtype=numpy.float64,requirements=['C','W'])
    delta= numpy.require(delta,dtype=numpy.float64,requirements=['C','W'])
    jr= numpy.require(jr,dtype=numpy.float64,requirements=['C','W'])
    jz= numpy.require(jz,dtype=numpy.float64,requirements=['C','W'])
    Omegar= numpy.require(Omegar,dtype=numpy.float64,requirements=['C','W'])
    Omegaphi= numpy.require(Omegaphi,dtype=numpy.float64,
                            requirements=['C','W'])
    Omegaz= numpy.require(Omegaz,dtype=numpy.float64,requirements=['C','W'])

    #Run the C code
    actionAngleStaeckel_actionsFunc(len(R),
                                    R,
                                    vR,
                                    vT,
                                    z,
                                    vz,
                                    u0,
                                    ctypes.c_int(npot),
                                    pot_type,
                                    pot_args,
                                    ctypes.c_int(ndelta),
                                    delta,
                                    ctypes.c_int(order),
                                    jr,
                                    jz,
                                    Omegar,
                                    Omegaphi,
                                    Omegaz,
                                    ctypes.byref(err))

    #Reset input arrays
    if f_cont[0]: R= numpy.asfortranarray(R)
    if f_cont[1]: vR= numpy.asfortranarray(vR)
    if f_cont[2]: vT= numpy.asfortranarray(vT)
    if f_cont[3]: z= numpy.asfortranarray(z)
    if f_cont[4]: vz= numpy.asfortranarray(vz)
    if f_cont[5]: u0= numpy.asfortranarray(u0)
    if f_cont[6]: delta= numpy.asfortranarray(delta)

    return (jr,jz,Omegar,Omegaphi,Omegaz,err.value)

def actionAngleFreqAngleStaeckel_c(pot,delta,R,vR,vT,z,vz,phi,
                                   u0=None,order=10):
    """
    NAME:
       actionAngleFreqAngleStaeckel_c
    PURPOSE:
       Use C to calculate actions, frequencies, and angles
       using the Staeckel approximation
    INPUT:
       pot - Potential or list of such instances
       delta - focal length of prolate spheroidal coordinates
       R, vR, vT, z, vz, phi - coordinates (arrays)
       u0= (None) if set, u0 to use
       order= (10) order of Gauss-Legendre integration of the relevant integrals
    OUTPUT:
       (jr,jz,Omegar,Omegaphi,Omegaz,Angler,Anglephi,Anglez,err)
       jr,jz,Omegar,Omegaphi,Omegaz,Angler,Anglephi,Anglez : array, shape (len(R))
       err - non-zero if error occured
    HISTORY:
       2013-08-27 - Written - Bovy (IAS)
    """
    if u0 is None:
        u0, dummy= bovy_coords.Rz_to_uv(R,z,delta=numpy.atleast_1d(delta))
    #Parse the potential
    from galpy.orbit.integrateFullOrbit import _parse_pot
    npot, pot_type, pot_args= _parse_pot(pot,potforactions=True)

    #Parse delta
    delta= numpy.atleast_1d(delta)
    ndelta= len(delta)

    #Set up result arrays
    jr= numpy.empty(len(R))
    jz= numpy.empty(len(R))
    Omegar= numpy.empty(len(R))
    Omegaphi= numpy.empty(len(R))
    Omegaz= numpy.empty(len(R))
    Angler= numpy.empty(len(R))
    Anglephi= numpy.empty(len(R))
    Anglez= numpy.empty(len(R))
    err= ctypes.c_int(0)

    #Set up the C code
    ndarrayFlags= ('C_CONTIGUOUS','WRITEABLE')
    actionAngleStaeckel_actionsFunc= _lib.actionAngleStaeckel_actionsFreqsAngles
    actionAngleStaeckel_actionsFunc.argtypes= [ctypes.c_int,
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ctypes.c_int,
                               ndpointer(dtype=numpy.int32,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ctypes.c_int,
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ctypes.c_int,
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ctypes.POINTER(ctypes.c_int)]

    #Array requirements, first store old order
    f_cont= [R.flags['F_CONTIGUOUS'],
             vR.flags['F_CONTIGUOUS'],
             vT.flags['F_CONTIGUOUS'],
             z.flags['F_CONTIGUOUS'],
             vz.flags['F_CONTIGUOUS'],
             u0.flags['F_CONTIGUOUS'],
             delta.flags['F_CONTIGUOUS']]
    R= numpy.require(R,dtype=numpy.float64,requirements=['C','W'])
    vR= numpy.require(vR,dtype=numpy.float64,requirements=['C','W'])
    vT= numpy.require(vT,dtype=numpy.float64,requirements=['C','W'])
    z= numpy.require(z,dtype=numpy.float64,requirements=['C','W'])
    vz= numpy.require(vz,dtype=numpy.float64,requirements=['C','W'])
    u0= numpy.require(u0,dtype=numpy.float64,requirements=['C','W'])
    delta= numpy.require(delta,dtype=numpy.float64,requirements=['C','W'])
    jr= numpy.require(jr,dtype=numpy.float64,requirements=['C','W'])
    jz= numpy.require(jz,dtype=numpy.float64,requirements=['C','W'])
    Omegar= numpy.require(Omegar,dtype=numpy.float64,requirements=['C','W'])
    Omegaphi= numpy.require(Omegaphi,dtype=numpy.float64,
                            requirements=['C','W'])
    Omegaz= numpy.require(Omegaz,dtype=numpy.float64,requirements=['C','W'])
    Angler= numpy.require(Angler,dtype=numpy.float64,requirements=['C','W'])
    Anglephi= numpy.require(Anglephi,dtype=numpy.float64,
                            requirements=['C','W'])
    Anglez= numpy.require(Anglez,dtype=numpy.float64,requirements=['C','W'])

    #Run the C code
    actionAngleStaeckel_actionsFunc(len(R),
                                    R,
                                    vR,
                                    vT,
                                    z,
                                    vz,
                                    u0,
                                    ctypes.c_int(npot),
                                    pot_type,
                                    pot_args,
                                    ctypes.c_int(ndelta),
                                    delta,
                                    ctypes.c_int(order),
                                    jr,
                                    jz,
                                    Omegar,
                                    Omegaphi,
                                    Omegaz,
                                    Angler,
                                    Anglephi,
                                    Anglez,
                                    ctypes.byref(err))

    #Reset input arrays
    if f_cont[0]: R= numpy.asfortranarray(R)
    if f_cont[1]: vR= numpy.asfortranarray(vR)
    if f_cont[2]: vT= numpy.asfortranarray(vT)
    if f_cont[3]: z= numpy.asfortranarray(z)
    if f_cont[4]: vz= numpy.asfortranarray(vz)
    if f_cont[5]: u0= numpy.asfortranarray(u0)
    if f_cont[6]: delta= numpy.asfortranarray(delta)
    
    badAngle = Anglephi != 9999.99
    Anglephi[badAngle]= (Anglephi[badAngle] + phi[badAngle] % (2.*numpy.pi)) % (2.*numpy.pi)
    Anglephi[Anglephi < 0.]+= 2.*numpy.pi

    return (jr,jz,Omegar,Omegaphi,Omegaz,Angler,
            Anglephi,Anglez,err.value)

def actionAngleUminUmaxVminStaeckel_c(pot,delta,R,vR,vT,z,vz,u0=None):
    """
    NAME:
       actionAngleUminUmaxVminStaeckel_c
    PURPOSE:
       Use C to calculate umin, umax, and vmin using the Staeckel approximation
    INPUT:
       pot - Potential or list of such instances
       delta - focal length of prolate spheroidal coordinates
       R, vR, vT, z, vz - coordinates (arrays)
    OUTPUT:
       (umin,umax,vmin,err)
       umin,umax,vmin : array, shape (len(R))
       err - non-zero if error occured
    HISTORY:
       2017-12-12 - Written - Bovy (UofT)
    """
    if u0 is None:
        u0, dummy= bovy_coords.Rz_to_uv(R,z,delta=numpy.atleast_1d(delta))
    #Parse the potential
    from galpy.orbit.integrateFullOrbit import _parse_pot
    npot, pot_type, pot_args= _parse_pot(pot,potforactions=True)

    #Parse delta
    delta= numpy.atleast_1d(delta)
    ndelta= len(delta)

    #Set up result arrays
    umin= numpy.empty(len(R))
    umax= numpy.empty(len(R))
    vmin= numpy.empty(len(R))
    err= ctypes.c_int(0)

    #Set up the C code
    ndarrayFlags= ('C_CONTIGUOUS','WRITEABLE')
    actionAngleStaeckel_actionsFunc= _lib.actionAngleStaeckel_uminUmaxVmin
    actionAngleStaeckel_actionsFunc.argtypes= [ctypes.c_int,
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ctypes.c_int,
                               ndpointer(dtype=numpy.int32,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ctypes.c_int,
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ctypes.POINTER(ctypes.c_int)]

    #Array requirements, first store old order
    f_cont= [R.flags['F_CONTIGUOUS'],
             vR.flags['F_CONTIGUOUS'],
             vT.flags['F_CONTIGUOUS'],
             z.flags['F_CONTIGUOUS'],
             vz.flags['F_CONTIGUOUS'],
             u0.flags['F_CONTIGUOUS'],
             delta.flags['F_CONTIGUOUS']]
    R= numpy.require(R,dtype=numpy.float64,requirements=['C','W'])
    vR= numpy.require(vR,dtype=numpy.float64,requirements=['C','W'])
    vT= numpy.require(vT,dtype=numpy.float64,requirements=['C','W'])
    z= numpy.require(z,dtype=numpy.float64,requirements=['C','W'])
    vz= numpy.require(vz,dtype=numpy.float64,requirements=['C','W'])
    u0= numpy.require(u0,dtype=numpy.float64,requirements=['C','W'])
    delta= numpy.require(delta,dtype=numpy.float64,requirements=['C','W'])
    umin= numpy.require(umin,dtype=numpy.float64,requirements=['C','W'])
    umax= numpy.require(umax,dtype=numpy.float64,requirements=['C','W'])
    vmin= numpy.require(vmin,dtype=numpy.float64,requirements=['C','W'])

    #Run the C code
    actionAngleStaeckel_actionsFunc(len(R),
                                    R,
                                    vR,
                                    vT,
                                    z,
                                    vz,
                                    u0,
                                    ctypes.c_int(npot),
                                    pot_type,
                                    pot_args,
                                    ctypes.c_int(ndelta),
                                    delta,
                                    umin,
                                    umax,
                                    vmin,
                                    ctypes.byref(err))

    #Reset input arrays
    if f_cont[0]: R= numpy.asfortranarray(R)
    if f_cont[1]: vR= numpy.asfortranarray(vR)
    if f_cont[2]: vT= numpy.asfortranarray(vT)
    if f_cont[3]: z= numpy.asfortranarray(z)
    if f_cont[4]: vz= numpy.asfortranarray(vz)
    if f_cont[5]: u0= numpy.asfortranarray(u0)
    if f_cont[6]: delta= numpy.asfortranarray(delta)

    return (umin,umax,vmin,err.value)

