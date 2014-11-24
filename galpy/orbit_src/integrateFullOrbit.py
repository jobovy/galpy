import sys
import warnings
import numpy as nu
import ctypes
import ctypes.util
from numpy.ctypeslib import ndpointer
import os
from galpy import potential
from galpy.util import galpyWarning
from galpy.orbit_src.integratePlanarOrbit import _parse_integrator, _parse_tol
#Find and load the library
_lib= None
outerr= None
for path in sys.path:
    try:
        _lib = ctypes.CDLL(os.path.join(path,'galpy_integrate_c.so'))
    except OSError, e:
        if os.path.exists(os.path.join(path,'galpy_integrate_c.so')): #pragma: no cover
            outerr= e
        _lib = None
    else:
        break
if _lib is None: #pragma: no cover
    if not outerr is None:
        warnings.warn("integrateFullOrbit_c extension module not loaded, because of error '%s' " % outerr,
                      galpyWarning)
    else:
        warnings.warn("integrateFullOrbit_c extension module not loaded, because galpy_integrate_c.so image was not found",
                      galpyWarning)
    _ext_loaded= False
else:
    _ext_loaded= True

def _parse_pot(pot,potforactions=False):
    """Parse the potential so it can be fed to C"""
    #Figure out what's in pot
    if not isinstance(pot,list):
        pot= [pot]
    #Initialize everything
    pot_type= []
    pot_args= []
    npot= len(pot)
    for p in pot:
        if isinstance(p,potential.LogarithmicHaloPotential):
            pot_type.append(0)
            pot_args.extend([p._amp,p._q,p._core2])
        elif isinstance(p,potential.MiyamotoNagaiPotential):
            pot_type.append(5)
            pot_args.extend([p._amp,p._a,p._b])
        elif isinstance(p,potential.PowerSphericalPotential):
            pot_type.append(7)
            pot_args.extend([p._amp,p.alpha])
        elif isinstance(p,potential.HernquistPotential):
            pot_type.append(8)
            pot_args.extend([p._amp,p.a])
        elif isinstance(p,potential.NFWPotential):
            pot_type.append(9)
            pot_args.extend([p._amp,p.a])
        elif isinstance(p,potential.JaffePotential):
            pot_type.append(10)
            pot_args.extend([p._amp,p.a])
        elif isinstance(p,potential.DoubleExponentialDiskPotential):
            pot_type.append(11)
            pot_args.extend([p._amp,p._alpha,p._beta,p._kmaxFac,
                             p._nzeros,p._glorder])
            pot_args.extend([p._glx[ii] for ii in range(p._glorder)])
            pot_args.extend([p._glw[ii] for ii in range(p._glorder)])
            pot_args.extend([p._j0zeros[ii] for ii in range(p._nzeros+1)])
            pot_args.extend([p._dj0zeros[ii] for ii in range(p._nzeros+1)])
            pot_args.extend([p._j1zeros[ii] for ii in range(p._nzeros+1)])
            pot_args.extend([p._dj1zeros[ii] for ii in range(p._nzeros+1)])
            pot_args.extend([p._kp._amp,p._kp.alpha])
        elif isinstance(p,potential.FlattenedPowerPotential):
            pot_type.append(12)
            pot_args.extend([p._amp,p.alpha,p.q2,p.core2])
        elif isinstance(p,potential.interpRZPotential):
            pot_type.append(13)
            pot_args.extend([len(p._rgrid),len(p._zgrid)])
            if p._logR:
                pot_args.extend([p._logrgrid[ii] for ii in range(len(p._rgrid))])
            else:
                pot_args.extend([p._rgrid[ii] for ii in range(len(p._rgrid))])
            pot_args.extend([p._zgrid[ii] for ii in range(len(p._zgrid))])
            if potforactions:
                pot_args.extend([x for x in p._potGrid_splinecoeffs.flatten(order='C')])
            else:
                pot_args.extend([x for x in p._rforceGrid_splinecoeffs.flatten(order='C')])
                pot_args.extend([x for x in p._zforceGrid_splinecoeffs.flatten(order='C')])
            pot_args.extend([p._amp,int(p._logR)])
        elif isinstance(p,potential.IsochronePotential):
            pot_type.append(14)
            pot_args.extend([p._amp,p.b])
        elif isinstance(p,potential.PowerSphericalPotentialwCutoff):
            pot_type.append(15)
            pot_args.extend([p._amp,p.alpha,p.rc])
    pot_type= nu.array(pot_type,dtype=nu.int32,order='C')
    pot_args= nu.array(pot_args,dtype=nu.float64,order='C')
    return (npot,pot_type,pot_args)

def integrateFullOrbit_c(pot,yo,t,int_method,rtol=None,atol=None):
    """
    NAME:
       integrateFullOrbit_c
    PURPOSE:
       C integrate an ode for a FullOrbit
    INPUT:
       pot - Potential or list of such instances
       yo - initial condition [q,p]
       t - set of times at which one wants the result
       int_method= 'leapfrog_c', 'rk4_c', 'rk6_c', 'symplec4_c'
       rtol, atol
    OUTPUT:
       (y,err)
       y : array, shape (len(y0), len(t))
       Array containing the value of y for each desired time in t, \
       with the initial value y0 in the first row.
       err: error message, if not zero: 1 means maximum step reduction happened for adaptive integrators
    HISTORY:
       2011-11-13 - Written - Bovy (IAS)
    """
    rtol, atol= _parse_tol(rtol,atol)
    npot, pot_type, pot_args= _parse_pot(pot)
    int_method_c= _parse_integrator(int_method)

    #Set up result array
    result= nu.empty((len(t),6))
    err= ctypes.c_int(0)

    #Set up the C code
    ndarrayFlags= ('C_CONTIGUOUS','WRITEABLE')
    integrationFunc= _lib.integrateFullOrbit
    integrationFunc.argtypes= [ndpointer(dtype=nu.float64,flags=ndarrayFlags),
                               ctypes.c_int,                             
                               ndpointer(dtype=nu.float64,flags=ndarrayFlags),
                               ctypes.c_int,
                               ndpointer(dtype=nu.int32,flags=ndarrayFlags),
                               ndpointer(dtype=nu.float64,flags=ndarrayFlags),
                               ctypes.c_double,
                               ctypes.c_double,
                               ndpointer(dtype=nu.float64,flags=ndarrayFlags),
                               ctypes.POINTER(ctypes.c_int),
                               ctypes.c_int]

    #Array requirements, first store old order
    f_cont= [yo.flags['F_CONTIGUOUS'],
             t.flags['F_CONTIGUOUS']]
    yo= nu.require(yo,dtype=nu.float64,requirements=['C','W'])
    t= nu.require(t,dtype=nu.float64,requirements=['C','W'])
    result= nu.require(result,dtype=nu.float64,requirements=['C','W'])

    #Run the C code
    integrationFunc(yo,
                    ctypes.c_int(len(t)),
                    t,
                    ctypes.c_int(npot),
                    pot_type,
                    pot_args,
                    ctypes.c_double(rtol),ctypes.c_double(atol),
                    result,
                    ctypes.byref(err),
                    ctypes.c_int(int_method_c))

    #Reset input arrays
    if f_cont[0]: yo= nu.asfortranarray(yo)
    if f_cont[1]: t= nu.asfortranarray(t)

    return (result,err.value)

def integrateFullOrbit_dxdv_c(pot,yo,dyo,t,int_method,rtol=None,atol=None): #pragma: no cover because not included in v1, uncover when included
    """
    NAME:
       integrateFullOrbit_dxdv_c
    PURPOSE:
       C integrate an ode for a planarOrbit+phase space volume dxdv
    INPUT:
       pot - Potential or list of such instances
       yo - initial condition [q,p]
       dyo - initial condition [dq,dp]
       t - set of times at which one wants the result
       int_method= 'leapfrog_c', 'rk4_c', 'rk6_c', 'symplec4_c'
       rtol, atol
    OUTPUT:
       (y,err)
       y : array, shape (len(y0), len(t))
       Array containing the value of y for each desired time in t, \
       with the initial value y0 in the first row.
       err: error message if not zero, 1: maximum step reduction happened for adaptive integrators
    HISTORY:
       2011-11-13 - Written - Bovy (IAS)
    """
    rtol, atol= _parse_tol(rtol,atol)
    npot, pot_type, pot_args= _parse_pot(pot)
    int_method_c= _parse_integrator(int_method)
    yo= nu.concatenate((yo,dyo))

    #Set up result array
    result= nu.empty((len(t),12))
    err= ctypes.c_int(0)

    #Set up the C code
    ndarrayFlags= ('C_CONTIGUOUS','WRITEABLE')
    integrationFunc= _lib.integrateFullOrbit_dxdv
    integrationFunc.argtypes= [ndpointer(dtype=nu.float64,flags=ndarrayFlags),
                               ctypes.c_int,                             
                               ndpointer(dtype=nu.float64,flags=ndarrayFlags),
                               ctypes.c_int,
                               ndpointer(dtype=nu.int32,flags=ndarrayFlags),
                               ndpointer(dtype=nu.float64,flags=ndarrayFlags),
                               ctypes.c_double,
                               ctypes.c_double,
                               ndpointer(dtype=nu.float64,flags=ndarrayFlags),
                               ctypes.POINTER(ctypes.c_int),
                               ctypes.c_int]

    #Array requirements, first store old order
    f_cont= [yo.flags['F_CONTIGUOUS'],
             t.flags['F_CONTIGUOUS']]
    yo= nu.require(yo,dtype=nu.float64,requirements=['C','W'])
    t= nu.require(t,dtype=nu.float64,requirements=['C','W'])
    result= nu.require(result,dtype=nu.float64,requirements=['C','W'])

    #Run the C code
    integrationFunc(yo,
                    ctypes.c_int(len(t)),
                    t,
                    ctypes.c_int(npot),
                    pot_type,
                    pot_args,
                    ctypes.c_double(rtol),ctypes.c_double(atol),
                    result,
                    ctypes.byref(err),
                    ctypes.c_int(int_method_c))

    #Reset input arrays
    if f_cont[0]: yo= nu.asfortranarray(yo)
    if f_cont[1]: t= nu.asfortranarray(t)

    return (result,err.value)
