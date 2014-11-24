import sys
import warnings
import numpy as nu
import ctypes
import ctypes.util
from numpy.ctypeslib import ndpointer
import os
from galpy import potential, potential_src
from galpy.util import galpyWarning
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
        warnings.warn("integratePlanarOrbit_c extension module not loaded, because of error '%s' " % outerr,
                      galpyWarning)
    else:
        warnings.warn("integratePlanarOrbit_c extension module not loaded, because galpy_integrate_c.so image was not found",
                      galpyWarning)
    _ext_loaded= False
else:
    _ext_loaded= True

def _parse_pot(pot):
    """Parse the potential so it can be fed to C"""
    #Figure out what's in pot
    if not isinstance(pot,list):
        pot= [pot]
    #Initialize everything
    pot_type= []
    pot_args= []
    npot= len(pot)
    for p in pot:
        if isinstance(p,potential_src.planarPotential.planarPotentialFromRZPotential) \
                 and isinstance(p._RZPot,potential.LogarithmicHaloPotential):
            pot_type.append(0)
            pot_args.extend([p._RZPot._amp,p._RZPot._core2])
        elif isinstance(p,potential.DehnenBarPotential):
            pot_type.append(1)
            pot_args.extend([p._amp,p._tform,p._tsteady,p._rb,p._af,p._omegab,
                             p._barphi])
        elif isinstance(p,potential.TransientLogSpiralPotential):
            pot_type.append(2)
            pot_args.extend([p._amp,p._A,p._to,p._sigma2,p._alpha,p._m,
                             p._omegas,p._gamma])
        elif isinstance(p,potential.SteadyLogSpiralPotential):
            pot_type.append(3)
            if p._tform is None:
                pot_args.extend([p._amp,float('nan'), float('nan'),
                                 p._A,p._alpha,p._m,
                                 p._omegas,p._gamma])
            else:
                pot_args.extend([p._amp,p._tform,p._tsteady,p._A,p._alpha,p._m,
                                 p._omegas,p._gamma])
        elif isinstance(p,potential.EllipticalDiskPotential):
            pot_type.append(4)
            if p._tform is None:
                pot_args.extend([p._amp,float('nan'), float('nan'),
                                 p._twophio,p._p,p._phib])
            else:
                pot_args.extend([p._amp,p._tform,p._tsteady,
                                 p._twophio,p._p,p._phib])
        elif isinstance(p,potential_src.planarPotential.planarPotentialFromRZPotential) \
                 and isinstance(p._RZPot,potential.MiyamotoNagaiPotential):
            pot_type.append(5)
            pot_args.extend([p._RZPot._amp,p._RZPot._a,p._RZPot._b])
        elif isinstance(p,potential.LopsidedDiskPotential):
            pot_type.append(6)
            if p._tform is None:
                pot_args.extend([p._amp,float('nan'), float('nan'),
                                 p._mphio,p._p,p._phib])
            else:
                pot_args.extend([p._amp,p._tform,p._tsteady,
                                 p._mphio,p._p,p._phib])
        elif isinstance(p,potential_src.planarPotential.planarPotentialFromRZPotential) \
                 and isinstance(p._RZPot,potential.PowerSphericalPotential):
            pot_type.append(7)
            pot_args.extend([p._RZPot._amp,p._RZPot.alpha])
        elif isinstance(p,potential_src.planarPotential.planarPotentialFromRZPotential) \
                 and isinstance(p._RZPot,potential.HernquistPotential):
            pot_type.append(8)
            pot_args.extend([p._RZPot._amp,p._RZPot.a])
        elif isinstance(p,potential_src.planarPotential.planarPotentialFromRZPotential) \
                 and isinstance(p._RZPot,potential.NFWPotential):
            pot_type.append(9)
            pot_args.extend([p._RZPot._amp,p._RZPot.a])
        elif isinstance(p,potential_src.planarPotential.planarPotentialFromRZPotential) \
                 and isinstance(p._RZPot,potential.JaffePotential):
            pot_type.append(10)
            pot_args.extend([p._RZPot._amp,p._RZPot.a])
        elif isinstance(p,potential_src.planarPotential.planarPotentialFromRZPotential) \
                and isinstance(p._RZPot,potential.DoubleExponentialDiskPotential):
            pot_type.append(11)
            pot_args.extend([p._RZPot._amp,p._RZPot._alpha,
                             p._RZPot._beta,p._RZPot._kmaxFac,
                             p._RZPot._nzeros,p._RZPot._glorder])
            pot_args.extend([p._RZPot._glx[ii] for ii in range(p._RZPot._glorder)])
            pot_args.extend([p._RZPot._glw[ii] for ii in range(p._RZPot._glorder)])
            pot_args.extend([p._RZPot._j0zeros[ii] for ii in range(p._RZPot._nzeros+1)])
            pot_args.extend([p._RZPot._dj0zeros[ii] for ii in range(p._RZPot._nzeros+1)])
            pot_args.extend([p._RZPot._j1zeros[ii] for ii in range(p._RZPot._nzeros+1)])
            pot_args.extend([p._RZPot._dj1zeros[ii] for ii in range(p._RZPot._nzeros+1)])
            pot_args.extend([p._RZPot._kp._amp,p._RZPot._kp.alpha])
        elif isinstance(p,potential_src.planarPotential.planarPotentialFromRZPotential) \
                and isinstance(p._RZPot,potential.FlattenedPowerPotential):
            pot_type.append(12)
            pot_args.extend([p._RZPot._amp,p._RZPot.alpha,p._RZPot.core2])
        elif isinstance(p,potential_src.planarPotential.planarPotentialFromRZPotential) \
                 and isinstance(p._RZPot,potential.IsochronePotential):
            pot_type.append(14)
            pot_args.extend([p._RZPot._amp,p._RZPot.b])
        elif isinstance(p,potential_src.planarPotential.planarPotentialFromRZPotential) \
                 and isinstance(p._RZPot,potential.PowerSphericalPotentialwCutoff):
            pot_type.append(15)
            pot_args.extend([p._RZPot._amp,p._RZPot.alpha,p._RZPot.rc])
    pot_type= nu.array(pot_type,dtype=nu.int32,order='C')
    pot_args= nu.array(pot_args,dtype=nu.float64,order='C')
    return (npot,pot_type,pot_args)

def _parse_integrator(int_method):
    """parse the integrator method to pass to C"""
    #Pick integrator
    if int_method.lower() == 'rk4_c':
        int_method_c= 1
    elif int_method.lower() == 'rk6_c':
        int_method_c= 2
    elif int_method.lower() == 'symplec4_c':
        int_method_c= 3
    elif int_method.lower() == 'symplec6_c':
        int_method_c= 4
    elif int_method.lower() == 'dopr54_c':
        int_method_c= 5
    else:
        int_method_c= 0
    return int_method_c
            
def _parse_tol(rtol,atol):
    """Parse the tolerance keywords"""
    #Process atol and rtol
    if rtol is None:
        rtol= -12.*nu.log(10.)
    else: #pragma: no cover
        rtol= nu.log(rtol)
    if atol is None:
        atol= -12.*nu.log(10.)
    else: #pragma: no cover
        atol= nu.log(atol)
    return (rtol,atol)

def integratePlanarOrbit_c(pot,yo,t,int_method,rtol=None,atol=None):
    """
    NAME:
       integratePlanarOrbit_c
    PURPOSE:
       C integrate an ode for a planarOrbit
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
       2011-10-03 - Written - Bovy (IAS)
    """
    rtol, atol= _parse_tol(rtol,atol)
    npot, pot_type, pot_args= _parse_pot(pot)
    int_method_c= _parse_integrator(int_method)

    #Set up result array
    result= nu.empty((len(t),4))
    err= ctypes.c_int(0)

    #Set up the C code
    ndarrayFlags= ('C_CONTIGUOUS','WRITEABLE')
    integrationFunc= _lib.integratePlanarOrbit
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


def integratePlanarOrbit_dxdv_c(pot,yo,dyo,t,int_method,rtol=None,atol=None):
    """
    NAME:
       integratePlanarOrbit_dxdv_c
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
       2011-10-19 - Written - Bovy (IAS)
    """
    rtol, atol= _parse_tol(rtol,atol)
    npot, pot_type, pot_args= _parse_pot(pot)
    int_method_c= _parse_integrator(int_method)
    yo= nu.concatenate((yo,dyo))

    #Set up result array
    result= nu.empty((len(t),8))
    err= ctypes.c_int(0)

    #Set up the C code
    ndarrayFlags= ('C_CONTIGUOUS','WRITEABLE')
    integrationFunc= _lib.integratePlanarOrbit_dxdv
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
