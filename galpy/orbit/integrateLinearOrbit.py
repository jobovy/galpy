import sys
import distutils.sysconfig as sysconfig
import warnings
import numpy as nu
import ctypes
import ctypes.util
from numpy.ctypeslib import ndpointer
import os
from galpy import potential
from galpy.util import galpyWarning
from .integratePlanarOrbit import _parse_integrator, _parse_tol
from .integrateFullOrbit import _parse_pot as _parse_pot_full
from galpy.potential.verticalPotential import verticalPotential
from galpy.potential.WrapperPotential import parentWrapperPotential
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
        _lib = ctypes.CDLL(os.path.join(path,'galpy_integrate_c%s' % _ext_suffix))
    except OSError as e:
        if os.path.exists(os.path.join(path,'galpy_integrate_c%s' % _ext_suffix)): #pragma: no cover
            outerr= e
        _lib = None
    else:
        break
if _lib is None: #pragma: no cover
    if not outerr is None:
        warnings.warn("integrateLinearOrbit_c extension module not loaded, because of error '%s' " % outerr,
                      galpyWarning)
    else:
        warnings.warn("integrateLinearOrbit_c extension module not loaded, because galpy_integrate_c%s image was not found" % _ext_suffix,
                      galpyWarning)
    _ext_loaded= False
else:
    _ext_loaded= True

def _parse_pot(pot):
    """Parse the potential so it can be fed to C"""
    from .integrateFullOrbit import _parse_scf_pot
    #Figure out what's in pot
    if not isinstance(pot,list):
        pot= [pot]
    #Initialize everything
    pot_type= []
    pot_args= []
    npot= len(pot)
    for p in pot:
        # Prepare for wrappers NOT CURRENTLY SUPPORTED, SEE PLANAR OR FULL
        if isinstance(p,verticalPotential) \
                and isinstance(p._Pot,potential.MN3ExponentialDiskPotential):
            # Need to do this one separately, because combination of many parts
            # Three Miyamoto-Nagai disks
            npot+= 2
            pot_type.extend([5,5,5])
            pot_args.extend([p._Pot._amp*p._Pot._mn3[0]._amp,
                             p._Pot._mn3[0]._a,p._Pot._mn3[0]._b,
                             p._R,p._phi,
                             p._Pot._amp*p._Pot._mn3[1]._amp,
                             p._Pot._mn3[1]._a,p._Pot._mn3[1]._b,
                             p._R,p._phi,
                             p._Pot._amp*p._Pot._mn3[2]._amp,
                             p._Pot._mn3[2]._a,p._Pot._mn3[2]._b,
                             p._R,p._phi])
        elif isinstance(p,verticalPotential) \
                and isinstance(p._Pot,potential.DiskSCFPotential):
            # Need to do this one separately, because combination of many parts
            # Need to pull this apart into: (a) SCF part, (b) constituent
            # [Sigma_i,h_i] parts
            # (a) SCF, multiply in any add'l amp
            pt,pa= _parse_scf_pot(p._Pot._scf,extra_amp=p._Pot._amp)
            pot_type.append(pt)
            pot_args.extend(pa)
            pot_args.extend([p._R,p._phi])
            # (b) constituent [Sigma_i,h_i] parts
            for Sigma,hz in zip(p._Pot._Sigma_dict,p._Pot._hz_dict):
                npot+= 1
                pot_type.append(26)
                stype= Sigma.get('type','exp')
                if stype == 'exp' \
                        or (stype == 'exp' and 'Rhole' in Sigma):
                    pot_args.extend([3,0,
                                     4.*nu.pi*Sigma.get('amp',1.)*p._Pot._amp,
                                     Sigma.get('h',1./3.)])
                elif stype == 'expwhole' \
                        or (stype == 'exp' and 'Rhole' in Sigma):
                    pot_args.extend([4,1,
                                     4.*nu.pi*Sigma.get('amp',1.)*p._Pot._amp,
                                     Sigma.get('h',1./3.),
                                     Sigma.get('Rhole',0.5)])
                hztype= hz.get('type','exp')
                if hztype == 'exp':
                    pot_args.extend([0,hz.get('h',0.0375)])
                elif hztype == 'sech2':
                    pot_args.extend([1,hz.get('h',0.0375)])
                pot_args.extend([p._R,p._phi])
        elif isinstance(p,potential.KGPotential):
            pot_type.append(31)
            pot_args.extend([p._amp,p._K,p._D2,2.*p._F]) 
        # All other potentials can be handled in the same way as follows:
        elif isinstance(p,verticalPotential):
            _,pt,pa= _parse_pot_full(p._Pot)
            pot_type.extend(pt)
            pot_args.extend(pa)
            pot_args.append(p._R)
            pot_args.append(p._phi)
    pot_type= nu.array(pot_type,dtype=nu.int32,order='C')
    pot_args= nu.array(pot_args,dtype=nu.float64,order='C')
    return (npot,pot_type,pot_args)

def integrateLinearOrbit_c(pot,yo,t,int_method,rtol=None,atol=None,dt=None):
    """
    NAME:
       integrateLinearOrbit_c
    PURPOSE:
       C integrate an ode for a LinearOrbit
    INPUT:
       pot - Potential or list of such instances
       yo - initial condition [q,p], can be [N,2] or [2]
       t - set of times at which one wants the result
       int_method= 'leapfrog_c', 'rk4_c', 'rk6_c', 'symplec4_c'
       rtol, atol
       dt= (None) force integrator to use this stepsize (default is to automatically determine one))
    OUTPUT:
       (y,err)
       y : array, shape (len(y0), len(t))
       Array containing the value of y for each desired time in t, \
       with the initial value y0 in the first row.
       err: error message, if not zero: 1 means maximum step reduction happened for adaptive integrators
    HISTORY:
       2018-10-06 - Written - Bovy (UofT)
    """
    if len(yo.shape) == 1: single_obj= True
    else: single_obj= False
    yo= nu.atleast_2d(yo)
    nobj= len(yo)
    rtol, atol= _parse_tol(rtol,atol)
    npot, pot_type, pot_args= _parse_pot(pot)
    int_method_c= _parse_integrator(int_method)
    if dt is None: 
        dt= -9999.99

    #Set up result array
    result= nu.empty((nobj,len(t),2))
    err= nu.zeros(nobj,dtype=nu.int32)

    #Set up the C code
    ndarrayFlags= ('C_CONTIGUOUS','WRITEABLE')
    integrationFunc= _lib.integrateLinearOrbit
    integrationFunc.argtypes= [ctypes.c_int,
                               ndpointer(dtype=nu.float64,flags=ndarrayFlags),
                               ctypes.c_int,                             
                               ndpointer(dtype=nu.float64,flags=ndarrayFlags),
                               ctypes.c_int,
                               ndpointer(dtype=nu.int32,flags=ndarrayFlags),
                               ndpointer(dtype=nu.float64,flags=ndarrayFlags),
                               ctypes.c_double,
                               ctypes.c_double,
                               ctypes.c_double,
                               ndpointer(dtype=nu.float64,flags=ndarrayFlags),
                               ndpointer(dtype=nu.int32,flags=ndarrayFlags),
                               ctypes.c_int]

    #Array requirements, first store old order
    f_cont= [yo.flags['F_CONTIGUOUS'],
             t.flags['F_CONTIGUOUS']]
    yo= nu.require(yo,dtype=nu.float64,requirements=['C','W'])
    t= nu.require(t,dtype=nu.float64,requirements=['C','W'])
    result= nu.require(result,dtype=nu.float64,requirements=['C','W'])
    err= nu.require(err,dtype=nu.int32,requirements=['C','W'])

    #Run the C code
    integrationFunc(ctypes.c_int(nobj),
                    yo,
                    ctypes.c_int(len(t)),
                    t,
                    ctypes.c_int(npot),
                    pot_type,
                    pot_args,
                    ctypes.c_double(dt),
                    ctypes.c_double(rtol),ctypes.c_double(atol),
                    result,
                    err,
                    ctypes.c_int(int_method_c))
    
    if nu.any(err == -10): #pragma: no cover
        raise KeyboardInterrupt("Orbit integration interrupted by CTRL-C (SIGINT)")

    #Reset input arrays
    if f_cont[0]: yo= nu.asfortranarray(yo)
    if f_cont[1]: t= nu.asfortranarray(t)

    if single_obj: return (result[0],err[0])
    else: return (result,err)

