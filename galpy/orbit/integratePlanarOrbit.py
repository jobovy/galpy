import sys
import distutils.sysconfig as sysconfig
import warnings
import numpy as nu
import ctypes
import ctypes.util
from numpy.ctypeslib import ndpointer
import os
from galpy import potential
from galpy.potential.planarPotential import planarPotentialFromFullPotential, \
    planarPotentialFromRZPotential
from galpy.potential.WrapperPotential import parentWrapperPotential
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
        _lib = ctypes.CDLL(os.path.join(path,'galpy_integrate_c%s' % _ext_suffix))
    except OSError as e:
        if os.path.exists(os.path.join(path,'galpy_integrate_c%s' % _ext_suffix)): #pragma: no cover
            outerr= e
        _lib = None
    else:
        break
if _lib is None: #pragma: no cover
    if not outerr is None:
        warnings.warn("integratePlanarOrbit_c extension module not loaded, because of error '%s' " % outerr,
                      galpyWarning)
    else:
        warnings.warn("integratePlanarOrbit_c extension module not loaded, because galpy_integrate_c%s image was not found" % _ext_suffix,
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
        # Prepare for wrappers
        if ((isinstance(p,planarPotentialFromFullPotential) \
          or isinstance(p,planarPotentialFromRZPotential)) \
          and isinstance(p._Pot,parentWrapperPotential)) \
        or isinstance(p,parentWrapperPotential):
            if not isinstance(p,parentWrapperPotential):
                wrap_npot, wrap_pot_type, wrap_pot_args= \
                    _parse_pot(potential.toPlanarPotential(p._Pot._pot))
            else:
                wrap_npot, wrap_pot_type, wrap_pot_args= _parse_pot(p._pot)
        if (isinstance(p,planarPotentialFromRZPotential) 
            or isinstance(p,planarPotentialFromFullPotential) ) \
                 and isinstance(p._Pot,potential.LogarithmicHaloPotential):
            pot_type.append(0)
            if p._Pot.isNonAxi:
                pot_args.extend([p._Pot._amp,p._Pot._q,
                                 p._Pot._core2,p._Pot._1m1overb2])
            else:
                pot_args.extend([p._Pot._amp,p._Pot._q,p._Pot._core2,2.]) # 1m1overb2 > 1: axi
        elif isinstance(p,planarPotentialFromFullPotential) \
                 and isinstance(p._Pot,potential.DehnenBarPotential):
            pot_type.append(1)
            pot_args.extend([p._Pot._amp*p._Pot._af,p._Pot._tform,
                             p._Pot._tsteady,p._Pot._rb,p._Pot._omegab,
                             p._Pot._barphi])
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
        elif isinstance(p,planarPotentialFromRZPotential) \
                 and isinstance(p._Pot,potential.MiyamotoNagaiPotential):
            pot_type.append(5)
            pot_args.extend([p._Pot._amp,p._Pot._a,p._Pot._b])
        elif isinstance(p,potential.LopsidedDiskPotential):
            pot_type.append(6)
            pot_args.extend([p._amp,p._mphio,p._p,p._phib])
        elif isinstance(p,planarPotentialFromRZPotential) \
                 and isinstance(p._Pot,potential.PowerSphericalPotential):
            pot_type.append(7)
            pot_args.extend([p._Pot._amp,p._Pot.alpha])
        elif isinstance(p,planarPotentialFromRZPotential) \
                 and isinstance(p._Pot,potential.HernquistPotential):
            pot_type.append(8)
            pot_args.extend([p._Pot._amp,p._Pot.a])
        elif isinstance(p,planarPotentialFromRZPotential) \
                 and isinstance(p._Pot,potential.NFWPotential):
            pot_type.append(9)
            pot_args.extend([p._Pot._amp,p._Pot.a])
        elif isinstance(p,planarPotentialFromRZPotential) \
                 and isinstance(p._Pot,potential.JaffePotential):
            pot_type.append(10)
            pot_args.extend([p._Pot._amp,p._Pot.a])
        elif isinstance(p,planarPotentialFromRZPotential) \
                and isinstance(p._Pot,potential.DoubleExponentialDiskPotential):
            pot_type.append(11)
            pot_args.extend([p._Pot._amp,p._Pot._alpha,
                             p._Pot._beta,p._Pot._kmaxFac,
                             p._Pot._nzeros,p._Pot._glorder])
            pot_args.extend([p._Pot._glx[ii] for ii in range(p._Pot._glorder)])
            pot_args.extend([p._Pot._glw[ii] for ii in range(p._Pot._glorder)])
            pot_args.extend([p._Pot._j0zeros[ii] for ii in range(p._Pot._nzeros+1)])
            pot_args.extend([p._Pot._dj0zeros[ii] for ii in range(p._Pot._nzeros+1)])
            pot_args.extend([p._Pot._j1zeros[ii] for ii in range(p._Pot._nzeros+1)])
            pot_args.extend([p._Pot._dj1zeros[ii] for ii in range(p._Pot._nzeros+1)])
            pot_args.extend([p._Pot._kp._amp,p._Pot._kp.alpha])
        elif isinstance(p,planarPotentialFromRZPotential) \
                and isinstance(p._Pot,potential.FlattenedPowerPotential):
            pot_type.append(12)
            pot_args.extend([p._Pot._amp,p._Pot.alpha,p._Pot.core2])
        elif isinstance(p,planarPotentialFromRZPotential) \
                 and isinstance(p._Pot,potential.IsochronePotential):
            pot_type.append(14)
            pot_args.extend([p._Pot._amp,p._Pot.b])
        elif isinstance(p,planarPotentialFromRZPotential) \
                 and isinstance(p._Pot,potential.PowerSphericalPotentialwCutoff):
            pot_type.append(15)
            pot_args.extend([p._Pot._amp,p._Pot.alpha,p._Pot.rc])
        elif isinstance(p,planarPotentialFromRZPotential) \
                 and isinstance(p._Pot,potential.MN3ExponentialDiskPotential):
            # Three Miyamoto-Nagai disks
            npot+= 2
            pot_type.extend([5,5,5])
            pot_args.extend([p._Pot._amp*p._Pot._mn3[0]._amp,
                             p._Pot._mn3[0]._a,p._Pot._mn3[0]._b,
                             p._Pot._amp*p._Pot._mn3[1]._amp,
                             p._Pot._mn3[1]._a,p._Pot._mn3[1]._b,
                             p._Pot._amp*p._Pot._mn3[2]._amp,
                             p._Pot._mn3[2]._a,p._Pot._mn3[2]._b])
        elif isinstance(p,planarPotentialFromRZPotential) \
                 and isinstance(p._Pot,potential.KuzminKutuzovStaeckelPotential):
            pot_type.append(16)
            pot_args.extend([p._Pot._amp,p._Pot._ac,p._Pot._Delta])
        elif isinstance(p,planarPotentialFromRZPotential) \
                 and isinstance(p._Pot,potential.PlummerPotential):
            pot_type.append(17)
            pot_args.extend([p._Pot._amp,p._Pot._b])
        elif isinstance(p,planarPotentialFromRZPotential) \
                 and isinstance(p._Pot,potential.PseudoIsothermalPotential):
            pot_type.append(18)
            pot_args.extend([p._Pot._amp,p._Pot._a])
        elif isinstance(p,planarPotentialFromRZPotential) \
                 and isinstance(p._Pot,potential.KuzminDiskPotential):
            pot_type.append(19)
            pot_args.extend([p._Pot._amp,p._Pot._a])
        elif isinstance(p,planarPotentialFromRZPotential) \
                 and isinstance(p._Pot,potential.BurkertPotential):
            pot_type.append(20)
            pot_args.extend([p._Pot._amp,p._Pot.a])
        elif (isinstance(p,planarPotentialFromFullPotential) or isinstance(p,planarPotentialFromRZPotential)) \
                and isinstance(p._Pot,potential.EllipsoidalPotential.EllipsoidalPotential):
            pot_args.append(p._Pot._amp)
            pot_args.extend([0.,0.,0.,0.,0.,0.]) # for caching
            if isinstance(p._Pot,potential.TriaxialHernquistPotential):
                pot_type.append(21)
                pot_args.extend([2,p._Pot.a,p._Pot.a4]) # for psi, mdens, mdens_deriv
            if isinstance(p._Pot,potential.TriaxialNFWPotential):
                pot_type.append(22)
                pot_args.extend([2,p._Pot.a,p._Pot.a3]) # for psi, mdens, mdens_deriv
            if isinstance(p._Pot,potential.TriaxialJaffePotential):
                pot_type.append(23)
                pot_args.extend([2,p._Pot.a,p._Pot.a2]) # for psi, mdens, mdens_deriv
            elif isinstance(p._Pot,potential.PerfectEllipsoidPotential):
                pot_type.append(30)
                pot_args.extend([1,p._Pot.a2]) # for psi, mdens, mdens_deriv
            pot_args.extend([p._Pot._b2,p._Pot._c2,
                             int(p._Pot._aligned)]) # Reg. Ellipsoidal
            if not p._Pot._aligned:
                pot_args.extend(list(p._Pot._rot.flatten()))
            else:
                pot_args.extend(list(nu.eye(3).flatten())) # not actually used
            pot_args.append(p._Pot._glorder)
            pot_args.extend([p._Pot._glx[ii] for ii in range(p._Pot._glorder)])
            # this adds some common factors to the integration weights
            pot_args.extend([-4.*nu.pi*p._Pot._glw[ii]*p._Pot._b*p._Pot._c\
                            /nu.sqrt(( 1.+(p._Pot._b2-1.)*p._Pot._glx[ii]**2.)
                                     *(1.+(p._Pot._c2-1.)*p._Pot._glx[ii]**2.))
                             for ii in range(p._Pot._glorder)])
        elif (isinstance(p,planarPotentialFromFullPotential) or isinstance(p,planarPotentialFromRZPotential)) \
                 and isinstance(p._Pot,potential.SCFPotential):
            pt,pa= _parse_scf_pot(p._Pot)
            pot_type.append(pt)
            pot_args.extend(pa)
        elif isinstance(p,planarPotentialFromFullPotential) \
                 and isinstance(p._Pot,potential.SoftenedNeedleBarPotential):
            pot_type.append(25)
            pot_args.extend([p._Pot._amp,p._Pot._a,p._Pot._b,p._Pot._c2,
                             p._Pot._pa,p._Pot._omegab])
            pot_args.extend([0.,0.,0.,0.,0.,0.,0.]) # for caching
        elif (isinstance(p,planarPotentialFromFullPotential) or isinstance(p,planarPotentialFromRZPotential)) \
                and isinstance(p._Pot,potential.DiskSCFPotential):
            # Need to pull this apart into: (a) SCF part, (b) constituent
            # [Sigma_i,h_i] parts
            # (a) SCF, multiply in any add'l amp
            pt,pa= _parse_scf_pot(p._Pot._scf,extra_amp=p._Pot._amp)
            pot_type.append(pt)
            pot_args.extend(pa)
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
        elif isinstance(p,planarPotentialFromFullPotential) \
                and isinstance(p._Pot, potential.SpiralArmsPotential):
            pot_type.append(27)
            pot_args.extend([len(p._Pot._Cs), p._Pot._amp, p._Pot._N, p._Pot._sin_alpha,
                             p._Pot._tan_alpha, p._Pot._r_ref, p._Pot._phi_ref, p._Pot._Rs, p._Pot._H, p._Pot._omega])
            pot_args.extend(p._Pot._Cs)
        elif isinstance(p,potential.CosmphiDiskPotential):
            pot_type.append(28)
            pot_args.extend([p._amp,p._mphio,p._p,p._mphib,p._m,
                             p._rb,p._rbp,p._rb2p,p._r1p])
        elif isinstance(p,potential.HenonHeilesPotential):
            pot_type.append(29)
            pot_args.extend([p._amp])
        # 30: PerfectEllipsoidPotential, done with other EllipsoidalPotentials above
        ############################## WRAPPERS ###############################
        elif ((isinstance(p,planarPotentialFromFullPotential) or isinstance(p,planarPotentialFromRZPotential)) \
              and isinstance(p._Pot,potential.DehnenSmoothWrapperPotential)) \
              or isinstance(p,potential.DehnenSmoothWrapperPotential):
            if not isinstance(p,potential.DehnenSmoothWrapperPotential):
                p= p._Pot
            pot_type.append(-1)
            # wrap_pot_type, args, and npot obtained before this horrible if
            pot_args.append(wrap_npot)
            pot_type.extend(wrap_pot_type)
            pot_args.extend(wrap_pot_args)
            pot_args.extend([p._amp,p._tform,p._tsteady])
        elif ((isinstance(p,planarPotentialFromFullPotential) or isinstance(p,planarPotentialFromRZPotential)) \
          and isinstance(p._Pot,potential.SolidBodyRotationWrapperPotential)) \
          or isinstance(p,potential.SolidBodyRotationWrapperPotential):
            if not isinstance(p,potential.SolidBodyRotationWrapperPotential):
                p= p._Pot
            pot_type.append(-2)
            # wrap_pot_type, args, and npot obtained before this horrible if
            pot_args.append(wrap_npot)
            pot_type.extend(wrap_pot_type)
            pot_args.extend(wrap_pot_args)
            pot_args.extend([p._amp,p._omega,p._pa])
        elif ((isinstance(p,planarPotentialFromFullPotential) or isinstance(p,planarPotentialFromRZPotential)) \
          and isinstance(p._Pot,potential.CorotatingRotationWrapperPotential)) \
          or isinstance(p,potential.CorotatingRotationWrapperPotential):
            if not isinstance(p,potential.CorotatingRotationWrapperPotential):
                p= p._Pot
            pot_type.append(-4)
            # wrap_pot_type, args, and npot obtained before this horrible if
            pot_args.append(wrap_npot)
            pot_type.extend(wrap_pot_type)
            pot_args.extend(wrap_pot_args)
            pot_args.extend([p._amp,p._vpo,p._beta,p._pa,p._to])
        elif ((isinstance(p,planarPotentialFromFullPotential) or isinstance(p,planarPotentialFromRZPotential)) \
              and isinstance(p._Pot,potential.GaussianAmplitudeWrapperPotential)) \
              or isinstance(p,potential.GaussianAmplitudeWrapperPotential):
            if not isinstance(p,potential.GaussianAmplitudeWrapperPotential):
                p= p._Pot
            pot_type.append(-5)
            # wrap_pot_type, args, and npot obtained before this horrible if
            pot_args.append(wrap_npot)
            pot_type.extend(wrap_pot_type)
            pot_args.extend(wrap_pot_args)
            pot_args.extend([p._amp,p._to,p._sigma2])
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

def integratePlanarOrbit_c(pot,yo,t,int_method,rtol=None,atol=None,
                           dt=None):
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
       dt= (None) force integrator to use this stepsize (default is to automatically determine one))
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
    if dt is None: 
        dt= -9999.99

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
                    ctypes.c_double(dt),                    
                    ctypes.c_double(rtol),ctypes.c_double(atol),
                    result,
                    ctypes.byref(err),
                    ctypes.c_int(int_method_c))

    if err.value == -10: #pragma: no cover
        raise KeyboardInterrupt("Orbit integration interrupted by CTRL-C (SIGINT)")

    #Reset input arrays
    if f_cont[0]: yo= nu.asfortranarray(yo)
    if f_cont[1]: t= nu.asfortranarray(t)

    return (result,err.value)


def integratePlanarOrbit_dxdv_c(pot,yo,dyo,t,int_method,rtol=None,atol=None,
                                dt=None):
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
       dt= (None) force integrator to use this stepsize (default is to automatically determine one))
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
    if dt is None: 
        dt= -9999.99
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
                    ctypes.c_double(dt),                    
                    ctypes.c_double(rtol),ctypes.c_double(atol),
                    result,
                    ctypes.byref(err),
                    ctypes.c_int(int_method_c))

    if err.value == -10: #pragma: no cover
        raise KeyboardInterrupt("Orbit integration interrupted by CTRL-C (SIGINT)")

    #Reset input arrays
    if f_cont[0]: yo= nu.asfortranarray(yo)
    if f_cont[1]: t= nu.asfortranarray(t)

    return (result,err.value)
