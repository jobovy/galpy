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
        warnings.warn("integrateFullOrbit_c extension module not loaded, because of error '%s' " % outerr,
                      galpyWarning)
    else:
        warnings.warn("integrateFullOrbit_c extension module not loaded, because galpy_integrate_c%s image was not found" % _ext_suffix,
                      galpyWarning)
    _ext_loaded= False
else:
    _ext_loaded= True

def _parse_pot(pot,potforactions=False,potfortorus=False):
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
            if p.isNonAxi:
                pot_args.extend([p._amp,p._q,p._core2,p._1m1overb2])
            else:
                pot_args.extend([p._amp,p._q,p._core2,2.]) # 1m1overb2 > 1: axi
        elif isinstance(p,potential.DehnenBarPotential):
            pot_type.append(1)
            pot_args.extend([p._amp*p._af,p._tform,p._tsteady,p._rb,p._omegab,
                             p._barphi])
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
            if hasattr(p,'_potGrid_splinecoeffs'):
                pot_args.extend([x for x in p._potGrid_splinecoeffs.flatten(order='C')])
            else: # pragma: no cover
                warnings.warn("You are attempting to use the C implementation of interpRZPotential, but have not interpolated the potential itself; if you think this is needed for what you want to do, initialize the interpRZPotential instance with interpPot=True",
                      galpyWarning)
                pot_args.extend(list(nu.ones(len(p._rgrid)*len(p._zgrid))))
            if hasattr(p,'_rforceGrid_splinecoeffs'):
                pot_args.extend([x for x in p._rforceGrid_splinecoeffs.flatten(order='C')])
            else: # pragma: no cover
                warnings.warn("You are attempting to use the C implementation of interpRZPotential, but have not interpolated the Rforce; if you think this is needed for what you want to do, initialize the interpRZPotential instance with interpRforce=True",
                      galpyWarning)
                pot_args.extend(list(nu.ones(len(p._rgrid)*len(p._zgrid))))
            if hasattr(p,'_zforceGrid_splinecoeffs'):
                pot_args.extend([x for x in p._zforceGrid_splinecoeffs.flatten(order='C')])
            else: # pragma: no cover
                warnings.warn("You are attempting to use the C implementation of interpRZPotential, but have not interpolated the zforce; if you think this is needed for what you want to do, initialize the interpRZPotential instance with interpzforce=True",
                      galpyWarning)
                pot_args.extend(list(nu.ones(len(p._rgrid)*len(p._zgrid))))
            pot_args.extend([p._amp,int(p._logR)])
        elif isinstance(p,potential.IsochronePotential):
            pot_type.append(14)
            pot_args.extend([p._amp,p.b])
        elif isinstance(p,potential.PowerSphericalPotentialwCutoff):
            pot_type.append(15)
            pot_args.extend([p._amp,p.alpha,p.rc])
        elif isinstance(p,potential.MN3ExponentialDiskPotential):
            # Three Miyamoto-Nagai disks
            npot+= 2
            pot_type.extend([5,5,5])
            pot_args.extend([p._amp*p._mn3[0]._amp,
                             p._mn3[0]._a,p._mn3[0]._b,
                             p._amp*p._mn3[1]._amp,
                             p._mn3[1]._a,p._mn3[1]._b,
                             p._amp*p._mn3[2]._amp,
                             p._mn3[2]._a,p._mn3[2]._b])
        elif isinstance(p,potential.KuzminKutuzovStaeckelPotential):
            pot_type.append(16)
            pot_args.extend([p._amp,p._ac,p._Delta])
        elif isinstance(p,potential.PlummerPotential):
            pot_type.append(17)
            pot_args.extend([p._amp,p._b])
        elif isinstance(p,potential.PseudoIsothermalPotential):
            pot_type.append(18)
            pot_args.extend([p._amp,p._a])
        elif isinstance(p,potential.KuzminDiskPotential):
            pot_type.append(19)
            pot_args.extend([p._amp,p._a])
        elif isinstance(p,potential.BurkertPotential):
            pot_type.append(20)
            pot_args.extend([p._amp,p.a])
        elif isinstance(p,potential.EllipsoidalPotential.EllipsoidalPotential):
            pot_args.append(p._amp)
            pot_args.extend([0.,0.,0.,0.,0.,0.]) # for caching
            # Potential specific parameters
            if isinstance(p,potential.TriaxialHernquistPotential):
                pot_type.append(21)
                pot_args.extend([2,p.a,p.a4]) # for psi, mdens, mdens_deriv
            elif isinstance(p,potential.TriaxialNFWPotential):
                pot_type.append(22)
                pot_args.extend([2,p.a,p.a3]) # for psi, mdens, mdens_deriv
            elif isinstance(p,potential.TriaxialJaffePotential):
                pot_type.append(23)
                pot_args.extend([2,p.a,p.a2]) # for psi, mdens, mdens_deriv
            elif isinstance(p,potential.PerfectEllipsoidPotential):
                pot_type.append(30)
                pot_args.extend([1,p.a2]) # for psi, mdens, mdens_deriv
            pot_args.extend([p._b2,p._c2,int(p._aligned)]) # Reg. Ellipsoidal
            if not p._aligned:
                pot_args.extend(list(p._rot.flatten()))
            else:
                pot_args.extend(list(nu.eye(3).flatten())) # not actually used
            pot_args.append(p._glorder)
            pot_args.extend([p._glx[ii] for ii in range(p._glorder)])
            # this adds some common factors to the integration weights
            pot_args.extend([-4.*nu.pi*p._glw[ii]*p._b*p._c\
                                  /nu.sqrt(( 1.+(p._b2-1.)*p._glx[ii]**2.)
                                           *(1.+(p._c2-1.)*p._glx[ii]**2.))
                             for ii in range(p._glorder)])
        elif isinstance(p,potential.SCFPotential):
            # Type 24, see stand-alone parser below
            pt,pa= _parse_scf_pot(p)
            pot_type.append(pt)
            pot_args.extend(pa)
        elif isinstance(p,potential.SoftenedNeedleBarPotential):
            pot_type.append(25)
            pot_args.extend([p._amp,p._a,p._b,p._c2,p._pa,p._omegab])
            pot_args.extend([0.,0.,0.,0.,0.,0.,0.]) # for caching
        elif isinstance(p,potential.DiskSCFPotential):
            # Need to pull this apart into: (a) SCF part, (b) constituent
            # [Sigma_i,h_i] parts
            # (a) SCF, multiply in any add'l amp
            pt,pa= _parse_scf_pot(p._scf,extra_amp=p._amp)
            pot_type.append(pt)
            pot_args.extend(pa)
            # (b) constituent [Sigma_i,h_i] parts
            for Sigma,hz in zip(p._Sigma_dict,p._hz_dict):
                npot+= 1
                pot_type.append(26)
                stype= Sigma.get('type','exp')
                if stype == 'exp' \
                        or (stype == 'exp' and 'Rhole' in Sigma):
                    pot_args.extend([3,0,
                                     4.*nu.pi*Sigma.get('amp',1.)*p._amp,
                                     Sigma.get('h',1./3.)])
                elif stype == 'expwhole' \
                        or (stype == 'exp' and 'Rhole' in Sigma):
                    pot_args.extend([4,1,
                                     4.*nu.pi*Sigma.get('amp',1.)*p._amp,
                                     Sigma.get('h',1./3.),
                                     Sigma.get('Rhole',0.5)])
                hztype= hz.get('type','exp')
                if hztype == 'exp':
                    pot_args.extend([0,hz.get('h',0.0375)])
                elif hztype == 'sech2':
                    pot_args.extend([1,hz.get('h',0.0375)])
        elif isinstance(p, potential.SpiralArmsPotential):
            pot_type.append(27)
            pot_args.extend([len(p._Cs), p._amp, p._N, p._sin_alpha, p._tan_alpha, p._r_ref, p._phi_ref,
                             p._Rs, p._H, p._omega])
            pot_args.extend(p._Cs)
        # 30: PerfectEllipsoidPotential, done with others above
        ############################## WRAPPERS ###############################
        elif isinstance(p,potential.DehnenSmoothWrapperPotential):
            pot_type.append(-1)
            wrap_npot, wrap_pot_type, wrap_pot_args= \
                _parse_pot(p._pot,
                           potforactions=potforactions,potfortorus=potfortorus)
            pot_args.append(wrap_npot)
            pot_type.extend(wrap_pot_type)
            pot_args.extend(wrap_pot_args)
            pot_args.extend([p._amp,p._tform,p._tsteady])
        elif isinstance(p,potential.SolidBodyRotationWrapperPotential):
            pot_type.append(-2)
            # Not sure how to easily avoid this duplication
            wrap_npot, wrap_pot_type, wrap_pot_args= \
                _parse_pot(p._pot,
                           potforactions=potforactions,potfortorus=potfortorus)
            pot_args.append(wrap_npot)
            pot_type.extend(wrap_pot_type)
            pot_args.extend(wrap_pot_args)
            pot_args.extend([p._amp,p._omega,p._pa])
        elif isinstance(p,potential.CorotatingRotationWrapperPotential):
            pot_type.append(-4)
            # Not sure how to easily avoid this duplication
            wrap_npot, wrap_pot_type, wrap_pot_args= \
                _parse_pot(p._pot,
                           potforactions=potforactions,potfortorus=potfortorus)
            pot_args.append(wrap_npot)
            pot_type.extend(wrap_pot_type)
            pot_args.extend(wrap_pot_args)
            pot_args.extend([p._amp,p._vpo,p._beta,p._pa,p._to])
        elif isinstance(p,potential.GaussianAmplitudeWrapperPotential):
            pot_type.append(-5)
            wrap_npot, wrap_pot_type, wrap_pot_args= \
                _parse_pot(p._pot,
                           potforactions=potforactions,potfortorus=potfortorus)
            pot_args.append(wrap_npot)
            pot_type.extend(wrap_pot_type)
            pot_args.extend(wrap_pot_args)
            pot_args.extend([p._amp,p._to,p._sigma2])
    pot_type= nu.array(pot_type,dtype=nu.int32,order='C')
    pot_args= nu.array(pot_args,dtype=nu.float64,order='C')
    return (npot,pot_type,pot_args)

def _parse_scf_pot(p,extra_amp=1.):
    # Stand-alone parser for SCF, bc re-used
    isNonAxi= p.isNonAxi
    pot_args= [p._a, isNonAxi]
    pot_args.extend(p._Acos.shape)
    pot_args.extend(extra_amp*p._amp*p._Acos.flatten(order='C'))
    if isNonAxi:
        pot_args.extend(extra_amp*p._amp*p._Asin.flatten(order='C'))   
    pot_args.extend([-1.,0,0,0,0,0,0])    
    return (24,pot_args)

def integrateFullOrbit_c(pot,yo,t,int_method,rtol=None,atol=None,dt=None):
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
       dt= (None) force integrator to use this stepsize (default is to automatically determine one))
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
    if dt is None: 
        dt= -9999.99

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
    
    if int(err.value) == -10: #pragma: no cover
        raise KeyboardInterrupt("Orbit integration interrupted by CTRL-C (SIGINT)")

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

    if int(err.value) == -10: #pragma: no cover
        raise KeyboardInterrupt("Orbit integration interrupted by CTRL-C (SIGINT)")

    #Reset input arrays
    if f_cont[0]: yo= nu.asfortranarray(yo)
    if f_cont[1]: t= nu.asfortranarray(t)

    return (result,err.value)
