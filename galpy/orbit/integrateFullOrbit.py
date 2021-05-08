import warnings
import ctypes
import ctypes.util
from numpy.ctypeslib import ndpointer
import numpy
from scipy import integrate
from .. import potential
from ..util import galpyWarning
from ..potential.Potential import _evaluateRforces, _evaluatezforces,\
    _evaluatephiforces
from .integratePlanarOrbit import _parse_integrator, _parse_tol
from ..util.multi import parallel_map
from ..util.leung_dop853 import dop853
from ..util import symplecticode
from ..util import _load_extension_libs

_lib, _ext_loaded= _load_extension_libs.load_libgalpy()

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
            pot_args.extend([p._amp,-4.*numpy.pi*p._alpha*p._amp,
                             p._alpha,p._beta,len(p._de_j1_xs)])
            pot_args.extend(p._de_j0_xs)
            pot_args.extend(p._de_j1_xs)
            pot_args.extend(p._de_j0_weights)
            pot_args.extend(p._de_j1_weights)
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
                pot_args.extend(list(numpy.ones(len(p._rgrid)*len(p._zgrid))))
            if hasattr(p,'_rforceGrid_splinecoeffs'):
                pot_args.extend([x for x in p._rforceGrid_splinecoeffs.flatten(order='C')])
            else: # pragma: no cover
                warnings.warn("You are attempting to use the C implementation of interpRZPotential, but have not interpolated the Rforce; if you think this is needed for what you want to do, initialize the interpRZPotential instance with interpRforce=True",
                      galpyWarning)
                pot_args.extend(list(numpy.ones(len(p._rgrid)*len(p._zgrid))))
            if hasattr(p,'_zforceGrid_splinecoeffs'):
                pot_args.extend([x for x in p._zforceGrid_splinecoeffs.flatten(order='C')])
            else: # pragma: no cover
                warnings.warn("You are attempting to use the C implementation of interpRZPotential, but have not interpolated the zforce; if you think this is needed for what you want to do, initialize the interpRZPotential instance with interpzforce=True",
                      galpyWarning)
                pot_args.extend(list(numpy.ones(len(p._rgrid)*len(p._zgrid))))
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
            elif isinstance(p,potential.TriaxialGaussianPotential):
                pot_type.append(37)
                pot_args.extend([1,-p._twosigma2]) # for psi, mdens, mdens_deriv
            elif isinstance(p,potential.PowerTriaxialPotential):
                pot_type.append(38)
                pot_args.extend([1,p.alpha]) # for psi, mdens, mdens_deriv
            pot_args.extend([p._b2,p._c2,int(p._aligned)]) # Reg. Ellipsoidal
            if not p._aligned:
                pot_args.extend(list(p._rot.flatten()))
            else:
                pot_args.extend(list(numpy.eye(3).flatten())) # not actually used
            pot_args.append(p._glorder)
            pot_args.extend([p._glx[ii] for ii in range(p._glorder)])
            # this adds some common factors to the integration weights
            pot_args.extend([-4.*numpy.pi*p._glw[ii]*p._b*p._c\
                                  /numpy.sqrt(( 1.+(p._b2-1.)*p._glx[ii]**2.)
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
                if stype == 'exp' and not 'Rhole' in Sigma:
                    pot_args.extend([3,0,
                                     4.*numpy.pi*Sigma.get('amp',1.)*p._amp,
                                     Sigma.get('h',1./3.)])
                elif stype == 'expwhole' \
                        or (stype == 'exp' and 'Rhole' in Sigma):
                    pot_args.extend([4,1,
                                     4.*numpy.pi*Sigma.get('amp',1.)*p._amp,
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
        # 31: KGPotential
        # 32: IsothermalDiskPotential
        elif isinstance(p,potential.DehnenCoreSphericalPotential):
            pot_type.append(33)
            pot_args.extend([p._amp,p.a])
        elif isinstance(p,potential.DehnenSphericalPotential):
            pot_type.append(34)
            pot_args.extend([p._amp,p.a,p.alpha])
        elif isinstance(p,potential.HomogeneousSpherePotential):
            pot_type.append(35)
            pot_args.extend([p._amp,p._R2,p._R3])
        elif isinstance(p,potential.interpSphericalPotential):
            pot_type.append(36)
            pot_args.append(len(p._rgrid))
            pot_args.extend(p._rgrid)
            pot_args.extend(p._rforce_grid)
            pot_args.extend([p._amp,p._rmin,p._rmax,p._total_mass,
                             p._Phi0,p._Phimax])
        # 37: TriaxialGaussianPotential, done with others above
        # 38: PowerTriaxialPotential, done with others above
        ############################## WRAPPERS ###############################
        elif isinstance(p,potential.DehnenSmoothWrapperPotential):
            pot_type.append(-1)
            wrap_npot, wrap_pot_type, wrap_pot_args= \
                _parse_pot(p._pot,
                           potforactions=potforactions,potfortorus=potfortorus)
            pot_args.append(wrap_npot)
            pot_type.extend(wrap_pot_type)
            pot_args.extend(wrap_pot_args)
            pot_args.extend([p._amp,p._tform,p._tsteady,int(p._grow)])
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
        elif isinstance(p,potential.MovingObjectPotential):
            pot_type.append(-6)
            wrap_npot, wrap_pot_type, wrap_pot_args= \
                _parse_pot(p._pot,
                           potforactions=potforactions,potfortorus=potfortorus)
            pot_args.append(wrap_npot)
            pot_type.extend(wrap_pot_type)
            pot_args.extend(wrap_pot_args)
            pot_args.extend([len(p._orb.t)])
            pot_args.extend(p._orb.t)
            pot_args.extend(p._orb.x(p._orb.t,use_physical=False))
            pot_args.extend(p._orb.y(p._orb.t,use_physical=False))
            pot_args.extend(p._orb.z(p._orb.t,use_physical=False))
            pot_args.extend([p._amp])
            pot_args.extend([p._orb.t[0],p._orb.t[-1]]) #t_0, t_f
        elif isinstance(p,potential.ChandrasekharDynamicalFrictionForce):
            pot_type.append(-7)
            wrap_npot, wrap_pot_type, wrap_pot_args= \
                _parse_pot(p._dens_pot,
                           potforactions=potforactions,potfortorus=potfortorus)
            pot_args.append(wrap_npot)
            pot_type.extend(wrap_pot_type)
            pot_args.extend(wrap_pot_args)
            pot_args.extend([len(p._sigmar_rs_4interp)])
            pot_args.extend(p._sigmar_rs_4interp)
            pot_args.extend(p._sigmars_4interp)
            pot_args.extend([p._amp])
            pot_args.extend([-1.,0.,0.,0.,0.,0.,0.,0.]) # for caching
            pot_args.extend([p._ms,p._rhm,p._gamma**2.,
                             -1 if not p._lnLambda else p._lnLambda,
                             p._minr**2.])
            pot_args.extend([p._sigmar_rs_4interp[0],
                             p._sigmar_rs_4interp[-1]]) #r_0, r_f
        elif isinstance(p,potential.RotateAndTiltWrapperPotential):
            pot_type.append(-8)
            # Not sure how to easily avoid this duplication
            wrap_npot, wrap_pot_type, wrap_pot_args= \
                _parse_pot(p._pot,
                           potforactions=potforactions,potfortorus=potfortorus)
            pot_args.append(wrap_npot)
            pot_type.extend(wrap_pot_type)
            pot_args.extend(wrap_pot_args)
            pot_args.extend([p._amp])
            pot_args.extend([0.,0.,0.,0.,0.,0.]) # for caching
            pot_args.extend(list(p._rot.flatten()))
    pot_type= numpy.array(pot_type,dtype=numpy.int32,order='C')
    pot_args= numpy.array(pot_args,dtype=numpy.float64,order='C')
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
       yo - initial condition [q,p] , can be [N,6] or [6]
       t - set of times at which one wants the result
       int_method= 'leapfrog_c', 'rk4_c', 'rk6_c', 'symplec4_c'
       rtol, atol
       dt= (None) force integrator to use this stepsize (default is to automatically determine one; only for C-based integrators)
    OUTPUT:
       (y,err)
       y : array, shape (N,len(t),6)  or (len(t),6) if N = 1
       Array containing the value of y for each desired time in t, \
       with the initial value y0 in the first row.
       err: error message, if not zero: 1 means maximum step reduction happened for adaptive integrators
    HISTORY:
       2011-11-13 - Written - Bovy (IAS)
       2018-12-21 - Adapted to allow multiple objects - Bovy (UofT)
    """
    if len(yo.shape) == 1: single_obj= True
    else: single_obj= False
    yo= numpy.atleast_2d(yo)
    nobj= len(yo)
    rtol, atol= _parse_tol(rtol,atol)
    npot, pot_type, pot_args= _parse_pot(pot)
    int_method_c= _parse_integrator(int_method)
    if dt is None:
        dt= -9999.99

    #Set up result array
    result= numpy.empty((nobj,len(t),6))
    err= numpy.zeros(nobj,dtype=numpy.int32)

    #Set up the C code
    ndarrayFlags= ('C_CONTIGUOUS','WRITEABLE')
    integrationFunc= _lib.integrateFullOrbit
    integrationFunc.argtypes= [ctypes.c_int,
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ctypes.c_int,
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ctypes.c_int,
                               ndpointer(dtype=numpy.int32,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ctypes.c_double,
                               ctypes.c_double,
                               ctypes.c_double,
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.int32,flags=ndarrayFlags),
                               ctypes.c_int]

    #Array requirements, first store old order
    f_cont= [yo.flags['F_CONTIGUOUS'],
             t.flags['F_CONTIGUOUS']]
    yo= numpy.require(yo,dtype=numpy.float64,requirements=['C','W'])
    t= numpy.require(t,dtype=numpy.float64,requirements=['C','W'])
    result= numpy.require(result,dtype=numpy.float64,requirements=['C','W'])
    err= numpy.require(err,dtype=numpy.int32,requirements=['C','W'])

    #Run the C code
    integrationFunc(ctypes.c_int(nobj),
                    yo,
                    ctypes.c_int(len(t)),
                    t,
                    ctypes.c_int(npot),
                    pot_type,
                    pot_args,
                    ctypes.c_double(dt),
                    ctypes.c_double(rtol),
                    ctypes.c_double(atol),
                    result,
                    err,
                    ctypes.c_int(int_method_c))

    if numpy.any(err == -10): #pragma: no cover
        raise KeyboardInterrupt("Orbit integration interrupted by CTRL-C (SIGINT)")

    #Reset input arrays
    if f_cont[0]: yo= numpy.asfortranarray(yo)
    if f_cont[1]: t= numpy.asfortranarray(t)

    if single_obj: return (result[0],err[0])
    else: return (result,err)

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
    yo= numpy.concatenate((yo,dyo))

    #Set up result array
    result= numpy.empty((len(t),12))
    err= ctypes.c_int(0)

    #Set up the C code
    ndarrayFlags= ('C_CONTIGUOUS','WRITEABLE')
    integrationFunc= _lib.integrateFullOrbit_dxdv
    integrationFunc.argtypes= [ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ctypes.c_int,
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ctypes.c_int,
                               ndpointer(dtype=numpy.int32,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ctypes.c_double,
                               ctypes.c_double,
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ctypes.POINTER(ctypes.c_int),
                               ctypes.c_int]

    #Array requirements, first store old order
    f_cont= [yo.flags['F_CONTIGUOUS'],
             t.flags['F_CONTIGUOUS']]
    yo= numpy.require(yo,dtype=numpy.float64,requirements=['C','W'])
    t= numpy.require(t,dtype=numpy.float64,requirements=['C','W'])
    result= numpy.require(result,dtype=numpy.float64,requirements=['C','W'])

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
    if f_cont[0]: yo= numpy.asfortranarray(yo)
    if f_cont[1]: t= numpy.asfortranarray(t)

    return (result,err.value)

def integrateFullOrbit(pot,yo,t,int_method,rtol=None,atol=None,numcores=1,
                       dt=None):
    """
    NAME:
       integrateFullOrbit
    PURPOSE:
       Integrate an ode for a FullOrbit
    INPUT:
       pot - Potential or list of such instances
       yo - initial condition [q,p], shape [N,5] or [N,6]
       t - set of times at which one wants the result
       int_method= 'leapfrog', 'odeint', or 'dop853'
       rtol, atol= tolerances (not always used...)
       numcores= (1) number of cores to use for multi-processing
       dt= (None) force integrator to use this stepsize (default is to automatically determine one; only for C-based integrators)
    OUTPUT:
       (y,err)
       y : array, shape (N,len(t),5/6)
       Array containing the value of y for each desired time in t, \
       with the initial value y0 in the first row.
       err: error message, always zero for now
    HISTORY:
       2010-08-01 - Written - Bovy (NYU)
       2019-04-09 - Adapted to allow multiple objects and parallel mapping - Bovy (UofT)
    """
    nophi= False
    if not int_method.lower() == 'dop853' and not int_method == 'odeint':
        if len(yo[0]) == 5:
            nophi= True
            #We hack this by putting in a dummy phi=0
            yo= numpy.pad(yo,((0,0),(0,1)),'constant',constant_values=0)
    if int_method.lower() == 'leapfrog':
        if rtol is None: rtol= 1e-8
        def integrate_for_map(vxvv):
            #go to the rectangular frame
            this_vxvv= numpy.array([vxvv[0]*numpy.cos(vxvv[5]),
                                 vxvv[0]*numpy.sin(vxvv[5]),
                                 vxvv[3],
                                 vxvv[1]*numpy.cos(vxvv[5])
                                     -vxvv[2]*numpy.sin(vxvv[5]),
                                 vxvv[2]*numpy.cos(vxvv[5])
                                     +vxvv[1]*numpy.sin(vxvv[5]),
                                 vxvv[4]])
            #integrate
            out= symplecticode.leapfrog(_rectForce,this_vxvv,
                                        t,args=(pot,),rtol=rtol)
            #go back to the cylindrical frame
            R= numpy.sqrt(out[:,0]**2.+out[:,1]**2.)
            phi= numpy.arccos(out[:,0]/R)
            phi[(out[:,1] < 0.)]= 2.*numpy.pi-phi[(out[:,1] < 0.)]
            vR= out[:,3]*numpy.cos(phi)+out[:,4]*numpy.sin(phi)
            vT= out[:,4]*numpy.cos(phi)-out[:,3]*numpy.sin(phi)
            out[:,3]= out[:,2]
            out[:,4]= out[:,5]
            out[:,0]= R
            out[:,1]= vR
            out[:,2]= vT
            out[:,5]= phi
            return out
    elif int_method.lower() == 'dop853' or int_method.lower() == 'odeint':
        if rtol is None: rtol= 1e-8
        if int_method.lower() == 'dop853':
            integrator= dop853
            extra_kwargs= {}
        else:
            integrator= integrate.odeint
            extra_kwargs= {'rtol':rtol}
        if len(yo[0]) == 5:
            def integrate_for_map(vxvv):
                l= vxvv[0]*vxvv[2]
                l2= l**2.
                init= [vxvv[0],vxvv[1],vxvv[3],vxvv[4]]
                intOut= integrator(_RZEOM,init,t=t,args=(pot,l2),
                                   **extra_kwargs)
                out= numpy.zeros((len(t),5))
                out[:,0]= intOut[:,0]
                out[:,1]= intOut[:,1]
                out[:,3]= intOut[:,2]
                out[:,4]= intOut[:,3]
                out[:,2]= l/out[:,0]
                #post-process to remove negative radii
                neg_radii= (out[:,0] < 0.)
                out[neg_radii,0]= -out[neg_radii,0]
                return out
        else:
            def integrate_for_map(vxvv):
                vphi= vxvv[2]/vxvv[0]
                init= [vxvv[0],vxvv[1],vxvv[5],vphi,vxvv[3],vxvv[4]]
                intOut= integrator(_EOM,init,t=t,args=(pot,))
                out= numpy.zeros((len(t),6))
                out[:,0]= intOut[:,0]
                out[:,1]= intOut[:,1]
                out[:,2]= out[:,0]*intOut[:,3]
                out[:,3]= intOut[:,4]
                out[:,4]= intOut[:,5]
                out[:,5]= intOut[:,2]
                #post-process to remove negative radii
                neg_radii= (out[:,0] < 0.)
                out[neg_radii,0]= -out[neg_radii,0]
                out[neg_radii,3]+= numpy.pi
                return out
    else: # Assume we are forcing parallel_mapping of a C integrator...
        def integrate_for_map(vxvv):
            return integrateFullOrbit_c(pot,numpy.copy(vxvv),
                                        t,int_method,dt=dt)[0]
    if len(yo) == 1: # Can't map a single value...
        out= numpy.atleast_3d(integrate_for_map(yo[0]).T).T
    else:
        out= numpy.array((parallel_map(integrate_for_map,yo,numcores=numcores)))
    if nophi:
        out= out[:,:,:5]
    return out, numpy.zeros(len(yo))

def _RZEOM(y,t,pot,l2):
    """
    NAME:
       _RZEOM
    PURPOSE:
       implements the EOM, i.e., the right-hand side of the differential
       equation, for a 3D orbit assuming conservation of angular momentum
    INPUT:
       y - current phase-space position
       t - current time
       pot - (list of) Potential instance(s)
       l2 - angular momentum squared
    OUTPUT:
       dy/dt
    HISTORY:
       2010-04-16 - Written - Bovy (NYU)
    """
    return [y[1],
            l2/y[0]**3.+_evaluateRforces(pot,y[0],y[2],t=t),
            y[3],
            _evaluatezforces(pot,y[0],y[2],t=t)]

def _EOM(y,t,pot):
    """
    NAME:
       _EOM
    PURPOSE:
       implements the EOM, i.e., the right-hand side of the differential
       equation, for a 3D orbit
    INPUT:
       y - current phase-space position
       t - current time
       pot - (list of) Potential instance(s)
    OUTPUT:
       dy/dt
    HISTORY:
       2010-04-16 - Written - Bovy (NYU)
    """
    l2= (y[0]**2.*y[3])**2.
    return [y[1],
            l2/y[0]**3.+_evaluateRforces(pot,y[0],y[4],phi=y[2],t=t,
                                         v=[y[1],y[0]*y[3],y[5]]),
            y[3],
            1./y[0]**2.*(_evaluatephiforces(pot,y[0],y[4],phi=y[2],t=t,
                                            v=[y[1],y[0]*y[3],y[5]])
                         -2.*y[0]*y[1]*y[3]),
            y[5],
            _evaluatezforces(pot,y[0],y[4],phi=y[2],t=t,
                             v=[y[1],y[0]*y[3],y[5]])]

def _rectForce(x,pot,t=0.):
    """
    NAME:
       _rectForce
    PURPOSE:
       returns the force in the rectangular frame
    INPUT:
       x - current position
       t - current time
       pot - (list of) Potential instance(s)
    OUTPUT:
       force
    HISTORY:
       2011-02-02 - Written - Bovy (NYU)
    """
    #x is rectangular so calculate R and phi
    R= numpy.sqrt(x[0]**2.+x[1]**2.)
    phi= numpy.arccos(x[0]/R)
    sinphi= x[1]/R
    cosphi= x[0]/R
    if x[1] < 0.: phi= 2.*numpy.pi-phi
    #calculate forces
    Rforce= _evaluateRforces(pot,R,x[2],phi=phi,t=t)
    phiforce= _evaluatephiforces(pot,R,x[2],phi=phi,t=t)
    return numpy.array([cosphi*Rforce-1./R*sinphi*phiforce,
                     sinphi*Rforce+1./R*cosphi*phiforce,
                     _evaluatezforces(pot,R,x[2],phi=phi,t=t)])
