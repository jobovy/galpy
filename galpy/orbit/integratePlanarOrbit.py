import ctypes
import ctypes.util
from numpy.ctypeslib import ndpointer
import numpy
from scipy import integrate
from .. import potential
from ..potential.planarPotential import planarPotentialFromFullPotential, \
    planarPotentialFromRZPotential
from ..potential.planarPotential import _evaluateplanarRforces,\
    _evaluateplanarphiforces, _evaluateplanarPotentials
from ..potential.WrapperPotential import parentWrapperPotential
from ..util.multi import parallel_map
from ..util.leung_dop853 import dop853
from ..util import symplecticode
from ..util import _load_extension_libs

_lib, _ext_loaded= _load_extension_libs.load_libgalpy()

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
            pot_args.extend([p._Pot._amp,
                             -4.*numpy.pi*p._Pot._alpha*p._Pot._amp,
                             p._Pot._alpha,p._Pot._beta,len(p._Pot._de_j1_xs)])
            pot_args.extend(p._Pot._de_j0_xs)
            pot_args.extend(p._Pot._de_j1_xs)
            pot_args.extend(p._Pot._de_j0_weights)
            pot_args.extend(p._Pot._de_j1_weights)
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
            elif isinstance(p._Pot,potential.TriaxialGaussianPotential):
                pot_type.append(37)
                pot_args.extend([1,-p._Pot._twosigma2]) # for psi, mdens, mdens_deriv
            elif isinstance(p._Pot,potential.PowerTriaxialPotential):
                pot_type.append(38)
                pot_args.extend([1,p._Pot.alpha]) # for psi, mdens, mdens_deriv
            pot_args.extend([p._Pot._b2,p._Pot._c2,
                             int(p._Pot._aligned)]) # Reg. Ellipsoidal
            if not p._Pot._aligned:
                pot_args.extend(list(p._Pot._rot.flatten()))
            else:
                pot_args.extend(list(numpy.eye(3).flatten())) # not actually used
            pot_args.append(p._Pot._glorder)
            pot_args.extend([p._Pot._glx[ii] for ii in range(p._Pot._glorder)])
            # this adds some common factors to the integration weights
            pot_args.extend([-4.*numpy.pi*p._Pot._glw[ii]*p._Pot._b*p._Pot._c\
                            /numpy.sqrt(( 1.+(p._Pot._b2-1.)*p._Pot._glx[ii]**2.)
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
                if stype == 'exp' and not 'Rhole' in Sigma:
                    pot_args.extend([3,0,
                                     4.*numpy.pi*Sigma.get('amp',1.)*p._Pot._amp,
                                     Sigma.get('h',1./3.)])
                elif stype == 'expwhole' \
                        or (stype == 'exp' and 'Rhole' in Sigma):
                    pot_args.extend([4,1,
                                     4.*numpy.pi*Sigma.get('amp',1.)*p._Pot._amp,
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
        # 31: KGPotential
        # 32: IsothermalDiskPotential
        elif isinstance(p, planarPotentialFromRZPotential) \
                and isinstance(p._Pot,potential.DehnenCoreSphericalPotential):
            pot_type.append(33)
            pot_args.extend([p._Pot._amp,p._Pot.a])
        elif isinstance(p, planarPotentialFromRZPotential) \
                and isinstance(p._Pot,potential.DehnenSphericalPotential):
            pot_type.append(34)
            pot_args.extend([p._Pot._amp,p._Pot.a,p._Pot.alpha])
        # 35: HomogeneousSpherePotential
        elif isinstance(p,planarPotentialFromRZPotential) \
             and isinstance(p._Pot,potential.HomogeneousSpherePotential):
            pot_type.append(35)
            pot_args.extend([p._Pot._amp,p._Pot._R2,p._Pot._R3])
        # 36: interpSphericalPotential
        elif isinstance(p,planarPotentialFromRZPotential) \
             and isinstance(p._Pot,potential.interpSphericalPotential):
            pot_type.append(36)
            pot_args.append(len(p._Pot._rgrid))
            pot_args.extend(p._Pot._rgrid)
            pot_args.extend(p._Pot._rforce_grid)
            pot_args.extend([p._Pot._amp,p._Pot._rmin,p._Pot._rmax,
                             p._Pot._total_mass,p._Pot._Phi0,p._Pot._Phimax])
        # 37: TriaxialGaussianPotential, done with other EllipsoidalPotentials above
        # 38: PowerTriaxialPotential, done with other EllipsoidalPotentials above
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
            pot_args.extend([p._amp,p._tform,p._tsteady,int(p._grow)])
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
        elif ((isinstance(p,planarPotentialFromFullPotential) or isinstance(p,planarPotentialFromRZPotential)) \
              and isinstance(p._Pot,potential.MovingObjectPotential)) \
              or isinstance(p,potential.MovingObjectPotential):
            if not isinstance(p,potential.MovingObjectPotential):
                p= p._Pot
            pot_type.append(-6)
            wrap_npot, wrap_pot_type, wrap_pot_args= \
                    _parse_pot(potential.toPlanarPotential(p._pot))
            pot_args.append(wrap_npot)
            pot_type.extend(wrap_pot_type)
            pot_args.extend(wrap_pot_args)
            pot_args.extend([len(p._orb.t)])
            pot_args.extend(p._orb.t)
            pot_args.extend(p._orb.x(p._orb.t,use_physical=False))
            pot_args.extend(p._orb.y(p._orb.t,use_physical=False))
            pot_args.extend([p._amp])
            pot_args.extend([p._orb.t[0],p._orb.t[-1]]) #t_0, t_f
        elif ((isinstance(p,planarPotentialFromFullPotential) or isinstance(p,planarPotentialFromRZPotential)) \
              and isinstance(p._Pot,potential.RotateAndTiltWrapperPotential)) \
              or isinstance(p,potential.RotateAndTiltWrapperPotential): # pragma: no cover
            raise NotImplementedError('Planar orbit integration in C for RotateAndTiltWrapperPotential not implemented; please integrate an orbit with (z,vz) = (0,0) instead')
            # Note that potential.RotateAndTiltWrapperPotential would be -8
    pot_type= numpy.array(pot_type,dtype=numpy.int32,order='C')
    pot_args= numpy.array(pot_args,dtype=numpy.float64,order='C')
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
    elif int_method.lower() == 'dop853_c':
        int_method_c= 6
    else:
        int_method_c= 0
    return int_method_c

def _parse_tol(rtol,atol):
    """Parse the tolerance keywords"""
    #Process atol and rtol
    if rtol is None:
        rtol= -12.*numpy.log(10.)
    else: #pragma: no cover
        rtol= numpy.log(rtol)
    if atol is None:
        atol= -12.*numpy.log(10.)
    else: #pragma: no cover
        atol= numpy.log(atol)
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
       yo - initial condition [q,p], can be [N,4] or [4]
       t - set of times at which one wants the result
       int_method= 'leapfrog_c', 'rk4_c', 'rk6_c', 'symplec4_c', ...
       rtol, atol
       dt= (None) force integrator to use this stepsize (default is to automatically determine one)
   OUTPUT:
       (y,err)
       y : array, shape (len(y0),len(t),4)
       Array containing the value of y for each desired time in t, \
       with the initial value y0 in the first row.
       err: error message, if not zero: 1 means maximum step reduction happened for adaptive integrators
    HISTORY:
       2011-10-03 - Written - Bovy (IAS)
       2018-12-20 - Adapted to allow multiple objects - Bovy (UofT)
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
    result= numpy.empty((nobj,len(t),4))
    err= numpy.zeros(nobj,dtype=numpy.int32)

    #Set up the C code
    ndarrayFlags= ('C_CONTIGUOUS','WRITEABLE')
    integrationFunc= _lib.integratePlanarOrbit
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
       y,dy : array, shape (len(y0),len(t),8)
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
    yo= numpy.concatenate((yo,dyo))

    #Set up result array
    result= numpy.empty((len(t),8))
    err= ctypes.c_int(0)

    #Set up the C code
    ndarrayFlags= ('C_CONTIGUOUS','WRITEABLE')
    integrationFunc= _lib.integratePlanarOrbit_dxdv
    integrationFunc.argtypes= [ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ctypes.c_int,
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ctypes.c_int,
                               ndpointer(dtype=numpy.int32,flags=ndarrayFlags),
                               ndpointer(dtype=numpy.float64,flags=ndarrayFlags),
                               ctypes.c_double,
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
                    ctypes.c_double(dt),
                    ctypes.c_double(rtol),ctypes.c_double(atol),
                    result,
                    ctypes.byref(err),
                    ctypes.c_int(int_method_c))

    if err.value == -10: #pragma: no cover
        raise KeyboardInterrupt("Orbit integration interrupted by CTRL-C (SIGINT)")

    #Reset input arrays
    if f_cont[0]: yo= numpy.asfortranarray(yo)
    if f_cont[1]: t= numpy.asfortranarray(t)

    return (result,err.value)

def integratePlanarOrbit(pot,yo,t,int_method,rtol=None,atol=None,numcores=1,
                         dt=None):
    """
    NAME:
       integratePlanarOrbit
    PURPOSE:
       Integrate an ode for a planarOrbit
    INPUT:
       pot - Potential or list of such instances
       yo - initial condition [q,p], shape [N,3] or [N,4]
       t - set of times at which one wants the result
       int_method= 'leapfrog', 'odeint', or 'dop853'
       rtol, atol= tolerances (not always used...)
       numcores= (1) number of cores to use for multi-processing
       dt= (None) force integrator to use this stepsize (default is to automatically determine one; only for C-based integrators!)
    OUTPUT:
       (y,err)
       y : array, shape (N,len(t),3/4)
       Array containing the value of y for each desired time in t, \
       with the initial value y0 in the first row.
       err: error message, always zero for now
    HISTORY:
       2010-07-20 - Written - Bovy (NYU)
       2019-04-09 - Adapted to allow multiple objects and parallel mapping - Bovy (UofT)
    """
    nophi= False
    if not int_method.lower() == 'dop853' and not int_method == 'odeint':
        if len(yo[0]) == 3:
            nophi= True
            #We hack this by putting in a dummy phi=0
            yo= numpy.pad(yo,((0,0),(0,1)),'constant',constant_values=0)
    if int_method.lower() == 'leapfrog':
        if rtol is None: rtol= 1e-8
        def integrate_for_map(vxvv):
            #go to the rectangular frame
            this_vxvv= numpy.array([vxvv[0]*numpy.cos(vxvv[3]),
                                 vxvv[0]*numpy.sin(vxvv[3]),
                                 vxvv[1]*numpy.cos(vxvv[3])
                                     -vxvv[2]*numpy.sin(vxvv[3]),
                                 vxvv[2]*numpy.cos(vxvv[3])
                                     +vxvv[1]*numpy.sin(vxvv[3])])
            #integrate
            tmp_out= symplecticode.leapfrog(_planarRectForce,this_vxvv,
                                            t,args=(pot,),rtol=rtol)
            #go back to the cylindrical frame
            R= numpy.sqrt(tmp_out[:,0]**2.+tmp_out[:,1]**2.)
            phi= numpy.arccos(tmp_out[:,0]/R)
            phi[(tmp_out[:,1] < 0.)]= 2.*numpy.pi-phi[(tmp_out[:,1] < 0.)]
            vR= tmp_out[:,2]*numpy.cos(phi)+tmp_out[:,3]*numpy.sin(phi)
            vT= tmp_out[:,3]*numpy.cos(phi)-tmp_out[:,2]*numpy.sin(phi)
            out= numpy.zeros((len(t),4))
            out[:,0]= R
            out[:,1]= vR
            out[:,2]= vT
            out[:,3]= phi
            return out
    elif int_method.lower() == 'dop853' or int_method.lower() == 'odeint':
        if rtol is None: rtol= 1e-8
        if int_method.lower() == 'dop853':
            integrator= dop853
            extra_kwargs= {}
        else:
            integrator= integrate.odeint
            extra_kwargs= {'rtol':rtol}
        if len(yo[0]) == 3:
            def integrate_for_map(vxvv):
                l= vxvv[0]*vxvv[2]
                l2= l**2.
                init= [vxvv[0],vxvv[1]]
                intOut= integrator(_planarREOM,init,t=t,args=(pot,l2),
                                   **extra_kwargs)
                out= numpy.zeros((len(t),3))
                out[:,0]= intOut[:,0]
                out[:,1]= intOut[:,1]
                out[:,2]= l/out[:,0]
                #post-process to remove negative radii
                neg_radii= (out[:,0] < 0.)
                out[neg_radii,0]= -out[neg_radii,0]
                return out
        else:
            def integrate_for_map(vxvv):
                vphi= vxvv[2]/vxvv[0]
                init= [vxvv[0],vxvv[1],vxvv[3],vphi]
                intOut= integrator(_planarEOM,init,t=t,args=(pot,),
                                   **extra_kwargs)
                out= numpy.zeros((len(t),4))
                out[:,0]= intOut[:,0]
                out[:,1]= intOut[:,1]
                out[:,3]= intOut[:,2]
                out[:,2]= out[:,0]*intOut[:,3]
                #post-process to remove negative radii
                neg_radii= (out[:,0] < 0.)
                out[neg_radii,0]= -out[neg_radii,0]
                out[neg_radii,3]+= numpy.pi
                return out
    else: # Assume we are forcing parallel_mapping of a C integrator...
        def integrate_for_map(vxvv):
            return integratePlanarOrbit_c(pot,numpy.copy(vxvv),
                                          t,int_method,dt=dt)[0]
    if len(yo) == 1: # Can't map a single value...
        out= numpy.atleast_3d(integrate_for_map(yo[0]).T).T
    else:
        out= numpy.array((parallel_map(integrate_for_map,yo,numcores=numcores)))
    if nophi:
        out= out[:,:,:3]
    return out, numpy.zeros(len(yo))

def integratePlanarOrbit_dxdv(pot,yo,dyo,t,int_method,
                              rectIn,rectOut,
                              rtol=None,atol=None,
                              dt=None,numcores=1):
    """
    NAME:
       integratePlanarOrbit_dxdv
    PURPOSE:
       Integrate an ode for a planarOrbit+phase space volume dxdv
    INPUT:
       pot - Potential or list of such instances
       yo - initial condition [q,p], shape [N,4]
       dyo - initial condition [dq,dp], shape [N,4]
       t - set of times at which one wants the result
       int_method= 'odeint', 'dop853', 'dopr54_c', 'rk4_c', 'rk6_c'
       rectIn= (False) if True, input dyo is in rectangular coordinates
       rectOut= (False) if True, output dyo is in rectangular coordinates
       rtol, atol= tolerances (not always used...)
       numcores= (1) number of cores to use for multi-processing
       dt= (None) force integrator to use this stepsize (default is to automatically determine one; only for C-based integrators)
    OUTPUT:
       (y,err)
       y : array, shape (N,len(t),8)
       Array containing the value of y for each desired time in t, \
       with the initial value y0 in the first row.
       err: error message, always zero for now
    HISTORY:
       2011-10-17 - Written - Bovy (IAS)
       2019-05-21 - Adapted to allow multiple objects and parallel mapping - Bovy (UofT)
    """
    #go to the rectangular frame
    this_yo= numpy.array([yo[:,0]*numpy.cos(yo[:,3]),
                         yo[:,0]*numpy.sin(yo[:,3]),
                         yo[:,1]*numpy.cos(yo[:,3])
                           -yo[:,2]*numpy.sin(yo[:,3]),
                         yo[:,2]*numpy.cos(yo[:,3])
                           +yo[:,1]*numpy.sin(yo[:,3])]).T
    if not rectIn:
        this_dyo= numpy.array([numpy.cos(yo[:,3])*dyo[:,0]
                              -yo[:,0]*numpy.sin(yo[:,3])*dyo[:,3],
                            numpy.sin(yo[:,3])*dyo[:,0]
                              +yo[:,0]*numpy.cos(yo[:,3])*dyo[:,3],
                            -(yo[:,1]*numpy.sin(yo[:,3])
                              +yo[:,2]*numpy.cos(yo[:,3]))*dyo[:,3]
                              +numpy.cos(yo[:,3])*dyo[:,1]
                              -numpy.sin(yo[:,3])*dyo[:,2],
                            (yo[:,1]*numpy.cos(yo[:,3])
                              -yo[:,2]*numpy.sin(yo[:,3]))*dyo[:,3]
                              +numpy.sin(yo[:,3])*dyo[:,1]
                              +numpy.cos(yo[:,3])*dyo[:,2]]).T
    else:
        this_dyo= dyo
    this_yo= numpy.hstack((this_yo,this_dyo))
    if int_method.lower() == 'dop853' or int_method.lower() == 'odeint':
        if rtol is None: rtol= 1e-8
        if int_method.lower() == 'dop853':
            integrator= dop853
            extra_kwargs= {}
        else:
            integrator= integrate.odeint
            extra_kwargs= {'rtol':rtol}
        def integrate_for_map(vxvv):
            return integrator(_planarEOM_dxdv,vxvv,t=t,args=(pot,),
                              **extra_kwargs)
    else: # Assume we are forcing parallel_mapping of a C integrator...
        def integrate_for_map(vxvv):
            return integratePlanarOrbit_dxdv_c(pot,numpy.copy(vxvv[:4]),
                                               numpy.copy(vxvv[4:]),
                                               t,int_method,dt=dt,
                                               rtol=rtol,atol=atol)[0]
    if len(this_yo) == 1: # Can't map a single value...
        out= numpy.atleast_3d(integrate_for_map(this_yo[0]).T).T
    else:
        out= numpy.array((parallel_map(integrate_for_map,this_yo,
                                    numcores=numcores)))
    #go back to the cylindrical frame
    R= numpy.sqrt(out[...,0]**2.+out[...,1]**2.)
    phi= numpy.arccos(out[...,0]/R)
    phi[(out[...,1] < 0.)]= 2.*numpy.pi-phi[(out[...,1] < 0.)]
    vR= out[...,2]*numpy.cos(phi)+out[...,3]*numpy.sin(phi)
    vT= out[...,3]*numpy.cos(phi)-out[...,2]*numpy.sin(phi)
    cp= numpy.cos(phi)
    sp= numpy.sin(phi)
    out[...,0]= R
    out[...,1]= vR
    out[...,2]= vT
    out[...,3]= phi
    if rectOut:
        out[...,4:]= out[...,4:]
    else:
        dR= cp*out[...,4]+sp*out[...,5]
        dphi= (cp*out[...,5]-sp*out[...,4])/R
        dvR= cp*out[...,6]+sp*out[...,7]+vT*dphi
        dvT= cp*out[...,7]-sp*out[...,6]-vR*dphi
        out[...,4]= dR
        out[...,7]= dphi
        out[...,5]= dvR
        out[...,6]= dvT
    return out, numpy.zeros(len(yo))

def _planarREOM(y,t,pot,l2):
    """
    NAME:
       _planarREOM
    PURPOSE:
       implements the EOM, i.e., the right-hand side of the differential
       equation, for integrating a planar Orbit assuming angular momentum
       conservation
    INPUT:
       y - current phase-space position
       t - current time
       pot - (list of) Potential instance(s)
       l2 - angular momentum squared
    OUTPUT:
       dy/dt
    HISTORY:
       2010-07-20 - Written - Bovy (NYU)
    """
    return [y[1],
            l2/y[0]**3.+_evaluateplanarRforces(pot,y[0],t=t)]

def _planarEOM(y,t,pot):
    """
    NAME:
       _planarEOM
    PURPOSE:
       implements the EOM, i.e., the right-hand side of the differential
       equation, for integrating a general planar Orbit
    INPUT:
       y - current phase-space position
       t - current time
       pot - (list of) Potential instance(s)
    OUTPUT:
       dy/dt
    HISTORY:
       2010-07-20 - Written - Bovy (NYU)
    """
    l2= (y[0]**2.*y[3])**2.
    return [y[1],
            l2/y[0]**3.+_evaluateplanarRforces(pot,y[0],phi=y[2],t=t),
            y[3],
            1./y[0]**2.*(_evaluateplanarphiforces(pot,y[0],phi=y[2],t=t)-
                         2.*y[0]*y[1]*y[3])]

def _planarEOM_dxdv(x,t,pot):
    """
    NAME:
       _planarEOM_dxdv
    PURPOSE:
       implements the EOM, i.e., the right-hand side of the differential
       equation, for integrating phase space differences, rectangular
    INPUT:
       x - current phase-space position
       t - current time
       pot - (list of) Potential instance(s)
    OUTPUT:
       dy/dt
    HISTORY:
       2011-10-18 - Written - Bovy (IAS)
    """
    #x is rectangular so calculate R and phi
    R= numpy.sqrt(x[0]**2.+x[1]**2.)
    phi= numpy.arccos(x[0]/R)
    sinphi= x[1]/R
    cosphi= x[0]/R
    if x[1] < 0.: phi= 2.*numpy.pi-phi
    #calculate forces
    Rforce= _evaluateplanarRforces(pot,R,phi=phi,t=t)
    phiforce= _evaluateplanarphiforces(pot,R,phi=phi,t=t)
    R2deriv= _evaluateplanarPotentials(pot,R,phi=phi,t=t,dR=2)
    phi2deriv= _evaluateplanarPotentials(pot,R,phi=phi,t=t,dphi=2)
    Rphideriv= _evaluateplanarPotentials(pot,R,phi=phi,t=t,dR=1,dphi=1)
    #Calculate derivatives and derivatives+time derivatives
    dFxdx= -cosphi**2.*R2deriv\
           +2.*cosphi*sinphi/R**2.*phiforce\
           +sinphi**2./R*Rforce\
           +2.*sinphi*cosphi/R*Rphideriv\
           -sinphi**2./R**2.*phi2deriv
    dFxdy= -sinphi*cosphi*R2deriv\
           +(sinphi**2.-cosphi**2.)/R**2.*phiforce\
           -cosphi*sinphi/R*Rforce\
           -(cosphi**2.-sinphi**2.)/R*Rphideriv\
           +cosphi*sinphi/R**2.*phi2deriv
    dFydx= -cosphi*sinphi*R2deriv\
           +(sinphi**2.-cosphi**2.)/R**2.*phiforce\
           +(sinphi**2.-cosphi**2.)/R*Rphideriv\
           -sinphi*cosphi/R*Rforce\
           +sinphi*cosphi/R**2.*phi2deriv
    dFydy= -sinphi**2.*R2deriv\
           -2.*sinphi*cosphi/R**2.*phiforce\
           -2.*sinphi*cosphi/R*Rphideriv\
           +cosphi**2./R*Rforce\
           -cosphi**2./R**2.*phi2deriv
    return numpy.array([x[2],x[3],
                     cosphi*Rforce-1./R*sinphi*phiforce,
                     sinphi*Rforce+1./R*cosphi*phiforce,
                     x[6],x[7],
                     dFxdx*x[4]+dFxdy*x[5],
                     dFydx*x[4]+dFydy*x[5]])

def _planarRectForce(x,pot,t=0.):
    """
    NAME:
       _planarRectForce
    PURPOSE:
       returns the planar force in the rectangular frame
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
    Rforce= _evaluateplanarRforces(pot,R,phi=phi,t=t)
    phiforce= _evaluateplanarphiforces(pot,R,phi=phi,t=t)
    return numpy.array([cosphi*Rforce-1./R*sinphi*phiforce,
                     sinphi*Rforce+1./R*cosphi*phiforce])
