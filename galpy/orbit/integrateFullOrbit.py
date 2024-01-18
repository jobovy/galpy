import ctypes
import ctypes.util
import warnings

import numpy
from numpy.ctypeslib import ndpointer
from scipy import integrate

from .. import potential
from ..potential.Potential import (
    _evaluatephitorques,
    _evaluateRforces,
    _evaluatezforces,
)
from ..util import _load_extension_libs, galpyWarning, symplecticode
from ..util._optional_deps import _TQDM_LOADED
from ..util.leung_dop853 import dop853
from ..util.multi import parallel_map
from .integratePlanarOrbit import (
    _parse_integrator,
    _parse_scf_pot,
    _parse_tol,
    _prep_tfuncs,
)

if _TQDM_LOADED:
    import tqdm

_lib, _ext_loaded = _load_extension_libs.load_libgalpy()


def _parse_pot(pot, potforactions=False, potfortorus=False):
    """Parse the potential so it can be fed to C"""
    # Figure out what's in pot
    if not isinstance(pot, list):
        pot = [pot]
    if (potforactions or potfortorus) and (
        (len(pot) == 1 and isinstance(pot[0], potential.NullPotential))
        or numpy.all([isinstance(p, potential.NullPotential) for p in pot])
    ):
        raise NotImplementedError(
            "Evaluating actions using the C backend is not supported for NullPotential instances"
        )
    # Remove NullPotentials from list of Potentials containing other potentials
    purged_pot = [p for p in pot if not isinstance(p, potential.NullPotential)]
    if len(purged_pot) > 0:
        pot = purged_pot
    # Initialize everything
    pot_type = []
    pot_args = []
    pot_tfuncs = []
    npot = len(pot)
    for p in pot:
        if isinstance(p, potential.LogarithmicHaloPotential):
            pot_type.append(0)
            if p.isNonAxi:
                pot_args.extend([p._amp, p._q, p._core2, p._1m1overb2])
            else:
                pot_args.extend([p._amp, p._q, p._core2, 2.0])  # 1m1overb2 > 1: axi
        elif isinstance(p, potential.DehnenBarPotential):
            pot_type.append(1)
            pot_args.extend(
                [p._amp * p._af, p._tform, p._tsteady, p._rb, p._omegab, p._barphi]
            )
        elif isinstance(p, potential.MiyamotoNagaiPotential):
            pot_type.append(5)
            pot_args.extend([p._amp, p._a, p._b])
        elif isinstance(p, potential.PowerSphericalPotential):
            pot_type.append(7)
            pot_args.extend([p._amp, p.alpha])
        elif isinstance(p, potential.HernquistPotential):
            pot_type.append(8)
            pot_args.extend([p._amp, p.a])
        elif isinstance(p, potential.NFWPotential):
            pot_type.append(9)
            pot_args.extend([p._amp, p.a])
        elif isinstance(p, potential.JaffePotential):
            pot_type.append(10)
            pot_args.extend([p._amp, p.a])
        elif isinstance(p, potential.DoubleExponentialDiskPotential):
            pot_type.append(11)
            pot_args.extend(
                [
                    p._amp,
                    -4.0 * numpy.pi * p._alpha * p._amp,
                    p._alpha,
                    p._beta,
                    len(p._de_j1_xs),
                ]
            )
            pot_args.extend(p._de_j0_xs)
            pot_args.extend(p._de_j1_xs)
            pot_args.extend(p._de_j0_weights)
            pot_args.extend(p._de_j1_weights)
        elif isinstance(p, potential.FlattenedPowerPotential):
            pot_type.append(12)
            pot_args.extend([p._amp, p.alpha, p.q2, p.core2])
        elif isinstance(p, potential.interpRZPotential):
            pot_type.append(13)
            pot_args.extend([len(p._rgrid), len(p._zgrid)])
            if p._logR:
                pot_args.extend([p._logrgrid[ii] for ii in range(len(p._rgrid))])
            else:
                pot_args.extend([p._rgrid[ii] for ii in range(len(p._rgrid))])
            pot_args.extend([p._zgrid[ii] for ii in range(len(p._zgrid))])
            if hasattr(p, "_potGrid_splinecoeffs"):
                pot_args.extend([x for x in p._potGrid_splinecoeffs.flatten(order="C")])
            else:  # pragma: no cover
                warnings.warn(
                    "You are attempting to use the C implementation of interpRZPotential, but have not interpolated the potential itself; if you think this is needed for what you want to do, initialize the interpRZPotential instance with interpPot=True",
                    galpyWarning,
                )
                pot_args.extend(list(numpy.ones(len(p._rgrid) * len(p._zgrid))))
            if hasattr(p, "_rforceGrid_splinecoeffs"):
                pot_args.extend(
                    [x for x in p._rforceGrid_splinecoeffs.flatten(order="C")]
                )
            else:  # pragma: no cover
                warnings.warn(
                    "You are attempting to use the C implementation of interpRZPotential, but have not interpolated the Rforce; if you think this is needed for what you want to do, initialize the interpRZPotential instance with interpRforce=True",
                    galpyWarning,
                )
                pot_args.extend(list(numpy.ones(len(p._rgrid) * len(p._zgrid))))
            if hasattr(p, "_zforceGrid_splinecoeffs"):
                pot_args.extend(
                    [x for x in p._zforceGrid_splinecoeffs.flatten(order="C")]
                )
            else:  # pragma: no cover
                warnings.warn(
                    "You are attempting to use the C implementation of interpRZPotential, but have not interpolated the zforce; if you think this is needed for what you want to do, initialize the interpRZPotential instance with interpzforce=True",
                    galpyWarning,
                )
                pot_args.extend(list(numpy.ones(len(p._rgrid) * len(p._zgrid))))
            pot_args.extend([p._amp, int(p._logR)])
        elif isinstance(p, potential.IsochronePotential):
            pot_type.append(14)
            pot_args.extend([p._amp, p.b])
        elif isinstance(p, potential.PowerSphericalPotentialwCutoff):
            pot_type.append(15)
            pot_args.extend([p._amp, p.alpha, p.rc])
        elif isinstance(p, potential.MN3ExponentialDiskPotential):
            # Three Miyamoto-Nagai disks
            npot += 2
            pot_type.extend([5, 5, 5])
            pot_args.extend(
                [
                    p._amp * p._mn3[0]._amp,
                    p._mn3[0]._a,
                    p._mn3[0]._b,
                    p._amp * p._mn3[1]._amp,
                    p._mn3[1]._a,
                    p._mn3[1]._b,
                    p._amp * p._mn3[2]._amp,
                    p._mn3[2]._a,
                    p._mn3[2]._b,
                ]
            )
        elif isinstance(p, potential.KuzminKutuzovStaeckelPotential):
            pot_type.append(16)
            pot_args.extend([p._amp, p._ac, p._Delta])
        elif isinstance(p, potential.PlummerPotential):
            pot_type.append(17)
            pot_args.extend([p._amp, p._b])
        elif isinstance(p, potential.PseudoIsothermalPotential):
            pot_type.append(18)
            pot_args.extend([p._amp, p._a])
        elif isinstance(p, potential.KuzminDiskPotential):
            pot_type.append(19)
            pot_args.extend([p._amp, p._a])
        elif isinstance(p, potential.BurkertPotential):
            pot_type.append(20)
            pot_args.extend([p._amp, p.a])
        elif isinstance(p, potential.EllipsoidalPotential.EllipsoidalPotential):
            pot_args.append(p._amp)
            pot_args.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # for caching
            # Potential specific parameters
            if isinstance(p, potential.TriaxialHernquistPotential):
                pot_type.append(21)
                pot_args.extend([2, p.a, p.a4])  # for psi, mdens, mdens_deriv
            elif isinstance(p, potential.TriaxialNFWPotential):
                pot_type.append(22)
                pot_args.extend([2, p.a, p.a3])  # for psi, mdens, mdens_deriv
            elif isinstance(p, potential.TriaxialJaffePotential):
                pot_type.append(23)
                pot_args.extend([2, p.a, p.a2])  # for psi, mdens, mdens_deriv
            elif isinstance(p, potential.PerfectEllipsoidPotential):
                pot_type.append(30)
                pot_args.extend([1, p.a2])  # for psi, mdens, mdens_deriv
            elif isinstance(p, potential.TriaxialGaussianPotential):
                pot_type.append(37)
                pot_args.extend([1, -p._twosigma2])  # for psi, mdens, mdens_deriv
            elif isinstance(p, potential.PowerTriaxialPotential):
                pot_type.append(38)
                pot_args.extend([1, p.alpha])  # for psi, mdens, mdens_deriv
            pot_args.extend([p._b2, p._c2, int(p._aligned)])  # Reg. Ellipsoidal
            if not p._aligned:
                pot_args.extend(list(p._rot.flatten()))
            else:
                pot_args.extend(list(numpy.eye(3).flatten()))  # not actually used
            pot_args.append(p._glorder)
            pot_args.extend([p._glx[ii] for ii in range(p._glorder)])
            # this adds some common factors to the integration weights
            pot_args.extend(
                [
                    -4.0
                    * numpy.pi
                    * p._glw[ii]
                    * p._b
                    * p._c
                    / numpy.sqrt(
                        (1.0 + (p._b2 - 1.0) * p._glx[ii] ** 2.0)
                        * (1.0 + (p._c2 - 1.0) * p._glx[ii] ** 2.0)
                    )
                    for ii in range(p._glorder)
                ]
            )
        elif isinstance(p, potential.SCFPotential):
            # Type 24, see stand-alone parser below
            pt, pa, ptf = _parse_scf_pot(p)
            pot_type.append(pt)
            pot_args.extend(pa)
            pot_tfuncs.extend(ptf)
        elif isinstance(p, potential.SoftenedNeedleBarPotential):
            pot_type.append(25)
            pot_args.extend([p._amp, p._a, p._b, p._c2, p._pa, p._omegab])
            pot_args.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # for caching
        elif isinstance(p, potential.DiskSCFPotential):
            # Need to pull this apart into: (a) SCF part, (b) constituent
            # [Sigma_i,h_i] parts
            # (a) SCF, multiply in any add'l amp
            pt, pa, ptf = _parse_scf_pot(p._scf, extra_amp=p._amp)
            pot_type.append(pt)
            pot_args.extend(pa)
            pot_tfuncs.extend(ptf)
            # (b) constituent [Sigma_i,h_i] parts
            for Sigma, hz in zip(p._Sigma_dict, p._hz_dict):
                npot += 1
                pot_type.append(26)
                stype = Sigma.get("type", "exp")
                if stype == "exp" and not "Rhole" in Sigma:
                    pot_args.extend(
                        [
                            3,
                            0,
                            4.0 * numpy.pi * Sigma.get("amp", 1.0) * p._amp,
                            Sigma.get("h", 1.0 / 3.0),
                        ]
                    )
                elif stype == "expwhole" or (stype == "exp" and "Rhole" in Sigma):
                    pot_args.extend(
                        [
                            4,
                            1,
                            4.0 * numpy.pi * Sigma.get("amp", 1.0) * p._amp,
                            Sigma.get("h", 1.0 / 3.0),
                            Sigma.get("Rhole", 0.5),
                        ]
                    )
                hztype = hz.get("type", "exp")
                if hztype == "exp":
                    pot_args.extend([0, hz.get("h", 0.0375)])
                elif hztype == "sech2":
                    pot_args.extend([1, hz.get("h", 0.0375)])
        elif isinstance(p, potential.SpiralArmsPotential):
            pot_type.append(27)
            pot_args.extend(
                [
                    len(p._Cs),
                    p._amp,
                    p._N,
                    p._sin_alpha,
                    p._tan_alpha,
                    p._r_ref,
                    p._phi_ref,
                    p._Rs,
                    p._H,
                    p._omega,
                ]
            )
            pot_args.extend(p._Cs)
        # 30: PerfectEllipsoidPotential, done with others above
        # 31: KGPotential
        # 32: IsothermalDiskPotential
        elif isinstance(p, potential.DehnenCoreSphericalPotential):
            pot_type.append(33)
            pot_args.extend([p._amp, p.a])
        elif isinstance(p, potential.DehnenSphericalPotential):
            pot_type.append(34)
            pot_args.extend([p._amp, p.a, p.alpha])
        elif isinstance(p, potential.HomogeneousSpherePotential):
            pot_type.append(35)
            pot_args.extend([p._amp, p._R2, p._R3])
        elif isinstance(p, potential.interpSphericalPotential):
            pot_type.append(36)
            pot_args.append(len(p._rgrid))
            pot_args.extend(p._rgrid)
            pot_args.extend(p._rforce_grid)
            pot_args.extend(
                [p._amp, p._rmin, p._rmax, p._total_mass, p._Phi0, p._Phimax]
            )
        # 37: TriaxialGaussianPotential, done with others above
        # 38: PowerTriaxialPotential, done with others above
        elif isinstance(p, potential.NonInertialFrameForce):
            pot_type.append(39)
            pot_args.append(p._amp)
            pot_args.extend(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            )  # for caching
            pot_args.extend(
                [
                    p._rot_acc,
                    p._lin_acc,
                    p._omegaz_only,
                    p._const_freq,
                    p._Omega_as_func,
                ]
            )
            if p._Omega_as_func:
                pot_args.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            else:
                if p._omegaz_only:
                    pot_args.extend([0.0, 0.0, p._Omega])
                else:
                    pot_args.extend(p._Omega)
                pot_args.append(p._Omega2)
                if not p._const_freq and p._omegaz_only:
                    pot_args.extend([0.0, 0.0, p._Omegadot])
                elif not p._const_freq:
                    pot_args.extend(p._Omegadot)
                else:
                    pot_args.extend([0.0, 0.0, 0.0])
            if p._lin_acc:
                pot_tfuncs.extend([p._a0[0], p._a0[1], p._a0[2]])
                if p._rot_acc:
                    pot_tfuncs.extend([p._x0[0], p._x0[1], p._x0[2]])
                    pot_tfuncs.extend([p._v0[0], p._v0[1], p._v0[2]])
            if p._Omega_as_func:
                if p._omegaz_only:
                    pot_tfuncs.extend([p._Omega, p._Omegadot])
                else:
                    pot_tfuncs.extend(
                        [
                            p._Omega[0],
                            p._Omega[1],
                            p._Omega[2],
                            p._Omegadot[0],
                            p._Omegadot[1],
                            p._Omegadot[2],
                        ]
                    )
        elif isinstance(p, potential.NullPotential):
            pot_type.append(40)
            # No arguments, zero forces
        ############################## WRAPPERS ###############################
        elif isinstance(p, potential.DehnenSmoothWrapperPotential):
            pot_type.append(-1)
            wrap_npot, wrap_pot_type, wrap_pot_args, wrap_pot_tfuncs = _parse_pot(
                p._pot, potforactions=potforactions, potfortorus=potfortorus
            )
            pot_args.append(wrap_npot)
            pot_type.extend(wrap_pot_type)
            pot_args.extend(wrap_pot_args)
            pot_tfuncs.extend(wrap_pot_tfuncs)
            pot_args.extend([p._amp, p._tform, p._tsteady, int(p._grow)])
        elif isinstance(p, potential.SolidBodyRotationWrapperPotential):
            pot_type.append(-2)
            # Not sure how to easily avoid this duplication
            wrap_npot, wrap_pot_type, wrap_pot_args, wrap_pot_tfuncs = _parse_pot(
                p._pot, potforactions=potforactions, potfortorus=potfortorus
            )
            pot_args.append(wrap_npot)
            pot_type.extend(wrap_pot_type)
            pot_args.extend(wrap_pot_args)
            pot_tfuncs.extend(wrap_pot_tfuncs)
            pot_args.extend([p._amp, p._omega, p._pa])
        elif isinstance(p, potential.CorotatingRotationWrapperPotential):
            pot_type.append(-4)
            # Not sure how to easily avoid this duplication
            wrap_npot, wrap_pot_type, wrap_pot_args, wrap_pot_tfuncs = _parse_pot(
                p._pot, potforactions=potforactions, potfortorus=potfortorus
            )
            pot_args.append(wrap_npot)
            pot_type.extend(wrap_pot_type)
            pot_args.extend(wrap_pot_args)
            pot_tfuncs.extend(wrap_pot_tfuncs)
            pot_args.extend([p._amp, p._vpo, p._beta, p._pa, p._to])
        elif isinstance(p, potential.GaussianAmplitudeWrapperPotential):
            pot_type.append(-5)
            wrap_npot, wrap_pot_type, wrap_pot_args, wrap_pot_tfuncs = _parse_pot(
                p._pot, potforactions=potforactions, potfortorus=potfortorus
            )
            pot_args.append(wrap_npot)
            pot_type.extend(wrap_pot_type)
            pot_args.extend(wrap_pot_args)
            pot_tfuncs.extend(wrap_pot_tfuncs)
            pot_args.extend([p._amp, p._to, p._sigma2])
        elif isinstance(p, potential.MovingObjectPotential):
            pot_type.append(-6)
            wrap_npot, wrap_pot_type, wrap_pot_args, wrap_pot_tfuncs = _parse_pot(
                p._pot, potforactions=potforactions, potfortorus=potfortorus
            )
            pot_args.append(wrap_npot)
            pot_type.extend(wrap_pot_type)
            pot_args.extend(wrap_pot_args)
            pot_tfuncs.extend(wrap_pot_tfuncs)
            pot_args.extend([len(p._orb.t)])
            pot_args.extend(p._orb.t)
            pot_args.extend(p._orb.x(p._orb.t, use_physical=False))
            pot_args.extend(p._orb.y(p._orb.t, use_physical=False))
            pot_args.extend(p._orb.z(p._orb.t, use_physical=False))
            pot_args.extend([p._amp])
            pot_args.extend([p._orb.t[0], p._orb.t[-1]])  # t_0, t_f
        elif isinstance(p, potential.ChandrasekharDynamicalFrictionForce):
            pot_type.append(-7)
            wrap_npot, wrap_pot_type, wrap_pot_args, wrap_pot_tfuncs = _parse_pot(
                p._dens_pot, potforactions=potforactions, potfortorus=potfortorus
            )
            pot_args.append(wrap_npot)
            pot_type.extend(wrap_pot_type)
            pot_args.extend(wrap_pot_args)
            pot_tfuncs.extend(wrap_pot_tfuncs)
            pot_args.extend([len(p._sigmar_rs_4interp)])
            pot_args.extend(p._sigmar_rs_4interp)
            pot_args.extend(p._sigmars_4interp)
            pot_args.extend([p._amp])
            pot_args.extend([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # for caching
            pot_args.extend(
                [
                    p._ms,
                    p._rhm,
                    p._gamma**2.0,
                    -1 if not p._lnLambda else p._lnLambda,
                    p._minr**2.0,
                ]
            )
            pot_args.extend(
                [p._sigmar_rs_4interp[0], p._sigmar_rs_4interp[-1]]
            )  # r_0, r_f
        elif isinstance(p, potential.RotateAndTiltWrapperPotential):
            pot_type.append(-8)
            # Not sure how to easily avoid this duplication
            wrap_npot, wrap_pot_type, wrap_pot_args, wrap_pot_tfuncs = _parse_pot(
                p._pot, potforactions=potforactions, potfortorus=potfortorus
            )
            pot_args.append(wrap_npot)
            pot_type.extend(wrap_pot_type)
            pot_args.extend(wrap_pot_args)
            pot_tfuncs.extend(wrap_pot_tfuncs)
            pot_args.extend([p._amp])
            pot_args.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # for caching
            pot_args.extend(list(p._rot.flatten()))
            pot_args.append(not p._norot)
            pot_args.append(not p._offset is None)
            pot_args.extend(
                list(p._offset) if not p._offset is None else [0.0, 0.0, 0.0]
            )
        elif isinstance(p, potential.TimeDependentAmplitudeWrapperPotential):
            pot_type.append(-9)
            # Not sure how to easily avoid this duplication
            wrap_npot, wrap_pot_type, wrap_pot_args, wrap_pot_tfuncs = _parse_pot(
                p._pot, potforactions=potforactions, potfortorus=potfortorus
            )
            pot_args.append(wrap_npot)
            pot_type.extend(wrap_pot_type)
            pot_args.extend(wrap_pot_args)
            pot_tfuncs.extend(wrap_pot_tfuncs)
            pot_args.append(p._amp)
            pot_tfuncs.append(p._A)
        elif isinstance(p, potential.KuzminLikeWrapperPotential):
            pot_type.append(-10)
            # Not sure how to easily avoid this duplication
            wrap_npot, wrap_pot_type, wrap_pot_args, wrap_pot_tfuncs = _parse_pot(
                p._pot, potforactions=potforactions, potfortorus=potfortorus
            )
            pot_args.append(wrap_npot)
            pot_type.extend(wrap_pot_type)
            pot_args.extend(wrap_pot_args)
            pot_tfuncs.extend(wrap_pot_tfuncs)
            pot_args.extend([p._amp, p._a, p._b2])
    pot_type = numpy.array(pot_type, dtype=numpy.int32, order="C")
    pot_args = numpy.array(pot_args, dtype=numpy.float64, order="C")
    return (npot, pot_type, pot_args, pot_tfuncs)


def integrateFullOrbit_c(
    pot, yo, t, int_method, rtol=None, atol=None, progressbar=True, dt=None
):
    """
    Integrate an ode for a FullOrbit.

    Parameters
    ----------
    pot : Potential or list of such instances
        The potential (or list thereof) to evaluate the orbit in.
    yo : numpy.ndarray
        Initial condition [q,p], can be [N,6] or [6].
    t : numpy.ndarray
        Set of times at which one wants the result.
    int_method : str
        Integration method. One of 'leapfrog_c', 'rk4_c', 'rk6_c', 'symplec4_c'.
    rtol : float, optional
        Relative tolerance.
    atol : float, optional
        Absolute tolerance.
    progressbar : bool, optional
        If True, display a tqdm progress bar when integrating multiple orbits (requires tqdm to be installed!).
    dt : float, optional
        Force integrator to use this stepsize (default is to automatically determine one; only for C-based integrators).

    Returns
    -------
    tuple
        (y, err)
        y : array, shape (N,len(t),6)  or (len(t),6) if N = 1
            Array containing the value of y for each desired time in t, with the initial value y0 in the first row.
        err : int or array of ints
            Error message, if not zero: 1 means maximum step reduction happened for adaptive integrators.

    Notes
    -----
    - 2011-11-13 - Written - Bovy (IAS)
    - 2018-12-21 - Adapted to allow multiple objects - Bovy (UofT)
    - 2022-04-12 - Add progressbar - Bovy (UofT)
    """
    if len(yo.shape) == 1:
        single_obj = True
    else:
        single_obj = False
    yo = numpy.atleast_2d(yo)
    nobj = len(yo)
    rtol, atol = _parse_tol(rtol, atol)
    npot, pot_type, pot_args, pot_tfuncs = _parse_pot(pot)
    pot_tfuncs = _prep_tfuncs(pot_tfuncs)
    int_method_c = _parse_integrator(int_method)
    if dt is None:
        dt = -9999.99

    # Set up result array
    result = numpy.empty((nobj, len(t), 6))
    err = numpy.zeros(nobj, dtype=numpy.int32)

    # Set up progressbar
    progressbar *= _TQDM_LOADED
    if nobj > 1 and progressbar:
        pbar = tqdm.tqdm(total=nobj, leave=False)
        pbar_func_ctype = ctypes.CFUNCTYPE(None)
        pbar_c = pbar_func_ctype(pbar.update)
    else:  # pragma: no cover
        pbar_c = None

    # Set up the C code
    ndarrayFlags = ("C_CONTIGUOUS", "WRITEABLE")
    integrationFunc = _lib.integrateFullOrbit
    integrationFunc.argtypes = [
        ctypes.c_int,
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_int,
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_int,
        ndpointer(dtype=numpy.int32, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.int32, flags=ndarrayFlags),
        ctypes.c_int,
        ctypes.c_void_p,
    ]

    # Array requirements, first store old order
    f_cont = [yo.flags["F_CONTIGUOUS"], t.flags["F_CONTIGUOUS"]]
    yo = numpy.require(yo, dtype=numpy.float64, requirements=["C", "W"])
    t = numpy.require(t, dtype=numpy.float64, requirements=["C", "W"])
    result = numpy.require(result, dtype=numpy.float64, requirements=["C", "W"])
    err = numpy.require(err, dtype=numpy.int32, requirements=["C", "W"])

    # Run the C code
    integrationFunc(
        ctypes.c_int(nobj),
        yo,
        ctypes.c_int(len(t)),
        t,
        ctypes.c_int(npot),
        pot_type,
        pot_args,
        pot_tfuncs,
        ctypes.c_double(dt),
        ctypes.c_double(rtol),
        ctypes.c_double(atol),
        result,
        err,
        ctypes.c_int(int_method_c),
        pbar_c,
    )

    if nobj > 1 and progressbar:
        pbar.close()

    if numpy.any(err == -10):  # pragma: no cover
        raise KeyboardInterrupt("Orbit integration interrupted by CTRL-C (SIGINT)")

    # Reset input arrays
    if f_cont[0]:
        yo = numpy.asfortranarray(yo)
    if f_cont[1]:
        t = numpy.asfortranarray(t)

    if single_obj:
        return (result[0], err[0])
    else:
        return (result, err)


def integrateFullOrbit_dxdv_c(
    pot, yo, dyo, t, int_method, rtol=None, atol=None
):  # pragma: no cover because not included in v1, uncover when included
    """
    Integrate an ode for a planarOrbit+phase space volume dxdv.

    Parameters
    ----------
    pot : Potential or list of such instances
        The potential (or list thereof) to evaluate the orbit in.
    yo : numpy.ndarray
        Initial condition [q,p].
    dyo : numpy.ndarray
        Initial condition [dq,dp].
    t : numpy.ndarray
        Set of times at which one wants the result.
    int_method : str
        Integration method. One of 'leapfrog_c', 'rk4_c', 'rk6_c', 'symplec4_c'.
    rtol : float, optional
        Relative tolerance.
    atol : float, optional
        Absolute tolerance.

    Returns
    -------
    tuple
        (y,err)
        y : array, shape (len(y0), len(t))
            Array containing the value of y for each desired time in t, with the initial value y0 in the first row.
        err : int
            Error message if not zero, 1: maximum step reduction happened for adaptive integrators.

    Notes
    -----
    - 2011-11-13 - Written - Bovy (IAS)
    """
    rtol, atol = _parse_tol(rtol, atol)
    npot, pot_type, pot_args, pot_tfuncs = _parse_pot(pot)
    pot_tfuncs = _prep_tfuncs(pot_tfuncs)
    int_method_c = _parse_integrator(int_method)
    yo = numpy.concatenate((yo, dyo))

    # Set up result array
    result = numpy.empty((len(t), 12))
    err = ctypes.c_int(0)

    # Set up the C code
    ndarrayFlags = ("C_CONTIGUOUS", "WRITEABLE")
    integrationFunc = _lib.integrateFullOrbit_dxdv
    integrationFunc.argtypes = [
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_int,
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_int,
        ndpointer(dtype=numpy.int32, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.c_double,
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]

    # Array requirements, first store old order
    f_cont = [yo.flags["F_CONTIGUOUS"], t.flags["F_CONTIGUOUS"]]
    yo = numpy.require(yo, dtype=numpy.float64, requirements=["C", "W"])
    t = numpy.require(t, dtype=numpy.float64, requirements=["C", "W"])
    result = numpy.require(result, dtype=numpy.float64, requirements=["C", "W"])

    # Run the C code
    integrationFunc(
        yo,
        ctypes.c_int(len(t)),
        t,
        ctypes.c_int(npot),
        pot_type,
        pot_args,
        pot_tfuncs,
        ctypes.c_double(rtol),
        ctypes.c_double(atol),
        result,
        ctypes.byref(err),
        ctypes.c_int(int_method_c),
    )

    if int(err.value) == -10:  # pragma: no cover
        raise KeyboardInterrupt("Orbit integration interrupted by CTRL-C (SIGINT)")

    # Reset input arrays
    if f_cont[0]:
        yo = numpy.asfortranarray(yo)
    if f_cont[1]:
        t = numpy.asfortranarray(t)

    return (result, err.value)


def integrateFullOrbit(
    pot, yo, t, int_method, rtol=None, atol=None, numcores=1, progressbar=True, dt=None
):
    """
    Integrate an ode for a FullOrbit

    Parameters
    ----------
    pot : Potential or list of such instances
        The potential (or list thereof) to evaluate the orbit in.
    yo : numpy.ndarray
        Initial condition [q,p], shape [N,5] or [N,6]
    t : numpy.ndarray
        Set of times at which one wants the result.
    int_method : str
        Integration method. One of 'leapfrog', 'odeint', 'dop853'.
    rtol : float, optional
        Relative tolerance.
    atol : float, optional
        Absolute tolerance.
    numcores : int, optional
        Number of cores to use for multi-processing.
    progressbar : bool, optional
        If True, display a tqdm progress bar when integrating multiple orbits (requires tqdm to be installed!).
    dt : float, optional
        Force integrator to use this stepsize (default is to automatically determine one; only for C-based integrators).

    Returns
    -------
    tuple
        (y,err)
        y : array, shape (N,len(t),5/6)
            Array containing the value of y for each desired time in t, with the initial value y0 in the first row.
        err : int or array of ints
            Error message, if not zero: 1 means maximum step reduction happened for adaptive integrators.

    Notes
    -----
    - 2010-08-01 - Written - Bovy (NYU)
    - 2019-04-09 - Adapted to allow multiple objects and parallel mapping - Bovy (UofT)
    - 2022-04-12 - Add progressbar - Bovy (UofT)
    """
    nophi = False
    if not int_method.lower() == "dop853" and not int_method == "odeint":
        if len(yo[0]) == 5:
            nophi = True
            # We hack this by putting in a dummy phi=0
            yo = numpy.pad(yo, ((0, 0), (0, 1)), "constant", constant_values=0)
    if int_method.lower() == "leapfrog":
        if rtol is None:
            rtol = 1e-8

        def integrate_for_map(vxvv):
            # go to the rectangular frame
            this_vxvv = numpy.array(
                [
                    vxvv[0] * numpy.cos(vxvv[5]),
                    vxvv[0] * numpy.sin(vxvv[5]),
                    vxvv[3],
                    vxvv[1] * numpy.cos(vxvv[5]) - vxvv[2] * numpy.sin(vxvv[5]),
                    vxvv[2] * numpy.cos(vxvv[5]) + vxvv[1] * numpy.sin(vxvv[5]),
                    vxvv[4],
                ]
            )
            # integrate
            out = symplecticode.leapfrog(
                _rectForce, this_vxvv, t, args=(pot,), rtol=rtol
            )
            # go back to the cylindrical frame
            R = numpy.sqrt(out[:, 0] ** 2.0 + out[:, 1] ** 2.0)
            phi = numpy.arccos(out[:, 0] / R)
            phi[(out[:, 1] < 0.0)] = 2.0 * numpy.pi - phi[(out[:, 1] < 0.0)]
            vR = out[:, 3] * numpy.cos(phi) + out[:, 4] * numpy.sin(phi)
            vT = out[:, 4] * numpy.cos(phi) - out[:, 3] * numpy.sin(phi)
            out[:, 3] = out[:, 2]
            out[:, 4] = out[:, 5]
            out[:, 0] = R
            out[:, 1] = vR
            out[:, 2] = vT
            out[:, 5] = phi
            return out

    elif int_method.lower() == "dop853" or int_method.lower() == "odeint":
        if rtol is None:
            rtol = 1e-8
        if int_method.lower() == "dop853":
            integrator = dop853
            extra_kwargs = {}
        else:
            integrator = integrate.odeint
            extra_kwargs = {"rtol": rtol}
        if len(yo[0]) == 5:

            def integrate_for_map(vxvv):
                l = vxvv[0] * vxvv[2]
                l2 = l**2.0
                init = [vxvv[0], vxvv[1], vxvv[3], vxvv[4]]
                intOut = integrator(_RZEOM, init, t=t, args=(pot, l2), **extra_kwargs)
                out = numpy.zeros((len(t), 5))
                out[:, 0] = intOut[:, 0]
                out[:, 1] = intOut[:, 1]
                out[:, 3] = intOut[:, 2]
                out[:, 4] = intOut[:, 3]
                out[:, 2] = l / out[:, 0]
                # post-process to remove negative radii
                neg_radii = out[:, 0] < 0.0
                out[neg_radii, 0] = -out[neg_radii, 0]
                return out

        else:

            def integrate_for_map(vxvv):
                vphi = vxvv[2] / vxvv[0]
                init = [vxvv[0], vxvv[1], vxvv[5], vphi, vxvv[3], vxvv[4]]
                intOut = integrator(_EOM, init, t=t, args=(pot,))
                out = numpy.zeros((len(t), 6))
                out[:, 0] = intOut[:, 0]
                out[:, 1] = intOut[:, 1]
                out[:, 2] = out[:, 0] * intOut[:, 3]
                out[:, 3] = intOut[:, 4]
                out[:, 4] = intOut[:, 5]
                out[:, 5] = intOut[:, 2]
                # post-process to remove negative radii
                neg_radii = out[:, 0] < 0.0
                out[neg_radii, 0] = -out[neg_radii, 0]
                out[neg_radii, 3] += numpy.pi
                return out

    else:  # Assume we are forcing parallel_mapping of a C integrator...

        def integrate_for_map(vxvv):
            return integrateFullOrbit_c(pot, numpy.copy(vxvv), t, int_method, dt=dt)[0]

    if len(yo) == 1:  # Can't map a single value...
        out = numpy.atleast_3d(integrate_for_map(yo[0]).T).T
    else:
        out = numpy.array(
            parallel_map(
                integrate_for_map, yo, numcores=numcores, progressbar=progressbar
            )
        )
    if nophi:
        out = out[:, :, :5]
    return out, numpy.zeros(len(yo))


def integrateFullOrbit_sos_c(
    pot, yo, psi, t0, int_method, rtol=None, atol=None, progressbar=True, dpsi=None
):
    """
    Integrate an ode for a FullOrbit for integrate_sos in C

    Parameters
    ----------
    pot : Potential or list of such instances
        The potential (or list thereof) to evaluate the orbit in.
    yo : numpy.ndarray
        initial condition [q,p]
    psi : numpy.ndarray
        set of increment angles at which one wants the result [increments wrt initial angle]
    t0 : float or numpy.ndarray
        initial time
    int_method : str
        'rk4_c', 'rk6_c', 'dopr54_c', or 'dop853_c'
    rtol : float, optional
        tolerances (not always used...)
    atol : float, optional
        tolerances (not always used...)
    progressbar : bool, optional
        if True, display a tqdm progress bar when integrating multiple orbits (requires tqdm to be installed!)
    dpsi : float, optional
        force integrator to use this stepsize (default is to automatically determine one; only for C-based integrators)

    Returns
    -------
    tuple
        (y,err)
        y : array, shape (N,len(psi),7) where the last of the last dimension is the time
            Array containing the value of y for each desired angle in psi, \
            with the initial value y0 in the first row.
        err : int
            error message, always zero for now

    Notes
    -----
    - 2023-03-17 - Written based on integrateFullOrbit_c - Bovy (UofT)
    """
    if len(yo.shape) == 1:
        single_obj = True
    else:
        single_obj = False
    yo = numpy.atleast_2d(yo)
    nobj = len(yo)
    rtol, atol = _parse_tol(rtol, atol)
    npot, pot_type, pot_args, pot_tfuncs = _parse_pot(pot)
    pot_tfuncs = _prep_tfuncs(pot_tfuncs)
    int_method_c = _parse_integrator(int_method)
    if dpsi is None:
        dpsi = -9999.99
    t0 = numpy.atleast_1d(t0)
    yoo = numpy.empty((nobj, 7))
    yoo[:, :6] = yo[:, :6]
    if len(t0) == 1:
        yoo[:, 6] = t0[0]
    else:
        yoo[:, 6] = t0
    npsi = len(psi.T)  # .T to make npsi always the first dim

    # Set up result array
    result = numpy.empty((nobj, npsi, 7))
    err = numpy.zeros(nobj, dtype=numpy.int32)

    # Set up progressbar
    progressbar *= _TQDM_LOADED
    if nobj > 1 and progressbar:
        pbar = tqdm.tqdm(total=nobj, leave=False)
        pbar_func_ctype = ctypes.CFUNCTYPE(None)
        pbar_c = pbar_func_ctype(pbar.update)
    else:  # pragma: no cover
        pbar_c = None

    # Set up the C code
    ndarrayFlags = ("C_CONTIGUOUS", "WRITEABLE")
    integrationFunc = _lib.integrateFullOrbit_sos
    integrationFunc.argtypes = [
        ctypes.c_int,
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_int,
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_int,
        ctypes.c_int,
        ndpointer(dtype=numpy.int32, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.int32, flags=ndarrayFlags),
        ctypes.c_int,
        ctypes.c_void_p,
    ]

    # Array requirements, first store old order
    f_cont = [yoo.flags["F_CONTIGUOUS"], psi.flags["F_CONTIGUOUS"]]
    yoo = numpy.require(yoo, dtype=numpy.float64, requirements=["C", "W"])
    psi = numpy.require(psi, dtype=numpy.float64, requirements=["C", "W"])
    result = numpy.require(result, dtype=numpy.float64, requirements=["C", "W"])
    err = numpy.require(err, dtype=numpy.int32, requirements=["C", "W"])

    # Run the C code)
    integrationFunc(
        ctypes.c_int(nobj),
        yoo,
        ctypes.c_int(npsi),
        psi,
        ctypes.c_int(len(psi.shape) > 1),
        ctypes.c_int(npot),
        pot_type,
        pot_args,
        pot_tfuncs,
        ctypes.c_double(dpsi),
        ctypes.c_double(rtol),
        ctypes.c_double(atol),
        result,
        err,
        ctypes.c_int(int_method_c),
        pbar_c,
    )

    if nobj > 1 and progressbar:
        pbar.close()

    if numpy.any(err == -10):  # pragma: no cover
        raise KeyboardInterrupt("Orbit integration interrupted by CTRL-C (SIGINT)")

    # Reset input arrays
    if f_cont[0]:
        yoo = numpy.asfortranarray(yoo)
    if f_cont[1]:
        psi = numpy.asfortranarray(psi)

    if single_obj:
        return (result[0], err[0])
    else:
        return (result, err)


def integrateFullOrbit_sos(
    pot,
    yo,
    psi,
    t0,
    int_method,
    rtol=None,
    atol=None,
    numcores=1,
    progressbar=True,
    dpsi=None,
):
    """
    Integrate an ode for a FullOrbit for integrate_sos

    Parameters
    ----------
    pot : Potential or list of such instances
        The potential (or list thereof) to evaluate the orbit in.
    yo : numpy.ndarray
        Initial condition [q,p], shape [N,5] or [N,6]
    psi : numpy.ndarray
        Set of increment angles at which one wants the result [increments wrt initial angle]
    t0 : float or numpy.ndarray
        Initial time
    int_method : str
        Integration method. One of 'leapfrog', 'odeint', or 'dop853'
    rtol : float, optional
        Relative tolerance. Default is None.
    atol : float, optional
        Absolute tolerance. Default is None.
    numcores : int, optional
        Number of cores to use for multi-processing. Default is 1.
    progressbar : bool, optional
        If True, display a tqdm progress bar when integrating multiple orbits (requires tqdm to be installed!). Default is True.
    dpsi : float, optional
        Force integrator to use this stepsize (default is to automatically determine one; only for C-based integrators). Default is None.

    Returns
    -------
    tuple
        (y,err)
        y : array, shape (N,len(psi),6/7) where the last of the last dimension is the time
            Array containing the value of y for each desired angle in psi, with the initial value y0 in the first row.
        err : float
            Error message, always zero for now

    Notes
    -----
    - 2023-03-16 - Written based on integrateFullOrbit - Bovy (UofT)

    """
    nophi = False
    if len(yo[0]) == 5:
        nophi = True
        # We hack this by putting in a dummy phi=0
        yo = numpy.pad(yo, ((0, 0), (0, 1)), "constant", constant_values=0)
    if not "_c" in int_method:
        if rtol is None:
            rtol = 1e-8
        if int_method.lower() == "dop853":
            integrator = dop853
            extra_kwargs = {}
        else:
            integrator = integrate.odeint
            extra_kwargs = {"rtol": rtol}

        def integrate_for_map(vxvv, psi, t0):
            # go to the transformed plane: (x,vx,y,vy,A,t)
            init_psi = numpy.arctan2(vxvv[3], vxvv[4])
            init = numpy.array(
                [
                    vxvv[0] * numpy.cos(vxvv[5]),
                    vxvv[1] * numpy.cos(vxvv[5]) - vxvv[2] * numpy.sin(vxvv[5]),
                    vxvv[0] * numpy.sin(vxvv[5]),
                    vxvv[2] * numpy.cos(vxvv[5]) + vxvv[1] * numpy.sin(vxvv[5]),
                    numpy.sqrt(vxvv[3] ** 2.0 + vxvv[4] ** 2.0),
                    t0,
                ]
            )
            # integrate
            intOut = integrator(
                _SOSEOM, init, t=psi + init_psi, args=(pot,), **extra_kwargs
            )
            # go back to the cylindrical frame
            out = numpy.zeros((len(psi), 7))
            out[:, 0] = numpy.sqrt(intOut[:, 0] ** 2.0 + intOut[:, 2] ** 2.0)
            out[:, 5] = numpy.arctan2(intOut[:, 2], intOut[:, 0])
            out[:, 1] = intOut[:, 1] * numpy.cos(out[:, 5]) + intOut[:, 3] * numpy.sin(
                out[:, 5]
            )
            out[:, 2] = intOut[:, 3] * numpy.cos(out[:, 5]) - intOut[:, 1] * numpy.sin(
                out[:, 5]
            )
            out[:, 3] = intOut[:, 4] * numpy.sin(psi + init_psi)
            out[:, 4] = intOut[:, 4] * numpy.cos(psi + init_psi)
            out[:, 6] = intOut[:, 5]
            return out

    else:  # Assume we are forcing parallel_mapping of a C integrator...

        def integrate_for_map(vxvv, psi, t0):
            return integrateFullOrbit_sos_c(
                pot, numpy.copy(vxvv), psi, t0, int_method, dpsi=dpsi
            )[0]

    if len(yo) == 1:  # Can't map a single value...
        out = numpy.atleast_3d(integrate_for_map(yo[0], psi.flatten(), t0).T).T
    else:
        out = numpy.array(
            parallel_map(
                lambda ii: integrate_for_map(
                    yo[ii],
                    psi[ii] if len(psi.shape) > 1 else psi,
                    t0[0] if len(t0) == 1 else t0[ii],
                ),
                range(len(yo)),
                numcores=numcores,
                progressbar=progressbar,
            )
        )
    if nophi:
        phi_mask = numpy.ones(out.shape[2], dtype="bool")
        phi_mask[5] = False
        out = out[:, :, phi_mask]
    return out, numpy.zeros(len(yo))


def _RZEOM(y, t, pot, l2):
    """
    Implements the EOM, i.e., the right-hand side of the differential equation, for a 3D orbit assuming conservation of angular momentum.

    Parameters
    ----------
    y : list or numpy.ndarray
        Current phase-space position.
    t : float
        Current time.
    pot : list of Potential instance(s)
        Potential instance(s).
    l2 : float
        Angular momentum squared.

    Returns
    -------
    list
        Derivative of the phase-space position.

    Notes
    -----
    - 2010-04-16 - Written - Bovy (NYU).
    """
    return [
        y[1],
        l2 / y[0] ** 3.0 + _evaluateRforces(pot, y[0], y[2], t=t),
        y[3],
        _evaluatezforces(pot, y[0], y[2], t=t),
    ]


def _EOM(y, t, pot):
    """
    Implements the EOM, i.e., the right-hand side of the differential equation, for a 3D orbit.

    Parameters
    ----------
    y : list or numpy.ndarray
        Current phase-space position.
    t : float
        Current time.
    pot : list of Potential instance(s)
        Potential instance(s).

    Returns
    -------
    list
        Derivative of the phase-space position.

    Notes
    -----
    - 2010-04-16 - Written - Bovy (NYU)
    """
    l2 = (y[0] ** 2.0 * y[3]) ** 2.0
    return [
        y[1],
        l2 / y[0] ** 3.0
        + _evaluateRforces(pot, y[0], y[4], phi=y[2], t=t, v=[y[1], y[0] * y[3], y[5]]),
        y[3],
        1.0
        / y[0] ** 2.0
        * (
            _evaluatephitorques(
                pot, y[0], y[4], phi=y[2], t=t, v=[y[1], y[0] * y[3], y[5]]
            )
            - 2.0 * y[0] * y[1] * y[3]
        ),
        y[5],
        _evaluatezforces(pot, y[0], y[4], phi=y[2], t=t, v=[y[1], y[0] * y[3], y[5]]),
    ]


def _SOSEOM(y, psi, pot):
    """
    Implements the EOM, i.e., the right-hand side of the differential equation, for the SOS integration of a 3D orbit

    Parameters
    ----------
    y : numpy.ndarray
        Current phase-space position
    psi : float
        Current angle
    pot : list of Potential instance(s)
        Potential instance(s)

    Returns
    -------
    numpy.ndarray
        dy/dpsi

    Notes
    -----
    - 2023-03-16 - Written - Bovy (UofT)
    """

    # y = (x,vx,y,vy,A,t)
    # Calculate z, vz
    sp, cp = numpy.sin(psi), numpy.cos(psi)
    z = y[4] * sp
    gxyz = _rectForce([y[0], y[2], z], pot, t=y[5], vx=[y[1], y[3], y[4] * cp])
    psidot = cp**2.0 - sp / y[4] * gxyz[2]
    Adot = y[4] * cp * sp + gxyz[2] * cp
    return numpy.array([y[1], gxyz[0], y[3], gxyz[1], Adot, 1.0]) / psidot


def _rectForce(x, pot, t=0.0, vx=None):
    """
    Returns the force in the rectangular frame

    Parameters
    ----------
    x : numpy.ndarray
        Current position
    t : float, optional
        Current time (default is 0.0)
    pot : (list of) Potential instance(s)
        The potential (or list thereof) to evaluate the force for
    vx : numpy.ndarray, optional
        If set, use this [vx,vy,vz] when evaluating dissipative forces (default is None)

    Returns
    -------
    numpy.ndarray
        The force in the rectangular frame

    Notes
    -----
    - 2011-02-02 - Written - Bovy (NYU)
    """
    # x is rectangular so calculate R and phi
    R = numpy.sqrt(x[0] ** 2.0 + x[1] ** 2.0)
    phi = numpy.arccos(x[0] / R)
    sinphi = x[1] / R
    cosphi = x[0] / R
    if x[1] < 0.0:
        phi = 2.0 * numpy.pi - phi
    if not vx is None:
        vR = vx[0] * cosphi + vx[1] * sinphi
        vT = -vx[0] * sinphi + vx[1] * cosphi
        vx = [vR, vT, vx[2]]
    # calculate forces
    Rforce = _evaluateRforces(pot, R, x[2], phi=phi, t=t, v=vx)
    phitorque = _evaluatephitorques(pot, R, x[2], phi=phi, t=t, v=vx)
    return numpy.array(
        [
            cosphi * Rforce - 1.0 / R * sinphi * phitorque,
            sinphi * Rforce + 1.0 / R * cosphi * phitorque,
            _evaluatezforces(pot, R, x[2], phi=phi, t=t, v=vx),
        ]
    )
