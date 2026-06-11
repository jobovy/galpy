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
    evaluatephi2derivs,
    evaluatephizderivs,
    evaluateR2derivs,
    evaluateRphiderivs,
    evaluateRzderivs,
    evaluatez2derivs,
    potential_list_of_potentials_input,
)
from ..util import _load_extension_libs, coords, galpyWarning, symplecticode
from ..util._optional_deps import _TQDM_LOADED
from ..util.leung_dop853 import dop853
from ..util.multi import parallel_map
from .integratePlanarOrbit import (
    _finalize_pot_args,
    _parse_disk_approx_pairs,
    _parse_integrator,
    _parse_multipole_expansion_pot,
    _parse_noninertial_frame_force,
    _parse_scf_pot,
    _parse_tol,
    _prep_tfuncs,
)

if _TQDM_LOADED:
    import tqdm

_lib, _ext_loaded = _load_extension_libs.load_libgalpy()


def _parse_pot(pot, potforactions=False, potfortorus=False, t=None):
    """Parse the potential so it can be fed to C

    ``t`` is the integration time array (when available), used to build
    on-the-fly C spline interpolations for a NonInertialFrameForce with
    ``cinterp=True`` (see _parse_noninertial_frame_force).
    """
    # Remove NullPotentials from the potential (iterate directly without casting to list first)
    purged_pot = [p for p in pot if not isinstance(p, potential.NullPotential)]
    # Use purged_pot if it's not empty, otherwise use original
    if len(purged_pot) > 0:
        pot = purged_pot
    if (potforactions or potfortorus) and numpy.all(
        [isinstance(p, potential.NullPotential) for p in pot]
    ):
        raise NotImplementedError(
            "Evaluating actions using the C backend is not supported for NullPotential instances"
        )
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
            pot_args.extend([0.0, 0.0])  # for caching
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
            pot_args.extend(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            )  # for caching (x,y,z,Fx,Fy,Fz,x2,y2,z2,phixx,phixy,phixz,phiyy,phiyz,phizz)
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
            elif isinstance(p, potential.TwoPowerTriaxialPotential):
                pot_type.append(43)
                pot_args.extend(
                    [
                        7,
                        p.a,
                        p.alpha,
                        p.beta,
                        p.betaminusalpha,
                        p.twominusalpha,
                        p.threeminusalpha,
                        p.psi_inf if p.twominusalpha != 0.0 else 0.0,
                    ]
                )  # for psi, mdens, mdens_deriv
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
        elif isinstance(p, potential.MultipoleExpansionPotential):
            pot_type.append(44)
            _mep_args = p._serialize_for_c()
            if isinstance(_mep_args, numpy.ndarray):
                pot_args.append(_mep_args)
            else:
                pot_args.extend(_mep_args)
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
        elif isinstance(p, potential.DiskMultipoleExpansionPotential):
            # Need to pull this apart into: (a) MultipoleExpansion part,
            # (b) constituent [Sigma_i,h_i] parts
            # (a) MultipoleExpansion, multiply in any add'l amp
            pt, pa = _parse_multipole_expansion_pot(p._me, extra_amp=p._amp)
            pot_type.append(pt)
            if isinstance(pa, numpy.ndarray):
                pot_args.append(pa)
            else:
                pot_args.extend(pa)
            # (b) constituent [Sigma_i,h_i] parts
            dpts, dpa = _parse_disk_approx_pairs(p, extra_amp=p._amp)
            for dpt in dpts:
                npot += 1
                pot_type.append(dpt)
            pot_args.extend(dpa)
        elif isinstance(p, potential.DiskSCFPotential):
            # Need to pull this apart into: (a) SCF part, (b) constituent
            # [Sigma_i,h_i] parts
            # (a) SCF, multiply in any add'l amp
            pt, pa, ptf = _parse_scf_pot(p._me, extra_amp=p._amp)
            pot_type.append(pt)
            pot_args.extend(pa)
            pot_tfuncs.extend(ptf)
            # (b) constituent [Sigma_i,h_i] parts
            dpts, dpa = _parse_disk_approx_pairs(p, extra_amp=p._amp)
            for dpt in dpts:
                npot += 1
                pot_type.append(dpt)
            pot_args.extend(dpa)
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
            # pot_type 39 (functions called from C) or 45 (on-the-fly C spline
            # interpolation when cinterp=True); see _parse_noninertial_frame_force.
            _code, _nip_args, _nip_tfuncs = _parse_noninertial_frame_force(p, t)
            pot_type.append(_code)
            pot_args.extend(_nip_args)
            pot_tfuncs.extend(_nip_tfuncs)
        elif isinstance(p, potential.NullPotential):
            pot_type.append(40)
        elif isinstance(p, potential.EinastoPotential):
            pot_type.append(41)
            pot_args.extend([p._amp, p.h, p.n])
        elif isinstance(p, potential.TwoPowerSphericalPotential):
            pot_type.append(42)
            pot_args.extend([p._amp, p.a, p.alpha, p.beta])
            # No arguments, zero forces
        ############################## WRAPPERS ###############################
        elif isinstance(p, potential.DehnenSmoothWrapperPotential):
            pot_type.append(-1)
            wrap_npot, wrap_pot_type, wrap_pot_args, wrap_pot_tfuncs = _parse_pot(
                p._pot, potforactions=potforactions, potfortorus=potfortorus, t=t
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
                p._pot, potforactions=potforactions, potfortorus=potfortorus, t=t
            )
            pot_args.append(wrap_npot)
            pot_type.extend(wrap_pot_type)
            pot_args.extend(wrap_pot_args)
            pot_tfuncs.extend(wrap_pot_tfuncs)
            pot_args.extend([p._amp, p._omega, p._pa])
        elif isinstance(p, potential.OblateStaeckelWrapperPotential):
            pot_type.append(-3)
            # Not sure how to easily avoid this duplication
            wrap_npot, wrap_pot_type, wrap_pot_args, wrap_pot_tfuncs = _parse_pot(
                p._pot, potforactions=potforactions, potfortorus=potfortorus, t=t
            )
            pot_args.append(wrap_npot)
            pot_type.extend(wrap_pot_type)
            pot_args.extend(wrap_pot_args)
            pot_tfuncs.extend(wrap_pot_tfuncs)
            pot_args.extend([p._amp, p._delta, p._u0, p._v0, p._refpot])
        elif isinstance(p, potential.CorotatingRotationWrapperPotential):
            pot_type.append(-4)
            # Not sure how to easily avoid this duplication
            wrap_npot, wrap_pot_type, wrap_pot_args, wrap_pot_tfuncs = _parse_pot(
                p._pot, potforactions=potforactions, potfortorus=potfortorus, t=t
            )
            pot_args.append(wrap_npot)
            pot_type.extend(wrap_pot_type)
            pot_args.extend(wrap_pot_args)
            pot_tfuncs.extend(wrap_pot_tfuncs)
            pot_args.extend([p._amp, p._vpo, p._beta, p._pa, p._to])
        elif isinstance(p, potential.GaussianAmplitudeWrapperPotential):
            pot_type.append(-5)
            wrap_npot, wrap_pot_type, wrap_pot_args, wrap_pot_tfuncs = _parse_pot(
                p._pot, potforactions=potforactions, potfortorus=potfortorus, t=t
            )
            pot_args.append(wrap_npot)
            pot_type.extend(wrap_pot_type)
            pot_args.extend(wrap_pot_args)
            pot_tfuncs.extend(wrap_pot_tfuncs)
            pot_args.extend([p._amp, p._to, p._sigma2])
        elif isinstance(p, potential.MovingObjectPotential):
            pot_type.append(-6)
            wrap_npot, wrap_pot_type, wrap_pot_args, wrap_pot_tfuncs = _parse_pot(
                p._pot, potforactions=potforactions, potfortorus=potfortorus, t=t
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
        # Note that this one is out of pot_type order because it's closely associated
        # with ChandrasekharDynamicalFrictionForce and its a subclass, so needs to be
        # caught first
        elif isinstance(p, potential.FDMDynamicalFrictionForce):
            pot_type.append(-11)
            # Manually wrap the instance as a ChandrasekharDynamicalFrictionForce
            pot_args.append(1)
            pot_type.append(-7)  # Wrapping as ChandrasekharDynamicalFrictionForce
            wrap_npot, wrap_pot_type, wrap_pot_args, wrap_pot_tfuncs = _parse_pot(
                p._dens_pot, potforactions=potforactions, potfortorus=potfortorus, t=t
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
            # Now we wrap the FDM part, which repeats a lot of the above, because we need it separately in C
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
            pot_args.extend(
                [p._mhbar, -1 if not p._const_FDMfactor else p._const_FDMfactor]
            )  # Any additional FDM arguments must be added here, because r_0 and r_f need to stay in the same place for the C spline transfer to work
        elif isinstance(
            p, potential.ChandrasekharDynamicalFrictionForce
        ):  # not isinstance(p, potential.FDMDynamicalFrictionForce):
            pot_type.append(-7)
            wrap_npot, wrap_pot_type, wrap_pot_args, wrap_pot_tfuncs = _parse_pot(
                p._dens_pot, potforactions=potforactions, potfortorus=potfortorus, t=t
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
                p._pot, potforactions=potforactions, potfortorus=potfortorus, t=t
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
                p._pot, potforactions=potforactions, potfortorus=potfortorus, t=t
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
                p._pot, potforactions=potforactions, potfortorus=potfortorus, t=t
            )
            pot_args.append(wrap_npot)
            pot_type.extend(wrap_pot_type)
            pot_args.extend(wrap_pot_args)
            pot_tfuncs.extend(wrap_pot_tfuncs)
            pot_args.extend([p._amp, p._a, p._b2])
        elif isinstance(p, potential.CylindricallySeparablePotentialWrapper):
            pot_type.append(-12)
            # Not sure how to easily avoid this duplication
            wrap_npot, wrap_pot_type, wrap_pot_args, wrap_pot_tfuncs = _parse_pot(
                p._pot, potforactions=potforactions, potfortorus=potfortorus, t=t
            )
            pot_args.append(wrap_npot)
            pot_type.extend(wrap_pot_type)
            pot_args.extend(wrap_pot_args)
            pot_tfuncs.extend(wrap_pot_tfuncs)
            pot_args.extend([p._amp, p._Rp, p._refpot])
    pot_type = numpy.array(pot_type, dtype=numpy.int32, order="C")
    pot_args = _finalize_pot_args(pot_args)
    return (npot, pot_type, pot_args, pot_tfuncs)


def integrateFullOrbit_c(
    pot, yo, t, int_method, rtol=None, atol=None, progressbar=True, dt=None
):
    """
    Integrate an ode for a FullOrbit.

    Parameters
    ----------
    pot : Potential or a combined potential formed using addition (pot1+pot2+…)
        The potential to evaluate the orbit in.
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
    npot, pot_type, pot_args, pot_tfuncs = _parse_pot(pot, t=t)
    pot_tfuncs = _prep_tfuncs(pot_tfuncs)
    int_method_c = _parse_integrator(int_method)
    if dt is None:
        dt = -9999.99
    # t can be 1D (shared across orbits) or 2D (per-orbit, shape (nobj,nt))
    indiv_t = len(t.shape) > 1
    nt = t.shape[-1]

    # Set up result array
    result = numpy.empty((nobj, nt, 6))
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
        ctypes.c_int(nt),
        t,
        ctypes.c_int(indiv_t),
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
    pot, yo, dyo, t, int_method, dt=None, rtol=None, atol=None
):
    """
    Integrate an ode for a fullOrbit+phase space volume dxdv in C.

    Both the input state ``yo`` and deviation ``dyo`` as well as the output are
    in the rectangular frame (x,y,z,vx,vy,vz | dx,dy,dz,dvx,dvy,dvz); the
    cylindrical<->rectangular transforms are handled by the calling
    ``integrateFullOrbit_dxdv``.

    Parameters
    ----------
    pot : Potential or a combined potential formed using addition (pot1+pot2+…)
        The potential to evaluate the orbit in.
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
    if dt is None:
        dt = -9999.99
    npot, pot_type, pot_args, pot_tfuncs = _parse_pot(pot, t=t)
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
        ctypes.c_double(dt),
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


def integrateFullOrbit_dxdv(
    pot,
    yo,
    dyo,
    t,
    int_method,
    rectIn,
    rectOut,
    rtol=None,
    atol=None,
    progressbar=True,
    dt=None,
    numcores=1,
):
    """
    Integrate an ode for a fullOrbit+phase space volume dxdv.

    The 3D analog of ``integratePlanarOrbit_dxdv``: handles the
    cylindrical<->rectangular transform of both the base state and the
    phase-space deviation (via the chain rule, using the Jacobian of the
    cyl->rect transform), then integrates the 12D rectangular variational
    system in C (or, for the pure-Python ``dop853``/``odeint`` methods, via
    ``_EOM_dxdv``).

    Parameters
    ----------
    pot : Potential or a combined potential formed using addition (pot1+pot2+…)
    yo : numpy.ndarray
        Initial condition [q,p], shape [N,6] in cylindrical (R,vR,vT,z,vz,phi).
    dyo : numpy.ndarray
        Initial condition [dq,dp], shape [N,6]; cylindrical (dR,dvR,dvT,dz,dvz,dphi)
        unless ``rectIn`` (then rectangular dx,dy,dz,dvx,dvy,dvz).
    t : numpy.ndarray
        Set of times at which one wants the result.
    int_method : str
        Integration method. One of 'dopr54_c', 'dop853_c', 'rk4_c', 'rk6_c',
        'dop853', 'odeint'.
    rectIn : bool
        If True, input ``dyo`` is in rectangular coordinates.
    rectOut : bool
        If True, output deviation is in rectangular coordinates.
    rtol : float, optional
        Relative tolerance.
    atol : float, optional
        Absolute tolerance.
    progressbar : bool, optional
        If True, display a tqdm progress bar when integrating multiple orbits.
    dt : float, optional
        Force integrator to use this stepsize (default is to automatically determine one).
    numcores : int, optional
        Number of cores to use for multi-processing.

    Returns
    -------
    tuple
        (out,err)
        out : array, shape (N,len(t),12)
            base orbit (cylindrical R,vR,vT,z,vz,phi) in [...,:6] and the
            deviation in [...,6:] (rectangular if ``rectOut`` else cylindrical).
        err : array
            Error message per orbit (0 unless maximum step reduction happened).

    Notes
    -----
    - 2026-06-03 - Written based on integratePlanarOrbit_dxdv - Bovy (UofT)
    """
    # Go to the rectangular frame: base state (R,vR,vT,z,vz,phi) -> (x,y,z,vx,vy,vz)
    R, vR, vT, z, vz, phi = (
        yo[:, 0],
        yo[:, 1],
        yo[:, 2],
        yo[:, 3],
        yo[:, 4],
        yo[:, 5],
    )
    X, Y, Z = coords.cyl_to_rect(R, phi, z)
    vX, vY, vZ = coords.cyl_to_rect_vec(vR, vT, vz, phi)
    this_yo = numpy.array([X, Y, Z, vX, vY, vZ]).T
    if not rectIn:
        # Chain rule: rect deviation = J . cyl deviation, with J the Jacobian
        # of (x,y,z,vx,vy,vz) wrt (R,vR,vT,z,vz,phi).
        this_dyo = numpy.empty_like(dyo)
        for ii in range(len(yo)):
            jac = coords.cyl_to_rect_jac(R[ii], vR[ii], vT[ii], z[ii], vz[ii], phi[ii])
            this_dyo[ii] = numpy.dot(jac, dyo[ii])
    else:
        this_dyo = dyo
    this_yo = numpy.hstack((this_yo, this_dyo))
    if int_method.lower() == "dop853" or int_method.lower() == "odeint":
        from ..potential.DissipativeForce import _isDissipative

        if _isDissipative(pot):
            # The pure-Python variational RHS (_EOM_dxdv) only implements the
            # conservative A=[[0,I],[K,0]] system: it neither passes the
            # velocity to the force evaluators nor includes the dissipative
            # dF/dx and dF/dv Jacobian blocks, so it would silently produce a
            # wrong deviation. Fail loudly instead; the C path supports
            # dissipative forces with a C implementation of the
            # velocity-dependent force Jacobian (hasC_dxdv3d=True).
            raise NotImplementedError(
                "integrate_dxdv with dissipative forces is not supported by the "
                "pure-Python methods ('odeint', 'dop853'); use a C method (e.g. "
                "'dopr54_c') with dissipative forces that have a C implementation "
                "of the velocity-dependent force Jacobian (hasC_dxdv3d=True)"
            )
        if int_method.lower() == "dop853":
            if rtol is None:
                rtol = 1e-12
            if atol is None:
                atol = 1e-12
            integrator = dop853
            extra_kwargs = {"rtol": rtol, "atol": atol}
        else:
            integrator = integrate.odeint
            extra_kwargs = {"rtol": rtol, "atol": atol}

        def integrate_for_map(vxvv):
            return integrator(_EOM_dxdv, vxvv, t=t, args=(pot,), **extra_kwargs)

    else:  # Assume we are forcing parallel_mapping of a C integrator...

        def integrate_for_map(vxvv):
            return integrateFullOrbit_dxdv_c(
                pot,
                numpy.copy(vxvv[:6]),
                numpy.copy(vxvv[6:]),
                t,
                int_method,
                dt=dt,
                rtol=rtol,
                atol=atol,
            )[0]

    if len(this_yo) == 1:  # Can't map a single value...
        out = numpy.atleast_3d(integrate_for_map(this_yo[0]).T).T
    else:
        out = numpy.array(
            parallel_map(
                integrate_for_map, this_yo, progressbar=progressbar, numcores=numcores
            )
        )
    # Go back to the cylindrical frame: base state out[...,:6] is rectangular
    # (x,y,z,vx,vy,vz); convert to (R,vR,vT,z,vz,phi) and (optionally) the
    # deviation out[...,6:] from rectangular to cylindrical.
    Rout, phiout, Zout = coords.rect_to_cyl(out[..., 0], out[..., 1], out[..., 2])
    vRout, vTout, vzout = coords.rect_to_cyl_vec(
        out[..., 3], out[..., 4], out[..., 5], out[..., 0], out[..., 1], out[..., 2]
    )
    # rect_to_cyl/rect_to_cyl_vec pass Z/vz through BY REFERENCE, so Zout and
    # vzout are views into out[...,2]/out[...,5]; copy them before the in-place
    # assignments below overwrite those columns (otherwise out[...,3] ends up
    # holding vT instead of z, corrupting the returned base orbit and any
    # restart that uses it, e.g. the lyapunov renormalization segments)
    Zout = numpy.copy(Zout)
    vzout = numpy.copy(vzout)
    out[..., 0] = Rout
    out[..., 1] = vRout
    out[..., 2] = vTout
    out[..., 3] = Zout
    out[..., 4] = vzout
    out[..., 5] = phiout
    if not rectOut:
        # cyl deviation = J^{-1} . rect deviation, with J the cyl->rect Jacobian
        # evaluated at each (R,vR,vT,z,vz,phi) along the orbit.
        shp = Rout.shape
        Rf = Rout.ravel()
        vRf = vRout.ravel()
        vTf = vTout.ravel()
        Zf = Zout.ravel()
        vzf = vzout.ravel()
        phif = phiout.ravel()
        dev_rect = out[..., 6:].reshape((-1, 6))
        dev_cyl = numpy.empty_like(dev_rect)
        for ii in range(dev_rect.shape[0]):
            jac = coords.cyl_to_rect_jac(
                Rf[ii], vRf[ii], vTf[ii], Zf[ii], vzf[ii], phif[ii]
            )
            dev_cyl[ii] = numpy.linalg.solve(jac, dev_rect[ii])
        out[..., 6:] = dev_cyl.reshape(shp + (6,))
    return out, numpy.zeros(len(yo))


@potential_list_of_potentials_input
def integrateFullOrbit(
    pot, yo, t, int_method, rtol=None, atol=None, numcores=1, progressbar=True, dt=None
):
    """
    Integrate an ode for a FullOrbit

    Parameters
    ----------
    pot : Potential, CompositePotential, or a combined potential formed using addition (pot1+pot2+…)
        The potential to evaluate the orbit in. Lists are deprecated and will be converted to CompositePotential.
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
    # Per-orbit time arrays: when t is 2D (shape (norbit,nt)), each orbit gets its own row
    indiv_t = len(t.shape) > 1

    def _t_for(idx):
        return t[idx] if indiv_t else t

    if int_method.lower() == "leapfrog":
        if rtol is None:
            rtol = 1e-8
        if atol is None:
            atol = 1e-8

        def integrate_for_map(idx):
            vxvv = yo[idx]
            this_t = _t_for(idx)
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
                _rectForce, this_vxvv, this_t, args=(pot,), rtol=rtol, atol=atol
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
        if int_method.lower() == "dop853":
            if rtol is None:
                rtol = 1e-12
            if atol is None:
                atol = 1e-12
            integrator = dop853
            extra_kwargs = {"rtol": rtol, "atol": atol}
        else:
            integrator = integrate.odeint
            extra_kwargs = {"rtol": rtol, "atol": atol}
        if len(yo[0]) == 5:

            def integrate_for_map(idx):
                vxvv = yo[idx]
                this_t = _t_for(idx)
                l = vxvv[0] * vxvv[2]
                l2 = l**2.0
                init = [vxvv[0], vxvv[1], vxvv[3], vxvv[4]]
                intOut = integrator(
                    _RZEOM, init, t=this_t, args=(pot, l2), **extra_kwargs
                )
                out = numpy.zeros((len(this_t), 5))
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

            def integrate_for_map(idx):
                vxvv = yo[idx]
                this_t = _t_for(idx)
                vphi = vxvv[2] / vxvv[0]
                init = [vxvv[0], vxvv[1], vxvv[5], vphi, vxvv[3], vxvv[4]]
                intOut = integrator(
                    _EOM, init, t=this_t, args=(pot,), rtol=rtol, atol=atol
                )
                out = numpy.zeros((len(this_t), 6))
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

        def integrate_for_map(idx):
            vxvv = yo[idx]
            this_t = _t_for(idx)
            return integrateFullOrbit_c(
                pot, numpy.copy(vxvv), this_t, int_method, dt=dt, rtol=rtol, atol=atol
            )[0]

    if len(yo) == 1:  # Can't map a single value...
        out = numpy.atleast_3d(integrate_for_map(0).T).T
    else:
        out = numpy.array(
            parallel_map(
                integrate_for_map,
                numpy.arange(len(yo)),
                numcores=numcores,
                progressbar=progressbar,
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
    pot : Potential or a combined potential formed using addition (pot1+pot2+…)
        The potential to evaluate the orbit in.
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


@potential_list_of_potentials_input
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
    pot : Potential, CompositePotential, or a combined potential formed using addition (pot1+pot2+…)
        The potential to evaluate the orbit in. Lists are deprecated and will be converted to CompositePotential.
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
        if int_method.lower() == "dop853":
            if rtol is None:
                rtol = 1e-12
            if atol is None:
                atol = 1e-12
            integrator = dop853
            extra_kwargs = {"rtol": rtol, "atol": atol}
        else:
            integrator = integrate.odeint
            extra_kwargs = {"rtol": rtol, "atol": atol}

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
    pot : Potential instance
        Potential instance.
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
    pot : Potential instance
        Potential instance.

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
    pot : Potential instance
        Potential instance

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
    pot : Potential instance or a combined potential formed using addition (pot1+pot2+…)
        The potential to evaluate the force for
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


def _EOM_dxdv(x, t, pot):
    """
    Implements the EOM, i.e., the right-hand side of the differential
    equation, for integrating a 3D orbit + phase-space deviation, in the
    rectangular frame.

    Parameters
    ----------
    x : numpy.ndarray
        Current 12D rectangular phase-space state
        (x,y,z,vx,vy,vz | dx,dy,dz,dvx,dvy,dvz).
    t : float
        Current time.
    pot : Potential instance or a combined potential formed using addition (pot1+pot2+…)

    Returns
    -------
    numpy.ndarray
        dx/dt (12D).

    Notes
    -----
    - 2026-06-03 - Written, mirroring the C evalRectDeriv_dxdv - Bovy (UofT)
    """
    # x is rectangular so calculate R and phi
    R = numpy.sqrt(x[0] ** 2.0 + x[1] ** 2.0)
    phi = numpy.arccos(x[0] / R)
    sinphi = x[1] / R
    cosphi = x[0] / R
    if x[1] < 0.0:
        phi = 2.0 * numpy.pi - phi
    z = x[2]
    # Cartesian forces -> accelerations
    Rforce = _evaluateRforces(pot, R, z, phi=phi, t=t)
    phitorque = _evaluatephitorques(pot, R, z, phi=phi, t=t)
    zforce = _evaluatezforces(pot, R, z, phi=phi, t=t)
    # Cylindrical second derivatives of the potential
    R2deriv = evaluateR2derivs(pot, R, z, phi=phi, t=t)
    phi2deriv = evaluatephi2derivs(pot, R, z, phi=phi, t=t)
    Rphideriv = evaluateRphiderivs(pot, R, z, phi=phi, t=t)
    z2deriv = evaluatez2derivs(pot, R, z, phi=phi, t=t)
    Rzderiv = evaluateRzderivs(pot, R, z, phi=phi, t=t)
    zphideriv = evaluatephizderivs(pot, R, z, phi=phi, t=t)
    # Symmetric Cartesian tidal tensor K = -grad grad Phi; in-plane (x,y) block
    # identical to the verified 2D variational equations (z enters only through
    # the second-derivative values above).
    dFxdx = (
        -(cosphi**2.0) * R2deriv
        + 2.0 * cosphi * sinphi / R**2.0 * phitorque
        + sinphi**2.0 / R * Rforce
        + 2.0 * sinphi * cosphi / R * Rphideriv
        - sinphi**2.0 / R**2.0 * phi2deriv
    )
    dFxdy = (
        -sinphi * cosphi * R2deriv
        + (sinphi**2.0 - cosphi**2.0) / R**2.0 * phitorque
        - cosphi * sinphi / R * Rforce
        - (cosphi**2.0 - sinphi**2.0) / R * Rphideriv
        + cosphi * sinphi / R**2.0 * phi2deriv
    )
    dFydy = (
        -(sinphi**2.0) * R2deriv
        - 2.0 * sinphi * cosphi / R**2.0 * phitorque
        - 2.0 * sinphi * cosphi / R * Rphideriv
        + cosphi**2.0 / R * Rforce
        - cosphi**2.0 / R**2.0 * phi2deriv
    )
    # z-coupling (K symmetric: dFzdx=dFxdz, dFzdy=dFydz, dFydx=dFxdy)
    dFxdz = -cosphi * Rzderiv + sinphi / R * zphideriv
    dFydz = -sinphi * Rzderiv - cosphi / R * zphideriv
    dFzdz = -z2deriv
    dx, dy, dz = x[6], x[7], x[8]
    return numpy.array(
        [
            x[3],
            x[4],
            x[5],
            cosphi * Rforce - 1.0 / R * sinphi * phitorque,
            sinphi * Rforce + 1.0 / R * cosphi * phitorque,
            zforce,
            x[9],
            x[10],
            x[11],
            dFxdx * dx + dFxdy * dy + dFxdz * dz,
            dFxdy * dx + dFydy * dy + dFydz * dz,
            dFxdz * dx + dFydz * dy + dFzdz * dz,
        ]
    )
