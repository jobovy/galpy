import ctypes
import ctypes.util

import numpy
from numpy.ctypeslib import ndpointer
from scipy import integrate

from .. import potential
from ..potential.planarDissipativeForce import (
    planarDissipativeForceFromFullDissipativeForce,
)
from ..potential.planarPotential import (
    _evaluateplanarphitorques,
    _evaluateplanarPotentials,
    _evaluateplanarRforces,
    planarPotentialFromFullPotential,
    planarPotentialFromRZPotential,
)
from ..potential.WrapperPotential import WrapperPotential, parentWrapperPotential
from ..util import _load_extension_libs, symplecticode
from ..util._optional_deps import _NUMBA_LOADED, _TQDM_LOADED
from ..util.leung_dop853 import dop853
from ..util.multi import parallel_map

if _TQDM_LOADED:
    import tqdm
if _NUMBA_LOADED:
    from numba import cfunc, types

_lib, _ext_loaded = _load_extension_libs.load_libgalpy()


def _parse_pot(pot):
    """Parse the potential so it can be fed to C"""
    # Figure out what's in pot
    if not isinstance(pot, list):
        pot = [pot]
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
        # Prepare for wrappers
        if (
            (
                isinstance(p, planarPotentialFromFullPotential)
                or isinstance(p, planarPotentialFromRZPotential)
            )
            and isinstance(p._Pot, (parentWrapperPotential, WrapperPotential))
        ) or isinstance(p, (parentWrapperPotential, WrapperPotential)):
            if not isinstance(p, (parentWrapperPotential, WrapperPotential)):
                wrap_npot, wrap_pot_type, wrap_pot_args, wrap_pot_tfuncs = _parse_pot(
                    potential.toPlanarPotential(p._Pot._pot)
                )
            else:
                wrap_npot, wrap_pot_type, wrap_pot_args, wrap_pot_tfuncs = _parse_pot(
                    p._pot
                )
        if (
            isinstance(p, planarPotentialFromRZPotential)
            or isinstance(p, planarPotentialFromFullPotential)
        ) and isinstance(p._Pot, potential.LogarithmicHaloPotential):
            pot_type.append(0)
            if p._Pot.isNonAxi:
                pot_args.extend(
                    [p._Pot._amp, p._Pot._q, p._Pot._core2, p._Pot._1m1overb2]
                )
            else:
                pot_args.extend(
                    [p._Pot._amp, p._Pot._q, p._Pot._core2, 2.0]
                )  # 1m1overb2 > 1: axi
        elif isinstance(p, planarPotentialFromFullPotential) and isinstance(
            p._Pot, potential.DehnenBarPotential
        ):
            pot_type.append(1)
            pot_args.extend(
                [
                    p._Pot._amp * p._Pot._af,
                    p._Pot._tform,
                    p._Pot._tsteady,
                    p._Pot._rb,
                    p._Pot._omegab,
                    p._Pot._barphi,
                ]
            )
        elif isinstance(p, potential.TransientLogSpiralPotential):
            pot_type.append(2)
            pot_args.extend(
                [p._amp, p._A, p._to, p._sigma2, p._alpha, p._m, p._omegas, p._gamma]
            )
        elif isinstance(p, potential.SteadyLogSpiralPotential):
            pot_type.append(3)
            if p._tform is None:
                pot_args.extend(
                    [
                        p._amp,
                        float("nan"),
                        float("nan"),
                        p._A,
                        p._alpha,
                        p._m,
                        p._omegas,
                        p._gamma,
                    ]
                )
            else:
                pot_args.extend(
                    [
                        p._amp,
                        p._tform,
                        p._tsteady,
                        p._A,
                        p._alpha,
                        p._m,
                        p._omegas,
                        p._gamma,
                    ]
                )
        elif isinstance(p, potential.EllipticalDiskPotential):
            pot_type.append(4)
            if p._tform is None:
                pot_args.extend(
                    [p._amp, float("nan"), float("nan"), p._twophio, p._p, p._phib]
                )
            else:
                pot_args.extend(
                    [p._amp, p._tform, p._tsteady, p._twophio, p._p, p._phib]
                )
        elif isinstance(p, planarPotentialFromRZPotential) and isinstance(
            p._Pot, potential.MiyamotoNagaiPotential
        ):
            pot_type.append(5)
            pot_args.extend([p._Pot._amp, p._Pot._a, p._Pot._b])
        elif isinstance(p, potential.LopsidedDiskPotential):
            pot_type.append(6)
            pot_args.extend([p._amp, p._mphio, p._p, p._phib])
        elif isinstance(p, planarPotentialFromRZPotential) and isinstance(
            p._Pot, potential.PowerSphericalPotential
        ):
            pot_type.append(7)
            pot_args.extend([p._Pot._amp, p._Pot.alpha])
        elif isinstance(p, planarPotentialFromRZPotential) and isinstance(
            p._Pot, potential.HernquistPotential
        ):
            pot_type.append(8)
            pot_args.extend([p._Pot._amp, p._Pot.a])
        elif isinstance(p, planarPotentialFromRZPotential) and isinstance(
            p._Pot, potential.NFWPotential
        ):
            pot_type.append(9)
            pot_args.extend([p._Pot._amp, p._Pot.a])
        elif isinstance(p, planarPotentialFromRZPotential) and isinstance(
            p._Pot, potential.JaffePotential
        ):
            pot_type.append(10)
            pot_args.extend([p._Pot._amp, p._Pot.a])
        elif isinstance(p, planarPotentialFromRZPotential) and isinstance(
            p._Pot, potential.DoubleExponentialDiskPotential
        ):
            pot_type.append(11)
            pot_args.extend(
                [
                    p._Pot._amp,
                    -4.0 * numpy.pi * p._Pot._alpha * p._Pot._amp,
                    p._Pot._alpha,
                    p._Pot._beta,
                    len(p._Pot._de_j1_xs),
                ]
            )
            pot_args.extend(p._Pot._de_j0_xs)
            pot_args.extend(p._Pot._de_j1_xs)
            pot_args.extend(p._Pot._de_j0_weights)
            pot_args.extend(p._Pot._de_j1_weights)
        elif isinstance(p, planarPotentialFromRZPotential) and isinstance(
            p._Pot, potential.FlattenedPowerPotential
        ):
            pot_type.append(12)
            pot_args.extend([p._Pot._amp, p._Pot.alpha, p._Pot.core2])
        elif isinstance(p, planarPotentialFromRZPotential) and isinstance(
            p._Pot, potential.IsochronePotential
        ):
            pot_type.append(14)
            pot_args.extend([p._Pot._amp, p._Pot.b])
        elif isinstance(p, planarPotentialFromRZPotential) and isinstance(
            p._Pot, potential.PowerSphericalPotentialwCutoff
        ):
            pot_type.append(15)
            pot_args.extend([p._Pot._amp, p._Pot.alpha, p._Pot.rc])
        elif isinstance(p, planarPotentialFromRZPotential) and isinstance(
            p._Pot, potential.MN3ExponentialDiskPotential
        ):
            # Three Miyamoto-Nagai disks
            npot += 2
            pot_type.extend([5, 5, 5])
            pot_args.extend(
                [
                    p._Pot._amp * p._Pot._mn3[0]._amp,
                    p._Pot._mn3[0]._a,
                    p._Pot._mn3[0]._b,
                    p._Pot._amp * p._Pot._mn3[1]._amp,
                    p._Pot._mn3[1]._a,
                    p._Pot._mn3[1]._b,
                    p._Pot._amp * p._Pot._mn3[2]._amp,
                    p._Pot._mn3[2]._a,
                    p._Pot._mn3[2]._b,
                ]
            )
        elif isinstance(p, planarPotentialFromRZPotential) and isinstance(
            p._Pot, potential.KuzminKutuzovStaeckelPotential
        ):
            pot_type.append(16)
            pot_args.extend([p._Pot._amp, p._Pot._ac, p._Pot._Delta])
        elif isinstance(p, planarPotentialFromRZPotential) and isinstance(
            p._Pot, potential.PlummerPotential
        ):
            pot_type.append(17)
            pot_args.extend([p._Pot._amp, p._Pot._b])
        elif isinstance(p, planarPotentialFromRZPotential) and isinstance(
            p._Pot, potential.PseudoIsothermalPotential
        ):
            pot_type.append(18)
            pot_args.extend([p._Pot._amp, p._Pot._a])
        elif isinstance(p, planarPotentialFromRZPotential) and isinstance(
            p._Pot, potential.KuzminDiskPotential
        ):
            pot_type.append(19)
            pot_args.extend([p._Pot._amp, p._Pot._a])
        elif isinstance(p, planarPotentialFromRZPotential) and isinstance(
            p._Pot, potential.BurkertPotential
        ):
            pot_type.append(20)
            pot_args.extend([p._Pot._amp, p._Pot.a])
        elif (
            isinstance(p, planarPotentialFromFullPotential)
            or isinstance(p, planarPotentialFromRZPotential)
        ) and isinstance(p._Pot, potential.EllipsoidalPotential.EllipsoidalPotential):
            pot_args.append(p._Pot._amp)
            pot_args.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # for caching
            if isinstance(p._Pot, potential.TriaxialHernquistPotential):
                pot_type.append(21)
                pot_args.extend([2, p._Pot.a, p._Pot.a4])  # for psi, mdens, mdens_deriv
            if isinstance(p._Pot, potential.TriaxialNFWPotential):
                pot_type.append(22)
                pot_args.extend([2, p._Pot.a, p._Pot.a3])  # for psi, mdens, mdens_deriv
            if isinstance(p._Pot, potential.TriaxialJaffePotential):
                pot_type.append(23)
                pot_args.extend([2, p._Pot.a, p._Pot.a2])  # for psi, mdens, mdens_deriv
            elif isinstance(p._Pot, potential.PerfectEllipsoidPotential):
                pot_type.append(30)
                pot_args.extend([1, p._Pot.a2])  # for psi, mdens, mdens_deriv
            elif isinstance(p._Pot, potential.TriaxialGaussianPotential):
                pot_type.append(37)
                pot_args.extend([1, -p._Pot._twosigma2])  # for psi, mdens, mdens_deriv
            elif isinstance(p._Pot, potential.PowerTriaxialPotential):
                pot_type.append(38)
                pot_args.extend([1, p._Pot.alpha])  # for psi, mdens, mdens_deriv
            pot_args.extend(
                [p._Pot._b2, p._Pot._c2, int(p._Pot._aligned)]
            )  # Reg. Ellipsoidal
            if not p._Pot._aligned:
                pot_args.extend(list(p._Pot._rot.flatten()))
            else:
                pot_args.extend(list(numpy.eye(3).flatten()))  # not actually used
            pot_args.append(p._Pot._glorder)
            pot_args.extend([p._Pot._glx[ii] for ii in range(p._Pot._glorder)])
            # this adds some common factors to the integration weights
            pot_args.extend(
                [
                    -4.0
                    * numpy.pi
                    * p._Pot._glw[ii]
                    * p._Pot._b
                    * p._Pot._c
                    / numpy.sqrt(
                        (1.0 + (p._Pot._b2 - 1.0) * p._Pot._glx[ii] ** 2.0)
                        * (1.0 + (p._Pot._c2 - 1.0) * p._Pot._glx[ii] ** 2.0)
                    )
                    for ii in range(p._Pot._glorder)
                ]
            )
        elif (
            isinstance(p, planarPotentialFromFullPotential)
            or isinstance(p, planarPotentialFromRZPotential)
        ) and isinstance(p._Pot, potential.SCFPotential):
            pt, pa, ptf = _parse_scf_pot(p._Pot)
            pot_type.append(pt)
            pot_args.extend(pa)
            pot_tfuncs.extend(ptf)
        elif isinstance(p, planarPotentialFromFullPotential) and isinstance(
            p._Pot, potential.SoftenedNeedleBarPotential
        ):
            pot_type.append(25)
            pot_args.extend(
                [
                    p._Pot._amp,
                    p._Pot._a,
                    p._Pot._b,
                    p._Pot._c2,
                    p._Pot._pa,
                    p._Pot._omegab,
                ]
            )
            pot_args.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # for caching
        elif (
            isinstance(p, planarPotentialFromFullPotential)
            or isinstance(p, planarPotentialFromRZPotential)
        ) and isinstance(p._Pot, potential.DiskSCFPotential):
            # Need to pull this apart into: (a) SCF part, (b) constituent
            # [Sigma_i,h_i] parts
            # (a) SCF, multiply in any add'l amp
            pt, pa, ptf = _parse_scf_pot(p._Pot._scf, extra_amp=p._Pot._amp)
            pot_type.append(pt)
            pot_args.extend(pa)
            pot_tfuncs.extend(ptf)
            # (b) constituent [Sigma_i,h_i] parts
            for Sigma, hz in zip(p._Pot._Sigma_dict, p._Pot._hz_dict):
                npot += 1
                pot_type.append(26)
                stype = Sigma.get("type", "exp")
                if stype == "exp" and not "Rhole" in Sigma:
                    pot_args.extend(
                        [
                            3,
                            0,
                            4.0 * numpy.pi * Sigma.get("amp", 1.0) * p._Pot._amp,
                            Sigma.get("h", 1.0 / 3.0),
                        ]
                    )
                elif stype == "expwhole" or (stype == "exp" and "Rhole" in Sigma):
                    pot_args.extend(
                        [
                            4,
                            1,
                            4.0 * numpy.pi * Sigma.get("amp", 1.0) * p._Pot._amp,
                            Sigma.get("h", 1.0 / 3.0),
                            Sigma.get("Rhole", 0.5),
                        ]
                    )
                hztype = hz.get("type", "exp")
                if hztype == "exp":
                    pot_args.extend([0, hz.get("h", 0.0375)])
                elif hztype == "sech2":
                    pot_args.extend([1, hz.get("h", 0.0375)])
        elif isinstance(p, planarPotentialFromFullPotential) and isinstance(
            p._Pot, potential.SpiralArmsPotential
        ):
            pot_type.append(27)
            pot_args.extend(
                [
                    len(p._Pot._Cs),
                    p._Pot._amp,
                    p._Pot._N,
                    p._Pot._sin_alpha,
                    p._Pot._tan_alpha,
                    p._Pot._r_ref,
                    p._Pot._phi_ref,
                    p._Pot._Rs,
                    p._Pot._H,
                    p._Pot._omega,
                ]
            )
            pot_args.extend(p._Pot._Cs)
        elif isinstance(p, potential.CosmphiDiskPotential):
            pot_type.append(28)
            pot_args.extend(
                [p._amp, p._mphio, p._p, p._mphib, p._m, p._rb, p._rbp, p._rb2p, p._r1p]
            )
        elif isinstance(p, potential.HenonHeilesPotential):
            pot_type.append(29)
            pot_args.extend([p._amp])
        # 30: PerfectEllipsoidPotential, done with other EllipsoidalPotentials above
        # 31: KGPotential
        # 32: IsothermalDiskPotential
        elif isinstance(p, planarPotentialFromRZPotential) and isinstance(
            p._Pot, potential.DehnenCoreSphericalPotential
        ):
            pot_type.append(33)
            pot_args.extend([p._Pot._amp, p._Pot.a])
        elif isinstance(p, planarPotentialFromRZPotential) and isinstance(
            p._Pot, potential.DehnenSphericalPotential
        ):
            pot_type.append(34)
            pot_args.extend([p._Pot._amp, p._Pot.a, p._Pot.alpha])
        # 35: HomogeneousSpherePotential
        elif isinstance(p, planarPotentialFromRZPotential) and isinstance(
            p._Pot, potential.HomogeneousSpherePotential
        ):
            pot_type.append(35)
            pot_args.extend([p._Pot._amp, p._Pot._R2, p._Pot._R3])
        # 36: interpSphericalPotential
        elif isinstance(p, planarPotentialFromRZPotential) and isinstance(
            p._Pot, potential.interpSphericalPotential
        ):
            pot_type.append(36)
            pot_args.append(len(p._Pot._rgrid))
            pot_args.extend(p._Pot._rgrid)
            pot_args.extend(p._Pot._rforce_grid)
            pot_args.extend(
                [
                    p._Pot._amp,
                    p._Pot._rmin,
                    p._Pot._rmax,
                    p._Pot._total_mass,
                    p._Pot._Phi0,
                    p._Pot._Phimax,
                ]
            )
        # 37: TriaxialGaussianPotential, done with other EllipsoidalPotentials above
        # 38: PowerTriaxialPotential, done with other EllipsoidalPotentials above
        elif isinstance(
            p, planarDissipativeForceFromFullDissipativeForce
        ) and isinstance(p._Pot, potential.NonInertialFrameForce):
            pot_type.append(39)
            pot_args.append(p._Pot._amp)
            pot_args.extend(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            )  # for caching
            pot_args.extend(
                [
                    p._Pot._rot_acc,
                    p._Pot._lin_acc,
                    p._Pot._omegaz_only,
                    p._Pot._const_freq,
                    p._Pot._Omega_as_func,
                ]
            )
            if p._Pot._Omega_as_func:
                pot_args.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            else:
                if p._Pot._omegaz_only:
                    pot_args.extend([0.0, 0.0, p._Pot._Omega])
                else:
                    pot_args.extend(p._Pot._Omega)
                pot_args.append(p._Pot._Omega2)
                if not p._Pot._const_freq and p._Pot._omegaz_only:
                    pot_args.extend([0.0, 0.0, p._Pot._Omegadot])
                elif not p._Pot._const_freq:
                    pot_args.extend(p._Pot._Omegadot)
                else:
                    pot_args.extend([0.0, 0.0, 0.0])
            if p._Pot._lin_acc:
                pot_tfuncs.extend([p._Pot._a0[0], p._Pot._a0[1], p._Pot._a0[2]])
                if p._Pot._rot_acc:
                    pot_tfuncs.extend([p._Pot._x0[0], p._Pot._x0[1], p._Pot._x0[2]])
                    pot_tfuncs.extend([p._Pot._v0[0], p._Pot._v0[1], p._Pot._v0[2]])
            if p._Pot._Omega_as_func:
                if p._Pot._omegaz_only:
                    pot_tfuncs.extend([p._Pot._Omega, p._Pot._Omegadot])
                else:
                    pot_tfuncs.extend(
                        [
                            p._Pot._Omega[0],
                            p._Pot._Omega[1],
                            p._Pot._Omega[2],
                            p._Pot._Omegadot[0],
                            p._Pot._Omegadot[1],
                            p._Pot._Omegadot[2],
                        ]
                    )
        elif isinstance(p, planarPotentialFromRZPotential) and isinstance(
            p._Pot, potential.NullPotential
        ):
            pot_type.append(40)
        ############################## WRAPPERS ###############################
        elif (
            (
                isinstance(p, planarPotentialFromFullPotential)
                or isinstance(p, planarPotentialFromRZPotential)
            )
            and isinstance(p._Pot, potential.DehnenSmoothWrapperPotential)
        ) or isinstance(p, potential.DehnenSmoothWrapperPotential):
            if not isinstance(p, potential.DehnenSmoothWrapperPotential):
                p = p._Pot
            pot_type.append(-1)
            # wrap_pot_type, args, and npot obtained before this horrible if
            pot_args.append(wrap_npot)
            pot_type.extend(wrap_pot_type)
            pot_args.extend(wrap_pot_args)
            pot_tfuncs.extend(wrap_pot_tfuncs)
            pot_args.extend([p._amp, p._tform, p._tsteady, int(p._grow)])
        elif (
            (
                isinstance(p, planarPotentialFromFullPotential)
                or isinstance(p, planarPotentialFromRZPotential)
            )
            and isinstance(p._Pot, potential.SolidBodyRotationWrapperPotential)
        ) or isinstance(p, potential.SolidBodyRotationWrapperPotential):
            if not isinstance(p, potential.SolidBodyRotationWrapperPotential):
                p = p._Pot
            pot_type.append(-2)
            # wrap_pot_type, args, and npot obtained before this horrible if
            pot_args.append(wrap_npot)
            pot_type.extend(wrap_pot_type)
            pot_args.extend(wrap_pot_args)
            pot_tfuncs.extend(wrap_pot_tfuncs)
            pot_args.extend([p._amp, p._omega, p._pa])
        elif (
            (
                isinstance(p, planarPotentialFromFullPotential)
                or isinstance(p, planarPotentialFromRZPotential)
            )
            and isinstance(p._Pot, potential.CorotatingRotationWrapperPotential)
        ) or isinstance(p, potential.CorotatingRotationWrapperPotential):
            if not isinstance(p, potential.CorotatingRotationWrapperPotential):
                p = p._Pot
            pot_type.append(-4)
            # wrap_pot_type, args, and npot obtained before this horrible if
            pot_args.append(wrap_npot)
            pot_type.extend(wrap_pot_type)
            pot_args.extend(wrap_pot_args)
            pot_tfuncs.extend(wrap_pot_tfuncs)
            pot_args.extend([p._amp, p._vpo, p._beta, p._pa, p._to])
        elif (
            (
                isinstance(p, planarPotentialFromFullPotential)
                or isinstance(p, planarPotentialFromRZPotential)
            )
            and isinstance(p._Pot, potential.GaussianAmplitudeWrapperPotential)
        ) or isinstance(p, potential.GaussianAmplitudeWrapperPotential):
            if not isinstance(p, potential.GaussianAmplitudeWrapperPotential):
                p = p._Pot
            pot_type.append(-5)
            # wrap_pot_type, args, and npot obtained before this horrible if
            pot_args.append(wrap_npot)
            pot_type.extend(wrap_pot_type)
            pot_args.extend(wrap_pot_args)
            pot_tfuncs.extend(wrap_pot_tfuncs)
            pot_args.extend([p._amp, p._to, p._sigma2])
        elif (
            (
                isinstance(p, planarPotentialFromFullPotential)
                or isinstance(p, planarPotentialFromRZPotential)
            )
            and isinstance(p._Pot, potential.MovingObjectPotential)
        ) or isinstance(p, potential.MovingObjectPotential):
            if not isinstance(p, potential.MovingObjectPotential):
                p = p._Pot
            pot_type.append(-6)
            wrap_npot, wrap_pot_type, wrap_pot_args, wrap_pot_tfuncs = _parse_pot(
                potential.toPlanarPotential(p._pot)
            )
            pot_args.append(wrap_npot)
            pot_type.extend(wrap_pot_type)
            pot_args.extend(wrap_pot_args)
            pot_tfuncs.extend(wrap_pot_tfuncs)
            pot_args.extend([len(p._orb.t)])
            pot_args.extend(p._orb.t)
            pot_args.extend(p._orb.x(p._orb.t, use_physical=False))
            pot_args.extend(p._orb.y(p._orb.t, use_physical=False))
            pot_args.extend([p._amp])
            pot_args.extend([p._orb.t[0], p._orb.t[-1]])  # t_0, t_f
        elif (
            (
                isinstance(p, planarPotentialFromFullPotential)
                or isinstance(p, planarPotentialFromRZPotential)
            )
            and isinstance(p._Pot, potential.RotateAndTiltWrapperPotential)
        ) or isinstance(p, potential.RotateAndTiltWrapperPotential):  # pragma: no cover
            raise NotImplementedError(
                "Planar orbit integration in C for RotateAndTiltWrapperPotential not implemented; please integrate an orbit with (z,vz) = (0,0) instead"
            )
            # Note that potential.RotateAndTiltWrapperPotential would be -8
        elif (
            (
                isinstance(p, planarPotentialFromFullPotential)
                or isinstance(p, planarPotentialFromRZPotential)
            )
            and isinstance(p._Pot, potential.TimeDependentAmplitudeWrapperPotential)
        ) or isinstance(p, potential.TimeDependentAmplitudeWrapperPotential):
            if not isinstance(p, potential.TimeDependentAmplitudeWrapperPotential):
                p = p._Pot
            pot_type.append(-9)
            # wrap_pot_type, args, and npot obtained before this horrible if
            pot_args.append(wrap_npot)
            pot_type.extend(wrap_pot_type)
            pot_args.extend(wrap_pot_args)
            pot_tfuncs.extend(wrap_pot_tfuncs)
            pot_args.append(p._amp)
            pot_tfuncs.append(p._A)
        elif (
            (
                isinstance(p, planarPotentialFromFullPotential)
                or isinstance(p, planarPotentialFromRZPotential)
            )
            and isinstance(p._Pot, potential.KuzminLikeWrapperPotential)
        ) or isinstance(p, potential.KuzminLikeWrapperPotential):
            if not isinstance(p, potential.KuzminLikeWrapperPotential):
                p = p._Pot
            pot_type.append(-10)
            # wrap_pot_type, args, and npot obtained before this horrible if
            pot_args.append(wrap_npot)
            pot_type.extend(wrap_pot_type)
            pot_args.extend(wrap_pot_args)
            pot_tfuncs.extend(wrap_pot_tfuncs)
            pot_args.extend([p._amp, p._a, p._b2])
    pot_type = numpy.array(pot_type, dtype=numpy.int32, order="C")
    pot_args = numpy.array(pot_args, dtype=numpy.float64, order="C")
    return (npot, pot_type, pot_args, pot_tfuncs)


def _parse_integrator(int_method):
    """parse the integrator method to pass to C"""
    # Pick integrator
    if int_method.lower() == "rk4_c":
        int_method_c = 1
    elif int_method.lower() == "rk6_c":
        int_method_c = 2
    elif int_method.lower() == "symplec4_c":
        int_method_c = 3
    elif int_method.lower() == "symplec6_c":
        int_method_c = 4
    elif int_method.lower() == "dopr54_c":
        int_method_c = 5
    elif int_method.lower() == "dop853_c":
        int_method_c = 6
    else:
        int_method_c = 0
    return int_method_c


def _parse_tol(rtol, atol):
    """Parse the tolerance keywords"""
    # Process atol and rtol
    if rtol is None:
        rtol = -12.0 * numpy.log(10.0)
    else:  # pragma: no cover
        rtol = numpy.log(rtol)
    if atol is None:
        atol = -12.0 * numpy.log(10.0)
    else:  # pragma: no cover
        atol = numpy.log(atol)
    return (rtol, atol)


def _parse_scf_pot(p, extra_amp=1.0):
    # Stand-alone parser for SCF, bc re-used
    isNonAxi = p.isNonAxi
    pot_args = [p._a, isNonAxi]
    pot_args.extend(p._Acos.shape)
    pot_args.extend(extra_amp * p._amp * p._Acos.flatten(order="C"))
    if isNonAxi:
        pot_args.extend(extra_amp * p._amp * p._Asin.flatten(order="C"))
    pot_args.extend([-1.0, 0, 0, 0, 0, 0, 0])
    return (24, pot_args, [])  # latter is pot_tfuncs


def _prep_tfuncs(pot_tfuncs):
    if len(pot_tfuncs) == 0:
        pot_tfuncs = None  # NULL
    else:
        func_ctype = ctypes.CFUNCTYPE(
            ctypes.c_double,
            ctypes.c_double,  # Return type
        )  # time
        try:  # using numba
            if not _NUMBA_LOADED:
                raise
            nb_c_sig = types.double(types.double)
            func_pyarr = [cfunc(nb_c_sig, nopython=True)(a).ctypes for a in pot_tfuncs]
        except:  # Any Exception, switch to regular ctypes wrapping
            func_pyarr = [func_ctype(a) for a in pot_tfuncs]
        pot_tfuncs = (func_ctype * len(func_pyarr))(*func_pyarr)
    return pot_tfuncs


def integratePlanarOrbit_c(
    pot, yo, t, int_method, rtol=None, atol=None, progressbar=True, dt=None
):
    """
    Integrate an ode for a planarOrbit.

    Parameters
    ----------
    pot : Potential or list of such instances
    yo : numpy.ndarray
        Initial condition [q,p], can be [N,4] or [4].
    t : numpy.ndarray
        Set of times at which one wants the result.
    int_method : str
        Integration method. Options are 'leapfrog_c', 'rk4_c', 'rk6_c', 'symplec4_c', ...
    rtol : float, optional
        Relative tolerance.
    atol : float, optional
        Absolute tolerance.
    progressbar : bool, optional
        If True, display a tqdm progress bar when integrating multiple orbits (requires tqdm to be installed!).
    dt : float, optional
        Force integrator to use this stepsize (default is to automatically determine one).

    Returns
    -------
    tuple
        (y,err)
        y : array, shape (len(y0),len(t),4)
            Array containing the value of y for each desired time in t, with the initial value y0 in the first row.
        err : int
            Error message, if not zero: 1 means maximum step reduction happened for adaptive integrators.

    Notes
    -----
    - 2011-10-03 - Written - Bovy (IAS)
    - 2018-12-20 - Adapted to allow multiple objects - Bovy (UofT)
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
    result = numpy.empty((nobj, len(t), 4))
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
    integrationFunc = _lib.integratePlanarOrbit
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


def integratePlanarOrbit_dxdv_c(
    pot, yo, dyo, t, int_method, rtol=None, atol=None, dt=None
):
    """
    Integrate an ode for a planarOrbit+phase space volume dxdv

    Parameters
    ----------
    pot : Potential or list of such instances
    yo : numpy.ndarray
        Initial condition [q,p]
    dyo : numpy.ndarray
        Initial condition [dq,dp]
    t : numpy.ndarray
        Set of times at which one wants the result
    int_method : str
        Integration method. One of 'leapfrog_c', 'rk4_c', 'rk6_c', 'symplec4_c'
    rtol : float, optional
        Relative tolerance. Default is None
    atol : float, optional
        Absolute tolerance. Default is None
    dt : float, optional
        Force integrator to use this stepsize (default is to automatically determine one)

    Returns
    -------
    tuple
        (y,err)
        y,dy : array, shape (len(y0),len(t),8)
        Array containing the value of y for each desired time in t, \
        with the initial value y0 in the first row.
        err: error message if not zero, 1: maximum step reduction happened for adaptive integrators

    Notes
    -----
    - 2011-10-19 - Written - Bovy (IAS)

    """
    rtol, atol = _parse_tol(rtol, atol)
    npot, pot_type, pot_args, pot_tfuncs = _parse_pot(pot)
    pot_tfuncs = _prep_tfuncs(pot_tfuncs)
    int_method_c = _parse_integrator(int_method)
    if dt is None:
        dt = -9999.99
    yo = numpy.concatenate((yo, dyo))

    # Set up result array
    result = numpy.empty((len(t), 8))
    err = ctypes.c_int(0)

    # Set up the C code
    ndarrayFlags = ("C_CONTIGUOUS", "WRITEABLE")
    integrationFunc = _lib.integratePlanarOrbit_dxdv
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

    if err.value == -10:  # pragma: no cover
        raise KeyboardInterrupt("Orbit integration interrupted by CTRL-C (SIGINT)")

    # Reset input arrays
    if f_cont[0]:
        yo = numpy.asfortranarray(yo)
    if f_cont[1]:
        t = numpy.asfortranarray(t)

    return (result, err.value)


def integratePlanarOrbit(
    pot, yo, t, int_method, rtol=None, atol=None, numcores=1, progressbar=True, dt=None
):
    """
    Integrate an ode for a planarOrbit

    Parameters
    ----------
    pot : Potential or list of such instances
    yo : numpy.ndarray
        Initial condition [q,p], shape [N,3] or [N,4]
    t : numpy.ndarray
        Set of times at which one wants the result
    int_method : str
        Integration method. One of 'leapfrog', 'odeint', 'dop853'
    rtol : float, optional
        Relative tolerance. Default is None
    atol : float, optional
        Absolute tolerance. Default is None
    numcores : int, optional
        Number of cores to use for multi-processing
    progressbar : bool, optional
        If True, display a tqdm progress bar when integrating multiple orbits (requires tqdm to be installed!).
    dt : float, optional
        Force integrator to use this stepsize (default is to automatically determine one)

    Returns
    -------
    tuple
        (y,err)
        y : array, shape (N,len(t),3/4)
        Array containing the value of y for each desired time in t, \
        with the initial value y0 in the first row.
        err: error message, always zero for now

    Notes
    -----
    - 2010-07-20 - Written - Bovy (NYU)
    - 2019-04-09 - Adapted to allow multiple objects and parallel mapping - Bovy (UofT)
    - 2022-04-12 - Add progressbar - Bovy (UofT)
    """
    nophi = False
    if not int_method.lower() == "dop853" and not int_method == "odeint":
        if len(yo[0]) == 3:
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
                    vxvv[0] * numpy.cos(vxvv[3]),
                    vxvv[0] * numpy.sin(vxvv[3]),
                    vxvv[1] * numpy.cos(vxvv[3]) - vxvv[2] * numpy.sin(vxvv[3]),
                    vxvv[2] * numpy.cos(vxvv[3]) + vxvv[1] * numpy.sin(vxvv[3]),
                ]
            )
            # integrate
            tmp_out = symplecticode.leapfrog(
                _planarRectForce, this_vxvv, t, args=(pot,), rtol=rtol
            )
            # go back to the cylindrical frame
            R = numpy.sqrt(tmp_out[:, 0] ** 2.0 + tmp_out[:, 1] ** 2.0)
            phi = numpy.arccos(tmp_out[:, 0] / R)
            phi[(tmp_out[:, 1] < 0.0)] = 2.0 * numpy.pi - phi[(tmp_out[:, 1] < 0.0)]
            vR = tmp_out[:, 2] * numpy.cos(phi) + tmp_out[:, 3] * numpy.sin(phi)
            vT = tmp_out[:, 3] * numpy.cos(phi) - tmp_out[:, 2] * numpy.sin(phi)
            out = numpy.zeros((len(t), 4))
            out[:, 0] = R
            out[:, 1] = vR
            out[:, 2] = vT
            out[:, 3] = phi
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
        if len(yo[0]) == 3:

            def integrate_for_map(vxvv):
                l = vxvv[0] * vxvv[2]
                l2 = l**2.0
                init = [vxvv[0], vxvv[1]]
                intOut = integrator(
                    _planarREOM, init, t=t, args=(pot, l2), **extra_kwargs
                )
                out = numpy.zeros((len(t), 3))
                out[:, 0] = intOut[:, 0]
                out[:, 1] = intOut[:, 1]
                out[:, 2] = l / out[:, 0]
                # post-process to remove negative radii
                neg_radii = out[:, 0] < 0.0
                out[neg_radii, 0] = -out[neg_radii, 0]
                return out

        else:

            def integrate_for_map(vxvv):
                vphi = vxvv[2] / vxvv[0]
                init = [vxvv[0], vxvv[1], vxvv[3], vphi]
                intOut = integrator(_planarEOM, init, t=t, args=(pot,), **extra_kwargs)
                out = numpy.zeros((len(t), 4))
                out[:, 0] = intOut[:, 0]
                out[:, 1] = intOut[:, 1]
                out[:, 3] = intOut[:, 2]
                out[:, 2] = out[:, 0] * intOut[:, 3]
                # post-process to remove negative radii
                neg_radii = out[:, 0] < 0.0
                out[neg_radii, 0] = -out[neg_radii, 0]
                out[neg_radii, 3] += numpy.pi
                return out

    else:  # Assume we are forcing parallel_mapping of a C integrator...

        def integrate_for_map(vxvv):
            return integratePlanarOrbit_c(pot, numpy.copy(vxvv), t, int_method, dt=dt)[
                0
            ]

    if len(yo) == 1:  # Can't map a single value...
        out = numpy.atleast_3d(integrate_for_map(yo[0]).T).T
    else:
        out = numpy.array(
            parallel_map(
                integrate_for_map, yo, numcores=numcores, progressbar=progressbar
            )
        )
    if nophi:
        out = out[:, :, :3]
    return out, numpy.zeros(len(yo))


def integratePlanarOrbit_dxdv(
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
    Integrate an ode for a planarOrbit+phase space volume dxdv

    Parameters
    ----------
    pot : Potential or list of such instances
    yo : numpy.ndarray
        Initial condition [q,p], shape [N,4]
    dyo : numpy.ndarray
        Initial condition [dq,dp], shape [N,4]
    t : numpy.ndarray
        Set of times at which one wants the result
    int_method : str
        Integration method. One of 'leapfrog', 'odeint', 'dop853'
    rectIn : bool
        If True, input dyo is in rectangular coordinates
    rectOut : bool
        If True, output dyo is in rectangular coordinates
    rtol : float, optional
        Relative tolerance. Default is None
    atol : float, optional
        Absolute tolerance. Default is None
    progressbar : bool, optional
        If True, display a tqdm progress bar when integrating multiple orbits (requires tqdm to be installed!).
    dt : float, optional
        Force integrator to use this stepsize (default is to automatically determine one)
    numcores : int, optional
        Number of cores to use for multi-processing

    Returns
    -------
    tuple
        (y,err)
        y,dy : array, shape (N,len(t),8)
        Array containing the value of y for each desired time in t, \
        with the initial value y0 in the first row.
        err: error message if not zero, 1: maximum step reduction happened for adaptive integrators

    Notes
    -----
    - 2011-10-17 - Written - Bovy (IAS)
    - 2019-05-21 - Adapted to allow multiple objects and parallel mapping - Bovy (UofT)
    - 2022-04-12 - Add progressbar - Bovy (UofT)
    """
    # go to the rectangular frame
    this_yo = numpy.array(
        [
            yo[:, 0] * numpy.cos(yo[:, 3]),
            yo[:, 0] * numpy.sin(yo[:, 3]),
            yo[:, 1] * numpy.cos(yo[:, 3]) - yo[:, 2] * numpy.sin(yo[:, 3]),
            yo[:, 2] * numpy.cos(yo[:, 3]) + yo[:, 1] * numpy.sin(yo[:, 3]),
        ]
    ).T
    if not rectIn:
        this_dyo = numpy.array(
            [
                numpy.cos(yo[:, 3]) * dyo[:, 0]
                - yo[:, 0] * numpy.sin(yo[:, 3]) * dyo[:, 3],
                numpy.sin(yo[:, 3]) * dyo[:, 0]
                + yo[:, 0] * numpy.cos(yo[:, 3]) * dyo[:, 3],
                -(yo[:, 1] * numpy.sin(yo[:, 3]) + yo[:, 2] * numpy.cos(yo[:, 3]))
                * dyo[:, 3]
                + numpy.cos(yo[:, 3]) * dyo[:, 1]
                - numpy.sin(yo[:, 3]) * dyo[:, 2],
                (yo[:, 1] * numpy.cos(yo[:, 3]) - yo[:, 2] * numpy.sin(yo[:, 3]))
                * dyo[:, 3]
                + numpy.sin(yo[:, 3]) * dyo[:, 1]
                + numpy.cos(yo[:, 3]) * dyo[:, 2],
            ]
        ).T
    else:
        this_dyo = dyo
    this_yo = numpy.hstack((this_yo, this_dyo))
    if int_method.lower() == "dop853" or int_method.lower() == "odeint":
        if rtol is None:
            rtol = 1e-8
        if int_method.lower() == "dop853":
            integrator = dop853
            extra_kwargs = {}
        else:
            integrator = integrate.odeint
            extra_kwargs = {"rtol": rtol}

        def integrate_for_map(vxvv):
            return integrator(_planarEOM_dxdv, vxvv, t=t, args=(pot,), **extra_kwargs)

    else:  # Assume we are forcing parallel_mapping of a C integrator...

        def integrate_for_map(vxvv):
            return integratePlanarOrbit_dxdv_c(
                pot,
                numpy.copy(vxvv[:4]),
                numpy.copy(vxvv[4:]),
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
    # go back to the cylindrical frame
    R = numpy.sqrt(out[..., 0] ** 2.0 + out[..., 1] ** 2.0)
    phi = numpy.arccos(out[..., 0] / R)
    phi[(out[..., 1] < 0.0)] = 2.0 * numpy.pi - phi[(out[..., 1] < 0.0)]
    vR = out[..., 2] * numpy.cos(phi) + out[..., 3] * numpy.sin(phi)
    vT = out[..., 3] * numpy.cos(phi) - out[..., 2] * numpy.sin(phi)
    cp = numpy.cos(phi)
    sp = numpy.sin(phi)
    out[..., 0] = R
    out[..., 1] = vR
    out[..., 2] = vT
    out[..., 3] = phi
    if rectOut:
        out[..., 4:] = out[..., 4:]
    else:
        dR = cp * out[..., 4] + sp * out[..., 5]
        dphi = (cp * out[..., 5] - sp * out[..., 4]) / R
        dvR = cp * out[..., 6] + sp * out[..., 7] + vT * dphi
        dvT = cp * out[..., 7] - sp * out[..., 6] - vR * dphi
        out[..., 4] = dR
        out[..., 7] = dphi
        out[..., 5] = dvR
        out[..., 6] = dvT
    return out, numpy.zeros(len(yo))


def integratePlanarOrbit_sos_c(
    pot,
    yo,
    psi,
    t0,
    int_method,
    surface="x",
    rtol=None,
    atol=None,
    progressbar=True,
    dpsi=None,
):
    """
    Integrate an ode for a PlanarOrbit for integrate_sos in C

    Parameters
    ----------
    pot : Potential or list of such instances
    yo : numpy.ndarray
        Initial condition [q,p], shape [N,5] or [N,6]
    psi : numpy.ndarray
        Set of increment angles at which one wants the result [increments wrt initial angle]
    t0 : float or numpy.ndarray
        Initial time
    int_method : str
        'rk4_c', 'rk6_c', 'dopr54_c', or 'dop853_c'
    surface : str, optional
        Surface to use ('x' for finding x=0, vx>0; 'y' for finding y=0, vy>0), by default "x"
    rtol : float, optional
        Relative tolerance, by default None
    atol : float, optional
        Absolute tolerance, by default None
    progressbar : bool, optional
        If True, display a tqdm progress bar when integrating multiple orbits (requires tqdm to be installed!), by default True
    dpsi : float, optional
        Force integrator to use this stepsize (default is to automatically determine one; only for C-based integrators), by default None

    Returns
    -------
    tuple
        (y,err)
        y : array, shape (N,len(psi),5) where the last of the last dimension is the time
            Array containing the value of y for each desired angle in psi, with the initial value y0 in the first row.
        err : int
            Error message, always zero for now

    Notes
    -----
    - 2023-03-17 - Written based on integrateFullOrbit_sos_c - Bovy (UofT)
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
    yoo = numpy.empty((nobj, 5))
    yoo[:, :4] = yo[:, :4]
    if len(t0) == 1:
        yoo[:, 4] = t0[0]
    else:
        yoo[:, 4] = t0
    npsi = len(psi.T)  # .T to make npsi always the first dim

    # Set up result array
    result = numpy.empty((nobj, npsi, 5))
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
    integrationFunc = _lib.integratePlanarOrbit_sos
    integrationFunc.argtypes = [
        ctypes.c_int,
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_int,
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_int,
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
        ctypes.c_int(1 if surface == "y" else 0),
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


def integratePlanarOrbit_sos(
    pot,
    yo,
    psi,
    t0,
    int_method,
    surface="x",
    rtol=None,
    atol=None,
    numcores=1,
    progressbar=True,
    dpsi=None,
):
    """
    Integrate an ode for a PlanarOrbit for integrate_sos

    Parameters
    ----------
    pot : Potential or list of such instances
    yo : numpy.ndarray
        Initial condition [q,p], shape [N,5] or [N,6]
    psi : numpy.ndarray
        Set of increment angles at which one wants the result [increments wrt initial angle]
    t0 : float or numpy.ndarray
        Initial time
    surface : str, optional
        Surface to use ('x' for finding x=0, vx>0; 'y' for finding y=0, vy>0), by default "x"
    int_method : str
        Integration method to use. One of 'leapfrog', 'odeint', or 'dop853'
    rtol : float, optional
        Relative tolerance, by default None
    atol : float, optional
        Absolute tolerance, by default None
    numcores : int, optional
        Number of cores to use for multi-processing, by default 1
    progressbar : bool, optional
        If True, display a tqdm progress bar when integrating multiple orbits (requires tqdm to be installed!), by default True
    dpsi : float, optional
        Force integrator to use this stepsize (default is to automatically determine one; only for C-based integrators), by default None

    Returns
    -------
    tuple
        (y,err)
        y : array, shape (N,len(psi),4/5) where the last of the last dimension is the time
            Array containing the value of y for each desired angle in psi, with the initial value y0 in the first row.
        err : int
            Error message, always zero for now

    Notes
    -----
    - 2023-03-24 - Written based on integrateFullOrbi_sos - Bovy (UofT)
    """
    if surface is None:
        surface = "x"
    nophi = False
    if len(yo[0]) == 3:
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
            # go to the transformed plane: (A,t,y,vy) or (x,vx,A,t)
            if surface.lower() == "x":
                x = vxvv[0] * numpy.cos(vxvv[3])
                vx = vxvv[1] * numpy.cos(vxvv[3]) - vxvv[2] * numpy.sin(vxvv[3])
                init_psi = numpy.arctan2(x, vx)
                this_vxvv = numpy.array(
                    [
                        vxvv[0] * numpy.sin(vxvv[3]),
                        vxvv[2] * numpy.cos(vxvv[3]) + vxvv[1] * numpy.sin(vxvv[3]),
                        numpy.sqrt(x**2.0 + vx**2.0),
                        t0,
                    ]
                )
                # integrate
                intOut = integrator(
                    _planarSOSEOMx,
                    this_vxvv,
                    t=psi + init_psi,
                    args=(pot,),
                    **extra_kwargs,
                )
                # go back to the cylindrical frame
                x = intOut[:, 2] * numpy.sin(psi + init_psi)
                vx = intOut[:, 2] * numpy.cos(psi + init_psi)
                y = intOut[:, 0]
                vy = intOut[:, 1]
            else:
                y = vxvv[0] * numpy.sin(vxvv[3])
                vy = vxvv[2] * numpy.cos(vxvv[3]) + vxvv[1] * numpy.sin(vxvv[3])
                init_psi = numpy.arctan2(y, vy)
                this_vxvv = numpy.array(
                    [
                        vxvv[0] * numpy.cos(vxvv[3]),
                        vxvv[1] * numpy.cos(vxvv[3]) - vxvv[2] * numpy.sin(vxvv[3]),
                        numpy.sqrt(y**2.0 + vy**2.0),
                        t0,
                    ]
                )
                # integrate
                intOut = integrator(
                    _planarSOSEOMy,
                    this_vxvv,
                    t=psi + init_psi,
                    args=(pot,),
                    **extra_kwargs,
                )
                # go back to the cylindrical frame
                x = intOut[:, 0]
                vx = intOut[:, 1]
                y = intOut[:, 2] * numpy.sin(psi + init_psi)
                vy = intOut[:, 2] * numpy.cos(psi + init_psi)
            out = numpy.zeros((len(psi), 5))
            out[:, 0] = numpy.sqrt(x**2.0 + y**2.0)
            out[:, 3] = numpy.arctan2(y, x)
            out[:, 1] = vx * numpy.cos(out[:, 3]) + vy * numpy.sin(out[:, 3])
            out[:, 2] = vy * numpy.cos(out[:, 3]) - vx * numpy.sin(out[:, 3])
            out[:, 4] = intOut[:, 3]
            return out

    else:  # Assume we are forcing parallel_mapping of a C integrator...

        def integrate_for_map(vxvv, psi, t0):
            return integratePlanarOrbit_sos_c(
                pot, numpy.copy(vxvv), psi, t0, int_method, surface=surface, dpsi=dpsi
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
        phi_mask[3] = False
        out = out[:, :, phi_mask]
    return out, numpy.zeros(len(yo))


def _planarREOM(y, t, pot, l2):
    """
    Implements the EOM, i.e., the right-hand side of the differential
    equation, for integrating a planar Orbit assuming angular momentum
    conservation.

    Parameters
    ----------
    y : numpy.ndarray
        Current phase-space position.
    t : float
        Current time.
    pot : list of Potential instance(s)
        Potential instance(s).
    l2 : float
        Angular momentum squared.

    Returns
    -------
    numpy.ndarray
        dy/dt.

    Notes
    -----
    - 2010-07-20 - Written - Bovy (NYU)
    """
    return [y[1], l2 / y[0] ** 3.0 + _evaluateplanarRforces(pot, y[0], t=t)]


def _planarEOM(y, t, pot):
    """
    Implements the EOM, i.e., the right-hand side of the differential equation, for integrating a general planar Orbit

    Parameters
    ----------
    y : numpy.ndarray
        Current phase-space position
    t : float
        Current time
    pot : (list of) Potential instance(s)

    Returns
    -------
    numpy.ndarray
        dy/dt

    Notes
    -----
    - 2010-07-20 - Written - Bovy (NYU)

    """
    l2 = (y[0] ** 2.0 * y[3]) ** 2.0
    return [
        y[1],
        l2 / y[0] ** 3.0
        + _evaluateplanarRforces(pot, y[0], phi=y[2], t=t, v=[y[1], y[0] * y[3]]),
        y[3],
        1.0
        / y[0] ** 2.0
        * (
            _evaluateplanarphitorques(pot, y[0], phi=y[2], t=t, v=[y[1], y[0] * y[3]])
            - 2.0 * y[0] * y[1] * y[3]
        ),
    ]


def _planarEOM_dxdv(x, t, pot):
    """
    Implements the EOM, i.e., the right-hand side of the differential
    equation, for integrating phase space differences, rectangular

    Parameters
    ----------
    x : numpy.ndarray
        Current phase-space position
    t : float
        Current time
    pot : (list of) Potential instance(s)

    Returns
    -------
    numpy.ndarray
        dy/dt

    Notes
    -----
    - 2011-10-18 - Written - Bovy (IAS)
    """
    # x is rectangular so calculate R and phi
    R = numpy.sqrt(x[0] ** 2.0 + x[1] ** 2.0)
    phi = numpy.arccos(x[0] / R)
    sinphi = x[1] / R
    cosphi = x[0] / R
    if x[1] < 0.0:
        phi = 2.0 * numpy.pi - phi
    # calculate forces
    Rforce = _evaluateplanarRforces(pot, R, phi=phi, t=t)
    phitorque = _evaluateplanarphitorques(pot, R, phi=phi, t=t)
    R2deriv = _evaluateplanarPotentials(pot, R, phi=phi, t=t, dR=2)
    phi2deriv = _evaluateplanarPotentials(pot, R, phi=phi, t=t, dphi=2)
    Rphideriv = _evaluateplanarPotentials(pot, R, phi=phi, t=t, dR=1, dphi=1)
    # Calculate derivatives and derivatives+time derivatives
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
    dFydx = (
        -cosphi * sinphi * R2deriv
        + (sinphi**2.0 - cosphi**2.0) / R**2.0 * phitorque
        + (sinphi**2.0 - cosphi**2.0) / R * Rphideriv
        - sinphi * cosphi / R * Rforce
        + sinphi * cosphi / R**2.0 * phi2deriv
    )
    dFydy = (
        -(sinphi**2.0) * R2deriv
        - 2.0 * sinphi * cosphi / R**2.0 * phitorque
        - 2.0 * sinphi * cosphi / R * Rphideriv
        + cosphi**2.0 / R * Rforce
        - cosphi**2.0 / R**2.0 * phi2deriv
    )
    return numpy.array(
        [
            x[2],
            x[3],
            cosphi * Rforce - 1.0 / R * sinphi * phitorque,
            sinphi * Rforce + 1.0 / R * cosphi * phitorque,
            x[6],
            x[7],
            dFxdx * x[4] + dFxdy * x[5],
            dFydx * x[4] + dFydy * x[5],
        ]
    )


def _planarSOSEOMx(y, psi, pot):
    """
    Implements the EOM, i.e., the right-hand side of the differential
    equation, for integrating a general planar Orbit in the SOS style

    Parameters
    ----------
    y : numpy.ndarray
        Current phase-space position
    psi : float
        Current angle
    pot : (list of) Potential instance(s)

    Returns
    -------
    numpy.ndarray
        dy/dt

    Notes
    -----
    - 2023-03-24 - Written - Bovy (UofT)
    """
    # y = (y,vy,A,t)
    # Calculate x, vx
    sp, cp = numpy.sin(psi), numpy.cos(psi)
    gxyz = _planarRectForce([y[2] * sp, y[0]], pot, t=y[3], vx=[y[2] * cp, y[1]])
    psidot = cp**2.0 - sp / y[2] * gxyz[0]
    Adot = y[2] * cp * sp + gxyz[0] * cp
    return numpy.array([y[1], gxyz[1], Adot, 1.0]) / psidot


def _planarSOSEOMy(y, psi, pot):
    """
    Implements the EOM, i.e., the right-hand side of the differential
    equation, for integrating a general planar Orbit in the SOS style

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
        dy/dt

    Notes
    -----
    - 2023-03-24 - Written - Bovy (UofT)
    """

    # y = (x,vx,A,t)
    # Calculate y
    sp, cp = numpy.sin(psi), numpy.cos(psi)
    gxyz = _planarRectForce([y[0], y[2] * sp], pot, t=y[3], vx=[y[1], y[2] * cp])
    psidot = cp**2.0 - sp / y[2] * gxyz[1]
    Adot = y[2] * cp * sp + gxyz[1] * cp
    return numpy.array([y[1], gxyz[0], Adot, 1.0]) / psidot


def _planarRectForce(x, pot, t=0.0, vx=None):
    """
    Returns the planar force in the rectangular frame.

    Parameters
    ----------
    x : numpy.ndarray
        Current position.
    t : float, optional
        Current time (default is 0.0).
    pot : list or Potential instance(s)
        Potential instance(s).
    vx : numpy.ndarray, optional
        If set, use this [vx,vy] when evaluating dissipative forces (default is None).

    Returns
    -------
    numpy.ndarray
        Force.

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
        vx = [vR, vT]
    # calculate forces
    Rforce = _evaluateplanarRforces(pot, R, phi=phi, t=t, v=vx)
    phitorque = _evaluateplanarphitorques(pot, R, phi=phi, t=t, v=vx)
    return numpy.array(
        [
            cosphi * Rforce - 1.0 / R * sinphi * phitorque,
            sinphi * Rforce + 1.0 / R * cosphi * phitorque,
        ]
    )
