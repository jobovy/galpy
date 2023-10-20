import ctypes
import ctypes.util

import numpy
from numpy.ctypeslib import ndpointer
from scipy import integrate

from .. import potential
from ..potential.linearPotential import _evaluatelinearForces
from ..potential.verticalPotential import verticalPotential
from ..util import _load_extension_libs, symplecticode
from ..util._optional_deps import _TQDM_LOADED
from ..util.leung_dop853 import dop853
from ..util.multi import parallel_map
from .integrateFullOrbit import _parse_pot as _parse_pot_full
from .integratePlanarOrbit import _parse_integrator, _parse_tol, _prep_tfuncs

if _TQDM_LOADED:
    import tqdm

_lib, _ext_loaded = _load_extension_libs.load_libgalpy()


def _parse_pot(pot):
    """Parse the potential so it can be fed to C"""
    from .integrateFullOrbit import _parse_scf_pot

    # Figure out what's in pot
    if not isinstance(pot, list):
        pot = [pot]
    # Initialize everything
    pot_type = []
    pot_args = []
    pot_tfuncs = []
    npot = len(pot)
    for p in pot:
        # Prepare for wrappers NOT CURRENTLY SUPPORTED, SEE PLANAR OR FULL
        if isinstance(p, verticalPotential) and isinstance(
            p._Pot, potential.MN3ExponentialDiskPotential
        ):
            # Need to do this one separately, because combination of many parts
            # Three Miyamoto-Nagai disks
            npot += 2
            pot_type.extend([5, 5, 5])
            pot_args.extend(
                [
                    p._Pot._amp * p._Pot._mn3[0]._amp,
                    p._Pot._mn3[0]._a,
                    p._Pot._mn3[0]._b,
                    p._R,
                    p._phi,
                    p._Pot._amp * p._Pot._mn3[1]._amp,
                    p._Pot._mn3[1]._a,
                    p._Pot._mn3[1]._b,
                    p._R,
                    p._phi,
                    p._Pot._amp * p._Pot._mn3[2]._amp,
                    p._Pot._mn3[2]._a,
                    p._Pot._mn3[2]._b,
                    p._R,
                    p._phi,
                ]
            )
        elif isinstance(p, verticalPotential) and isinstance(
            p._Pot, potential.DiskSCFPotential
        ):
            # Need to do this one separately, because combination of many parts
            # Need to pull this apart into: (a) SCF part, (b) constituent
            # [Sigma_i,h_i] parts
            # (a) SCF, multiply in any add'l amp
            pt, pa, ptf = _parse_scf_pot(p._Pot._scf, extra_amp=p._Pot._amp)
            pot_type.append(pt)
            pot_args.extend(pa)
            pot_tfuncs.extend(ptf)
            pot_args.extend([p._R, p._phi])
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
                pot_args.extend([p._R, p._phi])
        elif isinstance(p, potential.KGPotential):
            pot_type.append(31)
            pot_args.extend([p._amp, p._K, p._D2, 2.0 * p._F])
        elif isinstance(p, potential.IsothermalDiskPotential):
            pot_type.append(32)
            pot_args.extend([p._amp * p._sigma2 / p._H, 2.0 * p._H])
        # All other potentials can be handled in the same way as follows:
        elif isinstance(p, verticalPotential):
            _, pt, pa, ptf = _parse_pot_full(p._Pot)
            pot_type.extend(pt)
            pot_args.extend(pa)
            pot_tfuncs.extend(ptf)
            pot_args.append(p._R)
            pot_args.append(p._phi)
    pot_type = numpy.array(pot_type, dtype=numpy.int32, order="C")
    pot_args = numpy.array(pot_args, dtype=numpy.float64, order="C")
    return (npot, pot_type, pot_args, pot_tfuncs)


def integrateLinearOrbit_c(
    pot, yo, t, int_method, rtol=None, atol=None, progressbar=True, dt=None
):
    """
    C integrate an ode for a LinearOrbit

    Parameters
    ----------
    pot : Potential or list of such instances
    yo : numpy.ndarray
        initial condition [q,p], shape [N,2] or [2]
    t : numpy.ndarray
        set of times at which one wants the result
    int_method : str
        integration method
    rtol : float, optional
        relative tolerance
    atol : float, optional
        absolute tolerance
    progressbar : bool, optional
        if True, display a tqdm progress bar
    dt : float, optional
        force integrator to use this stepsize (default is to automatically determine one; only for C-based integrators)

    Returns
    -------
    tuple
        (y,err)
        y : Array containing the value of y for each desired time in t, with the initial value y0 in the first row.
        err: error message, if not zero: 1 means maximum step reduction happened for adaptive integrators

    Notes
    -----
    - 2018-10-06 - Written - Bovy (UofT)
    - 2018-10-14 - Adapted to allow multiple orbits to be integrated at once - Bovy (UofT)
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
    result = numpy.empty((nobj, len(t), 2))
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
    integrationFunc = _lib.integrateLinearOrbit
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


# Python integration functions
def integrateLinearOrbit(
    pot, yo, t, int_method, rtol=None, atol=None, numcores=1, progressbar=True, dt=None
):
    """
    Integrate an ode for a LinearOrbit

    Parameters
    ----------
    pot : Potential or list of such instances
    yo : numpy.ndarray
        initial condition [q,p], shape [N,2] or [2]
    t : numpy.ndarray
        set of times at which one wants the result
    int_method : str
        integration method
    rtol : float, optional
        relative tolerance
    atol : float, optional
        absolute tolerance
    numcores : int, optional
        number of cores to use
    progressbar : bool, optional
        if True, display a tqdm progress bar
    dt : float, optional
        force integrator to use this stepsize (default is to automatically determine one; only for C-based integrators)

    Returns
    -------
    tuple
        (y,err)
        y : Array containing the value of y for each desired time in t, with the initial value y0 in the first row.
        err: error message, if not zero: 1 means maximum step reduction happened for adaptive integrators

    Notes
    -----
    - 2010-07-13- Written - Bovy (NYU)
    - 2019-04-08 - Adapted to allow multiple orbits to be integrated at once and moved to integrateLinearOrbit.py - Bovy (UofT)
    - 2022-04-12 - Add progressbar - Bovy (UofT)
    """
    if int_method.lower() == "leapfrog":
        if rtol is None:
            rtol = 1e-8

        def integrate_for_map(vxvv):
            return symplecticode.leapfrog(
                lambda x, t=t: _evaluatelinearForces(pot, x, t=t),
                numpy.array(vxvv),
                t,
                rtol=rtol,
            )

    elif int_method.lower() == "dop853":
        if rtol is None:
            rtol = 1e-8

        def integrate_for_map(vxvv):
            return dop853(func=_linearEOM, x=vxvv, t=t, args=(pot,))

    elif int_method.lower() == "odeint":
        if rtol is None:
            rtol = 1e-8

        def integrate_for_map(vxvv):
            return integrate.odeint(_linearEOM, vxvv, t, args=(pot,), rtol=rtol)

    else:  # Assume we are forcing parallel_mapping of a C integrator...

        def integrate_for_map(vxvv):
            return integrateLinearOrbit_c(pot, numpy.copy(vxvv), t, int_method, dt=dt)[
                0
            ]

    if len(yo) == 1:  # Can't map a single value...
        return numpy.atleast_3d(integrate_for_map(yo[0]).T).T, 0
    else:
        return (
            numpy.array(
                parallel_map(
                    integrate_for_map, yo, numcores=numcores, progressbar=progressbar
                )
            ),
            numpy.zeros(len(yo)),
        )


def _linearEOM(y, t, pot):
    """
    The one-dimensional equation-of-motion

    Parameters
    ----------
    y : list or numpy.ndarray
        Current phase-space position
    t : float
        Current time
    pot : list
        (list of) linearPotential instance(s)

    Returns
    -------
    list
        dy/dt

    Notes
    -----
    - 2010-07-13 - Bovy (NYU)

    """
    return [y[1], _evaluatelinearForces(pot, y[0], t=t)]
