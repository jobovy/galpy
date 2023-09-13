import ctypes
import ctypes.util

import numpy
from numpy.ctypeslib import ndpointer

from ..util import _load_extension_libs

_lib, _ext_loaded = _load_extension_libs.load_libgalpy()


def actionAngleAdiabatic_c(pot, gamma, R, vR, vT, z, vz):
    """
    Use C to calculate actions using the adiabatic approximation

    Parameters
    ----------
    pot : Potential or list of such instances
        Gravitational potential to compute actions in
    gamma : float
        as in Lz -> Lz+gamma * J_z
    R : numpy.ndarray
        R coordinate.
    vR : numpy.ndarray
        vR coordinate.
    vT : numpy.ndarray
        vT coordinate.
    z : numpy.ndarray
        z coordinate.
    vz : numpy.ndarray
        vz coordinate.

    Returns
    -------
    tuple:
       (jr,jz,err) with radial, vertical action (numpy.ndarrays), and error (non-zero if error occurred)

    Notes
    -----
    - 2012-12-10 - Written - Bovy (IAS)
    """

    # Parse the potential
    from ..orbit.integrateFullOrbit import _parse_pot
    from ..orbit.integratePlanarOrbit import _prep_tfuncs

    npot, pot_type, pot_args, pot_tfuncs = _parse_pot(pot, potforactions=True)
    pot_tfuncs = _prep_tfuncs(pot_tfuncs)

    # Set up result arrays
    jr = numpy.empty(len(R))
    jz = numpy.empty(len(R))
    err = ctypes.c_int(0)

    # Set up the C code
    ndarrayFlags = ("C_CONTIGUOUS", "WRITEABLE")
    actionAngleAdiabatic_actionsFunc = _lib.actionAngleAdiabatic_actions
    actionAngleAdiabatic_actionsFunc.argtypes = [
        ctypes.c_int,
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_int,
        ndpointer(dtype=numpy.int32, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_void_p,
        ctypes.c_double,
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.POINTER(ctypes.c_int),
    ]

    # Array requirements, first store old order
    f_cont = [
        R.flags["F_CONTIGUOUS"],
        vR.flags["F_CONTIGUOUS"],
        vT.flags["F_CONTIGUOUS"],
        z.flags["F_CONTIGUOUS"],
        vz.flags["F_CONTIGUOUS"],
    ]
    R = numpy.require(R, dtype=numpy.float64, requirements=["C", "W"])
    vR = numpy.require(vR, dtype=numpy.float64, requirements=["C", "W"])
    vT = numpy.require(vT, dtype=numpy.float64, requirements=["C", "W"])
    z = numpy.require(z, dtype=numpy.float64, requirements=["C", "W"])
    vz = numpy.require(vz, dtype=numpy.float64, requirements=["C", "W"])
    jr = numpy.require(jr, dtype=numpy.float64, requirements=["C", "W"])
    jz = numpy.require(jz, dtype=numpy.float64, requirements=["C", "W"])

    # Run the C code
    actionAngleAdiabatic_actionsFunc(
        len(R),
        R,
        vR,
        vT,
        z,
        vz,
        ctypes.c_int(npot),
        pot_type,
        pot_args,
        pot_tfuncs,
        ctypes.c_double(gamma),
        jr,
        jz,
        ctypes.byref(err),
    )

    # Reset input arrays
    if f_cont[0]:
        R = numpy.asfortranarray(R)
    if f_cont[1]:
        vR = numpy.asfortranarray(vR)
    if f_cont[2]:
        vT = numpy.asfortranarray(vT)
    if f_cont[3]:
        z = numpy.asfortranarray(z)
    if f_cont[4]:
        vz = numpy.asfortranarray(vz)

    return (jr, jz, err.value)


def actionAngleRperiRapZmaxAdiabatic_c(pot, gamma, R, vR, vT, z, vz):
    """
    Calculate planar (Rperi,Rap) and the maximum height Zmax using the adiabatic approximation (rap = sqrt(Rap^2+Zmax^2))

    Parameters
    ----------
    pot : Potential or list of such instances
        Gravitational potential to compute actions in
    gamma : float
        as in Lz -> Lz+gamma * J_z
    R : numpy.ndarray
        R coordinate.
    vR : numpy.ndarray
        vR coordinate.
    vT : numpy.ndarray
        vT coordinate.
    z : numpy.ndarray
        z coordinate.
    vz : numpy.ndarray
        vz coordinate.

    Returns
    -------
    tuple
        (Rperi,Rap,Zmax,err)
        Rperi,Rap,Zmax : numpy.ndarray, shape (len(R))
        err - non-zero if error occurred

    Notes
    -----
    - 2017-12-21 - Written - Bovy (UofT)
    """
    # Parse the potential
    from ..orbit.integrateFullOrbit import _parse_pot
    from ..orbit.integratePlanarOrbit import _prep_tfuncs

    npot, pot_type, pot_args, pot_tfuncs = _parse_pot(pot, potforactions=True)
    pot_tfuncs = _prep_tfuncs(pot_tfuncs)

    # Set up result arrays
    rperi = numpy.empty(len(R))
    rap = numpy.empty(len(R))
    zmax = numpy.empty(len(R))
    err = ctypes.c_int(0)

    # Set up the C code
    ndarrayFlags = ("C_CONTIGUOUS", "WRITEABLE")
    actionAngleAdiabatic_actionsFunc = _lib.actionAngleAdiabatic_RperiRapZmax
    actionAngleAdiabatic_actionsFunc.argtypes = [
        ctypes.c_int,
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_int,
        ndpointer(dtype=numpy.int32, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_void_p,
        ctypes.c_double,
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.POINTER(ctypes.c_int),
    ]

    # Array requirements, first store old order
    f_cont = [
        R.flags["F_CONTIGUOUS"],
        vR.flags["F_CONTIGUOUS"],
        vT.flags["F_CONTIGUOUS"],
        z.flags["F_CONTIGUOUS"],
        vz.flags["F_CONTIGUOUS"],
    ]
    R = numpy.require(R, dtype=numpy.float64, requirements=["C", "W"])
    vR = numpy.require(vR, dtype=numpy.float64, requirements=["C", "W"])
    vT = numpy.require(vT, dtype=numpy.float64, requirements=["C", "W"])
    z = numpy.require(z, dtype=numpy.float64, requirements=["C", "W"])
    vz = numpy.require(vz, dtype=numpy.float64, requirements=["C", "W"])
    rperi = numpy.require(rperi, dtype=numpy.float64, requirements=["C", "W"])
    rap = numpy.require(rap, dtype=numpy.float64, requirements=["C", "W"])
    zmax = numpy.require(zmax, dtype=numpy.float64, requirements=["C", "W"])

    # Run the C code
    actionAngleAdiabatic_actionsFunc(
        len(R),
        R,
        vR,
        vT,
        z,
        vz,
        ctypes.c_int(npot),
        pot_type,
        pot_args,
        pot_tfuncs,
        ctypes.c_double(gamma),
        rperi,
        rap,
        zmax,
        ctypes.byref(err),
    )

    # Reset input arrays
    if f_cont[0]:
        R = numpy.asfortranarray(R)
    if f_cont[1]:
        vR = numpy.asfortranarray(vR)
    if f_cont[2]:
        vT = numpy.asfortranarray(vT)
    if f_cont[3]:
        z = numpy.asfortranarray(z)
    if f_cont[4]:
        vz = numpy.asfortranarray(vz)

    return (rperi, rap, zmax, err.value)
