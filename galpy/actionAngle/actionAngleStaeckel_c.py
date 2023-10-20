import ctypes
import ctypes.util

import numpy
from numpy.ctypeslib import ndpointer

from ..util import _load_extension_libs, coords

_lib, _ext_loaded = _load_extension_libs.load_libgalpy()


def actionAngleStaeckel_c(pot, delta, R, vR, vT, z, vz, u0=None, order=10):
    """
    Use C to calculate actions using the Staeckel approximation

    Parameters
    ----------
    pot : Potential or list of such instances
        Potential
    delta : float
        Focal length of prolate spheroidal coordinates
    R : float
        Galactocentric radius
    vR : float
        Galactocentric radial velocity
    vT : float
        Galactocentric tangential velocity
    z : float
        Height
    vz : float
        Vertical velocity
    u0 : float, optional
        If set, u0 to use
    order : int, optional
        Order of Gauss-Legendre integration of the relevant integrals

    Returns
    -------
    tuple
        (jr,jz,err) where:
           * jr,jz : array, shape (len(R))
           * err - non-zero if error occurred

    Notes
    -----
    - 2012-12-01 - Written - Bovy (IAS)
    """
    if u0 is None:
        u0, dummy = coords.Rz_to_uv(R, z, delta=numpy.atleast_1d(delta))
    # Parse the potential
    from ..orbit.integrateFullOrbit import _parse_pot
    from ..orbit.integratePlanarOrbit import _prep_tfuncs

    npot, pot_type, pot_args, pot_tfuncs = _parse_pot(pot, potforactions=True)
    pot_tfuncs = _prep_tfuncs(pot_tfuncs)

    # Parse delta
    delta = numpy.atleast_1d(delta)
    ndelta = len(delta)

    # Set up result arrays
    jr = numpy.empty(len(R))
    jz = numpy.empty(len(R))
    err = ctypes.c_int(0)

    # Set up the C code
    ndarrayFlags = ("C_CONTIGUOUS", "WRITEABLE")
    actionAngleStaeckel_actionsFunc = _lib.actionAngleStaeckel_actions
    actionAngleStaeckel_actionsFunc.argtypes = [
        ctypes.c_int,
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_int,
        ndpointer(dtype=numpy.int32, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_void_p,
        ctypes.c_int,
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_int,
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
        u0.flags["F_CONTIGUOUS"],
        delta.flags["F_CONTIGUOUS"],
    ]
    R = numpy.require(R, dtype=numpy.float64, requirements=["C", "W"])
    vR = numpy.require(vR, dtype=numpy.float64, requirements=["C", "W"])
    vT = numpy.require(vT, dtype=numpy.float64, requirements=["C", "W"])
    z = numpy.require(z, dtype=numpy.float64, requirements=["C", "W"])
    vz = numpy.require(vz, dtype=numpy.float64, requirements=["C", "W"])
    u0 = numpy.require(u0, dtype=numpy.float64, requirements=["C", "W"])
    delta = numpy.require(delta, dtype=numpy.float64, requirements=["C", "W"])
    jr = numpy.require(jr, dtype=numpy.float64, requirements=["C", "W"])
    jz = numpy.require(jz, dtype=numpy.float64, requirements=["C", "W"])

    # Run the C code
    actionAngleStaeckel_actionsFunc(
        len(R),
        R,
        vR,
        vT,
        z,
        vz,
        u0,
        ctypes.c_int(npot),
        pot_type,
        pot_args,
        pot_tfuncs,
        ctypes.c_int(ndelta),
        delta,
        ctypes.c_int(order),
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
    if f_cont[5]:
        u0 = numpy.asfortranarray(u0)
    if f_cont[6]:
        delta = numpy.asfortranarray(delta)

    return (jr, jz, err.value)


def actionAngleStaeckel_calcu0(E, Lz, pot, delta):
    """
    Use C to calculate u0 in the Staeckel approximation

    Parameters
    ----------
    E : numpy.ndarray
        Energy.
    Lz : numpy.ndarray
        Angular momentum.
    pot : Potential or list of such instances
        Potential or list of such instances.
    delta : numpy.ndarray
        Focal length of prolate spheroidal coordinates.

    Returns
    -------
    tuple
        (u0,err)
        u0 : array, shape (len(E))
        err - non-zero if error occurred

    Notes
    -----
    - 2012-12-03 - Written - Bovy (IAS)
    """
    # Parse the potential
    from ..orbit.integrateFullOrbit import _parse_pot
    from ..orbit.integratePlanarOrbit import _prep_tfuncs

    npot, pot_type, pot_args, pot_tfuncs = _parse_pot(pot, potforactions=True)
    pot_tfuncs = _prep_tfuncs(pot_tfuncs)

    # Set up result arrays
    u0 = numpy.empty(len(E))
    err = ctypes.c_int(0)

    # Parse delta
    delta = numpy.atleast_1d(delta)
    ndelta = len(delta)

    # Set up the C code
    ndarrayFlags = ("C_CONTIGUOUS", "WRITEABLE")
    actionAngleStaeckel_actionsFunc = _lib.calcu0
    actionAngleStaeckel_actionsFunc.argtypes = [
        ctypes.c_int,
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_int,
        ndpointer(dtype=numpy.int32, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_void_p,
        ctypes.c_int,
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.POINTER(ctypes.c_int),
    ]

    # Array requirements, first store old order
    f_cont = [
        E.flags["F_CONTIGUOUS"],
        Lz.flags["F_CONTIGUOUS"],
        delta.flags["F_CONTIGUOUS"],
    ]
    E = numpy.require(E, dtype=numpy.float64, requirements=["C", "W"])
    Lz = numpy.require(Lz, dtype=numpy.float64, requirements=["C", "W"])
    delta = numpy.require(delta, dtype=numpy.float64, requirements=["C", "W"])
    u0 = numpy.require(u0, dtype=numpy.float64, requirements=["C", "W"])

    # Run the C code
    actionAngleStaeckel_actionsFunc(
        len(E),
        E,
        Lz,
        ctypes.c_int(npot),
        pot_type,
        pot_args,
        pot_tfuncs,
        ctypes.c_int(ndelta),
        delta,
        u0,
        ctypes.byref(err),
    )

    # Reset input arrays
    if f_cont[0]:
        E = numpy.asfortranarray(E)
    if f_cont[1]:
        Lz = numpy.asfortranarray(Lz)
    if f_cont[2]:
        delta = numpy.asfortranarray(delta)

    return (u0, err.value)


def actionAngleFreqStaeckel_c(pot, delta, R, vR, vT, z, vz, u0=None, order=10):
    """
    Use C to calculate actions and frequencies using the Staeckel approximation

    Parameters
    ----------
    pot : Potential or list of such instances
        Potential or list of such instances.
    delta : float
        Focal length of prolate spheroidal coordinates.
    R : float
        Galactocentric radius.
    vR : float
        Galactocentric radial velocity.
    vT : float
        Galactocentric tangential velocity.
    z : float
        Height.
    vz : float
        Vertical velocity.
    u0 : float, optional
        If set, u0 to use.
    order : int, optional
        Order of Gauss-Legendre integration of the relevant integrals.

    Returns
    -------
    tuple
        (jr,jz,Omegar,Omegaphi,Omegaz,err) where:
            * jr,jz,Omegar,Omegaphi,Omegaz : array, shape (len(R))
            * err - non-zero if error occurred

    Notes
    -----
    - 2012-12-01 - Written - Bovy (IAS)
    """
    if u0 is None:
        u0, dummy = coords.Rz_to_uv(R, z, delta=numpy.atleast_1d(delta))
    # Parse the potential
    from ..orbit.integrateFullOrbit import _parse_pot
    from ..orbit.integratePlanarOrbit import _prep_tfuncs

    npot, pot_type, pot_args, pot_tfuncs = _parse_pot(pot, potforactions=True)
    pot_tfuncs = _prep_tfuncs(pot_tfuncs)

    # Parse delta
    delta = numpy.atleast_1d(delta)
    ndelta = len(delta)

    # Set up result arrays
    jr = numpy.empty(len(R))
    jz = numpy.empty(len(R))
    Omegar = numpy.empty(len(R))
    Omegaphi = numpy.empty(len(R))
    Omegaz = numpy.empty(len(R))
    err = ctypes.c_int(0)

    # Set up the C code
    ndarrayFlags = ("C_CONTIGUOUS", "WRITEABLE")
    actionAngleStaeckel_actionsFunc = _lib.actionAngleStaeckel_actionsFreqs
    actionAngleStaeckel_actionsFunc.argtypes = [
        ctypes.c_int,
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_int,
        ndpointer(dtype=numpy.int32, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_void_p,
        ctypes.c_int,
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_int,
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
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
        u0.flags["F_CONTIGUOUS"],
        delta.flags["F_CONTIGUOUS"],
    ]
    R = numpy.require(R, dtype=numpy.float64, requirements=["C", "W"])
    vR = numpy.require(vR, dtype=numpy.float64, requirements=["C", "W"])
    vT = numpy.require(vT, dtype=numpy.float64, requirements=["C", "W"])
    z = numpy.require(z, dtype=numpy.float64, requirements=["C", "W"])
    vz = numpy.require(vz, dtype=numpy.float64, requirements=["C", "W"])
    u0 = numpy.require(u0, dtype=numpy.float64, requirements=["C", "W"])
    delta = numpy.require(delta, dtype=numpy.float64, requirements=["C", "W"])
    jr = numpy.require(jr, dtype=numpy.float64, requirements=["C", "W"])
    jz = numpy.require(jz, dtype=numpy.float64, requirements=["C", "W"])
    Omegar = numpy.require(Omegar, dtype=numpy.float64, requirements=["C", "W"])
    Omegaphi = numpy.require(Omegaphi, dtype=numpy.float64, requirements=["C", "W"])
    Omegaz = numpy.require(Omegaz, dtype=numpy.float64, requirements=["C", "W"])

    # Run the C code
    actionAngleStaeckel_actionsFunc(
        len(R),
        R,
        vR,
        vT,
        z,
        vz,
        u0,
        ctypes.c_int(npot),
        pot_type,
        pot_args,
        pot_tfuncs,
        ctypes.c_int(ndelta),
        delta,
        ctypes.c_int(order),
        jr,
        jz,
        Omegar,
        Omegaphi,
        Omegaz,
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
    if f_cont[5]:
        u0 = numpy.asfortranarray(u0)
    if f_cont[6]:
        delta = numpy.asfortranarray(delta)

    return (jr, jz, Omegar, Omegaphi, Omegaz, err.value)


def actionAngleFreqAngleStaeckel_c(
    pot, delta, R, vR, vT, z, vz, phi, u0=None, order=10
):
    """
    Use C to calculate actions, frequencies, and angles using the Staeckel approximation

    Parameters
    ----------
    pot : Potential or list of such instances
        Potential or list of such instances.
    delta : float
        Focal length of prolate spheroidal coordinates.
    R : float
        Galactocentric radius.
    vR : float
        Galactocentric radial velocity.
    vT : float
        Galactocentric tangential velocity.
    z : float
        Height.
    vz : float
        Vertical velocity.
    phi : float
        Azimuth.
    u0 : float, optional
        If set, u0 to use.
    order : int, optional
        Order of Gauss-Legendre integration of the relevant integrals.

    Returns
    -------
    tuple
        (jr,jz,Omegar,Omegaphi,Omegaz,Angler,Anglephi,Anglez,err) where:
            * jr,jz,Omegar,Omegaphi,Omegaz,Angler,Anglephi,Anglez : array, shape (len(R))
            * err - non-zero if error occurred

    Notes
    -----
    - 2013-08-27 - Written - Bovy (IAS)
    """
    if u0 is None:
        u0, dummy = coords.Rz_to_uv(R, z, delta=numpy.atleast_1d(delta))
    # Parse the potential
    from ..orbit.integrateFullOrbit import _parse_pot
    from ..orbit.integratePlanarOrbit import _prep_tfuncs

    npot, pot_type, pot_args, pot_tfuncs = _parse_pot(pot, potforactions=True)
    pot_tfuncs = _prep_tfuncs(pot_tfuncs)

    # Parse delta
    delta = numpy.atleast_1d(delta)
    ndelta = len(delta)

    # Set up result arrays
    jr = numpy.empty(len(R))
    jz = numpy.empty(len(R))
    Omegar = numpy.empty(len(R))
    Omegaphi = numpy.empty(len(R))
    Omegaz = numpy.empty(len(R))
    Angler = numpy.empty(len(R))
    Anglephi = numpy.empty(len(R))
    Anglez = numpy.empty(len(R))
    err = ctypes.c_int(0)

    # Set up the C code
    ndarrayFlags = ("C_CONTIGUOUS", "WRITEABLE")
    actionAngleStaeckel_actionsFunc = _lib.actionAngleStaeckel_actionsFreqsAngles
    actionAngleStaeckel_actionsFunc.argtypes = [
        ctypes.c_int,
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_int,
        ndpointer(dtype=numpy.int32, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_void_p,
        ctypes.c_int,
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_int,
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
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
        u0.flags["F_CONTIGUOUS"],
        delta.flags["F_CONTIGUOUS"],
    ]
    R = numpy.require(R, dtype=numpy.float64, requirements=["C", "W"])
    vR = numpy.require(vR, dtype=numpy.float64, requirements=["C", "W"])
    vT = numpy.require(vT, dtype=numpy.float64, requirements=["C", "W"])
    z = numpy.require(z, dtype=numpy.float64, requirements=["C", "W"])
    vz = numpy.require(vz, dtype=numpy.float64, requirements=["C", "W"])
    u0 = numpy.require(u0, dtype=numpy.float64, requirements=["C", "W"])
    delta = numpy.require(delta, dtype=numpy.float64, requirements=["C", "W"])
    jr = numpy.require(jr, dtype=numpy.float64, requirements=["C", "W"])
    jz = numpy.require(jz, dtype=numpy.float64, requirements=["C", "W"])
    Omegar = numpy.require(Omegar, dtype=numpy.float64, requirements=["C", "W"])
    Omegaphi = numpy.require(Omegaphi, dtype=numpy.float64, requirements=["C", "W"])
    Omegaz = numpy.require(Omegaz, dtype=numpy.float64, requirements=["C", "W"])
    Angler = numpy.require(Angler, dtype=numpy.float64, requirements=["C", "W"])
    Anglephi = numpy.require(Anglephi, dtype=numpy.float64, requirements=["C", "W"])
    Anglez = numpy.require(Anglez, dtype=numpy.float64, requirements=["C", "W"])

    # Run the C code
    actionAngleStaeckel_actionsFunc(
        len(R),
        R,
        vR,
        vT,
        z,
        vz,
        u0,
        ctypes.c_int(npot),
        pot_type,
        pot_args,
        pot_tfuncs,
        ctypes.c_int(ndelta),
        delta,
        ctypes.c_int(order),
        jr,
        jz,
        Omegar,
        Omegaphi,
        Omegaz,
        Angler,
        Anglephi,
        Anglez,
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
    if f_cont[5]:
        u0 = numpy.asfortranarray(u0)
    if f_cont[6]:
        delta = numpy.asfortranarray(delta)

    badAngle = Anglephi != 9999.99
    Anglephi[badAngle] = (Anglephi[badAngle] + phi[badAngle] % (2.0 * numpy.pi)) % (
        2.0 * numpy.pi
    )
    Anglephi[Anglephi < 0.0] += 2.0 * numpy.pi

    return (jr, jz, Omegar, Omegaphi, Omegaz, Angler, Anglephi, Anglez, err.value)


def actionAngleUminUmaxVminStaeckel_c(pot, delta, R, vR, vT, z, vz, u0=None):
    """
    Use C to calculate umin, umax, and vmin using the Staeckel approximation

    Parameters
    ----------
    pot : Potential or list of such instances
        Potential or list of such instances.
    delta : float
        Focal length of prolate spheroidal coordinates.
    R : float
        Galactocentric radius.
    vR : float
        Galactocentric radial velocity.
    vT : float
        Galactocentric tangential velocity.
    z : float
        Height.
    vz : float
        Vertical velocity.
    u0 : float, optional
        If set, u0 to use.

    Returns
    -------
    tuple
        (umin,umax,vmin,err) where:
            * umin,umax,vmin : array, shape (len(R))
            * err - non-zero if error occurred

    Notes
    -----
    - 2017-12-12 - Written - Bovy (UofT)
    """
    if u0 is None:
        u0, dummy = coords.Rz_to_uv(R, z, delta=numpy.atleast_1d(delta))
    # Parse the potential
    from ..orbit.integrateFullOrbit import _parse_pot
    from ..orbit.integratePlanarOrbit import _prep_tfuncs

    npot, pot_type, pot_args, pot_tfuncs = _parse_pot(pot, potforactions=True)
    pot_tfuncs = _prep_tfuncs(pot_tfuncs)

    # Parse delta
    delta = numpy.atleast_1d(delta)
    ndelta = len(delta)

    # Set up result arrays
    umin = numpy.empty(len(R))
    umax = numpy.empty(len(R))
    vmin = numpy.empty(len(R))
    err = ctypes.c_int(0)

    # Set up the C code
    ndarrayFlags = ("C_CONTIGUOUS", "WRITEABLE")
    actionAngleStaeckel_actionsFunc = _lib.actionAngleStaeckel_uminUmaxVmin
    actionAngleStaeckel_actionsFunc.argtypes = [
        ctypes.c_int,
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_int,
        ndpointer(dtype=numpy.int32, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_void_p,
        ctypes.c_int,
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
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
        u0.flags["F_CONTIGUOUS"],
        delta.flags["F_CONTIGUOUS"],
    ]
    R = numpy.require(R, dtype=numpy.float64, requirements=["C", "W"])
    vR = numpy.require(vR, dtype=numpy.float64, requirements=["C", "W"])
    vT = numpy.require(vT, dtype=numpy.float64, requirements=["C", "W"])
    z = numpy.require(z, dtype=numpy.float64, requirements=["C", "W"])
    vz = numpy.require(vz, dtype=numpy.float64, requirements=["C", "W"])
    u0 = numpy.require(u0, dtype=numpy.float64, requirements=["C", "W"])
    delta = numpy.require(delta, dtype=numpy.float64, requirements=["C", "W"])
    umin = numpy.require(umin, dtype=numpy.float64, requirements=["C", "W"])
    umax = numpy.require(umax, dtype=numpy.float64, requirements=["C", "W"])
    vmin = numpy.require(vmin, dtype=numpy.float64, requirements=["C", "W"])

    # Run the C code
    actionAngleStaeckel_actionsFunc(
        len(R),
        R,
        vR,
        vT,
        z,
        vz,
        u0,
        ctypes.c_int(npot),
        pot_type,
        pot_args,
        pot_tfuncs,
        ctypes.c_int(ndelta),
        delta,
        umin,
        umax,
        vmin,
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
    if f_cont[5]:
        u0 = numpy.asfortranarray(u0)
    if f_cont[6]:
        delta = numpy.asfortranarray(delta)

    return (umin, umax, vmin, err.value)
