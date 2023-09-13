import ctypes
import ctypes.util

import numpy
from numpy.ctypeslib import ndpointer

from ..util import _load_extension_libs

_lib, _ext_loaded = _load_extension_libs.load_libgalpy_actionAngleTorus()


def actionAngleTorus_xvFreqs_c(pot, jr, jphi, jz, angler, anglephi, anglez, tol=0.003):
    """
    Compute configuration (x,v) and frequencies of a set of angles on a single torus

    Parameters
    ----------
    pot : Potential object or list thereof
    jr : float
        Radial action
    jphi : float
        Azimuthal action
    jz : float
        Vertical action
    angler : numpy.ndarray
        Radial angle
    anglephi : numpy.ndarray
        Azimuthal angle
    anglez : numpy.ndarray
        Vertical angle
    tol : float, optional
        Goal for |dJ|/|J| along the torus

    Returns
    -------
    tuple
        (R,vR,vT,z,vz,phi,Omegar,Omegaphi,Omegaz,flag)

    Notes
    -----
    - 2015-08-05/07 - Written - Bovy (UofT)
    """
    # Parse the potential
    from ..orbit.integrateFullOrbit import _parse_pot
    from ..orbit.integratePlanarOrbit import _prep_tfuncs

    npot, pot_type, pot_args, pot_tfuncs = _parse_pot(pot, potfortorus=True)
    pot_tfuncs = _prep_tfuncs(pot_tfuncs)

    # Set up result arrays
    R = numpy.empty(len(angler))
    vR = numpy.empty(len(angler))
    vT = numpy.empty(len(angler))
    z = numpy.empty(len(angler))
    vz = numpy.empty(len(angler))
    phi = numpy.empty(len(angler))
    Omegar = numpy.empty(1)
    Omegaphi = numpy.empty(1)
    Omegaz = numpy.empty(1)
    flag = ctypes.c_int(0)

    # Set up the C code
    ndarrayFlags = ("C_CONTIGUOUS", "WRITEABLE")
    actionAngleTorus_xvFreqsFunc = _lib.actionAngleTorus_xvFreqs
    actionAngleTorus_xvFreqsFunc.argtypes = [
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
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
        angler.flags["F_CONTIGUOUS"],
        anglephi.flags["F_CONTIGUOUS"],
        anglez.flags["F_CONTIGUOUS"],
    ]
    angler = numpy.require(angler, dtype=numpy.float64, requirements=["C", "W"])
    anglephi = numpy.require(anglephi, dtype=numpy.float64, requirements=["C", "W"])
    anglez = numpy.require(anglez, dtype=numpy.float64, requirements=["C", "W"])
    R = numpy.require(R, dtype=numpy.float64, requirements=["C", "W"])
    vR = numpy.require(vR, dtype=numpy.float64, requirements=["C", "W"])
    vT = numpy.require(vT, dtype=numpy.float64, requirements=["C", "W"])
    z = numpy.require(z, dtype=numpy.float64, requirements=["C", "W"])
    vz = numpy.require(vz, dtype=numpy.float64, requirements=["C", "W"])
    phi = numpy.require(phi, dtype=numpy.float64, requirements=["C", "W"])
    Omegar = numpy.require(Omegar, dtype=numpy.float64, requirements=["C", "W"])
    Omegaphi = numpy.require(Omegaphi, dtype=numpy.float64, requirements=["C", "W"])
    Omegaz = numpy.require(Omegaz, dtype=numpy.float64, requirements=["C", "W"])

    # Run the C code
    actionAngleTorus_xvFreqsFunc(
        ctypes.c_double(jr),
        ctypes.c_double(jphi),
        ctypes.c_double(jz),
        ctypes.c_int(len(angler)),
        angler,
        anglephi,
        anglez,
        ctypes.c_int(npot),
        pot_type,
        pot_args,
        pot_tfuncs,
        ctypes.c_double(tol),
        R,
        vR,
        vT,
        z,
        vz,
        phi,
        Omegar,
        Omegaphi,
        Omegaz,
        ctypes.byref(flag),
    )

    # Reset input arrays
    if f_cont[0]:
        angler = numpy.asfortranarray(angler)
    if f_cont[1]:
        anglephi = numpy.asfortranarray(anglephi)
    if f_cont[2]:
        anglez = numpy.asfortranarray(anglez)

    return (R, vR, vT, z, vz, phi, Omegar[0], Omegaphi[0], Omegaz[0], flag.value)


def actionAngleTorus_Freqs_c(pot, jr, jphi, jz, tol=0.003):
    """
    Compute frequencies on a single torus

    Parameters
    ----------
    pot : Potential object or list thereof
    jr : float
        Radial action
    jphi : float
        Azimuthal action
    jz : float
        Vertical action
    tol : float, optional
        Goal for |dJ|/|J| along the torus

    Returns
    -------
    tuple
        (Omegar,Omegaphi,Omegaz,flag)

    Notes
    -----
    - 2015-08-05/07 - Written - Bovy (UofT)
    """
    # Parse the potential
    from ..orbit.integrateFullOrbit import _parse_pot
    from ..orbit.integratePlanarOrbit import _prep_tfuncs

    npot, pot_type, pot_args, pot_tfuncs = _parse_pot(pot, potfortorus=True)
    pot_tfuncs = _prep_tfuncs(pot_tfuncs)

    # Set up result
    Omegar = numpy.empty(1)
    Omegaphi = numpy.empty(1)
    Omegaz = numpy.empty(1)
    flag = ctypes.c_int(0)

    # Set up the C code
    ndarrayFlags = ("C_CONTIGUOUS", "WRITEABLE")
    actionAngleTorus_FreqsFunc = _lib.actionAngleTorus_Freqs
    actionAngleTorus_FreqsFunc.argtypes = [
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
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

    # Array requirements
    Omegar = numpy.require(Omegar, dtype=numpy.float64, requirements=["C", "W"])
    Omegaphi = numpy.require(Omegaphi, dtype=numpy.float64, requirements=["C", "W"])
    Omegaz = numpy.require(Omegaz, dtype=numpy.float64, requirements=["C", "W"])

    # Run the C code
    actionAngleTorus_FreqsFunc(
        ctypes.c_double(jr),
        ctypes.c_double(jphi),
        ctypes.c_double(jz),
        ctypes.c_int(npot),
        pot_type,
        pot_args,
        pot_tfuncs,
        ctypes.c_double(tol),
        Omegar,
        Omegaphi,
        Omegaz,
        ctypes.byref(flag),
    )

    return (Omegar[0], Omegaphi[0], Omegaz[0], flag.value)


def actionAngleTorus_hessian_c(pot, jr, jphi, jz, tol=0.003, dJ=0.001):
    """
    Compute dO/dJ on a single torus

    Parameters
    ----------
    pot : Potential object or list thereof
    jr : float
        Radial action
    jphi : float
        Azimuthal action
    jz : float
        Vertical action
    tol : float, optional
        Goal for |dJ|/|J| along the torus
    dJ : float, optional
        Action difference when computing derivatives (Hessian or Jacobian)

    Returns
    -------
    tuple
        (dO/dJ,Omegar,Omegaphi,Omegaz,flag)

    Notes
    -----
    - 2016-07-15 - Written - Bovy (UofT)
    """
    # Parse the potential
    from ..orbit.integrateFullOrbit import _parse_pot
    from ..orbit.integratePlanarOrbit import _prep_tfuncs

    npot, pot_type, pot_args, pot_tfuncs = _parse_pot(pot, potfortorus=True)
    pot_tfuncs = _prep_tfuncs(pot_tfuncs)

    # Set up result
    dOdJT = numpy.empty(9)
    Omegar = numpy.empty(1)
    Omegaphi = numpy.empty(1)
    Omegaz = numpy.empty(1)
    flag = ctypes.c_int(0)

    # Set up the C code
    ndarrayFlags = ("C_CONTIGUOUS", "WRITEABLE")
    actionAngleTorus_HessFunc = _lib.actionAngleTorus_hessianFreqs
    actionAngleTorus_HessFunc.argtypes = [
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ndpointer(dtype=numpy.int32, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.c_double,
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.POINTER(ctypes.c_int),
    ]

    # Array requirements
    dOdJT = numpy.require(dOdJT, dtype=numpy.float64, requirements=["C", "W"])
    Omegar = numpy.require(Omegar, dtype=numpy.float64, requirements=["C", "W"])
    Omegaphi = numpy.require(Omegaphi, dtype=numpy.float64, requirements=["C", "W"])
    Omegaz = numpy.require(Omegaz, dtype=numpy.float64, requirements=["C", "W"])

    # Run the C code
    actionAngleTorus_HessFunc(
        ctypes.c_double(jr),
        ctypes.c_double(jphi),
        ctypes.c_double(jz),
        ctypes.c_int(npot),
        pot_type,
        pot_args,
        pot_tfuncs,
        ctypes.c_double(tol),
        ctypes.c_double(dJ),
        dOdJT,
        Omegar,
        Omegaphi,
        Omegaz,
        ctypes.byref(flag),
    )

    return (dOdJT.reshape((3, 3)).T, Omegar[0], Omegaphi[0], Omegaz[0], flag.value)


def actionAngleTorus_jacobian_c(
    pot, jr, jphi, jz, angler, anglephi, anglez, tol=0.003, dJ=0.001
):
    """
    Compute d(x,v)/d(J,theta) on a single torus, also compute dO/dJ and the frequencies

    Parameters
    ----------
    pot : Potential object or list thereof
    jr : float
        Radial action
    jphi : float
        Azimuthal action
    jz : float
        Vertical action
    angler : numpy.ndarray
        Radial angle
    anglephi : numpy.ndarray
        Azimuthal angle
    anglez : numpy.ndarray
        Vertical angle
    tol : float, optional
        Goal for |dJ|/|J| along the torus
    dJ : float, optional
        Action difference when computing derivatives (Hessian or Jacobian)

    Returns
    -------
    tuple
        (d[R,vR,vT,z,vz,phi]/d[J,theta],Omegar,Omegaphi,Omegaz,Autofit error message)

    Notes
    -----
    - 2016-07-19 - Written - Bovy (UofT)
    """
    # Parse the potential
    from ..orbit.integrateFullOrbit import _parse_pot
    from ..orbit.integratePlanarOrbit import _prep_tfuncs

    npot, pot_type, pot_args, pot_tfuncs = _parse_pot(pot, potfortorus=True)
    pot_tfuncs = _prep_tfuncs(pot_tfuncs)

    # Set up result
    R = numpy.empty(len(angler))
    vR = numpy.empty(len(angler))
    vT = numpy.empty(len(angler))
    z = numpy.empty(len(angler))
    vz = numpy.empty(len(angler))
    phi = numpy.empty(len(angler))
    dxvOdJaT = numpy.empty(36 * len(angler))
    dOdJT = numpy.empty(9)
    Omegar = numpy.empty(1)
    Omegaphi = numpy.empty(1)
    Omegaz = numpy.empty(1)
    flag = ctypes.c_int(0)

    # Set up the C code
    ndarrayFlags = ("C_CONTIGUOUS", "WRITEABLE")
    actionAngleTorus_JacFunc = _lib.actionAngleTorus_jacobianFreqs
    actionAngleTorus_JacFunc.argtypes = [
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_int,
        ndpointer(dtype=numpy.int32, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.c_double,
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
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
        angler.flags["F_CONTIGUOUS"],
        anglephi.flags["F_CONTIGUOUS"],
        anglez.flags["F_CONTIGUOUS"],
    ]
    angler = numpy.require(angler, dtype=numpy.float64, requirements=["C", "W"])
    anglephi = numpy.require(anglephi, dtype=numpy.float64, requirements=["C", "W"])
    anglez = numpy.require(anglez, dtype=numpy.float64, requirements=["C", "W"])
    R = numpy.require(R, dtype=numpy.float64, requirements=["C", "W"])
    vR = numpy.require(vR, dtype=numpy.float64, requirements=["C", "W"])
    vT = numpy.require(vT, dtype=numpy.float64, requirements=["C", "W"])
    z = numpy.require(z, dtype=numpy.float64, requirements=["C", "W"])
    vz = numpy.require(vz, dtype=numpy.float64, requirements=["C", "W"])
    phi = numpy.require(phi, dtype=numpy.float64, requirements=["C", "W"])
    dxvOdJaT = numpy.require(dxvOdJaT, dtype=numpy.float64, requirements=["C", "W"])
    dOdJT = numpy.require(dOdJT, dtype=numpy.float64, requirements=["C", "W"])
    Omegar = numpy.require(Omegar, dtype=numpy.float64, requirements=["C", "W"])
    Omegaphi = numpy.require(Omegaphi, dtype=numpy.float64, requirements=["C", "W"])
    Omegaz = numpy.require(Omegaz, dtype=numpy.float64, requirements=["C", "W"])

    # Run the C code
    actionAngleTorus_JacFunc(
        ctypes.c_double(jr),
        ctypes.c_double(jphi),
        ctypes.c_double(jz),
        ctypes.c_int(len(angler)),
        angler,
        anglephi,
        anglez,
        ctypes.c_int(npot),
        pot_type,
        pot_args,
        pot_tfuncs,
        ctypes.c_double(tol),
        ctypes.c_double(dJ),
        R,
        vR,
        vT,
        z,
        vz,
        phi,
        dxvOdJaT,
        dOdJT,
        Omegar,
        Omegaphi,
        Omegaz,
        ctypes.byref(flag),
    )

    # Reset input arrays
    if f_cont[0]:
        angler = numpy.asfortranarray(angler)
    if f_cont[1]:
        anglephi = numpy.asfortranarray(anglephi)
    if f_cont[2]:
        anglez = numpy.asfortranarray(anglez)

    dxvOdJaT = numpy.reshape(dxvOdJaT, (len(angler), 6, 6), order="C")
    dxvOdJa = numpy.swapaxes(dxvOdJaT, 1, 2)

    return (
        R,
        vR,
        vT,
        z,
        vz,
        phi,
        dxvOdJa,
        dOdJT.reshape((3, 3)).T,
        Omegar[0],
        Omegaphi[0],
        Omegaz[0],
        flag.value,
    )
