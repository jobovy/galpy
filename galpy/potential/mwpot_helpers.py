# mwpot_helpers.py: auxiliary functions to help in setting up various
# Milky-Way-like potentials
# (for now, functions required to setup the Dehnen & Binney (1998),
#  Binney & Tremaine (2008), and McMillan (2017) potentials)
import numpy

from ..backend import promote_scalars
from ..backend._namespaces import namespace_from_arrays

# The densities follow their (R, z) data: numpy (as before) unless handed a
# backend array, e.g. by DiskSCFPotential's phi_ME density under a forced backend.


def _xp(R, z):
    xp = namespace_from_arrays((R, z)) or numpy
    if xp is not numpy:
        R, z = promote_scalars(xp, R, z)
    return xp, R, z


def _fabs(xp):
    return numpy.fabs if xp is numpy else xp.abs


def expexp_dens(R, z, Rd, zd, Sigma0):
    """rho(R,z) = Sigma_0/(2zd) exp(-|z|/zd-R/Rd)"""
    xp, R, z = _xp(R, z)
    return Sigma0 / (2 * zd) * xp.exp(-_fabs(xp)(z) / zd - R / Rd)


def expexp_dens_with_hole(R, z, Rd, Rm, zd, Sigma0):
    """rho(R,z) = Sigma0 / (4zd) exp(-Rm/R-R/Rd-|z|/zd)"""
    xp, R, z = _xp(R, z)
    if xp is numpy:
        if R == 0.0:
            return 0.0
        return Sigma0 / (2 * zd) * numpy.exp(-Rm / R - R / Rd - numpy.fabs(z) / zd)
    Rs = xp.where(R == 0.0, 1.0, R)  # safe dead branch for AD
    out = Sigma0 / (2 * zd) * xp.exp(-Rm / Rs - Rs / Rd - xp.abs(z) / zd)
    return xp.where(R == 0.0, 0.0, out)


def expsech2_dens_with_hole(R, z, Rd, Rm, zd, Sigma0):
    """rho(R,z) = Sigma0 / (4zd) exp(-Rm/R-R/Rd)*sech(z/[2zd])^2"""
    xp, R, z = _xp(R, z)
    if xp is numpy:
        if R == 0.0:
            return 0.0
        return (
            Sigma0
            / (4 * zd)
            * numpy.exp(-Rm / R - R / Rd)
            / numpy.cosh(z / (2 * zd)) ** 2
        )
    Rs = xp.where(R == 0.0, 1.0, R)  # safe dead branch for AD
    out = Sigma0 / (4 * zd) * xp.exp(-Rm / Rs - Rs / Rd) / xp.cosh(z / (2 * zd)) ** 2
    return xp.where(R == 0.0, 0.0, out)


def core_pow_dens_with_cut(R, z, alpha, r0, rcut, rho0, q):
    """rho(R,z) = rho0(1+r'/r0)^-alpha exp(-[r'/rcut]^2)
    r' = sqrt(R^2+z^2/q^2"""
    xp, R, z = _xp(R, z)
    rdash = xp.sqrt(R**2 + (z / q) ** 2)
    return rho0 / (1 + rdash / r0) ** alpha * xp.exp(-((rdash / rcut) ** 2))


def pow_dens_with_cut(R, z, alpha, r0, rcut, rho0, q):
    """rho(R,z) = rho0(1+r'/r0)^-alpha exp(-[r'/rcut]^2)
    r' = sqrt(R^2+z^2/q^2"""
    xp, R, z = _xp(R, z)
    rdash = xp.sqrt(R**2 + (z / q) ** 2)
    return rho0 / (rdash / r0) ** alpha * xp.exp(-((rdash / rcut) ** 2))
