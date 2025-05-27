# mwpot_helpers.py: auxiliary functions to help in setting up various
# Milky-Way-like potentials
# (for now, functions required to setup the Dehnen & Binney (1998),
#  Binney & Tremaine (2008), and McMillan (2017) potentials)
import numpy


def expexp_dens(R, z, Rd, zd, Sigma0):
    """rho(R,z) = Sigma_0/(2zd) exp(-|z|/zd-R/Rd)"""
    return Sigma0 / (2 * zd) * numpy.exp(-numpy.fabs(z) / zd - R / Rd)


def expexp_dens_with_hole(R, z, Rd, Rm, zd, Sigma0):
    """rho(R,z) = Sigma0 / (4zd) exp(-Rm/R-R/Rd-|z|/zd)"""
    if R == 0.0:
        return 0.0
    return Sigma0 / (2 * zd) * numpy.exp(-Rm / R - R / Rd - numpy.fabs(z) / zd)


def expsech2_dens_with_hole(R, z, Rd, Rm, zd, Sigma0):
    """rho(R,z) = Sigma0 / (4zd) exp(-Rm/R-R/Rd)*sech(z/[2zd])^2"""
    if R == 0.0:
        return 0.0
    return (
        Sigma0 / (4 * zd) * numpy.exp(-Rm / R - R / Rd) / numpy.cosh(z / (2 * zd)) ** 2
    )


def core_pow_dens_with_cut(R, z, alpha, r0, rcut, rho0, q):
    """rho(R,z) = rho0(1+r'/r0)^-alpha exp(-[r'/rcut]^2)
    r' = sqrt(R^2+z^2/q^2"""
    rdash = numpy.sqrt(R**2 + (z / q) ** 2)
    return rho0 / (1 + rdash / r0) ** alpha * numpy.exp(-((rdash / rcut) ** 2))


def pow_dens_with_cut(R, z, alpha, r0, rcut, rho0, q):
    """rho(R,z) = rho0(1+r'/r0)^-alpha exp(-[r'/rcut]^2)
    r' = sqrt(R^2+z^2/q^2"""
    rdash = numpy.sqrt(R**2 + (z / q) ** 2)
    return rho0 / (rdash / r0) ** alpha * numpy.exp(-((rdash / rcut) ** 2))
