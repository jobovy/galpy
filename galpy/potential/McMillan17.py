# McMillan (2017) potential as first implemented in the galpy framework by
# Mackereth & Bovy (2018)
import numpy

from ..potential import (
    DiskSCFPotential,
    NFWPotential,
    SCFPotential,
    mwpot_helpers,
    scf_compute_coeffs_axi,
)
from ..util import conversion

# Suppress the numpy floating-point warnings that this code generates...
old_error_settings = numpy.seterr(all="ignore")
# Unit normalizations
ro = 8.21
vo = 233.1
sigo = conversion.surfdens_in_msolpc2(vo=vo, ro=ro)
rhoo = conversion.dens_in_msolpc3(vo=vo, ro=ro)
# gas disk parameters (fixed in McMillan 2017...)
Rd_HI = 7.0 / ro
Rm_HI = 4.0 / ro
zd_HI = 0.085 / ro
Sigma0_HI = 53.1 / sigo
Rd_H2 = 1.5 / ro
Rm_H2 = 12.0 / ro
zd_H2 = 0.045 / ro
Sigma0_H2 = 2180.0 / sigo
# parameters of best-fitting model in McMillan (2017)
# stellar disks
Sigma0_thin = 896.0 / sigo
Rd_thin = 2.5 / ro
zd_thin = 0.3 / ro
Sigma0_thick = 183.0 / sigo
Rd_thick = 3.02 / ro
zd_thick = 0.9 / ro
# bulge
rho0_bulge = 98.4 / rhoo
r0_bulge = 0.075 / ro
rcut = 2.1 / ro
# DM halo
rho0_halo = 0.00854 / rhoo
rh = 19.6 / ro


def gas_dens(R, z):
    return mwpot_helpers.expsech2_dens_with_hole(
        R, z, Rd_HI, Rm_HI, zd_HI, Sigma0_HI
    ) + mwpot_helpers.expsech2_dens_with_hole(R, z, Rd_H2, Rm_H2, zd_H2, Sigma0_H2)


def stellar_dens(R, z):
    return mwpot_helpers.expexp_dens(
        R, z, Rd_thin, zd_thin, Sigma0_thin
    ) + mwpot_helpers.expexp_dens(R, z, Rd_thick, zd_thick, Sigma0_thick)


def bulge_dens(R, z):
    return mwpot_helpers.core_pow_dens_with_cut(
        R, z, 1.8, r0_bulge, rcut, rho0_bulge, 0.5
    )


# dicts used in DiskSCFPotential
sigmadict = [
    {"type": "exp", "h": Rd_HI, "amp": Sigma0_HI, "Rhole": Rm_HI},
    {"type": "exp", "h": Rd_H2, "amp": Sigma0_H2, "Rhole": Rm_H2},
    {"type": "exp", "h": Rd_thin, "amp": Sigma0_thin},
    {"type": "exp", "h": Rd_thick, "amp": Sigma0_thick},
]

hzdict = [
    {"type": "sech2", "h": zd_HI},
    {"type": "sech2", "h": zd_H2},
    {"type": "exp", "h": zd_thin},
    {"type": "exp", "h": zd_thick},
]

# generate separate disk and halo potential - and combined potential
McMillan_bulge = SCFPotential(
    Acos=scf_compute_coeffs_axi(bulge_dens, 20, 10, a=0.1)[0], a=0.1, ro=ro, vo=vo
)
McMillan_disk = DiskSCFPotential(
    dens=lambda R, z: gas_dens(R, z) + stellar_dens(R, z),
    Sigma=sigmadict,
    hz=hzdict,
    a=2.5,
    N=30,
    L=30,
    ro=ro,
    vo=vo,
)
McMillan_halo = NFWPotential(amp=rho0_halo * (4 * numpy.pi * rh**3), a=rh, ro=ro, vo=vo)
# Go back to old floating-point warnings settings
numpy.seterr(**old_error_settings)
McMillan17 = McMillan_disk + McMillan_halo + McMillan_bulge
