# Cautun (2020) potential
# Thanks to Thomas Callingham (Durham University, UK) which implemented the potential within galpy
import numpy

from ..potential import (
    AdiabaticContractionWrapperPotential,
    DiskSCFPotential,
    NFWPotential,
    PowerSphericalPotentialwCutoff,
    SCFPotential,
    mwpot_helpers,
    scf_compute_coeffs_axi,
)
from ..util import conversion

# Suppress the numpy floating-point warnings that this code generates...
old_error_settings = numpy.seterr(all="ignore")
# Unit normalizations
ro = 8.122
vo = 229
sigo = conversion.surfdens_in_msolpc2(vo=vo, ro=ro)
rhoo = conversion.dens_in_msolpc3(vo=vo, ro=ro)
# Cautun DM halo
fb = 4.825 / 30.7  # Planck 1 baryon fraction
m200 = 0.969e12  # the DM halo mass
conc = 8.76
# Cautun Bulge
r0_bulge = 0.075 / ro
rcut_bulge = 2.1 / ro
rho0_bulge = 103 / rhoo
# Cautun Stellar Discs
zd_thin = 0.3 / ro
Rd_thin = 2.63 / ro
Sigma0_thin = 731.0 / sigo
zd_thick = 0.9 / ro
Rd_thick = 3.80 / ro
Sigma0_thick = 101.0 / sigo
# Cautun Gas Discs
Rd_HI = 7.0 / ro
Rm_HI = 4.0 / ro
zd_HI = 0.085 / ro
Sigma0_HI = 53 / sigo
Rd_H2 = 1.5 / ro
Rm_H2 = 12.0 / ro
zd_H2 = 0.045 / ro
Sigma0_H2 = 2200 / sigo
# Cautun CGM
A = 0.19
Beta = -1.46
critz0 = 127.5e-9 / rhoo
R200 = 219 / ro  # R200 for cgm
cgm_amp = 200 * critz0 * A * fb


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
        R, z, 1.8, r0_bulge, rcut_bulge, rho0_bulge, 0.5
    )


# dicts used in DiskSCFPotential
sigmadict = [
    {"type": "exp", "h": Rd_HI, "amp": Sigma0_HI, "Rhole": Rm_HI},
    {"type": "exp", "h": Rd_H2, "amp": Sigma0_H2, "Rhole": Rm_H2},
    {"type": "exp", "h": Rd_thin, "amp": Sigma0_thin, "Rhole": 0.0},
    {"type": "exp", "h": Rd_thick, "amp": Sigma0_thick, "Rhole": 0.0},
]

hzdict = [
    {"type": "sech2", "h": zd_HI},
    {"type": "sech2", "h": zd_H2},
    {"type": "exp", "h": zd_thin},
    {"type": "exp", "h": zd_thick},
]

# generate separate disk and halo potential - and combined potential
Cautun_bulge = SCFPotential(
    Acos=scf_compute_coeffs_axi(bulge_dens, 20, 10, a=0.1)[0], a=0.1, ro=ro, vo=vo
)
Cautun_cgm = PowerSphericalPotentialwCutoff(
    amp=cgm_amp, alpha=-Beta, r1=R200, rc=2.0 * R200, ro=ro, vo=vo
)
Cautun_disk = DiskSCFPotential(
    dens=lambda R, z: gas_dens(R, z) + stellar_dens(R, z),
    Sigma=sigmadict,
    hz=hzdict,
    a=2.5,
    N=30,
    L=30,
    ro=ro,
    vo=vo,
)
Cautun_halo = AdiabaticContractionWrapperPotential(
    pot=NFWPotential(
        conc=conc,
        mvir=m200 / 1.0e12,
        vo=vo,
        ro=ro,
        H=67.77,
        Om=0.307,
        overdens=200.0 * (1.0 - fb),
        wrtcrit=True,
    ),
    baryonpot=Cautun_bulge + Cautun_cgm + Cautun_disk,
    f_bar=fb,
    method="cautun",
    ro=ro,
    vo=vo,
)
Cautun20 = Cautun_halo + Cautun_disk + Cautun_bulge + Cautun_cgm
# Go back to old floating-point warnings settings
numpy.seterr(**old_error_settings)
