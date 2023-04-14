# Dehnen & Binney (1998) potentials: models 1 through 4
import numpy

from ..potential import (
    DiskSCFPotential,
    SCFPotential,
    TwoPowerTriaxialPotential,
    mwpot_helpers,
    scf_compute_coeffs_axi,
)
from ..util import conversion

# Unit normalization, all models have R0 = 8 kpc
ro = 8.0
# Parameters in common between all models
Rm_ISM = 4.0 / ro
zd_ISM = 0.040 / ro
zd_thin = 0.180 / ro
zd_thick = 1.0 / ro
Rd_ISM_over_Rd = 2.0
Rd_thin_over_Rd = 1.0
Rd_thick_over_Rd = 1.0
SigmaR0_ISM_over_SigmaR0 = 0.25
SigmaR0_thin_over_SigmaR0 = 0.70
SigmaR0_thick_over_SigmaR0 = 0.05
alpha_bulge = 1.8
q_bulge = 0.6
r0_bulge = 1.0 / ro
rc_bulge = 1.9 / ro
q_halo = 0.8


def define_dehnenbinney98_models(model=1):
    # Suppress the numpy floating-point warnings that this code generates...
    old_error_settings = numpy.seterr(all="ignore")
    if model == 1:
        vo = 222.0
        Rd_star = 0.25
        Sigma0 = 1905.0
        rho0_bulge = 0.4271
        rho0_halo = 0.7110
        alpha_halo = -2.0
        beta_halo = 2.959
        r0_halo = 3.83 / ro
    elif model == 2:
        vo = 217.0
        Rd_star = 0.30
        Sigma0 = 1208.0
        rho0_bulge = 0.7561
        rho0_halo = 1.263
        alpha_halo = -2.0
        beta_halo = 2.207
        r0_halo = 1.09 / ro
    elif model == 3:
        vo = 217.0
        Rd_star = 0.35
        Sigma0 = 778.4
        rho0_bulge = 0.3
        rho0_halo = 0.1179
        alpha_halo = 1.8
        beta_halo = 2.002
        r0_halo = 2.29 / ro
    elif model == 4:
        vo = 220.0
        Rd_star = 0.40
        Sigma0 = 536.0
        rho0_bulge = 0.3
        rho0_halo = 0.2659
        alpha_halo = 1.629
        beta_halo = 2.167
        r0_halo = 1.899 / ro
    sigo = conversion.surfdens_in_msolpc2(vo=vo, ro=ro)
    rhoo = conversion.dens_in_msolpc3(vo=vo, ro=ro)
    Sigma0 /= sigo
    rho0_bulge /= rhoo
    rho0_halo /= rhoo
    Rd_ISM = Rd_star * Rd_ISM_over_Rd
    Rd_thin = Rd_star * Rd_thin_over_Rd
    Rd_thick = Rd_star * Rd_thick_over_Rd
    SigmaR0 = Sigma0 / (
        SigmaR0_ISM_over_SigmaR0
        / (2.0 * zd_ISM)
        / mwpot_helpers.expexp_dens_with_hole(1.0, 0.0, Rd_ISM, Rm_ISM, zd_ISM, 1.0)
        + SigmaR0_thin_over_SigmaR0
        * mwpot_helpers.expexp_dens(0.0, 0.0, Rd_thin, zd_thin, 1.0)
        / mwpot_helpers.expexp_dens(1.0, 0.0, Rd_thin, zd_thin, 1.0)
        + SigmaR0_thick_over_SigmaR0
        * mwpot_helpers.expexp_dens(0.0, 0.0, Rd_thick, zd_thick, 1.0)
        / mwpot_helpers.expexp_dens(1.0, 0.0, Rd_thick, zd_thick, 1.0)
    )
    Sigma0_ISM = (
        SigmaR0_ISM_over_SigmaR0
        * SigmaR0
        / (2.0 * zd_ISM)
        / mwpot_helpers.expexp_dens_with_hole(1.0, 0.0, Rd_ISM, Rm_ISM, zd_ISM, 1.0)
    )
    Sigma0_thin = (
        SigmaR0_thin_over_SigmaR0
        * SigmaR0
        * mwpot_helpers.expexp_dens(0.0, 0.0, Rd_thin, zd_thin, 1.0)
        / mwpot_helpers.expexp_dens(1.0, 0.0, Rd_thin, zd_thin, 1.0)
    )
    Sigma0_thick = (
        SigmaR0_thick_over_SigmaR0
        * SigmaR0
        * mwpot_helpers.expexp_dens(0.0, 0.0, Rd_thick, zd_thick, 1.0)
        / mwpot_helpers.expexp_dens(1.0, 0.0, Rd_thick, zd_thick, 1.0)
    )
    # now define gas and disk functions for DiskSCF
    gas_dens = lambda R, z: mwpot_helpers.expexp_dens_with_hole(
        R, z, Rd_ISM, Rm_ISM, zd_ISM, Sigma0_ISM
    )
    disk_dens = lambda R, z: mwpot_helpers.expexp_dens(
        R, z, Rd_thin, zd_thin, Sigma0_thin
    ) + mwpot_helpers.expexp_dens(R, z, Rd_thick, zd_thick, Sigma0_thick)
    bulge_dens = lambda R, z: mwpot_helpers.pow_dens_with_cut(
        R, z, alpha_bulge, r0_bulge, rc_bulge, rho0_bulge, q_bulge
    )
    # dicts used in DiskSCFPotential
    sigmadict = [
        {"type": "expwhole", "h": Rd_ISM, "amp": Sigma0_ISM, "Rhole": Rm_ISM},
        {"type": "exp", "h": Rd_thin, "amp": Sigma0_thin},
        {"type": "exp", "h": Rd_thick, "amp": Sigma0_thick},
    ]
    hzdict = [
        {"type": "exp", "h": zd_ISM},
        {"type": "exp", "h": zd_thin},
        {"type": "exp", "h": zd_thick},
    ]
    # Now put together the potential
    DB98_bulge = SCFPotential(
        Acos=scf_compute_coeffs_axi(bulge_dens, 30, 10, a=0.025)[0],
        a=0.025,
        ro=ro,
        vo=vo,
    )
    DB98_disk = DiskSCFPotential(
        dens=lambda R, z: gas_dens(R, z) + disk_dens(R, z),
        Sigma=sigmadict,
        hz=hzdict,
        a=2.5,
        N=30,
        L=30,
        ro=ro,
        vo=vo,
    )
    DB98_halo = TwoPowerTriaxialPotential(
        amp=rho0_halo * (4 * numpy.pi * r0_halo**3),
        alpha=alpha_halo,
        beta=beta_halo,
        a=r0_halo,
        ro=ro,
        vo=vo,
        c=q_halo,
    )
    # Go back to old floating-point warnings settings
    numpy.seterr(**old_error_settings)
    return DB98_bulge + DB98_disk + DB98_halo
