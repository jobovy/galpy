# Milky-Way mass models from Irrgang et al. (2013)
import numpy

from ..potential import (
    MiyamotoNagaiPotential,
    NFWPotential,
    PlummerPotential,
    SCFPotential,
    scf_compute_coeffs_spherical,
)
from ..util import conversion

# Their mass unit
mgal_in_msun = 1e5 / conversion._G
# Model I: updated version of Allen & Santillan
# Unit normalizations
ro, vo = 8.4, 242.0
Irrgang13I_bulge = PlummerPotential(
    amp=409.0 * mgal_in_msun / conversion.mass_in_msol(vo, ro),
    b=0.23 / ro,
    ro=ro,
    vo=vo,
)
Irrgang13I_disk = MiyamotoNagaiPotential(
    amp=2856.0 * mgal_in_msun / conversion.mass_in_msol(vo, ro),
    a=4.22 / ro,
    b=0.292 / ro,
    ro=ro,
    vo=vo,
)


# The halo is a little more difficult, because the Irrgang13I halo model is
# not in galpy, so we use SCF to represent it (because we're lazy...). The
# sharp cut-off in the Irrgang13I halo model makes SCF difficult, so we
# replace it with a smooth cut-off; this only affects the very outer halo
def Irrgang13I_halo_dens(
    r,
    amp=1018 * mgal_in_msun / conversion.mass_in_msol(vo, ro),
    ah=2.562 / ro,
    gamma=2.0,
    Lambda=200.0 / ro,
):
    r_over_ah_gamma = (r / ah) ** (gamma - 1.0)
    return (
        amp
        / 4.0
        / numpy.pi
        / ah
        * r_over_ah_gamma
        * (r_over_ah_gamma + gamma)
        / r**2
        / (1.0 + r_over_ah_gamma) ** 2.0
        * ((1.0 - numpy.tanh((r - Lambda) / (Lambda / 20.0))) / 2.0)
    )


a_for_scf = 20.0
# scf_compute_coeffs_spherical currently seems to require a function of 3 parameters...
Acos = scf_compute_coeffs_spherical(
    lambda r, z, p: Irrgang13I_halo_dens(r), 40, a=a_for_scf
)[0]
Irrgang13I_halo = SCFPotential(Acos=Acos, a=a_for_scf, ro=ro, vo=vo)
# Final model I
Irrgang13I = Irrgang13I_bulge + Irrgang13I_disk + Irrgang13I_halo

# Model II
# Unit normalizations
ro, vo = 8.35, 240.4
Irrgang13II_bulge = PlummerPotential(
    amp=175.0 * mgal_in_msun / conversion.mass_in_msol(vo, ro),
    b=0.184 / ro,
    ro=ro,
    vo=vo,
)
Irrgang13II_disk = MiyamotoNagaiPotential(
    amp=2829.0 * mgal_in_msun / conversion.mass_in_msol(vo, ro),
    a=4.85 / ro,
    b=0.305 / ro,
    ro=ro,
    vo=vo,
)


# Again use SCF because the Irrgang13II halo model is not in galpy; because
# the halo model is quite different from Hernquist both in the inner and outer
# part, need quite a few basis functions...
def Irrgang13II_halo_dens(
    r, amp=69725 * mgal_in_msun / conversion.mass_in_msol(vo, ro), ah=200.0 / ro
):
    return amp / 4.0 / numpy.pi * ah**2.0 / r**2.0 / (r**2.0 + ah**2.0) ** 1.5


a_for_scf = 0.15
# scf_compute_coeffs_spherical currently seems to require a function of 3 parameters...
Acos = scf_compute_coeffs_spherical(
    lambda r, z, p: Irrgang13II_halo_dens(r), 75, a=a_for_scf
)[0]
Irrgang13II_halo = SCFPotential(Acos=Acos, a=a_for_scf, ro=ro, vo=vo)
# Final model II
Irrgang13II = Irrgang13II_bulge + Irrgang13II_disk + Irrgang13II_halo

# Model III
# Unit normalizations
ro, vo = 8.33, 239.7
Irrgang13III_bulge = PlummerPotential(
    amp=439.0 * mgal_in_msun / conversion.mass_in_msol(vo, ro),
    b=0.236 / ro,
    ro=ro,
    vo=vo,
)
Irrgang13III_disk = MiyamotoNagaiPotential(
    amp=3096.0 * mgal_in_msun / conversion.mass_in_msol(vo, ro),
    a=3.262 / ro,
    b=0.289 / ro,
    ro=ro,
    vo=vo,
)
Irrgang13III_halo = NFWPotential(
    amp=142200.0 * mgal_in_msun / conversion.mass_in_msol(vo, ro),
    a=45.02 / ro,
    ro=ro,
    vo=vo,
)
# Final model III
Irrgang13III = Irrgang13III_bulge + Irrgang13III_disk + Irrgang13III_halo
