# Cautun (2020) potential 
# Thanks to Thomas Callingham (Durham University, UK) which implemented the potential within galpy

import numpy as np
from galpy.potential import NFWPotential
from galpy.potential import DiskSCFPotential
from galpy.potential import SCFPotential
from galpy.potential import PowerSphericalPotentialwCutoff
from galpy.potential import interpSphericalPotential
from galpy.potential import scf_compute_coeffs_axi
from galpy.potential import scf_compute_coeffs_spherical
from galpy.potential import mwpot_helpers
from galpy.util import conversion
# Suppress the np floating-point warnings that this code generates...
old_error_settings= np.seterr(all='ignore')




ro= 8.122
vo= 229
sigo = conversion.surfdens_in_msolpc2(vo=vo,ro=ro)
rhoo = conversion.dens_in_msolpc3(vo=vo,ro=ro)

#Cautun DM halo
fb   = 4.825 / 30.7 # Planck 1 baryon fraction
m200 = 0.97e12  # the DM halo mass
conc = 9.4

#Cautun Bulge
r0_bulge  = 0.075/ro
rcut_bulge= 2.1/ro
rho0_bulge= 103/rhoo

#Cautun Stellar Discs
zd_thin    = 0.3/ro
Rd_thin    =2.63/ro
Sigma0_thin= 731./sigo
zd_thick    = 0.9/ro
Rd_thick    = 3.80/ro
Sigma0_thick= 101./sigo

#Cautun Gas Discs
Rd_HI= 7./ro
Rm_HI= 4./ro
zd_HI= 0.085/ro
Sigma0_HI= 53/sigo
Rd_H2= 1.5/ro
Rm_H2= 12./ro
zd_H2= 0.045/ro
Sigma0_H2= 2200/sigo

# Cautun CGM
A = 0.19
Beta = -1.46
critz0 = 127.5e-9/rhoo
R200   = 219/ro #R200 for cgm
cgm_amp = 200 * critz0 * A * fb


def gas_dens(R,z):
    return mwpot_helpers.expsech2_dens_with_hole(R,z,Rd_HI,Rm_HI,zd_HI,Sigma0_HI) + mwpot_helpers.expsech2_dens_with_hole(R,z,Rd_H2,Rm_H2,zd_H2,Sigma0_H2)

def stellar_dens(R,z):
    return mwpot_helpers.expexp_dens(R,z,Rd_thin,zd_thin,Sigma0_thin) + mwpot_helpers.expexp_dens(R,z,Rd_thick,zd_thick,Sigma0_thick)

def bulge_dens(R,z):
    return mwpot_helpers.core_pow_dens_with_cut(R,z,1.8,r0_bulge,rcut_bulge,
                                                rho0_bulge,0.5)


#dicts used in DiskSCFPotential 
sigmadict = [{'type':'exp','h':Rd_HI,'amp':Sigma0_HI, 'Rhole':Rm_HI},
             {'type':'exp','h':Rd_H2,'amp':Sigma0_H2, 'Rhole':Rm_H2},
             {'type':'exp','h':Rd_thin,'amp':Sigma0_thin, 'Rhole':0.},
             {'type':'exp','h':Rd_thick,'amp':Sigma0_thick, 'Rhole':0.}]

hzdict = [{'type':'sech2', 'h':zd_HI},
          {'type':'sech2', 'h':zd_H2},
          {'type':'exp', 'h':zd_thin},
          {'type':'exp', 'h':zd_thick}]


#generate separate disk and halo potential - and combined potential
Cautun_bulge= SCFPotential(\
    Acos=scf_compute_coeffs_axi(bulge_dens,20,10,a=0.1)[0], a=0.1, ro=ro, vo=vo )

Cautun_cgm= PowerSphericalPotentialwCutoff( amp=cgm_amp, alpha=-Beta,\
    r1=R200, rc=2.*R200, ro=ro, vo=vo )

Cautun_disk= DiskSCFPotential( dens=lambda R,z: gas_dens(R,z) + stellar_dens(R,z), Sigma=sigmadict, \
                              hz=hzdict, a=2.5, N=30, L=30, ro=ro, vo=vo )


Cautun_halo_unContracted = NFWPotential( conc=conc, mvir=m200/1.e12, vo=vo, ro=ro, H=67.77, Om=0.307, overdens=200.0 * (1.-fb), wrtcrit=True )


# functions for calculating the contraction of the DM halo given the baryonic mass distribution
def Baryon_rforce(R,z):
    return Cautun_bulge.rforce(R,z) + Cautun_disk.rforce(R,z)

def Baryon_enclosed_mass(r,G_Newton):
    from scipy import integrate
    _mass = np.empty_like(r)
    _I = lambda theta, ri: Baryon_rforce( ri*np.sin(theta), ri*np.cos(theta) ) * np.sin(theta)
    for i in range( len(r) ):
        _mass[i] = integrate.quad( _I, 0., np.pi/2, args=(r[i],), epsabs=0., epsrel=1.e-4 )[0] * r[i]**2 / (-G_Newton)
    return _mass + Cautun_cgm.mass(r)


def contract_factor_enclosed_mass( mass_DM, mass_bar, f_bar=0.157 ):
    """ Returns the contracted DM enclosed mass given the 'uncontracted' profile and that of the baryonic distribution.
   
   Args:
      mass_DM       : enclosed mass in the DM component in the absence of baryons. 
                          It corresponds to '(1-baryon_fraction) * enclosed mass in
                          DMO (dark matter only) simulations'.
      mass_bar      : enclosed baryonic mass for which to calculate the DM profile.
      f_bar         : optional cosmic baryonic fraction.
   Returns:
      Array of 'contracted' enclosed masses.
   """
    eta_bar = mass_bar / mass_DM * (1.-f_bar) / f_bar  # the last two terms account for transforming the DM mass into the corresponding baryonic mass in DMO simulations
    increase_factor = 0.45 + 0.38 * (eta_bar + 1.16)**0.53
    return increase_factor


# calculate thee gravitational constant in code units
halo_rgrid = np.logspace( -1, np.log10(R200), 31 )
_G_Newton = np.mean( [Cautun_halo_unContracted.rforce(r,0.)*-r**2/Cautun_halo_unContracted.mass(r) for r in halo_rgrid] )

# calculate the enclosed DM and baryonic masses needed for halo contraction
_Mass_encl_DM = Cautun_halo_unContracted.mass( halo_rgrid )
_Mass_encl_bar= Baryon_enclosed_mass(halo_rgrid,_G_Newton)

_contraction_factor = contract_factor_enclosed_mass( _Mass_encl_DM, _Mass_encl_bar, f_bar=fb )
from scipy import interpolate
halo_contraction_factor = interpolate.interp1d( halo_rgrid, _contraction_factor, fill_value="extrapolate", bounds_error=False )


# find the normalization factor for the "radial force" input to the "interpSphericalPotential" class 
halo_rgrid = np.logspace( -1, np.log10(R200), 301 )
halo_rforce = lambda R : Cautun_halo_unContracted.rforce(R,0)
_temp = interpSphericalPotential(rforce=halo_rforce, rgrid=halo_rgrid, vo=vo, ro=ro)
_norm_factor = np.mean( [Cautun_halo_unContracted.rforce(R,0)/_temp.rforce(R,0.) for R in halo_rgrid[::10]] )

# calculate the contracted halo profile
halo_rforce = lambda R : Cautun_halo_unContracted.rforce(R,0) * halo_contraction_factor(R) * _norm_factor
Cautun_halo = interpSphericalPotential(rforce=halo_rforce, rgrid=halo_rgrid, vo=vo, ro=ro)

Cautun20 = Cautun_halo + Cautun_disk + Cautun_bulge + Cautun_cgm


# Go back to old floating-point warnings settings
np.seterr(**old_error_settings)

    
    
