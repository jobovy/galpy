# Cautun (2020) potential 
# Thanks to Thomas Callingham (Durham University, UK) which implemented the potential within galpy

import numpy as np
from galpy.potential import NFWPotential
from galpy.potential import DiskSCFPotential
from galpy.potential import SCFPotential
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
critz0 = ((127.5/(1e9))/rhoo)
R200   = 219/ro #R200 for cgm


def gas_dens(R,z):
    return mwpot_helpers.expsech2_dens_with_hole(R,z,Rd_HI,Rm_HI,zd_HI,Sigma0_HI) + mwpot_helpers.expsech2_dens_with_hole(R,z,Rd_H2,Rm_H2,zd_H2,Sigma0_H2)

def stellar_dens(R,z):
    return mwpot_helpers.expexp_dens(R,z,Rd_thin,zd_thin,Sigma0_thin) + mwpot_helpers.expexp_dens(R,z,Rd_thick,zd_thick,Sigma0_thick)

def bulge_dens(R,z):
    return mwpot_helpers.core_pow_dens_with_cut(R,z,1.8,r0_bulge,rcut_bulge,
                                                rho0_bulge,0.5)

def cgm_dens(R,z):
    r = np.sqrt(R**2+(z**2))
    dens_cgm = 200 * critz0 * A * fb * (r/R200)**Beta 
    if r>R200:
        dens_cgm*=np.exp(1-r/R200)
    return dens_cgm



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

Cautun_cgm= SCFPotential(\
    Acos=scf_compute_coeffs_spherical(cgm_dens,20,a=20)[0], a=20, ro=ro, vo=vo )

Cautun_disk= DiskSCFPotential( dens=lambda R,z: gas_dens(R,z) + stellar_dens(R,z), Sigma=sigmadict, \
                              hz=hzdict, a=2.5, N=30, L=30, ro=ro, vo=vo )


Cautun_unContracted = NFWPotential( conc=conc, mvir=m200/1.e12, vo=vo, ro=ro, H=67.77, Om=0.307, overdens=200.0 * (1.-fb), wrtcrit=True )



# functions for calculating the contraction of the DM halo given the baryonic mass distribution
def DM_Dens_uncontracted(R,z): 
    return Cautun_unContracted.dens( R, z, use_physical=False )

def Baryon_Dens(R,z): #Total baryon profile in array friendly format
    TotalDens = gas_dens(R,z) + stellar_dens(R,z) + bulge_dens(R,z) + cgm_dens(R,z)
    return TotalDens

def enclosedMass( Func, rbins, N=2000 ):  #Gives the enclosed mass in r for (R,z) func
    r1 = np.logspace( np.log10(rbins[0]*1.e-3), np.log10(rbins[-1]*1.1), N+1 )  # the bins used for the enclosed mass calculation
    r  = np.sqrt( r1[1:] * r1[:-1] )
    dr = r1[1:] - r1[:-1]
    
    I = lambda x, xr: Func( R=xr * np.cos(x), z=xr * np.sin(x) ) * 4*np.pi * xr**2 * np.cos(x) 
    
    shellMass = np.zeros( r.shape[0] )
    for i in range( r.shape[0] ):
        shellMass[i] = integrate.quad( I, 0., np.pi/2, args=( r[i], ) )[0] * dr[i]
    return np.interp( rbins, r1[1:], shellMass.cumsum() )

def enclosedMass_spherical( Func, rbins ):  #Gives the enclosed mass in r for spherically symmetric (R,z) func
    I = lambda x: Func(x,0) * 4.* np.pi * x*x
    r_range = np.column_stack( ( np.hstack( (1.e-3*rbins[0],rbins[:-1]) ), rbins ) )
    out = np.zeros( len(rbins) )
    for i in range( len(rbins) ):
        out[i] = integrate.quad( I, r_range[i,0], r_range[i,1] )[0]
    return out.cumsum()

def sphericalAverage( Func, rbins ):  #Gives the spherical average in r for (R,z) func
    I = lambda x, xr: Func( R=xr * np.cos(x), z=xr * np.sin(x) ) * np.cos(x) 
    meanValue = np.zeros( rbins.shape[0] )
    for i in range( rbins.shape[0] ):
        meanValue[i] = integrate.quad( I, 0., np.pi/2, args=( rbins[i], ) )[0]
    return meanValue


def contract_density( density_DM, density_bar, mass_DM, mass_bar, f_bar=0.157 ):
    """ Returns the contracted DM density profile given the 'uncontracted' density and that of the baryonic distribution.
    It uses the differential (d/dr) form of Eq. (11) from Cautun et al (2020).
   
   Args:
      density_DM    : array of DM densities. 
                          It corresponds to '(1-baryon_fraction) * density in
                          DMO (dark matter only) simulations'.
      density_bar   : array of baryonic densities.
      mass_DM       : enclosed mass in the DM component in the absence of baryons. 
                          It corresponds to '(1-baryon_fraction) * enclosed mass in
                          DMO (dark matter only) simulations'.
      mass_bar      : enclosed baryonic mass for which to calculate the DM profile.
      f_bar         : optional cosmic baryonic fraction.
   Returns:
      Array of 'contracted' DM densities.
   """
        
    eta_bar = mass_bar / mass_DM * (1.-f_bar) / f_bar  # the last two terms account for transforming the DM mass into the corresponding baryonic mass in DMO simulations
    first_factor = 0.45 + 0.38 * (eta_bar + 1.16)**0.53
    temp         = density_bar - eta_bar * density_DM * f_bar / (1.-f_bar)
    const_term   = 0.38 * 0.53 * (eta_bar + 1.16)**(0.53-1.) * (1.-f_bar) / f_bar * temp
    
    return density_DM * first_factor + const_term


def potential_contract_DM_halo( rho_DMO_func, rho_Baryon_func, f_bar=0.157 ):
    """ Returns the contracted DM density in galpy units for the given baryonic profile.
   
   Args:
      rho_DM_func        : function that gives the 'uncontracted' DM density at coordinates (R,z).
      rho_Baryon_func    : function that gives the baryonic density at coordinates (R,z).
      f_bar              : optional cosmic baryonic fraction.
   Returns:
      Contracted_rho_dm  : contracted DM density at a set of radial distances.
      rvals              : array of radial distances for which the density was calculated.
      MCum_bar           : the enclosed baryonic mass (at a different set of distances from the density).
      MCum_DM            : the enclosed input DM mass. 
      MCum_DM_contracted : the enclosed contracted DM mass.
      rspace             : the radial values for which the enclosed mass was calculated.
   """
    # Create logarithmic r grid
    rspace = np.logspace( -2, 2, 201 )
    rvals  = np.sqrt( rspace[1:] * rspace[:-1] )
    
    # calculate masses and densities on r grid
    MCum_DM  = enclosedMass_spherical( rho_DMO_func, rspace )      # the enclosed mass in DM
    MCum_bar = enclosedMass( rho_Baryon_func, rspace )             # the enclosed mass in baryons
    rho_DM   = rho_DMO_func( rspace, 0 )                           # DM density at each bin position
    rho_bar  = sphericalAverage( rho_Baryon_func, rspace )         # baryonic density at each bin position
    
    # contract the DM density profile
    rho_dm_contracted = contract_density( rho_DM, rho_bar, MCum_DM, MCum_bar, f_bar=f_bar )
    
    return rho_dm_contracted, rspace


rho_DM_contracted, rspace = \
        potential_contract_DM_halo( DM_Dens_uncontracted, Baryon_Dens, f_bar=fb )

from scipy import interpolate
interpolated_rho_DM_contracted = interpolate.interp1d( rspace, rho_DM_contracted, fill_value="extrapolate" )

def Contracted_DM_dens(R,z):
    r = np.sqrt( (R**2.) + (z**2.) )
    if r>rspace[-1]: 
        return rho_DM_contracted[-1] * np.exp( 1 - ((r/rspace[-1])**2) )
    return interpolated_rho_DM_contracted( r )

Cautun_halo= SCFPotential(\
    Acos=scf_compute_coeffs_spherical( Contracted_DM_dens,60,a=50 )[0], a=50, ro=ro, vo=vo )


Cautun20 = Cautun_halo + Cautun_disk + Cautun_bulge + Cautun_cgm


# Go back to old floating-point warnings settings
np.seterr(**old_error_settings)

    
    