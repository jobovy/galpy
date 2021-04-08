# Tests of spherical distribution functions
import platform
WIN32= platform.system() == 'Windows'
import pytest
import numpy
from scipy import special
from galpy import potential
from galpy.df import isotropicHernquistdf, constantbetaHernquistdf, kingdf, \
    isotropicPlummerdf, osipkovmerrittHernquistdf, isotropicNFWdf, \
    eddingtondf, osipkovmerrittdf, osipkovmerrittNFWdf, constantbetadf
from galpy.df import jeans

############################# ISOTROPIC HERNQUIST DF ##########################
# Note that we use the Hernquist case to check a bunch of code in the
# sphericaldf realm that doesn't need to be check for each new spherical DF
def test_isotropic_hernquist_dens_spherically_symmetric():
    pot= potential.HernquistPotential(amp=2.3,a=1.3)
    dfh= isotropicHernquistdf(pot=pot)
    numpy.random.seed(10)
    samp= dfh.sample(n=100000)
    # Check spherical symmetry for different harmonics l,m
    tol= 1e-2
    check_spherical_symmetry(samp,0,0,tol)
    check_spherical_symmetry(samp,1,0,tol)
    check_spherical_symmetry(samp,1,-1,tol)
    check_spherical_symmetry(samp,1,1,tol)
    check_spherical_symmetry(samp,2,0,tol)
    check_spherical_symmetry(samp,2,-1,tol)
    check_spherical_symmetry(samp,2,-2,tol)
    check_spherical_symmetry(samp,2,1,tol)
    check_spherical_symmetry(samp,2,2,tol)
    # and some higher order ones
    check_spherical_symmetry(samp,3,1,tol)
    check_spherical_symmetry(samp,9,-6,tol)
    return None
    
def test_isotropic_hernquist_dens_massprofile():
    pot= potential.HernquistPotential(amp=2.3,a=1.3)
    dfh= isotropicHernquistdf(pot=pot)
    numpy.random.seed(10)
    samp= dfh.sample(n=100000)
    tol= 5*1e-3
    check_spherical_massprofile(samp,
                                lambda r: pot.mass(r)\
                                   /pot.mass(numpy.amax(samp.r()),
                                             ),
                                tol,skip=1000)
    return None

def test_isotropic_hernquist_singler_is_atsingler():
    pot= potential.HernquistPotential(amp=2.3,a=1.3)
    dfh= isotropicHernquistdf(pot=pot)
    numpy.random.seed(10)
    samp= dfh.sample(R=1.3,z=0.,n=100000)
    assert numpy.all(numpy.fabs(samp.r()-1.3) < 1e-8), 'Sampling a spherical distribution function at a single r does not produce orbits at a single r'
    return None

def test_isotropic_hernquist_singler_is_atrandomphi():
    pot= potential.HernquistPotential(amp=2.3,a=1.3)
    dfh= isotropicHernquistdf(pot=pot)
    numpy.random.seed(10)
    samp= dfh.sample(R=1.3,z=0.,n=100000)
    tol= 1e-2
    check_azimuthal_symmetry(samp,0,tol)
    check_azimuthal_symmetry(samp,1,tol)
    check_azimuthal_symmetry(samp,2,tol)
    check_azimuthal_symmetry(samp,3,tol)
    check_azimuthal_symmetry(samp,4,tol)
    check_azimuthal_symmetry(samp,5,tol)
    check_azimuthal_symmetry(samp,6,tol)
    return None

def test_isotropic_hernquist_singlerphi_is_atsinglephi():
    pot= potential.HernquistPotential(amp=2.3,a=1.3)
    dfh= isotropicHernquistdf(pot=pot)
    numpy.random.seed(10)
    samp= dfh.sample(R=1.3,z=0.,phi=numpy.pi-0.3,n=100000)
    assert numpy.all(numpy.fabs(samp.phi()-numpy.pi+0.3) < 1e-8), 'Sampling a spherical distribution function at a single r and phi oes not produce orbits at a single phi'
    return None

def test_isotropic_hernquist_givenr_are_atgivenr():
    pot= potential.HernquistPotential(amp=2.3,a=1.3)
    dfh= isotropicHernquistdf(pot=pot)
    numpy.random.seed(10)
    r= numpy.linspace(0.1,10.,1001)
    theta= numpy.random.uniform(size=len(r))*numpy.pi
    # n should be ignored in the following
    samp= dfh.sample(R=r*numpy.sin(theta),z=r*numpy.cos(theta),n=100000)
    assert len(samp) == len(r), 'Length of sample with given r array is not equal to length of r'
    assert numpy.all(numpy.fabs(samp.r()-r) < 1e-8), 'Sampling a spherical distribution function at given r does not produce orbits at these given r'
    assert numpy.all(numpy.fabs(samp.R()-r*numpy.sin(theta)) < 1e-8), 'Sampling a spherical distribution function at given R does not produce orbits at these given R'
    assert numpy.all(numpy.fabs(samp.z()-r*numpy.cos(theta)) < 1e-8), 'Sampling a spherical distribution function at given z does not produce orbits at these given z'
    return None

def test_isotropic_hernquist_dens_massprofile_forcemassinterpolation():
    pot= potential.HernquistPotential(amp=2.3,a=1.3)
    # Remove the inverse cumulative mass function to force its interpolation
    class isotropicHernquistdfNoICMF(isotropicHernquistdf):
        _icmf= property()
    dfh= isotropicHernquistdfNoICMF(pot=pot)
    numpy.random.seed(10)
    samp= dfh.sample(n=100000)
    tol= 5*1e-3
    check_spherical_massprofile(samp,
                                lambda r: pot.mass(r)\
                                   /pot.mass(numpy.amax(samp.r()),
                                             ),
                                tol,skip=1000)
    return None

def test_isotropic_hernquist_sigmar():
    pot= potential.HernquistPotential(amp=2.3,a=1.3)
    dfh= isotropicHernquistdf(pot=pot)
    numpy.random.seed(10)
    samp= dfh.sample(n=300000)
    tol= 0.05
    check_sigmar_against_jeans(samp,pot,tol,beta=0.,
                               rmin=pot._scale/10.,rmax=pot._scale*10.,bins=31)
    return None

def test_isotropic_hernquist_singler_sigmar():
    pot= potential.HernquistPotential(amp=2.3,a=1.3)
    dfh= isotropicHernquistdf(pot=pot)
    numpy.random.seed(10)
    for r in [0.3,1.3,2.3]:
        samp= dfh.sample(R=r,z=0.,n=100000)
        tol= 0.01
        check_sigmar_against_jeans(samp,pot,tol,beta=0.,
                                   rmin=r-0.1,rmax=r+0.1,bins=1)
    return None

def test_isotropic_hernquist_beta():
    pot= potential.HernquistPotential(amp=2.3,a=1.3)
    dfh= isotropicHernquistdf(pot=pot)
    numpy.random.seed(10)
    samp= dfh.sample(n=1000000)
    tol= 6*1e-2
    check_beta(samp,pot,tol,beta=0.,
               rmin=pot._scale/10.,rmax=pot._scale*10.,bins=31)
    return None

def test_isotropic_hernquist_dens_directint():
    pot= potential.HernquistPotential(amp=2.3,a=1.3)
    dfh= isotropicHernquistdf(pot=pot)
    tol= 1e-8
    check_dens_directint(dfh,pot,tol,
                         lambda r: pot.dens(r,0),
                         rmin=pot._scale/10.,
                         rmax=pot._scale*10.,bins=31)
    return None

def test_isotropic_hernquist_meanvr_directint():
    pot= potential.HernquistPotential(amp=2.3,a=1.3)
    dfh= isotropicHernquistdf(pot=pot)
    tol= 1e-8
    check_meanvr_directint(dfh,pot,tol,beta=0.,rmin=pot._scale/10.,
                           rmax=pot._scale*10.,bins=31)
    return None

def test_isotropic_hernquist_sigmar_directint():
    pot= potential.HernquistPotential(amp=2.3,a=1.3)
    dfh= isotropicHernquistdf(pot=pot)
    tol= 1e-5
    check_sigmar_against_jeans_directint(dfh,pot,tol,beta=0.,
                                         rmin=pot._scale/10.,
                                         rmax=pot._scale*10.,
                                         bins=31)
    return None

def test_isotropic_hernquist_sigmar_directint_forcevmoment():
    pot= potential.HernquistPotential(amp=2.3,a=1.3)
    dfh= isotropicHernquistdf(pot=pot)
    tol= 1e-5
    check_sigmar_against_jeans_directint_forcevmoment(dfh,pot,tol,beta=0.,
                                                      rmin=pot._scale/10.,
                                                      rmax=pot._scale*10.,
                                                      bins=31)
    return None

def test_isotropic_hernquist_beta_directint():
    pot= potential.HernquistPotential(amp=2.3,a=1.3)
    dfh= isotropicHernquistdf(pot=pot)
    tol= 1e-8
    check_beta_directint(dfh,tol,beta=0.,
                         rmin=pot._scale/10.,
                         rmax=pot._scale*10.,
                         bins=31)
    return None

def test_isotropic_hernquist_energyoutofbounds():
    pot= potential.HernquistPotential(amp=2.3,a=1.3)
    dfh= isotropicHernquistdf(pot=pot)
    assert numpy.all(numpy.fabs(dfh((numpy.arange(0.1,10.,0.1),))) < 1e-8), 'Evaluating the isotropic Hernquist DF at E > 0 does not give zero'
    assert numpy.all(numpy.fabs(dfh((pot(0,0)-1e-4,))) < 1e-8), 'Evaluating the isotropic Hernquist DF at E < -GM/a does not give zero'
    return None

# Check that samples of R,vR,.. are the same as orbit samples
def test_isotropic_hernquist_phasespacesamples_vs_orbitsamples():
    pot= potential.HernquistPotential(amp=2.3,a=1.3)
    dfh= isotropicHernquistdf(pot=pot)
    numpy.random.seed(10)    
    samp_orbits= dfh.sample(n=1000)
    # Reset seed such that we should get the same
    numpy.random.seed(10)
    samp_RvR= dfh.sample(n=1000,return_orbit=False)
    assert numpy.all(numpy.fabs(samp_orbits.R()-samp_RvR[0]) < 1e-8), 'Sampling R,vR,... from spherical DF does not give the same as sampling equivalent orbits'
    assert numpy.all(numpy.fabs(samp_orbits.vR()-samp_RvR[1]) < 1e-8), 'Sampling R,vR,... from spherical DF does not give the same as sampling equivalent orbits'
    assert numpy.all(numpy.fabs(samp_orbits.vT()-samp_RvR[2]) < 1e-8), 'Sampling R,vR,... from spherical DF does not give the same as sampling equivalent orbits'
    assert numpy.all(numpy.fabs(samp_orbits.z()-samp_RvR[3]) < 1e-8), 'Sampling R,vR,... from spherical DF does not give the same as sampling equivalent orbits'
    assert numpy.all(numpy.fabs(samp_orbits.vz()-samp_RvR[4]) < 1e-8), 'Sampling R,vR,... from spherical DF does not give the same as sampling equivalent orbits'
    assert numpy.all(numpy.fabs(samp_orbits.phi()-samp_RvR[5]) < 1e-8), 'Sampling R,vR,... from spherical DF does not give the same as sampling equivalent orbits'
    return None

def test_isotropic_hernquist_diffcalls():
    from galpy.orbit import Orbit
    pot= potential.HernquistPotential(amp=2.3,a=1.3)
    dfh= isotropicHernquistdf(pot=pot)
    # R,vR... vs. E
    R,vR,vT,z,vz,phi= 1.1,0.3,0.2,0.9,-0.2,2.4
    # Calculate E directly
    assert numpy.fabs(dfh(R,vR,vT,z,vz,phi)-dfh((pot(R,z)+0.5*(vR**2.+vT**2.+vz**2.),))) < 1e-8, 'Calling the isotropic Hernquist DF with R,vR,... or E[R,vR,...] does not give the same answer'
    # Also L
    assert numpy.fabs(dfh(R,vR,vT,z,vz,phi)-dfh((pot(R,z)+0.5*(vR**2.+vT**2.+vz**2.),numpy.sqrt(numpy.sum(Orbit([R,vR,vT,z,vz,phi]).L()**2.))))) < 1e-8, 'Calling the isotropic Hernquist DF with R,vR,... or E[R,vR,...] does not give the same answer'
    # Also as orbit
    assert numpy.fabs(dfh(R,vR,vT,z,vz,phi)-dfh(Orbit([R,vR,vT,z,vz,phi]))) < 1e-8, 'Calling the isotropic Hernquist DF with R,vR,... or E[R,vR,...] does not give the same answer'   
    return None

############################# ANISOTROPIC HERNQUIST DF ########################
def test_anisotropic_hernquist_dens_spherically_symmetric():
    pot= potential.HernquistPotential(amp=2.3,a=1.3)
    betas= [-0.7,-0.5,-0.4,0.,0.3,0.5]
    for beta in betas:
        dfh= constantbetaHernquistdf(pot=pot,beta=beta)
        numpy.random.seed(10)
        samp= dfh.sample(n=100000)
        # Check spherical symmetry for different harmonics l,m
        tol= 1e-2
        check_spherical_symmetry(samp,0,0,tol)
        check_spherical_symmetry(samp,1,0,tol)
        check_spherical_symmetry(samp,1,-1,tol)
        check_spherical_symmetry(samp,1,1,tol)
        check_spherical_symmetry(samp,2,0,tol)
        check_spherical_symmetry(samp,2,-1,tol)
        check_spherical_symmetry(samp,2,-2,tol)
        check_spherical_symmetry(samp,2,1,tol)
        check_spherical_symmetry(samp,2,2,tol)
        # and some higher order ones
        check_spherical_symmetry(samp,3,1,tol)
        check_spherical_symmetry(samp,9,-6,tol)
    return None
    
def test_anisotropic_hernquist_dens_massprofile():
    pot= potential.HernquistPotential(amp=2.3,a=1.3)
    betas= [-0.7,-0.5,-0.4,0.,0.3,0.5]
    for beta in betas:
        dfh= constantbetaHernquistdf(pot=pot,beta=beta)
        numpy.random.seed(10)
        samp= dfh.sample(n=100000)
        tol= 5*1e-3
        check_spherical_massprofile(samp,lambda r: pot.mass(r)\
                                    /pot.mass(numpy.amax(samp.r())),
                                    tol,skip=1000)
    return None

def test_anisotropic_hernquist_sigmar():
    pot= potential.HernquistPotential(amp=2.3,a=1.3)
    betas= [-0.7,-0.5,-0.4,0.,0.3,0.5]
    for beta in betas:
        dfh= constantbetaHernquistdf(pot=pot,beta=beta)
        numpy.random.seed(10)
        samp= dfh.sample(n=300000)
        tol= 0.05
        check_sigmar_against_jeans(samp,pot,tol,beta=beta,
                                   rmin=pot._scale/10.,rmax=pot._scale*10.,
                                   bins=31)
    return None

def test_anisotropic_hernquist_beta():
    pot= potential.HernquistPotential(amp=2.3,a=1.3)
    betas= [-0.7,-0.5,-0.4,0.,0.3,0.5]
    for beta in betas:
        dfh= constantbetaHernquistdf(pot=pot,beta=beta)
        numpy.random.seed(10)
        samp= dfh.sample(n=1000000)
        tol= 8*1e-2 * (beta > -0.7) + 0.12 * (beta == -0.7)
        check_beta(samp,pot,tol,beta=beta,
                   rmin=pot._scale/10.,rmax=pot._scale*10.,bins=31)
    return None

def test_anisotropic_hernquist_dens_directint():
    pot= potential.HernquistPotential(amp=2.3,a=1.3)
    betas= [-0.7,-0.5,-0.4,0.,0.3,0.5]
    for beta in betas:
        dfh= constantbetaHernquistdf(pot=pot,beta=beta)
        tol= 1e-7
        check_dens_directint(dfh,pot,tol,
                             lambda r: pot.dens(r,0),
                             rmin=pot._scale/10.,
                             rmax=pot._scale*10.,bins=31)
    return None

def test_anisotropic_hernquist_meanvr_directint():
    pot= potential.HernquistPotential(amp=2.3,a=1.3)
    betas= [-0.7,-0.5,-0.4,0.,0.3,0.5]
    for beta in betas:
        dfh= constantbetaHernquistdf(pot=pot,beta=beta)
        tol= 1e-8
        check_meanvr_directint(dfh,pot,tol,beta=beta,rmin=pot._scale/10.,
                               rmax=pot._scale*10.,bins=31)
    return None

def test_anisotropic_hernquist_sigmar_directint():
    pot= potential.HernquistPotential(amp=2.3,a=1.3)
    betas= [-0.7,-0.5,-0.4,0.,0.3,0.5]
    for beta in betas:
        dfh= constantbetaHernquistdf(pot=pot,beta=beta)
        tol= 1e-5
        check_sigmar_against_jeans_directint(dfh,pot,tol,beta=beta,
                                             rmin=pot._scale/10.,
                                             rmax=pot._scale*10.,
                                             bins=31)
    return None

def test_anisotropic_hernquist_beta_directint():
    pot= potential.HernquistPotential(amp=2.3,a=1.3)
    betas= [-0.7,-0.5,-0.4,0.,0.3,0.5]
    for beta in betas:
        dfh= constantbetaHernquistdf(pot=pot,beta=beta)
        tol= 1e-8
        check_beta_directint(dfh,tol,beta=beta,
                             rmin=pot._scale/10.,
                             rmax=pot._scale*10.,
                             bins=31)
    return None

def test_anisotropic_hernquist_energyoutofbounds():
    pot= potential.HernquistPotential(amp=2.3,a=1.3)
    betas= [-0.7,-0.5,-0.4,0.,0.3,0.5]
    for beta in betas:
        dfh= constantbetaHernquistdf(pot=pot,beta=beta)
        assert numpy.all(numpy.fabs(dfh((numpy.arange(0.1,10.,0.1),1.1))) < 1e-8), 'Evaluating the anisotropic Hernquist DF at E > 0 does not give zero'
        assert numpy.all(numpy.fabs(dfh((pot(0,0)-1e-4,1.1))) < 1e-8), 'Evaluating the anisotropic Hernquist DF at E < -GM/a does not give zero'
    return None

def test_anisotropic_hernquist_diffcalls():
    from galpy.orbit import Orbit
    pot= potential.HernquistPotential(amp=2.3,a=1.3)
    betas= [-0.7,-0.5,-0.4,0.,0.3,0.5]
    for beta in betas:
        dfh= constantbetaHernquistdf(pot=pot,beta=beta)
        # R,vR... vs. E
        R,vR,vT,z,vz,phi= 1.1,0.3,0.2,0.9,-0.2,2.4
        # Calculate E directly and L from Orbit
        assert numpy.fabs(dfh(R,vR,vT,z,vz,phi)-dfh((pot(R,z)+0.5*(vR**2.+vT**2.+vz**2.),numpy.sqrt(numpy.sum(Orbit([R,vR,vT,z,vz,phi]).L()**2.))))) < 1e-8, 'Calling the anisotropic Hernquist DF with R,vR,... or E[R,vR,...] does not give the same answer'
        # Also as orbit
        assert numpy.fabs(dfh(R,vR,vT,z,vz,phi)-dfh(Orbit([R,vR,vT,z,vz,phi]))) < 1e-8, 'Calling the anisotropic Hernquist DF with R,vR,... or E[R,vR,...] does not give the same answer'   
    return None

########################### OSIPKOV-MERRITT HERNQUIST DF ######################
def test_osipkovmerritt_hernquist_dens_spherically_symmetric():
    pot= potential.HernquistPotential(amp=2.3,a=1.3)
    ras= [0.3,2.3,5.7]
    for ra in ras:
        dfh= osipkovmerrittHernquistdf(pot=pot,ra=ra)
        numpy.random.seed(10)
        samp= dfh.sample(n=100000)
        # Check spherical symmetry for different harmonics l,m
        tol= 1e-2
        check_spherical_symmetry(samp,0,0,tol)
        check_spherical_symmetry(samp,1,0,tol)
        check_spherical_symmetry(samp,1,-1,tol)
        check_spherical_symmetry(samp,1,1,tol)
        check_spherical_symmetry(samp,2,0,tol)
        check_spherical_symmetry(samp,2,-1,tol)
        check_spherical_symmetry(samp,2,-2,tol)
        check_spherical_symmetry(samp,2,1,tol)
        check_spherical_symmetry(samp,2,2,tol)
        # and some higher order ones
        check_spherical_symmetry(samp,3,1,tol)
        check_spherical_symmetry(samp,9,-6,tol)
    return None
    
def test_osipkovmerritt_hernquist_dens_massprofile():
    pot= potential.HernquistPotential(amp=2.3,a=1.3)
    ras= [0.3,2.3,5.7]
    for ra in ras:
        dfh= osipkovmerrittHernquistdf(pot=pot,ra=ra)
        numpy.random.seed(10)
        samp= dfh.sample(n=100000)
        tol= 5*1e-3
        check_spherical_massprofile(samp,lambda r: pot.mass(r)\
                                    /pot.mass(numpy.amax(samp.r())),
                                    tol,skip=1000)
    return None

def test_osipkovmerritt_hernquist_sigmar():
    pot= potential.HernquistPotential(amp=2.3,a=1.3)
    ras= [0.3,2.3,5.7]
    for ra in ras:
        dfh= osipkovmerrittHernquistdf(pot=pot,ra=ra)
        numpy.random.seed(10)
        samp= dfh.sample(n=100000)
        tol= 0.05
        check_sigmar_against_jeans(samp,pot,tol,
                                   beta=lambda r: 1./(1.+ra**2./r**2.),
                                   rmin=pot._scale/10.,rmax=pot._scale*10.,
                                   bins=31)
    return None

def test_osipkovmerritt_hernquist_beta():
    pot= potential.HernquistPotential(amp=2.3,a=1.3)
    ras= [0.3,2.3,5.7]
    for ra in ras:
        dfh= osipkovmerrittHernquistdf(pot=pot,ra=ra)
        numpy.random.seed(10)
        samp= dfh.sample(n=1000000)
        tol= 0.06
        check_beta(samp,pot,tol,beta=lambda r: 1./(1.+ra**2./r**2.),
                   rmin=pot._scale/10.,rmax=pot._scale*10.,bins=31)
    return None

def test_osipkovmerritt_hernquist_dens_directint():
    pot= potential.HernquistPotential(amp=2.3,a=1.3)
    ras= [0.3,2.3,5.7]
    for ra in ras:
        dfh= osipkovmerrittHernquistdf(pot=pot,ra=ra)
        tol= 1e-5
        check_dens_directint(dfh,pot,tol,
                             lambda r: pot.dens(r,0),
                             rmin=pot._scale/10.,
                             rmax=pot._scale*10.,bins=6)
    return None

def test_osipkovmerritt_hernquist_meanvr_directint():
    pot= potential.HernquistPotential(amp=2.3,a=1.3)
    ras= [0.3,2.3,5.7]
    for ra in ras:
        dfh= osipkovmerrittHernquistdf(pot=pot,ra=ra)
        tol= 1e-8
        check_meanvr_directint(dfh,pot,tol,rmin=pot._scale/10.,
                               rmax=pot._scale*10.,bins=6)
    return None

def test_osipkovmerritt_hernquist_sigmar_directint():
    pot= potential.HernquistPotential(amp=2.3,a=1.3)
    ras= [0.3,2.3,5.7]
    for ra in ras:
        dfh= osipkovmerrittHernquistdf(pot=pot,ra=ra)
        tol= 1e-4
        check_sigmar_against_jeans_directint(dfh,pot,tol,
                                             beta=lambda r: 1./(1.+ra**2./r**2.),
                                             rmin=pot._scale/10.,
                                             rmax=pot._scale*10.,
                                             bins=6)
    return None

def test_osipkovmerritt_hernquist_beta_directint():
    pot= potential.HernquistPotential(amp=2.3,a=1.3)
    ras= [0.3,2.3,5.7]
    for ra in ras:
        dfh= osipkovmerrittHernquistdf(pot=pot,ra=ra)
        tol= 1e-8
        check_beta_directint(dfh,tol,beta=lambda r: 1./(1.+ra**2./r**2.),
                             rmin=pot._scale/10.,
                             rmax=pot._scale*10.,
                             bins=6)
    return None

def test_osipkovmerritt_hernquist_Qoutofbounds():
    pot= potential.HernquistPotential(amp=2.3,a=1.3)
    ras= [0.3,2.3,5.7]
    for ra in ras:
        dfh= osipkovmerrittHernquistdf(pot=pot,ra=ra)
        assert numpy.all(numpy.fabs(dfh((numpy.arange(0.1,10.,0.1),1.1))) < 1e-8), 'Evaluating the Osipkov-Merritt Hernquist DF at E > 0 does not give zero'
        # The next one is not actually a physical orbit...
        assert numpy.all(numpy.fabs(dfh((pot(0,0)-1e-1,0.1))) < 1e-8), 'Evaluating the Osipkov-Merritt Hernquist DF at E < -GM/a does not give zero'
        assert numpy.all(numpy.fabs(dfh((-1e-4,1.1))) < 1e-8), 'Evaluating the Osipkov-Merritt Hernquist DF at Q < 0 does not give zero'
    return None

def test_osipkovmerritt_hernquist_diffcalls():
    from galpy.orbit import Orbit
    pot= potential.HernquistPotential(amp=2.3,a=1.3)
    ras= [0.3,2.3,5.7]
    for ra in ras:
        dfh= osipkovmerrittHernquistdf(pot=pot,ra=ra)
        # R,vR... vs. E
        R,vR,vT,z,vz,phi= 1.1,0.3,0.2,0.9,-0.2,2.4
        # Calculate E directly and L from Orbit
        assert numpy.fabs(dfh(R,vR,vT,z,vz,phi)-dfh((pot(R,z)+0.5*(vR**2.+vT**2.+vz**2.),numpy.sqrt(numpy.sum(Orbit([R,vR,vT,z,vz,phi]).L()**2.))))) < 1e-8, 'Calling the Osipkov-Merritt anisotropic Hernquist DF with R,vR,... or E[R,vR,...] does not give the same answer'
        # Also as orbit
        assert numpy.fabs(dfh(R,vR,vT,z,vz,phi)-dfh(Orbit([R,vR,vT,z,vz,phi]))) < 1e-8, 'Calling the Osipkov-Merritt isotropic Hernquist DF with R,vR,... or E[R,vR,...] does not give the same answer'   
    return None

############################## OSIPKOV-MERRITT NFW DF #########################
def test_osipkovmerritt_nfw_dens_spherically_symmetric():
    pot= potential.NFWPotential(amp=2.3,a=1.3)
    ras= [2.3,5.7]
    for ra in ras:
        dfh= osipkovmerrittNFWdf(pot=pot,ra=ra)
        numpy.random.seed(10)
        samp= dfh.sample(n=100000)
        # Check spherical symmetry for different harmonics l,m
        tol= 1e-2
        check_spherical_symmetry(samp,0,0,tol)
        check_spherical_symmetry(samp,1,0,tol)
        check_spherical_symmetry(samp,1,-1,tol)
        check_spherical_symmetry(samp,1,1,tol)
        check_spherical_symmetry(samp,2,0,tol)
        check_spherical_symmetry(samp,2,-1,tol)
        check_spherical_symmetry(samp,2,-2,tol)
        check_spherical_symmetry(samp,2,1,tol)
        check_spherical_symmetry(samp,2,2,tol)
        # and some higher order ones
        check_spherical_symmetry(samp,3,1,tol)
        check_spherical_symmetry(samp,9,-6,tol)
    return None
    
def test_osipkovmerritt_nfw_dens_massprofile():
    pot= potential.NFWPotential(amp=2.3,a=1.3)
    ras= [2.3,5.7]
    for ra in ras:
        dfh= osipkovmerrittNFWdf(pot=pot,ra=ra)
        numpy.random.seed(10)
        samp= dfh.sample(n=100000)
        tol= 5*1e-3
        check_spherical_massprofile(samp,lambda r: pot.mass(r)\
                                    /pot.mass(numpy.amax(samp.r())),
                                    tol,skip=1000)
    return None

def test_osipkovmerritt_nfw_sigmar():
    pot= potential.NFWPotential(amp=2.3,a=1.3)
    ras= [2.3,5.7]
    for ra in ras:
        dfh= osipkovmerrittNFWdf(pot=pot,ra=ra)
        numpy.random.seed(10)
        samp= dfh.sample(n=1000000)
        tol= 0.17
        check_sigmar_against_jeans(samp,pot,tol,
                                   beta=lambda r: 1./(1.+ra**2./r**2.),
                                   rmin=pot._scale/10.,rmax=pot._scale*10.,
                                   bins=31)
    return None

def test_osipkovmerritt_nfw_beta():
    pot= potential.NFWPotential(amp=2.3,a=1.3)
    ras= [2.3,5.7]
    for ra in ras:
        dfh= osipkovmerrittNFWdf(pot=pot,ra=ra)
        numpy.random.seed(10)
        samp= dfh.sample(n=3000000)
        tol= 0.15
        check_beta(samp,pot,tol,beta=lambda r: 1./(1.+ra**2./r**2.),
                   rmin=pot._scale/10.,rmax=pot._scale*10.,bins=31)
    return None

def test_osipkovmerritt_nfw_dens_directint():
    pot= potential.NFWPotential(amp=2.3,a=1.3)
    ras= [2.3,5.7]
    for ra in ras:
        dfh= osipkovmerrittNFWdf(pot=pot,ra=ra,rmax=numpy.inf)
        tol= 0.01 # 1%
        check_dens_directint(dfh,pot,tol,
                             lambda r: pot.dens(r,0),
                             rmin=pot._scale/10.,
                             rmax=pot._scale*10.,bins=6)
    return None

def test_osipkovmerritt_nfw_meanvr_directint():
    pot= potential.NFWPotential(amp=2.3,a=1.3)
    ras= [2.3,5.7]
    for ra in ras:
        dfh= osipkovmerrittNFWdf(pot=pot,ra=ra,rmax=numpy.inf)
        tol= 1e-8
        check_meanvr_directint(dfh,pot,tol,rmin=pot._scale/10.,
                               rmax=pot._scale*10.,bins=6)
    return None

def test_osipkovmerritt_nfw_sigmar_directint():
    pot= potential.NFWPotential(amp=2.3,a=1.3)
    ras= [2.3,5.7]
    for ra in ras:
        dfh= osipkovmerrittNFWdf(pot=pot,ra=ra,rmax=numpy.inf)
        tol= 1e-2 # 1%
        check_sigmar_against_jeans_directint(dfh,pot,tol,
                                             beta=lambda r: 1./(1.+ra**2./r**2.),
                                             rmin=pot._scale/10.,
                                             rmax=pot._scale*10.,
                                             bins=6)
    return None

def test_osipkovmerritt_nfw_beta_directint():
    pot= potential.NFWPotential(amp=2.3,a=1.3)
    ras= [2.3,5.7]
    for ra in ras:
        dfh= osipkovmerrittNFWdf(pot=pot,ra=ra,rmax=numpy.inf)
        tol= 1e-8
        check_beta_directint(dfh,tol,beta=lambda r: 1./(1.+ra**2./r**2.),
                             rmin=pot._scale/10.,
                             rmax=pot._scale*10.,
                             bins=6)
    return None

def test_osipkovmerritt_nfw_Qoutofbounds():
    pot= potential.NFWPotential(amp=2.3,a=1.3)
    ras= [2.3,5.7]
    for ra in ras:
        dfh= osipkovmerrittNFWdf(pot=pot,ra=ra)
        assert numpy.all(numpy.fabs(dfh((numpy.arange(0.1,10.,0.1),1.1))) < 1e-8), 'Evaluating the Osipkov-Merritt NFW DF at E > 0 does not give zero'
        # The next one is not actually a physical orbit...
        assert numpy.all(numpy.fabs(dfh((pot(0,0)-1e-1,0.1))) < 1e-8), 'Evaluating the Osipkov-Merritt NFW DF at E < -GM/a does not give zero'
        assert numpy.all(numpy.fabs(dfh((-1e-4,1.1))) < 1e-8), 'Evaluating the Osipkov-Merritt NFW DF at Q < 0 does not give zero'
    return None

############################# ISOTROPIC PLUMMER DF ############################
def test_isotropic_plummer_dens_spherically_symmetric():
    pot= potential.PlummerPotential(amp=2.3,b=1.3)
    dfp= isotropicPlummerdf(pot=pot)
    numpy.random.seed(10)
    samp= dfp.sample(n=100000)
    # Check spherical symmetry for different harmonics l,m
    tol= 1e-2
    check_spherical_symmetry(samp,0,0,tol)
    check_spherical_symmetry(samp,1,0,tol)
    check_spherical_symmetry(samp,1,-1,tol)
    check_spherical_symmetry(samp,1,1,tol)
    check_spherical_symmetry(samp,2,0,tol)
    check_spherical_symmetry(samp,2,-1,tol)
    check_spherical_symmetry(samp,2,-2,tol)
    check_spherical_symmetry(samp,2,1,tol)
    check_spherical_symmetry(samp,2,2,tol)
    # and some higher order ones
    check_spherical_symmetry(samp,3,1,tol)
    check_spherical_symmetry(samp,9,-6,tol)
    return None
    
def test_isotropic_plummer_dens_massprofile():
    pot= potential.PlummerPotential(amp=2.3,b=1.3)
    dfp= isotropicPlummerdf(pot=pot)
    numpy.random.seed(10)
    samp= dfp.sample(n=100000)
    tol= 5*1e-3
    check_spherical_massprofile(samp,lambda r: pot.mass(r)\
                                /pot.mass(numpy.amax(samp.r())),
                                tol,skip=1000)
    return None

def test_isotropic_plummer_sigmar():
    pot= potential.PlummerPotential(amp=2.3,b=1.3)
    dfp= isotropicPlummerdf(pot=pot)
    numpy.random.seed(10)
    samp= dfp.sample(n=1000000)
    tol= 0.05
    check_sigmar_against_jeans(samp,pot,tol,
                               rmin=pot._scale/10.,rmax=pot._scale*10.,
                               bins=31)
    return None

def test_isotropic_plummer_beta():
    pot= potential.PlummerPotential(amp=2.3,b=1.3)
    dfp= isotropicPlummerdf(pot=pot)
    numpy.random.seed(10)
    samp= dfp.sample(n=1000000)
    tol= 6*1e-2
    check_beta(samp,pot,tol,rmin=pot._scale/10.,rmax=pot._scale*10.,bins=31)
    return None

def test_isotropic_plummer_dens_directint():
    pot= potential.PlummerPotential(amp=2.3,b=1.3)
    dfp= isotropicPlummerdf(pot=pot)
    tol= 1e-7
    check_dens_directint(dfp,pot,tol,
                         lambda r: pot.dens(r,0),
                         rmin=pot._scale/10.,
                         rmax=pot._scale*10.,bins=31)
    return None

def test_isotropic_plummer_meanvr_directint():
    pot= potential.PlummerPotential(amp=2.3,b=1.3)
    dfp= isotropicPlummerdf(pot=pot)
    tol= 1e-8
    check_meanvr_directint(dfp,pot,tol,rmin=pot._scale/10.,
                           rmax=pot._scale*10.,bins=31)
    return None

def test_isotropic_plummer_sigmar_directint():
    pot= potential.PlummerPotential(amp=2.3,b=1.3)
    dfp= isotropicPlummerdf(pot=pot)
    tol= 1e-5
    check_sigmar_against_jeans_directint(dfp,pot,tol,
                                         rmin=pot._scale/10.,
                                         rmax=pot._scale*10.,
                                         bins=31)
    return None

def test_isotropic_plummer_beta_directint():
    pot= potential.PlummerPotential(amp=2.3,b=1.3)
    dfp= isotropicPlummerdf(pot=pot)
    tol= 1e-8
    check_beta_directint(dfp,tol,rmin=pot._scale/10.,rmax=pot._scale*10.,
                         bins=31)
    return None

def test_isotropic_plummer_energyoutofbounds():
    pot= potential.PlummerPotential(amp=2.3,b=1.3)
    dfp= isotropicPlummerdf(pot=pot)
    assert numpy.all(numpy.fabs(dfp((numpy.arange(0.1,10.,0.1),1.1))) < 1e-8), 'Evaluating the isotropic Plummer DF at E > 0 does not give zero'
    assert numpy.all(numpy.fabs(dfp((pot(0,0)-1e-4,1.1))) < 1e-8), 'Evaluating the isotropic Plummer DF at E < Phi(0) does not give zero'
    return None

############################# ISOTROPIC NFW DF ############################
def test_isotropic_nfw_dens_spherically_symmetric():
    pot= potential.NFWPotential(amp=2.3,a=1.3)
    dfp= isotropicNFWdf(pot=pot)
    numpy.random.seed(10)
    samp= dfp.sample(n=100000)
    # Check spherical symmetry for different harmonics l,m
    tol= 1e-2
    check_spherical_symmetry(samp,0,0,tol)
    check_spherical_symmetry(samp,1,0,tol)
    check_spherical_symmetry(samp,1,-1,tol)
    check_spherical_symmetry(samp,1,1,tol)
    check_spherical_symmetry(samp,2,0,tol)
    check_spherical_symmetry(samp,2,-1,tol)
    check_spherical_symmetry(samp,2,-2,tol)
    check_spherical_symmetry(samp,2,1,tol)
    check_spherical_symmetry(samp,2,2,tol)
    # and some higher order ones
    check_spherical_symmetry(samp,3,1,tol)
    check_spherical_symmetry(samp,9,-6,tol)
    return None
    
def test_isotropic_nfw_dens_massprofile():
    pot= potential.NFWPotential(amp=2.3,a=1.3)
    dfp= isotropicNFWdf(pot=pot)
    numpy.random.seed(10)
    samp= dfp.sample(n=100000)
    tol= 5*1e-3
    check_spherical_massprofile(samp,lambda r: pot.mass(r)\
                                /pot.mass(numpy.amax(samp.r())),
                                tol,skip=1000)
    return None

def test_isotropic_nfw_sigmar():
    pot= potential.NFWPotential(amp=2.3,a=1.3)
    dfp= isotropicNFWdf(pot=pot)
    numpy.random.seed(10)
    samp= dfp.sample(n=1000000)
    tol= 0.08
    check_sigmar_against_jeans(samp,pot,tol,
                               rmin=pot._scale/10.,rmax=pot._scale*10.,
                               bins=31)
    return None

def test_isotropic_nfw_beta():
    pot= potential.NFWPotential(amp=2.3,a=1.3)
    dfp= isotropicNFWdf(pot=pot)
    numpy.random.seed(10)
    samp= dfp.sample(n=3000000)
    tol= 8*1e-2
    check_beta(samp,pot,tol,rmin=pot._scale/5.,rmax=pot._scale*10.,bins=31)
    return None

def test_isotropic_nfw_dens_directint():
    pot= potential.NFWPotential(amp=2.3,a=1.3)
    dfp= isotropicNFWdf(pot=pot)
    tol= 1e-2 # only approx, normally 1e-7
    check_dens_directint(dfp,pot,tol,
                         lambda r: pot.dens(r,0),
                         rmin=pot._scale/10.,
                         rmax=pot._scale*10.,bins=31)
    return None

def test_isotropic_nfw_meanvr_directint():
    pot= potential.NFWPotential(amp=2.3,a=1.3)
    dfp= isotropicNFWdf(pot=pot)
    tol= 1e-8
    check_meanvr_directint(dfp,pot,tol,rmin=pot._scale/10.,
                           rmax=pot._scale*10.,bins=31)
    return None

def test_isotropic_nfw_sigmar_directint():
    pot= potential.NFWPotential(amp=2.3,a=1.3)
    dfp= isotropicNFWdf(pot=pot)
    tol= 1e-3 # only approx. normally 1e-5
    check_sigmar_against_jeans_directint(dfp,pot,tol,
                                         rmin=pot._scale/10.,
                                         rmax=pot._scale*10.,
                                         bins=31)
    return None

def test_isotropic_nfw_beta_directint():
    pot= potential.NFWPotential(amp=2.3,a=1.3)
    dfp= isotropicNFWdf(pot=pot)
    tol= 1e-8
    check_beta_directint(dfp,tol,rmin=pot._scale/10.,rmax=pot._scale*10.,
                         bins=31)
    return None

def test_isotropic_nfw_energyoutofbounds():
    pot= potential.NFWPotential(amp=2.3,a=1.3)
    dfp= isotropicNFWdf(pot=pot)
    assert numpy.all(numpy.fabs(dfp((numpy.arange(0.1,10.,0.1),1.1))) < 1e-8), 'Evaluating the isotropic NFW DF at E > 0 does not give zero'
    assert numpy.all(numpy.fabs(dfp((pot(0,0)-1e-4,1.1))) < 1e-8), 'Evaluating the isotropic NFW DF at E < Phi(0) does not give zero'
    return None

def test_isotropic_nfw_widrow_against_improved():
    # Test that using the Widrow (2000) prescription gives almost the same f(E)
    pot= potential.NFWPotential(amp=2.3,a=1.3)
    dfp= isotropicNFWdf(pot=pot)
    dfpw= isotropicNFWdf(pot=pot,widrow=True)
    Es= numpy.linspace(-dfp._Etildemax*0.999,0,101,endpoint=False)
    assert numpy.all(numpy.fabs(1.-dfp.fE(Es)/dfpw.fE(Es)) < 1e-2), 'isotropic NFW with widrow=True does not agree on f(E) with widrow=False'
    return None    

################################# EDDINGTON DF ################################
# For the following tests, we use a DehnenCoreSphericalPotential
def test_isotropic_eddington_selfconsist_dehnencore_dens_spherically_symmetric():
    pot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    dfp= eddingtondf(pot=pot)
    numpy.random.seed(10)
    samp= dfp.sample(n=100000)
    # Check spherical symmetry for different harmonics l,m
    tol= 1e-2
    check_spherical_symmetry(samp,0,0,tol)
    check_spherical_symmetry(samp,1,0,tol)
    check_spherical_symmetry(samp,1,-1,tol)
    check_spherical_symmetry(samp,1,1,tol)
    check_spherical_symmetry(samp,2,0,tol)
    check_spherical_symmetry(samp,2,-1,tol)
    check_spherical_symmetry(samp,2,-2,tol)
    check_spherical_symmetry(samp,2,1,tol)
    check_spherical_symmetry(samp,2,2,tol)
    # and some higher order ones
    check_spherical_symmetry(samp,3,1,tol)
    check_spherical_symmetry(samp,9,-6,tol)
    return None
    
def test_isotropic_eddington_selfconsist_dehnencore_dens_massprofile():
    # Do one with pot as list
    pot= [potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)]
    dfp= eddingtondf(pot=pot)
    numpy.random.seed(10)
    samp= dfp.sample(n=100000)
    tol= 5*1e-3
    check_spherical_massprofile(samp,lambda r: potential.mass(pot,r)\
                                /potential.mass(pot,numpy.amax(samp.r())),
                                tol,skip=1000)
    return None

def test_isotropic_eddington_selfconsist_dehnencore_sigmar():
    pot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    dfp= eddingtondf(pot=pot)
    numpy.random.seed(10)
    samp= dfp.sample(n=1000000)
    tol= 0.08
    check_sigmar_against_jeans(samp,pot,tol,
                               rmin=pot._scale/10.,rmax=pot._scale*10.,
                               bins=31)
    return None

def test_isotropic_eddington_selfconsist_dehnencore_beta():
    pot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    dfp= eddingtondf(pot=pot)
    numpy.random.seed(10)
    samp= dfp.sample(n=3000000)
    tol= 8*1e-2
    check_beta(samp,pot,tol,rmin=pot._scale/5.,rmax=pot._scale*10.,bins=31)
    return None

def test_isotropic_eddington_selfconsist_dehnencore_dens_directint():
    pot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    dfp= eddingtondf(pot=pot)
    tol= 1e-2 # only approx, normally 1e-7
    check_dens_directint(dfp,pot,tol,
                         lambda r: pot.dens(r,0),
                         rmin=pot._scale/10.,
                         rmax=pot._scale*10.,bins=31)
    return None

def test_isotropic_eddington_selfconsist_dehnencore_meanvr_directint():
    pot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    dfp= eddingtondf(pot=pot)
    tol= 1e-8
    check_meanvr_directint(dfp,pot,tol,rmin=pot._scale/10.,
                           rmax=pot._scale*10.,bins=31)
    return None

def test_isotropic_eddington_selfconsist_dehnencore_sigmar_directint():
    pot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    dfp= eddingtondf(pot=pot)
    tol= 1e-3 # only approx. normally 1e-5
    check_sigmar_against_jeans_directint(dfp,pot,tol,
                                         rmin=pot._scale/10.,
                                         rmax=pot._scale*10.,
                                         bins=31)
    return None

def test_isotropic_eddington_selfconsist_dehnencore_beta_directint():
    pot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    dfp= eddingtondf(pot=pot)
    tol= 1e-8
    check_beta_directint(dfp,tol,rmin=pot._scale/10.,rmax=pot._scale*10.,
                         bins=31)
    return None

def test_isotropic_eddington_selfconsist_dehnencore_energyoutofbounds():
    pot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    dfp= eddingtondf(pot=pot)
    assert numpy.all(numpy.fabs(dfp((numpy.arange(0.1,10.,0.1),1.1))) < 1e-8), 'Evaluating the eddington DF at E > 0 does not give zero'
    assert numpy.all(numpy.fabs(dfp((pot(0,0)-1e-4,1.1))) < 1e-8), 'Evaluating the isotropic NFW DF at E < Phi(0) does not give zero'
    return None

# For the following tests, we use a DehnenCoreSphericalPotential embedded in
# an NFW halo
def test_isotropic_eddington_dehnencore_in_nfw_dens_spherically_symmetric():
    pot= potential.NFWPotential(amp=2.3,a=1.3)
    denspot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    dfp= eddingtondf(pot=pot,denspot=denspot)
    numpy.random.seed(10)
    samp= dfp.sample(n=100000)
    # Check spherical symmetry for different harmonics l,m
    tol= 1e-2
    check_spherical_symmetry(samp,0,0,tol)
    check_spherical_symmetry(samp,1,0,tol)
    check_spherical_symmetry(samp,1,-1,tol)
    check_spherical_symmetry(samp,1,1,tol)
    check_spherical_symmetry(samp,2,0,tol)
    check_spherical_symmetry(samp,2,-1,tol)
    check_spherical_symmetry(samp,2,-2,tol)
    check_spherical_symmetry(samp,2,1,tol)
    check_spherical_symmetry(samp,2,2,tol)
    # and some higher order ones
    check_spherical_symmetry(samp,3,1,tol)
    check_spherical_symmetry(samp,9,-6,tol)
    return None
    
def test_isotropic_eddington_dehnencore_in_nfw_dens_massprofile():
    pot= potential.NFWPotential(amp=2.3,a=1.3)
    denspot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    dfp= eddingtondf(pot=pot,denspot=denspot)
    numpy.random.seed(10)
    samp= dfp.sample(n=100000)
    tol= 5*1e-3
    check_spherical_massprofile(samp,lambda r: denspot.mass(r)\
                                /denspot.mass(numpy.amax(samp.r())),
                                tol,skip=1000)
    return None

def test_isotropic_eddington_dehnencore_in_nfw_sigmar():
    # Use list
    pot= [potential.NFWPotential(amp=2.3,a=1.3)]
    denspot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    dfp= eddingtondf(pot=pot,denspot=denspot)
    numpy.random.seed(10)
    samp= dfp.sample(n=1000000)
    tol= 0.08
    check_sigmar_against_jeans(samp,pot,tol,
                               dens=lambda r: denspot.dens(r,0,use_physical=False),
                               rmin=pot[0]._scale/10.,rmax=pot[0]._scale*10.,
                               bins=31)
    return None

def test_isotropic_eddington_dehnencore_in_nfw_beta():
    pot= potential.NFWPotential(amp=2.3,a=1.3)
    # Use list
    denspot= [potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)]
    dfp= eddingtondf(pot=pot,denspot=denspot)
    numpy.random.seed(10)
    samp= dfp.sample(n=3000000)
    tol= 8*1e-2
    check_beta(samp,pot,tol,rmin=pot._scale/5.,rmax=pot._scale*10.,bins=31)
    return None

def test_isotropic_eddington_dehnencore_in_nfw_dens_directint():
    # Lists for all!
    pot= [potential.NFWPotential(amp=2.3,a=1.3)]
    denspot= [potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)]
    dfp= eddingtondf(pot=pot,denspot=denspot)
    tol= 1e-2 # only approx, normally 1e-7
    check_dens_directint(dfp,pot,tol,
                         lambda r: potential.evaluateDensities(denspot,r,0),
                         rmin=pot[0]._scale/10.,
                         rmax=pot[0]._scale*10.,bins=31)
    return None

def test_isotropic_eddington_dehnencore_in_nfw_meanvr_directint():
    pot= potential.NFWPotential(amp=2.3,a=1.3)
    denspot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    dfp= eddingtondf(pot=pot,denspot=denspot)
    tol= 1e-8
    check_meanvr_directint(dfp,pot,tol,rmin=pot._scale/10.,
                           rmax=pot._scale*10.,bins=31)
    return None

def test_isotropic_eddington_dehnencore_in_nfw_sigmar_directint():
    pot= potential.NFWPotential(amp=2.3,a=1.3)
    denspot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    dfp= eddingtondf(pot=pot,denspot=denspot)
    tol= 1e-3 # only approx. normally 1e-5
    check_sigmar_against_jeans_directint(dfp,pot,tol,
                                         dens=lambda r: denspot.dens(r,0),
                                         rmin=pot._scale/10.,
                                         rmax=pot._scale*10.,
                                         bins=31)
    return None

def test_isotropic_eddington_dehnencore_in_nfw_beta_directint():
    pot= potential.NFWPotential(amp=2.3,a=1.3)
    denspot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    dfp= eddingtondf(pot=pot,denspot=denspot)
    tol= 1e-8
    check_beta_directint(dfp,tol,rmin=pot._scale/10.,rmax=pot._scale*10.,
                         bins=31)
    return None

def test_isotropic_eddington_dehnencore_in_nfw_energyoutofbounds():
    pot= potential.NFWPotential(amp=2.3,a=1.3)
    denspot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    dfp= eddingtondf(pot=pot,denspot=denspot)
    assert numpy.all(numpy.fabs(dfp((numpy.arange(0.1,10.,0.1),1.1))) < 1e-8), 'Evaluating the isotropic NFW DF at E > 0 does not give zero'
    assert numpy.all(numpy.fabs(dfp((pot(0,0)-1e-4,1.1))) < 1e-8), 'Evaluating the isotropic NFW DF at E < Phi(0) does not give zero'
    return None

################################# KING DF #####################################
def test_king_dens_spherically_symmetric():
    dfk= kingdf(W0=3.,M=2.3,rt=1.76)
    numpy.random.seed(10)
    samp= dfk.sample(n=100000)
    # Check spherical symmetry for different harmonics l,m
    tol= 1e-2
    check_spherical_symmetry(samp,0,0,tol)
    check_spherical_symmetry(samp,1,0,tol)
    check_spherical_symmetry(samp,1,-1,tol)
    check_spherical_symmetry(samp,1,1,tol)
    check_spherical_symmetry(samp,2,0,tol)
    check_spherical_symmetry(samp,2,-1,tol)
    check_spherical_symmetry(samp,2,-2,tol)
    check_spherical_symmetry(samp,2,1,tol)
    check_spherical_symmetry(samp,2,2,tol)
    # and some higher order ones
    check_spherical_symmetry(samp,3,1,tol)
    check_spherical_symmetry(samp,9,-6,tol)
    return None
    
def test_king_dens_massprofile():
    pot= potential.KingPotential(W0=3.,M=2.3,rt=1.76)
    dfk= kingdf(W0=3.,M=2.3,rt=1.76)
    numpy.random.seed(10)
    samp= dfk.sample(n=100000)
    tol= 1e-2
    check_spherical_massprofile(samp,lambda r: pot.mass(r)\
                                /pot.mass(numpy.amax(samp.r())),
                                tol,skip=4000)
    return None

def test_king_sigmar():
    W0s= [1.,3.,9.]
    for W0 in W0s:
        pot= potential.KingPotential(W0=W0,M=2.3,rt=1.76)
        dfk= kingdf(W0=W0,M=2.3,rt=1.76)
        numpy.random.seed(10)
        samp= dfk.sample(n=1000000)
        # lower tolerance closer to rt because fewer stars there
        tol= 0.1
        check_sigmar_against_jeans(samp,pot,tol,beta=0.,
                                   rmin=dfk._scale/10.,rmax=dfk.rt*0.7,bins=31)
        tol= 0.2
        check_sigmar_against_jeans(samp,pot,tol,beta=0.,
                                   rmin=dfk.rt*0.8,rmax=dfk.rt*0.95,bins=5)
    return None

def test_king_beta():
    pot= potential.KingPotential(W0=3.,M=2.3,rt=1.76)
    dfk= kingdf(W0=3.,M=2.3,rt=1.76)
    numpy.random.seed(10)
    samp= dfk.sample(n=1000000)
    tol= 6*1e-2
    # lower tolerance closer to rt because fewer stars there
    tol= 0.135
    check_beta(samp,pot,tol,beta=0.,rmin=dfk._scale/10.,rmax=dfk.rt,
               bins=31)
    return None
               
def test_king_dens_directint():
    pot= potential.KingPotential(W0=3.,M=2.3,rt=1.76)
    dfk= kingdf(W0=3.,M=2.3,rt=1.76)
    tol= 0.02
    check_dens_directint(dfk,pot,tol,
                         lambda r: dfk.dens(r),
                         rmin=dfk._scale/10.,
                         rmax=dfk.rt*0.7,bins=31)
    return None

def test_king_sigmar_directint():
    pot= potential.KingPotential(W0=3.,M=2.3,rt=1.76)
    dfk= kingdf(W0=3.,M=2.3,rt=1.76)
    tol= 0.05 # Jeans isn't that accurate for this rather difficult case
    check_sigmar_against_jeans_directint(dfk,pot,tol,beta=0.,
                                         rmin=dfk._scale/10.,
                                         rmax=dfk.rt*0.7,bins=31)
    return None

def test_king_beta_directint():
    dfk= kingdf(W0=3.,M=2.3,rt=1.76)
    tol= 1e-8
    check_beta_directint(dfk,tol,beta=0.,
                         rmin=dfk._scale/10.,rmax=dfk.rt*0.7,bins=31)
    return None

############################### OSIPKOV-MERRITT DF ############################
# For the following tests, we use a DehnenCoreSphericalPotential
osipkovmerritt_dfs_selfconsist= None # re-use in other tests
def test_osipkovmerritt_selfconsist_dehnencore_dens_spherically_symmetric():
    pot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    ras= [2.3,5.7]
    global osipkovmerritt_dfs_selfconsist
    osipkovmerritt_dfs_selfconsist= []
    for ra in ras:
        dfh= osipkovmerrittdf(pot=pot,ra=ra)
        osipkovmerritt_dfs_selfconsist.append(dfh)
        numpy.random.seed(10)
        samp= dfh.sample(n=100000)
        # Check spherical symmetry for different harmonics l,m
        tol= 1e-2
        check_spherical_symmetry(samp,0,0,tol)
        check_spherical_symmetry(samp,1,0,tol)
        check_spherical_symmetry(samp,1,-1,tol)
        check_spherical_symmetry(samp,1,1,tol)
        check_spherical_symmetry(samp,2,0,tol)
        check_spherical_symmetry(samp,2,-1,tol)
        check_spherical_symmetry(samp,2,-2,tol)
        check_spherical_symmetry(samp,2,1,tol)
        check_spherical_symmetry(samp,2,2,tol)
        # and some higher order ones
        check_spherical_symmetry(samp,3,1,tol)
        check_spherical_symmetry(samp,9,-6,tol)
    return None
    
def test_osipkovmerritt_selfconsist_dehnencore_dens_massprofile():
    pot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    ras= [2.3,5.7]
    for ra,dfh in zip(ras,osipkovmerritt_dfs_selfconsist):
        numpy.random.seed(10)
        samp= dfh.sample(n=100000)
        tol= 5*1e-3
        check_spherical_massprofile(samp,lambda r: pot.mass(r)\
                                    /pot.mass(numpy.amax(samp.r())),
                                    tol,skip=1000)
    return None

def test_osipkovmerritt_selfconsist_dehnencore_sigmar():
    pot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    ras= [2.3,5.7]
    for ra,dfh in zip(ras,osipkovmerritt_dfs_selfconsist):
        numpy.random.seed(10)
        samp= dfh.sample(n=300000)
        tol= 0.1
        check_sigmar_against_jeans(samp,pot,tol,
                                   beta=lambda r: 1./(1.+ra**2./r**2.),
                                   rmin=pot._scale/10.,rmax=pot._scale*10.,
                                   bins=31)
    return None

def test_osipkovmerritt_selfconsist_dehnencore_beta():
    pot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    ras= [2.3,5.7]
    for ra,dfh in zip(ras,osipkovmerritt_dfs_selfconsist):
        numpy.random.seed(10)
        samp= dfh.sample(n=300000)
        tol= 0.1
        # rmin larger than usual to avoid low number sampling
        check_beta(samp,pot,tol,beta=lambda r: 1./(1.+ra**2./r**2.),
                   rmin=pot._scale/3.,rmax=pot._scale*10.,bins=31)
    return None

def test_osipkovmerritt_selfconsist_dehnencore_dens_directint():
    pot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    ras= [2.3,5.7]
    for ra,dfh in zip(ras[:1],osipkovmerritt_dfs_selfconsist[:1]):
        tol= 1e-4
        check_dens_directint(dfh,pot,tol,
                             lambda r: pot.dens(r,0),
                             rmin=pot._scale/10.,
                             rmax=pot._scale*10.,bins=3)
    return None

def test_osipkovmerritt_selfconsist_dehnencore_meanvr_directint():
    pot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    ras= [2.3,5.7]
    for ra,dfh in zip(ras[1:],osipkovmerritt_dfs_selfconsist[1:]):
        tol= 1e-8
        check_meanvr_directint(dfh,pot,tol,rmin=pot._scale/10.,
                               rmax=pot._scale*10.,bins=3)
    return None

def test_osipkovmerritt_selfconsist_dehnencore_sigmar_directint():
    pot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    ras= [2.3,5.7]
    for ra,dfh in zip(ras[:1],osipkovmerritt_dfs_selfconsist[:1]):
        tol= 1e-4
        check_sigmar_against_jeans_directint(dfh,pot,tol,
                                             beta=lambda r: 1./(1.+ra**2./r**2.),
                                             rmin=pot._scale/10.,
                                             rmax=pot._scale*10.,
                                             bins=3)
    return None

def test_osipkovmerritt_selfconsist_dehnencore_beta_directint():
    pot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    ras= [2.3,5.7]
    for ra,dfh in zip(ras[1:],osipkovmerritt_dfs_selfconsist[1:]):
        tol= 1e-8
        check_beta_directint(dfh,tol,beta=lambda r: 1./(1.+ra**2./r**2.),
                             rmin=pot._scale/10.,
                             rmax=pot._scale*10.,
                             bins=3)
    return None

def test_osipkovmerritt_selfconsist_dehnencore_Qoutofbounds():
    pot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    ras= [2.3,5.7]
    for ra,dfh in zip(ras,osipkovmerritt_dfs_selfconsist):
        assert numpy.all(numpy.fabs(dfh((numpy.arange(0.1,10.,0.1),1.1))) < 1e-8), 'Evaluating the Osipkov-Merritt DF at E > 0 does not give zero'
        # The next one is not actually a physical orbit...
        assert numpy.all(numpy.fabs(dfh((pot(0,0)-1e-1,0.1))) < 1e-8), 'Evaluating the Osipkov-Merritt DF at E < -GM/a does not give zero'
        assert numpy.all(numpy.fabs(dfh((-1e-4,1.1))) < 1e-8), 'Evaluating the Osipkov-Merritt DF at Q < 0 does not give zero'
    return None

# For the following tests, we use a DehnenCoreSphericalPotential embedded in
# an NFW halo
osipkovmerritt_dfs_dehnencore_in_nfw= None # re-use in other tests
def test_osipkovmerritt_dehnencore_in_nfw_dens_spherically_symmetric():
    pot= potential.NFWPotential(amp=2.3,a=1.3)
    denspot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    ras= [2.3,5.7]
    global osipkovmerritt_dfs_dehnencore_in_nfw
    osipkovmerritt_dfs_dehnencore_in_nfw= []
    for ra in ras:
        dfh= osipkovmerrittdf(pot=pot,denspot=denspot,ra=ra)
        osipkovmerritt_dfs_dehnencore_in_nfw.append(dfh)
        numpy.random.seed(10)
        samp= dfh.sample(n=100000)
        # Check spherical symmetry for different harmonics l,m
        tol= 1e-2
        check_spherical_symmetry(samp,0,0,tol)
        check_spherical_symmetry(samp,1,0,tol)
        check_spherical_symmetry(samp,1,-1,tol)
        check_spherical_symmetry(samp,1,1,tol)
        check_spherical_symmetry(samp,2,0,tol)
        check_spherical_symmetry(samp,2,-1,tol)
        check_spherical_symmetry(samp,2,-2,tol)
        check_spherical_symmetry(samp,2,1,tol)
        check_spherical_symmetry(samp,2,2,tol)
        # and some higher order ones
        check_spherical_symmetry(samp,3,1,tol)
        check_spherical_symmetry(samp,9,-6,tol)
    return None
    
def test_osipkovmerritt_dehnencore_in_nfw_dens_massprofile():
    pot= potential.NFWPotential(amp=2.3,a=1.3)
    denspot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    ras= [2.3,5.7]
    for ra,dfh in zip(ras,osipkovmerritt_dfs_dehnencore_in_nfw):
        numpy.random.seed(10)
        samp= dfh.sample(n=100000)
        tol= 5*1e-3
        check_spherical_massprofile(samp,lambda r: denspot.mass(r)\
                                    /denspot.mass(numpy.amax(samp.r())),
                                    tol,skip=1000)
    return None

def test_osipkovmerritt_dehnencore_in_nfw_sigmar():
    # Use list
    pot= [potential.NFWPotential(amp=2.3,a=1.3)]
    denspot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    ras= [2.3,5.7]
    for ra,dfh in zip(ras[:1],osipkovmerritt_dfs_dehnencore_in_nfw[:1]):
        numpy.random.seed(10)
        samp= dfh.sample(n=300000)
        tol= 0.07
        # rmin larger than usual to avoid low number sampling  
        check_sigmar_against_jeans(samp,pot,tol,
                                   dens=lambda r: denspot.dens(r,0),
                                   beta=lambda r: 1./(1.+ra**2./r**2.),
                                   rmin=pot[0]._scale/3.,
                                   rmax=pot[0]._scale*10.,
                                   bins=31)
    return None

def test_osipkovmerritt_dehnencore_in_nfw_beta():
    # Use list
    pot= potential.NFWPotential(amp=2.3,a=1.3)
    denspot= [potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)]
    ras= [2.3,5.7]
    for ra,dfh in zip(ras[1:],osipkovmerritt_dfs_dehnencore_in_nfw[1:]):
        numpy.random.seed(10)
        samp= dfh.sample(n=300000)
        tol= 0.07
        # rmin larger than usual to avoid low number sampling
        check_beta(samp,pot,tol,beta=lambda r: 1./(1.+ra**2./r**2.),
                   rmin=pot._scale/3.,rmax=pot._scale*10.,bins=31)
    return None

def test_osipkovmerritt_dehnencore_in_nfw_dens_directint():
    # Use list for both
    pot= [potential.NFWPotential(amp=2.3,a=1.3)]
    denspot= [potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)]
    ras= [2.3,5.7]
    for ra,dfh in zip(ras[:1],osipkovmerritt_dfs_dehnencore_in_nfw[:1]):
        tol= 3e-4
        check_dens_directint(dfh,pot,tol,
                             lambda r: denspot[0].dens(r,0),
                             rmin=pot[0]._scale/10.,
                             rmax=pot[0]._scale*10.,bins=3)
    return None

def test_osipkovmerritt_dehnencore_in_nfw_meanvr_directint():
    pot= potential.NFWPotential(amp=2.3,a=1.3)
    denspot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    ras= [2.3,5.7]
    for ra,dfh in zip(ras[1:],osipkovmerritt_dfs_dehnencore_in_nfw[1:]):
        tol= 1e-8
        check_meanvr_directint(dfh,pot,tol,rmin=pot._scale/10.,
                               rmax=pot._scale*10.,bins=3)
    return None

def test_osipkovmerritt_dehnencore_in_nfw_sigmar_directint():
    pot= potential.NFWPotential(amp=2.3,a=1.3)
    denspot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    ras= [2.3,5.7]
    for ra,dfh in zip(ras[:1],osipkovmerritt_dfs_dehnencore_in_nfw[:1]):
        tol= 2e-4
        check_sigmar_against_jeans_directint(dfh,pot,tol,
                                             dens=lambda r: denspot.dens(r,0),
                                             beta=lambda r: 1./(1.+ra**2./r**2.),
                                             rmin=pot._scale/10.,
                                             rmax=pot._scale*10.,
                                             bins=3)
    return None

def test_osipkovmerritt_dehnencore_in_nfw_beta_directint():
    pot= potential.NFWPotential(amp=2.3,a=1.3)
    denspot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    ras= [2.3,5.7]
    for ra,dfh in zip(ras[1:],osipkovmerritt_dfs_dehnencore_in_nfw[1:]):
        tol= 1e-8
        check_beta_directint(dfh,tol,beta=lambda r: 1./(1.+ra**2./r**2.),
                             rmin=pot._scale/10.,
                             rmax=pot._scale*10.,
                             bins=3)
    return None

def test_osipkovmerritt_dehnencore_in_nfw_Qoutofbounds():
    pot= potential.NFWPotential(amp=2.3,a=1.3)
    denspot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    ras= [2.3,5.7]
    for ra,dfh in zip(ras,osipkovmerritt_dfs_dehnencore_in_nfw):
        assert numpy.all(numpy.fabs(dfh((numpy.arange(0.1,10.,0.1),1.1))) < 1e-8), 'Evaluating the Osipkov-Merritt DF at E > 0 does not give zero'
        # The next one is not actually a physical orbit...
        assert numpy.all(numpy.fabs(dfh((pot(0,0)-1e-1,0.1))) < 1e-8), 'Evaluating the Osipkov-Merritt DF at E < -GM/a does not give zero'
        assert numpy.all(numpy.fabs(dfh((-1e-4,1.1))) < 1e-8), 'Evaluating the Osipkov-Merritt DF at Q < 0 does not give zero'
    return None

################################ CONSTANT-BETA DF #############################
# For the following tests, we use a DehnenCoreSphericalPotential
constantbeta_dfs_selfconsist= None # re-use in other tests
def test_constantbeta_selfconsist_dehnencore_dens_spherically_symmetric():
    if WIN32: return None # skip on appveyor, because no JAX
    pot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    twobetas= [-1]
    global constantbeta_dfs_selfconsist
    constantbeta_dfs_selfconsist= []
    for twobeta in twobetas:
        dfh= constantbetadf(pot=pot,twobeta=twobeta)
        constantbeta_dfs_selfconsist.append(dfh)
        numpy.random.seed(10)
        samp= dfh.sample(n=100000)
        # Check spherical symmetry for different harmonics l,m
        tol= 1e-2
        check_spherical_symmetry(samp,0,0,tol)
        check_spherical_symmetry(samp,1,0,tol)
        check_spherical_symmetry(samp,1,-1,tol)
        check_spherical_symmetry(samp,1,1,tol)
        check_spherical_symmetry(samp,2,0,tol)
        check_spherical_symmetry(samp,2,-1,tol)
        check_spherical_symmetry(samp,2,-2,tol)
        check_spherical_symmetry(samp,2,1,tol)
        check_spherical_symmetry(samp,2,2,tol)
        # and some higher order ones
        check_spherical_symmetry(samp,3,1,tol)
        check_spherical_symmetry(samp,9,-6,tol)
    return None
    
def test_constantbeta_selfconsist_dehnencore_dens_massprofile():
    if WIN32: return None # skip on appveyor, because no JAX
    pot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    twobetas= [-1]
    for twobeta,dfh in zip(twobetas,constantbeta_dfs_selfconsist):
        numpy.random.seed(10)
        samp= dfh.sample(n=100000)
        tol= 5*1e-3
        check_spherical_massprofile(samp,lambda r: pot.mass(r)\
                                    /pot.mass(numpy.amax(samp.r())),
                                    tol,skip=1000)
    return None

def test_constantbeta_selfconsist_dehnencore_sigmar():
    if WIN32: return None # skip on appveyor, because no JAX
    pot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    twobetas= [-1]
    for twobeta,dfh in zip(twobetas,constantbeta_dfs_selfconsist):
        numpy.random.seed(10)
        samp= dfh.sample(n=1000000)
        tol= 0.1
        check_sigmar_against_jeans(samp,pot,tol,
                                   beta=twobeta/2.,
                                   rmin=pot._scale/10.,rmax=pot._scale*10.,
                                   bins=31)
    return None

def test_constantbeta_selfconsist_dehnencore_beta():
    if WIN32: return None # skip on appveyor, because no JAX
    pot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    twobetas= [-1]
    for twobeta,dfh in zip(twobetas,constantbeta_dfs_selfconsist):
        numpy.random.seed(10)
        samp= dfh.sample(n=1000000)
        tol= 0.1
        # rmin larger than usual to avoid low number sampling
        check_beta(samp,pot,tol,beta=twobeta/2.,
                   rmin=pot._scale/3.,rmax=pot._scale*10.,bins=31)
    return None

def test_constantbeta_selfconsist_dehnencore_dens_directint():
    if WIN32: return None # skip on appveyor, because no JAX
    pot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    twobetas= [-1]
    for twobeta,dfh in zip(twobetas,constantbeta_dfs_selfconsist):
        tol= 1e-4
        check_dens_directint(dfh,pot,tol,
                             lambda r: pot.dens(r,0),
                             rmin=pot._scale/10.,
                             rmax=pot._scale*10.,bins=3)
    return None

def test_constantbeta_selfconsist_dehnencore_meanvr_directint():
    if WIN32: return None # skip on appveyor, because no JAX
    pot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    twobetas= [-1]
    for twobeta,dfh in zip(twobetas,constantbeta_dfs_selfconsist):
        tol= 1e-8
        check_meanvr_directint(dfh,pot,tol,rmin=pot._scale/10.,
                               rmax=pot._scale*10.,bins=3)
    return None

def test_constantbeta_selfconsist_dehnencore_sigmar_directint():
    if WIN32: return None # skip on appveyor, because no JAX
    pot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    twobetas= [-1]
    for twobeta,dfh in zip(twobetas,constantbeta_dfs_selfconsist):
        tol= 1e-4
        check_sigmar_against_jeans_directint(dfh,pot,tol,
                                             beta=twobeta/2.,
                                             rmin=pot._scale/10.,
                                             rmax=pot._scale*10.,
                                             bins=3)
    return None

# We don't do this test, because it is trivially satisfied by
# any f(E,L) = L^(-2beta) f1(E)
#def test_constantbeta_selfconsist_dehnencore_beta_directint():
#    if WIN32: return None # skip on appveyor, because no JAX
#    pot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
#    twobetas= [-1]
#    for twobeta,dfh in zip(twobetas,constantbeta_dfs_selfconsist):
#        tol= 1e-8
#        check_beta_directint(dfh,tol,beta=twobeta/2.,
#                             rmin=pot._scale/10.,
#                             rmax=pot._scale*10.,
#                             bins=3)
#    return None

def test_constantbeta_selfconsist_dehnencore_Qoutofbounds():
    if WIN32: return None # skip on appveyor, because no JAX
    pot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    twobetas= [-1]
    for twobeta,dfh in zip(twobetas,constantbeta_dfs_selfconsist):
        assert numpy.all(numpy.fabs(dfh((numpy.arange(0.1,10.,0.1),1.1))) < 1e-8), 'Evaluating the constant-beta DF at E > 0 does not give zero'
        # The next one is not actually a physical orbit...
        assert numpy.all(numpy.fabs(dfh((pot(0,0)-1e-1,0.1))) < 1e-8), 'Evaluating the constant-beta DF at E < -GM/a does not give zero'
        assert numpy.all(numpy.fabs(dfh((-1e-4,1.1))) < 1e-8), 'Evaluating the Osipkov-Merritt DF at Q < 0 does not give zero'
    return None

# Also some tests with rmin in sampling
def test_constantbeta_selfconsist_dehnencore_rmin_inbounds():
    if WIN32: return None # skip on appveyor, because no JAX
    pot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    twobetas= [-1]
    rmin = 0.5
    for twobeta,dfh in zip(twobetas,constantbeta_dfs_selfconsist):
        samp= dfh.sample(n=1000000,rmin=rmin)
        assert numpy.min(samp.r()) >= rmin, 'Sample minimum r less than rmin'
        # Change rmin
        samp= dfh.sample(n=1000000,rmin=rmin+1.)
        assert numpy.min(samp.r()) >= rmin+1., 'Sample minimum r less than rmin'
    return None

# For the following tests, we use a DehnenCoreSphericalPotential embedded in
# an NFW halo
constantbeta_dfs_dehnencore_in_nfw= None # re-use in other tests
def test_constantbeta_dehnencore_in_nfw_dens_spherically_symmetric():
    if WIN32: return None # skip on appveyor, because no JAX
    pot= potential.NFWPotential(amp=2.3,a=1.3)
    denspot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    twobetas= [0.5]
    global constantbeta_dfs_dehnencore_in_nfw
    constantbeta_dfs_dehnencore_in_nfw= []   
    for twobeta in twobetas:
        dfh= constantbetadf(pot=pot,denspot=denspot,twobeta=twobeta)
        constantbeta_dfs_dehnencore_in_nfw.append(dfh)
        numpy.random.seed(10)
        samp= dfh.sample(n=100000)
        # Check spherical symmetry for different harmonics l,m
        tol= 1e-2
        check_spherical_symmetry(samp,0,0,tol)
        check_spherical_symmetry(samp,1,0,tol)
        check_spherical_symmetry(samp,1,-1,tol)
        check_spherical_symmetry(samp,1,1,tol)
        check_spherical_symmetry(samp,2,0,tol)
        check_spherical_symmetry(samp,2,-1,tol)
        check_spherical_symmetry(samp,2,-2,tol)
        check_spherical_symmetry(samp,2,1,tol)
        check_spherical_symmetry(samp,2,2,tol)
        # and some higher order ones
        check_spherical_symmetry(samp,3,1,tol)
        check_spherical_symmetry(samp,9,-6,tol)
    return None
    
def test_constantbeta_dehnencore_in_nfw_dens_massprofile():
    if WIN32: return None # skip on appveyor, because no JAX
    pot= potential.NFWPotential(amp=2.3,a=1.3)
    denspot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    twobetas= [0.5]
    for twobeta,dfh in zip(twobetas,constantbeta_dfs_dehnencore_in_nfw):
        numpy.random.seed(10)
        samp= dfh.sample(n=100000)
        tol= 5*1e-3
        check_spherical_massprofile(samp,lambda r: denspot.mass(r)\
                                    /denspot.mass(numpy.amax(samp.r())),
                                    tol,skip=1000)
    return None

def test_constantbeta_dehnencore_in_nfw_sigmar():
    if WIN32: return None # skip on appveyor, because no JAX
    # Use list
    pot= [potential.NFWPotential(amp=2.3,a=1.3)]
    denspot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    twobetas= [0.5]
    for twobeta,dfh in zip(twobetas,constantbeta_dfs_dehnencore_in_nfw):
        numpy.random.seed(10)
        samp= dfh.sample(n=1000000)
        tol= 0.07
        # rmin larger than usual to avoid low number sampling  
        check_sigmar_against_jeans(samp,pot,tol,
                                   dens=lambda r: denspot.dens(r,0),
                                   beta=twobeta/2.,
                                   rmin=pot[0]._scale/3.,
                                   rmax=pot[0]._scale*10.,
                                   bins=31)
    return None

def test_constantbeta_dehnencore_in_nfw_beta():
    if WIN32: return None # skip on appveyor, because no JAX
    # Use list
    pot= potential.NFWPotential(amp=2.3,a=1.3)
    denspot= [potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)]
    twobetas= [0.5]
    for twobeta,dfh in zip(twobetas,constantbeta_dfs_dehnencore_in_nfw):
        numpy.random.seed(10)
        samp= dfh.sample(n=1000000)
        tol= 0.07
        # rmin larger than usual to avoid low number sampling
        check_beta(samp,pot,tol,beta=twobeta/2.,
                   rmin=pot._scale/3.,rmax=pot._scale*10.,bins=31)
    return None

# Here in this case so it gets run before fE is changed for directint tests
def test_constantbeta_dehnencore_in_nfw_Qoutofbounds():
    if WIN32: return None # skip on appveyor, because no JAX
    pot= potential.NFWPotential(amp=2.3,a=1.3)
    denspot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    twobetas= [0.5]
    for twobeta,dfh in zip(twobetas,constantbeta_dfs_dehnencore_in_nfw):
        assert numpy.all(numpy.fabs(dfh((numpy.arange(0.1,10.,0.1),1.1))) < 1e-8), 'Evaluating the Osipkov-Merritt DF at E > 0 does not give zero'
        # The next one is not actually a physical orbit...
        assert numpy.all(numpy.fabs(dfh((pot(0,0)-1e-1,0.1))) < 1e-8), 'Evaluating the Osipkov-Merritt DF at E < -GM/a does not give zero'
        assert numpy.all(numpy.fabs(dfh((-1e-4,1.1))) < 1e-8), 'Evaluating the Osipkov-Merritt DF at Q < 0 does not give zero'
    return None

def test_constantbeta_dehnencore_in_nfw_dens_directint():
    if WIN32: return None # skip on appveyor, because no JAX
    # Use list for both
    pot= [potential.NFWPotential(amp=2.3,a=1.3)]
    denspot= [potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)]
    twobetas= [0.5]
    for twobeta,dfh in zip(twobetas,constantbeta_dfs_dehnencore_in_nfw):
        dfh.fE= lambda x: dfh._fE_interp(x)
        tol= 3e-4
        check_dens_directint(dfh,pot,tol,
                             lambda r: denspot[0].dens(r,0),
                             rmin=pot[0]._scale/10.,
                             rmax=pot[0]._scale*10.,bins=3)
    return None

def test_constantbeta_dehnencore_in_nfw_meanvr_directint():
    if WIN32: return None # skip on appveyor, because no JAX
    pot= potential.NFWPotential(amp=2.3,a=1.3)
    denspot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    twobetas= [0.5]
    for twobeta,dfh in zip(twobetas,constantbeta_dfs_dehnencore_in_nfw):
        dfh.fE= lambda x: dfh._fE_interp(x)
        tol= 1e-8
        check_meanvr_directint(dfh,pot,tol,rmin=pot._scale/10.,
                               rmax=pot._scale*10.,bins=3)
    return None

def test_constantbeta_dehnencore_in_nfw_sigmar_directint():
    if WIN32: return None # skip on appveyor, because no JAX
    pot= potential.NFWPotential(amp=2.3,a=1.3)
    denspot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
    twobetas= [0.5]
    for twobeta,dfh in zip(twobetas,constantbeta_dfs_dehnencore_in_nfw):
        dfh.fE= lambda x: dfh._fE_interp(x)
        tol= 2e-4
        check_sigmar_against_jeans_directint(dfh,pot,tol,
                                             dens=lambda r: denspot.dens(r,0),
                                             beta=twobeta/2.,
                                             rmin=pot._scale/10.,
                                             rmax=pot._scale*10.,
                                             bins=3)
    return None

#def test_constantbeta_dehnencore_in_nfw_beta_directint():
#    if WIN32: return None # skip on appveyor, because no JAX
#    pot= potential.NFWPotential(amp=2.3,a=1.3)
#    denspot= potential.DehnenCoreSphericalPotential(amp=2.5,a=1.15)
#    twobetas= [0.5]
#    for twobeta,dfh in zip(twobetas,constantbeta_dfs_dehnencore_in_nfw):
#        dfh.fE= lambda x: dfh._fE_interp(x)
#        tol= 1e-8
#        check_beta_directint(dfh,tol,beta=twobeta/2.,
#                             rmin=pot._scale/10.,
#                             rmax=pot._scale*10.,
#                             bins=3)
#    return None

########################### TESTS OF ERRORS AND WARNINGS#######################

def test_isotropic_hernquist_nopot():
    with pytest.raises(AssertionError)  as excinfo:
        dfh= isotropicHernquistdf()
    assert str(excinfo.value) == 'pot= must be potential.HernquistPotential', 'Error message when not supplying the potential is incorrect'
    return None

def test_isotropic_hernquist_wrongpot():
    pot= potential.JaffePotential(amp=2.,a=1.3)   
    with pytest.raises(AssertionError)  as excinfo:
        dfh= isotropicHernquistdf(pot=pot)
    assert str(excinfo.value) == 'pot= must be potential.HernquistPotential', 'Error message when not supplying the potential is incorrect'
    return None

def test_anisotropic_hernquist_nopot():
    with pytest.raises(AssertionError)  as excinfo:
        dfh= constantbetaHernquistdf()
    assert str(excinfo.value) == 'pot= must be potential.HernquistPotential', 'Error message when not supplying the potential is incorrect'
    return None

def test_anisotropic_hernquist_wrongpot():
    pot= potential.JaffePotential(amp=2.,a=1.3)   
    with pytest.raises(AssertionError)  as excinfo:
        dfh= constantbetaHernquistdf(pot=pot)
    assert str(excinfo.value) == 'pot= must be potential.HernquistPotential', 'Error message when not supplying the potential is incorrect'
    return None

def test_anisotropic_hernquist_negdf():
    pot= potential.HernquistPotential(amp=2.3,a=1.3)
    # beta > 0.5 has negative DF parts
    dfh= constantbetaHernquistdf(pot=pot,beta=0.7)
    with pytest.warns(None) as record:
        samp= dfh.sample(n=100)
    raisedWarning= False
    for rec in record:
        # check that the message matches
        raisedWarning+= (str(rec.message.args[0]) == "The DF appears to have negative regions; we'll try to ignore these for sampling the DF, but this may adversely affect the generated samples. Proceed with care!")
    assert raisedWarning, "Using an anisotropic Hernquist DF that has negative parts should have raised a warning, but didn't"
    
############################# TESTS OF UNIT HANDLING###########################

# Test that setting up a DF with unit conversion parameters that are
# incompatible with that of the underlying potential fails
def test_isotropic_hernquist_incompatibleunits():
    pot= potential.HernquistPotential(amp=2.,a=1.3,ro=9.,vo=210.)
    with pytest.raises(RuntimeError):
        dfh= isotropicHernquistdf(pot=pot,ro=8.,vo=210.)
    with pytest.raises(RuntimeError):
        dfh= isotropicHernquistdf(pot=pot,ro=9.,vo=230.)
    with pytest.raises(RuntimeError):
        dfh= isotropicHernquistdf(pot=pot,ro=8.,vo=230.)
    return None

# Test that setting up a DF with unit conversion parameters that are
# incompatible between the potential and the density fails
def test_eddington_pot_denspot_incompatibleunits():
    pot= potential.HernquistPotential(amp=2.,a=1.3,ro=9.,vo=210.)
    denspot= potential.NFWPotential(amp=2.,a=1.3,ro=8.,vo=200.)
    with pytest.raises(RuntimeError):
        denspot= potential.NFWPotential(amp=2.,a=1.3,ro=8.,vo=210.)
        dfh= eddingtondf(pot=pot,denspot=denspot)
    with pytest.raises(RuntimeError):
        denspot= potential.NFWPotential(amp=2.,a=1.3,ro=9.,vo=230.)
        dfh= eddingtondf(pot=pot,denspot=denspot)
    with pytest.raises(RuntimeError):
        denspot= potential.NFWPotential(amp=2.,a=1.3,ro=8.,vo=230.)
        dfh= eddingtondf(pot=pot,denspot=denspot)
    return None

# Test that the unit system is correctly transfered
def test_isotropic_hernquist_unittransfer():
    from galpy.util import conversion
    ro, vo= 9., 210.
    pot= potential.HernquistPotential(amp=2.,a=1.3,ro=ro,vo=vo)
    dfh= isotropicHernquistdf(pot=pot)
    phys= conversion.get_physical(dfh,include_set=True)
    assert phys['roSet'], "sphericaldf's ro not set when that of the underlying potential is set"
    assert phys['voSet'], "sphericaldf's vo not set when that of the underlying potential is set"
    assert numpy.fabs(phys['ro']-ro) < 1e-8, "Potential's unit system not correctly transfered to sphericaldf's"
    assert numpy.fabs(phys['vo']-vo) < 1e-8, "Potential's unit system not correctly transfered to sphericaldf's"
    # Following should not be on
    pot= potential.HernquistPotential(amp=2.,a=1.3)
    dfh= isotropicHernquistdf(pot=pot)
    phys= conversion.get_physical(dfh,include_set=True)
    assert not phys['roSet'], "sphericaldf's ro set when that of the underlying potential is not set"
    assert not phys['voSet'], "sphericaldf's vo set when that of the underlying potential is not set"
    return None

# Test that output orbits from sampling correctly have units on or off
def test_isotropic_hernquist_unitsofsamples():
    from galpy.util import conversion
    ro, vo= 9., 210.
    pot= potential.HernquistPotential(amp=2.,a=1.3,ro=ro,vo=vo)
    dfh= isotropicHernquistdf(pot=pot)
    samp= dfh.sample(n=100)
    assert conversion.get_physical(samp,include_set=True)['roSet'], 'Orbit samples from spherical DF with units on do not have units on'
    assert conversion.get_physical(samp,include_set=True)['voSet'], 'Orbit samples from spherical DF with units on do not have units on'
    assert numpy.fabs(conversion.get_physical(samp,include_set=True)['ro']-ro) < 1e-8, 'Orbit samples from spherical DF with units on do not have correct ro'
    assert numpy.fabs(conversion.get_physical(samp,include_set=True)['vo']-vo) < 1e-8, 'Orbit samples from spherical DF with units on do not have correct vo'
    # Also test a case where they should be off
    pot= potential.HernquistPotential(amp=2.,a=1.3)
    dfh= isotropicHernquistdf(pot=pot)
    samp= dfh.sample(n=100)
    assert not conversion.get_physical(samp,include_set=True)['roSet'], 'Orbit samples from spherical DF with units off do not have units off'
    assert not conversion.get_physical(samp,include_set=True)['voSet'], 'Orbit samples from spherical DF with units off do not have units off'
    return None

############################### HELPER FUNCTIONS ##############################
def check_spherical_symmetry(samp,l,m,tol):
    """Check for spherical symmetry by Monte Carlo integration of the
    spherical harmonic Y_lm over the sample, should be zero unless l=m=0"""
    thetas, phis= numpy.arctan2(samp.R(),samp.z()), samp.phi()
    assert numpy.fabs(numpy.sum(special.lpmv(m,l,numpy.cos(thetas))*numpy.cos(m*phis))/samp.size-(l==0)*(m==0)) < tol, 'Sample does not appear to be spherically symmetric, fails spherical harmonics test for (l,m) = ({},{})'.format(l,m)
    return None

def check_azimuthal_symmetry(samp,m,tol):
    """Check for spherical symmetry by Monte Carlo integration of the
    spherical harmonic Y_lm over the sample, should be zero unless l=m=0"""
    thetas, phis= numpy.arctan2(samp.R(),samp.z()), samp.phi()
    assert numpy.fabs(numpy.sum(numpy.cos(m*phis))/samp.size-(m==0)) < tol, 'Sample does not appear to be azimuthally symmetric, fails Fourier test for m = {}'.format(m)
    return None

def check_spherical_massprofile(samp,mass_profile,tol,skip=100):
    """Check that the cumulative distribution of radii follows the 
    cumulative mass profile (normalized such that total mass = 1)"""
    rs= samp.r()
    cumul_rs= numpy.sort(rs)
    cumul_mass= numpy.linspace(0.,1.,len(rs))
    for ii in range(len(rs)//skip-1):
        indx= (ii+1)*skip
        assert numpy.fabs(cumul_mass[indx]-mass_profile(cumul_rs[indx])) < tol, 'Mass profile of samples does not agree with analytical one'
    return None

def check_sigmar_against_jeans(samp,pot,tol,beta=0.,dens=None,
                               rmin=None,rmax=None,bins=31):
    """Check that sigma_r(r) obtained from a sampling agrees with that coming 
    from the Jeans equation
    Does this by logarithmically binning in r between rmin and rmax"""
    vrs= (samp.vR()*samp.R()+samp.vz()*samp.z())/samp.r()
    logrs= numpy.log(samp.r())
    if rmin is None: numpy.exp(numpy.amin(logrs))
    if rmax is None: numpy.exp(numpy.amax(logrs))
    w,e= numpy.histogram(logrs,range=[numpy.log(rmin),numpy.log(rmax)],
                         bins=bins,weights=numpy.ones_like(logrs))
    mv2,_= numpy.histogram(logrs,range=[numpy.log(rmin),numpy.log(rmax)],
                           bins=bins,weights=vrs**2.)
    samp_sigr= numpy.sqrt(mv2/w)
    brs= numpy.exp((numpy.roll(e,-1)+e)[:-1]/2.)
    for ii,br in enumerate(brs):
        assert numpy.fabs(samp_sigr[ii]/jeans.sigmar(pot,br,beta=beta,dens=dens
                                                     )-1.) < tol, \
                                                     "sigma_r(r) from samples does not agree with that obtained from the Jeans equation"
    return None

def check_beta(samp,pot,tol,beta=0.,
               rmin=None,rmax=None,bins=31):
    """Check that beta(r) obtained from a sampling agrees with the expected
    value
    Does this by logarithmically binning in r between rmin and rmax"""
    vrs= (samp.vR()*samp.R()+samp.vz()*samp.z())/samp.r()
    vthetas=(samp.z()*samp.vR()-samp.R()*samp.vz())/samp.r()
    vphis= samp.vT()    
    logrs= numpy.log(samp.r())
    if rmin is None: numpy.exp(numpy.amin(logrs))
    if rmax is None: numpy.exp(numpy.amax(logrs))
    w,e= numpy.histogram(logrs,range=[numpy.log(rmin),numpy.log(rmax)],
                         bins=bins,weights=numpy.ones_like(logrs))
    mvr2,_= numpy.histogram(logrs,range=[numpy.log(rmin),numpy.log(rmax)],
                            bins=bins,weights=vrs**2.)
    mvt2,_= numpy.histogram(logrs,range=[numpy.log(rmin),numpy.log(rmax)],
                            bins=bins,weights=vthetas**2.)
    mvp2,_= numpy.histogram(logrs,range=[numpy.log(rmin),numpy.log(rmax)],
                            bins=bins,weights=vphis**2.)
    samp_sigr= numpy.sqrt(mvr2/w)
    samp_sigt= numpy.sqrt(mvt2/w)
    samp_sigp= numpy.sqrt(mvp2/w)
    samp_beta= 1.-(samp_sigt**2.+samp_sigp**2.)/2./samp_sigr**2.
    brs= numpy.exp((numpy.roll(e,-1)+e)[:-1]/2.)
    if not callable(beta):
        beta_func= lambda r: beta
    else:
        beta_func= beta
    assert numpy.all(numpy.fabs(samp_beta-beta_func(brs)) < tol), "beta(r) from samples does not agree with the expected value for beta = {}".format(beta)
    return None

def check_dens_directint(dfi,pot,tol,dens,
                         rmin=None,rmax=None,bins=31):
    """Check that the density obtained from integrating over the DF agrees 
    with the expected density"""
    rs= numpy.linspace(rmin,rmax,bins)
    intdens= numpy.array([dfi.vmomentdensity(r,0,0) for r in rs])
    expdens= numpy.array([dens(r) for r in rs])
    assert numpy.all(numpy.fabs(intdens/expdens-1.) < tol), \
        "Density from direct integration is not equal to the expected value"
    return None

def check_meanvr_directint(dfi,pot,tol,beta=0.,
                           rmin=None,rmax=None,bins=31):
    """Check that the mean v_r(r) obtained from integrating over the DF agrees 
    with the expected zero"""
    rs= numpy.linspace(rmin,rmax,bins)
    intmvr= numpy.array([dfi.vmomentdensity(r,1,0)/dfi.vmomentdensity(r,0,0)
                         for r in rs])
    assert numpy.all(numpy.fabs(intmvr) < tol), \
        "mean v_r(r) from direct integration is not zero"
    return None

def check_sigmar_against_jeans_directint(dfi,pot,tol,beta=0.,
                                         dens=None,
                                         rmin=None,rmax=None,bins=31):
    """Check that sigma_r(r) obtained from integrating over the DF agrees 
    with that coming from the Jeans equation"""
    rs= numpy.linspace(rmin,rmax,bins)
    intsr= numpy.array([dfi.sigmar(r,use_physical=False) for r in rs])
    jeanssr= numpy.array([jeans.sigmar(pot,r,beta=beta,dens=dens,use_physical=False) for r in rs])
    assert numpy.all(numpy.fabs(intsr/jeanssr-1) < tol), \
                     "sigma_r(r) from direct integration does not agree with that obtained from the Jeans equation"
    return None

def check_sigmar_against_jeans_directint_forcevmoment(dfi,pot,tol,beta=0.,
                                                      rmin=None,rmax=None,
                                                      bins=31):
    """Check that sigma_r(r) obtained from integrating over the DF agrees 
    with that coming from the Jeans equation, using the general sphericaldf
    class' vmomentdensity"""
    from galpy.df.sphericaldf import sphericaldf
    rs= numpy.linspace(rmin,rmax,bins)
    intsr= numpy.array([numpy.sqrt(sphericaldf._vmomentdensity(dfi,r,2,0)/
                                   sphericaldf._vmomentdensity(dfi,r,0,0))
                        for r in rs])
    jeanssr= numpy.array([jeans.sigmar(pot,r,beta=beta,use_physical=False) for r in rs])
    assert numpy.all(numpy.fabs(intsr/jeanssr-1) < tol), \
                     "sigma_r(r) from direct integration does not agree with that obtained from the Jeans equation"
    return None

def check_beta_directint(dfi,tol,beta=0.,rmin=None,rmax=None,bins=31):
    """Check that beta(r) obtained from integrating over the DF agrees 
    with the expected behavior"""
    rs= numpy.linspace(rmin,rmax,bins)
    intbeta= numpy.array([dfi.beta(r) for r in rs])
    if not callable(beta):
        beta_func= lambda r: beta
    else:
        beta_func= beta
    assert numpy.all(numpy.fabs(intbeta-beta_func(rs)) < tol), \
        "beta(r) from direct integration does not agree with the expected value"
    return None

