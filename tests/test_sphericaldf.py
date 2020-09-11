# Tests of spherical distribution functions
import pytest
import numpy
from scipy import special
from galpy import potential
from galpy.df import isotropicHernquistdf, constantbetaHernquistdf, kingdf
from galpy.df import jeans

############################# ISOTROPIC HERNQUIST DF ##########################
# Note that we use the Hernquist case to check a bunch of code in the
# sphericaldf realm that doesn't need to be check for each new spherical DF
def test_isotropic_hernquist_dens_spherically_symmetric():
    pot= potential.HernquistPotential(amp=2.,a=1.3)
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
    pot= potential.HernquistPotential(amp=2.,a=1.3)
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
    pot= potential.HernquistPotential(amp=2.,a=1.3)
    dfh= isotropicHernquistdf(pot=pot)
    numpy.random.seed(10)
    samp= dfh.sample(R=1.3,z=0.,n=100000)
    assert numpy.all(numpy.fabs(samp.r()-1.3) < 1e-8), 'Sampling a spherical distribution function at a single r does not produce orbits at a single r'
    return None

def test_isotropic_hernquist_singler_is_atrandomphi():
    pot= potential.HernquistPotential(amp=2.,a=1.3)
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
    pot= potential.HernquistPotential(amp=2.,a=1.3)
    dfh= isotropicHernquistdf(pot=pot)
    numpy.random.seed(10)
    samp= dfh.sample(R=1.3,z=0.,phi=numpy.pi-0.3,n=100000)
    assert numpy.all(numpy.fabs(samp.phi()-numpy.pi+0.3) < 1e-8), 'Sampling a spherical distribution function at a single r and phi oes not produce orbits at a single phi'
    return None

def test_isotropic_hernquist_givenr_are_atgivenr():
    pot= potential.HernquistPotential(amp=2.,a=1.3)
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
    pot= potential.HernquistPotential(amp=2.,a=1.3)
    # Remove the inverse cumulative mass function to force its interpolation
    class isotropicHernquistdfNoICMF(isotropicHernquistdf):
        _icmf= property()
    dfh= isotropicHernquistdfNoICMF(pot=pot)
    print(hasattr(dfh,'_icmf'))
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
    pot= potential.HernquistPotential(amp=2.,a=1.3)
    dfh= isotropicHernquistdf(pot=pot)
    numpy.random.seed(10)
    samp= dfh.sample(n=100000)
    tol= 0.05
    check_sigmar_against_jeans(samp,pot,tol,beta=0.,
                               rmin=pot._scale/10.,rmax=pot._scale*10.,bins=31)
    return None

def test_isotropic_hernquist_singler_sigmar():
    pot= potential.HernquistPotential(amp=2.,a=1.3)
    dfh= isotropicHernquistdf(pot=pot)
    numpy.random.seed(10)
    for r in [0.3,1.3,2.3]:
        samp= dfh.sample(R=r,z=0.,n=100000)
        tol= 0.01
        check_sigmar_against_jeans(samp,pot,tol,beta=0.,
                                   rmin=r-0.1,rmax=r+0.1,bins=1)
    return None

def test_isotropic_hernquist_beta():
    pot= potential.HernquistPotential(amp=2.,a=1.3)
    dfh= isotropicHernquistdf(pot=pot)
    numpy.random.seed(10)
    samp= dfh.sample(n=1000000)
    tol= 6*1e-2
    check_beta(samp,pot,tol,beta=0.,
               rmin=pot._scale/10.,rmax=pot._scale*10.,bins=31)
    return None

def test_isotropic_hernquist_meanvr_directint():
    pot= potential.HernquistPotential(amp=2.,a=1.3)
    dfh= isotropicHernquistdf(pot=pot)
    tol= 1e-8
    check_meanvr_directint(dfh,pot,tol,beta=0.,rmin=pot._scale/10.,
                           rmax=pot._scale*10.,bins=31)
    return None

def test_isotropic_hernquist_sigmar_directint():
    pot= potential.HernquistPotential(amp=2.,a=1.3)
    dfh= isotropicHernquistdf(pot=pot)
    tol= 1e-5
    check_sigmar_against_jeans_directint(dfh,pot,tol,beta=0.,
                                         rmin=pot._scale/10.,
                                         rmax=pot._scale*10.,
                                         bins=31)
    return None

def test_isotropic_hernquist_sigmar_directint_forcevmoment():
    pot= potential.HernquistPotential(amp=2.,a=1.3)
    dfh= isotropicHernquistdf(pot=pot)
    tol= 1e-5
    check_sigmar_against_jeans_directint_forcevmoment(dfh,pot,tol,beta=0.,
                                                      rmin=pot._scale/10.,
                                                      rmax=pot._scale*10.,
                                                      bins=31)
    return None

def test_isotropic_hernquist_beta_directint():
    pot= potential.HernquistPotential(amp=2.,a=1.3)
    dfh= isotropicHernquistdf(pot=pot)
    tol= 1e-8
    check_beta_directint(dfh,tol,beta=0.,
                         rmin=pot._scale/10.,
                         rmax=pot._scale*10.,
                         bins=31)
    return None

def test_isotropic_hernquist_energyoutofbounds():
    pot= potential.HernquistPotential(amp=2.,a=1.3)
    dfh= isotropicHernquistdf(pot=pot)
    assert numpy.all(numpy.fabs(dfh((numpy.arange(0.1,10.,0.1),))) < 1e-8), 'Evaluating the isotropic Hernquist DF at E > 0 does not give zero'
    assert numpy.all(numpy.fabs(dfh((pot(0,0)-1e-4,))) < 1e-8), 'Evaluating the isotropic Hernquist DF at E < -GM/a does not give zero'
    return None

# Check that samples of R,vR,.. are the same as orbit samples
def test_isotropic_hernquist_phasespacesamples_vs_orbitsamples():
    pot= potential.HernquistPotential(amp=2.,a=1.3)
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
    pot= potential.HernquistPotential(amp=2.,a=1.3)
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
    pot= potential.HernquistPotential(amp=2.,a=1.3)
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
    pot= potential.HernquistPotential(amp=2.,a=1.3)
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
    pot= potential.HernquistPotential(amp=2.,a=1.3)
    betas= [-0.7,-0.5,-0.4,0.,0.3,0.5]
    for beta in betas:
        dfh= constantbetaHernquistdf(pot=pot,beta=beta)
        numpy.random.seed(10)
        samp= dfh.sample(n=100000)
        tol= 0.05
        check_sigmar_against_jeans(samp,pot,tol,beta=beta,
                                   rmin=pot._scale/10.,rmax=pot._scale*10.,
                                   bins=31)
    return None

def test_anisotropic_hernquist_beta():
    pot= potential.HernquistPotential(amp=2.,a=1.3)
    betas= [-0.7,-0.5,-0.4,0.,0.3,0.5]
    for beta in betas:
        dfh= constantbetaHernquistdf(pot=pot,beta=beta)
        numpy.random.seed(10)
        samp= dfh.sample(n=1000000)
        tol= 8*1e-2 * (beta > -0.7) + 0.12 * (beta == -0.7)
        check_beta(samp,pot,tol,beta=beta,
                   rmin=pot._scale/10.,rmax=pot._scale*10.,bins=31)
    return None
               
def test_anisotropic_hernquist_meanvr_directint():
    pot= potential.HernquistPotential(amp=2.,a=1.3)
    betas= [-0.7,-0.5,-0.4,0.,0.3,0.5]
    for beta in betas:
        dfh= constantbetaHernquistdf(pot=pot,beta=beta)
        tol= 1e-8
        check_meanvr_directint(dfh,pot,tol,beta=beta,rmin=pot._scale/10.,
                               rmax=pot._scale*10.,bins=31)
    return None

def test_anisotropic_hernquist_sigmar_directint():
    pot= potential.HernquistPotential(amp=2.,a=1.3)
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
    pot= potential.HernquistPotential(amp=2.,a=1.3)
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
    pot= potential.HernquistPotential(amp=2.,a=1.3)
    betas= [-0.7,-0.5,-0.4,0.,0.3,0.5]
    for beta in betas:
        dfh= constantbetaHernquistdf(pot=pot,beta=beta)
        assert numpy.all(numpy.fabs(dfh((numpy.arange(0.1,10.,0.1),1.1))) < 1e-8), 'Evaluating the isotropic Hernquist DF at E > 0 does not give zero'
        assert numpy.all(numpy.fabs(dfh((pot(0,0)-1e-4,1.1))) < 1e-8), 'Evaluating the isotropic Hernquist DF at E < -GM/a does not give zero'
    return None

def test_anisotropic_hernquist_diffcalls():
    from galpy.orbit import Orbit
    pot= potential.HernquistPotential(amp=2.,a=1.3)
    betas= [-0.7,-0.5,-0.4,0.,0.3,0.5]
    for beta in betas:
        dfh= constantbetaHernquistdf(pot=pot,beta=beta)
        # R,vR... vs. E
        R,vR,vT,z,vz,phi= 1.1,0.3,0.2,0.9,-0.2,2.4
        # Calculate E directly and L from Orbit
        assert numpy.fabs(dfh(R,vR,vT,z,vz,phi)-dfh((pot(R,z)+0.5*(vR**2.+vT**2.+vz**2.),numpy.sqrt(numpy.sum(Orbit([R,vR,vT,z,vz,phi]).L()**2.))))) < 1e-8, 'Calling the isotropic Hernquist DF with R,vR,... or E[R,vR,...] does not give the same answer'
        # Also as orbit
        assert numpy.fabs(dfh(R,vR,vT,z,vz,phi)-dfh(Orbit([R,vR,vT,z,vz,phi]))) < 1e-8, 'Calling the isotropic Hernquist DF with R,vR,... or E[R,vR,...] does not give the same answer'   
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
        tol= 0.09
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
    tol= 0.12
    check_beta(samp,pot,tol,beta=0.,rmin=dfk._scale/10.,rmax=dfk.rt,
               bins=31)
    return None
               
def test_king_dens_directint():
    pot= potential.KingPotential(W0=3.,M=2.3,rt=1.76)
    dfk= kingdf(W0=3.,M=2.3,rt=1.76)
    tol= 0.02
    check_dens_directint(dfk,pot,tol,
                         lambda r: dfk.dens(r)/2.3, # need to divide by mass
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

################################ TESTS OF ERRORS###############################

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
    spherical harmonic |Y_mn|^2 over the sample, should be zero unless l=m=0"""
    thetas, phis= numpy.arctan2(samp.R(),samp.z()), samp.phi()
    assert numpy.fabs(numpy.sum(special.lpmv(m,l,numpy.cos(thetas))*numpy.cos(m*phis))/samp.size-(l==0)*(m==0)) < tol, 'Sample does not appear to be spherically symmetric, fails spherical harmonics test for (l,m) = ({},{})'.format(l,m)
    return None

def check_azimuthal_symmetry(samp,m,tol):
    """Check for spherical symmetry by Monte Carlo integration of the
    spherical harmonic |Y_mn|^2 over the sample, should be zero unless l=m=0"""
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

def check_sigmar_against_jeans(samp,pot,tol,beta=0.,
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
        assert numpy.fabs(samp_sigr[ii]/jeans.sigmar(pot,br,beta=beta,
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
                                         rmin=None,rmax=None,bins=31):
    """Check that sigma_r(r) obtained from integrating over the DF agrees 
    with that coming from the Jeans equation"""
    rs= numpy.linspace(rmin,rmax,bins)
    intsr= numpy.array([dfi.sigmar(r,use_physical=False) for r in rs])
    jeanssr= numpy.array([jeans.sigmar(pot,r,beta=beta,use_physical=False) for r in rs])
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
    intsr= numpy.array([numpy.sqrt(sphericaldf.vmomentdensity(dfi,r,2,0)/
                                   sphericaldf.vmomentdensity(dfi,r,0,0))
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

