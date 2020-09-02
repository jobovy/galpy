# Tests of spherical distribution functions
import numpy
from scipy import special
from galpy import potential
from galpy.df import isotropicHernquistdf, constantbetaHernquistdf, kingdf
from galpy.df import jeans

############################# ISOTROPIC HERNQUIST DF ##########################
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
                                lambda r: pot.mass(r,use_physical=False)\
                                   /pot.mass(numpy.amax(samp.r(use_physical=False)),
                                             use_physical=False),
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

def test_isotropic_hernquist_beta():
    pot= potential.HernquistPotential(amp=2.,a=1.3)
    dfh= isotropicHernquistdf(pot=pot)
    numpy.random.seed(10)
    samp= dfh.sample(n=1000000)
    tol= 6*1e-2
    check_beta(samp,pot,tol,beta=0.,
               rmin=pot._scale/10.,rmax=pot._scale*10.,bins=31)
    return None
               
############################# ANISOTROPIC HERNQUIST DF ########################
def test_anisotropic_hernquist_dens_spherically_symmetric():
    pot= potential.HernquistPotential(amp=2.,a=1.3)
    betas= [-0.4,0.5]
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
    betas= [-0.4,0.5]
    for beta in betas:
        dfh= constantbetaHernquistdf(pot=pot,beta=beta)
        numpy.random.seed(10)
        samp= dfh.sample(n=100000)
        tol= 5*1e-3
        check_spherical_massprofile(samp,
                                    lambda r: pot.mass(r,use_physical=False)\
                                   /pot.mass(numpy.amax(samp.r(use_physical=False)),
                                             use_physical=False),
                                tol,skip=1000)
    return None

def test_anisotropic_hernquist_sigmar():
    pot= potential.HernquistPotential(amp=2.,a=1.3)
    betas= [-0.4,0.5]
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
    betas= [-0.4,0.5]
    for beta in betas:
        dfh= constantbetaHernquistdf(pot=pot,beta=beta)
        numpy.random.seed(10)
        samp= dfh.sample(n=1000000)
        tol= 8*1e-2
        check_beta(samp,pot,tol,beta=beta,
                   rmin=pot._scale/10.,rmax=pot._scale*10.,bins=31)
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
    check_spherical_massprofile(samp,
                                lambda r: pot.mass(r,use_physical=False)\
                                   /pot.mass(numpy.amax(samp.r(use_physical=False)),
                                             use_physical=False),
                                tol,skip=4000)
    return None

def test_king_sigmar():
    pot= potential.KingPotential(W0=3.,M=2.3,rt=1.76)
    dfk= kingdf(W0=3.,M=2.3,rt=1.76)
    numpy.random.seed(10)
    samp= dfk.sample(n=1000000)
    # lower tolerance closer to rt because fewer stars there
    tol= 0.07
    check_sigmar_against_jeans(samp,pot,tol,beta=0.,
                               rmin=dfk._scale/10.,rmax=dfk.rt*0.7,bins=31)
    tol= 0.2
    check_sigmar_against_jeans(samp,pot,tol,beta=0.,
                               rmin=dfk.rt*0.8,rmax=dfk.rt,bins=5)
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
               
def check_spherical_symmetry(samp,l,m,tol):
    """Check for spherical symmetry by Monte Carlo integration of the
    spherical harmonic |Y_mn|^2 over the sample, should be zero unless l=m=0"""
    thetas, phis= numpy.arctan2(samp.R(use_physical=False),samp.z(use_physical=False)), samp.phi(use_physical=False)
    assert numpy.fabs(numpy.sum(special.lpmv(m,l,numpy.cos(thetas))*numpy.cos(m*phis))/samp.size-(l==0)*(m==0)) < tol, 'Sample does not appear to be spherically symmetric, fails spherical harmonics test for (l,m) = ({},{})'.format(l,m)
    return None

def check_spherical_massprofile(samp,mass_profile,tol,skip=100):
    """Check that the cumulative distribution of radii follows the 
    cumulative mass profile (normalized such that total mass = 1)"""
    rs= samp.r(use_physical=False)
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
    vrs= (samp.vR(use_physical=False)*samp.R(use_physical=False)
          +samp.vz(use_physical=False)*samp.z(use_physical=False))\
          /samp.r(use_physical=False)
    logrs= numpy.log(samp.r(use_physical=False))
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
                                                     use_physical=False)-1.) < tol, \
                                                     "sigma_r(r) from samples does not agree with that obtained from the Jeans equation"
    return None

def check_beta(samp,pot,tol,beta=0.,
               rmin=None,rmax=None,bins=31):
    """Check that beta(r) obtained from a sampling agrees with the expected
    value
    Does this by logarithmically binning in r between rmin and rmax"""
    vrs= (samp.vR(use_physical=False)*samp.R(use_physical=False)
           +samp.vz(use_physical=False)*samp.z(use_physical=False))\
          /samp.r(use_physical=False)
    vthetas=(samp.z(use_physical=False)*samp.vR(use_physical=False)
             -samp.R(use_physical=False)*samp.vz(use_physical=False))\
             /samp.r(use_physical=False)
    vphis= samp.vT(use_physical=False)    
    logrs= numpy.log(samp.r(use_physical=False))
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
    assert numpy.all(numpy.fabs(samp_beta-beta_func(brs)) < tol), "beta(r) from samples does not agree with the expected value"
    return None
