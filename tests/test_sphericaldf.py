# Tests of spherical distribution functions
import numpy
from scipy import special
from galpy import potential
from galpy.df import isotropicHernquistdf

# Test that the density distribution of the isotropic Hernquist is correct
def test_isotropic_hernquist_dens_spherically_symmetric():
    pot = potential.HernquistPotential(amp=2.,a=1.3)
    dfh= isotropicHernquistdf(pot=pot)
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
    
def check_spherical_symmetry(samp,l,m,tol):
    """Check for spherical symmetry by Monte Carlo integration of the
    spherical harmonic |Y_mn|^2 over the sample, should be zero unless l=m=0"""
    thetas, phis= numpy.arctan2(samp.R(),samp.z()), samp.phi()
    assert numpy.fabs(numpy.sum(special.lpmv(m,l,numpy.cos(thetas))*numpy.cos(m*phis))/samp.size-(l==0)*(m==0)) < tol, 'Sample does not appear to be spherically symmetric, fails spherical harmonics test for (l,m) = ({},{})'.format(l,m)
    return None
    
