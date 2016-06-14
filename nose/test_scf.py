############################TESTS ON POTENTIALS################################
from __future__ import print_function, division
import os
import sys
import numpy
import pynbody
from galpy import potential
from galpy.potential import SCFPotential
from galpy.util import bovy_coords
from galpy.orbit import Orbit
_TRAVIS= bool(os.getenv('TRAVIS'))

EPS = 1e-13 ## default epsilon

DEFAULT_R= numpy.array([0.5,1.,2.])
DEFAULT_Z= numpy.array([0.,.125,-.125,0.25,-0.25])
DEFAULT_PHI= numpy.array([0.,0.5,-0.5,1.,-1.,
                       numpy.pi,0.5+numpy.pi,
                       1.+numpy.pi])

 
## tests whether scf_compute_spherical computes the correct coefficients for a Hernquist Potential
def test_scf_compute_spherical_hernquist():
    Acos, Asin = potential.scf_compute_coeffs_spherical(sphericalHernquistDensity, 10)
    spherical_coeffsTest(Acos, Asin)
    assert numpy.fabs(Acos[0,0,0] - 1.) < EPS, "Acos(n=0,l=0,m=0) = 1 fails. Found to be Acos(n=0,l=0,m=0) = {0}".format(Acos[0,0,0])
    
    
## tests whether scf_compute_spherical computes the correct coefficients for Zeeuw's Potential
def test_scf_compute_spherical_zeeuw():
    Acos, Asin = potential.scf_compute_coeffs_spherical(rho_Zeeuw, 10)
    spherical_coeffsTest(Acos, Asin)
    assert numpy.fabs(Acos[0,0,0] - 2*3./4) < EPS, "Acos(n=0,l=0,m=0) = 3/2 fails. Found to be Acos(n=0,l=0,m=0) = {0}".format(Acos[0,0,0])
    assert numpy.fabs(Acos[1,0,0] - 2*1./12) < EPS, "Acos(n=1,l=0,m=0) = 1/6 fails. Found to be Acos(n=0,l=0,m=0) = {0}".format(Acos[0,0,0])
 
##Tests that the numerically calculated results from axi_density1 matches with the analytic results
def test_scf_compute_axi_density1():
    A = potential.scf_compute_coeffs_axi(axi_density1, 10,10)
    axi_coeffsTest(A[0], A[1])
    analytically_calculated = numpy.array([[4./3, 7.* 3**(-5/2.), 2*11*5**(-5./2), 0],
                                            [0, 0,0,0],
                                            [0,11. / (3.**(5./2) * 5 * 7. * 2), 1. / (2*3.*5**.5*7.), 0]])
    numerically_calculated = A[0][:3,:4,0]
    shape = numerically_calculated.shape
    for n in range(shape[0]):
        for l in range(shape[1]):
            assert numpy.fabs(numerically_calculated[n,l] - analytically_calculated[n,l]) < EPS, \
        "Acos(n={0},l={1},0) = {2}, whereas it was analytically calculated to be {3}".format(n,l, numerically_calculated[n,l], analytically_calculated[n,l])


##Tests that the numerically calculated results from axi_density2 matches with the analytic results
def test_scf_compute_axi_density2():
    A = potential.scf_compute_coeffs_axi(axi_density2, 10,10)
    axi_coeffsTest(A[0], A[1])
    analytically_calculated = 2*numpy.array([[1., 7.* 3**(-3/2.) /4., 3*11*5**(-5./2)/2., 0],
                                            [0,0,0,0], ##I never did analytically solve for n=1
                                            [0, 11./(7*5*3**(3./2)*2**(3.)), (7 * 5**(.5)*2**3.)**-1., 0]])
    numerically_calculated = A[0][:3,:4,0]
    shape = numerically_calculated.shape
    for n in range(shape[0]):
        if n ==1: continue 
        for l in range(shape[1]):
            assert numpy.fabs(numerically_calculated[n,l] - analytically_calculated[n,l]) < EPS, \
        "Acos(n={0},l={1},0) = {2}, whereas it was analytically calculated to be {3}".format(n,l, numerically_calculated[n,l], analytically_calculated[n,l])
 
def test_scf_compute_nfw(): 
    Acos, Asin = potential.scf_compute_coeffs_spherical(rho_NFW, 10)
    spherical_coeffsTest(Acos, Asin)

## Tests whether scf_compute_axi reduces to scf_compute_spherical for the Hernquist Potential   
def test_scf_axiHernquistCoeffs_ReducesToSpherical():
    Aspherical = potential.scf_compute_coeffs_spherical(sphericalHernquistDensity, 10)
    Aaxi = potential.scf_compute_coeffs_axi(sphericalHernquistDensity, 10,10)
    axi_reducesto_spherical(Aspherical,Aaxi, "Hernquist Potential")
    
## Tests whether scf_compute_axi reduces to scf_compute_spherical for Zeeuw's Potential   
def test_scf_axiZeeuwCoeffs_ReducesToSpherical():
    Aspherical = potential.scf_compute_coeffs_spherical(rho_Zeeuw, 10)
    Aaxi = potential.scf_compute_coeffs_axi(rho_Zeeuw, 10,10)
    axi_reducesto_spherical(Aspherical,Aaxi, "Zeeuw Potential")
    
## Tests whether scf_compute reduces to scf_compute_spherical for Hernquist Potential   
def test_scf_HernquistCoeffs_ReducesToSpherical():
    Aspherical = potential.scf_compute_coeffs_spherical(sphericalHernquistDensity, 5)
    Aaxi = potential.scf_compute_coeffs(sphericalHernquistDensity, 5,5)
    reducesto_spherical(Aspherical,Aaxi, "Hernquist Potential")
       
## Tests whether scf_compute reduces to scf_compute_spherical for Zeeuw's Potential   
def test_scf_ZeeuwCoeffs_ReducesToSpherical():
    Aspherical = potential.scf_compute_coeffs_spherical(rho_Zeeuw, 5)
    Aaxi = potential.scf_compute_coeffs(rho_Zeeuw, 5,5)
    reducesto_spherical(Aspherical,Aaxi, "Zeeuw Potential")
    
## Tests whether scf density matches with Hernquist density
def test_densMatches_hernquist():
    h = potential.HernquistPotential()
    Acos, Asin = potential.scf_compute_coeffs_spherical(sphericalHernquistDensity,10)
    scf = SCFPotential()
    assertmsg = "Comparing the density of Hernquist Potential with SCF fails at R={0}, Z={1}, phi={2}"
    compareFunctions(h.dens,scf.dens, assertmsg) 
    
## Tests whether scf density matches with Zeeuw density
def test_densMatches_zeeuw():
    Acos, Asin = potential.scf_compute_coeffs_spherical(rho_Zeeuw,10)
    scf = SCFPotential(amp=1, Acos=Acos, Asin=Asin)
    assertmsg = "Comparing the density of Zeeuw's perfect ellipsoid with SCF fails at R={0}, Z={1}, phi={2}"
    compareFunctions(rho_Zeeuw,scf.dens, assertmsg) 
    
## Tests whether scf density matches with axi_density1
def test_densMatches_axi_density1():
    Acos, Asin = potential.scf_compute_coeffs_axi(axi_density1,10,10)
    scf = SCFPotential(amp=1, Acos=Acos, Asin=Asin)
    assertmsg = "Comparing axi_density1 with SCF fails at R={0}, Z={1}, phi={2}"
    compareFunctions(axi_density1,scf.dens, assertmsg) 
    
## Tests whether scf density matches with axi_density2
def test_densMatches_axi_density2():
    Acos, Asin = potential.scf_compute_coeffs_axi(axi_density2,10,10)
    scf = SCFPotential(amp=1, Acos=Acos, Asin=Asin)
    assertmsg = "Comparing axi_density2 with SCF fails at R={0}, Z={1}, phi={2}"
    compareFunctions(axi_density2,scf.dens, assertmsg) 
    
    
## Tests whether scf density matches with NFW 
def test_densMatches_nfw():
    Acos, Asin = potential.scf_compute_coeffs_spherical(rho_NFW,10)
    scf = SCFPotential(amp=1, Acos=Acos, Asin=Asin)
    assertmsg = "Comparing nfw with SCF fails at R={0}, Z={1}, phi={2}"
    compareFunctions(rho_NFW,scf.dens, assertmsg) 

## Tests whether scf potential matches with Hernquist potential
def test_potentialMatches_hernquist():
    h = potential.HernquistPotential()
    Acos, Asin = potential.scf_compute_coeffs_spherical(sphericalHernquistDensity,10)
    scf = SCFPotential()
    assertmsg = "Comparing the potential of Hernquist Potential with SCF fails at R={0}, Z={1}, phi={2}"
    compareFunctions(h,scf, assertmsg)
  
## Tests whether scf Potential matches with NFW 
def test_densMatches_nfw():
    nfw = potential.NFWPotential()
    Acos, Asin = potential.scf_compute_coeffs_spherical(rho_NFW,10)
    scf = SCFPotential(amp=1, Acos=Acos, Asin=Asin)
    assertmsg = "Comparing nfw with SCF fails at R={0}, Z={1}, phi={2}"
    compareFunctions(nfw,scf, assertmsg) 
      
## Tests whether scf Rforce matches with Hernquist Rforce
def test_RforceMatches_hernquist():
    h = potential.HernquistPotential()
    Acos, Asin = potential.scf_compute_coeffs_spherical(sphericalHernquistDensity,10)
    scf = SCFPotential()
    assertmsg = "Comparing the radial force of Hernquist Potential with SCF fails at R={0}, Z={1}, phi={2}"
    compareFunctions(h.Rforce,scf.Rforce, assertmsg)
    
## Tests whether scf zforce matches with Hernquist Rforce
def test_zforceMatches_hernquist():
    h = potential.HernquistPotential()
    Acos, Asin = potential.scf_compute_coeffs_spherical(sphericalHernquistDensity,10)
    scf = SCFPotential()
    assertmsg = "Comparing the vertical force of Hernquist Potential with SCF fails at R={0}, Z={1}, phi={2}"
    compareFunctions(h.zforce,scf.zforce, assertmsg)
    
    
## Tests whether scf phiforce matches with Hernquist Rforce
def test_phiforceMatches_hernquist():
    h = potential.HernquistPotential()
    Acos, Asin = potential.scf_compute_coeffs_spherical(sphericalHernquistDensity,10)
    scf = SCFPotential()
    assertmsg = "Comparing the azimuth force of Hernquist Potential with SCF fails at R={0}, Z={1}, phi={2}"
    compareFunctions(h.phiforce,scf.phiforce, assertmsg)
    
    
## Tests whether scf Rforce matches with NFW Rforce
def test_RforceMatches_nfw():
    nfw = potential.NFWPotential()
    Acos, Asin = potential.scf_compute_coeffs_spherical(rho_NFW,10)
    scf = SCFPotential()
    assertmsg = "Comparing the radial force of NFW Potential with SCF fails at R={0}, Z={1}, phi={2}"
    compareFunctions(nfw.Rforce,scf.Rforce, assertmsg)
      
## Tests whether scf zforce matches with NFW Rforce
def test_zforceMatches_nfw():
    nfw = potential.NFWPotential()
    Acos, Asin = potential.scf_compute_coeffs_spherical(rho_NFW,10)
    scf = SCFPotential()
    assertmsg = "Comparing the vertical force of NFW Potential with SCF fails at R={0}, Z={1}, phi={2}"
    compareFunctions(nfw.zforce,scf.zforce, assertmsg)
    
## Tests whether scf phiforce matches with NFW Rforce
def test_phiforceMatches_nfw():
    nfw = potential.NFWPotential()
    Acos, Asin = potential.scf_compute_coeffs_spherical(rho_NFW,10)
    scf = SCFPotential()
    assertmsg = "Comparing the azimuth force of NFW Potential with SCF fails at R={0}, Z={1}, phi={2}"
    compareFunctions(nfw.phiforce,scf.phiforce, assertmsg)

## Checks Energy conservation for the SCF Hernquist Potential    
def test_scfHernquist_energyConserved():
    Acos, Asin = potential.scf_compute_coeffs_spherical(sphericalHernquistDensity,10)
    energyConservationTest(Acos, Asin, "Hernquist")
  
  
## Checks Energy conservation for the SCF Zeeuw Potential    
def test_scfZeuuw_energyConserved():
    Acos, Asin = potential.scf_compute_coeffs_spherical(rho_Zeeuw,10)
    energyConservationTest(Acos, Asin, "Zeeuw")
    
## Checks Energy conservation for the SCF axi_density1 Potential    
def test_scfaxi_density1_energyConserved():
    Acos, Asin = potential.scf_compute_coeffs_axi(axi_density1,10,10)
    energyConservationTest(Acos, Asin, "axi_density1")
    
## Checks Energy conservation for the SCF axi_density2 Potential    
def test_scfaxi_density2_energyConserved():
    Acos, Asin = potential.scf_compute_coeffs_axi(axi_density2,10,10)
    energyConservationTest(Acos, Asin, "axi_density2")
  
def test_scfNFW_energyConserved():
    Acos, Asin = potential.scf_compute_coeffs_spherical(rho_NFW,10)
    energyConservationTest(Acos, Asin, "NFW")
    
    

 
##############GENERIC FUNCTIONS BELOW###############

## This is used to compare scf functions with its corresponding galpy function
def compareFunctions(galpyFunc, scfFunc, assertmsg, Rs=DEFAULT_R, Zs = DEFAULT_Z, phis = DEFAULT_PHI, eps=EPS):
    ##Assert msg must have 3 placeholders ({}) for Rs, Zs, and phis                      
    for ii in range(len(Rs)):
        for jj in range(len(Zs)):
            for kk in range(len(phis)):
                assert numpy.fabs(galpyFunc(Rs[ii],Zs[jj],phis[kk]) - scfFunc(Rs[ii],Zs[jj],phis[kk])) < eps, \
                assertmsg.format(Rs[ii],Zs[jj],phis[kk])

##General function that tests whether coefficients for a spherical density has the expected property
def spherical_coeffsTest(Acos, Asin, eps=EPS):
    ## We expect Asin to be zero
    assert numpy.all(numpy.fabs(Asin) <eps), "Conforming Asin = 0 fails"
    ## We expect that the only non-zero values occur at (n,l=0,m=0)
    assert numpy.all(numpy.fabs(Acos[:, 1:, :]) < eps) and numpy.all(numpy.fabs(Acos[:,:,1:]) < eps), \
    "Non Zero value found outside (n,l,m) = (n,0,0)" 
    
##General function that tests whether coefficients for an axi symmetric density has the expected property
def axi_coeffsTest(Acos, Asin):
    ## We expect Asin to be zero
    assert numpy.all(numpy.fabs(Asin) <EPS), "Conforming Asin = 0 fails"
    ## We expect that the only non-zero values occur at (n,l,m=0)
    assert numpy.all(numpy.fabs(Acos[:,:,1:]) < EPS), "Non Zero value found outside (n,l,m) = (n,0,0)" 
    
## Tests whether the coefficients of a spherical density computed using the scf_compute_coeffs_axi reduces to 
## The coefficients computed using the scf_compute_coeffs_spherical 
def axi_reducesto_spherical(Aspherical,Aaxi,potentialName):
    Acos_s, Asin_s = Aspherical
    Acos_a, Asin_a = Aaxi
 
    spherical_coeffsTest(Acos_a, Asin_a, eps=1e-10) 
    n = min(Acos_s.shape[0], Acos_a.shape[0])
    assert numpy.all(numpy.fabs(Acos_s[:n,0,0] - Acos_a[:n,0,0]) < EPS), \
    "The axi symmetric Acos(n,l=0,m=0) does not reduce to the spherical Acos(n,l=0,m=0) for {0}".format(potentialName)
    
    
## Tests whether the coefficients of a spherical density computed using the scf_compute_coeffs reduces to 
## The coefficients computed using the scf_compute_coeffs_spherical 
def reducesto_spherical(Aspherical,A,potentialName):
    Acos_s, Asin_s = Aspherical
    Acos, Asin = A
 
    spherical_coeffsTest(Acos, Asin, eps=1e-10) 
    n = min(Acos_s.shape[0], Acos.shape[0])
    assert numpy.all(numpy.fabs(Acos_s[:n,0,0] - Acos[:n,0,0]) < EPS), \
    "Acos(n,l=0,m=0) as generated by scf_compute_coeffs does not reduce to the spherical Acos(n,l=0,m=0) for {0}".format(potentialName)
    
##Tests energy conservation
def energyConservationTest(Acos, Asin, name):
    scf = SCFPotential(amp=1, Acos=Acos,Asin=Asin)
    times= numpy.linspace(0.,280.,1001)
    o= Orbit(vxvv=[1.,0.1,1.1,0.,0.1])
    o.integrate(times,scf,method='odeint')
    tEs= o.E(times)
    
    assert (numpy.std(tEs)/numpy.mean(tEs))**2. < EPS, "SCF {0} Conserved Energy fails.".format(name)
    
## Hernquist potential as a function of r
def sphericalHernquistDensity(R, z=0, phi=0):
    h = potential.HernquistPotential()
    return h.dens(R,z,phi)

def rho_Zeeuw(R, z=0., phi=0., a=1.):
    r, theta, phi = bovy_coords.cyl_to_spher(R,z, phi)
    return 3./(4*numpy.pi) * numpy.power((a + r),-4.) * a
    
def rho_NFW(R, z=0, phi=0.):
    nfw = potential.NFWPotential()
    return nfw.dens(R,z, phi)
    
    
def axi_density1(R, z=0, phi=0.):
    r, theta, phi = bovy_coords.cyl_to_spher(R,z, phi)
    h = potential.HernquistPotential()
    return h.dens(R, z, phi)*(1 + numpy.cos(theta) + numpy.cos(theta)**2.)
    
    
def axi_density2(R, z=0, phi=0.):
    spherical_coords = bovy_coords.cyl_to_spher(R,z, phi)
    theta = spherical_coords[1]
    return rho_Zeeuw(R,z,phi)*(1 + numpy.cos(theta) + numpy.cos(theta)**2)