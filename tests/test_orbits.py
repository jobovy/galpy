##########################TESTS ON MULTIPLE ORBITS#############################
import sys
import numpy
import astropy.units as u
import astropy.coordinates as apycoords
import pytest
from galpy import potential
import astropy
_APY3= astropy.__version__ > '3'

# Test that initializing an Orbit (not an Orbits) with an array of SkyCoords
# processes the input correctly into the Orbit._orb.vxvv attribute;
# The Orbits class depends on this to process arrays SkyCoords itself quickly
def test_orbit_initialization_SkyCoordarray():
    # Only run this for astropy>3
    if not _APY3: return None
    from galpy.orbit import Orbit
    numpy.random.seed(1)
    nrand= 30
    ras= numpy.random.uniform(size=nrand)*360.*u.deg
    decs= 90.*(2.*numpy.random.uniform(size=nrand)-1.)*u.deg
    dists= numpy.random.uniform(size=nrand)*10.*u.kpc
    pmras= 20.*(2.*numpy.random.uniform(size=nrand)-1.)*20.*u.mas/u.yr
    pmdecs= 20.*(2.*numpy.random.uniform(size=nrand)-1.)*20.*u.mas/u.yr
    vloss= 200.*(2.*numpy.random.uniform(size=nrand)-1.)*u.km/u.s
    # Without any custom coordinate-transformation parameters
    co= apycoords.SkyCoord(ra=ras,dec=decs,distance=dists, 
                           pm_ra_cosdec=pmras,pm_dec=pmdecs,
                           radial_velocity=vloss,
                           frame='icrs')
    os= Orbit(co)
    vxvv= numpy.array(os._orb.vxvv).T
    for ii in range(nrand):
        to= Orbit(co[ii])
        assert numpy.all(numpy.fabs(numpy.array(to._orb.vxvv)-vxvv[ii]) < 1e-10), 'Orbit initialization with an array of SkyCoords does not give the same result as processing each SkyCoord individually'
    # With custom coordinate-transformation parameters
    v_sun= apycoords.CartesianDifferential([-11.1,215.,3.25]*u.km/u.s)
    co= apycoords.SkyCoord(ra=ras,dec=decs,distance=dists, 
                           pm_ra_cosdec=pmras,pm_dec=pmdecs,
                           radial_velocity=vloss,
                           frame='icrs',
                           galcen_distance=10.*u.kpc,z_sun=1.*u.kpc,
                           galcen_v_sun=v_sun)
    os= Orbit(co)
    vxvv= numpy.array(os._orb.vxvv).T
    for ii in range(nrand):
        to= Orbit(co[ii])
        assert numpy.all(numpy.fabs(numpy.array(to._orb.vxvv)-vxvv[ii]) < 1e-10), 'Orbit initialization with an array of SkyCoords does not give the same result as processing each SkyCoord individually'
    return None

# Test Orbits initialization
def test_initialization_vxvv():
    from galpy.orbit import Orbit, Orbits
    # 1D
    vxvvs= [[1.,0.1],[0.1,3.]]
    orbits= Orbits(vxvvs)
    assert orbits.dim() == 1, 'Orbits initialization with vxvv in 1D does not work as expected'
    assert orbits.phasedim() == 2, 'Orbits initialization with vxvv in 1D does not work as expected'
    assert numpy.fabs(orbits.x()[0]-1.) < 1e-10, 'Orbits initialization with vxvv in 1D does not work as expected'
    assert numpy.fabs(orbits.x()[1]-0.1) < 1e-10, 'Orbits initialization with vxvv in 1D does not work as expected'
    assert numpy.fabs(orbits.vx()[0]-0.1) < 1e-10, 'Orbits initialization with vxvv in 1D does not work as expected'
    assert numpy.fabs(orbits.vx()[1]-3.) < 1e-10, 'Orbits initialization with vxvv in 1D does not work as expected'
    # 2D, 3 phase-D
    vxvvs= [[1.,0.1,1.],[0.1,3.,1.1]]
    orbits= Orbits(vxvvs)
    assert orbits.dim() == 2, 'Orbits initialization with vxvv in 2D, 3 phase-D does not work as expected'
    assert orbits.phasedim() == 3, 'Orbits initialization with vxvv in 2D, 3 phase-D does not work as expected'
    assert numpy.fabs(orbits.R()[0]-1.) < 1e-10, 'Orbits initialization with vxvv in 2D, 3 phase-D does not work as expected'
    assert numpy.fabs(orbits.R()[1]-0.1) < 1e-10, 'Orbits initialization with vxvv in 2D, 3 phase-D does not work as expected'
    assert numpy.fabs(orbits.vR()[0]-0.1) < 1e-10, 'Orbits initialization with vxvv in 2D, 3 phase-D does not work as expected'
    assert numpy.fabs(orbits.vR()[1]-3.) < 1e-10, 'Orbits initialization with vxvv in 2D, 3 phase-D does not work as expected'
    assert numpy.fabs(orbits.vT()[0]-1.) < 1e-10, 'Orbits initialization with vxvv in 2D, 3 phase-D does not work as expected'
    assert numpy.fabs(orbits.vT()[1]-1.1) < 1e-10, 'Orbits initialization with vxvv in 2D, 3 phase-D does not work as expected'
    # 2D, 4 phase-D
    vxvvs= [[1.,0.1,1.,1.5],[0.1,3.,1.1,2.]]
    orbits= Orbits(vxvvs)
    assert orbits.dim() == 2, 'Orbits initialization with vxvv 2D, 4 phase-D does not work as expected'
    assert orbits.phasedim() == 4, 'Orbits initialization with vxvv 2D, 4 phase-D does not work as expected'
    assert numpy.fabs(orbits.R()[0]-1.) < 1e-10, 'Orbits initialization with vxvv 2D, 4 phase-D does not work as expected'
    assert numpy.fabs(orbits.R()[1]-0.1) < 1e-10, 'Orbits initialization with vxvv 2D, 4 phase-D does not work as expected'
    assert numpy.fabs(orbits.vR()[0]-0.1) < 1e-10, 'Orbits initialization with vxvv 2D, 4 phase-D does not work as expected'
    assert numpy.fabs(orbits.vR()[1]-3.) < 1e-10, 'Orbits initialization with vxvv 2D, 4 phase-D does not work as expected'
    assert numpy.fabs(orbits.vT()[0]-1.) < 1e-10, 'Orbits initialization with vxvv 2D, 4 phase-D does not work as expected'
    assert numpy.fabs(orbits.vT()[1]-1.1) < 1e-10, 'Orbits initialization with vxvv 2D, 4 phase-D does not work as expected'
    assert numpy.fabs(orbits.phi()[0]-1.5) < 1e-10, 'Orbits initialization with vxvv 2D, 4 phase-D does not work as expected'
    assert numpy.fabs(orbits.phi()[1]-2.) < 1e-10, 'Orbits initialization with vxvv 2D, 4 phase-D does not work as expected'
    # 3D, 5 phase-D
    vxvvs= [[1.,0.1,1.,0.1,-0.2],[0.1,3.,1.1,-0.3,0.4]]
    orbits= Orbits(vxvvs)
    assert orbits.dim() == 3, 'Orbits initialization with vxvv 3D, 5 phase-D does not work as expected'
    assert orbits.phasedim() == 5, 'Orbits initialization with vxvv 3D, 5 phase-D does not work as expected'
    assert numpy.fabs(orbits.R()[0]-1.) < 1e-10, 'Orbits initialization with vxvv 3D, 5 phase-D does not work as expected'
    assert numpy.fabs(orbits.R()[1]-0.1) < 1e-10, 'Orbits initialization with vxvv 3D, 5 phase-D does not work as expected'
    assert numpy.fabs(orbits.vR()[0]-0.1) < 1e-10, 'Orbits initialization with vxvv 3D, 5 phase-D does not work as expected'
    assert numpy.fabs(orbits.vR()[1]-3.) < 1e-10, 'Orbits initialization with vxvv 3D, 5 phase-D does not work as expected'
    assert numpy.fabs(orbits.vT()[0]-1.) < 1e-10, 'Orbits initialization with vxvv 3D, 5 phase-D does not work as expected'
    assert numpy.fabs(orbits.vT()[1]-1.1) < 1e-10, 'Orbits initialization with vxvv 3D, 5 phase-D does not work as expected'
    assert numpy.fabs(orbits.z()[0]-0.1) < 1e-10, 'Orbits initialization with vxvv 3D, 5 phase-D does not work as expected'
    assert numpy.fabs(orbits.z()[1]+0.3) < 1e-10, 'Orbits initialization with vxvv 3D, 5 phase-D does not work as expected'
    assert numpy.fabs(orbits.vz()[0]+0.2) < 1e-10, 'Orbits initialization with vxvv 3D, 5 phase-D does not work as expected'
    assert numpy.fabs(orbits.vz()[1]-0.4) < 1e-10, 'Orbits initialization with vxvv 3D, 5 phase-D does not work as expected'
    # 3D, 6 phase-D
    vxvvs= [[1.,0.1,1.,0.1,-0.2,1.5],[0.1,3.,1.1,-0.3,0.4,2.]]
    orbits= Orbits(vxvvs)
    assert orbits.dim() == 3, 'Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected'
    assert orbits.phasedim() == 6, 'Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected'
    assert numpy.fabs(orbits.R()[0]-1.) < 1e-10, 'Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected'
    assert numpy.fabs(orbits.R()[1]-0.1) < 1e-10, 'Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected'
    assert numpy.fabs(orbits.vR()[0]-0.1) < 1e-10, 'Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected'
    assert numpy.fabs(orbits.vR()[1]-3.) < 1e-10, 'Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected'
    assert numpy.fabs(orbits.vT()[0]-1.) < 1e-10, 'Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected'
    assert numpy.fabs(orbits.vT()[1]-1.1) < 1e-10, 'Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected'
    assert numpy.fabs(orbits.z()[0]-0.1) < 1e-10, 'Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected'
    assert numpy.fabs(orbits.z()[1]+0.3) < 1e-10, 'Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected'
    assert numpy.fabs(orbits.vz()[0]+0.2) < 1e-10, 'Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected'
    assert numpy.fabs(orbits.vz()[1]-0.4) < 1e-10, 'Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected'
    assert numpy.fabs(orbits.phi()[0]-1.5) < 1e-10, 'Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected'
    assert numpy.fabs(orbits.phi()[1]-2.) < 1e-10, 'Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected'
    return None

def test_initialization_SkyCoord():
    # Only run this for astropy>3
    if not _APY3: return None
    from galpy.orbit import Orbit, Orbits
    numpy.random.seed(1)
    nrand= 30
    ras= numpy.random.uniform(size=nrand)*360.*u.deg
    decs= 90.*(2.*numpy.random.uniform(size=nrand)-1.)*u.deg
    dists= numpy.random.uniform(size=nrand)*10.*u.kpc
    pmras= 20.*(2.*numpy.random.uniform(size=nrand)-1.)*20.*u.mas/u.yr
    pmdecs= 20.*(2.*numpy.random.uniform(size=nrand)-1.)*20.*u.mas/u.yr
    vloss= 200.*(2.*numpy.random.uniform(size=nrand)-1.)*u.km/u.s
    # Without any custom coordinate-transformation parameters
    co= apycoords.SkyCoord(ra=ras,dec=decs,distance=dists, 
                           pm_ra_cosdec=pmras,pm_dec=pmdecs,
                           radial_velocity=vloss,
                           frame='icrs')
    orbits= Orbits(co)
    assert orbits.dim() == 3, 'Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected'
    assert orbits.phasedim() == 6, 'Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected'
    for ii in range(nrand):
        to= Orbit(co[ii])
        assert numpy.fabs(orbits.R()[ii]-to.R()) < 1e-10, 'Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected'
        assert numpy.fabs(orbits.vR()[ii]-to.vR()) < 1e-10, 'Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected'
        assert numpy.fabs(orbits.vT()[ii]-to.vT()) < 1e-10, 'Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected'
        assert numpy.fabs(orbits.z()[ii]-to.z()) < 1e-10, 'Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected'
        assert numpy.fabs(orbits.vz()[ii]-to.vz()) < 1e-10, 'Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected'
        assert numpy.fabs(orbits.phi()[ii]-to.phi()) < 1e-10, 'Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected'
    # With custom coordinate-transformation parameters
    v_sun= apycoords.CartesianDifferential([-11.1,215.,3.25]*u.km/u.s)
    co= apycoords.SkyCoord(ra=ras,dec=decs,distance=dists, 
                           pm_ra_cosdec=pmras,pm_dec=pmdecs,
                           radial_velocity=vloss,
                           frame='icrs',
                           galcen_distance=10.*u.kpc,z_sun=1.*u.kpc,
                           galcen_v_sun=v_sun)
    orbits= Orbits(co)
    assert orbits.dim() == 3, 'Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected'
    assert orbits.phasedim() == 6, 'Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected'
    for ii in range(nrand):
        to= Orbit(co[ii])
        assert numpy.fabs(orbits.R()[ii]-to.R()) < 1e-10, 'Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected'
        assert numpy.fabs(orbits.vR()[ii]-to.vR()) < 1e-10, 'Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected'
        assert numpy.fabs(orbits.vT()[ii]-to.vT()) < 1e-10, 'Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected'
        assert numpy.fabs(orbits.z()[ii]-to.z()) < 1e-10, 'Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected'
        assert numpy.fabs(orbits.vz()[ii]-to.vz()) < 1e-10, 'Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected'
        assert numpy.fabs(orbits.phi()[ii]-to.phi()) < 1e-10, 'Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected'
    return None

# Test that attempting to initialize Orbits with radec, lb, or uvw gives 
# an error
def test_initialization_radecetc_error():
    from galpy.orbit import Orbits
    with pytest.raises(NotImplementedError) as excinfo:
        Orbits([[0.,0.,0.,0.,0.,0.,]],radec=True)
    with pytest.raises(NotImplementedError) as excinfo:
        Orbits([[0.,0.,0.,0.,0.,0.,]],lb=True)
    with pytest.raises(NotImplementedError) as excinfo:
        Orbits([[0.,0.,0.,0.,0.,0.,]],radec=True,uvw=True)
    return None

# Tests that integrating Orbits agrees with integrating multiple Orbit 
# instances
def test_integration_1d():
    from galpy.orbit import Orbit, Orbits
    times= numpy.linspace(0.,10.,1001)
    orbits_list= [Orbit([1.,0.1]),Orbit([0.1,1.]),Orbit([-0.2,0.3])]
    orbits= Orbits(orbits_list)
    # Integrate as Orbits, twice to make sure initial cond. isn't changed
    orbits.integrate(times,
                     potential.toVerticalPotential(potential.MWPotential2014,1.))
    orbits.integrate(times,
                     potential.toVerticalPotential(potential.MWPotential2014,1.))
    # Integrate as multiple Orbits
    for o in orbits_list:
        o.integrate(times,
                    potential.toVerticalPotential(potential.MWPotential2014,1.))
    # Compare
    for ii in range(len(orbits)):
        assert numpy.amax(numpy.fabs(orbits_list[ii].x(times)-orbits.x(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_list[ii].vx(times)-orbits.vx(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
    return None
    
def test_integration_2d():
    from galpy.orbit import Orbit, Orbits
    times= numpy.linspace(0.,10.,1001)
    orbits_list= [Orbit([1.,0.1,1.,0.]),Orbit([.9,0.3,1.,-0.3]),
                  Orbit([1.2,-0.3,0.7,5.])]
    orbits= Orbits(orbits_list)
    # Integrate as Orbits, twice to make sure initial cond. isn't changed
    orbits.integrate(times,potential.MWPotential)
    orbits.integrate(times,potential.MWPotential)
    # Integrate as multiple Orbits
    for o in orbits_list:
        o.integrate(times,potential.MWPotential)
    # Compare
    for ii in range(len(orbits)):
        assert numpy.amax(numpy.fabs(orbits_list[ii].x(times)-orbits.x(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_list[ii].vx(times)-orbits.vx(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_list[ii].y(times)-orbits.y(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_list[ii].vy(times)-orbits.vy(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_list[ii].R(times)-orbits.R(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_list[ii].vR(times)-orbits.vR(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_list[ii].vT(times)-orbits.vT(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(((orbits_list[ii].phi(times)-orbits.phi(times)[ii]+numpy.pi) % (2.*numpy.pi)) - numpy.pi)) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
    return None
    
def test_integration_p3d():
    # 3D phase-space integration
    from galpy.orbit import Orbit, Orbits
    times= numpy.linspace(0.,10.,1001)
    orbits_list= [Orbit([1.,0.1,1.]),Orbit([.9,0.3,1.]),
                  Orbit([1.2,-0.3,0.7])]
    orbits= Orbits(orbits_list)
    # Integrate as Orbits, twice to make sure initial cond. isn't changed
    orbits.integrate(times,potential.MWPotential2014)
    orbits.integrate(times,potential.MWPotential2014)
    # Integrate as multiple Orbits
    for o in orbits_list:
        o.integrate(times,potential.MWPotential2014)
    # Compare
    for ii in range(len(orbits)):
        assert numpy.amax(numpy.fabs(orbits_list[ii].R(times)-orbits.R(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_list[ii].vR(times)-orbits.vR(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_list[ii].vT(times)-orbits.vT(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
    return None
    
def test_integration_3d():
    from galpy.orbit import Orbit, Orbits
    times= numpy.linspace(0.,10.,1001)
    orbits_list= [Orbit([1.,0.1,1.,0.,0.1,0.]),Orbit([.9,0.3,1.,-0.3,0.4,3.]),
                  Orbit([1.2,-0.3,0.7,.5,-0.5,6.])]
    orbits= Orbits(orbits_list)
    # Integrate as Orbits, twice to make sure initial cond. isn't changed
    orbits.integrate(times,potential.MWPotential2014)
    orbits.integrate(times,potential.MWPotential2014)
    # Integrate as multiple Orbits
    for o in orbits_list:
        o.integrate(times,potential.MWPotential2014)
    # Compare
    for ii in range(len(orbits)):
        assert numpy.amax(numpy.fabs(orbits_list[ii].x(times)-orbits.x(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_list[ii].vx(times)-orbits.vx(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_list[ii].y(times)-orbits.y(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_list[ii].vy(times)-orbits.vy(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_list[ii].z(times)-orbits.z(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_list[ii].vz(times)-orbits.vz(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_list[ii].R(times)-orbits.R(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_list[ii].vR(times)-orbits.vR(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_list[ii].vT(times)-orbits.vT(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs((((orbits_list[ii].phi(times)-orbits.phi(times)[ii])+numpy.pi) % (2.*numpy.pi)) - numpy.pi)) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
    return None
    
def test_integration_p5d():
    # 5D phase-space integration
    from galpy.orbit import Orbit, Orbits
    times= numpy.linspace(0.,10.,1001)
    orbits_list= [Orbit([1.,0.1,1.,0.,0.1]),Orbit([.9,0.3,1.,-0.3,0.4]),
                  Orbit([1.2,-0.3,0.7,.5,-0.5])]
    orbits= Orbits(orbits_list)
    # Integrate as Orbits, twice to make sure initial cond. isn't changed
    orbits.integrate(times,potential.MWPotential2014)
    orbits.integrate(times,potential.MWPotential2014)
    # Integrate as multiple Orbits
    for o in orbits_list:
        o.integrate(times,potential.MWPotential2014)
    # Compare
    for ii in range(len(orbits)):
        assert numpy.amax(numpy.fabs(orbits_list[ii].z(times)-orbits.z(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_list[ii].vz(times)-orbits.vz(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_list[ii].R(times)-orbits.R(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_list[ii].vR(times)-orbits.vR(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_list[ii].vT(times)-orbits.vT(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
    return None
    
# Tests that integrating Orbits agrees with integrating multiple Orbit 
# instances when using parallel_map Python paralleliization
def test_integration_forcemap_1d():
    from galpy.orbit import Orbit, Orbits
    times= numpy.linspace(0.,10.,1001)
    orbits_list= [Orbit([1.,0.1]),Orbit([0.1,1.]),Orbit([-0.2,0.3])]
    orbits= Orbits(orbits_list)
    # Integrate as Orbits
    orbits.integrate(times,
                     potential.toVerticalPotential(potential.MWPotential2014,1.),
                     force_map=True)
    # Integrate as multiple Orbits
    for o in orbits_list:
        o.integrate(times,
                    potential.toVerticalPotential(potential.MWPotential2014,1.))
    # Compare
    for ii in range(len(orbits)):
        assert numpy.amax(numpy.fabs(orbits_list[ii].x(times)-orbits.x(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_list[ii].vx(times)-orbits.vx(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
    return None
    
def test_integration_forcemap_2d():
    from galpy.orbit import Orbit, Orbits
    times= numpy.linspace(0.,10.,1001)
    orbits_list= [Orbit([1.,0.1,1.,0.]),Orbit([.9,0.3,1.,-0.3]),
                  Orbit([1.2,-0.3,0.7,5.])]
    orbits= Orbits(orbits_list)
    # Integrate as Orbits
    orbits.integrate(times,potential.MWPotential2014,force_map=True)
    # Integrate as multiple Orbits
    for o in orbits_list:
        o.integrate(times,potential.MWPotential2014)
    # Compare
    for ii in range(len(orbits)):
        assert numpy.amax(numpy.fabs(orbits_list[ii].x(times)-orbits.x(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_list[ii].vx(times)-orbits.vx(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_list[ii].y(times)-orbits.y(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_list[ii].vy(times)-orbits.vy(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_list[ii].R(times)-orbits.R(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_list[ii].vR(times)-orbits.vR(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_list[ii].vT(times)-orbits.vT(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs((((orbits_list[ii].phi(times)-orbits.phi(times)[ii])+numpy.pi) % (2.*numpy.pi)) - numpy.pi)) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
    return None
    
def test_integration_forcemap_3d():
    from galpy.orbit import Orbit, Orbits
    times= numpy.linspace(0.,10.,1001)
    orbits_list= [Orbit([1.,0.1,1.,0.,0.1,0.]),Orbit([.9,0.3,1.,-0.3,0.4,3.]),
                  Orbit([1.2,-0.3,0.7,.5,-0.5,6.])]
    orbits= Orbits(orbits_list)
    # Integrate as Orbits
    orbits.integrate(times,potential.MWPotential2014,force_map=True)
    # Integrate as multiple Orbits
    for o in orbits_list:
        o.integrate(times,potential.MWPotential2014)
    # Compare
    for ii in range(len(orbits)):
        assert numpy.amax(numpy.fabs(orbits_list[ii].x(times)-orbits.x(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_list[ii].vx(times)-orbits.vx(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_list[ii].y(times)-orbits.y(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_list[ii].vy(times)-orbits.vy(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_list[ii].z(times)-orbits.z(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_list[ii].vz(times)-orbits.vz(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_list[ii].R(times)-orbits.R(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_list[ii].vR(times)-orbits.vR(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_list[ii].vT(times)-orbits.vT(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs((((orbits_list[ii].phi(times)-orbits.phi(times)[ii])+numpy.pi) % (2.*numpy.pi)) - numpy.pi)) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
    return None

# Test slicing of orbits
def test_slice_singleobject():
    from galpy.orbit import Orbit, Orbits
    times= numpy.linspace(0.,10.,1001)
    orbits_list= [Orbit([1.,0.1,1.,0.,0.1,0.]),Orbit([.9,0.3,1.,-0.3,0.4,3.]),
                  Orbit([1.2,-0.3,0.7,.5,-0.5,6.])]
    orbits= Orbits(orbits_list)
    orbits.integrate(times,potential.MWPotential2014)
    indices= [0,1,-1]
    for ii in indices:
        assert numpy.amax(numpy.fabs(orbits[ii].x(times)-orbits.x(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits[ii].vx(times)-orbits.vx(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits[ii].y(times)-orbits.y(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits[ii].vy(times)-orbits.vy(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits[ii].z(times)-orbits.z(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits[ii].vz(times)-orbits.vz(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits[ii].R(times)-orbits.R(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits[ii].vR(times)-orbits.vR(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits[ii].vT(times)-orbits.vT(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs((((orbits[ii].phi(times)-orbits.phi(times)[ii])+numpy.pi) % (2.*numpy.pi)) - numpy.pi)) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
    return None
    
# Test slicing of orbits
def test_slice_multipleobjects():
    from galpy.orbit import Orbit, Orbits
    times= numpy.linspace(0.,10.,1001)
    orbits_list= [Orbit([1.,0.1,1.,0.,0.1,0.]),
                  Orbit([.9,0.3,1.,-0.3,0.4,3.]),
                  Orbit([1.2,-0.3,0.7,.5,-0.5,6.]),
                  Orbit([0.6,-0.4,0.4,.25,-0.5,6.]),
                  Orbit([1.1,-0.13,0.17,.35,-0.5,2.])]
    orbits= Orbits(orbits_list)
    # Pre-integration
    orbits_slice= orbits[1:4]
    for ii in range(3):
        assert numpy.amax(numpy.fabs(orbits_slice.x()[ii]-orbits.x()[ii+1])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_slice.vx()[ii]-orbits.vx()[ii+1])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_slice.y()[ii]-orbits.y()[ii+1])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_slice.vy()[ii]-orbits.vy()[ii+1])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_slice.z()[ii]-orbits.z()[ii+1])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_slice.vz()[ii]-orbits.vz()[ii+1])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_slice.R()[ii]-orbits.R()[ii+1])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_slice.vR()[ii]-orbits.vR()[ii+1])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_slice.vT()[ii]-orbits.vT()[ii+1])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_slice.phi()[ii]-orbits.phi()[ii+1])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
    # After integration
    orbits.integrate(times,potential.MWPotential2014)
    orbits_slice= orbits[1:4]
    for ii in range(3):
        assert numpy.amax(numpy.fabs(orbits_slice.x(times)[ii]-orbits.x(times)[ii+1])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_slice.vx(times)[ii]-orbits.vx(times)[ii+1])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_slice.y(times)[ii]-orbits.y(times)[ii+1])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_slice.vy(times)[ii]-orbits.vy(times)[ii+1])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_slice.z(times)[ii]-orbits.z(times)[ii+1])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_slice.vz(times)[ii]-orbits.vz(times)[ii+1])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_slice.R(times)[ii]-orbits.R(times)[ii+1])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_slice.vR(times)[ii]-orbits.vR(times)[ii+1])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_slice.vT(times)[ii]-orbits.vT(times)[ii+1])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_slice.phi(times)[ii]-orbits.phi(times)[ii+1])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
    return None

def test_slice_integratedorbit_wrapperpot_367():
    # Test related to issue 367: slicing orbits with a potential that includes 
    # a wrapper potential (from Ted Mackereth)
    from galpy.orbit import Orbit, Orbits
    from galpy.potential import DehnenSmoothWrapperPotential, \
        DehnenBarPotential, LogarithmicHaloPotential
    #initialise a wrapper potential
    tform= -10.
    tsteady= 5. 
    omega= 1.85 
    angle=25./180.*numpy.pi 
    dp= DehnenBarPotential(omegab=omega,rb=3.5/8.,Af=(1./75.),
                           tform=tform,tsteady=tsteady,barphi=angle)
    lhp=LogarithmicHaloPotential(normalize=1.)
    dswp= DehnenSmoothWrapperPotential(pot=dp,tform=-4.*2.*numpy.pi/dp.OmegaP(),
                                       tsteady=2.*2.*numpy.pi/dp.OmegaP())
    pot= [lhp,dswp]
    #initialise 2 random orbits
    r = numpy.random.randn(2)*0.01+1.
    z = numpy.random.randn(2)*0.01+0.2
    phi = numpy.random.randn(2)*0.01+0.
    vR = numpy.random.randn(2)*0.01+0.
    vT = numpy.random.randn(2)*0.01+1.
    vz = numpy.random.randn(2)*0.01+0.02
    vxvv = numpy.dstack([r,vR,vT,z,vz,phi])[0]
    os = Orbits(vxvv)
    times = numpy.linspace(0.,100.,3000)
    os.integrate(times,pot)
    # This failed in #367
    assert not os[0] is None, 'Slicing an integrated Orbits instance with a WrapperPotential does not work'
    return None
 
# Test that initializing Orbits with orbits with different phase-space
# dimensions raises an error
def test_initialize_diffphasedim_error():
    from galpy.orbit import Orbits
    # 2D with 3D
    with pytest.raises(RuntimeError) as excinfo:
        Orbits([[1.,0.1],[1.,0.1,1.]])
    # 2D with 4D
    with pytest.raises(RuntimeError) as excinfo:
        Orbits([[1.,0.1],[1.,0.1,1.,0.1]])
    # 2D with 5D
    with pytest.raises(RuntimeError) as excinfo:
        Orbits([[1.,0.1],[1.,0.1,1.,0.1,0.2]])
    # 2D with 6D
    with pytest.raises(RuntimeError) as excinfo:
        Orbits([[1.,0.1],[1.,0.1,1.,0.1,0.2,3.]])
    # 3D with 4D
    with pytest.raises(RuntimeError) as excinfo:
        Orbits([[1.,0.1,1.],[1.,0.1,1.,0.1]])
    # 3D with 5D
    with pytest.raises(RuntimeError) as excinfo:
        Orbits([[1.,0.1,1.],[1.,0.1,1.,0.1,0.2]])
    # 3D with 6D
    with pytest.raises(RuntimeError) as excinfo:
        Orbits([[1.,0.1,1.],[1.,0.1,1.,0.1,0.2,6.]])
    # 4D with 5D
    with pytest.raises(RuntimeError) as excinfo:
        Orbits([[1.,0.1,1.,2.],[1.,0.1,1.,0.1,0.2]])
    # 4D with 6D
    with pytest.raises(RuntimeError) as excinfo:
        Orbits([[1.,0.1,1.,2.],[1.,0.1,1.,0.1,0.2,6.]])
    # 5D with 6D
    with pytest.raises(RuntimeError) as excinfo:
        Orbits([[1.,0.1,1.,0.2,-0.2],[1.,0.1,1.,0.1,0.2,6.]])
    return None

def test_orbits_consistentro():
    from galpy.orbit import Orbit, Orbits
    ro= 7.
    # Initialize Orbits from list of Orbit instances
    orbits_list= [Orbit([1.,0.1,1.,0.1,0.2,-3.],ro=ro),
                  Orbit([1.,0.1,1.,0.1,0.2,-4.],ro=ro)]
    orbits= Orbits(orbits_list)
    # Check that ro is taken correctly
    assert numpy.fabs(orbits._ro-orbits_list[0]._ro) < 1e-10, "Orbits' ro not correctly taken from input list of Orbit instances"
    assert orbits._roSet, "Orbits' ro not correctly taken from input list of Orbit instances"
    # Check that consistency of ros is enforced
    with pytest.raises(RuntimeError) as excinfo:
        orbits= Orbits(orbits_list,ro=6.)
    orbits_list= [Orbit([1.,0.1,1.,0.1,0.2,-3.],ro=ro),
                  Orbit([1.,0.1,1.,0.1,0.2,-4.],ro=ro*1.2)]
    with pytest.raises(RuntimeError) as excinfo:
        orbits= Orbits(orbits_list,ro=ro)
    return None

def test_orbits_consistentvo():
    from galpy.orbit import Orbit, Orbits
    vo= 230.
    # Initialize Orbits from list of Orbit instances
    orbits_list= [Orbit([1.,0.1,1.,0.1,0.2,-3.],vo=vo),
                  Orbit([1.,0.1,1.,0.1,0.2,-4.],vo=vo)]
    orbits= Orbits(orbits_list)
    # Check that vo is taken correctly
    assert numpy.fabs(orbits._vo-orbits_list[0]._vo) < 1e-10, "Orbits' vo not correctly taken from input list of Orbit instances"
    assert orbits._voSet, "Orbits' vo not correctly taken from input list of Orbit instances"
    # Check that consistency of vos is enforced
    with pytest.raises(RuntimeError) as excinfo:
        orbits= Orbits(orbits_list,vo=210.)
    orbits_list= [Orbit([1.,0.1,1.,0.1,0.2,-3.],vo=vo),
                  Orbit([1.,0.1,1.,0.1,0.2,-4.],vo=vo*1.2)]
    with pytest.raises(RuntimeError) as excinfo:
        orbits= Orbits(orbits_list,vo=vo)
    return None

def test_orbits_consistentzo():
    from galpy.orbit import Orbit, Orbits
    zo= 0.015
    # Initialize Orbits from list of Orbit instances
    orbits_list= [Orbit([1.,0.1,1.,0.1,0.2,-3.],zo=zo),
                  Orbit([1.,0.1,1.,0.1,0.2,-4.],zo=zo)]
    orbits= Orbits(orbits_list)
    # Check that zo is taken correctly
    assert numpy.fabs(orbits._zo-orbits_list[0]._orb._zo) < 1e-10, "Orbits' zo not correctly taken from input list of Orbit instances"
    # Check that consistency of zos is enforced
    with pytest.raises(RuntimeError) as excinfo:
        orbits= Orbits(orbits_list,zo=0.045)
    orbits_list= [Orbit([1.,0.1,1.,0.1,0.2,-3.],zo=zo),
                  Orbit([1.,0.1,1.,0.1,0.2,-4.],zo=zo*1.2)]
    with pytest.raises(RuntimeError) as excinfo:
        orbits= Orbits(orbits_list,zo=zo)
    return None

def test_orbits_consistentsolarmotion():
    from galpy.orbit import Orbit, Orbits
    solarmotion= numpy.array([-10.,20.,30.])
    # Initialize Orbits from list of Orbit instances
    orbits_list= [Orbit([1.,0.1,1.,0.1,0.2,-3.],solarmotion=solarmotion),
                  Orbit([1.,0.1,1.,0.1,0.2,-4.],solarmotion=solarmotion)]
    orbits= Orbits(orbits_list)
    # Check that solarmotion is taken correctly
    assert numpy.all(numpy.fabs(orbits._solarmotion-orbits_list[0]._orb._solarmotion) < 1e-10), "Orbits' solarmotion not correctly taken from input list of Orbit instances"
    # Check that consistency of solarmotions is enforced
    with pytest.raises(RuntimeError) as excinfo:
        orbits= Orbits(orbits_list,solarmotion=numpy.array([15.,20.,30]))
    with pytest.raises(RuntimeError) as excinfo:
        orbits= Orbits(orbits_list,solarmotion=numpy.array([-10.,25.,30]))
    with pytest.raises(RuntimeError) as excinfo:
        orbits= Orbits(orbits_list,solarmotion=numpy.array([-10.,20.,-30]))
    orbits_list= [Orbit([1.,0.1,1.,0.1,0.2,-3.],solarmotion=solarmotion),
                  Orbit([1.,0.1,1.,0.1,0.2,-4.],solarmotion=solarmotion*1.2)]
    with pytest.raises(RuntimeError) as excinfo:
        orbits= Orbits(orbits_list,solarmotion=solarmotion)
    return None

def test_orbits_stringsolarmotion():
    from galpy.orbit import Orbit, Orbits
    solarmotion= 'hogg'
    orbits_list= [Orbit([1.,0.1,1.,0.1,0.2,-3.],solarmotion=solarmotion),
                  Orbit([1.,0.1,1.,0.1,0.2,-4.],solarmotion=solarmotion)]
    orbits= Orbits(orbits_list,solarmotion='hogg')
    assert numpy.all(numpy.fabs(orbits._solarmotion-numpy.array([-10.1,4.0,6.7])) < 1e-10), 'String solarmotion not parsed correcty'
    return None
                     
def test_orbits_dim_2dPot_3dOrb():
    # Test that orbit integration throws an error when using a potential that
    # is lower dimensional than the orbit (using ~Plevne's example)
    from galpy.util import bovy_conversion
    from galpy.orbit import Orbit, Orbits
    b_p= potential.PowerSphericalPotentialwCutoff(\
        alpha=1.8,rc=1.9/8.,normalize=0.05)
    ell_p= potential.EllipticalDiskPotential()
    pota=[b_p,ell_p]
    o= Orbits([Orbit(vxvv=[20.,10.,2.,3.2,3.4,-100.],
                     radec=True,ro=8.0,vo=220.0),
               Orbit(vxvv=[20.,10.,2.,3.2,3.4,-100.],
                     radec=True,ro=8.0,vo=220.0)])
    ts= numpy.linspace(0.,3.5/bovy_conversion.time_in_Gyr(vo=220.0,ro=8.0),
                       1000,endpoint=True)
    with pytest.raises(AssertionError) as excinfo:
        o.integrate(ts,pota,method="odeint")
    return None

def test_orbit_dim_1dPot_3dOrb():
    # Test that orbit integration throws an error when using a potential that
    # is lower dimensional than the orbit, for a 1D potential
    from galpy.util import bovy_conversion
    from galpy.orbit import Orbit, Orbits
    b_p= potential.PowerSphericalPotentialwCutoff(\
        alpha=1.8,rc=1.9/8.,normalize=0.05)
    pota= potential.RZToverticalPotential(b_p,1.1)
    o= Orbits([Orbit(vxvv=[20.,10.,2.,3.2,3.4,-100.],
                     radec=True,ro=8.0,vo=220.0),
               Orbit(vxvv=[20.,10.,2.,3.2,3.4,-100.],
                     radec=True,ro=8.0,vo=220.0)])
    ts= numpy.linspace(0.,3.5/bovy_conversion.time_in_Gyr(vo=220.0,ro=8.0),
                       1000,endpoint=True)
    with pytest.raises(AssertionError) as excinfo:
        o.integrate(ts,pota,method="odeint")
    return None

def test_orbit_dim_1dPot_2dOrb():
    # Test that orbit integration throws an error when using a potential that
    # is lower dimensional than the orbit, for a 1D potential
    from galpy.orbit import Orbit, Orbits
    b_p= potential.PowerSphericalPotentialwCutoff(\
        alpha=1.8,rc=1.9/8.,normalize=0.05)
    pota= [b_p.toVertical(1.1)]
    o= Orbits([Orbit(vxvv=[1.1,0.1,1.1,0.1]),Orbit(vxvv=[1.1,0.1,1.1,0.1])])
    ts= numpy.linspace(0.,10.,1001)
    with pytest.raises(AssertionError) as excinfo:
        o.integrate(ts,pota,method="leapfrog")
    with pytest.raises(AssertionError) as excinfo:
        o.integrate(ts,pota,method="dop853")
    return None

# Test the error for when explicit stepsize does not divide the output stepsize
def test_check_integrate_dt():
    from galpy.orbit import Orbit, Orbits
    from galpy.potential import LogarithmicHaloPotential
    lp= LogarithmicHaloPotential(normalize=1.,q=0.9)
    o= Orbits([Orbit([1.,0.1,1.2,0.3,0.2,2.]),
               Orbit([1.,0.1,1.2,0.3,0.2,2.])])
    times= numpy.linspace(0.,7.,251)
    # This shouldn't work
    try:
        o.integrate(times,lp,dt=(times[1]-times[0])/4.*1.1)
    except ValueError: pass
    else: raise AssertionError('dt that is not an integer divisor of the output step size does not raise a ValueError')
    # This should
    try:
        o.integrate(times,lp,dt=(times[1]-times[0])/4.)
    except ValueError:
        raise AssertionError('dt that is an integer divisor of the output step size raises a ValueError')
    return None

# Test that evaluating coordinate functions for integrated orbits works
def test_coordinate_interpolation():
    from galpy.orbit import Orbit, Orbits
    from galpy.potential import MWPotential2014
    numpy.random.seed(1)
    nrand= 10
    Rs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)+1.
    vRs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)
    vTs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)+1.
    zs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)
    vzs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)
    phis= 2.*numpy.pi*(2.*numpy.random.uniform(size=nrand)-1.)
    os= Orbits(list(zip(Rs,vRs,vTs,zs,vzs,phis)))
    list_os= [Orbit([R,vR,vT,z,vz,phi])
              for R,vR,vT,z,vz,phi in zip(Rs,vRs,vTs,zs,vzs,phis)]
    # Before integration
    for ii in range(nrand):
        # .time is special, just a single array
        assert numpy.all(numpy.fabs(os.time()-list_os[ii].time()) < 1e-10), 'Evaluating Orbits time does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.R()[ii]-list_os[ii].R()) < 1e-10), 'Evaluating Orbits R does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.r()[ii]-list_os[ii].r()) < 1e-10), 'Evaluating Orbits r does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vR()[ii]-list_os[ii].vR()) < 1e-10), 'Evaluating Orbits vR does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vT()[ii]-list_os[ii].vT()) < 1e-10), 'Evaluating Orbits vT does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.z()[ii]-list_os[ii].z()) < 1e-10), 'Evaluating Orbits z does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vz()[ii]-list_os[ii].vz()) < 1e-10), 'Evaluating Orbits vz does not agree with Orbit'
        assert numpy.all(numpy.fabs(((os.phi()[ii]-list_os[ii].phi()+numpy.pi) % (2.*numpy.pi)) - numpy.pi) < 1e-10), 'Evaluating Orbits phi does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.x()[ii]-list_os[ii].x()) < 1e-10), 'Evaluating Orbits x does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.y()[ii]-list_os[ii].y()) < 1e-10), 'Evaluating Orbits y does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vx()[ii]-list_os[ii].vx()) < 1e-10), 'Evaluating Orbits vx does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vy()[ii]-list_os[ii].vy()) < 1e-10), 'Evaluating Orbits vy does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vphi()[ii]-list_os[ii].vphi()) < 1e-10), 'Evaluating Orbits vphi does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.ra()[ii]-list_os[ii].ra()) < 1e-10), 'Evaluating Orbits ra  does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.dec()[ii]-list_os[ii].dec()) < 1e-10), 'Evaluating Orbits dec does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.dist()[ii]-list_os[ii].dist()) < 1e-10), 'Evaluating Orbits dist does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.ll()[ii]-list_os[ii].ll()) < 1e-10), 'Evaluating Orbits ll does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.bb()[ii]-list_os[ii].bb()) < 1e-10), 'Evaluating Orbits bb  does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.pmra()[ii]-list_os[ii].pmra()) < 1e-10), 'Evaluating Orbits pmra does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.pmdec()[ii]-list_os[ii].pmdec()) < 1e-10), 'Evaluating Orbits pmdec does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.pmll()[ii]-list_os[ii].pmll()) < 1e-10), 'Evaluating Orbits pmll does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.pmbb()[ii]-list_os[ii].pmbb()) < 1e-10), 'Evaluating Orbits pmbb does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vra()[ii]-list_os[ii].vra()) < 1e-10), 'Evaluating Orbits vra does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vdec()[ii]-list_os[ii].vdec()) < 1e-10), 'Evaluating Orbits vdec does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vll()[ii]-list_os[ii].vll()) < 1e-10), 'Evaluating Orbits vll does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vbb()[ii]-list_os[ii].vbb()) < 1e-10), 'Evaluating Orbits vbb does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vlos()[ii]-list_os[ii].vlos()) < 1e-10), 'Evaluating Orbits vlos does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.helioX()[ii]-list_os[ii].helioX()) < 1e-10), 'Evaluating Orbits helioX does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.helioY()[ii]-list_os[ii].helioY()) < 1e-10), 'Evaluating Orbits helioY does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.helioZ()[ii]-list_os[ii].helioZ()) < 1e-10), 'Evaluating Orbits helioZ does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.U()[ii]-list_os[ii].U()) < 1e-10), 'Evaluating Orbits U does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.V()[ii]-list_os[ii].V()) < 1e-10), 'Evaluating Orbits V does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.W()[ii]-list_os[ii].W()) < 1e-10), 'Evaluating Orbits W does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.SkyCoord().ra[ii]-list_os[ii].SkyCoord().ra).to(u.deg).value < 1e-10), 'Evaluating Orbits SkyCoord does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.SkyCoord().dec[ii]-list_os[ii].SkyCoord().dec).to(u.deg).value < 1e-10), 'Evaluating Orbits SkyCoord does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.SkyCoord().distance[ii]-list_os[ii].SkyCoord().distance).to(u.kpc).value < 1e-10), 'Evaluating Orbits SkyCoord does not agree with Orbit'
        if _APY3:
            assert numpy.all(numpy.fabs(os.SkyCoord().pm_ra_cosdec[ii]-list_os[ii].SkyCoord().pm_ra_cosdec).to(u.mas/u.yr).value < 1e-10), 'Evaluating Orbits SkyCoord does not agree with Orbit'
            assert numpy.all(numpy.fabs(os.SkyCoord().pm_dec[ii]-list_os[ii].SkyCoord().pm_dec).to(u.mas/u.yr).value < 1e-10), 'Evaluating Orbits SkyCoord does not agree with Orbit'
            assert numpy.all(numpy.fabs(os.SkyCoord().radial_velocity[ii]-list_os[ii].SkyCoord().radial_velocity).to(u.km/u.s).value < 1e-10), 'Evaluating Orbits SkyCoord does not agree with Orbit'
    # Integrate all
    times= numpy.linspace(0.,10.,1001)
    os.integrate(times,MWPotential2014)
    [o.integrate(times,MWPotential2014) for o in list_os]
    # Test exact times of integration
    for ii in range(nrand):
        # .time is special, just a single array
        assert numpy.all(numpy.fabs(os.time(times)-list_os[ii].time(times)) < 1e-10), 'Evaluating Orbits time does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.R(times)[ii]-list_os[ii].R(times)) < 1e-10), 'Evaluating Orbits R does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.r(times)[ii]-list_os[ii].r(times)) < 1e-10), 'Evaluating Orbits r does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vR(times)[ii]-list_os[ii].vR(times)) < 1e-10), 'Evaluating Orbits vR does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vT(times)[ii]-list_os[ii].vT(times)) < 1e-10), 'Evaluating Orbits vT does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.z(times)[ii]-list_os[ii].z(times)) < 1e-10), 'Evaluating Orbits z does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vz(times)[ii]-list_os[ii].vz(times)) < 1e-10), 'Evaluating Orbits vz does not agree with Orbit'
        assert numpy.all(numpy.fabs(((os.phi(times)[ii]-list_os[ii].phi(times)+numpy.pi) % (2.*numpy.pi)) - numpy.pi) < 1e-10), 'Evaluating Orbits phi does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.x(times)[ii]-list_os[ii].x(times)) < 1e-10), 'Evaluating Orbits x does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.y(times)[ii]-list_os[ii].y(times)) < 1e-10), 'Evaluating Orbits y does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vx(times)[ii]-list_os[ii].vx(times)) < 1e-10), 'Evaluating Orbits vx does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vy(times)[ii]-list_os[ii].vy(times)) < 1e-10), 'Evaluating Orbits vy does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vphi(times)[ii]-list_os[ii].vphi(times)) < 1e-10), 'Evaluating Orbits vphi does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.ra(times)[ii]-list_os[ii].ra(times)) < 1e-10), 'Evaluating Orbits ra  does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.dec(times)[ii]-list_os[ii].dec(times)) < 1e-10), 'Evaluating Orbits dec does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.dist(times)[ii]-list_os[ii].dist(times)) < 1e-10), 'Evaluating Orbits dist does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.ll(times)[ii]-list_os[ii].ll(times)) < 1e-10), 'Evaluating Orbits ll does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.bb(times)[ii]-list_os[ii].bb(times)) < 1e-10), 'Evaluating Orbits bb  does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.pmra(times)[ii]-list_os[ii].pmra(times)) < 1e-10), 'Evaluating Orbits pmra does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.pmdec(times)[ii]-list_os[ii].pmdec(times)) < 1e-10), 'Evaluating Orbits pmdec does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.pmll(times)[ii]-list_os[ii].pmll(times)) < 1e-10), 'Evaluating Orbits pmll does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.pmbb(times)[ii]-list_os[ii].pmbb(times)) < 1e-10), 'Evaluating Orbits pmbb does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vra(times)[ii]-list_os[ii].vra(times)) < 1e-10), 'Evaluating Orbits vra does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vdec(times)[ii]-list_os[ii].vdec(times)) < 1e-10), 'Evaluating Orbits vdec does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vll(times)[ii]-list_os[ii].vll(times)) < 1e-10), 'Evaluating Orbits vll does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vbb(times)[ii]-list_os[ii].vbb(times)) < 1e-10), 'Evaluating Orbits vbb does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vlos(times)[ii]-list_os[ii].vlos(times)) < 1e-9), 'Evaluating Orbits vlos does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.helioX(times)[ii]-list_os[ii].helioX(times)) < 1e-10), 'Evaluating Orbits helioX does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.helioY(times)[ii]-list_os[ii].helioY(times)) < 1e-10), 'Evaluating Orbits helioY does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.helioZ(times)[ii]-list_os[ii].helioZ(times)) < 1e-10), 'Evaluating Orbits helioZ does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.U(times)[ii]-list_os[ii].U(times)) < 1e-10), 'Evaluating Orbits U does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.V(times)[ii]-list_os[ii].V(times)) < 1e-10), 'Evaluating Orbits V does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.W(times)[ii]-list_os[ii].W(times)) < 1e-10), 'Evaluating Orbits W does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.SkyCoord(times).ra[ii]-list_os[ii].SkyCoord(times).ra).to(u.deg).value < 1e-10), 'Evaluating Orbits SkyCoord does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.SkyCoord(times).dec[ii]-list_os[ii].SkyCoord(times).dec).to(u.deg).value < 1e-10), 'Evaluating Orbits SkyCoord does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.SkyCoord(times).distance[ii]-list_os[ii].SkyCoord(times).distance).to(u.kpc).value < 1e-10), 'Evaluating Orbits SkyCoord does not agree with Orbit'
        if _APY3:
            assert numpy.all(numpy.fabs(os.SkyCoord(times).pm_ra_cosdec[ii]-list_os[ii].SkyCoord(times).pm_ra_cosdec).to(u.mas/u.yr).value < 1e-10), 'Evaluating Orbits SkyCoord does not agree with Orbit'
            assert numpy.all(numpy.fabs(os.SkyCoord(times).pm_dec[ii]-list_os[ii].SkyCoord(times).pm_dec).to(u.mas/u.yr).value < 1e-10), 'Evaluating Orbits SkyCoord does not agree with Orbit'
            assert numpy.all(numpy.fabs(os.SkyCoord(times).radial_velocity[ii]-list_os[ii].SkyCoord(times).radial_velocity).to(u.km/u.s).value < 1e-9), 'Evaluating Orbits SkyCoord does not agree with Orbit'
        # Also a single time in the array ...
        # .time is special, just a single array
        assert numpy.all(numpy.fabs(os.time(times[1])-list_os[ii].time(times[1])) < 1e-10), 'Evaluating Orbits time does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.R(times[1])[ii]-list_os[ii].R(times[1])) < 1e-10), 'Evaluating Orbits R does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.r(times[1])[ii]-list_os[ii].r(times[1])) < 1e-10), 'Evaluating Orbits r does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vR(times[1])[ii]-list_os[ii].vR(times[1])) < 1e-10), 'Evaluating Orbits vR does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vT(times[1])[ii]-list_os[ii].vT(times[1])) < 1e-10), 'Evaluating Orbits vT does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.z(times[1])[ii]-list_os[ii].z(times[1])) < 1e-10), 'Evaluating Orbits z does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vz(times[1])[ii]-list_os[ii].vz(times[1])) < 1e-10), 'Evaluating Orbits vz does not agree with Orbit'
        assert numpy.all(numpy.fabs(((os.phi(times[1])[ii]-list_os[ii].phi(times[1])+numpy.pi) % (2.*numpy.pi)) - numpy.pi) < 1e-10), 'Evaluating Orbits phi does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.x(times[1])[ii]-list_os[ii].x(times[1])) < 1e-10), 'Evaluating Orbits x does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.y(times[1])[ii]-list_os[ii].y(times[1])) < 1e-10), 'Evaluating Orbits y does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vx(times[1])[ii]-list_os[ii].vx(times[1])) < 1e-10), 'Evaluating Orbits vx does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vy(times[1])[ii]-list_os[ii].vy(times[1])) < 1e-10), 'Evaluating Orbits vy does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vphi(times[1])[ii]-list_os[ii].vphi(times[1])) < 1e-10), 'Evaluating Orbits vphi does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.ra(times[1])[ii]-list_os[ii].ra(times[1])) < 1e-10), 'Evaluating Orbits ra  does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.dec(times[1])[ii]-list_os[ii].dec(times[1])) < 1e-10), 'Evaluating Orbits dec does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.dist(times[1])[ii]-list_os[ii].dist(times[1])) < 1e-10), 'Evaluating Orbits dist does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.ll(times[1])[ii]-list_os[ii].ll(times[1])) < 1e-10), 'Evaluating Orbits ll does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.bb(times[1])[ii]-list_os[ii].bb(times[1])) < 1e-10), 'Evaluating Orbits bb  does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.pmra(times[1])[ii]-list_os[ii].pmra(times[1])) < 1e-10), 'Evaluating Orbits pmra does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.pmdec(times[1])[ii]-list_os[ii].pmdec(times[1])) < 1e-10), 'Evaluating Orbits pmdec does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.pmll(times[1])[ii]-list_os[ii].pmll(times[1])) < 1e-10), 'Evaluating Orbits pmll does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.pmbb(times[1])[ii]-list_os[ii].pmbb(times[1])) < 1e-10), 'Evaluating Orbits pmbb does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vra(times[1])[ii]-list_os[ii].vra(times[1])) < 1e-10), 'Evaluating Orbits vra does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vdec(times[1])[ii]-list_os[ii].vdec(times[1])) < 1e-10), 'Evaluating Orbits vdec does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vll(times[1])[ii]-list_os[ii].vll(times[1])) < 1e-10), 'Evaluating Orbits vll does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vbb(times[1])[ii]-list_os[ii].vbb(times[1])) < 1e-10), 'Evaluating Orbits vbb does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vlos(times[1])[ii]-list_os[ii].vlos(times[1])) < 1e-10), 'Evaluating Orbits vlos does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.helioX(times[1])[ii]-list_os[ii].helioX(times[1])) < 1e-10), 'Evaluating Orbits helioX does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.helioY(times[1])[ii]-list_os[ii].helioY(times[1])) < 1e-10), 'Evaluating Orbits helioY does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.helioZ(times[1])[ii]-list_os[ii].helioZ(times[1])) < 1e-10), 'Evaluating Orbits helioZ does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.U(times[1])[ii]-list_os[ii].U(times[1])) < 1e-10), 'Evaluating Orbits U does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.V(times[1])[ii]-list_os[ii].V(times[1])) < 1e-10), 'Evaluating Orbits V does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.W(times[1])[ii]-list_os[ii].W(times[1])) < 1e-10), 'Evaluating Orbits W does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.SkyCoord(times[1]).ra[ii]-list_os[ii].SkyCoord(times[1]).ra).to(u.deg).value < 1e-10), 'Evaluating Orbits SkyCoord does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.SkyCoord(times[1]).dec[ii]-list_os[ii].SkyCoord(times[1]).dec).to(u.deg).value < 1e-10), 'Evaluating Orbits SkyCoord does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.SkyCoord(times[1]).distance[ii]-list_os[ii].SkyCoord(times[1]).distance).to(u.kpc).value < 1e-10), 'Evaluating Orbits SkyCoord does not agree with Orbit'
        if _APY3:
            assert numpy.all(numpy.fabs(os.SkyCoord(times[1]).pm_ra_cosdec[ii]-list_os[ii].SkyCoord(times[1]).pm_ra_cosdec).to(u.mas/u.yr).value < 1e-10), 'Evaluating Orbits SkyCoord does not agree with Orbit'
            assert numpy.all(numpy.fabs(os.SkyCoord(times[1]).pm_dec[ii]-list_os[ii].SkyCoord(times[1]).pm_dec).to(u.mas/u.yr).value < 1e-10), 'Evaluating Orbits SkyCoord does not agree with Orbit'
            assert numpy.all(numpy.fabs(os.SkyCoord(times[1]).radial_velocity[ii]-list_os[ii].SkyCoord(times[1]).radial_velocity).to(u.km/u.s).value < 1e-10), 'Evaluating Orbits SkyCoord does not agree with Orbit'
    # Test actual interpolated
    itimes= times[:-2]+(times[1]-times[0])/2.
    for ii in range(nrand):
        assert numpy.all(numpy.fabs(os.R(itimes)[ii]-list_os[ii].R(itimes)) < 1e-10), 'Evaluating Orbits R does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.r(itimes)[ii]-list_os[ii].r(itimes)) < 1e-10), 'Evaluating Orbits r does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vR(itimes)[ii]-list_os[ii].vR(itimes)) < 1e-10), 'Evaluating Orbits vR does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vT(itimes)[ii]-list_os[ii].vT(itimes)) < 1e-10), 'Evaluating Orbits vT does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.z(itimes)[ii]-list_os[ii].z(itimes)) < 1e-10), 'Evaluating Orbits z does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vz(itimes)[ii]-list_os[ii].vz(itimes)) < 1e-10), 'Evaluating Orbits vz does not agree with Orbit'
        assert numpy.all(numpy.fabs(((os.phi(itimes)[ii]-list_os[ii].phi(itimes)+numpy.pi) % (2.*numpy.pi)) - numpy.pi) < 1e-10), 'Evaluating Orbits phi does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.x(itimes)[ii]-list_os[ii].x(itimes)) < 1e-10), 'Evaluating Orbits x does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.y(itimes)[ii]-list_os[ii].y(itimes)) < 1e-10), 'Evaluating Orbits y does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vx(itimes)[ii]-list_os[ii].vx(itimes)) < 1e-10), 'Evaluating Orbits vx does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vy(itimes)[ii]-list_os[ii].vy(itimes)) < 1e-10), 'Evaluating Orbits vy does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vphi(itimes)[ii]-list_os[ii].vphi(itimes)) < 1e-10), 'Evaluating Orbits vphidoes not agree with Orbit'
        assert numpy.all(numpy.fabs(os.ra(itimes)[ii]-list_os[ii].ra(itimes)) < 1e-10), 'Evaluating Orbits ra  does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.dec(itimes)[ii]-list_os[ii].dec(itimes)) < 1e-10), 'Evaluating Orbits dec does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.dist(itimes)[ii]-list_os[ii].dist(itimes)) < 1e-10), 'Evaluating Orbits dist does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.ll(itimes)[ii]-list_os[ii].ll(itimes)) < 1e-10), 'Evaluating Orbits ll does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.bb(itimes)[ii]-list_os[ii].bb(itimes)) < 1e-10), 'Evaluating Orbits bb  does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.pmra(itimes)[ii]-list_os[ii].pmra(itimes)) < 1e-10), 'Evaluating Orbits pmra does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.pmdec(itimes)[ii]-list_os[ii].pmdec(itimes)) < 1e-10), 'Evaluating Orbits pmdec does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.pmll(itimes)[ii]-list_os[ii].pmll(itimes)) < 1e-10), 'Evaluating Orbits pmll does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.pmbb(itimes)[ii]-list_os[ii].pmbb(itimes)) < 1e-10), 'Evaluating Orbits pmbb does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vra(itimes)[ii]-list_os[ii].vra(itimes)) < 1e-10), 'Evaluating Orbits vra does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vdec(itimes)[ii]-list_os[ii].vdec(itimes)) < 1e-10), 'Evaluating Orbits vdec does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vll(itimes)[ii]-list_os[ii].vll(itimes)) < 1e-10), 'Evaluating Orbits ll does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vbb(itimes)[ii]-list_os[ii].vbb(itimes)) < 1e-10), 'Evaluating Orbits vbb does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vlos(itimes)[ii]-list_os[ii].vlos(itimes)) < 1e-10), 'Evaluating Orbits vlos does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.helioX(itimes)[ii]-list_os[ii].helioX(itimes)) < 1e-10), 'Evaluating Orbits helioX does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.helioY(itimes)[ii]-list_os[ii].helioY(itimes)) < 1e-10), 'Evaluating Orbits helioY does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.helioZ(itimes)[ii]-list_os[ii].helioZ(itimes)) < 1e-10), 'Evaluating Orbits helioZ does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.U(itimes)[ii]-list_os[ii].U(itimes)) < 1e-10), 'Evaluating Orbits U does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.V(itimes)[ii]-list_os[ii].V(itimes)) < 1e-10), 'Evaluating Orbits V does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.W(itimes)[ii]-list_os[ii].W(itimes)) < 1e-10), 'Evaluating Orbits W does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.SkyCoord(itimes).ra[ii]-list_os[ii].SkyCoord(itimes).ra).to(u.deg).value < 1e-10), 'Evaluating Orbits SkyCoord does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.SkyCoord(itimes).dec[ii]-list_os[ii].SkyCoord(itimes).dec).to(u.deg).value < 1e-10), 'Evaluating Orbits SkyCoord does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.SkyCoord(itimes).distance[ii]-list_os[ii].SkyCoord(itimes).distance).to(u.kpc).value < 1e-10), 'Evaluating Orbits SkyCoord does not agree with Orbit'
        if _APY3:
            assert numpy.all(numpy.fabs(os.SkyCoord(itimes).pm_ra_cosdec[ii]-list_os[ii].SkyCoord(itimes).pm_ra_cosdec).to(u.mas/u.yr).value < 1e-10), 'Evaluating Orbits SkyCoord does not agree with Orbit'
            assert numpy.all(numpy.fabs(os.SkyCoord(itimes).pm_dec[ii]-list_os[ii].SkyCoord(itimes).pm_dec).to(u.mas/u.yr).value < 1e-10), 'Evaluating Orbits SkyCoord does not agree with Orbit'
            assert numpy.all(numpy.fabs(os.SkyCoord(itimes).radial_velocity[ii]-list_os[ii].SkyCoord(itimes).radial_velocity).to(u.km/u.s).value < 1e-10), 'Evaluating Orbits SkyCoord does not agree with Orbit'
        # Also a single time in the array ...
        assert numpy.all(numpy.fabs(os.R(itimes[1])[ii]-list_os[ii].R(itimes[1])) < 1e-10), 'Evaluating Orbits R does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.r(itimes[1])[ii]-list_os[ii].r(itimes[1])) < 1e-10), 'Evaluating Orbits r does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vR(itimes[1])[ii]-list_os[ii].vR(itimes[1])) < 1e-10), 'Evaluating Orbits vR does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vT(itimes[1])[ii]-list_os[ii].vT(itimes[1])) < 1e-10), 'Evaluating Orbits vT does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.z(itimes[1])[ii]-list_os[ii].z(itimes[1])) < 1e-10), 'Evaluating Orbits z does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vz(itimes[1])[ii]-list_os[ii].vz(itimes[1])) < 1e-10), 'Evaluating Orbits vz does not agree with Orbit'
        assert numpy.all(numpy.fabs(((os.phi(itimes[1])[ii]-list_os[ii].phi(itimes[1])+numpy.pi) % (2.*numpy.pi)) - numpy.pi) < 1e-10), 'Evaluating Orbits phi does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.ra(itimes[1])[ii]-list_os[ii].ra(itimes[1])) < 1e-10), 'Evaluating Orbits ra  does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.dec(itimes[1])[ii]-list_os[ii].dec(itimes[1])) < 1e-10), 'Evaluating Orbits dec does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.dist(itimes[1])[ii]-list_os[ii].dist(itimes[1])) < 1e-10), 'Evaluating Orbits dist does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.ll(itimes[1])[ii]-list_os[ii].ll(itimes[1])) < 1e-10), 'Evaluating Orbits ll does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.bb(itimes[1])[ii]-list_os[ii].bb(itimes[1])) < 1e-10), 'Evaluating Orbits bb  does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.pmra(itimes[1])[ii]-list_os[ii].pmra(itimes[1])) < 1e-10), 'Evaluating Orbits pmra does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.pmdec(itimes[1])[ii]-list_os[ii].pmdec(itimes[1])) < 1e-10), 'Evaluating Orbits pmdec does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.pmll(itimes[1])[ii]-list_os[ii].pmll(itimes[1])) < 1e-10), 'Evaluating Orbits pmll does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.pmbb(itimes[1])[ii]-list_os[ii].pmbb(itimes[1])) < 1e-10), 'Evaluating Orbits pmbb does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vra(itimes[1])[ii]-list_os[ii].vra(itimes[1])) < 1e-10), 'Evaluating Orbits vra does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vdec(itimes[1])[ii]-list_os[ii].vdec(itimes[1])) < 1e-10), 'Evaluating Orbits vdec does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vll(itimes[1])[ii]-list_os[ii].vll(itimes[1])) < 1e-10), 'Evaluating Orbits vll does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vbb(itimes[1])[ii]-list_os[ii].vbb(itimes[1])) < 1e-10), 'Evaluating Orbits vbb does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vlos(itimes[1])[ii]-list_os[ii].vlos(itimes[1])) < 1e-10), 'Evaluating Orbits vlos does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.helioX(itimes[1])[ii]-list_os[ii].helioX(itimes[1])) < 1e-10), 'Evaluating Orbits helioX does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.helioY(itimes[1])[ii]-list_os[ii].helioY(itimes[1])) < 1e-10), 'Evaluating Orbits helioY does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.helioZ(itimes[1])[ii]-list_os[ii].helioZ(itimes[1])) < 1e-10), 'Evaluating Orbits helioZ does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.U(itimes[1])[ii]-list_os[ii].U(itimes[1])) < 1e-10), 'Evaluating Orbits U does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.V(itimes[1])[ii]-list_os[ii].V(itimes[1])) < 1e-10), 'Evaluating Orbits V does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.W(itimes[1])[ii]-list_os[ii].W(itimes[1])) < 1e-10), 'Evaluating Orbits W does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.SkyCoord(itimes[1]).ra[ii]-list_os[ii].SkyCoord(itimes[1]).ra).to(u.deg).value < 1e-10), 'Evaluating Orbits SkyCoord does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.SkyCoord(itimes[1]).dec[ii]-list_os[ii].SkyCoord(itimes[1]).dec).to(u.deg).value < 1e-10), 'Evaluating Orbits SkyCoord does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.SkyCoord(itimes[1]).distance[ii]-list_os[ii].SkyCoord(itimes[1]).distance).to(u.kpc).value < 1e-10), 'Evaluating Orbits SkyCoord does not agree with Orbit'
        if _APY3:
            assert numpy.all(numpy.fabs(os.SkyCoord(itimes[1]).pm_ra_cosdec[ii]-list_os[ii].SkyCoord(itimes[1]).pm_ra_cosdec).to(u.mas/u.yr).value < 1e-10), 'Evaluating Orbits SkyCoord does not agree with Orbit'
            assert numpy.all(numpy.fabs(os.SkyCoord(itimes[1]).pm_dec[ii]-list_os[ii].SkyCoord(itimes[1]).pm_dec).to(u.mas/u.yr).value < 1e-10), 'Evaluating Orbits SkyCoord does not agree with Orbit'
            assert numpy.all(numpy.fabs(os.SkyCoord(itimes[1]).radial_velocity[ii]-list_os[ii].SkyCoord(itimes[1]).radial_velocity).to(u.km/u.s).value < 1e-10), 'Evaluating Orbits SkyCoord does not agree with Orbit'
    return None

# Test that evaluating coordinate functions for integrated orbits works, 
# for 5D orbits
def test_coordinate_interpolation_5d():
    from galpy.orbit import Orbit, Orbits
    from galpy.potential import MWPotential2014
    numpy.random.seed(1)
    nrand= 20
    Rs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)+1.
    vRs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)
    vTs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)+1.
    zs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)
    vzs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)
    os= Orbits(list(zip(Rs,vRs,vTs,zs,vzs)))
    list_os= [Orbit([R,vR,vT,z,vz])
              for R,vR,vT,z,vz in zip(Rs,vRs,vTs,zs,vzs)]
    # Before integration
    for ii in range(nrand):
        assert numpy.all(numpy.fabs(os.R()[ii]-list_os[ii].R()) < 1e-10), 'Evaluating Orbits R does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.r()[ii]-list_os[ii].r()) < 1e-10), 'Evaluating Orbits r does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vR()[ii]-list_os[ii].vR()) < 1e-10), 'Evaluating Orbits vR does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vT()[ii]-list_os[ii].vT()) < 1e-10), 'Evaluating Orbits vT does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.z()[ii]-list_os[ii].z()) < 1e-10), 'Evaluating Orbits z does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vz()[ii]-list_os[ii].vz()) < 1e-10), 'Evaluating Orbits vz does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vphi()[ii]-list_os[ii].vphi()) < 1e-10), 'Evaluating Orbits vphi does not agree with Orbit'
    # Integrate all
    times= numpy.linspace(0.,10.,1001)
    os.integrate(times,MWPotential2014)
    [o.integrate(times,MWPotential2014) for o in list_os]
    # Test exact times of integration
    for ii in range(nrand):
        assert numpy.all(numpy.fabs(os.R(times)[ii]-list_os[ii].R(times)) < 1e-10), 'Evaluating Orbits R does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.r(times)[ii]-list_os[ii].r(times)) < 1e-10), 'Evaluating Orbits r does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vR(times)[ii]-list_os[ii].vR(times)) < 1e-10), 'Evaluating Orbits vR does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vT(times)[ii]-list_os[ii].vT(times)) < 1e-10), 'Evaluating Orbits vT does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.z(times)[ii]-list_os[ii].z(times)) < 1e-10), 'Evaluating Orbits z does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vz(times)[ii]-list_os[ii].vz(times)) < 1e-10), 'Evaluating Orbits vz does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vphi(times)[ii]-list_os[ii].vphi(times)) < 1e-10), 'Evaluating Orbits vphi does not agree with Orbit'
        # Also a single time in the array ...
        assert numpy.all(numpy.fabs(os.R(times[1])[ii]-list_os[ii].R(times[1])) < 1e-10), 'Evaluating Orbits R does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.r(times[1])[ii]-list_os[ii].r(times[1])) < 1e-10), 'Evaluating Orbits r does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vR(times[1])[ii]-list_os[ii].vR(times[1])) < 1e-10), 'Evaluating Orbits vR does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vT(times[1])[ii]-list_os[ii].vT(times[1])) < 1e-10), 'Evaluating Orbits vT does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.z(times[1])[ii]-list_os[ii].z(times[1])) < 1e-10), 'Evaluating Orbits z does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vz(times[1])[ii]-list_os[ii].vz(times[1])) < 1e-10), 'Evaluating Orbits vz does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vphi(times[1])[ii]-list_os[ii].vphi(times[1])) < 1e-10), 'Evaluating Orbits vphi does not agree with Orbit'
    # Test actual interpolated
    itimes= times[:-2]+(times[1]-times[0])/2.
    for ii in range(nrand):
        assert numpy.all(numpy.fabs(os.R(itimes)[ii]-list_os[ii].R(itimes)) < 1e-10), 'Evaluating Orbits R does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.r(itimes)[ii]-list_os[ii].r(itimes)) < 1e-10), 'Evaluating Orbits r does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vR(itimes)[ii]-list_os[ii].vR(itimes)) < 1e-10), 'Evaluating Orbits vR does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vT(itimes)[ii]-list_os[ii].vT(itimes)) < 1e-10), 'Evaluating Orbits vT does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.z(itimes)[ii]-list_os[ii].z(itimes)) < 1e-10), 'Evaluating Orbits z does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vz(itimes)[ii]-list_os[ii].vz(itimes)) < 1e-10), 'Evaluating Orbits vz does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vphi(itimes)[ii]-list_os[ii].vphi(itimes)) < 1e-10), 'Evaluating Orbits vphi does not agree with Orbit'
        # Also a single time in the array ...
        assert numpy.all(numpy.fabs(os.R(itimes[1])[ii]-list_os[ii].R(itimes[1])) < 1e-10), 'Evaluating Orbits R does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.r(itimes[1])[ii]-list_os[ii].r(itimes[1])) < 1e-10), 'Evaluating Orbits r does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vR(itimes[1])[ii]-list_os[ii].vR(itimes[1])) < 1e-10), 'Evaluating Orbits vR does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vT(itimes[1])[ii]-list_os[ii].vT(itimes[1])) < 1e-10), 'Evaluating Orbits vT does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.z(itimes[1])[ii]-list_os[ii].z(itimes[1])) < 1e-10), 'Evaluating Orbits z does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vz(itimes[1])[ii]-list_os[ii].vz(itimes[1])) < 1e-10), 'Evaluating Orbits vz does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vphi(itimes[1])[ii]-list_os[ii].vphi(itimes[1])) < 1e-10), 'Evaluating Orbits vphi does not agree with Orbit'
    return None

# Test interpolation with backwards orbit integration
def test_backinterpolation():
    from galpy.orbit import Orbit, Orbits
    from galpy.potential import MWPotential2014
    numpy.random.seed(1)
    nrand= 20
    Rs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)+1.
    vRs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)
    vTs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)+1.
    zs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)
    vzs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)
    phis= 2.*numpy.pi*(2.*numpy.random.uniform(size=nrand)-1.)
    os= Orbits(list(zip(Rs,vRs,vTs,zs,vzs,phis)))
    list_os= [Orbit([R,vR,vT,z,vz,phi])
              for R,vR,vT,z,vz,phi in zip(Rs,vRs,vTs,zs,vzs,phis)]
    # Integrate all
    times= numpy.linspace(0.,-10.,1001)
    os.integrate(times,MWPotential2014)
    [o.integrate(times,MWPotential2014) for o in list_os]
    # Test actual interpolated
    itimes= times[:-2]+(times[1]-times[0])/2.
    for ii in range(nrand):
        assert numpy.all(numpy.fabs(os.R(itimes)[ii]-list_os[ii].R(itimes)) < 1e-10), 'Evaluating Orbits R does not agree with Orbit'
        # Also a single time in the array ...
        assert numpy.all(numpy.fabs(os.R(itimes[1])[ii]-list_os[ii].R(itimes[1])) < 1e-10), 'Evaluating Orbits R does not agree with Orbit'
    return None

# Test that evaluating coordinate functions for integrated orbits works for
# a single orbit
def test_coordinate_interpolation_oneorbit():
    from galpy.orbit import Orbit, Orbits
    from galpy.potential import MWPotential2014
    numpy.random.seed(1)
    nrand= 1
    Rs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)+1.
    vRs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)
    vTs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)+1.
    zs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)
    vzs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)
    phis= 2.*numpy.pi*(2.*numpy.random.uniform(size=nrand)-1.)
    os= Orbits(list(zip(Rs,vRs,vTs,zs,vzs,phis)))
    list_os= [Orbit([R,vR,vT,z,vz,phi])
              for R,vR,vT,z,vz,phi in zip(Rs,vRs,vTs,zs,vzs,phis)]
    # Before integration
    for ii in range(nrand):
        # .time is special, just a single array
        assert numpy.all(numpy.fabs(os.time()-list_os[ii].time()) < 1e-10), 'Evaluating Orbits time does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.R()[ii]-list_os[ii].R()) < 1e-10), 'Evaluating Orbits R does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.r()[ii]-list_os[ii].r()) < 1e-10), 'Evaluating Orbits r does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vR()[ii]-list_os[ii].vR()) < 1e-10), 'Evaluating Orbits vR does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vT()[ii]-list_os[ii].vT()) < 1e-10), 'Evaluating Orbits vT does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.z()[ii]-list_os[ii].z()) < 1e-10), 'Evaluating Orbits z does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vz()[ii]-list_os[ii].vz()) < 1e-10), 'Evaluating Orbits vz does not agree with Orbit'
        assert numpy.all(numpy.fabs(((os.phi()[ii]-list_os[ii].phi()+numpy.pi) % (2.*numpy.pi)) - numpy.pi) < 1e-10), 'Evaluating Orbits phi does not agree with Orbit'
    # Integrate all
    times= numpy.linspace(0.,10.,1001)
    os.integrate(times,MWPotential2014)
    [o.integrate(times,MWPotential2014) for o in list_os]
    # Test exact times of integration
    for ii in range(nrand):
        # .time is special, just a single array
        assert numpy.all(numpy.fabs(os.time(times)-list_os[ii].time(times)) < 1e-10), 'Evaluating Orbits time does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.R(times)[ii]-list_os[ii].R(times)) < 1e-10), 'Evaluating Orbits R does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.r(times)[ii]-list_os[ii].r(times)) < 1e-10), 'Evaluating Orbits r does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vR(times)[ii]-list_os[ii].vR(times)) < 1e-10), 'Evaluating Orbits vR does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vT(times)[ii]-list_os[ii].vT(times)) < 1e-10), 'Evaluating Orbits vT does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.z(times)[ii]-list_os[ii].z(times)) < 1e-10), 'Evaluating Orbits z does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vz(times)[ii]-list_os[ii].vz(times)) < 1e-10), 'Evaluating Orbits vz does not agree with Orbit'
        assert numpy.all(numpy.fabs(((os.phi(times)[ii]-list_os[ii].phi(times)+numpy.pi) % (2.*numpy.pi)) - numpy.pi) < 1e-10), 'Evaluating Orbits phi does not agree with Orbit'
        # Also a single time in the array ...
        # .time is special, just a single array
        assert numpy.all(numpy.fabs(os.time(times[1])-list_os[ii].time(times[1])) < 1e-10), 'Evaluating Orbits time does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.R(times[1])[ii]-list_os[ii].R(times[1])) < 1e-10), 'Evaluating Orbits R does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.r(times[1])[ii]-list_os[ii].r(times[1])) < 1e-10), 'Evaluating Orbits r does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vR(times[1])[ii]-list_os[ii].vR(times[1])) < 1e-10), 'Evaluating Orbits vR does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vT(times[1])[ii]-list_os[ii].vT(times[1])) < 1e-10), 'Evaluating Orbits vT does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.z(times[1])[ii]-list_os[ii].z(times[1])) < 1e-10), 'Evaluating Orbits z does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vz(times[1])[ii]-list_os[ii].vz(times[1])) < 1e-10), 'Evaluating Orbits vz does not agree with Orbit'
        assert numpy.all(numpy.fabs(((os.phi(times[1])[ii]-list_os[ii].phi(times[1])+numpy.pi) % (2.*numpy.pi)) - numpy.pi) < 1e-10), 'Evaluating Orbits phi does not agree with Orbit'
    # Test actual interpolated
    itimes= times[:-2]+(times[1]-times[0])/2.
    for ii in range(nrand):
        assert numpy.all(numpy.fabs(os.R(itimes)[ii]-list_os[ii].R(itimes)) < 1e-10), 'Evaluating Orbits R does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.r(itimes)[ii]-list_os[ii].r(itimes)) < 1e-10), 'Evaluating Orbits r does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vR(itimes)[ii]-list_os[ii].vR(itimes)) < 1e-10), 'Evaluating Orbits vR does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vT(itimes)[ii]-list_os[ii].vT(itimes)) < 1e-10), 'Evaluating Orbits vT does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.z(itimes)[ii]-list_os[ii].z(itimes)) < 1e-10), 'Evaluating Orbits z does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vz(itimes)[ii]-list_os[ii].vz(itimes)) < 1e-10), 'Evaluating Orbits vz does not agree with Orbit'
        assert numpy.all(numpy.fabs(((os.phi(itimes)[ii]-list_os[ii].phi(itimes)+numpy.pi) % (2.*numpy.pi)) - numpy.pi) < 1e-10), 'Evaluating Orbits phi does not agree with Orbit'
        # Also a single time in the array ...
        assert numpy.all(numpy.fabs(os.R(itimes[1])[ii]-list_os[ii].R(itimes[1])) < 1e-10), 'Evaluating Orbits R does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.r(itimes[1])[ii]-list_os[ii].r(itimes[1])) < 1e-10), 'Evaluating Orbits r does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vR(itimes[1])[ii]-list_os[ii].vR(itimes[1])) < 1e-10), 'Evaluating Orbits vR does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vT(itimes[1])[ii]-list_os[ii].vT(itimes[1])) < 1e-10), 'Evaluating Orbits vT does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.z(itimes[1])[ii]-list_os[ii].z(itimes[1])) < 1e-10), 'Evaluating Orbits z does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.vz(itimes[1])[ii]-list_os[ii].vz(itimes[1])) < 1e-10), 'Evaluating Orbits vz does not agree with Orbit'
        assert numpy.all(numpy.fabs(((os.phi(itimes[1])[ii]-list_os[ii].phi(itimes[1])+numpy.pi) % (2.*numpy.pi)) - numpy.pi) < 1e-10), 'Evaluating Orbits phi does not agree with Orbit'
    return None

# Test that an error is raised when evaluating an orbit outside of the 
# integration range
def test_interpolate_outsiderange():
    from galpy.orbit import Orbit, Orbits
    from galpy.potential import MWPotential2014
    numpy.random.seed(1)
    nrand= 3
    Rs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)+1.
    vRs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)
    vTs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)+1.
    zs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)
    vzs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)
    phis= 2.*numpy.pi*(2.*numpy.random.uniform(size=nrand)-1.)
    os= Orbits(list(zip(Rs,vRs,vTs,zs,vzs,phis)))
    # Integrate all                                                            
    times= numpy.linspace(0.,10.,1001)
    os.integrate(times,MWPotential2014)
    with pytest.raises(ValueError) as excinfo:
        os.R(11.)
    with pytest.raises(ValueError) as excinfo:
        os.R(-1.)
    # Also for arrays that partially overlap
    with pytest.raises(ValueError) as excinfo:
        os.R(numpy.linspace(5.,11.,1001))
    with pytest.raises(ValueError) as excinfo:
        os.R(numpy.linspace(-5.,5.,1001))

def test_call_issue256():
    # Same as for Orbit instances: non-integrated orbit with t=/=0 should return eror
    from galpy.orbit import Orbits
    o = Orbits(vxvv=[[5.,-1.,0.8, 3, -0.1, 0]])
    # no integration of the orbit
    with pytest.raises(ValueError) as excinfo:
        o.R(30)
    return None

# Test that the energy, angular momentum, and Jacobi functions work as expected
def test_energy_jacobi_angmom():
    from galpy.orbit import Orbit, Orbits
    numpy.random.seed(1)
    nrand= 10
    Rs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)+1.
    vRs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)
    vTs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)+1.
    zs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)
    vzs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)
    phis= 2.*numpy.pi*(2.*numpy.random.uniform(size=nrand)-1.)
    # 6D
    os= Orbits(list(zip(Rs,vRs,vTs,zs,vzs,phis)))
    list_os= [Orbit([R,vR,vT,z,vz,phi])
              for R,vR,vT,z,vz,phi in zip(Rs,vRs,vTs,zs,vzs,phis)]
    _check_energy_jacobi_angmom(os,list_os)
    # 5D
    os= Orbits(list(zip(Rs,vRs,vTs,zs,vzs)))
    list_os= [Orbit([R,vR,vT,z,vz])
              for R,vR,vT,z,vz in zip(Rs,vRs,vTs,zs,vzs)]
    _check_energy_jacobi_angmom(os,list_os)
    # 4D
    os= Orbits(list(zip(Rs,vRs,vTs,phis)))
    list_os= [Orbit([R,vR,vT,phi])
              for R,vR,vT,phi in zip(Rs,vRs,vTs,phis)]
    _check_energy_jacobi_angmom(os,list_os)
    # 3D
    os= Orbits(list(zip(Rs,vRs,vTs)))
    list_os= [Orbit([R,vR,vT])
              for R,vR,vT in zip(Rs,vRs,vTs)]
    _check_energy_jacobi_angmom(os,list_os)
    # 2D
    os= Orbits(list(zip(zs,vzs)))
    list_os= [Orbit([z,vz])
              for z,vz in zip(zs,vzs)]
    _check_energy_jacobi_angmom(os,list_os)
    return None

def _check_energy_jacobi_angmom(os,list_os):
    nrand= len(os)
    from galpy.potential import MWPotential2014, SpiralArmsPotential, \
        DehnenBarPotential, LogarithmicHaloPotential
    sp= SpiralArmsPotential()
    dp= DehnenBarPotential()
    lp= LogarithmicHaloPotential(normalize=1.)
    if os.dim() == 1:
        from galpy.potential import toVerticalPotential
        MWPotential2014= toVerticalPotential(MWPotential2014,1.)
        lp= toVerticalPotential(lp,1.)
    # Before integration
    for ii in range(nrand):
        assert numpy.all(numpy.fabs(os.E(pot=MWPotential2014)[ii]/list_os[ii].E(pot=MWPotential2014)-1.) < 10.**-10.), 'Evaluating Orbits E does not agree with Orbit'
        if os.dim() == 3:
            assert numpy.all(numpy.fabs(os.ER(pot=MWPotential2014)[ii]/list_os[ii].ER(pot=MWPotential2014)-1.) < 10.**-10.), 'Evaluating Orbits ER does not agree with Orbit'
            assert numpy.all(numpy.fabs(os.Ez(pot=MWPotential2014)[ii]/list_os[ii].Ez(pot=MWPotential2014)-1.) < 10.**-10.), 'Evaluating Orbits Ez does not agree with Orbit'
        if os.phasedim() % 2 == 0 and os.dim() != 1:
            assert numpy.all(numpy.fabs(os.L()[ii]/list_os[ii].L()-1.) < 10.**-10.), 'Evaluating Orbits L does not agree with Orbit'
        if os.dim() != 1:
            assert numpy.all(numpy.fabs(os.Lz()[ii]/list_os[ii].Lz()-1.) < 10.**-10.), 'Evaluating Orbits Lz does not agree with Orbit'
        if os.phasedim() % 2 == 0 and os.dim() != 1:
            assert numpy.all(numpy.fabs(os.Jacobi(pot=MWPotential2014)[ii]/list_os[ii].Jacobi(pot=MWPotential2014)-1.) < 10.**-10.), 'Evaluating Orbits Jacobi does not agree with Orbit'
            # Also explicitly set OmegaP
            assert numpy.all(numpy.fabs(os.Jacobi(pot=MWPotential2014,OmegaP=0.6)[ii]/list_os[ii].Jacobi(pot=MWPotential2014,OmegaP=0.6)-1.) < 10.**-10.), 'Evaluating Orbits Jacobi does not agree with Orbit'
    # Potential for which array evaluation definitely works
    for ii in range(nrand):
        assert numpy.all(numpy.fabs(os.E(pot=lp)[ii]/list_os[ii].E(pot=lp)-1.) < 10.**-10.), 'Evaluating Orbits E does not agree with Orbit'
        if os.dim() == 3:
            assert numpy.all(numpy.fabs(os.ER(pot=lp)[ii]/list_os[ii].ER(pot=lp)-1.) < 10.**-10.), 'Evaluating Orbits ER does not agree with Orbit'
            assert numpy.all(numpy.fabs(os.Ez(pot=lp)[ii]/list_os[ii].Ez(pot=lp)-1.) < 10.**-10.), 'Evaluating Orbits Ez does not agree with Orbit'
        if os.phasedim() % 2 == 0 and os.dim() != 1:
            assert numpy.all(numpy.fabs(os.L()[ii]/list_os[ii].L()-1.) < 10.**-10.), 'Evaluating Orbits L does not agree with Orbit'
        if os.dim() != 1:
            assert numpy.all(numpy.fabs(os.Lz()[ii]/list_os[ii].Lz()-1.) < 10.**-10.), 'Evaluating Orbits Lz does not agree with Orbit'
        if os.phasedim() % 2 == 0 and os.dim() != 1:
            assert numpy.all(numpy.fabs(os.Jacobi(pot=lp)[ii]/list_os[ii].Jacobi(pot=lp)-1.) < 10.**-10.), 'Evaluating Orbits Jacobi does not agree with Orbit'
            # Also explicitly set OmegaP
            assert numpy.all(numpy.fabs(os.Jacobi(pot=lp,OmegaP=0.6)[ii]/list_os[ii].Jacobi(pot=lp,OmegaP=0.6)-1.) < 10.**-10.), 'Evaluating Orbits Jacobi does not agree with Orbit'
    # Integrate all
    times= numpy.linspace(0.,10.,1001)
    os.integrate(times,MWPotential2014)
    [o.integrate(times,MWPotential2014) for o in list_os]
    for ii in range(nrand):
        assert numpy.all(numpy.fabs(os.E(times,pot=MWPotential2014)[ii]/list_os[ii].E(times,pot=MWPotential2014)-1.) < 10.**-10.), 'Evaluating Orbits E does not agree with Orbit'
        if os.dim() == 3:
            assert numpy.all(numpy.fabs(os.ER(times,pot=MWPotential2014)[ii]/list_os[ii].ER(times,pot=MWPotential2014)-1.) < 10.**-10.), 'Evaluating Orbits ER does not agree with Orbit'
            assert numpy.all(numpy.fabs(os.Ez(times,pot=MWPotential2014)[ii]/list_os[ii].Ez(times,pot=MWPotential2014)-1.) < 10.**-10.), 'Evaluating Orbits Ez does not agree with Orbit'
        if os.phasedim() % 2 == 0 and os.dim() != 1:
            assert numpy.all(numpy.fabs(os.L(times)[ii]/list_os[ii].L(times)-1.) < 10.**-10.), 'Evaluating Orbits L does not agree with Orbit'
        if os.dim() != 1:
            assert numpy.all(numpy.fabs(os.Lz(times)[ii]/list_os[ii].Lz(times)-1.) < 10.**-10.), 'Evaluating Orbits Lz does not agree with Orbit'
        if os.phasedim() % 2 == 0 and os.dim() != 1:
            assert numpy.all(numpy.fabs(os.Jacobi(times,pot=MWPotential2014)[ii]/list_os[ii].Jacobi(times,pot=MWPotential2014)-1.) < 10.**-10.), 'Evaluating Orbits Jacobi does not agree with Orbit'
            # Also explicitly set OmegaP
            assert numpy.all(numpy.fabs(os.Jacobi(times,pot=MWPotential2014,OmegaP=0.6)[ii]/list_os[ii].Jacobi(times,pot=MWPotential2014,OmegaP=0.6)-1.) < 10.**-10.), 'Evaluating Orbits Jacobi does not agree with Orbit'
    # Don't do non-axi for odd-D Orbits or 1D
    if os.phasedim() % 2 == 1 or os.dim() == 1: return None
    # Add bar and spiral
    for ii in range(nrand):
        assert numpy.all(numpy.fabs(os.E(pot=MWPotential2014+dp+sp)[ii]/list_os[ii].E(pot=MWPotential2014+dp+sp)-1.) < 10.**-10.), 'Evaluating Orbits E does not agree with Orbit'
        if os.dim() == 3:
            assert numpy.all(numpy.fabs(os.ER(pot=MWPotential2014+dp+sp)[ii]/list_os[ii].ER(pot=MWPotential2014+dp+sp)-1.) < 10.**-10.), 'Evaluating Orbits ER does not agree with Orbit'
            assert numpy.all(numpy.fabs(os.Ez(pot=MWPotential2014+dp+sp)[ii]/list_os[ii].Ez(pot=MWPotential2014+dp+sp)-1.) < 10.**-10.), 'Evaluating Orbits Ez does not agree with Orbit'
        if os.phasedim() % 2 == 0 and os.dim() != 1:
            assert numpy.all(numpy.fabs(os.L()[ii]/list_os[ii].L()-1.) < 10.**-10.), 'Evaluating Orbits L does not agree with Orbit'
        if os.dim() != 1:
            assert numpy.all(numpy.fabs(os.Lz()[ii]/list_os[ii].Lz()-1.) < 10.**-10.), 'Evaluating Orbits Lz does not agree with Orbit'
        if os.phasedim() % 2 == 0 and os.dim() != 1:
            assert numpy.all(numpy.fabs(os.Jacobi(pot=MWPotential2014+dp+sp)[ii]/list_os[ii].Jacobi(pot=MWPotential2014+dp+sp)-1.) < 10.**-10.), 'Evaluating Orbits Jacobi does not agree with Orbit'
            # Also explicitly set OmegaP
            assert numpy.all(numpy.fabs(os.Jacobi(pot=MWPotential2014+dp+sp,OmegaP=0.6)[ii]/list_os[ii].Jacobi(pot=MWPotential2014+dp+sp,OmegaP=0.6)-1.) < 10.**-10.), 'Evaluating Orbits Jacobi does not agree with Orbit'
    return None

# Test that we can still get outputs when there aren't enough points for an actual interpolation
# Test whether Orbits evaluation methods sound warning when called with
# unitless time when orbit is integrated with unitfull times
def test_orbits_method_integrate_t_asQuantity_warning():
    from galpy.potential import MWPotential2014
    from galpy.orbit import Orbits
    from astropy import units
    from test_orbit import check_integrate_t_asQuantity_warning
    # Setup and integrate orbit
    ts= numpy.linspace(0.,10.,1001)*units.Gyr
    o= Orbits([[1.1,0.1,1.1,0.1,0.1,0.2],
               [1.1,0.1,1.1,0.1,0.1,0.2]])
    o.integrate(ts,MWPotential2014)
    # Now check
    check_integrate_t_asQuantity_warning(o,'R')
    return None

# Test new orbits formed from __call__
def test_newOrbits():
    from galpy.orbit import Orbits
    o= Orbits([[1.,0.1,1.1,0.1,0.,0.],
               [1.1,0.3,0.9,-0.2,0.3,2.]])
    ts= numpy.linspace(0.,1.,21) #v. quick orbit integration
    lp= potential.LogarithmicHaloPotential(normalize=1.)
    o.integrate(ts,lp)
    no= o(ts[-1]) #new Orbits
    assert numpy.all(no.R() == o.R(ts[-1])), "New Orbits formed from calling an old orbit does not have the correct R"
    assert numpy.all(no.vR() == o.vR(ts[-1])), "New Orbits formed from calling an old orbit does not have the correct vR"
    assert numpy.all(no.vT() == o.vT(ts[-1])), "New Orbits formed from calling an old orbit does not have the correct vT"
    assert numpy.all(no.z() == o.z(ts[-1])), "New Orbits formed from calling an old orbit does not have the correct z"
    assert numpy.all(no.vz() == o.vz(ts[-1])), "New Orbits formed from calling an old orbit does not have the correct vz"
    assert numpy.all(no.phi() == o.phi(ts[-1])), "New Orbits formed from calling an old orbit does not have the correct phi"
    assert not no._roSet, "New Orbits formed from calling an old orbit does not have the correct roSet"
    assert not no._voSet, "New Orbits formed from calling an old orbit does not have the correct roSet"
    #Also test this for multiple time outputs
    nos= o(ts[-2:]) #new orbits
    #First t
    assert numpy.all(numpy.fabs(nos[0].R()-o.R(ts[-2])) < 10.**-10.), "New Orbits formed from calling an old orbit does not have the correct R"
    assert numpy.all(numpy.fabs(nos[0].vR()-o.vR(ts[-2])) < 10.**-10.), "New Orbits formed from calling an old orbit does not have the correct vR"
    assert numpy.all(numpy.fabs(nos[0].vT()-o.vT(ts[-2])) < 10.**-10.), "New Orbits formed from calling an old orbit does not have the correct vT"
    assert numpy.all(numpy.fabs(nos[0].z()-o.z(ts[-2])) < 10.**-10.), "New Orbits formed from calling an old orbit does not have the correct z"
    assert numpy.all(numpy.fabs(nos[0].vz()-o.vz(ts[-2])) < 10.**-10.), "New Orbits formed from calling an old orbit does not have the correct vz"
    assert numpy.all(numpy.fabs(nos[0].phi()-o.phi(ts[-2])) < 10.**-10.), "New Orbits formed from calling an old orbit does not have the correct phi"
    assert not nos[0]._roSet, "New Orbits formed from calling an old orbit does not have the correct roSet"
    assert not nos[0]._voSet, "New Orbits formed from calling an old orbit does not have the correct roSet"
    #Second t
    assert numpy.all(numpy.fabs(nos[1].R()-o.R(ts[-1])) < 10.**-10.), "New Orbits formed from calling an old orbit does not have the correct R"
    assert numpy.all(numpy.fabs(nos[1].vR()-o.vR(ts[-1])) < 10.**-10.), "New Orbits formed from calling an old orbit does not have the correct vR"
    assert numpy.all(numpy.fabs(nos[1].vT()-o.vT(ts[-1])) < 10.**-10.), "New Orbits formed from calling an old orbit does not have the correct vT"
    assert numpy.all(numpy.fabs(nos[1].z()-o.z(ts[-1])) < 10.**-10.), "New Orbits formed from calling an old orbit does not have the correct z"
    assert numpy.all(numpy.fabs(nos[1].vz()-o.vz(ts[-1])) < 10.**-10.), "New Orbits formed from calling an old orbit does not have the correct vz"
    assert numpy.all(numpy.fabs(nos[1].phi()-o.phi(ts[-1])) < 10.**-10.), "New Orbits formed from calling an old orbit does not have the correct phi"
    assert not nos[1]._roSet, "New Orbits formed from calling an old orbit does not have the correct roSet"
    assert not nos[1]._voSet, "New Orbits formed from calling an old orbit does not have the correct roSet"
    return None

# Test new orbits formed from __call__, before integration
def test_newOrbit_b4integration():
    from galpy.orbit import Orbits
    o= Orbits([[1.,0.1,1.1,0.1,0.,0.],
               [1.1,0.3,0.9,-0.2,0.3,2.]])
    no= o() #New Orbits formed before integration
    assert numpy.all(numpy.fabs(no.R()-o.R()) < 10.**-10.), "New Orbits formed from calling an old orbit does not have the correct R"
    assert numpy.all(numpy.fabs(no.vR()-o.vR()) < 10.**-10.), "New Orbits formed from calling an old orbit does not have the correct vR"
    assert numpy.all(numpy.fabs(no.vT()-o.vT()) < 10.**-10.), "New Orbits formed from calling an old orbit does not have the correct vT"
    assert numpy.all(numpy.fabs(no.z()-o.z()) < 10.**-10.), "New Orbits formed from calling an old orbit does not have the correct z"
    assert numpy.all(numpy.fabs(no.vz()-o.vz()) < 10.**-10.), "New Orbits formed from calling an old orbit does not have the correct vz"
    assert numpy.all(numpy.fabs(no.phi()-o.phi()) < 10.**-10.), "New Orbits formed from calling an old orbit does not have the correct phi"
    assert not no._roSet, "New Orbits formed from calling an old orbit does not have the correct roSet"
    assert not no._voSet, "New Orbits formed from calling an old orbit does not have the correct roSet"
    return None

# Check plotting routines
def test_plotting():
    from galpy.orbit import Orbit, Orbits
    from galpy.potential import LogarithmicHaloPotential
    o= Orbits([Orbit([1.,0.1,1.1,0.1,0.2,2.]),Orbit([1.,0.1,1.1,0.1,0.2,2.])])
    oa= Orbits([Orbit([1.,0.1,1.1,0.1,0.2]),Orbit([1.,0.1,1.1,0.1,0.2])])
    times= numpy.linspace(0.,7.,251)
    lp= LogarithmicHaloPotential(normalize=1.,q=0.8)
    # Integrate
    o.integrate(times,lp)
    oa.integrate(times,lp)
    # Some plots
    # Energy
    o.plotE()
    o.plotE(normed=True)
    o.plotE(pot=lp,d1='R')
    o.plotE(pot=lp,d1='vR')
    o.plotE(pot=lp,d1='vT')
    o.plotE(pot=lp,d1='z')
    o.plotE(pot=lp,d1='vz')
    o.plotE(pot=lp,d1='phi')
    oa.plotE()
    oa.plotE(pot=lp,d1='R')
    oa.plotE(pot=lp,d1='vR')
    oa.plotE(pot=lp,d1='vT')
    oa.plotE(pot=lp,d1='z')
    oa.plotE(pot=lp,d1='vz')
    # Vertical energy
    o.plotEz()
    o.plotEz(normed=True)
    o.plotEz(pot=lp,d1='R')
    o.plotEz(pot=lp,d1='vR')
    o.plotEz(pot=lp,d1='vT')
    o.plotEz(pot=lp,d1='z')
    o.plotEz(pot=lp,d1='vz')
    o.plotEz(pot=lp,d1='phi')
    oa.plotEz()
    oa.plotEz(normed=True)
    oa.plotEz(pot=lp,d1='R')
    oa.plotEz(pot=lp,d1='vR')
    oa.plotEz(pot=lp,d1='vT')
    oa.plotEz(pot=lp,d1='z')
    oa.plotEz(pot=lp,d1='vz')
    # Radial energy
    o.plotER()
    o.plotER(normed=True)
    # Radial energy
    oa.plotER()
    oa.plotER(normed=True)
    # EzJz
    o.plotEzJz()
    o.plotEzJz(pot=lp,d1='R')
    o.plotEzJz(pot=lp,d1='vR')
    o.plotEzJz(pot=lp,d1='vT')
    o.plotEzJz(pot=lp,d1='z')
    o.plotEzJz(pot=lp,d1='vz')
    o.plotEzJz(pot=lp,d1='phi')
    oa.plotEzJz()
    oa.plotEzJz(pot=lp,d1='R')
    oa.plotEzJz(pot=lp,d1='vR')
    oa.plotEzJz(pot=lp,d1='vT')
    oa.plotEzJz(pot=lp,d1='z')
    oa.plotEzJz(pot=lp,d1='vz')
    # Jacobi
    o.plotJacobi()
    o.plotJacobi(normed=True)
    o.plotJacobi(pot=lp,d1='R',OmegaP=1.)
    o.plotJacobi(pot=lp,d1='vR')
    o.plotJacobi(pot=lp,d1='vT')
    o.plotJacobi(pot=lp,d1='z')
    o.plotJacobi(pot=lp,d1='vz')
    o.plotJacobi(pot=lp,d1='phi')
    oa.plotJacobi()
    oa.plotJacobi(pot=lp,d1='R',OmegaP=1.)
    oa.plotJacobi(pot=lp,d1='vR')
    oa.plotJacobi(pot=lp,d1='vT')
    oa.plotJacobi(pot=lp,d1='z')
    oa.plotJacobi(pot=lp,d1='vz')
    # Plot the orbit itself
    o.plot() #defaults
    oa.plot()
    o.plot(d1='vR')
    o.plotR()
    o.plotvR(d1='vT')
    o.plotvT(d1='z')
    o.plotz(d1='vz')
    o.plotvz(d1='phi')
    o.plotphi(d1='vR')
    o.plotx(d1='vx')
    o.plotvx(d1='y')
    o.ploty(d1='vy')
    o.plotvy(d1='x')
    # Remaining attributes
    o.plot(d1='ra',d2='dec')
    o.plot(d2='ra',d1='dec')
    o.plot(d1='pmra',d2='pmdec')
    o.plot(d2='pmra',d1='pmdec')
    o.plot(d1='ll',d2='bb')
    o.plot(d2='ll',d1='bb')
    o.plot(d1='pmll',d2='pmbb')
    o.plot(d2='pmll',d1='pmbb')
    o.plot(d1='vlos',d2='dist')
    o.plot(d2='vlos',d1='dist')
    o.plot(d1='helioX',d2='U')
    o.plot(d2='helioX',d1='U')
    o.plot(d1='helioY',d2='V')
    o.plot(d2='helioY',d1='V')
    o.plot(d1='helioZ',d2='W')
    o.plot(d2='helioZ',d1='W')
    o.plot(d2='r',d1='R')
    o.plot(d2='R',d1='r')
    # Some more energies etc.
    o.plot(d1='E',d2='R')
    o.plot(d1='Enorm',d2='R')
    o.plot(d1='Ez',d2='R')
    o.plot(d1='Eznorm',d2='R')
    o.plot(d1='ER',d2='R')
    o.plot(d1='ERnorm',d2='R')
    o.plot(d1='Jacobi',d2='R')
    o.plot(d1='Jacobinorm',d2='R')
    # callables don't work
    # Expressions
    o.plot(d1='t',d2='r*R/vR')
    return None

def test_integrate_method_warning():
    """ Test Orbits.integrate raises an error if method is unvalid """
    from galpy.potential import MWPotential2014
    from galpy.orbit import Orbit, Orbits
    o = Orbits([Orbit(vxvv=[1.0, 0.1, 0.1, 0.5, 0.1, 0.0]),
                Orbit(vxvv=[1.0, 0.1, 0.1, 0.5, 0.1, 0.0])])
    t = numpy.arange(0.0, 10.0, 0.001)
    with pytest.raises(ValueError):
        o.integrate(t, MWPotential2014, method='rk4')

# Test that fallback onto Python integrators works for Orbits
def test_integrate_Cfallback_symplec():
    from test_potential import BurkertPotentialNoC
    from galpy.orbit import Orbit, Orbits
    times= numpy.linspace(0.,10.,1001)
    orbits_list= [Orbit([1.,0.1,1.]),Orbit([.9,0.3,1.]),
                  Orbit([1.2,-0.3,0.7])]
    orbits= Orbits(orbits_list)
    # Integrate as Orbits
    pot= BurkertPotentialNoC()
    pot.normalize(1.)
    orbits.integrate(times,pot,method='symplec4_c')
    # Integrate as multiple Orbits
    for o in orbits_list:
        o.integrate(times,pot,method='symplec4_c')
    # Compare
    for ii in range(len(orbits)):
        assert numpy.amax(numpy.fabs(orbits_list[ii].R(times)-orbits.R(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_list[ii].vR(times)-orbits.vR(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_list[ii].vT(times)-orbits.vT(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
    return None
    
def test_integrate_Cfallback_nonsymplec():
    from test_potential import BurkertPotentialNoC
    from galpy.orbit import Orbit, Orbits
    times= numpy.linspace(0.,10.,1001)
    orbits_list= [Orbit([1.,0.1,1.]),Orbit([.9,0.3,1.]),
                  Orbit([1.2,-0.3,0.7])]
    orbits= Orbits(orbits_list)
    # Integrate as Orbits
    pot= BurkertPotentialNoC()
    pot.normalize(1.)
    orbits.integrate(times,pot,method='dop853_c')
    # Integrate as multiple Orbits
    for o in orbits_list:
        o.integrate(times,pot,method='dop853_c')
    # Compare
    for ii in range(len(orbits)):
        assert numpy.amax(numpy.fabs(orbits_list[ii].R(times)-orbits.R(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_list[ii].vR(times)-orbits.vR(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
        assert numpy.amax(numpy.fabs(orbits_list[ii].vT(times)-orbits.vT(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
    return None
    
# Test flippingg an orbit
def setup_orbits_flip(tp,ro,vo,zo,solarmotion,axi=False):
    from galpy.orbit import Orbits
    if isinstance(tp,potential.linearPotential):
        o= Orbits([[1.,1.],[0.2,-0.3]],
                  ro=ro,vo=vo,zo=zo,solarmotion=solarmotion)
    elif isinstance(tp,potential.planarPotential):
        if axi:
            o= Orbits([[1.,1.1,1.1],[1.1,-0.1,0.9]],
                      ro=ro,vo=vo,zo=zo,solarmotion=solarmotion)
        else:
            o= Orbits([[1.,1.1,1.1,0.],[1.1,-1.2,-0.9,2.]],
                      ro=ro,vo=vo,zo=zo,solarmotion=solarmotion)
    else:
        if axi:
            o= Orbits([[1.,1.1,1.1,0.1,0.1],[1.1,-0.7,1.4,-0.1,0.3]],
                      ro=ro,vo=vo,zo=zo,
                      solarmotion=solarmotion)
        else:
            o= Orbits([[1.,1.1,1.1,0.1,0.1,0.],[0.6,-0.4,-1.,-0.3,-0.5,2.]],
                      ro=ro,vo=vo,zo=zo,
                      solarmotion=solarmotion)
    return o
def test_flip():
    from galpy.potential import LogarithmicHaloPotential
    lp= LogarithmicHaloPotential(normalize=1.,q=0.9)
    plp= lp.toPlanar()
    llp= lp.toVertical(1.)
    for ii in range(5):
        #Scales to test that these are properly propagated to the new Orbit
        ro,vo,zo,solarmotion= 10.,300.,0.01,'schoenrich'
        if ii == 0: #axi, full
            o= setup_orbits_flip(lp,ro,vo,zo,solarmotion,axi=True)
        elif ii == 1: #track azimuth, full
            o= setup_orbits_flip(lp,ro,vo,zo,solarmotion,axi=False)
        elif ii == 2: #axi, planar
            o= setup_orbits_flip(plp,ro,vo,zo,solarmotion,axi=True)
        elif ii == 3: #track azimuth, full
            o= setup_orbits_flip(plp,ro,vo,zo,solarmotion,axi=False)
        elif ii == 4: #linear orbit
            o= setup_orbits_flip(llp,ro,vo,None,None,axi=False)
        of= o.flip()
        #First check that the scales have been propagated properly
        assert numpy.fabs(o._ro-of._ro) < 10.**-15., 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        assert numpy.fabs(o._vo-of._vo) < 10.**-15., 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        if ii == 4:
            assert (o._zo is None)*(of._zo is None), 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
            assert (o._solarmotion is None)*(of._solarmotion is None), 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        else:
            assert numpy.fabs(o._zo-of._zo) < 10.**-15., 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
            assert numpy.all(numpy.fabs(o._solarmotion-of._solarmotion) < 10.**-15.), 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        assert o._roSet == of._roSet, 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        assert o._voSet == of._voSet, 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        if ii == 4:
            assert numpy.all(numpy.abs(o.x()-of.x()) < 10.**-10.), 'o.flip() did not work as expected'
            assert numpy.all(numpy.abs(o.vx()+of.vx()) < 10.**-10.), 'o.flip() did not work as expected'
        else:
            assert numpy.all(numpy.abs(o.R()-of.R()) < 10.**-10.), 'o.flip() did not work as expected'
            assert numpy.all(numpy.abs(o.vR()+of.vR()) < 10.**-10.), 'o.flip() did not work as expected'
            assert numpy.all(numpy.abs(o.vT()+of.vT()) < 10.**-10.), 'o.flip() did not work as expected'
        if ii % 2 == 1:
            assert numpy.all(numpy.abs(o.phi()-of.phi()) < 10.**-10.), 'o.flip() did not work as expected'
        if ii < 2:
            assert numpy.all(numpy.abs(o.z()-of.z()) < 10.**-10.), 'o.flip() did not work as expected'
            assert numpy.all(numpy.abs(o.vz()+of.vz()) < 10.**-10.), 'o.flip() did not work as expected'
    return None

# Test flippingg an orbit inplace
def test_flip_inplace():
    from galpy.potential import LogarithmicHaloPotential
    lp= LogarithmicHaloPotential(normalize=1.,q=0.9)
    plp= lp.toPlanar()
    llp= lp.toVertical(1.)
    for ii in range(5):
        #Scales (not really necessary for this test)
        ro,vo,zo,solarmotion= 10.,300.,0.01,'schoenrich'
        if ii == 0: #axi, full
            o= setup_orbits_flip(lp,ro,vo,zo,solarmotion,axi=True)
        elif ii == 1: #track azimuth, full
            o= setup_orbits_flip(lp,ro,vo,zo,solarmotion,axi=False)
        elif ii == 2: #axi, planar
            o= setup_orbits_flip(plp,ro,vo,zo,solarmotion,axi=True)
        elif ii == 3: #track azimuth, full
            o= setup_orbits_flip(plp,ro,vo,zo,solarmotion,axi=False)
        elif ii == 4: #linear orbit
            o= setup_orbits_flip(llp,ro,vo,None,None,axi=False)
        of= o()
        of.flip(inplace=True)
        #First check that the scales have been propagated properly
        assert numpy.fabs(o._ro-of._ro) < 10.**-15., 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        assert numpy.fabs(o._vo-of._vo) < 10.**-15., 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        if ii == 4:
            assert (o._zo is None)*(of._zo is None), 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
            assert (o._solarmotion is None)*(of._solarmotion is None), 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        else:
            assert numpy.fabs(o._zo-of._zo) < 10.**-15., 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
            assert numpy.all(numpy.fabs(o._solarmotion-of._solarmotion) < 10.**-15.), 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        assert o._roSet == of._roSet, 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        assert o._voSet == of._voSet, 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        if ii == 4:
            assert numpy.all(numpy.abs(o.x()-of.x()) < 10.**-10.), 'o.flip() did not work as expected'
            assert numpy.all(numpy.abs(o.vx()+of.vx()) < 10.**-10.), 'o.flip() did not work as expected'
        else:
            assert numpy.all(numpy.abs(o.R()-of.R()) < 10.**-10.), 'o.flip() did not work as expected'
            assert numpy.all(numpy.abs(o.vR()+of.vR()) < 10.**-10.), 'o.flip() did not work as expected'
            assert numpy.all(numpy.abs(o.vT()+of.vT()) < 10.**-10.), 'o.flip() did not work as expected'
        if ii % 2 == 1:
            assert numpy.all(numpy.abs(o.phi()-of.phi()) < 10.**-10.), 'o.flip() did not work as expected'
        if ii < 2:
            assert numpy.all(numpy.abs(o.z()-of.z()) < 10.**-10.), 'o.flip() did not work as expected'
            assert numpy.all(numpy.abs(o.vz()+of.vz()) < 10.**-10.), 'o.flip() did not work as expected'
    return None

# Test flippingg an orbit inplace after orbit integration
def test_flip_inplace_integrated():
    from galpy.potential import LogarithmicHaloPotential
    lp= LogarithmicHaloPotential(normalize=1.,q=0.9)
    plp= lp.toPlanar()
    llp= lp.toVertical(1.)
    ts= numpy.linspace(0.,1.,11)
    for ii in range(5):
        #Scales (not really necessary for this test)
        ro,vo,zo,solarmotion= 10.,300.,0.01,'schoenrich'
        if ii == 0: #axi, full
            o= setup_orbits_flip(lp,ro,vo,zo,solarmotion,axi=True)
        elif ii == 1: #track azimuth, full
            o= setup_orbits_flip(lp,ro,vo,zo,solarmotion,axi=False)
        elif ii == 2: #axi, planar
            o= setup_orbits_flip(plp,ro,vo,zo,solarmotion,axi=True)
        elif ii == 3: #track azimuth, full
            o= setup_orbits_flip(plp,ro,vo,zo,solarmotion,axi=False)
        elif ii == 4: #linear orbit
            o= setup_orbits_flip(llp,ro,vo,None,None,axi=False)
        of= o()
        if ii < 2 or ii == 3:
            o.integrate(ts,lp)
            of.integrate(ts,lp)
        elif ii == 2:
            o.integrate(ts,plp)
            of.integrate(ts,plp)
        else:
            o.integrate(ts,llp)
            of.integrate(ts,llp)
        of.flip(inplace=True)
        # Just check one time, allows code duplication!
        o= o(0.5)
        of= of(0.5)
        #First check that the scales have been propagated properly
        assert numpy.fabs(o._ro-of._ro) < 10.**-15., 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        assert numpy.fabs(o._vo-of._vo) < 10.**-15., 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        if ii == 4:
            assert (o._zo is None)*(of._zo is None), 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
            assert (o._solarmotion is None)*(of._solarmotion is None), 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        else:
            assert numpy.fabs(o._zo-of._zo) < 10.**-15., 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
            assert numpy.all(numpy.fabs(o._solarmotion-of._solarmotion) < 10.**-15.), 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        assert o._roSet == of._roSet, 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        assert o._voSet == of._voSet, 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        if ii == 4:
            assert numpy.all(numpy.abs(o.x()-of.x()) < 10.**-10.), 'o.flip() did not work as expected'
            assert numpy.all(numpy.abs(o.vx()+of.vx()) < 10.**-10.), 'o.flip() did not work as expected'
        else:
            assert numpy.all(numpy.abs(o.R()-of.R()) < 10.**-10.), 'o.flip() did not work as expected'
            assert numpy.all(numpy.abs(o.vR()+of.vR()) < 10.**-10.), 'o.flip() did not work as expected'
            assert numpy.all(numpy.abs(o.vT()+of.vT()) < 10.**-10.), 'o.flip() did not work as expected'
        if ii % 2 == 1:
            assert numpy.all(numpy.abs(o.phi()-of.phi()) < 10.**-10.), 'o.flip() did not work as expected'
        if ii < 2:
            assert numpy.all(numpy.abs(o.z()-of.z()) < 10.**-10.), 'o.flip() did not work as expected'
            assert numpy.all(numpy.abs(o.vz()+of.vz()) < 10.**-10.), 'o.flip() did not work as expected'
    return None

# Test flippingg an orbit inplace after orbit integration, and after having
# once evaluated the orbit before flipping inplace (#345)
# only difference wrt previous test is a line that evaluates of before
# flipping
def test_flip_inplace_integrated_evaluated():
    from galpy.potential import LogarithmicHaloPotential
    lp= LogarithmicHaloPotential(normalize=1.,q=0.9)
    plp= lp.toPlanar()
    llp= lp.toVertical(1.)
    ts= numpy.linspace(0.,1.,11)
    for ii in range(5):
        #Scales (not really necessary for this test)
        ro,vo,zo,solarmotion= 10.,300.,0.01,'schoenrich'
        if ii == 0: #axi, full
            o= setup_orbits_flip(lp,ro,vo,zo,solarmotion,axi=True)
        elif ii == 1: #track azimuth, full
            o= setup_orbits_flip(lp,ro,vo,zo,solarmotion,axi=False)
        elif ii == 2: #axi, planar
            o= setup_orbits_flip(plp,ro,vo,zo,solarmotion,axi=True)
        elif ii == 3: #track azimuth, full
            o= setup_orbits_flip(plp,ro,vo,zo,solarmotion,axi=False)
        elif ii == 4: #linear orbit
            o= setup_orbits_flip(llp,ro,vo,None,None,axi=False)
        of= o()
        if ii < 2 or ii == 3:
            o.integrate(ts,lp)
            of.integrate(ts,lp)
        elif ii == 2:
            o.integrate(ts,plp)
            of.integrate(ts,plp)
        else:
            o.integrate(ts,llp)
            of.integrate(ts,llp)
        # Evaluate, make sure it is at an interpolated time!
        dum= of.R(0.52)
        # Now flip
        of.flip(inplace=True)
        # Just check one time, allows code duplication!
        o= o(0.52)
        of= of(0.52)
        #First check that the scales have been propagated properly
        assert numpy.fabs(o._ro-of._ro) < 10.**-15., 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        assert numpy.fabs(o._vo-of._vo) < 10.**-15., 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        if ii == 4:
            assert (o._zo is None)*(of._zo is None), 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
            assert (o._solarmotion is None)*(of._solarmotion is None), 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        else:
            assert numpy.fabs(o._zo-of._zo) < 10.**-15., 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
            assert numpy.all(numpy.fabs(o._solarmotion-of._solarmotion) < 10.**-15.), 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        assert o._roSet == of._roSet, 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        assert o._voSet == of._voSet, 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        if ii == 4:
            assert numpy.all(numpy.abs(o.x()-of.x()) < 10.**-10.), 'o.flip() did not work as expected'
            assert numpy.all(numpy.abs(o.vx()+of.vx()) < 10.**-10.), 'o.flip() did not work as expected'
        else:
            assert numpy.all(numpy.abs(o.R()-of.R()) < 10.**-10.), 'o.flip() did not work as expected'
            assert numpy.all(numpy.abs(o.vR()+of.vR()) < 10.**-10.), 'o.flip() did not work as expected'
            assert numpy.all(numpy.abs(o.vT()+of.vT()) < 10.**-10.), 'o.flip() did not work as expected'
        if ii % 2 == 1:
            assert numpy.all(numpy.abs(o.phi()-of.phi()) < 10.**-10.), 'o.flip() did not work as expected'
        if ii < 2:
            assert numpy.all(numpy.abs(o.z()-of.z()) < 10.**-10.), 'o.flip() did not work as expected'
            assert numpy.all(numpy.abs(o.vz()+of.vz()) < 10.**-10.), 'o.flip() did not work as expected'
    return None

# test getOrbit
def test_getOrbit():
    from galpy.orbit import Orbits
    from galpy.potential import LogarithmicHaloPotential
    lp= LogarithmicHaloPotential(normalize=1.,q=0.9)
    o= Orbits([[1.,0.1,1.2,0.3,0.2,2.],
               [1.,-0.1,1.1,-0.3,0.2,5.]])
    times= numpy.linspace(0.,7.,251)
    o.integrate(times,lp)
    Rs= o.R(times)
    vRs= o.vR(times)
    vTs= o.vT(times)
    zs= o.z(times)
    vzs= o.vz(times)
    phis= o.phi(times)
    orbarray= o.getOrbit()
    assert numpy.all(numpy.fabs(Rs-orbarray[...,0])) < 10.**-16., \
        'getOrbit does not work as expected for R'
    assert numpy.all(numpy.fabs(vRs-orbarray[...,1])) < 10.**-16., \
        'getOrbit does not work as expected for vR'
    assert numpy.all(numpy.fabs(vTs-orbarray[...,2])) < 10.**-16., \
        'getOrbit does not work as expected for vT'
    assert numpy.all(numpy.fabs(zs-orbarray[...,3])) < 10.**-16., \
        'getOrbit does not work as expected for z'
    assert numpy.all(numpy.fabs(vzs-orbarray[...,4])) < 10.**-16., \
        'getOrbit does not work as expected for vz'
    assert numpy.all(numpy.fabs(phis-orbarray[...,5])) < 10.**-16., \
        'getOrbit does not work as expected for phi'
    return None

# Test that the eccentricity, zmax, rperi, and rap calculated numerically by
# Orbits agrees with that calculated numerically using Orbit
def test_EccZmaxRperiRap_num_againstorbit_3d():
    from galpy.orbit import Orbit, Orbits
    from galpy.potential import MWPotential2014
    numpy.random.seed(1)
    nrand= 10
    Rs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)+1.
    vRs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)
    vTs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)+1.
    zs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)
    vzs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)
    phis= 2.*numpy.pi*(2.*numpy.random.uniform(size=nrand)-1.)
    os= Orbits(list(zip(Rs,vRs,vTs,zs,vzs,phis)))
    list_os= [Orbit([R,vR,vT,z,vz,phi])
              for R,vR,vT,z,vz,phi in zip(Rs,vRs,vTs,zs,vzs,phis)]
    # First test AttributeError when not integrated
    with pytest.raises(AttributeError):
        os.e()
    with pytest.raises(AttributeError):
        os.zmax()
    with pytest.raises(AttributeError):
        os.rperi()
    with pytest.raises(AttributeError):
        os.rap()
    # Integrate all
    times= numpy.linspace(0.,10.,1001)
    os.integrate(times,MWPotential2014)
    [o.integrate(times,MWPotential2014) for o in list_os]
    for ii in range(nrand):
        assert numpy.all(numpy.fabs(os.e()[ii]-list_os[ii].e()) < 1e-10), 'Evaluating Orbits e does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.zmax()[ii]-list_os[ii].zmax()) < 1e-10), 'Evaluating Orbits zmax does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.rperi()[ii]-list_os[ii].rperi()) < 1e-10), 'Evaluating Orbits rperi does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.rap()[ii]-list_os[ii].rap()) < 1e-10), 'Evaluating Orbits rap does not agree with Orbit'
    return None

def test_EccZmaxRperiRap_num_againstorbit_2d():
    from galpy.orbit import Orbit, Orbits
    from galpy.potential import MWPotential2014
    numpy.random.seed(1)
    nrand= 10
    Rs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)+1.
    vRs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)
    vTs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)+1.
    phis= 2.*numpy.pi*(2.*numpy.random.uniform(size=nrand)-1.)
    os= Orbits(list(zip(Rs,vRs,vTs,phis)))
    list_os= [Orbit([R,vR,vT,phi])
              for R,vR,vT,phi in zip(Rs,vRs,vTs,phis)]
    # Integrate all
    times= numpy.linspace(0.,10.,1001)
    os.integrate(times,MWPotential2014)
    [o.integrate(times,MWPotential2014) for o in list_os]
    for ii in range(nrand):
        assert numpy.all(numpy.fabs(os.e()[ii]-list_os[ii].e()) < 1e-10), 'Evaluating Orbits e does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.rperi()[ii]-list_os[ii].rperi()) < 1e-10), 'Evaluating Orbits rperi does not agree with Orbit'
        assert numpy.all(numpy.fabs(os.rap()[ii]-list_os[ii].rap()) < 1e-10), 'Evaluating Orbits rap does not agree with Orbit'
    return None

# Test that the eccentricity, zmax, rperi, and rap calculated analytically by
# Orbits agrees with that calculated analytically using Orbit
def test_EccZmaxRperiRap_analytic_againstorbit_3d():
    from galpy.orbit import Orbit, Orbits
    from galpy.potential import MWPotential2014
    numpy.random.seed(1)
    nrand= 10
    Rs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)+1.
    vRs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)
    vTs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)+1.
    zs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)
    vzs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)
    phis= 2.*numpy.pi*(2.*numpy.random.uniform(size=nrand)-1.)
    os= Orbits(list(zip(Rs,vRs,vTs,zs,vzs,phis)))
    list_os= [Orbit([R,vR,vT,z,vz,phi])
              for R,vR,vT,z,vz,phi in zip(Rs,vRs,vTs,zs,vzs,phis)]
    # First test AttributeError when no potential and not integrated
    with pytest.raises(AttributeError):
        os.e(analytic=True)
    with pytest.raises(AttributeError):
        os.zmax(analytic=True)
    with pytest.raises(AttributeError):
        os.rperi(analytic=True)
    with pytest.raises(AttributeError):
        os.rap(analytic=True)
    for type in ['spherical','staeckel','adiabatic']:
        for ii in range(nrand):
            assert numpy.all(numpy.fabs(os.e(pot=MWPotential2014,analytic=True,type=type)[ii]-list_os[ii].e(pot=MWPotential2014,analytic=True,type=type)) < 1e-10), 'Evaluating Orbits e analytically does not agree with Orbit for type={}'.format(type)
        assert numpy.all(numpy.fabs(os.zmax(pot=MWPotential2014,analytic=True,type=type)[ii]-list_os[ii].zmax(pot=MWPotential2014,analytic=True,type=type)) < 1e-10), 'Evaluating Orbits zmax analytically does not agree with Orbit for type={}'.format(type)
        assert numpy.all(numpy.fabs(os.rperi(pot=MWPotential2014,analytic=True,type=type)[ii]-list_os[ii].rperi(pot=MWPotential2014,analytic=True,type=type)) < 1e-10), 'Evaluating Orbits rperi analytically does not agree with Orbit for type={}'.format(type)
        assert numpy.all(numpy.fabs(os.rap(pot=MWPotential2014,analytic=True,type=type)[ii]-list_os[ii].rap(pot=MWPotential2014,analytic=True,type=type)) < 1e-10), 'Evaluating Orbits rap analytically does not agree with Orbit for type={}'.format(type)
    return None

def test_EccZmaxRperiRap_analytic_againstorbit_2d():
    from galpy.orbit import Orbit, Orbits
    from galpy.potential import MWPotential2014
    numpy.random.seed(1)
    nrand= 10
    Rs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)+1.
    vRs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)
    vTs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)+1.
    phis= 2.*numpy.pi*(2.*numpy.random.uniform(size=nrand)-1.)
    os= Orbits(list(zip(Rs,vRs,vTs,phis)))
    list_os= [Orbit([R,vR,vT,phi])
              for R,vR,vT,phi in zip(Rs,vRs,vTs,phis)]
    # No matter the type, should always be using adiabtic, not specified in 
    # Orbit
    for type in ['spherical','staeckel','adiabatic']:
        for ii in range(nrand):
            assert numpy.all(numpy.fabs(os.e(pot=MWPotential2014,analytic=True,type=type)[ii]-list_os[ii].e(pot=MWPotential2014,analytic=True)) < 1e-10), 'Evaluating Orbits e analytically does not agree with Orbit for type={}'.format(type)
        assert numpy.all(numpy.fabs(os.rperi(pot=MWPotential2014,analytic=True,type=type)[ii]-list_os[ii].rperi(pot=MWPotential2014,analytic=True,type=type)) < 1e-10), 'Evaluating Orbits rperi analytically does not agree with Orbit for type={}'.format(type)
        assert numpy.all(numpy.fabs(os.rap(pot=MWPotential2014,analytic=True,type=type)[ii]-list_os[ii].rap(pot=MWPotential2014,analytic=True)) < 1e-10), 'Evaluating Orbits rap analytically does not agree with Orbit for type={}'.format(type)
    return None

def test_rguiding():
    from galpy.orbit import Orbit, Orbits
    from galpy.potential import MWPotential2014
    numpy.random.seed(1)
    nrand= 10
    Rs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)+1.
    vRs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)
    vTs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)+1.
    zs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)
    vzs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)
    phis= 2.*numpy.pi*(2.*numpy.random.uniform(size=nrand)-1.)
    os= Orbits(list(zip(Rs,vRs,vTs,zs,vzs,phis)))
    list_os= [Orbit([R,vR,vT,z,vz,phi])
              for R,vR,vT,z,vz,phi in zip(Rs,vRs,vTs,zs,vzs,phis)]
    # First test that if potential is not given, error is raised
    with pytest.raises(RuntimeError):
        os.rguiding()
    # With small number, calculation is direct
    for ii in range(nrand):
        assert numpy.all(numpy.fabs(os.rguiding(pot=MWPotential2014)[ii]/list_os[ii].rguiding(pot=MWPotential2014)-1.) < 10.**-10.), 'Evaluating Orbits rguiding analytically does not agree with Orbit'
    # With large number, calculation is interpolated
    nrand= 1002
    Rs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)+1.
    vRs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)
    vTs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)+1.
    zs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)
    vzs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)
    phis= 2.*numpy.pi*(2.*numpy.random.uniform(size=nrand)-1.)
    os= Orbits(list(zip(Rs,vRs,vTs,zs,vzs,phis)))
    list_os= [Orbit([R,vR,vT,z,vz,phi])
              for R,vR,vT,z,vz,phi in zip(Rs,vRs,vTs,zs,vzs,phis)]
    rgs= os.rguiding(pot=MWPotential2014)
    for ii in range(nrand):
        assert numpy.all(numpy.fabs(rgs[ii]/list_os[ii].rguiding(pot=MWPotential2014)-1.) < 10.**-10.), 'Evaluating Orbits rguiding analytically does not agree with Orbit'
    return None

# Test that the actions, frequencies/periods, and angles calculated 
# analytically by Orbits agrees with that calculated analytically using Orbit
def test_actionsFreqsAngles_againstorbit_3d():
    from galpy.orbit import Orbit, Orbits
    from galpy.potential import MWPotential2014
    numpy.random.seed(1)
    nrand= 10
    Rs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)+1.
    vRs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)
    vTs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)+1.
    zs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)
    vzs= 0.2*(2.*numpy.random.uniform(size=nrand)-1.)
    phis= 2.*numpy.pi*(2.*numpy.random.uniform(size=nrand)-1.)
    os= Orbits(list(zip(Rs,vRs,vTs,zs,vzs,phis)))
    list_os= [Orbit([R,vR,vT,z,vz,phi])
              for R,vR,vT,z,vz,phi in zip(Rs,vRs,vTs,zs,vzs,phis)]
    # First test AttributeError when no potential and not integrated
    with pytest.raises(AttributeError):
        os.jr(analytic=True)
    with pytest.raises(AttributeError):
        os.jp(analytic=True)
    with pytest.raises(AttributeError):
        os.jz(analytic=True)
    with pytest.raises(AttributeError):
        os.wr(analytic=True)
    with pytest.raises(AttributeError):
        os.wp(analytic=True)
    with pytest.raises(AttributeError):
        os.wz(analytic=True)
    with pytest.raises(AttributeError):
        os.Or(analytic=True)
    with pytest.raises(AttributeError):
        os.Op(analytic=True)
    with pytest.raises(AttributeError):
        os.Oz(analytic=True)
    with pytest.raises(AttributeError):
        os.Tr(analytic=True)
    with pytest.raises(AttributeError):
        os.Tp(analytic=True)
    with pytest.raises(AttributeError):
        os.TrTp(analytic=True)
    with pytest.raises(AttributeError):
        os.Tz(analytic=True)
    # Tolerance for jr, jp, jz, diff. for isochroneApprox, because currently
    # not implemented in exactly the same way in Orbit and Orbits (Orbit uses
    # __call__ for the actions, Orbits uses actionsFreqsAngles, which is diff.)
    tol= {}
    tol['spherical']= -12.
    tol['staeckel']=-12.
    tol['adiabatic']= -12.
    tol['isochroneApprox']= -2.
    # For now we skip adiabatic here, because frequencies and angles not 
    # implemented yet
#    for type in ['spherical','staeckel','adiabatic']:
    for type in ['spherical','staeckel','isochroneApprox']:
        for ii in range(nrand):
            assert numpy.all(numpy.fabs(os.jr(pot=MWPotential2014,analytic=True,type=type,b=0.8)[ii]/list_os[ii].jr(pot=MWPotential2014,analytic=True,type=type,b=0.8)-1.) < 10.**tol[type]), 'Evaluating Orbits jr analytically does not agree with Orbit for type={}'.format(type)
            assert numpy.all(numpy.fabs(os.jp(pot=MWPotential2014,analytic=True,type=type,b=0.8)[ii]/list_os[ii].jp(pot=MWPotential2014,analytic=True,type=type,b=0.8)-1.) < 10.**tol[type]), 'Evaluating Orbits jp analytically does not agree with Orbit for type={}'.format(type)
            assert numpy.all(numpy.fabs(os.jz(pot=MWPotential2014,analytic=True,type=type,b=0.8)[ii]/list_os[ii].jz(pot=MWPotential2014,analytic=True,type=type,b=0.8)-1.) < 10.**tol[type]), 'Evaluating Orbits jz analytically does not agree with Orbit for type={}'.format(type)
            assert numpy.all(numpy.fabs(os.wr(pot=MWPotential2014,analytic=True,type=type,b=0.8)[ii]/list_os[ii].wr(pot=MWPotential2014,analytic=True,type=type,b=0.8)-1.) < 1e-10), 'Evaluating Orbits wr analytically does not agree with Orbit for type={}'.format(type)
            assert numpy.all(numpy.fabs(os.wp(pot=MWPotential2014,analytic=True,type=type,b=0.8)[ii]/list_os[ii].wp(pot=MWPotential2014,analytic=True,type=type,b=0.8)-1.) < 1e-10), 'Evaluating Orbits wp analytically does not agree with Orbit for type={}'.format(type)
            assert numpy.all(numpy.fabs(os.wz(pot=MWPotential2014,analytic=True,type=type,b=0.8)[ii]/list_os[ii].wz(pot=MWPotential2014,analytic=True,type=type,b=0.8)-1.) < 1e-10), 'Evaluating Orbits wz analytically does not agree with Orbit for type={}'.format(type)
            assert numpy.all(numpy.fabs(os.Or(pot=MWPotential2014,analytic=True,type=type,b=0.8)[ii]/list_os[ii].Or(pot=MWPotential2014,analytic=True,type=type,b=0.8)-1.) < 1e-10), 'Evaluating Orbits Or analytically does not agree with Orbit for type={}'.format(type)
            assert numpy.all(numpy.fabs(os.Op(pot=MWPotential2014,analytic=True,type=type,b=0.8)[ii]/list_os[ii].Op(pot=MWPotential2014,analytic=True,type=type,b=0.8)-1.) < 1e-10), 'Evaluating Orbits Op analytically does not agree with Orbit for type={}'.format(type)
            assert numpy.all(numpy.fabs(os.Oz(pot=MWPotential2014,analytic=True,type=type,b=0.8)[ii]/list_os[ii].Oz(pot=MWPotential2014,analytic=True,type=type,b=0.8)-1.) < 1e-10), 'Evaluating Orbits Oz analytically does not agree with Orbit for type={}'.format(type)
            assert numpy.all(numpy.fabs(os.Tr(pot=MWPotential2014,analytic=True,type=type,b=0.8)[ii]/list_os[ii].Tr(pot=MWPotential2014,analytic=True,type=type,b=0.8)-1.) < 1e-10), 'Evaluating Orbits Tr analytically does not agree with Orbit for type={}'.format(type)
            assert numpy.all(numpy.fabs(os.Tp(pot=MWPotential2014,analytic=True,type=type,b=0.8)[ii]/list_os[ii].Tp(pot=MWPotential2014,analytic=True,type=type,b=0.8)-1.) < 1e-10), 'Evaluating Orbits Tp analytically does not agree with Orbit for type={}'.format(type)
            assert numpy.all(numpy.fabs(os.TrTp(pot=MWPotential2014,analytic=True,type=type,b=0.8)[ii]/list_os[ii].TrTp(pot=MWPotential2014,analytic=True,type=type,b=0.8)-1.) < 1e-10), 'Evaluating Orbits TrTp analytically does not agree with Orbit for type={}'.format(type)
            assert numpy.all(numpy.fabs(os.Tz(pot=MWPotential2014,analytic=True,type=type,b=0.8)[ii]/list_os[ii].Tz(pot=MWPotential2014,analytic=True,type=type,b=0.8)-1.) < 1e-10), 'Evaluating Orbits Tz analytically does not agree with Orbit for type={}'.format(type)
            if type == 'isochroneApprox': break # otherwise takes too long
    return None

# Test that the delta parameter is properly dealt with when using the staeckel
# approximation: when it changes, need to re-do the aA calcs.
def test_actionsFreqsAngles_staeckeldelta():
    from galpy.potential import MWPotential2014
    from galpy.orbit import Orbits
    os= Orbits([None,None]) # Just twice the Sun!
    # First with delta
    jr= os.jr(delta=0.4,pot=MWPotential2014)
    # Now without, should be different
    jrn= os.jr(pot=MWPotential2014)
    assert numpy.all(numpy.fabs(jr-jrn) > 1e-4), 'Action calculation in Orbits using Staeckel approximation not updated when going from specifying delta to not specifying it'
    # Again, now the other way around
    os= Orbits([None,None]) # Just twice the Sun!
    # First without delta
    jrn= os.jr(pot=MWPotential2014)
    # Now with, should be different
    jr= os.jr(delta=0.4,pot=MWPotential2014)
    assert numpy.all(numpy.fabs(jr-jrn) > 1e-4), 'Action calculation in Orbits using Staeckel approximation not updated when going from specifying delta to not specifying it'
    return None

# Test that the b / ip parameters are properly dealt with when using the 
# isochroneapprox approximation: when they change, need to re-do the aA calcs.
def test_actionsFreqsAngles_isochroneapproxb():
    from galpy.potential import MWPotential2014, IsochronePotential
    from galpy.orbit import Orbits
    os= Orbits([None,None]) # Just twice the Sun!
    # First with one b
    jr= os.jr(type='isochroneapprox',b=0.8,pot=MWPotential2014)
    # Now with another b, should be different
    jrn= os.jr(type='isochroneapprox',b=1.8,pot=MWPotential2014)
    assert numpy.all(numpy.fabs(jr-jrn) > 1e-4), 'Action calculation in Orbits using isochroneapprox approximation not updated when going from specifying b to not specifying it'
    # Again, now specifying ip
    os= Orbits([None,None]) # Just twice the Sun!
    # First with one
    jrn= os.jr(pot=MWPotential2014,type='isochroneapprox',
               ip=IsochronePotential(normalize=1.1,b=0.8))
    # Now with another one, should be different
    jr= os.jr(pot=MWPotential2014,type='isochroneapprox',
               ip=IsochronePotential(normalize=0.99,b=1.8))
    assert numpy.all(numpy.fabs(jr-jrn) > 1e-4), 'Action calculation in Orbits using isochroneapprox approximation not updated when going from specifying delta to not specifying it'
    return None

def test_actionsFreqsAngles_RuntimeError_1d():
    from galpy.orbit import Orbits
    os= Orbits([[1.,0.1],[0.2,0.3]])
    with pytest.raises(RuntimeError):
        os.jz(analytic=True)
    return None

@pytest.mark.xfail(sys.platform != 'win32',strict=True,raises=ValueError,
                   reason="Does not fail on Windows...")
def test_ChandrasekharDynamicalFrictionForce_constLambda():
    # Test from test_potential for Orbits now! Currently fails because Chandra
    # can't be pickled for parallel_map...
    #
    # Test that the ChandrasekharDynamicalFrictionForce with constant Lambda
    # agrees with analytical solutions for circular orbits:
    # assuming that a mass remains on a circular orbit in an isothermal halo 
    # with velocity dispersion sigma and for constant Lambda:
    # r_final^2 - r_initial^2 = -0.604 ln(Lambda) GM/sigma t 
    # (e.g., B&T08, p. 648)
    from galpy.util import bovy_conversion
    from galpy.orbit import Orbit, Orbits
    ro,vo= 8.,220.
    # Parameters
    GMs= 10.**9./bovy_conversion.mass_in_msol(vo,ro)
    const_lnLambda= 7.
    r_inits= [2.,2.5]
    dt= 2./bovy_conversion.time_in_Gyr(vo,ro)
    # Compute
    lp= potential.LogarithmicHaloPotential(normalize=1.,q=1.)
    cdfc= potential.ChandrasekharDynamicalFrictionForce(\
        GMs=GMs,const_lnLambda=const_lnLambda,
        dens=lp) # don't provide sigmar, so it gets computed using galpy.df.jeans
    o= Orbits([Orbit([r_inits[0],0.,1.,0.,0.,0.]),
               Orbit([r_inits[1],0.,1.,0.,0.,0.])])
    ts= numpy.linspace(0.,dt,1001)
    o.integrate(ts,[lp,cdfc],method='odeint')
    r_pred= numpy.sqrt(numpy.array(o.r())**2.
                       -0.604*const_lnLambda*GMs*numpy.sqrt(2.)*dt)
    assert numpy.all(numpy.fabs(r_pred-numpy.array(o.r(ts[-1]))) < 0.015), 'ChandrasekharDynamicalFrictionForce with constant lnLambda for circular orbits does not agree with analytical prediction'
    return None

# Check that toPlanar works
def test_toPlanar():
    from galpy.orbit import Orbits
    obs= Orbits([[1.,0.1,1.1,0.3,0.,2.],
                [1.,-0.2,1.3,-0.3,0.,5.]])
    obsp= obs.toPlanar()
    assert obsp.dim() == 2, 'toPlanar does not generate an Orbit w/ dim=2 for FullOrbit'
    assert numpy.all(obsp.R() == obs.R()), 'Planar orbit generated w/ toPlanar does not have the correct R'
    assert numpy.all(obsp.vR() == obs.vR()), 'Planar orbit generated w/ toPlanar does not have the correct vR'
    assert numpy.all(obsp.vT() == obs.vT()), 'Planar orbit generated w/ toPlanar does not have the correct vT'
    assert numpy.all(obsp.phi() == obs.phi()), 'Planar orbit generated w/ toPlanar does not have the correct phi'
    obs= Orbits([[1.,0.1,1.1,0.3,0.],
                [1.,-0.2,1.3,-0.3,0.]])
    obsp= obs.toPlanar()
    assert obsp.dim() == 2, 'toPlanar does not generate an Orbit w/ dim=2 for RZOrbit'
    assert numpy.all(obsp.R() == obs.R()), 'Planar orbit generated w/ toPlanar does not have the correct R'
    assert numpy.all(obsp.vR() == obs.vR()), 'Planar orbit generated w/ toPlanar does not have the correct vR'
    assert numpy.all(obsp.vT() == obs.vT()), 'Planar orbit generated w/ toPlanar does not have the correct vT'
    ro,vo,zo,solarmotion= 10.,300.,0.01,'schoenrich'
    obs= Orbits([[1.,0.1,1.1,0.3,0.,2.],
                [1.,-0.2,1.3,-0.3,0.,5.]],
                ro=ro,vo=vo,zo=zo,solarmotion=solarmotion)
    obsp= obs.toPlanar()
    assert obsp.dim() == 2, 'toPlanar does not generate an Orbit w/ dim=2 for RZOrbit'
    assert numpy.all(obsp.R() == obs.R()), 'Planar orbit generated w/ toPlanar does not have the correct R'
    assert numpy.all(obsp.vR() == obs.vR()), 'Planar orbit generated w/ toPlanar does not have the correct vR'
    assert numpy.all(obsp.vT() == obs.vT()), 'Planar orbit generated w/ toPlanar does not have the correct vT'
    assert numpy.fabs(obs._ro-obsp._ro) < 10.**-15., 'Planar orbit generated w/ toPlanar does not have the proper physical scale and coordinate-transformation parameters associated with it'
    assert numpy.fabs(obs._vo-obsp._vo) < 10.**-15., 'Planar orbit generated w/ toPlanar does not have the proper physical scale and coordinate-transformation parameters associated with it'
    assert numpy.fabs(obs._zo-obsp._zo) < 10.**-15., 'Planar orbit generated w/ toPlanar does not have the proper physical scale and coordinate-transformation parameters associated with it'
    assert numpy.all(numpy.fabs(obs._solarmotion-obsp._solarmotion) < 10.**-15.), 'Planar orbit generated w/ toPlanar does not have the proper physical scale and coordinate-transformation parameters associated with it'
    assert obs._roSet == obsp._roSet, 'Planar orbit generated w/ toPlanar does not have the proper physical scale and coordinate-transformation parameters associated with it'
    assert obs._voSet == obsp._voSet, 'Planar orbit generated w/ toPlanar does not have the proper physical scale and coordinate-transformation parameters associated with it'
    obs= Orbits([[1.,0.1,1.1,0.3],
                 [1.,-0.2,1.3,-0.3]])
    try:
        obs.toPlanar()
    except AttributeError:
        pass
    else:
        raise AttributeError('toPlanar() applied to a planar Orbit did not raise an AttributeError')
    return None

# Check that toLinear works
def test_toLinear():
    from galpy.orbit import Orbits
    obs= Orbits([[1.,0.1,1.1,0.3,0.,2.],
                 [1.,-0.2,1.3,-0.3,0.,5.]])
    obsl= obs.toLinear()
    assert obsl.dim() == 1, 'toLinear does not generate an Orbit w/ dim=1 for FullOrbit'
    assert numpy.all(obsl.x() == obs.z()), 'Linear orbit generated w/ toLinear does not have the correct z'
    assert numpy.all(obsl.vx() == obs.vz()), 'Linear orbit generated w/ toLinear does not have the correct vx'
    obs= Orbits([[1.,0.1,1.1,0.3,0.],
                 [1.,-0.2,1.3,-0.3,0.]])
    obsl= obs.toLinear()
    assert obsl.dim() == 1, 'toLinear does not generate an Orbit w/ dim=1 for FullOrbit'
    assert numpy.all(obsl.x() == obs.z()), 'Linear orbit generated w/ toLinear does not have the correct z'
    assert numpy.all(obsl.vx() == obs.vz()), 'Linear orbit generated w/ toLinear does not have the correct vx'
    obs= Orbits([[1.,0.1,1.1,0.3],
                 [1.,-0.2,1.3,-0.3]])
    try:
        obs.toLinear()
    except AttributeError:
        pass
    else:
        raise AttributeError('toLinear() applied to a planar Orbit did not raise an AttributeError')
    # w/ scales
    ro,vo= 10.,300.
    obs= Orbits([[1.,0.1,1.1,0.3,0.,2.],
                 [1.,-0.2,1.3,-0.3,0.,5.]],ro=ro,vo=vo)
    obsl= obs.toLinear()
    assert obsl.dim() == 1, 'toLinwar does not generate an Orbit w/ dim=1 for FullOrbit'
    assert numpy.all(obsl.x() == obs.z()), 'Linear orbit generated w/ toLinear does not have the correct z'
    assert numpy.all(obsl.vx() == obs.vz()), 'Linear orbit generated w/ toLinear does not have the correct vx'
    assert numpy.fabs(obs._ro-obsl._ro) < 10.**-15., 'Linear orbit generated w/ toLinear does not have the proper physical scale and coordinate-transformation parameters associated with it'
    assert numpy.fabs(obs._vo-obsl._vo) < 10.**-15., 'Linear orbit generated w/ toLinear does not have the proper physical scale and coordinate-transformation parameters associated with it'
    assert (obsl._zo is None), 'Linear orbit generated w/ toLinear does not have the proper physical scale and coordinate-transformation parameters associated with it'
    assert (obsl._solarmotion is None), 'Linear orbit generated w/ toLinear does not have the proper physical scale and coordinate-transformation parameters associated with it'
    assert obs._roSet == obsl._roSet, 'Linear orbit generated w/ toLinear does not have the proper physical scale and coordinate-transformation parameters associated with it'
    assert obs._voSet == obsl._voSet, 'Linear orbit generated w/ toLinear does not have the proper physical scale and coordinate-transformation parameters associated with it'
    return None

# Check that the routines that should return physical coordinates are turned off by turn_physical_off
def test_physical_output_off():
    from galpy.orbit import Orbits
    from galpy.potential import LogarithmicHaloPotential
    lp= LogarithmicHaloPotential(normalize=1.)
    o= Orbits()
    ro= o._ro
    vo= o._vo
    #turn off
    o.turn_physical_off()
    #Test positions
    assert numpy.fabs(o.R()-o.R(use_physical=False)) < 10.**-10., 'o.R() output for Orbit setup with ro= does not work as expected when turned off'
    assert numpy.fabs(o.x()-o.x(use_physical=False)) < 10.**-10., 'o.x() output for Orbit setup with ro= does not work as expected when turned off'
    assert numpy.fabs(o.y()-o.y(use_physical=False)) < 10.**-10., 'o.y() output for Orbit setup with ro= does not work as expected when turned off'
    assert numpy.fabs(o.z()-o.z(use_physical=False)) < 10.**-10., 'o.z() output for Orbit setup with ro= does not work as expected when turned off'
    assert numpy.fabs(o.r()-o.r(use_physical=False)) < 10.**-10., 'o.r() output for Orbit setup with ro= does not work as expected when turned off'
    #Test velocities
    assert numpy.fabs(o.vR()-o.vR(use_physical=False)) < 10.**-10., 'o.vR() output for Orbit setup with vo= does not work as expected when turned off'
    assert numpy.fabs(o.vT()-o.vT(use_physical=False)) < 10.**-10., 'o.vT() output for Orbit setup with vo= does not work as expected'
    assert numpy.fabs(o.vphi()-o.vphi(use_physical=False)) < 10.**-10., 'o.vphi() output for Orbit setup with vo= does not work as expected when turned off'
    assert numpy.fabs(o.vx()-o.vx(use_physical=False)) < 10.**-10., 'o.vx() output for Orbit setup with vo= does not work as expected when turned off'
    assert numpy.fabs(o.vy()-o.vy(use_physical=False)) < 10.**-10., 'o.vy() output for Orbit setup with vo= does not work as expected when turned off'
    assert numpy.fabs(o.vz()-o.vz(use_physical=False)) < 10.**-10., 'o.vz() output for Orbit setup with vo= does not work as expected when turned off'
    #Test energies
    assert numpy.fabs(o.E(pot=lp)-o.E(pot=lp,use_physical=False)) < 10.**-10., 'o.E() output for Orbit setup with vo= does not work as expected when turned off'
    assert numpy.fabs(o.Jacobi(pot=lp)-o.Jacobi(pot=lp,use_physical=False)) < 10.**-10., 'o.E() output for Orbit setup with vo= does not work as expected when turned off'
    assert numpy.fabs(o.ER(pot=lp)-o.ER(pot=lp,use_physical=False)) < 10.**-10., 'o.ER() output for Orbit setup with vo= does not work as expected when turned off'
    assert numpy.fabs(o.Ez(pot=lp)-o.Ez(pot=lp,use_physical=False)) < 10.**-10., 'o.Ez() output for Orbit setup with vo= does not work as expected when turned off'
    #Test angular momentun
    assert numpy.all(numpy.fabs(o.L()-o.L(use_physical=False)) < 10.**-10.), 'o.L() output for Orbit setup with ro=,vo= does not work as expected when turned off'
    # Test action-angle functions
    assert numpy.fabs(o.jr(pot=lp,type='staeckel',delta=0.5)-o.jr(pot=lp,type='staeckel',delta=0.5,use_physical=False)) < 10.**-10., 'o.jr() output for Orbit setup with ro=,vo= does not work as expected'
    assert numpy.fabs(o.jp(pot=lp,type='staeckel',delta=0.5)-o.jp(pot=lp,type='staeckel',delta=0.5,use_physical=False)) < 10.**-10., 'o.jp() output for Orbit setup with ro=,vo= does not work as expected'
    assert numpy.fabs(o.jz(pot=lp,type='staeckel',delta=0.5)-o.jz(pot=lp,type='staeckel',delta=0.5,use_physical=False)) < 10.**-10., 'o.jz() output for Orbit setup with ro=,vo= does not work as expected'
    assert numpy.fabs(o.Tr(pot=lp,type='staeckel',delta=0.5)-o.Tr(pot=lp,type='staeckel',delta=0.5,use_physical=False)) < 10.**-10., 'o.Tr() output for Orbit setup with ro=,vo= does not work as expected'
    assert numpy.fabs(o.Tp(pot=lp,type='staeckel',delta=0.5)-o.Tp(pot=lp,type='staeckel',delta=0.5,use_physical=False)) < 10.**-10., 'o.Tp() output for Orbit setup with ro=,vo= does not work as expected'
    assert numpy.fabs(o.Tz(pot=lp,type='staeckel',delta=0.5)-o.Tz(pot=lp,type='staeckel',delta=0.5,use_physical=False)) < 10.**-10., 'o.Tz() output for Orbit setup with ro=,vo= does not work as expected'
    assert numpy.fabs(o.Or(pot=lp,type='staeckel',delta=0.5)-o.Or(pot=lp,type='staeckel',delta=0.5,use_physical=False)) < 10.**-10., 'o.Or() output for Orbit setup with ro=,vo= does not work as expected'
    assert numpy.fabs(o.Op(pot=lp,type='staeckel',delta=0.5)-o.Op(pot=lp,type='staeckel',delta=0.5,use_physical=False)) < 10.**-10., 'o.Op() output for Orbit setup with ro=,vo= does not work as expected'
    assert numpy.fabs(o.Oz(pot=lp,type='staeckel',delta=0.5)-o.Oz(pot=lp,type='staeckel',delta=0.5,use_physical=False)) < 10.**-10., 'o.Oz() output for Orbit setup with ro=,vo= does not work as expected'
    #Also test the times
    assert numpy.fabs((o.time(1.)-1.)) < 10.**-10., 'o.time() in physical coordinates does not work as expected when turned off'
    assert numpy.fabs((o.time(1.,ro=ro,vo=vo)-ro/vo/1.0227121655399913)) < 10.**-10., 'o.time() in physical coordinates does not work as expected when turned off'
    return None

# Check that the routines that should return physical coordinates are turned
# back on by turn_physical_on
def test_physical_output_on():
    from galpy.orbit import Orbits
    from galpy.potential import LogarithmicHaloPotential
    from astropy import units
    lp= LogarithmicHaloPotential(normalize=1.)
    o= Orbits()
    ro= o._ro
    vo= o._vo
    o_orig= o()
    #turn off and on
    o.turn_physical_off()
    for ii in range(3):
        if ii == 0:
            o.turn_physical_on(ro=ro,vo=vo)
        elif ii == 1:
            o.turn_physical_on(ro=ro*units.kpc,vo=vo*units.km/units.s)
        else:
            o.turn_physical_on()
        #Test positions
        assert numpy.fabs(o.R()-o_orig.R(use_physical=True)) < 10.**-10., 'o.R() output for Orbit setup with ro= does not work as expected when turned back on'
        assert numpy.fabs(o.x()-o_orig.x(use_physical=True)) < 10.**-10., 'o.x() output for Orbit setup with ro= does not work as expected when turned back on'
        assert numpy.fabs(o.y()-o_orig.y(use_physical=True)) < 10.**-10., 'o.y() output for Orbit setup with ro= does not work as expected when turned back on'
        assert numpy.fabs(o.z()-o_orig.z(use_physical=True)) < 10.**-10., 'o.z() output for Orbit setup with ro= does not work as expected when turned back on'
        #Test velocities
        assert numpy.fabs(o.vR()-o_orig.vR(use_physical=True)) < 10.**-10., 'o.vR() output for Orbit setup with vo= does not work as expected when turned back on'
        assert numpy.fabs(o.vT()-o_orig.vT(use_physical=True)) < 10.**-10., 'o.vT() output for Orbit setup with vo= does not work as expected'
        assert numpy.fabs(o.vphi()-o_orig.vphi(use_physical=True)) < 10.**-10., 'o.vphi() output for Orbit setup with vo= does not work as expected when turned back on'
        assert numpy.fabs(o.vx()-o_orig.vx(use_physical=True)) < 10.**-10., 'o.vx() output for Orbit setup with vo= does not work as expected when turned back on'
        assert numpy.fabs(o.vy()-o_orig.vy(use_physical=True)) < 10.**-10., 'o.vy() output for Orbit setup with vo= does not work as expected when turned back on'
        assert numpy.fabs(o.vz()-o_orig.vz(use_physical=True)) < 10.**-10., 'o.vz() output for Orbit setup with vo= does not work as expected when turned back on'
        #Test energies
        assert numpy.fabs(o.E(pot=lp)-o_orig.E(pot=lp,use_physical=True)) < 10.**-10., 'o.E() output for Orbit setup with vo= does not work as expected when turned back on'
        assert numpy.fabs(o.Jacobi(pot=lp)-o_orig.Jacobi(pot=lp,use_physical=True)) < 10.**-10., 'o.E() output for Orbit setup with vo= does not work as expected when turned back on'
        assert numpy.fabs(o.ER(pot=lp)-o_orig.ER(pot=lp,use_physical=True)) < 10.**-10., 'o.ER() output for Orbit setup with vo= does not work as expected when turned back on'
        assert numpy.fabs(o.Ez(pot=lp)-o_orig.Ez(pot=lp,use_physical=True)) < 10.**-10., 'o.Ez() output for Orbit setup with vo= does not work as expected when turned back on'
        #Test angular momentun
        assert numpy.all(numpy.fabs(o.L()-o_orig.L(use_physical=True)) < 10.**-10.), 'o.L() output for Orbit setup with ro=,vo= does not work as expected when turned back on'
        # Test action-angle functions
        assert numpy.fabs(o.jr(pot=lp,type='staeckel',delta=0.5)-o_orig.jr(pot=lp,type='staeckel',delta=0.5,use_physical=True)) < 10.**-10., 'o.jr() output for Orbit setup with ro=,vo= does not work as expected'
        assert numpy.fabs(o.jp(pot=lp,type='staeckel',delta=0.5)-o_orig.jp(pot=lp,type='staeckel',delta=0.5,use_physical=True)) < 10.**-10., 'o.jp() output for Orbit setup with ro=,vo= does not work as expected'
        assert numpy.fabs(o.jz(pot=lp,type='staeckel',delta=0.5)-o_orig.jz(pot=lp,type='staeckel',delta=0.5,use_physical=True)) < 10.**-10., 'o.jz() output for Orbit setup with ro=,vo= does not work as expected'
        assert numpy.fabs(o.Tr(pot=lp,type='staeckel',delta=0.5)-o_orig.Tr(pot=lp,type='staeckel',delta=0.5,use_physical=True)) < 10.**-10., 'o.Tr() output for Orbit setup with ro=,vo= does not work as expected'
        assert numpy.fabs(o.Tp(pot=lp,type='staeckel',delta=0.5)-o_orig.Tp(pot=lp,type='staeckel',delta=0.5,use_physical=True)) < 10.**-10., 'o.Tp() output for Orbit setup with ro=,vo= does not work as expected'
        assert numpy.fabs(o.Tz(pot=lp,type='staeckel',delta=0.5)-o_orig.Tz(pot=lp,type='staeckel',delta=0.5,use_physical=True)) < 10.**-10., 'o.Tz() output for Orbit setup with ro=,vo= does not work as expected'
        assert numpy.fabs(o.Or(pot=lp,type='staeckel',delta=0.5)-o_orig.Or(pot=lp,type='staeckel',delta=0.5,use_physical=True)) < 10.**-10., 'o.Or() output for Orbit setup with ro=,vo= does not work as expected'
        assert numpy.fabs(o.Op(pot=lp,type='staeckel',delta=0.5)-o_orig.Op(pot=lp,type='staeckel',delta=0.5,use_physical=True)) < 10.**-10., 'o.Op() output for Orbit setup with ro=,vo= does not work as expected'
        assert numpy.fabs(o.Oz(pot=lp,type='staeckel',delta=0.5)-o_orig.Oz(pot=lp,type='staeckel',delta=0.5,use_physical=True)) < 10.**-10., 'o.Oz() output for Orbit setup with ro=,vo= does not work as expected'
    #Also test the times
    assert numpy.fabs((o.time(1.)-o_orig.time(1.,use_physical=True))) < 10.**-10., 'o_orig.time() in physical coordinates does not work as expected when turned back on'
    return None
