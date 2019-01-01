##########################TESTS ON MULTIPLE ORBITS#############################
import numpy
import pytest
from galpy import potential

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
        assert numpy.amax(numpy.fabs(orbits_list[ii].phi(times)-orbits.phi(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
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
        assert numpy.amax(numpy.fabs(orbits_list[ii].phi(times)-orbits.phi(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
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
        assert numpy.amax(numpy.fabs(orbits_list[ii].phi(times)-orbits.phi(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
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
        assert numpy.amax(numpy.fabs(orbits_list[ii].phi(times)-orbits.phi(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
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

# Check plotting routines
def test_plotting():
    from galpy.orbit import Orbit, Orbits
    from galpy.potential import LogarithmicHaloPotential
    o= Orbits([Orbit([1.,0.1,1.1,0.1,0.2,2.]),Orbit([1.,0.1,1.1,0.1,0.2,2.])])
    times= numpy.linspace(0.,7.,251)
    lp= LogarithmicHaloPotential(normalize=1.,q=0.8)
    # Integrate
    o.integrate(times,lp)
    # Some plots
    o.plotE()
    # Plot the orbit itself
    o.plot() #defaults
    o.plot(d1='vR')
    o.plotR()
    o.plotvR(d1='vT')
    o.plotvT(d1='z')
    o.plotz(d1='vz')
    return None
