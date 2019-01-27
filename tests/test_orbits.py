##########################TESTS ON MULTIPLE ORBITS#############################
import numpy
import pytest
from galpy import potential

# Test Orbits initialization
def test_initialization_vxvv():
    from galpy.orbit import Orbit, Orbits
    # 1D
    vxvvs= [[1.,0.1],[0.1,3.]]
    orbits= Orbits(vxvvs)
    assert orbits.dim() == 1, 'Orbit initialization with vxvv in 1D does not work as expected'
    assert orbits.phasedim() == 2, 'Orbit initialization with vxvv in 1D does not work as expected'
    assert numpy.fabs(orbits.x()[0]-1.) < 1e-10, 'Orbit initialization with vxvv in 1D does not work as expected'
    assert numpy.fabs(orbits.x()[1]-0.1) < 1e-10, 'Orbit initialization with vxvv in 1D does not work as expected'
    assert numpy.fabs(orbits.vx()[0]-0.1) < 1e-10, 'Orbit initialization with vxvv in 1D does not work as expected'
    assert numpy.fabs(orbits.vx()[1]-3.) < 1e-10, 'Orbit initialization with vxvv in 1D does not work as expected'
    # 2D, 3 phase-D
    vxvvs= [[1.,0.1,1.],[0.1,3.,1.1]]
    orbits= Orbits(vxvvs)
    assert orbits.dim() == 2, 'Orbit initialization with vxvv in 2D, 3 phase-D does not work as expected'
    assert orbits.phasedim() == 3, 'Orbit initialization with vxvv in 2D, 3 phase-D does not work as expected'
    assert numpy.fabs(orbits.R()[0]-1.) < 1e-10, 'Orbit initialization with vxvv in 2D, 3 phase-D does not work as expected'
    assert numpy.fabs(orbits.R()[1]-0.1) < 1e-10, 'Orbit initialization with vxvv in 2D, 3 phase-D does not work as expected'
    assert numpy.fabs(orbits.vR()[0]-0.1) < 1e-10, 'Orbit initialization with vxvv in 2D, 3 phase-D does not work as expected'
    assert numpy.fabs(orbits.vR()[1]-3.) < 1e-10, 'Orbit initialization with vxvv in 2D, 3 phase-D does not work as expected'
    assert numpy.fabs(orbits.vT()[0]-1.) < 1e-10, 'Orbit initialization with vxvv in 2D, 3 phase-D does not work as expected'
    assert numpy.fabs(orbits.vT()[1]-1.1) < 1e-10, 'Orbit initialization with vxvv in 2D, 3 phase-D does not work as expected'
    # 2D, 4 phase-D
    vxvvs= [[1.,0.1,1.,1.5],[0.1,3.,1.1,2.]]
    orbits= Orbits(vxvvs)
    assert orbits.dim() == 2, 'Orbit initialization with vxvv 2D, 4 phase-D does not work as expected'
    assert orbits.phasedim() == 4, 'Orbit initialization with vxvv 2D, 4 phase-D does not work as expected'
    assert numpy.fabs(orbits.R()[0]-1.) < 1e-10, 'Orbit initialization with vxvv 2D, 4 phase-D does not work as expected'
    assert numpy.fabs(orbits.R()[1]-0.1) < 1e-10, 'Orbit initialization with vxvv 2D, 4 phase-D does not work as expected'
    assert numpy.fabs(orbits.vR()[0]-0.1) < 1e-10, 'Orbit initialization with vxvv 2D, 4 phase-D does not work as expected'
    assert numpy.fabs(orbits.vR()[1]-3.) < 1e-10, 'Orbit initialization with vxvv 2D, 4 phase-D does not work as expected'
    assert numpy.fabs(orbits.vT()[0]-1.) < 1e-10, 'Orbit initialization with vxvv 2D, 4 phase-D does not work as expected'
    assert numpy.fabs(orbits.vT()[1]-1.1) < 1e-10, 'Orbit initialization with vxvv 2D, 4 phase-D does not work as expected'
    assert numpy.fabs(orbits.phi()[0]-1.5) < 1e-10, 'Orbit initialization with vxvv 2D, 4 phase-D does not work as expected'
    assert numpy.fabs(orbits.phi()[1]-2.) < 1e-10, 'Orbit initialization with vxvv 2D, 4 phase-D does not work as expected'
    # 3D, 5 phase-D
    vxvvs= [[1.,0.1,1.,0.1,-0.2],[0.1,3.,1.1,-0.3,0.4]]
    orbits= Orbits(vxvvs)
    assert orbits.dim() == 3, 'Orbit initialization with vxvv 3D, 5 phase-D does not work as expected'
    assert orbits.phasedim() == 5, 'Orbit initialization with vxvv 3D, 5 phase-D does not work as expected'
    assert numpy.fabs(orbits.R()[0]-1.) < 1e-10, 'Orbit initialization with vxvv 3D, 5 phase-D does not work as expected'
    assert numpy.fabs(orbits.R()[1]-0.1) < 1e-10, 'Orbit initialization with vxvv 3D, 5 phase-D does not work as expected'
    assert numpy.fabs(orbits.vR()[0]-0.1) < 1e-10, 'Orbit initialization with vxvv 3D, 5 phase-D does not work as expected'
    assert numpy.fabs(orbits.vR()[1]-3.) < 1e-10, 'Orbit initialization with vxvv 3D, 5 phase-D does not work as expected'
    assert numpy.fabs(orbits.vT()[0]-1.) < 1e-10, 'Orbit initialization with vxvv 3D, 5 phase-D does not work as expected'
    assert numpy.fabs(orbits.vT()[1]-1.1) < 1e-10, 'Orbit initialization with vxvv 3D, 5 phase-D does not work as expected'
    assert numpy.fabs(orbits.z()[0]-0.1) < 1e-10, 'Orbit initialization with vxvv 3D, 5 phase-D does not work as expected'
    assert numpy.fabs(orbits.z()[1]+0.3) < 1e-10, 'Orbit initialization with vxvv 3D, 5 phase-D does not work as expected'
    assert numpy.fabs(orbits.vz()[0]+0.2) < 1e-10, 'Orbit initialization with vxvv 3D, 5 phase-D does not work as expected'
    assert numpy.fabs(orbits.vz()[1]-0.4) < 1e-10, 'Orbit initialization with vxvv 3D, 5 phase-D does not work as expected'
    # 3D, 5 phase-D
    vxvvs= [[1.,0.1,1.,0.1,-0.2,1.5],[0.1,3.,1.1,-0.3,0.4,2.]]
    orbits= Orbits(vxvvs)
    assert orbits.dim() == 3, 'Orbit initialization with vxvv in 3D, 6 phase-D does not work as expected'
    assert orbits.phasedim() == 6, 'Orbit initialization with vxvv in 3D, 6 phase-D does not work as expected'
    assert numpy.fabs(orbits.R()[0]-1.) < 1e-10, 'Orbit initialization with vxvv in 3D, 6 phase-D does not work as expected'
    assert numpy.fabs(orbits.R()[1]-0.1) < 1e-10, 'Orbit initialization with vxvv in 3D, 6 phase-D does not work as expected'
    assert numpy.fabs(orbits.vR()[0]-0.1) < 1e-10, 'Orbit initialization with vxvv in 3D, 6 phase-D does not work as expected'
    assert numpy.fabs(orbits.vR()[1]-3.) < 1e-10, 'Orbit initialization with vxvv in 3D, 6 phase-D does not work as expected'
    assert numpy.fabs(orbits.vT()[0]-1.) < 1e-10, 'Orbit initialization with vxvv in 3D, 6 phase-D does not work as expected'
    assert numpy.fabs(orbits.vT()[1]-1.1) < 1e-10, 'Orbit initialization with vxvv in 3D, 6 phase-D does not work as expected'
    assert numpy.fabs(orbits.z()[0]-0.1) < 1e-10, 'Orbit initialization with vxvv in 3D, 6 phase-D does not work as expected'
    assert numpy.fabs(orbits.z()[1]+0.3) < 1e-10, 'Orbit initialization with vxvv in 3D, 6 phase-D does not work as expected'
    assert numpy.fabs(orbits.vz()[0]+0.2) < 1e-10, 'Orbit initialization with vxvv in 3D, 6 phase-D does not work as expected'
    assert numpy.fabs(orbits.vz()[1]-0.4) < 1e-10, 'Orbit initialization with vxvv in 3D, 6 phase-D does not work as expected'
    assert numpy.fabs(orbits.phi()[0]-1.5) < 1e-10, 'Orbit initialization with vxvv in 3D, 6 phase-D does not work as expected'
    assert numpy.fabs(orbits.phi()[1]-2.) < 1e-10, 'Orbit initialization with vxvv in 3D, 6 phase-D does not work as expected'
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
        assert numpy.amax(numpy.fabs(orbits[ii].phi(times)-orbits.phi(times)[ii])) < 1e-10, 'Integration of multiple orbits as Orbits does not agree with integrating multiple orbits'
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
    
@pytest.mark.xfail(strict=True,raises=ValueError)
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

