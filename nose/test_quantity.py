# Make sure to set configuration, needs to be before any galpy imports
from galpy.util import config
config.__config__.set('astropy','astropy-units','True')
import numpy
from astropy import units

def test_orbit_setup_radec_basic():
    from galpy.orbit import Orbit
    o= Orbit([10.*units.deg,-20.*units.deg,3.*units.kpc,
              -3.*units.mas/units.yr,2.*units.mas/units.yr,
              130.*units.km/units.s],radec=True)
    assert numpy.fabs(o.ra(use_physical=False)-10.) < 10.**-8., 'Orbit initialization with RA as Quantity does not work as expected'
    assert numpy.fabs(o.dec(use_physical=False)+20.) < 10.**-8., 'Orbit initialization with Dec as Quantity does not work as expected'
    assert numpy.fabs(o.dist(use_physical=False)-3.) < 10.**-8., 'Orbit initialization with distance as Quantity does not work as expected'
    assert numpy.fabs(o.pmra(use_physical=False)+3.) < 10.**-8., 'Orbit initialization with pmra as Quantity does not work as expected'
    assert numpy.fabs(o.pmdec(use_physical=False)-2.) < 10.**-8., 'Orbit initialization with pmdec as Quantity does not work as expected'
    assert numpy.fabs(o.vlos(use_physical=False)-130.) < 10.**-8., 'Orbit initialization with vlos as Quantity does not work as expected'
    return None

def test_orbit_setup_radec_oddunits():
    from galpy.orbit import Orbit
    o= Orbit([1.*units.rad,-0.25*units.rad,3000.*units.lyr,
              -3.*units.mas/units.s,2.*units.mas/units.kyr,
              130.*units.pc/units.Myr],radec=True)
    assert numpy.fabs(o.ra(use_physical=False)-1./numpy.pi*180.) < 10.**-8., 'Orbit initialization with RA as Quantity does not work as expected'
    assert numpy.fabs(o.dec(use_physical=False)+.25/numpy.pi*180.) < 10.**-8., 'Orbit initialization with Dec as Quantity does not work as expected'
    assert numpy.fabs(o.dist(use_physical=False)-3./3.26156) < 10.**-5., 'Orbit initialization with distance as Quantity does not work as expected'
    assert numpy.fabs((o.pmra(use_physical=False)+3.*units.yr.to(units.s))/o.pmra(use_physical=False)) < 10.**-8., 'Orbit initialization with pmra as Quantity does not work as expected'
    assert numpy.fabs((o.pmdec(use_physical=False)-2./10.**3.)/o.pmdec(use_physical=False)) < 10.**-4., 'Orbit initialization with pmdec as Quantity does not work as expected'
    assert numpy.fabs(o.vlos(use_physical=False)-130./1.0227121655399913) < 10.**-5., 'Orbit initialization with vlos as Quantity does not work as expected'
    return None

def test_orbit_setup_radec_uvw():
    from galpy.orbit import Orbit
    o= Orbit([1.*units.rad,-0.25*units.rad,3000.*units.pc,
              -30.*units.km/units.s,20.*units.km/units.s,
              130.*units.km/units.s],radec=True,uvw=True)
    assert numpy.fabs(o.ra(use_physical=False)-1./numpy.pi*180.) < 10.**-8., 'Orbit initialization with RA as Quantity does not work as expected'
    assert numpy.fabs(o.dec(use_physical=False)+.25/numpy.pi*180.) < 10.**-8., 'Orbit initialization with Dec as Quantity does not work as expected'
    assert numpy.fabs(o.dist(use_physical=False)-3.) < 10.**-8., 'Orbit initialization with distance as Quantity does not work as expected'
    assert numpy.fabs(o.U(use_physical=False)+30.) < 10.**-8., 'Orbit initialization with U as Quantity does not work as expected'
    assert numpy.fabs(o.V(use_physical=False)-20.) < 10.**-8., 'Orbit initialization with V as Quantity does not work as expected'
    assert numpy.fabs(o.W(use_physical=False)-130.) < 10.**-8., 'Orbit initialization with W as Quantity does not work as expected'
    return None

def test_orbit_setup_radec_uvw_oddunits():
    from galpy.orbit import Orbit
    o= Orbit([1.*units.rad,-0.25*units.rad,3000.*units.pc,
              -30.*units.pc/units.Myr,20.*units.pc/units.Myr,
              130.*units.pc/units.Myr],radec=True,uvw=True)
    assert numpy.fabs(o.ra(use_physical=False)-1./numpy.pi*180.) < 10.**-8., 'Orbit initialization with RA as Quantity does not work as expected'
    assert numpy.fabs(o.dec(use_physical=False)+.25/numpy.pi*180.) < 10.**-8., 'Orbit initialization with Dec as Quantity does not work as expected'
    assert numpy.fabs(o.dist(use_physical=False)-3.) < 10.**-8., 'Orbit initialization with distance as Quantity does not work as expected'
    assert numpy.fabs(o.U(use_physical=False)+30./1.0227121655399913) < 10.**-5., 'Orbit initialization with U as Quantity does not work as expected'
    assert numpy.fabs(o.V(use_physical=False)-20./1.0227121655399913) < 10.**-5., 'Orbit initialization with V as Quantity does not work as expected'
    assert numpy.fabs(o.W(use_physical=False)-130./1.0227121655399913) < 10.**-5., 'Orbit initialization with W as Quantity does not work as expected'
    return None

def test_orbit_setup_lb_basic():
    from galpy.orbit import Orbit
    o= Orbit([10.*units.deg,-20.*units.deg,3.*units.kpc,
              -3.*units.mas/units.yr,2.*units.mas/units.yr,
              130.*units.km/units.s],lb=True)
    assert numpy.fabs(o.ll(use_physical=False)-10.) < 10.**-8., 'Orbit initialization with ll as Quantity does not work as expected'
    assert numpy.fabs(o.bb(use_physical=False)+20.) < 10.**-8., 'Orbit initialization with bb as Quantity does not work as expected'
    assert numpy.fabs(o.dist(use_physical=False)-3.) < 10.**-8., 'Orbit initialization with distance as Quantity does not work as expected'
    assert numpy.fabs(o.pmll(use_physical=False)+3.) < 10.**-8., 'Orbit initialization with pmra as Quantity does not work as expected'
    assert numpy.fabs(o.pmbb(use_physical=False)-2.) < 10.**-8., 'Orbit initialization with pmdec as Quantity does not work as expected'
    assert numpy.fabs(o.vlos(use_physical=False)-130.) < 10.**-8., 'Orbit initialization with vlos as Quantity does not work as expected'
    return None

def test_orbit_setup_lb_oddunits():
    from galpy.orbit import Orbit
    o= Orbit([1.*units.rad,-0.25*units.rad,3000.*units.lyr,
              -3.*units.mas/units.s,2.*units.mas/units.kyr,
              130.*units.pc/units.Myr],lb=True)
    assert numpy.fabs(o.ll(use_physical=False)-1./numpy.pi*180.) < 10.**-8., 'Orbit initialization with ll as Quantity does not work as expected'
    assert numpy.fabs(o.bb(use_physical=False)+.25/numpy.pi*180.) < 10.**-8., 'Orbit initialization with bb as Quantity does not work as expected'
    assert numpy.fabs(o.dist(use_physical=False)-3./3.26156) < 10.**-5., 'Orbit initialization with distance as Quantity does not work as expected'
    assert numpy.fabs((o.pmll(use_physical=False)+3.*units.yr.to(units.s))/o.pmll(use_physical=False)) < 10.**-8., 'Orbit initialization with pmll as Quantity does not work as expected'
    assert numpy.fabs((o.pmbb(use_physical=False)-2./10.**3.)/o.pmbb(use_physical=False)) < 10.**-4., 'Orbit initialization with pmbb as Quantity does not work as expected'
    assert numpy.fabs(o.vlos(use_physical=False)-130./1.0227121655399913) < 10.**-5., 'Orbit initialization with vlos as Quantity does not work as expected'
    return None

def test_orbit_setup_lb_uvw():
    from galpy.orbit import Orbit
    o= Orbit([1.*units.rad,-0.25*units.rad,3000.*units.pc,
              -30.*units.km/units.s,20.*units.km/units.s,
              130.*units.km/units.s],lb=True,uvw=True)
    assert numpy.fabs(o.ll(use_physical=False)-1./numpy.pi*180.) < 10.**-8., 'Orbit initialization with ll as Quantity does not work as expected'
    assert numpy.fabs(o.bb(use_physical=False)+.25/numpy.pi*180.) < 10.**-8., 'Orbit initialization with bb as Quantity does not work as expected'
    assert numpy.fabs(o.dist(use_physical=False)-3.) < 10.**-8., 'Orbit initialization with distance as Quantity does not work as expected'
    assert numpy.fabs(o.U(use_physical=False)+30.) < 10.**-8., 'Orbit initialization with pmll as Quantity does not work as expected'
    assert numpy.fabs(o.V(use_physical=False)-20.) < 10.**-8., 'Orbit initialization with pmbb as Quantity does not work as expected'
    assert numpy.fabs(o.W(use_physical=False)-130.) < 10.**-8., 'Orbit initialization with W as Quantity does not work as expected'
    return None

def test_orbit_setup_lb_uvw_oddunits():
    from galpy.orbit import Orbit
    o= Orbit([1.*units.rad,-0.25*units.rad,3000.*units.pc,
              -30.*units.pc/units.Myr,20.*units.pc/units.Myr,
              130.*units.pc/units.Myr],lb=True,uvw=True)
    assert numpy.fabs(o.ll(use_physical=False)-1./numpy.pi*180.) < 10.**-8., 'Orbit initialization with ll as Quantity does not work as expected'
    assert numpy.fabs(o.bb(use_physical=False)+.25/numpy.pi*180.) < 10.**-8., 'Orbit initialization with bb as Quantity does not work as expected'
    assert numpy.fabs(o.dist(use_physical=False)-3.) < 10.**-8., 'Orbit initialization with distance as Quantity does not work as expected'
    assert numpy.fabs(o.U(use_physical=False)+30./1.0227121655399913) < 10.**-5., 'Orbit initialization with U as Quantity does not work as expected'
    assert numpy.fabs(o.V(use_physical=False)-20./1.0227121655399913) < 10.**-5., 'Orbit initialization with V as Quantity does not work as expected'
    assert numpy.fabs(o.W(use_physical=False)-130./1.0227121655399913) < 10.**-5., 'Orbit initialization with W as Quantity does not work as expected'
    return None

def test_orbit_setup_vxvv_fullorbit():
    from galpy.orbit import Orbit
    o= Orbit([10.*units.kpc,-20.*units.km/units.s,210.*units.km/units.s,
              500.*units.pc,-12.*units.km/units.s,45.*units.deg])
    assert numpy.fabs(o.R(use_physical=False)*o._ro-10.) < 10.**-8., 'Orbit initialization with vxvv as Quantity does not work as expected for FullOrbit'
    assert numpy.fabs(o.vR(use_physical=False)*o._vo+20.) < 10.**-8., 'Orbit initialization with vxvv as Quantity does not work as expected for FullOrbit'
    assert numpy.fabs(o.vT(use_physical=False)*o._vo-210.) < 10.**-8., 'Orbit initialization with vxvv as Quantity does not work as expected for FullOrbit'
    assert numpy.fabs(o.z(use_physical=False)*o._ro-0.5) < 10.**-8., 'Orbit initialization with vxvv as Quantity does not work as expected for FullOrbit'
    assert numpy.fabs(o.vz(use_physical=False)*o._vo+12) < 10.**-8., 'Orbit initialization with vxvv as Quantity does not work as expected for FullOrbit'
    assert numpy.fabs(o.phi(use_physical=False)-45./180.*numpy.pi) < 10.**-8., 'Orbit initialization with vxvv as Quantity does not work as expected for FullOrbit'
    return None

def test_orbit_setup_vxvv_rzorbit():
    from galpy.orbit import Orbit
    o= Orbit([10000.*units.lyr,-20.*units.km/units.s,210.*units.km/units.s,
              500.*units.pc,-12.*units.pc/units.Myr])
    assert numpy.fabs(o.R(use_physical=False)*o._ro-10./3.26156) < 10.**-5., 'Orbit initialization with vxvv as Quantity does not work as expected for RZOrbit'
    assert numpy.fabs(o.vR(use_physical=False)*o._vo+20.) < 10.**-8., 'Orbit initialization with vxvv as Quantity does not work as expected for RZOrbit'
    assert numpy.fabs(o.vT(use_physical=False)*o._vo-210.) < 10.**-8., 'Orbit initialization with vxvv as Quantity does not work as expected for RZOrbit'
    assert numpy.fabs(o.z(use_physical=False)*o._ro-0.5) < 10.**-8., 'Orbit initialization with vxvv as Quantity does not work as expected for RZOrbit'
    assert numpy.fabs(o.vz(use_physical=False)*o._vo+12./1.0227121655399913) < 10.**-5., 'Orbit initialization with vxvv as Quantity does not work as expected for RZOrbit'
    return None

def test_orbit_setup_vxvv_planarorbit():
    from galpy.orbit import Orbit
    o= Orbit([10000.*units.lyr,-20.*units.km/units.s,210.*units.km/units.s,
              3.*units.rad])
    assert numpy.fabs(o.R(use_physical=False)*o._ro-10./3.26156) < 10.**-5., 'Orbit initialization with vxvv as Quantity does not work as expected for RZOrbit'
    assert numpy.fabs(o.vR(use_physical=False)*o._vo+20.) < 10.**-8., 'Orbit initialization with vxvv as Quantity does not work as expected for RZOrbit'
    assert numpy.fabs(o.vT(use_physical=False)*o._vo-210.) < 10.**-8., 'Orbit initialization with vxvv as Quantity does not work as expected for RZOrbit'
    assert numpy.fabs(o.phi(use_physical=False)-3.) < 10.**-8., 'Orbit initialization with vxvv as Quantity does not work as expected for FullOrbit'
    return None

def test_orbit_setup_vxvv_planarrorbit():
    from galpy.orbit import Orbit
    o= Orbit([7.*units.kpc,-2.*units.km/units.s,210.*units.km/units.s],
             ro=10.,vo=150.)
    assert numpy.fabs(o.R(use_physical=False)*o._ro-7.) < 10.**-8., 'Orbit initialization with vxvv as Quantity does not work as expected for RZOrbit'
    assert numpy.fabs(o.vR(use_physical=False)*o._vo+2.) < 10.**-8., 'Orbit initialization with vxvv as Quantity does not work as expected for RZOrbit'
    assert numpy.fabs(o.vT(use_physical=False)*o._vo-210.) < 10.**-8., 'Orbit initialization with vxvv as Quantity does not work as expected for RZOrbit'
    return None

def test_orbit_setup_vxvv_linearorbit():
    from galpy.orbit import Orbit
    o= Orbit([7.*units.kpc,-21.*units.pc/units.Myr])
    assert numpy.fabs(o.x(use_physical=False)*o._ro-7.) < 10.**-8., 'Orbit initialization with vxvv as Quantity does not work as expected for RZOrbit'
    assert numpy.fabs(o.vx(use_physical=False)*o._vo+21./1.0227121655399913) < 10.**-5., 'Orbit initialization with vxvv as Quantity does not work as expected for RZOrbit'
    return None
