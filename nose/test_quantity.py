# Make sure to set configuration, needs to be before any galpy imports
from nose.tools import assert_raises
from galpy.util import config
config.__config__.set('astropy','astropy-units','True')
import numpy
from astropy import units, constants

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

def test_orbit_setup_solarmotion():
    from galpy.orbit import Orbit
    o= Orbit([1.,0.1,1.1,0.2,0.1,0.],
             solarmotion=units.Quantity([13.,25.,8.],unit=units.km/units.s))
    assert numpy.fabs(o._orb._solarmotion[0]-13.) < 10.**-8., 'solarmotion in Orbit setup as Quantity does not work as expected'
    assert numpy.fabs(o._orb._solarmotion[1]-25.) < 10.**-8., 'solarmotion in Orbit setup as Quantity does not work as expected'
    assert numpy.fabs(o._orb._solarmotion[2]-8.) < 10.**-8., 'solarmotion in Orbit setup as Quantity does not work as expected'
    return None

def test_orbit_setup_solarmotion_oddunits():
    from galpy.orbit import Orbit
    o= Orbit([1.,0.1,1.1,0.2,0.1,0.],
             solarmotion=units.Quantity([13.,25.,8.],unit=units.kpc/units.Gyr))
    assert numpy.fabs(o._orb._solarmotion[0]-13./1.0227121655399913) < 10.**-5., 'solarmotion in Orbit setup as Quantity does not work as expected'
    assert numpy.fabs(o._orb._solarmotion[1]-25./1.0227121655399913) < 10.**-5., 'solarmotion in Orbit setup as Quantity does not work as expected'
    assert numpy.fabs(o._orb._solarmotion[2]-8./1.0227121655399913) < 10.**-5., 'solarmotion in Orbit setup as Quantity does not work as expected'
    return None

def test_orbit_setup_roAsQuantity():
    from galpy.orbit import Orbit
    o= Orbit([1.,0.1,1.1,0.2,0.1,0.],ro=11*units.kpc)
    assert numpy.fabs(o._ro-11.) < 10.**-10., 'ro in Orbit setup as Quantity does not work as expected'
    assert numpy.fabs(o._orb._ro-11.) < 10.**-10., 'ro in Orbit setup as Quantity does not work as expected'
    return None

def test_orbit_setup_roAsQuantity_oddunits():
    from galpy.orbit import Orbit
    o= Orbit([1.,0.1,1.1,0.2,0.1,0.],ro=11*units.lyr)
    assert numpy.fabs(o._ro-11.*units.lyr.to(units.kpc)) < 10.**-10., 'ro in Orbit setup as Quantity does not work as expected'
    assert numpy.fabs(o._orb._ro-11.*units.lyr.to(units.kpc)) < 10.**-10., 'ro in Orbit setup as Quantity does not work as expected'
    return None

def test_orbit_setup_voAsQuantity():
    from galpy.orbit import Orbit
    o= Orbit([1.,0.1,1.1,0.2,0.1,0.],vo=210*units.km/units.s)
    assert numpy.fabs(o._vo-210.) < 10.**-10., 'vo in Orbit setup as Quantity does not work as expected'
    assert numpy.fabs(o._orb._vo-210.) < 10.**-10., 'vo in Orbit setup as Quantity does not work as expected'
    return None

def test_orbit_setup_voAsQuantity_oddunits():
    from galpy.orbit import Orbit
    o= Orbit([1.,0.1,1.1,0.2,0.1,0.],vo=210*units.pc/units.Myr)
    assert numpy.fabs(o._vo-210.*(units.pc/units.Myr).to(units.km/units.s)) < 10.**-10., 'vo in Orbit setup as Quantity does not work as expected'
    assert numpy.fabs(o._orb._vo-210.*(units.pc/units.Myr).to(units.km/units.s)) < 10.**-10., 'vo in Orbit setup as Quantity does not work as expected'
    return None

def test_orbit_setup_zoAsQuantity():
    from galpy.orbit import Orbit
    o= Orbit([1.,0.1,1.1,0.2,0.1,0.],zo=12*units.pc)
    assert numpy.fabs(o._orb._zo-0.012) < 10.**-10., 'zo in Orbit setup as Quantity does not work as expected'
    return None

def test_orbit_setup_zoAsQuantity_oddunits():
    from galpy.orbit import Orbit
    o= Orbit([1.,0.1,1.1,0.2,0.1,0.],zo=13*units.lyr)
    assert numpy.fabs(o._orb._zo-13.*units.lyr.to(units.kpc)) < 10.**-10., 'zo in Orbit setup as Quantity does not work as expected'
    return None

def test_orbit_method_returntype_scalar():
    from galpy.orbit import Orbit
    o= Orbit([10.*units.kpc,-20.*units.km/units.s,210.*units.km/units.s,
              500.*units.pc,-12.*units.km/units.s,45.*units.deg])
    from galpy.potential import MWPotential2014
    assert isinstance(o.E(pot=MWPotential2014),units.Quantity), 'Orbit method E does not return Quantity when it should'
    assert isinstance(o.ER(pot=MWPotential2014),units.Quantity), 'Orbit method ER does not return Quantity when it should'
    assert isinstance(o.Ez(pot=MWPotential2014),units.Quantity), 'Orbit method Ez does not return Quantity when it should'
    assert isinstance(o.Jacobi(pot=MWPotential2014),units.Quantity), 'Orbit method Jacobi does not return Quantity when it should'
    assert isinstance(o.L(),units.Quantity), 'Orbit method L does not return Quantity when it should'
    assert isinstance(o.rap(pot=MWPotential2014,analytic=True),units.Quantity), 'Orbit method rap does not return Quantity when it should'
    assert isinstance(o.rperi(pot=MWPotential2014,analytic=True),units.Quantity), 'Orbit method rperi does not return Quantity when it should'
    assert isinstance(o.zmax(pot=MWPotential2014,analytic=True),units.Quantity), 'Orbit method zmax does not return Quantity when it should'
    assert isinstance(o.jr(pot=MWPotential2014,type='staeckel',delta=0.5),units.Quantity), 'Orbit method jr does not return Quantity when it should'
    assert isinstance(o.jp(pot=MWPotential2014,type='staeckel',delta=0.5),units.Quantity), 'Orbit method jp does not return Quantity when it should'
    assert isinstance(o.jz(pot=MWPotential2014,type='staeckel',delta=0.5),units.Quantity), 'Orbit method jz does not return Quantity when it should'
    assert isinstance(o.wr(pot=MWPotential2014,type='staeckel',delta=0.5),units.Quantity), 'Orbit method wr does not return Quantity when it should'
    assert isinstance(o.wp(pot=MWPotential2014,type='staeckel',delta=0.5),units.Quantity), 'Orbit method wp does not return Quantity when it should'
    assert isinstance(o.wz(pot=MWPotential2014,type='staeckel',delta=0.5),units.Quantity), 'Orbit method wz does not return Quantity when it should'
    assert isinstance(o.Tr(pot=MWPotential2014,type='staeckel',delta=0.5),units.Quantity), 'Orbit method Tr does not return Quantity when it should'
    assert isinstance(o.Tp(pot=MWPotential2014,type='staeckel',delta=0.5),units.Quantity), 'Orbit method Tp does not return Quantity when it should'
    assert isinstance(o.Tz(pot=MWPotential2014,type='staeckel',delta=0.5),units.Quantity), 'Orbit method Tz does not return Quantity when it should'
    assert isinstance(o.Or(pot=MWPotential2014,type='staeckel',delta=0.5),units.Quantity), 'Orbit method Or does not return Quantity when it should'
    assert isinstance(o.Op(pot=MWPotential2014,type='staeckel',delta=0.5),units.Quantity), 'Orbit method Op does not return Quantity when it should'
    assert isinstance(o.Oz(pot=MWPotential2014,type='staeckel',delta=0.5),units.Quantity), 'Orbit method Oz does not return Quantity when it should'
    assert isinstance(o.time(),units.Quantity), 'Orbit method time does not return Quantity when it should'
    assert isinstance(o.R(),units.Quantity), 'Orbit method R does not return Quantity when it should'
    assert isinstance(o.vR(),units.Quantity), 'Orbit method vR does not return Quantity when it should'
    assert isinstance(o.vT(),units.Quantity), 'Orbit method vT does not return Quantity when it should'
    assert isinstance(o.z(),units.Quantity), 'Orbit method z does not return Quantity when it should'
    assert isinstance(o.vz(),units.Quantity), 'Orbit method vz does not return Quantity when it should'
    assert isinstance(o.phi(),units.Quantity), 'Orbit method phi does not return Quantity when it should'
    assert isinstance(o.vphi(),units.Quantity), 'Orbit method vphi does not return Quantity when it should'
    assert isinstance(o.x(),units.Quantity), 'Orbit method x does not return Quantity when it should'
    assert isinstance(o.y(),units.Quantity), 'Orbit method y does not return Quantity when it should'
    assert isinstance(o.vx(),units.Quantity), 'Orbit method vx does not return Quantity when it should'
    assert isinstance(o.vy(),units.Quantity), 'Orbit method vy does not return Quantity when it should'
    assert isinstance(o.ra(),units.Quantity), 'Orbit method ra does not return Quantity when it should'
    assert isinstance(o.dec(),units.Quantity), 'Orbit method dec does not return Quantity when it should'
    assert isinstance(o.ll(),units.Quantity), 'Orbit method ll does not return Quantity when it should'
    assert isinstance(o.bb(),units.Quantity), 'Orbit method bb does not return Quantity when it should'
    assert isinstance(o.dist(),units.Quantity), 'Orbit method dist does not return Quantity when it should'
    assert isinstance(o.pmra(),units.Quantity), 'Orbit method pmra does not return Quantity when it should'
    assert isinstance(o.pmdec(),units.Quantity), 'Orbit method pmdec does not return Quantity when it should'
    assert isinstance(o.pmll(),units.Quantity), 'Orbit method pmll does not return Quantity when it should'
    assert isinstance(o.pmbb(),units.Quantity), 'Orbit method pmbb does not return Quantity when it should'
    assert isinstance(o.vlos(),units.Quantity), 'Orbit method vlos does not return Quantity when it should'
    assert isinstance(o.vra(),units.Quantity), 'Orbit method vra does not return Quantity when it should'
    assert isinstance(o.vdec(),units.Quantity), 'Orbit method vdec does not return Quantity when it should'
    assert isinstance(o.vll(),units.Quantity), 'Orbit method vll does not return Quantity when it should'
    assert isinstance(o.vbb(),units.Quantity), 'Orbit method vbb does not return Quantity when it should'
    assert isinstance(o.helioX(),units.Quantity), 'Orbit method helioX does not return Quantity when it should'
    assert isinstance(o.helioY(),units.Quantity), 'Orbit method helioY does not return Quantity when it should'
    assert isinstance(o.helioZ(),units.Quantity), 'Orbit method helioZ does not return Quantity when it should'
    assert isinstance(o.U(),units.Quantity), 'Orbit method U does not return Quantity when it should'
    assert isinstance(o.V(),units.Quantity), 'Orbit method V does not return Quantity when it should'
    assert isinstance(o.W(),units.Quantity), 'Orbit method W does not return Quantity when it should'
    return None

def test_orbit_method_returntype():
    from galpy.orbit import Orbit
    o= Orbit([10.*units.kpc,-20.*units.km/units.s,210.*units.km/units.s,
              500.*units.pc,-12.*units.km/units.s,45.*units.deg])
    from galpy.potential import MWPotential2014
    ts= numpy.linspace(0.,6.,1001)
    o.integrate(ts,MWPotential2014)
    assert isinstance(o.E(ts),units.Quantity), 'Orbit method E does not return Quantity when it should'
    assert isinstance(o.ER(ts),units.Quantity), 'Orbit method ER does not return Quantity when it should'
    assert isinstance(o.Ez(ts),units.Quantity), 'Orbit method Ez does not return Quantity when it should'
    assert isinstance(o.Jacobi(ts),units.Quantity), 'Orbit method Jacobi does not return Quantity when it should'
    assert isinstance(o.L(ts),units.Quantity), 'Orbit method L does not return Quantity when it should'
    assert isinstance(o.time(ts),units.Quantity), 'Orbit method time does not return Quantity when it should'
    assert isinstance(o.R(ts),units.Quantity), 'Orbit method R does not return Quantity when it should'
    assert isinstance(o.vR(ts),units.Quantity), 'Orbit method vR does not return Quantity when it should'
    assert isinstance(o.vT(ts),units.Quantity), 'Orbit method vT does not return Quantity when it should'
    assert isinstance(o.z(ts),units.Quantity), 'Orbit method z does not return Quantity when it should'
    assert isinstance(o.vz(ts),units.Quantity), 'Orbit method vz does not return Quantity when it should'
    assert isinstance(o.phi(ts),units.Quantity), 'Orbit method phi does not return Quantity when it should'
    assert isinstance(o.vphi(ts),units.Quantity), 'Orbit method vphi does not return Quantity when it should'
    assert isinstance(o.x(ts),units.Quantity), 'Orbit method x does not return Quantity when it should'
    assert isinstance(o.y(ts),units.Quantity), 'Orbit method y does not return Quantity when it should'
    assert isinstance(o.vx(ts),units.Quantity), 'Orbit method vx does not return Quantity when it should'
    assert isinstance(o.vy(ts),units.Quantity), 'Orbit method vy does not return Quantity when it should'
    assert isinstance(o.ra(ts),units.Quantity), 'Orbit method ra does not return Quantity when it should'
    assert isinstance(o.dec(ts),units.Quantity), 'Orbit method dec does not return Quantity when it should'
    assert isinstance(o.ll(ts),units.Quantity), 'Orbit method ll does not return Quantity when it should'
    assert isinstance(o.bb(ts),units.Quantity), 'Orbit method bb does not return Quantity when it should'
    assert isinstance(o.dist(ts),units.Quantity), 'Orbit method dist does not return Quantity when it should'
    assert isinstance(o.pmra(ts),units.Quantity), 'Orbit method pmra does not return Quantity when it should'
    assert isinstance(o.pmdec(ts),units.Quantity), 'Orbit method pmdec does not return Quantity when it should'
    assert isinstance(o.pmll(ts),units.Quantity), 'Orbit method pmll does not return Quantity when it should'
    assert isinstance(o.pmbb(ts),units.Quantity), 'Orbit method pmbb does not return Quantity when it should'
    assert isinstance(o.vlos(ts),units.Quantity), 'Orbit method vlos does not return Quantity when it should'
    assert isinstance(o.vra(ts),units.Quantity), 'Orbit method vra does not return Quantity when it should'
    assert isinstance(o.vdec(ts),units.Quantity), 'Orbit method vdec does not return Quantity when it should'
    assert isinstance(o.vll(ts),units.Quantity), 'Orbit method vll does not return Quantity when it should'
    assert isinstance(o.vbb(ts),units.Quantity), 'Orbit method vbb does not return Quantity when it should'
    assert isinstance(o.helioX(ts),units.Quantity), 'Orbit method helioX does not return Quantity when it should'
    assert isinstance(o.helioY(ts),units.Quantity), 'Orbit method helioY does not return Quantity when it should'
    assert isinstance(o.helioZ(ts),units.Quantity), 'Orbit method helioZ does not return Quantity when it should'
    assert isinstance(o.U(ts),units.Quantity), 'Orbit method U does not return Quantity when it should'
    assert isinstance(o.V(ts),units.Quantity), 'Orbit method V does not return Quantity when it should'
    assert isinstance(o.W(ts),units.Quantity), 'Orbit method W does not return Quantity when it should'
    return None

def test_orbit_method_returnunit():
    from galpy.orbit import Orbit
    o= Orbit([10.*units.kpc,-20.*units.km/units.s,210.*units.km/units.s,
              500.*units.pc,-12.*units.km/units.s,45.*units.deg])
    from galpy.potential import MWPotential2014
    try:
        o.E(pot=MWPotential2014).to(units.km**2/units.s**2)
    except units.UnitConversionError:
        raise AssertionError('Orbit method E does not return Quantity with the right units')
    try:
        o.ER(pot=MWPotential2014).to(units.km**2/units.s**2)
    except units.UnitConversionError:
        raise AssertionError('Orbit method ER does not return Quantity with the right units')
    try:
        o.Ez(pot=MWPotential2014).to(units.km**2/units.s**2)
    except units.UnitConversionError:
        raise AssertionError('Orbit method Ez does not return Quantity with the right units')
    try:
        o.Jacobi(pot=MWPotential2014).to(units.km**2/units.s**2)
    except units.UnitConversionError:
        raise AssertionError('Orbit method Jacobi does not return Quantity with the right units')
    try:
        o.L().to(units.km**2/units.s)
    except units.UnitConversionError:
        raise AssertionError('Orbit method L does not return Quantity with the right units')
    try:
        o.rap(pot=MWPotential2014,analytic=True).to(units.kpc)
    except units.UnitConversionError:
        raise AssertionError('Orbit method rap does not return Quantity with the right units')
    try:
        o.rperi(pot=MWPotential2014,analytic=True).to(units.kpc)
    except units.UnitConversionError:
        raise AssertionError('Orbit method rperi does not return Quantity with the right units')
    try:
        o.zmax(pot=MWPotential2014,analytic=True).to(units.kpc)
    except units.UnitConversionError:
        raise AssertionError('Orbit method zmax does not return Quantity with the right units')
    try:
        o.jr(pot=MWPotential2014,type='staeckel',delta=0.5).to(units.km**2/units.s)
    except units.UnitConversionError:
        raise AssertionError('Orbit method jr does not return Quantity with the right units')
    try:
        o.jp(pot=MWPotential2014,type='staeckel',delta=0.5).to(units.km**2/units.s)
    except units.UnitConversionError:
        raise AssertionError('Orbit method jp does not return Quantity with the right units')
    try:
        o.jz(pot=MWPotential2014,type='staeckel',delta=0.5).to(units.km**2/units.s)
    except units.UnitConversionError:
        raise AssertionError('Orbit method jz does not return Quantity with the right units')
    try:
        o.wr(pot=MWPotential2014,type='staeckel',delta=0.5).to(units.rad)
    except units.UnitConversionError:
        raise AssertionError('Orbit method wr does not return Quantity with the right units')
    try:
        o.wp(pot=MWPotential2014,type='staeckel',delta=0.5).to(units.rad)
    except units.UnitConversionError:
        raise AssertionError('Orbit method wp does not return Quantity with the right units')
    try:
        o.wz(pot=MWPotential2014,type='staeckel',delta=0.5).to(units.rad)
    except units.UnitConversionError:
        raise AssertionError('Orbit method wz does not return Quantity with the right units')
    try:
        o.Tr(pot=MWPotential2014,type='staeckel',delta=0.5).to(units.yr)
    except units.UnitConversionError:
        raise AssertionError('Orbit method Tr does not return Quantity with the right units')
    try:
        o.Tp(pot=MWPotential2014,type='staeckel',delta=0.5).to(units.yr)
    except units.UnitConversionError:
        raise AssertionError('Orbit method Tp does not return Quantity with the right units')
    try:
        o.Tz(pot=MWPotential2014,type='staeckel',delta=0.5).to(units.yr)
    except units.UnitConversionError:
        raise AssertionError('Orbit method Tz does not return Quantity with the right units')
    try:
        o.Or(pot=MWPotential2014,type='staeckel',delta=0.5).to(1/units.yr)
    except units.UnitConversionError:
        raise AssertionError('Orbit method Or does not return Quantity with the right units')
    try:
        o.Op(pot=MWPotential2014,type='staeckel',delta=0.5).to(1/units.yr)
    except units.UnitConversionError:
        raise AssertionError('Orbit method Op does not return Quantity with the right units')
    try:
        o.Oz(pot=MWPotential2014,type='staeckel',delta=0.5).to(1/units.yr)
    except units.UnitConversionError:
        raise AssertionError('Orbit method Oz does not return Quantity with the right units')
    try:
        o.time().to(units.yr)
    except units.UnitConversionError:
        raise AssertionError('Orbit method time does not return Quantity with the right units')
    try:
        o.R().to(units.pc)
    except units.UnitConversionError:
        raise AssertionError('Orbit method R does not return Quantity with the right units')
    try:
        o.vR().to(units.km/units.s)
    except units.UnitConversionError:
        raise AssertionError('Orbit method vR does not return Quantity with the right units')
    try:
        o.vT().to(units.km/units.s)
    except units.UnitConversionError:
        raise AssertionError('Orbit method vT does not return Quantity with the right units')
    try:
        o.z().to(units.pc)
    except units.UnitConversionError:
        raise AssertionError('Orbit method z does not return Quantity with the right units')
    try:
        o.vz().to(units.km/units.s)
    except units.UnitConversionError:
        raise AssertionError('Orbit method vz does not return Quantity with the right units')
    try:
        o.phi().to(units.deg)
    except units.UnitConversionError:
        raise AssertionError('Orbit method phi does not return Quantity with the right units')
    try:
        o.vphi().to(units.km/units.s)
    except units.UnitConversionError:
        raise AssertionError('Orbit method vphi does not return Quantity with the right units')
    try:
        o.x().to(units.pc)
    except units.UnitConversionError:
        raise AssertionError('Orbit method x does not return Quantity with the right units')
    try:
        o.y().to(units.pc)
    except units.UnitConversionError:
        raise AssertionError('Orbit method y does not return Quantity with the right units')
    try:
        o.vx().to(units.km/units.s)
    except units.UnitConversionError:
        raise AssertionError('Orbit method vx does not return Quantity with the right units')
    try:
        o.vy().to(units.km/units.s)
    except units.UnitConversionError:
        raise AssertionError('Orbit method vy does not return Quantity with the right units')
    try:
        o.ra().to(units.rad)
    except units.UnitConversionError:
        raise AssertionError('Orbit method ra does not return Quantity with the right units')
    try:
        o.dec().to(units.rad)
    except units.UnitConversionError:
        raise AssertionError('Orbit method dec does not return Quantity with the right units')
    try:
        o.ll().to(units.rad)
    except units.UnitConversionError:
        raise AssertionError('Orbit method ll does not return Quantity with the right units')
    try:
        o.bb().to(units.rad)
    except units.UnitConversionError:
        raise AssertionError('Orbit method bb does not return Quantity with the right units')
    try:
        o.dist().to(units.kpc)
    except units.UnitConversionError:
        raise AssertionError('Orbit method dist does not return Quantity with the right units')
    try:
        o.pmra().to(units.mas/units.yr)
    except units.UnitConversionError:
        raise AssertionError('Orbit method pmra does not return Quantity with the right units')
    try:
        o.pmdec().to(units.mas/units.yr)
    except units.UnitConversionError:
        raise AssertionError('Orbit method pmdec does not return Quantity with the right units')
    try:
        o.pmll().to(units.mas/units.yr)
    except units.UnitConversionError:
        raise AssertionError('Orbit method pmll does not return Quantity with the right units')
    try:
        o.pmbb().to(units.mas/units.yr)
    except units.UnitConversionError:
        raise AssertionError('Orbit method pmbb does not return Quantity with the right units')
    try:
        o.vlos().to(units.km/units.s)
    except units.UnitConversionError:
        raise AssertionError('Orbit method vlos does not return Quantity with the right units')
    try:
        o.vra().to(units.km/units.s)
    except units.UnitConversionError:
        raise AssertionError('Orbit method vra does not return Quantity with the right units')
    try:
        o.vdec().to(units.km/units.s)
    except units.UnitConversionError:
        raise AssertionError('Orbit method vdec does not return Quantity with the right units')
    try:
        o.vll().to(units.km/units.s)
    except units.UnitConversionError:
        raise AssertionError('Orbit method vll does not return Quantity with the right units')
    try:
        o.vbb().to(units.km/units.s)
    except units.UnitConversionError:
        raise AssertionError('Orbit method vbb does not return Quantity with the right units')
    try:
        o.helioX().to(units.pc)
    except units.UnitConversionError:
        raise AssertionError('Orbit method helioX does not return Quantity with the right units')
    try:
        o.helioY().to(units.pc)
    except units.UnitConversionError:
        raise AssertionError('Orbit method helioY does not return Quantity with the right units')
    try:
        o.helioZ().to(units.pc)
    except units.UnitConversionError:
        raise AssertionError('Orbit method helioZ does not return Quantity with the right units')
    try:
        o.U().to(units.km/units.s)
    except units.UnitConversionError:
        raise AssertionError('Orbit method U does not return Quantity with the right units')
    try:
        o.V().to(units.km/units.s)
    except units.UnitConversionError:
        raise AssertionError('Orbit method V does not return Quantity with the right units')
    try:
        o.W().to(units.km/units.s)
    except units.UnitConversionError:
        raise AssertionError('Orbit method W does not return Quantity with the right units')
    return None

def test_orbit_method_value():
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014
    from galpy.util import bovy_conversion
    o= Orbit([10.*units.kpc,-20.*units.km/units.s,210.*units.km/units.s,
              500.*units.pc,-12.*units.km/units.s,45.*units.deg])
    oc= o()
    oc.turn_physical_off()
    assert numpy.fabs(o.E(pot=MWPotential2014).to(units.km**2/units.s**2).value-oc.E(pot=MWPotential2014)*o._vo**2.) < 10.**-8., 'Orbit method E does not return the correct value as Quantity'
    assert numpy.fabs(o.ER(pot=MWPotential2014).to(units.km**2/units.s**2).value-oc.ER(pot=MWPotential2014)*o._vo**2.) < 10.**-8., 'Orbit method ER does not return the correct value as Quantity'
    assert numpy.fabs(o.Ez(pot=MWPotential2014).to(units.km**2/units.s**2).value-oc.Ez(pot=MWPotential2014)*o._vo**2.) < 10.**-8., 'Orbit method Ez does not return the correct value as Quantity'
    assert numpy.fabs(o.Jacobi(pot=MWPotential2014).to(units.km**2/units.s**2).value-oc.Jacobi(pot=MWPotential2014)*o._vo**2.) < 10.**-8., 'Orbit method Jacobi does not return the correct value as Quantity'
    assert numpy.all(numpy.fabs(o.L(pot=MWPotential2014).to(units.km/units.s*units.kpc).value-oc.L(pot=MWPotential2014)*o._ro*o._vo) < 10.**-8.), 'Orbit method L does not return the correct value as Quantity'
    assert numpy.fabs(o.rap(pot=MWPotential2014,analytic=True).to(units.kpc).value-oc.rap(pot=MWPotential2014,analytic=True)*o._ro) < 10.**-8., 'Orbit method rap does not return the correct value as Quantity'
    assert numpy.fabs(o.rperi(pot=MWPotential2014,analytic=True).to(units.kpc).value-oc.rperi(pot=MWPotential2014,analytic=True)*o._ro) < 10.**-8., 'Orbit method rperi does not return the correct value as Quantity'
    assert numpy.fabs(o.zmax(pot=MWPotential2014,analytic=True).to(units.kpc).value-oc.zmax(pot=MWPotential2014,analytic=True)*o._ro) < 10.**-8., 'Orbit method zmax does not return the correct value as Quantity'
    assert numpy.fabs(o.jr(pot=MWPotential2014,type='staeckel',delta=0.5).to(units.km/units.s*units.kpc).value-oc.jr(pot=MWPotential2014,type='staeckel',delta=0.5)*o._ro*o._vo) < 10.**-8., 'Orbit method jr does not return the correct value as Quantity'
    assert numpy.fabs(o.jp(pot=MWPotential2014,type='staeckel',delta=4.*units.kpc).to(units.km/units.s*units.kpc).value-oc.jp(pot=MWPotential2014,type='staeckel',delta=0.5)*o._ro*o._vo) < 10.**-8., 'Orbit method jp does not return the correct value as Quantity'
    assert numpy.fabs(o.jz(pot=MWPotential2014,type='isochroneapprox',b=0.8*8.*units.kpc).to(units.km/units.s*units.kpc).value-oc.jz(pot=MWPotential2014,type='isochroneapprox',b=0.8)*o._ro*o._vo) < 10.**-8., 'Orbit method jz does not return the correct value as Quantity'
    assert numpy.fabs(o.wr(pot=MWPotential2014,type='staeckel',delta=0.5).to(units.rad).value-oc.wr(pot=MWPotential2014,type='staeckel',delta=0.5)) < 10.**-8., 'Orbit method wr does not return the correct value as Quantity'
    assert numpy.fabs(o.wp(pot=MWPotential2014,type='staeckel',delta=0.5).to(units.rad).value-oc.wp(pot=MWPotential2014,type='staeckel',delta=0.5)) < 10.**-8., 'Orbit method wp does not return the correct value as Quantity'
    assert numpy.fabs(o.wz(pot=MWPotential2014,type='staeckel',delta=0.5).to(units.rad).value-oc.wz(pot=MWPotential2014,type='staeckel',delta=0.5)) < 10.**-8., 'Orbit method wz does not return the correct value as Quantity'
    assert numpy.fabs(o.Tr(pot=MWPotential2014,type='staeckel',delta=0.5).to(units.Gyr).value-oc.Tr(pot=MWPotential2014,type='staeckel',delta=0.5)*bovy_conversion.time_in_Gyr(o._vo,o._ro)) < 10.**-8., 'Orbit method Tr does not return the correct value as Quantity'
    assert numpy.fabs(o.Tp(pot=MWPotential2014,type='staeckel',delta=0.5).to(units.Gyr).value-oc.Tp(pot=MWPotential2014,type='staeckel',delta=0.5)*bovy_conversion.time_in_Gyr(o._vo,o._ro)) < 10.**-8., 'Orbit method Tp does not return the correct value as Quantity'
    assert numpy.fabs(o.Tz(pot=MWPotential2014,type='staeckel',delta=0.5).to(units.Gyr).value-oc.Tz(pot=MWPotential2014,type='staeckel',delta=0.5)*bovy_conversion.time_in_Gyr(o._vo,o._ro)) < 10.**-8., 'Orbit method Tz does not return the correct value as Quantity'
    assert numpy.fabs(o.Or(pot=MWPotential2014,type='staeckel',delta=0.5).to(1/units.Gyr).value-oc.Or(pot=MWPotential2014,type='staeckel',delta=0.5)*bovy_conversion.freq_in_Gyr(o._vo,o._ro)) < 10.**-8., 'Orbit method Or does not return the correct value as Quantity'
    assert numpy.fabs(o.Op(pot=MWPotential2014,type='staeckel',delta=0.5).to(1/units.Gyr).value-oc.Op(pot=MWPotential2014,type='staeckel',delta=0.5)*bovy_conversion.freq_in_Gyr(o._vo,o._ro)) < 10.**-8., 'Opbit method Or does not return the correct value as Quantity'
    assert numpy.fabs(o.Oz(pot=MWPotential2014,type='staeckel',delta=0.5).to(1/units.Gyr).value-oc.Oz(pot=MWPotential2014,type='staeckel',delta=0.5)*bovy_conversion.freq_in_Gyr(o._vo,o._ro)) < 10.**-8., 'Ozbit method Or does not return the correct value as Quantity'
    assert numpy.fabs(o.time().to(units.Gyr).value-oc.time()*bovy_conversion.time_in_Gyr(o._vo,o._ro)) < 10.**-8., 'Orbit method time does not return the correct value as Quantity'
    assert numpy.fabs(o.R().to(units.kpc).value-oc.R()*o._ro) < 10.**-8., 'Orbit method R does not return the correct value as Quantity'
    assert numpy.fabs(o.vR().to(units.km/units.s).value-oc.vR()*o._vo) < 10.**-8., 'Orbit method vR does not return the correct value as Quantity'
    assert numpy.fabs(o.vT().to(units.km/units.s).value-oc.vT()*o._vo) < 10.**-8., 'Orbit method vT does not return the correct value as Quantity'
    assert numpy.fabs(o.z().to(units.kpc).value-oc.z()*o._ro) < 10.**-8., 'Orbit method z does not return the correct value as Quantity'
    assert numpy.fabs(o.vz().to(units.km/units.s).value-oc.vz()*o._vo) < 10.**-8., 'Orbit method vz does not return the correct value as Quantity'
    assert numpy.fabs(o.phi().to(units.rad).value-oc.phi()) < 10.**-8., 'Orbit method phi does not return the correct value as Quantity'
    assert numpy.fabs(o.vphi().to(units.km/units.s).value-oc.vphi()*o._vo) < 10.**-8., 'Orbit method vphi does not return the correct value as Quantity'
    assert numpy.fabs(o.x().to(units.kpc).value-oc.x()*o._ro) < 10.**-8., 'Orbit method x does not return the correct value as Quantity'
    assert numpy.fabs(o.y().to(units.kpc).value-oc.y()*o._ro) < 10.**-8., 'Orbit method y does not return the correct value as Quantity'
    assert numpy.fabs(o.vx().to(units.km/units.s).value-oc.vx()*o._vo) < 10.**-8., 'Orbit method vx does not return the correct value as Quantity'
    assert numpy.fabs(o.vy().to(units.km/units.s).value-oc.vy()*o._vo) < 10.**-8., 'Orbit method vy does not return the correct value as Quantity'
    assert numpy.fabs(o.ra().to(units.deg).value-oc.ra()) < 10.**-8., 'Orbit method ra does not return the correct value as Quantity'
    assert numpy.fabs(o.dec().to(units.deg).value-oc.dec()) < 10.**-8., 'Orbit method dec does not return the correct value as Quantity'
    assert numpy.fabs(o.ll().to(units.deg).value-oc.ll()) < 10.**-8., 'Orbit method ll does not return the correct value as Quantity'
    assert numpy.fabs(o.bb().to(units.deg).value-oc.bb()) < 10.**-8., 'Orbit method bb does not return the correct value as Quantity'
    assert numpy.fabs(o.dist().to(units.kpc).value-oc.dist()) < 10.**-8., 'Orbit method dist does not return the correct value as Quantity'
    assert numpy.fabs(o.pmra().to(units.mas/units.yr).value-oc.pmra()) < 10.**-8., 'Orbit method pmra does not return the correct value as Quantity'
    assert numpy.fabs(o.pmdec().to(units.mas/units.yr).value-oc.pmdec()) < 10.**-8., 'Orbit method pmdec does not return the correct value as Quantity'
    assert numpy.fabs(o.pmll().to(units.mas/units.yr).value-oc.pmll()) < 10.**-8., 'Orbit method pmll does not return the correct value as Quantity'
    assert numpy.fabs(o.pmbb().to(units.mas/units.yr).value-oc.pmbb()) < 10.**-8., 'Orbit method pmbb does not return the correct value as Quantity'
    assert numpy.fabs(o.vlos().to(units.km/units.s).value-oc.vlos()) < 10.**-8., 'Orbit method vlos does not return the correct value as Quantity'
    assert numpy.fabs(o.vra().to(units.km/units.s).value-oc.vra()) < 10.**-8., 'Orbit method vra does not return the correct value as Quantity'
    assert numpy.fabs(o.vdec().to(units.km/units.s).value-oc.vdec()) < 10.**-8., 'Orbit method vdec does not return the correct value as Quantity'
    assert numpy.fabs(o.vll().to(units.km/units.s).value-oc.vll()) < 10.**-8., 'Orbit method vll does not return the correct value as Quantity'
    assert numpy.fabs(o.vbb().to(units.km/units.s).value-oc.vbb()) < 10.**-8., 'Orbit method vbb does not return the correct value as Quantity'
    assert numpy.fabs(o.helioX().to(units.kpc).value-oc.helioX()) < 10.**-8., 'Orbit method helioX does not return the correct value as Quantity'
    assert numpy.fabs(o.helioY().to(units.kpc).value-oc.helioY()) < 10.**-8., 'Orbit method helioY does not return the correct value as Quantity'
    assert numpy.fabs(o.helioZ().to(units.kpc).value-oc.helioZ()) < 10.**-8., 'Orbit method helioZ does not return the correct value as Quantity'
    assert numpy.fabs(o.U().to(units.km/units.s).value-oc.U()) < 10.**-8., 'Orbit method U does not return the correct value as Quantity'
    assert numpy.fabs(o.V().to(units.km/units.s).value-oc.V()) < 10.**-8., 'Orbit method V does not return the correct value as Quantity'
    assert numpy.fabs(o.W().to(units.km/units.s).value-oc.W()) < 10.**-8., 'Orbit method W does not return the correct value as Quantity'
    return None

def test_integrate_timeAsQuantity():
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential
    from galpy.util import bovy_conversion
    import copy
    ro, vo= 8., 200.
    o= Orbit([10.*units.kpc,-20.*units.km/units.s,210.*units.km/units.s,
              500.*units.pc,-12.*units.km/units.s,45.*units.deg],
             ro=ro,vo=vo)
    oc= o()
    ts_nounits= numpy.linspace(0.,1.,1001)
    ts= units.Quantity(copy.copy(ts_nounits),unit=units.Gyr)
    ts_nounits/= bovy_conversion.time_in_Gyr(vo,ro)
    # Integrate both with Quantity time and with unitless time
    o.integrate(ts,MWPotential)
    oc.integrate(ts_nounits,MWPotential)
    assert numpy.all(numpy.fabs(o.x(ts)-oc.x(ts_nounits)).value < 10.**-8.), 'Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array'
    assert numpy.all(numpy.fabs(o.y(ts)-oc.y(ts_nounits)).value < 10.**-8.), 'Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array'
    assert numpy.all(numpy.fabs(o.z(ts)-oc.z(ts_nounits)).value < 10.**-8.), 'Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array'
    assert numpy.all(numpy.fabs(o.vx(ts)-oc.vx(ts_nounits)).value < 10.**-8.), 'Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array'
    assert numpy.all(numpy.fabs(o.vy(ts)-oc.vy(ts_nounits)).value < 10.**-8.), 'Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array'
    assert numpy.all(numpy.fabs(o.vz(ts)-oc.vz(ts_nounits)).value < 10.**-8.), 'Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array'
    return None

def test_integrate_timeAsQuantity_Myr():
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential
    from galpy.util import bovy_conversion
    import copy
    ro, vo= 8., 200.
    o= Orbit([10.*units.kpc,-20.*units.km/units.s,210.*units.km/units.s,
              500.*units.pc,-12.*units.km/units.s,45.*units.deg],
             ro=ro,vo=vo)
    oc= o()
    ts_nounits= numpy.linspace(0.,1000.,1001)
    ts= units.Quantity(copy.copy(ts_nounits),unit=units.Myr)
    ts_nounits/= bovy_conversion.time_in_Gyr(vo,ro)*1000.
    # Integrate both with Quantity time and with unitless time
    o.integrate(ts,MWPotential)
    oc.integrate(ts_nounits,MWPotential)
    assert numpy.all(numpy.fabs(o.x(ts)-oc.x(ts_nounits)).value < 10.**-8.), 'Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array'
    assert numpy.all(numpy.fabs(o.y(ts)-oc.y(ts_nounits)).value < 10.**-8.), 'Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array'
    assert numpy.all(numpy.fabs(o.z(ts)-oc.z(ts_nounits)).value < 10.**-8.), 'Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array'
    assert numpy.all(numpy.fabs(o.vx(ts)-oc.vx(ts_nounits)).value < 10.**-8.), 'Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array'
    assert numpy.all(numpy.fabs(o.vy(ts)-oc.vy(ts_nounits)).value < 10.**-8.), 'Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array'
    assert numpy.all(numpy.fabs(o.vz(ts)-oc.vz(ts_nounits)).value < 10.**-8.), 'Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array'
    return None

def test_integrate_dxdv_timeAsQuantity():
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential
    from galpy.util import bovy_conversion
    import copy
    ro, vo= 8., 200.
    o= Orbit([10.*units.kpc,-20.*units.km/units.s,210.*units.km/units.s,
              45.*units.deg],
             ro=ro,vo=vo)
    oc= o()
    ts_nounits= numpy.linspace(0.,1.,1001)
    ts= units.Quantity(copy.copy(ts_nounits),unit=units.Gyr)
    ts_nounits/= bovy_conversion.time_in_Gyr(vo,ro)
    # Integrate both with Quantity time and with unitless time
    o.integrate_dxdv([1.,0.3,0.4,0.2],ts,MWPotential,
                     rectIn=True,rectOut=True)
    oc.integrate_dxdv([1.,0.3,0.4,0.2],ts_nounits,MWPotential,
                      rectIn=True,rectOut=True)
    dx= o.getOrbit_dxdv()
    dxc= oc.getOrbit_dxdv()
    assert numpy.all(numpy.fabs(dx-dxc) < 10.**-8.), 'Orbit integrated_dxdv with times specified as Quantity does not agree with Orbit integrated_dxdv with time specified as array'
    return None

def test_integrate_dxdv_timeAsQuantity_Myr():
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential
    from galpy.util import bovy_conversion
    import copy
    ro, vo= 8., 200.
    o= Orbit([10.*units.kpc,-20.*units.km/units.s,210.*units.km/units.s,
              45.*units.deg],
             ro=ro,vo=vo)
    oc= o()
    ts_nounits= numpy.linspace(0.,1.,1001)
    ts= units.Quantity(copy.copy(ts_nounits),unit=units.Myr)
    ts_nounits/= bovy_conversion.time_in_Gyr(vo,ro)*1000.
    # Integrate both with Quantity time and with unitless time
    o.integrate_dxdv([1.,0.3,0.4,0.2],ts,MWPotential,
                     rectIn=True,rectOut=True)
    oc.integrate_dxdv([1.,0.3,0.4,0.2],ts_nounits,MWPotential,
                      rectIn=True,rectOut=True)
    dx= o.getOrbit_dxdv()
    dxc= oc.getOrbit_dxdv()
    assert numpy.all(numpy.fabs(dx-dxc) < 10.**-8.), 'Orbit integrated_dxdv with times specified as Quantity does not agree with Orbit integrated_dxdv with time specified as array'
    return None

def test_change_ro_config():
    from galpy.orbit import Orbit
    from galpy.util import config
    o= Orbit([10.*units.kpc,-20.*units.km/units.s,210.*units.km/units.s,
              45.*units.deg])
    assert numpy.fabs(o._ro-8.) < 10.**-10., 'Default ro value not as expected'
    assert numpy.fabs(o._orb._ro-8.) < 10.**-10., 'Default ro value not as expected'
    # Change value
    newro= 9.
    config.set_ro(newro)
    o= Orbit([10.*units.kpc,-20.*units.km/units.s,210.*units.km/units.s,
              45.*units.deg])
    assert numpy.fabs(o._ro-newro) < 10.**-10., 'Default ro value not as expected'
    assert numpy.fabs(o._orb._ro-newro) < 10.**-10., 'Default ro value not as expected'
    # Change value as Quantity
    newro= 9.*units.kpc
    config.set_ro(newro)
    o= Orbit([10.*units.kpc,-20.*units.km/units.s,210.*units.km/units.s,
              45.*units.deg])
    assert numpy.fabs(o._ro-newro.value) < 10.**-10., 'Default ro value not as expected'
    assert numpy.fabs(o._orb._ro-newro.value) < 10.**-10., 'Default ro value not as expected'
    # Back to default
    config.set_ro(8.)
    return None

def test_change_vo_config():
    from galpy.orbit import Orbit
    from galpy.util import config
    o= Orbit([10.*units.kpc,-20.*units.km/units.s,210.*units.km/units.s,
              45.*units.deg])
    assert numpy.fabs(o._vo-220.) < 10.**-10., 'Default ro value not as expected'
    assert numpy.fabs(o._orb._vo-220.) < 10.**-10., 'Default ro value not as expected'
    # Change value
    newvo= 250.
    config.set_vo(newvo)
    o= Orbit([10.*units.kpc,-20.*units.km/units.s,210.*units.km/units.s,
              45.*units.deg])
    assert numpy.fabs(o._vo-newvo) < 10.**-10., 'Default ro value not as expected'
    assert numpy.fabs(o._orb._vo-newvo) < 10.**-10., 'Default ro value not as expected'
    # Change value as Quantity
    newvo= 250.*units.km/units.s
    config.set_vo(newvo)
    o= Orbit([10.*units.kpc,-20.*units.km/units.s,210.*units.km/units.s,
              45.*units.deg])
    assert numpy.fabs(o._vo-newvo.value) < 10.**-10., 'Default ro value not as expected'
    assert numpy.fabs(o._orb._vo-newvo.value) < 10.**-10., 'Default ro value not as expected'
    # Back to default
    config.set_vo(220.)
    return None

def test_potential_method_returntype():
    from galpy.potential import PlummerPotential
    pot= PlummerPotential(normalize=True,ro=8.,vo=220.)
    assert isinstance(pot(1.1,0.1),units.Quantity), 'Potential method __call__ does not return Quantity when it should'
    assert isinstance(pot.Rforce(1.1,0.1),units.Quantity), 'Potential method Rforce does not return Quantity when it should'
    assert isinstance(pot.zforce(1.1,0.1),units.Quantity), 'Potential method zforce does not return Quantity when it should'
    assert isinstance(pot.phiforce(1.1,0.1),units.Quantity), 'Potential method phiforce does not return Quantity when it should'
    assert isinstance(pot.dens(1.1,0.1),units.Quantity), 'Potential method dens does not return Quantity when it should'
    assert isinstance(pot.mass(1.1,0.1),units.Quantity), 'Potential method mass does not return Quantity when it should'
    assert isinstance(pot.R2deriv(1.1,0.1),units.Quantity), 'Potential method R2deriv does not return Quantity when it should'
    assert isinstance(pot.z2deriv(1.1,0.1),units.Quantity), 'Potential method z2deriv does not return Quantity when it should'
    assert isinstance(pot.Rzderiv(1.1,0.1),units.Quantity), 'Potential method Rzderiv does not return Quantity when it should'
    assert isinstance(pot.Rphideriv(1.1,0.1),units.Quantity), 'Potential method Rphideriv does not return Quantity when it should'
    assert isinstance(pot.phi2deriv(1.1,0.1),units.Quantity), 'Potential method phi2deriv does not return Quantity when it should'
    assert isinstance(pot.flattening(1.1,0.1),units.Quantity), 'Potential method flattening does not return Quantity when it should'
    assert isinstance(pot.vcirc(1.1),units.Quantity), 'Potential method vcirc does not return Quantity when it should'
    assert isinstance(pot.dvcircdR(1.1),units.Quantity), 'Potential method dvcircdR does not return Quantity when it should'
    assert isinstance(pot.omegac(1.1),units.Quantity), 'Potential method omegac does not return Quantity when it should'
    assert isinstance(pot.epifreq(1.1),units.Quantity), 'Potential method epifreq does not return Quantity when it should'
    assert isinstance(pot.verticalfreq(1.1),units.Quantity), 'Potential method verticalfreq does not return Quantity when it should'
    assert pot.lindbladR(0.9) is None, 'Potential method lindbladR does not return None, even when it should return a Quantity, when it should'
    assert isinstance(pot.lindbladR(0.9,m='corot'),units.Quantity), 'Potential method lindbladR does not return Quantity when it should'
    assert isinstance(pot.vesc(1.3),units.Quantity), 'Potential method vesc does not return Quantity when it should'
    assert isinstance(pot.rl(1.3),units.Quantity), 'Potential method rl does not return Quantity when it should'
    assert isinstance(pot.vterm(45.),units.Quantity), 'Potential method vterm does not return Quantity when it should'
    return None

def test_planarPotential_method_returntype():
    from galpy.potential import PlummerPotential
    pot= PlummerPotential(normalize=True,ro=8.,vo=220.).toPlanar()
    assert isinstance(pot(1.1),units.Quantity), 'Potential method __call__ does not return Quantity when it should'
    assert isinstance(pot.Rforce(1.1),units.Quantity), 'Potential method Rforce does not return Quantity when it should'
    assert isinstance(pot.phiforce(1.1),units.Quantity), 'Potential method phiforce does not return Quantity when it should'
    assert isinstance(pot.R2deriv(1.1),units.Quantity), 'Potential method R2deriv does not return Quantity when it should'
    assert isinstance(pot.Rphideriv(1.1),units.Quantity), 'Potential method Rphideriv does not return Quantity when it should'
    assert isinstance(pot.phi2deriv(1.1),units.Quantity), 'Potential method phi2deriv does not return Quantity when it should'
    assert isinstance(pot.vcirc(1.1),units.Quantity), 'Potential method vcirc does not return Quantity when it should'
    assert isinstance(pot.omegac(1.1),units.Quantity), 'Potential method omegac does not return Quantity when it should'
    assert isinstance(pot.epifreq(1.1),units.Quantity), 'Potential method epifreq does not return Quantity when it should'
    assert pot.lindbladR(0.9) is None, 'Potential method lindbladR does not return None, even when it should return a Quantity, when it should'
    assert isinstance(pot.lindbladR(0.9,m='corot'),units.Quantity), 'Potential method lindbladR does not return Quantity when it should'
    assert isinstance(pot.vesc(1.3),units.Quantity), 'Potential method vesc does not return Quantity when it should'
    return None

def test_linearPotential_method_returntype():
    from galpy.potential import PlummerPotential
    pot= PlummerPotential(normalize=True,ro=8.,vo=220.).toVertical(1.1)
    assert isinstance(pot(1.1),units.Quantity), 'Potential method __call__ does not return Quantity when it should'
    assert isinstance(pot.force(1.1),units.Quantity), 'Potential method Rforce does not return Quantity when it should'
    return None

def test_potential_method_returnunit():
    from galpy.potential import PlummerPotential
    pot= PlummerPotential(normalize=True,ro=8.,vo=220.)
    try:
        pot(1.1,0.1).to(units.km**2/units.s**2)
    except units.UnitConversionError:
        raise AssertionError('Potential method __call__ does not return Quantity with the right units')
    try:
        pot.Rforce(1.1,0.1).to(units.km/units.s**2)
    except units.UnitConversionError:
        raise AssertionError('Potential method Rforce does not return Quantity with the right units')
    try:
        pot.zforce(1.1,0.1).to(units.km/units.s**2)
    except units.UnitConversionError:
        raise AssertionError('Potential method zforce does not return Quantity with the right units')
    try:
        pot.phiforce(1.1,0.1).to(units.km/units.s**2)
    except units.UnitConversionError:
        raise AssertionError('Potential method phiforce does not return Quantity with the right units')
    try:
        pot.dens(1.1,0.1).to(units.kg/units.m**3)
    except units.UnitConversionError:
        raise AssertionError('Potential method dens does not return Quantity with the right units')
    try:
        pot.mass(1.1,0.1).to(units.kg)
    except units.UnitConversionError:
        raise AssertionError('Potential method mass does not return Quantity with the right units')
    try:
        pot.R2deriv(1.1,0.1).to(1/units.s**2)
    except units.UnitConversionError:
        raise AssertionError('Potential method R2deriv does not return Quantity with the right units')
    try:
        pot.z2deriv(1.1,0.1).to(1/units.s**2)
    except units.UnitConversionError:
        raise AssertionError('Potential method z2deriv does not return Quantity with the right units')
    try:
        pot.Rzderiv(1.1,0.1).to(1/units.s**2)
    except units.UnitConversionError:
        raise AssertionError('Potential method Rzderiv does not return Quantity with the right units')
    try:
        pot.phi2deriv(1.1,0.1).to(1/units.s**2)
    except units.UnitConversionError:
        raise AssertionError('Potential method phi2deriv does not return Quantity with the right units')
    try:
        pot.Rphideriv(1.1,0.1).to(1/units.s**2)
    except units.UnitConversionError:
        raise AssertionError('Potential method Rphideriv does not return Quantity with the right units')
    try:
        pot.flattening(1.1,0.1).to(units.dimensionless_unscaled)
    except units.UnitConversionError:
        raise AssertionError('Potential method flattening does not return Quantity with the right units')
    try:
        pot.vcirc(1.1).to(units.km/units.s)
    except units.UnitConversionError:
        raise AssertionError('Potential method vcirc does not return Quantity with the right units')
    try:
        pot.dvcircdR(1.1).to(1./units.s)
    except units.UnitConversionError:
        raise AssertionError('Potential method dvcircdR does not return Quantity with the right units')
    try:
        pot.omegac(1.1).to(1./units.s)
    except units.UnitConversionError:
        raise AssertionError('Potential method omegac does not return Quantity with the right units')
    try:
        pot.epifreq(1.1).to(1./units.s)
    except units.UnitConversionError:
        raise AssertionError('Potential method epifreq does not return Quantity with the right units')
    try:
        pot.verticalfreq(1.1).to(1./units.s)
    except units.UnitConversionError:
        raise AssertionError('Potential method verticalfreq does not return Quantity with the right units')
    try:
        pot.lindbladR(0.9,m='corot').to(units.km)
    except units.UnitConversionError:
        raise AssertionError('Potential method lindbladR does not return Quantity with the right units')
    try:
        pot.vesc(1.3).to(units.km/units.s)
    except units.UnitConversionError:
        raise AssertionError('Potential method vesc does not return Quantity with the right units')
    try:
        pot.rl(1.3).to(units.km)
    except units.UnitConversionError:
        raise AssertionError('Potential method rl does not return Quantity with the right units')
    try:
        pot.vterm(45.).to(units.km/units.s)
    except units.UnitConversionError:
        raise AssertionError('Potential method vter does not return Quantity with the right units')
    return None

def test_planarPotential_method_returnunit():
    from galpy.potential import PlummerPotential
    pot= PlummerPotential(normalize=True,ro=8.,vo=220.).toPlanar()
    try:
        pot(1.1).to(units.km**2/units.s**2)
    except units.UnitConversionError:
        raise AssertionError('Potential method __call__ does not return Quantity with the right units')
    try:
        pot.Rforce(1.1).to(units.km/units.s**2)
    except units.UnitConversionError:
        raise AssertionError('Potential method Rforce does not return Quantity with the right units')
    try:
        pot.phiforce(1.1).to(units.km/units.s**2)
    except units.UnitConversionError:
        raise AssertionError('Potential method phiforce does not return Quantity with the right units')
    try:
        pot.R2deriv(1.1).to(1/units.s**2)
    except units.UnitConversionError:
        raise AssertionError('Potential method R2deriv does not return Quantity with the right units')
    try:
        pot.phi2deriv(1.1).to(1/units.s**2)
    except units.UnitConversionError:
        raise AssertionError('Potential method phi2deriv does not return Quantity with the right units')
    try:
        pot.Rphideriv(1.1).to(1/units.s**2)
    except units.UnitConversionError:
        raise AssertionError('Potential method Rphideriv does not return Quantity with the right units')
    try:
        pot.vcirc(1.1).to(units.km/units.s)
    except units.UnitConversionError:
        raise AssertionError('Potential method vcirc does not return Quantity with the right units')
    try:
        pot.omegac(1.1).to(1./units.s)
    except units.UnitConversionError:
        raise AssertionError('Potential method omegac does not return Quantity with the right units')
    try:
        pot.epifreq(1.1).to(1./units.s)
    except units.UnitConversionError:
        raise AssertionError('Potential method epifreq does not return Quantity with the right units')
    try:
        pot.lindbladR(0.9,m='corot').to(units.km)
    except units.UnitConversionError:
        raise AssertionError('Potential method lindbladR does not return Quantity with the right units')
    try:
        pot.vesc(1.3).to(units.km/units.s)
    except units.UnitConversionError:
        raise AssertionError('Potential method vesc does not return Quantity with the right units')
    return None

def test_linearPotential_method_returnunit():
    from galpy.potential import PlummerPotential
    pot= PlummerPotential(normalize=True,ro=8.,vo=220.).toVertical(1.1)
    try:
        pot(1.1).to(units.km**2/units.s**2)
    except units.UnitConversionError:
        raise AssertionError('Potential method __call__ does not return Quantity with the right units')
    try:
        pot.force(1.1).to(units.km/units.s**2)
    except units.UnitConversionError:
        raise AssertionError('Potential method force does not return Quantity with the right units')
    return None

def test_potential_method_value():
    from galpy.potential import PlummerPotential
    from galpy.util import bovy_conversion
    ro, vo= 8., 220.
    pot= PlummerPotential(normalize=True,ro=ro,vo=vo)
    potu= PlummerPotential(normalize=True)
    assert numpy.fabs(pot(1.1,0.1).to(units.km**2/units.s**2).value-potu(1.1,0.1)*vo**2.) < 10.**-8., 'Potential method __call__ does not return the correct value as Quantity'
    assert numpy.fabs(pot.Rforce(1.1,0.1).to(units.km/units.s**2).value*10.**13.-potu.Rforce(1.1,0.1)*bovy_conversion.force_in_10m13kms2(vo,ro)) < 10.**-4., 'Potential method Rforce does not return the correct value as Quantity'
    assert numpy.fabs(pot.zforce(1.1,0.1).to(units.km/units.s**2).value*10.**13.-potu.zforce(1.1,0.1)*bovy_conversion.force_in_10m13kms2(vo,ro)) < 10.**-4., 'Potential method zforce does not return the correct value as Quantity'
    assert numpy.fabs(pot.phiforce(1.1,0.1).to(units.km/units.s**2).value*10.**13.-potu.phiforce(1.1,0.1)*bovy_conversion.force_in_10m13kms2(vo,ro)) < 10.**-4., 'Potential method phiforce does not return the correct value as Quantity'
    assert numpy.fabs(pot.dens(1.1,0.1).to(units.Msun/units.pc**3).value-potu.dens(1.1,0.1)*bovy_conversion.dens_in_msolpc3(vo,ro)) < 10.**-8., 'Potential method dens does not return the correct value as Quantity'
    assert numpy.fabs(pot.mass(1.1,0.1).to(units.Msun).value/10.**10.-potu.mass(1.1,0.1)*bovy_conversion.mass_in_1010msol(vo,ro)) < 10.**-8., 'Potential method mass does not return the correct value as Quantity'
    assert numpy.fabs(pot.R2deriv(1.1,0.1).to(units.km**2/units.s**2./units.kpc**2).value-potu.R2deriv(1.1,0.1)*vo**2./ro**2.) < 10.**-8., 'Potential method R2deriv does not return the correct value as Quantity'
    assert numpy.fabs(pot.z2deriv(1.1,0.1).to(units.km**2/units.s**2./units.kpc**2).value-potu.z2deriv(1.1,0.1)*vo**2./ro**2.) < 10.**-8., 'Potential method z2deriv does not return the correct value as Quantity'
    assert numpy.fabs(pot.Rzderiv(1.1,0.1).to(units.km**2/units.s**2./units.kpc**2).value-potu.Rzderiv(1.1,0.1)*vo**2./ro**2.) < 10.**-8., 'Potential method Rzderiv does not return the correct value as Quantity'
    assert numpy.fabs(pot.Rphideriv(1.1,0.1).to(units.km**2/units.s**2./units.kpc**2).value-potu.Rphideriv(1.1,0.1)*vo**2./ro**2.) < 10.**-8., 'Potential method Rphideriv does not return the correct value as Quantity'
    assert numpy.fabs(pot.phi2deriv(1.1,0.1).to(units.km**2/units.s**2./units.kpc**2).value-potu.phi2deriv(1.1,0.1)*vo**2./ro**2.) < 10.**-8., 'Potential method phi2deriv does not return the correct value as Quantity'
    assert numpy.fabs(pot.flattening(1.1,0.1).value-potu.flattening(1.1,0.1)) < 10.**-8., 'Potential method flattening does not return the correct value as Quantity'
    assert numpy.fabs(pot.vcirc(1.1).to(units.km/units.s).value-potu.vcirc(1.1)*vo) < 10.**-8., 'Potential method vcirc does not return the correct value as Quantity'
    assert numpy.fabs(pot.dvcircdR(1.1).to(units.km/units.s/units.kpc).value-potu.dvcircdR(1.1)*vo/ro) < 10.**-8., 'Potential method dvcircdR does not return the correct value as Quantity'
    assert numpy.fabs(pot.omegac(1.1).to(units.km/units.s/units.kpc).value-potu.omegac(1.1)*vo/ro) < 10.**-8., 'Potential method omegac does not return the correct value as Quantity'
    assert numpy.fabs(pot.epifreq(1.1).to(units.km/units.s/units.kpc).value-potu.epifreq(1.1)*vo/ro) < 10.**-8., 'Potential method epifreq does not return the correct value as Quantity'
    assert numpy.fabs(pot.verticalfreq(1.1).to(units.km/units.s/units.kpc).value-potu.verticalfreq(1.1)*vo/ro) < 10.**-8., 'Potential method verticalfreq does not return the correct value as Quantity'
    assert numpy.fabs(pot.lindbladR(0.9,m='corot').to(units.kpc).value-potu.lindbladR(0.9,m='corot')*ro) < 10.**-8., 'Potential method lindbladR does not return the correct value as Quantity'
    assert numpy.fabs(pot.vesc(1.1).to(units.km/units.s).value-potu.vesc(1.1)*vo) < 10.**-8., 'Potential method vesc does not return the correct value as Quantity'
    assert numpy.fabs(pot.rl(1.1).to(units.kpc).value-potu.rl(1.1)*ro) < 10.**-8., 'Potential method rl does not return the correct value as Quantity'
    assert numpy.fabs(pot.vterm(45.).to(units.km/units.s).value-potu.vterm(45.)*vo) < 10.**-8., 'Potential method vterm does not return the correct value as Quantity'
    return None

def test_planarPotential_method_value():
    from galpy.potential import PlummerPotential
    from galpy.util import bovy_conversion
    ro, vo= 8., 220.
    pot= PlummerPotential(normalize=True,ro=ro,vo=vo).toPlanar()
    potu= PlummerPotential(normalize=True).toPlanar()
    assert numpy.fabs(pot(1.1).to(units.km**2/units.s**2).value-potu(1.1)*vo**2.) < 10.**-8., 'Potential method __call__ does not return the correct value as Quantity'
    assert numpy.fabs(pot.Rforce(1.1).to(units.km/units.s**2).value*10.**13.-potu.Rforce(1.1)*bovy_conversion.force_in_10m13kms2(vo,ro)) < 10.**-4., 'Potential method Rforce does not return the correct value as Quantity'
    assert numpy.fabs(pot.phiforce(1.1).to(units.km/units.s**2).value*10.**13.-potu.phiforce(1.1)*bovy_conversion.force_in_10m13kms2(vo,ro)) < 10.**-4., 'Potential method phiforce does not return the correct value as Quantity'
    assert numpy.fabs(pot.R2deriv(1.1).to(units.km**2/units.s**2./units.kpc**2).value-potu.R2deriv(1.1)*vo**2./ro**2.) < 10.**-8., 'Potential method R2deriv does not return the correct value as Quantity'
    assert numpy.fabs(pot.Rphideriv(1.1).to(units.km**2/units.s**2./units.kpc**2).value-potu.Rphideriv(1.1)*vo**2./ro**2.) < 10.**-8., 'Potential method Rphideriv does not return the correct value as Quantity'
    assert numpy.fabs(pot.phi2deriv(1.1).to(units.km**2/units.s**2./units.kpc**2).value-potu.phi2deriv(1.1)*vo**2./ro**2.) < 10.**-8., 'Potential method phi2deriv does not return the correct value as Quantity'
    assert numpy.fabs(pot.vcirc(1.1).to(units.km/units.s).value-potu.vcirc(1.1)*vo) < 10.**-8., 'Potential method vcirc does not return the correct value as Quantity'
    assert numpy.fabs(pot.omegac(1.1).to(units.km/units.s/units.kpc).value-potu.omegac(1.1)*vo/ro) < 10.**-8., 'Potential method omegac does not return the correct value as Quantity'
    assert numpy.fabs(pot.epifreq(1.1).to(units.km/units.s/units.kpc).value-potu.epifreq(1.1)*vo/ro) < 10.**-8., 'Potential method epifreq does not return the correct value as Quantity'
    assert numpy.fabs(pot.vesc(1.1).to(units.km/units.s).value-potu.vesc(1.1)*vo) < 10.**-8., 'Potential method vesc does not return the correct value as Quantity'
    return None

def test_linearPotential_method_value():
    from galpy.potential import PlummerPotential
    from galpy.util import bovy_conversion
    ro, vo= 8., 220.
    pot= PlummerPotential(normalize=True,ro=ro,vo=vo).toVertical(1.1)
    potu= PlummerPotential(normalize=True).toVertical(1.1)
    assert numpy.fabs(pot(1.1).to(units.km**2/units.s**2).value-potu(1.1)*vo**2.) < 10.**-8., 'Potential method __call__ does not return the correct value as Quantity'
    assert numpy.fabs(pot.force(1.1).to(units.km/units.s**2).value*10.**13.-potu.force(1.1)*bovy_conversion.force_in_10m13kms2(vo,ro)) < 10.**-4., 'Potential method force does not return the correct value as Quantity'
    return None

def test_potential_function_returntype():
    from galpy.potential import PlummerPotential
    from galpy import potential
    pot= [PlummerPotential(normalize=True,ro=8.,vo=220.)]
    assert isinstance(potential.evaluatePotentials(pot,1.1,0.1),units.Quantity), 'Potential function __call__ does not return Quantity when it should'
    assert isinstance(potential.evaluateRforces(pot,1.1,0.1),units.Quantity), 'Potential function Rforce does not return Quantity when it should'
    assert isinstance(potential.evaluatezforces(pot,1.1,0.1),units.Quantity), 'Potential function zforce does not return Quantity when it should'
    assert isinstance(potential.evaluatephiforces(pot,1.1,0.1),units.Quantity), 'Potential function phiforce does not return Quantity when it should'
    assert isinstance(potential.evaluateDensities(pot,1.1,0.1),units.Quantity), 'Potential function dens does not return Quantity when it should'
    assert isinstance(potential.evaluateR2derivs(pot,1.1,0.1),units.Quantity), 'Potential function R2deriv does not return Quantity when it should'
    assert isinstance(potential.evaluatez2derivs(pot,1.1,0.1),units.Quantity), 'Potential function z2deriv does not return Quantity when it should'
    assert isinstance(potential.evaluateRzderivs(pot,1.1,0.1),units.Quantity), 'Potential function Rzderiv does not return Quantity when it should'
    assert isinstance(potential.flattening(pot,1.1,0.1),units.Quantity), 'Potential function flattening does not return Quantity when it should'
    assert isinstance(potential.vcirc(pot,1.1),units.Quantity), 'Potential function vcirc does not return Quantity when it should'
    assert isinstance(potential.dvcircdR(pot,1.1),units.Quantity), 'Potential function dvcircdR does not return Quantity when it should'
    assert isinstance(potential.omegac(pot,1.1),units.Quantity), 'Potential function omegac does not return Quantity when it should'
    assert isinstance(potential.epifreq(pot,1.1),units.Quantity), 'Potential function epifreq does not return Quantity when it should'
    assert isinstance(potential.verticalfreq(pot,1.1),units.Quantity), 'Potential function verticalfreq does not return Quantity when it should'
    assert potential.lindbladR(pot,0.9) is None, 'Potential function lindbladR does not return None, even when it should return a Quantity, when it should'
    assert isinstance(potential.lindbladR(pot,0.9,m='corot'),units.Quantity), 'Potential function lindbladR does not return Quantity when it should'
    assert isinstance(potential.vesc(pot,1.3),units.Quantity), 'Potential function vesc does not return Quantity when it should'
    assert isinstance(potential.rl(pot,1.3),units.Quantity), 'Potential function rl does not return Quantity when it should'
    assert isinstance(potential.vterm(pot,45.),units.Quantity), 'Potential function vterm does not return Quantity when it should'
    return None

def test_planarPotential_function_returntype():
    from galpy.potential import PlummerPotential
    from galpy import potential
    pot= [PlummerPotential(normalize=True,ro=8.,vo=220.).toPlanar()]
    assert isinstance(potential.evaluateplanarPotentials(pot,1.1),units.Quantity), 'Potential function __call__ does not return Quantity when it should'
    assert isinstance(potential.evaluateplanarRforces(pot,1.1),units.Quantity), 'Potential function Rforce does not return Quantity when it should'
    assert isinstance(potential.evaluateplanarphiforces(pot,1.1),units.Quantity), 'Potential function phiforce does not return Quantity when it should'
    assert isinstance(potential.evaluateplanarR2derivs(pot,1.1),units.Quantity), 'Potential function R2deriv does not return Quantity when it should'
    assert isinstance(potential.vcirc(pot,1.1),units.Quantity), 'Potential function vcirc does not return Quantity when it should'
    assert isinstance(potential.omegac(pot,1.1),units.Quantity), 'Potential function omegac does not return Quantity when it should'
    assert isinstance(potential.epifreq(pot,1.1),units.Quantity), 'Potential function epifreq does not return Quantity when it should'
    assert potential.lindbladR(pot,0.9) is None, 'Potential function lindbladR does not return None, even when it should return a Quantity, when it should'
    assert isinstance(potential.lindbladR(pot,0.9,m='corot'),units.Quantity), 'Potential function lindbladR does not return Quantity when it should'
    assert isinstance(potential.vesc(pot,1.3),units.Quantity), 'Potential function vesc does not return Quantity when it should'
    return None

def test_linearPotential_function_returntype():
    from galpy.potential import PlummerPotential
    from galpy import potential
    pot= [PlummerPotential(normalize=True,ro=8.,vo=220.).toVertical(1.1)]
    assert isinstance(potential.evaluatelinearPotentials(pot,1.1),units.Quantity), 'Potential function __call__ does not return Quantity when it should'
    assert isinstance(potential.evaluatelinearForces(pot,1.1),units.Quantity), 'Potential function Rforce does not return Quantity when it should'
    return None

def test_potential_function_returnunit():
    from galpy.potential import PlummerPotential
    from galpy import potential
    pot= [PlummerPotential(normalize=True,ro=8.,vo=220.)]
    try:
        potential.evaluatePotentials(pot,1.1,0.1).to(units.km**2/units.s**2)
    except units.UnitConversionError:
        raise AssertionError('Potential function __call__ does not return Quantity with the right units')
    try:
        potential.evaluateRforces(pot,1.1,0.1).to(units.km/units.s**2)
    except units.UnitConversionError:
        raise AssertionError('Potential function Rforce does not return Quantity with the right units')
    try:
        potential.evaluatezforces(pot,1.1,0.1).to(units.km/units.s**2)
    except units.UnitConversionError:
        raise AssertionError('Potential function zforce does not return Quantity with the right units')
    try:
        potential.evaluatephiforces(pot,1.1,0.1).to(units.km/units.s**2)
    except units.UnitConversionError:
        raise AssertionError('Potential function phiforce does not return Quantity with the right units')
    try:
        potential.evaluateDensities(pot,1.1,0.1).to(units.kg/units.m**3)
    except units.UnitConversionError:
        raise AssertionError('Potential function dens does not return Quantity with the right units')
    try:
        potential.evaluateR2derivs(pot,1.1,0.1).to(1/units.s**2)
    except units.UnitConversionError:
        raise AssertionError('Potential function R2deriv does not return Quantity with the right units')
    try:
        potential.evaluatez2derivs(pot,1.1,0.1).to(1/units.s**2)
    except units.UnitConversionError:
        raise AssertionError('Potential function z2deriv does not return Quantity with the right units')
    try:
        potential.evaluateRzderivs(pot,1.1,0.1).to(1/units.s**2)
    except units.UnitConversionError:
        raise AssertionError('Potential function Rzderiv does not return Quantity with the right units')
    try:
        potential.flattening(pot,1.1,0.1).to(units.dimensionless_unscaled)
    except units.UnitConversionError:
        raise AssertionError('Potential function flattening does not return Quantity with the right units')
    try:
        potential.vcirc(pot,1.1).to(units.km/units.s)
    except units.UnitConversionError:
        raise AssertionError('Potential function vcirc does not return Quantity with the right units')
    try:
        potential.dvcircdR(pot,1.1).to(1./units.s)
    except units.UnitConversionError:
        raise AssertionError('Potential function dvcircdR does not return Quantity with the right units')
    try:
        potential.omegac(pot,1.1).to(1./units.s)
    except units.UnitConversionError:
        raise AssertionError('Potential function omegac does not return Quantity with the right units')
    try:
        potential.epifreq(pot,1.1).to(1./units.s)
    except units.UnitConversionError:
        raise AssertionError('Potential function epifreq does not return Quantity with the right units')
    try:
        potential.verticalfreq(pot,1.1).to(1./units.s)
    except units.UnitConversionError:
        raise AssertionError('Potential function verticalfreq does not return Quantity with the right units')
    try:
        potential.lindbladR(pot,0.9,m='corot').to(units.km)
    except units.UnitConversionError:
        raise AssertionError('Potential function lindbladR does not return Quantity with the right units')
    try:
        potential.vesc(pot,1.3).to(units.km/units.s)
    except units.UnitConversionError:
        raise AssertionError('Potential function vesc does not return Quantity with the right units')
    try:
        potential.rl(pot,1.3).to(units.km)
    except units.UnitConversionError:
        raise AssertionError('Potential function rl does not return Quantity with the right units')
    try:
        potential.vterm(pot,45.).to(units.km/units.s)
    except units.UnitConversionError:
        raise AssertionError('Potential function vter does not return Quantity with the right units')
    return None

def test_planarPotential_function_returnunit():
    from galpy.potential import PlummerPotential, LopsidedDiskPotential
    from galpy import potential
    pot= [PlummerPotential(normalize=True,ro=8.,vo=220.).toPlanar(),
          LopsidedDiskPotential(ro=8.*units.kpc,vo=220.*units.km/units.s)]
    try:
        potential.evaluateplanarPotentials(pot,1.1,phi=0.1).to(units.km**2/units.s**2)
    except units.UnitConversionError:
        raise AssertionError('Potential function __call__ does not return Quantity with the right units')
    try:
        potential.evaluateplanarRforces(pot,1.1,phi=0.1).to(units.km/units.s**2)
    except units.UnitConversionError:
        raise AssertionError('Potential function Rforce does not return Quantity with the right units')
    try:
        potential.evaluateplanarphiforces(pot,1.1,phi=0.1).to(units.km/units.s**2)
    except units.UnitConversionError:
        raise AssertionError('Potential function phiforce does not return Quantity with the right units')
    try:
        potential.evaluateplanarR2derivs(pot,1.1,phi=0.1).to(1/units.s**2)
    except units.UnitConversionError:
        raise AssertionError('Potential function R2deriv does not return Quantity with the right units')
    pot.pop()
    try:
        potential.vcirc(pot,1.1).to(units.km/units.s)
    except units.UnitConversionError:
        raise AssertionError('Potential function vcirc does not return Quantity with the right units')
    try:
        potential.omegac(pot,1.1).to(1./units.s)
    except units.UnitConversionError:
        raise AssertionError('Potential function omegac does not return Quantity with the right units')
    try:
        potential.epifreq(pot,1.1).to(1./units.s)
    except units.UnitConversionError:
        raise AssertionError('Potential function epifreq does not return Quantity with the right units')
    try:
        potential.lindbladR(pot,0.9,m='corot').to(units.km)
    except units.UnitConversionError:
        raise AssertionError('Potential function lindbladR does not return Quantity with the right units')
    try:
        potential.vesc(pot,1.3).to(units.km/units.s)
    except units.UnitConversionError:
        raise AssertionError('Potential function vesc does not return Quantity with the right units')
    return None

def test_linearPotential_function_returnunit():
    from galpy.potential import KGPotential
    from galpy import potential
    pot= [KGPotential(ro=8.*units.kpc,vo=220.*units.km/units.s)]
    try:
        potential.evaluatelinearPotentials(pot,1.1).to(units.km**2/units.s**2)
    except units.UnitConversionError:
        raise AssertionError('Potential function __call__ does not return Quantity with the right units')
    try:
        potential.evaluatelinearForces(pot,1.1).to(units.km/units.s**2)
    except units.UnitConversionError:
        raise AssertionError('Potential function force does not return Quantity with the right units')
    return None

def test_potential_function_value():
    from galpy.potential import PlummerPotential
    from galpy import potential
    from galpy.util import bovy_conversion
    ro, vo= 8., 220.
    pot= [PlummerPotential(normalize=True,ro=ro,vo=vo)]
    potu= [PlummerPotential(normalize=True)]
    assert numpy.fabs(potential.evaluatePotentials(pot,1.1,0.1).to(units.km**2/units.s**2).value-potential.evaluatePotentials(potu,1.1,0.1)*vo**2.) < 10.**-8., 'Potential function __call__ does not return the correct value as Quantity'
    assert numpy.fabs(potential.evaluateRforces(pot,1.1,0.1).to(units.km/units.s**2).value*10.**13.-potential.evaluateRforces(potu,1.1,0.1)*bovy_conversion.force_in_10m13kms2(vo,ro)) < 10.**-4., 'Potential function Rforce does not return the correct value as Quantity'
    assert numpy.fabs(potential.evaluatezforces(pot,1.1,0.1).to(units.km/units.s**2).value*10.**13.-potential.evaluatezforces(potu,1.1,0.1)*bovy_conversion.force_in_10m13kms2(vo,ro)) < 10.**-4., 'Potential function zforce does not return the correct value as Quantity'
    assert numpy.fabs(potential.evaluatephiforces(pot,1.1,0.1).to(units.km/units.s**2).value*10.**13.-potential.evaluatephiforces(potu,1.1,0.1)*bovy_conversion.force_in_10m13kms2(vo,ro)) < 10.**-4., 'Potential function phiforce does not return the correct value as Quantity'
    assert numpy.fabs(potential.evaluateDensities(pot,1.1,0.1).to(units.Msun/units.pc**3).value-potential.evaluateDensities(potu,1.1,0.1)*bovy_conversion.dens_in_msolpc3(vo,ro)) < 10.**-8., 'Potential function dens does not return the correct value as Quantity'
    assert numpy.fabs(potential.evaluateR2derivs(pot,1.1,0.1).to(units.km**2/units.s**2./units.kpc**2).value-potential.evaluateR2derivs(potu,1.1,0.1)*vo**2./ro**2.) < 10.**-8., 'Potential function R2deriv does not return the correct value as Quantity'
    assert numpy.fabs(potential.evaluatez2derivs(pot,1.1,0.1).to(units.km**2/units.s**2./units.kpc**2).value-potential.evaluatez2derivs(potu,1.1,0.1)*vo**2./ro**2.) < 10.**-8., 'Potential function z2deriv does not return the correct value as Quantity'
    assert numpy.fabs(potential.evaluateRzderivs(pot,1.1,0.1).to(units.km**2/units.s**2./units.kpc**2).value-potential.evaluateRzderivs(potu,1.1,0.1)*vo**2./ro**2.) < 10.**-8., 'Potential function Rzderiv does not return the correct value as Quantity'
    assert numpy.fabs(potential.flattening(pot,1.1,0.1).value-potential.flattening(potu,1.1,0.1)) < 10.**-8., 'Potential function flattening does not return the correct value as Quantity'
    assert numpy.fabs(potential.vcirc(pot,1.1).to(units.km/units.s).value-potential.vcirc(potu,1.1)*vo) < 10.**-8., 'Potential function vcirc does not return the correct value as Quantity'
    assert numpy.fabs(potential.dvcircdR(pot,1.1).to(units.km/units.s/units.kpc).value-potential.dvcircdR(potu,1.1)*vo/ro) < 10.**-8., 'Potential function dvcircdR does not return the correct value as Quantity'
    assert numpy.fabs(potential.omegac(pot,1.1).to(units.km/units.s/units.kpc).value-potential.omegac(potu,1.1)*vo/ro) < 10.**-8., 'Potential function omegac does not return the correct value as Quantity'
    assert numpy.fabs(potential.epifreq(pot,1.1).to(units.km/units.s/units.kpc).value-potential.epifreq(potu,1.1)*vo/ro) < 10.**-8., 'Potential function epifreq does not return the correct value as Quantity'
    assert numpy.fabs(potential.verticalfreq(pot,1.1).to(units.km/units.s/units.kpc).value-potential.verticalfreq(potu,1.1)*vo/ro) < 10.**-8., 'Potential function verticalfreq does not return the correct value as Quantity'
    assert numpy.fabs(potential.lindbladR(pot,0.9,m='corot').to(units.kpc).value-potential.lindbladR(potu,0.9,m='corot')*ro) < 10.**-8., 'Potential function lindbladR does not return the correct value as Quantity'
    assert numpy.fabs(potential.vesc(pot,1.1).to(units.km/units.s).value-potential.vesc(potu,1.1)*vo) < 10.**-8., 'Potential function vesc does not return the correct value as Quantity'
    assert numpy.fabs(potential.rl(pot,1.1).to(units.kpc).value-potential.rl(potu,1.1)*ro) < 10.**-8., 'Potential function rl does not return the correct value as Quantity'
    assert numpy.fabs(potential.vterm(pot,45.).to(units.km/units.s).value-potential.vterm(potu,45.)*vo) < 10.**-8., 'Potential function vterm does not return the correct value as Quantity'
    return None

def test_planarPotential_function_value():
    from galpy.potential import PlummerPotential
    from galpy import potential
    from galpy.util import bovy_conversion
    ro, vo= 8., 220.
    pot= [PlummerPotential(normalize=True,ro=ro,vo=vo).toPlanar()]
    potu= [PlummerPotential(normalize=True).toPlanar()]
    assert numpy.fabs(potential.evaluateplanarPotentials(pot,1.1).to(units.km**2/units.s**2).value-potential.evaluateplanarPotentials(potu,1.1)*vo**2.) < 10.**-8., 'Potential function __call__ does not return the correct value as Quantity'
    assert numpy.fabs(potential.evaluateplanarRforces(pot,1.1).to(units.km/units.s**2).value*10.**13.-potential.evaluateplanarRforces(potu,1.1)*bovy_conversion.force_in_10m13kms2(vo,ro)) < 10.**-4., 'Potential function Rforce does not return the correct value as Quantity'
    assert numpy.fabs(potential.evaluateplanarphiforces(pot,1.1).to(units.km/units.s**2).value*10.**13.-potential.evaluateplanarphiforces(potu,1.1)*bovy_conversion.force_in_10m13kms2(vo,ro)) < 10.**-4., 'Potential function phiforce does not return the correct value as Quantity'
    assert numpy.fabs(potential.evaluateplanarR2derivs(pot,1.1).to(units.km**2/units.s**2./units.kpc**2).value-potential.evaluateplanarR2derivs(potu,1.1)*vo**2./ro**2.) < 10.**-8., 'Potential function R2deriv does not return the correct value as Quantity'
    assert numpy.fabs(potential.vcirc(pot,1.1).to(units.km/units.s).value-potential.vcirc(potu,1.1)*vo) < 10.**-8., 'Potential function vcirc does not return the correct value as Quantity'
    assert numpy.fabs(potential.omegac(pot,1.1).to(units.km/units.s/units.kpc).value-potential.omegac(potu,1.1)*vo/ro) < 10.**-8., 'Potential function omegac does not return the correct value as Quantity'
    assert numpy.fabs(potential.epifreq(pot,1.1).to(units.km/units.s/units.kpc).value-potential.epifreq(potu,1.1)*vo/ro) < 10.**-8., 'Potential function epifreq does not return the correct value as Quantity'
    assert numpy.fabs(potential.vesc(pot,1.1).to(units.km/units.s).value-potential.vesc(potu,1.1)*vo) < 10.**-8., 'Potential function vesc does not return the correct value as Quantity'
    return None

def test_linearPotential_function_value():
    from galpy.potential import PlummerPotential
    from galpy import potential
    from galpy.util import bovy_conversion
    ro, vo= 8., 220.
    pot= [PlummerPotential(normalize=True,ro=ro,vo=vo).toVertical(1.1)]
    potu= [PlummerPotential(normalize=True).toVertical(1.1)]
    assert numpy.fabs(potential.evaluatelinearPotentials(pot,1.1).to(units.km**2/units.s**2).value-potential.evaluatelinearPotentials(potu,1.1)*vo**2.) < 10.**-8., 'Potential function __call__ does not return the correct value as Quantity'
    assert numpy.fabs(potential.evaluatelinearForces(pot,1.1).to(units.km/units.s**2).value*10.**13.-potential.evaluatelinearForces(potu,1.1)*bovy_conversion.force_in_10m13kms2(vo,ro)) < 10.**-4., 'Potential function force does not return the correct value as Quantity'
    return None

def test_potential_method_inputAsQuantity():
    from galpy.potential import PlummerPotential
    from galpy.util import bovy_conversion
    ro, vo= 8.*units.kpc, 220.
    pot= PlummerPotential(normalize=True,ro=ro,vo=vo)
    potu= PlummerPotential(normalize=True)
    assert numpy.fabs(pot(1.1*ro,0.1*ro,phi=10.*units.deg,t=10.*units.Gyr,use_physical=False)-potu(1.1,0.1)) < 10.**-8., 'Potential method __call__ does not return the correct value when input is Quantity'
    assert numpy.fabs(pot.Rforce(1.1*ro,0.1*ro,phi=10.*units.deg,t=10.*units.Gyr,use_physical=False)-potu.Rforce(1.1,0.1)) < 10.**-4., 'Potential method Rforce does not return the correct value when input is Quantity'
    # Few more cases for Rforce
    assert numpy.fabs(pot.Rforce(1.1*ro,0.1*ro,phi=10.*units.deg,t=10.*units.Gyr,ro=9.,use_physical=False)-potu.Rforce(1.1*8./9.,0.1*8./9.)) < 10.**-4., 'Potential method Rforce does not return the correct value when input is Quantity'
    assert numpy.fabs(pot.Rforce(1.1*ro,0.1*ro,phi=10.*units.deg,t=10.*units.Gyr,vo=230.,use_physical=False)-potu.Rforce(1.1,0.1)) < 10.**-4., 'Potential method Rforce does not return the correct value when input is Quantity'
    assert numpy.fabs(pot.zforce(1.1*ro,0.1*ro,phi=10.*units.deg,t=10.*units.Gyr,use_physical=False)-potu.zforce(1.1,0.1)) < 10.**-4., 'Potential method zforce does not return the correct value when input is Quantity'
    assert numpy.fabs(pot.phiforce(1.1*ro,0.1*ro,phi=10.*units.deg,t=10.*units.Gyr,use_physical=False)-potu.phiforce(1.1,0.1)) < 10.**-4., 'Potential method phiforce does not return the correct value when input is Quantity'
    assert numpy.fabs(pot.dens(1.1*ro,0.1*ro,phi=10.*units.deg,t=10.*units.Gyr,use_physical=False)-potu.dens(1.1,0.1)) < 10.**-8., 'Potential method dens does not return the correct value when input is Quantity'
    assert numpy.fabs(pot.mass(1.1*ro,0.1*ro,use_physical=False)-potu.mass(1.1,0.1)) < 10.**-8., 'Potential method mass does not return the correct value when input is Quantity'
    assert numpy.fabs(pot.R2deriv(1.1*ro,0.1*ro,phi=10.*units.deg,t=10.*units.Gyr,use_physical=False)-potu.R2deriv(1.1,0.1)) < 10.**-8., 'Potential method R2deriv does not return the correct value when input is Quantity'
    assert numpy.fabs(pot.z2deriv(1.1*ro,0.1*ro,phi=10.*units.deg,t=10.*units.Gyr,use_physical=False)-potu.z2deriv(1.1,0.1)) < 10.**-8., 'Potential method z2deriv does not return the correct value when input is Quantity'
    assert numpy.fabs(pot.Rzderiv(1.1*ro,0.1*ro,phi=10.*units.deg,t=10.*units.Gyr,use_physical=False)-potu.Rzderiv(1.1,0.1)) < 10.**-8., 'Potential method Rzderiv does not return the correct value when input is Quantity'
    assert numpy.fabs(pot.Rphideriv(1.1*ro,0.1*ro,phi=10.*units.deg,t=10.*units.Gyr,use_physical=False)-potu.Rphideriv(1.1,0.1)) < 10.**-8., 'Potential method Rphideriv does not return the correct value when input is Quantity'
    assert numpy.fabs(pot.phi2deriv(1.1*ro,0.1*ro,phi=10.*units.deg,t=10.*units.Gyr,use_physical=False)-potu.phi2deriv(1.1,0.1)) < 10.**-8., 'Potential method phi2deriv does not return the correct value when input is Quantity'
    assert numpy.fabs(pot.flattening(1.1*ro,0.1*ro,use_physical=False)-potu.flattening(1.1,0.1)) < 10.**-8., 'Potential method flattening does not return the correct value when input is Quantity'
    assert numpy.fabs(pot.vcirc(1.1*ro,use_physical=False)-potu.vcirc(1.1)) < 10.**-8., 'Potential method vcirc does not return the correct value when input is Quantity'
    assert numpy.fabs(pot.dvcircdR(1.1*ro,use_physical=False)-potu.dvcircdR(1.1)) < 10.**-8., 'Potential method dvcircdR does not return the correct value when input is Quantity'
    assert numpy.fabs(pot.omegac(1.1*ro,use_physical=False)-potu.omegac(1.1)) < 10.**-8., 'Potential method omegac does not return the correct value when input is Quantity'
    assert numpy.fabs(pot.epifreq(1.1*ro,use_physical=False)-potu.epifreq(1.1)) < 10.**-8., 'Potential method epifreq does not return the correct value when input is Quantity'
    assert numpy.fabs(pot.verticalfreq(1.1*ro,use_physical=False)-potu.verticalfreq(1.1)) < 10.**-8., 'Potential method verticalfreq does not return the correct value when input is Quantity'
    assert numpy.fabs(pot.vesc(1.1*ro,use_physical=False)-potu.vesc(1.1)) < 10.**-8., 'Potential method vesc does not return the correct value when input is Quantity'
    assert numpy.fabs(pot.lindbladR(0.9*bovy_conversion.freq_in_Gyr(vo,ro.value)/units.Gyr,m='corot',use_physical=False)-potu.lindbladR(0.9,m='corot')) < 10.**-8., 'Potential method lindbladR does not return the correct value when input is Quantity'
    assert numpy.fabs(pot.rl(1.1*vo*ro*units.km/units.s,use_physical=False)-potu.rl(1.1)) < 10.**-8., 'Potential function rl does not return the correct value when input is Quantity'
    assert numpy.fabs(pot.vterm(45.*units.deg,use_physical=False)-potu.vterm(45.)) < 10.**-8., 'Potential function vterm does not return the correct value when input is Quantity'
    return None

def test_planarPotential_method_inputAsQuantity():
    from galpy.potential import PlummerPotential
    from galpy.util import bovy_conversion
    ro, vo= 8.*units.kpc, 220.
    pot= PlummerPotential(normalize=True,ro=ro,vo=vo)
    # Force planarPotential setup with default
    pot._ro= None
    pot._roSet= False
    pot._vo= None
    pot._voSet= False
    pot= pot.toPlanar()
    potu= PlummerPotential(normalize=True).toPlanar()
    assert numpy.fabs(pot(1.1*ro,use_physical=False)-potu(1.1)) < 10.**-8., 'Potential method __call__ does not return the correct value as Quantity'
    assert numpy.fabs(pot.Rforce(1.1*ro,use_physical=False)-potu.Rforce(1.1)) < 10.**-4., 'Potential method Rforce does not return the correct value as Quantity'
    assert numpy.fabs(pot.phiforce(1.1*ro,use_physical=False)-potu.phiforce(1.1)) < 10.**-4., 'Potential method phiforce does not return the correct value as Quantity'
    assert numpy.fabs(pot.R2deriv(1.1*ro,use_physical=False)-potu.R2deriv(1.1)) < 10.**-8., 'Potential method R2deriv does not return the correct value as Quantity'
    assert numpy.fabs(pot.Rphideriv(1.1*ro,use_physical=False)-potu.Rphideriv(1.1)) < 10.**-8., 'Potential method Rphideriv does not return the correct value as Quantity'
    assert numpy.fabs(pot.phi2deriv(1.1*ro,use_physical=False)-potu.phi2deriv(1.1)) < 10.**-8., 'Potential method phi2deriv does not return the correct value as Quantity'
    assert numpy.fabs(pot.vcirc(1.1*ro,use_physical=False)-potu.vcirc(1.1)) < 10.**-8., 'Potential method vcirc does not return the correct value as Quantity'
    assert numpy.fabs(pot.omegac(1.1*ro,use_physical=False)-potu.omegac(1.1)) < 10.**-8., 'Potential method omegac does not return the correct value as Quantity'
    assert numpy.fabs(pot.epifreq(1.1*ro,use_physical=False)-potu.epifreq(1.1)) < 10.**-8., 'Potential method epifreq does not return the correct value as Quantity'
    assert numpy.fabs(pot.vesc(1.1*ro,use_physical=False)-potu.vesc(1.1)) < 10.**-8., 'Potential method vesc does not return the correct value as Quantity'
    assert numpy.fabs(pot.lindbladR(0.9*bovy_conversion.freq_in_Gyr(vo,ro.value)/units.Gyr,m='corot',use_physical=False)-potu.lindbladR(0.9,m='corot')) < 10.**-8., 'Potential method lindbladR does not return the correct value when input is Quantity'
    return None

def test_linearPotential_method_inputAsQuantity():
    from galpy.potential import PlummerPotential
    from galpy import potential
    ro, vo= 8.*units.kpc, 220.*units.km/units.s
    pot= PlummerPotential(normalize=True,ro=ro,vo=vo)
    # Force linearPotential setup with default
    pot._ro= None
    pot._roSet= False
    pot._vo= None
    pot._voSet= False
    pot= pot.toVertical(1.1)
    potu= potential.RZToverticalPotential(PlummerPotential(normalize=True),
                                          1.1*ro)
    assert numpy.fabs(pot(1.1*ro,use_physical=False)-potu(1.1)) < 10.**-8., 'Potential method __call__ does not return the correct value as Quantity'
    assert numpy.fabs(pot.force(1.1*ro,use_physical=False)-potu.force(1.1)) < 10.**-4., 'Potential method force does not return the correct value as Quantity'
    return None

def test_potential_function_inputAsQuantity():
    from galpy.potential import PlummerPotential
    from galpy.util import bovy_conversion
    from galpy import potential
    ro, vo= 8.*units.kpc, 220.
    pot= [PlummerPotential(normalize=True,ro=ro,vo=vo)]
    potu= [PlummerPotential(normalize=True)]
    assert numpy.fabs(potential.evaluatePotentials(pot,1.1*ro,0.1*ro,phi=10.*units.deg,t=10.*units.Gyr,use_physical=False)-potential.evaluatePotentials(potu,1.1,0.1)) < 10.**-8., 'Potential function __call__ does not return the correct value when input is Quantity'
    assert numpy.fabs(potential.evaluateRforces(pot,1.1*ro,0.1*ro,phi=10.*units.deg,t=10.*units.Gyr,ro=8.*units.kpc,vo=220.*units.km/units.s,use_physical=False)-potential.evaluateRforces(potu,1.1,0.1)) < 10.**-4., 'Potential function Rforce does not return the correct value when input is Quantity'
    assert numpy.fabs(potential.evaluatezforces(pot,1.1*ro,0.1*ro,phi=10.*units.deg,t=10.*units.Gyr,use_physical=False)-potential.evaluatezforces(potu,1.1,0.1)) < 10.**-4., 'Potential function zforce does not return the correct value when input is Quantity'
    assert numpy.fabs(potential.evaluatephiforces(pot,1.1*ro,0.1*ro,phi=10.*units.deg,t=10.*units.Gyr,use_physical=False)-potential.evaluatephiforces(potu,1.1,0.1)) < 10.**-4., 'Potential function phiforce does not return the correct value when input is Quantity'
    assert numpy.fabs(potential.evaluateDensities(pot,1.1*ro,0.1*ro,phi=10.*units.deg,t=10.*units.Gyr,use_physical=False)-potential.evaluateDensities(potu,1.1,0.1)) < 10.**-8., 'Potential function dens does not return the correct value when input is Quantity'
    assert numpy.fabs(potential.evaluateR2derivs(pot,1.1*ro,0.1*ro,phi=10.*units.deg,t=10.*units.Gyr,use_physical=False)-potential.evaluateR2derivs(potu,1.1,0.1)) < 10.**-8., 'Potential function R2deriv does not return the correct value when input is Quantity'
    assert numpy.fabs(potential.evaluatez2derivs(pot,1.1*ro,0.1*ro,phi=10.*units.deg,t=10.*units.Gyr,use_physical=False)-potential.evaluatez2derivs(potu,1.1,0.1)) < 10.**-8., 'Potential function z2deriv does not return the correct value when input is Quantity'
    assert numpy.fabs(potential.evaluateRzderivs(pot,1.1*ro,0.1*ro,phi=10.*units.deg,t=10.*units.Gyr,use_physical=False)-potential.evaluateRzderivs(potu,1.1,0.1)) < 10.**-8., 'Potential function Rzderiv does not return the correct value when input is Quantity'
    assert numpy.fabs(potential.flattening(pot,1.1*ro,0.1*ro,use_physical=False)-potential.flattening(potu,1.1,0.1)) < 10.**-8., 'Potential function flattening does not return the correct value when input is Quantity'
    assert numpy.fabs(potential.vcirc(pot,1.1*ro,use_physical=False)-potential.vcirc(potu,1.1)) < 10.**-8., 'Potential function vcirc does not return the correct value when input is Quantity'
    assert numpy.fabs(potential.dvcircdR(pot,1.1*ro,use_physical=False)-potential.dvcircdR(potu,1.1)) < 10.**-8., 'Potential function dvcircdR does not return the correct value when input is Quantity'
    assert numpy.fabs(potential.omegac(pot,1.1*ro,use_physical=False)-potential.omegac(potu,1.1)) < 10.**-8., 'Potential function omegac does not return the correct value when input is Quantity'
    assert numpy.fabs(potential.epifreq(pot,1.1*ro,use_physical=False)-potential.epifreq(potu,1.1)) < 10.**-8., 'Potential function epifreq does not return the correct value when input is Quantity'
    assert numpy.fabs(potential.verticalfreq(pot,1.1*ro,use_physical=False)-potential.verticalfreq(potu,1.1)) < 10.**-8., 'Potential function verticalfreq does not return the correct value when input is Quantity'
    assert numpy.fabs(potential.vesc(pot,1.1*ro,use_physical=False)-potential.vesc(potu,1.1)) < 10.**-8., 'Potential function vesc does not return the correct value when input is Quantity'
    assert numpy.fabs(potential.lindbladR(pot,0.9*bovy_conversion.freq_in_Gyr(vo,ro.value)/units.Gyr,m='corot',use_physical=False)-potential.lindbladR(potu,0.9,m='corot')) < 10.**-8., 'Potential method lindbladR does not return the correct value when input is Quantity'
    assert numpy.fabs(potential.lindbladR(pot[0],0.9*bovy_conversion.freq_in_Gyr(vo,ro.value)/units.Gyr,m='corot',use_physical=False)-potential.lindbladR(potu,0.9,m='corot')) < 10.**-8., 'Potential method lindbladR does not return the correct value when input is Quantity'
    assert numpy.fabs(potential.rl(pot,1.1*vo*ro*units.km/units.s,use_physical=False)-potential.rl(potu,1.1)) < 10.**-8., 'Potential function rl does not return the correct value when input is Quantity'
    assert numpy.fabs(potential.rl(pot[0],1.1*vo*ro*units.km/units.s,use_physical=False)-potential.rl(potu,1.1)) < 10.**-8., 'Potential function rl does not return the correct value when input is Quantity'
    assert numpy.fabs(potential.vterm(pot,45.*units.deg,use_physical=False)-potential.vterm(potu,45.)) < 10.**-8., 'Potential function vterm does not return the correct value when input is Quantity'
    return None

def test_planarPotential_function_inputAsQuantity():
    from galpy.potential import PlummerPotential
    from galpy import potential
    ro, vo= 8.*units.kpc, 220.
    pot= [PlummerPotential(normalize=True,ro=ro,vo=vo).toPlanar()]
    potu= [PlummerPotential(normalize=True).toPlanar()]
    assert numpy.fabs(potential.evaluateplanarPotentials(pot,1.1*ro,use_physical=False)-potential.evaluateplanarPotentials(potu,1.1)) < 10.**-8., 'Potential function __call__ does not return the correct value as Quantity'
    assert numpy.fabs(potential.evaluateplanarRforces(pot,1.1*ro,use_physical=False)-potential.evaluateplanarRforces(potu,1.1)) < 10.**-4., 'Potential function Rforce does not return the correct value as Quantity'
    assert numpy.fabs(potential.evaluateplanarphiforces(pot,1.1*ro,use_physical=False)-potential.evaluateplanarphiforces(potu,1.1)) < 10.**-4., 'Potential function phiforce does not return the correct value as Quantity'
    assert numpy.fabs(potential.evaluateplanarR2derivs(pot,1.1*ro,use_physical=False)-potential.evaluateplanarR2derivs(potu,1.1)) < 10.**-8., 'Potential function R2deriv does not return the correct value as Quantity'
    assert numpy.fabs(potential.vcirc(pot,1.1*ro,use_physical=False)-potential.vcirc(potu,1.1)) < 10.**-8., 'Potential function vcirc does not return the correct value as Quantity'
    assert numpy.fabs(potential.omegac(pot,1.1*ro,use_physical=False)-potential.omegac(potu,1.1)) < 10.**-8., 'Potential function omegac does not return the correct value as Quantity'
    assert numpy.fabs(potential.epifreq(pot,1.1*ro,use_physical=False)-potential.epifreq(potu,1.1)) < 10.**-8., 'Potential function epifreq does not return the correct value as Quantity'
    assert numpy.fabs(potential.vesc(pot,1.1*ro,use_physical=False)-potential.vesc(potu,1.1)) < 10.**-8., 'Potential function vesc does not return the correct value as Quantity'
    return None

def test_linearPotential_function_inputAsQuantity():
    from galpy.potential import PlummerPotential
    from galpy import potential
    ro, vo= 8.*units.kpc, 220.
    pot= [PlummerPotential(normalize=True,ro=ro,vo=vo).toVertical(1.1*ro)]
    potu= potential.RZToverticalPotential([PlummerPotential(normalize=True)],
                                          1.1*ro)
    assert numpy.fabs(potential.evaluatelinearPotentials(pot,1.1*ro,use_physical=False)-potential.evaluatelinearPotentials(potu,1.1)) < 10.**-8., 'Potential function __call__ does not return the correct value as Quantity'
    assert numpy.fabs(potential.evaluatelinearForces(pot,1.1*ro,use_physical=False)-potential.evaluatelinearForces(potu,1.1)) < 10.**-4., 'Potential function force does not return the correct value as Quantity'
    return None

def test_plotting_inputAsQuantity():
    from galpy.potential import PlummerPotential
    from galpy import potential
    ro, vo= 8.*units.kpc, 220.
    pot= PlummerPotential(normalize=True,ro=ro,vo=vo)
    pot.plot(rmin=1.*units.kpc,rmax=4.*units.kpc,
             zmin=-4.*units.kpc,zmax=4.*units.kpc)
    pot.plotDensity(rmin=1.*units.kpc,rmax=4.*units.kpc,
                    zmin=-4.*units.kpc,zmax=4.*units.kpc)
    potential.plotPotentials(pot,rmin=1.*units.kpc,rmax=4.*units.kpc,
                             zmin=-4.*units.kpc,zmax=4.*units.kpc)
    potential.plotPotentials([pot],rmin=1.*units.kpc,rmax=4.*units.kpc,
                             zmin=-4.*units.kpc,zmax=4.*units.kpc)
    potential.plotDensities(pot,rmin=1.*units.kpc,rmax=4.*units.kpc,
                            zmin=-4.*units.kpc,zmax=4.*units.kpc)
    potential.plotDensities([pot],rmin=1.*units.kpc,rmax=4.*units.kpc,
                             zmin=-4.*units.kpc,zmax=4.*units.kpc)
    # Planar
    plpot= pot.toPlanar()
    plpot.plot(Rrange=[1.*units.kpc,8.*units.kpc],
               xrange=[-4.*units.kpc,4.*units.kpc],
               yrange=[-6.*units.kpc,7.*units.kpc])
    potential.plotplanarPotentials(plpot,
                                   Rrange=[1.*units.kpc,8.*units.kpc],
                                   xrange=[-4.*units.kpc,4.*units.kpc],
                                   yrange=[-6.*units.kpc,7.*units.kpc])
    potential.plotplanarPotentials([plpot],
                                   Rrange=[1.*units.kpc,8.*units.kpc],
                                   xrange=[-4.*units.kpc,4.*units.kpc],
                                   yrange=[-6.*units.kpc,7.*units.kpc])
    # Rotcurve
    pot.plotRotcurve(Rrange=[1.*units.kpc,8.*units.kpc],ro=10.,vo=250.)
    plpot.plotRotcurve(Rrange=[1.*units.kpc,8.*units.kpc],
                       ro=10.*units.kpc,vo=250.*units.km/units.s)
    potential.plotRotcurve(pot,Rrange=[1.*units.kpc,8.*units.kpc])
    potential.plotRotcurve([pot],Rrange=[1.*units.kpc,8.*units.kpc])
    # Escapecurve
    pot.plotEscapecurve(Rrange=[1.*units.kpc,8.*units.kpc],ro=10.,vo=250.)
    plpot.plotEscapecurve(Rrange=[1.*units.kpc,8.*units.kpc],
                          ro=10.*units.kpc,vo=250.*units.km/units.s)
    potential.plotEscapecurve(pot,Rrange=[1.*units.kpc,8.*units.kpc])
    potential.plotEscapecurve([pot],Rrange=[1.*units.kpc,8.*units.kpc])
    return None

def test_potential_ampunits():
    # Test that input units for potential amplitudes behave as expected
    from galpy import potential
    from galpy.util import bovy_conversion
    ro, vo= 9., 210.
    # Burkert
    pot= potential.BurkertPotential(amp=0.1*units.Msun/units.pc**3.,
                                    a=2.,ro=ro,vo=vo)
    # density at r=a should be amp/4
    assert numpy.fabs(pot.dens(2.,0.,use_physical=False)*bovy_conversion.dens_in_msolpc3(vo,ro)-0.1/4.) < 10.**-8., "BurkertPotential w/ amp w/ units does not behave as expected"
    # DoubleExponentialDiskPotential
    pot= potential.DoubleExponentialDiskPotential(\
        amp=0.1*units.Msun/units.pc**3.,hr=2.,hz=0.2,ro=ro,vo=vo)
    # density at zero should be amp
    assert numpy.fabs(pot.dens(0.,0.,use_physical=False)*bovy_conversion.dens_in_msolpc3(vo,ro)-0.1) < 10.**-8., "DoubleExponentialDiskPotential w/ amp w/ units does not behave as expected"
    # TwoPowerSphericalPotential
    pot= potential.TwoPowerSphericalPotential(amp=20.*units.Msun,a=2.,
                                              alpha=1.5,beta=3.5,ro=ro,vo=vo)
    # Check density at r=a
    assert numpy.fabs(pot.dens(2.,0.,use_physical=False)*bovy_conversion.dens_in_msolpc3(vo,ro)-20./4./numpy.pi/8./ro**3./10.**9./4.) < 10.**-8., "TwoPowerSphericalPotential w/ amp w/ units does not behave as expected"
    # TwoPowerSphericalPotential with integer powers
    pot= potential.TwoPowerSphericalPotential(amp=20.*units.Msun,a=2.,
                                              alpha=2.,beta=5.,ro=ro,vo=vo)
    # Check density at r=a
    assert numpy.fabs(pot.dens(2.,0.,use_physical=False)*bovy_conversion.dens_in_msolpc3(vo,ro)-20./4./numpy.pi/8./ro**3./10.**9./8.) < 10.**-8., "TwoPowerSphericalPotential w/ amp w/ units does not behave as expected"
    # JaffePotential
    pot= potential.JaffePotential(amp=20.*units.Msun,a=2.,ro=ro,vo=vo)
    # Check density at r=a
    assert numpy.fabs(pot.dens(2.,0.,use_physical=False)*bovy_conversion.dens_in_msolpc3(vo,ro)-20./4./numpy.pi/8./ro**3./10.**9./4.) < 10.**-8., "JaffePotential w/ amp w/ units does not behave as expected"
    # HernquistPotential
    pot= potential.HernquistPotential(amp=20.*units.Msun,a=2.,ro=ro,vo=vo)
    # Check density at r=a
    assert numpy.fabs(pot.dens(2.,0.,use_physical=False)*bovy_conversion.dens_in_msolpc3(vo,ro)-20./4./numpy.pi/8./ro**3./10.**9./8.) < 10.**-8., "HernquistPotential w/ amp w/ units does not behave as expected"
    # NFWPotential
    pot= potential.NFWPotential(amp=20.*units.Msun,a=2.,ro=ro,vo=vo)
    # Check density at r=a
    assert numpy.fabs(pot.dens(2.,0.,use_physical=False)*bovy_conversion.dens_in_msolpc3(vo,ro)-20./4./numpy.pi/8./ro**3./10.**9./4.) < 10.**-8., "NFWPotential w/ amp w/ units does not behave as expected"
    # FlattenedPowerPotential
    pot= potential.FlattenedPowerPotential(amp=40000.*units.km**2/units.s**2,
                                           r1=1.,q=0.9,alpha=0.5,core=0.,
                                           ro=ro,vo=vo)
    # Check potential
    assert numpy.fabs(pot(2.,1.,use_physical=False)*vo**2.+40000./0.5/(2.**2.+(1./0.9)**2.)**0.25) < 10.**-8., "FlattenedPowerPotential w/ amp w/ units does not behave as expected"
    # IsochronePotential
    pot= potential.IsochronePotential(amp=20.*units.Msun,b=2.,ro=ro,vo=vo)
    # Check potential
    assert numpy.fabs(pot(4.,0.,use_physical=False)*vo**2.+(20.*units.Msun*constants.G).to(units.pc*units.km**2/units.s**2).value/(2.+numpy.sqrt(4.+16.))/ro/1000.) < 10.**-8., "IsochronePotential w/ amp w/ units does not behave as expected"   
    # KeplerPotential
    pot= potential.KeplerPotential(amp=20.*units.Msun,ro=ro,vo=vo)
    # Check mass
    assert numpy.fabs(pot.mass(100.,use_physical=False)*bovy_conversion.mass_in_msol(vo,ro)-20.) < 10.**-8., "KeplerPotential w/ amp w/ units does not behave as expected"   
    # KuzminKutuzovStaeckelPotential
    pot= potential.KuzminKutuzovStaeckelPotential(amp=20.*units.Msun,
                                                  Delta=2.,ro=ro,vo=vo)
    pot_nounits= potential.KuzminKutuzovStaeckelPotential(\
        amp=(20.*units.Msun*constants.G).to(units.kpc*units.km**2/units.s**2).value/ro/vo**2,
        Delta=2.,ro=ro,vo=vo)
    # Check potential
    assert numpy.fabs(pot(4.,0.,use_physical=False)-pot_nounits(4.,0.,use_physical=False)) < 10.**-8., "KuzminKutuzovStaeckelPotential w/ amp w/ units does not behave as expected"   
    # LogarithmicHaloPotential
    pot= potential.LogarithmicHaloPotential(amp=40000*units.km**2/units.s**2,
                                            core=0.,ro=ro,vo=vo)
    # Check potential
    assert numpy.fabs(pot(4.,0.,use_physical=False)*vo**2.-20000*numpy.log(16.)) < 10.**-8., "LogarithmicHaloPotential w/ amp w/ units does not behave as expected"   
    # MiyamotoNagaiPotential
    pot= potential.MiyamotoNagaiPotential(amp=20*units.Msun,
                                          a=2.,b=0.5,ro=ro,vo=vo)
    # Check potential
    assert numpy.fabs(pot(4.,1.,use_physical=False)*vo**2.+(20.*units.Msun*constants.G).to(units.pc*units.km**2/units.s**2).value/numpy.sqrt(16.+(2.+numpy.sqrt(1.+0.25))**2.)/ro/1000.) < 10.**-8., "MiyamotoNagaiPotential( w/ amp w/ units does not behave as expected"   
    # MN3ExponentialDiskPotential
    pot= potential.MN3ExponentialDiskPotential(\
        amp=0.1*units.Msun/units.pc**3.,hr=2.,hz=0.2,ro=ro,vo=vo)
    # density at hr should be 
    assert numpy.fabs(pot.dens(2.,0.2,use_physical=False)*bovy_conversion.dens_in_msolpc3(vo,ro)-0.1*numpy.exp(-2.)) < 10.**-3., "MN3ExponentialDiskPotential w/ amp w/ units does not behave as expected"
    # PlummerPotential
    pot= potential.PlummerPotential(amp=20*units.Msun,
                                    b=0.5,ro=ro,vo=vo)
    # Check potential
    assert numpy.fabs(pot(4.,0.,use_physical=False)*vo**2.+(20.*units.Msun*constants.G).to(units.pc*units.km**2/units.s**2).value/numpy.sqrt(16.+0.25)/ro/1000.) < 10.**-8., "PlummerPotential w/ amp w/ units does not behave as expected"   
    # PowerSphericalPotential
    pot= potential.PowerSphericalPotential(amp=10.**10.*units.Msun,
                                           r1=1.,alpha=2.,ro=ro,vo=vo)
    # density at r1
    assert numpy.fabs(pot.dens(1.,0.,use_physical=False)*bovy_conversion.dens_in_msolpc3(vo,ro)-10./ro**3.) < 10.**-8., "PowerSphericalPotential w/ amp w/ units does not behave as expected"
    # PowerSphericalPotentialwCutoff
    pot= potential.PowerSphericalPotentialwCutoff(amp=0.1*units.Msun/units.pc**3,
                                           r1=1.,alpha=2.,rc=2.,ro=ro,vo=vo)
    # density at r1
    assert numpy.fabs(pot.dens(1.,0.,use_physical=False)*bovy_conversion.dens_in_msolpc3(vo,ro)-0.1*numpy.exp(-0.25)) < 10.**-8., "PowerSphericalPotentialwCutoff w/ amp w/ units does not behave as expected"
    # PseudoIsothermalPotential
    pot= potential.PseudoIsothermalPotential(amp=10.**10.*units.Msun,
                                             a=2.,ro=ro,vo=vo)
    # density at a
    assert numpy.fabs(pot.dens(2.,0.,use_physical=False)*bovy_conversion.dens_in_msolpc3(vo,ro)-10./4./numpy.pi/8./2./ro**3.) < 10.**-8., "PseudoIsothermalPotential w/ amp w/ units does not behave as expected"
    # RazorThinExponentialDiskPotential
    pot= potential.RazorThinExponentialDiskPotential(amp=40.*units.Msun/units.pc**2,
                                                     hr=2.,ro=ro,vo=vo)
    pot_nounits= potential.RazorThinExponentialDiskPotential(\
        amp=(40.*units.Msun/units.pc**2*constants.G).to(1/units.kpc*units.km**2/units.s**2).value*ro/vo**2,
        hr=2.,ro=ro,vo=vo)
    # Check potential
    assert numpy.fabs(pot(4.,0.,use_physical=False)-pot_nounits(4.,0.,use_physical=False)) < 10.**-8., "RazorThinExponentialDiskPotential w/ amp w/ units does not behave as expected"   
    return None

def test_potential_ampunits_altunits():
    # Test that input units for potential amplitudes behave as expected, alternative where G*M is given
    from galpy import potential
    from galpy.util import bovy_conversion
    ro, vo= 9., 210.
    # Burkert
    pot= potential.BurkertPotential(amp=0.1*units.Msun/units.pc**3.*constants.G,
                                    a=2.,ro=ro,vo=vo)
    # density at r=a should be amp/4
    assert numpy.fabs(pot.dens(2.,0.,use_physical=False)*bovy_conversion.dens_in_msolpc3(vo,ro)-0.1/4.) < 10.**-8., "BurkertPotential w/ amp w/ units does not behave as expected"
    # DoubleExponentialDiskPotential
    pot= potential.DoubleExponentialDiskPotential(\
        amp=0.1*units.Msun/units.pc**3.*constants.G,hr=2.,hz=0.2,ro=ro,vo=vo)
    # density at zero should be amp
    assert numpy.fabs(pot.dens(0.,0.,use_physical=False)*bovy_conversion.dens_in_msolpc3(vo,ro)-0.1) < 10.**-8., "DoubleExponentialDiskPotential w/ amp w/ units does not behave as expected"
    # TwoPowerSphericalPotential
    pot= potential.TwoPowerSphericalPotential(amp=20.*units.Msun*constants.G,a=2.,
                                              alpha=1.5,beta=3.5,ro=ro,vo=vo)
    # Check density at r=a
    assert numpy.fabs(pot.dens(2.,0.,use_physical=False)*bovy_conversion.dens_in_msolpc3(vo,ro)-20./4./numpy.pi/8./ro**3./10.**9./4.) < 10.**-8., "TwoPowerSphericalPotential w/ amp w/ units does not behave as expected"
    # TwoPowerSphericalPotential with integer powers
    pot= potential.TwoPowerSphericalPotential(amp=20.*units.Msun*constants.G,a=2.,
                                              alpha=2.,beta=5.,ro=ro,vo=vo)
    # Check density at r=a
    assert numpy.fabs(pot.dens(2.,0.,use_physical=False)*bovy_conversion.dens_in_msolpc3(vo,ro)-20./4./numpy.pi/8./ro**3./10.**9./8.) < 10.**-8., "TwoPowerSphericalPotential w/ amp w/ units does not behave as expected"
    # JaffePotential
    pot= potential.JaffePotential(amp=20.*units.Msun*constants.G,a=2.,ro=ro,vo=vo)
    # Check density at r=a
    assert numpy.fabs(pot.dens(2.,0.,use_physical=False)*bovy_conversion.dens_in_msolpc3(vo,ro)-20./4./numpy.pi/8./ro**3./10.**9./4.) < 10.**-8., "JaffePotential w/ amp w/ units does not behave as expected"
    # HernquistPotential
    pot= potential.HernquistPotential(amp=20.*units.Msun*constants.G,a=2.,ro=ro,vo=vo)
    # Check density at r=a
    assert numpy.fabs(pot.dens(2.,0.,use_physical=False)*bovy_conversion.dens_in_msolpc3(vo,ro)-20./4./numpy.pi/8./ro**3./10.**9./8.) < 10.**-8., "HernquistPotential w/ amp w/ units does not behave as expected"
    # NFWPotential
    pot= potential.NFWPotential(amp=20.*units.Msun*constants.G,a=2.,ro=ro,vo=vo)
    # Check density at r=a
    assert numpy.fabs(pot.dens(2.,0.,use_physical=False)*bovy_conversion.dens_in_msolpc3(vo,ro)-20./4./numpy.pi/8./ro**3./10.**9./4.) < 10.**-8., "NFWPotential w/ amp w/ units does not behave as expected"
    # IsochronePotential
    pot= potential.IsochronePotential(amp=20.*units.Msun*constants.G,b=2.,ro=ro,vo=vo)
    # Check potential
    assert numpy.fabs(pot(4.,0.,use_physical=False)*vo**2.+(20.*units.Msun*constants.G).to(units.pc*units.km**2/units.s**2).value/(2.+numpy.sqrt(4.+16.))/ro/1000.) < 10.**-8., "IsochronePotential w/ amp w/ units does not behave as expected"   
    # KeplerPotential
    pot= potential.KeplerPotential(amp=20.*units.Msun*constants.G,ro=ro,vo=vo)
    # Check mass
    assert numpy.fabs(pot.mass(100.,use_physical=False)*bovy_conversion.mass_in_msol(vo,ro)-20.) < 10.**-8., "KeplerPotential w/ amp w/ units does not behave as expected"   
    # KuzminKutuzovStaeckelPotential
    pot= potential.KuzminKutuzovStaeckelPotential(amp=20.*units.Msun*constants.G,
                                                  Delta=2.,ro=ro,vo=vo)
    pot_nounits= potential.KuzminKutuzovStaeckelPotential(\
        amp=(20.*units.Msun*constants.G).to(units.kpc*units.km**2/units.s**2).value/ro/vo**2,
        Delta=2.,ro=ro,vo=vo)
    # Check potential
    assert numpy.fabs(pot(4.,0.,use_physical=False)-pot_nounits(4.,0.,use_physical=False)) < 10.**-8., "KuzminKutuzovStaeckelPotential w/ amp w/ units does not behave as expected"   
    # MiyamotoNagaiPotential
    pot= potential.MiyamotoNagaiPotential(amp=20*units.Msun*constants.G,
                                          a=2.,b=0.5,ro=ro,vo=vo)
    # Check potential
    assert numpy.fabs(pot(4.,1.,use_physical=False)*vo**2.+(20.*units.Msun*constants.G).to(units.pc*units.km**2/units.s**2).value/numpy.sqrt(16.+(2.+numpy.sqrt(1.+0.25))**2.)/ro/1000.) < 10.**-8., "MiyamotoNagaiPotential( w/ amp w/ units does not behave as expected"   
    # MN3ExponentialDiskPotential
    pot= potential.MN3ExponentialDiskPotential(\
        amp=0.1*units.Msun*constants.G/units.pc**3.,hr=2.,hz=0.2,ro=ro,vo=vo)
    # density at hr should be 
    assert numpy.fabs(pot.dens(2.,0.2,use_physical=False)*bovy_conversion.dens_in_msolpc3(vo,ro)-0.1*numpy.exp(-2.)) < 10.**-3., "MN3ExponentialDiskPotential w/ amp w/ units does not behave as expected"
    # PlummerPotential
    pot= potential.PlummerPotential(amp=20*units.Msun*constants.G,
                                    b=0.5,ro=ro,vo=vo)
    # Check potential
    assert numpy.fabs(pot(4.,0.,use_physical=False)*vo**2.+(20.*units.Msun*constants.G).to(units.pc*units.km**2/units.s**2).value/numpy.sqrt(16.+0.25)/ro/1000.) < 10.**-8., "PlummerPotential w/ amp w/ units does not behave as expected"   
    # PowerSphericalPotential
    pot= potential.PowerSphericalPotential(amp=10.**10.*units.Msun*constants.G,
                                           r1=1.,alpha=2.,ro=ro,vo=vo)
    # density at r1
    assert numpy.fabs(pot.dens(1.,0.,use_physical=False)*bovy_conversion.dens_in_msolpc3(vo,ro)-10./ro**3.) < 10.**-8., "PowerSphericalPotential w/ amp w/ units does not behave as expected"
    # PowerSphericalPotentialwCutoff
    pot= potential.PowerSphericalPotentialwCutoff(amp=0.1*units.Msun*constants.G/units.pc**3,
                                           r1=1.,alpha=2.,rc=2.,ro=ro,vo=vo)
    # density at r1
    assert numpy.fabs(pot.dens(1.,0.,use_physical=False)*bovy_conversion.dens_in_msolpc3(vo,ro)-0.1*numpy.exp(-0.25)) < 10.**-8., "PowerSphericalPotentialwCutoff w/ amp w/ units does not behave as expected"
    # PseudoIsothermalPotential
    pot= potential.PseudoIsothermalPotential(amp=10.**10.*units.Msun*constants.G,
                                             a=2.,ro=ro,vo=vo)
    # density at a
    assert numpy.fabs(pot.dens(2.,0.,use_physical=False)*bovy_conversion.dens_in_msolpc3(vo,ro)-10./4./numpy.pi/8./2./ro**3.) < 10.**-8., "PseudoIsothermalPotential w/ amp w/ units does not behave as expected"
    # RazorThinExponentialDiskPotential
    pot= potential.RazorThinExponentialDiskPotential(amp=40.*units.Msun*constants.G/units.pc**2,
                                                     hr=2.,ro=ro,vo=vo)
    pot_nounits= potential.RazorThinExponentialDiskPotential(\
        amp=(40.*units.Msun/units.pc**2*constants.G).to(1/units.kpc*units.km**2/units.s**2).value*ro/vo**2,
        hr=2.,ro=ro,vo=vo)
    # Check potential
    assert numpy.fabs(pot(4.,0.,use_physical=False)-pot_nounits(4.,0.,use_physical=False)) < 10.**-8., "RazorThinExponentialDiskPotential w/ amp w/ units does not behave as expected"   
    return None

def test_potential_ampunits_wrongunits():
    # Test that input units for potential amplitudes behave as expected
    from galpy import potential
    ro, vo= 9., 210.
    # Burkert
    assert_raises(units.UnitConversionError,
                  lambda x: potential.BurkertPotential(amp=0.1*units.Msun/units.pc**2.,
                                                       a=2.,ro=ro,vo=vo),())
    # DoubleExponentialDiskPotential
    assert_raises(units.UnitConversionError,
                  lambda x: potential.DoubleExponentialDiskPotential(\
            amp=0.1*units.Msun/units.pc**2.*constants.G,hr=2.,hz=0.2,ro=ro,vo=vo),())
    # TwoPowerSphericalPotential
    assert_raises(units.UnitConversionError,
                  lambda x:potential.TwoPowerSphericalPotential(amp=20.*units.Msun/units.pc**3,a=2.,
                                              alpha=1.5,beta=3.5,ro=ro,vo=vo),())
    # TwoPowerSphericalPotential with integer powers
    assert_raises(units.UnitConversionError,
                  lambda x:potential.TwoPowerSphericalPotential(amp=20.*units.Msun/units.pc**3*constants.G,a=2.,
                                                                alpha=2.,beta=5.,ro=ro,vo=vo),
                  ())
    # JaffePotential
    assert_raises(units.UnitConversionError,
                  lambda x:potential.JaffePotential(amp=20.*units.kpc,a=2.,ro=ro,vo=vo),())
    # HernquistPotential
    assert_raises(units.UnitConversionError,
                  lambda x:potential.HernquistPotential(amp=20.*units.Msun/units.pc**3,a=2.,ro=ro,vo=vo),())
    # NFWPotential
    assert_raises(units.UnitConversionError,
                  lambda x:potential.NFWPotential(amp=20.*units.km**2/units.s**2,a=2.,ro=ro,vo=vo),())
    # FlattenedPowerPotential
    assert_raises(units.UnitConversionError,
                  lambda x: potential.FlattenedPowerPotential(amp=40000.*units.km**2/units.s,
                                                              r1=1.,q=0.9,alpha=0.5,core=0.,
                                                              ro=ro,vo=vo),())
    # IsochronePotential
    assert_raises(units.UnitConversionError,
                  lambda x: potential.IsochronePotential(amp=20.*units.km**2/units.s**2,b=2.,ro=ro,vo=vo),())
    # KeplerPotential
    assert_raises(units.UnitConversionError,
                  lambda x:potential.KeplerPotential(amp=20.*units.Msun/units.pc**3,ro=ro,vo=vo),())
    # KuzminKutuzovStaeckelPotential
    assert_raises(units.UnitConversionError,
                  lambda x:potential.KuzminKutuzovStaeckelPotential(amp=20.*units.Msun/units.pc**2,
                                                                    Delta=2.,ro=ro,vo=vo),())
    # LogarithmicHaloPotential
    assert_raises(units.UnitConversionError,
                  lambda x: potential.LogarithmicHaloPotential(amp=40*units.Msun,
                                                          core=0.,ro=ro,vo=vo),())
    # MiyamotoNagaiPotential
    assert_raises(units.UnitConversionError,
                  lambda x:potential.MiyamotoNagaiPotential(amp=20*units.km**2/units.s**2,
                                          a=2.,b=0.5,ro=ro,vo=vo),())
    # MN3ExponentialDiskPotential
    assert_raises(units.UnitConversionError,
                  lambda x:potential.MN3ExponentialDiskPotential(\
            amp=0.1*units.Msun*constants.G,hr=2.,hz=0.2,ro=ro,vo=vo),())
    # PlummerPotential
    assert_raises(units.UnitConversionError,
                  lambda x:potential.PlummerPotential(amp=20*units.km**2/units.s**2,
                                    b=0.5,ro=ro,vo=vo),())
    # PowerSphericalPotential
    assert_raises(units.UnitConversionError,
                  lambda x:potential.PowerSphericalPotential(amp=10.**10.*units.Msun/units.pc**3,
                                           r1=1.,alpha=2.,ro=ro,vo=vo),())
    # PowerSphericalPotentialwCutoff
    assert_raises(units.UnitConversionError,
                  lambda x:potential.PowerSphericalPotentialwCutoff(amp=0.1*units.Msun/units.pc**2,
                                                                    r1=1.,alpha=2.,rc=2.,ro=ro,vo=vo),())
    # PseudoIsothermalPotential
    assert_raises(units.UnitConversionError,
                  lambda x:potential.PseudoIsothermalPotential(amp=10.**10.*units.Msun/units.pc**3,
                                             a=2.,ro=ro,vo=vo),())
    # RazorThinExponentialDiskPotential
    assert_raises(units.UnitConversionError,
                  lambda x:potential.RazorThinExponentialDiskPotential(amp=40.*units.Msun/units.pc**3,
                                                     hr=2.,ro=ro,vo=vo),())
    return None

def test_potential_paramunits():
    # Test that input units for potential parameters other than the amplitude
    # behave as expected
    from galpy import potential
    from galpy.util import bovy_conversion
    ro, vo= 7., 230.
    # Burkert
    pot= potential.BurkertPotential(amp=0.1*units.Msun/units.pc**3.,
                                    a=2.*units.kpc,ro=ro,vo=vo)
    # density at r=a should be amp/4
    assert numpy.fabs(pot.dens(2./ro,0.,use_physical=False)*bovy_conversion.dens_in_msolpc3(vo,ro)-0.1/4.) < 10.**-8., "BurkertPotential w/ parameters w/ units does not behave as expected"
    # DoubleExponentialDiskPotential
    pot= potential.DoubleExponentialDiskPotential(\
        amp=0.1*units.Msun/units.pc**3.,hr=4.*units.kpc,hz=200.*units.pc,
        ro=ro,vo=vo)
    # density at zero should be amp
    assert numpy.fabs(pot.dens(0.,0.,use_physical=False)*bovy_conversion.dens_in_msolpc3(vo,ro)-0.1) < 10.**-8., "DoubleExponentialDiskPotential w/ parameters w/ units does not behave as expected"
    # density at 1. is...
    assert numpy.fabs(pot.dens(1.,0.1,use_physical=False)*bovy_conversion.dens_in_msolpc3(vo,ro)-0.1*numpy.exp(-1./4.*ro-0.1/0.2*ro)) < 10.**-8., "DoubleExponentialDiskPotential w/ parameters w/ units does not behave as expected"
    # TwoPowerSphericalPotential
    pot= potential.TwoPowerSphericalPotential(amp=20.*units.Msun,
                                              a=10.*units.kpc,
                                              alpha=1.5,beta=3.5,ro=ro,vo=vo)
    # Check density at r=a
    assert numpy.fabs(pot.dens(10./ro,0.,use_physical=False)*bovy_conversion.dens_in_msolpc3(vo,ro)-20./4./numpy.pi/8./ro**3./10.**9./4.) < 10.**-8., "TwoPowerSphericalPotential w/ parameters w/ units does not behave as expected"
    # TwoPowerSphericalPotential with integer powers
    pot= potential.TwoPowerSphericalPotential(amp=20.*units.Msun,
                                              a=12000.*units.lyr,
                                              alpha=2.,
                                              beta=5.,ro=ro,vo=vo)
    # Check density at r=a
    assert numpy.fabs(pot.dens((12000.*units.lyr).to(units.kpc).value/ro,0.,use_physical=False)*bovy_conversion.dens_in_msolpc3(vo,ro)-20./4./numpy.pi/8./ro**3./10.**9./8.) < 10.**-8., "TwoPowerSphericalPotential w/ parameters w/ units does not behave as expected"
    # JaffePotential
    pot= potential.JaffePotential(amp=20.*units.Msun,a=0.02*units.Mpc,
                                  ro=ro,vo=vo)
    # Check density at r=a
    assert numpy.fabs(pot.dens(20./ro,0.,use_physical=False)*bovy_conversion.dens_in_msolpc3(vo,ro)-20./4./numpy.pi/8./ro**3./10.**9./4.) < 10.**-8., "JaffePotential w/ parameters w/ units does not behave as expected"
    # HernquistPotential
    pot= potential.HernquistPotential(amp=20.*units.Msun,a=10.*units.kpc,
                                      ro=ro,vo=vo)
    # Check density at r=a
    assert numpy.fabs(pot.dens(10./ro,0.,use_physical=False)*bovy_conversion.dens_in_msolpc3(vo,ro)-20./4./numpy.pi/8./ro**3./10.**9./8.) < 10.**-8., "HernquistPotential w/ parameters w/ units does not behave as expected"
    # NFWPotential
    pot= potential.NFWPotential(amp=20.*units.Msun,a=15.*units.kpc,ro=ro,vo=vo)
    # Check density at r=a
    assert numpy.fabs(pot.dens(15./ro,0.,use_physical=False)*bovy_conversion.dens_in_msolpc3(vo,ro)-20./4./numpy.pi/8./ro**3./10.**9./4.) < 10.**-8., "NFWPotential w/ parameters w/ units does not behave as expected"
    # FlattenedPowerPotential
    pot= potential.FlattenedPowerPotential(amp=40000.*units.km**2/units.s**2,
                                           r1=10.*units.kpc,
                                           q=0.9,alpha=0.5,core=1.*units.kpc,
                                           ro=ro,vo=vo)
    # Check potential
    assert numpy.fabs(pot(2.,1.,use_physical=False)*vo**2.+40000.*(10./ro)**0.5/0.5/(2.**2.+(1./0.9)**2.+(1./ro)**2.)**0.25) < 10.**-8., "FlattenedPowerPotential w/ parameters w/ units does not behave as expected"
    # IsochronePotential
    pot= potential.IsochronePotential(amp=20.*units.Msun,b=10.*units.kpc,
                                      ro=ro,vo=vo)
    # Check potential
    assert numpy.fabs(pot(4.,0.,use_physical=False)*vo**2.+(20.*units.Msun*constants.G).to(units.pc*units.km**2/units.s**2).value/(10./ro+numpy.sqrt((10./ro)**2.+16.))/ro/1000.) < 10.**-8., "IsochronePotential w/ parameters w/ units does not behave as expected"   
    # KuzminKutuzovStaeckelPotential
    pot= potential.KuzminKutuzovStaeckelPotential(amp=20.*units.Msun,
                                                  Delta=10.*units.kpc,
                                                  ro=ro,vo=vo)
    pot_nounits= potential.KuzminKutuzovStaeckelPotential(\
        amp=(20.*units.Msun*constants.G).to(units.kpc*units.km**2/units.s**2).value/ro/vo**2,
        Delta=10./ro,ro=ro,vo=vo)
    # Check potential
    assert numpy.fabs(pot(4.,0.,use_physical=False)-pot_nounits(4.,0.,use_physical=False)) < 10.**-8., "KuzminKutuzovStaeckelPotential w/ parameters w/ units does not behave as expected"   
    # LogarithmicHaloPotential
    pot= potential.LogarithmicHaloPotential(amp=40000*units.km**2/units.s**2,
                                            core=1.*units.kpc,ro=ro,vo=vo)
    # Check potential
    assert numpy.fabs(pot(4.,0.,use_physical=False)*vo**2.-20000*numpy.log(16.+(1./ro)**2.)) < 10.**-8., "LogarithmicHaloPotential w/ parameters w/ units does not behave as expected"   
    # MiyamotoNagaiPotential
    pot= potential.MiyamotoNagaiPotential(amp=20*units.Msun,
                                          a=5.*units.kpc,b=300.*units.pc,
                                          ro=ro,vo=vo)
    # Check potential
    assert numpy.fabs(pot(4.,1.,use_physical=False)*vo**2.+(20.*units.Msun*constants.G).to(units.pc*units.km**2/units.s**2).value/numpy.sqrt(16.+(5./ro+numpy.sqrt(1.+(0.3/ro)**2.))**2.)/ro/1000.) < 10.**-8., "MiyamotoNagaiPotential( w/ parameters w/ units does not behave as expected"   
    # MN3ExponentialDiskPotential
    pot= potential.MN3ExponentialDiskPotential(\
        amp=0.1*units.Msun/units.pc**3.,hr=6.*units.kpc,hz=300.*units.pc,
        ro=ro,vo=vo)
    # density at hr should be 
    assert numpy.fabs(pot.dens(6./ro,0.3/ro,use_physical=False)*bovy_conversion.dens_in_msolpc3(vo,ro)-0.1*numpy.exp(-2.)) < 10.**-3., "MN3ExponentialDiskPotential w/ parameters w/ units does not behave as expected"
    # MovingObjectPotential
    from galpy.orbit import Orbit
    pot= potential.MovingObjectPotential(Orbit([1.1,0.1,1.1,0.1,0.1,0.3]),
                                         GM=20*units.Msun,
                                         softening_length=5.*units.kpc,
                                         ro=ro,vo=vo)
    pot_nounits= potential.MovingObjectPotential(\
        Orbit([1.1,0.1,1.1,0.1,0.1,0.3]),
        GM=(20*units.Msun*constants.G).to(units.kpc*units.km**2/units.s**2).value/ro/vo**2,
        softening_length=5./ro,ro=ro,vo=vo)
    # Check potential
    assert numpy.fabs(pot(4.,0.1,t=0.,use_physical=False)-pot_nounits(4.,0.1,t=0.,use_physical=False)) < 10.**-8., "PlummerPotential w/ parameters w/ units does not behave as expected"   
    # MovingObjectPotential w/ Orbit w/ units
    from galpy.orbit import Orbit
    pot= potential.MovingObjectPotential(\
        Orbit([1.1*ro*units.kpc,0.1*vo*units.km/units.s,1.1*vo*units.km/units.s,
               0.1*ro*units.kpc,0.1*vo*units.km/units.s,0.3*units.rad]),
                                         GM=20*units.Msun,
                                         softening_length=5.*units.kpc,
                                         ro=ro,vo=vo)
    pot_nounits= potential.MovingObjectPotential(\
        Orbit([1.1,0.1,1.1,0.1,0.1,0.3]),
        GM=(20*units.Msun*constants.G).to(units.kpc*units.km**2/units.s**2).value/ro/vo**2,
        softening_length=5./ro,ro=ro,vo=vo)
    # Check potential
    assert numpy.fabs(pot(4.,0.1,t=0.,use_physical=False)-pot_nounits(4.,0.1,t=0.,use_physical=False)) < 10.**-8., "PlummerPotential w/ parameters w/ units does not behave as expected"   
    # PlummerPotential
    pot= potential.PlummerPotential(amp=20*units.Msun,
                                    b=5.*units.kpc,ro=ro,vo=vo)
    # Check potential
    assert numpy.fabs(pot(4.,0.,use_physical=False)*vo**2.+(20.*units.Msun*constants.G).to(units.pc*units.km**2/units.s**2).value/numpy.sqrt(16.+(5./ro)**2.)/ro/1000.) < 10.**-8., "PlummerPotential w/ parameters w/ units does not behave as expected"   
    # PowerSphericalPotential
    pot= potential.PowerSphericalPotential(amp=10.**10.*units.Msun,
                                           r1=10.*units.kpc,
                                           alpha=2.,ro=ro,vo=vo)
    # density at r1
    assert numpy.fabs(pot.dens(10./ro,0.,use_physical=False)*bovy_conversion.dens_in_msolpc3(vo,ro)-10./ro**3./(10./ro)**3.) < 10.**-8., "PowerSphericalPotential w/ parameters w/ units does not behave as expected"
    # PowerSphericalPotentialwCutoff
    pot= potential.PowerSphericalPotentialwCutoff(amp=0.1*units.Msun/units.pc**3,
                                                  r1=10.*units.kpc,
                                                  alpha=2.,rc=12.*units.kpc,
                                                  ro=ro,vo=vo)
    # density at r1
    assert numpy.fabs(pot.dens(10./ro,0.,use_physical=False)*bovy_conversion.dens_in_msolpc3(vo,ro)-0.1*numpy.exp(-(10./12.)**2.)) < 10.**-8., "PowerSphericalPotentialwCutoff w/ parameters w/ units does not behave as expected"
    # PseudoIsothermalPotential
    pot= potential.PseudoIsothermalPotential(amp=10.**10.*units.Msun,
                                             a=20.*units.kpc,ro=ro,vo=vo)
    # density at a
    assert numpy.fabs(pot.dens(20./ro,0.,use_physical=False)*bovy_conversion.dens_in_msolpc3(vo,ro)-10./4./numpy.pi/(20./ro)**3./2./ro**3.) < 10.**-8., "PseudoIsothermalPotential w/ parameters w/ units does not behave as expected"
    # RazorThinExponentialDiskPotential
    pot= potential.RazorThinExponentialDiskPotential(amp=40.*units.Msun/units.pc**2,
                                                     hr=10.*units.kpc,
                                                     ro=ro,vo=vo)
    pot_nounits= potential.RazorThinExponentialDiskPotential(\
        amp=(40.*units.Msun/units.pc**2*constants.G).to(1/units.kpc*units.km**2/units.s**2).value*ro/vo**2,
        hr=10./ro,ro=ro,vo=vo)
    # Check potential
    assert numpy.fabs(pot(4.,0.,use_physical=False)-pot_nounits(4.,0.,use_physical=False)) < 10.**-8., "RazorThinExponentialDiskPotential w/ parameters w/ units does not behave as expected"   
    return None

def test_potential_paramunits_2d():
    # Test that input units for potential parameters other than the amplitude
    # behave as expected
    from galpy import potential
    from galpy.util import bovy_conversion
    ro, vo= 11., 180.
    # DehnenBarPotential
    pot= potential.DehnenBarPotential(amp=1.,
                                      omegab=50.*units.km/units.s/units.kpc,
                                      rb=4.*units.kpc,
                                      Af=1290.*units.km**2/units.s**2,
                                      barphi=20.*units.deg,
                                      ro=ro,vo=vo)
    pot_nounits= potential.DehnenBarPotential(amp=1.,
                                              omegab=50.*ro/vo,
                                              rb=4./ro,
                                              Af=1290./vo**2.,
                                              barphi=20./180.*numpy.pi,
                                              ro=ro,vo=vo)
    # Check potential
    assert numpy.fabs(pot(1.5,phi=0.1,use_physical=False)-pot_nounits(1.5,phi=0.1,use_physical=False)) < 10.**-8., "DehnenBarPotential w/ parameters w/ units does not behave as expected"   
    # DehnenBarPotential, alternative setup
    pot= potential.DehnenBarPotential(amp=1.,
                                      rolr=8.*units.kpc,
                                      chi=0.8,
                                      alpha=0.02,
                                      beta=0.2,
                                      barphi=20.*units.deg,
                                      ro=ro,vo=vo)
    pot_nounits= potential.DehnenBarPotential(amp=1.,
                                              rolr=8./ro,
                                              chi=0.8,
                                              alpha=0.02,
                                              beta=0.2,
                                              barphi=20./180.*numpy.pi,
                                              ro=ro,vo=vo)
    # Check potential
    assert numpy.fabs(pot(1.5,phi=0.1,use_physical=False)-pot_nounits(1.5,phi=0.1,use_physical=False)) < 10.**-8., "DehnenBarPotential w/ parameters w/ units does not behave as expected"   
    # CosmphiDiskPotential
    pot= potential.CosmphiDiskPotential(amp=1.,
                                        m=3,
                                        tform=1.*units.Gyr,
                                        tsteady=3.*units.Gyr,
                                        phib=20.*units.deg,
                                        phio=1290.*units.km**2/units.s**2,
                                        r1=8.*units.kpc,
                                        ro=ro,vo=vo)
    pot_nounits= potential.CosmphiDiskPotential(amp=1.,
                                                m=3,
                                                tform=1./bovy_conversion.time_in_Gyr(vo,ro),
                                                tsteady=3./bovy_conversion.time_in_Gyr(vo,ro),
                                                phib=20./180.*numpy.pi,
                                                phio=1290./vo**2.,
                                                r1=8./ro,
                                                ro=ro,vo=vo)
    # Check potential
    assert numpy.fabs(pot(1.5,phi=0.1,t=2./bovy_conversion.time_in_Gyr(vo,ro),use_physical=False)-pot_nounits(1.5,phi=0.1,t=2./bovy_conversion.time_in_Gyr(vo,ro),use_physical=False)) < 10.**-8., "CosmphiDiskPotential w/ parameters w/ units does not behave as expected"   
    # CosmphiDiskPotential, alternative setup
    pot= potential.CosmphiDiskPotential(amp=1.,
                                        m=3,
                                        tform=1.*units.Gyr,
                                        tsteady=3.*units.Gyr,
                                        cp=1000.*units.km**2/units.s**2.,
                                        sp=300.*units.km**2/units.s**2.,
                                        r1=8.*units.kpc,
                                        ro=ro,vo=vo)
    pot_nounits= potential.CosmphiDiskPotential(amp=1.,
                                                m=3,
                                                tform=1./bovy_conversion.time_in_Gyr(vo,ro),
                                                tsteady=3./bovy_conversion.time_in_Gyr(vo,ro),
                                                cp=1000./vo**2.,
                                                sp=300./vo**2.,
                                                r1=8./ro,
                                                ro=ro,vo=vo)
    # Check potential
    assert numpy.fabs(pot(1.5,phi=0.1,t=2./bovy_conversion.time_in_Gyr(vo,ro),use_physical=False)-pot_nounits(1.5,phi=0.1,t=2./bovy_conversion.time_in_Gyr(vo,ro),use_physical=False)) < 10.**-8., "CosmphiDiskPotential w/ parameters w/ units does not behave as expected"   
    # EllipticalDiskPotential
    pot= potential.EllipticalDiskPotential(amp=1.,
                                           tform=1.*units.Gyr,
                                           tsteady=3.*units.Gyr,
                                           phib=20.*units.deg,
                                           twophio=1290.*units.km**2/units.s**2,
                                           r1=8.*units.kpc,
                                           ro=ro,vo=vo)
    pot_nounits= potential.EllipticalDiskPotential(amp=1.,
                                                   tform=1./bovy_conversion.time_in_Gyr(vo,ro),
                                                   tsteady=3./bovy_conversion.time_in_Gyr(vo,ro),
                                                   phib=20./180.*numpy.pi,
                                                   twophio=1290./vo**2.,
                                                   r1=8./ro,
                                                   ro=ro,vo=vo)
    # Check potential
    assert numpy.fabs(pot(1.5,phi=0.1,t=2./bovy_conversion.time_in_Gyr(vo,ro),use_physical=False)-pot_nounits(1.5,phi=0.1,t=2./bovy_conversion.time_in_Gyr(vo,ro),use_physical=False)) < 10.**-8., "EllipticalDiskPotential w/ parameters w/ units does not behave as expected"   
    # EllipticalDiskPotential, alternative setup
    pot= potential.EllipticalDiskPotential(amp=1.,
                                           tform=1.*units.Gyr,
                                           tsteady=3.*units.Gyr,
                                           cp=1000.*units.km**2/units.s**2.,
                                           sp=300.*units.km**2/units.s**2.,
                                           r1=8.*units.kpc,
                                           ro=ro,vo=vo)
    pot_nounits= potential.EllipticalDiskPotential(amp=1.,
                                                   tform=1./bovy_conversion.time_in_Gyr(vo,ro),
                                                   tsteady=3./bovy_conversion.time_in_Gyr(vo,ro),
                                                   cp=1000./vo**2.,
                                                   sp=300./vo**2.,
                                                   r1=8./ro,
                                                   ro=ro,vo=vo)
    # Check potential
    assert numpy.fabs(pot(1.5,phi=0.1,t=2./bovy_conversion.time_in_Gyr(vo,ro),use_physical=False)-pot_nounits(1.5,phi=0.1,t=2./bovy_conversion.time_in_Gyr(vo,ro),use_physical=False)) < 10.**-8., "EllipticalDiskPotential w/ parameters w/ units does not behave as expected"   
    # LopsidedDiskPotential
    pot= potential.LopsidedDiskPotential(amp=1.,
                                         tform=1.*units.Gyr,
                                         tsteady=3.*units.Gyr,
                                         phib=20.*units.deg,
                                         phio=1290.*units.km**2/units.s**2,
                                         r1=8.*units.kpc,
                                         ro=ro,vo=vo)
    pot_nounits= potential.LopsidedDiskPotential(amp=1.,
                                                 tform=1./bovy_conversion.time_in_Gyr(vo,ro),
                                                 tsteady=3./bovy_conversion.time_in_Gyr(vo,ro),
                                                 phib=20./180.*numpy.pi,
                                                 phio=1290./vo**2.,
                                                 r1=8./ro,
                                                 ro=ro,vo=vo)
    # Check potential
    assert numpy.fabs(pot(1.5,phi=0.1,t=2./bovy_conversion.time_in_Gyr(vo,ro),use_physical=False)-pot_nounits(1.5,phi=0.1,t=2./bovy_conversion.time_in_Gyr(vo,ro),use_physical=False)) < 10.**-8., "LopsidedDiskPotential w/ parameters w/ units does not behave as expected"   
    # LopsidedDiskPotential, alternative setup
    pot= potential.LopsidedDiskPotential(amp=1.,
                                         tform=1.*units.Gyr,
                                         tsteady=3.*units.Gyr,
                                         cp=1000.*units.km**2/units.s**2.,
                                         sp=300.*units.km**2/units.s**2.,
                                         r1=8.*units.kpc,
                                         ro=ro,vo=vo)
    pot_nounits= potential.LopsidedDiskPotential(amp=1.,
                                                 tform=1./bovy_conversion.time_in_Gyr(vo,ro),
                                                 tsteady=3./bovy_conversion.time_in_Gyr(vo,ro),
                                                 cp=1000./vo**2.,
                                                 sp=300./vo**2.,
                                                 r1=8./ro,
                                                 ro=ro,vo=vo)
    # Check potential
    assert numpy.fabs(pot(1.5,phi=0.1,t=2./bovy_conversion.time_in_Gyr(vo,ro),use_physical=False)-pot_nounits(1.5,phi=0.1,t=2./bovy_conversion.time_in_Gyr(vo,ro),use_physical=False)) < 10.**-8., "LopsidedDiskPotential w/ parameters w/ units does not behave as expected"   
    # SteadyLogSpiralPotential
    pot= potential.SteadyLogSpiralPotential(amp=1.,
                                            m=4,
                                            omegas=50.*units.km/units.s/units.kpc,
                                            A=1700.*units.km**2/units.s**2,
                                            gamma=21.*units.deg,
                                            alpha=-9.,
                                            ro=ro,vo=vo)
    pot_nounits= potential.SteadyLogSpiralPotential(amp=1.,
                                                    m=4,
                                                    omegas=50.*ro/vo,
                                                    A=1700./vo**2.,
                                                    gamma=21./180.*numpy.pi,
                                                    alpha=-9.,
                                                    ro=ro,vo=vo)
    # Check potential
    assert numpy.fabs(pot(1.5,phi=0.1,t=2./bovy_conversion.time_in_Gyr(vo,ro),use_physical=False)-pot_nounits(1.5,phi=0.1,t=2./bovy_conversion.time_in_Gyr(vo,ro),use_physical=False)) < 10.**-8., "LopsidedDiskPotential w/ parameters w/ units does not behave as expected"   
    # SteadyLogSpiralPotential, alternative setup
    pot= potential.SteadyLogSpiralPotential(amp=1.,
                                            m=4,
                                            omegas=50.*units.km/units.s/units.kpc,
                                            A=1700.*units.km**2/units.s**2,
                                            gamma=21.*units.deg,
                                            p=10.*units.deg,
                                            ro=ro,vo=vo)
    pot_nounits= potential.SteadyLogSpiralPotential(amp=1.,
                                                    m=4,
                                                    omegas=50.*ro/vo,
                                                    A=1700./vo**2.,
                                                    gamma=21./180.*numpy.pi,
                                                    p=10./180.*numpy.pi,
                                                    ro=ro,vo=vo)
    # Check potential
    assert numpy.fabs(pot(1.5,phi=0.1,t=2./bovy_conversion.time_in_Gyr(vo,ro),use_physical=False)-pot_nounits(1.5,phi=0.1,t=2./bovy_conversion.time_in_Gyr(vo,ro),use_physical=False)) < 10.**-8., "LopsidedDiskPotential w/ parameters w/ units does not behave as expected"   
    # TransientLogSpiralPotential
    pot= potential.TransientLogSpiralPotential(amp=1.,
                                               m=4,
                                               omegas=50.*units.km/units.s/units.kpc,
                                               A=1700.*units.km**2/units.s**2,
                                               gamma=21.*units.deg,
                                               alpha=-9.,
                                               to=2.*units.Gyr,
                                               sigma=1.*units.Gyr,
                                               ro=ro,vo=vo)
    pot_nounits= potential.TransientLogSpiralPotential(amp=1.,
                                                       m=4,
                                                       omegas=50.*ro/vo,
                                                       A=1700./vo**2.,
                                                       gamma=21./180.*numpy.pi,
                                                       alpha=-9.,
                                                       to=2./bovy_conversion.time_in_Gyr(vo,ro),
                                                       sigma=1./bovy_conversion.time_in_Gyr(vo,ro),
                                                       ro=ro,vo=vo)
    # Check potential
    assert numpy.fabs(pot(1.5,phi=0.1,t=2./bovy_conversion.time_in_Gyr(vo,ro),use_physical=False)-pot_nounits(1.5,phi=0.1,t=2./bovy_conversion.time_in_Gyr(vo,ro),use_physical=False)) < 10.**-8., "LopsidedDiskPotential w/ parameters w/ units does not behave as expected"   
    # TransientLogSpiralPotential, alternative setup
    pot= potential.TransientLogSpiralPotential(amp=1.,
                                               m=4,
                                               omegas=50.*units.km/units.s/units.kpc,
                                               A=1700.*units.km**2/units.s**2,
                                               gamma=21.*units.deg,
                                               p=10.*units.deg,
                                               to=2.*units.Gyr,
                                               sigma=1.*units.Gyr,
                                               ro=ro,vo=vo)
    pot_nounits= potential.TransientLogSpiralPotential(amp=1.,
                                                       m=4,
                                                       omegas=50.*ro/vo,
                                                       A=1700./vo**2.,
                                                       gamma=21./180.*numpy.pi,
                                                       p=10./180.*numpy.pi,
                                                       to=2./bovy_conversion.time_in_Gyr(vo,ro),
                                                       sigma=1./bovy_conversion.time_in_Gyr(vo,ro),
                                                       ro=ro,vo=vo)
    # Check potential
    assert numpy.fabs(pot(1.5,phi=0.1,t=2./bovy_conversion.time_in_Gyr(vo,ro),use_physical=False)-pot_nounits(1.5,phi=0.1,t=2./bovy_conversion.time_in_Gyr(vo,ro),use_physical=False)) < 10.**-8., "LopsidedDiskPotential w/ parameters w/ units does not behave as expected"   
    return None

def test_potential_paramunits_1d():
    # Test that input units for potential parameters other than the amplitude
    # behave as expected
    from galpy import potential
    from galpy.util import bovy_conversion
    ro, vo= 10.5, 195.
    # KGPotential
    pot= potential.KGPotential(amp=1.,
                               K=40.*units.Msun/units.pc**2,
                               F=0.02*units.Msun/units.pc**3,
                               D=200*units.pc,ro=ro,vo=vo)
    pot_nounits= potential.KGPotential(amp=1.,
                                       K=40./bovy_conversion.surfdens_in_msolpc2(vo,ro)*2.*numpy.pi,
                                       F=0.02/bovy_conversion.dens_in_msolpc3(vo,ro)*4.*numpy.pi,
                                       D=0.2/ro,ro=ro,vo=vo)
    # Check potential
    assert numpy.fabs(pot(1.5,use_physical=False)-pot_nounits(1.5,use_physical=False)) < 10.**-8., "KGPotential w/ parameters w/ units does not behave as expected"   
    # KGPotential, alternative setup
    pot= potential.KGPotential(amp=1.,
                               K=40.*units.Msun/units.pc**2*constants.G,
                               F=0.02*units.Msun/units.pc**3*constants.G,
                               D=200*units.pc,ro=ro,vo=vo)
    pot_nounits= potential.KGPotential(amp=1.,
                                       K=40./bovy_conversion.surfdens_in_msolpc2(vo,ro),
                                       F=0.02/bovy_conversion.dens_in_msolpc3(vo,ro),
                                       D=0.2/ro,ro=ro,vo=vo)
    # Check potential
    assert numpy.fabs(pot(1.5,use_physical=False)-pot_nounits(1.5,use_physical=False)) < 10.**-8., "KGPotential w/ parameters w/ units does not behave as expected"   
    return None

def test_potential_paramunits_1d_wrongunits():
    # Test that input units for potential amplitudes behave as expected
    from galpy import potential
    ro, vo= 9., 210.
    # KGPotential
    assert_raises(units.UnitConversionError,
                  lambda x: \
                      potential.KGPotential(amp=1.,
                                            K=40.*units.Msun/units.pc**3,
                                            F=0.02*units.Msun/units.pc**3,
                                            D=200*units.pc,ro=ro,vo=vo),())
    assert_raises(units.UnitConversionError,
                  lambda x: \
                      potential.KGPotential(amp=1.,
                                            K=40.*units.Msun/units.pc**2,
                                            F=0.02*units.Msun/units.pc**2,
                                            D=200*units.pc,ro=ro,vo=vo),())
    return None

def test_potential_method_turnphysicalon():
    from galpy import potential
    # 3D
    pot= potential.BurkertPotential(ro=7.*units.kpc)
    pot.turn_physical_on()
    assert isinstance(pot(1.1,0.1),units.Quantity), 'Potential method does not return Quantity when turn_physical_on has been called'
    # 2D
    pot= potential.DehnenBarPotential(ro=6.*units.kpc)
    pot.turn_physical_on()
    assert isinstance(pot(1.1,phi=0.1),units.Quantity), 'Potential method does not return Quantity when turn_physical_on has been called'
    # 1D
    pot= potential.KGPotential(ro=5.*units.kpc)
    pot.turn_physical_on()
    assert isinstance(pot(1.1),units.Quantity), 'Potential method does not return Quantity when turn_physical_on has been called'
    return None

def test_potential_method_turnphysicaloff():
    from galpy import potential
    # 3D
    pot= potential.BurkertPotential(ro=7.*units.kpc)
    pot.turn_physical_off()
    assert isinstance(pot(1.1,0.1),float), 'Potential method does not return float when turn_physical_on has been called'
    # 2D
    pot= potential.DehnenBarPotential(ro=6.*units.kpc)
    pot.turn_physical_off()
    assert isinstance(pot(1.1,phi=0.1),float), 'Potential method does not return float when turn_physical_on has been called'
    # 1D
    pot= potential.KGPotential(ro=5.*units.kpc)
    pot.turn_physical_off()
    assert isinstance(pot(1.1),float), 'Potential method does not return float when turn_physical_on has been called'
    return None

def test_potential_function_turnphysicalon():
    from galpy import potential
    # 3D
    pot= potential.BurkertPotential(ro=7.*units.kpc)
    potential.turn_physical_on(pot)
    assert isinstance(potential.evaluatePotentials(pot,1.1,0.1),units.Quantity), 'Potential function does not return Quantity when function turn_physical_on has been called'
    pot= potential.BurkertPotential(ro=7.*units.kpc)
    potential.turn_physical_on([pot])
    assert isinstance(potential.evaluatePotentials([pot],1.1,0.1),units.Quantity), 'Potential function does not return Quantity when function turn_physical_on has been called'
    # 2D
    pot= potential.DehnenBarPotential(ro=6.*units.kpc)
    potential.turn_physical_on(pot)
    assert isinstance(potential.evaluateplanarPotentials(pot,1.1,phi=0.1),units.Quantity), 'Potential function does not return Quantity when function turn_physical_on has been called'
    potential.turn_physical_on([pot])
    assert isinstance(potential.evaluateplanarPotentials([pot],1.1,phi=0.1),units.Quantity), 'Potential function does not return Quantity when function turn_physical_on has been called'
    # 1D
    pot= potential.KGPotential(ro=5.*units.kpc)
    potential.turn_physical_on(pot)
    assert isinstance(potential.evaluatelinearPotentials(pot,1.1),units.Quantity), 'Potential function does not return Quantity when function turn_physical_on has been called'
    potential.turn_physical_on([pot])
    assert isinstance(potential.evaluatelinearPotentials([pot],1.1),units.Quantity), 'Potential function does not return Quantity when function turn_physical_on has been called'
    return None

def test_potential_function_turnphysicaloff():
    from galpy import potential
    # 3D
    pot= potential.BurkertPotential(ro=7.*units.kpc)
    potential.turn_physical_off(pot)
    assert isinstance(potential.evaluatePotentials(pot,1.1,0.1),float), 'Potential function does not return float when function turn_physical_off has been called'
    pot= potential.BurkertPotential(ro=7.*units.kpc)
    potential.turn_physical_off([pot])
    assert isinstance(potential.evaluatePotentials([pot],1.1,0.1),float), 'Potential function does not return float when function turn_physical_off has been called'
    # 2D
    pot= potential.DehnenBarPotential(ro=6.*units.kpc)
    potential.turn_physical_off(pot)
    assert isinstance(potential.evaluateplanarPotentials(pot,1.1,phi=0.1),float), 'Potential function does not return float when function turn_physical_off has been called'
    potential.turn_physical_off([pot])
    assert isinstance(potential.evaluateplanarPotentials([pot],1.1,phi=0.1),float), 'Potential function does not return float when function turn_physical_off has been called'
    # 1D
    pot= potential.KGPotential(ro=5.*units.kpc)
    potential.turn_physical_off(pot)
    assert isinstance(potential.evaluatelinearPotentials(pot,1.1),float), 'Potential function does not return float when function turn_physical_off has been called'
    potential.turn_physical_off([pot])
    assert isinstance(potential.evaluatelinearPotentials([pot],1.1),float), 'Potential function does not return float when function turn_physical_off has been called'
    return None

def test_potential_setup_roAsQuantity():
    from galpy import potential
    # 3D
    pot= potential.BurkertPotential(ro=7.*units.kpc)
    assert numpy.fabs(pot._ro-7.) < 10.**-10., 'ro in 3D potential setup as Quantity does not work as expected'
    # 2D
    pot= potential.DehnenBarPotential(ro=6.*units.kpc)
    assert numpy.fabs(pot._ro-6.) < 10.**-10., 'ro in 2D potential setup as Quantity does not work as expected'
    # 1D
    pot= potential.KGPotential(ro=5.*units.kpc)
    assert numpy.fabs(pot._ro-5.) < 10.**-10., 'ro in 1D potential setup as Quantity does not work as expected'
    return None

def test_potential_setup_roAsQuantity_oddunits():
    from galpy import potential
    # 3D
    pot= potential.BurkertPotential(ro=7.*units.lyr)
    assert numpy.fabs(pot._ro-7.*units.lyr.to(units.kpc)) < 10.**-10., 'ro in 3D potential setup as Quantity does not work as expected'
    # 2D
    pot= potential.DehnenBarPotential(ro=6.*units.lyr)
    assert numpy.fabs(pot._ro-6.*units.lyr.to(units.kpc)) < 10.**-10., 'ro in 2D potential setup as Quantity does not work as expected'
    # 1D
    pot= potential.KGPotential(ro=5.*units.lyr)
    assert numpy.fabs(pot._ro-5.*units.lyr.to(units.kpc)) < 10.**-10., 'ro in 1D potential setup as Quantity does not work as expected'
    return None

def test_potential_setup_voAsQuantity():
    from galpy import potential
    # 3D
    pot= potential.BurkertPotential(vo=210.*units.km/units.s)
    assert numpy.fabs(pot._vo-210.) < 10.**-10., 'vo in 3D potential setup as Quantity does not work as expected'
    # 2D
    pot= potential.DehnenBarPotential(vo=230.*units.km/units.s)
    assert numpy.fabs(pot._vo-230.) < 10.**-10., 'vo in 2D potential setup as Quantity does not work as expected'
    # 1D
    pot= potential.KGPotential(vo=250.*units.km/units.s)
    assert numpy.fabs(pot._vo-250.) < 10.**-10., 'vo in 1D potential setup as Quantity does not work as expected'
    return None

def test_potential_setup_voAsQuantity_oddunits():
    from galpy import potential
    # 3D
    pot= potential.BurkertPotential(vo=210.*units.pc/units.Myr)
    assert numpy.fabs(pot._vo-210.*(units.pc/units.Myr).to(units.km/units.s)) < 10.**-10., 'vo in 3D potential setup as Quantity does not work as expected'
    # 2D
    pot= potential.DehnenBarPotential(vo=230.*units.pc/units.Myr)
    assert numpy.fabs(pot._vo-230.*(units.pc/units.Myr).to(units.km/units.s)) < 10.**-10., 'vo in 2D potential setup as Quantity does not work as expected'
    # 1D
    pot= potential.KGPotential(vo=250.*units.pc/units.Myr)
    assert numpy.fabs(pot._vo-250.*(units.pc/units.Myr).to(units.km/units.s)) < 10.**-10., 'vo in 1D potential setup as Quantity does not work as expected'
    return None

def test_interpRZPotential_ro():
    # Test that ro is correctly propagated to interpRZPotential
    from galpy.potential import BurkertPotential, interpRZPotential
    ro= 9.
    # ro on, single pot
    bp= BurkertPotential(ro=ro)
    ip= interpRZPotential(bp)
    assert numpy.fabs(ip._ro-bp._ro) < 10.**-10., 'ro not correctly propagated to interpRZPotential'
    assert ip._roSet, 'roSet not correctly propagated to interpRZPotential'
    # ro on, list pot
    ip= interpRZPotential([bp])
    assert numpy.fabs(ip._ro-bp._ro) < 10.**-10., 'ro not correctly propagated to interpRZPotential'
    assert ip._roSet, 'roSet not correctly propagated to interpRZPotential'
    # ro off, single pot
    bp= BurkertPotential()
    ip= interpRZPotential(bp)
    assert numpy.fabs(ip._ro-bp._ro) < 10.**-10., 'ro not correctly propagated to interpRZPotential'
    assert not ip._roSet, 'roSet not correctly propagated to interpRZPotential'
    # ro off, list pot
    bp= BurkertPotential()
    ip= interpRZPotential([bp])
    assert numpy.fabs(ip._ro-bp._ro) < 10.**-10., 'ro not correctly propagated to interpRZPotential'
    assert not ip._roSet, 'roSet not correctly propagated to interpRZPotential'
    return None

def test_interpRZPotential_vo():
    # Test that vo is correctly propagated to interpRZPotential
    from galpy.potential import BurkertPotential, interpRZPotential
    vo= 200.
    # vo on, single pot
    bp= BurkertPotential(vo=vo)
    ip= interpRZPotential(bp)
    assert numpy.fabs(ip._vo-bp._vo) < 10.**-10., 'vo not correctly propagated to interpRZPotential'
    assert ip._voSet, 'voSet not correctly propagated to interpRZPotential'
    # vo on, list pot
    ip= interpRZPotential([bp])
    assert numpy.fabs(ip._vo-bp._vo) < 10.**-10., 'vo not correctly propagated to interpRZPotential'
    assert ip._voSet, 'voSet not correctly propagated to interpRZPotential'
    # vo off, single pot
    bp= BurkertPotential()
    ip= interpRZPotential(bp)
    assert numpy.fabs(ip._vo-bp._vo) < 10.**-10., 'vo not correctly propagated to interpRZPotential'
    assert not ip._voSet, 'voSet not correctly propagated to interpRZPotential'
    # vo off, list pot
    bp= BurkertPotential()
    ip= interpRZPotential([bp])
    assert numpy.fabs(ip._vo-bp._vo) < 10.**-10., 'vo not correctly propagated to interpRZPotential'
    assert not ip._voSet, 'voSet not correctly propagated to interpRZPotential'
    return None

def test_actionAngle_method_returntype():
    from galpy.actionAngle import actionAngleIsochrone, actionAngleSpherical, \
        actionAngleAdiabatic, actionAngleStaeckel, actionAngleIsochroneApprox
    from galpy.potential import PlummerPotential, MWPotential
    # actionAngleIsochrone
    aA= actionAngleIsochrone(b=0.8,ro=8.,vo=220.)
    for ii in range(3):
        assert isinstance(aA(1.1,0.1,1.1,0.1,0.2,0.)[ii],units.Quantity), 'actionAngleIsochrone method __call__ does not return Quantity when it should'
    for ii in range(6):
        assert isinstance(aA.actionsFreqs(1.1,0.1,1.1,0.1,0.2,0.)[ii],units.Quantity), 'actionAngleIsochrone method actionsFreqs does not return Quantity when it should'
    for ii in range(9):
        assert isinstance(aA.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii],units.Quantity), 'actionAngleIsochrone method actionsFreqsAngles does not return Quantity when it should'
    # actionAngleSpherical
    pot= PlummerPotential(normalize=1.,b=0.7)
    aA= actionAngleSpherical(pot=pot,ro=8.,vo=220.)
    for ii in range(3):
        assert isinstance(aA(1.1,0.1,1.1,0.1,0.2,0.)[ii],units.Quantity), 'actionAngleIsochrone method __call__ does not return Quantity when it should'
    for ii in range(6):
        assert isinstance(aA.actionsFreqs(1.1,0.1,1.1,0.1,0.2,0.)[ii],units.Quantity), 'actionAngleIsochrone method actionsFreqs does not return Quantity when it should'
    for ii in range(9):
        assert isinstance(aA.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii],units.Quantity), 'actionAngleIsochrone method actionsFreqsAngles does not return Quantity when it should'
    # actionAngleAdiabatic
    aA= actionAngleAdiabatic(pot=MWPotential,ro=8.,vo=220.)
    for ii in range(3):
        assert isinstance(aA(1.1,0.1,1.1,0.1,0.2,0.)[ii],units.Quantity), 'actionAngleIsochrone method __call__ does not return Quantity when it should'
    # actionAngleStaeckel
    aA= actionAngleStaeckel(pot=MWPotential,delta=0.45,ro=8.,vo=220.)
    for ii in range(3):
        assert isinstance(aA(1.1,0.1,1.1,0.1,0.2,0.)[ii],units.Quantity), 'actionAngleIsochrone method __call__ does not return Quantity when it should'
    for ii in range(6):
        assert isinstance(aA.actionsFreqs(1.1,0.1,1.1,0.1,0.2,0.)[ii],units.Quantity), 'actionAngleIsochrone method actionsFreqs does not return Quantity when it should'
    for ii in range(9):
        assert isinstance(aA.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii],units.Quantity), 'actionAngleIsochrone method actionsFreqsAngles does not return Quantity when it should'
    # actionAngleIsochroneApprox
    aA= actionAngleIsochroneApprox(pot=MWPotential,b=0.8,ro=8.,vo=220.)
    for ii in range(3):
        assert isinstance(aA(1.1,0.1,1.1,0.1,0.2,0.)[ii],units.Quantity), 'actionAngleIsochrone method __call__ does not return Quantity when it should'
    for ii in range(6):
        assert isinstance(aA.actionsFreqs(1.1,0.1,1.1,0.1,0.2,0.)[ii],units.Quantity), 'actionAngleIsochrone method actionsFreqs does not return Quantity when it should'
    for ii in range(9):
        assert isinstance(aA.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii],units.Quantity), 'actionAngleIsochrone method actionsFreqsAngles does not return Quantity when it should'
    return None

def test_actionAngle_method_returnunit():
    from galpy.actionAngle import actionAngleIsochrone, actionAngleSpherical, \
        actionAngleAdiabatic, actionAngleStaeckel, actionAngleIsochroneApprox
    from galpy.potential import PlummerPotential, MWPotential
    # actionAngleIsochrone
    aA= actionAngleIsochrone(b=0.8,ro=8.,vo=220.)
    for ii in range(3):
        try:
            aA(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(units.kpc*units.km/units.s)
        except units.UnitConversionError:
            raise AssertionError('actionAngle function __call__ does not return Quantity with the right units')
    for ii in range(3):
        try:
            aA.actionsFreqs(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(units.kpc*units.km/units.s)
        except units.UnitConversionError:
            raise AssertionError('actionAngle function actionsFreqs does not return Quantity with the right units')
    for ii in range(3,6):
        try:
            aA.actionsFreqs(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(1/units.Gyr)
        except units.UnitConversionError:
            raise AssertionError('actionAngle function actionsFreqs does not return Quantity with the right units')
    for ii in range(3):
        try:
            aA.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(units.kpc*units.km/units.s)
        except units.UnitConversionError:
            raise AssertionError('actionAngle function actionsFreqsAngles does not return Quantity with the right units')
    for ii in range(3,6):
        try:
            aA.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(1/units.Gyr)
        except units.UnitConversionError:
            raise AssertionError('actionAngle function actionsFreqsAngles does not return Quantity with the right units')
    for ii in range(6,9):
        try:
            aA.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(units.rad)
        except units.UnitConversionError:
            raise AssertionError('actionAngle function actionsFreqsAngles does not return Quantity with the right units')
    # actionAngleSpherical
    pot= PlummerPotential(normalize=1.,b=0.7)
    aA= actionAngleSpherical(pot=pot,ro=8.,vo=220.)
    for ii in range(3):
        try:
            aA(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(units.kpc*units.km/units.s)
        except units.UnitConversionError:
            raise AssertionError('actionAngle function __call__ does not return Quantity with the right units')
    for ii in range(3):
        try:
            aA.actionsFreqs(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(units.kpc*units.km/units.s)
        except units.UnitConversionError:
            raise AssertionError('actionAngle function actionsFreqs does not return Quantity with the right units')
    for ii in range(3,6):
        try:
            aA.actionsFreqs(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(1/units.Gyr)
        except units.UnitConversionError:
            raise AssertionError('actionAngle function actionsFreqs does not return Quantity with the right units')
    for ii in range(3):
        try:
            aA.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(units.kpc*units.km/units.s)
        except units.UnitConversionError:
            raise AssertionError('actionAngle function actionsFreqsAngles does not return Quantity with the right units')
    for ii in range(3,6):
        try:
            aA.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(1/units.Gyr)
        except units.UnitConversionError:
            raise AssertionError('actionAngle function actionsFreqsAngles does not return Quantity with the right units')
    for ii in range(6,9):
        try:
            aA.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(units.rad)
        except units.UnitConversionError:
            raise AssertionError('actionAngle function actionsFreqsAngles does not return Quantity with the right units')
    # actionAngleAdiabatic
    aA= actionAngleAdiabatic(pot=MWPotential,ro=8.,vo=220.)
    for ii in range(3):
        try:
            aA(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(units.kpc*units.km/units.s)
        except units.UnitConversionError:
            raise AssertionError('actionAngle function __call__ does not return Quantity with the right units')
    # actionAngleStaeckel
    aA= actionAngleStaeckel(pot=MWPotential,delta=0.45,ro=8.,vo=220.)
    for ii in range(3):
        try:
            aA(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(units.kpc*units.km/units.s)
        except units.UnitConversionError:
            raise AssertionError('actionAngle function __call__ does not return Quantity with the right units')
    for ii in range(3):
        try:
            aA.actionsFreqs(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(units.kpc*units.km/units.s)
        except units.UnitConversionError:
            raise AssertionError('actionAngle function actionsFreqs does not return Quantity with the right units')
    for ii in range(3,6):
        try:
            aA.actionsFreqs(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(1/units.Gyr)
        except units.UnitConversionError:
            raise AssertionError('actionAngle function actionsFreqs does not return Quantity with the right units')
    for ii in range(3):
        try:
            aA.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(units.kpc*units.km/units.s)
        except units.UnitConversionError:
            raise AssertionError('actionAngle function actionsFreqsAngles does not return Quantity with the right units')
    for ii in range(3,6):
        try:
            aA.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(1/units.Gyr)
        except units.UnitConversionError:
            raise AssertionError('actionAngle function actionsFreqsAngles does not return Quantity with the right units')
    for ii in range(6,9):
        try:
            aA.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(units.rad)
        except units.UnitConversionError:
            raise AssertionError('actionAngle function actionsFreqsAngles does not return Quantity with the right units')
    # actionAngleIsochroneApprox
    aA= actionAngleIsochroneApprox(pot=MWPotential,b=0.8,ro=8.,vo=220.)
    for ii in range(3):
        try:
            aA(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(units.kpc*units.km/units.s)
        except units.UnitConversionError:
            raise AssertionError('actionAngle function __call__ does not return Quantity with the right units')
    for ii in range(3):
        try:
            aA.actionsFreqs(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(units.kpc*units.km/units.s)
        except units.UnitConversionError:
            raise AssertionError('actionAngle function actionsFreqs does not return Quantity with the right units')
    for ii in range(3,6):
        try:
            aA.actionsFreqs(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(1/units.Gyr)
        except units.UnitConversionError:
            raise AssertionError('actionAngle function actionsFreqs does not return Quantity with the right units')
    for ii in range(3):
        try:
            aA.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(units.kpc*units.km/units.s)
        except units.UnitConversionError:
            raise AssertionError('actionAngle function actionsFreqsAngles does not return Quantity with the right units')
    for ii in range(3,6):
        try:
            aA.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(1/units.Gyr)
        except units.UnitConversionError:
            raise AssertionError('actionAngle function actionsFreqsAngles does not return Quantity with the right units')
    for ii in range(6,9):
        try:
            aA.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(units.rad)
        except units.UnitConversionError:
            raise AssertionError('actionAngle function actionsFreqsAngles does not return Quantity with the right units')
    return None

def test_actionAngle_method_value():
    from galpy.actionAngle import actionAngleIsochrone, actionAngleSpherical, \
        actionAngleAdiabatic, actionAngleStaeckel, actionAngleIsochroneApprox
    from galpy.potential import PlummerPotential, MWPotential
    from galpy.util import bovy_conversion
    ro,vo= 9.,230.
    # actionAngleIsochrone
    aA= actionAngleIsochrone(b=0.8,ro=ro,vo=vo)
    aAnu= actionAngleIsochrone(b=0.8)
    for ii in range(3):
        assert numpy.fabs(aA(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(units.kpc*units.km/units.s).value-aAnu(1.1,0.1,1.1,0.1,0.2,0.)[ii]*ro*vo) < 10.**-8., 'actionAngle function __call__ does not return Quantity with the right value'
    for ii in range(3):
        assert numpy.fabs(aA.actionsFreqs(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(units.kpc*units.km/units.s).value-aAnu.actionsFreqs(1.1,0.1,1.1,0.1,0.2,0.)[ii]*ro*vo) < 10.**-8., 'actionAngle function actionsFreqs does not return Quantity with the right value'
    for ii in range(3,6):
        assert numpy.fabs(aA.actionsFreqs(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(1/units.Gyr).value-aAnu.actionsFreqs(1.1,0.1,1.1,0.1,0.2,0.)[ii]*bovy_conversion.freq_in_Gyr(vo,ro)) < 10.**-8., 'actionAngle function actionsFreqs does not return Quantity with the right value'
    for ii in range(3):
        assert numpy.fabs(aA.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(units.kpc*units.km/units.s).value-aAnu.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii]*ro*vo) < 10.**-8., 'actionAngle function actionsFreqsAngles does not return Quantity with the right value'
    for ii in range(3,6):
        assert numpy.fabs(aA.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(1/units.Gyr).value-aAnu.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii]*bovy_conversion.freq_in_Gyr(vo,ro)) < 10.**-8., 'actionAngle function actionsFreqsAngles does not return Quantity with the right value'
    for ii in range(6,9):
        assert numpy.fabs(aA.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(units.rad).value-aAnu.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii]) < 10.**-8., 'actionAngle function actionsFreqsAngles does not return Quantity with the right value'
    # actionAngleSpherical
    pot= PlummerPotential(normalize=1.,b=0.7)
    aA= actionAngleSpherical(pot=pot,ro=ro,vo=vo)
    aAnu= actionAngleSpherical(pot=pot)
    for ii in range(3):
        assert numpy.fabs(aA(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(units.kpc*units.km/units.s).value-aAnu(1.1,0.1,1.1,0.1,0.2,0.)[ii]*ro*vo) < 10.**-8., 'actionAngle function __call__ does not return Quantity with the right value'
    for ii in range(3):
        assert numpy.fabs(aA.actionsFreqs(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(units.kpc*units.km/units.s).value-aAnu.actionsFreqs(1.1,0.1,1.1,0.1,0.2,0.)[ii]*ro*vo) < 10.**-8., 'actionAngle function actionsFreqs does not return Quantity with the right value'
    for ii in range(3,6):
        assert numpy.fabs(aA.actionsFreqs(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(1/units.Gyr).value-aAnu.actionsFreqs(1.1,0.1,1.1,0.1,0.2,0.)[ii]*bovy_conversion.freq_in_Gyr(vo,ro)) < 10.**-8., 'actionAngle function actionsFreqs does not return Quantity with the right value'
    for ii in range(3):
        assert numpy.fabs(aA.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(units.kpc*units.km/units.s).value-aAnu.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii]*ro*vo) < 10.**-8., 'actionAngle function actionsFreqsAngles does not return Quantity with the right value'
    for ii in range(3,6):
        assert numpy.fabs(aA.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(1/units.Gyr).value-aAnu.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii]*bovy_conversion.freq_in_Gyr(vo,ro)) < 10.**-8., 'actionAngle function actionsFreqsAngles does not return Quantity with the right value'
    for ii in range(6,9):
        assert numpy.fabs(aA.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(units.rad).value-aAnu.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii]) < 10.**-8., 'actionAngle function actionsFreqsAngles does not return Quantity with the right value'
    # actionAngleAdiabatic
    aA= actionAngleAdiabatic(pot=MWPotential,ro=ro,vo=vo)
    aAnu= actionAngleAdiabatic(pot=MWPotential)
    for ii in range(3):
        assert numpy.fabs(aA(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(units.kpc*units.km/units.s).value-aAnu(1.1,0.1,1.1,0.1,0.2,0.)[ii]*ro*vo) < 10.**-8., 'actionAngle function __call__ does not return Quantity with the right value'
    # actionAngleStaeckel
    aA= actionAngleStaeckel(pot=MWPotential,delta=0.45,ro=ro,vo=vo)
    aAnu= actionAngleStaeckel(pot=MWPotential,delta=0.45)
    for ii in range(3):
        assert numpy.fabs(aA(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(units.kpc*units.km/units.s).value-aAnu(1.1,0.1,1.1,0.1,0.2,0.)[ii]*ro*vo) < 10.**-8., 'actionAngle function __call__ does not return Quantity with the right value'
    for ii in range(3):
        assert numpy.fabs(aA.actionsFreqs(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(units.kpc*units.km/units.s).value-aAnu.actionsFreqs(1.1,0.1,1.1,0.1,0.2,0.)[ii]*ro*vo) < 10.**-8., 'actionAngle function actionsFreqs does not return Quantity with the right value'
    for ii in range(3,6):
        assert numpy.fabs(aA.actionsFreqs(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(1/units.Gyr).value-aAnu.actionsFreqs(1.1,0.1,1.1,0.1,0.2,0.)[ii]*bovy_conversion.freq_in_Gyr(vo,ro)) < 10.**-8., 'actionAngle function actionsFreqs does not return Quantity with the right value'
    for ii in range(3):
        assert numpy.fabs(aA.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(units.kpc*units.km/units.s).value-aAnu.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii]*ro*vo) < 10.**-8., 'actionAngle function actionsFreqsAngles does not return Quantity with the right value'
    for ii in range(3,6):
        assert numpy.fabs(aA.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(1/units.Gyr).value-aAnu.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii]*bovy_conversion.freq_in_Gyr(vo,ro)) < 10.**-8., 'actionAngle function actionsFreqsAngles does not return Quantity with the right value'
    for ii in range(6,9):
        assert numpy.fabs(aA.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(units.rad).value-aAnu.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii]) < 10.**-8., 'actionAngle function actionsFreqsAngles does not return Quantity with the right value'
    # actionAngleIsochroneApprox
    aA= actionAngleIsochroneApprox(pot=MWPotential,b=0.8,ro=ro,vo=vo)
    aAnu= actionAngleIsochroneApprox(pot=MWPotential,b=0.8)
    for ii in range(3):
        assert numpy.fabs(aA(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(units.kpc*units.km/units.s).value-aAnu(1.1,0.1,1.1,0.1,0.2,0.)[ii]*ro*vo) < 10.**-8., 'actionAngle function __call__ does not return Quantity with the right value'
    for ii in range(3):
        assert numpy.fabs(aA.actionsFreqs(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(units.kpc*units.km/units.s).value-aAnu.actionsFreqs(1.1,0.1,1.1,0.1,0.2,0.)[ii]*ro*vo) < 10.**-8., 'actionAngle function actionsFreqs does not return Quantity with the right value'
    for ii in range(3,6):
        assert numpy.fabs(aA.actionsFreqs(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(1/units.Gyr).value-aAnu.actionsFreqs(1.1,0.1,1.1,0.1,0.2,0.)[ii]*bovy_conversion.freq_in_Gyr(vo,ro)) < 10.**-8., 'actionAngle function actionsFreqs does not return Quantity with the right value'
    for ii in range(3):
        assert numpy.fabs(aA.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(units.kpc*units.km/units.s).value-aAnu.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii]*ro*vo) < 10.**-8., 'actionAngle function actionsFreqsAngles does not return Quantity with the right value'
    for ii in range(3,6):
        assert numpy.fabs(aA.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(1/units.Gyr).value-aAnu.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii]*bovy_conversion.freq_in_Gyr(vo,ro)) < 10.**-8., 'actionAngle function actionsFreqsAngles does not return Quantity with the right value'
    for ii in range(6,9):
        assert numpy.fabs(aA.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii].to(units.rad).value-aAnu.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii]) < 10.**-8., 'actionAngle function actionsFreqsAngles does not return Quantity with the right value'
    return None

def test_actionAngle_setup_roAsQuantity():
    from galpy.actionAngle import actionAngleIsochrone, actionAngleSpherical, \
        actionAngleAdiabatic, actionAngleStaeckel, actionAngleIsochroneApprox
    from galpy.potential import PlummerPotential, MWPotential
    # actionAngleIsochrone
    aA= actionAngleIsochrone(b=0.8,ro=7.*units.kpc)
    assert numpy.fabs(aA._ro-7.) < 10.**-10., 'ro in actionAngle setup as Quantity does not work as expected'
    # actionAngleSpherical
    pot= PlummerPotential(normalize=1.,b=0.7)
    aA= actionAngleSpherical(pot=pot,ro=7.*units.kpc)
    assert numpy.fabs(aA._ro-7.) < 10.**-10., 'ro in actionAngle setup as Quantity does not work as expected'
    # actionAngleAdiabatic
    aA= actionAngleAdiabatic(pot=MWPotential,ro=9.*units.kpc)
    assert numpy.fabs(aA._ro-9.) < 10.**-10., 'ro in actionAngle setup as Quantity does not work as expected'
    # actionAngleStaeckel
    aA= actionAngleStaeckel(pot=MWPotential,delta=0.45,ro=7.*units.kpc)
    assert numpy.fabs(aA._ro-7.) < 10.**-10., 'ro in actionAngle setup as Quantity does not work as expected'
    # actionAngleIsochroneApprox
    aA= actionAngleIsochroneApprox(pot=MWPotential,b=0.8,ro=7.*units.kpc)
    assert numpy.fabs(aA._ro-7.) < 10.**-10., 'ro in actionAngle setup as Quantity does not work as expected'
    return None

def test_actionAngle_setup_roAsQuantity_oddunits():
    from galpy.actionAngle import actionAngleIsochrone, actionAngleSpherical, \
        actionAngleAdiabatic, actionAngleStaeckel, actionAngleIsochroneApprox
    from galpy.potential import PlummerPotential, MWPotential
    # actionAngleIsochrone
    aA= actionAngleIsochrone(b=0.8,ro=7.*units.lyr)
    assert numpy.fabs(aA._ro-7.*units.lyr.to(units.kpc)) < 10.**-10., 'ro in actionAngle setup as Quantity does not work as expected'
    # actionAngleSpherical
    pot= PlummerPotential(normalize=1.,b=0.7)
    aA= actionAngleSpherical(pot=pot,ro=7.*units.lyr)
    assert numpy.fabs(aA._ro-7.*units.lyr.to(units.kpc)) < 10.**-10., 'ro in actionAngle setup as Quantity does not work as expected'
    # actionAngleAdiabatic
    aA= actionAngleAdiabatic(pot=MWPotential,ro=7.*units.lyr)
    assert numpy.fabs(aA._ro-7.*units.lyr.to(units.kpc)) < 10.**-10., 'ro in actionAngle setup as Quantity does not work as expected'
    # actionAngleStaeckel
    aA= actionAngleStaeckel(pot=MWPotential,delta=0.45,ro=7.*units.lyr)
    assert numpy.fabs(aA._ro-7.*units.lyr.to(units.kpc)) < 10.**-10., 'ro in actionAngle setup as Quantity does not work as expected'
    # actionAngleIsochroneApprox
    aA= actionAngleIsochroneApprox(pot=MWPotential,b=0.8,ro=7.*units.lyr)
    assert numpy.fabs(aA._ro-7.*units.lyr.to(units.kpc)) < 10.**-10., 'ro in actionAngle setup as Quantity does not work as expected'
    return None

def test_actionAngle_setup_voAsQuantity():
    from galpy.actionAngle import actionAngleIsochrone, actionAngleSpherical, \
        actionAngleAdiabatic, actionAngleStaeckel, actionAngleIsochroneApprox
    from galpy.potential import PlummerPotential, MWPotential
    # actionAngleIsochrone
    aA= actionAngleIsochrone(b=0.8,vo=230.*units.km/units.s)
    assert numpy.fabs(aA._vo-230.) < 10.**-10., 'ro in actionAngle setup as Quantity does not work as expected'
    # actionAngleSpherical
    pot= PlummerPotential(normalize=1.,b=0.7)
    aA= actionAngleSpherical(pot=pot,vo=230.*units.km/units.s)
    assert numpy.fabs(aA._vo-230.) < 10.**-10., 'ro in actionAngle setup as Quantity does not work as expected'
    # actionAngleAdiabatic
    aA= actionAngleAdiabatic(pot=MWPotential,ro=9.*units.kpc)
    assert numpy.fabs(aA._ro-9.) < 10.**-10., 'ro in actionAngle setup as Quantity does not work as expected'
    # actionAngleStaeckel
    aA= actionAngleStaeckel(pot=MWPotential,delta=0.45,vo=230.*units.km/units.s)
    assert numpy.fabs(aA._vo-230.) < 10.**-10., 'ro in actionAngle setup as Quantity does not work as expected'
    # actionAngleIsochroneApprox
    aA= actionAngleIsochroneApprox(pot=MWPotential,b=0.8,vo=230.*units.km/units.s)
    assert numpy.fabs(aA._vo-230.) < 10.**-10., 'ro in actionAngle setup as Quantity does not work as expected'
    return None

def test_actionAngle_setup_voAsQuantity_oddunits():
    from galpy.actionAngle import actionAngleIsochrone, actionAngleSpherical, \
        actionAngleAdiabatic, actionAngleStaeckel, actionAngleIsochroneApprox
    from galpy.potential import PlummerPotential, MWPotential
    # actionAngleIsochrone
    aA= actionAngleIsochrone(b=0.8,vo=230.*units.pc/units.Myr)
    assert numpy.fabs(aA._vo-230.*(units.pc/units.Myr).to(units.km/units.s)) < 10.**-10., 'ro in actionAngle setup as Quantity does not work as expected'
    # actionAngleSpherical
    pot= PlummerPotential(normalize=1.,b=0.7)
    aA= actionAngleSpherical(pot=pot,vo=230.*units.pc/units.Myr)
    assert numpy.fabs(aA._vo-230.*(units.pc/units.Myr).to(units.km/units.s)) < 10.**-10., 'ro in actionAngle setup as Quantity does not work as expected'
    # actionAngleAdiabatic
    aA= actionAngleAdiabatic(pot=MWPotential,ro=9.*units.kpc)
    assert numpy.fabs(aA._ro-9.) < 10.**-10., 'ro in actionAngle setup as Quantity does not work as expected'
    # actionAngleStaeckel
    aA= actionAngleStaeckel(pot=MWPotential,delta=0.45,vo=230.*units.pc/units.Myr)
    assert numpy.fabs(aA._vo-230.*(units.pc/units.Myr).to(units.km/units.s)) < 10.**-10., 'ro in actionAngle setup as Quantity does not work as expected'
    # actionAngleIsochroneApprox
    aA= actionAngleIsochroneApprox(pot=MWPotential,b=0.8,vo=230.*units.pc/units.Myr)
    assert numpy.fabs(aA._vo-230.*(units.pc/units.Myr).to(units.km/units.s)) < 10.**-10., 'ro in actionAngle setup as Quantity does not work as expected'
    return None

def test_actionAngleStaeckel_setup_delta_units():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.potential import MWPotential
    ro= 9.
    aA= actionAngleStaeckel(pot=MWPotential,delta=0.45*ro*units.kpc,ro=ro)
    aAu= actionAngleStaeckel(pot=MWPotential,delta=0.45)
    assert numpy.fabs(aA._delta-aAu._delta) < 10.**-10., 'delta with units in actionAngleStaeckel setup does not work as expected'
    return None

def test_actionAngleIsochrone_setup_b_units():
    from galpy.actionAngle import actionAngleIsochrone
    ro= 9.
    aA= actionAngleIsochrone(b=0.7*ro*units.kpc,ro=ro)
    aAu= actionAngleIsochrone(b=0.7)
    assert numpy.fabs(aA.b-aAu.b) < 10.**-10., 'b with units in actionAngleIsochrone setup does not work as expected'
    return None

def test_actionAngleIsochroneApprix_setup_b_units():
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.potential import MWPotential
    ro= 9.
    aA= actionAngleIsochroneApprox(pot=MWPotential,b=0.7*ro*units.kpc,ro=ro)
    aAu= actionAngleIsochroneApprox(pot=MWPotential,b=0.7)
    assert numpy.fabs(aA._aAI.b-aAu._aAI.b) < 10.**-10., 'b with units in actionAngleIsochroneApprox setup does not work as expected'
    return None

def test_actionAngleIsochroneApprix_setup_tintJ_units():
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.potential import MWPotential
    from galpy.util import bovy_conversion
    ro= 9.
    vo= 230.
    aA= actionAngleIsochroneApprox(pot=MWPotential,b=0.7,
                                   tintJ=11.*units.Gyr,ro=ro,vo=vo)
    aAu= actionAngleIsochroneApprox(pot=MWPotential,b=0.7,
                                    tintJ=11./bovy_conversion.time_in_Gyr(vo,ro))
    assert numpy.fabs(aA._tintJ-aAu._tintJ) < 10.**-10., 'tintJ with units in actionAngleIsochroneApprox setup does not work as expected'
    return None

def test_actionAngle_method_inputAsQuantity():
    from galpy.actionAngle import actionAngleIsochrone, actionAngleSpherical, \
        actionAngleAdiabatic, actionAngleStaeckel, actionAngleIsochroneApprox
    from galpy.potential import PlummerPotential, MWPotential
    ro,vo= 9.,230.
    # actionAngleIsochrone
    aA= actionAngleIsochrone(b=0.8,ro=ro,vo=vo)
    aAnu= actionAngleIsochrone(b=0.8)
    for ii in range(3):
        assert numpy.fabs(aA(1.1*ro*units.kpc,0.1*vo*units.km/units.s,1.1*vo*units.km/units.s,0.1*ro*units.kpc,0.2*vo*units.km/units.s,0.*units.rad,use_physical=False)[ii]-aAnu(1.1,0.1,1.1,0.1,0.2,0.)[ii]) < 10.**-8., 'actionAngle method __call__ does not return the correct value when input is Quantity'
    for ii in range(3):
        assert numpy.fabs(aA.actionsFreqs(1.1*ro*units.kpc,0.1*vo*units.km/units.s,1.1*vo*units.km/units.s,0.1*ro*units.kpc,0.2*vo*units.km/units.s,0.*units.rad,use_physical=False)[ii]-aAnu.actionsFreqs(1.1,0.1,1.1,0.1,0.2,0.)[ii]) < 10.**-8., 'actionAngle method actionsFreqs does not return the correct value when input is Quantity'
    for ii in range(3,6):
        assert numpy.fabs(aA.actionsFreqs(1.1*ro*units.kpc,0.1*vo*units.km/units.s,1.1*vo*units.km/units.s,0.1*ro*units.kpc,0.2*vo*units.km/units.s,0.*units.rad,use_physical=False)[ii]-aAnu.actionsFreqs(1.1,0.1,1.1,0.1,0.2,0.)[ii]) < 10.**-8., 'actionAngle method actionsFreqs does not return the correct value when input is Quantity'
    for ii in range(3):
        assert numpy.fabs(aA.actionsFreqsAngles(1.1*ro*units.kpc,0.1*vo*units.km/units.s,1.1*vo*units.km/units.s,0.1*ro*units.kpc,0.2*vo*units.km/units.s,0.*units.rad,use_physical=False)[ii]-aAnu.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii]) < 10.**-8., 'actionAngle method actionsFreqsAngles does not return the correct value when input is Quantity'
    for ii in range(3,6):
        assert numpy.fabs(aA.actionsFreqsAngles(1.1*ro*units.kpc,0.1*vo*units.km/units.s,1.1*vo*units.km/units.s,0.1*ro*units.kpc,0.2*vo*units.km/units.s,0.*units.rad,use_physical=False)[ii]-aAnu.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii]) < 10.**-8., 'actionAngle method actionsFreqsAngles does not return the correct value when input is Quantity'
    for ii in range(6,9):
        assert numpy.fabs(aA.actionsFreqsAngles(1.1*ro*units.kpc,0.1*vo*units.km/units.s,1.1*vo*units.km/units.s,0.1*ro*units.kpc,0.2*vo*units.km/units.s,0.*units.rad,use_physical=False)[ii]-aAnu.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii]) < 10.**-8., 'actionAngle method actionsFreqsAngles does not return the correct value when input is Quantity'
    # actionAngleSpherical
    pot= PlummerPotential(normalize=1.,b=0.7)
    aA= actionAngleSpherical(pot=pot,ro=ro,vo=vo)
    aAnu= actionAngleSpherical(pot=pot)
    for ii in range(3):
        assert numpy.fabs(aA(1.1*ro*units.kpc,0.1*vo*units.km/units.s,1.1*vo*units.km/units.s,0.1*ro*units.kpc,0.2*vo*units.km/units.s,0.*units.rad,use_physical=False)[ii]-aAnu(1.1,0.1,1.1,0.1,0.2,0.)[ii]) < 10.**-8., 'actionAngle method __call__ does not return the correct value when input is Quantity'
    for ii in range(3):
        assert numpy.fabs(aA.actionsFreqs(1.1*ro*units.kpc,0.1*vo*units.km/units.s,1.1*vo*units.km/units.s,0.1*ro*units.kpc,0.2*vo*units.km/units.s,0.*units.rad,use_physical=False)[ii]-aAnu.actionsFreqs(1.1,0.1,1.1,0.1,0.2,0.)[ii]) < 10.**-8., 'actionAngle method actionsFreqs does not return the correct value when input is Quantity'
    for ii in range(3,6):
        assert numpy.fabs(aA.actionsFreqs(1.1*ro*units.kpc,0.1*vo*units.km/units.s,1.1*vo*units.km/units.s,0.1*ro*units.kpc,0.2*vo*units.km/units.s,0.*units.rad,use_physical=False)[ii]-aAnu.actionsFreqs(1.1,0.1,1.1,0.1,0.2,0.)[ii]) < 10.**-8., 'actionAngle method actionsFreqs does not return the correct value when input is Quantity'
    for ii in range(3):
        assert numpy.fabs(aA.actionsFreqsAngles(1.1*ro*units.kpc,0.1*vo*units.km/units.s,1.1*vo*units.km/units.s,0.1*ro*units.kpc,0.2*vo*units.km/units.s,0.*units.rad,use_physical=False)[ii]-aAnu.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii]) < 10.**-8., 'actionAngle method actionsFreqsAngles does not return the correct value when input is Quantity'
    for ii in range(3,6):
        assert numpy.fabs(aA.actionsFreqsAngles(1.1*ro*units.kpc,0.1*vo*units.km/units.s,1.1*vo*units.km/units.s,0.1*ro*units.kpc,0.2*vo*units.km/units.s,0.*units.rad,use_physical=False)[ii]-aAnu.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii]) < 10.**-8., 'actionAngle method actionsFreqsAngles does not return the correct value when input is Quantity'
    for ii in range(6,9):
        assert numpy.fabs(aA.actionsFreqsAngles(1.1*ro*units.kpc,0.1*vo*units.km/units.s,1.1*vo*units.km/units.s,0.1*ro*units.kpc,0.2*vo*units.km/units.s,0.*units.rad,use_physical=False)[ii]-aAnu.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii]) < 10.**-8., 'actionAngle method actionsFreqsAngles does not return the correct value when input is Quantity'
    # actionAngleAdiabatic
    aA= actionAngleAdiabatic(pot=MWPotential,ro=ro,vo=vo)
    aAnu= actionAngleAdiabatic(pot=MWPotential)
    for ii in range(3):
        assert numpy.fabs(aA(1.1*ro*units.kpc,0.1*vo*units.km/units.s,1.1*vo*units.km/units.s,0.1*ro*units.kpc,0.2*vo*units.km/units.s,0.*units.rad,use_physical=False)[ii]-aAnu(1.1,0.1,1.1,0.1,0.2,0.)[ii]) < 10.**-8., 'actionAngle method __call__ does not return the correct value when input is Quantity'
    # actionAngleStaeckel
    aA= actionAngleStaeckel(pot=MWPotential,delta=0.45,ro=ro,vo=vo)
    aAnu= actionAngleStaeckel(pot=MWPotential,delta=0.45)
    for ii in range(3):
        assert numpy.fabs(aA(1.1*ro*units.kpc,0.1*vo*units.km/units.s,1.1*vo*units.km/units.s,0.1*ro*units.kpc,0.2*vo*units.km/units.s,0.*units.rad,use_physical=False)[ii]-aAnu(1.1,0.1,1.1,0.1,0.2,0.)[ii]) < 10.**-8., 'actionAngle method __call__ does not return the correct value when input is Quantity'
    for ii in range(3):
        assert numpy.fabs(aA.actionsFreqs(1.1*ro*units.kpc,0.1*vo*units.km/units.s,1.1*vo*units.km/units.s,0.1*ro*units.kpc,0.2*vo*units.km/units.s,0.*units.rad,use_physical=False)[ii]-aAnu.actionsFreqs(1.1,0.1,1.1,0.1,0.2,0.)[ii]) < 10.**-8., 'actionAngle method actionsFreqs does not return the correct value when input is Quantity'
    for ii in range(3,6):
        assert numpy.fabs(aA.actionsFreqs(1.1*ro*units.kpc,0.1*vo*units.km/units.s,1.1*vo*units.km/units.s,0.1*ro*units.kpc,0.2*vo*units.km/units.s,0.*units.rad,use_physical=False)[ii]-aAnu.actionsFreqs(1.1,0.1,1.1,0.1,0.2,0.)[ii]) < 10.**-8., 'actionAngle method actionsFreqs does not return the correct value when input is Quantity'
    for ii in range(3):
        assert numpy.fabs(aA.actionsFreqsAngles(1.1*ro*units.kpc,0.1*vo*units.km/units.s,1.1*vo*units.km/units.s,0.1*ro*units.kpc,0.2*vo*units.km/units.s,0.*units.rad,use_physical=False)[ii]-aAnu.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii]) < 10.**-8., 'actionAngle method actionsFreqsAngles does not return the correct value when input is Quantity'
    for ii in range(3,6):
        assert numpy.fabs(aA.actionsFreqsAngles(1.1*ro*units.kpc,0.1*vo*units.km/units.s,1.1*vo*units.km/units.s,0.1*ro*units.kpc,0.2*vo*units.km/units.s,0.*units.rad,use_physical=False)[ii]-aAnu.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii]) < 10.**-8., 'actionAngle method actionsFreqsAngles does not return the correct value when input is Quantity'
    for ii in range(6,9):
        assert numpy.fabs(aA.actionsFreqsAngles(1.1*ro*units.kpc,0.1*vo*units.km/units.s,1.1*vo*units.km/units.s,0.1*ro*units.kpc,0.2*vo*units.km/units.s,0.*units.rad,use_physical=False)[ii]-aAnu.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii]) < 10.**-8., 'actionAngle method actionsFreqsAngles does not return the correct value when input is Quantity'
    # actionAngleIsochroneApprox
    aA= actionAngleIsochroneApprox(pot=MWPotential,b=0.8,ro=ro,vo=vo)
    aAnu= actionAngleIsochroneApprox(pot=MWPotential,b=0.8)
    for ii in range(3):
        assert numpy.fabs(aA(1.1*ro*units.kpc,0.1*vo*units.km/units.s,1.1*vo*units.km/units.s,0.1*ro*units.kpc,0.2*vo*units.km/units.s,0.*units.rad,use_physical=False)[ii]-aAnu(1.1,0.1,1.1,0.1,0.2,0.)[ii]) < 10.**-8., 'actionAngle method __call__ does not return the correct value when input is Quantity'
    for ii in range(3):
        assert numpy.fabs(aA.actionsFreqs(1.1*ro*units.kpc,0.1*vo*units.km/units.s,1.1*vo*units.km/units.s,0.1*ro*units.kpc,0.2*vo*units.km/units.s,0.*units.rad,use_physical=False)[ii]-aAnu.actionsFreqs(1.1,0.1,1.1,0.1,0.2,0.)[ii]) < 10.**-8., 'actionAngle method actionsFreqs does not return the correct value when input is Quantity'
    for ii in range(3,6):
        assert numpy.fabs(aA.actionsFreqs(1.1*ro*units.kpc,0.1*vo*units.km/units.s,1.1*vo*units.km/units.s,0.1*ro*units.kpc,0.2*vo*units.km/units.s,0.*units.rad,use_physical=False)[ii]-aAnu.actionsFreqs(1.1,0.1,1.1,0.1,0.2,0.)[ii]) < 10.**-8., 'actionAngle method actionsFreqs does not return the correct value when input is Quantity'
    for ii in range(3):
        assert numpy.fabs(aA.actionsFreqsAngles(1.1*ro*units.kpc,0.1*vo*units.km/units.s,1.1*vo*units.km/units.s,0.1*ro*units.kpc,0.2*vo*units.km/units.s,0.*units.rad,use_physical=False)[ii]-aAnu.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii]) < 10.**-8., 'actionAngle method actionsFreqsAngles does not return the correct value when input is Quantity'
    for ii in range(3,6):
        assert numpy.fabs(aA.actionsFreqsAngles(1.1*ro*units.kpc,0.1*vo*units.km/units.s,1.1*vo*units.km/units.s,0.1*ro*units.kpc,0.2*vo*units.km/units.s,0.*units.rad,use_physical=False)[ii]-aAnu.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii]) < 10.**-8., 'actionAngle method actionsFreqsAngles does not return the correct value when input is Quantity'
    for ii in range(6,9):
        assert numpy.fabs(aA.actionsFreqsAngles(1.1*ro*units.kpc,0.1*vo*units.km/units.s,1.1*vo*units.km/units.s,0.1*ro*units.kpc,0.2*vo*units.km/units.s,0.*units.rad,use_physical=False)[ii]-aAnu.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii]) < 10.**-8., 'actionAngle method actionsFreqsAngles does not return the correct value when input is Quantity'
    return None

def test_actionAngleIsochroneApprox_method_ts_units():   
    from galpy.potential import IsochronePotential
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.orbit import Orbit
    from galpy.util import bovy_conversion
    ip= IsochronePotential(normalize=1.,b=1.2)
    ro, vo= 7.5, 215.
    aAIA= actionAngleIsochroneApprox(pot=ip,b=0.8,ro=ro,vo=vo)
    R,vR,vT,z,vz,phi= 1.1, 0.3, 1.2, 0.2,0.5,2.
    #Setup an orbit, and integrated it first
    o= Orbit([R,vR,vT,z,vz,phi])
    ts= numpy.linspace(0.,10.,25000)*units.Gyr #Integrate for a long time, not the default
    o.integrate(ts,ip)
    jiaO= aAIA.actionsFreqs(o,ts=ts)
    jiaOu= aAIA.actionsFreqs(o,ts=ts.value/bovy_conversion.time_in_Gyr(vo,ro))
    dOr= numpy.fabs((jiaO[3]-jiaOu[3])/jiaO[3])
    dOp= numpy.fabs((jiaO[4]-jiaOu[4])/jiaO[4])
    dOz= numpy.fabs((jiaO[5]-jiaOu[5])/jiaO[5])
    assert dOr < 10.**-6., 'actionAngleIsochroneApprox with ts with units fails'
    assert dOp < 10.**-6., 'actionAngleIsochroneApprox with ts with units fails'
    assert dOz < 10.**-6., 'actionAngleIsochroneApprox with ts with units fails'
    # Same for actionsFreqsAngles
    jiaO= aAIA.actionsFreqsAngles(o,ts=ts)
    jiaOu= aAIA.actionsFreqsAngles(o,ts=ts.value/bovy_conversion.time_in_Gyr(vo,ro))
    dOr= numpy.fabs((jiaO[3]-jiaOu[3])/jiaO[3])
    dOp= numpy.fabs((jiaO[4]-jiaOu[4])/jiaO[4])
    dOz= numpy.fabs((jiaO[5]-jiaOu[5])/jiaO[5])
    assert dOr < 10.**-6., 'actionAngleIsochroneApprox with ts with units fails'
    assert dOp < 10.**-6., 'actionAngleIsochroneApprox with ts with units fails'
    assert dOz < 10.**-6., 'actionAngleIsochroneApprox with ts with units fails'
    return None

def test_actionAngle_inconsistentPotentialUnits_error():
    from galpy.actionAngle import actionAngleIsochrone, actionAngleSpherical, \
        actionAngleAdiabatic, actionAngleStaeckel, actionAngleIsochroneApprox
    from galpy.potential import PlummerPotential, IsochronePotential
    # actionAngleIsochrone
    pot= IsochronePotential(normalize=1.,ro=7.,vo=220.)
    assert_raises(AssertionError,
                  lambda x: actionAngleIsochrone(ip=pot,ro=8.,vo=220.),())
    pot= IsochronePotential(normalize=1.,ro=8.,vo=230.)
    assert_raises(AssertionError,
                  lambda x: actionAngleIsochrone(ip=pot,ro=8.,vo=220.),())
    # actionAngleSpherical
    pot= PlummerPotential(normalize=1.,b=0.7,ro=7.,vo=220.)
    assert_raises(AssertionError,
                  lambda x: actionAngleSpherical(pot=pot,ro=8.,vo=220.),())
    pot= PlummerPotential(normalize=1.,b=0.7,ro=8.,vo=230.)
    assert_raises(AssertionError,
                  lambda x: actionAngleSpherical(pot=pot,ro=8.,vo=220.),())
    # actionAngleAdiabatic
    pot= PlummerPotential(normalize=1.,b=0.7,ro=7.,vo=220.)
    assert_raises(AssertionError,
                  lambda x: actionAngleAdiabatic(pot=[pot],ro=8.,vo=220.),())
    pot= PlummerPotential(normalize=1.,b=0.7,ro=8.,vo=230.)
    assert_raises(AssertionError,
                  lambda x: actionAngleAdiabatic(pot=[pot],ro=8.,vo=220.),())
    # actionAngleStaeckel
    pot= PlummerPotential(normalize=1.,b=0.7,ro=7.,vo=220.)
    assert_raises(AssertionError,
                  lambda x: actionAngleStaeckel(delta=0.45,pot=pot,ro=8.,vo=220.),())
    pot= PlummerPotential(normalize=1.,b=0.7,ro=8.,vo=230.)
    assert_raises(AssertionError,
                  lambda x: actionAngleStaeckel(delta=0.45,pot=pot,ro=8.,vo=220.),())
    # actionAngleIsochroneApprox
    pot= PlummerPotential(normalize=1.,b=0.7,ro=7.,vo=220.)
    assert_raises(AssertionError,
                  lambda x: actionAngleIsochroneApprox(b=0.8,pot=pot,ro=8.,vo=220.),())
    pot= PlummerPotential(normalize=1.,b=0.7,ro=8.,vo=230.)
    assert_raises(AssertionError,
                  lambda x: actionAngleIsochroneApprox(b=0.8,pot=pot,ro=8.,vo=220.),())
    return None

def test_estimateDeltaStaeckel_method_returntype():
    from galpy.potential import MiyamotoNagaiPotential
    from galpy.actionAngle import estimateDeltaStaeckel
    pot= MiyamotoNagaiPotential(normalize=True,ro=8.,vo=220.)
    assert isinstance(estimateDeltaStaeckel(pot,1.1,0.1),units.Quantity), 'estimateDeltaStaeckel function does not return Quantity when it should'
    assert isinstance(estimateDeltaStaeckel(pot,1.1*numpy.ones(3),0.1*numpy.ones(3)),units.Quantity), 'estimateDeltaStaeckel function does not return Quantity when it should'
    return None

def test_estimateDeltaStaeckel_method_returnunit():
    from galpy.potential import MiyamotoNagaiPotential
    from galpy.actionAngle import estimateDeltaStaeckel
    pot= MiyamotoNagaiPotential(normalize=True,ro=8.,vo=220.)
    try:
        estimateDeltaStaeckel(pot,1.1,0.1).to(units.kpc)
    except units.UnitConversionError:
        raise AssertionError('estimateDeltaStaeckel function does not return Quantity with the right units')
    try:
        estimateDeltaStaeckel(pot,1.1*numpy.ones(3),0.1*numpy.ones(3)).to(units.kpc)
    except units.UnitConversionError:
        raise AssertionError('estimateDeltaStaeckel function does not return Quantity with the right units')
    return None

def test_estimateDeltaStaeckel_method_value():
    from galpy.potential import MiyamotoNagaiPotential
    from galpy.actionAngle import estimateDeltaStaeckel
    ro, vo= 9., 230.
    pot= MiyamotoNagaiPotential(normalize=True,ro=ro,vo=vo)
    potu= MiyamotoNagaiPotential(normalize=True)
    assert numpy.fabs(estimateDeltaStaeckel(pot,1.1*ro*units.kpc,0.1*ro*units.kpc).to(units.kpc).value-estimateDeltaStaeckel(potu,1.1,0.1)*ro) < 10.**-8., 'estimateDeltaStaeckel function does not return Quantity with the right value'
    assert numpy.all(numpy.fabs(estimateDeltaStaeckel(pot,1.1*numpy.ones(3),0.1*numpy.ones(3)).to(units.kpc).value-estimateDeltaStaeckel(potu,1.1*numpy.ones(3),0.1*numpy.ones(3))*ro) < 10.**-8.), 'estimateDeltaStaeckel function does not return Quantity with the right value'
    return None

def test_estimateBIsochrone_method_returntype():
    from galpy.potential import MiyamotoNagaiPotential
    from galpy.actionAngle import estimateBIsochrone
    pot= MiyamotoNagaiPotential(normalize=True,ro=8.,vo=220.)
    assert isinstance(estimateBIsochrone(pot,1.1,0.1),units.Quantity), 'estimateBIsochrone function does not return Quantity when it should'
    for ii in range(3):
        assert isinstance(estimateBIsochrone(pot,1.1*numpy.ones(3),0.1*numpy.ones(3))[ii],units.Quantity), 'estimateBIsochrone function does not return Quantity when it should'
    return None

def test_estimateBIsochrone_method_returnunit():
    from galpy.potential import MiyamotoNagaiPotential
    from galpy.actionAngle import estimateBIsochrone
    pot= MiyamotoNagaiPotential(normalize=True,ro=8.,vo=220.)
    try:
        estimateBIsochrone(pot,1.1,0.1).to(units.kpc)
    except units.UnitConversionError:
        raise AssertionError('estimateBIsochrone function does not return Quantity with the right units')
    for ii in range(3):
        try:
            estimateBIsochrone(pot,1.1*numpy.ones(3),0.1*numpy.ones(3))[ii].to(units.kpc)
        except units.UnitConversionError:
            raise AssertionError('estimateBIsochrone function does not return Quantity with the right units')
    return None

def test_estimateBIsochrone_method_value():
    from galpy.potential import MiyamotoNagaiPotential
    from galpy.actionAngle import estimateBIsochrone
    ro, vo= 9., 230.
    pot= MiyamotoNagaiPotential(normalize=True,ro=ro,vo=vo)
    potu= MiyamotoNagaiPotential(normalize=True)
    assert numpy.fabs(estimateBIsochrone(pot,1.1*ro*units.kpc,0.1*ro*units.kpc).to(units.kpc).value-estimateBIsochrone(potu,1.1,0.1)*ro) < 10.**-8., 'estimateBIsochrone function does not return Quantity with the right value'
    for ii in range(3):
        assert numpy.all(numpy.fabs(estimateBIsochrone(pot,1.1*numpy.ones(3),0.1*numpy.ones(3))[ii].to(units.kpc).value-estimateBIsochrone(potu,1.1*numpy.ones(3),0.1*numpy.ones(3))[ii]*ro) < 10.**-8.), 'estimateBIsochrone function does not return Quantity with the right value'
    return None

