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
