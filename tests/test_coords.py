from __future__ import print_function, division
import numpy
from galpy.util import bovy_coords
import pytest

def test_radec_to_lb_ngp():
    _turn_off_apy()
    # Test that the NGP is at b=90
    ra, dec= 192.25, 27.4
    lb= bovy_coords.radec_to_lb(ra,dec,degree=True,epoch=1950.)
    assert numpy.fabs(lb[1]-90.) < 10.**-8., 'Galactic latitude of the NGP given in ra,dec is not 90'
    # Also test this for degree=False
    lb= bovy_coords.radec_to_lb(ra/180.*numpy.pi,dec/180.*numpy.pi,
                                degree=False,epoch=1950.)
    assert numpy.fabs(lb[1]-numpy.pi/2.) < 10.**-8., 'Galactic latitude of the NGP given in ra,dec is not pi/2'
    _turn_on_apy()
    return None

def test_radec_to_lb_ngp_apyangles():
    # Test, but using transformation angles derived from astropy
    _turn_off_apy(keep_loaded=True)
    # Test that the NGP is at b=90
    ra, dec= 192.25, 27.4
    lb= bovy_coords.radec_to_lb(ra,dec,degree=True,epoch='B1950')
    assert numpy.fabs(lb[1]-90.) < 10.**-4., 'Galactic latitude of the NGP given in ra,dec is not 90'
    # Also test this for degree=False
    lb= bovy_coords.radec_to_lb(ra/180.*numpy.pi,dec/180.*numpy.pi,
                                degree=False,epoch='B1950')
    assert numpy.fabs(lb[1]-numpy.pi/2.) < 10.**-4., 'Galactic latitude of the NGP given in ra,dec is not pi/2'
    _turn_on_apy()
    return None

def test_radec_to_lb_ngp_apy():
    # Test that the NGP is at b=90, using astropy's coordinate transformations
    ra, dec= 192.25, 27.4
    lb= bovy_coords.radec_to_lb(ra,dec,degree=True,epoch=1950.)
    assert numpy.fabs(lb[1]-90.) < 10.**-4., 'Galactic latitude of the NGP given in ra,dec is not 90'
    # Also test this for degree=False
    lb= bovy_coords.radec_to_lb(ra/180.*numpy.pi,dec/180.*numpy.pi,
                                degree=False,epoch=1950.)
    assert numpy.fabs(lb[1]-numpy.pi/2.) < 10.**-4., 'Galactic latitude of the NGP given in ra,dec is not pi/2'
    return None

def test_radec_to_lb_ngp_j2000():
    _turn_off_apy()
    # Test that the NGP is at b=90
    ra, dec= 192.8594812065348, 27.12825118085622
    lb= bovy_coords.radec_to_lb(ra,dec,degree=True,epoch=2000.)
    assert numpy.fabs(lb[1]-90.) < 10.**-8., 'Galactic latitude of the NGP given in ra,dec is not 90'
    # Also test this for degree=False
    lb= bovy_coords.radec_to_lb(ra/180.*numpy.pi,dec/180.*numpy.pi,
                                degree=False,epoch=2000.)
    assert numpy.fabs(lb[1]-numpy.pi/2.) < 10.**-8., 'Galactic latitude of the NGP given in ra,dec is not pi/2'
    _turn_on_apy()
    return None

def test_radec_to_lb_ngp_j2000_apy():
    # Test that the NGP is at b=90
    ra, dec= 192.8594812065348, 27.12825118085622
    lb= bovy_coords.radec_to_lb(ra,dec,degree=True,epoch=2000.)
    assert numpy.fabs(lb[1]-90.) < 10.**-4., 'Galactic latitude of the NGP given in ra,dec is not 90'
    # Also test this for degree=False
    lb= bovy_coords.radec_to_lb(ra/180.*numpy.pi,dec/180.*numpy.pi,
                                degree=False,epoch=2000.)
    assert numpy.fabs(lb[1]-numpy.pi/2.) < 10.**-4., 'Galactic latitude of the NGP given in ra,dec is not pi/2'
    return None

def test_radec_to_lb_ngp_j2000_apyangles():
    # Same test, but using transformation angles derived from astropy
    _turn_off_apy(keep_loaded=True)
    # Test that the NGP is at b=90
    ra, dec= 192.8594812065348, 27.12825118085622
    lb= bovy_coords.radec_to_lb(ra,dec,degree=True,epoch='J2000')
    assert numpy.fabs(lb[1]-90.) < 10.**-4., 'Galactic latitude of the NGP given in ra,dec is not 90'
    # Also test this for degree=False
    lb= bovy_coords.radec_to_lb(ra/180.*numpy.pi,dec/180.*numpy.pi,
                                degree=False,epoch='J2000')
    assert numpy.fabs(lb[1]-numpy.pi/2.) < 10.**-4., 'Galactic latitude of the NGP given in ra,dec is not pi/2'
    _turn_on_apy()
    return None

def test_radec_to_lb_ngp_j2000_apyangles_icrs():
    # Test, but using transformation angles derived from astropy, for ICRS
    _turn_off_apy(keep_loaded=True)
    # Test that the NGP is at b=90
    ra, dec= 192.8594812065348, 27.12825118085622
    lb= bovy_coords.radec_to_lb(ra,dec,degree=True,epoch=None)
    assert numpy.fabs(lb[1]-90.) < 10.**-4., 'Galactic latitude of the NGP given in ra,dec is not 90'
    # Also test this for degree=False
    lb= bovy_coords.radec_to_lb(ra/180.*numpy.pi,dec/180.*numpy.pi,
                                degree=False,epoch=None)
    assert numpy.fabs(lb[1]-numpy.pi/2.) < 10.**-4., 'Galactic latitude of the NGP given in ra,dec is not pi/2'
    _turn_on_apy()
    return None

def test_radec_to_lb_sgp():
    _turn_off_apy()
    # Test that the SGP is at b=90
    ra, dec= 12.25, -27.4
    lb= bovy_coords.radec_to_lb(ra,dec,degree=True,epoch=1950.)
    assert numpy.fabs(lb[1]+90.) < 10.**-8., 'Galactic latitude of the SGP given in ra,dec is not 90'
    # Also test this for degree=False
    lb= bovy_coords.radec_to_lb(ra/180.*numpy.pi,dec/180.*numpy.pi,
                                degree=False,epoch=1950.)
    assert numpy.fabs(lb[1]+numpy.pi/2.) < 10.**-8., 'Galactic latitude of the SGP given in ra,dec is not pi/2'
    _turn_on_apy()
    return None

# Test the longitude of the north celestial pole
def test_radec_to_lb_ncp():
    _turn_off_apy()
    ra, dec= 180., 90.
    lb= bovy_coords.radec_to_lb(ra,dec,degree=True,epoch=1950.)
    assert numpy.fabs(lb[0]-123.) < 10.**-8., 'Galactic longitude of the NCP given in ra,dec is not 123'
    # Also test this for degree=False
    lb= bovy_coords.radec_to_lb(ra/180.*numpy.pi,dec/180.*numpy.pi,
                                degree=False,epoch=1950.)
    assert numpy.fabs(lb[0]-123./180.*numpy.pi) < 10.**-8., 'Galactic longitude of the NCP given in ra,dec is not 123'
    # Also test the latter for vector inputs
    os= numpy.ones(2)
    lb= bovy_coords.radec_to_lb(os*ra/180.*numpy.pi,os*dec/180.*numpy.pi,
                                degree=False,epoch=1950.)
    assert numpy.all(numpy.fabs(lb[:,0]-123./180.*numpy.pi) < 10.**-8.), 'Galactic longitude of the NCP given in ra,dec is not 123'
    _turn_on_apy()
    return None

def test_radec_to_lb_ncp_apyangles():
    _turn_off_apy(keep_loaded=True)
    ra, dec= 180., 90.
    lb= bovy_coords.radec_to_lb(ra,dec,degree=True,epoch='B1950')
    assert numpy.fabs(lb[0]-123.) < 10.**-4., 'Galactic longitude of the NCP given in ra,dec is not 123'
    _turn_on_apy()
    return None

# Test the longitude of the north celestial pole
def test_radec_to_lb_ncp_j2000():
    _turn_off_apy()
    ra, dec= 180., 90.
    lb= bovy_coords.radec_to_lb(ra,dec,degree=True,epoch=2000.)
    assert numpy.fabs(lb[0]-122.9319185680026) < 10.**-8., 'Galactic longitude of the NCP given in ra,dec is not 122.9319185680026'
    # Also test this for degree=False
    lb= bovy_coords.radec_to_lb(ra/180.*numpy.pi,dec/180.*numpy.pi,
                                degree=False,epoch=2000.)
    assert numpy.fabs(lb[0]-122.9319185680026/180.*numpy.pi) < 10.**-8., 'Galactic longitude of the NCP given in ra,dec is not 122.9319185680026'
    # Also test the latter for vector inputs
    os= numpy.ones(2)
    lb= bovy_coords.radec_to_lb(os*ra/180.*numpy.pi,os*dec/180.*numpy.pi,
                                degree=False,epoch=2000.)
    assert numpy.all(numpy.fabs(lb[:,0]-122.9319185680026/180.*numpy.pi) < 10.**-8.), 'Galactic longitude of the NCP given in ra,dec is not 122.9319185680026'
    _turn_on_apy()
    return None

def test_radec_to_lb_ncp_j2000_apyangles():
    _turn_off_apy(keep_loaded=True)
    ra, dec= 180., 90.
    lb= bovy_coords.radec_to_lb(ra,dec,degree=True,epoch='J2000')
    assert numpy.fabs(lb[0]-122.9319185680026) < 10.**-4., 'Galactic longitude of the NCP given in ra,dec is not 122.9319185680026'
    _turn_on_apy()
    return None

# Test that other epochs do not work when not using astropy
def test_radec_to_lb_otherepochs():
    _turn_off_apy()
    ra, dec= 180., 90.
    try:
        lb= bovy_coords.radec_to_lb(ra/180.*numpy.pi,dec/180.*numpy.pi,
                                    degree=False,epoch=1975.)   
    except IOError:
        pass
    else:
        raise AssertionError('radec functions with epoch not equal to 1950 or 2000 did not raise IOError')
    _turn_on_apy()
    return None

# Test that other epochs do work when using astropy
def test_radec_to_lb_otherepochs_apy():
    _turn_off_apy(keep_loaded=True)
    ra, dec= 180., 90.
    try:
        lb= bovy_coords.radec_to_lb(ra/180.*numpy.pi,dec/180.*numpy.pi,
                                    degree=False,epoch='J2015')   
    except IOError:
        raise AssertionError('radec functions with epoch not equal to 1950 or 2000 did not raise IOError')
    else:
        pass
    _turn_on_apy()
    return None

# Test that radec_to_lb and lb_to_radec are each other's inverse
def test_lb_to_radec():
    _turn_off_apy()
    ra, dec= 120, 60.
    lb= bovy_coords.radec_to_lb(ra,dec,degree=True,epoch=2000.)
    rat, dect= bovy_coords.lb_to_radec(lb[0],lb[1],degree=True,epoch=2000.)
    assert numpy.fabs(ra-rat) < 10.**-10., 'lb_to_radec is not the inverse of radec_to_lb'
    assert numpy.fabs(dec-dect) < 10.**-10., 'lb_to_radec is not the inverse of radec_to_lb'
    # Also test this for degree=False
    lb= bovy_coords.radec_to_lb(ra/180.*numpy.pi,dec/180.*numpy.pi,
                                degree=False,epoch=2000.)
    rat, dect= bovy_coords.lb_to_radec(lb[0],lb[1],degree=False,epoch=2000.)
    assert numpy.fabs(ra/180.*numpy.pi-rat) < 10.**-10., 'lb_to_radec is not the inverse of radec_to_lb'
    assert numpy.fabs(dec/180.*numpy.pi-dect) < 10.**-10., 'lb_to_radec is not the inverse of radec_to_lb'
    # And also test this for arrays
    os= numpy.ones(2)
    lb= bovy_coords.radec_to_lb(os*ra/180.*numpy.pi,os*dec/180.*numpy.pi,
                                degree=False,epoch=2000.)
    ratdect= bovy_coords.lb_to_radec(lb[:,0],lb[:,1],degree=False,epoch=2000.)
    rat= ratdect[:,0]
    dect= ratdect[:,1]
    assert numpy.all(numpy.fabs(ra/180.*numpy.pi-rat) < 10.**-10.), 'lb_to_radec is not the inverse of radec_to_lb'
    assert numpy.all(numpy.fabs(dec/180.*numpy.pi-dect) < 10.**-10.), 'lb_to_radec is not the inverse of radec_to_lb'   
    #Also test for a negative l
    l,b= 240., 60.
    ra,dec= bovy_coords.lb_to_radec(l,b,degree=True)
    lt,bt= bovy_coords.radec_to_lb(ra,dec,degree=True)
    assert numpy.fabs(lt-l) < 10.**-10., 'lb_to_radec is not the inverse of radec_to_lb'   
    assert numpy.fabs(bt-b) < 10.**-10., 'lb_to_radec is not the inverse of radec_to_lb'   
    _turn_on_apy()
    return None

# Test that radec_to_lb and lb_to_radec are each other's inverse, using astropy
def test_lb_to_radec_apy():
    ra, dec= 120, 60.
    lb= bovy_coords.radec_to_lb(ra,dec,degree=True,epoch=2000.)
    rat, dect= bovy_coords.lb_to_radec(lb[0],lb[1],degree=True,epoch=2000.)
    assert numpy.fabs(ra-rat) < 10.**-10., 'lb_to_radec is not the inverse of radec_to_lb'
    assert numpy.fabs(dec-dect) < 10.**-10., 'lb_to_radec is not the inverse of radec_to_lb'
    # Also test this for degree=False
    lb= bovy_coords.radec_to_lb(ra/180.*numpy.pi,dec/180.*numpy.pi,
                                degree=False,epoch=2000.)
    rat, dect= bovy_coords.lb_to_radec(lb[0],lb[1],degree=False,epoch=2000.)
    assert numpy.fabs(ra/180.*numpy.pi-rat) < 10.**-10., 'lb_to_radec is not the inverse of radec_to_lb'
    assert numpy.fabs(dec/180.*numpy.pi-dect) < 10.**-10., 'lb_to_radec is not the inverse of radec_to_lb'
    # And also test this for arrays
    os= numpy.ones(2)
    lb= bovy_coords.radec_to_lb(os*ra/180.*numpy.pi,os*dec/180.*numpy.pi,
                                degree=False,epoch=2000.)
    ratdect= bovy_coords.lb_to_radec(lb[:,0],lb[:,1],degree=False,epoch=2000.)
    rat= ratdect[:,0]
    dect= ratdect[:,1]
    assert numpy.all(numpy.fabs(ra/180.*numpy.pi-rat) < 10.**-10.), 'lb_to_radec is not the inverse of radec_to_lb'
    assert numpy.all(numpy.fabs(dec/180.*numpy.pi-dect) < 10.**-10.), 'lb_to_radec is not the inverse of radec_to_lb'   
    #Also test for a negative l
    l,b= 240., 60.
    ra,dec= bovy_coords.lb_to_radec(l,b,degree=True)
    lt,bt= bovy_coords.radec_to_lb(ra,dec,degree=True)
    assert numpy.fabs(lt-l) < 10.**-10., 'lb_to_radec is not the inverse of radec_to_lb'   
    assert numpy.fabs(bt-b) < 10.**-10., 'lb_to_radec is not the inverse of radec_to_lb'   
    return None

# Test that radec_to_lb and lb_to_radec are each other's inverse, using astropy
def test_lb_to_radec_apy_icrs():
    ra, dec= 120, 60.
    lb= bovy_coords.radec_to_lb(ra,dec,degree=True,epoch=None)
    rat, dect= bovy_coords.lb_to_radec(lb[0],lb[1],degree=True,epoch=None)
    assert numpy.fabs(ra-rat) < 10.**-10., 'lb_to_radec is not the inverse of radec_to_lb'
    assert numpy.fabs(dec-dect) < 10.**-10., 'lb_to_radec is not the inverse of radec_to_lb'
    # Also test this for degree=False
    lb= bovy_coords.radec_to_lb(ra/180.*numpy.pi,dec/180.*numpy.pi,
                                degree=False,epoch=None)
    rat, dect= bovy_coords.lb_to_radec(lb[0],lb[1],degree=False,epoch=None)
    assert numpy.fabs(ra/180.*numpy.pi-rat) < 10.**-10., 'lb_to_radec is not the inverse of radec_to_lb'
    assert numpy.fabs(dec/180.*numpy.pi-dect) < 10.**-10., 'lb_to_radec is not the inverse of radec_to_lb'
    # And also test this for arrays
    os= numpy.ones(2)
    lb= bovy_coords.radec_to_lb(os*ra/180.*numpy.pi,os*dec/180.*numpy.pi,
                                degree=False,epoch=None)
    ratdect= bovy_coords.lb_to_radec(lb[:,0],lb[:,1],degree=False,epoch=None)
    rat= ratdect[:,0]
    dect= ratdect[:,1]
    assert numpy.all(numpy.fabs(ra/180.*numpy.pi-rat) < 10.**-10.), 'lb_to_radec is not the inverse of radec_to_lb'
    assert numpy.all(numpy.fabs(dec/180.*numpy.pi-dect) < 10.**-10.), 'lb_to_radec is not the inverse of radec_to_lb'   
    #Also test for a negative l
    l,b= 240., 60.
    ra,dec= bovy_coords.lb_to_radec(l,b,degree=True)
    lt,bt= bovy_coords.radec_to_lb(ra,dec,degree=True)
    assert numpy.fabs(lt-l) < 10.**-10., 'lb_to_radec is not the inverse of radec_to_lb'   
    assert numpy.fabs(bt-b) < 10.**-10., 'lb_to_radec is not the inverse of radec_to_lb'   
    return None

def test_radec_to_lb_galpyvsastropy():
    # Test that galpy's radec_to_lb agrees with astropy's
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    _turn_off_apy(keep_loaded=True)
    ra, dec= 33., -20.
    # using galpy
    lg,bg= bovy_coords.radec_to_lb(ra,dec,degree=True,epoch=2000.0)
    # using astropy
    c= SkyCoord(ra=ra*u.deg,dec=dec*u.deg,frame='fk5',equinox='J2000')
    c= c.transform_to('galactic')
    la,ba= c.l.to(u.deg).value,c.b.to(u.deg).value
    assert numpy.fabs(lg-la) < 1e-12, "radec_to_lb using galpy's own transformations does not agree with astropy's"
    assert numpy.fabs(bg-ba) < 1e-12, "radec_to_lb using galpy's own transformations does not agree with astropy's"
    _turn_on_apy()
    return None

def test_radec_to_lb__1950_galpyvsastropy():
    # Test that galpy's radec_to_lb agrees with astropy's
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    _turn_off_apy(keep_loaded=True)
    ra, dec= 33., -20.
    # using galpy
    lg,bg= bovy_coords.radec_to_lb(ra,dec,degree=True,epoch=1950.0)
    # using astropy
    c= SkyCoord(ra=ra*u.deg,dec=dec*u.deg,frame='fk4noeterms',equinox='B1950')
    c= c.transform_to('galactic')
    la,ba= c.l.to(u.deg).value,c.b.to(u.deg).value
    assert numpy.fabs(lg-la) < 1e-12, "radec_to_lb using galpy's own transformations does not agree with astropy's"
    assert numpy.fabs(bg-ba) < 1e-12, "radec_to_lb using galpy's own transformations does not agree with astropy's"
    _turn_on_apy()
    return None

# Test lb_to_XYZ
def test_lbd_to_XYZ():
    l,b,d= 90., 30.,1.
    XYZ= bovy_coords.lbd_to_XYZ(l,b,d,degree=True)
    assert numpy.fabs(XYZ[0]) <10.**-10., 'lbd_to_XYZ conversion does not work as expected'
    assert numpy.fabs(XYZ[1]-numpy.sqrt(3.)/2.) < 10.**-10., 'lbd_to_XYZ conversion does not work as expected'
    assert numpy.fabs(XYZ[2]-0.5) < 10.**-10., 'lbd_to_XYZ conversion does not work as expected'
    # Also test for degree=False
    XYZ= bovy_coords.lbd_to_XYZ(l/180.*numpy.pi,b/180.*numpy.pi,d,degree=False)
    assert numpy.fabs(XYZ[0]) <10.**-10., 'lbd_to_XYZ conversion does not work as expected'
    assert numpy.fabs(XYZ[1]-numpy.sqrt(3.)/2.) < 10.**-10., 'lbd_to_XYZ conversion does not work as expected'
    assert numpy.fabs(XYZ[2]-0.5) < 10.**-10., 'lbd_to_XYZ conversion does not work as expected'
    # Also test for arrays
    os= numpy.ones(2)
    XYZ= bovy_coords.lbd_to_XYZ(os*l/180.*numpy.pi,os*b/180.*numpy.pi,
                                os*d,degree=False)
    assert numpy.all(numpy.fabs(XYZ[:,0]) <10.**-10.), 'lbd_to_XYZ conversion does not work as expected'
    assert numpy.all(numpy.fabs(XYZ[:,1]-numpy.sqrt(3.)/2.) < 10.**-10.), 'lbd_to_XYZ conversion does not work as expected'
    assert numpy.all(numpy.fabs(XYZ[:,2]-0.5) < 10.**-10.), 'lbd_to_XYZ conversion does not work as expected'
    return None

# Test that XYZ_to_lbd is the inverse of lbd_to_XYZ
def test_XYZ_to_lbd():
    l,b,d= 90., 30.,1.
    XYZ= bovy_coords.lbd_to_XYZ(l,b,d,degree=True)
    lt,bt,dt= bovy_coords.XYZ_to_lbd(XYZ[0],XYZ[1],XYZ[2],degree=True)
    assert numpy.fabs(lt-l) <10.**-10., 'XYZ_to_lbd conversion does not work as expected'
    assert numpy.fabs(bt-b) < 10.**-10., 'XYZ_to_lbd conversion does not work as expected'
    assert numpy.fabs(dt-d) < 10.**-10., 'XYZ_to_lbd conversion does not work as expected'
    # Also test for degree=False
    XYZ= bovy_coords.lbd_to_XYZ(l/180.*numpy.pi,b/180.*numpy.pi,d,degree=False)
    lt,bt,dt= bovy_coords.XYZ_to_lbd(XYZ[0],XYZ[1],XYZ[2],degree=False)
    assert numpy.fabs(lt-l/180.*numpy.pi) <10.**-10., 'XYZ_to_lbd conversion does not work as expected'
    assert numpy.fabs(bt-b/180.*numpy.pi) < 10.**-10., 'XYZ_to_lbd conversion does not work as expected'
    assert numpy.fabs(dt-d) < 10.**-10., 'XYZ_to_lbd conversion does not work as expected'
    # Also test for arrays
    os= numpy.ones(2)
    XYZ= bovy_coords.lbd_to_XYZ(os*l/180.*numpy.pi,os*b/180.*numpy.pi,
                                os*d,degree=False)
    lbdt= bovy_coords.XYZ_to_lbd(XYZ[:,0],XYZ[:,1],XYZ[:,2],degree=False)
    assert numpy.all(numpy.fabs(lbdt[:,0]-l/180.*numpy.pi) <10.**-10.), 'XYZ_to_lbd conversion does not work as expected'
    assert numpy.all(numpy.fabs(lbdt[:,1]-b/180.*numpy.pi) < 10.**-10.), 'XYZ_to_lbd conversion does not work as expected'
    assert numpy.all(numpy.fabs(lbdt[:,2]-d) < 10.**-10.), 'XYZ_to_lbd conversion does not work as expected'
    return None

def test_vrpmllpmbb_to_vxvyvz():
    l,b,d= 90., 0.,1.
    vr,pmll,pmbb= 10.,20./4.740470463496208,-10./4.740470463496208
    vxvyvz= bovy_coords.vrpmllpmbb_to_vxvyvz(vr,pmll,pmbb,l,b,d,
                                             degree=True,XYZ=False)
    assert numpy.fabs(vxvyvz[0]+20.) < 10.**-10., 'vrpmllpmbb_to_vxvyvz conversion did not work as expected'
    assert numpy.fabs(vxvyvz[1]-10.) < 10.**-10., 'vrpmllpmbb_to_vxvyvz conversion did not work as expected'
    assert numpy.fabs(vxvyvz[2]+10.) < 10.**-10., 'vrpmllpmbb_to_vxvyvz conversion did not work as expected'
    vxvyvz= bovy_coords.vrpmllpmbb_to_vxvyvz(vr,pmll,pmbb,l/180.*numpy.pi,
                                             b/180.*numpy.pi,d,
                                             degree=False,XYZ=False)
    assert numpy.fabs(vxvyvz[0]+20.) < 10.**-10., 'vrpmllpmbb_to_vxvyvz conversion did not work as expected'
    assert numpy.fabs(vxvyvz[1]-10.) < 10.**-10., 'vrpmllpmbb_to_vxvyvz conversion did not work as expected'
    assert numpy.fabs(vxvyvz[2]+10.) < 10.**-10., 'vrpmllpmbb_to_vxvyvz conversion did not work as expected'
    vxvyvz= bovy_coords.vrpmllpmbb_to_vxvyvz(vr,pmll,pmbb,0.,1,0.,
                                             XYZ=True)
    assert numpy.fabs(vxvyvz[0]+20.) < 10.**-10., 'vrpmllpmbb_to_vxvyvz conversion did not work as expected'
    assert numpy.fabs(vxvyvz[1]-10.) < 10.**-10., 'vrpmllpmbb_to_vxvyvz conversion did not work as expected'
    assert numpy.fabs(vxvyvz[2]+10.) < 10.**-10., 'vrpmllpmbb_to_vxvyvz conversion did not work as expected'
    vxvyvz= bovy_coords.vrpmllpmbb_to_vxvyvz(vr,pmll,pmbb,0.,1,0.,
                                             XYZ=True,degree=True)
    assert numpy.fabs(vxvyvz[0]+20.) < 10.**-10., 'vrpmllpmbb_to_vxvyvz conversion did not work as expected'
    assert numpy.fabs(vxvyvz[1]-10.) < 10.**-10., 'vrpmllpmbb_to_vxvyvz conversion did not work as expected'
    assert numpy.fabs(vxvyvz[2]+10.) < 10.**-10., 'vrpmllpmbb_to_vxvyvz conversion did not work as expected'
    #Also test for arrays
    os= numpy.ones(2)
    vxvyvz= bovy_coords.vrpmllpmbb_to_vxvyvz(os*vr,os*pmll,os*pmbb,os*l,os*b,
                                             os*d,degree=True,XYZ=False)
    assert numpy.all(numpy.fabs(vxvyvz[:,0]+20.) < 10.**-10.), 'vrpmllpmbb_to_vxvyvz conversion did not work as expected'
    assert numpy.all(numpy.fabs(vxvyvz[:,1]-10.) < 10.**-10.), 'vrpmllpmbb_to_vxvyvz conversion did not work as expected'
    assert numpy.all(numpy.fabs(vxvyvz[:,2]+10.) < 10.**-10.), 'vrpmllpmbb_to_vxvyvz conversion did not work as expected'
    return None

def test_vxvyvz_to_vrpmllpmbb():
    vx,vy,vz= -20.*4.740470463496208,10.,-10.*4.740470463496208
    X,Y,Z= 0.,1.,0.
    vrpmllpmbb= bovy_coords.vxvyvz_to_vrpmllpmbb(vx,vy,vz,X,Y,Z,
                                                 XYZ=True)
    assert numpy.fabs(vrpmllpmbb[0]-10.) < 10.**-10., 'vxvyvz_to_vrpmllpmbb conversion did not work as expected'
    assert numpy.fabs(vrpmllpmbb[1]-20.) < 10.**-10., 'vxvyvz_to_vrpmllpmbb conversion did not work as expected'
    assert numpy.fabs(vrpmllpmbb[2]+10.) < 10.**-10., 'vxvyvz_to_vrpmllpmbb conversion did not work as expected'
    # also try with degree=True (that shouldn't fail!)
    vrpmllpmbb= bovy_coords.vxvyvz_to_vrpmllpmbb(vx,vy,vz,X,Y,Z,
                                                 XYZ=True,
                                                 degree=True)
    assert numpy.fabs(vrpmllpmbb[0]-10.) < 10.**-10., 'vxvyvz_to_vrpmllpmbb conversion did not work as expected'
    assert numpy.fabs(vrpmllpmbb[1]-20.) < 10.**-10., 'vxvyvz_to_vrpmllpmbb conversion did not work as expected'
    assert numpy.fabs(vrpmllpmbb[2]+10.) < 10.**-10., 'vxvyvz_to_vrpmllpmbb conversion did not work as expected'
    # also for lbd
    vrpmllpmbb= bovy_coords.vxvyvz_to_vrpmllpmbb(vx,vy,vz,90.,0.,1.,
                                                 XYZ=False,degree=True)
    assert numpy.fabs(vrpmllpmbb[0]-10.) < 10.**-10., 'vxvyvz_to_vrpmllpmbb conversion did not work as expected'
    assert numpy.fabs(vrpmllpmbb[1]-20.) < 10.**-10., 'vxvyvz_to_vrpmllpmbb conversion did not work as expected'
    assert numpy.fabs(vrpmllpmbb[2]+10.) < 10.**-10., 'vxvyvz_to_vrpmllpmbb conversion did not work as expected'
    # also for lbd, not in degree
    vrpmllpmbb= bovy_coords.vxvyvz_to_vrpmllpmbb(vx,vy,vz,numpy.pi/2.,0.,1.,
                                                 XYZ=False,degree=False)
    assert numpy.fabs(vrpmllpmbb[0]-10.) < 10.**-10., 'vxvyvz_to_vrpmllpmbb conversion did not work as expected'
    assert numpy.fabs(vrpmllpmbb[1]-20.) < 10.**-10., 'vxvyvz_to_vrpmllpmbb conversion did not work as expected'
    assert numpy.fabs(vrpmllpmbb[2]+10.) < 10.**-10., 'vxvyvz_to_vrpmllpmbb conversion did not work as expected'
    # and for arrays
    os= numpy.ones(2)
    vrpmllpmbb= bovy_coords.vxvyvz_to_vrpmllpmbb(os*vx,os*vy,os*vz,
                                                 os*numpy.pi/2.,os*0.,os,
                                                 XYZ=False,degree=False)
    assert numpy.all(numpy.fabs(vrpmllpmbb[:,0]-10.) < 10.**-10.), 'vxvyvz_to_vrpmllpmbb conversion did not work as expected'
    assert numpy.all(numpy.fabs(vrpmllpmbb[:,1]-20.) < 10.**-10.), 'vxvyvz_to_vrpmllpmbb conversion did not work as expected'
    assert numpy.all(numpy.fabs(vrpmllpmbb[:,2]+10.) < 10.**-10.), 'vxvyvz_to_vrpmllpmbb conversion did not work as expected'
    return None

def test_XYZ_to_galcenrect():
    X,Y,Z= 1.,3.,-2.
    gcXYZ= bovy_coords.XYZ_to_galcenrect(X,Y,Z,Xsun=1.,Zsun=0.)
    assert numpy.fabs(gcXYZ[0]) < 10.**-5., 'XYZ_to_galcenrect conversion did not work as expected'
    assert numpy.fabs(gcXYZ[1]-3.) < 10.**-5., 'XYZ_to_galcenrect conversion did not work as expected'
    assert numpy.fabs(gcXYZ[2]+2.) < 10.**-5., 'XYZ_to_galcenrect conversion did not work as expected'
    #Another test
    X,Y,Z= -1.,3.,-2.
    gcXYZ= bovy_coords.XYZ_to_galcenrect(X,Y,Z,Xsun=1.,Zsun=0.)
    assert numpy.fabs(gcXYZ[0]-2.) < 10.**-5., 'XYZ_to_galcenrect conversion did not work as expected'
    assert numpy.fabs(gcXYZ[1]-3.) < 10.**-5., 'XYZ_to_galcenrect conversion did not work as expected'
    assert numpy.fabs(gcXYZ[2]+2.) < 10.**-5., 'XYZ_to_galcenrect conversion did not work as expected'
    return None

def test_XYZ_to_galcenrect_negXsun():
    # Check that XYZ_to_galcenrect works for negative Xsun
    X,Y,Z= 0.3,2.1,-1.2
    gcXYZ= bovy_coords.XYZ_to_galcenrect(X,Y,Z,Xsun=1.2,Zsun=0.2)
    gcXYZn= bovy_coords.XYZ_to_galcenrect(X,Y,Z,Xsun=-1.2,Zsun=0.2)
    assert numpy.fabs(gcXYZ[0]+gcXYZn[0]) < 10.**-10., 'XYZ_to_galcenrect conversion did not work as expected for negative Xsun'
    assert numpy.fabs(gcXYZ[1]-gcXYZn[1]) < 10.**-10., 'XYZ_to_galcenrect conversion did not work as expected for negative Xsun'
    assert numpy.fabs(gcXYZ[2]-gcXYZn[2]) < 10.**-10., 'XYZ_to_galcenrect conversion did not work as expected for negative Xsun'

def test_lbd_to_galcenrect_galpyvsastropy():
    # Test that galpy's transformations agree with astropy's
    from astropy.coordinates import SkyCoord, Galactocentric
    import astropy.units as u
    _turn_off_apy()
    l,b,d= 32., -12., 3.
    Zsun= 0.025
    # Using galpy
    X,Y,Z= bovy_coords.lbd_to_XYZ(l,b,d,degree=True)
    gcXYZ= bovy_coords.XYZ_to_galcenrect(X,Y,Z,Xsun=8.,Zsun=Zsun)
    # Using astropy
    c= SkyCoord(l=l*u.deg,b=b*u.deg,distance=d*u.kpc,frame='galactic')
    gc_frame= Galactocentric(galcen_distance=numpy.sqrt(8.**2.+Zsun**2.)*u.kpc,
                             z_sun=Zsun*u.kpc)
    c= c.transform_to(gc_frame)
    # galpy is left-handed, astropy right-handed
    assert numpy.fabs(gcXYZ[0]+c.x.to(u.kpc).value) < 10.**-10., "lbd to galcenrect conversion using galpy's methods does not agree with astropy"
    assert numpy.fabs(gcXYZ[1]-c.y.to(u.kpc).value) < 10.**-10., "lbd to galcenrect conversion using galpy's methods does not agree with astropy"
    assert numpy.fabs(gcXYZ[2]-c.z.to(u.kpc).value) < 10.**-10., "lbd to galcenrect conversion using galpy's methods does not agree with astropy"
    # Also with negative Xsun
    l,b,d= 32., -12., 3.
    Zsun= 0.025
    # Using galpy
    X,Y,Z= bovy_coords.lbd_to_XYZ(l,b,d,degree=True)
    gcXYZ= bovy_coords.XYZ_to_galcenrect(X,Y,Z,Xsun=-8.,Zsun=Zsun)
    # Using astropy
    c= SkyCoord(l=l*u.deg,b=b*u.deg,distance=d*u.kpc,frame='galactic')
    gc_frame= Galactocentric(galcen_distance=numpy.sqrt(8.**2.+Zsun**2.)*u.kpc,
                             z_sun=Zsun*u.kpc)
    c= c.transform_to(gc_frame)
    # galpy is now right-handed, astropy right-handed
    assert numpy.fabs(gcXYZ[0]-c.x.to(u.kpc).value) < 10.**-10., "lbd to galcenrect conversion using galpy's methods does not agree with astropy"
    assert numpy.fabs(gcXYZ[1]-c.y.to(u.kpc).value) < 10.**-10., "lbd to galcenrect conversion using galpy's methods does not agree with astropy"
    assert numpy.fabs(gcXYZ[2]-c.z.to(u.kpc).value) < 10.**-10., "lbd to galcenrect conversion using galpy's methods does not agree with astropy"
    _turn_on_apy()
    return None

def test_lbd_to_galcencyl_galpyvsastropy():
    # Test that galpy's transformations agree with astropy's
    from astropy.coordinates import SkyCoord, Galactocentric
    import astropy.units as u
    _turn_off_apy()
    l,b,d= 32., -12., 3.
    Zsun= 0.025
    # Using galpy
    X,Y,Z= bovy_coords.lbd_to_XYZ(l,b,d,degree=True)
    gcRpZ= bovy_coords.XYZ_to_galcencyl(X,Y,Z,Xsun=8.,Zsun=Zsun)
    # Using astropy
    c= SkyCoord(l=l*u.deg,b=b*u.deg,distance=d*u.kpc,frame='galactic')
    gc_frame= Galactocentric(galcen_distance=numpy.sqrt(8.**2.+Zsun**2.)*u.kpc,
                             z_sun=Zsun*u.kpc)
    c= c.transform_to(gc_frame)
    c.representation= 'cylindrical'
    # galpy is left-handed, astropy right-handed
    assert numpy.fabs(gcRpZ[0]-c.rho.to(u.kpc).value) < 10.**-10., "lbd to galcencyl conversion using galpy's methods does not agree with astropy"
    assert numpy.fabs(gcRpZ[1]-numpy.pi+c.phi.to(u.rad).value) < 10.**-10., "lbd to galcencyl conversion using galpy's methods does not agree with astropy"
    assert numpy.fabs(gcRpZ[2]-c.z.to(u.kpc).value) < 10.**-10., "lbd to galcencyl conversion using galpy's methods does not agree with astropy"
    # Also with negative Xsun
    l,b,d= 32., -12., 3.
    Zsun= 0.025
    # Using galpy
    X,Y,Z= bovy_coords.lbd_to_XYZ(l,b,d,degree=True)
    gcRpZ= bovy_coords.XYZ_to_galcencyl(X,Y,Z,Xsun=-8.,Zsun=Zsun)
    # Using astropy
    c= SkyCoord(l=l*u.deg,b=b*u.deg,distance=d*u.kpc,frame='galactic')
    gc_frame= Galactocentric(galcen_distance=numpy.sqrt(8.**2.+Zsun**2.)*u.kpc,
                             z_sun=Zsun*u.kpc)
    c= c.transform_to(gc_frame)
    c.representation= 'cylindrical'
    # galpy is now right-handed, astropy right-handed
    assert numpy.fabs(gcRpZ[0]-c.rho.to(u.kpc).value) < 10.**-10., "lbd to galcencyl conversion using galpy's methods does not agree with astropy"
    assert numpy.fabs(gcRpZ[1]-c.phi.to(u.rad).value) < 10.**-10., "lbd to galcencyl conversion using galpy's methods does not agree with astropy"
    assert numpy.fabs(gcRpZ[2]-c.z.to(u.kpc).value) < 10.**-10., "lbd to galcencyl conversion using galpy's methods does not agree with astropy"
    _turn_on_apy()
    return None

def test_galcenrect_to_XYZ_negXsun():
    gcX, gcY, gcZ= -1.,4.,2.
    XYZ= numpy.array(bovy_coords.galcenrect_to_XYZ(gcX,gcY,gcZ,Xsun=1.,Zsun=0.2))
    XYZn= numpy.array(bovy_coords.galcenrect_to_XYZ(-gcX,gcY,gcZ,Xsun=-1.,Zsun=0.2))
    assert numpy.all(numpy.fabs(XYZ-XYZn) < 10.**-10.), 'galcenrect_to_XYZ conversion did not work as expected for negative Xsun'
    return None

def test_galcenrect_to_XYZ():
    gcX, gcY, gcZ= -1.,4.,2.
    XYZ= bovy_coords.galcenrect_to_XYZ(gcX,gcY,gcZ,Xsun=1.,Zsun=0.)
    assert numpy.fabs(XYZ[0]-2.) < 10.**-10., 'galcenrect_to_XYZ conversion did not work as expected'
    assert numpy.fabs(XYZ[1]-4.) < 10.**-10., 'galcenrect_to_XYZ conversion did not work as expected'
    assert numpy.fabs(XYZ[2]-2.) < 10.**-10., 'galcenrect_to_XYZ conversion did not work as expected'
    # Also for arrays
    s= numpy.arange(2)+1
    XYZ= bovy_coords.galcenrect_to_XYZ(gcX*s,gcY*s,gcZ*s,Xsun=1.,Zsun=0.)
    assert numpy.fabs(XYZ[0,0]-2.) < 10.**-10., 'galcenrect_to_XYZ conversion did not work as expected'
    assert numpy.fabs(XYZ[0,1]-4.) < 10.**-10., 'galcenrect_to_XYZ conversion did not work as expected'
    assert numpy.fabs(XYZ[0,2]-2.) < 10.**-10., 'galcenrect_to_XYZ conversion did not work as expected'
    # Check 2nd one
    assert numpy.fabs(XYZ[1,0]-3.) < 10.**-10., 'galcenrect_to_XYZ conversion did not work as expected'
    assert numpy.fabs(XYZ[1,1]-8.) < 10.**-10., 'galcenrect_to_XYZ conversion did not work as expected'
    assert numpy.fabs(XYZ[1,2]-4.) < 10.**-10., 'galcenrect_to_XYZ conversion did not work as expected'
    # Also for arrays with Xsun/Zsun also arrays
    s= numpy.arange(2)+1
    XYZ= bovy_coords.galcenrect_to_XYZ(gcX*s,gcY*s,gcZ*s,Xsun=1.*s,Zsun=0.*s)
    assert numpy.fabs(XYZ[0,0]-2.) < 10.**-10., 'galcenrect_to_XYZ conversion did not work as expected'
    assert numpy.fabs(XYZ[0,1]-4.) < 10.**-10., 'galcenrect_to_XYZ conversion did not work as expected'
    assert numpy.fabs(XYZ[0,2]-2.) < 10.**-10., 'galcenrect_to_XYZ conversion did not work as expected'
    # Check 2nd one
    assert numpy.fabs(XYZ[1,0]-4.) < 10.**-10., 'galcenrect_to_XYZ conversion did not work as expected'
    assert numpy.fabs(XYZ[1,1]-8.) < 10.**-10., 'galcenrect_to_XYZ conversion did not work as expected'
    assert numpy.fabs(XYZ[1,2]-4.) < 10.**-10., 'galcenrect_to_XYZ conversion did not work as expected'
    return None

def test_XYZ_to_galcencyl():
    X,Y,Z= 5.,4.,-2.
    gcRpZ= bovy_coords.XYZ_to_galcencyl(X,Y,Z,Xsun=8.,Zsun=0.)
    assert numpy.fabs(gcRpZ[0]-5.) < 10.**-5., 'XYZ_to_galcencyl conversion did not work as expected'
    assert numpy.fabs(gcRpZ[1]-numpy.arctan(4./3.)) < 10.**-5., 'XYZ_to_galcencyl conversion did not work as expected'
    assert numpy.fabs(gcRpZ[2]+2.) < 10.**-4.8, 'XYZ_to_galcencyl conversion did not work as expected'
    #Another X
    X,Y,Z= 11.,4.,-2.
    gcRpZ= bovy_coords.XYZ_to_galcencyl(X,Y,Z,Xsun=8.,Zsun=0.)
    assert numpy.fabs(gcRpZ[0]-5.) < 10.**-5., 'XYZ_to_galcencyl conversion did not work as expected'
    assert numpy.fabs(gcRpZ[1]-numpy.pi+numpy.arctan(4./3.)) < 10.**-5., 'XYZ_to_galcencyl conversion did not work as expected'
    assert numpy.fabs(gcRpZ[2]+2.) < 10.**-4.6, 'XYZ_to_galcencyl conversion did not work as expected'
    return None

def test_galcencyl_to_XYZ():
    gcR, gcp, gcZ= 5.,numpy.arctan(4./3.),2.
    XYZ= bovy_coords.galcencyl_to_XYZ(gcR,gcp,gcZ,Xsun=8.,Zsun=0.)
    assert numpy.fabs(XYZ[0]-5.) < 10.**-10., 'galcencyl_to_XYZ conversion did not work as expected'
    assert numpy.fabs(XYZ[1]-4.) < 10.**-10., 'galcencyl_to_XYZ conversion did not work as expected'
    assert numpy.fabs(XYZ[2]-2.) < 10.**-10., 'galcencyl_to_XYZ conversion did not work as expected'
    # Also for arrays
    s= numpy.arange(2)+1
    XYZ= bovy_coords.galcencyl_to_XYZ(gcR*s,gcp*s,gcZ*s,Xsun=8.,Zsun=0.)
    assert numpy.fabs(XYZ[0,0]-5.) < 10.**-10., 'galcencyl_to_XYZ conversion did not work as expected'
    assert numpy.fabs(XYZ[0,1]-4.) < 10.**-10., 'galcencyl_to_XYZ conversion did not work as expected'
    assert numpy.fabs(XYZ[0,2]-2.) < 10.**-10., 'galcencyl_to_XYZ conversion did not work as expected'
    # Also test the second one
    assert numpy.fabs(XYZ[1,0]-10.8) < 10.**-10., 'galcencyl_to_XYZ conversion did not work as expected'
    assert numpy.fabs(XYZ[1,1]-9.6) < 10.**-10., 'galcencyl_to_XYZ conversion did not work as expected'
    assert numpy.fabs(XYZ[1,2]-4.0) < 10.**-10., 'galcencyl_to_XYZ conversion did not work as expected'
    # Also for arrays where Xsun/Zsun are also arrays
    s= numpy.arange(2)+1
    XYZ= bovy_coords.galcencyl_to_XYZ(gcR*s,gcp*s,gcZ*s,Xsun=8.*s,Zsun=0.*s)
    assert numpy.fabs(XYZ[0,0]-5.) < 10.**-10., 'galcencyl_to_XYZ conversion did not work as expected'
    assert numpy.fabs(XYZ[0,1]-4.) < 10.**-10., 'galcencyl_to_XYZ conversion did not work as expected'
    assert numpy.fabs(XYZ[0,2]-2.) < 10.**-10., 'galcencyl_to_XYZ conversion did not work as expected'
    # Also test the second one
    assert numpy.fabs(XYZ[1,0]-18.8) < 10.**-10., 'galcencyl_to_XYZ conversion did not work as expected'
    assert numpy.fabs(XYZ[1,1]-9.6) < 10.**-10., 'galcencyl_to_XYZ conversion did not work as expected'
    assert numpy.fabs(XYZ[1,2]-4.0) < 10.**-10., 'galcencyl_to_XYZ conversion did not work as expected'
    return None

def test_vxvyvz_to_galcenrect():
    vx,vy,vz= 10.,-20.,30
    vgc= bovy_coords.vxvyvz_to_galcenrect(vx,vy,vz,vsun=[-5.,10.,5.])
    assert numpy.fabs(vgc[0]+15.) < 10.**-4., 'vxvyvz_to_galcenrect conversion did not work as expected'
    assert numpy.fabs(vgc[1]+10.) < 10.**-4., 'vxvyvz_to_galcenrect conversion did not work as expected'
    assert numpy.fabs(vgc[2]-35.) < 10.**-4., 'vxvyvz_to_galcenrect conversion did not work as expected'
    return None

def test_vxvyvz_to_galcenrect_negXsun():
    vx,vy,vz= 10.,-20.,30
    vgc= bovy_coords.vxvyvz_to_galcenrect(vx,vy,vz,vsun=[-5.,10.,5.],
                                          Xsun=1.1,Zsun=0.2)
    vgcn= bovy_coords.vxvyvz_to_galcenrect(vx,vy,vz,vsun=[5.,10.,5.],
                                           Xsun=-1.1,Zsun=0.2)
    assert numpy.fabs(vgc[0]+vgcn[0]) < 10.**-4., 'vxvyvz_to_galcenrect conversion did not work as expected for negative Xsun'
    assert numpy.fabs(vgc[1]-vgcn[1]) < 10.**-4., 'vxvyvz_to_galcenrect conversion did not work as expected for negative Xsun'
    assert numpy.fabs(vgc[2]-vgcn[2]) < 10.**-4., 'vxvyvz_to_galcenrect conversion did not work as expected for negative Xsun'
    return None

def test_vrpmllpmbb_to_galcenrect_galpyvsastropy():
    # Test that galpy's transformations agree with astropy's
    from astropy.coordinates import SkyCoord, Galactocentric, \
        CartesianDifferential
    import astropy.units as u
    _turn_off_apy()
    l,b,d= 32., -12., 3.
    vr,pmll,pmbb= -112., -13.,5.
    Zsun= 0.025
    Rsun= 8.
    vsun= [-10.,230.,7.]
    # Using galpy
    vx,vy,vz= bovy_coords.vrpmllpmbb_to_vxvyvz(vr,pmll,pmbb,l,b,d,degree=True)
    vXYZg= bovy_coords.vxvyvz_to_galcenrect(vx,vy,vz,vsun=vsun,Xsun=Rsun,
                                            Zsun=Zsun)
    # Using astropy
    c= SkyCoord(l=l*u.deg,b=b*u.deg,distance=d*u.kpc,
                radial_velocity=vr*u.km/u.s,pm_l_cosb=pmll*u.mas/u.yr,
                pm_b=pmbb*u.mas/u.yr,frame='galactic')
    gc_frame= Galactocentric(\
        galcen_distance=numpy.sqrt(Rsun**2.+Zsun**2.)*u.kpc,z_sun=Zsun*u.kpc,
        galcen_v_sun=CartesianDifferential(numpy.array([-vsun[0],vsun[1],vsun[2]])*u.km/u.s))
    c= c.transform_to(gc_frame)
    c.representation= 'cartesian'
    # galpy is left-handed, astropy right-handed
    assert numpy.fabs(vXYZg[0]+c.v_x.to(u.km/u.s).value) < 10.**-8., "vrpmllpmbblbd to galcenrect conversion using galpy's methods does not agree with astropy"
    assert numpy.fabs(vXYZg[1]-c.v_y.to(u.km/u.s).value) < 10.**-8., "vrpmllpmbb to galcenrect conversion using galpy's methods does not agree with astropy"
    assert numpy.fabs(vXYZg[2]-c.v_z.to(u.km/u.s).value) < 10.**-8., "vrpmllpmbb to galcenrect conversion using galpy's methods does not agree with astropy"
    # Also with negative Xsun
    l,b,d= 32., -12., 3.
    Zsun= 0.025
    Rsun= -8.
    vsun= numpy.array([-10.,230.,7.])
    # Using galpy
    vx,vy,vz= bovy_coords.vrpmllpmbb_to_vxvyvz(vr,pmll,pmbb,l,b,d,degree=True)
    vXYZg= bovy_coords.vxvyvz_to_galcenrect(vx,vy,vz,vsun=vsun,Xsun=Rsun,
                                            Zsun=Zsun)
    # Using astropy
    c= SkyCoord(l=l*u.deg,b=b*u.deg,distance=d*u.kpc,
                radial_velocity=vr*u.km/u.s,pm_l_cosb=pmll*u.mas/u.yr,
                pm_b=pmbb*u.mas/u.yr,frame='galactic')
    gc_frame= Galactocentric(\
        galcen_distance=numpy.sqrt(Rsun**2.+Zsun**2.)*u.kpc,z_sun=Zsun*u.kpc,
        galcen_v_sun=CartesianDifferential(numpy.array([vsun[0],vsun[1],vsun[2]])*u.km/u.s))
    c= c.transform_to(gc_frame)
    c.representation= 'cartesian'
    # galpy is now right-handed, astropy right-handed
    assert numpy.fabs(vXYZg[0]-c.v_x.to(u.km/u.s).value) < 10.**-8., "vrpmllpmbblbd to galcenrect conversion using galpy's methods does not agree with astropy"
    assert numpy.fabs(vXYZg[1]-c.v_y.to(u.km/u.s).value) < 10.**-8., "vrpmllpmbb to galcenrect conversion using galpy's methods does not agree with astropy"
    assert numpy.fabs(vXYZg[2]-c.v_z.to(u.km/u.s).value) < 10.**-8., "vrpmllpmbb to galcenrect conversion using galpy's methods does not agree with astropy"
    _turn_on_apy()
    return None

def test_vxvyvz_to_galcencyl():
    X,Y,Z= 3.,4.,2.
    vx,vy,vz= 10.,-20.,30
    vgc= bovy_coords.vxvyvz_to_galcencyl(vx,vy,vz,X,Y,Z,vsun=[-5.,10.,5.])
    assert numpy.fabs(vgc[0]+17.) < 10.**-4., 'vxvyvz_to_galcenrect conversion did not work as expected'
    assert numpy.fabs(vgc[1]-6.) < 10.**-4., 'vxvyvz_to_galcenrect conversion did not work as expected'
    assert numpy.fabs(vgc[2]-35.) < 10.**-4., 'vxvyvz_to_galcenrect conversion did not work as expected'
    #with galcen=True
    vgc= bovy_coords.vxvyvz_to_galcencyl(vx,vy,vz,5.,numpy.arctan(4./3.),Z,
                                         vsun=[-5.,10.,5.],galcen=True)
    assert numpy.fabs(vgc[0]+17.) < 10.**-4., 'vxvyvz_to_galcenrect conversion did not work as expected'
    assert numpy.fabs(vgc[1]-6.) < 10.**-4., 'vxvyvz_to_galcenrect conversion did not work as expected'
    assert numpy.fabs(vgc[2]-35.) < 10.**-4., 'vxvyvz_to_galcenrect conversion did not work as expected'
    return None
    
def test_vrpmllpmbb_to_galcencyl_galpyvsastropy():
    # Test that galpy's transformations agree with astropy's
    from astropy.coordinates import SkyCoord, Galactocentric, \
        CartesianDifferential
    import astropy.units as u
    _turn_off_apy()
    l,b,d= 32., -12., 3.
    vr,pmll,pmbb= -112., -13.,5.
    Zsun= 0.025
    Rsun= 8.
    vsun= [-10.,230.,7.]
    # Using galpy
    X,Y,Z= bovy_coords.lbd_to_XYZ(l,b,d,degree=True)
    gcXYZ= bovy_coords.XYZ_to_galcenrect(X,Y,Z,Xsun=Rsun,Zsun=Zsun)
    vx,vy,vz= bovy_coords.vrpmllpmbb_to_vxvyvz(vr,pmll,pmbb,l,b,d,degree=True)
    vRTZg= bovy_coords.vxvyvz_to_galcencyl(vx,vy,vz,gcXYZ[0],gcXYZ[1],gcXYZ[2],
                                           vsun=vsun,Xsun=Rsun,Zsun=Zsun)
    # Using astropy
    c= SkyCoord(l=l*u.deg,b=b*u.deg,distance=d*u.kpc,
                radial_velocity=vr*u.km/u.s,pm_l_cosb=pmll*u.mas/u.yr,
                pm_b=pmbb*u.mas/u.yr,frame='galactic')
    gc_frame= Galactocentric(\
        galcen_distance=numpy.sqrt(Rsun**2.+Zsun**2.)*u.kpc,z_sun=Zsun*u.kpc,
        galcen_v_sun=CartesianDifferential(numpy.array([-vsun[0],vsun[1],vsun[2]])*u.km/u.s))
    c= c.transform_to(gc_frame)
    c.representation= 'cylindrical'
    # galpy is left-handed, astropy right-handed
    assert numpy.fabs(vRTZg[0]-c.d_rho.to(u.km/u.s).value) < 10.**-8., "vrpmllpmbblbd to galcencyl conversion using galpy's methods does not agree with astropy"
    assert numpy.fabs(vRTZg[1]+(c.d_phi*c.rho).to(u.km/u.s,
        equivalencies=u.dimensionless_angles()).value) < 10.**-8., "vrpmllpmbb to galcencyl conversion using galpy's methods does not agree with astropy"
    assert numpy.fabs(vRTZg[2]-c.d_z.to(u.km/u.s).value) < 10.**-8., "vrpmllpmbb to galcencyl conversion using galpy's methods does not agree with astropy"
    # Also with negative Xsun
    l,b,d= 32., -12., 3.
    Zsun= 0.025
    Rsun= -8.
    vsun= numpy.array([-10.,230.,7.])
    # Using galpy
    X,Y,Z= bovy_coords.lbd_to_XYZ(l,b,d,degree=True)
    gcXYZ= bovy_coords.XYZ_to_galcenrect(X,Y,Z,Xsun=Rsun,Zsun=Zsun)
    vx,vy,vz= bovy_coords.vrpmllpmbb_to_vxvyvz(vr,pmll,pmbb,l,b,d,degree=True)
    vRTZg= bovy_coords.vxvyvz_to_galcencyl(vx,vy,vz,gcXYZ[0],gcXYZ[1],gcXYZ[2],
                                           vsun=vsun,Xsun=Rsun,Zsun=Zsun)
    # Using astropy
    c= SkyCoord(l=l*u.deg,b=b*u.deg,distance=d*u.kpc,
                radial_velocity=vr*u.km/u.s,pm_l_cosb=pmll*u.mas/u.yr,
                pm_b=pmbb*u.mas/u.yr,frame='galactic')
    gc_frame= Galactocentric(\
        galcen_distance=numpy.sqrt(Rsun**2.+Zsun**2.)*u.kpc,z_sun=Zsun*u.kpc,
        galcen_v_sun=CartesianDifferential(numpy.array([vsun[0],vsun[1],vsun[2]])*u.km/u.s))
    c= c.transform_to(gc_frame)
    c.representation= 'cylindrical'
    # galpy is left-handed, astropy right-handed
    assert numpy.fabs(vRTZg[0]-c.d_rho.to(u.km/u.s).value) < 10.**-8., "vrpmllpmbblbd to galcencyl conversion using galpy's methods does not agree with astropy"
    assert numpy.fabs(vRTZg[1]-(c.d_phi*c.rho).to(u.km/u.s,
        equivalencies=u.dimensionless_angles()).value) < 10.**-8., "vrpmllpmbb to galcencyl conversion using galpy's methods does not agree with astropy"
    assert numpy.fabs(vRTZg[2]-c.d_z.to(u.km/u.s).value) < 10.**-8., "vrpmllpmbb to galcencyl conversion using galpy's methods does not agree with astropy"
    _turn_on_apy()
    return None

def test_galcenrect_to_vxvyvz():
    vxg,vyg,vzg= -15.,-10.,35.
    vxyz= bovy_coords.galcenrect_to_vxvyvz(vxg,vyg,vzg,vsun=[-5.,10.,5.])
    assert numpy.fabs(vxyz[0]-10.) < 10.**-10., 'galcenrect_to_vxvyvz conversion did not work as expected'
    assert numpy.fabs(vxyz[1]+20.) < 10.**-10., 'galcenrect_to_vxvyvz conversion did not work as expected'
    assert numpy.fabs(vxyz[2]-30.) < 10.**-10., 'galcenrect_to_vxvyvz conversion did not work as expected'
    #Also for arrays
    os= numpy.ones(2)
    vxyz= bovy_coords.galcenrect_to_vxvyvz(os*vxg,os*vyg,os*vzg,
                                           vsun=[-5.,10.,5.])
    assert numpy.all(numpy.fabs(vxyz[:,0]-10.) < 10.**-10.), 'galcenrect_to_vxvyvz conversion did not work as expected'
    assert numpy.all(numpy.fabs(vxyz[:,1]+20.) < 10.**-10.), 'galcenrect_to_vxvyvz conversion did not work as expected'
    assert numpy.all(numpy.fabs(vxyz[:,2]-30.) < 10.**-10.), 'galcenrect_to_vxvyvz conversion did not work as expected'
    return None

def test_galcenrect_to_vxvyvz_negXsun():
    vxg,vyg,vzg= -15.,-10.,35.
    vxyz= bovy_coords.galcenrect_to_vxvyvz(vxg,vyg,vzg,vsun=[-5.,10.,5.],
                                           Xsun=1.1,Zsun=0.2)
    vxyzn= bovy_coords.galcenrect_to_vxvyvz(-vxg,vyg,vzg,vsun=[5.,10.,5.],
                                             Xsun=-1.1,Zsun=0.2)
    assert numpy.all(numpy.fabs(numpy.array(vxyz)-numpy.array(vxyzn)) < 10.**-10.), 'galcenrect_to_vxvyvz conversion did not work as expected'
    return None

def test_galcencyl_to_vxvyvz():
    vr,vp,vz= -17.,6.,35.
    phi= numpy.arctan(4./3.)
    vxyz= bovy_coords.galcencyl_to_vxvyvz(vr,vp,vz,phi,vsun=[-5.,10.,5.])
    assert numpy.fabs(vxyz[0]-10.) < 10.**-10., 'galcenrect_to_vxvyvz conversion did not work as expected'
    assert numpy.fabs(vxyz[1]+20.) < 10.**-10., 'galcenrect_to_vxvyvz conversion did not work as expected'
    assert numpy.fabs(vxyz[2]-30.) < 10.**-10., 'galcenrect_to_vxvyvz conversion did not work as expected'
    return None   

def test_sphergal_to_rectgal():
    l,b,d= 90.,0.,1.
    vr,pmll,pmbb= 10.,-20./4.740470463496208,30./4.740470463496208
    X,Y,Z,vx,vy,vz= bovy_coords.sphergal_to_rectgal(l,b,d,vr,pmll,pmbb,
                                                    degree=True)
    assert numpy.fabs(X-0.) < 10.**-10., 'sphergal_to_rectgal conversion did not work as expected'
    assert numpy.fabs(Y-1.) < 10.**-10., 'sphergal_to_rectgal conversion did not work as expected'
    assert numpy.fabs(Z-0.) < 10.**-10., 'sphergal_to_rectgal conversion did not work as expected'
    assert numpy.fabs(vx-20.) < 10.**-10., 'sphergal_to_rectgal conversion did not work as expected'
    assert numpy.fabs(vy-10.) < 10.**-10., 'sphergal_to_rectgal conversion did not work as expected'
    assert numpy.fabs(vz-30.) < 10.**-10., 'sphergal_to_rectgal conversion did not work as expected'
    #Also test for degree=False
    X,Y,Z,vx,vy,vz= bovy_coords.sphergal_to_rectgal(l/180.*numpy.pi,
                                                    b/180.*numpy.pi,
                                                    d,vr,pmll,pmbb,
                                                    degree=False)
    assert numpy.fabs(X-0.) < 10.**-10., 'sphergal_to_rectgal conversion did not work as expected'
    assert numpy.fabs(Y-1.) < 10.**-10., 'sphergal_to_rectgal conversion did not work as expected'
    assert numpy.fabs(Z-0.) < 10.**-10., 'sphergal_to_rectgal conversion did not work as expected'
    assert numpy.fabs(vx-20.) < 10.**-10., 'sphergal_to_rectgal conversion did not work as expected'
    assert numpy.fabs(vy-10.) < 10.**-10., 'sphergal_to_rectgal conversion did not work as expected'
    assert numpy.fabs(vz-30.) < 10.**-10., 'sphergal_to_rectgal conversion did not work as expected'
    #Also test for arrays
    os= numpy.ones(2)
    XYZvxvyvz= bovy_coords.sphergal_to_rectgal(os*l,os*b,os*d,
                                                    os*vr,os*pmll,os*pmbb,
                                                    degree=True)
    X= XYZvxvyvz[:,0]
    Y= XYZvxvyvz[:,1]
    Z= XYZvxvyvz[:,2]
    vx= XYZvxvyvz[:,3]
    vy= XYZvxvyvz[:,4]
    vz= XYZvxvyvz[:,5]
    assert numpy.all(numpy.fabs(X-0.) < 10.**-10.), 'sphergal_to_rectgal conversion did not work as expected'
    assert numpy.all(numpy.fabs(Y-1.) < 10.**-10.), 'sphergal_to_rectgal conversion did not work as expected'
    assert numpy.all(numpy.fabs(Z-0.) < 10.**-10.), 'sphergal_to_rectgal conversion did not work as expected'
    assert numpy.all(numpy.fabs(vx-20.) < 10.**-10.), 'sphergal_to_rectgal conversion did not work as expected'
    assert numpy.all(numpy.fabs(vy-10.) < 10.**-10.), 'sphergal_to_rectgal conversion did not work as expected'
    assert numpy.all(numpy.fabs(vz-30.) < 10.**-10.), 'sphergal_to_rectgal conversion did not work as expected'
    return None

def test_rectgal_to_sphergal():
    #Test that this is the inverse of sphergal_to_rectgal
    l,b,d= 90.,30.,1.
    vr,pmll,pmbb= 10.,-20.,30.
    X,Y,Z,vx,vy,vz= bovy_coords.sphergal_to_rectgal(l,b,d,vr,pmll,pmbb,
                                                    degree=True)
    lt,bt,dt,vrt,pmllt,pmbbt= bovy_coords.rectgal_to_sphergal(X,Y,Z,
                                                              vx,vy,vz,
                                                              degree=True)
    assert numpy.fabs(lt-l) < 10.**-10., 'rectgal_to_sphergal conversion did not work as expected'
    assert numpy.fabs(bt-b) < 10.**-10., 'rectgal_to_sphergal conversion did not work as expected'
    assert numpy.fabs(dt-d) < 10.**-10., 'rectgal_to_sphergal conversion did not work as expected'
    assert numpy.fabs(vrt-vr) < 10.**-10., 'rectgal_to_sphergal conversion did not work as expected'
    assert numpy.fabs(pmllt-pmll) < 10.**-10., 'rectgal_to_sphergal conversion did not work as expected'
    assert numpy.fabs(pmbbt-pmbb) < 10.**-10., 'rectgal_to_sphergal conversion did not work as expected'
    #Also test for degree=False
    lt,bt,dt,vrt,pmllt,pmbbt= bovy_coords.rectgal_to_sphergal(X,Y,Z,
                                                              vx,vy,vz,
                                                              degree=False)
    assert numpy.fabs(lt-l/180.*numpy.pi) < 10.**-10., 'rectgal_to_sphergal conversion did not work as expected'
    assert numpy.fabs(bt-b/180.*numpy.pi) < 10.**-10., 'rectgal_to_sphergal conversion did not work as expected'
    assert numpy.fabs(dt-d) < 10.**-10., 'rectgal_to_sphergal conversion did not work as expected'
    assert numpy.fabs(vrt-vr) < 10.**-10., 'rectgal_to_sphergal conversion did not work as expected'
    assert numpy.fabs(pmllt-pmll) < 10.**-10., 'rectgal_to_sphergal conversion did not work as expected'
    assert numpy.fabs(pmbbt-pmbb) < 10.**-10., 'rectgal_to_sphergal conversion did not work as expected'
    #Also test for arrays
    os= numpy.ones(2)
    lbdvrpmllpmbbt= bovy_coords.rectgal_to_sphergal(os*X,os*Y,os*Z,
                                                              os*vx,os*vy,
                                                              os*vz,
                                                              degree=True)
    lt= lbdvrpmllpmbbt[:,0]
    bt= lbdvrpmllpmbbt[:,1]
    dt= lbdvrpmllpmbbt[:,2]
    vrt= lbdvrpmllpmbbt[:,3]
    pmllt= lbdvrpmllpmbbt[:,4]
    pmbbt= lbdvrpmllpmbbt[:,5]
    assert numpy.all(numpy.fabs(lt-l) < 10.**-10.), 'rectgal_to_sphergal conversion did not work as expected'
    assert numpy.all(numpy.fabs(bt-b) < 10.**-10.), 'rectgal_to_sphergal conversion did not work as expected'
    assert numpy.all(numpy.fabs(dt-d) < 10.**-10.), 'rectgal_to_sphergal conversion did not work as expected'
    assert numpy.all(numpy.fabs(vrt-vr) < 10.**-10.), 'rectgal_to_sphergal conversion did not work as expected'
    assert numpy.all(numpy.fabs(pmllt-pmll) < 10.**-10.), 'rectgal_to_sphergal conversion did not work as expected'
    assert numpy.all(numpy.fabs(pmbbt-pmbb) < 10.**-10.), 'rectgal_to_sphergal conversion did not work as expected'    
    return None

def test_pmrapmdec_to_pmllpmbb():
    #This is a random ra,dec
    ra, dec= 132., -20.4
    pmra, pmdec= 10., 20.
    pmll, pmbb= bovy_coords.pmrapmdec_to_pmllpmbb(pmra,pmdec,
                                              ra,dec,degree=True,epoch=1950.)
    assert numpy.fabs(numpy.sqrt(pmll**2.+pmbb**2.)-numpy.sqrt(pmra**2.+pmdec**2.)) < 10.**-10., 'pmrapmdec_to_pmllpmbb conversion did not work as expected'
    # This is close to the NGP at 1950.
    ra, dec= 192.24, 27.39
    pmra, pmdec= 10., 20.
    os= numpy.ones(2)
    pmllpmbb= bovy_coords.pmrapmdec_to_pmllpmbb(os*pmra,os*pmdec,
                                                  os*ra,os*dec,
                                                  degree=True,epoch=1950.)
    
    pmll= pmllpmbb[:,0]
    pmbb= pmllpmbb[:,1]
    assert numpy.all(numpy.fabs(numpy.sqrt(pmll**2.+pmbb**2.)-numpy.sqrt(pmra**2.+pmdec**2.)) < 10.**-10.), 'pmrapmdec_to_pmllpmbb conversion did not work as expected close to the NGP'
    # This is the NGP at 1950.
    ra, dec= 192.25, 27.4
    pmra, pmdec= 10., 20.
    os= numpy.ones(2)
    pmllpmbb= bovy_coords.pmrapmdec_to_pmllpmbb(os*pmra,os*pmdec,
                                                  os*ra,os*dec,
                                                  degree=True,epoch=1950.)
    
    pmll= pmllpmbb[:,0]
    pmbb= pmllpmbb[:,1]
    assert numpy.all(numpy.fabs(numpy.sqrt(pmll**2.+pmbb**2.)-numpy.sqrt(pmra**2.+pmdec**2.)) < 10.**-10.), 'pmrapmdec_to_pmllpmbb conversion did not work as expected for the NGP'
    # This is the NCP
    ra, dec= numpy.pi, numpy.pi/2.
    pmra, pmdec= 10., 20.
    pmll, pmbb= bovy_coords.pmrapmdec_to_pmllpmbb(pmra,pmdec,
                                                  ra,dec,degree=False,
                                                  epoch=1950.)
    assert numpy.fabs(numpy.sqrt(pmll**2.+pmbb**2.)-numpy.sqrt(pmra**2.+pmdec**2.)) < 10.**-10., 'pmrapmdec_to_pmllpmbb conversion did not work as expected for the NCP'
    return None

def test_pmllpmbb_to_pmrapmdec():
    #This is a random l,b
    ll, bb= 132., -20.4
    pmll, pmbb= 10., 20.
    pmra, pmdec= bovy_coords.pmllpmbb_to_pmrapmdec(pmll,pmbb,
                                                   ll,bb,
                                                   degree=True,epoch=1950.)
    assert numpy.fabs(numpy.sqrt(pmll**2.+pmbb**2.)-numpy.sqrt(pmra**2.+pmdec**2.)) < 10.**-10., 'pmllpmbb_to_pmrapmdec conversion did not work as expected for a random l,b'
    # This is close to the NGP
    ll,bb= numpy.pi-0.001, numpy.pi/2.-0.001
    pmll, pmbb= 10., 20.
    os= numpy.ones(2)
    pmrapmdec= bovy_coords.pmllpmbb_to_pmrapmdec(os*pmll,os*pmbb,
                                                 os*ll,os*bb,
                                                 degree=False,epoch=1950.)
    pmra= pmrapmdec[:,0]
    pmdec= pmrapmdec[:,1]
    assert numpy.all(numpy.fabs(numpy.sqrt(pmll**2.+pmbb**2.)-numpy.sqrt(pmra**2.+pmdec**2.)) < 10.**-10.), 'pmllpmbb_to_pmrapmdec conversion did not work as expected close to the NGP'
    # This is the NGP
    ll,bb= numpy.pi, numpy.pi/2.
    pmll, pmbb= 10., 20.
    os= numpy.ones(2)
    pmrapmdec= bovy_coords.pmllpmbb_to_pmrapmdec(os*pmll,os*pmbb,
                                                 os*ll,os*bb,
                                                 degree=False,epoch=1950.)
    pmra= pmrapmdec[:,0]
    pmdec= pmrapmdec[:,1]
    assert numpy.all(numpy.fabs(numpy.sqrt(pmll**2.+pmbb**2.)-numpy.sqrt(pmra**2.+pmdec**2.)) < 10.**-10.), 'pmllpmbb_to_pmrapmdec conversion did not work as expected at the NGP'
    # This is the NCP
    ra, dec= numpy.pi, numpy.pi/2.
    ll, bb= bovy_coords.radec_to_lb(ra,dec,degree=False,epoch=1950.)
    pmll, pmbb= 10., 20.
    pmra, pmdec= bovy_coords.pmllpmbb_to_pmrapmdec(pmll,pmbb,
                                                   ll,bb,
                                                   degree=False,epoch=1950.)
    assert numpy.fabs(numpy.sqrt(pmll**2.+pmbb**2.)-numpy.sqrt(pmra**2.+pmdec**2.)) < 10.**-10., 'pmllpmbb_to_pmrapmdec conversion did not work as expected at the NCP'
    return None

def test_cov_pmradec_to_pmllbb():
    # This is the NGP at 1950., for this the parallactic angle is 180
    ra, dec= 192.25, 27.4
    cov_pmrapmdec= numpy.array([[100.,100.],[100.,400.]])
    cov_pmllpmbb= bovy_coords.cov_pmrapmdec_to_pmllpmbb(cov_pmrapmdec,
                                                       ra,dec,
                                                       degree=True,
                                                       epoch=1950.)
    
    assert numpy.fabs(cov_pmllpmbb[0,0]-100.) < 10.**-10., 'cov_pmradec_to_pmllbb conversion did not work as expected'
    assert numpy.fabs(cov_pmllpmbb[0,1]-100.) < 10.**-10., 'cov_pmradec_to_pmllbb conversion did not work as expected'
    assert numpy.fabs(cov_pmllpmbb[1,0]-100.) < 10.**-10., 'cov_pmradec_to_pmllbb conversion did not work as expected'
    assert numpy.fabs(cov_pmllpmbb[1,1]-400.) < 10.**-10., 'cov_pmradec_to_pmllbb conversion did not work as expected'
    # This is a random position, check that the conversion makes sense
    ra, dec= 132.25, -23.4
    cov_pmrapmdec= numpy.array([[100.,100.],[100.,400.]])
    cov_pmllpmbb= bovy_coords.cov_pmrapmdec_to_pmllpmbb(cov_pmrapmdec,
                                                        ra/180.*numpy.pi,
                                                        dec/180.*numpy.pi,
                                                        degree=False,
                                                        epoch=1950.)
    assert numpy.fabs(numpy.linalg.det(cov_pmllpmbb)-numpy.linalg.det(cov_pmrapmdec)) < 10.**-10., 'cov_pmradec_to_pmllbb conversion did not work as expected'
    assert numpy.fabs(numpy.trace(cov_pmllpmbb)-numpy.trace(cov_pmrapmdec)) < 10.**-10., 'cov_pmradec_to_pmllbb conversion did not work as expected'
    # This is a random position, check that the conversion makes sense, arrays
    ra, dec= 132.25, -23.4
    icov_pmrapmdec= numpy.array([[100.,100.],[100.,400.]])
    cov_pmrapmdec= numpy.empty((3,2,2))
    for ii in range(3): cov_pmrapmdec[ii,:,:]= icov_pmrapmdec
    os= numpy.ones(3)
    cov_pmllpmbb= bovy_coords.cov_pmrapmdec_to_pmllpmbb(cov_pmrapmdec,
                                                        os*ra,
                                                        os*dec,
                                                        degree=True,
                                                        epoch=1950.)
    for ii in range(3):
        assert numpy.fabs(numpy.linalg.det(cov_pmllpmbb[ii,:,:])-numpy.linalg.det(cov_pmrapmdec[ii,:,:])) < 10.**-10., 'cov_pmradec_to_pmllbb conversion did not work as expected'
        assert numpy.fabs(numpy.trace(cov_pmllpmbb[ii,:,:])-numpy.trace(cov_pmrapmdec[ii,:,:])) < 10.**-10., 'cov_pmradec_to_pmllbb conversion did not work as expected'
    return None

def test_cov_dvrpmllbb_to_vxyz():
    l,b,d= 90., 0., 2.
    e_d, e_vr= 0.2, 2.
    cov_pmllpmbb= numpy.array([[100.,0.],[0.,400.]])
    pmll,pmbb= 20.,30.
    cov_vxvyvz= bovy_coords.cov_dvrpmllbb_to_vxyz(d,e_d,e_vr,
                                                  pmll,pmbb,
                                                  cov_pmllpmbb,
                                                  l,b,
                                                  degree=True,
                                                  plx=False)
    assert numpy.fabs(numpy.sqrt(cov_vxvyvz[0,0])
                      -d*4.740470463496208*pmll*numpy.sqrt((e_d/d)**2.+(10./pmll)**2.)) < 10.**-10., 'cov_dvrpmllbb_to_vxyz coversion did not work as expected'
    assert numpy.fabs(numpy.sqrt(cov_vxvyvz[1,1])-e_vr) < 10.**-10., 'cov_dvrpmllbb_to_vxyz coversion did not work as expected'
    assert numpy.fabs(numpy.sqrt(cov_vxvyvz[2,2])
                      -d*4.740470463496208*pmbb*numpy.sqrt((e_d/d)**2.+(20./pmbb)**2.)) < 10.**-10., 'cov_dvrpmllbb_to_vxyz coversion did not work as expected'
    #Another one
    l,b,d= 180., 0., 1./2.
    e_d, e_vr= 0.05, 2.
    cov_pmllpmbb= numpy.array([[100.,0.],[0.,400.]])
    pmll,pmbb= 20.,30.
    cov_vxvyvz= bovy_coords.cov_dvrpmllbb_to_vxyz(d,e_d,e_vr,
                                                  pmll,pmbb,
                                                  cov_pmllpmbb,
                                                  l/180.*numpy.pi,
                                                  b/180.*numpy.pi,
                                                  degree=False,
                                                  plx=True)
    assert numpy.fabs(numpy.sqrt(cov_vxvyvz[0,0])-e_vr) < 10.**-10., 'cov_dvrpmllbb_to_vxyz coversion did not work as expected'
    assert numpy.fabs(numpy.sqrt(cov_vxvyvz[1,1])
                      -1./d*4.740470463496208*pmll*numpy.sqrt((e_d/d)**2.+(10./pmll)**2.)) < 10.**-10., 'cov_dvrpmllbb_to_vxyz coversion did not work as expected'
    assert numpy.fabs(numpy.sqrt(cov_vxvyvz[2,2])
                      -1./d*4.740470463496208*pmbb*numpy.sqrt((e_d/d)**2.+(20./pmbb)**2.)) < 10.**-10., 'cov_dvrpmllbb_to_vxyz coversion did not work as expected'
    #Another one, w/ arrays
    l,b,d= 90., 90., 2.
    e_d, e_vr= 0.2, 2.
    tcov_pmllpmbb= numpy.array([[100.,0.],[0.,400.]])
    cov_pmllpmbb= numpy.empty((3,2,2))
    for ii in range(3): cov_pmllpmbb[ii,:,:]= tcov_pmllpmbb
    pmll,pmbb= 20.,30.
    os= numpy.ones(3)
    cov_vxvyvz= bovy_coords.cov_dvrpmllbb_to_vxyz(os*d,os*e_d,os*e_vr,
                                                  os*pmll,os*pmbb,
                                                  cov_pmllpmbb,
                                                  os*l,os*b,
                                                  degree=True,
                                                  plx=False)
    for ii in range(3):
        assert numpy.fabs(numpy.sqrt(cov_vxvyvz[ii,0,0])
                          -d*4.740470463496208*pmll*numpy.sqrt((e_d/d)**2.+(10./pmll)**2.)) < 10.**-10., 'cov_dvrpmllbb_to_vxyz coversion did not work as expected'
        assert numpy.fabs(numpy.sqrt(cov_vxvyvz[ii,1,1])
                          -d*4.740470463496208*pmbb*numpy.sqrt((e_d/d)**2.+(20./pmbb)**2.)) < 10.**-10., 'cov_dvrpmllbb_to_vxyz coversion did not work as expected'
        assert numpy.fabs(numpy.sqrt(cov_vxvyvz[ii,2,2])-e_vr) < 10.**-10., 'cov_dvrpmllbb_to_vxyz coversion did not work as expected'
    return None

def test_dl_to_rphi_2d():
    #This is a tangent point
    l= numpy.arcsin(0.75)
    d= 6./numpy.tan(l)
    r,phi= bovy_coords.dl_to_rphi_2d(d,l,degree=False,ro=8.,phio=0.)
    assert numpy.fabs(r-6.) < 10.**-10., 'dl_to_rphi_2d conversion did not work as expected'
    assert numpy.fabs(phi-numpy.arccos(0.75)) < 10.**-10., 'dl_to_rphi_2d conversion did not work as expected'
    #This is a different point
    d,l= 2., 45.
    r,phi= bovy_coords.dl_to_rphi_2d(d,l,degree=True,ro=2.*numpy.sqrt(2.),
                                     phio=10.)
    assert numpy.fabs(r-2.) < 10.**-10., 'dl_to_rphi_2d conversion did not work as expected'
    assert numpy.fabs(phi-55.) < 10.**-10., 'dl_to_rphi_2d conversion did not work as expected'
    #This is a different point, for array
    d,l= 2., 45.
    os= numpy.ones(2)
    r,phi= bovy_coords.dl_to_rphi_2d(os*d,os*l,degree=True,
                                     ro=2.*numpy.sqrt(2.),
                                     phio=0.)
    assert numpy.all(numpy.fabs(r-2.) < 10.**-10.), 'dl_to_rphi_2d conversion did not work as expected'
    assert numpy.all(numpy.fabs(phi-45.) < 10.**-10.), 'dl_to_rphi_2d conversion did not work as expected'
    #This is a different point, for list (which I support for some reason)
    d,l= 2., 45.
    r,phi= bovy_coords.dl_to_rphi_2d([d,d],[l,l],degree=True,
                                     ro=2.*numpy.sqrt(2.),
                                     phio=0.)
    r= numpy.array(r)
    phi= numpy.array(phi)
    assert numpy.all(numpy.fabs(r-2.) < 10.**-10.), 'dl_to_rphi_2d conversion did not work as expected'
    assert numpy.all(numpy.fabs(phi-45.) < 10.**-10.), 'dl_to_rphi_2d conversion did not work as expected'
    return None

def test_rphi_to_dl_2d():
    #This is a tangent point
    r,phi= 6., numpy.arccos(0.75)
    d,l= bovy_coords.rphi_to_dl_2d(r,phi,degree=False,ro=8.,phio=0.)
    l= numpy.arcsin(0.75)
    d= 6./numpy.tan(l)
    assert numpy.fabs(d-6./numpy.tan(numpy.arcsin(0.75))) < 10.**-10., 'dl_to_rphi_2d conversion did not work as expected'
    assert numpy.fabs(l-numpy.arcsin(0.75)) < 10.**-10., 'rphi_to_dl_2d conversion did not work as expected'
    #This is another point
    r,phi= 2., 55.
    d,l= bovy_coords.rphi_to_dl_2d(r,phi,degree=True,ro=2.*numpy.sqrt(2.),
                                   phio=10.)
    assert numpy.fabs(d-2.) < 10.**-10., 'rphi_to_dl_2d conversion did not work as expected'
    assert numpy.fabs(l-45.) < 10.**-10., 'rphi_to_dl_2d conversion did not work as expected'
    #This is another point, for arrays
    r,phi= 2., 45.
    os= numpy.ones(2)
    d,l= bovy_coords.rphi_to_dl_2d(os*r,os*phi,
                                   degree=True,ro=2.*numpy.sqrt(2.),
                                   phio=0.)
    assert numpy.all(numpy.fabs(d-2.) < 10.**-10.), 'rphi_to_dl_2d conversion did not work as expected'
    assert numpy.all(numpy.fabs(l-45.) < 10.**-10.), 'rphi_to_dl_2d conversion did not work as expected'
    #This is another point, for lists, which for some reason I support
    r,phi= 2., 45.
    d,l= bovy_coords.rphi_to_dl_2d([r,r],[phi,phi],
                                   degree=True,ro=2.*numpy.sqrt(2.),
                                   phio=0.)
    d= numpy.array(d)
    l= numpy.array(l)
    assert numpy.all(numpy.fabs(d-2.) < 10.**-10.), 'rphi_to_dl_2d conversion did not work as expected'
    assert numpy.all(numpy.fabs(l-45.) < 10.**-10.), 'rphi_to_dl_2d conversion did not work as expected'
    return None

def test_uv_to_Rz():
    u, v= numpy.arccosh(5./3.), numpy.pi/6.
    R,z= bovy_coords.uv_to_Rz(u,v,delta=3.)
    assert numpy.fabs(R-2.) < 10.**-10., 'uv_to_Rz conversion did not work as expected'
    assert numpy.fabs(z-2.5*numpy.sqrt(3.)) < 10.**-10., 'uv_to_Rz conversion did not work as expected'
    #Also test for arrays
    os= numpy.ones(2)
    R,z= bovy_coords.uv_to_Rz(os*u,os*v,delta=3.)
    assert numpy.all(numpy.fabs(R-2.) < 10.**-10.), 'uv_to_Rz conversion did not work as expected'
    assert numpy.all(numpy.fabs(z-2.5*numpy.sqrt(3.)) < 10.**-10.), 'uv_to_Rz conversion did not work as expected'
    return None

def test_Rz_to_uv():
    u, v= numpy.arccosh(5./3.), numpy.pi/6.
    ut,vt= bovy_coords.Rz_to_uv(*bovy_coords.uv_to_Rz(u,v,delta=3.),delta=3.)
    assert numpy.fabs(ut-u) < 10.**-10., 'Rz_to_uvz conversion did not work as expected'
    assert numpy.fabs(vt-v) < 10.**-10., 'Rz_to_uv conversion did not work as expected'
    #Also test for arrays
    os= numpy.ones(2)
    ut,vt= bovy_coords.Rz_to_uv(*bovy_coords.uv_to_Rz(u*os,v*os,delta=3.),delta=3.)
    assert numpy.all(numpy.fabs(ut-u) < 10.**-10.), 'Rz_to_uvz conversion did not work as expected'
    assert numpy.all(numpy.fabs(vt-v) < 10.**-10.), 'Rz_to_uv conversion did not work as expected'
    return None

def test_Rz_to_coshucosv():
    u, v= numpy.arccosh(5./3.), numpy.pi/3.
    R,z= bovy_coords.uv_to_Rz(u,v,delta=3.)
    coshu,cosv= bovy_coords.Rz_to_coshucosv(R,z,delta=3.)
    assert numpy.fabs(coshu-5./3.) < 10.**-10., 'Rz_to_coshucosv conversion did notwork as expected'
    assert numpy.fabs(cosv-0.5) < 10.**-10., 'Rz_to_coshucosv conversion did notwork as expected'
    #Also test for arrays
    os= numpy.ones(2)
    coshu,cosv= bovy_coords.Rz_to_coshucosv(R*os,z*os,delta=3.)
    assert numpy.all(numpy.fabs(coshu-5./3.) < 10.**-10.), 'Rz_to_coshucosv conversion did notwork as expected'
    assert numpy.all(numpy.fabs(cosv-0.5) < 10.**-10.), 'Rz_to_coshucosv conversion did notwork as expected'
    return None

def test_uv_to_Rz_oblate():
    u, v= numpy.arccosh(5./3.), numpy.pi/6.
    R,z= bovy_coords.uv_to_Rz(u,v,delta=3.,oblate=True)
    assert numpy.fabs(R-2.5) < 10.**-10., 'uv_to_Rz conversion did not work as expected'
    assert numpy.fabs(z-2.*numpy.sqrt(3.)) < 10.**-10., 'uv_to_Rz conversion did not work as expected'
    #Also test for arrays
    os= numpy.ones(2)
    R,z= bovy_coords.uv_to_Rz(os*u,os*v,delta=3.,oblate=True)
    assert numpy.all(numpy.fabs(R-2.5) < 10.**-10.), 'uv_to_Rz conversion did not work as expected'
    assert numpy.all(numpy.fabs(z-2.*numpy.sqrt(3.)) < 10.**-10.), 'uv_to_Rz conversion did not work as expected'
    return None

def test_Rz_to_uv_oblate():
    u, v= numpy.arccosh(5./3.), numpy.pi/6.
    ut,vt= bovy_coords.Rz_to_uv(*bovy_coords.uv_to_Rz(u,v,
                                                      delta=3.,oblate=True),
                                 delta=3.,oblate=True)
    assert numpy.fabs(ut-u) < 10.**-10., 'Rz_to_uvz conversion did not work as expected'
    assert numpy.fabs(vt-v) < 10.**-10., 'Rz_to_uv conversion did not work as expected'
    #Also test for arrays
    os= numpy.ones(2)
    ut,vt= bovy_coords.Rz_to_uv(*bovy_coords.uv_to_Rz(u*os,v*os,
                                                      delta=3.,oblate=True),
                                 delta=3.,oblate=True)
    assert numpy.all(numpy.fabs(ut-u) < 10.**-10.), 'Rz_to_uvz conversion did not work as expected'
    assert numpy.all(numpy.fabs(vt-v) < 10.**-10.), 'Rz_to_uv conversion did not work as expected'
    return None

def test_Rz_to_coshucosv_oblate():
    u, v= numpy.arccosh(5./3.), numpy.pi/3.
    R,z= bovy_coords.uv_to_Rz(u,v,delta=3.,oblate=True)
    coshu,cosv= bovy_coords.Rz_to_coshucosv(R,z,delta=3.,oblate=True)
    assert numpy.fabs(coshu-5./3.) < 10.**-10., 'Rz_to_coshucosv conversion did notwork as expected'
    assert numpy.fabs(cosv-0.5) < 10.**-10., 'Rz_to_coshucosv conversion did notwork as expected'
    #Also test for arrays
    os= numpy.ones(2)
    coshu,cosv= bovy_coords.Rz_to_coshucosv(R*os,z*os,delta=3.,oblate=True)
    assert numpy.all(numpy.fabs(coshu-5./3.) < 10.**-10.), 'Rz_to_coshucosv conversion did notwork as expected'
    assert numpy.all(numpy.fabs(cosv-0.5) < 10.**-10.), 'Rz_to_coshucosv conversion did notwork as expected'
    return None

def test_vRvz_to_pupv():
    # Some sanity checks
    # At R,z << Delta --> p_u ~ delta vR, p_v ~ -delta vz
    delta= 0.5
    R,z= delta/100., delta/300.
    vR, vz= 0.2,-0.5
    assert numpy.fabs(bovy_coords.vRvz_to_pupv(vR,vz,R,z,delta=delta)[0]-delta*vR) < 10.**-3., 'vRvz_to_pupv at small R,z does not behave as expected'
    assert numpy.fabs(bovy_coords.vRvz_to_pupv(vR,vz,R,z,delta=delta)[1]+delta*vz) < 10.**-3., 'vRvz_to_pupv at small R,z does not behave as expected'
    # At R,z >> Delta --> p_u ~ r v_r, p_v ~ r v_theta, spherical velocities
    delta= 0.5
    R,z= delta*100., delta*300.
    vR, vz= 0.2,-0.5
    # Compute spherical velocities
    r= numpy.sqrt(R**2.+z**2.)
    costheta= z/r
    sintheta= R/r
    vr= vR*sintheta+vz*costheta
    vt= -vz*sintheta+vR*costheta
    assert numpy.fabs(bovy_coords.vRvz_to_pupv(vR,vz,R,z,delta=delta)[0]-r*vr) < 10.**-3., 'vRvz_to_pupv at large R,z does not behave as expected'
    assert numpy.fabs(bovy_coords.vRvz_to_pupv(vR,vz,R,z,delta=delta)[1]-r*vt) < 10.**-3., 'vRvz_to_pupv at large R,z does not behave as expected'
    # Also check that it does not matter whether we give R,z or u,v
    delta= 0.5
    R,z= delta*2., delta/3.
    vR, vz= 0.2,-0.5
    assert numpy.fabs(bovy_coords.vRvz_to_pupv(vR,vz,R,z,delta=delta)[0]-bovy_coords.vRvz_to_pupv(vR,vz,*bovy_coords.Rz_to_uv(R,z,delta=delta),delta=delta,uv=True)[0]) < 10.**-3., 'vRvz_to_pupv with and without pre-computed u,v do not agree'
    assert numpy.fabs(bovy_coords.vRvz_to_pupv(vR,vz,R,z,delta=delta)[1]-bovy_coords.vRvz_to_pupv(vR,vz,*bovy_coords.Rz_to_uv(R,z,delta=delta),delta=delta,uv=True)[1]) < 10.**-3., 'vRvz_to_pupv with and without pre-computed u,v do not agree'
    return None

def test_vRvz_to_pupv_oblate():
    # Some sanity checks
    # At R,z << Delta --> p_u ~ delta vz, p_v ~ delta vR
    delta= 0.5
    R,z= delta/100., delta/300.
    vR, vz= 0.2,-0.5
    assert numpy.fabs(bovy_coords.vRvz_to_pupv(vR,vz,R,z,delta=delta,oblate=True)[0]-delta*vz) < 10.**-3., 'vRvz_to_pupv at small R,z does not behave as expected for oblate spheroidal coordinates'
    assert numpy.fabs(bovy_coords.vRvz_to_pupv(vR,vz,R,z,delta=delta,oblate=True)[1]-delta*vR) < 10.**-3., 'vRvz_to_pupv at small R,z does not behave as expected for oblate spheroidal coordinates'
    # At R,z >> Delta --> p_u ~ r v_r, p_v ~ r v_theta, spherical velocities
    delta= 0.5
    R,z= delta*100., delta*300.
    vR, vz= 0.2,-0.5
    # Compute spherical velocities
    r= numpy.sqrt(R**2.+z**2.)
    costheta= z/r
    sintheta= R/r
    vr= vR*sintheta+vz*costheta
    vt= -vz*sintheta+vR*costheta
    assert numpy.fabs(bovy_coords.vRvz_to_pupv(vR,vz,R,z,delta=delta,oblate=True)[0]-r*vr) < 10.**-3., 'vRvz_to_pupv at large R,z does not behave as expected for oblate spheroidal coordinates'
    assert numpy.fabs(bovy_coords.vRvz_to_pupv(vR,vz,R,z,delta=delta,oblate=True)[1]-r*vt) < 10.**-3., 'vRvz_to_pupv at large R,z does not behave as expected for oblate spheroidal coordinates'
    # Also check that it does not matter whether we give R,z or u,v
    delta= 0.5
    R,z= delta*2., delta/3.
    vR, vz= 0.2,-0.5
    assert numpy.fabs(bovy_coords.vRvz_to_pupv(vR,vz,R,z,delta=delta,oblate=True)[0]-bovy_coords.vRvz_to_pupv(vR,vz,*bovy_coords.Rz_to_uv(R,z,delta=delta,oblate=True),delta=delta,oblate=True,uv=True)[0]) < 10.**-3., 'vRvz_to_pupv with and without pre-computed u,v do not agree for oblate spheroidal coordinates'
    assert numpy.fabs(bovy_coords.vRvz_to_pupv(vR,vz,R,z,delta=delta,oblate=True)[1]-bovy_coords.vRvz_to_pupv(vR,vz,*bovy_coords.Rz_to_uv(R,z,delta=delta,oblate=True),delta=delta,oblate=True,uv=True)[1]) < 10.**-3., 'vRvz_to_pupv with and without pre-computed u,v do not agree for oblate spheroidal coordinates'
    return None

def test_pupv_to_vRvz():
    # Test that this is the inverse of vRvz_to_pupv
    delta= 0.5
    R,z= delta/2., delta*3.
    vR, vz= 0.2,-0.5
    u,v= bovy_coords.Rz_to_uv(R,z,delta=delta)
    pu,pv= bovy_coords.vRvz_to_pupv(vR,vz,R,z,delta=delta)
    assert numpy.fabs(bovy_coords.pupv_to_vRvz(pu,pv,u,v,delta=delta)[0]-vR) < 1e-8, 'pupv_to_vRvz is not the inverse of vRvz_to_pupv'
    assert numpy.fabs(bovy_coords.pupv_to_vRvz(pu,pv,u,v,delta=delta)[1]-vz) < 1e-8, 'pupv_to_vRvz is not the inverse of vRvz_to_pupv'
    # Another one
    delta= 1.5
    R,z= delta*2., -delta/3.
    vR, vz= -0.2,0.5
    u,v= bovy_coords.Rz_to_uv(R,z,delta=delta)
    pu,pv= bovy_coords.vRvz_to_pupv(vR,vz,R,z,delta=delta)
    assert numpy.fabs(bovy_coords.pupv_to_vRvz(pu,pv,u,v,delta=delta)[0]-vR) < 1e-8, 'pupv_to_vRvz is not the inverse of vRvz_to_pupv'
    assert numpy.fabs(bovy_coords.pupv_to_vRvz(pu,pv,u,v,delta=delta)[1]-vz) < 1e-8, 'pupv_to_vRvz is not the inverse of vRvz_to_pupv'
    return None

def test_pupv_to_vRvz_oblate():
    # Test that this is the inverse of vRvz_to_pupv
    delta= 0.5
    R,z= delta/2., delta*3.
    vR, vz= 0.2,-0.5
    u,v= bovy_coords.Rz_to_uv(R,z,delta=delta,oblate=True)
    pu,pv= bovy_coords.vRvz_to_pupv(vR,vz,R,z,delta=delta,oblate=True)
    assert numpy.fabs(bovy_coords.pupv_to_vRvz(pu,pv,u,v,delta=delta,oblate=True)[0]-vR) < 1e-8, 'pupv_to_vRvz is not the inverse of vRvz_to_pupv'
    assert numpy.fabs(bovy_coords.pupv_to_vRvz(pu,pv,u,v,delta=delta,oblate=True)[1]-vz) < 1e-8, 'pupv_to_vRvz is not the inverse of vRvz_to_pupv'
    # Another one
    delta= 1.5
    R,z= delta*2., -delta/3.
    vR, vz= -0.2,0.5
    u,v= bovy_coords.Rz_to_uv(R,z,delta=delta,oblate=True)
    pu,pv= bovy_coords.vRvz_to_pupv(vR,vz,R,z,delta=delta,oblate=True)
    assert numpy.fabs(bovy_coords.pupv_to_vRvz(pu,pv,u,v,delta=delta,oblate=True)[0]-vR) < 1e-8, 'pupv_to_vRvz is not the inverse of vRvz_to_pupv'
    assert numpy.fabs(bovy_coords.pupv_to_vRvz(pu,pv,u,v,delta=delta,oblate=True)[1]-vz) < 1e-8, 'pupv_to_vRvz is not the inverse of vRvz_to_pupv'
    return None

def test_lbd_to_XYZ_jac():
    #Just position
    l,b,d= 180.,30.,2.
    jac= bovy_coords.lbd_to_XYZ_jac(l,b,d,degree=True)
    assert numpy.fabs(jac[0,0]-0.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[0,1]-numpy.pi/180.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[0,2]+numpy.sqrt(3.)/2.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[1,0]+numpy.sqrt(3.)*numpy.pi/180.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[1,1]-0.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[1,2]-0.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[2,0]-0.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[2,1]-numpy.sqrt(3.)*numpy.pi/180.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[2,2]-0.5) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    #6D
    l,b,d= 3.*numpy.pi/2.,numpy.pi/6.,2.
    vr,pmll,pmbb= 10.,20.,-30.
    jac= bovy_coords.lbd_to_XYZ_jac(l,b,d,vr,pmll,pmbb,degree=False)
    assert numpy.fabs(jac[0,0]-numpy.sqrt(3.)) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[0,1]-0.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[0,2]-0.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[1,0]-0.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[1,1]-1.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[1,2]+numpy.sqrt(3.)/2.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[2,0]-0.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[2,1]-numpy.sqrt(3.)) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[2,2]-0.5) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.all(numpy.fabs(jac[:3,3:]) < 10.**-10.), 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[3,0]-numpy.sqrt(3.)/2.*vr+0.5*pmbb*d*4.740470463496208) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[3,1]-0.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[3,2]-pmll*4.740470463496208) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[3,3]-0.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[3,4]-d*4.740470463496208) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[3,5]-0.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[4,0]-pmll*d*4.740470463496208) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[4,1]-vr/2.-numpy.sqrt(3.)/2.*d*pmbb*4.740470463496208) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[4,2]-0.5*4.740470463496208*pmbb) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[4,3]+numpy.sqrt(3.)/2.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[4,4]-0.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[4,5]-4.740470463496208) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[5,0]-0.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[5,1]+0.5*d*4.740470463496208*pmbb-numpy.sqrt(3.)/2.*vr) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[5,2]-numpy.sqrt(3.)/2.*4.740470463496208*pmbb) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[5,3]-0.5) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[5,4]-0.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[5,5]-numpy.sqrt(3.)/2.*d*4.740470463496208) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    return None

def test_cyl_to_spher():
    # Just a few quick tests
    r,t,p= bovy_coords.cyl_to_spher(1.2,3.2,1.)
    assert numpy.fabs(r**2.-1.2**2.-3.2**2.) < 10.**-8., 'cyl_to_spher does not work as expected'
    assert numpy.fabs(r*numpy.cos(t)-3.2) < 10.**-8., 'cyl_to_spher does not work as expected'
    assert numpy.fabs(p-1.) < 10.**-8., 'cyl_to_spher does not work as expected'
    r,t,p= bovy_coords.cyl_to_spher(1.2,-3.2,4.)
    assert numpy.fabs(r**2.-1.2**2.-3.2**2.) < 10.**-8., 'cyl_to_spher does not work as expected'
    assert numpy.fabs(r*numpy.cos(t)+3.2) < 10.**-8., 'cyl_to_spher does not work as expected'
    assert numpy.fabs(p-4.) < 10.**-8., 'cyl_to_spher does not work as expected'
    return None

def test_spher_to_cyl():
    # Just a few quick tests
    R,z,p= bovy_coords.spher_to_cyl(5.,numpy.arccos(3./5.),1.)
    assert numpy.fabs(R-4.) < 10.**-8., 'spher_to_cyl does not work as expected'
    assert numpy.fabs(z-3.) < 10.**-8., 'spher_to_cyl does not work as expected'
    assert numpy.fabs(p-1.) < 10.**-8., 'spher_to_cyl does not work as expected'
    R,z,p= bovy_coords.spher_to_cyl(5.,numpy.arccos(-3./5.),4.)
    assert numpy.fabs(R-4.) < 10.**-8., 'spher_to_cyl does not work as expected'
    assert numpy.fabs(z+3.) < 10.**-8., 'spher_to_cyl does not work as expected'
    assert numpy.fabs(p-4.) < 10.**-8., 'spher_to_cyl does not work as expected'
    return None

def test_cyl_to_rect_jac():
    #Just position
    R,phi,Z= 2., numpy.pi, 1.
    jac= bovy_coords.cyl_to_rect_jac(R,phi,Z)
    assert numpy.fabs(numpy.linalg.det(jac)-R) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[0,0]+1.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[0,1]-0.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[0,2]-0.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[1,0]-0.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[1,1]+2.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[1,2]-0.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[2,0]-0.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[2,1]-0.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[2,2]-1.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    #6D
    R,phi,Z= 2., numpy.pi, 1.
    vR,vT,vZ= 1.,2.,3.
    jac= bovy_coords.cyl_to_rect_jac(R,vR,vT,Z,vZ,phi)
    vindx= numpy.array([False,True,True,False,True,False],dtype='bool')
    assert numpy.fabs(numpy.linalg.det(jac)-R) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[0,0]+1.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[0,5]-0.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[0,3]-0.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.all(numpy.fabs(jac[0,vindx]) < 10.**-10.), 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[1,0]-0.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[1,5]+2.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[1,3]-0.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.all(numpy.fabs(jac[1,vindx]) < 10.**-10.), 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[2,0]-0.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[2,5]-0.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[2,3]-1.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.all(numpy.fabs(jac[2,vindx]) < 10.**-10.), 'cyl_to_rect_jac calculation did not work as expected'
    #Velocities
    assert numpy.fabs(jac[3,0]-0.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[3,1]+1.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[3,2]-0.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[3,3]-0.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[3,4]-0.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[3,5]-2.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[4,0]-0.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[4,1]-0.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[4,2]+1.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[4,3]-0.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[4,4]-0.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[4,5]+1.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.all(numpy.fabs(jac[5,numpy.array([True,True,True,True,False,True],dtype='bool')]-0.) < 10.**-10.), 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[5,4]-1.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    return None

def test_radec_to_custom_valueerror():
    # Test the radec_to_custom without T raises a ValueError
    with pytest.raises(ValueError):
        xieta= bovy_coords.radec_to_custom(20.,30.)
    return None

def test_radec_to_custom_againstlb():
    _turn_off_apy()
    ra, dec= 20., 30.
    theta,dec_ngp,ra_ngp= bovy_coords.get_epoch_angles(2000.)
    T= numpy.dot(numpy.array([[numpy.cos(ra_ngp),-numpy.sin(ra_ngp),0.],
                              [numpy.sin(ra_ngp),numpy.cos(ra_ngp),0.],
                              [0.,0.,1.]]),
                 numpy.dot(numpy.array([[-numpy.sin(dec_ngp),0.,
                                          numpy.cos(dec_ngp)],
                                        [0.,1.,0.],
                                        [numpy.cos(dec_ngp),0.,
                                         numpy.sin(dec_ngp)]]),
                           numpy.array([[numpy.cos(theta),numpy.sin(theta),0.],
                                        [numpy.sin(theta),-numpy.cos(theta),0.],
                                        [0.,0.,1.]])))
    lb_direct= bovy_coords.radec_to_lb(ra,dec,degree=True)
    lb_custom= bovy_coords.radec_to_custom(ra,dec,T=T.T,degree=True)
    assert numpy.fabs(lb_direct[0]-lb_custom[0]) < 10.**-8., 'radec_to_custom for transformation to l,b does not work properly'
    assert numpy.fabs(lb_direct[1]-lb_custom[1]) < 10.**-8., 'radec_to_custom for transformation to l,b does not work properly'
    # Array
    s= numpy.arange(2)
    lb_direct= bovy_coords.radec_to_lb(ra*s,dec*s,degree=True)
    lb_custom= bovy_coords.radec_to_custom(ra*s,dec*s,T=T.T,degree=True)
    assert numpy.all(numpy.fabs(lb_direct-lb_custom) < 10.**-8.), 'radec_to_custom for transformation to l,b does not work properly'
    _turn_on_apy()
    return None

def test_radec_to_custom_pal5():
    # Test the custom ra,dec transformation for Pal 5
    _RAPAL5= 229.018/180.*numpy.pi
    _DECPAL5= -0.124/180.*numpy.pi
    _TPAL5= numpy.dot(numpy.array([[numpy.cos(_DECPAL5),0.,numpy.sin(_DECPAL5)],
                                   [0.,1.,0.],
                                   [-numpy.sin(_DECPAL5),0.,numpy.cos(_DECPAL5)]]),
                      numpy.array([[numpy.cos(_RAPAL5),numpy.sin(_RAPAL5),0.],
                                   [-numpy.sin(_RAPAL5),numpy.cos(_RAPAL5),0.],
                                   [0.,0.,1.]]))
    xieta= bovy_coords.radec_to_custom(_RAPAL5,_DECPAL5,T=_TPAL5,degree=False)
    assert numpy.fabs(xieta[0]) < 10.**-8., 'radec_to_custom does not work properly for Pal 5 transformation'
    assert numpy.fabs(xieta[1]) < 10.**-8., 'radec_to_custom does not work properly for Pal 5 transformation'
    # One more, rough estimate based on visual inspection of plot
    xieta= bovy_coords.radec_to_custom(240.,6.,T=_TPAL5,degree=True)
    assert numpy.fabs(xieta[0]-11.) < 0.2, 'radec_to_custom does not work properly for Pal 5 transformation'
    assert numpy.fabs(xieta[1]-6.) < 0.2, 'radec_to_custom does not work properly for Pal 5 transformation'
    return None

def test_pmrapmdec_to_custom_valueerror():
    # Test the pmrapmdec_to_custom without T raises a ValueError
    with pytest.raises(ValueError):
        xieta= bovy_coords.pmrapmdec_to_custom(1.,1.,20.,30.)
    return None

def test_pmrapmdec_to_custom_againstlb():
    _turn_off_apy()
    ra, dec= 20., 30.
    pmra, pmdec= -3.,4.
    theta,dec_ngp,ra_ngp= bovy_coords.get_epoch_angles(2000.)
    T= numpy.dot(numpy.array([[numpy.cos(ra_ngp),-numpy.sin(ra_ngp),0.],
                              [numpy.sin(ra_ngp),numpy.cos(ra_ngp),0.],
                              [0.,0.,1.]]),
                 numpy.dot(numpy.array([[-numpy.sin(dec_ngp),0.,
                                          numpy.cos(dec_ngp)],
                                        [0.,1.,0.],
                                        [numpy.cos(dec_ngp),0.,
                                         numpy.sin(dec_ngp)]]),
                           numpy.array([[numpy.cos(theta),numpy.sin(theta),0.],
                                        [numpy.sin(theta),-numpy.cos(theta),0.],
                                        [0.,0.,1.]])))
    pmlb_direct= bovy_coords.pmrapmdec_to_pmllpmbb(pmra,pmdec,ra,dec,
                                                   degree=True)
    pmlb_custom= bovy_coords.pmrapmdec_to_custom(pmra,pmdec,ra,dec,
                                                 T=T.T,degree=True)
    assert numpy.fabs(pmlb_direct[0]-pmlb_custom[0]) < 10.**-8., 'pmrapmdec_to_custom for transformation to pml,pmb does not work properly'
    assert numpy.fabs(pmlb_direct[1]-pmlb_custom[1]) < 10.**-8., 'pmrapmdec_to_custom for transformation to pml,pmb does not work properly'
    # Array
    s= numpy.arange(2)
    pmlb_direct= bovy_coords.pmrapmdec_to_pmllpmbb(pmra*s,pmdec*s,
                                                   ra*s,dec*s,degree=True)
    pmlb_custom= bovy_coords.pmrapmdec_to_custom(pmra*s,pmdec*s,
                                                 ra*s,dec*s,T=T.T,degree=True)
    assert numpy.all(numpy.fabs(pmlb_direct-pmlb_custom) < 10.**-8.), 'pmrapmdec_to_custom for transformation to pml,pmb does not work properly'
    _turn_on_apy()
    return None

# 02/06/2018 (JB): Edited for cases where astropy coords are always turned off
# [case at hand: einsum bug in numpy 1.14 / python2.7 astropy]
def _turn_off_apy(keep_loaded=False):
    bovy_coords._APY_COORDS_ORIG= bovy_coords._APY_COORDS
    bovy_coords._APY_COORDS= False
    if not keep_loaded:
        bovy_coords._APY_LOADED= False

def _turn_on_apy():
    bovy_coords._APY_COORDS= bovy_coords._APY_COORDS_ORIG
    bovy_coords._APY_LOADED= True
