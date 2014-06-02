import numpy
from galpy.util import bovy_coords

def test_radec_to_lb_ngp():
    # Test that the NGP is at b=90
    ra, dec= 192.25, 27.4
    lb= bovy_coords.radec_to_lb(ra,dec,degree=True,epoch=1950.)
    assert numpy.fabs(lb[1]-90.) < 10.**-8., 'Galactic latitude of the NGP given in ra,dec is not 90'
    # Also test this for degree=False
    lb= bovy_coords.radec_to_lb(ra/180.*numpy.pi,dec/180.*numpy.pi,
                                degree=False,epoch=1950.)
    assert numpy.fabs(lb[1]-numpy.pi/2.) < 10.**-8., 'Galactic latitude of the NGP given in ra,dec is not pi/2'
    return None

def test_radec_to_lb_sgp():
    # Test that the SGP is at b=90
    ra, dec= 12.25, -27.4
    lb= bovy_coords.radec_to_lb(ra,dec,degree=True,epoch=1950.)
    assert numpy.fabs(lb[1]+90.) < 10.**-8., 'Galactic latitude of the SGP given in ra,dec is not 90'
    # Also test this for degree=False
    lb= bovy_coords.radec_to_lb(ra/180.*numpy.pi,dec/180.*numpy.pi,
                                degree=False,epoch=1950.)
    assert numpy.fabs(lb[1]+numpy.pi/2.) < 10.**-8., 'Galactic latitude of the SGP given in ra,dec is not pi/2'
    return None

# Test the longitude of the north celestial pole
def test_radec_to_lb_ncp():
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
    return None

# Test that other epochs do not work
def test_radec_to_lb_otherepochs():
    ra, dec= 180., 90.
    try:
        lb= bovy_coords.radec_to_lb(ra/180.*numpy.pi,dec/180.*numpy.pi,
                                    degree=False,epoch=1975.)   
    except IOError:
        pass
    else:
        raise AssertionError('radec functions with epoch not equal to 1950 or 2000 did not raise IOError')

# Test that radec_to_lb and lb_to_radec are each other's inverse
def test_lb_to_radec():
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
    vr,pmll,pmbb= 10.,20./4.74047,-10./4.74047
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
    #Also test for arrays
    os= numpy.ones(2)
    vxvyvz= bovy_coords.vrpmllpmbb_to_vxvyvz(os*vr,os*pmll,os*pmbb,os*l,os*b,
                                             os*d,degree=True,XYZ=False)
    assert numpy.all(numpy.fabs(vxvyvz[:,0]+20.) < 10.**-10.), 'vrpmllpmbb_to_vxvyvz conversion did not work as expected'
    assert numpy.all(numpy.fabs(vxvyvz[:,1]-10.) < 10.**-10.), 'vrpmllpmbb_to_vxvyvz conversion did not work as expected'
    assert numpy.all(numpy.fabs(vxvyvz[:,2]+10.) < 10.**-10.), 'vrpmllpmbb_to_vxvyvz conversion did not work as expected'
    return None

def test_vxvyvz_to_vrpmllpmbb():
    vx,vy,vz= -20.*4.74047,10.,-10.*4.74047
    X,Y,Z= 0.,1.,0.
    vrpmllpmbb= bovy_coords.vxvyvz_to_vrpmllpmbb(vx,vy,vz,X,Y,Z,
                                                 XYZ=True)
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
    gcXYZ= bovy_coords.XYZ_to_galcenrect(X,Y,Z,Xsun=1.,Ysun=0.,Zsun=0.)
    assert numpy.fabs(gcXYZ[0]) < 10.**-10., 'XYZ_to_galcenrect conversion did not work as expected'
    assert numpy.fabs(gcXYZ[1]-3.) < 10.**-10., 'XYZ_to_galcenrect conversion did not work as expected'
    assert numpy.fabs(gcXYZ[2]+2.) < 10.**-10., 'XYZ_to_galcenrect conversion did not work as expected'
    return None
