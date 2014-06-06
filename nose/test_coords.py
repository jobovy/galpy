import numpy
from galpy.util import bovy_coords
from test_streamdf import expected_failure

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
    #Another test
    X,Y,Z= -1.,3.,-2.
    gcXYZ= bovy_coords.XYZ_to_galcenrect(X,Y,Z,Xsun=1.,Ysun=0.,Zsun=0.)
    assert numpy.fabs(gcXYZ[0]-2.) < 10.**-10., 'XYZ_to_galcenrect conversion did not work as expected'
    assert numpy.fabs(gcXYZ[1]-3.) < 10.**-10., 'XYZ_to_galcenrect conversion did not work as expected'
    assert numpy.fabs(gcXYZ[2]+2.) < 10.**-10., 'XYZ_to_galcenrect conversion did not work as expected'
    return None

def test_galcenrect_to_XYZ():
    gcX, gcY, gcZ= -1.,4.,2.
    XYZ= bovy_coords.galcenrect_to_XYZ(gcX,gcY,gcZ,Xsun=1.,Ysun=0.,Zsun=0.)
    assert numpy.fabs(XYZ[0]-2.) < 10.**-10., 'galcenrect_to_XYZ conversion did not work as expected'
    assert numpy.fabs(XYZ[1]-4.) < 10.**-10., 'galcenrect_to_XYZ conversion did not work as expected'
    assert numpy.fabs(XYZ[2]-2.) < 10.**-10., 'galcenrect_to_XYZ conversion did not work as expected'
    return None

def test_XYZ_to_galcencyl():
    X,Y,Z= 5.,4.,-2.
    gcRpZ= bovy_coords.XYZ_to_galcencyl(X,Y,Z,Xsun=8.,Ysun=0.,Zsun=0.)
    assert numpy.fabs(gcRpZ[0]-5.) < 10.**-10., 'XYZ_to_galcencyl conversion did not work as expected'
    assert numpy.fabs(gcRpZ[1]-numpy.arctan(4./3.)) < 10.**-10., 'XYZ_to_galcencyl conversion did not work as expected'
    assert numpy.fabs(gcRpZ[2]+2.) < 10.**-10., 'XYZ_to_galcencyl conversion did not work as expected'
    return None

def test_galcencyl_to_XYZ():
    gcR, gcp, gcZ= 5.,numpy.arctan(4./3.),2.
    XYZ= bovy_coords.galcencyl_to_XYZ(gcR,gcp,gcZ,Xsun=8.,Ysun=0.,Zsun=0.)
    assert numpy.fabs(XYZ[0]-5.) < 10.**-10., 'galcencyl_to_XYZ conversion did not work as expected'
    assert numpy.fabs(XYZ[1]-4.) < 10.**-10., 'galcencyl_to_XYZ conversion did not work as expected'
    assert numpy.fabs(XYZ[2]-2.) < 10.**-10., 'galcencyl_to_XYZ conversion did not work as expected'
    return None

def test_vxvyvz_to_galcenrect():
    vx,vy,vz= 10.,-20.,30
    vgc= bovy_coords.vxvyvz_to_galcenrect(vx,vy,vz,vsun=[-5.,10.,5.])
    assert numpy.fabs(vgc[0]+15.) < 10.**-10., 'vxvyvz_to_galcenrect conversion did not work as expected'
    assert numpy.fabs(vgc[1]+10.) < 10.**-10., 'vxvyvz_to_galcenrect conversion did not work as expected'
    assert numpy.fabs(vgc[2]-35.) < 10.**-10., 'vxvyvz_to_galcenrect conversion did not work as expected'
    return None

def test_vxvyvz_to_galcencyl():
    X,Y,Z= 3.,4.,2.
    vx,vy,vz= 10.,-20.,30
    vgc= bovy_coords.vxvyvz_to_galcencyl(vx,vy,vz,X,Y,Z,vsun=[-5.,10.,5.])
    assert numpy.fabs(vgc[0]+17.) < 10.**-10., 'vxvyvz_to_galcenrect conversion did not work as expected'
    assert numpy.fabs(vgc[1]-6.) < 10.**-10., 'vxvyvz_to_galcenrect conversion did not work as expected'
    assert numpy.fabs(vgc[2]-35.) < 10.**-10., 'vxvyvz_to_galcenrect conversion did not work as expected'
    return None
    
def test_sphergal_to_rectgal():
    l,b,d= 90.,0.,1.
    vr,pmll,pmbb= 10.,-20./4.74047,30./4.74047
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
