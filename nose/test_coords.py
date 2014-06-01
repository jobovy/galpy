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
