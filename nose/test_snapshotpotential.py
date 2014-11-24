################ TESTS OF THE SNAPSHOTPOTENTIAL CLASS AND RELATED #############
import numpy
import pynbody
from galpy import potential
from test_streamdf import expected_failure
def test_snapshotKeplerPotential_eval():
    # Set up a snapshot with just one unit mass at the origin
    s= pynbody.new(star=1)
    s['mass']= 1.
    s['eps']= 0.
    sp= potential.SnapshotPotential(s)
    kp= potential.KeplerPotential(amp=1.) #should be the same
    assert numpy.fabs(sp(1.,0.)-kp(1.,0.)) < 10.**-8., 'SnapshotPotential with single unit mass does not correspond to KeplerPotential'
    assert numpy.fabs(sp(0.5,0.)-kp(0.5,0.)) < 10.**-8., 'SnapshotPotential with single unit mass does not correspond to KeplerPotential'
    assert numpy.fabs(sp(1.,0.5)-kp(1.,0.5)) < 10.**-8., 'SnapshotPotential with single unit mass does not correspond to KeplerPotential'
    assert numpy.fabs(sp(1.,-0.5)-kp(1.,-0.5)) < 10.**-8., 'SnapshotPotential with single unit mass does not correspond to KeplerPotential'
    return None

def test_snapshotKeplerPotential_Rforce():
    # Set up a snapshot with just one unit mass at the origin
    s= pynbody.new(star=1)
    s['mass']= 1.
    s['eps']= 0.
    sp= potential.SnapshotPotential(s)
    kp= potential.KeplerPotential(amp=1.) #should be the same
    assert numpy.fabs(sp.Rforce(1.,0.)-kp.Rforce(1.,0.)) < 10.**-8., 'SnapshotPotential with single unit mass does not correspond to KeplerPotential'
    assert numpy.fabs(sp.Rforce(0.5,0.)-kp.Rforce(0.5,0.)) < 10.**-8., 'SnapshotPotential with single unit mass does not correspond to KeplerPotential'
    assert numpy.fabs(sp.Rforce(1.,0.5)-kp.Rforce(1.,0.5)) < 10.**-8., 'SnapshotPotential with single unit mass does not correspond to KeplerPotential'
    assert numpy.fabs(sp.Rforce(1.,-0.5)-kp.Rforce(1.,-0.5)) < 10.**-8., 'SnapshotPotential with single unit mass does not correspond to KeplerPotential'
    return None

@expected_failure
def test_snapshotKeplerPotential_zforce():
    # Set up a snapshot with just one unit mass at the origin
    s= pynbody.new(star=1)
    s['mass']= 1.
    s['eps']= 0.
    sp= potential.SnapshotPotential(s)
    kp= potential.KeplerPotential(amp=1.) #should be the same
    assert numpy.fabs(sp.zforce(1.,0.)-kp.zforce(1.,0.)) < 10.**-8., 'SnapshotPotential with single unit mass does not correspond to KeplerPotential'
    assert numpy.fabs(sp.zforce(0.5,0.)-kp.zforce(0.5,0.)) < 10.**-8., 'SnapshotPotential with single unit mass does not correspond to KeplerPotential'
    assert numpy.fabs(sp.zforce(1.,0.5)-kp.zforce(1.,0.5)) < 10.**-8., 'SnapshotPotential with single unit mass does not correspond to KeplerPotential'
    assert numpy.fabs(sp.zforce(1.,-0.5)-kp.zforce(1.,-0.5)) < 10.**-8., 'SnapshotPotential with single unit mass does not correspond to KeplerPotential'
    return None

