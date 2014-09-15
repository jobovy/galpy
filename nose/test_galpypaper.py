# Test that all of the examples in the galpy paper run
import numpy

def test_overview():
    from galpy.potential import NFWPotential
    np= NFWPotential(normalize=1.)
    from galpy.orbit import Orbit
    o= Orbit(vxvv=[1.,0.1,1.1,0.1,0.02,0.])
    from galpy.actionAngle import actionAngleSpherical
    aA= actionAngleSpherical(pot=np)
    js= aA(o)
    assert numpy.fabs((js[0]-0.00980542)/js[0]) < 10.**-3., 'Action calculation in the overview section has changed'
    assert numpy.fabs((js[1]-1.1)/js[0]) < 10.**-3., 'Action calculation in the overview section has changed'
    assert numpy.fabs((js[2]-0.00553155)/js[0]) < 10.**-3., 'Action calculation in the overview section has changed'
    from galpy.df import quasiisothermaldf
    qdf= quasiisothermaldf(1./3.,0.2,0.1,1.,1.,
                           pot=np,aA=aA)
    assert numpy.fabs((qdf(o)-61.57476085)/61.57476085) < 10.**-3., 'qdf calculation in the overview section has changed'
    return None
