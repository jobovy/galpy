import numpy

def test_actionAngleIsochroneApprox_otherIsochrone_actions():
    """Test the actionAngleIsochroneApprox against an isochrone potential: actions"""
    from galpy.potential import IsochronePotential
    from galpy.actionAngle import actionAngleIsochroneApprox, \
        actionAngleIsochrone
    ip= IsochronePotential(normalize=1.,b=1.2)
    aAI= actionAngleIsochrone(ip=ip)
    aAIA= actionAngleIsochroneApprox(pot=ip,b=0.8)
    R,vR,vT,z,vz,phi= 1.1, 0.3, 1.2, 0.2,0.5,2.
    ji= aAI(R,vR,vT,z,vz,phi)
    jia= aAIA(R,vR,vT,z,vz,phi)
    djr= numpy.fabs((ji[0]-jia[0])/ji[0])
    dlz= numpy.fabs((ji[1]-jia[1])/ji[1])
    djz= numpy.fabs((ji[2]-jia[2])/ji[2])
    assert djr < 10.**-2., 'actionAngleIsochroneApprox applied to isochrone potential fails for Jr at %f%%' % (djr*100.)
    #Lz and Jz are easy, because ip is a spherical potential
    assert dlz < 10.**-10., 'actionAngleIsochroneApprox applied to isochrone potential fails for Lz at %f%%' % (dlz*100.)
    assert djz < 10.**-10., 'actionAngleIsochroneApprox applied to isochrone potential fails for Jz at %f%%' % (djz*100.)
    return None

def test_actionAngleIsochroneApprox_otherIsochrone_freqs():
    """Test the actionAngleIsochroneApprox against an isochrone potential: frequencies"""
    from galpy.potential import IsochronePotential
    from galpy.actionAngle import actionAngleIsochroneApprox, \
        actionAngleIsochrone
    ip= IsochronePotential(normalize=1.,b=1.2)
    aAI= actionAngleIsochrone(ip=ip)
    aAIA= actionAngleIsochroneApprox(pot=ip,b=0.8)
    R,vR,vT,z,vz,phi= 1.1, 0.3, 1.2, 0.2,0.5,2.
    jiO= aAI.actionsFreqs(R,vR,vT,z,vz,phi)
    jiaO= aAIA.actionsFreqs(R,vR,vT,z,vz,phi)
    dOr= numpy.fabs((jiO[3]-jiaO[3])/jiO[3])
    dOp= numpy.fabs((jiO[4]-jiaO[4])/jiO[4])
    dOz= numpy.fabs((jiO[5]-jiaO[5])/jiO[5])
    assert dOr < 10.**-6., 'actionAngleIsochroneApprox applied to isochrone potential fails for Or at %f%%' % (dOr*100.)
    assert dOp < 10.**-6., 'actionAngleIsochroneApprox applied to isochrone potential fails for Op at %f%%' % (dOp*100.)
    assert dOz < 10.**-6., 'actionAngleIsochroneApprox applied to isochrone potential fails for Oz at %f%%' % (dOz*100.)
    return None

def test_actionAngleIsochroneApprox_bovy14():
    """Test the actionAngleIsochroneApprox used in Bovy (2014)"""
    from galpy.potential import LogarithmicHaloPotential
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.orbit import Orbit
    lp= LogarithmicHaloPotential(normalize=1.,q=0.9)
    aAI= actionAngleIsochroneApprox(pot=lp,b=0.8)
    obs= Orbit([1.56148083,0.35081535,-1.15481504,
                0.88719443,-0.47713334,0.12019596])
    times= numpy.linspace(0.,100.,51)
    obs.integrate(times,lp,method='dopr54_c')
    js= numpy.empty((3,len(times)))
    for ii in range(len(times)):
        tjr, tjp, tjz= aAI(obs(times[ii]))
        js[0,ii]= tjr
        js[1,ii]= tjp
        js[2,ii]= tjz
    maxdj= numpy.amax(numpy.fabs((js-numpy.tile(numpy.mean(js,axis=1),(len(times),1)).T)),axis=1)/numpy.mean(js,axis=1)
    assert maxdj[0] < 3.*10.**-2., 'Jr conservation for the GD-1 like orbit of Bovy (2014) fails at %f%%' % (100.*maxdj[0])
    assert maxdj[1] < 10.**-2., 'Lz conservation for the GD-1 like orbit of Bovy (2014) fails at %f%%' % (100.*maxdj[1])
    assert maxdj[2] < 2.*10.**-2., 'Jz conservation for the GD-1 like orbit of Bovy (2014) fails at %f%%' % (100.*maxdj[2])
    return None
