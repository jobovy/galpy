from __future__ import print_function, division
import os
import warnings
import numpy
from galpy.util import galpyWarning
_TRAVIS= bool(os.getenv('TRAVIS'))
# Print all galpyWarnings always for tests of warnings
warnings.simplefilter("always",galpyWarning)

#Basic sanity checking of the actionAngleIsochrone actions
def test_actionAngleIsochrone_basic_actions():
    from galpy.actionAngle import actionAngleIsochrone
    from galpy.orbit import Orbit
    aAI= actionAngleIsochrone(b=1.2)
    #circular orbit
    R,vR,vT,z,vz= 1.,0.,1.,0.,0. 
    js= aAI(R,vR,vT,z,vz)
    assert numpy.fabs(js[0]) < 10.**-16., 'Circular orbit in the isochrone potential does not have Jr=0'
    assert numpy.fabs(js[2]) < 10.**-16., 'Circular orbit in the isochrone potential does not have Jz=0'
    #Close-to-circular orbit
    R,vR,vT,z,vz= 1.01,0.01,1.,0.01,0.01 
    js= aAI(Orbit([R,vR,vT,z,vz]))
    assert numpy.fabs(js[0]) < 10.**-4., 'Close-to-circular orbit in the isochrone potential does not have small Jr'
    assert numpy.fabs(js[2]) < 10.**-4., 'Close-to-circular orbit in the isochrone potential does not have small Jz'
    #Close-to-circular orbit, called with time
    R,vR,vT,z,vz= 1.01,0.01,1.,0.01,0.01 
    js= aAI(Orbit([R,vR,vT,z,vz]),0.)
    assert numpy.fabs(js[0]) < 10.**-4., 'Close-to-circular orbit in the isochrone potential does not have small Jr'
    assert numpy.fabs(js[2]) < 10.**-4., 'Close-to-circular orbit in the isochrone potential does not have small Jz'
    return None

#Basic sanity checking of the actionAngleIsochrone actions
def test_actionAngleIsochrone_basic_freqs():
    from galpy.potential import IsochronePotential
    from galpy.actionAngle import actionAngleIsochrone
    from galpy.orbit import Orbit
    ip= IsochronePotential(normalize=1.,b=1.2)
    aAI= actionAngleIsochrone(ip=ip)
    #circular orbit
    R,vR,vT,z,vz= 1.,0.,1.,0.,0. 
    jos= aAI.actionsFreqs(R,vR,vT,z,vz)
    assert numpy.fabs((jos[3]-ip.epifreq(1.))/ip.epifreq(1.)) < 10.**-12., 'Circular orbit in the isochrone potential does not have Or=kappa at %g%%' % (100.*numpy.fabs((jos[3]-ip.epifreq(1.))/ip.epifreq(1.)))
    assert numpy.fabs((jos[4]-ip.omegac(1.))/ip.omegac(1.)) < 10.**-12., 'Circular orbit in the isochrone potential does not have Op=Omega at %g%%' % (100.*numpy.fabs((jos[4]-ip.omegac(1.))/ip.omegac(1.)))
    assert numpy.fabs((jos[5]-ip.verticalfreq(1.))/ip.verticalfreq(1.)) < 10.**-12., 'Circular orbit in the isochrone potential does not have Oz=nu at %g%%' % (100.*numpy.fabs((jos[5]-ip.verticalfreq(1.))/ip.verticalfreq(1.)))
    #close-to-circular orbit
    R,vR,vT,z,vz= 1.,0.01,1.01,0.01,0.01 
    jos= aAI.actionsFreqs(Orbit([R,vR,vT,z,vz]))
    assert numpy.fabs((jos[3]-ip.epifreq(1.))/ip.epifreq(1.)) < 10.**-2., 'Close-to-circular orbit in the isochrone potential does not have Or=kappa at %g%%' % (100.*numpy.fabs((jos[3]-ip.epifreq(1.))/ip.epifreq(1.)))
    assert numpy.fabs((jos[4]-ip.omegac(1.))/ip.omegac(1.)) < 10.**-2., 'Close-to-circular orbit in the isochrone potential does not have Op=Omega at %g%%' % (100.*numpy.fabs((jos[4]-ip.omegac(1.))/ip.omegac(1.)))
    assert numpy.fabs((jos[5]-ip.verticalfreq(1.))/ip.verticalfreq(1.)) < 10.**-2., 'Close-to-circular orbit in the isochrone potential does not have Oz=nu at %g%%' % (100.*numpy.fabs((jos[5]-ip.verticalfreq(1.))/ip.verticalfreq(1.)))
    return None

#Test the actions of an actionAngleIsochrone
def test_actionAngleIsochrone_conserved_actions():
    from galpy.potential import IsochronePotential
    from galpy.actionAngle import actionAngleIsochrone
    from galpy.orbit import Orbit
    ip= IsochronePotential(normalize=1.,b=1.2)
    aAI= actionAngleIsochrone(ip=ip)
    obs= Orbit([1.1, 0.3, 1.2, 0.2,0.5])
    from galpy.orbit_src.FullOrbit import ext_loaded
    if not ext_loaded: #odeint is not as accurate as dopr54_c
        check_actionAngle_conserved_actions(aAI,obs,ip,-5.,-5.,-5.)
    else:
        check_actionAngle_conserved_actions(aAI,obs,ip,-8.,-8.,-8.)
    return None

#Test that the angles of an actionAngleIsochrone increase linearly
def test_actionAngleIsochrone_linear_angles():
    from galpy.potential import IsochronePotential
    from galpy.actionAngle import actionAngleIsochrone
    from galpy.orbit import Orbit
    ip= IsochronePotential(normalize=1.,b=1.2)
    aAI= actionAngleIsochrone(ip=ip)
    obs= Orbit([1.1, 0.3, 1.2, 0.2,0.5,2.])
    from galpy.orbit_src.FullOrbit import ext_loaded
    if not ext_loaded: #odeint is not as accurate as dopr54_c
        check_actionAngle_linear_angles(aAI,obs,ip,
                                        -5.,-5.,-5.,
                                        -6.,-6.,-6.,
                                        -5.,-5.,-5.)
    else:
        check_actionAngle_linear_angles(aAI,obs,ip,
                                        -6.,-6.,-6.,
                                        -8.,-8.,-8.,
                                        -8.,-8.,-8.)
    return None

#Test that the Kelperian limit of the isochrone actions/angles works
def test_actionAngleIsochrone_kepler_actions():
    from galpy.potential import IsochronePotential
    from galpy.actionAngle import actionAngleIsochrone
    from galpy.orbit import Orbit
    ip= IsochronePotential(normalize=1.,b=0.)
    aAI= actionAngleIsochrone(ip=ip)
    obs= Orbit([1.1, 0.3, 1.2, 0.2,0.5,2.])
    times= numpy.linspace(0.,100.,101)
    obs.integrate(times,ip,method='dopr54_c')
    jrs,jps,jzs= aAI(obs.R(times),obs.vR(times),obs.vT(times),
                     obs.z(times),obs.vz(times),obs.phi(times))
    jc= ip._amp/numpy.sqrt(-2.*obs.E())
    L= numpy.sqrt(numpy.sum(obs.L()**2.))
    # Jr = Jc-L
    assert numpy.all(numpy.fabs(jrs-(jc-L)) < 10.**-5.), 'Radial action for the Kepler potential not correct'
    assert numpy.all(numpy.fabs(jps-obs.R()*obs.vT()) < 10.**-10.), 'Azimuthal action for the Kepler potential not correct'
    assert numpy.all(numpy.fabs(jzs-(L-numpy.fabs(obs.R()*obs.vT()))) < 10.**-10.), 'Vertical action for the Kepler potential not correct'
    return None

def test_actionAngleIsochrone_kepler_freqs():
    from galpy.potential import IsochronePotential
    from galpy.actionAngle import actionAngleIsochrone
    from galpy.orbit import Orbit
    ip= IsochronePotential(normalize=1.,b=0.)
    aAI= actionAngleIsochrone(ip=ip)
    obs= Orbit([1.1, 0.3, 1.2, 0.2,0.5,2.])
    times= numpy.linspace(0.,100.,101)
    obs.integrate(times,ip,method='dopr54_c')
    _, _, _, ors,ops,ozs= aAI.actionsFreqs(obs.R(times),obs.vR(times),
                                           obs.vT(times),obs.z(times),
                                           obs.vz(times),obs.phi(times))
    jc= ip._amp/numpy.sqrt(-2.*obs.E())
    oc= ip._amp**2./jc**3. # (BT08 eqn. E4)
    assert numpy.all(numpy.fabs(ors-oc) < 10.**-10.), 'Radial frequency for the Kepler potential not correct'
    assert numpy.all(numpy.fabs(ops-oc) < 10.**-10.), 'Azimuthal frequency for the Kepler potential not correct'
    assert numpy.all(numpy.fabs(ozs-numpy.sign(obs.R()*obs.vT())*oc) < 10.**-10.), 'Vertical frequency for the Kepler potential not correct'
    return None

def test_actionAngleIsochrone_kepler_angles():
    from galpy.potential import IsochronePotential
    from galpy.actionAngle import actionAngleIsochrone
    from galpy.orbit import Orbit
    ip= IsochronePotential(normalize=1.,b=0.)
    aAI= actionAngleIsochrone(ip=ip)
    obs= Orbit([1.1, 0.3, 1.2, 0.2,0.5,2.])
    times= numpy.linspace(0.,100.,101)
    obs.integrate(times,ip,method='dopr54_c')
    _, _, _, _, _, _,ars,aps,azs= \
        aAI.actionsFreqsAngles(obs.R(times),obs.vR(times),
                               obs.vT(times),obs.z(times),
                               obs.vz(times),obs.phi(times))
    jc= ip._amp/numpy.sqrt(-2.*obs.E())
    oc= ip._amp**2./jc**3. # (BT08 eqn. E4)
    # theta_r = Or x times + theta_r,0
    assert numpy.all(numpy.fabs(ars-oc*times-ars[0]) < 10.**-10.), 'Radial angle for the Kepler potential not correct'
    assert numpy.all(numpy.fabs(aps-oc*times-aps[0]) < 10.**-10.), 'Azimuthal angle for the Kepler potential not correct'
    assert numpy.all(numpy.fabs(azs-oc*times-azs[0]) < 10.**-10.), 'Vertical angle for the Kepler potential not correct'
    return None

#Basic sanity checking of the actionAngleSpherical actions
def test_actionAngleSpherical_basic_actions():
    from galpy.actionAngle import actionAngleSpherical
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    lp= LogarithmicHaloPotential(normalize=1.,q=1.)
    aAS= actionAngleSpherical(pot=lp)
    #circular orbit
    R,vR,vT,z,vz= 1.,0.,1.,0.,0. 
    js= aAS(Orbit([R,vR,vT]))
    assert numpy.fabs(js[0]) < 10.**-16., 'Circular orbit in the spherical LogarithmicHaloPotential does not have Jr=0'
    assert numpy.fabs(js[2]) < 10.**-16., 'Circular orbit in the spherical LogarithmicHaloPotential does not have Jz=0'
    #Close-to-circular orbit
    R,vR,vT,z,vz= 1.01,0.01,1.,0.01,0.01 
    js= aAS(Orbit([R,vR,vT,z,vz])._orb) #with OrbitTop
    assert numpy.fabs(js[0]) < 10.**-4., 'Close-to-circular orbit in the spherical LogarithmicHaloPotential does not have small Jr'
    assert numpy.fabs(js[2]) < 10.**-4., 'Close-to-circular orbit in the spherical LogarithmicHaloPotential does not have small Jz'
    return None

#Basic sanity checking of the actionAngleSpherical actions
def test_actionAngleSpherical_basic_freqs():
    from galpy.potential import LogarithmicHaloPotential
    from galpy.actionAngle import actionAngleSpherical
    from galpy.orbit import Orbit
    lp= LogarithmicHaloPotential(normalize=1.,q=1.)
    aAS= actionAngleSpherical(pot=[lp])
    #circular orbit
    R,vR,vT,z,vz= 1.,0.,1.,0.,0. 
    jos= aAS.actionsFreqs(R,vR,vT,z,vz)
    assert numpy.fabs((jos[3]-lp.epifreq(1.))/lp.epifreq(1.)) < 10.**-12., 'Circular orbit in the spherical LogarithmicHaloPotential does not have Or=kappa at %g%%' % (100.*numpy.fabs((jos[3]-lp.epifreq(1.))/lp.epifreq(1.)))
    assert numpy.fabs((jos[4]-lp.omegac(1.))/lp.omegac(1.)) < 10.**-12., 'Circular orbit in the spherical LogarithmicHaloPotential does not have Op=Omega at %g%%' % (100.*numpy.fabs((jos[4]-lp.omegac(1.))/lp.omegac(1.)))
    assert numpy.fabs((jos[5]-lp.verticalfreq(1.))/lp.verticalfreq(1.)) < 10.**-12., 'Circular orbit in the spherical LogarithmicHaloPotential does not have Oz=nu at %g%%' % (100.*numpy.fabs((jos[5]-lp.verticalfreq(1.))/lp.verticalfreq(1.)))
    #close-to-circular orbit
    R,vR,vT,z,vz= 1.,0.01,1.01,0.01,0.01 
    jos= aAS.actionsFreqs(Orbit([R,vR,vT,z,vz]))
    assert numpy.fabs((jos[3]-lp.epifreq(1.))/lp.epifreq(1.)) < 10.**-1.9, 'Close-to-circular orbit in the spherical LogarithmicHaloPotential does not have Or=kappa at %g%%' % (100.*numpy.fabs((jos[3]-lp.epifreq(1.))/lp.epifreq(1.)))
    assert numpy.fabs((jos[4]-lp.omegac(1.))/lp.omegac(1.)) < 10.**-1.9, 'Close-to-circular orbit in the spherical LogarithmicHaloPotential does not have Op=Omega at %g%%' % (100.*numpy.fabs((jos[4]-lp.omegac(1.))/lp.omegac(1.)))
    assert numpy.fabs((jos[5]-lp.verticalfreq(1.))/lp.verticalfreq(1.)) < 10.**-1.9, 'Close-to-circular orbit in the spherical LogarithmicHaloPotential does not have Oz=nu at %g%%' % (100.*numpy.fabs((jos[5]-lp.verticalfreq(1.))/lp.verticalfreq(1.)))

#Basic sanity checking of the actionAngleSpherical actions
def test_actionAngleSpherical_basic_freqsAngles():
    from galpy.potential import LogarithmicHaloPotential
    from galpy.actionAngle import actionAngleSpherical
    from galpy.orbit import Orbit
    lp= LogarithmicHaloPotential(normalize=1.,q=1.)
    aAS= actionAngleSpherical(pot=lp)
    #v. close-to-circular orbit using actionsFreqsAngles
    R,vR,vT,z,vz= 1.,10.**-8.,1.,10.**-8.,0.
    jos= aAS.actionsFreqsAngles(R,vR,vT,z,vz,0.)
    assert numpy.fabs((jos[3]-lp.epifreq(1.))/lp.epifreq(1.)) < 10.**-1.9, 'Close-to-circular orbit in the spherical LogarithmicHaloPotential does not have Or=kappa at %g%%' % (100.*numpy.fabs((jos[3]-lp.epifreq(1.))/lp.epifreq(1.)))
    assert numpy.fabs((jos[4]-lp.omegac(1.))/lp.omegac(1.)) < 10.**-1.9, 'Close-to-circular orbit in the spherical LogarithmicHaloPotential does not have Op=Omega at %g%%' % (100.*numpy.fabs((jos[4]-lp.omegac(1.))/lp.omegac(1.)))
    assert numpy.fabs((jos[5]-lp.verticalfreq(1.))/lp.verticalfreq(1.)) < 10.**-1.9, 'Close-to-circular orbit in the spherical LogarithmicHaloPotential does not have Oz=nu at %g%%' % (100.*numpy.fabs((jos[5]-lp.verticalfreq(1.))/lp.verticalfreq(1.)))
    return None

#Test the actions of an actionAngleSpherical
def test_actionAngleSpherical_conserved_actions():
    from galpy import potential
    from galpy.actionAngle import actionAngleSpherical
    from galpy.orbit import Orbit
    lp= potential.LogarithmicHaloPotential(normalize=1.,q=1.)
    aAS= actionAngleSpherical(pot=lp)
    obs= Orbit([1.1, 0.3, 1.2, 0.2,0.5])
    from galpy.orbit_src.FullOrbit import ext_loaded
    if not ext_loaded: #odeint is not as accurate as dopr54_c
        check_actionAngle_conserved_actions(aAS,obs,lp,-5.,-5.,-5.,ntimes=101)
    else:
        check_actionAngle_conserved_actions(aAS,obs,lp,-8.,-8.,-8.,ntimes=101)
    return None

#Test the actions of an actionAngleSpherical
def test_actionAngleSpherical_conserved_actions_fixed_quad():
    from galpy.potential import LogarithmicHaloPotential
    from galpy.actionAngle import actionAngleSpherical
    from galpy.orbit import Orbit
    lp= LogarithmicHaloPotential(normalize=1.,q=1.)
    aAS= actionAngleSpherical(pot=lp)
    obs= Orbit([1.1, 0.3, 1.2, 0.2,0.5])
    from galpy.orbit_src.FullOrbit import ext_loaded
    if not ext_loaded: #odeint is not as accurate as dopr54_c
        check_actionAngle_conserved_actions(aAS,obs,lp,-5.,-5.,-5.,ntimes=101,
                                            fixed_quad=True)
    else:
        check_actionAngle_conserved_actions(aAS,obs,lp,-8.,-8.,-8.,ntimes=101,
                                            fixed_quad=True)
    return None

#Test that the angles of an actionAngleIsochrone increase linearly
def test_actionAngleSpherical_linear_angles():
    from galpy.potential import LogarithmicHaloPotential
    from galpy.actionAngle import actionAngleSpherical
    from galpy.orbit import Orbit
    lp= LogarithmicHaloPotential(normalize=1.,q=1.)
    aAS= actionAngleSpherical(pot=lp)
    obs= Orbit([1.1, 0.3, 1.2, 0.2,0.5,2.])
    from galpy.orbit_src.FullOrbit import ext_loaded
    if not ext_loaded: #odeint is not as accurate as dopr54_c
        check_actionAngle_linear_angles(aAS,obs,lp,
                                        -4.,-4.,-4.,
                                        -4.,-4.,-4.,
                                        -4.,-4.,-4.,
                                        ntimes=501) #need fine sampling for de-period
    else:
        check_actionAngle_linear_angles(aAS,obs,lp,
                                        -6.,-6.,-6.,
                                        -8.,-8.,-8.,
                                        -8.,-8.,-8.,
                                        ntimes=501) #need fine sampling for de-period
    return None
  
#Test that the angles of an actionAngleIsochrone increase linearly
def test_actionAngleSpherical_linear_angles_fixed_quad():
    from galpy.potential import LogarithmicHaloPotential
    from galpy.actionAngle import actionAngleSpherical
    from galpy.orbit import Orbit
    lp= LogarithmicHaloPotential(normalize=1.,q=1.)
    aAS= actionAngleSpherical(pot=lp)
    obs= Orbit([1.1, 0.3, 1.2, 0.2,0.5,2.])
    from galpy.orbit_src.FullOrbit import ext_loaded
    if not ext_loaded: #odeint is not as accurate as dopr54_c
        check_actionAngle_linear_angles(aAS,obs,lp,
                                        -4.,-4.,-4.,
                                        -4.,-4.,-4.,
                                        -4.,-4.,-4.,
                                        ntimes=501, #need fine sampling for de-period
                                        fixed_quad=True)
    else:
        check_actionAngle_linear_angles(aAS,obs,lp,
                                        -6.,-6.,-6.,
                                        -8.,-8.,-8.,
                                        -8.,-8.,-8.,
                                        ntimes=501, #need fine sampling for de-period
                                        fixed_quad=True)
    return None
  
#Test the actionAngleSpherical against an isochrone potential: actions
def test_actionAngleSpherical_otherIsochrone_actions():
    from galpy.potential import IsochronePotential
    from galpy.actionAngle import actionAngleSpherical, \
        actionAngleIsochrone
    ip= IsochronePotential(normalize=1.,b=1.2)
    aAI= actionAngleIsochrone(ip=ip)
    aAS= actionAngleSpherical(pot=ip)
    R,vR,vT,z,vz,phi= 1.1, 0.3, 1.2, 0.2,0.5,2.
    ji= aAI(R,vR,vT,z,vz,phi)
    jia= aAS(R,vR,vT,z,vz,phi)
    djr= numpy.fabs((ji[0]-jia[0])/ji[0])
    dlz= numpy.fabs((ji[1]-jia[1])/ji[1])
    djz= numpy.fabs((ji[2]-jia[2])/ji[2])
    assert djr < 10.**-10., 'actionAngleSpherical applied to isochrone potential fails for Jr at %g%%' % (djr*100.)
    #Lz and Jz are easy, because ip is a spherical potential
    assert dlz < 10.**-10., 'actionAngleSpherical applied to isochrone potential fails for Lz at %g%%' % (dlz*100.)
    assert djz < 10.**-10., 'actionAngleSpherical applied to isochrone potential fails for Jz at %g%%' % (djz*100.)
    return None

#Test the actionAngleSpherical against an isochrone potential: frequencies
def test_actionAngleSpherical_otherIsochrone_freqs():   
    from galpy.potential import IsochronePotential
    from galpy.actionAngle import actionAngleSpherical, \
        actionAngleIsochrone
    ip= IsochronePotential(normalize=1.,b=1.2)
    aAI= actionAngleIsochrone(ip=ip)
    aAS= actionAngleSpherical(pot=ip)
    R,vR,vT,z,vz,phi= 1.1, 0.3, 1.2, 0.2,0.5,2.
    jiO= aAI.actionsFreqs(R,vR,vT,z,vz,phi)
    jiaO= aAS.actionsFreqs(R,vR,vT,z,vz,phi)
    dOr= numpy.fabs((jiO[3]-jiaO[3])/jiO[3])
    dOp= numpy.fabs((jiO[4]-jiaO[4])/jiO[4])
    dOz= numpy.fabs((jiO[5]-jiaO[5])/jiO[5])
    assert dOr < 10.**-6., 'actionAngleSpherical applied to isochrone potential fails for Or at %g%%' % (dOr*100.)
    assert dOp < 10.**-6., 'actionAngleSpherical applied to isochrone potential fails for Op at %g%%' % (dOp*100.)
    assert dOz < 10.**-6., 'actionAngleSpherical applied to isochrone potential fails for Oz at %g%%' % (dOz*100.)
    return None

#Test the actionAngleSpherical against an isochrone potential: frequencies
def test_actionAngleSpherical_otherIsochrone_freqs_fixed_quad():   
    from galpy.potential import IsochronePotential
    from galpy.actionAngle import actionAngleSpherical, \
        actionAngleIsochrone
    from galpy.orbit import Orbit
    ip= IsochronePotential(normalize=1.,b=1.2)
    aAI= actionAngleIsochrone(ip=ip)
    aAS= actionAngleSpherical(pot=ip)
    R,vR,vT,z,vz,phi= 1.1, 0.3, 1.2, 0.2,0.5,2.
    jiO= aAI.actionsFreqs(R,vR,vT,z,vz,phi)
    jiaO= aAS.actionsFreqs(Orbit([R,vR,vT,z,vz,phi]),fixed_quad=True)
    dOr= numpy.fabs((jiO[3]-jiaO[3])/jiO[3])
    dOp= numpy.fabs((jiO[4]-jiaO[4])/jiO[4])
    dOz= numpy.fabs((jiO[5]-jiaO[5])/jiO[5])
    assert dOr < 10.**-6., 'actionAngleSpherical applied to isochrone potential fails for Or at %g%%' % (dOr*100.)
    assert dOp < 10.**-6., 'actionAngleSpherical applied to isochrone potential fails for Op at %g%%' % (dOp*100.)
    assert dOz < 10.**-6., 'actionAngleSpherical applied to isochrone potential fails for Oz at %g%%' % (dOz*100.)
    return None

#Test the actionAngleSpherical against an isochrone potential: angles
def test_actionAngleSpherical_otherIsochrone_angles():   
    from galpy.potential import IsochronePotential
    from galpy.actionAngle import actionAngleSpherical, \
        actionAngleIsochrone
    ip= IsochronePotential(normalize=1.,b=1.2)
    aAI= actionAngleIsochrone(ip=ip)
    aAS= actionAngleSpherical(pot=ip,b=0.8)
    R,vR,vT,z,vz,phi= 1.1, 0.3, 1.2, 0.2,0.5,2.
    jiO= aAI.actionsFreqsAngles(R,vR,vT,z,vz,phi)
    jiaO= aAS.actionsFreqsAngles(R,vR,vT,z,vz,phi)
    dar= numpy.fabs((jiO[6]-jiaO[6])/jiO[6])
    dap= numpy.fabs((jiO[7]-jiaO[7])/jiO[7])
    daz= numpy.fabs((jiO[8]-jiaO[8])/jiO[8])
    assert dar < 10.**-6., 'actionAngleSpherical applied to isochrone potential fails for ar at %g%%' % (dar*100.)
    assert dap < 10.**-6., 'actionAngleSpherical applied to isochrone potential fails for ap at %g%%' % (dap*100.)
    assert daz < 10.**-6., 'actionAngleSpherical applied to isochrone potential fails for az at %g%%' % (daz*100.)
    return None

#Basic sanity checking of the actionAngleAdiabatic actions
def test_actionAngleAdiabatic_basic_actions():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential
    aAA= actionAngleAdiabatic(pot=MWPotential,gamma=1.)
    #circular orbit
    R,vR,vT,phi= 1.,0.,1.,2. 
    js= aAA(Orbit([R,vR,vT,phi]))
    assert numpy.fabs(js[0]) < 10.**-16., 'Circular orbit in the MWPotential does not have Jr=0'
    assert numpy.fabs(js[2]) < 10.**-16., 'Circular orbit in the MWPotential does not have Jz=0'
    #Close-to-circular orbit
    R,vR,vT,z,vz= 1.01,0.01,1.,0.01,0.01 
    js= aAA(Orbit([R,vR,vT,z,vz]))
    assert numpy.fabs(js[0]) < 10.**-4., 'Close-to-circular orbit in the MWPotential does not have small Jr'
    assert numpy.fabs(js[2]) < 10.**-3., 'Close-to-circular orbit in the MWPotentialspherical LogarithmicHalo does not have small Jz'
    #Another close-to-circular orbit
    R,vR,vT,z,vz= 1.0,0.0,0.99,0.0,0.0
    js= aAA(Orbit([R,vR,vT,z,vz]))
    assert numpy.fabs(js[0]) < 10.**-4., 'Close-to-circular orbit in the MWPotential does not have small Jr'
    assert numpy.fabs(js[2]) < 10.**-3., 'Close-to-circular orbit in the MWPotentialspherical LogarithmicHalo does not have small Jz'
    #Another close-to-circular orbit
    R,vR,vT,z,vz= 1.0,0.0,1.01,0.0,0.0
    js= aAA(Orbit([R,vR,vT,z,vz]))
    assert numpy.fabs(js[0]) < 10.**-4., 'Close-to-circular orbit in the MWPotential does not have small Jr'
    assert numpy.fabs(js[2]) < 10.**-3., 'Close-to-circular orbit in the MWPotentialspherical LogarithmicHalo does not have small Jz'
    return None

#Basic sanity checking of the actionAngleAdiabatic actions
def test_actionAngleAdiabatic_basic_actions_gamma0():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential
    aAA= actionAngleAdiabatic(pot=MWPotential,gamma=0.)
    #circular orbit
    R,vR,vT,phi= 1.,0.,1.,2. 
    js= aAA(Orbit([R,vR,vT,phi]))
    assert numpy.fabs(js[0]) < 10.**-16., 'Circular orbit in the MWPotential does not have Jr=0'
    assert numpy.fabs(js[2]) < 10.**-16., 'Circular orbit in the MWPotential does not have Jz=0'
    #Close-to-circular orbit
    R,vR,vT,z,vz= 1.01,0.01,1.,0.01,0.01 
    js= aAA(Orbit([R,vR,vT,z,vz]))
    assert numpy.fabs(js[0]) < 10.**-4., 'Close-to-circular orbit in the MWPotential does not have small Jr'
    assert numpy.fabs(js[2]) < 10.**-3., 'Close-to-circular orbit in the MWPotentialspherical LogarithmicHalo does not have small Jz'
    #Another close-to-circular orbit
    R,vR,vT,z,vz= 1.0,0.0,0.99,0.0,0.0
    js= aAA(Orbit([R,vR,vT,z,vz]))
    assert numpy.fabs(js[0]) < 10.**-4., 'Close-to-circular orbit in the MWPotential does not have small Jr'
    assert numpy.fabs(js[2]) < 10.**-3., 'Close-to-circular orbit in the MWPotentialspherical LogarithmicHalo does not have small Jz'
    #Another close-to-circular orbit
    R,vR,vT,z,vz= 1.0,0.0,1.01,0.0,0.0
    js= aAA(Orbit([R,vR,vT,z,vz]))
    assert numpy.fabs(js[0]) < 10.**-4., 'Close-to-circular orbit in the MWPotential does not have small Jr'
    assert numpy.fabs(js[2]) < 10.**-3., 'Close-to-circular orbit in the MWPotentialspherical LogarithmicHalo does not have small Jz'
    return None

#Basic sanity checking of the actionAngleAdiabatic actions
def test_actionAngleAdiabatic_basic_actions_c():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential
    aAA= actionAngleAdiabatic(pot=MWPotential,c=True)
    #circular orbit
    R,vR,vT,z,vz= 1.,0.,1.,0.,0. 
    js= aAA(R,vR,vT,z,vz)
    assert numpy.fabs(js[0]) < 10.**-16., 'Circular orbit in the MWPotential does not have Jr=0'
    assert numpy.fabs(js[2]) < 10.**-16., 'Circular orbit in the MWPotential does not have Jz=0'
    #Close-to-circular orbit
    R,vR,vT,z,vz= 1.01,0.01,1.,0.01,0.01 
    js= aAA(Orbit([R,vR,vT,z,vz]))
    assert numpy.fabs(js[0]) < 10.**-4., 'Close-to-circular orbit in the MWPotential does not have small Jr'
    assert numpy.fabs(js[2]) < 10.**-3., 'Close-to-circular orbit in the MWPotentialspherical LogarithmicHalo does not have small Jz'

#Basic sanity checking of the actionAngleAdiabatic actions
def test_actionAngleAdiabatic_unboundz_actions_c():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.potential import MWPotential
    aAA= actionAngleAdiabatic(pot=MWPotential,c=True,gamma=0.)
    #Unbound in z, so jz should be very large
    R,vR,vT,z,vz= 1.,0.,1.,0., 10.
    js= aAA(R,vR,vT,z,vz)
    assert js[2] > 1000., 'Unbound orbit in z in the MWPotential does not have large Jz'
    return None

#Basic sanity checking of the actionAngleAdiabatic actions
def test_actionAngleAdiabatic_zerolz_actions_c():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.potential import MWPotential
    aAA= actionAngleAdiabatic(pot=MWPotential,c=True,gamma=0.)
    #Zero angular momentum, so rperi=0, but should have finite jr
    R,vR,vT,z,vz= 1.,0.,0.,0., 0.
    js= aAA(R,vR,vT,z,vz)
    R,vR,vT,z,vz= 1.,0.,0.0000001,0., 0.
    js2= aAA(R,vR,vT,z,vz)
    assert numpy.fabs(js[0]-js2[0]) < 10.**-6., 'Orbit with zero angular momentum does not have the correct Jr'
    #Zero angular momentum, so rperi=0, but should have finite jr
    R,vR,vT,z,vz= 1.,-0.5,0.,0., 0.
    js= aAA(R,vR,vT,z,vz)
    R,vR,vT,z,vz= 1.,-0.5,0.0000001,0., 0.
    js2= aAA(R,vR,vT,z,vz)
    assert numpy.fabs(js[0]-js2[0]) < 10.**-6., 'Orbit with zero angular momentum does not have the correct Jr'
    return None

#Test the actions of an actionAngleAdiabatic
def test_actionAngleAdiabatic_conserved_actions():
    from galpy.potential import MWPotential
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.orbit import Orbit
    aAA= actionAngleAdiabatic(pot=MWPotential,c=False)
    obs= Orbit([1.05, 0.02, 1.05, 0.03,0.])
    check_actionAngle_conserved_actions(aAA,obs,MWPotential,
                                        -1.2,-8.,-1.7,ntimes=101)
    return None

#Test the actions of an actionAngleAdiabatic
def test_actionAngleAdiabatic_conserved_actions_c():
    from galpy.potential import MWPotential
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.orbit import Orbit
    obs= Orbit([1.05, 0.02, 1.05, 0.03,0.])
    aAA= actionAngleAdiabatic(pot=MWPotential,c=True)
    check_actionAngle_conserved_actions(aAA,obs,MWPotential,
                                        -1.4,-8.,-1.7,ntimes=101)
    return None

#Test the actions of an actionAngleAdiabatic, single pot
def test_actionAngleAdiabatic_conserved_actions_singlepot():
    from galpy.potential import MiyamotoNagaiPotential
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.orbit import Orbit
    mp= MiyamotoNagaiPotential(normalize=1.)
    obs= Orbit([1.05, 0.02, 1.05, 0.03,0.,2.])
    aAA= actionAngleAdiabatic(pot=mp,c=False)
    check_actionAngle_conserved_actions(aAA,obs,mp,
                                        -1.5,-8.,-2.,ntimes=101,
                                        inclphi=True)
    return None

#Test the actions of an actionAngleAdiabatic, single pot, C
def test_actionAngleAdiabatic_conserved_actions_singlepot_c():
    from galpy.potential import MiyamotoNagaiPotential
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.orbit import Orbit
    mp= MiyamotoNagaiPotential(normalize=1.)
    obs= Orbit([1.05, 0.02, 1.05, 0.03,0.,2.])
    aAA= actionAngleAdiabatic(pot=mp,c=True)
    check_actionAngle_conserved_actions(aAA,obs,mp,
                                        -1.5,-8.,-2.,ntimes=101,
                                        inclphi=True)
    return None

#Test the actions of an actionAngleAdiabatic, interpolated pot
def test_actionAngleAdiabatic_conserved_actions_interppot_c():
    from galpy.potential import MWPotential, interpRZPotential
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.orbit import Orbit
    ip= interpRZPotential(RZPot=MWPotential,
                          rgrid=(numpy.log(0.01),numpy.log(20.),101),
                          zgrid=(0.,1.,101),logR=True,use_c=True,enable_c=True,
                          interpPot=True,interpRforce=True,interpzforce=True)
    obs= Orbit([1.05, 0.02, 1.05, 0.03,0.,2.])
    aAA= actionAngleAdiabatic(pot=ip,c=True)
    check_actionAngle_conserved_actions(aAA,obs,ip,
                                        -1.4,-8.,-1.7,ntimes=101)
    return None

#Test the actionAngleAdiabatic against an isochrone potential: actions
def test_actionAngleAdiabatic_Isochrone_actions():
    from galpy.potential import IsochronePotential
    from galpy.actionAngle import actionAngleAdiabatic, \
        actionAngleIsochrone
    ip= IsochronePotential(normalize=1.,b=1.2)
    aAI= actionAngleIsochrone(ip=ip)
    aAA= actionAngleAdiabatic(pot=ip,c=True)
    R,vR,vT,z,vz,phi= 1.01, 0.05, 1.05, 0.05,0.,2.
    ji= aAI(R,vR,vT,z,vz,phi)
    jia= aAA(R,vR,vT,z,vz,phi)
    djr= numpy.fabs((ji[0]-jia[0])/ji[0])
    dlz= numpy.fabs((ji[1]-jia[1])/ji[1])
    djz= numpy.fabs((ji[2]-jia[2])/ji[2])
    assert djr < 10.**-1.2, 'actionAngleAdiabatic applied to isochrone potential fails for Jr at %f%%' % (djr*100.)
    #Lz and Jz are easy, because ip is a spherical potential
    assert dlz < 10.**-10., 'actionAngleAdiabatic applied to isochrone potential fails for Lz at %f%%' % (dlz*100.)
    assert djz < 10.**-1.2, 'actionAngleAdiabatic applied to isochrone potential fails for Jz at %f%%' % (djz*100.)
    return None

#Basic sanity checking of the actionAngleAdiabatic actions (incl. conserved, bc takes a lot of time)
def test_actionAngleAdiabaticGrid_basicAndConserved_actions():
    from galpy.actionAngle import actionAngleAdiabaticGrid
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential
    aAA= actionAngleAdiabaticGrid(pot=MWPotential,gamma=1.,c=False)
    #circular orbit
    R,vR,vT,z,vz= 1.,0.,1.,0.,0. 
    js= aAA(R,vR,vT,z,vz,0.)
    assert numpy.fabs(js[0]) < 10.**-16., 'Circular orbit in the MWPotential does not have Jr=0'
    assert numpy.fabs(aAA.Jz(R,vR,vT,z,vz,0.)) < 10.**-16., 'Circular orbit in the MWPotential does not have Jz=0'
    #setup w/ multi
    aAA= actionAngleAdiabaticGrid(pot=MWPotential,gamma=1.,c=False,numcores=2)
    #Close-to-circular orbit
    R,vR,vT,z,vz= 1.01,0.01,1.,0.01,0.01 
    js= aAA(Orbit([R,vR,vT,z,vz]))
    assert numpy.fabs(js[0]) < 10.**-4., 'Close-to-circular orbit in the MWPotential does not have small Jr'
    assert numpy.fabs(js[2]) < 10.**-3., 'Close-to-circular orbit in the MWPotentialspherical LogarithmicHalo does not have small Jz'
    #Check that actions are conserved along the orbit
    obs= Orbit([1.05, 0.02, 1.05, 0.03,0.])
    check_actionAngle_conserved_actions(aAA,obs,MWPotential,
                                        -1.2,-8.,-1.7,ntimes=101)
    return None

#Basic sanity checking of the actionAngleAdiabatic actions
def test_actionAngleAdiabaticGrid_basic_actions_c():
    from galpy.actionAngle import actionAngleAdiabaticGrid
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential
    aAA= actionAngleAdiabaticGrid(pot=MWPotential,c=True)
    #circular orbit
    R,vR,vT,z,vz= 1.,0.,1.,0.,0. 
    js= aAA(R,vR,vT,z,vz)
    assert numpy.fabs(js[0]) < 10.**-16., 'Circular orbit in the MWPotential does not have Jr=0'
    assert numpy.fabs(js[2]) < 10.**-16., 'Circular orbit in the MWPotential does not have Jz=0'
    #Close-to-circular orbit
    R,vR,vT,z,vz= 1.01,0.01,1.,0.01,0.01 
    js= aAA(Orbit([R,vR,vT,z,vz]))
    assert numpy.fabs(js[0]) < 10.**-4., 'Close-to-circular orbit in the MWPotential does not have small Jr'
    assert numpy.fabs(js[2]) < 10.**-3., 'Close-to-circular orbit in the MWPotentialspherical LogarithmicHalo does not have small Jz'

#actionAngleAdiabaticGrid actions outside the grid
def test_actionAngleAdiabaticGrid_outsidegrid_c():
    from galpy.actionAngle import actionAngleAdiabaticGrid, \
        actionAngleAdiabatic
    from galpy.potential import MWPotential
    aA= actionAngleAdiabatic(pot=MWPotential,c=True)
    aAA= actionAngleAdiabaticGrid(pot=MWPotential,c=True,Rmax=2.,zmax=0.2)
    R,vR,vT,z,vz,phi= 3.,0.1,1.,0.1,0.1,2.
    js= aA(R,vR,vT,z,vz,phi)
    jsa= aAA(R,vR,vT,z,vz,phi)
    assert numpy.fabs(js[0]-jsa[0]) < 10.**-8., 'actionAngleAdiabaticGrid evaluation outside of the grid fails'
    assert numpy.fabs(js[2]-jsa[2]) < 10.**-8., 'actionAngleAdiabaticGrid evaluation outside of the grid fails'
    assert numpy.fabs(js[2]-aAA.Jz(R,vR,vT,z,vz,phi)) < 10.**-8., 'actionAngleAdiabaticGrid evaluation outside of the grid fails'
    #Also for array
    s= numpy.ones(2)
    js= aA(R,vR,vT,z,vz,phi)
    jsa= aAA(R*s,vR*s,vT*s,z*s,vz*s,phi*s)
    assert numpy.all(numpy.fabs(js[0]-jsa[0]) < 10.**-8.), 'actionAngleAdiabaticGrid evaluation outside of the grid fails'
    assert numpy.all(numpy.fabs(js[2]-jsa[2]) < 10.**-8.), 'actionAngleAdiabaticGrid evaluation outside of the grid fails'
    return None

#Test the actions of an actionAngleAdiabatic
def test_actionAngleAdiabaticGrid_conserved_actions_c():
    from galpy.potential import MWPotential
    from galpy.actionAngle import actionAngleAdiabaticGrid
    from galpy.orbit import Orbit
    obs= Orbit([1.05, 0.02, 1.05, 0.03,0.])
    aAA= actionAngleAdiabaticGrid(pot=MWPotential,c=True)
    check_actionAngle_conserved_actions(aAA,obs,MWPotential,
                                        -1.4,-8.,-1.7,ntimes=101)
    return None

#Test the actionAngleAdiabatic against an isochrone potential: actions
def test_actionAngleAdiabaticGrid_Isochrone_actions():
    from galpy.potential import IsochronePotential
    from galpy.actionAngle import actionAngleAdiabaticGrid, \
        actionAngleIsochrone
    ip= IsochronePotential(normalize=1.,b=1.2)
    aAI= actionAngleIsochrone(ip=ip)
    aAA= actionAngleAdiabaticGrid(pot=ip,c=True)
    R,vR,vT,z,vz,phi= 1.01, 0.05, 1.05, 0.05,0.,2.
    ji= aAI(R,vR,vT,z,vz,phi)
    jia= aAA(R,vR,vT,z,vz,phi)
    djr= numpy.fabs((ji[0]-jia[0])/ji[0])
    dlz= numpy.fabs((ji[1]-jia[1])/ji[1])
    djz= numpy.fabs((ji[2]-jia[2])/ji[2])
    assert djr < 10.**-1.2, 'actionAngleAdiabatic applied to isochrone potential fails for Jr at %f%%' % (djr*100.)
    #Lz and Jz are easy, because ip is a spherical potential
    assert dlz < 10.**-10., 'actionAngleAdiabatic applied to isochrone potential fails for Lz at %f%%' % (dlz*100.)
    assert djz < 10.**-1.2, 'actionAngleAdiabatic applied to isochrone potential fails for Jz at %f%%' % (djz*100.)
    return None

#Basic sanity checking of the actionAngleStaeckel actions
def test_actionAngleStaeckel_basic_actions():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential
    aAS= actionAngleStaeckel(pot=MWPotential,delta=0.71,c=False)
    #circular orbit
    R,vR,vT,z,vz= 1.,0.,1.,0.,0. 
    js= aAS(R,vR,vT,z,vz)
    assert numpy.fabs(js[0][0]) < 10.**-16., 'Circular orbit in the MWPotential does not have Jr=0'
    assert numpy.fabs(js[2][0]) < 10.**-16., 'Circular orbit in the MWPotential does not have Jz=0'
    #Close-to-circular orbit
    R,vR,vT,z,vz= 1.01,0.01,1.,0.01,0.01
    js= aAS(Orbit([R,vR,vT,z,vz]))
    assert numpy.fabs(js[0]) < 10.**-4., 'Close-to-circular orbit in the MWPotential does not have small Jr'
    assert numpy.fabs(js[2]) < 2.*10.**-4., 'Close-to-circular orbit in the MWPotential does not have small Jz'
    #Another close-to-circular orbit
    R,vR,vT,z,vz= 1.0,0.0,0.99,0.0,0.0
    js= aAS(Orbit([R,vR,vT,z,vz]))
    assert numpy.fabs(js[0]) < 10.**-4., 'Close-to-circular orbit in the MWPotential does not have small Jr'
    assert numpy.fabs(js[2]) < 2.*10.**-4., 'Close-to-circular orbit in the MWPotential does not have small Jz'
    #Another close-to-circular orbit
    R,vR,vT,z,vz= 1.0,0.0,1.,0.01,0.0
    js= aAS(Orbit([R,vR,vT,z,vz]))
    assert numpy.fabs(js[0]) < 10.**-4., 'Close-to-circular orbit in the MWPotential does not have small Jr'
    assert numpy.fabs(js[2]) < 2.*10.**-4., 'Close-to-circular orbit in the MWPotential does not have small Jz'
    return None

#Basic sanity checking of the actionAngleStaeckel actions
def test_actionAngleStaeckel_basic_actions_u0():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential
    aAS= actionAngleStaeckel(pot=MWPotential,delta=0.71,c=False,useu0=True)
    #circular orbit
    R,vR,vT,z,vz= 1.,0.,1.,0.,0. 
    js= aAS(R,vR,vT,z,vz)
    assert numpy.fabs(js[0][0]) < 10.**-16., 'Circular orbit in the MWPotential does not have Jr=0'
    assert numpy.fabs(js[2][0]) < 10.**-16., 'Circular orbit in the MWPotential does not have Jz=0'
    #Close-to-circular orbit
    R,vR,vT,z,vz= 1.01,0.01,1.,0.01,0.01 
    js= aAS(Orbit([R,vR,vT,z,vz]))
    assert numpy.fabs(js[0]) < 10.**-4., 'Close-to-circular orbit in the MWPotential does not have small Jr'
    assert numpy.fabs(js[2]) < 2.*10.**-4., 'Close-to-circular orbit in the MWPotential does not have small Jz'
    return None

#Basic sanity checking of the actionAngleStaeckel actions
def test_actionAngleStaeckel_basic_actions_u0_c():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential
    aAS= actionAngleStaeckel(pot=MWPotential,delta=0.71,c=True,useu0=True)
    #circular orbit
    R,vR,vT,z,vz= 1.,0.,1.,0.,0. 
    js= aAS(R,vR,vT,z,vz)
    assert numpy.fabs(js[0][0]) < 10.**-16., 'Circular orbit in the MWPotential does not have Jr=0'
    assert numpy.fabs(js[2][0]) < 10.**-16., 'Circular orbit in the MWPotential does not have Jz=0'
    #Close-to-circular orbit
    R,vR,vT,z,vz= 1.01,0.01,1.,0.01,0.01 
    js= aAS(Orbit([R,vR,vT,z,vz]),u0=1.15)
    assert numpy.fabs(js[0]) < 10.**-4., 'Close-to-circular orbit in the MWPotential does not have small Jr'
    assert numpy.fabs(js[2]) < 2.*10.**-4., 'Close-to-circular orbit in the MWPotential does not have small Jz'
    return None

#Basic sanity checking of the actionAngleStaeckel actions, w/ u0, and interppot
def test_actionAngleStaeckel_basic_actions_u0_interppot_c():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential, interpRZPotential
    ip= interpRZPotential(RZPot=MWPotential,
                          rgrid=(numpy.log(0.01),numpy.log(20.),101),
                          zgrid=(0.,1.,101),logR=True,use_c=True,enable_c=True,
                          interpPot=True)
    aAS= actionAngleStaeckel(pot=ip,delta=0.71,c=True,useu0=True)
    #circular orbit
    R,vR,vT,z,vz= 1.,0.,1.,0.,0. 
    js= aAS(R,vR,vT,z,vz)
    assert numpy.fabs(js[0][0]) < 10.**-16., 'Circular orbit in the MWPotential does not have Jr=0'
    assert numpy.fabs(js[2][0]) < 10.**-16., 'Circular orbit in the MWPotential does not have Jz=0'
    #Close-to-circular orbit
    R,vR,vT,z,vz= 1.01,0.01,1.,0.01,0.01 
    js= aAS(Orbit([R,vR,vT,z,vz]))
    assert numpy.fabs(js[0]) < 10.**-4., 'Close-to-circular orbit in the MWPotential does not have small Jr'
    assert numpy.fabs(js[2]) < 2.*10.**-4., 'Close-to-circular orbit in the MWPotential does not have small Jz'
    return None

#Basic sanity checking of the actionAngleStaeckel actions
def test_actionAngleStaeckel_basic_actions_c():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential
    aAS= actionAngleStaeckel(pot=MWPotential,delta=0.71,c=True)
    #circular orbit
    R,vR,vT,z,vz= 1.,0.,1.,0.,0. 
    js= aAS(R,vR,vT,z,vz)
    assert numpy.fabs(js[0]) < 10.**-16., 'Circular orbit in the MWPotential does not have Jr=0'
    assert numpy.fabs(js[2]) < 10.**-16., 'Circular orbit in the MWPotential does not have Jz=0'
    #Close-to-circular orbit
    R,vR,vT,z,vz= 1.01,0.01,1.,0.01,0.01 
    js= aAS(Orbit([R,vR,vT,z,vz]))
    assert numpy.fabs(js[0]) < 10.**-4., 'Close-to-circular orbit in the MWPotential does not have small Jr'
    assert numpy.fabs(js[2]) < 2.*10.**-4., 'Close-to-circular orbit in the MWPotential does not have small Jz'
    return None

#Basic sanity checking of the actionAngleStaeckel actions, unbound
def test_actionAngleStaeckel_unboundr_actions_c():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.potential import MWPotential
    aAS= actionAngleStaeckel(pot=MWPotential,delta=0.71,c=True)
    #Unbound orbit, shouldn't fail
    R,vR,vT,z,vz= 1.,0.,10.,0.1,0.
    js= aAS(R,vR,vT,z,vz)
    assert js[0] > 1000., 'Unbound in R orbit in the MWPotential does not have large Jr'
    #Another unbound orbit, shouldn't fail
    R,vR,vT,z,vz= 1.,0.1,10.,0.1,0.
    js= aAS(R,vR,vT,z,vz)
    assert js[0] > 1000., 'Unbound in R orbit in the MWPotential does not have large Jr'
    return None

#Basic sanity checking of the actionAngleStaeckel actions
def test_actionAngleStaeckel_zerolz_actions_c():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.potential import MWPotential
    aAS= actionAngleStaeckel(pot=MWPotential,c=True,delta=0.71)
    #Zero angular momentum, so rperi=0, but should have finite jr
    R,vR,vT,z,vz= 1.,0.,0.,0., 0.
    js= aAS(R,vR,vT,z,vz)
    R,vR,vT,z,vz= 1.,0.,0.0000001,0., 0.
    js2= aAS(R,vR,vT,z,vz)
    assert numpy.fabs(js[0]-js2[0]) < 10.**-6., 'Orbit with zero angular momentum does not have the correct Jr'
    #Zero angular momentum, so rperi=0, but should have finite jr
    R,vR,vT,z,vz= 1.,-0.5,0.,0., 0.
    js= aAS(R,vR,vT,z,vz)
    R,vR,vT,z,vz= 1.,-0.5,0.0000001,0., 0.
    js2= aAS(R,vR,vT,z,vz)
    assert numpy.fabs(js[0]-js2[0]) < 10.**-6., 'Orbit with zero angular momentum does not have the correct Jr'
    return None

#Basic sanity checking of the actionAngleStaeckel frequencies
def test_actionAngleStaeckel_basic_freqs_c():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.potential import MWPotential, epifreq, omegac, verticalfreq
    from galpy.orbit import Orbit
    aAS= actionAngleStaeckel(pot=MWPotential,delta=0.71,c=True)
    #circular orbit
    R,vR,vT,z,vz= 1.,0.,1.,0.,0. 
    jos= aAS.actionsFreqs(R,vR,vT,z,vz)
    assert numpy.fabs((jos[3]-epifreq(MWPotential,1.))/epifreq(MWPotential,1.)) < 10.**-12., 'Circular orbit in the MWPotential does not have Or=kappa at %g%%' % (100.*numpy.fabs((jos[3]-epifreq(MWPotential,1.))/epifreq(MWPotential,1.)))
    assert numpy.fabs((jos[4]-omegac(MWPotential,1.))/omegac(MWPotential,1.)) < 10.**-12., 'Circular orbit in the MWPotential does not have Op=Omega at %g%%' % (100.*numpy.fabs((jos[4]-omegac(MWPotential,1.))/omegac(MWPotential,1.)))
    assert numpy.fabs((jos[5]-verticalfreq(MWPotential,1.))/verticalfreq(MWPotential,1.)) < 10.**-12., 'Circular orbit in the MWPotential does not have Oz=nu at %g%%' % (100.*numpy.fabs((jos[5]-verticalfreq(MWPotential,1.))/verticalfreq(MWPotential,1.)))
    #close-to-circular orbit
    R,vR,vT,z,vz= 1.,0.01,1.01,0.01,0.01 
    jos= aAS.actionsFreqs(Orbit([R,vR,vT,z,vz]))
    assert numpy.fabs((jos[3]-epifreq(MWPotential,1.))/epifreq(MWPotential,1.)) < 10.**-1.9, 'Close-to-circular orbit in the MWPotential does not have Or=kappa at %g%%' % (100.*numpy.fabs((jos[3]-epifreq(MWPotential,1.))/epifreq(MWPotential,1.)))
    assert numpy.fabs((jos[4]-omegac(MWPotential,1.))/omegac(MWPotential,1.)) < 10.**-1.9, 'Close-to-circular orbit in the MWPotential does not have Op=Omega at %g%%' % (100.*numpy.fabs((jos[4]-omegac(MWPotential,1.))/omegac(MWPotential,1.)))
    assert numpy.fabs((jos[5]-verticalfreq(MWPotential,1.))/verticalfreq(MWPotential,1.)) < 10.**-1.5, 'Close-to-circular orbit in the MWPotential does not have Oz=nu at %g%%' % (100.*numpy.fabs((jos[5]-verticalfreq(MWPotential,1.))/verticalfreq(MWPotential,1.)))
    #another close-to-circular orbit
    R,vR,vT,z,vz= 1.,0.03,1.02,0.03,0.01 
    jos= aAS.actionsFreqs(Orbit([R,vR,vT,z,vz,2.]))
    assert numpy.fabs((jos[3]-epifreq(MWPotential,1.))/epifreq(MWPotential,1.)) < 10.**-1.5, 'Close-to-circular orbit in the MWPotential does not have Or=kappa at %g%%' % (100.*numpy.fabs((jos[3]-epifreq(MWPotential,1.))/epifreq(MWPotential,1.)))
    assert numpy.fabs((jos[4]-omegac(MWPotential,1.))/omegac(MWPotential,1.)) < 10.**-1.5, 'Close-to-circular orbit in the MWPotential does not have Op=Omega at %g%%' % (100.*numpy.fabs((jos[4]-omegac(MWPotential,1.))/omegac(MWPotential,1.)))
    assert numpy.fabs((jos[5]-verticalfreq(MWPotential,1.))/verticalfreq(MWPotential,1.)) < 10.**-0.9, 'Close-to-circular orbit in the MWPotential does not have Oz=nu at %g%%' % (100.*numpy.fabs((jos[5]-verticalfreq(MWPotential,1.))/verticalfreq(MWPotential,1.)))
    return None

#Basic sanity checking of the actionAngleStaeckel actions
def test_actionAngleStaeckel_basic_freqsAngles():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.potential import MWPotential, epifreq, omegac, verticalfreq
    from galpy.orbit import Orbit
    aAS= actionAngleStaeckel(pot=MWPotential,delta=0.71,c=True)
    #v. close-to-circular orbit
    R,vR,vT,z,vz= 1.,10.**-4.,1.,10.**-4.,0.
    jos= aAS.actionsFreqs(Orbit([R,vR,vT,z,vz,2.]))
    assert numpy.fabs((jos[3]-epifreq(MWPotential,1.))/epifreq(MWPotential,1.)) < 10.**-1.9, 'Close-to-circular orbit in the MWPotential does not have Or=kappa at %g%%' % (100.*numpy.fabs((jos[3]-epifreq(MWPotential,1.))/epifreq(MWPotential,1.)))
    assert numpy.fabs((jos[4]-omegac(MWPotential,1.))/omegac(MWPotential,1.)) < 10.**-1.9, 'Close-to-circular orbit in the MWPotential does not have Op=Omega at %g%%' % (100.*numpy.fabs((jos[4]-omegac(MWPotential,1.))/omegac(MWPotential,1.)))
    assert numpy.fabs((jos[5]-verticalfreq(MWPotential,1.))/verticalfreq(MWPotential,1.)) < 10.**-1.9, 'Close-to-circular orbit in the MWPotential does not have Oz=nu at %g%%' % (100.*numpy.fabs((jos[5]-verticalfreq(MWPotential,1.))/verticalfreq(MWPotential,1.)))
    return None

#Basic sanity checking of the actionAngleStaeckel frequencies
def test_actionAngleStaeckel_basic_freqs_c_u0():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.potential import MWPotential, epifreq, omegac, verticalfreq
    from galpy.orbit import Orbit
    aAS= actionAngleStaeckel(pot=MWPotential,delta=0.71,c=True,useu0=True)
    #circular orbit
    R,vR,vT,z,vz= 1.,0.,1.,0.,0. 
    jos= aAS.actionsFreqs(R,vR,vT,z,vz)
    assert numpy.fabs((jos[3]-epifreq(MWPotential,1.))/epifreq(MWPotential,1.)) < 10.**-12., 'Circular orbit in the MWPotential does not have Or=kappa at %g%%' % (100.*numpy.fabs((jos[3]-epifreq(MWPotential,1.))/epifreq(MWPotential,1.)))
    assert numpy.fabs((jos[4]-omegac(MWPotential,1.))/omegac(MWPotential,1.)) < 10.**-12., 'Circular orbit in the MWPotential does not have Op=Omega at %g%%' % (100.*numpy.fabs((jos[4]-omegac(MWPotential,1.))/omegac(MWPotential,1.)))
    assert numpy.fabs((jos[5]-verticalfreq(MWPotential,1.))/verticalfreq(MWPotential,1.)) < 10.**-12., 'Circular orbit in the MWPotential does not have Oz=nu at %g%%' % (100.*numpy.fabs((jos[5]-verticalfreq(MWPotential,1.))/verticalfreq(MWPotential,1.)))
    #close-to-circular orbit
    R,vR,vT,z,vz= 1.,0.01,1.01,0.01,0.01 
    jos= aAS.actionsFreqs(Orbit([R,vR,vT,z,vz]),u0=1.15)
    assert numpy.fabs((jos[3]-epifreq(MWPotential,1.))/epifreq(MWPotential,1.)) < 10.**-1.9, 'Close-to-circular orbit in the MWPotential does not have Or=kappa at %g%%' % (100.*numpy.fabs((jos[3]-epifreq(MWPotential,1.))/epifreq(MWPotential,1.)))
    assert numpy.fabs((jos[4]-omegac(MWPotential,1.))/omegac(MWPotential,1.)) < 10.**-1.9, 'Close-to-circular orbit in the MWPotential does not have Op=Omega at %g%%' % (100.*numpy.fabs((jos[4]-omegac(MWPotential,1.))/omegac(MWPotential,1.)))
    assert numpy.fabs((jos[5]-verticalfreq(MWPotential,1.))/verticalfreq(MWPotential,1.)) < 10.**-1.5, 'Close-to-circular orbit in the MWPotential does not have Oz=nu at %g%%' % (100.*numpy.fabs((jos[5]-verticalfreq(MWPotential,1.))/verticalfreq(MWPotential,1.)))
    return None

#Basic sanity checking of the actionAngleStaeckel actions
def test_actionAngleStaeckel_basic_freqs_u0():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.potential import MWPotential, epifreq, omegac, verticalfreq, \
        interpRZPotential
    from galpy.orbit import Orbit
    ip= interpRZPotential(RZPot=MWPotential,
                          rgrid=(numpy.log(0.01),numpy.log(20.),101),
                          zgrid=(0.,1.,101),logR=True,use_c=True,enable_c=True,
                          interpPot=True)
    aAS= actionAngleStaeckel(pot=ip,delta=0.71,c=True,useu0=True)
    #v. close-to-circular orbit
    R,vR,vT,z,vz= 1.,10.**-4.,1.,10.**-4.,0.
    jos= aAS.actionsFreqs(Orbit([R,vR,vT,z,vz,2.]))
    assert numpy.fabs((jos[3]-epifreq(MWPotential,1.))/epifreq(MWPotential,1.)) < 10.**-1.9, 'Close-to-circular orbit in the MWPotential does not have Or=kappa at %g%%' % (100.*numpy.fabs((jos[3]-epifreq(MWPotential,1.))/epifreq(MWPotential,1.)))
    assert numpy.fabs((jos[4]-omegac(MWPotential,1.))/omegac(MWPotential,1.)) < 10.**-1.9, 'Close-to-circular orbit in the MWPotential does not have Op=Omega at %g%%' % (100.*numpy.fabs((jos[4]-omegac(MWPotential,1.))/omegac(MWPotential,1.)))
    assert numpy.fabs((jos[5]-verticalfreq(MWPotential,1.))/verticalfreq(MWPotential,1.)) < 10.**-1.9, 'Close-to-circular orbit in the MWPotential does not have Oz=nu at %g%%' % (100.*numpy.fabs((jos[5]-verticalfreq(MWPotential,1.))/verticalfreq(MWPotential,1.)))
    return None

#Basic sanity checking of the actionAngleStaeckel actions, unbound
def test_actionAngleStaeckel_unboundr_freqs_c():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.potential import MWPotential
    aAS= actionAngleStaeckel(pot=MWPotential,delta=0.71,c=True)
    #Unbound orbit, shouldn't fail
    R,vR,vT,z,vz= 1.,0.1,10.,0.1,0.
    js= aAS.actionsFreqs(R,vR,vT,z,vz)
    assert js[0] > 1000., 'Unbound in R orbit in the MWPotential does not have large Jr'
    assert js[3] > 1000., 'Unbound in R orbit in the MWPotential does not have large Or'
    assert js[4] > 1000., 'Unbound in R orbit in the MWPotential does not have large Op'
    assert js[5] > 1000., 'Unbound in R orbit in the MWPotential does not have large Oz'
    return None

#Basic sanity checking of the actionAngleStaeckel actions, unbound
def test_actionAngleStaeckel_unboundr_angles_c():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.potential import MWPotential
    aAS= actionAngleStaeckel(pot=MWPotential,delta=0.71,c=True)
    #Unbound orbit, shouldn't fail
    R,vR,vT,z,vz,phi= 1.,0.1,10.,0.1,0.,0.
    js= aAS.actionsFreqsAngles(R,vR,vT,z,vz,phi)
    assert js[0] > 1000., 'Unbound in R orbit in the MWPotential does not have large Jr'
    assert js[6] > 1000., 'Unbound in R orbit in the MWPotential does not have large ar'
    assert js[7] > 1000., 'Unbound in R orbit in the MWPotential does not have large ap'
    assert js[8] > 1000., 'Unbound in R orbit in the MWPotential does not have large az'
    return None

#Basic sanity checking of the actionAngleStaeckel actions, unbound
def test_actionAngleStaeckel_circular_angles_c():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.potential import MWPotential
    aAS= actionAngleStaeckel(pot=MWPotential,delta=0.71,c=True)
    #Circular orbits, have zero r and z angles in our implementation
    R,vR,vT,z,vz,phi= 1.,0.,1.,0.,0.,1.
    js= aAS.actionsFreqsAngles(R,vR,vT,z,vz,phi)
    assert numpy.fabs(js[6]) < 10.**-8., 'Circular orbit does not have zero angles'
    assert numpy.fabs(js[8]) < 10.**-8., 'Circular orbit does not have zero angles'
    return None

#Basic sanity checking of the actionAngleStaeckel ecc, zmax, rperi, rap calc.
def test_actionAngleStaeckel_basic_EccZmaxRperiRap():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.potential import MWPotential
    aAS= actionAngleStaeckel(pot=MWPotential,delta=0.71,c=False)
    #circular orbit
    R,vR,vT,z,vz= 1.,0.,1.,0.,0. 
    te,tzmax,_,_= aAS.EccZmaxRperiRap(R,vR,vT,z,vz)
    assert numpy.fabs(te) < 10.**-16., 'Circular orbit in the MWPotential does not have e=0'
    assert numpy.fabs(tzmax) < 10.**-16., 'Circular orbit in the MWPotential does not have zmax=0'
    #Close-to-circular orbit
    R,vR,vT,z,vz= 1.01,0.01,1.,0.01,0.01
    te,tzmax,_,_= aAS.EccZmaxRperiRap(R,vR,vT,z,vz)
    assert numpy.fabs(te) < 10.**-2., 'Close-to-circular orbit in the MWPotential does not have small eccentricity'
    assert numpy.fabs(tzmax) < 2.*10.**-2., 'Close-to-circular orbit in the MWPotential does not have small zmax'
    #Another close-to-circular orbit
    R,vR,vT,z,vz= 1.0,0.0,0.99,0.0,0.0
    te,tzmax,_,_= aAS.EccZmaxRperiRap(R,vR,vT,z,vz)
    assert numpy.fabs(te) < 10.**-2., 'Close-to-circular orbit in the MWPotential does not have small eccentricity'
    assert numpy.fabs(tzmax) < 2.*10.**-2., 'Close-to-circular orbit in the MWPotential does not have small zmax'
    #Another close-to-circular orbit
    R,vR,vT,z,vz= 1.0,0.0,1.,0.01,0.0
    te,tzmax,_,_= aAS.EccZmaxRperiRap(R,vR,vT,z,vz)
    assert numpy.fabs(te) < 10.**-2., 'Close-to-circular orbit in the MWPotential does not have small eccentricity'
    assert numpy.fabs(tzmax) < 2.*10.**-2., 'Close-to-circular orbit in the MWPotential does not have small zmax'
    return None

#Basic sanity checking of the actionAngleStaeckel ecc, zmax, rperi, rap calc.
def test_actionAngleStaeckel_basic_EccZmaxRperiRap_u0():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.potential import MWPotential
    aAS= actionAngleStaeckel(pot=MWPotential,delta=0.71,c=False,useu0=True)
    #circular orbit
    R,vR,vT,z,vz= 1.,0.,1.,0.,0. 
    te,tzmax,_,_= aAS.EccZmaxRperiRap(R,vR,vT,z,vz)
    assert numpy.fabs(te) < 10.**-16., 'Circular orbit in the MWPotential does not have e=0'
    assert numpy.fabs(tzmax) < 10.**-16., 'Circular orbit in the MWPotential does not have zmax=0'
    #Close-to-circular orbit
    R,vR,vT,z,vz= 1.01,0.01,1.,0.01,0.01 
    te,tzmax,_,_= aAS.EccZmaxRperiRap(R,vR,vT,z,vz)
    assert numpy.fabs(te) < 10.**-2., 'Close-to-circular orbit in the MWPotential does not have small eccentricity'
    assert numpy.fabs(tzmax) < 2.*10.**-2., 'Close-to-circular orbit in the MWPotential does not have small zmax'
    return None

#Basic sanity checking of the actionAngleStaeckel ecc, zmax, rperi, rap calc.
def test_actionAngleStaeckel_basic_EccZmaxRperiRap_u0_c():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.potential import MWPotential
    aAS= actionAngleStaeckel(pot=MWPotential,delta=0.71,c=True,useu0=True)
    #circular orbit
    R,vR,vT,z,vz= 1.,0.,1.,0.,0. 
    te,tzmax,_,_= aAS.EccZmaxRperiRap(R,vR,vT,z,vz)
    assert numpy.fabs(te) < 10.**-16., 'Circular orbit in the MWPotential does not have e=0'
    assert numpy.fabs(tzmax) < 10.**-16., 'Circular orbit in the MWPotential does not have zmax=0'
    #Close-to-circular orbit
    R,vR,vT,z,vz= 1.01,0.01,1.,0.01,0.01 
    te,tzmax,_,_= aAS.EccZmaxRperiRap(R,vR,vT,z,vz)
    assert numpy.fabs(te) < 10.**-2., 'Close-to-circular orbit in the MWPotential does not have small eccentricity'
    assert numpy.fabs(tzmax) < 2.*10.**-2., 'Close-to-circular orbit in the MWPotential does not have small zmax'
    return None

#Test the actions of an actionAngleStaeckel
def test_actionAngleStaeckel_conserved_actions():
    from galpy.potential import MWPotential
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    aAS= actionAngleStaeckel(pot=MWPotential,c=False,delta=0.71)
    obs= Orbit([1.05, 0.02, 1.05, 0.03,0.])
    check_actionAngle_conserved_actions(aAS,obs,MWPotential,
                                        -2.,-8.,-2.,ntimes=101)
    return None

#Test the actions of an actionAngleStaeckel, more eccentric orbit
def test_actionAngleStaeckel_conserved_actions_ecc():
    from galpy.potential import MWPotential
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    aAS= actionAngleStaeckel(pot=MWPotential,c=False,delta=0.71)
    obs= Orbit([1.1,0.2, 1.3, 0.3,0.])
    check_actionAngle_conserved_actions(aAS,obs,MWPotential,
                                        -1.5,-8.,-1.4,ntimes=101)
    return None

#Test the actions of an actionAngleStaeckel
def test_actionAngleStaeckel_conserved_actions_c():
    from galpy.potential import MWPotential, DoubleExponentialDiskPotential, \
        FlattenedPowerPotential, interpRZPotential, KuzminDiskPotential, \
        TriaxialHernquistPotential, TriaxialJaffePotential, \
        TriaxialNFWPotential, SCFPotential, DiskSCFPotential
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.orbit_src.FullOrbit import ext_loaded
    ip= interpRZPotential(RZPot=MWPotential,
                          rgrid=(numpy.log(0.01),numpy.log(20.),101),
                          zgrid=(0.,1.,101),logR=True,use_c=True,enable_c=True,
                          interpPot=True,interpRforce=True,interpzforce=True)
    pots= [MWPotential,
           DoubleExponentialDiskPotential(normalize=1.),
           FlattenedPowerPotential(normalize=1.),
           FlattenedPowerPotential(normalize=1.,alpha=0.),
           KuzminDiskPotential(normalize=1.,a=1./8.),
           TriaxialHernquistPotential(normalize=1.,c=0.2,pa=1.1), # tests rot, but not well
           TriaxialNFWPotential(normalize=1.,c=0.3,pa=1.1),
           TriaxialJaffePotential(normalize=1.,c=0.4,pa=1.1),
           SCFPotential(normalize=1.),
           DiskSCFPotential(normalize=1.),
           ip]
    for pot in pots:
        aAS= actionAngleStaeckel(pot=pot,c=True,delta=0.71)
        obs= Orbit([1.05, 0.02, 1.05, 0.03,0.,2.])
        if not ext_loaded: #odeint is not as accurate as dopr54_c
            check_actionAngle_conserved_actions(aAS,obs,pot,
                                                -1.6,-6.,-1.6,ntimes=101,
                                                inclphi=True)
        else:
            check_actionAngle_conserved_actions(aAS,obs,pot,
                                                -1.6,-8.,-1.65,ntimes=101,
                                                inclphi=True)
    return None

#Test the actions of an actionAngleStaeckel, for a dblexp disk far away from the center
def test_actionAngleStaeckel_conserved_actions_c_specialdblexp():
    from galpy.potential import DoubleExponentialDiskPotential
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    pot= DoubleExponentialDiskPotential(normalize=1.)
    aAS= actionAngleStaeckel(pot=pot,c=True,delta=0.01)
    #Close to circular in the Keplerian regime
    obs= Orbit([7.05, 0.002,pot.vcirc(7.05), 0.003,0.,2.])
    check_actionAngle_conserved_actions(aAS,obs,pot,
                                        -2.,-7.,-2.,ntimes=101,
                                        inclphi=True)
    return None

#Test the actions of an actionAngleStaeckel
def test_actionAngleStaeckel_wSpherical_conserved_actions_c():
    from galpy import potential
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.orbit_src.FullOrbit import ext_loaded
    from test_potential import mockSCFZeeuwPotential, \
        mockSphericalSoftenedNeedleBarPotential, \
        mockSmoothedLogarithmicHaloPotential
    lp= potential.LogarithmicHaloPotential(normalize=1.,q=1.)
    lpb= potential.LogarithmicHaloPotential(normalize=1.,q=1.,b=1.) # same |^
    hp= potential.HernquistPotential(normalize=1.)
    jp= potential.JaffePotential(normalize=1.)
    np= potential.NFWPotential(normalize=1.)
    ip= potential.IsochronePotential(normalize=1.,b=1.)
    pp= potential.PowerSphericalPotential(normalize=1.)
    lp2= potential.PowerSphericalPotential(normalize=1.,alpha=2.)
    ppc= potential.PowerSphericalPotentialwCutoff(normalize=1.)
    plp= potential.PlummerPotential(normalize=1.)
    psp= potential.PseudoIsothermalPotential(normalize=1.)
    bp= potential.BurkertPotential(normalize=1.)
    scfp= potential.SCFPotential(normalize=1.)
    scfzp = mockSCFZeeuwPotential(); scfzp.normalize(1.); 
    msoftneedlep= mockSphericalSoftenedNeedleBarPotential()
    msmlp= mockSmoothedLogarithmicHaloPotential()
    pots= [lp,lpb,hp,jp,np,ip,pp,lp2,ppc,plp,psp,bp,scfp,scfzp,
           msoftneedlep,msmlp]
    for pot in pots:
        aAS= actionAngleStaeckel(pot=pot,c=True,delta=0.01)
        obs= Orbit([1.1, 0.3, 1.2, 0.2,0.5,2.])
        if not ext_loaded: #odeint is not as accurate as dopr54_c
            check_actionAngle_conserved_actions(aAS,obs,pot,
                                                -2.,-5.,-2.,ntimes=101,
                                                inclphi=True)
        else:
            check_actionAngle_conserved_actions(aAS,obs,pot,
                                                -2.,-8.,-2.,ntimes=101,
                                                inclphi=True)
    return None
#Test the actions of an actionAngleStaeckel
def test_actionAngleStaeckel_conserved_actions_fixed_quad():
    from galpy.potential import MWPotential
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.orbit_src.FullOrbit import ext_loaded
    aAS= actionAngleStaeckel(pot=MWPotential,c=False,delta=0.71)
    obs= Orbit([1.05, 0.02, 1.05, 0.03,0.,2.])
    if not ext_loaded: #odeint is not as accurate as dopr54_c
        check_actionAngle_conserved_actions(aAS,obs,MWPotential,
                                            -2.,-5.,-2.,ntimes=101,
                                            fixed_quad=True,inclphi=True)
    else:
        check_actionAngle_conserved_actions(aAS,obs,MWPotential,
                                            -2.,-8.,-2.,ntimes=101,
                                            fixed_quad=True,inclphi=True)
    return None

#Test that the angles of an actionAngleStaeckel increase linearly
def test_actionAngleStaeckel_linear_angles():
    from galpy.potential import MWPotential
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    aAS= actionAngleStaeckel(pot=MWPotential,delta=0.71,c=True)
    obs= Orbit([1.05, 0.02, 1.05, 0.03,0.,2.])
    check_actionAngle_linear_angles(aAS,obs,MWPotential,
                                    -2.,-4.,-3.,
                                    -3.,-3.,-2.,
                                    -2.,-3.5,-2.,
                                    ntimes=1001) #need fine sampling for de-period
    return None

#Test that the angles of an actionAngleStaeckel increase linearly, interppot
def test_actionAngleStaeckel_linear_angles_interppot():
    from galpy.potential import MWPotential, interpRZPotential
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    ip= interpRZPotential(RZPot=MWPotential,
                          rgrid=(numpy.log(0.01),numpy.log(20.),101),
                          zgrid=(0.,1.,101),logR=True,use_c=True,enable_c=True,
                          interpPot=True,interpRforce=True,interpzforce=True)
    aAS= actionAngleStaeckel(pot=ip,delta=0.71,c=True)
    obs= Orbit([1.05, 0.02, 1.05, 0.03,0.,2.])
    check_actionAngle_linear_angles(aAS,obs,MWPotential,
                                    -2.,-4.,-3.,
                                    -3.,-3.,-2.,
                                    -2.,-3.5,-2.,
                                    ntimes=1001) #need fine sampling for de-period
    return None

#Test that the angles of an actionAngleStaeckel increase linearly
def test_actionAngleStaeckel_linear_angles_u0():
    from galpy.potential import MWPotential
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    aAS= actionAngleStaeckel(pot=MWPotential,delta=0.71,c=True,useu0=True)
    obs= Orbit([1.05, 0.02, 1.05, 0.03,0.,2.])
    check_actionAngle_linear_angles(aAS,obs,MWPotential,
                                    -2.,-4.,-3.,
                                    -3.,-3.,-2.,
                                    -2.,-3.5,-2.,
                                    ntimes=1001) #need fine sampling for de-period
    #specifying u0
    check_actionAngle_linear_angles(aAS,obs,MWPotential,
                                    -2.,-4.,-3.,
                                    -3.,-3.,-2.,
                                    -2.,-3.5,-2.,
                                    ntimes=1001,u0=1.23) #need fine sampling for de-period
    return None

#Test the conservation of ecc, zmax, rperi, rap of an actionAngleStaeckel
def test_actionAngleStaeckel_conserved_EccZmaxRperiRap():
    from galpy.potential import MWPotential
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    aAS= actionAngleStaeckel(pot=MWPotential,c=False,delta=0.71)
    obs= Orbit([1.05, 0.02, 1.05, 0.03,0.,0.])
    check_actionAngle_conserved_EccZmaxRperiRap(aAS,obs,MWPotential,
                                                -2.,-2.,-2.,-2.,ntimes=101)
    return None

#Test the conservation of ecc, zmax, rperi, rap of an actionAngleStaeckel
def test_actionAngleStaeckel_conserved_EccZmaxRperiRap_ecc():
    from galpy.potential import MWPotential
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    aAS= actionAngleStaeckel(pot=MWPotential,c=False,delta=0.71)
    obs= Orbit([1.1,0.2, 1.3, 0.3,0.,2.])
    check_actionAngle_conserved_EccZmaxRperiRap(aAS,obs,MWPotential,
                                                -1.8,-1.4,-1.8,-1.8,ntimes=101)
    return None

#Test the conservation of ecc, zmax, rperi, rap of an actionAngleStaeckel
def test_actionAngleStaeckel_conserved_EccZmaxRperiRap_c():
    from galpy.potential import MWPotential, DoubleExponentialDiskPotential, \
        FlattenedPowerPotential, interpRZPotential, KuzminDiskPotential, \
        TriaxialHernquistPotential, TriaxialJaffePotential, \
        TriaxialNFWPotential, SCFPotential, DiskSCFPotential
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.orbit_src.FullOrbit import ext_loaded
    ip= interpRZPotential(RZPot=MWPotential,
                          rgrid=(numpy.log(0.01),numpy.log(20.),101),
                          zgrid=(0.,1.,101),logR=True,use_c=True,enable_c=True,
                          interpPot=True,interpRforce=True,interpzforce=True)
    pots= [MWPotential,
           DoubleExponentialDiskPotential(normalize=1.),
           FlattenedPowerPotential(normalize=1.),
           FlattenedPowerPotential(normalize=1.,alpha=0.),
           KuzminDiskPotential(normalize=1.,a=1./8.),
           TriaxialHernquistPotential(normalize=1.,c=0.2,pa=1.1), # tests rot, but not well
           TriaxialNFWPotential(normalize=1.,c=0.3,pa=1.1),
           TriaxialJaffePotential(normalize=1.,c=0.4,pa=1.1),
           SCFPotential(normalize=1.),
           DiskSCFPotential(normalize=1.),
           ip]
    for pot in pots:
        aAS= actionAngleStaeckel(pot=pot,c=True,delta=0.71)
        obs= Orbit([1.05, 0.02, 1.05, 0.03,0.,2.])
        check_actionAngle_conserved_EccZmaxRperiRap(aAS,obs,pot,
                                                    -1.8,-1.3,-1.8,-1.8,
                                                    ntimes=101)
    return None

#HERE

#Test the actionAngleStaeckel against an isochrone potential: actions
def test_actionAngleStaeckel_otherIsochrone_actions():
    from galpy.potential import IsochronePotential
    from galpy.actionAngle import actionAngleStaeckel, \
        actionAngleIsochrone, estimateDeltaStaeckel
    ip= IsochronePotential(normalize=1.,b=1.2)
    aAI= actionAngleIsochrone(ip=ip)
    aAA= actionAngleStaeckel(pot=ip,c=False,delta=0.1) #not ideal
    R,vR,vT,z,vz,phi= 1.01, 0.05, 1.05, 0.05,0.,2.
    ji= aAI(R,vR,vT,z,vz,phi)
    jia= aAA(R,vR,vT,z,vz,phi)
    djr= numpy.fabs((ji[0]-jia[0])/ji[0])
    dlz= numpy.fabs((ji[1]-jia[1])/ji[1])
    djz= numpy.fabs((ji[2]-jia[2])/ji[2])
    assert djr < 10.**-3., 'actionAngleStaeckel applied to isochrone potential fails for Jr at %f%%' % (djr*100.)
    #Lz and Jz are easy, because ip is a spherical potential
    assert dlz < 10.**-10., 'actionAngleStaeckel applied to isochrone potential fails for Lz at %f%%' % (dlz*100.)
    assert djz < 10.**-3., 'actionAngleStaeckel applied to isochrone potential fails for Jz at %f%%' % (djz*100.)
    return None

#Test the actionAngleStaeckel against an isochrone potential: actions
def test_actionAngleStaeckel_otherIsochrone_actions_fixed_quad():
    from galpy.potential import IsochronePotential
    from galpy.actionAngle import actionAngleStaeckel, \
        actionAngleIsochrone, estimateDeltaStaeckel
    ip= IsochronePotential(normalize=1.,b=1.2)
    aAI= actionAngleIsochrone(ip=ip)
    aAA= actionAngleStaeckel(pot=ip,c=False,delta=0.1) #not ideal
    R,vR,vT,z,vz,phi= 1.01, 0.05, 1.05, 0.05,0.,2.
    ji= aAI(R,vR,vT,z,vz,phi)
    jia= aAA(R,vR,vT,z,vz,phi,fixed_quad=True)
    djr= numpy.fabs((ji[0]-jia[0])/ji[0])
    dlz= numpy.fabs((ji[1]-jia[1])/ji[1])
    djz= numpy.fabs((ji[2]-jia[2])/ji[2])
    assert djr < 10.**-3., 'actionAngleStaeckel applied to isochrone potential fails for Jr at %f%%' % (djr*100.)
    #Lz and Jz are easy, because ip is a spherical potential
    assert dlz < 10.**-10., 'actionAngleStaeckel applied to isochrone potential fails for Lz at %f%%' % (dlz*100.)
    assert djz < 10.**-3., 'actionAngleStaeckel applied to isochrone potential fails for Jz at %f%%' % (djz*100.)
    return None

#Test the actionAngleStaeckel against an isochrone potential: actions
def test_actionAngleStaeckel_otherIsochrone_actions_c():
    from galpy.potential import IsochronePotential
    from galpy.actionAngle import actionAngleStaeckel, \
        actionAngleIsochrone, estimateDeltaStaeckel
    ip= IsochronePotential(normalize=1.,b=1.2)
    aAI= actionAngleIsochrone(ip=ip)
    aAA= actionAngleStaeckel(pot=ip,c=True,delta=0.1) #not ideal
    R,vR,vT,z,vz,phi= 1.01, 0.05, 1.05, 0.05,0.,2.
    ji= aAI(R,vR,vT,z,vz,phi)
    jia= aAA(R,vR,vT,z,vz,phi)
    djr= numpy.fabs((ji[0]-jia[0])/ji[0])
    dlz= numpy.fabs((ji[1]-jia[1])/ji[1])
    djz= numpy.fabs((ji[2]-jia[2])/ji[2])
    assert djr < 10.**-3., 'actionAngleStaeckel applied to isochrone potential fails for Jr at %f%%' % (djr*100.)
    #Lz and Jz are easy, because ip is a spherical potential
    assert dlz < 10.**-10., 'actionAngleStaeckel applied to isochrone potential fails for Lz at %f%%' % (dlz*100.)
    assert djz < 10.**-3., 'actionAngleStaeckel applied to isochrone potential fails for Jz at %f%%' % (djz*100.)
    return None

#Test the actionAngleStaeckel against an isochrone potential: frequencies
def test_actionAngleStaeckel_otherIsochrone_freqs():   
    from galpy.potential import IsochronePotential
    from galpy.actionAngle import actionAngleStaeckel, \
        actionAngleIsochrone
    ip= IsochronePotential(normalize=1.,b=1.2)
    aAI= actionAngleIsochrone(ip=ip)
    aAS= actionAngleStaeckel(pot=ip,delta=0.1,c=True)
    R,vR,vT,z,vz,phi= 1.01, 0.05, 1.05, 0.05,0.,2.
    jiO= aAI.actionsFreqs(R,vR,vT,z,vz,phi)
    jiaO= aAS.actionsFreqs(R,vR,vT,z,vz,phi)
    dOr= numpy.fabs((jiO[3]-jiaO[3])/jiO[3])
    dOp= numpy.fabs((jiO[4]-jiaO[4])/jiO[4])
    dOz= numpy.fabs((jiO[5]-jiaO[5])/jiO[5])
    assert dOr < 10.**-5., 'actionAngleStaeckel applied to isochrone potential fails for Or at %g%%' % (dOr*100.)
    assert dOp < 10.**-5., 'actionAngleStaeckel applied to isochrone potential fails for Op at %g%%' % (dOp*100.)
    assert dOz < 1.5*10.**-4., 'actionAngleStaeckel applied to isochrone potential fails for Oz at %g%%' % (dOz*100.)
    return None

#Test the actionAngleStaeckel against an isochrone potential: angles
def test_actionAngleStaeckel_otherIsochrone_angles():   
    from galpy.potential import IsochronePotential
    from galpy.actionAngle import actionAngleStaeckel, \
        actionAngleIsochrone
    ip= IsochronePotential(normalize=1.,b=1.2)
    aAI= actionAngleIsochrone(ip=ip)
    aAS= actionAngleStaeckel(pot=ip,delta=0.1,c=True)
    R,vR,vT,z,vz,phi= 1.01, 0.05, 1.05, 0.03,-0.01,2.
    jiO= aAI.actionsFreqsAngles(R,vR,vT,z,vz,phi)
    jiaO= aAS.actionsFreqsAngles(R,vR,vT,z,vz,phi)
    dar= numpy.fabs((jiO[6]-jiaO[6])/jiO[6])
    dap= numpy.fabs((jiO[7]-jiaO[7])/jiO[7])
    daz= numpy.fabs((jiO[8]-jiaO[8])/jiO[8])
    assert dar < 10.**-4., 'actionAngleStaeckel applied to isochrone potential fails for ar at %g%%' % (dar*100.)
    assert dap < 10.**-6., 'actionAngleStaeckel applied to isochrone potential fails for ap at %g%%' % (dap*100.)
    assert daz < 10.**-4., 'actionAngleStaeckel applied to isochrone potential fails for az at %g%%' % (daz*100.)
    return None

#Basic sanity checking of the actionAngleStaeckelGrid actions (incl. conserved, bc takes a lot of time)
def test_actionAngleStaeckelGrid_basicAndConserved_actions():
    from galpy.actionAngle import actionAngleStaeckelGrid
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential
    aAA= actionAngleStaeckelGrid(pot=MWPotential,delta=0.71,c=False,nLz=20)
    #circular orbit
    R,vR,vT,z,vz= 1.,0.,1.,0.,0. 
    assert numpy.fabs(aAA.JR(R,vR,vT,z,vz,0.)) < 10.**-16., 'Circular orbit in the MWPotential does not have Jr=0'
    assert numpy.fabs(aAA.Jz(R,vR,vT,z,vz,0.)) < 10.**-16., 'Circular orbit in the MWPotential does not have Jz=0'
    #Close-to-circular orbit
    R,vR,vT,z,vz= 1.01,0.01,1.,0.01,0.01 
    js= aAA(Orbit([R,vR,vT,z,vz]))
    assert numpy.fabs(js[0]) < 10.**-4., 'Close-to-circular orbit in the MWPotential does not have small Jr'
    assert numpy.fabs(js[2]) < 10.**-3., 'Close-to-circular orbit in the MWPotential does not have small Jz'
    #Check that actions are conserved along the orbit
    obs= Orbit([1.05, 0.02, 1.05, 0.03,0.])
    check_actionAngle_conserved_actions(aAA,obs,MWPotential,
                                        -1.2,-8.,-1.7,ntimes=101)
    return None

#Basic sanity checking of the actionAngleStaeckel actions
def test_actionAngleStaeckelGrid_basic_actions_c():
    from galpy.actionAngle import actionAngleStaeckelGrid
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential, interpRZPotential
    rzpot= interpRZPotential(RZPot=MWPotential,
                             rgrid=(numpy.log(0.01),numpy.log(20.),201),
                             logR=True,
                             zgrid=(0.,1.,101),
                             interpPot=True,use_c=True,enable_c=True,
                             zsym=True)
    aAA= actionAngleStaeckelGrid(pot=rzpot,delta=0.71,c=True)
    #circular orbit
    R,vR,vT,z,vz= 1.,0.,1.,0.,0. 
    js= aAA(R,vR,vT,z,vz)
    assert numpy.fabs(js[0]) < 10.**-8., 'Circular orbit in the MWPotential does not have Jr=0'
    assert numpy.fabs(js[2]) < 10.**-8., 'Circular orbit in the MWPotential does not have Jz=0'
    #Close-to-circular orbit
    R,vR,vT,z,vz= 1.01,0.01,1.,0.01,0.01 
    js= aAA(Orbit([R,vR,vT,z,vz]))
    assert numpy.fabs(js[0]) < 10.**-4., 'Close-to-circular orbit in the MWPotential does not have small Jr'
    assert numpy.fabs(js[2]) < 10.**-3., 'Close-to-circular orbit in the MWPotentialspherical LogarithmicHalo does not have small Jz'

#Test the actions of an actionAngleStaeckel
def test_actionAngleStaeckelGrid_conserved_actions_c():
    from galpy.potential import MWPotential
    from galpy.actionAngle import actionAngleStaeckelGrid
    from galpy.orbit import Orbit
    obs= Orbit([1.05, 0.02, 1.05, 0.03,0.])
    aAA= actionAngleStaeckelGrid(pot=MWPotential,delta=0.71,c=True)
    check_actionAngle_conserved_actions(aAA,obs,MWPotential,
                                        -1.4,-8.,-1.7,ntimes=101)
    return None

#Test the setup of an actionAngleStaeckelGrid
def test_actionAngleStaeckelGrid_setuperrs():
    from galpy.potential import MWPotential
    from galpy.actionAngle import actionAngleStaeckelGrid
    try:
        aAA= actionAngleStaeckelGrid()
    except IOError: pass
    else: raise AssertionError('actionAngleStaeckelGrid w/o pot does not give IOError')
    try:
        aAA= actionAngleStaeckelGrid(pot=MWPotential)
    except IOError: pass
    else: raise AssertionError('actionAngleStaeckelGrid w/o delta does not give IOError')
    return None

#Test the actionAngleStaeckel against an isochrone potential: actions
def test_actionAngleStaeckelGrid_Isochrone_actions():
    from galpy.potential import IsochronePotential
    from galpy.actionAngle import actionAngleStaeckelGrid, \
        actionAngleIsochrone
    ip= IsochronePotential(normalize=1.,b=1.2)
    aAI= actionAngleIsochrone(ip=ip)
    aAA= actionAngleStaeckelGrid(pot=ip,delta=0.1,c=True)
    R,vR,vT,z,vz,phi= 1.01, 0.05, 1.05, 0.05,0.,2.
    ji= aAI(R,vR,vT,z,vz,phi)
    jia= aAA(R,vR,vT,z,vz,phi)
    djr= numpy.fabs((ji[0]-jia[0])/ji[0])
    dlz= numpy.fabs((ji[1]-jia[1])/ji[1])
    djz= numpy.fabs((ji[2]-jia[2])/ji[2])
    assert djr < 10.**-1.2, 'actionAngleStaeckel applied to isochrone potential fails for Jr at %f%%' % (djr*100.)
    #Lz and Jz are easy, because ip is a spherical potential
    assert dlz < 10.**-10., 'actionAngleStaeckel applied to isochrone potential fails for Lz at %f%%' % (dlz*100.)
    assert djz < 10.**-1.2, 'actionAngleStaeckel applied to isochrone potential fails for Jz at %f%%' % (djz*100.)
    return None

#Test the actionAngleIsochroneApprox against an isochrone potential: actions
def test_actionAngleIsochroneApprox_otherIsochrone_actions():
    from galpy.potential import IsochronePotential
    from galpy.actionAngle import actionAngleIsochroneApprox, \
        actionAngleIsochrone
    from galpy.orbit_src.FullOrbit import ext_loaded
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
    if not ext_loaded: #odeint is less accurate than dopr54_c
        assert djz < 10.**-6., 'actionAngleIsochroneApprox applied to isochrone potential fails for Jz at %f%%' % (djz*100.)
    else:
        assert djz < 10.**-10., 'actionAngleIsochroneApprox applied to isochrone potential fails for Jz at %f%%' % (djz*100.)
    return None

#Test the actionAngleIsochroneApprox against an isochrone potential: frequencies
def test_actionAngleIsochroneApprox_otherIsochrone_freqs():   
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
    #Same with _firstFlip, shouldn't be different bc doesn't do anything for R,vR,... input
    jiaO= aAIA.actionsFreqs(R,vR,vT,z,vz,phi,_firstFlip=True)
    dOr= numpy.fabs((jiO[3]-jiaO[3])/jiO[3])
    dOp= numpy.fabs((jiO[4]-jiaO[4])/jiO[4])
    dOz= numpy.fabs((jiO[5]-jiaO[5])/jiO[5])
    assert dOr < 10.**-6., 'actionAngleIsochroneApprox applied to isochrone potential fails for Or at %f%%' % (dOr*100.)
    assert dOp < 10.**-6., 'actionAngleIsochroneApprox applied to isochrone potential fails for Op at %f%%' % (dOp*100.)
    assert dOz < 10.**-6., 'actionAngleIsochroneApprox applied to isochrone potential fails for Oz at %f%%' % (dOz*100.)
    return None

#Test the actionAngleIsochroneApprox against an isochrone potential: angles
def test_actionAngleIsochroneApprox_otherIsochrone_angles():   
    from galpy.potential import IsochronePotential
    from galpy.actionAngle import actionAngleIsochroneApprox, \
        actionAngleIsochrone
    ip= IsochronePotential(normalize=1.,b=1.2)
    aAI= actionAngleIsochrone(ip=ip)
    aAIA= actionAngleIsochroneApprox(pot=ip,b=0.8)
    R,vR,vT,z,vz,phi= 1.1, 0.3, 1.2, 0.2,0.5,2.
    jiO= aAI.actionsFreqsAngles(R,vR,vT,z,vz,phi)
    jiaO= aAIA.actionsFreqsAngles(R,vR,vT,z,vz,phi)
    dar= numpy.fabs((jiO[6]-jiaO[6])/jiO[6])
    dap= numpy.fabs((jiO[7]-jiaO[7])/jiO[7])
    daz= numpy.fabs((jiO[8]-jiaO[8])/jiO[8])
    assert dar < 10.**-4., 'actionAngleIsochroneApprox applied to isochrone potential fails for ar at %f%%' % (dar*100.)
    assert dap < 10.**-4., 'actionAngleIsochroneApprox applied to isochrone potential fails for ap at %f%%' % (dap*100.)
    assert daz < 10.**-4., 'actionAngleIsochroneApprox applied to isochrone potential fails for az at %f%%' % (daz*100.)
    #Same with _firstFlip, shouldn't be different bc doesn't do anything for R,vR,... input
    jiaO= aAIA.actionsFreqsAngles(R,vR,vT,z,vz,phi,_firstFlip=True)
    dar= numpy.fabs((jiO[6]-jiaO[6])/jiO[6])
    dap= numpy.fabs((jiO[7]-jiaO[7])/jiO[7])
    daz= numpy.fabs((jiO[8]-jiaO[8])/jiO[8])
    assert dar < 10.**-4., 'actionAngleIsochroneApprox applied to isochrone potential fails for ar at %f%%' % (dar*100.)
    assert dap < 10.**-4., 'actionAngleIsochroneApprox applied to isochrone potential fails for ap at %f%%' % (dap*100.)
    assert daz < 10.**-4., 'actionAngleIsochroneApprox applied to isochrone potential fails for az at %f%%' % (daz*100.)
    return None

#Test the actionAngleIsochroneApprox against an isochrone potential: actions, cumul
def test_actionAngleIsochroneApprox_otherIsochrone_actions_cumul():
    from galpy.potential import IsochronePotential
    from galpy.actionAngle import actionAngleIsochroneApprox, \
        actionAngleIsochrone
    from galpy.orbit_src.FullOrbit import ext_loaded
    ip= IsochronePotential(normalize=1.,b=1.2)
    aAI= actionAngleIsochrone(ip=ip)
    aAIA= actionAngleIsochroneApprox(pot=ip,b=0.8)
    R,vR,vT,z,vz,phi= 1.1, 0.3, 1.2, 0.2,0.5,2.
    ji= aAI(R,vR,vT,z,vz,phi)
    jia= aAIA(R,vR,vT,z,vz,phi,cumul=True)
    djr= numpy.fabs((ji[0]-jia[0][0,-1])/ji[0])
    djz= numpy.fabs((ji[2]-jia[2][0,-1])/ji[2])
    assert djr < 10.**-2., 'actionAngleIsochroneApprox applied to isochrone potential fails for Jr at %f%%' % (djr*100.)
    #Lz and Jz are easy, because ip is a spherical potential
    if not ext_loaded: #odeint is less accurate than dopr54_c
        assert djz < 10.**-6., 'actionAngleIsochroneApprox applied to isochrone potential fails for Jz at %f%%' % (djz*100.)
    else:
        assert djz < 10.**-10., 'actionAngleIsochroneApprox applied to isochrone potential fails for Jz at %f%%' % (djz*100.)
    return None

#Test the actionAngleIsochroneApprox against an isochrone potential: actions; planarOrbit
def test_actionAngleIsochroneApprox_otherIsochrone_planarOrbit_actions():
    from galpy.potential import IsochronePotential
    from galpy.actionAngle import actionAngleIsochroneApprox, \
        actionAngleIsochrone
    ip= IsochronePotential(normalize=1.,b=1.2)
    aAI= actionAngleIsochrone(ip=ip)
    aAIA= actionAngleIsochroneApprox(pot=ip,b=0.8)
    R,vR,vT,phi= 1.1, 0.3, 1.2, 2.
    ji= aAI(R,vR,vT,0.,0.,phi)
    jia= aAIA(R,vR,vT,phi)
    djr= numpy.fabs((ji[0]-jia[0])/ji[0])
    dlz= numpy.fabs((ji[1]-jia[1])/ji[1])
    assert djr < 10.**-2., 'actionAngleIsochroneApprox applied to isochrone potential for planarOrbit fails for Jr at %f%%' % (djr*100.)
    #Lz and Jz are easy, because ip is a spherical potential
    assert dlz < 10.**-10., 'actionAngleIsochroneApprox applied to isochrone potential for planarOrbit fails for Lz at %f%%' % (dlz*100.)
    return None

#Test the actionAngleIsochroneApprox against an isochrone potential: actions; integrated planarOrbit
def test_actionAngleIsochroneApprox_otherIsochrone_planarOrbit_integratedOrbit_actions():
    from galpy.potential import IsochronePotential
    from galpy.actionAngle import actionAngleIsochroneApprox, \
        actionAngleIsochrone
    from galpy.orbit import Orbit
    ip= IsochronePotential(normalize=1.,b=1.2)
    aAI= actionAngleIsochrone(ip=ip)
    aAIA= actionAngleIsochroneApprox(pot=ip,b=0.8)
    R,vR,vT,phi= 1.1, 0.3, 1.2, 2.
    ji= aAI(R,vR,vT,0.,0.,phi)
    o= Orbit([R,vR,vT,phi])
    ts= numpy.linspace(0.,250.,25000)
    o.integrate(ts,ip)
    jia= aAIA(o)
    djr= numpy.fabs((ji[0]-jia[0])/ji[0])
    dlz= numpy.fabs((ji[1]-jia[1])/ji[1])
    assert djr < 10.**-2., 'actionAngleIsochroneApprox applied to isochrone potential for planarOrbit fails for Jr at %f%%' % (djr*100.)
    #Lz and Jz are easy, because ip is a spherical potential
    assert dlz < 10.**-10., 'actionAngleIsochroneApprox applied to isochrone potential for planarOrbit fails for Lz at %f%%' % (dlz*100.)
    return None

#Test the actionAngleIsochroneApprox against an isochrone potential: actions; for an integrated orbit
def test_actionAngleIsochroneApprox_otherIsochrone_integratedOrbit_actions():
    from galpy.potential import IsochronePotential
    from galpy.actionAngle import actionAngleIsochroneApprox, \
        actionAngleIsochrone
    from galpy.orbit_src.FullOrbit import ext_loaded
    from galpy.orbit import Orbit
    ip= IsochronePotential(normalize=1.,b=1.2)
    aAI= actionAngleIsochrone(ip=ip)
    aAIA= actionAngleIsochroneApprox(pot=ip,b=0.8)
    R,vR,vT,z,vz,phi= 1.1, 0.3, 1.2, 0.2,0.5,2.
    ji= aAI(R,vR,vT,z,vz,phi)
    #Setup an orbit, and integrated it first
    o= Orbit([R,vR,vT,z,vz,phi])
    ts= numpy.linspace(0.,250.,25000) #Integrate for a long time, not the default
    o.integrate(ts,ip)
    jia= aAIA(o) #actions, with an integrated orbit
    djr= numpy.fabs((ji[0]-jia[0])/ji[0])
    dlz= numpy.fabs((ji[1]-jia[1])/ji[1])
    djz= numpy.fabs((ji[2]-jia[2])/ji[2])
    assert djr < 10.**-2., 'actionAngleIsochroneApprox applied to isochrone potential fails for Jr at %f%%' % (djr*100.)
    #Lz and Jz are easy, because ip is a spherical potential
    assert dlz < 10.**-10., 'actionAngleIsochroneApprox applied to isochrone potential fails for Lz at %f%%' % (dlz*100.)
    if not ext_loaded: #odeint is less accurate than dopr54_c
        assert djz < 10.**-6., 'actionAngleIsochroneApprox applied to isochrone potential fails for Jz at %f%%' % (djz*100.)
    else:
        assert djz < 10.**-10., 'actionAngleIsochroneApprox applied to isochrone potential fails for Jz at %f%%' % (djz*100.)
    return None

#Test the actionAngleIsochroneApprox against an isochrone potential: frequencies; for an integrated orbit
def test_actionAngleIsochroneApprox_otherIsochrone_integratedOrbit_freqs():   
    from galpy.potential import IsochronePotential
    from galpy.actionAngle import actionAngleIsochroneApprox, \
        actionAngleIsochrone
    from galpy.orbit import Orbit
    ip= IsochronePotential(normalize=1.,b=1.2)
    aAI= actionAngleIsochrone(ip=ip)
    aAIA= actionAngleIsochroneApprox(pot=ip,b=0.8)
    R,vR,vT,z,vz,phi= 1.1, 0.3, 1.2, 0.2,0.5,2.
    jiO= aAI.actionsFreqs(R,vR,vT,z,vz,phi)
    #Setup an orbit, and integrated it first
    o= Orbit([R,vR,vT,z,vz,phi])
    ts= numpy.linspace(0.,250.,25000) #Integrate for a long time, not the default
    o.integrate(ts,ip)
    jiaO= aAIA.actionsFreqs([o]) #for list
    dOr= numpy.fabs((jiO[3]-jiaO[3])/jiO[3])
    dOp= numpy.fabs((jiO[4]-jiaO[4])/jiO[4])
    dOz= numpy.fabs((jiO[5]-jiaO[5])/jiO[5])
    assert dOr < 10.**-6., 'actionAngleIsochroneApprox applied to isochrone potential fails for Or at %f%%' % (dOr*100.)
    assert dOp < 10.**-6., 'actionAngleIsochroneApprox applied to isochrone potential fails for Op at %f%%' % (dOp*100.)
    assert dOz < 10.**-6., 'actionAngleIsochroneApprox applied to isochrone potential fails for Oz at %f%%' % (dOz*100.)
    #Same with specifying ts
    jiaO= aAIA.actionsFreqs(o,ts=ts)
    dOr= numpy.fabs((jiO[3]-jiaO[3])/jiO[3])
    dOp= numpy.fabs((jiO[4]-jiaO[4])/jiO[4])
    dOz= numpy.fabs((jiO[5]-jiaO[5])/jiO[5])
    assert dOr < 10.**-6., 'actionAngleIsochroneApprox applied to isochrone potential fails for Or at %f%%' % (dOr*100.)
    assert dOp < 10.**-6., 'actionAngleIsochroneApprox applied to isochrone potential fails for Op at %f%%' % (dOp*100.)
    assert dOz < 10.**-6., 'actionAngleIsochroneApprox applied to isochrone potential fails for Oz at %f%%' % (dOz*100.)
    return None

#Test the actionAngleIsochroneApprox against an isochrone potential: angles; for an integrated orbit
def test_actionAngleIsochroneApprox_otherIsochrone_integratedOrbit_angles():   
    from galpy.potential import IsochronePotential
    from galpy.actionAngle import actionAngleIsochroneApprox, \
        actionAngleIsochrone
    from galpy.orbit import Orbit
    ip= IsochronePotential(normalize=1.,b=1.2)
    aAI= actionAngleIsochrone(ip=ip)
    aAIA= actionAngleIsochroneApprox(pot=ip,b=0.8)
    R,vR,vT,z,vz,phi= 1.1, 0.3, 1.2, 0.2,0.5,2.
    jiO= aAI.actionsFreqsAngles(R,vR,vT,z,vz,phi)
    #Setup an orbit, and integrated it first
    o= Orbit([R,vR,vT,z,vz,phi])
    ts= numpy.linspace(0.,250.,25000) #Integrate for a long time, not the default
    o.integrate(ts,ip)
    jiaO= aAIA.actionsFreqsAngles(o)
    dar= numpy.fabs((jiO[6]-jiaO[6])/jiO[6])
    dap= numpy.fabs((jiO[7]-jiaO[7])/jiO[7])
    daz= numpy.fabs((jiO[8]-jiaO[8])/jiO[8])
    assert dar < 10.**-4., 'actionAngleIsochroneApprox applied to isochrone potential fails for ar at %f%%' % (dar*100.)
    assert dap < 10.**-4., 'actionAngleIsochroneApprox applied to isochrone potential fails for ap at %f%%' % (dap*100.)
    assert daz < 10.**-4., 'actionAngleIsochroneApprox applied to isochrone potential fails for az at %f%%' % (daz*100.)
    #Same with specifying ts
    jiaO= aAIA.actionsFreqsAngles(o,ts=ts)
    dar= numpy.fabs((jiO[6]-jiaO[6])/jiO[6])
    dap= numpy.fabs((jiO[7]-jiaO[7])/jiO[7])
    daz= numpy.fabs((jiO[8]-jiaO[8])/jiO[8])
    assert dar < 10.**-4., 'actionAngleIsochroneApprox applied to isochrone potential fails for ar at %f%%' % (dar*100.)
    assert dap < 10.**-4., 'actionAngleIsochroneApprox applied to isochrone potential fails for ap at %f%%' % (dap*100.)
    assert daz < 10.**-4., 'actionAngleIsochroneApprox applied to isochrone potential fails for az at %f%%' % (daz*100.)
    return None

#Check that actionAngleIsochroneApprox gives the same answer for different setups
def test_actionAngleIsochroneApprox_diffsetups(): 
    from galpy.potential import LogarithmicHaloPotential, \
        IsochronePotential
    from galpy.actionAngle import actionAngleIsochroneApprox, \
        actionAngleIsochrone
    from galpy.orbit import Orbit
    lp= LogarithmicHaloPotential(normalize=1.,q=0.9)
    #Different setups
    aAI= actionAngleIsochroneApprox(pot=lp,b=0.8)
    aAIip= actionAngleIsochroneApprox(pot=lp,
                                      ip=IsochronePotential(normalize=1.,
                                                            b=0.8))
    aAIaAIip= actionAngleIsochroneApprox(pot=lp,
                                         aAI=actionAngleIsochrone(ip=IsochronePotential(normalize=1.,
                                                                                        b=0.8)))
    aAIrk6= actionAngleIsochroneApprox(pot=lp,b=0.8,integrate_method='rk6_c')
    aAIlong= actionAngleIsochroneApprox(pot=lp,b=0.8,tintJ=200.)
    aAImany= actionAngleIsochroneApprox(pot=lp,b=0.8,ntintJ=20000)
    #Orbit to test on
    obs= Orbit([1.56148083,0.35081535,-1.15481504,
                0.88719443,-0.47713334,0.12019596])
    #Actions, frequencies, angles
    acfs= numpy.array(list(aAI.actionsFreqsAngles(obs()))).flatten()
    acfsip= numpy.array(list(aAIip.actionsFreqsAngles(obs()))).flatten()
    acfsaAIip= numpy.array(list(aAIaAIip.actionsFreqsAngles(obs()))).flatten()
    acfsrk6= numpy.array(list(aAIrk6.actionsFreqsAngles(obs()))).flatten()
    acfslong= numpy.array(list(aAIlong.actionsFreqsAngles(obs()))).flatten()
    acfsmany= numpy.array(list(aAImany.actionsFreqsAngles(obs()))).flatten()
    acfsfirstFlip= numpy.array(list(aAI.actionsFreqsAngles(obs(),_firstFlip=True))).flatten()
    #Check that they are the same
    assert numpy.amax(numpy.fabs((acfs-acfsip)/acfs)) < 10.**-16., \
        'actionAngleIsochroneApprox calculated w/ b= and ip= set to the equivalent IsochronePotential do not agree'
    assert numpy.amax(numpy.fabs((acfs-acfsaAIip)/acfs)) < 10.**-16., \
        'actionAngleIsochroneApprox calculated w/ b= and aAI= set to the equivalent IsochronePotential do not agree'
    assert numpy.amax(numpy.fabs((acfs-acfsrk6)/acfs)) < 10.**-8., \
        'actionAngleIsochroneApprox calculated w/ integrate_method=dopr54_c and rk6_c do not agree at %g%%' %(100.*numpy.amax(numpy.fabs((acfs-acfsrk6)/acfs)))
    assert numpy.amax(numpy.fabs((acfs-acfslong)/acfs)) < 10.**-2., \
        'actionAngleIsochroneApprox calculated w/ tintJ=100 and 200 do not agree at %g%%' % (100.*numpy.amax(numpy.fabs((acfs-acfslong)/acfs)))
    assert numpy.amax(numpy.fabs((acfs-acfsmany)/acfs)) < 10.**-4., \
        'actionAngleIsochroneApprox calculated w/ ntintJ=10000 and 20000 do not agree at %g%%' % (100.*numpy.amax(numpy.fabs((acfs-acfsmany)/acfs)))
    assert numpy.amax(numpy.fabs((acfs-acfsfirstFlip)/acfs)) < 10.**-4., \
        'actionAngleIsochroneApprox calculated w/ _firstFlip and w/o do not agree at %g%%' % (100.*numpy.amax(numpy.fabs((acfs-acfsmany)/acfs)))
    return None

#Check that actionAngleIsochroneApprox gives the same answer w/ and w/o firstFlip
def test_actionAngleIsochroneApprox_firstFlip(): 
    from galpy.potential import LogarithmicHaloPotential, \
        IsochronePotential
    from galpy.actionAngle import actionAngleIsochroneApprox, \
        actionAngleIsochrone
    from galpy.orbit import Orbit
    lp= LogarithmicHaloPotential(normalize=1.,q=0.9)
    aAI= actionAngleIsochroneApprox(pot=lp,b=0.8)
    #Orbit to test on
    obs= Orbit([1.56148083,0.35081535,-1.15481504,
                0.88719443,-0.47713334,0.12019596])
    #Actions, frequencies, angles
    acfs= numpy.array(list(aAI.actionsFreqsAngles(obs()))).flatten()
    acfsfirstFlip= numpy.array(list(aAI.actionsFreqsAngles(obs(),_firstFlip=True))).flatten()
    #Check that they are the same
    assert numpy.amax(numpy.fabs((acfs-acfsfirstFlip)/acfs)) < 10.**-4., \
        'actionAngleIsochroneApprox calculated w/ _firstFlip and w/o do not agree at %g%%' % (100.*numpy.amax(numpy.fabs((acfs-acfsfirstFlip)/acfs)))
    #Also test that this still works when the orbit was already integrated
    obs= Orbit([1.56148083,0.35081535,-1.15481504,
                0.88719443,-0.47713334,0.12019596])
    ts= numpy.linspace(0.,250.,25000)
    obs.integrate(ts,lp)
    acfs= numpy.array(list(aAI.actionsFreqsAngles(obs()))).flatten()
    acfsfirstFlip= numpy.array(list(aAI.actionsFreqsAngles(obs(),
                                                           _firstFlip=True))).flatten()
    #Check that they are the same
    assert numpy.amax(numpy.fabs((acfs-acfsfirstFlip)/acfs)) < 10.**-4., \
        'actionAngleIsochroneApprox calculated w/ _firstFlip and w/o do not agree at %g%%' % (100.*numpy.amax(numpy.fabs((acfs-acfsfirstFlip)/acfs)))
    return None

#Test the actionAngleIsochroneApprox used in Bovy (2014)
def test_actionAngleIsochroneApprox_bovy14():   
    from galpy.potential import LogarithmicHaloPotential
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.orbit import Orbit
    lp= LogarithmicHaloPotential(normalize=1.,q=0.9)
    aAI= actionAngleIsochroneApprox(pot=lp,b=0.8)
    obs= Orbit([1.56148083,0.35081535,-1.15481504,
                0.88719443,-0.47713334,0.12019596])
    times= numpy.linspace(0.,100.,51)
    obs.integrate(times,lp,method='dopr54_c')
    js= aAI(obs.R(times),obs.vR(times),obs.vT(times),obs.z(times),
            obs.vz(times),obs.phi(times))
    maxdj= numpy.amax(numpy.fabs((js-numpy.tile(numpy.mean(js,axis=1),(len(times),1)).T)),axis=1)/numpy.mean(js,axis=1)
    assert maxdj[0] < 3.*10.**-2., 'Jr conservation for the GD-1 like orbit of Bovy (2014) fails at %f%%' % (100.*maxdj[0])
    assert maxdj[1] < 10.**-2., 'Lz conservation for the GD-1 like orbit of Bovy (2014) fails at %f%%' % (100.*maxdj[1])
    assert maxdj[2] < 2.*10.**-2., 'Jz conservation for the GD-1 like orbit of Bovy (2014) fails at %f%%' % (100.*maxdj[2])
    return None

#Test the actionAngleIsochroneApprox for a triaxial potential
def test_actionAngleIsochroneApprox_triaxialnfw_conserved_actions():   
    from galpy.potential import TriaxialNFWPotential
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.orbit import Orbit
    tnp= TriaxialNFWPotential(b=.9,c=.8,normalize=1.)
    aAI= actionAngleIsochroneApprox(pot=tnp,b=0.8,tintJ=200.)
    obs= Orbit([1.,0.2,1.1,0.1,0.1,0.])
    check_actionAngle_conserved_actions(aAI,obs,tnp,
                                        -1.7,-2.,-1.7,ntimes=51,inclphi=True)
    return None

def test_actionAngleIsochroneApprox_triaxialnfw_linear_angles():   
    from galpy.potential import TriaxialNFWPotential
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.orbit import Orbit
    tnp= TriaxialNFWPotential(b=.9,c=.8,normalize=1.)
    aAI= actionAngleIsochroneApprox(pot=tnp,b=0.8,tintJ=200.)
    obs= Orbit([1.,0.2,1.1,0.1,0.1,0.])
    check_actionAngle_linear_angles(aAI,obs,tnp,
                                    -5.,-5.,-5.,
                                    -5.,-5.,-5.,
                                    -4.,-4.,-4.,
                                    separate_times=True, # otherwise, memory issues on travis
                                    maxt=4.,ntimes=51) # quick, essentially tests that nothing is grossly wrong
    return None

def test_actionAngleIsochroneApprox_plotting():   
    from galpy.potential import LogarithmicHaloPotential
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.orbit import Orbit
    lp= LogarithmicHaloPotential(normalize=1.,q=0.9)
    aAI= actionAngleIsochroneApprox(pot=lp,b=0.8)
    obs= Orbit([1.56148083,0.35081535,-1.15481504,
                0.88719443,-0.47713334,0.12019596])
    #Various plots that should be produced
    aAI.plot(obs)
    aAI.plot(obs,type='jr')
    aAI.plot(numpy.reshape(obs.R(obs._orb.t),(1,len(obs._orb.t))),
             numpy.reshape(obs.vR(obs._orb.t),(1,len(obs._orb.t))),
             numpy.reshape(obs.vT(obs._orb.t),(1,len(obs._orb.t))),
             numpy.reshape(obs.z(obs._orb.t),(1,len(obs._orb.t))),
             numpy.reshape(obs.vz(obs._orb.t),(1,len(obs._orb.t))),
             numpy.reshape(obs.phi(obs._orb.t),(1,len(obs._orb.t))),
             type='lz')
    aAI.plot(obs,type='jz')
    aAI.plot(obs,type='jr',downsample=True)
    aAI.plot(obs,type='lz',downsample=True)
    aAI.plot(obs,type='jz',downsample=True)
    aAI.plot(obs,type='araz')
    aAI.plot(obs,type='araz',downsample=True)
    aAI.plot(obs,type='araz',deperiod=True)
    aAI.plot(obs,type='araphi',deperiod=True)
    aAI.plot(obs,type='azaphi',deperiod=True)
    aAI.plot(obs,type='araphi',deperiod=True,downsample=True)
    aAI.plot(obs,type='azaphi',deperiod=True,downsample=True)
    #With integrated orbit, just to make sure we're covering this
    obs= Orbit([1.56148083,0.35081535,-1.15481504,
                0.88719443,-0.47713334,0.12019596])
    obs.integrate(numpy.linspace(0.,200.,20001),lp)
    aAI.plot(obs,type='jr')   
    return None

#Basic sanity checking: circular orbit should have constant R, zero vR, vT=vc
def test_actionAngleTorus_basic():
    from galpy.actionAngle import actionAngleTorus
    from galpy.potential import MWPotential, rl, vcirc, \
        FlattenedPowerPotential, PlummerPotential
    tol= -4.
    jr= 10.**-10.
    jz= 10.**-10.
    aAT= actionAngleTorus(pot=MWPotential)
    # at R=1, Lz=1
    jphi= 1.
    angler= numpy.linspace(0.,2.*numpy.pi,101)
    anglephi= numpy.linspace(0.,2.*numpy.pi,101)+1.
    anglez= numpy.linspace(0.,2.*numpy.pi,101)+2.
    RvR= aAT(jr,jphi,jz,angler,anglephi,anglez).T
    assert numpy.all(numpy.fabs(RvR[0]-rl(MWPotential,jphi)) < 10.**tol), \
        'circular orbit does not have constant radius for actionAngleTorus'
    assert numpy.all(numpy.fabs(RvR[1]) < 10.**tol), \
        'circular orbit does not have zero radial velocity for actionAngleTorus'
    assert numpy.all(numpy.fabs(RvR[2]-vcirc(MWPotential,rl(MWPotential,jphi))) < 10.**tol), \
        'circular orbit does not have constant vT=vc for actionAngleTorus'
    assert numpy.all(numpy.fabs(RvR[3]) < 10.**tol), \
        'circular orbit does not have zero vertical height for actionAngleTorus'
    assert numpy.all(numpy.fabs(RvR[4]) < 10.**tol), \
        'circular orbit does not have zero vertical velocity for actionAngleTorus'
    # at Lz=1.5, using Plummer
    tol= -3.25
    pp= PlummerPotential(normalize=1.)
    aAT= actionAngleTorus(pot=pp)
    jphi= 1.5
    RvR= aAT(jr,jphi,jz,angler,anglephi,anglez).T
    assert numpy.all(numpy.fabs(RvR[0]-rl(pp,jphi)) < 10.**tol), \
        'circular orbit does not have constant radius for actionAngleTorus'
    assert numpy.all(numpy.fabs(RvR[1]) < 10.**tol), \
        'circular orbit does not have zero radial velocity for actionAngleTorus'
    assert numpy.all(numpy.fabs(RvR[2]-vcirc(pp,rl(pp,jphi))) < 10.**tol), \
        'circular orbit does not have constant vT=vc for actionAngleTorus'
    assert numpy.all(numpy.fabs(RvR[3]) < 10.**tol), \
        'circular orbit does not have zero vertical height for actionAngleTorus'
    assert numpy.all(numpy.fabs(RvR[4]) < 10.**tol), \
        'circular orbit does not have zero vertical velocity for actionAngleTorus'
    # at Lz=0.5, using FlattenedPowerPotential
    tol= -4.
    fp= FlattenedPowerPotential(normalize=1.)
    aAT= actionAngleTorus(pot=fp)
    jphi= 0.5
    RvR= aAT(jr,jphi,jz,angler,anglephi,anglez).T
    assert numpy.all(numpy.fabs(RvR[0]-rl(fp,jphi)) < 10.**tol), \
        'circular orbit does not have constant radius for actionAngleTorus'
    assert numpy.all(numpy.fabs(RvR[1]) < 10.**tol), \
        'circular orbit does not have zero radial velocity for actionAngleTorus'
    assert numpy.all(numpy.fabs(RvR[2]-vcirc(fp,rl(fp,jphi))) < 10.**tol), \
        'circular orbit does not have constant vT=vc for actionAngleTorus'
    assert numpy.all(numpy.fabs(RvR[3]) < 10.**tol), \
        'circular orbit does not have zero vertical height for actionAngleTorus'
    assert numpy.all(numpy.fabs(RvR[4]) < 10.**tol), \
        'circular orbit does not have zero vertical velocity for actionAngleTorus'
    return None

#Basic sanity checking: close-to-circular orbit should have freq. = epicycle freq.
def test_actionAngleTorus_basic_freqs():
    from galpy.actionAngle import actionAngleTorus
    from galpy.potential import epifreq, omegac, verticalfreq, rl, \
        JaffePotential, PowerSphericalPotential, HernquistPotential
    tol= -3.
    jr= 10.**-6.
    jz= 10.**-6.
    jp= JaffePotential(normalize=1.)
    aAT= actionAngleTorus(pot=jp)
    # at Lz=1
    jphi= 1.
    om= aAT.Freqs(jr,jphi,jz)
    assert numpy.fabs((om[0]-epifreq(jp,rl(jp,jphi)))/om[0]) < 10.**tol, \
        'Close-to-circular orbit does not have Or=kappa for actionAngleTorus'
    assert numpy.fabs((om[1]-omegac(jp,rl(jp,jphi)))/om[1]) < 10.**tol, \
        'Close-to-circular orbit does not have Ophi=omega for actionAngleTorus'
    assert numpy.fabs((om[2]-verticalfreq(jp,rl(jp,jphi)))/om[2]) < 10.**tol, \
        'Close-to-circular orbit does not have Oz=nu for actionAngleTorus'
    # at Lz=1.5, w/ different potential
    pp= PowerSphericalPotential(normalize=1.)
    aAT= actionAngleTorus(pot=pp)
    jphi= 1.5
    om= aAT.Freqs(jr,jphi,jz)
    assert numpy.fabs((om[0]-epifreq(pp,rl(pp,jphi)))/om[0]) < 10.**tol, \
        'Close-to-circular orbit does not have Or=kappa for actionAngleTorus'
    assert numpy.fabs((om[1]-omegac(pp,rl(pp,jphi)))/om[1]) < 10.**tol, \
        'Close-to-circular orbit does not have Ophi=omega for actionAngleTorus'
    assert numpy.fabs((om[2]-verticalfreq(pp,rl(pp,jphi)))/om[2]) < 10.**tol, \
        'Close-to-circular orbit does not have Oz=nu for actionAngleTorus'
    # at Lz=0.5, w/ different potential
    tol= -2.5 # appears more difficult
    hp= HernquistPotential(normalize=1.)
    aAT= actionAngleTorus(pot=hp)
    jphi= 0.5
    om= aAT.Freqs(jr,jphi,jz)
    assert numpy.fabs((om[0]-epifreq(hp,rl(hp,jphi)))/om[0]) < 10.**tol, \
        'Close-to-circular orbit does not have Or=kappa for actionAngleTorus'
    assert numpy.fabs((om[1]-omegac(hp,rl(hp,jphi)))/om[1]) < 10.**tol, \
        'Close-to-circular orbit does not have Ophi=omega for actionAngleTorus'
    assert numpy.fabs((om[2]-verticalfreq(hp,rl(hp,jphi)))/om[2]) < 10.**tol, \
        'Close-to-circular orbit does not have Oz=nu for actionAngleTorus'
    return None

#Test that orbit from actionAngleTorus is the same as an integrated orbit
def test_actionAngleTorus_orbit():
    from galpy.actionAngle import actionAngleTorus
    from galpy.potential import MWPotential2014
    from galpy.orbit import Orbit
    # Set up instance
    aAT= actionAngleTorus(pot=MWPotential2014,tol=10.**-5.)
    jr,jphi,jz= 0.05,1.1,0.025
    # First calculate frequencies and the initial RvR
    RvRom= aAT.xvFreqs(jr,jphi,jz,
                       numpy.array([0.]),
                       numpy.array([1.]),
                       numpy.array([2.]))
    om= RvRom[1:]
    # Angles along an orbit
    ts= numpy.linspace(0.,100.,1001)
    angler= ts*om[0]
    anglephi= 1.+ts*om[1]
    anglez= 2.+ts*om[2]
    # Calculate the orbit using actionAngleTorus
    RvR= aAT(jr,jphi,jz,angler,anglephi,anglez).T
    # Calculate the orbit using orbit integration
    orb= Orbit([RvRom[0][:,0],RvRom[0][:,1],RvRom[0][:,2],
                RvRom[0][:,3],RvRom[0][:,4],RvRom[0][:,5]])
    orb.integrate(ts,MWPotential2014)
    # Compare
    tol= -3.
    assert numpy.all(numpy.fabs(orb.R(ts)-RvR[0]) < 10.**tol), \
        'Integrated orbit does not agree with torus orbit in R'
    assert numpy.all(numpy.fabs(orb.vR(ts)-RvR[1]) < 10.**tol), \
        'Integrated orbit does not agree with torus orbit in vR'
    assert numpy.all(numpy.fabs(orb.vT(ts)-RvR[2]) < 10.**tol), \
        'Integrated orbit does not agree with torus orbit in vT'
    assert numpy.all(numpy.fabs(orb.z(ts)-RvR[3]) < 10.**tol), \
        'Integrated orbit does not agree with torus orbit in z'
    assert numpy.all(numpy.fabs(orb.vz(ts)-RvR[4]) < 10.**tol), \
        'Integrated orbit does not agree with torus orbit in vz'
    assert numpy.all(numpy.fabs(orb.phi(ts)-RvR[5]) < 10.**tol), \
        'Integrated orbit does not agree with torus orbit in phi'
    return None

# Test that actionAngleTorus w/ interp pot gives same freqs as regular pot
# Doesn't work well: TM aborts because our interpolated forces aren't
# consistent enough with the potential for TM's taste, but we test that it at
# at least works somewhat
def test_actionAngleTorus_interppot_freqs():
    from galpy.actionAngle import actionAngleTorus
    from galpy.potential import LogarithmicHaloPotential, interpRZPotential
    lp= LogarithmicHaloPotential(normalize=1.)
    ip= interpRZPotential(RZPot=lp,
                          interpPot=True,
                          interpDens=True,interpRforce=True,interpzforce=True,
                          enable_c=True)
    aAT= actionAngleTorus(pot=lp)
    aATi= actionAngleTorus(pot=ip)
    jr,jphi,jz= 0.05,1.1,0.02
    om= aAT.Freqs(jr,jphi,jz)
    omi= aATi.Freqs(jr,jphi,jz)
    assert numpy.fabs((om[0]-omi[0])/om[0]) < 0.2, 'Radial frequency computed using the torus machine does not agree between potential and interpolated potential'
    assert numpy.fabs((om[1]-omi[1])/om[1]) < 0.2, 'Azimuthal frequency computed using the torus machine does not agree between potential and interpolated potential'
    assert numpy.fabs((om[2]-omi[2])/om[2]) < 0.8, 'Vertical frequency computed using the torus machine does not agree between potential and interpolated potential'
    return None

#Test the actionAngleTorus against an isochrone potential: actions
def test_actionAngleTorus_Isochrone_actions():
    from galpy.potential import IsochronePotential
    from galpy.actionAngle import actionAngleTorus, \
        actionAngleIsochrone
    ip= IsochronePotential(normalize=1.,b=1.2)
    aAI= actionAngleIsochrone(ip=ip)
    tol= -6.
    aAT= actionAngleTorus(pot=ip,tol=tol)
    jr,jphi,jz= 0.075,1.1,0.05
    angler= numpy.array([0.])
    anglephi= numpy.array([numpy.pi])
    anglez= numpy.array([numpy.pi/2.])
    # Calculate position from aAT
    RvR= aAT(jr,jphi,jz,angler,anglephi,anglez).T
    # Calculate actions from aAI
    ji= aAI(*RvR)
    djr= numpy.fabs((ji[0]-jr)/jr)
    dlz= numpy.fabs((ji[1]-jphi)/jphi)
    djz= numpy.fabs((ji[2]-jz)/jz)
    assert djr < 10.**tol, 'actionAngleTorus and actionAngleIsochrone applied to isochrone potential disagree for Jr at %f%%' % (djr*100.)
    assert dlz < 10.**tol, 'actionAngleTorus and actionAngleIsochrone applied to isochrone potential disagree for Jr at %f%%' % (dlz*100.) 
    assert djz < 10.**tol, 'actionAngleTorus and actionAngleIsochrone applied to isochrone potential disagree for Jr at %f%%' % (djz*100.)
    return None

#Test the actionAngleTorus against an isochrone potential: frequencies and angles
def test_actionAngleTorus_Isochrone_freqsAngles():
    from galpy.potential import IsochronePotential
    from galpy.actionAngle import actionAngleTorus, \
        actionAngleIsochrone
    ip= IsochronePotential(normalize=1.,b=1.2)
    aAI= actionAngleIsochrone(ip=ip)
    tol= -6.
    aAT= actionAngleTorus(pot=ip,tol=tol)
    jr,jphi,jz= 0.075,1.1,0.05
    angler= numpy.array([0.1])+numpy.linspace(0.,numpy.pi,101)
    angler= angler % (2.*numpy.pi)
    anglephi= numpy.array([numpy.pi])+numpy.linspace(0.,numpy.pi,101)
    anglephi= anglephi % (2.*numpy.pi)
    anglez= numpy.array([numpy.pi/2.])+numpy.linspace(0.,numpy.pi,101)
    anglez= anglez % (2.*numpy.pi)
    # Calculate position from aAT
    RvRom= aAT.xvFreqs(jr,jphi,jz,angler,anglephi,anglez)
    # Calculate actions, frequencies, and angles from aAI
    ws= aAI.actionsFreqsAngles(*RvRom[0].T)
    dOr= numpy.fabs((ws[3]-RvRom[1]))
    dOp= numpy.fabs((ws[4]-RvRom[2]))
    dOz= numpy.fabs((ws[5]-RvRom[3]))
    dar= numpy.fabs((ws[6]-angler))
    dap= numpy.fabs((ws[7]-anglephi))
    daz= numpy.fabs((ws[8]-anglez))
    dar[dar > numpy.pi]-= 2.*numpy.pi
    dar[dar < -numpy.pi]+= 2.*numpy.pi
    dap[dap > numpy.pi]-= 2.*numpy.pi
    dap[dap < -numpy.pi]+= 2.*numpy.pi
    daz[daz > numpy.pi]-= 2.*numpy.pi
    daz[daz < -numpy.pi]+= 2.*numpy.pi
    assert numpy.all(dOr < 10.**tol), 'actionAngleTorus and actionAngleIsochrone applied to isochrone potential disagree for Or at %f%%' % (numpy.nanmax(dOr)*100.)
    assert numpy.all(dOp < 10.**tol), 'actionAngleTorus and actionAngleIsochrone applied to isochrone potential disagree for Ophi at %f%%' % (numpy.nanmax(dOp)*100.) 
    assert numpy.all(dOz < 10.**tol), 'actionAngleTorus and actionAngleIsochrone applied to isochrone potential disagree for Oz at %f%%' % (numpy.nanmax(dOz)*100.)
    assert numpy.all(dar < 10.**tol), 'actionAngleTorus and actionAngleIsochrone applied to isochrone potential disagree for ar at %f' % (numpy.nanmax(dar))
    assert numpy.all(dap < 10.**tol), 'actionAngleTorus and actionAngleIsochrone applied to isochrone potential disagree for aphi at %f' % (numpy.nanmax(dap))
    assert numpy.all(daz < 10.**tol), 'actionAngleTorus and actionAngleIsochrone applied to isochrone potential disagree for az at %f' % (numpy.nanmax(daz))
    return None

#Test the actionAngleTorus against a Staeckel potential: actions
def test_actionAngleTorus_Staeckel_actions():
    from galpy.potential import KuzminKutuzovStaeckelPotential
    from galpy.actionAngle import actionAngleTorus, \
        actionAngleStaeckel
    delta= 1.2
    kp= KuzminKutuzovStaeckelPotential(normalize=1.,Delta=delta)
    aAS= actionAngleStaeckel(pot=kp,delta=delta,c=True)
    tol= -3.
    aAT= actionAngleTorus(pot=kp,tol=tol)
    jr,jphi,jz= 0.075,1.1,0.05
    angler= numpy.array([0.])
    anglephi= numpy.array([numpy.pi])
    anglez= numpy.array([numpy.pi/2.])
    # Calculate position from aAT
    RvR= aAT(jr,jphi,jz,angler,anglephi,anglez).T
    # Calculate actions from aAI
    ji= aAS(*RvR)
    djr= numpy.fabs((ji[0]-jr)/jr)
    dlz= numpy.fabs((ji[1]-jphi)/jphi)
    djz= numpy.fabs((ji[2]-jz)/jz)
    assert djr < 10.**tol, 'actionAngleTorus and actionAngleStaeckel applied to Staeckel potential disagree for Jr at %f%%' % (djr*100.)
    assert dlz < 10.**tol, 'actionAngleTorus and actionAngleStaeckel applied to Staeckel potential disagree for Jr at %f%%' % (dlz*100.) 
    assert djz < 10.**tol, 'actionAngleTorus and actionAngleStaeckel applied to Staeckel potential disagree for Jr at %f%%' % (djz*100.)
    return None

#Test the actionAngleTorus against an isochrone potential: frequencies and angles
def test_actionAngleTorus_Staeckel_freqsAngles():
    from galpy.potential import KuzminKutuzovStaeckelPotential
    from galpy.actionAngle import actionAngleTorus, \
        actionAngleStaeckel
    delta= 1.2
    kp= KuzminKutuzovStaeckelPotential(normalize=1.,Delta=delta)
    aAS= actionAngleStaeckel(pot=kp,delta=delta,c=True)
    tol= -3.
    aAT= actionAngleTorus(pot=kp,tol=tol)
    jr,jphi,jz= 0.075,1.1,0.05
    angler= numpy.array([0.1])+numpy.linspace(0.,numpy.pi,101)
    angler= angler % (2.*numpy.pi)
    anglephi= numpy.array([numpy.pi])+numpy.linspace(0.,numpy.pi,101)
    anglephi= anglephi % (2.*numpy.pi)
    anglez= numpy.array([numpy.pi/2.])+numpy.linspace(0.,numpy.pi,101)
    anglez= anglez % (2.*numpy.pi)
    # Calculate position from aAT
    RvRom= aAT.xvFreqs(jr,jphi,jz,angler,anglephi,anglez)
    # Calculate actions, frequencies, and angles from aAI
    ws= aAS.actionsFreqsAngles(*RvRom[0].T)
    dOr= numpy.fabs((ws[3]-RvRom[1]))
    dOp= numpy.fabs((ws[4]-RvRom[2]))
    dOz= numpy.fabs((ws[5]-RvRom[3]))
    dar= numpy.fabs((ws[6]-angler))
    dap= numpy.fabs((ws[7]-anglephi))
    daz= numpy.fabs((ws[8]-anglez))
    dar[dar > numpy.pi]-= 2.*numpy.pi
    dar[dar < -numpy.pi]+= 2.*numpy.pi
    dap[dap > numpy.pi]-= 2.*numpy.pi
    dap[dap < -numpy.pi]+= 2.*numpy.pi
    daz[daz > numpy.pi]-= 2.*numpy.pi
    daz[daz < -numpy.pi]+= 2.*numpy.pi
    assert numpy.all(dOr < 10.**tol), 'actionAngleTorus and actionAngleStaeckel applied to Staeckel potential disagree for Or at %f%%' % (numpy.nanmax(dOr)*100.)
    assert numpy.all(dOp < 10.**tol), 'actionAngleTorus and actionAngleStaeckel applied to Staeckel potential disagree for Ophi at %f%%' % (numpy.nanmax(dOp)*100.) 
    assert numpy.all(dOz < 10.**tol), 'actionAngleTorus and actionAngleStaeckel applied to Staeckel potential disagree for Oz at %f%%' % (numpy.nanmax(dOz)*100.)
    assert numpy.all(dar < 10.**tol), 'actionAngleTorus and actionAngleStaeckel applied to Staeckel potential disagree for ar at %f' % (numpy.nanmax(dar))
    assert numpy.all(dap < 10.**tol), 'actionAngleTorus and actionAngleStaeckel applied to Staeckel potential disagree for aphi at %f' % (numpy.nanmax(dap))
    assert numpy.all(daz < 10.**tol), 'actionAngleTorus and actionAngleStaeckel applied to Staeckel potential disagree for az at %f' % (numpy.nanmax(daz))
    return None

#Test the actionAngleTorus against a general potential w/ actionAngleIsochroneApprox: actions
def test_actionAngleTorus_isochroneApprox_actions():
    from galpy.potential import MWPotential2014
    from galpy.actionAngle import actionAngleTorus, \
        actionAngleIsochroneApprox
    aAIA= actionAngleIsochroneApprox(pot=MWPotential2014,b=0.8)
    tol= -2.5
    aAT= actionAngleTorus(pot=MWPotential2014,tol=tol)
    jr,jphi,jz= 0.075,1.1,0.05
    angler= numpy.array([0.])
    anglephi= numpy.array([numpy.pi])
    anglez= numpy.array([numpy.pi/2.])
    # Calculate position from aAT
    RvR= aAT(jr,jphi,jz,angler,anglephi,anglez).T
    # Calculate actions from aAIA
    ji= aAIA(*RvR)
    djr= numpy.fabs((ji[0]-jr)/jr)
    dlz= numpy.fabs((ji[1]-jphi)/jphi)
    djz= numpy.fabs((ji[2]-jz)/jz)
    assert djr < 10.**tol, 'actionAngleTorus and actionAngleIsochroneApprox applied to MWPotential2014 potential disagree for Jr at %f%%' % (djr*100.)
    assert dlz < 10.**tol, 'actionAngleTorus and actionAngleIsochroneApprox applied to MWPotential2014 potential disagree for Jr at %f%%' % (dlz*100.) 
    assert djz < 10.**tol, 'actionAngleTorus and actionAngleMWPotential2014 applied to MWPotential2014 potential disagree for Jr at %f%%' % (djz*100.)
    return None

#Test the actionAngleTorus against a general potential w/ actionAngleIsochrone: frequencies and angles
def test_actionAngleTorus_isochroneApprox_freqsAngles():
    from galpy.potential import MWPotential2014
    from galpy.actionAngle import actionAngleTorus, \
        actionAngleIsochroneApprox
    aAIA= actionAngleIsochroneApprox(pot=MWPotential2014,b=0.8)
    tol= -3.5
    aAT= actionAngleTorus(pot=MWPotential2014,tol=tol)
    jr,jphi,jz= 0.075,1.1,0.05
    angler= numpy.array([0.1])+numpy.linspace(0.,numpy.pi,21)
    angler= angler % (2.*numpy.pi)
    anglephi= numpy.array([numpy.pi])+numpy.linspace(0.,numpy.pi,21)
    anglephi= anglephi % (2.*numpy.pi)
    anglez= numpy.array([numpy.pi/2.])+numpy.linspace(0.,numpy.pi,21)
    anglez= anglez % (2.*numpy.pi)
    # Calculate position from aAT
    RvRom= aAT.xvFreqs(jr,jphi,jz,angler,anglephi,anglez)
    # Calculate actions, frequencies, and angles from aAI
    ws= aAIA.actionsFreqsAngles(*RvRom[0].T)
    dOr= numpy.fabs((ws[3]-RvRom[1]))
    dOp= numpy.fabs((ws[4]-RvRom[2]))
    dOz= numpy.fabs((ws[5]-RvRom[3]))
    dar= numpy.fabs((ws[6]-angler))
    dap= numpy.fabs((ws[7]-anglephi))
    daz= numpy.fabs((ws[8]-anglez))
    dar[dar > numpy.pi]-= 2.*numpy.pi
    dar[dar < -numpy.pi]+= 2.*numpy.pi
    dap[dap > numpy.pi]-= 2.*numpy.pi
    dap[dap < -numpy.pi]+= 2.*numpy.pi
    daz[daz > numpy.pi]-= 2.*numpy.pi
    daz[daz < -numpy.pi]+= 2.*numpy.pi
    assert numpy.all(dOr < 10.**tol), 'actionAngleTorus and actionAngleIsochroneApprox applied to MWPotential2014 potential disagree for Or at %f%%' % (numpy.nanmax(dOr)*100.)
    assert numpy.all(dOp < 10.**tol), 'actionAngleTorus and actionAngleIsochroneApprox applied to MWPotential2014 potential disagree for Ophi at %f%%' % (numpy.nanmax(dOp)*100.) 
    assert numpy.all(dOz < 10.**tol), 'actionAngleTorus and actionAngleIsochroneApprox applied to MWPotential2014 potential disagree for Oz at %f%%' % (numpy.nanmax(dOz)*100.)
    assert numpy.all(dar < 10.**tol), 'actionAngleTorus and actionAngleIsochroneApprox applied to MWPotential2014 potential disagree for ar at %f' % (numpy.nanmax(dar))
    assert numpy.all(dap < 10.**tol), 'actionAngleTorus and actionAngleIsochroneApprox applied to MWPotential2014 potential disagree for aphi at %f' % (numpy.nanmax(dap))
    assert numpy.all(daz < 10.**tol), 'actionAngleTorus and actionAngleIsochroneApprox applied to MWPotential2014 potential disagree for az at %f' % (numpy.nanmax(daz))
    return None

# Test that the frequencies returned by hessianFreqs are the same as those returned by Freqs
def test_actionAngleTorus_hessian_freqs():
    from galpy.potential import MWPotential2014
    from galpy.actionAngle import actionAngleTorus
    aAT= actionAngleTorus(pot=MWPotential2014)
    jr,jphi,jz= 0.075,1.1,0.05
    fO= aAT.Freqs(jr,jphi,jz)[:3]
    hO= aAT.hessianFreqs(jr,jphi,jz)[1:4]
    assert numpy.all(numpy.fabs(numpy.array(fO)-numpy.array(hO)) < 10.**-8.), 'actionAngleTorus methods Freqs and hessianFreqs return different frequencies'
    return None

# Test that the Hessian is approximately symmetric
def test_actionAngleTorus_hessian_symm():
    from galpy.potential import MWPotential2014
    from galpy.actionAngle import actionAngleTorus
    aAT= actionAngleTorus(pot=MWPotential2014)
    jr,jphi,jz= 0.075,1.1,0.05
    h= aAT.hessianFreqs(jr,jphi,jz,tol=0.0001,nosym=True)[0]
    assert numpy.all(numpy.fabs((h-h.T)/h) < 0.03), 'actionAngleTorus Hessian is not symmetric'
    return None

# Test that the Hessian is approximately correct
def test_actionAngleTorus_hessian_linear():
    from galpy.potential import MWPotential2014
    from galpy.actionAngle import actionAngleTorus
    aAT= actionAngleTorus(pot=MWPotential2014)
    jr,jphi,jz= 0.075,1.1,0.05
    h= aAT.hessianFreqs(jr,jphi,jz,tol=0.0001,nosym=True)[0]
    dj= numpy.array([0.02,0.005,-0.01])
    do_fromhessian= numpy.dot(h,dj)
    O= numpy.array(aAT.Freqs(jr,jphi,jz)[:3])
    do= numpy.array(aAT.Freqs(jr+dj[0],jphi+dj[1],jz+dj[2])[:3])-O
    assert numpy.all(numpy.fabs((do_fromhessian-do)/O)< 0.001), 'actionAngleTorus Hessian does not return good approximation to dO/dJ'
    return None

# Test that the frequencies returned by xvJacobianFreqs are the same as those returned by Freqs
def test_actionAngleTorus_jacobian_freqs():
    from galpy.potential import MWPotential2014
    from galpy.actionAngle import actionAngleTorus
    aAT= actionAngleTorus(pot=MWPotential2014)
    jr,jphi,jz= 0.075,1.1,0.05
    fO= aAT.Freqs(jr,jphi,jz)[:3]
    hO= aAT.xvJacobianFreqs(jr,jphi,jz,
                            numpy.array([0.]),numpy.array([1.]),
                            numpy.array([2.]))[3:6]
    assert numpy.all(numpy.fabs(numpy.array(fO)-numpy.array(hO)) < 10.**-8.), 'actionAngleTorus methods Freqs and xvJacobianFreqs return different frequencies'
    return None

# Test that the Hessian returned by xvJacobianFreqs are the same as those returned by hessianFreqs
def test_actionAngleTorus_jacobian_hessian():
    from galpy.potential import MWPotential2014
    from galpy.actionAngle import actionAngleTorus
    aAT= actionAngleTorus(pot=MWPotential2014)
    jr,jphi,jz= 0.075,1.1,0.05
    fO= aAT.hessianFreqs(jr,jphi,jz)[0]
    hO= aAT.xvJacobianFreqs(jr,jphi,jz,
                            numpy.array([0.]),numpy.array([1.]),
                            numpy.array([2.]))[2]
    assert numpy.all(numpy.fabs(numpy.array(fO)-numpy.array(hO)) < 10.**-8.), 'actionAngleTorus methods hessianFreqs and xvJacobianFreqs return different Hessians'
    return None

# Test that the xv returned by xvJacobianFreqs are the same as those returned by __call__
def test_actionAngleTorus_jacobian_xv():
    from galpy.potential import MWPotential2014
    from galpy.actionAngle import actionAngleTorus
    aAT= actionAngleTorus(pot=MWPotential2014)
    jr,jphi,jz= 0.075,1.1,0.05
    angler= numpy.array([0.,1.])
    anglephi= numpy.array([1.,2.])
    anglez= numpy.array([2.,3.])
    fO= aAT(jr,jphi,jz,angler,anglephi,anglez)
    hO= aAT.xvJacobianFreqs(jr,jphi,jz,angler,anglephi,anglez)[0]
    assert numpy.all(numpy.fabs(numpy.array(fO)-numpy.array(hO)) < 10.**-8.), 'actionAngleTorus methods __call__ and xvJacobianFreqs return different xv'
    return None

# Test that the determinant of the Jacobian returned by xvJacobianFreqs is close to 1/R (should be 1 for rectangular coordinates, 1/R for cylindrical
def test_actionAngleTorus_jacobian_detone():
    from galpy.potential import MWPotential2014
    from galpy.actionAngle import actionAngleTorus
    aAT= actionAngleTorus(pot=MWPotential2014,dJ=0.0001)
    jr,jphi,jz= 0.075,1.1,0.05
    angler= numpy.array([0.,1.])
    anglephi= numpy.array([1.,2.])
    anglez= numpy.array([2.,3.])
    jf= aAT.xvJacobianFreqs(jr,jphi,jz,angler,anglephi,anglez)
    assert numpy.fabs(jf[0][0,0]*numpy.fabs(numpy.linalg.det(jf[1][0]))-1) < 0.01, 'Jacobian returned by actionAngleTorus method xvJacobianFreqs does not have the expected determinant'
    assert numpy.fabs(jf[0][1,0]*numpy.fabs(numpy.linalg.det(jf[1][1]))-1) < 0.01, 'Jacobian returned by actionAngleTorus method xvJacobianFreqs does not have the expected determinant'
    return None

# Test that Jacobian returned by xvJacobianFreqs is approximately correct
def test_actionAngleTorus_jacobian_linear():
    from galpy.potential import MWPotential2014
    from galpy.actionAngle import actionAngleTorus
    aAT= actionAngleTorus(pot=MWPotential2014)
    jr,jphi,jz= 0.075,1.1,0.05
    angler= numpy.array([0.5])
    anglephi= numpy.array([1.])
    anglez= numpy.array([2.])
    jf= aAT.xvJacobianFreqs(jr,jphi,jz,angler,anglephi,anglez)
    xv= aAT(jr,jphi,jz,angler,anglephi,anglez)
    dja= 2.*numpy.array([0.001,0.002,0.003,-0.002,0.004,0.002])
    xv_direct= aAT(jr+dja[0],jphi+dja[1],jz+dja[2],
                   angler+dja[3],anglephi+dja[4],anglez+dja[5])
    xv_fromjac= xv+numpy.dot(jf[1],dja)
    assert numpy.all(numpy.fabs((xv_fromjac-xv_direct)/xv_direct) < 0.01), 'Jacobian returned by actionAngleTorus method xvJacobianFreqs does not appear to be correct'
    return None

#Test error when potential is not implemented in C
def test_actionAngleTorus_nocerr():
    from galpy.actionAngle import actionAngleTorus
    from test_potential import BurkertPotentialNoC
    bp= BurkertPotentialNoC()
    try:
        aAT= actionAngleTorus(pot=bp)
    except RuntimeError: pass
    else:
        raise AssertionError("actionAngleTorus initialization with potential w/o C should have given a RuntimeError, but didn't")
    return None

#Test error when potential is not axisymmetric
def test_actionAngleTorus_nonaxierr():
    from galpy.actionAngle import actionAngleTorus
    from galpy.potential import TriaxialNFWPotential
    np= TriaxialNFWPotential(normalize=1.,b=0.9)
    try:
        aAT= actionAngleTorus(pot=np)
    except RuntimeError: pass
    else:
        raise AssertionError("actionAngleTorus initialization with non-axisymmetric potential should have given a RuntimeError, but didn't")
    return None

# Test the Autofit torus warnings
def test_actionAngleTorus_AutoFitWarning():
    from galpy.potential import LogarithmicHaloPotential
    from galpy.actionAngle import actionAngleTorus
    lp= LogarithmicHaloPotential(normalize=1.,q=0.9)
    aAT= actionAngleTorus(pot=lp,tol=10.**-8.)
    # These should give warnings
    jr, jp, jz= 0.27209033, 1.80253892, 0.6078445
    ar, ap, az= numpy.array([1.95732492]), numpy.array([6.16753224]), \
        numpy.array([4.08233059])
    #Turn warnings into errors to test for them
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always",galpyWarning)
        aAT(jr,jp,jz,ar,ap,az)
        # Should raise warning bc of Autofit, might raise others
        raisedWarning= False
        for wa in w:
            raisedWarning= (str(wa.message) == "actionAngleTorus' AutoFit exited with non-zero return status -3: Fit failed the goal by more than 2")
            if raisedWarning: break
        assert raisedWarning, "actionAngleTorus with flattened LogarithmicHaloPotential and a particular orbit should have thrown a warning, but didn't"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always",galpyWarning)
        aAT.xvFreqs(jr,jp,jz,ar,ap,az)
        # Should raise warning bc of Autofit, might raise others
        raisedWarning= False
        for wa in w:
            raisedWarning= (str(wa.message) == "actionAngleTorus' AutoFit exited with non-zero return status -3: Fit failed the goal by more than 2")
            if raisedWarning: break
        assert raisedWarning, "actionAngleTorus with flattened LogarithmicHaloPotential and a particular orbit should have thrown a warning, but didn't"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always",galpyWarning)
        aAT.Freqs(jr,jp,jz)
        # Should raise warning bc of Autofit, might raise others
        raisedWarning= False
        for wa in w:
            raisedWarning= (str(wa.message) == "actionAngleTorus' AutoFit exited with non-zero return status -3: Fit failed the goal by more than 2")
            if raisedWarning: break
        assert raisedWarning, "actionAngleTorus with flattened LogarithmicHaloPotential and a particular orbit should have thrown a warning, but didn't"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always",galpyWarning)
        aAT.hessianFreqs(jr,jp,jz)
        # Should raise warning bc of Autofit, might raise others
        raisedWarning= False
        for wa in w:
            raisedWarning= (str(wa.message) == "actionAngleTorus' AutoFit exited with non-zero return status -3: Fit failed the goal by more than 2")
            if raisedWarning: break
        assert raisedWarning, "actionAngleTorus with flattened LogarithmicHaloPotential and a particular orbit should have thrown a warning, but didn't"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always",galpyWarning)
        aAT.xvJacobianFreqs(jr,jp,jz,ar,ap,az)
        # Should raise warning bc of Autofit, might raise others
        raisedWarning= False
        for wa in w:
            raisedWarning= (str(wa.message) == "actionAngleTorus' AutoFit exited with non-zero return status -3: Fit failed the goal by more than 2")
            if raisedWarning: break
        assert raisedWarning, "actionAngleTorus with flattened LogarithmicHaloPotential and a particular orbit should have thrown a warning, but didn't"
    return None

#Test the Orbit interface
def test_orbit_interface_spherical():
    from galpy.potential import LogarithmicHaloPotential, NFWPotential
    from galpy.orbit import Orbit
    from galpy.actionAngle import actionAngleSpherical
    lp= LogarithmicHaloPotential(normalize=1.,q=1.)
    obs= Orbit([1., 0.2, 1.5, 0.3,0.1,2.])
    assert not obs.resetaA(), 'obs.resetaA() does not return False when called before having set up an actionAngle instance'
    aAS= actionAngleSpherical(pot=lp)
    acfs= numpy.array(list(aAS.actionsFreqsAngles(obs))).reshape(9)
    type= 'spherical'
    acfso= numpy.array([obs.jr(pot=lp,type=type),
                        obs.jp(pot=lp,type=type),
                        obs.jz(pot=lp,type=type),
                        obs.Or(pot=lp,type=type),
                        obs.Op(pot=lp,type=type),
                        obs.Oz(pot=lp,type=type),
                        obs.wr(pot=lp,type=type),
                        obs.wp(pot=lp,type=type),
                        obs.wz(pot=lp,type=type)])
    maxdev= numpy.amax(numpy.abs(acfs-acfso))
    assert maxdev < 10.**-16., 'Orbit interface for actionAngleSpherical does not return the same as actionAngle interface'
    assert numpy.abs(obs.Tr(pot=lp,type=type)-2.*numpy.pi/acfs[3]) < 10.**-16., \
        'Orbit.Tr does not agree with actionAngleSpherical frequency'
    assert numpy.abs(obs.Tp(pot=lp,type=type)-2.*numpy.pi/acfs[4]) < 10.**-16., \
        'Orbit.Tp does not agree with actionAngleSpherical frequency'
    assert numpy.abs(obs.Tz(pot=lp,type=type)-2.*numpy.pi/acfs[5]) < 10.**-16., \
        'Orbit.Tz does not agree with actionAngleSpherical frequency'
    assert numpy.abs(obs.TrTp(pot=lp,type=type)-acfs[4]/acfs[3]*numpy.pi) < 10.**-16., \
        'Orbit.TrTp does not agree with actionAngleSpherical frequency'
    #Different spherical potential
    np= NFWPotential(normalize=1.)
    aAS= actionAngleSpherical(pot=np)
    acfs= numpy.array(list(aAS.actionsFreqsAngles(obs))).reshape(9)
    type= 'spherical'
    assert obs.resetaA(pot=np), 'obs.resetaA() does not return True after having set up an actionAngle instance'
    try:
        obs.jr(type=type)
    except AttributeError:
        pass #should raise this, as we have not specified a potential
    else:
        raise AssertionError('obs.jr w/o pot= does not raise AttributeError before the orbit was integrated')
    obs.integrate(numpy.linspace(0.,1.,11),np) #to test that not specifying the potential works
    acfso= numpy.array([obs.jr(type=type),
                        obs.jp(type=type),
                        obs.jz(type=type),
                        obs.Or(type=type),
                        obs.Op(type=type),
                        obs.Oz(type=type),
                        obs.wr(type=type),
                        obs.wp(type=type),
                        obs.wz(type=type)])
    maxdev= numpy.amax(numpy.abs(acfs-acfso))
    assert maxdev < 10.**-16., 'Orbit interface for actionAngleSpherical does not return the same as actionAngle interface'   
    #Directly test _resetaA
    assert obs._orb._resetaA(pot=lp), 'OrbitTop._resetaA does not return True when resetting the actionAngle instance'
    #Test that unit conversions to physical units are handled correctly
    ro, vo=8., 220.
    obs= Orbit([1., 0.2, 1.5, 0.3,0.1,2.],ro=ro,vo=vo)
    aAS= actionAngleSpherical(pot=lp)
    acfs= numpy.array(list(aAS.actionsFreqsAngles(obs))).reshape(9)
    type= 'spherical'
    acfso= numpy.array([obs.jr(pot=lp,type=type)/ro/vo,
                        obs.jp(pot=lp,type=type)/ro/vo,
                        obs.jz(pot=lp,type=type)/ro/vo,
                        obs.Or(pot=lp,type=type)/vo*ro/1.0227121655399913,
                        obs.Op(pot=lp,type=type)/vo*ro/1.0227121655399913,
                        obs.Oz(pot=lp,type=type)/vo*ro/1.0227121655399913,
                        obs.wr(pot=lp,type=type),
                        obs.wp(pot=lp,type=type),
                        obs.wz(pot=lp,type=type)])
    maxdev= numpy.amax(numpy.abs(acfs-acfso))
    assert maxdev < 10.**-9., 'Orbit interface for actionAngleSpherical does not return the same as actionAngle interface when using physical coordinates'
    assert numpy.abs(obs.Tr(pot=lp,type=type)/ro*vo*1.0227121655399913-2.*numpy.pi/acfs[3]) < 10.**-8., \
        'Orbit.Tr does not agree with actionAngleSpherical frequency when using physical coordinates'
    assert numpy.abs(obs.Tp(pot=lp,type=type)/ro*vo*1.0227121655399913-2.*numpy.pi/acfs[4]) < 10.**-8., \
        'Orbit.Tp does not agree with actionAngleSpherical frequency when using physical coordinates'
    assert numpy.abs(obs.Tz(pot=lp,type=type)/ro*vo*1.0227121655399913-2.*numpy.pi/acfs[5]) < 10.**-8., \
        'Orbit.Tz does not agree with actionAngleSpherical frequency when using physical coordinates'
    assert numpy.abs(obs.TrTp(pot=lp,type=type)-acfs[4]/acfs[3]*numpy.pi) < 10.**-8., \
        'Orbit.TrTp does not agree with actionAngleSpherical frequency when using physical coordinates'
    #Test frequency in km/s/kpc
    assert numpy.abs(obs.Or(pot=lp,type=type,kmskpc=True)/vo*ro-acfs[3]) < 10.**-8., \
        'Orbit.Or does not agree with actionAngleSpherical frequency when using physical coordinates with km/s/kpc'
    assert numpy.abs(obs.Op(pot=lp,type=type,kmskpc=True)/vo*ro-acfs[4]) < 10.**-8., \
        'Orbit.Op does not agree with actionAngleSpherical frequency when using physical coordinates with km/s/kpc'
    assert numpy.abs(obs.Oz(pot=lp,type=type,kmskpc=True)/vo*ro-acfs[5]) < 10.**-8., \
        'Orbit.Oz does not agree with actionAngleSpherical frequency when using physical coordinates with km/s/kpc'
    return None

# Test the Orbit interface for actionAngleStaeckel
def test_orbit_interface_staeckel():
    from galpy.potential import MWPotential
    from galpy.orbit import Orbit
    from galpy.actionAngle import actionAngleStaeckel
    obs= Orbit([1.05, 0.02, 1.05, 0.03,0.,2.])
    aAS= actionAngleStaeckel(pot=MWPotential,delta=0.71)
    acfs= numpy.array(list(aAS.actionsFreqsAngles(obs))).reshape(9)
    type= 'staeckel'
    acfso= numpy.array([obs.jr(pot=MWPotential,type=type,delta=0.71),
                        obs.jp(pot=MWPotential,type=type),
                        obs.jz(pot=MWPotential,type=type),
                        obs.Or(pot=MWPotential,type=type),
                        obs.Op(pot=MWPotential,type=type),
                        obs.Oz(pot=MWPotential,type=type),
                        obs.wr(pot=MWPotential,type=type),
                        obs.wp(pot=MWPotential,type=type),
                        obs.wz(pot=MWPotential,type=type)])
    maxdev= numpy.amax(numpy.abs(acfs-acfso))
    assert maxdev < 10.**-16., 'Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface'
    return None

# Test the Orbit interface for actionAngleAdiabatic
def test_orbit_interface_adiabatic():
    from galpy.potential import MWPotential
    from galpy.orbit import Orbit
    from galpy.actionAngle import actionAngleAdiabatic
    obs= Orbit([1.05, 0.02, 1.05, 0.03,0.,2.])
    aAS= actionAngleAdiabatic(pot=MWPotential)
    acfs= numpy.array(list(aAS(obs))).reshape(3)
    type= 'adiabatic'
    acfso= numpy.array([obs.jr(pot=MWPotential,type=type),
                        obs.jp(pot=MWPotential,type=type),
                        obs.jz(pot=MWPotential,type=type)])
    maxdev= numpy.amax(numpy.abs(acfs-acfso))
    assert maxdev < 10.**-16., 'Orbit interface for actionAngleAdiabatic does not return the same as actionAngle interface'
    return None

def test_orbit_interface_actionAngleIsochroneApprox():
    from galpy.potential import MWPotential
    from galpy.orbit import Orbit
    from galpy.actionAngle import actionAngleIsochroneApprox
    obs= Orbit([1.05, 0.02, 1.05, 0.03,0.,2.])
    aAS= actionAngleIsochroneApprox(pot=MWPotential,b=0.8)
    acfs= list(aAS(obs))
    acfs.extend(list(aAS.actionsFreqsAngles([obs()]))[3:])
    acfs= numpy.array(acfs).reshape(9)
    type= 'isochroneApprox'
    acfso= numpy.array([obs.jr(pot=MWPotential,type=type,b=0.8),
                        obs.jp(pot=MWPotential,type=type),
                        obs.jz(pot=MWPotential,type=type),
                        obs.Or(pot=MWPotential,type=type),
                        obs.Op(pot=MWPotential,type=type),
                        obs.Oz(pot=MWPotential,type=type),
                        obs.wr(pot=MWPotential,type=type),
                        obs.wp(pot=MWPotential,type=type),
                        obs.wz(pot=MWPotential,type=type)])
    maxdev= numpy.amax(numpy.abs(acfs-acfso))
    assert maxdev < 10.**-16., 'Orbit interface for actionAngleIsochroneApproxStaeckel does not return the same as actionAngle interface'
    assert numpy.abs(obs.Tr(pot=MWPotential,type=type)-2.*numpy.pi/acfso[3]) < 10.**-16., \
        'Orbit.Tr does not agree with actionAngleSpherical frequency'
    assert numpy.abs(obs.Tp(pot=MWPotential,type=type)-2.*numpy.pi/acfso[4]) < 10.**-16., \
        'Orbit.Tp does not agree with actionAngleSpherical frequency'
    assert numpy.abs(obs.Tz(pot=MWPotential,type=type)-2.*numpy.pi/acfso[5]) < 10.**-16., \
        'Orbit.Tz does not agree with actionAngleSpherical frequency'
    assert numpy.abs(obs.TrTp(pot=MWPotential,type=type)-acfso[4]/acfso[3]*numpy.pi) < 10.**-16., \
        'Orbit.TrTp does not agree with actionAngleSpherical frequency'
    return None

# Test physical output for actionAngleStaeckel
def test_physical_staeckel():
    from galpy.potential import MWPotential
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.util import bovy_conversion
    ro,vo= 7., 230.
    aA= actionAngleStaeckel(pot=MWPotential,delta=0.71,ro=ro,vo=vo)
    aAnu= actionAngleStaeckel(pot=MWPotential,delta=0.71)
    for ii in range(3):
        assert numpy.fabs(aA(1.1,0.1,1.1,0.1,0.2,0.)[ii]-aAnu(1.1,0.1,1.1,0.1,0.2,0.)[ii]*ro*vo) < 10.**-8., 'actionAngle function __call__ does not return Quantity with the right value'
    for ii in range(3):
        assert numpy.fabs(aA.actionsFreqs(1.1,0.1,1.1,0.1,0.2,0.)[ii]-aAnu.actionsFreqs(1.1,0.1,1.1,0.1,0.2,0.)[ii]*ro*vo) < 10.**-8., 'actionAngle function actionsFreqs does not return Quantity with the right value'
    for ii in range(3,6):
        assert numpy.fabs(aA.actionsFreqs(1.1,0.1,1.1,0.1,0.2,0.)[ii]-aAnu.actionsFreqs(1.1,0.1,1.1,0.1,0.2,0.)[ii]*bovy_conversion.freq_in_Gyr(vo,ro)) < 10.**-8., 'actionAngle function actionsFreqs does not return Quantity with the right value'
    for ii in range(3):
        assert numpy.fabs(aA.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii]-aAnu.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii]*ro*vo) < 10.**-8., 'actionAngle function actionsFreqsAngles does not return Quantity with the right value'
    for ii in range(3,6):
        assert numpy.fabs(aA.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii]-aAnu.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii]*bovy_conversion.freq_in_Gyr(vo,ro)) < 10.**-8., 'actionAngle function actionsFreqsAngles does not return Quantity with the right value'
    for ii in range(6,9):
        assert numpy.fabs(aA.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii]-aAnu.actionsFreqsAngles(1.1,0.1,1.1,0.1,0.2,0.)[ii]) < 10.**-8., 'actionAngle function actionsFreqsAngles does not return Quantity with the right value'
    return None

#Test the b estimation
def test_estimateBIsochrone():
    from galpy.potential import IsochronePotential
    from galpy.actionAngle import estimateBIsochrone
    from galpy.orbit import Orbit
    ip= IsochronePotential(normalize=1.,b=1.2)
    o= Orbit([1.1, 0.3, 1.2, 0.2,0.5,2.])
    times= numpy.linspace(0.,100.,1001)
    o.integrate(times,ip)
    bmin, bmed, bmax= estimateBIsochrone(ip,o.R(times),o.z(times))
    assert numpy.fabs(bmed-1.2) < 10.**-15., \
        'Estimated scale parameter b when estimateBIsochrone is applied to an IsochronePotential is wrong'
    return None

#Test the focal delta estimation
def test_estimateDeltaStaeckel():
    from galpy.potential import MWPotential
    from galpy.actionAngle import estimateDeltaStaeckel
    from galpy.orbit import Orbit
    o= Orbit([1.1, 0.05, 1.1, 0.05,0.,2.])
    times= numpy.linspace(0.,100.,1001)
    o.integrate(times,MWPotential)
    delta= estimateDeltaStaeckel(MWPotential,o.R(times),o.z(times))
    assert numpy.fabs(delta-0.71) < 10.**-3., \
        'Estimated focal parameter delta when estimateDeltaStaeckel is applied to the MWPotential is wrong'
    return None

#Test the focal delta estimation
def test_estimateDeltaStaeckel_spherical():
    from galpy.potential import LogarithmicHaloPotential
    from galpy.actionAngle import estimateDeltaStaeckel
    from galpy.orbit import Orbit
    o= Orbit([1.1, 0.05, 1.1, 0.05,0.,2.])
    times= numpy.linspace(0.,100.,1001)
    lp= LogarithmicHaloPotential(normalize=1.,q=1.)
    o.integrate(times,lp)
    delta= estimateDeltaStaeckel(lp,o.R(),o.z())
    assert numpy.fabs(delta) < 10.**-6., \
        'Estimated focal parameter delta when estimateDeltaStaeckel is applied to a spherical potential is wrong'
    delta= estimateDeltaStaeckel(lp,o.R(times),o.z(times))
    assert numpy.fabs(delta) < 10.**-16., \
        'Estimated focal parameter delta when estimateDeltaStaeckel is applied to a spherical potential is wrong'
    return None

# Test that setting up the non-spherical actionAngle routines raises a warning when using MWPotential, see #229
def test_MWPotential_warning_adiabatic():
    # Test that using MWPotential throws a warning, see #229
    from galpy.actionAngle import actionAngleAdiabatic, \
        actionAngleAdiabaticGrid
    from galpy.potential import MWPotential
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always",galpyWarning)
        aAA= actionAngleAdiabatic(pot=MWPotential,gamma=1.)
        # Should raise warning bc of MWPotential, might raise others
        raisedWarning= False
        for wa in w:
            raisedWarning= (str(wa.message) == "Use of MWPotential as a Milky-Way-like potential is deprecated; galpy.potential.MWPotential2014, a potential fit to a large variety of dynamical constraints (see Bovy 2015), is the preferred Milky-Way-like potential in galpy")
            if raisedWarning: break
        assert raisedWarning, "actionAngleAdiabatic with MWPotential should have thrown a warning, but didn't"
    #Grid
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always",galpyWarning)
        aAA= actionAngleAdiabaticGrid(pot=MWPotential,gamma=1.,nEz=5,nEr=5,
                                      nLz=5,nR=5)
        # Should raise warning bc of MWPotential, might raise others
        raisedWarning= False
        for wa in w:
            raisedWarning= (str(wa.message) == "Use of MWPotential as a Milky-Way-like potential is deprecated; galpy.potential.MWPotential2014, a potential fit to a large variety of dynamical constraints (see Bovy 2015), is the preferred Milky-Way-like potential in galpy")
            if raisedWarning: break
        assert raisedWarning, "actionAngleAdiabaticGrid with MWPotential should have thrown a warning, but didn't"
    return None

def test_MWPotential_warning_staeckel():
    # Test that using MWPotential throws a warning, see #229
    from galpy.actionAngle import actionAngleStaeckel, \
        actionAngleStaeckelGrid
    from galpy.potential import MWPotential
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always",galpyWarning)
        aAA= actionAngleStaeckel(pot=MWPotential,delta=0.5)
        # Should raise warning bc of MWPotential, might raise others
        raisedWarning= False
        for wa in w:
            raisedWarning= (str(wa.message) == "Use of MWPotential as a Milky-Way-like potential is deprecated; galpy.potential.MWPotential2014, a potential fit to a large variety of dynamical constraints (see Bovy 2015), is the preferred Milky-Way-like potential in galpy")
            if raisedWarning: break
        assert raisedWarning, "actionAngleStaeckel with MWPotential should have thrown a warning, but didn't"
    #Grid
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always",galpyWarning)
        aAA= actionAngleStaeckelGrid(pot=MWPotential,delta=0.5,
                                     nE=5,npsi=5,nLz=5)
        # Should raise warning bc of MWPotential, might raise others
        raisedWarning= False
        for wa in w:
            raisedWarning= (str(wa.message) == "Use of MWPotential as a Milky-Way-like potential is deprecated; galpy.potential.MWPotential2014, a potential fit to a large variety of dynamical constraints (see Bovy 2015), is the preferred Milky-Way-like potential in galpy")
            if raisedWarning: break
        assert raisedWarning, "actionAngleStaeckelGrid with MWPotential should have thrown a warning, but didn't"
    return None

def test_MWPotential_warning_isochroneapprox():
    # Test that using MWPotential throws a warning, see #229
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.potential import MWPotential
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always",galpyWarning)
        aAA= actionAngleIsochroneApprox(pot=MWPotential,b=1.)
        # Should raise warning bc of MWPotential, might raise others
        raisedWarning= False
        for wa in w:
            raisedWarning= (str(wa.message) == "Use of MWPotential as a Milky-Way-like potential is deprecated; galpy.potential.MWPotential2014, a potential fit to a large variety of dynamical constraints (see Bovy 2015), is the preferred Milky-Way-like potential in galpy")
            if raisedWarning: break
        assert raisedWarning, "actionAngleIsochroneApprox with MWPotential should have thrown a warning, but didn't"
    return None

def test_MWPotential_warning_torus():
    # Test that using MWPotential throws a warning, see #229
    from galpy.actionAngle import actionAngleTorus
    from galpy.potential import MWPotential
    warnings.simplefilter("error",galpyWarning)
    try:
        aAA= actionAngleTorus(pot=MWPotential)
    except: pass
    else:
        raise AssertionError("actionAngleTorus with MWPotential should have thrown a warning, but didn't")
    #Turn warnings back into warnings
    warnings.simplefilter("always",galpyWarning)
    return None

#Test that the actions are conserved along an orbit
def check_actionAngle_conserved_actions(aA,obs,pot,toljr,toljp,toljz,
                                        ntimes=1001,fixed_quad=False,
                                        inclphi=False):
    times= numpy.linspace(0.,100.,ntimes)
    obs.integrate(times,pot,method='dopr54_c')
    if fixed_quad and inclphi:
        js= aA(obs.R(times),obs.vR(times),obs.vT(times),obs.z(times),
               obs.vz(times),obs.phi(times),fixed_quad=True)
    elif fixed_quad and not inclphi:
        js= aA(obs.R(times),obs.vR(times),obs.vT(times),obs.z(times),
               obs.vz(times),fixed_quad=True)
    elif inclphi:
        js= aA(obs.R(times),obs.vR(times),obs.vT(times),obs.z(times),
               obs.vz(times),obs.phi(times))
    else:
        js= aA(obs.R(times),obs.vR(times),obs.vT(times),obs.z(times),
               obs.vz(times))
    maxdj= numpy.amax(numpy.fabs((js-numpy.tile(numpy.mean(js,axis=1),(len(times),1)).T)),axis=1)/numpy.mean(js,axis=1)
    assert maxdj[0] < 10.**toljr, 'Jr conservation fails at %g%%' % (100.*maxdj[0])
    assert maxdj[1] < 10.**toljp, 'Lz conservation fails at %g%%' % (100.*maxdj[1])
    assert maxdj[2] < 10.**toljz, 'Jz conservation fails at %g%%' % (100.*maxdj[2])
    return None

#Test that the angles increase linearly
def check_actionAngle_linear_angles(aA,obs,pot,
                                    tolinitar,tolinitap,tolinitaz,
                                    tolor,tolop,toloz,
                                    toldar,toldap,toldaz,
                                    maxt=100.,ntimes=1001,separate_times=False,
                                    fixed_quad=False,
                                    u0=None):
    from galpy.actionAngle import dePeriod
    times= numpy.linspace(0.,maxt,ntimes)
    obs.integrate(times,pot,method='dopr54_c')
    if fixed_quad:
        acfs_init= aA.actionsFreqsAngles(obs,fixed_quad=True) #to check the init. angles
        acfs= aA.actionsFreqsAngles(obs.R(times),obs.vR(times),obs.vT(times),
                                    obs.z(times),obs.vz(times),obs.phi(times),
                                    fixed_quad=True)
    elif not u0 is None:
        acfs_init= aA.actionsFreqsAngles(obs,u0=u0) #to check the init. angles
        acfs= aA.actionsFreqsAngles(obs.R(times),obs.vR(times),obs.vT(times),
                                    obs.z(times),obs.vz(times),obs.phi(times),
                                    u0=(u0+times*0.)) #array
    else:
        acfs_init= aA.actionsFreqsAngles(obs()) #to check the init. angles
        if separate_times:
            acfs= numpy.array([aA.actionsFreqsAngles(obs.R(t),obs.vR(t),
                                                     obs.vT(t),obs.z(t),
                                                     obs.vz(t),obs.phi(t))
                               for t in times])[:,:,0].T
            acfs= (acfs[0],acfs[1],acfs[2],
                   acfs[3],acfs[4],acfs[5],
                   acfs[6],acfs[7],acfs[8])
        else:
            acfs= aA.actionsFreqsAngles(obs.R(times),obs.vR(times),
                                        obs.vT(times),obs.z(times),
                                        obs.vz(times),obs.phi(times))
    ar= dePeriod(numpy.reshape(acfs[6],(1,len(times)))).flatten()
    ap= dePeriod(numpy.reshape(acfs[7],(1,len(times)))).flatten()
    az= dePeriod(numpy.reshape(acfs[8],(1,len(times)))).flatten()
    # Do linear fit to radial angle, check that deviations are small, check 
    # that the slope is the frequency
    linfit= numpy.polyfit(times,ar,1)
    assert numpy.fabs((linfit[1]-acfs_init[6])/acfs_init[6]) < 10.**tolinitar, \
        'Radial angle obtained by fitting linear trend to the orbit does not agree with the initially-calculated angle by %g%%' % (100.*numpy.fabs((linfit[1]-acfs_init[6])/acfs_init[6]))
    assert numpy.fabs(linfit[0]-acfs_init[3]) < 10.**tolor, \
        'Radial frequency obtained by fitting linear trend to the orbit does not agree with the initially-calculated frequency by %g%%' % (100.*numpy.fabs((linfit[0]-acfs_init[3])/acfs_init[3]))
    devs= (ar-linfit[0]*times-linfit[1])
    maxdev= numpy.amax(numpy.fabs(devs))
    assert maxdev < 10.**toldar, 'Maximum deviation from linear trend in the radial angles is %g' % maxdev
    # Do linear fit to azimuthal angle, check that deviations are small, check 
    # that the slope is the frequency
    linfit= numpy.polyfit(times,ap,1)
    assert numpy.fabs((linfit[1]-acfs_init[7])/acfs_init[7]) < 10.**tolinitap, \
        'Azimuthal angle obtained by fitting linear trend to the orbit does not agree with the initially-calculated angle by %g%%' % (100.*numpy.fabs((linfit[1]-acfs_init[7])/acfs_init[7]))
    assert numpy.fabs(linfit[0]-acfs_init[4]) < 10.**tolop, \
        'Azimuthal frequency obtained by fitting linear trend to the orbit does not agree with the initially-calculated frequency by %g%%' % (100.*numpy.fabs((linfit[0]-acfs_init[4])/acfs_init[4]))
    devs= (ap-linfit[0]*times-linfit[1])
    maxdev= numpy.amax(numpy.fabs(devs))
    assert maxdev < 10.**toldap, 'Maximum deviation from linear trend in the azimuthal angle is %g' % maxdev
    # Do linear fit to vertical angle, check that deviations are small, check 
    # that the slope is the frequency
    linfit= numpy.polyfit(times,az,1)
    assert numpy.fabs((linfit[1]-acfs_init[8])/acfs_init[8]) < 10.**tolinitaz, \
        'Vertical angle obtained by fitting linear trend to the orbit does not agree with the initially-calculated angle by %g%%' % (100.*numpy.fabs((linfit[1]-acfs_init[8])/acfs_init[8]))
    assert numpy.fabs(linfit[0]-acfs_init[5]) < 10.**toloz, \
        'Vertical frequency obtained by fitting linear trend to the orbit does not agree with the initially-calculated frequency by %g%%' % (100.*numpy.fabs((linfit[0]-acfs_init[5])/acfs_init[5]))
    devs= (az-linfit[0]*times-linfit[1])
    maxdev= numpy.amax(numpy.fabs(devs))
    assert maxdev < 10.**toldaz, 'Maximum deviation from linear trend in the vertical angles is %g' % maxdev
    return None

#Test that the ecc, zmax, rperi, rap are conserved along an orbit
def check_actionAngle_conserved_EccZmaxRperiRap(aA,obs,pot,tole,tolzmax,
                                                tolrperi,tolrap,
                                                ntimes=1001):
    times= numpy.linspace(0.,100.,ntimes)
    obs.integrate(times,pot,method='dopr54_c')
    es,zmaxs,rperis,raps= aA.EccZmaxRperiRap(\
        obs.R(times),obs.vR(times),obs.vT(times),obs.z(times),
        obs.vz(times),obs.phi(times))
    assert numpy.amax(numpy.fabs(es/numpy.mean(es)-1)) < 10.**tole, 'Eccentricity conservation fails at %g%%' % (100.*numpy.amax(numpy.fabs(es/numpy.mean(es)-1)))
    assert numpy.amax(numpy.fabs(zmaxs/numpy.mean(zmaxs)-1)) < 10.**tolzmax, 'Zmax conservation fails at %g%%' % (100.*numpy.amax(numpy.fabs(zmaxs/numpy.mean(zmaxs)-1)))
    assert numpy.amax(numpy.fabs(rperis/numpy.mean(rperis)-1)) < 10.**tolrperi, 'Rperi conservation fails at %g%%' % (100.*numpy.amax(numpy.fabs(rperis/numpy.mean(rperis)-1)))
    assert numpy.amax(numpy.fabs(raps/numpy.mean(raps)-1)) < 10.**tolrap, 'Rap conservation fails at %g%%' % (100.*numpy.amax(numpy.fabs(raps/numpy.mean(raps)-1)))
    return None

