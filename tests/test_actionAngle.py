from __future__ import print_function, division
import os
import sys
import pytest
import warnings
import numpy
from galpy.util import galpyWarning
_TRAVIS= bool(os.getenv('TRAVIS'))
PY2= sys.version < '3'
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

# Test that EccZmaxRperiRap for an IsochronePotential are correctly computed
# by comparing to a numerical orbit integration
def test_actionAngleIsochrone_EccZmaxRperiRap_againstOrbit():
    from galpy.potential import IsochronePotential
    from galpy.orbit import Orbit
    from galpy.actionAngle import actionAngleIsochrone
    ip= IsochronePotential(normalize=1.,b=1.2)
    aAI= actionAngleIsochrone(ip=ip)
    o= Orbit([1.,0.1,1.1,0.2,0.03,0.])
    ecc, zmax, rperi, rap= aAI.EccZmaxRperiRap(o)
    ts= numpy.linspace(0.,100.,100001)
    o.integrate(ts,ip)
    assert numpy.fabs(ecc-o.e()) < 1e-10, 'Analytically calculated eccentricity does not agree with numerically calculated one for an IsochronePotential'
    assert numpy.fabs(zmax-o.zmax()) < 1e-5, 'Analytically calculated zmax does not agree with numerically calculated one for an IsochronePotential'
    assert numpy.fabs(rperi-o.rperi()) < 1e-10, 'Analytically calculated rperi does not agree with numerically calculated one for an IsochronePotential'
    assert numpy.fabs(rap-o.rap()) < 1e-10, 'Analytically calculated rap does not agree with numerically calculated one for an IsochronePotential'
    # Another one
    o= Orbit([1.,0.1,1.1,0.2,-0.3,0.])
    ecc, zmax, rperi, rap= aAI.EccZmaxRperiRap(o.R(),o.vR(),o.vT(),
                                               o.z(),o.vz(),o.phi())
    ts= numpy.linspace(0.,100.,100001)
    o.integrate(ts,ip)
    assert numpy.fabs(ecc-o.e()) < 1e-10, 'Analytically calculated eccentricity does not agree with numerically calculated one for an IsochronePotential'
    assert numpy.fabs(zmax-o.zmax()) < 1e-3, 'Analytically calculated zmax does not agree with numerically calculated one for an IsochronePotential'
    assert numpy.fabs(rperi-o.rperi()) < 1e-10, 'Analytically calculated rperi does not agree with numerically calculated one for an IsochronePotential'
    assert numpy.fabs(rap-o.rap()) < 1e-10, 'Analytically calculated rap does not agree with numerically calculated one for an IsochronePotential'
    return None
    
# Test that EccZmaxRperiRap for an IsochronePotential are correctly computed
# by comparing to a numerical orbit integration for a Kepler potential
def test_actionAngleIsochrone_EccZmaxRperiRap_againstOrbit_kepler():
    from galpy.potential import IsochronePotential
    from galpy.orbit import Orbit
    from galpy.actionAngle import actionAngleIsochrone
    ip= IsochronePotential(normalize=1.,b=0)
    aAI= actionAngleIsochrone(ip=ip)
    o= Orbit([1.,0.1,1.1,0.2,0.03,0.])
    ecc, zmax, rperi, rap= aAI.EccZmaxRperiRap(o.R(),o.vR(),o.vT(),o.z(),o.vz())
    ts= numpy.linspace(0.,100.,100001)
    o.integrate(ts,ip)
    assert numpy.fabs(ecc-o.e()) < 1e-10, 'Analytically calculated eccentricity does not agree with numerically calculated one for an IsochronePotential'
    # Don't do zmax, because zmax for Kepler is approximate
    assert numpy.fabs(rperi-o.rperi()) < 1e-10, 'Analytically calculated rperi does not agree with numerically calculated one for an IsochronePotential'
    assert numpy.fabs(rap-o.rap()) < 1e-10, 'Analytically calculated rap does not agree with numerically calculated one for an IsochronePotential'
    return None


#Test the actions of an actionAngleIsochrone
def test_actionAngleIsochrone_conserved_actions():
    from galpy.potential import IsochronePotential
    from galpy.actionAngle import actionAngleIsochrone
    from galpy.orbit import Orbit
    ip= IsochronePotential(normalize=1.,b=1.2)
    aAI= actionAngleIsochrone(ip=ip)
    obs= Orbit([1.1, 0.3, 1.2, 0.2,0.5])
    from galpy.orbit.Orbits import ext_loaded
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
    from galpy.orbit.Orbits import ext_loaded
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
    js= aAS(Orbit([R,vR,vT,z,vz]))
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

# Test that EccZmaxRperiRap for a spherical potential are correctly computed
# by comparing to a numerical orbit integration
def test_actionAngleSpherical_EccZmaxRperiRap_againstOrbit():
    from galpy.potential import LogarithmicHaloPotential
    from galpy.orbit import Orbit
    from galpy.actionAngle import actionAngleSpherical
    lp= LogarithmicHaloPotential(normalize=1.,q=1.)
    aAS= actionAngleSpherical(pot=lp)
    o= Orbit([1.,0.1,1.1,0.2,0.03,0.])
    ecc, zmax, rperi, rap= aAS.EccZmaxRperiRap(o)
    ts= numpy.linspace(0.,100.,100001)
    o.integrate(ts,lp)
    assert numpy.fabs(ecc-o.e()) < 1e-9, 'Analytically calculated eccentricity does not agree with numerically calculated one for a spherical potential'
    assert numpy.fabs(zmax-o.zmax()) < 1e-4, 'Analytically calculated zmax does not agree with numerically calculated one for a spherical potential'
    assert numpy.fabs(rperi-o.rperi()) < 1e-8, 'Analytically calculated rperi does not agree with numerically calculated one for a spherical potential'
    assert numpy.fabs(rap-o.rap()) < 1e-8, 'Analytically calculated rap does not agree with numerically calculated one for a spherical potential'
    # Another one
    o= Orbit([1.,0.1,1.1,0.2,-0.3,0.])
    ecc, zmax, rperi, rap= aAS.EccZmaxRperiRap(o.R(),o.vR(),o.vT(),
                                               o.z(),o.vz())
    ts= numpy.linspace(0.,100.,100001)
    o.integrate(ts,lp)
    assert numpy.fabs(ecc-o.e()) < 1e-9, 'Analytically calculated eccentricity does not agree with numerically calculated one for a spherical potential'
    assert numpy.fabs(zmax-o.zmax()) < 1e-3, 'Analytically calculated zmax does not agree with numerically calculated one for a spherical potential'
    assert numpy.fabs(rperi-o.rperi()) < 1e-8, 'Analytically calculated rperi does not agree with numerically calculated one for a spherical potential'
    assert numpy.fabs(rap-o.rap()) < 1e-8, 'Analytically calculated rap does not agree with numerically calculated one for a spherical potential'
    return None

#Test the actions of an actionAngleSpherical
def test_actionAngleSpherical_conserved_actions():
    from galpy import potential
    from galpy.actionAngle import actionAngleSpherical
    from galpy.orbit import Orbit
    lp= potential.LogarithmicHaloPotential(normalize=1.,q=1.)
    aAS= actionAngleSpherical(pot=lp)
    obs= Orbit([1.1, 0.3, 1.2, 0.2,0.5])
    from galpy.orbit.Orbits import ext_loaded
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
    from galpy.orbit.Orbits import ext_loaded
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
    from galpy.orbit.Orbits import ext_loaded
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
    from galpy.orbit.Orbits import ext_loaded
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
  
#Test the conservation of ecc, zmax, rperi, rap of an actionAngleSpherical
def test_actionAngleSpherical_conserved_EccZmaxRperiRap_ecc():
    from galpy.potential import NFWPotential
    from galpy.actionAngle import actionAngleSpherical
    from galpy.orbit import Orbit
    np= NFWPotential(normalize=1.,a=2.)
    aAS= actionAngleSpherical(pot=np)
    obs= Orbit([1.1,0.2, 1.3, 0.1,0.,2.])
    check_actionAngle_conserved_EccZmaxRperiRap(aAS,obs,np,
                                                -1.1,-0.4,-1.8,-1.8,ntimes=101,
                                                inclphi=True)
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
    aAA= actionAngleAdiabatic(pot=[MWPotential[0],MWPotential[1:]],gamma=0.)
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
    # test nested list of potentials
    aAA= actionAngleAdiabatic(pot=[MWPotential[0],MWPotential[1:]],c=True)
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

#Basic sanity checking of the actionAngleAdiabatic ecc, zmax, rperi, rap calc.
def test_actionAngleAdiabatic_basic_EccZmaxRperiRap():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.potential import MWPotential
    aAA= actionAngleAdiabatic(pot=MWPotential,gamma=1.)
    #circular orbit
    R,vR,vT,z,vz= 1.,0.,1.,0.,0. 
    te,tzmax,_,_= aAA.EccZmaxRperiRap(R,vR,vT,z,vz)
    assert numpy.fabs(te) < 10.**-16., 'Circular orbit in the MWPotential does not have e=0'
    assert numpy.fabs(tzmax) < 10.**-16., 'Circular orbit in the MWPotential does not have zmax=0'
    #Close-to-circular orbit
    R,vR,vT,z,vz= 1.01,0.01,1.,0.01,0.01
    te,tzmax,_,_= aAA.EccZmaxRperiRap(R,vR,vT,z,vz)
    assert numpy.fabs(te) < 10.**-2., 'Close-to-circular orbit in the MWPotential does not have small eccentricity'
    assert numpy.fabs(tzmax) < 2.*10.**-2., 'Close-to-circular orbit in the MWPotential does not have small zmax'
    #Another close-to-circular orbit
    R,vR,vT,z,vz= 1.0,0.0,0.99,0.0,0.0
    te,tzmax,_,_= aAA.EccZmaxRperiRap(R,vR,vT,z,vz)
    assert numpy.fabs(te) < 10.**-2., 'Close-to-circular orbit in the MWPotential does not have small eccentricity'
    assert numpy.fabs(tzmax) < 2.*10.**-2., 'Close-to-circular orbit in the MWPotential does not have small zmax'
    #Another close-to-circular orbit
    R,vR,vT,z,vz= 1.0,0.0,1.,0.01,0.0
    te,tzmax,_,_= aAA.EccZmaxRperiRap(R,vR,vT,z,vz)
    assert numpy.fabs(te) < 10.**-2., 'Close-to-circular orbit in the MWPotential does not have small eccentricity'
    assert numpy.fabs(tzmax) < 2.*10.**-2., 'Close-to-circular orbit in the MWPotential does not have small zmax'
    return None

#Basic sanity checking of the actionAngleAdiabatic ecc, zmax, rperi, rap calc.
def test_actionAngleAdiabatic_basic_EccZmaxRperiRap_gamma0():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.potential import MiyamotoNagaiPotential
    mp= MiyamotoNagaiPotential(normalize=1.,a=1.5,b=0.3)
    aAA= actionAngleAdiabatic(pot=mp,gamma=0.,c=False)
    #circular orbit
    R,vR,vT,z,vz= 1.,0.,1.,0.,0. 
    te,tzmax,_,_= aAA.EccZmaxRperiRap(R,vR,vT,z,vz)
    assert numpy.fabs(te) < 10.**-16., 'Circular orbit in the MWPotential does not have e=0'
    assert numpy.fabs(tzmax) < 10.**-16., 'Circular orbit in the MWPotential does not have zmax=0'
    #Close-to-circular orbit
    R,vR,vT,z,vz= 1.01,0.01,1.,0.01,0.01 
    te,tzmax,_,_= aAA.EccZmaxRperiRap(R,vR,vT,z,vz)
    assert numpy.fabs(te) < 10.**-2., 'Close-to-circular orbit in the MWPotential does not have small eccentricity'
    assert numpy.fabs(tzmax) < 2.*10.**-2., 'Close-to-circular orbit in the MWPotential does not have small zmax'
    return None

#Basic sanity checking of the actionAngleAdiabatic ecc, zmax, rperi, rap calc.
def test_actionAngleAdiabatic_basic_EccZmaxRperiRap_gamma_c():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.potential import MWPotential
    from galpy.orbit import Orbit
    aAA= actionAngleAdiabatic(pot=MWPotential,gamma=1.,c=True)
    #circular orbit
    R,vR,vT,z,vz,phi= 1.,0.,1.,0.,0.,2.
    te,tzmax,_,_= aAA.EccZmaxRperiRap(Orbit([R,vR,vT,z,vz,phi]))
    assert numpy.fabs(te) < 10.**-16., 'Circular orbit in the MWPotential does not have e=0'
    assert numpy.fabs(tzmax) < 10.**-16., 'Circular orbit in the MWPotential does not have zmax=0'
    #Close-to-circular orbit
    R,vR,vT,z,vz,phi= 1.01,0.01,1.,0.01,0.01,2.
    te,tzmax,_,_= aAA.EccZmaxRperiRap(R,vR,vT,z,vz,phi)
    assert numpy.fabs(te) < 10.**-2., 'Close-to-circular orbit in the MWPotential does not have small eccentricity'
    assert numpy.fabs(tzmax) < 2.*10.**-2., 'Close-to-circular orbit in the MWPotential does not have small zmax'
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

#Test the conservation of ecc, zmax, rperi, rap of an actionAngleAdiabatic
def test_actionAngleAdiabatic_conserved_EccZmaxRperiRap():
    from galpy.potential import MWPotential
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.orbit import Orbit
    aAA= actionAngleAdiabatic(pot=MWPotential,c=False,gamma=1.)
    obs= Orbit([1.05, 0.02, 1.05, 0.03,0.,0.])
    check_actionAngle_conserved_EccZmaxRperiRap(aAA,obs,MWPotential,
                                                -1.7,-1.4,-2.,-2.,ntimes=101)
    return None

#Test the conservation of ecc, zmax, rperi, rap of an actionAngleAdiabatic
def test_actionAngleAdiabatic_conserved_EccZmaxRperiRap_ecc():
    from galpy.potential import MWPotential
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.orbit import Orbit
    aAA= actionAngleAdiabatic(pot=MWPotential,c=False,gamma=1.)
    obs= Orbit([1.1,0.2, 1.3, 0.1,0.,2.])
    check_actionAngle_conserved_EccZmaxRperiRap(aAA,obs,MWPotential,
                                                -1.1,-0.4,-1.8,-1.8,ntimes=101,
                                                inclphi=True)
    return None

#Test the conservation of ecc, zmax, rperi, rap of an actionAngleAdiabatic
def test_actionAngleAdiabatic_conserved_EccZmaxRperiRap_singlepot_c():
    from galpy.potential import MiyamotoNagaiPotential
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.orbit import Orbit
    mp= MiyamotoNagaiPotential(normalize=1.)
    obs= Orbit([1.05, 0.02, 1.05, 0.03,0.,2.])
    aAA= actionAngleAdiabatic(pot=mp,c=True)
    check_actionAngle_conserved_EccZmaxRperiRap(aAA,obs,mp,
                                                -1.7,-1.4,-2.,-2.,ntimes=101)
    return None

#Test the conservation of ecc, zmax, rperi, rap of an actionAngleAdiabatic
def test_actionAngleAdiabatic_conserved_EccZmaxRperiRa_interppot_c():
    from galpy.potential import MWPotential, interpRZPotential
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.orbit import Orbit
    ip= interpRZPotential(RZPot=MWPotential,
                          rgrid=(numpy.log(0.01),numpy.log(20.),101),
                          zgrid=(0.,1.,101),logR=True,use_c=True,enable_c=True,
                          interpPot=True,interpRforce=True,interpzforce=True)
    obs= Orbit([1.05, 0.02, 1.05, 0.03,0.,2.])
    aAA= actionAngleAdiabatic(pot=ip,c=True)
    check_actionAngle_conserved_EccZmaxRperiRap(aAA,obs,ip,
                                                -1.7,-1.4,-2.,-2.,ntimes=101)
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
    # test nested list of potentials
    aAS= actionAngleStaeckel(pot=[MWPotential[0],MWPotential[1:]],
                             delta=0.71,c=False,useu0=True)
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
    # test nested list of potentials
    aAS= actionAngleStaeckel(pot=[MWPotential[0],MWPotential[1:]],
                             delta=0.71,c=True,useu0=True)
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
    assert numpy.fabs(js[0][0]) < 10.**-12., 'Circular orbit in the MWPotential does not have Jr=0'
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

# Check that precision increases with increasing Gauss-Legendre order
def test_actionAngleStaeckel_actions_order():
    from galpy.potential import KuzminKutuzovStaeckelPotential
    from galpy.orbit import Orbit
    from galpy.actionAngle import actionAngleStaeckel
    kksp= KuzminKutuzovStaeckelPotential(normalize=1.,ac=4.,Delta=1.4)
    o= Orbit([1.,0.5,1.1,0.2,-0.3,0.4])
    aAS= actionAngleStaeckel(pot=kksp,delta=kksp._Delta,c=False)
    # We'll assume that order=10000 is the truth, so 50 should be better than 5
    jrt,jpt,jzt= aAS(o,order=10000,fixed_quad=True)
    jr1,jp1,jz1= aAS(o,order=5,fixed_quad=True)
    jr2,jp2,jz2= aAS(o,order=50,fixed_quad=True)
    assert numpy.fabs(jr1-jrt) > numpy.fabs(jr2-jrt), 'Accuracy of actionAngleStaeckel does not increase with increasing order of integration'
    assert numpy.fabs(jz1-jzt) > numpy.fabs(jz2-jzt), 'Accuracy of actionAngleStaeckel does not increase with increasing order of integration'
    return None

def test_actionAngleStaeckel_actions_order_c():
    from galpy.potential import KuzminKutuzovStaeckelPotential
    from galpy.orbit import Orbit
    from galpy.actionAngle import actionAngleStaeckel
    kksp= KuzminKutuzovStaeckelPotential(normalize=1.,ac=4.,Delta=1.4)
    o= Orbit([1.,0.5,1.1,0.2,-0.3,0.4])
    aAS= actionAngleStaeckel(pot=kksp,delta=kksp._Delta,c=True)
    # We'll assume that order=10000 is the truth, so 50 should be better than 5
    jrt,jpt,jzt= aAS(o,order=10000)
    jr1,jp1,jz1= aAS(o,order=5)
    jr2,jp2,jz2= aAS(o,order=50)
    assert numpy.fabs(jr1-jrt) > numpy.fabs(jr2-jrt), 'Accuracy of actionAngleStaeckel does not increase with increasing order of integration'
    assert numpy.fabs(jz1-jzt) > numpy.fabs(jz2-jzt), 'Accuracy of actionAngleStaeckel does not increase with increasing order of integration'
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

# Check that precision increases with increasing Gauss-Legendre order
def test_actionAngleStaeckel_freqs_order_c():
    from galpy.potential import KuzminKutuzovStaeckelPotential
    from galpy.orbit import Orbit
    from galpy.actionAngle import actionAngleStaeckel
    kksp= KuzminKutuzovStaeckelPotential(normalize=1.,ac=4.,Delta=1.4)
    o= Orbit([1.,0.5,1.1,0.2,-0.3,0.4])
    aAS= actionAngleStaeckel(pot=kksp,delta=kksp._Delta,c=True)
    # We'll assume that order=10000 is the truth, so 50 should be better than 5
    jrt,jpt,jzt,ort,opt,ozt= aAS.actionsFreqs(o,order=10000)
    jr1,jp1,jz1,or1,op1,oz1= aAS.actionsFreqs(o,order=5)
    jr2,jp2,jz2,or2,op2,oz2= aAS.actionsFreqs(o,order=50)
    assert numpy.fabs(jr1-jrt) > numpy.fabs(jr2-jrt), 'Accuracy of actionAngleStaeckel does not increase with increasing order of integration'
    assert numpy.fabs(jz1-jzt) > numpy.fabs(jz2-jzt), 'Accuracy of actionAngleStaeckel does not increase with increasing order of integration'
    assert numpy.fabs(or1-ort) > numpy.fabs(or2-ort), 'Accuracy of actionAngleStaeckel does not increase with increasing order of integration'
    assert numpy.fabs(op1-opt) > numpy.fabs(op2-opt), 'Accuracy of actionAngleStaeckel does not increase with increasing order of integration'
    assert numpy.fabs(oz1-ozt) > numpy.fabs(oz2-ozt), 'Accuracy of actionAngleStaeckel does not increase with increasing order of integration'
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

# Check that precision increases with increasing Gauss-Legendre order
def test_actionAngleStaeckel_angles_order_c():
    from galpy.potential import KuzminKutuzovStaeckelPotential
    from galpy.orbit import Orbit
    from galpy.actionAngle import actionAngleStaeckel
    kksp= KuzminKutuzovStaeckelPotential(normalize=1.,ac=4.,Delta=1.4)
    o= Orbit([1.,0.5,1.1,0.2,-0.3,0.4])
    aAS= actionAngleStaeckel(pot=kksp,delta=kksp._Delta,c=True)
    # We'll assume that order=10000 is the truth, so 50 should be better than 5
    jrt,jpt,jzt,ort,opt,ozt,art,apt,azt= aAS.actionsFreqsAngles(o,order=10000)
    jr1,jp1,jz1,or1,op1,oz1,ar1,ap1,az1= aAS.actionsFreqsAngles(o,order=5)
    jr2,jp2,jz2,or2,op2,oz2,ar2,ap2,az2= aAS.actionsFreqsAngles(o,order=50)
    assert numpy.fabs(jr1-jrt) > numpy.fabs(jr2-jrt), 'Accuracy of actionAngleStaeckel does not increase with increasing order of integration'
    assert numpy.fabs(jz1-jzt) > numpy.fabs(jz2-jzt), 'Accuracy of actionAngleStaeckel does not increase with increasing order of integration'
    assert numpy.fabs(or1-ort) > numpy.fabs(or2-ort), 'Accuracy of actionAngleStaeckel does not increase with increasing order of integration'
    assert numpy.fabs(op1-opt) > numpy.fabs(op2-opt), 'Accuracy of actionAngleStaeckel does not increase with increasing order of integration'
    assert numpy.fabs(oz1-ozt) > numpy.fabs(oz2-ozt), 'Accuracy of actionAngleStaeckel does not increase with increasing order of integration'
    assert numpy.fabs(ar1-art) > numpy.fabs(ar2-art), 'Accuracy of actionAngleStaeckel does not increase with increasing order of integration'
    assert numpy.fabs(ap1-apt) > numpy.fabs(ap2-apt), 'Accuracy of actionAngleStaeckel does not increase with increasing order of integration'
    assert numpy.fabs(az1-azt) > numpy.fabs(az2-azt), 'Accuracy of actionAngleStaeckel does not increase with increasing order of integration'
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
    from galpy.orbit import Orbit
    aAS= actionAngleStaeckel(pot=MWPotential,delta=0.71,c=True,useu0=True)
    #circular orbit
    R,vR,vT,z,vz= 1.,0.,1.,0.,0. 
    te,tzmax,_,_= aAS.EccZmaxRperiRap(Orbit([R,vR,vT,z,vz]))
    assert numpy.fabs(te) < 10.**-16., 'Circular orbit in the MWPotential does not have e=0'
    assert numpy.fabs(tzmax) < 10.**-16., 'Circular orbit in the MWPotential does not have zmax=0'
    #Close-to-circular orbit
    R,vR,vT,z,vz= 1.01,0.01,1.,0.01,0.01 
    te,tzmax,_,_= aAS.EccZmaxRperiRap(R,vR,vT,z,vz,u0=1.15)
    assert numpy.fabs(te) < 10.**-2., 'Close-to-circular orbit in the MWPotential does not have small eccentricity'
    assert numpy.fabs(tzmax) < 2.*10.**-2., 'Close-to-circular orbit in the MWPotential does not have small zmax'
    return None

#Test that using different delta for different phase-space points works
def test_actionAngleStaeckel_indivdelta_actions():
    from galpy.potential import MWPotential2014
    from galpy.orbit import Orbit
    from galpy.actionAngle import actionAngleStaeckel
    # Briefly integrate orbit to get multiple points
    o= Orbit([1.,0.1,1.1,0.,0.25,1.])
    ts= numpy.linspace(0.,1.,101)
    o.integrate(ts,MWPotential2014)
    deltas= [0.2,0.4]
    # actions with one delta
    aAS= actionAngleStaeckel(pot=MWPotential2014,delta=deltas[0],c=False)
    jr0,jp0,jz0= aAS(o.R(ts[:2]),o.vR(ts[:2]),o.vT(ts[:2]),
                     o.z(ts[:2]),o.vz(ts[:2]))
    # actions with another delta
    aAS= actionAngleStaeckel(pot=MWPotential2014,delta=deltas[1],c=False)
    jr1,jp1,jz1= aAS(o.R(ts[:2]),o.vR(ts[:2]),o.vT(ts[:2]),
                     o.z(ts[:2]),o.vz(ts[:2]))
    # actions with individual delta
    jri,jpi,jzi= aAS(o.R(ts[:2]),o.vR(ts[:2]),o.vT(ts[:2]),
                     o.z(ts[:2]),o.vz(ts[:2]),delta=deltas)
    # Check that they agree as expected
    assert numpy.fabs(jr0[0]-jri[0]) < 1e-10, 'Radial action computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(jr1[1]-jri[1]) < 1e-10, 'Radial action computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(jz0[0]-jzi[0]) < 1e-10, 'Vertical action computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(jz1[1]-jzi[1]) < 1e-10, 'Vertical action computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    return None
    
# Test that no_median option for estimateDeltaStaeckel returns the same results as when
# individual values are calculated separately
def test_estimateDeltaStaeckel_no_median():
	from galpy.potential import MWPotential2014
	from galpy.orbit import Orbit
	from galpy.actionAngle import estimateDeltaStaeckel
	# Briefly integrate orbit to get multiple points
	o= Orbit([1.,0.1,1.1,0.001,0.25,1.])
	ts= numpy.linspace(0.,1.,101)
	o.integrate(ts,MWPotential2014)
	#generate no_median deltas
	nomed = estimateDeltaStaeckel(MWPotential2014, o.R(ts[:10]), o.z(ts[:10]), no_median=True)
	#and the individual ones
	indiv = numpy.array([estimateDeltaStaeckel(MWPotential2014, o.R(ts[i]), o.z(ts[i])) for i in range(10)])
	#check that values agree
	assert (numpy.fabs(nomed-indiv) < 1e-10).all(), 'no_median option returns different values to individual Delta estimation'
	return None

def test_actionAngleStaeckel_indivdelta_actions_c():
    from galpy.potential import MWPotential2014
    from galpy.orbit import Orbit
    from galpy.actionAngle import actionAngleStaeckel
    # Briefly integrate orbit to get multiple points
    o= Orbit([1.,0.1,1.1,0.,0.25,1.])
    ts= numpy.linspace(0.,1.,101)
    o.integrate(ts,MWPotential2014)
    deltas= [0.2,0.4]
    # actions with one delta
    aAS= actionAngleStaeckel(pot=MWPotential2014,delta=deltas[0],c=True)
    jr0,jp0,jz0= aAS(o.R(ts[:2]),o.vR(ts[:2]),o.vT(ts[:2]),
                     o.z(ts[:2]),o.vz(ts[:2]))
    # actions with another delta
    aAS= actionAngleStaeckel(pot=MWPotential2014,delta=deltas[1],c=True)
    jr1,jp1,jz1= aAS(o.R(ts[:2]),o.vR(ts[:2]),o.vT(ts[:2]),
                     o.z(ts[:2]),o.vz(ts[:2]))
    # actions with individual delta
    jri,jpi,jzi= aAS(o.R(ts[:2]),o.vR(ts[:2]),o.vT(ts[:2]),
                     o.z(ts[:2]),o.vz(ts[:2]),delta=deltas)
    # Check that they agree as expected
    assert numpy.fabs(jr0[0]-jri[0]) < 1e-10, 'Radial action computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(jr1[1]-jri[1]) < 1e-10, 'Radial action computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(jz0[0]-jzi[0]) < 1e-10, 'Vertical action computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(jz1[1]-jzi[1]) < 1e-10, 'Vertical action computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    return None

def test_actionAngleStaeckel_indivdelta_freqs_c():
    from galpy.potential import MWPotential2014
    from galpy.orbit import Orbit
    from galpy.actionAngle import actionAngleStaeckel
    # Briefly integrate orbit to get multiple points
    o= Orbit([1.,0.1,1.1,0.,0.25,1.])
    ts= numpy.linspace(0.,1.,101)
    o.integrate(ts,MWPotential2014)
    deltas= [0.2,0.4]
    # actions with one delta
    aAS= actionAngleStaeckel(pot=MWPotential2014,delta=deltas[0],c=True)
    jr0,jp0,jz0,or0,op0,oz0= aAS.actionsFreqs(o.R(ts[:2]),o.vR(ts[:2]),
                                              o.vT(ts[:2]),o.z(ts[:2]),
                                              o.vz(ts[:2]),o.phi(ts[:2]))
    # actions with another delta
    aAS= actionAngleStaeckel(pot=MWPotential2014,delta=deltas[1],c=True)
    jr1,jp1,jz1,or1,op1,oz1= aAS.actionsFreqs(o.R(ts[:2]),o.vR(ts[:2]),
                                              o.vT(ts[:2]),o.z(ts[:2]),
                                              o.vz(ts[:2]),o.phi(ts[:2]))
    # actions with individual delta
    jri,jpi,jzi,ori,opi,ozi= aAS.actionsFreqs(o.R(ts[:2]),o.vR(ts[:2]),
                                              o.vT(ts[:2]),o.z(ts[:2]),
                                              o.vz(ts[:2]),o.phi(ts[:2]),
                                              delta=deltas)
    # Check that they agree as expected
    assert numpy.fabs(jr0[0]-jri[0]) < 1e-10, 'Radial action computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(jr1[1]-jri[1]) < 1e-10, 'Radial action computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(jz0[0]-jzi[0]) < 1e-10, 'Vertical action computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(jz1[1]-jzi[1]) < 1e-10, 'Vertical action computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(or0[0]-ori[0]) < 1e-10, 'Radial frequencyaction computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(or1[1]-ori[1]) < 1e-10, 'Radial frequency computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(op0[0]-opi[0]) < 1e-10, 'Azimuthal computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(op1[1]-opi[1]) < 1e-10, 'Azimuthal computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(oz0[0]-ozi[0]) < 1e-10, 'Azimuthal frequency computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(oz1[1]-ozi[1]) < 1e-10, 'Vertical frequency computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    return None

def test_actionAngleStaeckel_indivdelta_angles_c():
    from galpy.potential import MWPotential2014
    from galpy.orbit import Orbit
    from galpy.actionAngle import actionAngleStaeckel
    # Briefly integrate orbit to get multiple points
    o= Orbit([1.,0.1,1.1,0.,0.25,1.])
    ts= numpy.linspace(0.,1.,101)
    o.integrate(ts,MWPotential2014)
    deltas= [0.2,0.4]
    # actions with one delta
    aAS= actionAngleStaeckel(pot=MWPotential2014,delta=deltas[0],c=True)
    jr0,jp0,jz0,or0,op0,oz0,ar0,ap0,az0=\
        aAS.actionsFreqsAngles(o.R(ts[:2]),o.vR(ts[:2]),
                               o.vT(ts[:2]),o.z(ts[:2]),
                               o.vz(ts[:2]),o.phi(ts[:2]))
    # actions with another delta
    aAS= actionAngleStaeckel(pot=MWPotential2014,delta=deltas[1],c=True)
    jr1,jp1,jz1,or1,op1,oz1,ar1,ap1,az1=\
        aAS.actionsFreqsAngles(o.R(ts[:2]),o.vR(ts[:2]),
                               o.vT(ts[:2]),o.z(ts[:2]),
                               o.vz(ts[:2]),o.phi(ts[:2]))
    # actions with individual delta
    jri,jpi,jzi,ori,opi,ozi,ari,api,azi=\
        aAS.actionsFreqsAngles(o.R(ts[:2]),o.vR(ts[:2]),
                               o.vT(ts[:2]),o.z(ts[:2]),
                               o.vz(ts[:2]),o.phi(ts[:2]),
                               delta=deltas)
    # Check that they agree as expected
    assert numpy.fabs(jr0[0]-jri[0]) < 1e-10, 'Radial action computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(jr1[1]-jri[1]) < 1e-10, 'Radial action computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(jz0[0]-jzi[0]) < 1e-10, 'Vertical action computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(jz1[1]-jzi[1]) < 1e-10, 'Vertical action computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(or0[0]-ori[0]) < 1e-10, 'Radial frequencyaction computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(or1[1]-ori[1]) < 1e-10, 'Radial frequency computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(op0[0]-opi[0]) < 1e-10, 'Azimuthal computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(op1[1]-opi[1]) < 1e-10, 'Azimuthal computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(oz0[0]-ozi[0]) < 1e-10, 'Azimuthal frequency computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(oz1[1]-ozi[1]) < 1e-10, 'Vertical frequency computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(ar0[0]-ari[0]) < 1e-10, 'Radial frequencyaction computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(ar1[1]-ari[1]) < 1e-10, 'Radial frequency computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(ap0[0]-api[0]) < 1e-10, 'Azimuthal computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(ap1[1]-api[1]) < 1e-10, 'Azimuthal computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(az0[0]-azi[0]) < 1e-10, 'Azimuthal frequency computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(az1[1]-azi[1]) < 1e-10, 'Vertical frequency computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    return None

def test_actionAngleStaeckel_indivdelta_EccZmaxRperiRap():
    from galpy.potential import MWPotential2014
    from galpy.orbit import Orbit
    from galpy.actionAngle import actionAngleStaeckel
    # Briefly integrate orbit to get multiple points
    o= Orbit([1.,0.1,1.1,0.,0.25,1.])
    ts= numpy.linspace(0.,1.,101)
    o.integrate(ts,MWPotential2014)
    deltas= [0.2,0.4]
    # with one delta
    aAS= actionAngleStaeckel(pot=MWPotential2014,delta=deltas[0],c=False)
    e0,z0,rp0,ra0= aAS.EccZmaxRperiRap(o.R(ts[:2]),o.vR(ts[:2]),o.vT(ts[:2]),
                                       o.z(ts[:2]),o.vz(ts[:2]))
    # actions with another delta
    aAS= actionAngleStaeckel(pot=MWPotential2014,delta=deltas[1],c=False)
    e1,z1,rp1,ra1= aAS.EccZmaxRperiRap(o.R(ts[:2]),o.vR(ts[:2]),o.vT(ts[:2]),
                                       o.z(ts[:2]),o.vz(ts[:2]))
    # actions with individual delta
    ei,zi,rpi,rai= aAS.EccZmaxRperiRap(o.R(ts[:2]),o.vR(ts[:2]),o.vT(ts[:2]),
                                       o.z(ts[:2]),o.vz(ts[:2]),delta=deltas)
    # Check that they agree as expected
    assert numpy.fabs(e0[0]-ei[0]) < 1e-10, 'Eccentricity computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(e1[1]-ei[1]) < 1e-10, 'Eccentricity computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(z0[0]-zi[0]) < 1e-10, 'Zmax computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(z1[1]-zi[1]) < 1e-10, 'Zmax computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(rp0[0]-rpi[0]) < 1e-10, 'Pericenter computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(rp1[1]-rpi[1]) < 1e-10, 'Pericenter computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(ra0[0]-rai[0]) < 1e-10, 'Apocenter computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(ra1[1]-rai[1]) < 1e-10, 'Apocenter computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    return None

def test_actionAngleStaeckel_indivdelta_EccZmaxRperiRap_c():
    from galpy.potential import MWPotential2014
    from galpy.orbit import Orbit
    from galpy.actionAngle import actionAngleStaeckel
    # Briefly integrate orbit to get multiple points
    o= Orbit([1.,0.1,1.1,0.,0.25,1.])
    ts= numpy.linspace(0.,1.,101)
    o.integrate(ts,MWPotential2014)
    deltas= [0.2,0.4]
    # with one delta
    aAS= actionAngleStaeckel(pot=MWPotential2014,delta=deltas[0],c=True)
    e0,z0,rp0,ra0= aAS.EccZmaxRperiRap(o.R(ts[:2]),o.vR(ts[:2]),o.vT(ts[:2]),
                                       o.z(ts[:2]),o.vz(ts[:2]))
    # actions with another delta
    aAS= actionAngleStaeckel(pot=MWPotential2014,delta=deltas[1],c=True)
    e1,z1,rp1,ra1= aAS.EccZmaxRperiRap(o.R(ts[:2]),o.vR(ts[:2]),o.vT(ts[:2]),
                                       o.z(ts[:2]),o.vz(ts[:2]))
    # actions with individual delta
    ei,zi,rpi,rai= aAS.EccZmaxRperiRap(o.R(ts[:2]),o.vR(ts[:2]),o.vT(ts[:2]),
                                       o.z(ts[:2]),o.vz(ts[:2]),delta=deltas)
    # Check that they agree as expected
    assert numpy.fabs(e0[0]-ei[0]) < 1e-10, 'Eccentricity computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(e1[1]-ei[1]) < 1e-10, 'Eccentricity computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(z0[0]-zi[0]) < 1e-10, 'Zmax computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(z1[1]-zi[1]) < 1e-10, 'Zmax computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(rp0[0]-rpi[0]) < 1e-10, 'Pericenter computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(rp1[1]-rpi[1]) < 1e-10, 'Pericenter computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(ra0[0]-rai[0]) < 1e-10, 'Apocenter computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
    assert numpy.fabs(ra1[1]-rai[1]) < 1e-10, 'Apocenter computed with invidual delta does not agree with that computed using the fixed orbit-wide default'
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
        TriaxialNFWPotential, SCFPotential, DiskSCFPotential, \
        PerfectEllipsoidPotential
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.orbit.Orbits import ext_loaded
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
           ip,
           PerfectEllipsoidPotential(normalize=1.,c=0.98)]
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
    from galpy.orbit.Orbits import ext_loaded
    from test_potential import mockSCFZeeuwPotential, \
        mockSphericalSoftenedNeedleBarPotential, \
        mockSmoothedLogarithmicHaloPotential, \
        mockGaussianAmplitudeSmoothedLogarithmicHaloPotential
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
    mgasmlp= mockGaussianAmplitudeSmoothedLogarithmicHaloPotential()
    pots= [lp,lpb,hp,jp,np,ip,pp,lp2,ppc,plp,psp,bp,scfp,scfzp,
           msoftneedlep,msmlp,mgasmlp]
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
    from galpy.orbit.Orbits import ext_loaded
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
                                                -1.8,-1.4,-1.8,-1.8,ntimes=101,
                                                inclphi=True)
    return None

#Test the conservation of ecc, zmax, rperi, rap of an actionAngleStaeckel
def test_actionAngleStaeckel_conserved_EccZmaxRperiRap_c():
    from galpy.potential import MWPotential, DoubleExponentialDiskPotential, \
        FlattenedPowerPotential, interpRZPotential, KuzminDiskPotential, \
        TriaxialHernquistPotential, TriaxialJaffePotential, \
        TriaxialNFWPotential, SCFPotential, DiskSCFPotential, \
        PerfectEllipsoidPotential
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.orbit.Orbits import ext_loaded
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
           ip,
           PerfectEllipsoidPotential(normalize=1.,c=0.98)]
    for pot in pots:
        aAS= actionAngleStaeckel(pot=pot,c=True,delta=0.71)
        obs= Orbit([1.05, 0.02, 1.05, 0.03,0.,2.])
        check_actionAngle_conserved_EccZmaxRperiRap(aAS,obs,pot,
                                                    -1.8,-1.3,-1.8,-1.8,
                                                    ntimes=101)
    return None

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

#Basic sanity checking of the actionAngleStaeckelGrid actions (incl. conserved and ecc etc., bc takes a lot of time)
def test_actionAngleStaeckelGrid_basicAndConserved_actions():
    from galpy.actionAngle import actionAngleStaeckelGrid
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential
    aAA= actionAngleStaeckelGrid(pot=MWPotential,delta=0.71,c=False,nLz=20,
                                 interpecc=True)
    #circular orbit
    R,vR,vT,z,vz= 1.,0.,1.,0.,0. 
    assert numpy.fabs(aAA.JR(R,vR,vT,z,vz,0.)) < 10.**-16., 'Circular orbit in the MWPotential does not have Jr=0'
    assert numpy.fabs(aAA.Jz(R,vR,vT,z,vz,0.)) < 10.**-16., 'Circular orbit in the MWPotential does not have Jz=0'
    te,tzmax,_,_= aAA.EccZmaxRperiRap(R,vR,vT,z,vz)
    assert numpy.fabs(te) < 10.**-16., 'Circular orbit in the MWPotential does not have e=0'
    assert numpy.fabs(tzmax) < 10.**-16., 'Circular orbit in the MWPotential does not have zmax=0'
    #Close-to-circular orbit
    R,vR,vT,z,vz= 1.01,0.01,1.,0.01,0.01 
    js= aAA(Orbit([R,vR,vT,z,vz]))
    assert numpy.fabs(js[0]) < 10.**-4., 'Close-to-circular orbit in the MWPotential does not have small Jr'
    assert numpy.fabs(js[2]) < 10.**-3., 'Close-to-circular orbit in the MWPotential does not have small Jz'
    te,tzmax,_,_= aAA.EccZmaxRperiRap(R,vR,vT,z,vz)
    assert numpy.fabs(te) < 10.**-2., 'Close-to-circular orbit in the MWPotential does not have small eccentricity'
    assert numpy.fabs(tzmax) < 2.*10.**-2., 'Close-to-circular orbit in the MWPotential does not have small zmax'
    #Check that actions are conserved along the orbit
    obs= Orbit([1.05, 0.02, 1.05, 0.03,0.])
    check_actionAngle_conserved_actions(aAA,obs,MWPotential,
                                        -1.2,-8.,-1.7,ntimes=101)
    # and the eccentricity etc.
    check_actionAngle_conserved_EccZmaxRperiRap(aAA,obs,MWPotential,
                                                -2.,-2.,-2.,-2.,ntimes=101)
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

#Basic sanity checking of the actionAngleStaeckelGrid eccentricity etc.
def test_actionAngleStaeckelGrid_basic_EccZmaxRperiRap_c():
    from galpy.actionAngle import actionAngleStaeckelGrid
    from galpy.potential import MWPotential, interpRZPotential
    from galpy.orbit import Orbit
    rzpot= interpRZPotential(RZPot=MWPotential,
                             rgrid=(numpy.log(0.01),numpy.log(20.),201),
                             logR=True,
                             zgrid=(0.,1.,101),
                             interpPot=True,use_c=True,enable_c=True,
                             zsym=True)
    aAA= actionAngleStaeckelGrid(pot=rzpot,delta=0.71,c=True,interpecc=True)
    #circular orbit
    R,vR,vT,z,vz= 1.,0.,1.,0.,0. 
    te,tzmax,_,_= aAA.EccZmaxRperiRap(R,vR,vT,z,vz)
    assert numpy.fabs(te) < 10.**-16., 'Circular orbit in the MWPotential does not have e=0'
    assert numpy.fabs(tzmax) < 10.**-16., 'Circular orbit in the MWPotential does not have zmax=0'
    #Close-to-circular orbit
    R,vR,vT,z,vz= 1.01,0.01,1.,0.01,0.01
    te,tzmax,_,_= aAA.EccZmaxRperiRap(R,vR,vT,z,vz)
    assert numpy.fabs(te) < 10.**-2., 'Close-to-circular orbit in the MWPotential does not have small eccentricity'
    assert numpy.fabs(tzmax) < 2.*10.**-2., 'Close-to-circular orbit in the MWPotential does not have small zmax'
    #Another close-to-circular orbit
    R,vR,vT,z,vz= 1.0,0.0,0.99,0.0,0.0
    te,tzmax,_,_= aAA.EccZmaxRperiRap(R,vR,vT,z,vz)
    assert numpy.fabs(te) < 10.**-2., 'Close-to-circular orbit in the MWPotential does not have small eccentricity'
    assert numpy.fabs(tzmax) < 2.*10.**-2., 'Close-to-circular orbit in the MWPotential does not have small zmax'
    #Another close-to-circular orbit
    R,vR,vT,z,vz= 1.0,0.0,1.,0.01,0.0
    te,tzmax,_,_= aAA.EccZmaxRperiRap(Orbit([R,vR,vT,z,vz]))
    assert numpy.fabs(te) < 10.**-2., 'Close-to-circular orbit in the MWPotential does not have small eccentricity'
    assert numpy.fabs(tzmax) < 2.*10.**-2., 'Close-to-circular orbit in the MWPotential does not have small zmax'
    return None

#Test the actions of an actionAngleStaeckel
def test_actionAngleStaeckelGrid_conserved_EccZmaxRperiRap_c():
    from galpy.potential import MWPotential
    from galpy.actionAngle import actionAngleStaeckelGrid
    from galpy.orbit import Orbit
    obs= Orbit([1.05, 0.02, 1.05, 0.03,0.,2.])
    aAA= actionAngleStaeckelGrid(pot=MWPotential,delta=0.71,c=True,
                                 interpecc=True)
    check_actionAngle_conserved_EccZmaxRperiRap(aAA,obs,MWPotential,
                                                -2.,-2.,-2.,-2.,ntimes=101,
                                                inclphi=True)
    return None

#Test the actionAngleIsochroneApprox against an isochrone potential: actions
def test_actionAngleIsochroneApprox_otherIsochrone_actions():
    from galpy.potential import IsochronePotential
    from galpy.actionAngle import actionAngleIsochroneApprox, \
        actionAngleIsochrone
    from galpy.orbit.Orbits import ext_loaded
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
    from galpy.orbit.Orbits import ext_loaded
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
    from galpy.orbit.Orbits import ext_loaded
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
    aAI.plot(numpy.reshape(obs.R(obs.t),(1,len(obs.t))),
             numpy.reshape(obs.vR(obs.t),(1,len(obs.t))),
             numpy.reshape(obs.vT(obs.t),(1,len(obs.t))),
             numpy.reshape(obs.z(obs.t),(1,len(obs.t))),
             numpy.reshape(obs.vz(obs.t),(1,len(obs.t))),
             numpy.reshape(obs.phi(obs.t),(1,len(obs.t))),
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

#Test the Orbit interface
def test_orbit_interface_spherical():
    from galpy.potential import LogarithmicHaloPotential, NFWPotential
    from galpy.orbit import Orbit
    from galpy.actionAngle import actionAngleSpherical
    lp= LogarithmicHaloPotential(normalize=1.,q=1.)
    obs= Orbit([1., 0.2, 1.5, 0.3,0.1,2.])
    # resetaA has been deprecated
    #assert not obs.resetaA(), 'obs.resetaA() does not return False when called before having set up an actionAngle instance'
    aAS= actionAngleSpherical(pot=lp)
    acfs= numpy.array(list(aAS.actionsFreqsAngles(obs))).reshape(9)
    type= 'spherical'
    try:
        obs.jr(type=type)
    except AttributeError:
        pass #should raise this, as we have not specified a potential
    else:
        raise AssertionError('obs.jr w/o pot= does not raise AttributeError before the orbit was integrated')
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
    # resetaA has been deprecated
    #assert obs.resetaA(pot=np), 'obs.resetaA() does not return True after having set up an actionAngle instance'
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
    #Directly test _resetaA --> deprecated
    #assert obs._orb._resetaA(pot=lp), 'OrbitTop._resetaA does not return True when resetting the actionAngle instance'
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
                        obs.jp(pot=MWPotential,type=type,delta=0.71),
                        obs.jz(pot=MWPotential,type=type,delta=0.71),
                        obs.Or(pot=MWPotential,type=type,delta=0.71),
                        obs.Op(pot=MWPotential,type=type,delta=0.71),
                        obs.Oz(pot=MWPotential,type=type,delta=0.71),
                        obs.wr(pot=MWPotential,type=type,delta=0.71),
                        obs.wp(pot=MWPotential,type=type,delta=0.71),
                        obs.wz(pot=MWPotential,type=type,delta=0.71)])
    maxdev= numpy.amax(numpy.abs(acfs-acfso))
    assert maxdev < 10.**-16., 'Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface'
    return None

# Further tests of the Orbit interface for actionAngleStaeckel
def test_orbit_interface_staeckel_defaultdelta():
    from galpy.potential import MWPotential2014
    from galpy.orbit import Orbit
    from galpy.actionAngle import actionAngleStaeckel, estimateDeltaStaeckel
    obs= Orbit([1.05, 0.02, 1.05, 0.03,0.,2.])
    est_delta= estimateDeltaStaeckel(MWPotential2014,obs.R(),obs.z())
    # Just need to trigger delta estimation in orbit
    jr_orb= obs.jr(pot=MWPotential2014,type='staeckel')
    assert numpy.fabs(est_delta-obs._aA._delta) < 1e-10, 'Directly estimated delta does not agree with Orbit-interface-estimated delta'
    aAS= actionAngleStaeckel(pot=MWPotential2014,delta=est_delta)
    acfs= numpy.array(list(aAS.actionsFreqsAngles(obs))).reshape(9)
    type= 'staeckel'
    acfso= numpy.array([obs.jr(pot=MWPotential2014,type=type),
                        obs.jp(pot=MWPotential2014,type=type),
                        obs.jz(pot=MWPotential2014,type=type),
                        obs.Or(pot=MWPotential2014,type=type),
                        obs.Op(pot=MWPotential2014,type=type),
                        obs.Oz(pot=MWPotential2014,type=type),
                        obs.wr(pot=MWPotential2014,type=type),
                        obs.wp(pot=MWPotential2014,type=type),
                        obs.wz(pot=MWPotential2014,type=type)])
    maxdev= numpy.amax(numpy.abs(acfs-acfso))
    assert maxdev < 10.**-16., 'Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface'
    return None

def test_orbit_interface_staeckel_PotentialErrors():
    # staeckel approx. w/ automatic delta should fail if delta cannot be found
    from galpy.potential import TwoPowerSphericalPotential, SpiralArmsPotential
    from galpy.potential import PotentialError
    from galpy.orbit import Orbit
    obs= Orbit([1.05, 0.02, 1.05, 0.03,0.,2.])
    # Currently doesn't have second derivs
    tp= TwoPowerSphericalPotential(normalize=1.,alpha=1.2,beta=2.5)
    # Check that this potential indeed does not have second derivs
    with pytest.raises(PotentialError) as excinfo:
        dummy= tp.R2deriv(1.,0.1)
        pytest.fail('TwoPowerSphericalPotential appears to now have second derivatives, means that it cannot be used to test exceptions based on not having the second derivatives any longer')
    # Now check that estimating delta fails
    with pytest.raises(PotentialError) as excinfo:
        obs.jr(pot=tp,type='staeckel')
        pytest.fail('TwoPowerSphericalPotential appears to now have second derivatives, means that it cannot be used to test exceptions based on not having the second derivatives any longer')
    assert 'second derivatives' in str(excinfo.value), 'Estimating delta for potential lacking second derivatives should have failed with a message about the lack of second derivatives'
    # Generic non-axi
    sp= SpiralArmsPotential()
    with pytest.raises(PotentialError) as excinfo:
        obs.jr(pot=sp,type='staeckel')
        pytest.fail('TwoPowerSphericalPotential appears to now have second derivatives, means that it cannot be used to test exceptions based on not having the second derivatives any longer')
    assert 'not axisymmetric' in str(excinfo.value), 'Estimating delta for a non-axi potential should have failed with a message about the fact that the potential is non-axisymmetric'
    return None

def test_orbits_interface_staeckel_PotentialErrors():
    # staeckel approx. w/ automatic delta should fail if delta cannot be found
    from galpy.potential import TwoPowerSphericalPotential, SpiralArmsPotential
    from galpy.potential import PotentialError
    from galpy.orbit import Orbit
    obs= Orbit([[1.05, 0.02, 1.05, 0.03,0.,2.],
                [1.15, -0.02, 1.02, -0.03,0.,2.]])
    # Currently doesn't have second derivs
    tp= TwoPowerSphericalPotential(normalize=1.,alpha=1.2,beta=2.5)
    # Check that this potential indeed does not have second derivs
    with pytest.raises(PotentialError) as excinfo:
        dummy= tp.R2deriv(1.,0.1)
        pytest.fail('TwoPowerSphericalPotential appears to now have second derivatives, means that it cannot be used to test exceptions based on not having the second derivatives any longer')
    # Now check that estimating delta fails
    with pytest.raises(PotentialError) as excinfo:
        obs.jr(pot=tp,type='staeckel')
        pytest.fail('TwoPowerSphericalPotential appears to now have second derivatives, means that it cannot be used to test exceptions based on not having the second derivatives any longer')
    assert 'second derivatives' in str(excinfo.value), 'Estimating delta for potential lacking second derivatives should have failed with a message about the lack of second derivatives'
    # Generic non-axi
    sp= SpiralArmsPotential()
    with pytest.raises(PotentialError) as excinfo:
        obs.jr(pot=sp,type='staeckel')
        pytest.fail('TwoPowerSphericalPotential appears to now have second derivatives, means that it cannot be used to test exceptions based on not having the second derivatives any longer')
    assert 'not axisymmetric' in str(excinfo.value), 'Estimating delta for a non-axi potential should have failed with a message about the fact that the potential is non-axisymmetric'
    return None

# Test the Orbit interface for actionAngleAdiabatic
# currently fails bc actionAngleAdiabatic doesn't have actionsFreqsAngles
@pytest.mark.xfail(raises=NotImplementedError,strict=True)
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
    acfs= aAS.actionsFreqsAngles([obs()])
    acfs= numpy.array(acfs).reshape(9)
    type= 'isochroneApprox'
    acfso= numpy.array([obs.jr(pot=MWPotential,type=type,b=0.8),
                        obs.jp(pot=MWPotential,type=type,b=0.8),
                        obs.jz(pot=MWPotential,type=type,b=0.8),
                        obs.Or(pot=MWPotential,type=type,b=0.8),
                        obs.Op(pot=MWPotential,type=type,b=0.8),
                        obs.Oz(pot=MWPotential,type=type,b=0.8),
                        obs.wr(pot=MWPotential,type=type,b=0.8),
                        obs.wp(pot=MWPotential,type=type,b=0.8),
                        obs.wz(pot=MWPotential,type=type,b=0.8)])
    maxdev= numpy.amax(numpy.abs(acfs-acfso))
    assert maxdev < 10.**-16., 'Orbit interface for actionAngleIsochroneApprox does not return the same as actionAngle interface'
    assert numpy.abs(obs.Tr(pot=MWPotential,type=type,b=0.8)-2.*numpy.pi/acfso[3]) < 10.**-16., \
        'Orbit.Tr does not agree with actionAngleIsochroneApprox frequency'
    assert numpy.abs(obs.Tp(pot=MWPotential,type=type,b=0.8)-2.*numpy.pi/acfso[4]) < 10.**-16., \
        'Orbit.Tp does not agree with actionAngleIsochroneApprox frequency'
    assert numpy.abs(obs.Tz(pot=MWPotential,type=type,b=0.8)-2.*numpy.pi/acfso[5]) < 10.**-16., \
        'Orbit.Tz does not agree with actionAngleIsochroneApprox frequency'
    assert numpy.abs(obs.TrTp(pot=MWPotential,type=type,b=0.8)-acfso[4]/acfso[3]*numpy.pi) < 10.**-16., \
        'Orbit.TrTp does not agree with actionAngleIsochroneApprox frequency'
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
        if PY2: reset_warning_registry('galpy')
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
        if PY2: reset_warning_registry('galpy')
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
        if PY2: reset_warning_registry('galpy')
        warnings.simplefilter("always",galpyWarning)
        aAA= actionAngleIsochroneApprox(pot=MWPotential,b=1.)
        # Should raise warning bc of MWPotential, might raise others
        raisedWarning= False
        for wa in w:
            raisedWarning= (str(wa.message) == "Use of MWPotential as a Milky-Way-like potential is deprecated; galpy.potential.MWPotential2014, a potential fit to a large variety of dynamical constraints (see Bovy 2015), is the preferred Milky-Way-like potential in galpy")
            if raisedWarning: break
        assert raisedWarning, "actionAngleIsochroneApprox with MWPotential should have thrown a warning, but didn't"
    return None

# Test of the fix to issue 361
def test_actionAngleAdiabatic_issue361():
    from galpy.potential import MWPotential2014
    from galpy import actionAngle
    aA_adi = actionAngle.actionAngleAdiabatic(pot=MWPotential2014, c=True) 
    R = 8.7007/8.
    vT = 188.5/220.
    jr_good,_,_= aA_adi(R, -0.1/220., vT, 0, 0)
    jr_bad,_,_= aA_adi(R, -0.09/220., vT, 0, 0)
    assert numpy.fabs(jr_good-jr_bad) < 1e-6, 'Nearby JR for orbit near apocenter disagree too much, likely because one completely fails: Jr_good = {}, Jr_bad = {}'.format(jr_good,jr_bad)
    return None

# Test that evaluating actionAngle with multi-dimensional orbit doesn't work
def test_actionAngle_orbitInput_multid_error():
    from galpy.potential import MWPotential2014
    from galpy.orbit import Orbit
    from galpy.actionAngle import actionAngleStaeckel
    orbits= Orbit(numpy.array([[[1.,0.1,1.1,-0.1,-0.2,0.],
                                [1.,0.2,1.2,0.,-0.1,1.]],
                               [[1.,-0.2,0.9,0.2,0.2,2.],
                                [1.2,-0.4,1.1,-0.1,0.,-2.]],
                               [[1., 0.2,0.9,0.3,-0.2,0.1],
                                [1.2, 0.4,1.1,-0.2,0.05,4.]]]))
    aAS= actionAngleStaeckel(pot=MWPotential2014,delta=0.45,c=True)
    with pytest.raises(RuntimeError) as excinfo:
        aAS(orbits)
        pytest.fail('Evaluating actionAngle methods with Orbit instances with multi-dimensional shapes is not support')
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
        # Test Orbit with multiple objects case, but calling
        js= aA(obs(times))
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
                                                ntimes=1001,inclphi=False):
    times= numpy.linspace(0.,100.,ntimes)
    obs.integrate(times,pot,method='dopr54_c')
    if inclphi:
        es,zmaxs,rperis,raps= aA.EccZmaxRperiRap(\
            obs.R(times),obs.vR(times),obs.vT(times),obs.z(times),
            obs.vz(times),obs.phi(times))
    else:
        es,zmaxs,rperis,raps= aA.EccZmaxRperiRap(\
            obs.R(times),obs.vR(times),obs.vT(times),obs.z(times),
            obs.vz(times))
    assert numpy.amax(numpy.fabs(es/numpy.mean(es)-1)) < 10.**tole, 'Eccentricity conservation fails at %g%%' % (100.*numpy.amax(numpy.fabs(es/numpy.mean(es)-1)))
    assert numpy.amax(numpy.fabs(zmaxs/numpy.mean(zmaxs)-1)) < 10.**tolzmax, 'Zmax conservation fails at %g%%' % (100.*numpy.amax(numpy.fabs(zmaxs/numpy.mean(zmaxs)-1)))
    assert numpy.amax(numpy.fabs(rperis/numpy.mean(rperis)-1)) < 10.**tolrperi, 'Rperi conservation fails at %g%%' % (100.*numpy.amax(numpy.fabs(rperis/numpy.mean(rperis)-1)))
    assert numpy.amax(numpy.fabs(raps/numpy.mean(raps)-1)) < 10.**tolrap, 'Rap conservation fails at %g%%' % (100.*numpy.amax(numpy.fabs(raps/numpy.mean(raps)-1)))
    return None

# Python 2 bug: setting simplefilter to 'always' still does not display 
# warnings that were already displayed using 'once' or 'default', so some
# warnings tests fail; need to reset the registry
# Has become an issue at pytest 3.8.0, which seems to have changed the scope of
# filterwarnings (global one at the start is ignored)
def reset_warning_registry(pattern=".*"):
    "clear warning registry for all match modules"
    import re
    import sys
    key = "__warningregistry__"
    for mod in sys.modules.values():
        if hasattr(mod, key) and re.match(pattern, mod.__name__):
            getattr(mod, key).clear()

