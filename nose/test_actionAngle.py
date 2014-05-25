import numpy

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
    #Very eccentric orbit
    R,vR,vT,z,vz= 1.,2.,1.,2.,1.
    js= aAI(R,vR,vT,z,vz)
    assert numpy.fabs(js[0]) > 10.**1., 'Very eccentric orbit in the isochrone potential does not have large Jr'
    assert numpy.fabs(js[2]) < 10.**1., 'Very eccentric orbit in the isochrone potential does not have large Jz'
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
    check_actionAngle_linear_angles(aAI,obs,ip,
                                    -6.,-6.,-6.,
                                    -8.,-8.,-8.,
                                    -8.,-8.,-8.)
    return None

   
#Test the actionAngleIsochroneApprox against an isochrone potential: actions
def test_actionAngleIsochroneApprox_otherIsochrone_actions():
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
    acfs= numpy.array(list(aAI.actionsFreqsAngles(obs))).flatten()
    acfsip= numpy.array(list(aAIip.actionsFreqsAngles(obs))).flatten()
    acfsaAIip= numpy.array(list(aAIaAIip.actionsFreqsAngles(obs))).flatten()
    acfsrk6= numpy.array(list(aAIrk6.actionsFreqsAngles(obs))).flatten()
    acfslong= numpy.array(list(aAIlong.actionsFreqsAngles(obs))).flatten()
    acfsmany= numpy.array(list(aAImany.actionsFreqsAngles(obs))).flatten()
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
            obs.vz(times),obs.phi(times),nonaxi=True) #nonaxi to test that part of the code
    maxdj= numpy.amax(numpy.fabs((js-numpy.tile(numpy.mean(js,axis=1),(len(times),1)).T)),axis=1)/numpy.mean(js,axis=1)
    assert maxdj[0] < 3.*10.**-2., 'Jr conservation for the GD-1 like orbit of Bovy (2014) fails at %f%%' % (100.*maxdj[0])
    assert maxdj[1] < 10.**-2., 'Lz conservation for the GD-1 like orbit of Bovy (2014) fails at %f%%' % (100.*maxdj[1])
    assert maxdj[2] < 2.*10.**-2., 'Jz conservation for the GD-1 like orbit of Bovy (2014) fails at %f%%' % (100.*maxdj[2])
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
    aAI.plot(obs,type='lz')
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
    bmin, bmed, bmax= estimateBIsochrone(o.R(times),o.z(times),pot=ip)
    assert numpy.fabs(bmed-1.2) < 10.**-15., \
        'Estimated scale parameter b when estimateBIsochrone is applied to an IsochronePotential is wrong'
    return None

#Test that the actions are conserved along an orbit
def check_actionAngle_conserved_actions(aA,obs,pot,toljr,toljp,toljz):
    times= numpy.linspace(0.,100.,1001)
    obs.integrate(times,pot,method='dopr54_c')
    js= aA(obs.R(times),obs.vR(times),obs.vT(times),obs.z(times),
           obs.vz(times))
    maxdj= numpy.amax(numpy.fabs((js-numpy.tile(numpy.mean(js,axis=1),(len(times),1)).T)),axis=1)/numpy.mean(js,axis=1)
    assert maxdj[0] < 10.**toljr, 'Jr conservation fails at %g%%' % (100.*maxdj[0])
    assert maxdj[1] < 10.**toljp, 'Lz conservation fails at %g%%' % (100.*maxdj[1])
    assert maxdj[2] < 10.**-toljp, 'Jz conservation fails at %g%%' % (100.*maxdj[2])
    return None

#Test that the angles increase linearly
def check_actionAngle_linear_angles(aA,obs,pot,
                                    tolinitar,tolinitap,tolinitaz,
                                    tolor,tolop,toloz,
                                    toldar,toldap,toldaz):
    from galpy.actionAngle import dePeriod
    times= numpy.linspace(0.,100.,1001)
    obs.integrate(times,pot,method='dopr54_c')
    acfs_init= aA.actionsFreqsAngles(obs) #to check the init. angles
    acfs= aA.actionsFreqsAngles(obs.R(times),obs.vR(times),obs.vT(times),
                                obs.z(times),obs.vz(times),obs.phi(times))
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
    assert maxdev < 10.**toldap, 'Maximum deviation from linear trend in the azimuthal angles is %g' % maxdev
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
