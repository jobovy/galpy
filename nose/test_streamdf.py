import functools
import nose
import numpy
from scipy import interpolate
sdf_bovy14= None #so we can set this up and then use in other tests

# Decorator for expected failure
def expected_failure(test):
    @functools.wraps(test)
    def inner(*args, **kwargs):
        try:
            test(*args, **kwargs)
        except Exception:
            raise nose.SkipTest
        else:
            raise AssertionError('Test is expected to fail, but passed instead')
    return inner

#Exact setup from Bovy (2014); should reproduce those results (which have been
# sanity checked
def test_bovy14_setup():
    #Imports
    from galpy.df import streamdf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.util import bovy_conversion #for unit conversions
    lp= LogarithmicHaloPotential(normalize=1.,q=0.9)
    aAI= actionAngleIsochroneApprox(pot=lp,b=0.8)
    obs= Orbit([1.56148083,0.35081535,-1.15481504,
                0.88719443,-0.47713334,0.12019596])
    sigv= 0.365 #km/s
    global sdf_bovy14
    sdf_bovy14= streamdf(sigv/220.,progenitor=obs,pot=lp,aA=aAI,
                         leading=True,
                         nTrackChunks=11,
                         tdisrupt=4.5/bovy_conversion.time_in_Gyr(220.,8.))
    assert not sdf_bovy14 is None, 'bovy14 streamdf setup did not work'
    return None

def test_bovy14_freqratio():
    #Test the frequency ratio
    assert (sdf_bovy14.freqEigvalRatio()-30.)**2. < 10.**0., 'streamdf model from Bovy (2014) does not give a frequency ratio of about 30'
    assert (sdf_bovy14.freqEigvalRatio(isotropic=True)-34.)**2. < 10.**0., 'streamdf model from Bovy (2014) does not give an isotropic frequency ratio of about 34'
    return None

def test_bovy14_misalignment():
    #Test the misalignment
    assert (sdf_bovy14.misalignment()+0.5)**2. <10.**-2., 'streamdf model from Bovy (2014) does not give a misalighment of about -0.5 degree'
    assert (sdf_bovy14.misalignment(isotropic=True)-1.3)**2. <10.**-2., 'streamdf model from Bovy (2014) does not give an isotropic misalighment of about 1.3 degree'
    return None

def test_bovy14_track_prog_diff():
    #Test that the stream and the progenitor are close together, for both leading and trailing
    check_track_prog_diff(sdf_bovy14,'R','Z',0.1)
    check_track_prog_diff(sdf_bovy14,'R','Z',0.8,phys=True) #do 1 with phys
    check_track_prog_diff(sdf_bovy14,'R','X',0.1)
    check_track_prog_diff(sdf_bovy14,'R','Y',0.1)
    check_track_prog_diff(sdf_bovy14,'R','vZ',0.03)
    check_track_prog_diff(sdf_bovy14,'R','vZ',6.6,phys=True) #do 1 with phys
    check_track_prog_diff(sdf_bovy14,'R','vX',0.05)
    check_track_prog_diff(sdf_bovy14,'R','vY',0.05)
    check_track_prog_diff(sdf_bovy14,'R','vT',0.05)
    check_track_prog_diff(sdf_bovy14,'R','vR',0.05)
    check_track_prog_diff(sdf_bovy14,'ll','bb',0.3)
    check_track_prog_diff(sdf_bovy14,'ll','dist',0.5)
    check_track_prog_diff(sdf_bovy14,'ll','vlos',4.)
    check_track_prog_diff(sdf_bovy14,'ll','pmll',0.3)
    check_track_prog_diff(sdf_bovy14,'ll','pmbb',0.25)
    return None

def test_bovy14_track_spread():
    #Test that the spreads are small
    check_track_spread(sdf_bovy14,'R','Z',0.01,0.005)
    check_track_spread(sdf_bovy14,'R','Z',0.08,0.04,phys=True) #do 1 with phys
    check_track_spread(sdf_bovy14,'R','Z',0.01,0.005,interp=False) #do 1 with interp
    check_track_spread(sdf_bovy14,'X','Y',0.01,0.005)
    check_track_spread(sdf_bovy14,'X','Y',0.08,0.04,phys=True) #do 1 with phys
    check_track_spread(sdf_bovy14,'R','phi',0.01,0.005)
    check_track_spread(sdf_bovy14,'vR','vT',0.005,0.005)
    check_track_spread(sdf_bovy14,'vR','vT',1.1,1.1,phys=True) #do 1 with phys
    check_track_spread(sdf_bovy14,'vR','vZ',0.005,0.005)
    check_track_spread(sdf_bovy14,'vX','vY',0.005,0.005)
    check_track_spread(sdf_bovy14,'ll','bb',0.5,0.5)
    check_track_spread(sdf_bovy14,'dist','vlos',0.5,5.)
    check_track_spread(sdf_bovy14,'pmll','pmbb',0.5,0.5)
    return None

def test_closest_trackpoint():
    #Check that we can find the closest trackpoint properly
    check_closest_trackpoint(sdf_bovy14,50)
    check_closest_trackpoint(sdf_bovy14,230,usev=True)
    check_closest_trackpoint(sdf_bovy14,330,usev=True,xy=False)
    check_closest_trackpoint(sdf_bovy14,40,xy=False)
    check_closest_trackpoint(sdf_bovy14,4,interp=False)
    check_closest_trackpoint(sdf_bovy14,6,interp=False,usev=True,xy=False)
    return None

def test_closest_trackpointLB():
    #Check that we can find the closest trackpoint properly in LB
    check_closest_trackpointLB(sdf_bovy14,50)
    check_closest_trackpointLB(sdf_bovy14,230,usev=True)
    check_closest_trackpointLB(sdf_bovy14,4,interp=False)
    check_closest_trackpointLB(sdf_bovy14,8,interp=False,usev=True)
    check_closest_trackpointLB(sdf_bovy14,-1,interp=False,usev=False)
    check_closest_trackpointLB(sdf_bovy14,-2,interp=False,usev=True)
    check_closest_trackpointLB(sdf_bovy14,-3,interp=False,usev=True)
    return None
    
def test_closest_trackpointaA():
    #Check that we can find the closest trackpoint properly in AA
    check_closest_trackpointaA(sdf_bovy14,50)
    check_closest_trackpointaA(sdf_bovy14,4,interp=False)
    return None

def test_meanOmega():
    #Test that meanOmega is close to constant and the mean Omega close to the progenitor
    assert numpy.all(numpy.fabs(sdf_bovy14.meanOmega(0.1)-sdf_bovy14._progenitor_Omega) < 10.**-2.), 'meanOmega near progenitor not close to mean Omega for Bovy14 stream'
    assert numpy.all(numpy.fabs(sdf_bovy14.meanOmega(0.5)-sdf_bovy14._progenitor_Omega) < 10.**-2.), 'meanOmega near progenitor not close to mean Omega for Bovy14 stream'
    return None

def test_meanOmega_oned():
    #Test that meanOmega is close to constant and the mean Omega close to the progenitor
    assert numpy.fabs(sdf_bovy14.meanOmega(0.1,oned=True)) < 10.**-2., 'One-dimensional meanOmega near progenitor not close to zero for Bovy14 stream'
    assert numpy.fabs(sdf_bovy14.meanOmega(0.5,oned=True)) < 10.**-2., 'Oned-dimensional meanOmega near progenitor not close to zero for Bovy14 stream'
    return None

def test_sigOmega_constant():
    #Test that sigOmega is close to constant close to the progenitor
    assert numpy.fabs(sdf_bovy14.sigOmega(0.1)-sdf_bovy14.sigOmega(0.5)) < 10.**-4., 'sigOmega near progenitor not close to constant for Bovy14 stream'
    return None

def test_sigOmega_small():
    #Test that sigOmega is smaller than the total spread
    assert sdf_bovy14.sigOmega(0.1) < numpy.sqrt(sdf_bovy14._sortedSigOEig[2]), 'sigOmega near progenitor not smaller than the total Omega spread'
    assert sdf_bovy14.sigOmega(0.5) < numpy.sqrt(sdf_bovy14._sortedSigOEig[2]), 'sigOmega near progenitor not smaller than the total Omega spread'
    assert sdf_bovy14.sigOmega(1.2) < numpy.sqrt(sdf_bovy14._sortedSigOEig[2]), 'sigOmega near progenitor not smaller than the total Omega spread'
    return None

def test_meantdAngle():
    #Test that the mean td for a given angle is close to what's expected
    assert numpy.fabs((sdf_bovy14.meantdAngle(0.1)-0.1/sdf_bovy14._meandO)/sdf_bovy14.meantdAngle(0.1)) < 10.**-1.5, 'mean td close to the progenitor is not dangle/dO'
    assert numpy.fabs((sdf_bovy14.meantdAngle(0.4)-0.4/sdf_bovy14._meandO)/sdf_bovy14.meantdAngle(0.1)) < 10.**-0.9, 'mean td close to the progenitor is not dangle/dO'
    return None

def test_sigtdAngle():
    #Test that the sigma of td for a given angle is small
    assert sdf_bovy14.sigtdAngle(0.1) < 0.2*0.1/sdf_bovy14._meandO, 'sigma of td close to the progenitor is not small'
    assert sdf_bovy14.sigtdAngle(0.5) > 0.2*0.1/sdf_bovy14._meandO, 'sigma of td in the middle of the stream is not large'
    return None

def test_ptdAngle():
    #Test that the probability distribution for p(td|angle) is reasonable
    #at 0.1
    da= 0.1
    expected_max= da/sdf_bovy14._meandO
    assert sdf_bovy14.ptdAngle(expected_max,da) > \
        sdf_bovy14.ptdAngle(2.*expected_max,da), 'ptdAngle does not peak close to where it is expected to peak'
    assert sdf_bovy14.ptdAngle(expected_max,da) > \
        sdf_bovy14.ptdAngle(0.5*expected_max,da), 'ptdAngle does not peak close to where it is expected to peak'
    #at 0.6
    da= 0.6
    expected_max= da/sdf_bovy14._meandO
    assert sdf_bovy14.ptdAngle(expected_max,da) > \
        sdf_bovy14.ptdAngle(2.*expected_max,da), 'ptdAngle does not peak close to where it is expected to peak'
    assert sdf_bovy14.ptdAngle(expected_max,da) > \
        sdf_bovy14.ptdAngle(0.5*expected_max,da), 'ptdAngle does not peak close to where it is expected to peak'
    #Now test that the mean and sigma calculated with a simple Riemann sum agrees with meantdAngle
    da= 0.2
    ts= numpy.linspace(0.,100.,1001)
    pts= numpy.array([sdf_bovy14.ptdAngle(t,da) for t in ts])
    assert numpy.fabs((numpy.sum(ts*pts)/numpy.sum(pts)\
                           -sdf_bovy14.meantdAngle(da))/sdf_bovy14.meantdAngle(da)) < 10.**-2., 'mean td at angle 0.2 calculated with Riemann sum does not agree with that calculated by meantdAngle'
    assert numpy.fabs((numpy.sqrt(numpy.sum(ts**2.*pts)/numpy.sum(pts)-(numpy.sum(ts*pts)/numpy.sum(pts))**2.)\
                           -sdf_bovy14.sigtdAngle(da))/sdf_bovy14.sigtdAngle(da)) < 10.**-1.5, 'sig td at angle 0.2 calculated with Riemann sum does not agree with that calculated by meantdAngle'
    return None

def test_meanangledAngle():
    #Test that the mean perpendicular angle at a given angle is zero
    da= 0.1
    assert numpy.fabs(sdf_bovy14.meanangledAngle(da,smallest=False)) < 10.**-2, 'mean perpendicular angle not zero'
    assert numpy.fabs(sdf_bovy14.meanangledAngle(da,smallest=True)) < 10.**-2, 'mean perpendicular angle not zero'
    da= 1.1
    assert numpy.fabs(sdf_bovy14.meanangledAngle(da,smallest=False)) < 10.**-2, 'mean perpendicular angle not zero'
    assert numpy.fabs(sdf_bovy14.meanangledAngle(da,smallest=True)) < 10.**-2, 'mean perpendicular angle not zero'
    return None

def test_sigangledAngle():
    #Test that the spread in perpendicular angle is much smaller than 1 (the typical spread in the parallel angle)
    da= 0.1
    assert sdf_bovy14.sigangledAngle(da,assumeZeroMean=True,smallest=False,
                                     simple=False) \
                                     < 1./sdf_bovy14.freqEigvalRatio(), \
                                     'spread in perpendicular angle is not small'
    assert sdf_bovy14.sigangledAngle(da,assumeZeroMean=True,smallest=True,
                                     simple=False) \
                                     < 1./sdf_bovy14.freqEigvalRatio(), \
                                     'spread in perpendicular angle is not small'
    da= 1.1
    assert sdf_bovy14.sigangledAngle(da,assumeZeroMean=True,smallest=False,
                                     simple=False) \
                                     < 1./sdf_bovy14.freqEigvalRatio(), \
                                     'spread in perpendicular angle is not small'
    assert sdf_bovy14.sigangledAngle(da,assumeZeroMean=True,smallest=True,
                                     simple=False) \
                                     < 1./sdf_bovy14.freqEigvalRatio(), \
                                     'spread in perpendicular angle is not small'
    #w/o assuming zeroMean
    da= 0.1
    assert sdf_bovy14.sigangledAngle(da,assumeZeroMean=False,smallest=False,
                                     simple=False) \
                                     < 1./sdf_bovy14.freqEigvalRatio(), \
                                     'spread in perpendicular angle is not small'
    assert sdf_bovy14.sigangledAngle(da,assumeZeroMean=False,smallest=True,
                                     simple=False) \
                                     < 1./sdf_bovy14.freqEigvalRatio(), \
                                     'spread in perpendicular angle is not small'
    #simple estimate
    da= 0.1
    assert sdf_bovy14.sigangledAngle(da,assumeZeroMean=False,smallest=False,
                                     simple=True) \
                                     < 1./sdf_bovy14.freqEigvalRatio(), \
                                     'spread in perpendicular angle is not small'
    assert sdf_bovy14.sigangledAngle(da,assumeZeroMean=False,smallest=True,
                                     simple=True) \
                                     < 1./sdf_bovy14.freqEigvalRatio(), \
                                     'spread in perpendicular angle is not small'
    return None

def test_pangledAngle():
    #Sanity check pangledAngle, does it peak near zero? Does the mean agree with meandAngle, does the sigma agree with sigdAngle?
    da= 0.1
    assert sdf_bovy14.pangledAngle(0.,da,smallest=False) > \
        sdf_bovy14.pangledAngle(0.1,da,smallest=False), 'pangledAngle does not peak near zero'
    assert sdf_bovy14.pangledAngle(0.,da,smallest=False) > \
        sdf_bovy14.pangledAngle(-0.1,da,smallest=False), 'pangledAngle does not peak near zero'
    #also for smallest
    assert sdf_bovy14.pangledAngle(0.,da,smallest=True) > \
        sdf_bovy14.pangledAngle(0.1,da,smallest=False), 'pangledAngle does not peak near zero'
    assert sdf_bovy14.pangledAngle(0.,da,smallest=True) > \
        sdf_bovy14.pangledAngle(-0.1,da,smallest=False), 'pangledAngle does not peak near zero'
    dangles= numpy.linspace(-0.01,0.01,201)
    pdangles= (numpy.array([sdf_bovy14.pangledAngle(pda,da,smallest=False) for pda in dangles])).flatten()               
    assert numpy.fabs(numpy.sum(dangles*pdangles)/numpy.sum(pdangles)) < 10.**-2., 'mean calculated using Riemann sum of pangledAngle does not agree with actual mean'
    acsig= sdf_bovy14.sigangledAngle(da,assumeZeroMean=True,smallest=False,simple=False)
    assert numpy.fabs((numpy.sqrt(numpy.sum(dangles**2.*pdangles)/numpy.sum(pdangles))-acsig)/acsig) < 10.**-2., 'sigma calculated using Riemann sum of pangledAngle does not agree with actual sigma'
    #also for smallest
    pdangles= (numpy.array([sdf_bovy14.pangledAngle(pda,da,smallest=True) for pda in dangles])).flatten()
    assert numpy.fabs(numpy.sum(dangles*pdangles)/numpy.sum(pdangles)) < 10.**-2., 'mean calculated using Riemann sum of pangledAngle does not agree with actual mean'
    acsig= sdf_bovy14.sigangledAngle(da,assumeZeroMean=True,smallest=True,simple=False)
    assert numpy.fabs((numpy.sqrt(numpy.sum(dangles**2.*pdangles)/numpy.sum(pdangles))-acsig)/acsig) < 10.**-1.95, 'sigma calculated using Riemann sum of pangledAngle does not agree with actual sigma'
    return None

def test_plotting():
    #Check plotting routines
    check_track_plotting(sdf_bovy14,'R','Z')
    check_track_plotting(sdf_bovy14,'R','Z',phys=True) #do 1 with phys
    check_track_plotting(sdf_bovy14,'R','Z',interp=False) #do 1 w/o interp
    check_track_plotting(sdf_bovy14,'R','X',spread=0)
    check_track_plotting(sdf_bovy14,'R','Y',spread=0)
    check_track_plotting(sdf_bovy14,'R','phi')
    check_track_plotting(sdf_bovy14,'R','vZ')
    check_track_plotting(sdf_bovy14,'R','vZ',phys=True) #do 1 with phys
    check_track_plotting(sdf_bovy14,'R','vZ',interp=False) #do 1 w/o interp
    check_track_plotting(sdf_bovy14,'R','vX',spread=0)
    check_track_plotting(sdf_bovy14,'R','vY',spread=0)
    check_track_plotting(sdf_bovy14,'R','vT')
    check_track_plotting(sdf_bovy14,'R','vR')
    check_track_plotting(sdf_bovy14,'ll','bb')
    check_track_plotting(sdf_bovy14,'ll','bb',interp=False) #do 1 w/o interp
    check_track_plotting(sdf_bovy14,'ll','dist')
    check_track_plotting(sdf_bovy14,'ll','vlos')
    check_track_plotting(sdf_bovy14,'ll','pmll')
    delattr(sdf_bovy14,'_ObsTrackLB') #rm, to test that this gets recalculated
    check_track_plotting(sdf_bovy14,'ll','pmbb')
    return None

@expected_failure
def test_diff_pot():
    raise AssertionError()

def check_track_prog_diff(sdf,d1,d2,tol,phys=False):
    observe= [sdf._R0,0.,sdf._Zsun]
    observe.extend(sdf._vsun)
    #Test that the stream and the progenitor are close together in Z
    trackR= sdf._parse_track_dim(d1,interp=True,phys=phys) #bit hacky to use private function
    trackZ= sdf._parse_track_dim(d2,interp=True,phys=phys) #bit hacky to use private function
    ts= sdf._progenitor._orb.t[sdf._progenitor._orb.t < sdf._trackts[-1]]
    progR= sdf._parse_progenitor_dim(d1,ts,
                                     ro=sdf._Rnorm,vo=sdf._Vnorm,
                                     obs=observe,
                                     phys=phys)
    progZ= sdf._parse_progenitor_dim(d2,ts,
                                     ro=sdf._Rnorm,vo=sdf._Vnorm,
                                     obs=observe,
                                     phys=phys)
    #Interpolate progenitor, st we can put it on the same grid as the stream
    interpProgZ= interpolate.InterpolatedUnivariateSpline(progR,progZ,k=3)
    maxdevZ= numpy.amax(numpy.fabs(interpProgZ(trackR)-trackZ))
    assert maxdevZ < tol, "Stream track deviates more from progenitor track in %s vs. %s than expected; max. deviation = %f" % (d2,d1,maxdevZ)
    return None

def check_track_spread(sdf,d1,d2,tol1,tol2,phys=False,interp=True):
    #Check that the spread around the track is small
    addx, addy= sdf._parse_track_spread(d1,d2,interp=interp,phys=phys) 
    assert numpy.amax(addx) < tol1, "Stream track spread is larger in %s than expected; max. deviation = %f" % (d1,numpy.amax(addx))
    assert numpy.amax(addy) < tol2, "Stream track spread is larger in %s than expected; max. deviation = %f" % (d2,numpy.amax(addy))
    return None

def check_track_plotting(sdf,d1,d2,phys=False,interp=True,spread=2,ls='-'):
    #Test that we can plot the stream track
    if not phys and d1 == 'R' and d2 == 'Z': #one w/ default
        sdf.plotTrack(d1=d1,d2=d2,interp=interp,spread=spread)
        sdf.plotProgenitor(d1=d1,d2=d2)
    else:
        sdf.plotTrack(d1=d1,d2=d2,interp=interp,spread=spread,
                      scaleToPhysical=phys,ls='none',linestyle='--',
                      color='k',lw=2.,marker='.')
        sdf.plotProgenitor(d1=d1,d2=d2,scaleToPhysical=phys)
    return None

def check_closest_trackpoint(sdf,trackp,usev=False,xy=True,interp=True):
    # Check that the closest trackpoint (close )to a trackpoint is the trackpoint
    if interp:
        if xy:
            RvR= sdf._interpolatedObsTrackXY[trackp,:]
        else:
            RvR= sdf._interpolatedObsTrack[trackp,:]
    else:
        if xy:
            RvR= sdf._ObsTrackXY[trackp,:]
        else:
            RvR= sdf._ObsTrack[trackp,:]
    R= RvR[0]
    vR= RvR[1]
    vT= RvR[2]
    z= RvR[3]
    vz= RvR[4]
    phi= RvR[5]
    indx= sdf.find_closest_trackpoint(R,vR,vT,z,vz,phi,interp=interp,
                                      xy=xy,usev=usev)
    assert indx == trackp, 'Closest trackpoint to a trackpoint is not that trackpoint'
    #Same test for a slight offset
    doff= 10.**-5.
    indx= sdf.find_closest_trackpoint(R+doff,vR+doff,vT+doff,
                                      z+doff,vz+doff,phi+doff,
                                      interp=interp,
                                      xy=xy,usev=usev)
    assert indx == trackp, 'Closest trackpoint to close to a trackpoint is not that trackpoint (%i,%i)' % (indx,trackp)
    return None

def check_closest_trackpointLB(sdf,trackp,usev=False,interp=True):
    # Check that the closest trackpoint (close )to a trackpoint is the trackpoint
    if trackp == -1: # in this case, rm some coordinates
        trackp= 1
        if interp:
            RvR= sdf._interpolatedObsTrackLB[trackp,:]
        else:
            RvR= sdf._ObsTrackLB[trackp,:]
        R= RvR[0]
        vR= None
        vT= RvR[2]
        z= None
        vz= RvR[4]
        phi= None
    elif trackp == -2: # in this case, rm some coordinates
        trackp= 1
        if interp:
            RvR= sdf._interpolatedObsTrackLB[trackp,:]
        else:
            RvR= sdf._ObsTrackLB[trackp,:]
        R= None
        vR= RvR[1]
        vT= None
        z= RvR[3]
        vz= None
        phi= RvR[5]
    elif trackp == -3: # in this case, rm some coordinates
        trackp= 1
        if interp:
            RvR= sdf._interpolatedObsTrackLB[trackp,:]
        else:
            RvR= sdf._ObsTrackLB[trackp,:]
        R= RvR[0]
        vR= RvR[1]
        vT= None
        z= None
        vz= None
        phi= None
    else:
        if interp:
            RvR= sdf._interpolatedObsTrackLB[trackp,:]
        else:
            RvR= sdf._ObsTrackLB[trackp,:]
        R= RvR[0]
        vR= RvR[1]
        vT= RvR[2]
        z= RvR[3]
        vz= RvR[4]
        phi= RvR[5]
    indx= sdf.find_closest_trackpointLB(R,vR,vT,z,vz,phi,interp=interp,
                                      usev=usev)
    assert indx == trackp, 'Closest trackpoint to a trackpoint is not that trackpoint in LB'
    #Same test for a slight offset
    doff= 10.**-5.
    if not R is None: R= R+doff
    if not vR is None: vR= vR+doff
    if not vT is None: vT= vT+doff
    if not z is None: z= z+doff
    if not vz  is None: vz= vz+doff
    if not phi is None: phi= phi+doff
    indx= sdf.find_closest_trackpointLB(R,vR,vT,z,vz,phi,
                                        interp=interp,
                                        usev=usev)
    assert indx == trackp, 'Closest trackpoint to close to a trackpoint is not that trackpoint in LB (%i,%i)' % (indx,trackp)
    return None

def check_closest_trackpointaA(sdf,trackp,interp=True):
    # Check that the closest trackpoint (close )to a trackpoint is the trackpoint
    if interp:
        RvR= sdf._interpolatedObsTrackAA[trackp,:]
    else:
        RvR= sdf._ObsTrackAA[trackp,:]
    #These aren't R,vR etc., but frequencies and angles
    R= RvR[0]
    vR= RvR[1]
    vT= RvR[2]
    z= RvR[3]
    vz= RvR[4]
    phi= RvR[5]
    indx= sdf._find_closest_trackpointaA(R,vR,vT,z,vz,phi,interp=interp)
    assert indx == trackp, 'Closest trackpoint to a trackpoint is not that trackpoint for AA'
    #Same test for a slight offset
    doff= 10.**-5.
    indx= sdf._find_closest_trackpointaA(R+doff,vR+doff,vT+doff,
                                        z+doff,vz+doff,phi+doff,
                                        interp=interp)
    assert indx == trackp, 'Closest trackpoint to close to a trackpoint is not that trackpoint for AA (%i,%i)' % (indx,trackp)
    return None

