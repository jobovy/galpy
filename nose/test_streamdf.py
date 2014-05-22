import functools
import nose
import numpy
from scipy import interpolate

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
def test_bovy14():
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
    sdfl= streamdf(sigv/220.,progenitor=obs,pot=lp,aA=aAI,leading=True,
                   nTrackChunks=11,
                   tdisrupt=4.5/bovy_conversion.time_in_Gyr(220.,8.))
    #Test the frequency ratio
    try: #Should be about 30
        assert((sdfl.freqEigvalRatio()-30.)**2. < 10.**0.)
    except AssertionError:
        raise AssertionError('streamdf model from Bovy (2014) does not give a frequency ratio of about 30')
    try: #Should be about 30
        assert((sdfl.freqEigvalRatio(isotropic=True)-34.)**2. < 10.**0.)
    except AssertionError:
        raise AssertionError('streamdf model from Bovy (2014) does not give an isotropic frequency ratio of about 34')
    #Test the misalignment
    try:
        assert((sdfl.misalignment()+0.5)**2. <10.**-2.)
    except AssertionError:
        raise AssertionError('streamdf model from Bovy (2014) does not give a misalighment of about -0.5 degree')
    try:
        assert((sdfl.misalignment(isotropic=True)-1.3)**2. <10.**-2.)
    except AssertionError:
        raise AssertionError('streamdf model from Bovy (2014) does not give an isotropic misalighment of about 1.3 degree')
    #Test that the stream and the progenitor are close together, for both leading and trailing
    check_track_prog_diff(sdfl,'R','Z',0.1)
    check_track_prog_diff(sdfl,'R','Z',0.8,phys=True) #do 1 with phys
    check_track_prog_diff(sdfl,'R','Y',0.1)
    check_track_prog_diff(sdfl,'R','vZ',0.03)
    check_track_prog_diff(sdfl,'R','vZ',6.6,phys=True) #do 1 with phys
    check_track_prog_diff(sdfl,'R','vX',0.05)
    check_track_prog_diff(sdfl,'R','vY',0.05)
    check_track_prog_diff(sdfl,'R','vT',0.05)
    check_track_prog_diff(sdfl,'R','vR',0.05)
    check_track_prog_diff(sdfl,'ll','bb',0.3)
    check_track_prog_diff(sdfl,'ll','dist',0.5)
    check_track_prog_diff(sdfl,'ll','vlos',4.)
    check_track_prog_diff(sdfl,'ll','pmll',0.3)
    check_track_prog_diff(sdfl,'ll','pmbb',0.25)
    #Test that the spreads are small
    check_track_spread(sdfl,'R','Z',0.01,0.005)
    check_track_spread(sdfl,'R','Z',0.08,0.04,phys=True) #do 1 with phys
    check_track_spread(sdfl,'X','Y',0.01,0.005)
    check_track_spread(sdfl,'R','phi',0.01,0.005)
    check_track_spread(sdfl,'vR','vT',0.005,0.005)
    check_track_spread(sdfl,'vR','vT',1.1,1.1,phys=True) #do 1 with phys
    check_track_spread(sdfl,'vR','vZ',0.005,0.005)
    check_track_spread(sdfl,'vX','vY',0.005,0.005)
    check_track_spread(sdfl,'ll','bb',0.5,0.5)
    check_track_spread(sdfl,'dist','vlos',0.5,5.)
    check_track_spread(sdfl,'pmll','pmbb',0.5,0.5)
    return None

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
    try:
        assert(maxdevZ < tol)
    except AssertionError:
        raise AssertionError("Stream track deviates more from progenitor track in %s vs. %s than expected; max. deviation = %f" % (d2,d1,maxdevZ))
    return None

def check_track_spread(sdf,d1,d2,tol1,tol2,phys=False):
    #Check that the spread around the track is small
    addx, addy= sdf._parse_track_spread(d1,d2,interp=True,phys=phys) 
    try:
        assert(numpy.amax(addx) < tol1)
    except AssertionError:
        raise AssertionError("Stream track spread is larger in %s than expected; max. deviation = %f" % (d1,numpy.amax(addx)))
    try:
        assert(numpy.amax(addy) < tol2)
    except AssertionError:
        raise AssertionError("Stream track spread is larger in %s than expected; max. deviation = %f" % (d2,numpy.amax(addy)))
    return None

@expected_failure
def test_diff_pot():
    raise AssertionError()
