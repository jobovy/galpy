import numpy
import pytest
from galpy.df import streamdf, streamspraydf
from galpy.orbit import Orbit
from galpy.potential import LogarithmicHaloPotential
from galpy.actionAngle import actionAngleIsochroneApprox
from galpy.util import conversion #for unit conversions

################################ Tests against streamdf ######################

# Setup both DFs    
@pytest.fixture(scope='module')
def setup_testStreamsprayAgainstStreamdf():
    lp= LogarithmicHaloPotential(normalize=1.,q=0.9)
    aAI= actionAngleIsochroneApprox(pot=lp,b=0.8)
    obs= Orbit([1.56148083,0.35081535,-1.15481504,
                0.88719443,-0.47713334,0.12019596])
    ro, vo= 8., 220.
    # Set up streamdf
    sigv= 0.365 #km/s
    sdf_bovy14= streamdf(sigv/220.,progenitor=obs,pot=lp,aA=aAI,
                         leading=True,
                         nTrackChunks=11,
                         tdisrupt=4.5/conversion.time_in_Gyr(vo,ro))
    # Set up streamspraydf
    spdf_bovy14= streamspraydf(2*10.**4./conversion.mass_in_msol(vo,ro),
                               progenitor=obs,
                               pot=lp,
                               tdisrupt=4.5/conversion.time_in_Gyr(vo,ro))
    return sdf_bovy14, spdf_bovy14
    
def test_sample_bovy14(setup_testStreamsprayAgainstStreamdf):
    # Load objects that were setup above
    sdf_bovy14, spdf_bovy14= setup_testStreamsprayAgainstStreamdf
    numpy.random.seed(1)
    RvR_sdf = sdf_bovy14.sample(n=1000)
    RvR_spdf= spdf_bovy14.sample(n=1000,integrate=True)
    #Sanity checks
    # Range in Z
    indx= (RvR_sdf[3] > 4./8.)*(RvR_sdf[3] < 5./8.)
    #mean
    assert numpy.fabs(numpy.mean(RvR_sdf[0][indx])-numpy.mean(RvR_spdf[0][indx])) < 3e-2, 'streamdf and streamspraydf do not generate similar samples for the Bovy (2014) stream (mean)'
    assert numpy.fabs(numpy.mean(RvR_sdf[1][indx])-numpy.mean(RvR_spdf[1][indx])) < 3e-2, 'streamdf and streamspraydf do not generate similar samples for the Bovy (2014) stream (mean)'
    assert numpy.fabs(numpy.mean(RvR_sdf[2][indx])-numpy.mean(RvR_spdf[2][indx])) < 3e-2, 'streamdf and streamspraydf do not generate similar samples for the Bovy (2014) stream (mean)'
    assert numpy.fabs(numpy.mean(RvR_sdf[4][indx])-numpy.mean(RvR_spdf[4][indx])) < 3e-2, 'streamdf and streamspraydf do not generate similar samples for the Bovy (2014) stream (mean)'
    assert numpy.fabs(numpy.mean(RvR_sdf[5][indx])-numpy.mean(RvR_spdf[5][indx])) < 4e-2, 'streamdf and streamspraydf do not generate similar samples for the Bovy (2014) stream (mean)'
    # Another range in Z
    indx= (RvR_sdf[3] > 5./8.)*(RvR_sdf[3] < 6./8.)
    #mean
    assert numpy.fabs(numpy.mean(RvR_sdf[0][indx])-numpy.mean(RvR_spdf[0][indx])) < 1e-1, 'streamdf and streamspraydf do not generate similar samples for the Bovy (2014) stream (mean)'
    assert numpy.fabs(numpy.mean(RvR_sdf[1][indx])-numpy.mean(RvR_spdf[1][indx])) < 3e-2, 'streamdf and streamspraydf do not generate similar samples for the Bovy (2014) stream (mean)'
    assert numpy.fabs(numpy.mean(RvR_sdf[2][indx])-numpy.mean(RvR_spdf[2][indx])) < 4e-2, 'streamdf and streamspraydf do not generate similar samples for the Bovy (2014) stream (mean)'
    assert numpy.fabs(numpy.mean(RvR_sdf[4][indx])-numpy.mean(RvR_spdf[4][indx])) < 3e-2, 'streamdf and streamspraydf do not generate similar samples for the Bovy (2014) stream (mean)'
    assert numpy.fabs(numpy.mean(RvR_sdf[5][indx])-numpy.mean(RvR_spdf[5][indx])) < 1e-1, 'streamdf and streamspraydf do not generate similar samples for the Bovy (2014) stream (mean)'
    return None

@pytest.mark.xfail
def test_bovy14_sampleXY(setup_testStreamsprayAgainstStreamdf):
    # Load objects that were setup above
    sdf_bovy14, spdf_bovy14= setup_testStreamsprayAgainstStreamdf
    numpy.random.seed(1)
    XvX_sdf = sdf_bovy14.sample(n=1000,xy=True)
    XvX_spdf= spdf_bovy14.sample(n=1000,xy=True)
    #Sanity checks
    # Range in Z
    indx= (XvX_sdf[2] > 4./8.)*(XvX_sdf[2] < 5./8.)
    #mean
    assert numpy.fabs(numpy.mean(XvX_sdf[0][indx])-numpy.mean(XvX_spdf[0][indx])) < 3e-2, 'streamdf and streamspraydf do not generate similar samples for the Bovy (2014) stream (mean, xy)'
    assert numpy.fabs(numpy.mean(XvX_sdf[1][indx])-numpy.mean(XvX_spdf[1][indx])) < 3e-2, 'streamdf and streamspraydf do not generate similar samples for the Bovy (2014) stream (mean, xy)'
    assert numpy.fabs(numpy.mean(XvX_sdf[4][indx])-numpy.mean(XvX_spdf[4][indx])) < 3e-2, 'streamdf and streamspraydf do not generate similar samples for the Bovy (2014) stream (mean, xy)'
    return None

@pytest.mark.xfail
def test_bovy14_sampleLB(setup_testStreamsprayAgainstStreamdf):
    # Load objects that were setup above
    sdf_bovy14, spdf_bovy14= setup_testStreamsprayAgainstStreamdf
    numpy.random.seed(1)
    LB_sdf = sdf_bovy14.sample(n=1000,lb=True)
    LB_spdf = spdf_bovy14.sample(n=1000,lb=True)
    #Sanity checks
    # Range in l
    indx= (LB_sdf[0] > 212.5)*(LB_sdf[0] < 217.5)
    #mean
    assert numpy.fabs(numpy.mean(LB_spdf[0][indx])/numpy.mean(LB_sdf[0][indx])-1.) < 3e-2, 'streamdf and streamspraydf do not generate similar samples for the Bovy (2014) stream (mean, lb)'
    assert numpy.fabs(numpy.mean(LB_spdf[1][indx])/numpy.mean(LB_sdf[1][indx])-1.) < 3e-2, 'streamdf and streamspraydf do not generate similar samples for the Bovy (2014) stream (mean, lb)'
    assert numpy.fabs(numpy.mean(LB_spdf[2][indx])/numpy.mean(LB_sdf[2][indx])-1.) < 3e-2, 'streamdf and streamspraydf do not generate similar samples for the Bovy (2014) stream (mean, lb)'
    return None

def test_integrate(setup_testStreamsprayAgainstStreamdf):
    # Test that sampling at stripping + integrate == sampling at the end
    # Load objects that were setup above
    _, spdf_bovy14= setup_testStreamsprayAgainstStreamdf
    # Sample at at stripping
    numpy.random.seed(4)
    RvR_noint,dt_noint= spdf_bovy14.sample(n=100,returndt=True,integrate=False)
    # and integrate
    for ii in range(len(dt_noint)):
        to= Orbit(RvR_noint[:,ii])
        to.integrate(numpy.linspace(-dt_noint[ii],0.,1001),spdf_bovy14._pot)
        RvR_noint[:,ii]= [to.R(0.),to.vR(0.),to.vT(0.),
                          to.z(0.),to.vz(0.),to.phi(0.)]
    # Sample today
    numpy.random.seed(4)
    RvR,dt= spdf_bovy14.sample(n=100,returndt=True,integrate=True)
    # Should agree
    assert numpy.amax(numpy.fabs(dt-dt_noint)) < 1e-10, 'Times not the same when sampling with and without integrating'
    assert numpy.amax(numpy.fabs(RvR-RvR_noint)) < 1e-7, 'Phase-space points not the same when sampling with and without integrating'
    return None
