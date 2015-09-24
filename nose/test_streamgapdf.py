import numpy
numpy.random.seed(1)
sdf_sanders15= None #so we can set this up and then use in other tests

#Exact setup from Section 5 of Sanders, Bovy, and Erkal (2015); should reproduce those results (which have been checked against a simulation)
def test_sanders15_setup():
    #Imports
    from galpy.df import streamgapdf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.util import bovy_conversion #for unit conversions
    lp= LogarithmicHaloPotential(normalize=1.,q=0.9)
    aAI= actionAngleIsochroneApprox(pot=lp,b=0.8)
    prog_unp_peri= Orbit([2.6556151742081835,
                          0.2183747276300308,
                          0.67876510797240575,
                          -2.0143395648974671,
                          -0.3273737682604374,
                          0.24218273922966019])
    global sdf_sanders15
    V0, R0= 220., 8.
    sigv= 0.365*(10./2.)**(1./3.) # km/s
    sdf_sanders15= streamgapdf(sigv/V0,progenitor=prog_unp_peri,pot=lp,aA=aAI,
                               leading=False,nTrackChunks=26,
                               nTrackIterations=1,
                               sigMeanOffset=4.5,
                               tdisrupt=10.88\
                                   /bovy_conversion.time_in_Gyr(V0,R0),
                               Vnorm=V0,Rnorm=R0,
                               impactb=0.,
                               subhalovel=numpy.array([6.82200571,132.7700529,
                                                       149.4174464])/V0,
                               timpact=0.88/bovy_conversion.time_in_Gyr(V0,R0),
                               impact_angle=-2.34,
                               GM=10.**-2.\
                                   /bovy_conversion.mass_in_1010msol(V0,R0),
                               rs=0.625/R0)
    assert not sdf_sanders15 is None, 'sanders15 streamdf setup did not work'
    return None

# Some very basic tests
def test_nTrackIterations():
    assert sdf_sanders15.nTrackIterations == 1, 'nTrackIterations should have been 1'
    return None
def test_nTrackChunks():
    assert sdf_sanders15._nTrackChunks == 26, 'nTrackChunks should have been 26'
    return None
def test_deltaAngleTrackImpact():
    assert numpy.fabs(sdf_sanders15._deltaAngleTrackImpact-4.31) < 0.01, 'deltaAngleTrackImpact should have been ~4.31'
    return None

# Tests of the track near the impact
def test_trackNearImpact():
    # Sanity checks against numbers taken from plots of the simulation
    # Make sure we're near 14.5
    assert numpy.fabs(sdf_sanders15._gap_ObsTrack[14,0]*sdf_sanders15._Rnorm
                      -14.5) < 0.2, '14th point along track near the impact is not near 14.5 kpc'
    assert numpy.fabs(sdf_sanders15._gap_ObsTrack[14,1]*sdf_sanders15._Vnorm
                      -80) < 3., 'Point along the track near impact near R=14.5 does not have the correct radial velocity'
    assert numpy.fabs(sdf_sanders15._gap_ObsTrack[14,2]*sdf_sanders15._Vnorm
                      -220.) < 3., 'Point along the track near impact near R=14.5 does not have the correct tangential velocity'
    assert numpy.fabs(sdf_sanders15._gap_ObsTrack[14,3]*sdf_sanders15._Rnorm
                      -0.) < 1., 'Point along the track near impact near R=14.5 does not have the correct vertical height'
    assert numpy.fabs(sdf_sanders15._gap_ObsTrack[14,4]*sdf_sanders15._Vnorm
                      -200.) < 5., 'Point along the track near impact near R=14.5 does not have the correct vertical velocity'
    # Another one!
    assert numpy.fabs(sdf_sanders15._gap_ObsTrack[27,0]*sdf_sanders15._Rnorm
                      -16.25) < 0.2, '27th point along track near the impact is not near 16.25 kpc'
    assert numpy.fabs(sdf_sanders15._gap_ObsTrack[27,1]*sdf_sanders15._Vnorm
                      +130) < 3., 'Point along the track near impact near R=16.25 does not have the correct radial velocity'
    assert numpy.fabs(sdf_sanders15._gap_ObsTrack[27,2]*sdf_sanders15._Vnorm
                      -200.) < 3., 'Point along the track near impact near R=16.25 does not have the correct tangential velocity'
    assert numpy.fabs(sdf_sanders15._gap_ObsTrack[27,3]*sdf_sanders15._Rnorm
                      +12.) < 1., 'Point along the track near impact near R=16.25 does not have the correct vertical height'
    assert numpy.fabs(sdf_sanders15._gap_ObsTrack[27,4]*sdf_sanders15._Vnorm
                      -25.) < 5., 'Point along the track near impact near R=16.25 does not have the correct vertical velocity'   
    assert numpy.fabs(sdf_sanders15._gap_ObsTrack[27,5]-1.2) < .2, 'Point along the track near impact near R=16.25 does not have the correct azimuth'   
    return None

def test_interpolatedTrackNearImpact():
    # Sanity checks against numbers taken from plots of the simulation
    # Make sure we're near X=-10.9
    theta= 2.7
    assert numpy.fabs(sdf_sanders15._kick_interpTrackX(theta)*sdf_sanders15._Rnorm
                      +10.9) < 0.2, 'Point along track near the impact at theta=2.7 is not near X=-10.9 kpc'
    assert numpy.fabs(sdf_sanders15._kick_interpTrackY(theta)*sdf_sanders15._Rnorm
                      -6.) < 0.5, 'Point along track near the impact at theta=2.7 is not near Y=6. kpc'
    assert numpy.fabs(sdf_sanders15._kick_interpTrackZ(theta)*sdf_sanders15._Rnorm
                      +5.) < 0.5, 'Point along track near the impact at theta=2.7 is not near Z=5. kpc'
    assert numpy.fabs(sdf_sanders15._kick_interpTrackvX(theta)*sdf_sanders15._Vnorm
                      +180.) < 5, 'Point along track near the impact at theta=2.7 is not near vX=-180 km/s'
    assert numpy.fabs(sdf_sanders15._kick_interpTrackvY(theta)*sdf_sanders15._Vnorm
                      +190.) < 5., 'Point along track near the impact at theta=2.7 is not near vY=190 km/s'
    assert numpy.fabs(sdf_sanders15._kick_interpTrackvZ(theta)*sdf_sanders15._Vnorm
                      -170.) < 5., 'Point along track near the impact at theta=2.7 is not near vZ=170 km/s'
    return None

# Test the calculation of the kicks in dv
def test_kickdv():
    # Closest one to the impact point, should be close to zero
    tIndx= numpy.argmin(numpy.fabs(sdf_sanders15._kick_interpolatedThetasTrack\
                                       -sdf_sanders15._impact_angle))
    assert numpy.all(numpy.fabs(sdf_sanders15._kick_deltav[tIndx]*sdf_sanders15._Vnorm) < 0.3), 'Kick near the impact point not close to zero'
    # The peak, size and location
    assert numpy.fabs(numpy.amax(numpy.fabs(sdf_sanders15._kick_deltav[:,0]*sdf_sanders15._Vnorm))-0.3) < 0.06, 'Peak dvx incorrect'
    assert sdf_sanders15._kick_interpolatedThetasTrack[numpy.argmax(sdf_sanders15._kick_deltav[:,0]*sdf_sanders15._Vnorm)]-sdf_sanders15._impact_angle < 0., 'Location of peak dvx incorrect'
    assert numpy.fabs(numpy.amax(numpy.fabs(sdf_sanders15._kick_deltav[:,1]*sdf_sanders15._Vnorm))-0.3) < 0.06, 'Peak dvy incorrect'
    assert sdf_sanders15._kick_interpolatedThetasTrack[numpy.argmax(sdf_sanders15._kick_deltav[:,1]*sdf_sanders15._Vnorm)]-sdf_sanders15._impact_angle > 0., 'Location of peak dvy incorrect'
    assert numpy.fabs(numpy.amax(numpy.fabs(sdf_sanders15._kick_deltav[:,2]*sdf_sanders15._Vnorm))-1.8) < 0.06, 'Peak dvz incorrect'
    assert sdf_sanders15._kick_interpolatedThetasTrack[numpy.argmax(sdf_sanders15._kick_deltav[:,2]*sdf_sanders15._Vnorm)]-sdf_sanders15._impact_angle > 0., 'Location of peak dvz incorrect'
    # Close to zero far from impact point
    tIndx= numpy.argmin(numpy.fabs(sdf_sanders15._kick_interpolatedThetasTrack\
                                       -sdf_sanders15._impact_angle-1.5))
    assert numpy.all(numpy.fabs(sdf_sanders15._kick_deltav[tIndx]*sdf_sanders15._Vnorm) < 0.3), 'Kick far the impact point not close to zero'
    return None

# Test the calculation of the kicks in dO
def test_kickdO():
    from galpy.util import bovy_conversion
    # Closest one to the impact point, should be close to zero
    tIndx= numpy.argmin(numpy.fabs(sdf_sanders15._kick_interpolatedThetasTrack\
                                       -sdf_sanders15._impact_angle))
    assert numpy.all(numpy.fabs(sdf_sanders15._kick_dOap[tIndx,:3]*bovy_conversion.freq_in_Gyr(sdf_sanders15._Vnorm,sdf_sanders15._Rnorm)) < 0.03), 'Kick near the impact point not close to zero'
    # The peak, size and location
    assert numpy.fabs(numpy.amax(numpy.fabs(sdf_sanders15._kick_dOap[:,0]*bovy_conversion.freq_in_Gyr(sdf_sanders15._Vnorm,sdf_sanders15._Rnorm)))-0.085) < 0.01, 'Peak dOR incorrect'
    assert sdf_sanders15._kick_interpolatedThetasTrack[numpy.argmax(sdf_sanders15._kick_dOap[:,0])]-sdf_sanders15._impact_angle < 0., 'Location of peak dOR incorrect'
    assert numpy.fabs(numpy.amax(numpy.fabs(sdf_sanders15._kick_dOap[:,1]*bovy_conversion.freq_in_Gyr(sdf_sanders15._Vnorm,sdf_sanders15._Rnorm)))-0.07) < 0.01, 'Peak dOp incorrect'
    assert sdf_sanders15._kick_interpolatedThetasTrack[numpy.argmax(sdf_sanders15._kick_dOap[:,1])]-sdf_sanders15._impact_angle < 0., 'Location of peak dvy incorrect'
    assert numpy.fabs(numpy.amax(numpy.fabs(sdf_sanders15._kick_dOap[:,2]*bovy_conversion.freq_in_Gyr(sdf_sanders15._Vnorm,sdf_sanders15._Rnorm)))-0.075) < 0.01, 'Peak dOz incorrect'
    assert sdf_sanders15._kick_interpolatedThetasTrack[numpy.argmax(sdf_sanders15._kick_dOap[:,2])]-sdf_sanders15._impact_angle < 0., 'Location of peak dOz incorrect'
    # Close to zero far from impact point
    tIndx= numpy.argmin(numpy.fabs(sdf_sanders15._kick_interpolatedThetasTrack\
                                       -sdf_sanders15._impact_angle-1.5))
    assert numpy.all(numpy.fabs(sdf_sanders15._kick_dOap[tIndx,:3]*bovy_conversion.freq_in_Gyr(sdf_sanders15._Vnorm,sdf_sanders15._Rnorm)) < 0.03), 'Kick far the impact point not close to zero'
    return None

def test_kickda():
    # All angle kicks should be small, just test that they are smaller than dO/O
    assert numpy.all(numpy.fabs(sdf_sanders15._kick_dOap[:,3:]) < (numpy.fabs(sdf_sanders15._kick_dOap[:,3:]/sdf_sanders15._progenitor_Omega))), 'angle kicks not smaller than the frequency kicks'
    return None

# Test the interpolation of the kicks
def test_interpKickdO():
    from galpy.util import bovy_conversion
    freqConv= bovy_conversion.freq_in_Gyr(sdf_sanders15._Vnorm,sdf_sanders15._Rnorm)
    # Bunch of spot checks at some interesting angles
    # Impact angle
    theta= sdf_sanders15._impact_angle
    assert numpy.fabs(sdf_sanders15._kick_interpdOpar(theta)*freqConv) < 10.**-4., 'Frequency kick at the impact point is not zero'
    assert numpy.fabs(sdf_sanders15._kick_interpdOperp0(theta)*freqConv) < 10.**-4., 'Frequency kick at the impact point is not zero'
    assert numpy.fabs(sdf_sanders15._kick_interpdOperp1(theta)*freqConv) < 10.**-4., 'Frequency kick at the impact point is not zero'
    assert numpy.fabs(sdf_sanders15._kick_interpdOr(theta)*freqConv) < 10.**-4., 'Frequency kick at the impact point is not zero'
    assert numpy.fabs(sdf_sanders15._kick_interpdOp(theta)*freqConv) < 10.**-4., 'Frequency kick at the impact point is not zero'
    assert numpy.fabs(sdf_sanders15._kick_interpdOz(theta)*freqConv) < 10.**-4., 'Frequency kick at the impact point is not zero'
    # random one
    theta= sdf_sanders15._impact_angle-0.25
    assert numpy.fabs(sdf_sanders15._kick_interpdOpar(theta)*freqConv+0.07) < 0.002, 'Frequency kick near the impact point is not zero'
    assert numpy.fabs(sdf_sanders15._kick_interpdOperp0(theta)*freqConv) < 0.002, 'Frequency kick near the impact point is not zero'
    assert numpy.fabs(sdf_sanders15._kick_interpdOperp1(theta)*freqConv) < 0.003, 'Frequency kick near the impact point is not zero'
    assert numpy.fabs(sdf_sanders15._kick_interpdOr(theta)*freqConv-0.05) < 0.01, 'Frequency kick near the impact point is not zero'
    assert numpy.fabs(sdf_sanders15._kick_interpdOp(theta)*freqConv-0.035) < 0.01, 'Frequency kick near the impact point is not zero'
    assert numpy.fabs(sdf_sanders15._kick_interpdOz(theta)*freqConv-0.04) < 0.01, 'Frequency kick near the impact point is not zero'
    return None

# Test rewind_angle

# Test the sampling of present-day perturbed points based on the model

# Test the routine that rotates vectors to an arbitrary vector
def test_rotate_to_arbitrary_vector():
    from galpy.df_src import streamgapdf
    tol= -10.
    v= numpy.array([[1.,0.,0.]])
    # Rotate to 90 deg off
    ma= streamgapdf._rotate_to_arbitrary_vector(v,[0,1.,0])
    assert numpy.fabs(ma[0,0,1]+1.) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    assert numpy.fabs(ma[0,1,0]-1.) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    assert numpy.fabs(ma[0,2,2]-1.) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    assert numpy.fabs(ma[0,0,0]) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    assert numpy.fabs(ma[0,0,2]) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    assert numpy.fabs(ma[0,1,1]) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    assert numpy.fabs(ma[0,1,2]) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    assert numpy.fabs(ma[0,2,0]) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    assert numpy.fabs(ma[0,2,1]) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    # Rotate to 90 deg off
    ma= streamgapdf._rotate_to_arbitrary_vector(v,[0,0,1.])
    assert numpy.fabs(ma[0,0,2]+1.) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    assert numpy.fabs(ma[0,2,0]-1.) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    assert numpy.fabs(ma[0,1,1]-1.) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    assert numpy.fabs(ma[0,0,0]) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    assert numpy.fabs(ma[0,0,1]) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    assert numpy.fabs(ma[0,2,2]) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    assert numpy.fabs(ma[0,2,1]) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    assert numpy.fabs(ma[0,1,0]) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    assert numpy.fabs(ma[0,1,2]) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    # Rotate to same should be unit matrix
    ma= streamgapdf._rotate_to_arbitrary_vector(v,v[0])
    assert numpy.all(numpy.fabs(numpy.diag(ma[0])-1.) < 10.**tol), \
        'Rotation matrix to same vector is not unity'
    assert numpy.fabs(numpy.sum(ma**2.)-3.)< 10.**tol, \
        'Rotation matrix to same vector is not unity'
    # Rotate to -same should be -unit matrix
    ma= streamgapdf._rotate_to_arbitrary_vector(v,-v[0])
    assert numpy.all(numpy.fabs(numpy.diag(ma[0])+1.) < 10.**tol), \
        'Rotation matrix to minus same vector is not minus unity'
    assert numpy.fabs(numpy.sum(ma**2.)-3.)< 10.**tol, \
        'Rotation matrix to minus same vector is not minus unity'
    return None

# Test that the rotation routine works for multiple vectors
def test_rotate_to_arbitrary_vector_multi():
    from galpy.df_src import streamgapdf
    tol= -10.
    v= numpy.array([[1.,0.,0.],[0.,1.,0.]])
    # Rotate to 90 deg off
    ma= streamgapdf._rotate_to_arbitrary_vector(v,[0,0,1.])
    assert numpy.fabs(ma[0,0,2]+1.) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    assert numpy.fabs(ma[0,2,0]-1.) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    assert numpy.fabs(ma[0,1,1]-1.) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    assert numpy.fabs(ma[0,0,0]) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    assert numpy.fabs(ma[0,0,1]) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    assert numpy.fabs(ma[0,2,2]) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    assert numpy.fabs(ma[0,2,1]) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    assert numpy.fabs(ma[0,1,0]) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    assert numpy.fabs(ma[0,1,2]) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    # 2nd
    assert numpy.fabs(ma[1,1,2]+1.) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    assert numpy.fabs(ma[1,2,1]-1.) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    assert numpy.fabs(ma[1,0,0]-1.) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    assert numpy.fabs(ma[1,0,1]) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    assert numpy.fabs(ma[1,0,2]) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    assert numpy.fabs(ma[1,1,0]) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    assert numpy.fabs(ma[1,1,1]) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    assert numpy.fabs(ma[1,2,0]) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    assert numpy.fabs(ma[1,2,2]) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    return None

# Test the inverse of the routine that rotates vectors to an arbitrary vector
def test_rotate_to_arbitrary_vector_inverse():
    from galpy.df_src import streamgapdf
    tol= -10.
    v= numpy.array([[1.,0.,0.]])
    # Rotate to random vector and back
    a= numpy.random.uniform(size=3)
    a/= numpy.sqrt(numpy.sum(a**2.))
    ma= streamgapdf._rotate_to_arbitrary_vector(v,a)
    ma_inv= streamgapdf._rotate_to_arbitrary_vector(v,a,inv=True)
    ma= numpy.dot(ma[0],ma_inv[0])
    assert numpy.all(numpy.fabs(ma-numpy.eye(3)) < 10.**tol), 'Inverse rotation matrix incorrect'
    return None

# Test that rotating to vy in particular works as expected
def test_rotation_vy():
    from galpy.df_src import streamgapdf
    tol= -10.
    v= numpy.array([[1.,0.,0.]])
    # Rotate to 90 deg off
    ma= streamgapdf._rotation_vy(v)
    assert numpy.fabs(ma[0,0,1]+1.) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    assert numpy.fabs(ma[0,1,0]-1.) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    assert numpy.fabs(ma[0,2,2]-1.) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    assert numpy.fabs(ma[0,0,0]) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    assert numpy.fabs(ma[0,0,2]) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    assert numpy.fabs(ma[0,1,1]) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    assert numpy.fabs(ma[0,1,2]) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    assert numpy.fabs(ma[0,2,0]) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'
    assert numpy.fabs(ma[0,2,1]) < 10.**tol, 'Rotation matrix to 90 deg off incorrect'

# Test the Plummer calculation for a perpendicular impact, B&T ex. 8.7
def test_impulse_deltav_plummer_subhalo_perpendicular():
    from galpy.df_src import streamgapdf
    tol= -10.
    kick= streamgapdf.impulse_deltav_plummer(numpy.array([[0.,numpy.pi,0.]]),
                                             numpy.array([0.]),
                                             3.,
                                             numpy.array([0.,numpy.pi/2.,0.]),
                                             1.5,4.)
    # Should be B&T (8.152)
    assert numpy.fabs(kick[0,0]-2.*1.5*3./numpy.pi*2./25.) < 10.**tol, 'Perpendicular kick of subhalo perpendicular not as expected'
    assert numpy.fabs(kick[0,2]+2.*1.5*3./numpy.pi*2./25.) < 10.**tol, 'Perpendicular kick of subhalo perpendicular not as expected'
    # Same for along z
    kick= streamgapdf.impulse_deltav_plummer(numpy.array([[0.,0.,numpy.pi]]),
                                             numpy.array([0.]),
                                             3.,
                                             numpy.array([0.,0.,numpy.pi/2.]),
                                             1.5,4.)
    # Should be B&T (8.152)
    assert numpy.fabs(kick[0,0]-2.*1.5*3./numpy.pi*2./25.) < 10.**tol, 'Perpendicular kick of subhalo perpendicular not as expected'
    assert numpy.fabs(kick[0,1]-2.*1.5*3./numpy.pi*2./25.) < 10.**tol, 'Perpendicular kick of subhalo perpendicular not as expected'
    return None

# Test the Plummer curved calculation for a perpendicular impact
def test_impulse_deltav_plummer_curved_subhalo_perpendicular():
    from galpy.df_src import streamgapdf
    tol= -10.
    kick= streamgapdf.impulse_deltav_plummer(numpy.array([[3.4,0.,0.]]),
                                             numpy.array([4.]),
                                             3.,
                                             numpy.array([0.,numpy.pi/2.,0.]),
                                             1.5,4.)
    curved_kick= streamgapdf.impulse_deltav_plummer_curvedstream(\
        numpy.array([[3.4,0.,0.]]),
        numpy.array([[4.,0.,0.]]),
        3.,
        numpy.array([0.,numpy.pi/2.,0.]),
        numpy.array([0.,0.,0.]),
        numpy.array([3.4,0.,0.]),
        1.5,4.)
    # Should be equal
    assert numpy.all(numpy.fabs(kick-curved_kick) < 10.**tol), 'curved Plummer kick does not agree with straight kick for straight track'
    # Same for a bunch of positions
    v= numpy.zeros((100,3))
    v[:,0]= 3.4
    xpos= numpy.random.normal(size=100)
    kick= streamgapdf.impulse_deltav_plummer(v,
                                             xpos,
                                             3.,
                                             numpy.array([0.,numpy.pi/2.,0.]),
                                             1.5,4.)
    xpos= numpy.array([xpos,numpy.zeros(100),numpy.zeros(100)]).T
    curved_kick= streamgapdf.impulse_deltav_plummer_curvedstream(\
        v,
        xpos,
        3.,
        numpy.array([0.,numpy.pi/2.,0.]),
        numpy.array([0.,0.,0.]),
        numpy.array([3.4,0.,0.]),
        1.5,4.)
    # Should be equal
    assert numpy.all(numpy.fabs(kick-curved_kick) < 10.**tol), 'curved Plummer kick does not agree with straight kick for straight track'
    return None

# Test general impulse vs. Plummer
def test_impulse_deltav_general():
    from galpy.df_src import streamgapdf
    from galpy.potential import PlummerPotential
    tol= -10.
    kick= streamgapdf.impulse_deltav_plummer(numpy.array([[3.4,0.,0.]]),
                                             numpy.array([4.]),
                                             3.,
                                             numpy.array([0.,numpy.pi/2.,0.]),
                                             1.5,4.)
    pp= PlummerPotential(amp=1.5,b=4.)
    general_kick=\
        streamgapdf.impulse_deltav_general(numpy.array([[3.4,0.,0.]]),
                                           numpy.array([4.]),
                                           3.,
                                           numpy.array([0.,numpy.pi/2.,0.]),
                                           pp)
    assert numpy.all(numpy.fabs(kick-general_kick) < 10.**tol), 'general kick calculation does not agree with Plummer calculation for a Plummer potential'
    # Same for a bunch of positions
    v= numpy.zeros((100,3))
    v[:,0]= 3.4
    xpos= numpy.random.normal(size=100)
    kick= streamgapdf.impulse_deltav_plummer(v,
                                             xpos,
                                             3.,
                                             numpy.array([0.,numpy.pi/2.,0.]),
                                             numpy.pi,numpy.exp(1.))
    pp= PlummerPotential(amp=numpy.pi,b=numpy.exp(1.))
    general_kick=\
        streamgapdf.impulse_deltav_general(v,
                                           xpos,
                                           3.,
                                           numpy.array([0.,numpy.pi/2.,0.]),
                                           pp)
    assert numpy.all(numpy.fabs(kick-general_kick) < 10.**tol), 'general kick calculation does not agree with Plummer calculation for a Plummer potential'
    return None

# Test general impulse vs. Plummer for curved stream
def test_impulse_deltav_general_curved():
    from galpy.df_src import streamgapdf
    from galpy.potential import PlummerPotential
    tol= -10.
    kick= streamgapdf.impulse_deltav_plummer_curvedstream(\
        numpy.array([[3.4,0.,0.]]),
        numpy.array([[4.,0.,0.]]),
        3.,
        numpy.array([0.,numpy.pi/2.,0.]),
        numpy.array([0.,0.,0.]),
        numpy.array([3.4,0.,0.]),
        1.5,4.)
    pp= PlummerPotential(amp=1.5,b=4.)
    general_kick= streamgapdf.impulse_deltav_general_curvedstream(\
        numpy.array([[3.4,0.,0.]]),
        numpy.array([[4.,0.,0.]]),
        3.,
        numpy.array([0.,numpy.pi/2.,0.]),
        numpy.array([0.,0.,0.]),
        numpy.array([3.4,0.,0.]),
        pp)
    assert numpy.all(numpy.fabs(kick-general_kick) < 10.**tol), 'general kick calculation does not agree with Plummer calculation for a Plummer potential, for curved stream'
    # Same for a bunch of positions
    v= numpy.zeros((100,3))
    v[:,0]= 3.4
    xpos= numpy.random.normal(size=100)
    xpos= numpy.array([xpos,numpy.zeros(100),numpy.zeros(100)]).T
    kick= streamgapdf.impulse_deltav_plummer_curvedstream(\
        v,
        xpos,
        3.,
        numpy.array([0.,numpy.pi/2.,0.]),
        numpy.array([0.,0.,0.]),
        numpy.array([3.4,0.,0.]),
        numpy.pi,numpy.exp(1.))
    pp= PlummerPotential(amp=numpy.pi,b=numpy.exp(1.))
    general_kick=\
        streamgapdf.impulse_deltav_general_curvedstream(\
        v,
        xpos,
        3.,
        numpy.array([0.,numpy.pi/2.,0.]),
        numpy.array([0.,0.,0.]),
        numpy.array([3.4,0.,0.]),
        pp)
    assert numpy.all(numpy.fabs(kick-general_kick) < 10.**tol), 'general kick calculation does not agree with Plummer calculation for a Plummer potential, for curved stream'
    return None

# Test general impulse vs. Hernquist
def test_impulse_deltav_general_hernquist():
    from galpy.df_src import streamgapdf
    from galpy.potential import HernquistPotential
    GM = 1.5
    tol= -10.
    kick= streamgapdf.impulse_deltav_hernquist(numpy.array([[3.4,0.,0.]]),
                                             numpy.array([4.]),
                                             3.,
                                             numpy.array([0.,numpy.pi/2.,0.]),
                                             GM,4.)
    # Note factor of 2 in definition of GM and amp
    pp= HernquistPotential(amp=2.*GM,a=4.)
    general_kick=\
        streamgapdf.impulse_deltav_general(numpy.array([[3.4,0.,0.]]),
                                           numpy.array([4.]),
                                           3.,
                                           numpy.array([0.,numpy.pi/2.,0.]),
                                           pp)
    assert numpy.all(numpy.fabs(kick-general_kick) < 10.**tol), 'general kick calculation does not agree with Hernquist calculation for a Hernquist potential'
    # Same for a bunch of positions
    GM = numpy.pi
    v= numpy.zeros((100,3))
    v[:,0]= 3.4
    xpos= numpy.random.normal(size=100)
    kick= streamgapdf.impulse_deltav_hernquist(v,
                                             xpos,
                                             3.,
                                             numpy.array([0.,numpy.pi/2.,0.]),
                                             GM,numpy.exp(1.))
    pp= HernquistPotential(amp=2.*GM,a=numpy.exp(1.))
    general_kick=\
        streamgapdf.impulse_deltav_general(v,
                                           xpos,
                                           3.,
                                           numpy.array([0.,numpy.pi/2.,0.]),
                                           pp)
    assert numpy.all(numpy.fabs(kick-general_kick) < 10.**tol), 'general kick calculation does not agree with Hernquist calculation for a Hernquist potential'
    return None

# Test general impulse vs. Hernquist for curved stream
def test_impulse_deltav_general_curved_hernquist():
    from galpy.df_src import streamgapdf
    from galpy.potential import HernquistPotential
    GM = 1.5
    tol= -10.
    kick= streamgapdf.impulse_deltav_hernquist_curvedstream(\
        numpy.array([[3.4,0.,0.]]),
        numpy.array([[4.,0.,0.]]),
        3.,
        numpy.array([0.,numpy.pi/2.,0.]),
        numpy.array([0.,0.,0.]),
        numpy.array([3.4,0.,0.]),
        GM,4.)
    # Note factor of 2 in definition of GM and amp
    pp= HernquistPotential(amp=2.*GM,a=4.)
    general_kick= streamgapdf.impulse_deltav_general_curvedstream(\
        numpy.array([[3.4,0.,0.]]),
        numpy.array([[4.,0.,0.]]),
        3.,
        numpy.array([0.,numpy.pi/2.,0.]),
        numpy.array([0.,0.,0.]),
        numpy.array([3.4,0.,0.]),
        pp)
    assert numpy.all(numpy.fabs(kick-general_kick) < 10.**tol), 'general kick calculation does not agree with Hernquist calculation for a Hernquist potential, for curved stream'
    # Same for a bunch of positions
    GM = numpy.pi
    v= numpy.zeros((100,3))
    v[:,0]= 3.4
    xpos= numpy.random.normal(size=100)
    xpos= numpy.array([xpos,numpy.zeros(100),numpy.zeros(100)]).T
    kick= streamgapdf.impulse_deltav_hernquist_curvedstream(\
        v,
        xpos,
        3.,
        numpy.array([0.,numpy.pi/2.,0.]),
        numpy.array([0.,0.,0.]),
        numpy.array([3.4,0.,0.]),
        GM,numpy.exp(1.))
    pp= HernquistPotential(amp=2.*GM,a=numpy.exp(1.))
    general_kick=\
        streamgapdf.impulse_deltav_general_curvedstream(\
        v,
        xpos,
        3.,
        numpy.array([0.,numpy.pi/2.,0.]),
        numpy.array([0.,0.,0.]),
        numpy.array([3.4,0.,0.]),
        pp)
    assert numpy.all(numpy.fabs(kick-general_kick) < 10.**tol), 'general kick calculation does not agree with Hernquist calculation for a Hernquist potential, for curved stream'
    return None

from nose.tools import raises
@raises(ValueError)
def test_hernquistX_negative():
    from galpy.df_src import streamgapdf
    streamgapdf.HernquistX(-1.)
    return None

def test_hernquistX_unity():
    from galpy.df_src import streamgapdf
    assert streamgapdf.HernquistX(1.)==1., 'Hernquist X function not returning 1 with argument 1'
    return None

# Test general impulse vs. full orbit integration for zero force
def test_impulse_deltav_general_orbit_zeroforce():
    from galpy.df_src import streamgapdf
    from galpy.potential import PlummerPotential
    tol= -6.
    rcurv=10.
    vp=220.
    x0 = numpy.array([rcurv,0.,0.])
    v0 = numpy.array([0.,vp,0.])
    w = numpy.array([1.,numpy.pi/2.,0.])
    plummer_kick= streamgapdf.impulse_deltav_plummer_curvedstream(\
        v0,x0,3.,w,x0,v0,1.5,4.)
    pp= PlummerPotential(amp=1.5,b=4.)
    vang=vp/rcurv
    angrange=numpy.pi
    maxt=5.*angrange/vang
    galpot = constantPotential()
    orbit_kick= streamgapdf.impulse_deltav_general_orbitintegration(\
        v0,x0,3.,w,x0,v0,pp,maxt,galpot)
    assert numpy.all(numpy.fabs(orbit_kick-plummer_kick) < 10.**tol), \
        'general kick with acceleration calculation does not agree with Plummer calculation for a Plummer potential, for straight'
    # Same for a bunch of positions
    tol= -5.
    pp= PlummerPotential(amp=numpy.pi,b=numpy.exp(1.))
    theta = numpy.linspace(-numpy.pi/4.,numpy.pi/4.,100)
    xc,yc = rcurv*numpy.cos(theta),rcurv*numpy.sin(theta)
    Xc = numpy.zeros((100,3))
    Xc[:,0]=xc
    Xc[:,1]=yc
    vx,vy = -vp*numpy.sin(theta),vp*numpy.cos(theta)
    V = numpy.zeros((100,3))
    V[:,0]=vx
    V[:,1]=vy
    plummer_kick= streamgapdf.impulse_deltav_plummer_curvedstream(\
        V,Xc,3.,w,x0,v0,numpy.pi,numpy.exp(1.))
    orbit_kick=streamgapdf.impulse_deltav_general_orbitintegration(\
        V,Xc,3.,w,x0,v0,pp,
        maxt,
        galpot)
    assert numpy.all(numpy.fabs(orbit_kick-plummer_kick) < 10.**tol), \
            'general kick calculation does not agree with Plummer calculation for a Plummer potential, for curved stream'
    return None

# Test general impulse vs. full stream and halo integration for zero force
def test_impulse_deltav_general_fullintegration_zeroforce():
    from galpy.df_src import streamgapdf
    tol= -3.
    rcurv=10.
    vp=220.
    GM=1.5
    rs=4.
    x0 = numpy.array([rcurv,0.,0.])
    v0 = numpy.array([0.,vp,0.])
    w = numpy.array([1.,numpy.pi/4.*vp,0.])
    plummer_kick= streamgapdf.impulse_deltav_plummer_curvedstream(\
        v0,x0,3.,w,x0,v0,GM,rs)
    galpot = constantPotential()
    orbit_kick= streamgapdf.impulse_deltav_general_fullplummerintegration(\
        v0,x0,3.,w,x0,v0,galpot,GM,rs,tmaxfac=100.,N=1000)
    nzeroIndx= numpy.fabs(plummer_kick) > 10.**tol
    assert numpy.all(numpy.fabs((orbit_kick-plummer_kick)/plummer_kick)[nzeroIndx] < 10.**tol), \
        'general kick with acceleration calculation does not agree with Plummer calculation for a Plummer potential, for straight'
    assert numpy.all(numpy.fabs(orbit_kick-plummer_kick)[True-nzeroIndx] < 10.**tol), \
        'general kick with acceleration calculation does not agree with Plummer calculation for a Plummer potential, for straight'
    # Same for a bunch of positions
    tol= -2.5
    GM=numpy.pi
    rs=numpy.exp(1.)
    theta = numpy.linspace(-numpy.pi/4.,numpy.pi/4.,10)
    xc,yc = rcurv*numpy.cos(theta),rcurv*numpy.sin(theta)
    Xc = numpy.zeros((10,3))
    Xc[:,0]=xc
    Xc[:,1]=yc
    vx,vy = -vp*numpy.sin(theta),vp*numpy.cos(theta)
    V = numpy.zeros((10,3))
    V[:,0]=vx
    V[:,1]=vy
    plummer_kick= streamgapdf.impulse_deltav_plummer_curvedstream(\
        V,Xc,3.,w,x0,v0,GM,rs)
    orbit_kick=streamgapdf.impulse_deltav_general_fullplummerintegration(\
        V,Xc,3.,w,x0,v0,galpot,GM,rs,tmaxfac=100.)
    nzeroIndx= numpy.fabs(plummer_kick) > 10.**tol
    assert numpy.all(numpy.fabs((orbit_kick-plummer_kick)/plummer_kick)[nzeroIndx] < 10.**tol), \
        'full stream+halo integration calculation does not agree with Plummer calculation for a Plummer potential, for curved stream'
    assert numpy.all(numpy.fabs(orbit_kick-plummer_kick)[True-nzeroIndx] < 10.**tol), \
        'full stream+halo integration calculation does not agree with Plummer calculation for a Plummer potential, for curved stream'
    return None

# Test general impulse vs. full stream and halo integration for fast encounter
def test_impulse_deltav_general_fullintegration_fastencounter():
    from galpy.df_src import streamgapdf
    from galpy.potential import PlummerPotential, LogarithmicHaloPotential
    tol= -2.
    GM=1.5
    rs=4.
    x0 = numpy.array([1.5,0.,0.])
    v0 = numpy.array([0.,1.,0.]) #circular orbit
    w = numpy.array([0.,0.,100.]) # very fast compared to v=1
    lp= LogarithmicHaloPotential(normalize=1.)
    pp= PlummerPotential(amp=GM,b=rs)
    orbit_kick= streamgapdf.impulse_deltav_general_orbitintegration(\
        v0,x0,3.,w,x0,v0,pp,5.*numpy.pi,lp)
    full_kick= streamgapdf.impulse_deltav_general_fullplummerintegration(\
        v0,x0,3.,w,x0,v0,lp,GM,rs,tmaxfac=10.,N=1000)
    # Kick should be in the X direction
    assert numpy.fabs((orbit_kick-full_kick)/full_kick)[0,0] < 10.**tol, \
        'Acceleration kick does not agree with full-orbit-integration kick for fast encounter'
    assert numpy.all(numpy.fabs((orbit_kick-full_kick))[0,1:] < 10.**tol), \
        'Acceleration kick does not agree with full-orbit-integration kick for fast encounter'
    return None

from galpy.potential import Potential
class constantPotential(Potential):
    def __init__(self):
        Potential.__init__(self,amp=1.)
        self.hasC= False
        return None
    def _Rforce(self,R,z,phi=0.,t=0.):
        return 0.
    def _zforce(self,R,z,phi=0.,t=0.):
        return 0.
