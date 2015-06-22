# The DF of a gap in a tidal stream
import numpy
import warnings
import multiprocessing
from scipy import integrate, interpolate
from galpy.util import galpyWarning
from galpy.orbit import Orbit
from galpy.potential import evaluateRforces
import galpy.df_src.streamdf
from galpy.df_src.streamdf import _determine_stream_track_single
from galpy.util import bovy_coords, multi
class streamgapdf(galpy.df_src.streamdf.streamdf):
    """The DF of a tidal stream"""
    def __init__(self,*args,**kwargs):
        """
        NAME:

           __init__

        PURPOSE:

           Initialize the DF of a gap in a stellar stream

        INPUT:

           streamdf args and kwargs

           Subhalo and impact parameters:

              impactb= impact parameter

              subhalovel= velocity of the subhalo shape=(3)

              timpact time since impact 
              
              impact_angle= angle offset from progenitor at which the impact occurred (rad)

              Subhalo: specify either 1( mass and size of Plummer sphere or 2( general spherical-potential object (kick is numerically computed)

                 1( GM= mass of the subhalo

                    rs= size parameter of the subhalo

                 2( subhalopot= galpy potential object or list thereof (should be spherical)
              
           deltaAngleTrackImpact= (None) angle to estimate the stream track over to determine the effect of the impact [similar to deltaAngleTrack] (rad)

           nTrackChunksImpact= (floor(deltaAngleTrack/0.15)+1) number of chunks to divide the progenitor track in near the impact [similar to nTrackChunks]

           nKickPoints= (10xnTrackChunksImpact) number of points along the stream to compute the kicks at (kicks are then interpolated)

        OUTPUT:

           object

        HISTORY:

           2015-06-02 - Started - Bovy (IAS)

        """
        # Parse kwargs
        impactb= kwargs.pop('impactb',1.)
        subhalovel= kwargs.pop('subhalovel',numpy.array([0.,1.,0.]))
        GM= kwargs.pop('GM',None)
        rs= kwargs.pop('rs',None)
        subhalopot= kwargs.pop('subhalopot',None)
        timpact= kwargs.pop('timpact',1.)
        impact_angle= kwargs.pop('impact_angle',1.)
        deltaAngleTrackImpact= kwargs.pop('deltaAngleTrackImpact',None)
        nTrackChunksImpact= kwargs.pop('nTrackChunksImpact',None)
        nKickPoints= kwargs.pop('nKickPoints',None)
        # Now run the regular streamdf setup, but without calculating the 
        # stream track (nosetup=True)
        kwargs['nosetup']= True
        super(streamgapdf,self).__init__(*args,**kwargs)
        # Setup the machinery to go between (x,v) and (Omega,theta) 
        # near the impact
        self._determine_nTrackIterations(kwargs.get('nTrackIterations',None))
        self._determine_deltaAngleTrackImpact(deltaAngleTrackImpact,timpact)
        self._determine_impact_coordtransform(self._deltaAngleTrackImpact,
                                              nTrackChunksImpact,
                                              timpact,impact_angle)
        # Compute \Delta Omega ( \Delta \theta_perp) and \Delta theta, 
        # setup interpolating function
        self._determine_deltav_kick(impactb,subhalovel,
                                    GM,rs,subhalopot,
                                    nKickPoints)
        self._determine_deltaOmegaTheta_kick()
        # (c) Write new meanOmega function based on this?
        # (d) then pass everything to the streamdf setup, should work?
        # (e) First do this for the Plummer sphere, then generalize
        # Determine the necessary number of iterations
        return None

    def _determine_deltav_kick(self,impactb,subhalovel,
                               GM,rs,subhalopot,
                               nKickPoints):
        # Store some impact parameters
        self._impactb= impactb
        self._subhalovel= subhalovel
        # First set nKickPoints
        if nKickPoints is None:
            self._nKickPoints= 10*self._nTrackChunksImpact
        else:
            self._nKickPoints= nKickPoints
        # Analytical Plummer or general potential?
        self._general_kick= GM is None or rs is None
        if self._general_kick and subhalopot is None:
            raise IOError("One of (GM=, rs=) or subhalopot= needs to be set to specify the subhalo's structure")
        if self._general_kick:
            self._subhalopot= subhalopot
        else:
            self._GM= GM
            self._rs= rs
        # Interpolate the track near the gap in (x,v) at the kick_thetas
        self._interpolate_stream_track_kick()
        self._interpolate_stream_track_kick_aA()
        # Then compute delta v along the track
        """
        print "WARNING: SETTING Z and VZ TO ZERO TO DEAL WITH DENIS' EXAMPLE"
        self._gap_ObsTrackXY[:,2]= 0.
        self._gap_ObsTrackXY[:,5]= 0.
        self._kick_interpolatedObsTrackXY[:,2]= 0.
        self._kick_interpolatedObsTrackXY[:,5]= 0.
        self._gap_ObsTrack[:,3]= 0.
        self._gap_ObsTrack[:,4]= 0.
        self._kick_interpolatedObsTrack[:,3]= 0.
        self._kick_interpolatedObsTrack[:,4]= 0.
        """
        if self._general_kick:
            self._kick_deltav=\
                impulse_deltav_general_curvedstream(self._kick_interpolatedObsTrackXY[:,3:],
                                                    self._kick_interpolatedObsTrackXY[:,:3],
                                                    self._impactb,
                                                    self._subhalovel,
                                                    self._kick_ObsTrackXY_closest[:3],
                                                    self._kick_ObsTrackXY_closest[3:],
                                                    self._subhalopot)
        else:
            self._kick_deltav= \
                impulse_deltav_plummer_curvedstream(self._kick_interpolatedObsTrackXY[:,3:],
                                                    self._kick_interpolatedObsTrackXY[:,:3],
                                                    self._impactb,
                                                    self._subhalovel,
                                                    self._kick_ObsTrackXY_closest[:3],
                                                    self._kick_ObsTrackXY_closest[3:],
                                                    self._GM,self._rs)
        return None

    def _determine_deltaOmegaTheta_kick(self):
        # Propagate deltav(angle) -> delta (Omega,theta) [angle]
        # Cylindrical coordinates of the perturbed points
        vXp= self._kick_interpolatedObsTrackXY[:,3]+self._kick_deltav[:,0]
        vYp= self._kick_interpolatedObsTrackXY[:,4]+self._kick_deltav[:,1]
        vZp= self._kick_interpolatedObsTrackXY[:,5]+self._kick_deltav[:,2]
        vRp,vTp,vZp=\
            bovy_coords.rect_to_cyl_vec(vXp,vYp,vZp,
                                        self._kick_interpolatedObsTrack[:,0],
                                        self._kick_interpolatedObsTrack[:,5],
                                        self._kick_interpolatedObsTrack[:,3],
                                        cyl=True)
        # We will abuse streamdf functions for doing the (O,a) -> (R,vR)
        # coordinate transformation, to do this, we assign some of the 
        # attributes related to the track near the impact to the equivalent
        # attributes related to the track at the present time, carefully
        # removing this again to avoid confusion (as much as possible)
        self._interpolatedObsTrack= self._kick_interpolatedObsTrack
        self._ObsTrack= self._gap_ObsTrack
        self._interpolatedObsTrackXY= self._kick_interpolatedObsTrackXY
        self._ObsTrackXY= self._gap_ObsTrackXY
        self._alljacsTrack= self._gap_alljacsTrack
        self._interpolatedObsTrackAA= self._kick_interpolatedObsTrackAA
        self._ObsTrackAA= self._gap_ObsTrackAA
        Oap= self._approxaA(self._kick_interpolatedObsTrack[:,0],
                            vRp,vTp,
                            self._kick_interpolatedObsTrack[:,3],
                            vZp,
                            self._kick_interpolatedObsTrack[:,5],interp=True,
                            cindx=range(len(self._kick_interpolatedObsTrackAA)))
        # Remove attributes again to avoid confusion later
        delattr(self,'_interpolatedObsTrack')
        delattr(self,'_ObsTrack')
        delattr(self,'_interpolatedObsTrackXY')
        delattr(self,'_ObsTrackXY')
        delattr(self,'_alljacsTrack')
        delattr(self,'_interpolatedObsTrackAA')
        delattr(self,'_ObsTrackAA')
        # Generate (dO,da)[angle_offset] and interpolate
        self._kick_dOap= Oap.T-self._kick_interpolatedObsTrackAA
        self._kick_interpdOr=\
            interpolate.InterpolatedUnivariateSpline(\
            self._kick_interpolatedThetasTrack,self._kick_dOap[:,0],k=3)
        self._kick_interpdOp=\
            interpolate.InterpolatedUnivariateSpline(\
            self._kick_interpolatedThetasTrack,self._kick_dOap[:,1],k=3)
        self._kick_interpdOz=\
            interpolate.InterpolatedUnivariateSpline(\
            self._kick_interpolatedThetasTrack,self._kick_dOap[:,2],k=3)
        self._kick_interpdar=\
            interpolate.InterpolatedUnivariateSpline(\
            self._kick_interpolatedThetasTrack,self._kick_dOap[:,3],k=3)
        self._kick_interpdap=\
            interpolate.InterpolatedUnivariateSpline(\
            self._kick_interpolatedThetasTrack,self._kick_dOap[:,4],k=3)
        self._kick_interpdaz=\
            interpolate.InterpolatedUnivariateSpline(\
            self._kick_interpolatedThetasTrack,self._kick_dOap[:,5],k=3)
        return None

    def _interpolate_stream_track_kick(self):
        """Build interpolations of the stream track near the kick"""
        if hasattr(self,'_kick_interpolatedThetasTrack'):
            return None #Already did this
        # Setup the trackpoints where the kick will be computed, covering the 
        # full length of the stream
        self._kick_interpolatedThetasTrack= \
            numpy.linspace(self._gap_thetasTrack[0],
                           self._gap_thetasTrack[-1],
                           self._nKickPoints)
        TrackX= self._gap_ObsTrack[:,0]*numpy.cos(self._gap_ObsTrack[:,5])
        TrackY= self._gap_ObsTrack[:,0]*numpy.sin(self._gap_ObsTrack[:,5])
        TrackZ= self._gap_ObsTrack[:,3]
        TrackvX, TrackvY, TrackvZ=\
            bovy_coords.cyl_to_rect_vec(self._gap_ObsTrack[:,1],
                                        self._gap_ObsTrack[:,2],
                                        self._gap_ObsTrack[:,4],
                                        self._gap_ObsTrack[:,5])
        #Interpolate
        self._kick_interpTrackX=\
            interpolate.InterpolatedUnivariateSpline(self._gap_thetasTrack,
                                                     TrackX,k=3)
        self._kick_interpTrackY=\
            interpolate.InterpolatedUnivariateSpline(self._gap_thetasTrack,
                                                     TrackY,k=3)
        self._kick_interpTrackZ=\
            interpolate.InterpolatedUnivariateSpline(self._gap_thetasTrack,
                                                     TrackZ,k=3)
        self._kick_interpTrackvX=\
            interpolate.InterpolatedUnivariateSpline(self._gap_thetasTrack,
                                                     TrackvX,k=3)
        self._kick_interpTrackvY=\
            interpolate.InterpolatedUnivariateSpline(self._gap_thetasTrack,
                                                     TrackvY,k=3)
        self._kick_interpTrackvZ=\
            interpolate.InterpolatedUnivariateSpline(self._gap_thetasTrack,
                                                     TrackvZ,k=3)
        #Now store an interpolated version of the stream track
        self._kick_interpolatedObsTrackXY= numpy.empty((len(self._kick_interpolatedThetasTrack),6))
        self._kick_interpolatedObsTrackXY[:,0]=\
            self._kick_interpTrackX(self._kick_interpolatedThetasTrack)
        self._kick_interpolatedObsTrackXY[:,1]=\
            self._kick_interpTrackY(self._kick_interpolatedThetasTrack)
        self._kick_interpolatedObsTrackXY[:,2]=\
            self._kick_interpTrackZ(self._kick_interpolatedThetasTrack)
        self._kick_interpolatedObsTrackXY[:,3]=\
            self._kick_interpTrackvX(self._kick_interpolatedThetasTrack)
        self._kick_interpolatedObsTrackXY[:,4]=\
            self._kick_interpTrackvY(self._kick_interpolatedThetasTrack)
        self._kick_interpolatedObsTrackXY[:,5]=\
            self._kick_interpTrackvZ(self._kick_interpolatedThetasTrack)
        #Also in cylindrical coordinates
        self._kick_interpolatedObsTrack= \
            numpy.empty((len(self._kick_interpolatedThetasTrack),6))
        tR,tphi,tZ= bovy_coords.rect_to_cyl(self._kick_interpolatedObsTrackXY[:,0],
                                            self._kick_interpolatedObsTrackXY[:,1],
                                            self._kick_interpolatedObsTrackXY[:,2])
        tvR,tvT,tvZ=\
            bovy_coords.rect_to_cyl_vec(self._kick_interpolatedObsTrackXY[:,3],
                                        self._kick_interpolatedObsTrackXY[:,4],
                                        self._kick_interpolatedObsTrackXY[:,5],
                                        tR,tphi,tZ,cyl=True)
        self._kick_interpolatedObsTrack[:,0]= tR
        self._kick_interpolatedObsTrack[:,1]= tvR
        self._kick_interpolatedObsTrack[:,2]= tvT
        self._kick_interpolatedObsTrack[:,3]= tZ
        self._kick_interpolatedObsTrack[:,4]= tvZ
        self._kick_interpolatedObsTrack[:,5]= tphi
        # Also store (x,v) for the point of closest approach
        self._kick_ObsTrackXY_closest= numpy.array([\
                self._kick_interpTrackX(self._impact_angle),
                self._kick_interpTrackY(self._impact_angle),
                self._kick_interpTrackZ(self._impact_angle),
                self._kick_interpTrackvX(self._impact_angle),
                self._kick_interpTrackvY(self._impact_angle),
                self._kick_interpTrackvZ(self._impact_angle)])
        return None

    def _interpolate_stream_track_kick_aA(self):
        """Build interpolations of the stream track near the impact in action-angle coordinates"""
        if hasattr(self,'_kick_interpolatedObsTrackAA'):
            return None #Already did this
        #Calculate 1D meanOmega on a fine grid in angle and interpolate
        dmOs= numpy.array([self.meanOmega(da,oned=True,tdisrupt=self._tdisrupt-self._timpact)
                           for da in self._kick_interpolatedThetasTrack])
        self._kick_interpTrackAAdmeanOmegaOneD=\
            interpolate.InterpolatedUnivariateSpline(\
            self._kick_interpolatedThetasTrack,dmOs,k=3)
        #Build the interpolated AA
        self._kick_interpolatedObsTrackAA=\
            numpy.empty((len(self._kick_interpolatedThetasTrack),6))
        for ii in range(len(self._kick_interpolatedThetasTrack)):
            self._kick_interpolatedObsTrackAA[ii,:3]=\
                self._progenitor_Omega+dmOs[ii]*self._dsigomeanProgDirection\
                *self._gap_sigMeanSign
            self._kick_interpolatedObsTrackAA[ii,3:]=\
                self._progenitor_angle+self._kick_interpolatedThetasTrack[ii]\
                *self._dsigomeanProgDirection*self._gap_sigMeanSign\
                -self._timpact*self._progenitor_Omega
            self._kick_interpolatedObsTrackAA[ii,3:]=\
                numpy.mod(self._kick_interpolatedObsTrackAA[ii,3:],2.*numpy.pi)
        return None

    def _determine_deltaAngleTrackImpact(self,deltaAngleTrackImpact,timpact):
        self._timpact= timpact
        deltaAngleTrackLim = (self._sigMeanOffset+4.) * numpy.sqrt(
            self._sortedSigOEig[2]) * (self._tdisrupt-self._timpact)
        if deltaAngleTrackImpact is None: 
            deltaAngleTrackImpact= deltaAngleTrackLim
        else:
            if deltaAngleTrackImpact > deltaAngleTrackLim:
                warnings.warn("WARNING: deltaAngleTrackImpact angle range large compared to plausible value.", galpyWarning)
        self._deltaAngleTrackImpact= deltaAngleTrackImpact
        return None

    def _determine_impact_coordtransform(self,deltaAngleTrackImpact,
                                         nTrackChunksImpact,
                                         timpact,impact_angle):
        """Function that sets up the transformation between (x,v) and (O,theta)
        SHOULD REALLY REPLACE THIS WITH MORE GENERAL FUNCTION THAT SETS UP
        THE TRANSFORMATION AT ANY TIME IN THE PAST (ISN'T THAT WHAT THIS IS :-)"""
        # Integrate the progenitor backward to the time of impact
        self._gap_progenitor_setup()
        # Sign of delta angle tells us whether the impact happens to the 
        # leading or trailing arm, self._sigMeanSign contains this info
        if impact_angle > 0.:
            self._gap_leading= True
        else:
            self._gap_leading= False
        self._impact_angle= numpy.fabs(impact_angle)
        self._gap_sigMeanSign= 1.
        if self._gap_leading and self._progenitor_Omega_along_dOmega/self._sigMeanSign < 0.:
            self._gap_sigMeanSign= -1.
        elif not self._gap_leading and self._progenitor_Omega_along_dOmega/self._sigMeanSign > 0.:
            self._gap_sigMeanSign= -1.
        # Determine how much orbital time is necessary for the progenitor's orbit at the time of impact to cover the part of the stream near the impact; we cover the whole leading (or trailing) part of the stream
        if nTrackChunksImpact is None:
            #default is floor(self._deltaAngleTrackImpact/0.15)+1
            self._nTrackChunksImpact= int(numpy.floor(self._deltaAngleTrackImpact/0.15))+1
        else:
            self._nTrackChunksImpact= nTrackChunksImpact
        dt= self._deltaAngleTrackImpact\
            /self._progenitor_Omega_along_dOmega\
            /self._sigMeanSign*self._gap_sigMeanSign
        self._gap_trackts= numpy.linspace(0.,2*dt,2*self._nTrackChunksImpact-1) #to be sure that we cover it
        #Instantiate an auxiliaryTrack, which is an Orbit instance at the mean frequency of the stream, and zero angle separation wrt the progenitor; prog_stream_offset is the offset between this track and the progenitor at zero angle (same as in streamdf, but just done at the time of impact rather than the current time)
        prog_stream_offset=\
            _determine_stream_track_single(self._aA,
                                           self._gap_progenitor,
                                           self._timpact, # around the t of imp
                                           self._progenitor_angle-self._timpact*self._progenitor_Omega,
                                           self._gap_sigMeanSign,
                                           self._dsigomeanProgDirection,
                                           lambda da: self.meanOmega(da,offset_sign=self._gap_sigMeanSign,tdisrupt=self._tdisrupt-self._timpact),
                                           0.) #angle = 0
        auxiliaryTrack= Orbit(prog_stream_offset[3])
        if dt < 0.:
            self._gap_trackts= numpy.linspace(0.,-2.*dt,2.*self._nTrackChunksImpact-1)
            #Flip velocities before integrating
            auxiliaryTrack= auxiliaryTrack.flip()
        auxiliaryTrack.integrate(self._gap_trackts,self._pot)
        if dt < 0.:
            #Flip velocities again
            auxiliaryTrack._orb.orbit[:,1]= -auxiliaryTrack._orb.orbit[:,1]
            auxiliaryTrack._orb.orbit[:,2]= -auxiliaryTrack._orb.orbit[:,2]
            auxiliaryTrack._orb.orbit[:,4]= -auxiliaryTrack._orb.orbit[:,4]
        #Calculate the actions, frequencies, and angle for this auxiliary orbit
        acfs= self._aA.actionsFreqs(auxiliaryTrack(0.),maxn=3)
        auxiliary_Omega= numpy.array([acfs[3],acfs[4],acfs[5]]).reshape(3\
)
        auxiliary_Omega_along_dOmega= \
            numpy.dot(auxiliary_Omega,self._dsigomeanProgDirection)
        # compute the transformation using _determine_stream_track_single
        allAcfsTrack= numpy.empty((self._nTrackChunksImpact,9))
        alljacsTrack= numpy.empty((self._nTrackChunksImpact,6,6))
        allinvjacsTrack= numpy.empty((self._nTrackChunksImpact,6,6))
        thetasTrack= numpy.linspace(0.,self._deltaAngleTrackImpact,
                                    self._nTrackChunksImpact)
        ObsTrack= numpy.empty((self._nTrackChunksImpact,6))
        ObsTrackAA= numpy.empty((self._nTrackChunksImpact,6))
        detdOdJps= numpy.empty((self._nTrackChunksImpact))
        if self._multi is None:
            for ii in range(self._nTrackChunksImpact):
                multiOut= _determine_stream_track_single(self._aA,
                                           auxiliaryTrack,
                                           self._gap_trackts[ii]*numpy.fabs(self._progenitor_Omega_along_dOmega/auxiliary_Omega_along_dOmega), #this factor accounts for the difference in frequency between the progenitor and the auxiliary track, no timpact bc gap_tracks is relative to timpact
                                           self._progenitor_angle-self._timpact*self._progenitor_Omega,
                                           self._gap_sigMeanSign,
                                           self._dsigomeanProgDirection,
                                           lambda da: self.meanOmega(da,offset_sign=self._gap_sigMeanSign,tdisrupt=self._tdisrupt-self._timpact),
                                           thetasTrack[ii])
                allAcfsTrack[ii,:]= multiOut[0]
                alljacsTrack[ii,:,:]= multiOut[1]
                allinvjacsTrack[ii,:,:]= multiOut[2]
                ObsTrack[ii,:]= multiOut[3]
                ObsTrackAA[ii,:]= multiOut[4]
                detdOdJps[ii]= multiOut[5]
        else:
            multiOut= multi.parallel_map(\
                (lambda x: _determine_stream_track_single(self._aA,
                                           auxiliaryTrack,
                                           self._gap_trackts[x]*numpy.fabs(self._progenitor_Omega_along_dOmega/auxiliary_Omega_along_dOmega), #this factor accounts for the difference in frequency between the progenitor and the auxiliary track, no timpact bc gap_tracks is relative to timpact
                                           self._progenitor_angle-self._timpact*self._progenitor_Omega,
                                           self._gap_sigMeanSign,
                                           self._dsigomeanProgDirection,
                                           lambda da: self.meanOmega(da,offset_sign=self._gap_sigMeanSign,tdisrupt=self._tdisrupt-self._timpact),
                                           thetasTrack[x])),
                range(self._nTrackChunksImpact),
                numcores=numpy.amin([self._nTrackChunksImpact,
                                     multiprocessing.cpu_count(),
                                     self._multi]))
            for ii in range(self._nTrackChunksImpact):
                allAcfsTrack[ii,:]= multiOut[ii][0]
                alljacsTrack[ii,:,:]= multiOut[ii][1]
                allinvjacsTrack[ii,:,:]= multiOut[ii][2]
                ObsTrack[ii,:]= multiOut[ii][3]
                ObsTrackAA[ii,:]= multiOut[ii][4]
                detdOdJps[ii]= multiOut[ii][5]
        #Repeat the track calculation using the previous track, to get closer to it
        for nn in range(self.nTrackIterations):
            if self._multi is None:
                for ii in range(self._nTrackChunksImpact):
                    multiOut= _determine_stream_track_single(self._aA,
                                                             Orbit(ObsTrack[ii,:]),
                                                             0.,
                                                             self._progenitor_angle-self._timpact*self._progenitor_Omega,
                                                             self._gap_sigMeanSign,
                                                             self._dsigomeanProgDirection,
                                                             lambda da: self.meanOmega(da,offset_sign=self._gap_sigMeanSign),
                                                             thetasTrack[ii])
                    allAcfsTrack[ii,:]= multiOut[0]
                    alljacsTrack[ii,:,:]= multiOut[1]
                    allinvjacsTrack[ii,:,:]= multiOut[2]
                    ObsTrack[ii,:]= multiOut[3]
                    ObsTrackAA[ii,:]= multiOut[4]
                    detdOdJps[ii]= multiOut[5]
            else:
                multiOut= multi.parallel_map(\
                    (lambda x: _determine_stream_track_single(self._aA,Orbit(ObsTrack[x,:]),0.,
                                                              self._progenitor_angle-self._timpact*self._progenitor_Omega,
                                                              self._gap_sigMeanSign,
                                                              self._dsigomeanProgDirection,
                                           lambda da: self.meanOmega(da,offset_sign=self._gap_sigMeanSign),
                                                              thetasTrack[x])),
                    range(self._nTrackChunksImpact),
                    numcores=numpy.amin([self._nTrackChunksImpact,
                                         multiprocessing.cpu_count(),
                                         self._multi]))
                for ii in range(self._nTrackChunksImpact):
                    allAcfsTrack[ii,:]= multiOut[ii][0]
                    alljacsTrack[ii,:,:]= multiOut[ii][1]
                    allinvjacsTrack[ii,:,:]= multiOut[ii][2]
                    ObsTrack[ii,:]= multiOut[ii][3]
                    ObsTrackAA[ii,:]= multiOut[ii][4]
                    detdOdJps[ii]= multiOut[ii][5]           
        #Store the track
        self._gap_thetasTrack= thetasTrack
        self._gap_ObsTrack= ObsTrack
        self._gap_ObsTrackAA= ObsTrackAA
        self._gap_allAcfsTrack= allAcfsTrack
        self._gap_alljacsTrack= alljacsTrack
        self._gap_allinvjacsTrack= allinvjacsTrack
        self._gap_detdOdJps= detdOdJps
        self._gap_meandetdOdJp= numpy.mean(self._gap_detdOdJps)
        self._gap_logmeandetdOdJp= numpy.log(self._gap_meandetdOdJp)
        #Also calculate _ObsTrackXY in XYZ,vXYZ coordinates
        self._gap_ObsTrackXY= numpy.empty_like(self._gap_ObsTrack)
        TrackX= self._gap_ObsTrack[:,0]*numpy.cos(self._gap_ObsTrack[:,5])
        TrackY= self._gap_ObsTrack[:,0]*numpy.sin(self._gap_ObsTrack[:,5])
        TrackZ= self._gap_ObsTrack[:,3]
        TrackvX, TrackvY, TrackvZ=\
            bovy_coords.cyl_to_rect_vec(self._gap_ObsTrack[:,1],
                                        self._gap_ObsTrack[:,2],
                                        self._gap_ObsTrack[:,4],
                                        self._gap_ObsTrack[:,5])
        self._gap_ObsTrackXY[:,0]= TrackX
        self._gap_ObsTrackXY[:,1]= TrackY
        self._gap_ObsTrackXY[:,2]= TrackZ
        self._gap_ObsTrackXY[:,3]= TrackvX
        self._gap_ObsTrackXY[:,4]= TrackvY
        self._gap_ObsTrackXY[:,5]= TrackvZ       
        return None

    def _gap_progenitor_setup(self):
        """Setup an Orbit instance that's the progenitor integrated backwards"""
        self._gap_progenitor= self._progenitor().flip() # new orbit, flip velocities
        # Make sure we do not use physical coordinates
        self._gap_progenitor.turn_physical_off()
        # Now integrate backward in time until tdisrupt
        ts= numpy.linspace(0.,self._tdisrupt,1001)
        self._gap_progenitor.integrate(ts,self._pot)
        # Flip its velocities, should really write a function for this
        self._gap_progenitor._orb.orbit[:,1]= -self._gap_progenitor._orb.orbit[:,1]
        self._gap_progenitor._orb.orbit[:,2]= -self._gap_progenitor._orb.orbit[:,2]
        self._gap_progenitor._orb.orbit[:,4]= -self._gap_progenitor._orb.orbit[:,4]
        return None

def impulse_deltav_plummer(v,y,b,w,GM,rs):
    """
    NAME:
       impulse_deltav_plummer
    PURPOSE:
       calculate the delta velocity to due an encounter with a Plummer sphere in the impulse approximation; allows for arbitrary velocity vectors, but y is input as the position along the stream
    INPUT:
       v - velocity of the stream (nstar,3)
       y - position along the stream (nstar)
       b - impact parameter
       w - velocity of the Plummer sphere (3)
       GM - mass of the Plummer sphere (in natural units)
       rs - size of the Plummer sphere
    OUTPUT:
       deltav (nstar,3)
    HISTORY:
       2015-04-30 - Written based on Erkal's expressions - Bovy (IAS)
    """
    if len(v.shape) == 1: v= numpy.reshape(v,(1,3))
    nv= v.shape[0]
    # Build the rotation matrices and their inverse
    rot= _rotation_vy(v)
    rotinv= _rotation_vy(v,inv=True)
    # Rotate the Plummer sphere's velocity to the stream frames
    tilew= numpy.sum(rot*numpy.tile(w,(nv,3,1)),axis=-1)
    # Use Denis' expressions
    wperp= numpy.sqrt(tilew[:,0]**2.+tilew[:,2]**2.)
    wpar= numpy.sqrt(numpy.sum(v**2.,axis=1))-tilew[:,1]
    wmag2= wpar**2.+wperp**2.
    wmag= numpy.sqrt(wmag2)
    out= numpy.empty_like(v)
    denom= wmag*((b**2.+rs**2.)*wmag2+wperp**2.*y**2.)
    out[:,0]= (b*wmag2*tilew[:,2]/wperp-y*wpar*tilew[:,0])/denom
    out[:,1]= -wperp**2.*y/denom
    out[:,2]= -(b*wmag2*tilew[:,0]/wperp+y*wpar*tilew[:,2])/denom
    # Rotate back to the original frame
    return 2.0*GM*numpy.sum(rotinv*numpy.tile(out.T,(3,1,1)).T,axis=-1)

def impulse_deltav_plummer_curvedstream(v,x,b,w,x0,v0,GM,rs):
    """
    NAME:
       impulse_deltav_plummer_curvedstream
    PURPOSE:
       calculate the delta velocity to due an encounter with a Plummer sphere in the impulse approximation; allows for arbitrary velocity vectors, and arbitrary position along the stream
    INPUT:
       v - velocity of the stream (nstar,3)
       x - position along the stream (nstar,3)
       b - impact parameter
       w - velocity of the Plummer sphere (3)
       x0 - point of closest approach
       v0 - velocity of point of closest approach
       GM - mass of the Plummer sphere (in natural units)
       rs - size of the Plummer sphere
    OUTPUT:
       deltav (nstar,3)
    HISTORY:
       2015-05-04 - Written based on above - SANDERS
    """
    if len(v.shape) == 1: v= numpy.reshape(v,(1,3))
    if len(x.shape) == 1: x= numpy.reshape(x,(1,3))
    b0 = numpy.cross(w,v0)
    b0 *= b/numpy.sqrt(numpy.sum(b0**2))
    b_ = b0+x-x0
    w = w-v
    wmag = numpy.sqrt(numpy.sum(w**2,axis=1))
    bdotw=numpy.sum(b_*w,axis=1)/wmag
    denom= wmag*(numpy.sum(b_**2,axis=1)+rs**2-bdotw**2)
    denom = 1./denom
    return -2.0*GM*((b_.T-bdotw*w.T/wmag)*denom).T

def _a_integrand(T,y,b,w,pot,compt):
    t = T/(1-T*T)
    X = b+w*t+y*numpy.array([0,1,0])
    r = numpy.sqrt(numpy.sum(X**2))
    return (1+T*T)/(1-T*T)**2*evaluateRforces(r,0.,pot)*X[compt]/r

def _deltav_integrate(y,b,w,pot):
    return numpy.array([integrate.quad(_a_integrand,-1.,1.,args=(y,b,w,pot,i))[0] for i in range(3)])

def impulse_deltav_general(v,y,b,w,pot):
    """
    NAME:
       impulse_deltav_general
    PURPOSE:
       calculate the delta velocity to due an encounter with a general spherical potential in the impulse approximation; allows for arbitrary velocity vectors, but y is input as the position along the stream
    INPUT:
       v - velocity of the stream (nstar,3)
       y - position along the stream (nstar)
       b - impact parameter
       w - velocity of the subhalo (3)
       pot - Potential object or list thereof (should be spherical)
    OUTPUT:
       deltav (nstar,3)
    HISTORY:
       2015-05-04 - SANDERS
       2015-06-15 - Tweak to use galpy' potential objects - Bovy (IAS)
    """
    if len(v.shape) == 1: v= numpy.reshape(v,(1,3))
    nv= v.shape[0]
    # Build the rotation matrices and their inverse
    rot= _rotation_vy(v)
    rotinv= _rotation_vy(v,inv=True)
    # Rotate the subhalo's velocity to the stream frames
    tilew= numpy.sum(rot*numpy.tile(w,(nv,3,1)),axis=-1)
    tilew[:,1]-=numpy.sqrt(numpy.sum(v**2.,axis=1))
    wmag = numpy.sqrt(w[0]**2+w[2]**2)
    b0 = b*numpy.array([-w[2]/wmag,0,w[0]/wmag])
    return numpy.array(map(lambda i:numpy.sum(i[2]
                       *_deltav_integrate(i[0],b0,i[1],pot).T,axis=-1)
                        ,zip(y,tilew,rotinv)))

def impulse_deltav_general_curvedstream(v,x,b,w,x0,v0,pot):
    """
    NAME:
       impulse_deltav_general_curvedstream
    PURPOSE:
       calculate the delta velocity to due an encounter with a general spherical potential in the impulse approximation; allows for arbitrary velocity vectors and arbitrary shaped streams
    INPUT:
       v - velocity of the stream (nstar,3)
       x - position along the stream (nstar,3)
       b - impact parameter
       w - velocity of the subhalo (3)
       x0 - position of closest approach (3)
       v0 - velocity of stream at closest approach (3)
       pot - Potential object or list thereof (should be spherical)
    OUTPUT:
       deltav (nstar,3)
    HISTORY:
       2015-05-04 - SANDERS
       2015-06-15 - Tweak to use galpy' potential objects - Bovy (IAS)
    """
    if len(v.shape) == 1: v= numpy.reshape(v,(1,3))
    if len(x.shape) == 1: x= numpy.reshape(x,(1,3))
    b0 = numpy.cross(w,v0)
    b0 *= b/numpy.sqrt(numpy.sum(b0**2))
    b_ = b0+x-x0
    return numpy.array(map(lambda i:_deltav_integrate(0.,i[1],i[0],pot)
                        ,zip(w-v,b_)))

def _rotation_vy(v,inv=False):
    return _rotate_to_arbitrary_vector(v,[0,1,0],inv)

def _rotate_to_arbitrary_vector(v,a,inv=False):
    """ Return a rotation matrix that rotates v to align with vector a
        i.e. R . v = |v|\hat{a} """
    normv= v/numpy.tile(numpy.sqrt(numpy.sum(v**2.,axis=1)),(3,1)).T
    rotaxis= numpy.cross(normv,a)
    rotaxis/= numpy.tile(numpy.sqrt(numpy.sum(rotaxis**2.,axis=1)),(3,1)).T
    crossmatrix= numpy.empty((len(v),3,3))
    crossmatrix[:,0,:]= numpy.cross(rotaxis,[1,0,0])
    crossmatrix[:,1,:]= numpy.cross(rotaxis,[0,1,0])
    crossmatrix[:,2,:]= numpy.cross(rotaxis,[0,0,1])
    costheta= normv[:,1]
    sintheta= numpy.sqrt(1.-costheta**2.)
    if inv: sgn= 1.
    else: sgn= -1.
    out= numpy.tile(costheta,(3,3,1)).T*numpy.tile(numpy.eye(3),(len(v),1,1))\
        +sgn*numpy.tile(sintheta,(3,3,1)).T*crossmatrix\
        +numpy.tile(1.-costheta,(3,3,1)).T\
        *(rotaxis[:,:,numpy.newaxis]*rotaxis[:,numpy.newaxis,:])
    return out
