# The DF of a gap in a tidal stream
from functools import wraps
import numpy
import warnings
import multiprocessing
from scipy import integrate, interpolate, optimize
from galpy.util import galpyWarning, bovy_coords, multi
from galpy.orbit import Orbit
from galpy.potential import evaluateRforces, MovingObjectPotential
import galpy.df_src.streamdf
from galpy.df_src.streamdf import _determine_stream_track_single
def impact_check_range(func):
    """Decorator to check the range of interpolated kicks"""
    @wraps(func)
    def impact_wrapper(*args,**kwargs):
        if isinstance(args[1],numpy.ndarray):
            out= numpy.zeros(len(args[1]))
            goodIndx= (args[1] < args[0]._deltaAngleTrackImpact)*(args[1] > 0.)
            out[goodIndx]= func(args[0],args[1][goodIndx])
            return out
        elif args[1] >= args[0]._deltaAngleTrackImpact or args[1] <= 0.:
            return 0.
        else:
            return func(*args,**kwargs)
    return impact_wrapper
class streamgapdf(galpy.df_src.streamdf.streamdf):
    """The DF of a gap in a tidal stream"""
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
        # For setup later
        nTrackChunks= kwargs.pop('nTrackChunks',None)
        interpTrack= kwargs.pop('interpTrack',
                                galpy.df_src.streamdf._INTERPDURINGSETUP)
        useInterp= kwargs.pop('useInterp',
                              galpy.df_src.streamdf._USEINTERP)
        # Analytical Plummer or general potential?
        self._general_kick= GM is None or rs is None
        if self._general_kick and subhalopot is None:
            raise IOError("One of (GM=, rs=) or subhalopot= needs to be set to specify the subhalo's structure")
        if self._general_kick:
            self._subhalopot= subhalopot
        else:
            self._GM= GM
            self._rs= rs
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
        # Then pass everything to the normal streamdf setup
        self.nInterpolatedTrackChunks= 201 #more expensive now
        super(streamgapdf,self)._determine_stream_track(nTrackChunks)
        self._useInterp= useInterp
        if interpTrack or self._useInterp:
            super(streamgapdf,self)._interpolate_stream_track()
            super(streamgapdf,self)._interpolate_stream_track_aA()
        super(streamgapdf,self).calc_stream_lb()
        return None

    def pOparapar(self,Opar,apar):
        """
        NAME:

           pOparapar

        PURPOSE:

           return the probability of a given parallel (frequency,angle) offset pair

        INPUT:

           Opar - parallel frequency offset (array)

           apar - parallel angle offset along the stream (scalar)

        OUTPUT:

           p(Opar,apar)

        HISTORY:

           2015-11-17 - Written - Bovy (UofT)

        """
        if isinstance(Opar,(int,float,numpy.float32,numpy.float64)):
            Opar= numpy.array([Opar])
        out= numpy.zeros(len(Opar))
        # Compute ts and where they were at impact for all
        ts= apar/Opar
        apar_impact= apar-Opar*self._timpact
        dOpar_impact= self._kick_interpdOpar(apar_impact)
        Opar_b4impact= Opar-dOpar_impact
        ts_b4impact= apar_impact/Opar_b4impact
        # Evaluate the two regimes: stripped before or after impact
        out[ts < self._timpact]=\
            numpy.exp(-0.5*(Opar-self._meandO)**2.\
                           /self._sortedSigOEig[2])/\
                           numpy.sqrt(self._sortedSigOEig[2])
        out[(ts >= self._timpact)\
                *(self._timpact+ts_b4impact < self._tdisrupt)]=\
            numpy.exp(-0.5*(Opar_b4impact-self._meandO)**2.\
                           /self._sortedSigOEig[2])/\
                           numpy.sqrt(self._sortedSigOEig[2])
        return out

    def density_par(self,dangle,tdisrupt=None):
        """
        NAME:

           density_par

        PURPOSE:

           calculate the density as a function of parallel angle, assuming a uniform time distribution up to a maximum time

        INPUT:

           dangle - angle offset

        OUTPUT:

           density(angle)

        HISTORY:

           2015-11-17 - Written - Bovy (UofT)

        """
        if tdisrupt is None: tdisrupt= self._tdisrupt
        Tlow= 1./2./self._sigMeanOffset\
            -numpy.sqrt(1.-(1./2./self._sigMeanOffset)**2.)
        return integrate.quad(lambda T: numpy.sqrt(self._sortedSigOEig[2])\
                                  *(1+T*T)/(1-T*T)**2.\
                                  *self.pOparapar(T/(1-T*T)\
                                                      *numpy.sqrt(self._sortedSigOEig[2])\
                                                      +self._meandO,dangle),
                              Tlow,1.)[0]

    def meanOmega(self,dangle,oned=False,tdisrupt=None):
        """
        NAME:

           meanOmega

        PURPOSE:

           calculate the mean frequency as a function of angle, assuming a uniform time distribution up to a maximum time

        INPUT:

           dangle - angle offset

           oned= (False) if True, return the 1D offset from the progenitor (along the direction of disruption)

        OUTPUT:

           mean Omega

        HISTORY:

           2015-11-17 - Written - Bovy (UofT)

        """
        if tdisrupt is None: tdisrupt= self._tdisrupt
        Tlow= 1./2./self._sigMeanOffset\
            -numpy.sqrt(1.-(1./2./self._sigMeanOffset)**2.)
        num=\
            integrate.quad(lambda T: (T/(1-T*T)\
                                          *numpy.sqrt(self._sortedSigOEig[2])\
                                          +self._meandO)\
                               *numpy.sqrt(self._sortedSigOEig[2])\
                               *(1+T*T)/(1-T*T)**2.\
                               *self.pOparapar(T/(1-T*T)\
                                                   *numpy.sqrt(self._sortedSigOEig[2])\
                                                   +self._meandO,dangle),
                           Tlow,1.)[0]
        denom=\
            integrate.quad(lambda T: numpy.sqrt(self._sortedSigOEig[2])\
                               *(1+T*T)/(1-T*T)**2.\
                               *self.pOparapar(T/(1-T*T)\
                                                   *numpy.sqrt(self._sortedSigOEig[2])\
                                                   +self._meandO,dangle),
                           Tlow,1.)[0]
        dO1D= num/denom
        if oned: return dO1D
        else:
            return self._progenitor_Omega+dO1D*self._dsigomeanProgDirection\
                *self._sigMeanSign

    def _rewind_angle_impact(self,dangle):
        """
        NAME:
           _rewind_angle_impact
        PURPOSE:
           Find the parallel angle at which a star was just after the impact
        INPUT:
           dangle - current parallel angle in rad
        OUTPUT:
           angle at impact
        HISTORY:
           2015-06-22 - Written - Bovy (IAS)
        """
        # First subtract meanOmega evolution
        dangle-= super(streamgapdf,self).meanOmega(dangle,oned=True)\
            *self._timpact
        # Now solve the kick's evolution  itself
        if not hasattr(self,'_daparMax'):
            # Maximum dOpar
            self._daparMax= self._kick_interpdOpar_dapar.roots()
            self._dOparMax= self._kick_interpdOpar(self._daparMax)
            self._daparMax-= self._impact_angle
            # Angle where caustic forms at the current time, not yet used
            self._daparCau= interpolate.InterpolatedUnivariateSpline(\
                self._kick_interpolatedThetasTrack,
                numpy.dot(self._kick_dOap[:,:3],self._dsigomeanProgDirection)\
                    +self._kick_interpolatedThetasTrack/self._timpact,
                k=4).derivative(1).roots()
            self._dOparCau= self._kick_interpdOpar(self._daparCau)
            self._daparCau-= self._impact_angle
            if len(self._daparCau) == 0: self._caustic_phase= False
            else: self._caustic_phase= True
        if not self._caustic_phase: # Single-valued everywhere
            guess= dangle
            dguess= numpy.fabs(self._dOparMax[0])*self._timpact*1.2
            out= optimize.brentq(lambda da: dangle
                                 -self._kick_interpdOpar(da)*self._timpact
                                 -da,
                                 guess-dguess,guess+dguess)
        """
        except ValueError:
            # Only get into trouble at the edges; assume the kick is zero
            out= dangle-self._meandO*self._sigMeanSign*self._timpact
        """
        return out

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
        # Interpolate the track near the gap in (x,v) at the kick_thetas
        self._interpolate_stream_track_kick()
        self._interpolate_stream_track_kick_aA()
        # Then compute delta v along the track
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
        self._nTrackChunks= self._nTrackChunksImpact
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
        delattr(self,'_nTrackChunks')
        # Generate (dO,da)[angle_offset] and interpolate (raw here, see below
        # for form that checks range)
        self._kick_dOap= Oap.T-self._kick_interpolatedObsTrackAA
        self._kick_interpdOr_raw=\
            interpolate.InterpolatedUnivariateSpline(\
            self._kick_interpolatedThetasTrack,self._kick_dOap[:,0],k=3)
        self._kick_interpdOp_raw=\
            interpolate.InterpolatedUnivariateSpline(\
            self._kick_interpolatedThetasTrack,self._kick_dOap[:,1],k=3)
        self._kick_interpdOz_raw=\
            interpolate.InterpolatedUnivariateSpline(\
            self._kick_interpolatedThetasTrack,self._kick_dOap[:,2],k=3)
        self._kick_interpdar_raw=\
            interpolate.InterpolatedUnivariateSpline(\
            self._kick_interpolatedThetasTrack,self._kick_dOap[:,3],k=3)
        self._kick_interpdap_raw=\
            interpolate.InterpolatedUnivariateSpline(\
            self._kick_interpolatedThetasTrack,self._kick_dOap[:,4],k=3)
        self._kick_interpdaz_raw=\
            interpolate.InterpolatedUnivariateSpline(\
            self._kick_interpolatedThetasTrack,self._kick_dOap[:,5],k=3)
        # Also interpolate parallel and perpendicular frequencies
        self._kick_dOaparperp=\
            numpy.dot(self._kick_dOap[:,:3],
                      self._sigomatrixEig[1][:,self._sigomatrixEigsortIndx])
        self._kick_interpdOpar_raw=\
            interpolate.InterpolatedUnivariateSpline(\
            self._kick_interpolatedThetasTrack,
            numpy.dot(self._kick_dOap[:,:3],self._dsigomeanProgDirection),k=4) # to get zeros with sproot
        self._kick_interpdOperp0_raw=\
            interpolate.InterpolatedUnivariateSpline(\
            self._kick_interpolatedThetasTrack,self._kick_dOaparperp[:,0],k=3)
        self._kick_interpdOperp1_raw=\
            interpolate.InterpolatedUnivariateSpline(\
            self._kick_interpolatedThetasTrack,self._kick_dOaparperp[:,1],k=3)
        # Also construct derivative of dOpar
        self._kick_interpdOpar_dapar= self._kick_interpdOpar_raw.derivative(1)
        return None

    # Functions that evaluate the interpolated kicks, but also check the range
    @impact_check_range
    def _kick_interpdOpar(self,da):
        return self._kick_interpdOpar_raw(da)
    @impact_check_range
    def _kick_interpdOperp0(self,da):
        return self._kick_interpdOperp0_raw(da)
    @impact_check_range
    def _kick_interpdOperp1(self,da):
        return self._kick_interpdOperp1_raw(da)
    @impact_check_range
    def _kick_interpdOr(self,da):
        return self._kick_interpdOr_raw(da)
    @impact_check_range
    def _kick_interpdOp(self,da):
        return self._kick_interpdOp_raw(da)
    @impact_check_range
    def _kick_interpdOz(self,da):
        return self._kick_interpdOz_raw(da)
    @impact_check_range
    def _kick_interpdar(self,da):
        return self._kick_interpdar_raw(da)
    @impact_check_range
    def _kick_interpdap(self,da):
        return self._kick_interpdap_raw(da)
    @impact_check_range
    def _kick_interpdaz(self,da):
        return self._kick_interpdaz_raw(da)

    def _interpolate_stream_track_kick(self):
        """Build interpolations of the stream track near the kick"""
        if hasattr(self,'_kick_interpolatedThetasTrack'): #pragma: no cover
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
        if hasattr(self,'_kick_interpolatedObsTrackAA'): #pragma: no cover
            return None #Already did this
        #Calculate 1D meanOmega on a fine grid in angle and interpolate
        dmOs= numpy.array([\
                super(streamgapdf,self).meanOmega(da,oned=True,
                                                  tdisrupt=self._tdisrupt
                                                  -self._timpact)
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
                warnings.warn("WARNING: deltaAngleTrackImpact angle range large compared to plausible value", galpyWarning)
        self._deltaAngleTrackImpact= deltaAngleTrackImpact
        return None

    def _determine_impact_coordtransform(self,deltaAngleTrackImpact,
                                         nTrackChunksImpact,
                                         timpact,impact_angle):
        """Function that sets up the transformation between (x,v) and (O,theta)"""
        # Integrate the progenitor backward to the time of impact
        self._gap_progenitor_setup()
        # Sign of delta angle tells us whether the impact happens to the
        # leading or trailing arm, self._sigMeanSign contains this info
        if impact_angle > 0.:
            self._gap_leading= True
        else:
            self._gap_leading= False
        if (self._gap_leading and not self._leading) \
                or (not self._gap_leading and self._leading):
            raise ValueError('Modeling leading (trailing) impact for trailing (leading) arm; this is not allowed because it is nonsensical in this framework')
        self._impact_angle= numpy.fabs(impact_angle)
        self._gap_sigMeanSign= 1.
        if (self._gap_leading and self._progenitor_Omega_along_dOmega/self._sigMeanSign < 0.) \
                or (not self._gap_leading and self._progenitor_Omega_along_dOmega/self._sigMeanSign > 0.):
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
                                           lambda da: super(streamgapdf,self).meanOmega(da,offset_sign=self._gap_sigMeanSign,tdisrupt=self._tdisrupt-self._timpact),
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
                                           lambda da: super(streamgapdf,self).meanOmega(da,offset_sign=self._gap_sigMeanSign,tdisrupt=self._tdisrupt-self._timpact),
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
                                           lambda da: super(streamgapdf,self).meanOmega(da,offset_sign=self._gap_sigMeanSign,tdisrupt=self._tdisrupt-self._timpact),
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
                                                             lambda da: super(streamgapdf,self).meanOmega(da,offset_sign=self._gap_sigMeanSign,tdisrupt=self._tdisrupt-self._timpact),
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
                                                              lambda da: super(streamgapdf,self).meanOmega(da,offset_sign=self._gap_sigMeanSign,tdisrupt=self._tdisrupt-self._timpact),
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

################################SAMPLE THE DF##################################
    def _sample_aAt(self,n):
        """Sampling frequencies, angles, and times part of sampling, for stream with gap"""
        # Use streamdf's _sample_aAt to generate unperturbed frequencies,
        # angles
        Om,angle,dt= super(streamgapdf,self)._sample_aAt(n)
        # Now rewind angles by timpact, apply the kicks, and run forward again
        dangle_at_impact= angle-numpy.tile(self._progenitor_angle.T,(n,1)).T\
            -(Om-numpy.tile(self._progenitor_Omega.T,(n,1)).T)*self._timpact
        dangle_par_at_impact= numpy.dot(dangle_at_impact.T,
                                        self._dsigomeanProgDirection)\
                                        *self._gap_sigMeanSign
        # Calculate and apply kicks (points not yet released have zero kick)
        dOr= self._kick_interpdOr(dangle_par_at_impact)
        dOp= self._kick_interpdOp(dangle_par_at_impact)
        dOz= self._kick_interpdOz(dangle_par_at_impact)
        Om[0,:]+= dOr
        Om[1,:]+= dOp
        Om[2,:]+= dOz
        angle[0,:]+=\
            self._kick_interpdar(dangle_par_at_impact)+dOr*self._timpact
        angle[1,:]+=\
            self._kick_interpdap(dangle_par_at_impact)+dOp*self._timpact
        angle[2,:]+=\
            self._kick_interpdaz(dangle_par_at_impact)+dOz*self._timpact
        return (Om,angle,dt)

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
    if len(v.shape) == 1:
        v= numpy.reshape(v,(1,3))
        y= numpy.reshape(y,(1,1))
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
    # deal w/ perpendicular impacts
    wperp0Indx= numpy.fabs(wperp) < 10.**-10.
    out[wperp0Indx,0]= (b*wmag2[wperp0Indx]-y[wperp0Indx]*wpar[wperp0Indx]*tilew[wperp0Indx,0])/denom[wperp0Indx]
    out[wperp0Indx,2]=-(b*wmag2[wperp0Indx]+y[wperp0Indx]*wpar[wperp0Indx]*tilew[wperp0Indx,2])/denom[wperp0Indx]
    # Rotate back to the original frame
    return 2.0*GM*numpy.sum(\
        rotinv*numpy.swapaxes(numpy.tile(out.T,(3,1,1)).T,1,2),axis=-1)

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

def HernquistX(s):
    """
      Computes X function from equations (33) & (34) of Hernquist (1990)
    """
    if(s<0.):
        raise ValueError("s must be positive in Hernquist X function")
    elif(s<1.):
        return numpy.log((1+numpy.sqrt(1-s*s))/s)/numpy.sqrt(1-s*s)
    elif(s==1.):
        return 1.
    else:
        return numpy.arccos(1./s)/numpy.sqrt(s*s-1)

def impulse_deltav_hernquist(v,y,b,w,GM,rs):
    """
    NAME:

       impulse_deltav_hernquist

    PURPOSE:

       calculate the delta velocity to due an encounter with a Hernquist sphere in the impulse approximation; allows for arbitrary velocity vectors, but y is input as the position along the stream

    INPUT:

       v - velocity of the stream (nstar,3)

       y - position along the stream (nstar)

       b - impact parameter

       w - velocity of the Hernquist sphere (3)

       GM - mass of the Hernquist sphere (in natural units)

       rs - size of the Hernquist sphere

    OUTPUT:

       deltav (nstar,3)

    HISTORY:

       2015-08-13 SANDERS, using Wyn Evans calculation

    """
    if len(v.shape) == 1: v= numpy.reshape(v,(1,3))
    nv= v.shape[0]
    # Build the rotation matrices and their inverse
    rot= _rotation_vy(v)
    rotinv= _rotation_vy(v,inv=True)
    # Rotate the Plummer sphere's velocity to the stream frames
    tilew= numpy.sum(rot*numpy.tile(w,(nv,3,1)),axis=-1)
    wperp= numpy.sqrt(tilew[:,0]**2.+tilew[:,2]**2.)
    wpar= numpy.sqrt(numpy.sum(v**2.,axis=1))-tilew[:,1]
    wmag2= wpar**2.+wperp**2.
    wmag= numpy.sqrt(wmag2)
    B = numpy.sqrt(b**2.+wperp**2.*y**2./wmag2)
    denom= wmag*(B**2-rs**2)
    denom=1./denom
    s = numpy.sqrt(2.*B/(rs+B))
    HernquistXv=numpy.vectorize(HernquistX)
    Xfac = 1.-2.*rs/(rs+B)*HernquistXv(s)
    out= numpy.empty_like(v)
    out[:,0]= (b*tilew[:,2]/wperp-y*wpar*tilew[:,0]/wmag2)*denom*Xfac
    out[:,1]= -wperp**2.*y*denom*Xfac/wmag2
    out[:,2]= -(b*tilew[:,0]/wperp+y*wpar*tilew[:,2]/wmag2)*denom*Xfac
    # deal w/ perpendicular impacts
    wperp0Indx= numpy.fabs(wperp) < 10.**-10.
    out[wperp0Indx,0]= (b-y[wperp0Indx]*wpar[wperp0Indx]*tilew[wperp0Indx,0]/wmag2[wperp0Indx])*denom[wperp0Indx]*Xfac[wperp0Indx]
    out[wperp0Indx,2]=-(b+y[wperp0Indx]*wpar[wperp0Indx]*tilew[wperp0Indx,2]/wmag2[wperp0Indx])*denom[wperp0Indx]*Xfac[wperp0Indx]
    # Rotate back to the original frame
    return 2.0*GM*numpy.sum(\
        rotinv*numpy.swapaxes(numpy.tile(out.T,(3,1,1)).T,1,2),axis=-1)

def impulse_deltav_hernquist_curvedstream(v,x,b,w,x0,v0,GM,rs):
    """
    NAME:

       impulse_deltav_plummer_hernquist

    PURPOSE:

       calculate the delta velocity to due an encounter with a Hernquist sphere in the impulse approximation; allows for arbitrary velocity vectors, and arbitrary position along the stream

    INPUT:

       v - velocity of the stream (nstar,3)

       x - position along the stream (nstar,3)

       b - impact parameter

       w - velocity of the Hernquist sphere (3)

       x0 - point of closest approach

       v0 - velocity of point of closest approach

       GM - mass of the Hernquist sphere (in natural units)

       rs - size of the Hernquist sphere

    OUTPUT:

       deltav (nstar,3)

    HISTORY:

       2015-08-13 - SANDERS, using Wyn Evans calculation

    """
    if len(v.shape) == 1: v= numpy.reshape(v,(1,3))
    if len(x.shape) == 1: x= numpy.reshape(x,(1,3))
    b0 = numpy.cross(w,v0)
    b0 *= b/numpy.sqrt(numpy.sum(b0**2))
    b_ = b0+x-x0
    w = w-v
    wmag = numpy.sqrt(numpy.sum(w**2,axis=1))
    bdotw=numpy.sum(b_*w,axis=1)/wmag
    B = numpy.sqrt(numpy.sum(b_**2,axis=1)-bdotw**2)
    denom= wmag*(B**2-rs**2)
    denom=1./denom
    s = numpy.sqrt(2.*B/(rs+B))
    HernquistXv=numpy.vectorize(HernquistX)
    Xfac = 1.-2.*rs/(rs+B)*HernquistXv(s)
    return -2.0*GM*((b_.T-bdotw*w.T/wmag)*Xfac*denom).T

def _a_integrand(T,y,b,w,pot,compt):
    t = T/(1-T*T)
    X = b+w*t+y*numpy.array([0,1,0])
    r = numpy.sqrt(numpy.sum(X**2))
    return (1+T*T)/(1-T*T)**2*evaluateRforces(pot,r,0.)*X[compt]/r

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
    wmag = numpy.sqrt(tilew[:,0]**2+tilew[:,2]**2)
    b0 = b*numpy.array([-tilew[:,2]/wmag,numpy.zeros(nv),tilew[:,0]/wmag]).T
    return numpy.array(list(map(lambda i:numpy.sum(i[3]
                       *_deltav_integrate(i[0],i[1],i[2],pot).T,axis=-1)
                        ,zip(y,b0,tilew,rotinv))))

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
    return numpy.array(list(map(lambda i:_deltav_integrate(0.,i[1],i[0],pot)
                        ,zip(w-v,b_))))

def impulse_deltav_general_orbitintegration(v,x,b,w,x0,v0,pot,tmax,galpot,
                                            tmaxfac=10.,nsamp=1000,
                                            integrate_method='symplec4_c'):
    """
    NAME:

       impulse_deltav_general_orbitintegration

    PURPOSE:

       calculate the delta velocity to due an encounter with a general spherical potential NOT in the impulse approximation by integrating each particle in the underlying galactic potential; allows for arbitrary velocity vectors and arbitrary shaped streams.

    INPUT:

       v - velocity of the stream (nstar,3)

       x - position along the stream (nstar,3)

       b - impact parameter

       w - velocity of the subhalo (3)

       x0 - position of closest approach (3)

       v0 - velocity of stream at closest approach (3)

       pot - Potential object or list thereof (should be spherical)

       tmax - maximum integration time

       galpot - galpy Potential object or list thereof

       nsamp(1000) - number of forward integration points

       integrate_method= ('symplec4_c') orbit integrator to use (see Orbit.integrate)

    OUTPUT:

       deltav (nstar,3)

    HISTORY:

       2015-08-17 - SANDERS

    """
    if len(v.shape) == 1: v= numpy.reshape(v,(1,3))
    if len(x.shape) == 1: x= numpy.reshape(x,(1,3))
    nstar,ndim=numpy.shape(v)
    b0 = numpy.cross(w,v0)
    b0 *= b/numpy.sqrt(numpy.sum(b0**2))
    times = numpy.linspace(0.,tmax,nsamp)
    xres = numpy.zeros(shape=(len(x),nsamp*2-1,3))
    R, phi, z= bovy_coords.rect_to_cyl(x[:,0],x[:,1],x[:,2])
    vR, vp, vz= bovy_coords.rect_to_cyl_vec(v[:,0],v[:,1],v[:,2],
                                            R,phi,z,cyl=True)
    for i in range(nstar):
        o = Orbit([R[i],vR[i],vp[i],z[i],vz[i],phi[i]])
        o.integrate(times,galpot,method=integrate_method)
        xres[i,nsamp:,0]=o.x(times)[1:]
        xres[i,nsamp:,1]=o.y(times)[1:]
        xres[i,nsamp:,2]=o.z(times)[1:]
        oreverse = o.flip()
        oreverse.integrate(times,galpot,method=integrate_method)
        xres[i,:nsamp,0]=oreverse.x(times)[::-1]
        xres[i,:nsamp,1]=oreverse.y(times)[::-1]
        xres[i,:nsamp,2]=oreverse.z(times)[::-1]
    times = numpy.concatenate((-times[::-1],times[1:]))
    nsamp = len(times)
    X = b0+xres-x0-numpy.outer(times,w)
    r = numpy.sqrt(numpy.sum(X**2,axis=-1))
    acc = (numpy.reshape(evaluateRforces(pot,r.flatten(),0.),(nstar,nsamp))/r)[:,:,numpy.newaxis]*X
    return integrate.simps(acc,x=times,axis=1)

def impulse_deltav_general_fullplummerintegration(v,x,b,w,x0,v0,galpot,GM,rs,
                                                  tmaxfac=10.,N=1000,
                                                integrate_method='symplec4_c'):
    """
    NAME:

       impulse_deltav_general_fullplummerintegration

    PURPOSE:

       calculate the delta velocity to due an encounter with a moving Plummer sphere and galactic potential relative to just in galactic potential

    INPUT:

       v - velocity of the stream (nstar,3)

       x - position along the stream (nstar,3)

       b - impact parameter

       w - velocity of the subhalo (3)

       x0 - position of closest approach (3)

       v0 - velocity of stream at closest approach (3)

       galpot - Galaxy Potential object

       GM - mass of Plummer

       rs - scale of Plummer

       tmaxfac(10) - multiple of rs/|w-v0| to use for time integration interval

       N(1000) - number of forward integration points

       integrate_method('symplec4_c') - orbit integrator to use (see Orbit.integrate)

    OUTPUT:

       deltav (nstar,3)

    HISTORY:

       2015-08-18 - SANDERS

    """
    if len(v.shape) == 1: v= numpy.reshape(v,(1,3))
    if len(x.shape) == 1: x= numpy.reshape(x,(1,3))

    nstar,ndim=numpy.shape(v)
    b0 = numpy.cross(w,v0)
    b0 *= b/numpy.sqrt(numpy.sum(b0**2))
    X = x0-b0

    # Setup Plummer orbit
    R, phi, z= bovy_coords.rect_to_cyl(X[0],X[1],X[2])
    vR, vp, vz= bovy_coords.rect_to_cyl_vec(w[0],w[1],w[2],R,phi,z,cyl=True)
    tmax = tmaxfac*rs/numpy.sqrt(numpy.sum((w-v0)**2))
    times = numpy.linspace(0.,tmax,N)
    dtimes = numpy.linspace(-tmax,tmax,2*N)
    o = Orbit(vxvv=[R,-vR,-vp,z,-vz,phi])
    o.integrate(times,galpot,method=integrate_method)
    oplum = o(times[-1]).flip()
    oplum.integrate(dtimes,galpot,method=integrate_method)
    plumpot = MovingObjectPotential(orbit=oplum, GM=GM, softening_model='plummer', softening_length=rs)

    # Now integrate each particle backwards in galaxy potential, forwards in combined potential and backwards again in galaxy and take diff

    deltav = numpy.zeros((nstar,3))
    R, phi, z= bovy_coords.rect_to_cyl(x[:,0],x[:,1],x[:,2])
    vR, vp, vz= bovy_coords.rect_to_cyl_vec(v[:,0],v[:,1],v[:,2],
                                            R,phi,z,cyl=True)
    for i in range(nstar):
      ostar= Orbit(vxvv=[R[i],-vR[i],-vp[i],z[i],-vz[i],phi[i]])
      ostar.integrate(times,galpot,method=integrate_method)
      oboth = ostar(times[-1]).flip()
      oboth.integrate(dtimes,[galpot,plumpot],method=integrate_method)
      ogalpot = oboth(times[-1]).flip()
      ogalpot.integrate(times,galpot,method=integrate_method)
      deltav[i][0] = -ogalpot.vx(times[-1]) - v[i][0]
      deltav[i][1] = -ogalpot.vy(times[-1]) - v[i][1]
      deltav[i][2] = -ogalpot.vz(times[-1]) - v[i][2]
    return deltav

def _astream_integrand_x(t,y,v,b,w,b2,w2,wperp,wperp2,wpar,GSigma,rs2):
    return GSigma(t)*(b*w2*w[2]/wperp-(y-v*t)*wpar*w[0])\
        /((b2+rs2)*w2+wperp2*(y-v*t)**2.)
def _astream_integrand_y(t,y,v,b2,w2,wperp2,GSigma,rs2):
    return GSigma(t)*(y-v*t)/((b2+rs2)*w2+wperp2*(y-v*t)**2.)
def _astream_integrand_z(t,y,v,b,w,b2,w2,wperp,wperp2,wpar,GSigma,rs2):
    return -GSigma(t)*(b*w2*w[0]/wperp+(y-v*t)*wpar*w[2])\
        /((b2+rs2)*w2+wperp2*(y-v*t)**2.)

def impulse_deltav_plummerstream(v,y,b,w,GSigma,rs,tmin=None,tmax=None):
    """
    NAME:

       impulse_deltav_plummerstream

    PURPOSE:

       calculate the delta velocity to due an encounter with a Plummer-softened stream in the impulse approximation; allows for arbitrary velocity vectors, but y is input as the position along the stream

    INPUT:

       v - velocity of the stream (nstar,3)

       y - position along the stream (nstar)

       b - impact parameter

       w - velocity of the Plummer sphere (3)

       GSigma - surface density of the Plummer-softened stream (in natural units); should be a function of time

       rs - size of the Plummer sphere

       tmin, tmax= (None) minimum and maximum time to consider for GSigma (need to be set)

    OUTPUT:

       deltav (nstar,3)

    HISTORY:

       2015-11-14 - Written - Bovy (UofT)

    """
    if len(v.shape) == 1: 
        v= numpy.reshape(v,(1,3))
        y= numpy.reshape(y,(1,1))
    if tmax is None or tmax is None:
        raise ValueError("tmin= and tmax= need to be set")
    nv= v.shape[0]
    vmag= numpy.sqrt(numpy.sum(v**2.,axis=1))
    # Build the rotation matrices and their inverse
    rot= _rotation_vy(v)
    rotinv= _rotation_vy(v,inv=True)
    # Rotate the perturbing stream's velocity to the stream frames
    tilew= numpy.sum(rot*numpy.tile(w,(nv,3,1)),axis=-1)
    # Use similar expressions to Denis'
    wperp= numpy.sqrt(tilew[:,0]**2.+tilew[:,2]**2.)
    wpar= numpy.sqrt(numpy.sum(v**2.,axis=1))-tilew[:,1]
    wmag2= wpar**2.+wperp**2.
    wmag= numpy.sqrt(wmag2)
    b2= b**2.
    rs2= rs**2.
    wperp2= wperp**2.
    out= numpy.empty_like(v)
    out[:,0]= [1./wmag[ii]\
                   *integrate.quad(_astream_integrand_x,
                                   tmin,tmax,args=(y[ii],
                                                    vmag[ii],b,tilew[ii],
                                                    b2,wmag2[ii],
                                                    wperp[ii],wperp2[ii],
                                                    wpar[ii],
                                                    GSigma,rs2))[0]
               for ii in range(len(y))]
    out[:,1]= [-wperp2[ii]/wmag[ii]\
                    *integrate.quad(_astream_integrand_y,
                                    tmin,tmax,args=(y[ii],
                                                    vmag[ii],
                                                    b2,wmag2[ii],
                                                    wperp2[ii],
                                                    GSigma,rs2))[0]
                for ii in range(len(y))]
    out[:,2]= [1./wmag[ii]\
                    *integrate.quad(_astream_integrand_z       ,
                                    tmin,tmax,args=(y[ii],
                                                    vmag[ii],b,tilew[ii],
                                                    b2,wmag2[ii],
                                                    wperp[ii],wperp2[ii],
                                                    wpar[ii],
                                                    GSigma,rs2))[0]
               for ii in range(len(y))]
    # Rotate back to the original frame
    return 2.0*numpy.sum(\
        rotinv*numpy.swapaxes(numpy.tile(out.T,(3,1,1)).T,1,2),axis=-1)

def _astream_integrand(t,b_,orb,tx,w,GSigma,rs2,tmin,compt):
    teval= tx-tmin-t
    b__= b_+numpy.array([orb.x(teval),orb.y(teval),orb.z(teval)])
    w = w-numpy.array([orb.vx(teval),orb.vy(teval),orb.vz(teval)])
    wmag = numpy.sqrt(numpy.sum(w**2))
    bdotw=numpy.sum(b__*w)/wmag
    denom= wmag*(numpy.sum(b__**2)+rs2-bdotw**2)
    denom = 1./denom
    return -2.0*GSigma(t)*(((b__.T-bdotw*w.T/wmag)*denom).T)[compt]

def _astream_integrate(b_,orb,tx,w,GSigma,rs2,otmin,tmin,tmax):
    return numpy.array([integrate.quad(_astream_integrand,tmin,tmax,
                                       args=(b_,orb,tx,w,GSigma,rs2,
                                             otmin,i))[0] \
                            for i in range(3)])

def impulse_deltav_plummerstream_curvedstream(v,x,t,b,w,x0,v0,GSigma,rs,
                                              galpot,tmin=None,tmax=None):
    """
    NAME:

       impulse_deltav_plummerstream_curvedstream

    PURPOSE:

       calculate the delta velocity to due an encounter with a Plummer sphere in the impulse approximation; allows for arbitrary velocity vectors, and arbitrary position along the stream; velocities and positions are assumed to lie along an orbit

    INPUT:

       v - velocity of the stream (nstar,3)

       x - position along the stream (nstar,3)

       t - times at which (v,x) are reached, wrt the closest impact t=0 (nstar)

       b - impact parameter

       w - velocity of the Plummer sphere (3)

       x0 - point of closest approach

       v0 - velocity of point of closest approach

       GSigma - surface density of the Plummer-softened stream (in natural units); should be a function of time

       rs - size of the Plummer sphere

       galpot - galpy Potential object or list thereof

       tmin, tmax= (None) minimum and maximum time to consider for GSigma

    OUTPUT:

       deltav (nstar,3)

    HISTORY:

       2015-11-20 - Written based on Plummer sphere above - Bovy (UofT)

    """
    if len(v.shape) == 1: v= numpy.reshape(v,(1,3))
    if len(x.shape) == 1: x= numpy.reshape(x,(1,3))
    # Integrate an orbit to use to figure out where each (v,x) is at each time
    R, phi, z= bovy_coords.rect_to_cyl(x0[0],x0[1],x0[2])
    vR, vT, vz= bovy_coords.rect_to_cyl_vec(v0[0],v0[1],v0[2],R,phi,z,cyl=True)
    # First back, then forward to cover the entire range with 1 orbit
    o= Orbit([R,vR,vT,z,vz,phi]).flip()
    ts= numpy.linspace(0.,numpy.fabs(numpy.amin(t)+tmin),101)
    o.integrate(ts,galpot)
    o= o(ts[-1]).flip()
    ts= numpy.linspace(0.,numpy.amax(t)+tmax-numpy.amin(t)-tmin,201)
    o.integrate(ts,galpot)
    # Calculate kicks
    b0 = numpy.cross(w,v0)
    b0 *= b/numpy.sqrt(numpy.sum(b0**2))
    return numpy.array(list(map(lambda i:_astream_integrate(\
                    b0-x0,o,i,w,GSigma,rs**2.,numpy.amin(t)+tmin,
                    tmin,tmax),t)))

def _rotation_vy(v,inv=False):
    return _rotate_to_arbitrary_vector(v,[0,1,0],inv)

def _rotate_to_arbitrary_vector(v,a,inv=False):
    """ Return a rotation matrix that rotates v to align with unit vector a
        i.e. R . v = |v|\hat{a} """
    normv= v/numpy.tile(numpy.sqrt(numpy.sum(v**2.,axis=1)),(3,1)).T
    rotaxis= numpy.cross(normv,a)
    rotaxis/= numpy.tile(numpy.sqrt(numpy.sum(rotaxis**2.,axis=1)),(3,1)).T
    crossmatrix= numpy.empty((len(v),3,3))
    crossmatrix[:,0,:]= numpy.cross(rotaxis,[1,0,0])
    crossmatrix[:,1,:]= numpy.cross(rotaxis,[0,1,0])
    crossmatrix[:,2,:]= numpy.cross(rotaxis,[0,0,1])
    costheta= numpy.dot(normv,a)
    sintheta= numpy.sqrt(1.-costheta**2.)
    if inv: sgn= 1.
    else: sgn= -1.
    out= numpy.tile(costheta,(3,3,1)).T*numpy.tile(numpy.eye(3),(len(v),1,1))\
        +sgn*numpy.tile(sintheta,(3,3,1)).T*crossmatrix\
        +numpy.tile(1.-costheta,(3,3,1)).T\
        *(rotaxis[:,:,numpy.newaxis]*rotaxis[:,numpy.newaxis,:])
    out[numpy.fabs(costheta-1.) < 10.**-10.]= numpy.eye(3)
    out[numpy.fabs(costheta+1.) < 10.**-10.]= -numpy.eye(3)
    return out
