#The DF of a tidal stream
import numpy
from galpy import potential
from galpy.orbit import Orbit
from galpy.util import bovy_coords
class streamdf:
    """The DF of a tidal stream"""
    def __init__(self,sigv,sigx,progenitor=None,pot=None,aA=None,
                 ts=None,integrate_method='dopr54_c'):
        """
        NAME:
           __init__
        PURPOSE:
           Initialize a quasi-isothermal DF
        INPUT:
           sigv - radial velocity dispersion of the 'progenitor'
           sigx - spatial velocity dispersion of the 'progenitor'
           progenitor= progenitor orbit as Orbit instance 
                       (if not integrated, given with +t velocity)
           pot= Potential instance or list thereof
           aA= actionAngle instance used to convert (x,v) to actions if
               action-angle coordinates are used to fit the stream
           ts= times used in orbit integrations
           integrate_method= (default: 'dopr54_c') integration method to use
        OUTPUT:
           object
        HISTORY:
           2013-09-16 - Started - Bovy (IAS)
        """
        self._sigv= sigv
        if pot is None:
            raise IOError("pot= must be set")
        self._pot= pot
        self._integrate_method= integrate_method
        self._aA= aA
        self._progenitor= progenitor
        self._ts= ts
        if not self._ts is None: self._nts= len(self._ts)
        if True:
            if not isinstance(self._progenitor,Orbit):
                raise IOError('progenitor= kwargs needs to be an Orbit instance')
            if not hasattr(self._progenitor,'orbit'): #not integrated yet
                #First flip the velocity for backward integration
                self._progenitor._orb.vxvv[1]= -self._progenitor._orb.vxvv[1]
                self._progenitor._orb.vxvv[2]= -self._progenitor._orb.vxvv[2]
                self._progenitor._orb.vxvv[4]= -self._progenitor._orb.vxvv[4]
                self._progenitor.integrate(self._ts,self._pot,
                                           method=self._integrate_method)
            self._progenitor_R= self._progenitor.R(self._ts)
            self._progenitor_vR= self._progenitor.vR(self._ts)
            self._progenitor_vT= self._progenitor.vT(self._ts)
            self._progenitor_z= self._progenitor.z(self._ts)
            self._progenitor_vz= self._progenitor.vz(self._ts)
            self._progenitor_phi= self._progenitor.phi(self._ts)
            self._progenitor_rperi= numpy.amin(self._progenitor_R**2.+
                                               self._progenitor_Z**2.)
            self._progenitor_rap= numpy.amax(self._progenitor_R**2.+
                                             self._progenitor_Z**2.)
        if self._aA is None: #direct integration, so we need the progenitor orbit in rectangular coordinates
            tX, tY, tZ= bovy_coords.cyl_to_rect(self._progenitor_R,
                                                self._progenitor_phi,
                                                self._progenitor_z)
            tvX, tvY, tvZ= bovy_coords.cyl_to_rect_vec(self._progenitor_vR,
                                                       self._progenitor_vT,
                                                       self._progenitor_vz,
                                                       self._progenitor_phi)
            self._progenitor_X= tX
            self._progenitor_Y= tY
            self._progenitor_Z= tZ
            self._progenitor_vX= tvX
            self._progenitor_vY= tvY
            self._progenitor_vZ= tvZ
            #Assume that dr/rperi ~ sigmar/Vcirc(rperi)
            if sigx is None:
                self._sigx= self._sigv/potential.vcirc(self._pot,
                                                       self._progenitor_rperi)\
                                                       *self._progenitor_rperi
            else:
                self._sigx= sigx
            self._sigx2= self._sigx**2.
            self._sigv2= self._sigv**2.
        else: #calculate estimates sigmas for the actions
            self._sigjr= (self._progenitor.rap()-self._progenitor.rperi())/numpy.pi*self._sigv
            self._siglz= self._progenitor.rperi()*self._sigv
            self._sigjz= 2.*self._progenitor.zmax()/numpy.pi*self._sigv
            if sigx is None:
                self._sigx= self._siglz/self._progenitor.R()/self._progenitor.vT() #estimate spread in angles as dimensionless actions-spread
            else:
                self._sigx= sigx
            self._sigangle= self._sigx
            self._sigjr2= self._sigjr**2.
            self._siglz2= self._siglz**2.
            self._sigjz2= self._sigjz**2.
            self._sigangle2= self._sigangle**2.
        return None
        
    def __call__(self,*args,**kwargs):
        """
        NAME:
           __call__
        PURPOSE:
           evaluate the DF
        INPUT:
           Either:
              a)(jr,lz,jz,Omegar,Omegaphi,Omegaz,angler,anglephi,anglez) tuple
                 where:
                    jr - radial action
                    lz - z-component of angular momentum
                    jz - vertical action
                    Omegar - radial frequency
                    Omegaphi - azimuthal frequency
                    Omegaz - vertical frequency
                    angler - radial angle
                    anglephi - azimuthal angle
                    anglez - vertical angle
              b) R,vR,vT,z,vz,phi ndarray [nobjects,ntimes]
              c) Orbit instance or list thereof
           log= if True, return the natural log
           rect= if True, R,vR,vT,z,vz,phi is actually X,Y,Z,vX,vY,vZ
        OUTPUT:
           value of DF
        HISTORY:
           2013-09-16 - Written - Bovy (IAS)
        """
        if self._aA is None:
            return self._callDirectIntegration(*args,**kwargs)
        else:
            return self._callActionAngleMethod(*args,**kwargs)

    def _callDirectIntegration(self,*args,**kwargs):
        """Evaluate the DF using direct integration"""
        #First parse log
        if kwargs.has_key('log'):
            log= kwargs['log']
            kwargs.pop('log')
        else:
            log= False
        X,Y,Z,vX,vY,vZ= self.prepData4Direct(*args,**kwargs)
        #For each object, marginalize over time
        logdft= -0.5*(1./self._sigx2\
                          *((X-self._progenitor_X)**2.
                            +(Y-self._progenitor_Y)**2.
                            +(Z-self._progenitor_Z)**2.)
                      +1./self._sigv2\
                          *((vX-self._progenitor_vX)**2.
                            +(vY-self._progenitor_vY)**2.
                            +(vZ-self._progenitor_vZ)**2.))
        logdf= _mylogsumexp(logdft,axis=1)
        out= numpy.sum(logdf)-3.*X.shape[0]*numpy.log(self._sigv*self._sigx)
        if log:
            return out
        else:
            return numpy.exp(out)

    def _callActionAngleMethod(self,*args,**kwargs):
        """Evaluate the DF using the action-angle formalism"""
        #First parse log
        if kwargs.has_key('log'):
            log= kwargs['log']
            kwargs.pop('log')
        else:
            log= False
        raise NotImplementedError("Using action-angle framework not immplemented yet")

    def prepData4Direct(self,*args,**kwargs):
        """
        NAME:
           prepData4Direct
        PURPOSE:
           prepare stream data for the direct integration method
           (integrate and transform, can then be re-used for different DF but
           same potential)
        INPUT:
           __call__ inputs
        OUTPUT:
           X,Y,Z,vX,vY,vZ [nobj,ntimes]
        HISTORY:
           2013-09-16 - Written - Bovy (IAS)
        """
        R,vR,vT,z,vz,phi= self._parse_call_args(True,*args)
        if kwargs.has_key('rect') and kwargs['rect']:
            X,Y,Z,vX,vY,vZ= R,vR,vT,z,vz,phi
        else:
            X,Y,Z= bovy_coords.cyl_to_rect(R,phi,z)
            vX,vY,vZ= bovy_coords.cyl_to_rect_vec(vR,vT,vz,phi)
        return (X,Y,Z,vX,vY,vZ)

    def _parse_call_args(self,directIntegration=True,*args):
        """Helper function to parse the arguments to the __call__ and related functions"""
        if directIntegration:
            RasOrbit= False
            if len(args) == 5:
                raise IOError("Must specify phi for streamdf")
            if len(args) == 6:
                R,vR,vT,z,vz,phi= args
                if isinstance(R,float):
                    o= Orbit([R,-vR,-vT,z,-vz,phi])
                    o.integrate(self._ts,
                                pot=self._pot,
                                method=self._integrate_method)
                    this_orbit= o.getOrbit()
                    R= numpy.reshape(this_orbit[:,0],(1,self._nts))
                    vR= numpy.reshape(this_orbit[:,1],(1,self._nts))
                    vT= numpy.reshape(this_orbit[:,2],(1,self._nts))
                    z= numpy.reshape(this_orbit[:,3],(1,self._nts))
                    vz= numpy.reshape(this_orbit[:,4],(1,self._nts))
                    phi= numpy.reshape(this_orbit[:,5],(1,self._nts))
                if len(R.shape) == 1: #not integrated yet
                    os= [Orbit([R[ii],vR[ii],vT[ii],z[ii],vz[ii],phi[ii]]) for ii in range(R.shape[0])]
                    RasOrbit= True
            if isinstance(args[0],Orbit) \
                    or (isinstance(args[0],list) and isinstance(args[0][0],Orbit)) \
                    or RasOrbit:
                if RasOrbit:
                    pass
                elif not isinstance(args[0],list):
                    os= [args[0]]
                else:
                    os= args[0]
                if not hasattr(os[0],'orbit'): #not integrated yet
                    for o in os: #flip velocities for backwards integration
                        o._orb.vxvv[1]= -o._orb.vxvv[1]
                        o._orb.vxvv[2]= -o._orb.vxvv[2]
                        o._orb.vxvv[4]= -o._orb.vxvv[4]
                    [o.integrate(self._ts,pot=self._pot,
                                 method=self._integrate_method) for o in os]
                    nts= os[0].getOrbit().shape[0]
                    no= len(os)
                    R= numpy.empty((no,nts))
                    vR= numpy.empty((no,nts))
                    vT= numpy.empty((no,nts))
                    z= numpy.empty((no,nts))
                    vz= numpy.empty((no,nts))
                    phi= numpy.empty((no,nts))
                    for ii in range(len(os)):
                        this_orbit= os[ii].getOrbit()
                        R[ii,:]= this_orbit[:,0]
                        vR[ii,:]= this_orbit[:,1]
                        vT[ii,:]= this_orbit[:,2]
                        z[ii,:]= this_orbit[:,3]
                        vz[ii,:]= this_orbit[:,4]
                        phi[ii,:]= this_orbit[:,5]
            return (R,vR,vT,z,vz,phi)

def _mylogsumexp(arr,axis=0):
    """Faster logsumexp?"""
    minarr= numpy.amax(arr,axis=axis)
    if axis == 1:
        minarr= numpy.reshape(minarr,(arr.shape[0],1))
    if axis == 0:
        minminarr= numpy.tile(minarr,(arr.shape[0],1))
    elif axis == 1:
        minminarr= numpy.tile(minarr,(1,arr.shape[1]))
    elif axis == None:
        minminarr= numpy.tile(minarr,arr.shape)
    else:
        raise NotImplementedError("'_mylogsumexp' not implemented for axis > 2")
    if axis == 1:
        minarr= numpy.reshape(minarr,(arr.shape[0]))
    return minarr+numpy.log(numpy.sum(numpy.exp(arr-minminarr),axis=axis))
