#The DF of a tidal stream
import numpy
from galpy.orbit import Orbit
from galpy.util import bovy_coords, stable_cholesky
class streamdf:
    """The DF of a tidal stream"""
    def __init__(self,sigv,progenitor=None,pot=None,aA=None,
                 deltaAngle=0.3):
        """
        NAME:
           __init__
        PURPOSE:
           Initialize a quasi-isothermal DF
        INPUT:
           sigv - radial velocity dispersion of the progenitor
           progenitor= progenitor orbit as Orbit instance 
           pot= Potential instance or list thereof
           aA= actionAngle instance used to convert (x,v) to actions
           deltaAngle= (0.3) estimate of 'dispersion' in largest angle
        OUTPUT:
           object
        HISTORY:
           2013-09-16 - Started - Bovy (IAS)
           2013-11-25 - Started over - Bovy (IAS)
        """
        self._sigv= sigv
        self._deltaAngle= deltaAngle
        if pot is None:
            raise IOError("pot= must be set")
        self._pot= pot
        self._aA= aA
        self._progenitor= progenitor
        #Progenitor orbit: Calculate actions, frequencies, and angles for the progenitor
        acfs= aA.actionsFreqsAngles(self._progenitor,maxn=3)
        self._progenitor_jr= acfs[0][0]
        self._progenitor_lz= acfs[1][0]
        self._progenitor_jz= acfs[2][0]
        self._progenitor_Omegar= acfs[3]
        self._progenitor_Omegaphi= acfs[4]
        self._progenitor_Omegaz= acfs[5]
        self._progenitor_angler= acfs[6]
        self._progenitor_anglephi= acfs[7]
        self._progenitor_anglez= acfs[8]
        #Calculate dO/dJ Jacobian at the progenitor
        self._dOdJp= calcaAJac(self._progenitor._orb.vxvv,
                               self._aA,dxv=None,dOdJ=True,
                               _initacfs=acfs)
        #From the progenitor orbit, determine the sigmas in J and angle
        self._sigjr= (self._progenitor.rap()-self._progenitor.rperi())/numpy.pi*self._sigv
        self._siglz= self._progenitor.rperi()*self._sigv
        self._sigjz= 2.*self._progenitor.zmax()/numpy.pi*self._sigv
        #Estimate the frequency covariance matrix from a diagonal J matrix x dOdJ
        self._sigjmatrix= numpy.diag([self._sigjr**2.,
                                      self._siglz**2.,
                                      self._sigjz**2.])
        self._sigomatrix= numpy.dot(self._dOdJp,
                                    numpy.dot(self._sigjmatrix,self._dOdJp.T))
        #Estimate angle spread as the ratio of the largest to the middle eigenvalue
        self._sigomatrixEig= numpy.linalg.eig(self._sigomatrix)
        sortedEig= sorted(self._sigomatrixEig[0])
        self._sigangle2= sortedEig[1]/sortedEig[2]*self._deltaAngle
        self._sigangle= numpy.sqrt(self._sigangle2)
        self._lnsigangle= numpy.log(self._sigangle)
        #Store cholesky of sigomatrix for fast evaluation
        self._sigomatrixL= stable_cholesky(self._sigomatrix,10.**-8.)
        self._sigomatrixDet= numpy.linalg.det(self._sigomatrix)
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
        out= numpy.sum(logdf)-3.*X.shape[0]*(self._lnsigv+self._lnsigx)
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
        djr,dlz,djz,dOr,dOphi,dOz,dar,daphi,daz= self.prepData4aA(*args,**kwargs)
        no= len(djr)
        logdfJ= numpy.sum(-0.5*(1./self._sigjr2*djr**2.
                                +1./self._siglz2*dlz**2.
                                +1./self._sigjz2*djz**2.))\
                                -no*(self._lnsigjr+self._lnsiglz+self._lnsigjz)
        da2= dar**2.+daphi**2.+daz**2.
        do2= dOr**2.+dOphi**2.+dOz**2.
        doa= dar*dOr+daphi*dOphi+daz*dOz
        logdfA= numpy.sum(-0.5/self._sigangle2*(da2-doa**2./do2)\
                               -0.5*numpy.log(do2))-2.*no*self._lnsigangle
        out= logdfA+logdfJ
        if log:
            return out
        else:
            return numpy.exp(out)

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
           X,Y,Z,vX,vY,vZ each [nobj,ntimes]
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

    def prepData4aA(self,*args,**kwargs):
        """
        NAME:
           prepData4aA
        PURPOSE:
           prepare stream data for the action-angle method
        INPUT:
           __call__ inputs
        OUTPUT:
           djr,dlz,djz,dOmegar,dOmegaphi,dOmegaz,dangler,danglephi,danglez; each [nobj,ntimes]; differences wrt the progenitor
        HISTORY:
           2013-09-17 - Written - Bovy (IAS)
        """
        if len(args) == 9: #actions, frequencies, and angles are given
            return args
        R,vR,vT,z,vz,phi= self._parse_call_args(False,*args)
        jr,lz,jz,Or,Ophi,Oz,ar,aphi,az= self._aA.actionsFreqsAngles(R,vR,vT,z,vz,phi,maxn=3)
        djr= jr-self._progenitor_jr
        dlz= lz-self._progenitor_lz
        djz= jz-self._progenitor_jz
        dOr= Or-self._progenitor_Omegar
        dOphi= Ophi-self._progenitor_Omegaphi
        dOz= Oz-self._progenitor_Omegaz
        dar= ar-self._progenitor_angler
        daphi= aphi-self._progenitor_anglephi
        daz= az-self._progenitor_anglez
        #Assuming single wrap, resolve large angle differences (wraps should be marginalized over)
        dar[(dar < -4.)]+= 2.*numpy.pi
        dar[(dar > 4.)]-= 2.*numpy.pi
        daphi[(daphi < -4.)]+= 2.*numpy.pi
        daphi[(daphi > 4.)]-= 2.*numpy.pi
        daz[(daz < -4.)]+= 2.*numpy.pi
        daz[(daz > 4.)]-= 2.*numpy.pi
        return (djr,dlz,djz,dOr,dOphi,dOz,dar,daphi,daz)

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
        else:
            if len(args) == 5:
                raise IOError("Must specify phi for streamdf")
            elif len(args) == 6:
                return args
            elif isinstance(args[0],Orbit):
                return (o.R(),o.vR(),o.vT(),o.z(),o.vz(),o.phi())
            elif isinstance(args[0],list) and isinstance(args[0][0],Orbit):
                R, vR, vT, z, vz, phi= [], [], [], [], [], []
                for o in args[0]:
                    R.append(o.R())
                    vR.append(o.vR())
                    vT.append(o.vT())
                    z.append(o.z())
                    vz.append(o.vz())
                    phi.append(o.phi())
                return (numpy.array(R),numpy.array(vR),numpy.array(vT),
                        numpy.array(z),numpy.array(vz),numpy.array(phi))

def calcaAJac(xv,aA,dxv=None,freqs=False,dOdJ=False,
              lb=False,coordFunc=None,
              Vnorm=220.,Rnorm=8.,R0=8.,Zsun=0.025,vsun=[-11.1,8.*30.24,7.25],
              _initacfs=None):
    """
    NAME:
       calcaAJac
    PURPOSE:
       calculate the Jacobian d(J,theta)/d(x,v)
    INPUT:
       xv - phase-space point: Either
          1) [R,vR,vT,z,vz,phi]
          2) [l,b,D,vlos,pmll,pmbb] (if lb=True, see below)
          3) list/array of 6 numbers that can be transformed into (normalized) R,vR,vT,z,vz,phi using coordFunc

       aA - actionAngle instance

       dxv - infinitesimal to use (rescaled for lb, so think fractionally))

       freqs= (False) if True, go to frequencies rather than actions

       dOdJ= (False), actually calculate d Frequency / d action

       lb= (False) if True, start with (l,b,D,vlos,pmll,pmbb) in (deg,deg,kpc,km/s,mas/yr,mas/yr)
       Vnorm= (220) circular velocity to normalize with when lb=True
       Rnorm= (8) Galactocentric radius to normalize with when lb=True
       R0= (8) Galactocentric radius of the Sun (kpc)
       Zsun= (0.025) Sun's height above the plane (kpc)
       vsun= ([-11.1,241.92,7.25]) Sun's motion in cylindrical coordinates (vR positive away from center)

       coordFunc= (None) if set, this is a function that takes xv and returns R,vR,vT,z,vz,phi in normalized units (units where vc=1 at r=1 if the potential is normalized that way, for example)

    OUTPUT:
       Jacobian matrix
    HISTORY:
       2013-11-25 - Written - Bovy (IAS) 
    """
    if lb:
        coordFunc= lambda x: lbCoordFunc(xv,Vnorm,Rnorm,R0,Zsun,vsun)
    if not coordFunc is None:
        R, vR, vT, z, vz, phi= coordFunc(xv)
    else:
        R, vR, vT, z, vz, phi= xv[0],xv[1],xv[2],xv[3],xv[4],xv[5]
    if dxv is None:
        dxv= 10.**-8.*numpy.ones(6)
    if lb:
        #Re-scale some of the differences, to be more natural
        dxv[0]*= 180./numpy.pi
        dxv[1]*= 180./numpy.pi
        dxv[2]*= Rnorm
        dxv[3]*= Vnorm
        dxv[4]*= Vnorm/4.74047/xv[2]
        dxv[5]*= Vnorm/4.74047/xv[2]
    jac= numpy.zeros((6,6))
    if dOdJ:
        jac2= numpy.zeros((6,6))
    if _initacfs is None:
        jr,lz,jz,Or,Ophi,Oz,ar,aphi,az\
            = aA.actionsFreqsAngles(R,vR,vT,z,vz,phi,maxn=3)
    else:
        jr,lz,jz,Or,Ophi,Oz,ar,aphi,az\
            = _initacfs
    for ii in range(6):
        temp= xv[ii]+dxv[ii] #Trick to make sure dxv is representable
        dxv[ii]= temp-xv[ii]
        xv[ii]+= dxv[ii]
        if not coordFunc is None:
            tR, tvR, tvT, tz, tvz, tphi= coordFunc(xv)
        else:
            tR, tvR, tvT, tz, tvz, tphi= xv[0],xv[1],xv[2],xv[3],xv[4],xv[5]
        tjr,tlz,tjz,tOr,tOphi,tOz,tar,taphi,taz\
            = aA.actionsFreqsAngles(tR,tvR,tvT,tz,tvz,tphi,maxn=3)
        xv[ii]-= dxv[ii]
        if freqs:
            jac[0,ii]= (tOr-Or)/dxv[ii]
            jac[1,ii]= (tOphi-Ophi)/dxv[ii]
            jac[2,ii]= (tOz-Oz)/dxv[ii]
        else:        
            jac[0,ii]= (tjr-jr)/dxv[ii]
            jac[1,ii]= (tlz-lz)/dxv[ii]
            jac[2,ii]= (tjz-jz)/dxv[ii]
        if dOdJ:
            jac2[0,ii]= (tOr-Or)/dxv[ii]
            jac2[1,ii]= (tOphi-Ophi)/dxv[ii]
            jac2[2,ii]= (tOz-Oz)/dxv[ii]
        jac[3,ii]= (tar-ar)/dxv[ii]
        jac[4,ii]= (taphi-aphi)/dxv[ii]
        jac[5,ii]= (taz-az)/dxv[ii]
    if dOdJ:
        jac2[3,:]= jac[3,:]
        jac2[4,:]= jac[4,:]
        jac2[5,:]= jac[5,:]
        jac= numpy.dot(jac2,numpy.linalg.inv(jac))[0:3,0:3]
    return jac

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

def lbCoordFunc(xv,Vnorm,Rnorm,R0,Zsun,vsun):
    #Input is (l,b,D,vlos,pmll,pmbb) in (deg,deg,kpc,km/s,mas/yr,mas/yr)
    X,Y,Z= bovy_coords.lbd_to_XYZ(xv[0],xv[1],xv[2],degree=True)
    R,phi,Z= bovy_coords.XYZ_to_galcencyl(X,Y,Z,
                                          Xsun=R0,Ysun=0.,Zsun=Zsun)
    vx,vy,vz= bovy_coords.vrpmllpmbb_to_vxvyvz(xv[3],xv[4],xv[5],
                                               X,Y,Z,XYZ=True)
    vR,vT,vZ= bovy_coords.vxvyvz_to_galcencyl(vx,vy,vz,R,phi,Z,galcen=True,
                                              vsun=vsun)
    R/= Rnorm
    Z/= Rnorm
    vR/= Vnorm
    vT/= Vnorm
    vZ/= Vnorm
    return (R,vR,vT,Z,vZ,phi)
