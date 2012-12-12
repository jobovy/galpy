#A 'Binney' quasi-isothermal DF
import math
import numpy
from scipy import optimize, interpolate, integrate, linalg
from galpy import potential
from galpy import actionAngle
_NSIGMA=4
_DEFAULTNGL=20
class quasiisothermaldf:
    """Class that represents a 'Binney' quasi-isothermal DF"""
    def __init__(self,hr,sr,sz,hsr,hsz,pot=None,aA=None,
                 cutcounter=False,
                 _precomputerg=True,_precomputergrmax=None,
                 _precomputergnLz=51,
                 ro=1.,lo=10./220.*8.):
        """
        NAME:
           __init__
        PURPOSE:
           Initialize a quasi-isothermal DF
        INPUT:
           hr - radial scale length
           sr - radial velocity dispersion at the solar radius
           sz - vertical velocity dispersion at the solar radius
           hsr - radial-velocity-dispersion scale length
           hsz - vertial-velocity-dispersion scale length
           pot= Potential instance or list thereof
           aA= actionAngle instance used to convert (x,v) to actions
           cutcounter= if True, set counter-rotating stars' DF to zero
           ro= reference radius for surface mass and sigmas
           lo= reference angular momentum below where there are significant numbers of retrograde stars
        OTHER INPUTS:
           _precomputerg= if True (default), pre-compute the rL(L)
           _precomputergrmax= if set, this is the maximum R for which to pre-compute rg (default: 5*hr)
           _precomputergnLz if set, number of Lz to pre-compute rg for (default: 51)
        OUTPUT:
           object
        HISTORY:
           2012-07-25 - Started - Bovy (IAS@MPIA)
        """
        self._hr= hr
        self._sr= sr
        self._sz= sz
        self._hsr= hsr
        self._hsz= hsz
        self._ro= ro
        self._lo= lo
        self._lnsr= math.log(self._sr)
        self._lnsz= math.log(self._sz)
        if pot is None:
            raise IOError("pot= must be set")
        self._pot= pot
        if aA is None:
            raise IOError("aA= must be set")
        self._aA= aA
        self._cutcounter= cutcounter
        if _precomputerg:
            if _precomputergrmax is None:
                _precomputergrmax= 5*self._hr
            self._precomputergrmax= _precomputergrmax
            self._precomputergnLz= _precomputergnLz
            self._precomputergLzmin= 0.01
            self._precomputergLzmax= self._precomputergrmax\
                *potential.vcirc(self._pot,self._precomputergrmax)
            self._precomputergLzgrid= numpy.linspace(self._precomputergLzmin,self._precomputergLzmax,self._precomputergnLz)
            self._rls= numpy.array([potential.rl(self._pot,l) for l in self._precomputergLzgrid])
            #Spline interpolate
            self._rgInterp= interpolate.InterpolatedUnivariateSpline(self._precomputergLzgrid,self._rls,k=3)
        else:
            self._precomputergrmax= 0.
            self._rgInterp= None
            self._rls= None
            self._precomputergnr= None
            self._precomputergLzgrid= None
        self._precomputerg= _precomputerg
        return None

    def __call__(self,*args,**kwargs):
        """
        NAME:
           __call__
        PURPOSE:
           return the DF
        INPUT:
           Either:
              a)(jr,lz,jz) tuple
                 where:
                    jr - radial action
                    lz - z-component of angular momentum
                    jz - vertical action
              b) R,vR,vT,z,vz
              c) Orbit instance: initial condition used if that's it, orbit(t)
                 if there is a time given as well

           log= if True, return the natural log
           +scipy.integrate.quadrature kwargs
           func= function of (jr,lz,jz) to multiply f with (useful for moments)
        OUTPUT:
           value of DF
        HISTORY:
           2012-07-25 - Written - Bovy (IAS@MPIA)
        NOTE:
           For Miyamoto-Nagai/adiabatic approximation this seems to take 
           about 30 ms / evaluation in the extended Solar neighborhood
           For a MWPotential/adiabatic approximation this takes about 
           50 ms / evaluation in the extended Solar neighborhood

           For adiabatic-approximation grid this seems to take 
           about 0.67 to 0.75 ms / evaluation in the extended Solar 
           neighborhood (includes some out of the grid)

           up to 200x faster when called with vector R,vR,vT,z,vz
        """
        #First parse log
        if kwargs.has_key('log'):
            log= kwargs['log']
            kwargs.pop('log')
        else:
            log= False
        #First parse args
        if len(args) == 1: #(jr,lz,jz)
            jr,lz,jz= args[0]
        else:
            #Use self._aA to calculate the actions
            try:
                jr,lz,jz= self._aA(*args,**kwargs)
            except actionAngle.UnboundError:
                if log: return -numpy.finfo(numpy.dtype(numpy.float64)).max
                else: return 0.
            #if isinstance(jr,(list,numpy.ndarray)) and len(jr) > 1: jr= jr[0]
            #if isinstance(jz,(list,numpy.ndarray)) and len(jz) > 1: jz= jz[0]
        if not isinstance(lz,numpy.ndarray) and self._cutcounter and lz < 0.:
            if log: return -numpy.finfo(numpy.dtype(numpy.float64)).max
            else: return 0.
        #First calculate rg
        thisrg= self.rg(lz)
        #Then calculate the epicycle and vertical frequencies
        kappa, nu= self._calc_epifreq(thisrg), self._calc_verticalfreq(thisrg)
        Omega= numpy.fabs(lz)/thisrg/thisrg
        #calculate surface-densities and sigmas
        lnsurfmass= (self._ro-thisrg)/self._hr
        lnsr= self._lnsr+(self._ro-thisrg)/self._hsr
        lnsz= self._lnsz+(self._ro-thisrg)/self._hsz
        #Calculate func
        if kwargs.has_key('func'):
            if log:
                funcTerm= numpy.log(kwargs['func'](jr,lz,jz))
            else:
                funcFactor= kwargs['func'](jr,lz,jz)
        #Calculate fsr
        else:
            if log:
                funcTerm= 0.
            else:
                funcFactor= 1.            
        if log:
            lnfsr= numpy.log(Omega)+lnsurfmass-2.*lnsr-math.log(math.pi)\
                -numpy.log(kappa)\
                +numpy.log(1.+numpy.tanh(lz/self._lo))\
                -kappa*jr*numpy.exp(-2.*lnsr)
            lnfsz= numpy.log(nu)-math.log(2.*math.pi)\
                -2.*lnsz-nu*jz*numpy.exp(-2.*lnsz)
            out= lnfsr+lnfsz+funcTerm
            if isinstance(lz,numpy.ndarray):
                out[numpy.isnan(out)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
                if self._cutcounter: out[(lz < 0.)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
            elif numpy.isnan(out): return -numpy.finfo(numpy.dtype(numpy.float64)).max
            return out
        else:
            srm2= numpy.exp(-2.*lnsr)
            fsr= Omega*numpy.exp(lnsurfmass)*srm2/math.pi/kappa\
                *(1.+numpy.tanh(lz/self._lo))\
                *numpy.exp(-kappa*jr*srm2)
            szm2= numpy.exp(-2.*lnsz)
            fsz= nu/2./math.pi*szm2*numpy.exp(-nu*jz*szm2)
            out= fsr*fsz*funcFactor
            if isinstance(lz,numpy.ndarray):
                out[numpy.isnan(out)]= 0.
                if self._cutcounter: out[(lz < 0.)]= 0.
            elif numpy.isnan(out): return 0.
            return out

    def estimate_hr(self,R,z=0.,nR=11,dR=2./3.,**kwargs):
        """
        NAME:
           estimate_hr
        PURPOSE:
           estimate the exponential scale length at R
        INPUT:
           R - Galactocentric radius
           z= height (default: 0 pc)
           nR= number of Rs to use to estimate
           dR- range in R to use
           Rmax=m minimum R to use
           surfacemass kwargs
        OUTPUT:
           estimated hR
        HISTORY:
           2012-09-11 - Written - Bovy (IAS)
        """
        Rs= numpy.linspace(R-dR/2.,R+dR/2.,nR)
        if z is None:
            sf= numpy.array([self.surfacemass_z(r) for r in Rs])
        else:
            sf= numpy.array([self.surfacemass(r,z,**kwargs) for r in Rs])
        indx= (sf != 0.)
        Rs= Rs[indx]
        sf= sf[indx]
        nR= numpy.sum(indx)
        lsf= numpy.log(sf)
        #Fit
        #Put the dat in the appropriate arrays and matrices
        Y= lsf
        A= numpy.ones((nR,2))
        A[:,1]= Rs
        #Now compute the best fit and the uncertainties
        bestfit= numpy.dot(A.T,Y.T)
        bestfitvar= numpy.dot(A.T,A)
        bestfitvar= linalg.inv(bestfitvar)
        bestfit= numpy.dot(bestfitvar,bestfit)
        return -1./bestfit[1]

    def estimate_hz(self,R,nz=11,zmin=0.1,zmax=0.3,**kwargs):
        """
        NAME:
           estimate_hz
        PURPOSE:
           estimate the exponential scale height at R
        INPUT:
           R - Galactocentric radius
           nz= number of zs to use to estimate
           zmin=m minimum z to use
           zmax=m minimum z to use
           surfacemass kwargs
        OUTPUT:
           estimated hz
        HISTORY:
           2012-08-30 - Written - Bovy (IAS)
        """
        zs= numpy.linspace(zmin,zmax,nz)
        sf= numpy.array([self.surfacemass(R,z,**kwargs) for z in zs])
        indx= (sf != 0.)
        zs= zs[indx]
        sf= sf[indx]
        nz= numpy.sum(indx)
        lsf= numpy.log(sf)
        #Fit
        #Put the dat in the appropriate arrays and matrices
        Y= lsf
        A= numpy.ones((nz,2))
        A[:,1]= zs
        #Now compute the best fit and the uncertainties
        bestfit= numpy.dot(A.T,Y.T)
        bestfitvar= numpy.dot(A.T,A)
        bestfitvar= linalg.inv(bestfitvar)
        bestfit= numpy.dot(bestfitvar,bestfit)
        return -1./bestfit[1]

    def surfacemass_z(self,R,nz=7,zmax=1.,**kwargs):
        """
        NAME:
           surfacemass_z
        PURPOSE:
           calculate the vertically-integrated surface density
        INPUT:
           R - Galactocentric radius
           nz= number of zs to use to estimate
           zmax=m minimum z to use
           surfacemass kwargs
        OUTPUT:
           \Sigma(R)
        HISTORY:
           2012-08-30 - Written - Bovy (IAS)
        """
        zs= numpy.linspace(0.,zmax,nz)
        sf= numpy.array([self.surfacemass(R,z,**kwargs) for z in zs])
        lsf= numpy.log(sf)
        #Interpolate
        lsfInterp= interpolate.UnivariateSpline(zs,
                                                lsf,
                                                k=3)
        #Integrate
        return 2.*integrate.quad((lambda x: numpy.exp(lsfInterp(x))),
                                 0.,1.)[0]

    def vmomentsurfacemass(self,R,z,n,m,o,nsigma=None,mc=False,nmc=10000,
                           _returnmc=False,_vrs=None,_vts=None,_vzs=None,
                           _rawgausssamples=False,
                           gl=False,ngl=_DEFAULTNGL,_returngl=False,_glqeval=None,
                           gh=False,
                           ghgl=False,
                           **kwargs):
        """
        NAME:
           vmomentsurfacemass
        PURPOSE:
           calculate the an arbitrary moment of the velocity distribution 
           at R times the surfacmass
        INPUT:
           R - radius at which to calculate the moment(/ro)
           n - vR^n
           m - vT^m
           o - vz^o
        OPTIONAL INPUT:
           nsigma - number of sigma to integrate the velocities over (when doing explicit numerical integral)
           mc= if True, calculate using Monte Carlo integration
           nmc= if mc, use nmc samples
        OUTPUT:
           <vR^n vT^m  x surface-mass> at R
        HISTORY:
           2012-08-06 - Written - Bovy (IAS@MPIA)
        """
        if nsigma == None:
            nsigma= _NSIGMA
        logSigmaR= (self._ro-R)/self._hr
        sigmaR1= self._sr*numpy.exp((self._ro-R)/self._hsr)
        sigmaz1= self._sz*numpy.exp((self._ro-R)/self._hsz)
        thisvc= potential.vcirc(self._pot,R)
        #Use the asymmetric drift equation to estimate va
        gamma= numpy.sqrt(0.5)
        va= sigmaR1**2./2./thisvc\
            *(gamma**2.-1. #Assume close to flat rotation curve, sigphi2/sigR2 =~ 0.5
               +R*(1./self._hr+2./self._hsr))
        if math.fabs(va) > sigmaR1: va = 0.#To avoid craziness near the center
        if gh:
            mvT= thisvc-va
            #Use Gauss-Hermite integration
            ngh= 5
            ghx, ghw= numpy.polynomial.hermite.hermgauss(ngh)
            #Evaluate everywhere
            vRgh= ghx*numpy.sqrt(2.)*sigmaR1
            vTgh= ghx*numpy.sqrt(2.)*sigmaR1*gamma+mvT
            vzgh= ghx*numpy.sqrt(2.)*sigmaz1
            #Tile everything
            vTgh= numpy.tile(vTgh,(ngh,ngh,1)).T
            vRgh= numpy.tile(numpy.reshape(vRgh,(1,ngh)).T,(ngh,1,ngh))
            vzgh= numpy.tile(vzgh,(ngh,ngh,1))
            vTghw= numpy.tile(ghw,(ngh,ngh,1)).T #also tile weights
            vRghw= numpy.tile(numpy.reshape(ghw,(1,ngh)).T,(ngh,1,ngh))
            vzghw= numpy.tile(ghw,(ngh,ngh,1))
            #evaluate
            logqeval= numpy.reshape(self(R+numpy.zeros(ngh*ngh*ngh),
                                         vRgh.flatten(),
                                         vTgh.flatten(),
                                         z+numpy.zeros(ngh*ngh*ngh),
                                         vzgh.flatten(),
                                         log=True),
                                    (ngh,ngh,ngh))
            logqeval-= vRgh**2./2./sigmaR1**2.+vzgh**2./2./sigmaz1**2.\
                +(vTgh-mvT)**2./2./sigmaR1**2./gamma**2.
            return numpy.sum(numpy.exp(logqeval)*vRgh**n*vTgh**m*vzgh**o
                             *vTghw*vRghw*vzghw)*sigmaR1**2.*sigmaz1
        elif ghgl:
            #Use Gauss-Hermite integration for vR and vz, Gauss-Legendre for vT
            ngh= 20
            ngl= 50
            ghx, ghw= numpy.polynomial.hermite.hermgauss(ngh)
            glx, glw= numpy.polynomial.legendre.leggauss(ngl)
            #Evaluate everywhere
            vRgh= ghx*numpy.sqrt(2.)*sigmaR1
            vTgl= 1.5/2.*(glx+1.)
            vzgh= ghx*numpy.sqrt(2.)*sigmaz1
            #Tile everything
            vTgl= numpy.tile(vTgl,(ngh,ngh,1)).T
            vRgh= numpy.tile(numpy.reshape(vRgh,(1,ngh)).T,(ngl,1,ngh))
            vzgh= numpy.tile(vzgh,(ngl,ngh,1))
            vTglw= numpy.tile(glw,(ngh,ngh,1)).T #also tile weights
            vRghw= numpy.tile(numpy.reshape(ghw,(1,ngh)).T,(ngl,1,ngh))
            vzghw= numpy.tile(ghw,(ngl,ngh,1))
            #evaluate
            logqeval= numpy.reshape(self(R+numpy.zeros(ngl*ngh*ngh),
                                         vRgh.flatten(),
                                         vTgl.flatten(),
                                         z+numpy.zeros(ngl*ngh*ngh),
                                         vzgh.flatten(),
                                         log=True),
                                    (ngl,ngh,ngh))
            logqeval-= vRgh**2./2./sigmaR1**2.+vzgh**2./2./sigmaz1**2.
            return numpy.sum(numpy.exp(logqeval)*vRgh**n*vTgl**m*vzgh**o
                             *vTglw*vRghw*vzghw)*sigmaR1*sigmaz1
        elif gl:
            if not _glqeval is None and ngl != _glqeval.shape[0]:
                _glqeval= None
            #Use Gauss-Legendre integration for all
            glx, glw= numpy.polynomial.legendre.leggauss(ngl)
            #Evaluate everywhere
            vRgl= 3./2.*(glx+1.)
            vTgl= 1.5/2.*(glx+1.)
            vzgl= 3./2.*(glx+1.)
            #Tile everything
            vTgl= numpy.tile(vTgl,(ngl,ngl,1)).T
            vRgl= numpy.tile(numpy.reshape(vRgl,(1,ngl)).T,(ngl,1,ngl))
            vzgl= numpy.tile(vzgl,(ngl,ngl,1))
            vTglw= numpy.tile(glw,(ngl,ngl,1)).T #also tile weights
            vRglw= numpy.tile(numpy.reshape(glw,(1,ngl)).T,(ngl,1,ngl))
            vzglw= numpy.tile(glw,(ngl,ngl,1))
            #evaluate
            if _glqeval is None:
                logqeval= numpy.reshape(self(R+numpy.zeros(ngl*ngl*ngl),
                                             vRgl.flatten(),
                                             vTgl.flatten(),
                                             z+numpy.zeros(ngl*ngl*ngl),
                                             vzgl.flatten(),
                                             log=True),
                                        (ngl,ngl,ngl))
            else:
                logqeval= _glqeval
            if _returngl:
                return (numpy.sum(numpy.exp(logqeval)*vRgl**n*vTgl**m*vzgl**o
                                  *vTglw*vRglw*vzglw),
                        logqeval)
            else:
                return numpy.sum(numpy.exp(logqeval)*vRgl**n*vTgl**m*vzgl**o
                                 *vTglw*vRglw*vzglw)
        elif mc:
            mvT= (thisvc-va)/gamma/sigmaR1
            if _vrs is None:
                vrs= numpy.random.normal(size=nmc)
            else:
                vrs= _vrs
            if _vts is None:
                vts= numpy.random.normal(size=nmc)+mvT
            else:
                if _rawgausssamples:
                    vts= _vts+mvT
                else:
                    vts= _vts
            if _vzs is None:
                vzs= numpy.random.normal(size=nmc)
            else:
                vzs= _vzs
            Is= _vmomentsurfaceMCIntegrand(vzs,vrs,vts,numpy.ones(nmc)*R,
                                           numpy.ones(nmc)*z,
                                           self,sigmaR1,gamma,sigmaz1,mvT,
                                           n,m,o)
            if _returnmc:
                if _rawgausssamples:
                    return (numpy.mean(Is)*sigmaR1**(2.+n+m)*gamma**(1.+m)*sigmaz1**(1.+o),
                        vrs,vts-mvT,vzs)
                else:
                    return (numpy.mean(Is)*sigmaR1**(2.+n+m)*gamma**(1.+m)*sigmaz1**(1.+o),
                            vrs,vts,vzs)
            else:
                return numpy.mean(Is)*sigmaR1**(2.+n+m)*gamma**(1.+m)*sigmaz1**(1.+o)
        else:
            return integrate.tplquad(_vmomentsurfaceIntegrand,
                                     1./gamma*(thisvc-va)/sigmaR1-nsigma,
                                     1./gamma*(thisvc-va)/sigmaR1+nsigma,
                                     lambda x: 0., lambda x: nsigma,
                                     lambda x,y: 0., lambda x,y: nsigma,
                                     (R,z,self,sigmaR1,gamma,sigmaz1,n,m,o),
                                     **kwargs)[0]*sigmaR1**(2.+n+m)*gamma**(1.+m)*sigmaz1**(1.+o)
        
    def jmomentsurfacemass(self,R,z,n,m,o,nsigma=None,mc=True,nmc=10000,
                           _returnmc=False,_vrs=None,_vts=None,_vzs=None,
                           **kwargs):
        """
        NAME:
           jmomentsurfacemass
        PURPOSE:
           calculate the an arbitrary moment of an action
           of the velocity distribution 
           at R times the surfacmass
        INPUT:
           R - radius at which to calculate the moment(/ro)
           n - jr^n
           m - lz^m
           o - jz^o
        OPTIONAL INPUT:
           nsigma - number of sigma to integrate the velocities over (when doing explicit numerical integral)
           mc= if True, calculate using Monte Carlo integration
           nmc= if mc, use nmc samples
        OUTPUT:
           <jr^n lz^m jz^o  x surface-mass> at R
        HISTORY:
           2012-08-09 - Written - Bovy (IAS@MPIA)
        """
        if nsigma == None:
            nsigma= _NSIGMA
        logSigmaR= (self._ro-R)/self._hr
        sigmaR1= self._sr*numpy.exp((self._ro-R)/self._hsr)
        sigmaz1= self._sz*numpy.exp((self._ro-R)/self._hsz)
        thisvc= potential.vcirc(self._pot,R)
        #Use the asymmetric drift equation to estimate va
        gamma= numpy.sqrt(0.5)
        va= sigmaR1**2./2./thisvc\
            *(gamma**2.-1. #Assume close to flat rotation curve, sigphi2/sigR2 =~ 0.5
               +R*(1./self._hr+2./self._hsr))
        if math.fabs(va) > sigmaR1: va = 0.#To avoid craziness near the center
        if mc:
            mvT= (thisvc-va)/gamma/sigmaR1
            if _vrs is None:
                vrs= numpy.random.normal(size=nmc)
            else:
                vrs= _vrs
            if _vts is None:
                vts= numpy.random.normal(size=nmc)+mvT
            else:
                vts= _vts
            if _vzs is None:
                vzs= numpy.random.normal(size=nmc)
            else:
                vzs= _vzs
            Is= _jmomentsurfaceMCIntegrand(vzs,vrs,vts,nump.ones(nmc)*R,numpy.ones(nmc)*z,self,sigmaR1,gamma,sigmaz1,mvT,n,m,o)
            if _returnmc:
                return (numpy.mean(Is)*sigmaR1**2.*gamma*sigmaz1,
                        vrs,vts,vzs)
            else:
                return numpy.mean(Is)*sigmaR1**2.*gamma*sigmaz1
        else:
            return integrate.tplquad(_jmomentsurfaceIntegrand,
                                     1./gamma*(thisvc-va)/sigmaR1-nsigma,
                                     1./gamma*(thisvc-va)/sigmaR1+nsigma,
                                     lambda x: 0., lambda x: nsigma,
                                     lambda x,y: 0., lambda x,y: nsigma,
                                     (R,z,self,sigmaR1,gamma,sigmaz1,n,m,o),
                                     **kwargs)[0]*sigmaR1**2.*gamma*sigmaz1
        
    def surfacemass(self,R,z,nsigma=None,mc=False,nmc=10000,
                    gl=True,ngl=_DEFAULTNGL,**kwargs):
        """
        NAME:
           surfacemass
        PURPOSE:
           calculate the surface-mass at R by marginalizing over velocity
        INPUT:
           R - radius at which to calculate the surfacemass density
           z - height at which to calculate the surfacemass density
        OPTIONAL INPUT:
           nsigma - number of sigma to integrate the velocities over
           scipy.integrate.tplquad kwargs epsabs and epsrel
           mc= if True, calculate using Monte Carlo integration
           nmc= if mc, use nmc samples
           gl= if True, calculate using Gauss-Legendre integration
           ngl= if gl, use ngl-th order Gauss-Legendre integration for each dimension
        OUTPUT:
           surface mass at (R,z)
        HISTORY:
           2012-07-26 - Written - Bovy (IAS@MPIA)
        """
        return self.vmomentsurfacemass(R,z,0.,0.,0.,
                                       nsigma=nsigma,mc=mc,nmc=nmc,
                                       gl=gl,ngl=ngl,
                                       **kwargs)
    
    def sigmaR2(self,R,z,nsigma=None,mc=False,nmc=10000,
                gl=True,ngl=_DEFAULTNGL,**kwargs):
        """
        NAME:
           sigmaR2
        PURPOSE:
           calculate sigma_R^2 by marginalizing over velocity
        INPUT:
           R - radius at which to calculate this
           z - height at which to calculate this
        OPTIONAL INPUT:
           nsigma - number of sigma to integrate the velocities over
           scipy.integrate.tplquad kwargs epsabs and epsrel
           mc= if True, calculate using Monte Carlo integration
           nmc= if mc, use nmc samples
           gl= if True, calculate using Gauss-Legendre integration
           ngl= if gl, use ngl-th order Gauss-Legendre integration for each dimension
        OUTPUT:
           sigma_R^2
        HISTORY:
           2012-07-30 - Written - Bovy (IAS@MPIA)
        """
        if mc:
            surfmass, vrs, vts, vzs= self.vmomentsurfacemass(R,z,0.,0.,0.,
                                                             nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=True,
                                                             **kwargs)
            return self.vmomentsurfacemass(R,z,2.,0.,0.,
                                                             nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=False,
                                           _vrs=vrs,_vts=vts,_vzs=vzs,
                                                             **kwargs)/surfmass
        elif gl:
            surfmass, glqeval= self.vmomentsurfacemass(R,z,0.,0.,0.,
                                                       gl=gl,ngl=ngl,
                                                       _returngl=True,
                                                       **kwargs)
            return self.vmomentsurfacemass(R,z,2.,0.,0.,
                                           ngl=ngl,gl=gl,
                                           _glqeval=glqeval,
                                           **kwargs)/surfmass
        else:
            return (self.vmomentsurfacemass(R,z,2.,0.,0.,
                                           nsigma=nsigma,mc=mc,nmc=nmc,
                                           **kwargs)/
                    self.vmomentsurfacemass(R,z,0.,0.,0.,
                                            nsigma=nsigma,mc=mc,nmc=nmc,
                                            **kwargs))
        
    def sigmaRz(self,R,z,nsigma=None,mc=False,nmc=10000,
                gl=True,ngl=_DEFAULTNGL,**kwargs):
        """
        NAME:
           sigmaRz
        PURPOSE:
           calculate sigma_RZ^2 by marginalizing over velocity
        INPUT:
           R - radius at which to calculate this
           z - height at which to calculate this
        OPTIONAL INPUT:
           nsigma - number of sigma to integrate the velocities over
           scipy.integrate.tplquad kwargs epsabs and epsrel
           mc= if True, calculate using Monte Carlo integration
           nmc= if mc, use nmc samples
           gl= if True, calculate using Gauss-Legendre integration
           ngl= if gl, use ngl-th order Gauss-Legendre integration for each dimension
        OUTPUT:
           sigma_Rz^2
        HISTORY:
           2012-07-30 - Written - Bovy (IAS@MPIA)
        """
        if mc:
            surfmass, vrs, vts, vzs= self.vmomentsurfacemass(R,z,0.,0.,0.,
                                                             nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=True,
                                                             **kwargs)
            return self.vmomentsurfacemass(R,z,1.,0.,1.,
                                                             nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=False,
                                           _vrs=vrs,_vts=vts,_vzs=vzs,
                                                             **kwargs)/surfmass
        elif gl:
            surfmass, glqeval= self.vmomentsurfacemass(R,z,0.,0.,0.,
                                                       gl=gl,ngl=ngl,
                                                       _returngl=True,
                                                       **kwargs)
            return self.vmomentsurfacemass(R,z,1.,0.,1.,
                                           ngl=ngl,gl=gl,
                                           _glqeval=glqeval,
                                           **kwargs)/surfmass
        else:
            return (self.vmomentsurfacemass(R,z,1.,0.,1.,
                                           nsigma=nsigma,mc=mc,nmc=nmc,
                                           **kwargs)/
                    self.vmomentsurfacemass(R,z,0.,0.,0.,
                                            nsigma=nsigma,mc=mc,nmc=nmc,
                                            **kwargs))
        
    def sigmaz2(self,R,z,nsigma=None,mc=False,nmc=10000,
                gl=True,ngl=_DEFAULTNGL,**kwargs):
        """
        NAME:
           sigmaz2
        PURPOSE:
           calculate sigma_z^2 by marginalizing over velocity
        INPUT:
           R - radius at which to calculate this
           z - height at which to calculate this
        OPTIONAL INPUT:
           nsigma - number of sigma to integrate the velocities over
           scipy.integrate.tplquad kwargs epsabs and epsrel
           mc= if True, calculate using Monte Carlo integration
           nmc= if mc, use nmc samples
           gl= if True, calculate using Gauss-Legendre integration
           ngl= if gl, use ngl-th order Gauss-Legendre integration for each dimension
        OUTPUT:
           sigma_z^2
        HISTORY:
           2012-07-30 - Written - Bovy (IAS@MPIA)
        """
        if mc:
            surfmass, vrs, vts, vzs= self.vmomentsurfacemass(R,z,0.,0.,0.,
                                                             nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=True,
                                                             **kwargs)
            return self.vmomentsurfacemass(R,z,0.,0.,2.,
                                           nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=False,
                                           _vrs=vrs,_vts=vts,_vzs=vzs,
                                                             **kwargs)/surfmass
        elif gl:
            surfmass, glqeval= self.vmomentsurfacemass(R,z,0.,0.,0.,
                                                       gl=gl,ngl=ngl,
                                                       _returngl=True,
                                                       **kwargs)
            return self.vmomentsurfacemass(R,z,0.,0.,2.,
                                           ngl=ngl,gl=gl,
                                           _glqeval=glqeval,
                                           **kwargs)/surfmass
        else:
            return (self.vmomentsurfacemass(R,z,0.,0.,2.,
                                           nsigma=nsigma,mc=mc,nmc=nmc,
                                           **kwargs)/
                    self.vmomentsurfacemass(R,z,0.,0.,0.,
                                            nsigma=nsigma,mc=mc,nmc=nmc,
                                            **kwargs))
        
    def meanvT(self,R,z,nsigma=None,mc=False,nmc=10000,
                gl=True,ngl=_DEFAULTNGL,**kwargs):
        """
        NAME:
           meanvT
        PURPOSE:
           calculate sigma_R^2 by marginalizing over velocity
        INPUT:
           R - radius at which to calculate this
           z - height at which to calculate this
        OPTIONAL INPUT:
           nsigma - number of sigma to integrate the velocities over
           scipy.integrate.tplquad kwargs epsabs and epsrel
           mc= if True, calculate using Monte Carlo integration
           nmc= if mc, use nmc samples
           gl= if True, calculate using Gauss-Legendre integration
           ngl= if gl, use ngl-th order Gauss-Legendre integration for each dimension
        OUTPUT:
           meanvT
        HISTORY:
           2012-07-30 - Written - Bovy (IAS@MPIA)
        """
        if mc:
            surfmass, vrs, vts, vzs= self.vmomentsurfacemass(R,z,0.,0.,0.,
                                                             nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=True,
                                                             **kwargs)
            return self.vmomentsurfacemass(R,z,0.,1.,0.,
                                                             nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=False,
                                           _vrs=vrs,_vts=vts,_vzs=vzs,
                                                             **kwargs)/surfmass
        elif gl:
            surfmass, glqeval= self.vmomentsurfacemass(R,z,0.,0.,0.,
                                                       gl=gl,ngl=ngl,
                                                       _returngl=True,
                                                       **kwargs)
            return self.vmomentsurfacemass(R,z,0.,1.,0.,
                                           ngl=ngl,gl=gl,
                                           _glqeval=glqeval,
                                           **kwargs)/surfmass
        else:
            return (self.vmomentsurfacemass(R,z,0.,1.,0.,
                                           nsigma=nsigma,mc=mc,nmc=nmc,
                                           **kwargs)/
                    self.vmomentsurfacemass(R,z,0.,0.,0.,
                                            nsigma=nsigma,mc=mc,nmc=nmc,
                                            **kwargs))
        
    def sigmaT2(self,R,z,nsigma=None,mc=False,nmc=10000,
                gl=True,ngl=_DEFAULTNGL,**kwargs):
        """
        NAME:
           sigmaT2
        PURPOSE:
           calculate sigma_T^2 by marginalizing over velocity
        INPUT:
           R - radius at which to calculate this
           z - height at which to calculate this
        OPTIONAL INPUT:
           nsigma - number of sigma to integrate the velocities over
           scipy.integrate.tplquad kwargs epsabs and epsrel
           mc= if True, calculate using Monte Carlo integration
           nmc= if mc, use nmc samples
           gl= if True, calculate using Gauss-Legendre integration
           ngl= if gl, use ngl-th order Gauss-Legendre integration for each dimension
        OUTPUT:
           sigma_T^2
        HISTORY:
           2012-07-30 - Written - Bovy (IAS@MPIA)
        """
        if mc:
            surfmass, vrs, vts, vzs= self.vmomentsurfacemass(R,z,0.,0.,0.,
                                                             nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=True,
                                                             **kwargs)
            mvt= self.vmomentsurfacemass(R,z,0.,1.,0.,
                                                             nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=False,
                                           _vrs=vrs,_vts=vts,_vzs=vzs,
                                                             **kwargs)/surfmass
            return self.vmomentsurfacemass(R,z,0.,2.,0.,
                                           nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=False,
                                           _vrs=vrs,_vts=vts,_vzs=vzs,
                                           **kwargs)/surfmass\
                                           -mvt**2.
        elif gl:
            surfmass, glqeval= self.vmomentsurfacemass(R,z,0.,0.,0.,
                                                       gl=gl,ngl=ngl,
                                                       _returngl=True,
                                                       **kwargs)
            mvt= self.vmomentsurfacemass(R,z,0.,1.,0.,
                                         ngl=ngl,gl=gl,
                                         _glqeval=glqeval,
                                         **kwargs)/surfmass
            return self.vmomentsurfacemass(R,z,0.,2.,0.,
                                           ngl=ngl,gl=gl,
                                           _glqeval=glqeval,
                                           **kwargs)/surfmass-mvt**2.

        else:
            surfmass= self.vmomentsurfacemass(R,z,0.,0.,0.,
                                              nsigma=nsigma,mc=mc,nmc=nmc,
                                              **kwargs)
            return (self.vmomentsurfacemass(R,z,0.,2.,0.,
                                           nsigma=nsigma,mc=mc,nmc=nmc,
                                           **kwargs)/surfmass\
                        -(self.vmomentsurfacemass(R,z,0.,2.,0.,
                                           nsigma=nsigma,mc=mc,nmc=nmc,
                                           **kwargs)/surfmass)**2.)

    def meanjr(self,R,z,nsigma=None,mc=True,nmc=10000,**kwargs):
        """
        NAME:
           meanjr
        PURPOSE:
           calculate the mean radial action by marginalizing over velocity
        INPUT:
           R - radius at which to calculate this
           z - height at which to calculate this
        OPTIONAL INPUT:
           nsigma - number of sigma to integrate the velocities over
           scipy.integrate.tplquad kwargs epsabs and epsrel
           mc= if True, calculate using Monte Carlo integration
           nmc= if mc, use nmc samples
        OUTPUT:
           meanjr
        HISTORY:
           2012-08-09 - Written - Bovy (IAS@MPIA)
        """
        if mc:
            surfmass, vrs, vts, vzs= self.vmomentsurfacemass(R,z,0.,0.,0.,
                                                             nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=True,
                                                             **kwargs)
            return self.jmomentsurfacemass(R,z,1.,0.,0.,
                                           nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=False,
                                           _vrs=vrs,_vts=vts,_vzs=vzs,
                                                             **kwargs)/surfmass
        else:
            return (self.jmomentsurfacemass(R,z,1.,0.,0.,
                                           nsigma=nsigma,mc=mc,nmc=nmc,
                                           **kwargs)/
                    self.vmomentsurfacemass(R,z,0.,0.,0.,
                                            nsigma=nsigma,mc=mc,nmc=nmc,
                                            **kwargs))
        
    def meanlz(self,R,z,nsigma=None,mc=True,nmc=10000,**kwargs):
        """
        NAME:
           meanlz
        PURPOSE:
           calculate the mean angular momemtum by marginalizing over velocity
        INPUT:
           R - radius at which to calculate this
           z - height at which to calculate this
        OPTIONAL INPUT:
           nsigma - number of sigma to integrate the velocities over
           scipy.integrate.tplquad kwargs epsabs and epsrel
           mc= if True, calculate using Monte Carlo integration
           nmc= if mc, use nmc samples
        OUTPUT:
           meanlz
        HISTORY:
           2012-08-09 - Written - Bovy (IAS@MPIA)
        """
        if mc:
            surfmass, vrs, vts, vzs= self.vmomentsurfacemass(R,z,0.,0.,0.,
                                                             nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=True,
                                                             **kwargs)
            return self.jmomentsurfacemass(R,z,0.,1.,0.,
                                           nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=False,
                                           _vrs=vrs,_vts=vts,_vzs=vzs,
                                                             **kwargs)/surfmass
        else:
            return (self.jmomentsurfacemass(R,z,0.,1.,0.,
                                           nsigma=nsigma,mc=mc,nmc=nmc,
                                           **kwargs)/
                    self.vmomentsurfacemass(R,z,0.,0.,0.,
                                            nsigma=nsigma,mc=mc,nmc=nmc,
                                            **kwargs))
        
    def meanjz(self,R,z,nsigma=None,mc=True,nmc=10000,**kwargs):
        """
        NAME:
           meanjz
        PURPOSE:
           calculate the mean vertical action by marginalizing over velocity
        INPUT:
           R - radius at which to calculate this
           z - height at which to calculate this
        OPTIONAL INPUT:
           nsigma - number of sigma to integrate the velocities over
           scipy.integrate.tplquad kwargs epsabs and epsrel
           mc= if True, calculate using Monte Carlo integration
           nmc= if mc, use nmc samples
        OUTPUT:
           meanjz
        HISTORY:
           2012-08-09 - Written - Bovy (IAS@MPIA)
        """
        if mc:
            surfmass, vrs, vts, vzs= self.vmomentsurfacemass(R,z,0.,0.,0.,
                                                             nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=True,
                                                             **kwargs)
            return self.jmomentsurfacemass(R,z,0.,0.,1.,
                                           nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=False,
                                           _vrs=vrs,_vts=vts,_vzs=vzs,
                                                             **kwargs)/surfmass
        else:
            return (self.jmomentsurfacemass(R,z,0.,0.,1.,
                                           nsigma=nsigma,mc=mc,nmc=nmc,
                                           **kwargs)/
                    self.vmomentsurfacemass(R,z,0.,0.,0.,
                                            nsigma=nsigma,mc=mc,nmc=nmc,
                                            **kwargs))
        
    def _calc_epifreq(self,r):
        """
        NAME:
           _calc_epifreq
        PURPOSE:
           calculate the epicycle frequency at r
        INPUT:
           r - radius
        OUTPUT:
           kappa
        HISTORY:
           2012-07-25 - Written - Bovy (IAS@MPIA)
        NOTE:
           takes about 0.1 ms for a Miyamoto-Nagai potential
        """
        return potential.epifreq(self._pot,r)

    def _calc_verticalfreq(self,r):
        """
        NAME:
           _calc_verticalfreq
        PURPOSE:
           calculate the vertical frequency at r
        INPUT:
           r - radius
        OUTPUT:
           nu
        HISTORY:
           2012-07-25 - Written - Bovy (IAS@MPIA)
        NOTE:
           takes about 0.05 ms for a Miyamoto-Nagai potential
        """
        return potential.verticalfreq(self._pot,r)

    def rg(self,lz):
        """
        NAME:
           rg
        PURPOSE:
           calculate the radius of a circular orbit of Lz
        INPUT:
           lz - Angular momentum
        OUTPUT:
           radius
        HISTORY:
           2012-07-25 - Written - Bovy (IAS@MPIA)
        NOTE:
           seems to take about ~0.5 ms for a Miyamoto-Nagai potential; 
           ~0.75 ms for a MWPotential
           about the same with or without interpolation of the rotation curve

           Not sure what to do about negative lz...
        """
        if isinstance(lz,numpy.ndarray):
            indx= (lz > self._precomputergLzmax)*(lz < self._precomputergLzmin)
            indxc= True-indx
            out= numpy.empty(lz.shape)
            out[indxc]= self._rgInterp(lz[indxc])
            out[indx]= numpy.array([potential.rl(self._pot,lz[indx][ii]) for ii in range(numpy.sum(indx))])
            return out
        else:
            if lz > self._precomputergLzmax or lz < self._precomputergLzmin:
                return potential.rl(self._pot,lz)
            return self._rgInterp(lz)

def _surfaceIntegrand(vz,vR,vT,R,z,df,sigmaR1,gamma,sigmaz1):
    """Internal function that is the integrand for the surface mass integration"""
    return df(R,vR*sigmaR1,vT*sigmaR1*gamma,z,vz*sigmaz1)

def _surfaceMCIntegrand(vz,vR,vT,R,z,df,sigmaR1,gamma,sigmaz1,mvT):
    """Internal function that is the integrand for the surface mass integration"""
    return df(R,vR*sigmaR1,vT*sigmaR1*gamma,z,vz*sigmaz1)*numpy.exp(vR**2./2.+(vT-mvT)**2./2.+vz**2./2.)

def _vmomentsurfaceIntegrand(vz,vR,vT,R,z,df,sigmaR1,gamma,sigmaz1,n,m,o):
    """Internal function that is the integrand for the vmomentsurface mass integration"""
    return vR**n*vT**m*vz**o*df(R,vR*sigmaR1,vT*sigmaR1*gamma,z,vz*sigmaz1)

def _vmomentsurfaceMCIntegrand(vz,vR,vT,R,z,df,sigmaR1,gamma,sigmaz1,mvT,n,m,o):
    """Internal function that is the integrand for the vmomentsurface mass integration"""
    return vR**n*vT**m*vz**o*df(R,vR*sigmaR1,vT*sigmaR1*gamma,z,vz*sigmaz1)*numpy.exp(vR**2./2.+(vT-mvT)**2./2.+vz**2./2.)

def _jmomentsurfaceIntegrand(vz,vR,vT,R,z,df,sigmaR1,gamma,sigmaz1,n,m,o):
    """Internal function that is the integrand for the vmomentsurface mass integration"""
    return df(R,vR*sigmaR1,vT*sigmaR1*gamma,z,vz*sigmaz1,
              func= (lambda x,y,z: x**n*y**m*z**o))

def _jmomentsurfaceMCIntegrand(vz,vR,vT,R,z,df,sigmaR1,gamma,sigmaz1,mvT,n,m,o):
    """Internal function that is the integrand for the vmomentsurface mass integration"""
    return df(R,vR*sigmaR1,vT*sigmaR1*gamma,z,vz*sigmaz1,
              func=(lambda x,y,z: x**n*y**m*z**o))\
              *numpy.exp(vR**2./2.+(vT-mvT)**2./2.+vz**2./2.)

def _sigmaR2surfaceIntegrand(vz,vR,vT,R,z,df,sigmaR1,gamma,sigmaz1):
    """Internal function that is the integrand for the sigma-squared times
    surface mass integration"""
    return (vR*sigmaR1)**2.*df(R,vR*sigmaR1,vT*sigmaR1*gamma,z,vz*sigmaz1)

def _sigmaz2surfaceIntegrand(vz,vR,vT,R,z,df,sigmaR1,gamma,sigmaz1):
    """Internal function that is the integrand for the sigma-squared times
    surface mass integration"""
    return (vz*sigmaz1)**2.*df(R,vR*sigmaR1,vT*sigmaR1*gamma,z,vz*sigmaz1)

def _meanvphisurfaceIntegrand(vz,vR,vT,R,z,df,sigmaR1,gamma,sigmaz1):
    """Internal function that is the integrand for the <vT> times
    surface mass integration"""
    return (vT*sigmaR1*gamma)*df(R,vR*sigmaR1,vT*sigmaR1*gamma,
                                     z,vz*sigmaz1)

def _meanvphi2surfaceIntegrand(vz,vR,vT,R,z,df,sigmaR1,gamma,sigmaz1):
    """Internal function that is the integrand for the sigma-squared times
    surface mass integration"""
    return (vT*sigmaR1*gamma)**2.*df(R,vR*sigmaR1,vT*sigmaR1*gamma,
                                     z,vz*sigmaz1)

