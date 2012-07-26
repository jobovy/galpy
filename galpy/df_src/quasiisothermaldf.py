#A 'Binney' quasi-isothermal DF
import math
import numpy
from scipy import optimize, interpolate, integrate
from galpy import potential
from galpy import actionAngle
_NSIGMA=4
class quasiisothermaldf:
    """Class that represents a 'Binney' quasi-isothermal DF"""
    def __init__(self,hr,sr,sz,hsr,hsz,pot=None,aA=None,
                 _precomputevcirc=True,_precomputevcircrmax=None,
                 _precomputevcircnr=51,
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
           ro= reference radius for surface mass and sigmas
           lo= reference angular momentum below where there are significant numbers of retrograde stars
        OTHER INPUTS:
           _precomputevcirc= if True (default), pre-compute the circular velocity curve
           _precomputevcircrmax= if set, this is the maximum R for which to pre-compute vcirc (default: 5*hr
           _precomputevcircnr if set, number of R to pre-compute vc for (default: 51)
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
        if _precomputevcirc:
            if _precomputevcircrmax is None:
                _precomputevcircrmax= 5*self._hr
            self._precomputevcircrmax= _precomputevcircrmax
            self._precomputevcircnr= _precomputevcircnr
            self._precomputevcircrgrid= numpy.linspace(0.00001,self._precomputevcircrmax,self._precomputevcircnr)
            self._vcircs= numpy.array([potential.vcirc(self._pot,r) for r in self._precomputevcircrgrid])
            #Spline interpolate
            self._vcircInterp= interpolate.InterpolatedUnivariateSpline(self._precomputevcircrgrid,self._vcircs,k=3)
        else:
            self._precomputevcircrmax= 0.
            self._vcircInterp= None
            self._vcircs= None
            self._precomputevcircnr= None
            self._precomputevcircrgrid= None
        self._precomputevcirc= _precomputevcirc
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
        OUTPUT:
           value of DF
        HISTORY:
           2012-07-25 - Written - Bovy (IAS@MPIA)
        NOTE:
           For Miyamoto-Nagai/adiabatic approximation this seems to take 
           about 30 ms / evaluation in the extended Solar neighborhood
           For a MWPotential/adiabatic approximation this takes about 
           50 ms / evaluation in the extended Solar neighborhood
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
            if len(jr) > 1: jr= jr[0]
            if len(jz) > 1: jz= jz[0]
        #First calculate rg
        thisrg= self.rg(lz)
        #Then calculate the epicycle and vertical frequencies
        kappa, nu= self._calc_epifreq(thisrg), self._calc_verticalfreq(thisrg)
        Omega= lz/thisrg/thisrg
        #calculate surface-densities and sigmas
        lnsurfmass= (self._ro-thisrg)/self._hr
        lnsr= self._lnsr+(self._ro-thisrg)/self._hsr
        lnsz= self._lnsz+(self._ro-thisrg)/self._hsz
        #Calculate fsr
        if log:
            lnfsr= numpy.log(Omega)+lnsurfmass-2.*lnsr-math.log(math.pi)\
                -numpy.log(kappa)\
                +numpy.log(1.+numpy.tanh(lz/self._lo))\
                -kappa*jr*numpy.exp(-2.*lnsr)
            lnfsz= numpy.log(nu)-math.log(2.*math.pi)\
                -2.*lnsz-nu*jz*numpy.exp(-2.*lnsz)
            return lnfsr+lnfsz
        else:
            srm2= numpy.exp(-2.*lnsr)
            fsr= Omega*numpy.exp(lnsurfmass)*srm2/math.pi/kappa\
                *(1.+numpy.tanh(lz/self._lo))\
                *numpy.exp(-kappa*jr*srm2)
            szm2= numpy.exp(-2.*lnsz)
            fsz= nu/2./math.pi*szm2*numpy.exp(-nu*jz*szm2)
            return fsr*fsz

    def surfacemass(self,R,z,nsigma=None,**kwargs):
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
        OUTPUT:
           surface mass at (R,z)
        HISTORY:
           2012-07-26 - Written - Bovy (IAS@MPIA)
        """
        if nsigma == None:
            nsigma= _NSIGMA
        logSigmaR= (self._ro-R)/self._hr
        sigmaR1= self._sr*numpy.exp((self._ro-R)/self._hsr)
        sigmaz1= self._sz*numpy.exp((self._ro-R)/self._hsz)
        thisvc= potential.vcirc(self._pot,R)
        #Use the asymmetric drift equation to estimate va
        gamma= 0.5
        va= sigmaR1**2./2./thisvc\
            *(gamma-1. #Assume close to flat rotation curve, sigphi2/sigR2 =~ 0.5
               +R*(1./self._hr+2./self._hsr))
        if math.fabs(va) > sigmaR1: va = 0.#To avoid craziness near the center
        return integrate.tplquad(_surfaceIntegrand,
                                 gamma*(thisvc-va)/sigmaR1-nsigma,
                                 gamma*(thisvc-va)/sigmaR1+nsigma,
                                 lambda x: 0., lambda x: nsigma,
                                 lambda x,y: 0., lambda x,y: nsigma,
                                 (R,z,self,sigmaR1,gamma,sigmaz1),
                                 **kwargs)[0]
    
    def sigma2surfacemass(self,R,romberg=False,nsigma=None,
                                relative=False):
        """
        NAME:
           sigma2surfacemass
        PURPOSE:
           calculate the product sigma_R^2 x surface-mass at R by 
           marginalizing over velocity
        INPUT:
           R - radius at which to calculate the sigma_R^2 x surfacemass 
               density (/ro)
        OPTIONAL INPUT:
           nsigma - number of sigma to integrate the velocities over
        KEYWORDS:
           romberg - if True, use a romberg integrator (default: False)
        OUTPUT:
           sigma_R^2 x surface-mass at R
        HISTORY:
           2010-03-XX - Written - Bovy (NYU)
        """
        if nsigma == None:
            nsigma= _NSIGMA
        logSigmaR= self.targetSurfacemass(R,log=True)
        sigmaR2= self.targetSigma2(R)
        sigmaR1= sc.sqrt(sigmaR2)
        logsigmaR2= sc.log(sigmaR2)
        if relative:
            norm= 1.
        else:
            norm= sc.exp(logSigmaR+logsigmaR2)
        #Use the asymmetric drift equation to estimate va
        va= sigmaR2/2./R**self._beta*(1./self._gamma**2.-1.
                                      -R*self._surfaceSigmaProfile.surfacemassDerivative(R,log=True)
                                      -R*self._surfaceSigmaProfile.sigma2Derivative(R,log=True))
        if math.fabs(va) > sigmaR1: va = 0. #To avoid craziness near the center
        if romberg:
            return bovy_dblquad(_sigma2surfaceIntegrand,
                                self._gamma*(R**self._beta-va)/sigmaR1-nsigma,
                                self._gamma*(R**self._beta-va)/sigmaR1+nsigma,
                                lambda x: 0., lambda x: nsigma,
                                [R,self,logSigmaR,logsigmaR2,sigmaR1,
                                 self._gamma],
                                tol=10.**-8)/sc.pi*norm
        else:
            return integrate.dblquad(_sigma2surfaceIntegrand,
                                     self._gamma*(R**self._beta-va)/sigmaR1-nsigma,
                                     self._gamma*(R**self._beta-va)/sigmaR1+nsigma,
                                     lambda x: 0., lambda x: nsigma,
                                     (R,self,logSigmaR,logsigmaR2,sigmaR1,
                                      self._gamma),
                                     epsrel=_EPSREL)[0]/sc.pi*norm
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
        #Find interval
        rstart= _rgFindStart(math.fabs(lz),#assumes vo=1.
                             self._vcircInterp,math.fabs(lz),
                             self._precomputevcircrmax,
                             self._pot)
        return optimize.brentq(_rgfunc,0.0000001,rstart,
                               args=(self._vcircInterp,math.fabs(lz),
                                     self._precomputevcircrmax,self._pot))
        
def _rgfunc(rg,vcircInterp,lz,rmax,pot):
    """Function that gives rvc-lz"""
    if rg >= rmax:
        thisvcirc= potential.vcirc(pot,rg)
    else:
        thisvcirc= vcircInterp(rg)
    return rg*thisvcirc-lz

def _rgFindStart(rg,vcircInterp,lz,rmax,pot):
    """find a starting interval for rg"""
    rtry= 2.*rg
    while _rgfunc(rtry,vcircInterp,lz,rmax,pot) < 0.:
        rtry*= 2.
    return rtry

def _surfaceIntegrand(vz,vR,vT,R,z,df,sigmaR1,gamma,sigmaz1):
    """Internal function that is the integrand for the surface mass integration"""
    return df(R,vR*sigmaR1,vT*sigmaR1*gamma,z,vz*sigmaz1)

def _sigmar2surfaceIntegrand(vz,vR,vT,R,z,df,sigmaR1,gamma,sigmaz1):
    """Internal function that is the integrand for the sigma-squared times
    surface mass integration"""
    return vR**2.*df(R,vR*sigmaR1,vT*sigmaR1*gamma,z,vz*sigmaz1)

def _sigmaz2surfaceIntegrand(vz,vR,vT,R,z,df,sigmaR1,gamma,sigmaz1):
    """Internal function that is the integrand for the sigma-squared times
    surface mass integration"""
    return vz**2.*df(R,vR*sigmaR1,vT*sigmaR1*gamma,z,vz*sigmaz1)

