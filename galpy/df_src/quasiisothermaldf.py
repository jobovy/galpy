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
            if isinstance(jr,(list,numpy.ndarray)) and len(jr) > 1: jr= jr[0]
            if isinstance(jz,(list,numpy.ndarray)) and len(jz) > 1: jz= jz[0]
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
                                 1./gamma*(thisvc-va)/sigmaR1-nsigma,
                                 1./gamma*(thisvc-va)/sigmaR1+nsigma,
                                 lambda x: 0., lambda x: nsigma,
                                 lambda x,y: 0., lambda x,y: nsigma,
                                 (R,z,self,sigmaR1,gamma,sigmaz1),
                                 **kwargs)[0]*8.*numpy.pi
    
    def sigmaR2surfacemass(self,R,z,nsigma=None,**kwargs):
        """
        NAME:
           sigmaR2surfacemass
        PURPOSE:
           calculate the surface-mass at R x sigma_R^2 
           by marginalizing over velocity
        INPUT:
           R - radius at which to calculate this
           z - height at which to calculate this
        OPTIONAL INPUT:
           nsigma - number of sigma to integrate the velocities over
           scipy.integrate.tplquad kwargs epsabs and epsrel
        OUTPUT:
           surface mass at (R,z) x sigma_R^2
        HISTORY:
           2012-07-30 - Written - Bovy (IAS@MPIA)
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
        return integrate.tplquad(_sigmaR2surfaceIntegrand,
                                 1./gamma*(thisvc-va)/sigmaR1-nsigma,
                                 1./gamma*(thisvc-va)/sigmaR1+nsigma,
                                 lambda x: 0., lambda x: nsigma,
                                 lambda x,y: 0., lambda x,y: nsigma,
                                 (R,z,self,sigmaR1,gamma,sigmaz1),
                                 **kwargs)[0]*8.*numpy.pi
    
    def sigmaz2surfacemass(self,R,z,nsigma=None,**kwargs):
        """
        NAME:
           sigmaz2surfacemass
        PURPOSE:
           calculate the surface-mass at R x sigma_z^2 
           by marginalizing over velocity
        INPUT:
           R - radius at which to calculate this
           z - height at which to calculate this
        OPTIONAL INPUT:
           nsigma - number of sigma to integrate the velocities over
           scipy.integrate.tplquad kwargs epsabs and epsrel
        OUTPUT:
           surface mass at (R,z) x sigma_z^2
        HISTORY:
           2012-07-30 - Written - Bovy (IAS@MPIA)
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
        return integrate.tplquad(_sigmaz2surfaceIntegrand,
                                 1./gamma*(thisvc-va)/sigmaR1-nsigma,
                                 1./gamma*(thisvc-va)/sigmaR1+nsigma,
                                 lambda x: 0., lambda x: nsigma,
                                 lambda x,y: 0., lambda x,y: nsigma,
                                 (R,z,self,sigmaR1,gamma,sigmaz1),
                                 **kwargs)[0]*8.*numpy.pi
    
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
        if lz > self._precomputergLzmax or lz < self._precomputergLzmin:
            return potential.rl(self._pot,lz)
        return self._rgInterp(lz)

def _surfaceIntegrand(vz,vR,vT,R,z,df,sigmaR1,gamma,sigmaz1):
    """Internal function that is the integrand for the surface mass integration"""
    return df(R,vR*sigmaR1,vT*sigmaR1*gamma,z,vz*sigmaz1)

def _sigmaR2surfaceIntegrand(vz,vR,vT,R,z,df,sigmaR1,gamma,sigmaz1):
    """Internal function that is the integrand for the sigma-squared times
    surface mass integration"""
    return (vR*sigmaR1)**2.*df(R,vR*sigmaR1,vT*sigmaR1*gamma,z,vz*sigmaz1)

def _sigmaz2surfaceIntegrand(vz,vR,vT,R,z,df,sigmaR1,gamma,sigmaz1):
    """Internal function that is the integrand for the sigma-squared times
    surface mass integration"""
    return (vz*sigmaz1)**2.*df(R,vR*sigmaR1,vT*sigmaR1*gamma,z,vz*sigmaz1)

