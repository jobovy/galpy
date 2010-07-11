###############################################################################
#   diskdf.py: module that interprets (E,Lz) pairs in terms of a 
#              distribution function
#
#   This module contains the following classes:
#
#      diskdf - top-level class that represents a distribution function
#      dehnendf - inherits from diskdf, implements Dehnen's 'new' DF
#      shudf - inherits from diskdf, implements Shu's DF
#      DFcorrection - class that represents corrections to the input Sigma(R)
#                     and sigma_R(R) to get closer to the targets
###############################################################################
_EPSREL=10.**-14.
_NSIGMA= 4.
_INTERPDEGREE= 3
_RMIN=10.**-10.
_CORRECTIONSDIR='./data'
import copy
import os, os.path
import cPickle as pickle
import math as m
import scipy as sc
import scipy.integrate as integrate
import scipy.interpolate as interpolate
from surfaceSigmaProfile import *
from galpy.orbit.Orbit import Orbit
class diskdf:
    """Class that represents a disk DF"""
    def __init__(self,dftype='dehnen',
                 surfaceSigma=expSurfaceSigmaProfile,
                 profileParams=(1./3.,1.0,0.2),
                 correct=False,
                 beta=0.,**kwargs):
        """
        NAME:
           __init__
        PURPOSE:
           Initialize a DF
        INPUT:
           dftype= 'dehnen' or 'corrected-dehnen', 'shu' or 'corrected-shu'
           surfaceSigma - instance or class name of the target 
                      surface density and sigma_R profile 
                      (default: both exponential)
           profileParams - parameters of the surface and sigma_R profile:
                      (xD,xS,Sro) where
                        xD - disk surface mass scalelength / Ro
                        xS - disk velocity dispersion scalelength / Ro
                        Sro - disk velocity dispersion at Ro (/vo)
                        Directly given to the 'surfaceSigmaProfile class, so
                        could be anything that class takes
           beta - power-law index of the rotation curve
           correct - correct the DF (i.e., DFcorrection kwargs are also given)
           + DFcorrection kwargs (except for those already specified)
        OUTPUT:
        HISTORY:
            2010-03-10 - Written - Bovy (NYU)
        """
        self._dftype= dftype
        if isinstance(surfaceSigma,surfaceSigmaProfile):
            self._surfaceSigmaProfile= surfaceSigma
        else:
            self._surfaceSigmaProfile= surfaceSigma(profileParams)
        self._beta= beta
        if correct or kwargs.has_key('corrections') or kwargs.has_key('rmax') or kwargs.has_key('niter') or kwargs.has_key('npoints'):
            self._correct= True
            #Load corrections
            self._corr= DFcorrection(dftype=self.__class__,
                                     surfaceSigmaProfile=self._surfaceSigmaProfile,
                                     beta=beta,**kwargs)
        else:
            self._correct= False
        return None
    
    def __call__(self,*args):
        """
        NAME:
           __call__
        PURPOSE:
           evaluate the distribution function
        INPUT:
           either an orbit instance or E,Lz

           1) Orbit instance:
              a) Orbit instance alone: use vxvv member
              b) Orbit instance + t: call the Orbit instance

           2)
              E - energy (/vo^2)
              L - angular momentun (/ro/vo)
        OUTPUT:
           DF(orbit/E,L)
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        if isinstance(args[0],Orbit):
            if len(args) == 1:
                return self.eval(*vRvTRToEL(args[0].vxvv[1],
                                            args[0].vxvv[2],
                                            args[0].vxvv[0],
                                            self._beta))
            else:
                vxvv= args[0](args[1])
                return self.eval(*vRvTRToEL(vxvv[1],
                                            vxvv[2],
                                            vxvv[0],
                                            self._beta))
        else:
            return self.eval(*args)

    def targetSigma2(self,R,log=False):
        """
        NAME:
           targetSigma2
        PURPOSE:
           evaluate the target Sigma_R^2(R)
        INPUT:
           R - radius at which to evaluate (/ro)
        OUTPUT:
           target Sigma_R^2(R)
           log - if True, return the log (default: False)
        HISTORY:
           2010-03-28 - Written - Bovy (NYU)
        """
        return self._surfaceSigmaProfile.sigma2(R,log=log)
    
    def targetSurfacemass(self,R,log=False):
        """
        NAME:
           targetSurfacemass
        PURPOSE:
           evaluate the target surface mass at R
        INPUT:
           R - radius at which to evaluate
           log - if True, return the log (default: False)
        OUTPUT:
           Sigma(R)
        HISTORY:
           2010-03-28 - Written - Bovy (NYU)
        """
        return self._surfaceSigmaProfile.surfacemass(R,log=log)
        
    def surfacemass(self,R,romberg=False,nsigma=None,relative=False):
        """
        NAME:
           surfacemass
        PURPOSE:
           calculate the surface-mass at R by marginalizing over velocity
        INPUT:
           R - radius at which to calculate the surfacemass density (/ro)
        OPTIONAL INPUT:
           nsigma - number of sigma to integrate the velocities over
        KEYWORDS:
           romberg - if True, use a romberg integrator (default: False)
        OUTPUT:
           surface mass at R
        HISTORY:
           2010-03-XX - Written - Bovy (NYU)
        """
        if nsigma == None:
            nsigma= _NSIGMA
        logSigmaR= self.targetSurfacemass(R,log=True)
        sigmaR2= self.targetSigma2(R)
        sigmaR1= sc.sqrt(sigmaR2)
        logsigmaR2= sc.log(sigmaR2)
        gamma= sc.sqrt(2./(1.+self._beta))
        if relative:
            norm= 1.
        else:
            norm= sc.exp(logSigmaR)
        if romberg:
            return bovy_dblquad(_surfaceIntegrand,
                                gamma*R**self._beta/sigmaR1-nsigma,
                                gamma*R**self._beta/sigmaR1+nsigma,
                                lambda x: 0., lambda x: nsigma,
                                [R,self,logSigmaR,logsigmaR2,sigmaR1,gamma],
                                tol=10.**-8)/sc.pi*norm
        else:
            return integrate.dblquad(_surfaceIntegrand,
                                     gamma*R**self._beta/sigmaR1-nsigma,
                                     gamma*R**self._beta/sigmaR1+nsigma,
                                     lambda x: 0., lambda x: nsigma,
                                     (R,self,logSigmaR,logsigmaR2,sigmaR1,
                                      gamma),
                                     epsrel=_EPSREL)[0]/sc.pi*norm

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
        gamma= sc.sqrt(2./(1.+self._beta))
        if relative:
            norm= 1.
        else:
            norm= sc.exp(logSigmaR+logsigmaR2)
        if romberg:
            return bovy_dblquad(_sigma2surfaceIntegrand,
                                gamma*R**self._beta/sigmaR1-nsigma,
                                gamma*R**self._beta/sigmaR1+nsigma,
                                lambda x: 0., lambda x: nsigma,
                                [R,self,logSigmaR,logsigmaR2,sigmaR1,gamma],
                                tol=10.**-8)/sc.pi*norm
        else:
            return integrate.dblquad(_sigma2surfaceIntegrand,
                                     gamma*R**self._beta/sigmaR1-nsigma,
                                     gamma*R**self._beta/sigmaR1+nsigma,
                                     lambda x: 0., lambda x: nsigma,
                                     (R,self,logSigmaR,logsigmaR2,sigmaR1,
                                      gamma),
                                     epsrel=_EPSREL)[0]/sc.pi*norm

    def sigma2(self,R,romberg=False,nsigma=None):
        """
        NAME:
           sigma2
        PURPOSE:
           calculate sigma_R^2 at R by marginalizing over velocity
        INPUT:
           R - radius at which to calculate sigma_R^2 density (/ro)
        OPTIONAL INPUT:
           nsigma - number of sigma to integrate the velocities over
        KEYWORDS:
           romberg - if True, use a romberg integrator (default: False)
        OUTPUT:
           sigma_R^2 at R
        HISTORY:
           2010-03-XX - Written - Bovy (NYU)
        """
        return self.sigma2surfacemass(R,romberg,nsigma)/self.surfacemass(R,romberg,nsigma)


class dehnendf(diskdf):
    """Dehnen's 'new' df"""
    def __init__(self,surfaceSigma=expSurfaceSigmaProfile,
                 profileParams=(1./3.,1.0,0.2),
                 correct=False,
                 beta=0.,**kwargs):
        """
        NAME:
           __init__
        PURPOSE:
           Initialize a Dehnen 'new' DF
        INPUT:
           surfaceSigma - instance or class name of the target 
                      surface density and sigma_R profile 
                      (default: both exponential)
           profileParams - parameters of the surface and sigma_R profile:
                      (xD,xS,Sro) where
                        xD - disk surface mass scalelength / Ro
                        xS - disk velocity dispersion scalelength / Ro
                        Sro - disk velocity dispersion at Ro (/vo)
                        Directly given to the 'surfaceSigmaProfile class, so
                        could be anything that class takes
           beta - power-law index of the rotation curve
           correct - if True, correct the DF
           + DFcorrection kwargs (except for those already specified)
        OUTPUT:
        HISTORY:
            2010-03-10 - Written - Bovy (NYU)
        """
        return diskdf.__init__(self,surfaceSigma=surfaceSigma,
                               profileParams=profileParams,
                               correct=correct,dftype='dehnen',
                               beta=beta,**kwargs)
        
    def eval(self,E,L,logSigmaR=0.,logsigmaR2=0.):
        """
        NAME:
           eval
        PURPOSE:
           evaluate the distribution function
        INPUT:
           E - energy (/vo^2)
           L - angular momentun (/ro/vo)
       OUTPUT:
           DF(E,L)
        HISTORY:
           2010-03-10 - Written - Bovy (NYU)
           2010-03-28 - Moved to dehnenDF - Bovy (NYU)
        """
        #Calculate Re,LE, OmegaE
        if self._beta == 0.:
            xE= sc.exp(E-.5)
            logOLLE= sc.log(L/xE-1.)
        else: #non-flat rotation curve
            xE= (2.*E/(1.+1./self._beta))**(1./2./self._beta)
            logOLLE= self._beta*sc.log(xE)+sc.log(L/xE-xE**self._beta)
        if self._correct: 
            correction= self._corr.correct(xE,log=True)
        else:
            correction= sc.zeros(2)
        SRE2= self.targetSigma2(xE,log=True)+correction[1]
        return sc.exp(logsigmaR2-SRE2+self.targetSurfacemass(xE,log=True)-logSigmaR+sc.exp(logOLLE-SRE2)+correction[0])


    def sample(self,n=1,rrange=None):
        """
        NAME:
           sample
        PURPOSE:
           sample from this DF
        INPUT:
           n - number of desired sample (specifying this rather than calling 
               this routine n times is more efficient)
           rrange - if you only want samples in this rrange, set this keyword
        OUTPUT:
           list of [[E,Lz],...]
        HISTORY:
           2010-07-10 - Started  - Bovy (NYU)
        """
        #First sample xE
        #Then sample Lz

class shudf(diskdf):
    """Shu's df (1969)"""
    def __init__(self,surfaceSigma=expSurfaceSigmaProfile,
                 profileParams=(1./3.,1.0,0.2),
                 correct=False,
                 beta=0.,**kwargs):
        """
        NAME:
           __init__
        PURPOSE:
           Initialize a Shu DF
        INPUT:
           surfaceSigma - instance or class name of the target 
                      surface density and sigma_R profile 
                      (default: both exponential)
           profileParams - parameters of the surface and sigma_R profile:
                      (xD,xS,Sro) where
                        xD - disk surface mass scalelength / Ro
                        xS - disk velocity dispersion scalelength / Ro
                        Sro - disk velocity dispersion at Ro (/vo)
                        Directly given to the 'surfaceSigmaProfile class, so
                        could be anything that class takes
           beta - power-law index of the rotation curve
           correct - if True, correct the DF
           + DFcorrection kwargs (except for those already specified)
        OUTPUT:
        HISTORY:
            2010-05-09 - Written - Bovy (NYU)
        """
        return distkdf.__init__(self,surfaceSigma=surfaceSigma,
                                profileParams=profileParams,
                                correct=correct,dftype='shu',
                                beta=beta,**kwargs)
    
    def eval(self,E,L,logSigmaR=0.,logsigmaR2=0.):
        """
        NAME:
           eval
        PURPOSE:
           evaluate the distribution function
        INPUT:
           E - energy (/vo^2)
           L - angular momentun (/ro/vo)
       OUTPUT:
           DF(E,L)
        HISTORY:
           2010-05-09 - Written - Bovy (NYU)
        """
        #Calculate RL,LL, OmegaL
        if self._beta == 0.:
            xL= L
            logECLE= sc.log(-sc.log(xL)-0.5+E)
        else: #non-flat rotation curve
            xL= L**(1./(self._beta+1.))
            logECLE= sc.log(-0.5*(1./self._beta+1.)*xL**(2.*self._beta)+E)
        if xL < 0.: #We must remove counter-rotating mass
            return 0.
        if self._correct: 
            correction= self._corr.correct(xL,log=True)
        else:
            correction= sc.zeros(2)
        SRE2= self.targetSigma2(xL,log=True)+correction[1]
        return sc.exp(logsigmaR2-SRE2+self.targetSurfacemass(xL,log=True)-logSigmaR-sc.exp(logECLE-SRE2)+correction[0])

def _surfaceIntegrand(vR,vT,R,df,logSigmaR,logsigmaR2,sigmaR1,gamma):
    """Internal function that is the integrand for the surface mass integration"""
    E,L= _vRpvTpRToEL(vR,vT,R,df._beta,sigmaR1,gamma)
    return df.eval(E,L,logSigmaR,logsigmaR2)

def _sigma2surfaceIntegrand(vR,vT,R,df,logSigmaR,logsigmaR2,sigmaR1,gamma):
    """Internal function that is the integrand for the sigma-squared times
    surface mass integration"""
    E,L= _vRpvTpRToEL(vR,vT,R,df._beta,sigmaR1,gamma)
    return vR**2.*df.eval(E,L,logSigmaR,logsigmaR2)

def _vRpvTpRToEL(vR,vT,R,beta,sigmaR1,gamma):
    """Internal function that calculates E and L given velocities normalized by the velocity dispersion"""
    vR*= sigmaR1
    vT*= sigmaR1/gamma
    return vRvTRToEL(vR,vT,R,beta)

def _oned_intFunc(x,twodfunc,gfun,hfun,tol,args):
    """Internal function for bovy_dblquad"""
    thisargs= copy.deepcopy(args)
    thisargs.insert(0,x)
    return integrate.romberg(twodfunc,gfun(x),hfun(x),args=thisargs,tol=tol)

def bovy_dblquad(func, a, b, gfun, hfun, args=(), tol=1.48e-08):
    """
    NAME:
       bovy_dblquad
    PURPOSE:
       like scipy.integrate's dblquad, but using Romberg integration for the one-d integrals and using tol
    INPUT:
       same as scipy.integrate.dblquad except for tol and epsrel,epsabs
    OUTPUT:
       value
    HISTORY:
       2010-03-11 - Written - Bpvy (NYU)
    """
    return integrate.romberg(_oned_intFunc,a,b,args=(func,gfun,hfun,tol,args),tol=tol)


class DFcorrection:
    """Class that contains the corrections necessary to reach
    exponential profiles"""
    def __init__(self,**kwargs):
        """
        NAME:
           __init__
        PURPOSE:
           initialize the corrections: set them, load them, or calculate
           and save them
        OPTIONAL INPUTS:
           corrections - if Set, these are the corrections and they should
                         be used as such
           npoints - number of points from 0 to Rmax
           rmax - correct up to this radius (/ro) (default: 5)
           savedir - save the corrections in this directory
           surfaceSigmaProfile - target surfacemass and sigma_R^2 instance
           beta - power-law index of the rotation curve (when calculating)
           dftype - classname of the DF
           niter - number of iterations to perform to calculate the corrections
           interp1d_kind - 'kind' keyword to give to interp1d
        OUTPUT:
        HISTORY:
           2010-03-10 - Written - Bovy (NYU)
        """
        if not kwargs.has_key('surfaceSigmaProfile'):
            raise DFcorrectionError("surfaceSigmaProfile not given")
        else:
            self._surfaceSigmaProfile= kwargs['surfaceSigmaProfile']
        if not kwargs.has_key('rmax'):
            self._rmax= 5.
        else:
            self._rmax= kwargs['rmax']
        if not kwargs.has_key('niter'):
            self._niter= 1
        else:
            self._niter= kwargs['niter']
        if not kwargs.has_key('npoints'):
            if kwargs.has_key('corrections'):
                self._npoints= kwargs['corrections'].shape[0]
            else:
                self._npoints= 151
        else:
            self._npoints= kwargs['npoints']
        if kwargs.has_key('dftype'):
            self._dftype= kwargs['dftype']
        else:
            self._dftype= dehnenDF
        if kwargs.has_key('beta'):
            self._beta= kwargs['beta']
        else:
            self._beta= 0.
        self._rs= sc.linspace(_RMIN,self._rmax,self._npoints)
        if kwargs.has_key('interp1d_kind'):
            self._interp1d_kind= kwargs['interp1d_kind']
        else:
            self._interp1d_kind= _INTERPDEGREE
        if kwargs.has_key('corrections'):
            self._corrections= kwargs['corrections']
            if not len(self._corrections) == self._npoints:
                raise DFcorrectionError("Number of corrections has to be equal to the number of points npoints")
        else:
            if kwargs.has_key('savedir'):
                self._savedir= kwargs['savedir']
            else:
                self._savedir= _CORRECTIONSDIR
            self._savefilename= self._createSavefilename(self._niter)
            if os.path.exists(self._savefilename):
                savefile= open(self._savefilename,'rb')
                self._corrections= sc.array(pickle.load(savefile))
                savefile.close()
            else: #Calculate the corrections
                self._corrections= self._calc_corrections()
        #Interpolation; smoothly go to zero
        interpRs= sc.append(self._rs,2.*self._rmax)
        self._surfaceInterpolate= interpolate.interp1d(interpRs,
                                                       sc.log(sc.append(self._corrections[:,0],1.)),
                                                       kind=self._interp1d_kind,
                                                       bounds_error=False,
                                                       fill_value=0.)
        self._sigma2Interpolate= interpolate.interp1d(interpRs,
                                                      sc.log(sc.append(self._corrections[:,1],1.)),
                                                      kind=self._interp1d_kind,
                                                      bounds_error=False,
                                                      fill_value=0.)
        #Interpolation for R < _RMIN
        surfaceInterpolateSmallR= interpolate.UnivariateSpline(interpRs[0:_INTERPDEGREE+2],sc.log(self._corrections[0:_INTERPDEGREE+2,0]),k=_INTERPDEGREE)
        self._surfaceDerivSmallR= surfaceInterpolateSmallR.derivatives(interpRs[0])[1]
        sigma2InterpolateSmallR= interpolate.UnivariateSpline(interpRs[0:_INTERPDEGREE+2],sc.log(self._corrections[0:_INTERPDEGREE+2,1]),k=_INTERPDEGREE)
        self._sigma2DerivSmallR= sigma2InterpolateSmallR.derivatives(interpRs[0])[1]
        return None

    def _createSavefilename(self,niter):
        #Form surfaceSigmaProfile string
        sspFormat= self._surfaceSigmaProfile.formatStringParams()
        sspString= ''
        for format in sspFormat:
            sspString+= format+'_'
        return os.path.join(self._savedir,'dfcorrection_'+
                            self._dftype.__name__+'_'+
                            self._surfaceSigmaProfile.__class__.__name__+'_'+
                            sspString % self._surfaceSigmaProfile.outputParams()+
                            '%4.2f_%i_%4.2f_%i.sav'
                            % (self._beta,self._npoints,self._rmax,niter))

    def correct(self,R,log=False):
        """
        NAME:
           correct
        PURPOSE:
           calculate the correction in Sigma and sigma2 at R
        INPUT:
           R - Galactocentric radius(/ro)
           log - if True, return the log of the correction
        OUTPUT:
           [Sigma correction, sigma2 correction]
        HISTORY:
           2010-03-10 - Written - Bovy (NYU)
        """
        if R < _RMIN:
            out= sc.array([sc.log(self._corrections[0,0])+self._surfaceDerivSmallR*(R-_RMIN),
                           sc.log(self._corrections[0,1])+self._sigma2DerivSmallR*(R-_RMIN)])
        else:
            out= sc.array([self._surfaceInterpolate(R),
                           self._sigma2Interpolate(R)])
        if log:
            return out
        else:
            return sc.exp(out)
            

    def _calc_corrections(self):
        """Internal function that calculates the corrections"""     
        searchIter= self._niter-1
        while searchIter > 0:
            trySavefilename= self._createSavefilename(searchIter)
            if os.path.exists(trySavefilename):
                trySavefile= open(trySavefilename,'rb')
                corrections= sc.array(pickle.load(trySavefile))
                trySavefile.close()
                break
            else:
                searchIter-= 1
        if searchIter == 0:
            corrections= sc.ones((self._npoints,2))
        for ii in range(searchIter,self._niter):
            if ii == 0:
                currentDF= self._dftype(surfaceSigma=self._surfaceSigmaProfile,
                                        beta=self._beta)
            else:
                currentDF= self._dftype(surfaceSigma=self._surfaceSigmaProfile,
                                        beta=self._beta,
                                        corrections=corrections,
                                        npoints=self._npoints,
                                        rmax=self._rmax,
                                        savedir=self._savedir,
                                        interp1d_kind=self._interp1d_kind)
            newcorrections= sc.zeros((self._npoints,2))
            for jj in range(self._npoints):
                thisSurface= currentDF.surfacemass(self._rs[jj])
                newcorrections[jj,0]= currentDF.targetSurfacemass(self._rs[jj])/thisSurface
                newcorrections[jj,1]= currentDF.targetSigma2(self._rs[jj])*thisSurface/currentDF.sigma2surfacemass(self._rs[jj])
                #print jj, newcorrections[jj,:]
            corrections*= newcorrections
        #Save
        savefile= open(self._savefilename,'w')
        picklethis= []
        for arr in list(corrections):
            picklethis.append([float(a) for a in arr])
        pickle.dump(picklethis,savefile)#We pickle a list for platform-independence
        savefile.close()
        return corrections
    
class DFcorrectionError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def vRvTRToEL(vR,vT,R,beta):
    """
    NAME:
       vRvTRToEL
    PURPOSE:
       calculate the energy and angular momentum
    INPUT:
       vR - radial velocity
       vT - tangential velocity
       R - Galactocentric radius
    OUTPUT:
    HISTORY:
       2010-03-10 - Written - Bovy (NYU)
    """
    return (axipotential(R,beta)+0.5*vR**2.+0.5*vT**2.,vT*R)

def axipotential(R,beta=0.):
    """
    NAME:
       axipotential
    PURPOSE:
       return the axisymmetric potential at R/Ro
    INPUT:
       R - Galactocentric radius
       beta - rotation curve power-law
    OUTPUT:
       Pot(R)/vo**2.
    HISTORY:
       2010-03-01 - Written - Bovy (NYU)
    """
    if beta == 0.:
        return m.log(R)
    else: #non-flat rotation curve
        return R**(2.*beta)/2./beta

