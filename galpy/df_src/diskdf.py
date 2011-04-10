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
_MAXD_REJECTLOS= 4.
import copy
import re
import os, os.path
import cPickle as pickle
import math as m
import numpy as nu
import scipy as sc
import scipy.integrate as integrate
import scipy.interpolate as interpolate
from scipy import linalg
from scipy import stats
from scipy import optimize
from surfaceSigmaProfile import *
from galpy.orbit import Orbit
from galpy.util.bovy_ars import bovy_ars
from galpy.potential import PowerSphericalPotential
from galpy.actionAngle import actionAngleFlat, actionAnglePower
from galpy.actionAngle_src.actionAngleFlat import calcRapRperiFromELFlat #HACK
from galpy.actionAngle_src.actionAnglePower import \
    calcRapRperiFromELPower #HACK
#scipy version
try:
    sversion=re.split(r'\.',sc.__version__)
    _SCIPYVERSION=float(sversion[0])+float(sversion[1])/10.
except:
    raise ImportError( "scipy.__version__ not understood, contact galpy developer, send scipy.__version__")
_CORRECTIONSDIR=os.path.join(os.path.dirname(os.path.realpath(__file__)),'data')
_DEGTORAD= m.pi/180.
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
        self._gamma= sc.sqrt(2./(1.+self._beta))
        if correct or kwargs.has_key('corrections') or kwargs.has_key('rmax') or kwargs.has_key('niter') or kwargs.has_key('npoints'):
            self._correct= True
            #Load corrections
            self._corr= DFcorrection(dftype=self.__class__,
                                     surfaceSigmaProfile=self._surfaceSigmaProfile,
                                     beta=beta,**kwargs)
        else:
            self._correct= False
        return None
    
    def __call__(self,*args,**kwargs):
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

        KWARGS:

           marginalizeVperp - marginalize over perpendicular velocity (only supported with 1a) above) + nsigma, +scipy.integrate.quad keywords

           marginalizeVlos - marginalize over line-of-sight velocity (only supported with 1a) above) + nsigma, +scipy.integrate.quad keywords

        OUTPUT:

           DF(orbit/E,L)

        HISTORY:

           2010-07-10 - Written - Bovy (NYU)

        """
        if isinstance(args[0],Orbit):
            if len(args) == 1:
                if kwargs.has_key('marginalizeVperp') and \
                        kwargs['marginalizeVperp']:
                    kwargs.pop('marginalizeVperp')
                    return self._call_marginalizevperp(args[0],**kwargs)
                elif kwargs.has_key('marginalizeVlos') and \
                        kwargs['marginalizeVlos']:
                    kwargs.pop('marginalizeVlos')
                    return self._call_marginalizevlos(args[0],**kwargs)
                else:
                    return sc.real(self.eval(*vRvTRToEL(args[0].vxvv[1],
                                                        args[0].vxvv[2],
                                                        args[0].vxvv[0],
                                                        self._beta)))
            else:
                vxvv= args[0](args[1])
                return sc.real(self.eval(*vRvTRToEL(vxvv[1],
                                                    vxvv[2],
                                                    vxvv[0],
                                                    self._beta)))
        else:
            return sc.real(self.eval(*args))

    def _call_marginalizevperp(self,o,**kwargs):
        """Call the DF, marginalizing over perpendicular velocity"""
        #Get d, l, vlos
        d= o.dist(ro=1.,obs=[1.,0.,0.])
        l= o.ll(obs=[1.,0.,0.],ro=1.)*_DEGTORAD
        vlos= o.vlos(ro=1.,vo=1.,obs=[1.,0.,0.,0.,0.,0.])
        R= o.R()
        phi= o.phi()
        #Get local circular velocity, projected onto the los
        vcirc= R**self._beta
        vcirclos= vcirc*m.sin(phi+l)
        #Marginalize
        alphalos= phi+l
        if not kwargs.has_key('nsigma') or (kwargs.has_key('nsigma') and \
                                                kwargs['nsigma'] is None):
            nsigma= _NSIGMA
        else:
            nsigma= kwargs['nsigma']
        if kwargs.has_key('nsigma'): kwargs.pop('nsigma')
        sigmaR2= self.targetSigma2(R)
        sigmaR1= sc.sqrt(sigmaR2)
        #Use the asymmetric drift equation to estimate va
        va= sigmaR2/2./R**self._beta*(1./self._gamma**2.-1.
                                      -R*self._surfaceSigmaProfile.surfacemassDerivative(R,log=True)
                                      -R*self._surfaceSigmaProfile.sigma2Derivative(R,log=True))
        if m.fabs(m.sin(alphalos)) < m.sqrt(1./2.):
            cosalphalos= m.cos(alphalos)
            tanalphalos= m.tan(alphalos)
            return integrate.quad(_marginalizeVperpIntegrandSinAlphaSmall,
                                  -self._gamma*va/sigmaR1-nsigma,
                                  -self._gamma*va/sigmaR1+nsigma,
                                  args=(self,R,cosalphalos,tanalphalos,
                                        vlos-vcirclos,vcirc,
                                        sigmaR1/self._gamma),
                                  **kwargs)[0]/m.fabs(cosalphalos)
        else:
            sinalphalos= m.sin(alphalos)
            cotalphalos= 1./m.tan(alphalos)
            return integrate.quad(_marginalizeVperpIntegrandSinAlphaLarge,
                                  -nsigma,nsigma,
                                  args=(self,R,sinalphalos,cotalphalos,
                                        vlos-vcirclos,vcirc,sigmaR1),
                                  **kwargs)[0]/m.fabs(sinalphalos)
        
    def _call_marginalizevlos(self,o,**kwargs):
        """Call the DF, marginalizing over line-of-sight velocity"""
        #Get d, l, vperp
        d= o.dist(ro=1.,obs=[1.,0.,0.])
        l= o.ll(obs=[1.,0.,0.],ro=1.)*_DEGTORAD
        vperp= o.vll(ro=1.,vo=1.,obs=[1.,0.,0.,0.,0.,0.])
        R= o.R()
        phi= o.phi()
        #Get local circular velocity, projected onto the perpendicular 
        #direction
        vcirc= R**self._beta
        vcircperp= vcirc*m.cos(phi+l)
        #Marginalize
        alphaperp= m.pi/2.+phi+l
        if not kwargs.has_key('nsigma') or (kwargs.has_key('nsigma') and \
                                                kwargs['nsigma'] is None):
            nsigma= _NSIGMA
        else:
            nsigma= kwargs['nsigma']
        if kwargs.has_key('nsigma'): kwargs.pop('nsigma')
        sigmaR2= self.targetSigma2(R)
        sigmaR1= sc.sqrt(sigmaR2)
        #Use the asymmetric drift equation to estimate va
        va= sigmaR2/2./R**self._beta*(1./self._gamma**2.-1.
                                      -R*self._surfaceSigmaProfile.surfacemassDerivative(R,log=True)
                                      -R*self._surfaceSigmaProfile.sigma2Derivative(R,log=True))
        if m.fabs(m.sin(alphaperp)) < m.sqrt(1./2.):
            cosalphaperp= m.cos(alphaperp)
            tanalphaperp= m.tan(alphaperp)
            #we can reuse the VperpIntegrand, since it is just another angle
            return integrate.quad(_marginalizeVperpIntegrandSinAlphaSmall,
                                  -self._gamma*va/sigmaR1-nsigma,
                                  -self._gamma*va/sigmaR1+nsigma,
                                  args=(self,R,cosalphaperp,tanalphaperp,
                                        vperp-vcircperp,vcirc,
                                        sigmaR1/self._gamma),
                                  **kwargs)[0]/m.fabs(cosalphaperp)
        else:
            sinalphaperp= m.sin(alphaperp)
            cotalphaperp= 1./m.tan(alphaperp)
            #we can reuse the VperpIntegrand, since it is just another angle
            return integrate.quad(_marginalizeVperpIntegrandSinAlphaLarge,
                                  -nsigma,nsigma,
                                  args=(self,R,sinalphaperp,cotalphaperp,
                                        vperp-vcircperp,vcirc,sigmaR1),
                                  **kwargs)[0]/m.fabs(sinalphaperp)
        
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
        
    def targetSurfacemassLOS(self,d,l,log=False,deg=True):
        """
        NAME:
           targetSurfacemassLOS
        PURPOSE:
           evaluate the target surface mass along the LOS given l and d
        INPUT:
           d - distance along the line of sight
           l - Galactic longitude (in deg, unless deg=False)
           deg= if False, l is in radians
           log - if True, return the log (default: False)
        OUTPUT:
           Sigma(d,l)
        HISTORY:
           2011-03-23 - Written - Bovy (NYU)
        """
        #Calculate R and phi
        if deg:
            lrad= l*_DEGTORAD
        else:
            lrad= l
        R, phi= _dlToRphi(d,lrad)
        #Evaluate Jacobian
        jac= _jacobian_rphi_dl(d,lrad,R=R,phi=phi)
        if log:
            return self._surfaceSigmaProfile.surfacemass(R,log=log)\
                +m.log(m.fabs(jac))+m.log(R)
            pass
        else:
            return self._surfaceSigmaProfile.surfacemass(R,log=log)\
                *m.fabs(jac)*R
        
    def surfacemassLOS(self,d,l,deg=True,
                       romberg=False,nsigma=None,relative=None):
        """
        NAME:
           surfacemassLOS
        PURPOSE:
           evaluate the surface mass along the LOS given l and d
        INPUT:
           d - distance along the line of sight
           l - Galactic longitude (in deg, unless deg=False)
        OPTIONAL INPUT:
           nsigma - number of sigma to integrate the velocities over
        KEYWORDS:
           romberg - if True, use a romberg integrator (default: False)
           deg= if False, l is in radians
        OUTPUT:
           Sigma(d,l)
        HISTORY:
           2011-03-24 - Written - Bovy (NYU)
        """
        #Calculate R and phi
        if deg:
            lrad= l*_DEGTORAD
        else:
            lrad= l
        R, phi= _dlToRphi(d,lrad)
        #Evaluate Jacobian
        jac= _jacobian_rphi_dl(d,lrad,R=R,phi=phi)
        return self.surfacemass(R,romberg=romberg,nsigma=nsigma,
                                relative=relative)\
                                *m.fabs(jac)*R

    def sampledSurfacemassLOS(self,l,n=1,maxd=None,target=True):
        """
        NAME:
           sampledSurfacemassLOS
        PURPOSE:
           sample a distance along the line of sight
        INPUT:
           l - Galactic longitude (in rad)
           n= number of distances to sample
           maxd= maximum distance to consider (for the rejection sampling)
           target= if True, sample from the 'target' surface mass density,
                   rather than the actual surface mass density (default=True)
        OUTPUT:
           list of samples
        HISTORY:
           2011-03-24 - Written - Bovy (NYU)
        """
        #First calculate where the maximum is
        if l == 0.:
            maxSM= self.targetSurfacemass(0.)
        elif l >= m.pi/2. and l <= 3.*m.pi/2.:
            maxSM= self.targetSurfacemass(1.)
        elif l < m.pi/2. or l > 3.*m.pi/2.:
            if target:
                minR= optimize.fmin_bfgs(lambda x: \
                                             -x*self.targetSurfacemassLOS(x,l,
                                                                          deg=False),
                                         0.,disp=False)[0]
                maxSM= self.targetSurfacemassLOS(minR,l,deg=False)*minR
            else:
                minR= optimize.fmin_bfgs(lambda x: \
                                             -x*self.surfacemassLOS(x,l,
                                                                    deg=False),
                                         0.,disp=False)[0]
                maxSM= self.surfacemassLOS(minR,l,deg=False)*minR
        #Now rejection-sample
        if maxd is None:
            maxd= _MAXD_REJECTLOS
        out= []
        while len(out) < n:
            #sample
            prop= nu.random.random()*maxd
            if target:
                surfmassatprop= self.targetSurfacemassLOS(prop,l,deg=False)*prop
            else:
                surfmassatprop= self.surfacemassLOS(prop,l,deg=False)*prop
            if surfmassatprop/maxSM > nu.random.random(): #accept
                out.append(prop)
        return nu.array(out)

    def sampleVRVT(self,R,n=1,nsigma=None,target=True):
        """
        NAME:
           sampleVRVT
        PURPOSE:
           sample a radial and azimuthal velocity at R
        INPUT:
           R - Galactocentric distance
           n= number of distances to sample
           nsigma= number of sigma to rejection-sample on
           target= if True, sample using the 'target' sigma_R
                   rather than the actual sigma_R (default=True)
        OUTPUT:
           list of samples
        BUGS:
           should use the fact that vR and vT separate
        HISTORY:
           2011-03-24 - Written - Bovy (NYU)
        """
        #Determine where the max of the v-distribution is using asymmetric drift
        maxVR= 0.
        maxVT= optimize.brentq(_vtmaxEq,0.,1.,(R,self))
        maxVD= self(Orbit([R,maxVR,maxVT]))
        #Now rejection-sample
        if nsigma == None:
            nsigma= _NSIGMA
        out= []
        if target:
            sigma= m.sqrt(self.targetSigma2(R))
        else:
            sigma= m.sqrt(self.sigma2(R))
        while len(out) < n:
            #sample
            propvR= nu.random.normal()*_NSIGMA*sigma
            propvT= nu.random.normal()*_NSIGMA*sigma/self._gamma+1.
            VDatprop= self(Orbit([R,propvR,propvT]))
            if VDatprop/maxVD > nu.random.random(): #accept
                out.append(sc.array([propvR,propvT]))
        return nu.array(out)

    def sampleLOS(self,los,n=1,deg=True,maxd=None,nsigma=None,
                  target=True):
        """
        NAME:
           sampleLOS
        PURPOSE:
           sample along a given LOS
        INPUT:
           los - line of sight (in deg, unless deg=False)
           n= number of desired samples
           deg= los in degrees? (default=True)
           target= if True, use target surface mass and sigma2 profiles
                   (default=True)
        OUTPUT:
           returns list of Orbits
        BUGS:
           target=False uses target distribution for derivatives (this is a detail)
        HISTORY:
           2011-03-24 - Started  - Bovy (NYU)
        """
        if deg:
            l= los*_DEGTORAD
        else:
            l= los
        out= []
        #sample distances
        ds= self.sampledSurfacemassLOS(l,n=n,maxd=maxd,target=target)
        for ii in range(int(n)):
            #Calculate R and phi
            thisR,thisphi= _dlToRphi(ds[ii],l)
            #sample velocities
            vv= self.sampleVRVT(thisR,n=1,nsigma=nsigma,target=target)[0]
            out.append(Orbit([thisR,vv[0],vv[1],thisphi]))
        return out

    def asymmetricdrift(self,R):
        """
        NAME:
           asymmetricdrift
        PURPOSE:
           calculate the asymmetric drift (vc-mean-vphi)
        INPUT:
           R - radius at which to calculate the asymmetric drift (/ro)
        OUTPUT:
           asymmetric drift at R
        HISTORY:
           2011-04-02 - Written - Bovy (NYU)
        """
        sigmaR2= self.targetSigma2(R)
        return sigmaR2/2./R**self._beta*(1./self._gamma**2.-1.
                                         -R*self._surfaceSigmaProfile.surfacemassDerivative(R,log=True)
                                         -R*self._surfaceSigmaProfile.sigma2Derivative(R,log=True))


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
        if relative:
            norm= 1.
        else:
            norm= sc.exp(logSigmaR)
        #Use the asymmetric drift equation to estimate va
        va= sigmaR2/2./R**self._beta*(1./self._gamma**2.-1.
                                      -R*self._surfaceSigmaProfile.surfacemassDerivative(R,log=True)
                                      -R*self._surfaceSigmaProfile.sigma2Derivative(R,log=True))
        if romberg:
            return bovy_dblquad(_surfaceIntegrand,
                                self._gamma*(R**self._beta-va)/sigmaR1-nsigma,
                                self._gamma*(R**self._beta-va)/sigmaR1+nsigma,
                                lambda x: 0., lambda x: nsigma,
                                [R,self,logSigmaR,logsigmaR2,sigmaR1,
                                 self._gamma],
                                tol=10.**-8)/sc.pi*norm
        else:
            return integrate.dblquad(_surfaceIntegrand,
                                     self._gamma*(R**self._beta-va)/sigmaR1-nsigma,
                                     self._gamma*(R**self._beta-va)/sigmaR1+nsigma,
                                     lambda x: 0., lambda x: nsigma,
                                     (R,self,logSigmaR,logsigmaR2,sigmaR1,
                                      self._gamma),
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
        if relative:
            norm= 1.
        else:
            norm= sc.exp(logSigmaR+logsigmaR2)
        #Use the asymmetric drift equation to estimate va
        va= sigmaR2/2./R**self._beta*(1./self._gamma**2.-1.
                                      -R*self._surfaceSigmaProfile.surfacemassDerivative(R,log=True)
                                      -R*self._surfaceSigmaProfile.sigma2Derivative(R,log=True))
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

    def vmomentsurfacemass(self,R,n,m,romberg=False,nsigma=None,
                           relative=False,phi=0.):
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
        OPTIONAL INPUT:
           nsigma - number of sigma to integrate the velocities over
        KEYWORDS:
           romberg - if True, use a romberg integrator (default: False)
        OUTPUT:
           <vR^n vT^m  x surface-mass> at R
        HISTORY:
           2011-03-30 - Written - Bovy (NYU)
        """
        #odd moments of vR are zero
        if isinstance(n,int) and n%2 == 1:
            return 0.
        if nsigma == None:
            nsigma= _NSIGMA
        logSigmaR= self.targetSurfacemass(R,log=True)
        sigmaR2= self.targetSigma2(R)
        sigmaR1= sc.sqrt(sigmaR2)
        logsigmaR2= sc.log(sigmaR2)
        if relative:
            norm= 1.
        else:
            norm= sc.exp(logSigmaR+logsigmaR2*(n+m)/2.)/self._gamma**m
        #Use the asymmetric drift equation to estimate va
        va= sigmaR2/2./R**self._beta*(1./self._gamma**2.-1.
                                      -R*self._surfaceSigmaProfile.surfacemassDerivative(R,log=True)
                                      -R*self._surfaceSigmaProfile.sigma2Derivative(R,log=True))
        if romberg:
            return bovy_dblquad(_vmomentsurfaceIntegrand,
                                self._gamma*(R**self._beta-va)/sigmaR1-nsigma,
                                self._gamma*(R**self._beta-va)/sigmaR1+nsigma,
                                lambda x: 0., lambda x: nsigma,
                                [R,self,logSigmaR,logsigmaR2,sigmaR1,
                                 self._gamma,n,m],
                                tol=10.**-8)/sc.pi*norm
        else:
            return integrate.dblquad(_vmomentsurfaceIntegrand,
                                     self._gamma*(R**self._beta-va)/sigmaR1-nsigma,
                                     self._gamma*(R**self._beta-va)/sigmaR1+nsigma,
                                     lambda x: 0., lambda x: nsigma,
                                     (R,self,logSigmaR,logsigmaR2,sigmaR1,
                                      self._gamma,n,m),
                                     epsrel=_EPSREL)[0]/sc.pi*norm

    def sigma2(self,R,romberg=False,nsigma=None,phi=0.):
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

    def sigmaT2(self,R,romberg=False,nsigma=None,phi=0.):
        """
        NAME:
           sigmaT2
        PURPOSE:
           calculate sigma_T^2 at R by marginalizing over velocity
        INPUT:
           R - radius at which to calculate sigma_T^2 (/ro)
        OPTIONAL INPUT:
           nsigma - number of sigma to integrate the velocities over
        KEYWORDS:
           romberg - if True, use a romberg integrator (default: False)
        OUTPUT:
           sigma_T^2 at R
        HISTORY:
           2011-03-30 - Written - Bovy (NYU)
        """
        surfmass= self.surfacemass(R,romberg=romberg,nsigma=nsigma)
        return (self.vmomentsurfacemass(R,0,2,romberg=romberg,nsigma=nsigma)
                -self.vmomentsurfacemass(R,0,1,romberg=romberg,nsigma=nsigma)\
                    **2.\
                    /surfmass)/surfmass

    def sigmaR2(self,R,romberg=False,nsigma=None,phi=0.):
        """
        NAME:
           sigmaR2 (duplicate of sigma2 for consistency)
        PURPOSE:
           calculate sigma_R^2 at R by marginalizing over velocity
        INPUT:
           R - radius at which to calculate sigma_R^2 (/ro)
        OPTIONAL INPUT:
           nsigma - number of sigma to integrate the velocities over
        KEYWORDS:
           romberg - if True, use a romberg integrator (default: False)
        OUTPUT:
           sigma_R^2 at R
        HISTORY:
           2011-03-30 - Written - Bovy (NYU)
        """
        return self.sigma2(R,romberg=romberg,nsigma=nsigma)

    def meanvT(self,R,romberg=False,nsigma=None,phi=0.):
        """
        NAME:
           meanvT
        PURPOSE:
           calculate <vT> at R by marginalizing over velocity
        INPUT:
           R - radius at which to calculate <vT> (/ro)
        OPTIONAL INPUT:
           nsigma - number of sigma to integrate the velocities over
        KEYWORDS:
           romberg - if True, use a romberg integrator (default: False)
        OUTPUT:
           <vT> at R
        HISTORY:
           2011-03-30 - Written - Bovy (NYU)
        """
        return self.vmomentsurfacemass(R,0,1,romberg=romberg,nsigma=nsigma)\
            /self.surfacemass(R,romberg=romberg,nsigma=nsigma)

    def meanvR(self,R,romberg=False,nsigma=None,phi=0.):
        """
        NAME:
           meanvR
        PURPOSE:
           calculate <vR> at R by marginalizing over velocity
        INPUT:
           R - radius at which to calculate <vR> (/ro)
        OPTIONAL INPUT:
           nsigma - number of sigma to integrate the velocities over
        KEYWORDS:
           romberg - if True, use a romberg integrator (default: False)
        OUTPUT:
           <vR> at R
        HISTORY:
           2011-03-30 - Written - Bovy (NYU)
        """
        return self.vmomentsurfacemass(R,1,0,romberg=romberg,nsigma=nsigma)\
            /self.surfacemass(R,romberg=romberg,nsigma=nsigma)

    def _ELtowRRapRperi(self,E,L):
        """
        NAME:
           _ELtowRRapRperi
        PURPOSE:
           calculate the radial frequency based on E,L, also return rap and 
           rperi
        INPUT:
           E - energy
           L - angular momentum
        OUTPUT:
           wR(E.L)
        HISTORY:
           2010-07-11 - Written - Bovy (NYU)
        """
        if self._beta == 0.:
            rperi, rap= calcRapRperiFromELFlat(E,L,vc=1.,ro=1.)
            aA= actionAngleFlat(rperi,0.,L/rperi)
        else:
            rperi, rap= calcRapRperiFromELPower(E,L,vc=1.,ro=1.)
            aA= actionAnglePower(rperi,0.,L/rperi,beta=self._beta)
        TR= aA.TR()[0]
        return (2.*m.pi/TR, rap, rperi)

    def sample(self,n=1,rrange=None,returnROrbit=True,returnOrbit=False,
               nphi=1.,los=None,losdeg=True,nsigma=None,maxd=None,target=True):
        """
        NAME:
           sample
        PURPOSE:
           sample n*nphi points from this DF
        INPUT:
           n - number of desired sample (specifying this rather than calling 
               this routine n times is more efficient)
           rrange - if you only want samples in this rrange, set this keyword 
                    (only works when asking for an (RZ)Orbit
           returnROrbit - if True, return a planarROrbit instance: 
                          [R,vR,vT] (default)
           returnOrbit - if True, return a planarOrbit instance (including phi)
           nphi - number of azimuths to sample for each E,L
           los= line of sight sampling along this line of sight
           losdeg= los in degrees? (default=True)
           target= if True, use target surface mass and sigma2 profiles
                   (default=True)
           nsigma= number of sigma to rejection-sample on
           maxd= maximum distance to consider (for the rejection sampling)
        OUTPUT:
           n*nphi list of [[E,Lz],...] or list of planar(R)Orbits
           CAUTION: lists of EL need to be post-processed to account for the 
                    \kappa/\omega_R discrepancy
        HISTORY:
           2010-07-10 - Started  - Bovy (NYU)
        """
        raise AttributeError("'sample' method for this disk df is not implemented")

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
        return self._gamma*sc.exp(logsigmaR2-SRE2+self.targetSurfacemass(xE,log=True)-logSigmaR+sc.exp(logOLLE-SRE2)+correction[0])/2./nu.pi

    def sample(self,n=1,rrange=None,returnROrbit=True,returnOrbit=False,
               nphi=1.,los=None,losdeg=True,nsigma=None,target=True,
               maxd=None):
        """
        NAME:
           sample
        PURPOSE:
           sample n*nphi points from this DF
        INPUT:
           n - number of desired sample (specifying this rather than calling 
               this routine n times is more efficient)
           rrange - if you only want samples in this rrange, set this keyword 
                    (only works when asking for an (RZ)Orbit
           returnROrbit - if True, return a planarROrbit instance: 
                          [R,vR,vT] (default)
           returnOrbit - if True, return a planarOrbit instance (including phi)
           nphi - number of azimuths to sample for each E,L
           los= if set, sample along this line of sight (deg) (assumes that the Sun is located at R=1,phi=0)
           losdeg= if False, los is in radians (default=True)
           target= if True, use target surface mass and sigma2 profiles
                   (default=True)
           nsigma= number of sigma to rejection-sample on
           maxd= maximum distance to consider (for the rejection sampling)
        OUTPUT:
           n*nphi list of [[E,Lz],...] or list of planar(R)Orbits
           CAUTION: lists of EL need to be post-processed to account for the 
                    \kappa/\omega_R discrepancy
        HISTORY:
           2010-07-10 - Started  - Bovy (NYU)
        """
        if not los is None:
            return self.sampleLOS(los,deg=losdeg,n=n,maxd=maxd,
                                  nsigma=nsigma,target=target)
        #First sample xE
        if self._correct:
            xE= sc.array(bovy_ars([0.,0.],[True,False],[0.05,2.],_ars_hx,
                                  _ars_hpx,nsamples=n,
                                  hxparams=(self._surfaceSigmaProfile,
                                            self._corr)))
        else:
            xE= sc.array(bovy_ars([0.,0.],[True,False],[0.05,2.],_ars_hx,
                                  _ars_hpx,nsamples=n,
                                  hxparams=(self._surfaceSigmaProfile,
                                            None)))
        #Calculate E
        if self._beta == 0.:
            E= sc.log(xE)+0.5
        else: #non-flat rotation curve
            E= .5*xE**(2.*self._beta)*(1.+1./self._beta)
        #Then sample Lz
        LCE= xE**(self._beta+1.)
        OR= xE**(self._beta-1.)
        Lz= self._surfaceSigmaProfile.sigma2(xE)*sc.log(stats.uniform.rvs(size=n))/OR
        if self._correct:
            for ii in range(len(xE)):
                Lz[ii]*= self._corr.correct(xE[ii],log=False)[1]
        Lz+= LCE
        if not returnROrbit and not returnOrbit:
            out= [[e,l] for e,l in zip(E,Lz)]
        else:
            if not hasattr(self,'_psp'):
                self._psp= PowerSphericalPotential(alpha=2.-self._beta,normalize=True).toPlanar()
            out= []
            for ii in range(int(n)):
                try:
                    wR, rap, rperi= self._ELtowRRapRperi(E[ii],Lz[ii])
                except ValueError:
                    continue
                TR= 2.*m.pi/wR
                tr= stats.uniform.rvs()*TR
                if tr > TR/2.:
                    tr-= TR/2.
                    thisOrbit= Orbit([rperi,0.,Lz[ii]/rperi])
                else:
                    thisOrbit= Orbit([rap,0.,Lz[ii]/rap])
                thisOrbit.integrate(sc.array([0.,tr]),self._psp)
                if returnOrbit:
                    vxvv= thisOrbit(tr).vxvv
                    if not los is None: #Sample along a given line of sight
                        if losdeg: l= los*_DEGTORAD
                        else: l= los
                        if l > (2.*m.pi): l-= 2.*m.pi
                        if l < 0: l+= 2.*m.pi
                        sinphil= 1./vxvv[0]*m.sin(l)
                        if m.fabs(sinphil) > 1.: continue
                        if stats.uniform.rvs() < 0.5:
                            phil= m.asin(sinphil)
                        else:
                            phil= m.pi-m.asin(sinphil)
                        phi= phil-l
                        if phi > (2.*m.pi): phi-= 2.*m.pi
                        if phi < 0: phi+= 2.*m.pi
                        #make sure this is on the right side of the los
                        if l >= 0. and l <= m.pi/2. and phi > m.pi: continue
                        elif l >= m.pi/2. and l <= m.pi and phi > m.pi/2.: \
                                continue
                        elif l >= m.pi and l <= 3.*m.pi/2. and phi < 3.*m.pi/2.: continue
                        elif l >= 3.*m.pi/2. and phi < m.pi: continue
                            
                        thisOrbit= Orbit(vxvv=sc.array([vxvv[0],vxvv[1],
                                                        vxvv[2],
                                                        phi]).reshape(4))
                    else:
                        thisOrbit= Orbit(vxvv=sc.array([vxvv[0],vxvv[1],vxvv[2],
                                                        stats.uniform.rvs()\
                                                            *m.pi*2.])\
                                             .reshape(4))
                else:
                    thisOrbit= Orbit(thisOrbit(tr))
                kappa= _kappa(thisOrbit.vxvv[0],self._beta)
                if not rrange == None:
                    if thisOrbit.vxvv[0] < rrange[0] \
                            or thisOrbit.vxvv[0] > rrange[1]:
                        continue
                if los is None:
                    mult= sc.ceil(kappa/wR*nphi)-1.
                    kappawR= kappa/wR*nphi-mult
                    while mult > 0:
                        if returnOrbit:
                            out.append(Orbit(vxvv=sc.array([vxvv[0],vxvv[1],
                                                            vxvv[2],
                                                            stats.uniform.rvs()*m.pi*2.]).reshape(4)))
                        else:
                            out.append(thisOrbit)
                        mult-= 1
                else:
                    if losdeg: l= los*_DEGTORAD
                    else: l= los
                    if l > (2.*m.pi): l-= 2.*m.pi
                    if l < 0: l+= 2.*m.pi
                    sinphil= 1./vxvv[0]*m.sin(l)
                    if m.fabs(sinphil) > 1.: continue
                    if stats.uniform.rvs() < 0.5:
                        phil= m.asin(sinphil)
                    else:
                        phil= m.pi-m.asin(sinphil)
                    phi= phil-l
                    if phi > (2.*m.pi): phi-= 2.*m.pi
                    if phi < 0: phi+= 2.*m.pi
                                #make sure this is on the right side of the los
                    if l >= 0. and l <= m.pi/2. and phi > m.pi: continue
                    elif l >= m.pi/2. and l <= m.pi and phi > m.pi/2.: \
                            continue
                    elif l >= m.pi and l <= 3.*m.pi/2. and phi < 3.*m.pi/2.: \
                            continue
                    elif l >= 3.*m.pi/2. and phi < m.pi: continue
                    #Calcualte dphidl
                    dphidl= m.fabs(1./vxvv[0]*m.cos(l)/m.cos(l+phi)-1.)
                    mult= sc.ceil(dphidl*kappa/wR*nphi)-1.
                    kappawR= kappa/wR*nphi-mult
                    while mult > 0:
                        out.append(Orbit(vxvv=sc.array([vxvv[0],vxvv[1],
                                                        vxvv[2],
                                                        phi]).reshape(4)))
                        mult-= 1
                if stats.uniform.rvs() > kappawR:
                    continue
                out.append(thisOrbit)
        #Recurse to get enough
        if len(out) < n*nphi:
            out.extend(self.sample(n=int(n-len(out)/nphi),rrange=rrange,
                                   returnROrbit=returnROrbit,
                                   returnOrbit=returnOrbit,nphi=int(nphi),
                                   los=los,losdeg=losdeg))
        if len(out) > n*nphi:
            print n, nphi, n*nphi
            out= out[0:int(n*nphi)]
        return out

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
        return diskdf.__init__(self,surfaceSigma=surfaceSigma,
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
        return self._gamma*sc.exp(logsigmaR2-SRE2+self.targetSurfacemass(xL,log=True)-logSigmaR-sc.exp(logECLE-SRE2)+correction[0])/2./nu.pi

    def sample(self,n=1,rrange=None,returnROrbit=True,returnOrbit=False,
               nphi=1.,los=None,losdeg=True,nsigma=None,maxd=None,target=True):
        """
        NAME:
           sample
        PURPOSE:
           sample n*nphi points from this DF
        INPUT:
           n - number of desired sample (specifying this rather than calling 
               this routine n times is more efficient)
           rrange - if you only want samples in this rrange, set this keyword 
                    (only works when asking for an (RZ)Orbit
           returnROrbit - if True, return a planarROrbit instance: 
                          [R,vR,vT] (default)
           returnOrbit - if True, return a planarOrbit instance (including phi)
           nphi - number of azimuths to sample for each E,L
           los= if set, sample along this line of sight (deg) (assumes that the Sun is located at R=1,phi=0)
           losdeg= if False, los is in radians (default=True)
           target= if True, use target surface mass and sigma2 profiles
                   (default=True)
           nsigma= number of sigma to rejection-sample on
           maxd= maximum distance to consider (for the rejection sampling)
        OUTPUT:
           n*nphi list of [[E,Lz],...] or list of planar(R)Orbits
           CAUTION: lists of EL need to be post-processed to account for the 
                    \kappa/\omega_R discrepancy
        HISTORY:
           2010-07-10 - Started  - Bovy (NYU)
        """
        if not los is None:
            return self.sampleLOS(los,n=n,maxd=maxd,nsigma=nsigma,target=target)
        #First sample xL
        if self._correct:
            xL= sc.array(bovy_ars([0.,0.],[True,False],[0.05,2.],_ars_hx,
                                  _ars_hpx,nsamples=n,
                                  hxparams=(self._surfaceSigmaProfile,
                                            self._corr)))
        else:
            xL= sc.array(bovy_ars([0.,0.],[True,False],[0.05,2.],_ars_hx,
                                  _ars_hpx,nsamples=n,
                                  hxparams=(self._surfaceSigmaProfile,
                                            None)))
        #Calculate Lz
        Lz= xL**(self._beta+1.)
        #Then sample E
        if self._beta == 0.:
            ECL= sc.log(xL)+0.5
        else:
            ECL= 0.5*(1./self._beta+1.)*xL**(2.*self._beta)
        E= -self._surfaceSigmaProfile.sigma2(xL)*sc.log(stats.uniform.rvs(size=n))
        if self._correct:
            for ii in range(len(xL)):
                E[ii]*= self._corr.correct(xL[ii],log=False)[1]
        E+= ECL
        if not returnROrbit and not returnOrbit:
            out= [[e,l] for e,l in zip(E,Lz)]
        else:
            if not hasattr(self,'_psp'):
                self._psp= PowerSphericalPotential(alpha=2.-self._beta,normalize=True).toPlanar()
            out= []
            for ii in range(n):
                try:
                    wR, rap, rperi= self._ELtowRRapRperi(E[ii],Lz[ii])
                except ValueError:
                    continue
                TR= 2.*m.pi/wR
                tr= stats.uniform.rvs()*TR
                if tr > TR/2.:
                    tr-= TR/2.
                    thisOrbit= Orbit([rperi,0.,Lz[ii]/rperi])
                else:
                    thisOrbit= Orbit([rap,0.,Lz[ii]/rap])
                thisOrbit.integrate(sc.array([0.,tr]),self._psp)
                if returnOrbit:
                    vxvv= thisOrbit(tr).vxvv
                    thisOrbit= Orbit(vxvv=sc.array([vxvv[0],vxvv[1],vxvv[2],
                                                    stats.uniform.rvs()*m.pi*2.]).reshape(4))
                else:
                    thisOrbit= Orbit(thisOrbit(tr))
                kappa= _kappa(thisOrbit.vxvv[0],self._beta)
                if not rrange == None:
                    if thisOrbit.vxvv[0] < rrange[0] or thisOrbit.vxvv[0] > rrange[1]:
                        continue
                mult= sc.ceil(kappa/wR*nphi)-1.
                kappawR= kappa/wR*nphi-mult
                while mult > 0:
                    if returnOrbit:
                        out.append(Orbit(vxvv=sc.array([vxvv[0],vxvv[1],
                                                        vxvv[2],
                                                        stats.uniform.rvs()*m.pi*2.]).reshape(4)))
                    else:
                        out.append(thisOrbit)
                    mult-= 1
                if stats.uniform.rvs() > kappawR:
                    continue
                out.append(thisOrbit)
        #Recurse to get enough
        if len(out) < n*nphi:
            out.extend(self.sample(n=n-len(out)/nphi,rrange=rrange,
                                   returnROrbit=returnROrbit,
                                   returnOrbit=returnOrbit,nphi=nphi))
        if len(out) > n*nphi:
            out= out[0:n*nphi]
        return out

def _surfaceIntegrand(vR,vT,R,df,logSigmaR,logsigmaR2,sigmaR1,gamma):
    """Internal function that is the integrand for the surface mass integration"""
    E,L= _vRpvTpRToEL(vR,vT,R,df._beta,sigmaR1,gamma)
    return df.eval(E,L,logSigmaR,logsigmaR2)*2.*nu.pi/df._gamma #correct

def _sigma2surfaceIntegrand(vR,vT,R,df,logSigmaR,logsigmaR2,sigmaR1,gamma):
    """Internal function that is the integrand for the sigma-squared times
    surface mass integration"""
    E,L= _vRpvTpRToEL(vR,vT,R,df._beta,sigmaR1,gamma)
    return vR**2.*df.eval(E,L,logSigmaR,logsigmaR2)*2.*nu.pi/df._gamma #correct

def _vmomentsurfaceIntegrand(vR,vT,R,df,logSigmaR,logsigmaR2,sigmaR1,gamma,
                             n,m):
    """Internal function that is the integrand for the velocity moment times
    surface mass integration"""
    E,L= _vRpvTpRToEL(vR,vT,R,df._beta,sigmaR1,gamma)
    return vR**n*vT**m*df.eval(E,L,logSigmaR,logsigmaR2)*2.*nu.pi/df._gamma #correct

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
           interp_k - 'k' keyword to give to InterpolatedUnivariateSpline
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
        if kwargs.has_key('interp_k'):
            self._interp_k= kwargs['interp_k']
        else:
            self._interp_k= _INTERPDEGREE
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
        self._surfaceInterpolate= interpolate.InterpolatedUnivariateSpline(interpRs,
                                                       sc.log(sc.append(self._corrections[:,0],1.)),
                                                       k=self._interp_k)
        self._sigma2Interpolate= interpolate.InterpolatedUnivariateSpline(interpRs,
                                                      sc.log(sc.append(self._corrections[:,1],1.)),
                                                      k=self._interp_k)
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
        elif R > (2.*self._rmax):
            out= sc.array([0.,0.])
        else:
            if _SCIPYVERSION >= 0.9:
                out= sc.array([self._surfaceInterpolate(R),
                               self._sigma2Interpolate(R)])
            else:
                out= sc.array([self._surfaceInterpolate(R)[0],
                               self._sigma2Interpolate(R)[0]])
        if log:
            return out
        else:
            return sc.exp(out)
            

    def derivLogcorrect(self,R):
        """
        NAME:
           derivLogcorrect
        PURPOSE:
           calculate the derivative of the log of the correction in Sigma 
           and sigma2 at R
        INPUT:
           R - Galactocentric radius(/ro)
        OUTPUT:
           [d log(Sigma correction)/dR, d log(sigma2 correction)/dR]
        HISTORY:
           2010-03-10 - Written - Bovy (NYU)
        """
        if R < _RMIN:
            out= sc.array([self._surfaceDerivSmallR,
                           self._sigma2DerivSmallR])
        elif R > (2.*self._rmax):
            out= sc.array([0.,0.])
        else:
            if _SCIPYVERSION >= 0.9:
                out= sc.array([self._surfaceInterpolate(R,nu=1)[0],
                               self._sigma2Interpolate(R,nu=1)[0]])
            else:
                out= sc.array([self._surfaceInterpolate(R,nu=1)[0],
                               self._sigma2Interpolate(R,nu=1)[0]])
        return out
            

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
                                        interp_k=self._interp_k)
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

def _ars_hx(x,args):
    """
    NAME:
       _ars_hx
    PURPOSE:
       h(x) for ARS sampling of the input surfacemass profile
    INPUT:
       x - R(/ro)
       args= (surfaceSigma, dfcorr)
          surfaceSigma - surfaceSigmaProfile instance
          dfcorr - DFcorrection instance
    OUTPUT:
       log(x)+log surface(x) + log(correction)
    HISTORY:
       2010-07-11 - Written - Bovy (NYU)
    """
    surfaceSigma, dfcorr= args
    if dfcorr is None:
        return m.log(x)+surfaceSigma.surfacemass(x,log=True)
    else:
        return m.log(x)+surfaceSigma.surfacemass(x,log=True)+dfcorr.correct(x)[0]

def _ars_hpx(x,args):
    """
    NAME:
       _ars_hpx
    PURPOSE:
       h'(x) for ARS sampling of the input surfacemass profile
    INPUT:
       x - R(/ro)
       args= (surfaceSigma, dfcorr)
          surfaceSigma - surfaceSigmaProfile instance
          dfcorr - DFcorrection instance
    OUTPUT:
       derivative of log(x)+log surface(x) + log(correction) wrt x
    HISTORY:
       2010-07-11 - Written - Bovy (NYU)
    """
    surfaceSigma, dfcorr= args
    if dfcorr is None:
        return 1./x+surfaceSigma.surfacemassDerivative(x,log=True)
    else:
        return 1./x+surfaceSigma.surfacemassDerivative(x,log=True)+dfcorr.derivLogcorrect(x)[0]

def _kappa(R,beta):
    """Internal function to give kappa(r)"""
    return m.sqrt(2.*(1.+beta))*R**(beta-1)

def _jacobian_rphi_dl(d,l,R=None,phi=None):
    """Compute the jacobian for transforming Galactocentric coordinates to Galactic coordinates, /d"""
    if R is None or phi is None:
        R, phi= _dlToRphi(d,l)
    matrix= sc.zeros((2,2))
    cosphi= m.cos(phi)
    sinphi= m.sin(phi)
    sinl= m.sin(l)
    cosl= m.cos(l)
    matrix[0,0]= 1./R*sinl
    matrix[1,0]= 1./R*(d-cosl)
    if m.fabs(cosphi) > m.sqrt(2.)/2.: #use 1./cosphi expression       
        matrix[0,1]= 1./cosphi*(1./R*cosl-d/R**3.*sinl**2.)
        matrix[1,1]= 1./cosphi*(sinl/R-d**2./R**3.*sinl+d/R**3.*sinl*cosl)
    else:
        matrix[0,1]= -(R-cosphi)/R**2./sinphi*sinl
        matrix[1,1]= 1./R/sinphi*(d-(R-cosphi)/R*(d-cosl))
    return linalg.det(matrix)

def _dlToRphi(d,l):
    """Convert d and l to R and phi, l is in radians"""
    R= m.sqrt(1.+d**2.-2.*d*m.cos(l))
    if 1./m.cos(l) < d and m.cos(l) > 0.:
        theta= m.pi-m.asin(d/R*m.sin(l))
    else:
        theta= m.asin(d/R*m.sin(l))
    return (R,theta)
    
def _vtmaxEq(vT,R,diskdf):
    """Equation to solve to find the max vT at R"""
    #Calculate a bunch of stuff that we need
    if diskdf._beta == 0.:
        E= vT**2./2.+sc.log(R)
        xE= sc.exp(E-.5)
        OE= xE**-1.
        LCE= xE
        dxEdvT= xE*vT
    else: #non-flat rotation curve
        E= vT**2./2.+1./2./diskdf._beta*R**(2.*diskdf._beta)
        xE= (2.*E/(1.+1./self._beta))**(1./2./self._beta)
        OE= xE**(diskdf._beta-1.)
        LCE= xE**(diskdf._beta+1.)
        dxEdvT= xE/2./diskdf._beta/E*vT
    L= R*vT
    sigma2xE= diskdf._surfaceSigmaProfile.sigma2(xE,log=False)
    return OE*R/sigma2xE+\
        (diskdf._surfaceSigmaProfile.surfacemassDerivative(xE,log=True)\
             -(1.+OE*(L-LCE)/sigma2xE)*diskdf._surfaceSigmaProfile.sigma2Derivative(xE,log=True)\
             +(L-LCE)/sigma2xE*(diskdf._beta-1.)*xE**(diskdf._beta-2.)\
             -OE*(diskdf._beta+1.)/sigma2xE*xE**diskdf._beta)\
             *dxEdvT

def _marginalizeVperpIntegrandSinAlphaLarge(vR,df,R,sinalpha,cotalpha,
                                            vlos,vcirc,sigma):
    return df(*vRvTRToEL(vR*sigma,cotalpha*vR*sigma+vlos/sinalpha+vcirc,
                        R,df._beta))

def _marginalizeVperpIntegrandSinAlphaSmall(vT,df,R,cosalpha,tanalpha,
                                            vlos,vcirc,sigma):
    return df(*vRvTRToEL(tanalpha*vT*sigma-vlos/cosalpha,vT*sigma+vcirc,
                        R,df._beta))

