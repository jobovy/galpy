###############################################################################
#   diskdf.py: module that interprets (E,Lz) pairs in terms of a 
#              distribution function (following Dehnen 1999)
#
#   This module contains the following classes:
#
#      diskdf - top-level class that represents a distribution function
#      dehnendf - inherits from diskdf, implements Dehnen's 'new' DF
#      shudf - inherits from diskdf, implements Shu's DF
#      DFcorrection - class that represents corrections to the input Sigma(R)
#                     and sigma_R(R) to get closer to the targets
###############################################################################
from __future__ import print_function
_EPSREL=10.**-14.
_NSIGMA= 4.
_INTERPDEGREE= 3
_RMIN=10.**-10.
_MAXD_REJECTLOS= 4.
_PROFILE= False
import copy
import re
import os, os.path
import pickle
import math
import numpy as nu
import scipy as sc
import scipy.integrate as integrate
import scipy.interpolate as interpolate
from scipy import linalg
from scipy import stats
from scipy import optimize
from .surfaceSigmaProfile import *
from galpy.orbit import Orbit
from galpy.util.bovy_ars import bovy_ars
from galpy.util import save_pickles
from galpy.util.bovy_conversion import physical_conversion, \
    potential_physical_input, _APY_UNITS, surfdens_in_msolpc2
from galpy.potential import PowerSphericalPotential
from galpy.actionAngle import actionAngleAdiabatic, actionAngleAxi
from .df import df, _APY_LOADED
if _APY_LOADED:
    from astropy import units
#scipy version
try:
    sversion=re.split(r'\.',sc.__version__)
    _SCIPYVERSION=float(sversion[0])+float(sversion[1])/10.
except: #pragma: no cover
    raise ImportError( "scipy.__version__ not understood, contact galpy developer, send scipy.__version__")
_CORRECTIONSDIR=os.path.join(os.path.dirname(os.path.realpath(__file__)),'data')
_DEGTORAD= math.pi/180.
class diskdf(df):
    """Class that represents a disk DF"""
    def __init__(self,dftype='dehnen',
                 surfaceSigma=expSurfaceSigmaProfile,
                 profileParams=(1./3.,1.0,0.2),
                 correct=False,
                 beta=0.,ro=None,vo=None,**kwargs):
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
        df.__init__(self,ro=ro,vo=vo)
        self._dftype= dftype
        if isinstance(surfaceSigma,surfaceSigmaProfile):
            self._surfaceSigmaProfile= surfaceSigma
        else:
            if _APY_LOADED and isinstance(profileParams[0],units.Quantity):
                newprofileParams=\
                    (profileParams[0].to(units.kpc).value/self._ro,
                     profileParams[1].to(units.kpc).value/self._ro,
                     profileParams[2].to(units.km/units.s).value/self._vo)
                self._roSet= True
                self._voSet= True
                profileParams= newprofileParams
            self._surfaceSigmaProfile= surfaceSigma(profileParams)
        self._beta= beta
        self._gamma= sc.sqrt(2./(1.+self._beta))
        if correct or 'corrections' in kwargs or 'rmax' in kwargs \
                or 'niter' in kwargs or 'npoints' in kwargs:
            self._correct= True
            #Load corrections
            self._corr= DFcorrection(dftype=self.__class__,
                                     surfaceSigmaProfile=self._surfaceSigmaProfile,
                                     beta=beta,**kwargs)
        else:
            self._correct= False
        #Setup aA objects for frequency and rap,rperi calculation
        self._aA= actionAngleAdiabatic(pot=PowerSphericalPotential(normalize=1.,
                                                                   alpha=2.-2.*self._beta),gamma=0.)
        return None
    
    @physical_conversion('phasespacedensity2d',pop=True)
    def __call__(self,*args,**kwargs):
        """
        NAME:

           __call__

        PURPOSE:

           evaluate the distribution function

        INPUT:

           either an orbit instance, a list of such instances,  or E,Lz

           1) Orbit instance or list:
              a) Orbit instance alone: use vxvv member
              b) Orbit instance + t: call the Orbit instance (for list, each instance is called at t)

           2)
              E - energy (/vo^2; or can be Quantity)
              L - angular momentun (/ro/vo; or can be Quantity)

           3) array vxvv [3/4,nt] [must be in natural units /vo,/ro; use Orbit interface for physical-unit input)

        KWARGS:

           marginalizeVperp - marginalize over perpendicular velocity (only supported with 1a) for single orbits above)


           marginalizeVlos - marginalize over line-of-sight velocity (only supported with 1a) for single orbits above)

           nsigma= number of sigma to integrate over when marginalizing

           +scipy.integrate.quad keywords

        OUTPUT:

           DF(orbit/E,L)

        HISTORY:

           2010-07-10 - Written - Bovy (NYU)

        """
        if isinstance(args[0],Orbit):
            if len(args) == 1:
                if kwargs.pop('marginalizeVperp',False):
                    return self._call_marginalizevperp(args[0],**kwargs)
                elif kwargs.pop('marginalizeVlos',False):
                    return self._call_marginalizevlos(args[0],**kwargs)
                else:
                    return sc.real(self.eval(*vRvTRToEL(args[0]._orb.vxvv[1],
                                                        args[0]._orb.vxvv[2],
                                                        args[0]._orb.vxvv[0],
                                                        self._beta,
                                                        self._dftype)))
            else:
                no= args[0](args[1])
                return sc.real(self.eval(*vRvTRToEL(no._orb.vxvv[1],
                                                    no._orb.vxvv[2],
                                                    no._orb.vxvv[0],
                                                    self._beta,
                                                    self._dftype)))
        elif isinstance(args[0],list) \
                 and isinstance(args[0][0],Orbit):
            #Grab all of the vR, vT, and R
            vR= nu.array([o._orb.vxvv[1] for o in args[0]])
            vT= nu.array([o._orb.vxvv[2] for o in args[0]])
            R= nu.array([o._orb.vxvv[0] for o in args[0]])
            return sc.real(self.eval(*vRvTRToEL(vR,vT,R,self._beta,
                                                self._dftype)))
        elif isinstance(args[0],nu.ndarray) and \
                not (hasattr(args[0],'isscalar') and args[0].isscalar):
            #Grab all of the vR, vT, and R
            vR= args[0][1]
            vT= args[0][2]
            R= args[0][0]
            return sc.real(self.eval(*vRvTRToEL(vR,vT,R,self._beta,
                                                self._dftype)))
        else:
            return sc.real(self.eval(*args))

    def _call_marginalizevperp(self,o,**kwargs):
        """Call the DF, marginalizing over perpendicular velocity"""
        #Get l, vlos
        l= o.ll(obs=[1.,0.,0.],ro=1.)*_DEGTORAD
        vlos= o.vlos(ro=1.,vo=1.,obs=[1.,0.,0.,0.,0.,0.])
        R= o.R(use_physical=False)
        phi= o.phi(use_physical=False)
        #Get local circular velocity, projected onto the los
        vcirc= R**self._beta
        vcirclos= vcirc*math.sin(phi+l)
        #Marginalize
        alphalos= phi+l
        if not 'nsigma' in kwargs or ('nsigma' in kwargs and \
                                          kwargs['nsigma'] is None):
            nsigma= _NSIGMA
        else:
            nsigma= kwargs['nsigma']
        kwargs.pop('nsigma',None)
        sigmaR2= self.targetSigma2(R,use_physical=False)
        sigmaR1= sc.sqrt(sigmaR2)
        #Use the asymmetric drift equation to estimate va
        va= sigmaR2/2./R**self._beta*(1./self._gamma**2.-1.
                                      -R*self._surfaceSigmaProfile.surfacemassDerivative(R,log=True)
                                      -R*self._surfaceSigmaProfile.sigma2Derivative(R,log=True))
        if math.fabs(va) > sigmaR1: va = 0. #To avoid craziness near the center
        if math.fabs(math.sin(alphalos)) < math.sqrt(1./2.):
            cosalphalos= math.cos(alphalos)
            tanalphalos= math.tan(alphalos)            
            return integrate.quad(_marginalizeVperpIntegrandSinAlphaSmall,
                                  -self._gamma*va/sigmaR1-nsigma,
                                  -self._gamma*va/sigmaR1+nsigma,
                                  args=(self,R,cosalphalos,tanalphalos,
                                        vlos-vcirclos,vcirc,
                                        sigmaR1/self._gamma),
                                  **kwargs)[0]/math.fabs(cosalphalos)\
                                  *sigmaR1/self._gamma
        else:
            sinalphalos= math.sin(alphalos)
            cotalphalos= 1./math.tan(alphalos)
            return integrate.quad(_marginalizeVperpIntegrandSinAlphaLarge,
                                  -nsigma,nsigma,
                                  args=(self,R,sinalphalos,cotalphalos,
                                        vlos-vcirclos,vcirc,sigmaR1),
                                  **kwargs)[0]/math.fabs(sinalphalos)*sigmaR1
        
    def _call_marginalizevlos(self,o,**kwargs):
        """Call the DF, marginalizing over line-of-sight velocity"""
        #Get d, l, vperp
        l= o.ll(obs=[1.,0.,0.],ro=1.)*_DEGTORAD
        vperp= o.vll(ro=1.,vo=1.,obs=[1.,0.,0.,0.,0.,0.])
        R= o.R(use_physical=False)
        phi= o.phi(use_physical=False)
        #Get local circular velocity, projected onto the perpendicular 
        #direction
        vcirc= R**self._beta
        vcircperp= vcirc*math.cos(phi+l)
        #Marginalize
        alphaperp= math.pi/2.+phi+l
        if not 'nsigma' in kwargs or ('nsigma' in kwargs and \
                                          kwargs['nsigma'] is None):
            nsigma= _NSIGMA
        else:
            nsigma= kwargs['nsigma']
        kwargs.pop('nsigma',None)
        sigmaR2= self.targetSigma2(R,use_physical=False)
        sigmaR1= sc.sqrt(sigmaR2)
        #Use the asymmetric drift equation to estimate va
        va= sigmaR2/2./R**self._beta*(1./self._gamma**2.-1.
                                      -R*self._surfaceSigmaProfile.surfacemassDerivative(R,log=True)
                                      -R*self._surfaceSigmaProfile.sigma2Derivative(R,log=True))
        if math.fabs(va) > sigmaR1: va = 0. #To avoid craziness near the center
        if math.fabs(math.sin(alphaperp)) < math.sqrt(1./2.):
            cosalphaperp= math.cos(alphaperp)
            tanalphaperp= math.tan(alphaperp)
            #we can reuse the VperpIntegrand, since it is just another angle
            return integrate.quad(_marginalizeVperpIntegrandSinAlphaSmall,
                                  -self._gamma*va/sigmaR1-nsigma,
                                  -self._gamma*va/sigmaR1+nsigma,
                                  args=(self,R,cosalphaperp,tanalphaperp,
                                        vperp-vcircperp,vcirc,
                                        sigmaR1/self._gamma),
                                  **kwargs)[0]/math.fabs(cosalphaperp)\
                                  *sigmaR1/self._gamma
        else:
            sinalphaperp= math.sin(alphaperp)
            cotalphaperp= 1./math.tan(alphaperp)
            #we can reuse the VperpIntegrand, since it is just another angle
            return integrate.quad(_marginalizeVperpIntegrandSinAlphaLarge,
                                  -nsigma,nsigma,
                                  args=(self,R,sinalphaperp,cotalphaperp,
                                        vperp-vcircperp,vcirc,sigmaR1),
                                  **kwargs)[0]/math.fabs(sinalphaperp)*sigmaR1
        
    @potential_physical_input
    @physical_conversion('velocity2',pop=True)        
    def targetSigma2(self,R,log=False):
        """
        NAME:

           targetSigma2

        PURPOSE:

           evaluate the target Sigma_R^2(R)

        INPUT:

            R - radius at which to evaluate (can be Quantity)

        OUTPUT:

           target Sigma_R^2(R)

           log - if True, return the log (default: False)

        HISTORY:

           2010-03-28 - Written - Bovy (NYU)

        """
        return self._surfaceSigmaProfile.sigma2(R,log=log)

    @potential_physical_input
    @physical_conversion('surfacedensity',pop=True)        
    def targetSurfacemass(self,R,log=False):
         """
         NAME:

            targetSurfacemass

         PURPOSE:

            evaluate the target surface mass at R

         INPUT:

            R - radius at which to evaluate (can be Quantity)

            log - if True, return the log (default: False)

         OUTPUT:

            Sigma(R)

         HISTORY:

            2010-03-28 - Written - Bovy (NYU)

         """
         return self._surfaceSigmaProfile.surfacemass(R,log=log)

    @physical_conversion('surfacedensitydistance',pop=True)        
    def targetSurfacemassLOS(self,d,l,log=False,deg=True):
        """
        NAME:

            targetSurfacemassLOS

        PURPOSE:

            evaluate the target surface mass along the LOS given l and d

        INPUT:

            d - distance along the line of sight (can be Quantity)

            l - Galactic longitude (in deg, unless deg=False; can be Quantity)

            deg= if False, l is in radians

            log - if True, return the log (default: False)

        OUTPUT:

            Sigma(d,l) x d

        HISTORY:

            2011-03-23 - Written - Bovy (NYU)

        """
        #Calculate R and phi
        if _APY_LOADED and isinstance(l,units.Quantity):
            lrad= l.to(units.rad).value
        elif deg:
            lrad= l*_DEGTORAD
        else:
            lrad= l
        if _APY_LOADED and isinstance(d,units.Quantity):
            d= d.to(units.kpc).value/self._ro
        R, phi= _dlToRphi(d,lrad)
        if log:
            return self._surfaceSigmaProfile.surfacemass(R,log=log)\
                +math.log(d)
        else:
            return self._surfaceSigmaProfile.surfacemass(R,log=log)\
                *d

    @physical_conversion('surfacedensitydistance',pop=True)        
    def surfacemassLOS(self,d,l,deg=True,target=True,
                       romberg=False,nsigma=None,relative=None):
        """
        NAME:

           surfacemassLOS

        PURPOSE:

           evaluate the surface mass along the LOS given l and d

        INPUT:

           d - distance along the line of sight (can be Quantity)

           l - Galactic longitude (in deg, unless deg=False; can be Quantity)

        OPTIONAL INPUT:

           nsigma - number of sigma to integrate the velocities over

        KEYWORDS:

           target= if True, use target surfacemass (default)

           romberg - if True, use a romberg integrator (default: False)

           deg= if False, l is in radians

        OUTPUT:

           Sigma(d,l) x d

        HISTORY:

           2011-03-24 - Written - Bovy (NYU)

        """
        #Calculate R and phi
        if _APY_LOADED and isinstance(l,units.Quantity):
            lrad= l.to(units.rad).value
        elif deg:
            lrad= l*_DEGTORAD
        else:
            lrad= l
        if _APY_LOADED and isinstance(d,units.Quantity):
            d= d.to(units.kpc).value/self._ro
        R, phi= _dlToRphi(d,lrad)
        if target:
            if relative: return d
            else: return self.targetSurfacemass(R,use_physical=False)*d
        else:
            return self.surfacemass(R,romberg=romberg,nsigma=nsigma,
                                    relative=relative,use_physical=False)\
                                    *d

    @physical_conversion('position',pop=True)
    def sampledSurfacemassLOS(self,l,n=1,maxd=None,target=True):
        """
        NAME:

           sampledSurfacemassLOS

        PURPOSE:

           sample a distance along the line of sight

        INPUT:

           l - Galactic longitude (in rad; can be Quantity)

           n= number of distances to sample

           maxd= maximum distance to consider (for the rejection sampling) (can be Quantity)

           target= if True, sample from the 'target' surface mass density, rather than the actual surface mass density (default=True)

        OUTPUT:

           list of samples

        HISTORY:

           2011-03-24 - Written - Bovy (NYU)

        """
        #First calculate where the maximum is
        if target:
            minR= optimize.fmin_bfgs(lambda x: \
                                         -self.targetSurfacemassLOS(x,l,
                                                                    use_physical=False,
                                                                    deg=False),
                                     0.,disp=False)[0]
            maxSM= self.targetSurfacemassLOS(minR,l,deg=False,
                                             use_physical=False)
        else:
            minR= optimize.fmin_bfgs(lambda x: \
                                         -self.surfacemassLOS(x,l,
                                                              deg=False,
                                                              use_physical=False),
                                     0.,disp=False)[0]
            maxSM= self.surfacemassLOS(minR,l,deg=False,use_physical=False)
        #Now rejection-sample
        if _APY_LOADED and isinstance(l,units.Quantity):
                l= l.to(units.rad).value
        if _APY_LOADED and isinstance(maxd,units.Quantity):
            maxd= maxd.to(units.kpc).value/self._ro
        if maxd is None:
            maxd= _MAXD_REJECTLOS
        out= []
        while len(out) < n:
            #sample
            prop= nu.random.random()*maxd
            if target:
                surfmassatprop= self.targetSurfacemassLOS(prop,l,deg=False,
                                                          use_physical=False)
            else:
                surfmassatprop= self.surfacemassLOS(prop,l,deg=False,
                                                    use_physical=False)
            if surfmassatprop/maxSM > nu.random.random(): #accept
                out.append(prop)
        return nu.array(out)

    @potential_physical_input
    @physical_conversion('velocity',pop=True)
    def sampleVRVT(self,R,n=1,nsigma=None,target=True):
        """
        NAME:

           sampleVRVT

        PURPOSE:

           sample a radial and azimuthal velocity at R

        INPUT:

           R - Galactocentric distance (can be Quantity)

           n= number of distances to sample

           nsigma= number of sigma to rejection-sample on

           target= if True, sample using the 'target' sigma_R rather than the actual sigma_R (default=True)

        OUTPUT:

           list of samples

        BUGS:

           should use the fact that vR and vT separate

        HISTORY:

           2011-03-24 - Written - Bovy (NYU)

        """
        #Determine where the max of the v-distribution is using asymmetric drift
        maxVR= 0.
        maxVT= optimize.brentq(_vtmaxEq,0.,R**self._beta+0.2,(R,self))
        maxVD= self(Orbit([R,maxVR,maxVT]))
        #Now rejection-sample
        if nsigma == None:
            nsigma= _NSIGMA
        out= []
        if target:
            sigma= math.sqrt(self.targetSigma2(R,use_physical=False))
        else:
            sigma= math.sqrt(self.sigma2(R,use_physical=False))
        while len(out) < n:
            #sample
            vrg, vtg= nu.random.normal(), nu.random.normal()
            propvR= vrg*nsigma*sigma
            propvT= vtg*nsigma*sigma/self._gamma+maxVT
            VDatprop= self(Orbit([R,propvR,propvT]))
            if VDatprop/maxVD > nu.random.uniform()*nu.exp(-0.5*(vrg**2.+vtg**2.)): #accept
                out.append(sc.array([propvR,propvT]))
        return nu.array(out)

    def sampleLOS(self,los,n=1,deg=True,maxd=None,nsigma=None,
                  targetSurfmass=True,targetSigma2=True):
        """
        NAME:

           sampleLOS

        PURPOSE:

           sample along a given LOS

        INPUT:

           los - line of sight (in deg, unless deg=False; can be Quantity)

           n= number of desired samples

           deg= los in degrees? (default=True)

           targetSurfmass, targetSigma2= if True, use target surface mass and sigma2 profiles, respectively (there is not much point to doing the latter)
                   (default=True)

        OUTPUT:

           returns list of Orbits

        BUGS:
           target=False uses target distribution for derivatives (this is a detail)

        HISTORY:

           2011-03-24 - Started  - Bovy (NYU)

        """
        if _APY_LOADED and isinstance(los,units.Quantity):
            l= los.to(units.rad).value
        elif deg:
            l= los*_DEGTORAD
        else:
            l= los
        out= []
        #sample distances
        ds= self.sampledSurfacemassLOS(l,n=n,maxd=maxd,target=targetSurfmass,
                                       use_physical=False)
        for ii in range(int(n)):
            #Calculate R and phi
            thisR,thisphi= _dlToRphi(ds[ii],l)
            #sample velocities
            vv= self.sampleVRVT(thisR,n=1,nsigma=nsigma,target=targetSigma2,
                                use_physical=False)[0]
            if self._roSet and self._voSet:
                out.append(Orbit([thisR,vv[0],vv[1],thisphi],ro=self._ro,
                                 vo=self._vo))
            else:
                out.append(Orbit([thisR,vv[0],vv[1],thisphi]))
        return out

    @potential_physical_input
    @physical_conversion('velocity',pop=True)
    def asymmetricdrift(self,R):
        """
        NAME:

           asymmetricdrift

        PURPOSE:

           estimate the asymmetric drift (vc-mean-vphi) from an approximation to the Jeans equation

        INPUT:

           R - radius at which to calculate the asymmetric drift (can be Quantity)

        OUTPUT:

           asymmetric drift at R

        HISTORY:

           2011-04-02 - Written - Bovy (NYU)

        """
        sigmaR2= self.targetSigma2(R,use_physical=False)
        return sigmaR2/2./R**self._beta*(1./self._gamma**2.-1.
                                         -R*self._surfaceSigmaProfile.surfacemassDerivative(R,log=True)
                                         -R*self._surfaceSigmaProfile.sigma2Derivative(R,log=True))


    @potential_physical_input
    @physical_conversion('surfacedensity',pop=True)        
    def surfacemass(self,R,romberg=False,nsigma=None,relative=False):
        """
        NAME:

           surfacemass

        PURPOSE:

           calculate the surface-mass at R by marginalizing over velocity

        INPUT:

           R - radius at which to calculate the surfacemass density (can be Quantity)

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
        logSigmaR= self.targetSurfacemass(R,log=True,use_physical=False)
        sigmaR2= self.targetSigma2(R,use_physical=False)
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
        if math.fabs(va) > sigmaR1: va = 0.#To avoid craziness near the center
        if romberg:
            return sc.real(bovy_dblquad(_surfaceIntegrand,
                                        self._gamma*(R**self._beta-va)/sigmaR1-nsigma,
                                        self._gamma*(R**self._beta-va)/sigmaR1+nsigma,
                                        lambda x: 0., lambda x: nsigma,
                                        [R,self,logSigmaR,logsigmaR2,sigmaR1,
                                         self._gamma],
                                        tol=10.**-8)/sc.pi*norm)
        else:
            return integrate.dblquad(_surfaceIntegrand,
                                     self._gamma*(R**self._beta-va)/sigmaR1-nsigma,
                                     self._gamma*(R**self._beta-va)/sigmaR1+nsigma,
                                     lambda x: 0., lambda x: nsigma,
                                     (R,self,logSigmaR,logsigmaR2,sigmaR1,
                                      self._gamma),
                                     epsrel=_EPSREL)[0]/sc.pi*norm

    @potential_physical_input
    @physical_conversion('velocity2surfacedensity',pop=True)
    def sigma2surfacemass(self,R,romberg=False,nsigma=None,
                                relative=False):
        """

        NAME:

           sigma2surfacemass

        PURPOSE:

           calculate the product sigma_R^2 x surface-mass at R by marginalizing over velocity

        INPUT:

           R - radius at which to calculate the sigma_R^2 x surfacemass density (can be Quantity)

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
        logSigmaR= self.targetSurfacemass(R,log=True,use_physical=False)
        sigmaR2= self.targetSigma2(R,use_physical=False)
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
            return sc.real(bovy_dblquad(_sigma2surfaceIntegrand,
                                        self._gamma*(R**self._beta-va)/sigmaR1-nsigma,
                                        self._gamma*(R**self._beta-va)/sigmaR1+nsigma,
                                        lambda x: 0., lambda x: nsigma,
                                        [R,self,logSigmaR,logsigmaR2,sigmaR1,
                                         self._gamma],
                                        tol=10.**-8)/sc.pi*norm)
        else:
            return integrate.dblquad(_sigma2surfaceIntegrand,
                                     self._gamma*(R**self._beta-va)/sigmaR1-nsigma,
                                     self._gamma*(R**self._beta-va)/sigmaR1+nsigma,
                                     lambda x: 0., lambda x: nsigma,
                                     (R,self,logSigmaR,logsigmaR2,sigmaR1,
                                      self._gamma),
                                     epsrel=_EPSREL)[0]/sc.pi*norm

    def vmomentsurfacemass(self,*args,**kwargs):
        """
        NAME:

           vmomentsurfacemass
           
        PURPOSE:

           calculate the an arbitrary moment of the velocity distribution 
           at R times the surfacmass

        INPUT:

           R - radius at which to calculate the moment (in natural units)

           n - vR^n

           m - vT^m

        OPTIONAL INPUT:

           nsigma - number of sigma to integrate the velocities over

        KEYWORDS:

           romberg - if True, use a romberg integrator (default: False)

           deriv= None, 'R', or 'phi': calculates derivative of the moment wrt R or phi

        OUTPUT:

           <vR^n vT^m  x surface-mass> at R (no support for units)

        HISTORY:

           2011-03-30 - Written - Bovy (NYU)

        """
        use_physical= kwargs.pop('use_physical',True)
        ro= kwargs.pop('ro',None)
        if ro is None and hasattr(self,'_roSet') and self._roSet:
            ro= self._ro
        if _APY_LOADED and isinstance(ro,units.Quantity):
            ro= ro.to(units.kpc).value
        vo= kwargs.pop('vo',None)
        if vo is None and hasattr(self,'_voSet') and self._voSet:
            vo= self._vo
        if _APY_LOADED and isinstance(vo,units.Quantity):
            vo= vo.to(units.km/units.s).value
        if use_physical and not vo is None and not ro is None:
            fac= surfdens_in_msolpc2(vo,ro)*vo**(args[1]+args[2])
            if _APY_UNITS:
                u= units.Msun/units.pc**2*(units.km/units.s)**(args[1]+args[2])
            out= self._vmomentsurfacemass(*args,**kwargs)
            if _APY_UNITS:
                return units.Quantity(out*fac,unit=u)
            else:
                return out*fac
        else:
            return self._vmomentsurfacemass(*args,**kwargs)
          
    def _vmomentsurfacemass(self,R,n,m,romberg=False,nsigma=None,
                           relative=False,phi=0.,deriv=None):
        """Non-physical version of vmomentsurfacemass, otherwise the same"""
        #odd moments of vR are zero
        if isinstance(n,int) and n%2 == 1:
            return 0.
        if nsigma == None:
            nsigma= _NSIGMA
        logSigmaR= self.targetSurfacemass(R,log=True,use_physical=False)
        sigmaR2= self.targetSigma2(R,use_physical=False)
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
        if math.fabs(va) > sigmaR1: va = 0. #To avoid craziness near the center
        if deriv is None:
            if romberg:
                return sc.real(bovy_dblquad(_vmomentsurfaceIntegrand,
                                            self._gamma*(R**self._beta-va)/sigmaR1-nsigma,
                                            self._gamma*(R**self._beta-va)/sigmaR1+nsigma,
                                            lambda x: -nsigma, lambda x: nsigma,
                                            [R,self,logSigmaR,logsigmaR2,sigmaR1,
                                             self._gamma,n,m],
                                            tol=10.**-8)/sc.pi*norm/2.)
            else:
                return integrate.dblquad(_vmomentsurfaceIntegrand,
                                         self._gamma*(R**self._beta-va)/sigmaR1-nsigma,
                                         self._gamma*(R**self._beta-va)/sigmaR1+nsigma,
                                         lambda x: -nsigma, lambda x: nsigma,
                                         (R,self,logSigmaR,logsigmaR2,sigmaR1,
                                          self._gamma,n,m),
                                         epsrel=_EPSREL)[0]/sc.pi*norm/2.
        else:
            if romberg:
                return sc.real(bovy_dblquad(_vmomentderivsurfaceIntegrand,
                                            self._gamma*(R**self._beta-va)/sigmaR1-nsigma,
                                            self._gamma*(R**self._beta-va)/sigmaR1+nsigma,
                                            lambda x: -nsigma, lambda x: nsigma,
                                            [R,self,logSigmaR,logsigmaR2,sigmaR1,
                                             self._gamma,n,m,deriv],
                                            tol=10.**-8)/sc.pi*norm/2.)
            else:
                return integrate.dblquad(_vmomentderivsurfaceIntegrand,
                                         self._gamma*(R**self._beta-va)/sigmaR1-nsigma,
                                         self._gamma*(R**self._beta-va)/sigmaR1+nsigma,
                                         lambda x: -nsigma, lambda x: nsigma,
                                         (R,self,logSigmaR,logsigmaR2,sigmaR1,
                                          self._gamma,n,m,deriv),
                                         epsrel=_EPSREL)[0]/sc.pi*norm/2.

    @potential_physical_input
    @physical_conversion('frequency-kmskpc',pop=True)
    def oortA(self,R,romberg=False,nsigma=None,phi=0.):
        """

        NAME:

           oortA

        PURPOSE:

           calculate the Oort function A

        INPUT:

           R - radius at which to calculate A (can be Quantity)

        OPTIONAL INPUT:

           nsigma - number of sigma to integrate the velocities over

        KEYWORDS:

           romberg - if True, use a romberg integrator (default: False)

        OUTPUT:

           Oort A at R

        HISTORY:

           2011-04-19 - Written - Bovy (NYU)

        BUGS:

           could be made more efficient, e.g., surfacemass is calculated multiple times

        """
        #2A= meanvphi/R-dmeanvR/R/dphi-dmeanvphi/dR
        meanvphi= self.meanvT(R,romberg=romberg,nsigma=nsigma,phi=phi,
                              use_physical=False)
        dmeanvRRdphi= 0. #We know this, since the DF does not depend on phi
        surfmass= self._vmomentsurfacemass(R,0,0,phi=phi,romberg=romberg,nsigma=nsigma)
        dmeanvphidR= self._vmomentsurfacemass(R,0,1,deriv='R',phi=phi,romberg=romberg,nsigma=nsigma)/\
            surfmass\
            -self._vmomentsurfacemass(R,0,1,phi=phi,romberg=romberg,nsigma=nsigma)\
            /surfmass**2.\
            *self._vmomentsurfacemass(R,0,0,deriv='R',phi=phi,romberg=romberg,nsigma=nsigma)
        return 0.5*(meanvphi/R-dmeanvRRdphi/R-dmeanvphidR)

    @potential_physical_input
    @physical_conversion('frequency-kmskpc',pop=True)
    def oortB(self,R,romberg=False,nsigma=None,phi=0.):
        """
        NAME:

           oortB

        PURPOSE:

           calculate the Oort function B

        INPUT:

           R - radius at which to calculate B (can be Quantity)

        OPTIONAL INPUT:

           nsigma - number of sigma to integrate the velocities over

        KEYWORDS:

           romberg - if True, use a romberg integrator (default: False)

        OUTPUT:

           Oort B at R

        HISTORY:

           2011-04-19 - Written - Bovy (NYU)

        BUGS:

           could be made more efficient, e.g., surfacemass is calculated multiple times

        """
        #2B= -meanvphi/R+dmeanvR/R/dphi-dmeanvphi/dR
        meanvphi= self.meanvT(R,romberg=romberg,nsigma=nsigma,phi=phi,
                              use_physical=False)
        dmeanvRRdphi= 0. #We know this, since the DF does not depend on phi
        surfmass= self._vmomentsurfacemass(R,0,0,phi=phi,romberg=romberg,nsigma=nsigma)
        dmeanvphidR= self._vmomentsurfacemass(R,0,1,deriv='R',phi=phi,romberg=romberg,nsigma=nsigma)/\
            surfmass\
            -self._vmomentsurfacemass(R,0,1,phi=phi,romberg=romberg,nsigma=nsigma)\
            /surfmass**2.\
            *self._vmomentsurfacemass(R,0,0,deriv='R',phi=phi,romberg=romberg,nsigma=nsigma)
        return 0.5*(-meanvphi/R+dmeanvRRdphi/R-dmeanvphidR)

    @potential_physical_input
    @physical_conversion('frequency-kmskpc',pop=True)
    def oortC(self,R,romberg=False,nsigma=None,phi=0.):
        """
        NAME:

           oortC

        PURPOSE:

           calculate the Oort function C

        INPUT:

           R - radius at which to calculate C (can be Quantity)

        OPTIONAL INPUT:

           nsigma - number of sigma to integrate the velocities over

        KEYWORDS:

           romberg - if True, use a romberg integrator (default: False)

        OUTPUT:

           Oort C at R

        HISTORY:

           2011-04-19 - Written - Bovy (NYU)

        BUGS:

           could be made more efficient, e.g., surfacemass is calculated multiple times
           we know this is zero, but it is calculated anyway (bug or feature?)

        """
        #2C= -meanvR/R-dmeanvphi/R/dphi+dmeanvR/dR
        meanvr= self.meanvR(R,romberg=romberg,nsigma=nsigma,phi=phi,
                            use_physical=False)
        dmeanvphiRdphi= 0. #We know this, since the DF does not depend on phi
        surfmass= self._vmomentsurfacemass(R,0,0,phi=phi,romberg=romberg,nsigma=nsigma)
        dmeanvRdR= self._vmomentsurfacemass(R,1,0,deriv='R',phi=phi,romberg=romberg,nsigma=nsigma)/\
            surfmass #other terms is zero because f is even in vR
        return 0.5*(-meanvr/R-dmeanvphiRdphi/R+dmeanvRdR)

    @potential_physical_input
    @physical_conversion('frequency-kmskpc',pop=True)
    def oortK(self,R,romberg=False,nsigma=None,phi=0.):
        """
        NAME:

           oortK

        PURPOSE:

           calculate the Oort function K

        INPUT:

           R - radius at which to calculate K (can be Quantity)

        OPTIONAL INPUT:

           nsigma - number of sigma to integrate the velocities over

        KEYWORDS:

           romberg - if True, use a romberg integrator (default: False)

        OUTPUT:

           Oort K at R

        HISTORY:

           2011-04-19 - Written - Bovy (NYU)

        BUGS:

           could be made more efficient, e.g., surfacemass is calculated multiple times
           we know this is zero, but it is calculated anyway (bug or feature?)

        """
        #2K= meanvR/R+dmeanvphi/R/dphi+dmeanvR/dR
        meanvr= self.meanvR(R,romberg=romberg,nsigma=nsigma,phi=phi,
                            use_physical=False)
        dmeanvphiRdphi= 0. #We know this, since the DF does not depend on phi
        surfmass= self._vmomentsurfacemass(R,0,0,phi=phi,romberg=romberg,nsigma=nsigma)
        dmeanvRdR= self._vmomentsurfacemass(R,1,0,deriv='R',phi=phi,romberg=romberg,nsigma=nsigma)/\
            surfmass #other terms is zero because f is even in vR
        return 0.5*(+meanvr/R+dmeanvphiRdphi/R+dmeanvRdR)

    @potential_physical_input
    @physical_conversion('velocity2',pop=True)        
    def sigma2(self,R,romberg=False,nsigma=None,phi=0.):
        """
        NAME:

           sigma2

        PURPOSE:

           calculate sigma_R^2 at R by marginalizing over velocity

        INPUT:

           R - radius at which to calculate sigma_R^2 density (can be Quantity)

        OPTIONAL INPUT:

           nsigma - number of sigma to integrate the velocities over

        KEYWORDS:

           romberg - if True, use a romberg integrator (default: False)

        OUTPUT:

           sigma_R^2 at R

        HISTORY:

           2010-03-XX - Written - Bovy (NYU)

        """
        return self.sigma2surfacemass(R,romberg,nsigma,use_physical=False)\
            /self.surfacemass(R,romberg,nsigma,use_physical=False)

    @potential_physical_input
    @physical_conversion('velocity2',pop=True)        
    def sigmaT2(self,R,romberg=False,nsigma=None,phi=0.):
        """

        NAME:

           sigmaT2

        PURPOSE:

           calculate sigma_T^2 at R by marginalizing over velocity

        INPUT:

           R - radius at which to calculate sigma_T^2 (can be Quantity)

        OPTIONAL INPUT:

           nsigma - number of sigma to integrate the velocities over

        KEYWORDS:

           romberg - if True, use a romberg integrator (default: False)

        OUTPUT:

           sigma_T^2 at R

        HISTORY:

           2011-03-30 - Written - Bovy (NYU)

        """
        surfmass= self.surfacemass(R,romberg=romberg,nsigma=nsigma,
                                   use_physical=False)
        return (self._vmomentsurfacemass(R,0,2,romberg=romberg,nsigma=nsigma)
                -self._vmomentsurfacemass(R,0,1,romberg=romberg,nsigma=nsigma)\
                    **2.\
                    /surfmass)/surfmass

    @potential_physical_input
    @physical_conversion('velocity2',pop=True)        
    def sigmaR2(self,R,romberg=False,nsigma=None,phi=0.):
        """
        NAME:

           sigmaR2 (duplicate of sigma2 for consistency)

        PURPOSE:

           calculate sigma_R^2 at R by marginalizing over velocity

        INPUT:

           R - radius at which to calculate sigma_R^2 (can be Quantity)

        OPTIONAL INPUT:

           nsigma - number of sigma to integrate the velocities over

        KEYWORDS:

           romberg - if True, use a romberg integrator (default: False)

        OUTPUT:

           sigma_R^2 at R

        HISTORY:

           2011-03-30 - Written - Bovy (NYU)

        """
        return self.sigma2(R,romberg=romberg,nsigma=nsigma,use_physical=False)

    @potential_physical_input
    @physical_conversion('velocity',pop=True)
    def meanvT(self,R,romberg=False,nsigma=None,phi=0.):
        """
        NAME:

           meanvT

        PURPOSE:

           calculate <vT> at R by marginalizing over velocity

        INPUT:

           R - radius at which to calculate <vT> (can be Quantity)

        OPTIONAL INPUT:

           nsigma - number of sigma to integrate the velocities over

        KEYWORDS:

           romberg - if True, use a romberg integrator (default: False)

        OUTPUT:

           <vT> at R

        HISTORY:

           2011-03-30 - Written - Bovy (NYU)

        """
        return self._vmomentsurfacemass(R,0,1,romberg=romberg,nsigma=nsigma)\
            /self.surfacemass(R,romberg=romberg,nsigma=nsigma,
                              use_physical=False)

    @potential_physical_input
    @physical_conversion('velocity',pop=True)
    def meanvR(self,R,romberg=False,nsigma=None,phi=0.):
        """
        NAME:

           meanvR

        PURPOSE:

           calculate <vR> at R by marginalizing over velocity

        INPUT:

           R - radius at which to calculate <vR> (can be Quantity)

        OPTIONAL INPUT:

           nsigma - number of sigma to integrate the velocities over

        KEYWORDS:

           romberg - if True, use a romberg integrator (default: False)

        OUTPUT:

           <vR> at R

        HISTORY:

           2011-03-30 - Written - Bovy (NYU)

        """
        return self._vmomentsurfacemass(R,1,0,romberg=romberg,nsigma=nsigma)\
            /self.surfacemass(R,romberg=romberg,nsigma=nsigma,
                              use_physical=False)

    @potential_physical_input
    def skewvT(self,R,romberg=False,nsigma=None,phi=0.):
        """
        NAME:

           skewvT

        PURPOSE:

           calculate skew in vT at R by marginalizing over velocity

        INPUT:

           R - radius at which to calculate <vR> (can be Quantity)

        OPTIONAL INPUT:

           nsigma - number of sigma to integrate the velocities over

        KEYWORDS:

           romberg - if True, use a romberg integrator (default: False)

        OUTPUT:

           skewvT

        HISTORY:

           2011-12-07 - Written - Bovy (NYU)

        """
        surfmass= self.surfacemass(R,romberg=romberg,nsigma=nsigma,
                                   use_physical=False)
        vt= self._vmomentsurfacemass(R,0,1,romberg=romberg,nsigma=nsigma)\
            /surfmass
        vt2= self._vmomentsurfacemass(R,0,2,romberg=romberg,nsigma=nsigma)\
            /surfmass
        vt3= self._vmomentsurfacemass(R,0,3,romberg=romberg,nsigma=nsigma)\
            /surfmass
        s2= vt2-vt**2.
        return (vt3-3.*vt*vt2+2.*vt**3.)*s2**(-1.5)

    @potential_physical_input
    def skewvR(self,R,romberg=False,nsigma=None,phi=0.):
        """
        NAME:

           skewvR

        PURPOSE:

           calculate skew in vR at R by marginalizing over velocity

        INPUT:

           R - radius at which to calculate <vR> (can be Quantity)

        OPTIONAL INPUT:

           nsigma - number of sigma to integrate the velocities over

        KEYWORDS:

           romberg - if True, use a romberg integrator (default: False)

        OUTPUT:

           skewvR

        HISTORY:

           2011-12-07 - Written - Bovy (NYU)

        """
        surfmass= self.surfacemass(R,romberg=romberg,nsigma=nsigma,
                                   use_physical=False)
        vr= self._vmomentsurfacemass(R,1,0,romberg=romberg,nsigma=nsigma)\
            /surfmass
        vr2= self._vmomentsurfacemass(R,2,0,romberg=romberg,nsigma=nsigma)\
            /surfmass
        vr3= self._vmomentsurfacemass(R,3,0,romberg=romberg,nsigma=nsigma)\
            /surfmass
        s2= vr2-vr**2.
        return (vr3-3.*vr*vr2+2.*vr**3.)*s2**(-1.5)

    @potential_physical_input
    def kurtosisvT(self,R,romberg=False,nsigma=None,phi=0.):
        """
        NAME:

           kurtosisvT

        PURPOSE:

           calculate excess kurtosis in vT at R by marginalizing over velocity

        INPUT:

           R - radius at which to calculate <vR> (can be Quantity)

        OPTIONAL INPUT:

           nsigma - number of sigma to integrate the velocities over

        KEYWORDS:

           romberg - if True, use a romberg integrator (default: False)

        OUTPUT:

           kurtosisvT

        HISTORY:

           2011-12-07 - Written - Bovy (NYU)

        """
        surfmass= self.surfacemass(R,romberg=romberg,nsigma=nsigma,
                                   use_physical=False)
        vt= self._vmomentsurfacemass(R,0,1,romberg=romberg,nsigma=nsigma)\
            /surfmass
        vt2= self._vmomentsurfacemass(R,0,2,romberg=romberg,nsigma=nsigma)\
            /surfmass
        vt3= self._vmomentsurfacemass(R,0,3,romberg=romberg,nsigma=nsigma)\
            /surfmass
        vt4= self._vmomentsurfacemass(R,0,4,romberg=romberg,nsigma=nsigma)\
            /surfmass
        s2= vt2-vt**2.
        return (vt4-4.*vt*vt3+6.*vt**2.*vt2-3.*vt**4.)*s2**(-2.)-3.

    @potential_physical_input
    def kurtosisvR(self,R,romberg=False,nsigma=None,phi=0.):
        """
        NAME:

           kurtosisvR

        PURPOSE:

           calculate excess kurtosis in vR at R by marginalizing over velocity

        INPUT:

           R - radius at which to calculate <vR> (can be Quantity)

        OPTIONAL INPUT:

           nsigma - number of sigma to integrate the velocities over

        KEYWORDS:

           romberg - if True, use a romberg integrator (default: False)

        OUTPUT:

           kurtosisvR

        HISTORY:

           2011-12-07 - Written - Bovy (NYU)

        """
        surfmass= self.surfacemass(R,romberg=romberg,nsigma=nsigma,
                                   use_physical=False)
        vr= self._vmomentsurfacemass(R,1,0,romberg=romberg,nsigma=nsigma)\
            /surfmass
        vr2= self._vmomentsurfacemass(R,2,0,romberg=romberg,nsigma=nsigma)\
            /surfmass
        vr3= self._vmomentsurfacemass(R,3,0,romberg=romberg,nsigma=nsigma)\
            /surfmass
        vr4= self._vmomentsurfacemass(R,4,0,romberg=romberg,nsigma=nsigma)\
            /surfmass
        s2= vr2-vr**2.
        return (vr4-4.*vr*vr3+6.*vr**2.*vr2-3.*vr**4.)*s2**(-2.)-3.

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
            xE= sc.exp(E-.5)
        else: #non-flat rotation curve                                      
            xE= (2.*E/(1.+1./self._beta))**(1./2./self._beta)
        rperi,rap= self._aA.calcRapRperi(xE,0.,L/xE,0.,0.)
        #Replace the above w/
        aA= actionAngleAxi(xE,0.,L/xE,
                           pot=PowerSphericalPotential(normalize=1.,
                                                       alpha=2.-2.*self._beta).toPlanar())
        TR= aA.TR()
        return (2.*math.pi/TR,rap,rperi)

    def sample(self,n=1,rrange=None,returnROrbit=True,returnOrbit=False,
               nphi=1.,los=None,losdeg=True,nsigma=None,maxd=None,target=True):
        """
        NAME:

           sample

        PURPOSE:

           sample n*nphi points from this DF

        INPUT:

           n - number of desired sample (specifying this rather than calling this routine n times is more efficient)

           rrange - if you only want samples in this rrange, set this keyword (only works when asking for an (RZ)Orbit) (can be Quantity)

           returnROrbit - if True, return a planarROrbit instance: 
                          [R,vR,vT] (default)

           returnOrbit - if True, return a planarOrbit instance (including phi)

           nphi - number of azimuths to sample for each E,L

           los= line of sight sampling along this line of sight (can be Quantity)

           losdeg= los in degrees? (default=True)

           target= if True, use target surface mass and sigma2 profiles (default=True)

           nsigma= number of sigma to rejection-sample on

           maxd= maximum distance to consider (for the rejection sampling)

        OUTPUT:

           n*nphi list of [[E,Lz],...] or list of planar(R)Orbits

           CAUTION: lists of EL need to be post-processed to account for the 
                    \kappa/\omega_R discrepancy

        HISTORY:

           2010-07-10 - Started  - Bovy (NYU)

        """
        raise NotImplementedError("'sample' method for this disk df is not implemented")

    def _estimatemeanvR(self,R,phi=0.,log=False):
        """
        NAME:
           _estimatemeanvR
        PURPOSE:
            quickly estimate meanvR (useful in evolveddiskdf where we
            need an estimate of this but we do not want to spend too
            much time on it)
        INPUT:
           R - radius at which to evaluate (/ro)
           phi= azimuth (not used)
        OUTPUT:
           target Sigma_R^2(R)
           log - if True, return the log (default: False)
        HISTORY:
           2010-03-28 - Written - Bovy (NYU)
        """
        return 0.

    def _estimatemeanvT(self,R,phi=0.,log=False):
        """
        NAME:
           _estimatemeanvT
        PURPOSE:
            quickly estimate meanvR (useful in evolveddiskdf where we
            need an estimate of this but we do not want to spend too
            much time on it)
        INPUT:
           R - radius at which to evaluate (/ro)
           phi= azimuth (not used)
        OUTPUT:
           target Sigma_R^2(R)
        HISTORY:
           2010-03-28 - Written - Bovy (NYU)
        """
        return R**self._beta-self.asymmetricdrift(R,use_physical=False)

    def _estimateSigmaR2(self,R,phi=0.,log=False):
        """
        NAME:
           _estimateSigmaR2
        PURPOSE:
            quickly estimate SigmaR2 (useful in evolveddiskdf where we
            need an estimate of this but we do not want to spend too
            much time on it)
        INPUT:
           R - radius at which to evaluate (/ro)
           phi= azimuth (not used)
        OUTPUT:
           target Sigma_R^2(R)
           log - if True, return the log (default: False)
        HISTORY:
           2010-03-28 - Written - Bovy (NYU)
        """
        return self.targetSigma2(R,log=log,use_physical=False)

    def _estimateSigmaT2(self,R,phi=0.,log=False):
        """
        NAME:
           _estimateSigmaT2
        PURPOSE:
            quickly estimate SigmaT2 (useful in evolveddiskdf where we
            need an estimate of this but we do not want to spend too
            much time on it)
        INPUT:
           R - radius at which to evaluate (/ro)
           phi= azimuth (not used)
        OUTPUT:
           target Sigma_R^2(R)
           log - if True, return the log (default: False)
        HISTORY:
           2010-03-28 - Written - Bovy (NYU)
        """
        if log:
            return self.targetSigma2(R,log=log,use_physical=False)\
                -2.*nu.log(self._gamma)
        else:
            return self.targetSigma2(R,log=log,use_physical=False)\
                /self._gamma**2.


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

                        xD - disk surface mass scalelength (can be Quantity)

                        xS - disk velocity dispersion scalelength (can be Quantity)

                        Sro - disk velocity dispersion at Ro (can be Quantity)

                        Directly given to the 'surfaceSigmaProfile class, so
                        could be anything that class takes

           beta - power-law index of the rotation curve

           correct - if True, correct the DF

           ro= distance from vantage point to GC (kpc; can be Quantity)

           vo= circular velocity at ro (km/s; can be Quantity)

           +DFcorrection kwargs (except for those already specified)

        OUTPUT:

           instance

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
           E - energy (can be Quantity)
           L - angular momentum (can be Quantity)
       OUTPUT:
           DF(E,L)
        HISTORY:
           2010-03-10 - Written - Bovy (NYU)
           2010-03-28 - Moved to dehnenDF - Bovy (NYU)
        """
        if _PROFILE: #pragma: no cover
            import time
            start= time.time()
        if _APY_LOADED and isinstance(E,units.Quantity):
            E= E.to(units.km**2/units.s**2).value/self._vo**2.
        if _APY_LOADED and isinstance(L,units.Quantity):
            L= L.to(units.kpc*units.km/units.s).value/self._ro/self._vo
        #Calculate Re,LE, OmegaE
        if self._beta == 0.:
            xE= sc.exp(E-.5)
            logOLLE= sc.log(L/xE-1.)
        else: #non-flat rotation curve
            xE= (2.*E/(1.+1./self._beta))**(1./2./self._beta)
            logOLLE= self._beta*sc.log(xE)+sc.log(L/xE-xE**self._beta)
        if _PROFILE: #pragma: no cover
            one_time= (time.time()-start)
            start= time.time()
        if self._correct: 
            correction= self._corr.correct(xE,log=True)
        else:
            correction= sc.zeros(2)
        if _PROFILE: #pragma: no cover
            corr_time= (time.time()-start)
            start= time.time()
        SRE2= self.targetSigma2(xE,log=True,use_physical=False)+correction[1]
        if _PROFILE: #pragma: no cover
            targSigma_time= (time.time()-start)
            start= time.time()
            out= self._gamma*sc.exp(logsigmaR2-SRE2+self.targetSurfacemass(xE,log=True,use_physical=False)-logSigmaR+sc.exp(logOLLE-SRE2)+correction[0])/2./nu.pi
            out_time= (time.time()-start)
            tot_time= one_time+corr_time+targSigma_time+out_time
            print(one_time/tot_time, corr_time/tot_time, targSigma_time/tot_time, out_time/tot_time, tot_time)
            return out
        else:
            return self._gamma*sc.exp(logsigmaR2-SRE2+self.targetSurfacemass(xE,log=True,use_physical=False)-logSigmaR+sc.exp(logOLLE-SRE2)+correction[0])/2./nu.pi

    def sample(self,n=1,rrange=None,returnROrbit=True,returnOrbit=False,
               nphi=1.,los=None,losdeg=True,nsigma=None,targetSurfmass=True,
               targetSigma2=True,
               maxd=None,**kwargs):
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
           targetSurfmass, targetSigma2= if True, use target surface mass and sigma2 profiles, respectively (there is not much point to doing the latter)
                   (default=True)
           nsigma= number of sigma to rejection-sample on
           maxd= maximum distance to consider (for the rejection sampling)
        OUTPUT:
           n*nphi list of [[E,Lz],...] or list of planar(R)Orbits
           CAUTION: lists of EL need to be post-processed to account for the 
                    \kappa/\omega_R discrepancy; EL not returned in physical units        
        HISTORY:
           2010-07-10 - Started  - Bovy (NYU)
        """
        if not los is None:
            return self.sampleLOS(los,deg=losdeg,n=n,maxd=maxd,
                                  nsigma=nsigma,targetSurfmass=targetSurfmass,
                                  targetSigma2=targetSigma2)
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
            Lz*= self._corr.correct(xE,log=False)[1,:]
        Lz+= LCE
        if not returnROrbit and not returnOrbit:
            out= [[e,l] for e,l in zip(E,Lz)]
        else:
            if not rrange is None \
                    and _APY_LOADED and isinstance(rrange[0],units.Quantity):
                rrange[0]= rrange[0].to(units.kpc).value/self._ro
                rrange[1]= rrange[1].to(units.kpc).value/self._ro
            if not hasattr(self,'_psp'):
                self._psp= PowerSphericalPotential(alpha=2.-self._beta,normalize=True).toPlanar()
            out= []
            for ii in range(int(n)):
                try:
                    wR, rap, rperi= self._ELtowRRapRperi(E[ii],Lz[ii])
                except ValueError:
                    continue
                TR= 2.*math.pi/wR
                tr= stats.uniform.rvs()*TR
                if tr > TR/2.:
                    tr-= TR/2.
                    thisOrbit= Orbit([rperi,0.,Lz[ii]/rperi])
                else:
                    thisOrbit= Orbit([rap,0.,Lz[ii]/rap])
                thisOrbit.integrate(sc.array([0.,tr]),self._psp)
                if returnOrbit:
                    vxvv= thisOrbit(tr)._orb.vxvv
                    thisOrbit= Orbit(vxvv=sc.array([vxvv[0],vxvv[1],vxvv[2],
                                                    stats.uniform.rvs()\
                                                        *math.pi*2.])\
                                         .reshape(4))
                else:
                    thisOrbit= thisOrbit(tr)
                kappa= _kappa(thisOrbit._orb.vxvv[0],self._beta)
                if not rrange == None:
                    if thisOrbit._orb.vxvv[0] < rrange[0] \
                            or thisOrbit._orb.vxvv[0] > rrange[1]:
                        continue
                mult= sc.ceil(kappa/wR*nphi)-1.
                kappawR= kappa/wR*nphi-mult
                while mult > 0:
                    if returnOrbit:
                        out.append(Orbit(vxvv=sc.array([vxvv[0],vxvv[1],
                                                            vxvv[2],
                                                            stats.uniform.rvs()*math.pi*2.]).reshape(4)))
                    else:
                        out.append(thisOrbit)
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
            print(n, nphi, n*nphi)
            out= out[0:int(n*nphi)]
        if kwargs.get('use_physical',True) and \
                self._roSet and self._voSet:
            if isinstance(out[0],Orbit):
                dum= [o.turn_physical_on(ro=self._ro,vo=self._vo) for o in out]
        return out

    def _dlnfdR(self,R,vR,vT):
        #Calculate a bunch of stuff that we need
        if self._beta == 0.:
            E= vR**2./2.+vT**2./2.+sc.log(R)
            xE= sc.exp(E-.5)
            OE= xE**-1.
            LCE= xE
            dRedR= xE/R
        else: #non-flat rotation curve
            E= vR**2./2.+vT**2./2.+1./2./self._beta*R**(2.*self._beta)
            xE= (2.*E/(1.+1./self._beta))**(1./2./self._beta)
            OE= xE**(self._beta-1.)
            LCE= xE**(self._beta+1.)
            dRedR= xE/2./self._beta/E*R**(2.*self._beta-1.)
        return self._dlnfdRe(R,vR,vT,E=E,xE=xE,OE=OE,LCE=LCE)*dRedR\
            +self._dlnfdl(R,vR,vT,E=E,xE=xE,OE=OE)*vT
            
    def _dlnfdvR(self,R,vR,vT):
        #Calculate a bunch of stuff that we need
        if self._beta == 0.:
            E= vR**2./2.+vT**2./2.+sc.log(R)
            xE= sc.exp(E-.5)
            OE= xE**-1.
            LCE= xE
            dRedvR= xE*vR
        else: #non-flat rotation curve
            E= vR**2./2.+vT**2./2.+1./2./self._beta*R**(2.*self._beta)
            xE= (2.*E/(1.+1./self._beta))**(1./2./self._beta)
            OE= xE**(self._beta-1.)
            LCE= xE**(self._beta+1.)
            dRedvR= xE/2./self._beta/E*vR
        return self._dlnfdRe(R,vR,vT,E=E,xE=xE,OE=OE,LCE=LCE)*dRedvR
            
    def _dlnfdvT(self,R,vR,vT):
        #Calculate a bunch of stuff that we need
        if self._beta == 0.:
            E= vR**2./2.+vT**2./2.+sc.log(R)
            xE= sc.exp(E-.5)
            OE= xE**-1.
            LCE= xE
            dRedvT= xE*vT
        else: #non-flat rotation curve
            E= vR**2./2.+vT**2./2.+1./2./self._beta*R**(2.*self._beta)
            xE= (2.*E/(1.+1./self._beta))**(1./2./self._beta)
            OE= xE**(self._beta-1.)
            LCE= xE**(self._beta+1.)
            dRedvT= xE/2./self._beta/E*vT
        return self._dlnfdRe(R,vR,vT,E=E,xE=xE,OE=OE,LCE=LCE)*dRedvT\
            +self._dlnfdl(R,vR,vT,E=E,xE=xE,OE=OE)*R
           
    def _dlnfdRe(self,R,vR,vT,E=None,xE=None,OE=None,LCE=None):
        """d ln f(x,v) / d R_e"""
        #Calculate a bunch of stuff that we need
        if E is None or xE is None or OE is None or LCE is None:
            if self._beta == 0.:
                E= vR**2./2.+vT**2./2.+sc.log(R)
                xE= sc.exp(E-.5)
                OE= xE**-1.
                LCE= xE
            else: #non-flat rotation curve
                E= vR**2./2.+vT**2./2.+1./2./self._beta*R**(2.*self._beta)
                xE= (2.*E/(1.+1./self._beta))**(1./2./self._beta)
                OE= xE**(self._beta-1.)
                LCE= xE**(self._beta+1.)
        L= R*vT
        sigma2xE= self._surfaceSigmaProfile.sigma2(xE,log=False)
        return (self._surfaceSigmaProfile.surfacemassDerivative(xE,log=True)\
                 -(1.+OE*(L-LCE)/sigma2xE)*self._surfaceSigmaProfile.sigma2Derivative(xE,log=True)\
                 +(L-LCE)/sigma2xE*(self._beta-1.)*xE**(self._beta-2.)\
                 -OE*(self._beta+1.)/sigma2xE*xE**self._beta)

    def _dlnfdl(self,R,vR,vT,E=None,xE=None,OE=None):
        #Calculate a bunch of stuff that we need
        if E is None or xE is None or OE is None:
            if self._beta == 0.:
                E= vR**2./2.+vT**2./2.+sc.log(R)
                xE= sc.exp(E-.5)
                OE= xE**-1.
            else: #non-flat rotation curve
                E= vR**2./2.+vT**2./2.+1./2./self._beta*R**(2.*self._beta)
                xE= (2.*E/(1.+1./self._beta))**(1./2./self._beta)
                OE= xE**(self._beta-1.)
        sigma2xE= self._surfaceSigmaProfile.sigma2(xE,log=False)
        return OE/sigma2xE

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
          
                        xD - disk surface mass scalelength (can be Quantity)
              
                        xS - disk velocity dispersion scalelength (can be Quantity)
                        
                        Sro - disk velocity dispersion at Ro (can be Quantity)
                        
                        Directly given to the 'surfaceSigmaProfile class, so
                        could be anything that class takes

           beta - power-law index of the rotation curve

           correct - if True, correct the DF

           ro= distance from vantage point to GC (kpc; can be Quantity)

           vo= circular velocity at ro (km/s; can be Quantity)

           +DFcorrection kwargs (except for those already specified)

        OUTPUT:

           instance

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
        if _APY_LOADED and isinstance(E,units.Quantity):
            E= E.to(units.km**2/units.s**2).value/self._vo**2.
        if _APY_LOADED and isinstance(L,units.Quantity):
            L= L.to(units.kpc*units.km/units.s).value/self._ro/self._vo
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
        SRE2= self.targetSigma2(xL,log=True,use_physical=False)+correction[1]
        return self._gamma*sc.exp(logsigmaR2-SRE2+self.targetSurfacemass(xL,log=True,use_physical=False)-logSigmaR-sc.exp(logECLE-SRE2)+correction[0])/2./nu.pi

    def sample(self,n=1,rrange=None,returnROrbit=True,returnOrbit=False,
               nphi=1.,los=None,losdeg=True,nsigma=None,maxd=None,
               targetSurfmass=True,targetSigma2=True,**kwargs):
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
           targetSurfmass, targetSigma2= if True, use target surface mass and sigma2 profiles, respectively (there is not much point to doing the latter)
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
            return self.sampleLOS(los,n=n,maxd=maxd,
                                  nsigma=nsigma,targetSurfmass=targetSurfmass,
                                  targetSigma2=targetSigma2)
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
            E*= self._corr.correct(xL,log=False)[1,:]
        E+= ECL
        if not returnROrbit and not returnOrbit:
            out= [[e,l] for e,l in zip(E,Lz)]
        else:
            if not rrange is None \
                    and _APY_LOADED and isinstance(rrange[0],units.Quantity):
                rrange[0]= rrange[0].to(units.kpc).value/self._ro
                rrange[1]= rrange[1].to(units.kpc).value/self._ro
            if not hasattr(self,'_psp'):
                self._psp= PowerSphericalPotential(alpha=2.-self._beta,normalize=True).toPlanar()
            out= []
            for ii in range(n):
                try:
                    wR, rap, rperi= self._ELtowRRapRperi(E[ii],Lz[ii])
                except ValueError: #pragma: no cover
                    continue
                TR= 2.*math.pi/wR
                tr= stats.uniform.rvs()*TR
                if tr > TR/2.:
                    tr-= TR/2.
                    thisOrbit= Orbit([rperi,0.,Lz[ii]/rperi])
                else:
                    thisOrbit= Orbit([rap,0.,Lz[ii]/rap])
                thisOrbit.integrate(sc.array([0.,tr]),self._psp)
                if returnOrbit:
                    vxvv= thisOrbit(tr)._orb.vxvv
                    thisOrbit= Orbit(vxvv=sc.array([vxvv[0],vxvv[1],vxvv[2],
                                                    stats.uniform.rvs()*math.pi*2.]).reshape(4))
                else:
                    thisOrbit= thisOrbit(tr)
                kappa= _kappa(thisOrbit._orb.vxvv[0],self._beta)
                if not rrange == None:
                    if thisOrbit._orb.vxvv[0] < rrange[0] \
                            or thisOrbit._orb.vxvv[0] > rrange[1]:
                        continue
                mult= sc.ceil(kappa/wR*nphi)-1.
                kappawR= kappa/wR*nphi-mult
                while mult > 0:
                    if returnOrbit:
                        out.append(Orbit(vxvv=sc.array([vxvv[0],vxvv[1],
                                                        vxvv[2],
                                                        stats.uniform.rvs()*math.pi*2.]).reshape(4)))
                    else:
                        out.append(thisOrbit)
                    mult-= 1
                if stats.uniform.rvs() > kappawR:
                    continue
                out.append(thisOrbit)
        #Recurse to get enough
        if len(out) < n*nphi:
            out.extend(self.sample(n=int(n-len(out)/nphi),rrange=rrange,
                                   returnROrbit=returnROrbit,
                                   returnOrbit=returnOrbit,nphi=nphi))
        if len(out) > n*nphi:
            out= out[0:int(n*nphi)]
        if kwargs.get('use_physical',True) and \
                self._roSet and self._voSet:
            if isinstance(out[0],Orbit):
                dum= [o.turn_physical_on(ro=self._ro,vo=self._vo) for o in out]
        return out

    def _dlnfdR(self,R,vR,vT):
        #Calculate a bunch of stuff that we need
        E, L= vRvTRToEL(vR,vT,R,self._beta,self._dftype)
        if self._beta == 0.:
            xL= L
            dRldR= vT
            ECL= sc.log(xL)+0.5
            dECLEdR= 0.
        else: #non-flat rotation curve
            xL= L**(1./(self._beta+1.))
            dRldR= L**(1./(self._beta+1.))/R/(self._beta+1.)
            ECL= 0.5*(1./self._beta+1.)*xL**(2.*self._beta)
            dECLdRl= (1.+self._beta)*xL**(2.*self._beta-1)
            dEdR= R**(2.*self._beta-1.)
            dECLEdR= dECLdRl*dRldR-dEdR
        sigma2xL= self._surfaceSigmaProfile.sigma2(xL,log=False)
        return (self._surfaceSigmaProfile.surfacemassDerivative(xL,log=True)\
                 -(1.+(ECL-E)/sigma2xL)*self._surfaceSigmaProfile.sigma2Derivative(xL,log=True))*dRldR\
                 +dECLEdR/sigma2xL
    
    def _dlnfdvR(self,R,vR,vT):
        #Calculate a bunch of stuff that we need
        E, L= vRvTRToEL(vR,vT,R,self._beta,self._dftype)
        if self._beta == 0.:
            xL= L
        else: #non-flat rotation curve
            xL= L**(1./(self._beta+1.))
        sigma2xL= self._surfaceSigmaProfile.sigma2(xL,log=False)
        return -vR/sigma2xL
    
    def _dlnfdvT(self,R,vR,vT):
        #Calculate a bunch of stuff that we need
        E, L= vRvTRToEL(vR,vT,R,self._beta,self._dftype)
        if self._beta == 0.:
            xL= L
            dRldvT= R
            ECL= sc.log(xL)+0.5
            dECLEdvT= 1./vT-vT
        else: #non-flat rotation curve
            xL= L**(1./(self._beta+1.))
            dRldvT= L**(1./(self._beta+1.))/vT/(self._beta+1.)
            ECL= 0.5*(1./self._beta+1.)*xL**(2.*self._beta)
            dECLdRl= (1.+self._beta)*xL**(2.*self._beta-1)
            dEdvT= vT
            dECLEdvT= dECLdRl*dRldvT-dEdvT
        sigma2xL= self._surfaceSigmaProfile.sigma2(xL,log=False)
        return (self._surfaceSigmaProfile.surfacemassDerivative(xL,log=True)\
                 -(1.+(ECL-E)/sigma2xL)*self._surfaceSigmaProfile.sigma2Derivative(xL,log=True))*dRldvT\
                 +dECLEdvT/sigma2xL
    
class schwarzschilddf(shudf):
    """Schwarzschild's df"""
    def __init__(self,surfaceSigma=expSurfaceSigmaProfile,
                 profileParams=(1./3.,1.0,0.2),
                 correct=False,
                 beta=0.,**kwargs):
        """
        NAME:
           __init__
        PURPOSE:
           Initialize a Schwarzschild DF
        INPUT:
           surfaceSigma - instance or class name of the target 
                      surface density and sigma_R profile 
                      (default: both exponential)
           profileParams - parameters of the surface and sigma_R profile:
                      (xD,xS,Sro) where
          
                        xD - disk surface mass scalelength (can be Quantity)
              
                        xS - disk velocity dispersion scalelength (can be Quantity)
                        
                        Sro - disk velocity dispersion at Ro (can be Quantity)
                        
                        Directly given to the 'surfaceSigmaProfile class, so
                        could be anything that class takes

           beta - power-law index of the rotation curve

           correct - if True, correct the DF

           ro= distance from vantage point to GC (kpc; can be Quantity)

           vo= circular velocity at ro (km/s; can be Quantity)

           +DFcorrection kwargs (except for those already specified)

        OUTPUT:

           instance

        HISTORY:

            2017-09-17 - Written - Bovy (UofT)

        """
        # Schwarzschild == Shu w/ energy computed in epicycle approx.
        # so all functions are the same as in Shu, only thing different is
        # how E is computed
        return diskdf.__init__(self,surfaceSigma=surfaceSigma,
                               profileParams=profileParams,
                               correct=correct,dftype='schwarzschild',
                               beta=beta,**kwargs)
    

def _surfaceIntegrand(vR,vT,R,df,logSigmaR,logsigmaR2,sigmaR1,gamma):
    """Internal function that is the integrand for the surface mass integration"""
    E,L= _vRpvTpRToEL(vR,vT,R,df._beta,sigmaR1,gamma,df._dftype)
    return df.eval(E,L,logSigmaR,logsigmaR2)*2.*nu.pi/df._gamma #correct

def _sigma2surfaceIntegrand(vR,vT,R,df,logSigmaR,logsigmaR2,sigmaR1,gamma):
    """Internal function that is the integrand for the sigma-squared times
    surface mass integration"""
    E,L= _vRpvTpRToEL(vR,vT,R,df._beta,sigmaR1,gamma,df._dftype)
    return vR**2.*df.eval(E,L,logSigmaR,logsigmaR2)*2.*nu.pi/df._gamma #correct

def _vmomentsurfaceIntegrand(vR,vT,R,df,logSigmaR,logsigmaR2,sigmaR1,gamma,
                             n,m):
    """Internal function that is the integrand for the velocity moment times
    surface mass integration"""
    E,L= _vRpvTpRToEL(vR,vT,R,df._beta,sigmaR1,gamma,df._dftype)
    return vR**n*vT**m*df.eval(E,L,logSigmaR,logsigmaR2)*2.*nu.pi/df._gamma #correct

def _vmomentderivsurfaceIntegrand(vR,vT,R,df,logSigmaR,logsigmaR2,sigmaR1,
                                  gamma,n,m,deriv):
    """Internal function that is the integrand for the derivative of velocity 
    moment times surface mass integration"""
    E,L= _vRpvTpRToEL(vR,vT,R,df._beta,sigmaR1,gamma,df._dftype)
    if deriv.lower() == 'r':
        return vR**n*vT**m*df.eval(E,L,logSigmaR,logsigmaR2)*2.*nu.pi/df._gamma*df._dlnfdR(R,vR*sigmaR1,vT*sigmaR1/gamma) #correct
    else:
        return 0.

def _vRpvTpRToEL(vR,vT,R,beta,sigmaR1,gamma,dftype='dehnen'):
    """Internal function that calculates E and L given velocities normalized by the velocity dispersion"""
    vR*= sigmaR1
    vT*= sigmaR1/gamma
    return vRvTRToEL(vR,vT,R,beta,dftype)

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


class DFcorrection(object):
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
        if not 'surfaceSigmaProfile' in kwargs:
            raise DFcorrectionError("surfaceSigmaProfile not given")
        else:
            self._surfaceSigmaProfile= kwargs['surfaceSigmaProfile']
        self._rmax= kwargs.get('rmax',5.)
        self._niter= kwargs.get('niter',20)
        if not 'npoints' in kwargs:
            if 'corrections' in kwargs:
                self._npoints= kwargs['corrections'].shape[0]
            else: #pragma: no cover
                self._npoints= 151 #would take too long to cover
        else:
            self._npoints= kwargs['npoints']
        self._dftype= kwargs.get('dftype',dehnendf)
        self._beta= kwargs.get('beta',0.)
        self._rs= sc.linspace(_RMIN,self._rmax,self._npoints)
        self._interp_k= kwargs.get('interp_k',_INTERPDEGREE)
        if 'corrections' in kwargs:
            self._corrections= kwargs['corrections']
            if not len(self._corrections) == self._npoints:
                raise DFcorrectionError("Number of corrections has to be equal to the number of points npoints")
        else:
            self._savedir= kwargs.get('savedir',_CORRECTIONSDIR)
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
                            '%6.4f_%i_%6.4f_%i.sav'
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
        if isinstance(R,nu.ndarray):
            out= nu.empty((2,len(R)))
            #R < _RMIN
            rmin_indx= (R < _RMIN)
            if nu.sum(rmin_indx) > 0:
                out[0,rmin_indx]= math.log(self._corrections[0,0])\
                                  +self._surfaceDerivSmallR*(R[rmin_indx]-_RMIN)
                out[1,rmin_indx]= math.log(self._corrections[0,1])\
                                  +self._sigma2DerivSmallR*(R[rmin_indx]-_RMIN)
            #R > 2rmax
            rmax_indx= (R > (2.*self._rmax))
            if nu.sum(rmax_indx) > 0:
                out[:,rmax_indx]= 0.
            #'normal' R
            r_indx= (R >= _RMIN)*(R <= (2.*self._rmax))
            if nu.sum(r_indx) > 0:
                out[0,r_indx]= self._surfaceInterpolate(R[r_indx])
                out[1,r_indx]= self._sigma2Interpolate(R[r_indx])
            if log: return out
            else: return nu.exp(out)
        if R < _RMIN:
            out= sc.array([sc.log(self._corrections[0,0])+self._surfaceDerivSmallR*(R-_RMIN),
                           sc.log(self._corrections[0,1])+self._sigma2DerivSmallR*(R-_RMIN)])
        elif R > (2.*self._rmax):
            out= sc.array([0.,0.])
        else:
            if _SCIPYVERSION >= 0.9:
                out= sc.array([self._surfaceInterpolate(R),
                               self._sigma2Interpolate(R)])
            else: #pragma: no cover
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
                out= sc.array([self._surfaceInterpolate(R,nu=1),
                               self._sigma2Interpolate(R,nu=1)])
            else: #pragma: no cover
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
                thisSurface= currentDF.surfacemass(self._rs[jj],
                                                   use_physical=False)
                newcorrections[jj,0]= currentDF.targetSurfacemass(self._rs[jj],use_physical=False)/thisSurface
                newcorrections[jj,1]= currentDF.targetSigma2(self._rs[jj],use_physical=False)*thisSurface\
                    /currentDF.sigma2surfacemass(self._rs[jj],
                                                 use_physical=False)
                #print(jj, newcorrections[jj,:])
            corrections*= newcorrections
        #Save
        picklethis= []
        for arr in list(corrections):
            picklethis.append([float(a) for a in arr])
        save_pickles(self._savefilename,picklethis) #We pickle a list for platform-independence)
        return corrections
    
class DFcorrectionError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def vRvTRToEL(vR,vT,R,beta,dftype='dehnen'):
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
    if dftype == 'schwarzschild':
        # Compute E in the epicycle approximation
        gamma= sc.sqrt(2./(1.+beta))
        L= R*vT
        if beta == 0.:
            xL= L
        else: #non-flat rotation curve
            xL= L**(1./(beta+1.))   
        return (0.5*vR**2.+0.5*gamma**2.*(vT-R**beta)**2.
                +xL**(2.*beta)/2.+axipotential(xL,beta=beta),
                L)
    else:
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
        if nu.any(R == 0.):
            out= nu.empty(R.shape)
            out[R == 0.]= math.log(_RMIN)
            out[R != 0.]= nu.log(R[R != 0.])
            return out
        else:
            return nu.log(R)
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
        return math.log(x)+surfaceSigma.surfacemass(x,log=True)
    else:
        return math.log(x)+surfaceSigma.surfacemass(x,log=True)+dfcorr.correct(x)[0]

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
    return math.sqrt(2.*(1.+beta))*R**(beta-1)

def _dlToRphi(d,l):
    """Convert d and l to R and phi, l is in radians"""
    R= math.sqrt(1.+d**2.-2.*d*math.cos(l))
    if R == 0.:
        R+= 0.0001
        d+= 0.0001
    if 1./math.cos(l) < d and math.cos(l) > 0.:
        theta= math.pi-math.asin(d/R*math.sin(l))
    else:
        theta= math.asin(d/R*math.sin(l))
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
        xE= (2.*E/(1.+1./diskdf._beta))**(1./2./diskdf._beta)
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
                        R,df._beta,df._dftype))

def _marginalizeVperpIntegrandSinAlphaSmall(vT,df,R,cosalpha,tanalpha,
                                            vlos,vcirc,sigma):
    return df(*vRvTRToEL(tanalpha*vT*sigma-vlos/cosalpha,vT*sigma+vcirc,
                        R,df._beta,df._dftype))

