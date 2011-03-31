###############################################################################
#   evolveddiskdf.py: module that builds a distribution function as a 
#                     steady-state DF + subsequent evolution
#
#   This module contains the following classes:
#
#      evolveddiskdf - top-level class that represents a distribution function
###############################################################################
_EPSREL=10.**-14.
_NSIGMA= 4.
_NTS= 1000
_RMIN=10.**-10.
_MAXD_REJECTLOS= 4.
import math as m
import numpy as nu
from scipy import integrate
from galpy.orbit import Orbit
_DEGTORAD= m.pi/180.
class evolveddiskdf:
    """Class that represents a diskdf as initial DF + subsequent secular evolution"""
    def __init__(self,initdf,pot,to=0.):
        """
        NAME:
           __init__
        PURPOSE:
           initialize
        INPUT:
           initdf - the df at the start of the evolution (at to)
           pot - potential to integrate orbits in
           to= initial time (time at which initdf is evaluated; orbits are 
               integrated from current t back to to)
        OUTPUT:
        HISTORY:
           2011-03-30 - Written - Bovy (NYU)
        """
        self._initdf= initdf
        self._pot= pot
        self._to= to

    def __call__(self,*args):
        """
        NAME:
           __call__
        PURPOSE:
           evaluate the distribution function
        INPUT:
           Orbit instance:
              a) Orbit instance alone: use initial state and t=0
              b) Orbit instance + t: Orbit instance *NOT* called (i.e., Orbit's initial condition is used, call Orbit yourself)
        OUTPUT:
           DF(orbit,t)
        HISTORY:
           2011-03-30 - Written - Bovy (NYU)
        """
        if isinstance(args[0],Orbit):
            if len(args) == 1:
                t= 0.
            else:
                t= args[1]
        else:
            raise IOError("Input to __call__ not understood; this has to be an Orbit instance with optional time")
        #Integrate back
        if self._to == t:
            return self._initdf(args[0])
        ts= nu.linspace(0.,self._to-t,_NTS)
        o= args[0]
        #integrate orbit
        o.integrate(ts,self._pot)
        #Now evaluate the DF
        return self._initdf(o(self._to-t))

    def vmomentsurfacemass(self,R,phi,n,m,t=0.,nsigma=None,deg=False,
                           epsrel=-1.e-02,epsabs=1.e-05):
        """
        NAME:
           vmomentsurfacemass
        PURPOSE:
           calculate the an arbitrary moment of the velocity distribution 
           at (R,phi) times the surfacmass
        INPUT:
           R - radius at which to calculate the moment(/ro)
           phi= azimuth (rad unless deg=True)
           n - vR^n
           m - vT^m
           t= time at which to evaluate the DF
        OPTIONAL INPUT:
           nsigma - number of sigma to integrate the velocities over (based on an estimate, so be generous)
           deg= azimuth is in degree (default=False)
           epsrel, epsabs - scipy.integrate keywords (the integration 
                            calculates the ration of this vmoment to that 
                            of the initial DF)
        OUTPUT:
           <vR^n vT^m  x surface-mass> at R,phi
        HISTORY:
           2011-03-30 - Written - Bovy (NYU)
        """
        if deg: az= phi*_DEGTORAD
        else: az= phi
        if nsigma == None: nsigma= _NSIGMA
        sigmaR1= nu.sqrt(self._initdf.sigmaR2(R))
        sigmaT1= nu.sqrt(self._initdf.sigmaT2(R))
        meanvR= self._initdf.meanvR(R)
        meanvT= self._initdf.meanvT(R)
        #Calculate the initdf moment and then calculate the ratio
        initvmoment= self._initdf.vmomentsurfacemass(R,n,m,nsigma=nsigma)
        if initvmoment == 0.: initvmoment= 1.
        norm= sigmaR1**(n+1)*sigmaT1**(m+1)*initvmoment
        return integrate.dblquad(_vmomentsurfaceIntegrand,
                                 meanvT/sigmaT1-nsigma,
                                 meanvT/sigmaT1+nsigma,
                                 lambda x: meanvR/sigmaR1-nsigma, 
                                 lambda x: meanvR/sigmaR1+nsigma, 
                                 (R,az,self,n,m,sigmaR1,sigmaT1,t,initvmoment),
                                 epsrel=epsrel,epsabs=epsabs)[0]*norm

def _vmomentsurfaceIntegrand(vR,vT,R,az,df,n,m,sigmaR1,sigmaT1,t,initvmoment):
    """Internal function that is the integrand for the velocity moment times
    surface mass integration"""
    o= Orbit([R,vR*sigmaR1,vT*sigmaT1,az])
    return vR**n*vT**m*df(o,t)/initvmoment

