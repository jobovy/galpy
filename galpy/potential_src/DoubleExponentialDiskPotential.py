###############################################################################
#   DoubleExponentialDiskPotential.py: class that implements the double
#                                      exponential disk potential
#
#                                      rho(R,z) = rho_0 e^-R/h_R e^-|z|/h_z
###############################################################################
import numpy as nu
from scipy import special, integrate
from Potential import Potential
_TOL= 1.4899999999999999e-15
_MAXITER= 20
class DoubleExponentialDiskPotential(Potential):
    """Class that implements the double exponential disk potential
    rho(R,z) = rho_0 e^-R/h_R e^-|z|/h_z"""
    def __init__(self,amp=1.,ro=1.,hr=1./3.,hz=1./16.,
                 maxiter=_MAXITER,tol=0.001,normalize=False):
        """
        NAME:
           __init__
        PURPOSE:
           initialize a double-exponential disk potential
        INPUT:
           amp - amplitude to be applied to the potential (default: 1)
           hr - disk scale-length in terms of ro
           hz - scale-height
           ro - representative disk-radius at which rhoo is given
           tol - relative accuracy of potential-evaluations
           maxiter - scipy.integrate keyword
           normalize - if True, normalize such that vc(1.,0.)=1., or, if 
                       given as a number, such that the force is this fraction 
                       of the force necessary to make vc(1.,0.)=1.
        OUTPUT:
           DoubleExponentialDiskPotential object
        HISTORY:
           2010-04-16 - Written - Bovy (NYU)
        """
        Potential.__init__(self,amp=amp)
        self._ro= ro
        self._hr= hr
        self._hz= hz
        self._alpha= 1./self._hr
        self._beta= 1./self._hz
        self._gamma= self._alpha/self._beta
        self._maxiter= maxiter
        self._tol= tol
        self._zforceNotSetUp= True #We have not calculated a typical Kz yet
        if normalize:
            self.normalize(normalize)
        
    def _evaluate(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _evaluate
        PURPOSE:
           evaluate the potential at (R,z)
        INPUT:
           R - Cylindrical Galactocentric radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           potential at (R,z)
        HISTORY:
           2010-04-16 - Written - Bovy (NYU)
        DOCTEST:
           >>> doubleExpPot= DoubleExponentialDiskPotential()
           >>> r= doubleExpPot(1.,0) #doctest: +ELLIPSIS
           ...
           >>> assert( r+1.89595350484)**2.< 10.**-6.
        """
        notConvergedSmall= True
        notConvergedLarge= True
        smallkIntegral= integrate.quadrature(_doubleExponentialDiskPotentialPotentialIntegrandSmallk,
                                             0.,1./self._gamma,
                                             args=(self._alpha*R,
                                                   self._beta*nu.fabs(z),
                                                   self._gamma),tol=_TOL,
                                             maxiter=self._maxiter,
                                             vec_func=False)
        largekIntegral= integrate.quadrature(_doubleExponentialDiskPotentialPotentialIntegrandLargek,
                                             0.,self._gamma,
                                             args=(self._alpha*R,
                                                   self._beta*nu.fabs(z),
                                                   self._gamma),tol=_TOL,
                                             maxiter=self._maxiter,
                                             vec_func=False)
        maxiterFactorSmall= 2.
        maxiterFactorLarge= 2.
        if nu.fabs(smallkIntegral[1]/(smallkIntegral[0]+largekIntegral[0])) <= self._tol:
            notConvergedSmall= False
        if nu.fabs(largekIntegral[1]/(largekIntegral[0]+smallkIntegral[0])) <= self._tol:
            notConvergedLarge= False
        while notConvergedSmall or notConvergedLarge:
            if notConvergedSmall:
                smallkIntegral= integrate.quadrature(_doubleExponentialDiskPotentialPotentialIntegrandSmallk,
                                                     0.,1./self._gamma,
                                                     args=(self._alpha*R,
                                                           self._beta*nu.fabs(z),
                                                           self._gamma),tol=_TOL,
                                                     maxiter= maxiterFactorSmall*self._maxiter,
                                                     vec_func=False)
                if nu.fabs(smallkIntegral[1]/(smallkIntegral[0]+largekIntegral[0])) > self._tol:
                    maxiterFactorSmall*= 2
                else:
                    notConvergedSmall= False
            if notConvergedLarge:
                largekIntegral= integrate.quadrature(_doubleExponentialDiskPotentialPotentialIntegrandLargek,
                                                     0.,self._gamma,
                                                     args=(self._alpha*R,
                                                           self._beta*nu.fabs(z),
                                                           self._gamma),tol=_TOL,
                                                     maxiter=maxiterFactorLarge*self._maxiter,
                                                     vec_func=False)
                if nu.fabs(largekIntegral[1]/(largekIntegral[0]+smallkIntegral[0])) > self._tol:
                    maxiterFactorLarge*= 2
                else:
                    notConvergedLarge= False
        return -4.*nu.pi/self._alpha/self._beta*(smallkIntegral[0]+largekIntegral[0])
    
    def _Rforce(self,R,z,phi=0.,t=0.):
        """
        NAME:
           Rforce
        PURPOSE:
           evaluate radial force K_R  (R,z)
        INPUT:
           R - Cylindrical Galactocentric radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           K_R (R,z)
        HISTORY:
           2010-04-16 - Written - Bovy (NYU)
        DOCTEST:
        """
        notConvergedSmall= True
        notConvergedLarge= True
        smallkIntegral= integrate.quadrature(_doubleExponentialDiskPotentialRForceIntegrandSmallk,
                                             0.,1./self._gamma,
                                             args=(self._alpha*R,
                                                   self._beta*nu.fabs(z),
                                                   self._gamma),tol=_TOL,
                                             maxiter= 2*self._maxiter,
                                                 vec_func=False)
        largekIntegral= integrate.quadrature(_doubleExponentialDiskPotentialRForceIntegrandLargek,
                                             0.,self._gamma,
                                             args=(self._alpha*R,
                                                   self._beta*nu.fabs(z),
                                                   self._gamma),tol=_TOL,
                                             maxiter= 2*self._maxiter,
                                             vec_func=False)
        maxiterFactorSmall= 4.
        maxiterFactorLarge= 4.
        if nu.fabs(smallkIntegral[1]/(smallkIntegral[0]+largekIntegral[0])) <= self._tol:
            notConvergedSmall= False
        if nu.fabs(largekIntegral[1]/(largekIntegral[0]+smallkIntegral[0])) <= self._tol:
            notConvergedLarge= False
        while notConvergedSmall or notConvergedLarge:
            if notConvergedSmall:
                smallkIntegral= integrate.quadrature(_doubleExponentialDiskPotentialRForceIntegrandSmallk,
                                                     0.,1./self._gamma,
                                                     args=(self._alpha*R,
                                                           self._beta*nu.fabs(z),
                                                           self._gamma),
                                                     tol=_TOL,
                                                     maxiter= maxiterFactorSmall*self._maxiter,
                                                     vec_func=False)
                if nu.fabs(smallkIntegral[1]/(smallkIntegral[0]+largekIntegral[0])) > self._tol:
                    maxiterFactorSmall*= 2
                else:
                    notConvergedSmall= False
            if notConvergedLarge:
                largekIntegral= integrate.quadrature(_doubleExponentialDiskPotentialRForceIntegrandLargek,
                                                     0.,self._gamma,
                                                     args=(self._alpha*R,
                                                           self._beta*nu.fabs(z),
                                                           self._gamma),
                                                     tol=_TOL,
                                                     maxiter=maxiterFactorLarge*self._maxiter,
                                                     vec_func=False)
            if nu.fabs(largekIntegral[1]/(largekIntegral[0]+smallkIntegral[0])) > self._tol:
                maxiterFactorLarge*= 2
            else:
                notConvergedLarge= False
        return -4.*nu.pi/self._beta*(smallkIntegral[0]+largekIntegral[0])
    
    def _zforce(self,R,z,phi=0.,t=0.):
        """
        NAME:
           zforce
        PURPOSE:
           evaluate vertical force K_z  (R,z)
        INPUT:
           R - Cylindrical Galactocentric radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           K_z (R,z)
        HISTORY:
           2010-04-16 - Written - Bovy (NYU)
        DOCTEST:
        """
        if self._zforceNotSetUp:
            self._zforceNotSetUp= False
            self._typicalKz= self._zforce(self._ro,self._hz)
        notConvergedSmall= True
        notConvergedLarge= True
        smallkIntegral= integrate.quadrature(_doubleExponentialDiskPotentialzForceIntegrandSmallk,
                                             0.,1./self._gamma,
                                             args=(self._alpha*R,
                                                   self._beta*nu.fabs(z),
                                                   self._gamma),tol=_TOL,
                                             maxiter= 2*self._maxiter,
                                             vec_func=False)
        largekIntegral= integrate.quadrature(_doubleExponentialDiskPotentialzForceIntegrandLargek,
                                             0.,self._gamma,
                                             args=(self._alpha*R,
                                                   self._beta*nu.fabs(z),
                                                   self._gamma),tol=_TOL,
                                             maxiter=2*self._maxiter,
                                             vec_func=False)
        maxiterFactorSmall= 4.
        maxiterFactorLarge= 4.
        try:
            if smallkIntegral[1]/self._typicalKz <= self._tol:
                notConvergedSmall= False
        except AttributeError:
            if nu.fabs(smallkIntegral[1]/(smallkIntegral[0]+largekIntegral[0])) <= self._tol:
                notConvergedSmall= False
        try:
            if largekIntegral[1]/self._typicalKz <= self._tol:
                notConvergedLarge= False
        except AttributeError:                
            if nu.fabs(largekIntegral[1]/(largekIntegral[0]+smallkIntegral[0])) <= self._tol:
                notConvergedLarge= False
        while notConvergedSmall or notConvergedLarge:
            if notConvergedSmall:
                smallkIntegral= integrate.quadrature(_doubleExponentialDiskPotentialzForceIntegrandSmallk,
                                                     0.,1./self._gamma,
                                                     args=(self._alpha*R,
                                                           self._beta*nu.fabs(z),
                                                           self._gamma),tol=_TOL,
                                                     maxiter= maxiterFactorSmall*self._maxiter,
                                                     vec_func=False)
                try:
                    if smallkIntegral[1]/self._typicalKz > self._tol:
                        maxiterFactorSmall*= 2
                    else:
                        notConvergedSmall= False
                except AttributeError:
                    if nu.fabs(smallkIntegral[1]/(smallkIntegral[0]+largekIntegral[0])) > self._tol:
                        maxiterFactorSmall*= 2
                    else:
                        notConvergedSmall= False
            if notConvergedLarge:
                largekIntegral= integrate.quadrature(_doubleExponentialDiskPotentialzForceIntegrandLargek,
                                                     0.,self._gamma,
                                                     args=(self._alpha*R,
                                                           self._beta*nu.fabs(z),
                                                           self._gamma),tol=_TOL,
                                                     maxiter=maxiterFactorLarge*self._maxiter,
                                                     vec_func=False)
                try:
                    if largekIntegral[1]/self._typicalKz > self._tol:
                        maxiterFactorLarge*= 2
                    else:
                        notConvergedLarge= False
                except AttributeError:
                    if largekIntegral[1]/(largekIntegral[0]+smallkIntegral[0]) > self._tol:
                        maxiterFactorLarge*= 2
                    else:
                        notConvergedLarge= False
        if z < 0.:
            return 4.*nu.pi/self._beta*(smallkIntegral[0]+largekIntegral[0])
        else:
            return -4.*nu.pi/self._beta*(smallkIntegral[0]+largekIntegral[0])

    def _dens(self,R,z,phi=0.,t=0.):
        """
        NAME:
           dens
        PURPOSE:
           evaluate the density
        INPUT:
           R - Cylindrical Galactocentric radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           rho (R,z)
        HISTORY:
           2010-08-08 - Written - Bovy (NYU)
        """
        return nu.exp(-self._alpha*R-self._beta*nu.fabs(z))

def _doubleExponentialDiskPotentialPotentialIntegrandSmallk(k,R,z,gamma):
    """Internal function that gives the integrand for the double
    exponential disk potential for k < 1/gamma"""
    gammak= gamma*k
    return special.jn(0,k*R)*(1.+k**2.)**-1.5*(nu.exp(-gammak*z)
                                               -gammak*nu.exp(-z))/(1.-gammak**2.)

def _doubleExponentialDiskPotentialPotentialIntegrandLargek(k,R,z,gamma):
    """Internal function that gives the integrand for the double
    exponential disk potential for k > 1/gamma"""
    return 1./k**2.*_doubleExponentialDiskPotentialPotentialIntegrandSmallk(1./k,R,z,gamma)

def _doubleExponentialDiskPotentialRForceIntegrandSmallk(k,R,z,gamma):
    """Internal function that gives the integrand for the double
    exponential disk radial force for k < 1/gamma"""
    gammak= gamma*k
    return k*special.jn(1,k*R)*(1.+k**2.)**-1.5*(nu.exp(-gammak*z)
                                               -gammak*nu.exp(-z))/(1.-gammak**2.)

def _doubleExponentialDiskPotentialRForceIntegrandLargek(k,R,z,gamma):
    """Internal function that gives the integrand for the double
    exponential disk radial force for k > 1/gamma"""
    return 1./k**2.*_doubleExponentialDiskPotentialRForceIntegrandSmallk(1./k,R,z,gamma)

def _doubleExponentialDiskPotentialzForceIntegrandSmallk(k,R,z,gamma):
    """Internal function that gives the integrand for the double
    exponential disk vertical force for k < 1/gamma"""
    gammak= gamma*k
    return k*special.jn(0,k*R)*(1.+k**2.)**-1.5*(nu.exp(-gammak*z)
                                                 -nu.exp(-z))/(1.-gammak**2.)

def _doubleExponentialDiskPotentialzForceIntegrandLargek(k,R,z,gamma):
    """Internal function that gives the integrand for the double
    exponential disk vertical force for k > 1/gamma"""
    return 1./k**2.*_doubleExponentialDiskPotentialzForceIntegrandSmallk(1./k,R,z,gamma)


if __name__ == '__main__':
    print "doctesting ..."
    import doctest
    doctest.testmod(verbose=True)
        
    import time, sys
    import numpy as nu
    nTrials = 100
    doubleExpPot= DoubleExponentialDiskPotential()
    print "Timing ..."
    start= time.time()
    for ii in range(nTrials):
        doubleExpPot(nu.random.random()*2./3.+2./3.,
                     nu.random.random()*1./4.-1./8.)
    deltatpot= time.time()-start
    print "Potential evaluation @ %.3f s per evaluation" % (deltatpot/nTrials)
    #sys.exit(-1)
    start= time.time()
    for ii in range(nTrials):
        doubleExpPot.Rforce(nu.random.random()*2./3.+2./3.,
                            nu.random.random()*1./4.-1./8.)
    deltatRforce= time.time()-start

    #doubleExpPot._zforceNotSetUp= False
    start= time.time()
    for ii in range(nTrials):
        doubleExpPot.zforce(nu.random.random()*2./3.+2./3.,
                            nu.random.random()*1./4.-1./8.)
    deltatzforce= time.time()-start
    print "Potential evaluation @ %.3f s per evaluation" % (deltatpot/nTrials)
    print "Radial force evaluation @ %.3f s per evaluation" % (deltatRforce/nTrials)
    print "Vertical force evaluation @ %.3f s per evaluation" % (deltatzforce/nTrials)
