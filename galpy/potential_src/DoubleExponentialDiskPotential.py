###############################################################################
#   DoubleExponentialDiskPotential.py: class that implements the double
#                                      exponential disk potential
#
#                                      rho(R,z) = rho_0 e^-R/h_R e^-|z|/h_z
###############################################################################
import numpy as nu
import warnings
from scipy import special, integrate
from galpy.util import galpyWarning
from Potential import Potential
from PowerSphericalPotential import KeplerPotential
_TOL= 1.4899999999999999e-15
_MAXITER= 20
class DoubleExponentialDiskPotential(Potential):
    """Class that implements the double exponential disk potential
    rho(R,z) = rho_0 e^-R/h_R e^-|z|/h_z"""
    def __init__(self,amp=1.,ro=1.,hr=1./3.,hz=1./16.,
                 maxiter=_MAXITER,tol=0.001,normalize=False,
                 new=True,kmaxFac=2.,glorder=10):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a double-exponential disk potential

        INPUT:

           amp - amplitude to be applied to the potential (default: 1)

           hr - disk scale-length in terms of ro

           hz - scale-height

           tol - relative accuracy of potential-evaluations

           maxiter - scipy.integrate keyword

           normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.

        OUTPUT:

           DoubleExponentialDiskPotential object

        HISTORY:

           2010-04-16 - Written - Bovy (NYU)

           2013-01-01 - Re-implemented using faster integration techniques - Bovy (IAS)

        """
        Potential.__init__(self,amp=amp)
        self.hasC= True
        self._new= new
        self._kmaxFac= kmaxFac
        self._glorder= glorder
        self._ro= ro
        self._hr= hr
        self._hz= hz
        self._alpha= 1./self._hr
        self._beta= 1./self._hz
        self._gamma= self._alpha/self._beta
        self._maxiter= maxiter
        self._tol= tol
        self._zforceNotSetUp= True #We have not calculated a typical Kz yet
        #Setup j0 zeros etc.
        self._glx, self._glw= nu.polynomial.legendre.leggauss(self._glorder)
        self._nzeros=100
        #j0 for potential and z
        self._j0zeros= nu.zeros(self._nzeros+1)
        self._j0zeros[1:self._nzeros+1]= special.jn_zeros(0,self._nzeros)
        self._dj0zeros= self._j0zeros-nu.roll(self._j0zeros,1)
        self._dj0zeros[0]= self._j0zeros[0]
        #j1 for R
        self._j1zeros= nu.zeros(self._nzeros+1)
        self._j1zeros[1:self._nzeros+1]= special.jn_zeros(1,self._nzeros)
        self._dj1zeros= self._j1zeros-nu.roll(self._j1zeros,1)
        self._dj1zeros[0]= self._j1zeros[0]
        #j2 for R2deriv
        self._j2zeros= nu.zeros(self._nzeros+1)
        self._j2zeros[1:self._nzeros+1]= special.jn_zeros(2,self._nzeros)
        self._dj2zeros= self._j2zeros-nu.roll(self._j2zeros,1)
        self._dj2zeros[0]= self._j2zeros[0]
        if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)):
            self.normalize(normalize)
        #Load Kepler potential for large R
        self._kp= KeplerPotential(normalize=4.*nu.pi/self._alpha**2./self._beta)

    def _evaluate(self,R,z,phi=0.,t=0.,dR=0,dphi=0):
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
           2012-12-26 - New method using Gaussian quadrature between zeros - Bovy (IAS)
        DOCTEST:
           >>> doubleExpPot= DoubleExponentialDiskPotential()
           >>> r= doubleExpPot(1.,0) #doctest: +ELLIPSIS
           ...
           >>> assert( r+1.89595350484)**2.< 10.**-6.
        """
        if dR == 1 and dphi == 0:
            return -self._Rforce(R,z,phi=phi,t=t)
        elif dR == 0 and dphi == 1:
            return -self._phiforce(R,z,phi=phi,t=t)
        elif dR == 2 and dphi == 0:
            return self._R2deriv(R,z,phi=phi,t=t)
        elif dR != 0 and dphi != 0:
            warnings.warn("High-order derivatives for DoubleExponentialDiskPotential not implemented",galpyWarning)
            return None
        if self._new:
            if isinstance(R,float):
                floatIn= True
                R= nu.array([R])
                z= nu.array([z])
            else:
                floatIn= False
            out= nu.empty(len(R))
            indx= (R <= 6.)
            out[True-indx]= self._kp(R[True-indx],z[True-indx])
            R4max= nu.copy(R)
            R4max[(R < 1.)]= 1.
            kmax= self._kmaxFac*self._beta
            for jj in range(len(R)):
                if not indx[jj]: continue
                maxj0zeroIndx= nu.argmin((self._j0zeros-kmax*R4max[jj])**2.) #close enough
                ks= nu.array([0.5*(self._glx+1.)*self._dj0zeros[ii+1] + self._j0zeros[ii] for ii in range(maxj0zeroIndx)]).flatten()
                weights= nu.array([self._glw*self._dj0zeros[ii+1] for ii in range(maxj0zeroIndx)]).flatten()
                evalInt= special.jn(0,ks*R[jj])*(self._alpha**2.+ks**2.)**-1.5*(self._beta*nu.exp(-ks*nu.fabs(z[jj]))-ks*nu.exp(-self._beta*nu.fabs(z[jj])))/(self._beta**2.-ks**2.)
                out[jj]= -2.*nu.pi*self._alpha*nu.sum(weights*evalInt)
            if floatIn: return out[0]
            else: return out
        #Old code, uses scipy's quadrature to do the relevant integrals, split into two
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
        maxiterFactorSmall= 2
        maxiterFactorLarge= 2
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
        if self._new:
            if isinstance(R,nu.ndarray):
                if not isinstance(z,nu.ndarray): z= nu.ones_like(R)*z
                out= nu.array([self._Rforce(rr,zz) for rr,zz in zip(R,z)])
                return out
            if R > 6.: return self._kp.Rforce(R,z)
            if R < 1.: R4max= 1.
            else: R4max= R
            kmax= self._kmaxFac*self._beta
            kmax= 2.*self._kmaxFac*self._beta
            maxj1zeroIndx= nu.argmin((self._j1zeros-kmax*R4max)**2.) #close enough
            ks= nu.array([0.5*(self._glx+1.)*self._dj1zeros[ii+1] + self._j1zeros[ii] for ii in range(maxj1zeroIndx)]).flatten()
            weights= nu.array([self._glw*self._dj1zeros[ii+1] for ii in range(maxj1zeroIndx)]).flatten()
            evalInt= ks*special.jn(1,ks*R)*(self._alpha**2.+ks**2.)**-1.5*(self._beta*nu.exp(-ks*nu.fabs(z))-ks*nu.exp(-self._beta*nu.fabs(z)))/(self._beta**2.-ks**2.)
            return -2.*nu.pi*self._alpha*nu.sum(weights*evalInt)
        #Old code, uses scipy's quadrature to do the relevant integrals, split into two
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
        maxiterFactorSmall= 4
        maxiterFactorLarge= 4
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
        if self._new:
            if isinstance(R,nu.ndarray):
                if not isinstance(z,nu.ndarray): z= nu.ones_like(R)*z
                out= nu.array([self._zforce(rr,zz) for rr,zz in zip(R,z)])
                return out
            if R > 6.: return self._kp.zforce(R,z)
            if R < 1.: R4max= 1.
            else: R4max= R
            kmax= self._kmaxFac*self._beta
            maxj0zeroIndx= nu.argmin((self._j0zeros-kmax*R4max)**2.) #close enough
            ks= nu.array([0.5*(self._glx+1.)*self._dj0zeros[ii+1] + self._j0zeros[ii] for ii in range(maxj0zeroIndx)]).flatten()
            weights= nu.array([self._glw*self._dj0zeros[ii+1] for ii in range(maxj0zeroIndx)]).flatten()
            evalInt= ks*special.jn(0,ks*R)*(self._alpha**2.+ks**2.)**-1.5*(nu.exp(-ks*nu.fabs(z))-nu.exp(-self._beta*nu.fabs(z)))/(self._beta**2.-ks**2.)
            if z > 0.:
                return -2.*nu.pi*self._alpha*self._beta*nu.sum(weights*evalInt)
            else:
                return 2.*nu.pi*self._alpha*self._beta*nu.sum(weights*evalInt)
        #Old code, uses scipy's quadrature to do the relevant integrals, split into two
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
        maxiterFactorSmall= 4
        maxiterFactorLarge= 4
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

    def _R2deriv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           R2deriv
        PURPOSE:
           evaluate R2 derivative
        INPUT:
           R - Cylindrical Galactocentric radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           -d K_R (R,z) d R
        HISTORY:
           2012-12-27 - Written - Bovy (IAS)
        """
        if self._new:
            if isinstance(R,nu.ndarray):
                if not isinstance(z,nu.ndarray): z= nu.ones_like(R)*z
                out= nu.array([self._R2deriv(rr,zz) for rr,zz in zip(R,z)])
                return out
            if R > 16.*self._hr or R > 6.: return self._kp.R2deriv(R,z)
            if R < 1.: R4max= 1.
            else: R4max= R
            kmax= 2.*self._kmaxFac*self._beta
            maxj0zeroIndx= nu.argmin((self._j0zeros-kmax*R4max)**2.) #close enough
            maxj2zeroIndx= nu.argmin((self._j2zeros-kmax*R4max)**2.) #close enough
            ks0= nu.array([0.5*(self._glx+1.)*self._dj0zeros[ii+1] + self._j0zeros[ii] for ii in range(maxj0zeroIndx)]).flatten()
            weights0= nu.array([self._glw*self._dj0zeros[ii+1] for ii in range(maxj0zeroIndx)]).flatten()
            ks2= nu.array([0.5*(self._glx+1.)*self._dj2zeros[ii+1] + self._j2zeros[ii] for ii in range(maxj2zeroIndx)]).flatten()
            weights2= nu.array([self._glw*self._dj2zeros[ii+1] for ii in range(maxj2zeroIndx)]).flatten()
            evalInt0= ks0**2.*special.jn(0,ks0*R)*(self._alpha**2.+ks0**2.)**-1.5*(self._beta*nu.exp(-ks0*nu.fabs(z))-ks0*nu.exp(-self._beta*nu.fabs(z)))/(self._beta**2.-ks0**2.)
            evalInt2= ks2**2.*special.jn(2,ks2*R)*(self._alpha**2.+ks2**2.)**-1.5*(self._beta*nu.exp(-ks2*nu.fabs(z))-ks2*nu.exp(-self._beta*nu.fabs(z)))/(self._beta**2.-ks2**2.)
            return nu.pi*self._alpha*(nu.sum(weights0*evalInt0)
                                      -nu.sum(weights2*evalInt2))
        #Old code, uses scipy's quadrature to do the relevant integrals, split into two
        notConvergedSmall= True
        notConvergedLarge= True
        smallkIntegral= integrate.quadrature(_doubleExponentialDiskPotentialR2derivIntegrandSmallk,
                                             0.,1./self._gamma,
                                             args=(self._alpha*R,
                                                   self._beta*nu.fabs(z),
                                                   self._gamma),tol=_TOL,
                                             maxiter= 2*self._maxiter,
                                                 vec_func=True)
        largekIntegral= integrate.quadrature(_doubleExponentialDiskPotentialR2derivIntegrandLargek,
                                             0.,self._gamma,
                                             args=(self._alpha*R,
                                                   self._beta*nu.fabs(z),
                                                   self._gamma),tol=_TOL,
                                             maxiter= 2*self._maxiter,
                                             vec_func=True)
        maxiterFactorSmall= 4
        maxiterFactorLarge= 4
        if nu.fabs(smallkIntegral[1]/(smallkIntegral[0]+largekIntegral[0])) <= self._tol:
            notConvergedSmall= False
        if nu.fabs(largekIntegral[1]/(largekIntegral[0]+smallkIntegral[0])) <= self._tol:
            notConvergedLarge= False
        while notConvergedSmall or notConvergedLarge:
            if notConvergedSmall:
                smallkIntegral= integrate.quadrature(_doubleExponentialDiskPotentialR2derivIntegrandSmallk,
                                                     0.,1./self._gamma,
                                                     args=(self._alpha*R,
                                                           self._beta*nu.fabs(z),
                                                           self._gamma),
                                                     tol=_TOL,
                                                     maxiter= maxiterFactorSmall*self._maxiter,
                                                     vec_func=True)
                if nu.fabs(smallkIntegral[1]/(smallkIntegral[0]+largekIntegral[0])) > self._tol:
                    maxiterFactorSmall*= 2
                else:
                    notConvergedSmall= False
            if notConvergedLarge:
                largekIntegral= integrate.quadrature(_doubleExponentialDiskPotentialR2derivIntegrandLargek,
                                                     0.,self._gamma,
                                                     args=(self._alpha*R,
                                                           self._beta*nu.fabs(z),
                                                           self._gamma),
                                                     tol=_TOL,
                                                     maxiter=maxiterFactorLarge*self._maxiter,
                                                     vec_func=True)
            if nu.fabs(largekIntegral[1]/(largekIntegral[0]+smallkIntegral[0])) > self._tol:
                maxiterFactorLarge*= 2
            else:
                notConvergedLarge= False
        return 4.*nu.pi*self._alpha/self._beta*(smallkIntegral[0]+largekIntegral[0])
    
    def _z2deriv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           z2deriv
        PURPOSE:
           evaluate z2 derivative
        INPUT:
           R - Cylindrical Galactocentric radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           -d K_Z (R,z) d Z
        HISTORY:
           2012-12-26 - Written - Bovy (IAS)
        """
        if self._new:
            if isinstance(R,nu.ndarray):
                if not isinstance(z,nu.ndarray): z= nu.ones_like(R)*z
                out= nu.array([self._z2deriv(rr,zz) for rr,zz in zip(R,z)])
                return out
            if R > 6.: return self._kp.z2deriv(R,z)
            if R < 1.: R4max= 1.
            else: R4max= R
            kmax= self._kmaxFac*self._beta
            maxj0zeroIndx= nu.argmin((self._j0zeros-kmax*R4max)**2.) #close enough
            ks= nu.array([0.5*(self._glx+1.)*self._dj0zeros[ii+1] + self._j0zeros[ii] for ii in range(maxj0zeroIndx)]).flatten()
            weights= nu.array([self._glw*self._dj0zeros[ii+1] for ii in range(maxj0zeroIndx)]).flatten()
            evalInt= ks*special.jn(0,ks*R)*(self._alpha**2.+ks**2.)**-1.5*(ks*nu.exp(-ks*nu.fabs(z))-self._beta*nu.exp(-self._beta*nu.fabs(z)))/(self._beta**2.-ks**2.)
            return -2.*nu.pi*self._alpha*self._beta*nu.sum(weights*evalInt)
        raise NotImplementedError("none 'new' z2deriv not implemented for DoubleExponentialDiskPotential")

    def _Rzderiv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           Rzderiv
        PURPOSE:
           evaluate the mixed R,z derivative
        INPUT:
           R - Cylindrical Galactocentric radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           d2phi/dR/dz
        HISTORY:
           2013-08-28 - Written - Bovy (IAS)
        """
        if self._new:
            if isinstance(R,nu.ndarray):
                if not isinstance(z,nu.ndarray): z= nu.ones_like(R)*z
                out= nu.array([self._Rzderiv(rr,zz) for rr,zz in zip(R,z)])
                return out
            if R > 6.: return self._kp.Rzderiv(R,z)
            if R < 1.: R4max= 1.
            else: R4max= R
            kmax= 2.*self._kmaxFac*self._beta
            maxj1zeroIndx= nu.argmin((self._j1zeros-kmax*R4max)**2.) #close enough
            ks= nu.array([0.5*(self._glx+1.)*self._dj1zeros[ii+1] + self._j1zeros[ii] for ii in range(maxj1zeroIndx)]).flatten()
            weights= nu.array([self._glw*self._dj1zeros[ii+1] for ii in range(maxj1zeroIndx)]).flatten()
            evalInt= ks**2.*special.jn(1,ks*R)*(self._alpha**2.+ks**2.)**-1.5*(nu.exp(-ks*nu.fabs(z))-nu.exp(-self._beta*nu.fabs(z)))/(self._beta**2.-ks**2.)
            if z >= 0.:
                return -2.*nu.pi*self._alpha*self._beta*nu.sum(weights*evalInt)
            else:
                return 2.*nu.pi*self._alpha*self._beta*nu.sum(weights*evalInt)
        raise NotImplementedError("none 'new' Rzderiv not implemented for DoubleExponentialDiskPotential")

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

def _doubleExponentialDiskPotentialR2derivIntegrandSmallk(k,R,z,gamma):
    """Internal function that gives the integrand for the double
    exponential disk radial force for k < 1/gamma"""
    gammak= gamma*k
    return k*k*0.5*(special.jn(0,k*R)-special.jn(2,k*R))\
        *(1.+k**2.)**-1.5*(nu.exp(-gammak*z)
                           -gammak*nu.exp(-z))/(1.-gammak**2.)

def _doubleExponentialDiskPotentialR2derivIntegrandLargek(k,R,z,gamma):
    """Internal function that gives the integrand for the double
    exponential disk radial force for k > 1/gamma"""
    return 1./k**2.*_doubleExponentialDiskPotentialR2derivIntegrandSmallk(1./k,R,z,gamma)


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
