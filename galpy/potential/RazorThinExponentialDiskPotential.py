###############################################################################
#   RazorThinExponentialDiskPotential.py: class that implements the razor thin
#                                         exponential disk potential
#
#                                      rho(R,z) = rho_0 e^-R/h_R delta(z)
###############################################################################
import numpy as nu
import warnings
from scipy import special, integrate
from galpy.util import galpyWarning
from .Potential import Potential, _APY_LOADED
if _APY_LOADED:
    from astropy import units
_TOL= 1.4899999999999999e-15
_MAXITER= 20
class RazorThinExponentialDiskPotential(Potential):
    """Class that implements the razor-thin exponential disk potential

    .. math::

        \\rho(R,z) = \\mathrm{amp}\\,\\exp\\left(-R/h_R\\right)\\,\\delta(z)

    """
    def __init__(self,amp=1.,hr=1./3.,
                 maxiter=_MAXITER,tol=0.001,normalize=False,
                 ro=None,vo=None,
                 new=True,glorder=100):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a razor-thin-exponential disk potential

        INPUT:

           amp - amplitude to be applied to the potential (default: 1); can be a Quantity with units of surface-mass or Gxsurface-mass

           hr - disk scale-length (can be Quantity)

           tol - relative accuracy of potential-evaluations

           maxiter - scipy.integrate keyword

           normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.

           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

        OUTPUT:

           RazorThinExponentialDiskPotential object

        HISTORY:

           2012-12-27 - Written - Bovy (IAS)

        """
        Potential.__init__(self,amp=amp,ro=ro,vo=vo,amp_units='surfacedensity')
        if _APY_LOADED and isinstance(hr,units.Quantity):
            hr= hr.to(units.kpc).value/self._ro
        self._new= new
        self._glorder= glorder
        self._hr= hr
        self._scale= self._hr
        self._alpha= 1./self._hr
        self._maxiter= maxiter
        self._tol= tol
        self._glx, self._glw= nu.polynomial.legendre.leggauss(self._glorder)
        if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)): #pragma: no cover
            self.normalize(normalize)
        #Load Kepler potential for large R
        #self._kp= KeplerPotential(normalize=4.*nu.pi/self._alpha**2./self._beta)

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
           2012-12-26 - Written - Bovy (IAS)
        """
        if self._new:
            #if R > 6.: return self._kp(R,z)
            if nu.fabs(z) < 10.**-6.:
                y= 0.5*self._alpha*R
                return -nu.pi*R*(special.i0(y)*special.k1(y)-special.i1(y)*special.k0(y))
            kalphamax= 10.
            ks= kalphamax*0.5*(self._glx+1.)
            weights= kalphamax*self._glw
            sqrtp= nu.sqrt(z**2.+(ks+R)**2.)
            sqrtm= nu.sqrt(z**2.+(ks-R)**2.)
            evalInt= nu.arcsin(2.*ks/(sqrtp+sqrtm))*ks*special.k0(self._alpha*ks)
            return -2.*self._alpha*nu.sum(weights*evalInt)
        raise NotImplementedError("Not new=True not implemented for RazorThinExponentialDiskPotential")

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
           2012-12-27 - Written - Bovy (IAS)
        """
        if self._new:
            #if R > 6.: return self._kp(R,z)
            if nu.fabs(z) < 10.**-6.:
                y= 0.5*self._alpha*R
                return -2.*nu.pi*y*(special.i0(y)*special.k0(y)-special.i1(y)*special.k1(y))
            kalphamax1= R
            ks1= kalphamax1*0.5*(self._glx+1.)
            weights1= kalphamax1*self._glw
            sqrtp= nu.sqrt(z**2.+(ks1+R)**2.)
            sqrtm= nu.sqrt(z**2.+(ks1-R)**2.)
            evalInt1= ks1**2.*special.k0(ks1*self._alpha)*((ks1+R)/sqrtp-(ks1-R)/sqrtm)/nu.sqrt(R**2.+z**2.-ks1**2.+sqrtp*sqrtm)/(sqrtp+sqrtm)
            if R < 10.:
                kalphamax2= 10.
                ks2= (kalphamax2-kalphamax1)*0.5*(self._glx+1.)+kalphamax1
                weights2= (kalphamax2-kalphamax1)*self._glw
                sqrtp= nu.sqrt(z**2.+(ks2+R)**2.)
                sqrtm= nu.sqrt(z**2.+(ks2-R)**2.)
                evalInt2= ks2**2.*special.k0(ks2*self._alpha)*((ks2+R)/sqrtp-(ks2-R)/sqrtm)/nu.sqrt(R**2.+z**2.-ks2**2.+sqrtp*sqrtm)/(sqrtp+sqrtm)
                return -2.*nu.sqrt(2.)*self._alpha*nu.sum(weights1*evalInt1
                                                          +weights2*evalInt2)
            else:
                return -2.*nu.sqrt(2.)*self._alpha*nu.sum(weights1*evalInt1)
        raise NotImplementedError("Not new=True not implemented for RazorThinExponentialDiskPotential")

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
           2012-12-27 - Written - Bovy (IAS)
        """
        if self._new:
            #if R > 6.: return self._kp(R,z)
            if nu.fabs(z) < 10.**-6.:
                return 0.
            kalphamax1= R
            ks1= kalphamax1*0.5*(self._glx+1.)
            weights1= kalphamax1*self._glw
            sqrtp= nu.sqrt(z**2.+(ks1+R)**2.)
            sqrtm= nu.sqrt(z**2.+(ks1-R)**2.)
            evalInt1= ks1**2.*special.k0(ks1*self._alpha)*(1./sqrtp+1./sqrtm)/nu.sqrt(R**2.+z**2.-ks1**2.+sqrtp*sqrtm)/(sqrtp+sqrtm)
            if R < 10.:
                kalphamax2= 10.
                ks2= (kalphamax2-kalphamax1)*0.5*(self._glx+1.)+kalphamax1
                weights2= (kalphamax2-kalphamax1)*self._glw
                sqrtp= nu.sqrt(z**2.+(ks2+R)**2.)
                sqrtm= nu.sqrt(z**2.+(ks2-R)**2.)
                evalInt2= ks2**2.*special.k0(ks2*self._alpha)*(1./sqrtp+1./sqrtm)/nu.sqrt(R**2.+z**2.-ks2**2.+sqrtp*sqrtm)/(sqrtp+sqrtm)
                return -z*2.*nu.sqrt(2.)*self._alpha*nu.sum(weights1*evalInt1
                                                            +weights2*evalInt2)
            else:
                return -z*2.*nu.sqrt(2.)*self._alpha*nu.sum(weights1*evalInt1)
        raise NotImplementedError("Not new=True not implemented for RazorThinExponentialDiskPotential")


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
            if nu.fabs(z) < 10.**-6.:
                y= 0.5*self._alpha*R
                return nu.pi*self._alpha*(special.i0(y)*special.k0(y)-special.i1(y)*special.k1(y)) \
                    +nu.pi/4.*self._alpha**2.*R*(special.i1(y)*(3.*special.k0(y)+special.kn(2,y))-special.k1(y)*(3.*special.i0(y)+special.iv(2,y)))
            raise AttributeError("'R2deriv' for RazorThinExponentialDisk not implemented for z =/= 0")

    def _z2deriv(self,R,z,phi=0.,t=0.): #pragma: no cover
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
           -d K_z (R,z) d z
        HISTORY:
           2012-12-27 - Written - Bovy (IAS)
        """
        return nu.infty

    def _surfdens(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _surfdens
        PURPOSE:
           evaluate the surface density
        INPUT:
           R - Cylindrical Galactocentric radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           Sigma (R,z)
        HISTORY:
           2018-08-19 - Written - Bovy (UofT)
        """
        return nu.exp(-self._alpha*R)
