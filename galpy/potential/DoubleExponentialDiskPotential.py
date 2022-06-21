###############################################################################
#   DoubleExponentialDiskPotential.py: class that implements the double
#                                      exponential disk potential
#
#                                      rho(R,z) = rho_0 e^-R/h_R e^-|z|/h_z
###############################################################################
import numpy
from scipy import special
from ..util import conversion
from .Potential import Potential, check_potential_inputs_not_arrays

def _de_psi(t):
    return t*numpy.tanh(numpy.pi/2.*numpy.sinh(t))
def _de_psiprime(t):
    return (numpy.sinh(numpy.pi*numpy.sinh(t))
            +numpy.pi*t*numpy.cosh(t))/(numpy.cosh(numpy.pi*numpy.sinh(t))+1)

class DoubleExponentialDiskPotential(Potential):
    """Class that implements the double exponential disk potential

    .. math::

        \\rho(R,z) = \\mathrm{amp}\\,\\exp\\left(-R/h_R-|z|/h_z\\right)

    """
    def __init__(self,amp=1.,hr=1./3.,hz=1./16.,normalize=False,
                 ro=None,vo=None,
                 de_h=1e-3,de_n=10000):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a double-exponential disk potential

        INPUT:

           amp - amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass density or Gxmass density

           hr - disk scale-length (can be Quantity)

           hz - scale-height (can be Quantity)

           normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.

           de_h= (1e-3) step used in numerical integration (use 1000 for a lower accuracy version that is typically still high accuracy enough, but faster)

           de_b= (10000) number of points used in numerical integration

           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

        OUTPUT:

           DoubleExponentialDiskPotential object

        HISTORY:

           2010-04-16 - Written - Bovy (NYU)

           2013-01-01 - Re-implemented using faster integration techniques - Bovy (IAS)

           2020-12-24 - Re-implemented again using more accurate integration techniques for Bessel integrals - Bovy (UofT)

        """
        Potential.__init__(self,amp=amp,ro=ro,vo=vo,amp_units='density')
        hr= conversion.parse_length(hr,ro=self._ro)
        hz= conversion.parse_length(hz,ro=self._ro)
        self.hasC= True
        self.hasC_dens= True
        self._hr= hr
        self._scale= self._hr
        self._hz= hz
        self._alpha= 1./self._hr
        self._beta= 1./self._hz
        self._zforceNotSetUp= True #We have not calculated a typical Kz yet
        # For double-exponential formula
        self._de_h= de_h
        self._de_n= de_n
        self._de_j0zeros= special.jn_zeros(0,self._de_n)/numpy.pi
        self._de_j1zeros= special.jn_zeros(1,self._de_n)/numpy.pi
        self._de_j0_xs= numpy.pi/self._de_h\
            *_de_psi(self._de_h*self._de_j0zeros)
        self._de_j0_weights= 2./(numpy.pi*self._de_j0zeros\
                               *special.j1(numpy.pi*self._de_j0zeros)**2.)\
                               *special.j0(self._de_j0_xs)\
                               *_de_psiprime(self._de_h*self._de_j0zeros)
        self._de_j1_xs= numpy.pi/self._de_h\
            *_de_psi(self._de_h*self._de_j1zeros)
        self._de_j1_weights= 2./(numpy.pi*self._de_j1zeros\
                               *special.jv(2,numpy.pi*self._de_j1zeros)**2.)\
                               *special.j1(self._de_j1_xs)\
                               *_de_psiprime(self._de_h*self._de_j1zeros)
        # Potential at zero in case we want that
        _gamma= self._beta/self._alpha
        _gamma2= _gamma**2.
        self._pot_zero= (2.*(_gamma-1.)*numpy.sqrt(1.+_gamma2)
                         +2.*numpy.arctanh(1./numpy.sqrt(1.+_gamma2))
                         -numpy.log(1.-_gamma/numpy.sqrt(1.+_gamma2))
                         +numpy.log(1.+_gamma/numpy.sqrt(1.+_gamma2)))\
                         /(2.*(1.+_gamma2)**1.5)
        self._pot_zero*= -4.*numpy.pi/self._alpha**2.
        # Normalize?
        if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)): #pragma: no cover
            self.normalize(normalize)

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
           2020-12-24 - New method using Ogata's Bessel integral formula - Bovy (UofT)
        """
        if isinstance(R,(float,int)):
            floatIn= True
            R= numpy.array([R])
            z= numpy.array([z])
        else:
            if isinstance(z,float):
                z= z*numpy.ones_like(R)
            floatIn= False
            outShape= R.shape # this code can't do arbitrary shapes
            R= R.flatten()
            z= z.flatten()
        fun= lambda x: (self._alpha**2.+(x/R[:,numpy.newaxis])**2.)**-1.5\
            *(self._beta*numpy.exp(-x/R[:,numpy.newaxis]*numpy.fabs(z[:,numpy.newaxis]))
              -x/R[:,numpy.newaxis]*numpy.exp(-self._beta*numpy.fabs(z[:,numpy.newaxis])))\
              /(self._beta**2.-(x/R[:,numpy.newaxis])**2.)
        out= -4.*numpy.pi*self._alpha/R*\
            numpy.nansum(fun(self._de_j0_xs)*self._de_j0_weights,
                         axis=1)
        out[(R == 0)*(z == 0)]= self._pot_zero
        if floatIn: return out[0]
        else: return numpy.reshape(out,outShape)
    
    @check_potential_inputs_not_arrays
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
           2012-12-26 - New method using Gaussian quadrature between zeros - Bovy (IAS)
           2020-12-24 - New method using Ogata's Bessel integral formula - Bovy (UofT)
        """
        fun=  lambda x: x*(self._alpha**2.+(x/R)**2.)**-1.5\
            *(self._beta*numpy.exp(-x/R*numpy.fabs(z))
              -x/R*numpy.exp(-self._beta*numpy.fabs(z)))\
              /(self._beta**2.-(x/R)**2.)
        return -4.*numpy.pi*self._alpha/R**2.\
            *numpy.nansum(fun(self._de_j1_xs)*self._de_j1_weights)
    
    @check_potential_inputs_not_arrays
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
           2012-12-26 - New method using Gaussian quadrature between zeros - Bovy (IAS)
           2020-12-24 - New method using Ogata's Bessel integral formula - Bovy (UofT)
        """
        fun= lambda x: (self._alpha**2.+(x/R)**2.)**-1.5*x/R\
            *(numpy.exp(-x/R*numpy.fabs(z))
              -numpy.exp(-self._beta*numpy.fabs(z)))\
              /(self._beta**2.-(x/R)**2.)
        out= -4.*numpy.pi*self._alpha*self._beta/R*\
            numpy.nansum(fun(self._de_j0_xs)*self._de_j0_weights)
        if z > 0.:
            return out
        else:
            return -out

    @check_potential_inputs_not_arrays
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
           2020-12-24 - New method using Ogata's Bessel integral formula - Bovy (UofT)
        """
        fun=  lambda x: x**2*(self._alpha**2.+(x/R)**2.)**-1.5\
            *(self._beta*numpy.exp(-x/R*numpy.fabs(z))
              -x/R*numpy.exp(-self._beta*numpy.fabs(z)))\
              /(self._beta**2.-(x/R)**2.)
        return 4.*numpy.pi*self._alpha/R**3.\
            *numpy.nansum(fun(self._de_j0_xs)*self._de_j0_weights
                          -fun(self._de_j1_xs)/self._de_j1_xs\
                          *self._de_j1_weights)
    
    @check_potential_inputs_not_arrays
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
           2020-12-24 - New method using Ogata's Bessel integral formula - Bovy (UofT)
        """
        fun= lambda x: (self._alpha**2.+(x/R)**2.)**-1.5*x/R\
            *(x/R*numpy.exp(-x/R*numpy.fabs(z))
              -self._beta*numpy.exp(-self._beta*numpy.fabs(z)))\
              /(self._beta**2.-(x/R)**2.)
        return -4.*numpy.pi*self._alpha*self._beta/R*\
            numpy.nansum(fun(self._de_j0_xs)*self._de_j0_weights)

    @check_potential_inputs_not_arrays
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
           2020-12-24 - New method using Ogata's Bessel integral formula - Bovy (UofT)
        """
        fun= lambda x: (self._alpha**2.+(x/R)**2.)**-1.5*(x/R)**2.\
            *(numpy.exp(-x/R*numpy.fabs(z))
              -numpy.exp(-self._beta*numpy.fabs(z)))\
              /(self._beta**2.-(x/R)**2.)
        out= -4.*numpy.pi*self._alpha*self._beta/R*\
            numpy.nansum(fun(self._de_j1_xs)*self._de_j1_weights)
        if z > 0.:
            return out
        else:
            return -out

    def _dens(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _dens
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
        return numpy.exp(-self._alpha*R-self._beta*numpy.fabs(z))

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
        return 2.*numpy.exp(-self._alpha*R)/self._beta\
            *(1.-numpy.exp(-self._beta*numpy.fabs(z)))
