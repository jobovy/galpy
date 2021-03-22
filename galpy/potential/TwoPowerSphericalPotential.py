###############################################################################
#   TwoPowerSphericalPotential.py: General class for potentials derived from 
#                                  densities with two power-laws
#
#                                                    amp
#                             rho(r)= ------------------------------------
#                                      (r/a)^\alpha (1+r/a)^(\beta-\alpha)
###############################################################################
import numpy
from scipy import special, optimize
from ..util import conversion
from .Potential import Potential, kms_to_kpcGyrDecorator, _APY_LOADED
if _APY_LOADED:
    from astropy import units
class TwoPowerSphericalPotential(Potential):
    """Class that implements spherical potentials that are derived from 
    two-power density models

    .. math::

        \\rho(r) = \\frac{\\mathrm{amp}}{4\\,\\pi\\,a^3}\\,\\frac{1}{(r/a)^\\alpha\\,(1+r/a)^{\\beta-\\alpha}}
    """
    def __init__(self,amp=1.,a=5.,alpha=1.5,beta=3.5,normalize=False,
                 ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a two-power-density potential

        INPUT:

           amp - amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass or Gxmass

           a - scale radius (can be Quantity)

           alpha - inner power

           beta - outer power

           normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.

           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

        OUTPUT:

           (none)

        HISTORY:

           2010-07-09 - Started - Bovy (NYU)

        """
        # Instantiate
        Potential.__init__(self,amp=amp,ro=ro,vo=vo,amp_units='mass')
        # _specialSelf for special cases (Dehnen class, Dehnen core, Hernquist, Jaffe, NFW)
        self._specialSelf= None
        if ((self.__class__ == TwoPowerSphericalPotential) &
            (alpha == round(alpha)) & (beta == round(beta))):
            if int(alpha) == 0 and int(beta) == 4:
                self._specialSelf=\
                        DehnenCoreSphericalPotential(amp=1.,a=a,
                                                     normalize=False)
            elif int(alpha) == 1 and int(beta) == 4:
                self._specialSelf=\
                        HernquistPotential(amp=1.,a=a,normalize=False)
            elif int(alpha) == 2 and int(beta) == 4:
                self._specialSelf= JaffePotential(amp=1.,a=a,normalize=False)
            elif int(alpha) == 1 and int(beta) == 3:
                self._specialSelf= NFWPotential(amp=1.,a=a,normalize=False)
        # correcting quantities
        a= conversion.parse_length(a,ro=self._ro)
        # setting properties
        self.a= a
        self._scale= self.a
        self.alpha= alpha
        self.beta= beta
        if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)): #pragma: no cover
            self.normalize(normalize)
        return None

    def _evaluate(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _evaluate
        PURPOSE:
           evaluate the potential at R,z
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           Phi(R,z)
        HISTORY:
           2010-07-09 - Started - Bovy (NYU)
        """
        if self._specialSelf is not None:
            return self._specialSelf._evaluate(R,z,phi=phi,t=t)
        elif self.beta == 3.:
            r= numpy.sqrt(R**2.+z**2.)
            return (1./self.a)\
                *(r-self.a*(r/self.a)**(3.-self.alpha)/(3.-self.alpha)\
                      *special.hyp2f1(3.-self.alpha,
                                      2.-self.alpha,
                                      4.-self.alpha,
                                      -r/self.a))/(self.alpha-2.)/r
        else:
            r= numpy.sqrt(R**2.+z**2.)
            return special.gamma(self.beta-3.)\
                *((r/self.a)**(3.-self.beta)/special.gamma(self.beta-1.)\
                      *special.hyp2f1(self.beta-3.,
                                      self.beta-self.alpha,
                                      self.beta-1.,
                                      -self.a/r)
                  -special.gamma(3.-self.alpha)/special.gamma(self.beta-self.alpha))/r

    def _Rforce(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _Rforce
        PURPOSE:
           evaluate the radial force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the radial force
        HISTORY:
           2010-07-09 - Written - Bovy (NYU)
        """
        if self._specialSelf is not None:
            return self._specialSelf._Rforce(R,z,phi=phi,t=t)
        else:
            r= numpy.sqrt(R**2.+z**2.)
            return -R/r**self.alpha*self.a**(self.alpha-3.)/(3.-self.alpha)\
                *special.hyp2f1(3.-self.alpha,
                                self.beta-self.alpha,
                                4.-self.alpha,
                                -r/self.a)

    def _zforce(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _zforce
        PURPOSE:
           evaluate the vertical force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the vertical force
        HISTORY:
           2010-07-09 - Written - Bovy (NYU)
        """
        if self._specialSelf is not None:
            return self._specialSelf._zforce(R,z,phi=phi,t=t)
        else:
            r= numpy.sqrt(R**2.+z**2.)
            return -z/r**self.alpha*self.a**(self.alpha-3.)/(3.-self.alpha)\
                *special.hyp2f1(3.-self.alpha,
                                self.beta-self.alpha,
                                4.-self.alpha,
                                -r/self.a)

    def _dens(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _dens
        PURPOSE:
           evaluate the density for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the density
        HISTORY:
           2010-08-08 - Written - Bovy (NYU)
        """
        r= numpy.sqrt(R**2.+z**2.)
        return (self.a/r)**self.alpha/(1.+r/self.a)**(self.beta-self.alpha)/4./numpy.pi/self.a**3.

    def _ddensdr(self,r,t=0.):
        """
        NAME:
           _ddensdr
        PURPOSE:
s           evaluate the radial density derivative for this potential
        INPUT:
           r - spherical radius
           t= time
        OUTPUT:
           the density derivative
        HISTORY:
           2021-02-05 - Written - Bovy (UofT)
        """
        return -self._amp*(self.a/r)**(self.alpha-1.)\
            *(1.+r/self.a)**(self.alpha-self.beta-1.)\
            *(self.a*self.alpha+r*self.beta)/r**2/4./numpy.pi/self.a**3.

    def _d2densdr2(self,r,t=0.):
        """
        NAME:
           _d2densdr2
        PURPOSE:
           evaluate the second radial density derivative for this potential
        INPUT:
           r - spherical radius
           t= time
        OUTPUT:
           the 2nd density derivative
        HISTORY:
           2021-02-05 - Written - Bovy (UofT)
        """
        return self._amp*(self.a/r)**(self.alpha-2.)\
            *(1.+r/self.a)**(self.alpha-self.beta-2.)\
            *(self.alpha*(self.alpha+1.)*self.a**2+
              2.*self.alpha*self.a*(self.beta+1.)*r
              +self.beta*(self.beta+1.)*r**2)/r**4/4./numpy.pi/self.a**3.

    def _ddenstwobetadr(self,r,beta=0):
        """
        NAME:
           _ddenstwobetadr
        PURPOSE:
           evaluate the radial density derivative x r^(2beta) for this potential
        INPUT:
           r - spherical radius
           beta= (0)
        OUTPUT:
           d (rho x r^{2beta} ) / d r
        HISTORY:
           2021-02-14 - Written - Bovy (UofT)
        """
        return self._amp/4./numpy.pi/self.a**3.\
            *r**(2.*beta-2.)*(self.a/r)**(self.alpha-1.)\
            *(1.+r/self.a)**(self.alpha-self.beta-1.)\
            *(self.a*(2.*beta-self.alpha)+r*(2.*beta-self.beta))
    
    def _R2deriv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _R2deriv
        PURPOSE:
           evaluate the second cylindrically radial derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t- time
        OUTPUT:
           the second cylindrically radial derivative
        HISTORY:
           2020-11-23 - Written - Beane (CfA)
        """
        r = numpy.sqrt(R**2.+z**2.)
        A = self.a**(self.alpha-3.)/(3.-self.alpha)
        hyper = special.hyp2f1(3.-self.alpha,
                                self.beta-self.alpha,
                                4.-self.alpha,
                                -r/self.a)
        hyper_deriv = (3.-self.alpha) * (self.beta - self.alpha) / (4.-self.alpha) \
               * special.hyp2f1(4.-self.alpha,
                                1.+self.beta-self.alpha,
                                5.-self.alpha,
                                -r/self.a)
        
        term1 = A * r**(-self.alpha) * hyper
        term2 = -self.alpha * A * R**2. * r**(-self.alpha-2.) * hyper
        term3 = -A * R**2 * r**(-self.alpha-1.) / self.a * hyper_deriv
        return term1 + term2 + term3

    def _Rzderiv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _R2deriv
        PURPOSE:
           evaluate the mixed radial/vertical derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t- time
        OUTPUT:
           the mixed radial/vertical derivative
        HISTORY:
           2020-11-28 - Written - Beane (CfA)
        """
        r = numpy.sqrt(R**2.+z**2.)
        A = self.a**(self.alpha-3.)/(3.-self.alpha)
        hyper = special.hyp2f1(3.-self.alpha,
                                self.beta-self.alpha,
                                4.-self.alpha,
                                -r/self.a)
        hyper_deriv = (3.-self.alpha) * (self.beta - self.alpha) / (4.-self.alpha) \
               * special.hyp2f1(4.-self.alpha,
                                1.+self.beta-self.alpha,
                                5.-self.alpha,
                                -r/self.a)
        
        term1 = -self.alpha * A * R * r**(-self.alpha-2.) * z * hyper
        term2 = -A * R * r**(-self.alpha-1.) * z / self.a * hyper_deriv
        return term1 + term2

    def _z2deriv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _z2deriv
        PURPOSE:
           evaluate the second vertical derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t- time
        OUTPUT:
           the second vertical derivative
        HISTORY:
           2012-07-26 - Written - Bovy (IAS@MPIA)
        """
        return self._R2deriv(numpy.fabs(z),R) #Spherical potential

    def _mass(self,R,z=None,t=0.):
        """
        NAME:
           _mass
        PURPOSE:
           evaluate the mass within R for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           t - time
        OUTPUT:
           the mass enclosed
        HISTORY:
           2014-04-01 - Written - Erkal (IoA)
        """
        if z is not None: raise AttributeError # use general implementation
        return (R/self.a)**(3.-self.alpha)/(3.-self.alpha)\
            *special.hyp2f1(3.-self.alpha,-self.alpha+self.beta,
                            4.-self.alpha,-R/self.a)

class DehnenSphericalPotential(TwoPowerSphericalPotential):
    """Class that implements the Dehnen Spherical Potential from `Dehnen (1993) <https://ui.adsabs.harvard.edu/abs/1993MNRAS.265..250D>`_

    .. math::

          \\rho(r) = \\frac{\\mathrm{amp}(3-\\alpha)}{4\\,\\pi\\,a^3}\\,\\frac{1}{(r/a)^{\\alpha}\\,(1+r/a)^{4-\\alpha}}
    """

    def __init__(self,amp=1.,a=1.,alpha=1.5,normalize=False,ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a Dehnen Spherical Potential; note that the amplitude definitio used here does NOT match that of Dehnen (1993)

        INPUT:

           amp - amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass or Gxmass

           a - scale radius (can be Quantity)

           alpha - inner power, restricted to [0, 3)

           normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.

           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

        OUTPUT:

           (none)

        HISTORY:

           2019-10-07 - Started - Starkman (UofT)

        """
        if (alpha < 0.) or (alpha >= 3.):
            raise IOError('DehnenSphericalPotential requires 0 <= alpha < 3')
        # instantiate
        TwoPowerSphericalPotential.__init__(
            self,amp=amp,a=a,alpha=alpha,beta=4,
            normalize=normalize,ro=ro,vo=vo)
        # make special-self and protect subclasses
        self._specialSelf= None
        if ((self.__class__ == DehnenSphericalPotential) &
            (alpha == round(alpha))):
            if round(alpha) == 0:
                self._specialSelf=\
                        DehnenCoreSphericalPotential(amp=1.,a=a,
                                                     normalize=False)
            elif round(alpha) == 1:
                self._specialSelf=\
                        HernquistPotential(amp=1.,a=a,normalize=False)
            elif round(alpha) == 2:
                self._specialSelf= JaffePotential(amp=1.,a=a,normalize=False)
        # set properties
        self.hasC= True
        self.hasC_dxdv= True
        self.hasC_dens= True
        return None

    def _evaluate(self,R,z,phi=0.,t=0.):
      """
      NAME:
         _evaluate
      PURPOSE:
         evaluate the potential at R,z
      INPUT:
         R - Galactocentric cylindrical radius
         z - vertical height
         phi - azimuth
         t - time
      OUTPUT:
         Phi(R,z)
      HISTORY:
         2019-11-20 - Written - Starkman (UofT)
      """
      if self._specialSelf is not None:
          return self._specialSelf._evaluate(R,z,phi=phi,t=t)
      else:  # valid for alpha != 2, 3
        r= numpy.sqrt(R**2.+z**2.)
        return -(1.-1./(1.+self.a/r)**(2.-self.alpha))/\
                 (self.a * (2.-self.alpha) * (3.-self.alpha))

    def _Rforce(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _Rforce
        PURPOSE:
           evaluate the radial force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the radial force
        HISTORY:
           2019-11-20 - Written - Starkman (UofT)
        """
        if self._specialSelf is not None:
            return self._specialSelf._Rforce(R,z,phi=phi,t=t)
        else:
            r= numpy.sqrt(R**2.+z**2.)
            return -R/r**self.alpha*(self.a+r)**(self.alpha-3.)/(3.-self.alpha)

    def _R2deriv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _R2deriv
        PURPOSE:
           evaluate the second radial derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t- time
        OUTPUT:
           the second radial derivative
        HISTORY:
           2019-10-11 - Written - Starkman (UofT)
        """
        if self._specialSelf is not None:
            return self._specialSelf._R2deriv(R, z, phi=phi, t=t)
        a, alpha = self.a, self.alpha
        r = numpy.sqrt(R**2. + z**2.)
        # formula not valid for alpha=2,3, (integers?)
        return (numpy.power(r, -2.-alpha)*numpy.power(r+a, alpha-4.)*
                (-a*r**2. + (2.*R**2.-z**2.)*r + a*alpha*R**2.)/
                (alpha - 3.))

    def _zforce(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _zforce
        PURPOSE:
           evaluate the vertical force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the vertical force
        HISTORY:
           2019-11-21 - Written - Starkman (UofT)
        """
        if self._specialSelf is not None:
            return self._specialSelf._zforce(R,z,phi=phi,t=t)
        else:
            r= numpy.sqrt(R**2.+z**2.)
            return -z/r**self.alpha*(self.a+r)**(self.alpha-3.)/(3.-self.alpha)

    def _z2deriv(self,R,z,phi=0.,t=0.):
        r"""
        NAME:
           _z2deriv
        PURPOSE:
           evaluate the second vertical derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t- time
        OUTPUT:
           the second vertical derivative
        HISTORY:
           2019-10-20 - Written - Starkman (UofT)
        """
        return self._R2deriv(z,R,phi=phi,t=t)

    def _Rzderiv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _Rzderiv
        PURPOSE:
           evaluate the mixed R,z derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t- time
        OUTPUT:
           d2phi/dR/dz
        HISTORY:
           2019-10-11 - Written - Starkman (UofT)
        """
        if self._specialSelf is not None:
            return self._specialSelf._Rzderiv(R, z, phi=phi, t=t)
        a, alpha= self.a, self.alpha
        r= numpy.sqrt(R**2.+z**2.)
        return ((R*z*numpy.power(r,-2.-alpha)*numpy.power(a+r,alpha-4.)
                 *(3*r+a*alpha))/(alpha-3))

    def _dens(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _dens
        PURPOSE:
           evaluate the density for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the density
        HISTORY:
           2019-11-20 - Written - Starkman (UofT)
        """
        r= numpy.sqrt(R**2.+z**2.)
        return (self.a/r)**self.alpha/(1.+r/self.a)**(4.-self.alpha)/4./numpy.pi/self.a**3.

    def _mass(self,R,z=None,t=0.):
        """
        NAME:
           _mass
        PURPOSE:
           evaluate the mass within R for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           t - time
        OUTPUT:
           the mass enclosed
        HISTORY:
           2019-11-20 - Written - Starkman (UofT)
        """
        if z is not None: raise AttributeError # use general implementation
        return 1./(1.+self.a/R)**(3.-self.alpha)/(3.-self.alpha) # written so it works for r=numpy.inf 

class DehnenCoreSphericalPotential(DehnenSphericalPotential):
    """Class that implements the Dehnen Spherical Potential from `Dehnen (1993) <https://ui.adsabs.harvard.edu/abs/1993MNRAS.265..250D>`_ with alpha=0 (corresponding to an inner core)

    .. math::

          \\rho(r) = \\frac{\\mathrm{amp}}{12\\,\\pi\\,a^3}\\,\\frac{1}{(1+r/a)^{4}}
    """

    def __init__(self,amp=1.,a=1.,normalize=False,ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a cored Dehnen Spherical Potential; note that the amplitude definition used here does NOT match that of Dehnen (1993)

        INPUT:

           amp - amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass or Gxmass

           a - scale radius (can be Quantity)

           alpha - inner power, restricted to [0, 3)

           normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.

           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

        OUTPUT:

           (none)

        HISTORY:

           2019-10-07 - Started - Starkman (UofT)

        """
        DehnenSphericalPotential.__init__(
            self,amp=amp,a=a,alpha=0,
            normalize=normalize,ro=ro,vo=vo)
        # set properties explicitly
        self.hasC= True
        self.hasC_dxdv= True
        self.hasC_dens= True
        return None

    def _evaluate(self,R,z,phi=0.,t=0.):
      """
      NAME:
         _evaluate
      PURPOSE:
         evaluate the potential at R,z
      INPUT:
         R - Galactocentric cylindrical radius
         z - vertical height
         phi - azimuth
         t - time
      OUTPUT:
         Phi(R,z)
      HISTORY:
         2019-11-20 - Written - Starkman (UofT)
      """
      r= numpy.sqrt(R**2.+z**2.)
      return -(1.-1./(1.+self.a/r)**2.)/(6.*self.a)

    def _Rforce(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _Rforce
        PURPOSE:
           evaluate the radial force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the radial force
        HISTORY:
           2019-11-20 - Written - Starkman (UofT)
        """
        return -R/numpy.power(numpy.sqrt(R**2.+z**2.)+self.a,3.)/3.

    def _rforce_jax(self,r):
        """
        NAME:
           _rforce_jax
        PURPOSE:
           evaluate the spherical radial force for this potential using JAX
        INPUT:
           r - Galactocentric spherical radius
        OUTPUT:
           the radial force
        HISTORY:
           2021-02-25 - Written - Bovy (UofT)
        """
        # No need for actual JAX!
        return -self._amp*r/(r+self.a)**3./3.

    def _R2deriv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _R2deriv
        PURPOSE:
           evaluate the second radial derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t- time
        OUTPUT:
           the second radial derivative
        HISTORY:
           2019-10-11 - Written - Starkman (UofT)
        """
        r = numpy.sqrt(R**2.+z**2.)
        return -(((2.*R**2.-z**2.)-self.a*r)/(3.*r*numpy.power(r+self.a,4.)))

    def _zforce(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _zforce
        PURPOSE:
           evaluate the vertical force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the vertical force
        HISTORY:
           2019-11-21 - Written - Starkman (UofT)
        """
        r= numpy.sqrt(R**2.+z**2.)
        return -z/numpy.power(self.a+r,3.)/3.

    def _z2deriv(self,R,z,phi=0.,t=0.):
        r"""
        NAME:
           _z2deriv
        PURPOSE:
           evaluate the second vertical derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t- time
        OUTPUT:
           the second vertical derivative
        HISTORY:
           2019-10-20 - Written - Starkman (UofT)
        """
        return self._R2deriv(z,R,phi=phi,t=t)

    def _Rzderiv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _Rzderiv
        PURPOSE:
           evaluate the mixed R,z derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t- time
        OUTPUT:
           d2phi/dR/dz
        HISTORY:
           2019-10-11 - Written - Starkman (UofT)
        """
        a= self.a
        r= numpy.sqrt(R**2.+z**2.)
        return -(R * z/r/numpy.power(a+r,4.))

    def _dens(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _dens
        PURPOSE:
           evaluate the density for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the density
        HISTORY:
           2019-11-20 - Written - Starkman (UofT)
        """
        r= numpy.sqrt(R**2.+z**2.)
        return 1./(1.+r/self.a)**4./4./numpy.pi/self.a**3.

    def _mass(self,R,z=None,t=0.):
        """
        NAME:
           _mass
        PURPOSE:
           evaluate the mass within R for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           t - time
        OUTPUT:
           the mass enclosed
        HISTORY:
           2019-11-20 - Written - Starkman (UofT)
        """
        if z is not None: raise AttributeError # use general implementation
        return 1./(1.+self.a/R)**3./3. # written so it works for r=numpy.inf 

class HernquistPotential(DehnenSphericalPotential):
    """Class that implements the Hernquist potential

    .. math::

        \\rho(r) = \\frac{\\mathrm{amp}}{4\\,\\pi\\,a^3}\\,\\frac{1}{(r/a)\\,(1+r/a)^{3}}

    """
    def __init__(self,amp=1.,a=1.,normalize=False,
                 ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           Initialize a Hernquist potential

        INPUT:

           amp - amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass or Gxmass (note that amp is 2 x [total mass] for the chosen definition of the Hernquist potential)

           a - scale radius (can be Quantity)

           normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.

           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

        OUTPUT:

           (none)

        HISTORY:

           2010-07-09 - Written - Bovy (NYU)

        """
        DehnenSphericalPotential.__init__(
            self,amp=amp,a=a,alpha=1,
            normalize=normalize,ro=ro,vo=vo)
        self._nemo_accname= 'Dehnen'
        # set properties explicitly
        self.hasC= True
        self.hasC_dxdv= True
        self.hasC_dens= True
        return None

    def _evaluate(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _evaluate
        PURPOSE:
           evaluate the potential at R,z
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           Phi(R,z)
        HISTORY:
           2010-07-09 - Started - Bovy (NYU)
        """
        return -1./(1.+numpy.sqrt(R**2.+z**2.)/self.a)/2./self.a

    def _Rforce(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _Rforce
        PURPOSE:
           evaluate the radial force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t- time
        OUTPUT:
           the radial force
        HISTORY:
           2010-07-09 - Written - Bovy (NYU)
        """
        sqrtRz= numpy.sqrt(R**2.+z**2.)
        return -R/self.a/sqrtRz/(1.+sqrtRz/self.a)**2./2./self.a

    def _zforce(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _zforce
        PURPOSE:
           evaluate the vertical force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           t - time
        OUTPUT:
           the vertical force
        HISTORY:
           2010-07-09 - Written - Bovy (NYU)
        """
        sqrtRz= numpy.sqrt(R**2.+z**2.)
        return -z/self.a/sqrtRz/(1.+sqrtRz/self.a)**2./2./self.a

    def _rforce_jax(self,r):
        """
        NAME:
           _rforce_jax
        PURPOSE:
           evaluate the spherical radial force for this potential using JAX
        INPUT:
           r - Galactocentric spherical radius
        OUTPUT:
           the radial force
        HISTORY:
           2021-02-14 - Written - Bovy (UofT)
        """
        # No need for actual JAX!
        return -self._amp/2./(r+self.a)**2.

    def _R2deriv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _R2deriv
        PURPOSE:
           evaluate the second radial derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t- time
        OUTPUT:
           the second radial derivative
        HISTORY:
           2011-10-09 - Written - Bovy (IAS)
        """
        sqrtRz= numpy.sqrt(R**2.+z**2.)
        return (self.a*z**2.+(z**2.-2.*R**2.)*sqrtRz)/sqrtRz**3.\
            /(self.a+sqrtRz)**3./2.

    def _Rzderiv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _Rzderiv
        PURPOSE:
           evaluate the mixed R,z derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t- time
        OUTPUT:
           d2phi/dR/dz
        HISTORY:
           2013-08-28 - Written - Bovy (IAS)
        """
        sqrtRz= numpy.sqrt(R**2.+z**2.)
        return -R*z*(self.a+3.*sqrtRz)*(sqrtRz*(self.a+sqrtRz))**-3./2.

    def _surfdens(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _surfdens
        PURPOSE:
           evaluate the surface density for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the surface density
        HISTORY:
           2018-08-19 - Written - Bovy (UofT)
        """
        r= numpy.sqrt(R**2.+z**2.)
        Rma= numpy.sqrt(R**2.-self.a**2.+0j)
        if Rma == 0.:
            return (-12.*self.a**3-5.*self.a*z**2
                      +numpy.sqrt(1.+z**2/self.a**2)\
                         *(12.*self.a**3-self.a*z**2+2/self.a*z**4))\
                          /30./numpy.pi*z**-5.
        else:
            return self.a*((2.*self.a**2.+R**2.)*Rma**-5\
                               *(numpy.arctan(z/Rma)-numpy.arctan(self.a*z/r/Rma))
                           +z*(5.*self.a**3.*r-4.*self.a**4
                               +self.a**2*(2.*r**2.+R**2)
                               -self.a*r*(5.*R**2.+3.*z**2.)+R**2.*r**2.)
                           /(self.a**2.-R**2.)**2.
                           /(r**2-self.a**2.)**2.).real/4./numpy.pi

    def _mass(self,R,z=None,t=0.):
        """
        NAME:
           _mass
        PURPOSE:
           calculate the mass out to a given radius
        INPUT:
           R - radius at which to return the enclosed mass
           z - (don't specify this) vertical height
        OUTPUT:
           mass in natural units
        HISTORY:
           2014-01-29 - Written - Bovy (IAS)
        """
        if z is not None: raise AttributeError # use general implementation
        return 1./(1.+self.a/R)**2./2. # written so it works for r=numpy.inf

    @kms_to_kpcGyrDecorator
    def _nemo_accpars(self,vo,ro):
        """
        NAME:

           _nemo_accpars

        PURPOSE:

           return the accpars potential parameters for use of this potential with NEMO

        INPUT:

           vo - velocity unit in km/s

           ro - length unit in kpc

        OUTPUT:

           accpars string

        HISTORY:

           2018-09-14 - Written - Bovy (UofT)

        """
        GM= self._amp*vo**2.*ro/2.
        return "0,1,%s,%s,0" % (GM,self.a*ro)

class JaffePotential(DehnenSphericalPotential):
    """Class that implements the Jaffe potential

    .. math::

        \\rho(r) = \\frac{\\mathrm{amp}}{4\\,\\pi\\,a^3}\\,\\frac{1}{(r/a)^2\\,(1+r/a)^{2}}

    """
    def __init__(self,amp=1.,a=1.,normalize=False,
                 ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           Initialize a Jaffe potential

        INPUT:

           amp - amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass or Gxmass

           a - scale radius (can be Quantity)

           normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.

           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

        OUTPUT:

           (none)

        HISTORY:

           2010-07-09 - Written - Bovy (NYU)

        """
        Potential.__init__(self,amp=amp,ro=ro,vo=vo,amp_units='mass')
        a= conversion.parse_length(a,ro=self._ro)
        self.a= a
        self._scale= self.a
        self.alpha= 2
        self.beta= 4
        if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)): #pragma: no cover
            self.normalize(normalize)
        self.hasC= True
        self.hasC_dxdv= True
        self.hasC_dens= True
        return None

    def _evaluate(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _evaluate
        PURPOSE:
           evaluate the potential at R,z
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           Phi(R,z)
        HISTORY:
           2010-07-09 - Started - Bovy (NYU)
        """
        return -numpy.log(1.+self.a/numpy.sqrt(R**2.+z**2.))/self.a

    def _Rforce(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _Rforce
        PURPOSE:
           evaluate the radial force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the radial force
        HISTORY:
           2010-07-09 - Written - Bovy (NYU)
        """
        sqrtRz= numpy.sqrt(R**2.+z**2.)
        return -R/sqrtRz**3./(1.+self.a/sqrtRz)

    def _zforce(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _zforce
        PURPOSE:
           evaluate the vertical force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the vertical force
        HISTORY:
           2010-07-09 - Written - Bovy (NYU)
        """
        sqrtRz= numpy.sqrt(R**2.+z**2.)
        return -z/sqrtRz**3./(1.+self.a/sqrtRz)

    def _R2deriv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _R2deriv
        PURPOSE:
           evaluate the second radial derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the second radial derivative
        HISTORY:
           2011-10-09 - Written - Bovy (IAS)
        """
        sqrtRz= numpy.sqrt(R**2.+z**2.)
        return (self.a*(z**2.-R**2.)+(z**2.-2.*R**2.)*sqrtRz)\
            /sqrtRz**4./(self.a+sqrtRz)**2.

    def _Rzderiv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _Rzderiv
        PURPOSE:
           evaluate the mixed R,z derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           d2phi/dR/dz
        HISTORY:
           2013-08-28 - Written - Bovy (IAS)
        """
        sqrtRz= numpy.sqrt(R**2.+z**2.)
        return -R*z*(2.*self.a+3.*sqrtRz)*sqrtRz**-4.\
            *(self.a+sqrtRz)**-2.

    def _surfdens(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _surfdens
        PURPOSE:
           evaluate the surface density for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the surface density
        HISTORY:
           2018-08-19 - Written - Bovy (UofT)
        """
        r= numpy.sqrt(R**2.+z**2.)
        Rma= numpy.sqrt(R**2.-self.a**2.+0j)
        if Rma == 0.:
            return (3.*z**2.-2.*self.a**2.
                    +2.*numpy.sqrt(1.+(z/self.a)**2.)\
                        *(self.a**2.-2.*z**2.)
                    +3.*z**3./self.a*numpy.arctan(z/self.a))\
                    /self.a/z**3./6./numpy.pi
        else:
            return ((2.*self.a**2.-R**2.)*Rma**-3\
                        *(numpy.arctan(z/Rma)-numpy.arctan(self.a*z/r/Rma))
                    +numpy.arctan(z/R)/R
                    -self.a*z/(R**2-self.a**2)/(r+self.a)).real\
                    /self.a/2./numpy.pi

    def _mass(self,R,z=None,t=0.):
        """
        NAME:
           _mass
        PURPOSE:
           calculate the mass out to a given radius
        INPUT:
           R - radius at which to return the enclosed mass
           z - (don't specify this) vertical height
        OUTPUT:
           mass in natural units
        HISTORY:
           2014-01-29 - Written - Bovy (IAS)
        """
        if z is not None: raise AttributeError # use general implementation
        return 1./(1.+self.a/R) # written so it works for r=numpy.inf 

class NFWPotential(TwoPowerSphericalPotential):
    """Class that implements the NFW potential

    .. math::

        \\rho(r) = \\frac{\\mathrm{amp}}{4\\,\\pi\\,a^3}\\,\\frac{1}{(r/a)\\,(1+r/a)^{2}}

    """
    def __init__(self,amp=1.,a=1.,normalize=False,
                 rmax=None,vmax=None,
                 conc=None,mvir=None,
                 vo=None,ro=None,
                 H=70.,Om=0.3,overdens=200.,wrtcrit=False):
        """
        NAME:

           __init__

        PURPOSE:

           Initialize a NFW potential

        INPUT:

           amp - amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass or Gxmass

           a - scale radius (can be Quantity)

           normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.


           Alternatively, NFW potentials can be initialized in the following two manners:

           a)

              rmax= radius where the rotation curve peaks (can be a Quantity, otherwise assumed to be in internal units)

              vmax= maximum circular velocity (can be a Quantity, otherwise assumed to be in internal units)

           b)

              conc= concentration

              mvir= virial mass in 10^12 Msolar

           in which case you also need to supply the following keywords
           
              H= (default: 70) Hubble constant in km/s/Mpc
           
              Om= (default: 0.3) Omega matter
       
              overdens= (200) overdensity which defines the virial radius

              wrtcrit= (False) if True, the overdensity is wrt the critical density rather than the mean matter density
           
           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

        OUTPUT:

           (none)

        HISTORY:

           2010-07-09 - Written - Bovy (NYU)

           2014-04-03 - Initialization w/ concentration and mass - Bovy (IAS)

           2020-04-29 - Initialization w/ rmax and vmax - Bovy (UofT)
           
        """
        Potential.__init__(self,amp=amp,ro=ro,vo=vo,amp_units='mass')
        a= conversion.parse_length(a,ro=self._ro)
        if conc is None and rmax is None:
            self.a= a
            self.alpha= 1
            self.beta= 3
            if normalize or \
                    (isinstance(normalize,(int,float)) \
                         and not isinstance(normalize,bool)):
                self.normalize(normalize)
        elif not rmax is None:
            if _APY_LOADED and isinstance(rmax,units.Quantity):
                rmax= conversion.parse_length(rmax,ro=self._ro)
                self._roSet= True
            if _APY_LOADED and isinstance(vmax,units.Quantity):
                vmax= conversion.parse_velocity(vmax,vo=self._vo)
                self._voSet= True
            self.a= rmax/2.1625815870646098349
            self._amp= vmax**2.*self.a/0.21621659550187311005
        else:
            if wrtcrit:
                od= overdens/conversion.dens_in_criticaldens(self._vo,
                                                                  self._ro,H=H)
            else:
                od= overdens/conversion.dens_in_meanmatterdens(self._vo,
                                                                    self._ro,
                                                                    H=H,Om=Om)
            mvirNatural= mvir*100./conversion.mass_in_1010msol(self._vo,
                                                                    self._ro)
            rvir= (3.*mvirNatural/od/4./numpy.pi)**(1./3.)
            self.a= rvir/conc
            self._amp= mvirNatural/(numpy.log(1.+conc)-conc/(1.+conc))
        self._scale= self.a
        self.hasC= True
        self.hasC_dxdv= True
        self.hasC_dens= True
        self._nemo_accname= 'NFW'
        return None

    def _evaluate(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _evaluate
        PURPOSE:
           evaluate the potential at R,z
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           Phi(R,z)
        HISTORY:
           2010-07-09 - Started - Bovy (NYU)
        """
        r= numpy.sqrt(R**2.+z**2.)
        if isinstance(r,(float,int)) and r == 0:
            return -1./self.a
        elif isinstance(r,(float,int)):
            return -special.xlogy(1./r,1.+r/self.a) # stable as r -> infty
        else:
            out= -special.xlogy(1./r,1.+r/self.a) # stable as r -> infty
            out[r == 0]= -1./self.a
            return out

    def _Rforce(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _Rforce
        PURPOSE:
           evaluate the radial force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the radial force
        HISTORY:
           2010-07-09 - Written - Bovy (NYU)
        """
        Rz= R**2.+z**2.
        sqrtRz= numpy.sqrt(Rz)
        return R*(1./Rz/(self.a+sqrtRz)-numpy.log(1.+sqrtRz/self.a)/sqrtRz/Rz)

    def _zforce(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _zforce
        PURPOSE:
           evaluate the vertical force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the vertical force
        HISTORY:
           2010-07-09 - Written - Bovy (NYU)
        """
        Rz= R**2.+z**2.
        sqrtRz= numpy.sqrt(Rz)
        return z*(1./Rz/(self.a+sqrtRz)-numpy.log(1.+sqrtRz/self.a)/sqrtRz/Rz)

    def _rforce_jax(self,r):
        """
        NAME:
           _rforce_jax
        PURPOSE:
           evaluate the spherical radial force for this potential using JAX
        INPUT:
           r - Galactocentric spherical radius
        OUTPUT:
           the radial force
        HISTORY:
           2021-02-14 - Written - Bovy (UofT)
        """
        try:
            import jax.numpy as jnp
        except ImportError: # pragma: no cover
            raise ImportError("Making use of _rforce_jax function requires the google/jax library")
        return self._amp*(1./r/(self.a+r)-jnp.log(1.+r/self.a)/r**2.)

    def _R2deriv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _R2deriv
        PURPOSE:
           evaluate the second radial derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the second radial derivative
        HISTORY:
           2011-10-09 - Written - Bovy (IAS)
        """
        Rz= R**2.+z**2.
        sqrtRz= numpy.sqrt(Rz)
        return (3.*R**4.+2.*R**2.*(z**2.+self.a*sqrtRz)\
                    -z**2.*(z**2.+self.a*sqrtRz)\
                    -(2.*R**2.-z**2.)*(self.a**2.+R**2.+z**2.+2.*self.a*sqrtRz)\
                    *numpy.log(1.+sqrtRz/self.a))\
                    /Rz**2.5/(self.a+sqrtRz)**2.

    def _Rzderiv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _Rzderiv
        PURPOSE:
           evaluate the mixed R,z derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           d2phi/dR/dz
        HISTORY:
           2013-08-28 - Written - Bovy (IAS)
        """
        Rz= R**2.+z**2.
        sqrtRz= numpy.sqrt(Rz)
        return -R*z*(-4.*Rz-3.*self.a*sqrtRz+3.*(self.a**2.+Rz+2.*self.a*sqrtRz)*numpy.log(1.+sqrtRz/self.a))*Rz**-2.5*(self.a+sqrtRz)**-2.

    def _surfdens(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _surfdens
        PURPOSE:
           evaluate the surface density for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the surface density
        HISTORY:
           2018-08-19 - Written - Bovy (UofT)
        """
        r= numpy.sqrt(R**2.+z**2.)
        Rma= numpy.sqrt(R**2.-self.a**2.+0j)
        if Rma == 0.:
            za2= (z/self.a)**2
            return self.a*(2.+numpy.sqrt(za2+1.)*(za2-2.))/6./numpy.pi/z**3
        else:
            return (self.a*Rma**-3\
                        *(numpy.arctan(self.a*z/r/Rma)-numpy.arctan(z/Rma))
                    +z/(r+self.a)/(R**2.-self.a**2.)).real/2./numpy.pi

    def _mass(self,R,z=None,t=0.):
        """
        NAME:
           _mass
        PURPOSE:
           calculate the mass out to a given radius
        INPUT:
           R - radius at which to return the enclosed mass
           z - (don't specify this) vertical height
        OUTPUT:
           mass in natural units
        HISTORY:
           2014-01-29 - Written - Bovy (IAS)
        """
        if z is not None: raise AttributeError # use general implementation
        return numpy.log(1+R/self.a)-R/self.a/(1.+R/self.a)

    @conversion.physical_conversion('position',pop=False)
    def rvir(self,H=70.,Om=0.3,t=0.,overdens=200.,wrtcrit=False,ro=None,vo=None,
             use_physical=False): # use_physical necessary bc of pop=False, does nothing inside
        """
        NAME:

           rvir

        PURPOSE:

           calculate the virial radius for this density distribution

        INPUT:

           H= (default: 70) Hubble constant in km/s/Mpc
           
           Om= (default: 0.3) Omega matter
       
           overdens= (200) overdensity which defines the virial radius

           wrtcrit= (False) if True, the overdensity is wrt the critical density rather than the mean matter density

           ro= distance scale in kpc or as Quantity (default: object-wide, which if not set is 8 kpc))

           vo= velocity scale in km/s or as Quantity (default: object-wide, which if not set is 220 km/s))

        OUTPUT:
        
           virial radius
        
        HISTORY:

           2014-01-29 - Written - Bovy (IAS)

        """
        if ro is None: ro= self._ro
        if vo is None: vo= self._vo
        if wrtcrit:
            od= overdens/conversion.dens_in_criticaldens(vo,ro,H=H)
        else:
            od= overdens/conversion.dens_in_meanmatterdens(vo,ro,
                                                                H=H,Om=Om)
        dc= 12.*self.dens(self.a,0.,t=t,use_physical=False)/od
        x= optimize.brentq(lambda y: (numpy.log(1.+y)-y/(1.+y))/y**3.-1./dc,
                           0.01,100.)
        return x*self.a

    @conversion.physical_conversion('position',pop=True)
    def rmax(self):
        """
        NAME:

           rmax

        PURPOSE:

           calculate the radius at which the rotation curve peaks

        INPUT:

           (none)

        OUTPUT:
        
           Radius at which the rotation curve peaks
        
        HISTORY:

           2020-02-05 - Written - Bovy (UofT)

        """
        # Magical number, solve(derivative (ln(1+x)-x/(1+x))/x wrt x=0,x)
        return 2.1625815870646098349*self.a

    @conversion.physical_conversion('velocity',pop=True)
    def vmax(self):
        """
        NAME:

           vmax

        PURPOSE:

           calculate the maximum rotation curve velocity

        INPUT:

           (none)

        OUTPUT:
        
           Peak velocity in the rotation curve
        
        HISTORY:

           2020-02-05 - Written - Bovy (UofT)

        """
        # 0.21621659550187311005 = (numpy.log(1.+rmax)-rmax/(1.+rmax))/rmax 
        return numpy.sqrt(0.21621659550187311005*self._amp/self.a)

    @kms_to_kpcGyrDecorator
    def _nemo_accpars(self,vo,ro):
        """
        NAME:

           _nemo_accpars

        PURPOSE:

           return the accpars potential parameters for use of this potential with NEMO

        INPUT:

           vo - velocity unit in km/s

           ro - length unit in kpc

        OUTPUT:

           accpars string

        HISTORY:

           2014-12-18 - Written - Bovy (IAS)

        """
        ampl= self._amp*vo**2.*ro
        vmax= numpy.sqrt(ampl/self.a/ro*0.2162165954) #Take that factor directly from gyrfalcon
        return "0,%s,%s" % (self.a*ro,vmax)
