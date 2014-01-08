###############################################################################
#   TwoPowerSphericalPotential.py: General class for potentials derived from 
#                                  densities with two power-laws
#
#                                                    amp
#                             rho(r)= ------------------------------------
#                                      (r/a)^\alpha (1+r/a)^(\beta-\alpha)
###############################################################################
import math as m
import numpy
from scipy import special, integrate
from Potential import Potential
class TwoPowerSphericalPotential(Potential):
    """Class that implements spherical potentials that are derived from 
    two-power density models

                              A
    rho(r)= ------------------------------------
             (r/a)^\alpha (1+r/a)^(\beta-\alpha)
    """
    def __init__(self,amp=1.,a=1.,alpha=1.,beta=3.,normalize=False):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a two-power-density potential

        INPUT:

           amp - amplitude to be applied to the potential (default: 1)

           a - "scale" (in terms of Ro)

           alpha - inner power

           beta - outer power

           normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.

        OUTPUT:

           (none)

        HISTORY:

           2010-07-09 - Started - Bovy (NYU)

        """
        self.a= a
        self.alpha= alpha
        self.beta= beta
        if alpha == round(alpha) and beta == round(beta):
            integerSelf= TwoPowerIntegerSphericalPotential(amp=amp,a=a,
                                                           alpha=int(alpha),
                                                           beta=int(beta),
                                                           normalize=normalize)
            self.integerSelf= integerSelf
        else:
            Potential.__init__(self,amp=amp)
            self.integerSelf= None
            if normalize or \
                    (isinstance(normalize,(int,float)) \
                         and not isinstance(normalize,bool)):
                self.normalize(normalize)
        return None

    def _evaluate(self,R,z,phi=0.,t=0.,dR=0,dphi=0):
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
        if dR == 0 and dphi == 0:
            if not self.integerSelf == None:
                return self.integerSelf._evaluate(R,z,phi=phi,t=t)
            else:
                r= numpy.sqrt(R**2.+z**2.)
                return integrate.quadrature(_potIntegrandTransform,
                                            0.,self.a/r,
                                            args=(self.alpha,self.beta))[0]
        elif dR == 1 and dphi == 0:
            return -self._Rforce(R,z,phi=phi,t=t)
        elif dR == 0 and dphi == 1:
            return -self._phiforce(R,z,phi=phi,t=t)

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
        if not self.integerSelf == None:
            return self.integerSelf._Rforce(R,z,phi=phi,t=t)
        else:
            r= numpy.sqrt(R**2.+z**2.)
            return -R/r**self.alpha*special.hyp2f1(3.-self.alpha,
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
        if not self.integerSelf == None:
            return self.integerSelf._zforce(R,z,phi=phi,t=t)
        else:
            r= numpy.sqrt(R**2.+z**2.)
            return -z/r**self.alpha*special.hyp2f1(3.-self.alpha,
                                                   self.beta-self.alpha,
                                                   4.-self.alpha,
                                                   -r/self.a)

    def _dens(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _dens
        PURPOSE:
           evaluate the density force for this potential
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
        return (self.a/r)**self.alpha/(1.+r/self.a)**(self.beta-self.alpha)/4./m.pi/self.a**3.

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
        return self._R2deriv(z,R) #Spherical potential

def _potIntegrandTransform(t,alpha,beta):
    """Internal function that transforms the integrand such that the integral becomes finite-ranged"""
    return 1./t**2.*_potIntegrand(1./t,alpha,beta)

def _potIntegrand(t,alpha,beta):
    """Internal function that holds the straight integrand to get the potential"""
    return -t**(1.-alpha)*special.hyp2f1(3.-alpha,
                                          beta-alpha,
                                          4.-alpha,
                                          -t)


class TwoPowerIntegerSphericalPotential(TwoPowerSphericalPotential):
    """Class that implements the two-power-density spherical potentials in 
    the case of integer powers"""
    def __init__(self,amp=1.,a=1.,alpha=1,beta=3,normalize=False):
        """
        NAME:
           __init__
        PURPOSE:
           initialize a two-power-density potential for integer powers
        INPUT:
           amp - amplitude to be applied to the potential (default: 1)
           a - "scale" (in terms of Ro)
           alpha - inner power (default: NFW)
           beta - outer power (default: NFW)
           normalize - if True, normalize such that vc(1.,0.)=1., or, if 
                       given as a number, such that the force is this fraction 
                       of the force necessary to make vc(1.,0.)=1.
        OUTPUT:
           (none)
        HISTORY:
           2010-07-09 - Started - Bovy (NYU)
        """
        if alpha == 1 and beta == 4:
            HernquistSelf= HernquistPotential(amp=amp,a=a,normalize=normalize)
            self.HernquistSelf= HernquistSelf
            self.JaffeSelf= None
            self.NFWSelf= None
        elif alpha == 2 and beta == 4:
            JaffeSelf= JaffePotential(amp=amp,a=a,normalize=normalize)
            self.HernquistSelf= None
            self.JaffeSelf= JaffeSelf
            self.NFWSelf= None
        elif alpha == 1 and beta == 3:
            NFWSelf= NFWPotential(amp=amp,a=a,normalize=normalize)
            self.HernquistSelf= None
            self.JaffeSelf= None
            self.NFWSelf= NFWSelf
        else:
            Potential.__init__(self,amp=amp)
            self.HernquistSelf= None
            self.JaffeSelf= None
            self.NFWSelf= None
            if normalize or \
                    (isinstance(normalize,(int,float)) \
                         and not isinstance(normalize,bool)):
                self.normalize(normalize)
        return None

    def _evaluate(self,R,z,phi=0.,t=0.,dR=0,dphi=0):
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
        if dR == 0 and dphi == 0:
            if not self.HernquistSelf == None:
                return self.HernquistSelf._evaluate(R,z,phi=phi,t=t)
            elif not self.JaffeSelf == None:
                return self.JaffeSelf._evaluate(R,z,phi=phi,t=t)
            elif not self.NFWSelf == None:
                return self.NFWSelf._evaluate(R,z,phi=phi,t=t)
            else:
                raise AttributeError
        elif dR == 1 and dphi == 0:
            return -self._Rforce(R,z,phi=phi,t=t)
        elif dR == 0 and dphi == 1:
            return -self._phiforce(R,z,phi=phi,t=t)

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
        if not self.HernquistSelf == None:
            return self.HernquistSelf._Rforce(R,z,phi=phi,t=t)
        elif not self.JaffeSelf == None:
            return self.JaffeSelf._Rforce(R,z,phi=phi,t=t)
        elif not self.NFWSelf == None:
            return self.NFWSelf._Rforce(R,z,phi=phi,t=t)
        else:
            raise AttributeError

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
        if not self.HernquistSelf == None:
            return self.HernquistSelf._zforce(R,z,phi=phi,t=t)
        elif not self.JaffeSelf == None:
            return self.JaffeSelf._zforce(R,z,phi=phi,t=t)
        elif not self.NFWSelf == None:
            return self.NFWSelf._zforce(R,z,phi=phi,t=t)
        else:
            raise AttributeError

class HernquistPotential(TwoPowerIntegerSphericalPotential):
    """Class that implements the Hernquist potential"""
    def __init__(self,amp=1.,a=1.,normalize=False):
        """
        NAME:

           __init__

        PURPOSE:

           Initialize a Hernquist potential

        INPUT:

           amp - amplitude to be applied to the potential

           a - "scale" (in terms of Ro)

           normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.

        OUTPUT:

           (none)

        HISTORY:

           2010-07-09 - Written - Bovy (NYU)

        """
        Potential.__init__(self,amp=amp)
        self.a= a
        self.alpha= 1
        self.beta= 4
        if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)):
            self.normalize(normalize)
        self.hasC= True
        return None

    def _evaluate(self,R,z,phi=0.,t=0.,dR=0,dphi=0):
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
        if dR == 0 and dphi == 0:
            return -1./(1.+numpy.sqrt(R**2.+z**2.)/self.a)/2./self.a
        elif dR == 1 and dphi == 0:
            return -self._Rforce(R,z,phi=phi,t=t)
        elif dR == 0 and dphi == 1:
            return -self._phiforce(R,z,phi=phi,t=t)

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

class JaffePotential(TwoPowerIntegerSphericalPotential):
    """Class that implements the Jaffe potential"""
    def __init__(self,amp=1.,a=1.,normalize=False):
        """
        NAME:

           __init__

        PURPOSE:

           Initialize a Jaffe potential

        INPUT:

           amp - amplitude to be applied to the potential

           a - "scale" (in terms of Ro)

           normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.

        OUTPUT:

           (none)

        HISTORY:

           2010-07-09 - Written - Bovy (NYU)

        """
        Potential.__init__(self,amp=amp)
        self.a= a
        self.alpha= 2
        self.beta= 4
        if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)):
            self.normalize(normalize)
        self.hasC= True
        return None

    def _evaluate(self,R,z,phi=0.,t=0.,dR=0,dphi=0):
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
        if dR == 0 and dphi == 0:
            return -numpy.log(1.+self.a/numpy.sqrt(R**2.+z**2.))/self.a
        elif dR == 1 and dphi == 0:
            return -self._Rforce(R,z,phi=phi,t=t)
        elif dR == 0 and dphi == 1:
            return -self._phiforce(R,z,phi=phi,t=t)

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

class NFWPotential(TwoPowerIntegerSphericalPotential):
    """Class that implements the NFW potential"""
    def __init__(self,amp=1.,a=1.,normalize=False):
        """
        NAME:

           __init__

        PURPOSE:

           Initialize a NFW potential

        INPUT:

           amp - amplitude to be applied to the potential

           a - "scale" (in terms of Ro)

           normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.

        OUTPUT:

           (none)

        HISTORY:

           2010-07-09 - Written - Bovy (NYU)

        """
        Potential.__init__(self,amp=amp)
        self.a= a
        self.alpha= 1
        self.beta= 3
        if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)):
            self.normalize(normalize)
        self.hasC= True
        return None

    def _evaluate(self,R,z,phi=0.,t=0.,dR=0,dphi=0):
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
        if dR == 0 and dphi == 0:
            r= numpy.sqrt(R**2.+z**2.)
            return -numpy.log(1.+r/self.a)/r
        elif dR == 1 and dphi == 0:
            return -self._Rforce(R,z,phi=phi,t=t)
        elif dR == 0 and dphi == 1:
            return -self._phiforce(R,z,phi=phi,t=t)

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
