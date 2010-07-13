###############################################################################
#   TwoPowerSphericalPotential.py: General class for potentials derived from 
#                                  densities with two power-laws
#
#                                                    amp
#                             rho(r)= ------------------------------------
#                                      (r/a)^\alpha (1+r/a)^(\beta-\alpha)
###############################################################################
import math as m
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
           normalize - if True, normalize such that vc(1.,0.)=1., or, if 
                       given as a number, such that the force is this fraction 
                       of the force necessary to make vc(1.,0.)=1.
        OUTPUT:
           (none)
        HISTORY:
           2010-07-09 - Started - Bovy (NYU)
        """
        if alpha == round(alpha) and beta == round(beta):
            integerSelf= TwoPowerIntegerSphericalPotential(amp=amp,a=a,
                                                           alpha=int(alpha),
                                                           beta=int(beta),
                                                           normalize=normalize)
            self.integerSelf= integerSelf
        else:
            Potential.__init__(self,amp=amp)
            self.integerSelf= None
            self.a= a
            self.alpha= alpha
            self.beta= beta
            if normalize:
                self.normalize(normalize)
        return None

    def _evaluate(self,R,z):
        """
        NAME:
           _evaluate
        PURPOSE:
           evaluate the potential at R,z
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
        OUTPUT:
           Phi(R,z)
        HISTORY:
           2010-07-09 - Started - Bovy (NYU)
        """
        if not self.integerSelf == None:
            return self.integerSelf._evaluate(R,z)
        else:
            r= m.sqrt(R**2.+z**2.)
            return integrate.quadrature(_potIntegrandTransform,
                                        0.,self.a/r,
                                        args=(self.alpha,self.beta))[0]

    def _Rforce(self,R,z):
        """
        NAME:
           _Rforce
        PURPOSE:
           evaluate the radial force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
        OUTPUT:
           the radial force
        HISTORY:
           2010-07-09 - Written - Bovy (NYU)
        """
        if not self.integerSelf == None:
            return self.integerSelf._Rforce(R,z)
        else:
            r= m.sqrt(R**2.+z**2.)
            return R/r**self.alpha*special.hyp2f1(3.-self.alpha,
                                                  self.beta-self.alpha,
                                                  4.-self.alpha,
                                                  -r/self.a)
    def _zforce(self,R,z):
        """
        NAME:
           _zforce
        PURPOSE:
           evaluate the vertical force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
        OUTPUT:
           the vertical force
        HISTORY:
           2010-07-09 - Written - Bovy (NYU)
        """
        if not self.integerSelf == None:
            return self.integerSelf._zforce(R,z)
        else:
            r= m.sqrt(R**2.+z**2.)
            return z/r**self.alpha*special.hyp2f1(3.-self.alpha,
                                                  self.beta-self.alpha,
                                                  4.-self.alpha,
                                                  -r/self.a)

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
            if normalize:
                self.normalize(normalize)
        return None

    def _evaluate(self,R,z):
        """
        NAME:
           _evaluate
        PURPOSE:
           evaluate the potential at R,z
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
        OUTPUT:
           Phi(R,z)
        HISTORY:
           2010-07-09 - Started - Bovy (NYU)
        """
        if not self.HernquistSelf == None:
            return self.HernquistSelf._evaluate(R,z)
        elif not self.JaffeSelf == None:
            return self.JaffeSelf._evaluate(R,z)
        elif not self.NFWSelf == None:
            return self.NFWSelf._evaluate(R,z)
        else:
            raise AttributeError

    def _Rforce(self,R,z):
        """
        NAME:
           _Rforce
        PURPOSE:
           evaluate the radial force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
        OUTPUT:
           the radial force
        HISTORY:
           2010-07-09 - Written - Bovy (NYU)
        """
        if not self.HernquistSelf == None:
            return self.HernquistSelf._Rforce(R,z)
        elif not self.JaffeSelf == None:
            return self.JaffeSelf._Rforce(R,z)
        elif not self.NFWSelf == None:
            return self.NFWSelf._Rforce(R,z)
        else:
            raise AttributeError

    def _zforce(self,R,z):
        """
        NAME:
           _zforce
        PURPOSE:
           evaluate the vertical force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
        OUTPUT:
           the vertical force
        HISTORY:
           2010-07-09 - Written - Bovy (NYU)
        """
        if not self.HernquistSelf == None:
            return self.HernquistSelf._zforce(R,z)
        elif not self.JaffeSelf == None:
            return self.JaffeSelf._zforce(R,z)
        elif not self.NFWSelf == None:
            return self.NFWSelf._zforce(R,z)
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
           normalize - if True, normalize such that vc(1.,0.)=1., or, if 
                       given as a number, such that the force is this fraction 
                       of the force necessary to make vc(1.,0.)=1.
        OUTPUT:
           (none)
        HISTORY:
           2010-07-09 - Written - Bovy (NYU)
        """
        Potential.__init__(self,amp=amp)
        self.a= a
        if normalize:
            self.normalize(normalize)
        return None

    def _evaluate(self,R,z):
        """
        NAME:
           _evaluate
        PURPOSE:
           evaluate the potential at R,z
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
        OUTPUT:
           Phi(R,z)
        HISTORY:
           2010-07-09 - Started - Bovy (NYU)
        """
        return -1./(1.+m.sqrt(R**2.+z**2.)/self.a)

    def _Rforce(self,R,z):
        """
        NAME:
           _Rforce
        PURPOSE:
           evaluate the radial force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
        OUTPUT:
           the radial force
        HISTORY:
           2010-07-09 - Written - Bovy (NYU)
        """
        sqrtRz= m.sqrt(R**2.+z**2.)
        return -R/self.a/sqrtRz/(1.+sqrtRz)**2.

    def _zforce(self,R,z):
        """
        NAME:
           _zforce
        PURPOSE:
           evaluate the vertical force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
        OUTPUT:
           the vertical force
        HISTORY:
           2010-07-09 - Written - Bovy (NYU)
        """
        sqrtRz= m.sqrt(R**2.+z**2.)
        return -z/self.a/sqrtRz/(1.+sqrtRz)**2.

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
           normalize - if True, normalize such that vc(1.,0.)=1., or, if 
                       given as a number, such that the force is this fraction 
                       of the force necessary to make vc(1.,0.)=1.
        OUTPUT:
           (none)
        HISTORY:
           2010-07-09 - Written - Bovy (NYU)
        """
        Potential.__init__(self,amp=amp)
        self.a= a
        if normalize:
            self.normalize(normalize)
        return None

    def _evaluate(self,R,z):
        """
        NAME:
           _evaluate
        PURPOSE:
           evaluate the potential at R,z
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
        OUTPUT:
           Phi(R,z)
        HISTORY:
           2010-07-09 - Started - Bovy (NYU)
        """
        return -m.log(1.+self.a/m.sqrt(R**2.+z**2.))

    def _Rforce(self,R,z):
        """
        NAME:
           _Rforce
        PURPOSE:
           evaluate the radial force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
        OUTPUT:
           the radial force
        HISTORY:
           2010-07-09 - Written - Bovy (NYU)
        """
        sqrtRz= m.sqrt(R**2.+z**2.)
        return -self.a*R/sqrtRz**3./(1.+self.a/sqrtRz)

    def _zforce(self,R,z):
        """
        NAME:
           _zforce
        PURPOSE:
           evaluate the vertical force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
        OUTPUT:
           the vertical force
        HISTORY:
           2010-07-09 - Written - Bovy (NYU)
        """
        sqrtRz= m.sqrt(R**2.+z**2.)
        return -self.a*z/sqrtRz**3./(1.+self.a/sqrtRz)

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
           normalize - if True, normalize such that vc(1.,0.)=1., or, if 
                       given as a number, such that the force is this fraction 
                       of the force necessary to make vc(1.,0.)=1.
        OUTPUT:
           (none)
        HISTORY:
           2010-07-09 - Written - Bovy (NYU)
        """
        Potential.__init__(self,amp=amp)
        self.a= a
        if normalize:
            self.normalize(normalize)
        return None

    def _evaluate(self,R,z):
        """
        NAME:
           _evaluate
        PURPOSE:
           evaluate the potential at R,z
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
        OUTPUT:
           Phi(R,z)
        BUGS:
           Might not be accurate enough in all circumstances
        HISTORY:
           2010-07-09 - Started - Bovy (NYU)
        """
        r= m.sqrt(R**2.+z**2.)
        return -m.log(1.+r/self.a)/r

    def _Rforce(self,R,z):
        """
        NAME:
           _Rforce
        PURPOSE:
           evaluate the radial force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
        OUTPUT:
           the radial force
        HISTORY:
           2010-07-09 - Written - Bovy (NYU)
        """
        Rz= R**2.+z**2.
        sqrtRz= m.sqrt(Rz)
        return self.a*R*(1./Rz/(self.a+sqrtRz)-m.log(1.+sqrtRz/self.a)/sqrtRz/Rz)

    def _zforce(self,R,z):
        """
        NAME:
           _zforce
        PURPOSE:
           evaluate the vertical force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
        OUTPUT:
           the vertical force
        HISTORY:
           2010-07-09 - Written - Bovy (NYU)
        """
        Rz= R**2.+z**2.
        sqrtRz= m.sqrt(Rz)
        return self.a*z*(1./Rz/(self.a+sqrtRz)-m.log(1.+sqrtRz/self.a)/sqrtRz/Rz)
