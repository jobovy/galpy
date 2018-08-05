###############################################################################
#   MiyamotoNagaiPotential.py: class that implements the Miyamoto-Nagai 
#                              potential
#                                                           GM
#                              phi(R,z) = -  ---------------------------------
#                                             \sqrt(R^2+(a+\sqrt(z^2+b^2))^2)
###############################################################################
import numpy as nu
from .Potential import Potential, kms_to_kpcGyrDecorator, _APY_LOADED
if _APY_LOADED:
    from astropy import units
class MiyamotoNagaiPotential(Potential):
    """Class that implements the Miyamoto-Nagai potential

    .. math::

        \\Phi(R,z) = -\\frac{\\mathrm{amp}}{\\sqrt{R^2+(a+\\sqrt{z^2+b^2})^2}}

    with :math:`\\mathrm{amp} = GM` the total mass.
    """
    def __init__(self,amp=1.,a=1.,b=0.1,normalize=False,
                 ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a Miyamoto-Nagai potential

        INPUT:

           amp - amplitude to be applied to the potential, the total mass (default: 1); can be a Quantity with units of mass or Gxmass

           a - scale length (can be Quantity)

           b - scale height (can be Quantity)

           normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.

           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

        OUTPUT:

           (none)

        HISTORY:

           2010-07-09 - Started - Bovy (NYU)

        """
        Potential.__init__(self,amp=amp,ro=ro,vo=vo,amp_units='mass')
        if _APY_LOADED and isinstance(a,units.Quantity):
            a= a.to(units.kpc).value/self._ro
        if _APY_LOADED and isinstance(b,units.Quantity):
            b= b.to(units.kpc).value/self._ro
        self._a= a
        self._scale= self._a
        self._b= b
        self._b2= self._b**2.
        if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)):
            self.normalize(normalize)
        self.hasC= True
        self.hasC_dxdv= True
        self._nemo_accname= 'MiyamotoNagai'

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
        return -1./nu.sqrt(R**2.+(self._a+nu.sqrt(z**2.+self._b2))**2.)

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
        return -R/(R**2.+(self._a+nu.sqrt(z**2.+self._b2))**2.)**(3./2.)

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
        sqrtbz= nu.sqrt(self._b2+z**2.)
        asqrtbz= self._a+sqrtbz
        if isinstance(R,float) and sqrtbz == asqrtbz:
            return (-z/
                     (R**2.+(self._a+nu.sqrt(z**2.+self._b2))**2.)**(3./2.))
        else:
            return (-z*asqrtbz/sqrtbz/
                     (R**2.+(self._a+nu.sqrt(z**2.+self._b2))**2.)**(3./2.))

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
        sqrtbz= nu.sqrt(self._b2+z**2.)
        asqrtbz= self._a+sqrtbz
        if isinstance(R,float) and sqrtbz == asqrtbz:
            return 3./\
                (R**2.+sqrtbz**2.)**2.5/4./nu.pi*self._b2
        else:
            return (self._a*R**2.+(self._a+3.*sqrtbz)*asqrtbz**2.)/\
                (R**2.+asqrtbz**2.)**2.5/sqrtbz**3./4./nu.pi*self._b2

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
        return 1./(R**2.+(self._a+nu.sqrt(z**2.+self._b2))**2.)**1.5 \
            -3.*R**2./(R**2.+(self._a+nu.sqrt(z**2.+self._b2))**2.)**2.5

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
           t - time
        OUTPUT:
           the second vertical derivative
        HISTORY:
           2012-07-25 - Written - Bovy (IAS@MPIA)
        """
        sqrtbz= nu.sqrt(self._b2+z**2.)
        asqrtbz= self._a+sqrtbz
        if isinstance(R,float) and sqrtbz == asqrtbz:
            return (self._b2+R**2.-2.*z**2.)*(self._b2+R**2.+z**2.)**-2.5
        else:
            return ((self._a**3.*self._b2 + 
                     self._a**2.*(3.*self._b2 - 2.* z**2.)
                     *nu.sqrt(self._b2 + z**2.)
                     + (self._b2 + R**2. - 2.*z**2.)*(self._b2 + z**2.)**1.5
                     +self._a* (3.*self._b2**2. - 4.*z**4. + self._b2*(R**2. - z**2.)))/
                    ((self._b2 + z**2.)**1.5* (R**2. + asqrtbz**2.)**2.5))

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
        sqrtbz= nu.sqrt(self._b2+z**2.)
        asqrtbz= self._a+sqrtbz
        if isinstance(R,float) and sqrtbz == asqrtbz:
            return -(3.*R*z/(R**2.+asqrtbz**2.)**2.5)
        else:
            return -(3.*R*z*asqrtbz
                     /sqrtbz/(R**2.+asqrtbz**2.)**2.5)

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
        return "0,%s,%s,%s" % (ampl,self._a*ro,self._b*ro)
