###############################################################################
#   LogarithmicHaloPotential.py: class that implements the logarithmic 
#                            potential Phi(r) = vc**2 ln(r)
###############################################################################
import warnings
import numpy as nu
from .Potential import Potential, kms_to_kpcGyrDecorator, _APY_LOADED
if _APY_LOADED:
    from astropy import units
from galpy.util import galpyWarning
_CORE=10**-8
class LogarithmicHaloPotential(Potential):
    """Class that implements the logarithmic potential

    .. math::

        \\Phi(R,z) = \\frac{\\mathrm{amp}}{2}\\,\\ln\\left[R^2+\\left(\\frac{z}{q}\\right)^2+\\mathrm{core}^2\\right]

    Alternatively, the potential can be made triaxial by adding a parameter :math:`b`

    .. math::

        \\Phi(x,y,z) = \\frac{\\mathrm{amp}}{2}\\,\\ln\\left[x^2+\\left(\\frac{y}{b}\\right)^2+\\left(\\frac{z}{q}\\right)^2+\\mathrm{core}^2\\right]

    With these definitions, :math:`\\sqrt{\mathrm{amp}}` is the circular velocity at :math:`r \gg \mathrm{core}` at :math:`(y,z) = (0,0)`.

    """
    def __init__(self,amp=1.,core=_CORE,q=1.,b=None,normalize=False,
                 ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a logarithmic potential

        INPUT:

           amp - amplitude to be applied to the potential (default: 1); can be a Quantity with units of velocity-squared

           core - core radius at which the logarithm is cut (can be Quantity)

           q - potential flattening (z/q)**2.
           
           b= (None) if set, shape parameter in y-direction (y --> y/b; see definition)

           normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.

           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

        OUTPUT:

           (none)

        HISTORY:

           2010-04-02 - Started - Bovy (NYU)

        """
        Potential.__init__(self,amp=amp,ro=ro,vo=vo,amp_units='velocity2')
        if _APY_LOADED and isinstance(core,units.Quantity):
            core= core.to(units.kpc).value/self._ro
        self.hasC= True
        self.hasC_dxdv= True
        self._core2= core**2.
        self._q= q
        self._b= b
        if not self._b is None:
            self.isNonAxi= True
            self._1m1overb2= 1.-1./self._b**2.
        if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)): #pragma: no cover 
            self.normalize(normalize)
        self._nemo_accname= 'LogPot'
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
           2010-04-02 - Started - Bovy (NYU)
           2010-04-30 - Adapted for R,z - Bovy (NYU)
        """
        if self.isNonAxi:
            return 1./2.*nu.log(R**2.*(1.-self._1m1overb2*nu.sin(phi)**2.)
                                +(z/self._q)**2.+self._core2)
        else:
            return 1./2.*nu.log(R**2.+(z/self._q)**2.+self._core2)

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
        """
        if self.isNonAxi:
            Rt2= R**2.*(1.-self._1m1overb2*nu.sin(phi)**2.)
            return -Rt2/R/(Rt2+(z/self._q)**2.+self._core2)
        else:
            return -R/(R**2.+(z/self._q)**2.+self._core2)

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
        """
        if self.isNonAxi:
            Rt2= R**2.*(1.-self._1m1overb2*nu.sin(phi)**2.)
            return -z/self._q**2./(Rt2+(z/self._q)**2.+self._core2)
        else:
            return -z/self._q**2./(R**2.+(z/self._q)**2.+self._core2)

    def _phiforce(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _phiforce
        PURPOSE:
           evaluate the azimuthal force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the azimuthal force
        HISTORY:
        """
        if self.isNonAxi:
            Rt2= R**2.*(1.-self._1m1overb2*nu.sin(phi)**2.)
            return R**2./(Rt2+(z/self._q)**2.+self._core2)\
                *nu.sin(2.*phi)*self._1m1overb2/2.
        else:
            return 0

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
        """
        if self.isNonAxi:
            R2= R**2.
            Rt2= R2*(1.-self._1m1overb2*nu.sin(phi)**2.)
            denom= 1./(Rt2+(z/self._q)**2.+self._core2)
            denom2= denom**2.
            return 1./4./nu.pi\
                *(2.*Rt2/R2*(denom-Rt2*denom2)\
                      +denom/self._q**2.-2.*z**2.*denom**2./self._q**4.\
                      -self._1m1overb2\
                      *(2.*R2*nu.sin(2.*phi)**2./4.*self._1m1overb2\
                            *denom**2.+denom*nu.cos(2.*phi)))
        else:
            return 1./4./nu.pi/self._q**2.*((2.*self._q**2.+1.)*self._core2+R**2.\
                                                +(2.-self._q**-2.)*z**2.)/\
                                                (R**2.+(z/self._q)**2.+self._core2)**2.

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
        if self.isNonAxi:
            Rt2= R**2.*(1.-self._1m1overb2*nu.sin(phi)**2.)
            denom= 1./(Rt2+(z/self._q)**2.+self._core2)
            return (denom-2.*Rt2*denom**2.)*Rt2/R**2.
        else:
            denom= 1./(R**2.+(z/self._q)**2.+self._core2)
            return denom-2.*R**2.*denom**2.

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
        if self.isNonAxi:
            Rt2= R**2.*(1.-self._1m1overb2*nu.sin(phi)**2.)
            denom= 1./(Rt2+(z/self._q)**2.+self._core2)
            return denom/self._q**2.-2.*z**2.*denom**2./self._q**4.
        else:
            denom= 1./(R**2.+(z/self._q)**2.+self._core2)
            return denom/self._q**2.-2.*z**2.*denom**2./self._q**4.

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
           d2Phi/dR/dz
        HISTORY:
           2013-08-28 - Written - Bovy (IAS)
        """
        if self.isNonAxi:
            Rt2= R**2.*(1.-self._1m1overb2*nu.sin(phi)**2.)
            return -2.*Rt2/R*z/self._q**2./(Rt2+(z/self._q)**2.+self._core2)**2.
        else:
            return -2.*R*z/self._q**2./(R**2.+(z/self._q)**2.+self._core2)**2.

    def _phi2deriv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _phi2deriv
        PURPOSE:
           evaluate the second azimuthal derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the second azimuthal derivative
        HISTORY:
           2017-10-15 - Written - Bovy (UofT)
        """
        if self.isNonAxi:
            Rt2= R**2.*(1.-self._1m1overb2*nu.sin(phi)**2.)
            denom= 1./(Rt2+(z/self._q)**2.+self._core2)
            return -self._1m1overb2\
                *(R**4.*nu.sin(2.*phi)**2./2.*self._1m1overb2\
                      *denom**2.
                  +R**2.*denom*nu.cos(2.*phi))
        else:
            return 0.

    def _Rphideriv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _Rphideriv
        PURPOSE:
           evaluate the mixed R,phi derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           d2Phi/dR/dphi
        HISTORY:
           2017-10-15 - Written - Bovy (UofT)
        """
        if self.isNonAxi:
            Rt2= R**2.*(1.-self._1m1overb2*nu.sin(phi)**2.)
            denom= 1./(Rt2+(z/self._q)**2.+self._core2)
            return -(denom-Rt2*denom**2.)*R*nu.sin(2.*phi)*self._1m1overb2 
        else:
            return 0.

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
        warnings.warn("NEMO's LogPot does not allow flattening in z (for some reason); therefore, flip y and z in NEMO wrt galpy; also does not allow the triaxial b parameter",galpyWarning)
        ampl= self._amp*vo**2.
        return "0,%s,%s,1.0,%s" % (ampl,
                                  self._core2*ro**2.*self._q**(2./3.), #somewhat weird gyrfalcon implementation
                                  self._q)
