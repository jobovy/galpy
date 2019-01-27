###############################################################################
#   TransientLogSpiralPotential: a transient spiral potential
###############################################################################
import math
from galpy.util import bovy_conversion
from .planarPotential import planarPotential, _APY_LOADED
if _APY_LOADED:
    from astropy import units
_degtorad= math.pi/180.
class TransientLogSpiralPotential(planarPotential):
    """Class that implements a steady-state spiral potential
    
    .. math::

        \\Phi(R,\\phi) = \\frac{\\mathrm{amp}(t)}{\\alpha}\\,\\cos\\left(\\alpha\,\ln R - m\\,(\\phi-\\Omega_s\\,t-\\gamma)\\right)

    where

    .. math::

        \\mathrm{amp}(t) = \\mathrm{amp}\\,\\times A\\,\\exp\\left(-\\frac{[t-t_0]^2}{2\\,\\sigma^2}\\right)

    """
    def __init__(self,amp=1.,omegas=0.65,A=-0.035,
                 alpha=-7.,m=2,gamma=math.pi/4.,p=None,
                 sigma=1.,to=0.,ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a transient logarithmic spiral potential localized 
           around to

        INPUT:

           amp - amplitude to be applied to the potential (default:
           1., A below)

           gamma - angle between sun-GC line and the line connecting the peak of the spiral pattern at the Solar radius (in rad; default=45 degree; can be Quantity)
        
           A - amplitude (alpha*potential-amplitude; default=0.035; can be Quantity)

           omegas= - pattern speed (default=0.65; can be Quantity)

           m= number of arms
           
           to= time at which the spiral peaks (can be Quantity)

           sigma= "spiral duration" (sigma in Gaussian amplitude; can be Quantity)
           
           Either provide:

              a) alpha=
               
              b) p= pitch angle (rad; can be Quantity)
              
        OUTPUT:

           (none)

        HISTORY:

           2011-03-27 - Started - Bovy (NYU)

        """
        planarPotential.__init__(self,amp=amp,ro=ro,vo=vo)
        if _APY_LOADED and isinstance(gamma,units.Quantity):
            gamma= gamma.to(units.rad).value
        if _APY_LOADED and isinstance(p,units.Quantity):
            p= p.to(units.rad).value
        if _APY_LOADED and isinstance(A,units.Quantity):
            A= A.to(units.km**2/units.s**2).value/self._vo**2.
        if _APY_LOADED and isinstance(omegas,units.Quantity):
            omegas= omegas.to(units.km/units.s/units.kpc).value\
                /bovy_conversion.freq_in_kmskpc(self._vo,self._ro)
        if _APY_LOADED and isinstance(to,units.Quantity):
            to= to.to(units.Gyr).value\
                /bovy_conversion.time_in_Gyr(self._vo,self._ro)
        if _APY_LOADED and isinstance(sigma,units.Quantity):
            sigma= sigma.to(units.Gyr).value\
                /bovy_conversion.time_in_Gyr(self._vo,self._ro)
        self._omegas= omegas
        self._A= A
        self._m= m
        self._gamma= gamma
        self._to= to
        self._sigma2= sigma**2.
        if not p is None:
            self._alpha= self._m/math.tan(p)
        else:
            self._alpha= alpha
        self.hasC= True

    def _evaluate(self,R,phi=0.,t=0.):
        """
        NAME:
           _evaluate
        PURPOSE:
           evaluate the potential at R,phi,t
        INPUT:
           R - Galactocentric cylindrical radius
           phi - azimuth
           t - time
        OUTPUT:
           Phi(R,phi,t)
        HISTORY:
           2011-03-27 - Started - Bovy (NYU)
        """
        return self._A*math.exp(-(t-self._to)**2./2./self._sigma2)\
            /self._alpha*math.cos(self._alpha*math.log(R)
                                  -self._m*(phi-self._omegas*t-self._gamma))

    def _Rforce(self,R,phi=0.,t=0.):
        """
        NAME:
           _Rforce
        PURPOSE:
           evaluate the radial force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           phi - azimuth
           t - time
        OUTPUT:
           the radial force
        HISTORY:
           2010-11-24 - Written - Bovy (NYU)
        """
        return self._A*math.exp(-(t-self._to)**2./2./self._sigma2)\
            /R*math.sin(self._alpha*math.log(R)
                        -self._m*(phi-self._omegas*t-self._gamma))
    
    def _phiforce(self,R,phi=0.,t=0.):
        """
        NAME:
           _phiforce
        PURPOSE:
           evaluate the azimuthal force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           phi - azimuth
           t - time
        OUTPUT:
           the azimuthal force
        HISTORY:
           2010-11-24 - Written - Bovy (NYU)
        """
        return -self._A*math.exp(-(t-self._to)**2./2./self._sigma2)\
            /self._alpha*self._m*math.sin(self._alpha*math.log(R)
                                          -self._m*(phi-self._omegas*t
                                                    -self._gamma))

    def OmegaP(self):
        """
        NAME:


           OmegaP

        PURPOSE:

           return the pattern speed

        INPUT:

           (none)

        OUTPUT:

           pattern speed

        HISTORY:

           2011-10-10 - Written - Bovy (IAS)

        """
        return self._omegas
