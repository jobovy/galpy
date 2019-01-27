###############################################################################
#   CosmphiDiskPotential: cos(mphi) potential
###############################################################################
import math
from .planarPotential import planarPotential, _APY_LOADED
if _APY_LOADED:
    from astropy import units
_degtorad= math.pi/180.
class CosmphiDiskPotential(planarPotential):
    """Class that implements the disk potential

    .. math::

        \\Phi(R,\\phi) = \\mathrm{amp}\\,\\phi_0\\,\\,\\cos\\left[m\\,(\\phi-\\phi_b)\\right]\\times \\begin{cases}
        \\left(\\frac{R}{R_1}\\right)^p\\,, & \\text{for}\\ R \\geq R_b\\\\
        \\left[2-\\left(\\frac{R_b}{R}\\right)^p\\right]\\times\\left(\\frac{R_b}{R_1}\\right)^p\\,, & \\text{for}\\ R\\leq R_b.
        \\end{cases}

    This potential can be grown between  :math:`t_{\mathrm{form}}` and  :math:`t_{\mathrm{form}}+T_{\mathrm{steady}}` in a similar way as DehnenBarPotential by wrapping it with a DehnenSmoothWrapperPotential

   """
    def __init__(self,amp=1.,phib=25.*_degtorad,
                 p=1.,phio=0.01,m=4,r1=1.,rb=None,
                 cp=None,sp=None,ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize an cosmphi disk potential

        INPUT:

           amp= amplitude to be applied to the potential (default:
           1.), degenerate with phio below, but kept for overall
           consistency with potentials

           m= cos( m * (phi - phib) ), integer

           p= power-law index of the phi(R) = (R/Ro)^p part

           r1= (1.) normalization radius for the amplitude (can be Quantity); amp x phio is only the potential at (R,phi) = (r1,pib) when r1 > rb; otherwise more complicated

           rb= (None) if set, break radius for power-law: potential R^p at R > Rb, R^-p at R < Rb, potential and force continuous at Rb


           Either:
           
              a) phib= angle (in rad; default=25 degree; or can be Quantity)

                 phio= potential perturbation (in terms of phio/vo^2 if vo=1 at Ro=1; or can be Quantity with units of velocity-squared)
                 
              b) cp, sp= m * phio * cos(m * phib), m * phio * sin(m * phib); can be Quantity with units of velocity-squared)

        OUTPUT:

           (none)

        HISTORY:

           2011-10-27 - Started - Bovy (IAS)

           2017-09-16 - Added break radius rb - Bovy (UofT)

        """
        planarPotential.__init__(self,amp=amp,ro=ro,vo=vo)
        if _APY_LOADED and isinstance(phib,units.Quantity):
            phib= phib.to(units.rad).value
        if _APY_LOADED and isinstance(r1,units.Quantity):
            r1= r1.to(units.kpc).value/self._ro
        if _APY_LOADED and isinstance(rb,units.Quantity):
            rb= rb.to(units.kpc).value/self._ro
        if _APY_LOADED and isinstance(phio,units.Quantity):
            phio= phio.to(units.km**2/units.s**2).value/self._vo**2.
        if _APY_LOADED and isinstance(cp,units.Quantity):
            cp= cp.to(units.km**2/units.s**2).value/self._vo**2.
        if _APY_LOADED and isinstance(sp,units.Quantity):
            sp= sp.to(units.km**2/units.s**2).value/self._vo**2.
        # Back to old definition
        self._r1p= r1**p
        self._amp/= self._r1p
        self.hasC= False
        self._m= int(m) # make sure this is an int
        if cp is None or sp is None:
            self._phib= phib
            self._mphio= phio*self._m
        else:
            self._mphio= math.sqrt(cp*cp+sp*sp)
            self._phib= math.atan(sp/cp)/self._m
            if m < 2. and cp < 0.:
                self._phib= math.pi+self._phib
        self._p= p
        if rb is None:
            self._rb= 0.
            self._rbp= 1. # never used, but for p < 0 general expr fails
            self._rb2p= 1.
        else:
            self._rb= rb
            self._rbp= self._rb**self._p
            self._rb2p= self._rbp**2.
        self._mphib= self._m*self._phib
        self.hasC= True
        self.hasC_dxdv= True

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
           2011-10-19 - Started - Bovy (IAS)
        """
        if R < self._rb:
            return self._mphio/self._m*math.cos(self._m*phi-self._mphib)\
                *self._rbp*(2.*self._r1p-self._rbp/R**self._p)
        else:
            return self._mphio/self._m*R**self._p\
                *math.cos(self._m*phi-self._mphib)
        
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
           2011-10-19 - Written - Bovy (IAS)
        """
        if R < self._rb:
            return -self._p*self._mphio/self._m*self._rb2p/R**(self._p+1.)\
                *math.cos(self._m*phi-self._mphib)
        else:
            return -self._p*self._mphio/self._m*R**(self._p-1.)\
                *math.cos(self._m*phi-self._mphib)
        
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
           2011-10-19 - Written - Bovy (IAS)
        """
        if R < self._rb:
            return self._mphio*math.sin(self._m*phi-self._mphib)\
                *self._rbp*(2.*self._r1p-self._rbp/R**self._p)
        else:
            return self._mphio*R**self._p*math.sin(self._m*phi-self._mphib)

    def _R2deriv(self,R,phi=0.,t=0.):
        if R < self._rb:
            return -self._p*(self._p+1.)*self._mphio/self._m\
                *self._rb2p/R**(self._p+2.)*math.cos(self._m*phi-self._mphib)
        else:
            return self._p*(self._p-1.)/self._m*self._mphio*R**(self._p-2.)\
                *math.cos(self._m*phi-self._mphib)
        
    def _phi2deriv(self,R,phi=0.,t=0.):
        if R < self._rb:
            return -self._m*self._mphio*math.cos(self._m*phi-self._mphib)\
                *self._rbp*(2.*self._r1p-self._rbp/R**self._p)
        else:
            return -self._m*self._mphio*R**self._p\
                *math.cos(self._m*phi-self._mphib)

    def _Rphideriv(self,R,phi=0.,t=0.):
        if R < self._rb:
            return -self._p*self._mphio/self._m*self._rb2p/R**(self._p+1.)\
                *math.sin(self._m*phi-self._mphib)
        else:
            return -self._p*self._mphio*R**(self._p-1.)*\
                math.sin(self._m*phi-self._mphib)

class LopsidedDiskPotential(CosmphiDiskPotential):
    """Class that implements the disk potential

    .. math::

        \\Phi(R,\\phi) = \\mathrm{amp}\\,\\phi_0\\,\\left(\\frac{R}{R_1}\\right)^p\\,\\cos\\left(\\phi-\\phi_b\\right)

   Special case of CosmphiDiskPotential with m=1; see documentation for CosmphiDiskPotential
   """
    def __init__(self,amp=1.,phib=25.*_degtorad,
                 p=1.,phio=0.01,r1=1.,
                 cp=None,sp=None,ro=None,vo=None):
        CosmphiDiskPotential.__init__(self,
                                      amp=amp,
                                      phib=phib,
                                      p=p,phio=phio,m=1.,
                                      cp=cp,sp=sp,ro=ro,vo=vo)
        self.hasC= True
        self.hasC_dxdv= True
