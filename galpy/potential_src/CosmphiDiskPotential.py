###############################################################################
#   CosmphiDiskPotential: cos(mphi) potential
###############################################################################
import math
from galpy.potential_src.planarPotential import planarPotential, _APY_LOADED
if _APY_LOADED:
    from astropy import units
_degtorad= math.pi/180.
class CosmphiDiskPotential(planarPotential):
    """Class that implements the disk potential

    .. math::

        \\Phi(R,\\phi) = \\mathrm{amp}\\,\\phi_0\\,\\left(\\frac{R}{R_1}\\right)^p\\,\\cos\\left(m\\,(\\phi-\\phi_b)\\right)

    This potential can be grown between  :math:`t_{\mathrm{form}}` and  :math:`t_{\mathrm{form}}+T_{\mathrm{steady}}` in a similar way as DehnenBarPotential by wrapping it with a DehnenSmoothWrapperPotential

   """
    def __init__(self,amp=1.,phib=25.*_degtorad,
                 p=1.,phio=0.01,m=4,r1=1.,
                 cp=None,sp=None,ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize an cosmphi disk potential

           phi(R,phi) = phio (R/Ro)^p cos[m(phi-phib)]

        INPUT:

           amp=  amplitude to be applied to the potential (default:
           1.), see phio below

           m= cos( m * (phi - phib) ), integer

           p= power-law index of the phi(R) = (R/Ro)^p part

           r1= (1.) normalization radius for the amplitude (can be Quantity)

           Either:
           
              a) phib= angle (in rad; default=25 degree; or can be Quantity)

                 phio= potential perturbation (in terms of phio/vo^2 if vo=1 at Ro=1; or can be Quantity with units of velocity-squared)
                 
              b) cp, sp= m * phio * cos(m * phib), m * phio * sin(m * phib); can be Quantity with units of velocity-squared)

        OUTPUT:

           (none)

        HISTORY:

           2011-10-27 - Started - Bovy (IAS)

        """
        planarPotential.__init__(self,amp=amp,ro=ro,vo=vo)
        if _APY_LOADED and isinstance(phib,units.Quantity):
            phib= phib.to(units.rad).value
        if _APY_LOADED and isinstance(r1,units.Quantity):
            r1= r1.to(units.kpc).value/self._ro
        if _APY_LOADED and isinstance(phio,units.Quantity):
            phio= phio.to(units.km**2/units.s**2).value/self._vo**2.
        if _APY_LOADED and isinstance(cp,units.Quantity):
            cp= cp.to(units.km**2/units.s**2).value/self._vo**2.
        if _APY_LOADED and isinstance(sp,units.Quantity):
            sp= sp.to(units.km**2/units.s**2).value/self._vo**2.
        # Back to old definition
        self._amp/= r1**p
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
        return self._mphio*R**self._p*math.sin(self._m*phi-self._mphib)

    def _R2deriv(self,R,phi=0.,t=0.):
        return self._p*(self._p-1.)/self._m*self._mphio*R**(self._p-2.)\
            *math.cos(self._m*phi-self._mphib)
        
    def _phi2deriv(self,R,phi=0.,t=0.):
        return -self._m*self._mphio*R**self._p\
            *math.cos(self._m*phi-self._mphib)

    def _Rphideriv(self,R,phi=0.,t=0.):
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
