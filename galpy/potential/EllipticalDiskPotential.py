###############################################################################
#   EllipticalDiskPotential: Kuijken & Tremaine (1994)'s elliptical disk 
#   potential
###############################################################################
import numpy
from ..util import conversion
from .planarPotential import planarPotential
_degtorad= numpy.pi/180.
class EllipticalDiskPotential(planarPotential):
    """Class that implements the Elliptical disk potential of Kuijken & Tremaine (1994) 

    .. math::

        \\Phi(R,\\phi) = \\mathrm{amp}\\,\\phi_0\\,\\left(\\frac{R}{R_1}\\right)^p\\,\\cos\\left(2\\,(\\phi-\\phi_b)\\right)

    This potential can be grown between  :math:`t_{\mathrm{form}}` and  :math:`t_{\mathrm{form}}+T_{\mathrm{steady}}` in a similar way as DehnenBarPotential, but times are given directly in galpy time units

   """
    def __init__(self,amp=1.,phib=25.*_degtorad,
                 p=1.,twophio=0.01,r1=1.,
                 tform=None,tsteady=None,
                 cp=None,sp=None,ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize an Elliptical disk potential

           phi(R,phi) = phio (R/Ro)^p cos[2(phi-phib)]

        INPUT:

           amp=  amplitude to be applied to the potential (default:
           1.), see twophio below

           tform= start of growth (to smoothly grow this potential (can be Quantity)

           tsteady= time delay at which the perturbation is fully grown (default: 2.; can be Quantity)

           p= power-law index of the phi(R) = (R/Ro)^p part

           r1= (1.) normalization radius for the amplitude (can be Quantity)

           Either:
           
              a) phib= angle (in rad; default=25 degree; or can be Quantity)

                 twophio= potential perturbation (in terms of 2phio/vo^2 if vo=1 at Ro=1; can be Quantity with units of velocity-squared)
                 
              b) cp, sp= twophio * cos(2phib), twophio * sin(2phib) (can be Quantity with units of velocity-squared)

        OUTPUT:

           (none)

        HISTORY:

           2011-10-19 - Started - Bovy (IAS)

        """
        planarPotential.__init__(self,amp=amp,ro=ro,vo=vo)
        phib= conversion.parse_angle(phib)
        r1= conversion.parse_length(r1,ro=self._ro)
        tform= conversion.parse_time(tform,ro=self._ro,vo=self._vo)
        tsteady= conversion.parse_time(tsteady,ro=self._ro,vo=self._vo)
        twophio= conversion.parse_energy(twophio,vo=self._vo)
        cp= conversion.parse_energy(cp,vo=self._vo)
        sp= conversion.parse_energy(sp,vo=self._vo)
        # Back to old definition
        self._amp/= r1**p
        self.hasC= True
        self.hasC_dxdv= True
        if cp is None or sp is None:
            self._phib= phib
            self._twophio= twophio
        else:
            self._twophio= numpy.sqrt(cp*cp+sp*sp)
            self._phib= numpy.arctan2(sp,cp)/2.
        self._p= p
        if not tform is None:
            self._tform= tform
        else:
            self._tform= None
        if not tsteady is None:
            self._tsteady= self._tform+tsteady
        else:
            if self._tform is None: self._tsteady= None
            else: self._tsteady= self._tform+2.

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
        #Calculate relevant time
        if not self._tform is None:
            if t < self._tform:
                smooth= 0.
            elif t < self._tsteady:
                deltat= t-self._tform
                xi= 2.*deltat/(self._tsteady-self._tform)-1.
                smooth= (3./16.*xi**5.-5./8*xi**3.+15./16.*xi+.5)
            else: #fully on
                smooth= 1.
        else:
            smooth= 1.
        return smooth*self._twophio/2.*R**self._p\
            *numpy.cos(2.*(phi-self._phib))
        
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
        #Calculate relevant time
        if not self._tform is None:
            if t < self._tform:
                smooth= 0.
            elif t < self._tsteady:
                deltat= t-self._tform
                xi= 2.*deltat/(self._tsteady-self._tform)-1.
                smooth= (3./16.*xi**5.-5./8*xi**3.+15./16.*xi+.5)
            else: #fully on
                smooth= 1.
        else:
            smooth= 1.
        return -smooth*self._p*self._twophio/2.*R**(self._p-1.)\
            *numpy.cos(2.*(phi-self._phib))
        
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
        #Calculate relevant time
        if not self._tform is None:
            if t < self._tform:
                smooth= 0.
            elif t < self._tsteady:
                deltat= t-self._tform
                xi= 2.*deltat/(self._tsteady-self._tform)-1.
                smooth= (3./16.*xi**5.-5./8*xi**3.+15./16.*xi+.5)
            else: #fully on
                smooth= 1.
        else:
            smooth= 1.
        return smooth*self._twophio*R**self._p*numpy.sin(2.*(phi-self._phib))

    def _R2deriv(self,R,phi=0.,t=0.):
        #Calculate relevant time
        if not self._tform is None:
            if t < self._tform:
                smooth= 0.
            elif t < self._tsteady:
                deltat= t-self._tform
                xi= 2.*deltat/(self._tsteady-self._tform)-1.
                smooth= (3./16.*xi**5.-5./8*xi**3.+15./16.*xi+.5)
            else: #fully on
                smooth= 1.
        else:
            smooth= 1.
        return smooth*self._p*(self._p-1.)/2.*self._twophio*R**(self._p-2.)\
            *numpy.cos(2.*(phi-self._phib))
        
    def _phi2deriv(self,R,phi=0.,t=0.):
        #Calculate relevant time
        if not self._tform is None:
            if t < self._tform:
                smooth= 0.
            elif t < self._tsteady:
                deltat= t-self._tform
                xi= 2.*deltat/(self._tsteady-self._tform)-1.
                smooth= (3./16.*xi**5.-5./8*xi**3.+15./16.*xi+.5)
            else: #perturbation is fully on
                smooth= 1.
        else:
            smooth= 1.
        return -2.*smooth*self._twophio*R**self._p*numpy.cos(2.*(phi-self._phib))

    def _Rphideriv(self,R,phi=0.,t=0.):
        #Calculate relevant time
        if not self._tform is None:
            if t < self._tform:
                smooth= 0.
            elif t < self._tsteady:
                deltat= t-self._tform
                xi= 2.*deltat/(self._tsteady-self._tform)-1.
                smooth= (3./16.*xi**5.-5./8*xi**3.+15./16.*xi+.5)
            else: #perturbation is fully on
                smooth= 1.
        else:
            smooth= 1.
        return -smooth*self._p*self._twophio*R**(self._p-1.)*numpy.sin(2.*(phi-self._phib))

    def tform(self): #pragma: no cover
        """
        NAME:

           tform

        PURPOSE:

           return formation time of the perturbation

        INPUT:

           (none)

        OUTPUT:

           tform in normalized units

        HISTORY:

           2011-10-19 - Written - Bovy (IAS)

        """
        return self._tform
