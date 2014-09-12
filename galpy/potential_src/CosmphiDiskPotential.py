###############################################################################
#   EllipticalDiskPotential: Kuijken & Tremaine (1994)'s elliptical disk 
#   potential
###############################################################################
import math
from planarPotential import planarPotential
_degtorad= math.pi/180.
class CosmphiDiskPotential(planarPotential):
    """Class that implements the disk potential

    .. math::

        \\Phi(R,\\phi) = \\phi_0\\,R^p\\,\\cos\\left(m\\,(\\phi-\\phi_b)\\right)

    This potential can be grown between  :math:`t_{\mathrm{form}}` and  :math:`t_{\mathrm{form}}+T_{\mathrm{steady}}` in a similar way as DehnenBarPotential, but times are given directly in galpy time units

   """
    def __init__(self,amp=1.,phib=25.*_degtorad,
                 p=1.,phio=0.01,m=1.,
                 tform=None,tsteady=None,
                 cp=None,sp=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize an cosmphi disk potential

           phi(R,phi) = phio (R/Ro)^p cos[m(phi-phib)]

        INPUT:

           amp=  amplitude to be applied to the potential (default:
           1.), see twophio below

           tform= start of growth (to smoothly grow this potential

           tsteady= time delay at which the perturbation is fully grown (default: 2.)

           m= cos( m * (phi - phib) )

           p= power-law index of the phi(R) = (R/Ro)^p part

           Either:
           
              a) phib= angle (in rad; default=25 degree)

                 phio= potential perturbation (in terms of phio/vo^2 if vo=1 at Ro=1)
                 
              b) cp, sp= m * phio * cos(m * phib), m * phio * sin(m * phib)

        OUTPUT:

           (none)

        HISTORY:

           2011-10-27 - Started - Bovy (IAS)

        """
        planarPotential.__init__(self,amp=amp)
        self.hasC= False
        self._m= m
        if cp is None or sp is None:
            self._phib= phib
            self._mphio= phio*self._m
        else:
            self._mphio= math.sqrt(cp*cp+sp*sp)
            self._phib= math.atan(sp/cp)/self._m
            if m < 2. and cp < 0.:
                self._phib= math.pi+self._phib
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
        return smooth*self._mphio/self._m*R**self._p\
            *math.cos(self._m*(phi-self._phib))
        
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
        return -smooth*self._p*self._mphio/self._m*R**(self._p-1.)\
            *math.cos(self._m*(phi-self._phib))
        
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
        return smooth*self._mphio*R**self._p*math.sin(self._m*(phi-self._phib))

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
        return smooth*self._p*(self._p-1.)/self._m*self._mphio*R**(self._p-2.)\
            *math.cos(self._m*(phi-self._phib))
        
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
        return -self._m*smooth*self._mphio*R**self._p*math.cos(self._m*(phi-self._phib))

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
        return -smooth*self._p*self._mphio*R**(self._p-1.)*math.sin(self._m*(phi-self._phib))

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

class LopsidedDiskPotential(CosmphiDiskPotential):
    """Class that implements the disk potential

    .. math::

        \\Phi(R,\\phi) = \\phi_0\\,R^p\\,\\cos\\left(\\phi-\\phi_b\\right)

    See documentation for CosmphiDiskPotential
   """
    def __init__(self,amp=1.,phib=25.*_degtorad,
                 p=1.,phio=0.01,
                 tform=None,tsteady=None,
                 cp=None,sp=None):
        CosmphiDiskPotential.__init__(self,
                                      amp=amp,
                                      phib=phib,
                                      p=p,phio=phio,m=1.,
                                      tform=tform,tsteady=tsteady,
                                      cp=cp,sp=sp)
        self.hasC= True
        self.hasC_dxdv= True
