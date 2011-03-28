###############################################################################
#   SteadyLogSpiralPotential: a steady-state spiral potential
###############################################################################
import math as m
from planarPotential import planarPotential
_degtorad= m.pi/180.
class SteadyLogSpiralPotential(planarPotential):
    """Class that implements a steady-state spiral potential
    
    V(r,phi,t) = A/alpha cos(alpha ln(r) + m(phi - Omegas*t-gamma))

    """
    def __init__(self,amp=1.,omegas=0.65,A=0.035,
                 alpha=7.,m=2,gamma=m.pi/4.,p=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a logarithmic spiral potential

        INPUT:

           amp - amplitude to be applied to the potential (default:
           1., A below)

           gamma - angle between sun-GC line and the line connecting the 
                  peak of the spiral pattern at the Solar radius 
                  (in rad; default=45 degree)
        
           A - force amplitude (alpha*potential-amplitude; default=0.035)

           omegas= - pattern speed (default=0.65)

           m= number of arms
           
           Either provide:

              a) alpha=
               
              b) p= pitch angle
              
        OUTPUT:

           (none)

        HISTORY:

           2011-03-27 - Started - Bovy (NYU)

        """
        planarPotential.__init__(self,amp=amp)
        self._omegas= omegas
        self._A= A
        self._m= m
        self._gamma= gamma
        if not p is None:
            self._alpha= self._m*m.cot(p)
        else:
            self._alpha= alpha

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
        return self._A/self._alpha*m.cos(self._alpha*m.log(R)
                                         +self._m*(phi-self._omegas*t
                                                   -self._gamma))

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
        return self._A/R*m.cos(self._alpha*m.log(R)
                               +self._m*(phi-self._omegas*t
                                         -self._gamma))
        
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
        return self._A/self._alpha*self._m*m.cos(self._alpha*m.log(R)
                                                 +self._m*(phi-self._omegas*t
                                                           -self._gamma))
