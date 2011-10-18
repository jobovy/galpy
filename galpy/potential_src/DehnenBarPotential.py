###############################################################################
#   DehnenBarPotential: Dehnen (2000)'s bar potential
###############################################################################
import math as m
from planarPotential import planarPotential
_degtorad= m.pi/180.
class DehnenBarPotential(planarPotential):
    """Class that implements the Dehnen bar potential (Dehnen 2000)
    """
    def __init__(self,amp=1.,omegab=None,rb=None,chi=0.8,
                 rolr=0.9,barphi=25.*_degtorad,
                 tform=-4.,tsteady=None,beta=0.,
                 alpha=0.01,Af=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a Dehnen bar potential

        INPUT:

           amp - amplitude to be applied to the potential (default:
           1., see alpha or Ab below)

           barphi - angle between sun-GC line and the bar's major axis
           (in rad; default=25 degree)

           tform - start of bar growth / bar period (default: -4)

           tsteady - time at which the bar is fully grown / bar period
           (default: tform/2)

           Either provide:

              a) rolr - radius of the Outer Lindblad Resonance for a
                 circular orbit
              
                 chi - fraction R_bar / R_CR (corotation radius of bar)

                 alpha - relative bar strength (default: 0.01)

                 beta - power law index of rotation curve (to
                 calculate OLR, etc.)
               
              b) omegab - rotation speed of the bar
              
                 rb - bar radius
                 
                 Af - bar strength
              
        OUTPUT:

           (none)

        HISTORY:

           2010-11-24 - Started - Bovy (NYU)

        """
        planarPotential.__init__(self,amp=amp)
        self.hasC= True
        self._barphi= barphi
        if omegab is None:
            self._rolr= rolr
            self._chi= chi
            self._beta= beta
            #Calculate omegab and rb
            self._omegab= 1./((self._rolr**(1.-self._beta))/(1.+m.sqrt((1.+self._beta)/2.)))
            self._rb= self._chi*self._omegab**(1./(self._beta-1.))
            self._alpha= alpha
            self._af= self._alpha/3./self._rb**3.
        else:
            self._omegab= omegab
            self._rb= rb
            self._af= Af
        self._tb= 2.*m.pi/self._omegab
        self._tform= tform*self._tb
        if tsteady is None:
            self._tsteady= self._tform/2.
        else:
            self._tsteady= tsteady*self._tb

    def _evaluate(self,R,phi=0.,t=0.,dR=0,dphi=0):
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
           2010-11-24 - Started - Bovy (NYU)
        """
        #Calculate relevant time
        if t < self._tform:
            smooth= 0.
        elif t < self._tsteady:
            deltat= t-self._tform
            xi= 2.*deltat/(self._tsteady-self._tform)-1.
            smooth= (3./16.*xi**5.-5./8*xi**3.+15./16.*xi+.5)
        else: #bar is fully on
            smooth= 1.
        if dR == 0 and dphi == 0:
            if R <= self._rb:
                return self._af*smooth*m.cos(2.*(phi-self._omegab*t-self._barphi))\
                       *((R/self._rb)**3.-2.)
            else:
                return -self._af*smooth*m.cos(2.*(phi-self._omegab*t-
                                                  self._barphi))\
                                                  *(self._rb/R)**3.
        elif dR == 1 and dphi == 0:
            return -self._Rforce(R,phi=phi,t=t)
        elif dR == 0 and dphi == 1:
            return -self._phiforce(R,phi=phi,t=t)
        elif dR == 2 and dphi == 0:
            return self._R2deriv(R,phi=phi,t=t)
        elif dR == 0 and dphi == 2:
            return self._phi2deriv(R,phi=phi,t=t)
        elif dR == 1 and dphi == 1:
            return self._Rphideriv(R,phi=phi,t=t)
        
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
        #Calculate relevant time
        if t < self._tform:
            smooth= 0.
        elif t < self._tsteady:
            deltat= t-self._tform
            xi= 2.*deltat/(self._tsteady-self._tform)-1.
            smooth= (3./16.*xi**5.-5./8*xi**3.+15./16.*xi+.5)
        else: #bar is fully on
            smooth= 1.
        if R <= self._rb:
            return -3.*self._af*smooth*m.cos(2.*(phi-self._omegab*t
                                              -self._barphi))\
                                              *(R/self._rb)**3./R
        else:
            return -3.*self._af*smooth*m.cos(2.*(phi-self._omegab*t-
                                              self._barphi))\
                                              *(self._rb/R)**3./R
        
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
        #Calculate relevant time
        if t < self._tform:
            smooth= 0.
        elif t < self._tsteady:
            deltat= t-self._tform
            xi= 2.*deltat/(self._tsteady-self._tform)-1.
            smooth= (3./16.*xi**5.-5./8*xi**3.+15./16.*xi+.5)
        else: #bar is fully on
            smooth= 1.
        if R <= self._rb:
            return 2.*self._af*smooth*m.sin(2.*(phi-self._omegab*t-
                                                self._barphi))\
                                                *((R/self._rb)**3.-2.)
        else:
            return -2.*self._af*smooth*m.sin(2.*(phi-self._omegab*t-
                                                 self._barphi))\
                                                 *(self._rb/R)**3.

    def _R2deriv(self,R,phi=0.,t=0.):
        #Calculate relevant time
        if t < self._tform:
            smooth= 0.
        elif t < self._tsteady:
            deltat= t-self._tform
            xi= 2.*deltat/(self._tsteady-self._tform)-1.
            smooth= (3./16.*xi**5.-5./8*xi**3.+15./16.*xi+.5)
        else: #bar is fully on
            smooth= 1.
        if R <= self._rb:
            return 6.*self._af*smooth*m.cos(2.*(phi-self._omegab*t
                                              -self._barphi))\
                                              *(R/self._rb)**3./R**2.
        else:
            return -12.*self._af*smooth*m.cos(2.*(phi-self._omegab*t-
                                                  self._barphi))\
                                                  *(self._rb/R)**3./R**2.
        
    def _phi2deriv(self,R,phi=0.,t=0.):
        #Calculate relevant time
        if t < self._tform:
            smooth= 0.
        elif t < self._tsteady:
            deltat= t-self._tform
            xi= 2.*deltat/(self._tsteady-self._tform)-1.
            smooth= (3./16.*xi**5.-5./8*xi**3.+15./16.*xi+.5)
        else: #bar is fully on
            smooth= 1.
        if R <= self._rb:
            return -4.*self._af*smooth*m.cos(2.*(phi-self._omegab*t-
                                                 self._barphi))\
                                                 *((R/self._rb)**3.-2.)
        else:
            return 4.*self._af*smooth*m.cos(2.*(phi-self._omegab*t-
                                                 self._barphi))\
                                                 *(self._rb/R)**3.       

    def _Rphideriv(self,R,phi=0.,t=0.):
        #Calculate relevant time
        if t < self._tform:
            smooth= 0.
        elif t < self._tsteady:
            deltat= t-self._tform
            xi= 2.*deltat/(self._tsteady-self._tform)-1.
            smooth= (3./16.*xi**5.-5./8*xi**3.+15./16.*xi+.5)
        else: #bar is fully on
            smooth= 1.
        if R <= self._rb:
            return -6.*self._af*smooth*m.sin(2.*(phi-self._omegab*t
                                              -self._barphi))\
                                              *(R/self._rb)**3./R
        else:
            return -6.*self._af*smooth*m.sin(2.*(phi-self._omegab*t-
                                              self._barphi))\
                                              *(self._rb/R)**3./R

    def tform(self):
        """
        NAME:

           tform

        PURPOSE:

           return formation time of the bar

        INPUT:

           (none)

        OUTPUT:

           tform in normalized units

        HISTORY:

           2011-03-08 - Written - Bovy (NYU)

        """
        return self._tform

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
        return self._omegab
