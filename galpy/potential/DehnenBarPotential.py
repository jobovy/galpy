###############################################################################
#   DehnenBarPotential: Dehnen (2000)'s bar potential
###############################################################################
import numpy
from galpy.util import bovy_conversion
from .Potential import Potential, _APY_LOADED
if _APY_LOADED:
    from astropy import units
_degtorad= numpy.pi/180.
class DehnenBarPotential(Potential):
    """Class that implements the Dehnen bar potential (`Dehnen 2000 <http://adsabs.harvard.edu/abs/2000AJ....119..800D>`__), generalized to 3D following `Monari et al. (2016) <http://adsabs.harvard.edu/abs/2016MNRAS.461.3835M>`__

    .. math::

        \\Phi(R,z,\\phi) = A_b(t)\\,\\cos\\left(2\\,(\\phi-\\Omega_b\\,t)\\right))\\,\\left(\\frac{R}{r}\\right)^2\\,\\times \\begin{cases}
        -(R_b/r)^3\\,, & \\text{for}\\ r \\geq R_b\\\\
        (r/R_b)^3-2\\,, & \\text{for}\\ r\\leq R_b.
        \\end{cases}

    where :math:`r^2 = R^2+z^2` is the spherical radius and

    .. math::

        A_b(t) = A_f\\,\\left(\\frac{3}{16}\\xi^5-\\frac{5}{8}\\xi^3+\\frac{15}{16}\\xi+\\frac{1}{2}\\right)\\,, \\xi = 2\\frac{t/T_b-t_\\mathrm{form}}{T_\mathrm{steady}}-1\\,,\ \mathrm{if}\ t_\\mathrm{form} \\leq \\frac{t}{T_b} \\leq t_\\mathrm{form}+T_\\mathrm{steady}

    and 

    .. math::

        A_b(t) = \\begin{cases}
        0\\,, & \\frac{t}{T_b} < t_\mathrm{form}\\\\
        A_f\\,, & \\frac{t}{T_b} > t_\mathrm{form}+T_\mathrm{steady}
        \\end{cases}

    where

    .. math::

       T_b = \\frac{2\pi}{\\Omega_b}

    is the bar period and the strength can also be specified using :math:`\\alpha`

    .. math::

       \\alpha = 3\\,\\frac{A_f}{v_0^2}\\,\\left(\\frac{R_b}{r_0}\\right)^3\,.

    """
    normalize= property() # turn off normalize
    def __init__(self,amp=1.,omegab=None,rb=None,chi=0.8,
                 rolr=0.9,barphi=25.*_degtorad,
                 tform=-4.,tsteady=None,beta=0.,
                 alpha=0.01,Af=None,ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a Dehnen bar potential

        INPUT:

           amp - amplitude to be applied to the potential (default:
           1., see alpha or Ab below)

           barphi - angle between sun-GC line and the bar's major axis
           (in rad; default=25 degree; or can be Quantity))

           tform - start of bar growth / bar period (default: -4)

           tsteady - time from tform at which the bar is fully grown / bar period (default: -tform/2, st the perturbation is fully grown at tform/2)

           Either provide:

              a) rolr - radius of the Outer Lindblad Resonance for a
                 circular orbit (can be Quantity)
              
                 chi - fraction R_bar / R_CR (corotation radius of bar)

                 alpha - relative bar strength (default: 0.01)

                 beta - power law index of rotation curve (to
                 calculate OLR, etc.)
               
              b) omegab - rotation speed of the bar (can be Quantity)
              
                 rb - bar radius (can be Quantity)
                 
                 Af - bar strength (can be Quantity)
              
        OUTPUT:

           (none)

        HISTORY:

           2010-11-24 - Started - Bovy (NYU)

           2017-06-23 - Converted to 3D following Monari et al. (2016) - Bovy (UofT/CCA)

        """
        Potential.__init__(self,amp=amp,ro=ro,vo=vo)
        if _APY_LOADED and isinstance(barphi,units.Quantity):
            barphi= barphi.to(units.rad).value
        if _APY_LOADED and isinstance(rolr,units.Quantity):
            rolr= rolr.to(units.kpc).value/self._ro
        if _APY_LOADED and isinstance(rb,units.Quantity):
            rb= rb.to(units.kpc).value/self._ro
        if _APY_LOADED and isinstance(omegab,units.Quantity):
            omegab= omegab.to(units.km/units.s/units.kpc).value\
                /bovy_conversion.freq_in_kmskpc(self._vo,self._ro)
        if _APY_LOADED and isinstance(Af,units.Quantity):
            Af= Af.to(units.km**2/units.s**2).value/self._vo**2.
        self.hasC= True
        self.hasC_dxdv= True
        self.isNonAxi= True
        self._barphi= barphi
        if omegab is None:
            self._rolr= rolr
            self._chi= chi
            self._beta= beta
            #Calculate omegab and rb
            self._omegab= 1./((self._rolr**(1.-self._beta))/(1.+numpy.sqrt((1.+self._beta)/2.)))
            self._rb= self._chi*self._omegab**(1./(self._beta-1.))
            self._alpha= alpha
            self._af= self._alpha/3./self._rb**3.
        else:
            self._omegab= omegab
            self._rb= rb
            self._af= Af
        self._tb= 2.*numpy.pi/self._omegab
        self._tform= tform*self._tb
        if tsteady is None:
            self._tsteady= self._tform/2.
        else:
            self._tsteady= self._tform+tsteady*self._tb

    def _evaluate(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _evaluate
        PURPOSE:
           evaluate the potential at R,phi,t
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           Phi(R,z,phi,t)
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
        r2= R**2.+z**2.
        r= numpy.sqrt(r2)
        if r <= self._rb:
            return self._af*smooth*numpy.cos(2.*(phi-self._omegab*t-self._barphi))\
                *((r/self._rb)**3.-2.)*R**2./r2
        else:
            return -self._af*smooth*numpy.cos(2.*(phi-self._omegab*t-
                                                  self._barphi))\
                                                  *(self._rb/r)**3.\
                                                  *R**2./r2

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
        r= numpy.sqrt(R**2.+z**2.)
        if r <= self._rb:
            return -self._af*smooth*numpy.cos(2.*(phi-self._omegab*t
                                                  -self._barphi))\
                    *((r/self._rb)**3.*R*(3.*R**2.+2.*z**2.)-4.*R*z**2.)/r**4.
        else:
            return -self._af*smooth*numpy.cos(2.*(phi-self._omegab*t-
                                                  self._barphi))\
                    *(self._rb/r)**3.*R/r**4.*(3.*R**2.-2.*z**2.)
        
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
        r2= R**2.+z**2.
        r= numpy.sqrt(r2)
        if r <= self._rb:
            return 2.*self._af*smooth*numpy.sin(2.*(phi-self._omegab*t-
                                                    self._barphi))\
                                                *((r/self._rb)**3.-2.)*R**2./r2
        else:
            return -2.*self._af*smooth*numpy.sin(2.*(phi-self._omegab*t-
                                                     self._barphi))\
                                                     *(self._rb/r)**3.*R**2./r2

    def _zforce(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _zforce
        PURPOSE:
           evaluate the vertical  force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the vertical force
        HISTORY:
           2017-06-23 - Written - Bovy (NYU)
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
        r= numpy.sqrt(R**2.+z**2.)
        if r <= self._rb:
            return -self._af*smooth*numpy.cos(2.*(phi-self._omegab*t
                                                  -self._barphi))\
                   *((r/self._rb)**3.+4.)*R**2.*z/r**4.
        else:
            return -5.*self._af*smooth*numpy.cos(2.*(phi-self._omegab*t-
                                                  self._barphi))\
                    *(self._rb/r)**3.*R**2.*z/r**4.
        
    def _R2deriv(self,R,z,phi=0.,t=0.):
        #Calculate relevant time
        if t < self._tform:
            smooth= 0.
        elif t < self._tsteady:
            deltat= t-self._tform
            xi= 2.*deltat/(self._tsteady-self._tform)-1.
            smooth= (3./16.*xi**5.-5./8*xi**3.+15./16.*xi+.5)
        else: #bar is fully on
            smooth= 1.
        r= numpy.sqrt(R**2.+z**2.)
        if r <= self._rb:
            return self._af*smooth*numpy.cos(2.*(phi-self._omegab*t
                                                 -self._barphi))\
                *((r/self._rb)**3.*((9.*R**2.+2.*z**2.)/r**4.
                                    -R**2./r**6.*(3.*R**2.+2.*z**2.))\
                      +4.*z**2./r**6.*(4.*R**2.-r**2.))
        else:
            return self._af*smooth*numpy.cos(2.*(phi-self._omegab*t-
                                                 self._barphi))\
                *(self._rb/r)**3./r**6.*((r**2.-7.*R**2.)*(3.*R**2.-2.*z**2.)\
                                             +6.*R**2.*r**2.)
        
    def _phi2deriv(self,R,z,phi=0.,t=0.):
        #Calculate relevant time
        if t < self._tform:
            smooth= 0.
        elif t < self._tsteady:
            deltat= t-self._tform
            xi= 2.*deltat/(self._tsteady-self._tform)-1.
            smooth= (3./16.*xi**5.-5./8*xi**3.+15./16.*xi+.5)
        else: #bar is fully on
            smooth= 1.
        r= numpy.sqrt(R**2.+z**2.)
        if r <= self._rb:
            return -4.*self._af*smooth*numpy.cos(2.*(phi-self._omegab*t-
                                                     self._barphi))\
                                            *((r/self._rb)**3.-2.)*R**2./r**2.
        else:
            return 4.*self._af*smooth*numpy.cos(2.*(phi-self._omegab*t-
                                                    self._barphi))\
                                                 *(self._rb/r)**3.*R**2./r**2.

    def _Rphideriv(self,R,z,phi=0.,t=0.):
        #Calculate relevant time
        if t < self._tform:
            smooth= 0.
        elif t < self._tsteady:
            deltat= t-self._tform
            xi= 2.*deltat/(self._tsteady-self._tform)-1.
            smooth= (3./16.*xi**5.-5./8*xi**3.+15./16.*xi+.5)
        else: #bar is fully on
            smooth= 1.
        r= numpy.sqrt(R**2.+z**2.)
        if r <= self._rb:
            return -2.*self._af*smooth*numpy.sin(2.*(phi-self._omegab*t
                                                  -self._barphi))\
                    *((r/self._rb)**3.*R*(3.*R**2.+2.*z**2.)-4.*R*z**2.)/r**4.
        else:
            return -2.*self._af*smooth*numpy.sin(2.*(phi-self._omegab*t-
                                                  self._barphi))\
                    *(self._rb/r)**3.*R/r**4.*(3.*R**2.-2.*z**2.)

    def _z2deriv(self,R,z,phi=0.,t=0.):
        #Calculate relevant time
        if t < self._tform:
            smooth= 0.
        elif t < self._tsteady:
            deltat= t-self._tform
            xi= 2.*deltat/(self._tsteady-self._tform)-1.
            smooth= (3./16.*xi**5.-5./8*xi**3.+15./16.*xi+.5)
        else: #bar is fully on
            smooth= 1.
        r= numpy.sqrt(R**2.+z**2.)
        if r <= self._rb:
            return self._af*smooth*numpy.cos(2.*(phi-self._omegab*t
                                                  -self._barphi))\
                   *R**2./r**6.*((r/self._rb)**3.*(r**2.-z**2.)
                                 +4.*(r**2.-4.*z**2.))
        else:
            return 5.*self._af*smooth*numpy.cos(2.*(phi-self._omegab*t-
                                                  self._barphi))\
                    *(self._rb/r)**3.*R**2./r**6.*(r**2.-7.*z**2.)
        
    def _Rzderiv(self,R,z,phi=0.,t=0.):
        #Calculate relevant time
        if t < self._tform:
            smooth= 0.
        elif t < self._tsteady:
            deltat= t-self._tform
            xi= 2.*deltat/(self._tsteady-self._tform)-1.
            smooth= (3./16.*xi**5.-5./8*xi**3.+15./16.*xi+.5)
        else: #bar is fully on
            smooth= 1.
        r= numpy.sqrt(R**2.+z**2.)
        if r <= self._rb:
            return self._af*smooth*numpy.cos(2.*(phi-self._omegab*t
                                                  -self._barphi))\
                   *R*z/r**6.*((r/self._rb)**3.*(2.*r**2.-R**2.)
                                 +8.*(r**2.-2.*R**2.))
        else:
            return 5.*self._af*smooth*numpy.cos(2.*(phi-self._omegab*t-
                                                  self._barphi))\
                    *(self._rb/r)**3.*R*z/r**6.*(2.*r**2.-7.*R**2.)
        
    def tform(self): #pragma: no cover
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
