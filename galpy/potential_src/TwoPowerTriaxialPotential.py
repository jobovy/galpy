###############################################################################
#   TwoPowerTriaxialPotential.py: General class for triaxial potentials 
#                                 derived from densities with two power-laws
#
#                                                    amp/[4pia^3]
#                             rho(r)= ------------------------------------
#                                      (m/a)^\alpha (1+m/a)^(\beta-\alpha)
#
#                             with
#
#                             m^2 = x^2 + y^2/b^2 + z^2/c^2
###############################################################################
import numpy
from scipy import integrate
from galpy.util import bovy_conversion, bovy_coords
from galpy.potential_src.Potential import Potential, _APY_LOADED
if _APY_LOADED:
    from astropy import units
class TwoPowerTriaxialPotential(Potential):
    """Class that implements triaxial potentials that are derived from 
    two-power density models

    .. math::

        \\rho(x,y,z) = \\frac{\\mathrm{amp}}{4\\,\\pi\\,a^3}\\,\\frac{1}{(m/a)^\\alpha\\,(1+m/a)^{\\beta-\\alpha}}

    with

    .. math::

        m^2 = x^2 + \\frac{y^2}{b^2}+\\frac{z^2}{c^2}
    """
    def __init__(self,amp=1.,a=5.,alpha=1.5,beta=3.5,b=1.,c=1.,normalize=False,
                 ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a triaxial two-power-density potential

        INPUT:

           amp - amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass or Gxmass

           a - scale radius (can be Quantity)

           alpha - inner power

           beta - outer power

           b - y-to-x axis ratio of the density

           c - z-to-x axis ratio of the density

           normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.

           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

        OUTPUT:

           (none)

        HISTORY:

           2016-05-30 - Started - Bovy (UofT)

        """
        if alpha == 1 and beta == 4:
            Potential.__init__(self,amp=amp,ro=ro,vo=vo,amp_units='mass')
            HernquistSelf= TriaxialHernquistPotential(amp=1.,a=a,b=b,c=c,
                                                      normalize=False)
            self.HernquistSelf= HernquistSelf
            self.JaffeSelf= None
            self.NFWSelf= None
        elif alpha == 2 and beta == 4:
            Potential.__init__(self,amp=amp,ro=ro,vo=vo,amp_units='mass')
            JaffeSelf= TriaxialJaffePotential(amp=1.,a=a,b=b,c=c,
                                              normalize=False)
            self.HernquistSelf= None
            self.JaffeSelf= JaffeSelf
            self.NFWSelf= None
        elif alpha == 1 and beta == 3:
            Potential.__init__(self,amp=amp,ro=ro,vo=vo,amp_units='mass')
            NFWSelf= TriaxialNFWPotential(amp=1.,a=a,b=b,c=c,normalize=False)
            self.HernquistSelf= None
            self.JaffeSelf= None
            self.NFWSelf= NFWSelf
        else:
            Potential.__init__(self,amp=amp,ro=ro,vo=vo,amp_units='mass')
            self.HernquistSelf= None
            self.JaffeSelf= None
            self.NFWSelf= None
        if _APY_LOADED and isinstance(a,units.Quantity):
            a= a.to(units.kpc).value/self._ro
        self.a= a
        self._scale= self.a
        self.alpha= alpha
        self.beta= beta
        self._b= b
        self._c= c
        self._b2= self._b**2.
        self._c2= self._c**2.
        if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)): #pragma: no cover
            self.normalize(normalize)
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
           2016-05-30 - Started - Bovy (UofT)
        """
        if not self.HernquistSelf == None:
            return self.HernquistSelf._evaluate(R,z,phi=phi,t=t)
        elif not self.JaffeSelf == None:
            return self.JaffeSelf._evaluate(R,z,phi=phi,t=t)
        elif not self.NFWSelf == None:
            return self.NFWSelf._evaluate(R,z,phi=phi,t=t)
        else:
            raise NotImplementedError("General potential expression not yet implemented")

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
           2010-07-09 - Written - Bovy (UofT)
        """
        if not self.HernquistSelf == None:
            return self.HernquistSelf._Rforce(R,z,phi=phi,t=t)
        elif not self.JaffeSelf == None:
            return self.JaffeSelf._Rforce(R,z,phi=phi,t=t)
        elif not self.NFWSelf == None:
            return self.NFWSelf._Rforce(R,z,phi=phi,t=t)
        else:
            raise NotImplementedError("General potential expression not yet implemented")

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
           2010-07-09 - Written - Bovy (UofT)
        """
        if not self.HernquistSelf == None:
            return self.HernquistSelf._zforce(R,z,phi=phi,t=t)
        elif not self.JaffeSelf == None:
            return self.JaffeSelf._zforce(R,z,phi=phi,t=t)
        elif not self.NFWSelf == None:
            return self.NFWSelf._zforce(R,z,phi=phi,t=t)
        else:
            raise NotImplementedError("General potential expression not yet implemented")

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
           2016-05-31 - Written - Bovy (UofT)
        """
        x,y,z= bovy_coords.cyl_to_rect(R,phi,z)
        m= numpy.sqrt(x**2.+y**2./self._b2+z**2./self._c2)
        return (self.a/m)**self.alpha/(1.+m/self.a)**(self.beta-self.alpha)/4./numpy.pi/self.a**3.
        
class TriaxialHernquistPotential(TwoPowerTriaxialPotential):
    """Class that implements the triaxial Hernquist potential

    .. math::

        \\rho(x,y,z) = \\frac{\\mathrm{amp}}{4\\,\\pi\\,a^3}\\,\\frac{1}{(m/a)\\,(1+m/a)^{3}}

    with

    .. math::

        m^2 = x^2 + \\frac{y^2}{b^2}+\\frac{z^2}{c^2}
    """
    def __init__(self,amp=1.,a=1.,normalize=False,b=1.,c=1.,
                 ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           Initialize a triaxial Hernquist potential

        INPUT:

           amp - amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass or Gxmass

           a - scale radius (can be Quantity)

           b - y-to-x axis ratio of the density

           c - z-to-x axis ratio of the density

           normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.

           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

        OUTPUT:

           (none)

        HISTORY:

           2010-07-09 - Written - Bovy (UofT)

        """
        Potential.__init__(self,amp=amp,ro=ro,vo=vo,amp_units='mass')
        if _APY_LOADED and isinstance(a,units.Quantity):
            a= a.to(units.kpc).value/self._ro
        self.a= a
        self._scale= self.a
        self.alpha= 1
        self.beta= 4
        self._b= b
        self._c= c
        if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)):
            self.normalize(normalize)
        self.hasC= False
        self.hasC_dxdv= False
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
           2010-07-09 - Started - Bovy (UofT)
        """
        raise NotImplementedError("Triaxial Hernquist potential expression not yet implemented")

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
           t- time
        OUTPUT:
           the radial force
        HISTORY:
           2010-07-09 - Written - Bovy (UofT)
        """
        raise NotImplementedError("Triaxial Hernquist potential expression not yet implemented")

    def _zforce(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _zforce
        PURPOSE:
           evaluate the vertical force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           t - time
        OUTPUT:
           the vertical force
        HISTORY:
           2010-07-09 - Written - Bovy (UofT)
        """
        raise NotImplementedError("Triaxial Hernquist potential expression not yet implemented")

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
           t- time
        OUTPUT:
           the second radial derivative
        HISTORY:
           2011-10-09 - Written - Bovy (UofT)
        """
        raise NotImplementedError("Triaxial Hernquist potential expression not yet implemented")

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
           t- time
        OUTPUT:
           d2phi/dR/dz
        HISTORY:
           2013-08-28 - Written - Bovy (UofT)
        """
        raise NotImplementedError("Triaxial Hernquist potential expression not yet implemented")

    def _mass(self,R,z=0.,t=0.):
        """
        NAME:
           _mass
        PURPOSE:
           calculate the mass out to a given radius
        INPUT:
           R - radius at which to return the enclosed mass
           z - (don't specify this) vertical height
        OUTPUT:
           mass in natural units
        HISTORY:
           2014-01-29 - Written - Bovy (UofT)
        """
        if z is None: r= R
        else: r= numpy.sqrt(R**2.+z**2.)
        return (r/self.a)**2./2./(1.+r/self.a)**2.

class TriaxialJaffePotential(TwoPowerTriaxialPotential):
    """Class that implements the Jaffe potential

    .. math::

        \\rho(x,y,z) = \\frac{\\mathrm{amp}}{4\\,\\pi\\,a^3}\\,\\frac{1}{(m/a)^2\\,(1+m/a)^{2}}

    with

    .. math::

        m^2 = x^2 + \\frac{y^2}{b^2}+\\frac{z^2}{c^2}
    """
    def __init__(self,amp=1.,a=1.,b=1.,c=1.,normalize=False,
                 ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           Initialize a Jaffe potential

        INPUT:

           amp - amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass or Gxmass

           a - scale radius (can be Quantity)

           b - y-to-x axis ratio of the density

           c - z-to-x axis ratio of the density

           normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.

           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

        OUTPUT:

           (none)

        HISTORY:

           2010-07-09 - Written - Bovy (UofT)

        """
        Potential.__init__(self,amp=amp,ro=ro,vo=vo,amp_units='mass')
        if _APY_LOADED and isinstance(a,units.Quantity):
            a= a.to(units.kpc).value/self._ro
        self.a= a
        self._scale= self.a
        self.alpha= 2
        self.beta= 4
        if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)): #pragma: no cover
            self.normalize(normalize)
        self.hasC= False
        self.hasC_dxdv= False
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
           2010-07-09 - Started - Bovy (UofT)
        """
        raise NotImplementedError("Triaxial Jaffe potential expression not yet implemented")

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
           2010-07-09 - Written - Bovy (UofT)
        """
        raise NotImplementedError("Triaxial Jaffe potential expression not yet implemented")

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
           2010-07-09 - Written - Bovy (UofT)
        """
        raise NotImplementedError("Triaxial Jaffe potential expression not yet implemented")

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
           2011-10-09 - Written - Bovy (UofT)
        """
        raise NotImplementedError("Triaxial Jaffe potential expression not yet implemented")

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
           2013-08-28 - Written - Bovy (UofT)
        """
        raise NotImplementedError("Triaxial Jaffe potential expression not yet implemented")

    def _mass(self,R,z=0.,t=0.):
        """
        NAME:
           _mass
        PURPOSE:
           calculate the mass out to a given radius
        INPUT:
           R - radius at which to return the enclosed mass
           z - (don't specify this) vertical height
        OUTPUT:
           mass in natural units
        HISTORY:
           2014-01-29 - Written - Bovy (UofT)
        """
        if z is None: r= R
        else: r= numpy.sqrt(R**2.+z**2.)
        return r/self.a/(1.+r/self.a)

class TriaxialNFWPotential(TwoPowerTriaxialPotential):
    """Class that implements the triaxial NFW potential

    .. math::

        \\rho(r) = \\frac{\\mathrm{amp}}{4\\,\\pi\\,a^3}\\,\\frac{1}{(r/a)\\,(1+r/a)^{2}}

    with

    .. math::

        m^2 = x^2 + \\frac{y^2}{b^2}+\\frac{z^2}{c^2}
    """
    def __init__(self,amp=1.,a=1.,b=1.,c=1.,normalize=False,
                 conc=None,mvir=None,
                 vo=None,ro=None,
                 H=70.,Om=0.3,overdens=200.,wrtcrit=False):
        """
        NAME:

           __init__

        PURPOSE:

           Initialize a NFW potential

        INPUT:

           amp - amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass or Gxmass

           a - scale radius (can be Quantity)

           b - y-to-x axis ratio of the density

           c - z-to-x axis ratio of the density

           normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.


           Alternatively, NFW potentials can be initialized using 

              conc= concentration

              mvir= virial mass in 10^12 Msolar

           in which case you also need to supply the following keywords
           
              H= (default: 70) Hubble constant in km/s/Mpc
           
              Om= (default: 0.3) Omega matter
       
              overdens= (200) overdensity which defines the virial radius

              wrtcrit= (False) if True, the overdensity is wrt the critical density rather than the mean matter density
           
           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

        OUTPUT:

           (none)

        HISTORY:

           2016-05-30 - Written - Bovy (UofT)

        """
        Potential.__init__(self,amp=amp,ro=ro,vo=vo,amp_units='mass')
        if _APY_LOADED and isinstance(a,units.Quantity):
            a= a.to(units.kpc).value/self._ro
        if conc is None:
            self.a= a
            self.alpha= 1
            self.beta= 3
            if normalize or \
                    (isinstance(normalize,(int,float)) \
                         and not isinstance(normalize,bool)):
                self.normalize(normalize)
        else:
            if wrtcrit:
                od= overdens/bovy_conversion.dens_in_criticaldens(self._vo,
                                                                  self._ro,H=H)
            else:
                od= overdens/bovy_conversion.dens_in_meanmatterdens(self._vo,
                                                                    self._ro,
                                                                    H=H,Om=Om)
            mvirNatural= mvir*100./bovy_conversion.mass_in_1010msol(self._vo,
                                                                    self._ro)
            rvir= (3.*mvirNatural/od/4./numpy.pi)**(1./3.)
            self.a= rvir/conc
            self._amp= mvirNatural/(numpy.log(1.+conc)-conc/(1.+conc))
        self._scale= self.a
        self._b= b
        self._b2= self._b**2.
        self._c= c
        self._c2= self._c**2.
        self.hasC= False
        self.hasC_dxdv= False
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
           2016-05-30 - Started - Bovy (UofT)
        """
        x,y,z= bovy_coords.cyl_to_rect(R,phi,z)
        psi= lambda m: 1./(1.+m/self.a)
        return -self._b*self._c/self.a\
            *_potInt(x,y,z,psi,self._b2,self._c2)

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
           2010-07-09 - Written - Bovy (UofT)
        """
        raise NotImplementedError("Triaxial NFW potential expression not yet implemented")

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
           2010-07-09 - Written - Bovy (UofT)
        """
        raise NotImplementedError("Triaxial NFW potential expression not yet implemented")

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
           2011-10-09 - Written - Bovy (UofT)
        """
        raise NotImplementedError("Triaxial NFW potential expression not yet implemented")

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
           2013-08-28 - Written - Bovy (UofT)
        """
        raise NotImplementedError("Triaxial NFW potential expression not yet implemented")

    def _mass(self,R,z=0.,t=0.):
        """
        NAME:
           _mass
        PURPOSE:
           calculate the mass out to a given radius
        INPUT:
           R - radius at which to return the enclosed mass
           z - (don't specify this) vertical height
        OUTPUT:
           mass in natural units
        HISTORY:
           2014-01-29 - Written - Bovy (UofT)
        """
        if z is None: r= R
        else: r= numpy.sqrt(R**2.+z**2.)
        return numpy.log(1+r/self.a)-r/self.a/(1.+r/self.a)

    @bovy_conversion.physical_conversion('position',pop=True)
    def rvir(self,H=70.,Om=0.3,overdens=200.,wrtcrit=False):
        """
        NAME:

           rvir

        PURPOSE:

           calculate the virial radius for this density distribution

        INPUT:

           H= (default: 70) Hubble constant in km/s/Mpc
           
           Om= (default: 0.3) Omega matter
       
           overdens= (200) overdensity which defines the virial radius

           wrtcrit= (False) if True, the overdensity is wrt the critical density rather than the mean matter density

           ro= distance scale in kpc or as Quantity (default: object-wide, which if not set is 8 kpc))

           vo= velocity scale in km/s or as Quantity (default: object-wide, which if not set is 220 km/s))

        OUTPUT:
        
           virial radius
        
        HISTORY:

           2014-01-29 - Written - Bovy (UofT)

        """
        if wrtcrit:
            od= overdens/bovy_conversion.dens_in_criticaldens(self._vo,
                                                              self._ro,H=H)
        else:
            od= overdens/bovy_conversion.dens_in_meanmatterdens(self._vo,
                                                                self._ro,
                                                                H=H,Om=Om)
        dc= 12.*self.dens(self.a,0.,use_physical=False)/od
        x= optimize.brentq(lambda y: (numpy.log(1.+y)-y/(1.+y))/y**3.-1./dc,
                           0.01,100.)
        return x*self.a

def _potInt(x,y,z,psi,b2,c2):
    """int_0^\infty psi~(m))/sqrt([1+tau]x[b^2+tau]x[c^2+tau])dtau, 
    where psi~(m) = [psi(\infty)-psi(m)]/[2Aa^2], with A=amp/[4pia^3]"""
    def integrand(s):
        t= 1/s**2.-1.
        return psi(numpy.sqrt(x**2./(1.+t)+y**2./(b2+t)+z**2./(c2+t)))\
            /numpy.sqrt((1.+(b2-1.)*s**2.)*(1.+(c2-1.)*s**2.))
    return integrate.quad(integrand,0.,1.)[0]                              
