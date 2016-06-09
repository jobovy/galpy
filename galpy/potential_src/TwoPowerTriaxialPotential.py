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
import hashlib
import numpy
from scipy import integrate
from galpy.util import bovy_conversion, bovy_coords
from galpy.util import _rotate_to_arbitrary_vector
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
    def __init__(self,amp=1.,a=5.,alpha=1.5,beta=3.5,b=1.,c=1.,
                 zvec=None,pa=None,
                 normalize=False,ro=None,vo=None):
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

           zvec= (None) If set, a unit vector that corresponds to the z axis

           pa= (None) If set, the position angle of the x axis (rad or Quantity)

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
                                                      zvec=zvec,pa=pa,
                                                      normalize=False)
            self.HernquistSelf= HernquistSelf
            self.JaffeSelf= None
            self.NFWSelf= None
        elif alpha == 2 and beta == 4:
            Potential.__init__(self,amp=amp,ro=ro,vo=vo,amp_units='mass')
            JaffeSelf= TriaxialJaffePotential(amp=1.,a=a,b=b,c=c,
                                              zvec=zvec,pa=pa,
                                              normalize=False)
            self.HernquistSelf= None
            self.JaffeSelf= JaffeSelf
            self.NFWSelf= None
        elif alpha == 1 and beta == 3:
            Potential.__init__(self,amp=amp,ro=ro,vo=vo,amp_units='mass')
            NFWSelf= TriaxialNFWPotential(amp=1.,a=a,b=b,c=c,pa=pa,
                                          zvec=zvec,normalize=False)
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
        self._setup_zvec_pa(zvec,pa)
        if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)): #pragma: no cover
            self.normalize(normalize)
        if numpy.fabs(self._b-1.) > 10.**-10.:
            self.isNonAxi= True
        return None

    def _setup_zvec_pa(self,zvec,pa):
        if zvec is None and pa is None:
            self._aligned= True
        else:
            self._aligned= False
            if not pa is None:
                if _APY_LOADED and isinstance(pa,units.Quantity):
                    pa= pa.to(units.rad).value
                pa_rot= numpy.array([[numpy.cos(pa),numpy.sin(pa),0.],
                                     [-numpy.sin(pa),numpy.cos(pa),0.],
                                     [0.,0.,1.]])
            else:
                pa_rot= numpy.eye(3)
            if not zvec is None:
                if not isinstance(zvec,numpy.ndarray):
                    zvec= numpy.array(zvec)
                zvec/= numpy.sqrt(numpy.sum(zvec**2.))
                zvec_rot= _rotate_to_arbitrary_vector(\
                    numpy.array([[0.,0.,1.]]),zvec,inv=True)[0]
            else:
                zvec_rot= numpy.eye(3)
            self._rot= numpy.dot(pa_rot,zvec_rot)
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
        if self._aligned:
            return self._evaluate_xyz(x,y,z)
        else:
            xyzp= numpy.dot(self._rot,numpy.array([x,y,z]))
            return self._evaluate_xyz(xyzp[0],xyzp[1],xyzp[2])

    def _evaluate_xyz(self,x,y,z):
        if not self.HernquistSelf == None:
            return self.HernquistSelf._evaluate_xyz(x,y,z)
        elif not self.JaffeSelf == None:
            return self.JaffeSelf._evaluate_xyz(x,y,z)
        elif not self.NFWSelf == None:
            return self.NFWSelf._evaluate_xyz(x,y,z)
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
           2016-05-31 - Written - Bovy (UofT)
        """
        return 0.

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
        if numpy.fabs(self._b-1.) > 10.**-10.:
            self.isNonAxi= True
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
        if numpy.fabs(self._b-1.) > 10.**-10.:
            self.isNonAxi= True
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

class TriaxialNFWPotential(TwoPowerTriaxialPotential):
    """Class that implements the triaxial NFW potential

    .. math::

        \\rho(r) = \\frac{\\mathrm{amp}}{4\\,\\pi\\,a^3}\\,\\frac{1}{(r/a)\\,(1+r/a)^{2}}

    with

    .. math::

        m^2 = x^2 + \\frac{y^2}{b^2}+\\frac{z^2}{c^2}
    """
    def __init__(self,amp=1.,a=1.5,b=0.9,c=0.7,zvec=None,pa=None,
                 normalize=False,
                 conc=None,mvir=None,
                 vo=None,ro=None,
                 H=70.,Om=0.3,overdens=200.,wrtcrit=False):
        """
        NAME:

           __init__

        PURPOSE:

           Initialize a triaxial NFW potential

        INPUT:

           amp - amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass or Gxmass

           a - scale radius (can be Quantity)

           b - y-to-x axis ratio of the density

           c - z-to-x axis ratio of the density

           zvec= (None) If set, a unit vector that corresponds to the z axis

           pa= (None) If set, the position angle of the x axis

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
        self._b= b
        self._b2= self._b**2.
        self._c= c
        self._c2= self._c**2.
        self._setup_zvec_pa(zvec,pa)
        self._force_hash= None
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
        self.hasC= False
        self.hasC_dxdv= False
        if numpy.fabs(self._b-1.) > 10.**-10.:
            self.isNonAxi= True
        return None

    def _evaluate_xyz(self,x,y,z):
        """
        NAME:
           __evaluate_xyz
        PURPOSE:
           evaluate the potential at x,y,z
        INPUT:
           x,y,z- rectangular position
        OUTPUT:
           Phi(x,y,z)
        HISTORY:
           2016-05-30 - Started - Bovy (UofT)
           2016-06-08 - Re-written to be a function of (x,y,z) - Bovy (UofT)
        """
        psi= lambda m: 1./(1.+m/self.a)
        return -self._b*self._c/self.a\
            *_potInt(x,y,z,psi,self._b2,self._c2)

    def _xforce(self,x,y,z):
        return -self._b*self._c/self.a**3.\
            *_forceInt(x,y,z,
                       lambda m: (self.a/m)**self.alpha/(1.+m/self.a)**(self.beta-self.alpha),
                       self._b2,self._c2,0)
        
    def _yforce(self,x,y,z):
        return -self._b*self._c/self.a**3.\
            *_forceInt(x,y,z,
                       lambda m: (self.a/m)**self.alpha/(1.+m/self.a)**(self.beta-self.alpha),
                       self._b2,self._c2,1)

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
        x,y,z= bovy_coords.cyl_to_rect(R,phi,z)
        new_hash= hashlib.md5(numpy.array([x,y,z])).hexdigest()
        if new_hash == self._force_hash:
            Fx= self._cached_Fx
            Fy= self._cached_Fy
        else:
            Fx= self._xforce(x,y,z)
            Fy= self._yforce(x,y,z)
            self._force_hash= new_hash
            self._cached_Fx= Fx
            self._cached_Fy= Fy
        return numpy.cos(phi)*Fx+numpy.sin(phi)*Fy

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
        x,y,z= bovy_coords.cyl_to_rect(R,phi,z)
        return -self._b*self._c/self.a**3.\
            *_forceInt(x,y,z,
                       lambda m: (self.a/m)**self.alpha/(1.+m/self.a)**(self.beta-self.alpha),
                       self._b2,self._c2,2)

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
           the radial force
        HISTORY:
           2010-07-09 - Written - Bovy (UofT)
        """
        x,y,z= bovy_coords.cyl_to_rect(R,phi,z)
        new_hash= hashlib.md5(numpy.array([x,y,z])).hexdigest()
        if new_hash == self._force_hash:
            Fx= self._cached_Fx
            Fy= self._cached_Fy
        else:
            Fx= self._xforce(x,y,z)
            Fy= self._yforce(x,y,z)
            self._force_hash= new_hash
            self._cached_Fx= Fx
            self._cached_Fy= Fy
        return R*(-numpy.sin(phi)*Fx+numpy.cos(phi)*Fy)

def _potInt(x,y,z,psi,b2,c2):
    """int_0^\infty psi~(m))/sqrt([1+tau]x[b^2+tau]x[c^2+tau])dtau, 
    where psi~(m) = [psi(\infty)-psi(m)]/[2Aa^2], with A=amp/[4pia^3]"""
    def integrand(s):
        t= 1/s**2.-1.
        return psi(numpy.sqrt(x**2./(1.+t)+y**2./(b2+t)+z**2./(c2+t)))\
            /numpy.sqrt((1.+(b2-1.)*s**2.)*(1.+(c2-1.)*s**2.))
    return integrate.quad(integrand,0.,1.)[0]                              

def _forceInt(x,y,z,dens,b2,c2,i):
    """Integral that gives the force in x,y,z"""
    def integrand(s):
        t= 1/s**2.-1.
        return dens(numpy.sqrt(x**2./(1.+t)+y**2./(b2+t)+z**2./(c2+t)))\
            *(x/(1.+t)*(i==0)+y/(b2+t)*(i==1)+z/(c2+t)*(i==2))\
            /numpy.sqrt((1.+(b2-1.)*s**2.)*(1.+(c2-1.)*s**2.))
    return integrate.quad(integrand,0.,1.)[0]                              
