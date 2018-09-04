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
from scipy import integrate, special
from .Potential import _APY_LOADED
from .EllipsoidalPotential import EllipsoidalPotential
if _APY_LOADED:
    from astropy import units
class TwoPowerTriaxialPotential(EllipsoidalPotential):
    """Class that implements triaxial potentials that are derived from 
    two-power density models

    .. math::

        \\rho(x,y,z) = \\frac{\\mathrm{amp}}{4\\,\\pi\\,a^3}\\,\\frac{1}{(m/a)^\\alpha\\,(1+m/a)^{\\beta-\\alpha}}

    with

    .. math::

        m^2 = x'^2 + \\frac{y'^2}{b^2}+\\frac{z'^2}{c^2}

    and :math:`(x',y',z')` is a rotated frame wrt :math:`(x,y,z)` specified by parameters ``zvec`` and ``pa`` which specify (a) ``zvec``: the location of the :math:`z'` axis in the :math:`(x,y,z)` frame and (b) ``pa``: the position angle of the :math:`x'` axis wrt the :math:`\\tilde{x}` axis, that is, the :math:`x` axis after rotating to ``zvec``.

    Note that this general class of potentials does *not* automatically revert to the special TriaxialNFWPotential, TriaxialHernquistPotential, or TriaxialJaffePotential when using their (alpha,beta) values (like TwoPowerSphericalPotential).
    """
    def __init__(self,amp=1.,a=5.,alpha=1.5,beta=3.5,b=1.,c=1.,
                 zvec=None,pa=None,glorder=50,
                 normalize=False,ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a triaxial two-power-density potential

        INPUT:

           amp - amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass or Gxmass

           a - scale radius (can be Quantity)

           alpha - inner power (0 <= alpha < 3)

           beta - outer power ( beta > 2)

           b - y-to-x axis ratio of the density

           c - z-to-x axis ratio of the density

           zvec= (None) If set, a unit vector that corresponds to the z axis

           pa= (None) If set, the position angle of the x axis (rad or Quantity)

           glorder= (50) if set, compute the relevant force and potential integrals with Gaussian quadrature of this order

           normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.

           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

        OUTPUT:

           (none)

        HISTORY:

           2016-05-30 - Started - Bovy (UofT)

           2018-08-07 - Re-written using the general EllipsoidalPotential class - Bovy (UofT)

        """
        EllipsoidalPotential.__init__(self,amp=amp,b=b,c=c,
                                      zvec=zvec,pa=pa,glorder=glorder,
                                      ro=ro,vo=vo,amp_units='mass')
        if _APY_LOADED and isinstance(a,units.Quantity):
            a= a.to(units.kpc).value/self._ro
        self.a= a
        self._scale= self.a
        if beta <= 2. or alpha < 0. or alpha >= 3.:
            raise IOError('TwoPowerTriaxialPotential requires 0 <= alpha < 3 and beta > 2')
        self.alpha= alpha
        self.beta= beta
        self.betaminusalpha= self.beta-self.alpha
        self.twominusalpha= 2.-self.alpha
        self.threeminusalpha= 3.-self.alpha
        if self.twominusalpha != 0.:
            self.psi_inf= special.gamma(self.beta-2.)\
                *special.gamma(3.-self.alpha)\
                /special.gamma(self.betaminusalpha)
        # Adjust amp
        self._amp/= (4.*numpy.pi*self.a**3)
        if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)): #pragma: no cover
            self.normalize(normalize)
        return None

    def _psi(self,m):
        """\psi(m) = -\int_m^\infty d m^2 \rho(m^2)"""
        if self.twominusalpha == 0.:
            return -2.*self.a**2*(self.a/m)**self.betaminusalpha\
                      /self.betaminusalpha\
                *special.hyp2f1(self.betaminusalpha,
                                self.betaminusalpha,
                                self.betaminusalpha+1,
                                -self.a/m)
        else:
            return -2.*self.a**2\
                *(self.psi_inf-(m/self.a)**self.twominusalpha\
                      /self.twominusalpha\
                      *special.hyp2f1(self.twominusalpha,
                                      self.betaminusalpha,
                                      self.threeminusalpha,
                                      -m/self.a))

    def _mdens(self,m):
        """Density as a function of m"""
        return (self.a/m)**self.alpha/(1.+m/self.a)**(self.betaminusalpha)

    def _mdens_deriv(self,m):
        """Derivative of the density as a function of m"""
        return -self._mdens(m)*(self.a*self.alpha+self.beta*m)/m/(self.a+m)

class TriaxialHernquistPotential(EllipsoidalPotential):
    """Class that implements the triaxial Hernquist potential

    .. math::

        \\rho(x,y,z) = \\frac{\\mathrm{amp}}{4\\,\\pi\\,a^3}\\,\\frac{1}{(m/a)\\,(1+m/a)^{3}}

    with

    .. math::

        m^2 = x'^2 + \\frac{y'^2}{b^2}+\\frac{z'^2}{c^2}

    and :math:`(x',y',z')` is a rotated frame wrt :math:`(x,y,z)` specified by parameters ``zvec`` and ``pa`` which specify (a) ``zvec``: the location of the :math:`z'` axis in the :math:`(x,y,z)` frame and (b) ``pa``: the position angle of the :math:`x'` axis wrt the :math:`\\tilde{x}` axis, that is, the :math:`x` axis after rotating to ``zvec``.

    """
    def __init__(self,amp=1.,a=2.,normalize=False,b=1.,c=1.,zvec=None,pa=None,
                 glorder=50,ro=None,vo=None):
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

           zvec= (None) If set, a unit vector that corresponds to the z axis

           pa= (None) If set, the position angle of the x axis

           glorder= (50) if set, compute the relevant force and potential integrals with Gaussian quadrature of this order

           normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.

           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

        OUTPUT:

           (none)

        HISTORY:

           2010-07-09 - Written - Bovy (UofT)

           2018-08-07 - Re-written using the general EllipsoidalPotential class - Bovy (UofT)

        """
        EllipsoidalPotential.__init__(self,amp=amp,b=b,c=c,
                                      zvec=zvec,pa=pa,glorder=glorder,
                                      ro=ro,vo=vo,amp_units='mass')
        if _APY_LOADED and isinstance(a,units.Quantity):
            a= a.to(units.kpc).value/self._ro
        self.a= a
        self._scale= self.a
        # Adjust amp
        self.a4= self.a**4
        self._amp/= (4.*numpy.pi*self.a**3)
        if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)):
            self.normalize(normalize)
        self.hasC= not self._glorder is None
        self.hasC_dxdv= False
        return None

    def _psi(self,m):
        """\psi(m) = -\int_m^\infty d m^2 \rho(m^2)"""
        return -self.a4/(m+self.a)**2.

    def _mdens(self,m):
        """Density as a function of m"""
        return self.a4/m/(m+self.a)**3

    def _mdens_deriv(self,m):
        """Derivative of the density as a function of m"""
        return -self.a4*(self.a+4.*m)/m**2/(self.a+m)**4

class TriaxialJaffePotential(EllipsoidalPotential):
    """Class that implements the Jaffe potential

    .. math::

        \\rho(x,y,z) = \\frac{\\mathrm{amp}}{4\\,\\pi\\,a^3}\\,\\frac{1}{(m/a)^2\\,(1+m/a)^{2}}

    with

    .. math::

        m^2 = x'^2 + \\frac{y'^2}{b^2}+\\frac{z'^2}{c^2}

    and :math:`(x',y',z')` is a rotated frame wrt :math:`(x,y,z)` specified by parameters ``zvec`` and ``pa`` which specify (a) ``zvec``: the location of the :math:`z'` axis in the :math:`(x,y,z)` frame and (b) ``pa``: the position angle of the :math:`x'` axis wrt the :math:`\\tilde{x}` axis, that is, the :math:`x` axis after rotating to ``zvec``.

    """
    def __init__(self,amp=1.,a=2.,b=1.,c=1.,zvec=None,pa=None,normalize=False,
                 glorder=50,ro=None,vo=None):
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

           zvec= (None) If set, a unit vector that corresponds to the z axis

           pa= (None) If set, the position angle of the x axis

           glorder= (50) if set, compute the relevant force and potential integrals with Gaussian quadrature of this order

           normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.

           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

        OUTPUT:

           (none)

        HISTORY:

           2010-07-09 - Written - Bovy (UofT)

           2018-08-07 - Re-written using the general EllipsoidalPotential class - Bovy (UofT)

        """
        EllipsoidalPotential.__init__(self,amp=amp,b=b,c=c,
                                      zvec=zvec,pa=pa,glorder=glorder,
                                      ro=ro,vo=vo,amp_units='mass')
        if _APY_LOADED and isinstance(a,units.Quantity):
            a= a.to(units.kpc).value/self._ro
        self.a= a
        self._scale= self.a
        # Adjust amp
        self.a2= self.a**2
        self._amp/= (4.*numpy.pi*self.a2*self.a)
        if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)): #pragma: no cover
            self.normalize(normalize)
        self.hasC= not self._glorder is None
        self.hasC_dxdv= False
        return None

    def _psi(self,m):
        """\psi(m) = -\int_m^\infty d m^2 \rho(m^2)"""
        return 2.*self.a2*(1./(1.+m/self.a)+numpy.log(m/(m+self.a)))

    def _mdens(self,m):
        """Density as a function of m"""
        return self.a2/m**2/(1.+m/self.a)**2

    def _mdens_deriv(self,m):
        """Derivative of the density as a function of m"""
        return -2.*self.a2**2*(self.a+2.*m)/m**3/(self.a+m)**3
  
class TriaxialNFWPotential(EllipsoidalPotential):
    """Class that implements the triaxial NFW potential

    .. math::

        \\rho(x,y,z) = \\frac{\\mathrm{amp}}{4\\,\\pi\\,a^3}\\,\\frac{1}{(m/a)\\,(1+m/a)^{2}}

    with

    .. math::

        m^2 = x'^2 + \\frac{y'^2}{b^2}+\\frac{z'^2}{c^2}

    and :math:`(x',y',z')` is a rotated frame wrt :math:`(x,y,z)` specified by parameters ``zvec`` and ``pa`` which specify (a) ``zvec``: the location of the :math:`z'` axis in the :math:`(x,y,z)` frame and (b) ``pa``: the position angle of the :math:`x'` axis wrt the :math:`\\tilde{x}` axis, that is, the :math:`x` axis after rotating to ``zvec``.

    """
    def __init__(self,amp=1.,a=2.,b=1.,c=1.,zvec=None,pa=None,
                 normalize=False,
                 conc=None,mvir=None,
                 glorder=50,vo=None,ro=None,
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

           glorder= (50) if set, compute the relevant force and potential integrals with Gaussian quadrature of this order

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

           2018-08-06 - Re-written using the general EllipsoidalPotential class - Bovy (UofT)

        """
        EllipsoidalPotential.__init__(self,amp=amp,b=b,c=c,
                                      zvec=zvec,pa=pa,glorder=glorder,
                                      ro=ro,vo=vo,amp_units='mass')
        if _APY_LOADED and isinstance(a,units.Quantity):
            a= a.to(units.kpc).value/self._ro
        if conc is None:
            self.a= a
        else:
            from galpy.potential import NFWPotential
            dum= NFWPotential(mvir=mvir,conc=conc,ro=self._ro,vo=self._vo,
                              H=H,Om=Om,wrtcrit=wrtcrit,overdens=overdens)
            self.a= dum.a
            self._amp= dum._amp
        self._scale= self.a
        self.hasC= not self._glorder is None
        self.hasC_dxdv= False
        # Adjust amp
        self.a3= self.a**3
        self._amp/= (4.*numpy.pi*self.a3)
        if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)):
            self.normalize(normalize)
        return None

    def _psi(self,m):
        """\psi(m) = -\int_m^\infty d m^2 \rho(m^2)"""
        return -2.*self.a3/(self.a+m)

    def _mdens(self,m):
        """Density as a function of m"""
        return self.a/m/(1.+m/self.a)**2

    def _mdens_deriv(self,m):
        """Derivative of the density as a function of m"""
        return -self.a3*(self.a+3.*m)/m**2/(self.a+m)**3
