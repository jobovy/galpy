###############################################################################
#   FerrersPotential.py: General class for triaxial Ferrers Potential
#
#                             rho(r) = amp/[4pia^3] (1 - (m/a)^2)^n
#
#                             with
#
#                             m^2 = x^2 + y^2/b^2 + z^2/c^2
###############################################################################
import numpy
import hashlib
from scipy import integrate, special
from scipy.optimize import fsolve
from galpy.util import bovy_conversion, bovy_coords
from galpy.util import _rotate_to_arbitrary_vector
from galpy.potential_src.Potential import Potential, _APY_LOADED
from galpy.orbit_src import Orbit
if _APY_LOADED:
    from astropy import units

class FerrersPotential(Potential):
    """Class that implements triaxial Ferrers potentials

    .. math::

        \\rho(x,y,z) = \\frac{\\mathrm{amp}}{4\\,\\pi\\,a^3}\\,(1-(m/a)^2)^n

    with

    .. math::
   
        m^2 = x^2 + \\frac{y^2}{b^2}+\\frac{z^2}{c^2}
    """
    
    def __init__(self,amp=1.,a=1.,n=2,b=0.35,c=0.2375,omegab=0.001,
                 pa=0.,normalize=False,ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a triaxial two-power-density potential
        
        INPUT:
        
           amp - amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass or Gxmass
        
           a - scale radius (can be Quantity)
        
           n - power of Ferrers density (n > 0)
        
           b - y-to-x axis ratio of the density
        
           c - z-to-x axis ratio of the density

           omegab - rotation speed of the bar (can be Quantity)
        
           pa= (None) If set, the position angle of the x axis (rad or Quantity)
        
           normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.
        
           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)
        
        OUTPUT:
        
           (none)
        
        HISTORY:
        
           2016-06-30 - Started - Semyeong Oh
        
        """
        Potential.__init__(self,amp=amp,ro=ro,vo=vo,amp_units='mass')
        if _APY_LOADED and isinstance(a,units.Quantity):
            a= a.to(units.kpc).value/self._ro
        if _APY_LOADED and isinstance(omegab,units.Quantity):
            omegab= omegab.to(units.km/units.s/units.kpc).value\
                /bovy_conversion.freq_in_kmskpc(self._vo,self._ro)
        if _APY_LOADED and isinstance(pa,units.Quantity):
            pa= pa.to(units.rad).value
        self.a= a
        self._scale= self.a
        if n <= 0:
            raise IOError('FerrersPotential requires n > 0')
        self.n= n
        self._b= b
        self._c= c
        self._omegab= omegab
        self._a2= self.a**2
        self._b2= self._b**2.
        self._c2= self._c**2.
        self._force_hash= None
        self._pa = pa
        if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)): #pragma: no cover
            self.normalize(normalize)
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
           2016-05-30 - Started - Bovy (UofT)
        """
        if not self.isNonAxi:
            phi= 0.
        x,y,z= bovy_coords.cyl_to_rect(R,phi,z)
        xy= numpy.dot(self.rot(t),numpy.array([x,y]))
        x,y= xy[0],xy[1]     
        return self._evaluate_xyz(x,y,z)

    def _evaluate_xyz(self,x,y,z=0.):
        """Evaluation of the potential as a function of (x,y,z) in the 
        aligned coordinate frame"""
        return -1/4/(self.n+1)*self._b*self._c*_potInt(x,y,z,self._a2,self._b2,self._c2,self.n)

    def _Rforce(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _Rforce
        PURPOSE:T
           evaluate the radial force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the radial force
        HISTORY:
           2016-06-09 - Written - Bovy (UofT)
        """
        if not self.isNonAxi:
            phi= 0.
        x,y,z= bovy_coords.cyl_to_rect(R,phi,z)
        xy= numpy.dot(self.rot(t),numpy.array([x,y]))
        x,y= xy[0],xy[1]  
        # Compute all rectangular forces
        new_hash= hashlib.md5(numpy.array([x,y,z])).hexdigest()
        if new_hash == self._force_hash:
            Fx= self._cached_Fx
            Fy= self._cached_Fy
            Fz= self._cached_Fz
        else:
            Fx= self._xforce_xyz(x,y,z)
            Fy= self._yforce_xyz(x,y,z)
            Fz= self._zforce_xyz(x,y,z)
            self._force_hash= new_hash
            self._cached_Fx= Fx
            self._cached_Fy= Fy
            self._cached_Fz= Fz
        Fxy= numpy.dot(self.rot(t, transposed = True),numpy.array([Fx,Fy]))
        Fx, Fy= Fxy[0], Fxy[1]
        return numpy.cos(phi)*Fx+numpy.sin(phi)*Fy


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
           2016-06-09 - Written - Bovy (UofT)
        """
        if not self.isNonAxi:
            phi= 0.
        x,y,z= bovy_coords.cyl_to_rect(R,phi,z)
        xy= numpy.dot(self.rot(t),numpy.array([x,y]))
        x,y= xy[0],xy[1]  
        # Compute all rectangular forces
        new_hash= hashlib.md5(numpy.array([x,y,z])).hexdigest()
        if new_hash == self._force_hash:
            Fx= self._cached_Fx
            Fy= self._cached_Fy
            Fz= self._cached_Fz
        else:
            Fx= self._xforce_xyz(x,y,z)
            Fy= self._yforce_xyz(x,y,z)
            Fz= self._zforce_xyz(x,y,z)
            self._force_hash= new_hash
            self._cached_Fx= Fx
            self._cached_Fy= Fy
            self._cached_Fz= Fz
        Fxy= numpy.dot(self.rot(t, transposed = True),numpy.array([Fx,Fy]))
        Fx, Fy= Fxy[0], Fxy[1]
        return R*(-numpy.sin(phi)*Fx+numpy.cos(phi)*Fy)

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
           2016-06-09 - Written - Bovy (UofT)
        """
        if not self.isNonAxi:
            phi= 0.
        x,y,z= bovy_coords.cyl_to_rect(R,phi,z)
        xy= numpy.dot(self.rot(t),numpy.array([x,y]))
        x,y= xy[0],xy[1] 
        # Compute all rectangular forces
        new_hash= hashlib.md5(numpy.array([x,y,z])).hexdigest()
        if new_hash == self._force_hash:
            Fz= self._cached_Fz
        else:
            Fz= self._zforce_xyz(x,y,z)
            self._force_hash= new_hash
            self._cached_Fz= Fz
        return Fz

    def _xforce_xyz(self,x,y,z):
        """Evaluation of the x force as a function of (x,y,z) in the aligned
        coordinate frame"""
        return 1/2*self._b*self._c*_forceInt(x,y,z,self._a2,self._b2,self._c2,0,self.n)         
            
    def _yforce_xyz(self,x,y,z):
        """Evaluation of the y force as a function of (x,y,z) in the aligned
        coordinate frame"""
        return 1/2*self._b*self._c*_forceInt(x,y,z,self._a2,self._b2,self._c2,1,self.n)  

    def _zforce_xyz(self,x,y,z):
        """Evaluation of the z force as a function of (x,y,z) in the aligned
        coordinate frame"""
        return 1/2*self._b*self._c*_forceInt(x,y,z,self._a2,self._b2,self._c2,2,self.n)  

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
           2016-06-15 - Written - Bovy (UofT)
        """
        if not self.isNonAxi:
            phi= 0.
        x,y,z= bovy_coords.cyl_to_rect(R,phi-self._pa-self._omegab*t,z)
      #  if not self._aligned:
      #      raise NotImplementedError("2nd potential derivatives of TwoPowerTriaxialPotential not implemented for rotated coordinated frames (non-trivial zvec and pa)")
        phixx= self._2ndderiv_xyz(x,y,z,0,0)
        phixy= self._2ndderiv_xyz(x,y,z,0,1)
        phiyy= self._2ndderiv_xyz(x,y,z,1,1)
        return numpy.cos(phi)**2.*phixx+numpy.sin(phi)**2.*phiyy\
            +2.*numpy.cos(phi)*numpy.sin(phi)*phixy

    def _Rzderiv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _Rzderiv
        PURPOSE:
           evaluate the mixed radial, vertical derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the mixed radial, vertical derivative
        HISTORY:
           2016-06-15 - Written - Bovy (UofT)
        """
        if not self.isNonAxi:
            phi= 0.
        x,y,z= bovy_coords.cyl_to_rect(R,phi-self._pa-self._omegab*t,z)
      #  if not self._aligned:
      #      raise NotImplementedError("2nd potential derivatives of TwoPowerTriaxialPotential not implemented for rotated coordinated frames (non-trivial zvec and pa)")
        phixz= self._2ndderiv_xyz(x,y,z,0,2)
        phiyz= self._2ndderiv_xyz(x,y,z,1,2)
        return numpy.cos(phi)*phixz+numpy.sin(phi)*phiyz

    def _z2deriv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _z2deriv
        PURPOSE:
           evaluate the second vertical derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the second vertical derivative
        HISTORY:
           2016-06-15 - Written - Bovy (UofT)
        """
        if not self.isNonAxi:
            phi= 0.
        x,y,z= bovy_coords.cyl_to_rect(R,phi-self._pa-self._omegab*t,z)
      #  if not self._aligned:
      #      raise NotImplementedError("2nd potential derivatives of TwoPowerTriaxialPotential not implemented for rotated coordinated frames (non-trivial zvec and pa)")
        return self._2ndderiv_xyz(x,y,z,2,2)

    def _phi2deriv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _phi2deriv
        PURPOSE:
           evaluate the second azimuthal derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the second azimuthal derivative
        HISTORY:
           2016-06-15 - Written - Bovy (UofT)
        """
        if not self.isNonAxi:
            phi= 0.
        x,y,z= bovy_coords.cyl_to_rect(R,phi-self._omegab*t,z)
      #  if not self._aligned:
      #      raise NotImplementedError("2nd potential derivatives of TwoPowerTriaxialPotential not implemented for rotated coordinated frames (non-trivial zvec and pa)")
        Fx= self._xforce_xyz(x,y,z)
        Fy= self._yforce_xyz(x,y,z)
        phixx= self._2ndderiv_xyz(x,y,z,0,0)
        phixy= self._2ndderiv_xyz(x,y,z,0,1)
        phiyy= self._2ndderiv_xyz(x,y,z,1,1)
        return R**2.*(numpy.sin(phi)**2.*phixx+numpy.cos(phi)**2.*phiyy\
                          -2.*numpy.cos(phi)*numpy.sin(phi)*phixy)\
                          +R*(numpy.cos(phi)*Fx+numpy.sin(phi)*Fy)

    def _Rphideriv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _Rphideriv
        PURPOSE:
           evaluate the mixed radial, azimuthal derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the mixed radial, azimuthal derivative
        HISTORY:
           2016-06-15 - Written - Bovy (UofT)
        """
        if not self.isNonAxi:
            phi= 0.
        x,y,z= bovy_coords.cyl_to_rect(R,phi-self._omegab*t,z)
      #  if not self._aligned:
      #      raise NotImplementedError("2nd potential derivatives of TwoPowerTriaxialPotential not implemented for rotated coordinated frames (non-trivial zvec and pa)")
        Fx= self._xforce_xyz(x,y,z)
        Fy= self._yforce_xyz(x,y,z)
        phixx= self._2ndderiv_xyz(x,y,z,0,0)
        phixy= self._2ndderiv_xyz(x,y,z,0,1)
        phiyy= self._2ndderiv_xyz(x,y,z,1,1)
        return R*numpy.cos(phi)*numpy.sin(phi)*\
            (phiyy-phixx)+R*numpy.cos(2.*(phi))*phixy\
            +numpy.sin(phi)*Fx-numpy.cos(phi)*Fy

    def _2ndderiv_xyz(self,x,y,z,i,j):
        """General 2nd derivative of the potential as a function of (x,y,z)
        in the aligned coordinate frame"""
                         
        return -1/4*self._b*self._c*_2ndDerivInt(x,y,z,self._a2,self._b2,self._c2,i,j,self.n)
        
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
        xy= numpy.dot(self.rot(t),numpy.array([x,y]))
        x,y= xy[0],xy[1] 
        m2 = x**2/self._a2+y**2/self._b2+z**2/self._c2
        if m2 < 1:
            return 1/(4*numpy.pi*self.a**3)*(1-m2/self.a**2)**self.n
        else:
            return 0

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
        return self._omegab
    
    def rot(self, t=0., transposed = False):
        """Rotation matrix for non-zero pa or pattern speed"""
        rotmat = numpy.array([[numpy.cos(self._pa+self._omegab*t),numpy.sin(self._pa+self._omegab*t)],
                        [-numpy.sin(self._pa+self._omegab*t),numpy.cos(self._pa+self._omegab*t)]])
        if transposed:
            return rotmat.T
        else:
            return rotmat

def _potInt(x,y,z,a2,b2,c2,n):
    """Integral that gives the potential in x,y,z"""
    def integrand(tau):
        return _FracInt(x,y,z,a2,b2,c2,tau,n + 1)
    return integrate.quad(integrand,lowerlim(x,y,z,a2,b2,c2),numpy.inf)[0]                              

def _forceInt(x,y,z,a2,b2,c2,i,n):
    """Integral that gives the force in x,y,z"""
    def integrand(tau):
        return -(x*(i==0) + y*(i==1) + z*(i==2))/(a2*(i==0) + a2*b2*(i==1) + a2*c2*(i==2) + tau)*_FracInt(x,y,z,a2,b2,c2,tau,n)
    return integrate.quad(integrand,lowerlim(x,y,z,a2,b2,c2),numpy.inf)[0]                              

def _2ndDerivInt(x,y,z,a2,b2,c2,i,j,n):
    """Integral that gives the 2nd derivative of the potential in x,y,z"""
    def integrand(tau):
        if i!=j:
            return _FracInt(x,y,z,a2,b2,c2,tau,n-1)*n*(1+(-1-2*x/(tau+a2))*(i==0 or j==0))*(1+(-1-2*y/(tau+a2*b2))*(i==1 or j==1))*(1+(-1-2*z/(tau+a2*c2))*(i==2 or j==2))
        else:
            var2 = x**2*(i==0) + y**2*(i==1) + z**2*(i==2)
            coef2 = a2*(i==0) + a2*b2*(i==1) + a2*c2*(i==2)
            return _FracInt(x,y,z,a2,b2,c2,tau,n-1)*n*(4*var2)/(tau+coef2)**2 + _FracInt(x,y,z,a2,b2,c2,tau,n)*(-2/(tau+coef2))
    return integrate.quad(integrand,lowerlim(x,y,z,a2,b2,c2),numpy.inf)[0]

def _FracInt(x,y,z,a2,b2,c2,tau,expon):
    """Integrand present in most class functions"""
    return (1 - x**2/(a2 + tau) - y**2/(a2*b2 + tau) - z**2/(a2*c2 + tau))**expon/numpy.sqrt((a2 + tau)*(a2*b2 + tau)*(a2*c2 + tau))

def lowerlim(x,y,z,a2,b2,c2):
    """Lower limit of the integrals"""
    def func(tau):
        return x**2/(a2+tau) + y**2/(a2*b2+tau) + z**2/(a2*c2+tau) - 1

    if numpy.sqrt(x**2/a2 + y**2/(a2*b2) + z**2/(a2*c2)) >= 1:
        return fsolve(func,0)[0]
    else:
        return 0