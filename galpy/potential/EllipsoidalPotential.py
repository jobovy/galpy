###############################################################################
#   EllipsoidalPotential.py: base class for potentials corresponding to 
#                            density profiles that are stratified on
#                            ellipsoids: 
#
#                            \rho(x,y,z) ~ \rho(m)
#
#                            with m^2 = x^2+y^2/b^2+z^2/c^2
#
###############################################################################
import numpy
import hashlib
from scipy import integrate
from galpy.util import bovy_coords
from galpy.util import _rotate_to_arbitrary_vector
from .Potential import Potential, _APY_LOADED
if _APY_LOADED:
    from astropy import units
class EllipsoidalPotential(Potential):
    """Base class for potentials corresponding to density profiles that are stratified on ellipsoids: 

    .. math::

        \\rho(x,y,z) \\equiv \\rho(m)

    where :math:`m^2 = x^2+y^2/b^2+z^2/c^2`. Note that :math:`b` and :math:`c` are defined to be the axis ratios (rather than using :math:`m^2 = x^2/a^2+y^2/b^2+z^2/c^2` as is common).

    Implement a specific density distribution with this form by inheriting from this class and defining the functions: _psi(self,m) = -\int_m^\infty d m^2 \rho(m^2), _mdens = \rho(m), and _mdens_deriv= d \rho(m) / d m. See PerfectEllipsoidPotential for an example.
    """
    def __init__(self,amp=1.,
                 b=1.,c=1.,
                 zvec=None,pa=None,glorder=50,
                 ro=None,vo=None,amp_units=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a ellipsoidal potential

        INPUT:

           amp - amplitude to be applied to the potential (default: 1); can be a Quantity with units that depend on the specific spheroidal potential

           b - y-to-x axis ratio of the density

           c - z-to-x axis ratio of the density

           zvec= (None) If set, a unit vector that corresponds to the z axis

           pa= (None) If set, the position angle of the x axis (rad or Quantity)

           glorder= (50) if set, compute the relevant force and potential integrals with Gaussian quadrature of this order

           amp_units - ('mass', 'velocity2', 'density') type of units that amp should have if it has units (passed to Potential.__init__

           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

        OUTPUT:

           (none)

        HISTORY:

           2018-08-06 - Started - Bovy (UofT)

        """
        Potential.__init__(self,amp=amp,ro=ro,vo=vo,amp_units=amp_units)
        # Setup axis ratios
        self._b= b
        self._c= c
        self._b2= self._b**2.
        self._c2= self._c**2.
        self._force_hash= None
        # Setup rotation
        self._setup_zvec_pa(zvec,pa)
        # Setup integration
        self._setup_gl(glorder)
        if not self._aligned or numpy.fabs(self._b-1.) > 10.**-10.:
            self.isNonAxi= True
        return None

    def _setup_zvec_pa(self,zvec,pa):
        if not pa is None:
            if _APY_LOADED and isinstance(pa,units.Quantity):
                pa= pa.to(units.rad).value
        if zvec is None and (pa is None or numpy.fabs(pa) < 10.**-10.):
            self._aligned= True
        else:
            self._aligned= False
            if not pa is None:
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

    def _setup_gl(self,glorder):
        self._glorder= glorder
        if self._glorder is None:
            self._glx, self._glw= None, None
        else:
            self._glx, self._glw=\
                numpy.polynomial.legendre.leggauss(self._glorder)
            # Interval change
            self._glx= 0.5*self._glx+0.5
            self._glw*= 0.5
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
        if self._aligned:
            return self._evaluate_xyz(x,y,z)
        else:
            xyzp= numpy.dot(self._rot,numpy.array([x,y,z]))
            return self._evaluate_xyz(xyzp[0],xyzp[1],xyzp[2])

    def _evaluate_xyz(self,x,y,z):
        """Evaluation of the potential as a function of (x,y,z) in the 
        aligned coordinate frame"""
        return 2.*numpy.pi*self._b*self._c\
            *_potInt(x,y,z,self._psi,
                     self._b2,self._c2,glx=self._glx,glw=self._glw)

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
           2016-06-09 - Written - Bovy (UofT)
        """
        if not self.isNonAxi:
            phi= 0.
        x,y,z= bovy_coords.cyl_to_rect(R,phi,z)
        # Compute all rectangular forces
        new_hash= hashlib.md5(numpy.array([x,y,z])).hexdigest()
        if new_hash == self._force_hash:
            Fx= self._cached_Fx
            Fy= self._cached_Fy
            Fz= self._cached_Fz
        else:
            if self._aligned:
                xp, yp, zp= x, y, z
            else:
                xyzp= numpy.dot(self._rot,numpy.array([x,y,z]))
                xp, yp, zp= xyzp[0], xyzp[1], xyzp[2]
            Fx= self._force_xyz(xp,yp,zp,0)
            Fy= self._force_xyz(xp,yp,zp,1)
            Fz= self._force_xyz(xp,yp,zp,2)
            self._force_hash= new_hash
            self._cached_Fx= Fx
            self._cached_Fy= Fy
            self._cached_Fz= Fz
        if not self._aligned:
            Fxyz= numpy.dot(self._rot.T,numpy.array([Fx,Fy,Fz]))
            Fx, Fy= Fxyz[0], Fxyz[1]
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
        # Compute all rectangular forces
        new_hash= hashlib.md5(numpy.array([x,y,z])).hexdigest()
        if new_hash == self._force_hash:
            Fx= self._cached_Fx
            Fy= self._cached_Fy
            Fz= self._cached_Fz
        else:
            if self._aligned:
                xp, yp, zp= x, y, z
            else:
                xyzp= numpy.dot(self._rot,numpy.array([x,y,z]))
                xp, yp, zp= xyzp[0], xyzp[1], xyzp[2]
            Fx= self._force_xyz(xp,yp,zp,0)
            Fy= self._force_xyz(xp,yp,zp,1)
            Fz= self._force_xyz(xp,yp,zp,2)
            self._force_hash= new_hash
            self._cached_Fx= Fx
            self._cached_Fy= Fy
            self._cached_Fz= Fz
        if not self._aligned:
            Fxyz= numpy.dot(self._rot.T,numpy.array([Fx,Fy,Fz]))
            Fx, Fy= Fxyz[0], Fxyz[1]
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
        # Compute all rectangular forces
        new_hash= hashlib.md5(numpy.array([x,y,z])).hexdigest()
        if new_hash == self._force_hash:
            Fx= self._cached_Fx
            Fy= self._cached_Fy
            Fz= self._cached_Fz
        else:
            if self._aligned:
                xp, yp, zp= x, y, z
            else:
                xyzp= numpy.dot(self._rot,numpy.array([x,y,z]))
                xp, yp, zp= xyzp[0], xyzp[1], xyzp[2]
            Fx= self._force_xyz(xp,yp,zp,0)
            Fy= self._force_xyz(xp,yp,zp,1)
            Fz= self._force_xyz(xp,yp,zp,2)
            self._force_hash= new_hash
            self._cached_Fx= Fx
            self._cached_Fy= Fy
            self._cached_Fz= Fz
        if not self._aligned:
            Fxyz= numpy.dot(self._rot.T,numpy.array([Fx,Fy,Fz]))
            Fz= Fxyz[2]
        return Fz

    def _force_xyz(self,x,y,z,i):
        """Evaluation of the i-th force component as a function of (x,y,z)"""
        return -4.*numpy.pi*self._b*self._c\
            *_forceInt(x,y,z,
                       lambda m: self._mdens(m),
                       self._b2,self._c2,i,glx=self._glx,glw=self._glw)
        
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
        x,y,z= bovy_coords.cyl_to_rect(R,phi,z)
        if not self._aligned:
            raise NotImplementedError("2nd potential derivatives of TwoPowerTriaxialPotential not implemented for rotated coordinated frames (non-trivial zvec and pa)")
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
        x,y,z= bovy_coords.cyl_to_rect(R,phi,z)
        if not self._aligned:
            raise NotImplementedError("2nd potential derivatives of TwoPowerTriaxialPotential not implemented for rotated coordinated frames (non-trivial zvec and pa)")
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
        x,y,z= bovy_coords.cyl_to_rect(R,phi,z)
        if not self._aligned:
            raise NotImplementedError("2nd potential derivatives of TwoPowerTriaxialPotential not implemented for rotated coordinated frames (non-trivial zvec and pa)")
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
        x,y,z= bovy_coords.cyl_to_rect(R,phi,z)
        if not self._aligned:
            raise NotImplementedError("2nd potential derivatives of TwoPowerTriaxialPotential not implemented for rotated coordinated frames (non-trivial zvec and pa)")
        Fx= self._force_xyz(x,y,z,0)
        Fy= self._force_xyz(x,y,z,1)
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
        x,y,z= bovy_coords.cyl_to_rect(R,phi,z)
        if not self._aligned:
            raise NotImplementedError("2nd potential derivatives of TwoPowerTriaxialPotential not implemented for rotated coordinated frames (non-trivial zvec and pa)")
        Fx= self._force_xyz(x,y,z,0)
        Fy= self._force_xyz(x,y,z,1)
        phixx= self._2ndderiv_xyz(x,y,z,0,0)
        phixy= self._2ndderiv_xyz(x,y,z,0,1)
        phiyy= self._2ndderiv_xyz(x,y,z,1,1)
        return R*numpy.cos(phi)*numpy.sin(phi)*\
            (phiyy-phixx)+R*numpy.cos(2.*phi)*phixy\
            +numpy.sin(phi)*Fx-numpy.cos(phi)*Fy

    def _2ndderiv_xyz(self,x,y,z,i,j):
        """General 2nd derivative of the potential as a function of (x,y,z)
        in the aligned coordinate frame"""
        return 4.*numpy.pi*self._b*self._c\
            *_2ndDerivInt(x,y,z,
                          lambda m: self._mdens(m),
                          lambda m: self._mdens_deriv(m),
                          self._b2,self._c2,i,j,glx=self._glx,glw=self._glw)
                 
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
           2018-08-06 - Written - Bovy (UofT)
        """
        x,y,z= bovy_coords.cyl_to_rect(R,phi,z)
        if self._aligned:
            xp, yp, zp= x, y, z
        else:
            xyzp= numpy.dot(self._rot,numpy.array([x,y,z]))
            xp, yp, zp= xyzp[0], xyzp[1], xyzp[2]
        m= numpy.sqrt(xp**2.+yp**2./self._b2+zp**2./self._c2)
        return self._mdens(m)
        
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

def _potInt(x,y,z,psi,b2,c2,glx=None,glw=None):
    """int_0^\infty [psi(m)-psi(\infy)]/sqrt([1+tau]x[b^2+tau]x[c^2+tau])dtau"""
    def integrand(s):
        t= 1/s**2.-1.
        return psi(numpy.sqrt(x**2./(1.+t)+y**2./(b2+t)+z**2./(c2+t)))\
            /numpy.sqrt((1.+(b2-1.)*s**2.)*(1.+(c2-1.)*s**2.))
    if glx is None:
        return integrate.quad(integrand,0.,1.)[0]                              
    else:
        return numpy.sum(glw*integrand(glx))

def _forceInt(x,y,z,dens,b2,c2,i,glx=None,glw=None):
    """Integral that gives the force in x,y,z"""
    def integrand(s):
        t= 1/s**2.-1.
        return dens(numpy.sqrt(x**2./(1.+t)+y**2./(b2+t)+z**2./(c2+t)))\
            *(x/(1.+t)*(i==0)+y/(b2+t)*(i==1)+z/(c2+t)*(i==2))\
            /numpy.sqrt((1.+(b2-1.)*s**2.)*(1.+(c2-1.)*s**2.))
    if glx is None:
        return integrate.quad(integrand,0.,1.)[0]                              
    else:
        return numpy.sum(glw*integrand(glx))

def _2ndDerivInt(x,y,z,dens,densDeriv,b2,c2,i,j,glx=None,glw=None):
    """Integral that gives the 2nd derivative of the potential in x,y,z"""
    def integrand(s):
        t= 1/s**2.-1.
        m= numpy.sqrt(x**2./(1.+t)+y**2./(b2+t)+z**2./(c2+t))
        return (densDeriv(m)
                *(x/(1.+t)*(i==0)+y/(b2+t)*(i==1)+z/(c2+t)*(i==2))
                *(x/(1.+t)*(j==0)+y/(b2+t)*(j==1)+z/(c2+t)*(j==2))/m\
                    +dens(m)*(i==j)*((1./(1.+t)*(i==0)+1./(b2+t)*(i==1)+1./(c2+t)*(i==2))))\
                    /numpy.sqrt((1.+(b2-1.)*s**2.)*(1.+(c2-1.)*s**2.))
    if glx is None:
        return integrate.quad(integrand,0.,1.)[0]
    else:
        return numpy.sum(glw*integrand(glx))

