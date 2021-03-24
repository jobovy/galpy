###############################################################################
#   Potential.py: top-level class for a full potential
#
#   Evaluate by calling the instance: Pot(R,z,phi)
#
#   API for Potentials:
#      function _evaluate(self,R,z,phi) returns Phi(R,z,phi)
#    for orbit integration you need
#      function _Rforce(self,R,z,phi) return -d Phi d R
#      function _zforce(self,R,z,phi) return - d Phi d Z
#    density
#      function _dens(self,R,z,phi) return BOVY??
#    for epicycle frequency
#      function _R2deriv(self,R,z,phi) return d2 Phi dR2
###############################################################################
from __future__  import division, print_function

import os, os.path
import pickle
from functools import wraps
import warnings
import numpy
from scipy import optimize, integrate
from ..util import plot, coords, conversion
from ..util.conversion import velocity_in_kpcGyr, \
    physical_conversion, potential_physical_input, freq_in_Gyr, \
    get_physical
from ..util import galpyWarning
from .plotRotcurve import plotRotcurve, vcirc
from .plotEscapecurve import _INF, plotEscapecurve
from .DissipativeForce import DissipativeForce, _isDissipative
from .Force import Force, _APY_LOADED
if _APY_LOADED:
    from astropy import units
def check_potential_inputs_not_arrays(func):
    """
    NAME:
       check_potential_inputs_not_arrays
    PURPOSE:
       Decorator to check inputs and throw TypeError if any of the inputs are arrays for Potentials that do not support array evaluation
    HISTORY:
       2017-summer - Written for SpiralArmsPotential - Jack Hong (UBC)
       2019-05-23 - Moved to Potential for more general use - Bovy (UofT)
       
    """
    @wraps(func)
    def func_wrapper(self,R,z,phi,t):
        if (hasattr(R,'shape') and R.shape != () and len(R) > 1) \
                or (hasattr(z,'shape') and z.shape != () and len(z) > 1) \
                or (hasattr(phi,'shape') and phi.shape != () and len(phi) > 1) \
                or (hasattr(t,'shape') and t.shape != () and len(t) > 1):
            raise TypeError('Methods in {} do not accept array inputs. Please input scalars'.format(self.__class__.__name__))
        return func(self,R,z,phi,t)
    return func_wrapper


class Potential(Force):
    """Top-level class for a potential"""
    def __init__(self,amp=1.,ro=None,vo=None,amp_units=None):
        """
        NAME:
           __init__
        PURPOSE:
        INPUT:
           amp - amplitude to be applied when evaluating the potential and its forces
           amp_units - ('mass', 'velocity2', 'density') type of units that amp should have if it has units
        OUTPUT:
        HISTORY:
        """
        Force.__init__(self,amp=amp,ro=ro,vo=vo,amp_units=amp_units)
        self.dim= 3
        self.isRZ= True
        self.isNonAxi= False
        self.hasC= False
        self.hasC_dxdv= False
        self.hasC_dens= False
        return None

    @potential_physical_input
    @physical_conversion('energy',pop=True)
    def __call__(self,R,z,phi=0.,t=0.,dR=0,dphi=0):
        """
        NAME:

           __call__

        PURPOSE:

           evaluate the potential at (R,z,phi,t)

        INPUT:

           R - Cylindrical Galactocentric radius (can be Quantity)

           z - vertical height (can be Quantity)

           phi - azimuth (optional; can be Quantity)

           t - time (optional; can be Quantity)

        OUTPUT:

           Phi(R,z,t)

        HISTORY:

           2010-04-16 - Written - Bovy (NYU)

        """
        return self._call_nodecorator(R,z,phi=phi,t=t,dR=dR,dphi=dphi)

    def _call_nodecorator(self,R,z,phi=0.,t=0.,dR=0.,dphi=0):
        if dR == 0 and dphi == 0:
            try:
                rawOut= self._evaluate(R,z,phi=phi,t=t)
            except AttributeError: #pragma: no cover
                raise PotentialError("'_evaluate' function not implemented for this potential")
            if rawOut is None: return rawOut
            else: return self._amp*rawOut
        elif dR == 1 and dphi == 0:
            return -self.Rforce(R,z,phi=phi,t=t,use_physical=False)
        elif dR == 0 and dphi == 1:
            return -self.phiforce(R,z,phi=phi,t=t,use_physical=False)
        elif dR == 2 and dphi == 0:
            return self.R2deriv(R,z,phi=phi,t=t,use_physical=False)
        elif dR == 0 and dphi == 2:
            return self.phi2deriv(R,z,phi=phi,t=t,use_physical=False)
        elif dR == 1 and dphi == 1:
            return self.Rphideriv(R,z,phi=phi,t=t,use_physical=False)
        elif dR != 0 or dphi != 0:
            raise NotImplementedError('Higher-order derivatives not implemented for this potential')
        
    @potential_physical_input
    @physical_conversion('force',pop=True)
    def Rforce(self,R,z,phi=0.,t=0.):
        """
        NAME:

           Rforce

        PURPOSE:

           evaluate cylindrical radial force F_R  (R,z)

        INPUT:

           R - Cylindrical Galactocentric radius (can be Quantity)

           z - vertical height (can be Quantity)

           phi - azimuth (optional; can be Quantity)

           t - time (optional; can be Quantity)

        OUTPUT:

           F_R (R,z,phi,t)

        HISTORY:

           2010-04-16 - Written - Bovy (NYU)

        """
        return self._Rforce_nodecorator(R,z,phi=phi,t=t)

    def _Rforce_nodecorator(self,R,z,phi=0.,t=0.):
        # Separate, so it can be used during orbit integration
        try:
            return self._amp*self._Rforce(R,z,phi=phi,t=t)
        except AttributeError: #pragma: no cover
            raise PotentialError("'_Rforce' function not implemented for this potential")
        
    @potential_physical_input
    @physical_conversion('force',pop=True)
    def zforce(self,R,z,phi=0.,t=0.):
        """
        NAME:

           zforce

        PURPOSE:

           evaluate the vertical force F_z  (R,z,t)

        INPUT:

           R - Cylindrical Galactocentric radius (can be Quantity)

           z - vertical height (can be Quantity)

           phi - azimuth (optional; can be Quantity)

           t - time (optional; can be Quantity)

        OUTPUT:

           F_z (R,z,phi,t)

        HISTORY:

           2010-04-16 - Written - Bovy (NYU)

        """
        return self._zforce_nodecorator(R,z,phi=phi,t=t)

    def _zforce_nodecorator(self,R,z,phi=0.,t=0.):
        # Separate, so it can be used during orbit integration
        try:
            return self._amp*self._zforce(R,z,phi=phi,t=t)
        except AttributeError: #pragma: no cover
            raise PotentialError("'_zforce' function not implemented for this potential")

    @potential_physical_input
    @physical_conversion('forcederivative',pop=True)
    def r2deriv(self,R,z,phi=0.,t=0.):
        """
        NAME:

           r2deriv

        PURPOSE:

           evaluate the second spherical radial derivative

        INPUT:

           R - Cylindrical Galactocentric radius (can be Quantity)

           z - vertical height (can be Quantity)

           phi - azimuth (optional; can be Quantity)

           t - time (optional; can be Quantity)

        OUTPUT:

           d2phi/dr2

        HISTORY:

           2018-03-21 - Written - Webb (UofT)

        """
       
        r= numpy.sqrt(R**2.+z**2.)       
        return (self.R2deriv(R,z,phi=phi,t=t,use_physical=False)*R/r\
            +self.Rzderiv(R,z,phi=phi,t=t,use_physical=False)*z/r)*R/r\
            +(self.Rzderiv(R,z,phi=phi,t=t,use_physical=False)*R/r\
            +self.z2deriv(R,z,phi=phi,t=t,use_physical=False)*z/r)*z/r

    @potential_physical_input
    @physical_conversion('density',pop=True)
    def dens(self,R,z,phi=0.,t=0.,forcepoisson=False):
        """
        NAME:

           dens

        PURPOSE:

           evaluate the density rho(R,z,t)

        INPUT:

           R - Cylindrical Galactocentric radius (can be Quantity)

           z - vertical height (can be Quantity)

           phi - azimuth (optional; can be Quantity)

           t - time (optional; can be Quantity)

        KEYWORDS:

           forcepoisson= if True, calculate the density through the Poisson equation, even if an explicit expression for the density exists

        OUTPUT:

           rho (R,z,phi,t)

        HISTORY:

           2010-08-08 - Written - Bovy (NYU)

        """
        try:
            if forcepoisson: raise AttributeError #Hack!
            return self._amp*self._dens(R,z,phi=phi,t=t)
        except AttributeError:
            #Use the Poisson equation to get the density
            return (-self.Rforce(R,z,phi=phi,t=t,use_physical=False)/R
                     +self.R2deriv(R,z,phi=phi,t=t,use_physical=False)
                     +self.phi2deriv(R,z,phi=phi,t=t,use_physical=False)/R**2.
                     +self.z2deriv(R,z,phi=phi,t=t,use_physical=False))/4./numpy.pi

    @potential_physical_input
    @physical_conversion('surfacedensity',pop=True)
    def surfdens(self,R,z,phi=0.,t=0.,forcepoisson=False):
        """
        NAME:

           surfdens

        PURPOSE:

           evaluate the surface density :math:`\\Sigma(R,z,\\phi,t) = \\int_{-z}^{+z} dz' \\rho(R,z',\\phi,t)`

        INPUT:

           R - Cylindrical Galactocentric radius (can be Quantity)

           z - vertical height (can be Quantity)

           phi - azimuth (optional; can be Quantity)

           t - time (optional; can be Quantity)

        KEYWORDS:

           forcepoisson= if True, calculate the surface density through the Poisson equation, even if an explicit expression for the surface density exists

        OUTPUT:

           Sigma (R,z,phi,t)

        HISTORY:

           2018-08-19 - Written - Bovy (UofT)

        """
        try:
            if forcepoisson: raise AttributeError #Hack!
            return self._amp*self._surfdens(R,z,phi=phi,t=t)
        except AttributeError:
            #Use the Poisson equation to get the surface density
            return (-self.zforce(R,numpy.fabs(z),phi=phi,t=t,use_physical=False)
                    +integrate.quad(\
                lambda x: -self.Rforce(R,x,phi=phi,t=t,use_physical=False)/R
                +self.R2deriv(R,x,phi=phi,t=t,use_physical=False)
                +self.phi2deriv(R,x,phi=phi,t=t,use_physical=False)/R**2.,
                0.,numpy.fabs(z))[0])/2./numpy.pi

    def _surfdens(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _surfdens
        PURPOSE:
           evaluate the surface density for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the surface density
        HISTORY:
           2018-08-19 - Written - Bovy (UofT)
        """
        return 2.*integrate.quad(lambda x: self._dens(R,x,phi=phi,t=t),0,z)[0]

    @potential_physical_input
    @physical_conversion('mass',pop=True)
    def mass(self,R,z=None,t=0.,forceint=False):
        """
        NAME:

           mass

        PURPOSE:

           evaluate the mass enclosed

        INPUT:

           R - Cylindrical Galactocentric radius (can be Quantity)

           z= (None) vertical height up to which to integrate (can be Quantity)

           t - time (optional; can be Quantity)

           forceint= if True, calculate the mass through integration of the density, even if an explicit expression for the mass exists

        OUTPUT:

           Mass enclosed within the spherical shell with radius R if z is None else mass in the slab <R and between -z and z; except: potentials inheriting from EllipsoidalPotential, which if z is None return the mass within the ellipsoidal shell with semi-major axis R

        HISTORY:

           2014-01-29 - Written - Bovy (IAS)

           2019-08-15 - Added spherical warning - Bovy (UofT)

           2021-03-15 - Changed to integrate to spherical shell for z is None slab otherwise - Bovy (UofT)

           2021-03-18 - Switched to using Gauss' theorem - Bovy (UofT)

        """
        from .EllipsoidalPotential import EllipsoidalPotential
        if self.isNonAxi and not isinstance(self,EllipsoidalPotential):
            raise NotImplementedError('mass for non-axisymmetric potentials that are not EllipsoidalPotentials is not currently supported')
        try:
            if forceint: raise AttributeError #Hack!
            return self._amp*self._mass(R,z=z,t=t)
        except AttributeError:
            #Use numerical integration to get the mass, using Gauss' theorem
            if z is None: # Within spherical shell
                def _integrand(theta):
                    tz= R*numpy.cos(theta)
                    tR= R*numpy.sin(theta)
                    return self.rforce(tR,tz,t=t,use_physical=False)\
                        *numpy.sin(theta)
                return -R**2.*integrate.quad(_integrand,0.,numpy.pi)[0]/2.
            else: # Within disk at <R, -z --> z
                return -R*integrate.quad(lambda x: self.Rforce(R,x,t=t,
                                                        use_physical=False),
                                         -z,z)[0]/2.\
                        -integrate.quad(lambda x: x*self.zforce(x,z,t=t,
                                                        use_physical=False),
                                        0.,R)[0]

    @physical_conversion('position',pop=True)
    def rhalf(self,t=0.,INF=numpy.inf):
        """
            
        NAME:
            
            rhalf
            
        PURPOSE:

            calculate the half-mass radius, the radius of the spherical shell that contains half the total mass

        INPUT:

            t= (0.) time (optional; can be Quantity)

            INF= (numpy.inf) radius at which the total mass is calculated (internal units, just set this to something very large)

        OUTPUT:

            half-mass radius

        HISTORY:

            2021-03-18 - Written - Bovy (UofT)

        """
        return rhalf(self,t=t,INF=INF,use_physical=False)

    @potential_physical_input
    @physical_conversion('time',pop=True)
    def tdyn(self,R,t=0.):
        """
        NAME:
        
           tdyn

        PURPOSE:

           calculate the dynamical time from tdyn^2 = 3pi/[G<rho>]

        INPUT:

           R - Galactocentric radius (can be Quantity)

           t= (0.) time (optional; can be Quantity)
        
        OUTPUT:

           Dynamical time

        HISTORY:

           2021-03-18 - Written - Bovy (UofT)

        """
        return 2.*numpy.pi*R*numpy.sqrt(R/self.mass(R,use_physical=False))

    @physical_conversion('mass',pop=False)
    def mvir(self,H=70.,Om=0.3,t=0.,overdens=200.,wrtcrit=False,
             forceint=False,ro=None,vo=None,
             use_physical=False): # use_physical necessary bc of pop=False, does nothing inside
        """
        NAME:

           mvir

        PURPOSE:

           calculate the virial mass

        INPUT:

           H= (default: 70) Hubble constant in km/s/Mpc
           
           Om= (default: 0.3) Omega matter
       
           overdens= (200) overdensity which defines the virial radius

           wrtcrit= (False) if True, the overdensity is wrt the critical density rather than the mean matter density
           
           ro= distance scale in kpc or as Quantity (default: object-wide, which if not set is 8 kpc))

           vo= velocity scale in km/s or as Quantity (default: object-wide, which if not set is 220 km/s))

        KEYWORDS:

           forceint= if True, calculate the mass through integration of the density, even if an explicit expression for the mass exists

        OUTPUT:

           M(<rvir)

        HISTORY:

           2014-09-12 - Written - Bovy (IAS)

        """
        if ro is None: ro= self._ro
        if vo is None: vo= self._vo
        #Evaluate the virial radius
        try:
            rvir= self.rvir(H=H,Om=Om,t=t,overdens=overdens,wrtcrit=wrtcrit,
                            use_physical=False,ro=ro,vo=vo)
        except AttributeError:
            raise AttributeError("This potential does not have a '_scale' defined to base the concentration on or does not support calculating the virial radius")
        return self.mass(rvir,t=t,forceint=forceint,use_physical=False,ro=ro,vo=vo)

    @potential_physical_input
    @physical_conversion('forcederivative',pop=True)
    def R2deriv(self,R,Z,phi=0.,t=0.):
        """
        NAME:

           R2deriv

        PURPOSE:

           evaluate the second radial derivative

        INPUT:

           R - Galactocentric radius (can be Quantity)

           Z - vertical height (can be Quantity)

           phi - Galactocentric azimuth (can be Quantity)

           t - time (can be Quantity)

        OUTPUT:

           d2phi/dR2

        HISTORY:

           2011-10-09 - Written - Bovy (IAS)

        """
        try:
            return self._amp*self._R2deriv(R,Z,phi=phi,t=t)
        except AttributeError: #pragma: no cover
            raise PotentialError("'_R2deriv' function not implemented for this potential")      

    @potential_physical_input
    @physical_conversion('forcederivative',pop=True)
    def z2deriv(self,R,Z,phi=0.,t=0.):
        """
        NAME:

           z2deriv

        PURPOSE:

           evaluate the second vertical derivative

        INPUT:

           R - Galactocentric radius (can be Quantity)

           Z - vertical height (can be Quantity)

           phi - Galactocentric azimuth (can be Quantity)

           t - time (can be Quantity)

        OUTPUT:

           d2phi/dz2

        HISTORY:

           2012-07-25 - Written - Bovy (IAS@MPIA)

        """
        try:
            return self._amp*self._z2deriv(R,Z,phi=phi,t=t)
        except AttributeError: #pragma: no cover
            raise PotentialError("'_z2deriv' function not implemented for this potential")      

    @potential_physical_input
    @physical_conversion('forcederivative',pop=True)
    def Rzderiv(self,R,Z,phi=0.,t=0.):
        """
        NAME:

           Rzderiv

        PURPOSE:

           evaluate the mixed R,z derivative

        INPUT:

           R - Galactocentric radius (can be Quantity)

           Z - vertical height (can be Quantity)

           phi - Galactocentric azimuth (can be Quantity)

           t - time (can be Quantity)

        OUTPUT:

           d2phi/dz/dR

        HISTORY:

           2013-08-26 - Written - Bovy (IAS)

        """
        try:
            return self._amp*self._Rzderiv(R,Z,phi=phi,t=t)
        except AttributeError: #pragma: no cover
            raise PotentialError("'_Rzderiv' function not implemented for this potential")      

    def normalize(self,norm):
        """
        NAME:

           normalize

        PURPOSE:

           normalize a potential in such a way that vc(R=1,z=0)=1., or a 
           fraction of this

        INPUT:

           norm - normalize such that Rforce(R=1,z=0) is such that it is 'norm' of the force necessary to make vc(R=1,z=0)=1 (if True, norm=1)

        OUTPUT:
           
           (none)

        HISTORY:


           2010-07-10 - Written - Bovy (NYU)

        """
        self._amp*= norm/numpy.fabs(self.Rforce(1.,0.,use_physical=False))

    @potential_physical_input
    @physical_conversion('energy',pop=True)
    def phiforce(self,R,z,phi=0.,t=0.):
        """
        NAME:

           phiforce

        PURPOSE:

           evaluate the azimuthal force F_phi = -d Phi / d phi (R,z,phi,t) [note that this is a torque, not a force!)

        INPUT:

           R - Cylindrical Galactocentric radius (can be Quantity)

           z - vertical height (can be Quantity)

           phi - azimuth (rad; can be Quantity)

           t - time (optional; can be Quantity)

        OUTPUT:

           F_phi (R,z,phi,t)

        HISTORY:

           2010-07-10 - Written - Bovy (NYU)

        """
        return self._phiforce_nodecorator(R,z,phi=phi,t=t)

    def _phiforce_nodecorator(self,R,z,phi=0.,t=0.):
        # Separate, so it can be used during orbit integration
        try:
            return self._amp*self._phiforce(R,z,phi=phi,t=t)
        except AttributeError: #pragma: no cover
            if self.isNonAxi:
                raise PotentialError("'_phiforce' function not implemented for this non-axisymmetric potential")
            return 0.

    @potential_physical_input
    @physical_conversion('forcederivative',pop=True)
    def phi2deriv(self,R,Z,phi=0.,t=0.):
        """
        NAME:

           phi2deriv

        PURPOSE:

           evaluate the second azimuthal derivative

        INPUT:

           R - Galactocentric radius (can be Quantity)

           Z - vertical height (can be Quantity)

           phi - Galactocentric azimuth (can be Quantity)

           t - time (can be Quantity)

        OUTPUT:

           d2Phi/dphi2

        HISTORY:

           2013-09-24 - Written - Bovy (IAS)

        """
        try:
            return self._amp*self._phi2deriv(R,Z,phi=phi,t=t)
        except AttributeError: #pragma: no cover
            if self.isNonAxi:
                raise PotentialError("'_phi2deriv' function not implemented for this non-axisymmetric potential")
            return 0.

    @potential_physical_input
    @physical_conversion('forcederivative',pop=True)
    def Rphideriv(self,R,Z,phi=0.,t=0.):
        """
        NAME:

           Rphideriv

        PURPOSE:

           evaluate the mixed radial, azimuthal derivative

        INPUT:

           R - Galactocentric radius (can be Quantity)

           Z - vertical height (can be Quantity)

           phi - Galactocentric azimuth (can be Quantity)

           t - time (can be Quantity)

        OUTPUT:

           d2Phi/dphidR

        HISTORY:

           2014-06-30 - Written - Bovy (IAS)

        """
        try:
            return self._amp*self._Rphideriv(R,Z,phi=phi,t=t)
        except AttributeError: #pragma: no cover
            if self.isNonAxi:
                raise PotentialError("'_Rphideriv' function not implemented for this non-axisymmetric potential")
            return 0.

    def toPlanar(self):
        """
        NAME:

           toPlanar

        PURPOSE:

           convert a 3D potential into a planar potential in the mid-plane

        INPUT:

           (none)

        OUTPUT:

           planarPotential

        HISTORY:

           unknown

        """
        from ..potential import toPlanarPotential
        return toPlanarPotential(self)

    def toVertical(self,R,phi=None,t0=0.):
        """
        NAME:

           toVertical

        PURPOSE:

           convert a 3D potential into a linear (vertical) potential at R

        INPUT:

           R - Galactocentric radius at which to create the vertical potential (can be Quantity)

           phi= (None) Galactocentric azimuth at which to create the vertical potential (can be Quantity); required for non-axisymmetric potential

           t0= (0.) time at which to create the vertical potential (can be Quantity)

        OUTPUT:

           linear (vertical) potential: Phi(z,phi,t) = Phi(R,z,phi,t)-Phi(R,0.,phi0,t0) where phi0 and t0 are the phi and t inputs

        HISTORY

           unknown

        """
        from ..potential import toVerticalPotential
        return toVerticalPotential(self,R,phi=phi,t0=t0)

    def plot(self,t=0.,rmin=0.,rmax=1.5,nrs=21,zmin=-0.5,zmax=0.5,nzs=21,
             effective=False,Lz=None,phi=None,xy=False,
             xrange=None,yrange=None,
             justcontours=False,levels=None,cntrcolors=None,
             ncontours=21,savefilename=None):
        """
        NAME:

           plot

        PURPOSE:

           plot the potential

        INPUT:

           t= time to plot potential at

           rmin= minimum R (can be Quantity) [xmin if xy]

           rmax= maximum R (can be Quantity) [ymax if xy]

           nrs= grid in R

           zmin= minimum z (can be Quantity) [ymin if xy]

           zmax= maximum z (can be Quantity) [ymax if xy]

           nzs= grid in z

           phi= (None) azimuth to use for non-axisymmetric potentials

           xy= (False) if True, plot the potential in X-Y

           effective= (False) if True, plot the effective potential Phi + Lz^2/2/R^2

           Lz= (None) angular momentum to use for the effective potential when effective=True

           justcontours= (False) if True, just plot contours

           savefilename - save to or restore from this savefile (pickle)

           xrange, yrange= can be specified independently from rmin,zmin, etc.

           levels= (None) contours to plot

           ncontours - number of contours when levels is None

           cntrcolors= (None) colors of the contours (single color or array with length ncontours)

        OUTPUT:

           plot to output device

        HISTORY:

           2010-07-09 - Written - Bovy (NYU)

           2014-04-08 - Added effective= - Bovy (IAS)

        """
        rmin= conversion.parse_length(rmin,ro=self._ro)
        rmax= conversion.parse_length(rmax,ro=self._ro)
        zmin= conversion.parse_length(zmin,ro=self._ro)
        zmax= conversion.parse_length(zmax,ro=self._ro)
        if xrange is None: xrange= [rmin,rmax]
        if yrange is None: yrange= [zmin,zmax]
        if not savefilename is None and os.path.exists(savefilename):
            print("Restoring savefile "+savefilename+" ...")
            savefile= open(savefilename,'rb')
            potRz= pickle.load(savefile)
            Rs= pickle.load(savefile)
            zs= pickle.load(savefile)
            savefile.close()
        else:
            if effective and Lz is None:
                raise RuntimeError("When effective=True, you need to specify Lz=")
            Rs= numpy.linspace(xrange[0],xrange[1],nrs)
            zs= numpy.linspace(yrange[0],yrange[1],nzs)
            potRz= numpy.zeros((nrs,nzs))
            for ii in range(nrs):
                for jj in range(nzs):
                    if xy:
                        R,phi,z= coords.rect_to_cyl(Rs[ii],zs[jj],0.)
                    else:
                        R,z= Rs[ii], zs[jj]
                    potRz[ii,jj]= evaluatePotentials(self,
                                                     R,z,t=t,phi=phi,
                                                     use_physical=False)
                if effective:
                    potRz[ii,:]+= 0.5*Lz**2/Rs[ii]**2.
            #Don't plot outside of the desired range
            potRz[Rs < rmin,:]= numpy.nan
            potRz[Rs > rmax,:]= numpy.nan
            potRz[:,zs < zmin]= numpy.nan
            potRz[:,zs > zmax]= numpy.nan
            if not savefilename == None:
                print("Writing savefile "+savefilename+" ...")
                savefile= open(savefilename,'wb')
                pickle.dump(potRz,savefile)
                pickle.dump(Rs,savefile)
                pickle.dump(zs,savefile)
                savefile.close()
        if xy:
            xlabel= r'$x/R_0$'
            ylabel= r'$y/R_0$'
        else:
            xlabel=r"$R/R_0$"
            ylabel=r"$z/R_0$"
        if levels is None:
            levels= numpy.linspace(numpy.nanmin(potRz),numpy.nanmax(potRz),ncontours)
        if cntrcolors is None:
            cntrcolors= 'k'
        return plot.dens2d(potRz.T,origin='lower',cmap='gist_gray',contours=True,
                           xlabel=xlabel,ylabel=ylabel,
                           xrange=xrange,
                           yrange=yrange,
                           aspect=.75*(rmax-rmin)/(zmax-zmin),
                           cntrls='-',
                           justcontours=justcontours,
                           levels=levels,cntrcolors=cntrcolors)

    def plotDensity(self,t=0.,
                    rmin=0.,rmax=1.5,nrs=21,zmin=-0.5,zmax=0.5,nzs=21,
                    phi=None,xy=False,
                    ncontours=21,savefilename=None,aspect=None,log=False,
                    justcontours=False):
        """
        NAME:

           plotDensity

        PURPOSE:

           plot the density of this potential

        INPUT:

           t= time to plot potential at

           rmin= minimum R (can be Quantity) [xmin if xy]

           rmax= maximum R (can be Quantity) [ymax if xy]

           nrs= grid in R

           zmin= minimum z (can be Quantity) [ymin if xy]

           zmax= maximum z (can be Quantity) [ymax if xy]

           nzs= grid in z

           phi= (None) azimuth to use for non-axisymmetric potentials

           xy= (False) if True, plot the density in X-Y

           ncontours= number of contours

           justcontours= (False) if True, just plot contours

           savefilename= save to or restore from this savefile (pickle)

           log= if True, plot the log density

        OUTPUT:

           plot to output device

        HISTORY:

           2014-01-05 - Written - Bovy (IAS)

        """
        return plotDensities(self,rmin=rmin,rmax=rmax,nrs=nrs,
                             zmin=zmin,zmax=zmax,nzs=nzs,phi=phi,xy=xy,t=t,
                             ncontours=ncontours,savefilename=savefilename,
                             justcontours=justcontours,
                             aspect=aspect,log=log)

    def plotSurfaceDensity(self,t=0.,z=numpy.inf,
                           xmin=0.,xmax=1.5,nxs=21,ymin=-0.5,ymax=0.5,nys=21,
                           ncontours=21,savefilename=None,aspect=None,
                           log=False,justcontours=False):
        """
        NAME:

           plotSurfaceDensity

        PURPOSE:

           plot the surface density of this potential

        INPUT:

           t= time to plot potential at

           z= (inf) height between which to integrate the density (from -z to z; can be a Quantity) 

           xmin= minimum x (can be Quantity)

           xmax= maximum x (can be Quantity)

           nxs= grid in x

           ymin= minimum y (can be Quantity)

           ymax= maximum y (can be Quantity)

           nys= grid in y

           ncontours= number of contours

           justcontours= (False) if True, just plot contours

           savefilename= save to or restore from this savefile (pickle)

           log= if True, plot the log density

        OUTPUT:

           plot to output device

        HISTORY:

           2020-08-19 - Written - Bovy (UofT)

        """
        return plotSurfaceDensities(self,xmin=xmin,xmax=xmax,nxs=nxs,
                                    ymin=ymin,ymax=ymax,nys=nys,t=t,z=z,
                                    ncontours=ncontours,
                                    savefilename=savefilename,
                                    justcontours=justcontours,
                                    aspect=aspect,log=log)
    
    @potential_physical_input
    @physical_conversion('velocity',pop=True)
    def vcirc(self,R,phi=None,t=0.):
        """
        
        NAME:
        
            vcirc
        
        PURPOSE:
        
            calculate the circular velocity at R in this potential

        INPUT:
        
            R - Galactocentric radius (can be Quantity)
        
            phi= (None) azimuth to use for non-axisymmetric potentials

            t - time (optional; can be Quantity)

        OUTPUT:
        
            circular rotation velocity
        
        HISTORY:
        
            2011-10-09 - Written - Bovy (IAS)
        
            2016-06-15 - Added phi= keyword for non-axisymmetric potential - Bovy (UofT)

        """  
        return numpy.sqrt(R*-self.Rforce(R,0.,phi=phi,t=t,use_physical=False))

    @potential_physical_input
    @physical_conversion('frequency',pop=True)
    def dvcircdR(self,R,phi=None,t=0.):
        """
        
        NAME:
        
            dvcircdR
        
        PURPOSE:
        
            calculate the derivative of the circular velocity at R wrt R
            in this potential

        INPUT:
        
            R - Galactocentric radius (can be Quantity)
        
            phi= (None) azimuth to use for non-axisymmetric potentials

            t - time (optional; can be Quantity)

        OUTPUT:
        
            derivative of the circular rotation velocity wrt R
        
        HISTORY:
        
            2013-01-08 - Written - Bovy (IAS)
        
            2016-06-28 - Added phi= keyword for non-axisymmetric potential - Bovy (UofT)

        """
        return 0.5*(-self.Rforce(R,0.,phi=phi,t=t,use_physical=False)\
                         +R*self.R2deriv(R,0.,phi=phi,t=t,use_physical=False))\
                         /self.vcirc(R,phi=phi,t=t,use_physical=False)

    @potential_physical_input
    @physical_conversion('frequency',pop=True)
    def omegac(self,R,t=0.):
        """
        
        NAME:
        
            omegac
        
        PURPOSE:
        
            calculate the circular angular speed at R in this potential

        INPUT:
        
            R - Galactocentric radius (can be Quantity)

            t - time (optional; can be Quantity)
        
        OUTPUT:
        
            circular angular speed
        
        HISTORY:
        
            2011-10-09 - Written - Bovy (IAS)
        
        """
        return numpy.sqrt(-self.Rforce(R,0.,t=t,use_physical=False)/R)

    @potential_physical_input
    @physical_conversion('frequency',pop=True)
    def epifreq(self,R,t=0.):
        """
        
        NAME:
        
           epifreq
        
        PURPOSE:
        
           calculate the epicycle frequency at R in this potential
        
        INPUT:
        
           R - Galactocentric radius (can be Quantity)

           t - time (optional; can be Quantity)
        
        OUTPUT:
        
           epicycle frequency
        
        HISTORY:
        
           2011-10-09 - Written - Bovy (IAS)
        
        """
        return numpy.sqrt(self.R2deriv(R,0.,t=t,use_physical=False)\
                           -3./R*self.Rforce(R,0.,t=t,use_physical=False))

    @potential_physical_input
    @physical_conversion('frequency',pop=True)
    def verticalfreq(self,R,t=0.):
        """
        
        NAME:
        
           verticalfreq
        
        PURPOSE:
        
           calculate the vertical frequency at R in this potential
        
        INPUT:
        
           R - Galactocentric radius (can be Quantity)

           t - time (optional; can be Quantity)
        
        OUTPUT:
        
           vertical frequency
        
        HISTORY:
        
           2012-07-25 - Written - Bovy (IAS@MPIA)
        
        """
        return numpy.sqrt(self.z2deriv(R,0.,t=t,use_physical=False))

    @physical_conversion('position',pop=True)
    def lindbladR(self,OmegaP,m=2,t=0.,**kwargs):
        """
        
        NAME:
        
           lindbladR
        
        PURPOSE:
        
            calculate the radius of a Lindblad resonance
        
        INPUT:
        
           OmegaP - pattern speed (can be Quantity)

           m= order of the resonance (as in m(O-Op)=kappa (negative m for outer)
              use m='corotation' for corotation
              +scipy.optimize.brentq xtol,rtol,maxiter kwargs

           t - time (optional; can be Quantity)
        
        OUTPUT:
        
           radius of Linblad resonance, None if there is no resonance
        
        HISTORY:
        
           2011-10-09 - Written - Bovy (IAS)
        
        """
        OmegaP= conversion.parse_frequency(OmegaP,ro=self._ro,vo=self._vo)
        return lindbladR(self,OmegaP,m=m,t=t,use_physical=False,**kwargs)

    @potential_physical_input
    @physical_conversion('velocity',pop=True)
    def vesc(self,R,t=0.):
        """

        NAME:

            vesc

        PURPOSE:

            calculate the escape velocity at R for this potential

        INPUT:

            R - Galactocentric radius (can be Quantity)

            t - time (optional; can be Quantity)

        OUTPUT:

            escape velocity

        HISTORY:

            2011-10-09 - Written - Bovy (IAS)

        """
        return numpy.sqrt(2.*(self(_INF,0.,t=t,use_physical=False)\
                               -self(R,0.,t=t,use_physical=False)))
        
    @physical_conversion('position',pop=True)
    def rl(self,lz,t=0.):
        """
        NAME:
        
            rl
        
        PURPOSE:
        
            calculate the radius of a circular orbit of Lz
        
        INPUT:
        
        
            lz - Angular momentum (can be Quantity)

            t - time (optional; can be Quantity)
        
        OUTPUT:
        
            radius
        
        HISTORY:
        
            2012-07-30 - Written - Bovy (IAS@MPIA)
        
        NOTE:
        
            seems to take about ~0.5 ms for a Miyamoto-Nagai potential; 
            ~0.75 ms for a MWPotential
        
        """
        lz= conversion.parse_angmom(lz,ro=self._ro,vo=self._vo)
        return rl(self,lz,t=t,use_physical=False)

    @potential_physical_input
    @physical_conversion('dimensionless',pop=True)
    def flattening(self,R,z,t=0.):
        """
        
        NAME:
        
           flattening
        
        PURPOSE:
        
           calculate the potential flattening, defined as sqrt(fabs(z/R F_R/F_z))
        
        INPUT:
        
           R - Galactocentric radius (can be Quantity)

           z - height (can be Quantity)

           t - time (optional; can be Quantity)
        
        OUTPUT:
        
           flattening
        
        HISTORY:
        
           2012-09-13 - Written - Bovy (IAS)
        
        """
        return numpy.sqrt(numpy.fabs(z/R*self.Rforce(R,z,t=t,use_physical=False)\
                                   /self.zforce(R,z,t=t,use_physical=False)))

    @physical_conversion('velocity',pop=True)
    def vterm(self,l,t=0.,deg=True):
        """
        
        NAME:
        
            vterm
        
        PURPOSE:
        
            calculate the terminal velocity at l in this potential

        INPUT:
        
            l - Galactic longitude [deg/rad; can be Quantity)

            t - time (optional; can be Quantity)

            deg= if True (default), l in deg
        
        OUTPUT:
        
            terminal velocity
        
        HISTORY:
        
            2013-05-31 - Written - Bovy (IAS)
        
        """
        if _APY_LOADED and isinstance(l,units.Quantity):
            l= conversion.parse_angle(l)
            deg= False
        if deg:
            sinl= numpy.sin(l/180.*numpy.pi)
        else:
            sinl= numpy.sin(l)
        return sinl*(self.omegac(numpy.fabs(sinl),t=t,use_physical=False)\
                         -self.omegac(1.,t=t,use_physical=False))

    def plotRotcurve(self,*args,**kwargs):
        """
        NAME:

           plotRotcurve

        PURPOSE:

           plot the rotation curve for this potential (in the z=0 plane for
           non-spherical potentials)

        INPUT:

           Rrange - range (can be Quantity)

           grid= number of points to plot

           savefilename=- save to or restore from this savefile (pickle)

           +galpy.util.plot.plot(*args,**kwargs)

        OUTPUT:

           plot to output device

        HISTORY:

           2010-07-10 - Written - Bovy (NYU)

        """
        return plotRotcurve(self,*args,**kwargs)

    def plotEscapecurve(self,*args,**kwargs):
        """
        NAME:

           plotEscapecurve

        PURPOSE:

           plot the escape velocity  curve for this potential 
           (in the z=0 plane for non-spherical potentials)

        INPUT:

           Rrange - range (can be Quantity)

           grid= number of points to plot

           savefilename= save to or restore from this savefile (pickle)

           +galpy.util.plot.plot(*args,**kwargs)

        OUTPUT:

           plot to output device

        HISTORY:

           2010-08-08 - Written - Bovy (NYU)

        """
        return plotEscapecurve(self.toPlanar(),*args,**kwargs)

    def conc(self,H=70.,Om=0.3,t=0.,overdens=200.,wrtcrit=False,
             ro=None,vo=None):
        """
        NAME:

           conc

        PURPOSE:

           return the concentration

        INPUT:

           H= (default: 70) Hubble constant in km/s/Mpc
           
           Om= (default: 0.3) Omega matter

           t - time (optional; can be Quantity)
       
           overdens= (200) overdensity which defines the virial radius

           wrtcrit= (False) if True, the overdensity is wrt the critical density rather than the mean matter density
           
           ro= distance scale in kpc or as Quantity (default: object-wide, which if not set is 8 kpc))

           vo= velocity scale in km/s or as Quantity (default: object-wide, which if not set is 220 km/s))

        OUTPUT:

           concentration (scale/rvir)

        HISTORY:

           2014-04-03 - Written - Bovy (IAS)

        """
        if ro is None: ro= self._ro
        if vo is None: vo= self._vo
        try:
            return self.rvir(H=H,Om=Om,t=t,overdens=overdens,wrtcrit=wrtcrit,
                             ro=ro,vo=vo,use_physical=False)/self._scale
        except AttributeError:
            raise AttributeError("This potential does not have a '_scale' defined to base the concentration on or does not support calculating the virial radius")

    def nemo_accname(self):
        """
        NAME:

           nemo_accname

        PURPOSE:

           return the accname potential name for use of this potential with NEMO

        INPUT:

           (none)

        OUTPUT:

           Acceleration name

        HISTORY:

           2014-12-18 - Written - Bovy (IAS)

        """
        try:
            return self._nemo_accname
        except AttributeError:
            raise AttributeError('NEMO acceleration name not supported for %s' % self.__class__.__name__)

    def nemo_accpars(self,vo,ro):
        """
        NAME:

           nemo_accpars

        PURPOSE:

           return the accpars potential parameters for use of this potential with NEMO

        INPUT:

           vo - velocity unit in km/s

           ro - length unit in kpc

        OUTPUT:

           accpars string

        HISTORY:

           2014-12-18 - Written - Bovy (IAS)

        """
        try:
            return self._nemo_accpars(vo,ro)
        except AttributeError:
            raise AttributeError('NEMO acceleration parameters not supported for %s' % self.__class__.__name__)

    @potential_physical_input
    @physical_conversion('position',pop=True)
    def rtide(self,R,z,phi=0.,t=0.,M=None):
        """
            
        NAME:
            
            rtide
            
        PURPOSE:
            
            Calculate the tidal radius for object of mass M assuming a circular orbit as

            .. math::

               r_t^3 = \\frac{GM_s}{\\Omega^2-\\mathrm{d}^2\\Phi/\\mathrm{d}r^2}

            where :math:`M_s` is the cluster mass, :math:`\\Omega` is the circular frequency, and :math:`\Phi` is the gravitational potential. For non-spherical potentials, we evaluate :math:`\\Omega^2 = (1/r)(\\mathrm{d}\\Phi/\\mathrm{d}r)` and evaluate the derivatives at the given position of the cluster.

        INPUT:
        
            R - Galactocentric radius (can be Quantity)
            
            z - height (can be Quantity)
            
            phi - azimuth (optional; can be Quantity)
            
            t - time (optional; can be Quantity)
            
            M - (default = None) Mass of object (can be Quantity)
            
        OUTPUT:
            
            Tidal Radius
        
        HISTORY:
            
            2018-03-21 - Written - Webb (UofT)
            
        """
        if M is None:
            #Make sure an object mass is given
            raise PotentialError("Mass parameter M= needs to be set to compute tidal radius")
        r= numpy.sqrt(R**2.+z**2.)
        omegac2= -self.rforce(R,z,phi=phi,t=t,use_physical=False)/r
        d2phidr2= self.r2deriv(R,z,phi=phi,t=t,use_physical=False)
        return (M/(omegac2-d2phidr2))**(1./3.)

    @potential_physical_input
    @physical_conversion('forcederivative',pop=True)
    def ttensor(self,R,z,phi=0.,t=0.,eigenval=False):
        """
            
        NAME:
        
            ttensor
            
        PURPOSE:
        
            Calculate the tidal tensor Tij=-d(Psi)(dxidxj)
            
        INPUT:
        
            R - Galactocentric radius (can be Quantity)
            
            z - height (can be Quantity)
            
            phi - azimuth (optional; can be Quantity)
            
            t - time (optional; can be Quantity)
            
            eigenval - return eigenvalues if true (optional; boolean)
            
        OUTPUT:
        
            Tidal Tensor
        
        HISTORY:
        
            2018-03-21 - Written - Webb (UofT)

        """
        if self.isNonAxi:
            raise PotentialError("Tidal tensor calculation is currently only implemented for axisymmetric potentials")
        #Evaluate forces, angles and derivatives
        Rderiv= -self.Rforce(R,z,phi=phi,t=t,use_physical=False)       
        phideriv= -self.phiforce(R,z,phi=phi,t=t,use_physical=False)
        R2deriv= self.R2deriv(R,z,phi=phi,t=t,use_physical=False)
        z2deriv= self.z2deriv(R,z,phi=phi,t=t,use_physical=False)
        phi2deriv= self.phi2deriv(R,z,phi=phi,t=t,use_physical=False)
        Rzderiv= self.Rzderiv(R,z,phi=phi,t=t,use_physical=False)
        Rphideriv= self.Rphideriv(R,z,phi=phi,t=t,use_physical=False)
        #Temporarily set zphideriv to zero until zphideriv is added to Class
        zphideriv=0.0
        cosphi=numpy.cos(phi)
        sinphi=numpy.sin(phi)
        cos2phi=cosphi**2.0
        sin2phi=sinphi**2.0
        R2=R**2.0
        R3=R**3.0
        # Tidal tensor
        txx= R2deriv*cos2phi-Rphideriv*2.*cosphi*sinphi/R+Rderiv*sin2phi/R\
            +phi2deriv*sin2phi/R2+phideriv*2.*cosphi*sinphi/R2
        tyx= R2deriv*sinphi*cosphi+Rphideriv*(cos2phi-sin2phi)/R\
            -Rderiv*sinphi*cosphi/R-phi2deriv*sinphi*cosphi/R2\
            +phideriv*(sin2phi-cos2phi)/R2      
        tzx=Rzderiv*cosphi-zphideriv*sinphi/R
        tyy=R2deriv*sin2phi+Rphideriv*2.*cosphi*sinphi/R+Rderiv*cos2phi/R\
            +phi2deriv*cos2phi/R2-phideriv*2.*sinphi*cosphi/R2
        txy=tyx
        tzy=Rzderiv*sinphi+zphideriv*cosphi/R
        txz=tzx
        tyz=tzy
        tzz=z2deriv
        tij=-numpy.array([[txx,txy,txz],[tyx,tyy,tyz],[tzx,tzy,tzz]])
        if eigenval:
           return numpy.linalg.eigvals(tij)
        else:
            return tij

    @physical_conversion('position',pop=True)
    def zvc(self,R,E,Lz,phi=0.,t=0.):
        """
        
        NAME:
        
           zvc
            
        PURPOSE:
        
           Calculate the zero-velocity curve: z such that Phi(R,z) + Lz/[2R^2] = E (assumes that F_z(R,z) = negative at positive z such that there is a single solution)
            
        INPUT:
        
           R - Galactocentric radius (can be Quantity)
            
           E - Energy (can be Quantity)

           Lz - Angular momentum (can be Quantity)
            
           phi - azimuth (optional; can be Quantity)
            
           t - time (optional; can be Quantity)
            
        OUTPUT:
        
           z such that Phi(R,z) + Lz/[2R^2] = E
        
        HISTORY:
        
           2020-08-20 - Written - Bovy (UofT)
        """
        return zvc(self,R,E,Lz,phi=phi,t=t,use_physical=False)
    
    @physical_conversion('position',pop=True)
    def zvc_range(self,E,Lz,phi=0.,t=0.):
        """
            
        NAME:
        
           zvc_range
            
        PURPOSE:
        
          Calculate the minimum and maximum radius for which the zero-velocity curve exists for this energy and angular momentum (R such that Phi(R,0) + Lz/[2R^2] = E)
            
        INPUT:
        
           E - Energy (can be Quantity)

           Lz - Angular momentum (can be Quantity)
            
           phi - azimuth (optional; can be Quantity)
            
           t - time (optional; can be Quantity)
            
        OUTPUT:
        
           Solutions R such that Phi(R,0) + Lz/[2R^2] = E
        
        HISTORY:
        
           2020-08-20 - Written - Bovy (UofT)
        """
        return zvc_range(self,E,Lz,phi=phi,t=t,use_physical=False)
    
class PotentialError(Exception): #pragma: no cover
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

@potential_physical_input
@physical_conversion('energy',pop=True)
def evaluatePotentials(Pot,R,z,phi=None,t=0.,dR=0,dphi=0):
    """
    NAME:

       evaluatePotentials

    PURPOSE:

       convenience function to evaluate a possible sum of potentials

    INPUT:

       Pot - potential or list of potentials (dissipative forces in such a list are ignored)

       R - cylindrical Galactocentric distance (can be Quantity)

       z - distance above the plane (can be Quantity)

       phi - azimuth (can be Quantity)

       t - time (can be Quantity)

       dR= dphi=, if set to non-zero integers, return the dR, dphi't derivative instead

    OUTPUT:

       Phi(R,z)

    HISTORY:

       2010-04-16 - Written - Bovy (NYU)

    """
    return _evaluatePotentials(Pot,R,z,phi=phi,t=t,dR=dR,dphi=dphi)

def _evaluatePotentials(Pot,R,z,phi=None,t=0.,dR=0,dphi=0):
    """Raw, undecorated function for internal use"""
    nonAxi= _isNonAxi(Pot)
    if nonAxi and phi is None:
        raise PotentialError("The (list of) Potential instances is non-axisymmetric, but you did not provide phi")
    isList= isinstance(Pot,list)
    if isList:
        out= 0.
        for pot in Pot:
            if not isinstance(pot,DissipativeForce):
                out+= pot._call_nodecorator(R,z,phi=phi,t=t,dR=dR,dphi=dphi)
        return out
    elif isinstance(Pot,Potential):
        return Pot._call_nodecorator(R,z,phi=phi,t=t,dR=dR,dphi=dphi)
    else: #pragma: no cover 
        raise PotentialError("Input to 'evaluatePotentials' is neither a Potential-instance or a list of such instances")

@potential_physical_input
@physical_conversion('density',pop=True)
def evaluateDensities(Pot,R,z,phi=None,t=0.,forcepoisson=False):
    """
    NAME:

       evaluateDensities

    PURPOSE:

       convenience function to evaluate a possible sum of densities

    INPUT:

       Pot - potential or list of potentials (dissipative forces in such a list are ignored)

       R - cylindrical Galactocentric distance (can be Quantity)

       z - distance above the plane (can be Quantity)

       phi - azimuth (can be Quantity)

       t - time (can be Quantity)

       forcepoisson= if True, calculate the density through the Poisson equation, even if an explicit expression for the density exists

    OUTPUT:

       rho(R,z)

    HISTORY:

       2010-08-08 - Written - Bovy (NYU)

       2013-12-28 - Added forcepoisson - Bovy (IAS)

    """
    isList= isinstance(Pot,list)
    nonAxi= _isNonAxi(Pot)
    if nonAxi and phi is None:
        raise PotentialError("The (list of) Potential instances is non-axisymmetric, but you did not provide phi")
    if isList:
        out= 0.
        for pot in Pot:
            if not isinstance(pot,DissipativeForce):
                out+= pot.dens(R,z,phi=phi,t=t,forcepoisson=forcepoisson,
                               use_physical=False)
        return out
    elif isinstance(Pot,Potential):
        return Pot.dens(R,z,phi=phi,t=t,forcepoisson=forcepoisson,
                        use_physical=False)
    else: #pragma: no cover 
        raise PotentialError("Input to 'evaluateDensities' is neither a Potential-instance or a list of such instances")

@potential_physical_input
@physical_conversion('surfacedensity',pop=True)
def evaluateSurfaceDensities(Pot,R,z,phi=None,t=0.,forcepoisson=False):
    """
    NAME:

       evaluateSurfaceDensities

    PURPOSE:

       convenience function to evaluate a possible sum of surface densities

    INPUT:

       Pot - potential or list of potentials (dissipative forces in such a list are ignored)

       R - cylindrical Galactocentric distance (can be Quantity)

       z - distance above the plane (can be Quantity)

       phi - azimuth (can be Quantity)

       t - time (can be Quantity)

       forcepoisson= if True, calculate the surface density through the Poisson equation, even if an explicit expression for the surface density exists

    OUTPUT:

       Sigma(R,z)

    HISTORY:

       2018-08-20 - Written - Bovy (UofT)

    """
    isList= isinstance(Pot,list)
    nonAxi= _isNonAxi(Pot)
    if nonAxi and phi is None:
        raise PotentialError("The (list of) Potential instances is non-axisymmetric, but you did not provide phi")
    if isList:
        out= 0.
        for pot in Pot:
            if not isinstance(pot,DissipativeForce):
                out+= pot.surfdens(R,z,phi=phi,t=t,forcepoisson=forcepoisson,
                                   use_physical=False)
        return out
    elif isinstance(Pot,Potential):
        return Pot.surfdens(R,z,phi=phi,t=t,forcepoisson=forcepoisson,
                            use_physical=False)
    else: #pragma: no cover 
        raise PotentialError("Input to 'evaluateSurfaceDensities' is neither a Potential-instance or a list of such instances")

@potential_physical_input
@physical_conversion('mass',pop=True)
def mass(Pot,R,z=None,t=0.,forceint=False):
    """
    NAME:

       mass

    PURPOSE:

       convenience function to evaluate a possible sum of masses

    INPUT:

       Pot - potential or list of potentials (dissipative forces in such a list are ignored)

       R - cylindrical Galactocentric distance (can be Quantity)

       z= (None) vertical height up to which to integrate (can be Quantity) 

       t - time (can be Quantity)

       forceint= if True, calculate the mass through integration of the density, even if an explicit expression for the mass exists


    OUTPUT:

       Mass enclosed within the spherical shell with radius R if z is None else mass in the slab <R and between -z and z

    HISTORY:

       2021-02-07 - Written - Bovy (UofT)

       2021-03-15 - Changed to integrate to spherical shell for z is None slab otherwise - Bovy (UofT)

    """
    Pot= flatten(Pot)
    isList= isinstance(Pot,list)
    nonAxi= _isNonAxi(Pot)
    if nonAxi:
        raise NotImplementedError('mass for non-axisymmetric potentials is not currently supported')
    if isList:
        out= 0.
        for pot in Pot:
            if not isinstance(pot,DissipativeForce):
                out+= pot.mass(R,z=z,t=t,forceint=forceint,use_physical=False)
        return out
    elif isinstance(Pot,Potential):
        return Pot.mass(R,z=z,t=t,forceint=forceint,use_physical=False)
    else: #pragma: no cover 
        raise PotentialError("Input to 'mass' is neither a Potential-instance or a list of such instances")

@potential_physical_input
@physical_conversion('force',pop=True)
def evaluateRforces(Pot,R,z,phi=None,t=0.,v=None):
    """
    NAME:

       evaluateRforce

    PURPOSE:

       convenience function to evaluate a possible sum of potentials

    INPUT:

       Pot - a potential or list of potentials

       R - cylindrical Galactocentric distance (can be Quantity)

       z - distance above the plane (can be Quantity)

       phi - azimuth (optional; can be Quantity))

       t - time (optional; can be Quantity)

       v - current velocity in cylindrical coordinates (optional, but required when including dissipative forces; can be a Quantity)

    OUTPUT:

       F_R(R,z,phi,t)

    HISTORY:

       2010-04-16 - Written - Bovy (NYU)

       2018-03-16 - Added velocity input for dissipative forces - Bovy (UofT)

    """
    return _evaluateRforces(Pot,R,z,phi=phi,t=t,v=v)

def _evaluateRforces(Pot,R,z,phi=None,t=0.,v=None):
    """Raw, undecorated function for internal use"""
    isList= isinstance(Pot,list)
    nonAxi= _isNonAxi(Pot)
    if nonAxi and phi is None:
        raise PotentialError("The (list of) Potential instances is non-axisymmetric, but you did not provide phi")
    dissipative= _isDissipative(Pot)
    if dissipative and v is None:
        raise PotentialError("The (list of) Potential instances includes dissipative, but you did not provide the 3D velocity (required for dissipative forces")
    if isList:
        out= 0.
        for pot in Pot:
            if isinstance(pot,DissipativeForce):
                out+= pot._Rforce_nodecorator(R,z,phi=phi,t=t,v=v)
            else:
                out+= pot._Rforce_nodecorator(R,z,phi=phi,t=t)
        return out
    elif isinstance(Pot,Potential):
        return Pot._Rforce_nodecorator(R,z,phi=phi,t=t)
    elif isinstance(Pot,DissipativeForce):
        return Pot._Rforce_nodecorator(R,z,phi=phi,t=t,v=v)
    else: #pragma: no cover 
        raise PotentialError("Input to 'evaluateRforces' is neither a Potential-instance, DissipativeForce-instance or a list of such instances")

@potential_physical_input
@physical_conversion('energy',pop=True)
def evaluatephiforces(Pot,R,z,phi=None,t=0.,v=None):
    """
    NAME:

       evaluatephiforces

    PURPOSE:

       convenience function to evaluate a possible sum of potentials

    INPUT:
       Pot - a potential or list of potentials

       R - cylindrical Galactocentric distance (can be Quantity)

       z - distance above the plane (can be Quantity)

       phi - azimuth (optional; can be Quantity)

       t - time (optional; can be Quantity)

       v - current velocity in cylindrical coordinates (optional, but required when including dissipative forces; can be a Quantity)

    OUTPUT:

       F_phi(R,z,phi,t)

    HISTORY:

       2010-04-16 - Written - Bovy (NYU)

       2018-03-16 - Added velocity input for dissipative forces - Bovy (UofT)

    """
    return _evaluatephiforces(Pot,R,z,phi=phi,t=t,v=v)

def _evaluatephiforces(Pot,R,z,phi=None,t=0.,v=None):
    """Raw, undecorated function for internal use"""
    isList= isinstance(Pot,list)
    nonAxi= _isNonAxi(Pot)
    if nonAxi and phi is None:
        raise PotentialError("The (list of) Potential instances is non-axisymmetric, but you did not provide phi")
    dissipative= _isDissipative(Pot)
    if dissipative and v is None:
        raise PotentialError("The (list of) Potential instances includes dissipative, but you did not provide the 3D velocity (required for dissipative forces")
    if isList:
        out= 0.
        for pot in Pot:
            if isinstance(pot,DissipativeForce):
                out+= pot._phiforce_nodecorator(R,z,phi=phi,t=t,v=v)
            else:
                out+= pot._phiforce_nodecorator(R,z,phi=phi,t=t)
        return out
    elif isinstance(Pot,Potential):
        return Pot._phiforce_nodecorator(R,z,phi=phi,t=t)
    elif isinstance(Pot,DissipativeForce):
        return Pot._phiforce_nodecorator(R,z,phi=phi,t=t,v=v)
    else: #pragma: no cover 
        raise PotentialError("Input to 'evaluatephiforces' is neither a Potential-instance, DissipativeForce-instance or a list of such instances")

@potential_physical_input
@physical_conversion('force',pop=True)
def evaluatezforces(Pot,R,z,phi=None,t=0.,v=None):
    """
    NAME:

       evaluatezforces

    PURPOSE:

       convenience function to evaluate a possible sum of potentials

    INPUT:

       Pot - a potential or list of potentials

       R - cylindrical Galactocentric distance (can be Quantity)

       z - distance above the plane (can be Quantity)

       phi - azimuth (optional; can be Quantity)

       t - time (optional; can be Quantity)

       v - current velocity in cylindrical coordinates (optional, but required when including dissipative forces; can be a Quantity)

    OUTPUT:

       F_z(R,z,phi,t)

    HISTORY:

       2010-04-16 - Written - Bovy (NYU)

       2018-03-16 - Added velocity input for dissipative forces - Bovy (UofT)

    """
    return _evaluatezforces(Pot,R,z,phi=phi,t=t,v=v)

def _evaluatezforces(Pot,R,z,phi=None,t=0.,v=None):
    """Raw, undecorated function for internal use"""
    isList= isinstance(Pot,list)
    nonAxi= _isNonAxi(Pot)
    if nonAxi and phi is None:
        raise PotentialError("The (list of) Potential instances is non-axisymmetric, but you did not provide phi")
    dissipative= _isDissipative(Pot)
    if dissipative and v is None:
        raise PotentialError("The (list of) Potential instances includes dissipative, but you did not provide the 3D velocity (required for dissipative forces")
    if isList:
        out= 0.
        for pot in Pot:
            if isinstance(pot,DissipativeForce):
                out+= pot._zforce_nodecorator(R,z,phi=phi,t=t,v=v)
            else:
                out+= pot._zforce_nodecorator(R,z,phi=phi,t=t)
        return out
    elif isinstance(Pot,Potential):
        return Pot._zforce_nodecorator(R,z,phi=phi,t=t)
    elif isinstance(Pot,DissipativeForce):
        return Pot._zforce_nodecorator(R,z,phi=phi,t=t,v=v)
    else: #pragma: no cover 
        raise PotentialError("Input to 'evaluatezforces' is neither a Potential-instance, DissipativeForce-instance or a list of such instances")

@potential_physical_input
@physical_conversion('force',pop=True)
def evaluaterforces(Pot,R,z,phi=None,t=0.,v=None):
    """
    NAME:

       evaluaterforces

    PURPOSE:

       convenience function to evaluate a possible sum of potentials

    INPUT:

       Pot - a potential or list of potentials

       R - cylindrical Galactocentric distance (can be Quantity)

       z - distance above the plane (can be Quantity)

       phi - azimuth (optional; can be Quantity)

       t - time (optional; can be Quantity)

       v - current velocity in cylindrical coordinates (optional, but required when including dissipative forces; can be a Quantity)

    OUTPUT:

       F_r(R,z,phi,t)

    HISTORY:

       2016-06-10 - Written - Bovy (UofT)

    """
    isList= isinstance(Pot,list)
    nonAxi= _isNonAxi(Pot)
    if nonAxi and phi is None:
        raise PotentialError("The (list of) Potential instances is non-axisymmetric, but you did not provide phi")
    dissipative= _isDissipative(Pot)
    if dissipative and v is None:
        raise PotentialError("The (list of) Potential instances includes dissipative, but you did not provide the 3D velocity (required for dissipative forces")
    if isList:
        out= 0.
        for pot in Pot:
            if isinstance(pot,DissipativeForce):
                out+= pot.rforce(R,z,phi=phi,t=t,v=v,use_physical=False)
            else:
                out+= pot.rforce(R,z,phi=phi,t=t,use_physical=False)
        return out
    elif isinstance(Pot,Potential):
        return Pot.rforce(R,z,phi=phi,t=t,use_physical=False)
    elif isinstance(Pot,DissipativeForce):
        return Pot.rforce(R,z,phi=phi,t=t,v=v,use_physical=False)
    else: #pragma: no cover 
        raise PotentialError("Input to 'evaluaterforces' is neither a Potential-instance or a list of such instances")

@potential_physical_input
@physical_conversion('forcederivative',pop=True)
def evaluateR2derivs(Pot,R,z,phi=None,t=0.):
    """
    NAME:

       evaluateR2derivs

    PURPOSE:

       convenience function to evaluate a possible sum of potentials

    INPUT:

       Pot - a potential or list of potentials (dissipative forces in such a list are ignored)

       R - cylindrical Galactocentric distance (can be Quantity)

       z - distance above the plane (can be Quantity)

       phi - azimuth (optional; can be Quantity)

       t - time (optional; can be Quantity)

    OUTPUT:

       d2Phi/d2R(R,z,phi,t)

    HISTORY:

       2012-07-25 - Written - Bovy (IAS)

    """
    isList= isinstance(Pot,list)
    nonAxi= _isNonAxi(Pot)
    if nonAxi and phi is None:
        raise PotentialError("The (list of) Potential instances is non-axisymmetric, but you did not provide phi")
    if isList:
        out= 0.
        for pot in Pot:
            if not isinstance(pot,DissipativeForce):
                out+= pot.R2deriv(R,z,phi=phi,t=t,use_physical=False)
        return out
    elif isinstance(Pot,Potential):
        return Pot.R2deriv(R,z,phi=phi,t=t,use_physical=False)
    else: #pragma: no cover 
        raise PotentialError("Input to 'evaluateR2derivs' is neither a Potential-instance or a list of such instances")

@potential_physical_input
@physical_conversion('forcederivative',pop=True)
def evaluatez2derivs(Pot,R,z,phi=None,t=0.):
    """
    NAME:

       evaluatez2derivs

    PURPOSE:

       convenience function to evaluate a possible sum of potentials

    INPUT:

       Pot - a potential or list of potentials (dissipative forces in such a list are ignored)

       R - cylindrical Galactocentric distance (can be Quantity)

       z - distance above the plane (can be Quantity)

       phi - azimuth (optional; can be Quantity)

       t - time (optional; can be Quantity)

    OUTPUT:

       d2Phi/d2z(R,z,phi,t)

    HISTORY:

       2012-07-25 - Written - Bovy (IAS)

    """
    isList= isinstance(Pot,list)
    nonAxi= _isNonAxi(Pot)
    if nonAxi and phi is None:
        raise PotentialError("The (list of) Potential instances is non-axisymmetric, but you did not provide phi")
    if isList:
        out= 0.
        for pot in Pot:
            if not isinstance(pot,DissipativeForce):
                out+= pot.z2deriv(R,z,phi=phi,t=t,use_physical=False)
        return out
    elif isinstance(Pot,Potential):
        return Pot.z2deriv(R,z,phi=phi,t=t,use_physical=False)
    else: #pragma: no cover 
        raise PotentialError("Input to 'evaluatez2derivs' is neither a Potential-instance or a list of such instances")

@potential_physical_input
@physical_conversion('forcederivative',pop=True)
def evaluateRzderivs(Pot,R,z,phi=None,t=0.):
    """
    NAME:

       evaluateRzderivs

    PURPOSE:

       convenience function to evaluate a possible sum of potentials

    INPUT:

       Pot - a potential or list of potentials (dissipative forces in such a list are ignored)

       R - cylindrical Galactocentric distance (can be Quantity)

       z - distance above the plane (can be Quantity)

       phi - azimuth (optional; can be Quantity)

       t - time (optional; can be Quantity)

    OUTPUT:

       d2Phi/dz/dR(R,z,phi,t)

    HISTORY:

       2013-08-28 - Written - Bovy (IAS)

    """
    isList= isinstance(Pot,list)
    nonAxi= _isNonAxi(Pot)
    if nonAxi and phi is None:
        raise PotentialError("The (list of) Potential instances is non-axisymmetric, but you did not provide phi")
    if isList:
        out= 0.
        for pot in Pot:
            if not isinstance(pot,DissipativeForce):
                out+= pot.Rzderiv(R,z,phi=phi,t=t,use_physical=False)
        return out
    elif isinstance(Pot,Potential):
        return Pot.Rzderiv(R,z,phi=phi,t=t,use_physical=False)
    else: #pragma: no cover 
        raise PotentialError("Input to 'evaluateRzderivs' is neither a Potential-instance or a list of such instances")

@potential_physical_input
@physical_conversion('forcederivative',pop=True)
def evaluatephi2derivs(Pot,R,z,phi=None,t=0.):
    """
    NAME:

       evaluatephi2derivs

    PURPOSE:

       convenience function to evaluate a possible sum of potentials

    INPUT:

       Pot - a potential or list of potentials

       R - cylindrical Galactocentric distance (can be Quantity)

       z - distance above the plane (can be Quantity)

       phi - azimuth (optional; can be Quantity)

       t - time (optional; can be Quantity)

    OUTPUT:

       d2Phi/d2phi(R,z,phi,t)

    HISTORY:

       2018-03-28 - Written - Bovy (UofT)

    """
    isList= isinstance(Pot,list)
    nonAxi= _isNonAxi(Pot)
    if nonAxi and phi is None:
        raise PotentialError("The (list of) Potential instances is non-axisymmetric, but you did not provide phi")
    if isList:
        out= 0.
        for pot in Pot:
            if not isinstance(pot,DissipativeForce):
                out+= pot.phi2deriv(R,z,phi=phi,t=t,use_physical=False)
        return out
    elif isinstance(Pot,Potential):
        return Pot.phi2deriv(R,z,phi=phi,t=t,use_physical=False)
    else: #pragma: no cover 
        raise PotentialError("Input to 'evaluatephi2derivs' is neither a Potential-instance or a list of such instances")

@potential_physical_input
@physical_conversion('forcederivative',pop=True)
def evaluateRphiderivs(Pot,R,z,phi=None,t=0.):
    """
    NAME:

       evaluateRphiderivs

    PURPOSE:

       convenience function to evaluate a possible sum of potentials

    INPUT:

       Pot - a potential or list of potentials

       R - cylindrical Galactocentric distance (can be Quantity)

       z - distance above the plane (can be Quantity)

       phi - azimuth (optional; can be Quantity)

       t - time (optional; can be Quantity)

    OUTPUT:

       d2Phi/d2R(R,z,phi,t)

    HISTORY:

       2012-07-25 - Written - Bovy (IAS)

    """
    isList= isinstance(Pot,list)
    nonAxi= _isNonAxi(Pot)
    if nonAxi and phi is None:
        raise PotentialError("The (list of) Potential instances is non-axisymmetric, but you did not provide phi")
    if isList:
        out= 0.
        for pot in Pot:
            if not isinstance(pot,DissipativeForce):
                out+= pot.Rphideriv(R,z,phi=phi,t=t,use_physical=False)
        return out
    elif isinstance(Pot,Potential):
        return Pot.Rphideriv(R,z,phi=phi,t=t,use_physical=False)
    else: #pragma: no cover 
        raise PotentialError("Input to 'evaluateRphiderivs' is neither a Potential-instance or a list of such instances")

@potential_physical_input
@physical_conversion('forcederivative',pop=True)
def evaluater2derivs(Pot,R,z,phi=None,t=0.):
    """
    NAME:

       evaluater2derivs

    PURPOSE:

       convenience function to evaluate a possible sum of potentials

    INPUT:

       Pot - a potential or list of potentials

       R - cylindrical Galactocentric distance (can be Quantity)

       z - distance above the plane (can be Quantity)

       phi - azimuth (optional; can be Quantity)

       t - time (optional; can be Quantity)

    OUTPUT:

       d2phi/dr2(R,z,phi,t)

    HISTORY:

       2018-03-28 - Written - Bovy (UofT)

    """
    isList= isinstance(Pot,list)
    nonAxi= _isNonAxi(Pot)
    if nonAxi and phi is None:
        raise PotentialError("The (list of) Potential instances is non-axisymmetric, but you did not provide phi")
    if isList:
        out= 0.
        for pot in Pot:
            if not isinstance(pot,DissipativeForce):
                out+= pot.r2deriv(R,z,phi=phi,t=t,use_physical=False)
        return out
    elif isinstance(Pot,Potential):
        return Pot.r2deriv(R,z,phi=phi,t=t,use_physical=False)
    else: #pragma: no cover 
        raise PotentialError("Input to 'evaluater2derivs' is neither a Potential-instance or a list of such instances")

def plotPotentials(Pot,rmin=0.,rmax=1.5,nrs=21,zmin=-0.5,zmax=0.5,nzs=21,
                   phi=None,xy=False,t=0.,effective=False,Lz=None,
                   ncontours=21,savefilename=None,aspect=None,
                   justcontours=False,levels=None,cntrcolors=None):
        """
        NAME:

           plotPotentials

        PURPOSE:

           plot a set of potentials

        INPUT:

           Pot - Potential or list of Potential instances

           rmin= minimum R (can be Quantity) [xmin if xy]

           rmax= maximum R (can be Quantity) [ymax if xy]

           nrs= grid in R

           zmin= minimum z (can be Quantity) [ymin if xy]

           zmax= maximum z (can be Quantity) [ymax if xy]

           nzs= grid in z

           phi= (None) azimuth to use for non-axisymmetric potentials

           t= (0.) time to use to evaluate potential

           xy= (False) if True, plot the potential in X-Y

           effective= (False) if True, plot the effective potential Phi + Lz^2/2/R^2

           Lz= (None) angular momentum to use for the effective potential when effective=True

           justcontours= (False) if True, just plot contours

           levels= (None) contours to plot

           ncontours - number of contours when levels is None

           cntrcolors= (None) colors of the contours (single color or array with length ncontours)

           savefilename= save to or restore from this savefile (pickle)

        OUTPUT:

           plot to output device

        HISTORY:

           2010-07-09 - Written - Bovy (NYU)

        """
        Pot= flatten(Pot)
        rmin= conversion.parse_length(rmin,**get_physical(Pot))
        rmax= conversion.parse_length(rmax,**get_physical(Pot))
        zmin= conversion.parse_length(zmin,**get_physical(Pot))
        zmax= conversion.parse_length(zmax,**get_physical(Pot))
        if not savefilename == None and os.path.exists(savefilename):
            print("Restoring savefile "+savefilename+" ...")
            savefile= open(savefilename,'rb')
            potRz= pickle.load(savefile)
            Rs= pickle.load(savefile)
            zs= pickle.load(savefile)
            savefile.close()
        else:
            if effective and Lz is None:
                raise RuntimeError("When effective=True, you need to specify Lz=")
            Rs= numpy.linspace(rmin,rmax,nrs)
            zs= numpy.linspace(zmin,zmax,nzs)
            potRz= numpy.zeros((nrs,nzs))
            for ii in range(nrs):
                for jj in range(nzs):
                    if xy:
                        R,phi,z= coords.rect_to_cyl(Rs[ii],zs[jj],0.)
                    else:
                        R,z= Rs[ii], zs[jj]
                    potRz[ii,jj]= evaluatePotentials(Pot,numpy.fabs(R),
                                                     z,phi=phi,t=t,
                                                     use_physical=False)
                if effective:
                    potRz[ii,:]+= 0.5*Lz**2/Rs[ii]**2.
            if not savefilename == None:
                print("Writing savefile "+savefilename+" ...")
                savefile= open(savefilename,'wb')
                pickle.dump(potRz,savefile)
                pickle.dump(Rs,savefile)
                pickle.dump(zs,savefile)
                savefile.close()
        if aspect is None:
            aspect=.75*(rmax-rmin)/(zmax-zmin)
        if xy:
            xlabel= r'$x/R_0$'
            ylabel= r'$y/R_0$'
        else:
            xlabel=r"$R/R_0$"
            ylabel=r"$z/R_0$"
        if levels is None:
            levels= numpy.linspace(numpy.nanmin(potRz),numpy.nanmax(potRz),ncontours)
        if cntrcolors is None:
            cntrcolors= 'k'
        return plot.dens2d(potRz.T,origin='lower',cmap='gist_gray',contours=True,
                           xlabel=xlabel,ylabel=ylabel,
                           aspect=aspect,
                           xrange=[rmin,rmax],
                           yrange=[zmin,zmax],
                           cntrls='-',
                           justcontours=justcontours,
                           levels=levels,cntrcolors=cntrcolors)

def plotDensities(Pot,rmin=0.,rmax=1.5,nrs=21,zmin=-0.5,zmax=0.5,nzs=21,
                  phi=None,xy=False,t=0.,
                  ncontours=21,savefilename=None,aspect=None,log=False,
                  justcontours=False):
        """
        NAME:

           plotDensities

        PURPOSE:

           plot the density a set of potentials

        INPUT:

           Pot - Potential or list of Potential instances

           rmin= minimum R (can be Quantity) [xmin if xy]

           rmax= maximum R (can be Quantity) [ymax if xy]

           nrs= grid in R

           zmin= minimum z (can be Quantity) [ymin if xy]

           zmax= maximum z (can be Quantity) [ymax if xy]

           nzs= grid in z

           phi= (None) azimuth to use for non-axisymmetric potentials

           t= (0.) time to use to evaluate potential

           xy= (False) if True, plot the density in X-Y

           ncontours= number of contours

           justcontours= (False) if True, just plot contours

           savefilename= save to or restore from this savefile (pickle)

           log= if True, plot the log density

        OUTPUT:

           plot to output device

        HISTORY:

           2013-07-05 - Written - Bovy (IAS)

        """
        Pot= flatten(Pot)
        rmin= conversion.parse_length(rmin,**get_physical(Pot))
        rmax= conversion.parse_length(rmax,**get_physical(Pot))
        zmin= conversion.parse_length(zmin,**get_physical(Pot))
        zmax= conversion.parse_length(zmax,**get_physical(Pot))
        if not savefilename == None and os.path.exists(savefilename):
            print("Restoring savefile "+savefilename+" ...")
            savefile= open(savefilename,'rb')
            potRz= pickle.load(savefile)
            Rs= pickle.load(savefile)
            zs= pickle.load(savefile)
            savefile.close()
        else:
            Rs= numpy.linspace(rmin,rmax,nrs)
            zs= numpy.linspace(zmin,zmax,nzs)
            potRz= numpy.zeros((nrs,nzs))
            for ii in range(nrs):
                for jj in range(nzs):
                    if xy:
                        R,phi,z= coords.rect_to_cyl(Rs[ii],zs[jj],0.)
                    else:
                        R,z= Rs[ii], zs[jj]
                    potRz[ii,jj]= evaluateDensities(Pot,numpy.fabs(R),z,phi=phi,
                                                    t=t,
                                                    use_physical=False)
            if not savefilename == None:
                print("Writing savefile "+savefilename+" ...")
                savefile= open(savefilename,'wb')
                pickle.dump(potRz,savefile)
                pickle.dump(Rs,savefile)
                pickle.dump(zs,savefile)
                savefile.close()
        if aspect is None:
            aspect=.75*(rmax-rmin)/(zmax-zmin)
        if log:
            potRz= numpy.log(potRz)
        if xy:
            xlabel= r'$x/R_0$'
            ylabel= r'$y/R_0$'
        else:
            xlabel=r"$R/R_0$"
            ylabel=r"$z/R_0$"
        return plot.dens2d(potRz.T,origin='lower',
                           cmap='gist_yarg',contours=True,
                           xlabel=xlabel,ylabel=ylabel,
                           aspect=aspect,
                           xrange=[rmin,rmax],
                           yrange=[zmin,zmax],
                           cntrls='-',
                           justcontours=justcontours,
                           levels=numpy.linspace(numpy.nanmin(potRz),numpy.nanmax(potRz),
                                                 ncontours))

def plotSurfaceDensities(Pot,
                         xmin=-1.5,xmax=1.5,nxs=21,ymin=-1.5,ymax=1.5,nys=21,
                         z=numpy.inf,t=0.,
                         ncontours=21,savefilename=None,aspect=None,
                         log=False,justcontours=False):
        """
        NAME:

           plotSurfaceDensities

        PURPOSE:

           plot the surface density a set of potentials

        INPUT:

           Pot - Potential or list of Potential instances

           xmin= minimum x (can be Quantity)

           xmax= maximum x (can be Quantity)

           nxs= grid in x

           ymin= minimum y (can be Quantity)

           ymax= maximum y (can be Quantity)

           nys= grid in y

           z= (inf) height between which to integrate the density (from -z to z; can be a Quantity)

           t= (0.) time to use to evaluate potential

           ncontours= number of contours

           justcontours= (False) if True, just plot contours

           savefilename= save to or restore from this savefile (pickle)

           log= if True, plot the log density

        OUTPUT:

           plot to output device

        HISTORY:

           2020-08-19 - Written - Bovy (UofT)

        """
        Pot= flatten(Pot)
        xmin= conversion.parse_length(xmin,**get_physical(Pot))
        xmax= conversion.parse_length(xmax,**get_physical(Pot))
        ymin= conversion.parse_length(ymin,**get_physical(Pot))
        ymax= conversion.parse_length(ymax,**get_physical(Pot))
        if not savefilename == None and os.path.exists(savefilename):
            print("Restoring savefile "+savefilename+" ...")
            savefile= open(savefilename,'rb')
            surfxy= pickle.load(savefile)
            xs= pickle.load(savefile)
            ys= pickle.load(savefile)
            savefile.close()
        else:
            xs= numpy.linspace(xmin,xmax,nxs)
            ys= numpy.linspace(ymin,ymax,nys)
            surfxy= numpy.zeros((nxs,nys))
            for ii in range(nxs):
                for jj in range(nys):
                    R,phi,_= coords.rect_to_cyl(xs[ii],ys[jj],0.)
                    surfxy[ii,jj]= evaluateSurfaceDensities(Pot,
                                                            numpy.fabs(R),z,
                                                            phi=phi,
                                                            t=t,
                                                            use_physical=False)
            if not savefilename == None:
                print("Writing savefile "+savefilename+" ...")
                savefile= open(savefilename,'wb')
                pickle.dump(surfxy,savefile)
                pickle.dump(xs,savefile)
                pickle.dump(ys,savefile)
                savefile.close()
        if aspect is None:
            aspect= 1.
        if log:
            surfxy= numpy.log(surfxy)
        xlabel= r'$x/R_0$'
        ylabel= r'$y/R_0$'
        return plot.dens2d(surfxy.T,origin='lower',
                           cmap='gist_yarg',contours=True,
                           xlabel=xlabel,ylabel=ylabel,
                           aspect=aspect,
                           xrange=[xmin,xmax],
                           yrange=[ymin,ymax],
                           cntrls='-',
                           justcontours=justcontours,
                           levels=numpy.linspace(numpy.nanmin(surfxy),
                                                 numpy.nanmax(surfxy),
                                                 ncontours))
    
@potential_physical_input
@physical_conversion('frequency',pop=True)
def epifreq(Pot,R,t=0.):
    """
    
    NAME:
    
        epifreq
    
    PURPOSE:
    
        calculate the epicycle frequency at R in the potential Pot
    
    INPUT:

        Pot - Potential instance or list thereof
    
        R - Galactocentric radius (can be Quantity)

        t - time (optional; can be Quantity)
    
    OUTPUT:
    
        epicycle frequency
    
    HISTORY:
    
        2012-07-25 - Written - Bovy (IAS)
    
    """
    from .planarPotential import planarPotential
    if isinstance(Pot,(Potential,planarPotential)):
        return Pot.epifreq(R,t=t,use_physical=False)
    from ..potential import evaluateplanarRforces, evaluateplanarR2derivs
    from ..potential import PotentialError
    try:
        return numpy.sqrt(evaluateplanarR2derivs(Pot,R,t=t,use_physical=False)
                       -3./R*evaluateplanarRforces(Pot,R,t=t,use_physical=False))
    except PotentialError:
        from ..potential import RZToplanarPotential
        Pot= RZToplanarPotential(Pot)
        return numpy.sqrt(evaluateplanarR2derivs(Pot,R,t=t,use_physical=False)
                       -3./R*evaluateplanarRforces(Pot,R,t=t,use_physical=False))

@potential_physical_input
@physical_conversion('frequency',pop=True)
def verticalfreq(Pot,R,t=0.):
    """
    
    NAME:
    
       verticalfreq
        
    PURPOSE:
    
        calculate the vertical frequency at R in the potential Pot
    
    INPUT:

       Pot - Potential instance or list thereof
    
       R - Galactocentric radius (can be Quantity)

       t - time (optional; can be Quantity)
    
    OUTPUT:
    
        vertical frequency
    
    HISTORY:
    
        2012-07-25 - Written - Bovy (IAS@MPIA)
    
    """
    from .planarPotential import planarPotential
    if isinstance(Pot,(Potential,planarPotential)):
        return Pot.verticalfreq(R,t=t,use_physical=False)
    return numpy.sqrt(evaluatez2derivs(Pot,R,0.,t=t,use_physical=False))

@potential_physical_input
@physical_conversion('dimensionless',pop=True)
def flattening(Pot,R,z,t=0.):
    """
    
    NAME:
    
        flattening
    
    PURPOSE:
    
       calculate the potential flattening, defined as sqrt(fabs(z/R F_R/F_z))
    
    INPUT:

        Pot - Potential instance or list thereof
    
        R - Galactocentric radius (can be Quantity)
        
        z - height (can be Quantity)

        t - time (optional; can be Quantity)
    
    OUTPUT:
    
        flattening
    
    HISTORY:
    
        2012-09-13 - Written - Bovy (IAS)
    
    """
    return numpy.sqrt(numpy.fabs(z/R*evaluateRforces(Pot,R,z,t=t,use_physical=False)\
                               /evaluatezforces(Pot,R,z,t=t,use_physical=False)))

@physical_conversion('velocity',pop=True)
def vterm(Pot,l,t=0.,deg=True):
    """
    
    NAME:
    
        vterm
        
    PURPOSE:
    
        calculate the terminal velocity at l in this potential

    INPUT:
    
        Pot - Potential instance
    
        l - Galactic longitude [deg/rad; can be Quantity)

        t - time (optional; can be Quantity)
        
        deg= if True (default), l in deg
        
    OUTPUT:
        
        terminal velocity
        
    HISTORY:
        
        2013-05-31 - Written - Bovy (IAS)
        
    """
    Pot= flatten(Pot)
    if _APY_LOADED and isinstance(l,units.Quantity):
        l= conversion.parse_angle(l)
        deg= False
    if deg:
        sinl= numpy.sin(l/180.*numpy.pi)
    else:
        sinl= numpy.sin(l)
    return sinl*(omegac(Pot,sinl,t=t,use_physical=False)
                 -omegac(Pot,1.,t=t,use_physical=False))

@physical_conversion('position',pop=True)
def rl(Pot,lz,t=0.):
    """
    NAME:

       rl

    PURPOSE:

       calculate the radius of a circular orbit of Lz

    INPUT:

       Pot - Potential instance or list thereof

       lz - Angular momentum (can be Quantity)

       t - time (optional; can be Quantity)

    OUTPUT:

       radius

    HISTORY:

       2012-07-30 - Written - Bovy (IAS@MPIA)

    NOTE:

       seems to take about ~0.5 ms for a Miyamoto-Nagai potential; 
       ~0.75 ms for a MWPotential

    """
    Pot= flatten(Pot)
    lz= conversion.parse_angmom(lz,**conversion.get_physical(Pot))
    #Find interval
    rstart= _rlFindStart(numpy.fabs(lz),#assumes vo=1.
                         numpy.fabs(lz),
                         Pot, t=t)
    try:
        return optimize.brentq(_rlfunc,10.**-5.,rstart,
                               args=(numpy.fabs(lz),
                                     Pot,
                                     t),
                               maxiter=200,disp=False)
    except ValueError: #Probably lz small and starting lz to great
        rlower= _rlFindStart(10.**-5.,
                             numpy.fabs(lz),
                             Pot,t=t,lower=True)
        return optimize.brentq(_rlfunc,rlower,rstart,
                               args=(numpy.fabs(lz),
                                     Pot,t))
        

def _rlfunc(rl,lz,pot,t=0.):
    """Function that gives rvc-lz"""
    thisvcirc= vcirc(pot,rl,t=t,use_physical=False)
    return rl*thisvcirc-lz

def _rlFindStart(rl,lz,pot,t=0.,lower=False):
    """find a starting interval for rl"""
    rtry= 2.*rl
    while (2.*lower-1.)*_rlfunc(rtry,lz,pot,t=t) > 0.:
        if lower:
            rtry/= 2.
        else:
            rtry*= 2.
    return rtry

@physical_conversion('position',pop=True)
def lindbladR(Pot,OmegaP,m=2,t=0.,**kwargs):
    """
    NAME:

       lindbladR

    PURPOSE:

       calculate the radius of a Lindblad resonance

    INPUT:

       Pot - Potential instance or list of such instances

       OmegaP - pattern speed (can be Quantity)

       m= order of the resonance (as in m(O-Op)=kappa (negative m for outer)
          use m='corotation' for corotation
       +scipy.optimize.brentq xtol,rtol,maxiter kwargs

       t - time (optional; can be Quantity)

    OUTPUT:

       radius of Linblad resonance, None if there is no resonance

    HISTORY:

       2011-10-09 - Written - Bovy (IAS)

    """
    Pot= flatten(Pot)
    OmegaP= conversion.parse_frequency(OmegaP,**conversion.get_physical(Pot))
    if isinstance(m,str):
        if 'corot' in m.lower():
            corotation= True
        else:
            raise IOError("'m' input not recognized, should be an integer or 'corotation'")
    else:
        corotation= False
    if corotation:
        try:
            out= optimize.brentq(_corotationR_eq,0.0000001,1000.,
                                 args=(Pot,OmegaP,t),**kwargs)
        except ValueError:
            try:
                # Sometimes 0.0000001 is numerically too small to start...
                out= optimize.brentq(_corotationR_eq,0.01,1000.,
                                     args=(Pot,OmegaP,t),**kwargs)
            except ValueError:
                return None
        except RuntimeError: #pragma: no cover 
            raise
        return out
    else:
        try:
            out= optimize.brentq(_lindbladR_eq,0.0000001,1000.,
                                 args=(Pot,OmegaP,m,t),**kwargs)
        except ValueError:
            return None
        except RuntimeError: #pragma: no cover 
            raise
        return out

def _corotationR_eq(R,Pot,OmegaP,t=0.):
    return omegac(Pot,R,t=t,use_physical=False)-OmegaP
def _lindbladR_eq(R,Pot,OmegaP,m,t=0.):
    return m*(omegac(Pot,R,t=t,use_physical=False)-OmegaP)\
        -epifreq(Pot,R,t=t,use_physical=False)

@potential_physical_input
@physical_conversion('frequency',pop=True)
def omegac(Pot,R,t=0.):
    """

    NAME:

       omegac

    PURPOSE:

       calculate the circular angular speed velocity at R in potential Pot

    INPUT:

       Pot - Potential instance or list of such instances

       R - Galactocentric radius (can be Quantity)

       t - time (optional; can be Quantity)

    OUTPUT:

       circular angular speed

    HISTORY:

       2011-10-09 - Written - Bovy (IAS)

    """
    from ..potential import evaluateplanarRforces
    try:
        return numpy.sqrt(-evaluateplanarRforces(Pot,R,t=t,use_physical=False)/R)
    except PotentialError:
        from ..potential import RZToplanarPotential
        Pot= RZToplanarPotential(Pot)
        return numpy.sqrt(-evaluateplanarRforces(Pot,R,t=t,use_physical=False)/R)

def nemo_accname(Pot):
    """
    NAME:
    
       nemo_accname
    
    PURPOSE:
    
       return the accname potential name for use of this potential or list of potentials with NEMO
    
    INPUT:
    
       Pot - Potential instance or list of such instances
       
    OUTPUT:
    
       Acceleration name in the correct format to give to accname=
    
    HISTORY:
    
       2014-12-18 - Written - Bovy (IAS)
    
    """
    Pot= flatten(Pot)
    if isinstance(Pot,list):
        out= ''
        for ii,pot in enumerate(Pot):
            if ii > 0: out+= '+'
            out+= pot.nemo_accname()
        return out
    elif isinstance(Pot,Potential):
        return Pot.nemo_accname()
    else: #pragma: no cover 
        raise PotentialError("Input to 'nemo_accname' is neither a Potential-instance or a list of such instances")
    
def nemo_accpars(Pot,vo,ro):
    """
    NAME:
    
       nemo_accpars
    
    PURPOSE:
    
       return the accpars potential parameters for use of this potential or list of potentials with NEMO
    
    INPUT:
    
       Pot - Potential instance or list of such instances

       vo - velocity unit in km/s
    
       ro - length unit in kpc
    
    OUTPUT:
    
       accpars string in the corrct format to give to accpars
    
    HISTORY:
    
       2014-12-18 - Written - Bovy (IAS)
    
    """
    Pot= flatten(Pot)
    if isinstance(Pot,list):
        out= ''
        for ii,pot in enumerate(Pot):
            if ii > 0: out+= '#'
            out+= pot.nemo_accpars(vo,ro)
        return out
    elif isinstance(Pot,Potential):
        return Pot.nemo_accpars(vo,ro)
    else: #pragma: no cover 
        raise PotentialError("Input to 'nemo_accpars' is neither a Potential-instance or a list of such instances")
    
def to_amuse(Pot,t=0.,tgalpy=0.,reverse=False,ro=None,vo=None): # pragma: no cover
    """
    NAME:
    
       to_amuse

    PURPOSE:

       Return an AMUSE representation of a galpy Potential or list of Potentials

    INPUT:

       Pot - Potential instance or list of such instances

       t= (0.) Initial time in AMUSE (can be in internal galpy units or AMUSE units)

       tgalpy= (0.) Initial time in galpy (can be in internal galpy units or AMUSE units); because AMUSE initial times have to be positive, this is useful to set if the initial time in galpy is negative

       reverse= (False) set whether the galpy potential evolves forwards or backwards in time (default: False); because AMUSE can only integrate forward in time, this is useful to integrate backward in time in AMUSE

       ro= (default taken from Pot) length unit in kpc

       vo= (default taken from Pot) velocity unit in km/s       

    OUTPUT:

       AMUSE representation of Pot

    HISTORY:

       2019-08-04 - Written - Bovy (UofT)

       2019-08-12 - Implemented actual function - Webb (UofT)

    """
    try:
        from . import amuse
    except ImportError:
        raise ImportError("To obtain an AMUSE representation of a galpy potential, you need to have AMUSE installed")
    Pot= flatten(Pot)
    if ro is None or vo is None:
        physical_dict= get_physical(Pot)
        if ro is None: ro= physical_dict.get('ro')
        if vo is None: vo= physical_dict.get('vo')
    return amuse.galpy_profile(Pot,t=t,tgalpy=tgalpy,ro=ro,vo=vo)

def turn_physical_off(Pot):
    """
    NAME:
    
       turn_physical_off

    PURPOSE:

       turn off automatic returning of outputs in physical units

    INPUT:

       (none)

    OUTPUT:

       (none)

    HISTORY:

       2016-01-30 - Written - Bovy (UofT)

    """
    if isinstance(Pot,list):
        for pot in Pot:
            turn_physical_off(pot)
    else:
        Pot.turn_physical_off()
    return None

def turn_physical_on(Pot,ro=None,vo=None):
    """
    NAME:
       
       turn_physical_on

    PURPOSE:
    
       turn on automatic returning of outputs in physical units
    
    INPUT:
    
       ro= reference distance (kpc; can be Quantity)
       
       vo= reference velocity (km/s; can be Quantity)

    OUTPUT:
    
        (none)
    
    HISTORY:
    
        2016-01-30 - Written - Bovy (UofT)
    
    """
    if isinstance(Pot,list):
        for pot in Pot:
            turn_physical_on(pot,ro=ro,vo=vo)
    else:
        Pot.turn_physical_on(ro=ro,vo=vo)
    return None

def _flatten_list(L):
    for item in L:
        try:
            for i in _flatten_list(item): yield i
        except TypeError:
            yield item

def flatten(Pot):
    """
    NAME:
       
       flatten

    PURPOSE:
    
       flatten a possibly nested list of Potential instances into a flat list
    
    INPUT:
    
       Pot - list (possibly nested) of Potential instances

    OUTPUT:
    
       Flattened list of Potential instances 
    
    HISTORY:
    
        2018-03-14 - Written - Bovy (UofT)
    
    """
    if isinstance(Pot, Potential):
        return Pot
    elif isinstance(Pot, list):
        return list(_flatten_list(Pot))
    else:
        return Pot

def _check_c(Pot,dxdv=False,dens=False):
    """

    NAME:

       _check_c

    PURPOSE:

       check whether a potential or list thereof has a C implementation

    INPUT:

       Pot - Potential instance or list of such instances

       dxdv= (False) check whether the potential has dxdv implementation

       dens= (False) check whether the potential has its density implemented in C

    OUTPUT:

       True if a C implementation exists, False otherwise

    HISTORY:

       2014-02-17 - Written - Bovy (IAS)

       2017-07-01 - Generalized to dxdv, added general support for WrapperPotentials, and added support for planarPotentials

    """
    Pot= flatten(Pot)
    from ..potential import planarPotential, linearPotential
    if dxdv: hasC_attr= 'hasC_dxdv'
    elif dens: hasC_attr= 'hasC_dens'
    else: hasC_attr= 'hasC'
    from .WrapperPotential import parentWrapperPotential
    if isinstance(Pot,list):
        return numpy.all(numpy.array([_check_c(p,dxdv=dxdv,dens=dens)
                                      for p in Pot],
                               dtype='bool'))
    elif isinstance(Pot,parentWrapperPotential):
        return bool(Pot.__dict__[hasC_attr]*_check_c(Pot._pot))
    elif isinstance(Pot,Force) or isinstance(Pot,planarPotential) \
            or isinstance(Pot,linearPotential):
        return Pot.__dict__[hasC_attr]

def _dim(Pot):
    """
    NAME:                                                                       
       _dim                                                                                
    PURPOSE:

       Determine the dimensionality of this potential

    INPUT:

       Pot - Potential instance or list of such instances

    OUTPUT:

       Minimum of the dimensionality of all potentials if list; otherwise Pot.dim

    HISTORY:

       2016-04-19 - Written - Bovy (UofT)
    """
    from ..potential import planarPotential, linearPotential
    if isinstance(Pot,list):
        return numpy.amin(numpy.array([_dim(p) for p in Pot],dtype='int'))
    elif isinstance(Pot,(Potential,planarPotential,linearPotential,
                         DissipativeForce)):
        return Pot.dim

def _isNonAxi(Pot):
    """
    NAME:

       _isNonAxi

    PURPOSE:

       Determine whether this potential is non-axisymmetric

    INPUT:

       Pot - Potential instance or list of such instances

    OUTPUT:

       True or False depending on whether the potential is non-axisymmetric (note that some potentials might return True, even though for some parameter values they are axisymmetric)

    HISTORY:

       2016-06-16 - Written - Bovy (UofT)

    """
    isList= isinstance(Pot,list)
    if isList:
        isAxis= [not _isNonAxi(p) for p in Pot]
        nonAxi= not numpy.prod(numpy.array(isAxis))
    else:
        nonAxi= Pot.isNonAxi
    return nonAxi

def kms_to_kpcGyrDecorator(func):
    """Decorator to convert velocities from km/s to kpc/Gyr"""
    @wraps(func)
    def kms_to_kpcGyr_wrapper(*args,**kwargs):
        return func(args[0],velocity_in_kpcGyr(args[1],1.),args[2],**kwargs)
    return kms_to_kpcGyr_wrapper

@potential_physical_input
@physical_conversion('position',pop=True)
def rtide(Pot,R,z,phi=0.,t=0.,M=None):
    """
            
    NAME:
        
        rtide
            
    PURPOSE:
        
        Calculate the tidal radius for object of mass M assuming a circular orbit as

        .. math::

           r_t^3 = \\frac{GM_s}{\\Omega^2-\\mathrm{d}^2\\Phi/\\mathrm{d}r^2}

        where :math:`M_s` is the cluster mass, :math:`\\Omega` is the circular frequency, and :math:`\Phi` is the gravitational potential. For non-spherical potentials, we evaluate :math:`\\Omega^2 = (1/r)(\\mathrm{d}\\Phi/\\mathrm{d}r)` and evaluate the derivatives at the given position of the cluster.

    INPUT:
        
        Pot - Potential instance or list of such instances

        R - Galactocentric radius (can be Quantity)
            
        z - height (can be Quantity)
            
        phi - azimuth (optional; can be Quantity)
            
        t - time (optional; can be Quantity)
        
        M - (default = None) Mass of object (can be Quantity)
            
    OUTPUT:
        
        Tidal Radius
        
    HISTORY:
        
        2018-03-21 - Written - Webb (UofT)
            
    """
    Pot= flatten(Pot)
    if M is None:
        #Make sure an object mass is given
        raise PotentialError("Mass parameter M= needs to be set to compute tidal radius")
    r= numpy.sqrt(R**2.+z**2.)
    omegac2=-evaluaterforces(Pot,R,z,phi=phi,t=t,use_physical=False)/r
    d2phidr2= evaluater2derivs(Pot,R,z,phi=phi,t=t,use_physical=False)
    return (M/(omegac2-d2phidr2))**(1./3.)

@potential_physical_input
@physical_conversion('forcederivative',pop=True)
def ttensor(Pot,R,z,phi=0.,t=0.,eigenval=False):
    """
            
    NAME:
        
        ttensor
            
    PURPOSE:
        
        Calculate the tidal tensor Tij=-d(Psi)(dxidxj)
            
    INPUT:
        
        Pot - Potential instance or list of such instances

        R - Galactocentric radius (can be Quantity)
            
        z - height (can be Quantity)
            
        phi - azimuth (optional; can be Quantity)
            
        t - time (optional; can be Quantity)
            
        eigenval - return eigenvalues if true (optional; boolean)
            
    OUTPUT:
        
        Tidal Tensor
        
    HISTORY:
        
        2018-03-21 - Written - Webb (UofT)
    """
    Pot= flatten(Pot)
    if _isNonAxi(Pot):
        raise PotentialError("Tidal tensor calculation is currently only implemented for axisymmetric potentials")
    #Evaluate forces, angles and derivatives
    Rderiv= -evaluateRforces(Pot,R,z,phi=phi,t=t,use_physical=False)
    phideriv= -evaluatephiforces(Pot,R,z,phi=phi,t=t,use_physical=False)
    R2deriv= evaluateR2derivs(Pot,R,z,phi=phi,t=t,use_physical=False)
    z2deriv= evaluatez2derivs(Pot,R,z,phi=phi,t=t,use_physical=False)
    phi2deriv= evaluatephi2derivs(Pot,R,z,phi=phi,t=t,use_physical=False)
    Rzderiv= evaluateRzderivs(Pot,R,z,phi=phi,t=t,use_physical=False)
    Rphideriv= evaluateRphiderivs(Pot,R,z,phi=phi,t=t,use_physical=False)
    #Temporarily set zphideriv to zero until zphideriv is added to Class
    zphideriv=0.0
    cosphi=numpy.cos(phi)
    sinphi=numpy.sin(phi)
    cos2phi=cosphi**2.0
    sin2phi=sinphi**2.0
    R2=R**2.0
    R3=R**3.0
    # Tidal tensor
    txx= R2deriv*cos2phi-Rphideriv*2.*cosphi*sinphi/R+Rderiv*sin2phi/R\
        +phi2deriv*sin2phi/R2+phideriv*2.*cosphi*sinphi/R2
    tyx= R2deriv*sinphi*cosphi+Rphideriv*(cos2phi-sin2phi)/R\
        -Rderiv*sinphi*cosphi/R-phi2deriv*sinphi*cosphi/R2+phideriv*(sin2phi-cos2phi)/R2
    tzx= Rzderiv*cosphi-zphideriv*sinphi/R
    tyy= R2deriv*sin2phi+Rphideriv*2.*cosphi*sinphi/R+Rderiv*cos2phi/R\
        +phi2deriv*cos2phi/R2-phideriv*2.*sinphi*cosphi/R2
    txy=tyx
    tzy= Rzderiv*sinphi+zphideriv*cosphi/R
    txz= tzx
    tyz= tzy
    tzz=z2deriv
    tij= -numpy.array([[txx,txy,txz],[tyx,tyy,tyz],[tzx,tzy,tzz]])
    if eigenval:
       return numpy.linalg.eigvals(tij)
    else:
       return tij

@physical_conversion('position',pop=True)
def zvc(Pot,R,E,Lz,phi=0.,t=0.):
    """
            
    NAME:
        
        zvc
            
    PURPOSE:
        
        Calculate the zero-velocity curve: z such that Phi(R,z) + Lz/[2R^2] = E (assumes that F_z(R,z) = negative at positive z such that there is a single solution)
            
    INPUT:
        
        Pot - Potential instance or list of such instances

        R - Galactocentric radius (can be Quantity)
            
        E - Energy (can be Quantity)

        Lz - Angular momentum (can be Quantity)
            
        phi - azimuth (optional; can be Quantity)
            
        t - time (optional; can be Quantity)
            
    OUTPUT:
        
        z such that Phi(R,z) + Lz/[2R^2] = E
        
    HISTORY:
        
        2020-08-20 - Written - Bovy (UofT)
    """
    Pot= flatten(Pot)
    R= conversion.parse_length(R,**get_physical(Pot))
    E= conversion.parse_energy(E,**get_physical(Pot))
    Lz= conversion.parse_angmom(Lz,**get_physical(Pot))
    Lz2over2R2= Lz**2./2./R**2.
    # Check z=0 and whether a solution exists
    if numpy.fabs(_evaluatePotentials(Pot,R,0.,phi=phi,t=t)+Lz2over2R2-E) < 1e-8:
        return 0.
    elif _evaluatePotentials(Pot,R,0.,phi=phi,t=t)+Lz2over2R2 > E:
        return numpy.nan # s.t. this does not get plotted
    # Find starting value
    zstart= 1.
    zmax= 1000.
    while E-_evaluatePotentials(Pot,R,zstart,phi=phi,t=t)-Lz2over2R2 > 0. \
          and zstart < zmax:
        zstart*= 2.
    try:
        out= optimize.brentq(\
                lambda z: _evaluatePotentials(Pot,R,z,phi=phi,t=t)+Lz2over2R2-E,
                0.,zstart)
    except ValueError:
        raise ValueError('No solution for the zero-velocity curve found for this combination of parameters')
    return out
    
@physical_conversion('position',pop=True)
def zvc_range(Pot,E,Lz,phi=0.,t=0.):
    """
            
    NAME:
        
        zvc_range
            
    PURPOSE:
        
        Calculate the minimum and maximum radius for which the zero-velocity curve exists for this energy and angular momentum (R such that Phi(R,0) + Lz/[2R^2] = E)
            
    INPUT:
        
        Pot - Potential instance or list of such instances

        E - Energy (can be Quantity)

        Lz - Angular momentum (can be Quantity)
            
        phi - azimuth (optional; can be Quantity)
            
        t - time (optional; can be Quantity)
            
    OUTPUT:
        
        Solutions R such that Phi(R,0) + Lz/[2R^2] = E
        
    HISTORY:
        
        2020-08-20 - Written - Bovy (UofT)
    """
    Pot= flatten(Pot)
    E= conversion.parse_energy(E,**get_physical(Pot))
    Lz= conversion.parse_angmom(Lz,**get_physical(Pot))
    Lz2over2= Lz**2./2.
    # Check whether a solution exists
    RLz= rl(Pot,Lz,t=t,use_physical=False)
    Rstart= RLz
    if _evaluatePotentials(Pot,Rstart,0.,phi=phi,t=t)+Lz2over2/Rstart**2. > E:
        return numpy.array([numpy.nan,numpy.nan])
    # Find starting value for Rmin
    Rstartmin= 1e-8
    while _evaluatePotentials(Pot,Rstart,0,phi=phi,t=t)\
          +Lz2over2/Rstart**2. < E and Rstart > Rstartmin:
        Rstart/= 2.
    Rmin= optimize.brentq(\
                          lambda R: _evaluatePotentials(Pot,R,0,phi=phi,t=t)
                          +Lz2over2/R**2.-E,Rstart,RLz)
    # Find starting value for Rmax
    Rstart= RLz
    Rstartmax= 1000.
    while _evaluatePotentials(Pot,Rstart,0,phi=phi,t=t)\
          +Lz2over2/Rstart**2. < E and Rstart < Rstartmax:
        Rstart*= 2.
    Rmax= optimize.brentq(\
                          lambda R: _evaluatePotentials(Pot,R,0,phi=phi,t=t)
                          +Lz2over2/R**2.-E,RLz,Rstart)
    return numpy.array([Rmin,Rmax])
    
@physical_conversion('position',pop=True)
def rhalf(Pot,t=0.,INF=numpy.inf):
    """
    NAME:

       rhalf

    PURPOSE:

       calculate the half-mass radius, the radius of the spherical shell that contains half the total mass

    INPUT:

       Pot - Potential instance or list thereof

       t= (0.) time (optional; can be Quantity)

       INF= (numpy.inf) radius at which the total mass is calculated (internal units, just set this to something very large)

    OUTPUT:

       half-mass radius

    HISTORY:

       2021-03-18 - Written - Bovy (UofT)

    """
    Pot= flatten(Pot)
    tot_mass= mass(Pot,INF,t=t)
    #Find interval
    rhi= _rhalfFindStart(1.,Pot,tot_mass,t=t)
    rlo= _rhalfFindStart(1.,Pot,tot_mass,t=t,lower=True)
    return optimize.brentq(_rhalffunc,rlo,rhi,
                           args=(Pot,tot_mass,t),
                           maxiter=200,disp=False)

def _rhalffunc(rh,pot,tot_mass,t=0.):
    return mass(pot,rh,t=t)/tot_mass-0.5

def _rhalfFindStart(rh,pot,tot_mass,t=0.,lower=False):
    """find a starting interval for rhalf"""
    rtry= 2.*rh
    while (2.*lower-1.)*_rhalffunc(rtry,pot,tot_mass,t=t) > 0.:
        if lower:
            rtry/= 2.
        else:
            rtry*= 2.
    return rtry

@potential_physical_input
@physical_conversion('time',pop=True)
def tdyn(Pot,R,t=0.):
    """
    NAME:

       tdyn

    PURPOSE:

       calculate the dynamical time from tdyn^2 = 3pi/[G<rho>]

    INPUT:

       Pot - Potential instance or list thereof

       R - Galactocentric radius (can be Quantity)

       t= (0.) time (optional; can be Quantity)

    OUTPUT:

       Dynamical time

    HISTORY:

       2021-03-18 - Written - Bovy (UofT)

    """
    return 2.*numpy.pi*R*numpy.sqrt(R/mass(Pot,R,use_physical=False))
