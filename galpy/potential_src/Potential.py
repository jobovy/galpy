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
import os, os.path
import cPickle as pickle
import math
import numpy as nu
from scipy import optimize, integrate
import galpy.util.bovy_plot as plot
from plotRotcurve import plotRotcurve, vcirc
from plotEscapecurve import plotEscapecurve, _INF
class Potential:
    """Top-level class for a potential"""
    def __init__(self,amp=1.):
        """
        NAME:
           __init__
        PURPOSE:
        INPUT:
           amp - amplitude to be applied when evaluating the potential and its forces
        OUTPUT:
        HISTORY:
        """
        self._amp= amp
        self.dim= 3
        self.isRZ= True
        self.isNonAxi= False
        self.hasC= False
        self.hasC_dxdv= False
        return None

    def __call__(self,R,z,phi=0.,t=0.,dR=0,dphi=0):
        """
        NAME:
           __call__
        PURPOSE:
           evaluate the potential at (R,z,phi,t)
        INPUT:
           R - Cylindrical Galactocentric radius

           z - vertical height

           phi - azimuth (optional)

           t - time (optional)

           dR= dphi=, if set to non-zero integers, return the dR, dphi't derivative instead
           
        OUTPUT:
           Phi(R,z,t)
        HISTORY:
           2010-04-16 - Written - Bovy (NYU)
        """
        if dR == 0 and dphi == 0:
            try:
                rawOut= self._evaluate(R,z,phi=phi,t=t)
            except AttributeError: #pragma: no cover
                raise PotentialError("'_evaluate' function not implemented for this potential")
            if rawOut is None: return rawOut
            else: return self._amp*rawOut
        elif dR == 1 and dphi == 0:
            return -self.Rforce(R,z,phi=phi,t=t)
        elif dR == 0 and dphi == 1:
            return -self.phiforce(R,z,phi=phi,t=t)
        elif dR == 2 and dphi == 0:
            return self.R2deriv(R,z,phi=phi,t=t)
        elif dR == 0 and dphi == 2:
            return self.phi2deriv(R,z,phi=phi,t=t)
        elif dR == 1 and dphi == 1:
            return self.Rphideriv(R,z,phi=phi,t=t)           
        elif dR != 0 or dphi != 0:
            raise NotImplementedError('Higher-order derivatives not implemented for this potential')
        
    def Rforce(self,R,z,phi=0.,t=0.):
        """
        NAME:

           Rforce

        PURPOSE:

           evaluate radial force F_R  (R,z)

        INPUT:

           R - Cylindrical Galactocentric radius

           z - vertical height

           phi - azimuth (optional)

           t - time (optional)

        OUTPUT:

           F_R (R,z,phi,t)

        HISTORY:

           2010-04-16 - Written - Bovy (NYU)

        """
        try:
            return self._amp*self._Rforce(R,z,phi=phi,t=t)
        except AttributeError: #pragma: no cover
            raise PotentialError("'_Rforce' function not implemented for this potential")
        
    def zforce(self,R,z,phi=0.,t=0.):
        """
        NAME:

           zforce

        PURPOSE:

           evaluate the vertical force F_z  (R,z,t)

        INPUT:

           R - Cylindrical Galactocentric radius

           z - vertical height

           phi - azimuth (optional)

           t - time (optional)

        OUTPUT:

           F_z (R,z,phi,t)

        HISTORY:

           2010-04-16 - Written - Bovy (NYU)

        """
        try:
            return self._amp*self._zforce(R,z,phi=phi,t=t)
        except AttributeError: #pragma: no cover
            raise PotentialError("'_zforce' function not implemented for this potential")

    def dens(self,R,z,phi=0.,t=0.,forcepoisson=False):
        """
        NAME:

           dens

        PURPOSE:

           evaluate the density rho(R,z,t)

        INPUT:

           R - Cylindrical Galactocentric radius

           z - vertical height

           phi - azimuth (optional)

           t - time (optional)

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
            return (-self.Rforce(R,z,phi=phi,t=t)/R
                     +self.R2deriv(R,z,phi=phi,t=t)
                     +self.phi2deriv(R,z,phi=phi,t=t)/R**2.
                     +self.z2deriv(R,z,phi=phi,t=t))/4./nu.pi

    def mass(self,R,z=None,t=0.,forceint=False):
        """
        NAME:

           mass

        PURPOSE:

           evaluate the mass enclosed

        INPUT:

           R - Cylindrical Galactocentric radius

           z= (None) vertical height

           t - time (optional)

        KEYWORDS:

           forceint= if True, calculate the mass through integration of the density, even if an explicit expression for the mass exists

        OUTPUT:

           1) for spherical potentials: M(<R) [or if z is None], when the mass is implemented explicitly, the mass enclosed within  r = sqrt(R^2+z^2) is returned when not z is None; forceint will integrate between -z and z, so the two are inconsistent (If you care to have this changed, raise an issue on github)

           2) for axisymmetric potentials: M(<R,<|Z|)

        HISTORY:

           2014-01-29 - Written - Bovy (IAS)

        """
        if self.isNonAxi:
            raise NotImplementedError('mass for non-axisymmetric potentials is not currently supported')
        try:
            if forceint: raise AttributeError #Hack!
            return self._amp*self._mass(R,z=z,t=t)
        except AttributeError:
            #Use numerical integration to get the mass
            if z is None:
                return 4.*nu.pi\
                    *integrate.quad(lambda x: x**2.*self.dens(x,0.,),
                                    0.,R)[0]
            else:
                return 4.*nu.pi\
                    *integrate.dblquad(lambda y,x: x*self.dens(x,y),
                                       0.,R,lambda x: 0., lambda x: z)[0]

    def mvir(self,vo,ro,H=70.,Om=0.3,overdens=200.,wrtcrit=False,
             forceint=False):
        """
        NAME:

           mvir

        PURPOSE:

           calculate the virial mass

        INPUT:

           vo - velocity unit in km/s

           ro - length unit in kpc

           H= (default: 70) Hubble constant in km/s/Mpc
           
           Om= (default: 0.3) Omega matter
       
           overdens= (200) overdensity which defines the virial radius

           wrtcrit= (False) if True, the overdensity is wrt the critical density rather than the mean matter density
           
        KEYWORDS:

           forceint= if True, calculate the mass through integration of the density, even if an explicit expression for the mass exists

        OUTPUT:

           M(<rvir)

        HISTORY:

           2014-09-12 - Written - Bovy (IAS)

        """
        #Evaluate the virial radius
        try:
            rvir= self.rvir(vo,ro,H=H,Om=Om,overdens=overdens,wrtcrit=wrtcrit)
        except AttributeError:
            raise AttributeError("This potential does not have a '_scale' defined to base the concentration on or does not support calculating the virial radius")
        return self.mass(rvir,forceint=forceint)

    def R2deriv(self,R,Z,phi=0.,t=0.):
        """
        NAME:

           R2deriv

        PURPOSE:

           evaluate the second radial derivative

        INPUT:

           R - Galactocentric radius

           Z - vertical height

           phi - Galactocentric azimuth

           t - time

        OUTPUT:

           d2phi/dR2

        HISTORY:

           2011-10-09 - Written - Bovy (IAS)

        """
        try:
            return self._amp*self._R2deriv(R,Z,phi=phi,t=t)
        except AttributeError: #pragma: no cover
            raise PotentialError("'_R2deriv' function not implemented for this potential")      

    def z2deriv(self,R,Z,phi=0.,t=0.):
        """
        NAME:

           z2deriv

        PURPOSE:

           evaluate the second vertical derivative

        INPUT:

           R - Galactocentric radius

           Z - vertical height

           phi - Galactocentric azimuth

           t - time

        OUTPUT:

           d2phi/dz2

        HISTORY:

           2012-07-25 - Written - Bovy (IAS@MPIA)

        """
        try:
            return self._amp*self._z2deriv(R,Z,phi=phi,t=t)
        except AttributeError: #pragma: no cover
            raise PotentialError("'_z2deriv' function not implemented for this potential")      

    def Rzderiv(self,R,Z,phi=0.,t=0.):
        """
        NAME:

           Rzderiv

        PURPOSE:

           evaluate the mixed R,z derivative

        INPUT:

           R - Galactocentric radius

           Z - vertical height

           phi - Galactocentric azimuth

           t - time

        OUTPUT:

           d2phi/dz/dR

        HISTORY:

           2013-08-26 - Written - Bovy (IAS)

        """
        try:
            return self._amp*self._Rzderiv(R,Z,phi=phi,t=t)
        except AttributeError: #pragma: no cover
            raise PotentialError("'_Rzderiv' function not implemented for this potential")      

    def normalize(self,norm,t=0.):
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
        self._amp*= norm/nu.fabs(self.Rforce(1.,0.,t=t))

    def phiforce(self,R,z,phi=0.,t=0.):
        """
        NAME:

           phiforce

        PURPOSE:

           evaluate the azimuthal force F_phi  (R,z,phi,t)

        INPUT:

           R - Cylindrical Galactocentric radius

           z - vertical height

           phi - azimuth (rad)

           t - time (optional)

        OUTPUT:

           F_phi (R,z,phi,t)

        HISTORY:

           2010-07-10 - Written - Bovy (NYU)

        """
        try:
            return self._amp*self._phiforce(R,z,phi=phi,t=t)
        except AttributeError: #pragma: no cover
            return 0.

    def phi2deriv(self,R,Z,phi=0.,t=0.):
        """
        NAME:

           phi2deriv

        PURPOSE:

           evaluate the second azimuthal derivative

        INPUT:

           R - Galactocentric radius

           Z - vertical height

           phi - Galactocentric azimuth

           t - time

        OUTPUT:

           d2Phi/dphi2

        HISTORY:

           2013-09-24 - Written - Bovy (IAS)

        """
        try:
            return self._amp*self._phi2deriv(R,Z,phi=phi,t=t)
        except AttributeError: #pragma: no cover
            return 0.

    def Rphideriv(self,R,Z,phi=0.,t=0.):
        """
        NAME:

           Rphideriv

        PURPOSE:

           evaluate the mixed radial, azimuthal derivative

        INPUT:

           R - Galactocentric radius

           Z - vertical height

           phi - Galactocentric azimuth

           t - time

        OUTPUT:

           d2Phi/dphidR

        HISTORY:

           2014-06-30 - Written - Bovy (IAS)

        """
        try:
            return self._amp*self._Rphideriv(R,Z,phi=phi,t=t)
        except AttributeError: #pragma: no cover
            return 0.

    def _phiforce(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _phiforce
        PURPOSE:
           evaluate the azimuthal force F_phi  (R,z,phi,t)
        INPUT:
           R - Cylindrical Galactocentric radius
           z - vertical height
           phi - azimuth (rad)
           t - time (optional)
        OUTPUT:
           F_phi (R,z,phi,t)
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        return 0. #default is to assume axisymmetry

    def _phi2deriv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _phi2deriv
        PURPOSE:
           evaluate the azimuthal second derivative of the potential
        INPUT:
           R - Cylindrical Galactocentric radius
           z - vertical height
           phi - azimuth (rad)
           t - time (optional)
        OUTPUT:
           d2Phi/dphi2
        HISTORY:
           2013-09-24 - Written - Bovy (NYU)
        """
        return 0. #default is to assume axisymmetry

    def _Rphideriv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _Rphideriv
        PURPOSE:
           evaluate the mixed radial and azimuthal derivative of the potential
        INPUT:
           R - Cylindrical Galactocentric radius
           z - vertical height
           phi - azimuth (rad)
           t - time (optional)
        OUTPUT:
           d2Phi/dphidR
        HISTORY:
           2014-06-30 - Written - Bovy (IAS)
        """
        return 0. #default is to assume axisymmetry

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
        HISTORY
        """
        from planarPotential import RZToplanarPotential
        return RZToplanarPotential(self)

    def toVertical(self,R):
        """
        NAME:
           toVertical
        PURPOSE:
           convert a 3D potential into a linear (vertical) potential at R
        INPUT:
           R - Galactocentric radius at which to create the vertical potential
        OUTPUT:
           linear (vertical) potential
        HISTORY
        """
        from verticalPotential import RZToverticalPotential
        return RZToverticalPotential(self,R)

    def plot(self,t=0.,rmin=0.,rmax=1.5,nrs=21,zmin=-0.5,zmax=0.5,nzs=21,
             effective=False,Lz=None,
             xrange=None,yrange=None,
             justcontours=False,
             ncontours=21,savefilename=None):
        """
        NAME:

           plot

        PURPOSE:

           plot the potential

        INPUT:

           t= time tp plot potential at

           rmin= minimum R at which to calculate

           rmax= maximum R

           nrs= grid in R

           zmin= minimum z

           zmax= maximum z

           nzs= grid in z

           effective= (False) if True, plot the effective potential Phi + Lz^2/2/R^2

           Lz= (None) angular momentum to use for the effective potential when effective=True

           ncontours - number of contours

           justcontours= (False) if True, just plot contours

           savefilename - save to or restore from this savefile (pickle)

           xrange, yrange= can be specified independently from rmin,zmin, etc.

        OUTPUT:

           plot to output device

        HISTORY:

           2010-07-09 - Written - Bovy (NYU)

           2014-04-08 - Added effective= - Bovy (IAS)

        """
        if xrange is None: xrange= [rmin,rmax]
        if yrange is None: yrange= [zmin,zmax]
        if not savefilename is None and os.path.exists(savefilename):
            print "Restoring savefile "+savefilename+" ..."
            savefile= open(savefilename,'rb')
            potRz= pickle.load(savefile)
            Rs= pickle.load(savefile)
            zs= pickle.load(savefile)
            savefile.close()
        else:
            if effective and Lz is None:
                raise RuntimeError("When effective=True, you need to specify Lz=")
            Rs= nu.linspace(xrange[0],xrange[1],nrs)
            zs= nu.linspace(yrange[0],yrange[1],nzs)
            potRz= nu.zeros((nrs,nzs))
            for ii in range(nrs):
                for jj in range(nzs):
                    potRz[ii,jj]= self._evaluate(Rs[ii],zs[jj],t=t)
                if effective:
                    potRz[ii,:]+= 0.5*Lz**2/Rs[ii]**2.
            #Don't plot outside of the desired range
            potRz[Rs < rmin,:]= nu.nan
            potRz[Rs > rmax,:]= nu.nan
            potRz[:,zs < zmin]= nu.nan
            potRz[:,zs > zmax]= nu.nan
            if not savefilename == None:
                print "Writing savefile "+savefilename+" ..."
                savefile= open(savefilename,'wb')
                pickle.dump(potRz,savefile)
                pickle.dump(Rs,savefile)
                pickle.dump(zs,savefile)
                savefile.close()
        return plot.bovy_dens2d(potRz.T,origin='lower',cmap='gist_gray',contours=True,
                                xlabel=r"$R/R_0$",ylabel=r"$z/R_0$",
                                xrange=xrange,
                                yrange=yrange,
                                aspect=.75*(rmax-rmin)/(zmax-zmin),
                                cntrls='-',
                                justcontours=justcontours,
                                levels=nu.linspace(nu.nanmin(potRz),nu.nanmax(potRz),
                                                   ncontours))
        
    def plotDensity(self,rmin=0.,rmax=1.5,nrs=21,zmin=-0.5,zmax=0.5,nzs=21,
                    ncontours=21,savefilename=None,aspect=None,log=False,
                    justcontours=False):
        """
        NAME:
           plotDensity
        PURPOSE:
           plot the density of this potential
        INPUT:

           rmin= minimum R

           rmax= maximum R

           nrs= grid in R

           zmin= minimum z

           zmax= maximum z

           nzs= grid in z

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
                             zmin=zmin,zmax=zmax,nzs=nzs,
                             ncontours=ncontours,savefilename=savefilename,
                             justcontours=justcontours,
                             aspect=aspect,log=log)

    def vcirc(self,R):
        """
        
        NAME:
        
            vcirc
        
        PURPOSE:
        
            calculate the circular velocity at R in this potential

        INPUT:
        
            R - Galactocentric radius
        
        OUTPUT:
        
            circular rotation velocity
        
        HISTORY:
        
            2011-10-09 - Written - Bovy (IAS)
        
        """
        return nu.sqrt(R*-self.Rforce(R,0.))

    def dvcircdR(self,R):
        """
        
        NAME:
        
            dvcircdR
        
        PURPOSE:
        
            calculate the derivative of the circular velocity at R wrt R
            in this potential

        INPUT:
        
            R - Galactocentric radius
        
        OUTPUT:
        
            derivative of the circular rotation velocity wrt R
        
        HISTORY:
        
            2013-01-08 - Written - Bovy (IAS)
        
        """
        return 0.5*(-self.Rforce(R,0.)+R*self.R2deriv(R,0.))/self.vcirc(R)

    def omegac(self,R):
        """
        
        NAME:
        
            omegac
        
        PURPOSE:
        
            calculate the circular angular speed at R in this potential

        INPUT:
        
            R - Galactocentric radius
        
        OUTPUT:
        
            circular angular speed
        
        HISTORY:
        
            2011-10-09 - Written - Bovy (IAS)
        
        """
        return nu.sqrt(-self.Rforce(R,0.)/R)

    def epifreq(self,R):
        """
        
        NAME:
        
           epifreq
        
        PURPOSE:
        
           calculate the epicycle frequency at R in this potential
        
        INPUT:
        
           R - Galactocentric radius
        
        OUTPUT:
        
           epicycle frequency
        
        HISTORY:
        
           2011-10-09 - Written - Bovy (IAS)
        
        """
        return nu.sqrt(self.R2deriv(R,0.)-3./R*self.Rforce(R,0.))

    def verticalfreq(self,R):
        """
        
        NAME:
        
           verticalfreq
        
        PURPOSE:
        
           calculate the vertical frequency at R in this potential
        
        INPUT:
        
           R - Galactocentric radius
        
        OUTPUT:
        
           vertical frequency
        
        HISTORY:
        
           2012-07-25 - Written - Bovy (IAS@MPIA)
        
        """
        return nu.sqrt(self.z2deriv(R,0.))

    def lindbladR(self,OmegaP,m=2,**kwargs):
        """
        
        NAME:
        
           lindbladR
        
        PURPOSE:
        
            calculate the radius of a Lindblad resonance
        
        INPUT:
        
           OmegaP - pattern speed

           m= order of the resonance (as in m(O-Op)=kappa (negative m for outer)
              use m='corotation' for corotation
              +scipy.optimize.brentq xtol,rtol,maxiter kwargs
        
        OUTPUT:
        
           radius of Linblad resonance, None if there is no resonance
        
        HISTORY:
        
           2011-10-09 - Written - Bovy (IAS)
        
        """
        return lindbladR(self,OmegaP,m=m,**kwargs)

    def vesc(self,R):
        """

        NAME:

            vesc

        PURPOSE:

            calculate the escape velocity at R for this potential

        INPUT:

            R - Galactocentric radius

        OUTPUT:

            escape velocity

        HISTORY:

            2011-10-09 - Written - Bovy (IAS)

        """
        return nu.sqrt(2.*(self(_INF,0.)-self(R,0.)))
        
    def rl(self,lz):
        """
        NAME:
        
            rl
        
        PURPOSE:
        
            calculate the radius of a circular orbit of Lz
        
        INPUT:
        
        
            lz - Angular momentum
        
        OUTPUT:
        
            radius
        
        HISTORY:
        
            2012-07-30 - Written - Bovy (IAS@MPIA)
        
        NOTE:
        
            seems to take about ~0.5 ms for a Miyamoto-Nagai potential; 
            ~0.75 ms for a MWPotential
        
        """
        return rl(self,lz)

    def flattening(self,R,z):
        """
        
        NAME:
        
           flattening
        
        PURPOSE:
        
           calculate the potential flattening, defined as sqrt(|z/R F_R/F_z|)
        
        INPUT:
        
           R - Galactocentric radius

           z - height
        
        OUTPUT:
        
           flattening
        
        HISTORY:
        
           2012-09-13 - Written - Bovy (IAS)
        
        """
        return nu.sqrt(nu.fabs(z/R*self.Rforce(R,z)/self.zforce(R,z)))

    def vterm(self,l,deg=True):
        """
        
        NAME:
        
            vterm
        
        PURPOSE:
        
            calculate the terminal velocity at l in this potential

        INPUT:
        
            l - Galactic longitude [deg/rad]

            deg= if True (default), l in deg
        
        OUTPUT:
        
            terminal velocity
        
        HISTORY:
        
            2013-05-31 - Written - Bovy (IAS)
        
        """
        if deg:
            sinl= nu.sin(l/180.*nu.pi)
        else:
            sinl= nu.sin(l)
        return sinl*(self.omegac(sinl)-self.omegac(1.))

    def plotRotcurve(self,*args,**kwargs):
        """
        NAME:

           plotRotcurve

        PURPOSE:

           plot the rotation curve for this potential (in the z=0 plane for
           non-spherical potentials)

        INPUT:

           Rrange - range

           grid= number of points to plot

           savefilename=- save to or restore from this savefile (pickle)

           +bovy_plot(*args,**kwargs)

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

           Rrange - range

           grid= number of points to plot

           savefilename= save to or restore from this savefile (pickle)

           +bovy_plot(*args,**kwargs)

        OUTPUT:

           plot to output device

        HISTORY:

           2010-08-08 - Written - Bovy (NYU)

        """
        return plotEscapecurve(self.toPlanar(),*args,**kwargs)

    def conc(self,vo,ro,H=70.,Om=0.3,overdens=200.,wrtcrit=False):
        """
        NAME:

           conc

        PURPOSE:

           return the concentration

        INPUT:

           vo - velocity unit in km/s

           ro - length unit in kpc

           H= (default: 70) Hubble constant in km/s/Mpc
           
           Om= (default: 0.3) Omega matter
       
           overdens= (200) overdensity which defines the virial radius

           wrtcrit= (False) if True, the overdensity is wrt the critical density rather than the mean matter density
           
        OUTPUT:

           concentration (scale/rvir)

        HISTORY:

           2014-04-03 - Written - Bovy (IAS)

        """
        try:
            return self.rvir(vo,ro,H=H,Om=Om,overdens=overdens,wrtcrit=wrtcrit)/self._scale
        except AttributeError:
            raise AttributeError("This potential does not have a '_scale' defined to base the concentration on or does not support calculating the virial radius")

class PotentialError(Exception): #pragma: no cover
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def evaluatePotentials(R,z,Pot,phi=0.,t=0.,dR=0,dphi=0):
    """
    NAME:
       evaluatePotentials
    PURPOSE:
       convenience function to evaluate a possible sum of potentials
    INPUT:
       R - cylindrical Galactocentric distance

       z - distance above the plane

       Pot - potential or list of potentials

       phi - azimuth

       t - time

       dR= dphi=, if set to non-zero integers, return the dR, dphi't derivative instead
    OUTPUT:
       Phi(R,z)
    HISTORY:
       2010-04-16 - Written - Bovy (NYU)
    """
    if isinstance(Pot,list):
        sum= 0.
        for pot in Pot:
            sum+= pot(R,z,phi=phi,t=t,dR=dR,dphi=dphi)
        return sum
    elif isinstance(Pot,Potential):
        return Pot(R,z,phi=phi,t=t,dR=dR,dphi=dphi)
    else: #pragma: no cover 
        raise PotentialError("Input to 'evaluatePotentials' is neither a Potential-instance or a list of such instances")

def evaluateDensities(R,z,Pot,phi=0.,t=0.,forcepoisson=False):
    """
    NAME:

       evaluateDensities

    PURPOSE:

       convenience function to evaluate a possible sum of densities

    INPUT:

       R - cylindrical Galactocentric distance

       z - distance above the plane

       Pot - potential or list of potentials

       phi - azimuth

       t - time

       forcepoisson= if True, calculate the density through the Poisson equation, even if an explicit expression for the density exists

    OUTPUT:

       rho(R,z)

    HISTORY:

       2010-08-08 - Written - Bovy (NYU)

       2013-12-28 - Added forcepoisson - Bovy (IAS)

    """
    if isinstance(Pot,list):
        sum= 0.
        for pot in Pot:
            sum+= pot.dens(R,z,phi=phi,t=t,forcepoisson=forcepoisson)
        return sum
    elif isinstance(Pot,Potential):
        return Pot.dens(R,z,phi=phi,t=t,forcepoisson=forcepoisson)
    else: #pragma: no cover 
        raise PotentialError("Input to 'evaluateDensities' is neither a Potential-instance or a list of such instances")

def evaluateRforces(R,z,Pot,phi=0.,t=0.):
    """
    NAME:
       evaluateRforce
    PURPOSE:
       convenience function to evaluate a possible sum of potentials
    INPUT:
       R - cylindrical Galactocentric distance

       z - distance above the plane

       Pot - a potential or list of potentials

       phi - azimuth (optional)

       t - time (optional)
    OUTPUT:
       F_R(R,z,phi,t)
    HISTORY:
       2010-04-16 - Written - Bovy (NYU)
    """
    if isinstance(Pot,list):
        sum= 0.
        for pot in Pot:
            sum+= pot.Rforce(R,z,phi=phi,t=t)
        return sum
    elif isinstance(Pot,Potential):
        return Pot.Rforce(R,z,phi=phi,t=t)
    else: #pragma: no cover 
        raise PotentialError("Input to 'evaluateRforces' is neither a Potential-instance or a list of such instances")

def evaluatephiforces(R,z,Pot,phi=0.,t=0.):
    """
    NAME:

       evaluatephiforces

    PURPOSE:

       convenience function to evaluate a possible sum of potentials

    INPUT:
       R - cylindrical Galactocentric distance

       z - distance above the plane

       Pot - a potential or list of potentials

       phi - azimuth (optional)

       t - time (optional)

    OUTPUT:

       F_phi(R,z,phi,t)

    HISTORY:

       2010-04-16 - Written - Bovy (NYU)

    """
    if isinstance(Pot,list):
        sum= 0.
        for pot in Pot:
            sum+= pot.phiforce(R,z,phi=phi,t=t)
        return sum
    elif isinstance(Pot,Potential):
        return Pot.phiforce(R,z,phi=phi,t=t)
    else: #pragma: no cover 
        raise PotentialError("Input to 'evaluatephiforces' is neither a Potential-instance or a list of such instances")

def evaluatezforces(R,z,Pot,phi=0.,t=0.):
    """
    NAME:

       evaluatezforces

    PURPOSE:

       convenience function to evaluate a possible sum of potentials

    INPUT:

       R - cylindrical Galactocentric distance

       z - distance above the plane

       Pot - a potential or list of potentials

       phi - azimuth (optional)

       t - time (optional)

    OUTPUT:

       F_z(R,z,phi,t)

    HISTORY:

       2010-04-16 - Written - Bovy (NYU)

    """
    if isinstance(Pot,list):
        sum= 0.
        for pot in Pot:
            sum+= pot.zforce(R,z,phi=phi,t=t)
        return sum
    elif isinstance(Pot,Potential):
        return Pot.zforce(R,z,phi=phi,t=t)
    else: #pragma: no cover 
        raise PotentialError("Input to 'evaluatezforces' is neither a Potential-instance or a list of such instances")

def evaluateR2derivs(R,z,Pot,phi=0.,t=0.):
    """
    NAME:
       evaluateR2derivs
    PURPOSE:
       convenience function to evaluate a possible sum of potentials
    INPUT:
       R - cylindrical Galactocentric distance

       z - distance above the plane

       Pot - a potential or list of potentials

       phi - azimuth (optional)

       t - time (optional)
    OUTPUT:
       d2Phi/d2R(R,z,phi,t)
    HISTORY:
       2012-07-25 - Written - Bovy (IAS)
    """
    if isinstance(Pot,list):
        sum= 0.
        for pot in Pot:
            sum+= pot.R2deriv(R,z,phi=phi,t=t)
        return sum
    elif isinstance(Pot,Potential):
        return Pot.R2deriv(R,z,phi=phi,t=t)
    else: #pragma: no cover 
        raise PotentialError("Input to 'evaluateR2derivs' is neither a Potential-instance or a list of such instances")

def evaluatez2derivs(R,z,Pot,phi=0.,t=0.):
    """
    NAME:
       evaluatez2derivs
    PURPOSE:
       convenience function to evaluate a possible sum of potentials
    INPUT:
       R - cylindrical Galactocentric distance

       z - distance above the plane

       Pot - a potential or list of potentials

       phi - azimuth (optional)

       t - time (optional)
    OUTPUT:
       d2Phi/d2z(R,z,phi,t)
    HISTORY:
       2012-07-25 - Written - Bovy (IAS)
    """
    if isinstance(Pot,list):
        sum= 0.
        for pot in Pot:
            sum+= pot.z2deriv(R,z,phi=phi,t=t)
        return sum
    elif isinstance(Pot,Potential):
        return Pot.z2deriv(R,z,phi=phi,t=t)
    else: #pragma: no cover 
        raise PotentialError("Input to 'evaluatez2derivs' is neither a Potential-instance or a list of such instances")

def evaluateRzderivs(R,z,Pot,phi=0.,t=0.):
    """
    NAME:
       evaluateRzderivs
    PURPOSE:
       convenience function to evaluate a possible sum of potentials
    INPUT:
       R - cylindrical Galactocentric distance

       z - distance above the plane

       Pot - a potential or list of potentials

       phi - azimuth (optional)

       t - time (optional)
    OUTPUT:
       d2Phi/dz/dR(R,z,phi,t)
    HISTORY:
       2013-08-28 - Written - Bovy (IAS)
    """
    if isinstance(Pot,list):
        sum= 0.
        for pot in Pot:
            sum+= pot.Rzderiv(R,z,phi=phi,t=t)
        return sum
    elif isinstance(Pot,Potential):
        return Pot.Rzderiv(R,z,phi=phi,t=t)
    else: #pragma: no cover 
        raise PotentialError("Input to 'evaluateRzderivs' is neither a Potential-instance or a list of such instances")

def plotPotentials(Pot,rmin=0.,rmax=1.5,nrs=21,zmin=-0.5,zmax=0.5,nzs=21,
                   ncontours=21,savefilename=None,aspect=None,
                   justcontours=False):
        """
        NAME:

           plotPotentials

        PURPOSE:

           plot a set of potentials

        INPUT:

           Pot - Potential or list of Potential instances

           rmin= minimum R

           rmax= maximum R

           nrs= grid in R

           zmin= minimum z

           zmax= maximum z

           nzs= grid in z

           ncontours= number of contours

           justcontours= (False) if True, just plot contours

           savefilename= save to or restore from this savefile (pickle)

        OUTPUT:

           plot to output device

        HISTORY:

           2010-07-09 - Written - Bovy (NYU)

        """
        if not savefilename == None and os.path.exists(savefilename):
            print "Restoring savefile "+savefilename+" ..."
            savefile= open(savefilename,'rb')
            potRz= pickle.load(savefile)
            Rs= pickle.load(savefile)
            zs= pickle.load(savefile)
            savefile.close()
        else:
            Rs= nu.linspace(rmin,rmax,nrs)
            zs= nu.linspace(zmin,zmax,nzs)
            potRz= nu.zeros((nrs,nzs))
            for ii in range(nrs):
                for jj in range(nzs):
                    potRz[ii,jj]= evaluatePotentials(nu.fabs(Rs[ii]),
                                                     zs[jj],Pot)
            if not savefilename == None:
                print "Writing savefile "+savefilename+" ..."
                savefile= open(savefilename,'wb')
                pickle.dump(potRz,savefile)
                pickle.dump(Rs,savefile)
                pickle.dump(zs,savefile)
                savefile.close()
        if aspect is None:
            aspect=.75*(rmax-rmin)/(zmax-zmin)
        return plot.bovy_dens2d(potRz.T,origin='lower',cmap='gist_gray',contours=True,
                                xlabel=r"$R/R_0$",ylabel=r"$z/R_0$",
                                aspect=aspect,
                                xrange=[rmin,rmax],
                                yrange=[zmin,zmax],
                                cntrls='-',
                                justcontours=justcontours,
                                levels=nu.linspace(nu.nanmin(potRz),nu.nanmax(potRz),
                                                   ncontours))

def plotDensities(Pot,rmin=0.,rmax=1.5,nrs=21,zmin=-0.5,zmax=0.5,nzs=21,
                  ncontours=21,savefilename=None,aspect=None,log=False,
                  justcontours=False):
        """
        NAME:

           plotDensities

        PURPOSE:

           plot the density a set of potentials

        INPUT:

           Pot - Potential or list of Potential instances

           rmin= minimum R

           rmax= maximum R

           nrs= grid in R

           zmin= minimum z

           zmax= maximum z

           nzs= grid in z

           ncontours= number of contours

           justcontours= (False) if True, just plot contours

           savefilename= save to or restore from this savefile (pickle)

           log= if True, plot the log density
        OUTPUT:
           plot to output device
        HISTORY:
           2013-07-05 - Written - Bovy (IAS)
        """
        if not savefilename == None and os.path.exists(savefilename):
            print "Restoring savefile "+savefilename+" ..."
            savefile= open(savefilename,'rb')
            potRz= pickle.load(savefile)
            Rs= pickle.load(savefile)
            zs= pickle.load(savefile)
            savefile.close()
        else:
            Rs= nu.linspace(rmin,rmax,nrs)
            zs= nu.linspace(zmin,zmax,nzs)
            potRz= nu.zeros((nrs,nzs))
            for ii in range(nrs):
                for jj in range(nzs):
                    potRz[ii,jj]= evaluateDensities(nu.fabs(Rs[ii]),
                                                    zs[jj],Pot)
            if not savefilename == None:
                print "Writing savefile "+savefilename+" ..."
                savefile= open(savefilename,'wb')
                pickle.dump(potRz,savefile)
                pickle.dump(Rs,savefile)
                pickle.dump(zs,savefile)
                savefile.close()
        if aspect is None:
            aspect=.75*(rmax-rmin)/(zmax-zmin)
        if log:
            potRz= nu.log(potRz)
        return plot.bovy_dens2d(potRz.T,origin='lower',cmap='gist_yarg',contours=True,
                                xlabel=r"$R/R_0$",ylabel=r"$z/R_0$",
                                aspect=aspect,
                                xrange=[rmin,rmax],
                                yrange=[zmin,zmax],
                                cntrls='-',
                                justcontours=justcontours,
                                levels=nu.linspace(nu.nanmin(potRz),nu.nanmax(potRz),
                                                   ncontours))

def epifreq(Pot,R):
    """
    
    NAME:
    
        epifreq
    
    PURPOSE:
    
        calculate the epicycle frequency at R in the potential Pot
    
    INPUT:

        Pot - Potential instance or list thereof
    
        R - Galactocentric radius
    
    OUTPUT:
    
        epicycle frequency
    
    HISTORY:
    
        2012-07-25 - Written - Bovy (IAS)
    
    """
    from galpy.potential_src.planarPotential import planarPotential
    if isinstance(Pot,(Potential,planarPotential)):
        return Pot.epifreq(R)
    from planarPotential import evaluateplanarRforces, evaluateplanarR2derivs
    from Potential import PotentialError
    try:
        return nu.sqrt(evaluateplanarR2derivs(R,Pot)-3./R*evaluateplanarRforces(R,Pot))
    except PotentialError:
        from planarPotential import RZToplanarPotential
        Pot= RZToplanarPotential(Pot)
        return nu.sqrt(evaluateplanarR2derivs(R,Pot)-3./R*evaluateplanarRforces(R,Pot))

def verticalfreq(Pot,R):
    """
    
    NAME:
    
       verticalfreq
        
    PURPOSE:
    
        calculate the vertical frequency at R in the potential Pot
    
    INPUT:

       Pot - Potential instance or list thereof
    
       R - Galactocentric radius
    
    OUTPUT:
    
        vertical frequency
    
    HISTORY:
    
        2012-07-25 - Written - Bovy (IAS@MPIA)
    
    """
    from galpy.potential_src.planarPotential import planarPotential
    if isinstance(Pot,(Potential,planarPotential)):
        return Pot.verticalfreq(R)
    return nu.sqrt(evaluatez2derivs(R,0.,Pot))

def flattening(Pot,R,z):
    """
    
    NAME:
    
        flattening
    
    PURPOSE:
    
       calculate the potential flattening, defined as sqrt(|z/R F_R/F_z|)
    
    INPUT:

        Pot - Potential instance or list thereof
    
        R - Galactocentric radius
        
        z - height
    
    OUTPUT:
    
        flattening
    
    HISTORY:
    
        2012-09-13 - Written - Bovy (IAS)
    
    """
    return nu.sqrt(nu.fabs(z/R*evaluateRforces(R,z,Pot)/evaluatezforces(R,z,Pot)))

def vterm(Pot,l,deg=True):
    """
    
    NAME:
    
        vterm
        
    PURPOSE:
    
        calculate the terminal velocity at l in this potential

    INPUT:
    
        Pot - Potential instance
    
        l - Galactic longitude [deg/rad]
        
        deg= if True (default), l in deg
        
    OUTPUT:
        
        terminal velocity
        
    HISTORY:
        
        2013-05-31 - Written - Bovy (IAS)
        
    """
    if deg:
        sinl= nu.sin(l/180.*nu.pi)
    else:
        sinl= nu.sin(l)
    return sinl*(omegac(Pot,sinl)-omegac(Pot,1.))

def rl(Pot,lz):
    """
    NAME:

       rl

    PURPOSE:

       calculate the radius of a circular orbit of Lz

    INPUT:

       Pot - Potential instance or list thereof

       lz - Angular momentum

    OUTPUT:

       radius

    HISTORY:

       2012-07-30 - Written - Bovy (IAS@MPIA)

    NOTE:

       seems to take about ~0.5 ms for a Miyamoto-Nagai potential; 
       ~0.75 ms for a MWPotential

    """
    #Find interval
    rstart= _rlFindStart(math.fabs(lz),#assumes vo=1.
                         math.fabs(lz),
                         Pot)
    try:
        return optimize.brentq(_rlfunc,10.**-5.,rstart,
                               args=(math.fabs(lz),
                                     Pot),
                               maxiter=200,disp=False)
    except ValueError: #Probably lz small and starting lz to great
        rlower= _rlFindStart(10.**-5.,
                             math.fabs(lz),
                             Pot,lower=True)
        return optimize.brentq(_rlfunc,rlower,rstart,
                               args=(math.fabs(lz),
                                     Pot))
        

def _rlfunc(rl,lz,pot):
    """Function that gives rvc-lz"""
    thisvcirc= vcirc(pot,rl)
    return rl*thisvcirc-lz

def _rlFindStart(rl,lz,pot,lower=False):
    """find a starting interval for rl"""
    rtry= 2.*rl
    while (2.*lower-1.)*_rlfunc(rtry,lz,pot) > 0.:
        if lower:
            rtry/= 2.
        else:
            rtry*= 2.
    return rtry

def lindbladR(Pot,OmegaP,m=2,**kwargs):
    """
    NAME:

       lindbladR

    PURPOSE:

       calculate the radius of a Lindblad resonance

    INPUT:

       Pot - Potential instance or list of such instances

       OmegaP - pattern speed

       m= order of the resonance (as in m(O-Op)=kappa (negative m for outer)
          use m='corotation' for corotation
       +scipy.optimize.brentq xtol,rtol,maxiter kwargs

    OUTPUT:

       radius of Linblad resonance, None if there is no resonance

    HISTORY:

       2011-10-09 - Written - Bovy (IAS)

    """
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
                                 args=(Pot,OmegaP),**kwargs)
        except ValueError:
            return None
        except RuntimeError: #pragma: no cover 
            raise
        return out
    else:
        try:
            out= optimize.brentq(_lindbladR_eq,0.0000001,1000.,
                                 args=(Pot,OmegaP,m),**kwargs)
        except ValueError:
            return None
        except RuntimeError: #pragma: no cover 
            raise
        return out

def _corotationR_eq(R,Pot,OmegaP):
    return omegac(Pot,R)-OmegaP
def _lindbladR_eq(R,Pot,OmegaP,m):
    return m*(omegac(Pot,R)-OmegaP)-epifreq(Pot,R)

def omegac(Pot,R):
    """

    NAME:

       omegac

    PURPOSE:

       calculate the circular angular speed velocity at R in potential Pot

    INPUT:

       Pot - Potential instance or list of such instances

       R - Galactocentric radius

    OUTPUT:

       circular angular speed

    HISTORY:

       2011-10-09 - Written - Bovy (IAS)

    """
    from planarPotential import evaluateplanarRforces
    try:
        return nu.sqrt(-evaluateplanarRforces(R,Pot)/R)
    except PotentialError:
        from planarPotential import RZToplanarPotential
        Pot= RZToplanarPotential(Pot)
        return nu.sqrt(-evaluateplanarRforces(R,Pot)/R)

def _check_c(Pot):
    """

    NAME:

       _check_c

    PURPOSE:

       check whether a potential or list thereof has a C implementation

    INPUT:

       Pot - Potential instance or list of such instances

    OUTPUT:

       True if a C implementation exists, False otherwise

    HISTORY:

       2014-02-17 - Written - Bovy (IAS)

    """
    if isinstance(Pot,list):
        return nu.all(nu.array([p.hasC for p in Pot],dtype='bool'))
    elif isinstance(Pot,Potential):
        return Pot.hasC
