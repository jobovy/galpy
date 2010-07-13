###############################################################################
#   Potential.py: top-level class for a full (R,z) potential
#
#   Evaluate by calling the instance: Phi(z;R)= Pot(z,R)
#
#   API for Potentials:
#      function _evaluate(self,R,z) returns Phi(z;R)
#      function _Rforce(self,R,z) return K_R
#      function _zforce(self,R,z) return K_z
###############################################################################
import os, os.path
import cPickle as pickle
import numpy as nu
import galpy.util.bovy_plot as plot
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
        return None

    def __call__(self,R,z):
        """
        NAME:
           __call__
        PURPOSE:
           evaluate the (R,z)
        INPUT:
           R - Cylindrical Galactocentric radius
           z - vertical height
        OUTPUT:
           Phi(z;R)
        HISTORY:
           2010-04-16 - Written - Bovy (NYU)
        """
        try:
            return self._amp*self._evaluate(R,z)
        except AttributeError:
            raise PotentialError("'_evaluate' function not implemented for this potential")

    def Rforce(self,R,z):
        """
        NAME:
           Rforce
        PURPOSE:
           evaluate radial force K_R  (R,z)
        INPUT:
           R - Cylindrical Galactocentric radius
           z - vertical height
        OUTPUT:
           K_R (R,z)
        HISTORY:
           2010-04-16 - Written - Bovy (NYU)
        DOCTEST:
        """
        try:
            return self._amp*self._Rforce(R,z)
        except AttributeError:
            raise PotentialError("'_Rforce' function not implemented for this potential")
        
    def zforce(self,R,z):
        """
        NAME:
           zforce
        PURPOSE:
           evaluate the vertical force K_R  (R,z)
        INPUT:
           R - Cylindrical Galactocentric radius
           z - vertical height
        OUTPUT:
           K_z (R,z)
        HISTORY:
           2010-04-16 - Written - Bovy (NYU)
        DOCTEST:
        """
        try:
            return self._amp*self._zforce(R,z)
        except AttributeError:
            raise PotentialError("'_zforce' function not implemented for this potential")

    def normalize(self,norm):
        """
        NAME:
           normalize
        PURPOSE:
           normalize a potential in such a way that vc(R=1,z=0)=1., or a 
           fraction of this
        INPUT:
           norm - normalize such that Rforce(R=1,z=0) is such that it is
                  'norm' of the force necessary to make vc(R=1,z=0)=1
                  if True, norm=1
        OUTPUT:
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        self._amp*= norm/nu.fabs(self.Rforce(1.,0.))

    def _phiforce(self,R,z,phi):
        """
        NAME:
           _phiforce
        PURPOSE:
           evaluate the azimuthal force K_R  (R,z,phi)
        INPUT:
           R - Cylindrical Galactocentric radius
           z - vertical height
           phi - azimuth (rad)
        OUTPUT:
           K_phi (R,z,phi)
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        DOCTEST:
        """
        return 0.

    def plot(self,rmin=0.,rmax=1.5,nrs=21,zmin=-0.5,zmax=0.5,nzs=21,
             ncontours=21,savefilename=None):
        """
        NAME:
           plot
        PURPOSE:
           plot the potential
        INPUT:
           rmin - minimum R
           rmax - maximum R
           nrs - grid in R
           zmin - minimum z
           zmax - maximum z
           nzs - grid in z
           ncontours - number of contours
           savefilename - save to or restore from this savefile (pickle)
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
                    potRz[ii,jj]= self._evaluate(Rs[ii],zs[jj])
            if not savefilename == None:
                print "Writing savefile "+savefilename+" ..."
                savefile= open(savefilename,'wb')
                pickle.dump(potRz,savefile)
                pickle.dump(Rs,savefile)
                pickle.dump(zs,savefile)
                savefile.close()
        return plot.bovy_dens2d(potRz.T,origin='lower',cmap='gist_gray',contours=True,
                                xlabel=r"$R/R_0$",ylabel=r"$z/R_0$",
                                xrange=[rmin,rmax],
                                yrange=[zmin,zmax],
                                aspect=.75*(rmax-rmin)/(zmax-zmin),
                                cntrls='-',
                                levels=nu.linspace(nu.nanmin(potRz),nu.nanmax(potRz),
                                                   ncontours))
        

    def plotRotcurve(self,rmin=0.,rmax=1.5,nrs=21,
                     savefilename=None,*args,**kwargs):
        """
        NAME:
           plotRotcurve
        PURPOSE:
           plot the rotation curve for this potential (in the z=0 plane for
           non-spherical potentials)
        INPUT:
           rmin - minimum R
           rmax - maximum R
           nrs - grid in R
           savefilename - save to or restore from this savefile (pickle)
           +matploltib.plot args and kwargs
        OUTPUT:
           plot to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        if not savefilename == None and os.path.exists(savefilename):
            print "Restoring savefile "+savefilename+" ..."
            savefile= open(savefilename,'rb')
            rotcurve= pickle.load(savefile)
            Rs= pickle.load(savefile)
            savefile.close()
        else:
            Rs= nu.linspace(rmin,rmax,nrs)
            rotcurve= nu.zeros(nrs)
            for ii in range(nrs):
                rotcurve[ii]= nu.sqrt(Rs[ii]*-self.Rforce(Rs[ii],0.))
            if not savefilename == None:
                print "Writing savefile "+savefilename+" ..."
                savefile= open(savefilename,'wb')
                pickle.dump(rotcurve,savefile)
                pickle.dump(Rs,savefile)
                savefile.close()
        return plot.bovy_plot(Rs,rotcurve,*args,
                              xlabel=r"$R/R_0$",ylabel=r"$v_c(R)$",
                              xrange=[rmin,rmax],**kwargs)


class PotentialError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def evaluatePotentials(R,z,Pot):
    """
    NAME:
       evaluatePotentials
    PURPOSE:
       convenience function to evaluate a possible sum of potentials
    INPUT:
       R - cylindrical Galactocentric distance
       z - distance above the plane
       Pot - potential or list of potentials
    OUTPUT:
       Phi(R,z)
    HISTORY:
       2010-04-16 - Written - Bovy (NYU)
    """
    if isinstance(Pot,list):
        sum= 0.
        for pot in Pot:
            sum+= pot(R,z)
        return sum
    elif isinstance(Pot,Potential):
        return Pot(R,z)
    else:
        raise PotentialError("Input to 'evaluatePotentials' is neither a Potential-instance or a list of such instances")

def evaluateRforces(R,z,Pot):
    """
    NAME:
       evaluateRforce
    PURPOSE:
       convenience function to evaluate a possible sum of potentials
    INPUT:
       R - cylindrical Galactocentric distance
       z - distance above the plane
       Pot - a potential or list of potentials
    OUTPUT:
       K_R(R,z)
    HISTORY:
       2010-04-16 - Written - Bovy (NYU)
    """
    if isinstance(Pot,list):
        sum= 0.
        for pot in Pot:
            sum+= pot.Rforce(R,z)
        return sum
    elif isinstance(Pot,Potential):
        return Pot.Rforce(R,z)
    else:
        raise PotentialError("Input to 'evaluateRforces' is neither a Potential-instance or a list of such instances")

def evaluatezforces(R,z,Pot):
    """
    NAME:
       evaluatezforces
    PURPOSE:
       convenience function to evaluate a possible sum of potentials
    INPUT:
       R - cylindrical Galactocentric distance
       z - distance above the plane
       Pot - a potential or list of potentials
    OUTPUT:
       K_z(R,z)
    HISTORY:
       2010-04-16 - Written - Bovy (NYU)
    """
    if isinstance(Pot,list):
        sum= 0.
        for pot in Pot:
            sum+= pot.zforce(R,z)
        return sum
    elif isinstance(Pot,Potential):
        return Pot.zforce(R,z)
    else:
        raise PotentialError("Input to 'evaluatezforces' is neither a Potential-instance or a list of such instances")

def plotPotentials(Pot,rmin=0.,rmax=1.5,nrs=21,zmin=-0.5,zmax=0.5,nzs=21,
                   ncontours=21,savefilename=None):
        """
        NAME:
           plotPotentials
        PURPOSE:
           plot a set of potentials
        INPUT:
           Pot - Potential or list of Potential instances
           rmin - minimum R
           rmax - maximum R
           nrs - grid in R
           zmin - minimum z
           zmax - maximum z
           nzs - grid in z
           ncontours - number of contours
           savefilename - save to or restore from this savefile (pickle)
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
                    potRz[ii,jj]= evaluatePotentials(Rs[ii],zs[jj],Pot)
            if not savefilename == None:
                print "Writing savefile "+savefilename+" ..."
                savefile= open(savefilename,'wb')
                pickle.dump(potRz,savefile)
                pickle.dump(Rs,savefile)
                pickle.dump(zs,savefile)
                savefile.close()
        return plot.bovy_dens2d(potRz.T,origin='lower',cmap='gist_gray',contours=True,
                                xlabel=r"$R/R_0$",ylabel=r"$z/R_0$",
                                xrange=[rmin,rmax],
                                yrange=[zmin,zmax],
                                aspect=.75*(rmax-rmin)/(zmax-zmin),
                                cntrls='-',
                                levels=nu.linspace(nu.nanmin(potRz),nu.nanmax(potRz),
                                                   ncontours))
        

def plotRotcurve(Pot,rmin=0.,rmax=1.5,nrs=21,
                 savefilename=None,*args,**kwargs):
    """
    NAME:
       plotRotcurve
    PURPOSE:
       plot the rotation curve for this potential (in the z=0 plane for
       non-spherical potentials)
    INPUT:
       Pot - Potential or list of Potential instances
       rmin - minimum R
       rmax - maximum R
       nrs - grid in R
       savefilename - save to or restore from this savefile (pickle)
       +matploltib.plot args and kwargs
    OUTPUT:
       plot to output device
    HISTORY:
       2010-07-10 - Written - Bovy (NYU)
    """
    if not savefilename == None and os.path.exists(savefilename):
        print "Restoring savefile "+savefilename+" ..."
        savefile= open(savefilename,'rb')
        rotcurve= pickle.load(savefile)
        Rs= pickle.load(savefile)
        savefile.close()
    else:
        Rs= nu.linspace(rmin,rmax,nrs)
        rotcurve= nu.zeros(nrs)
        for ii in range(nrs):
            rotcurve[ii]= nu.sqrt(Rs[ii]*-evaluateRforces(Rs[ii],0.,Pot))
        if not savefilename == None:
            print "Writing savefile "+savefilename+" ..."
            savefile= open(savefilename,'wb')
            pickle.dump(rotcurve,savefile)
            pickle.dump(Rs,savefile)
            savefile.close()
    return plot.bovy_plot(Rs,rotcurve,*args,
                          xlabel=r"$R/R_0$",ylabel=r"$v_c(R)$",
                          xrange=[rmin,rmax],**kwargs)

