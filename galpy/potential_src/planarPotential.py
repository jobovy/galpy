from __future__ import division, print_function

import os
import pickle
import numpy as nu
from scipy import integrate
import galpy.util.bovy_plot as plot
from galpy.util.bovy_conversion import physical_conversion
from galpy.potential_src.Potential import Potential, PotentialError, lindbladR
from galpy.potential_src.plotRotcurve import plotRotcurve
from galpy.potential_src.plotEscapecurve import _INF, plotEscapecurve
class planarPotential(object):
    """Class representing 2D (R,\phi) potentials"""
    def __init__(self,amp=1.):
        self._amp= 1.
        self.dim= 2
        self.isNonAxi= True #Gets reset by planarAxiPotential
        self.isRZ= False
        self.hasC= False
        self.hasC_dxdv= False
        return None

    @physical_conversion('energy',pop=True)
    def __call__(self,R,phi=0.,t=0.,dR=0,dphi=0):
        """
        NAME:

           __call__

        PURPOSE:

           evaluate the potential

        INPUT: 

           R - Cylindrica radius

           phi= azimuth (optional)

           t= time (optional)

           dR=, dphi= if set to non-zero integers, return the dR,dphi't derivative

        OUTPUT:

           Phi(R(,phi,t)))

        HISTORY:

           2010-07-13 - Written - Bovy (NYU)

        """
        if dR == 0 and dphi == 0:
            try:
                return self._amp*self._evaluate(R,phi=phi,t=t)
            except AttributeError: #pragma: no cover
                raise PotentialError("'_evaluate' function not implemented for this potential")
        elif dR == 1 and dphi == 0:
            return -self.Rforce(R,phi=phi,t=t)
        elif dR == 0 and dphi == 1:
            return -self.phiforce(R,phi=phi,t=t)
        elif dR == 2 and dphi == 0:
            return self.R2deriv(R,phi=phi,t=t)
        elif dR == 0 and dphi == 2:
            return self.phi2deriv(R,phi=phi,t=t)
        elif dR == 1 and dphi == 1:
            return self.Rphideriv(R,phi=phi,t=t)

    @physical_conversion('force',pop=True)
    def Rforce(self,R,phi=0.,t=0.):
        """
        NAME:

           Rforce

        PURPOSE:

           evaluate the radial force

        INPUT:

           R - Cylindrical radius

           phi= azimuth (optional)

           t= time (optional)

        OUTPUT:

           F_R(R,(\phi,t)))

        HISTORY:

           2010-07-13 - Written - Bovy (NYU)

        """
        try:
            return self._amp*self._Rforce(R,phi=phi,t=t)
        except AttributeError: #pragma: no cover
            raise PotentialError("'_Rforce' function not implemented for this potential")

    @physical_conversion('force',pop=True)
    def phiforce(self,R,phi=0.,t=0.):
        """
        NAME:

           phiforce

        PURPOSE:

           evaluate the phi force

        INPUT:

           R - Cylindrical radius

           phi= azimuth (optional)

           t= time (optional)

        OUTPUT:

           F_\phi(R,(\phi,t)))

        HISTORY:

           2010-07-13 - Written - Bovy (NYU)

        """
        try:
            return self._amp*self._phiforce(R,phi=phi,t=t)
        except AttributeError: #pragma: no cover
            raise PotentialError("'_phiforce' function not implemented for this potential")

    @physical_conversion('forcederivative',pop=True)
    def R2deriv(self,R,phi=0.,t=0.):
        """
        NAME:

           R2deriv

        PURPOSE:

           evaluate the second radial derivative

        INPUT:

           R - Cylindrical radius

           phi= azimuth (optional)

           t= time (optional)

        OUTPUT:

           d2phi/dR2

        HISTORY:

           2011-10-09 - Written - Bovy (IAS)

        """
        try:
            return self._amp*self._R2deriv(R,phi=phi,t=t)
        except AttributeError: #pragma: no cover
            raise PotentialError("'_R2deriv' function not implemented for this potential")      

    @physical_conversion('forcederivative',pop=True)
    def phi2deriv(self,R,phi=0.,t=0.):
        """
        NAME:

           phi2deriv

        PURPOSE:

           evaluate the second azimuthal derivative

        INPUT:

           R - Cylindrical radius

           phi= azimuth (optional)

           t= time (optional)

        OUTPUT:

           d2phi/daz2

        HISTORY:

           2014-04-06 - Written - Bovy (IAS)

        """
        try:
            return self._amp*self._phi2deriv(R,phi=phi,t=t)
        except AttributeError: #pragma: no cover
            raise PotentialError("'_phi2deriv' function not implemented for this potential")      

    @physical_conversion('forcederivative',pop=True)
    def Rphideriv(self,R,phi=0.,t=0.):
        """
        NAME:

           Rphideriv

        PURPOSE:

           evaluate the mixed radial and azimuthal  derivative

        INPUT:

           R - Cylindrical radius

           phi= azimuth (optional)

           t= time (optional)

        OUTPUT:

           d2phi/dR d az

        HISTORY:

           2014-05-21 - Written - Bovy (IAS)

        """
        try:
            return self._amp*self._Rphideriv(R,phi=phi,t=t)
        except AttributeError: #pragma: no cover
            raise PotentialError("'_Rphideriv' function not implemented for this potential")      

    def plot(self,*args,**kwargs):
        """
        NAME:
           plot
        PURPOSE:
           plot the potential
        INPUT:
           Rrange - range
           grid - number of points to plot
           savefilename - save to or restore from this savefile (pickle)
           +bovy_plot(*args,**kwargs)
        OUTPUT:
           plot to output device
        HISTORY:
           2010-07-13 - Written - Bovy (NYU)
        """
        return plotplanarPotentials(self,*args,**kwargs)

class planarAxiPotential(planarPotential):
    """Class representing axisymmetric planar potentials"""
    def __init__(self,amp=1.):
        planarPotential.__init__(self,amp=amp)
        self.isNonAxi= False
        return None
    
    def _phiforce(self,R,phi=0.,t=0.):
        return 0.

    def _phi2deriv(self,R,phi=0.,t=0.): #pragma: no cover
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
           2011-10-17 - Written - Bovy (IAS)
        """
        return 0.

    def _Rphideriv(self,R,phi=0.,t=0.): #pragma: no cover
        """
        NAME:
           _Rphideriv
        PURPOSE:
           evaluate the radial+azimuthal derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the radial+azimuthal derivative
        HISTORY:
           2011-10-17 - Written - Bovy (IAS)
        """
        return 0.

    @physical_conversion('velocity',pop=True)
    def vcirc(self,R):
        """
        
        NAME:
        
            vcirc
        
        PURPOSE:
        
            calculate the circular velocity at R in potential Pot

        INPUT:
        
            Pot - Potential instance or list of such instances
        
            R - Galactocentric radius
        
        OUTPUT:
        
            circular rotation velocity
        
        HISTORY:
        
            2011-10-09 - Written - Bovy (IAS)
        
        """
        return nu.sqrt(R*-self.Rforce(R))       

    @physical_conversion('frequency',pop=True)
    def omegac(self,R):
        """
        
        NAME:
        
            omegac
        
        PURPOSE:
        
            calculate the circular angular speed at R in potential Pot

        INPUT:
        
            Pot - Potential instance or list of such instances
        
            R - Galactocentric radius
        
        OUTPUT:
        
            circular angular speed
        
        HISTORY:
        
            2011-10-09 - Written - Bovy (IAS)
        
        """
        return nu.sqrt(-self.Rforce(R)/R)       

    @physical_conversion('frequency',pop=True)
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
        return nu.sqrt(self.R2deriv(R)-3./R*self.Rforce(R))

    @physical_conversion('position',pop=True)
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

    @physical_conversion('velocity',pop=True)
    def vesc(self,R):
        """

        NAME:

            vesc

        PURPOSE:

            calculate the escape velocity at R for potential Pot

        INPUT:

            Pot - Potential instances or list thereof

            R - Galactocentric radius

        OUTPUT:

            escape velocity

        HISTORY:

            2011-10-09 - Written - Bovy (IAS)

        """
        return nu.sqrt(2.*(self(_INF)-self(R)))
        
    def plotRotcurve(self,*args,**kwargs):
        """
        NAME:

           plotRotcurve

        PURPOSE:

           plot the rotation curve for this potential

        INPUT:

           Rrange - range

           grid - number of points to plot

           savefilename - save to or restore from this savefile (pickle)

           +bovy_plot(*args,**kwargs)

        OUTPUT:

           plot to output device

        HISTORY:

           2010-07-13 - Written - Bovy (NYU)

        """
        return plotRotcurve(self,*args,**kwargs)

    def plotEscapecurve(self,*args,**kwargs):
        """
        NAME:

           plotEscapecurve

        PURPOSE:

           plot the escape velocity curve for this potential

        INPUT:

           Rrange - range

           grid - number of points to plot

           savefilename - save to or restore from this savefile (pickle)

           +bovy_plot(*args,**kwargs)

        OUTPUT:

           plot to output device

        HISTORY:

           2010-07-13 - Written - Bovy (NYU)

        """
        return plotEscapecurve(self,*args,**kwargs)

class planarPotentialFromRZPotential(planarAxiPotential):
    """Class that represents an axisymmetic planar potential derived from a 
    RZPotential"""
    def __init__(self,RZPot):
        """
        NAME:
           __init__
        PURPOSE:
           Initialize
        INPUT:
           RZPot - RZPotential instance
        OUTPUT:
           planarAxiPotential instance
        HISTORY:
           2010-07-13 - Written - Bovy (NYU)
        """
        planarAxiPotential.__init__(self,amp=1.)
        # Also transfer ro and vo
        self._ro= RZPot._ro
        self._roSet= RZPot._roSet
        self._vo= RZPot._vo
        self._voSet= RZPot._voSet
        self._RZPot= RZPot
        self.hasC= RZPot.hasC
        self.hasC_dxdv= RZPot.hasC_dxdv
        return None

    def _evaluate(self,R,phi=0.,t=0.):
        """
        NAME:
           _evaluate
        PURPOSE:
           evaluate the potential
        INPUT:
           R
           phi
           t
        OUTPUT:
          Pot(R(,\phi,t))
        HISTORY:
           2010-07-13 - Written - Bovy (NYU)
        """
        return self._RZPot(R,0.,t=t,use_physical=False)
            
    def _Rforce(self,R,phi=0.,t=0.):
        """
        NAME:
           _Rforce
        PURPOSE:
           evaluate the radial force
        INPUT:
           R
           phi
           t
        OUTPUT:
          F_R(R(,\phi,t))
        HISTORY:
           2010-07-13 - Written - Bovy (NYU)
        """
        return self._RZPot.Rforce(R,0.,t=t,use_physical=False)

    def _R2deriv(self,R,phi=0.,t=0.):
        """
        NAME:
           _R2deriv
        PURPOSE:
           evaluate the second radial derivative
        INPUT:
           R
           phi
           t
        OUTPUT:
           d2phi/dR2
        HISTORY:
           2011-10-09 - Written - Bovy (IAS)
        """
        return self._RZPot.R2deriv(R,0.,t=t,use_physical=False)
            
def RZToplanarPotential(RZPot):
    """
    NAME:

       RZToplanarPotential

    PURPOSE:

       convert an RZPotential to a planarPotential in the mid-plane (z=0)

    INPUT:

       RZPot - RZPotential instance or list of such instances (existing planarPotential instances are just copied to the output)

    OUTPUT:

       planarPotential instance(s)

    HISTORY:

       2010-07-13 - Written - Bovy (NYU)

    """
    if isinstance(RZPot,list):
        out= []
        for pot in RZPot:
            if isinstance(pot,planarPotential):
                out.append(pot)
            else:
                out.append(planarPotentialFromRZPotential(pot))
        return out
    elif isinstance(RZPot,Potential):
        return planarPotentialFromRZPotential(RZPot)
    elif isinstance(RZPot,planarPotential):
        return RZPot
    else:
        raise PotentialError("Input to 'RZToplanarPotential' is neither an RZPotential-instance or a list of such instances")

def evaluateplanarPotentials(R,Pot,phi=None,t=0.,dR=0,dphi=0):
    """
    NAME:

       evaluateplanarPotentials

    PURPOSE:

       evaluate a (list of) planarPotential instance(s)

    INPUT:

       R - Cylindrical radius

       Pot - (list of) planarPotential instance(s)

       phi= azimuth (optional)

       t= time (optional)

       dR=, dphi= if set to non-zero integers, return the dR,dphi't derivative instead

    OUTPUT:

       Phi(R(,phi,t))

    HISTORY:

       2010-07-13 - Written - Bovy (NYU)

    """
    isList= isinstance(Pot,list)
    if isList:
        isAxis= [not p.isNonAxi for p in Pot]
        nonAxi= not nu.prod(nu.array(isAxis))
    else:
        nonAxi= Pot.isNonAxi
    if nonAxi and phi is None:
        raise PotentialError("The (list of) planarPotential instances is non-axisymmetric, but you did not provide phi")
    if isinstance(Pot,list) \
            and nu.all([isinstance(p,planarPotential) for p in Pot]):
        sum= 0.
        for pot in Pot:
            if nonAxi:
                sum+= pot(R,phi=phi,t=t,dR=dR,dphi=dphi)
            else:
                sum+= pot(R,t=t,dR=dR,dphi=dphi)
        return sum
    elif isinstance(Pot,planarPotential):
        if nonAxi:
            return Pot(R,phi=phi,t=t,dR=dR,dphi=dphi)
        else:
            return Pot(R,t=t,dR=dR,dphi=dphi)
    else: #pragma: no cover 
        raise PotentialError("Input to 'evaluatePotentials' is neither a Potential-instance or a list of such instances")

def evaluateplanarRforces(R,Pot,phi=None,t=0.):
    """
    NAME:

       evaluateplanarRforces

    PURPOSE:

       evaluate the Rforce of a (list of) planarPotential instance(s)

    INPUT:

       R - Cylindrical radius

       Pot - (list of) planarPotential instance(s)

       phi= azimuth (optional)

       t= time (optional)

    OUTPUT:

       F_R(R(,phi,t))

    HISTORY:

       2010-07-13 - Written - Bovy (NYU)

    """
    isList= isinstance(Pot,list)
    if isList:
        isAxis= [not p.isNonAxi for p in Pot]
        nonAxi= not nu.prod(nu.array(isAxis))
    else:
        nonAxi= Pot.isNonAxi
    if nonAxi and phi is None:
        raise PotentialError("The (list of) planarPotential instances is non-axisymmetric, but you did not provide phi")
    if isinstance(Pot,list) \
            and nu.all([isinstance(p,planarPotential) for p in Pot]):
        sum= 0.
        for pot in Pot:
            if nonAxi:
                sum+= pot.Rforce(R,phi=phi,t=t)
            else:
                sum+= pot.Rforce(R,t=t)
        return sum
    elif isinstance(Pot,planarPotential):
        if nonAxi:
            return Pot.Rforce(R,phi=phi,t=t)
        else:
            return Pot.Rforce(R,t=t)
    else: #pragma: no cover 
        raise PotentialError("Input to 'evaluatePotentials' is neither a Potential-instance or a list of such instances")

def evaluateplanarphiforces(R,Pot,phi=None,t=0.):
    """
    NAME:

       evaluateplanarphiforces

    PURPOSE:

       evaluate the phiforce of a (list of) planarPotential instance(s)

    INPUT:

       R - Cylindrical radius

       Pot - (list of) planarPotential instance(s)

       phi= azimuth (optional)

       t= time (optional)

    OUTPUT:

       F_phi(R(,phi,t))

    HISTORY:

       2010-07-13 - Written - Bovy (NYU)

    """
    isList= isinstance(Pot,list)
    if isList:
        isAxis= [not p.isNonAxi for p in Pot]
        nonAxi= not nu.prod(nu.array(isAxis))
    else:
        nonAxi= Pot.isNonAxi
    if nonAxi and phi is None:
        raise PotentialError("The (list of) planarPotential instances is non-axisymmetric, but you did not provide phi")
    if isinstance(Pot,list) \
            and nu.all([isinstance(p,planarPotential) for p in Pot]):
        sum= 0.
        for pot in Pot:
            if nonAxi:
                sum+= pot.phiforce(R,phi=phi,t=t)
            else:
                sum+= pot.phiforce(R,t=t)
        return sum
    elif isinstance(Pot,planarPotential):
        if nonAxi:
            return Pot.phiforce(R,phi=phi,t=t)
        else:
            return Pot.phiforce(R,t=t)
    else: #pragma: no cover 
        raise PotentialError("Input to 'evaluatePotentials' is neither a Potential-instance or a list of such instances")

def evaluateplanarR2derivs(R,Pot,phi=None,t=0.):
    """
    NAME:

       evaluateplanarR2derivs

    PURPOSE:

       evaluate the second radial derivative of a (list of) planarPotential instance(s)

    INPUT:

       R - Cylindrical radius

       Pot - (list of) planarPotential instance(s)

       phi= azimuth (optional)

       t= time (optional)

    OUTPUT:

       F_R(R(,phi,t))

    HISTORY:

       2010-10-09 - Written - Bovy (IAS)

    """
    isList= isinstance(Pot,list)
    if isList:
        isAxis= [not p.isNonAxi for p in Pot]
        nonAxi= not nu.prod(nu.array(isAxis))
    else:
        nonAxi= Pot.isNonAxi
    if nonAxi and phi is None:
        raise PotentialError("The (list of) planarPotential instances is non-axisymmetric, but you did not provide phi")
    if isinstance(Pot,list) \
            and nu.all([isinstance(p,planarPotential) for p in Pot]):
        sum= 0.
        for pot in Pot:
            if nonAxi:
                sum+= pot.R2deriv(R,phi=phi,t=t)
            else:
                sum+= pot.R2deriv(R,t=t)
        return sum
    elif isinstance(Pot,planarPotential):
        if nonAxi:
            return Pot.R2deriv(R,phi=phi,t=t)
        else:
            return Pot.R2deriv(R,t=t)
    else: #pragma: no cover 
        raise PotentialError("Input to 'evaluatePotentials' is neither a Potential-instance or a list of such instances")

def LinShuReductionFactor(axiPot,R,sigmar,nonaxiPot=None,
                          k=None,m=None,OmegaP=None):
    """
    NAME:

       LinShuReductionFactor

    PURPOSE:

       Calculate the Lin & Shu (1966) reduction factor: the reduced linear response of a kinematically-warm stellar disk to a perturbation

    INPUT:

       axiPot - The background, axisymmetric potential

       R - Cylindrical radius
       
       sigmar - radial velocity dispersion of the population

       Then either provide:

       1) m= m in the perturbation's m x phi (number of arms for a spiral)

          k= wavenumber (see Binney & Tremaine 2008)

          OmegaP= pattern speed

       2) nonaxiPot= a non-axisymmetric Potential instance (such as SteadyLogSpiralPotential) that has functions that return OmegaP, m, and wavenumber

    OUTPUT:

       reduction factor

    HISTORY:

       2014-08-23 - Written - Bovy (IAS)

    """
    from galpy.potential import omegac, epifreq
    if nonaxiPot is None and (OmegaP is None or k is None or m is None):
        raise IOError("Need to specify either nonaxiPot= or m=, k=, OmegaP= for LinShuReductionFactor")
    elif not nonaxiPot is None:
        OmegaP= nonaxiPot.OmegaP()
        k= nonaxiPot.wavenumber(R)
        m= nonaxiPot.m()
    tepif= epifreq(axiPot,R)
    s= m*(OmegaP-omegac(axiPot,R))/tepif
    chi= sigmar**2.*k**2./tepif**2.
    return (1.-s**2.)/nu.sin(nu.pi*s)\
        *integrate.quad(lambda t: nu.exp(-chi*(1.+nu.cos(t)))\
                            *nu.sin(s*t)*nu.sin(t),
                        0.,nu.pi)[0]

def plotplanarPotentials(Pot,*args,**kwargs):
    """
    NAME:

       plotplanarPotentials

    PURPOSE:

       plot a planar potential

    INPUT:

       Rrange - range

       xrange, yrange - if relevant

       grid, gridx, gridy - number of points to plot

       savefilename - save to or restore from this savefile (pickle)

       ncontours - number of contours to plot (if applicable)

       +bovy_plot(*args,**kwargs) or bovy_dens2d(**kwargs)

    OUTPUT:

       plot to output device

    HISTORY:

       2010-07-13 - Written - Bovy (NYU)

    """
    Rrange= kwargs.pop('Rrange',[0.01,5.])
    xrange= kwargs.pop('xrange',[-5.,5.])
    yrange= kwargs.pop('yrange',[-5.,5.])
    grid= kwargs.pop('grid',100)
    gridx= kwargs.pop('gridx',100)
    gridy= kwargs.pop('gridy',gridx)
    savefilename= kwargs.pop('savefilename',None)
    isList= isinstance(Pot,list)
    nonAxi= ((isList and Pot[0].isNonAxi) or (not isList and Pot.isNonAxi))
    if not savefilename is None and os.path.exists(savefilename):
        print("Restoring savefile "+savefilename+" ...")
        savefile= open(savefilename,'rb')
        potR= pickle.load(savefile)
        if nonAxi:
            xs= pickle.load(savefile)
            ys= pickle.load(savefile)
        else:
            Rs= pickle.load(savefile)
        savefile.close()
    else:
        if nonAxi:
            xs= nu.linspace(xrange[0],xrange[1],gridx)
            ys= nu.linspace(yrange[0],yrange[1],gridy)
            potR= nu.zeros((gridx,gridy))
            for ii in range(gridx):
                for jj in range(gridy):
                    thisR= nu.sqrt(xs[ii]**2.+ys[jj]**2.)
                    if xs[ii] >= 0.:
                        thisphi= nu.arcsin(ys[jj]/thisR)
                    else:
                        thisphi= -nu.arcsin(ys[jj]/thisR)+nu.pi
                    potR[ii,jj]= evaluateplanarPotentials(thisR,Pot,
                                                          phi=thisphi)
        else:
            Rs= nu.linspace(Rrange[0],Rrange[1],grid)
            potR= nu.zeros(grid)
            for ii in range(grid):
                potR[ii]= evaluateplanarPotentials(Rs[ii],Pot)
        if not savefilename is None:
            print("Writing planar savefile "+savefilename+" ...")
            savefile= open(savefilename,'wb')
            pickle.dump(potR,savefile)
            if nonAxi:
                pickle.dump(xs,savefile)
                pickle.dump(ys,savefile)
            else:
                pickle.dump(Rs,savefile)
            savefile.close()
    if nonAxi:
        if not 'orogin' in kwargs:
            kwargs['origin']= 'lower'
        if not 'cmap' in kwargs:
            kwargs['cmap']= 'gist_yarg'
        if not 'contours' in kwargs:
            kwargs['contours']= True
        if not 'xlabel' in kwargs:
            kwargs['xlabel']= r"$x / R_0$"
        if not 'ylabel' in kwargs:
            kwargs['ylabel']= "$y / R_0$"
        if not 'aspect' in kwargs:
            kwargs['aspect']= 1.
        if not 'cntrls' in kwargs:
            kwargs['cntrls']= '-'
        ncontours= kwargs.pop('ncontours',10)
        if not 'levels' in kwargs:
            kwargs['levels']= nu.linspace(nu.nanmin(potR),nu.nanmax(potR),ncontours)
        return plot.bovy_dens2d(potR.T,
                                xrange=xrange,
                                yrange=yrange,**kwargs)
    else:
        kwargs['xlabel']=r"$R/R_0$"
        kwargs['ylabel']=r"$\Phi(R)$"
        kwargs['xrange']=Rrange
        return plot.bovy_plot(Rs,potR,*args,**kwargs)
                              
    
