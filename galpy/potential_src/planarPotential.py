import numpy as nu
import galpy.util.bovy_plot as plot
from Potential import PotentialError, Potential, lindbladR
from plotRotcurve import plotRotcurve
from plotEscapecurve import plotEscapecurve
_INF= 1000000.
class planarPotential:
    """Class representing 2D (R,\phi) potentials"""
    def __init__(self,amp=1.):
        self._amp= 1.
        self.dim= 2
        self.isNonAxi= True #Gets reset by planarAxiPotential
        self.isRZ= False
        self.hasC= False
        return None

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
        try:
            return self._amp*self._evaluate(R,phi=phi,t=t,dR=dR,dphi=dphi)
        except AttributeError:
            raise PotentialError("'_evaluate' function not implemented for this potential")

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
        except AttributeError:
            raise PotentialError("'_Rforce' function not implemented for this potential")

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
        except AttributeError:
            raise PotentialError("'_phiforce' function not implemented for this potential")

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
        except AttributeError:
            raise PotentialError("'_R2deriv' function not implemented for this potential")      

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
        plotplanarPotentials(self,*args,**kwargs)

class planarAxiPotential(planarPotential):
    """Class representing axisymmetric planar potentials"""
    def __init__(self,amp=1.):
        planarPotential.__init__(self,amp=amp)
        self.isNonAxi= False
        return None
    
    def _phiforce(self,R,phi=0.,t=0.):
        return 0.

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
           2011-10-17 - Written - Bovy (IAS)
        """
        return 0.

    def _Rphideriv(self,R,z,phi=0.,t=0.):
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
        plotRotcurve(self,*args,**kwargs)

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
        plotEscapecurve(self,*args,**kwargs)

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
        self._RZPot= RZPot
        return None

    def _evaluate(self,R,phi=0.,t=0.,dR=0,dphi=0):
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
        return self._RZPot(R,0.,t=t,dR=dR,dphi=dphi)
            
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
        return self._RZPot.Rforce(R,0.,t=t)

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
        return self._RZPot.R2deriv(R,0.,t=t)
            
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
    if isinstance(Pot,list):
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
    else:
        raise TypeError("Input to 'evaluateplanarPotentials' is neither a Potential-instance or a list of such instances")

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
    if isinstance(Pot,list):
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
    else:
        raise TypeError("Input to 'evaluateplanarRforces' is neither a Potential-instance or a list of such instances")

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
    if isinstance(Pot,list):
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
    else:
        raise TypeError("Input to 'evaluateplanarphiforces' is neither a Potential-instance or a list of such instances")

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
    if isinstance(Pot,list):
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
    else:
        raise TypeError("Input to 'evaluateplanarR2derivs' is neither a Potential-instance or a list of such instances")

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
    if kwargs.has_key('Rrange'):
        Rrange= kwargs['Rrange']
        kwargs.pop('Rrange')
    else:
        Rrange= [0.01,5.]
    if kwargs.has_key('xrange'):
        xrange= kwargs['xrange']
        kwargs.pop('xrange')
    else:
        xrange= [-5.,5.]
    if kwargs.has_key('yrange'):
        yrange= kwargs['yrange']
        kwargs.pop('yrange')
    else:
        yrange= [-5.,5.]
    if kwargs.has_key('grid'):
        grid= kwargs['grid']
        kwargs.pop('grid')
    else:
        grid= 1000 #avoid zero
    if kwargs.has_key('gridx'):
        gridx= kwargs['gridx']
        kwargs.pop('gridx')
    else:
        gridx= 1000 #avoid zero
    if kwargs.has_key('gridy'):
        gridy= kwargs['gridy']
        kwargs.pop('gridy')
    else:
        gridy= gridx
    if kwargs.has_key('savefilename'):
        savefilename= kwargs['savefilename']
        kwargs.pop('savefilename')
    else:
        savefilename= None
    isList= isinstance(Pot,list)
    nonAxi= ((isList and Pot[0].isNonAxi) or (not isList and Pot.isNonAxi))
    if not savefilename == None and os.path.exists(savefilename):
        print "Restoring savefile "+savefilename+" ..."
        savefile= open(savefilename,'rb')
        potR= pickle.load(savefile)
        Rs= pickle.load(savefile)
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
        if not savefilename == None:
            print "Writing savefile "+savefilename+" ..."
            savefile= open(savefilename,'wb')
            pickle.dump(potR,savefile)
            if nonAxi:
                pickle.dump(xs,savefile)
                pickle.dump(ys,savefile)
            else:
                pickle.dump(Rs,savefile)
                savefile.close()
    if nonAxi:
        if not kwargs.has_key('origin'):
            kwargs['origin']= 'lower'
        if not kwargs.has_key('cmap'):
            kwargs['cmap']= 'gist_yarg'
        if not kwargs.has_key('contours'):
            kwargs['contours']= True
        if not kwargs.has_key('xlabel'):
            kwargs['xlabel']= r"$x / R_0$"
        if not kwargs.has_key('ylabel'):
            kwargs['ylabel']= "$y / R_0$"
        if not kwargs.has_key('aspect'):
            kwargs['aspect']= 1.
        if not kwargs.has_key('cntrls'):
            kwargs['cntrls']= '-'
        if kwargs.has_key('ncontours'):
            ncontours= kwargs['ncontours']
            kwargs.pop('ncontours')
        else:
            ncontours=10
        if not kwargs.has_key('levels'):
            kwargs['levels']= nu.linspace(nu.nanmin(potR),nu.nanmax(potR),ncontours)
        return plot.bovy_dens2d(potR.T,
                                xrange=xrange,
                                yrange=yrange,**kwargs)
    else:
        kwargs['xlabel']=r"$R/R_0$"
        kwargs['ylabel']=r"$\Phi(R)$",
        kwargs['xrange']=Rrange
        return plot.bovy_plot(Rs,potR,*args,**kwargs)
                              
    
