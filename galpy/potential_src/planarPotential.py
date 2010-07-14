import numpy as nu
import galpy.util.bovy_plot as plot
from Potential import PotentialError, Potential
from plotRotcurve import plotRotcurve
class planarPotential:
    """Class representing 2D (R,\phi) potentials"""
    def __init__(self,amp=1.):
        self._amp= 1.
        self.dim= 2
        self.isNonAxi= True #Gets reset by planarAxiPotential
        self.isRZ= False
        return None

    def __call__(self,*args):
        """
        NAME:
           __call__
        PURPOSE:
           evaluate the potential
        INPUT: 
           Either: R or R,phi [rad]
        OUTPUT:
           Phi(R(,phi)))
        HISTORY:
           2010-07-13 - Written - Bovy (NYU)
        """
        try:
            return self._amp*self._evaluate(*args)
        except AttributeError:
            raise PotentialError("'_evaluate' function not implemented for this potential")

    def Rforce(self,*args):
        """
        NAME:
           Rforce
        PURPOSE:
           evaluate the radial force
        INPUT:
           Either: R or R,phi [rad]         
        OUTPUT:
           F_R(R,(\phi)))
        HISTORY:
           2010-07-13 - Written - Bovy (NYU)
        """
        try:
            return self._amp*self._Rforce(*args)
        except AttributeError:
            raise PotentialError("'_Rforce' function not implemented for this potential")

    def phiforce(self,*args):
        """
        NAME:
           phiforce
        PURPOSE:
           evaluate the phi force
        INPUT:
           Either: R or R,phi [rad]         
        OUTPUT:
           F_\phi(R,(\phi)))
        HISTORY:
           2010-07-13 - Written - Bovy (NYU)
        """
        try:
            return self._amp*self._phiforce(*args)
        except AttributeError:
            raise PotentialError("'_phiforce' function not implemented for this potential")


class planarAxiPotential(planarPotential):
    """Class representing axisymmetric planar potentials"""
    def __init__(self,amp=1.):
        planarPotential.__init__(self,amp=amp)
        self.isNonAxi= False
        return None
    
    def _phiforce(self,*args):
        return 0.

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

    def _evaluate(self,*args):
        """
        NAME:
           _evaluate
        PURPOSE:
           evaluate the potential
        INPUT:
           Either: R or R,phi [rad]      
        OUTPUT:
          Pot(R(,\phi))
        HISTORY:
           2010-07-13 - Written - Bovy (NYU)
        """
        R= args[0]
        return self._RZPot(R,0.)
            
    def _Rforce(self,*args):
        """
        NAME:
           _Rforce
        PURPOSE:
           evaluate the radial force
        INPUT:
           Either: R or R,phi [rad]      
        OUTPUT:
          F_R(R(,\phi))
        HISTORY:
           2010-07-13 - Written - Bovy (NYU)
        """
        R= args[0]
        return self._RZPot.Rforce(R,0.)
            
def RZToplanarPotential(RZPot):
    """
    NAME:
       RZToPlanarPotential
    PURPOSE:
       convert an RZPotential to a planarPotential in the mid-plane (z=0)
    INPUT:
       RZPot - RZPotential instance or list of such instances (existing 
               planarPotential instances are just copied to the output)
    OUTPUT:
       planarPotential instance
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
    else:
        raise PotentialError("Input to 'RZTolinearPotential' is neither an RZPotential-instance or a list of such instances")

def evaluateplanarPotentials(*args):
    """
    NAME:
       evaluateplanarPotentials
    PURPOSE:
       evaluate a (list of) planarPotential instance(s)
    INPUT:
       R (+phi optional)
       Pot - (list of) planarPotential instance(s)
    OUTPUT:
       Phi(R(,phi))
    HISTORY:
       2010-07-13 - Written - Bovy (NYU)
    """
    hasphi= (len(args) == 3)
    if hasphi:
        potindx= 2
    else:
        potindx= 1
    Pot= args[potindx]
    isList= isinstance(Pot,list)
    nonAxi= ((isList and Pot[0].isNonAxi) or (not isList and Pot.isNonAxi))
    if nonAxi and not hasphi:
        raise PotentialError("The (list of) planarPotential instances is non-axisymmetric, but you did not provide phi")
    if isinstance(Pot,list):
        sum= 0.
        for pot in Pot:
            if nonAxi:
                sum+= pot(args[0],args[1])
            else:
                sum+= pot(args[0])
        return sum
    elif isinstance(Pot,planarPotential):
        if nonAxi:
            return Pot(args[0],args[1])
        else:
            return Pot(args[0])
    else:
        raise PotentialError("Input to 'evaluatePotentials' is neither a Potential-instance or a list of such instances")

def evaluateplanarRforces(*args):
    """
    NAME:
       evaluateplanarRforces
    PURPOSE:
       evaluate the Rforce of a (list of) planarPotential instance(s)
    INPUT:
       R (+phi optional)
       Pot - (list of) planarPotential instance(s)
    OUTPUT:
       F_R(R(,phi))
    HISTORY:
       2010-07-13 - Written - Bovy (NYU)
    """
    hasphi= (len(args) == 3)
    if hasphi:
        potindx= 2
    else:
        potindx= 1
    Pot= args[potindx]
    isList= isinstance(Pot,list)
    nonAxi= ((isList and Pot[0].isNonAxi) or (not isList and Pot.isNonAxi))
    if nonAxi and not hasphi:
        raise PotentialError("The (list of) planarPotential instances is non-axisymmetric, but you did not provide phi")
    if isinstance(Pot,list):
        sum= 0.
        for pot in Pot:
            if nonAxi:
                sum+= pot.Rforce(args[0],args[1])
            else:
                sum+= pot.Rforce(args[0])
        return sum
    elif isinstance(Pot,planarPotential):
        if nonAxi:
            return Pot.Rforce(args[0],args[1])
        else:
            return Pot.Rforce(args[0])
    else:
        raise PotentialError("Input to 'evaluateRforces' is neither a Potential-instance or a list of such instances")

def evaluateplanarphiforces(*args):
    """
    NAME:
       evaluateplanarphiforces
    PURPOSE:
       evaluate the phiforce of a (list of) planarPotential instance(s)
    INPUT:
       R (+phi optional)
       Pot - (list of) planarPotential instance(s)
    OUTPUT:
       F_phi(R(,phi))
    HISTORY:
       2010-07-13 - Written - Bovy (NYU)
    """
    hasphi= (len(args) == 3)
    if hasphi:
        potindx= 2
    else:
        potindx= 1
    Pot= args[potindx]
    isList= ininstance(Pot,list)
    nonAxi= ((isList and Pot[0].isNonAxi) or (not isList and Pot.isNonAxi))
    if nonAxi and not hasphi:
        raise PotentialError("The (list of) planarPotential instances is non-axisymmetric, but you did not provide phi")
    if isinstance(Pot,list):
        sum= 0.
        for pot in Pot:
            if nonAxi:
                sum+= pot.phiforce(args[0],args[1])
            else:
                sum+= pot.phiforce(args[0])
        return sum
    elif isinstance(Pot,planarPotential):
        if nonAxi:
            return Pot.phiforce(args[0],args[1])
        else:
            return Pot.phiforce(args[0])
    else:
        raise PotentialError("Input to 'evaluatephiforces' is neither a Potential-instance or a list of such instances")

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
        grid= 1001
    if kwargs.has_key('gridx'):
        gridx= kwargs['gridx']
        kwargs.pop('gridx')
    else:
        gridx= 1001
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
                    potR[ii,jj]= evaluateplanarPotentials(thisR,thisphi,Pot)
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
            kwargs['xlabel']= r"$R/R_0$"
        if not kwargs.has_key('ylabel'):
            kwargs['ylabel']= "$z/R_0$",
        if not kwargs.has_key('aspect'):
            kwargs['aspect']= 1.
        if not kwargs.has_key('cntrls'):
            kwargs['cntrls']= '-'
        if kwargs.has_key('ncontours'):
            ncontours= kwargs['ncontours']
            kwargs.pop('ncontours')
        if not kwargs.has_key('levels'):
            kwargs['levels']= nu.linspace(nu.nanmin(potR),nu.nanmax(potR),ncontours)
        return plot.bovy_dens2d(potR.T,
                                xrange=xrange,
                                yrange=yrange,**kwargs)
    else:
        return plot.bovy_plot(Rs,potR,*args,
                              xlabel=r"$R/R_0$",ylabel=r"$\Phi(R)$",
                              xrange=Rrange,**kwargs)
    
