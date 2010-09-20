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

    def __call__(self,R,phi=0.):
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
            return self._amp*self._evaluate(R,phi=phi)
        except AttributeError:
            raise PotentialError("'_evaluate' function not implemented for this potential")

    def Rforce(self,R,phi=0.):
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
            return self._amp*self._Rforce(R,phi=phi)
        except AttributeError:
            raise PotentialError("'_Rforce' function not implemented for this potential")

    def phiforce(self,R,phi=0.):
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
            return self._amp*self._phiforce(R,phi=phi)
        except AttributeError:
            raise PotentialError("'_phiforce' function not implemented for this potential")


class planarAxiPotential(planarPotential):
    """Class representing axisymmetric planar potentials"""
    def __init__(self,amp=1.):
        planarPotential.__init__(self,amp=amp)
        self.isNonAxi= False
        return None
    
    def _phiforce(self,R,phi=0.):
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

    def _evaluate(self,R,phi=0.):
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
        return self._RZPot(R,0.)
            
    def _Rforce(self,R,phi=0.):
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
        return self._RZPot.Rforce(R,0.)
            
def RZToplanarPotential(RZPot):
    """
    NAME:
       RZToplanarPotential
    PURPOSE:
       convert an RZPotential to a planarPotential in the mid-plane (z=0)
    INPUT:
       RZPot - RZPotential instance or list of such instances (existing 
               planarPotential instances are just copied to the output)
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

def evaluateplanarPotentials(R,Pot,phi=0.):
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
    isList= isinstance(Pot,list)
    nonAxi= ((isList and Pot[0].isNonAxi) or (not isList and Pot.isNonAxi))
    if nonAxi and not hasphi:
        raise PotentialError("The (list of) planarPotential instances is non-axisymmetric, but you did not provide phi")
    if isinstance(Pot,list):
        sum= 0.
        for pot in Pot:
            if nonAxi:
                sum+= pot(R,phi=phi)
            else:
                sum+= pot(R)
        return sum
    elif isinstance(Pot,planarPotential):
        if nonAxi:
            return Pot(R,phi=phi)
        else:
            return Pot(R)
    else:
        raise PotentialError("Input to 'evaluateplanarPotentials' is neither a Potential-instance or a list of such instances")

def evaluateplanarRforces(R,Pot,phi=0.):
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
    isList= isinstance(Pot,list)
    nonAxi= ((isList and Pot[0].isNonAxi) or (not isList and Pot.isNonAxi))
    if nonAxi and not hasphi:
        raise PotentialError("The (list of) planarPotential instances is non-axisymmetric, but you did not provide phi")
    if isinstance(Pot,list):
        sum= 0.
        for pot in Pot:
            if nonAxi:
                sum+= pot.Rforce(R,phi=phi)
            else:
                sum+= pot.Rforce(R)
        return sum
    elif isinstance(Pot,planarPotential):
        if nonAxi:
            return Pot.Rforce(R,phi=phi)
        else:
            return Pot.Rforce(R)
    else:
        raise PotentialError("Input to 'evaluateplanarRforces' is neither a Potential-instance or a list of such instances")

def evaluateplanarphiforces(R,Pot,phi=0.):
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
    isList= isinstance(Pot,list)
    nonAxi= ((isList and Pot[0].isNonAxi) or (not isList and Pot.isNonAxi))
    if nonAxi and not hasphi:
        raise PotentialError("The (list of) planarPotential instances is non-axisymmetric, but you did not provide phi")
    if isinstance(Pot,list):
        sum= 0.
        for pot in Pot:
            if nonAxi:
                sum+= pot.phiforce(R,phi=phi)
            else:
                sum+= pot.phiforce(R)
        return sum
    elif isinstance(Pot,planarPotential):
        if nonAxi:
            return Pot.phiforce(R,phi=phi)
        else:
            return Pot.phiforce(R)
    else:
        raise PotentialError("Input to 'evaluateplanarphiforces' is neither a Potential-instance or a list of such instances")

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
        kwargs['xlabel']=r"$R/R_0$"
        kwargs['ylabel']=r"$\Phi(R)$",
        kwargs['xrange']=Rrange
        return plot.bovy_plot(Rs,potR,*args,**kwargs)
                              
    
