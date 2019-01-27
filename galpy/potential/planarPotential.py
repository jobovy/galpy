from __future__ import division, print_function

import os
import pickle
import numpy as nu
from scipy import integrate
import galpy.util.bovy_plot as plot
from galpy.util import config
from galpy.util.bovy_conversion import physical_conversion,\
    potential_physical_input, freq_in_Gyr
from .Potential import Potential, PotentialError, lindbladR, flatten
from .plotRotcurve import plotRotcurve
from .plotEscapecurve import _INF, plotEscapecurve
_APY_LOADED= True
try:
    from astropy import units
except ImportError:
    _APY_LOADED= False
class planarPotential(object):
    """Class representing 2D (R,\phi) potentials"""
    def __init__(self,amp=1.,ro=None,vo=None):
        self._amp= amp
        self.dim= 2
        self.isNonAxi= True #Gets reset by planarAxiPotential
        self.isRZ= False
        self.hasC= False
        self.hasC_dxdv= False
        # Parse ro and vo
        if ro is None:
            self._ro= config.__config__.getfloat('normalization','ro')
            self._roSet= False
        else:
            if _APY_LOADED and isinstance(ro,units.Quantity):
                ro= ro.to(units.kpc).value
            self._ro= ro
            self._roSet= True
        if vo is None:
            self._vo= config.__config__.getfloat('normalization','vo')
            self._voSet= False
        else:
            if _APY_LOADED and isinstance(vo,units.Quantity):
                vo= vo.to(units.km/units.s).value
            self._vo= vo
            self._voSet= True
        return None

    def turn_physical_off(self):
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
        self._roSet= False
        self._voSet= False
        return None

    def turn_physical_on(self,ro=None,vo=None):
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
        self._roSet= True
        self._voSet= True
        if not ro is None:
            if _APY_LOADED and isinstance(ro,units.Quantity):
                ro= ro.to(units.kpc).value
            self._ro= ro
        if not vo is None:
            if _APY_LOADED and isinstance(vo,units.Quantity):
                vo= vo.to(units.km/units.s).value
            self._vo= vo
        return None

    @potential_physical_input
    @physical_conversion('energy',pop=True)
    def __call__(self,R,phi=0.,t=0.,dR=0,dphi=0):
        """
        NAME:

           __call__

        PURPOSE:

           evaluate the potential

        INPUT: 

           R - Cylindrica radius (can be Quantity)

           phi= azimuth (optional; can be Quantity)

           t= time (optional; can be Quantity)

        OUTPUT:

           Phi(R(,phi,t)))

        HISTORY:

           2010-07-13 - Written - Bovy (NYU)

        """
        return self._call_nodecorator(R,phi=phi,t=t,dR=dR,dphi=dphi)

    def _call_nodecorator(self,R,phi=0.,t=0.,dR=0,dphi=0):
        # Separate, so it can be used during orbit integration
        if dR == 0 and dphi == 0:
            try:
                return self._amp*self._evaluate(R,phi=phi,t=t)
            except AttributeError: #pragma: no cover
                raise PotentialError("'_evaluate' function not implemented for this potential")
        elif dR == 1 and dphi == 0:
            return -self.Rforce(R,phi=phi,t=t,use_physical=False)
        elif dR == 0 and dphi == 1:
            return -self.phiforce(R,phi=phi,t=t,use_physical=False)
        elif dR == 2 and dphi == 0:
            return self.R2deriv(R,phi=phi,t=t,use_physical=False)
        elif dR == 0 and dphi == 2:
            return self.phi2deriv(R,phi=phi,t=t,use_physical=False)
        elif dR == 1 and dphi == 1:
            return self.Rphideriv(R,phi=phi,t=t,use_physical=False)
        elif dR != 0 or dphi != 0:
            raise NotImplementedError('Higher-order derivatives not implemented for this potential')

    @potential_physical_input
    @physical_conversion('force',pop=True)
    def Rforce(self,R,phi=0.,t=0.):
        """
        NAME:

           Rforce

        PURPOSE:

           evaluate the radial force

        INPUT:

           R - Cylindrical radius (can be Quantity)

           phi= azimuth (optional; can be Quantity)

           t= time (optional; can be Quantity)

        OUTPUT:

           F_R(R,(\phi,t)))

        HISTORY:

           2010-07-13 - Written - Bovy (NYU)

        """
        return self._Rforce_nodecorator(R,phi=phi,t=t)

    def _Rforce_nodecorator(self,R,phi=0.,t=0.):
        # Separate, so it can be used during orbit integration
        try:
            return self._amp*self._Rforce(R,phi=phi,t=t)
        except AttributeError: #pragma: no cover
            raise PotentialError("'_Rforce' function not implemented for this potential")

    @potential_physical_input
    @physical_conversion('force',pop=True)
    def phiforce(self,R,phi=0.,t=0.):
        """
        NAME:

           phiforce

        PURPOSE:

           evaluate the phi force

        INPUT:

           R - Cylindrical radius (can be Quantity)

           phi= azimuth (optional; can be Quantity)

           t= time (optional; can be Quantity)

        OUTPUT:

           F_phi(R,(phi,t)))

        HISTORY:

           2010-07-13 - Written - Bovy (NYU)

        """
        return self._phiforce_nodecorator(R,phi=phi,t=t)
       
    def _phiforce_nodecorator(self,R,phi=0.,t=0.):
        # Separate, so it can be used during orbit integration
        try:
            return self._amp*self._phiforce(R,phi=phi,t=t)
        except AttributeError: #pragma: no cover
            raise PotentialError("'_phiforce' function not implemented for this potential")

    @potential_physical_input
    @physical_conversion('forcederivative',pop=True)
    def R2deriv(self,R,phi=0.,t=0.):
        """
        NAME:

           R2deriv

        PURPOSE:

           evaluate the second radial derivative

        INPUT:

           R - Cylindrical radius (can be Quantity)

           phi= azimuth (optional; can be Quantity)

           t= time (optional; can be Quantity)

        OUTPUT:

           d2phi/dR2

        HISTORY:

           2011-10-09 - Written - Bovy (IAS)

        """
        try:
            return self._amp*self._R2deriv(R,phi=phi,t=t)
        except AttributeError: #pragma: no cover
            raise PotentialError("'_R2deriv' function not implemented for this potential")      

    @potential_physical_input
    @physical_conversion('forcederivative',pop=True)
    def phi2deriv(self,R,phi=0.,t=0.):
        """
        NAME:

           phi2deriv

        PURPOSE:

           evaluate the second azimuthal derivative

        INPUT:

           R - Cylindrical radius (can be Quantity)

           phi= azimuth (optional; can be Quantity)

           t= time (optional; can be Quantity)

        OUTPUT:

           d2phi/daz2

        HISTORY:

           2014-04-06 - Written - Bovy (IAS)

        """
        try:
            return self._amp*self._phi2deriv(R,phi=phi,t=t)
        except AttributeError: #pragma: no cover
            raise PotentialError("'_phi2deriv' function not implemented for this potential")      

    @potential_physical_input
    @physical_conversion('forcederivative',pop=True)
    def Rphideriv(self,R,phi=0.,t=0.):
        """
        NAME:

           Rphideriv

        PURPOSE:

           evaluate the mixed radial and azimuthal  derivative

        INPUT:

           R - Cylindrical radius (can be Quantity)

           phi= azimuth (optional can be Quantity)

           t= time (optional; can be Quantity)

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
           Rrange - range (can be Quantity)
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
    def __init__(self,amp=1.,ro=None,vo=None):
        planarPotential.__init__(self,amp=amp,ro=ro,vo=vo)
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

    @potential_physical_input
    @physical_conversion('velocity',pop=True)
    def vcirc(self,R,phi=None):
        """
        
        NAME:
        
            vcirc
        
        PURPOSE:
        
            calculate the circular velocity at R in potential Pot

        INPUT:
        
            Pot - Potential instance or list of such instances
        
            R - Galactocentric radius (can be Quantity)
        
            phi= (None) azimuth to use for non-axisymmetric potentials

        OUTPUT:
        
            circular rotation velocity
        
        HISTORY:
        
            2011-10-09 - Written - Bovy (IAS)
        
            2016-06-15 - Added phi= keyword for non-axisymmetric potential - Bovy (UofT)

        """
        return nu.sqrt(R*-self.Rforce(R,phi=phi,use_physical=False))

    @potential_physical_input
    @physical_conversion('frequency',pop=True)
    def omegac(self,R):
        """
        
        NAME:
        
            omegac
        
        PURPOSE:
        
            calculate the circular angular speed at R in potential Pot

        INPUT:
        
            Pot - Potential instance or list of such instances
        
            R - Galactocentric radius (can be Quantity)
        
        OUTPUT:
        
            circular angular speed
        
        HISTORY:
        
            2011-10-09 - Written - Bovy (IAS)
        
        """
        return nu.sqrt(-self.Rforce(R,use_physical=False)/R)       

    @potential_physical_input
    @physical_conversion('frequency',pop=True)
    def epifreq(self,R):
        """
        
        NAME:
        
           epifreq
        
        PURPOSE:
        
           calculate the epicycle frequency at R in this potential
        
        INPUT:
        
           R - Galactocentric radius (can be Quantity)
        
        OUTPUT:
        
           epicycle frequency
        
        HISTORY:
        
           2011-10-09 - Written - Bovy (IAS)
        
        """
        return nu.sqrt(self.R2deriv(R,use_physical=False)
                       -3./R*self.Rforce(R,use_physical=False))

    @physical_conversion('position',pop=True)
    def lindbladR(self,OmegaP,m=2,**kwargs):
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
        
        OUTPUT:
        
           radius of Linblad resonance, None if there is no resonance
        
        HISTORY:
        
           2011-10-09 - Written - Bovy (IAS)
        
        """
        if _APY_LOADED and isinstance(OmegaP,units.Quantity):
            OmegaP= OmegaP.to(1/units.Gyr).value/freq_in_Gyr(self._vo,self._ro)
        return lindbladR(self,OmegaP,m=m,use_physical=False,**kwargs)

    @potential_physical_input
    @physical_conversion('velocity',pop=True)
    def vesc(self,R):
        """

        NAME:

            vesc

        PURPOSE:

            calculate the escape velocity at R for potential Pot

        INPUT:

            Pot - Potential instances or list thereof

            R - Galactocentric radius (can be Quantity)

        OUTPUT:

            escape velocity

        HISTORY:

            2011-10-09 - Written - Bovy (IAS)

        """
        return nu.sqrt(2.*(self(_INF,use_physical=False)
                           -self(R,use_physical=False)))
        
    def plotRotcurve(self,*args,**kwargs):
        """
        NAME:

           plotRotcurve

        PURPOSE:

           plot the rotation curve for this potential

        INPUT:

           Rrange - range (can be Quantity)

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

           Rrange - range (can be Quantity)

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
        planarAxiPotential.__init__(self,amp=1.,ro=RZPot._ro,vo=RZPot._vo)
        # Also transfer roSet and voSet
        self._roSet= RZPot._roSet
        self._voSet= RZPot._voSet
        self._Pot= RZPot
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
        return self._Pot(R,0.,t=t,use_physical=False)
            
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
        return self._Pot.Rforce(R,0.,t=t,use_physical=False)

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
        return self._Pot.R2deriv(R,0.,t=t,use_physical=False)
            
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
    RZPot= flatten(RZPot)
    if isinstance(RZPot,list):
        out= []
        for pot in RZPot:
            if isinstance(pot,planarPotential) and not pot.isNonAxi:
                out.append(pot)
            elif isinstance(pot,Potential) and not pot.isNonAxi:
                out.append(planarPotentialFromRZPotential(pot))
            else:
                raise PotentialError("Input to 'RZToplanarPotential' is neither an RZPotential-instance or a list of such instances")
        return out
    elif isinstance(RZPot,Potential) and not RZPot.isNonAxi:
        return planarPotentialFromRZPotential(RZPot)
    elif isinstance(RZPot,planarPotential) and not RZPot.isNonAxi:
        return RZPot
    else:
        raise PotentialError("Input to 'RZToplanarPotential' is neither an RZPotential-instance or a list of such instances")

class planarPotentialFromFullPotential(planarPotential):
    """Class that represents a planar potential derived from a non-axisymmetric
    3D potential"""
    def __init__(self,Pot):
        """
        NAME:
           __init__
        PURPOSE:
           Initialize
        INPUT:
           Pot - Potential instance
        OUTPUT:
           planarPotential instance
        HISTORY:
           2016-06-02 - Written - Bovy (UofT)
        """
        planarPotential.__init__(self,amp=1.,ro=Pot._ro,vo=Pot._vo)
        # Also transfer roSet and voSet
        self._roSet= Pot._roSet
        self._voSet= Pot._voSet
        self._Pot= Pot
        self.hasC= Pot.hasC
        self.hasC_dxdv= Pot.hasC_dxdv
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
           2016-06-02 - Written - Bovy (UofT)
        """
        return self._Pot(R,0.,phi=phi,t=t,use_physical=False)
            
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
           2016-06-02 - Written - Bovy (UofT)
        """
        return self._Pot.Rforce(R,0.,phi=phi,t=t,use_physical=False)

    def _phiforce(self,R,phi=0.,t=0.):
        """
        NAME:
           _phiforce
        PURPOSE:
           evaluate the azimuthal force
        INPUT:
           R
           phi
           t
        OUTPUT:
          F_phi(R(,\phi,t))
        HISTORY:
           2016-06-02 - Written - Bovy (UofT)
        """
        return self._Pot.phiforce(R,0.,phi=phi,t=t,use_physical=False)

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
           2016-06-02 - Written - Bovy (UofT)
        """
        return self._Pot.R2deriv(R,0.,phi=phi,t=t,use_physical=False)
            
    def _phi2deriv(self,R,phi=0.,t=0.):
        """
        NAME:
           _phi2deriv
        PURPOSE:
           evaluate the second azimuthal derivative
        INPUT:
           R
           phi
           t
        OUTPUT:
           d2phi/dphi2
        HISTORY:
           2016-06-02 - Written - Bovy (UofT)
        """
        return self._Pot.phi2deriv(R,0.,phi=phi,t=t,use_physical=False)
            
    def _Rphideriv(self,R,phi=0.,t=0.):
        """
        NAME:
           _Rphideriv
        PURPOSE:
           evaluate the mixed radial-azimuthal derivative
        INPUT:
           R
           phi
           t
        OUTPUT:
           d2phi/dRdphi
        HISTORY:
           2016-06-02 - Written - Bovy (UofT)
        """
        return self._Pot.Rphideriv(R,0.,phi=phi,t=t,use_physical=False)
            
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
        return self._Pot.OmegaP()
            
def toPlanarPotential(Pot):
    """
    NAME:

       toPlanarPotential

    PURPOSE:

       convert an Potential to a planarPotential in the mid-plane (z=0)

    INPUT:

       Pot - Potential instance or list of such instances (existing planarPotential instances are just copied to the output)

    OUTPUT:

       planarPotential instance(s)

    HISTORY:

       2016-06-11 - Written - Bovy (UofT)

    """
    Pot= flatten(Pot)
    if isinstance(Pot,list):
        out= []
        for pot in Pot:
            if isinstance(pot,planarPotential):
                out.append(pot)
            elif isinstance(pot,Potential) and pot.isNonAxi:
                out.append(planarPotentialFromFullPotential(pot))
            elif isinstance(pot,Potential):
                out.append(planarPotentialFromRZPotential(pot))
            else:
                raise PotentialError("Input to 'toPlanarPotential' is neither an Potential-instance or a list of such instances")
        return out
    elif isinstance(Pot,Potential) and Pot.isNonAxi:
        return planarPotentialFromFullPotential(Pot)
    elif isinstance(Pot,Potential):
        return planarPotentialFromRZPotential(Pot)
    elif isinstance(Pot,planarPotential):
        return Pot
    else:
        raise PotentialError("Input to 'toPlanarPotential' is neither an Potential-instance or a list of such instances")

@potential_physical_input
@physical_conversion('energy',pop=True)
def evaluateplanarPotentials(Pot,R,phi=None,t=0.,dR=0,dphi=0):
    """
    NAME:

       evaluateplanarPotentials

    PURPOSE:

       evaluate a (list of) planarPotential instance(s)

    INPUT:

       Pot - (list of) planarPotential instance(s)

       R - Cylindrical radius (can be Quantity)

       phi= azimuth (optional; can be Quantity)

       t= time (optional; can be Quantity)

       dR=, dphi= if set to non-zero integers, return the dR,dphi't derivative instead

    OUTPUT:

       Phi(R(,phi,t))

    HISTORY:

       2010-07-13 - Written - Bovy (NYU)

    """
    return _evaluateplanarPotentials(Pot,R,phi=phi,t=t,dR=dR,dphi=dphi)

def _evaluateplanarPotentials(Pot,R,phi=None,t=0.,dR=0,dphi=0):
    from .Potential import _isNonAxi
    isList= isinstance(Pot,list)
    nonAxi= _isNonAxi(Pot)
    if nonAxi and phi is None:
        raise PotentialError("The (list of) planarPotential instances is non-axisymmetric, but you did not provide phi")
    if isList and nu.all([isinstance(p,planarPotential) for p in Pot]):
        sum= 0.
        for pot in Pot:
            if nonAxi:
                sum+= pot._call_nodecorator(R,phi=phi,t=t,dR=dR,dphi=dphi)
            else:
                sum+= pot._call_nodecorator(R,t=t,dR=dR,dphi=dphi)
        return sum
    elif isinstance(Pot,planarPotential):
        if nonAxi:
            return Pot._call_nodecorator(R,phi=phi,t=t,dR=dR,dphi=dphi)
        else:
            return Pot._call_nodecorator(R,t=t,dR=dR,dphi=dphi)
    else: #pragma: no cover 
        raise PotentialError("Input to 'evaluatePotentials' is neither a Potential-instance or a list of such instances")

@potential_physical_input
@physical_conversion('force',pop=True)
def evaluateplanarRforces(Pot,R,phi=None,t=0.):
    """
    NAME:

       evaluateplanarRforces

    PURPOSE:

       evaluate the Rforce of a (list of) planarPotential instance(s)

    INPUT:

       Pot - (list of) planarPotential instance(s)

       R - Cylindrical radius (can be Quantity)

       phi= azimuth (optional can be Quantity)

       t= time (optional; can be Quantity)

    OUTPUT:

       F_R(R(,phi,t))

    HISTORY:

       2010-07-13 - Written - Bovy (NYU)

    """
    return _evaluateplanarRforces(Pot,R,phi=phi,t=t)

def _evaluateplanarRforces(Pot,R,phi=None,t=0.):
    """Raw, undecorated function for internal use"""
    from .Potential import _isNonAxi
    isList= isinstance(Pot,list)
    nonAxi= _isNonAxi(Pot)
    if nonAxi and phi is None:
        raise PotentialError("The (list of) planarPotential instances is non-axisymmetric, but you did not provide phi")
    if isinstance(Pot,list) \
            and nu.all([isinstance(p,planarPotential) for p in Pot]):
        sum= 0.
        for pot in Pot:
            if nonAxi:
                sum+= pot._Rforce_nodecorator(R,phi=phi,t=t)
            else:
                sum+= pot._Rforce_nodecorator(R,t=t)
        return sum
    elif isinstance(Pot,planarPotential):
        if nonAxi:
            return Pot._Rforce_nodecorator(R,phi=phi,t=t)
        else:
            return Pot._Rforce_nodecorator(R,t=t)
    else: #pragma: no cover 
        raise PotentialError("Input to 'evaluatePotentials' is neither a Potential-instance or a list of such instances")

@potential_physical_input
@physical_conversion('force',pop=True)
def evaluateplanarphiforces(Pot,R,phi=None,t=0.):
    """
    NAME:

       evaluateplanarphiforces

    PURPOSE:

       evaluate the phiforce of a (list of) planarPotential instance(s)

    INPUT:

       Pot - (list of) planarPotential instance(s)

       R - Cylindrical radius (can be Quantity)

       phi= azimuth (optional; can be Quantity)

       t= time (optional; can be Quantity)

    OUTPUT:

       F_phi(R(,phi,t))

    HISTORY:

       2010-07-13 - Written - Bovy (NYU)

    """
    return _evaluateplanarphiforces(Pot,R,phi=phi,t=t)

def _evaluateplanarphiforces(Pot,R,phi=None,t=0.):
    from .Potential import _isNonAxi
    isList= isinstance(Pot,list)
    nonAxi= _isNonAxi(Pot)
    if nonAxi and phi is None:
        raise PotentialError("The (list of) planarPotential instances is non-axisymmetric, but you did not provide phi")
    if isinstance(Pot,list) \
            and nu.all([isinstance(p,planarPotential) for p in Pot]):
        sum= 0.
        for pot in Pot:
            if nonAxi:
                sum+= pot._phiforce_nodecorator(R,phi=phi,t=t)
            else:
                sum+= pot._phiforce_nodecorator(R,t=t)
        return sum
    elif isinstance(Pot,planarPotential):
        if nonAxi:
            return Pot._phiforce_nodecorator(R,phi=phi,t=t)
        else:
            return Pot._phiforce_nodecorator(R,t=t)
    else: #pragma: no cover 
        raise PotentialError("Input to 'evaluatePotentials' is neither a Potential-instance or a list of such instances")

@potential_physical_input
@physical_conversion('forcederivative',pop=True)
def evaluateplanarR2derivs(Pot,R,phi=None,t=0.):
    """
    NAME:

       evaluateplanarR2derivs

    PURPOSE:

       evaluate the second radial derivative of a (list of) planarPotential instance(s)

    INPUT:

       Pot - (list of) planarPotential instance(s)

       R - Cylindrical radius (can be Quantity)

       phi= azimuth (optional; can be Quantity)

       t= time (optional; can be Quantity)

    OUTPUT:

       F_R(R(,phi,t))

    HISTORY:

       2010-10-09 - Written - Bovy (IAS)

    """
    from .Potential import _isNonAxi
    isList= isinstance(Pot,list)
    nonAxi= _isNonAxi(Pot)
    if nonAxi and phi is None:
        raise PotentialError("The (list of) planarPotential instances is non-axisymmetric, but you did not provide phi")
    if isinstance(Pot,list) \
            and nu.all([isinstance(p,planarPotential) for p in Pot]):
        sum= 0.
        for pot in Pot:
            if nonAxi:
                sum+= pot.R2deriv(R,phi=phi,t=t,use_physical=False)
            else:
                sum+= pot.R2deriv(R,t=t,use_physical=False)
        return sum
    elif isinstance(Pot,planarPotential):
        if nonAxi:
            return Pot.R2deriv(R,phi=phi,t=t,use_physical=False)
        else:
            return Pot.R2deriv(R,t=t,use_physical=False)
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

       R - Cylindrical radius (can be Quantity)
       
       sigmar - radial velocity dispersion of the population (can be Quantity)

       Then either provide:

       1) m= m in the perturbation's m x phi (number of arms for a spiral)

          k= wavenumber (see Binney & Tremaine 2008)

          OmegaP= pattern speed (can be Quantity)

       2) nonaxiPot= a non-axisymmetric Potential instance (such as SteadyLogSpiralPotential) that has functions that return OmegaP, m, and wavenumber

    OUTPUT:

       reduction factor

    HISTORY:

       2014-08-23 - Written - Bovy (IAS)

    """
    axiPot= flatten(axiPot)
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

       Rrange - range (can be Quantity)

       xrange, yrange - if relevant (can be Quantity)

       grid, gridx, gridy - number of points to plot

       savefilename - save to or restore from this savefile (pickle)

       ncontours - number of contours to plot (if applicable)

       +bovy_plot(*args,**kwargs) or bovy_dens2d(**kwargs)

    OUTPUT:

       plot to output device

    HISTORY:

       2010-07-13 - Written - Bovy (NYU)

    """
    Pot= flatten(Pot)
    Rrange= kwargs.pop('Rrange',[0.01,5.])
    xrange= kwargs.pop('xrange',[-5.,5.])
    yrange= kwargs.pop('yrange',[-5.,5.])
    if _APY_LOADED:
        if hasattr(Pot,'_ro'):
            tro= Pot._ro
        else:
            tro= Pot[0]._ro
        if isinstance(Rrange[0],units.Quantity):
            Rrange[0]= Rrange[0].to(units.kpc).value/tro
        if isinstance(Rrange[1],units.Quantity):
            Rrange[1]= Rrange[1].to(units.kpc).value/tro
        if isinstance(xrange[0],units.Quantity):
            xrange[0]= xrange[0].to(units.kpc).value/tro
        if isinstance(xrange[1],units.Quantity):
            xrange[1]= xrange[1].to(units.kpc).value/tro
        if isinstance(yrange[0],units.Quantity):
            yrange[0]= yrange[0].to(units.kpc).value/tro
        if isinstance(yrange[1],units.Quantity):
            yrange[1]= yrange[1].to(units.kpc).value/tro
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
                    potR[ii,jj]= evaluateplanarPotentials(Pot,thisR,
                                                          phi=thisphi,
                                                          use_physical=False)
        else:
            Rs= nu.linspace(Rrange[0],Rrange[1],grid)
            potR= nu.zeros(grid)
            for ii in range(grid):
                potR[ii]= evaluateplanarPotentials(Pot,Rs[ii],
                                                   use_physical=False)
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
                              
    
