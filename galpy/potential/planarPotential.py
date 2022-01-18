from __future__ import division, print_function

import os
import copy
import pickle
import numpy
from scipy import integrate
from ..util import plot, config, conversion
from ..util.conversion import physical_conversion,\
    potential_physical_input, physical_compatible
from .Potential import Potential, PotentialError, lindbladR, flatten
from .DissipativeForce import _isDissipative
from .plotRotcurve import plotRotcurve
from .plotEscapecurve import _INF, plotEscapecurve
class planarPotential(object):
    """Class representing 2D (R,\phi) potentials"""
    def __init__(self,amp=1.,ro=None,vo=None):
        self._amp= amp
        self.dim= 2
        self.isNonAxi= True #Gets reset by planarAxiPotential
        self.isRZ= False
        self.hasC= False
        self.hasC_dxdv= False
        self.hasC_dens= False
        # Parse ro and vo
        if ro is None:
            self._ro= config.__config__.getfloat('normalization','ro')
            self._roSet= False
        else:
            self._ro= conversion.parse_length_kpc(ro)
            self._roSet= True
        if vo is None:
            self._vo= config.__config__.getfloat('normalization','vo')
            self._voSet= False
        else:
            self._vo= conversion.parse_velocity_kms(vo)
            self._voSet= True
        return None

    def __mul__(self,b):
        """
        NAME:

           __mul__

        PURPOSE:

           Multiply a planarPotential's amplitude by a number

        INPUT:

           b - number

        OUTPUT:

           New instance with amplitude = (old amplitude) x b

        HISTORY:

           2019-01-27 - Written - Bovy (UofT)

        """
        if not isinstance(b,(int,float)):
            raise TypeError("Can only multiply a planarPotential instance with a number")
        out= copy.deepcopy(self)
        out._amp*= b
        return out
    # Similar functions
    __rmul__= __mul__
    def __div__(self,b): return self.__mul__(1./b)
    __truediv__= __div__

    def __add__(self,b):
        """
        NAME:

           __add__

        PURPOSE:

           Add planarPotential instances together to create a multi-component potential (e.g., pot= pot1+pot2+pot3)

        INPUT:

           b - planarPotential instance or a list thereof

        OUTPUT:

           List of planarPotential instances that represents the combined potential

        HISTORY:

           2019-01-27 - Written - Bovy (UofT)

        """
        from ..potential import flatten as flatten_pot
        if not isinstance(flatten_pot([b])[0],(Potential,planarPotential)):
            raise TypeError("""Can only combine galpy Potential"""
                            """/planarPotential objects with """
                            """other such objects or lists thereof""")
        assert physical_compatible(self,b), \
            """Physical unit conversion parameters (ro,vo) are not """\
            """compatible between potentials to be combined"""
        if isinstance(b,list):
            return [self]+b
        else:
            return [self,b]
    # Define separately to keep order
    def __radd__(self,b):
        from ..potential import flatten as flatten_pot
        if not isinstance(flatten_pot([b])[0],(Potential,planarPotential)):
            raise TypeError("""Can only combine galpy Force objects with """
                            """other Force objects or lists thereof""")
        assert physical_compatible(self,b), \
            """Physical unit conversion parameters (ro,vo) are not """\
            """compatible between potentials to be combined"""
        # If we get here, b has to be a list
        return b+[self]

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

           2020-04-22 - Don't turn on a parameter when it is False - Bovy (UofT)

        """
        if not ro is False: self._roSet= True
        if not vo is False: self._voSet= True
        if not ro is None and ro:
            self._ro= conversion.parse_length_kpc(ro)
        if not vo is None and vo:
            self._vo= conversion.parse_velocity_kms(vo)
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
    @physical_conversion('energy',pop=True)
    def phiforce(self,R,phi=0.,t=0.):
        """
        NAME:

           phiforce

        PURPOSE:

           evaluate the phi force = - d Phi / d phi (note that this is a torque, not a force!)

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
    @physical_conversion('energy',pop=True)
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
    @physical_conversion('force',pop=True)
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
           +galpy.util.plot.plot(*args,**kwargs)
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
    def vcirc(self,R,phi=None,t=0.):
        """
        
        NAME:
        
            vcirc
        
        PURPOSE:
        
            calculate the circular velocity at R in potential Pot

        INPUT:
        
            Pot - Potential instance or list of such instances
        
            R - Galactocentric radius (can be Quantity)
        
            phi= (None) azimuth to use for non-axisymmetric potentials

            t - time (optional; can be Quantity)

        OUTPUT:
        
            circular rotation velocity
        
        HISTORY:
        
            2011-10-09 - Written - Bovy (IAS)
        
            2016-06-15 - Added phi= keyword for non-axisymmetric potential - Bovy (UofT)

        """
        return numpy.sqrt(R*-self.Rforce(R,phi=phi,t=t,use_physical=False))

    @potential_physical_input
    @physical_conversion('frequency',pop=True)
    def omegac(self,R,t=0.):
        """
        
        NAME:
        
            omegac
        
        PURPOSE:
        
            calculate the circular angular speed at R in potential Pot

        INPUT:
        
            Pot - Potential instance or list of such instances
        
            R - Galactocentric radius (can be Quantity)

            t - time (optional; can be Quantity)
        
        OUTPUT:
        
            circular angular speed
        
        HISTORY:
        
            2011-10-09 - Written - Bovy (IAS)
        
        """
        return numpy.sqrt(-self.Rforce(R,t=t,use_physical=False)/R)       

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
        return numpy.sqrt(self.R2deriv(R,t=t,use_physical=False)
                       -3./R*self.Rforce(R,t=t,use_physical=False))

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

            calculate the escape velocity at R for potential Pot

        INPUT:

            Pot - Potential instances or list thereof

            R - Galactocentric radius (can be Quantity)

            t - time (optional; can be Quantity)

        OUTPUT:

            escape velocity

        HISTORY:

            2011-10-09 - Written - Bovy (IAS)

        """
        return numpy.sqrt(2.*(self(_INF,t=t,use_physical=False)
                           -self(R,t=t,use_physical=False)))
        
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

           +galpy.util.plot.plot(*args,**kwargs)

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

           +galpy.util.plot.plot(*args,**kwargs)

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
        self.hasC_dens= RZPot.hasC_dens
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
    if _isDissipative(RZPot):
        raise NotImplementedError("Converting dissipative forces to 2D potentials is currently not supported")
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
        self.hasC_dens= Pot.hasC_dens
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
    if _isDissipative(Pot):
        raise NotImplementedError("Converting dissipative forces to 2D potentials is currently not supported")
    elif isinstance(Pot,list):
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
    if isList and numpy.all([isinstance(p,planarPotential) for p in Pot]):
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
            and numpy.all([isinstance(p,planarPotential) for p in Pot]):
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
@physical_conversion('energy',pop=True)
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
            and numpy.all([isinstance(p,planarPotential) for p in Pot]):
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
            and numpy.all([isinstance(p,planarPotential) for p in Pot]):
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
    from ..potential import omegac, epifreq
    if nonaxiPot is None and (OmegaP is None or k is None or m is None):
        raise IOError("Need to specify either nonaxiPot= or m=, k=, OmegaP= for LinShuReductionFactor")
    elif not nonaxiPot is None:
        OmegaP= nonaxiPot.OmegaP()
        k= nonaxiPot.wavenumber(R)
        m= nonaxiPot.m()
    tepif= epifreq(axiPot,R)
    s= m*(OmegaP-omegac(axiPot,R))/tepif
    chi= sigmar**2.*k**2./tepif**2.
    return (1.-s**2.)/numpy.sin(numpy.pi*s)\
        *integrate.quad(lambda t: numpy.exp(-chi*(1.+numpy.cos(t)))\
                            *numpy.sin(s*t)*numpy.sin(t),
                        0.,numpy.pi)[0]

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

       +galpy.util.plot.plot(*args,**kwargs) or galpy.util.plot.dens2d(**kwargs)

    OUTPUT:

       plot to output device

    HISTORY:

       2010-07-13 - Written - Bovy (NYU)

    """
    Pot= flatten(Pot)
    Rrange= kwargs.pop('Rrange',[0.01,5.])
    xrange= kwargs.pop('xrange',[-5.,5.])
    yrange= kwargs.pop('yrange',[-5.,5.])
    if hasattr(Pot,'_ro'):
        tro= Pot._ro
    else:
        tro= Pot[0]._ro
    Rrange[0]= conversion.parse_length(Rrange[0],ro=tro)
    Rrange[1]= conversion.parse_length(Rrange[1],ro=tro)
    xrange[0]= conversion.parse_length(xrange[0],ro=tro)
    xrange[1]= conversion.parse_length(xrange[1],ro=tro)
    yrange[0]= conversion.parse_length(yrange[0],ro=tro)
    yrange[1]= conversion.parse_length(yrange[1],ro=tro)
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
            xs= numpy.linspace(xrange[0],xrange[1],gridx)
            ys= numpy.linspace(yrange[0],yrange[1],gridy)
            potR= numpy.zeros((gridx,gridy))
            for ii in range(gridx):
                for jj in range(gridy):
                    thisR= numpy.sqrt(xs[ii]**2.+ys[jj]**2.)
                    if xs[ii] >= 0.:
                        thisphi= numpy.arcsin(ys[jj]/thisR)
                    else:
                        thisphi= -numpy.arcsin(ys[jj]/thisR)+numpy.pi
                    potR[ii,jj]= evaluateplanarPotentials(Pot,thisR,
                                                          phi=thisphi,
                                                          use_physical=False)
        else:
            Rs= numpy.linspace(Rrange[0],Rrange[1],grid)
            potR= numpy.zeros(grid)
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
            kwargs['levels']= numpy.linspace(numpy.nanmin(potR),numpy.nanmax(potR),ncontours)
        return plot.dens2d(potR.T,
                           xrange=xrange,
                           yrange=yrange,**kwargs)
    else:
        kwargs['xlabel']=r"$R/R_0$"
        kwargs['ylabel']=r"$\Phi(R)$"
        kwargs['xrange']=Rrange
        return plot.plot(Rs,potR,*args,**kwargs)
                              
    
