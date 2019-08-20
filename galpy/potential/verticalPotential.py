import numpy
from .linearPotential import linearPotential
from .planarPotential import planarPotential
from .Potential import PotentialError, Potential, flatten
from ..util import bovy_conversion
_APY_LOADED= True
try:
    from astropy import units
except ImportError:
    _APY_LOADED= False
class verticalPotential(linearPotential):
    """Class that represents a vertical potential derived from a 3D Potential:
    phi(z,t;R,phi)= phi(R,z,phi,t)-phi(R,0.,phi,t0)"""
    def __init__(self,Pot,R=1.,phi=None,t0=0.):
        """
        NAME:
           __init__
        PURPOSE:
           Initialize
        INPUT:
           Pot - Potential instance
           R  - Galactocentric radius at which to create the vertical potential
           phi= (None) Galactocentric azimuth at which to create the vertical potential (rad); necessary for 
           t0= (0.) time at which to create the vertical potential
        OUTPUT:
           verticalPotential instance
        HISTORY:
           2010-07-13 - Written - Bovy (NYU)
           2018-10-07 - Added support for non-axi potentials - Bovy (UofT)
           2019-08-19 - Added support for time-dependent potentials - Bovy (UofT)
        """
        linearPotential.__init__(self,amp=1.,ro=Pot._ro,vo=Pot._vo)
        self._Pot= Pot
        self._R= R
        if phi is None:
            if Pot.isNonAxi:
                raise PotentialError("The Potential instance to convert to a verticalPotential is non-axisymmetric, but you did not provide phi")
            self._phi= 0.
        else:
            self._phi= phi
        self._midplanePot= self._Pot(self._R,0.,phi=self._phi,
                                     t=t0,use_physical=False)
        self.hasC= Pot.hasC
        # Also transfer roSet and voSet
        self._roSet= Pot._roSet
        self._voSet= Pot._voSet
        return None

    def _evaluate(self,z,t=0.):
        """
        NAME:
           _evaluate
        PURPOSE:
           evaluate the potential
        INPUT:
           z
           t
        OUTPUT:
          Pot(z,t;R,phi)
        HISTORY:
           2010-07-13 - Written - Bovy (NYU)
        """
        tR= self._R if not hasattr(z,'__len__') else self._R*numpy.ones_like(z)
        tphi= self._phi if not hasattr(z,'__len__') else self._phi*numpy.ones_like(z)
        return self._Pot(tR,z,phi=tphi,t=t,use_physical=False)\
            -self._midplanePot
            
    def _force(self,z,t=0.):
        """
        NAME:
           _force
        PURPOSE:
           evaluate the force
        INPUT:
           z
           t
        OUTPUT:
          F_z(z,t;R,phi)
        HISTORY:
           2010-07-13 - Written - Bovy (NYU)
        """
        tR= self._R if not hasattr(z,'__len__') else self._R*numpy.ones_like(z)
        tphi= self._phi if not hasattr(z,'__len__') else self._phi*numpy.ones_like(z)
        return self._Pot.zforce(tR,z,phi=tphi,t=t,use_physical=False)

def RZToverticalPotential(RZPot,R):
    """
    NAME:

       RZToverticalPotential

    PURPOSE:

       convert a RZPotential to a vertical potential at a given R

    INPUT:

       RZPot - RZPotential instance or list of such instances

       R - Galactocentric radius at which to evaluate the vertical potential (can be Quantity)

    OUTPUT:

       (list of) linearPotential instance(s)

    HISTORY:

       2010-07-21 - Written - Bovy (NYU)

    """
    RZPot= flatten(RZPot)
    if _APY_LOADED and isinstance(R,units.Quantity):
        if hasattr(RZPot,'_ro'):
            R= R.to(units.kpc).value/RZPot._ro
        else:
            R= R.to(units.kpc).value/RZPot[0]._ro
    if isinstance(RZPot,list):
        out= []
        for pot in RZPot:
            if isinstance(pot,linearPotential):
                out.append(pot)
            elif isinstance(pot,Potential):
                out.append(verticalPotential(pot,R))
            elif isinstance(pot,planarPotential):
                raise PotentialError("Input to 'RZToverticalPotential' cannot be a planarPotential")
            else:
                raise PotentialError("Input to 'RZToverticalPotential' is neither an RZPotential-instance or a list of such instances")
        return out
    elif isinstance(RZPot,Potential):
        return verticalPotential(RZPot,R)
    elif isinstance(RZPot,linearPotential):
        return RZPot
    elif isinstance(RZPot,planarPotential):
        raise PotentialError("Input to 'RZToverticalPotential' cannot be a planarPotential")
    else:
        raise PotentialError("Input to 'RZToverticalPotential' is neither an RZPotential-instance or a list of such instances")

def toVerticalPotential(Pot,R,phi=None,t0=0.):
    """
    NAME:

       toVerticalPotential

    PURPOSE:

       convert a Potential to a vertical potential at a given R: Phi(z,phi,t) = Phi(R,z,phi,t)-Phi(R,0.,phi0,t0) where phi0 and t0 are the phi and t inputs

    INPUT:

       Pot - Potential instance or list of such instances

       R - Galactocentric radius at which to evaluate the vertical potential (can be Quantity)

       phi= (None) Galactocentric azimuth at which to evaluate the vertical potential (can be Quantity); required if Pot is non-axisymmetric

       t0= (0.) time at which to evaluate the vertical potential (can be Quantity)

    OUTPUT:

       (list of) linearPotential instance(s)

    HISTORY:

       2018-10-07 - Written - Bovy (UofT)

    """
    Pot= flatten(Pot)
    if _APY_LOADED:
        if isinstance(R,units.Quantity):
            if hasattr(Pot,'_ro'):
                R= R.to(units.kpc).value/Pot._ro
            else:
                R= R.to(units.kpc).value/Pot[0]._ro
        if isinstance(phi,units.Quantity):
            phi= phi.to(units.rad).value
        if isinstance(t0,units.Quantity):
            if hasattr(Pot,'_ro'):
                t0= t0.to(units.Gyr).value/bovy_conversion.time_in_Gyr(Pot._vo,
                                                                       Pot._ro)
            else:
                t0= t0.to(units.Gyr).value\
                    /bovy_conversion.time_in_Gyr(Pot[0]._vo,Pot[0]._ro)
    if isinstance(Pot,list):
        out= []
        for pot in Pot:
            if isinstance(pot,linearPotential):
                out.append(pot)
            elif isinstance(pot,Potential):
                out.append(verticalPotential(pot,R,phi=phi,t0=t0))
            elif isinstance(pot,planarPotential):
                raise PotentialError("Input to 'toVerticalPotential' cannot be a planarPotential")
            else:
                raise PotentialError("Input to 'toVerticalPotential' is neither an RZPotential-instance or a list of such instances")
        return out
    elif isinstance(Pot,Potential):
        return verticalPotential(Pot,R,phi=phi,t0=t0)
    elif isinstance(Pot,linearPotential):
        return Pot
    elif isinstance(Pot,planarPotential):
        raise PotentialError("Input to 'toVerticalPotential' cannot be a planarPotential")
    else:
        raise PotentialError("Input to 'toVerticalPotential' is neither an Potential-instance or a list of such instances")

