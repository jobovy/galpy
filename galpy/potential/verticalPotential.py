from .linearPotential import linearPotential
from .Potential import PotentialError, Potential
_APY_LOADED= True
try:
    from astropy import units
except ImportError:
    _APY_LOADED= False
class verticalPotential(linearPotential):
    """Class that represents a vertical potential derived from a RZPotential:
    phi(z;R)= phi(R,z)-phi(R,0.)"""
    def __init__(self,RZPot,R=1.):
        """
        NAME:
           __init__
        PURPOSE:
           Initialize
        INPUT:
           RZPot - RZPotential instance
           R  - Galactocentric radius at which to create the vertical potential
        OUTPUT:
           verticalPotential instance
        HISTORY:
           2010-07-13 - Written - Bovy (NYU)
        """
        linearPotential.__init__(self,amp=1.,ro=RZPot._ro,vo=RZPot._vo)
        self._Pot= RZPot
        self._R= R
        # Also transfer roSet and voSet
        self._roSet= RZPot._roSet
        self._voSet= RZPot._voSet
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
          Pot(z,t;R)
        HISTORY:
           2010-07-13 - Written - Bovy (NYU)
        """
        return self._Pot(self._R,z,t=t,use_physical=False)\
            -self._Pot(self._R,0.,t=t,use_physical=False)
            
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
          F_z(z,t;R)
        HISTORY:
           2010-07-13 - Written - Bovy (NYU)
        """
        return self._Pot.zforce(self._R,z,t=t,use_physical=False)\
            -self._Pot.zforce(self._R,0.,t=t,use_physical=False)

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
            else:
                out.append(verticalPotential(pot,R))
        return out
    elif isinstance(RZPot,Potential):
        return verticalPotential(RZPot,R)
    elif isinstance(RZPot,linearPotential):
        return RZPot
    else: #pragma: no cover
        raise PotentialError("Input to 'RZToverticalPotential' is neither an RZPotential-instance or a list of such instances")
