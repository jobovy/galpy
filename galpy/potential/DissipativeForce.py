###############################################################################
#   DissipativeForce.py: top-level class for non-conservative forces
###############################################################################
import numpy
from .Force import Force
from galpy.util.bovy_conversion import physical_conversion, \
    potential_physical_input
class DissipativeForce(Force):
    """Top-level class for non-conservative forces (cannot be derived from a potential function)"""
    def __init__(self,amp=1.,ro=None,vo=None,amp_units=None):
        """
        NAME:
           __init__
        PURPOSE:
        INPUT:
           amp - amplitude to be applied when evaluating the potential and its forces
        OUTPUT:
        HISTORY:
           2018-03-16 - Started - Bovy (UofT)
        """
        Force.__init__(self,amp=amp,ro=ro,vo=vo,amp_units=amp_units)
        self.dim= 3
        self.isNonAxi= True # Default: are non-axisymmetric
        self.hasC= False
        self.hasC_dxdv= False

    @potential_physical_input
    @physical_conversion('force',pop=True)
    def Rforce(self,R,z,phi=0.,t=0.,v=None):
        """
        NAME:

           Rforce

        PURPOSE:

           evaluate cylindrical radial force F_R  (R,z)

        INPUT:

           R - Cylindrical Galactocentric radius (can be Quantity)

           z - vertical height (can be Quantity)

           phi - azimuth (optional; can be Quantity)

           t - time (optional; can be Quantity)

           v - 3d velocity (optional; can be Quantity)

        OUTPUT:

           F_R (R,z,phi,t,v)

        HISTORY:

           2018-03-18 - Written - Bovy (UofT)

        """
        return self._Rforce_nodecorator(R,z,phi=phi,t=t,v=v)

    def _Rforce_nodecorator(self,R,z,phi=0.,t=0.,v=None):
        # Separate, so it can be used during orbit integration
        try:
            return self._amp*self._Rforce(R,z,phi=phi,t=t,v=v)
        except AttributeError: #pragma: no cover
            raise
            from .Potential import PotentialError
            raise PotentialError("'_Rforce' function not implemented for this DissipativeForce")
        
    @potential_physical_input
    @physical_conversion('force',pop=True)
    def zforce(self,R,z,phi=0.,t=0.,v=None):
        """
        NAME:

           zforce

        PURPOSE:

           evaluate the vertical force F_z  (R,z,t)

        INPUT:

           R - Cylindrical Galactocentric radius (can be Quantity)

           z - vertical height (can be Quantity)

           phi - azimuth (optional; can be Quantity)

           t - time (optional; can be Quantity)

           v - 3d velocity (optional; can be Quantity)

        OUTPUT:

           F_z (R,z,phi,t,v)

        HISTORY:

           2018-03-16 - Written - Bovy (UofT)

        """
        return self._zforce_nodecorator(R,z,phi=phi,t=t,v=v)

    def _zforce_nodecorator(self,R,z,phi=0.,t=0.,v=None):
        # Separate, so it can be used during orbit integration
        try:
            return self._amp*self._zforce(R,z,phi=phi,t=t,v=v)
        except AttributeError: #pragma: no cover
            from .Potential import PotentialError
            raise PotentialError("'_zforce' function not implemented for this DissipativeForce")

    @potential_physical_input
    @physical_conversion('force',pop=True)
    def phiforce(self,R,z,phi=0.,t=0.,v=None):
        """
        NAME:

           phiforce

        PURPOSE:

           evaluate the azimuthal force F_phi  (R,z,phi,t,v)

        INPUT:

           R - Cylindrical Galactocentric radius (can be Quantity)

           z - vertical height (can be Quantity)

           phi - azimuth (rad; can be Quantity)

           t - time (optional; can be Quantity)

           v - 3d velocity (optional; can be Quantity)

        OUTPUT:

           F_phi (R,z,phi,t,v)

        HISTORY:

           2018-03-16 - Written - Bovy (UofT)

        """
        return self._phiforce_nodecorator(R,z,phi=phi,t=t,v=v)

    def _phiforce_nodecorator(self,R,z,phi=0.,t=0.,v=None):
        # Separate, so it can be used during orbit integration
        try:
            return self._amp*self._phiforce(R,z,phi=phi,t=t,v=v)
        except AttributeError: #pragma: no cover
            if self.isNonAxi:
                from .Potential import PotentialError
                raise PotentialError("'_phiforce' function not implemented for this DissipativeForce")
            return 0.

def _isDissipative(obj):
    """
    NAME:

       _isDissipative

    PURPOSE:

       Determine whether this combination of potentials and forces is Dissipative

    INPUT:

       obj - Potential/DissipativeForce instance or list of such instances

    OUTPUT:

       True or False depending on whether the object is dissipative

    HISTORY:

       2018-03-16 - Written - Bovy (UofT)

    """
    from .Potential import flatten
    obj= flatten(obj)
    isList= isinstance(obj,list)
    if isList:
        isCons= [not isinstance(p,DissipativeForce) for p in obj]
        nonCons= not numpy.prod(numpy.array(isCons))
    else:
        nonCons= isinstance(obj,DissipativeForce)
    return nonCons

