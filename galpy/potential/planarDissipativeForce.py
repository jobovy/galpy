###############################################################################
#   planarDissipativeForce.py: top-level class for non-conservative 2D forces
###############################################################################
import numpy

from ..util.conversion import physical_conversion, potential_physical_input
from .planarForce import planarForce


class planarDissipativeForce(planarForce):
    """Top-level class for non-conservative forces (cannot be derived from a potential function)"""

    def __init__(self, amp=1.0, ro=None, vo=None, amp_units=None):
        """
        NAME:
           __init__
        PURPOSE:
        INPUT:
           amp - amplitude to be applied when evaluating the potential and its forces
        OUTPUT:
        HISTORY:
           2023-05-29 - Started - Bovy (UofT)
        """
        planarForce.__init__(self, amp=amp, ro=ro, vo=vo)

    @potential_physical_input
    @physical_conversion("force", pop=True)
    def Rforce(self, R, phi=0.0, t=0.0, v=None):
        """
        NAME:

           Rforce

        PURPOSE:

           evaluate cylindrical radial force F_R  (R,phi)

        INPUT:

           R - Cylindrical Galactocentric radius (can be Quantity)

           phi - azimuth (optional; can be Quantity)

           t - time (optional; can be Quantity)

           v - 2d cylindrical velocity (optional; can be Quantity)

        OUTPUT:

           F_R (R,phi,t,v)

        HISTORY:

           2023-05-29 - Written - Bovy (UofT)

        """
        return self._Rforce_nodecorator(R, phi=phi, t=t, v=v)

    def _Rforce_nodecorator(self, R, phi=0.0, t=0.0, v=None):
        # Separate, so it can be used during orbit integration
        try:
            return self._amp * self._Rforce(R, phi=phi, t=t, v=v)
        except AttributeError:  # pragma: no cover
            from .Potential import PotentialError

            raise PotentialError(
                "'_Rforce' function not implemented for this planarDissipativeForce"
            )

    @potential_physical_input
    @physical_conversion("force", pop=True)
    def phitorque(self, R, phi=0.0, t=0.0, v=None):
        """
        NAME:

           phitorque

        PURPOSE:

           evaluate the azimuthal torque F_phi  (R,phi,t,v)

        INPUT:

           R - Cylindrical Galactocentric radius (can be Quantity)

           phi - azimuth (rad; can be Quantity)

           t - time (optional; can be Quantity)

           v - 2d cylindrical velocity (optional; can be Quantity)

        OUTPUT:

           F_phi (R,phi,t,v)

        HISTORY:

           2023-05-29 - Written - Bovy (UofT)

        """
        return self._phitorque_nodecorator(R, phi=phi, t=t, v=v)

    def _phitorque_nodecorator(self, R, phi=0.0, t=0.0, v=None):
        # Separate, so it can be used during orbit integration
        try:
            return self._amp * self._phitorque(R, phi=phi, t=t, v=v)
        except AttributeError:  # pragma: no cover
            if self.isNonAxi:
                from .Potential import PotentialError

                raise PotentialError(
                    "'_phitorque' function not implemented for this DissipativeForce"
                )
            return 0.0


class planarDissipativeForceFromFullDissipativeForce(planarDissipativeForce):
    """Class that represents a planar dissipative force derived from a 3D dissipative force"""

    def __init__(self, Pot):
        """
        NAME:
           __init__
        PURPOSE:
           Initialize
        INPUT:
           Pot - DissipativeForce instance
        OUTPUT:
           planarDissipativeForce instance
        HISTORY:
           2023-05-29 - Written - Bovy (UofT)
        """
        planarDissipativeForce.__init__(self, amp=1.0, ro=Pot._ro, vo=Pot._vo)
        # Also transfer roSet and voSet
        self._roSet = Pot._roSet
        self._voSet = Pot._voSet
        self._Pot = Pot
        self.hasC = Pot.hasC
        self.hasC_dxdv = Pot.hasC_dxdv
        self.hasC_dens = Pot.hasC_dens
        return None

    def _Rforce(self, R, phi=0.0, t=0.0, v=None):
        r"""
        NAME:
           _Rforce
        PURPOSE:
           evaluate the radial force
        INPUT:
           R
           phi
           t
           v
        OUTPUT:
          F_R(R(,\phi,t,v))
        HISTORY:
           2023-09-29 - Written - Bovy (UofT)
        """
        return self._Pot.Rforce(
            R, 0.0, phi=phi, t=t, v=[v[0], v[1], 0.0], use_physical=False
        )

    def _phitorque(self, R, phi=0.0, t=0.0, v=None):
        r"""
        NAME:
           _phitorque
        PURPOSE:
           evaluate the azimuthal torque
        INPUT:
           R
           phi
           t
           v
        OUTPUT:
          tau_phi(R(,\phi,t,v))
        HISTORY:
           2029-05-29 - Written - Bovy (UofT)
        """
        return self._Pot.phitorque(
            R, 0.0, phi=phi, t=t, v=[v[0], v[1], 0.0], use_physical=False
        )
