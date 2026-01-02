###############################################################################
#   plabarForce.py: top-level class for a 2D force, conservative
#                   (planarPotential) or not (planarDissipativeForce)
#
###############################################################################
import copy

import numpy

from ..util import config, conversion
from ..util._optional_deps import _APY_LOADED
from ..util.conversion import (
    physical_compatible,
    physical_conversion,
    potential_physical_input,
)
from ._repr_utils import _build_repr
from .Force import Force

if _APY_LOADED:
    from astropy import units


class planarForce:
    """Top-level class for any 2D force, conservative or dissipative"""

    def __init__(self, amp=1.0, ro=None, vo=None):
        """
        Initialize 2D Force.

        Parameters
        ----------
        amp : float
            Amplitude to be applied when evaluating the potential and its forces.
        ro : float or Quantity, optional
            Physical distance scale (in kpc or as Quantity). Default is from the configuration file.
        vo : float or Quantity, optional
            Physical velocity scale (in km/s or as Quantity). Default is from the configuration file.

        Notes
        -----
        - 2023-05-29 - Written to generalize planarPotential to force that may or may not be conservative - Bovy (UofT)
        """
        self._amp = amp
        self.dim = 2
        self.isNonAxi = True
        self.isRZ = False
        self.hasC = False
        self.hasC_dxdv = False
        self.hasC_dens = False
        # Parse ro and vo
        if ro is None:
            self._ro = config.__config__.getfloat("normalization", "ro")
            self._roSet = False
        else:
            self._ro = conversion.parse_length_kpc(ro)
            self._roSet = True
        if vo is None:
            self._vo = config.__config__.getfloat("normalization", "vo")
            self._voSet = False
        else:
            self._vo = conversion.parse_velocity_kms(vo)
            self._voSet = True
        return None

    def __repr__(self):
        """
        Return a string representation of the planarForce instance.

        Returns
        -------
        str
            String representation showing the class name, internal parameters, and physical output status.

        Notes
        -----
        - 2025-12-09 - Written - Bovy (UofT)

        """
        return _build_repr(self)

    def __mul__(self, b):
        """
        Multiply a planarPotential's amplitude by a number.

        Parameters
        ----------
        b : number
            The number to multiply the amplitude by.

        Returns
        -------
        planarPotential instance
            A new instance with amplitude = (old amplitude) x b.

        Notes
        -----
        - 2019-01-27: Written - Bovy (UofT)

        """
        if not isinstance(b, (int, float)):
            raise TypeError(
                "Can only multiply a planarPotential instance with a number"
            )
        out = copy.deepcopy(self)
        out._amp *= b
        return out

    # Similar functions
    __rmul__ = __mul__

    def __div__(self, b):
        return self.__mul__(1.0 / b)

    __truediv__ = __div__

    def __add__(self, b):
        """
        Add planarPotential instances together to create a multi-component potential (e.g., pot= pot1+pot2+pot3)

        Parameters
        ----------
        b : planarForce instance or a combination thereof

        Returns
        -------
        planarCompositePotential
            Represents the combined potential

        Notes
        -----
        - 2019-01-27 - Written - Bovy (UofT)
        - 2024-11-27 - Updated to return planarCompositePotential - Copilot

        """
        from ..potential import flatten as flatten_pot
        from .planarCompositePotential import planarCompositePotential

        if not isinstance(flatten_pot([b])[0], (Force, planarForce)):
            raise TypeError(
                """Can only combine galpy Potential"""
                """/planarPotential objects with """
                """other such objects or combinations thereof"""
            )

        # If adding a 3D Force, convert it to planar
        if isinstance(b, Force) and hasattr(b, "dim") and b.dim == 3:
            return planarCompositePotential(self, b.toPlanar())

        # Physical compatibility is checked in planarCompositePotential.__init__
        return planarCompositePotential(self, b)

    # Define separately to catch errors
    def __radd__(self, b):
        from ..potential import flatten as flatten_pot
        from .planarCompositePotential import planarCompositePotential

        if not isinstance(flatten_pot([b])[0], (Force, planarForce)):
            raise TypeError(
                """Can only combine galpy Force objects with """
                """other Force objects or combinations thereof"""
            )

        # Can't add anything that isn't handled elsewhere, so no further code here

    def turn_physical_off(self):
        """
        Turn off automatic returning of outputs in physical units.

        Returns
        -------
        None

        Notes
        -----
        - 2016-01-30 - Written - Bovy (UofT)

        """
        self._roSet = False
        self._voSet = False
        return None

    def turn_physical_on(self, ro=None, vo=None):
        """
        Turn on automatic returning of outputs in physical units.

        Parameters
        ----------
        ro : float or Quantity, optional
            Reference distance in kpc. Default is None.
        vo : float or Quantity, optional
            Reference velocity in km/s. Default is None.

        Returns
        -------
        None

        Notes
        -----
        - 2016-01-30 - Written - Bovy (UofT)
        - 2020-04-22 - Don't turn on a parameter when it is False - Bovy (UofT)

        """
        if not ro is False:
            self._roSet = True
            ro = conversion.parse_length_kpc(ro)
            if not ro is None:
                self._ro = ro
        if not vo is False:
            self._voSet = True
            vo = conversion.parse_velocity_kms(vo)
            if not vo is None:
                self._vo = vo
        return None

    def _Rforce_nodecorator(self, R, phi=0.0, t=0.0, **kwargs):
        # Separate, so it can be used during orbit integration
        try:
            return self._amp * self._Rforce(R, phi=phi, t=t, **kwargs)
        except AttributeError:  # pragma: no cover
            from .Potential import PotentialError

            raise PotentialError(
                "'_Rforce' function not implemented for this planarDissipativeForce"
            )

    def _phitorque_nodecorator(self, R, phi=0.0, t=0.0, **kwargs):
        # Separate, so it can be used during orbit integration
        try:
            return self._amp * self._phitorque(R, phi=phi, t=t, **kwargs)
        except AttributeError:  # pragma: no cover
            if self.isNonAxi:
                from .Potential import PotentialError

                raise PotentialError(
                    "'_phitorque' function not implemented for this DissipativeForce"
                )
            return 0.0
