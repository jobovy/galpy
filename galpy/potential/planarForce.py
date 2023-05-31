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
from .Force import Force

if _APY_LOADED:
    from astropy import units


class planarForce:
    """Top-level class for any 2D force, conservative or dissipative"""

    def __init__(self, amp=1.0, ro=None, vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           Initialize 2D Force

        INPUT:

           amp - amplitude to be applied when evaluating the potential and its forces

           ro - physical distance scale (in kpc or as Quantity)

           vo - physical velocity scale (in km/s or as Quantity)

        OUTPUT:

        HISTORY:
           2023-05-29 - Written to generalize planarPotential to force that may or may not be conservative - Bovy (UofT)
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

    def __mul__(self, b):
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

        if not isinstance(flatten_pot([b])[0], (Force, planarForce)):
            raise TypeError(
                """Can only combine galpy Potential"""
                """/planarPotential objects with """
                """other such objects or lists thereof"""
            )
        assert physical_compatible(self, b), (
            """Physical unit conversion parameters (ro,vo) are not """
            """compatible between potentials to be combined"""
        )
        if isinstance(b, list):
            return [self] + b
        else:
            return [self, b]

    # Define separately to keep order
    def __radd__(self, b):
        from ..potential import flatten as flatten_pot

        if not isinstance(flatten_pot([b])[0], (Force, planarForce)):
            raise TypeError(
                """Can only combine galpy Force objects with """
                """other Force objects or lists thereof"""
            )
        assert physical_compatible(self, b), (
            """Physical unit conversion parameters (ro,vo) are not """
            """compatible between potentials to be combined"""
        )
        # If we get here, b has to be a list
        return b + [self]

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
        self._roSet = False
        self._voSet = False
        return None

    def turn_physical_on(self, ro=None, vo=None):
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
