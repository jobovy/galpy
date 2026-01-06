###############################################################################
#   Force.py: top-level class for a 3D force, conservative (Potential) or
#             not (DissipativeForce)
#
###############################################################################
import copy
from typing import Optional, Union

import numpy

from ..util import config, conversion
from ..util._optional_deps import _APY_LOADED
from ..util.conversion import (
    physical_compatible,
    physical_conversion,
    potential_physical_input,
)
from ._repr_utils import _build_repr

if _APY_LOADED:
    from astropy import units


class Force:
    """Top-level class for any force, conservative or dissipative"""

    def __init__(
        self,
        amp: float = 1.0,
        ro: Optional[float] = None,
        vo: Optional[float] = None,
        amp_units: Optional[str] = None,
    ) -> None:
        """
        Initialize Force.

        Parameters
        ----------
        amp : float, optional
            Amplitude to be applied when evaluating the potential and its forces.
        ro : float or Quantity, optional
            Physical distance scale (in kpc or as Quantity). Default is from the configuration file.
        vo : float or Quantity, optional
            Physical velocity scale (in km/s or as Quantity). Default is from the configuration file.
        amp_units : str, optional
            Type of units that `amp` should have if it has units. Must be one of
            'mass', 'velocity2', or 'density'.

        Notes
        -----
        - 2018-03-18 - Written to generalize Potential to force that may or may not be conservative - Bovy (UofT)

        """
        self._amp = amp
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
        # Parse amp if it has units
        if _APY_LOADED and isinstance(self._amp, units.Quantity):
            # Try a bunch of possible units
            unitFound = False
            units_to_try = [
                ("velocity2", lambda a: conversion.parse_energy(a, vo=self._vo)),
                ("mass", lambda a: conversion.parse_mass(a, ro=self._ro, vo=self._vo)),
                (
                    "density",
                    lambda a: conversion.parse_dens(a, ro=self._ro, vo=self._vo),
                ),
                (
                    "surfacedensity",
                    lambda a: conversion.parse_surfdens(a, ro=self._ro, vo=self._vo),
                ),
            ]
            for amp_units_try, parse_func in units_to_try:
                try:
                    self._amp = parse_func(self._amp)
                except units.UnitConversionError:
                    continue
                else:
                    unitFound = True
                    if not amp_units == amp_units_try:
                        raise units.UnitConversionError(
                            f"amp= parameter of {type(self).__name__} should have units of {amp_units}, but has units of {amp_units_try} instead"
                        )
                    break
            if not unitFound:
                raise units.UnitConversionError(
                    f"amp= parameter of {type(self).__name__} should have units of {amp_units}; given units are not understood"
                )
            else:
                # When amplitude is given with units, turn on physical output
                self._roSet = True
                self._voSet = True

    def __repr__(self) -> str:
        """
        Return a string representation of the Force instance.

        Returns
        -------
        str
            String representation showing the class name, internal parameters, and physical output status.

        Notes
        -----
        - 2025-12-09 - Written - Bovy (UofT)

        """
        return _build_repr(self)

    def __mul__(self, b: Union[int, float]) -> "Force":
        """
        Multiply a Force or Potential's amplitude by a number

        Parameters
        ----------
        b : int or float
            Number to multiply the amplitude with.

        Returns
        -------
        Force or Potential instance
            New instance with amplitude = (old amplitude) x b.

        Notes
        -----
        - 2019-01-27 - Written - Bovy (UofT)

        """
        if not isinstance(b, (int, float)):
            raise TypeError(
                "Can only multiply a Force or Potential instance with a number"
            )
        out = copy.deepcopy(self)
        out._amp *= b
        return out

    # Similar functions
    __rmul__ = __mul__

    def __div__(self, b: Union[int, float]) -> "Force":
        return self.__mul__(1.0 / b)

    __truediv__ = __div__

    def __add__(self, b):
        """
        Add Force or Potential instances together to create a multi-component potential (e.g., pot= pot1+pot2+pot3)

        Parameters
        ----------
        b : Force or Potential instance

        Returns
        -------
        CompositePotential instance
            Represents the combined potential

        Notes
        -----
        - 2019-01-27 - Written - Bovy (UofT)
        - 2020-04-22 - Added check that unit systems of combined potentials are compatible - Bovy (UofT)
        - 2024-11-24 - Modified to return CompositePotential instead of list - Bovy (UofT)

        """
        from ..potential import flatten as flatten_pot
        from ..potential import planarForce
        from .CompositePotential import CompositePotential

        if not isinstance(flatten_pot([b])[0], (Force, planarForce)):
            raise TypeError(
                """Can only combine galpy Force objects with """
                """other Force objects or combinations thereof"""
            )
        assert physical_compatible(self, b), (
            """Physical unit conversion parameters (ro,vo) are not """
            """compatible between potentials to be combined"""
        )

        # If adding a planarForce, convert this CompositePotential to planar
        if isinstance(b, planarForce) and hasattr(b, "dim") and b.dim == 2:
            from .planarCompositePotential import planarCompositePotential

            return planarCompositePotential(self.toPlanar(), b)

        return CompositePotential(self, b)

    # Define separately to keep order
    def __radd__(self, b):
        from ..potential import flatten as flatten_pot
        from ..potential import planarPotential
        from .CompositePotential import CompositePotential

        # Only way to get here is in a situation that isn't supported: adding a Force
        # to something that is not a Force or similar itself, which all implement __add__
        raise TypeError(
            """Can only combine galpy Force objects with """
            """other Force objects or combinations thereof"""
        )

    def turn_physical_off(self) -> None:
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

    def turn_physical_on(
        self, ro: Optional[float] = None, vo: Optional[float] = None
    ) -> None:
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
        if ro is not False:
            self._roSet = True
            ro = conversion.parse_length_kpc(ro)
            if ro is not None:
                self._ro = ro
        if vo is not False:
            self._voSet = True
            vo = conversion.parse_velocity_kms(vo)
            if vo is not None:
                self._vo = vo

    def _Rforce_nodecorator(self, R, z, **kwargs):
        # Separate, so it can be used during orbit integration
        try:
            return self._amp * self._Rforce(R, z, **kwargs)
        except AttributeError:  # pragma: no cover
            from .Potential import PotentialError

            raise PotentialError("'_Rforce' function not implemented for this Force")

    def _zforce_nodecorator(self, R, z, **kwargs):
        # Separate, so it can be used during orbit integration
        try:
            return self._amp * self._zforce(R, z, **kwargs)
        except AttributeError:  # pragma: no cover
            from .Potential import PotentialError

            raise PotentialError(
                "'_zforce' function not implemented for this potential"
            )

    def _phitorque_nodecorator(self, R, z, **kwargs):
        # Separate, so it can be used during orbit integration
        try:
            return self._amp * self._phitorque(R, z, **kwargs)
        except AttributeError:  # pragma: no cover
            if self.isNonAxi:
                from .Potential import PotentialError

                raise PotentialError(
                    "'_phitorque' function not implemented for this DissipativeForce"
                )
            return 0.0

    @potential_physical_input
    @physical_conversion("force", pop=True)
    def rforce(self, R: float, z: float, **kwargs) -> float:
        """
        Evaluate the spherical radial force F_r (R,z).

        Parameters
        ----------
        R : float or Quantity
            Cylindrical Galactocentric radius.
        z : float or Quantity
            Vertical height.
        phi : float or Quantity, optional
            Azimuth. Default is None.
        t : float or Quantity, optional
            Time. Default is 0.0.
        v : float or Quantity, optional
            Current velocity in cylindrical coordinates. Required when including dissipative forces. Default is None.

        Returns
        -------
        F_r : float or Quantity
            The spherical radial force F_r (R,z).

        Notes
        -----
        - 2016-06-20 - Written - Bovy (UofT)

        """
        r = numpy.sqrt(R**2.0 + z**2.0)
        kwargs["use_physical"] = False
        return self.Rforce(R, z, **kwargs) * R / r + self.zforce(R, z, **kwargs) * z / r

    def toPlanar(self) -> "Force":
        """
        Convert a 3D potential into a planar potential in the mid-plane.

        Returns
        -------
        planarPotential, planarAxiPotential, or planarDissipativeForce instance

        """
        from ..potential import toPlanarPotential

        return toPlanarPotential(self)
