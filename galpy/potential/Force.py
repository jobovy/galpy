###############################################################################
#   Force.py: top-level class for a 3D force, conservative (Potential) or
#             not (DissipativeForce)
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

if _APY_LOADED:
    from astropy import units


class Force:
    """Top-level class for any force, conservative or dissipative"""

    def __init__(self, amp=1.0, ro=None, vo=None, amp_units=None):
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
            # velocity^2
            try:
                self._amp = conversion.parse_energy(self._amp, vo=self._vo)
            except units.UnitConversionError:
                pass
            else:
                unitFound = True
                if not amp_units == "velocity2":
                    raise units.UnitConversionError(
                        f"amp= parameter of {type(self).__name__} should have units of {amp_units}, but has units of velocity2 instead"
                    )
            if not unitFound:
                # mass
                try:
                    self._amp = conversion.parse_mass(
                        self._amp, ro=self._ro, vo=self._vo
                    )
                except units.UnitConversionError:
                    pass
                else:
                    unitFound = True
                    if not amp_units == "mass":
                        raise units.UnitConversionError(
                            f"amp= parameter of {type(self).__name__} should have units of {amp_units}, but has units of mass instead"
                        )
            if not unitFound:
                # density
                try:
                    self._amp = conversion.parse_dens(
                        self._amp, ro=self._ro, vo=self._vo
                    )
                except units.UnitConversionError:
                    pass
                else:
                    unitFound = True
                    if not amp_units == "density":
                        raise units.UnitConversionError(
                            f"amp= parameter of {type(self).__name__} should have units of {amp_units}, but has units of density instead"
                        )
            if not unitFound:
                # surface density
                try:
                    self._amp = conversion.parse_surfdens(
                        self._amp, ro=self._ro, vo=self._vo
                    )
                except units.UnitConversionError:
                    pass
                else:
                    unitFound = True
                    if not amp_units == "surfacedensity":
                        raise units.UnitConversionError(
                            f"amp= parameter of {type(self).__name__} should have units of {amp_units}, but has units of surface density instead"
                        )
            if not unitFound:
                raise units.UnitConversionError(
                    f"amp= parameter of {type(self).__name__} should have units of {amp_units}; given units are not understood"
                )
            else:
                # When amplitude is given with units, turn on physical output
                self._roSet = True
                self._voSet = True
        return None

    def __mul__(self, b):
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

    def __div__(self, b):
        return self.__mul__(1.0 / b)

    __truediv__ = __div__

    def __add__(self, b):
        """
        Add Force or Potential instances together to create a multi-component potential (e.g., pot= pot1+pot2+pot3)

        Parameters
        ----------
        b : Force or Potential instance or a list thereof

        Returns
        -------
        list of Force or Potential instances
            Represents the combined potential

        Notes
        -----
        - 2019-01-27 - Written - Bovy (UofT)
        - 2020-04-22 - Added check that unit systems of combined potentials are compatible - Bovy (UofT)

        """
        from ..potential import flatten as flatten_pot
        from ..potential import planarPotential

        if not isinstance(flatten_pot([b])[0], (Force, planarPotential)):
            raise TypeError(
                """Can only combine galpy Force objects with """
                """other Force objects or lists thereof"""
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
        from ..potential import planarPotential

        if not isinstance(flatten_pot([b])[0], (Force, planarPotential)):
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

    @potential_physical_input
    @physical_conversion("force", pop=True)
    def rforce(self, R, z, **kwargs):
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

    def toPlanar(self):
        """
        Convert a 3D potential into a planar potential in the mid-plane.

        Returns
        -------
        planarPotential, planarAxiPotential, or planarDissipativeForce instance

        """
        from ..potential import toPlanarPotential

        return toPlanarPotential(self)
