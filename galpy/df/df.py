from ..util import config, conversion
from ..util.conversion import physical_compatible


class df:
    """Top-level class for DF classes"""

    def __init__(self, ro=None, vo=None):
        """
        Initialize a DF object.

        Parameters
        ----------
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2016-02-28 - Written - Bovy (UofT)

        """
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

    def _check_consistent_units(self):
        """Internal function to check that the set of units for this object is consistent with that for the potential"""
        assert physical_compatible(
            self, self._pot
        ), "Physical conversion for the DF object is not consistent with that of the Potential given to it"

    def turn_physical_off(self):
        """
        Turn off automatic returning of outputs in physical units.

        Notes
        -----
        - 2017-06-05 - Written - Bovy (UofT)

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
            Reference distance (kpc). If False, don't turn it on.
        vo : float or Quantity, optional
            Reference velocity (km/s). If False, don't turn it on.

        Notes
        -----
        - 2016-06-05 - Written - Bovy (UofT)
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
