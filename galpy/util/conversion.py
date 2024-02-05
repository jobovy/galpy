###############################################################################
#
# conversion: utilities to convert from galpy 'natural units' to physical
#             units
#
###############################################################################
import copy
import math as m
import numbers
import warnings
from functools import wraps
from typing import Any, Tuple

import numpy

from ..util._optional_deps import _APY_LOADED, _APY_UNITS
from ..util.config import __config__

if not _APY_LOADED:
    _G = 4.302 * 10.0**-3.0  # pc / Msolar (km/s)^2
    _kmsInPcMyr = 1.0227121655399913
    _PCIN10p18CM = 3.08567758  # 10^18 cm
    _CIN10p5KMS = 2.99792458  # 10^5 km/s
    _MSOLAR10p30KG = 1.9891  # 10^30 kg
    _EVIN10m19J = 1.60217657  # 10^-19 J
else:
    from astropy import constants, units

    _G = constants.G.to(units.pc / units.Msun * units.km**2 / units.s**2).value
    _kmsInPcMyr = (units.km / units.s).to(units.pc / units.Myr)
    _PCIN10p18CM = units.pc.to(units.cm) / 10.0**18.0  # 10^18 cm
    _CIN10p5KMS = constants.c.to(units.km / units.s).value / 10.0**5.0  # 10^5 km/s
    _MSOLAR10p30KG = units.Msun.to(units.kg) / 10.0**30.0  # 10^30 kg
    _EVIN10m19J = units.eV.to(units.J) * 10.0**19.0  # 10^-19 J
_MyrIn1013Sec = 3.65242198 * 0.24 * 3.6  # use tropical year, like for pms
_TWOPI = 2.0 * m.pi


def dens_in_criticaldens(vo, ro, H=70.0):
    """
    Convert density to units of the critical density.

    Parameters
    ----------
    vo : float
        Velocity unit in km/s.
    ro : float
        Length unit in kpc.
    H : float, optional
        Hubble constant in km/s/Mpc. Default is 70.0.

    Returns
    -------
    float
        Conversion from units where vo=1. at ro=1. to units of the critical density.

    Notes
    -----
    - 2014-01-28 - Written - Bovy (IAS)

    """
    return vo**2.0 / ro**2.0 * 10.0**6.0 / H**2.0 * 8.0 * numpy.pi / 3.0


def dens_in_meanmatterdens(vo, ro, H=70.0, Om=0.3):
    """
    Convert density to units of the mean matter density.

    Parameters
    ----------
    vo : float
        Velocity unit in km/s.
    ro : float
        Length unit in kpc.
    H : float, optional
        Hubble constant in km/s/Mpc. Default is 70.0.
    Om : float, optional
        Omega matter. Default is 0.3.

    Returns
    -------
    float
        Conversion from units where vo=1. at ro=1. to units of the mean matter density.

    Notes
    -----
    - 2014-01-28 - Written - Bovy (IAS)

    """
    return dens_in_criticaldens(vo, ro, H=H) / Om


def dens_in_gevcc(vo, ro):
    """
    Convert density to GeV / cm^3.

    Parameters
    ----------
    vo : float
        Velocity unit in km/s.
    ro : float
        Length unit in kpc.

    Returns
    -------
    float
        Conversion from units where vo=1. at ro=1. to GeV/cm^3.

    Notes
    -----
    - 2014-06-16 - Written - Bovy (IAS)

    """
    return (
        vo**2.0
        / ro**2.0
        / _G
        * _MSOLAR10p30KG
        * _CIN10p5KMS**2.0
        / _EVIN10m19J
        / _PCIN10p18CM**3.0
        * 10.0**-4.0
    )


def dens_in_msolpc3(vo, ro):
    """
    Convert density to Msolar / pc^3.

    Parameters
    ----------
    vo : float
        Velocity unit in km/s.
    ro : float
        Length unit in kpc.

    Returns
    -------
    float
        Conversion from units where vo=1. at ro=1. to Msolar/pc^3.

    Notes
    -----
    - 2013-09-01 - Written - Bovy (IAS)

    """
    return vo**2.0 / ro**2.0 / _G * 10.0**-6.0


def force_in_2piGmsolpc2(vo, ro):
    """
    Convert a force or acceleration to 2piG x Msolar / pc^2

    Parameters
    ----------
    vo : float
        Velocity unit in km/s
    ro : float
        Length unit in kpc

    Returns
    -------
    float
        Conversion from units where vo=1. at ro=1.

    Notes
    -----
    - 2013-09-01 - Written - Bovy (IAS)

    """
    return vo**2.0 / ro / _G * 10.0**-3.0 / _TWOPI


def force_in_pcMyr2(vo, ro):
    """
    Convert a force or acceleration to pc/Myr^2.

    Parameters
    ----------
    vo : float
        Velocity unit in km/s.
    ro : float
        Length unit in kpc.

    Returns
    -------
    float
        Conversion from units where vo=1. at ro=1.

    Notes
    -----
    - 2013-09-01 - Written - Bovy (IAS)

    """
    return vo**2.0 / ro * _kmsInPcMyr**2.0 * 10.0**-3.0


def force_in_kmsMyr(vo, ro):
    """
    Convert a force or acceleration to km/s/Myr.

    Parameters
    ----------
    vo : float
        Velocity unit in km/s.
    ro : float
        Length unit in kpc.

    Returns
    -------
    float
        Conversion from units where vo=1. at ro=1.

    Notes
    -----
    - 2013-09-01 - Written - Bovy (IAS)

    """
    return vo**2.0 / ro * _kmsInPcMyr * 10.0**-3.0


def force_in_10m13kms2(vo, ro):
    """
    Convert a force or acceleration to 10^(-13) km/s^2

    Parameters
    ----------
    vo : float
        Velocity unit in km/s
    ro : float
        Length unit in kpc

    Returns
    -------
    float
        Conversion from units where vo=1. at ro=1.

    Notes
    -----
    - 2014-01-22 - Written - Bovy (IAS)

    """
    return vo**2.0 / ro * _kmsInPcMyr * 10.0**-3.0 / _MyrIn1013Sec


def freq_in_Gyr(vo, ro):
    """
    Convert a frequency to 1/Gyr.

    Parameters
    ----------
    vo : float
        Velocity unit in km/s.
    ro : float
        Length unit in kpc.

    Returns
    -------
    float
        Conversion from units where vo=1. at ro=1.

    Notes
    -----
    - 2013-09-01 - Written - Bovy (IAS)

    """
    return vo / ro * _kmsInPcMyr


def freq_in_kmskpc(vo, ro):
    """
    Convert a frequency to km/s/kpc.

    Parameters
    ----------
    vo : float
        Velocity unit in km/s.
    ro : float
        Length unit in kpc.

    Returns
    -------
    float
        Conversion from units where vo=1. at ro=1.

    Notes
    -----
    - 2013-09-01 - Written - Bovy (IAS)

    """
    return vo / ro


def surfdens_in_msolpc2(vo, ro):
    """
    Convert a surface density to Msolar / pc^2.

    Parameters
    ----------
    vo : float
        Velocity unit in km/s.
    ro : float
        Length unit in kpc.

    Returns
    -------
    float
        Conversion from units where vo=1. at ro=1.

    Notes
    -----
    - 2013-09-01 - Written - Bovy (IAS)

    """
    return vo**2.0 / ro / _G * 10.0**-3.0


def mass_in_msol(vo, ro):
    """
    Convert a mass to Msolar.

    Parameters
    ----------
    vo : float
        Velocity unit in km/s.
    ro : float
        Length unit in kpc.

    Returns
    -------
    float
        Conversion from units where vo=1. at ro=1.

    Notes
    -----
    - 2013-09-01 - Written - Bovy (IAS)

    """
    return vo**2.0 * ro / _G * 10.0**3.0


def mass_in_1010msol(vo, ro):
    """
    Convert a mass to 10^10 x Msolar.

    Parameters
    ----------
    vo : float
        Velocity unit in km/s.
    ro : float
        Length unit in kpc.

    Returns
    -------
    float
        Conversion from units where vo=1. at ro=1.

    Notes
    -----
    - 2013-09-01 - Written - Bovy (IAS)

    """
    return vo**2.0 * ro / _G * 10.0**-7.0


def time_in_Gyr(vo, ro):
    """
    Convert a time to Gyr.

    Parameters
    ----------
    vo : float
        Velocity unit in km/s.
    ro : float
        Length unit in kpc.

    Returns
    -------
    float
        Conversion from units where vo=1. at ro=1.

    Notes
    -----
    - 2013-09-01 - Written - Bovy (IAS)

    """
    return ro / vo / _kmsInPcMyr


def velocity_in_kpcGyr(vo, ro):
    """
    Convert a velocity to kpc/Gyr.

    Parameters
    ----------
    vo : float
        Velocity unit in km/s.
    ro : float
        Length unit in kpc.

    Returns
    -------
    float
        Conversion from units where vo=1. at ro=1.

    Notes
    -----
    - 2014-12-19 - Written - Bovy (IAS)

    """
    return vo * _kmsInPcMyr


def get_physical(obj: Any, include_set: bool = False) -> dict:
    """
    Return the velocity and length units for converting between physical and internal units as a dictionary for any galpy object, so they can easily be fed to galpy routines.

    Parameters
    ----------
    obj : Any
        A galpy object or list of such objects (e.g., a Potential, list of Potentials, Orbit, actionAngle instance, DF instance).
    include_set : bool, optional
        If True, also include roSet and voSet, flags of whether the unit is explicitly set in the object. Default is False.

    Returns
    -------
    dict
        A dictionary {'ro':length unit in kpc,'vo':velocity unit in km/s}; note that this routine will *always* return these conversion units, even if the obj you provide does not have units turned on.

    Notes
    -----
    - 2019-08-03 - Written - Bovy (UofT)

    """
    # Try flattening the object in case it's a nested list of Potentials
    from ..potential import Force
    from ..potential import flatten as flatten_pot
    from ..potential import linearPotential, planarPotential

    try:
        new_obj = flatten_pot(obj)
    except:  # pragma: no cover
        pass  # hope for the best!
    else:  # only apply flattening for potentials
        if isinstance(new_obj, (Force, planarPotential, linearPotential)) or (
            isinstance(new_obj, list)
            and len(new_obj) > 0
            and isinstance(new_obj[0], (Force, planarPotential, linearPotential))
        ):
            obj = new_obj
    if isinstance(obj, list):
        out_obj = obj[0]
    else:
        out_obj = obj
    out = {"ro": out_obj._ro, "vo": out_obj._vo}
    if include_set:
        out.update({"roSet": out_obj._roSet, "voSet": out_obj._voSet})
    return out


def extract_physical_kwargs(kwargs: dict) -> dict:
    """
    Extract the physical kwargs from a kwargs dictionary.

    Parameters
    ----------
    kwargs : dict
        A dictionary of kwargs.

    Returns
    -------
    dict
        A dictionary with just the physical kwargs.

    Notes
    -----
    - 2023-04-24 - Written - Bovy (UofT)

    """
    out = {}
    for key in kwargs.copy():
        if key in ["use_physical", "ro", "vo", "quantity"]:
            out[key] = kwargs.pop(key)
    return out


def physical_compatible(obj: Any, other_obj: Any) -> bool:
    """
    Test whether the velocity and length units for converting between physical and internal units are compatible for two galpy objects.

    Parameters
    ----------
    obj : galpy object or list of such objects
        A galpy object or list of such objects (e.g., a Potential, list of Potentials, Orbit, actionAngle instance, DF instance)
    other_obj : galpy object or list of such objects
        Another galpy object or list of such objects (e.g., a Potential, list of Potentials, Orbit, actionAngle instance, DF instance)

    Returns
    -------
    bool
        True if the units are compatible, False if not (compatible means that the units are the same when they are set for both objects).

    Notes
    -----
    - 2020-04-22 - Written - Bovy (UofT)

    """
    if obj is None or other_obj is None:  # if one is None, just state compat
        return True
    phys = get_physical(obj, include_set=True)
    other_phys = get_physical(other_obj, include_set=True)
    out = True
    if phys["roSet"] and other_phys["roSet"]:
        out = out and m.fabs((phys["ro"] - other_phys["ro"]) / phys["ro"]) < 1e-8
    if phys["voSet"] and other_phys["voSet"]:
        out = out and m.fabs((phys["vo"] - other_phys["vo"]) / phys["vo"]) < 1e-8
    return out


# Parsers of different inputs with units
def check_parser_input_type(func):
    """
    Decorator to check the inputs to a parse_ function; should be either:
    a) a number
    b) an array of numbers
    c) an astropy Quantity (incl. arrays)

    Also parses ro/vo if they are provided and converts them to the correct
    internal representation
    """

    @wraps(func)
    def parse_x_wrapper(x, **kwargs):
        if (
            not x is None
            and not isinstance(x, numbers.Number)
            and not (
                isinstance(x, numpy.ndarray)
                and (x.size == 0 or isinstance(x.flatten()[0], numbers.Number))
            )
            and not (_APY_LOADED and isinstance(x, units.Quantity))
        ):
            raise RuntimeError(
                f"Input '{x}' not understood; should either be a number or an astropy Quantity"
            )
        # Also parse ro and vo inputs
        if "ro" in kwargs:
            if (
                not kwargs["ro"] is None
                and not isinstance(kwargs["ro"], numbers.Number)
                and not (_APY_LOADED and isinstance(kwargs["ro"], units.Quantity))
            ):
                raise RuntimeError(
                    f"Input 'ro={kwargs['ro']}' not understood; should either be a number or an astropy Quantity"
                )
            else:
                kwargs["ro"] = (
                    kwargs["ro"].to(units.kpc).value
                    if _APY_LOADED and isinstance(kwargs["ro"], units.Quantity)
                    else kwargs["ro"]
                )
        if "vo" in kwargs:
            if (
                not kwargs["vo"] is None
                and not isinstance(kwargs["vo"], numbers.Number)
                and not (_APY_LOADED and isinstance(kwargs["vo"], units.Quantity))
            ):
                raise RuntimeError(
                    f"Input 'vo={kwargs['vo']}' not understood; should either be a number or an astropy Quantity"
                )
            else:
                kwargs["vo"] = (
                    kwargs["vo"].to(units.km / units.s).value
                    if _APY_LOADED and isinstance(kwargs["vo"], units.Quantity)
                    else kwargs["vo"]
                )
        return func(x, **kwargs)

    return parse_x_wrapper


@check_parser_input_type
def parse_length(x, ro=None, vo=None):
    return (
        x.to(units.kpc).value / ro
        if _APY_LOADED and isinstance(x, units.Quantity)
        else x
    )


@check_parser_input_type
def parse_length_kpc(x):
    return x.to(units.kpc).value if _APY_LOADED and isinstance(x, units.Quantity) else x


@check_parser_input_type
def parse_velocity(x, ro=None, vo=None):
    return (
        x.to(units.km / units.s).value / vo
        if _APY_LOADED and isinstance(x, units.Quantity)
        else x
    )


@check_parser_input_type
def parse_velocity_kms(x):
    return (
        x.to(units.km / units.s).value
        if _APY_LOADED and isinstance(x, units.Quantity)
        else x
    )


@check_parser_input_type
def parse_angle(x):
    return x.to(units.rad).value if _APY_LOADED and isinstance(x, units.Quantity) else x


@check_parser_input_type
def parse_time(x, ro=None, vo=None):
    return (
        x.to(units.Gyr).value / time_in_Gyr(vo, ro)
        if _APY_LOADED and isinstance(x, units.Quantity)
        else x
    )


@check_parser_input_type
def parse_mass(x, ro=None, vo=None):
    try:
        return (
            x.to(units.pc * units.km**2 / units.s**2).value / mass_in_msol(vo, ro) / _G
            if _APY_LOADED and isinstance(x, units.Quantity)
            else x
        )
    except units.UnitConversionError:
        pass
    return (
        x.to(1e10 * units.Msun).value / mass_in_1010msol(vo, ro)
        if _APY_LOADED and isinstance(x, units.Quantity)
        else x
    )


@check_parser_input_type
def parse_energy(x, ro=None, vo=None):
    return (
        x.to(units.km**2 / units.s**2).value / vo**2.0
        if _APY_LOADED and isinstance(x, units.Quantity)
        else x
    )


@check_parser_input_type
def parse_angmom(x, ro=None, vo=None):
    return (
        x.to(units.kpc * units.km / units.s).value / ro / vo
        if _APY_LOADED and isinstance(x, units.Quantity)
        else x
    )


@check_parser_input_type
def parse_frequency(x, ro=None, vo=None):
    return (
        x.to(units.km / units.s / units.kpc).value / freq_in_kmskpc(vo, ro)
        if _APY_LOADED and isinstance(x, units.Quantity)
        else x
    )


@check_parser_input_type
def parse_force(x, ro=None, vo=None):
    try:
        return (
            x.to(units.pc / units.Myr**2).value / force_in_pcMyr2(vo, ro)
            if _APY_LOADED and isinstance(x, units.Quantity)
            else x
        )
    except units.UnitConversionError:
        pass
    return (
        x.to(units.Msun / units.pc**2).value / force_in_2piGmsolpc2(vo, ro)
        if _APY_LOADED and isinstance(x, units.Quantity)
        else x
    )


@check_parser_input_type
def parse_dens(x, ro=None, vo=None):
    try:
        return (
            x.to(units.Msun / units.pc**3).value / dens_in_msolpc3(vo, ro)
            if _APY_LOADED and isinstance(x, units.Quantity)
            else x
        )
    except units.UnitConversionError:
        pass
    # Try Gxdens
    return (
        x.to(units.km**2 / units.s**2 / units.pc**2).value
        / dens_in_msolpc3(vo, ro)
        / _G
        if _APY_LOADED and isinstance(x, units.Quantity)
        else x
    )


@check_parser_input_type
def parse_surfdens(x, ro=None, vo=None):
    try:
        return (
            x.to(units.Msun / units.pc**2).value / surfdens_in_msolpc2(vo, ro)
            if _APY_LOADED and isinstance(x, units.Quantity)
            else x
        )
    except units.UnitConversionError:
        pass
    # Try Gxsurfdens
    return (
        x.to(units.km**2 / units.s**2 / units.pc).value
        / surfdens_in_msolpc2(vo, ro)
        / _G
        if _APY_LOADED and isinstance(x, units.Quantity)
        else x
    )


@check_parser_input_type
def parse_numdens(x, ro=None, vo=None):
    return (
        x.to(1 / units.kpc**3).value * ro**3
        if _APY_LOADED and isinstance(x, units.Quantity)
        else x
    )


# Decorator to apply these transformations
# NOTE: names with underscores in them signify return values that *always* have
# units, which is depended on in the Orbit returns (see issue #326)
_roNecessary = {
    "time": True,
    "position": True,
    "position_kpc": True,
    "velocity": False,
    "velocity2": False,
    "velocity2surfacendensity": False,
    "velocity_kms": False,
    "energy": False,
    "density": True,
    "numberdensity": True,
    "force": True,
    "velocity2surfacedensity": True,
    "surfacedensity": True,
    "numbersurfacedensity": True,
    "surfacedensitydistance": True,
    "mass": True,
    "action": True,
    "frequency": True,
    "frequency-kmskpc": True,
    "forcederivative": True,
    "angle": True,
    "angle_deg": True,
    "proper-motion_masyr": True,
    "phasespacedensity": True,
    "phasespacedensity2d": True,
    "phasespacedensityvelocity": True,
    "phasespacedensityvelocity2": True,
    "energydensity": False,
    "dimensionless": False,
}
_voNecessary = copy.copy(_roNecessary)
_voNecessary["position"] = False
_voNecessary["position_kpc"] = False
_voNecessary["angle"] = False
_voNecessary["angle_deg"] = False
_voNecessary["velocity"] = True
_voNecessary["velocity2"] = True
_voNecessary["velocity_kms"] = True
_voNecessary["energy"] = True
_voNecessary["energydensity"] = True


# Determine whether or not outputs will be physical or not
def physical_output(obj: Any, kwargs: dict, quantity: str) -> Tuple[bool, float, float]:
    """
    Determine whether or not outputs will be physical or not

    Parameters
    ----------
    obj : galpy object (or list in case of potentials)
        galpy object.
    kwargs : dict
        Kwargs passed to the method.
    quantity : str
        Quantity to be returned.

    Returns
    -------
    tuple
        A tuple containing:
            - boolean that indicates whether or not to use physical units.
            - ro.
            - vo.

    Notes
    -----
    - 2023-04-24 - Written - Bovy (UofT).

    """
    use_physical = kwargs.get("use_physical", True) and not kwargs.get("log", False)
    # Parse whether ro or vo should be considered to be set, because
    # the return value will have units anyway
    # (like in Orbit methods that return numbers with units, like ra)
    roSet = "_" in quantity  # _ in quantity name means always units
    voSet = "_" in quantity  # _ in quantity name means always units
    use_physical = (
        use_physical or "_" in quantity
    )  # _ in quantity name means always units
    ro = kwargs.get("ro", None)
    if ro is None and (roSet or (hasattr(obj, "_roSet") and obj._roSet)):
        ro = obj._ro
    if (
        ro is None
        and isinstance(obj, list)
        and hasattr(obj[0], "_roSet")
        and obj[0]._roSet
    ):
        # For lists of Potentials
        ro = obj[0]._ro
    if _APY_LOADED and isinstance(ro, units.Quantity):
        ro = ro.to(units.kpc).value
    vo = kwargs.get("vo", None)
    if vo is None and (voSet or (hasattr(obj, "_voSet") and obj._voSet)):
        vo = obj._vo
    if (
        vo is None
        and isinstance(obj, list)
        and hasattr(obj[0], "_voSet")
        and obj[0]._voSet
    ):
        # For lists of Potentials
        vo = obj[0]._vo
    if _APY_LOADED and isinstance(vo, units.Quantity):
        vo = vo.to(units.km / units.s).value
    return (
        (
            use_physical
            and not (_voNecessary[quantity.lower()] and vo is None)
            and not (_roNecessary[quantity.lower()] and ro is None)
        ),
        ro,
        vo,
    )


def physical_conversion(quantity, pop=False):
    """Decorator to convert to physical coordinates:
    quantity = [position,velocity,time]"""

    def wrapper(method):
        @wraps(method)
        def wrapped(*args, **kwargs):
            # Determine whether or not to return outputs in physical units
            use_physical_output, ro, vo = physical_output(args[0], kwargs, quantity)
            # Determine whether physical outputs were explicitly asked for
            use_physical_explicitly_set = kwargs.get("use_physical", False)
            # Determine whether or not to return outputs as quantities
            _apy_units = kwargs.get("quantity", _APY_UNITS)
            # Remove ro, vo, use_physical, and quantity kwargs if necessary
            if pop:
                _ = extract_physical_kwargs(kwargs)
            if use_physical_output:
                from ..orbit import Orbit

                if quantity.lower() == "time":
                    fac = time_in_Gyr(vo, ro)
                    if _apy_units:
                        u = units.Gyr
                elif quantity.lower() == "position":
                    fac = ro
                    if _apy_units:
                        u = units.kpc
                elif quantity.lower() == "position_kpc":  # already in kpc
                    fac = 1.0
                    if _apy_units:
                        u = units.kpc
                elif quantity.lower() == "velocity":
                    fac = vo
                    if _apy_units:
                        u = units.km / units.s
                elif quantity.lower() == "velocity2":
                    fac = vo**2.0
                    if _apy_units:
                        u = (units.km / units.s) ** 2
                elif quantity.lower() == "velocity_kms":  # already in km/s
                    fac = 1.0
                    if _apy_units:
                        u = units.km / units.s
                elif quantity.lower() == "frequency":
                    if kwargs.get("kmskpc", False) and not _apy_units:
                        fac = freq_in_kmskpc(vo, ro)
                    else:
                        fac = freq_in_Gyr(vo, ro)
                        if _apy_units:
                            u = units.Gyr**-1.0
                elif quantity.lower() == "frequency-kmskpc":
                    fac = freq_in_kmskpc(vo, ro)
                    if _apy_units:
                        u = units.km / units.s / units.kpc
                elif quantity.lower() == "action":
                    fac = ro * vo
                    if _apy_units:
                        u = units.kpc * units.km / units.s
                elif quantity.lower() == "energy":
                    fac = vo**2.0
                    if _apy_units:
                        u = units.km**2.0 / units.s**2.0
                elif quantity.lower() == "angle":  # in rad
                    fac = 1.0
                    if _apy_units:
                        u = units.rad
                elif quantity.lower() == "angle_deg":  # already in deg
                    fac = 1.0
                    if _apy_units:
                        u = units.deg
                elif quantity.lower() == "proper-motion_masyr":  # already in mas/yr
                    fac = 1.0
                    if _apy_units:
                        u = units.mas / units.yr
                elif quantity.lower() == "force":
                    fac = force_in_kmsMyr(vo, ro)
                    if _apy_units:
                        u = units.km / units.s / units.Myr
                elif quantity.lower() == "density":
                    fac = dens_in_msolpc3(vo, ro)
                    if _apy_units:
                        u = units.Msun / units.pc**3
                elif quantity.lower() == "numberdensity":
                    fac = 1 / ro**3.0
                    if _apy_units:
                        u = 1 / units.kpc**3
                elif quantity.lower() == "velocity2surfacedensity":
                    fac = surfdens_in_msolpc2(vo, ro) * vo**2
                    if _apy_units:
                        u = units.Msun / units.pc**2 * (units.km / units.s) ** 2
                elif quantity.lower() == "surfacedensity":
                    fac = surfdens_in_msolpc2(vo, ro)
                    if _apy_units:
                        u = units.Msun / units.pc**2
                elif quantity.lower() == "numbersurfacedensity":
                    fac = 1.0 / ro**2.0
                    if _apy_units:
                        u = 1 / units.kpc**2
                elif quantity.lower() == "surfacedensitydistance":
                    fac = surfdens_in_msolpc2(vo, ro) * ro * 1000.0
                    if _apy_units:
                        u = units.Msun / units.pc
                elif quantity.lower() == "mass":
                    fac = mass_in_msol(vo, ro)
                    if _apy_units:
                        u = units.Msun
                elif quantity.lower() == "forcederivative":
                    fac = freq_in_Gyr(vo, ro) ** 2.0
                    if _apy_units:
                        u = units.Gyr**-2.0
                elif quantity.lower() == "phasespacedensity":
                    fac = 1.0 / vo**3.0 / ro**3.0
                    if _apy_units:
                        u = 1 / (units.km / units.s) ** 3 / units.kpc**3
                elif quantity.lower() == "phasespacedensity2d":
                    fac = 1.0 / vo**2.0 / ro**2.0
                    if _apy_units:
                        u = 1 / (units.km / units.s) ** 2 / units.kpc**2
                elif quantity.lower() == "phasespacedensityvelocity":
                    fac = 1.0 / vo**2.0 / ro**3.0
                    if _apy_units:
                        u = 1 / (units.km / units.s) ** 2 / units.kpc**3
                elif quantity.lower() == "phasespacedensityvelocity2":
                    fac = 1.0 / vo / ro**3.0
                    if _apy_units:
                        u = 1 / (units.km / units.s) / units.kpc**3
                elif quantity.lower() == "energydensity":
                    fac = 1.0 / vo**2.0
                    if _apy_units:
                        u = 1 / (units.km / units.s) ** 2
                elif quantity.lower() == "dimensionless":
                    fac = 1.0
                    if _apy_units:
                        u = units.dimensionless_unscaled
                out = method(*args, **kwargs)
                if out is None:
                    return out
                if _apy_units:
                    return units.Quantity(out * fac, unit=u)
                else:
                    # complicated logic for dealing with ro and vo arrays
                    return out * (
                        fac[:, numpy.newaxis]
                        if isinstance(fac, numpy.ndarray) and len(out.shape) > 1
                        else fac
                    )
            else:
                if use_physical_explicitly_set:
                    warnings.warn(
                        "Returning output(s) in internal units even though use_physical=True, because ro and/or vo not set"
                    )
                return method(*args, **kwargs)

        return wrapped

    return wrapper


def physical_conversion_tuple(quantities, pop=False):
    """Decorator to convert to physical coordinates for tuple outputs.
    So outputs are a tuple of quantities that each need to be converted,
    with possibly different conversions, e.g., (R,vR)"""

    def wrapper(method):
        @wraps(method)
        def wrapped(*args, **kwargs):
            rawOut = method(*args, **kwargs)
            out = ()
            for ii in range(len(rawOut)):
                # Apply physical conversion by converting a wrapped dummy function that returns the raw output
                out = out + (
                    physical_conversion(quantities[ii])(lambda x, **kwargs: rawOut[ii])(
                        args[0], **kwargs
                    ),
                )
            return out

        return wrapped

    return wrapper


def potential_physical_input(method):
    """Decorator to convert inputs to Potential functions from physical
    to internal coordinates"""

    @wraps(method)
    def wrapper(*args, **kwargs):
        from ..potential import flatten as flatten_potential

        Pot = flatten_potential(args[0])
        ro = kwargs.get("ro", None)
        if ro is None and hasattr(Pot, "_ro"):
            ro = Pot._ro
        if ro is None and isinstance(Pot, list) and hasattr(Pot[0], "_ro"):
            # For lists of Potentials
            ro = Pot[0]._ro
        if _APY_LOADED and isinstance(ro, units.Quantity):
            ro = ro.to(units.kpc).value
        if "t" in kwargs or "M" in kwargs:
            vo = kwargs.get("vo", None)
            if vo is None and hasattr(Pot, "_vo"):
                vo = Pot._vo
            if vo is None and isinstance(Pot, list) and hasattr(Pot[0], "_vo"):
                # For lists of Potentials
                vo = Pot[0]._vo
            if _APY_LOADED and isinstance(vo, units.Quantity):
                vo = vo.to(units.km / units.s).value
        # Loop through args
        newargs = (Pot,)
        for ii in range(1, len(args)):
            if _APY_LOADED and isinstance(args[ii], units.Quantity):
                newargs = newargs + (args[ii].to(units.kpc).value / ro,)
            else:
                newargs = newargs + (args[ii],)
        args = newargs
        # phi and t kwargs, also do R, z, and x in case these are given as kwargs
        if (
            "phi" in kwargs
            and _APY_LOADED
            and isinstance(kwargs["phi"], units.Quantity)
        ):
            kwargs["phi"] = kwargs["phi"].to(units.rad).value
        if "t" in kwargs and _APY_LOADED and isinstance(kwargs["t"], units.Quantity):
            kwargs["t"] = kwargs["t"].to(units.Gyr).value / time_in_Gyr(vo, ro)
        if "R" in kwargs and _APY_LOADED and isinstance(kwargs["R"], units.Quantity):
            kwargs["R"] = kwargs["R"].to(units.kpc).value / ro
        if "z" in kwargs and _APY_LOADED and isinstance(kwargs["z"], units.Quantity):
            kwargs["z"] = kwargs["z"].to(units.kpc).value / ro
        if "x" in kwargs and _APY_LOADED and isinstance(kwargs["x"], units.Quantity):
            kwargs["x"] = kwargs["x"].to(units.kpc).value / ro
        # v kwarg for dissipative forces
        if "v" in kwargs and _APY_LOADED and isinstance(kwargs["v"], units.Quantity):
            kwargs["v"] = kwargs["v"].to(units.km / units.s).value / vo
        # Mass kwarg for rtide
        if "M" in kwargs and _APY_LOADED and isinstance(kwargs["M"], units.Quantity):
            try:
                kwargs["M"] = kwargs["M"].to(units.Msun).value / mass_in_msol(vo, ro)
            except units.UnitConversionError:
                kwargs["M"] = (
                    kwargs["M"].to(units.pc * units.km**2 / units.s**2).value
                    / mass_in_msol(vo, ro)
                    / _G
                )
        # kwargs that come up in quasiisothermaldf
        # z done above
        if "dz" in kwargs and _APY_LOADED and isinstance(kwargs["dz"], units.Quantity):
            kwargs["dz"] = kwargs["dz"].to(units.kpc).value / ro
        if "dR" in kwargs and _APY_LOADED and isinstance(kwargs["dR"], units.Quantity):
            kwargs["dR"] = kwargs["dR"].to(units.kpc).value / ro
        if (
            "zmax" in kwargs
            and _APY_LOADED
            and isinstance(kwargs["zmax"], units.Quantity)
        ):
            kwargs["zmax"] = kwargs["zmax"].to(units.kpc).value / ro
        return method(*args, **kwargs)

    return wrapper


def physical_conversion_actionAngle(quantity, pop=False):
    """Decorator to convert to physical coordinates for the actionAngle methods:
    quantity= call, actionsFreqs, or actionsFreqsAngles (or EccZmaxRperiRap for actionAngleStaeckel)
    """

    def wrapper(method):
        @wraps(method)
        def wrapped(*args, **kwargs):
            use_physical = kwargs.get("use_physical", True)
            ro = kwargs.get("ro", None)
            if ro is None and hasattr(args[0], "_roSet") and args[0]._roSet:
                ro = args[0]._ro
            if _APY_LOADED and isinstance(ro, units.Quantity):
                ro = ro.to(units.kpc).value
            vo = kwargs.get("vo", None)
            if vo is None and hasattr(args[0], "_voSet") and args[0]._voSet:
                vo = args[0]._vo
            if _APY_LOADED and isinstance(vo, units.Quantity):
                vo = vo.to(units.km / units.s).value
            # Remove ro and vo kwargs if necessary
            if pop and "use_physical" in kwargs:
                kwargs.pop("use_physical")
            if pop and "ro" in kwargs:
                kwargs.pop("ro")
            if pop and "vo" in kwargs:
                kwargs.pop("vo")
            if use_physical and not vo is None and not ro is None:
                out = method(*args, **kwargs)
                if "call" in quantity or "actions" in quantity:
                    if "actions" in quantity and len(out) < 4:  # 1D system
                        fac = [ro * vo]
                        if _APY_UNITS:
                            u = [units.kpc * units.km / units.s]
                    else:
                        fac = [ro * vo, ro * vo, ro * vo]
                        if _APY_UNITS:
                            u = [
                                units.kpc * units.km / units.s,
                                units.kpc * units.km / units.s,
                                units.kpc * units.km / units.s,
                            ]
                if "Freqs" in quantity:
                    FreqsFac = freq_in_Gyr(vo, ro)
                    if len(out) < 4:  # 1D system
                        fac.append(FreqsFac)
                        if _APY_UNITS:
                            Freqsu = units.Gyr**-1.0
                            u.append(Freqsu)
                    else:
                        fac.extend([FreqsFac, FreqsFac, FreqsFac])
                        if _APY_UNITS:
                            Freqsu = units.Gyr**-1.0
                            u.extend([Freqsu, Freqsu, Freqsu])
                if "Angles" in quantity:
                    if len(out) < 4:  # 1D system
                        fac.append(1.0)
                        if _APY_UNITS:
                            Freqsu = units.Gyr**-1.0
                            u.append(units.rad)
                    else:
                        fac.extend([1.0, 1.0, 1.0])
                        if _APY_UNITS:
                            Freqsu = units.Gyr**-1.0
                            u.extend([units.rad, units.rad, units.rad])
                if "EccZmaxRperiRap" in quantity:
                    fac = [1.0, ro, ro, ro]
                    if _APY_UNITS:
                        u = [1.0, units.kpc, units.kpc, units.kpc]
                if _APY_UNITS:
                    newOut = ()
                    try:
                        for ii in range(len(out)):
                            newOut = newOut + (
                                units.Quantity(out[ii] * fac[ii], unit=u[ii]),
                            )
                    except TypeError:  # happens if out = scalar
                        newOut = units.Quantity(out * fac[0], unit=u[0])
                else:
                    newOut = ()
                    try:
                        for ii in range(len(out)):
                            newOut = newOut + (out[ii] * fac[ii],)
                    except TypeError:  # happens if out = scalar
                        newOut = out * fac[0]
                return newOut
            else:
                return method(*args, **kwargs)

        return wrapped

    return wrapper


def actionAngle_physical_input(method):
    """Decorator to convert inputs to actionAngle functions from physical
    to internal coordinates"""

    @wraps(method)
    def wrapper(*args, **kwargs):
        if len(args) < 3:  # orbit input
            return method(*args, **kwargs)
        ro = kwargs.get("ro", None)
        if ro is None and hasattr(args[0], "_ro"):
            ro = args[0]._ro
        if _APY_LOADED and isinstance(ro, units.Quantity):
            ro = ro.to(units.kpc).value
        vo = kwargs.get("vo", None)
        if vo is None and hasattr(args[0], "_vo"):
            vo = args[0]._vo
        if _APY_LOADED and isinstance(vo, units.Quantity):
            vo = vo.to(units.km / units.s).value
        # Loop through args
        newargs = ()
        for ii in range(len(args)):
            if _APY_LOADED and isinstance(args[ii], units.Quantity):
                try:
                    targ = args[ii].to(units.kpc).value / ro
                except units.UnitConversionError:
                    try:
                        targ = args[ii].to(units.km / units.s).value / vo
                    except units.UnitConversionError:
                        try:
                            targ = args[ii].to(units.rad).value
                        except units.UnitConversionError:
                            raise units.UnitConversionError(
                                "Input units not understood"
                            )
                newargs = newargs + (targ,)
            else:
                newargs = newargs + (args[ii],)
        args = newargs
        return method(*args, **kwargs)

    return wrapper


def physical_conversion_actionAngleInverse(quantity, pop=False):
    """Decorator to convert to physical coordinates for the actionAngleInverse methods:
    quantity= call, xvFreqs, or Freqs"""

    def wrapper(method):
        @wraps(method)
        def wrapped(*args, **kwargs):
            use_physical = kwargs.get("use_physical", True)
            ro = kwargs.get("ro", None)
            if ro is None and hasattr(args[0], "_roSet") and args[0]._roSet:
                ro = args[0]._ro
            if _APY_LOADED and isinstance(ro, units.Quantity):
                ro = ro.to(units.kpc).value
            vo = kwargs.get("vo", None)
            if vo is None and hasattr(args[0], "_voSet") and args[0]._voSet:
                vo = args[0]._vo
            if _APY_LOADED and isinstance(vo, units.Quantity):
                vo = vo.to(units.km / units.s).value
            # Remove ro and vo kwargs if necessary
            if pop and "use_physical" in kwargs:
                kwargs.pop("use_physical")
            if pop and "ro" in kwargs:
                kwargs.pop("ro")
            if pop and "vo" in kwargs:
                kwargs.pop("vo")
            if use_physical and not vo is None and not ro is None:
                fac = []
                u = []
                out = method(*args, **kwargs)
                if "call" in quantity or "xv" in quantity:
                    if "xv" in quantity and len(out) < 4:  # 1D system
                        fac.extend([ro, vo])
                        if _APY_UNITS:
                            u.extend([units.kpc, units.km / units.s])
                    else:
                        fac.extend([ro, vo, vo, ro, vo, 1.0])
                        if _APY_UNITS:
                            u.extend(
                                [
                                    units.kpc,
                                    units.km / units.s,
                                    units.km / units.s,
                                    units.kpc,
                                    units.km / units.s,
                                    units.rad,
                                ]
                            )
                if "Freqs" in quantity:
                    FreqsFac = freq_in_Gyr(vo, ro)
                    if isinstance(out, float):  # 1D system
                        fac.append(FreqsFac)
                        if _APY_UNITS:
                            Freqsu = units.Gyr**-1.0
                            u.append(Freqsu)
                    else:
                        fac.extend([FreqsFac, FreqsFac, FreqsFac])
                        if _APY_UNITS:
                            Freqsu = units.Gyr**-1.0
                            u.extend([Freqsu, Freqsu, Freqsu])
                if _APY_UNITS:
                    newOut = ()
                    try:
                        for ii in range(len(out)):
                            newOut = newOut + (
                                units.Quantity(out[ii] * fac[ii], unit=u[ii]),
                            )
                    except TypeError:  # Happens when out == scalar
                        newOut = units.Quantity(out * fac[0], unit=u[0])
                else:
                    newOut = ()
                    try:
                        for ii in range(len(out)):
                            newOut = newOut + (out[ii] * fac[ii],)
                    except TypeError:  # Happens when out == scalar
                        newOut = out * fac[0]
                return newOut
            else:
                return method(*args, **kwargs)

        return wrapped

    return wrapper


def actionAngleInverse_physical_input(method):
    """Decorator to convert inputs to actionAngleInverse functions from
    physical to internal coordinates"""

    @wraps(method)
    def wrapper(*args, **kwargs):
        ro = kwargs.get("ro", None)
        if ro is None and hasattr(args[0], "_ro"):
            ro = args[0]._ro
        if _APY_LOADED and isinstance(ro, units.Quantity):
            ro = ro.to(units.kpc).value
        vo = kwargs.get("vo", None)
        if vo is None and hasattr(args[0], "_vo"):
            vo = args[0]._vo
        if _APY_LOADED and isinstance(vo, units.Quantity):
            vo = vo.to(units.km / units.s).value
        # Loop through args
        newargs = ()
        for ii in range(len(args)):
            if _APY_LOADED and isinstance(args[ii], units.Quantity):
                try:
                    targ = args[ii].to(units.kpc * units.km / units.s).value / ro / vo
                except units.UnitConversionError:
                    try:
                        targ = args[ii].to(units.rad).value
                    except units.UnitConversionError:
                        raise units.UnitConversionError("Input units not understood")
                newargs = newargs + (targ,)
            else:
                newargs = newargs + (args[ii],)
        args = newargs
        return method(*args, **kwargs)

    return wrapper
