import configparser
import copy
import os
import os.path
import warnings

# The default configuration
default_configuration = {
    "normalization": {"ro": "8.", "vo": "220."},
    "astropy": {"astropy-units": "False", "astropy-coords": "True"},
    "plot": {"seaborn-bovy-defaults": "False"},
    "warnings": {"verbose": "False"},
    "version-check": {
        "do-check": "True",
        "check-non-interactive": "True",
        "check-non-interactive-every": "1",
        "last-non-interactive-check": "2000-01-01",
    },
}
default_filename = os.path.join(os.path.expanduser("~"), ".galpyrc")


def check_config(configuration):
    # Check that the configuration is a valid galpy configuration
    for sec_key in default_configuration.keys():
        if not configuration.has_section(sec_key):
            return False
        for key in default_configuration[sec_key]:
            if not configuration.has_option(sec_key, key):
                return False
    return True


def fix_config(configuration=None):
    # Reduces to default if configuration is None
    # otherwise fixes the configuration if it is not standard
    fixedconfig = configparser.ConfigParser()
    # Write different sections
    for sec_key in default_configuration.keys():
        fixedconfig.add_section(sec_key)
        for key in default_configuration[sec_key]:
            if (
                configuration is None
                or not configuration.has_section(sec_key)
                or not configuration.has_option(sec_key, key)
            ):
                fixedconfig.set(sec_key, key, default_configuration[sec_key][key])
            else:
                fixedconfig.set(sec_key, key, configuration.get(sec_key, key))
    return fixedconfig


def write_config(filename, configuration):
    try:
        with open(filename, "w") as configfile:
            configuration.write(configfile)
    except Exception as e:  # pragma: no cover
        warnings.warn(
            f"""Could not write new/fixed galpy configuration to {filename},"""
            f""" because of \"{type(e).__name__}: {e.__str__()}\""""
        )
    return None


# Read the configuration file
__config__ = configparser.ConfigParser()
cfilename = __config__.read(".galpyrc")
if not cfilename:
    cfilename = __config__.read(default_filename)
    if not cfilename:
        __config__ = fix_config()
        write_config(default_filename, __config__)
        cfilename = [default_filename]
if not check_config(__config__):
    __config__ = fix_config(__config__)
    write_config(cfilename[-1], __config__)
# Store a version of the config in case we need to re-write parts of it,
# but don't want to apply changes that we don't want to re-write
configfilename = cfilename[-1]
__orig__config__ = copy.deepcopy(__config__)


# Set configuration variables on the fly
def set_ro(ro):
    """
    Set the global configuration value of ro (distance scale)

    Parameters
    ----------
    ro : float or astropy Quantity
        Scale in kpc or astropy Quantity

    Returns
    -------
    None

    Notes
    -----
    - 2016-01-05 - Written - Bovy (UofT)
    """
    from ..util._optional_deps import _APY_LOADED

    if _APY_LOADED:
        from astropy import units
    if _APY_LOADED and isinstance(ro, units.Quantity):
        ro = ro.to(units.kpc).value
    __config__.set("normalization", "ro", str(ro))


def set_vo(vo):
    """
    Set the global configuration value of vo (velocity scale)

    Parameters
    ----------
    vo : float or astropy Quantity
        Scale in km/s or astropy Quantity

    Returns
    -------
    None

    Notes
    -----
    - 2016-01-05 - Written - Bovy (UofT)
    """
    from ..util._optional_deps import _APY_LOADED

    if _APY_LOADED:
        from astropy import units
    if _APY_LOADED and isinstance(vo, units.Quantity):
        vo = vo.to(units.km / units.s).value
    __config__.set("normalization", "vo", str(vo))
