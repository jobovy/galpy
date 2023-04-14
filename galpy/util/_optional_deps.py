# Central place to process optional dependencies
from packaging.version import parse as parse_version

# astropy
from ..util.config import __config__

_APY_UNITS = __config__.getboolean("astropy", "astropy-units")
_APY_LOADED = True
_APY3 = None
_APY_GE_31 = None
try:
    from astropy import constants, units
except ImportError:
    _APY_UNITS = False
    _APY_LOADED = False
else:
    import astropy

    _APY3 = parse_version(astropy.__version__) > parse_version("3")
    _APY_GE_31 = parse_version(astropy.__version__) > parse_version("3.0.5")
_APY_COORD_LOADED = True
try:
    from astropy.coordinates import SkyCoord
except ImportError:
    SkyCoord = None
    _APY_COORD_LOADED = False

# astroquery
_ASTROQUERY_LOADED = True
try:
    from astroquery.simbad import Simbad
except ImportError:
    _ASTROQUERY_LOADED = False

# numexpr
_NUMEXPR_LOADED = True
try:
    import numexpr
except ImportError:  # pragma: no cover
    _NUMEXPR_LOADED = False

# tqdm
_TQDM_LOADED = True
try:
    import tqdm
except ImportError:  # pragma: no cover
    _TQDM_LOADED = False

# numba
_NUMBA_LOADED = True
try:
    from numba import cfunc, types
except ImportError:
    _NUMBA_LOADED = False

# jax
_JAX_LOADED = True
try:
    from jax import grad, vmap
except ImportError:
    _JAX_LOADED = False

# pynbody
_PYNBODY_LOADED = True
try:
    import pynbody
except ImportError:  # pragma: no cover
    _PYNBODY_LOADED = False
