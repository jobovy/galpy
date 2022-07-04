# renamed to coords.py
import warnings

from .coords import *
from .coords import (_APY_COORDS, _APY_LOADED, _DEGTORAD, _K,
                     _parse_epoch_frame_apy)

warnings.warn('galpy.util.bovy_coords is being deprecated in favor of galpy.util.coords; all functions in there are the same; please switch to the new import, because the old import will be removed in v1.9',FutureWarning)
