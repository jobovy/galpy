# renamed to conversion.py
from .conversion import *
from .conversion import _APY_UNITS, _APY_LOADED, _G, _kmsInPcMyr, \
    _PCIN10p18CM, _CIN10p5KMS, _EVIN10m19J, _MyrIn1013Sec, _TWOPI
import warnings
warnings.warn('galpy.util.bovy_conversion is being deprecated in favor of galpy.util.conversion; all functions in there are the same; please switch to the new import, because the old import will be removed in v1.9',FutureWarning)
