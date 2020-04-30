# _load_extension_libs.py: centralized place to load the C extensions
import os
import sys
import distutils.sysconfig as sysconfig
import warnings
import ctypes
from ..util import galpyWarning, galpyWarningVerbose

PY3= sys.version > '3'
if PY3:
    _ext_suffix= sysconfig.get_config_var('EXT_SUFFIX')
else: #pragma: no cover
    _ext_suffix= '.so'

_libgalpy= None
_libgalpy_loaded= None

_libgalpy_actionAngleTorus= None
_libgalpy_actionAngleTorus_loaded= None

def load_libgalpy():
    global _libgalpy
    global _libgalpy_loaded
    if _libgalpy_loaded is False or not _libgalpy is None:
        return (_libgalpy,_libgalpy_loaded)
    outerr= None
    for path in sys.path:
        if not os.path.isdir(path): continue
        try:
            if sys.platform == 'win32' and sys.version_info >= (3,8): # pragma: no cover
                # winmode=0x008 is easy-going way to call LoadLibraryExA
                _lib = ctypes.CDLL(os.path.join(path,'libgalpy%s' % _ext_suffix),winmode=0x008)
            else:
                _lib = ctypes.CDLL(os.path.join(path,'libgalpy%s' % _ext_suffix))
        except OSError as e:
            if os.path.exists(os.path.join(path,'libgalpy%s' % _ext_suffix)): #pragma: no cover
                outerr= e
            _lib = None
        else:
            break
    if _lib is None: #pragma: no cover
        if not outerr is None:
            warnings.warn("libgalpy C extension module not loaded, because of error '%s' " % outerr,
                          galpyWarning)
        else:
            warnings.warn("libgalpy C extension module not loaded, because libgalpy%s image was not found" % _ext_suffix,
                          galpyWarning)
        _libgalpy_loaded= False    
    else:
        _libgalpy_loaded= True
    _libgalpy= _lib
    return (_libgalpy,_libgalpy_loaded)

def load_libgalpy_actionAngleTorus():
    global _libgalpy_actionAngleTorus
    global _libgalpy_actionAngleTorus_loaded
    if _libgalpy_actionAngleTorus_loaded is False \
            or not _libgalpy_actionAngleTorus is None:
        return (_libgalpy_actionAngleTorus,_libgalpy_actionAngleTorus_loaded)
    outerr= None
    for path in sys.path:
        if not os.path.isdir(path): continue
        try:
            if sys.platform == 'win32' and sys.version_info >= (3,8): # pragma: no cover
                # winmode=0x008 is easy-going way to call LoadLibraryExA
                _lib = ctypes.CDLL(os.path.join(path,'libgalpy_actionAngleTorus%s' % _ext_suffix),winmode=0x008)
            else:
                _lib = ctypes.CDLL(os.path.join(path,'libgalpy_actionAngleTorus%s' % _ext_suffix))
        except OSError as e:
            if os.path.exists(os.path.join(path,'libgalpy_actionAngleTorus%s' % _ext_suffix)): #pragma: no cover
                outerr= e
            _lib = None
        else:
            break
    if _lib is None: #pragma: no cover
        if not outerr is None:
            warnings.warn("libgalpy_actionAngleTorus C extension module not loaded, because of error '%s' " % outerr,
                          galpyWarningVerbose)
        else:
            warnings.warn("libgalpy_actionAngleTorus C extension module not loaded, because libgalpy%s image was not found" % _ext_suffix,
                          galpyWarningVerbose)
        _libgalpy_actionAngleTorus_loaded= False    
    else:
        _libgalpy_actionAngleTorus_loaded= True
    _libgalpy_actionAngleTorus= _lib
    return (_libgalpy_actionAngleTorus,_libgalpy_actionAngleTorus_loaded)
