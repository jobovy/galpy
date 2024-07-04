# _load_extension_libs.py: centralized place to load the C extensions
import ctypes
import re
import subprocess
import sys
import sysconfig
import warnings
from pathlib import Path

from ..util import galpyWarning, galpyWarningVerbose

PY3 = sys.version > "3"
if PY3:
    _ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
else:  # pragma: no cover
    _ext_suffix = ".so"

_libgalpy = None
_libgalpy_loaded = None

_libgalpy_actionAngleTorus = None
_libgalpy_actionAngleTorus_loaded = None

_checked_openmp_issue = False


def _detect_openmp_issue():
    # Check whether we get an error of the type "OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.", which occurs, e.g., when using pip-installed galpy with conda-installed numpy and causes segmentation faults and general issues
    # This is a known issue with OpenMP and is not galpy's fault (GitHub Copilot suggested this comment!)

    # Check this by running a subprocess that tries to import the C library without this check
    try:
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from galpy.util._load_extension_libs import load_libgalpy; load_libgalpy(check_openmp_issue=False)",
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:  # pragma: no cover
        if re.match(
            r"OMP: Error #15: Initializing libomp[0-9]*.dylib, but found libomp[0-9]*.dylib already initialized",
            e.stderr.decode("utf-8"),
        ):
            warnings.warn(
                "Encountered OpenMP issue with multiple OpenMP runtimes causing conflicts and segmentation faults (similar to https://github.com/pytorch/pytorch/issues/78490). "
                "This generally happens when you combine a pip-installed galpy with a conda-installed numpy. "
                "If you are using conda, your best bet is to install galpy with conda-forge, because there is little advantage to using pip. "
                "If you insist on using pip, you can try installing galpy with the --no-binary galpy flag, but note that you have to have the GSL and OpenMP installed and available for linking (e.g., using 'brew install gsl libomp' on a Mac, but make sure to add the OpenMP library to your CFLAGS/LDFLAGS/LD_LIBRARY_PATH as it is keg-only; please refer to galpy's installation page for more info). "
                "Please open an Issue or Discussion on the galpy GitHub repository if you require further assistance.\n"
                "For now, we will disable the libgalpy C extension module to avoid segmentation faults and other issues, but be aware that this significantly slows down the code.",
                galpyWarning,
            )
            return True
    else:
        return False
    finally:
        global _checked_openmp_issue
        _checked_openmp_issue = True


def load_libgalpy(check_openmp_issue=True):
    global _libgalpy
    global _libgalpy_loaded
    if _libgalpy_loaded is False or not _libgalpy is None:
        return (_libgalpy, _libgalpy_loaded)
    if check_openmp_issue and not _checked_openmp_issue:
        if _detect_openmp_issue():  # pragma: no cover
            _libgalpy_loaded = False
            return (_libgalpy, _libgalpy_loaded)
    outerr = None
    # Add top-level galpy repository directory for pip install (-e) .,
    # just becomes site-packages for regular install
    paths = sys.path
    paths.append(str(Path(__file__).parent.parent.parent.absolute()))
    for path in [Path(p) for p in paths]:
        if not path.is_dir():
            continue
        try:
            if sys.platform == "win32" and sys.version_info >= (
                3,
                8,
            ):  # pragma: no cover
                # winmode=0x008 is easy-going way to call LoadLibraryExA
                _lib = ctypes.CDLL(str(path / f"libgalpy{_ext_suffix}"), winmode=0x008)
            else:
                _lib = ctypes.CDLL(str(path / f"libgalpy{_ext_suffix}"))
        except OSError as e:
            if (path / f"libgalpy{_ext_suffix}").exists():  # pragma: no cover
                outerr = e
            _lib = None
        else:
            break
    if _lib is None:  # pragma: no cover
        if not outerr is None:
            warnings.warn(
                f"libgalpy C extension module not loaded, because of error '{outerr}'",
                galpyWarning,
            )
        else:
            warnings.warn(
                f"libgalpy C extension module not loaded, because libgalpy{_ext_suffix} image was not found",
                galpyWarning,
            )
        _libgalpy_loaded = False
    else:
        _libgalpy_loaded = True
    _libgalpy = _lib
    return (_libgalpy, _libgalpy_loaded)


def load_libgalpy_actionAngleTorus():
    global _libgalpy_actionAngleTorus
    global _libgalpy_actionAngleTorus_loaded
    if (
        _libgalpy_actionAngleTorus_loaded is False
        or not _libgalpy_actionAngleTorus is None
    ):
        return (_libgalpy_actionAngleTorus, _libgalpy_actionAngleTorus_loaded)
    outerr = None
    # Add top-level galpy repository directory for pip install (-e) .,
    # just becomes site-packages for regular install
    paths = sys.path
    paths.append(str(Path(__file__).parent.parent.parent.absolute()))
    for path in [Path(p) for p in paths]:
        if not path.is_dir():
            continue
        try:
            if sys.platform == "win32" and sys.version_info >= (
                3,
                8,
            ):  # pragma: no cover
                # winmode=0x008 is easy-going way to call LoadLibraryExA
                _lib = ctypes.CDLL(
                    str(path / f"libgalpy_actionAngleTorus{_ext_suffix}"), winmode=0x008
                )
            else:
                _lib = ctypes.CDLL(
                    str(path / f"libgalpy_actionAngleTorus{_ext_suffix}")
                )
        except OSError as e:
            if (
                path / f"libgalpy_actionAngleTorus{_ext_suffix}"
            ).exists():  # pragma: no cover
                outerr = e
            _lib = None
        else:
            break
    if _lib is None:  # pragma: no cover
        if not outerr is None:
            warnings.warn(
                f"libgalpy_actionAngleTorus C extension module not loaded, because of error '{outerr}'",
                galpyWarningVerbose,
            )
        else:
            warnings.warn(
                f"libgalpy_actionAngleTorus C extension module not loaded, because libgalpy{_ext_suffix} image was not found",
                galpyWarningVerbose,
            )
        _libgalpy_actionAngleTorus_loaded = False
    else:
        _libgalpy_actionAngleTorus_loaded = True
    _libgalpy_actionAngleTorus = _lib
    return (_libgalpy_actionAngleTorus, _libgalpy_actionAngleTorus_loaded)
