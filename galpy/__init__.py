__version__ = "1.10.0"
# Check whether a new version is available
import datetime
import http.client
import platform
import subprocess
import sys

from packaging.version import Version
from packaging.version import parse as parse_version

from .util.config import __config__, __orig__config__, configfilename, write_config


def latest_pypi_version(name):
    # First check whether there's internet, code below this works w/o
    # but is very slow without internet, so this is a quick check
    def online():
        try:
            conn = http.client.HTTPSConnection("8.8.8.8", timeout=5)
            conn.request("HEAD", "/")
            return True
        except Exception:  # pragma: no cover
            return False
        finally:
            conn.close()

    if not online():  # pragma: no cover
        return __version__
    # Essentially from https://stackoverflow.com/a/58649262
    try:  # Wrap everything in try/except to avoid any issues with connections
        latest_version = str(
            subprocess.run(
                [sys.executable, "-m", "pip", "install", f"{name}==random"],
                capture_output=True,
                text=True,
            )
        )
        latest_version = latest_version[latest_version.find("(from versions:") + 15 :]
        latest_version = latest_version[: latest_version.find(")")]
        latest_version = latest_version.replace(" ", "").split(",")
        # Remove any pre-releases
        latest_version = [v for v in latest_version if not Version(v).is_prerelease]
        # Latest is now the final one
        latest_version = latest_version[-1]
    except Exception:  # pragma: no cover
        return __version__
    else:
        # Offline can return 'none', then just return __version__ to avoid warning
        return __version__ if latest_version == "none" else latest_version


def check_pypi_version(name):
    latest_version = latest_pypi_version(name)
    if parse_version(latest_version) > parse_version(__version__):  # pragma: no cover
        return True
    else:
        return False


def print_version_warning():  # pragma: no cover
    print(
        "\033[91mA new version of galpy ({}) is available, please upgrade using pip/conda/... to get the latest features and bug fixes!\033[0m".format(
            latest_pypi_version("galpy")
        )
    )


_CHECK_VERSION_UPGRADE = (
    __config__.getboolean("version-check", "do-check")
    and not platform.system() == "Emscripten"
)
if _CHECK_VERSION_UPGRADE and hasattr(sys, "ps1"):  # pragma: no cover
    # Interactive session, https://stackoverflow.com/a/64523765
    if check_pypi_version("galpy"):
        print_version_warning()
elif _CHECK_VERSION_UPGRADE and __config__.getboolean(
    "version-check", "check-non-interactive"
):
    # Non-interactive session, only check once every
    # 'check-non-interactive-every' days
    today = datetime.date.today()
    last_check = datetime.date.fromisoformat(
        __config__.get("version-check", "last-non-interactive-check")
    )
    if today - last_check >= datetime.timedelta(
        days=__config__.getint("version-check", "check-non-interactive-every")
    ):
        if check_pypi_version("galpy"):  # pragma: no cover
            print_version_warning()
        # Also write the date of the last check to the configuration file
        __orig__config__.set(
            "version-check", "last-non-interactive-check", today.isoformat()
        )
        write_config(configfilename, __orig__config__)
