###############################################################################
# nemo_util.py: some utilities for handling NEMO snapshots
###############################################################################
import os
import subprocess
import tempfile

import numpy


def read(filename, ext=None, swapyz=False):
    """
    Read a NEMO snapshot file consisting of mass, position, velocity.

    Parameters
    ----------
    filename : str
        Name of the file.
    ext : str, optional
        If set, 'nemo' for NEMO binary format, otherwise assumed ASCII; if not set, gleaned from extension.
    swapyz : bool, optional
        If True, swap the y and z axes in the output (only for position and velocity).

    Returns
    -------
    ndarray
        Array of shape (nbody, ndim, nt).

    Notes
    -----
    - 2015-11-18 - Written - Bovy (UofT).
    """
    if ext is None and filename.split(".")[-1] == "nemo":
        ext = "nemo"
    elif ext is None:
        ext = "dat"
    # Convert to ASCII if necessary
    if ext.lower() == "nemo":
        file_handle, asciifilename = tempfile.mkstemp()
        os.close(file_handle)
        stderr = open("/dev/null", "w")
        try:
            subprocess.check_call(["s2a", filename, asciifilename])  # ,stderr=stderr)
        except subprocess.CalledProcessError:
            os.remove(asciifilename)
        finally:
            stderr.close()
    else:
        asciifilename = filename
    # Now read
    out = numpy.loadtxt(asciifilename, comments="#")
    if ext.lower() == "nemo":
        os.remove(asciifilename)
    if swapyz:
        out[:, [2, 3]] = out[:, [3, 2]]
        out[:, [5, 6]] = out[:, [6, 5]]
    # Get the number of snapshots
    nt = (_wc(asciifilename) - out.shape[0]) // 13  # 13 comments/snapshot
    out = numpy.reshape(out, (nt, out.shape[0] // nt, out.shape[1]))
    return numpy.swapaxes(numpy.swapaxes(out, 0, 1), 1, 2)


def _wc(filename):
    try:
        return int(subprocess.check_output(["wc", "-l", filename]).split()[0])
    except subprocess.CalledProcessError:
        return numpy.nan
