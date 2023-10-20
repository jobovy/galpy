import os
import pickle
import shutil
import tempfile
import warnings

import numpy
import scipy.linalg as linalg

from ..util.config import __config__

_SHOW_WARNINGS = __config__.getboolean("warnings", "verbose")


class galpyWarning(Warning):
    pass


# galpy warnings only shown if verbose = True in the configuration
class galpyWarningVerbose(galpyWarning):
    pass


def _warning(
    message, category=galpyWarning, filename="", lineno=-1, file=None, line=None
):
    if issubclass(category, galpyWarning):
        if not issubclass(category, galpyWarningVerbose) or _SHOW_WARNINGS:
            print("galpyWarning: " + str(message))
    else:
        print(warnings.formatwarning(message, category, filename, lineno))


warnings.showwarning = _warning


def save_pickles(savefilename, *args, **kwargs):
    """
    Save things to a pickle file.

    Parameters
    ----------
    savefilename : str
        Name of the file to save to; actual save operation will be performed on a temporary file and then that file will be shell mv'ed to savefilename.
    *args : tuple
        Things to pickle (as many as you want!).
    **kwargs : dict
        testKeyboardInterrupt : bool, optional
            If True, raises KeyboardInterrupt to test if the function can handle it.

    Returns
    -------
    None

    Notes
    -----
    - 2010-? - Written - Bovy (NYU)
    - 2011-08-23 - generalized and added to galpy.util - Bovy (NYU)
    """

    saving = True
    interrupted = False
    file, tmp_savefilename = tempfile.mkstemp()  # savefilename+'.tmp'
    os.close(file)  # Easier this way
    while saving:
        try:
            savefile = open(tmp_savefilename, "wb")
            file_open = True
            if kwargs.get("testKeyboardInterrupt", False) and not interrupted:
                raise KeyboardInterrupt
            for f in args:
                pickle.dump(f, savefile, pickle.HIGHEST_PROTOCOL)
            savefile.close()
            file_open = False
            shutil.move(tmp_savefilename, savefilename)
            saving = False
            if interrupted:
                raise KeyboardInterrupt
        except KeyboardInterrupt:
            if not saving:
                raise
            print("KeyboardInterrupt ignored while saving pickle ...")
            interrupted = True
        finally:
            if file_open:
                savefile.close()


def logsumexp(arr, axis=0):
    """Faster logsumexp?"""
    minarr = numpy.amax(arr, axis=axis)
    if axis == 1:
        minarr = numpy.reshape(minarr, (arr.shape[0], 1))
    if axis == 0:
        minminarr = numpy.tile(minarr, (arr.shape[0], 1))
    elif axis == 1:
        minminarr = numpy.tile(minarr, (1, arr.shape[1]))
    elif axis == None:
        minminarr = numpy.tile(minarr, arr.shape)
    else:
        raise NotImplementedError("'galpy.util.logsumexp' not implemented for axis > 2")
    if axis == 1:
        minarr = numpy.reshape(minarr, (arr.shape[0]))
    return minarr + numpy.log(numpy.sum(numpy.exp(arr - minminarr), axis=axis))


_TINY = 0.000000001


def stable_cho_factor(x, tiny=_TINY):
    """
    Stable version of the cholesky decomposition

    Parameters
    ----------
    x : numpy.ndarray
        Positive definite matrix
    tiny : float, optional
        Tiny number to add to the covariance matrix to make the decomposition stable (has a default)

    Returns
    -------
    tuple
        (L, lowerFlag) - output from scipy.linalg.cho_factor for lower=True

    Notes
    -----
    - 2009-09-25 - Written - Bovy (NYU)
    """

    return linalg.cho_factor(
        x + numpy.sum(numpy.diag(x)) * tiny * numpy.eye(x.shape[0]), lower=True
    )


def fast_cholesky_invert(A, logdet=False, tiny=_TINY):
    """
    Invert a positive definite matrix by using its Cholesky decomposition.

    Parameters
    ----------
    A : numpy.ndarray
        Matrix to be inverted.
    logdet : bool, optional
        If True, return the logarithm of the determinant as well.
    tiny : float, optional
        Tiny number to add to the covariance matrix to make the decomposition stable (has a default).

    Returns
    -------
    numpy.ndarray or tuple
        If logdet is False, the inverse of the input matrix. If logdet is True, the inverse and the logarithm of the determinant of the input matrix.

    Notes
    -----
    - 2009-10-07 - Written - Bovy (NYU)
    """

    L = stable_cho_factor(A, tiny=tiny)
    if logdet:
        return (
            linalg.cho_solve(L, numpy.eye(A.shape[0])),
            2.0 * numpy.sum(numpy.log(numpy.diag(L[0]))),
        )
    else:
        return linalg.cho_solve(L, numpy.eye(A.shape[0]))


def _rotate_to_arbitrary_vector(v, a, inv=False, _dontcutsmall=False):
    r"""Return a rotation matrix that rotates v to align with unit vector a
    i.e. R . v = |v|\hat{a}"""
    normv = v / numpy.tile(numpy.sqrt(numpy.sum(v**2.0, axis=1)), (3, 1)).T
    rotaxis = numpy.cross(normv, a)
    rotaxis /= numpy.tile(numpy.sqrt(numpy.sum(rotaxis**2.0, axis=1)), (3, 1)).T
    crossmatrix = numpy.empty((len(v), 3, 3))
    crossmatrix[:, 0, :] = numpy.cross(rotaxis, [1, 0, 0])
    crossmatrix[:, 1, :] = numpy.cross(rotaxis, [0, 1, 0])
    crossmatrix[:, 2, :] = numpy.cross(rotaxis, [0, 0, 1])
    costheta = numpy.dot(normv, a)
    sintheta = numpy.sqrt(1.0 - costheta**2.0)
    if inv:
        sgn = 1.0
    else:
        sgn = -1.0
    out = (
        numpy.tile(costheta, (3, 3, 1)).T * numpy.tile(numpy.eye(3), (len(v), 1, 1))
        + sgn * numpy.tile(sintheta, (3, 3, 1)).T * crossmatrix
        + numpy.tile(1.0 - costheta, (3, 3, 1)).T
        * (rotaxis[:, :, numpy.newaxis] * rotaxis[:, numpy.newaxis, :])
    )
    if not _dontcutsmall:
        out[numpy.fabs(costheta - 1.0) < 10.0**-10.0] = numpy.eye(3)
        out[numpy.fabs(costheta + 1.0) < 10.0**-10.0] = -numpy.eye(3)
    return out
