###############################################################################
#   special.py: shared spherical harmonics utilities
###############################################################################
import numpy
import scipy
from packaging.version import parse as parse_version
from scipy.special import gammaln

_SCIPY_VERSION = parse_version(scipy.__version__)
if _SCIPY_VERSION < parse_version("1.15"):  # pragma: no cover
    from scipy.special import lpmn
else:
    from scipy.special import assoc_legendre_p_all


def compute_legendre(costheta, L, M, deriv=False):
    """
    Compute associated Legendre polynomials P_l^m(cos(theta)).

    Parameters
    ----------
    costheta : float
        Cosine of the polar angle.
    L : int
        Maximum degree + 1 (compute for 0 <= l < L).
    M : int
        Maximum order + 1 (compute for 0 <= m < M).
    deriv : bool, optional
        If True, also compute the derivative with respect to costheta.

    Returns
    -------
    PP : numpy.ndarray
        Associated Legendre polynomials, shape (L, M).
    dPP : numpy.ndarray
        Derivative with respect to costheta, shape (L, M). Only returned if deriv=True.

    Notes
    -----
    - 2026-02-11 - Written - Bovy (UofT)
    - 2026-02-13 - Moved to galpy.util.special - Bovy (UofT)
    """
    if _SCIPY_VERSION < parse_version("1.15"):  # pragma: no cover
        if deriv:
            PP, dPP = lpmn(M - 1, L - 1, costheta)
            return PP.T, dPP.T
        return lpmn(M - 1, L - 1, costheta)[0].T
    if deriv:
        PP, dPP = assoc_legendre_p_all(L - 1, M - 1, costheta, branch_cut=2, diff_n=1)
        return (
            numpy.swapaxes(PP[:, :M], 0, 1).T,
            numpy.swapaxes(dPP[:, :M], 0, 1).T,
        )
    return numpy.swapaxes(
        assoc_legendre_p_all(L - 1, M - 1, costheta, branch_cut=2)[0, :, :M],
        0,
        1,
    ).T


def sph_harm_normalization(L, M):
    """
    Compute the spherical harmonics normalization factor N_lm.

    Returns beta_lm = sqrt((2l+1)/(4*pi) * (l-m)!/(l+m)!) * (2 - delta_{m,0})
    for 0 <= l < L and 0 <= m < M.

    Parameters
    ----------
    L : int
        Maximum degree + 1 (compute for 0 <= l < L).
    M : int
        Maximum order + 1 (compute for 0 <= m < M).

    Returns
    -------
    numpy.ndarray
        Normalization factors, shape (L, M). Entries where m > l are zero.

    Notes
    -----
    - 2016-05-16 - Written as _Nroot - Aladdin Seaifan (UofT)
    - 2026-02-13 - Moved to galpy.util.special - Bovy (UofT)
    """
    NN = numpy.zeros((L, M), float)
    l = numpy.arange(0, L)[:, numpy.newaxis]
    m = numpy.arange(0, M)[numpy.newaxis, :]
    nLn = gammaln(l - m + 1) - gammaln(l + m + 1)
    NN[:, :] = ((2 * l + 1.0) / (4.0 * numpy.pi) * numpy.e**nLn) ** 0.5 * 2
    NN[:, 0] /= 2.0
    NN = numpy.tril(NN)
    return NN
