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
    deriv : bool or int, optional
        If False, only return P. If True or 1, also return dP/dx. If 2, also return d²P/dx².

    Returns
    -------
    PP : numpy.ndarray
        Associated Legendre polynomials, shape (L, M).
    dPP : numpy.ndarray
        Derivative with respect to costheta, shape (L, M). Only returned if deriv >= 1.
    d2PP : numpy.ndarray
        Second derivative with respect to costheta, shape (L, M). Only returned if deriv == 2.

    Notes
    -----
    - 2026-02-11 - Written - Bovy (UofT)
    - 2026-02-13 - Moved to galpy.util.special - Bovy (UofT)
    - 2026-02-18 - Added deriv=2 support - Bovy (UofT)
    """
    if _SCIPY_VERSION < parse_version("1.15"):  # pragma: no cover
        if deriv:
            PP, dPP = lpmn(M - 1, L - 1, costheta)
            PP = PP.T
            dPP = dPP.T
            if deriv == 2:
                d2PP = _compute_legendre_2nd_deriv(PP, dPP, costheta, L, M)
                return PP, dPP, d2PP
            return PP, dPP
        return lpmn(M - 1, L - 1, costheta)[0].T
    if deriv == 2:
        result = assoc_legendre_p_all(L - 1, M - 1, costheta, branch_cut=2, diff_n=2)
        PP = numpy.swapaxes(result[0][:, :M], 0, 1).T
        dPP = numpy.swapaxes(result[1][:, :M], 0, 1).T
        d2PP = numpy.swapaxes(result[2][:, :M], 0, 1).T
        return PP, dPP, d2PP
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


def _compute_legendre_2nd_deriv(PP, dPP, x, L, M):  # pragma: no cover
    """
    Compute d²P_l^m/dx² from the Legendre differential equation for scipy < 1.15.

    Uses: (1-x²) d²P/dx² - 2x dP/dx + [l(l+1) - m²/(1-x²)] P = 0
    => d²P/dx² = [2x dP/dx - l(l+1) P + m²/(1-x²) P] / (1-x²)

    At the poles (x = ±1), d²P/dx² diverges for m > 0. To avoid numerical
    issues, we clamp x slightly away from ±1 and recompute P and dP there.
    This is safe because the divergent d²P/dx² is always multiplied by sin²θ
    in the physical second derivative d²P/dθ², making the product finite.

    Parameters
    ----------
    PP : numpy.ndarray
        Associated Legendre polynomials, shape (L, M).
    dPP : numpy.ndarray
        First derivatives, shape (L, M).
    x : float
        costheta value.
    L : int
        Maximum degree + 1.
    M : int
        Maximum order + 1.

    Returns
    -------
    d2PP : numpy.ndarray
        Second derivatives, shape (L, M).
    """
    if abs(1.0 - x * x) < 1e-14:
        x = numpy.sign(x) * (1.0 - 1e-7) if x != 0.0 else x
        PP, dPP = lpmn(M - 1, L - 1, x)
        PP = PP.T
        dPP = dPP.T
    d2PP = numpy.zeros((L, M))
    one_minus_x2 = 1.0 - x * x
    for l in range(L):
        for m in range(min(l + 1, M)):
            d2PP[l, m] = (
                2.0 * x * dPP[l, m]
                - l * (l + 1) * PP[l, m]
                + m * m / one_minus_x2 * PP[l, m]
            ) / one_minus_x2
    return d2PP


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
