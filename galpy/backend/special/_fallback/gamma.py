###############################################################################
#   Fallback for the gamma function on backends without a native gamma
#   (currently torch, which has gammaln but not gamma).
###############################################################################
import numpy

from .._router import gammaln


def gamma_fallback(xp, x):
    r"""Gamma(x) built from the native gammaln.

    For x > 0: Gamma(x) = exp(gammaln(x)).
    For x <= 0: the reflection formula Gamma(x) = pi / (sin(pi x) * Gamma(1-x))
    with Gamma(1-x) = exp(gammaln(1-x)) (1-x >= 1 > 0). This also yields the
    correct sign (from sin(pi x)) and the +/-inf poles at non-positive integers.

    Autodiff-friendly: under the eager xp.where both branches are evaluated
    everywhere, so each branch's argument is clamped into a safe region wherever
    the OTHER branch is selected -- otherwise gammaln's poles at non-positive
    integers would feed a NaN back through 0*inf and poison reverse-mode
    gradients at perfectly valid inputs.
    """
    x = xp.asarray(x) * 1.0
    positive = x > 0
    # negative integer poles (-1, -2, ...): scipy.special.gamma returns nan there;
    # the reflection formula would give a huge finite value (sin(pi*int) is ~1e-16,
    # not 0, in float64), so mask them out. x==0 is left to the reflection branch,
    # which yields +inf (sin(0)=0 -> pi/0), matching scipy.special.gamma(0)=inf.
    is_pole = (x < 0) & (x == xp.round(x))
    # x>0 branch: in the dead (x<=0) region use a safe +0.5 so gammaln stays finite.
    x_pos = xp.where(positive, x, 0.5 * xp.ones_like(x))
    pos = xp.exp(gammaln(x_pos))
    # x<=0 branch: keep the arg off the positive region AND the poles, so
    # sin(pi x) != 0 and gammaln(1-x) is finite (no NaN-poisoning of AD).
    x_neg = xp.where(positive | is_pole, -0.5 * xp.ones_like(x), x)
    refl = numpy.pi / (xp.sin(numpy.pi * x_neg) * xp.exp(gammaln(1.0 - x_neg)))
    out = xp.where(positive, pos, refl)
    return xp.where(is_pole, xp.full_like(x, numpy.nan), out)
