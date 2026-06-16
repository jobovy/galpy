###############################################################################
#   Fallback for the sine and cosine integrals Si(x), Ci(x) on real x > 0.
#   Needed on torch (torch.special has no sici); jax has a native sici and numpy
#   uses scipy, so the router only routes torch here.
#
#   ``sici_fallback`` uses two regimes, each ~1e-14 vs scipy and AD-friendly:
#     - x <= x0: the convergent power series
#         Si(x) = sum_{k>=0} a_k/(2k+1),  a_0 = x, a_k = a_{k-1}*(-x^2/((2k)(2k+1)))
#         Ci(x) = gamma + ln(x) + sum_{k>=1} b_k/(2k),
#                 b_0 = 1, b_k = b_{k-1}*(-x^2/((2k-1)(2k))).
#       Terms are built iteratively (no factorials -> no overflow).
#     - x  > x0: the auxiliary functions f, g via Gauss-Laguerre quadrature of
#         f(x) = int_0^inf e^{-x t}/(1+t^2) dt,
#         g(x) = int_0^inf t e^{-x t}/(1+t^2) dt,
#       with the substitution t = u/x (u the Laguerre variable for the weight
#       e^{-u} on [0, inf)):
#         f(x) = (1/x)  * sum_i w_i / (1 + (u_i/x)^2),
#         g(x) = (1/x^2)* sum_i w_i u_i / (1 + (u_i/x)^2),
#       then  Si(x) = pi/2 - f*cos(x) - g*sin(x),  Ci(x) = f*sin(x) - g*cos(x).
#       The integrands are smooth and decay like e^{-x t}, so a moderate-order
#       Gauss-Laguerre rule is accurate for all x > x0.
#   Each branch's argument is clamped into its valid region wherever the OTHER
#   branch is selected, so the unused branch cannot overflow (1/x at x->0 in the
#   asymptotic branch, large x^2 in the series branch) or NaN-poison reverse-mode
#   gradients.
###############################################################################
import numpy

from ..._namespaces import asarray_on_device, device_of

_GAMMA = 0.5772156649015328606  # Euler-Mascheroni
_X0 = 6.0  # regime split
_NSERIES = 40  # power-series terms (x <= x0); ~1e-14 at x0
_NLAG = 64  # Gauss-Laguerre order (x > x0)
# Gauss-Laguerre nodes/weights for int_0^inf e^{-u} h(u) du, kept float64.
_LAG_U, _LAG_W = numpy.polynomial.laguerre.laggauss(_NLAG)


def sici_fallback(xp, x):
    """Return (Si(x), Ci(x)) for real x > 0, ~1e-14 vs scipy, AD-friendly."""
    x = xp.asarray(x) * 1.0
    small = x <= _X0
    # Clamp the dead region of each branch into its valid domain.
    xs = xp.where(small, x, xp.ones_like(x))  # series branch (x <= x0)
    xa = xp.where(small, _X0 * xp.ones_like(x), x)  # asymptotic branch (x > x0)

    # --- convergent power series (x <= x0) ---
    xs2 = xs * xs
    # Si: a_0 = x; a_k = a_{k-1} * (-x^2/((2k)(2k+1)))
    a = xs
    Sis = a  # k = 0 term / (2*0+1)
    for k in range(1, _NSERIES):
        a = a * (-xs2 / ((2.0 * k) * (2.0 * k + 1.0)))
        Sis = Sis + a / (2.0 * k + 1.0)
    # Ci: b_0 = 1; b_k = b_{k-1} * (-x^2/((2k-1)(2k)))
    b = xp.ones_like(xs)
    Cis = _GAMMA + xp.log(xs)
    for k in range(1, _NSERIES):
        b = b * (-xs2 / ((2.0 * k - 1.0) * (2.0 * k)))
        Cis = Cis + b / (2.0 * k)

    # --- Gauss-Laguerre auxiliary functions (x > x0) ---
    dev = device_of(x)
    u = asarray_on_device(xp, _LAG_U, dev)  # (N,)
    w = asarray_on_device(xp, _LAG_W, dev)  # (N,)
    tau = u / xa[..., None]  # u_i / x, shape (..., N)
    denom = 1.0 + tau * tau
    f = xp.sum(w / denom, axis=-1) / xa
    g = xp.sum(w * tau / denom, axis=-1) / xa
    cosx = xp.cos(xa)
    sinx = xp.sin(xa)
    half_pi = numpy.pi / 2.0
    Sia = half_pi - f * cosx - g * sinx
    Cia = f * sinx - g * cosx

    return xp.where(small, Sis, Sia), xp.where(small, Cis, Cia)
