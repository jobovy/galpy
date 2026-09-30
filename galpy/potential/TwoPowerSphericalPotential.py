###############################################################################
#   TwoPowerSphericalPotential.py: General class for potentials derived from
#                                  densities with two power-laws
#
#                                                    amp
#                             rho(r)= ------------------------------------
#                                      (r/a)^\alpha (1+r/a)^(\beta-\alpha)
###############################################################################
import functools
import math

import numpy
from scipy import optimize, special

from ..backend import (
    branch_where,
    coerce_coords,
    get_namespace,
    radial_limits,
)
from ..backend._coerce import mask_where, power_series
from ..backend._namespaces import has_concrete_truth_value
from ..backend.special.incomplete_beta import (
    incomplete_beta_hi_xp,
    incomplete_beta_series_coeffs,
    incomplete_beta_series_xp,
    incomplete_beta_xp,
)
from ..util import conversion
from ..util._optional_deps import _APY_LOADED
from ..util.special import (
    hyp2f1_1,
    incomplete_beta,
    incomplete_beta_hi,
    incomplete_beta_split,
)
from .Potential import Potential, kms_to_kpcGyrDecorator

# NFW's closed forms subtract terms of order 1/r^2 that cancel to leading
# order, losing ~eps/x^2 (force, mass) and ~eps/x^3 (second derivatives) at
# x = r/a << 1 (e.g. R2deriv 3e2 off at x=1e-6). Below _NFW_SMALL_X they use,
# with t = x/(2+x) and S = sum_{m>=1} t^(2m+1)/(2m+1) (log1p(x) = 2 atanh(t),
# all terms positive; truncation < 1e-17 at x = 0.25)
#   h(x) = log1p(x) - x/(1+x) = 2 t^2/(1+t) + 2 S        [dPhi/dr = h/r^2]
#   k(x) = x^2/(1+x)^2 - 2h(x) = -4 t^3/(1+t)^2 - 4 S    [Phi'' = k/r^3]
# above it the original formulas. The C implementation (NFWPotential.c) does
# the same.
_NFW_SMALL_X = 0.25
_NFW_S = [1.0 / (2 * m + 3) for m in range(8)]


def _nfw_tu_S(xp, x):
    """(t, t/(1+t), S) for h and k above; S/t^3 = 1/3 + O(t^2) keeps the
    derivative finite at t = 0"""
    t = x / (2.0 + x)
    t2 = t * t
    S = t2 * t * (_NFW_S[0] + power_series(xp, t2, _NFW_S[1:], 1))
    return t, t / (1.0 + t), S


def _nfw_hx(xp, x):
    t, u, S = _nfw_tu_S(xp, x)
    return 2.0 * (t * u + S)


def _nfw_masked(xp, small, r, a):
    """(r, x = r/a) on the small-x mask, 0.05 a elsewhere (keeping the dead
    branches finite).

    Masks r BEFORE dividing by a: masking x = r/a after would leave d(r/a)/da
    = -inf (r = inf) in the dead branch's backward, 0 * inf = NaN. Use the
    returned r, not x * a: the latter's d/da cancels only in exact arithmetic."""
    rs = mask_where(xp, small, r, 0.05 * a)
    return rs, rs / a


def _nfw_h(xp, small, r, a):
    """h(x) / r^3 on the small-x mask (dPhi/dr = h / r^2)."""
    rs, xs = _nfw_masked(xp, small, r, a)
    return _nfw_hx(xp, xs) / (rs * rs * rs)


def _nfw_hk5(xp, small, r, a):
    """(h(x), k(x)) / r^5 on the small-x mask."""
    rs, xs = _nfw_masked(xp, small, r, a)
    t, u, S = _nfw_tu_S(xp, xs)
    r5 = rs * rs * rs * rs * rs
    return 2.0 * (t * u + S) / r5, -4.0 * (t * u * u + S) / r5


# TwoPowerSphericalPotential's potential through two incomplete beta integrals
# (w = x/(1+x), x = r/a):
#   Phi = -(1/a) [M(x)/x + O(x)],  M = B_w(3-alpha, beta-3),
#   O = int_x^inf t^(1-alpha) (1+t)^(alpha-beta) dt = B_{1-w}(beta-2, 2-alpha),
# (galpy.util.special.incomplete_beta). Unlike the closed forms in Gamma(beta-3)
# and hyp2f1(..., -a/r), this has no cancellation as beta -> 3 or alpha -> 2
# and no Gamma overflow at large beta.
# Forces and second derivatives from the same M (amp = 1; 4 pi rho a^3 = D =
# w^-alpha s^beta): dPhi/dr / r = M/(x a)^3 and, by Poisson,
#   Phi'' = 4 pi rho - 2 dPhi/dr / r,  Phi'' - dPhi/dr / r = 4 pi rho - 3 dPhi/dr / r.
# Below the split c of incomplete_beta, M = w^p s^q / p (1 + G) with
# G = 2F1(1, p+q; p+1; w) - 1 = (p+q)/(p+1) w 2F1(1, p+q+1; p+2; w), so with
# E = D/p these are E (1 + G), E (1 - alpha - 2 G) and -E (alpha + 3 G): the
# x^-alpha terms of 4 pi rho and k M/x^3 that cancel at alpha = 1 (k = 2) and
# alpha = 0 (k = 3) are subtracted in closed form. Above c, D - k M/x^3
# directly: those cancellations are small-x ones (and for alpha < beta,
# G >~ 1/2 there). No hyp2f1(..., -r/a): that was 3e-3 off at
# beta = 3 +- 1e-12 and NaN at large beta and r.
def _tp_radial(alpha, beta, w, s, hess):
    """(M/x^3,) or, if hess, (M/x^3, Phi'' a^3, (Phi'' - Phi'/r) a^3)"""
    p, q = 3.0 - alpha, beta - 3.0
    c = incomplete_beta_split(p, q)
    if numpy.ndim(w) == 0:
        if w <= c:
            return _tp_radial_lo(alpha, beta, w, s, hess)
        return _tp_radial_hi(alpha, beta, w, s, c, hess)
    w = numpy.asarray(w, dtype=float)
    s = numpy.asarray(s, dtype=float)
    lo = w <= c
    out = numpy.empty((3 if hess else 1,) + w.shape)
    if numpy.any(lo):
        out[:, lo] = _tp_radial_lo(alpha, beta, w[lo], s[lo], hess)
    if not numpy.all(lo):
        out[:, ~lo] = _tp_radial_hi(alpha, beta, w[~lo], s[~lo], c, hess)
    return tuple(out)


def _tp_radial_lo(alpha, beta, w, s, hess):
    p, q = 3.0 - alpha, beta - 3.0
    E = w**-alpha * s**beta / p
    G = (p + q) / (p + 1.0) * w * hyp2f1_1(p + q + 1.0, p + 2.0, w)
    if not hess:
        return (E * (1.0 + G),)
    return E * (1.0 + G), E * (1.0 - alpha - 2.0 * G), -E * (alpha + 3.0 * G)


def _tp_radial_hi(alpha, beta, w, s, c, hess):
    m = incomplete_beta_hi(3.0 - alpha, beta - 3.0, s, c) * (s / w) ** 3
    if not hess:
        return (m,)
    D = w**-alpha * s**beta
    return m, D - 2.0 * m, D - 3.0 * m


def _tp_radial_xp(xp, alpha, beta, w, s, hess):
    """_tp_radial for backend w, s (as xp.stack of its outputs)"""
    p, q = 3.0 - alpha, beta - 3.0
    c = incomplete_beta_split(p, q)
    lo = w <= c
    one = xp.ones_like(w * 1.0)

    def below():
        wl = xp.where(lo, w, 0.5 * c * one)
        sl = xp.where(lo, s, (1.0 - 0.5 * c) * one)
        E = wl**-alpha * sl**beta / p
        # G = 2F1(1, p+q; p+1; w) - 1: the M series without its n = 0 term
        sgn, loga, ns = incomplete_beta_series_coeffs("lo", p, q, c)
        G = incomplete_beta_series_xp(xp, (sgn[1:], loga[1:], ns[1:]), wl)
        if not hess:
            return xp.stack([E * (1.0 + G)])
        return xp.stack(
            [E * (1.0 + G), E * (1.0 - alpha - 2.0 * G), -E * (alpha + 3.0 * G)]
        )

    def above():
        sh = xp.where(lo, 0.5 * (1.0 - c) * one, s)
        wh = xp.where(lo, (1.0 - 0.5 * (1.0 - c)) * one, w)  # 1 - sh, exactly
        m = incomplete_beta_hi_xp(xp, p, q, sh, wh, c) * (sh / wh) ** 3
        if not hess:
            return xp.stack([m])
        D = wh**-alpha * sh**beta
        return xp.stack([m, D - 2.0 * m, D - 3.0 * m])

    return branch_where(xp, lo, below, above)


def _force_at_infinity(beta, a):
    """dPhi/dr (amp = 1) as r -> inf: M(x) / (x a)^2 -> 0 for beta > 1,
    1 / (2 a^2) at beta = 1 (M ~ x^2 / 2), inf for beta < 1"""
    if beta > 1.0:
        return 0.0
    return 0.5 / a**2.0 if beta == 1.0 else numpy.inf


def _limit_at_infinite_radius(component=None):
    """The value at r = inf, where the expressions are inf * 0 (R f(r),
    z^2 f(r) / r^2, ...). The force along an infinite coordinate is
    -dPhi/dr(inf) (_force_at_infinity); the transverse force and every second
    derivative -> 0. With both coordinates infinite the direction is
    undefined: 0 if dPhi/dr -> 0, else NaN. Only for beta > 0: at beta <= 0
    the density does not fall off and these limits are finite or divergent, so
    r = inf is NaN there. amp = 0 is 0 (no 0 * inf). ``component``: "R" or "z"
    for the forces, None for the second derivatives. Finite inputs are untouched;
    under a trace the infinite entries are masked (finite stand-ins, so the
    unused branch cannot NaN the gradient)."""

    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, R, z, phi=0.0, t=0.0):
            xp = get_namespace(R, z)
            R, z = coerce_coords(xp, R, z)
            Rinf, zinf = xp.isinf(R), xp.isinf(z)
            inf = Rinf | zinf
            anyinf = xp.any(inf)
            if has_concrete_truth_value(anyinf) and not bool(anyinf):
                return method(self, R, z, phi=phi, t=t)
            out = method(
                self, xp.where(inf, 1.0, R), xp.where(inf, 0.0, z), phi=phi, t=t
            )
            amp0 = self._amp == 0.0  # traced for a differentiated amp
            if has_concrete_truth_value(amp0):
                if bool(amp0):
                    return xp.where(inf, 0.0, out)
                amp0 = None
            if self.beta <= 0.0:
                lim = numpy.nan
            elif component is None:
                lim = 0.0
            else:
                F = _force_at_infinity(self.beta, self.a)
                X, Xinf, Yinf = (R, Rinf, zinf) if component == "R" else (z, zinf, Rinf)
                along = -F * xp.sign(xp.where(Xinf, X, 1.0))
                both = 0.0 if self.beta > 1.0 else numpy.nan
                lim = xp.where(Xinf & Yinf, both, xp.where(Xinf, along, 0.0))
            if amp0 is not None:
                lim = xp.where(amp0, 0.0, lim)
            return xp.where(inf, lim, out)

        return wrapper

    return decorator


if _APY_LOADED:
    from astropy import units


class TwoPowerSphericalPotential(Potential):
    """Class that implements spherical potentials that are derived from
    two-power density models

    .. math::

        \\rho(r) = \\frac{\\mathrm{amp}}{4\\,\\pi\\,a^3}\\,\\frac{1}{(r/a)^\\alpha\\,(1+r/a)^{\\beta-\\alpha}}
    """

    def __init__(
        self, amp=1.0, a=5.0, alpha=1.5, beta=3.5, normalize=False, ro=None, vo=None
    ):
        """
        Initialize a two-power-density potential.

        Parameters
        ----------
        amp : float or Quantity, optional
            Amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass or Gxmass.
        a : float or Quantity, optional
            Scale radius.
        alpha : float, optional
            Inner power.
        beta : float, optional
            Outer power.
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - Started - 2010-07-09 - Bovy (NYU)
        """
        # Instantiate
        Potential.__init__(self, amp=amp, ro=ro, vo=vo, amp_units="mass")
        # _specialSelf for special cases (Dehnen class, Dehnen core, Hernquist, Jaffe, NFW)
        self._specialSelf = None
        if (
            (self.__class__ == TwoPowerSphericalPotential)
            & (alpha == round(alpha))
            & (beta == round(beta))
        ):
            if int(alpha) == 0 and int(beta) == 4:
                self._specialSelf = DehnenCoreSphericalPotential(
                    amp=1.0, a=a, normalize=False
                )
            elif int(alpha) == 1 and int(beta) == 4:
                self._specialSelf = HernquistPotential(amp=1.0, a=a, normalize=False)
            elif int(alpha) == 2 and int(beta) == 4:
                self._specialSelf = JaffePotential(amp=1.0, a=a, normalize=False)
            elif int(alpha) == 1 and int(beta) == 3:
                self._specialSelf = NFWPotential(amp=1.0, a=a, normalize=False)
        # correcting quantities
        a = conversion.parse_length(a, ro=self._ro)
        # setting properties
        self.a = a
        self._scale = self.a
        self.alpha = alpha
        self.beta = beta
        self.hasC = True
        self._backend_compatible = True
        self.hasC_dxdv = True
        self.hasC_dxdv3d = True  # full 3D Hessian (R2deriv/z2deriv/Rzderiv) in C
        self.hasC_dens = True
        if normalize or (
            isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
        ):  # pragma: no cover
            self.normalize(normalize)
        return None

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        if self._specialSelf is not None:
            return self._specialSelf._evaluate(R, z, phi=phi, t=t)
        xp = get_namespace(R, z)
        R, z = coerce_coords(xp, R, z)
        # Phi(0) = -B(2-alpha, beta-2)/a is finite for alpha < 2 only
        phi0 = (
            -special.beta(2.0 - self.alpha, self.beta - 2.0) / self.a
            if self.alpha < 2.0
            else -numpy.inf
        )
        return radial_limits(
            xp.sqrt(R**2.0 + z**2.0),
            lambda r: self._evaluate_ibeta(xp, r),
            at0=phi0,
            atinf=0.0,
            numpy_too=True,
        )

    def _evaluate_ibeta(self, xp, r):
        """Phi = -(M(x)/x + O(x))/a as incomplete beta integrals (see incomplete_beta)"""
        ibeta = (
            incomplete_beta
            if xp is numpy
            else functools.partial(incomplete_beta_xp, xp)
        )
        x = r / self.a
        w, s = x / (1.0 + x), 1.0 / (1.0 + x)
        M = ibeta(3.0 - self.alpha, self.beta - 3.0, w, s)
        O = ibeta(self.beta - 2.0, 2.0 - self.alpha, s, w)
        return -(M / x + O) / self.a

    def _radial(self, xp, r, hess):
        """(dPhi/dr / r,) or (dPhi/dr / r, Phi'', Phi'' - dPhi/dr / r) at r
        (amp = 1; see _tp_radial)"""
        radial = _tp_radial if xp is numpy else functools.partial(_tp_radial_xp, xp)
        x = r / self.a
        out = radial(self.alpha, self.beta, x / (1.0 + x), 1.0 / (1.0 + x), hess)
        a3 = self.a**3.0
        return [f / a3 for f in out]

    @_limit_at_infinite_radius("R")
    def _Rforce(self, R, z, phi=0.0, t=0.0):
        if self._specialSelf is not None:
            return self._specialSelf._Rforce(R, z, phi=phi, t=t)
        xp = get_namespace(R, z)
        R, z = coerce_coords(xp, R, z)
        return -R * self._radial(xp, xp.sqrt(R**2.0 + z**2.0), False)[0]

    @_limit_at_infinite_radius("z")
    def _zforce(self, R, z, phi=0.0, t=0.0):
        if self._specialSelf is not None:
            return self._specialSelf._zforce(R, z, phi=phi, t=t)
        xp = get_namespace(R, z)
        R, z = coerce_coords(xp, R, z)
        return -z * self._radial(xp, xp.sqrt(R**2.0 + z**2.0), False)[0]

    def _dens(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        R, z = coerce_coords(xp, R, z)
        r = xp.sqrt(R**2.0 + z**2.0)
        return radial_limits(
            r,
            lambda r: (
                (self.a / r) ** self.alpha
                / (1.0 + r / self.a) ** (self.beta - self.alpha)
                / 4.0
                / math.pi
                / self.a**3.0
            ),
            atinf=0.0,
        )

    def _ddensdr(self, r, t=0.0):
        return (
            -self._amp
            * (self.a / r) ** (self.alpha - 1.0)
            * (1.0 + r / self.a) ** (self.alpha - self.beta - 1.0)
            * (self.a * self.alpha + r * self.beta)
            / r**2
            / 4.0
            / numpy.pi
            / self.a**3.0
        )

    def _d2densdr2(self, r, t=0.0):
        return (
            self._amp
            * (self.a / r) ** (self.alpha - 2.0)
            * (1.0 + r / self.a) ** (self.alpha - self.beta - 2.0)
            * (
                self.alpha * (self.alpha + 1.0) * self.a**2
                + 2.0 * self.alpha * self.a * (self.beta + 1.0) * r
                + self.beta * (self.beta + 1.0) * r**2
            )
            / r**4
            / 4.0
            / numpy.pi
            / self.a**3.0
        )

    def _ddenstwobetadr(self, r, beta=0):
        """
        Evaluate the radial density derivative x r^(2beta) for this potential.

        Parameters
        ----------
        r : float
            Spherical radius.
        beta : float, optional
            Power of r in the density derivative. Default is 0.

        Returns
        -------
        float
            The derivative of the density times r^(2beta).

        Notes
        -----
        - 2021-02-14 - Written - Bovy (UofT)

        """
        return (
            self._amp
            / 4.0
            / numpy.pi
            / self.a**3.0
            * r ** (2.0 * beta - 2.0)
            * (self.a / r) ** (self.alpha - 1.0)
            * (1.0 + r / self.a) ** (self.alpha - self.beta - 1.0)
            * (self.a * (2.0 * beta - self.alpha) + r * (2.0 * beta - self.beta))
        )

    @_limit_at_infinite_radius()
    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        R, z = coerce_coords(xp, R, z)
        r2 = R**2.0 + z**2.0
        f1, f0, _ = self._radial(xp, xp.sqrt(r2), True)
        return (R**2.0 * f0 + z**2.0 * f1) / r2

    @_limit_at_infinite_radius()
    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        R, z = coerce_coords(xp, R, z)
        r2 = R**2.0 + z**2.0
        return R * z * self._radial(xp, xp.sqrt(r2), True)[2] / r2

    def _z2deriv(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        R, z = coerce_coords(xp, R, z)
        return self._R2deriv(xp.abs(z), R)  # Spherical potential

    def _mass(self, R, z=None, t=0.0):
        if z is not None:
            raise AttributeError  # use general implementation
        # finite total mass B(3-alpha, beta-3) for beta > 3, divergent otherwise
        # (the formula is 0 * inf = NaN at R = inf)
        # special.beta, not a ratio of gammas: those overflow for beta > ~170
        mtot = (
            special.beta(3.0 - self.alpha, self.beta - 3.0)
            if self.beta > 3.0
            else numpy.inf
        )
        xp = get_namespace(R)
        (R,) = coerce_coords(xp, R)
        ibeta = (
            incomplete_beta
            if xp is numpy
            else functools.partial(incomplete_beta_xp, xp)
        )

        def M(R):
            x = R / self.a
            return ibeta(
                3.0 - self.alpha, self.beta - 3.0, x / (1.0 + x), 1.0 / (1.0 + x)
            )

        return radial_limits(R, M, at0=0.0, atinf=mtot, numpy_too=True)


class DehnenSphericalPotential(TwoPowerSphericalPotential):
    """Class that implements the Dehnen Spherical Potential from `Dehnen (1993) <https://ui.adsabs.harvard.edu/abs/1993MNRAS.265..250D>`_

    .. math::

          \\rho(r) = \\frac{\\mathrm{amp}(3-\\alpha)}{4\\,\\pi\\,a^3}\\,\\frac{1}{(r/a)^{\\alpha}\\,(1+r/a)^{4-\\alpha}}
    """

    def __init__(self, amp=1.0, a=1.0, alpha=1.5, normalize=False, ro=None, vo=None):
        """
        Initialize a Dehnen Spherical Potential.

        Parameters
        ----------
        amp : float or Quantity, optional
            Amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass or Gxmass.
        a : float or Quantity, optional
            Scale radius.
        alpha : float, optional
            Inner power, restricted to [0, 3).
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - Started - Starkman (UofT) - 2019-10-07
        """
        if (alpha < 0.0) or (alpha >= 3.0):
            raise OSError("DehnenSphericalPotential requires 0 <= alpha < 3")
        # instantiate
        TwoPowerSphericalPotential.__init__(
            self, amp=amp, a=a, alpha=alpha, beta=4, normalize=normalize, ro=ro, vo=vo
        )
        # make special-self and protect subclasses
        self._specialSelf = None
        if (self.__class__ == DehnenSphericalPotential) & (alpha == round(alpha)):
            if round(alpha) == 0:
                self._specialSelf = DehnenCoreSphericalPotential(
                    amp=1.0, a=a, normalize=False
                )
            elif round(alpha) == 1:
                self._specialSelf = HernquistPotential(amp=1.0, a=a, normalize=False)
            elif round(alpha) == 2:
                self._specialSelf = JaffePotential(amp=1.0, a=a, normalize=False)
        # set properties
        self.hasC = True
        self._backend_compatible = True
        self.hasC_dxdv = True
        self.hasC_dxdv3d = True  # full 3D Hessian (R2deriv/z2deriv/Rzderiv) in C
        self.hasC_dens = True
        return None

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        if self._specialSelf is not None:
            return self._specialSelf._evaluate(R, z, phi=phi, t=t)
        else:  # valid for alpha != 2, 3
            xp = get_namespace(R, z)
            R, z = coerce_coords(xp, R, z)
            r = xp.sqrt(R**2.0 + z**2.0)
            norm = self.a * (2.0 - self.alpha) * (3.0 - self.alpha)
            return radial_limits(
                r,
                lambda r: (
                    -(1.0 - 1.0 / (1.0 + self.a / r) ** (2.0 - self.alpha)) / norm
                ),
                at0=-1.0 / norm,
            )

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        if self._specialSelf is not None:
            return self._specialSelf._Rforce(R, z, phi=phi, t=t)
        else:
            xp = get_namespace(R, z)
            R, z = coerce_coords(xp, R, z)
            r = xp.sqrt(R**2.0 + z**2.0)
            return (
                -R
                / r**self.alpha
                * (self.a + r) ** (self.alpha - 3.0)
                / (3.0 - self.alpha)
            )

    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        if self._specialSelf is not None:
            return self._specialSelf._R2deriv(R, z, phi=phi, t=t)
        xp = get_namespace(R, z)
        R, z = coerce_coords(xp, R, z)
        a, alpha = self.a, self.alpha
        r = xp.sqrt(R**2.0 + z**2.0)
        # formula not valid for alpha=2,3, (integers?)
        return (
            r ** (-2.0 - alpha)
            * (r + a) ** (alpha - 4.0)
            * (-a * r**2.0 + (2.0 * R**2.0 - z**2.0) * r + a * alpha * R**2.0)
            / (alpha - 3.0)
        )

    def _zforce(self, R, z, phi=0.0, t=0.0):
        if self._specialSelf is not None:
            return self._specialSelf._zforce(R, z, phi=phi, t=t)
        else:
            xp = get_namespace(R, z)
            R, z = coerce_coords(xp, R, z)
            r = xp.sqrt(R**2.0 + z**2.0)
            return (
                -z
                / r**self.alpha
                * (self.a + r) ** (self.alpha - 3.0)
                / (3.0 - self.alpha)
            )

    def _z2deriv(self, R, z, phi=0.0, t=0.0):
        return self._R2deriv(z, R, phi=phi, t=t)

    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        if self._specialSelf is not None:
            return self._specialSelf._Rzderiv(R, z, phi=phi, t=t)
        xp = get_namespace(R, z)
        R, z = coerce_coords(xp, R, z)
        a, alpha = self.a, self.alpha
        r = xp.sqrt(R**2.0 + z**2.0)
        return (
            R * z * r ** (-2.0 - alpha) * (a + r) ** (alpha - 4.0) * (3 * r + a * alpha)
        ) / (alpha - 3)

    def _dens(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        R, z = coerce_coords(xp, R, z)
        r = xp.sqrt(R**2.0 + z**2.0)
        return radial_limits(
            r,
            lambda r: (
                (self.a / r) ** self.alpha
                / (1.0 + r / self.a) ** (4.0 - self.alpha)
                / 4.0
                / math.pi
                / self.a**3.0
            ),
            atinf=0.0,
        )

    def _mass(self, R, z=None, t=0.0):
        if z is not None:
            raise AttributeError  # use general implementation
        return radial_limits(  # written so it works for r=numpy.inf
            R,
            lambda R: (
                1.0 / (1.0 + self.a / R) ** (3.0 - self.alpha) / (3.0 - self.alpha)
            ),
            at0=0.0,
        )


class DehnenCoreSphericalPotential(DehnenSphericalPotential):
    """Class that implements the Dehnen Spherical Potential from `Dehnen (1993) <https://ui.adsabs.harvard.edu/abs/1993MNRAS.265..250D>`_ with alpha=0 (corresponding to an inner core)

    .. math::

          \\rho(r) = \\frac{\\mathrm{amp}}{12\\,\\pi\\,a^3}\\,\\frac{1}{(1+r/a)^{4}}
    """

    def __init__(self, amp=1.0, a=1.0, normalize=False, ro=None, vo=None):
        """
        Initialize a cored Dehnen Spherical Potential; note that the amplitude definition used here does NOT match that of Dehnen (1993)

        Parameters
        ----------
        amp : float, optional
            Amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass or Gxmass
        a : float or Quantity, optional
            Scale radius.
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2019-10-07 - Started - Starkman (UofT)
        """
        DehnenSphericalPotential.__init__(
            self, amp=amp, a=a, alpha=0, normalize=normalize, ro=ro, vo=vo
        )
        # set properties explicitly
        self.hasC = True
        self._backend_compatible = True
        self.hasC_dxdv = True
        self.hasC_dxdv3d = True  # full 3D Hessian (R2deriv/z2deriv/Rzderiv) in C
        self.hasC_dens = True
        return None

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        R, z = coerce_coords(xp, R, z)
        r = xp.sqrt(R**2.0 + z**2.0)
        return radial_limits(
            r,
            lambda r: -(1.0 - 1.0 / (1.0 + self.a / r) ** 2.0) / (6.0 * self.a),
            at0=-1.0 / (6.0 * self.a),
        )

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        R, z = coerce_coords(xp, R, z)
        return -R / (xp.sqrt(R**2.0 + z**2.0) + self.a) ** 3.0 / 3.0

    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        R, z = coerce_coords(xp, R, z)
        r = xp.sqrt(R**2.0 + z**2.0)
        return -(
            ((2.0 * R**2.0 - z**2.0) - self.a * r) / (3.0 * r * (r + self.a) ** 4.0)
        )

    def _zforce(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        R, z = coerce_coords(xp, R, z)
        r = xp.sqrt(R**2.0 + z**2.0)
        return -z / (self.a + r) ** 3.0 / 3.0

    def _z2deriv(self, R, z, phi=0.0, t=0.0):
        return self._R2deriv(z, R, phi=phi, t=t)

    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        R, z = coerce_coords(xp, R, z)
        a = self.a
        r = xp.sqrt(R**2.0 + z**2.0)
        return -(R * z / r / (a + r) ** 4.0)

    def _dens(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        R, z = coerce_coords(xp, R, z)
        r = xp.sqrt(R**2.0 + z**2.0)
        return radial_limits(
            r,
            lambda r: 1.0 / (1.0 + r / self.a) ** 4.0 / 4.0 / math.pi / self.a**3.0,
            atinf=0.0,
        )

    def _mass(self, R, z=None, t=0.0):
        if z is not None:
            raise AttributeError  # use general implementation
        return radial_limits(  # written so it works for r=numpy.inf
            R,
            lambda R: 1.0 / (1.0 + self.a / R) ** 3.0 / 3.0,
            at0=0.0,
        )


class HernquistPotential(DehnenSphericalPotential):
    """Class that implements the Hernquist potential

    .. math::

        \\rho(r) = \\frac{\\mathrm{amp}}{4\\,\\pi\\,a^3}\\,\\frac{1}{(r/a)\\,(1+r/a)^{3}}

    """

    def __init__(self, amp=1.0, a=1.0, normalize=False, ro=None, vo=None):
        """
        Initialize a Two Power Spherical Potential.

        Parameters
        ----------
        amp : float or Quantity, optional
            Amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass or Gxmass (note that amp is 2 x [total mass] for the chosen definition of the Two Power Spherical potential).
        a : float or Quantity, optional
            Scale radius.
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2010-07-09 - Written - Bovy (NYU).

        """
        DehnenSphericalPotential.__init__(
            self, amp=amp, a=a, alpha=1, normalize=normalize, ro=ro, vo=vo
        )
        self._nemo_accname = "Dehnen"
        # set properties explicitly
        self.hasC = True
        self._backend_compatible = True
        self.hasC_dxdv = True
        self.hasC_dxdv3d = True  # full 3D Hessian (R2deriv/z2deriv/Rzderiv) in C
        self.hasC_dens = True
        return None

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        R, z = coerce_coords(xp, R, z)
        return radial_limits(
            xp.sqrt(R**2.0 + z**2.0),
            lambda r: -1.0 / (1.0 + r / self.a) / 2.0 / self.a,
            atinf=0.0,
        )

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        R, z = coerce_coords(xp, R, z)
        sqrtRz = xp.sqrt(R**2.0 + z**2.0)
        return -R / self.a / sqrtRz / (1.0 + sqrtRz / self.a) ** 2.0 / 2.0 / self.a

    def _zforce(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        R, z = coerce_coords(xp, R, z)
        sqrtRz = xp.sqrt(R**2.0 + z**2.0)
        return -z / self.a / sqrtRz / (1.0 + sqrtRz / self.a) ** 2.0 / 2.0 / self.a

    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        R, z = coerce_coords(xp, R, z)
        sqrtRz = xp.sqrt(R**2.0 + z**2.0)
        return (
            (self.a * z**2.0 + (z**2.0 - 2.0 * R**2.0) * sqrtRz)
            / sqrtRz**3.0
            / (self.a + sqrtRz) ** 3.0
            / 2.0
        )

    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        R, z = coerce_coords(xp, R, z)
        sqrtRz = xp.sqrt(R**2.0 + z**2.0)
        return (
            -R
            * z
            * (self.a + 3.0 * sqrtRz)
            * (sqrtRz * (self.a + sqrtRz)) ** -3.0
            / 2.0
        )

    def _surfdens(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        R, z = coerce_coords(xp, R, z)
        r = xp.sqrt(R**2.0 + z**2.0)
        # R == a is a removable singularity of the generic branch (Rma -> 0,
        # (a^2-R^2) -> 0, (r^2-a^2) -> 0): use the closed-form limit there. Both
        # xp.where branches are evaluated, so guard the generic branch's zero
        # denominators with a safe R^2-a^2 that is never zero at the edge.
        at_edge = R == self.a
        d2 = R**2.0 - self.a**2.0
        safe_d2 = xp.where(at_edge, xp.ones_like(d2 * 1.0), d2)
        Rma = xp.sqrt(xp.astype(safe_d2, xp.complex128))
        # also guard (r^2 - a^2), which vanishes at the edge when z == 0
        d2r = r**2.0 - self.a**2.0
        safe_d2r = xp.where(at_edge, xp.ones_like(d2r * 1.0), d2r)
        # The edge branch carries a z**-5 that is singular at z == 0; under the
        # eager xp.where it is evaluated everywhere, so in its DEAD region
        # (generic, R != a) it must use a finite z or it NaN-poisons reverse-mode
        # gradients at z == 0 (a valid, finite input where surfdens == 0). Use the
        # real z on the edge (live) and 1 in the generic region (dead). At the
        # genuine edge point R == a, z == 0 the limb singularity is real (kept).
        z_edge = xp.where(at_edge, z, xp.ones_like(z * 1.0))
        edge = (
            (
                -12.0 * self.a**3
                - 5.0 * self.a * z_edge**2
                + xp.sqrt(1.0 + z_edge**2 / self.a**2)
                * (12.0 * self.a**3 - self.a * z_edge**2 + 2 / self.a * z_edge**4)
            )
            / 30.0
            / math.pi
            * z_edge**-5.0
        )
        generic = (
            self.a
            * xp.real(
                (2.0 * self.a**2.0 + R**2.0)
                * Rma**-5
                * (xp.arctan(z / Rma) - xp.arctan(self.a * z / r / Rma))
                + z
                * (
                    5.0 * self.a**3.0 * r
                    - 4.0 * self.a**4
                    + self.a**2 * (2.0 * r**2.0 + R**2)
                    - self.a * r * (5.0 * R**2.0 + 3.0 * z**2.0)
                    + R**2.0 * r**2.0
                )
                / safe_d2**2.0
                / safe_d2r**2.0
            )
            / 4.0
            / math.pi
        )
        return xp.where(at_edge, edge, generic)

    def _mass(self, R, z=None, t=0.0):
        if z is not None:
            raise AttributeError  # use general implementation
        return radial_limits(  # written so it works for r=numpy.inf
            R,
            lambda R: 1.0 / (1.0 + self.a / R) ** 2.0 / 2.0,
            at0=0.0,
        )

    @kms_to_kpcGyrDecorator
    def _nemo_accpars(self, vo, ro):
        """
        Return the accpars potential parameters for use of this potential with NEMO.

        Parameters
        ----------
        vo : float
            Velocity unit in km/s.
        ro : float
            Length unit in kpc.

        Returns
        -------
        str
            accpars string.

        Notes
        -----
        - 2018-09-14 - Written - Bovy (UofT)

        """
        GM = self._amp * vo**2.0 * ro / 2.0
        return f"0,1,{GM},{self.a * ro},0"


class JaffePotential(DehnenSphericalPotential):
    """Class that implements the Jaffe potential

    .. math::

        \\rho(r) = \\frac{\\mathrm{amp}}{4\\,\\pi\\,a^3}\\,\\frac{1}{(r/a)^2\\,(1+r/a)^{2}}

    """

    def __init__(self, amp=1.0, a=1.0, normalize=False, ro=None, vo=None):
        """
        Initialize a Jaffe Potential.

        Parameters
        ----------
        amp : float or Quantity, optional
            Amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass or Gxmass.
        a : float or Quantity, optional
            Scale radius (can be Quantity).
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2010-07-09 - Written - Bovy (NYU)
        """
        Potential.__init__(self, amp=amp, ro=ro, vo=vo, amp_units="mass")
        a = conversion.parse_length(a, ro=self._ro)
        self.a = a
        self._scale = self.a
        self.alpha = 2
        self.beta = 4
        self._backend_compatible = True
        if normalize or (
            isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
        ):  # pragma: no cover
            self.normalize(normalize)
        self.hasC = True
        self.hasC_dxdv = True
        self.hasC_dxdv3d = True  # full 3D Hessian (R2deriv/z2deriv/Rzderiv) in C
        self.hasC_dens = True
        return None

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        R, z = coerce_coords(xp, R, z)
        return -xp.log(1.0 + self.a / xp.sqrt(R**2.0 + z**2.0)) / self.a

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        R, z = coerce_coords(xp, R, z)
        sqrtRz = xp.sqrt(R**2.0 + z**2.0)
        return -R / sqrtRz**3.0 / (1.0 + self.a / sqrtRz)

    def _zforce(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        R, z = coerce_coords(xp, R, z)
        sqrtRz = xp.sqrt(R**2.0 + z**2.0)
        return -z / sqrtRz**3.0 / (1.0 + self.a / sqrtRz)

    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        R, z = coerce_coords(xp, R, z)
        sqrtRz = xp.sqrt(R**2.0 + z**2.0)
        return (
            (self.a * (z**2.0 - R**2.0) + (z**2.0 - 2.0 * R**2.0) * sqrtRz)
            / sqrtRz**4.0
            / (self.a + sqrtRz) ** 2.0
        )

    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        R, z = coerce_coords(xp, R, z)
        sqrtRz = xp.sqrt(R**2.0 + z**2.0)
        return (
            -R
            * z
            * (2.0 * self.a + 3.0 * sqrtRz)
            * sqrtRz**-4.0
            * (self.a + sqrtRz) ** -2.0
        )

    def _surfdens(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        R, z = coerce_coords(xp, R, z)
        r = xp.sqrt(R**2.0 + z**2.0)
        # R == a is a removable singularity of the generic branch (Rma -> 0,
        # R^2-a^2 -> 0): use the closed-form limit there. Both xp.where branches
        # are evaluated, so guard the generic branch's zero denominators.
        at_edge = R == self.a
        d2 = R**2.0 - self.a**2.0
        safe_d2 = xp.where(at_edge, xp.ones_like(d2 * 1.0), d2)
        Rma = xp.sqrt(xp.astype(safe_d2, xp.complex128))
        # The edge branch carries a z**-3 that is singular at z == 0; under the
        # eager xp.where it is evaluated everywhere, so in its DEAD region
        # (generic, R != a) it must use a finite z or it NaN-poisons reverse-mode
        # gradients at z == 0 (a valid, finite input where surfdens == 0). Use the
        # real z on the edge (live) and 1 in the generic region (dead). At the
        # genuine edge point R == a, z == 0 the limb singularity is real (kept).
        z_edge = xp.where(at_edge, z, xp.ones_like(z * 1.0))
        edge = (
            (
                3.0 * z_edge**2.0
                - 2.0 * self.a**2.0
                + 2.0
                * xp.sqrt(1.0 + (z_edge / self.a) ** 2.0)
                * (self.a**2.0 - 2.0 * z_edge**2.0)
                + 3.0 * z_edge**3.0 / self.a * xp.arctan(z_edge / self.a)
            )
            / self.a
            / z_edge**3.0
            / 6.0
            / math.pi
        )
        generic = (
            xp.real(
                (2.0 * self.a**2.0 - R**2.0)
                * Rma**-3
                * (xp.arctan(z / Rma) - xp.arctan(self.a * z / r / Rma))
                + xp.arctan(z / R) / R
                - self.a * z / safe_d2 / (r + self.a)
            )
            / self.a
            / 2.0
            / math.pi
        )
        return xp.where(at_edge, edge, generic)

    def _mass(self, R, z=None, t=0.0):
        if z is not None:
            raise AttributeError  # use general implementation
        return radial_limits(  # written so it works for r=numpy.inf
            R, lambda R: 1.0 / (1.0 + self.a / R), at0=0.0
        )


class NFWPotential(TwoPowerSphericalPotential):
    """Class that implements the NFW potential

    .. math::

        \\rho(r) = \\frac{\\mathrm{amp}}{4\\,\\pi\\,a^3}\\,\\frac{1}{(r/a)\\,(1+r/a)^{2}}

    """

    def __init__(
        self,
        amp=1.0,
        a=1.0,
        normalize=False,
        rmax=None,
        vmax=None,
        conc=None,
        mvir=None,
        vo=None,
        ro=None,
        H=70.0,
        Om=0.3,
        overdens=200.0,
        wrtcrit=False,
    ):
        """
        Initialize a NFW Potential.

        Parameters
        ----------
        amp : float or Quantity, optional
            Amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass or Gxmass.
        a : float or Quantity, optional
            Scale radius (can be Quantity).
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.
        rmax : float or Quantity, optional
            Radius where the rotation curve peak.
        vmax : float or Quantity, optional
            Maximum circular velocity.
        conc : float, optional
            Concentration.
        mvir : float, optional
            virial mass in 10^12 Msolar
        H : float, optional
            Hubble constant in km/s/Mpc.
        Om : float, optional
            Omega matter.
        overdens : float, optional
            Overdensity which defines the virial radius.
        wrtcrit : bool, optional
            If True, the overdensity is wrt the critical density rather than the mean matter density.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - Initialize with one of:
              * a and amp or normalize
              * rmax and vmax
              * conc, mvir, H, Om, overdens, wrtcrit
        - 2010-07-09 - Written - Bovy (NYU)
        - 2014-04-03 - Initialization w/ concentration and mass - Bovy (IAS)
        - 2020-04-29 - Initialization w/ rmax and vmax - Bovy (UofT)

        """
        Potential.__init__(self, amp=amp, ro=ro, vo=vo, amp_units="mass")
        a = conversion.parse_length(a, ro=self._ro)
        # Flag before the (conditional) normalize() so its Rforce(1.,0.) is coerced.
        self._backend_compatible = True
        if conc is None and rmax is None:
            self.a = a
            if normalize or (
                isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
            ):
                self.normalize(normalize)
        elif not rmax is None:
            if _APY_LOADED and isinstance(rmax, units.Quantity):
                rmax = conversion.parse_length(rmax, ro=self._ro)
                self._roSet = True
            if _APY_LOADED and isinstance(vmax, units.Quantity):
                vmax = conversion.parse_velocity(vmax, vo=self._vo)
                self._voSet = True
            self.a = rmax / 2.1625815870646098349
            self._amp = vmax**2.0 * self.a / 0.21621659550187311005
        else:
            if wrtcrit:
                od = overdens / conversion.dens_in_criticaldens(self._vo, self._ro, H=H)
            else:
                od = overdens / conversion.dens_in_meanmatterdens(
                    self._vo, self._ro, H=H, Om=Om
                )
            mvirNatural = mvir * 100.0 / conversion.mass_in_1010msol(self._vo, self._ro)
            rvir = (3.0 * mvirNatural / od / 4.0 / numpy.pi) ** (1.0 / 3.0)
            self.a = rvir / conc
            self._amp = mvirNatural / (numpy.log(1.0 + conc) - conc / (1.0 + conc))
            # Turn on physical output, because mass is given in 1e12 Msun (see #465)
            self._roSet = True
            self._voSet = True
        self.alpha = 1
        self.beta = 3
        self._scale = self.a
        self.hasC = True
        self.hasC_dxdv = True
        self.hasC_dxdv3d = True  # full 3D Hessian (R2deriv/z2deriv/Rzderiv) in C
        self.hasC_dens = True
        self._nemo_accname = "NFW"
        return None

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        R, z = coerce_coords(xp, R, z)
        r = xp.sqrt(R**2.0 + z**2.0)
        # -special.xlogy(1/r, 1+r/a) == -(1/r)*log(1+r/a), with xlogy's
        # convention that the result is 0 where the prefactor 1/r is 0 (i.e.
        # at r -> infty, where the bare product is 0*inf = NaN; this is the
        # "stable as r -> infty" behavior). Additionally Phi(0) = -1/a. Both
        # r == 0 and r == infty are handled with NaN-safe xp.where branches.
        at0 = r == 0.0
        atinf = xp.isinf(r)
        # safe r so neither dead branch divides by 0 or takes log(inf)
        safe = xp.where(at0 | atinf, xp.ones_like(r * 1.0), r)
        # log(1 + x) loses eps/x at x << 1
        bulk = branch_where(
            xp,
            safe < _NFW_SMALL_X * self.a,
            lambda: -(1.0 / safe) * xp.log1p(safe / self.a),
            lambda: -(1.0 / safe) * xp.log(1.0 + safe / self.a),
        )
        out = xp.where(atinf, xp.zeros_like(r * 1.0), bulk)
        return xp.where(at0, -1.0 / self.a * xp.ones_like(r * 1.0), out)

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        R, z = coerce_coords(xp, R, z)
        Rz = R**2.0 + z**2.0
        sqrtRz = xp.sqrt(Rz)
        small = sqrtRz < _NFW_SMALL_X * self.a
        return branch_where(
            xp,
            small,
            lambda: -R * _nfw_h(xp, small, sqrtRz, self.a),
            lambda: (
                R
                * (
                    1.0 / Rz / (self.a + sqrtRz)
                    - xp.log(1.0 + sqrtRz / self.a) / sqrtRz / Rz
                )
            ),
        )

    def _zforce(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        R, z = coerce_coords(xp, R, z)
        Rz = R**2.0 + z**2.0
        sqrtRz = xp.sqrt(Rz)
        small = sqrtRz < _NFW_SMALL_X * self.a
        return branch_where(
            xp,
            small,
            lambda: -z * _nfw_h(xp, small, sqrtRz, self.a),
            lambda: (
                z
                * (
                    1.0 / Rz / (self.a + sqrtRz)
                    - xp.log(1.0 + sqrtRz / self.a) / sqrtRz / Rz
                )
            ),
        )

    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        R, z = coerce_coords(xp, R, z)
        Rz = R**2.0 + z**2.0
        sqrtRz = xp.sqrt(Rz)
        small = sqrtRz < _NFW_SMALL_X * self.a

        def stable():  # d2Phi/dR2 = (k R^2 + h z^2) / r^5
            h5, k5 = _nfw_hk5(xp, small, sqrtRz, self.a)
            return k5 * R**2.0 + h5 * z**2.0

        return branch_where(
            xp,
            small,
            stable,
            lambda: (
                (
                    3.0 * R**4.0
                    + 2.0 * R**2.0 * (z**2.0 + self.a * sqrtRz)
                    - z**2.0 * (z**2.0 + self.a * sqrtRz)
                    - (2.0 * R**2.0 - z**2.0)
                    * (self.a**2.0 + R**2.0 + z**2.0 + 2.0 * self.a * sqrtRz)
                    * xp.log(1.0 + sqrtRz / self.a)
                )
                / Rz**2.5
                / (self.a + sqrtRz) ** 2.0
            ),
        )

    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        R, z = coerce_coords(xp, R, z)
        Rz = R**2.0 + z**2.0
        sqrtRz = xp.sqrt(Rz)
        small = sqrtRz < _NFW_SMALL_X * self.a

        def stable():  # d2Phi/dRdz = (k - h) R z / r^5
            h5, k5 = _nfw_hk5(xp, small, sqrtRz, self.a)
            return (k5 - h5) * R * z

        return branch_where(
            xp,
            small,
            stable,
            lambda: (
                -R
                * z
                * (
                    -4.0 * Rz
                    - 3.0 * self.a * sqrtRz
                    + 3.0
                    * (self.a**2.0 + Rz + 2.0 * self.a * sqrtRz)
                    * xp.log(1.0 + sqrtRz / self.a)
                )
                * Rz**-2.5
                * (self.a + sqrtRz) ** -2.0
            ),
        )

    def _surfdens(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        R, z = coerce_coords(xp, R, z)
        r = xp.sqrt(R**2.0 + z**2.0)
        # R == a is a removable singularity of the generic branch (Rma -> 0,
        # R^2-a^2 -> 0): use the closed-form limit there. Both xp.where branches
        # are evaluated, so guard the generic branch's zero denominators.
        at_edge = R == self.a
        d2 = R**2.0 - self.a**2.0
        safe_d2 = xp.where(at_edge, xp.ones_like(d2 * 1.0), d2)
        Rma = xp.sqrt(xp.astype(safe_d2, xp.complex128))
        # The edge branch carries a z**-3 that is singular at z == 0; under the
        # eager xp.where it is evaluated everywhere, so in its DEAD region
        # (generic, R != a) it must use a finite z or it NaN-poisons reverse-mode
        # gradients at z == 0 (a valid, finite input where surfdens == 0). Use the
        # real z on the edge (live) and 1 in the generic region (dead). At the
        # genuine edge point R == a, z == 0 the limb singularity is real (kept).
        z_edge = xp.where(at_edge, z, xp.ones_like(z * 1.0))
        za2 = (z_edge / self.a) ** 2
        edge = (
            self.a
            * (2.0 + xp.sqrt(za2 + 1.0) * (za2 - 2.0))
            / 6.0
            / math.pi
            / z_edge**3
        )
        generic = (
            xp.real(
                self.a
                * Rma**-3
                * (xp.arctan(self.a * z / r / Rma) - xp.arctan(z / Rma))
                + z / (r + self.a) / safe_d2
            )
            / 2.0
            / math.pi
        )
        return xp.where(at_edge, edge, generic)

    def _mass(self, R, z=None, t=0.0):
        if z is not None:
            raise AttributeError  # use general implementation
        xp = get_namespace(R)
        (R,) = coerce_coords(xp, R)

        def mass(R):
            small = R < _NFW_SMALL_X * self.a
            return branch_where(
                xp,
                small,
                lambda: _nfw_hx(xp, _nfw_masked(xp, small, R, self.a)[1]),
                lambda: xp.log(1 + R / self.a) - R / self.a / (1.0 + R / self.a),
            )

        # log-divergent: numpy's inf - inf/inf was NaN
        return radial_limits(R, mass, atinf=numpy.inf, numpy_too=True)

    @conversion.physical_conversion("position", pop=False)
    def rvir(
        self,
        H=70.0,
        Om=0.3,
        t=0.0,
        overdens=200.0,
        wrtcrit=False,
        ro=None,
        vo=None,
        use_physical=False,
    ):  # use_physical necessary bc of pop=False, does nothing inside
        """
        Calculate the virial radius for this density distribution.

        Parameters
        ----------
        H : float, optional
            Hubble constant in km/s/Mpc. Default is 70.0.
        Om : float, optional
            Omega matter. Default is 0.3.
        t : float, optional
            Time. Default is 0.0.
        overdens : float, optional
            Overdensity which defines the virial radius. Default is 200.0.
        wrtcrit : bool, optional
            If True, the overdensity is wrt the critical density rather than the mean matter density. Default is False.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default is the object-wide value).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default is the object-wide value).

        Returns
        -------
        float
            Virial radius.

        Notes
        -----
        - 2014-01-29 - Written - Bovy (IAS)

        """
        if ro is None:
            ro = self._ro
        if vo is None:
            vo = self._vo
        if wrtcrit:
            od = overdens / conversion.dens_in_criticaldens(vo, ro, H=H)
        else:
            od = overdens / conversion.dens_in_meanmatterdens(vo, ro, H=H, Om=Om)
        dc = 12.0 * self.dens(self.a, 0.0, t=t, use_physical=False) / od
        x = optimize.brentq(
            lambda y: (numpy.log(1.0 + y) - y / (1.0 + y)) / y**3.0 - 1.0 / dc,
            0.01,
            100.0,
        )
        return x * self.a

    @conversion.physical_conversion("position", pop=True)
    def rmax(self):
        """
        Calculate the radius at which the rotation curve peaks.

        Returns
        -------
        float
            Radius at which the rotation curve peaks.

        Notes
        -----
        - 2020-02-05 - Written - Bovy (UofT)

        """
        # Magical number, solve(derivative (ln(1+x)-x/(1+x))/x wrt x=0,x)
        return 2.1625815870646098349 * self.a

    @conversion.physical_conversion("velocity", pop=True)
    def vmax(self):
        """
        Calculate the maximum rotation curve velocity.

        Returns
        -------
        float
            Peak velocity in the rotation curve.

        Notes
        -----
        - 2020-02-05 - Written - Bovy (UofT)

        """
        # 0.21621659550187311005 = (numpy.log(1.+rmax/a)-rmax/(a+rmax))*a/rmax
        return numpy.sqrt(0.21621659550187311005 * self._amp / self.a)

    @kms_to_kpcGyrDecorator
    def _nemo_accpars(self, vo, ro):
        """
        Return the accpars potential parameters for use of this potential with NEMO

        Parameters
        ----------
        vo : float
            Velocity unit in km/s
        ro : float
            Length unit in kpc

        Returns
        -------
        str
            accpars string

        Notes
        -----
        - 2014-12-18 - Written - Bovy (IAS)

        """
        ampl = self._amp * vo**2.0 * ro
        vmax = numpy.sqrt(
            ampl / self.a / ro * 0.2162165954
        )  # Take that factor directly from gyrfalcon
        return f"0,{self.a * ro},{vmax}"
