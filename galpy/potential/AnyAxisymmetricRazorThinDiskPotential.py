###############################################################################
#   AnyAxisymmetricRazorThinDiskPotential.py: class that implements the
#                                             potential of an arbitrary
#                                             axisymmetric, razor-thin disk
###############################################################################
import numpy
from scipy import integrate, special

from ..util import conversion
from ..util._optional_deps import _APY_LOADED
from .Potential import Potential, check_potential_inputs_not_arrays

# Below this |z|, R2deriv is indistinguishable from its z=0 limit: measured
# against an mpmath Hankel reference the true value moves by only 5.9e-09 between
# z=0 and z=1e-10 (and ~1e-11 by z=1e-12), while the finite-|z| quadrature branch
# degrades there. Snapping costs ~6e-9 and avoids a decade in which neither
# branch is accurate.
_R2DERIV_ZFLOOR = 1e-9

if _APY_LOADED:
    from astropy import units


# The zforce integrand carries a factor 1/((a-R)^2+z^2): a peak of width ~|z|
# centred on a=R. quad subdivides adaptively from panels of order R, so once
# |z| << R it steps over the peak entirely and returns a value that is not
# merely inaccurate but wrong by orders of magnitude and of the wrong sign
# (zforce(0.5, 1e-8) came back +1.9e-8 instead of -1.86).
#
# Substituting a = R + |z| t maps the peak to unit width: (a-R)^2+z^2 becomes
# z^2 (1+t^2) *analytically*, so the small quantity is never formed by
# subtraction and the result stays correct however small |z| is; da = |z| dt
# then cancels the 1/|z| that the peak contributes, against the -4z prefactor.
#
# Only applied where that cancellation makes the substituted integrand finite,
# i.e. zforce and Rzderiv, both of which carry a z prefactor. Above the
# threshold, and at z == 0, the original quadrature is used verbatim.
#
# Rzderiv is a 1/z-sized residual of 1/z**2-sized pieces that cancel by their
# oddness about a = R, so its relative accuracy floors at ~eps/|z|: good to
# ~1e-8 at |z| = 1e-8, degrading below that. Cancelling analytically instead
# would need Sigma'(R), which is not available for a user-supplied callable.
_PEAK_SUB_ZR = 1e-4  # substitute when |z| < _PEAK_SUB_ZR * R
_PEAK_SUB_T = 1.0e4  # near-region half-width, in units of |z|


def _quad_apeak(f, R, z):
    """Integrate ``f(a, u, d2)`` over ``a`` in [0, inf), resolving the a=R peak.

    ``f`` receives the exact ``u = a - R`` and ``d2 = u**2 + z**2`` alongside
    ``a`` so the substituted branch can supply ``|z| t`` and ``z**2 (1+t**2)``
    rather than re-forming either by subtraction: for ``|z| << R`` both cancel
    catastrophically when built from ``a`` and ``R``.
    """
    az = numpy.fabs(z)
    g = lambda a: f(a, a - R, (a - R) ** 2.0 + z**2.0)
    # z == 0 is the genuinely singular case (callers special-case it) and the
    # substitution would divide by |z|, so it stays on the original path.
    if az == 0.0 or az >= _PEAK_SUB_ZR * R:
        return (
            integrate.quad(g, 0, 2 * R, points=[R])[0]
            + integrate.quad(g, 2 * R, numpy.inf)[0]
        )
    z2 = z**2.0
    tlim = min(_PEAK_SUB_T, R / az)
    lo, hi = R - az * tlim, R + az * tlim
    return (
        integrate.quad(
            lambda t: f(R + az * t, az * t, z2 * (1.0 + t * t)) * az,
            -tlim,
            tlim,
            limit=200,
        )[0]
        + integrate.quad(g, 0, lo, limit=200)[0]
        + integrate.quad(g, hi, 2 * R, limit=200)[0]
        + integrate.quad(g, 2 * R, numpy.inf, limit=200)[0]
    )


class AnyAxisymmetricRazorThinDiskPotential(Potential):
    """Class that implements the potential of an arbitrary axisymmetric, razor-thin disk with surface density :math:`\\Sigma(R)`"""

    def __init__(
        self,
        amp=1.0,
        surfdens=lambda R: 1.5 * numpy.exp(-3.0 * R),
        normalize=False,
        ro=None,
        vo=None,
    ):
        """
        Potential of an arbitrary axisymmetric disk.

        Parameters
        ----------
        amp : float, optional
            Amplitude to be applied to the potential. Default is 1.0.
        surfdens : callable, optional
            Function of a single variable that gives the surface density as a function of radius (can return a Quantity). Default is ``lambda R: 1.5 * numpy.exp(-3.0 * R)``.
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1. Default is False.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2021-01-04 - Written - Bovy (UofT)

        """
        Potential.__init__(self, amp=amp, ro=ro, vo=vo)
        # Parse surface density: does it have units? does it expect them?
        if _APY_LOADED:
            _sdens_unit_input = False
            try:
                surfdens(1)
            except (units.UnitConversionError, units.UnitTypeError):
                _sdens_unit_input = True
            _sdens_unit_output = False
            if _sdens_unit_input:
                try:
                    surfdens(1.0 * units.kpc).to(units.Msun / units.pc**2)
                except (AttributeError, units.UnitConversionError):
                    pass
                else:
                    _sdens_unit_output = True
            else:
                try:
                    surfdens(1.0).to(units.Msun / units.pc**2)
                except (AttributeError, units.UnitConversionError):
                    pass
                else:
                    _sdens_unit_output = True
            if _sdens_unit_input and _sdens_unit_output:
                self._sdens = lambda R: conversion.parse_surfdens(
                    surfdens(R * self._ro * units.kpc), ro=self._ro, vo=self._vo
                )
            elif _sdens_unit_input:
                self._sdens = lambda R: surfdens(R * self._ro * units.kpc)
            elif _sdens_unit_output:
                self._sdens = lambda R: conversion.parse_surfdens(
                    surfdens(R), ro=self._ro, vo=self._vo
                )
            if _sdens_unit_output:
                # When sdens (like other potential's amplitude) gives outputs in units,
                # turn on physical output
                self._roSet = True
                self._voSet = True
        if not hasattr(self, "_sdens"):  # unitless
            self._sdens = surfdens
        # The potential at zero, in case it's asked for
        self._pot_zero = (
            -2.0 * numpy.pi * integrate.quad(lambda a: self._sdens(a), 0, numpy.inf)[0]
        )
        if normalize or (
            isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
        ):  # pragma: no cover
            self.normalize(normalize)

    @check_potential_inputs_not_arrays
    def _evaluate(self, R, z, phi=0.0, t=0.0):
        if R == 0 and z == 0:
            return self._pot_zero
        elif numpy.isinf(R**2 + z**2):
            return 0.0
        potint = lambda a: (
            a
            * self._sdens(a)
            / numpy.sqrt((R + a) ** 2.0 + z**2.0)
            * special.ellipk(4 * R * a / ((R + a) ** 2.0 + z**2.0))
        )
        return -4 * (
            integrate.quad(potint, 0, 2 * R, points=[R])[0]
            + integrate.quad(potint, 2 * R, numpy.inf)[0]
        )

    @check_potential_inputs_not_arrays
    def _Rforce(self, R, z, phi=0.0, t=0.0):
        R2 = R**2
        z2 = z**2

        def rforceint(a):
            a2 = a**2
            aRz = (a + R) ** 2.0 + z2
            # m = 4aR/((a+R)^2+z^2), and 1-m = ((a-R)^2+z^2)/((a+R)^2+z^2) exactly.
            # Taking K from that complement never forms 1-m by subtraction, which
            # rounds to 0 for tiny |z| and sends ellipk to inf. E is bounded, so
            # clamping m only guards the >1 rounding artifact there.
            m = numpy.minimum(4 * a * R / aRz, 1.0)
            km1 = ((a - R) ** 2.0 + z2) / aRz
            return (
                a
                * self._sdens(a)
                * (
                    (a2 - R2 + z2) * special.ellipe(m)
                    - ((a - R) ** 2 + z2) * special.ellipkm1(km1)
                )
                / R
                / ((a - R) ** 2 + z2)
                / numpy.sqrt(aRz)
            )

        return 2 * (
            integrate.quad(rforceint, 0, 2 * R, points=[R])[0]
            + integrate.quad(rforceint, 2 * R, numpy.inf)[0]
        )

    @check_potential_inputs_not_arrays
    def _zforce(self, R, z, phi=0.0, t=0.0):
        if z == 0:
            return 0.0
        z2 = z**2

        def zforceint(a, u, d2):
            aRz = (a + R) ** 2.0 + z2
            # m = 4aR/((a+R)^2+z^2), and 1-m = ((a-R)^2+z^2)/((a+R)^2+z^2) exactly.
            # Taking K from that complement never forms 1-m by subtraction, which
            # rounds to 0 for tiny |z| and sends ellipk to inf. E is bounded, so
            # clamping m only guards the >1 rounding artifact there.
            m = numpy.minimum(4 * a * R / aRz, 1.0)
            km1 = ((a - R) ** 2.0 + z2) / aRz
            return a * self._sdens(a) * special.ellipe(m) / d2 / numpy.sqrt(aRz)

        return -4 * z * _quad_apeak(zforceint, R, z)

    @check_potential_inputs_not_arrays
    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        R2 = R**2
        az = numpy.fabs(z)
        if az < _R2DERIV_ZFLOOR:
            az = 0.0
        z2 = az**2

        def r2derivint(d):
            """The integrand as a function of the OFFSET d = a - R.

            Parameterised by d, never by a, so the offset is never formed by
            subtraction: at R=0.2 the float64 spacing is 2.8e-17, so recovering
            d=1e-10 from ``a - R`` carries 8e-8 relative error, which
            ``1/(d^2+z^2)^2`` then amplifies. Every ingredient is exact in d:
            ``a^2-R^2 = d(2R+d)`` and ``(a-R)^2+z^2 = d^2+z^2``.
            """
            a = R + d
            a2 = R2 + 2.0 * R * d + d * d
            dz = d * d + z2  # (a-R)^2 + z^2, exactly
            a2mR2 = d * (2.0 * R + d)  # a^2 - R^2, exactly
            aRz = (2.0 * R + d) ** 2.0 + z2
            m = numpy.minimum(4.0 * a * R / aRz, 1.0)
            return (
                a
                * self._sdens(a)
                * (
                    -(
                        (
                            (a2 - 3.0 * R2) * a2mR2**2
                            + (3.0 * a2**2 + 2.0 * a2 * R2 + 3.0 * R2**2) * z2
                            + (3.0 * a2 + 7.0 * R2) * z2**2
                            + z2**3
                        )
                        * special.ellipe(m)
                    )
                    + dz
                    * (a2mR2**2 + 2.0 * (a2 + 2.0 * R2) * z2 + z2**2)
                    * special.ellipkm1(dz / aRz)
                )
                / (2.0 * R2 * dz**2 * aRz**1.5)
            )

        # Symmetrise about d=0. The integrand behaves as C/d^2 + B/d + integrable
        # with C = Sigma(R)/2; pairing +-d cancels the odd B/d term exactly, so
        # only C has to be handled -- and C needs no derivative of Sigma.
        def sym(u):
            return r2derivint(u) + r2derivint(-u)

        C = self._sdens(R) / 2.0
        if z2 == 0.0:
            # At z=0 the integral exists only as a Hadamard finite part: the
            # integrand diverges as C/d^2 with the SAME sign either side, so the
            # two halves add rather than cancel. Subtract the singular model and
            # add back its finite part, -2C/R over the symmetric panel.
            inner = (
                integrate.quad(lambda u: sym(u) - 2.0 * C / (u * u), 0.0, R)[0]
                - 2.0 * C / R
            )
        else:
            # d = z*sinh(t) gives log-spaced coverage from z out to R with no
            # endpoint crowding, so the width-z peak is resolved at any z.
            # (d = z*tan(t) fails: arctan(R/z) lands within ~5e-10 of pi/2, where
            # tan is quantised -- the same representability trap one level up.)
            tmax = numpy.arcsinh(R / az)
            inner = integrate.quad(
                lambda t: sym(az * numpy.sinh(t)) * az * numpy.cosh(t), 0.0, tmax
            )[0]
        return -4 * (inner + integrate.quad(r2derivint, R, numpy.inf)[0])

    @check_potential_inputs_not_arrays
    def _z2deriv(self, R, z, phi=0.0, t=0.0):
        R2 = R**2
        z2 = z**2

        def z2derivint(a):
            a2 = a**2
            aRz = (a + R) ** 2.0 + z2
            # m = 4aR/((a+R)^2+z^2), and 1-m = ((a-R)^2+z^2)/((a+R)^2+z^2) exactly.
            # Taking K from that complement never forms 1-m by subtraction, which
            # rounds to 0 for tiny |z| and sends ellipk to inf. E is bounded, so
            # clamping m only guards the >1 rounding artifact there.
            m = numpy.minimum(4 * a * R / aRz, 1.0)
            km1 = ((a - R) ** 2.0 + z2) / aRz
            return (
                a
                * self._sdens(a)
                * (
                    -(
                        ((a2 - R2) ** 2 - 2.0 * (a2 + R2) * z2 - 3.0 * z**4)
                        * special.ellipe(m)
                    )
                    - z2 * ((a - R) ** 2 + z2) * special.ellipkm1(km1)
                )
                / (((a - R) ** 2 + z2) ** 2 * ((a + R) ** 2 + z2) ** 1.5)
            )

        return -4 * (
            integrate.quad(z2derivint, 0, 2 * R, points=[R])[0]
            + integrate.quad(z2derivint, 2 * R, numpy.inf)[0]
        )

    @check_potential_inputs_not_arrays
    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        R2 = R**2
        z2 = z**2

        def rzderivint(a, u, d2):
            aRz = (a + R) ** 2.0 + z2
            # m = 4aR/((a+R)^2+z^2), and 1-m = d2/aRz exactly. Taking K from that
            # complement never forms 1-m by subtraction, which rounds to 0 for
            # tiny |z| and sends ellipk to inf. E is bounded, so clamping m only
            # guards the >1 rounding artifact there.
            m = numpy.minimum(4 * a * R / aRz, 1.0)
            # Both coefficients are written in u = a-R, the point they are built
            # about: the E one vanishes at (a=R, z=0) and the K one factors
            # through d2. Forming either from a and R cancels away for |z| << R.
            ecoeff = (
                u * (16.0 * R2 * R + 4.0 * R * z2 + 4.0 * R * u * u)
                + u * u * (12.0 * R2 + 2.0 * z2)
                + u**4
                - 4.0 * R2 * z2
                + z2 * z2
            )
            kcoeff = d2 * (2.0 * R * u + d2)
            return (
                a
                * self._sdens(a)
                * (-ecoeff * special.ellipe(m) + kcoeff * special.ellipkm1(d2 / aRz))
                / R
                / d2**2
                / aRz**1.5
            )

        return -2 * z * _quad_apeak(rzderivint, R, z)

    def _surfdens(self, R, z, phi=0.0, t=0.0):
        return self._sdens(R)
