###############################################################################
#   AnyAxisymmetricRazorThinDiskPotential.py: class that implements the
#                                             potential of an arbitrary
#                                             axisymmetric, razor-thin disk
###############################################################################
import numpy
from scipy import integrate, special

from ..backend import (
    autodiff_ops,
    device_of,
    get_namespace,
    is_backend_array,
    match_input_dtype,
)
from ..backend import quadrature as _bquad
from ..backend import special as _bspecial
from ..backend import use
from ..util import conversion
from ..util._optional_deps import _APY_LOADED
from .Potential import (
    Potential,
    _normalize_requested,
    check_potential_inputs_not_arrays,
)

# Below this |z| both second derivatives are indistinguishable from their z=0
# limits, so we evaluate the z=0 branch: the finite-|z| branch degrades here (the
# offset d is not representable relative to R), and this avoids a decade in which
# neither branch is accurate.
#
# Worst case is at the floor itself. Relative movement of the TRUE value between
# z=0 and z=1e-9, vs an mpmath Hankel reference:
#
#     R      R2deriv    z2deriv
#     0.2    5.9e-08    1.9e-09
#     1.0    2.4e-09    4.5e-09
#
# i.e. up to ~6e-8, set by R2deriv at small R. The movement is exactly linear in
# z -- measured, not extrapolated: R=0.2 R2deriv gives 5.937e-08, 5.937e-09 and
# 5.937e-11 at z = 1e-9, 1e-10 and 1e-12. Both derivatives use this constant, and
# z2deriv moves LESS across the floor than R2deriv does.
_R2DERIV_ZFLOOR = 1e-9
# Relative floor on |z|/R for the backend zforce and Rzderiv, only so that z^2
# cannot underflow; the limit as z -> 0 is reached to O(|z|/R).
_ZFORCE_ZFLOOR_REL = 1e-14

if _APY_LOADED:
    from astropy import units


def _on_numpy(f, *args):
    """f(*args) with numpy semantics: the constructor's unit probes and Phi(0)
    pass surfdens plain floats, which a forced backend would otherwise hand an
    xp-agnostic surfdens as that backend's namespace"""
    with use("numpy", force=True):
        return f(*args)


# Gauss-Legendre nodes on the backend path (the numpy path keeps scipy's
# adaptive quad and is byte-identical): the tail a > 2R, and the fold about a = R,
# whose log subtraction at z = 0 converges algebraically (Rforce there: 5e-11 at
# 100 nodes, 6e-12 at 300; beyond, rounding in the sums wins off the plane).
_GLORDER = 100
_FOLD_N = 300


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


def _node_axes(v, n_axes):
    """``v`` with ``n_axes`` trailing axes added, to broadcast against quadrature nodes.

    The backend rules call their integrand with ONE trailing node axis. A no-op
    on a scalar/0-d ``v``, which is what lets the scalar path stay exactly as it
    was.
    """
    for _ in range(n_axes):
        v = _bquad.node_axis(v)
    return v


def _finite_part_quad(integrand_d, R, az, C):
    """Integrate ``integrand_d(d)`` over the disc, as a Hadamard finite part.

    ``integrand_d`` is parameterised by the OFFSET ``d = a - R`` so the offset is
    never formed by subtraction: at R=0.2 the float64 spacing is 2.8e-17, so
    recovering d=1e-10 from ``a - R`` carries 8e-8 relative error, which the
    ``1/(d^2+z^2)^2`` factor then amplifies.

    Near d=0 the integrand behaves as ``C/d^2 + B/d + integrable``. Symmetrising
    about d=0 cancels the odd ``B/d`` term exactly, so only ``C`` is needed --
    and ``C`` never involves a derivative of the surface density.

    At z=0 the integral exists only as a finite part: the integrand diverges as
    ``C/d^2`` with the SAME sign either side, so the halves add rather than
    cancel. Subtract the singular model and add back its finite part, ``-2C/R``.
    For z>0 there is a peak of width ~z instead; ``d = z*sinh(t)`` gives
    log-spaced coverage from z out to R with no endpoint crowding.
    (``d = z*tan(t)`` fails -- ``arctan(R/z)`` lands within ~5e-10 of pi/2, where
    tan is quantised: the same representability trap one level up.)
    """

    def sym(u):
        return integrand_d(u) + integrand_d(-u)

    if az == 0.0:
        inner = (
            integrate.quad(lambda u: sym(u) - 2.0 * C / (u * u), 0.0, R)[0]
            - 2.0 * C / R
        )
    else:
        tmax = numpy.arcsinh(R / az)
        inner = integrate.quad(
            lambda t: sym(az * numpy.sinh(t)) * az * numpy.cosh(t), 0.0, tmax
        )[0]
    return -4 * (inner + integrate.quad(integrand_d, R, numpy.inf)[0])


def _default_surfdens(R):
    # Backend-agnostic default so the GL/backend force path is jit/trace-safe: the
    # surface density is evaluated on the (traced) GL nodes, and bare ``numpy.exp`` calls
    # ``__array__()`` on a jax tracer. Dispatch on is_backend_array (NOT get_namespace):
    # a numpy / python-float / Quantity R stays on ``numpy.exp`` -> byte-identical and the
    # constructor's unit-detection probe is unchanged; only a genuine backend array takes
    # the namespace's exp.
    xp = get_namespace(R) if is_backend_array(R) else numpy
    return 1.5 * xp.exp(-3.0 * R)


class AnyAxisymmetricRazorThinDiskPotential(Potential):
    """Class that implements the potential of an arbitrary axisymmetric, razor-thin disk with surface density :math:`\\Sigma(R)`"""

    # Backend ARRAY input is accepted: the six _gl integrands broadcast over a
    # trailing node axis, which is what lets the traced surfdens/mass Poisson
    # quadratures hand this potential a whole Gauss-Legendre node array in ONE
    # call.
    _backend_accepts_arrays = True

    def __init__(
        self,
        amp=1.0,
        surfdens=_default_surfdens,
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
        # jax/torch-traceable forces (differentiable) via backend Gauss-Legendre
        # quadrature; the numpy path stays scipy-adaptive and byte-identical.
        self._backend_compatible = True
        # Parse surface density: does it have units? does it expect them?
        if _APY_LOADED:
            _sdens_unit_input = False
            try:
                _on_numpy(surfdens, 1)
            except (units.UnitConversionError, units.UnitTypeError):
                _sdens_unit_input = True
            _sdens_unit_output = False
            if _sdens_unit_input:
                try:
                    _on_numpy(surfdens, 1.0 * units.kpc).to(units.Msun / units.pc**2)
                except (AttributeError, TypeError, units.UnitConversionError):
                    pass
                else:
                    _sdens_unit_output = True
            else:
                try:
                    _on_numpy(surfdens, 1.0).to(units.Msun / units.pc**2)
                except (AttributeError, TypeError, units.UnitConversionError):
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
            -2.0
            * numpy.pi
            * integrate.quad(lambda a: _on_numpy(self._sdens, a), 0, numpy.inf)[0]
        )
        if _normalize_requested(normalize):  # pragma: no cover
            self.normalize(normalize)

    # -----------------------------------------------------------------------
    # Backend (jax/torch) helpers
    #
    # A backend input (concrete, traced, or grad-tracking) takes in-backend
    # Gauss-Legendre quadrature; a numpy input keeps scipy's adaptive quad and is
    # byte-identical. Every backend integrand is written in the EXACT offset
    # u = a - R, as numpy's _quad_apeak and _finite_part_quad are: forming a - R
    # by subtraction loses the digits that the 1/((a-R)^2+z^2) peak then
    # amplifies (Rzderiv, a 1/z residual of 1/z^2 pieces, came out 1e-2 wrong at
    # |z| = 1e-8). Measured against an mpmath Hankel reference at R = 0.05 .. 5,
    # z = 0 .. 3, the backend path is within 2e-11 for the potential and the
    # forces, 7e-11 for R2deriv, 1e-12 for z2deriv and 5e-7 for Rzderiv (scipy:
    # 5e-8, 6e-5, 2e-8, 1e-8, and 1e1 at |z| = 1e-10).
    # -----------------------------------------------------------------------
    def _bk_offset_quad(self, make_integrand, R, z, xp, dev):
        """int_0^inf f(a) da for an integrand written in the offset u = a - R.

        [0, 2R] is folded about a = R (finite_part_quad, which cancels the odd
        principal-value part exactly, subtracts the log singularity of the
        plane, and maps the |z|-wide peak off it with a sinh), plus the tail
        a > 2R. ``make_integrand(Rv)`` returns ``f(u)`` at radius ``Rv``.

        Both degenerate radii need a guard, or the backend returns NaN where
        numpy is finite: at R = 0 the fold has zero width and the tail is the
        whole integral; at R = inf everything vanishes. Selected with xp.where
        (no data-dependent branch, so it traces), with each dead branch at a
        harmless R so a nan cannot leak back through a gradient.
        """
        finite = xp.isfinite(R)
        Rs = xp.where(finite, R, xp.ones_like(R))
        positive = Rs > 0
        Rp = xp.where(positive, Rs, xp.ones_like(Rs))
        inner = _bquad.finite_part_quad(
            xp,
            make_integrand(Rp),
            Rp,
            c=xp.zeros_like(Rp),
            peak_width=xp.abs(z),
            n=_FOLD_N,
            device=dev,
        )
        out = xp.where(positive, inner, xp.zeros_like(inner)) + (
            _bquad.fixed_quad_semiinfinite(
                xp, make_integrand(Rs), Rs, n=_GLORDER, device=dev
            )
        )
        return xp.where(finite, out, xp.zeros_like(out))

    def _bk_offset_parts(self, Rv, z, u):
        """(a, z^2, (a-R)^2+z^2, (a+R)^2+z^2, 1-m) at the offset u, all exact
        in u: 1-m = ((a-R)^2+z^2)/((a+R)^2+z^2) is never formed as 1 - m, which
        rounds to 0 for |z| << R and sends K to inf."""
        Rb = _node_axes(Rv, 1)
        z2 = _node_axes(z, 1) ** 2
        d2 = u * u + z2
        aRz = (2.0 * Rb + u) ** 2 + z2
        return Rb + u, z2, d2, aRz, d2 / aRz

    @staticmethod
    def _bk_broadcast(R, z, xp):
        """R and z on their common shape (each point folds about its own R and
        resolves its own |z|); a no-op for scalars"""
        return R * xp.ones_like(z), z * xp.ones_like(R)

    def _bk_dispatch(self, gl_fn, R, z):
        xp = get_namespace(R, z)
        dev = device_of(R, z)
        R, z = self._bk_broadcast(R, z, xp)
        return match_input_dtype(gl_fn(R, z, xp, dev), R, z)

    # ------------------------------ potential ------------------------------
    @check_potential_inputs_not_arrays
    def _evaluate(self, R, z, phi=0.0, t=0.0):
        if is_backend_array(R) or is_backend_array(z):
            return self._bk_dispatch(self._evaluate_gl, R, z)
        return self._evaluate_numpy(R, z)

    def _evaluate_numpy(self, R, z):
        if R == 0 and z == 0:
            return self._pot_zero
        elif numpy.isinf(R**2 + z**2):
            return 0.0

        # Same a=R peak, and the same fix, as zforce and Rzderiv: see
        # _quad_apeak. Phi's peak is only logarithmic rather than 1/((a-R)^2+z^2),
        # so it is not wrong by orders of magnitude -- but for |z| << R quad
        # still steps over the O(|z|)-wide region entirely and misses the
        # 2*pi*Sigma(R)*|z| that Phi picks up off the midplane. That term IS the
        # vertical force, so without this Phi and zforce disagree: at R=0.5,
        # Phi(1e-8)-Phi(0) came back 4.4e-12 where zforce(0.5,1e-8) = -2.10295
        # demands 2.10e-8.
        #
        # m1 = 1-m is taken from the exact d2 = (a-R)^2+z^2 that _quad_apeak
        # supplies, never as 1 - 4aR/aRz: that subtraction rounds to 0 for
        # |z| << R and sends ellipk to inf right at the peak, which is why the
        # peak cannot simply be panelled with the old integrand.
        def potint(a, u, d2):
            aRz = (R + a) ** 2.0 + z**2.0
            return a * self._sdens(a) / numpy.sqrt(aRz) * special.ellipkm1(d2 / aRz)

        return -4 * _quad_apeak(potint, R, z)

    def _evaluate_gl(self, R, z, xp, dev):
        sdens = self._sdens

        def make_integrand(Rv):
            def potint(u):
                a, z2, d2, aRz, m1 = self._bk_offset_parts(Rv, z, u)
                return a * sdens(a) / xp.sqrt(aRz) * _bspecial.ellipkm1(m1)

            return potint

        return -4.0 * self._bk_offset_quad(make_integrand, R, z, xp, dev)

    # ------------------------------- Rforce --------------------------------
    @check_potential_inputs_not_arrays
    def _Rforce(self, R, z, phi=0.0, t=0.0):
        if is_backend_array(R) or is_backend_array(z):
            return self._bk_dispatch(self._Rforce_gl, R, z)
        return self._Rforce_numpy(R, z)

    def _Rforce_numpy(self, R, z):
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

    def _Rforce_gl(self, R, z, xp, dev):
        sdens = self._sdens

        def make_integrand(Rv):
            Rb = _node_axes(Rv, 1)

            def rforceint(u):
                # (a^2 - R^2 + z^2) E - ((a-R)^2 + z^2) K, a^2 - R^2 = u (2R + u)
                a, z2, d2, aRz, m1 = self._bk_offset_parts(Rv, z, u)
                return (
                    a
                    * sdens(a)
                    * (
                        (u * (2.0 * Rb + u) + z2) * _bspecial.ellipe(1.0 - m1)
                        - d2 * _bspecial.ellipkm1(m1)
                    )
                    / Rb
                    / d2
                    / xp.sqrt(aRz)
                )

            return rforceint

        return 2.0 * self._bk_offset_quad(make_integrand, R, z, xp, dev)

    # ------------------------------- zforce --------------------------------
    @check_potential_inputs_not_arrays
    def _zforce(self, R, z, phi=0.0, t=0.0):
        if is_backend_array(R) or is_backend_array(z):
            return self._bk_dispatch(self._zforce_gl, R, z)
        return self._zforce_numpy(R, z)

    def _zforce_numpy(self, R, z):
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

    def _zforce_gl(self, R, z, xp, dev):
        # z == 0 is the plane (zforce = 0 by symmetry); off it, -4 |z| I(|z|)
        # tends to -2 pi Sigma(R) as |z| -> 0. |z| is floored relative to R only
        # so that z^2 cannot underflow: below the floor the limit is reached to
        # O(|z|/R).
        az = xp.maximum(xp.abs(z), _ZFORCE_ZFLOOR_REL * xp.abs(R))
        az = xp.where(az > 0, az, xp.ones_like(az))  # R == z == 0: dead branch
        sdens = self._sdens

        def make_integrand(Rv):
            def zforceint(u):
                a, z2, d2, aRz, m1 = self._bk_offset_parts(Rv, az, u)
                return a * sdens(a) * _bspecial.ellipe(1.0 - m1) / d2 / xp.sqrt(aRz)

            return zforceint

        integral = self._bk_offset_quad(make_integrand, R, az, xp, dev)
        z_safe = xp.where(z == 0, xp.ones_like(z), z)
        return xp.where(
            z == 0, xp.zeros_like(z), -4.0 * xp.sign(z_safe) * az * integral
        )

    # ------------------------------ R2deriv --------------------------------
    @check_potential_inputs_not_arrays
    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        if is_backend_array(R) or is_backend_array(z):
            return self._bk_dispatch(self._R2deriv_gl, R, z)
        return self._R2deriv_numpy(R, z)

    def _R2deriv_numpy(self, R, z):
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

        return _finite_part_quad(r2derivint, R, az, self._sdens(R) / 2.0)

    def _R2deriv_gl(self, R, z, xp, dev):
        # -dRforce/dR, differentiated under the integral in the offset form: at
        # fixed u the peak (u^2 + z^2) does not move with R, so this is as
        # accurate as the force (the boundary terms cancel, leaving the
        # integrand's value at a = 0, which vanishes). Quadrature of the direct
        # second-derivative integrand is limited to ~eps/|z| near the plane,
        # where its 1/z^2-sized pieces cancel (8.6e-7 at |z| = 1e-8, against
        # 7e-11 here). Sigma'(a) comes from autodiff of the surface density.
        sdens = self._sdens
        grad, _ = autodiff_ops(xp)

        def make_integrand(Rv):
            Rb = _node_axes(Rv, 1)

            def r2derivint(u):
                a, z2, d2, aRz, m1 = self._bk_offset_parts(Rv, z, u)
                E, K = _bspecial.ellipe(1.0 - m1), _bspecial.ellipkm1(m1)
                w = 2.0 * Rb + u  # a + R; d aRz / dR = 4 w, dm/dR = 4 w m1 / aRz
                mm = 1.0 - m1
                dE = (E - K) * 2.0 * w * m1 / (mm * aRz)  # dE/dm = (E-K)/(2m)
                dK = (E - m1 * K) * 2.0 * w / (mm * aRz)  # dK/dm = (E-m1 K)/(2m m1)
                N = (u * w + z2) * E - d2 * K
                dN = 2.0 * u * E + (u * w + z2) * dE - d2 * dK
                dlnD = 1.0 / Rb + 2.0 * w / aRz
                s = sdens(a)
                ds = s + a * grad(lambda x: xp.sum(sdens(x)))(a)
                return (ds * N + a * s * (dN - N * dlnD)) / (Rb * d2 * xp.sqrt(aRz))

            return r2derivint

        return -2.0 * self._bk_offset_quad(make_integrand, R, z, xp, dev)

    # ------------------------------ z2deriv --------------------------------
    @check_potential_inputs_not_arrays
    def _z2deriv(self, R, z, phi=0.0, t=0.0):
        if is_backend_array(R) or is_backend_array(z):
            return self._bk_dispatch(self._z2deriv_gl, R, z)
        return self._z2deriv_numpy(R, z)

    def _z2deriv_numpy(self, R, z):
        R2 = R**2
        az = numpy.fabs(z)
        if az < _R2DERIV_ZFLOOR:
            az = 0.0
        z2 = az**2

        def z2derivint(d):
            """As for R2deriv: parameterised by the offset d = a - R."""
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
                        (a2mR2**2 - 2.0 * (a2 + R2) * z2 - 3.0 * z2**2)
                        * special.ellipe(m)
                    )
                    - z2 * dz * special.ellipkm1(dz / aRz)
                )
                / (dz**2 * aRz**1.5)
            )

        # At z=0 the K term carries a z^2 factor and drops out entirely, leaving
        # -a Sigma(a) E(m) / (d^2 (a+R)), so the singular coefficient is
        # -Sigma(R)/2 -- exactly minus R2deriv's. Verified to 5e-10.
        return _finite_part_quad(z2derivint, R, az, -self._sdens(R) / 2.0)

    def _z2deriv_gl(self, R, z, xp, dev):
        # Laplace's equation off the razor-thin disk, and its z -> 0 limit on
        # it (galpy's value at z = 0): Phi_zz = -Phi_RR - Phi_R / R. At R = 0,
        # Rforce / R tends to -R2deriv.
        R2 = self._R2deriv_gl(R, z, xp, dev)
        R_safe = xp.where(R > 0, R, xp.ones_like(R))
        Rforce_R = xp.where(R > 0, self._Rforce_gl(R, z, xp, dev) / R_safe, -R2)
        return -R2 + Rforce_R

    # ------------------------------ Rzderiv --------------------------------
    @check_potential_inputs_not_arrays
    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        if is_backend_array(R) or is_backend_array(z):
            return self._bk_dispatch(self._Rzderiv_gl, R, z)
        return self._Rzderiv_numpy(R, z)

    def _Rzderiv_numpy(self, R, z):
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

    def _Rzderiv_gl(self, R, z, xp, dev):
        # z == 0 -> Rzderiv = 0 by symmetry; off it, numpy's integrand in the
        # exact offset (its E and K coefficients written in u), floored like zforce
        az = xp.maximum(xp.abs(z), _ZFORCE_ZFLOOR_REL * xp.abs(R))
        az = xp.where(az > 0, az, xp.ones_like(az))
        sdens = self._sdens

        def make_integrand(Rv):
            Rb = _node_axes(Rv, 1)
            R2 = Rb**2

            def rzderivint(u):
                a, z2, d2, aRz, m1 = self._bk_offset_parts(Rv, az, u)
                ecoeff = (
                    u * (16.0 * R2 * Rb + 4.0 * Rb * z2 + 4.0 * Rb * u * u)
                    + u * u * (12.0 * R2 + 2.0 * z2)
                    + u**4
                    - 4.0 * R2 * z2
                    + z2 * z2
                )
                kcoeff = d2 * (2.0 * Rb * u + d2)
                return (
                    a
                    * sdens(a)
                    * (
                        -ecoeff * _bspecial.ellipe(1.0 - m1)
                        + kcoeff * _bspecial.ellipkm1(m1)
                    )
                    / Rb
                    / d2**2
                    / aRz**1.5
                )

            return rzderivint

        integral = self._bk_offset_quad(make_integrand, R, az, xp, dev)
        z_safe = xp.where(z == 0, xp.ones_like(z), z)
        return xp.where(
            z == 0, xp.zeros_like(z), -2.0 * xp.sign(z_safe) * az * integral
        )

    def _surfdens(self, R, z, phi=0.0, t=0.0):
        return self._sdens(R)
