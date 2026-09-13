###############################################################################
#   AnyAxisymmetricRazorThinDiskPotential.py: class that implements the
#                                             potential of an arbitrary
#                                             axisymmetric, razor-thin disk
###############################################################################
import numpy
from scipy import integrate, special

from ..backend import (
    asarray_on_device,
    device_of,
    get_namespace,
    is_backend_array,
    match_input_dtype,
)
from ..backend import quadrature as _bquad
from ..backend import special as _bspecial
from ..backend._namespaces import requires_backend_grad, under_trace
from ..util import conversion
from ..util._optional_deps import _APY_LOADED
from .Potential import Potential, check_potential_inputs_not_arrays

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
# Relative floor on |z|/R for the traced zforce; see _zforce_gl for why it is
# relative and where 1e-8 comes from (6 orders above the measured ~1e-14 onset,
# 4 orders inside the 3e-4 the small-z limit test asks for).
_ZFORCE_ZFLOOR_REL = 1e-8

if _APY_LOADED:
    from astropy import units

# Gauss-Legendre nodes per panel on the differentiable backend path (the numpy
# path keeps scipy's adaptive quad and is byte-identical). The a=R (m->1)
# singularity is handled by a symmetric plain-GL split at R (0, R, 2R, inf); the
# principal-value cancellation degrades if nodes cluster too close to R, so a
# moderate order is deliberate.
_GLORDER = 100


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
                surfdens(1)
            except (units.UnitConversionError, units.UnitTypeError):
                _sdens_unit_input = True
            _sdens_unit_output = False
            if _sdens_unit_input:
                try:
                    surfdens(1.0 * units.kpc).to(units.Msun / units.pc**2)
                except (AttributeError, TypeError, units.UnitConversionError):
                    pass
                else:
                    _sdens_unit_output = True
            else:
                try:
                    surfdens(1.0).to(units.Msun / units.pc**2)
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
            -2.0 * numpy.pi * integrate.quad(lambda a: self._sdens(a), 0, numpy.inf)[0]
        )
        if normalize or (
            isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
        ):  # pragma: no cover
            self.normalize(normalize)

    # -----------------------------------------------------------------------
    # Backend (jax/torch) helpers
    #
    # A backend-array input routes here. When a gradient is actually being taken
    # -- a jax/functorch tracer, or a grad-tracking torch tensor -- the compute
    # is in-backend fixed-order Gauss-Legendre so it differentiates. A PLAIN
    # concrete backend scalar instead reuses scipy's accurate adaptive quad
    # (byte-consistent with the numpy path; GL cannot resolve the a=R principal
    # value / small-z sheet structure that the derivative tests FD-probe at
    # z~1e-8, and cannot form the z=0 Hadamard-type second-derivative), and the
    # scipy value is wrapped as a backend array so no downstream ``xp.`` op meets
    # a bare python float.
    # -----------------------------------------------------------------------
    def _bk_split_quad(self, integrand, R, xp, dev, z, K=24, n=50):
        # int_0^inf integrand(a) da as [0, R] + [R, 2R] + [2R, inf); the
        # symmetric plain-GL split at R handles the a=R (m->1) singularity.
        #
        # Both degenerate radii need a guard, or the traced path returns NaN
        # where the concrete scipy path is finite:
        #   R = 0   -- the first two panels have zero width, but the integrand is
        #              0/0 at a=0, so they evaluate to 0*nan. The tail alone is
        #              then the whole [0, inf) integral, which is exactly right.
        #   R = inf -- every panel spans an infinite range. All of these
        #              quantities vanish as R -> inf, so the answer is 0.
        # Selected with xp.where (no data-dependent branch, so it traces), and
        # each dead branch is evaluated at a harmless R so a nan cannot leak back
        # through a gradient.
        finite = xp.isfinite(R)
        Rs = xp.where(finite, R, xp.ones_like(R))
        positive = Rs > 0
        Rp = xp.where(positive, Rs, xp.ones_like(Rs))
        # Dyadic panels graded toward the a=R peak (width ~|z|), issued as ONE
        # batched fixed_quad over a leading panel axis: as separate calls the
        # jit graph is ~16x more expensive to compile.
        #
        # dmin bounds how close panels may approach a=R. At z=0 the integral
        # is singular there and marching panels in makes a finite difference
        # of this potential in R blow up, so grading is switched off; for
        # z>0 the peak is genuinely resolvable and the FD stays clean.
        ones = xp.ones_like(Rp)
        half = 0.5 * Rp
        dmin = xp.where(
            xp.abs(z) * ones > 0.0,
            xp.minimum(4.0 * xp.abs(z) * ones, half),
            half,
        )
        # dk decreasing R/2 -> dmin, so R-dk ascends and R+dk descends -- the
        # right side is the one needing a reverse.
        #
        # Geometric in dmin/half rather than dyadic: a fixed R/2, R/4, ...
        # ladder only reaches R*2**-K, so for |z| < R*2**-K/4 it never gets
        # near the peak and the answer degrades (K=24 is 0.5% low at
        # |z|=1e-10, R=1). Spanning to dmin instead resolves the peak at any
        # |z| with the SAME K, so the jit graph does not grow.
        # ratio is 1 exactly at z==0, where grading is off by construction.
        ratio = dmin / half
        dk = [half * ratio ** (k / (K - 1.0)) for k in range(K)]
        left = [Rp - d for d in dk]  # ascending, ends near R
        right = [Rp + d for d in dk][::-1]  # ascending, starts near R
        lo = [xp.zeros_like(Rp)] + left + [Rp] + right
        hi = left + [Rp] + right + [2.0 * Rp]
        inner = xp.sum(
            _bquad.fixed_quad(
                xp,
                integrand,
                xp.stack(lo, axis=-1),
                xp.stack(hi, axis=-1),
                n=n,
                device=dev,
            ),
            axis=-1,
        )
        out = xp.where(positive, inner, xp.zeros_like(inner)) + (
            _bquad.fixed_quad_semiinfinite(
                xp, integrand, 2.0 * Rs, n=_GLORDER, device=dev
            )
        )
        return xp.where(finite, out, xp.zeros_like(out))

    @staticmethod
    def _bk_m1_aRz(a, R, z2, xp):
        """Return (1-m, aRz) for m = 4aR/((a+R)^2+z^2), complement formed exactly.

        1-m = ((a-R)^2+z^2)/((a+R)^2+z^2) identically, so it is built from the
        quantities the integrands already carry instead of by the subtraction
        1 - 4aR/aRz, which rounds to 0 for |z| << R and sends K to inf. It also
        lies in [0,1] by construction, so m = 1-m1 can never round above 1 and
        no clamp is needed to keep E out of sqrt(negative).
        """
        aRz = (a + R) ** 2.0 + z2
        m1 = ((a - R) ** 2.0 + z2) / aRz
        return m1, aRz

    def _bk_dispatch(self, numpy_fn, gl_fn, R, z):
        xp = get_namespace(R, z)
        dev = device_of(R, z)
        # torch.compile does not raise from the float() probe below (dynamo
        # makes it symbolic) and would trace scipy's quad instead: ask.
        if not (requires_backend_grad(R, z) or under_trace(R, z)):
            # plain concrete backend input: reuse scipy's accurate value. No
            # try/except around float(): every caller is guarded by
            # @check_potential_inputs_not_arrays, so R and z are scalar here, and
            # the tracer cases that used to need the guard (grad, jit, vmap,
            # vmap-of-grad, torch.compile) are all diverted by the requires_grad
            # / under_trace test above -- verified by tracing each of them. A
            # float() failure now would be a genuine surprise, and should raise
            # rather than silently switch to a different quadrature.
            Rf, zf = float(R), float(z)
            # numpy.asarray keeps scipy's float64 value at full precision
            # (asarray_on_device of a bare python float would drop to torch's
            # float32 default, and match_input_dtype's later up-cast cannot
            # recover the lost digits -> a derivative test's dr=1e-8 FD blows
            # up); match_input_dtype then honours the input's own dtype.
            return match_input_dtype(
                asarray_on_device(xp, numpy.asarray(numpy_fn(Rf, zf)), dev), R, z
            )
        return match_input_dtype(gl_fn(R, z, xp, dev), R, z)  # differentiate: GL

    # ------------------------------ potential ------------------------------
    @check_potential_inputs_not_arrays
    def _evaluate(self, R, z, phi=0.0, t=0.0):
        if is_backend_array(R) or is_backend_array(z):
            return self._bk_dispatch(self._evaluate_numpy, self._evaluate_gl, R, z)
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
        z2 = z**2
        sdens = self._sdens

        def potint(a):
            m1, aRz = self._bk_m1_aRz(a, R, z2, xp)
            return a * sdens(a) / xp.sqrt(aRz) * _bspecial.ellipkm1(m1)

        # NOT _bk_split_quad, unlike Rforce and zforce. Phi's peak at a=R is
        # LOGARITHMIC (ellipkm1 ~ -ln(m1)/2), not the 1/((a-R)^2+z^2) pole those
        # two carry, and _bk_split_quad switches its panel grading OFF at z == 0
        # -- which leaves the log sitting on a plain panel edge, where GL
        # converges only algebraically. Measured against scipy, at z = 0:
        #
        #     R        _bk_split_quad   transformed_quad
        #     0.2      2.966e-05        4.425e-11
        #     0.5      4.175e-05        6.072e-11
        #     1.0      3.179e-05        4.541e-11
        #
        # (n=100 sharpens those to ~1e-12.) transformed_quad splits at a=R and
        # clusters nodes into the boundary layer with X = xi^3, which resolves a
        # log; for z > 0 it agrees with the ladder to the last digit at every z
        # sampled (1e-8, 1e-4, 1e-2, 0.1, 1.0), so nothing regresses.
        #
        # It is NOT a drop-in for the other two: xi^3 clustering only reaches
        # ~(1/n)^3 R ~ 5e-7 of the peak, so on zforce's pole it degrades to
        # 1.1e-05 at z=1e-6 and 1.7e-02 at z=1e-8, where the geometric ladder --
        # which marches panel edges to dmin = 4|z| exactly -- holds 8e-13 and
        # 3.8e-10. Different singularity, different rule.
        Rp = xp.where(xp.isfinite(R) & (R > 0), R, xp.ones_like(R))
        inner = _bquad.transformed_quad(
            xp,
            potint,
            xp.zeros_like(Rp),
            2.0 * Rp,
            n=_GLORDER,
            interior_point=Rp,
            device=dev,
        )
        out = xp.where(R > 0, inner, xp.zeros_like(inner)) + (
            _bquad.fixed_quad_semiinfinite(
                xp,
                potint,
                2.0 * xp.where(xp.isfinite(R), R, xp.ones_like(R)),
                n=_GLORDER,
                device=dev,
            )
        )
        return -4.0 * xp.where(xp.isfinite(R), out, xp.zeros_like(out))

    # ------------------------------- Rforce --------------------------------
    @check_potential_inputs_not_arrays
    def _Rforce(self, R, z, phi=0.0, t=0.0):
        if is_backend_array(R) or is_backend_array(z):
            return self._bk_dispatch(self._Rforce_numpy, self._Rforce_gl, R, z)
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
        R2 = R**2
        z2 = z**2
        sdens = self._sdens

        def rforceint(a):
            a2 = a**2
            m1, aRz = self._bk_m1_aRz(a, R, z2, xp)
            return (
                a
                * sdens(a)
                * (
                    (a2 - R2 + z2) * _bspecial.ellipe(1.0 - m1)
                    - ((a - R) ** 2 + z2) * _bspecial.ellipkm1(m1)
                )
                / R
                / ((a - R) ** 2 + z2)
                / xp.sqrt(aRz)
            )

        return 2.0 * self._bk_split_quad(rforceint, R, xp, dev, z)

    # ------------------------------- zforce --------------------------------
    @check_potential_inputs_not_arrays
    def _zforce(self, R, z, phi=0.0, t=0.0):
        if is_backend_array(R) or is_backend_array(z):
            return self._bk_dispatch(self._zforce_numpy, self._zforce_gl, R, z)
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
        # z==0 is the disk plane (zforce=0 by symmetry) but the integrand has a
        # non-integrable 1/(a-R)^2 pole there; guard the dead branch with z_safe.
        #
        # |z| is also floored RELATIVE to R. The peak at a=R has width ~|z|, and
        # _bk_split_quad grades its panels to it by forming edges R +/- d and,
        # in the integrand, a - R -- both by subtraction. Once |z|/R falls to a
        # few tens of machine epsilon those stop being representable, the
        # innermost panels collapse to zero width and the 1/((a-R)^2+z^2) pole
        # is sampled at a == R. Measured onset (traced, vs -2 pi Sigma(R)):
        #
        #     R       0.05      0.2       0.5       2.0
        #     z/R     3.2e-15   3.2e-15   1.0e-14   1.0e-14      (~14-45 eps)
        #
        # i.e. SCALE-FREE in z/R, which is why the floor is relative -- an
        # absolute one would be wrong at small R. Flooring is not an
        # approximation being smuggled in: Fz(z->0+) is exactly -2 pi Sigma(R)
        # for a razor-thin disk, the integral goes as 1/|z|, so -4 |z| I(|z|)
        # evaluated AT the floor already is that limit. The floor sits 6 orders
        # above the 1e-14 onset, so the quadrature is only ever used where it is
        # accurate, and the limit's own error at the floor is O(z/R) ~ 1e-8.
        # numpy needs none of this: _quad_apeak splits on scipy's points=[R],
        # which stays finite, and it saturates INSIDE tolerance instead.
        az = xp.maximum(xp.abs(z), _ZFORCE_ZFLOOR_REL * xp.abs(R))
        # R == 0 (with z == 0) would leave az == 0 and put the pole back; that
        # branch is dead, but keep it finite so it cannot poison a gradient.
        az = xp.where(az > 0, az, xp.ones_like(az))
        z_safe = xp.where(z == 0, xp.ones_like(z), z)
        z2 = az**2
        sdens = self._sdens

        def zforceint(a):
            m1, aRz = self._bk_m1_aRz(a, R, z2, xp)
            return (
                a
                * sdens(a)
                * _bspecial.ellipe(1.0 - m1)
                / ((a - R) ** 2 + z2)
                / xp.sqrt(aRz)
            )

        # az, not z, on BOTH sides: the panel grading must see the floored width
        # it was integrated at, and -4 |z| I(|z|) is what tends to the limit. Fz
        # is odd in z, so the sign is reapplied here rather than carried through.
        integral = self._bk_split_quad(zforceint, R, xp, dev, az)
        return xp.where(
            z == 0, xp.zeros_like(z), -4.0 * xp.sign(z_safe) * az * integral
        )

    # ------------------------------ R2deriv --------------------------------
    @check_potential_inputs_not_arrays
    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        if is_backend_array(R) or is_backend_array(z):
            return self._bk_dispatch(self._R2deriv_numpy, self._R2deriv_gl, R, z)
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
        R2 = R**2
        # Mirror the numpy floor: below it the finite-|z| branch cannot resolve
        # the peak, and numpy evaluates the z=0 limit instead. Without it the
        # sinh branch of finite_part_quad is entered at a width where the
        # substitution cannot work.
        az = xp.abs(z)
        az = xp.where(az < _R2DERIV_ZFLOOR, xp.zeros_like(az), az)
        z2 = az**2
        sdens = self._sdens

        def r2derivint(d):
            # Parameterised by the OFFSET d = a - R, never by a: a^2-R^2 =
            # d(2R+d) and (a-R)^2+z^2 = d^2+z^2 are exact in d, whereas
            # recovering d from a-R loses the digits that 1/(d^2+z^2)^2 then
            # amplifies. m1 = dz/aRz is 1-m exactly, since aRz - dz = 4 R a.
            a = R + d
            a2 = R2 + 2.0 * R * d + d * d
            dz = d * d + z2
            a2mR2 = d * (2.0 * R + d)
            aRz = (2.0 * R + d) ** 2.0 + z2
            m1 = dz / aRz
            return (
                a
                * sdens(a)
                * (
                    -(
                        (
                            (a2 - 3.0 * R2) * a2mR2**2
                            + (3.0 * a2**2 + 2.0 * a2 * R2 + 3.0 * R2**2) * z2
                            + (3.0 * a2 + 7.0 * R2) * z2**2
                            + z2**3
                        )
                        * _bspecial.ellipe(1.0 - m1)
                    )
                    + dz
                    * (a2mR2**2 + 2.0 * (a2 + 2.0 * R2) * z2 + z2**2)
                    * _bspecial.ellipkm1(m1)
                )
                / (2.0 * R2 * dz**2 * aRz**1.5)
            )

        # Hadamard finite part, as the numpy path does: the a=R singularity is
        # a non-integrable c/d^2 divergence, so no amount of panel grading
        # converges it -- the generic split_quad rule used here before was ~1e4
        # off at the midplane.
        return -4.0 * (
            _bquad.finite_part_quad(
                xp, r2derivint, R, c=sdens(R) / 2.0, peak_width=az, device=dev
            )
            + _bquad.fixed_quad_semiinfinite(xp, r2derivint, R, device=dev)
        )

    # ------------------------------ z2deriv --------------------------------
    @check_potential_inputs_not_arrays
    def _z2deriv(self, R, z, phi=0.0, t=0.0):
        if is_backend_array(R) or is_backend_array(z):
            return self._bk_dispatch(self._z2deriv_numpy, self._z2deriv_gl, R, z)
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
        R2 = R**2
        # Mirror the numpy floor: below it the finite-|z| branch cannot resolve
        # the peak, and numpy evaluates the z=0 limit instead. Without it the
        # sinh branch of finite_part_quad is entered at a width where the
        # substitution cannot work.
        az = xp.abs(z)
        az = xp.where(az < _R2DERIV_ZFLOOR, xp.zeros_like(az), az)
        z2 = az**2
        sdens = self._sdens

        def z2derivint(d):
            # Parameterised by the OFFSET d = a - R, never by a: a^2-R^2 =
            # d(2R+d) and (a-R)^2+z^2 = d^2+z^2 are exact in d, whereas
            # recovering d from a-R loses the digits that 1/(d^2+z^2)^2 then
            # amplifies. m1 = dz/aRz is 1-m exactly, since aRz - dz = 4 R a.
            a = R + d
            a2 = R2 + 2.0 * R * d + d * d
            dz = d * d + z2
            a2mR2 = d * (2.0 * R + d)
            aRz = (2.0 * R + d) ** 2.0 + z2
            m1 = dz / aRz
            return (
                a
                * sdens(a)
                * (
                    -(
                        (a2mR2**2 - 2.0 * (a2 + R2) * z2 - 3.0 * z2**2)
                        * _bspecial.ellipe(1.0 - m1)
                    )
                    - z2 * dz * _bspecial.ellipkm1(m1)
                )
                / (dz**2 * aRz**1.5)
            )

        # At z=0 the K term carries a z^2 factor and drops out, leaving
        # -a Sigma(a) E(m) / (d^2 (a+R)), so the singular coefficient is
        # -Sigma(R)/2 -- exactly minus R2deriv's.
        return -4.0 * (
            _bquad.finite_part_quad(
                xp, z2derivint, R, c=-sdens(R) / 2.0, peak_width=az, device=dev
            )
            + _bquad.fixed_quad_semiinfinite(xp, z2derivint, R, device=dev)
        )

    # ------------------------------ Rzderiv --------------------------------
    @check_potential_inputs_not_arrays
    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        if is_backend_array(R) or is_backend_array(z):
            return self._bk_dispatch(self._Rzderiv_numpy, self._Rzderiv_gl, R, z)
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
        R2 = R**2
        # z==0 -> Rzderiv=0 by symmetry; guard the 1/(a-R)^2 pole with z_safe.
        z_safe = xp.where(z == 0, xp.ones_like(z), z)
        z2 = z_safe**2
        sdens = self._sdens

        def rzderivint(a):
            a2 = a**2
            m1, aRz = self._bk_m1_aRz(a, R, z2, xp)
            return (
                a
                * sdens(a)
                * (
                    -(
                        (
                            a**4
                            - 7.0 * R**4
                            - 6.0 * R2 * z2
                            + z2**2
                            + 2.0 * a2 * (3.0 * R2 + z2)
                        )
                        * _bspecial.ellipe(1.0 - m1)
                    )
                    + ((a - R) ** 2 + z2) * (a2 - R2 + z2) * _bspecial.ellipkm1(m1)
                )
                / R
                / ((a - R) ** 2 + z2) ** 2
                / aRz**1.5
            )

        integral = self._bk_split_quad(rzderivint, R, xp, dev, z)
        return xp.where(z == 0, xp.zeros_like(z), -2.0 * z * integral)

    def _surfdens(self, R, z, phi=0.0, t=0.0):
        return self._sdens(R)
