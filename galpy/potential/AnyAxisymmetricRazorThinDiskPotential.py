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
from ..backend._namespaces import under_trace
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

if _APY_LOADED:
    from astropy import units

# Gauss-Legendre nodes per panel on the differentiable backend path (the numpy
# path keeps scipy's adaptive quad and is byte-identical). The a=R (m->1)
# singularity is handled by a symmetric plain-GL split at R (0, R, 2R, inf); the
# principal-value cancellation degrades if nodes cluster too close to R, so a
# moderate order is deliberate.
_GLORDER = 100
# m = 4aR/((a+R)^2+z^2) <= 1 analytically (equality at a=R), but rounding can tip
# it just above 1 near a=R -> sqrt(1-m) of a negative -> NaN in the AGM elliptic
# fallback. Clamp strictly below 1 (only fires on the rounding artifact).
_ELLIP_M_CAP = 1.0 - 1e-15


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
    def _bk_split_quad(self, integrand, R, xp, dev):
        # int_0^inf integrand(a) da as [0, R] + [R, 2R] + [2R, inf); the
        # symmetric plain-GL split at R handles the a=R (m->1) singularity.
        zero = xp.zeros_like(R)
        two_R = 2.0 * R
        return (
            _bquad.fixed_quad(xp, integrand, zero, R, n=_GLORDER, device=dev)
            + _bquad.fixed_quad(xp, integrand, R, two_R, n=_GLORDER, device=dev)
            + _bquad.fixed_quad_semiinfinite(
                xp, integrand, two_R, n=_GLORDER, device=dev
            )
        )

    @staticmethod
    def _bk_m_aRz(a, R, z2, xp):
        aRz = (a + R) ** 2.0 + z2
        m = xp.minimum(4.0 * a * R / aRz, _ELLIP_M_CAP * xp.ones_like(aRz))
        return m, aRz

    def _bk_dispatch(self, numpy_fn, gl_fn, R, z):
        xp = get_namespace(R, z)
        dev = device_of(R, z)
        if not (
            getattr(R, "requires_grad", False)
            or getattr(z, "requires_grad", False)
            # torch.compile does not raise from the float() probe below (dynamo
            # makes it symbolic) and would trace scipy's quad instead: ask.
            or under_trace(R, z)
        ):
            try:  # plain concrete backend input: reuse scipy's accurate value
                Rf, zf = float(R), float(z)
            except Exception:  # tracer: in-backend differentiable GL
                pass
            else:
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

    def _evaluate_gl(self, R, z, xp, dev):
        z2 = z**2
        sdens = self._sdens

        def potint(a):
            m, aRz = self._bk_m_aRz(a, R, z2, xp)
            return a * sdens(a) / xp.sqrt(aRz) * _bspecial.ellipk(m)

        return -4.0 * self._bk_split_quad(potint, R, xp, dev)

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
            m, aRz = self._bk_m_aRz(a, R, z2, xp)
            return (
                a
                * sdens(a)
                * (
                    (a2 - R2 + z2) * _bspecial.ellipe(m)
                    - ((a - R) ** 2 + z2) * _bspecial.ellipk(m)
                )
                / R
                / ((a - R) ** 2 + z2)
                / xp.sqrt(aRz)
            )

        return 2.0 * self._bk_split_quad(rforceint, R, xp, dev)

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
        z_safe = xp.where(z == 0, xp.ones_like(z), z)
        z2 = z_safe**2
        sdens = self._sdens

        def zforceint(a):
            m, aRz = self._bk_m_aRz(a, R, z2, xp)
            return (
                a * sdens(a) * _bspecial.ellipe(m) / ((a - R) ** 2 + z2) / xp.sqrt(aRz)
            )

        integral = self._bk_split_quad(zforceint, R, xp, dev)
        return xp.where(z == 0, xp.zeros_like(z), -4.0 * z * integral)

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
        z2 = z**2
        sdens = self._sdens

        def r2derivint(a):
            a2 = a**2
            m, aRz = self._bk_m_aRz(a, R, z2, xp)
            return (
                a
                * sdens(a)
                * (
                    -(
                        (
                            (a2 - 3.0 * R2) * (a2 - R2) ** 2
                            + (3.0 * a2**2 + 2.0 * a2 * R2 + 3.0 * R2**2) * z2
                            + (3.0 * a2 + 7.0 * R2) * z**4
                            + z**6
                        )
                        * _bspecial.ellipe(m)
                    )
                    + ((a - R) ** 2 + z2)
                    * ((a2 - R2) ** 2 + 2.0 * (a2 + 2.0 * R2) * z2 + z**4)
                    * _bspecial.ellipk(m)
                )
                / (2.0 * R2 * ((a - R) ** 2 + z2) ** 2 * aRz**1.5)
            )

        return -4.0 * self._bk_split_quad(r2derivint, R, xp, dev)

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
        z2 = z**2
        sdens = self._sdens

        def z2derivint(a):
            a2 = a**2
            m, aRz = self._bk_m_aRz(a, R, z2, xp)
            return (
                a
                * sdens(a)
                * (
                    -(
                        ((a2 - R2) ** 2 - 2.0 * (a2 + R2) * z2 - 3.0 * z**4)
                        * _bspecial.ellipe(m)
                    )
                    - z2 * ((a - R) ** 2 + z2) * _bspecial.ellipk(m)
                )
                / (((a - R) ** 2 + z2) ** 2 * aRz**1.5)
            )

        return -4.0 * self._bk_split_quad(z2derivint, R, xp, dev)

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
            m, aRz = self._bk_m_aRz(a, R, z2, xp)
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
                        * _bspecial.ellipe(m)
                    )
                    + ((a - R) ** 2 + z2) * (a2 - R2 + z2) * _bspecial.ellipk(m)
                )
                / R
                / ((a - R) ** 2 + z2) ** 2
                / aRz**1.5
            )

        integral = self._bk_split_quad(rzderivint, R, xp, dev)
        return xp.where(z == 0, xp.zeros_like(z), -2.0 * z * integral)

    def _surfdens(self, R, z, phi=0.0, t=0.0):
        return self._sdens(R)
