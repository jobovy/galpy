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

if _APY_LOADED:
    from astropy import units

# Gauss-Legendre nodes per panel on the differentiable backend path (the numpy
# path keeps scipy's adaptive quad and is byte-identical). The a=R (m->1)
# singularity is handled by a symmetric plain-GL split at R (0, R, 2R, inf); the
# principal-value cancellation degrades if nodes cluster too close to R, so a
# moderate order is deliberate.
_GLORDER = 100


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
        inner = _bquad.fixed_quad(
            xp, integrand, xp.zeros_like(Rp), Rp, n=_GLORDER, device=dev
        ) + _bquad.fixed_quad(xp, integrand, Rp, 2.0 * Rp, n=_GLORDER, device=dev)
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
            m1, aRz = self._bk_m1_aRz(a, R, z2, xp)
            return a * sdens(a) / xp.sqrt(aRz) * _bspecial.ellipkm1(m1)

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
            faRoveraRz = 4 * a * R / aRz
            return (
                a
                * self._sdens(a)
                * (
                    (a2 - R2 + z2) * special.ellipe(faRoveraRz)
                    - ((a - R) ** 2 + z2) * special.ellipk(faRoveraRz)
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

        def zforceint(a):
            aRz = (a + R) ** 2.0 + z2
            faRoveraRz = 4 * a * R / aRz
            return (
                a
                * self._sdens(a)
                * special.ellipe(faRoveraRz)
                / ((a - R) ** 2 + z2)
                / numpy.sqrt(aRz)
            )

        return (
            -4
            * z
            * (
                integrate.quad(zforceint, 0, 2 * R, points=[R])[0]
                + integrate.quad(zforceint, 2 * R, numpy.inf)[0]
            )
        )

    def _zforce_gl(self, R, z, xp, dev):
        # z==0 is the disk plane (zforce=0 by symmetry) but the integrand has a
        # non-integrable 1/(a-R)^2 pole there; guard the dead branch with z_safe.
        z_safe = xp.where(z == 0, xp.ones_like(z), z)
        z2 = z_safe**2
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
        z2 = z**2

        def r2derivint(a):
            a2 = a**2
            aRz = (a + R) ** 2.0 + z2
            faRoveraRz = 4 * a * R / aRz
            return (
                a
                * self._sdens(a)
                * (
                    -(
                        (
                            (a2 - 3.0 * R2) * (a2 - R2) ** 2
                            + (3.0 * a2**2 + 2.0 * a2 * R2 + 3.0 * R2**2) * z2
                            + (3.0 * a2 + 7.0 * R2) * z**4
                            + z**6
                        )
                        * special.ellipe(faRoveraRz)
                    )
                    + ((a - R) ** 2 + z2)
                    * ((a2 - R2) ** 2 + 2.0 * (a2 + 2.0 * R2) * z2 + z**4)
                    * special.ellipk(faRoveraRz)
                )
                / (2.0 * R2 * ((a - R) ** 2 + z2) ** 2 * ((a + R) ** 2 + z2) ** 1.5)
            )

        return -4 * (
            integrate.quad(r2derivint, 0, 2 * R, points=[R])[0]
            + integrate.quad(r2derivint, 2 * R, numpy.inf)[0]
        )

    def _R2deriv_gl(self, R, z, xp, dev):
        R2 = R**2
        z2 = z**2
        sdens = self._sdens

        def r2derivint(a):
            a2 = a**2
            m1, aRz = self._bk_m1_aRz(a, R, z2, xp)
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
                        * _bspecial.ellipe(1.0 - m1)
                    )
                    + ((a - R) ** 2 + z2)
                    * ((a2 - R2) ** 2 + 2.0 * (a2 + 2.0 * R2) * z2 + z**4)
                    * _bspecial.ellipkm1(m1)
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
        z2 = z**2

        def z2derivint(a):
            a2 = a**2
            aRz = (a + R) ** 2.0 + z2
            faRoveraRz = 4 * a * R / aRz
            return (
                a
                * self._sdens(a)
                * (
                    -(
                        ((a2 - R2) ** 2 - 2.0 * (a2 + R2) * z2 - 3.0 * z**4)
                        * special.ellipe(faRoveraRz)
                    )
                    - z2 * ((a - R) ** 2 + z2) * special.ellipk(faRoveraRz)
                )
                / (((a - R) ** 2 + z2) ** 2 * ((a + R) ** 2 + z2) ** 1.5)
            )

        return -4 * (
            integrate.quad(z2derivint, 0, 2 * R, points=[R])[0]
            + integrate.quad(z2derivint, 2 * R, numpy.inf)[0]
        )

    def _z2deriv_gl(self, R, z, xp, dev):
        R2 = R**2
        z2 = z**2
        sdens = self._sdens

        def z2derivint(a):
            a2 = a**2
            m1, aRz = self._bk_m1_aRz(a, R, z2, xp)
            return (
                a
                * sdens(a)
                * (
                    -(
                        ((a2 - R2) ** 2 - 2.0 * (a2 + R2) * z2 - 3.0 * z**4)
                        * _bspecial.ellipe(1.0 - m1)
                    )
                    - z2 * ((a - R) ** 2 + z2) * _bspecial.ellipkm1(m1)
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

        def rzderivint(a):
            a2 = a**2
            aRz = (a + R) ** 2.0 + z2
            faRoveraRz = 4 * a * R / aRz
            return (
                a
                * self._sdens(a)
                * (
                    -(
                        (
                            a**4
                            - 7.0 * R**4
                            - 6.0 * R2 * z2
                            + z**4
                            + 2.0 * a2 * (3.0 * R2 + z2)
                        )
                        * special.ellipe(faRoveraRz)
                    )
                    + ((a - R) ** 2 + z**2)
                    * (a2 - R2 + z2)
                    * special.ellipk(faRoveraRz)
                )
                / R
                / ((a - R) ** 2 + z2) ** 2
                / ((a + R) ** 2 + z2) ** 1.5
            )

        return (
            -2
            * z
            * (
                integrate.quad(rzderivint, 0, 2 * R, points=[R])[0]
                + integrate.quad(rzderivint, 2 * R, numpy.inf)[0]
            )
        )

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

        integral = self._bk_split_quad(rzderivint, R, xp, dev)
        return xp.where(z == 0, xp.zeros_like(z), -2.0 * z * integral)

    def _surfdens(self, R, z, phi=0.0, t=0.0):
        return self._sdens(R)
