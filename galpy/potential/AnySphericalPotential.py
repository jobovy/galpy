###############################################################################
#   AnySphericalPotential: Potential of an arbitrary spherical density
###############################################################################
import numpy
from scipy import integrate

from ..backend import as_backend_constant, as_numpy, get_namespace, is_backend_array
from ..util import conversion
from ..util._optional_deps import _APY_LOADED
from ..util.quadpack import quad_over_limits
from .SphericalPotential import SphericalPotential

if _APY_LOADED:
    from astropy import units


class AnySphericalPotential(SphericalPotential):
    """Class that implements the potential of an arbitrary spherical density distribution :math:`\\rho(r)`"""

    def __init__(
        self,
        amp=1.0,
        dens=lambda r: 0.64 / r / (1 + r) ** 3,
        normalize=False,
        ro=None,
        vo=None,
    ):
        """
        Initialize the potential of an arbitrary spherical density distribution.

        Parameters
        ----------
        amp : float, optional
            Amplitude to be applied to the potential. Default is 1.0.
        dens : callable, optional
            A function of a single variable that gives the density as a function of radius (can return a Quantity). Default is ``lambda r: 0.64 / r / (1 + r) ** 3``.
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2021-01-05 - Written - Bovy (UofT)

        """
        SphericalPotential.__init__(self, amp=amp, ro=ro, vo=vo)
        # numpy path: _rawmass closes over scipy.integrate.quad with a SCALAR
        # upper limit (r[0] of an array), so the force is scalar-only -- an
        # array input silently collapses to its first element. Tell
        # Potential.mass to drive the backend GL quadrature node-by-node
        # (vectorized=False) instead of feeding the whole node array.
        self._force_accepts_arrays = False
        # A jax/torch coord routes _rawmass / the _revaluate tail to in-backend
        # Gauss-Legendre quadrature (see below); flag as backend-capable.
        self._backend_compatible = True
        # A units-based density is numpy-only (astropy Quantity arithmetic
        # strips a jax/torch node to numpy); on a backend node it is evaluated
        # on numpy and the result anchored back on the backend (see
        # _backend_dens). Set True below when the density involves units.
        self._dens_needs_numpy = False
        # Parse density: does it have units? does it expect them?
        if _APY_LOADED:
            _dens_unit_input = False
            try:
                dens(1)
            except (units.UnitConversionError, units.UnitTypeError):
                _dens_unit_input = True
            _dens_unit_output = False
            if _dens_unit_input:
                try:
                    dens(1.0 * units.kpc).to(units.Msun / units.pc**3)
                except (AttributeError, units.UnitConversionError, TypeError):
                    pass
                else:
                    _dens_unit_output = True
            else:
                try:
                    dens(1.0).to(units.Msun / units.pc**3)
                except (AttributeError, units.UnitConversionError, TypeError):
                    pass
                else:
                    _dens_unit_output = True
            if _dens_unit_input and _dens_unit_output:
                self._rawdens = lambda R: conversion.parse_dens(
                    dens(R * self._ro * units.kpc), ro=self._ro, vo=self._vo
                )
            elif _dens_unit_input:
                self._rawdens = lambda R: dens(R * self._ro * units.kpc)
            elif _dens_unit_output:
                self._rawdens = lambda R: conversion.parse_dens(
                    dens(R), ro=self._ro, vo=self._vo
                )
            self._dens_needs_numpy = _dens_unit_input or _dens_unit_output
        if not hasattr(self, "_rawdens"):  # unitless
            self._rawdens = dens
        # The potential at zero, try to figure out whether it's finite
        _zero_msg = integrate.quad(
            lambda a: a * self._rawdens(a), 0, numpy.inf, full_output=True
        )[-1]
        _infpotzero = "divergent" in _zero_msg or "maximum number" in _zero_msg
        self._pot_zero = (
            -numpy.inf
            if _infpotzero
            else -4.0
            * numpy.pi
            * integrate.quad(lambda a: a * self._rawdens(a), 0, numpy.inf)[0]
        )
        # The potential at infinity
        _infmass = (
            "divergent"
            in integrate.quad(
                lambda a: a**2.0 * self._rawdens(a), 0, numpy.inf, full_output=True
            )[-1]
        )
        self._pot_inf = 0.0 if not _infmass else numpy.inf
        # Normalize?
        if normalize or (
            isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
        ):  # pragma: no cover
            self.normalize(normalize)
        return None

    def _backend_dens(self, a):
        """Evaluate the density, keeping the backend-quadrature path type-clean.

        A units-based ``dens`` runs through astropy Quantity arithmetic, which
        strips a jax/torch node to numpy (emitting a numpy-2 ``__array__``
        deprecation) and yields numpy -- and ``numpy * Tensor`` then raises. Such
        a density is inherently non-differentiable, so on a backend node it is
        evaluated on the numpy node and the result anchored back on the node's
        backend/dtype/device. A backend-native (differentiable) density and the
        numpy path both pass through untouched (``is_backend_array(a)`` is False
        for numpy), so the numpy path stays byte-identical.
        """
        if is_backend_array(a) and self._dens_needs_numpy:
            d = numpy.asarray(self._rawdens(as_numpy(a)))
            return as_backend_constant(get_namespace(a), d, a)
        return self._rawdens(a)

    def _rawmass(self, r):
        r"""Enclosed mass :math:`4\pi\int_0^r a^2\rho(a)\,da`.

        numpy: scipy.integrate.quad with a SCALAR upper limit (byte-identical to
        the historical closure -- an array ``r`` collapses to ``r[0]``). A
        jax/torch ``r`` routes to in-backend fixed-order Gauss-Legendre so the
        mass (and the force / 2nd derivative built on it) differentiates w.r.t.
        ``r`` and through the density's parameters.
        """
        if is_backend_array(r):
            from ..backend.quadrature import quad as _bk_quad

            return (
                4.0
                * numpy.pi
                * _bk_quad(lambda a: a**2 * self._backend_dens(a), 0.0, r)
            )
        return (
            4.0
            * numpy.pi
            * integrate.quad(
                lambda a: a**2 * self._rawdens(a), 0, numpy.atleast_1d(r).flatten()[0]
            )[0]
        )

    def _revaluate(self, r, t=0.0):
        """Potential as a function of r and time"""
        if is_backend_array(r):
            from ..backend.quadrature import fixed_quad_semiinfinite

            xp = get_namespace(r)
            # -M(r)/r - 4 pi int_r^inf rho(a) a da (tail via the recip s=1/u^2-1
            # substitution, differentiable in r). The scalar edges r == 0
            # (M/r -> 0/0) and r == inf (both terms -> 0) DO reach this path from
            # the forced-backend test_potential; evaluate the bulk formula at a
            # safe r (keeps the dead where-branch finite for reverse-mode AD too)
            # and select the precomputed edge values.
            edge = (r == 0) | xp.isinf(r)
            r_safe = xp.where(edge, xp.ones_like(r), r)
            tail = fixed_quad_semiinfinite(
                xp, lambda a: self._backend_dens(a) * a, r_safe, kind="recip"
            )
            bulk = -self._rawmass(r_safe) / r_safe - 4.0 * numpy.pi * tail
            out = xp.where(r == 0, self._pot_zero, bulk)
            return xp.where(xp.isinf(r), self._pot_inf, out)
        # r == 0 / isinf(r) are per-element questions, so an array r has to be
        # handled element by element; scipy's quad wants a scalar limit anyway.
        if numpy.ndim(r) == 0:
            return self._revaluate_scalar(r)
        rr = numpy.asarray(r)
        return numpy.reshape([self._revaluate_scalar(x) for x in rr.ravel()], rr.shape)

    def _revaluate_scalar(self, r):
        if r == 0:
            return self._pot_zero
        elif numpy.isinf(r):
            return self._pot_inf
        else:
            return -self._rawmass(r) / r - 4.0 * numpy.pi * quad_over_limits(
                lambda a: self._rawdens(a) * a, r, numpy.inf
            )

    def _rforce(self, r, t=0.0):
        return -self._rawmass(r) / r**2

    def _r2deriv(self, r, t=0.0):
        return -2 * self._rawmass(r) / r**3.0 + 4.0 * numpy.pi * self._backend_dens(r)

    def _rdens(self, r, t=0.0):
        return self._backend_dens(r)
