###############################################################################
#   AnySphericalPotential: Potential of an arbitrary spherical density
###############################################################################
import numpy
from scipy import integrate

from ..util import conversion
from ..util._optional_deps import _APY_LOADED
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
                except (AttributeError, units.UnitConversionError):
                    pass
                else:
                    _dens_unit_output = True
            else:
                try:
                    dens(1.0).to(units.Msun / units.pc**3)
                except (AttributeError, units.UnitConversionError):
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
        if not hasattr(self, "_rawdens"):  # unitless
            self._rawdens = dens
        self._rawmass = (
            lambda r: 4.0
            * numpy.pi
            * integrate.quad(
                lambda a: a**2 * self._rawdens(a), 0, numpy.atleast_1d(r)[0]
            )[0]
        )
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

    def _revaluate(self, r, t=0.0):
        """Potential as a function of r and time"""
        if r == 0:
            return self._pot_zero
        elif numpy.isinf(r):
            return self._pot_inf
        else:
            return (
                -self._rawmass(r) / r
                - 4.0
                * numpy.pi
                * integrate.quad(
                    lambda a: self._rawdens(a) * a, numpy.atleast_1d(r)[0], numpy.inf
                )[0]
            )

    def _rforce(self, r, t=0.0):
        return -self._rawmass(r) / r**2

    def _r2deriv(self, r, t=0.0):
        return -2 * self._rawmass(r) / r**3.0 + 4.0 * numpy.pi * self._rawdens(r)

    def _rdens(self, r, t=0.0):
        return self._rawdens(r)
