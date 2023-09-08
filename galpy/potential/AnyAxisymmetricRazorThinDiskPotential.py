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

if _APY_LOADED:
    from astropy import units


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
        potint = (
            lambda a: a
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

    @check_potential_inputs_not_arrays
    def _zforce(self, R, z, phi=0.0, t=0.0):
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

    @check_potential_inputs_not_arrays
    def _R2deriv(self, R, z, phi=0.0, t=0.0):
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

    @check_potential_inputs_not_arrays
    def _z2deriv(self, R, z, phi=0.0, t=0.0):
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

    @check_potential_inputs_not_arrays
    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
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

    def _surfdens(self, R, z, phi=0.0, t=0.0):
        return self._sdens(R)
