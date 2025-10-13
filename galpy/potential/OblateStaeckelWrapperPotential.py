###############################################################################
#   OblateStaeckelWrapperPotential.py: Wrapper to turn an axisymmetric
#                                      potential into an oblate Staeckel
#                                      potential following Binney (2012)
#
#   NOT A TYPICAL WRAPPER, SO DON'T USE THIS BLINDLY AS A TEMPLATE FOR NEW
#   WRAPPERS
#
###############################################################################
import numpy

from galpy.util import coords

from .Potential import (
    _APY_LOADED,
    _evaluatePotentials,
    _evaluateRforces,
    _evaluatezforces,
    evaluateR2derivs,
    evaluateRzderivs,
    evaluatez2derivs,
)
from .WrapperPotential import parentWrapperPotential

if _APY_LOADED:
    from astropy import units


class OblateStaeckelWrapperPotential(parentWrapperPotential):
    """Potential wrapper class that approximates a given axisymmetric potential as an oblate Staeckel potential, following the scheme of Binney (2012)"""

    def __init__(self, amp=1.0, pot=None, delta=0.5, u0=0.0, ro=None, vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a OblateStaeckelWrapper Potential

        INPUT:

           amp - amplitude to be applied to the potential (default: 1.)

           pot - Potential instance or list thereof; this potential is made to rotate around the z axis by the wrapper

           delta= (0.5) the focal length

           u0= (None) reference u value

        OUTPUT:

           (none)

        HISTORY:

           2017-12-15 - Started - Bovy (UofT)

        """
        if _APY_LOADED and isinstance(delta, units.Quantity):
            delta = delta.to(units.kpc).value / self._ro
        self._delta = delta
        if u0 is None:  # pragma: no cover
            raise ValueError(
                "u0= needs to be given to setup OblateStaeckelWrapperPotential"
            )
        self._u0 = u0
        self._v0 = numpy.pi / 2.0  # so we know when we're using this
        R0, z0 = coords.uv_to_Rz(self._u0, self._v0, delta=self._delta)
        self._refpot = (
            _evaluatePotentials(self._pot, R0, z0) * numpy.cosh(self._u0) ** 2.0
        )
        self.hasC = True
        self.hasC_dxdv = False

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        """
        NAME:
           _evaluate
        PURPOSE:
           evaluate the potential at R,z
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           Phi(R,z)
        HISTORY:
           2017-12-15 - Written - Bovy (UofT)
        """
        u, v = coords.Rz_to_uv(R, z, delta=self._delta)
        return (self._U(u) - self._V(v)) / _staeckel_prefactor(u, v)

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        """
        NAME:
           _Rforce
        PURPOSE:
           evaluate the radial force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the radial force
        HISTORY:
           2017-12-15 - Written - Bovy (UofT)
        """
        u, v = coords.Rz_to_uv(R, z, delta=self._delta)
        prefac = _staeckel_prefactor(u, v)
        dprefacdu, dprefacdv = _dstaeckel_prefactordudv(u, v)
        return (
            (
                -self._dUdu(u) * self._delta * numpy.sin(v) * numpy.cosh(u)
                + self._dVdv(v) * numpy.tanh(u) * z
                + (self._U(u) - self._V(v))
                * (
                    dprefacdu * self._delta * numpy.sin(v) * numpy.cosh(u)
                    + dprefacdv * numpy.tanh(u) * z
                )
                / prefac
            )
            / self._delta**2.0
            / prefac**2.0
        )

    def _zforce(self, R, z, phi=0.0, t=0.0):
        """
        NAME:
           _zforce
        PURPOSE:
           evaluate the vertical force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the vertical force
        HISTORY:
           2017-12-15 - Written - Bovy (UofT)
        """
        u, v = coords.Rz_to_uv(R, z, delta=self._delta)
        prefac = _staeckel_prefactor(u, v)
        dprefacdu, dprefacdv = _dstaeckel_prefactordudv(u, v)
        return (
            (
                -self._dUdu(u) * R / numpy.tan(v)
                - self._dVdv(v) * self._delta * numpy.sin(v) * numpy.cosh(u)
                + (self._U(u) - self._V(v))
                * (
                    dprefacdu / numpy.tan(v) * R
                    - dprefacdv * self._delta * numpy.sin(v) * numpy.cosh(u)
                )
                / prefac
            )
            / self._delta**2.0
            / prefac**2.0
        )

    def _U(self, u):
        """Approximated U(u) = cosh^2(u) Phi(u,pi/2)"""
        Rz0 = coords.uv_to_Rz(u, self._v0, delta=self._delta)
        return numpy.cosh(u) ** 2.0 * _evaluatePotentials(self._pot, Rz0[0], Rz0[1])

    def _dUdu(self, u):
        Rz0 = coords.uv_to_Rz(u, self._v0, delta=self._delta)
        # 1e-12 bc force should win the 0/0 battle
        return 2.0 * numpy.cosh(u) * numpy.sinh(u) * _evaluatePotentials(
            self._pot, Rz0[0], Rz0[1]
        ) - numpy.cosh(u) ** 2.0 * (
            _evaluateRforces(self._pot, Rz0[0], Rz0[1])
            * Rz0[0]
            / (numpy.tanh(u) + 1e-12)
            + _evaluatezforces(self._pot, Rz0[0], Rz0[1]) * Rz0[1] * numpy.tanh(u)
        )

    def _V(self, v):
        """Approximated
        V(v) = cosh^2(u0) Phi(u0,pi/2) - (sinh^2(u0)+sin^2(v)) Phi(u0,v)"""
        R0z = coords.uv_to_Rz(self._u0, v, delta=self._delta)
        return self._refpot - _staeckel_prefactor(self._u0, v) * _evaluatePotentials(
            self._pot, R0z[0], R0z[1]
        )

    def _dVdv(self, v):
        R0z = coords.uv_to_Rz(self._u0, v, delta=self._delta)
        return -2.0 * numpy.sin(v) * numpy.cos(v) * _evaluatePotentials(
            self._pot, R0z[0], R0z[1]
        ) + _staeckel_prefactor(self._u0, v) * (
            _evaluateRforces(self._pot, R0z[0], R0z[1]) * R0z[0] / numpy.tan(v)
            - _evaluatezforces(self._pot, R0z[0], R0z[1]) * R0z[1] * numpy.tan(v)
        )


def _staeckel_prefactor(u, v):
    return numpy.sinh(u) ** 2.0 + numpy.sin(v) ** 2.0


def _dstaeckel_prefactordudv(u, v):
    return (2.0 * numpy.sinh(u) * numpy.cosh(u), 2.0 * numpy.sin(v) * numpy.cos(v))
