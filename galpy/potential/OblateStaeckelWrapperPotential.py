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

from galpy.util import conversion, coords

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
    r"""Potential wrapper class that approximates a given axisymmetric potential as an oblate Staeckel potential by defining (see `Binney 2012 <https://ui.adsabs.harvard.edu/abs/2012MNRAS.426.1324B/abstract>`__; `Bovy 2026 <https://galaxiesbook.org/chapters/II-03.-Orbits-in-Disks_4-Action-angle-coordinates-in-and-around-disks.html#specifically-choosing-a-reference-value-u-0>`__)

    .. math::
        :nowrap:

        \begin{align}
        U(u) & = \cosh^2 u \,\Phi\left(u,{\pi\over 2}\right)\,,\\
        V(v) & = \cosh^2 u_0 \,\Phi\left(u_0,{\pi\over 2}\right)-\left(\sinh^2 u_0+\sin^2 v\right)\,\Phi\left(u_0,v\right)\,.
        \end{align}

    in the prolate spheroidal coordinate system defined by the focal length :math:`\Delta`. Here :math:`u_0` is a reference value of :math:`u` at which the potential is split. The potential is then given by

    .. math::
        :nowrap:

        \begin{align}
        \Phi(u,v) & = {U(u)-V(v)\over \sinh^2 u + \sin^2 v}\,.
        \end{align}

    """

    def __init__(self, amp=1.0, pot=None, delta=0.5, u0=0.0, ro=None, vo=None):
        """Initialize an OblateStaeckelWrapper Potential.

        Parameters
        ----------
        amp : float, optional
            Amplitude to be applied to the potential. Default is 1.0.
        pot : Potential or a combined potential formed using addition (pot1+pot2+…)
            Potential instance or a combined potential formed using addition (pot1+pot2+…); this potential is made into an oblate Staeckel potential.
        delta : float or Quantity, optional
            The focal length. Default is 0.5.
        u0 : float or tuple or tuple of Quantity
            Reference u value; if a tuple is given, this is assumed to be a (R,z) value to be converted to u.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2017-12-15 - Started - Bovy (UofT)
        """
        self._delta = conversion.parse_length(delta, ro=ro)
        if u0 is None:  # pragma: no cover
            raise ValueError(
                "u0= needs to be given to setup OblateStaeckelWrapperPotential"
            )
        if isinstance(u0, (tuple, list, numpy.ndarray)):
            self._u0 = coords.Rz_to_uv(
                conversion.parse_length(u0[0], ro=ro),
                conversion.parse_length(u0[1], ro=ro),
                delta=self._delta,
            )[0]
        else:
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

    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        """
        NAME:
           _R2deriv
        PURPOSE:
           evaluate the 2nd radial derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the 2nd radial derivative
        HISTORY:
           2017-01-21 - Written - Bovy (UofT)
        """
        u, v = coords.Rz_to_uv(R, z, delta=self._delta)
        prefac = _staeckel_prefactor(u, v)
        dprefacdu, dprefacdv = _dstaeckel_prefactordudv(u, v)
        d2prefacdu2, d2prefacdv2 = _dstaeckel_prefactord2ud2v(u, v)
        umvfac = (
            dprefacdu * self._delta * numpy.sin(v) * numpy.cosh(u)
            + dprefacdv * numpy.tanh(u) * z
        ) / prefac  # xs (U-V) in Rforce
        U = self._U(u)
        dUdu = self._dUdu(u)
        d2Udu2 = self._d2Udu2(u)
        V = self._V(v)
        dVdv = self._dVdv(v)
        d2Vdv2 = self._d2Vdv2(v)
        return (
            d2Udu2 * numpy.sin(v) ** 2.0 * numpy.cosh(u) ** 2.0
            + dUdu * numpy.sinh(u) * numpy.cosh(u)
            - d2Vdv2 * numpy.sinh(u) ** 2.0 * numpy.cos(v) ** 2.0
            - dVdv * numpy.sin(v) * numpy.cos(v)
            + (
                (
                    -dUdu * numpy.cosh(u) * numpy.sin(v)
                    + dVdv * numpy.sinh(u) * numpy.cos(v)
                )
                / self._delta
                * umvfac
                + (U - V)
                * (
                    -d2prefacdu2 * numpy.cosh(u) ** 2.0 * numpy.sin(v) ** 2.0
                    - dprefacdu * numpy.sinh(u) * numpy.cosh(u)
                    - d2prefacdv2 * numpy.sinh(u) ** 2.0 * numpy.cos(v) ** 2.0
                    - dprefacdv * numpy.sin(v) * numpy.cos(v)
                )
                / prefac
                + (U - V)
                * umvfac
                / prefac
                / self._delta
                * (
                    dprefacdu * numpy.cosh(u) * numpy.sin(v)
                    + dprefacdv * numpy.sinh(u) * numpy.cos(v)
                )
            )
        ) / self._delta**2.0 / prefac**3.0 + 2.0 * self._Rforce(
            R, z, phi=phi, t=t
        ) / prefac**2.0 * (
            dprefacdu * numpy.cosh(u) * numpy.sin(v)
            + dprefacdv * numpy.sinh(u) * numpy.cos(v)
        ) / self._delta

    def _z2deriv(self, R, z, phi=0.0, t=0.0):
        """
        NAME:
           _z2deriv
        PURPOSE:
           evaluate the 2nd vertical derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the 2nd vertical derivative
        HISTORY:
           2017-01-21 - Written - Bovy (UofT)
        """
        u, v = coords.Rz_to_uv(R, z, delta=self._delta)
        prefac = _staeckel_prefactor(u, v)
        dprefacdu, dprefacdv = _dstaeckel_prefactordudv(u, v)
        d2prefacdu2, d2prefacdv2 = _dstaeckel_prefactord2ud2v(u, v)
        umvfac = (
            dprefacdu / numpy.tan(v) * R  # xs (U-V) in zforce
            - dprefacdv * self._delta * numpy.sin(v) * numpy.cosh(u)
        ) / prefac
        U = self._U(u)
        dUdu = self._dUdu(u)
        d2Udu2 = self._d2Udu2(u)
        V = self._V(v)
        dVdv = self._dVdv(v)
        d2Vdv2 = self._d2Vdv2(v)
        return (
            d2Udu2 * numpy.sinh(u) ** 2.0 * numpy.cos(v) ** 2.0
            + dUdu * numpy.cosh(u) * numpy.sinh(u)
            - d2Vdv2 * numpy.sin(v) ** 2.0 * numpy.cosh(u) ** 2.0
            - dVdv * numpy.cos(v) * numpy.sin(v)
            + (
                (
                    -dUdu * numpy.sinh(u) * numpy.cos(v)
                    - dVdv * numpy.cosh(u) * numpy.sin(v)
                )
                / self._delta
                * umvfac
                + (U - V)
                * (
                    -d2prefacdu2 * numpy.sinh(u) ** 2.0 * numpy.cos(v) ** 2.0
                    - dprefacdu * numpy.sinh(u) * numpy.cosh(u)
                    - d2prefacdv2 * numpy.sin(v) ** 2.0 * numpy.cosh(u) ** 2.0
                    - dprefacdv * numpy.cos(v) * numpy.sin(v)
                )
                / prefac
                - (U - V)
                * umvfac
                / prefac
                / self._delta
                * (
                    -dprefacdu * numpy.sinh(u) * numpy.cos(v)
                    + dprefacdv * numpy.cosh(u) * numpy.sin(v)
                )
            )
        ) / self._delta**2.0 / prefac**3.0 - 2.0 * self._zforce(
            R, z, phi=phi, t=t
        ) / prefac**2.0 * (
            -dprefacdu * numpy.sinh(u) * numpy.cos(v)
            + dprefacdv * numpy.cosh(u) * numpy.sin(v)
        ) / self._delta

    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        """
        NAME:
           _Rzderiv
        PURPOSE:
           evaluate the mixed radial and vertical derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the mixed radial and vertical derivative
        HISTORY:
           2017-01-22 - Written - Bovy (UofT)
        """
        u, v = coords.Rz_to_uv(R, z, delta=self._delta)
        prefac = _staeckel_prefactor(u, v)
        dprefacdu, dprefacdv = _dstaeckel_prefactordudv(u, v)
        d2prefacdu2, d2prefacdv2 = _dstaeckel_prefactord2ud2v(u, v)
        umvfac = (
            dprefacdu / numpy.tan(v) * R  # xs (U-V) in zforce
            - dprefacdv * self._delta * numpy.sin(v) * numpy.cosh(u)
        ) / prefac
        U = self._U(u)
        dUdu = self._dUdu(u)
        d2Udu2 = self._d2Udu2(u)
        V = self._V(v)
        dVdv = self._dVdv(v)
        d2Vdv2 = self._d2Vdv2(v)
        return (
            (d2Udu2 + d2Vdv2)
            * numpy.cosh(u)
            * numpy.sin(v)
            * numpy.cos(v)
            * numpy.sinh(u)
            + dUdu * numpy.sin(v) * numpy.cos(v)
            + dVdv * numpy.sinh(u) * numpy.cosh(u)
            + (
                (
                    -dUdu * numpy.cosh(u) * numpy.sin(v)
                    + dVdv * numpy.sinh(u) * numpy.cos(v)
                )
                / self._delta
                * umvfac
                + (U - V)
                * (
                    (-d2prefacdu2 + d2prefacdv2)
                    * numpy.sin(v)
                    * numpy.cosh(u)
                    * numpy.sinh(u)
                    * numpy.cos(v)
                    - dprefacdu * numpy.sin(v) * numpy.cos(v)
                    + dprefacdv * numpy.cosh(u) * numpy.sinh(u)
                )
                / prefac
                + (U - V)
                * umvfac
                / prefac
                / self._delta
                * (
                    dprefacdu * numpy.cosh(u) * numpy.sin(v)
                    + dprefacdv * numpy.sinh(u) * numpy.cos(v)
                )
            )
        ) / self._delta**2.0 / prefac**3.0 + 2.0 * self._zforce(
            R, z, phi=phi, t=t
        ) / prefac**2.0 * (
            dprefacdu * numpy.cosh(u) * numpy.sin(v)
            + dprefacdv * numpy.sinh(u) * numpy.cos(v)
        ) / self._delta

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

    def _d2Udu2(self, u):
        Rz0 = coords.uv_to_Rz(u, self._v0, delta=self._delta)
        tRforce = _evaluateRforces(self._pot, Rz0[0], Rz0[1])
        tzforce = _evaluatezforces(self._pot, Rz0[0], Rz0[1])
        return (
            2.0 * numpy.cosh(2 * u) * _evaluatePotentials(self._pot, Rz0[0], Rz0[1])
            - 4.0
            * numpy.cosh(u)
            * numpy.sinh(u)
            * (
                tRforce * Rz0[0] / (numpy.tanh(u) + 1e-12)
                + tzforce * Rz0[1] * numpy.tanh(u)
            )
            - numpy.cosh(u) ** 2.0
            * (
                -evaluateR2derivs(self._pot, Rz0[0], Rz0[1], use_physical=False)
                * Rz0[0] ** 2.0
                / (numpy.tanh(u) + 1e-12) ** 2.0
                - 2.0
                * evaluateRzderivs(self._pot, Rz0[0], Rz0[1], use_physical=False)
                * Rz0[0]
                * Rz0[1]
                + tRforce * Rz0[0]
                - evaluatez2derivs(self._pot, Rz0[0], Rz0[1], use_physical=False)
                * Rz0[1] ** 2.0
                * numpy.tanh(u) ** 2.0
                + tzforce * Rz0[1]
            )
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

    def _d2Vdv2(self, v):
        R0z = coords.uv_to_Rz(self._u0, v, delta=self._delta)
        tRforce = _evaluateRforces(self._pot, R0z[0], R0z[1])
        tzforce = _evaluatezforces(self._pot, R0z[0], R0z[1])
        return (
            -2.0 * numpy.cos(2.0 * v) * _evaluatePotentials(self._pot, R0z[0], R0z[1])
            + 2.0
            * numpy.sin(2.0 * v)
            * (tRforce * R0z[0] / numpy.tan(v) - tzforce * R0z[1] * numpy.tan(v))
            + _staeckel_prefactor(self._u0, v)
            * (
                -evaluateR2derivs(self._pot, R0z[0], R0z[1], use_physical=False)
                * R0z[0] ** 2.0
                / numpy.tan(v) ** 2.0
                + 2.0
                * evaluateRzderivs(self._pot, R0z[0], R0z[1], use_physical=False)
                * R0z[0]
                * R0z[1]
                - tRforce * R0z[0]
                - evaluatez2derivs(self._pot, R0z[0], R0z[1], use_physical=False)
                * R0z[1] ** 2.0
                * numpy.tan(v) ** 2.0
                - tzforce * R0z[1]
            )
        )


def _staeckel_prefactor(u, v):
    return numpy.sinh(u) ** 2.0 + numpy.sin(v) ** 2.0


def _dstaeckel_prefactordudv(u, v):
    return (2.0 * numpy.sinh(u) * numpy.cosh(u), 2.0 * numpy.sin(v) * numpy.cos(v))


def _dstaeckel_prefactord2ud2v(u, v):
    return (2.0 * numpy.cosh(2.0 * u), 2.0 * numpy.cos(2.0 * v))
