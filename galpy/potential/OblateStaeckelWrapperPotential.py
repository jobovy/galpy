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
from scipy.interpolate import CubicSpline

from galpy.util import conversion, coords
from galpy.util.coords import _promote_scalars_for

from ..backend import get_namespace
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

    def __init__(
        self,
        amp=1.0,
        pot=None,
        delta=0.5,
        u0=None,
        ntab=None,
        Rmax_tab=100.0,
        ro=None,
        vo=None,
    ):
        """Initialize an OblateStaeckelWrapper Potential.

        Parameters
        ----------
        amp : float, optional
            Amplitude to be applied to the potential. Default is 1.0.
        pot : Potential or a combined potential formed using addition (pot1+pot2+…)
            Potential instance or a combined potential formed using addition (pot1+pot2+…); this potential is made into an oblate Staeckel potential.
        delta : float or Quantity, optional
            The focal length. Default is 0.5.
        u0 : float or tuple or tuple of Quantity, optional
            Reference u value, the curve along which V(v) is built; if a tuple is given, this is assumed to be a (R,z) value to be converted to u. Defaults to arcsinh(1/delta), the value that places the reference curve at R=1 in the plane, whatever delta is. V(v) only represents the wrapped potential well near the reference curve unless the potential is exactly of Staeckel form, so this should sit near the orbits of interest; u0=0 is the symmetry axis and is degenerate for anything that is not exactly Staeckel.
        ntab : int, optional
            If set, tabulate the 1-D building blocks U(u), V(v) and their first and second derivatives on ntab-point grids at initialization and have the potential/force/Hessian routines (Python and C, which agree to machine precision) interpolate them (natural cubic splines) instead of evaluating the wrapped potential along the reference curves on every call; ~20x faster C orbit integration at spline accuracy (u covers [0, arcsinh(Rmax_tab/delta)], v covers [0, pi/2] with z-symmetry folding; u beyond the table is clamped). Default is None (exact evaluations).
        Rmax_tab : float, optional
            Cylindrical radius in the plane out to which the u table extends when ntab is set. Default is 100.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2017-12-15 - Started - Bovy (UofT)
        """
        self._delta = conversion.parse_length(delta, ro=ro)
        if u0 is None:
            # Place the reference curve at R=1, so that it tracks delta rather
            # than landing at an arbitrary radius; a fixed u0 means
            # R = delta sinh(u0), which drifts with delta
            u0 = numpy.arcsinh(1.0 / self._delta)
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
        self._ntab = 0 if ntab is None else int(ntab)
        # mode discriminator, mirroring spline1d != NULL in C: None = exact
        # (also while the tables below are being built from the exact
        # primitives), a list of six splines = interpolate
        self._splines = None
        if self._ntab:
            if self._ntab < 4:
                raise ValueError("ntab= must be at least 4")
            umax = float(numpy.arcsinh(Rmax_tab / self._delta))
            ugrid = numpy.linspace(0.0, umax, self._ntab)
            vgrid = numpy.linspace(0.0, numpy.pi / 2.0, self._ntab)
            # evaluate the v functions just off the axis (exact v=0 is 0/0
            # in dVdv's R0/tan v, whose limit is finite)
            veval = numpy.copy(vgrid)
            veval[0] = 1e-9
            uvals = [
                numpy.array([float(func(x)) for x in ugrid])
                for func in (self._U, self._dUdu, self._d2Udu2)
            ]
            vvals = [
                numpy.array([float(func(x)) for x in veval])
                for func in (self._V, self._dVdv, self._d2Vdv2)
            ]
            # raw (grid, values) tables; the C side builds natural cubic
            # splines from these with the house GSL 1D machinery at parse time
            tab = [float(self._ntab)]
            tab.extend(ugrid)
            for vals in uvals:
                tab.extend(vals)
            tab.extend(vgrid)
            for vals in vvals:
                tab.extend(vals)
            self._tabargs = tab
            # Python evaluation interpolates the same tables: a natural
            # CubicSpline is the same mathematical object as GSL's cspline on
            # the same knots, so Python and C agree to machine precision
            # also in tabulated mode
            self._splines = [
                CubicSpline(ugrid, vals, bc_type="natural") for vals in uvals
            ] + [CubicSpline(vgrid, vals, bc_type="natural") for vals in vvals]
        self.hasC = True
        # Advertise the (planar and 3D) C variational capabilities
        # unconditionally, as for hasC: _check_c recurses into the wrapped
        # potential's own flags (the wrapper's C Hessian chain-rules the
        # wrapped potential's forces and second derivatives along the U/V
        # reference curves through the prolate spheroidal coordinates, so it
        # is complete iff the wrapped potential's Hessian is in C).
        self.hasC_dxdv = True
        self.hasC_dxdv3d = True

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
        xp = get_namespace(R, z, phi, t)
        u, v = coords.Rz_to_uv(R, z, delta=self._delta)
        prefac = _staeckel_prefactor(u, v)
        dprefacdu, dprefacdv = _dstaeckel_prefactordudv(u, v)
        return (
            (
                -self._dUdu(u) * self._delta * xp.sin(v) * xp.cosh(u)
                + self._dVdv(v) * xp.tanh(u) * z
                + (self._U(u) - self._V(v))
                * (
                    dprefacdu * self._delta * xp.sin(v) * xp.cosh(u)
                    + dprefacdv * xp.tanh(u) * z
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
        xp = get_namespace(R, z, phi, t)
        u, v = coords.Rz_to_uv(R, z, delta=self._delta)
        prefac = _staeckel_prefactor(u, v)
        dprefacdu, dprefacdv = _dstaeckel_prefactordudv(u, v)
        return (
            (
                -self._dUdu(u) * R / xp.tan(v)
                - self._dVdv(v) * self._delta * xp.sin(v) * xp.cosh(u)
                + (self._U(u) - self._V(v))
                * (
                    dprefacdu / xp.tan(v) * R
                    - dprefacdv * self._delta * xp.sin(v) * xp.cosh(u)
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
        xp = get_namespace(R, z, phi, t)
        u, v = coords.Rz_to_uv(R, z, delta=self._delta)
        prefac = _staeckel_prefactor(u, v)
        dprefacdu, dprefacdv = _dstaeckel_prefactordudv(u, v)
        d2prefacdu2, d2prefacdv2 = _dstaeckel_prefactord2ud2v(u, v)
        umvfac = (
            dprefacdu * self._delta * xp.sin(v) * xp.cosh(u)
            + dprefacdv * xp.tanh(u) * z
        ) / prefac  # xs (U-V) in Rforce
        U = self._U(u)
        dUdu = self._dUdu(u)
        d2Udu2 = self._d2Udu2(u)
        V = self._V(v)
        dVdv = self._dVdv(v)
        d2Vdv2 = self._d2Vdv2(v)
        return (
            d2Udu2 * xp.sin(v) ** 2.0 * xp.cosh(u) ** 2.0
            + dUdu * xp.sinh(u) * xp.cosh(u)
            - d2Vdv2 * xp.sinh(u) ** 2.0 * xp.cos(v) ** 2.0
            - dVdv * xp.sin(v) * xp.cos(v)
            + (
                (-dUdu * xp.cosh(u) * xp.sin(v) + dVdv * xp.sinh(u) * xp.cos(v))
                / self._delta
                * umvfac
                + (U - V)
                * (
                    -d2prefacdu2 * xp.cosh(u) ** 2.0 * xp.sin(v) ** 2.0
                    - dprefacdu * xp.sinh(u) * xp.cosh(u)
                    - d2prefacdv2 * xp.sinh(u) ** 2.0 * xp.cos(v) ** 2.0
                    - dprefacdv * xp.sin(v) * xp.cos(v)
                )
                / prefac
                + (U - V)
                * umvfac
                / prefac
                / self._delta
                * (
                    dprefacdu * xp.cosh(u) * xp.sin(v)
                    + dprefacdv * xp.sinh(u) * xp.cos(v)
                )
            )
        ) / self._delta**2.0 / prefac**3.0 + 2.0 * self._Rforce(
            R, z, phi=phi, t=t
        ) / prefac**2.0 * (
            dprefacdu * xp.cosh(u) * xp.sin(v) + dprefacdv * xp.sinh(u) * xp.cos(v)
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
        xp = get_namespace(R, z, phi, t)
        u, v = coords.Rz_to_uv(R, z, delta=self._delta)
        prefac = _staeckel_prefactor(u, v)
        dprefacdu, dprefacdv = _dstaeckel_prefactordudv(u, v)
        d2prefacdu2, d2prefacdv2 = _dstaeckel_prefactord2ud2v(u, v)
        umvfac = (
            dprefacdu / xp.tan(v) * R  # xs (U-V) in zforce
            - dprefacdv * self._delta * xp.sin(v) * xp.cosh(u)
        ) / prefac
        U = self._U(u)
        dUdu = self._dUdu(u)
        d2Udu2 = self._d2Udu2(u)
        V = self._V(v)
        dVdv = self._dVdv(v)
        d2Vdv2 = self._d2Vdv2(v)
        return (
            d2Udu2 * xp.sinh(u) ** 2.0 * xp.cos(v) ** 2.0
            + dUdu * xp.cosh(u) * xp.sinh(u)
            - d2Vdv2 * xp.sin(v) ** 2.0 * xp.cosh(u) ** 2.0
            - dVdv * xp.cos(v) * xp.sin(v)
            + (
                (-dUdu * xp.sinh(u) * xp.cos(v) - dVdv * xp.cosh(u) * xp.sin(v))
                / self._delta
                * umvfac
                + (U - V)
                * (
                    -d2prefacdu2 * xp.sinh(u) ** 2.0 * xp.cos(v) ** 2.0
                    - dprefacdu * xp.sinh(u) * xp.cosh(u)
                    - d2prefacdv2 * xp.sin(v) ** 2.0 * xp.cosh(u) ** 2.0
                    - dprefacdv * xp.cos(v) * xp.sin(v)
                )
                / prefac
                - (U - V)
                * umvfac
                / prefac
                / self._delta
                * (
                    -dprefacdu * xp.sinh(u) * xp.cos(v)
                    + dprefacdv * xp.cosh(u) * xp.sin(v)
                )
            )
        ) / self._delta**2.0 / prefac**3.0 - 2.0 * self._zforce(
            R, z, phi=phi, t=t
        ) / prefac**2.0 * (
            -dprefacdu * xp.sinh(u) * xp.cos(v) + dprefacdv * xp.cosh(u) * xp.sin(v)
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
        xp = get_namespace(R, z, phi, t)
        u, v = coords.Rz_to_uv(R, z, delta=self._delta)
        prefac = _staeckel_prefactor(u, v)
        dprefacdu, dprefacdv = _dstaeckel_prefactordudv(u, v)
        d2prefacdu2, d2prefacdv2 = _dstaeckel_prefactord2ud2v(u, v)
        umvfac = (
            dprefacdu / xp.tan(v) * R  # xs (U-V) in zforce
            - dprefacdv * self._delta * xp.sin(v) * xp.cosh(u)
        ) / prefac
        U = self._U(u)
        dUdu = self._dUdu(u)
        d2Udu2 = self._d2Udu2(u)
        V = self._V(v)
        dVdv = self._dVdv(v)
        d2Vdv2 = self._d2Vdv2(v)
        return (
            (d2Udu2 + d2Vdv2) * xp.cosh(u) * xp.sin(v) * xp.cos(v) * xp.sinh(u)
            + dUdu * xp.sin(v) * xp.cos(v)
            + dVdv * xp.sinh(u) * xp.cosh(u)
            + (
                (-dUdu * xp.cosh(u) * xp.sin(v) + dVdv * xp.sinh(u) * xp.cos(v))
                / self._delta
                * umvfac
                + (U - V)
                * (
                    (-d2prefacdu2 + d2prefacdv2)
                    * xp.sin(v)
                    * xp.cosh(u)
                    * xp.sinh(u)
                    * xp.cos(v)
                    - dprefacdu * xp.sin(v) * xp.cos(v)
                    + dprefacdv * xp.cosh(u) * xp.sinh(u)
                )
                / prefac
                + (U - V)
                * umvfac
                / prefac
                / self._delta
                * (
                    dprefacdu * xp.cosh(u) * xp.sin(v)
                    + dprefacdv * xp.sinh(u) * xp.cos(v)
                )
            )
        ) / self._delta**2.0 / prefac**3.0 + 2.0 * self._zforce(
            R, z, phi=phi, t=t
        ) / prefac**2.0 * (
            dprefacdu * xp.cosh(u) * xp.sin(v) + dprefacdv * xp.sinh(u) * xp.cos(v)
        ) / self._delta

    def _evalUspline(self, u, k):
        # same semantics as evalUspline in C: u beyond the table is clamped
        return self._splines[k](numpy.clip(u, 0.0, self._splines[k].x[-1]))

    def _evalVspline(self, v, k):
        # same semantics as evalVspline in C: tables cover v in [0, pi/2];
        # V is even and V' odd about pi/2 (z-symmetry)
        sgn = 1.0
        if v > numpy.pi / 2.0:
            v = numpy.pi - v
            if k == 1:
                sgn = -1.0
        return sgn * self._splines[3 + k](numpy.clip(v, 0.0, numpy.pi / 2.0))

    def _U(self, u):
        """Approximated U(u) = cosh^2(u) Phi(u,pi/2)"""
        xp = get_namespace(u)
        # The tabulated spline (ntab=) is a scipy CubicSpline: numpy-only and
        # non-differentiable, so it serves the numpy path only. A backend keeps
        # the analytic evaluation below, which differentiates.
        if xp is numpy and self._splines is not None:
            return self._evalUspline(u, 0)
        Rz0 = coords.uv_to_Rz(u, self._v0, delta=self._delta)
        return xp.cosh(u) ** 2.0 * _evaluatePotentials(self._pot, Rz0[0], Rz0[1])

    def _dUdu(self, u):
        xp = get_namespace(u)
        # The tabulated spline (ntab=) is a scipy CubicSpline: numpy-only and
        # non-differentiable, so it serves the numpy path only. A backend keeps
        # the analytic evaluation below, which differentiates.
        if xp is numpy and self._splines is not None:
            return self._evalUspline(u, 1)
        Rz0 = coords.uv_to_Rz(u, self._v0, delta=self._delta)
        # 1e-12 bc force should win the 0/0 battle
        return 2.0 * xp.cosh(u) * xp.sinh(u) * _evaluatePotentials(
            self._pot, Rz0[0], Rz0[1]
        ) - xp.cosh(u) ** 2.0 * (
            _evaluateRforces(self._pot, Rz0[0], Rz0[1]) * Rz0[0] / (xp.tanh(u) + 1e-12)
            + _evaluatezforces(self._pot, Rz0[0], Rz0[1]) * Rz0[1] * xp.tanh(u)
        )

    def _d2Udu2(self, u):
        xp = get_namespace(u)
        # The tabulated spline (ntab=) is a scipy CubicSpline: numpy-only and
        # non-differentiable, so it serves the numpy path only. A backend keeps
        # the analytic evaluation below, which differentiates.
        if xp is numpy and self._splines is not None:
            return self._evalUspline(u, 2)
        Rz0 = coords.uv_to_Rz(u, self._v0, delta=self._delta)
        tRforce = _evaluateRforces(self._pot, Rz0[0], Rz0[1])
        tzforce = _evaluatezforces(self._pot, Rz0[0], Rz0[1])
        return (
            2.0 * xp.cosh(2 * u) * _evaluatePotentials(self._pot, Rz0[0], Rz0[1])
            - 4.0
            * xp.cosh(u)
            * xp.sinh(u)
            * (tRforce * Rz0[0] / (xp.tanh(u) + 1e-12) + tzforce * Rz0[1] * xp.tanh(u))
            - xp.cosh(u) ** 2.0
            * (
                -evaluateR2derivs(self._pot, Rz0[0], Rz0[1], use_physical=False)
                * Rz0[0] ** 2.0
                / (xp.tanh(u) + 1e-12) ** 2.0
                - 2.0
                * evaluateRzderivs(self._pot, Rz0[0], Rz0[1], use_physical=False)
                * Rz0[0]
                * Rz0[1]
                + tRforce * Rz0[0]
                - evaluatez2derivs(self._pot, Rz0[0], Rz0[1], use_physical=False)
                * Rz0[1] ** 2.0
                * xp.tanh(u) ** 2.0
                + tzforce * Rz0[1]
            )
        )

    def _V(self, v):
        """Approximated
        V(v) = cosh^2(u0) Phi(u0,pi/2) - (sinh^2(u0)+sin^2(v)) Phi(u0,v)"""
        if self._splines is not None:
            return self._evalVspline(v, 0)
        R0z = coords.uv_to_Rz(self._u0, v, delta=self._delta)
        return self._refpot - _staeckel_prefactor(self._u0, v) * _evaluatePotentials(
            self._pot, R0z[0], R0z[1]
        )

    def _dVdv(self, v):
        xp = get_namespace(v)
        # The tabulated spline (ntab=) is a scipy CubicSpline: numpy-only and
        # non-differentiable, so it serves the numpy path only. A backend keeps
        # the analytic evaluation below, which differentiates.
        if xp is numpy and self._splines is not None:
            return self._evalVspline(v, 1)
        R0z = coords.uv_to_Rz(self._u0, v, delta=self._delta)
        return -2.0 * xp.sin(v) * xp.cos(v) * _evaluatePotentials(
            self._pot, R0z[0], R0z[1]
        ) + _staeckel_prefactor(self._u0, v) * (
            _evaluateRforces(self._pot, R0z[0], R0z[1]) * R0z[0] / xp.tan(v)
            - _evaluatezforces(self._pot, R0z[0], R0z[1]) * R0z[1] * xp.tan(v)
        )

    def _d2Vdv2(self, v):
        xp = get_namespace(v)
        # The tabulated spline (ntab=) is a scipy CubicSpline: numpy-only and
        # non-differentiable, so it serves the numpy path only. A backend keeps
        # the analytic evaluation below, which differentiates.
        if xp is numpy and self._splines is not None:
            return self._evalVspline(v, 2)
        R0z = coords.uv_to_Rz(self._u0, v, delta=self._delta)
        tRforce = _evaluateRforces(self._pot, R0z[0], R0z[1])
        tzforce = _evaluatezforces(self._pot, R0z[0], R0z[1])
        return (
            -2.0 * xp.cos(2.0 * v) * _evaluatePotentials(self._pot, R0z[0], R0z[1])
            + 2.0
            * xp.sin(2.0 * v)
            * (tRforce * R0z[0] / xp.tan(v) - tzforce * R0z[1] * xp.tan(v))
            + _staeckel_prefactor(self._u0, v)
            * (
                -evaluateR2derivs(self._pot, R0z[0], R0z[1], use_physical=False)
                * R0z[0] ** 2.0
                / xp.tan(v) ** 2.0
                + 2.0
                * evaluateRzderivs(self._pot, R0z[0], R0z[1], use_physical=False)
                * R0z[0]
                * R0z[1]
                - tRforce * R0z[0]
                - evaluatez2derivs(self._pot, R0z[0], R0z[1], use_physical=False)
                * R0z[1] ** 2.0
                * xp.tan(v) ** 2.0
                - tzforce * R0z[1]
            )
        )


def _staeckel_prefactor(u, v):
    xp = get_namespace(u, v)
    u, v = _promote_scalars_for(xp, u, v)
    return xp.sinh(u) ** 2.0 + xp.sin(v) ** 2.0


def _dstaeckel_prefactordudv(u, v):
    xp = get_namespace(u, v)
    u, v = _promote_scalars_for(xp, u, v)
    return (2.0 * xp.sinh(u) * xp.cosh(u), 2.0 * xp.sin(v) * xp.cos(v))


def _dstaeckel_prefactord2ud2v(u, v):
    xp = get_namespace(u, v)
    u, v = _promote_scalars_for(xp, u, v)
    return (2.0 * xp.cosh(2.0 * u), 2.0 * xp.cos(2.0 * v))
