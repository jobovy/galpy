###############################################################################
#   CorotatingRotationWrapperPotential.py: Wrapper to make a potential rotate
#                                          with a fixed R x pattern speed,
#                                          around the z axis
###############################################################################
from ..util import conversion
from .WrapperPotential import parentWrapperPotential


class CorotatingRotationWrapperPotential(parentWrapperPotential):
    """Potential wrapper class that implements rotation with fixed R x pattern-speed around the z-axis. Can be used to make spiral structure that is everywhere co-rotating. The potential is rotated by replacing

    .. math::

        \\phi \\rightarrow \\phi + \\frac{V_p(R)}{R} \\times \\left(t-t_0\\right) + \\mathrm{pa}

    with :math:`V_p(R)` the circular velocity curve, :math:`t_0` a reference time---time at which the potential is unchanged by the wrapper---and :math:`\\mathrm{pa}` the position angle at :math:`t=0`. The circular velocity is parameterized as

    .. math::

       V_p(R) = V_{p,0}\\,\\left(\\frac{R}{R_0}\\right)^\\beta\\,.

    """

    def __init__(
        self, amp=1.0, pot=None, vpo=1.0, beta=0.0, to=0.0, pa=0.0, ro=None, vo=None
    ):
        """
        Initialize a CorotatingRotationWrapper Potential.

        Parameters
        ----------
        amp : float, optional
            Amplitude to be applied to the potential (default: 1.).
        pot : Potential instance or list thereof, optional
            This potential is made to rotate around the z axis by the wrapper.
        vpo : float or Quantity, optional
            Amplitude of the circular-velocity curve.
        beta : float, optional
            Power-law amplitude of the circular-velocity curve.
        to : float or Quantity, optional
            Reference time at which the potential == pot.
        pa : float or Quantity, optional
            The position angle.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2018-02-21 - Started - Bovy (UofT)

        """
        vpo = conversion.parse_velocity(vpo, vo=self._vo)
        to = conversion.parse_time(to, ro=self._ro, vo=self._vo)
        pa = conversion.parse_angle(pa)
        self._vpo = vpo
        self._beta = beta
        self._pa = pa
        self._to = to
        self.hasC = True
        self.hasC_dxdv = True

    def _wrap(self, attribute, *args, **kwargs):
        kwargs["phi"] = (
            kwargs.get("phi", 0.0)
            - self._vpo
            * args[0] ** (self._beta - 1.0)
            * (kwargs.get("t", 0.0) - self._to)
            - self._pa
        )
        return self._wrap_pot_func(attribute)(self._pot, *args, **kwargs)

    # Derivatives that involve R need to be adjusted, bc they require also
    # the R dependence of phi to be taken into account
    def _Rforce(self, *args, **kwargs):
        kwargs["phi"] = (
            kwargs.get("phi", 0.0)
            - self._vpo
            * args[0] ** (self._beta - 1.0)
            * (kwargs.get("t", 0.0) - self._to)
            - self._pa
        )
        return self._wrap_pot_func("_Rforce")(
            self._pot, *args, **kwargs
        ) - self._wrap_pot_func("_phitorque")(self._pot, *args, **kwargs) * (
            self._vpo
            * (self._beta - 1.0)
            * args[0] ** (self._beta - 2.0)
            * (kwargs.get("t", 0.0) - self._to)
        )

    def _R2deriv(self, *args, **kwargs):
        kwargs["phi"] = (
            kwargs.get("phi", 0.0)
            - self._vpo
            * args[0] ** (self._beta - 1.0)
            * (kwargs.get("t", 0.0) - self._to)
            - self._pa
        )
        phiRderiv = (
            -self._vpo
            * (self._beta - 1.0)
            * args[0] ** (self._beta - 2.0)
            * (kwargs.get("t", 0.0) - self._to)
        )
        return (
            self._wrap_pot_func("_R2deriv")(self._pot, *args, **kwargs)
            + 2.0
            * self._wrap_pot_func("_Rphideriv")(self._pot, *args, **kwargs)
            * phiRderiv
            + self._wrap_pot_func("_phi2deriv")(self._pot, *args, **kwargs)
            * phiRderiv**2.0
            + self._wrap_pot_func("_phitorque")(self._pot, *args, **kwargs)
            * (
                self._vpo
                * (self._beta - 1.0)
                * (self._beta - 2.0)
                * args[0] ** (self._beta - 3.0)
                * (kwargs.get("t", 0.0) - self._to)
            )
        )

    def _Rphideriv(self, *args, **kwargs):
        kwargs["phi"] = (
            kwargs.get("phi", 0.0)
            - self._vpo
            * args[0] ** (self._beta - 1.0)
            * (kwargs.get("t", 0.0) - self._to)
            - self._pa
        )
        return self._wrap_pot_func("_Rphideriv")(
            self._pot, *args, **kwargs
        ) - self._wrap_pot_func("_phi2deriv")(
            self._pot, *args, **kwargs
        ) * self._vpo * (self._beta - 1.0) * args[0] ** (self._beta - 2.0) * (
            kwargs.get("t", 0.0) - self._to
        )
