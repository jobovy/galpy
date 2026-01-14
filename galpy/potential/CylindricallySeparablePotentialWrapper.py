###############################################################################
#   CylindricallySeparablePotentialWrapper.py: Wrapper to turn an axisymmetric
#                                              potential into a potential
#                                              that is separable in cylindrical
#                                              coordinates
#
#   NOT A TYPICAL WRAPPER, SO DON'T USE THIS BLINDLY AS A TEMPLATE FOR NEW
#   WRAPPERS
#
###############################################################################
from galpy.util import conversion

from .Potential import (
    _APY_LOADED,
    _evaluatePotentials,
    _evaluateRforces,
    _evaluatezforces,
    evaluateR2derivs,
    evaluatez2derivs,
)
from .WrapperPotential import parentWrapperPotential


class CylindricallySeparablePotentialWrapper(parentWrapperPotential):
    r"""Potential wrapper class that approximates a given axisymmetric potential as a potential that is separable in cylindrical coordinates, by defining

    .. math::
        :nowrap:

        \begin{align}
        \Phi_R(R) & = \Phi(R,0)\,,\\
        \Phi_z(z) & = \Phi(R',z) - \Phi(R',0)\,,
        \end{align}

    where :math:`R'` is a reference radius (default: :math:`R'=1` in internal units). The potential is then given by

    .. math::
        :nowrap:

        \begin{align}
        \Phi(R,z) & = \Phi_R(R) + \Phi_z(z)\,.
        \end{align}

    This approximation is used in the adiabatic-approximation for action-angle coordinates (e.g., `Binney 2010 <https://ui.adsabs.harvard.edu/abs/2010MNRAS.401.2318B/abstract>`__; `Bovy 2026 <https://galaxiesbook.org/chapters/II-03.-Orbits-in-Disks_4-Action-angle-coordinates-in-and-around-disks.html#The-adiabatic-approximation>`__).
    """

    def __init__(self, amp=1.0, pot=None, Rp=1.0, ro=None, vo=None):
        """Initialize an CylindricallySeparablePotentialWrapper Potential.

        Parameters
        ----------
        amp : float, optional
            Amplitude to be applied to the potential. Default is 1.0.
        pot : Potential or a combined potential formed using addition (pot1+pot2+…)
            Potential instance or a combined potential formed using addition (pot1+pot2+…); this potential is made into a cylindrically separable potential.
        Rp : float or Quantity, optional
            Reference radius :math:`R'` (in internal units) at which to evaluate :math:`\\Phi(R',z)`; default is 1.0 (in internal units).
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2026-01-14 - Started - Bovy (UofT)
        """
        if Rp is None:  # pragma: no cover
            raise ValueError(
                "Rp= needs to be given to setup CylindricallySeparablePotentialWrapper"
            )
        self._Rp = conversion.parse_length(Rp, ro=ro)
        self._refpot = _evaluatePotentials(self._pot, self._Rp, 0.0)
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
           2026-01-14 - Written - Bovy (UofT)
        """
        return (
            _evaluatePotentials(self._pot, R, 0.0)
            + _evaluatePotentials(self._pot, self._Rp, z)
            - self._refpot
        )

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
           2026-01-14 - Written - Bovy (UofT)
        """
        return _evaluateRforces(self._pot, R, 0.0)

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
           2026-01-14 - Written - Bovy (UofT)
        """
        return _evaluatezforces(self._pot, self._Rp, z)

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
           2026-01-14 - Written - Bovy (UofT)
        """
        return evaluateR2derivs(self._pot, R, 0.0, use_physical=False)

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
           2026-01-14 - Written - Bovy (UofT)
        """
        return evaluatez2derivs(self._pot, self._Rp, z, use_physical=False)

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
           2026-01-14 - Written - Bovy (UofT)
        """
        return 0.0
