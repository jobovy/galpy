###############################################################################
#   SphericalShellPotential.py: The gravitational potential of a thin,
#                               spherical shell
###############################################################################
import math

from ..backend import get_namespace
from ..util import conversion
from .SphericalPotential import SphericalPotential


class SphericalShellPotential(SphericalPotential):
    """Class that implements the potential of an infinitesimally-thin, spherical shell

    .. math::

        \\rho(r) = \\frac{\\mathrm{amp}}{4\\pi\\,a^2}\\,\\delta(r-a)

    with :math:`\\mathrm{amp} = GM` the mass of the shell.
    """

    def __init__(self, amp=1.0, a=0.75, normalize=False, ro=None, vo=None):
        """
        Initialize a spherical shell potential.

        Parameters
        ----------
        amp : float or Quantity, optional
            Mass of the shell (default: 1); can be a Quantity with units of mass or Gxmass.
        a : float or Quantity, optional
            Radius of the shell (default: 0.75).
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.; note that because the force is always zero at r < a, this does not work if a > 1.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2018-08-04 - Written - Bovy (UofT)
        - 2020-03-30 - Re-implemented using SphericalPotential - Bovy (UofT)

        """
        SphericalPotential.__init__(self, amp=amp, ro=ro, vo=vo, amp_units="mass")
        a = conversion.parse_length(a, ro=self._ro)
        self.a = a
        self.a2 = a**2
        if normalize or (
            isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
        ):
            if self.a > 1.0:
                raise ValueError(
                    "SphericalShellPotential with normalize= for a > 1 is not supported (because the force is always 0 at r=1)"
                )
            self.normalize(normalize)
        self.hasC = False
        self.hasC_dxdv = False

    def _revaluate(self, r, t=0.0):
        """The potential as a function of r"""
        xp = get_namespace(r)
        inside = r <= self.a
        # safe r so the (dead) outside branch cannot produce -1/0 at r == 0
        safe = xp.where(inside, xp.ones_like(r * 1.0), r)
        return xp.where(inside, -1.0 / self.a * xp.ones_like(r * 1.0), -1.0 / safe)

    def _rforce(self, r, t=0.0):
        """The force as a function of r"""
        xp = get_namespace(r)
        inside = r <= self.a
        safe = xp.where(inside, xp.ones_like(r * 1.0), r)
        return xp.where(inside, xp.zeros_like(r * 1.0), -1 / safe**2.0)

    def _r2deriv(self, r, t=0.0):
        """The second radial derivative as a function of r"""
        xp = get_namespace(r)
        inside = r <= self.a
        safe = xp.where(inside, xp.ones_like(r * 1.0), r)
        return xp.where(inside, xp.zeros_like(r * 1.0), -2.0 / safe**3.0)

    def _rdens(self, r, t=0.0):
        """The density as a function of r"""
        xp = get_namespace(r)
        return xp.where(
            r != self.a, xp.zeros_like(r * 1.0), math.inf * xp.ones_like(r * 1.0)
        )

    def _surfdens(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        # h is real only where R <= a; use a safe argument so the dead
        # (R > a) branch cannot produce sqrt(negative) = NaN
        outside = R > self.a
        safe_arg = xp.where(outside, xp.ones_like(R * 1.0), self.a2 - R**2)
        h = xp.sqrt(safe_arg)
        safe_h = xp.where(h == 0.0, xp.ones_like(h), h)
        val = 1.0 / (2.0 * math.pi * self.a * safe_h)
        zero = xp.zeros_like((R + z) * 1.0)
        # zero for R > a or z < h, else val
        return xp.where(outside | (z < h), zero, val)
