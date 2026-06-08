###############################################################################
#   BurkertPotential.py: Potential with a Burkert density
###############################################################################
import math

from ..backend import get_namespace
from ..util import conversion
from .SphericalPotential import SphericalPotential


class BurkertPotential(SphericalPotential):
    """BurkertPotential.py: Potential with a Burkert density

    .. math::

        \\rho(r) = \\frac{\\mathrm{amp}}{(1+r/a)\\,(1+[r/a]^2)}

    """

    def __init__(self, amp=1.0, a=2.0, normalize=False, ro=None, vo=None):
        """
        Initialize a Burkert-density potential [1]_.

        Parameters
        ----------
        amp : float or Quantity
            Amplitude to be applied to the potential. Can be a Quantity with units of mass density or Gxmass density.
        a : float or Quantity
            Scale radius.
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1. Default is False.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2013-04-10 - Written - Bovy (IAS)
        - 2020-03-30 - Re-implemented using SphericalPotential - Bovy (UofT)

        References
        ----------
        .. [1] Burkert (1995), Astrophysical Journal, 447, L25. ADS: https://ui.adsabs.harvard.edu/abs/1995ApJ...447L..25B.
        """
        SphericalPotential.__init__(self, amp=amp, ro=ro, vo=vo, amp_units="density")
        a = conversion.parse_length(a, ro=self._ro, vo=self._vo)
        self.a = a
        self._scale = self.a
        if normalize or (
            isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
        ):  # pragma: no cover
            self.normalize(normalize)
        self.hasC = True
        self.hasC_dxdv = True
        self.hasC_dxdv3d = True  # full 3D Hessian (R2deriv/z2deriv/Rzderiv) in C
        self.hasC_dens = True
        return None

    def _revaluate(self, r, t=0.0):
        """Potential as a function of r and time"""
        xp = get_namespace(r)
        x = r / self.a
        # special.xlogy(2/x, 1+x**2) == (2/x)*log(1+x**2), but with the convention
        # that it is 0 where the prefactor is 0 (i.e. as x -> infty, where the bare
        # product would be 0*inf = NaN). Reproduce that backend-agnostically: the
        # prefactor 2/x only vanishes at x == infty, so guard exactly that point.
        pref = 2.0 / x
        # safe argument so the (dead) finite branch cannot make log(inf) at x==inf
        safe_x2 = xp.where(xp.isinf(x), xp.ones_like(x * 1.0), x**2.0)
        xlogy_term = xp.where(
            xp.isinf(x),
            xp.zeros_like(x * 1.0),
            pref * xp.log(1.0 + safe_x2),
        )
        return (
            -(self.a**2.0)
            * math.pi
            * (
                -math.pi / x
                + 2.0 * (1.0 / x + 1) * xp.arctan(1 / x)
                + (1.0 / x + 1) * xp.log((1.0 + 1.0 / x) ** 2.0 / (1.0 + 1 / x**2.0))
                + xlogy_term
            )
        )

    # Previous way, not stable as r -> infty
    # return -self.a**2.*numpy.pi/x*(-numpy.pi+2.*(1.+x)*numpy.arctan(1/x)
    #                                +2.*(1.+x)*numpy.log(1.+x)
    #                                +(1.-x)*numpy.log(1.+x**2.))

    def _rforce(self, r, t=0.0):
        xp = get_namespace(r)
        x = r / self.a
        return (
            self.a
            * math.pi
            / x**2.0
            * (
                math.pi
                - 2.0 * xp.arctan(1.0 / x)
                - 2.0 * xp.log(1.0 + x)
                - xp.log(1.0 + x**2.0)
            )
        )

    def _r2deriv(self, r, t=0.0):
        x = r / self.a
        return (
            4.0 * math.pi / (1.0 + x**2.0) / (1.0 + x)
            + 2.0 * self._rforce(r) / x / self.a
        )

    def _rdens(self, r, t=0.0):
        x = r / self.a
        return 1.0 / (1.0 + x) / (1.0 + x**2.0)

    def _surfdens(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        r = xp.sqrt(R**2.0 + z**2.0)
        x = r / self.a
        Rpa = xp.sqrt(R**2.0 + self.a**2.0)
        # R == a is a removable singularity of the generic (Rma != 0) branch:
        # there Rma -> 0 and the arctan/Rma terms blow up, so we use a separate
        # closed-form limit. xp.where evaluates BOTH branches, so the generic
        # branch must stay NaN-free at the edge: build Rma from a safe argument
        # that is never zero there (so 1/Rma, arctan(z/x/Rma) etc. stay finite).
        at_edge = R == self.a
        d2 = R**2.0 - self.a**2.0
        safe_d2 = xp.where(at_edge, xp.ones_like(d2 * 1.0), d2)
        Rma = xp.sqrt(xp.astype(safe_d2, xp.complex128))
        # Edge (R == a) branch. It carries a 1/z that is singular at z == 0;
        # under the eager xp.where it is evaluated everywhere, so in its DEAD
        # region (generic, R != a) it must use a finite z or it NaN-poisons
        # reverse-mode gradients at z == 0 (a valid, finite input where surfdens
        # == 0). Use the real z on the edge (live) and 1 in the generic region
        # (dead). At the genuine edge point R == a, z == 0 the limb singularity
        # is real (kept).
        z_edge = xp.where(at_edge, z, xp.ones_like(z * 1.0))
        za = z_edge / self.a
        edge = (
            self.a**2.0
            / 2.0
            * (
                (
                    2.0
                    - 2.0 * xp.sqrt(za**2.0 + 1)
                    + 2.0**0.5 * za * xp.arctan(za / 2.0**0.5)
                )
                / z_edge
                + xp.sqrt(2 * za**2.0 + 2.0)
                * xp.arctanh(za / xp.sqrt(2.0 * (za**2.0 + 1)))
                / xp.sqrt(self.a**2.0 + z_edge**2.0)
            )
        )
        # Generic (R != a) branch; .real of the complex combination
        generic = self.a**2.0 * xp.real(
            xp.arctan(z / x / Rma) / Rma
            + xp.arctanh(z / x / Rpa) / Rpa
            - xp.arctan(z / Rma) / Rma
            + xp.arctan(z / Rpa) / Rpa
        )
        return xp.where(at_edge, edge, generic)
