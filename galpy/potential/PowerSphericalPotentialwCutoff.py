###############################################################################
#   PowerSphericalPotentialwCutoff.py: spherical power-law potential w/ cutoff
#
#                                     amp
#                          rho(r)= ---------   e^{-(r/rc)^2}
#                                   r^\alpha
###############################################################################
import numpy
from scipy import special

from ..backend import get_namespace
from ..backend.special import gamma as _gamma
from ..backend.special import gammainc as _gammainc
from ..util import conversion
from ..util._optional_deps import _JAX_LOADED
from .Potential import Potential, kms_to_kpcGyrDecorator

if _JAX_LOADED:
    import jax.numpy as jnp
    import jax.scipy.special as jspecial


class PowerSphericalPotentialwCutoff(Potential):
    """Class that implements spherical potentials that are derived from
    power-law density models

    .. math::

        \\rho(r) = \\mathrm{amp}\\,\\left(\\frac{r_1}{r}\\right)^\\alpha\\,\\exp\\left(-(r/rc)^2\\right)

    """

    def __init__(
        self, amp=1.0, alpha=1.0, rc=1.0, normalize=False, r1=1.0, ro=None, vo=None
    ):
        """
        Initialize a power-law-density potential.

        Parameters
        ----------
        amp : float or Quantity, optional
            Amplitude to be applied to the potential. Can be a Quantity with units of mass density or Gxmass density.
        alpha : float, optional
            Inner power.
        rc : float or Quantity, optional
            Cut-off radius.
        r1 : float or Quantity, optional
            Reference radius for amplitude. Default is 1.0. Can be Quantity.
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.
        ro : float, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2013-06-28 - Written - Bovy (IAS)
        """
        Potential.__init__(self, amp=amp, ro=ro, vo=vo, amp_units="density")
        r1 = conversion.parse_length(r1, ro=self._ro)
        rc = conversion.parse_length(rc, ro=self._ro)
        self.alpha = alpha
        # Back to old definition
        self._amp *= r1**self.alpha
        self.rc = rc
        self._scale = self.rc
        if normalize or (
            isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
        ):  # pragma: no cover
            self.normalize(normalize)
        self.hasC = True
        self.hasC_dxdv = True
        self.hasC_dxdv3d = True  # full 3D Hessian (R2deriv/z2deriv/Rzderiv) in C
        self.hasC_dens = True
        self._nemo_accname = "PowSphwCut"

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        r = xp.sqrt(R**2.0 + z**2.0)
        # guard r=0 in the dead branch (the 1/r term -> 0/0=NaN there) so the
        # xp.where stays finite under autodiff/jit; the value at r=0 is 0.
        rsafe = xp.where(r == 0, 1.0, r)
        out = (
            2.0
            * numpy.pi
            * self.rc ** (3.0 - self.alpha)
            * (
                1
                / self.rc
                * _gamma(1.0 - self.alpha / 2.0)
                * _gammainc(1.0 - self.alpha / 2.0, (rsafe / self.rc) ** 2.0)
                - _gamma(1.5 - self.alpha / 2.0)
                * _gammainc(1.5 - self.alpha / 2.0, (rsafe / self.rc) ** 2.0)
                / rsafe
            )
        )
        return xp.where(r == 0, 0.0, out)

    def _rforce(self, r):
        # Radial force (amp-free): -M(<r)/r^2. The enclosed mass is written with
        # the regularized lower incomplete gamma P(a,x)=gammainc(a,x), so the
        # whole force path is backend-agnostic and jit-clean -- gammainc and
        # gamma are native (and reliable) on numpy/jax/torch, unlike the
        # hyp1f1 form of _mass (which only the numpy mass API still uses).
        return (
            -2.0
            * numpy.pi
            * self.rc ** (3.0 - self.alpha)
            * _gamma(1.5 - 0.5 * self.alpha)
            * _gammainc(1.5 - 0.5 * self.alpha, (r / self.rc) ** 2.0)
            / r**2.0
        )

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        r = xp.sqrt(R * R + z * z)
        return self._rforce(r) * R / r

    def _zforce(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        r = xp.sqrt(R * R + z * z)
        return self._rforce(r) * z / r

    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        r = xp.sqrt(R * R + z * z)
        return 4.0 * numpy.pi * r ** (-2.0 - self.alpha) * xp.exp(
            -((r / self.rc) ** 2.0)
        ) * R**2.0 + self._mass(r) / r**5.0 * (z**2.0 - 2.0 * R**2.0)

    def _z2deriv(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        r = xp.sqrt(R * R + z * z)
        return 4.0 * numpy.pi * r ** (-2.0 - self.alpha) * xp.exp(
            -((r / self.rc) ** 2.0)
        ) * z**2.0 + self._mass(r) / r**5.0 * (R**2.0 - 2.0 * z**2.0)

    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        r = xp.sqrt(R * R + z * z)
        return (
            R
            * z
            * (
                4.0
                * numpy.pi
                * r ** (-2.0 - self.alpha)
                * xp.exp(-((r / self.rc) ** 2.0))
                - 3.0 * self._mass(r) / r**5.0
            )
        )

    def _rforce_jax(self, r):
        if not _JAX_LOADED:  # pragma: no cover
            raise ImportError(
                "Making use of the _rforce_jax function requires the google/jax library"
            )
        return (
            -self._amp
            * 2.0
            * numpy.pi
            * self.rc ** (3.0 - self.alpha)
            * jspecial.gammainc(1.5 - 0.5 * self.alpha, (r / self.rc) ** 2.0)
            * numpy.exp(jspecial.gammaln(1.5 - 0.5 * self.alpha))
            / r**2
        )

    def _ddensdr(self, r, t=0.0):
        xp = get_namespace(r)
        return (
            -self._amp
            * r ** (-1.0 - self.alpha)
            * xp.exp(-((r / self.rc) ** 2.0))
            * (2.0 * r**2.0 / self.rc**2.0 + self.alpha)
        )

    def _d2densdr2(self, r, t=0.0):
        xp = get_namespace(r)
        return (
            self._amp
            * r ** (-2.0 - self.alpha)
            * xp.exp(-((r / self.rc) ** 2))
            * (
                self.alpha**2.0
                + self.alpha
                + 4 * self.alpha * r**2.0 / self.rc**2.0
                - 2.0 * r**2.0 / self.rc**2.0
                + 4.0 * r**4.0 / self.rc**4.0
            )
        )

    def _ddenstwobetadr(self, r, beta=0):
        """
        Evaluate the radial density derivative x r^(2beta) for this potential.

        Parameters
        ----------
        r : float
            Spherical radius.
        beta : int, optional
            Power of r in the density derivative. Default is 0.

        Returns
        -------
        float
            The derivative of rho x r^{2beta} / d r.

        Notes
        -----
        - 2021-03-15 - Written - Lane (UofT)
        """
        if not _JAX_LOADED:  # pragma: no cover
            raise ImportError(
                "Making use of _rforce_jax function requires the google/jax library"
            )
        return (
            -self._amp
            * jnp.exp(-((r / self.rc) ** 2.0))
            / r ** (self.alpha - 2.0 * beta)
            * ((self.alpha - 2.0 * beta) / r + 2.0 * r / self.rc**2.0)
        )

    def _dens(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        r = xp.sqrt(R**2.0 + z**2.0)
        return 1.0 / r**self.alpha * xp.exp(-((r / self.rc) ** 2.0))

    def _mass(self, R, z=None, t=0.0):
        if z is not None:
            raise AttributeError  # use general implementation
        # M(<R) via the regularized lower incomplete gamma P(a,x)=gammainc(a,x).
        # This is backend-agnostic/jit-clean and needs no separate R=inf branch:
        # gammainc(a, inf)=1 recovers the total mass automatically. (The previous
        # hyp1f1 form is an equivalent representation; values agree to ~1e-15.)
        return (
            2.0
            * numpy.pi
            * self.rc ** (3.0 - self.alpha)
            * _gamma(1.5 - 0.5 * self.alpha)
            * _gammainc(1.5 - 0.5 * self.alpha, (R / self.rc) ** 2.0)
        )

    @kms_to_kpcGyrDecorator
    def _nemo_accpars(self, vo, ro):
        ampl = self._amp * vo**2.0 * ro ** (self.alpha - 2.0)
        return f"0,{ampl},{self.alpha},{self.rc * ro}"
