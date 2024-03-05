###############################################################################
#   PowerSphericalPotential.py: General class for potentials derived from
#                               densities with two power-laws
#
#                                     amp
#                          rho(r)= ---------
#                                   r^\alpha
###############################################################################
import numpy
from scipy import special

from ..util import conversion
from .Potential import Potential


class PowerSphericalPotential(Potential):
    """Class that implements spherical potentials that are derived from power-law density models

    .. math::

        \\rho(r) = \\frac{\\mathrm{amp}}{r_1^3}\\,\\left(\\frac{r_1}{r}\\right)^{\\alpha}

    """

    def __init__(self, amp=1.0, alpha=1.0, normalize=False, r1=1.0, ro=None, vo=None):
        """
        Initialize a power-law-density potential

        Parameters
        ----------
        amp : float, optional
            Amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass or Gxmass
        alpha : float, optional
            Power-law exponent
        r1 : float, optional
            Reference radius for amplitude (can be Quantity)
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.
        ro : float, optional
            Distance scale for translation into internal units (default from configuration file)
        vo : float, optional
            Velocity scale for translation into internal units (default from configuration file)

        Notes
        -----
        - 2010-07-10 - Written - Bovy (NYU)
        """
        Potential.__init__(self, amp=amp, ro=ro, vo=vo, amp_units="mass")
        r1 = conversion.parse_length(r1, ro=self._ro)
        self.alpha = alpha
        # Back to old definition
        if self.alpha != 3.0:
            self._amp *= r1 ** (self.alpha - 3.0) * 4.0 * numpy.pi / (3.0 - self.alpha)
        if normalize or (
            isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
        ):
            self.normalize(normalize)
        self.hasC = True
        self.hasC_dxdv = True
        self.hasC_dens = True

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        """
        Evaluate the potential at R,z

        Parameters
        ----------
        R : float
            Galactocentric cylindrical radius
        z : float
            Vertical height
        phi : float, optional
            Azimuth (default: 0.0)
        t : float, optional
            Time (default: 0.0)

        Returns
        -------
        float
            Potential at (R, z)

        Notes
        -----
        - Started: 2010-07-10 by Bovy (NYU)
        """
        r2 = R**2.0 + z**2.0
        if self.alpha == 2.0:
            return numpy.log(r2) / 2.0
        elif isinstance(r2, (float, int)) and r2 == 0 and self.alpha > 2:
            return -numpy.inf
        else:
            out = -(r2 ** (1.0 - self.alpha / 2.0)) / (self.alpha - 2.0)
            if isinstance(r2, numpy.ndarray) and self.alpha > 2:
                out[r2 == 0] = -numpy.inf
            return out

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        """
        Evaluate the radial force for this potential.

        Parameters
        ----------
        R : float
            Galactocentric cylindrical radius
        z : float
            Vertical height
        phi : float, optional
            Azimuth (default: 0.0)
        t : float, optional
            Time (default: 0.0)

        Returns
        -------
        float
            The radial force.

        Notes
        -----
        - 2010-07-10 - Written - Bovy (NYU)
        """
        return -R / (R**2.0 + z**2.0) ** (self.alpha / 2.0)

    def _zforce(self, R, z, phi=0.0, t=0.0):
        """
        Evaluate the vertical force for this potential.

        Parameters
        ----------
        R : float
            Galactocentric cylindrical radius
        z : float
            Vertical height
        phi : float, optional
            Azimuth (default: 0.0)
        t : float, optional
            Time (default: 0.0)

        Returns
        -------
        float
            The vertical force.

        Notes
        -----
        - 2010-07-10 - Written - Bovy (NYU)
        """
        return -z / (R**2.0 + z**2.0) ** (self.alpha / 2.0)

    def _rforce_jax(self, r):
        """
        Evaluate the spherical radial force for this potential using JAX.

        Parameters
        ----------
        r : float
            Galactocentric spherical radius.

        Returns
        -------
        float
            The radial force.

        Notes
        -----
        - 2021-02-14 - Written - Bovy (UofT)
        """
        # No need for actual JAX!
        return -self._amp / r ** (self.alpha - 1.0)

    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        """
        Evaluate the second radial derivative for this potential.

        Parameters
        ----------
        R : float
            Galactocentric cylindrical radius.
        z : float
            Vertical height.
        phi : float, optional
            Azimuth (default: 0.0).
        t : float, optional
            Time (default: 0.0).

        Returns
        -------
        float
            The second radial derivative.

        Notes
        -----
        - 2011-10-09 - Written - Bovy (NYU)
        """
        return 1.0 / (R**2.0 + z**2.0) ** (self.alpha / 2.0) - self.alpha * R**2.0 / (
            R**2.0 + z**2.0
        ) ** (self.alpha / 2.0 + 1.0)

    def _z2deriv(self, R, z, phi=0.0, t=0.0):
        """
        Evaluate the second vertical derivative for this potential.

        Parameters
        ----------
        R : float
            Galactocentric cylindrical radius.
        z : float
            Vertical height.
        phi : float, optional
            Azimuth (default: 0.0).
        t : float, optional
            Time (default: 0.0).

        Returns
        -------
        float
            The second vertical derivative.

        Notes
        -----
        - 2012-07-26 - Written - Bovy (IAS@MPIA)
        """
        return self._R2deriv(z, R)  # Spherical potential

    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        """
        Evaluate the mixed R,z derivative for this potential.

        Parameters
        ----------
        R : float
            Galactocentric cylindrical radius.
        z : float
            Vertical height.
        phi : float, optional
            Azimuth (default: 0.0).
        t : float, optional
            Time (default: 0.0).

        Returns
        -------
        float
            The mixed R,z derivative.

        Notes
        -----
        - 2013-08-28 - Written - Bovy (IAs)
        """
        return -self.alpha * R * z * (R**2.0 + z**2.0) ** (-1.0 - self.alpha / 2.0)

    def _dens(self, R, z, phi=0.0, t=0.0):
        """
        Evaluate the density for this potential.

        Parameters
        ----------
        R : float
            Galactocentric cylindrical radius.
        z : float
            Vertical height.
        phi : float, optional
            Azimuth (default: 0.0).
        t : float, optional
            Time (default: 0.0).

        Returns
        -------
        float
            The density.

        Notes
        -----
        - 2013-01-09 - Written - Bovy (IAS)
        """
        r = numpy.sqrt(R**2.0 + z**2.0)
        return (3.0 - self.alpha) / 4.0 / numpy.pi / r**self.alpha

    def _ddensdr(self, r, t=0.0):
        """
        Evaluate the radial density derivative for this potential.

        Parameters
        ----------
        r : float
            Spherical radius.
        t : float, optional
            Time (default: 0.0).

        Returns
        -------
        float
            The density derivative.

        Notes
        -----
        - 2021-02-25 - Written - Bovy (UofT)
        """
        return (
            -self._amp
            * self.alpha
            * (3.0 - self.alpha)
            / 4.0
            / numpy.pi
            / r ** (self.alpha + 1.0)
        )

    def _d2densdr2(self, r, t=0.0):
        """
        Evaluate the second radial density derivative for this potential.

        Parameters
        ----------
        r : float
            Spherical radius.
        t : float, optional
            Time (default: 0.0).

        Returns
        -------
        float
            The second density derivative.

        Notes
        -----
        - 2021-02-25: Written by Bovy (UofT)
        """
        return (
            self._amp
            * (self.alpha + 1.0)
            * self.alpha
            * (3.0 - self.alpha)
            / 4.0
            / numpy.pi
            / r ** (self.alpha + 2.0)
        )

    def _ddenstwobetadr(self, r, beta=0):
        """
        Evaluate the radial density derivative x r^(2beta) for this potential.

        Parameters
        ----------
        r : float
            Spherical radius.
        beta : int, optional
            Power of r (default: 0).

        Returns
        -------
        float
            The d (rho x r^{2beta} ) / d r derivative.

        Notes
        -----
        - 2021-02-14: Written by Bovy (UofT)
        """
        return (
            -self._amp
            * (self.alpha - 2.0 * beta)
            * (3.0 - self.alpha)
            / 4.0
            / numpy.pi
            / r ** (self.alpha + 1.0 - 2.0 * beta)
        )

    def _surfdens(self, R, z, phi=0.0, t=0.0):
        """
        Evaluate the surface density for this potential.

        Parameters
        ----------
        R : float
            Galactocentric cylindrical radius.
        z : float
            Vertical height.
        phi : float, optional
            Azimuth (default: 0.0).
        t : float, optional
            Time (default: 0.0).

        Returns
        -------
        float
            The surface density.

        Notes
        -----
        - 2018-08-19: Written by Bovy (UofT)
        """
        return (
            (3.0 - self.alpha)
            / 2.0
            / numpy.pi
            * z
            * R**-self.alpha
            * special.hyp2f1(0.5, self.alpha / 2.0, 1.5, -((z / R) ** 2))
        )


class KeplerPotential(PowerSphericalPotential):
    """Class that implements the Kepler (point mass) potential

    .. math::

        \\Phi(r) = -\\frac{\\mathrm{amp}}{r}

    with :math:`\\mathrm{amp} = GM` the mass.
    """

    def __init__(self, amp=1.0, normalize=False, ro=None, vo=None):
        """
        Initialize a Kepler, point-mass potential.

        Parameters
        ----------
        amp : float or Quantity, optional
            Amplitude to be applied to the potential, the mass of the point mass (default: 1); can be a Quantity with units of mass or Gxmass.
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.
        ro : float, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float, optional
            Velocity scale for translation into internal units (default from configuration file).

        Returns
        -------
        None

        Notes
        -----
        - 2010-07-10: Written by Bovy (NYU)

        """
        PowerSphericalPotential.__init__(
            self, amp=amp, normalize=normalize, alpha=3.0, ro=ro, vo=vo
        )

    def _mass(self, R, z=None, t=0.0):
        """
        Evaluate the mass within R for this potential.

        Parameters
        ----------
        R : float
            Galactocentric cylindrical radius.
        z : float, optional
            Vertical height (default: None).
        t : float, optional
            Time (default: 0.0).

        Returns
        -------
        float
            The mass enclosed.

        Notes
        -----
        - Written on 2014-07-02 by Bovy (IAS).
        """
        return 1.0
