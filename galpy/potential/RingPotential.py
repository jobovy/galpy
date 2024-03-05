###############################################################################
#   RingPotential.py: The gravitational potential of a thin, circular ring
###############################################################################
import numpy
from scipy import special

from ..util import conversion
from .Potential import Potential


class RingPotential(Potential):
    """Class that implements the potential of an infinitesimally-thin, circular ring

    .. math::

        \\rho(R,z) = \\frac{\\mathrm{amp}}{2\\pi\\,R_0}\\,\\delta(R-R_0)\\,\\delta(z)

    with :math:`\\mathrm{amp} = GM` the mass of the ring.
    """

    def __init__(self, amp=1.0, a=0.75, normalize=False, ro=None, vo=None):
        """
        Class that implements a circular ring potential.

        Parameters
        ----------
        amp : float or Quantity, optional
            Mass of the ring (default: 1); can be a Quantity with units of mass or Gxmass.
        a : float or Quantity, optional
            Radius of the ring (default: 0.75).
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.; note that because the force is always positive at r < a, this does not work if a > 1.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2018-08-04 - Written - Bovy (UofT)
        """
        Potential.__init__(self, amp=amp, ro=ro, vo=vo, amp_units="mass")
        a = conversion.parse_length(a, ro=self._ro)
        self.a = a
        self.a2 = self.a**2
        self._amp /= 2.0 * numpy.pi * self.a
        if normalize or (
            isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
        ):
            if self.a > 1.0:
                raise ValueError(
                    "RingPotential with normalize= for a > 1 is not supported (because the force is always positive at r=1)"
                )
            self.normalize(normalize)
        self.hasC = False
        self.hasC_dxdv = False

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        # Stable as r -> infty
        m = 4.0 * self.a / ((numpy.sqrt(R) + self.a / numpy.sqrt(R)) ** 2 + z**2 / R)
        return -4.0 * self.a / numpy.sqrt((R + self.a) ** 2 + z**2) * special.ellipk(m)

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        m = 4.0 * R * self.a / ((R + self.a) ** 2 + z**2)
        return (
            -2.0
            * self.a
            / R
            / numpy.sqrt((R + self.a) ** 2 + z**2)
            * (
                m
                * (R**2 - self.a2 - z**2)
                / 4.0
                / (1.0 - m)
                / self.a
                / R
                * special.ellipe(m)
                + special.ellipk(m)
            )
        )

    def _zforce(self, R, z, phi=0.0, t=0.0):
        m = 4.0 * R * self.a / ((R + self.a) ** 2 + z**2)
        return (
            -4.0
            * z
            * self.a
            / (1.0 - m)
            * ((R + self.a) ** 2 + z**2) ** -1.5
            * special.ellipe(m)
        )

    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        Raz2 = (R + self.a) ** 2 + z**2
        Raz = numpy.sqrt(Raz2)
        m = 4.0 * R * self.a / Raz2
        R2ma2mz2o4aR1m = (R**2 - self.a2 - z**2) / 4.0 / self.a / R / (1.0 - m)
        return (2 * R**2 + self.a2 + 3 * R * self.a + z**2) / R / Raz2 * self._Rforce(
            R, z
        ) + 2.0 * self.a / R / Raz * (
            m
            * (R**2 + self.a2 + z**2)
            / 4.0
            / (1.0 - m)
            / self.a
            / R**2
            * special.ellipe(m)
            + (
                R2ma2mz2o4aR1m / (1.0 - m) * special.ellipe(m)
                + 0.5 * R2ma2mz2o4aR1m * (special.ellipe(m) - special.ellipk(m))
                + 0.5 * (special.ellipe(m) / (1.0 - m) - special.ellipk(m)) / m
            )
            * 4
            * self.a
            * (self.a2 + z**2 - R**2)
            / Raz2**2
        )

    def _z2deriv(self, R, z, phi=0.0, t=0.0):
        Raz2 = (R + self.a) ** 2 + z**2
        m = 4.0 * R * self.a / Raz2
        # Explicitly swapped in zforce here, so the z/z can be cancelled
        # and z=0 is handled properly
        return (
            -4.0
            * (
                3.0 * z**2 / Raz2
                - 1.0
                + 4.0
                * ((1.0 + m) / (1.0 - m) - special.ellipk(m) / special.ellipe(m))
                * self.a
                * R
                * z**2
                / Raz2**2
                / m
            )
            * self.a
            / (1.0 - m)
            * ((R + self.a) ** 2 + z**2) ** -1.5
            * special.ellipe(m)
        )

    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        Raz2 = (R + self.a) ** 2 + z**2
        m = 4.0 * R * self.a / Raz2
        return (
            3.0 * (R + self.a) / Raz2
            - 2.0
            * ((1.0 + m) / (1.0 - m) - special.ellipk(m) / special.ellipe(m))
            * self.a
            * (self.a2 + z**2 - R**2)
            / Raz2**2
            / m
        ) * self._zforce(R, z)
