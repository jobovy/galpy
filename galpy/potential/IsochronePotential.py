###############################################################################
#   IsochronePotential.py: The isochrone potential
#
#                                     - amp
#                          Phi(r)= ---------------------
#                                   b + sqrt{b^2+r^2}
###############################################################################
import numpy

from ..util import conversion
from .Potential import Potential


class IsochronePotential(Potential):
    """Class that implements the Isochrone potential

    .. math::

        \\Phi(r) = -\\frac{\\mathrm{amp}}{b+\\sqrt{b^2+r^2}}

    with :math:`\\mathrm{amp} = GM` the total mass.
    """

    def __init__(self, amp=1.0, b=1.0, normalize=False, ro=None, vo=None):
        """
        Initialize an isochrone potential.

        Parameters
        ----------
        amp : float or Quantity, optional
            Amplitude to be applied to the potential, the total mass. Can be a Quantity with units of mass or Gxmass.
        b : float or Quantity, optional
            Scale radius of the isochrone potential.
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1. Default is False.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2013-09-08 - Written - Bovy (IAS)
        """
        Potential.__init__(self, amp=amp, ro=ro, vo=vo, amp_units="mass")
        b = conversion.parse_length(b, ro=self._ro)
        self.b = b
        self._scale = self.b
        self.b2 = self.b**2.0
        if normalize or (
            isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
        ):  # pragma: no cover
            self.normalize(normalize)
        self.hasC = True
        self.hasC_dxdv = True
        self.hasC_dens = True

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        r2 = R**2.0 + z**2.0
        rb = numpy.sqrt(r2 + self.b2)
        return -1.0 / (self.b + rb)

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        r2 = R**2.0 + z**2.0
        rb = numpy.sqrt(r2 + self.b2)
        dPhidrr = -1.0 / rb / (self.b + rb) ** 2.0
        return dPhidrr * R

    def _zforce(self, R, z, phi=0.0, t=0.0):
        r2 = R**2.0 + z**2.0
        rb = numpy.sqrt(r2 + self.b2)
        dPhidrr = -1.0 / rb / (self.b + rb) ** 2.0
        return dPhidrr * z

    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        r2 = R**2.0 + z**2.0
        rb = numpy.sqrt(r2 + self.b2)
        return (
            -(
                -(self.b**3.0)
                - self.b * z**2.0
                + (2.0 * R**2.0 - z**2.0 - self.b**2.0) * rb
            )
            / rb**3.0
            / (self.b + rb) ** 3.0
        )

    def _z2deriv(self, R, z, phi=0.0, t=0.0):
        r2 = R**2.0 + z**2.0
        rb = numpy.sqrt(r2 + self.b2)
        return (
            -(
                -(self.b**3.0)
                - self.b * R**2.0
                - (R**2.0 - 2.0 * z**2.0 + self.b**2.0) * rb
            )
            / rb**3.0
            / (self.b + rb) ** 3.0
        )

    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        r2 = R**2.0 + z**2.0
        rb = numpy.sqrt(r2 + self.b2)
        return -R * z * (self.b + 3.0 * rb) / rb**3.0 / (self.b + rb) ** 3.0

    def _dens(self, R, z, phi=0.0, t=0.0):
        r2 = R**2.0 + z**2.0
        rb = numpy.sqrt(r2 + self.b2)
        return (
            (3.0 * (self.b + rb) * rb**2.0 - r2 * (self.b + 3.0 * rb))
            / rb**3.0
            / (self.b + rb) ** 3.0
            / 4.0
            / numpy.pi
        )

    def _surfdens(self, R, z, phi=0.0, t=0.0):
        r2 = R**2.0 + z**2.0
        rb = numpy.sqrt(r2 + self.b2)
        return (
            self.b
            * (
                (R * z) / r2
                - (self.b * R * z * (self.b**2 + 2.0 * R**2 + z**2))
                / ((self.b**2 + R**2) * r2 * rb)
                + numpy.arctan(z / R)
                - numpy.arctan(self.b * z / R / rb)
            )
            / R**3
            / 2.0
            / numpy.pi
        )
