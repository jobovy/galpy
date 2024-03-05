###############################################################################
#   KuzminDiskPotential.py: class that implements Kuzmin disk potential
#
#                                   - amp
#               Phi(R, z)=  ---------------------------
#                            \sqrt{R^2 + (a + |z|)^2}
###############################################################################
import numpy

from ..util import conversion
from .Potential import Potential


class KuzminDiskPotential(Potential):
    """Class that implements the Kuzmin Disk potential

    .. math::

        \\Phi(R,z) = -\\frac{\\mathrm{amp}}{\\sqrt{R^2 + (a + |z|)^2}}

    with :math:`\\mathrm{amp} = GM` the total mass.
    """

    def __init__(self, amp=1.0, a=1.0, normalize=False, ro=None, vo=None):
        """
        Initialize a Kuzmin disk Potential.

        Parameters
        ----------
        amp : float or Quantity, optional
            Amplitude to be applied to the potential, the total mass. Can be a Quantity with units of mass or Gxmass.
        a : float or Quantity, optional
            Scale length.
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.
        ro : float, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2016-05-09 - Written - Aladdin

        """
        Potential.__init__(self, amp=amp, ro=ro, vo=vo, amp_units="mass")
        a = conversion.parse_length(a, ro=self._ro)
        self._a = a  ## a must be greater or equal to 0.
        if normalize or (
            isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
        ):
            self.normalize(normalize)
        self.hasC = True
        self.hasC_dxdv = True
        return None

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        return -(self._denom(R, z) ** -0.5)

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        return -(self._denom(R, z) ** -1.5) * R

    def _zforce(self, R, z, phi=0.0, t=0.0):
        return -numpy.sign(z) * self._denom(R, z) ** -1.5 * (self._a + numpy.fabs(z))

    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        return self._denom(R, z) ** -1.5 - 3.0 * R**2 * self._denom(R, z) ** -2.5

    def _z2deriv(self, R, z, phi=0.0, t=0.0):
        a = self._a
        return (
            self._denom(R, z) ** -1.5
            - 3.0 * (a + numpy.fabs(z)) ** 2.0 * self._denom(R, z) ** -2.5
        )

    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        return (
            -3
            * numpy.sign(z)
            * R
            * (self._a + numpy.fabs(z))
            * self._denom(R, z) ** -2.5
        )

    def _surfdens(self, R, z, phi=0.0, t=0.0):
        return self._a * (R**2 + self._a**2) ** -1.5 / 2.0 / numpy.pi

    def _mass(self, R, z=None, t=0.0):
        return 1.0 - self._a / numpy.sqrt(R**2.0 + self._a**2.0)

    def _denom(self, R, z):
        return R**2.0 + (self._a + numpy.fabs(z)) ** 2.0
