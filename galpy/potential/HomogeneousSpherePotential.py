###############################################################################
#   HomogeneousSpherePotential.py: The potential of a homogeneous sphere
###############################################################################
import numpy

from ..util import conversion
from .Potential import Potential


class HomogeneousSpherePotential(Potential):
    """Class that implements the homogeneous sphere potential for :math:`\\rho(r) = \\rho_0 = \\mathrm{constant}` for all :math:`r < R` and zero otherwise. The potential is given by

    .. math::

        \\Phi(r) = \\mathrm{amp}\\times\\left\\{\\begin{array}{lr}
        (r^2-3R^2), & \\text{for } r < R\\\\
        -\\frac{2R^3}{r} & \\text{for } r \\geq R
        \\end{array}\\right.

    We have that :math:`\\rho_0 = 3\\,\\mathrm{amp}/[2\\pi G]`.
    """

    def __init__(self, amp=1.0, R=1.1, normalize=False, ro=None, vo=None):
        """
        Initialize a homogeneous sphere potential.

        Parameters
        ----------
        amp : float or Quantity, optional
            Amplitude to be applied to the potential. Can be a Quantity with units of mass density or Gxmass density.
        R : float or Quantity, optional
            Size of the sphere.
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2019-12-20 - Written - Bovy (UofT)
        """
        Potential.__init__(self, amp=amp, ro=ro, vo=vo, amp_units="density")
        R = conversion.parse_length(R, ro=self._ro)
        self.R = R
        self._R2 = self.R**2.0
        self._R3 = self.R**3.0
        if normalize or (
            isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
        ):  # pragma: no cover
            self.normalize(normalize)
        self.hasC = True
        self.hasC_dxdv = True
        self.hasC_dens = True

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        r2 = R**2.0 + z**2.0
        if r2 < self._R2:
            return r2 - 3.0 * self._R2
        else:
            return -2.0 * self._R3 / numpy.sqrt(r2)

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        r2 = R**2.0 + z**2.0
        if r2 < self._R2:
            return -2.0 * R
        else:
            return -2.0 * self._R3 * R / r2**1.5

    def _zforce(self, R, z, phi=0.0, t=0.0):
        r2 = R**2.0 + z**2.0
        if r2 < self._R2:
            return -2.0 * z
        else:
            return -2.0 * self._R3 * z / r2**1.5

    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        r2 = R**2.0 + z**2.0
        if r2 < self._R2:
            return 2.0
        else:
            return 2.0 * self._R3 / r2**1.5 - 6.0 * self._R3 * R**2.0 / r2**2.5

    def _z2deriv(self, R, z, phi=0.0, t=0.0):
        r2 = R**2.0 + z**2.0
        if r2 < self._R2:
            return 2.0
        else:
            return 2.0 * self._R3 / r2**1.5 - 6.0 * self._R3 * z**2.0 / r2**2.5

    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        r2 = R**2.0 + z**2.0
        if r2 < self._R2:
            return 0.0
        else:
            return -6.0 * self._R3 * R * z / r2**2.5

    def _dens(self, R, z, phi=0.0, t=0.0):
        r2 = R**2.0 + z**2.0
        if r2 < self._R2:
            return 1.5 / numpy.pi
        else:
            return 0.0
