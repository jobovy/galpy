###############################################################################
#   NullPotential.py: class that implements a constant potential
###############################################################################
from .Potential import Potential


class NullPotential(Potential):
    """Class that implements a constant potential with, thus, zero forces. Can be used, for example, for integrating orbits in the absence of forces or for adjusting the value of the total gravitational potential, say, at infinity"""

    normalize = property()  # turn off normalize

    def __init__(self, amp=1.0, ro=None, vo=None):
        """
        Initialize a null potential: a constant potential with, thus, zero forces

        Parameters
        ----------
        amp : float or Quantity, optional
            Constant value of the potential (default: 1); can be a Quantity with units of velocity-squared
        ro : float, optional
            Distance scale for translation into internal units (default from configuration file)
        vo : float, optional
            Velocity scale for translation into internal units (default from configuration file)

        Notes
        -----
        - 2022-03-18 - Written - Bovy (UofT)

        """
        Potential.__init__(self, amp=amp, ro=ro, vo=vo, amp_units="velocity2")
        self.hasC = True
        self.hasC_dxdv = True
        self.hasC_dens = True
        return None

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        return 1.0

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        return 0.0

    def _zforce(self, R, z, phi=0.0, t=0.0):
        return 0.0

    def _dens(self, R, z, phi=0.0, t=0.0):
        return 0.0

    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        return 0.0

    def _z2deriv(self, R, z, phi=0.0, t=0.0):
        return 0.0

    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        return 0.0
