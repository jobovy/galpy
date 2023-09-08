###############################################################################
#   SolidBodyRotationWrapperPotential.py: Wrapper to make a potential rotate
#                                         with a fixed pattern speed, around
#                                         the z axis
###############################################################################
from ..util import conversion
from .WrapperPotential import parentWrapperPotential


class SolidBodyRotationWrapperPotential(parentWrapperPotential):
    """Potential wrapper class that implements solid-body rotation around the z-axis. Can be used to make a bar or other perturbation rotate. The potential is rotated by replacing

    .. math::

        \\phi \\rightarrow \\phi + \\Omega \\times t + \\mathrm{pa}

    with :math:`\\Omega` the fixed pattern speed and :math:`\\mathrm{pa}` the position angle at :math:`t=0`.
    """

    def __init__(self, amp=1.0, pot=None, omega=1.0, pa=0.0, ro=None, vo=None):
        """
        Initialize a SolidBodyRotationWrapper Potential.

        Parameters
        ----------
        amp : float, optional
            Amplitude to be applied to the potential. Default is 1.0.
        pot : Potential instance or list thereof
            This potential is made to rotate around the z axis by the wrapper.
        omega : float or Quantity, optional
            The pattern speed. Default is 1.0.
        pa : float or Quantity, optional
            The position angle. Default is 0.0.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2017-08-22 - Started - Bovy (UofT)

        """
        omega = conversion.parse_frequency(omega, ro=self._ro, vo=self._vo)
        pa = conversion.parse_angle(pa)
        self._omega = omega
        self._pa = pa
        self.hasC = True
        self.hasC_dxdv = True

    def OmegaP(self):
        return self._omega

    def _wrap(self, attribute, *args, **kwargs):
        kwargs["phi"] = (
            kwargs.get("phi", 0.0) - self._omega * kwargs.get("t", 0.0) - self._pa
        )
        return self._wrap_pot_func(attribute)(self._pot, *args, **kwargs)
