###############################################################################
#   DehnenSmoothWrapperPotential.py: Wrapper to smoothly grow a potential
###############################################################################
from ..util import conversion
from .WrapperPotential import parentWrapperPotential


class DehnenSmoothWrapperPotential(parentWrapperPotential):
    """Potential wrapper class that implements the growth of a gravitational potential following `Dehnen (2000) <http://adsabs.harvard.edu/abs/2000AJ....119..800D>`__. The amplitude A applied to a potential wrapped by an instance of this class is changed as

    .. math::

        A(t) = amp\\,\\left(\\frac{3}{16}\\xi^5-\\frac{5}{8}\\xi^3+\\frac{15}{16}\\xi+\\frac{1}{2}\\right)

    where

    .. math::

        \\xi = \\begin{cases}
        -1 & t < t_\\mathrm{form}\\\\
        2\\left(\\frac{t-t_\\mathrm{form}}{t_\\mathrm{steady}}\\right)-1\\,, &  t_\\mathrm{form} \\leq t \\leq t_\\mathrm{form}+t_\\mathrm{steady}\\\\
        1 & t > t_\\mathrm{form}+t_\\mathrm{steady}
        \\end{cases}

    if ``decay=True``, the amplitude decays rather than grows as decay = 1 - grow.
    """

    def __init__(
        self, amp=1.0, pot=None, tform=-4.0, tsteady=None, decay=False, ro=None, vo=None
    ):
        """
        Initialize a DehnenSmoothWrapper Potential.

        Parameters
        ----------
        amp : float, optional
            Amplitude to be applied to the potential (default: 1.).
        pot : Potential instance or list thereof, optional
            The amplitude of this will be grown by this wrapper.
        tform : float or Quantity, optional
            Start of growth (default: -4.0).
        tsteady : float or Quantity, optional
            Time from tform at which the potential is fully grown (default: -tform/2, so the perturbation is fully grown at tform/2).
        decay : bool, optional
            If True, decay the amplitude instead of growing it (as 1-grow).
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2017-06-26 - Started - Bovy (UofT)
        - 2018-10-07 - Added 'decay' option - Bovy (UofT)

        """
        tform = conversion.parse_time(tform, ro=self._ro, vo=self._vo)
        tsteady = conversion.parse_time(tsteady, ro=self._ro, vo=self._vo)
        self._tform = tform
        if tsteady is None:
            self._tsteady = self._tform / 2.0
        else:
            self._tsteady = self._tform + tsteady
        self._grow = not decay
        self.hasC = True
        self.hasC_dxdv = True

    def _smooth(self, t):
        # Calculate relevant time
        if t < self._tform:
            smooth = 0.0
        elif t < self._tsteady:
            deltat = t - self._tform
            xi = 2.0 * deltat / (self._tsteady - self._tform) - 1.0
            smooth = 3.0 / 16.0 * xi**5.0 - 5.0 / 8 * xi**3.0 + 15.0 / 16.0 * xi + 0.5
        else:  # bar is fully on
            smooth = 1.0
        return smooth if self._grow else 1.0 - smooth

    def _wrap(self, attribute, *args, **kwargs):
        return self._smooth(kwargs.get("t", 0.0)) * self._wrap_pot_func(attribute)(
            self._pot, *args, **kwargs
        )
