###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleHarmonicInverse
#
#             Calculate (x,v) coordinates for the harmonic oscillator from
#             given actions-angle coordinates
#
###############################################################################
import numpy

from ..backend import get_namespace, promote_scalars
from ..util import conversion
from .actionAngleInverse import actionAngleInverse


class actionAngleHarmonicInverse(actionAngleInverse):
    """Inverse action-angle formalism for the one-dimensional harmonic oscillator"""

    def __init__(self, *args, **kwargs):
        """
        Initialize an actionAngleHarmonicInverse object.

        Parameters
        ----------
        omega : float or Quantity
            Frequency.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2018-04-08 - Started - Bovy (UofT)
        """
        actionAngleInverse.__init__(self, *args, **kwargs)
        if not "omega" in kwargs:  # pragma: no cover
            raise OSError("Must specify omega= for actionAngleHarmonic")
        omega = conversion.parse_frequency(
            kwargs.get("omega"), ro=self._ro, vo=self._vo
        )
        self._omega = omega
        return None

    def _evaluate(self, j, angle, **kwargs):
        """
        Evaluate the phase-space coordinates (x,v) for a number of angles on a single torus

        Parameters
        ----------
        j : float
            Action
        angle : numpy.ndarray
            Angle

        Returns
        -------
        x_vx : list
            A list containing the phase-space coordinates [x,vx]

        Notes
        -----
        - 2018-04-08 - Written - Bovy (UofT)
        """
        return self._xvFreqs(j, angle, **kwargs)[:2]

    def _xvFreqs(self, j, angle, **kwargs):
        """
        Evaluate the phase-space coordinates (x,v) for a number of angles on a single torus as well as the frequency

        Parameters
        ----------
        j : float
            Action.
        angle : numpy.ndarray
            Angle.

        Returns
        -------
        tuple
            Tuple containing:
                - x (numpy.ndarray): x-coordinate.
                - vx (numpy.ndarray): Velocity in x-direction.
                - Omega (float): Frequency.

        Notes
        -----
        - 2018-04-08 - Written - Bovy (UofT)

        """
        xp = get_namespace(j, angle)
        # omega joins the backend too: a plain-float frequency would be an island
        j, angle, omega = promote_scalars(xp, j, angle, self._omega)
        amp = xp.sqrt(2.0 * j / omega)
        x = amp * xp.sin(angle)
        vx = amp * omega * xp.cos(angle)
        return (x, vx, omega)

    def _Freqs(self, j, **kwargs):
        """
        Return the frequency corresponding to a torus

        Parameters
        ----------
        j : scalar
            action

        Returns
        -------
        Omega : float
            frequency

        Notes
        -----
        - 2018-04-08 - Written - Bovy (UofT)

        """
        return promote_scalars(get_namespace(j), j, self._omega)[1]
