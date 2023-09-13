###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleHarmonic
#
#             Calculate actions-angle coordinates for the harmonic-oscillator
#
#      methods:
#             __call__: returns (j)
#             actionsFreqs: returns (j,omega)
#             actionsFreqsAngles: returns (j,omega,a)
#
###############################################################################
import numpy

from ..util import conversion
from .actionAngle import actionAngle


class actionAngleHarmonic(actionAngle):
    """Action-angle formalism for the one-dimensional harmonic oscillator"""

    def __init__(self, *args, **kwargs):
        """
        Initialize an actionAngleHarmonic object.

        Parameters
        ----------
        omega : float or numpy.ndarray
            Frequencies (can be Quantity).
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2018-04-08 - Written - Bovy (Uoft)

        """
        actionAngle.__init__(self, ro=kwargs.get("ro", None), vo=kwargs.get("vo", None))
        if not "omega" in kwargs:  # pragma: no cover
            raise OSError("Must specify omega= for actionAngleHarmonic")
        self._omega = conversion.parse_frequency(
            kwargs.get("omega"), ro=self._ro, vo=self._vo
        )
        return None

    def _evaluate(self, *args, **kwargs):
        """
        Evaluate the action for the harmonic oscillator

        Parameters
        ----------
        Either:
            a) x,vx:
                1) floats: phase-space value for single object (each can be a Quantity)
                2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)

        Returns
        -------
        float or numpy.ndarray
            action

        Notes
        -----
        - 2018-04-08 - Written - Bovy (UofT)
        """
        if len(args) == 2:  # x,vx
            x, vx = args
            return (vx**2.0 / self._omega + self._omega * x**2.0) / 2.0
        else:  # pragma: no cover
            raise ValueError("actionAngleHarmonic __call__ input not understood")

    def _actionsFreqs(self, *args, **kwargs):
        """
        Evaluate the action and frequency for the harmonic oscillator

        Parameters
        ----------
        Either:
            a) x,vx:
                1) floats: phase-space value for single object (each can be a Quantity)
                2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)

        Returns
        -------
        tuple
            (action,frequency)

        Notes
        -----
        - 2018-04-08 - Written - Bovy (UofT)
        """
        if len(args) == 2:  # x,vx
            x, vx = args
            return (
                (vx**2.0 / self._omega + self._omega * x**2.0) / 2.0,
                self._omega * numpy.ones_like(x),
            )
        else:  # pragma: no cover
            raise ValueError("actionAngleHarmonic __call__ input not understood")

    def _actionsFreqsAngles(self, *args, **kwargs):
        """
        Evaluate the action, frequency, and angle for the harmonic oscillator

        Parameters
        ----------
        Either:
            a) x,vx:
                1) floats: phase-space value for single object (each can be a Quantity)
                2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)

        Returns
        -------
        tuple
            (action,frequency,angle)

        Notes
        -----
        - 2018-04-08 - Written - Bovy (UofT)
        """
        if len(args) == 2:  # x,vx
            x, vx = args
            return (
                (vx**2.0 / self._omega + self._omega * x**2.0) / 2.0,
                self._omega * numpy.ones_like(x),
                numpy.arctan2(self._omega * x, vx),
            )
        else:  # pragma: no cover
            raise ValueError("actionAngleHarmonic __call__ input not understood")
