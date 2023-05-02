###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleVertical
#
#      methods:
#             __call__: returns (j)
#             actionsFreqs: returns (j,omega)
#             actionsFreqsAngles: returns (j,omega,a)
#             calcxmax
###############################################################################
import numpy
from scipy import integrate, optimize

from ..potential.linearPotential import evaluatelinearPotentials
from .actionAngle import actionAngle


class actionAngleVertical(actionAngle):
    """Action-angle formalism for one-dimensional potentials (or of the vertical potential in a galactic disk in the adiabatic approximation, hence the name)"""

    def __init__(self, *args, **kwargs):
        """
        NAME:

           __init__

        PURPOSE:

           initialize an actionAngleVertical object

        INPUT:

           pot= potential or list of 1D potentials (linearPotential or verticalPotential)

           ro= distance from vantage point to GC (kpc; can be Quantity)

           vo= circular velocity at ro (km/s; can be Quantity)

        OUTPUT:

           instance

        HISTORY:

           2012-06-01 - Written - Bovy (IAS)

           2018-05-19 - Conformed to the general actionAngle framework - Bovy (UofT)

        """
        actionAngle.__init__(self, ro=kwargs.get("ro", None), vo=kwargs.get("vo", None))
        if not "pot" in kwargs:  # pragma: no cover
            raise OSError("Must specify pot= for actionAngleVertical")
        if not "pot" in kwargs:  # pragma: no cover
            raise OSError("Must specify pot= for actionAngleVertical")
        self._pot = kwargs["pot"]
        return None
        """
        self._parse_eval_args(*args,_noOrbUnitsCheck=True,**kwargs)
        self._z= self._eval_z
        self._vz= self._eval_vz
        self._verticalpot= kwargs['pot']
        return None
        """

    def _evaluate(self, *args, **kwargs):
        """
        NAME:
           __call__ (_evaluate)
        PURPOSE:
           evaluate the action
        INPUT:
           Either:
              a) x,vx:
                 1) floats: phase-space value for single object (each can be a Quantity)
                 2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
        OUTPUT:
           action
        HISTORY:
           2018-05-19 - Written based on re-write of existing code - Bovy (UofT)
        """
        if len(args) == 2:  # x,vx
            x, vx = args
            if isinstance(x, float):
                x = numpy.array([x])
                vx = numpy.array([vx])
            J = numpy.empty(len(x))
            for ii in range(len(x)):
                E = vx[ii] ** 2.0 / 2.0 + evaluatelinearPotentials(
                    self._pot, x[ii], use_physical=False
                )
                xmax = self.calcxmax(x[ii], vx[ii], E)
                if xmax == -9999.99:
                    J[ii] = 9999.99
                else:
                    J[ii] = (
                        2.0
                        * integrate.quad(
                            lambda xi: numpy.sqrt(
                                2.0
                                * (
                                    E
                                    - evaluatelinearPotentials(
                                        self._pot, xi, use_physical=False
                                    )
                                )
                            ),
                            0.0,
                            xmax,
                        )[0]
                        / numpy.pi
                    )
            return J
        else:  # pragma: no cover
            raise ValueError("actionAngleVertical __call__ input not understood")

    def _actionsFreqs(self, *args, **kwargs):
        """
        NAME:
           actionsFreqs (_actionsFreqs)
        PURPOSE:
           evaluate the action and frequency
        INPUT:
           Either:
              a) x,vx:
                 1) floats: phase-space value for single object (each can be a Quantity)
                 2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
        OUTPUT:
           action,frequency
        HISTORY:
           2018-05-19 - Written based on re-write of existing code - Bovy (UofT)
        """
        if len(args) == 2:  # x,vx
            x, vx = args
            if isinstance(x, float):
                x = numpy.array([x])
                vx = numpy.array([vx])
            J = numpy.empty(len(x))
            Omega = numpy.empty(len(x))
            for ii in range(len(x)):
                E = vx[ii] ** 2.0 / 2.0 + evaluatelinearPotentials(
                    self._pot, x[ii], use_physical=False
                )
                xmax = self.calcxmax(x[ii], vx[ii], E)
                if xmax == -9999.99:
                    J[ii] = 9999.99
                    Omega[ii] = 9999.99
                else:
                    J[ii] = (
                        2.0
                        * integrate.quad(
                            lambda xi: numpy.sqrt(
                                2.0
                                * (
                                    E
                                    - evaluatelinearPotentials(
                                        self._pot, xi, use_physical=False
                                    )
                                )
                            ),
                            0.0,
                            xmax,
                        )[0]
                        / numpy.pi
                    )
                    # Transformed x = xmax-t^2 for singularity
                    Omega[ii] = (
                        numpy.pi
                        / 2.0
                        / integrate.quad(
                            lambda t: 2.0
                            * t
                            / numpy.sqrt(
                                2.0
                                * (
                                    E
                                    - evaluatelinearPotentials(
                                        self._pot, xmax - t**2.0, use_physical=False
                                    )
                                )
                            ),
                            0,
                            numpy.sqrt(xmax),
                        )[0]
                    )
            return (J, Omega)
        else:  # pragma: no cover
            raise ValueError("actionAngleVertical actionsFreqs input not understood")

    def _actionsFreqsAngles(self, *args, **kwargs):
        """
        NAME:
           actionsFreqsAngles (_actionsFreqsAngles)
        PURPOSE:
           evaluate the action, frequency, and angle
        INPUT:
           Either:
              a) x,vx:
                 1) floats: phase-space value for single object (each can be a Quantity)
                 2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
        OUTPUT:
           action,frequency,angle
        HISTORY:
           2018-05-19 - Written based on re-write of existing code - Bovy (UofT)
        """
        if len(args) == 2:  # x,vx
            x, vx = args
            if isinstance(x, float):
                x = numpy.array([x])
                vx = numpy.array([vx])
            J = numpy.empty(len(x))
            Omega = numpy.empty(len(x))
            angle = numpy.empty(len(x))
            for ii in range(len(x)):
                E = vx[ii] ** 2.0 / 2.0 + evaluatelinearPotentials(
                    self._pot, x[ii], use_physical=False
                )
                xmax = self.calcxmax(x[ii], vx[ii], E)
                if xmax == -9999.99:
                    J[ii] = 9999.99
                    Omega[ii] = 9999.99
                    angle[ii] = 9999.99
                else:
                    J[ii] = (
                        2.0
                        * integrate.quad(
                            lambda xi: numpy.sqrt(
                                2.0
                                * (
                                    E
                                    - evaluatelinearPotentials(
                                        self._pot, xi, use_physical=False
                                    )
                                )
                            ),
                            0.0,
                            xmax,
                        )[0]
                        / numpy.pi
                    )
                    Omega[ii] = (
                        numpy.pi
                        / 2.0
                        / integrate.quad(
                            lambda t: 2.0
                            * t
                            / numpy.sqrt(
                                2.0
                                * (
                                    E
                                    - evaluatelinearPotentials(
                                        self._pot, xmax - t**2.0, use_physical=False
                                    )
                                )
                            ),
                            0,
                            numpy.sqrt(xmax),
                        )[0]
                    )
                    angle[ii] = integrate.quad(
                        lambda xi: 1.0
                        / numpy.sqrt(
                            2.0
                            * (
                                E
                                - evaluatelinearPotentials(
                                    self._pot, xi, use_physical=False
                                )
                            )
                        ),
                        0,
                        numpy.fabs(x[ii]),
                    )[0]
            angle *= Omega
            angle[(x >= 0.0) * (vx < 0.0)] = numpy.pi - angle[(x >= 0.0) * (vx < 0.0)]
            angle[(x < 0.0) * (vx <= 0.0)] = numpy.pi + angle[(x < 0.0) * (vx <= 0.0)]
            angle[(x < 0.0) * (vx > 0.0)] = (
                2.0 * numpy.pi - angle[(x < 0.0) * (vx > 0.0)]
            )
            return (J, Omega, angle % (2.0 * numpy.pi))
        else:  # pragma: no cover
            raise ValueError(
                "actionAngleVertical actionsFreqsAngles input not understood"
            )

    def calcxmax(self, x, vx, E=None):
        """
        NAME:
           calcxmax
        PURPOSE:
           calculate the maximum height
        INPUT:
           x - position
           vx - velocity
        OUTPUT:
           zmax
        HISTORY:
           2012-06-01 - Written - Bovy (IAS)
           2018-05-19 - Re-written for new framework - Bovy (UofT)
        """
        if E is None:
            E = E = vx**2.0 / 2.0 + evaluatelinearPotentials(
                self._pot, x, use_physical=False
            )
        if vx == 0.0:  # We are exactly at the maximum height
            xmax = numpy.fabs(x)
        else:
            xstart = x
            try:
                if x == 0.0:
                    xend = 0.00001
                else:
                    xend = 2.0 * numpy.fabs(x)
                while (
                    E - evaluatelinearPotentials(self._pot, xend, use_physical=False)
                ) > 0.0:
                    xend *= 2.0
                    if xend > 100.0:  # pragma: no cover
                        raise OverflowError
            except OverflowError:  # pragma: no cover
                xmax = -9999.99
            else:
                xmax = optimize.brentq(
                    lambda xm: E
                    - evaluatelinearPotentials(self._pot, xm, use_physical=False),
                    xstart,
                    xend,
                    xtol=1e-14,
                )
                while (
                    E - evaluatelinearPotentials(self._pot, xmax, use_physical=False)
                ) < 0:
                    xmax -= 1e-14
        return xmax
