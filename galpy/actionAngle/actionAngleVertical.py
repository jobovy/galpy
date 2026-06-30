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

from ..backend import device_of, get_namespace
from ..potential.linearPotential import evaluatelinearPotentials
from ..potential.Potential import _check_potential_list_and_deprecate
from .actionAngle import actionAngle

# Gauss-Legendre order for the backend (jax/torch) action/freq/angle quadratures
# (matches actionAngleSpherical's choice).
_BACKEND_GL_ORDER = 50


class actionAngleVertical(actionAngle):
    """Action-angle formalism for one-dimensional potentials (or of the vertical potential in a galactic disk in the adiabatic approximation, hence the name)"""

    def __init__(self, *args, **kwargs):
        """
        Initialize an actionAngleVertical object.

        Parameters
        ----------
        pot : potential or a combined potential formed using addition (pot1+pot2+…) of 1D potentials (linearPotential or verticalPotential)
            Potential or a combined potential formed using addition (pot1+pot2+…) of 1D potentials.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2012-06-01 - Written - Bovy (IAS)
        - 2018-05-19 - Conformed to the general actionAngle framework - Bovy (UofT)

        """
        actionAngle.__init__(self, ro=kwargs.get("ro", None), vo=kwargs.get("vo", None))
        if not "pot" in kwargs:  # pragma: no cover
            raise OSError("Must specify pot= for actionAngleVertical")
        if not "pot" in kwargs:  # pragma: no cover
            raise OSError("Must specify pot= for actionAngleVertical")
        self._pot = _check_potential_list_and_deprecate(kwargs["pot"])
        return None

    def _evaluate(self, *args, **kwargs):
        """
        Evaluate the action.

        Parameters
        ----------
        *args : tuple
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
        - 2018-05-19 - Written based on re-write of existing code - Bovy (UofT)
        """
        if len(args) == 2:  # x,vx
            x, vx = args
            if isinstance(x, float):
                x = numpy.array([x])
                vx = numpy.array([vx])
            xp = get_namespace(x, vx)
            if xp is not numpy:
                x, vx = xp.asarray(x), xp.asarray(vx)
                return self._evaluate_backend(x, vx)
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
        Evaluate the action and frequency.

        Parameters
        ----------
        *args : tuple
            Either:
              a) x,vx:
                 1) floats: phase-space value for single object (each can be a Quantity)
                 2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)

        Returns
        -------
        tuple
            action,frequency

        Notes
        -----
        - 2018-05-19 - Written based on re-write of existing code - Bovy (UofT)
        """
        if len(args) == 2:  # x,vx
            x, vx = args
            if isinstance(x, float):
                x = numpy.array([x])
                vx = numpy.array([vx])
            xp = get_namespace(x, vx)
            if xp is not numpy:
                x, vx = xp.asarray(x), xp.asarray(vx)
                return self._actionsFreqs_backend(x, vx)
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
                            lambda t: (
                                2.0
                                * t
                                / (
                                    numpy.sqrt(
                                        2.0
                                        * (
                                            E
                                            - evaluatelinearPotentials(
                                                self._pot,
                                                xmax - t**2.0,
                                                use_physical=False,
                                            )
                                        )
                                    )
                                    if t > 1e-6
                                    else 1.0
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
        Evaluate the action, frequency, and angle.

        Parameters
        ----------
        *args : tuple
            Either:
              a) x,vx:
                 1) floats: phase-space value for single object (each can be a Quantity)
                 2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)

        Returns
        -------
        tuple
            action,frequency,angle

        Notes
        -----
        - 2018-05-19 - Written based on re-write of existing code - Bovy (UofT)
        """
        if len(args) == 2:  # x,vx
            x, vx = args
            if isinstance(x, float):
                x = numpy.array([x])
                vx = numpy.array([vx])
            xp = get_namespace(x, vx)
            if xp is not numpy:
                x, vx = xp.asarray(x), xp.asarray(vx)
                return self._actionsFreqsAngles_backend(x, vx)
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
                            lambda t: (
                                2.0
                                * t
                                / numpy.sqrt(
                                    2.0
                                    * (
                                        E
                                        - evaluatelinearPotentials(
                                            self._pot, xmax - t**2.0, use_physical=False
                                        )
                                    )
                                )
                            ),
                            0,
                            numpy.sqrt(xmax),
                        )[0]
                    )
                    angle[ii] = integrate.quad(
                        lambda xi: (
                            1.0
                            / numpy.sqrt(
                                2.0
                                * (
                                    E
                                    - evaluatelinearPotentials(
                                        self._pot, xi, use_physical=False
                                    )
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

    # ------------------------------------------------ backend (jax/torch) path
    # Vectorised, differentiable mirror of the per-object numpy scipy loop above
    # (the numpy path is byte-identical and untouched; jax/torch inputs branch
    # here). The turning point xmax (E == Phi(xmax)) uses the shared
    # backend.optimize.brentq with a fixed-schedule expanding bracket; the
    # J/Omega/angle integrals use backend.quadrature.fixed_quad with the same
    # x = xmax - t^2 substitution the numpy Omega integral uses, the radicand
    # 2(E - Phi(x)) clipped >= 0 before the sqrt (sqrt'(0)=inf would NaN-poison
    # reverse-mode AD). This is the 1D vertical analog of actionAngleSpherical's
    # radial Jr/Or/ar.

    def _E_backend(self, x, vx):
        return vx**2.0 / 2.0 + evaluatelinearPotentials(
            self._pot, x, use_physical=False
        )

    def _calc_xmax_backend(self, x, vx, E):
        """Vectorised turning point xmax (E == Phi(xmax)) via backend brentq."""
        from ..backend.optimize import brentq as _backend_brentq

        xp = get_namespace(x)
        absx = xp.abs(x)

        def f(xm, E_):
            return E_ - evaluatelinearPotentials(self._pot, xm, use_physical=False)

        # Expanding upper bracket: double from 2|x| (1e-5 at x=0) until f <= 0.
        xend = xp.where(absx > 0.0, 2.0 * absx, 1e-5 * xp.ones_like(absx))
        for _ in range(80):
            xend = xp.where(f(xend, E) > 0.0, xend * 2.0, xend)
        # Lower bracket |x| (f(|x|) = vx^2/2 >= 0); at the turning point vx==0,
        # |x| IS xmax, so use a safe lower end there (dead-branch guard) and
        # override below.
        at_turn = vx == 0.0
        lo = xp.where(at_turn, absx / 2.0, absx)
        xmax = _backend_brentq(f, lo, xend, args=(E,))
        return xp.where(at_turn, absx, xmax)

    def _calc_J_backend(self, xp, xmax, E):
        """J = (2/pi) int_0^xmax sqrt(2(E-Phi)) dx via x = xmax - t^2."""
        from ..backend.quadrature import fixed_quad

        lim = xp.sqrt(xmax)

        def integrand(s):  # s: (n,) -> (N, n); t = lim*s, x = xmax - t^2
            t = lim[:, None] * s[None, :]
            xi = xmax[:, None] - t**2.0
            rad = 2.0 * (
                E[:, None] - evaluatelinearPotentials(self._pot, xi, use_physical=False)
            )
            rad = xp.where(rad > 0.0, rad, xp.zeros_like(rad))  # clip (AD guard)
            return xp.sqrt(rad) * 2.0 * t * lim[:, None]  # dx = 2t dt, dt = lim ds

        # device=: scalar limits, so anchor the GL nodes on the input device (xmax)
        # -- else torch raises on CUDA input. No-op on numpy (device_of -> None).
        return (
            2.0
            / numpy.pi
            * fixed_quad(
                xp, integrand, 0.0, 1.0, n=_BACKEND_GL_ORDER, device=device_of(xmax)
            )
        )

    def _calc_omega_backend(self, xp, xmax, E):
        """Omega = pi/2 / int_0^xmax dx/sqrt(2(E-Phi)) via x = xmax - t^2."""
        from ..backend.quadrature import fixed_quad

        lim = xp.sqrt(xmax)

        def integrand(s):
            t = lim[:, None] * s[None, :]
            xi = xmax[:, None] - t**2.0
            rad = 2.0 * (
                E[:, None] - evaluatelinearPotentials(self._pot, xi, use_physical=False)
            )
            rad = xp.where(rad > 0.0, rad, xp.ones_like(rad))  # 2t->0 there anyway
            return 2.0 * t / xp.sqrt(rad) * lim[:, None]

        return (
            numpy.pi
            / 2.0
            / fixed_quad(
                xp, integrand, 0.0, 1.0, n=_BACKEND_GL_ORDER, device=device_of(xmax)
            )
        )

    def _calc_angle_backend(self, xp, x, vx, xmax, E, Omega):
        """angle = Omega * int_0^|x| dx/sqrt(2(E-Phi)), then (x,vx)-quadrant fix.

        Integrate via xi = xmax - t^2 (t from sqrt(xmax-|x|) to sqrt(xmax)) so the
        1/sqrt turning-point singularity at xi=xmax is regularised by the 2t
        Jacobian (the orbit can sit arbitrarily close to xmax as vx -> 0).
        """
        from ..backend.quadrature import fixed_quad

        absx = xp.abs(x)
        lo = xp.sqrt(xp.where(xmax > absx, xmax - absx, xp.zeros_like(xmax)))
        hi = xp.sqrt(xmax)
        span = hi - lo

        def integrand(s):  # t = lo + span*s, xi = xmax - t^2
            t = lo[:, None] + span[:, None] * s[None, :]
            xi = xmax[:, None] - t**2.0
            rad = 2.0 * (
                E[:, None] - evaluatelinearPotentials(self._pot, xi, use_physical=False)
            )
            rad = xp.where(rad > 0.0, rad, xp.ones_like(rad))
            return 2.0 * t / xp.sqrt(rad) * span[:, None]  # dx = 2t dt, dt = span ds

        angle = Omega * fixed_quad(
            xp, integrand, 0.0, 1.0, n=_BACKEND_GL_ORDER, device=device_of(xmax)
        )
        # Quadrant assembly (mirror the numpy masked writes; disjoint conditions).
        angle = xp.where((x >= 0.0) & (vx < 0.0), numpy.pi - angle, angle)
        angle = xp.where((x < 0.0) & (vx <= 0.0), numpy.pi + angle, angle)
        angle = xp.where((x < 0.0) & (vx > 0.0), 2.0 * numpy.pi - angle, angle)
        return angle % (2.0 * numpy.pi)

    def _unbound_backend(self, xp, xmax, E):
        """Mask of unbound orbits: the bracket expansion found no turning point,
        so E still exceeds Phi(xmax). The numpy path returns the 9999.99 sentinel
        (calcxmax == -9999.99 on overflow); match it for the vectorised path."""
        return (
            E - evaluatelinearPotentials(self._pot, xmax, use_physical=False)
        ) > 1e-7

    def _evaluate_backend(self, x, vx):
        xp = get_namespace(x)
        E = self._E_backend(x, vx)
        xmax = self._calc_xmax_backend(x, vx, E)
        J = self._calc_J_backend(xp, xmax, E)
        return xp.where(self._unbound_backend(xp, xmax, E), 9999.99, J)

    def _actionsFreqs_backend(self, x, vx):
        xp = get_namespace(x)
        E = self._E_backend(x, vx)
        xmax = self._calc_xmax_backend(x, vx, E)
        unbound = self._unbound_backend(xp, xmax, E)
        return (
            xp.where(unbound, 9999.99, self._calc_J_backend(xp, xmax, E)),
            xp.where(unbound, 9999.99, self._calc_omega_backend(xp, xmax, E)),
        )

    def _actionsFreqsAngles_backend(self, x, vx):
        xp = get_namespace(x)
        E = self._E_backend(x, vx)
        xmax = self._calc_xmax_backend(x, vx, E)
        J = self._calc_J_backend(xp, xmax, E)
        Omega = self._calc_omega_backend(xp, xmax, E)
        angle = self._calc_angle_backend(xp, x, vx, xmax, E, Omega)
        unbound = self._unbound_backend(xp, xmax, E)
        # numpy unbound path sets angle=9999.99 BEFORE `angle *= Omega` (=9999.99)
        # then `% 2pi`, so the sentinel angle is (9999.99**2) % (2pi), not 9999.99.
        return (
            xp.where(unbound, 9999.99, J),
            xp.where(unbound, 9999.99, Omega),
            xp.where(unbound, (9999.99 * 9999.99) % (2.0 * numpy.pi), angle),
        )

    def calcxmax(self, x, vx, E=None):
        """
        Calculate the maximum height

        Parameters
        ----------
        x : float
            Position
        vx : float
            Velocity
        E : float, optional
            Energy (default is None)

        Returns
        -------
        float
            Maximum height

        Notes
        -----
        - 2012-06-01 - Written - Bovy (IAS)
        - 2018-05-19 - Re-written for new framework - Bovy (UofT)
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
                    lambda xm: (
                        E - evaluatelinearPotentials(self._pot, xm, use_physical=False)
                    ),
                    xstart,
                    xend,
                    xtol=1e-14,
                )
                while (
                    E - evaluatelinearPotentials(self._pot, xmax, use_physical=False)
                ) < 0:
                    xmax -= 1e-14
        return xmax
