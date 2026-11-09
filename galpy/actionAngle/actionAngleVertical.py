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
from scipy import optimize

from ..potential.linearPotential import evaluatelinearForces, evaluatelinearPotentials
from ..potential.Potential import _check_potential_list_and_deprecate
from .actionAngle import actionAngle

# the fixed-order Gauss-Legendre rules of the vectorized quadratures, on [0, 1]
# for the librations and on [-1, 1] for the force integral near the turning
# point, and the relative size of E - Phi below which that force form is used
# (the direct difference then carries a round-off above 1e-12 of its value)
_GL_ORDER = 40
_FORCE_ORDER = 8
_FORCE_BELOW = 1e-4
# the relative accuracy asked of E - Phi: where the potential's own absolute
# round-off (measured once per instance by probing it against the force
# integral near the midplane) exceeds this fraction of E - Phi, the force form
# is used instead of the direct difference
_FORCE_RTOL = 1e-12
_GL_NODES01, _GL_WEIGHTS01 = (
    0.5 * (v + 1.0) if ii == 0 else 0.5 * v
    for ii, v in enumerate(numpy.polynomial.legendre.leggauss(_GL_ORDER))
)
_FORCE_NODES, _FORCE_WEIGHTS = numpy.polynomial.legendre.leggauss(_FORCE_ORDER)


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
        - 2026-11-09 - Vectorized over the points, by fixed-order Gauss-Legendre quadrature in a turning-point-regular variable - Bovy (UofT)
        """
        if len(args) == 2:  # x,vx
            x, vx, E, xmax, bound = self._points(*args)
            J = numpy.full(x.shape, 9999.99)
            J[bound] = self._JOmega(xmax[bound], E[bound], freq=False)
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
        - 2026-11-09 - Vectorized over the points - Bovy (UofT)
        """
        if len(args) == 2:  # x,vx
            x, vx, E, xmax, bound = self._points(*args)
            J = numpy.full(x.shape, 9999.99)
            Omega = numpy.full(x.shape, 9999.99)
            J[bound], Omega[bound] = self._JOmega(xmax[bound], E[bound])
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
        - 2026-11-09 - Vectorized over the points - Bovy (UofT)
        """
        if len(args) == 2:  # x,vx
            x, vx, E, xmax, bound = self._points(*args)
            J = numpy.full(x.shape, 9999.99)
            Omega = numpy.full(x.shape, 9999.99)
            angle = numpy.full(x.shape, 9999.99)
            J[bound], Omega[bound] = self._JOmega(xmax[bound], E[bound])
            angle[bound] = self._angle(x[bound], xmax[bound], E[bound])
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

    # ---------- the vectorized quadratures
    #
    # Every integral along the libration is taken in t = sqrt(xmax - x), in
    # which the integrand is regular at the turning point (the square-root
    # vanishing of the momentum is absorbed by dx = 2 t dt), by fixed-order
    # Gauss-Legendre quadrature over all points at once. Near the turning
    # point the difference E - Phi(x), which is Phi(xmax) - Phi(x), is
    # computed as the integral of the force over [x, xmax], because the
    # direct difference loses digits there for a torus that is small
    # compared to the scale of the potential.
    def _points(self, x, vx):
        """The points as float arrays, their energies, their turning points, and which are bound"""
        x = numpy.atleast_1d(numpy.asarray(x, dtype="float"))
        vx = numpy.atleast_1d(numpy.asarray(vx, dtype="float"))
        x, vx = numpy.broadcast_arrays(x, vx)
        Phix = evaluatelinearPotentials(self._pot, x, use_physical=False)
        E = vx**2.0 / 2.0 + Phix
        Phi0 = evaluatelinearPotentials(self._pot, 0.0, use_physical=False)
        # a torus whose energy above the minimum is within the potential's
        # round-off, amplified to the accuracy asked, has its turning point
        # found from the force as well
        small = (E - Phi0) < self._phi_roundoff() / _FORCE_RTOL
        xmax = self._xmax(x, vx, E)
        xmax[small] = self._xmax_small(x[small], vx[small])
        return x, vx, E, xmax, xmax != -9999.99

    def _phi_roundoff(self):
        """The absolute round-off of the potential's evaluation, measured once:
        the largest discrepancy between Phi(x) - Phi(0) and the integral of
        the force over [0, x] for a set of small x, where both are exact but
        for round-off (a potential that is itself a difference, such as a
        vertical potential measured from the midplane, carries the round-off
        of the terms it subtracts)"""
        if not hasattr(self, "_phi_roundoff_value"):
            xs = numpy.logspace(-6.0, -2.0, 17)
            direct = evaluatelinearPotentials(
                self._pot, xs, use_physical=False
            ) - evaluatelinearPotentials(self._pot, 0.0, use_physical=False)
            byforce = self._dPhi(numpy.zeros_like(xs), xs)
            self._phi_roundoff_value = float(numpy.max(numpy.fabs(direct - byforce)))
        return self._phi_roundoff_value

    def _xmax(self, x, vx, E):
        """The turning points of the points: |x| where the velocity vanishes;
        otherwise the root of Phi(xm) = E beyond |x|, bracketed by doubling
        from |x| and found by bisection to round-off; -9999.99 for a point
        that is not bound (no bracket within 2^80 |x|, as calcxmax)"""
        absx = numpy.fabs(x)
        xmax = numpy.array(absx)
        todo = vx != 0.0
        if not numpy.any(todo):
            return xmax
        lo = absx[todo]
        hi = numpy.where(lo > 0.0, 2.0 * lo, 1e-5)
        Et = E[todo]
        for _ in range(80):
            below = evaluatelinearPotentials(self._pot, hi, use_physical=False) < Et
            if not numpy.any(below):
                break
            lo = numpy.where(below, hi, lo)
            hi = numpy.where(below, 2.0 * hi, hi)
        unbound = evaluatelinearPotentials(self._pot, hi, use_physical=False) < Et
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            below = evaluatelinearPotentials(self._pot, mid, use_physical=False) < Et
            lo = numpy.where(below, mid, lo)
            hi = numpy.where(below, hi, mid)
        root = 0.5 * (lo + hi)
        xmax[todo] = numpy.where(unbound, -9999.99, root)
        return xmax

    def _rad(self, t2, xmax, E):
        """2 [E - Phi(x)] at x = xmax - t2, for arrays of points (rows) and
        quadrature nodes (columns): the direct difference where it keeps its
        digits, and otherwise, where E - Phi is small against E or Phi, 2 t2
        times the mean force over [x, xmax] by Gauss-Legendre quadrature,
        which is Phi(xmax) - Phi(x) without cancellation (a torus small
        compared to the potential's scale uses it at every node)"""
        xi = xmax[:, None] - t2
        Phi = evaluatelinearPotentials(self._pot, xi, use_physical=False)
        diff = E[:, None] - Phi
        near = diff < numpy.maximum(
            _FORCE_BELOW * numpy.maximum(numpy.fabs(E[:, None]), numpy.fabs(Phi)),
            self._phi_roundoff() / _FORCE_RTOL,
        )
        rad = 2.0 * diff
        if numpy.any(near):
            xn = xi[near]
            dx = numpy.broadcast_to(t2, xi.shape)[near]
            rad[near] = 2.0 * self._dPhi(xn, xn + dx)
        return numpy.clip(rad, 0.0, None)

    def _dPhi(self, xa, xb):
        """Phi(xb) - Phi(xa) as the integral of minus the force over [xa, xb],
        by Gauss-Legendre quadrature, for arrays of end points"""
        dx = xb - xa
        xx = xa[..., None] + 0.5 * dx[..., None] * (_FORCE_NODES + 1.0)
        return (
            0.5
            * dx
            * numpy.sum(
                _FORCE_WEIGHTS
                * (-evaluatelinearForces(self._pot, xx, use_physical=False)),
                axis=-1,
            )
        )

    def _xmax_small(self, x, vx):
        """The turning point of a small torus: the root beyond |x| of
        vx^2 / 2 = Phi(xm) - Phi(|x|), the right-hand side from the force, by
        bracketing and bisection"""
        absx = numpy.fabs(x)
        if not len(absx):
            return absx
        lo = numpy.array(absx)
        hi = numpy.where(lo > 0.0, 2.0 * lo, 1e-5)
        target = 0.5 * vx**2.0
        for _ in range(200):
            below = self._dPhi(absx, hi) < target
            if not numpy.any(below):
                break
            lo = numpy.where(below, hi, lo)
            hi = numpy.where(below, 2.0 * hi, hi)
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            below = self._dPhi(absx, mid) < target
            lo = numpy.where(below, mid, lo)
            hi = numpy.where(below, hi, mid)
        return numpy.where(vx == 0.0, absx, 0.5 * (lo + hi))

    def _JOmega(self, xmax, E, freq=True):
        """J = (2/pi) int_0^xmax sqrt(2 [E - Phi]) dx and, if asked, Omega =
        (pi/2) / int_0^xmax dx / sqrt(2 [E - Phi]), both with x = xmax - t^2
        on the same nodes"""
        lim = numpy.sqrt(xmax)
        t = lim[:, None] * _GL_NODES01[None, :]
        rad = self._rad(t**2.0, xmax, E)
        J = (
            2.0
            / numpy.pi
            * numpy.sum(_GL_WEIGHTS01[None, :] * numpy.sqrt(rad) * 2.0 * t, axis=1)
            * lim
        )
        if not freq:
            return J
        with numpy.errstate(divide="ignore", invalid="ignore"):
            integrand = numpy.where(rad > 0.0, 2.0 * t / numpy.sqrt(rad), 0.0)
        return J, numpy.pi / 2.0 / (
            numpy.sum(_GL_WEIGHTS01[None, :] * integrand, axis=1) * lim
        )

    def _angle(self, x, xmax, E):
        """int_0^|x| dx / sqrt(2 [E - Phi]), the angle over the frequency, with
        x = xmax - t^2 from t = sqrt(xmax - |x|) to sqrt(xmax)"""
        lo = numpy.sqrt(numpy.clip(xmax - numpy.fabs(x), 0.0, None))
        span = numpy.sqrt(xmax) - lo
        t = lo[:, None] + span[:, None] * _GL_NODES01[None, :]
        rad = self._rad(t**2.0, xmax, E)
        with numpy.errstate(divide="ignore", invalid="ignore"):
            integrand = numpy.where(rad > 0.0, 2.0 * t / numpy.sqrt(rad), 0.0)
        return numpy.sum(_GL_WEIGHTS01[None, :] * integrand, axis=1) * span

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
