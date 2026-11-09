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

from ..backend import (
    as_numpy,
    asarray_on_device,
    device_of,
    get_namespace,
    lazy_cond,
    param_grad_namespace,
    promote_scalars,
    use,
)
from ..backend._namespaces import (
    has_concrete_truth_value,
    namespace_from_arrays,
    set_at,
    stop_gradient,
    under_trace,
)
from ..backend.optimize import brentq as _backend_brentq
from ..backend.optimize import iterate_to_fixed_point
from ..potential.linearPotential import (
    _evaluatelinearForces,
    _evaluatelinearPotentials,
    evaluatelinearPotentials,
)
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

    # a scalar point's outputs: numpy's shape for it
    _scalar_point_ndim = 1

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
            x, vx = args
            xp = get_namespace(x, vx)
            if xp is not numpy:
                x, vx = promote_scalars(xp, x, vx)
                return self._evaluate_backend(x, vx)
            x, vx, E, xmax, bound = self._points(x, vx)
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
            x, vx = args
            xp = get_namespace(x, vx)
            if xp is not numpy:
                x, vx = promote_scalars(xp, x, vx)
                return self._actionsFreqs_backend(x, vx)
            x, vx, E, xmax, bound = self._points(x, vx)
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
            x, vx = args
            xp = get_namespace(x, vx)
            if xp is not numpy:
                x, vx = promote_scalars(xp, x, vx)
                return self._actionsFreqsAngles_backend(x, vx)
            x, vx, E, xmax, bound = self._points(x, vx)
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
    def _per_object(self):
        """A vertical potential with one radius per object, R of shape (N,),
        which broadcasts against the leading axis of its z"""
        return numpy.ndim(getattr(self._pot, "_R", None)) > 0

    def _call(self, func, x, **kwargs):
        """The potential or force at the array x: in one call where the
        potential's implementation takes arrays, and element by element where
        it does not (one that branches on a scalar: numpy and jax raise a
        ValueError or TypeError there, torch a RuntimeError)"""
        try:
            return func(self._pot, x, **kwargs)
        except (ValueError, TypeError, RuntimeError):
            xp = get_namespace(x)
            flat = xp.reshape(x * 1.0, (-1,))
            vals = [func(self._pot, flat[k], **kwargs) for k in range(flat.shape[0])]
            return xp.reshape(xp.stack(vals), xp.asarray(x).shape)

    def _points(self, x, vx):
        """The points as float arrays, their energies, their turning points, and which are bound"""
        x = numpy.atleast_1d(numpy.asarray(x, dtype="float"))
        vx = numpy.atleast_1d(numpy.asarray(vx, dtype="float"))
        x, vx = numpy.broadcast_arrays(x, vx)
        Phix = self._call(evaluatelinearPotentials, x, use_physical=False)
        E = vx**2.0 / 2.0 + Phix
        Phi0 = self._call(evaluatelinearPotentials, 0.0, use_physical=False)
        # a torus whose energy above the minimum is within the potential's
        # round-off, amplified to the accuracy asked, has its turning point
        # found from the force as well
        small = (E - Phi0) < self._phi_roundoff() / _FORCE_RTOL
        xmax = self._xmax(x, vx, E)
        xmax[small] = self._xmax_small(x[small], vx[small])
        return x, vx, E, xmax, xmax != -9999.99

    def _phi_roundoff(self, ref=None):
        """The absolute round-off of the potential's evaluation, measured once:
        the largest discrepancy between Phi(x) - Phi(0) and the integral of
        the force over [0, x] for a set of small x, where both are exact but
        for round-off (a potential that is itself a difference, such as a
        vertical potential measured from the midplane, carries the round-off
        of the terms it subtracts). For a potential with backend parameters it
        is measured there (on ref's device), and afresh while they are
        traced"""
        if getattr(self, "_phi_roundoff_value", None) is not None:
            return self._phi_roundoff_value
        # a per-object vertical potential (R of shape (N,)) broadcasts against
        # z's leading axis: one row of heights per object
        R = getattr(self._pot, "_R", None)
        batch = self._per_object()
        xp = param_grad_namespace(self._pot, any_backend=True)
        if xp is None and batch:
            xp = namespace_from_arrays((R,))
        if xp is None:  # numpy's
            with use("numpy", force=True):
                xs = numpy.logspace(-6.0, -2.0, 17)
                if batch:
                    xs = numpy.tile(xs, (len(R), 1))
                direct = self._call(
                    evaluatelinearPotentials, xs, use_physical=False
                ) - self._call(
                    evaluatelinearPotentials,
                    0.0 * xs if batch else 0.0,
                    use_physical=False,
                )
                byforce = self._dPhi(numpy.zeros_like(xs), xs)
                self._phi_roundoff_value = float(
                    numpy.max(numpy.fabs(direct - byforce))
                )
            return self._phi_roundoff_value
        dev = None if ref is None else device_of(ref)
        xs = asarray_on_device(xp, numpy.logspace(-6.0, -2.0, 17), dev)
        if batch:
            xs = xs[None, :] + 0.0 * xp.reshape(R, (-1, 1))
        dev = (
            self._call(_evaluatelinearPotentials, xs)
            - self._call(_evaluatelinearPotentials, 0.0 * xs)
            - self._dPhi(0.0 * xs, xs)
        )
        out = xp.max(xp.abs(stop_gradient(dev)))
        if has_concrete_truth_value(out):
            self._phi_roundoff_value = float(as_numpy(out))
        return out

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
            below = self._call(evaluatelinearPotentials, hi, use_physical=False) < Et
            if not numpy.any(below):
                break
            lo = numpy.where(below, hi, lo)
            hi = numpy.where(below, 2.0 * hi, hi)
        unbound = self._call(evaluatelinearPotentials, hi, use_physical=False) < Et
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            below = self._call(evaluatelinearPotentials, mid, use_physical=False) < Et
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
        compared to the potential's scale uses it at every node). numpy
        takes the force form at those nodes only; a backend takes it at
        every node and selects (a where, differentiable and traceable)"""
        xp = get_namespace(t2, xmax, E)
        xi = xmax[..., None] - t2
        Phi = self._call(_evaluatelinearPotentials, xi)
        diff = E[..., None] - Phi
        # diff < max(the relative floor, the round-off floor)
        near = (diff < _FORCE_BELOW * xp.maximum(xp.abs(E[..., None]), xp.abs(Phi))) | (
            diff < self._phi_roundoff(xi) / _FORCE_RTOL
        )
        rad = 2.0 * diff
        if under_trace(xi, t2) or self._per_object():
            # keeps the shape, which a per-object potential broadcasts against
            rad = xp.where(near, 2.0 * self._dPhi(xi, xi + t2), rad)
        elif xp is numpy:
            if numpy.any(near):
                xn = xi[near]
                dx = numpy.broadcast_to(t2, xi.shape)[near]
                rad[near] = 2.0 * self._dPhi(xn, xn + dx)
        elif bool(xp.any(near)):  # eager: at the near nodes only, as numpy
            xn = xi[near]
            rad = set_at(
                xp, rad, near, 2.0 * self._dPhi(xn, xn + (t2 + 0.0 * xi)[near])
            )
        if xp is numpy:
            return numpy.clip(rad, 0.0, None)
        # a constant zero: no derivative into sqrt(0) below
        return xp.where(rad > 0.0, rad, xp.zeros_like(rad))

    def _dPhi(self, xa, xb):
        """Phi(xb) - Phi(xa) as the integral of minus the force over [xa, xb],
        by Gauss-Legendre quadrature, for arrays of end points"""
        xp = get_namespace(xa, xb)
        dev = device_of(xa, xb)
        nodes, weights = (
            asarray_on_device(xp, t, dev) for t in (_FORCE_NODES, _FORCE_WEIGHTS)
        )
        dx = xb - xa
        xx = xa[..., None] + 0.5 * dx[..., None] * (nodes + 1.0)
        return (
            0.5
            * dx
            * xp.sum(weights * (-self._call(_evaluatelinearForces, xx)), axis=-1)
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

    def _gl01(self, xp, ref):
        """The fixed-order Gauss-Legendre nodes and weights on [0, 1], on ref's
        device"""
        dev = device_of(ref)
        return (asarray_on_device(xp, t, dev) for t in (_GL_NODES01, _GL_WEIGHTS01))

    @staticmethod
    def _inv_sqrt_integrand(xp, t, rad):
        """2 t / sqrt(rad), zero where rad vanishes (the turning point, where
        2 t vanishes too), with no derivative through the dead branch"""
        pos = rad > 0.0
        return xp.where(pos, 2.0 * t / xp.sqrt(xp.where(pos, rad, 1.0)), 0.0 * t)

    def _JOmega(self, xmax, E, freq=True):
        """J = (2/pi) int_0^xmax sqrt(2 [E - Phi]) dx and, if asked, Omega =
        (pi/2) / int_0^xmax dx / sqrt(2 [E - Phi]), both with x = xmax - t^2
        on the same nodes"""
        xp = get_namespace(xmax, E)
        nodes, weights = self._gl01(xp, xmax)
        lim = xp.sqrt(xmax)
        t = lim[..., None] * nodes
        rad = self._rad(t**2.0, xmax, E)
        J = 2.0 / numpy.pi * xp.sum(weights * xp.sqrt(rad) * 2.0 * t, axis=-1) * lim
        if not freq:
            return J
        integrand = self._inv_sqrt_integrand(xp, t, rad)
        return J, numpy.pi / 2.0 / (xp.sum(weights * integrand, axis=-1) * lim)

    def _angle(self, x, xmax, E):
        """int_0^|x| dx / sqrt(2 [E - Phi]), the angle over the frequency, with
        x = xmax - t^2 from t = sqrt(xmax - |x|) to sqrt(xmax)"""
        xp = get_namespace(x, xmax, E)
        nodes, weights = self._gl01(xp, xmax)
        gap = xmax - xp.abs(x)
        lo = xp.sqrt(xp.where(gap > 0.0, gap, xp.zeros_like(gap)))
        span = xp.sqrt(xmax) - lo
        t = lo[..., None] + span[..., None] * nodes
        rad = self._rad(t**2.0, xmax, E)
        integrand = self._inv_sqrt_integrand(xp, t, rad)
        return xp.sum(weights * integrand, axis=-1) * span

    # ------------------------------------------------ backend (jax/torch) path
    # The same turning points and quadratures as numpy's, vectorized and
    # differentiable: the turning point xmax (E == Phi(xmax)) by the backend
    # brentq from a fixed-schedule expanding bracket (its final Newton step
    # carries the implicit derivative), from the force integral for a torus
    # within the potential's round-off (as _xmax_small); the action,
    # frequency, and angle by the shared quadratures above (_JOmega, _angle,
    # with the force form of E - Phi as a where).

    def _E_backend(self, x, vx):
        return vx**2.0 / 2.0 + self._call(_evaluatelinearPotentials, x)

    def _calc_xmax_backend(self, x, vx, E):
        """Vectorised turning point xmax (E == Phi(xmax)) via backend brentq;
        a torus within the potential's round-off from the force (as
        _points)"""
        xp = get_namespace(x)
        absx = xp.abs(x)

        def f(xm, E_):
            return E_ - self._call(_evaluatelinearPotentials, xm)

        xmax = self._xmax_root_backend(xp, absx, vx, f, (E,))
        Phi0 = self._call(_evaluatelinearPotentials, 0.0 * x)
        small = (E - Phi0) < self._phi_roundoff(x) / _FORCE_RTOL
        if under_trace(x, vx, E) or x.ndim == 0 or self._per_object():
            return lazy_cond(
                xp,
                xp.any(small),
                lambda: xp.where(small, self._xmax_small_backend(xp, x, vx), xmax),
                lambda: xmax,
            )
        if not bool(xp.any(small)):
            return xmax
        # eager: the small tori only, as numpy
        return set_at(
            xp, xmax, small, self._xmax_small_backend(xp, x[small], vx[small])
        )

    def _xmax_root_backend(self, xp, absx, vx, f, args, ndouble=80, harmonic=False):
        """The root beyond |x| of f(xm, *args), positive at |x|: the upper
        bracket doubled from 2|x| (1e-5 at x = 0) until f <= 0 there, each end
        frozen once bracketed so that the doubling stops there (a while_loop
        under a jax trace; the bracket carries no derivative, the root's
        implicit step does), then the backend brentq -- from the harmonic
        estimate by safeguarded Newton steps for a torus within round-off of
        the minimum (harmonic=True), whose f is a force integral, else by
        bisection; |x| itself at a turning point (vx == 0)"""
        sargs = tuple(stop_gradient(a) for a in args)
        xend = iterate_to_fixed_point(
            lambda xe: xp.where(f(xe, *sargs) > 0.0, xe * 2.0, xe),
            stop_gradient(xp.where(absx > 0.0, 2.0 * absx, 1e-5 + 0.0 * absx)),
            ndouble,
        )
        # at the turning point |x| IS xmax: a safe lower end there (dead-branch
        # guard), overridden below
        at_turn = vx == 0.0
        lo = xp.where(at_turn, absx / 2.0, absx)
        guess = None
        if harmonic:
            # vx^2 / 2 = k (xm^2 - x^2) / 2 with k = -F / x at the bracket's end
            sa, sv = stop_gradient(absx), stop_gradient(vx)
            k = -self._call(_evaluatelinearForces, xend) / xend
            guess = xp.sqrt(sa**2.0 + sv**2.0 / xp.where(k > 0.0, k, 1.0))
            guess = xp.where(k > 0.0, guess, 0.5 * (lo + xend))
        root = _backend_brentq(
            f, lo, xend, args=args, guess=guess, newton_steps=12 if harmonic else 10
        )
        # the implicit derivative once more, at the polished root: the slope at
        # the bisection's root (its ends an absolute 2e-12 apart) is off by a
        # relative ~2e-12 / xmax, 5e-6 for a turning point at 4e-8
        root = _backend_brentq(
            f, lo, xend, args=args, guess=stop_gradient(root), newton_steps=0
        )
        return xp.where(at_turn, absx, root)

    def _xmax_small_backend(self, xp, x, vx):
        """_xmax_small on a backend: the root beyond |x| of vx^2 / 2 = Phi(xm)
        - Phi(|x|), the right-hand side from the force"""
        absx = xp.abs(x)

        def f(xm, a, tgt):
            return tgt - self._dPhi(a, xm)

        return self._xmax_root_backend(
            xp, absx, vx, f, (absx, 0.5 * vx**2.0), ndouble=200, harmonic=True
        )

    def _unbound_backend(self, xp, xmax, E):
        """Mask of unbound orbits: the bracket expansion found no turning point,
        so E still exceeds Phi(xmax). The numpy path returns the 9999.99 sentinel
        (calcxmax == -9999.99 on overflow); match it for the vectorised path."""
        return (E - self._call(_evaluatelinearPotentials, xmax)) > 1e-7

    def _evaluate_backend(self, x, vx):
        xp = get_namespace(x)
        E = self._E_backend(x, vx)
        xmax = self._calc_xmax_backend(x, vx, E)
        J = self._JOmega(xmax, E, freq=False)
        return xp.where(self._unbound_backend(xp, xmax, E), 9999.99, J)

    def _actionsFreqs_backend(self, x, vx):
        xp = get_namespace(x)
        E = self._E_backend(x, vx)
        xmax = self._calc_xmax_backend(x, vx, E)
        unbound = self._unbound_backend(xp, xmax, E)
        J, Omega = self._JOmega(xmax, E)
        return xp.where(unbound, 9999.99, J), xp.where(unbound, 9999.99, Omega)

    def _actionsFreqsAngles_backend(self, x, vx):
        xp = get_namespace(x)
        E = self._E_backend(x, vx)
        xmax = self._calc_xmax_backend(x, vx, E)
        J, Omega = self._JOmega(xmax, E)
        angle = Omega * self._angle(x, xmax, E)
        # quadrant assembly (mirror the numpy masked writes; disjoint conditions)
        angle = xp.where((x >= 0.0) & (vx < 0.0), numpy.pi - angle, angle)
        angle = xp.where((x < 0.0) & (vx <= 0.0), numpy.pi + angle, angle)
        angle = xp.where((x < 0.0) & (vx > 0.0), 2.0 * numpy.pi - angle, angle)
        angle = angle % (2.0 * numpy.pi)
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
