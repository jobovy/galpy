# Class that implements isotropic spherical DFs computed using the Eddington
# formula
import numpy
from scipy import integrate, interpolate, optimize

from ..backend import as_numpy, get_namespace, is_backend_array, resolve_namespace
from ..backend._namespaces import has_concrete_truth_value
from ..backend.interpolate import Spline1D
from ..backend.quadrature import fixed_quad
from ..potential import CompositePotential, evaluateR2derivs
from ..potential.Potential import _evaluatePotentials, _evaluateRforces
from ..util import conversion
from .sphericaldf import _handle_rmin, isotropicsphericaldf, sphericaldf

# Backend Gauss-Legendre order for the two fE half-integrals; N=100 agrees with
# scipy adaptive quad to ~6e-8 -- higher N drifts as small-r nodes cluster into
# the r=rphi turning-point fp-cancellation zone (see tests/test_backend_eddingtondf.py).
_QUAD_N_FE = 100


class eddingtondf(isotropicsphericaldf):
    """Class that implements isotropic spherical DFs computed using the Eddington formula

    .. math::

        f(\\mathcal{E}) = \\frac{1}{\\sqrt{8}\\,\\pi^2}\\,\\left[\\int_0^\\mathcal{E}\\mathrm{d}\\Psi\\,\\frac{1}{\\sqrt{\\mathcal{E}-\\Psi}}\\,\\frac{\\mathrm{d}^2\\rho}{\\mathrm{d}\\Psi^2} +\\frac{1}{\\sqrt{\\mathcal{E}}}\\,\\frac{\\mathrm{d}\\rho}{\\mathrm{d}\\Psi}\\Bigg|_{\\Psi=0}\\right]\\,,

    where :math:`\\Psi = -\\Phi+\\Phi(\\infty)` is the relative potential, :math:`\\mathcal{E} = \\Psi-v^2/2` is the relative (binding) energy, and :math:`\\rho` is the density of the tracer population (not necessarily the density corresponding to :math:`\\Psi` according to the Poisson equation). Note that the second term on the right-hand side is currently assumed to be zero in the code.
    """

    def __init__(
        self, pot=None, denspot=None, rmax=1e4, rmin=None, scale=None, ro=None, vo=None
    ):
        """
        Initialize an isotropic distribution function computed using the Eddington inversion.

        Parameters
        ----------
        pot : Potential instance or a combined potential formed using addition (pot1+pot2+…)
            Represents the gravitational potential (assumed to be spherical).
        denspot : Potential instance or a combined potential formed using addition (pot1+pot2+…), optional
            Represents the density of the tracers (assumed to be spherical; if None, set equal to pot).
        rmax : float or Quantity, optional
            Maximum radius to consider. DF is cut off at E = Phi(rmax).
        rmin : float or Quantity, optional
            Minimum radius to consider. For divergent potentials (Phi(0) = -inf),
            this sets the inner boundary for the energy range. Auto-detected if
            not specified.
        scale : float or Quantity, optional
            Characteristic scale radius to aid sampling calculations. Optional and will also be overridden by value from pot if available.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2021-02-04 - Written - Bovy (UofT)

        """
        isotropicsphericaldf.__init__(
            self, pot=pot, denspot=denspot, rmax=rmax, scale=scale, ro=ro, vo=vo
        )

        # Handle rmin for divergent potentials
        self._rmin = _handle_rmin(
            rmin, self._pot, self._denspot, self._scale, self._ro, "eddingtondf"
        )

        self._dnudr = (
            self._denspot._ddensdr
            if not isinstance(self._denspot, CompositePotential)
            # builtin sum keeps r's shape (numpy.sum collapses the vectorized
            # backend node array); byte-identical for the scalar-r numpy path
            else lambda r: sum(p._ddensdr(r) for p in self._denspot)
        )
        self._d2nudr2 = (
            self._denspot._d2densdr2
            if not isinstance(self._denspot, CompositePotential)
            else lambda r: sum(p._d2densdr2(r) for p in self._denspot)
        )
        self._potInf = _evaluatePotentials(pot, self._rmax, 0)
        self._Emin = _evaluatePotentials(pot, self._rmin, 0)
        # inside jax.jit the potential has no concrete value (see _rInf, _rphi)
        self._jit = is_backend_array(self._Emin) and not has_concrete_truth_value(
            self._Emin == self._Emin
        )
        # Current calculation of the boundary term uses r -> inf limit
        try:
            # a discrete choice of the boundary radius, so bool() on the
            # namespace's own isfinite: numpy.isfinite cannot convert a
            # DIFFERENTIATED value, while bool() reads the concrete primal that
            # jax.grad and torch autograd both carry (see _handle_rmin).
            def _finite(v):
                # bool() on the namespace's own isfinite for a BACKEND value: a
                # DIFFERENTIATED value cannot go through numpy.isfinite, while
                # bool() reads the concrete primal jax.grad and torch autograd
                # carry. Gate on the VALUE, not the ambient namespace -- under a
                # forced backend these are often still plain floats and
                # torch.isfinite rejects a float.
                if not is_backend_array(v):
                    return bool(numpy.isfinite(v))
                return bool(resolve_namespace(v).isfinite(v))

            # NB the `and` short-circuits, and must keep doing so: evaluating
            # the forces at r=inf warns ("invalid value encountered in scalar
            # divide") for potentials where dnu/dr is already non-finite there,
            # which test_eddington_hernquist_no_warning forbids.
            if self._jit:
                # jax.jit: finiteness at infinity is undecidable without a value;
                # take the finite boundary radius (the r -> inf limit to ~1e-12)
                self._rInf = 1e12
            else:
                self._rInf = (
                    numpy.inf
                    if _finite(self._dnudr(numpy.inf))
                    and _finite(_evaluateRforces(self._pot, numpy.inf, 0))
                    else 1e12
                )
        except ZeroDivisionError:
            self._rInf = 1e12
        # Build interpolator r(pot), starting at rmin for divergent potentials
        self._rphi = self._setup_rphi_interpolator(
            r_a_min=resolve_namespace(self._scale).maximum(
                1e-6, self._rmin / self._scale
            )
            if self._jit
            else max(1e-6, self._rmin / self._scale)
        )

    def sample(
        self, R=None, z=None, phi=None, n=1, return_orbit=True, rmin=None, key=None
    ):
        # Slight over-write of superclass method to first build f(E) interp
        # No docstring so superclass' is used
        if rmin is None:
            rmin = self._rmin
        self._ensure_fE_interp()
        # isotropic: a backend key drives the general (no closed-form _icmf)
        # interp_linear inverse-CDF radial sampler + isotropic angles
        return sphericaldf.sample(
            self, R=R, z=z, phi=phi, n=n, return_orbit=return_orbit, rmin=rmin, key=key
        )

    def _rphi_root(self, E):
        """r(Phi = E) to machine precision, for Emin <= E < potInf, whose root
        [rmin, rmax] always brackets (an expanding finite one for rmax = inf)."""
        rmax = self._rmax
        if numpy.isinf(rmax):
            rmax = max(2.0 * self._rmin, self._scale)
            while _evaluatePotentials(self._pot, rmax, 0) < E and rmax < 1e300:
                rmax *= 2.0
        return optimize.brentq(
            lambda r: _evaluatePotentials(self._pot, r, 0) - E,
            self._rmin,
            rmax,
            xtol=1e-300,
            maxiter=500,
        )

    def _fE_integral(self, E, rphi):
        """The Eddington integral over r in [rphi, inf): r = rphi + t^2 up to
        rsplit = 2 rphi, log(r) on to the scale radius (the integrand's bulk,
        which a single t = 1/r quad over [0, 1/(2 rphi)] misses for rphi <<
        scale), t = 1/r beyond"""
        args = (self._pot, E, self._dnudr, self._d2nudr2)
        if rphi == 0.0:  # E = Emin = Phi(0): the endpoint limit
            # A tracer's density slope alone does not determine convergence:
            # the central force can cancel its leading contribution. With
            # r = t^2, the actual integrand is nonintegrable if it grows at
            # least as 1/t. Check its scaling over two small-r decades.
            ts = numpy.sqrt(self._scale) * numpy.array([1e-2, 1e-3, 1e-4])
            fs = [_fEintegrand_smallr(t, *args, 0.0) for t in ts]
            if (
                numpy.all(numpy.isfinite(fs))
                and numpy.fabs(fs[0]) > 0.0
                and numpy.fabs(fs[2]) >= 10.0 * numpy.fabs(fs[1])
                and numpy.fabs(fs[1]) >= 10.0 * numpy.fabs(fs[0])
            ):
                return numpy.copysign(numpy.inf, fs[2])
            rsplit = self._scale
        else:
            rsplit = 2.0 * rphi
        out = integrate.quad(
            lambda t: _fEintegrand_smallr(t, *args, rphi),
            0.0,
            numpy.sqrt(rsplit - rphi),
            points=[0.0],
        )[0]
        if rsplit < self._scale:
            out += integrate.quad(
                lambda u: numpy.exp(u) * _fEintegrand_raw(numpy.exp(u), *args),
                numpy.log(rsplit),
                numpy.log(self._scale),
            )[0]
            rsplit = self._scale
        return (
            out
            + integrate.quad(
                lambda t: _fEintegrand_larger(t, *args), 0.0, 1.0 / rsplit
            )[0]
        )

    def fE(self, E):
        """
        Calculate the energy portion of a DF computed using the Eddington inversion

        Parameters
        ----------
        E : float or Quantity
            The energy.

        Returns
        -------
        fE : ndarray
            The value of the energy portion of the DF.

        Notes
        -----
        - 2021-02-04 - Written - Bovy (UofT)
        """
        Eint = conversion.parse_energy(E, vo=self._vo)
        xp = resolve_namespace(Eint, self._potInf, self._Emin)
        if xp is numpy:
            out = numpy.zeros_like(Eint)
            indx = (Eint < self._potInf) * (Eint >= self._Emin)
            # rphi(E) as the exact root (the spline is off by 8e-5 below its
            # first knot, r = 1e-6 scale, i.e. within ~1e-6 of Emin)
            out[indx] = numpy.array(
                [self._fE_integral(tE, self._rphi_root(tE)) for tE in Eint[indx]]
            )
            # Add boundary term ~ 1 / sqrt(-E) dnu / dpsi | psi=0
            boundary_term = numpy.zeros_like(Eint)
            boundary_term[indx] = (
                self._dnudr(self._rInf)
                / _evaluateRforces(self._pot, self._rInf, 0)
                / numpy.sqrt(-Eint[indx])
            )
            # For some potentials, such as PowerSphericalPotential with infinite mass,
            # the boundary term as implemented is incorrect, but we'll just set it to zero,
            # because it essentially is
            boundary_term[~numpy.isfinite(boundary_term)] = 0.0
            out -= boundary_term
            return -out / (numpy.sqrt(8.0) * numpy.pi**2.0)
        # jax/torch: GL after the same substitutions as the numpy path (see
        # _fE_integral) -- r = t^2 + rphi(E) (cancels the sqrt turning point) up
        # to rsplit = 2 rphi(E), log(r) on to the scale radius, r = 1/t (tames
        # r->inf) beyond; GL nodes are strictly interior so no endpoint is
        # evaluated.
        Eb = xp.asarray(Eint) * 1.0
        dead = (Eb >= self._potInf) | (Eb < self._Emin)
        Esafe = xp.where(dead, 0.5 * (self._potInf + self._Emin), Eb)
        rphiE = xp.asarray(self._rphi(Esafe)) * 1.0
        # Newton to the exact root (differentiable): the spline is off by 8e-5
        # below its first knot (within ~1e-6 of Emin); from there 3 steps reach
        # machine precision. dPhi/dr = -Rforce.
        for _ in range(3):
            rphiE = rphiE + (
                _evaluatePotentials(self._pot, rphiE, 0) - Esafe
            ) / _evaluateRforces(self._pot, rphiE, 0)

        def _raw(r, Ei):
            # differentiable, dead-branch-guarded _fEintegrand_raw
            Fr = _evaluateRforces(self._pot, r, 0)
            num = Fr * self._d2nudr2(r) + self._dnudr(r) * evaluateR2derivs(
                self._pot, r, 0, use_physical=False
            )
            diff = _evaluatePotentials(self._pot, r, 0) - Ei
            diffsafe = xp.where(diff > 0.0, diff, xp.ones_like(diff))
            return xp.where(
                diff > 0.0,
                num / Fr**2.0 / xp.sqrt(diffsafe),
                xp.zeros_like(diff),
            )

        def _small_at(t, rphi):
            # r = rphi + t^2: 2 t / sqrt(Phi(r) - E) = 2 / sqrt(D), D the mean of
            # dPhi/dr over [rphi, r] (GL). No difference of O(1) potentials, so
            # it stays exact as E -> Emin, where Phi(r) - E is mostly rounding.
            u = t**2.0
            r = rphi + u
            Fr = _evaluateRforces(self._pot, r, 0)
            num = Fr * self._d2nudr2(r) + self._dnudr(r) * evaluateR2derivs(
                self._pot, r, 0, use_physical=False
            )
            gl_x = xp.asarray(_GL_X)
            D = -xp.sum(
                xp.asarray(_GL_W)
                * _evaluateRforces(self._pot, rphi[..., None] + u[..., None] * gl_x, 0),
                axis=-1,
            )
            return 2.0 * num / Fr**2.0 / xp.sqrt(D)

        def _small(t):
            return _small_at(t, rphi_b)

        # E = Emin = Phi(0): rphi = 0 and the small-r piece runs to the scale
        # radius (a core's endpoint limit; a cusp's is inf, below)
        atmin = Esafe <= self._Emin
        rphiE = xp.where(atmin, xp.zeros_like(rphiE), rphiE)
        rsplit = xp.where(atmin, self._scale * xp.ones_like(rphiE), 2.0 * rphiE)
        rbig = xp.maximum(rsplit, self._scale * xp.ones_like(rphiE))
        Es_b = Esafe[..., None]
        rphi_b = rphiE[..., None]
        small = fixed_quad(xp, _small, 0.0, xp.sqrt(rsplit - rphiE), n=_QUAD_N_FE)
        middle = fixed_quad(
            xp,
            lambda u: xp.exp(u) * _raw(xp.exp(u), Es_b),
            xp.log(rsplit),
            xp.log(rbig),
            n=_QUAD_N_FE,
        )
        large = fixed_quad(
            xp,
            lambda t: 1.0 / t**2.0 * _raw(1.0 / t, Es_b),
            0.0,
            1.0 / rbig,
            n=_QUAD_N_FE,
        )
        # boundary term ~ 1 / sqrt(-E) dnu / dpsi | psi=0 (essentially zero for
        # the finite-mass systems here); zeroed where non-finite as in numpy
        rInf = xp.asarray(self._rInf) * 1.0
        boundary = (
            self._dnudr(rInf) / _evaluateRforces(self._pot, rInf, 0) / xp.sqrt(-Esafe)
        )
        boundary = xp.where(xp.isfinite(boundary), boundary, xp.zeros_like(Eb))
        out = small + middle + large - boundary
        fE = -out / (numpy.sqrt(8.0) * numpy.pi**2.0)
        # E = Emin diverges iff the r = t^2 integrand grows at least as 1/t over
        # two small-r decades (the tracer's slope alone does not decide it: the
        # central force can cancel it), as on numpy
        tcheck = xp.sqrt(self._scale * xp.ones_like(rphiE))[..., None] * xp.asarray(
            [1e-2, 1e-3, 1e-4]
        )
        fs = _small_at(tcheck, xp.zeros_like(tcheck))
        afs = xp.abs(fs)
        cusp = (
            xp.all(xp.isfinite(fs), axis=-1)
            & (afs[..., 0] > 0.0)
            & (afs[..., 2] >= 10.0 * afs[..., 1])
            & (afs[..., 1] >= 10.0 * afs[..., 0])
        )
        fE = xp.where(atmin & cusp, -numpy.inf * xp.sign(fs[..., 2]), fE)  # fE = -out
        return xp.where(dead, xp.zeros_like(fE), fE)


def _fEintegrand_raw(r, pot, E, dnudr, d2nudr2):
    # The 'raw', i.e., direct integrand in the Eddington inversion
    Fr = _evaluateRforces(pot, r, 0)
    return (
        (Fr * d2nudr2(r) + dnudr(r) * evaluateR2derivs(pot, r, 0, use_physical=False))
        / Fr**2.0
        / numpy.sqrt(_evaluatePotentials(pot, r, 0) - E)
    )


# Gauss-Legendre nodes/weights on [0, 1] for Phi(rphi + u) - Phi(rphi) as the
# integral of dPhi/dr (see _fEintegrand_smallr)
_GL_X, _GL_W = numpy.polynomial.legendre.leggauss(12)
_GL_X, _GL_W = 0.5 * (_GL_X + 1.0), 0.5 * _GL_W


def _fEintegrand_smallr(t, pot, E, dnudr, d2nudr2, rmin):
    # The integrand at small r, r = rmin + t^2 (rmin = rphi(E), the exact root),
    # which cancels the sqrt divergence at the turning point. Phi(r) - E is a
    # small difference of O(1) numbers for E near Emin -- and pure rounding once
    # t^2 is below rmin's ulp -- so where the direct difference has lost more
    # than ~6 digits it is the integral of dPhi/dr = -Rforce over [rmin, r]
    # instead, with u = t^2 exact (E -> Phi(rmin), within an ulp of E).
    u = t**2.0
    r = rmin + u
    diff = _evaluatePotentials(pot, r, 0) - E
    if abs(diff) < 1e-6 * abs(E):
        diff = -u * numpy.sum(
            _GL_W * numpy.array([_evaluateRforces(pot, rmin + u * x, 0) for x in _GL_X])
        )
    Fr = _evaluateRforces(pot, r, 0)
    num = Fr * d2nudr2(r) + dnudr(r) * evaluateR2derivs(pot, r, 0, use_physical=False)
    return 2.0 * t * num / Fr**2.0 / numpy.sqrt(diff)


def _fEintegrand_larger(t, pot, E, dnudr, d2nudr2):
    # The integrand at large r, using transformation to deal with infinity
    return 1.0 / t**2 * _fEintegrand_raw(1.0 / t, pot, E, dnudr, d2nudr2)
