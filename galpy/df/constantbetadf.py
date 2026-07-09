# Class that implements DFs of the form f(E,L) = L^{-2\beta} f(E) with constant
# beta anisotropy parameter

import contextlib

import numpy
from scipy import integrate, interpolate, special

from ..backend import (
    as_numpy,
    autodiff_ops,
    get_namespace,
    is_backend_array,
    name_of_namespace,
    resolve_namespace,
    use,
)
from ..backend.quadrature import fixed_quad
from ..potential import evaluateRforces, interpSphericalPotential
from ..potential.Potential import _evaluatePotentials
from ..util import conversion, quadpack
from ..util._optional_deps import _JAX_LOADED, _TORCH_LOADED
from .sphericaldf import (
    _QUAD_N_DMDE,
    _QUAD_N_VMOM,
    _handle_rmin,
    anisotropicsphericaldf,
    sphericaldf,
)

# Gauss-Legendre order for the backend (jax/torch) fE inversion integral; the
# post-substitution integrand is smooth, so this matches scipy's adaptive numpy
# result to ~1e-6 in the physical range (the numpy path stays scipy-adaptive).
_QUAD_N_FE = 100


def _active_backend_name():
    """'torch'|'jax'|'numpy' for the active galpy backend (context/forced default)."""
    return name_of_namespace(get_namespace())


def _autodiff_xp():
    """Namespace whose autodiff builds the fE derivative chain.

    Prefer jax whenever it is available (``_JAX_LOADED``), so the numpy-eval fE
    path reproduces the historical jax-grad-fed-into-scipy computation
    bit-for-bit; fall back to torch on a torch-only install. The nested-grad
    closures differentiate correctly under either engine because they only call
    the backend-agnostic ``evaluateRforces``/``_ddenstwobetadr``.
    """
    if _JAX_LOADED:
        import jax.numpy as jnp

        return jnp
    import torch

    return torch


def _numpy_ctx(backend_name):
    """Force-numpy context under torch, else a no-op.

    fE and the DF setup are inherently numpy (scipy interpolators + quadrature)
    and only use the backend for the m-th density derivative (``_gradfunc``,
    which drives its own tensors). torch rejects the numpy/scalar coords these
    scipy paths hand to the (undecorated) potential evaluations, so under torch
    they run on numpy; jax accepts numpy inputs natively, so its path (and the
    numpy default) is a no-op here and stays byte-identical.
    """
    if backend_name == "torch":
        return use("numpy", force=True)
    return contextlib.nullcontext()


def _make_gradfunc(vmapped, name):
    """Wrap a vmapped derivative closure so it takes/returns numpy arrays.

    jax's ``vmap(func)`` already accepts numpy input (auto-converted) and its
    output flows through the numpy consumers unchanged, so it is returned as-is
    (byte-identical). torch.func.vmap requires Tensor input, so the wrapper
    coerces the (scipy-interpolator / quadrature) numpy input to a float64
    Tensor and casts the result back to numpy for the numpy-based fE machinery
    (only reached on a torch-only install; with jax present the numpy-eval fE
    path uses jax autodiff for byte-identity).
    """
    if name != "torch":
        return vmapped
    import torch

    def _gradfunc(r):
        return as_numpy(vmapped(torch.as_tensor(numpy.asarray(r), dtype=torch.float64)))

    return _gradfunc


# This is the general constantbeta superclass, implementation of general
# formula can be found following this class
class _constantbetadf(anisotropicsphericaldf):
    """Class that implements DFs of the form f(E,L) = L^{-2\beta} f(E) with constant beta anisotropy parameter"""

    def __init__(
        self, pot=None, denspot=None, beta=None, rmax=None, scale=None, ro=None, vo=None
    ):
        """
        Initialize a spherical DF with constant anisotropy parameter.

        Parameters
        ----------
        pot : Potential or a combined potential formed using addition (pot1+pot2+…), optional
            Spherical potential which determines the DF.
        denspot : Potential or a combined potential formed using addition (pot1+pot2+…), optional
            Potential instance or a combined potential formed using addition (pot1+pot2+…) that represent the density of the tracers (assumed to be spherical; if None, set equal to pot).
        beta : float, optional
            Anisotropy parameter. Default is None.
        rmax : float or Quantity, optional
            Maximum radius to consider; DF is cut off at E = Phi(rmax). Default is None.
        scale : float or Quantity, optional
            Characteristic scale radius to aid sampling calculations. Not necessary, and will also be overridden by value from pot if available. Default is None.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).
        """
        anisotropicsphericaldf.__init__(
            self, pot=pot, denspot=denspot, rmax=rmax, scale=scale, ro=ro, vo=vo
        )
        self._beta = beta

    def _call_internal(self, *args):
        """
        Evaluate the DF for a constant anisotropy Hernquist.

        Parameters
        ----------
        E : float
            The energy.
        L : float
            The angular momentum.

        Returns
        -------
        float
            The value of the DF.

        Notes
        -----
        - 2020-07-22 - Written - Lane (UofT)

        """
        E, L, _ = args
        return L ** (-2 * self._beta) * self.fE(E)

    def _dMdE(self, E):
        if not hasattr(self, "_rphi"):
            self._rphi = self._setup_rphi_interpolator()
        xp = resolve_namespace(E)
        if xp is numpy:
            fE = self.fE(E)
            out = numpy.zeros_like(E)
            out[fE > 0.0] = (
                (2.0 * numpy.pi) ** 2.5
                * special.gamma(1.0 - self._beta)
                / 2.0 ** (self._beta - 1.0)
                / special.gamma(1.5 - self._beta)
                * fE[fE > 0.0]
                * numpy.array(
                    [
                        integrate.quad(
                            lambda r: (
                                r ** (2.0 - 2.0 * self._beta)
                                * (tE - _evaluatePotentials(self._pot, r, 0.0))
                                ** (0.5 - self._beta)
                            ),
                            0.0,
                            self._rphi(tE),
                        )[0]
                        for ii, tE in enumerate(E)
                        if fE[ii] > 0.0
                    ]
                )
            )
            return out
        # jax/torch: GL after r = rphi(E) - s^2, which cancels the (E-Phi)^{...}
        # turning point at r = rphi(E) so fixed-order GL converges fast
        fE = xp.atleast_1d(self.fE(E))
        pos = fE > 0.0
        rphiE = xp.asarray(self._rphi(E)) * 1.0
        # dead-branch guard: out-of-bounds E gets a safe dummy radius, zeroed below
        rphiE = xp.where(pos, rphiE, xp.ones_like(rphiE))
        Eb = (xp.asarray(E) * 1.0)[..., None]

        def _integrand(s):
            r = rphiE[..., None] - s**2.0
            diff = Eb - _evaluatePotentials(self._pot, r, 0.0)
            # guard: numerical noise can push E - Phi below 0 at the turning point
            diffsafe = xp.where(diff > 0.0, diff, xp.ones_like(diff))
            return (
                r ** (2.0 - 2.0 * self._beta)
                * xp.where(
                    diff > 0.0, diffsafe ** (0.5 - self._beta), xp.zeros_like(diff)
                )
                * 2.0
                * s
            )

        integral = fixed_quad(xp, _integrand, 0.0, xp.sqrt(rphiE), n=_QUAD_N_DMDE)
        prefac = (
            (2.0 * numpy.pi) ** 2.5
            * special.gamma(1.0 - self._beta)
            / 2.0 ** (self._beta - 1.0)
            / special.gamma(1.5 - self._beta)
        )
        return xp.where(pos, prefac * fE * integral, xp.zeros_like(fE))

    def _sample_eta(self, r, n=1):
        """Sample the angle eta which defines radial vs tangential velocities"""
        if not hasattr(self, "_coseta_icmf_interp"):
            # Cumulative dist for cos(eta) =
            # 0.5 + x 2F1(0.5,beta,1.5,x^2)/sqrt(pi)/Gamma(1-beta)*Gamma(1.5-beta)
            cosetas = numpy.linspace(-1.0, 1.0, 20001)
            coseta_cmf = (
                cosetas
                * special.hyp2f1(0.5, self._beta, 1.5, cosetas**2.0)
                / numpy.sqrt(numpy.pi)
                / special.gamma(1.0 - self._beta)
                * special.gamma(1.5 - self._beta)
                + 0.5
            )
            self._coseta_icmf_interp = interpolate.interp1d(
                coseta_cmf, cosetas, bounds_error=False, fill_value="extrapolate"
            )
        return numpy.arccos(self._coseta_icmf_interp(numpy.random.uniform(size=n)))

    def _p_v_at_r(self, v, r):
        xp = resolve_namespace(v, r)
        if xp is not numpy:
            # coerce: a forced backend sees the numpy sampling grids here and
            # torch potentials reject numpy coords
            v = xp.asarray(v) * 1.0
            r = xp.asarray(r) * 1.0
        if hasattr(self, "_fE_interp"):
            return self._fE_interp(
                _evaluatePotentials(self._pot, r, 0) + 0.5 * v**2.0
            ) * v ** (2.0 - 2.0 * self._beta)
        else:
            return self.fE(_evaluatePotentials(self._pot, r, 0) + 0.5 * v**2.0) * v ** (
                2.0 - 2.0 * self._beta
            )

    def _vmomentdensity(self, r, n, m):
        if m % 2 == 1 or n % 2 == 1:
            return 0.0
        xp = resolve_namespace(r)
        if xp is numpy:
            return (
                2.0
                * numpy.pi
                * r ** (-2.0 * self._beta)
                * integrate.quad(
                    lambda v: (
                        v ** (2.0 - 2.0 * self._beta + m + n)
                        * self.fE(_evaluatePotentials(self._pot, r, 0) + 0.5 * v**2.0)
                    ),
                    0.0,
                    self._vmax_at_r(self._pot, r),
                )[0]
                * special.gamma(m / 2.0 - self._beta + 1.0)
                * special.gamma((n + 1) / 2.0)
                / special.gamma(0.5 * (m + n - 2.0 * self._beta + 3.0))
            )
        # jax/torch: fixed-order GL over v, differentiable in r through Phi(r)
        # and the vmax(r) integration limit; the node axis trails
        rb = xp.asarray(r) * 1.0  # coerce: torch potentials reject numpy coords
        Phir_b = (xp.asarray(_evaluatePotentials(self._pot, rb, 0)) * 1.0)[..., None]
        return (
            2.0
            * numpy.pi
            * rb ** (-2.0 * self._beta)
            * fixed_quad(
                xp,
                lambda v: (
                    v ** (2.0 - 2.0 * self._beta + m + n)
                    * self.fE(Phir_b + 0.5 * v**2.0)
                ),
                0.0,
                self._vmax_at_r(self._pot, rb),
                n=_QUAD_N_VMOM,
            )
            * special.gamma(m / 2.0 - self._beta + 1.0)
            * special.gamma((n + 1) / 2.0)
            / special.gamma(0.5 * (m + n - 2.0 * self._beta + 3.0))
        )

    def sample(self, R=None, z=None, phi=None, n=1, return_orbit=True, rmin=0.0):
        # No docstring so the superclass' is used. Sampling is a numpy-side
        # (stateful-RNG) operation drawn from the interpolated fE (built with
        # the backend's autodiff at construction); run it on numpy under torch
        # so the returned Orbit and its accessors are numpy (see _numpy_ctx).
        with _numpy_ctx(_active_backend_name()):
            return sphericaldf.sample(
                self, R=R, z=z, phi=phi, n=n, return_orbit=return_orbit, rmin=rmin
            )


class constantbetadf(_constantbetadf):
    """Class that implements DFs of the form :math:`f(E,L) = L^{-2\\beta} f_1(E)` with constant :math:`\\beta` anisotropy parameter for a given density profile"""

    def __init__(
        self,
        pot=None,
        denspot=None,
        beta=0.0,
        twobeta=None,
        rmax=None,
        rmin=None,
        scale=None,
        ro=None,
        vo=None,
    ):
        """
        Initialize a spherical DF with constant anisotropy parameter

        Parameters
        ----------
        pot : Potential instance or a combined potential formed using addition (pot1+pot2+…), optional
            Potential instance or a combined potential formed using addition (pot1+pot2+…)
        denspot : Potential instance or a combined potential formed using addition (pot1+pot2+…), optional
            Potential instance or a combined potential formed using addition (pot1+pot2+…) that represent the density of the tracers (assumed to be spherical; if None, set equal to pot)
        beta : float, optional
            anisotropy parameter
        twobeta : float, optional
            twice the anisotropy parameter (useful for \beta = half-integer, which is a special case); has priority over beta
        rmax : float or Quantity, optional
            maximum radius to consider; DF is cut off at E = Phi(rmax)
        rmin : float or Quantity, optional
            Minimum radius to consider; the distribution function is cut off at E = Phi(rmin). For potentials that diverge at r=0 (e.g., PowerSphericalPotential with alpha > 2), this is automatically set to a small value if not specified.
        scale : float or Quantity, optional
            Characteristic scale radius to aid sampling calculations. Optional and will also be overridden by value from pot if available.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2021-02-14 - Written - Bovy (UofT)

        """
        if not (_JAX_LOADED or _TORCH_LOADED):  # pragma: no cover
            raise ImportError(
                "galpy.df.constantbetadf requires the google/jax or pytorch library"
            )
        # Construction autodiff engine: jax when available (byte-identical numpy
        # fE path), else torch on a torch-only install.
        self._backend = "jax" if _JAX_LOADED else "torch"
        # Parse twobeta
        if twobeta is not None:
            beta = twobeta / 2.0
        else:
            twobeta = 2.0 * beta
        if (
            isinstance(pot, interpSphericalPotential) and beta < -0.5
        ):  # pragma: no cover
            raise RuntimeError(
                "constantbetadf with beta < -0.5 is not supported for use with interpSphericalPotential."
            )
        _constantbetadf.__init__(
            self,
            pot=pot,
            denspot=denspot,
            beta=beta,
            rmax=rmax,
            scale=scale,
            ro=ro,
            vo=vo,
        )
        # Handle rmin for divergent potentials
        self._rmin = _handle_rmin(
            rmin, self._pot, self._denspot, self._scale, self._ro, "constantbetadf"
        )

        self._twobeta = twobeta
        self._halfint = isinstance(self._twobeta, int) and self._twobeta % 2 == 1
        if self._halfint:
            self._m = (3 - self._twobeta) // 2
        else:
            self._m = int(numpy.floor(1.5 - self._beta))
            self._alpha = 1.5 - self._beta - self._m
            self._fE_prefactor = (
                2.0**self._beta
                / (2.0 * numpy.pi) ** 1.5
                / special.gamma(1.0 - self._alpha)
                / special.gamma(1.0 - self._beta)
            )
        # numpy-facing m-th (dens r^2beta)/dPsi^m derivative for the byte-identical
        # numpy fE path; the backend fE path reuses/rebuilds the raw vmapped
        # closure per eval-backend via _raw_gradfunc.
        self._gradfunc = _make_gradfunc(
            self._raw_gradfunc(_autodiff_xp()), self._backend
        )
        # Min and max energy (numpy scalars): under a forced non-numpy backend the
        # (undecorated) potential rejects the numpy/scalar limits, so coerce the
        # input and pull the value back to numpy (boundary coercion, not a compute
        # island -- the differentiable fE is the backend path below).
        xpc = get_namespace()
        if xpc is numpy:
            self._potInf = _evaluatePotentials(self._pot, self._rmax, 0)
            self._Emin = _evaluatePotentials(self._pot, self._rmin, 0)
        else:
            self._potInf = as_numpy(
                _evaluatePotentials(self._pot, xpc.asarray(self._rmax) * 1.0, 0)
            )
            self._Emin = as_numpy(
                _evaluatePotentials(self._pot, xpc.asarray(self._rmin) * 1.0, 0)
            )
        # Build interpolator r(pot), starting at rmin for divergent potentials
        self._rphi = self._setup_rphi_interpolator(
            r_a_min=max(1e-6, self._rmin / self._scale)
        )
        # Build interpolator for the lower limit of the integration (near the
        # 1/(Phi-E)^alpha divergence; at the end, we slightly adjust it up
        # to be sure to be above the point where things go haywire...
        if not self._halfint:
            # numpy-side calibration of the integration lower limit; run it
            # data-first (non-forced) so the jax/torch gradfunc autodiff traces
            # on its own tracer regardless of any forced backend (evaluateRforces
            # otherwise resolves the forced default and mismatches the tracer).
            # Byte-identical to the numpy default, where dispatch is already
            # data-first.
            with use("numpy", force=False):
                Es = numpy.linspace(
                    self._Emin, self._potInf + 1e-3 * (self._Emin - self._potInf), 51
                )
                guesspow = -17
                guesst = 10.0 ** (guesspow * (1.0 - self._alpha))
                startt = numpy.ones_like(Es) * guesst
                startval = numpy.zeros_like(Es)
                while numpy.any(startval == 0.0):
                    guesspow += 1
                    guesst = 10.0 ** (guesspow * (1.0 - self._alpha))
                    indx = startval == 0.0
                    startt[indx] = guesst
                    startval[indx] = _fEintegrand_smallr(
                        startt[indx],
                        self._pot,
                        Es[indx],
                        self._gradfunc,
                        self._alpha,
                        self._rphi(Es[indx]),
                    )
                self._logstartt = interpolate.InterpolatedUnivariateSpline(
                    Es, numpy.log10(startt) + 10.0 / 3.0 * (1.0 - self._alpha), k=3
                )

    def sample(self, R=None, z=None, phi=None, n=1, return_orbit=True, rmin=None):
        # Slight over-write of superclass method to first build f(E) interp
        # No docstring so superclass' is used
        # Use self._rmin as default if rmin is not specified
        if rmin is None:
            rmin = self._rmin
        self._ensure_fE_interp()
        # via _constantbetadf.sample so the torch->numpy sampling wrap applies
        return super().sample(
            R=R, z=z, phi=phi, n=n, return_orbit=return_orbit, rmin=rmin
        )

    def _ensure_fE_interp(self):
        """Build the f(E) interpolator if not already built."""
        if not hasattr(self, "_fE_interp"):
            Es4interp = numpy.hstack(
                (
                    numpy.geomspace(1e-8, 0.5, 101, endpoint=False),
                    sorted(1.0 - numpy.geomspace(1e-4, 0.5, 101)),
                )
            )
            Es4interp = (Es4interp * (self._Emin - self._potInf) + self._potInf)[::-1]
            # scipy spline over the numpy energy grid: pull the (backend) fE
            # values back to numpy for the frozen interpolator
            fE4interp = as_numpy(self.fE(Es4interp))
            iindx = numpy.isfinite(fE4interp)
            self._fE_interp = interpolate.InterpolatedUnivariateSpline(
                Es4interp[iindx], fE4interp[iindx], k=3, ext=3
            )

    def _make_func(self, grad):
        """Build the m-th (dens r^2beta)/dPsi^m derivative closure using ``grad``.

        d/dPsi = (d/dr)/F_r, applied m times; composed purely from the
        backend-agnostic ``_ddenstwobetadr`` / ``evaluateRforces`` (the radial
        force F_r = evaluateRforces(pot, r, 0)), so it differentiates natively
        under jax.grad or torch.func.grad. For a non-halfint DF the final 1/F_r
        is omitted (the fE integral is over Psi).
        """
        ddens = lambda r: self._denspot._ddenstwobetadr(r, beta=self._beta)
        rforce = lambda r: evaluateRforces(self._pot, r, 0.0, use_physical=False)
        if self._halfint:
            func = lambda r: ddens(r) / rforce(r)
            ii = self._m - 1
            while ii > 0:
                func = lambda r, func=func: grad(func)(r) / rforce(r)
                ii -= 1
        else:
            ii = self._m
            func = ddens if ii == 0 else (lambda r: ddens(r) / rforce(r))
            while ii > 0:
                if ii == 1:
                    func = lambda r, func=func: grad(func)(r)
                else:
                    func = lambda r, func=func: grad(func)(r) / rforce(r)
                ii -= 1
        return func

    def _raw_gradfunc(self, xp):
        """Cached vmapped m-th derivative built with ``xp``'s functional autodiff.

        Keyed on the canonical backend name of ``xp`` ("jax"/"torch"), not on
        construction: a DF built with jax autodiff may be evaluated under a
        forced-torch run, so the grad operator must match the eval backend.
        """
        name = "torch" if "torch" in getattr(xp, "__name__", "") else "jax"
        cache = self.__dict__.setdefault("_gradfunc_cache", {})
        if name not in cache:
            grad, vmap = autodiff_ops(xp)
            cache[name] = vmap(self._make_func(grad))
        return cache[name]

    def _deriv(self, xp, r):
        """m-th (dens r^2beta)/dPsi^m derivative at radii ``r`` (any shape),
        via the eval-backend's autodiff (vmap over a flattened radius axis)."""
        return self._raw_gradfunc(xp)(r.reshape(-1)).reshape(r.shape)

    def fE(self, E):
        """
        Calculate the energy portion of a constant-beta distribution function

        Parameters
        ----------
        E : float, numpy.ndarray, or Quantity
            The energy.

        Returns
        -------
        numpy.ndarray
            The value of the energy portion of the DF

        Notes
        -----
        - 2021-02-14 - Written - Bovy (UofT)
        """
        Ein = conversion.parse_energy(E, vo=self._vo)
        xp = resolve_namespace(Ein)
        if xp is numpy:  # byte-identical scipy-adaptive numpy path
            return self._fE_numpy(E, numpy.atleast_1d(Ein))
        return self._fE_backend(E, Ein, xp)  # backend GL fixed_quad (differentiable)

    def _fE_numpy(self, E, Eint):
        out = numpy.zeros_like(Eint)
        indx = (Eint < self._potInf) * (Eint >= self._Emin)
        if self._halfint:
            # fE is simply given by the relevant derivative
            out[indx] = self._gradfunc(self._rphi(Eint[indx]))
            return out.reshape(E.shape) / (
                2.0
                * numpy.pi**1.5
                * 2 ** (0.5 - self._beta)
                * special.gamma(1.0 - self._beta)
            )
        # Now need to integrate to get fE
        # Split integral at twice the lower limit to deal with divergence
        # at the lower end and infinity at the upper end
        out[indx] = numpy.array(
            [
                quadpack.quadrature(
                    lambda t: _fEintegrand_smallr(
                        t,
                        self._pot,
                        tE,
                        self._gradfunc,
                        self._alpha,
                        self._rphi(tE),
                    ),
                    10.0 ** self._logstartt(tE),
                    self._rphi(tE) ** (1.0 - self._alpha),
                )[0]
                for tE in Eint[indx]
            ]
        )
        # Add constant part at the beginning
        out[indx] += 10.0 ** self._logstartt(Eint[indx]) * _fEintegrand_smallr(
            10.0 ** self._logstartt(Eint[indx]),
            self._pot,
            Eint[indx],
            self._gradfunc,
            self._alpha,
            self._rphi(Eint[indx]),
        )
        # 2nd half of the integral
        out[indx] += numpy.array(
            [
                quadpack.quadrature(
                    lambda t: _fEintegrand_larger(
                        t, self._pot, tE, self._gradfunc, self._alpha
                    ),
                    0.0,
                    0.5 / self._rphi(tE),
                )[0]
                for tE in Eint[indx]
            ]
        )
        return -out.reshape(E.shape) * self._fE_prefactor

    def _fE_backend(self, E, Ein, xp):
        # Backend (jax/torch) Gauss-Legendre version of the inversion integral,
        # vectorized over E (node axis trails) and differentiable through the
        # integrand (Phi(r) and the m-th density derivative _deriv). The frozen
        # interpolators rphi/logstartt set the (non-differentiable) limits, as in
        # _dMdE. The post-substitution integrand is smooth, matching the scipy
        # numpy path to ~1e-6 in the physical range.
        pinf, emin = float(self._potInf), float(self._Emin)
        # torch.asarray rejects the negative strides of a reversed numpy grid
        # (e.g. the [::-1] fE-interp energies), so make numpy input contiguous
        Ein = Ein if is_backend_array(Ein) else numpy.ascontiguousarray(Ein)
        Eb = xp.atleast_1d(xp.asarray(Ein) * 1.0)
        indx = (Eb < pinf) & (Eb >= emin)
        # clamp out-of-bounds E for the frozen numpy interpolators (zeroed below)
        Enp = as_numpy(xp.where(indx, Eb, xp.ones_like(Eb) * emin))
        rphiE = xp.asarray(self._rphi(Enp)) * 1.0
        if self._halfint:
            val = self._deriv(xp, rphiE) / (
                2.0
                * numpy.pi**1.5
                * 2 ** (0.5 - self._beta)
                * special.gamma(1.0 - self._beta)
            )
            return xp.where(indx, val, xp.zeros_like(val)).reshape(E.shape)
        alpha = self._alpha
        lo = xp.asarray(10.0 ** self._logstartt(Enp)) * 1.0
        hi = rphiE ** (1.0 - alpha)
        Eb2 = Eb[..., None]

        def _raw(r):
            diff = _evaluatePotentials(self._pot, r, 0) - Eb2
            diffsafe = xp.where(diff > 0.0, diff, xp.ones_like(diff))
            return xp.where(
                diff > 0.0,
                self._deriv(xp, r) / diffsafe**alpha,
                xp.zeros_like(diff),
            )

        def _smallr(t):  # substitution r = rphiE + t^(1/(1-alpha)) regularizes
            r = t ** (1.0 / (1.0 - alpha)) + rphiE[..., None]
            return 1.0 / (1.0 - alpha) * t ** (alpha / (1.0 - alpha)) * _raw(r)

        def _larger(t):  # substitution r = 1/t handles the r -> inf tail
            return _raw(1.0 / t) / t**2.0

        i1 = fixed_quad(xp, _smallr, lo, hi, n=_QUAD_N_FE)
        # constant [0, lo] piece (integrand ~ const there): rectangle lo*smallr(lo)
        csmall = lo * _smallr(lo[..., None])[..., 0]
        i2 = fixed_quad(xp, _larger, xp.zeros_like(rphiE), 0.5 / rphiE, n=_QUAD_N_FE)
        out = -(i1 + csmall + i2) * self._fE_prefactor
        return xp.where(indx, out, xp.zeros_like(out)).reshape(E.shape)


def _evalpot_asnumpy(pot, r):
    """Phi(numpy r) as numpy, robust under a forced non-numpy backend.

    The numpy fE quadrature and the construction-time startt calibration hand
    numpy radii to the (undecorated) potential; a forced torch/jax context would
    otherwise push those numpy inputs onto the backend (torch rejects them). numpy
    is a strict pass-through (byte-identical); a forced backend coerces the input,
    evaluates natively, and pulls back to numpy (the boundary-coercion pattern
    used by _setup_rphi_interpolator, not a backend-compute island -- the
    differentiable fE lives in the backend path).
    """
    xp = get_namespace()
    if xp is numpy:
        return _evaluatePotentials(pot, r, 0)
    return as_numpy(_evaluatePotentials(pot, xp.asarray(r) * 1.0, 0))


def _fEintegrand_raw(r, pot, E, dmp1nudrmp1, alpha):
    # The 'raw', i.e., direct integrand in the constant-beta inversion
    out = numpy.zeros_like(r)  # Avoid JAX item assignment issues
    # print("r",r,dmp1nudrmp1(r),(_evaluatePotentials(pot,r,0)-E))
    out[:] = dmp1nudrmp1(r) / (_evalpot_asnumpy(pot, r) - E) ** alpha
    out[True ^ numpy.isfinite(out)] = (
        0.0  # assume these are where denom is slightly neg.
    )
    return out


def _fEintegrand_smallr(t, pot, E, dmp1nudrmp1, alpha, rmin):
    # The integrand at small r, using transformation to deal with divergence
    # print("t",t,rmin,t**(1./(1.-alpha))+rmin)
    return (
        1.0
        / (1.0 - alpha)
        * t ** (alpha / (1.0 - alpha))
        * _fEintegrand_raw(
            t ** (1.0 / (1.0 - alpha)) + rmin, pot, E, dmp1nudrmp1, alpha
        )
    )


def _fEintegrand_larger(t, pot, E, dmp1nudrmp1, alpha):
    # The integrand at large r, using transformation to deal with infinity
    return 1.0 / t**2 * _fEintegrand_raw(1.0 / t, pot, E, dmp1nudrmp1, alpha)
