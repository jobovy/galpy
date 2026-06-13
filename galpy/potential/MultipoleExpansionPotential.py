###############################################################################
#   MultipoleExpansionPotential.py: Potential via multipole expansion of a
#   given density function
###############################################################################
import inspect

import numpy
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import (
    BPoly,
    CubicSpline,
    InterpolatedUnivariateSpline,
    PPoly,
    make_interp_spline,
)

from ..backend import (
    asarray_on_device,
    device_of,
    get_namespace,
    match_input_dtype,
)
from ..backend.special import assoc_legendre
from ..util import conversion, coords
from ..util._optional_deps import _APY_LOADED
from ..util.special import compute_legendre, sph_harm_normalization
from .Potential import Potential
from .SphericalHarmonicPotentialMixin import SphericalHarmonicPotentialMixin

if _APY_LOADED:
    from astropy import units


class MultipoleExpansionPotential(Potential, SphericalHarmonicPotentialMixin):
    r"""Class that implements a gravitational potential computed via multipole expansion of an arbitrary density distribution.

    This class decomposes a user-supplied density function into real spherical harmonics on a radial grid,
    then evaluates the gravitational potential using classical multipole integrals (the real-valued form of
    `Bovy 2026 <https://galaxiesbook.org>`__, `Chapter 12.3.1 <https://galaxiesbook.org/chapters/III-01.-Gravitation-in-Elliptical-Galaxies-and-Dark-Matter-Halos_3-Multipole-and-basis-function-expansions.html>`__, equations `12.78 <https://galaxiesbook.org/chapters/III-01.-Gravitation-in-Elliptical-Galaxies-and-Dark-Matter-Halos_3-Multipole-and-basis-function-expansions.html#mjx-eqn-eq-triaxialgrav-body-decompose-2>`__–`12.79 <https://galaxiesbook.org/chapters/III-01.-Gravitation-in-Elliptical-Galaxies-and-Dark-Matter-Halos_3-Multipole-and-basis-function-expansions.html#mjx-eqn-eq-triaxialgrav-multipole-potential>`__):

    .. math::

        \rho(r,\theta,\phi) = \sum_{l=0}^{L-1}\sum_{m=0}^{l}\,\left[\rho^{\cos}_{lm}(r)\,\cos(m\phi) + \rho^{\sin}_{lm}(r)\,\sin(m\phi)\right]\,P_l^m(\cos\theta)\,,

    where the radial density coefficients are obtained by projection:

    .. math::

        \rho^{\cos}_{lm}(r) = \alpha_{lm} \int_0^{\pi}\!\int_0^{2\pi} \rho(r,\theta,\phi)\,P_l^m(\cos\theta)\,\cos(m\phi)\,\sin\theta\,\mathrm{d}\phi\,\mathrm{d}\theta\,,

        \rho^{\sin}_{lm}(r) = \alpha_{lm} \int_0^{\pi}\!\int_0^{2\pi} \rho(r,\theta,\phi)\,P_l^m(\cos\theta)\,\sin(m\phi)\,\sin\theta\,\mathrm{d}\phi\,\mathrm{d}\theta\,,

    with :math:`\alpha_{lm} = \sqrt{\frac{2l+1}{4\pi}\,\frac{(l-m)!}{(l+m)!}}`. The gravitational potential has an analogous expansion:

    .. math::

        \Phi(r, \theta, \phi) = \sum_{l=0}^{L-1}\sum_{m=0}^{l}\,\left[\Phi^{\cos}_{lm}(r)\,\cos(m\phi) + \Phi^{\sin}_{lm}(r)\,\sin(m\phi)\right]\,P_l^m(\cos\theta)\,,

    where the radial potential functions are given by:

    .. math::

        \Phi^{\cos,\sin}_{lm}(r) = -\frac{4\pi}{2l+1} \left[r^{-(l+1)} \int_0^r a^{l+2} \rho^{\cos,\sin}_{lm}(a) \, da + r^l \int_r^{\infty} a^{1-l} \rho^{\cos,\sin}_{lm}(a) \, da\right]

    The spherical harmonic decomposition is performed via Gauss-Legendre quadrature (theta) and trapezoidal
    rule (phi). Radial integrals are evaluated to high precision using spline integration and precomputed
    cumulative integrals (I_inner, I_outer), which are stored as Bernstein polynomials to ensure smooth
    derivatives that satisfy the radial Poisson equation exactly. Outside of the radial grid,
    below the minimum radius, a constant-density extrapolation is used (with density equal to the value
    at the minimum grid radius), while above the maximum radius, the density is assumed to be zero.

    **Time-dependent potentials** are supported in two ways:

    * Via direct initialization: pass ``rho_cos_splines`` (and optionally ``rho_sin_splines``)
      as lists of callables ``f(r, t)`` instead of ``InterpolatedUnivariateSpline`` instances,
      together with a ``tgrid`` array. Each callable should return the normalized radial density
      coefficient at the given radius and time (see the ``rho_cos_splines`` parameter documentation
      for details on the normalization convention).

    * Via ``from_density``: pass a density function that accepts a ``t`` keyword argument,
      i.e., ``dens(R, z, phi, t=0.0)``, together with a ``tgrid`` array of time grid points.
      The density function must be in galpy's internal units (astropy ``Quantity`` output is not
      supported for time-dependent densities).

    In both cases, the radial multipole integrals are precomputed at each time in ``tgrid``, and
    the resulting radial functions :math:`\Phi^{\cos,\sin}_{lm}(r, t)` are interpolated in time
    using piecewise polynomials. This allows efficient evaluation of the potential, forces, and
    second derivatives at arbitrary times within the ``tgrid`` range during orbit integration.
    """

    def __init__(
        self,
        amp=1.0,
        rho_cos_splines=None,
        rho_sin_splines=None,
        rgrid=numpy.geomspace(1e-3, 30, 1_001),
        tgrid=None,
        normalize=False,
        ro=None,
        vo=None,
    ):
        r"""
        Initialize a MultipoleExpansionPotential from precomputed density splines (use ``MultipoleExpansionPotential.from_density`` to initialize from a density function).

        Parameters
        ----------
        amp : float or Quantity, optional
            Amplitude to be applied to the potential (default: 1).
        rho_cos_splines : list of list of InterpolatedUnivariateSpline, optional
            Cosine density coefficient splines with the spherical harmonic normalization absorbed, shape ``[L][M]``. ``rho_cos_splines[l][m]`` is a spline representing :math:`\hat{\rho}^{\cos}_{lm}(r) = \beta_{lm}\,\rho^{\cos}_{lm}(r)` as a function of ``r``, where :math:`\rho^{\cos}_{lm}(r)` are the coefficients in the real spherical-harmonic expansion of the density :math:`\rho(r,\theta,\phi) = \sum_{l,m} [\rho^{\cos}_{lm}(r)\,\cos(m\phi) + \rho^{\sin}_{lm}(r)\,\sin(m\phi)]\,P_l^m(\cos\theta)` (the real-valued form of `Eq. 12.78 <https://galaxiesbook.org/chapters/III-01.-Gravitation-in-Elliptical-Galaxies-and-Dark-Matter-Halos_3-Multipole-and-basis-function-expansions.html#mjx-eqn-eq-triaxialgrav-body-decompose-2>`__ in `Bovy 2026 <https://galaxiesbook.org>`__) and :math:`\beta_{lm} = (2-\delta_{m0})\,\sqrt{\frac{2l+1}{4\pi}\,\frac{(l-m)!}{(l+m)!}}` is the real spherical harmonic normalization factor. If ``None``, computes a default Hernquist monopole. For time-dependent potentials, these can be callables ``f(r, t)`` instead of ``InterpolatedUnivariateSpline`` instances; in that case ``tgrid`` must also be provided.
        rho_sin_splines : list of list of InterpolatedUnivariateSpline, optional
            Like ``rho_cos_splines`` but for the sine coefficients: ``rho_sin_splines[l][m]`` represents :math:`\hat{\rho}^{\sin}_{lm}(r) = \beta_{lm}\,\rho^{\sin}_{lm}(r)`. If ``None``, set to zero splines matching ``rho_cos_splines``. For time-dependent potentials, these can be callables ``f(r, t)`` like ``rho_cos_splines``.
        rgrid : numpy.ndarray, optional
            Radial grid points (1D array). Default: ``numpy.geomspace(1e-3, 30, 1_001)``.
        tgrid : numpy.ndarray or None, optional
            Time grid for time-dependent potentials. If provided, ``rho_cos_splines`` and ``rho_sin_splines`` are interpreted as callables ``f(r, t)`` and the potential is time-dependent: BPoly radial integrals are precomputed at each time in ``tgrid`` and interpolated in time at evaluation. Default: ``None`` (static potential).
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2026-02-13 - Written - Bovy (UofT)
        - 2026-03-06 - Refactored to accept splines; density computation moved to from_density - Bovy (UofT)
        - 2026-03-23 - Added time-dependent support via tgrid - Bovy (UofT)
        """
        Potential.__init__(self, amp=amp, ro=ro, vo=vo)
        self._rgrid = rgrid
        self._k = 3
        self._tdep = tgrid is not None
        if rho_cos_splines is None:
            # Default: Hernquist monopole
            rho_cos_splines, rho_sin_splines = self._default_hernquist_splines(
                rgrid, self._k
            )
        if self._tdep:
            # Time-dependent path: splines are callables f(r, t)
            self._tgrid = numpy.asarray(tgrid)
            self._rho_cos_funcs = rho_cos_splines
            self._L = len(rho_cos_splines)
            self._M = len(rho_cos_splines[0])
            L = self._L
            M = self._M
            if rho_sin_splines is None:
                self._rho_sin_funcs = [
                    [lambda r, t: numpy.zeros_like(r) for _ in range(M)]
                    for _ in range(L)
                ]
            else:
                self._rho_sin_funcs = rho_sin_splines
            # Precompute 2D radial integrals and time interpolators
            self._precompute_radial_integrals_2d()
            # Determine isNonAxi: check whether m > 0 terms are
            # non-negligible at any time in tgrid
            if M > 1:
                _max_m0 = 0.0
                for l in range(L):
                    vals = self._rho_cos_interp[l][0](self._tgrid)
                    _max_m0 = max(_max_m0, numpy.max(numpy.abs(vals)))
                _tol = 1e-12 * max(_max_m0, 1e-16)
                has_nonaxi = False
                for l in range(L):
                    for m in range(1, min(l + 1, M)):
                        cos_vals = self._rho_cos_interp[l][m](self._tgrid)
                        if numpy.any(numpy.abs(cos_vals) > _tol):
                            has_nonaxi = True
                            break
                        if self._rho_sin_interp[l][m] is not None:
                            sin_vals = self._rho_sin_interp[l][m](self._tgrid)
                            if numpy.any(numpy.abs(sin_vals) > _tol):
                                has_nonaxi = True
                                break
                    if has_nonaxi:
                        break
                self.isNonAxi = has_nonaxi
            else:
                self.isNonAxi = False
            # Truncate to axisymmetric if non-axi terms are negligible
            if not self.isNonAxi and M > 1:
                self._M = 1
                M = 1
                self._rho_cos_funcs = [row[:1] for row in self._rho_cos_funcs]
                self._rho_sin_funcs = [row[:1] for row in self._rho_sin_funcs]
                self._I_inner_cos_interp = [row[:1] for row in self._I_inner_cos_interp]
                self._I_inner_sin_interp = [row[:1] for row in self._I_inner_sin_interp]
                self._I_outer_cos_interp = [row[:1] for row in self._I_outer_cos_interp]
                self._I_outer_sin_interp = [row[:1] for row in self._I_outer_sin_interp]
                self._rho_cos_interp = [row[:1] for row in self._rho_cos_interp]
                self._rho_sin_interp = [row[:1] for row in self._rho_sin_interp]
            self._force_cache_key = None
            self._2nd_deriv_cache_key = None
            self._cached_t = None
        else:
            # Static path: validate spline types
            for l, row in enumerate(rho_cos_splines):
                for m, s in enumerate(row):
                    if not isinstance(s, InterpolatedUnivariateSpline):
                        if callable(s):
                            raise ValueError(
                                f"rho_cos_splines[{l}][{m}] appears to be a "
                                "callable rather than an InterpolatedUnivariateSpline. "
                                "If it is a time-dependent function f(r, t), pass "
                                "tgrid=... to enable time-dependent evaluation."
                            )
                        raise TypeError(
                            f"rho_cos_splines[{l}][{m}] must be an "
                            "InterpolatedUnivariateSpline instance; use "
                            "MultipoleExpansionPotential.from_density to "
                            "initialize from a density function"
                        )
            if rho_sin_splines is not None:
                for l, row in enumerate(rho_sin_splines):
                    for m, s in enumerate(row):
                        if not isinstance(s, InterpolatedUnivariateSpline):
                            if callable(s):
                                raise ValueError(
                                    f"rho_sin_splines[{l}][{m}] appears to be a "
                                    "callable rather than an InterpolatedUnivariateSpline. "
                                    "If it is a time-dependent function f(r, t), pass "
                                    "tgrid=... to enable time-dependent evaluation."
                                )
                            raise TypeError(
                                f"rho_sin_splines[{l}][{m}] must be an "
                                "InterpolatedUnivariateSpline instance; use "
                                "MultipoleExpansionPotential.from_density to "
                                "initialize from a density function"
                            )
            self._rho_cos_splines = rho_cos_splines
            self._L = len(rho_cos_splines)
            self._M = len(rho_cos_splines[0])
            L = self._L
            M = self._M
            if rho_sin_splines is None:
                # Create zero splines matching rho_cos_splines
                self._rho_sin_splines = [
                    [
                        InterpolatedUnivariateSpline(
                            rgrid, numpy.zeros(len(rgrid)), k=self._k
                        )
                        for m in range(M)
                    ]
                    for l in range(L)
                ]
            else:
                self._rho_sin_splines = rho_sin_splines
            # Determine isNonAxi: if M > 1, check whether m > 0 splines are
            # non-negligible compared to the monopole
            if M > 1:
                _max_m0 = numpy.max(
                    numpy.abs(
                        numpy.column_stack(
                            [self._rho_cos_splines[l][0](rgrid) for l in range(L)]
                        )
                    )
                )
                _tol = 1e-12 * max(_max_m0, 1e-16)
                has_nonaxi = False
                for l in range(L):
                    for m in range(1, min(l + 1, M)):
                        if numpy.any(
                            numpy.abs(self._rho_cos_splines[l][m](rgrid)) > _tol
                        ) or numpy.any(
                            numpy.abs(self._rho_sin_splines[l][m](rgrid)) > _tol
                        ):
                            has_nonaxi = True
                            break
                    if has_nonaxi:
                        break
                self.isNonAxi = has_nonaxi
            else:
                self.isNonAxi = False
            # Truncate to axisymmetric if non-axi terms are negligible
            if not self.isNonAxi and M > 1:
                self._M = 1
                M = 1
                self._rho_cos_splines = [row[:1] for row in self._rho_cos_splines]
                self._rho_sin_splines = [row[:1] for row in self._rho_sin_splines]
            # Precompute radial integrals for potential
            # with -4*pi/(2l+1) absorbed into the spline data
            # (beta_lm is already baked into the density splines)
            self._precompute_radial_integrals()
            self._force_cache_key = None
            self._2nd_deriv_cache_key = None
        # Lazily-built constant tables for the jax/torch evaluation path
        self._backend_data = None
        self.hasC = True
        self.hasC_dxdv = True
        self.hasC_dens = True
        if normalize or (
            isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
        ):
            self.normalize(normalize)
        return None

    @staticmethod
    def _default_hernquist_splines(rgrid, k):
        """Compute default Hernquist monopole splines for the no-argument case."""
        hernquist_dens = lambda r: 1.0 / (2.0 * numpy.pi) / r / (1 + r) ** 3
        beta_00 = sph_harm_normalization(1, 1)[0, 0]
        alpha_00 = beta_00  # same for m=0
        rho_00 = numpy.array(
            [alpha_00 * 4.0 * numpy.pi * hernquist_dens(r) for r in rgrid]
        )
        rho_cos_splines = [[InterpolatedUnivariateSpline(rgrid, beta_00 * rho_00, k=k)]]
        rho_sin_splines = [
            [InterpolatedUnivariateSpline(rgrid, numpy.zeros(len(rgrid)), k=k)]
        ]
        return rho_cos_splines, rho_sin_splines

    @classmethod
    def from_density(
        cls,
        dens,
        L=6,
        rgrid=numpy.geomspace(1e-3, 30, 1_001),
        tgrid=None,
        symmetry=None,
        costheta_order=None,
        phi_order=None,
        amp=1.0,
        normalize=False,
        ro=None,
        vo=None,
    ):
        """
        Initialize a MultipoleExpansionPotential from a density function.

        Parameters
        ----------
        dens : callable or Potential
            Density function. Can take 1 arg (r), 2 args (R, z), or 3 args (R, z, phi). Can also be a galpy Potential instance. For time-dependent densities, add a ``t`` keyword argument (e.g., ``dens(R, z, phi, t=0.)``).
        L : int, optional
            Maximum spherical harmonic degree + 1 (l goes from 0 to L-1). Default: 6.
        rgrid : numpy.ndarray, optional
            Radial grid points (1D array). Default: ``numpy.geomspace(1e-3, 30, 1_001)``.
        tgrid : numpy.ndarray or None, optional
            Time grid for time-dependent potentials. Required when the density function accepts a ``t`` parameter. Default: ``None``.
        symmetry : str or None, optional
            ``'spherical'``, ``'axisymmetric'``, or ``None`` (general). Determines M.
        costheta_order : int, optional
            Gauss-Legendre quadrature order for theta. Default: ``max(20, L+1)``.
        phi_order : int, optional
            Number of uniform phi points for trapezoidal rule. Default: ``max(20, 2*L+1)``.
        amp : float or Quantity, optional
            Amplitude to be applied to the potential (default: 1).
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Returns
        -------
        MultipoleExpansionPotential
            A new MultipoleExpansionPotential instance.

        Notes
        -----
        - 2026-03-06 - Written - Bovy (UofT)
        - 2026-03-23 - Added time-dependent support - Bovy (UofT)
        """
        # Dummy instance for ro/vo parsing (following SCFPotential pattern)
        dumm = cls(ro=ro, vo=vo)
        internal_ro = dumm._ro
        internal_vo = dumm._vo
        # Parse density function
        dens_func, has_t = cls._parse_density(dens, internal_ro, internal_vo)
        # For Potential instances with tgrid, evaluate the density at each
        # time to create a time-dependent MultipoleExpansionPotential
        if isinstance(dens, Potential) and tgrid is not None:
            dens_func = lambda R, z, phi, t: dens.dens(
                R, z, phi, t=t, use_physical=False
            )
            has_t = True
        # Validate tgrid requirement
        if has_t and tgrid is None:
            raise ValueError(
                "tgrid is required when the density function depends on time "
                "(has a 't' parameter). Pass tgrid=numpy.linspace(t0, t1, Nt) "
                "to enable time-dependent evaluation."
            )
        # Set L, M based on symmetry
        if symmetry is not None and symmetry.startswith("spher"):
            L = 1
            M = 1
        elif symmetry is not None and symmetry.startswith("axi"):
            M = 1
        else:
            M = L
        # Quadrature orders
        if costheta_order is None:
            costheta_order = max(20, L + 1)
        if phi_order is None:
            phi_order = max(20, 2 * L + 1)
        if has_t:
            # Time-dependent path: compute rho_lm at all t in tgrid
            # using vectorized angular integration
            tgrid = numpy.asarray(tgrid)
            k = 3
            beta_lm = sph_harm_normalization(L, M)
            Nr = len(rgrid)
            Nt = len(tgrid)
            rho_cos_all, rho_sin_all = cls._compute_rho_lm_timedep(
                dens_func, rgrid, tgrid, L, M, costheta_order, phi_order
            )
            # Build time-dependent callable splines for each (l, m)
            # These are callables f(r, t) that return density values
            rho_cos_funcs = [[None for _ in range(M)] for _ in range(L)]
            rho_sin_funcs = [[None for _ in range(M)] for _ in range(L)]
            for l in range(L):
                for m in range(M):
                    # beta_lm * rho_cos_all[:, :, l, m] has shape (Nt, Nr)
                    cos_data = beta_lm[l, m] * rho_cos_all[:, :, l, m]
                    sin_data = beta_lm[l, m] * rho_sin_all[:, :, l, m]
                    # Build spline interpolator over t for each (l,m)
                    cos_t_interp = make_interp_spline(tgrid, cos_data, k=3)
                    sin_t_interp = make_interp_spline(tgrid, sin_data, k=3)
                    # Callable f(rgrid_eval, t) that returns density on rgrid
                    # For _precompute_radial_integrals_2d, this is called with
                    # the full rgrid and a single t
                    rho_cos_funcs[l][m] = lambda r, t, _interp=cos_t_interp: _interp(t)
                    rho_sin_funcs[l][m] = lambda r, t, _interp=sin_t_interp: _interp(t)
            # Handle astropy unit detection (following SCFPotential pattern)
            if cls._density_has_units(dens):
                ro = internal_ro
                vo = internal_vo
            return cls(
                amp=amp,
                rho_cos_splines=rho_cos_funcs,
                rho_sin_splines=rho_sin_funcs,
                rgrid=rgrid,
                tgrid=tgrid,
                normalize=normalize,
                ro=ro,
                vo=vo,
            )
        else:
            # Static path
            rho_cos, rho_sin = cls._compute_rho_lm(
                dens_func, rgrid, L, M, costheta_order, phi_order
            )
            # Normalization for angular reconstruction; absorbed into splines
            k = 3
            beta_lm = sph_harm_normalization(L, M)
            rho_cos_splines = [
                [
                    InterpolatedUnivariateSpline(
                        rgrid, beta_lm[l, m] * rho_cos[:, l, m], k=k
                    )
                    for m in range(M)
                ]
                for l in range(L)
            ]
            rho_sin_splines = [
                [
                    InterpolatedUnivariateSpline(
                        rgrid, beta_lm[l, m] * rho_sin[:, l, m], k=k
                    )
                    for m in range(M)
                ]
                for l in range(L)
            ]
            # Handle astropy unit detection (following SCFPotential pattern)
            if cls._density_has_units(dens):
                ro = internal_ro
                vo = internal_vo
            return cls(
                amp=amp,
                rho_cos_splines=rho_cos_splines,
                rho_sin_splines=rho_sin_splines,
                rgrid=rgrid,
                normalize=normalize,
                ro=ro,
                vo=vo,
            )

    @staticmethod
    def _density_has_units(dens):
        """
        Check whether a density callable returns astropy Quantity with density units.

        Parameters
        ----------
        dens : callable or Potential
            Density function.

        Returns
        -------
        bool
            True if the density returns astropy density units, False otherwise.

        Notes
        -----
        - 2026-03-31 - Written - Bovy (UofT)
        """
        if not _APY_LOADED or isinstance(dens, Potential):
            return False
        try:
            sig = inspect.signature(dens)
            params = list(sig.parameters.keys())
            spatial_params = [p for p in params if p != "t"]
            param = [1.0] * len(spatial_params)
            has_t = "t" in params
            if has_t:
                dens(*param, t=0.0).to(units.kg / units.m**3)
            else:
                dens(*param).to(units.kg / units.m**3)
        except (AttributeError, units.UnitConversionError, TypeError):
            return False
        return True

    @staticmethod
    def _parse_density(dens, ro, vo):
        """
        Parse the density input and return a callable taking (R, z, phi) or
        (R, z, phi, t) plus a flag indicating time-dependence.

        Parameters
        ----------
        dens : callable or Potential
            Density function or galpy Potential instance. May accept a ``t``
            keyword argument for time-dependent densities.
        ro : float
            Distance scale for unit conversion.
        vo : float
            Velocity scale for unit conversion.

        Returns
        -------
        dens_func : callable
            Density function taking ``(R, z, phi)`` (static) or
            ``(R, z, phi, t)`` (time-dependent).
        has_t : bool
            Whether the density depends on time.

        Notes
        -----
        - 2026-02-13 - Written - Bovy (UofT)
        - 2026-03-06 - Made static with explicit ro/vo - Bovy (UofT)
        - 2026-03-23 - Added time-dependence detection - Bovy (UofT)
        """
        # Handle galpy Potential instances (always static)
        if isinstance(dens, Potential):
            return (
                lambda R, z, phi: dens.dens(R, z, phi, use_physical=False),
                False,
            )
        # Detect t parameter via inspect.signature
        has_t = "t" in inspect.signature(dens).parameters
        # Determine number of spatial parameters via try/except
        numOfParam = 0
        if has_t:
            try:
                dens(1.0, 0.0, 0.0, t=0.0)
                numOfParam = 3
            except TypeError:
                try:
                    dens(1.0, 0.0, t=0.0)
                    numOfParam = 2
                except TypeError:
                    numOfParam = 1
        else:
            try:
                dens(1.0, 0.0, 0.0)
                numOfParam = 3
            except TypeError:
                try:
                    dens(1.0, 0.0)
                    numOfParam = 2
                except TypeError:
                    numOfParam = 1
        # Handle astropy units
        if has_t and MultipoleExpansionPotential._density_has_units(dens):
            import warnings

            from ..util import galpyWarning

            warnings.warn(
                "Time-dependent density appears to return an astropy "
                "Quantity. Unit conversion is not supported for "
                "time-dependent densities; pass the density in internal "
                "units (1/ro^3 * vo^2 / (4 pi G)) instead.",
                galpyWarning,
            )
        if not has_t and MultipoleExpansionPotential._density_has_units(dens):
            raw_dens = dens
            if numOfParam == 1:
                return (
                    lambda R, z, phi: conversion.parse_dens(
                        raw_dens(numpy.sqrt(R**2 + z**2)),
                        ro=ro,
                        vo=vo,
                    ),
                    False,
                )
            elif numOfParam == 2:
                return (
                    lambda R, z, phi: conversion.parse_dens(
                        raw_dens(R, z), ro=ro, vo=vo
                    ),
                    False,
                )
            else:
                return (
                    lambda R, z, phi: conversion.parse_dens(
                        raw_dens(R, z, phi), ro=ro, vo=vo
                    ),
                    False,
                )
        # Wrap based on number of spatial params
        if has_t:
            if numOfParam == 1:
                return (
                    lambda R, z, phi, t: dens(numpy.sqrt(R**2 + z**2), t=t),
                    True,
                )
            elif numOfParam == 2:
                return lambda R, z, phi, t: dens(R, z, t=t), True
            else:
                return lambda R, z, phi, t: dens(R, z, phi, t=t), True
        else:
            if numOfParam == 1:
                return (
                    lambda R, z, phi: dens(numpy.sqrt(R**2 + z**2)),
                    False,
                )
            elif numOfParam == 2:
                return lambda R, z, phi: dens(R, z), False
            else:
                return dens, False

    @staticmethod
    def _compute_rho_lm(dens_func, rgrid, L, M, costheta_order, phi_order):
        """
        Compute the spherical harmonic coefficients of the density on the radial grid.

        Parameters
        ----------
        dens_func : callable
            Density function taking (R, z, phi).
        rgrid : numpy.ndarray
            Radial grid points.
        L : int
            Maximum degree + 1.
        M : int
            Maximum order + 1.
        costheta_order : int
            Number of Gauss-Legendre quadrature points for cos(theta).
        phi_order : int
            Number of uniform phi points for trapezoidal rule.

        Returns
        -------
        rho_cos : numpy.ndarray
            Cosine coefficients, shape (Nr, L, M).
        rho_sin : numpy.ndarray
            Sine coefficients, shape (Nr, L, M).

        Notes
        -----
        - 2026-02-13 - Written - Bovy (UofT)
        """
        Nr = len(rgrid)
        rho_cos = numpy.zeros((Nr, L, M))
        rho_sin = numpy.zeros((Nr, L, M))
        # alpha_lm = sph_harm_normalization without the (2 - delta) factor
        # i.e., alpha_lm = sqrt((2l+1)/(4pi) * (l-m)!/(l+m)!)
        beta_lm = sph_harm_normalization(L, M)
        alpha_lm = beta_lm.copy()
        alpha_lm[:, 1:] /= 2.0  # Remove the factor of 2 for m > 0
        if L == 1 and M == 1:
            # Spherical case: density is just rho(r) * Y_00
            # Y_00 = 1/sqrt(4*pi), alpha_00 = 1/sqrt(4*pi)
            # integral rho * Y_00 dOmega = rho * 4*pi * alpha_00 = rho * sqrt(4*pi)
            for ir, r in enumerate(rgrid):
                rho_cos[ir, 0, 0] = (
                    alpha_lm[0, 0] * 4.0 * numpy.pi * dens_func(r, 0.0, 0.0)
                )
            return rho_cos, rho_sin
        # Gauss-Legendre quadrature for cos(theta)
        ct_nodes, ct_weights = leggauss(costheta_order)
        # Precompute Legendre polynomials at all quadrature nodes
        PP_all = numpy.zeros((costheta_order, L, M))
        for ict, ct in enumerate(ct_nodes):
            PP_all[ict] = compute_legendre(ct, L, M)
        if M == 1:
            # Axisymmetric: no phi integral needed, multiply by 2*pi for m=0
            for ir, r in enumerate(rgrid):
                for ict in range(costheta_order):
                    ct = ct_nodes[ict]
                    wt = ct_weights[ict]
                    sintheta = numpy.sqrt(1.0 - ct**2)
                    # For m=0, phi integral gives 2*pi
                    rho_cos[ir, :, 0] += (
                        wt
                        * PP_all[ict, :, 0]
                        * dens_func(r * sintheta, r * ct, 0.0)
                        * 2.0
                        * numpy.pi
                    )
            # Multiply by alpha_lm
            rho_cos *= alpha_lm[numpy.newaxis, :, :]
            return rho_cos, rho_sin
        # General case: full angular integration
        phi_nodes = numpy.linspace(0.0, 2.0 * numpy.pi, phi_order, endpoint=False)
        dphi = 2.0 * numpy.pi / phi_order
        # Precompute cos(m*phi) and sin(m*phi) at all phi nodes
        m_arr = numpy.arange(M)
        cos_mphi = numpy.cos(numpy.outer(phi_nodes, m_arr))  # (phi_order, M)
        sin_mphi = numpy.sin(numpy.outer(phi_nodes, m_arr))  # (phi_order, M)
        for ir, r in enumerate(rgrid):
            for ict in range(costheta_order):
                ct = ct_nodes[ict]
                wt = ct_weights[ict]
                sintheta = numpy.sqrt(1.0 - ct**2)
                R = r * sintheta
                z = r * ct
                # Evaluate density at all phi nodes (try vectorized first)
                try:
                    rho_vals = numpy.atleast_1d(
                        dens_func(
                            numpy.full(phi_order, R),
                            numpy.full(phi_order, z),
                            phi_nodes,
                        )
                    )
                except (TypeError, ValueError):
                    rho_vals = numpy.array(
                        [dens_func(R, z, p) for p in phi_nodes]
                    )  # (phi_order,)
                # Trapezoidal phi integrals
                phi_int_cos = (
                    numpy.sum(rho_vals[:, numpy.newaxis] * cos_mphi, axis=0) * dphi
                )  # (M,)
                phi_int_sin = (
                    numpy.sum(rho_vals[:, numpy.newaxis] * sin_mphi, axis=0) * dphi
                )  # (M,)
                # Accumulate over theta quadrature
                rho_cos[ir, :, :] += (
                    wt * PP_all[ict, :, :] * phi_int_cos[numpy.newaxis, :]
                )
                rho_sin[ir, :, :] += (
                    wt * PP_all[ict, :, :] * phi_int_sin[numpy.newaxis, :]
                )
        # Multiply by alpha_lm
        rho_cos *= alpha_lm[numpy.newaxis, :, :]
        rho_sin *= alpha_lm[numpy.newaxis, :, :]
        return rho_cos, rho_sin

    @staticmethod
    def _compute_rho_lm_timedep(
        dens_func, rgrid, tgrid, L, M, costheta_order, phi_order
    ):
        """Compute spherical harmonic density coefficients for all timesteps.

        Evaluates the density at all spatial quadrature points for all times,
        then performs angular integration vectorized over the time axis using
        numpy broadcasting.

        Parameters
        ----------
        dens_func : callable
            Density function taking (R, z, phi, t).
        rgrid : numpy.ndarray
            Radial grid points.
        tgrid : numpy.ndarray
            Time grid points.
        L : int
            Maximum degree + 1.
        M : int
            Maximum order + 1.
        costheta_order : int
            Number of Gauss-Legendre quadrature points for cos(theta).
        phi_order : int
            Number of uniform phi points for trapezoidal rule.

        Returns
        -------
        rho_cos_all : numpy.ndarray
            Cosine coefficients, shape (Nt, Nr, L, M).
        rho_sin_all : numpy.ndarray
            Sine coefficients, shape (Nt, Nr, L, M).

        Notes
        -----
        - 2026-03-27 - Written - Bovy (UofT)
        """
        Nr = len(rgrid)
        Nt = len(tgrid)
        # alpha_lm = normalization without the (2 - delta) factor
        beta_lm = sph_harm_normalization(L, M)
        alpha_lm = beta_lm.copy()
        alpha_lm[:, 1:] /= 2.0
        # Gauss-Legendre quadrature for cos(theta)
        ct_nodes, ct_weights = leggauss(costheta_order)
        # Precompute Legendre polynomials at all quadrature nodes
        PP_all = numpy.zeros((costheta_order, L, M))
        for ict, ct in enumerate(ct_nodes):
            PP_all[ict] = compute_legendre(ct, L, M)
        if M == 1:
            # Axisymmetric: no phi integral needed
            rho_cos_all = numpy.zeros((Nt, Nr, L, 1))
            rho_sin_all = numpy.zeros((Nt, Nr, L, 1))
            # Preallocate broadcasting arrays for vectorized path
            R_col = rgrid[:, numpy.newaxis]  # (Nr, 1)
            t_row = tgrid[numpy.newaxis, :]  # (1, Nt)
            # Try fully vectorized: evaluate density at all (r, t) at once
            _vectorized = True
            try:
                ct = ct_nodes[0]
                sintheta = numpy.sqrt(1.0 - ct**2)
                test = dens_func(R_col * sintheta, R_col * ct, 0.0, t_row)
                if numpy.shape(test) != (Nr, Nt):
                    _vectorized = False
            except (TypeError, ValueError):
                _vectorized = False
            for ict in range(costheta_order):
                ct = ct_nodes[ict]
                wt = ct_weights[ict]
                sintheta = numpy.sqrt(1.0 - ct**2)
                if _vectorized:
                    # (Nr, Nt) via broadcasting
                    rho_spatial = dens_func(
                        R_col * sintheta,
                        R_col * ct,
                        0.0,
                        t_row,
                    ).T  # -> (Nt, Nr)
                else:
                    rho_spatial = numpy.zeros((Nt, Nr))
                    for it, t in enumerate(tgrid):
                        for ir, r in enumerate(rgrid):
                            rho_spatial[it, ir] = dens_func(
                                r * sintheta, r * ct, 0.0, t
                            )
                rho_cos_all[:, :, :, 0] += (
                    (wt * rho_spatial[:, :, numpy.newaxis] * PP_all[ict, :, 0])
                    * 2.0
                    * numpy.pi
                )
            rho_cos_all *= alpha_lm[numpy.newaxis, numpy.newaxis, :, :]
            return rho_cos_all, rho_sin_all
        # General case: full angular integration
        phi_nodes = numpy.linspace(0.0, 2.0 * numpy.pi, phi_order, endpoint=False)
        dphi = 2.0 * numpy.pi / phi_order
        m_arr = numpy.arange(M)
        cos_mphi = numpy.cos(numpy.outer(phi_nodes, m_arr))  # (phi_order, M)
        sin_mphi = numpy.sin(numpy.outer(phi_nodes, m_arr))  # (phi_order, M)
        rho_cos_all = numpy.zeros((Nt, Nr, L, M))
        rho_sin_all = numpy.zeros((Nt, Nr, L, M))
        # Preallocate broadcasting arrays for vectorized path
        R_3d = rgrid[:, numpy.newaxis, numpy.newaxis]  # (Nr, 1, 1)
        t_3d = tgrid[numpy.newaxis, :, numpy.newaxis]  # (1, Nt, 1)
        phi_3d = phi_nodes[numpy.newaxis, numpy.newaxis, :]  # (1, 1, phi_order)
        # Try fully vectorized: evaluate density at all (r, t, phi) at once
        # per theta node. Shape: (Nr, Nt, phi_order)
        _vectorized = True
        try:
            ct = ct_nodes[0]
            sintheta = numpy.sqrt(1.0 - ct**2)
            test = dens_func(R_3d * sintheta, R_3d * ct, phi_3d, t_3d)
            if numpy.shape(test) != (Nr, Nt, phi_order):
                _vectorized = False
        except (TypeError, ValueError):
            _vectorized = False
        if _vectorized:
            for ict in range(costheta_order):
                ct = ct_nodes[ict]
                wt = ct_weights[ict]
                sintheta = numpy.sqrt(1.0 - ct**2)
                # rho_spatial: (Nr, Nt, phi_order)
                rho_spatial = dens_func(R_3d * sintheta, R_3d * ct, phi_3d, t_3d)
                # Phi integrals: (Nr, Nt, phi_order) @ (phi_order, M) -> (Nr, Nt, M)
                phi_int_cos = numpy.einsum("rtp,pm->rtm", rho_spatial, cos_mphi) * dphi
                phi_int_sin = numpy.einsum("rtp,pm->rtm", rho_spatial, sin_mphi) * dphi
                # Accumulate: transpose to (Nt, Nr, M) then broadcast with (L, M)
                rho_cos_all += (
                    wt
                    * PP_all[ict, numpy.newaxis, numpy.newaxis, :, :]
                    * phi_int_cos.transpose(1, 0, 2)[:, :, numpy.newaxis, :]
                )
                rho_sin_all += (
                    wt
                    * PP_all[ict, numpy.newaxis, numpy.newaxis, :, :]
                    * phi_int_sin.transpose(1, 0, 2)[:, :, numpy.newaxis, :]
                )
        else:
            for ir, r in enumerate(rgrid):
                for ict in range(costheta_order):
                    ct = ct_nodes[ict]
                    wt = ct_weights[ict]
                    sintheta = numpy.sqrt(1.0 - ct**2)
                    R = r * sintheta
                    z = r * ct
                    rho_spatial = numpy.zeros((Nt, phi_order))
                    for it, t in enumerate(tgrid):
                        try:
                            rho_spatial[it] = numpy.atleast_1d(
                                dens_func(
                                    numpy.full(phi_order, R),
                                    numpy.full(phi_order, z),
                                    phi_nodes,
                                    t,
                                )
                            )
                        except (TypeError, ValueError):
                            for ip, p in enumerate(phi_nodes):
                                rho_spatial[it, ip] = dens_func(R, z, p, t)
                    phi_int_cos = rho_spatial @ cos_mphi * dphi
                    phi_int_sin = rho_spatial @ sin_mphi * dphi
                    rho_cos_all[:, ir, :, :] += (
                        wt * PP_all[ict, :, :] * phi_int_cos[:, numpy.newaxis, :]
                    )
                    rho_sin_all[:, ir, :, :] += (
                        wt * PP_all[ict, :, :] * phi_int_sin[:, numpy.newaxis, :]
                    )
        rho_cos_all *= alpha_lm[numpy.newaxis, numpy.newaxis, :, :]
        rho_sin_all *= alpha_lm[numpy.newaxis, numpy.newaxis, :, :]
        return rho_cos_all, rho_sin_all

    @staticmethod
    def _quintic_hermite_ppoly_coeffs(vals, derivs, derivs2, dx):
        """Compute flattened PPoly coefficients for quintic Hermite interpolation.

        Given function values, first and second derivatives at breakpoints,
        computes the power-basis coefficients of the quintic C² piecewise
        polynomial that matches these constraints. This is equivalent to
        ``PPoly.from_bernstein_basis(BPoly.from_derivatives(...))`` but
        operates on arrays with an arbitrary leading batch dimension,
        avoiding per-element Python object construction.

        Parameters
        ----------
        vals : numpy.ndarray, shape ``(..., Nr)``
            Function values at breakpoints.
        derivs : numpy.ndarray, shape ``(..., Nr)``
            First derivatives at breakpoints.
        derivs2 : numpy.ndarray, shape ``(..., Nr)``
            Second derivatives at breakpoints.
        dx : numpy.ndarray, shape ``(Nr-1,)``
            Interval widths (``numpy.diff(rgrid)``).

        Returns
        -------
        numpy.ndarray, shape ``(..., 6*(Nr-1))``
            Flattened PPoly coefficients in interval-major order
            (matching ``PPoly.c.ravel(order='F')``).

        Notes
        -----
        - 2026-03-27 - Written - Bovy (UofT)
        """
        from math import comb

        # Extract left/right values for each interval
        f_L = vals[..., :-1]  # (..., Nr-1)
        f_R = vals[..., 1:]
        fp_L = derivs[..., :-1]
        fp_R = derivs[..., 1:]
        fpp_L = derivs2[..., :-1]
        fpp_R = derivs2[..., 1:]
        h = dx  # (Nr-1,); broadcasts with (..., Nr-1) batch dims via numpy rules
        # Bernstein coefficients for quintic (degree 5) Hermite interpolant
        b = numpy.empty(f_L.shape[:-1] + (6,) + f_L.shape[-1:])
        b[..., 0, :] = f_L
        b[..., 1, :] = f_L + h * fp_L / 5
        b[..., 2, :] = f_L + 2 * h * fp_L / 5 + h**2 * fpp_L / 20
        b[..., 3, :] = f_R - 2 * h * fp_R / 5 + h**2 * fpp_R / 20
        b[..., 4, :] = f_R - h * fp_R / 5
        b[..., 5, :] = f_R
        # Convert Bernstein to power basis using the exact same accumulation
        # as scipy's PPoly.from_bernstein_basis:
        #   c_s = sum_{a=0}^{s} (-1)^(a+s) C(5,a) C(5-a,s-a) b_a / h^s
        # where c_s is the coefficient of (x - x_i)^(5-s)
        c = numpy.zeros_like(b)
        for a in range(6):
            sign_a = (-1) ** a * comb(5, a)
            for s in range(a, 6):
                sign_as = sign_a * ((-1) ** s) * comb(5 - a, s - a)
                c[..., 5 - s, :] += sign_as * b[..., a, :] / dx**s
        # Flatten to interval-major: (..., Nr-1, 6) -> (..., 6*(Nr-1))
        # Move coefficient axis to last, then reshape
        # c is (..., 6, Nr-1), need (..., Nr-1, 6) for interval-major
        c_transposed = numpy.moveaxis(c, -2, -1)  # (..., Nr-1, 6)
        batch_shape = c_transposed.shape[:-2]
        return c_transposed.reshape(batch_shape + (6 * (c_transposed.shape[-2]),))

    def _precompute_radial_integrals(self):
        """
        Precompute cumulative radial integrals I_inner and I_outer for the potential.

        For each (l, m):
            I_inner(r) = integral_0^r a^{l+2} * rho_lm(a) da
            I_outer(r) = integral_r^inf a^{1-l} * rho_lm(a) da

        The prefactor ``-4*pi/(2l+1)`` is absorbed into the spline data so it
        need not be applied at evaluation time. The ``beta_lm`` normalization
        is already baked into the density splines.

        Notes
        -----
        - 2026-02-13 - Written - Bovy (UofT)
        - 2026-03-06 - Refactored to use density splines - Bovy (UofT)
        """
        rgrid = self._rgrid
        L = self._L
        M = self._M
        # Store cumulative integral splines; R, dR/dr, d²R/dr² are computed
        # analytically from these at evaluation time, avoiding dynamic-range
        # issues with high-l radial functions and ensuring the radial ODE
        # (and thus Poisson equation) is satisfied by construction.
        self._I_inner_cos = [[None for _ in range(M)] for _ in range(L)]
        self._I_inner_sin = [[None for _ in range(M)] for _ in range(L)]
        self._I_outer_cos = [[None for _ in range(M)] for _ in range(L)]
        self._I_outer_sin = [[None for _ in range(M)] for _ in range(L)]
        for l in range(L):
            pref = -4.0 * numpy.pi / (2 * l + 1)
            for m in range(min(l + 1, M)):
                # sin(m*phi) = 0 for m=0, so sin integrals are identically zero
                # and never accessed; skip their computation.
                pairs = [
                    (
                        self._rho_cos_splines[l][m],
                        self._I_inner_cos,
                        self._I_outer_cos,
                    )
                ]
                if m > 0:
                    pairs.append(
                        (
                            self._rho_sin_splines[l][m],
                            self._I_inner_sin,
                            self._I_outer_sin,
                        )
                    )
                for rho_spline, I_inner_store, I_outer_store in pairs:
                    # Evaluate density spline on grid (beta_lm already baked in)
                    rho_arr = rho_spline(rgrid)
                    # Inner integral: integrand = r^{l+2} * rho_lm(r)
                    f_inner = rgrid ** (l + 2) * rho_arr
                    # Use spline integration for higher accuracy than trapezoid
                    f_inner_spline = InterpolatedUnivariateSpline(
                        rgrid, f_inner, k=self._k
                    )
                    I_inner_vals = numpy.array(
                        [f_inner_spline.integral(rgrid[0], r) for r in rgrid]
                    )
                    # Outer integral: integrand = r^{1-l} * rho_lm(r)
                    f_outer = rgrid ** (1 - l) * rho_arr
                    f_outer_spline = InterpolatedUnivariateSpline(
                        rgrid, f_outer, k=self._k
                    )
                    total_outer = f_outer_spline.integral(rgrid[0], rgrid[-1])
                    I_outer_vals = numpy.array(
                        [
                            total_outer - f_outer_spline.integral(rgrid[0], r)
                            for r in rgrid
                        ]
                    )
                    # Compute 2nd derivatives of the integrals at grid points
                    # d²I_inner/dr² = d/dr[r^{l+2} ρ] = (l+2) r^{l+1} ρ + r^{l+2} ρ'
                    # d²I_outer/dr² = d/dr[-r^{1-l} ρ] = -(1-l) r^{-l} ρ - r^{1-l} ρ'
                    # Use density spline derivative for ρ'
                    rho_deriv = rho_spline(rgrid, 1)
                    d2I_inner = (l + 2) * rgrid ** (l + 1) * rho_arr + rgrid ** (
                        l + 2
                    ) * rho_deriv
                    d2I_outer = (
                        -(1 - l) * rgrid ** (-l) * rho_arr
                        - rgrid ** (1 - l) * rho_deriv
                    )
                    # Use BPoly.from_derivatives with 3 constraints:
                    # value, 1st derivative, 2nd derivative.
                    # This gives quintic C² piecewise polynomial, ensuring
                    # spline derivatives match analytical values at grid
                    # points for consistent R/dR/d²R evaluation.
                    # Absorb pref into the spline values so it doesn't
                    # need to be stored/passed separately
                    I_inner_store[l][m] = BPoly.from_derivatives(
                        rgrid,
                        numpy.column_stack(
                            [
                                pref * I_inner_vals,
                                pref * f_inner,
                                pref * d2I_inner,
                            ]
                        ),
                    )
                    I_outer_store[l][m] = BPoly.from_derivatives(
                        rgrid,
                        numpy.column_stack(
                            [
                                pref * I_outer_vals,
                                pref * (-f_outer),
                                pref * d2I_outer,
                            ]
                        ),
                    )

    def _precompute_radial_integrals_2d(self):
        """Precompute PPoly coefficients at each time in tgrid and build
        time-interpolators for each (l, m) component.

        For each (l, m, kind) pair, evaluate the density callable at all
        timesteps, compute cumulative radial integrals via spline
        integration, then vectorize the quintic Hermite PPoly
        construction across all timesteps at once using
        ``_quintic_hermite_ppoly_coeffs``.

        Notes
        -----
        - 2026-03-23 - Written - Bovy (UofT)
        - 2026-03-27 - Vectorized over timesteps - Bovy (UofT)
        """
        rgrid = self._rgrid
        tgrid = self._tgrid
        L = self._L
        M = self._M
        Nt = len(tgrid)
        Nr = len(rgrid)
        k = self._k
        dx = numpy.diff(rgrid)
        # Storage for time-interpolators
        self._I_inner_cos_interp = [[None for _ in range(M)] for _ in range(L)]
        self._I_inner_sin_interp = [[None for _ in range(M)] for _ in range(L)]
        self._I_outer_cos_interp = [[None for _ in range(M)] for _ in range(L)]
        self._I_outer_sin_interp = [[None for _ in range(M)] for _ in range(L)]
        self._rho_cos_interp = [[None for _ in range(M)] for _ in range(L)]
        self._rho_sin_interp = [[None for _ in range(M)] for _ in range(L)]
        for l in range(L):
            pref = -4.0 * numpy.pi / (2 * l + 1)
            for m in range(min(l + 1, M)):
                pairs = [
                    (self._rho_cos_funcs[l][m], "cos"),
                ]
                if m > 0:
                    pairs.append((self._rho_sin_funcs[l][m], "sin"))
                for rho_func, kind in pairs:
                    # Collect density, integrals, and derivatives at each t
                    rho_vals_all = numpy.zeros((Nt, Nr))
                    I_inner_all = numpy.zeros((Nt, Nr))
                    I_outer_all = numpy.zeros((Nt, Nr))
                    f_inner_all = numpy.zeros((Nt, Nr))
                    f_outer_all = numpy.zeros((Nt, Nr))
                    d2I_inner_all = numpy.zeros((Nt, Nr))
                    d2I_outer_all = numpy.zeros((Nt, Nr))
                    for it, t in enumerate(tgrid):
                        rho_arr = rho_func(rgrid, t)
                        rho_vals_all[it] = rho_arr
                        rho_spline = InterpolatedUnivariateSpline(rgrid, rho_arr, k=k)
                        # Inner integral
                        f_inner = rgrid ** (l + 2) * rho_arr
                        f_inner_spline = InterpolatedUnivariateSpline(
                            rgrid, f_inner, k=k
                        )
                        I_inner_all[it] = numpy.array(
                            [f_inner_spline.integral(rgrid[0], r) for r in rgrid]
                        )
                        f_inner_all[it] = f_inner
                        # Outer integral
                        f_outer = rgrid ** (1 - l) * rho_arr
                        f_outer_spline = InterpolatedUnivariateSpline(
                            rgrid, f_outer, k=k
                        )
                        total_outer = f_outer_spline.integral(rgrid[0], rgrid[-1])
                        I_outer_all[it] = numpy.array(
                            [
                                total_outer - f_outer_spline.integral(rgrid[0], r)
                                for r in rgrid
                            ]
                        )
                        f_outer_all[it] = f_outer
                        # 2nd derivatives
                        rho_deriv = rho_spline(rgrid, 1)
                        d2I_inner_all[it] = (l + 2) * rgrid ** (
                            l + 1
                        ) * rho_arr + rgrid ** (l + 2) * rho_deriv
                        d2I_outer_all[it] = (
                            -(1 - l) * rgrid ** (-l) * rho_arr
                            - rgrid ** (1 - l) * rho_deriv
                        )
                    # Vectorized PPoly construction over all timesteps
                    I_inner_flat_all = self._quintic_hermite_ppoly_coeffs(
                        pref * I_inner_all,
                        pref * f_inner_all,
                        pref * d2I_inner_all,
                        dx,
                    )
                    I_outer_flat_all = self._quintic_hermite_ppoly_coeffs(
                        pref * I_outer_all,
                        pref * (-f_outer_all),
                        pref * d2I_outer_all,
                        dx,
                    )
                    # Build CubicSpline time-interpolators over flattened
                    # PPoly coefficient arrays; CubicSpline is a PPoly
                    # subclass with breakpoints exactly equal to tgrid
                    I_inner_interp = CubicSpline(tgrid, I_inner_flat_all)
                    I_outer_interp = CubicSpline(tgrid, I_outer_flat_all)
                    rho_interp = CubicSpline(tgrid, rho_vals_all)
                    if kind == "cos":
                        self._I_inner_cos_interp[l][m] = I_inner_interp
                        self._I_outer_cos_interp[l][m] = I_outer_interp
                        self._rho_cos_interp[l][m] = rho_interp
                    else:
                        self._I_inner_sin_interp[l][m] = I_inner_interp
                        self._I_outer_sin_interp[l][m] = I_outer_interp
                        self._rho_sin_interp[l][m] = rho_interp

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        """
        Evaluate the potential at (R, z, phi).

        Parameters
        ----------
        R : float or numpy.ndarray
            Cylindrical Galactocentric radius.
        z : float or numpy.ndarray
            Vertical height.
        phi : float or numpy.ndarray, optional
            Azimuth.
        t : float, optional
            Time.

        Returns
        -------
        float or numpy.ndarray
            Potential at (R, z, phi).

        Notes
        -----
        - 2026-02-13 - Written - Bovy (UofT)
        - 2026-06-09 - Added jax/torch evaluation path - Bovy (UofT)
        """
        xp = get_namespace(R, z)
        # the expansion tables are deliberately float64 (precision); cast the
        # result to the input dtype at exit (no-op for float64/scalar inputs)
        if xp is not numpy:
            if not self.isNonAxi and phi is None:
                phi = 0.0
            return match_input_dtype(
                self._backend_evaluate(xp, R, z, phi, t), R, z, phi, t
            )
        if not self.isNonAxi and phi is None:
            phi = 0.0
        in_coords = (R, z, phi, t)
        R = numpy.array(R, dtype=float)
        z = numpy.array(z, dtype=float)
        phi = numpy.array(phi, dtype=float)
        t = numpy.array(t, dtype=float)
        shape = numpy.broadcast_shapes(R.shape, z.shape, phi.shape, t.shape)
        if shape == ():
            return match_input_dtype(
                self._evaluate_at_point(R, z, phi, t=t), *in_coords
            )
        R = R * numpy.ones(shape)
        z = z * numpy.ones(shape)
        phi = phi * numpy.ones(shape)
        t = t * numpy.ones(shape)
        result = numpy.zeros(shape, float)
        for idx in numpy.ndindex(*shape):
            result[idx] = self._evaluate_at_point(R[idx], z[idx], phi[idx], t=t[idx])
        return match_input_dtype(result, *in_coords)

    def _below_grid_integrals(self, r, l, I_inner_spline, I_outer_spline):
        """Compute extended I_inner and I_outer for r < rmin.

        Assumes rho_lm = constant = rho_lm(rmin) for r' < rmin.
        pref_blm is already absorbed into the splines.

        Returns (I_inner_ext, I_outer_ext, P_rho0) where P_rho0 = pref*rho0.
        """
        rmin = self._rgrid[0]
        # Extract pref*rho0 from the BPoly derivative:
        # I_inner'(rmin) = pref*rho0 * rmin^{l+2}
        P_rho0 = float(I_inner_spline(rmin, 1)) / rmin ** (l + 2)
        # Full inner integral from 0 to r with constant density
        I_inner_ext = P_rho0 / (l + 3) * r ** (l + 3)
        # Outer integral: stored I_outer(rmin) + integral from r to rmin
        I_outer_rmin = float(I_outer_spline(rmin))
        if l == 2:
            extra = P_rho0 * numpy.log(rmin / r)
        else:
            extra = P_rho0 / (2 - l) * (rmin ** (2 - l) - r ** (2 - l))
        I_outer_ext = I_outer_rmin + extra
        return I_inner_ext, I_outer_ext, P_rho0

    def _eval_R_lm(self, r, l, I_inner_spline, I_outer_spline):
        """Compute R_lm(r) = r^{-(l+1)} * I_inner + r^l * I_outer.

        pref_blm is already absorbed into the I_inner/I_outer splines.
        Extrapolation:
        - r < rmin: constant density = rho(rmin) assumed below grid
        - r > rmax: point-mass (rho=0 above grid, I_outer(rmax)=0)
        """
        rmin = self._rgrid[0]
        rmax = self._rgrid[-1]
        if r < rmin:
            I_inner_ext, I_outer_ext, _ = self._below_grid_integrals(
                r, l, I_inner_spline, I_outer_spline
            )
            return r ** (-(l + 1)) * I_inner_ext + r**l * I_outer_ext
        if r <= rmax:
            I_inner = float(I_inner_spline(r))
            I_outer = float(I_outer_spline(r))
            return r ** (-(l + 1)) * I_inner + r**l * I_outer
        # Above grid: rho=0 so I_outer(rmax)=0, use I_inner(rmax) only
        I_inner_rmax = float(I_inner_spline(rmax))
        return I_inner_rmax * r ** (-(l + 1))

    def _eval_dR_lm(self, r, l, I_inner_spline, I_outer_spline):
        """Compute dR_lm/dr from I_inner/I_outer spline values and derivatives.

        dR/dr = d/dr[r^{-(l+1)} I_inner + r^l I_outer]
              = -(l+1) r^{-(l+2)} I_inner + r^{-(l+1)} I'_inner
                + l r^{l-1} I_outer + r^l I'_outer

        pref_blm is already absorbed into the splines.
        For r < rmin, the dI/dr terms cancel analytically, leaving:
        dR/dr = -(l+1) r^{-(l+2)} I_inner_ext + l r^{l-1} I_outer_ext
        """
        rmin = self._rgrid[0]
        rmax = self._rgrid[-1]
        if r < rmin:
            I_inner_ext, I_outer_ext, _ = self._below_grid_integrals(
                r, l, I_inner_spline, I_outer_spline
            )
            return (
                -(l + 1) * r ** (-(l + 2)) * I_inner_ext
                + l * r ** (l - 1) * I_outer_ext
            )
        if r <= rmax:
            I_inner = float(I_inner_spline(r))
            I_outer = float(I_outer_spline(r))
            dI_inner = float(I_inner_spline(r, 1))
            dI_outer = float(I_outer_spline(r, 1))
            return (
                -(l + 1) * r ** (-(l + 2)) * I_inner
                + r ** (-(l + 1)) * dI_inner
                + l * r ** (l - 1) * I_outer
                + r**l * dI_outer
            )
        I_inner_rmax = float(I_inner_spline(rmax))
        return (-(l + 1)) * I_inner_rmax * r ** (-(l + 2))

    def _eval_d2R_lm(self, r, l, I_inner_spline, I_outer_spline):
        """Compute d²R_lm/dr² from I_inner/I_outer spline values and derivatives.

        d²R/dr² = d²/dr²[r^{-(l+1)} I_inner + r^l I_outer]
                = (l+1)(l+2) r^{-(l+3)} I_inner
                  - 2(l+1) r^{-(l+2)} I'_inner + r^{-(l+1)} I''_inner
                  + l(l-1) r^{l-2} I_outer
                  + 2l r^{l-1} I'_outer + r^l I''_outer

        pref_blm is already absorbed into the splines.
        For r < rmin, the dI/dr terms simplify to:
        d²R/dr² = (l+1)(l+2) r^{-(l+3)} I_inner_ext
                  + l(l-1) r^{l-2} I_outer_ext - (2l+1) * pref*rho0
        """
        rmin = self._rgrid[0]
        rmax = self._rgrid[-1]
        if r < rmin:
            I_inner_ext, I_outer_ext, P_rho0 = self._below_grid_integrals(
                r, l, I_inner_spline, I_outer_spline
            )
            return (
                (l + 1) * (l + 2) * r ** (-(l + 3)) * I_inner_ext
                + l * (l - 1) * r ** (l - 2) * I_outer_ext
                - (2 * l + 1) * P_rho0
            )
        if r <= rmax:
            I_inner = float(I_inner_spline(r))
            I_outer = float(I_outer_spline(r))
            dI_inner = float(I_inner_spline(r, 1))
            dI_outer = float(I_outer_spline(r, 1))
            d2I_inner = float(I_inner_spline(r, 2))
            d2I_outer = float(I_outer_spline(r, 2))
            return (
                (l + 1) * (l + 2) * r ** (-(l + 3)) * I_inner
                - 2 * (l + 1) * r ** (-(l + 2)) * dI_inner
                + r ** (-(l + 1)) * d2I_inner
                + l * (l - 1) * r ** (l - 2) * I_outer
                + 2 * l * r ** (l - 1) * dI_outer
                + r**l * d2I_outer
            )
        I_inner_rmax = float(I_inner_spline(rmax))
        return (l + 1) * (l + 2) * I_inner_rmax * r ** (-(l + 3))

    @staticmethod
    def _fused_ppoly_eval(cs, i_t, dt, i_r, dr, mode=0):
        """Evaluate PPoly-in-r coefficients at a specific (i_t, i_r) via cubics in t.

        Parameters
        ----------
        cs : CubicSpline
            Time interpolator. cs.c has shape (4, Nt-1, n_r) where n_r = 6*(Nr-1).
        i_t : int
            Time interval index.
        dt : float
            dt = t - tgrid[i_t].
        i_r : int
            Radial interval index.
        dr : float
            dr = r - rgrid[i_r].
        mode : int
            0=value, 1=value+deriv, 2=value+deriv+2nd deriv.

        Returns
        -------
        val : float
            PPoly value.
        d1 : float or None
            First derivative (if mode >= 1).
        d2 : float or None
            Second derivative (if mode >= 2).
        """
        # cs.c shape: (4, Nt-1, n_r) where n_r = 6*(Nr-1)
        # Extract 6 PPoly-in-r coefficients for interval i_r
        tc = cs.c[:, i_t, i_r * 6 : (i_r + 1) * 6]  # shape (4, 6)
        # Evaluate cubic in t for each of the 6 PPoly coefficients
        c = ((tc[0] * dt + tc[1]) * dt + tc[2]) * dt + tc[3]  # shape (6,)
        # Evaluate quintic in r via Horner
        val = ((((c[0] * dr + c[1]) * dr + c[2]) * dr + c[3]) * dr + c[4]) * dr + c[5]
        d1 = None
        d2 = None
        if mode >= 1:
            d1 = (
                ((5 * c[0] * dr + 4 * c[1]) * dr + 3 * c[2]) * dr + 2 * c[3]
            ) * dr + c[4]
        if mode >= 2:
            d2 = ((20 * c[0] * dr + 12 * c[1]) * dr + 6 * c[2]) * dr + 2 * c[3]
        return val, d1, d2

    def _eval_radial_lm_timedep(self, r, t, l, I_inner_cs, I_outer_cs, mode=0):
        """Fused time+radial evaluation of R_lm from CubicSpline interpolators.

        Parameters
        ----------
        r : float
            Radial coordinate.
        t : float
            Time.
        l : int
            Spherical harmonic degree.
        I_inner_cs : CubicSpline
            Time interpolator for I_inner PPoly coefficients.
        I_outer_cs : CubicSpline
            Time interpolator for I_outer PPoly coefficients.
        mode : int
            0=value, 1=value+deriv, 2=value+deriv+2nd deriv.

        Returns
        -------
        R_val : float
            R_lm(r, t).
        dR : float or None
            dR_lm/dr (if mode >= 1).
        d2R : float or None
            d²R_lm/dr² (if mode >= 2).
        """
        rgrid = self._rgrid
        tgrid = self._tgrid
        rmin = rgrid[0]
        rmax = rgrid[-1]
        Nr = len(rgrid)

        # Find time interval
        i_t = max(
            0, min(numpy.searchsorted(tgrid, t, side="right") - 1, len(tgrid) - 2)
        )
        dt = t - tgrid[i_t]

        if r < rmin:
            # Below grid: evaluate at i_r=0, dr=0
            I_inner_rmin, dI_inner_rmin, _ = self._fused_ppoly_eval(
                I_inner_cs, i_t, dt, 0, 0.0, mode=1
            )
            I_outer_rmin, _, _ = self._fused_ppoly_eval(
                I_outer_cs, i_t, dt, 0, 0.0, mode=0
            )
            P_rho0 = dI_inner_rmin / rmin ** (l + 2)
            I_inner_ext = P_rho0 / (l + 3) * r ** (l + 3)
            if l == 2:
                extra = P_rho0 * numpy.log(rmin / r)
            else:
                extra = P_rho0 / (2 - l) * (rmin ** (2 - l) - r ** (2 - l))
            I_outer_ext = I_outer_rmin + extra
            r_neg_lp1 = r ** (-(l + 1))
            r_l = r**l
            R_val = r_neg_lp1 * I_inner_ext + r_l * I_outer_ext
            dR = None
            d2R = None
            if mode >= 1:
                dR = -(l + 1) * r_neg_lp1 / r * I_inner_ext + l * r_l / r * I_outer_ext
            if mode >= 2:
                d2R = (
                    (l + 1) * (l + 2) * r_neg_lp1 / (r * r) * I_inner_ext
                    + l * (l - 1) * r_l / (r * r) * I_outer_ext
                    - (2 * l + 1) * P_rho0
                )
            return R_val, dR, d2R

        if r <= rmax:
            i_r = max(0, min(numpy.searchsorted(rgrid, r, side="right") - 1, Nr - 2))
            dr = r - rgrid[i_r]
            r_neg_lp1 = r ** (-(l + 1))
            r_l = r**l

            I_inner, dI_inner, d2I_inner = self._fused_ppoly_eval(
                I_inner_cs, i_t, dt, i_r, dr, mode=mode
            )
            I_outer, dI_outer, d2I_outer = self._fused_ppoly_eval(
                I_outer_cs, i_t, dt, i_r, dr, mode=mode
            )
            R_val = r_neg_lp1 * I_inner + r_l * I_outer
            dR = None
            d2R = None
            if mode >= 1:
                dR = (
                    -(l + 1) * r_neg_lp1 / r * I_inner
                    + r_neg_lp1 * dI_inner
                    + l * r_l / r * I_outer
                    + r_l * dI_outer
                )
            if mode >= 2:
                d2R = (
                    (l + 1) * (l + 2) * r_neg_lp1 / (r * r) * I_inner
                    - 2 * (l + 1) * r_neg_lp1 / r * dI_inner
                    + r_neg_lp1 * d2I_inner
                    + l * (l - 1) * r_l / (r * r) * I_outer
                    + 2 * l * r_l / r * dI_outer
                    + r_l * d2I_outer
                )
            return R_val, dR, d2R

        # r > rmax: evaluate I_inner at last interval endpoint
        i_r_last = Nr - 2
        dr_max = rgrid[-1] - rgrid[i_r_last]
        I_inner_rmax, _, _ = self._fused_ppoly_eval(
            I_inner_cs, i_t, dt, i_r_last, dr_max, mode=0
        )
        r_neg_lp1 = r ** (-(l + 1))
        R_val = I_inner_rmax * r_neg_lp1
        dR = None
        d2R = None
        if mode >= 1:
            dR = (-(l + 1)) * I_inner_rmax * r_neg_lp1 / r
        if mode >= 2:
            d2R = (l + 1) * (l + 2) * I_inner_rmax * r_neg_lp1 / (r * r)
        return R_val, dR, d2R

    def _evaluate_at_point(self, R, z, phi, t=0.0):
        """
        Evaluate the potential at a single point.

        Notes
        -----
        - 2026-02-13 - Written - Bovy (UofT)
        """
        L = self._L
        M = self._M
        r, theta, phi = coords.cyl_to_spher(R, z, phi)
        if not numpy.isfinite(r):
            return 0.0
        PP = compute_legendre(numpy.cos(theta), L, M)
        result = 0.0
        if r == 0.0:
            if self._tdep:
                R_00, _, _ = self._eval_radial_lm_timedep(
                    self._rgrid[0],
                    t,
                    0,
                    self._I_inner_cos_interp[0][0],
                    self._I_outer_cos_interp[0][0],
                )
            else:
                R_00 = float(self._I_outer_cos[0][0](self._rgrid[0]))
            return R_00 * PP[0, 0]
        for l in range(L):
            for m in range(min(l + 1, M)):
                if self._tdep:
                    radial_cos, _, _ = self._eval_radial_lm_timedep(
                        r,
                        t,
                        l,
                        self._I_inner_cos_interp[l][m],
                        self._I_outer_cos_interp[l][m],
                    )
                else:
                    radial_cos = self._eval_R_lm(
                        r,
                        l,
                        self._I_inner_cos[l][m],
                        self._I_outer_cos[l][m],
                    )
                contrib = PP[l, m] * numpy.cos(m * phi) * radial_cos
                if m > 0:
                    if self._tdep:
                        radial_sin, _, _ = self._eval_radial_lm_timedep(
                            r,
                            t,
                            l,
                            self._I_inner_sin_interp[l][m],
                            self._I_outer_sin_interp[l][m],
                        )
                    else:
                        radial_sin = self._eval_R_lm(
                            r,
                            l,
                            self._I_inner_sin[l][m],
                            self._I_outer_sin[l][m],
                        )
                    contrib += PP[l, m] * numpy.sin(m * phi) * radial_sin
                result += contrib
        return result

    def _dens(self, R, z, phi=0.0, t=0.0):
        """
        Evaluate the reconstructed density at (R, z, phi).

        Parameters
        ----------
        R : float or numpy.ndarray
            Cylindrical Galactocentric radius.
        z : float or numpy.ndarray
            Vertical height.
        phi : float or numpy.ndarray, optional
            Azimuth.
        t : float, optional
            Time.

        Returns
        -------
        float or numpy.ndarray
            Density at (R, z, phi).

        Notes
        -----
        - 2026-02-13 - Written - Bovy (UofT)
        - 2026-06-09 - Added jax/torch evaluation path - Bovy (UofT)
        """
        xp = get_namespace(R, z)
        # float64 table interior, input-dtype exit cast (see _evaluate)
        if xp is not numpy:
            return match_input_dtype(self._backend_dens(xp, R, z, phi, t), R, z, phi, t)
        if self._tdep:
            self._ensure_rho_for_time(t)
        if not self.isNonAxi and phi is None:
            phi = 0.0
        in_coords = (R, z, phi, t)
        R = numpy.array(R, dtype=float)
        z = numpy.array(z, dtype=float)
        phi = numpy.array(phi, dtype=float)
        shape = numpy.broadcast_shapes(R.shape, z.shape, phi.shape)
        if shape == ():
            return match_input_dtype(self._dens_at_point(R, z, phi), *in_coords)
        R = R * numpy.ones(shape)
        z = z * numpy.ones(shape)
        phi = phi * numpy.ones(shape)
        result = numpy.zeros(shape, float)
        for idx in numpy.ndindex(*shape):
            result[idx] = self._dens_at_point(R[idx], z[idx], phi[idx])
        return match_input_dtype(result, *in_coords)

    def _ensure_rho_for_time(self, t):
        """Lazy rho spline creation for time-dependent density evaluation.

        Only reconstructs the density splines (not I_inner/I_outer PPoly).
        """
        if not self._tdep or self._cached_t == t:
            return
        L = self._L
        M = self._M
        Nr = len(self._rgrid)
        rgrid = self._rgrid
        if self._cached_t is None:
            self._rho_cos_splines = [[None for _ in range(M)] for _ in range(L)]
            self._rho_sin_splines = [[None for _ in range(M)] for _ in range(L)]
        for l in range(L):
            for m in range(min(l + 1, M)):
                rho_vals = self._rho_cos_interp[l][m](t)
                self._rho_cos_splines[l][m] = InterpolatedUnivariateSpline(
                    rgrid, rho_vals, k=self._k
                )
                if m > 0:
                    rho_vals = self._rho_sin_interp[l][m](t)
                    self._rho_sin_splines[l][m] = InterpolatedUnivariateSpline(
                        rgrid, rho_vals, k=self._k
                    )
        self._cached_t = t

    def _dens_at_point(self, R, z, phi):
        """
        Evaluate the reconstructed density at a single point.

        Notes
        -----
        - 2026-02-13 - Written - Bovy (UofT)
        """
        L = self._L
        M = self._M
        r, theta, phi = coords.cyl_to_spher(R, z, phi)
        if not numpy.isfinite(r) or r > self._rgrid[-1]:
            return 0.0
        if r < self._rgrid[0]:
            r = self._rgrid[0]
        PP = compute_legendre(numpy.cos(theta), L, M)
        result = 0.0
        for l in range(L):
            for m in range(min(l + 1, M)):
                rho_cos_val = self._rho_cos_splines[l][m](r)
                contrib = PP[l, m] * numpy.cos(m * phi) * rho_cos_val
                if m > 0:
                    rho_sin_val = self._rho_sin_splines[l][m](r)
                    contrib += PP[l, m] * numpy.sin(m * phi) * rho_sin_val
                result += contrib
        return float(result)

    def _compute_spher_forces_at_point(self, R, z, phi, t=0.0):
        """
        Compute spherical force components dPhi/dr, dPhi/dtheta, dPhi/dphi at a single point.

        Parameters
        ----------
        R : float
            Cylindrical Galactocentric radius.
        z : float
            Vertical height.
        phi : float
            Azimuth.
        t : float, optional
            Time. Default: 0.0.

        Returns
        -------
        dPhi_dr : float
            Derivative of the potential with respect to r.
        dPhi_dtheta : float
            Derivative of the potential with respect to theta.
        dPhi_dphi : float
            Derivative of the potential with respect to phi.

        Notes
        -----
        - 2026-02-13 - Written - Bovy (UofT)
        """
        cache_key = (float(R), float(z), float(phi), float(t))
        if cache_key == self._force_cache_key:
            return (
                self._cached_dPhi_dr,
                self._cached_dPhi_dtheta,
                self._cached_dPhi_dphi,
            )
        L = self._L
        M = self._M
        r, theta, phi = coords.cyl_to_spher(R, z, phi)
        if r == 0.0 or not numpy.isfinite(r):
            self._force_cache_key = cache_key
            self._cached_dPhi_dr = 0.0
            self._cached_dPhi_dtheta = 0.0
            self._cached_dPhi_dphi = 0.0
            return 0.0, 0.0, 0.0
        PP, dPP = compute_legendre(numpy.cos(theta), L, M, deriv=True)
        sintheta = numpy.sin(theta)
        dPhi_dr = 0.0
        dPhi_dtheta = 0.0
        dPhi_dphi = 0.0
        for l in range(L):
            for m in range(min(l + 1, M)):
                if self._tdep:
                    radial_cos, dradial_cos, _ = self._eval_radial_lm_timedep(
                        r,
                        t,
                        l,
                        self._I_inner_cos_interp[l][m],
                        self._I_outer_cos_interp[l][m],
                        mode=1,
                    )
                else:
                    radial_cos = self._eval_R_lm(
                        r,
                        l,
                        self._I_inner_cos[l][m],
                        self._I_outer_cos[l][m],
                    )
                    dradial_cos = self._eval_dR_lm(
                        r,
                        l,
                        self._I_inner_cos[l][m],
                        self._I_outer_cos[l][m],
                    )
                cos_mphi = numpy.cos(m * phi)
                sin_mphi = numpy.sin(m * phi)
                dPhi_dr += PP[l, m] * cos_mphi * dradial_cos
                dPhi_dtheta += dPP[l, m] * (-sintheta) * cos_mphi * radial_cos
                dPhi_dphi += PP[l, m] * (-m * sin_mphi) * radial_cos
                if m > 0:
                    if self._tdep:
                        radial_sin, dradial_sin, _ = self._eval_radial_lm_timedep(
                            r,
                            t,
                            l,
                            self._I_inner_sin_interp[l][m],
                            self._I_outer_sin_interp[l][m],
                            mode=1,
                        )
                    else:
                        radial_sin = self._eval_R_lm(
                            r,
                            l,
                            self._I_inner_sin[l][m],
                            self._I_outer_sin[l][m],
                        )
                        dradial_sin = self._eval_dR_lm(
                            r,
                            l,
                            self._I_inner_sin[l][m],
                            self._I_outer_sin[l][m],
                        )
                    dPhi_dr += PP[l, m] * sin_mphi * dradial_sin
                    dPhi_dtheta += dPP[l, m] * (-sintheta) * sin_mphi * radial_sin
                    dPhi_dphi += PP[l, m] * (m * cos_mphi) * radial_sin
        # Negate to match convention: return negative gradient (force components)
        dPhi_dr = -dPhi_dr
        dPhi_dtheta = -dPhi_dtheta
        dPhi_dphi = -dPhi_dphi
        # Cache for reuse
        self._force_cache_key = cache_key
        self._cached_dPhi_dr = dPhi_dr
        self._cached_dPhi_dtheta = dPhi_dtheta
        self._cached_dPhi_dphi = dPhi_dphi
        return dPhi_dr, dPhi_dtheta, dPhi_dphi

    def _compute_spher_2nd_derivs_at_point(self, R, z, phi, t=0.0):
        """
        Compute spherical second derivatives of the potential at a single point.

        Parameters
        ----------
        R : float
            Cylindrical Galactocentric radius.
        z : float
            Vertical height.
        phi : float
            Azimuth.
        t : float, optional
            Time. Default: 0.0.

        Returns
        -------
        d2Phi_dr2 : float
            Second derivative with respect to r.
        d2Phi_dtheta2 : float
            Second derivative with respect to theta.
        d2Phi_dphi2 : float
            Second derivative with respect to phi.
        d2Phi_drdtheta : float
            Mixed derivative with respect to r and theta.
        d2Phi_drdphi : float
            Mixed derivative with respect to r and phi.
        d2Phi_dthetadphi : float
            Mixed derivative with respect to theta and phi.
        dPhi_dr : float
            First derivative with respect to r (needed for chain rule).
        dPhi_dtheta : float
            First derivative with respect to theta (needed for chain rule).

        Notes
        -----
        - 2026-02-18 - Written - Bovy (UofT)
        """
        cache_key = (float(R), float(z), float(phi), float(t))
        if cache_key == self._2nd_deriv_cache_key:
            return self._cached_2nd_derivs
        L = self._L
        M = self._M
        r, theta, phi = coords.cyl_to_spher(R, z, phi)
        if r == 0.0 or not numpy.isfinite(r):
            self._2nd_deriv_cache_key = cache_key
            self._cached_2nd_derivs = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            return self._cached_2nd_derivs
        costheta = numpy.cos(theta)
        sintheta = numpy.sin(theta)
        if M > 1 and abs(1.0 - costheta * costheta) < 1e-14:
            costheta = numpy.sign(costheta) * (1.0 - 1e-7)
            sintheta = numpy.sqrt(1.0 - costheta**2)
        PP, dPP, d2PP = compute_legendre(costheta, L, M, deriv=2)
        d2Phi_dr2 = 0.0
        d2Phi_dtheta2 = 0.0
        d2Phi_dphi2 = 0.0
        d2Phi_drdtheta = 0.0
        d2Phi_drdphi = 0.0
        d2Phi_dthetadphi = 0.0
        dPhi_dr = 0.0
        dPhi_dtheta = 0.0
        for l in range(L):
            for m in range(min(l + 1, M)):
                if self._tdep:
                    radial_cos, dradial_cos, d2radial_cos = (
                        self._eval_radial_lm_timedep(
                            r,
                            t,
                            l,
                            self._I_inner_cos_interp[l][m],
                            self._I_outer_cos_interp[l][m],
                            mode=2,
                        )
                    )
                else:
                    radial_cos = self._eval_R_lm(
                        r,
                        l,
                        self._I_inner_cos[l][m],
                        self._I_outer_cos[l][m],
                    )
                    dradial_cos = self._eval_dR_lm(
                        r,
                        l,
                        self._I_inner_cos[l][m],
                        self._I_outer_cos[l][m],
                    )
                    d2radial_cos = self._eval_d2R_lm(
                        r,
                        l,
                        self._I_inner_cos[l][m],
                        self._I_outer_cos[l][m],
                    )
                cos_mphi = numpy.cos(m * phi)
                sin_mphi = numpy.sin(m * phi)
                Plm = PP[l, m]
                dPlm_dx = dPP[l, m]
                d2Plm_dx2 = d2PP[l, m]
                # dP_l^m/dtheta = dP/dx * (-sintheta)
                dPlm_dtheta = dPlm_dx * (-sintheta)
                # d²P_l^m/dtheta² = d²P/dx² * sin²theta - dP/dx * costheta
                d2Plm_dtheta2 = d2Plm_dx2 * sintheta**2 - dPlm_dx * costheta
                # --- Cosine terms ---
                # First derivatives (needed for chain rule)
                dPhi_dr += Plm * cos_mphi * dradial_cos
                dPhi_dtheta += dPlm_dtheta * cos_mphi * radial_cos
                # d²Phi/dr²
                d2Phi_dr2 += Plm * cos_mphi * d2radial_cos
                # d²Phi/dtheta²
                d2Phi_dtheta2 += d2Plm_dtheta2 * cos_mphi * radial_cos
                # d²Phi/dphi² : cos(m*phi) -> -m²*cos(m*phi)
                d2Phi_dphi2 += Plm * (-m * m * cos_mphi) * radial_cos
                # d²Phi/drdtheta
                d2Phi_drdtheta += dPlm_dtheta * cos_mphi * dradial_cos
                # d²Phi/drdphi
                d2Phi_drdphi += Plm * (-m * sin_mphi) * dradial_cos
                # d²Phi/dthetadphi
                d2Phi_dthetadphi += dPlm_dtheta * (-m * sin_mphi) * radial_cos
                if m > 0:
                    if self._tdep:
                        radial_sin, dradial_sin, d2radial_sin = (
                            self._eval_radial_lm_timedep(
                                r,
                                t,
                                l,
                                self._I_inner_sin_interp[l][m],
                                self._I_outer_sin_interp[l][m],
                                mode=2,
                            )
                        )
                    else:
                        radial_sin = self._eval_R_lm(
                            r,
                            l,
                            self._I_inner_sin[l][m],
                            self._I_outer_sin[l][m],
                        )
                        dradial_sin = self._eval_dR_lm(
                            r,
                            l,
                            self._I_inner_sin[l][m],
                            self._I_outer_sin[l][m],
                        )
                        d2radial_sin = self._eval_d2R_lm(
                            r,
                            l,
                            self._I_inner_sin[l][m],
                            self._I_outer_sin[l][m],
                        )
                    # --- Sine terms ---
                    dPhi_dr += Plm * sin_mphi * dradial_sin
                    dPhi_dtheta += dPlm_dtheta * sin_mphi * radial_sin
                    d2Phi_dr2 += Plm * sin_mphi * d2radial_sin
                    d2Phi_dtheta2 += d2Plm_dtheta2 * sin_mphi * radial_sin
                    d2Phi_dphi2 += Plm * (-m * m * sin_mphi) * radial_sin
                    d2Phi_drdtheta += dPlm_dtheta * sin_mphi * dradial_sin
                    d2Phi_drdphi += Plm * (m * cos_mphi) * dradial_sin
                    d2Phi_dthetadphi += dPlm_dtheta * (m * cos_mphi) * radial_sin
        self._2nd_deriv_cache_key = cache_key
        self._cached_2nd_derivs = (
            d2Phi_dr2,
            d2Phi_dtheta2,
            d2Phi_dphi2,
            d2Phi_drdtheta,
            d2Phi_drdphi,
            d2Phi_dthetadphi,
            dPhi_dr,
            dPhi_dtheta,
        )
        return self._cached_2nd_derivs

    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        return self._evaluate_cyl_2nd_deriv("R2", R, z, phi, t=t)

    def _z2deriv(self, R, z, phi=0.0, t=0.0):
        return self._evaluate_cyl_2nd_deriv("z2", R, z, phi, t=t)

    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        return self._evaluate_cyl_2nd_deriv("Rz", R, z, phi, t=t)

    def _phi2deriv(self, R, z, phi=0.0, t=0.0):
        return self._evaluate_cyl_2nd_deriv("phi2", R, z, phi, t=t)

    def _Rphideriv(self, R, z, phi=0.0, t=0.0):
        return self._evaluate_cyl_2nd_deriv("Rphi", R, z, phi, t=t)

    def _phizderiv(self, R, z, phi=0.0, t=0.0):
        return self._evaluate_cyl_2nd_deriv("phiz", R, z, phi, t=t)

    ###########################################################################
    # Backend (jax/torch) evaluation path.
    #
    # The numpy path above evaluates the radial functions point-by-point
    # through the scipy BPoly/spline objects and is kept bit-for-bit
    # unchanged (the dispatchers below only route non-numpy inputs away from
    # it). For jax/torch arrays the same piecewise polynomials are evaluated
    # in a vectorized, autodiff/jit-friendly way: the BPoly radial integrals
    # are converted once to power-basis (PPoly) coefficient stacks evaluated
    # by searchsorted + Horner (exactly the representation _serialize_for_c
    # hands to the C code), and the angular part uses the backend-agnostic
    # galpy.backend.special.assoc_legendre router. Time-dependent (tgrid=...)
    # instances follow the same scheme with one extra (clamped) searchsorted
    # in tgrid: the CubicSpline-in-t coefficient stacks of the numpy path are
    # re-packed so a cubic Horner in (t - t_k) yields the effective radial
    # PPoly coefficients at time t (the vectorized twin of _fused_ppoly_eval /
    # _eval_radial_lm_timedep).
    ###########################################################################
    def _Rforce(self, R, z, phi=0, t=0):
        xp = get_namespace(R, z)
        if xp is not numpy:
            # float64 table interior, input-dtype exit cast (see _evaluate);
            # the numpy path inherits the same cast from the mixin
            return match_input_dtype(
                self._backend_cyl_force(xp, "R", R, z, phi, t), R, z, phi, t
            )
        return SphericalHarmonicPotentialMixin._Rforce(self, R, z, phi=phi, t=t)

    def _zforce(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        if xp is not numpy:
            return match_input_dtype(
                self._backend_cyl_force(xp, "z", R, z, phi, t), R, z, phi, t
            )
        return SphericalHarmonicPotentialMixin._zforce(self, R, z, phi=phi, t=t)

    def _phitorque(self, R, z, phi=0, t=0):
        xp = get_namespace(R, z)
        if xp is not numpy:
            return match_input_dtype(
                self._backend_cyl_force(xp, "phi", R, z, phi, t), R, z, phi, t
            )
        return SphericalHarmonicPotentialMixin._phitorque(self, R, z, phi=phi, t=t)

    def _evaluate_cyl_2nd_deriv(self, deriv_type, R, z, phi, t=0.0):
        xp = get_namespace(R, z)
        if xp is not numpy:
            return match_input_dtype(
                self._backend_cyl_2nd_deriv(xp, deriv_type, R, z, phi, t), R, z, phi, t
            )
        return SphericalHarmonicPotentialMixin._evaluate_cyl_2nd_deriv(
            self, deriv_type, R, z, phi, t=t
        )

    def _backend_comps(self):
        """(l, m, is_sin) component order shared by all backend tables."""
        comps = []
        for l in range(self._L):
            for m in range(min(l + 1, self._M)):
                comps.append((l, m, False))
                if m > 0:
                    comps.append((l, m, True))
        return comps

    @staticmethod
    def _backend_comp_consts(comps, rmin):
        """Per-component constant tables shared by the static and TD layouts.

        The below-grid (constant-density) extrapolation's l-dependent
        prefactors are precomputed so the l == 2 logarithmic special case
        reduces to a constant mask.
        """
        l_arr = numpy.array([c[0] for c in comps])
        m_arr = numpy.array([c[1] for c in comps])
        return {
            "l_idx": l_arr,
            "m_idx": m_arr,
            "lf": l_arr.astype(float)[:, None],
            "mf": m_arr.astype(float)[:, None],
            "is_sin": numpy.array([c[2] for c in comps])[:, None],
            "inv_lp3": (1.0 / (l_arr + 3.0))[:, None],
            "inv_2ml": numpy.where(
                l_arr == 2, 0.0, 1.0 / numpy.where(l_arr == 2, 1.0, 2.0 - l_arr)
            )[:, None],
            "l2mask": (l_arr == 2)[:, None],
            "rmin_pow_2ml": (rmin ** (2.0 - l_arr))[:, None],
            "rmin_pow_lp2": (rmin ** (l_arr + 2.0))[:, None],
        }

    def _backend_tables(self):
        """Constant numpy tables for the backend path (static or TD layout)."""
        return self._backend_tdep_data() if self._tdep else self._backend_static_data()

    def _backend_static_data(self):
        """Build (once) the numpy constant tables for the backend path.

        Converts the per-(l, m, cos/sin) BPoly radial integrals to stacked
        power-basis (PPoly) coefficient arrays plus the analytic
        extrapolation constants of _below_grid_integrals / _eval_R_lm, and
        the FITPACK density splines to cubic PPoly pieces. All entries are
        plain numpy arrays/floats; the evaluators convert them to the active
        backend with ``asarray_on_device`` (placed on the coordinate inputs'
        device).

        Notes
        -----
        - 2026-06-09 - Written - Bovy (UofT)
        """
        if self._backend_data is not None:
            return self._backend_data
        comps = self._backend_comps()
        rmin, rmax = self._rgrid[0], self._rgrid[-1]
        breaks, rho_breaks = None, None
        inner_c, outer_c, rho_c = [], [], []
        I_outer_rmin, I_inner_rmax = [], []
        for l, m, is_sin in comps:
            I_inner = (self._I_inner_sin if is_sin else self._I_inner_cos)[l][m]
            I_outer = (self._I_outer_sin if is_sin else self._I_outer_cos)[l][m]
            rho_sp = (self._rho_sin_splines if is_sin else self._rho_cos_splines)[l][m]
            pp_in = PPoly.from_bernstein_basis(I_inner)
            pp_out = PPoly.from_bernstein_basis(I_outer)
            if breaks is None:
                breaks = pp_in.x
            inner_c.append(pp_in.c)  # (6, Nr-1), highest power first
            outer_c.append(pp_out.c)
            I_outer_rmin.append(float(pp_out(rmin)))
            I_inner_rmax.append(float(pp_in(rmax)))
            # Density spline -> cubic PPoly pieces; drop FITPACK's zero-width
            # boundary-knot intervals so searchsorted lookup is well-defined
            pp_rho = PPoly.from_spline(rho_sp._eval_args)
            keep = numpy.diff(pp_rho.x) > 0
            if rho_breaks is None:
                rho_breaks = numpy.append(pp_rho.x[:-1][keep], pp_rho.x[-1])
            rho_c.append(pp_rho.c[:, keep])
        inner_c = numpy.stack(inner_c)
        # Below-grid constant-density constants: P_rho0 = I_inner'(rmin) /
        # rmin^{l+2} (= pref*rho(rmin), cf. _below_grid_integrals)
        self._backend_data = {
            "breaks": breaks,
            "inner_c": inner_c,  # (n_comp, 6, Nr-1)
            "outer_c": numpy.stack(outer_c),
            "rho_breaks": rho_breaks,
            "rho_c": numpy.stack(rho_c),  # (n_comp, 4, n_int)
            "I_outer_rmin": numpy.array(I_outer_rmin)[:, None],
            "I_inner_rmax": numpy.array(I_inner_rmax)[:, None],
            # value the numpy path returns at the exact center:
            # R_00(rmin) * P_00 with P_00 = 1
            "R00": I_outer_rmin[0],
            **self._backend_comp_consts(comps, rmin),
        }
        self._backend_data["P_rho0"] = (
            inner_c[:, 4, 0:1] / self._backend_data["rmin_pow_lp2"]
        )
        return self._backend_data

    def _backend_tdep_data(self):
        """Build (once) the numpy constant tables for the time-dependent
        backend path.

        The numpy TD path stores, per (l, m, cos/sin) component, CubicSpline
        time-interpolators over the flattened power-basis radial PPoly
        coefficients (``cs.c`` of shape ``(4, Nt-1, 6*(Nr-1))``) and
        evaluates them via the fused cubic-in-dt + quintic-in-dr Horner of
        ``_fused_ppoly_eval`` / ``_eval_radial_lm_timedep``. This method
        re-packs exactly those CubicSpline coefficient stacks for vectorized
        backend evaluation:

        * ``inner_tc`` / ``outer_tc``: ``(n_comp, 4, 6, (Nt-1)*(Nr-1))``
          tables gathered with a combined time-interval x radial-interval
          index; a cubic Horner in ``dt = t - tgrid[i_t]`` (same op order as
          ``_fused_ppoly_eval``) collapses them to the effective 6 radial
          PPoly coefficients at time t.
        * boundary tables at the first radial breakpoint (value and first
          derivative of the integrals at rmin) and the last radial interval
          (for I_inner(rmax)), feeding the analytic below-grid
          (constant-density) and above-grid (point-mass) extrapolations of
          ``_eval_radial_lm_timedep``.
        * ``rho_tc``: the numpy TD density path (``_ensure_rho_for_time``)
          builds a FITPACK interpolating spline through the
          time-interpolated grid values. Spline interpolation is linear in
          the data, so that spline equals the cubic-Horner-in-dt combination
          of the splines through each CubicSpline coefficient row; those
          per-(time-interval, dt-power) spline PPoly stacks are precomputed
          here, shape ``(n_comp, 4, 4, (Nt-1)*n_rint)``.

        Time extrapolation outside tgrid matches the numpy path (clamped
        interval index = edge-interval cubic extrapolation, scipy's
        ``extrapolate=True``). All entries are plain numpy arrays/floats;
        the evaluators convert them with ``asarray_on_device`` (placed on
        the coordinate inputs' device).

        Notes
        -----
        - 2026-06-10 - Written - Bovy (UofT)
        """
        if self._backend_data is not None:
            return self._backend_data
        comps = self._backend_comps()
        rgrid = self._rgrid
        rmin = rgrid[0]
        nint = len(rgrid) - 1
        rho_breaks = None
        inner_tc, outer_tc, rho_tc = [], [], []
        dIin_rmin_tc, Iin_rmin_tc, Iout_rmin_tc, inner_last_tc = [], [], [], []
        for l, m, is_sin in comps:
            I_inner_cs = (
                self._I_inner_sin_interp if is_sin else self._I_inner_cos_interp
            )[l][m]
            I_outer_cs = (
                self._I_outer_sin_interp if is_sin else self._I_outer_cos_interp
            )[l][m]
            rho_cs = (self._rho_sin_interp if is_sin else self._rho_cos_interp)[l][m]
            ci = I_inner_cs.c  # (4, Nt-1, 6*(Nr-1)), interval-major in r
            co = I_outer_cs.c
            # (4, Nt-1, 6*nint) -> (4, 6, (Nt-1)*nint), flat = i_t*nint + i_r
            inner_tc.append(
                ci.reshape(4, -1, nint, 6).transpose(0, 3, 1, 2).reshape(4, 6, -1)
            )
            outer_tc.append(
                co.reshape(4, -1, nint, 6).transpose(0, 3, 1, 2).reshape(4, 6, -1)
            )
            # i_r = 0, dr = 0 boundary stacks: PPoly value is the dr^0
            # coefficient (flat index 5), its derivative the dr^1 one (4)
            dIin_rmin_tc.append(ci[:, :, 4])
            Iin_rmin_tc.append(ci[:, :, 5])
            Iout_rmin_tc.append(co[:, :, 5])
            # full 6-coefficient stack of the last radial interval, for
            # I_inner(rmax) above the grid: (4, 6, Nt-1)
            inner_last_tc.append(ci[:, :, 6 * (nint - 1) :].transpose(0, 2, 1))
            # Density: FITPACK splines through each CubicSpline coefficient
            # row (linearity, see docstring); drop FITPACK's zero-width
            # boundary-knot intervals so searchsorted lookup is well-defined
            comp_rho = []
            for kt in range(4):
                row = []
                for it in range(rho_cs.c.shape[1]):
                    pp = PPoly.from_spline(
                        InterpolatedUnivariateSpline(
                            rgrid, rho_cs.c[kt, it], k=self._k
                        )._eval_args
                    )
                    keep = numpy.diff(pp.x) > 0
                    if rho_breaks is None:
                        rho_breaks = numpy.append(pp.x[:-1][keep], pp.x[-1])
                    row.append(pp.c[:, keep])  # (4, n_rint)
                comp_rho.append(numpy.stack(row, axis=1))  # (4, Nt-1, n_rint)
            # (4 dt-powers, 4 dr-powers, (Nt-1)*n_rint)
            rho_tc.append(numpy.stack(comp_rho).reshape(4, 4, -1))
        self._backend_data = {
            "tgrid": numpy.asarray(self._tgrid, dtype=float),
            "breaks": numpy.asarray(rgrid, dtype=float),
            "inner_tc": numpy.stack(inner_tc),  # (n_comp, 4, 6, (Nt-1)*nint)
            "outer_tc": numpy.stack(outer_tc),
            "dIin_rmin_tc": numpy.stack(dIin_rmin_tc),  # (n_comp, 4, Nt-1)
            "Iin_rmin_tc": numpy.stack(Iin_rmin_tc),
            "Iout_rmin_tc": numpy.stack(Iout_rmin_tc),
            "inner_last_tc": numpy.stack(inner_last_tc),  # (n_comp, 4, 6, Nt-1)
            "rho_breaks": rho_breaks,
            "rho_tc": numpy.stack(rho_tc),  # (n_comp, 4, 4, (Nt-1)*n_rint)
            **self._backend_comp_consts(comps, rmin),
        }
        return self._backend_data

    @staticmethod
    def _backend_prepare(xp, R, z, phi, t):
        """Broadcast (R, z, phi, t) to a common shape and flatten to 1-D.

        Scalar coordinates (e.g. the default phi=0.0/t=0.0) are placed on the
        device of the array coordinates (CUDA support): the broadcast below
        expands them to full-size arrays, which torch cannot mix with CUDA
        tensors (unlike 0-d scalars)."""
        dev = device_of(R, z, phi, t)
        R = asarray_on_device(xp, R, dev) * 1.0
        z = asarray_on_device(xp, z, dev) * 1.0
        phi = asarray_on_device(xp, phi, dev) * 1.0
        if isinstance(t, (int, float)):
            # route Python scalars through numpy so torch keeps float64
            t = numpy.asarray(t, dtype=float)
        t = asarray_on_device(xp, t, dev) * 1.0
        R, z, phi, t = xp.broadcast_arrays(R, z, phi, t)
        shape = R.shape
        return (
            xp.reshape(R, (-1,)),
            xp.reshape(z, (-1,)),
            xp.reshape(phi, (-1,)),
            xp.reshape(t, (-1,)),
            shape,
        )

    @staticmethod
    def _backend_horner_t(c, dt):
        """Cubic Horner in dt over axis 1 of a (n_comp, 4, ...) stack.

        Same op order as _fused_ppoly_eval's time Horner.
        """
        return ((c[:, 0] * dt + c[:, 1]) * dt + c[:, 2]) * dt + c[:, 3]

    @staticmethod
    def _backend_time_interval(xp, data, t):
        """Clamped time-interval lookup of _eval_radial_lm_timedep.

        Returns (i_t, dt = t - tgrid[i_t]); outside tgrid the clamped index
        reproduces scipy's edge-interval cubic extrapolation.
        """
        tg = asarray_on_device(xp, data["tgrid"], device_of(t))
        n_t = data["tgrid"].shape[0]
        i_t = xp.clip(xp.searchsorted(tg, t, side="right") - 1, 0, n_t - 2)
        return i_t, t - tg[i_t]

    def _backend_radial(self, xp, r, t, mode=0):
        """Vectorized R_lm(r[, t]) (and dR/dr, d2R/dr2) for every component.

        Mirrors _eval_R_lm / _eval_dR_lm / _eval_d2R_lm (static) and
        _eval_radial_lm_timedep (time-dependent): quintic Horner on the
        power-basis I_inner/I_outer coefficients in-grid, the analytic
        constant-density extrapolation below rmin and the point-mass form
        above rmax. For time-dependent instances the effective radial
        coefficients (and the rmin/rmax boundary integrals the
        extrapolations depend on) are first obtained by a cubic Horner in
        (t - t_k), exactly as in _fused_ppoly_eval. Each piecewise branch is
        evaluated with a 'safe' radius so its dead side stays finite under
        eager backends / autodiff.

        Parameters
        ----------
        xp : module
            Array namespace.
        r : array
            1-D backend array of spherical radii.
        t : array
            1-D backend array of times (broadcast with r); ignored for
            static instances.
        mode : int
            0=value, 1=value+deriv, 2=value+deriv+2nd deriv.

        Returns
        -------
        (R, dR, d2R) : tuple of arrays of shape (n_comp, len(r))
            dR / d2R are None when not requested via mode.

        Notes
        -----
        - 2026-06-09 - Written - Bovy (UofT)
        - 2026-06-10 - Added the time-dependent path - Bovy (UofT)
        """
        data = self._backend_tables()
        # the tables stay float64 (precision is the point; the callers
        # exit-cast) but must live on the input's device (CUDA support)
        dev = device_of(r)
        breaks = asarray_on_device(xp, data["breaks"], dev)
        lf = asarray_on_device(xp, data["lf"], dev)
        inv_lp3 = asarray_on_device(xp, data["inv_lp3"], dev)
        inv_2ml = asarray_on_device(xp, data["inv_2ml"], dev)
        l2mask = asarray_on_device(xp, data["l2mask"], dev)
        rmin_pow = asarray_on_device(xp, data["rmin_pow_2ml"], dev)
        rmin = float(data["breaks"][0])
        rmax = float(data["breaks"][-1])
        nint = data["breaks"].shape[0] - 1
        below = r < rmin
        in_grid = r <= rmax
        # --- in-grid branch: searchsorted + Horner on r clipped into the grid
        r_in = xp.clip(r, rmin, rmax)
        idx = xp.clip(xp.searchsorted(breaks, r_in, side="right") - 1, 0, nint - 1)
        dr = r_in - breaks[idx]
        if self._tdep:
            # effective radial PPoly coefficients and boundary integrals at
            # time t: gather (time interval, radial interval) then cubic
            # Horner in dt (the vectorized _fused_ppoly_eval)
            i_t, dt = self._backend_time_interval(xp, data, t)
            flat = i_t * nint + idx
            cin = self._backend_horner_t(
                asarray_on_device(xp, data["inner_tc"], dev)[:, :, :, flat], dt
            )  # (n_comp, 6, N)
            cout = self._backend_horner_t(
                asarray_on_device(xp, data["outer_tc"], dev)[:, :, :, flat], dt
            )
            P_rho0 = self._backend_horner_t(
                asarray_on_device(xp, data["dIin_rmin_tc"], dev)[:, :, i_t], dt
            ) / asarray_on_device(xp, data["rmin_pow_lp2"], dev)
            I_outer_rmin = self._backend_horner_t(
                asarray_on_device(xp, data["Iout_rmin_tc"], dev)[:, :, i_t], dt
            )
            # I_inner(rmax, t): dt-Horner of the last interval's stack, then
            # the quintic Horner at dr_max (same op order as the numpy path)
            clast = self._backend_horner_t(
                asarray_on_device(xp, data["inner_last_tc"], dev)[:, :, :, i_t], dt
            )  # (n_comp, 6, N)
            dr_max = float(data["breaks"][-1] - data["breaks"][-2])
            I_inner_rmax = clast[:, 0]
            for k in range(1, 6):
                I_inner_rmax = I_inner_rmax * dr_max + clast[:, k]
        else:
            cin = asarray_on_device(xp, data["inner_c"], dev)[:, :, idx]
            cout = asarray_on_device(xp, data["outer_c"], dev)[:, :, idx]
            P_rho0 = asarray_on_device(xp, data["P_rho0"], dev)
            I_outer_rmin = asarray_on_device(xp, data["I_outer_rmin"], dev)
            I_inner_rmax = asarray_on_device(xp, data["I_inner_rmax"], dev)

        def _val(c):
            out = c[:, 0]
            for k in range(1, 6):
                out = out * dr + c[:, k]
            return out

        def _d1(c):
            out = 5.0 * c[:, 0]
            for k, f in zip(range(1, 5), (4.0, 3.0, 2.0, 1.0)):
                out = out * dr + f * c[:, k]
            return out

        def _d2(c):
            out = 20.0 * c[:, 0]
            for k, f in zip(range(1, 4), (12.0, 6.0, 2.0)):
                out = out * dr + f * c[:, k]
            return out

        I_in, I_out = _val(cin), _val(cout)
        rl = r_in**lf
        rml = r_in ** (-(lf + 1.0))
        R_in = rml * I_in + rl * I_out
        # --- below-grid branch (constant density rho(rmin) below rmin):
        # r^{-(l+1)} I_inner_ext simplifies to P_rho0/(l+3) r^2, so the only
        # negative powers act on rb, which is guarded away from 0
        rb = xp.where(xp.logical_and(below, r > 0.0), r, rmin)
        extra = xp.where(
            l2mask,
            P_rho0 * xp.log(rmin / rb),
            P_rho0 * inv_2ml * (rmin_pow - rb ** (2.0 - lf)),
        )
        I_outer_ext = I_outer_rmin + extra
        R_below = P_rho0 * inv_lp3 * rb * rb + rb**lf * I_outer_ext
        # --- above-grid branch (point mass: I_outer(rmax) = 0)
        ra = xp.where(r > rmax, r, rmax)
        R_above = I_inner_rmax * ra ** (-(lf + 1.0))
        Rlm = xp.where(below, R_below, xp.where(in_grid, R_in, R_above))
        dRlm, d2Rlm = None, None
        if mode >= 1:
            dI_in, dI_out = _d1(cin), _d1(cout)
            dR_in = (
                -(lf + 1.0) * r_in ** (-(lf + 2.0)) * I_in
                + rml * dI_in
                + lf * r_in ** (lf - 1.0) * I_out
                + rl * dI_out
            )
            dR_below = (
                -(lf + 1.0) * P_rho0 * inv_lp3 * rb
                + lf * rb ** (lf - 1.0) * I_outer_ext
            )
            dR_above = -(lf + 1.0) * I_inner_rmax * ra ** (-(lf + 2.0))
            dRlm = xp.where(below, dR_below, xp.where(in_grid, dR_in, dR_above))
        if mode >= 2:
            d2I_in, d2I_out = _d2(cin), _d2(cout)
            d2R_in = (
                (lf + 1.0) * (lf + 2.0) * r_in ** (-(lf + 3.0)) * I_in
                - 2.0 * (lf + 1.0) * r_in ** (-(lf + 2.0)) * dI_in
                + rml * d2I_in
                + lf * (lf - 1.0) * r_in ** (lf - 2.0) * I_out
                + 2.0 * lf * r_in ** (lf - 1.0) * dI_out
                + rl * d2I_out
            )
            d2R_below = (
                (lf + 1.0) * (lf + 2.0) * P_rho0 * inv_lp3
                + lf * (lf - 1.0) * rb ** (lf - 2.0) * I_outer_ext
                - (2.0 * lf + 1.0) * P_rho0
            )
            d2R_above = (lf + 1.0) * (lf + 2.0) * I_inner_rmax * ra ** (-(lf + 3.0))
            d2Rlm = xp.where(below, d2R_below, xp.where(in_grid, d2R_in, d2R_above))
        return Rlm, dRlm, d2Rlm

    def _backend_angular(self, xp, costheta, phi, deriv=0):
        """Per-component Legendre tables and cos/sin(m phi) factors.

        Returns (tables, trig, dtrig): ``tables`` is a tuple of (n_comp, N)
        arrays (P, and its requested x-derivatives) selected per component
        from the assoc_legendre (L, M) table; ``trig`` is cos(m phi) for
        cosine components / sin(m phi) for sine components; ``dtrig`` its
        phi-derivative.

        Notes
        -----
        - 2026-06-09 - Written - Bovy (UofT)
        """
        data = self._backend_tables()
        # index/constant tables on the input's device (CUDA support)
        dev = device_of(costheta)
        l_idx = asarray_on_device(xp, data["l_idx"], dev)
        m_idx = asarray_on_device(xp, data["m_idx"], dev)
        mf = asarray_on_device(xp, data["mf"], dev)
        is_sin = asarray_on_device(xp, data["is_sin"], dev)
        tables = assoc_legendre(self._L, self._M, costheta, deriv=deriv)
        if deriv == 0:
            tables = (tables,)
        sel = tuple(xp.moveaxis(tab[..., l_idx, m_idx], -1, 0) for tab in tables)
        mphi = mf * phi
        cosm = xp.cos(mphi)
        sinm = xp.sin(mphi)
        trig = xp.where(is_sin, sinm, cosm)
        dtrig = mf * xp.where(is_sin, cosm, -sinm)
        return sel, trig, dtrig

    def _backend_evaluate(self, xp, R, z, phi, t):
        """Backend-array (jax/torch) version of _evaluate."""
        data = self._backend_tables()
        Rf, zf, phif, tf, shape = self._backend_prepare(xp, R, z, phi, t)
        r = xp.sqrt(Rf * Rf + zf * zf)
        ctheta = xp.cos(xp.arctan2(Rf, zf))
        (PP,), trig, _ = self._backend_angular(xp, ctheta, phif, deriv=0)
        Rlm, _, _ = self._backend_radial(xp, r, tf, mode=0)
        out = xp.sum(PP * trig * Rlm, axis=0)
        # exact center: the numpy path returns R_00(rmin[, t]) * P_00
        if self._tdep:
            # R_00(rmin, t) = rmin^{-1} I_inner,00(rmin, t) + I_outer,00(rmin, t)
            # (in-grid branch of _eval_radial_lm_timedep at i_r = 0, dr = 0)
            i_t, dt = self._backend_time_interval(xp, data, tf)
            dev = device_of(r)
            cin0 = asarray_on_device(xp, data["Iin_rmin_tc"], dev)[0][:, i_t]
            cout0 = asarray_on_device(xp, data["Iout_rmin_tc"], dev)[0][:, i_t]
            Iin00 = ((cin0[0] * dt + cin0[1]) * dt + cin0[2]) * dt + cin0[3]
            Iout00 = ((cout0[0] * dt + cout0[1]) * dt + cout0[2]) * dt + cout0[3]
            rmin = float(data["breaks"][0])
            R00 = rmin ** (-1.0) * Iin00 + Iout00
        else:
            R00 = data["R00"]
        out = xp.where(r == 0.0, R00, out)
        return xp.reshape(out, shape)

    def _backend_spher_forces(self, xp, r, ctheta, stheta, phi, t):
        """Backend version of _compute_spher_forces_at_point (vectorized)."""
        (PP, dPP), trig, dtrig = self._backend_angular(xp, ctheta, phi, deriv=1)
        Rlm, dRlm, _ = self._backend_radial(xp, r, t, mode=1)
        dPhi_dr = xp.sum(PP * trig * dRlm, axis=0)
        dPhi_dtheta = xp.sum(dPP * (-stheta) * trig * Rlm, axis=0)
        dPhi_dphi = xp.sum(PP * dtrig * Rlm, axis=0)
        return -dPhi_dr, -dPhi_dtheta, -dPhi_dphi

    def _backend_cyl_force(self, xp, which, R, z, phi, t):
        """Backend-array version of the mixin's _Rforce/_zforce/_phitorque."""
        if not self.isNonAxi and phi is None:
            phi = 0.0
        Rf, zf, phif, tf, shape = self._backend_prepare(xp, R, z, phi, t)
        r = xp.sqrt(Rf * Rf + zf * zf)
        theta = xp.arctan2(Rf, zf)
        Fr, Ftheta, Fphi = self._backend_spher_forces(
            xp, r, xp.cos(theta), xp.sin(theta), phif, tf
        )
        rsafe = xp.where(r == 0.0, 1.0, r)
        if which == "R":
            out = Rf / rsafe * Fr + zf / rsafe**2 * Ftheta
        elif which == "z":
            out = zf / rsafe * Fr - Rf / rsafe**2 * Ftheta
        else:  # phi torque
            out = Fphi
        out = xp.where(r == 0.0, 0.0, out)
        return xp.reshape(out, shape)

    def _backend_spher_2nd_derivs(self, xp, r, ctheta, stheta, phi, t):
        """Backend version of _compute_spher_2nd_derivs_at_point (vectorized)."""
        (PP, dPP, d2PP), trig, dtrig = self._backend_angular(xp, ctheta, phi, deriv=2)
        Rlm, dRlm, d2Rlm = self._backend_radial(xp, r, t, mode=2)
        mf = asarray_on_device(xp, self._backend_tables()["mf"], device_of(r))
        dP_dtheta = dPP * (-stheta)
        d2P_dtheta2 = d2PP * stheta**2 - dPP * ctheta
        Phi_r = xp.sum(PP * trig * dRlm, axis=0)
        Phi_t = xp.sum(dP_dtheta * trig * Rlm, axis=0)
        Phi_rr = xp.sum(PP * trig * d2Rlm, axis=0)
        Phi_tt = xp.sum(d2P_dtheta2 * trig * Rlm, axis=0)
        Phi_pp = xp.sum(-(mf**2) * PP * trig * Rlm, axis=0)
        Phi_rt = xp.sum(dP_dtheta * trig * dRlm, axis=0)
        Phi_rp = xp.sum(PP * dtrig * dRlm, axis=0)
        Phi_tp = xp.sum(dP_dtheta * dtrig * Rlm, axis=0)
        return Phi_rr, Phi_tt, Phi_pp, Phi_rt, Phi_rp, Phi_tp, Phi_r, Phi_t

    def _backend_cyl_2nd_deriv(self, xp, deriv_type, R, z, phi, t):
        """Backend-array version of the mixin's _evaluate_cyl_2nd_deriv."""
        if not self.isNonAxi and phi is None:
            phi = 0.0
        Rf, zf, phif, tf, shape = self._backend_prepare(xp, R, z, phi, t)
        r = xp.sqrt(Rf * Rf + zf * zf)
        theta = xp.arctan2(Rf, zf)
        ctheta = xp.cos(theta)
        stheta = xp.sin(theta)
        if self._M > 1:
            # mirror the numpy path's pole clamp (d2P/dx2 diverges at the
            # poles for m > 0); guard the dead side of the where
            polemask = xp.abs(1.0 - ctheta * ctheta) < 1e-14
            ctclamp = xp.sign(ctheta) * (1.0 - 1e-7)
            ctheta = xp.where(polemask, ctclamp, ctheta)
            stheta = xp.where(polemask, xp.sqrt(1.0 - ctclamp**2), stheta)
        (Phi_rr, Phi_tt, Phi_pp, Phi_rt, Phi_rp, Phi_tp, Phi_r, Phi_t) = (
            self._backend_spher_2nd_derivs(xp, r, ctheta, stheta, phif, tf)
        )
        rs = xp.where(r == 0.0, 1.0, r)
        rs2 = rs * rs
        rs3 = rs * rs2
        rs4 = rs2 * rs2
        if deriv_type == "R2":
            out = (
                Phi_rr * Rf * Rf / rs2
                + Phi_r * zf * zf / rs3
                + 2.0 * Phi_rt * Rf * zf / rs3
                + Phi_tt * zf * zf / rs4
                + Phi_t * (-2.0 * Rf * zf) / rs4
            )
        elif deriv_type == "z2":
            out = (
                Phi_rr * zf * zf / rs2
                + Phi_r * Rf * Rf / rs3
                - 2.0 * Phi_rt * zf * Rf / rs3
                + Phi_tt * Rf * Rf / rs4
                + Phi_t * 2.0 * Rf * zf / rs4
            )
        elif deriv_type == "Rz":
            out = (
                Phi_rr * Rf * zf / rs2
                + Phi_r * (-Rf * zf) / rs3
                + Phi_rt * (zf * zf - Rf * Rf) / rs3
                + Phi_tt * (-zf * Rf) / rs4
                + Phi_t * (Rf * Rf - zf * zf) / rs4
            )
        elif deriv_type == "phi2":
            out = Phi_pp
        elif deriv_type == "Rphi":
            out = Phi_rp * Rf / rs + Phi_tp * zf / rs2
        elif deriv_type == "phiz":
            out = Phi_rp * zf / rs + Phi_tp * (-Rf) / rs2
        out = xp.where(r == 0.0, 0.0, out)
        return xp.reshape(out, shape)

    def _backend_dens(self, xp, R, z, phi, t):
        """Backend-array (jax/torch) version of _dens."""
        data = self._backend_tables()
        if not self.isNonAxi and phi is None:
            phi = 0.0
        Rf, zf, phif, tf, shape = self._backend_prepare(xp, R, z, phi, t)
        r = xp.sqrt(Rf * Rf + zf * zf)
        ctheta = xp.cos(xp.arctan2(Rf, zf))
        (PP,), trig, _ = self._backend_angular(xp, ctheta, phif, deriv=0)
        dev = device_of(r)
        rho_breaks = asarray_on_device(xp, data["rho_breaks"], dev)
        rmin = float(data["rho_breaks"][0])
        rmax = float(data["rho_breaks"][-1])
        nint = data["rho_breaks"].shape[0] - 1
        # below rmin the numpy path clamps r to rmin; above rmax it returns 0
        rd = xp.clip(r, rmin, rmax)
        idx = xp.clip(xp.searchsorted(rho_breaks, rd, side="right") - 1, 0, nint - 1)
        dr = rd - rho_breaks[idx]
        if self._tdep:
            # effective cubic-in-r spline coefficients at time t (linearity
            # of spline interpolation in the data; cf. _backend_tdep_data)
            i_t, dt = self._backend_time_interval(xp, data, tf)
            flat = i_t * nint + idx
            c = self._backend_horner_t(
                asarray_on_device(xp, data["rho_tc"], dev)[:, :, :, flat], dt
            )
        else:
            c = asarray_on_device(xp, data["rho_c"], dev)[:, :, idx]
        rho = ((c[:, 0] * dr + c[:, 1]) * dr + c[:, 2]) * dr + c[:, 3]
        out = xp.sum(PP * trig * rho, axis=0)
        out = xp.where(r > rmax, 0.0, out)
        return xp.reshape(out, shape)

    def OmegaP(self):
        return 0

    def _serialize_for_c(self):
        """Serialize MultipoleExpansionPotential data for C consumption.

        I_inner/I_outer BPoly splines are converted to PPoly and their
        coefficients are passed directly (interval-major order) so C can
        evaluate them via Horner's method with exact derivative parity.
        Rho splines are sampled at grid points for GSL cubic interpolation.

        Uses the BPoly breakpoints as the radial grid (not self._rgrid), since
        the PPoly coefficients are defined relative to these breakpoints.

        Static data layout (Nt=0):
            Nr, L, M, isNonAxi,
            rgrid (Nr),
            amp, Nt=0,
            per (l,m):
                I_inner_cos PPoly coeffs (6*(Nr-1)),
                I_outer_cos PPoly coeffs (6*(Nr-1)),
                rho_cos values (Nr),
                [if m>0: I_inner_sin, I_outer_sin, rho_sin likewise]

        Time-dependent data layout (Nt>0):
            Nr, L, M, isNonAxi,
            rgrid (Nr),
            amp, Nt, tgrid (Nt),
            per (l,m):
                I_inner_cos time-PPoly (4*(Nt-1)*6*(Nr-1)),
                I_outer_cos time-PPoly (4*(Nt-1)*6*(Nr-1)),
                rho_cos time-PPoly (4*(Nt-1)*Nr),
                [if m>0: sin components likewise]

        Time-PPoly layout per block: data[i_t * 4 * n + k * n + j]
        where i_t is the time interval, k is the cubic power (0..3),
        and j indexes the r-coefficients.
        """
        if self._tdep:
            rgrid = self._rgrid
            Nr, L, M = len(rgrid), self._L, self._M
            Nt = len(self._tgrid)
            # Use numpy arrays and concatenate for speed (avoids creating
            # millions of Python float objects)
            chunks = [
                numpy.array([Nr, L, M, int(self.isNonAxi)], dtype=numpy.float64),
                numpy.asarray(rgrid, dtype=numpy.float64),
                numpy.array([self._amp, Nt], dtype=numpy.float64),
                numpy.asarray(self._tgrid, dtype=numpy.float64),
            ]
            for l in range(L):
                for m in range(min(l + 1, M)):
                    interp_sets = [
                        (
                            self._I_inner_cos_interp[l][m],
                            self._I_outer_cos_interp[l][m],
                            self._rho_cos_interp[l][m],
                        ),
                    ]
                    if m > 0:
                        interp_sets.append(
                            (
                                self._I_inner_sin_interp[l][m],
                                self._I_outer_sin_interp[l][m],
                                self._rho_sin_interp[l][m],
                            ),
                        )
                    for I_inner_cs, I_outer_cs, rho_cs in interp_sets:
                        # CubicSpline.c has shape (4, Nt-1, n_coeffs)
                        # Transpose to (Nt-1, n_coeffs, 4) for cache-friendly
                        # fused time+radial evaluation: 4 cubic coefficients
                        # are contiguous per radial coefficient
                        chunks.append(
                            numpy.ascontiguousarray(
                                I_inner_cs.c.transpose(1, 2, 0)
                            ).ravel()
                        )
                        chunks.append(
                            numpy.ascontiguousarray(
                                I_outer_cs.c.transpose(1, 2, 0)
                            ).ravel()
                        )
                        chunks.append(
                            numpy.ascontiguousarray(rho_cs.c.transpose(1, 2, 0)).ravel()
                        )
            return numpy.concatenate(chunks)
        # Static path (Nt=0)
        # Use BPoly breakpoints as the grid: PPoly coefficients are defined
        # relative to these breakpoints, so C must use them for interval lookup
        rgrid = self._I_inner_cos[0][0].x
        Nr, L, M = len(rgrid), self._L, self._M
        args = [Nr, L, M, int(self.isNonAxi)]
        args.extend(rgrid)
        args.append(self._amp)
        args.append(0)  # Nt=0 for static
        for l in range(L):
            for m in range(min(l + 1, M)):
                for I_inner, I_outer, rho_sp in [
                    (
                        self._I_inner_cos[l][m],
                        self._I_outer_cos[l][m],
                        self._rho_cos_splines[l][m],
                    ),
                    (
                        self._I_inner_sin[l][m] if m > 0 else None,
                        self._I_outer_sin[l][m] if m > 0 else None,
                        self._rho_sin_splines[l][m] if m > 0 else None,
                    ),
                ]:
                    if I_inner is None:
                        continue
                    # Convert BPoly to PPoly and flatten interval-major
                    pp_inner = PPoly.from_bernstein_basis(I_inner)
                    args.extend(pp_inner.c.ravel(order="F"))
                    pp_outer = PPoly.from_bernstein_basis(I_outer)
                    args.extend(pp_outer.c.ravel(order="F"))
                    # Rho: sampled values for GSL cubic spline
                    args.extend(rho_sp(rgrid))
        return args
