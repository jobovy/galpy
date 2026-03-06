###############################################################################
#   MultipoleExpansionPotential.py: Potential via multipole expansion of a
#   given density function
###############################################################################
import numpy
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import BPoly, InterpolatedUnivariateSpline, PPoly

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
    """

    def __init__(
        self,
        amp=1.0,
        rho_cos_splines=None,
        rho_sin_splines=None,
        rgrid=numpy.geomspace(1e-3, 30, 1_001),
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
            Cosine density coefficient splines with the spherical harmonic normalization absorbed, shape ``[L][M]``. ``rho_cos_splines[l][m]`` is a spline representing :math:`\hat{\rho}^{\cos}_{lm}(r) = \beta_{lm}\,\rho^{\cos}_{lm}(r)` as a function of ``r``, where :math:`\rho^{\cos}_{lm}(r)` are the coefficients in the real spherical-harmonic expansion of the density :math:`\rho(r,\theta,\phi) = \sum_{l,m} [\rho^{\cos}_{lm}(r)\,\cos(m\phi) + \rho^{\sin}_{lm}(r)\,\sin(m\phi)]\,P_l^m(\cos\theta)` (the real-valued form of `Eq. 12.78 <https://galaxiesbook.org/chapters/III-01.-Gravitation-in-Elliptical-Galaxies-and-Dark-Matter-Halos_3-Multipole-and-basis-function-expansions.html#mjx-eqn-eq-triaxialgrav-body-decompose-2>`__ in `Bovy 2026 <https://galaxiesbook.org>`__) and :math:`\beta_{lm} = (2-\delta_{m0})\,\sqrt{\frac{2l+1}{4\pi}\,\frac{(l-m)!}{(l+m)!}}` is the real spherical harmonic normalization factor. If ``None``, computes a default Hernquist monopole.
        rho_sin_splines : list of list of InterpolatedUnivariateSpline, optional
            Like ``rho_cos_splines`` but for the sine coefficients: ``rho_sin_splines[l][m]`` represents :math:`\hat{\rho}^{\sin}_{lm}(r) = \beta_{lm}\,\rho^{\sin}_{lm}(r)`. If ``None``, set to zero splines matching ``rho_cos_splines``.
        rgrid : numpy.ndarray, optional
            Radial grid points (1D array). Default: ``numpy.geomspace(1e-3, 30, 1_001)``.
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
        """
        Potential.__init__(self, amp=amp, ro=ro, vo=vo)
        self._rgrid = rgrid
        self._k = 3
        if rho_cos_splines is None:
            # Default: Hernquist monopole
            rho_cos_splines, rho_sin_splines = self._default_hernquist_splines(
                rgrid, self._k
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
            Density function. Can take 1 arg (r), 2 args (R, z), or 3 args (R, z, phi). Can also be a galpy Potential instance.
        L : int, optional
            Maximum spherical harmonic degree + 1 (l goes from 0 to L-1). Default: 6.
        rgrid : numpy.ndarray, optional
            Radial grid points (1D array). Default: ``numpy.geomspace(1e-3, 30, 1_001)``.
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
        """
        # Dummy instance for ro/vo parsing (following SCFPotential pattern)
        dumm = cls(ro=ro, vo=vo)
        internal_ro = dumm._ro
        internal_vo = dumm._vo
        # Parse density function
        dens_func = cls._parse_density(dens, internal_ro, internal_vo)
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
        # Compute rho_lm on radial grid
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
        if _APY_LOADED:
            numOfParam = 0
            try:
                dens(1.0, 0.0, 0.0)
                numOfParam = 3
            except TypeError:
                try:
                    dens(1.0, 0.0)
                    numOfParam = 2
                except TypeError:
                    numOfParam = 1
            if not isinstance(dens, Potential):
                param = [1.0] * numOfParam
                try:
                    dens(*param).to(units.kg / units.m**3)
                except (AttributeError, units.UnitConversionError):
                    pass
                else:
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
    def _parse_density(dens, ro, vo):
        """
        Parse the density input and return a callable taking (R, z, phi).

        Parameters
        ----------
        dens : callable or Potential
            Density function or galpy Potential instance.
        ro : float
            Distance scale for unit conversion.
        vo : float
            Velocity scale for unit conversion.

        Returns
        -------
        callable
            Density function taking (R, z, phi).

        Notes
        -----
        - 2026-02-13 - Written - Bovy (UofT)
        - 2026-03-06 - Made static with explicit ro/vo - Bovy (UofT)
        """
        # Handle galpy Potential instances
        if isinstance(dens, Potential):
            return lambda R, z, phi: dens.dens(R, z, phi, use_physical=False)
        # Determine number of parameters
        numOfParam = 0
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
        if _APY_LOADED:
            param = [1.0] * numOfParam
            _dens_unit_output = False
            try:
                dens(*param).to(units.kg / units.m**3)
            except (AttributeError, units.UnitConversionError):
                pass
            else:
                _dens_unit_output = True
            if _dens_unit_output:
                raw_dens = dens
                if numOfParam == 1:
                    return lambda R, z, phi: conversion.parse_dens(
                        raw_dens(numpy.sqrt(R**2 + z**2)),
                        ro=ro,
                        vo=vo,
                    )
                elif numOfParam == 2:
                    return lambda R, z, phi: conversion.parse_dens(
                        raw_dens(R, z), ro=ro, vo=vo
                    )
                else:
                    return lambda R, z, phi: conversion.parse_dens(
                        raw_dens(R, z, phi), ro=ro, vo=vo
                    )
        # No units, wrap based on number of params
        if numOfParam == 1:
            return lambda R, z, phi: dens(numpy.sqrt(R**2 + z**2))
        elif numOfParam == 2:
            return lambda R, z, phi: dens(R, z)
        else:
            return dens

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
                # Evaluate density at all phi nodes
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
        """
        if not self.isNonAxi and phi is None:
            phi = 0.0
        R = numpy.array(R, dtype=float)
        z = numpy.array(z, dtype=float)
        phi = numpy.array(phi, dtype=float)
        shape = numpy.broadcast_shapes(R.shape, z.shape, phi.shape)
        if shape == ():
            return self._evaluate_at_point(R, z, phi)
        R = R * numpy.ones(shape)
        z = z * numpy.ones(shape)
        phi = phi * numpy.ones(shape)
        result = numpy.zeros(shape, float)
        for idx in numpy.ndindex(*shape):
            result[idx] = self._evaluate_at_point(R[idx], z[idx], phi[idx])
        return result

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

    def _evaluate_at_point(self, R, z, phi):
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
            # At r=0, only l=0, m=0 contributes: R_00(0) = I_outer(0)
            R_00 = float(self._I_outer_cos[0][0](self._rgrid[0]))
            return R_00 * PP[0, 0]
        for l in range(L):
            for m in range(min(l + 1, M)):
                radial_cos = self._eval_R_lm(
                    r,
                    l,
                    self._I_inner_cos[l][m],
                    self._I_outer_cos[l][m],
                )
                contrib = PP[l, m] * numpy.cos(m * phi) * radial_cos
                if m > 0:
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
        """
        if not self.isNonAxi and phi is None:
            phi = 0.0
        R = numpy.array(R, dtype=float)
        z = numpy.array(z, dtype=float)
        phi = numpy.array(phi, dtype=float)
        shape = numpy.broadcast_shapes(R.shape, z.shape, phi.shape)
        if shape == ():
            return self._dens_at_point(R, z, phi)
        R = R * numpy.ones(shape)
        z = z * numpy.ones(shape)
        phi = phi * numpy.ones(shape)
        result = numpy.zeros(shape, float)
        for idx in numpy.ndindex(*shape):
            result[idx] = self._dens_at_point(R[idx], z[idx], phi[idx])
        return result

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

    def _compute_spher_forces_at_point(self, R, z, phi):
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
        cache_key = (float(R), float(z), float(phi))
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
                # dPhi/dr contribution
                dPhi_dr += PP[l, m] * cos_mphi * dradial_cos
                # dPhi/dtheta contribution: dP_l^m/dtheta = dP_l^m/d(costheta) * (-sin(theta))
                dPhi_dtheta += dPP[l, m] * (-sintheta) * cos_mphi * radial_cos
                # dPhi/dphi contribution
                dPhi_dphi += PP[l, m] * (-m * sin_mphi) * radial_cos
                if m > 0:
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

    def _compute_spher_2nd_derivs_at_point(self, R, z, phi):
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
        cache_key = (float(R), float(z), float(phi))
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
        # Clamp costheta slightly away from ±1 to avoid divergence in
        # dP/d(costheta) and d²P/d(costheta)² for m > 0; the divergent
        # terms are cancelled by sintheta factors in dP/dtheta and
        # d²P/dtheta², but inf * 0 = nan numerically
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
                    # sin(m*phi) -> -m²*sin(m*phi)
                    d2Phi_dphi2 += Plm * (-m * m * sin_mphi) * radial_sin
                    d2Phi_drdtheta += dPlm_dtheta * sin_mphi * dradial_sin
                    # sin(m*phi) -> m*cos(m*phi) for d/dphi
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
        return self._evaluate_cyl_2nd_deriv("R2", R, z, phi)

    def _z2deriv(self, R, z, phi=0.0, t=0.0):
        return self._evaluate_cyl_2nd_deriv("z2", R, z, phi)

    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        return self._evaluate_cyl_2nd_deriv("Rz", R, z, phi)

    def _phi2deriv(self, R, z, phi=0.0, t=0.0):
        return self._evaluate_cyl_2nd_deriv("phi2", R, z, phi)

    def _Rphideriv(self, R, z, phi=0.0, t=0.0):
        return self._evaluate_cyl_2nd_deriv("Rphi", R, z, phi)

    def _phizderiv(self, R, z, phi=0.0, t=0.0):
        return self._evaluate_cyl_2nd_deriv("phiz", R, z, phi)

    def OmegaP(self):
        return 0

    @staticmethod
    def _serialize_for_c(p):
        """Serialize MultipoleExpansionPotential data for C consumption.

        I_inner/I_outer BPoly splines are converted to PPoly and their
        coefficients are passed directly (interval-major order) so C can
        evaluate them via Horner's method with exact derivative parity.
        Rho splines are sampled at grid points for GSL cubic interpolation.

        Uses the BPoly breakpoints as the radial grid (not p._rgrid), since
        the PPoly coefficients are defined relative to these breakpoints.

        Data layout:
            Nr, L, M, isNonAxi,
            rgrid (Nr),
            amp,
            per (l,m):
                I_inner_cos PPoly coeffs (6*(Nr-1)),
                I_outer_cos PPoly coeffs (6*(Nr-1)),
                rho_cos values (Nr),
                [if m>0: I_inner_sin, I_outer_sin, rho_sin likewise]
        """
        # Use BPoly breakpoints as the grid: PPoly coefficients are defined
        # relative to these breakpoints, so C must use them for interval lookup
        rgrid = p._I_inner_cos[0][0].x
        Nr, L, M = len(rgrid), p._L, p._M
        args = [Nr, L, M, int(p.isNonAxi)]
        args.extend(rgrid)
        args.append(p._amp)
        for l in range(L):
            for m in range(min(l + 1, M)):
                for I_inner, I_outer, rho_sp in [
                    (
                        p._I_inner_cos[l][m],
                        p._I_outer_cos[l][m],
                        p._rho_cos_splines[l][m],
                    ),
                    (
                        p._I_inner_sin[l][m] if m > 0 else None,
                        p._I_outer_sin[l][m] if m > 0 else None,
                        p._rho_sin_splines[l][m] if m > 0 else None,
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
