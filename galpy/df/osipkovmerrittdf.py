# Class that implements anisotropic DFs of the Osipkov-Merritt type
import numpy
from scipy import integrate, interpolate, special

from ..backend import as_numpy, device_of, get_namespace
from ..backend import random as grandom
from ..backend import resolve_namespace
from ..backend.quadrature import fixed_quad, nested_quad
from ..potential import evaluateDensities
from ..potential.Potential import _evaluatePotentials
from ..util import conversion
from .eddingtondf import eddingtondf
from .sphericaldf import (
    _QUAD_N_VMOM,
    _QUAD_N_VMOM2D,
    anisotropicsphericaldf,
    sphericaldf,
)


# This is the general Osipkov-Merritt superclass, implementation of general
# formula can be found following this class
class _osipkovmerrittdf(anisotropicsphericaldf):
    """General Osipkov-Merritt superclass with useful functions for any DF of the Osipkov-Merritt type."""

    def __init__(
        self, pot=None, denspot=None, ra=1.4, rmax=None, scale=None, ro=None, vo=None
    ):
        """
        Initialize a DF with Osipkov-Merritt anisotropy.

        Parameters
        ----------
        pot : Potential instance or a combined potential formed using addition (pot1+pot2+…), optional
            Default: None
        denspot : Potential instance or a combined potential formed using addition (pot1+pot2+…) that represent the density of the tracers (assumed to be spherical; if None, set equal to pot), optional
            Default: None
        ra : float or Quantity, optional
            Anisotropy radius. Default: 1.4
        rmax : float or Quantity, optional
            Maximum radius to consider; DF is cut off at E = Phi(rmax). Default: None
        scale : float or Quantity, optional
            Characteristic scale radius to aid sampling calculations. Not necessary, and will also be overridden by value from pot if available. Default: None
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2020-11-12 - Written - Bovy (UofT)

        """
        anisotropicsphericaldf.__init__(
            self, pot=pot, denspot=denspot, rmax=rmax, scale=scale, ro=ro, vo=vo
        )
        self._ra = conversion.parse_length(ra, ro=self._ro)
        self._ra2 = self._ra**2.0

    def _call_internal(self, *args):
        """
        Evaluate the DF for an Osipkov-Merritt-anisotropy DF

        Parameters
        ----------
        E : float
            The energy
        L : float
            The angular momentum

        Returns
        -------
        float
            The value of the DF

        Notes
        -----
        - 2020-11-12 - Written - Bovy (UofT)

        """
        E, L, _ = args
        return self.fQ(-E - 0.5 * L**2.0 / self._ra2)

    def _dMdE(self, E):
        if not hasattr(self, "_rphi"):
            self._rphi = self._setup_rphi_interpolator()
        xp = resolve_namespace(E)
        if xp is numpy:

            def Lintegrand(t, L2lim, E):
                return self((E, numpy.sqrt(L2lim - t**2.0)), use_physical=False)

            # Integrate where Q > 0

            out = (
                16.0
                * numpy.pi**2.0
                * numpy.array(
                    [
                        integrate.quad(
                            lambda r: (
                                r
                                * integrate.quad(
                                    Lintegrand,
                                    numpy.sqrt(
                                        numpy.amax(
                                            [
                                                (0.0),
                                                (
                                                    2.0
                                                    * r**2.0
                                                    * (
                                                        tE
                                                        - _evaluatePotentials(
                                                            self._pot, r, 0.0
                                                        )
                                                    )
                                                    + 2.0 * tE * self._ra2
                                                ),
                                            ]
                                        )
                                    ),
                                    numpy.sqrt(
                                        2.0
                                        * r**2.0
                                        * (tE - _evaluatePotentials(self._pot, r, 0.0))
                                    ),
                                    args=(
                                        2.0
                                        * r**2.0
                                        * (tE - _evaluatePotentials(self._pot, r, 0.0)),
                                        tE,
                                    ),
                                )[0]
                            ),
                            0.0,
                            self._rphi(tE),
                        )[0]
                        for ii, tE in enumerate(E)
                    ]
                )
            )
            # Numerical issues can make the integrand's sqrt argument negative, only
            # happens at dMdE ~ 0, so just set to zero
            out[numpy.isnan(out)] = 0.0
            return out.reshape(E.shape)
        # jax/torch: nested GL over the Q>0 region after r = rphi - s^2 (outer
        # turning point) and t = Lmax sin(phi) with phi clustered as phi_low+span*w^2
        # (cancels the fQ sqrt(Q) endpoint at the Q=0 boundary phi_low)
        Eb = xp.asarray(E) * 1.0
        rphiE = xp.asarray(self._rphi(E)) * 1.0
        rpos = rphiE > 0.0
        smax = xp.where(rpos, xp.sqrt(xp.where(rpos, rphiE, xp.ones_like(rphiE))), 0.0)
        E_bb = Eb[..., None, None]
        rphi_bb = xp.where(rpos, rphiE, xp.ones_like(rphiE))[..., None, None]

        def _integrand(s, w):
            r = rphi_bb - s**2.0
            twoRsq = 2.0 * r**2.0 * (E_bb - _evaluatePotentials(self._pot, r, 0.0))
            live = twoRsq > 0.0
            Lmax = xp.where(
                live, xp.sqrt(xp.where(live, twoRsq, xp.ones_like(twoRsq))), 0.0
            )
            Llow2 = twoRsq + 2.0 * E_bb * self._ra2  # Q>=0 boundary in t^2
            Llow2 = xp.where(Llow2 > 0.0, Llow2, xp.zeros_like(Llow2))
            ratio = xp.sqrt(Llow2) / xp.where(live, Lmax, xp.ones_like(Lmax))
            ratio = xp.where(ratio < 1.0, ratio, xp.ones_like(ratio))
            phi_low = xp.arcsin(ratio)
            span = numpy.pi / 2.0 - phi_low
            phi = phi_low + span * w**2.0
            L = Lmax * xp.cos(phi)
            return (
                r
                * self._call_internal(E_bb, L, None)
                * L
                * span
                * (2.0 * w)
                * (2.0 * s)
            )

        return (
            16.0
            * numpy.pi**2.0
            * nested_quad(
                xp,
                _integrand,
                [[0.0, smax[..., None, None]], [0.0, 1.0]],
                n=_QUAD_N_VMOM2D,
            )
        )

    def _sample_eta(self, r, n=1, key=None):
        """Sample the angle eta which defines radial vs tangential velocities

        The cos(eta) inverse-CDF is CLOSED-FORM (r-dependent through
        A = (r/ra)^2), so no grid is needed: ``key=None`` draws from the global
        ``numpy.random`` (byte-identical); a backend key draws backend uniforms
        (magnitude + symmetric sign) and evaluates the SAME analytic inversion
        in-namespace -- so eta is a backend array differentiable in r."""
        # cumulative distribution of x = cos eta satisfies
        # x/(sqrt(A+1 -A* x^2)) = 2 b - 1 = c
        # where b \in [0,1] and A = (r/ra)^2
        # Solved by
        # x = c sqrt(1+[r/ra]^2) / sqrt( [r/ra]^2 c^2 + 1 ) for c > 0 [b > 0.5]
        # and symmetric wrt c
        if key is None:
            # numpy path (byte-identical)
            c = numpy.random.uniform(size=n)
            x = (
                c
                * numpy.sqrt(1 + r**2.0 / self._ra2)
                / numpy.sqrt(r**2.0 / self._ra2 * c**2.0 + 1)
            )
            x *= numpy.random.choice([1.0, -1.0], size=n)
            return numpy.arccos(x)
        # backend key: same analytic inversion, in-namespace and differentiable
        # in r; independent sub-keys for the magnitude uniform and the sign
        kc, ks = grandom.split(key, 2)
        c = grandom.uniform(kc, n)
        xp = get_namespace(c)
        A = xp.asarray(r) ** 2.0 / self._ra2  # coerce: r is the backend sample r
        x = c * xp.sqrt(1.0 + A) / xp.sqrt(A * c**2.0 + 1.0)
        sign = grandom.choice(ks, xp.asarray([1.0, -1.0]), shape=n)
        return xp.arccos(x * sign)

    def _p_v_at_r(self, v, r):
        """p( v*sqrt[1+r^2/ra^2*sin^2eta] | r) used in sampling"""
        xp = resolve_namespace(v, r)
        if hasattr(self, "_logfQ_interp"):
            # scipy interpolator (general OM df) is numpy-only; sampling numpy-side
            # (the potential eval stays on the active backend, so pull it numpy-side
            # before the scipy spline; no-op on the numpy path)
            v, r = as_numpy(v), as_numpy(r)
            return (
                numpy.exp(
                    self._logfQ_interp(
                        -as_numpy(_evaluatePotentials(self._pot, r, 0)) - 0.5 * v**2.0
                    )
                )
                * v**2.0
            )
        if xp is numpy:
            return (
                self.fQ(-_evaluatePotentials(self._pot, r, 0) - 0.5 * v**2.0) * v**2.0
            )
        # coerce: a forced backend sees numpy sampling grids; torch potentials
        # reject numpy coords
        v, r = xp.asarray(v) * 1.0, xp.asarray(r) * 1.0
        return self.fQ(-_evaluatePotentials(self._pot, r, 0) - 0.5 * v**2.0) * v**2.0

    def _sample_v(self, r, eta, n=1, key=None):
        """Generate velocity samples

        ``key=None`` is the byte-identical numpy path; a backend key returns a
        backend velocity (the base pvr sampler is native, so the r/eta transform
        below runs in-namespace, differentiable in r and eta)."""
        # Use super-class method to obtain v*[1+r^2/ra^2*sin^2eta]
        out = super()._sample_v(r, eta, n=n, key=key)
        # Transform to v
        if key is None:
            return out / numpy.sqrt(1.0 + r**2.0 / self._ra2 * numpy.sin(eta) ** 2.0)
        xp = get_namespace(out, eta)
        rb = xp.asarray(r) ** 2.0
        return out / xp.sqrt(1.0 + rb / self._ra2 * xp.sin(eta) ** 2.0)

    def _vmomentdensity(self, r, n, m):
        if m % 2 == 1 or n % 2 == 1:
            return 0.0
        xp = resolve_namespace(r)
        if xp is numpy:
            return (
                2.0
                * numpy.pi
                * integrate.quad(
                    lambda v: (
                        v ** (2.0 + m + n)
                        * self.fQ(-_evaluatePotentials(self._pot, r, 0) - 0.5 * v**2.0)
                    ),
                    0.0,
                    self._vmax_at_r(self._pot, r),
                )[0]
                * special.gamma(m / 2.0 + 1.0)
                * special.gamma((n + 1) / 2.0)
                / special.gamma(0.5 * (m + n + 3.0))
                / (1 + r**2.0 / self._ra2) ** (m / 2 + 1)
            )
        # jax/torch: GL after v = vmax sin(theta), which cancels the fQ endpoint
        # singularity (power-law fQ ~ Q^{-1/2} as Q -> 0 at v = vmax); node axis trails
        rb = xp.asarray(r) * 1.0  # coerce: torch potentials reject numpy coords
        Phir_b = (xp.asarray(_evaluatePotentials(self._pot, rb, 0)) * 1.0)[..., None]
        vmax = (xp.asarray(self._vmax_at_r(self._pot, rb)) * 1.0)[..., None]

        def _integrand(theta):
            v = vmax * xp.sin(theta)
            return (
                v ** (2.0 + m + n)
                * self.fQ(-Phir_b - 0.5 * v**2.0)
                * vmax
                * xp.cos(theta)
            )

        return (
            2.0
            * numpy.pi
            * fixed_quad(
                xp,
                _integrand,
                0.0,
                numpy.pi / 2.0,
                n=_QUAD_N_VMOM,
                device=device_of(rb),
            )
            * special.gamma(m / 2.0 + 1.0)
            * special.gamma((n + 1) / 2.0)
            / special.gamma(0.5 * (m + n + 3.0))
            / (1 + rb**2.0 / self._ra2) ** (m / 2 + 1)
        )


class osipkovmerrittdf(_osipkovmerrittdf):
    """Class that implements spherical DFs with Osipkov-Merritt-type orbital anisotropy

    .. math::

        \\beta(r) = \\frac{1}{1+r_a^2/r^2}

    with :math:`r_a` the anisotropy radius for arbitrary combinations of potential and density profile.
    """

    def __init__(
        self,
        pot=None,
        denspot=None,
        ra=1.4,
        rmax=1e4,
        rmin=None,
        scale=None,
        ro=None,
        vo=None,
    ):
        """
        Initialize a DF with Osipkov-Merritt anisotropy.

        Parameters
        ----------
        pot : Potential instance or a combined potential formed using addition (pot1+pot2+…), optional
            Default: None
        denspot : Potential instance or a combined potential formed using addition (pot1+pot2+…) that represent the density of the tracers (assumed to be spherical; if None, set equal to pot), optional
            Default: None
        ra : float or Quantity, optional
            Anisotropy radius. Default: 1.4
        rmax : float or Quantity, optional
            Maximum radius to consider; DF is cut off at E = Phi(rmax). Default: None
        rmin : float or Quantity, optional
            Minimum radius to consider. For divergent potentials (Phi(0) = -inf),
            this sets the inner boundary for the energy range. Auto-detected if
            not specified.
        scale : float or Quantity, optional
            Characteristic scale radius to aid sampling calculations. Not necessary, and will also be overridden by value from pot if available. Default: None
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2021-02-07 - Written - Bovy (UofT)
        """
        _osipkovmerrittdf.__init__(
            self, pot=pot, denspot=denspot, ra=ra, rmax=rmax, scale=scale, ro=ro, vo=vo
        )
        # Because f(Q) is the same integral as the Eddington conversion, but
        # using the augmented density rawdensx(1+r^2/ra^2), we use a helper
        # eddingtondf to do this integral, hacked to use the augmented density
        self._edf = eddingtondf(
            pot=self._pot,
            denspot=self._denspot,
            scale=scale,
            rmax=rmax,
            rmin=rmin,
            ro=ro,
            vo=vo,
        )
        # Copy rmin from the internal eddingtondf
        self._rmin = self._edf._rmin
        self._edf._dnudr = (
            (
                lambda r: (
                    self._denspot._ddensdr(r) * (1.0 + r**2.0 / self._ra2)
                    + 2.0 * self._denspot.dens(r, 0, use_physical=False) * r / self._ra2
                )
            )
            if not isinstance(self._denspot, list)
            else (
                lambda r: (
                    numpy.sum([p._ddensdr(r) for p in self._denspot])
                    * (1.0 + r**2.0 / self._ra2)
                    + 2.0
                    * evaluateDensities(self._denspot, r, 0, use_physical=False)
                    * r
                    / self._ra2
                )
            )
        )
        self._edf._d2nudr2 = (
            (
                lambda r: (
                    self._denspot._d2densdr2(r) * (1.0 + r**2.0 / self._ra2)
                    + 4.0 * self._denspot._ddensdr(r) * r / self._ra2
                    + 2.0 * self._denspot.dens(r, 0, use_physical=False) / self._ra2
                )
            )
            if not isinstance(self._denspot, list)
            else (
                lambda r: (
                    numpy.sum([p._d2densdr2(r) for p in self._denspot])
                    * (1.0 + r**2.0 / self._ra2)
                    + 4.0
                    * numpy.sum([p._ddensdr(r) for p in self._denspot])
                    * r
                    / self._ra2
                    + 2.0
                    * evaluateDensities(self._denspot, r, 0, use_physical=False)
                    / self._ra2
                )
            )
        )

    def sample(
        self, R=None, z=None, phi=None, n=1, return_orbit=True, rmin=None, key=None
    ):
        # Slight over-write of superclass method to first build f(Q) interp
        # No docstring so superclass' is used
        if rmin is None:
            rmin = self._rmin
        self._ensure_fQ_interp()
        # key=None keeps the whole assembly numpy (byte-identical); a backend key
        # makes the radial, angle (native analytic eta inverse-CDF), and velocity
        # sampling backend-native (differentiable, GPU/jit-able).
        return sphericaldf.sample(
            self, R=R, z=z, phi=phi, n=n, return_orbit=return_orbit, rmin=rmin, key=key
        )

    def _ensure_fQ_interp(self):
        """Build the f(Q) interpolator if not already built."""
        if not hasattr(self, "_logfQ_interp"):
            Qs4interp = numpy.hstack(
                (
                    numpy.geomspace(1e-8, 0.5, 101, endpoint=False),
                    sorted(1.0 - numpy.geomspace(1e-8, 0.5, 101)),
                )
            )
            # scipy spline table is inherently numpy (no backend spline here);
            # under a forced backend the potential bounds and fQ come back as
            # backend scalars, so pull them numpy-side (no-op on the numpy path)
            Emin = as_numpy(self._edf._Emin)
            potInf = as_numpy(self._edf._potInf)
            Qs4interp = -(Qs4interp * (Emin - potInf) + potInf)
            fQ4interp = numpy.log(as_numpy(self.fQ(Qs4interp)))
            iindx = numpy.isfinite(fQ4interp)
            self._logfQ_interp = interpolate.InterpolatedUnivariateSpline(
                Qs4interp[iindx], fQ4interp[iindx], k=3, ext=3
            )

    def fQ(self, Q):
        """
        Calculate the f(Q) portion of an Osipkov-Merritt Hernquist distribution function

        Parameters
        ----------
        Q : float
            The Osipkov-Merritt 'energy' E-L^2/[2ra^2]

        Returns
        -------
        float
            The value of the f(Q) portion of the DF

        Notes
        -----
        - 2021-02-07 - Written - Bovy (UofT)

        """
        return self._edf.fE(-Q)
