# Class that implements isotropic spherical DFs computed using the Eddington
# formula
from __future__ import annotations

import numpy
from scipy import integrate

from ..potential import CompositePotential, evaluateR2derivs
from ..potential.Potential import _evaluatePotentials, _evaluateRforces
from ..util import conversion
from .sphericaldf import (
    _handle_rmin,
    _select_fE_extrapolator,
    isotropicsphericaldf,
    sphericaldf,
)


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
            Transition radius for numerical/extrapolation boundary. Only applicable for potentials with divergent Phi(0). For E >= Phi(rmin), f(E) is computed via numerical Eddington integration. For E < Phi(rmin) (higher binding energies), f(E) is extrapolated using a power-law fitted near the transition. For divergent potentials, this is automatically set to a small value if not specified. Raises ValueError if specified for potentials with finite Phi(0).
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

        # Handle rmin: _handle_rmin validates and raises error if rmin is
        # provided for non-divergent potentials
        self._rmin = _handle_rmin(
            rmin, pot, denspot, self._scale, self._ro, "eddingtondf"
        )

        # Check if potential diverges at r=0 (determines if extrapolation is needed)
        phi_at_zero = _evaluatePotentials(pot, 0, 0)
        self._divergent = not numpy.isfinite(phi_at_zero)

        self._dnudr = (
            self._denspot._ddensdr
            if not isinstance(self._denspot, CompositePotential)
            else lambda r: numpy.sum([p._ddensdr(r) for p in self._denspot])
        )
        self._d2nudr2 = (
            self._denspot._d2densdr2
            if not isinstance(self._denspot, CompositePotential)
            else lambda r: numpy.sum([p._d2densdr2(r) for p in self._denspot])
        )
        self._potInf = _evaluatePotentials(pot, self._rmax, 0)
        self._Emin = _evaluatePotentials(pot, self._rmin, 0)
        # Build interpolator r(pot), starting at rmin for divergent potentials
        self._rphi = self._setup_rphi_interpolator(
            r_a_min=max(1e-6, self._rmin / self._scale) if self._divergent else 1e-6
        )

    def sample(self, R=None, z=None, phi=None, n=1, return_orbit=True, rmin=None):
        # Slight over-write of superclass method to first build f(E) interp
        # No docstring so superclass' is used
        # Use self._rmin as the spatial sampling boundary (needed for mass computation)
        # The f(E) interpolator uses power-law extrapolation for E < Phi(rmin)
        if rmin is None:
            rmin = self._rmin
        self._ensure_fE_interp()
        return sphericaldf.sample(
            self, R=R, z=z, phi=phi, n=n, return_orbit=return_orbit, rmin=rmin
        )

    def _ensure_fE_interp(self):
        """Build the f(E) interpolator with appropriate extrapolation if not already built.

        For PowerSphericalPotential: uses power-law extrapolation (exact)
        For other divergent potentials: uses Padé approximant extrapolation
        """
        if not hasattr(self, "_fE_interp"):
            Es4interp = numpy.hstack(
                (
                    numpy.geomspace(1e-8, 0.5, 101, endpoint=False),
                    sorted(1.0 - numpy.geomspace(1e-4, 0.5, 101)),
                )
            )
            Es4interp = (Es4interp * (self._Emin - self._potInf) + self._potInf)[::-1]
            fE4interp = self._fE_numerical(Es4interp)
            # Select appropriate extrapolator based on potential type
            self._fE_interp = _select_fE_extrapolator(
                self._pot, Es4interp, fE4interp, E_transition=self._Emin
            )

    def _fE_numerical(self, E):
        """Compute f(E) numerically via Eddington integration (only for E >= Emin)."""
        Eint = conversion.parse_energy(E, vo=self._vo)
        Eint = numpy.atleast_1d(Eint)
        out = numpy.zeros_like(Eint, dtype=float)
        indx = (Eint < self._potInf) * (Eint >= self._Emin)
        out[indx] = numpy.array(
            [
                integrate.quad(
                    lambda t: _fEintegrand_smallr(
                        t, self._pot, tE, self._dnudr, self._d2nudr2, self._rphi(tE)
                    ),
                    0.0,
                    numpy.sqrt(self._rphi(tE)),
                    points=[0.0],
                )[0]
                for tE in Eint[indx]
            ]
        )
        out[indx] += numpy.array(
            [
                integrate.quad(
                    lambda t: _fEintegrand_larger(
                        t, self._pot, tE, self._dnudr, self._d2nudr2
                    ),
                    0.0,
                    0.5 / self._rphi(tE),
                )[0]
                for tE in Eint[indx]
            ]
        )
        return -out / (numpy.sqrt(8.0) * numpy.pi**2.0)

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
        Eint = numpy.atleast_1d(Eint)
        out = numpy.zeros_like(Eint, dtype=float)

        # For E >= Emin: compute numerically
        numerical_mask = Eint >= self._Emin
        if numpy.any(numerical_mask):
            out[numerical_mask] = self._fE_numerical(Eint[numerical_mask])

        # For E < Emin: use power-law extrapolation only if potential diverges
        extrap_mask = Eint < self._Emin
        if numpy.any(extrap_mask):
            if self._divergent:
                # Divergent potential: use power-law extrapolation
                self._ensure_fE_interp()
                out[extrap_mask] = self._fE_interp(Eint[extrap_mask])
            # else: non-divergent potential, E < Emin is unphysical, leave as 0

        # Return scalar for single-element input
        return out if len(out) > 1 else out[0]


def _fEintegrand_raw(r, pot, E, dnudr, d2nudr2):
    # The 'raw', i.e., direct integrand in the Eddington inversion
    Fr = _evaluateRforces(pot, r, 0)
    return (
        (Fr * d2nudr2(r) + dnudr(r) * evaluateR2derivs(pot, r, 0, use_physical=False))
        / Fr**2.0
        / numpy.sqrt(_evaluatePotentials(pot, r, 0) - E)
    )


def _fEintegrand_smallr(t, pot, E, dnudr, d2nudr2, rmin):
    # The integrand at small r, using transformation to deal with sqrt diverge
    return 2.0 * t * _fEintegrand_raw(t**2.0 + rmin, pot, E, dnudr, d2nudr2)


def _fEintegrand_larger(t, pot, E, dnudr, d2nudr2):
    # The integrand at large r, using transformation to deal with infinity
    return 1.0 / t**2 * _fEintegrand_raw(1.0 / t, pot, E, dnudr, d2nudr2)
