# Class that implements isotropic spherical DFs computed using the Eddington
# formula
import numpy
from scipy import integrate, interpolate

from ..potential import evaluateR2derivs
from ..potential.Potential import _evaluatePotentials, _evaluateRforces
from ..util import conversion
from .sphericaldf import isotropicsphericaldf, sphericaldf


class eddingtondf(isotropicsphericaldf):
    """Class that implements isotropic spherical DFs computed using the Eddington formula

    .. math::

        f(\\mathcal{E}) = \\frac{1}{\\sqrt{8}\\,\\pi^2}\\,\\left[\\int_0^\\mathcal{E}\\mathrm{d}\\Psi\\,\\frac{1}{\\sqrt{\\mathcal{E}-\\Psi}}\\,\\frac{\\mathrm{d}^2\\rho}{\\mathrm{d}\\Psi^2} +\\frac{1}{\\sqrt{\\mathcal{E}}}\\,\\frac{\\mathrm{d}\\rho}{\\mathrm{d}\\Psi}\\Bigg|_{\\Psi=0}\\right]\\,,

    where :math:`\\Psi = -\\Phi+\\Phi(\\infty)` is the relative potential, :math:`\\mathcal{E} = \\Psi-v^2/2` is the relative (binding) energy, and :math:`\\rho` is the density of the tracer population (not necessarily the density corresponding to :math:`\\Psi` according to the Poisson equation). Note that the second term on the right-hand side is currently assumed to be zero in the code.
    """

    def __init__(self, pot=None, denspot=None, rmax=1e4, scale=None, ro=None, vo=None):
        """
        Initialize an isotropic distribution function computed using the Eddington inversion.

        Parameters
        ----------
        pot : Potential instance or list thereof
            Represents the gravitational potential (assumed to be spherical).
        denspot : Potential instance or list thereof, optional
            Represents the density of the tracers (assumed to be spherical; if None, set equal to pot).
        rmax : float or Quantity, optional
            Maximum radius to consider. DF is cut off at E = Phi(rmax).
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
        self._dnudr = (
            self._denspot._ddensdr
            if not isinstance(self._denspot, list)
            else lambda r: numpy.sum([p._ddensdr(r) for p in self._denspot])
        )
        self._d2nudr2 = (
            self._denspot._d2densdr2
            if not isinstance(self._denspot, list)
            else lambda r: numpy.sum([p._d2densdr2(r) for p in self._denspot])
        )
        self._potInf = _evaluatePotentials(pot, self._rmax, 0)
        self._Emin = _evaluatePotentials(pot, 0.0, 0)
        # Build interpolator r(pot)
        self._rphi = self._setup_rphi_interpolator()

    def sample(self, R=None, z=None, phi=None, n=1, return_orbit=True, rmin=0.0):
        # Slight over-write of superclass method to first build f(E) interp
        # No docstring so superclass' is used
        if not hasattr(self, "_fE_interp"):
            Es4interp = numpy.hstack(
                (
                    numpy.geomspace(1e-8, 0.5, 101, endpoint=False),
                    sorted(1.0 - numpy.geomspace(1e-4, 0.5, 101)),
                )
            )
            Es4interp = (Es4interp * (self._Emin - self._potInf) + self._potInf)[::-1]
            fE4interp = self.fE(Es4interp)
            iindx = numpy.isfinite(fE4interp)
            self._fE_interp = interpolate.InterpolatedUnivariateSpline(
                Es4interp[iindx], fE4interp[iindx], k=3, ext=3
            )
        return sphericaldf.sample(
            self, R=R, z=z, phi=phi, n=n, return_orbit=return_orbit, rmin=rmin
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
        out = numpy.zeros_like(Eint)
        indx = (Eint < self._potInf) * (Eint >= self._Emin)
        # Split integral at twice the lower limit to deal with divergence at
        # the lower end and infinity at the upper end
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
