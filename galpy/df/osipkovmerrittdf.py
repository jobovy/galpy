# Class that implements anisotropic DFs of the Osipkov-Merritt type
import numpy
from scipy import integrate, interpolate, special

from ..potential import evaluateDensities
from ..potential.Potential import _evaluatePotentials
from ..util import conversion
from .eddingtondf import eddingtondf
from .sphericaldf import anisotropicsphericaldf, sphericaldf


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
        pot : Potential instance or list thereof, optional
            Default: None
        denspot : Potential instance or list thereof that represent the density of the tracers (assumed to be spherical; if None, set equal to pot), optional
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

        def Lintegrand(t, L2lim, E):
            return self((E, numpy.sqrt(L2lim - t**2.0)), use_physical=False)

        # Integrate where Q > 0

        out = (
            16.0
            * numpy.pi**2.0
            * numpy.array(
                [
                    integrate.quad(
                        lambda r: r
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
                                                - _evaluatePotentials(self._pot, r, 0.0)
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
                        )[0],
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

    def _sample_eta(self, r, n=1):
        """Sample the angle eta which defines radial vs tangential velocities"""
        # cumulative distribution of x = cos eta satisfies
        # x/(sqrt(A+1 -A* x^2)) = 2 b - 1 = c
        # where b \in [0,1] and A = (r/ra)^2
        # Solved by
        # x = c sqrt(1+[r/ra]^2) / sqrt( [r/ra]^2 c^2 + 1 ) for c > 0 [b > 0.5]
        # and symmetric wrt c
        c = numpy.random.uniform(size=n)
        x = (
            c
            * numpy.sqrt(1 + r**2.0 / self._ra2)
            / numpy.sqrt(r**2.0 / self._ra2 * c**2.0 + 1)
        )
        x *= numpy.random.choice([1.0, -1.0], size=n)
        return numpy.arccos(x)

    def _p_v_at_r(self, v, r):
        """p( v*sqrt[1+r^2/ra^2*sin^2eta] | r) used in sampling"""
        if hasattr(self, "_logfQ_interp"):
            return (
                numpy.exp(
                    self._logfQ_interp(
                        -_evaluatePotentials(self._pot, r, 0) - 0.5 * v**2.0
                    )
                )
                * v**2.0
            )
        else:
            return (
                self.fQ(-_evaluatePotentials(self._pot, r, 0) - 0.5 * v**2.0) * v**2.0
            )

    def _sample_v(self, r, eta, n=1):
        """Generate velocity samples"""
        # Use super-class method to obtain v*[1+r^2/ra^2*sin^2eta]
        out = super()._sample_v(r, eta, n=n)
        # Transform to v
        return out / numpy.sqrt(1.0 + r**2.0 / self._ra2 * numpy.sin(eta) ** 2.0)

    def _vmomentdensity(self, r, n, m):
        if m % 2 == 1 or n % 2 == 1:
            return 0.0
        return (
            2.0
            * numpy.pi
            * integrate.quad(
                lambda v: v ** (2.0 + m + n)
                * self.fQ(-_evaluatePotentials(self._pot, r, 0) - 0.5 * v**2.0),
                0.0,
                self._vmax_at_r(self._pot, r),
            )[0]
            * special.gamma(m / 2.0 + 1.0)
            * special.gamma((n + 1) / 2.0)
            / special.gamma(0.5 * (m + n + 3.0))
            / (1 + r**2.0 / self._ra2) ** (m / 2 + 1)
        )


class osipkovmerrittdf(_osipkovmerrittdf):
    """Class that implements spherical DFs with Osipkov-Merritt-type orbital anisotropy

    .. math::

        \\beta(r) = \\frac{1}{1+r_a^2/r^2}

    with :math:`r_a` the anistropy radius for arbitrary combinations of potential and density profile.
    """

    def __init__(
        self, pot=None, denspot=None, ra=1.4, rmax=1e4, scale=None, ro=None, vo=None
    ):
        """
        Initialize a DF with Osipkov-Merritt anisotropy.

        Parameters
        ----------
        pot : Potential instance or list thereof, optional
            Default: None
        denspot : Potential instance or list thereof that represent the density of the tracers (assumed to be spherical; if None, set equal to pot), optional
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
        - 2021-02-07 - Written - Bovy (UofT)
        """
        _osipkovmerrittdf.__init__(
            self, pot=pot, denspot=denspot, ra=ra, rmax=rmax, scale=scale, ro=ro, vo=vo
        )
        # Because f(Q) is the same integral as the Eddington conversion, but
        # using the augmented density rawdensx(1+r^2/ra^2), we use a helper
        # eddingtondf to do this integral, hacked to use the augmented density
        self._edf = eddingtondf(
            pot=self._pot, denspot=self._denspot, scale=scale, rmax=rmax, ro=ro, vo=vo
        )
        self._edf._dnudr = (
            (
                lambda r: self._denspot._ddensdr(r) * (1.0 + r**2.0 / self._ra2)
                + 2.0 * self._denspot.dens(r, 0, use_physical=False) * r / self._ra2
            )
            if not isinstance(self._denspot, list)
            else (
                lambda r: numpy.sum([p._ddensdr(r) for p in self._denspot])
                * (1.0 + r**2.0 / self._ra2)
                + 2.0
                * evaluateDensities(self._denspot, r, 0, use_physical=False)
                * r
                / self._ra2
            )
        )
        self._edf._d2nudr2 = (
            (
                lambda r: self._denspot._d2densdr2(r) * (1.0 + r**2.0 / self._ra2)
                + 4.0 * self._denspot._ddensdr(r) * r / self._ra2
                + 2.0 * self._denspot.dens(r, 0, use_physical=False) / self._ra2
            )
            if not isinstance(self._denspot, list)
            else (
                lambda r: numpy.sum([p._d2densdr2(r) for p in self._denspot])
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

    def sample(self, R=None, z=None, phi=None, n=1, return_orbit=True, rmin=0.0):
        # Slight over-write of superclass method to first build f(Q) interp
        # No docstring so superclass' is used
        if not hasattr(self, "_logfQ_interp"):
            Qs4interp = numpy.hstack(
                (
                    numpy.geomspace(1e-8, 0.5, 101, endpoint=False),
                    sorted(1.0 - numpy.geomspace(1e-8, 0.5, 101)),
                )
            )
            Qs4interp = -(
                Qs4interp * (self._edf._Emin - self._edf._potInf) + self._edf._potInf
            )
            fQ4interp = numpy.log(self.fQ(Qs4interp))
            iindx = numpy.isfinite(fQ4interp)
            self._logfQ_interp = interpolate.InterpolatedUnivariateSpline(
                Qs4interp[iindx], fQ4interp[iindx], k=3, ext=3
            )
        return sphericaldf.sample(
            self, R=R, z=z, phi=phi, n=n, return_orbit=return_orbit, rmin=rmin
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
