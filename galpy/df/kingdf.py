# Class that represents a King DF
import numpy
from scipy import integrate, interpolate, special

from ..util import conversion
from .df import df
from .sphericaldf import isotropicsphericaldf

_FOURPI = 4.0 * numpy.pi
_TWOOVERSQRTPI = 2.0 / numpy.sqrt(numpy.pi)


class kingdf(isotropicsphericaldf):
    """Class that represents a King DF:

    .. math::

      f(\\mathcal{E}) = \\begin{cases} \\rho_1\\,(2\\pi\\sigma^2)^{-3/2}\\,\\left(e^{\\mathcal{E}/\\sigma^2}-1\\right), & \\mathcal{E} > 0\\\\0, & \\mathcal{E} \\leq 0\\end{cases}

    where :math:`\\mathcal{E}` is the binding energy. See also :ref:`King potential <king_potential_api>`.

    """

    def __init__(self, W0, M=1.0, rt=1.0, npt=1001, ro=None, vo=None):
        """
        Initialize a King DF

        Parameters
        ----------
        W0 : float
            Dimensionless central potential :math:`W_0 = \\Psi(0)/\\sigma^2` (in practice, needs to be :math:`\\lesssim 200`, where the DF is essentially isothermal).
        M : float or Quantity, optional
            Total mass.
        rt : float or Quantity, optional
            Tidal radius.
        npt : int, optional
            Number of points to use to solve for :math:`\\Psi(r)`.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2020-07-09 - Written - Bovy (UofT)
        """
        # Just run df init to set up unit-conversion parameters
        df.__init__(self, ro=ro, vo=vo)
        self.W0 = W0
        self.M = conversion.parse_mass(M, ro=self._ro, vo=self._vo)
        self.rt = conversion.parse_length(rt, ro=self._ro)
        # Solve (mass,rtidal)-scale-free model, which is the basis for
        # the full solution
        self._scalefree_kdf = _scalefreekingdf(self.W0)
        self._scalefree_kdf.solve(npt)
        # Set up scaling factors
        self._radius_scale = self.rt / self._scalefree_kdf.rt
        self._mass_scale = self.M / self._scalefree_kdf.mass
        self._velocity_scale = numpy.sqrt(self._mass_scale / self._radius_scale)
        self._density_scale = self._mass_scale / self._radius_scale**3.0
        # Store central density, r0...
        self.rho0 = self._scalefree_kdf.rho0 * self._density_scale
        self.r0 = self._scalefree_kdf.r0 * self._radius_scale
        self.c = self._scalefree_kdf.c  # invariant
        self.sigma = self._velocity_scale
        self._sigma2 = self.sigma**2.0
        self.rho1 = self._density_scale
        # Setup the potential, use original params in case they had units
        # because then the initialization will turn on units for this object
        from ..potential import KingPotential

        pot = KingPotential(
            W0=self.W0, M=M, rt=rt, _sfkdf=self._scalefree_kdf, ro=ro, vo=vo
        )
        # Now initialize the isotropic DF
        isotropicsphericaldf.__init__(
            self, pot=pot, scale=self.r0, rmax=self.rt, ro=ro, vo=vo
        )
        self._potInf = self._pot(self.rt, 0.0, use_physical=False)
        # Setup inverse cumulative mass function for radius sampling
        self._icmf = interpolate.InterpolatedUnivariateSpline(
            self._mass_scale * self._scalefree_kdf._cumul_mass / self.M,
            self._radius_scale * self._scalefree_kdf._r,
            k=3,
        )
        # Setup velocity DF interpolator for velocity sampling here
        self._rmin_sampling = 0.0
        self._v_vesc_pvr_interpolator = self._make_pvr_interpolator(
            r_a_end=numpy.log10(self.rt / self._scale)
        )

    def dens(self, r):
        return self._scalefree_kdf.dens(r / self._radius_scale) * self._density_scale

    def fE(self, E):
        out = numpy.zeros(numpy.atleast_1d(E).shape)
        varE = self._potInf - E
        if numpy.sum(varE > 0.0) > 0:
            out[varE > 0.0] = (
                (numpy.exp(varE[varE > 0.0] / self._sigma2) - 1.0)
                * (2.0 * numpy.pi * self._sigma2) ** -1.5
                * self.rho1
            )
        return out.reshape(E.shape)  # mass density, not /self.M as for number density


class _scalefreekingdf:
    """Internal helper class to solve the scale-free King DF model, that is, the one that only depends on W = Psi/sigma^2"""

    def __init__(self, W0):
        self.W0 = W0

    def solve(self, npt=1001):
        """Solve the model W(r) at npt points (note: not equally spaced in
        either r or W, because combination of two ODEs for different r ranges)"""
        # Set up arrays for outputs
        r = numpy.zeros(npt)
        W = numpy.zeros(npt)
        dWdr = numpy.zeros(npt)
        # Initialize (r[0]=0 already)
        W[0] = self.W0
        # Determine central density and r0
        self.rho0 = self._dens_W(self.W0)
        self.r0 = numpy.sqrt(9.0 / 4.0 / numpy.pi / self.rho0)
        # First solve Poisson equation ODE from r=0 to r0 using form
        # d^2 Psi / dr^2 =  ... (d psi / dr = v, r^2 dv / dr = RHS-2*r*v)
        if self.W0 < 2.0:
            rbreak = self.r0 / 100.0
        else:
            rbreak = self.r0
        # Using linspace focuses on what happens ~rbreak rather than on <<rbreak
        # which is what you want, because W ~ constant at r <~ r0
        r[: npt // 2] = numpy.linspace(0.0, rbreak, npt // 2)
        sol = integrate.solve_ivp(
            lambda t, y: [
                y[1],
                -_FOURPI * self._dens_W(y[0]) - (2.0 * y[1] / t if t > 0.0 else 0.0),
            ],
            [0.0, rbreak],
            [self.W0, 0.0],
            method="LSODA",
            t_eval=r[: npt // 2],
        )
        W[: npt // 2] = sol.y[0]
        dWdr[: npt // 2] = sol.y[1]
        # Then solve Poisson equation ODE from Psi(r0) to Psi=0 using form
        # d^2 r / d Psi^2 = ... (d r / d psi = 1/v, dv / dpsi = 1/v(RHS-2*r*v))
        # Added advantage that this becomes ~log-spaced in r, which is what
        # you want
        W[npt // 2 - 1 :] = numpy.linspace(sol.y[0, -1], 0.0, npt - npt // 2 + 1)
        sol = integrate.solve_ivp(
            lambda t, y: [
                1.0 / y[1],
                -1.0 / y[1] * (_FOURPI * self._dens_W(t) + 2.0 * y[1] / y[0]),
            ],
            [sol.y[0, -1], 0.0],
            [rbreak, sol.y[1, -1]],
            method="LSODA",
            t_eval=W[npt // 2 - 1 :],
        )
        r[npt // 2 - 1 :] = sol.y[0]
        dWdr[npt // 2 - 1 :] = sol.y[1]
        # Store solution
        self._r = r
        self._W = W
        self._dWdr = dWdr
        # Also store density at these points, and the tidal radius
        self._rho = self._dens_W(self._W)
        self.rt = r[-1]
        self.c = numpy.log10(self.rt / self.r0)
        # Interpolate solution
        self._W_from_r = interpolate.InterpolatedUnivariateSpline(self._r, self._W, k=3)
        # Compute the cumulative mass and store the total mass
        mass_shells = numpy.array(
            [
                integrate.quad(lambda r: _FOURPI * r**2 * self.dens(r), rlo, rhi)[0]
                for rlo, rhi in zip(r[:-1], r[1:])
            ]
        )
        self._cumul_mass = numpy.hstack(
            (
                integrate.quad(lambda r: _FOURPI * r**2 * self.dens(r), 0.0, r[0])[0],
                numpy.cumsum(mass_shells),
            )
        )
        self.mass = self._cumul_mass[-1]
        return None

    def _dens_W(self, W):
        """Density as a function of W"""
        sqW = numpy.sqrt(W)
        return numpy.exp(W) * special.erf(sqW) - _TWOOVERSQRTPI * sqW * (
            1.0 + 2.0 / 3.0 * W
        )

    def dens(self, r):
        return self._dens_W(self._W_from_r(r))
