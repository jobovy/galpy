###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleVerticalInverse
#
#             Calculate (x,v) coordinates for a one-dimensional potential
#             given actions-angle coordinates
#
###############################################################################
import copy
import warnings

import numpy
from matplotlib import cm, gridspec, pyplot
from matplotlib.ticker import NullFormatter
from numpy.polynomial import chebyshev, polynomial
from scipy import integrate, interpolate, ndimage, optimize

from ..potential import evaluatelinearForces, evaluatelinearPotentials
from ..potential.linearPotential import _evaluatelinearx2derivs
from ..potential.Potential import _check_potential_list_and_deprecate
from ..util import conversion, galpyWarning

if conversion._APY_LOADED:
    from astropy import units

from ..util import plot as plot
from .actionAngleHarmonic import actionAngleHarmonic
from .actionAngleHarmonicInverse import actionAngleHarmonicInverse
from .actionAngleInverse import actionAngleInverse
from .actionAngleVertical import actionAngleVertical

# Nodes/weights for composite 10-point Gauss-Legendre quadrature (used per
# interval of the chi mesh in the exact-point-transformation construction;
# the error per panel is O((pi/nchi)^20), i.e., machine precision)
_GLX, _GLW = numpy.polynomial.legendre.leggauss(10)


def _slope_at_zero(js, ys, dys):
    """Slope at J = 0 of the polynomial with value 0 there and the given
    values and slopes at the (one or two) actions js; ys may be 2D with the
    action along the first axis."""
    ys = numpy.atleast_1d(ys)
    dys = numpy.atleast_1d(dys)
    if len(js) == 1:
        return 2.0 * ys[0] / js[0] - dys[0]
    # quartic c1 J + c2 J^2 + c3 J^3 + c4 J^4 through (y, y') at two actions
    A = numpy.array(
        [
            [js[0], js[0] ** 2.0, js[0] ** 3.0, js[0] ** 4.0],
            [1.0, 2.0 * js[0], 3.0 * js[0] ** 2.0, 4.0 * js[0] ** 3.0],
            [js[1], js[1] ** 2.0, js[1] ** 3.0, js[1] ** 4.0],
            [1.0, 2.0 * js[1], 3.0 * js[1] ** 2.0, 4.0 * js[1] ** 3.0],
        ]
    )
    b = numpy.array([ys[0], dys[0], ys[1], dys[1]])
    return numpy.linalg.solve(A, b.reshape(4, -1))[0].reshape(numpy.shape(ys[0]))


class _linearHermite:
    """A single node with its slope, as the value and derivative of a
    linear function of the action: the one-torus family."""

    def __init__(self, j0, y0, dy0):
        self._j0, self._y0, self._dy0 = j0, numpy.array(y0), numpy.array(dy0)

    def __call__(self, j):
        return self._y0 + self._dy0 * (j - self._j0)

    def derivative(self):
        return _linearHermite(self._j0, self._dy0, 0.0 * self._dy0)


class actionAngleVerticalInverse(actionAngleInverse):
    """Inverse action-angle formalism for one dimensional systems"""

    def __init__(
        self,
        pot=None,
        Es=[0.1, 0.3],
        nta=128,
        setup_interp=False,
        use_pointtransform=False,
        pt_deg=7,
        pt_nxa=301,
        exact_pt_spl_deg=5,
        pt_only=False,
        maxiter=100,
        angle_tol=1e-12,
        bisect=False,
        momentum_matched=True,
        mm_npt=40,
        mm_nta=None,
        **kwargs,
    ):
        """
        Initialize an actionAngleVerticalInverse object

        Parameters
        ----------
        pot : Potential object or a combined potential formed using addition (pot1+pot2+…)
            a linearPotential/verticalPotential or a combined potential formed using addition (pot1+pot2+…)
        Es : numpy.ndarray
            energies of the orbits to map the tori for, will be forcibly sorted (needs to be a dense grid when setting up the object for interpolation with setup_interp=True)
        nta : int
            number of auxiliary angles to sample the torus at when mapping the torus (in the momentum-matched mode this only sets the default of mm_nta)
        setup_interp : bool
            if True, setup interpolation grids that allow any torus within the E range to be accessed through interpolation
        use_pointtransform : bool or str
            if True, use a point transformation to improve the accuracy of the mapping; use "exact" to solve for the point transformation that makes the torus exactly a harmonic-oscillator torus by solving an ordinary differential equation (rather than using a polynomial approximation)
        pt_deg : int
            degree of the point transformation polynomial (not used when use_pointtransform == "exact")
        pt_nxa : int
            number of points to use in the point transformation
        exact_pt_spl_deg : int
            degree of the spline used to represent the exact point transformation (only used when use_pointtransform == "exact")
        pt_only : bool
            if True, evaluate the inverse transformation using the exact point transformation alone, skipping the generating-function mapping, which is the identity for a perfect point transformation (only allowed when use_pointtransform == "exact"); the mapping coefficients are still computed at setup as a diagnostic of the accuracy of the point transformation and a warning is issued if they are not small
        maxiter : int
            maximum number of iterations of root-finding algorithms
        angle_tol : float
            tolerance for angle root-finding (f(x) is within tol of desired value)
        bisect : bool
            if True, use simple bisection for root-finding, otherwise first try Newton-Raphson (mainly useful for testing the bisection fallback)
        momentum_matched : bool
            if True (default), evaluate through the momentum-matched canonical map: the auxiliary torus carries the same action, corresponding points are those that have swept the same cumulative action, and the amplitude is stored as K = xmax^2/J. This is the same construction the spherical and Staeckel inverses use. Set to False to use the older evaluation instead. The canonical map is itself a point transformation, so it is not used when use_pointtransform is set, in which case the older evaluation runs; it works for any number of energies, down to a single torus, and always interpolates between the grid tori; setup_interp=True then only provides E(J) and J(E), from the same Hermite energy interpolant that the frequency derives from.
        mm_npt : int
            number of (even) harmonics of the momentum-matched anomaly map; the reconstruction converges spectrally in this, reaching round-off (~1e-13) at the default for tori within a few scale heights of the midplane (~1e-8 at 20); tori reaching further need more, growing as xmax over the scale height, and a warning is raised when the default does not suffice (only used when momentum_matched is True)
        mm_nta : int, optional
            number of anomaly samples per torus used to fit the momentum-matched anomaly map and to compute the torus's action and frequency; must exceed 4 * mm_npt for the samples to resolve the map's highest harmonic; default is 2 * nta, raised to 8 * mm_npt if that is larger (only used when momentum_matched is True)

        Notes
        -----
        - 2018-04-11 - Started - Bovy (UofT)
        - 2026-08-30 - Added the momentum-matched canonical map - Bovy (UofT)
        """
        actionAngleInverse.__init__(self, **kwargs)
        if pot is None:  # pragma: no cover
            raise OSError("Must specify pot= for actionAngleVerticalInverse")
        self._pot = _check_potential_list_and_deprecate(pot)
        self._aAV = actionAngleVertical(pot=self._pot)
        # Compute action, frequency, and xmax for each energy
        self._Es = numpy.sort(
            conversion._parse_grid_quantity(Es, conversion.parse_energy, vo=self._vo)
        )
        self._nE = len(self._Es)
        js = numpy.empty(self._nE)
        Omegas = numpy.empty(self._nE)
        xmaxs = numpy.empty(self._nE)
        for ii, E in enumerate(self._Es):
            if (
                E - evaluatelinearPotentials(self._pot, 0.0, use_physical=False)
            ) < 1e-14:
                # J=0, should be using vertical freq. from 2nd deriv.
                tJ, tO = self._aAV.actionsFreqs(
                    0.0,
                    numpy.sqrt(
                        2.0
                        * (
                            E
                            + 1e-5
                            - evaluatelinearPotentials(
                                self._pot, 0.0, use_physical=False
                            )
                        )
                    ),
                )
                js[ii] = 0.0
                Omegas[ii] = tO[0]
                xmaxs[ii] = 0.0
                continue
            tJ, tO = self._aAV.actionsFreqs(
                0.0,
                numpy.sqrt(
                    2.0
                    * (E - evaluatelinearPotentials(self._pot, 0.0, use_physical=False))
                ),
            )
            js[ii] = tJ[0]
            Omegas[ii] = tO[0]
            xmaxs[ii] = self._aAV.calcxmax(
                0.0,
                numpy.sqrt(
                    2.0
                    * (E - evaluatelinearPotentials(self._pot, 0.0, use_physical=False))
                ),
                E=E,
            )
        self._js = js
        self._Omegas = Omegas
        self._xmaxs = xmaxs
        # Set harmonic-oscillator frequencies == frequencies
        self._OmegaHO = copy.copy(Omegas)
        # The following work properly for arrays of omega
        self._hoaa = actionAngleHarmonic(omega=self._OmegaHO)
        self._hoaainv = actionAngleHarmonicInverse(omega=self._OmegaHO)
        self._nta = nta
        self._maxiter = maxiter
        self._angle_tol = angle_tol
        self._bisect = bisect
        # The zero-energy torus is the harmonic oscillator at the midplane,
        # whose frequency is sqrt(Phi''(0)), rather than the first torus's
        # frequency, which is off by O(J_1) and would spoil everything
        # interpolated through the bottom of the grid.
        if numpy.any(self._Es < 1e-10):
            omega0 = numpy.sqrt(_evaluatelinearx2derivs(self._pot, 0.0))
            self._OmegaHO[self._Es < 1e-10] = omega0
            self._Omegas[self._Es < 1e-10] = omega0
        # The momentum-matched canonical map replaces the evaluation rather
        # than adding to it: it needs only each torus's turning point and its
        # action and frequency, which its own sweep of the torus gives
        # spectrally, so none of the auxiliary-angle machinery below (the
        # angle grid, the S_n fits, the point transformation) is built for it.
        # The map is itself a point transformation, so asking for one
        # explicitly selects the older evaluation.
        self._momentum_matched = momentum_matched and not use_pointtransform
        if self._momentum_matched:
            if pt_only:
                raise ValueError(
                    'pt_only=True is only supported for use_pointtransform="exact"'
                )
            bad = (self._xmaxs < 0.0) | ~numpy.isfinite(self._js)
            if numpy.any(bad):
                raise RuntimeError(
                    "The turning point could not be found for energies: {}".format(
                        ", ".join(f"{E:g}" for E in self._Es[bad])
                    )
                )
            self._pt_exact = False
            self._pt_only = False
            self._pt_deg = 1
            self._check_consistent_units()
            if mm_nta is None:
                mm_nta = max(2 * nta, 8 * mm_npt)
            elif mm_nta <= 4 * mm_npt:
                # the map's highest harmonic is 2 mm_npt, which mm_nta uniform
                # samples only resolve below their Nyquist harmonic mm_nta / 2
                raise ValueError(
                    "mm_nta must exceed 4 * mm_npt for the anomaly samples to resolve the map's harmonics"
                )
            self._setup_momentum_matched_family(mm_npt=mm_npt, mm_nta=mm_nta)
            self._interp = bool(setup_interp)
            if self._interp:
                # the family interpolates on its own; these only serve
                # J(E) and E(J) at energies between the grid tori, from the
                # same Hermite energy interpolant whose derivative is the
                # frequency, so that E, J, and Freqs agree exactly
                self.E = self._mm_E
                self.J = self._mm_J_of_E
            return None
        if (
            isinstance(use_pointtransform, str)
            and use_pointtransform.lower() == "exact"
        ):
            self._pt_exact = True
            self._exact_pt_spl_deg = exact_pt_spl_deg
            self._pt_only = pt_only
            self._setup_pointtransform(pt_deg, pt_nxa)
        elif use_pointtransform and pt_deg > 1:
            if pt_only:
                raise ValueError(
                    'pt_only=True is only supported for use_pointtransform="exact"'
                )
            self._pt_exact = False
            self._pt_only = False
            self._exact_pt_spl_deg = None
            self._setup_pointtransform(pt_deg - (1 - pt_deg % 2), pt_nxa)  # make odd
        else:
            if pt_only:
                raise ValueError(
                    'pt_only=True is only supported for use_pointtransform="exact"'
                )
            # Setup identity point transformation
            self._pt_exact = False
            self._pt_only = False
            self._exact_pt_spl_deg = None
            self._pt_deg = 1
            self._pt_nxa = pt_nxa
            self._pt_xmaxs = self._xmaxs
            self._pt_coeffs = numpy.zeros((self._nE, 2))
            self._pt_coeffs[:, 1] = 1.0
            self._pt_deriv_coeffs = numpy.ones((self._nE, 1))
            self._pt_deriv2_coeffs = numpy.zeros((self._nE, 1))
        # Extra keyword arguments to pass to _anglea, _ja, ... for the exact
        # point transformation (which the polynomial evaluation doesn't need);
        # for the exact point transformation, the positional ptcoeffs-style
        # arguments of those functions instead carry the (possibly fractional)
        # row index of each point's torus in the grid
        self._pt_eval_kwargs = (
            dict(
                pt_exact=True,
                pt_filtered=self._pt_filtered,
                pt_nmesh=self._pt_nmesh,
                pt_spl_deg=self._exact_pt_spl_deg,
            )
            if self._pt_exact
            else dict()
        )
        # Now map all tori
        self._thetaa = numpy.linspace(0.0, 2.0 * numpy.pi * (1.0 - 1.0 / nta), nta)
        self._xgrid = self._create_xgrid()
        self._ja = _ja(
            self._xgrid,
            self._Egrid,
            self._pot,
            self._omegagrid,
            self._ptcoeffsgrid,
            self._ptderivcoeffsgrid,
            self._xmaxgrid,
            self._ptxmaxgrid,
            **self._pt_eval_kwargs,
        )
        self._djadj = (
            _djadj(
                self._xgrid,
                self._Egrid,
                self._pot,
                self._omegagrid,
                self._ptcoeffsgrid,
                self._ptderivcoeffsgrid,
                self._ptderiv2coeffsgrid,
                self._xmaxgrid,
                self._ptxmaxgrid,
                **self._pt_eval_kwargs,
            )
            * numpy.atleast_2d(self._Omegas / self._OmegaHO).T
        )  # In case not 1!
        self._djadj[self._js < 1e-10] = 1.0  # J = 0 special case
        # The mean of the auxiliary action over the auxiliary-angle grid is
        # the action of the POINT-TRANSFORMED torus, J^A = (1/2pi) times the
        # loop integral of v^A dx^A (the harmonic action-angle map is
        # canonical, so the mean over a regular grid in theta^A is that
        # integral, to spectral accuracy).  It is the base the Fourier structure below is built
        # on, and it is the torus's actual action only when the
        # transformation conserves the action: the identity and the exact
        # one do, a polynomial one in the v^A = v / pi' gauge used here does
        # not (4e-4 relative at degree 7, growing with energy).  The
        # difference has a closed form, J^A - J = (1/2pi) Int v (pi'^-2 - 1)
        # dx around the torus, computed here by quadrature from the potential and the fitted
        # transformation, so that _js is the actual action in every mode
        # (without a transformation the mean is the more accurate estimate
        # of it) and the internal base is _js + _jaoffset everywhere below.
        self._jaoffset = self._pt_action_offset(use_pointtransform)
        self._js = numpy.nanmean(self._ja, axis=1) - self._jaoffset
        # Store better approximation to Omega
        self._Omegas_orig = copy.copy(self._Omegas)
        self._Omegas /= numpy.nanmean(self._djadj, axis=1)
        # Compute Fourier expansions
        self._nforSn = numpy.arange(self._ja.shape[1] // 2 + 1)
        self._nSn = (
            numpy.real(
                numpy.fft.rfft(
                    self._ja - numpy.atleast_2d(numpy.nanmean(self._ja, axis=1)).T,
                    axis=1,
                )
            )[:, 1:]
            / self._ja.shape[1]
        )
        self._dSndJ = (
            numpy.real(
                numpy.fft.rfft(
                    self._djadj / numpy.atleast_2d(numpy.nanmean(self._djadj, axis=1)).T
                    - 1.0,
                    axis=1,
                )
            )[:, 1:]
            / self._ja.shape[1]
        )
        # Interpolation of small, noisy coeffs doesn't work, so set to zero
        if setup_interp:
            self._nSn[numpy.fabs(self._nSn) < 1e-16] = 0.0
            self._dSndJ[numpy.fabs(self._dSndJ) < 1e-15] = 0.0
        self._dSndJ /= numpy.atleast_2d(self._nforSn)[:, 1:]
        self._nforSn = self._nforSn[1:]
        self._js[self._Es < 1e-10] = 0.0
        self._nSn[self._js < 1e-10] = 0.0
        self._dSndJ[self._js < 1e-10] = 0.0
        # When evaluating using the point transformation only, the computed
        # mapping coefficients serve as a diagnostic of the accuracy of the
        # point transformation: they should all be close to zero
        if self._pt_exact and self._pt_only:
            relnSn = numpy.nanmax(numpy.fabs(self._nSn)) / numpy.nanmax(
                numpy.fabs(self._js) + 1e-15
            )
            maxdSndJ = numpy.nanmax(numpy.fabs(self._dSndJ))
            if relnSn > 1e-8 or maxdSndJ > 1e-8:
                warnings.warn(
                    "Point transformation is not accurate enough for pt_only=True evaluation: the generating-function mapping that pt_only skips is not negligible (max |nSn|/max J = {:.2e}, max |dSndJ| = {:.2e}); increase pt_nxa or use pt_only=False".format(
                        relnSn, maxdSndJ
                    ),
                    galpyWarning,
                )
        # Check the units
        self._check_consistent_units()
        # Setup interpolation if requested
        if setup_interp:
            self._interp = True
            self._setup_interp()
        else:
            self._interp = False
        return None

    def _pt_action_offset(self, use_pointtransform):
        """
        The offset between the point-transformed action J^A = <j^A> and the
        torus's actual action J, for every torus of the grid, in closed
        form.

        Because (x^A, v^A) -> (j^A, theta^A) is the harmonic action-angle
        map, 2 pi <j^A> is the loop integral of v^A dx^A = v pi'^-2 dx, so

            J^A - J = (1 / 2 pi) Int v(x) [pi'(x^A)^-2 - 1] dx ,

        a quadrature along the torus from the potential and the fitted
        transformation alone.  It vanishes for the identity and for the
        exact point transformation (v / pi' is then exactly the harmonic
        velocity), so it is only computed for a polynomial one.

        Parameters
        ----------
        use_pointtransform : bool or str
            The constructor's use_pointtransform.

        Returns
        -------
        numpy.ndarray
            J^A - J for each torus (zeros without a polynomial point
            transformation).

        Notes
        -----
        - 2026-09-18 - Written - Bovy (UofT)
        """
        offset = numpy.zeros(self._nE)
        if not use_pointtransform or self._pt_exact:
            return offset
        for ii in range(self._nE):
            if self._js[ii] <= 0.0:
                continue
            E, xmax, ptxmax = self._Es[ii], self._xmaxs[ii], self._pt_xmaxs[ii]
            c, dc = self._pt_coeffs[ii], self._pt_deriv_coeffs[ii]

            def integrand(u):
                # u = x^A / ptxmax on [-1, 1]; x = pi(x^A); dx = pi' dx^A
                x = polynomial.polyval(u, c) * xmax
                piprime = polynomial.polyval(u, dc) * xmax / ptxmax
                v = numpy.sqrt(
                    max(
                        2.0
                        * (
                            E
                            - evaluatelinearPotentials(self._pot, x, use_physical=False)
                        ),
                        0.0,
                    )
                )
                return v * (piprime**-2.0 - 1.0) * polynomial.polyval(u, dc) * xmax

            offset[ii] = (
                integrate.quad(
                    integrand, -1.0, 1.0, limit=200, epsabs=1e-13, epsrel=1e-12
                )[0]
                / numpy.pi
            )
        return offset

    def _setup_pointtransform(self, pt_deg, pt_nxa):
        # Setup a point transformation for each torus
        self._pt_deg = pt_deg
        self._pt_nxa = pt_nxa
        self._pt_xmaxs = numpy.sqrt(2.0 * self._js / self._OmegaHO)
        if self._pt_exact:
            return self._setup_pointtransform_exact(pt_nxa)
        xamesh = numpy.linspace(-1.0, 1.0, pt_nxa)
        self._pt_coeffs = numpy.empty((self._nE, pt_deg + 1))
        self._pt_deriv_coeffs = numpy.empty((self._nE, pt_deg))
        self._pt_deriv2_coeffs = numpy.empty((self._nE, pt_deg - 1))
        for ii in range(self._nE):
            if self._js[ii] < 1e-10:  # Just use identity for small J
                self._pt_coeffs[ii] = 0.0
                self._pt_coeffs[ii, 1] = 1.0
                self._pt_deriv_coeffs[ii] = 1.0
                self._pt_deriv2_coeffs[ii] = 0.0
                self._pt_xmaxs[ii] = self._xmaxs[ii] + 1e-10  # avoid /0
                coeffs = self._pt_coeffs[ii]  # to start next fit
                continue
            Ea = self._js[ii] * self._OmegaHO[ii]

            # Function to optimize with least squares: p-p
            def opt_func(coeffs):
                # constraints: symmetric, maps [-1,1] --> [-1,1]
                ccoeffs = numpy.zeros(pt_deg + 1)
                ccoeffs[1] = 1.0
                ccoeffs[3::2] = coeffs
                ccoeffs /= chebyshev.chebval(1, ccoeffs)
                pt = chebyshev.Chebyshev(ccoeffs)
                xmesh = pt(xamesh) * self._xmaxs[ii]
                # Compute v from (E,xmesh)
                v2mesh = 2.0 * (
                    self._Es[ii]
                    - evaluatelinearPotentials(self._pot, xmesh, use_physical=False)
                )
                v2mesh[v2mesh < 0.0] = 0.0
                vmesh = numpy.sqrt(v2mesh)
                # Compute v from va = 2(E-HO) and transform
                va2mesh = 2.0 * (
                    Ea
                    - self._OmegaHO[ii] ** 2.0
                    * (xamesh * self._pt_xmaxs[ii]) ** 2.0
                    / 2.0
                )
                va2mesh[va2mesh < 0.0] = 0.0
                vamesh = numpy.sqrt(va2mesh)
                piprime = pt.deriv()(xamesh) * self._xmaxs[ii] / self._pt_xmaxs[ii]
                vtildemesh = (
                    vamesh - numpy.sqrt(v2mesh) * (1.0 / piprime - piprime)
                ) / piprime
                return vmesh - vtildemesh

            if ii == 0:
                # Start from identity mapping
                start_coeffs = [0.0]
                start_coeffs.extend([0.0 for jj in range((pt_deg + 1) // 2 - 2)])
            else:
                # Start from previous best fit
                start_coeffs = coeffs[3::2] / coeffs[1]
            coeffs = optimize.leastsq(opt_func, start_coeffs)[0]
            # Extract full Chebyshev parameters from constrained optimization
            ccoeffs = numpy.zeros(pt_deg + 1)
            ccoeffs[1] = 1.0
            ccoeffs[3::2] = coeffs
            ccoeffs /= chebyshev.chebval(1, ccoeffs)  # map exact [-1,1] --> [-1,1]
            coeffs = ccoeffs
            # Store point transformation as simple polynomial
            self._pt_coeffs[ii] = chebyshev.cheb2poly(coeffs)
            self._pt_deriv_coeffs[ii] = polynomial.polyder(self._pt_coeffs[ii], m=1)
            self._pt_deriv2_coeffs[ii] = polynomial.polyder(self._pt_coeffs[ii], m=2)
        return None

    def _mm_sweep(self, E, xmax, mm_nta):
        """
        Sample a torus in its anomaly: positions, signed momenta, action
        flux, action, cumulative action, and frequency.

        On the uniform periodic grid tau_k = 2 pi k / mm_nta, x = -xmax cos
        tau and p = sgn(sin tau) sqrt(2 [E - Phi(x)]), the sign making p the
        signed momentum along the libration (with |p| the flux would be
        antisymmetric about tau = pi and its mean, the action, would
        vanish).  The action is the mean of the flux g = p dx/dtau, the loop
        integral by the trapezoid rule on a periodic grid; the cumulative
        action A(tau) is its spectral antiderivative (exact for the
        band-limited integrand, where a cumulative trapezoid is only second
        order and would floor the map's fit); and the frequency is 2 pi over
        the period, the loop integral of dx/p, whose integrand xmax sin(tau)
        / p is regular at the turning points with the limit
        sqrt(xmax / |Phi'(xmax)|), which is what the samples that fall
        exactly on them are set to.

        Parameters
        ----------
        E, xmax : float
            Energy and turning point of the torus.
        mm_nta : int
            Number of anomaly samples.

        Returns
        -------
        tuple
            (tau, x, p, g, J, A, Omega).

        Notes
        -----
        - 2026-09-18 - Written - Bovy (UofT)
        """
        tau = 2.0 * numpy.pi * numpy.arange(mm_nta) / mm_nta
        x = -xmax * numpy.cos(tau)
        sintau = numpy.sin(tau)
        p = numpy.sign(sintau) * numpy.sqrt(
            numpy.clip(
                2.0 * (E - evaluatelinearPotentials(self._pot, x, use_physical=False)),
                0.0,
                None,
            )
        )
        g = p * xmax * sintau
        J = numpy.mean(g)
        k = numpy.fft.fftfreq(mm_nta, d=1.0 / mm_nta)
        gh = numpy.fft.fft(g - J)
        ah = numpy.zeros_like(gh)
        ah[1:] = gh[1:] / (1j * k[1:])
        A = numpy.real(numpy.fft.ifft(ah))
        A = A - A[0] + J * tau
        # the period: dx / p is regular at the turning points; the samples
        # sitting on them (tau = 0 and tau = pi, both on the grid) take the
        # limit, which the floating-point ratio does not reproduce there
        turn = numpy.fabs(sintau) < 1e-8
        r = numpy.empty_like(tau)
        r[~turn] = xmax * sintau[~turn] / p[~turn]
        r[turn] = numpy.sqrt(
            xmax / numpy.fabs(evaluatelinearForces(self._pot, xmax, use_physical=False))
        )
        Om = 1.0 / numpy.mean(r)
        return tau, x, p, g, J, A, Om

    def _momentum_matched_map(self, ii, mm_npt=16, mm_nta=1024, D0=None):
        """
        Momentum-matched anomaly map of torus ii.

        The auxiliary torus is the harmonic oscillator with the SAME action,
        and corresponding points are those that have swept the same action:
        A^A(eta) = A(tau).  Because the two actions agree, eta - tau is
        periodic, and because the momentum is antisymmetric about the turning
        points it is odd under tau -> 2 pi - tau, so the map is a pure sine
        series.  For a potential symmetric about the midplane only the even
        harmonics survive.

        The auxiliary's cumulative action is A^A(eta) = J (eta - sin eta
        cos eta): the harmonic frequency cancels between omega and the
        squared amplitude 2J/omega, so the map does not depend on which
        harmonic auxiliary is used, only on its action.

        The coefficients are obtained by FITTING them to the matching
        condition rather than by inverting A^A pointwise.  That inverse is
        ill-conditioned at both turning points, where dA^A/deta vanishes like
        sin^2, and a pointwise construction converges only as 1/nta; fitting
        leaves the vanishing derivative as a factor rather than a divisor.

        Parameters
        ----------
        ii : int
            Index of the torus.
        mm_npt : int, optional
            Number of (even) harmonics to fit.
        mm_nta : int, optional
            Number of anomaly samples.
        D0 : numpy.ndarray, optional
            Starting coefficients for the fit (e.g., those of the previous
            torus of the grid); zero if not given.

        Returns
        -------
        tuple
            (D, K, J, Omega, perr) with D the sine coefficients of the even
            harmonics, K = xmax^2 / J the storage variable, which stays
            finite in the harmonic limit where xmax itself vanishes, the
            torus's action and frequency from the sweep, and the maximum
            relative error of the momentum that the truncated map
            reconstructs on the sweep.

        Notes
        -----
        - 2026-08-29 - Written - Bovy (UofT)
        """
        E, xmax = self._Es[ii], self._xmaxs[ii]
        tau, x, p, g, J, A, Om = self._mm_sweep(E, xmax, mm_nta)
        ms = 2 * numpy.arange(1, mm_npt + 1)
        S = numpy.sin(tau[:, None] * ms[None, :])

        def _resid(D):
            eta = tau + S @ D
            return J * (eta - numpy.sin(eta) * numpy.cos(eta)) - A

        def _jac(D):
            # d resid / d D_m = J (1 - cos 2 eta) sin(m tau) = 2 J sin^2(eta) sin(m tau)
            eta = tau + S @ D
            return (2.0 * J * numpy.sin(eta) ** 2.0)[:, None] * S

        sol = optimize.least_squares(
            _resid,
            numpy.zeros(len(ms)) if D0 is None else numpy.array(D0, dtype="float"),
            jac=_jac,
            xtol=1e-15,
            ftol=1e-15,
            gtol=1e-15,
        )
        # how well the truncated map reconstructs the momentum on the sweep,
        # which is where a too-short series shows first (the fit's own
        # residual is the cumulative action, one derivative smoother)
        eta = tau + S @ sol.x
        deta = 1.0 + numpy.cos(tau[:, None] * ms[None, :]) @ (ms * sol.x)
        nz = numpy.sin(tau) != 0.0
        pmap = (
            2.0 * J * numpy.sin(eta[nz]) ** 2.0 * deta[nz] / (xmax * numpy.sin(tau[nz]))
        )
        perr = numpy.amax(numpy.fabs(pmap - p[nz])) / numpy.amax(numpy.fabs(p))
        return sol.x, xmax**2.0 / J, J, Om, perr

    def _setup_momentum_matched_family(self, mm_npt=16, mm_nta=1024):
        """
        Build the momentum-matched family: the anomaly map and the storage
        variable K on every torus of the energy grid.

        This is the whole stored content of the canonical map in the new
        scheme.  There is no table of angle-fit coefficients: the auxiliary
        torus carries the target's content through the anomaly map alone,
        and the amplitude enters only through K = xmax^2 / J.

        K is stored rather than xmax because xmax vanishes with the action
        while K does not: in the harmonic limit xmax^2 -> 2 J / omega, so
        K -> 2 / omega, which is finite and O(1).  Storing xmax instead would
        put a square-root cusp at the bottom of the grid and spend the
        interpolant's resolution resolving it.

        Parameters
        ----------
        mm_npt : int, optional
            Number of (even) harmonics of the anomaly map.
        mm_nta : int, optional
            Number of anomaly samples used to fit them.

        Notes
        -----
        - 2026-08-29 - Written - Bovy (UofT)
        """
        D = numpy.zeros((self._nE, mm_npt))
        K = numpy.empty(self._nE)
        perr = numpy.zeros(self._nE)
        for ii in range(self._nE):
            if self._js[ii] <= 0.0:
                # The bottom of the grid IS a harmonic oscillator, so the
                # auxiliary torus is the torus, the anomaly map is the
                # identity (D = 0), and K takes its limit 2 / omega.
                K[ii] = 2.0 / self._Omegas[ii]
                continue
            # warm start from the previous torus: the coefficients vary
            # smoothly along the grid.  The torus's action and frequency
            # come from the same sweep, spectrally, and replace the forward
            # transformation's fixed-order quadratures.
            D[ii], K[ii], self._js[ii], self._Omegas[ii], perr[ii] = (
                self._momentum_matched_map(
                    ii,
                    mm_npt=mm_npt,
                    mm_nta=mm_nta,
                    D0=D[ii - 1] if ii > 0 and self._js[ii - 1] > 0.0 else None,
                )
            )
        self._mm_D = D
        self._mm_K = K
        self._mm_npt = mm_npt
        self._mm_nta = mm_nta
        if numpy.any(perr > 1e-6):
            warnings.warn(
                "The momentum-matched anomaly map is not converged for energies: {} (maximum relative error of the reconstructed momentum {:.1e}); increase mm_npt and, with it, mm_nta".format(
                    ", ".join(f"{E:g}" for E in self._Es[perr > 1e-6]),
                    numpy.amax(perr),
                ),
                galpyWarning,
            )
        # The action derivatives dD_m/dJ and dK/dJ that the angle needs are
        # known on every torus from that torus alone (the variation of the
        # matching condition, _momentum_matched_slopes), so they go INTO
        # the interpolants as Hermite constraints, the way E(J) below takes
        # the frequency: the evaluation still differentiates the same
        # interpolant it reads, which is what keeps the map symplectic
        # whatever the tables contain, but the derivative is now exact at
        # the nodes and one order better between them, and a family can be
        # as small as a single torus.  (Storing the slopes as separate
        # tables instead would be exactly what breaks manifest canonicity.)
        dD = numpy.zeros((self._nE, mm_npt))
        dK = numpy.zeros(self._nE)
        for ii in range(self._nE):
            if self._js[ii] > 0.0:
                dD[ii], dK[ii] = self._momentum_matched_slopes(
                    self._Es[ii],
                    self._xmaxs[ii],
                    self._js[ii],
                    self._Omegas[ii],
                    D[ii],
                    mm_nta=mm_nta,
                )
        # At the bottom of the grid the values are exact (D = 0, K = 2 /
        # omega) but their slopes are anharmonic quantities the harmonic limit
        # does not give, and the fit above degenerates as J -> 0.  They are
        # the slope at J = 0 of the polynomial through the exact bottom value
        # and the value AND slope of the next two tori (a quartic; with a
        # single torus above the bottom, the quadratic through it), which is
        # fourth (second) order in the spacing: for D_2 it reproduces the
        # perturbative value, the fourth derivative of Phi at 0 over
        # 96 omega^3, to ~1e-3 on a nine-node grid.
        pos = numpy.where(self._js > 0.0)[0]
        for ii in numpy.where(self._js <= 0.0)[0]:
            if len(pos) == 0:
                break
            dD[ii] = _slope_at_zero(self._js[pos[:2]], D[pos[:2]] - D[ii], dD[pos[:2]])
            dK[ii] = _slope_at_zero(self._js[pos[:2]], K[pos[:2]] - K[ii], dK[pos[:2]])
        self._mm_dD = dD
        self._mm_dK = dK
        if self._nE > 1:
            self._mm_Dspl = interpolate.CubicHermiteSpline(self._js, D, dD, axis=0)
            self._mm_Kspl = interpolate.CubicHermiteSpline(self._js, K, dK)
            # E(J) is interpolated as a Hermite spline, matching the energies
            # at the nodes AND their slopes, because those slopes are already
            # known exactly: dE/dJ is the frequency.  Fitting E alone and
            # differentiating would throw that information away and leave the
            # map's frequency disagreeing with the tabulated one.
            self._mm_E = interpolate.CubicHermiteSpline(
                self._js, self._Es, self._Omegas
            )
        else:
            # a single torus: the family is one node with its slopes
            self._mm_Dspl = _linearHermite(self._js[0], D[0], dD[0])
            self._mm_Kspl = _linearHermite(self._js[0], K[0], dK[0])
            self._mm_E = _linearHermite(self._js[0], self._Es[0], self._Omegas[0])
        self._mm_dDspl = self._mm_Dspl.derivative()
        self._mm_dKspl = self._mm_Kspl.derivative()
        self._mm_dEdj = self._mm_E.derivative()
        return None

    def _momentum_matched_slopes(self, E, xmax, J, Om, D, mm_nta=1024):
        """
        The action derivatives of the anomaly map and of the storage
        variable on a single torus, from that torus alone.

        Differentiating the matching condition A^A(eta(tau; J); J) =
        A(tau; J) with respect to J at fixed anomaly gives

            (eta - sin eta cos eta) + 2 J sin^2(eta) sum_m dD_m/dJ sin(m tau)
                = dA/dJ |_tau ,

        where the right-hand side varies through dE/dJ = Omega and through
        the moving turning point, dxmax/dJ = Omega / Phi'(xmax): its
        integrand, the J-derivative of the flux p dx/dtau, is regular at the
        turning points (numerator ~ tau^2 against p ~ tau) and is integrated
        spectrally like the flux itself.  The coefficients dD_m/dJ then
        follow from a LINEAR least-squares fit with the vanishing factor
        2 J sin^2(eta) multiplying the unknowns, as in the map's own fit.

        Parameters
        ----------
        E, xmax, J, Om : float
            Energy, turning point, action, and frequency of the torus.
        D : numpy.ndarray
            The torus's anomaly-map coefficients.
        mm_nta : int, optional
            Number of anomaly samples.

        Returns
        -------
        tuple
            (dD/dJ, dK/dJ) with K = xmax^2 / J.

        Notes
        -----
        - 2026-09-18 - Written - Bovy (UofT)
        """
        ms = 2.0 * numpy.arange(1, len(D) + 1)
        tau = 2.0 * numpy.pi * numpy.arange(mm_nta) / mm_nta
        x = -xmax * numpy.cos(tau)
        p = numpy.sign(numpy.sin(tau)) * numpy.sqrt(
            numpy.clip(
                2.0 * (E - evaluatelinearPotentials(self._pot, x, use_physical=False)),
                0.0,
                None,
            )
        )
        dPhi = -evaluatelinearForces(self._pot, x, use_physical=False)
        dxmaxdJ = Om / (-evaluatelinearForces(self._pot, xmax, use_physical=False))
        dxdJ = -dxmaxdJ * numpy.cos(tau)
        dpdJ = numpy.zeros_like(tau)
        nz = p != 0.0
        dpdJ[nz] = (Om - dPhi[nz] * dxdJ[nz]) / p[nz]
        gJ = (dpdJ * xmax + p * dxmaxdJ) * numpy.sin(tau)
        k = numpy.fft.fftfreq(mm_nta, d=1.0 / mm_nta)
        gh = numpy.fft.fft(gJ - numpy.mean(gJ))
        ah = numpy.zeros_like(gh)
        ah[1:] = gh[1:] / (1j * k[1:])
        dAdJ = numpy.real(numpy.fft.ifft(ah))
        dAdJ = dAdJ - dAdJ[0] + numpy.mean(gJ) * tau
        S = numpy.sin(tau[:, None] * ms[None, :])
        eta = tau + S @ D
        rhs = dAdJ - (eta - numpy.sin(eta) * numpy.cos(eta))
        dDdJ = numpy.linalg.lstsq(
            (2.0 * J * numpy.sin(eta) ** 2.0)[:, None] * S, rhs, rcond=None
        )[0]
        dKdJ = 2.0 * xmax * dxmaxdJ / J - xmax**2.0 / J**2.0
        return dDdJ, dKdJ

    def _mm_tables(self, j):
        """
        The anomaly map, the storage variable, and their exact action
        derivatives at action j.

        Both derivatives come from differentiating the stored interpolants;
        nothing is finite-differenced and no derivative is stored separately,
        which is what makes the resulting map symplectic whatever the tables
        happen to contain.  Outside the grid the splines extrapolate.

        Parameters
        ----------
        j : float
            Action.

        Returns
        -------
        tuple
            (D, dD/dj, K, dK/dj).

        Notes
        -----
        - 2026-08-29 - Written - Bovy (UofT)
        """
        return (
            self._mm_Dspl(j),
            self._mm_dDspl(j),
            float(self._mm_Kspl(j)),
            float(self._mm_dKspl(j)),
        )

    def _mm_xp_of_tau(self, j, tau):
        """
        Position and momentum at anomaly tau on the torus of action j, from
        the stored family alone.

        The auxiliary is the harmonic oscillator of the same action, so
        x^A = -sqrt(2 J / omega) cos eta and p^A = sqrt(2 J omega) sin eta,
        and the flux identity p dx/dtau = p^A (dx^A/deta)(deta/dtau) gives

            p = 2 J sin^2(eta) eta'(tau) / (xmax sin tau) ,

        with xmax = sqrt(K J).  The auxiliary frequency cancels between the
        momentum and the amplitude, which is the same cancellation that
        makes the anomaly map itself independent of which harmonic auxiliary
        is used: only its action matters.

        Both factors vanish at the turning points, where sin tau and
        sin^2 eta go to zero together and the momentum is zero; the ratio is
        taken only where sin tau does not vanish exactly, and the limit is
        supplied directly.

        Parameters
        ----------
        j : float
            Action.
        tau : float or numpy.ndarray
            Anomaly.

        Returns
        -------
        tuple
            (x, p) at the requested anomalies.

        Notes
        -----
        - 2026-08-29 - Written - Bovy (UofT)
        """
        tau = numpy.atleast_1d(numpy.array(tau, dtype="float"))
        if j <= 0.0:
            # the zero-action torus is the point at the bottom
            return numpy.zeros_like(tau), numpy.zeros_like(tau)
        D, _, K, _ = self._mm_tables(j)
        ms = 2.0 * numpy.arange(1, len(D) + 1)
        tau = numpy.atleast_1d(numpy.array(tau, dtype="float"))
        eta = tau + numpy.sin(tau[:, None] * ms[None, :]) @ D
        detadtau = 1.0 + numpy.cos(tau[:, None] * ms[None, :]) @ (ms * D)
        xmax = numpy.sqrt(K * j)
        x = -xmax * numpy.cos(tau)
        sintau = numpy.sin(tau)
        p = numpy.zeros_like(tau)
        nz = sintau != 0.0
        p[nz] = 2.0 * j * numpy.sin(eta[nz]) ** 2.0 * detadtau[nz] / (xmax * sintau[nz])
        return x, p

    def _mm_compensation(self, j, tau):
        """
        The compensation integrand of the momentum-matched map, grouped so
        that it is regular at the turning points.

        The turning points move with the action, so at fixed position

            d tau / d J |_x = (1 / xmax) (d xmax / d J) cos(tau) / sin(tau) ,

        which diverges at both of them.  It is multiplied by
        p^A (dx^A/deta)(deta/dtau) = 2 J sin^2(eta) eta'(tau), which vanishes
        there, and the product is finite: the sin(tau) cancels against the
        momentum and leaves

            p (d xmax / d J) cos(tau) .

        Computing the two factors separately returns nan at an anomaly
        sitting exactly on a turning point, one being infinite and the other
        zero; this grouped form is finite everywhere.  The amplitude
        derivative comes from the stored K,

            d xmax / d J = (K + J dK/dJ) / (2 sqrt(K J)) ,

        so it too differentiates the interpolant that the evaluation reads.

        Parameters
        ----------
        j : float
            Action.
        tau : float or numpy.ndarray
            Anomaly.

        Returns
        -------
        numpy.ndarray
            The compensation integrand at the requested anomalies.

        Notes
        -----
        - 2026-08-29 - Written - Bovy (UofT)
        """
        _, _, K, dKdj = self._mm_tables(j)
        _, p = self._mm_xp_of_tau(j, tau)
        tau = numpy.atleast_1d(numpy.array(tau, dtype="float"))
        dxmaxdj = (K + j * dKdj) / (2.0 * numpy.sqrt(K * j))
        return p * dxmaxdj * numpy.cos(tau)

    def _mm_angle_of_tau(self, j, tau):
        """
        The angle at anomaly tau on the torus of action j.

        The matching condition makes the generating function explicit: the
        cumulative action of the target equals that of its auxiliary, so
        W = J (eta - sin eta cos eta) with eta = eta(tau; J).  The angle is
        its action derivative at fixed position, and the chain rule splits
        into a term at fixed anomaly and the boundary term that the moving
        turning points contribute,

            theta = (eta - sin eta cos eta)
                    + 2 J sin^2(eta) sum_m (dD_m/dJ) sin(m tau)
                    + p (d xmax / d J) cos(tau) ,

        the last being the grouped compensation.  Every ingredient is either
        closed form or a derivative of the stored interpolants, so no
        quadrature and no separately tabulated derivative enters.

        A constant pi/2 is subtracted to put the result in the convention of
        the forward transformation, which measures the angle from the
        midplane while the anomaly is measured from the turning point.  The
        offset is a choice of origin and nothing more: it comes out at
        pi/2 to 4e-13 independently of the torus and of the grid, while the
        anomaly-dependent part of the difference converges away as the grid
        is refined (2.5e-3, 1.3e-5, 1.7e-8, 1.1e-9 for 9, 17, 33 and 65
        energies), which is the family interpolation of dD_m/dJ and not an
        error of this relation.

        Parameters
        ----------
        j : float
            Action.
        tau : float or numpy.ndarray
            Anomaly.

        Returns
        -------
        numpy.ndarray
            The angle at the requested anomalies.

        Notes
        -----
        - 2026-08-29 - Written - Bovy (UofT)
        """
        D, dDdj, _, _ = self._mm_tables(j)
        tau = numpy.atleast_1d(numpy.array(tau, dtype="float"))
        ms = 2.0 * numpy.arange(1, len(D) + 1)
        eta = tau + numpy.sin(tau[:, None] * ms[None, :]) @ D
        detadj = numpy.sin(tau[:, None] * ms[None, :]) @ dDdj
        return (
            eta
            - numpy.sin(eta) * numpy.cos(eta)
            + 2.0 * j * numpy.sin(eta) ** 2.0 * detadj
            + self._mm_compensation(j, tau)
            - 0.5 * numpy.pi
        )

    def _mm_eval_tau(self, j, tau, tables, deriv=False):
        """
        Position, momentum, and angle at anomaly tau on the torus of action
        j, from table values read once, in a single pass, optionally with
        the closed-form derivative of the angle with respect to the anomaly.

        This is the evaluation kernel: it computes exactly what
        _mm_xp_of_tau, _mm_compensation and _mm_angle_of_tau compute, but
        shares the table read and the trigonometry between them, and adds
        the derivative that the Newton inversion of the angle relation needs.

        Parameters
        ----------
        j : float
            Action.
        tau : numpy.ndarray
            Anomalies.
        tables : tuple
            (D, dD/dj, K, dK/dj) as returned by _mm_tables(j).
        deriv : bool, optional
            If True, also return d(angle)/d(tau).

        Returns
        -------
        tuple
            (x, p, angle) or (x, p, angle, dangle/dtau).

        Notes
        -----
        - 2026-09-17 - Written - Bovy (UofT)
        """
        D, dDdj, K, dKdj = tables
        ms = 2.0 * numpy.arange(1, len(D) + 1)
        tau = numpy.atleast_1d(numpy.array(tau, dtype="float"))
        mt = tau[:, None] * ms[None, :]
        smt, cmt = numpy.sin(mt), numpy.cos(mt)
        eta = tau + smt @ D
        deta = 1.0 + cmt @ (ms * D)
        detadj = smt @ dDdj
        xmax = numpy.sqrt(K * j)
        dxmaxdj = (K + j * dKdj) / (2.0 * xmax)
        se, ce = numpy.sin(eta), numpy.cos(eta)
        st, ct = numpy.sin(tau), numpy.cos(tau)
        x = -xmax * ct
        p = numpy.zeros_like(tau)
        nz = st != 0.0
        p[nz] = 2.0 * j * se[nz] ** 2.0 * deta[nz] / (xmax * st[nz])
        angle = (
            eta
            - se * ce
            + 2.0 * j * se**2.0 * detadj
            + p * dxmaxdj * ct
            - 0.5 * numpy.pi
        )
        if not deriv:
            return x, p, angle
        d2eta = -smt @ (ms**2.0 * D)
        ddetadj = cmt @ (ms * dDdj)
        dp = numpy.zeros_like(tau)
        dp[nz] = (
            2.0
            * j
            / xmax
            * (
                (2.0 * se[nz] * ce[nz] * deta[nz] ** 2.0 + se[nz] ** 2.0 * d2eta[nz])
                / st[nz]
                - se[nz] ** 2.0 * deta[nz] * ct[nz] / st[nz] ** 2.0
            )
        )
        # exactly on a turning point the ratio's limit is finite:
        # p ~ (2 J / xmax) eta'^3 cos(tau) (tau - tau_0), so dp/dtau there is
        # its slope
        dp[~nz] = 2.0 * j / xmax * deta[~nz] ** 3.0 * ct[~nz]
        dangle = (
            2.0 * se**2.0 * deta
            + 2.0 * j * (2.0 * se * ce * deta * detadj + se**2.0 * ddetadj)
            + dxmaxdj * (dp * ct - p * st)
        )
        return x, p, angle, dangle

    def _mm_tau_of_angle(self, j, angle, tables):
        """
        Invert the angle relation: the anomaly at a requested angle.

        The angle advances monotonically with the anomaly, by exactly 2 pi
        over a libration, so the root on [0, 2 pi) is unique and can be
        bracketed.  Safeguarded Newton steps on the closed-form
        d(angle)/d(tau), started from the requested angle itself (the
        relation is the auxiliary's angle plus small corrections), reach
        double precision in a few iterations; a step that would leave the
        bracket, which shrinks around the root as the iteration proceeds, is
        replaced by the bracket's midpoint, so the inversion cannot fail.

        Parameters
        ----------
        j : float
            Action.
        angle : float or numpy.ndarray
            Angle.
        tables : tuple
            (D, dD/dj, K, dK/dj) as returned by _mm_tables(j), already
            in hand.

        Returns
        -------
        numpy.ndarray
            The anomaly at the requested angles.

        Notes
        -----
        - 2026-08-29 - Written - Bovy (UofT)
        - 2026-09-17 - Bracketed Newton replaces the 60-step bisection - Bovy (UofT)
        """
        # Solve in the anomaly's own origin: the relation runs from 0 to
        # 2 pi there, so it is monotone and unwrapped, while the requested
        # angle is measured from the midplane.
        angle = numpy.mod(
            numpy.atleast_1d(numpy.array(angle, dtype="float")) + 0.5 * numpy.pi,
            2.0 * numpy.pi,
        )
        lo = numpy.zeros_like(angle)
        hi = numpy.zeros_like(angle) + 2.0 * numpy.pi
        # The relation is the auxiliary's own angle plus small corrections,
        # so the requested angle itself is a starting point within a few
        # percent of the root; the bracket only serves as a safeguard.
        tau = numpy.clip(angle, 1e-6, 2.0 * numpy.pi - 1e-6)
        conv = numpy.zeros(len(angle), dtype="bool")
        for _ in range(40):
            _, _, th, dth = self._mm_eval_tau(j, tau, tables, deriv=True)
            f = th + 0.5 * numpy.pi - angle
            with numpy.errstate(divide="ignore", invalid="ignore"):
                step = f / dth
            # converged: the residual is at round-off, or the step could no
            # longer move the anomaly
            conv |= (numpy.fabs(f) < 8.0 * numpy.finfo(float).eps * (1.0 + angle)) | (
                numpy.fabs(step) <= 4.0 * numpy.finfo(float).eps * (1.0 + tau)
            )
            if numpy.all(conv):
                break
            low = f < 0.0
            lo = numpy.where(low, tau, lo)
            hi = numpy.where(low, hi, tau)
            new = tau - step
            bad = ~numpy.isfinite(new) | (new <= lo) | (new >= hi)
            new = numpy.where(bad, 0.5 * (lo + hi), new)
            tau = numpy.where(conv, tau, new)
        return tau

    def _mm_J_of_E(self, E):
        """
        The action at energy E, as the inverse of the Hermite energy
        interpolant E(J) of the family (the one whose derivative is the
        frequency), so that E(J(E)) = E to round-off.

        Parameters
        ----------
        E : float or numpy.ndarray
            Energy.

        Returns
        -------
        numpy.ndarray
            Action.

        Notes
        -----
        - 2026-09-18 - Written - Bovy (UofT)
        """
        shape = numpy.shape(E)
        E = numpy.atleast_1d(numpy.array(E, dtype="float"))
        out = numpy.empty_like(E)
        for ii, tE in enumerate(E):
            if isinstance(self._mm_E, _linearHermite):
                out[ii] = self._mm_E._j0 + (tE - self._mm_E._y0) / self._mm_E._dy0
                continue
            roots = numpy.real(self._mm_E.solve(tE, extrapolate=True))
            # E(J) is monotonic on the grid; the extrapolated end pieces may
            # turn over and add far-away roots, so take the root nearest
            # the grid
            out[ii] = roots[
                numpy.argmin(
                    numpy.fabs(roots - numpy.clip(roots, self._js[0], self._js[-1]))
                )
            ]
        return out.reshape(shape)

    def _mm_xp_of_angle(self, j, angle):
        """
        The canonical evaluation: position and momentum at a requested
        action and angle, through the momentum-matched map.

        This is the composition the construction is built to deliver -- the
        angle shift, the auxiliary's inverse, and the inverse cotangent lift
        -- with every ingredient either closed form or a derivative of the
        stored interpolants.  The tables are read once per call.

        Parameters
        ----------
        j : float
            Action.
        angle : float or numpy.ndarray
            Angle.

        Returns
        -------
        tuple
            (x, p) at the requested angles.

        Notes
        -----
        - 2026-08-29 - Written - Bovy (UofT)
        """
        if j < 0.0:
            raise ValueError("The action must be non-negative")
        if j == 0.0:
            # the zero-action torus is the point at the bottom of the potential
            zero = numpy.zeros_like(numpy.atleast_1d(numpy.array(angle, dtype="float")))
            return zero, zero
        tables = self._mm_tables(j)
        tau = self._mm_tau_of_angle(j, angle, tables=tables)
        x, p, _ = self._mm_eval_tau(j, tau, tables)
        return x, p

    def _setup_pointtransform_exact(self, pt_nxa):
        # Setup the exact point transformation for each torus by direct
        # quadrature of the time-from-midplane profile and monotone spline
        # inversion (the map equates times along the true and auxiliary
        # orbits; its profile integrand is regular everywhere, including at
        # the turning point); the result is stored as the normalized mapping
        # x/xmax(xa/ptxmax) sampled on the fixed mesh self._pt_xamesh and is
        # evaluated using spline interpolation. _pt_deriv_coeffs and
        # _pt_deriv2_coeffs are not used in this case (derivatives come from
        # the spline), but are kept for shape compatibility
        # The mapping is stored sampled on a mesh that extends beyond the
        # [-1,1] range of the normalized coordinate, because the stored
        # arrays are evaluated using ndimage 2D-spline interpolation whose
        # mirror boundary condition would otherwise distort the interpolation
        # near the turning points; the padding pushes that edge artifact,
        # which decays geometrically, well below the ODE tolerance inside
        # [-1,1]
        self._pt_nmesh = 2 * pt_nxa - 1
        self._pt_pad = 4 * self._exact_pt_spl_deg + 16
        dmesh = 2.0 / (self._pt_nmesh - 1.0)
        self._pt_xamesh = numpy.linspace(
            -1.0 - self._pt_pad * dmesh,
            1.0 + self._pt_pad * dmesh,
            self._pt_nmesh + 2 * self._pt_pad,
        )
        # Initialize all tori to the identity mapping (which remains the
        # mapping used for small J); for the exact point transformation, the
        # deriv arrays hold the derivatives of the normalized mapping with
        # respect to the normalized coordinate sampled on the same mesh
        self._pt_coeffs = numpy.tile(self._pt_xamesh, (self._nE, 1))
        self._pt_deriv_coeffs = numpy.ones_like(self._pt_coeffs)
        self._pt_deriv2_coeffs = numpy.zeros_like(self._pt_coeffs)
        xanormmesh = numpy.linspace(0.0, 1.0, pt_nxa)
        # Just use the identity for small J
        zIndx = self._js < 1e-10
        self._pt_xmaxs[zIndx] = self._xmaxs[zIndx] + 1e-10  # avoid /0
        gIndx = True ^ zIndx
        if numpy.any(gIndx):
            # Aux. torus = harmonic-oscillator torus with the same action
            # (because omega = Omega, this is also the same-frequency torus)
            Es = self._Es[gIndx]
            xmaxs = self._xmaxs[gIndx]
            ng = numpy.sum(gIndx)
            # Construct the equal-time map by direct quadrature and monotone
            # inversion instead of integrating its defining ODE: writing the
            # normalized position as y = x/xmax = sin(chi), the time from the
            # midplane is t(chi) = xmax int_0^chi dchi' / sqrt(Q), with
            # Q = v^2/(1-y^2) regular over the whole quarter period
            # (Q -> -F(xmax) xmax at the turning point), and the map is
            # chi(thetaa) = t^{-1}(thetaa/omega). Using time normalized by
            # the quarter period enforces the exact turning-point closure
            # pi(xa_max) = xmax without relying on the accuracy of omega, and
            # the composite Gauss-Legendre quadrature has no tolerance knob:
            # every profile is at machine precision on the chi mesh
            Qmax = -evaluatelinearForces(self._pot, xmaxs, use_physical=False) * xmaxs
            nchimesh = numpy.amax([4 * pt_nxa, 801])
            chimesh = numpy.linspace(0.0, numpy.pi / 2.0, nchimesh)
            midp = 0.5 * (chimesh[:-1] + chimesh[1:])
            half = 0.5 * (chimesh[1:] - chimesh[:-1])
            nodes = (midp[:, None] + half[:, None] * _GLX[None, :]).ravel()
            sinn, cosn2 = numpy.sin(nodes), numpy.cos(nodes) ** 2.0
            v2 = 2.0 * (
                Es[:, None]
                - evaluatelinearPotentials(
                    self._pot,
                    (xmaxs[:, None] * sinn[None, :]).ravel(),
                    use_physical=False,
                ).reshape(ng, len(nodes))
            )
            v2[v2 < 0.0] = 0.0
            Q = numpy.empty_like(v2)
            reg = cosn2 > 1e-6
            Q[:, reg] = v2[:, reg] / cosn2[None, reg]
            Q[:, True ^ reg] = numpy.tile(Qmax[:, None], (1, numpy.sum(True ^ reg)))
            Q[Q < numpy.finfo(float).tiny] = numpy.finfo(float).tiny
            panels = (
                (half[None, :, None] * _GLW[None, None, :])
                * (1.0 / numpy.sqrt(Q)).reshape(ng, len(half), len(_GLX))
            ).sum(axis=-1)
            tmesh = numpy.hstack(
                (numpy.zeros((ng, 1)), numpy.cumsum(panels, axis=1))
            )  # time / xmax on the chi mesh; only ratios to t(pi/2) are used
            xanormmesh = numpy.linspace(0.0, 1.0, pt_nxa)
            ttargets = (
                numpy.arcsin(xanormmesh)[None, :]
                / (numpy.pi / 2.0)
                * tmesh[:, -1][:, None]
            )
            ynorm = numpy.empty((ng, pt_nxa))
            for jj in range(ng):
                ynorm[jj] = numpy.sin(
                    interpolate.InterpolatedUnivariateSpline(
                        tmesh[jj], chimesh, k=self._exact_pt_spl_deg
                    )(ttargets[jj])
                )
            ynorm[:, 0] = 0.0
            ynorm[:, -1] = 1.0
            # Odd reflection onto the full [-1,1] mesh (symmetric potential)
            ynormfull = numpy.hstack((-ynorm[:, :0:-1], ynorm))
            # Represent as a spline and sample the mapping and its
            # derivatives on the extended mesh (polynomial extrapolation of
            # the end pieces beyond [-1,1], see above)
            coremesh = numpy.linspace(-1.0, 1.0, self._pt_nmesh)
            for tynorm, ii in zip(ynormfull, numpy.arange(self._nE)[gIndx]):
                tspl = interpolate.InterpolatedUnivariateSpline(
                    coremesh, tynorm, k=self._exact_pt_spl_deg
                )
                self._pt_coeffs[ii] = tspl(self._pt_xamesh)
                self._pt_deriv_coeffs[ii] = tspl(self._pt_xamesh, nu=1)
                self._pt_deriv2_coeffs[ii] = tspl(self._pt_xamesh, nu=2)
        # Store spline-filtered versions for fast 2D-spline evaluation of the
        # mapping and its derivatives at (torus,xa/ptxmax) points
        self._pt_filtered = tuple(
            ndimage.spline_filter(arr, order=self._exact_pt_spl_deg)
            for arr in (
                self._pt_coeffs,
                self._pt_deriv_coeffs,
                self._pt_deriv2_coeffs,
            )
        )
        return None

    def _create_xgrid(self):
        # Find x grid for regular grid in auxiliary angle (thetaa)
        # in practice only need to map 0 < thetaa < pi/2  to +x with +v bc symm
        # To efficiently start the search, we first compute thetaa for a dense
        # grid in x (at +v)
        xgrid = numpy.linspace(-1.0, 1.0, 2 * self._nta)
        xs = xgrid * numpy.atleast_2d(self._pt_xmaxs).T
        if self._pt_exact:
            # For the exact point transformation, the positional
            # ptcoeffs-style arguments carry the row index of each point's
            # torus in the grid of tori instead of polynomial coefficients
            tptcoeffs = numpy.tile(
                numpy.arange(self._nE, dtype="float"), (xs.shape[1], 1)
            ).T
            tptderivcoeffs = tptcoeffs
        else:
            tptcoeffs = numpy.rollaxis(
                numpy.tile(self._pt_coeffs, (xs.shape[1], 1, 1)), 1
            )
            tptderivcoeffs = numpy.rollaxis(
                numpy.tile(self._pt_deriv_coeffs, (xs.shape[1], 1, 1)), 1
            )
        xta = _anglea(
            xs,
            numpy.tile(self._Es, (xs.shape[1], 1)).T,
            self._pot,
            numpy.tile(self._hoaa._omega, (xs.shape[1], 1)).T,
            tptcoeffs,
            tptderivcoeffs,
            numpy.tile(self._xmaxs, (xs.shape[1], 1)).T,
            numpy.tile(self._pt_xmaxs, (xs.shape[1], 1)).T,
            **self._pt_eval_kwargs,
        )
        xta[numpy.isnan(xta)] = 0.0  # Zero energy orbit -> NaN
        # Now use Newton-Raphson to iterate to a regular grid
        cindx = numpy.nanargmin(
            numpy.fabs(
                (xta - numpy.rollaxis(numpy.atleast_3d(self._thetaa), 1) + numpy.pi)
                % (2.0 * numpy.pi)
                - numpy.pi
            ),
            axis=2,
        )
        xgrid = xgrid[cindx].T * numpy.atleast_2d(self._pt_xmaxs).T
        if self._pt_exact:
            # With the exact point transformation, the auxiliary angle is
            # simply the harmonic-oscillator angle of xa on the auxiliary
            # torus, so the grid follows in closed form; the Newton-Raphson
            # iteration below then merely polishes this to be exactly
            # consistent with the stored spline representation
            xgrid = numpy.atleast_2d(self._pt_xmaxs).T * numpy.sin(self._thetaa)
        Egrid = numpy.tile(self._Es, (self._nta, 1)).T
        omegagrid = numpy.tile(self._hoaa._omega, (self._nta, 1)).T
        xmaxgrid = numpy.tile(self._xmaxs, (self._nta, 1)).T
        ptxmaxgrid = numpy.tile(self._pt_xmaxs, (self._nta, 1)).T
        if self._pt_exact:
            ptcoeffsgrid = numpy.tile(
                numpy.arange(self._nE, dtype="float"), (self._nta, 1)
            ).T
            ptderivcoeffsgrid = ptcoeffsgrid
            ptderiv2coeffsgrid = ptcoeffsgrid
        else:
            ptcoeffsgrid = numpy.rollaxis(
                numpy.tile(self._pt_coeffs, (self._nta, 1, 1)), 1
            )
            ptderivcoeffsgrid = numpy.rollaxis(
                numpy.tile(self._pt_deriv_coeffs, (self._nta, 1, 1)), 1
            )
            ptderiv2coeffsgrid = numpy.rollaxis(
                numpy.tile(self._pt_deriv2_coeffs, (self._nta, 1, 1)), 1
            )
        ta = _anglea(
            xgrid,
            Egrid,
            self._pot,
            omegagrid,
            ptcoeffsgrid,
            ptderivcoeffsgrid,
            xmaxgrid,
            ptxmaxgrid,
            **self._pt_eval_kwargs,
        )
        mta = numpy.tile(self._thetaa, (len(self._Es), 1))
        # Now iterate
        cntr = 0
        unconv = numpy.ones(xgrid.shape, dtype="bool")
        # We'll fill in the -v part using the +v, also remove the endpoints
        unconv[:, self._nta // 4 : 3 * self._nta // 4 + 1] = False
        # Also don't bother with J=0 torus
        unconv[numpy.tile(self._js, (self._nta, 1)).T < 1e-10] = False
        dta = (ta[unconv] - mta[unconv] + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
        unconv[unconv] = numpy.fabs(dta) > self._angle_tol
        # Don't allow too big steps
        maxdx = numpy.tile(self._pt_xmaxs / float(self._nta), (self._nta, 1)).T
        while not self._bisect:
            dtadx = _danglea(
                xgrid[unconv],
                Egrid[unconv],
                self._pot,
                omegagrid[unconv],
                ptcoeffsgrid[unconv],
                ptderivcoeffsgrid[unconv],
                ptderiv2coeffsgrid[unconv],
                xmaxgrid[unconv],
                ptxmaxgrid[unconv],
                **self._pt_eval_kwargs,
            )
            dta = (ta[unconv] - mta[unconv] + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
            dx = -dta / dtadx
            dx[numpy.fabs(dx) > maxdx[unconv]] = (numpy.sign(dx) * maxdx[unconv])[
                numpy.fabs(dx) > maxdx[unconv]
            ]
            xgrid[unconv] += dx
            xgrid[unconv * (xgrid > ptxmaxgrid)] = ptxmaxgrid[
                unconv * (xgrid > ptxmaxgrid)
            ]
            xgrid[unconv * (xgrid < -ptxmaxgrid)] = ptxmaxgrid[
                unconv * (xgrid < -ptxmaxgrid)
            ]
            newta = _anglea(
                xgrid[unconv],
                Egrid[unconv],
                self._pot,
                omegagrid[unconv],
                ptcoeffsgrid[unconv],
                ptderivcoeffsgrid[unconv],
                xmaxgrid[unconv],
                ptxmaxgrid[unconv],
                **self._pt_eval_kwargs,
            )
            ta[unconv] = newta
            unconv[unconv] = numpy.fabs(dta) > self._angle_tol
            cntr += 1
            if numpy.sum(unconv) == 0:
                break
            if cntr > self._maxiter:
                warnings.warn(
                    "Torus mapping with Newton-Raphson did not converge in {} iterations, falling back onto simple bisection (increase maxiter to try harder with Newton-Raphson)".format(
                        self._maxiter
                    ),
                    galpyWarning,
                )
                break
        if self._bisect or cntr > self._maxiter:
            # Reset cntr
            cntr = 0
            # Start from nearest guess from below
            new_xgrid = numpy.linspace(-1.0, 1.0, 2 * self._nta)
            da = (
                xta - numpy.rollaxis(numpy.atleast_3d(self._thetaa), 1) + numpy.pi
            ) % (2.0 * numpy.pi) - numpy.pi
            da[da >= 0.0] = -numpy.nanmax(numpy.fabs(da)) - 0.1
            cindx = numpy.nanargmax(da, axis=2)
            tryx_min = (new_xgrid[cindx].T * numpy.atleast_2d(self._pt_xmaxs).T)[unconv]
            dx = (
                2.0 / (2.0 * self._nta - 1) * ptxmaxgrid
            )  # delta of initial x grid above
            while True:
                dx *= 0.5
                xgrid[unconv] = tryx_min + dx[unconv]
                newta = (
                    _anglea(
                        xgrid[unconv],
                        Egrid[unconv],
                        self._pot,
                        omegagrid[unconv],
                        ptcoeffsgrid[unconv],
                        ptderivcoeffsgrid[unconv],
                        xmaxgrid[unconv],
                        ptxmaxgrid[unconv],
                        **self._pt_eval_kwargs,
                    )
                    + 2.0 * numpy.pi
                ) % (2.0 * numpy.pi)
                ta[unconv] = newta
                dta = (newta - mta[unconv] + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
                tryx_min[newta < mta[unconv]] = xgrid[unconv][newta < mta[unconv]]
                unconv[unconv] = numpy.fabs(dta) > self._angle_tol
                tryx_min = tryx_min[numpy.fabs(dta) > self._angle_tol]
                cntr += 1
                if numpy.sum(unconv) == 0:
                    break
                if cntr > self._maxiter:
                    warnings.warn(
                        "Torus mapping with bisection did not converge in {} iterations".format(
                            self._maxiter
                        )
                        + " for energies:"
                        + "".join(f" {k:g}" for k in sorted(set(Egrid[unconv]))),
                        galpyWarning,
                    )
                    break
        xgrid[:, self._nta // 4 + 1 : self._nta // 2 + 1] = xgrid[:, : self._nta // 4][
            :, ::-1
        ]
        xgrid[:, self._nta // 2 + 1 : 3 * self._nta // 4 + 1] = xgrid[
            :, 3 * self._nta // 4 :
        ][:, ::-1]
        ta[:, self._nta // 4 + 1 : 3 * self._nta // 4] = _anglea(
            xgrid[:, self._nta // 4 + 1 : 3 * self._nta // 4],
            Egrid[:, self._nta // 4 + 1 : 3 * self._nta // 4],
            self._pot,
            omegagrid[:, self._nta // 4 + 1 : 3 * self._nta // 4],
            ptcoeffsgrid[:, self._nta // 4 + 1 : 3 * self._nta // 4],
            ptderivcoeffsgrid[:, self._nta // 4 + 1 : 3 * self._nta // 4],
            xmaxgrid[:, self._nta // 4 + 1 : 3 * self._nta // 4],
            ptxmaxgrid[:, self._nta // 4 + 1 : 3 * self._nta // 4],
            vsign=-1.0,
            **self._pt_eval_kwargs,
        )
        self._dta = (ta - mta + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
        self._mta = mta
        # Store these, they are useful (obv. arbitrary to return xgrid
        # and not just store it...)
        self._Egrid = Egrid
        self._omegagrid = omegagrid
        self._ptcoeffsgrid = ptcoeffsgrid
        self._ptderivcoeffsgrid = ptderivcoeffsgrid
        self._ptderiv2coeffsgrid = ptderiv2coeffsgrid
        self._ptxmaxgrid = ptxmaxgrid
        self._xmaxgrid = xmaxgrid
        return xgrid

    def plot_convergence(
        self, E, overplot=False, return_gridspec=False, shift_action=None
    ):
        if self._momentum_matched:
            return self._plot_convergence_mm(
                E, overplot=overplot, return_gridspec=return_gridspec
            )
        if shift_action is None:
            shift_action = self._pt_deg > 1
        # First find the torus for this energy
        indx = numpy.nanargmin(numpy.fabs(E - self._Es))
        if numpy.fabs(E - self._Es[indx]) > 1e-10:
            raise ValueError(
                "Given energy not found; please specify an energy used in the initialization of the instance"
            )
        if not overplot:
            gs = gridspec.GridSpec(2, 3, height_ratios=[4, 1])
        else:
            gs = overplot  # confusingly, we overload the meaning of overplot
        # mapping of thetaa --> x
        pyplot.subplot(gs[0])
        plot.plot(
            self._thetaa,
            self._xgrid[indx],
            color="k",
            ls="--" if overplot else "-",
            ylabel=r"$x(\theta^A)$",
            gcf=True,
            overplot=overplot,
        )
        if not overplot:
            pyplot.gca().xaxis.set_major_formatter(NullFormatter())
        if not overplot:
            pyplot.subplot(gs[3])
            negv = (self._thetaa > numpy.pi / 2.0) * (
                self._thetaa < 3.0 * numpy.pi / 2.0
            )
            thetaa_out = numpy.empty_like(self._thetaa)
            one = numpy.ones(numpy.sum(True ^ negv))
            thetaa_out[True ^ negv] = _anglea(
                self._xgrid[indx][True ^ negv],
                E,
                self._pot,
                self._OmegaHO[indx],
                indx * one if self._pt_exact else self._pt_coeffs[indx],
                indx * one
                if self._pt_exact
                else numpy.tile(
                    self._pt_deriv_coeffs[indx], (numpy.sum(True ^ negv), 1)
                ),
                self._xmaxs[indx] * one,
                self._pt_xmaxs[indx] * one,
                vsign=1.0,
                **self._pt_eval_kwargs,
            )
            one = numpy.ones(numpy.sum(negv))
            thetaa_out[negv] = _anglea(
                self._xgrid[indx][negv],
                E,
                self._pot,
                self._OmegaHO[indx],
                indx * one if self._pt_exact else self._pt_coeffs[indx],
                indx * one
                if self._pt_exact
                else numpy.tile(self._pt_deriv_coeffs[indx], (numpy.sum(negv), 1)),
                self._xmaxs[indx] * one,
                self._pt_xmaxs[indx] * one,
                vsign=-1.0,
                **self._pt_eval_kwargs,
            )
            plot.plot(
                self._thetaa,
                ((thetaa_out - self._thetaa + numpy.pi) % (2.0 * numpy.pi)) - numpy.pi,
                color="k",
                gcf=True,
                xlabel=r"$\theta^A$",
                ylabel=r"$\theta^A[x(\theta^A)]-\theta^A$",
            )
        # Recovery of the nSn from J^A(theta^A) behavior
        pyplot.subplot(gs[1])
        plot.plot(
            self._thetaa,
            self._ja[indx],
            color="k",
            ls="--" if overplot else "-",
            ylabel=r"$J^A(\theta^A),J$",
            gcf=True,
            overplot=overplot,
        )
        pyplot.axhline(
            self._js[indx] + (0.0 if shift_action else self._jaoffset[indx]),
            color="k",
            ls="--",
        )
        if not overplot:
            pyplot.gca().xaxis.set_major_formatter(NullFormatter())
        if not overplot:
            pyplot.subplot(gs[4])
            plot.plot(
                self._thetaa,
                numpy.array(
                    [
                        self._js[indx]
                        + self._jaoffset[indx]
                        + 2.0 * numpy.sum(self._nSn[indx] * numpy.cos(self._nforSn * x))
                        for x in self._thetaa
                    ]
                )
                / self._ja[indx]
                - 1.0,
                color="k",
                xlabel=r"$\theta^A$",
                ylabel=r"$\delta J^A/J^A$",
                gcf=True,
            )
        # Recovery of the dSndJ from dJ^A/dJ(theta^A) behavior
        pyplot.subplot(gs[2])
        plot.plot(
            self._thetaa,
            self._djadj[indx] / numpy.nanmean(self._djadj[indx]),
            color="k",
            ls="--" if overplot else "-",
            ylabel=r"$\mathrm{d}J^A/\mathrm{d}J(\theta^A)$",
            gcf=True,
            overplot=overplot,
        )
        pyplot.axhline(1.0, color="k", ls="--")
        if not overplot:
            pyplot.gca().xaxis.set_major_formatter(NullFormatter())
        if not overplot:
            pyplot.subplot(gs[5])
            plot.plot(
                self._thetaa,
                numpy.array(
                    [
                        1.0
                        + 2.0
                        * numpy.sum(
                            self._nforSn
                            * self._dSndJ[indx]
                            * numpy.cos(self._nforSn * x)
                        )
                        for x in self._thetaa
                    ]
                )
                - self._djadj[indx] / numpy.nanmean(self._djadj[indx]),
                color="k",
                xlabel=r"$\theta^A$",
                ylabel=r"$\delta \mathrm{d}J^A/\mathrm{d}J(\theta^A)$",
                gcf=True,
            )
        pyplot.tight_layout()
        if return_gridspec:
            return gs
        else:
            return None

    def _plot_convergence_mm(self, E, overplot=False, return_gridspec=False):
        # The momentum-matched map on the torus of energy E: the anomaly map,
        # the matching residual, the reconstruction's energy error, and the
        # angle against the forward transformation
        indx = numpy.nanargmin(numpy.fabs(E - self._Es))
        if numpy.fabs(E - self._Es[indx]) > 1e-10:
            raise ValueError(
                "Given energy not found; please specify an energy used in the initialization of the instance"
            )
        J, xmax = self._js[indx], self._xmaxs[indx]
        tau, x, p, g, Jsweep, A, Om = self._mm_sweep(E, xmax, self._mm_nta)
        ms = 2.0 * numpy.arange(1, self._mm_npt + 1)
        eta = tau + numpy.sin(tau[:, None] * ms[None, :]) @ self._mm_D[indx]
        xr, pr = self._mm_xp_of_tau(J, tau)
        Er = 0.5 * pr**2.0 + evaluatelinearPotentials(self._pot, xr, use_physical=False)
        if J > 0.0:
            th = self._mm_angle_of_tau(J, tau)
            # the forward angle is undefined at the turning points
            keep = numpy.fabs(pr) > 1e-6 * numpy.amax(numpy.fabs(pr))
            _, _, thf = self._aAV.actionsFreqsAngles(xr[keep], pr[keep])
            dth = (th[keep] - thf + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
        else:
            # the zero-action torus is a point with no angle to compare
            keep = numpy.ones(len(tau), dtype="bool")
            dth = numpy.zeros(len(tau))
        if not overplot:
            gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)
        else:
            gs = overplot  # confusingly, we overload the meaning of overplot
        panels = (
            (tau, eta - tau, r"$\eta - \tau$"),
            (
                tau,
                (J * (eta - numpy.sin(eta) * numpy.cos(eta)) - A) / (2.0 * numpy.pi * J)
                if J > 0.0
                else 0.0 * tau,
                r"$[A^A(\eta)-A(\tau)]/2\pi J$",
            ),
            (tau, Er / E - 1.0 if E != 0.0 else Er, r"$E(x,p)/E - 1$"),
            (tau[keep], dth, r"$\theta - \theta_{\mathrm{forward}}$"),
        )
        for ii, (xx, yy, ylabel) in enumerate(panels):
            pyplot.subplot(gs[ii])
            plot.plot(
                xx,
                yy,
                color="k",
                ls="--" if overplot else "-",
                gcf=True,
                overplot=overplot,
                xrange=[0.0, 2.0 * numpy.pi],
                xlabel=r"$\tau$",
                ylabel=ylabel,
            )
        if not overplot:
            pyplot.suptitle(rf"$E = {E:g}$")
        if return_gridspec:
            return gs
        else:
            return None

    def plot_power(self, Es, symm=True, overplot=False, return_gridspec=False, ls="-"):
        Es = numpy.sort(numpy.atleast_1d(Es))
        if self._momentum_matched:
            # the momentum-matched map's harmonics (all even) and their action
            # derivatives play the role of n S_n and dS_n/dJ
            nfor = 2.0 * numpy.arange(1, self._mm_npt + 1)
            tab1, tab2 = self._mm_D, self._mm_dD
            lab1, lab2, xlab = r"$|D_m|$", r"$|\mathrm{d}D_m/\mathrm{d}J|$", r"$m$"
            sl = slice(None)
        else:
            nfor, tab1, tab2 = self._nforSn, self._nSn, self._dSndJ
            lab1, lab2, xlab = r"$|nS_n|$", r"$|\mathrm{d}S_n/\mathrm{d}J|$", r"$n$"
            sl = slice(symm, None, symm + 1)
        minn_for_cmap = 4
        if len(Es) < minn_for_cmap:
            if not overplot:
                gs = gridspec.GridSpec(1, 2)
            else:
                gs = overplot  # confusingly, we overload the meaning of overplot
        else:
            if not overplot:
                outer = gridspec.GridSpec(1, 2, width_ratios=[2.0, 0.05], wspace=0.05)
                gs = gridspec.GridSpecFromSubplotSpec(
                    1, 2, subplot_spec=outer[0], wspace=0.35
                )
            else:
                raise RuntimeError(
                    f"plot_power with >= {minn_for_cmap} energies and overplot=True is not supported"
                )
        for ii, E in enumerate(Es):
            # First find the torus for this energy
            indx = numpy.nanargmin(numpy.fabs(E - self._Es))
            if numpy.fabs(E - self._Es[indx]) > 1e-10:
                raise ValueError(
                    "Given energy not found; please specify an energy used in the initialization of the instance"
                )
            # n S_n
            y = numpy.fabs(tab1[indx, sl])
            if len(Es) > 1 and E == Es[0]:
                y4minmax = numpy.fabs(tab1[:, sl])
                ymin = numpy.amax(
                    [numpy.amin(y4minmax[numpy.isfinite(y4minmax)]), 1e-17]
                )
                ymax = numpy.amax(y4minmax[numpy.isfinite(y4minmax)])
            elif len(Es) == 1:
                ymin = numpy.amax([numpy.amin(y[numpy.isfinite(y)]), 1e-17])
                ymax = numpy.amax(y[numpy.isfinite(y)])
            if len(Es) < minn_for_cmap:
                label = rf"$E = {E:g}$"
                color = f"C{ii}"
            else:
                label = None
                color = cm.plasma((E - Es[0]) / (Es[-1] - Es[0]))
            pyplot.subplot(gs[0])
            plot.plot(
                numpy.fabs(nfor[sl]),
                y,
                yrange=[ymin, ymax],
                ls=ls,
                gcf=True,
                semilogy=True,
                overplot=overplot,
                xrange=[0.0, nfor[-1]],
                label=label,
                color=color,
                xlabel=xlab,
                ylabel=lab1,
            )
            # d S_n / d J
            y = numpy.fabs(tab2[indx, sl])
            if len(Es) > 1 and E == Es[0]:
                y4minmax = numpy.fabs(tab2[:, sl])
                ymin = numpy.amax(
                    [numpy.amin(y4minmax[numpy.isfinite(y4minmax)]), 1e-17]
                )
                ymax = numpy.amax(y4minmax[numpy.isfinite(y4minmax)])
            elif len(Es) == 1:
                ymin = numpy.amax([numpy.amin(y[numpy.isfinite(y)]), 1e-17])
                ymax = numpy.amax(y[numpy.isfinite(y)])
            if len(Es) < minn_for_cmap:
                label = rf"$E = {E:g}$"
                color = f"C{ii}"
            else:
                label = None
                color = cm.plasma((E - Es[0]) / (Es[-1] - Es[0]))
            pyplot.subplot(gs[1])
            plot.plot(
                numpy.fabs(nfor[sl]),
                y,
                yrange=[ymin, ymax],
                ls=ls,
                gcf=True,
                semilogy=True,
                overplot=overplot,
                xrange=[0.0, nfor[-1]],
                label=label,
                color=color,
                xlabel=xlab,
                ylabel=lab2,
            )
            if not overplot == gs:
                overplot = True
        if len(Es) < minn_for_cmap:
            if not overplot == gs:
                pyplot.subplot(gs[0])
                pyplot.legend(fontsize=17.0, frameon=False)
                pyplot.subplot(gs[1])
                pyplot.legend(fontsize=17.0, frameon=False)
                pyplot.tight_layout()
        else:
            pyplot.subplot(outer[1])
            sm = pyplot.cm.ScalarMappable(
                cmap=cm.plasma, norm=pyplot.Normalize(vmin=Es[0], vmax=Es[-1])
            )
            sm._A = []
            cbar = pyplot.colorbar(
                sm, cax=pyplot.gca(), use_gridspec=True, format=r"$%g$"
            )
            cbar.set_label(r"$E$")
            outer.tight_layout(pyplot.gcf())
        if return_gridspec:
            return gs
        else:
            return None

    def plot_orbit(self, E):
        ta = numpy.linspace(0.0, 2.0 * numpy.pi, 1001)
        if not self._interp:
            # First find the torus for this energy
            indx = numpy.nanargmin(numpy.fabs(E - self._Es))
            if numpy.fabs(E - self._Es[indx]) > 1e-10:
                raise ValueError(
                    "Given energy not found; please specify an energy used in the initialization of the instance"
                )
            tJ = self._js[indx]
        else:
            tJ = self.J(E)
        x, v = self(tJ, ta)
        # First plot orbit in x,v
        pyplot.subplot(1, 2, 1)
        plot.plot(
            x,
            v,
            xlabel=r"$x$",
            ylabel=r"$v$",
            gcf=True,
            color="k",
            xrange=[1.1 * numpy.amin(x), 1.1 * numpy.amax(x)],
            yrange=[1.1 * numpy.amin(v), 1.1 * numpy.amax(v)],
        )
        # Then plot energy
        pyplot.subplot(1, 2, 2)
        Eorbit = (
            v**2.0 / 2.0 + evaluatelinearPotentials(self._pot, x, use_physical=False)
        ) / E - 1.0
        ymin, ymax = numpy.amin(Eorbit), numpy.amax(Eorbit)
        plot.plot(
            ta,
            Eorbit,
            xrange=[0.0, 2.0 * numpy.pi],
            yrange=[ymin - (ymax - ymin) * 3.0, ymax + (ymax - ymin) * 3.0],
            gcf=True,
            color="k",
            xlabel=r"$\theta$",
            ylabel=r"$E/E_{\mathrm{true}}-1$",
        )
        pyplot.tight_layout()
        return None

    ################### FUNCTIONS FOR INTERPOLATION BETWEEN TORI###############
    def _setup_interp(self):
        self._Emin = self._Es[0]
        self._Emax = self._Es[-1]
        self._nnSn = self._nSn.shape[1]  # won't be confusing...
        self._nSnNormalize = numpy.ones(self._nnSn)
        self._nSnFiltered = ndimage.spline_filter(self._nSn, order=3)
        self._dSndJFiltered = ndimage.spline_filter(self._dSndJ, order=3)
        self.J = interpolate.InterpolatedUnivariateSpline(self._Es, self._js, k=3)
        self.E = interpolate.InterpolatedUnivariateSpline(self._js, self._Es, k=3)
        self.jaoffset = interpolate.InterpolatedUnivariateSpline(
            self._Es, self._jaoffset, k=3
        )
        self.OmegaHO = interpolate.InterpolatedUnivariateSpline(
            self._Es, self._OmegaHO, k=3
        )
        self.Omega = interpolate.InterpolatedUnivariateSpline(
            self._Es, self._Omegas, k=3
        )
        self.xmax = interpolate.InterpolatedUnivariateSpline(self._Es, self._xmaxs, k=3)
        self.ptxmax = interpolate.InterpolatedUnivariateSpline(
            self._Es, self._pt_xmaxs, k=3
        )
        self._nptcoeffs = self._pt_coeffs.shape[1]
        self._ptcoeffsFiltered = ndimage.spline_filter(self._pt_coeffs, order=3)
        self._ptderivcoeffsFiltered = ndimage.spline_filter(
            self._pt_deriv_coeffs, order=3
        )
        return None

    def _coords_for_map_coords(self, E):
        coords = numpy.empty((2, self._nnSn * len(E)))
        coords[0] = numpy.tile(
            (E - self._Emin) / (self._Emax - self._Emin) * (self._nE - 1.0),
            (self._nnSn, 1),
        ).T.flatten()
        coords[1] = numpy.tile(self._nforSn - 1, (len(E), 1)).flatten()
        return coords

    def nSn(self, E):
        if self._momentum_matched:
            raise RuntimeError(
                "nSn is part of the older evaluation and is not available for momentum_matched=True"
            )
        if not self._interp:
            raise RuntimeError(
                "To evaluate nSn, interpolation must be activated at instantiation using setup_interp=True"
            )
        evalE = numpy.atleast_1d(E)
        indxc = (evalE >= self._Emin) * (evalE <= self._Emax)
        coords = self._coords_for_map_coords(evalE[indxc])
        out = numpy.empty((len(evalE), self._nnSn))
        out[indxc] = numpy.reshape(
            ndimage.map_coordinates(
                self._nSnFiltered, coords, order=3, prefilter=False
            ),
            (numpy.sum(indxc), self._nnSn),
        )
        out[True ^ indxc] = numpy.nan
        return out

    def dSndJ(self, E):
        if self._momentum_matched:
            raise RuntimeError(
                "dSndJ is part of the older evaluation and is not available for momentum_matched=True"
            )
        if not self._interp:
            raise RuntimeError(
                "To evaluate dnSndJ, interpolation must be activated at instantiation using setup_interp=True"
            )
        evalE = numpy.atleast_1d(E)
        indxc = (evalE >= self._Emin) * (evalE <= self._Emax)
        coords = self._coords_for_map_coords(evalE[indxc])
        out = numpy.empty((len(evalE), self._nnSn))
        out[indxc] = numpy.reshape(
            ndimage.map_coordinates(
                self._dSndJFiltered, coords, order=3, prefilter=False
            ),
            (numpy.sum(indxc), self._nnSn),
        )
        out[True ^ indxc] = numpy.nan
        return out

    def _coords_for_map_coords_pt(self, E, deriv=False):
        coords = numpy.empty((2, (self._nptcoeffs - deriv) * len(E)))
        coords[0] = numpy.tile(
            (E - self._Emin) / (self._Emax - self._Emin) * (self._nE - 1.0),
            (self._nptcoeffs - deriv, 1),
        ).T.flatten()
        coords[1] = numpy.tile(
            numpy.arange(self._nptcoeffs - deriv), (len(E), 1)
        ).flatten()
        return coords

    def pt_coeffs(self, E):
        if self._momentum_matched:
            raise RuntimeError(
                "pt_coeffs is part of the older evaluation and is not available for momentum_matched=True"
            )
        if not self._interp:
            raise RuntimeError(
                "To evaluate pt_coeffs, interpolation must be activated at instantiation using setup_interp=True"
            )
        evalE = numpy.atleast_1d(E)
        indxc = (evalE >= self._Emin) * (evalE <= self._Emax)
        coords = self._coords_for_map_coords_pt(evalE[indxc], deriv=False)
        out = numpy.empty((len(evalE), self._nptcoeffs))
        out[indxc] = numpy.reshape(
            ndimage.map_coordinates(
                self._ptcoeffsFiltered, coords, order=3, prefilter=False
            ),
            (numpy.sum(indxc), self._nptcoeffs),
        )
        out[True ^ indxc] = numpy.nan
        return out

    def pt_deriv_coeffs(self, E):
        if self._momentum_matched:
            raise RuntimeError(
                "pt_deriv_coeffs is part of the older evaluation and is not available for momentum_matched=True"
            )
        if not self._interp:
            raise RuntimeError(
                "To evaluate pt_deriv_coeffs, interpolation must be activated at instantiation using setup_interp=True"
            )
        evalE = numpy.atleast_1d(E)
        indxc = (evalE >= self._Emin) * (evalE <= self._Emax)
        coords = self._coords_for_map_coords_pt(evalE[indxc], deriv=True)
        out = numpy.empty((len(evalE), self._nptcoeffs - 1))
        out[indxc] = numpy.reshape(
            ndimage.map_coordinates(
                self._ptderivcoeffsFiltered, coords, order=3, prefilter=False
            ),
            (numpy.sum(indxc), self._nptcoeffs - 1),
        )
        out[True ^ indxc] = numpy.nan
        return out

    def _plot_interp_mm(self, E):
        # The interpolated family at E against the torus built on its own (a
        # one-torus family): map coefficients, storage variable, and orbit
        truthaAV = actionAngleVerticalInverse(
            pot=self._pot,
            Es=[E],
            nta=self._nta,
            setup_interp=False,
            mm_npt=self._mm_npt,
            mm_nta=self._mm_nta,
        )
        J = float(self.J(E))
        D, _, K, _ = self._mm_tables(J)
        ms = 2.0 * numpy.arange(1, self._mm_npt + 1)
        pyplot.subplot(1, 3, 1)
        # absolute differences of the coefficients, which are themselves at
        # round-off beyond the first few tens of harmonics, and the relative
        # difference of the storage variable K
        y = numpy.fabs(D - truthaAV._mm_D[0])
        dK = numpy.fabs(K / truthaAV._mm_K[0] - 1.0)
        plot.plot(
            ms,
            y,
            color="k",
            gcf=True,
            semilogy=True,
            xrange=[0.0, ms[-1] + 1.0],
            yrange=[
                numpy.amax([numpy.amin(numpy.append(y, dK)), 1e-17]) / 3.0,
                numpy.amax([numpy.amax(numpy.append(y, dK)), 1e-16]) * 3.0,
            ],
            xlabel=r"$m$",
            ylabel=r"$|\Delta D_m|$ (solid), $|\Delta K/K|$ (dashed)",
        )
        pyplot.axhline(dK, color="k", ls="--")
        ta = numpy.linspace(0.0, 2.0 * numpy.pi, 1001)
        x, v = self(J, ta)
        xt, vt = truthaAV(truthaAV._js[0], ta)
        pyplot.subplot(1, 3, 2)
        plot.plot(
            ta,
            x - xt,
            color="k",
            gcf=True,
            xrange=[0.0, 2.0 * numpy.pi],
            xlabel=r"$\theta$",
            ylabel=r"$x^{\mathrm{interp}} - x^{\mathrm{truth}}$",
        )
        pyplot.subplot(1, 3, 3)
        plot.plot(
            ta,
            v - vt,
            color="k",
            gcf=True,
            xrange=[0.0, 2.0 * numpy.pi],
            xlabel=r"$\theta$",
            ylabel=r"$v^{\mathrm{interp}} - v^{\mathrm{truth}}$",
        )
        pyplot.tight_layout()
        return None

    def plot_interp(self, E, symm=True):
        if self._momentum_matched:
            return self._plot_interp_mm(E)
        truthaAV = actionAngleVerticalInverse(
            pot=self._pot,
            Es=[E],
            nta=self._nta,
            setup_interp=False,
            use_pointtransform="exact" if self._pt_exact else self._pt_deg > 1,
            pt_deg=self._pt_deg,
            pt_nxa=self._pt_nxa,
            exact_pt_spl_deg=(self._exact_pt_spl_deg if self._pt_exact else 5),
        )
        # Check whether S_n is matched
        pyplot.subplot(2, 3, 1)
        y = numpy.fabs(self.nSn(E)[0, symm :: symm + 1])
        ymin = numpy.amax([numpy.amin(y[numpy.isfinite(y)]), 1e-17])
        ymax = numpy.amax(y[numpy.isfinite(y)])
        plot.plot(
            numpy.fabs(self._nforSn[symm :: symm + 1]),
            y,
            yrange=[ymin, ymax],
            gcf=True,
            semilogy=True,
            xrange=[0.0, self._nforSn[-1]],
            label=r"$\mathrm{Interpolation}$",
            xlabel=r"$n$",
            ylabel=r"$|nS_n|$",
        )
        plot.plot(
            self._nforSn[symm :: symm + 1],
            truthaAV._nSn[0, symm :: symm + 1],
            overplot=True,
            label=r"$\mathrm{Direct}$",
        )
        pyplot.legend(fontsize=17.0, frameon=False)
        pyplot.subplot(2, 3, 4)
        y = ((self.nSn(E)[0] - truthaAV._nSn[0]) / truthaAV._nSn[0])[symm :: symm + 1]
        ymin = numpy.amin(y[numpy.isfinite(y)])
        ymax = numpy.amax(y[numpy.isfinite(y)])
        plot.plot(
            self._nforSn[symm :: symm + 1],
            y,
            yrange=[ymin, ymax],
            xrange=[0.0, self._nforSn[-1]],
            gcf=True,
            xlabel=r"$n$",
            ylabel=r"$S_{n,\mathrm{interp}}/S_{n,\mathrm{direct}}-1$",
        )
        # Check whether d S_n / d J is matched
        pyplot.subplot(2, 3, 2)
        y = numpy.fabs(self.dSndJ(E)[0, symm :: symm + 1])
        ymin = numpy.amax([numpy.amin(y[numpy.isfinite(y)]), 1e-18])
        ymax = numpy.amax(y[numpy.isfinite(y)])
        plot.plot(
            numpy.fabs(self._nforSn[symm :: symm + 1]),
            y,
            yrange=[ymin, ymax],
            xrange=[0.0, self._nforSn[-1]],
            gcf=True,
            semilogy=True,
            label=r"$\mathrm{Interpolation}$",
            xlabel=r"$n$",
            ylabel=r"$|\mathrm{d}S_n/\mathrm{d}J|$",
        )
        plot.plot(
            self._nforSn[symm :: symm + 1],
            numpy.fabs(truthaAV._dSndJ[0, symm :: symm + 1]),
            overplot=True,
            label=r"$\mathrm{Direct}$",
        )
        pyplot.legend(fontsize=17.0, frameon=False)
        pyplot.subplot(2, 3, 5)
        y = ((self.dSndJ(E)[0] - truthaAV._dSndJ[0]) / truthaAV._dSndJ[0])[
            symm :: symm + 1
        ]
        ymin = numpy.amin(y[numpy.isfinite(y)])
        ymax = numpy.amax(y[numpy.isfinite(y)])
        plot.plot(
            self._nforSn[symm :: symm + 1],
            y,
            yrange=[ymin, ymax],
            xrange=[0.0, self._nforSn[-1]],
            gcf=True,
            xlabel=r"$n$",
            ylabel=r"$(\mathrm{d}S_n/\mathrm{d}J)_{\mathrm{interp}}/(\mathrm{d}S_n/\mathrm{d}J)_{\mathrm{direct}}-1$",
        )
        # Check energy along the torus
        pyplot.subplot(2, 3, 3)
        ta = numpy.linspace(0.0, 2.0 * numpy.pi, 1001)
        # J(E) is the torus's label, which the public evaluator looks up
        # (with a polynomial point transformation the internal action _js
        # differs from it)
        x, v = truthaAV(truthaAV.J(E), ta)
        Edirect = v**2.0 / 2.0 + evaluatelinearPotentials(
            self._pot, x, use_physical=False
        )
        x, v = self(self.J(E), ta)
        Einterp = v**2.0 / 2.0 + evaluatelinearPotentials(
            self._pot, x, use_physical=False
        )
        ymin, ymax = numpy.amin([Edirect, Einterp]), numpy.amax([Edirect, Einterp])

        plot.plot(
            ta,
            Einterp,
            xrange=[0.0, 2.0 * numpy.pi],
            yrange=[ymin - (ymax - ymin) * 2.0, ymax + (ymax - ymin) * 2.0],
            gcf=True,
            label=r"$\mathrm{Interpolation}$",
            xlabel=r"$\theta$",
            ylabel=r"$E$",
        )
        plot.plot(ta, Edirect, overplot=True, label=r"$\mathrm{Direct}$")
        pyplot.legend(fontsize=17.0, frameon=False)
        pyplot.subplot(2, 3, 6)
        plot.plot(
            ta,
            Einterp / Edirect - 1.0,
            xrange=[0.0, 2.0 * numpy.pi],
            gcf=True,
            label=r"$\mathrm{Interpolation}$",
            xlabel=r"$\theta$",
            ylabel=r"$E_{\mathrm{interp}}/E_{\mathrm{direct}}-1$",
        )
        pyplot.tight_layout()
        return None

    def J(self, E):
        """
        Return the action for the given energy.

        Parameters
        ----------
        E : float
            Energy.

        Returns
        -------
        float
            Action.

        Notes
        -----
        - 2022-11-24 - Written - Bovy (UofT)

        """
        indx = numpy.nanargmin(numpy.fabs(E - self._Es))
        if numpy.fabs(E - self._Es[indx]) > 1e-10:
            raise ValueError(
                "Given energy not found; please specify an energy used in the initialization of the instance"
            )
        return self._js[indx]

    def _evaluate(self, j, angle, **kwargs):
        """
        Evaluate the phase-space coordinates (x,v) for a number of angles on a single torus

        Parameters
        ----------
        j : float
            Action
        angle : numpy.ndarray
            Angle
        Returns
        -------
        tuple
            Tuple containing the phase-space coordinates [x,vx]

        Notes
        -----
        - 2018-04-08 - Written - Bovy (UofT)

        """
        return self._xvFreqs(j, angle, **kwargs)[:2]

    def _mm_xvFreqs(self, j, angle):
        """
        The momentum-matched evaluation, in the form the public interface
        wants: position, velocity, and frequency.

        For H = p^2/2 + Phi the momentum is the velocity, and the frequency
        is dE/dJ, taken from the same interpolant the map reads rather than
        from a separate table.

        Parameters
        ----------
        j : float
            Action.
        angle : numpy.ndarray
            Angle.

        Returns
        -------
        tuple
            (x, v, frequency).

        Notes
        -----
        - 2026-08-29 - Written - Bovy (UofT)
        """
        # a single torus at a time: J(E) from the interpolant is a length-one array
        j = float(numpy.asarray(j, dtype="float").item())
        x, p = self._mm_xp_of_angle(j, angle)
        return x, p, float(self._mm_dEdj(j))

    def _xvFreqs(self, j, angle, **kwargs):
        """
        Evaluate the phase-space coordinates (x,v) for a number of angles on a single torus as well as the frequency.

        Parameters
        ----------
        j : float
            Action.
        angle : numpy.ndarray
            Angle.

        Returns
        -------
        tuple
            (x,v,frequency)

        Notes
        -----
        - 2018-04-15 - Written - Bovy (UofT)
        """
        if self._momentum_matched:
            # the canonical map replaces the evaluation entirely; there is
            # no fallback path through the old correspondence
            return self._mm_xvFreqs(j, angle)
        # Find torus
        if not self._interp:
            indx = numpy.nanargmin(numpy.fabs(j - self._js))
            if numpy.fabs(j - self._js[indx]) > 1e-10:
                raise ValueError(
                    "Given action/energy not found, to use interpolation, initialize with setup_interp=True"
                )
            tjaoffset = self._jaoffset[indx]
            tnSn = self._nSn[indx]
            tdSndJ = self._dSndJ[indx]
            tOmegaHO = self._OmegaHO[indx]
            tOmega = self._Omegas[indx]
            txmax = self._xmaxs[indx]
            tptxmax = self._pt_xmaxs[indx]
            tptcoeffs = self._pt_coeffs[indx]
            tptderivcoeffs = self._pt_deriv_coeffs[indx]
        else:
            tE = self.E(j)
            tjaoffset = float(self.jaoffset(tE))
            tnSn = self.nSn(tE)[0]
            tdSndJ = self.dSndJ(tE)[0]
            tOmegaHO = self.OmegaHO(tE)
            tOmega = self.Omega(tE)
            txmax = self.xmax(tE)
            tptxmax = self.ptxmax(tE)
            tptcoeffs = self.pt_coeffs(tE)[0]
            tptderivcoeffs = self.pt_deriv_coeffs(tE)[0]
        if self._pt_exact and self._pt_only:
            # For the exact point transformation, the generating-function
            # mapping (J,theta) -> (JA,thetaA) is the identity, so we can
            # skip solving for the auxiliary angles and action
            angle = numpy.atleast_1d(angle)
            anglea = copy.copy(angle)
            ja = (j + tjaoffset) * numpy.ones_like(angle)
        else:
            # First we need to solve for a<nglea
            angle = numpy.atleast_1d(angle)
            anglea = copy.copy(angle)
            # Now iterate Newton's method
            cntr = 0
            unconv = numpy.ones(len(angle), dtype="bool")
            ta = anglea + 2.0 * numpy.sum(
                tdSndJ * numpy.sin(self._nforSn * numpy.atleast_2d(anglea).T), axis=1
            )
            dta = (ta - angle + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
            unconv[unconv] = numpy.fabs(dta) > self._angle_tol
            # Don't allow too big steps
            maxda = 2.0 * numpy.pi / 101
            while not self._bisect:
                danglea = 1.0 + 2.0 * numpy.sum(
                    self._nforSn
                    * tdSndJ
                    * numpy.cos(self._nforSn * numpy.atleast_2d(anglea[unconv]).T),
                    axis=1,
                )
                dta = (ta[unconv] - angle[unconv] + numpy.pi) % (
                    2.0 * numpy.pi
                ) - numpy.pi
                da = -dta / danglea
                da[numpy.fabs(da) > maxda] = (numpy.sign(da) * maxda)[
                    numpy.fabs(da) > maxda
                ]
                anglea[unconv] += da
                unconv[unconv] = numpy.fabs(dta) > self._angle_tol
                newta = anglea[unconv] + 2.0 * numpy.sum(
                    tdSndJ
                    * numpy.sin(self._nforSn * numpy.atleast_2d(anglea[unconv]).T),
                    axis=1,
                )
                ta[unconv] = newta
                cntr += 1
                if numpy.sum(unconv) == 0:
                    break
                if cntr > self._maxiter:  # pragma: no cover
                    warnings.warn(
                        "Angle mapping with Newton-Raphson did not converge in {} iterations, falling back onto simple bisection (increase maxiter to try harder with Newton-Raphson)".format(
                            self._maxiter
                        ),
                        galpyWarning,
                    )
                    break
            # Fallback onto simple bisection in case of non-convergence
            if self._bisect or cntr > self._maxiter:
                # Reset cntr
                cntr = 0
                trya_min = numpy.zeros(numpy.sum(unconv))
                da = 2.0 * numpy.pi
                while True:
                    da *= 0.5
                    anglea[unconv] = trya_min + da
                    newta = (
                        anglea[unconv]
                        + 2.0
                        * numpy.sum(
                            tdSndJ
                            * numpy.sin(
                                self._nforSn * numpy.atleast_2d(anglea[unconv]).T
                            ),
                            axis=1,
                        )
                        + 2.0 * numpy.pi
                    ) % (2.0 * numpy.pi)
                    dta = (newta - angle[unconv] + numpy.pi) % (
                        2.0 * numpy.pi
                    ) - numpy.pi
                    trya_min[newta < angle[unconv]] = anglea[unconv][
                        newta < angle[unconv]
                    ]
                    unconv[unconv] = numpy.fabs(dta) > self._angle_tol
                    trya_min = trya_min[numpy.fabs(dta) > self._angle_tol]
                    cntr += 1
                    if numpy.sum(unconv) == 0:
                        break
                    if cntr > self._maxiter:  # pragma: no cover
                        warnings.warn(
                            "Angle mapping with bisection did not converge in {} iterations".format(
                                self._maxiter
                            )
                            + " for angles:"
                            + "".join(f" {k:g}" for k in sorted(set(angle[unconv]))),
                            galpyWarning,
                        )
                        break
            # Then compute the auxiliary action
            # the point-transformed action J^A = J + offset is the base of
            # the Fourier structure
            ja = (j + tjaoffset) + 2.0 * numpy.sum(
                tnSn * numpy.cos(self._nforSn * numpy.atleast_2d(anglea).T), axis=1
            )
        hoaainv = actionAngleHarmonicInverse(omega=tOmegaHO)
        xa, va = hoaainv(ja, anglea)
        if self._pt_exact:
            # Row coordinate of this torus in the grid of tori; fractional
            # for interpolated tori, in which case the 2D spline evaluation
            # interpolates the point transformation between the grid tori
            trowcoord = (
                float(indx)
                if not self._interp
                else float(
                    (tE - self._Emin) / (self._Emax - self._Emin) * (self._nE - 1.0)
                )
            )
            x = txmax * _ptxa_eval(
                xa / tptxmax,
                trowcoord,
                self._pt_filtered[0],
                self._pt_nmesh,
                self._exact_pt_spl_deg,
            )
            v = (
                va
                / tptxmax
                * txmax
                * _ptxa_eval(
                    xa / tptxmax,
                    trowcoord,
                    self._pt_filtered[1],
                    self._pt_nmesh,
                    self._exact_pt_spl_deg,
                )
            )
        else:
            x = (
                txmax
                * polynomial.polyval((xa / tptxmax).T, tptcoeffs.T, tensor=False).T
            )
            v = (
                va
                / tptxmax
                * txmax
                * polynomial.polyval((xa / tptxmax).T, tptderivcoeffs.T, tensor=False).T
            )
        return (x, v, tOmega)

    def _Freqs(self, j, **kwargs):
        """
        Return the frequency corresponding to a torus

        Parameters
        ----------
        j : float
            Action.

        Returns
        -------
        float
            Frequency corresponding to a torus.

        Notes
        -----
        - 2018-04-08 - Written - Bovy (UofT)

        """
        # Find torus
        if self._momentum_matched:
            # The map's own frequency: dE/dJ of the Hermite energy
            # interpolant, which is exactly the frequency of the (x, v)
            # trajectories the map returns (_xvFreqs), so the two public
            # answers agree exactly.  The frequency table is marginally
            # (~4e-10) closer to the isolated true frequency between its
            # nodes, but an answer inconsistent with the returned orbits
            # is the wrong kind of accurate.
            return float(self._mm_dEdj(float(numpy.asarray(j, dtype="float").item())))
        if not self._interp:
            indx = numpy.nanargmin(numpy.fabs(j - self._js))
            if numpy.fabs(j - self._js[indx]) > 1e-10:
                raise ValueError(
                    "Given action/energy not found, to use interpolation, initialize with setup_interp=True"
                )
            tOmega = self._Omegas[indx]
        else:
            tE = self.E(j)
            tOmega = self.Omega(tE)
        return tOmega


def _ptxa_eval(xanorm, rowcoord, pt_filtered_arr, pt_nmesh, pt_spl_deg):
    """
    Evaluate the exact point transformation (or one of its derivatives) with
    respect to the normalized coordinate xa/ptxmax, using 2D spline
    interpolation of the (torus,mesh) grid on which it is stored

    Parameters
    ----------
    xanorm : numpy.ndarray
        Normalized position(s) xa/ptxmax at which to evaluate.
    rowcoord : numpy.ndarray or float
        (Possibly fractional, for tori obtained through interpolation) row
        index of the torus of each evaluation point in the grid of tori;
        scalars are broadcast against xanorm.
    pt_filtered_arr : numpy.ndarray
        Spline-filtered (torus,mesh) grid of the normalized mapping x/xmax
        (or of one of its derivatives with respect to the normalized
        coordinate) sampled on the fixed mesh.
    pt_nmesh : int
        Number of mesh points (the size of pt_filtered_arr's second
        dimension).
    pt_spl_deg : int
        Degree of the interpolating spline (must match the order used to
        filter pt_filtered_arr).

    Returns
    -------
    numpy.ndarray
        The normalized mapping (or its derivative) at xanorm.

    Notes
    -----
    - 2026-08-13 - Written - Bovy (UofT)

    """
    xanorm = numpy.atleast_1d(numpy.asarray(xanorm, dtype="float"))
    rowcoord = numpy.broadcast_to(numpy.asarray(rowcoord, dtype="float"), xanorm.shape)
    meshcoord = (xanorm + 1.0) * (pt_nmesh - 1.0) / 2.0 + (
        pt_filtered_arr.shape[1] - pt_nmesh
    ) / 2.0
    return ndimage.map_coordinates(
        pt_filtered_arr,
        [rowcoord.reshape(-1), meshcoord.reshape(-1)],
        order=pt_spl_deg,
        prefilter=False,
        mode="mirror",
    ).reshape(xanorm.shape)


def _anglea(
    xa,
    E,
    pot,
    omega,
    ptcoeffs,
    ptderivcoeffs,
    xmax,
    ptxmax,
    vsign=1.0,
    pt_exact=False,
    pt_filtered=None,
    pt_nmesh=None,
    pt_spl_deg=5,
):
    """
    Compute the auxiliary angle in the harmonic-oscillator for a grid in x and E

    Parameters
    ----------
    xa : numpy.ndarray
        Position.
    E : float
        Energy.
    pot : Potential object
        The potential.
    omega : numpy.ndarray
        Harmonic-oscillator frequencies.
    ptcoeffs : numpy.ndarray
        Coefficients of the polynomial point transformation.
    ptderivcoeffs : numpy.ndarray
        Coefficients of the derivative of the polynomial point transformation.
    xmax : float
        Xmax of the true torus.
    ptxmax : float
        Xmax of the point-transformed torus.
    vsign : float, optional
        Sign of the velocity. Default is 1.0.

    Returns
    -------
    numpy.ndarray
        Auxiliary angles.

    Notes
    -----
    - 2018-04-13 - Written - Bovy (UofT)
    - 2018-11-19 - Added point transformation - Bovy (UofT)

    """
    # Compute v
    if pt_exact:
        x = xmax * _ptxa_eval(
            xa / ptxmax, ptcoeffs, pt_filtered[0], pt_nmesh, pt_spl_deg
        )
    else:
        x = xmax * polynomial.polyval((xa / ptxmax).T, ptcoeffs.T, tensor=False).T
    v2 = 2.0 * (E - evaluatelinearPotentials(pot, x, use_physical=False))
    v2[v2 < 0] = 0.0
    v2[numpy.fabs(xa) == ptxmax] = 0.0  # just in case the pt mapping has small issues
    if pt_exact:
        piprime = (
            xmax
            / ptxmax
            * _ptxa_eval(xa / ptxmax, ptcoeffs, pt_filtered[1], pt_nmesh, pt_spl_deg)
        )
    else:
        piprime = (
            xmax
            / ptxmax
            * polynomial.polyval((xa / ptxmax).T, ptderivcoeffs.T, tensor=False).T
        )
    # J=0 special case:
    zindx = (xmax == 0.0) * (ptxmax == xmax + 1e-10)
    if numpy.any(zindx):
        if pt_exact:
            piprime[zindx] = _ptxa_eval(
                xa[zindx] / ptxmax[zindx],
                numpy.asarray(ptcoeffs)[zindx]
                if numpy.ndim(ptcoeffs) > 0
                else ptcoeffs,
                pt_filtered[1],
                pt_nmesh,
                pt_spl_deg,
            )
        else:
            piprime[zindx] = polynomial.polyval(
                (xa[zindx] / ptxmax[zindx]).T,
                ptderivcoeffs[zindx].T,
                tensor=False,
            ).T
    return numpy.arctan2(omega * xa, vsign * numpy.sqrt(v2) / piprime)


def _danglea(
    xa,
    E,
    pot,
    omega,
    ptcoeffs,
    ptderivcoeffs,
    ptderiv2coeffs,
    xmax,
    ptxmax,
    vsign=1.0,
    pt_exact=False,
    pt_filtered=None,
    pt_nmesh=None,
    pt_spl_deg=5,
):
    """
    Compute the derivative of the auxiliary angle in the harmonic-oscillator for a grid in x and E at constant E

    Parameters
    ----------
    xa : numpy.ndarray
        Position.
    E : float
        Energy.
    pot : Potential object
        The potential.
    omega : numpy.ndarray
        Harmonic-oscillator frequencies.
    ptcoeffs : numpy.ndarray
        Coefficients of the polynomial point transformation.
    ptderivcoeffs : numpy.ndarray
        Coefficients of the derivative of the polynomial point transformation.
    ptderiv2coeffs : numpy.ndarray
        Coefficients of the second derivative of the polynomial point transformation.
    xmax : float
        Xmax of the true torus.
    ptxmax : float
        Xmax of the point-transformed torus.
    vsign : float, optional
        Sign of the velocity. Default is 1.0.

    Returns
    -------
    numpy.ndarray
        d auxiliary angles / d x (2D array)

    Notes
    -----
    - 2018-04-13 - Written - Bovy (UofT)
    - 2018-11-22 - Added point transformation - Bovy (UofT)

    """
    # Compute v
    if pt_exact:
        x = xmax * _ptxa_eval(
            xa / ptxmax, ptcoeffs, pt_filtered[0], pt_nmesh, pt_spl_deg
        )
        piprime = (
            xmax
            / ptxmax
            * _ptxa_eval(xa / ptxmax, ptcoeffs, pt_filtered[1], pt_nmesh, pt_spl_deg)
        )
        piprime2 = (
            xmax
            / ptxmax**2.0
            * _ptxa_eval(xa / ptxmax, ptcoeffs, pt_filtered[2], pt_nmesh, pt_spl_deg)
        )
    else:
        x = xmax * polynomial.polyval((xa / ptxmax).T, ptcoeffs.T, tensor=False).T
        piprime = (
            xmax
            / ptxmax
            * polynomial.polyval((xa / ptxmax).T, ptderivcoeffs.T, tensor=False).T
        )
        piprime2 = (
            xmax
            / ptxmax**2.0
            * polynomial.polyval((xa / ptxmax).T, ptderiv2coeffs.T, tensor=False).T
        )
    v2 = 2.0 * (E - evaluatelinearPotentials(pot, x, use_physical=False))
    v2[v2 < 1e-15] = 1e-15
    anglea = numpy.arctan2(omega * xa * piprime, vsign * numpy.sqrt(v2))
    return (
        omega
        * numpy.cos(anglea) ** 2.0
        * v2**-1.5
        * (
            v2 * (piprime + xa * piprime2)
            - xa * evaluatelinearForces(pot, x, use_physical=False) * piprime**2.0
        )
    )


def _ja(
    xa,
    E,
    pot,
    omega,
    ptcoeffs,
    ptderivcoeffs,
    xmax,
    ptxmax,
    pt_exact=False,
    pt_filtered=None,
    pt_nmesh=None,
    pt_spl_deg=5,
):
    """
    Compute the auxiliary action in the harmonic-oscillator for a grid in x and E

    Parameters
    ----------
    xa : numpy.ndarray
        position
    E : numpy.ndarray
        Energy
    pot : Potential object
        the potential
    omega : numpy.ndarray
        harmonic-oscillator frequencies
    ptcoeffs : numpy.ndarray
        coefficients of the polynomial point transformation
    ptderivcoeffs : numpy.ndarray
        coefficients of the derivative of the polynomial point transformation
    xmax : float
        xmax of the true torus
    ptxmax : float
        xmax of the point-transformed torus

    Returns
    -------
    numpy.ndarray
        auxiliary actions

    Notes
    -----
    - 2018-04-14 - Written - Bovy (UofT)
    - 2018-11-22 - Added point transformation - Bovy (UofT)

    """
    if pt_exact:
        x = xmax * _ptxa_eval(
            xa / ptxmax, ptcoeffs, pt_filtered[0], pt_nmesh, pt_spl_deg
        )
        piprime = (
            xmax
            / ptxmax
            * _ptxa_eval(xa / ptxmax, ptcoeffs, pt_filtered[1], pt_nmesh, pt_spl_deg)
        )
    else:
        x = xmax * polynomial.polyval((xa / ptxmax).T, ptcoeffs.T, tensor=False).T
        piprime = (
            xmax
            / ptxmax
            * polynomial.polyval((xa / ptxmax).T, ptderivcoeffs.T, tensor=False).T
        )
    v2over2 = E - evaluatelinearPotentials(pot, x, use_physical=False)
    v2over2[v2over2 < 0.0] = 0.0
    out = numpy.empty_like(xa)
    gIndx = True ^ ((xmax == 0.0) * (ptxmax == xmax + 1e-10))
    out[gIndx] = (
        v2over2[gIndx] / omega[gIndx] / piprime[gIndx] ** 2.0
        + omega[gIndx] * xa[gIndx] ** 2.0 / 2.0
    )
    # J=0 special case
    out[True ^ gIndx] = 0.0
    return out


def _djadj(
    xa,
    E,
    pot,
    omega,
    ptcoeffs,
    ptderivcoeffs,
    ptderiv2coeffs,
    xmax,
    ptxmax,
    pt_exact=False,
    pt_filtered=None,
    pt_nmesh=None,
    pt_spl_deg=5,
):
    """
    Compute the derivative of the auxiliary action in the harmonic-oscillator wrt the action for a grid in x and E

    Parameters
    ----------
    xa : numpy.ndarray
        position
    E : float
        Energy
    pot : galpy.potential.Potential
        the potential
    omega : numpy.ndarray
        harmonic-oscillator frequencies
    ptcoeffs : numpy.ndarray
        coefficients of the polynomial point transformation
    ptderivcoeffs : numpy.ndarray
        coefficients of the derivative of the polynomial point transformation
    ptderiv2coeffs : numpy.ndarray
        coefficients of the second derivative of the polynomial point transformation
    xmax : float
        xmax of the true torus
    ptxmax : float
        xmax of the point-transformed torus

    Returns
    -------
    numpy.ndarray
        d(auxiliary actions)/d(action)

    Notes
    -----
    - 2018-04-14 - Written - Bovy (UofT)
    - 2018-11-23 - Added point transformation - Bovy (UofT)
    """
    if pt_exact:
        x = xmax * _ptxa_eval(
            xa / ptxmax, ptcoeffs, pt_filtered[0], pt_nmesh, pt_spl_deg
        )
        piprime = (
            xmax
            / ptxmax
            * _ptxa_eval(xa / ptxmax, ptcoeffs, pt_filtered[1], pt_nmesh, pt_spl_deg)
        )
        piprime2 = (
            xmax
            / ptxmax**2.0
            * _ptxa_eval(xa / ptxmax, ptcoeffs, pt_filtered[2], pt_nmesh, pt_spl_deg)
        )
    else:
        x = xmax * polynomial.polyval((xa / ptxmax).T, ptcoeffs.T, tensor=False).T
        piprime = (
            xmax
            / ptxmax
            * polynomial.polyval((xa / ptxmax).T, ptderivcoeffs.T, tensor=False).T
        )
        piprime2 = (
            xmax
            / ptxmax**2.0
            * polynomial.polyval((xa / ptxmax).T, ptderiv2coeffs.T, tensor=False).T
        )
    v2 = 2.0 * (E - evaluatelinearPotentials(pot, x, use_physical=False))
    # J=0 special case:
    zindx = (xmax == 0.0) * (ptxmax == xmax + 1e-10)
    if numpy.any(zindx):
        if pt_exact:
            piprime[zindx] = _ptxa_eval(
                x[zindx] / ptxmax[zindx],
                numpy.asarray(ptcoeffs)[zindx]
                if numpy.ndim(ptcoeffs) > 0
                else ptcoeffs,
                pt_filtered[1],
                pt_nmesh,
                pt_spl_deg,
            )
        else:
            piprime[zindx] = polynomial.polyval(
                (x[zindx] / ptxmax[zindx]).T,
                ptderivcoeffs[zindx].T,
                tensor=False,
            ).T
    gIndx = True ^ ((xmax == 0.0) * (ptxmax == xmax + 1e-10))
    dxAdE = numpy.empty_like(xa)
    dxAdE[gIndx] = (
        xa[gIndx]
        * piprime[gIndx] ** 2.0
        / (
            v2[gIndx] * (1.0 + piprime2[gIndx] / piprime[gIndx] * xa[gIndx])
            - xa[gIndx]
            * evaluatelinearForces(pot, x[gIndx], use_physical=False)
            * piprime[gIndx]
        )
    )
    dxAdE[(xmax == 0.0) * (ptxmax == xmax + 1e-10)] = 1.0
    return (
        1.0
        + (
            evaluatelinearForces(pot, x, use_physical=False) / piprime
            + omega**2.0 * xa
            - piprime**-3.0 * piprime2 * v2
        )
        * dxAdE
    )
