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
        canonical=None,
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
            number of auxiliary angles to sample the torus at when mapping the torus
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
        canonical : bool, optional
            if True, use the canonical evaluation mode: the angle map uses the exact derivative of the interpolated nSn tables plus the closed-form compensation for the energy-dependent auxiliary frequency, so the assembled (J, angle) -> (x, v) map is exactly symplectic for any stored tables (the separately stored dSndJ tables are kept only as a consistency diagnostic). Default: True for interpolating instances without a point transformation (where the stored coefficients are gauge-compatible), False otherwise; canonical=True with a point transformation is not implemented yet.

        Notes
        -----
        - 2018-04-11 - Started - Bovy (UofT)
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
        self._nta = nta
        self._thetaa = numpy.linspace(0.0, 2.0 * numpy.pi * (1.0 - 1.0 / nta), nta)
        self._maxiter = maxiter
        self._angle_tol = angle_tol
        self._bisect = bisect
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
        # Store mean(ja), this is only a better approx. of j w/ no PT!
        self._js_orig = copy.copy(self._js)
        self._js = numpy.nanmean(self._ja, axis=1)
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
        # Should use sqrt(2nd deriv. pot), but currently not implemented for 1D
        if self._nE > 1:
            self._OmegaHO[self._Es < 1e-10] = self._OmegaHO[1]
            self._Omegas[self._Es < 1e-10] = self._Omegas[1]
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
        # Canonical evaluation: manifestly symplectic for any stored tables.
        # Default for interpolating instances without a point transformation,
        # where the stored nSn are the coefficients of the xi = 0 gauge (the
        # identity PT has no shear, so the equal-time and cotangent gauges
        # coincide and the tables can be reused as-is)
        self._identity_pt = not self._pt_exact and self._pt_deg == 1
        if canonical is None:
            canonical = self._interp and not self._pt_exact and not self._pt_only
        if canonical and not self._interp:
            raise ValueError("canonical=True requires setup_interp=True")
        if canonical and self._pt_exact:
            raise ValueError(
                'canonical=True is not implemented yet for use_pointtransform="exact"'
            )
        if canonical and self._pt_only:  # pragma: no cover
            raise ValueError(
                "canonical=True is incompatible with pt_only=True: pt_only "
                "is an equal-time-gauge construct and the canonical mode "
                "works in the shear-free gauge, where the exact point "
                "transformation's mapping coefficients do not vanish"
            )
        self._canonical = canonical
        if self._canonical:
            self._setup_canonical()
        return None

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
            self._js[indx] + shift_action * (self._js_orig[indx] - self._js[indx]),
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

    def plot_power(self, Es, symm=True, overplot=False, return_gridspec=False, ls="-"):
        Es = numpy.sort(numpy.atleast_1d(Es))
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
            y = numpy.fabs(self._nSn[indx, symm :: symm + 1])
            if len(Es) > 1 and E == Es[0]:
                y4minmax = numpy.fabs(self._nSn[:, symm :: symm + 1])
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
                numpy.fabs(self._nforSn[symm :: symm + 1]),
                y,
                yrange=[ymin, ymax],
                ls=ls,
                gcf=True,
                semilogy=True,
                overplot=overplot,
                xrange=[0.0, self._nforSn[-1]],
                label=label,
                color=color,
                xlabel=r"$n$",
                ylabel=r"$|nS_n|$",
            )
            # d S_n / d J
            y = numpy.fabs(self._dSndJ[indx, symm :: symm + 1])
            if len(Es) > 1 and E == Es[0]:
                y4minmax = numpy.fabs(self._dSndJ[:, symm :: symm + 1])
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
                numpy.fabs(self._nforSn[symm :: symm + 1]),
                y,
                yrange=[ymin, ymax],
                ls=ls,
                gcf=True,
                semilogy=True,
                overplot=overplot,
                xrange=[0.0, self._nforSn[-1]],
                label=label,
                color=color,
                xlabel=r"$n$",
                ylabel=r"$|\mathrm{d}S_n/\mathrm{d}J|$",
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

    def plot_interp(self, E, symm=True):
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
        x, v = truthaAV(truthaAV._js, ta)
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

    def _setup_canonical(self):
        """Build the exactly-differentiable chains of the canonical mode.

        Everything the evaluation uses is either an interpolant paired with
        its own exact derivative or a closed form. The mapping coefficients
        are recomputed here in the shear-free (cotangent-lift) gauge by
        direct sampling of each torus -- x_a sampled on the auxiliary loop,
        x = pi(x_a), p_a = v pi'(x_a), so no inversion of the point
        transformation is ever needed -- and the canonical action labels are
        the zero modes <J^A>, which are gauge-independent (the loop action).
        For the identity point transformation this reproduces the stored
        nSn to quadrature accuracy; for a polynomial point transformation
        the equal-time-gauge nSn are NOT gauge-compatible and the
        recomputation is essential. The separately stored dSndJ tables are
        never used by the canonical evaluation: a value table and an
        independent derivative table only satisfy the symplectic
        consistency relation when both are exact, and interpolation breaks
        it (the measured symplectic defect of the old mode is at the
        derivative-table interpolation error)."""
        ntau = 2 * self._nta
        tau = 2.0 * numpy.pi * (numpy.arange(ntau) + 0.5) / ntau
        nn = self._nforSn
        can_nSn = numpy.empty_like(self._nSn)
        can_js = numpy.empty(self._nE)
        for ii in range(self._nE):
            if self._js[ii] < 1e-10:
                can_nSn[ii] = 0.0
                can_js[ii] = 0.0
                continue
            xa = self._pt_xmaxs[ii] * numpy.sin(tau)
            x = self._xmaxs[ii] * polynomial.polyval(
                xa / self._pt_xmaxs[ii], self._pt_coeffs[ii]
            )
            piprime = (
                self._xmaxs[ii]
                / self._pt_xmaxs[ii]
                * polynomial.polyval(xa / self._pt_xmaxs[ii], self._pt_deriv_coeffs[ii])
            )
            v2 = 2.0 * (
                self._Es[ii]
                - evaluatelinearPotentials(self._pot, x, use_physical=False)
            )
            v2[v2 < 0.0] = 0.0
            pa = numpy.sign(numpy.cos(tau)) * numpy.sqrt(v2) * piprime
            om = self._OmegaHO[ii]
            jA = 0.5 * (pa**2 + om**2 * xa**2) / om
            thetaA = numpy.arctan2(om * xa, pa)
            # Fourier data by the pullback: the tau-grid is regular, theta^A
            # is not; spectral derivative of the periodic part supplies the
            # measure and sigma follows from one spectral antiderivative
            P = numpy.unwrap(thetaA - tau + numpy.pi) - numpy.pi
            k = numpy.fft.fftfreq(ntau, d=1.0 / ntau)
            dthetaAdtau = 1.0 + numpy.real(numpy.fft.ifft(1j * k * numpy.fft.fft(P)))
            can_js[ii] = numpy.mean(jA * dthetaAdtau)
            g = (jA - can_js[ii]) * dthetaAdtau
            gh = numpy.fft.fft(g)
            sh = numpy.zeros(ntau, dtype=complex)
            sh[1:] = gh[1:] / (1j * k[1:])
            sigma = numpy.real(numpy.fft.ifft(sh))
            for q, n in enumerate(nn):
                c = numpy.mean(sigma * numpy.exp(-1j * n * thetaA) * dthetaAdtau)
                can_nSn[ii, q] = -n * numpy.imag(c)
        self._can_nSn = can_nSn
        self._can_js = can_js
        self._can_nSn_c = ndimage.spline_filter1d(
            can_nSn, order=3, axis=0, mode="mirror"
        )
        self._can_E = interpolate.InterpolatedUnivariateSpline(can_js, self._Es, k=3)
        self._can_J = interpolate.InterpolatedUnivariateSpline(self._Es, can_js, k=3)
        self._can_dEdj = self._can_E.derivative()
        self._can_dOmHOdE = self.OmegaHO.derivative()
        # derivative chains of the point-transformation parameter functions,
        # for the PT compensation term (zero for the identity PT)
        self._can_xmax = interpolate.InterpolatedUnivariateSpline(
            self._Es, self._xmaxs, k=3
        )
        self._can_ptxmax = interpolate.InterpolatedUnivariateSpline(
            self._Es, self._pt_xmaxs, k=3
        )
        self._can_dxmaxdE = self._can_xmax.derivative()
        self._can_dptxmaxdE = self._can_ptxmax.derivative()
        self._can_ptcoeffs_c = ndimage.spline_filter1d(
            self._pt_coeffs, order=3, axis=0, mode="mirror"
        )
        # The canonical labels become the public ones: J(E) and E(j) must be
        # the same chains the evaluation differentiates, or callers mixing
        # J(E) with __call__ would straddle two gauges' labelings
        self.J = self._can_J
        self.E = self._can_E
        return None

    def _can_row(self, table_c, x):
        """Value and d/d(row) of a row-filtered table at fractional row x,
        by the four-point cubic B-spline stencil (mirror boundary)."""
        x = min(max(x, 0.0), self._nE - 1.0)
        i0 = int(numpy.floor(x))
        if i0 > self._nE - 2:  # pragma: no cover
            i0 = self._nE - 2
        t = x - i0
        taps = numpy.array([i0 - 1, i0, i0 + 1, i0 + 2])
        taps = numpy.abs(taps)
        taps[taps > self._nE - 1] = 2 * (self._nE - 1) - taps[taps > self._nE - 1]
        C = table_c[taps]
        w = numpy.array(
            [
                (1.0 - t) ** 3 / 6.0,
                (4.0 - 6.0 * t**2 + 3.0 * t**3) / 6.0,
                (1.0 + 3.0 * t + 3.0 * t**2 - 3.0 * t**3) / 6.0,
                t**3 / 6.0,
            ]
        )
        wd = numpy.array(
            [
                -((1.0 - t) ** 2) / 2.0,
                (-12.0 * t + 9.0 * t**2) / 6.0,
                (3.0 + 6.0 * t - 9.0 * t**2) / 6.0,
                t**2 / 2.0,
            ]
        )
        return w @ C, wd @ C

    def _canonical_tables(self, j):
        """nSn and d(nSn)/dj at action j, as value and exact derivative of
        one stored interpolant (cubic B-spline along the energy axis chained
        through the canonical E(j))."""
        tE = float(self._can_E(j))
        x = (tE - self._Emin) / (self._Emax - self._Emin) * (self._nE - 1.0)
        val, dval_dx = self._can_row(self._can_nSn_c, x)
        dval_dE = dval_dx * (self._nE - 1.0) / (self._Emax - self._Emin)
        return val, dval_dE * float(self._can_dEdj(j))

    def check_canonical_consistency(self, ntheta=256):
        """The three-way consistency relation between the two derivative
        routes (STAECKEL_CANONICAL_MATH.md C8): the stored dSndJ tables are
        the fixed-omega angle-fit fields, the canonical chain differentiates
        the value interpolant, and the two must differ by exactly the
        closed-form compensation for the varying auxiliary frequency,
        D_n = d(Sbar_n)/dj + [omega'/(2 omega)] g_n with g_n the sine
        coefficients of J^A(theta^A) sin(2 theta^A). Returns the maximum
        absolute mismatch over the interior nodes; a value at the
        table-interpolation error certifies both routes, a large value means
        a broken chain and localizes the offending node.

        Parameters
        ----------
        ntheta : int, optional
            Number of angles for the Fourier transform of the compensation.

        Returns
        -------
        float
            Maximum absolute mismatch of the consistency relation.

        Notes
        -----
        - 2026-08-25 - Written - Bovy (UofT)
        """
        if not self._identity_pt:
            raise RuntimeError(
                "check_canonical_consistency compares against the stored "
                "dSndJ tables, which live in the equal-time gauge; with a "
                "point transformation the gauges differ and the comparison "
                "is not meaningful"
            )
        th = 2.0 * numpy.pi * numpy.arange(ntheta) / ntheta
        mism = 0.0
        for idx in range(2, self._nE - 2):
            jn = float(self._can_js[idx])
            _, dnSndj = self._canonical_tables(jn)
            dSdj = dnSndj / self._nforSn
            tE = float(self._can_E(jn))
            om = float(self.OmegaHO(tE))
            domdj = float(self._can_dOmHOdE(tE)) * float(self._can_dEdj(jn))
            ja = jn + 2.0 * numpy.sum(
                self._nSn[idx] * numpy.cos(numpy.outer(th, self._nforSn)), axis=1
            )
            comp = ja * numpy.sin(2.0 * th)
            gn = -numpy.imag(numpy.fft.rfft(comp))[1 : len(self._nforSn) + 1] / ntheta
            pred = dSdj + domdj / (2.0 * om) * gn
            mism = max(mism, numpy.max(numpy.abs(pred - self._dSndJ[idx])))
        return mism

    def _xvFreqs_canonical(self, j, angle):
        """The canonical evaluation: every ingredient is an interpolant
        differentiated exactly or a closed form, so the map is exactly
        symplectic whatever the tables contain. The compensation for the
        j-dependent auxiliary frequency is closed form: dW^A/domega at fixed
        (x, J^A) equals x_a v_a / (2 omega) = [J^A sin(2 theta^A)] / (2
        omega)."""
        angle = numpy.atleast_1d(angle)
        nSn, dnSndj = self._canonical_tables(j)
        n = self._nforSn
        dSdj = dnSndj / n
        tE = float(self._can_E(j))
        om = float(self.OmegaHO(tE))
        dEdj = float(self._can_dEdj(j))
        domdj = float(self._can_dOmHOdE(tE)) * dEdj
        pref = domdj / (2.0 * om)
        # point-transformation parameter chains (identity PT: pi(xa) = xa and
        # all its parameter derivatives vanish)
        X = float(self._can_xmax(tE))
        PX = float(self._can_ptxmax(tE))
        dXdE = float(self._can_dxmaxdE(tE))
        dPXdE = float(self._can_dptxmaxdE(tE))
        xrow = (tE - self._Emin) / (self._Emax - self._Emin) * (self._nE - 1.0)
        cfs, dcfs_dx = self._can_row(self._can_ptcoeffs_c, xrow)
        dcfs_dE = dcfs_dx * (self._nE - 1.0) / (self._Emax - self._Emin)
        dcderiv = polynomial.polyder(cfs)

        def _residual(anglea):
            san = numpy.sin(n * numpy.atleast_2d(anglea).T)
            can = numpy.cos(n * numpy.atleast_2d(anglea).T)
            ja = j + 2.0 * numpy.sum(nSn * can, axis=1)
            xa = numpy.sqrt(2.0 * numpy.fabs(ja) / om) * numpy.sin(anglea)
            pa = numpy.sqrt(2.0 * numpy.fabs(ja) * om) * numpy.cos(anglea)
            u = xa / PX
            piprime = X / PX * polynomial.polyval(u, dcderiv)
            v = pa / piprime
            # d pi / dE at fixed xa, through every stored parameter chain
            dpidE = (
                dXdE * polynomial.polyval(u, cfs)
                + X * polynomial.polyval(u, dcfs_dE)
                - X * polynomial.polyval(u, dcderiv) * u * dPXdE / PX
            )
            comp_pt = -v * dpidE * dEdj
            return (
                anglea
                + 2.0 * numpy.sum(dSdj * san, axis=1)
                + pref * ja * numpy.sin(2.0 * anglea)
                + comp_pt
                - angle
            )

        anglea = copy.copy(angle)
        cntr = 0
        maxda = 2.0 * numpy.pi / 101.0
        hFD = 1e-7
        while cntr <= self._maxiter:
            F = (_residual(anglea) + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
            if numpy.max(numpy.fabs(F)) < self._angle_tol:
                break
            dF = (_residual(anglea + hFD) - _residual(anglea - hFD)) / (2.0 * hFD)
            da = -F / dF
            da[numpy.fabs(da) > maxda] = (numpy.sign(da) * maxda)[
                numpy.fabs(da) > maxda
            ]
            anglea = anglea + da
            cntr += 1
        else:  # pragma: no cover
            warnings.warn(
                "Canonical angle mapping did not converge in {} iterations".format(
                    self._maxiter
                ),
                galpyWarning,
            )
        ja = j + 2.0 * numpy.sum(
            nSn * numpy.cos(n * numpy.atleast_2d(anglea).T), axis=1
        )
        hoaainv = actionAngleHarmonicInverse(omega=om)
        xa, pa = hoaainv(ja, anglea)
        u = xa / PX
        x = X * polynomial.polyval(u, cfs)
        v = pa / (X / PX * polynomial.polyval(u, dcderiv))
        return (x, v, dEdj)

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
        if self._canonical:
            return self._xvFreqs_canonical(j, angle)
        # Find torus
        if not self._interp:
            indx = numpy.nanargmin(numpy.fabs(j - self._js))
            if numpy.fabs(j - self._js[indx]) > 1e-10:
                raise ValueError(
                    "Given action/energy not found, to use interpolation, initialize with setup_interp=True"
                )
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
            ja = j * numpy.ones_like(angle)
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
            ja = j + 2.0 * numpy.sum(
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
        if self._canonical:
            # the frequency of the interpolated Hamiltonian itself: exactly
            # consistent with the canonical chart (the integrator contract)
            return float(self._can_dEdj(j))
        # Find torus
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
