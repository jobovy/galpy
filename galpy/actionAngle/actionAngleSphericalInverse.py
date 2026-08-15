###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleSphericalInverse
#
#             Calculate (x,v) coordinates for any spherical potential from
#             given actions-angle coordinates
#
###############################################################################
import copy
import warnings

import numpy
from matplotlib import cm, gridspec, pyplot
from matplotlib.ticker import NullFormatter
from numpy.polynomial import polynomial
from scipy import integrate, interpolate, ndimage, optimize

from ..potential import (
    IsochronePotential,
    dvcircdR,
    epifreq,
    evaluatePotentials,
    evaluateRforces,
    rl,
    vcirc,
)
from ..potential.Potential import _evaluatePotentials
from ..util import galpyWarning
from ..util import plot as galpy_plot
from .actionAngleInverse import actionAngleInverse
from .actionAngleIsochrone import _actionAngleIsochroneHelper, actionAngleIsochrone
from .actionAngleIsochroneInverse import actionAngleIsochroneInverse
from .actionAngleSpherical import actionAngleSpherical

_APY_LOADED = True
try:
    from astropy import units
except ImportError:
    _APY_LOADED = False


def _evs(spl, iL, ix, **kwargs):
    # Scalar evaluation of a RectBivariateSpline (.ev returns a size-1 array)
    return spl.ev(iL, ix, **kwargs).item()


def _solve_kepler(k, theta, maxiter, tol=1e-14):
    """Solve the Kepler-like equation eta - k sin(eta) = theta for eta,
    for arrays of k and theta (in [0,2pi]) broadcast against each other.
    Because the left-hand side increases monotonically in eta, the root is
    always bracketed by theta and pi (eta >= theta before apocenter and
    eta <= theta after it), so we can safeguard Newton-Raphson with
    bisection: the Newton-Raphson iteration converges quadratically except
    near eta = 0 for nearly-radial tori (where k -> 1 and the derivative
    vanishes), where the bisection fallback takes over. Only the entries
    that have not converged yet are iterated."""
    k, theta = numpy.broadcast_arrays(k, theta)
    shape = theta.shape
    k = numpy.ascontiguousarray(k, dtype="float").flatten()
    theta = numpy.ascontiguousarray(theta, dtype="float").flatten()
    eta = copy.copy(theta)
    brlo = numpy.minimum(theta, numpy.pi)
    brhi = numpy.maximum(theta, numpy.pi)
    unconv = numpy.ones(theta.shape, dtype="bool")
    cntr = 0
    while numpy.any(unconv):
        teta, tk = eta[unconv], k[unconv]
        tbrlo, tbrhi = brlo[unconv], brhi[unconv]
        f = teta - tk * numpy.sin(teta) - theta[unconv]
        tbrlo = numpy.where(f <= 0.0, teta, tbrlo)
        tbrhi = numpy.where(f >= 0.0, teta, tbrhi)
        with numpy.errstate(divide="ignore", invalid="ignore"):
            newteta = teta - f / (1.0 - tk * numpy.cos(teta))
        # Fall back onto bisection when Newton-Raphson leaves the bracket
        bisect = (
            (newteta <= tbrlo) + (newteta >= tbrhi) + (True ^ numpy.isfinite(newteta))
        )
        newteta[bisect] = 0.5 * (tbrlo[bisect] + tbrhi[bisect])
        conv = numpy.fabs(newteta - teta) < tol
        eta[unconv] = newteta
        brlo[unconv] = tbrlo
        brhi[unconv] = tbrhi
        unconv[numpy.where(unconv)[0][conv]] = False
        cntr += 1
        if cntr > maxiter:
            warnings.warn(
                "Solving the Kepler-like equation for the auxiliary eccentric anomaly did not converge in {} iterations".format(
                    maxiter
                ),
                galpyWarning,
            )
            break
    return eta.reshape(shape)


class actionAngleSphericalInverse(actionAngleInverse):
    """Inverse action-angle formalism for spherical potentials"""

    def __init__(
        self,
        pot=None,
        Es=[0.1, 0.3],
        Ls=[1.0, 1.2],
        setup_interp=False,
        Rmax=5.0,
        Rinf=25.0,
        nL=31,
        nE=31,
        nta=128,
        use_pointtransform=False,
        pt_deg=7,
        pt_nra=301,
        exact_pt_spl_deg=5,
        exact_pt_tol=1e-12,
        pt_only=False,
        maxiter=100,
        angle_tol=1e-12,
        bisect=False,
    ):
        """
        NAME:

           __init__

        PURPOSE:

           initialize an actionAngleSphericalInverse object

        INPUT:

           pot= a Potential or list thereof, should be a spherical potential

           Either:

              a)

                 Es= energies of the orbits to map the tori for

                 Ls= angular momenta of the orbits to map the tori for

              b)

                 setup_interp= (False) if True, setup interpolation grids that allow any torus within the grid to be accessed through interpolation

                 Rmax= (5.) maximum radius to consider when building the L grid

                 Rinf= (5.) maximum radius to consider when building the E grid

                 nE= (31) number of energies to grid

                 nL= (31) number of angular momenta to grid

           nta= (128) number of auxiliary angles to sample the torus at when mapping the torus

           maxiter= (100) maximum number of iterations of root-finding algorithms

           angle_tol= (1e-12) tolerance for angle root-finding (f(x) is within tol of desired value)

           bisect= (False) if True, use simple bisection for root-finding, otherwise first try Newton-Raphson (mainly useful for testing the bisection fallback)

           use_pointtransform= (False) if True, setup a point transformation to, e.g., better handle highly radial orbits; use "exact" to solve for the point transformation that makes the torus exactly an isochrone torus, otherwise a simple polynomial point transformation is used

           pt_only= (False) if True, evaluate the inverse transformation using the exact point transformation alone, skipping the generating-function mapping, which is the identity for a perfect point transformation (only allowed when use_pointtransform == "exact"); the mapping coefficients are still computed at setup as a diagnostic and a warning is issued if they are not small

        OUTPUT:

           instance

        HISTORY:

           2017-11-21 - Started initial implementation that works for single (E,L) - Bovy (UofT)

           2018-11-02 - Started efficient implementation for multiple (E,L), like actionAngleVerticalInverse - Bovy (UofT)

        """
        # actionAngleInverse.__init__(self,*args,**kwargs)
        if pot is None:  # pragma: no cover
            raise OSError("Must specify pot= for actionAngleSphericalInverse")
        self._pot = pot
        self._aAS = actionAngleSpherical(pot=self._pot)
        # Determine gridding options
        if not setup_interp:
            self._Es = numpy.atleast_1d(Es)
            self._Ls = numpy.atleast_1d(Ls)
            self._nE = len(self._Es)
            self._nL = len(self._Ls)
            if self._nE != self._nL:
                raise ValueError("When grid=False, len(Es) has to equal len(Ls)")
            self._internal_Es = copy.copy(self._Es)
            self._internal_Ls = copy.copy(self._Ls)
        else:
            # Make grid, flatten so we can treat it as regular 1D input
            self._Rmax = Rmax
            self._Rinf = Rinf
            self._nE = nE
            self._nL = nL
            self._Lmin = 0.01
            self._Ls = numpy.linspace(
                self._Lmin, self._Rmax * vcirc(self._pot, self._Rmax), self._nL
            )
            self._Lmax = self._Ls[-1]
            # Calculate ER(vr=0,R=RL)
            self._RL = numpy.array([rl(self._pot, l) for l in self._Ls])
            # self._RLInterp= interpolate.InterpolatedUnivariateSpline(self._Ls,
            #                                                     self._RL,k=3)
            self._ERRL = (
                _evaluatePotentials(self._pot, self._RL, numpy.zeros(self._nL))
                + self._Ls**2.0 / 2.0 / self._RL**2.0
            )
            # self._ERRLmax= numpy.amax(self._ERRL)+1.
            # self._ERRLInterp= interpolate.InterpolatedUnivariateSpline(self._Ls,
            # numpy.log(-(self._ERRL-self._ERRLmax)),k=3)
            self._ERRa = (
                _evaluatePotentials(self._pot, self._Rinf, 0.0)
                + self._Ls**2.0 / 2.0 / self._Rinf**2.0
            )
            # self._ERRamax= numpy.amax(self._ERRa)+1.
            # self._ERRaInterp= interpolate.InterpolatedUnivariateSpline(self._Lzs,
            #                                                           numpy.log(-(self._ERRa-self._ERRamax)),k=3)
            # Space the normalized energies x quadratically, because torus
            # properties like rperi/rap and the Fourier/point-transformation
            # coefficients behave as sqrt(E-E_circular) near the circular-orbit
            # edge of the grid; on a quadratic grid they are smooth functions
            # of the grid index, so spline interpolation over the grid
            # converges at full order
            self._internal_Es = (
                numpy.tile(numpy.linspace(0.0, 1.0, self._nE) ** 2.0, (self._nL, 1))
            ).flatten() * (
                numpy.tile(self._ERRa - self._ERRL, (self._nE, 1)).T
            ).flatten() + (numpy.tile(self._ERRL, (self._nE, 1)).T).flatten()
            self._internal_Ls = (numpy.tile(self._Ls, (self._nE, 1)).T).flatten()
            # self._internal_Es,self._internal_Ls= \
            #    numpy.meshgrid(numpy.linspace(0.,1.,self._nE),
            #                   self._Ls,indexing='ij')
            # self._internal_Es= self._internal_Es.flatten()
            # self._internal_Ls= self._internal_Ls.flatten()
        self._L2 = self._internal_Ls**2
        # Total number of tori (nE*nL for the interpolation grid)
        self._ntori = len(self._internal_Es)
        # Compute actions, frequencies, and rperi/rap for each (E,L), to do
        # this, setup orbit at radius of circular orbit for given L
        rls = numpy.array(
            [
                optimize.newton(
                    lambda x: x * vcirc(self._pot, x) - L,
                    1.0,
                    lambda x: dvcircdR(self._pot, x) + x,
                )
                for L in self._internal_Ls
            ]
        )
        vrls = numpy.sqrt(
            2.0
            * (
                self._internal_Es
                - evaluatePotentials(self._pot, rls, numpy.zeros_like(rls))
            )
            - self._L2 / rls**2.0
        )
        self._circIndx = numpy.zeros(len(rls), dtype="bool")
        if setup_interp:
            # The lower edge of the energy grid consists of exactly circular
            # orbits, which we treat analytically in the epicyclic limit,
            # because the numerical action/frequency computation breaks down
            # there
            vrls[:: self._nE] = 0.0
            self._circIndx[:: self._nE] = True
        self._jr = numpy.zeros(len(rls))
        self._Omegar = numpy.empty(len(rls))
        self._Omegaz = numpy.empty(len(rls))
        self._rperi = numpy.empty(len(rls))
        self._rap = numpy.empty(len(rls))
        gIndx = True ^ self._circIndx
        if numpy.any(gIndx):
            (
                self._jr[gIndx],
                _,
                _,
                self._Omegar[gIndx],
                _,
                self._Omegaz[gIndx],
            ) = self._aAS.actionsFreqs(
                rls[gIndx],
                vrls[gIndx],
                self._internal_Ls[gIndx] / rls[gIndx],
                numpy.zeros(numpy.sum(gIndx)),
                numpy.zeros(numpy.sum(gIndx)),
            )
            # Also need rperi and rap
            _, _, self._rperi[gIndx], self._rap[gIndx] = self._aAS.EccZmaxRperiRap(
                rls[gIndx],
                vrls[gIndx],
                self._internal_Ls[gIndx] / rls[gIndx],
                numpy.zeros(numpy.sum(gIndx)),
                numpy.zeros(numpy.sum(gIndx)),
            )
        if numpy.any(self._circIndx):
            self._Omegar[self._circIndx] = numpy.array(
                [epifreq(self._pot, r) for r in rls[self._circIndx]]
            )
            self._Omegaz[self._circIndx] = (
                vcirc(self._pot, rls[self._circIndx]) / rls[self._circIndx]
            )
            self._rperi[self._circIndx] = rls[self._circIndx]
            self._rap[self._circIndx] = rls[self._circIndx]
        self._OmegazoverOmegar = self._Omegaz / self._Omegar
        # First need to determine appropriate IsochronePotentials
        ampb = (
            self._L2
            * self._Omegaz
            * (self._Omegar - self._Omegaz)
            / (2.0 * self._Omegaz - self._Omegar) ** 2.0
        )
        if numpy.any(ampb < 0.0):
            raise NotImplementedError(
                "actionAngleSphericalInverse not implemented for the case where Omegaz > Omegar or Omegaz < Omegar/2"
            )
        amp = numpy.sqrt(
            self._Omegar
            * (
                self._jr
                + numpy.sqrt(self._L2 + 4.0 * ampb) * self._Omegaz / self._Omegar
            )
            ** 3.0
        )
        self._amp = amp
        self._b = ampb / amp
        # This sets up objects with arrays of parameters, which is not
        # generally supported in galpy, but works here as long as the object
        # is evaluated with the same number of phase-space points as the
        # length of the parameter array
        self._ip = IsochronePotential(amp=self._amp, b=self._b)
        self._isoaa = actionAngleIsochrone(ip=self._ip)
        self._isoaainv = actionAngleIsochroneInverse(ip=self._ip)
        if use_pointtransform:
            if (
                not isinstance(use_pointtransform, bool)
                and use_pointtransform.lower() == "exact"
            ):
                self._pt_exact = True
                self._exact_pt_spl_deg = exact_pt_spl_deg
                self._exact_pt_tol = exact_pt_tol
                self._pt_only = pt_only
            else:
                if pt_only:
                    raise ValueError(
                        'pt_only=True is only supported for use_pointtransform="exact"'
                    )
                self._pt_exact = False
                self._pt_only = False
                self._exact_pt_spl_deg = None
            self._setup_pointtransform(pt_deg, pt_nra)
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
            self._pt_nra = pt_nra
            self._pt_rperi = self._rperi
            self._pt_rap = self._rap
            self._pt_coeffs = numpy.zeros((len(self._internal_Es), 2))
            self._pt_coeffs[:, 1] = 1.0
            self._pt_deriv_coeffs = numpy.ones((len(self._internal_Es), 1))
            self._pt_deriv2_coeffs = numpy.zeros((len(self._internal_Es), 1))
        # Extra keyword arguments to pass to _anglera, _jraora, ... for the
        # exact point transformation (which the polynomial evaluation doesn't
        # need); for the exact point transformation, the positional
        # ptcoeffs-style arguments of those functions instead carry the
        # (possibly fractional) row index of each point's torus in the grid
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
        # Determine the r grid for even-spaced theta_r grid
        self._rgrid = self._create_rgrid()
        # Compute mapping coefficients
        isoaa_helper = _actionAngleIsochroneHelper(
            ip=IsochronePotential(amp=self._ampgrid, b=self._bgrid)
        )
        self._jra, self._ora, self._Ea = _jraora(
            self._rgrid,
            self._Egrid,
            self._Lgrid,
            self._Lgrid**2.0,
            self._pot,
            isoaa_helper,
            self._ptcoeffsgrid,
            self._ptderivcoeffsgrid,
            self._rperigrid,
            self._rapgrid,
            self._ptrperigrid,
            self._ptrapgrid,
            **self._pt_eval_kwargs,
        )
        self._djradjr, self._djradLish, self._dEadE, self._dEadL = _djradjrLish(
            self._rgrid,
            self._Egrid,
            self._Lgrid,
            self._Lgrid**2.0,
            self._Omegargrid,
            self._pot,
            isoaa_helper,
            self._ptcoeffsgrid,
            self._ptderivcoeffsgrid,
            self._ptderiv2coeffsgrid,
            self._rperigrid,
            self._rapgrid,
            self._ptrperigrid,
            self._ptrapgrid,
            **self._pt_eval_kwargs,
        )
        # Store mean(jra) as probably a better approx. of jr
        self._jr_orig = copy.copy(self._jr)
        self._jr = numpy.nanmean(self._jra, axis=1)
        # Store better approximation to Omegar and Omegaz
        self._Omegar_orig = copy.copy(self._Omegar)
        self._Omegaz_orig = copy.copy(self._Omegaz)
        with numpy.errstate(invalid="ignore"):
            self._Omegar /= numpy.nanmean(self._djradjr, axis=1)
        self._Omegaz = self._OmegazoverOmegar * self._Omegar
        # The degenerate circular tori are treated in the epicyclic limit
        self._jr[self._circIndx] = 0.0
        self._Omegar[self._circIndx] = self._Omegar_orig[self._circIndx]
        self._Omegaz[self._circIndx] = self._Omegaz_orig[self._circIndx]
        # Compute Fourier expansions
        self._nforSn = numpy.arange(self._jra.shape[1] // 2 + 1)
        self._nSn = (
            numpy.real(
                numpy.fft.rfft(self._jra - numpy.atleast_2d(self._jr).T, axis=1)
            )[:, 1:]
            / self._jra.shape[1]
        )
        self._dSndJr = (
            numpy.real(numpy.fft.rfft(self._djradjr - 1.0, axis=1))[:, 1:]
            / self._jra.shape[1]
        )
        self._dSndLish = (
            numpy.real(numpy.fft.rfft(self._djradLish, axis=1))[:, 1:]
            / self._jra.shape[1]
        )
        self._dSndJr = (
            numpy.real(
                numpy.fft.rfft(
                    self._djradjr
                    / numpy.atleast_2d(numpy.nanmean(self._djradjr, axis=1)).T
                    - 1.0,
                    axis=1,
                )
            )[:, 1:]
            / self._jra.shape[1]
        )
        self._dSndLish = (
            numpy.real(
                numpy.fft.rfft(
                    self._djradLish
                    - numpy.atleast_2d(numpy.nanmean(self._djradLish, axis=1)).T,
                    axis=1,
                )
            )[:, 1:]
            / self._jra.shape[1]
        )

        # The degenerate circular tori have vanishing coefficients in the
        # epicyclic limit
        self._nSn[self._circIndx] = 0.0
        self._dSndJr[self._circIndx] = 0.0
        self._dSndLish[self._circIndx] = 0.0
        # Interpolation of small, noisy coeffs doesn't work, so set to zero
        if setup_interp:
            self._nSn[numpy.fabs(self._nSn) < 1e-16] = 0.0
            self._dSndJr[numpy.fabs(self._dSndJr) < 1e-15] = 0.0
            self._dSndLish[numpy.fabs(self._dSndLish) < 1e-15] = 0.0
        self._dSndJr /= numpy.atleast_2d(self._nforSn)[:, 1:]
        self._dSndLish /= numpy.atleast_2d(self._nforSn)[:, 1:]
        self._nforSn = self._nforSn[1:]
        # When evaluating using the point transformation only, the computed
        # mapping coefficients serve as a diagnostic of the accuracy of the
        # point transformation: they should all be close to zero
        if self._pt_exact and self._pt_only:
            relnSn = numpy.nanmax(numpy.fabs(self._nSn)) / numpy.nanmax(
                numpy.fabs(self._jr) + 1e-15
            )
            maxdSndJr = numpy.nanmax(numpy.fabs(self._dSndJr))
            maxdSndLish = numpy.nanmax(numpy.fabs(self._dSndLish))
            if relnSn > 1e-8 or maxdSndJr > 1e-8 or maxdSndLish > 1e-8:
                warnings.warn(
                    "Point transformation is not accurate enough for pt_only=True evaluation: the generating-function mapping that pt_only skips is not negligible (max |nSn|/max Jr = {:.2e}, max |dSndJr| = {:.2e}, max |dSndLish| = {:.2e}); decrease exact_pt_tol and/or increase pt_nra, or use pt_only=False".format(
                        relnSn, maxdSndJr, maxdSndLish
                    ),
                    galpyWarning,
                )
        # Setup interpolation if requested
        if setup_interp:
            self._interp = True
            self._setup_interp()
        else:
            self._interp = False
        # Check the units
        # self._check_consistent_units()
        return None

    def _setup_pointtransform(self, pt_deg, pt_nra):
        # Setup a point transformation for each torus
        self._pt_deg = pt_deg
        self._pt_nra = pt_nra
        Etilde = -0.5 * (self._amp * self._Omegar) ** (2.0 / 3.0)
        isoaa_helper = _actionAngleIsochroneHelper(ip=self._ip)
        self._pt_rperi, self._pt_rap = isoaa_helper.rperirap(Etilde, self._L2)
        if self._pt_exact:
            return self._setup_pointtransform_exact(pt_nra, Etilde)
        ramesh = numpy.linspace(0.0, 1.0, pt_nra)
        self._pt_coeffs = numpy.empty((self._ntori, pt_deg + 1))
        self._pt_deriv_coeffs = numpy.empty((self._ntori, pt_deg))
        self._pt_deriv2_coeffs = numpy.empty((self._ntori, pt_deg - 1))

        for ii in range(self._ntori):
            if self._jr[ii] < 1e-10:  # Just use identity for small J
                self._pt_coeffs[ii] = 0.0
                self._pt_coeffs[ii, 1] = 1.0
                self._pt_deriv_coeffs[ii] = 1.0
                self._pt_deriv2_coeffs[ii] = 0.0
                self._pt_rperi[ii] = self._rperi[ii]
                self._pt_rap[ii] = self._rap[ii]
                # Next fit warm-starts from the identity mapping, whose
                # free-parameter vector is all zeros
                coeffs = numpy.zeros(pt_deg - 3)
                continue
            Ea = Etilde[ii]
            ip = IsochronePotential(amp=self._amp[ii], b=self._b[ii])

            # Function to optimize with least squares for approximate point transform: p-p
            def opt_func(coeffs):
                # constraints: map [0,1] --> [0,1]

                # 1 + x + x2 + x3 + ...
                # 1 + 2x + 3x2 + ...
                # first 0 to map 0 --> 0
                # second 1 to have d = 1 at 0
                # sum rest = 0 to map 1 --> 1
                # sum n * rest = 0 to have d = 1 at 1

                # a + b + c + ... = 0 ==> a = -b -c - ...
                # 2a + 3b + 4c + ... = 0 ==> -2b -2c -... + 3b + 4c ... = b + 2c + 3d + ... = 0 ==> b = -2c -3d ...

                ccoeffs = numpy.zeros(pt_deg + 1)
                ccoeffs[1] = 1.0
                ccoeffs[3] = -numpy.sum(
                    polynomial.polyder(
                        numpy.hstack(
                            (
                                [
                                    0.0,
                                    0.0,
                                ],
                                coeffs,
                            )
                        )
                    )[1:]
                )
                ccoeffs[2] = -numpy.sum(coeffs) - ccoeffs[3]
                ccoeffs[4::] = coeffs
                # ccoeffs/= chebyshev.chebval(1,ccoeffs)
                # pt= chebyshev.Chebyshev(ccoeffs)

                pt = polynomial.Polynomial(ccoeffs)

                rmesh = (self._rap[ii] - self._rperi[ii]) * pt(ramesh) + self._rperi[ii]
                # Compute v from (E,L2,rmesh)
                vr2mesh = (
                    2.0
                    * (
                        self._internal_Es[ii]
                        - evaluatePotentials(self._pot, rmesh, numpy.zeros_like(rmesh))
                    )
                    - self._L2[ii] / rmesh**2.0
                )
                vr2mesh[vr2mesh < 0.0] = 0.0
                vrmesh = numpy.sqrt(vr2mesh)
                # Compute v from vra^2 = 2(Ea-isopot)-L2/ra^2 and transform
                real_ramesh = (
                    self._pt_rap[ii] - self._pt_rperi[ii]
                ) * ramesh + self._pt_rperi[ii]
                vra2mesh = (
                    2.0 * (Ea - ip(real_ramesh, numpy.zeros_like(real_ramesh)))
                    - self._L2[ii] / real_ramesh**2.0
                )
                vra2mesh[vra2mesh < 0.0] = 0.0
                vramesh = numpy.sqrt(vra2mesh)
                piprime = (
                    pt.deriv()(ramesh)
                    * (self._rap[ii] - self._rperi[ii])
                    / (self._pt_rap[ii] - self._pt_rperi[ii])
                )
                vrtildemesh = (
                    vramesh - numpy.sqrt(vr2mesh) * (1.0 / piprime - piprime)
                ) / piprime
                return vrmesh - vrtildemesh

            if ii == 0:
                # Start from identity mapping
                start_coeffs = [0.0]
                start_coeffs.extend([0.0 for jj in range(pt_deg - 4)])
            else:
                # Start from previous best fit's free-parameter vector
                start_coeffs = coeffs
            coeffs = optimize.leastsq(opt_func, start_coeffs)[0]
            # Extract full Chebyshev parameters from constrained optimization

            ccoeffs = numpy.zeros(pt_deg + 1)
            ccoeffs[1] = 1.0
            ccoeffs[3] = -numpy.sum(
                polynomial.polyder(
                    numpy.hstack(
                        (
                            [
                                0.0,
                                0.0,
                            ],
                            coeffs,
                        )
                    )
                )[1:]
            )
            ccoeffs[2] = -numpy.sum(coeffs) - ccoeffs[3]
            ccoeffs[4::] = coeffs
            """
            ccoeffs= numpy.zeros(pt_deg+1)
            ccoeffs[1]= 1.
            ccoeffs[2::]= coeffs
            ccoeffs/= chebyshev.chebval(1,ccoeffs)# map exact [0,1] --> [0,1]
            """

            # coeffs keeps the free-parameter vector to warm-start the next fit
            self._pt_coeffs[ii] = ccoeffs
            self._pt_deriv_coeffs[ii] = polynomial.polyder(ccoeffs, m=1)
            self._pt_deriv2_coeffs[ii] = polynomial.polyder(ccoeffs, m=2)
        return None

    def _setup_pointtransform_exact(self, pt_nra, Etilde):
        # Setup the exact point transformation for each torus by solving the
        # ordinary differential equation d pi / d ra = vr / vra that it
        # satisfies, together with the Delta-psi (zeta') shift of the
        # in-plane angle. The system is integrated using the auxiliary
        # torus' eccentric anomaly eta as the independent variable and
        # y = sin^2(chi/2) as the dependent variable for the normalized
        # radius, in which form it is regular at both turning points
        # (in particular, d Delta-psi / d eta = L [1/pi^2 - 1/ra^2] G(eta)
        # exactly, with no divergence at the turning points). The result is
        # stored as the normalized mapping y = (r-rperi)/(rap-rperi) as a
        # function of ya = (ra-ptrperi)/(ptrap-ptrperi) sampled on a fixed
        # mesh (padded beyond [0,1], because the 2D-spline evaluation's
        # mirror boundary would otherwise distort the interpolation near the
        # turning points) and is evaluated using 2D spline interpolation,
        # with the derivative and Delta-psi arrays sampled on the same mesh
        self._pt_nmesh = pt_nra
        self._pt_pad = 4 * self._exact_pt_spl_deg + 16
        dmesh = 1.0 / (pt_nra - 1.0)
        nmext = pt_nra + 2 * self._pt_pad
        self._pt_yamesh = numpy.linspace(
            -self._pt_pad * dmesh, 1.0 + self._pt_pad * dmesh, nmext
        )
        # Initialize all tori to the identity mapping (also used for small J)
        self._pt_coeffs = numpy.tile(self._pt_yamesh, (self._ntori, 1))
        self._pt_deriv_coeffs = numpy.ones((self._ntori, nmext))
        self._pt_deriv2_coeffs = numpy.zeros((self._ntori, nmext))
        self._pt_dpsi_coeffs = numpy.zeros((self._ntori, nmext))
        self._pt_dpsi_apo = numpy.zeros(self._ntori)
        # For a circular orbit, the auxiliary torus is circular as well and
        # its radius equals that of the true circular orbit, because matching
        # the frequencies matches the circular angular frequency L/r^2; the
        # turning points of the auxiliary torus are therefore those of the
        # true one (evaluating the general expression would divide by zero)
        zIndx = self._jr < 1e-10
        self._pt_rperi[zIndx] = self._rperi[zIndx]
        self._pt_rap[zIndx] = self._rap[zIndx]
        gIndx = True ^ zIndx
        if numpy.any(gIndx):
            Es = self._internal_Es[gIndx]
            L = self._internal_Ls[gIndx]
            L2 = self._L2[gIndx]
            amp = self._amp[gIndx]
            b = self._b[gIndx]
            Ea = Etilde[gIndx]
            rperi = self._rperi[gIndx]
            rap = self._rap[gIndx]
            ptrperi = self._pt_rperi[gIndx]
            ptrap = self._pt_rap[gIndx]
            deltar = rap - rperi
            ng = numpy.sum(gIndx)
            # Parameters of the auxiliary torus
            ca = -amp / 2.0 / Ea - b
            ea = numpy.sqrt(1.0 - L2 / amp / ca * (1.0 + b / ca))
            # Endpoint limits of Q = vr^2/[y(1-y)]:
            # W'(rperi) (rap-rperi) and -W'(rap) (rap-rperi),
            # where W(r) = vr^2(r) along the true torus
            Qperi = (
                2.0
                * (evaluateRforces(self._pot, rperi, numpy.zeros(ng)) + L2 / rperi**3.0)
                * deltar
            )
            Qap = (
                -2.0
                * (evaluateRforces(self._pot, rap, numpy.zeros(ng)) + L2 / rap**3.0)
                * deltar
            )

            def deriv_eta(eta, state):
                chi = state[:ng]
                dpsi_unused = state[ng:]  # noqa: F841
                y = numpy.sin(chi / 2.0) ** 2.0
                r = rperi + deltar * y
                # Radius along the auxiliary torus at this eccentric
                # anomaly; writing it in terms of u = b (s-1) - b, that is,
                # r^A = sqrt[u (u+2b)], avoids dividing by the isochrone
                # scale b and is regular in the Kepler limit b -> 0
                u = ca * (1.0 - ea * numpy.cos(eta))
                ra = numpy.sqrt(u * (u + 2.0 * b))
                # G = (d ra / d eta) / vra = b (s-1) sqrt[(b+c)/GM], which
                # is regular at the turning points
                G = numpy.sqrt(b**2.0 + ra**2.0) * numpy.sqrt((b + ca) / amp)
                vr2 = (
                    2.0 * (Es - evaluatePotentials(self._pot, r, numpy.zeros(ng)))
                    - L2 / r**2.0
                )
                vr2[vr2 < 0.0] = 0.0
                y1my = y * (1.0 - y)
                Q = numpy.empty(ng)
                reg = y1my > 1e-6
                Q[reg] = vr2[reg] / y1my[reg]
                lowIndx = (True ^ reg) * (y < 0.5)
                hiIndx = (True ^ reg) * (y >= 0.5)
                Q[lowIndx] = Qperi[lowIndx]
                Q[hiIndx] = Qap[hiIndx]
                Q[Q < 0.0] = 0.0
                dchideta = G / deltar * numpy.sqrt(Q)
                ddpsideta = L * (1.0 / r**2.0 - 1.0 / ra**2.0) * G
                return numpy.concatenate((dchideta, ddpsideta))

            etamesh = numpy.linspace(0.0, numpy.pi, numpy.amax([pt_nra, 201]))
            sol = integrate.solve_ivp(
                deriv_eta,
                [0.0, numpy.pi],
                numpy.zeros(2 * ng),
                t_eval=etamesh,
                rtol=self._exact_pt_tol,
                atol=self._exact_pt_tol,
                method="DOP853",
            )
            if not sol.success:  # pragma: no cover
                raise RuntimeError(
                    "Solving the ODE that defines the exact point transformation failed, full message: "
                    + sol.message
                )
            chis = sol.y[:ng]
            dpsis = sol.y[ng:]
            # The turning point must map exactly onto the turning point
            ys = numpy.sin(chis / 2.0) ** 2.0
            ys[:, 0] = 0.0
            ys[:, -1] = 1.0
            # Resample onto the fixed mesh in the normalized auxiliary radius
            # ya in two stages: first represent the solution as a spline in
            # the eccentric anomaly, on whose uniform grid the spline is
            # well-conditioned (the solution's sampling in ya clusters
            # quadratically near the turning points, where a direct spline
            # would be noisy), then evaluate it at the closed-form eta(ya) of
            # the uniform core ya mesh and build the final spline through
            # those, sampling it on the extended mesh (polynomial
            # extrapolation of the end pieces beyond [0,1])
            yacore = numpy.linspace(0.0, 1.0, self._pt_nmesh)
            # Delta-psi is a smooth, odd function of the auxiliary radial
            # angle around both turning points, but has a sqrt branch as a
            # function of ya there (ya is quadratic in time at the turning
            # points while Delta-psi is linear), so it is stored on a uniform
            # mesh in the auxiliary radial angle in [0,pi] (same index layout
            # as the ya mesh), padded with its exact odd extensions
            thetacore = numpy.linspace(0.0, numpy.pi, self._pt_nmesh)
            dth = numpy.pi / (self._pt_nmesh - 1.0)
            thlow = -dth * numpy.arange(self._pt_pad, 0, -1)
            thhigh = numpy.pi + dth * numpy.arange(1, self._pt_pad + 1)
            # Kepler-equation parameter of the auxiliary torus
            kepk = ea * ca / (ca + b)
            for jj, ii in enumerate(numpy.arange(self._ntori)[gIndx]):
                yspl_eta = interpolate.InterpolatedUnivariateSpline(
                    etamesh, ys[jj], k=self._exact_pt_spl_deg
                )
                dpsispl_eta = interpolate.InterpolatedUnivariateSpline(
                    etamesh, dpsis[jj], k=self._exact_pt_spl_deg
                )
                racore = ptrperi[jj] + (ptrap[jj] - ptrperi[jj]) * yacore
                # Inverse of the relation above: u = sqrt(b^2+r^2) - b
                cosetacore = (
                    1.0 - (numpy.sqrt(b[jj] ** 2.0 + racore**2.0) - b[jj]) / ca[jj]
                ) / ea[jj]
                etacore = numpy.arccos(numpy.clip(cosetacore, -1.0, 1.0))
                ycore = yspl_eta(etacore)
                ycore[0] = 0.0
                ycore[-1] = 1.0
                tspl = interpolate.InterpolatedUnivariateSpline(
                    yacore, ycore, k=self._exact_pt_spl_deg
                )
                self._pt_coeffs[ii] = tspl(self._pt_yamesh)
                self._pt_deriv_coeffs[ii] = tspl(self._pt_yamesh, nu=1)
                self._pt_deriv2_coeffs[ii] = tspl(self._pt_yamesh, nu=2)
                # Solve the Kepler equation for eta on the uniform mesh in
                # the auxiliary radial angle; eta - k sin(eta) is strictly
                # increasing, so bisection always converges (Newton diverges
                # near eta = 0 for the nearly-radial tori, where k -> 1)
                etlo = numpy.zeros(self._pt_nmesh)
                ethi = numpy.pi * numpy.ones(self._pt_nmesh)
                for _ in range(53):
                    etath = 0.5 * (etlo + ethi)
                    fmid = etath - kepk[jj] * numpy.sin(etath) - thetacore
                    ethi = numpy.where(fmid > 0.0, etath, ethi)
                    etlo = numpy.where(fmid > 0.0, etlo, etath)
                etath = 0.5 * (etlo + ethi)
                dpsith = dpsispl_eta(etath)
                dpsith[0] = 0.0
                dpsi_apo = dpsith[-1]
                self._pt_dpsi_apo[ii] = dpsi_apo
                dpsithspl = interpolate.InterpolatedUnivariateSpline(
                    thetacore, dpsith, k=self._exact_pt_spl_deg
                )
                self._pt_dpsi_coeffs[ii] = numpy.concatenate(
                    (
                        -dpsithspl(-thlow),
                        dpsith,
                        2.0 * dpsi_apo - dpsithspl(2.0 * numpy.pi - thhigh),
                    )
                )
        # Store spline-filtered versions for fast 2D-spline evaluation
        self._pt_filtered = tuple(
            ndimage.spline_filter(arr, order=self._exact_pt_spl_deg)
            for arr in (
                self._pt_coeffs,
                self._pt_deriv_coeffs,
                self._pt_deriv2_coeffs,
                self._pt_dpsi_coeffs,
            )
        )
        return None

    def _create_rgrid(self):
        # Find r grid for regular grid in auxiliary angle (thetara)
        # in practice only need to map 0 < thetara < pi  to r with +v bc symm
        # To efficiently start the search, first compute thetara for a dense
        # grid in r (at +v); also don't allow points to be exactly at
        # rperi or rap, because Newton derivative is inf there...
        rgrid = numpy.linspace(0.0, 1.0, 2 * self._nta)
        rs = (
            rgrid * numpy.atleast_2d(self._pt_rap - self._pt_rperi - 2 * 1e-8).T
            + numpy.atleast_2d(self._pt_rperi + 1e-8).T
        )
        # Setup helper for computing angles, and derivative
        isoaa_helper = _actionAngleIsochroneHelper(
            ip=IsochronePotential(
                amp=numpy.tile(self._amp, (rs.shape[1], 1)).T,
                b=numpy.tile(self._b, (rs.shape[1], 1)).T,
            )
        )
        rta = _anglera(
            rs,
            numpy.tile(self._internal_Es, (rs.shape[1], 1)).T,
            numpy.tile(self._internal_Ls, (rs.shape[1], 1)).T,
            numpy.tile(self._L2, (rs.shape[1], 1)).T,
            self._pot,
            isoaa_helper,
            (
                numpy.tile(numpy.arange(self._ntori, dtype="float"), (rs.shape[1], 1)).T
                if self._pt_exact
                else numpy.rollaxis(numpy.tile(self._pt_coeffs, (rs.shape[1], 1, 1)), 1)
            ),
            (
                numpy.tile(numpy.arange(self._ntori, dtype="float"), (rs.shape[1], 1)).T
                if self._pt_exact
                else numpy.rollaxis(
                    numpy.tile(self._pt_deriv_coeffs, (rs.shape[1], 1, 1)), 1
                )
            ),
            numpy.tile(self._rperi, (rs.shape[1], 1)).T,
            numpy.tile(self._rap, (rs.shape[1], 1)).T,
            numpy.tile(self._pt_rperi, (rs.shape[1], 1)).T,
            numpy.tile(self._pt_rap, (rs.shape[1], 1)).T,
            **self._pt_eval_kwargs,
        )
        rta[numpy.isnan(rta)] = 0.0  # Zero energy orbit -> NaN
        # Now use Newton-Raphson to iterate to a regular grid
        cindx = numpy.nanargmin(
            numpy.fabs(
                (rta - numpy.rollaxis(numpy.atleast_3d(self._thetaa), 1) + numpy.pi)
                % (2.0 * numpy.pi)
                - numpy.pi
            ),
            axis=2,
        )
        rgrid = (
            rgrid[cindx].T * numpy.atleast_2d(self._rap - self._rperi - 2 * 1e-8).T
            + numpy.atleast_2d(self._rperi + 1e-8).T
        )
        if self._pt_exact:
            # With the exact point transformation, the auxiliary angle is the
            # isochrone radial angle on the auxiliary torus, so the grid
            # follows from solving the Kepler-like equation
            # eta - [e c/(c+b)] sin eta = thetaa; the Newton-Raphson
            # iteration below then merely polishes the result to be exactly
            # consistent with the stored spline representation
            tamp = numpy.atleast_2d(self._amp).T
            tb = numpy.atleast_2d(self._b).T
            tL2 = numpy.atleast_2d(self._L2).T
            tptrperi = numpy.atleast_2d(self._pt_rperi).T
            Ettmp = (
                -tamp / (tb + numpy.sqrt(tb**2.0 + tptrperi**2.0))
                + tL2 / 2.0 / tptrperi**2.0
            )
            catmp = -tamp / 2.0 / Ettmp - tb
            eatmp = numpy.sqrt(
                numpy.clip(1.0 - tL2 / tamp / catmp * (1.0 + tb / catmp), 0.0, None)
            )
            ktmp = eatmp * catmp / (catmp + tb)
            eta = _solve_kepler(
                ktmp, numpy.tile(self._thetaa, (self._ntori, 1)), self._maxiter
            )
            utmp = catmp * (1.0 - eatmp * numpy.cos(eta))
            rgrid = numpy.sqrt(utmp * (utmp + 2.0 * tb))
        Egrid = numpy.tile(self._internal_Es, (self._nta, 1)).T
        Lgrid = numpy.tile(self._internal_Ls, (self._nta, 1)).T
        L2grid = Lgrid**2
        # Force rperi and rap to be thetar=0 and pi and don't optimize later
        rgrid[:, 0] = self._pt_rperi
        rgrid[:, self._nta // 2] = self._pt_rap
        # Need to adjust parameters of helpers
        ampgrid = numpy.tile(self._amp, (self._nta, 1)).T
        bgrid = numpy.tile(self._b, (self._nta, 1)).T
        isoaa_helper._ip = IsochronePotential(amp=ampgrid, b=bgrid)
        isoaa_helper.amp = ampgrid
        isoaa_helper.b = bgrid
        rperigrid = numpy.tile(self._rperi, (self._nta, 1)).T
        rapgrid = numpy.tile(self._rap, (self._nta, 1)).T
        ptrperigrid = numpy.tile(self._pt_rperi, (self._nta, 1)).T
        ptrapgrid = numpy.tile(self._pt_rap, (self._nta, 1)).T
        if self._pt_exact:
            # For the exact point transformation, the positional
            # ptcoeffs-style arguments carry the row index of each point's
            # torus in the grid of tori instead of polynomial coefficients
            ptcoeffsgrid = numpy.tile(
                numpy.arange(self._ntori, dtype="float"), (self._nta, 1)
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
        ta = _anglera(
            rgrid,
            Egrid,
            Lgrid,
            L2grid,
            self._pot,
            isoaa_helper,
            ptcoeffsgrid,
            ptderivcoeffsgrid,
            rperigrid,
            rapgrid,
            ptrperigrid,
            ptrapgrid,
            **self._pt_eval_kwargs,
        )
        mta = numpy.tile(self._thetaa, (len(self._internal_Es), 1))
        # Now iterate
        cntr = 0
        unconv = numpy.ones(rgrid.shape, dtype="bool")
        # We'll fill in the -v part using the +v, also remove rperi/rap
        unconv[:, 0] = False
        unconv[:, self._nta // 2 :] = False
        # Also don't bother with the degenerate circular tori
        unconv[self._circIndx] = False
        dta = (ta[unconv] - mta[unconv] + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
        unconv[unconv] = numpy.fabs(dta) > self._angle_tol
        # Don't allow too big steps
        maxdr = numpy.tile(
            (self._rap - self._rperi) / float(self._nta), (self._nta, 1)
        ).T
        isoaa_helper._ip = IsochronePotential(amp=ampgrid[unconv], b=bgrid[unconv])
        isoaa_helper.amp = ampgrid[unconv]
        isoaa_helper.b = bgrid[unconv]
        while not self._bisect:
            dtadr = _danglera(
                rgrid[unconv],
                Egrid[unconv],
                Lgrid[unconv],
                L2grid[unconv],
                self._pot,
                isoaa_helper,
                ptcoeffsgrid[unconv],
                ptderivcoeffsgrid[unconv],
                ptderiv2coeffsgrid[unconv],
                rperigrid[unconv],
                rapgrid[unconv],
                ptrperigrid[unconv],
                ptrapgrid[unconv],
                **self._pt_eval_kwargs,
            )
            dta = (ta[unconv] - mta[unconv] + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
            dr = -dta / dtadr
            dr[numpy.fabs(dr) > maxdr[unconv]] = (numpy.sign(dr) * maxdr[unconv])[
                numpy.fabs(dr) > maxdr[unconv]
            ]
            rgrid[unconv] += dr
            rgrid[unconv * (rgrid > rapgrid)] = rapgrid[unconv * (rgrid > rapgrid)]
            rgrid[unconv * (rgrid < rperigrid)] = rperigrid[
                unconv * (rgrid < rperigrid)
            ]
            unconv[unconv] = numpy.fabs(dta) > self._angle_tol
            if numpy.sum(unconv) == 0:
                break
            isoaa_helper._ip = IsochronePotential(amp=ampgrid[unconv], b=bgrid[unconv])
            isoaa_helper.amp = ampgrid[unconv]
            isoaa_helper.b = bgrid[unconv]
            newta = _anglera(
                rgrid[unconv],
                Egrid[unconv],
                Lgrid[unconv],
                L2grid[unconv],
                self._pot,
                isoaa_helper,
                ptcoeffsgrid[unconv],
                ptderivcoeffsgrid[unconv],
                rperigrid[unconv],
                rapgrid[unconv],
                ptrperigrid[unconv],
                ptrapgrid[unconv],
                **self._pt_eval_kwargs,
            )
            ta[unconv] = newta
            cntr += 1
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
            new_rgrid = numpy.linspace(0.0, 1.0, 2 * self._nta)
            da = (
                rta - numpy.rollaxis(numpy.atleast_3d(self._thetaa), 1) + numpy.pi
            ) % (2.0 * numpy.pi) - numpy.pi
            da[da >= 0.0] = -numpy.nanmax(numpy.fabs(da)) - 0.1
            cindx = numpy.nanargmax(da, axis=2)
            tryr_min = (
                new_rgrid[cindx].T
                * numpy.atleast_2d(self._pt_rap - self._pt_rperi - 2 * 1e-8).T
                + numpy.atleast_2d(self._pt_rperi + 1e-8).T
            )[unconv]
            dr = (
                2.0 / (2.0 * self._nta - 1) * (ptrapgrid - ptrperigrid)
            )  # delta of initial x grid above
            while True:
                dr *= 0.5
                rgrid[unconv] = tryr_min + dr[unconv]
                isoaa_helper._ip = IsochronePotential(
                    amp=ampgrid[unconv], b=bgrid[unconv]
                )
                isoaa_helper.amp = ampgrid[unconv]
                isoaa_helper.b = bgrid[unconv]
                newta = (
                    _anglera(
                        rgrid[unconv],
                        Egrid[unconv],
                        Lgrid[unconv],
                        L2grid[unconv],
                        self._pot,
                        isoaa_helper,
                        ptcoeffsgrid[unconv],
                        ptderivcoeffsgrid[unconv],
                        rperigrid[unconv],
                        rapgrid[unconv],
                        ptrperigrid[unconv],
                        ptrapgrid[unconv],
                        **self._pt_eval_kwargs,
                    )
                    + 2.0 * numpy.pi
                ) % (2.0 * numpy.pi)
                ta[unconv] = newta
                #                print(mta[unconv],rgrid[unconv],ta[unconv])
                dta = (newta - mta[unconv] + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
                tryr_min[newta < mta[unconv]] = rgrid[unconv][newta < mta[unconv]]
                unconv[unconv] = numpy.fabs(dta) > self._angle_tol
                tryr_min = tryr_min[numpy.fabs(dta) > self._angle_tol]
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
        rgrid[:, self._nta // 2 + 1 :] = rgrid[:, 1 : self._nta // 2][:, ::-1]
        isoaa_helper._ip = IsochronePotential(
            amp=ampgrid[:, self._nta // 2 + 1 :], b=bgrid[:, self._nta // 2 + 1 :]
        )
        isoaa_helper.amp = ampgrid[:, self._nta // 2 + 1 :]
        isoaa_helper.b = bgrid[:, self._nta // 2 + 1 :]
        ta[:, self._nta // 2 + 1 :] = _anglera(
            rgrid[:, self._nta // 2 + 1 :],
            Egrid[:, self._nta // 2 + 1 :],
            Lgrid[:, self._nta // 2 + 1 :],
            L2grid[:, self._nta // 2 + 1 :],
            self._pot,
            isoaa_helper,
            ptcoeffsgrid[:, self._nta // 2 + 1 :],
            ptderivcoeffsgrid[:, self._nta // 2 + 1 :],
            rperigrid[:, self._nta // 2 + 1 :],
            rapgrid[:, self._nta // 2 + 1 :],
            ptrperigrid[:, self._nta // 2 + 1 :],
            ptrapgrid[:, self._nta // 2 + 1 :],
            vrneg=True,
            **self._pt_eval_kwargs,
        )
        self._dta = (ta - mta + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
        self._mta = mta
        # Store these, they are useful (obv. arbitrary to return rgrid
        # and not just store it...)
        self._Egrid = Egrid
        self._Omegargrid = numpy.tile(self._Omegar, (self._nta, 1)).T
        self._Lgrid = Lgrid
        self._ampgrid = ampgrid
        self._bgrid = bgrid
        self._ptcoeffsgrid = ptcoeffsgrid
        self._ptderivcoeffsgrid = ptderivcoeffsgrid
        self._ptderiv2coeffsgrid = ptderiv2coeffsgrid
        self._rperigrid = rperigrid
        self._rapgrid = rapgrid
        self._ptrperigrid = ptrperigrid
        self._ptrapgrid = ptrapgrid
        return rgrid

    def plot_convergence(
        self, E, L, overplot=False, return_gridspec=False, shift_action=None
    ):
        if shift_action is None:
            shift_action = self._pt_deg > 1
        # First find the torus for this energy and angular momentum
        indx = numpy.nanargmin(
            numpy.fabs(E - self._internal_Es) * numpy.fabs(L - self._internal_Ls)
        )
        if (
            numpy.fabs(E - self._Es[indx]) > 1e-10
            or numpy.fabs(L - self._internal_Ls[indx]) > 1e-10
        ):
            raise ValueError(
                "Given energy and angular-momentum pair not found; please specify an energy/angular-momentum pair used in the initialization of the instance"
            )
        if not overplot:
            gs = gridspec.GridSpec(2, 4, height_ratios=[4, 1])
        else:
            gs = overplot  # confusingly, we overload the meaning of overplot
        # mapping of thetaa --> r
        pyplot.subplot(gs[0])
        galpy_plot.plot(
            self._thetaa,
            self._rgrid[indx],
            color="k",
            ls="--" if overplot else "-",
            ylabel=r"$r(\theta_r^A)$",
            gcf=True,
            overplot=overplot,
        )
        if not overplot:
            pyplot.gca().xaxis.set_major_formatter(NullFormatter())
        if not overplot:
            pyplot.subplot(gs[4])
            # Setup isochrone helper for relevant calculations
            isoaa_helper = _actionAngleIsochroneHelper(
                ip=IsochronePotential(
                    amp=self._ampgrid[indx][0], b=self._bgrid[indx][0]
                )
            )
            negv = self._thetaa >= numpy.pi
            thetaa_out = numpy.empty_like(self._thetaa)
            one = numpy.ones(numpy.sum(True ^ negv))
            thetaa_out[True ^ negv] = _anglera(
                self._rgrid[indx][True ^ negv],
                E,
                L,
                L**2.0,
                self._pot,
                isoaa_helper,
                indx * one if self._pt_exact else self._pt_coeffs[indx],
                indx * one if self._pt_exact else self._pt_deriv_coeffs[indx],
                self._rperi[indx] * one,
                self._rap[indx] * one,
                self._pt_rperi[indx] * one,
                self._pt_rap[indx] * one,
                vrneg=False,
                **self._pt_eval_kwargs,
            )
            one = numpy.ones(numpy.sum(negv))
            thetaa_out[negv] = _anglera(
                self._rgrid[indx][negv],
                E,
                L,
                L**2.0,
                self._pot,
                isoaa_helper,
                indx * one if self._pt_exact else self._pt_coeffs[indx],
                indx * one if self._pt_exact else self._pt_deriv_coeffs[indx],
                self._rperi[indx] * one,
                self._rap[indx] * one,
                self._pt_rperi[indx] * one,
                self._pt_rap[indx] * one,
                vrneg=True,
                **self._pt_eval_kwargs,
            )
            galpy_plot.plot(
                self._thetaa,
                ((thetaa_out - self._thetaa + numpy.pi) % (2.0 * numpy.pi)) - numpy.pi,
                color="k",
                gcf=True,
                xlabel=r"$\theta_r^A$",
                ylabel=r"$\theta_r^A[r(\theta)r^A)]-\theta_r^A$",
            )
        # Recovery of the nSn from J_r^A(theta_r^A) behavior
        pyplot.subplot(gs[1])
        galpy_plot.plot(
            self._thetaa,
            self._jra[indx],
            color="k",
            ls="--" if overplot else "-",
            ylabel=r"$J_r^A(\theta_r^A),J$",
            gcf=True,
            overplot=overplot,
        )
        pyplot.axhline(
            self._jr[indx] + shift_action * (self._jr_orig[indx] - self._jr[indx]),
            color="k",
            ls="--",
        )
        if not overplot:
            pyplot.gca().xaxis.set_major_formatter(NullFormatter())
        if not overplot:
            pyplot.subplot(gs[5])
            galpy_plot.plot(
                self._thetaa,
                numpy.array(
                    [
                        self._jr[indx]
                        + 2.0 * numpy.sum(self._nSn[indx] * numpy.cos(self._nforSn * x))
                        for x in self._thetaa
                    ]
                )
                / self._jra[indx]
                - 1.0,
                color="k",
                xlabel=r"$\theta_r^A$",
                ylabel=r"$\delta J_r^A/J_r^A$",
                gcf=True,
            )
        # Recovery of the dSndJr from dJ_r^A/dJ_r(theta^A) behavior
        pyplot.subplot(gs[2])
        galpy_plot.plot(
            self._thetaa,
            self._djradjr[indx],
            color="k",
            ls="--" if overplot else "-",
            ylabel=r"$\partial J_r^A/\partial J_r(\theta_r^A)$",
            gcf=True,
            overplot=overplot,
        )
        pyplot.axhline(1.0, color="k", ls="--")
        if not overplot:
            pyplot.gca().xaxis.set_major_formatter(NullFormatter())
        if not overplot:
            pyplot.subplot(gs[6])
            galpy_plot.plot(
                self._thetaa,
                numpy.array(
                    [
                        1.0
                        + 2.0
                        * numpy.sum(
                            self._nforSn
                            * self._dSndJr[indx]
                            * numpy.cos(self._nforSn * x)
                        )
                        for x in self._thetaa
                    ]
                )
                - self._djradjr[indx] / numpy.nanmean(self._djradjr[indx]),
                color="k",
                xlabel=r"$\theta_r^A$",
                ylabel=r"$\delta \partial J_r^A/\partial J_r(\theta_r^A)$",
                gcf=True,
            )
        # Recovery of the dSndL from dJ_r^A/dL(theta^A) behavior
        pyplot.subplot(gs[3])
        galpy_plot.plot(
            self._thetaa,
            (
                self._djradLish[indx]
                + self._OmegazoverOmegar[indx] * (self._djradjr[indx] - 1.0)
            ),
            color="k",
            ls="--" if overplot else "-",
            ylabel=r"$\partial J_r^A/\partial L(\theta_r^A)$",
            gcf=True,
            overplot=overplot,
        )
        pyplot.axhline(1.0, color="k", ls="--")
        if not overplot:
            pyplot.gca().xaxis.set_major_formatter(NullFormatter())
        if not overplot:
            pyplot.subplot(gs[7])
            galpy_plot.plot(
                self._thetaa,
                numpy.array(
                    [
                        2.0
                        * numpy.sum(
                            self._nforSn
                            * self._dSndLish[indx]
                            * numpy.cos(self._nforSn * x)
                        )
                        for x in self._thetaa
                    ]
                )
                - self._djradLish[indx]
                + numpy.nanmean(self._djradLish[indx]),
                color="k",
                xlabel=r"$\theta_r^A$",
                ylabel=r"$\delta \partial J_r^A/\partial L(\theta_r^A)$",
                gcf=True,
            )
        pyplot.tight_layout()
        if return_gridspec:
            return gs
        else:
            return None

    def plot_power(self, Es, Ls, overplot=False, return_gridspec=False, ls="-"):
        Ls = numpy.atleast_1d(Ls)[numpy.argsort(numpy.atleast_1d(Es))]
        Es = numpy.sort(numpy.atleast_1d(Es))
        minn_for_cmap = 4
        if len(Es) < minn_for_cmap:
            if not overplot:
                gs = gridspec.GridSpec(1, 3)
            else:
                gs = overplot  # confusingly, we overload the meaning of overplot
        else:
            if not overplot:
                outer = gridspec.GridSpec(1, 3, width_ratios=[2.0, 0.05], wspace=0.05)
                gs = gridspec.GridSpecFromSubplotSpec(
                    1, 3, subplot_spec=outer[0], wspace=0.35
                )
            else:
                raise RuntimeError(
                    f"plot_power with >= {minn_for_cmap} energies and overplot=True is not supported"
                )
        for ii, (E, L) in enumerate(zip(Es, Ls)):
            # First find the torus for this energy and angular momentum
            indx = numpy.nanargmin(
                numpy.fabs(E - self._internal_Es) * numpy.fabs(L - self._internal_Ls)
            )
            if (
                numpy.fabs(E - self._Es[indx]) > 1e-10
                or numpy.fabs(L - self._internal_Ls[indx]) > 1e-10
            ):
                raise ValueError(
                    "Given energy and angular-momentum pair not found; please specify an energy/angular-momentum pair used in the initialization of the instance"
                )
            # n S_n
            y = numpy.fabs(self._nSn[indx])
            if len(Es) > 1 and E == Es[0]:
                y4minmax = numpy.fabs(self._nSn[:])
                ymin = numpy.amax(
                    [numpy.amin(y4minmax[numpy.isfinite(y4minmax)]), 1e-17]
                )
                ymax = numpy.amax(y4minmax[numpy.isfinite(y4minmax)])
            elif len(Es) == 1:
                ymin = numpy.amax([numpy.amin(y[numpy.isfinite(y)]), 1e-17])
                ymax = numpy.amax(y[numpy.isfinite(y)])
            if len(Es) < minn_for_cmap:
                label = rf"$E, L = {E:g}, {L:g}$"
                color = f"C{ii}"
            else:
                label = None
                color = cm.plasma((E - Es[0]) / (Es[-1] - Es[0]))
            pyplot.subplot(gs[0])
            galpy_plot.plot(
                numpy.fabs(self._nforSn),
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
            # d S_n / d J_r
            y = numpy.fabs(self._dSndJr[indx])
            if len(Es) > 1 and E == Es[0]:
                y4minmax = numpy.fabs(self._dSndJr)
                ymin = numpy.amax(
                    [numpy.amin(y4minmax[numpy.isfinite(y4minmax)]), 1e-17]
                )
                ymax = numpy.amax(y4minmax[numpy.isfinite(y4minmax)])
            elif len(Es) == 1:
                ymin = numpy.amax([numpy.amin(y[numpy.isfinite(y)]), 1e-17])
                ymax = numpy.amax(y[numpy.isfinite(y)])
            if len(Es) < minn_for_cmap:
                label = rf"$E, L = {E:g}, {L:g}$"
                color = f"C{ii}"
            else:
                label = None
                color = cm.plasma((E - Es[0]) / (Es[-1] - Es[0]))
            pyplot.subplot(gs[1])
            galpy_plot.plot(
                numpy.fabs(self._nforSn),
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
                ylabel=r"$|\mathrm{d}S_n/\mathrm{d}J_r|$",
            )
            # d S_n / d Lish
            y = numpy.fabs(self._dSndLish[indx])
            if len(Es) > 1 and E == Es[0]:
                y4minmax = numpy.fabs(self._dSndLish)
                ymin = numpy.amax(
                    [numpy.amin(y4minmax[numpy.isfinite(y4minmax)]), 1e-17]
                )
                ymax = numpy.amax(y4minmax[numpy.isfinite(y4minmax)])
            elif len(Es) == 1:
                ymin = numpy.amax([numpy.amin(y[numpy.isfinite(y)]), 1e-17])
                ymax = numpy.amax(y[numpy.isfinite(y)])
            if len(Es) < minn_for_cmap:
                label = rf"$E, L = {E:g}, {L:g}$"
                color = f"C{ii}"
            else:
                label = None
                color = cm.plasma((E - Es[0]) / (Es[-1] - Es[0]))
            pyplot.subplot(gs[2])
            galpy_plot.plot(
                numpy.fabs(self._nforSn),
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
                ylabel=r"$|\mathrm{d}S_n/\mathrm{d}L|$",
            )
            if not overplot == gs:
                overplot = True
        if len(Es) < minn_for_cmap:
            if not overplot == gs:
                pyplot.subplot(gs[0])
                pyplot.legend(fontsize=17.0, frameon=False)
                pyplot.subplot(gs[1])
                pyplot.legend(fontsize=17.0, frameon=False)
                pyplot.subplot(gs[2])
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

    def plot_orbit(
        self,
        E,
        L,
        point=True,
        ls="-",
        xrange=None,
        yrange=None,
        include_isochrone_tori=False,
        orbit_only=False,
    ):
        """point=False: plot v_r^A vs. r^A, don't plot energy,
        include_isochrone_tori=True: include contours of constant action
                                     and angle for the auxiliary isochrone
                                     torus"""
        tar = numpy.linspace(0.0, 2.0 * numpy.pi, 1001)
        if not self._interp:
            # First find the torus for this energy and angular momentum
            indx = numpy.nanargmin(
                numpy.fabs(E - self._internal_Es) * numpy.fabs(L - self._internal_Ls)
            )
            if (
                numpy.fabs(E - self._Es[indx]) > 1e-10
                or numpy.fabs(L - self._internal_Ls[indx]) > 1e-10
            ):
                raise ValueError(
                    "Given energy and angular-momentum pair not found; please specify an energy/angular-momentum pair used in the initialization of the instance"
                )
            tJr = self._jr[indx]
        else:
            tJr = self.Jr(E)
        r, vr, _, _, _, _ = self(
            tJr, L, 0.0, tar, numpy.zeros_like(tar), numpy.zeros_like(tar), point=point
        )
        # First plot orbit in r,vr
        pyplot.subplot(1, 2 - orbit_only, 1)
        line2d = galpy_plot.plot(
            r, vr, xlabel=r"$r$", ylabel=r"$v_r$", gcf=True, ls=ls, lw=3.0
        )[0]
        if xrange is None:
            line2d.axes.autoscale(enable=True)
        else:
            pyplot.gca().set_xlim(xrange[0], xrange[1])
            pyplot.gca().set_ylim(yrange[0], yrange[1])
        if include_isochrone_tori:
            rmin, rmax = pyplot.gca().get_xlim()
            vrmin, vrmax = pyplot.gca().get_ylim()
            rr, vrr = numpy.meshgrid(
                numpy.linspace(rmin, rmax, 300), numpy.linspace(vrmin, vrmax, 300)
            )
            aAI = actionAngleIsochrone(
                ip=IsochronePotential(amp=self._amp[indx], b=self._b[indx])
            )
            jri, _, _, _, _, _, ari, _, _ = aAI.actionsFreqsAngles(
                rr,
                vrr,
                self._internal_Ls[indx] / rr,
                numpy.zeros_like(rr),
                numpy.zeros_like(rr),
                numpy.zeros_like(rr),
            )
            pyplot.contour(
                rr[0],
                vrr[:, 0],
                jri,
                colors="0.75",
                linewidths=0.9,
                levels=10.0
                ** numpy.linspace(numpy.log10(tJr) - 1.0, numpy.log10(tJr) + 1.0, 11),
            )
            """
            pyplot.contour(
                rr[0],
                vrr[:, 0],
                ari,
                colors="0.75",  # linestyles='dotted',
                linewidths=0.9,  # cmap='viridis',
                levels=numpy.linspace(
                    numpy.pi / 6.0, 2.0 * numpy.pi - numpy.pi / 6.0, 11
                ),
            )
            """
        if not point or orbit_only:
            return None
        # Then plot energy
        pyplot.subplot(1, 2, 2)
        Eorbit = (
            vr**2.0 / 2.0
            + L**2.0 / 2.0 / r**2.0
            + evaluatePotentials(self._pot, r, numpy.zeros_like(r))
        ) / E - 1.0
        ymin, ymax = numpy.amin(Eorbit), numpy.amax(Eorbit)
        galpy_plot.plot(
            tar,
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
        # Interpolate between the tori of the (L,E) grid, using as grid
        # coordinates (iL,ix), the (fractional) indices into the uniform grid
        # in L and into the uniform grid in the normalized energy
        # x = (E-E[R_L])/(E[Rinf,L]-E[R_L]); tori are looked up by (jr,L),
        # inverting the interpolated jr(iL,ix) for ix using Newton's method
        self._nnSn = self._nSn.shape[1]  # won't be confusing...
        shape2 = (self._nL, self._nE)
        iLs = numpy.arange(self._nL)
        ixs = numpy.arange(self._nE)

        def _rbs(arr):
            return interpolate.RectBivariateSpline(
                iLs, ixs, numpy.reshape(arr, shape2), kx=3, ky=3, s=0.0
            )

        self._jrInterp = _rbs(self._jr)
        self._EInterp = _rbs(self._internal_Es)
        self._OmegarInterp = _rbs(self._Omegar)
        self._OmegazInterp = _rbs(self._Omegaz)
        self._OmegazoverOmegarInterp = _rbs(self._OmegazoverOmegar)
        self._ampInterp = _rbs(self._amp)
        self._bInterp = _rbs(self._b)
        self._rperiInterp = _rbs(self._rperi)
        self._rapInterp = _rbs(self._rap)
        self._ptrperiInterp = _rbs(self._pt_rperi)
        self._ptrapInterp = _rbs(self._pt_rap)
        if self._pt_exact:
            self._dpsiapoInterp = _rbs(self._pt_dpsi_apo)
        # Interpolation of small, noisy coeffs doesn't work, so set to zero
        self._nSn[numpy.fabs(self._nSn) < 1e-16] = 0.0
        self._dSndJr[numpy.fabs(self._dSndJr) < 1e-15] = 0.0
        self._dSndLish[numpy.fabs(self._dSndLish) < 1e-15] = 0.0
        self._nSnFiltered = ndimage.spline_filter(
            numpy.reshape(self._nSn, shape2 + (self._nnSn,)), order=3
        )
        self._dSndJrFiltered = ndimage.spline_filter(
            numpy.reshape(self._dSndJr, shape2 + (self._nnSn,)), order=3
        )
        self._dSndLishFiltered = ndimage.spline_filter(
            numpy.reshape(self._dSndLish, shape2 + (self._nnSn,)), order=3
        )
        # Point transformation: reshape to (nL,nE,:) and filter, for direct
        # evaluation at fractional (iL,ix) grid coordinates
        if self._pt_exact:
            self._pt_filtered_interp = tuple(
                ndimage.spline_filter(
                    numpy.reshape(arr, shape2 + (arr.shape[1],)),
                    order=self._exact_pt_spl_deg,
                )
                for arr in (
                    self._pt_coeffs,
                    self._pt_deriv_coeffs,
                    self._pt_deriv2_coeffs,
                    self._pt_dpsi_coeffs,
                )
            )
        else:
            self._ptcoeffsFiltered = ndimage.spline_filter(
                numpy.reshape(self._pt_coeffs, shape2 + (self._pt_coeffs.shape[1],)),
                order=3,
            )
        return None

    def _gridcoords(self, jr, L):
        # Compute the (fractional) grid coordinates (iL,ix) of the torus
        # (jr,L), solving the interpolated jr(iL,ix) = jr for ix
        if L < self._Lmin or L > self._Lmax:
            raise ValueError(
                "Given angular momentum is outside of the interpolation grid"
            )
        iL = float((L - self._Lmin) / (self._Lmax - self._Lmin) * (self._nL - 1.0))
        ix = 0.5 * (self._nE - 1.0)
        for _ in range(100):
            djrdix = _evs(self._jrInterp, iL, ix, dy=1)
            dix = -(_evs(self._jrInterp, iL, ix) - jr) / djrdix
            ix = float(numpy.clip(ix + dix, 0.0, self._nE - 1.0))
            if numpy.fabs(dix) < 1e-12:
                break
        if numpy.fabs(_evs(self._jrInterp, iL, ix) - jr) > 1e-8 * (
            numpy.fabs(jr) + 1e-10
        ):
            raise ValueError("Given radial action is outside of the interpolation grid")
        return iL, ix

    def _coeffs_for_interp(self, filtered, iL, ix, order=3):
        # Evaluate a (nL,nE,ncoeff) spline-filtered array at fractional grid
        # coordinates (iL,ix), returning the interpolated coefficient row
        ncoeff = filtered.shape[2]
        return ndimage.map_coordinates(
            filtered,
            [
                iL * numpy.ones(ncoeff),
                ix * numpy.ones(ncoeff),
                numpy.arange(ncoeff, dtype="float"),
            ],
            order=order,
            prefilter=False,
            mode="mirror",
        )

    def _evaluate(self, jr, jphi, jz, angler, anglephi, anglez, **kwargs):
        """
        NAME:

           __call__

        PURPOSE:

           evaluate the phase-space coordinates (x,v) for a number of angles on a single torus

        INPUT:

           jr - radial action (scalar)

           jphi - azimuthal action (scalar)

           jz - vertical action (scalar)

           angler - radial angle (array [N])

           anglephi - azimuthal angle (array [N])

           anglez - vertical angle (array [N])

           tol= (object-wide value) goal for |dJ|/|J| along the torus

        OUTPUT:

           [R,vR,vT,z,vz,phi]

        HISTORY:

           2018-11-17 - Written - Bovy (UofT)

        """
        return self._xvFreqs(jr, jphi, jz, angler, anglephi, anglez, **kwargs)[:6]

    def _xvFreqs(self, jr, jphi, jz, angler, anglephi, anglez, point=True, **kwargs):
        """
        NAME:

           xvFreqs

        PURPOSE:

           evaluate the phase-space coordinates (x,v) for a number of angles on a single torus as well as the frequencies

        INPUT:

           jr - radial action (scalar)

           jphi - azimuthal action (scalar)

           jz - vertical action (scalar)

           angler - radial angle (array [N])

           anglephi - azimuthal angle (array [N])

           anglez - vertical angle (array [N])

           point= (True) if False, don't apply the point transformation, i.e., return (x^A,v^A)

        OUTPUT:

           ([R,vR,vT,z,vz,phi],OmegaR,Omegaphi,Omegaz)

        HISTORY:

           2018-11-17 - Written - Bovy (UofT)

           2020-05-22 - Started updating for point transformation - Bovy (UofT)

        """
        # Find torus
        if not self._interp:
            indx = numpy.nanargmin(numpy.fabs(jr - self._jr))
            if (
                numpy.fabs(jr - self._jr[indx]) > 1e-10
                or numpy.fabs(jz + numpy.fabs(jphi) - self._internal_Ls[indx]) > 1e-10
            ):
                raise ValueError(
                    "Given actions not found, to use interpolation, initialize with setup_interp=True"
                )
            tE = self._internal_Es[indx]
            tnSn = self._nSn[indx]
            tdSndJr = self._dSndJr[indx]
            tdSndLish = self._dSndLish[indx]
            tOmegazoverOmegar = self._OmegazoverOmegar[indx]
            tOmegar = self._Omegar[indx]
            tOmegaz = self._Omegaz[indx]
            isoaainv = actionAngleIsochroneInverse(
                ip=IsochronePotential(amp=self._amp[indx], b=self._b[indx])
            )
            tptcoeffs = self._pt_coeffs[indx]
            tptderivcoeffs = self._pt_deriv_coeffs[indx]
            tptderiv2coeffs = self._pt_deriv2_coeffs[indx]
            trperi = self._rperi[indx]
            trap = self._rap[indx]
            tptrperi = self._pt_rperi[indx]
            tptrap = self._pt_rap[indx]
        else:
            L = jz + numpy.fabs(jphi)
            iL, ix = self._gridcoords(jr, L)
            tE = _evs(self._EInterp, iL, ix)
            tnSn = self._coeffs_for_interp(self._nSnFiltered, iL, ix)
            tdSndJr = self._coeffs_for_interp(self._dSndJrFiltered, iL, ix)
            tdSndLish = self._coeffs_for_interp(self._dSndLishFiltered, iL, ix)
            tOmegazoverOmegar = _evs(self._OmegazoverOmegarInterp, iL, ix)
            tOmegar = _evs(self._OmegarInterp, iL, ix)
            tOmegaz = _evs(self._OmegazInterp, iL, ix)
            isoaainv = actionAngleIsochroneInverse(
                ip=IsochronePotential(
                    amp=_evs(self._ampInterp, iL, ix),
                    b=_evs(self._bInterp, iL, ix),
                )
            )
            trperi = _evs(self._rperiInterp, iL, ix)
            trap = _evs(self._rapInterp, iL, ix)
            tptrperi = _evs(self._ptrperiInterp, iL, ix)
            tptrap = _evs(self._ptrapInterp, iL, ix)
            if self._pt_exact:
                # Interpolated point transformation is evaluated directly at
                # the fractional (iL,ix) coordinates below
                tptcoeffs = tptderivcoeffs = tptderiv2coeffs = None
            else:
                tptcoeffs = self._coeffs_for_interp(self._ptcoeffsFiltered, iL, ix)
                tptderivcoeffs = polynomial.polyder(tptcoeffs, m=1)
                tptderiv2coeffs = polynomial.polyder(tptcoeffs, m=2)
        if self._pt_exact and self._pt_only:
            # For the exact point transformation, the generating-function
            # mapping (J,theta) -> (JA,thetaA) is the identity, so we can
            # skip solving for the auxiliary angles and action
            angler = numpy.atleast_1d(angler)
            anglera = copy.copy(angler)
            jra = jr * numpy.ones_like(angler)
            angleza = copy.copy(numpy.atleast_1d(anglez))
            anglephia = copy.copy(numpy.atleast_1d(anglephi))
        else:
            # First we need to solve for anglera
            angler = numpy.atleast_1d(angler)
            anglera = copy.copy(angler)
            # Now iterate Newton's method
            cntr = 0
            unconv = numpy.ones(len(angler), dtype="bool")
            tar = anglera + 2.0 * numpy.sum(
                tdSndJr * numpy.sin(self._nforSn * numpy.atleast_2d(anglera).T), axis=1
            )
            dtar = (tar - angler + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
            unconv[unconv] = numpy.fabs(dtar) > self._angle_tol
            # Don't allow too big steps
            maxdar = 2.0 * numpy.pi / 101
            while not self._bisect:
                danglear = 1.0 + 2.0 * numpy.sum(
                    self._nforSn
                    * tdSndJr
                    * numpy.cos(self._nforSn * numpy.atleast_2d(anglera[unconv]).T),
                    axis=1,
                )
                dtar = (tar[unconv] - angler[unconv] + numpy.pi) % (
                    2.0 * numpy.pi
                ) - numpy.pi
                dar = -dtar / danglear
                dar[numpy.fabs(dar) > maxdar] = (numpy.sign(dar) * maxdar)[
                    numpy.fabs(dar) > maxdar
                ]
                anglera[unconv] += dar
                unconv[unconv] = numpy.fabs(dtar) > self._angle_tol
                newtar = anglera[unconv] + 2.0 * numpy.sum(
                    tdSndJr
                    * numpy.sin(self._nforSn * numpy.atleast_2d(anglera[unconv]).T),
                    axis=1,
                )
                tar[unconv] = newtar
                cntr += 1
                if numpy.sum(unconv) == 0:
                    break
                if cntr > self._maxiter:
                    warnings.warn(
                        "Radial angle mapping with Newton-Raphson did not converge in {} iterations, falling back onto simple bisection (increase maxiter to try harder with Newton-Raphson)".format(
                            self._maxiter
                        ),
                        galpyWarning,
                    )
                    break
            # Fallback onto simple bisection in case of non-convergence
            if self._bisect or cntr > self._maxiter:
                # Reset cntr
                cntr = 0
                tryar_min = numpy.zeros(numpy.sum(unconv))
                dar = 2.0 * numpy.pi
                while True:
                    dar *= 0.5
                    anglera[unconv] = tryar_min + dar
                    newtar = (
                        anglera[unconv]
                        + 2.0
                        * numpy.sum(
                            tdSndJr
                            * numpy.sin(
                                self._nforSn * numpy.atleast_2d(anglera[unconv]).T
                            ),
                            axis=1,
                        )
                        + 2.0 * numpy.pi
                    ) % (2.0 * numpy.pi)
                    dtar = (newtar - angler[unconv] + numpy.pi) % (
                        2.0 * numpy.pi
                    ) - numpy.pi
                    tryar_min[newtar < angler[unconv]] = anglera[unconv][
                        newtar < angler[unconv]
                    ]
                    unconv[unconv] = numpy.fabs(dtar) > self._angle_tol
                    tryar_min = tryar_min[numpy.fabs(dtar) > self._angle_tol]
                    cntr += 1
                    if numpy.sum(unconv) == 0:
                        break
                    if cntr > self._maxiter:
                        warnings.warn(
                            "Radial angle mapping with bisection did not converge in {} iterations".format(
                                self._maxiter
                            )
                            + " for radial angles:"
                            + "".join(f" {k:g}" for k in sorted(set(angler[unconv]))),
                            galpyWarning,
                        )
                        break
            # Then compute the auxiliary action
            jra = jr + 2.0 * numpy.sum(
                tnSn * numpy.cos(self._nforSn * numpy.atleast_2d(anglera).T), axis=1
            )
            angleza = (
                anglez
                + tOmegazoverOmegar * (anglera - angler)
                - 2.0
                * numpy.sum(
                    tdSndLish * numpy.sin(self._nforSn * numpy.atleast_2d(anglera).T),
                    axis=1,
                )
            )
            anglephia = anglephi + numpy.sign(jphi) * (angleza - anglez)
        Ra, vRa, vTa, za, vza, phia = isoaainv(
            jra, jphi, jz, anglera, anglephia, angleza
        )
        if not point:
            return (
                Ra,
                vRa,
                vTa,
                za,
                vza,
                phia,
                tOmegar,
                numpy.sign(jphi) * tOmegaz,
                tOmegaz,
            )
        # Need to go back to orbital plane first...
        L = jz + numpy.fabs(jphi)  # total angular momentum
        lowerl = numpy.sqrt(1.0 - jphi**2.0 / L**2.0)
        ra = numpy.sqrt(Ra**2.0 + za**2.0)
        sinthetaa = Ra / ra
        costhetaa = za / ra
        vra = sinthetaa * vRa + costhetaa * vza
        vta = costhetaa * vRa - sinthetaa * vza
        # Also need to compute the angle psi in the orbital plane
        if lowerl <= 0.0:
            psia = phia
            asc = 0.0
        else:
            sinpsia = costhetaa / lowerl
            cospsia = -vta * ra * sinthetaa / L / lowerl
            psia = numpy.arctan2(sinpsia, cospsia)
            # Also compute the longitude of the ascending node Omega
            sinu = costhetaa / sinthetaa * jphi / L / lowerl
            pindx = (sinu > 1.0) * numpy.isfinite(sinu)
            sinu[pindx] = 1.0
            pindx = (sinu < -1.0) * numpy.isfinite(sinu)
            sinu[pindx] = -1.0
            u = numpy.arcsin(sinu)
            u[vta > 0.0] = numpy.pi - u[vta > 0.0]
            # In case we end up with a non-inclined orbit here, set u = phi (like in the if above)
            u[True ^ numpy.isfinite(u)] = phia[True ^ numpy.isfinite(u)]
            asc = phia - u
        # Now convert orbital-plane^A --> orbital-plane
        if self._pt_exact:
            yanorm = (ra - tptrperi) / (tptrap - tptrperi)
            # Delta-psi is stored as a function of the auxiliary radial angle
            # in [0,pi] (outgoing branch); the incoming branch follows from
            # its symmetry Delta-psi(2 pi - theta) = 2 Delta-psi(pi)
            #                                          - Delta-psi(theta)
            thetaafold = numpy.atleast_1d(anglera) % (2.0 * numpy.pi)
            dpsiflip = thetaafold > numpy.pi
            thetaafold[dpsiflip] = 2.0 * numpy.pi - thetaafold[dpsiflip]
            thnorm = thetaafold / numpy.pi
            if self._interp:

                def _pt_ev(arr3d, coord):
                    meshcoord = (
                        coord * (self._pt_nmesh - 1.0)
                        + (arr3d.shape[2] - self._pt_nmesh) / 2.0
                    )
                    return ndimage.map_coordinates(
                        arr3d,
                        [
                            iL * numpy.ones_like(coord),
                            ix * numpy.ones_like(coord),
                            meshcoord,
                        ],
                        order=self._exact_pt_spl_deg,
                        prefilter=False,
                        mode="mirror",
                    )

                yval = _pt_ev(self._pt_filtered_interp[0], yanorm)
                dyval = _pt_ev(self._pt_filtered_interp[1], yanorm)
                dpsival = _pt_ev(self._pt_filtered_interp[3], thnorm)
                tdpsiapo = _evs(self._dpsiapoInterp, iL, ix)
            else:
                yval = _ptra_eval(
                    yanorm,
                    indx,
                    self._pt_filtered[0],
                    self._pt_nmesh,
                    self._exact_pt_spl_deg,
                )
                dyval = _ptra_eval(
                    yanorm,
                    indx,
                    self._pt_filtered[1],
                    self._pt_nmesh,
                    self._exact_pt_spl_deg,
                )
                dpsival = _ptra_eval(
                    thnorm,
                    indx,
                    self._pt_filtered[3],
                    self._pt_nmesh,
                    self._exact_pt_spl_deg,
                )
                tdpsiapo = self._pt_dpsi_apo[indx]
            dpsival = numpy.where(dpsiflip, 2.0 * tdpsiapo - dpsival, dpsival)
            r = trperi + (trap - trperi) * yval
            piprime = (trap - trperi) / (tptrap - tptrperi) * dyval
            psi = psia + dpsival
        else:
            r = (trap - trperi) * polynomial.polyval(
                ((ra - tptrperi) / (tptrap - tptrperi)).T, tptcoeffs.T, tensor=False
            ).T + trperi
            piprime = (
                (trap - trperi)
                / (tptrap - tptrperi)
                * polynomial.polyval(
                    ((ra - tptrperi) / (tptrap - tptrperi)).T,
                    tptderivcoeffs.T,
                    tensor=False,
                ).T
            )
            psi = psia
        vr = vra * piprime
        vt = vta * ra / r  # conservation of angular momentum
        # and back from orbital-plane --> (x,v) in cyl coordinates
        cospsi = numpy.cos(psi)
        if lowerl > 0.0:
            sinpsi = numpy.sin(psi)
            costheta = sinpsi * lowerl
            sintheta = numpy.sqrt(1.0 - costheta**2.0)
        else:
            sintheta = sinthetaa
            costheta = costhetaa
        vtheta = -L * lowerl * cospsi / r / sintheta
        R = sintheta * r
        z = costheta * r
        if lowerl > 0.0:
            # Same as z / R / sqrt(L^2/jphi^2-1), but with the sign of jphi
            # included, as in the computation of the ascending node above
            sinu = z / R * jphi / L / lowerl
            pindx = (sinu > 1.0) * numpy.isfinite(sinu)
            sinu[pindx] = 1.0
            pindx = (sinu < -1.0) * numpy.isfinite(sinu)
            sinu[pindx] = -1.0
            u = numpy.arcsin(sinu)
            # Branch of u flips with the sign of the real polar velocity
            # vtheta, not the in-plane tangential velocity vt (they differ in
            # windows of width |Delta-psi| around psi = +/- pi/2)
            u[vtheta > 0.0] = numpy.pi - u[vtheta > 0.0]
            phi = asc + u
        else:
            phi = psi
        vR = vr * sintheta + vtheta * costheta
        vz = vr * costheta - vtheta * sintheta
        vT = jphi / R
        return (R, vR, vT, z, vz, phi, tOmegar, numpy.sign(jphi) * tOmegaz, tOmegaz)

    def _Freqs(self, jr, jphi, jz, **kwargs):
        """
        NAME:

           Freqs

        PURPOSE:

           return the frequencies corresponding to a torus

        INPUT:

           jr - radial action (scalar)

           jphi - azimuthal action (scalar)

           jz - vertical action (scalar)

        OUTPUT:

           (OmegaR,Omegaphi,Omegaz)

        HISTORY:

           2018-11-17 - Written - Bovy (UofT)

        """
        # Find torus
        if not self._interp:
            indx = numpy.nanargmin(numpy.fabs(jr - self._jr))
            if (
                numpy.fabs(jr - self._jr[indx]) > 1e-10
                or numpy.fabs(jz + numpy.fabs(jphi) - self._Ls[indx]) > 1e-10
            ):
                raise ValueError(
                    "Given actions not found, to use interpolation, initialize with setup_interp=True"
                )
            tOmegar = self._Omegar[indx]
            tOmegaz = self._Omegaz[indx]
        else:
            L = jz + numpy.fabs(jphi)
            iL, ix = self._gridcoords(jr, L)
            tOmegar = _evs(self._OmegarInterp, iL, ix)
            tOmegaz = _evs(self._OmegazInterp, iL, ix)
        return (tOmegar, numpy.sign(jphi) * tOmegaz, tOmegaz)


def _ptra_eval(yanorm, rowcoord, pt_filtered_arr, pt_nmesh, pt_spl_deg):
    """
    NAME:
       _ptra_eval
    PURPOSE:
       evaluate the exact point transformation (or one of its derivatives or
       its Delta-psi shift) with respect to the normalized coordinate
       ya = (ra-ptrperi)/(ptrap-ptrperi), using 2D spline interpolation of
       the (torus,mesh) grid on which it is stored
    INPUT:
       yanorm - normalized radial position(s) on the auxiliary torus
       rowcoord - (possibly fractional) row index of the torus of each
                  evaluation point in the grid of tori; scalars are
                  broadcast against yanorm
       pt_filtered_arr - spline-filtered (torus,mesh) grid of the normalized
                         mapping (or of one of its derivatives, or of
                         Delta-psi) sampled on the fixed padded mesh
       pt_nmesh - number of core (unpadded) mesh points
       pt_spl_deg - degree of the interpolating spline (must match the order
                    used to filter pt_filtered_arr)
    OUTPUT:
       the normalized mapping (or derivative/Delta-psi) at yanorm
    HISTORY:
       2026-08-14 - Written - Bovy (UofT)
    """
    yanorm = numpy.atleast_1d(numpy.asarray(yanorm, dtype="float"))
    rowcoord = numpy.broadcast_to(numpy.asarray(rowcoord, dtype="float"), yanorm.shape)
    meshcoord = yanorm * (pt_nmesh - 1.0) + (pt_filtered_arr.shape[1] - pt_nmesh) / 2.0
    return ndimage.map_coordinates(
        pt_filtered_arr,
        [rowcoord.reshape(-1), meshcoord.reshape(-1)],
        order=pt_spl_deg,
        prefilter=False,
        mode="mirror",
    ).reshape(yanorm.shape)


def _anglera(
    ra,
    E,
    L,
    L2,
    pot,
    isoaa_helper,
    ptcoeffs,
    ptderivcoeffs,
    rperi,
    rap,
    ptrperi,
    ptrap,
    vrneg=False,
    pt_exact=False,
    pt_filtered=None,
    pt_nmesh=None,
    pt_spl_deg=5,
):
    """
    NAME:
       _anglera
    PURPOSE:
       Compute the auxiliary radial angle in the isochrone potential for a grid in r and E, including the point transformation
    INPUT:
       ra - radial position
       E - energy
       L - angular momentum
       L2 - angular momentum squared
       pot - the potential
       isoaa_helper - _actionAngleIsochroneHelper instance for isochrone action-angle calculations
       ptcoeffs - coefficients of the polynomial point transformation
       ptderivcoeffs - coefficients of the derivative of the polynomial point transformation
       rperi, rap - peri- and apocenter of the true torus
       ptrperi, ptrap - peri- and apocenter of the point-transformed torus
       vrneg= (False) True if vr is negative
       pt_exact= (False) if True, use the exact point transformation instead of the polynomial approximation
       exact_pt_spl_deg= (5) degree of the spline to use for the exact point transformation
    OUTPUT:
       auxiliary radial angles
    HISTORY:
       2020-05-22 - Written based on earlier code - Bovy (UofT)
    """
    yanorm = (ra - ptrperi) / (ptrap - ptrperi)
    # Compute vr
    if pt_exact:
        r = rperi + (rap - rperi) * _ptra_eval(
            yanorm, ptcoeffs, pt_filtered[0], pt_nmesh, pt_spl_deg
        )
    else:
        r = (rap - rperi) * polynomial.polyval(
            yanorm.T, ptcoeffs.T, tensor=False
        ).T + rperi
    vr2 = 2.0 * (E - evaluatePotentials(pot, r, numpy.zeros_like(r))) - L2 / r**2.0
    vr2[vr2 < 0.0] = 0.0
    if pt_exact:
        piprime = (
            (rap - rperi)
            / (ptrap - ptrperi)
            * _ptra_eval(yanorm, ptcoeffs, pt_filtered[1], pt_nmesh, pt_spl_deg)
        )
    else:
        piprime = (
            (rap - rperi)
            / (ptrap - ptrperi)
            * polynomial.polyval(yanorm.T, ptderivcoeffs.T, tensor=False).T
        )
    return isoaa_helper.angler(ra, vr2 * piprime**-2.0, L, reuse=False, vrneg=vrneg)


def _danglera(
    ra,
    E,
    L,
    L2,
    pot,
    isoaa_helper,
    ptcoeffs,
    ptderivcoeffs,
    ptderiv2coeffs,
    rperi,
    rap,
    ptrperi,
    ptrap,
    vrneg=False,
    pt_exact=False,
    pt_filtered=None,
    pt_nmesh=None,
    pt_spl_deg=5,
):
    """
    NAME:
       _danglera
    PURPOSE:
       Compute the derivative of the auxiliary radial angle in the isochrone potential for a grid in r and E, including the point transformation wrt ra
    INPUT:
       ra - radial position
       E - energy
       L - angular momentum
       L2 - angular momentum squared
       pot - the potential
       isoaa_helepr - _actionAngleIsochroneHelper instance for isochrone action-angle calculations
       ptcoeffs - coefficients of the polynomial point transformation
       ptderivcoeffs - coefficients of the derivative of the polynomial point transformation
       ptderiv2coeffs - coefficients of the second derivative of the polynomial point transformation
       rperi, rap - peri- and apocenter of the true torus
       ptrperi, ptrap - peri- and apocenter of the point-transformed torus
       vrneg= (False) True if vr is negative
       pt_exact= (False) if True, use the exact point transformation instead of the polynomial approximation
       exact_pt_spl_deg= (5) degree of the spline to use for the exact point transformation
    OUTPUT:
       auxiliary radial angles
    HISTORY:
       2020-05-22 - Written based on earlier code - Bovy (UofT)
    """
    yanorm = (ra - ptrperi) / (ptrap - ptrperi)
    # Compute vr
    if pt_exact:
        r = rperi + (rap - rperi) * _ptra_eval(
            yanorm, ptcoeffs, pt_filtered[0], pt_nmesh, pt_spl_deg
        )
        piprime = (
            (rap - rperi)
            / (ptrap - ptrperi)
            * _ptra_eval(yanorm, ptcoeffs, pt_filtered[1], pt_nmesh, pt_spl_deg)
        )
        piprime2 = (
            (rap - rperi)
            / (ptrap - ptrperi) ** 2.0
            * _ptra_eval(yanorm, ptcoeffs, pt_filtered[2], pt_nmesh, pt_spl_deg)
        )
    else:
        r = (rap - rperi) * polynomial.polyval(
            yanorm.T, ptcoeffs.T, tensor=False
        ).T + rperi
        piprime = (
            (rap - rperi)
            / (ptrap - ptrperi)
            * polynomial.polyval(yanorm.T, ptderivcoeffs.T, tensor=False).T
        )
        piprime2 = (
            (rap - rperi)
            / (ptrap - ptrperi) ** 2.0
            * polynomial.polyval(yanorm.T, ptderiv2coeffs.T, tensor=False).T
        )
    vr2 = 2.0 * (E - evaluatePotentials(pot, r, numpy.zeros_like(r))) - L2 / r**2.0
    vr2[vr2 < 0.0] = 0.0
    dEadra = (
        -piprime2 * piprime**-3.0 * vr2
        + (L2 * r**-3.0 + evaluateRforces(pot, r, numpy.zeros_like(r))) / piprime
        - L2 * ra**-3.0
        - evaluateRforces(isoaa_helper._ip, ra, numpy.zeros_like(ra))
    )
    return isoaa_helper.danglerdr_constant_L(
        ra, vr2 * piprime**-2.0, L, dEadra, vrneg=vrneg
    )


def _jraora(
    ra,
    E,
    L,
    L2,
    pot,
    isoaa_helper,
    ptcoeffs,
    ptderivcoeffs,
    rperi,
    rap,
    ptrperi,
    ptrap,
    pt_exact=False,
    pt_filtered=None,
    pt_nmesh=None,
    pt_spl_deg=5,
):
    """
    NAME:
       _jraora
    PURPOSE:
       Compute the auxiliary radial action and frequency in the isochrone potential for a grid in r and E, including the point transformation
    INPUT:
       ra - radial position
       E - energy
       L - angular momentum
       L2 - angular momentum squared
       pot - the potential
       isoaa_helepr - _actionAngleIsochroneHelper instance for isochrone action-angle calculations
       ptcoeffs - coefficients of the polynomial point transformation
       ptderivcoeffs - coefficients of the derivative of the polynomial point transformation
       rperi, rap - peri- and apocenter of the true torus
       ptrperi, ptrap - peri- and apocenter of the point-transformed torus
       pt_exact= (False) if True, use the exact point transformation instead of the polynomial approximation
         exact_pt_spl_deg= (5) degree of the spline to use for the exact point transformation
    OUTPUT:
       auxiliary radial actions, auxiliary radial frequencies
    HISTORY:
       2020-05-23 - Written based on earlier code - Bovy (UofT)
    """
    yanorm = (ra - ptrperi) / (ptrap - ptrperi)
    # Compute vr
    if pt_exact:
        r = rperi + (rap - rperi) * _ptra_eval(
            yanorm, ptcoeffs, pt_filtered[0], pt_nmesh, pt_spl_deg
        )
        piprime = (
            (rap - rperi)
            / (ptrap - ptrperi)
            * _ptra_eval(yanorm, ptcoeffs, pt_filtered[1], pt_nmesh, pt_spl_deg)
        )
    else:
        r = (rap - rperi) * polynomial.polyval(
            yanorm.T, ptcoeffs.T, tensor=False
        ).T + rperi
        piprime = (
            (rap - rperi)
            / (ptrap - ptrperi)
            * polynomial.polyval(yanorm.T, ptderivcoeffs.T, tensor=False).T
        )
    vr2 = 2.0 * (E - evaluatePotentials(pot, r, numpy.zeros_like(r))) - L2 / r**2.0
    vr2[vr2 < 0.0] = 0.0
    Ea = 0.5 * (vr2 * piprime**-2.0 + L2 * ra**-2.0) + isoaa_helper._ip(
        ra, numpy.zeros_like(ra)
    )
    return isoaa_helper.Jr(Ea, L), isoaa_helper.Or(Ea), Ea


def _djradjrLish(
    ra,
    E,
    L,
    L2,
    Omegar,
    pot,
    isoaa_helper,
    ptcoeffs,
    ptderivcoeffs,
    ptderiv2coeffs,
    rperi,
    rap,
    ptrperi,
    ptrap,
    pt_exact=False,
    pt_filtered=None,
    pt_nmesh=None,
    pt_spl_deg=5,
):
    """
    NAME:
       _djradjrLish
    PURPOSE:
       Compute the derivative of the auxiliary radial action with respect to radial action and angular momentum-ish for a grid
    INPUT:
       ra - radial position
       E - energy
       L - angular momentum
       L2 - angular momentum squared
       Omegar - radial frequency
       pot - the potential
       isoaa_helepr - _actionAngleIsochroneHelper instance for isochrone action-angle calculations
       ptcoeffs - coefficients of the polynomial point transformation
       ptderivcoeffs - coefficients of the derivative of the polynomial point transformation
       ptderiv2coeffs - coefficients of the second derivative of the polynomial point transformation
       rperi, rap - peri- and apocenter of the true torus
       ptrperi, ptrap - peri- and apocenter of the point-transformed torus
       pt_exact= (False) if True, use the exact point transformation instead of the polynomial approximation
       exact_pt_spl_deg= (5) degree of the spline to use for the exact point transformation
    OUTPUT:
       derivative of the auxiliary radial actions wrt radial action and angular momentum
    HISTORY:
       2020-05-23 - Written based on earlier code - Bovy (UofT)
    """
    yanorm = (ra - ptrperi) / (ptrap - ptrperi)
    # Compute vr
    if pt_exact:
        r = rperi + (rap - rperi) * _ptra_eval(
            yanorm, ptcoeffs, pt_filtered[0], pt_nmesh, pt_spl_deg
        )
        piprime = (
            (rap - rperi)
            / (ptrap - ptrperi)
            * _ptra_eval(yanorm, ptcoeffs, pt_filtered[1], pt_nmesh, pt_spl_deg)
        )
        piprime2 = (
            (rap - rperi)
            / (ptrap - ptrperi) ** 2.0
            * _ptra_eval(yanorm, ptcoeffs, pt_filtered[2], pt_nmesh, pt_spl_deg)
        )
    else:
        r = (rap - rperi) * polynomial.polyval(
            yanorm.T, ptcoeffs.T, tensor=False
        ).T + rperi
        piprime = (
            (rap - rperi)
            / (ptrap - ptrperi)
            * polynomial.polyval(yanorm.T, ptderivcoeffs.T, tensor=False).T
        )
        piprime2 = (
            (rap - rperi)
            / (ptrap - ptrperi) ** 2.0
            * polynomial.polyval(yanorm.T, ptderiv2coeffs.T, tensor=False).T
        )
    vr2 = 2.0 * (E - evaluatePotentials(pot, r, numpy.zeros_like(r))) - L2 / r**2.0
    vr2[vr2 < 0.0] = 0.0
    Ea = 0.5 * (vr2 * piprime**-2.0 + L2 * ra**-2.0) + isoaa_helper._ip(
        ra, numpy.zeros_like(ra)
    )
    dEadra = (
        -piprime2 * piprime**-3.0 * vr2
        + (L2 * r**-3.0 + evaluateRforces(pot, r, numpy.zeros_like(r))) / piprime
        - L2 * ra**-3.0
        - evaluateRforces(isoaa_helper._ip, ra, numpy.zeros_like(ra))
    )
    pardEapardL = L * (ra**-2.0 - r**-2.0) * (1 - pt_exact)
    Ora = isoaa_helper.Or(Ea)
    # Compute dEA/dE and dEA/dL for dJr^A/d(Jr,L)
    dradE, dradL = isoaa_helper.drdEL_constant_angler(
        ra, vr2 * piprime**-2.0, Ea, L, dEadra, pardEapardL
    )
    dEadE = dradE * dEadra + 1.0
    dEadL = dradL * dEadra + pardEapardL
    djradjr = Omegar / Ora * dEadE
    djradLish = dEadL / Ora
    return djradjr, djradLish, dEadE, dEadL
