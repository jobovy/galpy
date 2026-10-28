###############################################################################
#   diskdf.py: module that interprets (E,Lz) pairs in terms of a
#              distribution function (following Dehnen 1999)
#
#   This module contains the following classes:
#
#      diskdf - top-level class that represents a distribution function
#      dehnendf - inherits from diskdf, implements Dehnen's 'new' DF
#      shudf - inherits from diskdf, implements Shu's DF
#      DFcorrection - class that represents corrections to the input Sigma(R)
#                     and sigma_R(R) to get closer to the targets
###############################################################################
_EPSREL = 10.0**-14.0
_NSIGMA = 4.0
_INTERPDEGREE = 3
_RMIN = 10.0**-10.0
_MAXD_REJECTLOS = 4.0
_PROFILE = False
_NQUAD = 50  # fixed Gauss-Legendre order for the backend (jax/torch) moment path
_NQUAD_CORRECTED = 100  # the correction splines' knots: 50 -> 4e-9, 100 -> 5e-10
import copy
import functools
import os
import os.path
import pickle

import numpy
import scipy

numpylog = numpy.lib.scimath.log  # somehow, this code produces log(negative), which scipy (now numpy.lib.scimath.log) implements as log(|negative|) + i pi while numpy gives NaN and we want the scipy behavior; not sure where the log(negative) comes from though! I think it's for sigma=0 DFs (this test fails with numpy.log) where the DF eval has a log(~zero) that can be slightly negative because of numerical precision issues
from scipy import integrate, interpolate

from ..actionAngle import actionAngleAdiabatic
from ..backend import (
    as_numpy,
    asarray_on_device,
    backend_input,
    coerce_coords,
    device_of,
    float64_default_if_torch_args,
    get_namespace,
    is_backend_array,
    prefer_backend_namespace,
)
from ..backend import random as grandom
from ..backend import resolve_namespace, scalar_like, to_host, use
from ..backend._namespaces import (
    has_concrete_truth_value,
    namespace_for_name,
    namespace_from_arrays,
    requires_backend_grad,
    under_trace,
)
from ..backend.interpolate import cubic_spline_coeffs, eval_ppoly, spline_to_ppoly
from ..backend.quadrature import nested_quad
from ..backend.sampling import (
    batched_inverse_cdf_sample,
    ensure_strictly_increasing,
    linear_inverse_cdf_sample,
    trapezoid_cdf,
    trapezoid_sum,
)
from ..orbit import Orbit
from ..potential import PowerSphericalPotential
from ..util import conversion, quadpack, save_pickles
from ..util.conversion import (
    _APY_LOADED,
    _APY_UNITS,
    physical_conversion,
    potential_physical_input,
    surfdens_in_msolpc2,
)
from .df import df
from .surfaceSigmaProfile import expSurfaceSigmaProfile, surfaceSigmaProfile

if _APY_LOADED:
    from astropy import units
# scipy version
from packaging.version import parse as parse_version

_SCIPY_VERSION = parse_version(scipy.__version__)
_SCIPY_VERSION_BREAK = parse_version("0.9")
_CORRECTIONSDIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
_DEGTORAD = numpy.pi / 180.0


def _loop_array_R(method):
    """scipy's dblquad takes a scalar R: evaluate a numpy array R elementwise
    (a differentiated profile takes the vectorized backend path instead)"""

    @functools.wraps(method)
    def wrapper(self, R, *args, **kwargs):
        if (
            isinstance(R, numpy.ndarray)
            and R.ndim > 0
            and not self._profile_differentiated()
        ):
            return numpy.array(
                [method(self, r, *args, **kwargs) for r in R.flat]
            ).reshape(R.shape)
        return method(self, R, *args, **kwargs)

    return wrapper


def _on_numpy(method):
    """A scipy/numpy-RNG method: plain inputs stay on numpy under a forced backend
    (the entry points it calls would otherwise coerce them onto it)"""

    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        with use("numpy", force=True):
            return method(*args, **kwargs)

    return wrapper


def _sqrt(x):
    """numpy.sqrt, or the backend's for a backend-array (e.g. a differentiated beta)"""
    if is_backend_array(x):
        return namespace_from_arrays((x,)).sqrt(x)
    return numpy.sqrt(x)


def _beta_branch(beta, at0, other):
    """at0() if beta == 0 else other(beta), for a beta that may be a backend
    array. Traced, both run (other() at a masked beta) and xp.where picks. The
    branches use different potential zero points, so AD at exactly beta == 0
    misses the branch's own beta dependence."""
    if not is_backend_array(beta):
        return at0() if beta == 0.0 else other(beta)
    zero = beta == 0.0
    if has_concrete_truth_value(zero):
        return at0() if bool(zero) else other(beta)
    xp = namespace_from_arrays((beta,))
    a, o = at0(), other(xp.where(zero, xp.ones_like(beta), beta))
    if isinstance(a, tuple):
        return tuple(xp.where(zero, x, y) for x, y in zip(a, o))
    return xp.where(zero, a, o)


class diskdf(df):
    """Class that represents a disk DF"""

    _backend_compatible = True

    def __init__(
        self,
        dftype="dehnen",
        surfaceSigma=expSurfaceSigmaProfile,
        profileParams=(1.0 / 3.0, 1.0, 0.2),
        correct=False,
        beta=0.0,
        ro=None,
        vo=None,
        **kwargs,
    ):
        """
        Initialize a DF

        Parameters
        ----------
        dftype : str, optional
            'dehnen' or 'corrected-dehnen', 'shu' or 'corrected-shu'
        surfaceSigma : instance or class name of the target surface density and sigma_R profile, optional
            (default: both exponential)
        profileParams : tuple, optional
            parameters of the surface and sigma_R profile: (xD,xS,Sro) where
              * xD - disk surface mass scalelength / Ro
              * xS - disk velocity dispersion scalelength / Ro
              * Sro - disk velocity dispersion at Ro (/vo)
            Directly given to the 'surfaceSigmaProfile class, so could be anything that class takes
        beta : float, optional
            power-law index of the rotation curve
        correct : bool, optional
            correct the DF (i.e., DFcorrection kwargs are also given)
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).
        **kwargs : dict, optional
            DFcorrection kwargs (except for those already specified)

        Notes
        -----
        - 2010-03-10 - Written - Bovy (NYU)
        """
        df.__init__(self, ro=ro, vo=vo)
        self._dftype = dftype
        if isinstance(surfaceSigma, surfaceSigmaProfile):
            self._surfaceSigmaProfile = surfaceSigma
        else:
            if _APY_LOADED and isinstance(profileParams[0], units.Quantity):
                newprofileParams = (
                    conversion.parse_length(profileParams[0], ro=self._ro),
                    conversion.parse_length(profileParams[1], ro=self._ro),
                    conversion.parse_velocity(profileParams[2], vo=self._vo),
                )
                self._roSet = True
                self._voSet = True
                profileParams = newprofileParams
            self._surfaceSigmaProfile = surfaceSigma(profileParams)
        self._beta = beta
        self._gamma = _sqrt(2.0 / (1.0 + self._beta))
        if (
            correct
            or "corrections" in kwargs
            or "rmax" in kwargs
            or "niter" in kwargs
            or "npoints" in kwargs
        ):
            self._correct = True
            # Load corrections
            self._corr = DFcorrection(
                dftype=self.__class__,
                surfaceSigmaProfile=self._surfaceSigmaProfile,
                beta=beta,
                **kwargs,
            )
        else:
            self._correct = False
        return None

    @functools.cached_property
    def _psp(self):
        # lazy: only the numpy samplers use it (and a traced beta cannot build it)
        return PowerSphericalPotential(
            normalize=1.0, alpha=2.0 - 2.0 * self._beta
        ).toPlanar()

    @functools.cached_property
    def _aA(self):
        # aA object for frequency and rap,rperi calculation
        return actionAngleAdiabatic(pot=self._psp, gamma=0.0)

    @physical_conversion("phasespacedensity2d", pop=True)
    def __call__(self, *args, **kwargs):
        """
        Evaluate the distribution function

        Parameters
        ----------
        *args : tuple
            Either:
                1) Orbit instance or list:
                    a) Orbit instance alone: use initial condition
                    b) Orbit instance + t: call the Orbit instance (for list, each instance is called at t)
                2) E,L - energy (/vo^2; or can be Quantity) and angular momentun (/ro/vo; or can be Quantity)
                3) array vxvv [3/4,nt] [must be in natural units /vo,/ro; use Orbit interface for physical-unit input]
        marginalizeVperp : bool, optional
            marginalize over perpendicular velocity (only supported with 1a) for single orbits above)
        marginalizeVlos : bool, optional
            marginalize over line-of-sight velocity (only supported with 1a) for single orbits above)
        nsigma : float, optional
            number of sigma to integrate over when marginalizing
        **kwargs: dict, optional
            scipy.integrate.quad keywords

        Returns
        -------
        float or numpy.ndarray
            value of DF

        Notes
        -----
        - 2010-07-10 - Written - Bovy (NYU)
        """
        if isinstance(args[0], Orbit):
            # _stored=True: the orbit's own coordinates (numpy for a numpy
            # orbit), not the accessors' forced-backend lift
            if len(args[0]) > 1:
                raise RuntimeError(
                    "Only single-object Orbit instances can be passed to DF instances at this point"
                )  # pragma: no cover
            if len(args) == 1:
                if kwargs.pop("marginalizeVperp", False):
                    return self._call_marginalizevperp(args[0], **kwargs)
                elif kwargs.pop("marginalizeVlos", False):
                    return self._call_marginalizevlos(args[0], **kwargs)
                else:
                    return self._real(
                        self.eval(
                            *vRvTRToEL(
                                args[0].vR(use_physical=False, _stored=True),
                                args[0].vT(use_physical=False, _stored=True),
                                args[0].R(use_physical=False, _stored=True),
                                self._beta,
                                self._dftype,
                            )
                        )
                    )
            else:
                no = args[0](args[1])
                return self._real(
                    self.eval(
                        *vRvTRToEL(
                            no.vR(use_physical=False, _stored=True),
                            no.vT(use_physical=False, _stored=True),
                            no.R(use_physical=False, _stored=True),
                            self._beta,
                            self._dftype,
                        )
                    )
                )
        elif isinstance(args[0], list) and isinstance(args[0][0], Orbit):
            if numpy.any([len(no) > 1 for no in args[0]]):
                raise RuntimeError(
                    "Only single-object Orbit instances can be passed to DF instances at this point"
                )  # pragma: no cover
            # Grab all of the vR, vT, and R
            vR = numpy.array([o.vR(use_physical=False, _stored=True) for o in args[0]])
            vT = numpy.array([o.vT(use_physical=False, _stored=True) for o in args[0]])
            R = numpy.array([o.R(use_physical=False, _stored=True) for o in args[0]])
            return self._real(
                self.eval(*vRvTRToEL(vR, vT, R, self._beta, self._dftype))
            )
        elif isinstance(args[0], numpy.ndarray) and not (
            hasattr(args[0], "isscalar") and args[0].isscalar
        ):
            # Grab all of the vR, vT, and R
            vR = args[0][1]
            vT = args[0][2]
            R = args[0][0]
            return self._real(
                self.eval(*vRvTRToEL(vR, vT, R, self._beta, self._dftype))
            )
        else:
            return self._real(self.eval(*args))

    @staticmethod
    def _real(x):
        # backend arrays are already real; numpy path unwraps scimath's +i*pi
        return x if is_backend_array(x) else numpy.real(x)

    @_on_numpy
    def _call_marginalizevperp(self, o, **kwargs):
        """Call the DF, marginalizing over perpendicular velocity"""
        # Get l, vlos
        # a scipy marginalization below: read the orbit on the host
        l = to_host(o.ll(obs=[1.0, 0.0, 0.0], ro=1.0)) * _DEGTORAD
        vlos = to_host(o.vlos(ro=1.0, vo=1.0, obs=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        R = to_host(o.R(use_physical=False, _stored=True))
        phi = to_host(o.phi(use_physical=False, _stored=True))
        # Get local circular velocity, projected onto the los
        vcirc = R**self._beta
        vcirclos = vcirc * numpy.sin(phi + l)
        # Marginalize
        alphalos = phi + l
        if not "nsigma" in kwargs or ("nsigma" in kwargs and kwargs["nsigma"] is None):
            nsigma = _NSIGMA
        else:
            nsigma = kwargs["nsigma"]
        kwargs.pop("nsigma", None)
        sigmaR2 = self._ssp("sigma2", R)
        sigmaR1 = numpy.sqrt(sigmaR2)
        # Use the asymmetric drift equation to estimate va
        va = (
            sigmaR2
            / 2.0
            / R**self._beta
            * (
                1.0 / self._gamma**2.0
                - 1.0
                - R * self._ssp("surfacemassDerivative", R, log=True)
                - R * self._ssp("sigma2Derivative", R, log=True)
            )
        )
        if numpy.fabs(va) > sigmaR1:
            va = 0.0  # To avoid craziness near the center
        if numpy.fabs(numpy.sin(alphalos)) < numpy.sqrt(1.0 / 2.0):
            cosalphalos = numpy.cos(alphalos)
            tanalphalos = numpy.tan(alphalos)
            return (
                integrate.quad(
                    _marginalizeVperpIntegrandSinAlphaSmall,
                    -self._gamma * va / sigmaR1 - nsigma,
                    -self._gamma * va / sigmaR1 + nsigma,
                    args=(
                        self,
                        R,
                        cosalphalos,
                        tanalphalos,
                        vlos - vcirclos,
                        vcirc,
                        sigmaR1 / self._gamma,
                    ),
                    **kwargs,
                )[0]
                / numpy.fabs(cosalphalos)
                * sigmaR1
                / self._gamma
            )
        else:
            sinalphalos = numpy.sin(alphalos)
            cotalphalos = 1.0 / numpy.tan(alphalos)
            return (
                integrate.quad(
                    _marginalizeVperpIntegrandSinAlphaLarge,
                    -nsigma,
                    nsigma,
                    args=(
                        self,
                        R,
                        sinalphalos,
                        cotalphalos,
                        vlos - vcirclos,
                        vcirc,
                        sigmaR1,
                    ),
                    **kwargs,
                )[0]
                / numpy.fabs(sinalphalos)
                * sigmaR1
            )

    @_on_numpy
    def _call_marginalizevlos(self, o, **kwargs):
        """Call the DF, marginalizing over line-of-sight velocity"""
        # Get d, l, vperp
        # a scipy marginalization below: read the orbit on the host
        l = to_host(o.ll(obs=[1.0, 0.0, 0.0], ro=1.0)) * _DEGTORAD
        vperp = to_host(o.vll(ro=1.0, vo=1.0, obs=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        R = to_host(o.R(use_physical=False, _stored=True))
        phi = to_host(o.phi(use_physical=False, _stored=True))
        # Get local circular velocity, projected onto the perpendicular
        # direction
        vcirc = R**self._beta
        vcircperp = vcirc * numpy.cos(phi + l)
        # Marginalize
        alphaperp = numpy.pi / 2.0 + phi + l
        if not "nsigma" in kwargs or ("nsigma" in kwargs and kwargs["nsigma"] is None):
            nsigma = _NSIGMA
        else:
            nsigma = kwargs["nsigma"]
        kwargs.pop("nsigma", None)
        sigmaR2 = self._ssp("sigma2", R)
        sigmaR1 = numpy.sqrt(sigmaR2)
        # Use the asymmetric drift equation to estimate va
        va = (
            sigmaR2
            / 2.0
            / R**self._beta
            * (
                1.0 / self._gamma**2.0
                - 1.0
                - R * self._ssp("surfacemassDerivative", R, log=True)
                - R * self._ssp("sigma2Derivative", R, log=True)
            )
        )
        if numpy.fabs(va) > sigmaR1:
            va = 0.0  # To avoid craziness near the center
        if numpy.fabs(numpy.sin(alphaperp)) < numpy.sqrt(1.0 / 2.0):
            cosalphaperp = numpy.cos(alphaperp)
            tanalphaperp = numpy.tan(alphaperp)
            # we can reuse the VperpIntegrand, since it is just another angle
            return (
                integrate.quad(
                    _marginalizeVperpIntegrandSinAlphaSmall,
                    -self._gamma * va / sigmaR1 - nsigma,
                    -self._gamma * va / sigmaR1 + nsigma,
                    args=(
                        self,
                        R,
                        cosalphaperp,
                        tanalphaperp,
                        vperp - vcircperp,
                        vcirc,
                        sigmaR1 / self._gamma,
                    ),
                    **kwargs,
                )[0]
                / numpy.fabs(cosalphaperp)
                * sigmaR1
                / self._gamma
            )
        else:
            sinalphaperp = numpy.sin(alphaperp)
            cotalphaperp = 1.0 / numpy.tan(alphaperp)
            # we can reuse the VperpIntegrand, since it is just another angle
            return (
                integrate.quad(
                    _marginalizeVperpIntegrandSinAlphaLarge,
                    -nsigma,
                    nsigma,
                    args=(
                        self,
                        R,
                        sinalphaperp,
                        cotalphaperp,
                        vperp - vcircperp,
                        vcirc,
                        sigmaR1,
                    ),
                    **kwargs,
                )[0]
                / numpy.fabs(sinalphaperp)
                * sigmaR1
            )

    @potential_physical_input
    @physical_conversion("velocity2", pop=True)
    @backend_input("R")
    def targetSigma2(self, R, log=False):
        """
        Evaluate the target Sigma_R^2(R)

        Parameters
        ----------
        R : float or Quantity
            Radius at which to evaluate.
        log : bool, optional
            If True, return the log (default: False).

        Returns
        -------
        float
            Target Sigma_R^2(R).

        Notes
        -----
        - 2010-03-28 - Written - Bovy (NYU)
        """
        return self._ssp("sigma2", R, log=log)

    @potential_physical_input
    @physical_conversion("surfacedensity", pop=True)
    @backend_input("R")
    def targetSurfacemass(self, R, log=False):
        """
        Evaluate the target surface mass at R.

        Parameters
        ----------
        R : float or Quantity
            Radius at which to evaluate.
        log : bool, optional
            If True, return the log (default: False).

        Returns
        -------
        float or Quantity
            Target surface mass at R.

        Notes
        -----
        - 2010-03-28 - Written - Bovy (NYU)

        """
        return self._ssp("surfacemass", R, log=log)

    # --- surfaceSigmaProfile boundary -------------------------------------
    # surfaceSigmaProfile is backend-native and its @backend_input boundary
    # coerces R under a FORCED context, so a numpy caller gets a Tensor back.
    # diskdf is NOT migrated: its numpy branches feed scipy quad and numpylog,
    # which cannot take Tensors (numpylog's complex result promotes against a
    # Tensor into a complex Tensor that _real cannot unwrap). Until diskdf
    # itself is migrated, the CONSUMER guards its own boundary here -- every
    # call into the profile goes through one of these four, so the guard cannot
    # be bypassed by adding a call site.
    #
    # Deliberately NOT a proxy object wrapping _surfaceSigmaProfile: the profile
    # is passed on to other df constructors and into ARS `hxparams`, and its
    # `__class__.__name__` builds the dfcorrection save filename -- a wrapper
    # would silently change that filename and regenerate corrections instead of
    # loading them.
    # Three cases, because the cost of getting this wrong is measured in
    # minutes: diskdf's scipy-quad path calls in here ~133k times per oortA().
    #   * R is a backend array -> the caller is on the backend; go straight in.
    #   * plain R, numpy active -> the common path. Skip the @backend_input
    #     wrapper via __wrapped__: it can only be a no-op here (coerce_coords
    #     returns numpy coords object-identical), and at this call count even a
    #     no-op wrapper is worth ~12%.
    #   * plain R, a backend FORCED -> evaluate on that backend (the scipy
    #     paths pin numpy with _on_numpy, so they never get here).
    def _profile_differentiated(self):
        """True when the surfaceSigmaProfile's parameters carry a gradient.

        These paths dispatch on R, but a DIFFERENTIATED profile parameter makes
        the result traced whatever R is (surfacemass is exp(-R/params[0])). With
        a plain float R the numpy branch would then run numpy ufuncs on a tracer:
        d/d(hr) raised on jax and DETACHED to a silent 0 on torch.
        """
        return any(
            under_trace(p) or requires_backend_grad(p)
            for p in getattr(self._surfaceSigmaProfile, "_params", ())
        )

    def _backend_path(self, *xs):
        """True when an evaluation must take the backend path: a backend input or
        beta, or a differentiated profile parameter"""
        return (
            any(is_backend_array(x) for x in xs)
            or is_backend_array(self._beta)
            or self._profile_differentiated()
        )

    def _forced_coerce(self, *xs):
        """(xp, xs) for a private evaluation: numpy and xs as given on numpy,
        else the backend's, with plain xs coerced onto a forced backend"""
        if self._backend_path(*xs) or resolve_namespace() is not numpy:
            return self._backend_coerce(*xs)
        return numpy, xs

    def _forced_logexp(self, *xs):
        """(xs, log, exp) on _forced_coerce's namespace (numpy's scimath log)"""
        xp, xs = self._forced_coerce(*xs)
        if xp is numpy:
            return xs, numpylog, numpy.exp
        return xs, xp.log, xp.exp

    @property
    def _nquad(self):
        return _NQUAD_CORRECTED if self._correct else _NQUAD

    def _backend_coerce(self, *xs):
        """(xp, xs on xp): the namespace of the inputs, beta and the profile
        parameters (a differentiated parameter at plain-float inputs is data)"""
        params = (*getattr(self._surfaceSigmaProfile, "_params", ()), self._beta)
        xp = prefer_backend_namespace(*xs, *params)
        return xp, coerce_coords(xp, *xs, device=device_of(*xs, *params))

    def _ssp(self, name, R, **kwargs):
        ssp = self._surfaceSigmaProfile
        bound = getattr(ssp, name)
        if is_backend_array(R) or self._profile_differentiated():
            return bound(R, **kwargs)
        if get_namespace(R) is numpy:
            inner = getattr(bound, "__wrapped__", None)
            return inner(ssp, R, **kwargs) if inner is not None else bound(R, **kwargs)
        return bound(R, **kwargs)

    @physical_conversion("surfacedensitydistance", pop=True)
    @backend_input("d", "l")
    def targetSurfacemassLOS(self, d, l, log=False, deg=True):
        """
        Evaluate the target surface mass along the line of sight given Galactic longitude and distance.

        Parameters
        ----------
        d : float or Quantity
            Distance along the line of sight.
        l : float or Quantity
            Galactic longitude in degrees, unless deg=False.
        deg : bool, optional
            If False, l is in radians. Default is True.
        log : bool, optional
            If True, return the logarithm of the surface mass. Default is False.

        Returns
        -------
        float or Quantity
            Surface mass times distance.

        Notes
        -----
        - 2011-03-23 - Written - Bovy (NYU)
        """
        # Calculate R and phi
        if _APY_LOADED and isinstance(l, units.Quantity):
            lrad = conversion.parse_angle(l)
        elif deg:
            lrad = l * _DEGTORAD
        else:
            lrad = l
        d = conversion.parse_length(d, ro=self._ro)
        R, phi = _dlToRphi(d, lrad)
        if log:
            return self._ssp("surfacemass", R, log=log) + (
                get_namespace(d).log(d) if is_backend_array(d) else numpylog(d)
            )
        else:
            return self._ssp("surfacemass", R, log=log) * d

    @physical_conversion("surfacedensitydistance", pop=True)
    @backend_input("d", "l")
    def surfacemassLOS(
        self, d, l, deg=True, target=True, romberg=False, nsigma=None, relative=None
    ):
        """
        Evaluate the surface mass along the line of sight (LOS) given Galactic longitude and distance.

        Parameters
        ----------
        d : float or Quantity
            Distance along the line of sight.
        l : float or Quantity
            Galactic longitude (in deg, unless deg=False).
        nsigma : float, optional
            Number of sigma to integrate the velocities over.
        target : bool, optional
            If True, use target surfacemass (default).
        romberg : bool, optional
            If True, use a romberg integrator (default: False).
        deg : bool, optional
            If False, l is in radians.
        relative : bool, optional
            If True, return d.

        Returns
        -------
        float
            Sigma(d,l) x d

        Notes
        -----
        - 2011-03-24 - Written - Bovy (NYU)
        """
        # Calculate R and phi
        if _APY_LOADED and isinstance(l, units.Quantity):
            lrad = conversion.parse_angle(l)
        elif deg:
            lrad = l * _DEGTORAD
        else:
            lrad = l
        d = conversion.parse_length(d, ro=self._ro)
        R, phi = _dlToRphi(d, lrad)
        if target:
            if relative:
                return d
            else:
                return self._ssp("surfacemass", R) * d
        else:
            return (
                self.surfacemass(
                    R,
                    romberg=romberg,
                    nsigma=nsigma,
                    relative=relative,
                    use_physical=False,
                )
                * d
            )

    # --- inverse-CDF samplers ---------------------------------------------------
    # Every sampler draws by inverting a tabulated CDF: the distance along a
    # line of sight from Sigma(R(d)) d, the velocities at a radius from the DF
    # on a velocity mesh (the marginal in vT, then vR given vT), the
    # guiding-centre radius from x Sigma(x) and the orbital phase from the
    # radial angle of the spherical action-angle quadrature (rejection
    # sampling, adaptive rejection sampling and an orbit integration per sample
    # before 2026-10). One path for numpy, jax and torch: ``key=None`` draws the
    # uniforms from numpy's global generator and returns the historical numpy
    # objects; a backend key draws them in that namespace, returns backend
    # arrays (one Orbit holding every sample) and makes the samples
    # differentiable in the DF's parameters through the tables.
    _SAMPLE_ND = 1001  # nodes of the line-of-sight distance table
    _SAMPLE_ND_ACTUAL = (
        201  # ... for the actual surface density (a quadrature per node)
    )
    _SAMPLE_NVT = 241  # velocity-mesh nodes in vT
    _SAMPLE_NVR = 121  # ... in vR (61: sigma_R 0.5% high)
    _SAMPLE_VNSIGMA = 5.0  # the mesh's half-width in dispersions
    _SAMPLE_NX = 1001  # nodes of the x Sigma(x) table
    _SAMPLE_XMAX = 40.0  # its extent in scale lengths (x Sigma(x) ~ 40 e^-40)
    _SAMPLE_NPHASE = 48  # radial-angle nodes per half orbit

    def _sample_namespace(self, key):
        """The namespace a sampler computes in: a backend key's, else that of a
        differentiated profile parameter or beta, else the active one"""
        xp = get_namespace()
        if xp is numpy and key is not None:
            xp = namespace_for_name(grandom._backend_of_key(key))
        if xp is numpy and self._backend_path():
            xp, _ = self._backend_coerce()
        return xp

    @staticmethod
    def _sample_uniform(xp, key, shape, like):
        """Uniform draws of ``like``'s dtype; numpy's global stream for key=None"""
        return xp.astype(xp.asarray(grandom.uniform(key, shape)), like.dtype)

    @staticmethod
    def _sample_rows_increasing(xp, rows):
        """Rows strictly increasing along the last axis
        (ensure_strictly_increasing works along the first)"""
        return xp.permute_dims(
            ensure_strictly_increasing(xp, xp.permute_dims(rows, (1, 0))), (1, 0)
        )

    @staticmethod
    def _sample_lerp_rows(xp, table, row, q):
        """``table[row]`` at the fractional index ``q``: the inverse-CDF helpers
        invert on the shared index axis, the per-row grid is interpolated here.
        Returns (value, cell index, fraction)."""
        n = table.shape[-1]
        i0 = xp.astype(xp.clip(xp.floor(q), 0, n - 2), xp.int64)
        frac = q - xp.astype(i0, q.dtype)
        return table[row, i0] * (1.0 - frac) + table[row, i0 + 1] * frac, i0, frac

    def _sample_orbits(self, vxvv, key, physical):
        """The sampled points as Orbits: a list of single Orbits for key=None
        (the historical contract), one Orbit holding every sample for a key"""
        kw = {"ro": self._ro, "vo": self._vo} if physical else {}
        if key is None:
            return [Orbit(vxvv=row, **kw) for row in as_numpy(vxvv)]
        return Orbit(vxvv=vxvv, **kw)

    def _sample_los_distances(self, l, n, maxd, target, key, xp):
        """n distances along the line of sight l (rad) from the inverse CDF of
        Sigma(R(d)) d tabulated up to maxd"""
        nnodes = self._SAMPLE_ND if target else self._SAMPLE_ND_ACTUAL
        dg = xp.linspace(0.0, 1.0, nnodes) * maxd
        Rg = xp.sqrt(1.0 + dg**2.0 - 2.0 * dg * xp.cos(xp.asarray(l)))
        if target:
            dens = self._ssp("surfacemass", Rg) * dg
        else:
            dens = self.surfacemass(Rg, use_physical=False) * dg
        F = ensure_strictly_increasing(xp, trapezoid_cdf(xp, dens, dg, 0))
        u = self._sample_uniform(xp, key, (n,), dg)
        return linear_inverse_cdf_sample(xp, dg, F, u)

    def _sample_velocities(self, R, n, nsigma, target, key, xp):
        """n (vR, vT) at the radii R ((1,) for every draw or (n,), one each)
        from the inverse CDF of the DF on a velocity mesh: the marginal in vT,
        then vR given vT (the vR CDF interpolated to the drawn vT). The mesh
        follows the DF: a coarse pass over a box of a few circular velocities
        finds the support in vT, and every vT row's extent in vR is nsigma
        dispersions at that row's guiding radius -- the inner, hotter
        populations reach large radii on eccentric orbits (the Shu DF at
        R = 2 has 6 per cent of its mass beyond five local dispersions).
        ``target`` is accepted for compatibility; the mesh is the DF's own."""
        nvT, nvR = self._SAMPLE_NVT, self._SAMPLE_NVR
        ns = self._SAMPLE_VNSIGMA if nsigma is None else nsigma
        nR = R.shape[0]
        vc = R**self._beta
        vTc = vc - self.asymmetricdrift(R, use_physical=False)
        unitT = xp.linspace(-1.0, 1.0, nvT)
        unitR = xp.linspace(-1.0, 1.0, nvR)

        def halfwidth(vTg):  # ns dispersions at the row's guiding radius
            xrow = (R[:, None] * xp.where(vTg > 0.0, vTg, 1e-8)) ** (
                1.0 / (self._beta + 1.0)
            )
            return ns * xp.sqrt(self._ssp("sigma2", xrow))

        def density(vTg, vRh, unitR=unitR):  # f: vT rows, vR columns (nR, nvT, nvR)
            VT = vTg[:, :, None] + 0.0 * unitR[None, None, :]
            VR = vRh[:, :, None] * unitR[None, None, :] + 0.0 * VT
            RR = R[:, None, None] + 0.0 * VT
            E, L = vRvTRToEL(
                xp.reshape(VR, (-1,)),
                xp.reshape(VT, (-1,)),
                xp.reshape(RR, (-1,)),
                self._beta,
                self._dftype,
            )
            return xp.reshape(self._eval_backend(E, L, 0.0, 0.0), VT.shape)

        # pass 1: the support in vT (the mass below 1e-7 on either side) on a
        # box of ns circular velocities, or dispersions when hotter, at half
        # the resolution
        W = ns * xp.maximum(vc, xp.sqrt(self._ssp("sigma2", R)))
        coarseT = xp.linspace(-1.0, 1.0, nvT // 2 + 1)
        vTg = vTc[:, None] + W[:, None] * coarseT[None, :]
        vRh = halfwidth(vTg)
        m = trapezoid_sum(xp, density(vTg, vRh, unitR[::2]), unitR[::2], 2) * vRh
        below = xp.cumulative_sum(m, axis=1) / xp.sum(m, axis=1, keepdims=True)
        keep = (below > 1e-7) & (below < 1.0 - 1e-7)
        d = 2.0 * W / (nvT // 2)
        vTlo = xp.min(xp.where(keep, vTg, xp.inf), axis=1) - d
        vThi = xp.max(xp.where(keep, vTg, -xp.inf), axis=1) + d
        # pass 2: the mesh on the support
        vTg = vTlo[:, None] + 0.5 * (vThi - vTlo)[:, None] * (unitT[None, :] + 1.0)
        vRh = halfwidth(vTg)
        f = density(vTg, vRh)
        FvT = self._sample_rows_increasing(
            xp, trapezoid_cdf(xp, trapezoid_sum(xp, f, unitR, 2) * vRh, unitT, 1)
        )
        FvR = trapezoid_cdf(xp, f, unitR, 2)
        u = self._sample_uniform(xp, key, (2, n), unitT)
        row = xp.arange(n) if nR == n else xp.zeros(n, dtype=xp.int64)
        qT = batched_inverse_cdf_sample(
            xp, xp.astype(xp.arange(nvT), unitT.dtype), FvT[row], u[0]
        )
        vT, iT, wT = self._sample_lerp_rows(xp, vTg, row, qT)
        # vR given vT: the conditional CDF and the row's extent, both
        # interpolated between the neighbouring vT rows
        FvRj = self._sample_rows_increasing(
            xp, (1.0 - wT)[:, None] * FvR[row, iT] + wT[:, None] * FvR[row, iT + 1]
        )
        vRhj = (1.0 - wT) * vRh[row, iT] + wT * vRh[row, iT + 1]
        qR = batched_inverse_cdf_sample(
            xp, xp.astype(xp.arange(nvR), unitR.dtype), FvRj, u[1]
        )
        vRu, _, _ = self._sample_lerp_rows(
            xp, unitR[None, :] + 0.0 * vRhj[:, None], xp.arange(n), qR
        )
        return xp.stack([vRu * vRhj, vT], axis=1)

    @float64_default_if_torch_args
    @physical_conversion("position", pop=True)
    def sampledSurfacemassLOS(self, l, n=1, maxd=None, target=True, key=None):
        """
        Sample a distance along the line of sight

        Parameters
        ----------
        l : float or Quantity
            Galactic longitude (rad).
        n : int, optional
            Number of distances to sample.
        maxd : float or Quantity, optional
            Maximum distance to consider (the distance table's extent).
        target : bool, optional
            If True, sample from the 'target' surface mass density, rather than the actual surface mass density (default=True).
        key : backend random key, optional
            Draw the uniforms with this key (``galpy.backend.random.key``) and
            return a backend array; ``None`` uses numpy's global generator and
            returns a numpy array.

        Returns
        -------
        numpy.ndarray or backend array
            The sampled distances.

        Notes
        -----
        - 2011-03-24 - Written - Bovy (NYU)
        - 2026-10-28 - Inverse-CDF sampling on every backend - Bovy (UofT)

        """
        xp = self._sample_namespace(key)
        l = conversion.parse_angle(l)
        maxd = conversion.parse_length(maxd, ro=self._ro)
        if maxd is None:
            maxd = _MAXD_REJECTLOS
        out = self._sample_los_distances(l, n, maxd, target, key, xp)
        return out if key is not None else as_numpy(out)

    @float64_default_if_torch_args
    @potential_physical_input
    @physical_conversion("velocity", pop=True)
    def sampleVRVT(self, R, n=1, nsigma=None, target=True, key=None):
        """
        Sample a radial and azimuthal velocity at R

        Parameters
        ----------
        R : float or Quantity
            Galactocentric distance.
        n : int, optional
            Number of velocities to sample.
        nsigma : float, optional
            Half-width of the velocity mesh in dispersions (default: 5).
        target : bool, optional
            If True, use the 'target' sigma_R rather than the actual sigma_R to size the velocity mesh (default=True).
        key : backend random key, optional
            Draw the uniforms with this key (``galpy.backend.random.key``) and
            return a backend array; ``None`` uses numpy's global generator and
            returns a numpy array.

        Returns
        -------
        numpy.ndarray or backend array
            [vR, vT] samples, shape (n, 2).

        Notes
        -----
        - 2011-03-24 - Written - Bovy (NYU)
        - 2026-10-28 - Inverse-CDF sampling on every backend - Bovy (UofT)
        """
        xp = self._sample_namespace(key)
        (R,) = coerce_coords(xp, R)
        out = self._sample_velocities(
            xp.reshape(xp.asarray(R), (-1,)), n, nsigma, target, key, xp
        )
        return out if key is not None else as_numpy(out)

    @float64_default_if_torch_args
    def sampleLOS(
        self,
        los,
        n=1,
        deg=True,
        maxd=None,
        nsigma=None,
        targetSurfmass=True,
        targetSigma2=True,
        key=None,
    ):
        """
        Sample along a given LOS

        Parameters
        ----------
        los : float or Quantity
            Line of sight Galactic longitude.
        n : int, optional
            Number of points to sample.
        deg : bool, optional
            If False, los is in radians.
        maxd : float or Quantity, optional
            Maximum distance to consider (the distance table's extent).
        nsigma : float, optional
            Half-width of the velocity mesh in dispersions (default: 5).
        targetSurfmass : bool, optional
            If True, use target surface mass (default=True).
        targetSigma2 : bool, optional
            If True, use target sigma_R^2 (default=True).
        key : backend random key, optional
            Draw the uniforms with this key (``galpy.backend.random.key``) and
            return one Orbit instance holding every sample; ``None`` uses
            numpy's global generator and returns a list of Orbits.

        Returns
        -------
        list of Orbit, or Orbit
            The sampled [R, vR, vT, phi] points.

        Notes
        -----
        - target=False uses target distribution for derivatives (this is a detail)
        - 2011-03-24 - Written - Bovy (NYU)
        - 2026-10-28 - Inverse-CDF sampling on every backend - Bovy (UofT)
        """
        xp = self._sample_namespace(key)
        if _APY_LOADED and isinstance(los, units.Quantity):
            l = conversion.parse_angle(los)
        elif deg:
            l = los * _DEGTORAD
        else:
            l = los
        maxd = conversion.parse_length(maxd, ro=self._ro)
        if maxd is None:
            maxd = _MAXD_REJECTLOS
        k1, k2 = grandom.split(key, 2)
        ds = self._sample_los_distances(l, n, maxd, targetSurfmass, k1, xp)
        R, phi = _dlToRphi(ds, l)
        vv = self._sample_velocities(R, n, nsigma, targetSigma2, k2, xp)
        vxvv = xp.stack([R, vv[:, 0], vv[:, 1], phi], axis=1)
        return self._sample_orbits(vxvv, key, self._roSet and self._voSet)

    @potential_physical_input
    @physical_conversion("velocity", pop=True)
    @backend_input("R")
    def asymmetricdrift(self, R):
        """
        Estimate the asymmetric drift (vc-mean-vphi) from an approximation to the Jeans equation.

        Parameters
        ----------
        R : float or Quantity
            Radius at which to calculate the asymmetric drift.

        Returns
        -------
        float
            Asymmetric drift at R.

        Notes
        -----
        - 2011-04-02 - Written - Bovy (NYU).
        """
        return self._asymmetricdrift(R)

    def _asymmetricdrift(self, R):
        _, (R,) = self._forced_coerce(R)
        sigmaR2 = self._ssp("sigma2", R)
        return (
            sigmaR2
            / 2.0
            / R**self._beta
            * (
                1.0 / self._gamma**2.0
                - 1.0
                - R * self._ssp("surfacemassDerivative", R, log=True)
                - R * self._ssp("sigma2Derivative", R, log=True)
            )
        )

    def _backend_moment_prep(self, R, nsigma):
        """Backend (jax/torch) prelude shared by the moment quadratures: returns
        (xp, R, logSigmaR, logsigmaR2, sigmaR1, vTlo, vThi, gamma) for the
        velocity box, R on xp: a differentiated profile parameter at a
        Python-float R is data. All carry two trailing length-1 axes for the
        (vR, vT) GL grid; gamma is the scalar_like-anchored self._gamma."""
        xp, (R,) = self._backend_coerce(R)
        R = xp.reshape(R, tuple(R.shape) + (1, 1))
        gamma = scalar_like(R, self._gamma)  # numpy.float64: see scalar_like
        logSigmaR = self._ssp("surfacemass", R, log=True)
        sigmaR2 = self._ssp("sigma2", R)
        sigmaR1 = xp.sqrt(sigmaR2)
        logsigmaR2 = xp.log(sigmaR2)
        va = (
            sigmaR2
            / 2.0
            / R**self._beta
            * (
                1.0 / gamma**2.0
                - 1.0
                - R * self._ssp("surfacemassDerivative", R, log=True)
                - R * self._ssp("sigma2Derivative", R, log=True)
            )
        )
        va = xp.where(xp.abs(va) > sigmaR1, 0.0, va)  # avoid craziness near center
        vTcen = gamma * (R**self._beta - va) / sigmaR1
        return (
            xp,
            R,
            logSigmaR,
            logsigmaR2,
            sigmaR1,
            vTcen - nsigma,
            vTcen + nsigma,
            gamma,
        )

    @potential_physical_input
    @physical_conversion("surfacedensity", pop=True)
    @backend_input("R")
    @_loop_array_R
    def surfacemass(self, R, romberg=False, nsigma=None, relative=False):
        """
        Calculate the surface-mass at R by marginalizing over velocity

        Parameters
        ----------
        R : float, numpy.ndarray or Quantity
            Radius at which to calculate the surfacemass density.
        romberg : bool, optional
            If True, use a romberg integrator (default: False)
        nsigma : float, optional
            Number of sigma to integrate the velocities over
        relative : bool, optional
            If True, return the relative surface mass at R (default: False)

        Returns
        -------
        float
            Surface mass at R

        Notes
        -----
        - 2011-03-XX - Bovy (NYU)
        """
        if nsigma == None:
            nsigma = _NSIGMA
        if self._backend_path(R):
            xp, R, logSigmaR, logsigmaR2, sigmaR1, vTlo, vThi, gamma = (
                self._backend_moment_prep(R, nsigma)
            )
            norm = 1.0 if relative else xp.exp(logSigmaR[..., 0, 0])
            return (
                nested_quad(
                    xp,
                    lambda vR, vT: _surfaceIntegrand(
                        vR, vT, R, self, logSigmaR, logsigmaR2, sigmaR1, gamma
                    ),
                    [(0.0, nsigma), (vTlo, vThi)],
                    n=self._nquad,
                )
                / numpy.pi
                * norm
            )
        logSigmaR = self._ssp("surfacemass", R, log=True)
        sigmaR2 = self._ssp("sigma2", R)
        sigmaR1 = numpy.sqrt(sigmaR2)
        logsigmaR2 = numpylog(sigmaR2)
        if relative:
            norm = 1.0
        else:
            norm = numpy.exp(logSigmaR)
        # Use the asymmetric drift equation to estimate va
        va = (
            sigmaR2
            / 2.0
            / R**self._beta
            * (
                1.0 / self._gamma**2.0
                - 1.0
                - R * self._ssp("surfacemassDerivative", R, log=True)
                - R * self._ssp("sigma2Derivative", R, log=True)
            )
        )
        if numpy.fabs(va) > sigmaR1:
            va = 0.0  # To avoid craziness near the center
        if romberg:
            return numpy.real(
                bovy_dblquad(
                    _surfaceIntegrand,
                    self._gamma * (R**self._beta - va) / sigmaR1 - nsigma,
                    self._gamma * (R**self._beta - va) / sigmaR1 + nsigma,
                    lambda x: 0.0,
                    lambda x: nsigma,
                    [R, self, logSigmaR, logsigmaR2, sigmaR1, self._gamma],
                    tol=10.0**-8,
                )
                / numpy.pi
                * norm
            )
        else:
            return (
                integrate.dblquad(
                    _surfaceIntegrand,
                    self._gamma * (R**self._beta - va) / sigmaR1 - nsigma,
                    self._gamma * (R**self._beta - va) / sigmaR1 + nsigma,
                    lambda x: 0.0,
                    lambda x: nsigma,
                    (R, self, logSigmaR, logsigmaR2, sigmaR1, self._gamma),
                    epsrel=_EPSREL,
                )[0]
                / numpy.pi
                * norm
            )

    @potential_physical_input
    @physical_conversion("velocity2surfacedensity", pop=True)
    @backend_input("R")
    @_loop_array_R
    def sigma2surfacemass(self, R, romberg=False, nsigma=None, relative=False):
        """
        Calculate the product sigma_R^2 x surface-mass at R by marginalizing over velocity.

        Parameters
        ----------
        R : float, numpy.ndarray or Quantity
            Radius at which to calculate the sigma_R^2 x surfacemass density.
        romberg : bool, optional
            If True, use a romberg integrator (default: False).
        nsigma : float, optional
            Number of sigma to integrate the velocities over.
        relative : bool, optional
            If True, return the relative density (default: False).

        Returns
        -------
        float
            Sigma_R^2 x surface-mass at R.

        Notes
        -----
        - 2010-03-XX - Written - Bovy (NYU).

        """
        if nsigma == None:
            nsigma = _NSIGMA
        if self._backend_path(R):
            xp, R, logSigmaR, logsigmaR2, sigmaR1, vTlo, vThi, gamma = (
                self._backend_moment_prep(R, nsigma)
            )
            norm = 1.0 if relative else xp.exp((logSigmaR + logsigmaR2)[..., 0, 0])
            return (
                nested_quad(
                    xp,
                    lambda vR, vT: _sigma2surfaceIntegrand(
                        vR, vT, R, self, logSigmaR, logsigmaR2, sigmaR1, gamma
                    ),
                    [(0.0, nsigma), (vTlo, vThi)],
                    n=self._nquad,
                )
                / numpy.pi
                * norm
            )
        logSigmaR = self._ssp("surfacemass", R, log=True)
        sigmaR2 = self._ssp("sigma2", R)
        sigmaR1 = numpy.sqrt(sigmaR2)
        logsigmaR2 = numpylog(sigmaR2)
        if relative:
            norm = 1.0
        else:
            norm = numpy.exp(logSigmaR + logsigmaR2)
        # Use the asymmetric drift equation to estimate va
        va = (
            sigmaR2
            / 2.0
            / R**self._beta
            * (
                1.0 / self._gamma**2.0
                - 1.0
                - R * self._ssp("surfacemassDerivative", R, log=True)
                - R * self._ssp("sigma2Derivative", R, log=True)
            )
        )
        if numpy.fabs(va) > sigmaR1:
            va = 0.0  # To avoid craziness near the center
        if romberg:
            return numpy.real(
                bovy_dblquad(
                    _sigma2surfaceIntegrand,
                    self._gamma * (R**self._beta - va) / sigmaR1 - nsigma,
                    self._gamma * (R**self._beta - va) / sigmaR1 + nsigma,
                    lambda x: 0.0,
                    lambda x: nsigma,
                    [R, self, logSigmaR, logsigmaR2, sigmaR1, self._gamma],
                    tol=10.0**-8,
                )
                / numpy.pi
                * norm
            )
        else:
            return (
                integrate.dblquad(
                    _sigma2surfaceIntegrand,
                    self._gamma * (R**self._beta - va) / sigmaR1 - nsigma,
                    self._gamma * (R**self._beta - va) / sigmaR1 + nsigma,
                    lambda x: 0.0,
                    lambda x: nsigma,
                    (R, self, logSigmaR, logsigmaR2, sigmaR1, self._gamma),
                    epsrel=_EPSREL,
                )[0]
                / numpy.pi
                * norm
            )

    def vmomentsurfacemass(self, *args, **kwargs):
        """
        Calculate the an arbitrary moment of the velocity distribution at R times the surfacmass

        Parameters
        ----------
        R: float, numpy.ndarray or Quantity
            Galactocentric radius at which to calculate the moment.
        n: int
            vR^n in the moment
        m: int
            vT^m in the moment
        nsigma : int, optional
            number of sigma to integrate the velocities over
        romberg : bool, optional
            If True, use a romberg integrator (default: False)
        deriv : str, optional
            Calculates derivative of the moment wrt R or phi (default: None)

        Returns
        -------
        float or Quantity
            <vR^n vT^m  x surface-mass> at R (no support for units)

        Notes
        -----
        - 2011-03-30 - Written - Bovy (NYU)
        """
        use_physical = kwargs.pop("use_physical", True)
        ro = kwargs.pop("ro", None)
        if ro is None and hasattr(self, "_roSet") and self._roSet:
            ro = self._ro
        ro = conversion.parse_length_kpc(ro)
        vo = kwargs.pop("vo", None)
        if vo is None and hasattr(self, "_voSet") and self._voSet:
            vo = self._vo
        vo = conversion.parse_velocity_kms(vo)
        if use_physical and not vo is None and not ro is None:
            fac = surfdens_in_msolpc2(vo, ro) * vo ** (args[1] + args[2])
            if _APY_UNITS:
                u = (
                    units.Msun
                    / units.pc**2
                    * (units.km / units.s) ** (args[1] + args[2])
                )
            out = self._vmomentsurfacemass(*args, **kwargs)
            if _APY_UNITS:
                return units.Quantity(to_host(out * fac), unit=u)
            else:
                return out * fac
        else:
            return self._vmomentsurfacemass(*args, **kwargs)

    @backend_input("R", "phi")
    @_loop_array_R
    def _vmomentsurfacemass(
        self, R, n, m, romberg=False, nsigma=None, relative=False, phi=0.0, deriv=None
    ):
        """Non-physical version of vmomentsurfacemass, otherwise the same"""
        # odd moments of vR are zero
        if isinstance(n, int) and n % 2 == 1:
            return (
                namespace_from_arrays((R,)).zeros_like(R * 1.0)
                if is_backend_array(R)
                else 0.0
            )
        if nsigma == None:
            nsigma = _NSIGMA
        if self._backend_path(R):
            xp, R, logSigmaR, logsigmaR2, sigmaR1, vTlo, vThi, gamma = (
                self._backend_moment_prep(R, nsigma)
            )
            norm = (
                1.0
                if relative
                else xp.exp((logSigmaR + logsigmaR2 * (n + m) / 2.0)[..., 0, 0])
                / gamma**m
            )
            if deriv is None:
                integ = lambda vR, vT: _vmomentsurfaceIntegrand(
                    vR, vT, R, self, logSigmaR, logsigmaR2, sigmaR1, gamma, n, m
                )
            else:
                integ = lambda vR, vT: _vmomentderivsurfaceIntegrand(
                    vR,
                    vT,
                    R,
                    self,
                    logSigmaR,
                    logsigmaR2,
                    sigmaR1,
                    gamma,
                    n,
                    m,
                    deriv,
                )
            return (
                nested_quad(xp, integ, [(-nsigma, nsigma), (vTlo, vThi)], n=self._nquad)
                / numpy.pi
                * norm
                / 2.0
            )
        logSigmaR = self._ssp("surfacemass", R, log=True)
        sigmaR2 = self._ssp("sigma2", R)
        sigmaR1 = numpy.sqrt(sigmaR2)
        logsigmaR2 = numpylog(sigmaR2)
        if relative:
            norm = 1.0
        else:
            norm = numpy.exp(logSigmaR + logsigmaR2 * (n + m) / 2.0) / self._gamma**m
        # Use the asymmetric drift equation to estimate va
        va = (
            sigmaR2
            / 2.0
            / R**self._beta
            * (
                1.0 / self._gamma**2.0
                - 1.0
                - R * self._ssp("surfacemassDerivative", R, log=True)
                - R * self._ssp("sigma2Derivative", R, log=True)
            )
        )
        if numpy.fabs(va) > sigmaR1:
            va = 0.0  # To avoid craziness near the center
        if deriv is None:
            if romberg:
                return numpy.real(
                    bovy_dblquad(
                        _vmomentsurfaceIntegrand,
                        self._gamma * (R**self._beta - va) / sigmaR1 - nsigma,
                        self._gamma * (R**self._beta - va) / sigmaR1 + nsigma,
                        lambda x: -nsigma,
                        lambda x: nsigma,
                        [R, self, logSigmaR, logsigmaR2, sigmaR1, self._gamma, n, m],
                        tol=10.0**-8,
                    )
                    / numpy.pi
                    * norm
                    / 2.0
                )
            else:
                return (
                    integrate.dblquad(
                        _vmomentsurfaceIntegrand,
                        self._gamma * (R**self._beta - va) / sigmaR1 - nsigma,
                        self._gamma * (R**self._beta - va) / sigmaR1 + nsigma,
                        lambda x: -nsigma,
                        lambda x: nsigma,
                        (R, self, logSigmaR, logsigmaR2, sigmaR1, self._gamma, n, m),
                        epsrel=_EPSREL,
                    )[0]
                    / numpy.pi
                    * norm
                    / 2.0
                )
        else:
            if romberg:
                return numpy.real(
                    bovy_dblquad(
                        _vmomentderivsurfaceIntegrand,
                        self._gamma * (R**self._beta - va) / sigmaR1 - nsigma,
                        self._gamma * (R**self._beta - va) / sigmaR1 + nsigma,
                        lambda x: -nsigma,
                        lambda x: nsigma,
                        [
                            R,
                            self,
                            logSigmaR,
                            logsigmaR2,
                            sigmaR1,
                            self._gamma,
                            n,
                            m,
                            deriv,
                        ],
                        tol=10.0**-8,
                    )
                    / numpy.pi
                    * norm
                    / 2.0
                )
            else:
                return (
                    integrate.dblquad(
                        _vmomentderivsurfaceIntegrand,
                        self._gamma * (R**self._beta - va) / sigmaR1 - nsigma,
                        self._gamma * (R**self._beta - va) / sigmaR1 + nsigma,
                        lambda x: -nsigma,
                        lambda x: nsigma,
                        (
                            R,
                            self,
                            logSigmaR,
                            logsigmaR2,
                            sigmaR1,
                            self._gamma,
                            n,
                            m,
                            deriv,
                        ),
                        epsrel=_EPSREL,
                    )[0]
                    / numpy.pi
                    * norm
                    / 2.0
                )

    @potential_physical_input
    @physical_conversion("frequency-kmskpc", pop=True)
    @backend_input("R", "phi")
    def oortA(self, R, romberg=False, nsigma=None, phi=0.0):
        """
        Calculate the Oort function A.

        Parameters
        ----------
        R : float or Quantity
            Radius at which to calculate A.
        phi : float, optional
            Azimuth (default: 0.0).
        nsigma : int, optional
            Number of sigma to integrate the velocities over.
        romberg : bool, optional
            If True, use a romberg integrator (default: False).

        Returns
        -------
        float or Quantity
            Oort A at R.

        Notes
        -----
        - 2011-04-19 - Written - Bovy (NYU)
        """
        # Could be made more efficient, e.g., surfacemass is calculated multiple times.
        # 2A= meanvphi/R-dmeanvR/R/dphi-dmeanvphi/dR
        meanvphi = self.meanvT(
            R, romberg=romberg, nsigma=nsigma, phi=phi, use_physical=False
        )
        dmeanvRRdphi = 0.0  # We know this, since the DF does not depend on phi
        surfmass = self._vmomentsurfacemass(
            R, 0, 0, phi=phi, romberg=romberg, nsigma=nsigma
        )
        dmeanvphidR = self._vmomentsurfacemass(
            R, 0, 1, deriv="R", phi=phi, romberg=romberg, nsigma=nsigma
        ) / surfmass - self._vmomentsurfacemass(
            R, 0, 1, phi=phi, romberg=romberg, nsigma=nsigma
        ) / surfmass**2.0 * self._vmomentsurfacemass(
            R, 0, 0, deriv="R", phi=phi, romberg=romberg, nsigma=nsigma
        )
        return 0.5 * (meanvphi / R - dmeanvRRdphi / R - dmeanvphidR)

    @potential_physical_input
    @physical_conversion("frequency-kmskpc", pop=True)
    @backend_input("R", "phi")
    def oortB(self, R, romberg=False, nsigma=None, phi=0.0):
        """
        Calculate the Oort function B.

        Parameters
        ----------
        R : float
            Radius at which to calculate B (can be Quantity).
        romberg : bool, optional
            If True, use a romberg integrator (default: False).
        nsigma : float, optional
            Number of sigma to integrate the velocities over.
        phi : float, optional
            Azimuth angle (in radians) at which to calculate B.

        Returns
        -------
        float or Quantity
            Oort B at R.

        Notes
        -----
        - 2011-04-19 - Written - Bovy (NYU).
        """
        # Could be made more efficient, e.g., surfacemass is calculated multiple times.
        # 2B= -meanvphi/R+dmeanvR/R/dphi-dmeanvphi/dR
        meanvphi = self.meanvT(
            R, romberg=romberg, nsigma=nsigma, phi=phi, use_physical=False
        )
        dmeanvRRdphi = 0.0  # We know this, since the DF does not depend on phi
        surfmass = self._vmomentsurfacemass(
            R, 0, 0, phi=phi, romberg=romberg, nsigma=nsigma
        )
        dmeanvphidR = self._vmomentsurfacemass(
            R, 0, 1, deriv="R", phi=phi, romberg=romberg, nsigma=nsigma
        ) / surfmass - self._vmomentsurfacemass(
            R, 0, 1, phi=phi, romberg=romberg, nsigma=nsigma
        ) / surfmass**2.0 * self._vmomentsurfacemass(
            R, 0, 0, deriv="R", phi=phi, romberg=romberg, nsigma=nsigma
        )
        return 0.5 * (-meanvphi / R + dmeanvRRdphi / R - dmeanvphidR)

    @potential_physical_input
    @physical_conversion("frequency-kmskpc", pop=True)
    @backend_input("R", "phi")
    def oortC(self, R, romberg=False, nsigma=None, phi=0.0):
        """
        Calculate the Oort function C.

        Parameters
        ----------
        R : float or Quantity
            Radius at which to calculate C (can be Quantity).
        nsigma : int, optional
            Number of sigma to integrate the velocities over.
        romberg : bool, optional
            If True, use a romberg integrator (default: False).
        phi : float, optional
            Azimuth (default: 0.0).

        Returns
        -------
        float or Quantity
            Oort C at R.

        Notes
        -----
        - 2011-04-19 - Written - Bovy (NYU)
        """
        # - Could be made more efficient, e.g., surfacemass is calculated multiple times.
        # - We know this is zero, but it is calculated anyway (bug or feature?).
        # 2C= -meanvR/R-dmeanvphi/R/dphi+dmeanvR/dR
        meanvr = self.meanvR(
            R, romberg=romberg, nsigma=nsigma, phi=phi, use_physical=False
        )
        dmeanvphiRdphi = 0.0  # We know this, since the DF does not depend on phi
        surfmass = self._vmomentsurfacemass(
            R, 0, 0, phi=phi, romberg=romberg, nsigma=nsigma
        )
        dmeanvRdR = (
            self._vmomentsurfacemass(
                R, 1, 0, deriv="R", phi=phi, romberg=romberg, nsigma=nsigma
            )
            / surfmass
        )  # other terms is zero because f is even in vR
        return 0.5 * (-meanvr / R - dmeanvphiRdphi / R + dmeanvRdR)

    @potential_physical_input
    @physical_conversion("frequency-kmskpc", pop=True)
    @backend_input("R", "phi")
    def oortK(self, R, romberg=False, nsigma=None, phi=0.0):
        """
        Calculate the Oort function K.

        Parameters
        ----------
        R : float
            Radius at which to calculate K (can be Quantity).
        phi : float, optional
            Azimuth angle (in radians) at which to calculate K.
        nsigma : int, optional
            Number of sigma to integrate the velocities over.
        romberg : bool, optional
            If True, use a romberg integrator (default: False).

        Returns
        -------
        float or Quantity
            Oort K at R.

        Notes
        -----
        - 2011-04-19 - Written - Bovy (NYU)
        """
        # - Could be made more efficient, e.g., surfacemass is calculated multiple times.
        # - We know this is zero, but it is calculated anyway (bug or feature?).
        # 2K= meanvR/R+dmeanvphi/R/dphi+dmeanvR/dR
        meanvr = self.meanvR(
            R, romberg=romberg, nsigma=nsigma, phi=phi, use_physical=False
        )
        dmeanvphiRdphi = 0.0  # We know this, since the DF does not depend on phi
        surfmass = self._vmomentsurfacemass(
            R, 0, 0, phi=phi, romberg=romberg, nsigma=nsigma
        )
        dmeanvRdR = (
            self._vmomentsurfacemass(
                R, 1, 0, deriv="R", phi=phi, romberg=romberg, nsigma=nsigma
            )
            / surfmass
        )  # other terms is zero because f is even in vR
        return 0.5 * (+meanvr / R + dmeanvphiRdphi / R + dmeanvRdR)

    @potential_physical_input
    @physical_conversion("velocity2", pop=True)
    @backend_input("R", "phi")
    def sigma2(self, R, romberg=False, nsigma=None, phi=0.0):
        """
        Calculate sigma_R^2 at R by marginalizing over velocity.

        Parameters
        ----------
        R : float
            Radius at which to calculate sigma_R^2 density.
        romberg : bool, optional
            If True, use a romberg integrator (default: False).
        nsigma : int, optional
            Number of sigma to integrate the velocities over.
        phi : float, optional
            Azimuth angle at which to calculate sigma_R^2 density.

        Returns
        -------
        float or Quantity
            Sigma_R^2 at R.

        Notes
        -----
        - 2010-03-XX - Written - Bovy (NYU)
        """

        return self.sigma2surfacemass(
            R, romberg, nsigma, use_physical=False
        ) / self.surfacemass(R, romberg, nsigma, use_physical=False)

    @potential_physical_input
    @physical_conversion("velocity2", pop=True)
    @backend_input("R", "phi")
    def sigmaT2(self, R, romberg=False, nsigma=None, phi=0.0):
        """
        Calculate sigma_T^2 at R by marginalizing over velocity

        Parameters
        ----------
        R : float
            Radius at which to calculate sigma_T^2 (can be Quantity)
        romberg : bool, optional
            If True, use a romberg integrator (default: False)
        nsigma : int, optional
            Number of sigma to integrate the velocities over
        phi : float, optional
            Azimuth (default: 0.0)

        Returns
        -------
        float or Quantity
            Sigma_T^2 at R

        Notes
        -----
        - 2011-03-30 - Written - Bovy (NYU)

        """
        surfmass = self.surfacemass(
            R, romberg=romberg, nsigma=nsigma, use_physical=False
        )
        return (
            self._vmomentsurfacemass(R, 0, 2, romberg=romberg, nsigma=nsigma)
            - self._vmomentsurfacemass(R, 0, 1, romberg=romberg, nsigma=nsigma) ** 2.0
            / surfmass
        ) / surfmass

    @potential_physical_input
    @physical_conversion("velocity2", pop=True)
    @backend_input("R", "phi")
    def sigmaR2(self, R, romberg=False, nsigma=None, phi=0.0):
        """
        Calculate sigma_R^2 at R by marginalizing over velocity.

        Parameters
        ----------
        R : float
            Radius at which to calculate sigma_R^2.
        romberg : bool, optional
            If True, use a romberg integrator (default: False).
        nsigma : int, optional
            Number of sigma to integrate the velocities over.
        phi : float, optional
            Azimuth (default: 0.0).

        Returns
        -------
        float or Quantity
            Sigma_R^2 at R.

        Notes
        -----
        - 2011-03-30 - Written - Bovy (NYU).

        """
        return self.sigma2(R, romberg=romberg, nsigma=nsigma, use_physical=False)

    @potential_physical_input
    @physical_conversion("velocity", pop=True)
    @backend_input("R", "phi")
    def meanvT(self, R, romberg=False, nsigma=None, phi=0.0):
        """
        Calculate the mean tangential velocity at a given radius by marginalizing over velocity.

        Parameters
        ----------
        R : float
            Radius at which to calculate the mean tangential velocity.
        romberg : bool, optional
            If True, use a Romberg integrator. Default is False.
        nsigma : float, optional
            Number of sigma to integrate the velocities over.
        phi : float, optional
            Azimuth angle at which to calculate the mean tangential velocity.

        Returns
        -------
        float or Quantity
            The mean tangential velocity at the given radius.

        Notes
        -----
        - 2011-03-30 - Written - Bovy (NYU)
        """
        return self._vmomentsurfacemass(
            R, 0, 1, romberg=romberg, nsigma=nsigma
        ) / self.surfacemass(R, romberg=romberg, nsigma=nsigma, use_physical=False)

    @potential_physical_input
    @physical_conversion("velocity", pop=True)
    @backend_input("R", "phi")
    def meanvR(self, R, romberg=False, nsigma=None, phi=0.0):
        """
        Calculate <vR> at R by marginalizing over velocity.

        Parameters
        ----------
        R : float
            Radius at which to calculate <vR>.
        romberg : bool, optional
            If True, use a romberg integrator (default: False).
        nsigma : float, optional
            Number of sigma to integrate the velocities over.
        phi : float, optional
            Azimuth angle at which to calculate <vR>.

        Returns
        -------
        float or Quantity
            <vR> at R.

        Notes
        -----
        - 2011-03-30 - Written - Bovy (NYU).
        """

        return self._vmomentsurfacemass(
            R, 1, 0, romberg=romberg, nsigma=nsigma
        ) / self.surfacemass(R, romberg=romberg, nsigma=nsigma, use_physical=False)

    @potential_physical_input
    @backend_input("R", "phi")
    def skewvT(self, R, romberg=False, nsigma=None, phi=0.0):
        """
        Calculate skew in vT at R by marginalizing over velocity

        Parameters
        ----------
        R : float
            Radius at which to calculate <vR>
        romberg : bool, optional
            If True, use a romberg integrator (default: False)
        nsigma : float, optional
            Number of sigma to integrate the velocities over
        phi : float, optional
            Azimuth (default: 0.0)

        Returns
        -------
        float
            Skew in vT

        Notes
        -----
        - 2011-12-07 - Written - Bovy (NYU)
        """
        surfmass = self.surfacemass(
            R, romberg=romberg, nsigma=nsigma, use_physical=False
        )
        vt = (
            self._vmomentsurfacemass(R, 0, 1, romberg=romberg, nsigma=nsigma) / surfmass
        )
        vt2 = (
            self._vmomentsurfacemass(R, 0, 2, romberg=romberg, nsigma=nsigma) / surfmass
        )
        vt3 = (
            self._vmomentsurfacemass(R, 0, 3, romberg=romberg, nsigma=nsigma) / surfmass
        )
        s2 = vt2 - vt**2.0
        return (vt3 - 3.0 * vt * vt2 + 2.0 * vt**3.0) * s2 ** (-1.5)

    @potential_physical_input
    @backend_input("R", "phi")
    def skewvR(self, R, romberg=False, nsigma=None, phi=0.0):
        """
        Calculate skew in vR at R by marginalizing over velocity.

        Parameters
        ----------
        R : float or Quantity
            Radius at which to calculate <vR>.
        romberg : bool, optional
            If True, use a romberg integrator (default: False).
        nsigma : float, optional
            Number of sigma to integrate the velocities over.
        phi : float, optional
            Azimuth (in radians) at which to calculate the skew in vR.

        Returns
        -------
        float
            Skew in vR.

        Notes
        -----
        - 2011-12-07 - Written - Bovy (NYU).
        """

        surfmass = self.surfacemass(
            R, romberg=romberg, nsigma=nsigma, use_physical=False
        )
        vr = (
            self._vmomentsurfacemass(R, 1, 0, romberg=romberg, nsigma=nsigma) / surfmass
        )
        vr2 = (
            self._vmomentsurfacemass(R, 2, 0, romberg=romberg, nsigma=nsigma) / surfmass
        )
        vr3 = (
            self._vmomentsurfacemass(R, 3, 0, romberg=romberg, nsigma=nsigma) / surfmass
        )
        s2 = vr2 - vr**2.0
        return (vr3 - 3.0 * vr * vr2 + 2.0 * vr**3.0) * s2 ** (-1.5)

    @potential_physical_input
    @backend_input("R", "phi")
    def kurtosisvT(self, R, romberg=False, nsigma=None, phi=0.0):
        """
        Calculate excess kurtosis in vT at R by marginalizing over velocity

        Parameters
        ----------
        R : float or Quantity
            Radius at which to calculate <vR>
        romberg : bool, optional
            If True, use a romberg integrator (default: False)
        nsigma : float, optional
            Number of sigma to integrate the velocities over
        phi : float, optional
            (default: 0.0)

        Returns
        -------
        float
            kurtosisvT

        Notes
        -----
        - 2011-12-07 - Written - Bovy (NYU)

        """
        surfmass = self.surfacemass(
            R, romberg=romberg, nsigma=nsigma, use_physical=False
        )
        vt = (
            self._vmomentsurfacemass(R, 0, 1, romberg=romberg, nsigma=nsigma) / surfmass
        )
        vt2 = (
            self._vmomentsurfacemass(R, 0, 2, romberg=romberg, nsigma=nsigma) / surfmass
        )
        vt3 = (
            self._vmomentsurfacemass(R, 0, 3, romberg=romberg, nsigma=nsigma) / surfmass
        )
        vt4 = (
            self._vmomentsurfacemass(R, 0, 4, romberg=romberg, nsigma=nsigma) / surfmass
        )
        s2 = vt2 - vt**2.0
        return (vt4 - 4.0 * vt * vt3 + 6.0 * vt**2.0 * vt2 - 3.0 * vt**4.0) * s2 ** (
            -2.0
        ) - 3.0

    @potential_physical_input
    @backend_input("R", "phi")
    def kurtosisvR(self, R, romberg=False, nsigma=None, phi=0.0):
        """
        Calculate excess kurtosis in vR at R by marginalizing over velocity

        Parameters
        ----------
        R : float or Quantity
            Radius at which to calculate <vR>
        romberg : bool, optional
            If True, use a romberg integrator (default: False)
        nsigma : float, optional
            Number of sigma to integrate the velocities over
        phi : float or Quantity, optional
            Azimuth (default: 0.0)

        Returns
        -------
        float
            KurtosisvR

        Notes
        -----
        - 2011-12-07 - Written - Bovy (NYU)
        """
        surfmass = self.surfacemass(
            R, romberg=romberg, nsigma=nsigma, use_physical=False
        )
        vr = (
            self._vmomentsurfacemass(R, 1, 0, romberg=romberg, nsigma=nsigma) / surfmass
        )
        vr2 = (
            self._vmomentsurfacemass(R, 2, 0, romberg=romberg, nsigma=nsigma) / surfmass
        )
        vr3 = (
            self._vmomentsurfacemass(R, 3, 0, romberg=romberg, nsigma=nsigma) / surfmass
        )
        vr4 = (
            self._vmomentsurfacemass(R, 4, 0, romberg=romberg, nsigma=nsigma) / surfmass
        )
        s2 = vr2 - vr**2.0
        return (vr4 - 4.0 * vr * vr3 + 6.0 * vr**2.0 * vr2 - 3.0 * vr**4.0) * s2 ** (
            -2.0
        ) - 3.0

    def _ELtowRRapRperi(self, E, L):
        """
        Calculate the radial frequency based on energy and angular momentum and return the pericenter and apocenter radii.

        Parameters
        ----------
        E : float
            Energy.
        L : float
            Angular momentum.

        Returns
        -------
        tuple
            Tuple containing:
            - wR(E.L) : float
                Radial frequency.
            - rap : float
                Apocenter radius.
            - rperi : float
                Pericenter radius.

        Notes
        -----
        - 2010-07-11 - Written - Bovy (NYU)
        """
        if self._beta == 0.0:
            xE = numpy.exp(E - 0.5)
        else:  # non-flat rotation curve
            xE = (2.0 * E / (1.0 + 1.0 / self._beta)) ** (1.0 / 2.0 / self._beta)
        _, _, rperi, rap = self._aA.EccZmaxRperiRap(
            xE,
            numpy.sqrt(2.0 * (E - to_host(self._psp(xE))) - L**2.0 / xE**2.0),
            L / xE,
            0.0,
            0.0,
        )
        return to_host(  # read by the numpy sampler
            (
                self._aA._aAS.actionsFreqs(xE, 0.0, L / xE, 0.0, 0.0)[3][0],
                rap[0],
                rperi[0],
            )
        )

    @float64_default_if_torch_args
    def sample(
        self,
        n=1,
        rrange=None,
        returnROrbit=True,
        returnOrbit=False,
        nphi=1.0,
        los=None,
        losdeg=True,
        nsigma=None,
        maxd=None,
        targetSurfmass=True,
        targetSigma2=True,
        **kwargs,
    ):
        """
        Sample n*nphi points from this DF.

        Parameters
        ----------
        n : int, optional
            Number of desired samples (specifying this rather than calling
            this routine n times is more efficient). Default is 1.
        rrange : list or tuple, optional
            If you only want samples in this rrange, set this keyword
            (only works when asking for an (RZ)Orbit). Default is None.
        returnROrbit : bool, optional
            If True, return a planarROrbit instance: [R,vR,vT] (default).
            Default is True.
        returnOrbit : bool, optional
            If True, return a planarOrbit instance (including phi).
            Default is False.
        nphi : float, optional
            Number of azimuths to sample for each E,L. Default is 1.0.
        los : float or Quantity, optional
            If set, sample along this line of sight (deg) (assumes that the Sun is located at R=1,phi=0).
            Default is None.
        losdeg : bool, optional
            If False, los is in radians (default=True). Default is True.
        nsigma : float, optional
            Half-width of the velocity mesh in dispersions, for los sampling (default: 5).
        maxd : float or Quantity, optional
            Maximum distance to consider, for los sampling. Default is None.
        targetSurfmass : bool, optional
            If True, use target surface mass profile. Default is True.
        targetSigma2 : bool, optional
            If True, use target sigma2 profile. Default is True.
        key : backend random key, optional
            Draw the uniforms with this key (``galpy.backend.random.key``) and
            return backend arrays: one Orbit instance holding every sample, or
            an (n*nphi, 2) array of [E, Lz]. ``None`` uses numpy's global
            generator and returns a list of Orbits, or of [E, Lz] pairs.
        **kwargs : dict, optional
            Additional keyword arguments (``use_physical``).

        Returns
        -------
        out : list, Orbit, or array
            n*nphi list of [[E,Lz],...] or list of planar(R)Orbits (key=None);
            one Orbit or an array with a key.
            CAUTION: lists of EL need to be post-processed to account for the
            \\kappa/\\omega_R discrepancy; EL not returned in physical units.

        Notes
        -----
        - The guiding-centre radius is drawn from the inverse CDF of x Sigma(x),
          the angular momentum (dehnendf) or energy (shudf) by its analytic
          inverse transform, the orbital phase from the inverse of the radial
          angle-radius relation of the action-angle quadrature, and each orbit
          is replicated by the ratio of the epicycle frequency at its guiding
          radius to its radial frequency; the points' R, vR and vT are
          differentiable in the DF's parameters, the sample count is not.
        - 2010-07-10 - Started  - Bovy (NYU)
        - 2026-10-28 - Inverse-CDF sampling on every backend - Bovy (UofT)
        """
        key = kwargs.pop("key", None)
        if not los is None:
            return self.sampleLOS(
                los,
                deg=losdeg,
                n=n,
                maxd=maxd,
                nsigma=nsigma,
                targetSurfmass=targetSurfmass,
                targetSigma2=targetSigma2,
                key=key,
            )
        xp = self._sample_namespace(key)
        if not rrange is None:
            rrange = [
                conversion.parse_length(rrange[0], ro=self._ro),
                conversion.parse_length(rrange[1], ro=self._ro),
            ]
        EL_only = not returnROrbit and not returnOrbit
        physical = kwargs.get("use_physical", True) and self._roSet and self._voSet
        target = int(n * nphi)
        chunks, total, need, k = [], 0, int(n), key
        while total < target:  # draw again when the weighting leaves too few
            k, k1, k2 = grandom.split(k, 3)
            chunk = self._sample_chunk(
                need, nphi, rrange, returnOrbit, EL_only, k1, k2, xp
            )
            chunks.append(chunk)
            total += int(chunk.shape[0])
            need = max(1, int(n - total / nphi))
        concat = getattr(xp, "concat", None) or xp.concatenate
        out = concat(chunks, axis=0)[:target]
        if EL_only:
            return [list(el) for el in as_numpy(out)] if key is None else out
        return self._sample_orbits(out, key, physical)

    def _sample_chunk(self, n, nphi, rrange, want_phi, EL_only, key, key_phi, xp):
        """n draws of (E, Lz), or the weighted [R, vR, vT(, phi)] points of
        their orbits"""
        xg, F = self._sample_x_table(xp)
        u = self._sample_uniform(xp, key, (4, n), xg)
        x = linear_inverse_cdf_sample(xp, xg, F, u[0])
        E, L = self._sample_EL(x, u[1], xp)
        if EL_only:
            return xp.stack([E, L], axis=1)
        R, vR, vT, Or = self._sample_phase(E, L, x, u[2], xp)
        # a draw with |L| above the circular orbit's at E is no orbit (the
        # rare far tail of the Lz transform): dropped, as the old sampler did
        valid = 2.0 * (E - self._psp(x, use_physical=False)) - L**2.0 / x**2.0 >= 0.0
        idx = self._sample_multiplicity(x, R, Or, nphi, u[3], rrange, valid, xp)
        cols = [R[idx], vR[idx], vT[idx]]
        if want_phi:
            cols.append(
                2.0 * numpy.pi * self._sample_uniform(xp, key_phi, (idx.shape[0],), xg)
            )
        return xp.stack(cols, axis=1)

    def _sample_x_table(self, xp):
        """(x grid, CDF of x Sigma(x) c_Sigma(x)): the guiding-centre radius of
        a circular orbit's energy (dehnendf) or angular momentum (shudf)"""
        # a unit grid scaled by the extent (torch rejects tensor limits, and a
        # differentiated scale length keeps the grid differentiable)
        xmax = self._SAMPLE_XMAX * self._surfaceSigmaProfile._params[0]
        xg = xp.linspace(0.0, 1.0, self._SAMPLE_NX) * xmax
        logp = self._ssp("surfacemass", xg, log=True)
        if self._correct:
            logp = logp + self._corr.correct(xg, log=True)[0]
        F = trapezoid_cdf(xp, xg * xp.exp(logp), xg, 0)
        return xg, ensure_strictly_increasing(xp, F)

    def _sample_EL(self, x, u, xp):  # pragma: no cover
        raise NotImplementedError("'sample' method for this disk df is not implemented")

    def _sample_phase(self, E, L, x, u, xp):
        """(R, vR, vT, omega_R) of the orbits (E, L) at the uniform phase u of
        their radial oscillation, in the DF's power-law potential: the turning
        points by bracketed root-finding of the radial kinetic energy rad(r) =
        2 [E - Phi(r)] - L^2 / r^2 (then anchored to vanish exactly at both),
        the radial period and the angle-radius table from the panels of the
        spherical action-angle quadrature (r = rperi + s^2 from the
        pericentre, r = rap - s^2 from the apocentre, the turning points'
        square roots absorbed), the radius at the angle 2 pi u by inverting
        the table, vR from rad with the angle's sign. x is the circular
        radius of E (dehnendf) or of L (shudf), a point on the orbit."""
        from ..backend.optimize import brentq, iterate_bracket

        aAS = self._aA._aAS
        K = self._SAMPLE_NPHASE
        beta = self._beta

        def _b(p, r):  # the per-orbit parameters against r's trailing axes
            return xp.reshape(p, tuple(p.shape) + (1,) * (r.ndim - 1))

        def rad(r):
            Phi = _beta_branch(
                beta, lambda: xp.log(r), lambda b: r ** (2.0 * b) / (2.0 * b)
            )
            return 2.0 * (_b(E, r) - Phi) - _b(L, r) ** 2.0 / r**2.0

        # brackets: halve from x down to the centrifugal barrier, double up to
        # the potential's wall (the draws are bound)
        rlo = iterate_bracket(lambda r: xp.where(rad(r) > 0.0, r / 2.0, r), x / 2.0, 80)
        rhi = iterate_bracket(lambda r: xp.where(rad(r) > 0.0, r * 2.0, r), 2.0 * x, 80)
        rperi = brentq(rad, rlo, x, xtol=1e-14)
        rap = brentq(rad, x, rhi, xtol=1e-14)
        dp, da = rad(rperi), rad(rap)
        span = rap - rperi
        span_s = xp.where(span > 0.0, span, xp.ones_like(span))

        def rad_a(r):  # anchored: zero at the turning points to round-off
            return rad(r) - (
                _b(dp, r) * (_b(rap, r) - r) + _b(da, r) * (r - _b(rperi, r))
            ) / _b(span_s, r)

        Rmean = aAS._rmean_backend(xp, rperi, rap)
        sin_ = aAS._sqrt_pos(xp, Rmean - rperi)
        sout = aAS._sqrt_pos(xp, rap - Rmean)
        T = 2.0 * (
            aAS._panel_backend(xp, lambda s: rperi[:, None] + s**2.0, sin_, rad_a)
            + aAS._panel_backend(xp, lambda s: rap[:, None] - s**2.0, sout, rad_a)
        )
        # a circular draw (no libration): the epicycle frequency
        kappa = _sqrt(2.0 * (1.0 + beta)) * x ** (beta - 1.0)
        Or = xp.where(T > 0.0, 2.0 * numpy.pi / xp.where(T > 0.0, T, 1.0), kappa)
        # the angle-radius table in the panels' variable: the inner panel from
        # the pericentre up to Rmean, the outer one from the apocentre down to it
        fin = xp.astype(xp.arange(K), sin_.dtype) / K  # 0 .. (K-1)/K
        fout = 1.0 - xp.astype(xp.arange(K + 1), sin_.dtype) / K  # 1 .. 0
        lin = sin_[:, None] * fin[None, :]  # (N, K) panel limits
        lout = sout[:, None] * fout[None, :]  # (N, K+1)
        th_in = Or[:, None] * aAS._panel_backend(
            xp, lambda s: rperi[:, None, None] + s**2.0, lin, rad_a
        )
        th_out = numpy.pi - Or[:, None] * aAS._panel_backend(
            xp, lambda s: rap[:, None, None] - s**2.0, lout, rad_a
        )
        concat = getattr(xp, "concat", None) or xp.concatenate
        theta = concat([th_in, th_out], axis=1) / numpy.pi
        rr = concat([rperi[:, None] + lin**2.0, rap[:, None] - lout**2.0], axis=1)
        # a circular draw's table is degenerate: keep every row finite and
        # increasing so that it cannot poison a gradient
        qgrid = xp.astype(xp.arange(2 * K + 1), theta.dtype)
        theta = xp.where(xp.isfinite(theta), theta, qgrid[None, :] / (2.0 * K))
        rr = xp.where(xp.isfinite(rr), rr, xp.ones_like(rr))
        theta = self._sample_rows_increasing(xp, theta)
        # the half orbit: outgoing (vR > 0) from the pericentre for u > 1/2,
        # ingoing from the apocentre otherwise, mirrored onto the same table
        out = u > 0.5
        up = xp.where(out, 2.0 * u - 1.0, 1.0 - 2.0 * u)
        q = batched_inverse_cdf_sample(xp, qgrid, theta, up)
        r, _, _ = self._sample_lerp_rows(xp, rr, xp.arange(q.shape[0]), q)
        vr = xp.where(out, 1.0, -1.0) * aAS._sqrt_pos(xp, rad_a(r))
        return r, vr, L / r, Or

    def _sample_multiplicity(self, x, R, Or, nphi, u, rrange, valid, xp):
        """The index of every returned point: each orbit replicated
        kappa(x)/omega_R nphi times on average (the integer part, plus one
        with the fractional probability), none outside rrange or invalid.

        The DF's measure is f(E, L) T_R dE dL; the draws of x and L (or E)
        carry x Sigma(x) dx times the DF's exponential, and dE (or dL) along
        the circular orbits is x kappa^2 / 2 (or x kappa^2 / 2 Omega) dx, so
        the weight is T_R kappa^2 / Omega, i.e. kappa(x) / omega_R for a
        power-law rotation curve (kappa / Omega constant). The sampler before
        2026-10 took kappa at the sampled radius and omega_R of the orbit with
        vR = 0 at x, which biased the mean azimuthal velocity by a few per
        cent."""
        kappa = _sqrt(2.0 * (1.0 + self._beta)) * x ** (self._beta - 1.0)
        ratio = kappa / Or * nphi
        mult = xp.ceil(ratio) - 1.0
        counts = xp.astype(mult + xp.astype(u < ratio - mult, mult.dtype), xp.int64)
        counts = xp.where(valid, counts, 0)
        if not rrange is None:
            counts = xp.where((R >= rrange[0]) & (R <= rrange[1]), counts, 0)
        return xp.repeat(xp.arange(R.shape[0]), counts)

    def _estimatemeanvR(self, R, phi=0.0, log=False):
        """
        Quickly estimate the mean radial velocity at a given radius R.

        Parameters
        ----------
        R : float
            Radius at which to evaluate (/ro).
        phi : float, optional
            Azimuth angle (not used).
        log : bool, optional
            If True, return the logarithm of the target Sigma_R^2(R).

        Returns
        -------
        float
            The target Sigma_R^2(R).

        Notes
        -----
        - 2010-03-28 - Written - Bovy (NYU)

        """
        return 0.0

    def _estimatemeanvT(self, R, phi=0.0, log=False):
        """
        Quickly estimate the mean tangential velocity at a given radius.

        Parameters
        ----------
        R : float
            Radius at which to evaluate (/ro).
        phi : float, optional
            Azimuth angle (not used).
        log : bool, optional
            If True, return the logarithm of the estimate.

        Returns
        -------
        float
            The estimated mean tangential velocity.

        Notes
        -----
        - 2010-03-28 - Written - Bovy (NYU)

        """
        return R**self._beta - self._asymmetricdrift(R)

    def _estimateSigmaR2(self, R, phi=0.0, log=False):
        """
        Quickly estimate SigmaR2.

        Parameters
        ----------
        R : float
            Radius at which to evaluate (/ro).
        phi : float, optional
            Azimuth (not used).
        log : bool, optional
            If True, return the log (default: False).

        Returns
        -------
        float
            Target Sigma_R^2(R).

        Notes
        -----
        - 2010-03-28 - Written - Bovy (NYU)
        """
        return self._ssp("sigma2", R, log=log)

    def _estimateSigmaT2(self, R, phi=0.0, log=False):
        """
        Quickly estimate SigmaT2.

        Parameters
        ----------
        R : float
            Radius at which to evaluate (/ro).
        phi : float, optional
            Azimuth (not used).
        log : bool, optional
            If True, return the log (default: False).

        Returns
        -------
        float
            Target Sigma_R^2(R).

        Notes
        -----
        - 2010-03-28 - Written - Bovy (NYU)

        """
        if log:
            return self._ssp("sigma2", R, log=log) - 2.0 * numpylog(self._gamma)
        else:
            return self._ssp("sigma2", R, log=log) / self._gamma**2.0


class dehnendf(diskdf):
    """Dehnen's 'new' df"""

    def __init__(
        self,
        surfaceSigma=expSurfaceSigmaProfile,
        profileParams=(1.0 / 3.0, 1.0, 0.2),
        correct=False,
        beta=0.0,
        **kwargs,
    ):
        """
        Initialize a Dehnen 'new' DF.

        Parameters
        ----------
        surfaceSigma : instance or class name of the target surface density and sigma_R profile, optional
            Default: both exponential.
        profileParams : tuple, optional
            Parameters of the surface and sigma_R profile: (xD,xS,Sro) where:
                * xD - disk surface mass scalelength (can be Quantity)
                * xS - disk velocity dispersion scalelength (can be Quantity)
                * Sro - disk velocity dispersion at Ro (can be Quantity)
            Directly given to the 'surfaceSigmaProfile class, so could be anything that class takes.
        beta : float, optional
            Power-law index of the rotation curve.
        correct : bool, optional
            If True, correct the DF.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).
        **kwargs: dict, optional
            DFcorrection kwargs (except for those already specified).

        Notes
        -----
        - 2010-03-10 - Written - Bovy (NYU)

        """
        return diskdf.__init__(
            self,
            surfaceSigma=surfaceSigma,
            profileParams=profileParams,
            correct=correct,
            dftype="dehnen",
            beta=beta,
            **kwargs,
        )

    @backend_input("E", "L")
    def eval(self, E, L, logSigmaR=0.0, logsigmaR2=0.0):
        """
        Evaluate the distribution function.

        Parameters
        ----------
        E : float or Quantity
            Energy.
        L : float or Quantity
            Angular momentum.
        logSigmaR : float, optional
            Logarithm of the radial velocity dispersion.
        logsigmaR2 : float, optional
            Logarithm of the square of the radial velocity dispersion.

        Returns
        -------
        float
            DF(E,L)

        Notes
        -----
        - 2010-03-10 - Written - Bovy (NYU).
        - 2010-03-28 - Moved to dehnenDF - Bovy (NYU).
        """
        if _PROFILE:  # pragma: no cover
            import time

            start = time.time()
        E = conversion.parse_energy(E, vo=self._vo)
        L = conversion.parse_angmom(L, ro=self._ro, vo=self._vo)
        if self._backend_path(E, L) or resolve_namespace() is not numpy:
            return self._eval_backend(E, L, logSigmaR, logsigmaR2)
        # Calculate Re,LE, OmegaE
        if self._beta == 0.0:
            xE = numpy.exp(E - 0.5)
            logOLLE = numpylog(L / xE - 1.0)
        else:  # non-flat rotation curve
            xE = (2.0 * E / (1.0 + 1.0 / self._beta)) ** (1.0 / 2.0 / self._beta)
            logOLLE = self._beta * numpylog(xE) + numpylog(L / xE - xE**self._beta)
        if _PROFILE:  # pragma: no cover
            one_time = time.time() - start
            start = time.time()
        if self._correct:
            correction = self._corr.correct(xE, log=True)
        else:
            correction = numpy.zeros(2)
        if _PROFILE:  # pragma: no cover
            corr_time = time.time() - start
            start = time.time()
        SRE2 = self._ssp("sigma2", xE, log=True) + correction[1]
        if _PROFILE:  # pragma: no cover
            targSigma_time = time.time() - start
            start = time.time()
            out = (
                self._gamma
                * numpy.exp(
                    logsigmaR2
                    - SRE2
                    + self._ssp("surfacemass", xE, log=True)
                    - logSigmaR
                    + numpy.exp(logOLLE - SRE2)
                    + correction[0]
                )
                / 2.0
                / numpy.pi
            )
            out_time = time.time() - start
            tot_time = one_time + corr_time + targSigma_time + out_time
            print(
                one_time / tot_time,
                corr_time / tot_time,
                targSigma_time / tot_time,
                out_time / tot_time,
                tot_time,
            )
            return out
        else:
            return (
                self._gamma
                * numpy.exp(
                    logsigmaR2
                    - SRE2
                    + self._ssp("surfacemass", xE, log=True)
                    - logSigmaR
                    + numpy.exp(logOLLE - SRE2)
                    + correction[0]
                )
                / 2.0
                / numpy.pi
            )

    def _eval_backend(self, E, L, logSigmaR, logsigmaR2):
        """Backend (jax/torch) eval; byte-matches the numpy scimath real part.
        exp(numpylog(X) - SRE2) = X * exp(-SRE2), so the scimath log/real pair is
        replaced by the signed algebraic factor X (differentiable, no complex)."""
        xp, (E, L) = self._backend_coerce(E, L)

        def flat():
            xE = xp.exp(E - 0.5)
            return xE, L / xE - 1.0, E == E  # Omega_E (L - L_E) numerator

        def nonflat(beta):
            # a falling rotation curve has unbound energies (E >= 0, no
            # circular radius): the DF vanishes there (the numpy path's NaN
            # was what the rejection samplers silently rejected); the guard
            # keeps the dead branch finite for AD
            base = 2.0 * E / (1.0 + 1.0 / beta)
            bound = base > 0.0
            xE = xp.where(bound, base, 1.0) ** (1.0 / 2.0 / beta)
            return xE, xE**beta * (L / xE - xE**beta), bound

        xE, olfac, bound = _beta_branch(self._beta, flat, nonflat)
        correction = self._corr.correct(xE, log=True) if self._correct else xp.zeros(2)
        SRE2 = self._ssp("sigma2", xE, log=True) + correction[1]
        value = (
            scalar_like(E, self._gamma)
            * xp.exp(
                logsigmaR2
                - SRE2
                + self._ssp("surfacemass", xE, log=True)
                - logSigmaR
                + olfac * xp.exp(-SRE2)
                + correction[0]
            )
            / 2.0
            / numpy.pi
        )
        return xp.where(bound, value, 0.0)

    def _sample_EL(self, xE, u, xp):
        """(E, Lz) at the guiding-centre radii xE: E the circular energy, Lz
        exponential below the circular angular momentum (the DF's L
        dependence), u uniform"""
        E = _beta_branch(
            self._beta,
            lambda: xp.log(xE) + 0.5,
            lambda beta: 0.5 * xE ** (2.0 * beta) * (1.0 + 1.0 / beta),
        )
        LCE = xE ** (self._beta + 1.0)
        OR = xE ** (self._beta - 1.0)
        Lz = self._ssp("sigma2", xE) * xp.log(u) / OR
        if self._correct:
            Lz = Lz * self._corr.correct(xE, log=False)[1]
        return E, Lz + LCE

    def _dlnfdR(self, R, vR, vT):
        # Calculate a bunch of stuff that we need
        (R, vR, vT), _log, _exp = self._forced_logexp(R, vR, vT)

        def flat():
            E = vR**2.0 / 2.0 + vT**2.0 / 2.0 + _log(R)
            xE = _exp(E - 0.5)
            return E, xE, xE**-1.0, xE, xE / R

        def nonflat(beta):  # non-flat rotation curve
            E = vR**2.0 / 2.0 + vT**2.0 / 2.0 + 1.0 / 2.0 / beta * R ** (2.0 * beta)
            xE = (2.0 * E / (1.0 + 1.0 / beta)) ** (1.0 / 2.0 / beta)
            OE = xE ** (beta - 1.0)
            LCE = xE ** (beta + 1.0)
            return E, xE, OE, LCE, xE / 2.0 / beta / E * R ** (2.0 * beta - 1.0)

        E, xE, OE, LCE, dRedR = _beta_branch(self._beta, flat, nonflat)
        return (
            self._dlnfdRe(R, vR, vT, E=E, xE=xE, OE=OE, LCE=LCE) * dRedR
            + self._dlnfdl(R, vR, vT, E=E, xE=xE, OE=OE) * vT
        )

    def _dlnfdvR(self, R, vR, vT):
        # Calculate a bunch of stuff that we need
        (R, vR, vT), _log, _exp = self._forced_logexp(R, vR, vT)
        if self._beta == 0.0:
            E = vR**2.0 / 2.0 + vT**2.0 / 2.0 + _log(R)
            xE = _exp(E - 0.5)
            OE = xE**-1.0
            LCE = xE
            dRedvR = xE * vR
        else:  # non-flat rotation curve
            E = (
                vR**2.0 / 2.0
                + vT**2.0 / 2.0
                + 1.0 / 2.0 / self._beta * R ** (2.0 * self._beta)
            )
            xE = (2.0 * E / (1.0 + 1.0 / self._beta)) ** (1.0 / 2.0 / self._beta)
            OE = xE ** (self._beta - 1.0)
            LCE = xE ** (self._beta + 1.0)
            dRedvR = xE / 2.0 / self._beta / E * vR
        return self._dlnfdRe(R, vR, vT, E=E, xE=xE, OE=OE, LCE=LCE) * dRedvR

    def _dlnfdvT(self, R, vR, vT):
        # Calculate a bunch of stuff that we need
        (R, vR, vT), _log, _exp = self._forced_logexp(R, vR, vT)
        if self._beta == 0.0:
            E = vR**2.0 / 2.0 + vT**2.0 / 2.0 + _log(R)
            xE = _exp(E - 0.5)
            OE = xE**-1.0
            LCE = xE
            dRedvT = xE * vT
        else:  # non-flat rotation curve
            E = (
                vR**2.0 / 2.0
                + vT**2.0 / 2.0
                + 1.0 / 2.0 / self._beta * R ** (2.0 * self._beta)
            )
            xE = (2.0 * E / (1.0 + 1.0 / self._beta)) ** (1.0 / 2.0 / self._beta)
            OE = xE ** (self._beta - 1.0)
            LCE = xE ** (self._beta + 1.0)
            dRedvT = xE / 2.0 / self._beta / E * vT
        return (
            self._dlnfdRe(R, vR, vT, E=E, xE=xE, OE=OE, LCE=LCE) * dRedvT
            + self._dlnfdl(R, vR, vT, E=E, xE=xE, OE=OE) * R
        )

    def _dlnfdRe(self, R, vR, vT, E=None, xE=None, OE=None, LCE=None):
        """d ln f(x,v) / d R_e"""
        # Calculate a bunch of stuff that we need
        (R, vR, vT), _log, _exp = self._forced_logexp(R, vR, vT)
        if E is None or xE is None or OE is None or LCE is None:
            if self._beta == 0.0:
                E = vR**2.0 / 2.0 + vT**2.0 / 2.0 + _log(R)
                xE = _exp(E - 0.5)
                OE = xE**-1.0
                LCE = xE
            else:  # non-flat rotation curve
                E = (
                    vR**2.0 / 2.0
                    + vT**2.0 / 2.0
                    + 1.0 / 2.0 / self._beta * R ** (2.0 * self._beta)
                )
                xE = (2.0 * E / (1.0 + 1.0 / self._beta)) ** (1.0 / 2.0 / self._beta)
                OE = xE ** (self._beta - 1.0)
                LCE = xE ** (self._beta + 1.0)
        L = R * vT
        sigma2xE = self._ssp("sigma2", xE, log=False)
        return (
            self._ssp("surfacemassDerivative", xE, log=True)
            - (1.0 + OE * (L - LCE) / sigma2xE)
            * self._ssp("sigma2Derivative", xE, log=True)
            + (L - LCE) / sigma2xE * (self._beta - 1.0) * xE ** (self._beta - 2.0)
            - OE * (self._beta + 1.0) / sigma2xE * xE**self._beta
        )

    def _dlnfdl(self, R, vR, vT, E=None, xE=None, OE=None):
        # Calculate a bunch of stuff that we need
        (R, vR, vT), _log, _exp = self._forced_logexp(R, vR, vT)
        if E is None or xE is None or OE is None:
            if self._beta == 0.0:
                E = vR**2.0 / 2.0 + vT**2.0 / 2.0 + _log(R)
                xE = _exp(E - 0.5)
                OE = xE**-1.0
            else:  # non-flat rotation curve
                E = (
                    vR**2.0 / 2.0
                    + vT**2.0 / 2.0
                    + 1.0 / 2.0 / self._beta * R ** (2.0 * self._beta)
                )
                xE = (2.0 * E / (1.0 + 1.0 / self._beta)) ** (1.0 / 2.0 / self._beta)
                OE = xE ** (self._beta - 1.0)
        sigma2xE = self._ssp("sigma2", xE, log=False)
        return OE / sigma2xE


class shudf(diskdf):
    """Shu's df (1969)"""

    def __init__(
        self,
        surfaceSigma=expSurfaceSigmaProfile,
        profileParams=(1.0 / 3.0, 1.0, 0.2),
        correct=False,
        beta=0.0,
        **kwargs,
    ):
        """
        Initialize a Shu DF.

        Parameters
        ----------
        surfaceSigma : instance or class name of the target surface density and sigma_R profile, optional
            Default: both exponential.
        profileParams : tuple, optional
            Parameters of the surface and sigma_R profile: (xD,xS,Sro) where
                * xD - disk surface mass scalelength (can be Quantity)
                * xS - disk velocity dispersion scalelength (can be Quantity)
                * Sro - disk velocity dispersion at Ro (can be Quantity)
            Directly given to the 'surfaceSigmaProfile class, so could be anything that class takes.
        beta : float, optional
            Power-law index of the rotation curve.
        correct : bool, optional
            If True, correct the DF.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).
        **kwargs: dict, optional
            DFcorrection kwargs (except for those already specified).

        Notes
        -----
        - 2010-05-09 - Written - Bovy (NYU)

        """
        return diskdf.__init__(
            self,
            surfaceSigma=surfaceSigma,
            profileParams=profileParams,
            correct=correct,
            dftype="shu",
            beta=beta,
            **kwargs,
        )

    @backend_input("E", "L")
    def eval(self, E, L, logSigmaR=0.0, logsigmaR2=0.0):
        """
        Evaluate the distribution function.

        Parameters
        ----------
        E : float
            Energy (/vo^2).
        L : float
            Angular momentum (/ro/vo).
        logSigmaR : float, optional
            Logarithm of the radial velocity dispersion squared.
        logsigmaR2 : float, optional
            Logarithm of the radial velocity dispersion squared.

        Returns
        -------
        float
            DF(E,L).

        Notes
        -----
        - 2010-05-09 - Written - Bovy (NYU)
        """
        E = conversion.parse_energy(E, vo=self._vo)
        L = conversion.parse_angmom(L, ro=self._ro, vo=self._vo)
        if self._backend_path(E, L) or resolve_namespace() is not numpy:
            return self._eval_backend(E, L, logSigmaR, logsigmaR2)
        # Calculate RL,LL, OmegaL
        if self._beta == 0.0:
            xL = L
            logECLE = numpylog(-numpylog(xL) - 0.5 + E)
        else:  # non-flat rotation curve
            xL = L ** (1.0 / (self._beta + 1.0))
            logECLE = numpylog(
                -0.5 * (1.0 / self._beta + 1.0) * xL ** (2.0 * self._beta) + E
            )
        if xL < 0.0:  # We must remove counter-rotating mass
            return 0.0
        if self._correct:
            correction = self._corr.correct(xL, log=True)
        else:
            correction = numpy.zeros(2)
        SRE2 = self._ssp("sigma2", xL, log=True) + correction[1]
        return (
            self._gamma
            * numpy.exp(
                logsigmaR2
                - SRE2
                + self._ssp("surfacemass", xL, log=True)
                - logSigmaR
                - numpy.exp(logECLE - SRE2)
                + correction[0]
            )
            / 2.0
            / numpy.pi
        )

    def _eval_backend(self, E, L, logSigmaR, logsigmaR2):
        """Backend (jax/torch) eval; byte-matches the numpy scimath real part.
        exp(numpylog(X) - SRE2) = X * exp(-SRE2) recovers the signed factor X; the
        xL<0 counter-rotating cut is a data branch -> xp.where (log guarded)."""
        xp, (E, L) = self._backend_coerce(E, L)

        # counter-rotating (L <= 0): no mass; the guards keep the dead branch
        # finite (a fractional power of a negative L is NaN, which the mask on
        # xL used to let through as a finite wrong value)
        Lsafe = xp.where(L > 0.0, L, 1.0)

        def flat():
            return L, Lsafe, E - 0.5 - xp.log(Lsafe)  # E - E_c(L)

        def nonflat(beta):
            xL = Lsafe ** (1.0 / (beta + 1.0))
            return xL, xL, E - 0.5 * (1.0 / beta + 1.0) * xL ** (2.0 * beta)

        xL, xLsafe, eclfac = _beta_branch(self._beta, flat, nonflat)
        correction = (
            self._corr.correct(xLsafe, log=True) if self._correct else xp.zeros(2)
        )
        SRE2 = self._ssp("sigma2", xLsafe, log=True) + correction[1]
        value = (
            scalar_like(E, self._gamma)
            * xp.exp(
                logsigmaR2
                - SRE2
                + self._ssp("surfacemass", xLsafe, log=True)
                - logSigmaR
                - eclfac * xp.exp(-SRE2)
                + correction[0]
            )
            / 2.0
            / numpy.pi
        )
        return xp.where(L <= 0.0, 0.0, value)

    def _sample_EL(self, xL, u, xp):
        """(E, Lz) at the guiding-centre radii xL: Lz the circular angular
        momentum, E exponential above the circular energy (the DF's E
        dependence), u uniform"""
        Lz = xL ** (self._beta + 1.0)
        ECL = _beta_branch(
            self._beta,
            lambda: xp.log(xL) + 0.5,
            lambda beta: 0.5 * (1.0 / beta + 1.0) * xL ** (2.0 * beta),
        )
        E = -self._ssp("sigma2", xL) * xp.log(u)
        if self._correct:
            E = E * self._corr.correct(xL, log=False)[1]
        return E + ECL, Lz

    def _dlnfdR(self, R, vR, vT):
        # Calculate a bunch of stuff that we need
        (R, vR, vT), _, _ = self._forced_logexp(R, vR, vT)
        E, L = vRvTRToEL(vR, vT, R, self._beta, self._dftype)
        if self._beta == 0.0:
            xL = L
            dRldR = vT
            if is_backend_array(xL):  # guard log on the masked (xL<0) side
                xp = get_namespace(xL)
                ECL = xp.log(xp.where(xL > 0.0, xL, 1.0)) + 0.5
            else:
                ECL = numpylog(xL) + 0.5
            dECLEdR = 0.0
        else:  # non-flat rotation curve
            xL = L ** (1.0 / (self._beta + 1.0))
            dRldR = L ** (1.0 / (self._beta + 1.0)) / R / (self._beta + 1.0)
            ECL = 0.5 * (1.0 / self._beta + 1.0) * xL ** (2.0 * self._beta)
            dECLdRl = (1.0 + self._beta) * xL ** (2.0 * self._beta - 1)
            dEdR = R ** (2.0 * self._beta - 1.0)
            dECLEdR = dECLdRl * dRldR - dEdR
        sigma2xL = self._ssp("sigma2", xL, log=False)
        return (
            self._ssp("surfacemassDerivative", xL, log=True)
            - (1.0 + (ECL - E) / sigma2xL) * self._ssp("sigma2Derivative", xL, log=True)
        ) * dRldR + dECLEdR / sigma2xL

    def _dlnfdvR(self, R, vR, vT):
        # Calculate a bunch of stuff that we need
        (R, vR, vT), _, _ = self._forced_logexp(R, vR, vT)
        E, L = vRvTRToEL(vR, vT, R, self._beta, self._dftype)
        if self._beta == 0.0:
            xL = L
        else:  # non-flat rotation curve
            xL = L ** (1.0 / (self._beta + 1.0))
        sigma2xL = self._ssp("sigma2", xL, log=False)
        return -vR / sigma2xL

    def _dlnfdvT(self, R, vR, vT):
        # Calculate a bunch of stuff that we need
        (R, vR, vT), _log, _ = self._forced_logexp(R, vR, vT)
        E, L = vRvTRToEL(vR, vT, R, self._beta, self._dftype)
        if self._beta == 0.0:
            xL = L
            dRldvT = R
            ECL = _log(xL) + 0.5
            dECLEdvT = 1.0 / vT - vT
        else:  # non-flat rotation curve
            xL = L ** (1.0 / (self._beta + 1.0))
            dRldvT = L ** (1.0 / (self._beta + 1.0)) / vT / (self._beta + 1.0)
            ECL = 0.5 * (1.0 / self._beta + 1.0) * xL ** (2.0 * self._beta)
            dECLdRl = (1.0 + self._beta) * xL ** (2.0 * self._beta - 1)
            dEdvT = vT
            dECLEdvT = dECLdRl * dRldvT - dEdvT
        sigma2xL = self._ssp("sigma2", xL, log=False)
        return (
            self._ssp("surfacemassDerivative", xL, log=True)
            - (1.0 + (ECL - E) / sigma2xL) * self._ssp("sigma2Derivative", xL, log=True)
        ) * dRldvT + dECLEdvT / sigma2xL


class schwarzschilddf(shudf):
    """Schwarzschild's df"""

    def __init__(
        self,
        surfaceSigma=expSurfaceSigmaProfile,
        profileParams=(1.0 / 3.0, 1.0, 0.2),
        correct=False,
        beta=0.0,
        **kwargs,
    ):
        """
        Initialize a Schwarzschild DF.

        Parameters
        ----------
        surfaceSigma : instance or class name of the target surface density and sigma_R profile, optional
            (default: both exponential)
        profileParams : tuple, optional
            Parameters of the surface and sigma_R profile: (xD,xS,Sro) where
                * xD - disk surface mass scalelength (can be Quantity)
                * xS - disk velocity dispersion scalelength (can be Quantity)
                * Sro - disk velocity dispersion at Ro (can be Quantity)
            Directly given to the 'surfaceSigmaProfile class, so could be anything that class takes
        beta : float, optional
            Power-law index of the rotation curve
        correct : bool, optional
            If True, correct the DF
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).
        **kwargs : dict, optional
            DFcorrection kwargs (except for those already specified)

        Notes
        -----
        - 2017-09-17 - Written - Bovy (UofT)

        """
        # Schwarzschild == Shu w/ energy computed in epicycle approx.
        # so all functions are the same as in Shu, only thing different is
        # how E is computed
        return diskdf.__init__(
            self,
            surfaceSigma=surfaceSigma,
            profileParams=profileParams,
            correct=correct,
            dftype="schwarzschild",
            beta=beta,
            **kwargs,
        )


def _surfaceIntegrand(vR, vT, R, df, logSigmaR, logsigmaR2, sigmaR1, gamma):
    """Internal function that is the integrand for the surface mass integration"""
    E, L = _vRpvTpRToEL(vR, vT, R, df._beta, sigmaR1, gamma, df._dftype)
    return df.eval(E, L, logSigmaR, logsigmaR2) * 2.0 * numpy.pi / df._gamma  # correct


def _sigma2surfaceIntegrand(vR, vT, R, df, logSigmaR, logsigmaR2, sigmaR1, gamma):
    """Internal function that is the integrand for the sigma-squared times
    surface mass integration"""
    E, L = _vRpvTpRToEL(vR, vT, R, df._beta, sigmaR1, gamma, df._dftype)
    return (
        vR**2.0 * df.eval(E, L, logSigmaR, logsigmaR2) * 2.0 * numpy.pi / df._gamma
    )  # correct


def _vmomentsurfaceIntegrand(
    vR, vT, R, df, logSigmaR, logsigmaR2, sigmaR1, gamma, n, m
):
    """Internal function that is the integrand for the velocity moment times
    surface mass integration"""
    E, L = _vRpvTpRToEL(vR, vT, R, df._beta, sigmaR1, gamma, df._dftype)
    return (
        vR**n
        * vT**m
        * df.eval(E, L, logSigmaR, logsigmaR2)
        * 2.0
        * numpy.pi
        / df._gamma
    )  # correct


def _vmomentderivsurfaceIntegrand(
    vR, vT, R, df, logSigmaR, logsigmaR2, sigmaR1, gamma, n, m, deriv
):
    """Internal function that is the integrand for the derivative of velocity
    moment times surface mass integration"""
    E, L = _vRpvTpRToEL(vR, vT, R, df._beta, sigmaR1, gamma, df._dftype)
    if deriv.lower() == "r":
        return (
            vR**n
            * vT**m
            * df.eval(E, L, logSigmaR, logsigmaR2)
            * 2.0
            * numpy.pi
            / df._gamma
            * df._dlnfdR(R, vR * sigmaR1, vT * sigmaR1 / gamma)
        )  # correct
    else:
        return 0.0


def _vRpvTpRToEL(vR, vT, R, beta, sigmaR1, gamma, dftype="dehnen"):
    """Internal function that calculates E and L given velocities normalized by the velocity dispersion"""
    vR = vR * sigmaR1  # out-of-place (backend arrays are immutable)
    vT = vT * (sigmaR1 / gamma)  # keep the *= grouping for numpy byte-identity
    return vRvTRToEL(vR, vT, R, beta, dftype)


def _oned_intFunc(x, twodfunc, gfun, hfun, tol, args):
    """Internal function for bovy_dblquad"""
    thisargs = copy.deepcopy(args)
    thisargs.insert(0, x)
    return quadpack.romberg(twodfunc, gfun(x), hfun(x), args=thisargs, tol=tol)


def bovy_dblquad(func, a, b, gfun, hfun, args=(), tol=1.48e-08):
    """
    Compute a double integral using Romberg integration for the one-dimensional integrals and a specified tolerance.

    Parameters
    ----------
    func : callable
        Function of two variables to integrate.
    a : float
        Lower limit of integration in the outer integral.
    b : float
        Upper limit of integration in the outer integral.
    gfun : callable
        Function of one variable that returns the lower limit of integration in the inner integral for a given value of the outer variable.
    hfun : callable
        Function of one variable that returns the upper limit of integration in the inner integral for a given value of the outer variable.
    args : tuple, optional
        Extra arguments to pass to the integrand function.
    tol : float, optional
        Desired absolute tolerance.

    Returns
    -------
    float
        The value of the double integral.

    Notes
    -----
    - 2010-03-11 - Written - Bovy (NYU)
    """
    return quadpack.romberg(
        _oned_intFunc, a, b, args=(func, gfun, hfun, tol, args), tol=tol
    )


class DFcorrection:
    """Class that contains the corrections necessary to reach
    exponential profiles"""

    def __init__(self, **kwargs):
        """
        Initialize the corrections: set them, load them, or calculate and save them.

        Parameters
        ----------
        corrections : numpy.ndarray, optional
            If set, these are the corrections and they should be used as such.
        npoints : int, optional
            Number of points from 0 to Rmax.
        rmax : float, optional
            Correct up to this radius (/ro) (default: 5).
        savedir : str, optional
            Save the corrections in this directory.
        surfaceSigmaProfile : object
            Target surfacemass and sigma_R^2 instance.
        beta : float, optional
            Power-law index of the rotation curve (when calculating).
        dftype : class, optional
            Classname of the DF.
        niter : int, optional
            Number of iterations to perform to calculate the corrections.
        interp_k : str, optional
            'k' keyword to give to InterpolatedUnivariateSpline.

        Notes
        -----
        - 2010-03-10 - Written - Bovy (NYU)

        """
        if not "surfaceSigmaProfile" in kwargs:
            raise DFcorrectionError("surfaceSigmaProfile not given")
        else:
            self._surfaceSigmaProfile = kwargs["surfaceSigmaProfile"]
        self._rmax = kwargs.get("rmax", 5.0)
        self._niter = kwargs.get("niter", 20)
        if not "npoints" in kwargs:
            if "corrections" in kwargs:
                self._npoints = kwargs["corrections"].shape[0]
            else:  # pragma: no cover
                self._npoints = 151  # would take too long to cover
        else:
            self._npoints = kwargs["npoints"]
        self._dftype = kwargs.get("dftype", dehnendf)
        self._beta = kwargs.get("beta", 0.0)
        self._rs = numpy.linspace(_RMIN, self._rmax, self._npoints)
        self._interp_k = kwargs.get("interp_k", _INTERPDEGREE)
        if "corrections" in kwargs:
            self._corrections = kwargs["corrections"]
            if not len(self._corrections) == self._npoints:
                raise DFcorrectionError(
                    "Number of corrections has to be equal to the number of points npoints"
                )
        elif any(
            is_backend_array(p)
            for p in (*getattr(self._surfaceSigmaProfile, "_params", ()), self._beta)
        ):
            # backend parameters: the parameter-keyed save file cannot hold (or
            # be keyed by) them, so compute on the backend, niter iterations
            self._corrections = self._calc_corrections_backend()
        else:
            self._savedir = kwargs.get("savedir", _CORRECTIONSDIR)
            self._savefilename = self._createSavefilename(self._niter)
            if os.path.exists(self._savefilename):
                savefile = open(self._savefilename, "rb")
                self._corrections = numpy.array(pickle.load(savefile))
                savefile.close()
            else:  # Calculate the corrections
                self._corrections = self._calc_corrections()
        if is_backend_array(self._corrections):
            self._backend_tables = self._backend_interp()
            return None
        # Interpolation; smoothly go to zero
        interpRs = numpy.append(self._rs, 2.0 * self._rmax)
        self._surfaceInterpolate = interpolate.InterpolatedUnivariateSpline(
            interpRs,
            numpylog(numpy.append(self._corrections[:, 0], 1.0)),
            k=self._interp_k,
        )
        self._sigma2Interpolate = interpolate.InterpolatedUnivariateSpline(
            interpRs,
            numpylog(numpy.append(self._corrections[:, 1], 1.0)),
            k=self._interp_k,
        )
        # Interpolation for R < _RMIN
        surfaceInterpolateSmallR = interpolate.UnivariateSpline(
            interpRs[0 : _INTERPDEGREE + 2],
            numpylog(self._corrections[0 : _INTERPDEGREE + 2, 0]),
            k=_INTERPDEGREE,
        )
        self._surfaceDerivSmallR = surfaceInterpolateSmallR.derivatives(interpRs[0])[1]
        sigma2InterpolateSmallR = interpolate.UnivariateSpline(
            interpRs[0 : _INTERPDEGREE + 2],
            numpylog(self._corrections[0 : _INTERPDEGREE + 2, 1]),
            k=_INTERPDEGREE,
        )
        self._sigma2DerivSmallR = sigma2InterpolateSmallR.derivatives(interpRs[0])[1]
        # Built here, not lazily: a first build inside a traced call compiles scipy
        self._backend_tables = self._numpy_tables()
        return None

    def _backend_interp(self):
        """(x, c, log corrections at _RMIN, d log corrections/dR at _RMIN) of the
        log-correction splines, for backend corrections: the not-a-knot cubic
        is InterpolatedUnivariateSpline(k=3)'s; the small-R slope is that of
        the least-squares cubic through the first _INTERPDEGREE+2 points, which
        is UnivariateSpline's (default s) for corrections this smooth"""
        xp = namespace_from_arrays((self._corrections,))
        x = numpy.append(self._rs, 2.0 * self._rmax)
        concat = getattr(xp, "concat", None) or xp.concatenate
        logc = xp.log(
            concat([self._corrections, xp.ones_like(self._corrections[:1])], axis=0)
        )
        if self._interp_k == 3:
            c = cubic_spline_coeffs(xp, x, logc, bc="not-a-knot")
        elif self._interp_k == 1:
            dx = asarray_on_device(xp, numpy.diff(x), device_of(logc))[:, None]
            c = xp.stack([(logc[1:] - logc[:-1]) / dx, logc[:-1]], axis=0)
        else:
            raise NotImplementedError(
                "DFcorrection with backend parameters supports interp_k=1 or 3"
            )
        xs = x[: _INTERPDEGREE + 2]
        V = numpy.vander(xs - xs[0], _INTERPDEGREE + 1, increasing=True)
        w = numpy.linalg.pinv(V)[1]  # d/dR at xs[0] of the least-squares cubic
        w = asarray_on_device(xp, w, device_of(logc))
        dsmall = xp.sum(w[:, None] * logc[: _INTERPDEGREE + 2], axis=0)
        return x, c, logc[0], dsmall

    def _numpy_tables(self):
        """_backend_interp's tables from the numpy splines (the same polynomials)"""
        x, c0 = spline_to_ppoly(self._surfaceInterpolate)
        _, c1 = spline_to_ppoly(self._sigma2Interpolate)
        return (
            x,
            numpy.stack([c0, c1], axis=-1),
            numpylog(self._corrections[0]),
            numpy.array([self._surfaceDerivSmallR, self._sigma2DerivSmallR]),
        )

    def _correct_backend(self, R, log):
        x, c, lc0, dsmall = self._backend_tables
        xp = prefer_backend_namespace(R, self._corrections)
        (R,) = coerce_coords(xp, R, device=device_of(R, self._corrections))
        lo, hi = R < _RMIN, R > 2.0 * self._rmax
        sp = eval_ppoly(xp, x, c, xp.where(lo | hi, self._rmax, R))
        out = [
            xp.where(
                lo, lc0[ii] + dsmall[ii] * (R - _RMIN), xp.where(hi, 0.0, sp[..., ii])
            )
            for ii in range(2)
        ]
        out = xp.stack(out, axis=0)
        return out if log else xp.exp(out)

    def _calc_corrections_backend(self):
        """_calc_corrections on the backend: niter iterations from scratch"""
        params = (*getattr(self._surfaceSigmaProfile, "_params", ()), self._beta)
        xp = prefer_backend_namespace(*params)
        rs = asarray_on_device(xp, self._rs, device_of(*params))
        corrections = None
        for ii in range(self._niter):
            kw = (
                {}
                if corrections is None
                else dict(
                    corrections=corrections,
                    npoints=self._npoints,
                    rmax=self._rmax,
                    interp_k=self._interp_k,
                )
            )
            currentDF = self._dftype(
                surfaceSigma=self._surfaceSigmaProfile, beta=self._beta, **kw
            )
            thisSurface = currentDF.surfacemass(rs, use_physical=False)
            new = xp.stack(
                [
                    currentDF._ssp("surfacemass", rs) / thisSurface,
                    currentDF._ssp("sigma2", rs)
                    * thisSurface
                    / currentDF.sigma2surfacemass(rs, use_physical=False),
                ],
                axis=1,
            )
            corrections = new if corrections is None else corrections * new
        return corrections

    def _createSavefilename(self, niter):
        # Form surfaceSigmaProfile string
        sspFormat = self._surfaceSigmaProfile.formatStringParams()
        sspString = ""
        for format in sspFormat:
            sspString += format + "_"
        return os.path.join(
            self._savedir,
            "dfcorrection_"
            + self._dftype.__name__
            + "_"
            + self._surfaceSigmaProfile.__class__.__name__
            + "_"
            + sspString % self._surfaceSigmaProfile.outputParams()
            + "%6.4f_%i_%6.4f_%i.sav" % (self._beta, self._npoints, self._rmax, niter),
        )

    def correct(self, R, log=False):
        """
        Calculate the correction in Sigma and sigma2 at R.

        Parameters
        ----------
        R : float
            Galactocentric radius (/ro).
        log : bool, optional
            If True, return the log of the correction.

        Returns
        -------
        tuple
            (Sigma correction, sigma2 correction).

        Notes
        -----
        - 2010-03-10 - Written - Bovy (NYU)
        """
        if is_backend_array(R) or is_backend_array(self._corrections):
            return self._correct_backend(R, log)
        if isinstance(R, numpy.ndarray):
            out = numpy.empty((2, len(R)))
            # R < _RMIN
            rmin_indx = R < _RMIN
            if numpy.sum(rmin_indx) > 0:
                out[0, rmin_indx] = numpylog(
                    self._corrections[0, 0]
                ) + self._surfaceDerivSmallR * (R[rmin_indx] - _RMIN)
                out[1, rmin_indx] = numpylog(
                    self._corrections[0, 1]
                ) + self._sigma2DerivSmallR * (R[rmin_indx] - _RMIN)
            # R > 2rmax
            rmax_indx = R > (2.0 * self._rmax)
            if numpy.sum(rmax_indx) > 0:
                out[:, rmax_indx] = 0.0
            #'normal' R
            r_indx = (R >= _RMIN) * (R <= (2.0 * self._rmax))
            if numpy.sum(r_indx) > 0:
                out[0, r_indx] = self._surfaceInterpolate(R[r_indx])
                out[1, r_indx] = self._sigma2Interpolate(R[r_indx])
            if log:
                return out
            else:
                return numpy.exp(out)
        if R < _RMIN:
            out = numpy.array(
                [
                    numpylog(self._corrections[0, 0])
                    + self._surfaceDerivSmallR * (R - _RMIN),
                    numpylog(self._corrections[0, 1])
                    + self._sigma2DerivSmallR * (R - _RMIN),
                ]
            )
        elif R > (2.0 * self._rmax):
            out = numpy.array([0.0, 0.0])
        else:
            if _SCIPY_VERSION >= _SCIPY_VERSION_BREAK:
                out = numpy.array(
                    [self._surfaceInterpolate(R), self._sigma2Interpolate(R)]
                )
            else:  # pragma: no cover
                out = numpy.array(
                    [self._surfaceInterpolate(R)[0], self._sigma2Interpolate(R)[0]]
                )
        if log:
            return out
        else:
            return numpy.exp(out)

    def derivLogcorrect(self, R):
        """
        Calculate the derivative of the log of the correction in Sigma and sigma2 at R.

        Parameters
        ----------
        R : float
            Galactocentric radius(/ro)

        Returns
        -------
        numpy.ndarray
            [d log(Sigma correction)/dR, d log(sigma2 correction)/dR]

        Notes
        -----
        - 2010-03-10 - Written - Bovy (NYU)
        """
        if R < _RMIN:
            out = numpy.array([self._surfaceDerivSmallR, self._sigma2DerivSmallR])
        elif R > (2.0 * self._rmax):
            out = numpy.array([0.0, 0.0])
        else:
            if _SCIPY_VERSION >= _SCIPY_VERSION_BREAK:
                out = numpy.array(
                    [
                        self._surfaceInterpolate(R, nu=1),
                        self._sigma2Interpolate(R, nu=1),
                    ]
                )
            else:  # pragma: no cover
                out = numpy.array(
                    [
                        self._surfaceInterpolate(R, nu=1)[0],
                        self._sigma2Interpolate(R, nu=1)[0],
                    ]
                )
        return out

    @_on_numpy
    def _calc_corrections(self):
        """Internal function that calculates the corrections"""
        searchIter = self._niter - 1
        while searchIter > 0:
            trySavefilename = self._createSavefilename(searchIter)
            if os.path.exists(trySavefilename):
                trySavefile = open(trySavefilename, "rb")
                corrections = numpy.array(pickle.load(trySavefile))
                trySavefile.close()
                break
            else:
                searchIter -= 1
        if searchIter == 0:
            corrections = numpy.ones((self._npoints, 2))
        for ii in range(searchIter, self._niter):
            if ii == 0:
                currentDF = self._dftype(
                    surfaceSigma=self._surfaceSigmaProfile, beta=self._beta
                )
            else:
                currentDF = self._dftype(
                    surfaceSigma=self._surfaceSigmaProfile,
                    beta=self._beta,
                    corrections=corrections,
                    npoints=self._npoints,
                    rmax=self._rmax,
                    savedir=self._savedir,
                    interp_k=self._interp_k,
                )
            newcorrections = numpy.zeros((self._npoints, 2))
            for jj in range(self._npoints):
                thisSurface = currentDF.surfacemass(self._rs[jj], use_physical=False)
                newcorrections[jj, 0] = (
                    currentDF.targetSurfacemass(self._rs[jj], use_physical=False)
                    / thisSurface
                )
                newcorrections[jj, 1] = (
                    currentDF.targetSigma2(self._rs[jj], use_physical=False)
                    * thisSurface
                    / currentDF.sigma2surfacemass(self._rs[jj], use_physical=False)
                )
                # print(jj, newcorrections[jj,:])
            corrections *= newcorrections
        # Save
        picklethis = []
        for arr in list(corrections):
            picklethis.append([float(a) for a in arr])
        save_pickles(
            self._savefilename, picklethis
        )  # We pickle a list for platform-independence)
        return corrections


class DFcorrectionError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def vRvTRToEL(vR, vT, R, beta, dftype="dehnen"):
    """
    Calculate the energy and angular momentum.

    Parameters
    ----------
    vR : float
        Radial velocity.
    vT : float
        Rotational velocity.
    R : float
        Galactocentric radius.
    beta : float
        Parameter that determines the shape of the rotation curve.
    dftype : str, optional
        Type of disk distribution function. Default is "dehnen".

    Returns
    -------
    tuple
        Energy and angular momentum.

    Notes
    -----
    - 2010-03-10 - Written - Bovy (NYU)

    """
    if dftype == "schwarzschild":
        # Compute E in the epicycle approximation
        gamma = _sqrt(2.0 / (1.0 + beta))
        L = R * vT
        xL = _beta_branch(beta, lambda: L, lambda b: L ** (1.0 / (b + 1.0)))
        return (
            0.5 * vR**2.0
            + 0.5 * gamma**2.0 * (vT - R**beta) ** 2.0
            + xL ** (2.0 * beta) / 2.0
            + axipotential(xL, beta=beta),
            L,
        )
    else:
        return (axipotential(R, beta) + 0.5 * vR**2.0 + 0.5 * vT**2.0, vT * R)


def axipotential(R, beta=0.0):
    """
    Return the axisymmetric potential at R/Ro.

    Parameters
    ----------
    R : float
        Galactocentric radius.
    beta : float, optional
        Rotation curve power-law.

    Returns
    -------
    float
        Pot(R)/vo**2.

    Notes
    -----
    - 2010-03-01 - Written - Bovy (NYU)

    """
    return _beta_branch(
        beta, lambda: _logpotential(R), lambda b: R ** (2.0 * b) / 2.0 / b
    )


def _logpotential(R):
    if is_backend_array(R):
        xp = get_namespace(R)
        # scimath.log real part is log|R|; floor R==0 at _RMIN (guards AD)
        return xp.log(xp.where(R == 0.0, _RMIN, xp.abs(R)))
    if numpy.any(R == 0.0):
        out = numpy.empty(R.shape)
        out[R == 0.0] = numpylog(_RMIN)
        out[R != 0.0] = numpylog(R[R != 0.0])
        return out
    else:
        return numpylog(R)


def _kappa(R, beta):
    """Internal function to give kappa(r)"""
    return numpy.sqrt(2.0 * (1.0 + beta)) * R ** (beta - 1)


def _dlToRphi(d, l):
    """Convert d and l to R and phi, l is in radians"""
    if (
        is_backend_array(d)
        or is_backend_array(l)
        or (isinstance(d, numpy.ndarray) and d.ndim > 0)  # the samplers' arrays
    ):
        xp = prefer_backend_namespace(d, l)
        d, l = coerce_coords(xp, d, l, device=device_of(d, l))
        R2 = 1.0 + d**2.0 - 2.0 * d * xp.cos(l)
        at0 = R2 == 0.0  # numpy's nudge; a guarded sqrt keeps d/dd finite there
        R = xp.where(at0, 0.0001, xp.sqrt(xp.where(at0, 1.0, R2)))
        d = xp.where(at0, d + 0.0001, d)
        theta = xp.arcsin(d / R * xp.sin(l))
        far = (1.0 / xp.cos(l) < d) & (xp.cos(l) > 0.0)
        return (R, xp.where(far, numpy.pi - theta, theta))
    R = numpy.sqrt(1.0 + d**2.0 - 2.0 * d * numpy.cos(l))
    if R == 0.0:
        R += 0.0001
        d += 0.0001
    if 1.0 / numpy.cos(l) < d and numpy.cos(l) > 0.0:
        theta = numpy.pi - numpy.arcsin(d / R * numpy.sin(l))
    else:
        theta = numpy.arcsin(d / R * numpy.sin(l))
    return (R, theta)


def _vtmaxEq(vT, R, diskdf):
    """Equation to solve to find the max vT at R"""
    # Calculate a bunch of stuff that we need
    if diskdf._beta == 0.0:
        E = vT**2.0 / 2.0 + numpylog(R)
        xE = numpy.exp(E - 0.5)
        OE = xE**-1.0
        LCE = xE
        dxEdvT = xE * vT
    else:  # non-flat rotation curve
        E = vT**2.0 / 2.0 + 1.0 / 2.0 / diskdf._beta * R ** (2.0 * diskdf._beta)
        xE = (2.0 * E / (1.0 + 1.0 / diskdf._beta)) ** (1.0 / 2.0 / diskdf._beta)
        OE = xE ** (diskdf._beta - 1.0)
        LCE = xE ** (diskdf._beta + 1.0)
        dxEdvT = xE / 2.0 / diskdf._beta / E * vT
    L = R * vT
    sigma2xE = diskdf._ssp("sigma2", xE, log=False)
    return (
        OE * R / sigma2xE
        + (
            diskdf._ssp("surfacemassDerivative", xE, log=True)
            - (1.0 + OE * (L - LCE) / sigma2xE)
            * diskdf._ssp("sigma2Derivative", xE, log=True)
            + (L - LCE) / sigma2xE * (diskdf._beta - 1.0) * xE ** (diskdf._beta - 2.0)
            - OE * (diskdf._beta + 1.0) / sigma2xE * xE**diskdf._beta
        )
        * dxEdvT
    )


def _marginalizeVperpIntegrandSinAlphaLarge(
    vR, df, R, sinalpha, cotalpha, vlos, vcirc, sigma
):
    return df(
        *vRvTRToEL(
            vR * sigma,
            cotalpha * vR * sigma + vlos / sinalpha + vcirc,
            R,
            df._beta,
            df._dftype,
        )
    )


def _marginalizeVperpIntegrandSinAlphaSmall(
    vT, df, R, cosalpha, tanalpha, vlos, vcirc, sigma
):
    return df(
        *vRvTRToEL(
            tanalpha * vT * sigma - vlos / cosalpha,
            vT * sigma + vcirc,
            R,
            df._beta,
            df._dftype,
        )
    )
