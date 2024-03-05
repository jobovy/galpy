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
import copy
import os
import os.path
import pickle

import numpy
import scipy

numpylog = numpy.lib.scimath.log  # somehow, this code produces log(negative), which scipy (now numpy.lib.scimath.log) implements as log(|negative|) + i pi while numpy gives NaN and we want the scipy behavior; not sure where the log(negative) comes from though! I think it's for sigma=0 DFs (this test fails with numpy.log) where the DF eval has a log(~zero) that can be slightly negative because of numerical precision issues
from scipy import integrate, interpolate, optimize, stats

from ..actionAngle import actionAngleAdiabatic
from ..orbit import Orbit
from ..potential import PowerSphericalPotential
from ..util import conversion, quadpack, save_pickles
from ..util.ars import ars
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


class diskdf(df):
    """Class that represents a disk DF"""

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
        self._gamma = numpy.sqrt(2.0 / (1.0 + self._beta))
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
        self._psp = PowerSphericalPotential(
            normalize=1.0, alpha=2.0 - 2.0 * self._beta
        ).toPlanar()
        # Setup aA objects for frequency and rap,rperi calculation
        self._aA = actionAngleAdiabatic(pot=self._psp, gamma=0.0)
        return None

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
                    return numpy.real(
                        self.eval(
                            *vRvTRToEL(
                                args[0].vR(use_physical=False),
                                args[0].vT(use_physical=False),
                                args[0].R(use_physical=False),
                                self._beta,
                                self._dftype,
                            )
                        )
                    )
            else:
                no = args[0](args[1])
                return numpy.real(
                    self.eval(
                        *vRvTRToEL(
                            no.vR(use_physical=False),
                            no.vT(use_physical=False),
                            no.R(use_physical=False),
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
            vR = numpy.array([o.vR(use_physical=False) for o in args[0]])
            vT = numpy.array([o.vT(use_physical=False) for o in args[0]])
            R = numpy.array([o.R(use_physical=False) for o in args[0]])
            return numpy.real(
                self.eval(*vRvTRToEL(vR, vT, R, self._beta, self._dftype))
            )
        elif isinstance(args[0], numpy.ndarray) and not (
            hasattr(args[0], "isscalar") and args[0].isscalar
        ):
            # Grab all of the vR, vT, and R
            vR = args[0][1]
            vT = args[0][2]
            R = args[0][0]
            return numpy.real(
                self.eval(*vRvTRToEL(vR, vT, R, self._beta, self._dftype))
            )
        else:
            return numpy.real(self.eval(*args))

    def _call_marginalizevperp(self, o, **kwargs):
        """Call the DF, marginalizing over perpendicular velocity"""
        # Get l, vlos
        l = o.ll(obs=[1.0, 0.0, 0.0], ro=1.0) * _DEGTORAD
        vlos = o.vlos(ro=1.0, vo=1.0, obs=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        R = o.R(use_physical=False)
        phi = o.phi(use_physical=False)
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
        sigmaR2 = self.targetSigma2(R, use_physical=False)
        sigmaR1 = numpy.sqrt(sigmaR2)
        # Use the asymmetric drift equation to estimate va
        va = (
            sigmaR2
            / 2.0
            / R**self._beta
            * (
                1.0 / self._gamma**2.0
                - 1.0
                - R * self._surfaceSigmaProfile.surfacemassDerivative(R, log=True)
                - R * self._surfaceSigmaProfile.sigma2Derivative(R, log=True)
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

    def _call_marginalizevlos(self, o, **kwargs):
        """Call the DF, marginalizing over line-of-sight velocity"""
        # Get d, l, vperp
        l = o.ll(obs=[1.0, 0.0, 0.0], ro=1.0) * _DEGTORAD
        vperp = o.vll(ro=1.0, vo=1.0, obs=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        R = o.R(use_physical=False)
        phi = o.phi(use_physical=False)
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
        sigmaR2 = self.targetSigma2(R, use_physical=False)
        sigmaR1 = numpy.sqrt(sigmaR2)
        # Use the asymmetric drift equation to estimate va
        va = (
            sigmaR2
            / 2.0
            / R**self._beta
            * (
                1.0 / self._gamma**2.0
                - 1.0
                - R * self._surfaceSigmaProfile.surfacemassDerivative(R, log=True)
                - R * self._surfaceSigmaProfile.sigma2Derivative(R, log=True)
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
        return self._surfaceSigmaProfile.sigma2(R, log=log)

    @potential_physical_input
    @physical_conversion("surfacedensity", pop=True)
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
        return self._surfaceSigmaProfile.surfacemass(R, log=log)

    @physical_conversion("surfacedensitydistance", pop=True)
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
            return self._surfaceSigmaProfile.surfacemass(R, log=log) + numpylog(d)
        else:
            return self._surfaceSigmaProfile.surfacemass(R, log=log) * d

    @physical_conversion("surfacedensitydistance", pop=True)
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
                return self.targetSurfacemass(R, use_physical=False) * d
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

    @physical_conversion("position", pop=True)
    def sampledSurfacemassLOS(self, l, n=1, maxd=None, target=True):
        """
        Sample a distance along the line of sight

        Parameters
        ----------
        l : float or Quantity
            Galactic longitude.
        n : int, optional
            Number of distances to sample.
        maxd : float or Quantity, optional
            Maximum distance to consider (for the rejection sampling).
        target : bool, optional
            If True, sample from the 'target' surface mass density, rather than the actual surface mass density (default=True).

        Returns
        -------
        list
            List of samples.

        Notes
        -----
        - 2011-03-24 - Written - Bovy (NYU)

        """
        # First calculate where the maximum is
        if target:
            minR = optimize.fmin_bfgs(
                lambda x: -self.targetSurfacemassLOS(
                    x, l, use_physical=False, deg=False
                ),
                0.0,
                disp=False,
            )[0]
            maxSM = self.targetSurfacemassLOS(minR, l, deg=False, use_physical=False)
        else:
            minR = optimize.fmin_bfgs(
                lambda x: -self.surfacemassLOS(x, l, deg=False, use_physical=False),
                0.0,
                disp=False,
            )[0]
            maxSM = self.surfacemassLOS(minR, l, deg=False, use_physical=False)
        # Now rejection-sample
        l = conversion.parse_angle(l)
        maxd = conversion.parse_length(maxd, ro=self._ro)
        if maxd is None:
            maxd = _MAXD_REJECTLOS
        out = []
        while len(out) < n:
            # sample
            prop = numpy.random.random() * maxd
            if target:
                surfmassatprop = self.targetSurfacemassLOS(
                    prop, l, deg=False, use_physical=False
                )
            else:
                surfmassatprop = self.surfacemassLOS(
                    prop, l, deg=False, use_physical=False
                )
            if surfmassatprop / maxSM > numpy.random.random():  # accept
                out.append(prop)
        return numpy.array(out)

    @potential_physical_input
    @physical_conversion("velocity", pop=True)
    def sampleVRVT(self, R, n=1, nsigma=None, target=True):
        """
        Sample a radial and azimuthal velocity at R

        Parameters
        ----------
        R : float or Quantity
            Galactocentric distance.
        n : int, optional
            Number of distances to sample.
        nsigma : float, optional
            Number of sigma to rejection-sample on.
        target : bool, optional
            If True, sample using the 'target' sigma_R rather than the actual sigma_R (default=True).

        Returns
        -------
        list
            List of samples.

        Notes
        -----
        - 2011-03-24 - Written - Bovy (NYU)
        """
        # Determine where the max of the v-distribution is using asymmetric drift
        maxVR = 0.0
        maxVT = optimize.brentq(_vtmaxEq, 0.0, R**self._beta + 0.2, (R, self))
        maxVD = self(Orbit([R, maxVR, maxVT]))
        # Now rejection-sample
        if nsigma == None:
            nsigma = _NSIGMA
        out = []
        if target:
            sigma = numpy.sqrt(self.targetSigma2(R, use_physical=False))
        else:
            sigma = numpy.sqrt(self.sigma2(R, use_physical=False))
        while len(out) < n:
            # sample
            vrg, vtg = numpy.random.normal(), numpy.random.normal()
            propvR = vrg * nsigma * sigma
            propvT = vtg * nsigma * sigma / self._gamma + maxVT
            VDatprop = self(Orbit([R, propvR, propvT]))
            if VDatprop / maxVD > numpy.random.uniform() * numpy.exp(
                -0.5 * (vrg**2.0 + vtg**2.0)
            ):  # accept
                out.append(numpy.array([propvR, propvT]))
        return numpy.array(out)

    def sampleLOS(
        self,
        los,
        n=1,
        deg=True,
        maxd=None,
        nsigma=None,
        targetSurfmass=True,
        targetSigma2=True,
    ):
        """
        Sample along a given LOS

        Parameters
        ----------
        los : float or Quantity
            Line of sight Galactic longitude.
        n : int, optional
            Number of distances to sample.
        deg : bool, optional
            If False, los is in radians.
        maxd : float or Quantity, optional
            Maximum distance to consider (for the rejection sampling).
        nsigma : float, optional
            Number of sigma to integrate the velocities over.
        targetSurfmass : bool, optional
            If True, use target surface mass (default=True).
        targetSigma2 : bool, optional
            If True, use target sigma_R^2 (default=True).

        Returns
        -------
        list
            List of Orbits sampled.

        Notes
        -----
        - target=False uses target distribution for derivatives (this is a detail)
        - 2011-03-24 - Written - Bovy (NYU)
        """
        if _APY_LOADED and isinstance(los, units.Quantity):
            l = conversion.parse_angle(los)
        elif deg:
            l = los * _DEGTORAD
        else:
            l = los
        out = []
        # sample distances
        ds = self.sampledSurfacemassLOS(
            l, n=n, maxd=maxd, target=targetSurfmass, use_physical=False
        )
        for ii in range(int(n)):
            # Calculate R and phi
            thisR, thisphi = _dlToRphi(ds[ii], l)
            # sample velocities
            vv = self.sampleVRVT(
                thisR, n=1, nsigma=nsigma, target=targetSigma2, use_physical=False
            )[0]
            if self._roSet and self._voSet:
                out.append(
                    Orbit([thisR, vv[0], vv[1], thisphi], ro=self._ro, vo=self._vo)
                )
            else:
                out.append(Orbit([thisR, vv[0], vv[1], thisphi]))
        return out

    @potential_physical_input
    @physical_conversion("velocity", pop=True)
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
        sigmaR2 = self.targetSigma2(R, use_physical=False)
        return (
            sigmaR2
            / 2.0
            / R**self._beta
            * (
                1.0 / self._gamma**2.0
                - 1.0
                - R * self._surfaceSigmaProfile.surfacemassDerivative(R, log=True)
                - R * self._surfaceSigmaProfile.sigma2Derivative(R, log=True)
            )
        )

    @potential_physical_input
    @physical_conversion("surfacedensity", pop=True)
    def surfacemass(self, R, romberg=False, nsigma=None, relative=False):
        """
        Calculate the surface-mass at R by marginalizing over velocity

        Parameters
        ----------
        R : float or Quantity
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
        logSigmaR = self.targetSurfacemass(R, log=True, use_physical=False)
        sigmaR2 = self.targetSigma2(R, use_physical=False)
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
                - R * self._surfaceSigmaProfile.surfacemassDerivative(R, log=True)
                - R * self._surfaceSigmaProfile.sigma2Derivative(R, log=True)
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
    def sigma2surfacemass(self, R, romberg=False, nsigma=None, relative=False):
        """
        Calculate the product sigma_R^2 x surface-mass at R by marginalizing over velocity.

        Parameters
        ----------
        R : float or Quantity
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
        logSigmaR = self.targetSurfacemass(R, log=True, use_physical=False)
        sigmaR2 = self.targetSigma2(R, use_physical=False)
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
                - R * self._surfaceSigmaProfile.surfacemassDerivative(R, log=True)
                - R * self._surfaceSigmaProfile.sigma2Derivative(R, log=True)
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
        R: float or Quantity
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
                return units.Quantity(out * fac, unit=u)
            else:
                return out * fac
        else:
            return self._vmomentsurfacemass(*args, **kwargs)

    def _vmomentsurfacemass(
        self, R, n, m, romberg=False, nsigma=None, relative=False, phi=0.0, deriv=None
    ):
        """Non-physical version of vmomentsurfacemass, otherwise the same"""
        # odd moments of vR are zero
        if isinstance(n, int) and n % 2 == 1:
            return 0.0
        if nsigma == None:
            nsigma = _NSIGMA
        logSigmaR = self.targetSurfacemass(R, log=True, use_physical=False)
        sigmaR2 = self.targetSigma2(R, use_physical=False)
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
                - R * self._surfaceSigmaProfile.surfacemassDerivative(R, log=True)
                - R * self._surfaceSigmaProfile.sigma2Derivative(R, log=True)
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
            numpy.sqrt(2.0 * (E - self._psp(xE)) - L**2.0 / xE**2.0),
            L / xE,
            0.0,
            0.0,
        )
        return (
            self._aA._aAS.actionsFreqs(xE, 0.0, L / xE, 0.0, 0.0)[3][0],
            rap[0],
            rperi[0],
        )

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
        target=True,
    ):
        """
        Sample n*nphi points from this disk DF.

        Parameters
        ----------
        n : int, optional
            Number of desired samples. Default is 1.
        rrange : list, optional
            If you only want samples in this rrange, set this keyword (only works when asking for an (RZ)Orbit).
        returnROrbit : bool, optional
            If True, return a planarROrbit instance: [R,vR,vT] (default).
        returnOrbit : bool, optional
            If True, return a planarOrbit instance (including phi).
        nphi : float, optional
            Number of azimuths to sample for each E,L.
        los : float, optional
            Line of sight sampling along this line of sight.
        losdeg : bool, optional
            If True, los is in degrees (default).
        nsigma : int, optional
            Number of sigma to rejection-sample on.
        maxd : float, optional
            Maximum distance to consider (for the rejection sampling).
        target : bool, optional
            If True, use target surface mass and sigma2 profiles (default).

        Returns
        -------
        list
            n*nphi list of [[E,Lz],...] or list of planar(R)Orbits.
            CAUTION: lists of EL need to be post-processed to account for the
                    \\kappa/\\omega_R discrepancy

        Notes
        -----
        - 2010-07-10 - Started  - Bovy (NYU)

        """
        raise NotImplementedError("'sample' method for this disk df is not implemented")

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
        return R**self._beta - self.asymmetricdrift(R, use_physical=False)

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
        return self.targetSigma2(R, log=log, use_physical=False)

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
            return self.targetSigma2(R, log=log, use_physical=False) - 2.0 * numpylog(
                self._gamma
            )
        else:
            return self.targetSigma2(R, log=log, use_physical=False) / self._gamma**2.0


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
        SRE2 = self.targetSigma2(xE, log=True, use_physical=False) + correction[1]
        if _PROFILE:  # pragma: no cover
            targSigma_time = time.time() - start
            start = time.time()
            out = (
                self._gamma
                * numpy.exp(
                    logsigmaR2
                    - SRE2
                    + self.targetSurfacemass(xE, log=True, use_physical=False)
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
                    + self.targetSurfacemass(xE, log=True, use_physical=False)
                    - logSigmaR
                    + numpy.exp(logOLLE - SRE2)
                    + correction[0]
                )
                / 2.0
                / numpy.pi
            )

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
        targetSurfmass=True,
        targetSigma2=True,
        maxd=None,
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
        nsigma : int, optional
            Number of sigma to rejection-sample on. Default is None.
        targetSurfmass : bool, optional
            If True, use target surface mass profile. Default is True.
        targetSigma2 : bool, optional
            If True, use target sigma2 profile. Default is True.
        maxd : float or Quantity, optional
            Maximum distance to consider (for the rejection sampling). Default is None.
        **kwargs : dict, optional
            Additional keyword arguments.

        Returns
        -------
        out : list
            n*nphi list of [[E,Lz],...] or list of planar(R)Orbits.
            CAUTION: lists of EL need to be post-processed to account for the
            \\kappa/\\omega_R discrepancy; EL not returned in physical units.

        Notes
        -----
        - 2010-07-10 - Started  - Bovy (NYU)
        """
        if not los is None:
            return self.sampleLOS(
                los,
                deg=losdeg,
                n=n,
                maxd=maxd,
                nsigma=nsigma,
                targetSurfmass=targetSurfmass,
                targetSigma2=targetSigma2,
            )
        # First sample xE
        if self._correct:
            xE = numpy.array(
                ars(
                    [0.0, 0.0],
                    [True, False],
                    [0.05, 2.0],
                    _ars_hx,
                    _ars_hpx,
                    nsamples=n,
                    hxparams=(self._surfaceSigmaProfile, self._corr),
                )
            )
        else:
            xE = numpy.array(
                ars(
                    [0.0, 0.0],
                    [True, False],
                    [0.05, 2.0],
                    _ars_hx,
                    _ars_hpx,
                    nsamples=n,
                    hxparams=(self._surfaceSigmaProfile, None),
                )
            )
        # Calculate E
        if self._beta == 0.0:
            E = numpylog(xE) + 0.5
        else:  # non-flat rotation curve
            E = 0.5 * xE ** (2.0 * self._beta) * (1.0 + 1.0 / self._beta)
        # Then sample Lz
        LCE = xE ** (self._beta + 1.0)
        OR = xE ** (self._beta - 1.0)
        Lz = (
            self._surfaceSigmaProfile.sigma2(xE)
            * numpylog(stats.uniform.rvs(size=n))
            / OR
        )
        if self._correct:
            Lz *= self._corr.correct(xE, log=False)[1, :]
        Lz += LCE
        if not returnROrbit and not returnOrbit:
            out = [[e, l] for e, l in zip(E, Lz)]
        else:
            if not rrange is None:
                rrange[0] = conversion.parse_length(rrange[0], ro=self._ro)
                rrange[1] = conversion.parse_length(rrange[1], ro=self._ro)
            out = []
            for ii in range(int(n)):
                try:
                    wR, rap, rperi = self._ELtowRRapRperi(E[ii], Lz[ii])
                except ValueError:  # pragma: no cover
                    # Tests don't get here anymore, because of improvements
                    # in the rperi/rap calculation, but leaving the try/except
                    # in because it can do no harm
                    continue
                TR = 2.0 * numpy.pi / wR
                tr = stats.uniform.rvs() * TR
                if tr > TR / 2.0:
                    tr -= TR / 2.0
                    thisOrbit = Orbit([rperi, 0.0, Lz[ii] / rperi])
                else:
                    thisOrbit = Orbit([rap, 0.0, Lz[ii] / rap])
                thisOrbit.integrate(numpy.array([0.0, tr]), self._psp)
                if returnOrbit:
                    vxvv = thisOrbit(tr).vxvv[0]
                    thisOrbit = Orbit(
                        vxvv=numpy.array(
                            [
                                vxvv[0],
                                vxvv[1],
                                vxvv[2],
                                stats.uniform.rvs() * numpy.pi * 2.0,
                            ]
                        ).reshape(4)
                    )
                else:
                    thisOrbit = thisOrbit(tr)
                kappa = _kappa(thisOrbit.vxvv[0, 0], self._beta)
                if not rrange == None:
                    if (
                        thisOrbit.vxvv[0, 0] < rrange[0]
                        or thisOrbit.vxvv[0, 0] > rrange[1]
                    ):
                        continue
                mult = numpy.ceil(kappa / wR * nphi) - 1.0
                kappawR = kappa / wR * nphi - mult
                while mult > 0:
                    if returnOrbit:
                        out.append(
                            Orbit(
                                vxvv=numpy.array(
                                    [
                                        vxvv[0],
                                        vxvv[1],
                                        vxvv[2],
                                        stats.uniform.rvs() * numpy.pi * 2.0,
                                    ]
                                ).reshape(4)
                            )
                        )
                    else:
                        out.append(thisOrbit)
                    mult -= 1
                if stats.uniform.rvs() > kappawR:
                    continue
                out.append(thisOrbit)
        # Recurse to get enough
        if len(out) < n * nphi:
            out.extend(
                self.sample(
                    n=int(n - len(out) / nphi),
                    rrange=rrange,
                    returnROrbit=returnROrbit,
                    returnOrbit=returnOrbit,
                    nphi=int(nphi),
                    los=los,
                    losdeg=losdeg,
                )
            )
        # Trim to make sure output has the right size
        out = out[0 : int(n * nphi)]
        if kwargs.get("use_physical", True) and self._roSet and self._voSet:
            if isinstance(out[0], Orbit):
                dumb = [o.turn_physical_on(ro=self._ro, vo=self._vo) for o in out]
        return out

    def _dlnfdR(self, R, vR, vT):
        # Calculate a bunch of stuff that we need
        if self._beta == 0.0:
            E = vR**2.0 / 2.0 + vT**2.0 / 2.0 + numpylog(R)
            xE = numpy.exp(E - 0.5)
            OE = xE**-1.0
            LCE = xE
            dRedR = xE / R
        else:  # non-flat rotation curve
            E = (
                vR**2.0 / 2.0
                + vT**2.0 / 2.0
                + 1.0 / 2.0 / self._beta * R ** (2.0 * self._beta)
            )
            xE = (2.0 * E / (1.0 + 1.0 / self._beta)) ** (1.0 / 2.0 / self._beta)
            OE = xE ** (self._beta - 1.0)
            LCE = xE ** (self._beta + 1.0)
            dRedR = xE / 2.0 / self._beta / E * R ** (2.0 * self._beta - 1.0)
        return (
            self._dlnfdRe(R, vR, vT, E=E, xE=xE, OE=OE, LCE=LCE) * dRedR
            + self._dlnfdl(R, vR, vT, E=E, xE=xE, OE=OE) * vT
        )

    def _dlnfdvR(self, R, vR, vT):
        # Calculate a bunch of stuff that we need
        if self._beta == 0.0:
            E = vR**2.0 / 2.0 + vT**2.0 / 2.0 + numpylog(R)
            xE = numpy.exp(E - 0.5)
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
        if self._beta == 0.0:
            E = vR**2.0 / 2.0 + vT**2.0 / 2.0 + numpylog(R)
            xE = numpy.exp(E - 0.5)
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
        if E is None or xE is None or OE is None or LCE is None:
            if self._beta == 0.0:
                E = vR**2.0 / 2.0 + vT**2.0 / 2.0 + numpylog(R)
                xE = numpy.exp(E - 0.5)
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
        sigma2xE = self._surfaceSigmaProfile.sigma2(xE, log=False)
        return (
            self._surfaceSigmaProfile.surfacemassDerivative(xE, log=True)
            - (1.0 + OE * (L - LCE) / sigma2xE)
            * self._surfaceSigmaProfile.sigma2Derivative(xE, log=True)
            + (L - LCE) / sigma2xE * (self._beta - 1.0) * xE ** (self._beta - 2.0)
            - OE * (self._beta + 1.0) / sigma2xE * xE**self._beta
        )

    def _dlnfdl(self, R, vR, vT, E=None, xE=None, OE=None):
        # Calculate a bunch of stuff that we need
        if E is None or xE is None or OE is None:
            if self._beta == 0.0:
                E = vR**2.0 / 2.0 + vT**2.0 / 2.0 + numpylog(R)
                xE = numpy.exp(E - 0.5)
                OE = xE**-1.0
            else:  # non-flat rotation curve
                E = (
                    vR**2.0 / 2.0
                    + vT**2.0 / 2.0
                    + 1.0 / 2.0 / self._beta * R ** (2.0 * self._beta)
                )
                xE = (2.0 * E / (1.0 + 1.0 / self._beta)) ** (1.0 / 2.0 / self._beta)
                OE = xE ** (self._beta - 1.0)
        sigma2xE = self._surfaceSigmaProfile.sigma2(xE, log=False)
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
        SRE2 = self.targetSigma2(xL, log=True, use_physical=False) + correction[1]
        return (
            self._gamma
            * numpy.exp(
                logsigmaR2
                - SRE2
                + self.targetSurfacemass(xL, log=True, use_physical=False)
                - logSigmaR
                - numpy.exp(logECLE - SRE2)
                + correction[0]
            )
            / 2.0
            / numpy.pi
        )

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
        nsigma : int, optional
            Number of sigma to rejection-sample on. Default is None.
        targetSurfmass : bool, optional
            If True, use target surface mass profile. Default is True.
        targetSigma2 : bool, optional
            If True, use target sigma2 profile. Default is True.
        maxd : float or Quantity, optional
            Maximum distance to consider (for the rejection sampling). Default is None.
        **kwargs : dict, optional
            Additional keyword arguments.

        Returns
        -------
        out : list
            n*nphi list of [[E,Lz],...] or list of planar(R)Orbits.
            CAUTION: lists of EL need to be post-processed to account for the
            \\kappa/\\omega_R discrepancy; EL not returned in physical units.

        Notes
        -----
        - 2010-07-10 - Started  - Bovy (NYU)
        """
        if not los is None:
            return self.sampleLOS(
                los,
                n=n,
                maxd=maxd,
                nsigma=nsigma,
                targetSurfmass=targetSurfmass,
                targetSigma2=targetSigma2,
            )
        # First sample xL
        if self._correct:
            xL = numpy.array(
                ars(
                    [0.0, 0.0],
                    [True, False],
                    [0.05, 2.0],
                    _ars_hx,
                    _ars_hpx,
                    nsamples=n,
                    hxparams=(self._surfaceSigmaProfile, self._corr),
                )
            )
        else:
            xL = numpy.array(
                ars(
                    [0.0, 0.0],
                    [True, False],
                    [0.05, 2.0],
                    _ars_hx,
                    _ars_hpx,
                    nsamples=n,
                    hxparams=(self._surfaceSigmaProfile, None),
                )
            )
        # Calculate Lz
        Lz = xL ** (self._beta + 1.0)
        # Then sample E
        if self._beta == 0.0:
            ECL = numpylog(xL) + 0.5
        else:
            ECL = 0.5 * (1.0 / self._beta + 1.0) * xL ** (2.0 * self._beta)
        E = -self._surfaceSigmaProfile.sigma2(xL) * numpylog(stats.uniform.rvs(size=n))
        if self._correct:
            E *= self._corr.correct(xL, log=False)[1, :]
        E += ECL
        if not returnROrbit and not returnOrbit:
            out = [[e, l] for e, l in zip(E, Lz)]
        else:
            if not rrange is None:
                rrange[0] = conversion.parse_length(rrange[0], ro=self._ro)
                rrange[1] = conversion.parse_length(rrange[1], ro=self._ro)
            out = []
            for ii in range(n):
                try:
                    wR, rap, rperi = self._ELtowRRapRperi(E[ii], Lz[ii])
                except ValueError:  # pragma: no cover
                    continue
                TR = 2.0 * numpy.pi / wR
                tr = stats.uniform.rvs() * TR
                if tr > TR / 2.0:
                    tr -= TR / 2.0
                    thisOrbit = Orbit([rperi, 0.0, Lz[ii] / rperi])
                else:
                    thisOrbit = Orbit([rap, 0.0, Lz[ii] / rap])
                thisOrbit.integrate(numpy.array([0.0, tr]), self._psp)
                if returnOrbit:
                    vxvv = thisOrbit(tr).vxvv[0]
                    thisOrbit = Orbit(
                        vxvv=numpy.array(
                            [
                                vxvv[0],
                                vxvv[1],
                                vxvv[2],
                                stats.uniform.rvs() * numpy.pi * 2.0,
                            ]
                        ).reshape(4)
                    )
                else:
                    thisOrbit = thisOrbit(tr)
                kappa = _kappa(thisOrbit.vxvv[0, 0], self._beta)
                if not rrange == None:
                    if (
                        thisOrbit.vxvv[0, 0] < rrange[0]
                        or thisOrbit.vxvv[0, 0] > rrange[1]
                    ):
                        continue
                mult = numpy.ceil(kappa / wR * nphi) - 1.0
                kappawR = kappa / wR * nphi - mult
                while mult > 0:
                    if returnOrbit:
                        out.append(
                            Orbit(
                                vxvv=numpy.array(
                                    [
                                        vxvv[0],
                                        vxvv[1],
                                        vxvv[2],
                                        stats.uniform.rvs() * numpy.pi * 2.0,
                                    ]
                                ).reshape(4)
                            )
                        )
                    else:
                        out.append(thisOrbit)
                    mult -= 1
                if stats.uniform.rvs() > kappawR:
                    continue
                out.append(thisOrbit)
        # Recurse to get enough
        if len(out) < n * nphi:
            out.extend(
                self.sample(
                    n=int(n - len(out) / nphi),
                    rrange=rrange,
                    returnROrbit=returnROrbit,
                    returnOrbit=returnOrbit,
                    nphi=nphi,
                )
            )
        # Trim to make sure output has the right size
        out = out[0 : int(n * nphi)]
        if kwargs.get("use_physical", True) and self._roSet and self._voSet:
            if isinstance(out[0], Orbit):
                dumb = [o.turn_physical_on(ro=self._ro, vo=self._vo) for o in out]
        return out

    def _dlnfdR(self, R, vR, vT):
        # Calculate a bunch of stuff that we need
        E, L = vRvTRToEL(vR, vT, R, self._beta, self._dftype)
        if self._beta == 0.0:
            xL = L
            dRldR = vT
            ECL = numpylog(xL) + 0.5
            dECLEdR = 0.0
        else:  # non-flat rotation curve
            xL = L ** (1.0 / (self._beta + 1.0))
            dRldR = L ** (1.0 / (self._beta + 1.0)) / R / (self._beta + 1.0)
            ECL = 0.5 * (1.0 / self._beta + 1.0) * xL ** (2.0 * self._beta)
            dECLdRl = (1.0 + self._beta) * xL ** (2.0 * self._beta - 1)
            dEdR = R ** (2.0 * self._beta - 1.0)
            dECLEdR = dECLdRl * dRldR - dEdR
        sigma2xL = self._surfaceSigmaProfile.sigma2(xL, log=False)
        return (
            self._surfaceSigmaProfile.surfacemassDerivative(xL, log=True)
            - (1.0 + (ECL - E) / sigma2xL)
            * self._surfaceSigmaProfile.sigma2Derivative(xL, log=True)
        ) * dRldR + dECLEdR / sigma2xL

    def _dlnfdvR(self, R, vR, vT):
        # Calculate a bunch of stuff that we need
        E, L = vRvTRToEL(vR, vT, R, self._beta, self._dftype)
        if self._beta == 0.0:
            xL = L
        else:  # non-flat rotation curve
            xL = L ** (1.0 / (self._beta + 1.0))
        sigma2xL = self._surfaceSigmaProfile.sigma2(xL, log=False)
        return -vR / sigma2xL

    def _dlnfdvT(self, R, vR, vT):
        # Calculate a bunch of stuff that we need
        E, L = vRvTRToEL(vR, vT, R, self._beta, self._dftype)
        if self._beta == 0.0:
            xL = L
            dRldvT = R
            ECL = numpylog(xL) + 0.5
            dECLEdvT = 1.0 / vT - vT
        else:  # non-flat rotation curve
            xL = L ** (1.0 / (self._beta + 1.0))
            dRldvT = L ** (1.0 / (self._beta + 1.0)) / vT / (self._beta + 1.0)
            ECL = 0.5 * (1.0 / self._beta + 1.0) * xL ** (2.0 * self._beta)
            dECLdRl = (1.0 + self._beta) * xL ** (2.0 * self._beta - 1)
            dEdvT = vT
            dECLEdvT = dECLdRl * dRldvT - dEdvT
        sigma2xL = self._surfaceSigmaProfile.sigma2(xL, log=False)
        return (
            self._surfaceSigmaProfile.surfacemassDerivative(xL, log=True)
            - (1.0 + (ECL - E) / sigma2xL)
            * self._surfaceSigmaProfile.sigma2Derivative(xL, log=True)
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
    vR *= sigmaR1
    vT *= sigmaR1 / gamma
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
        else:
            self._savedir = kwargs.get("savedir", _CORRECTIONSDIR)
            self._savefilename = self._createSavefilename(self._niter)
            if os.path.exists(self._savefilename):
                savefile = open(self._savefilename, "rb")
                self._corrections = numpy.array(pickle.load(savefile))
                savefile.close()
            else:  # Calculate the corrections
                self._corrections = self._calc_corrections()
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
        return None

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
        gamma = numpy.sqrt(2.0 / (1.0 + beta))
        L = R * vT
        if beta == 0.0:
            xL = L
        else:  # non-flat rotation curve
            xL = L ** (1.0 / (beta + 1.0))
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
    if beta == 0.0:
        if numpy.any(R == 0.0):
            out = numpy.empty(R.shape)
            out[R == 0.0] = numpylog(_RMIN)
            out[R != 0.0] = numpylog(R[R != 0.0])
            return out
        else:
            return numpylog(R)
    else:  # non-flat rotation curve
        return R ** (2.0 * beta) / 2.0 / beta


def _ars_hx(x, args):
    """
    h(x) for ARS sampling of the input surfacemass profile

    Parameters
    ----------
    x : float
        R(/ro)
    args : tuple
        surfaceSigma - surfaceSigmaProfile instance
        dfcorr - DFcorrection instance

    Returns
    -------
    float
        log(x)+log surface(x) + log(correction)

    Notes
    -----
    - 2010-07-11 - Written - Bovy (NYU)
    """

    surfaceSigma, dfcorr = args
    if dfcorr is None:
        return numpylog(x) + surfaceSigma.surfacemass(x, log=True)
    else:
        return (
            numpylog(x) + surfaceSigma.surfacemass(x, log=True) + dfcorr.correct(x)[0]
        )


def _ars_hpx(x, args):
    """
    h'(x) for ARS sampling of the input surfacemass profile

    Parameters
    ----------
    x : float
        R(/ro)
    args : tuple
        surfaceSigma - surfaceSigmaProfile instance
        dfcorr - DFcorrection instance

    Returns
    -------
    float
        derivative of log(x)+log surface(x) + log(correction) wrt x

    Notes
    -----
    - 2010-07-11 - Written - Bovy (NYU)
    """
    surfaceSigma, dfcorr = args
    if dfcorr is None:
        return 1.0 / x + surfaceSigma.surfacemassDerivative(x, log=True)
    else:
        return (
            1.0 / x
            + surfaceSigma.surfacemassDerivative(x, log=True)
            + dfcorr.derivLogcorrect(x)[0]
        )


def _kappa(R, beta):
    """Internal function to give kappa(r)"""
    return numpy.sqrt(2.0 * (1.0 + beta)) * R ** (beta - 1)


def _dlToRphi(d, l):
    """Convert d and l to R and phi, l is in radians"""
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
    sigma2xE = diskdf._surfaceSigmaProfile.sigma2(xE, log=False)
    return (
        OE * R / sigma2xE
        + (
            diskdf._surfaceSigmaProfile.surfacemassDerivative(xE, log=True)
            - (1.0 + OE * (L - LCE) / sigma2xE)
            * diskdf._surfaceSigmaProfile.sigma2Derivative(xE, log=True)
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
