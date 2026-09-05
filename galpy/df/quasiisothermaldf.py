# A 'Binney' quasi-isothermal DF
import hashlib
import warnings

import numpy
from scipy import integrate, interpolate, optimize

from .. import actionAngle, potential
from ..actionAngle import actionAngleIsochrone
from ..backend import (
    as_numpy,
    coerce_coords,
    get_namespace,
    is_backend_array,
    promote_scalars,
    use,
)
from ..backend.interpolate import Spline1D
from ..backend.quadrature import fixed_quad as _backend_fixed_quad
from ..orbit import Orbit
from ..potential import IsochronePotential
from ..potential.Potential import _check_potential_list_and_deprecate
from ..util import conversion, galpyWarning
from ..util._optional_deps import _APY_LOADED, _APY_UNITS
from ..util.conversion import (
    actionAngle_physical_input,
    parse_angmom,
    parse_length,
    parse_length_kpc,
    parse_velocity,
    parse_velocity_kms,
    physical_compatible,
    physical_conversion,
    potential_physical_input,
)
from .df import df

if _APY_LOADED:
    from astropy import units
_NSIGMA = 4
_DEFAULTNGL = 10
_DEFAULTNGL2 = 20


class quasiisothermaldf(df):
    """Class that represents a 'Binney' quasi-isothermal DF"""

    def __init__(
        self,
        hr,
        sr,
        sz,
        hsr,
        hsz,
        pot=None,
        aA=None,
        cutcounter=False,
        _precomputerg=True,
        _precomputergrmax=None,
        _precomputergnLz=51,
        refr=1.0,
        lo=10.0 / 220.0 / 8.0,
        ro=None,
        vo=None,
    ):
        """
        Initialize a quasi-isothermal DF

        Parameters
        ----------
        hr : float or Quantity
            Radial scale length.
        sr : float or Quantity
            Radial velocity dispersion at the solar radius.
        sz : float or Quantity
            Vertical velocity dispersion at the solar radius.
        hsr : float or Quantity
            Radial-velocity-dispersion scale length.
        hsz : float or Quantity
            Vertial-velocity-dispersion scale length.
        pot : Potential or a combined potential formed using addition (pot1+pot2+…)
            Potential or a combined potential formed using addition (pot1+pot2+…) of potentials that represents the underlying potential.
        aA : actionAngle instance
            ActionAngle instance used to convert (x,v) to actions [must be an instance of an actionAngle class that computes (J,Omega,angle) for a given (x,v)].
        cutcounter : bool, optional
            If True, set counter-rotating stars' DF to zero.
        refr : float or Quantity, optional
            Reference radius for dispersions (can be different from ro).
        lo : float or Quantity, optional
            Reference angular momentum below where there are significant numbers of retrograde stars.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).
        _precomputerg : bool, optional
            If True (default), pre-compute the rL(L).
        _precomputergrmax : float or Quantity, optional
            If set, this is the maximum R for which to pre-compute rg (default: 5*hr).
        _precomputergnLz : int, optional
            If set, number of Lz to pre-compute rg for (default: 51).

        Notes
        -----
        - 2012-07-25 - Started - Bovy (IAS@MPIA)
        """
        df.__init__(self, ro=ro, vo=vo)
        self._hr = parse_length(hr, ro=self._ro)
        self._sr = parse_velocity(sr, vo=self._vo)
        self._sz = parse_velocity(sz, vo=self._vo)
        self._hsr = parse_length(hsr, ro=self._ro)
        self._hsz = parse_length(hsz, ro=self._ro)
        self._refr = parse_length(refr, ro=self._ro)
        self._lo = parse_angmom(lo, ro=self._ro, vo=self._vo)
        self._lnsr = numpy.log(self._sr)
        self._lnsz = numpy.log(self._sz)
        self._maxVT_hash = None
        self._maxVT_ip = None
        if pot is None:
            raise OSError("pot= must be set")
        self._pot = _check_potential_list_and_deprecate(pot)
        if aA is None:
            raise OSError("aA= must be set")
        self._aA = aA
        if not self._aA._pot == self._pot:
            if not isinstance(self._aA, actionAngleIsochrone):
                raise OSError(
                    "Potential in aA does not appear to be the same as given potential pot"
                )
            elif (
                isinstance(self._pot, IsochronePotential)
                and not self._aA.b == self._pot.b
                and not self._aA.amp == self._pot._amp
            ):
                raise OSError(
                    "Potential in aA does not appear to be the same as given potential pot"
                )
        self._check_consistent_units()
        self._cutcounter = cutcounter
        if _precomputerg:
            if _precomputergrmax is None:
                _precomputergrmax = 5 * self._hr
            self._precomputergrmax = _precomputergrmax
            self._precomputergnLz = _precomputergnLz
            self._precomputergLzmin = 0.01
            # float(): under a forced backend vcirc returns a backend scalar, which
            # would make this grid bound a Tensor and break the numpy _rg branch's
            # `lz > self._precomputergLzmax` (ndarray > Tensor raises). Keep it a
            # Python scalar; the numpy path is byte-identical (linspace stop value).
            self._precomputergLzmax = float(
                self._precomputergrmax
                * potential.vcirc(self._pot, self._precomputergrmax)
            )
            self._precomputergLzgrid = numpy.linspace(
                self._precomputergLzmin, self._precomputergLzmax, self._precomputergnLz
            )
            self._rls = numpy.array(
                [potential.rl(self._pot, l) for l in self._precomputergLzgrid]
            )
            # Spline interpolate
            self._rgInterp = interpolate.InterpolatedUnivariateSpline(
                self._precomputergLzgrid, self._rls, k=3
            )
            # backend-array eval of the same spline (numpy path stays byte-identical)
            self._rgInterpBackend = Spline1D(self._precomputergLzgrid, self._rls, k=3)
        else:
            self._precomputergrmax = 0.0
            self._rgInterp = None
            self._rgInterpBackend = None
            self._rls = None
            self._precomputergnr = None
            self._precomputergLzgrid = None
            self._precomputergLzmin = numpy.finfo(numpy.dtype(numpy.float64)).max
            self._precomputergLzmax = numpy.finfo(numpy.dtype(numpy.float64)).min
        self._precomputerg = _precomputerg
        self._glxdef, self._glwdef = numpy.polynomial.legendre.leggauss(_DEFAULTNGL)
        self._glxdef2, self._glwdef2 = numpy.polynomial.legendre.leggauss(_DEFAULTNGL2)
        self._glxdef12, self._glwdef12 = numpy.polynomial.legendre.leggauss(
            _DEFAULTNGL // 2
        )
        return None

    @physical_conversion("phasespacedensity", pop=True)
    def __call__(self, *args, **kwargs):
        """
        Evaluate the DF

        Parameters
        ----------
        args: tuple or Orbit
            Either:
                a) (jr,lz,jz) tuple; each can be a Quantity
                    where:
                        * jr - radial action
                        * lz - z-component of angular momentum
                        * jz - vertical action
                b) R,vR,vT,z,vz
                c) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well
        log: bool, optional
            If True, return the natural log.
        func: function of (jr,lz,jz), optional
            Function of the actions to multiply the DF with (useful for moments).
        _return_actions: bool, optional
            If True, return the actions as well.
        _return_freqs: bool, optional
            If True, return the frequencies as well.
        _return_rgr: bool, optional
            If True, return the rg as well.
        kwargs: dict, optional
            scipy.integrate.quadrature kwargs.

        Returns
        -------
        float
            Value of DF.

        Notes
        -----
        - 2012-07-25 - Written - Bovy (IAS@MPIA)
        """
        # First parse log
        log = kwargs.pop("log", False)
        _return_actions = kwargs.pop("_return_actions", False)
        _return_freqs = kwargs.pop("_return_freqs", False)
        _func = kwargs.pop("func", None)
        if "rg" in kwargs:
            thisrg = kwargs.pop("rg")
            kappa = kwargs.pop("kappa")
            nu = kwargs.pop("nu")
            Omega = kwargs.pop("Omega")
        else:
            thisrg = None
            kappa = None
            nu = None
            Omega = None
        # First parse args
        if len(args) == 1 and not isinstance(args[0], Orbit):  # (jr,lz,jz)
            jr, lz, jz = args[0]
            jr = parse_angmom(jr, ro=self._ro, vo=self._vo)
            lz = parse_angmom(lz, ro=self._ro, vo=self._vo)
            jz = parse_angmom(jz, ro=self._ro, vo=self._vo)
        else:
            # Use self._aA to calculate the actions
            if isinstance(args[0], Orbit) and len(args[0].shape) > 1:
                raise RuntimeError(
                    "Evaluating quasiisothermaldf with Orbit instances with multi-dimensional shapes is not supported"
                )  # pragma: no cover
            try:
                jr, lz, jz = self._aA(*args, use_physical=False, **kwargs)
            except actionAngle.UnboundError:
                if log:
                    return -numpy.finfo(numpy.dtype(numpy.float64)).max
                else:
                    return 0.0
            # if isinstance(jr,(list,numpy.ndarray)) and len(jr) > 1: jr= jr[0]
            # if isinstance(jz,(list,numpy.ndarray)) and len(jz) > 1: jz= jz[0]
        xp = get_namespace(jr, lz, jz)
        jr, lz, jz = coerce_coords(xp, jr, lz, jz)  # torch rejects python-float xp.abs
        if (
            not isinstance(lz, numpy.ndarray)
            and not is_backend_array(lz)
            and self._cutcounter
            and lz < 0.0
        ):
            if log:
                return -numpy.finfo(numpy.dtype(numpy.float64)).max
            else:
                return 0.0
        # First calculate rg
        if thisrg is None:
            thisrg = self._rg(lz)
            # Then calculate the epicycle and vertical frequencies
            kappa, nu = self._calc_epifreq(thisrg), self._calc_verticalfreq(thisrg)
            Omega = xp.abs(lz) / thisrg / thisrg
        # calculate surface-densities and sigmas
        lnsurfmass = (self._refr - thisrg) / self._hr
        lnsr = self._lnsr + (self._refr - thisrg) / self._hsr
        lnsz = self._lnsz + (self._refr - thisrg) / self._hsz
        # Calculate func
        if not _func is None:
            if log:
                funcTerm = xp.log(_func(jr, lz, jz))
            else:
                funcFactor = _func(jr, lz, jz)
        # Calculate fsr
        else:
            if log:
                funcTerm = 0.0
            else:
                funcFactor = 1.0
        if log:
            lnfsr = (
                xp.log(Omega)
                + lnsurfmass
                - 2.0 * lnsr
                - numpy.log(numpy.pi)
                - xp.log(kappa)
                + xp.log(1.0 + xp.tanh(lz / self._lo))
                - kappa * jr * xp.exp(-2.0 * lnsr)
            )
            lnfsz = (
                xp.log(nu)
                - numpy.log(2.0 * numpy.pi)
                - 2.0 * lnsz
                - nu * jz * xp.exp(-2.0 * lnsz)
            )
            out = lnfsr + lnfsz + funcTerm
            if is_backend_array(out):
                sentinel = -xp.finfo(out.dtype).max
                out = xp.where(xp.isnan(out), sentinel, out)
                if self._cutcounter:
                    out = xp.where(lz < 0.0, sentinel, out)
            elif isinstance(lz, numpy.ndarray):
                out[numpy.isnan(out)] = -numpy.finfo(numpy.dtype(numpy.float64)).max
                if self._cutcounter:
                    out[(lz < 0.0)] = -numpy.finfo(numpy.dtype(numpy.float64)).max
            elif numpy.isnan(out):  # pragma: no cover
                out = -numpy.finfo(numpy.dtype(numpy.float64)).max
        else:
            srm2 = xp.exp(-2.0 * lnsr)
            fsr = (
                Omega
                * xp.exp(lnsurfmass)
                * srm2
                / numpy.pi
                / kappa
                * (1.0 + xp.tanh(lz / self._lo))
                * xp.exp(-kappa * jr * srm2)
            )
            szm2 = xp.exp(-2.0 * lnsz)
            fsz = nu / 2.0 / numpy.pi * szm2 * xp.exp(-nu * jz * szm2)
            out = fsr * fsz * funcFactor
            if is_backend_array(out):
                out = xp.where(xp.isnan(out), 0.0, out)
                if self._cutcounter:
                    out = xp.where(lz < 0.0, 0.0, out)
            elif isinstance(lz, numpy.ndarray):
                out[numpy.isnan(out)] = 0.0
                if self._cutcounter:
                    out[(lz < 0.0)] = 0.0
            elif numpy.isnan(out):  # pragma: no cover
                out = 0.0
        if _return_actions and _return_freqs:
            return (out, jr, lz, jz, thisrg, kappa, nu, Omega)
        elif _return_actions:
            return (out, jr, lz, jz)
        elif _return_freqs:
            return (out, thisrg, kappa, nu, Omega)
        else:
            return out

    @potential_physical_input
    @physical_conversion("position", pop=True)
    def estimate_hr(self, R, z=0.0, dR=10.0**-8.0, **kwargs):
        """
        Estimate the exponential scale length at R.

        Parameters
        ----------
        R : float or Quantity
            Galactocentric radius.
        z : float or Quantity, optional
            Height (default: 0 pc).
        dR : float or Quantity, optional
            Range in R to use.
        **kwargs
            Density kwargs.

        Returns
        -------
        float or Quantity
            Estimated hR.

        Notes
        -----
        - 2012-09-11 - Written - Bovy (IAS)
        - 2013-01-28 - Re-written - Bovy
        """
        Rs = [R - dR / 2.0, R + dR / 2.0]
        if z is None:
            sf = numpy.array(
                [self.surfacemass_z(r, use_physical=False, **kwargs) for r in Rs]
            )
        else:
            sf = numpy.array(
                [self.density(r, z, use_physical=False, **kwargs) for r in Rs]
            )
        lsf = numpy.log(sf)
        return -dR / (lsf[1] - lsf[0])

    @potential_physical_input
    @physical_conversion("position", pop=True)
    def estimate_hz(self, R, z, dz=10.0**-8.0, **kwargs):
        """
        Estimate the exponential scale height at R.

        Parameters
        ----------
        R : float or Quantity
            Galactocentric radius.
        z : float or Quantity
            Height above the Galactic plane.
        dz : float or Quantity, optional
            z range to use.
        **kwargs
            density kwargs.

        Returns
        -------
        float or Quantity
            Estimated hz.

        Notes
        -----
        - 2012-08-30 - Written - Bovy (IAS)
        - 2013-01-28 - Re-written - Bovy
        """
        if z == 0.0:
            zs = [z, z + dz]
        else:
            zs = [z - dz / 2.0, z + dz / 2.0]
        sf = numpy.array(
            [self.density(R, zz, use_physical=False, **kwargs) for zz in zs]
        )
        lsf = numpy.log(sf)
        return -dz / (lsf[1] - lsf[0])

    @potential_physical_input
    @physical_conversion("position", pop=True)
    def estimate_hsr(self, R, z=0.0, dR=10.0**-8.0, **kwargs):
        """
        Estimate the exponential scale length of the radial dispersion at R.

        Parameters
        ----------
        R : float or Quantity
            Galactocentric radius.
        z : float or Quantity, optional
            Height (default: 0 pc).
        dR : float or Quantity, optional
            Range in R to use.
        **kwargs
            Density kwargs.

        Returns
        -------
        float or Quantity
            Estimated hsR.

        Notes
        -----
        - 2013-03-08 - Written - Bovy (IAS)

        """
        Rs = [R - dR / 2.0, R + dR / 2.0]
        sf = numpy.array([self.sigmaR2(r, z, use_physical=False, **kwargs) for r in Rs])
        lsf = numpy.log(sf) / 2.0
        return -dR / (lsf[1] - lsf[0])

    @potential_physical_input
    @physical_conversion("position", pop=True)
    def estimate_hsz(self, R, z=0.0, dR=10.0**-8.0, **kwargs):
        """
        Estimate the exponential scale length of the vertical dispersion at R.

        Parameters
        ----------
        R : float or Quantity
            Galactocentric radius.
        z : float or Quantity, optional
            Height (default: 0 pc).
        dR : float or Quantity, optional
            Range in R to use.
        **kwargs
            Density kwargs.

        Returns
        -------
        float or Quantity
            Estimated hsz.

        Notes
        -----
        - 2013-03-08 - Written - Bovy (IAS)

        """
        Rs = [R - dR / 2.0, R + dR / 2.0]
        sf = numpy.array([self.sigmaz2(r, z, use_physical=False, **kwargs) for r in Rs])
        lsf = numpy.log(sf) / 2.0
        return -dR / (lsf[1] - lsf[0])

    @potential_physical_input
    @physical_conversion("numbersurfacedensity", pop=True)
    def surfacemass_z(
        self, R, nz=7, zmax=1.0, fixed_quad=True, fixed_order=8, **kwargs
    ):
        """
        Calculate the vertically-integrated surface density.

        Parameters
        ----------
        R : float or Quantity
            Galactocentric radius.
        nz : int, optional
            Number of zs to use to estimate. Default is 7.
        zmax : float or Quantity, optional
            Maximum z to use. Default is 1.0.
        fixed_quad : bool, optional
            If True (default), use Gauss-Legendre integration.
        fixed_order : int, optional
            Order of GL integration to use. Default is 8.
        **kwargs : dict
            Density kwargs.

        Returns
        -------
        float or Quantity
            Surface density at R.

        Notes
        -----
        - 2012-08-30 - Written - Bovy (IAS)
        """
        if fixed_quad:
            xp = get_namespace(R)
            if xp is not numpy:
                # backend GL quadrature (scipy fixed_quad multiplies its numpy
                # weights by the backend integrand -> breaks torch); numpy path
                # below is byte-identical (scipy).
                (R,) = promote_scalars(xp, R)
                return 2.0 * _backend_fixed_quad(
                    xp,
                    lambda x: self.density(R * xp.ones_like(x), x, use_physical=False),
                    0.0,
                    0.5,
                    n=fixed_order,
                )
            return (
                2.0
                * integrate.fixed_quad(
                    lambda x: self.density(
                        R * numpy.ones(fixed_order), x, use_physical=False
                    ),
                    0.0,
                    0.5,
                    n=fixed_order,
                )[0]
            )
        zs = numpy.linspace(0.0, zmax, nz)
        sf = numpy.array([self.density(R, z, use_physical=False, **kwargs) for z in zs])
        lsf = numpy.log(sf)
        # Interpolate
        lsfInterp = interpolate.UnivariateSpline(zs, lsf, k=3)
        # Integrate
        return 2.0 * integrate.quad((lambda x: numpy.exp(lsfInterp(x))), 0.0, 1.0)[0]

    def vmomentdensity(self, *args, **kwargs):
        """
        Calculate the an arbitrary moment of the velocity distribution at R times the density

        Parameters
        ----------
        R : float
            radius at which to calculate the moment(/ro)
        z : float
            height at which to calculate the moment(/ro)
        n : int
            vR^n
        m : int
            vT^m
        o : int
            vz^o
        nsigma : int, optional
            number of sigma to integrate the vR and vz velocities over (when doing explicit numerical integral; default: 4)
        vTmax : float, optional
            upper limit for integration over vT (default: 1.5)
        mc : bool, optional
            if True, calculate using Monte Carlo integration
        nmc : int, optional
            if mc, use nmc samples
        gl : bool, optional
            use Gauss-Legendre
        _returngl : bool, optional
            if True, return the evaluated DF
        _return_actions : bool, optional
            if True, return the evaluated actions (does not work with _returngl currently)
        _return_freqs : bool, optional
            if True, return the evaluated frequencies and rg (does not work with _returngl currently)

        Returns
        -------
        float
            <vR^n vT^m  x density> at R,z (no support for units)

        Notes
        -----
        - 2012-08-06 - Written - Bovy (IAS@MPIA)

        """
        use_physical = kwargs.pop("use_physical", True)
        ro = kwargs.pop("ro", None)
        if ro is None and hasattr(self, "_roSet") and self._roSet:
            ro = self._ro
        ro = parse_length_kpc(ro)
        vo = kwargs.pop("vo", None)
        if vo is None and hasattr(self, "_voSet") and self._voSet:
            vo = self._vo
        vo = parse_velocity_kms(vo)
        if use_physical and not vo is None and not ro is None:
            fac = vo ** (args[2] + args[3] + args[4]) / ro**3
            if _APY_UNITS:
                u = (
                    1
                    / units.kpc**3
                    * (units.km / units.s) ** (args[2] + args[3] + args[4])
                )
            out = self._vmomentdensity(*args, **kwargs)
            if _APY_UNITS:
                return units.Quantity(out * fac, unit=u)
            else:
                return out * fac
        else:
            return self._vmomentdensity(*args, **kwargs)

    def _vmomentdensity(
        self,
        R,
        z,
        n,
        m,
        o,
        nsigma=None,
        mc=False,
        nmc=10000,
        _returnmc=False,
        _vrs=None,
        _vts=None,
        _vzs=None,
        _rawgausssamples=False,
        gl=False,
        ngl=_DEFAULTNGL,
        _returngl=False,
        _glqeval=None,
        _return_actions=False,
        _jr=None,
        _lz=None,
        _jz=None,
        _return_freqs=False,
        _rg=None,
        _kappa=None,
        _nu=None,
        _Omega=None,
        _sigmaR1=None,
        _sigmaz1=None,
        **kwargs,
    ):
        """Non-physical version of vmomentdensity, otherwise the same"""
        xp = get_namespace(R, z)
        if getattr(R, "ndim", 0) > 0:
            # array R (numpy or backend): the GL grid below is per-scalar-R, so
            # recurse per (r,z) and collect on the resolved namespace -- xp.stack
            # under a forced backend (so numpy-array inputs run on the backend
            # too), numpy.array on numpy (byte-identical). 0-D backend scalars
            # from the recursion have ndim==0 and fall through to the scalar body.
            results = [
                self._vmomentdensity(
                    r,
                    zz,
                    n,
                    m,
                    o,
                    nsigma=nsigma,
                    mc=mc,
                    nmc=nmc,
                    gl=gl,
                    ngl=ngl,
                    **kwargs,
                )
                for r, zz in zip(R, z)
            ]
            return numpy.array(results) if xp is numpy else xp.stack(results)
        if isinstance(
            self._aA,
            (actionAngle.actionAngleAdiabatic, actionAngle.actionAngleAdiabaticGrid),
        ):
            if n % 2 == 1.0 or o % 2 == 1.0:
                return 0.0  # we know this must be the case
        if nsigma == None:
            nsigma = _NSIGMA
        if xp is not numpy:
            # promote the scalar (R,z) up to the backend so xp.exp(...) etc. run
            # on it (torch rejects Python floats); numpy path is a no-op.
            R, z = promote_scalars(xp, R, z)
        if _sigmaR1 is None:
            sigmaR1 = self._sr * xp.exp((self._refr - R) / self._hsr)
        else:
            sigmaR1 = _sigmaR1
        if _sigmaz1 is None:
            sigmaz1 = self._sz * xp.exp((self._refr - R) / self._hsz)
        else:
            sigmaz1 = _sigmaz1
        thisvc = potential.vcirc(self._pot, R, use_physical=False)
        # Use the asymmetric drift equation to estimate va
        gamma = numpy.sqrt(0.5)
        va = (
            sigmaR1**2.0
            / 2.0
            / thisvc
            * (
                gamma**2.0
                - 1.0  # Assume close to flat rotation curve, sigphi2/sigR2 =~ 0.5
                + R * (1.0 / self._hr + 2.0 / self._hsr)
            )
        )
        if is_backend_array(va):
            va = xp.where(xp.abs(va) > sigmaR1, 0.0, va)  # avoid craziness near center
        elif numpy.fabs(va) > sigmaR1:
            va = 0.0  # To avoid craziness near the center
        if gl:
            if ngl % 2 == 1:
                raise ValueError("ngl must be even")
            if not _glqeval is None and ngl != _glqeval.shape[0]:
                _glqeval = None
            # Use Gauss-Legendre integration for all
            if ngl == _DEFAULTNGL:
                glx, glw = self._glxdef, self._glwdef
                glx12, glw12 = self._glxdef12, self._glwdef12
            elif ngl == _DEFAULTNGL2:
                glx, glw = self._glxdef2, self._glwdef2
                glx12, glw12 = self._glxdef, self._glwdef
            else:
                glx, glw = numpy.polynomial.legendre.leggauss(ngl)
                glx12, glw12 = numpy.polynomial.legendre.leggauss(ngl // 2)
            if xp is not numpy:
                # promote the precomputed GL node/weight tables to the backend
                # (numpy path keeps the numpy tables -> byte-identical)
                glx, glw = xp.asarray(glx) * 1.0, xp.asarray(glw) * 1.0
                glx12, glw12 = xp.asarray(glx12) * 1.0, xp.asarray(glw12) * 1.0
            # Evaluate everywhere
            if isinstance(
                self._aA,
                (
                    actionAngle.actionAngleAdiabatic,
                    actionAngle.actionAngleAdiabaticGrid,
                ),
            ):
                vRgl = nsigma * sigmaR1 / 2.0 * (glx + 1.0)
                vzgl = nsigma * sigmaz1 / 2.0 * (glx + 1.0)
                vRglw = glw
                vzglw = glw
            else:
                vRgl = xp.concatenate(
                    [
                        nsigma * sigmaR1 / 2.0 * (glx12 + 1.0),
                        -nsigma * sigmaR1 / 2.0 * (glx12 + 1.0),
                    ]
                )
                vzgl = xp.concatenate(
                    [
                        nsigma * sigmaz1 / 2.0 * (glx12 + 1.0),
                        -nsigma * sigmaz1 / 2.0 * (glx12 + 1.0),
                    ]
                )
                vRglw = xp.concatenate([glw12, glw12])
                vzglw = xp.concatenate([glw12, glw12])
            vTmax = kwargs.get("vTmax", 1.5)
            vTgl = vTmax / 2.0 * (glx + 1.0)
            # Tile everything (permute_dims not .T: torch errors on 3-D .T under -W)
            vTgl = xp.permute_dims(xp.tile(vTgl, (ngl, ngl, 1)), (2, 1, 0))
            vRgl = xp.tile(xp.reshape(vRgl, (1, ngl)).T, (ngl, 1, ngl))
            vzgl = xp.tile(vzgl, (ngl, ngl, 1))
            vTglw = xp.permute_dims(xp.tile(glw, (ngl, ngl, 1)), (2, 1, 0))
            vRglw = xp.tile(xp.reshape(vRglw, (1, ngl)).T, (ngl, 1, ngl))
            vzglw = xp.tile(vzglw, (ngl, ngl, 1))
            # evaluate
            if _glqeval is None and _jr is None:
                logqeval, jr, lz, jz, rg, kappa, nu, Omega = self(
                    R + xp.zeros(ngl * ngl * ngl),
                    vRgl.flatten(),
                    vTgl.flatten(),
                    z + xp.zeros(ngl * ngl * ngl),
                    vzgl.flatten(),
                    log=True,
                    _return_actions=True,
                    _return_freqs=True,
                    use_physical=False,
                )
                logqeval = xp.reshape(logqeval, (ngl, ngl, ngl))
            elif not _jr is None and _rg is None:
                logqeval, jr, lz, jz, rg, kappa, nu, Omega = self(
                    (_jr, _lz, _jz),
                    log=True,
                    _return_actions=True,
                    _return_freqs=True,
                    use_physical=False,
                )
                logqeval = xp.reshape(logqeval, (ngl, ngl, ngl))
            elif not _jr is None and not _rg is None:
                logqeval, jr, lz, jz, rg, kappa, nu, Omega = self(
                    (_jr, _lz, _jz),
                    rg=_rg,
                    kappa=_kappa,
                    nu=_nu,
                    Omega=_Omega,
                    log=True,
                    _return_actions=True,
                    _return_freqs=True,
                    use_physical=False,
                )
                logqeval = xp.reshape(logqeval, (ngl, ngl, ngl))
            else:
                logqeval = _glqeval
            if _returngl:
                return (
                    xp.sum(
                        xp.exp(logqeval)
                        * vRgl**n
                        * vTgl**m
                        * vzgl**o
                        * vTglw
                        * vRglw
                        * vzglw
                    )
                    * sigmaR1
                    * sigmaz1
                    * 0.125
                    * vTmax
                    * nsigma**2,
                    logqeval,
                )
            elif _return_actions and _return_freqs:
                return (
                    xp.sum(
                        xp.exp(logqeval)
                        * vRgl**n
                        * vTgl**m
                        * vzgl**o
                        * vTglw
                        * vRglw
                        * vzglw
                    )
                    * sigmaR1
                    * sigmaz1
                    * 0.125
                    * vTmax
                    * nsigma**2,
                    jr,
                    lz,
                    jz,
                    rg,
                    kappa,
                    nu,
                    Omega,
                )
            elif _return_actions:
                return (
                    xp.sum(
                        xp.exp(logqeval)
                        * vRgl**n
                        * vTgl**m
                        * vzgl**o
                        * vTglw
                        * vRglw
                        * vzglw
                    )
                    * sigmaR1
                    * sigmaz1
                    * 0.125
                    * vTmax
                    * nsigma**2,
                    jr,
                    lz,
                    jz,
                )
            else:
                return xp.sum(
                    xp.exp(logqeval)
                    * vRgl**n
                    * vTgl**m
                    * vzgl**o
                    * vTglw
                    * vRglw
                    * vzglw
                    * sigmaR1
                    * sigmaz1
                    * 0.125
                    * vTmax
                    * nsigma**2
                )
        elif mc:
            mvT = (thisvc - va) / gamma / sigmaR1
            if _vrs is None:
                vrs = numpy.random.normal(size=nmc)
            else:
                vrs = _vrs
            # mvT is baked into the vt samples when freshly drawn or when raw
            # gaussians are supplied; defer the add so the numpy.random draw order
            # (hence the stream) is byte-identical to the original interleaving.
            add_mvT_to_vts = _vts is None or _rawgausssamples
            if _vts is None:
                vts = numpy.random.normal(size=nmc)
            else:
                vts = _vts
            if _vzs is None:
                vzs = numpy.random.normal(size=nmc)
            else:
                vzs = _vzs
            if xp is not numpy:  # promote the (numpy) draws to combine with backend
                vrs, vts, vzs = promote_scalars(xp, vrs, vts, vzs)
            if add_mvT_to_vts:
                vts = vts + mvT
            Is = _vmomentsurfaceMCIntegrand(
                vzs,
                vrs,
                vts,
                xp.ones(nmc) * R,
                xp.ones(nmc) * z,
                self,
                sigmaR1,
                gamma,
                sigmaz1,
                mvT,
                n,
                m,
                o,
            )
            if _returnmc:
                if _rawgausssamples:
                    return (
                        xp.mean(Is)
                        * sigmaR1 ** (2.0 + n + m)
                        * gamma ** (1.0 + m)
                        * sigmaz1 ** (1.0 + o),
                        vrs,
                        vts - mvT,
                        vzs,
                    )
                else:
                    return (
                        xp.mean(Is)
                        * sigmaR1 ** (2.0 + n + m)
                        * gamma ** (1.0 + m)
                        * sigmaz1 ** (1.0 + o),
                        vrs,
                        vts,
                        vzs,
                    )
            else:
                return (
                    xp.mean(Is)
                    * sigmaR1 ** (2.0 + n + m)
                    * gamma ** (1.0 + m)
                    * sigmaz1 ** (1.0 + o)
                )
        else:  # pragma: no cover because this is too slow; a warning is shown
            warnings.warn(
                "Calculations using direct numerical integration using tplquad is not recommended and extremely slow; it has also not been carefully tested",
                galpyWarning,
            )
            return (
                integrate.tplquad(
                    _vmomentsurfaceIntegrand,
                    1.0 / gamma * (thisvc - va) / sigmaR1 - nsigma,
                    1.0 / gamma * (thisvc - va) / sigmaR1 + nsigma,
                    lambda x: 0.0,
                    lambda x: nsigma,
                    lambda x, y: 0.0,
                    lambda x, y: nsigma,
                    (R, z, self, sigmaR1, gamma, sigmaz1, n, m, o),
                    **kwargs,
                )[0]
                * sigmaR1 ** (2.0 + n + m)
                * gamma ** (1.0 + m)
                * sigmaz1 ** (1.0 + o)
            )

    def jmomentdensity(self, *args, **kwargs):
        """
        Calculate the an arbitrary moment of an action of the velocity distribution at R times the surfacmass.

        Parameters
        ----------
        R : float
            radius at which to calculate the moment(/ro)
        z : float
            height at which to calculate the moment(/ro)
        n : int
            jr^n
        m : int
            lz^m
        o : int
            jz^o
        nsigma : int, optional
            Number of sigma to integrate the velocities over (when doing explicit numerical integral). Default is None.
        mc : bool, optional
            If True, calculate using Monte Carlo integration. Default is False.
        nmc : int, optional
            If mc is True, use nmc samples. Default is None.

        Returns
        -------
        float or Quantity
            <jr^n lz^m jz^o  x density> at R (no support for units)

        Notes
        -----
        - 2012-08-09 - Written - Bovy (IAS@MPIA)

        """
        use_physical = kwargs.pop("use_physical", True)
        ro = kwargs.pop("ro", None)
        if ro is None and hasattr(self, "_roSet") and self._roSet:
            ro = self._ro
        ro = parse_length_kpc(ro)
        vo = kwargs.pop("vo", None)
        if vo is None and hasattr(self, "_voSet") and self._voSet:
            vo = self._vo
        vo = parse_velocity_kms(vo)
        if use_physical and not vo is None and not ro is None:
            fac = (ro * vo) ** (args[2] + args[3] + args[4]) / ro**3
            if _APY_UNITS:
                u = (
                    1
                    / units.kpc**3
                    * (units.kpc * units.km / units.s) ** (args[2] + args[3] + args[4])
                )
            out = self._jmomentdensity(*args, **kwargs)
            if _APY_UNITS:
                return units.Quantity(out * fac, unit=u)
            else:
                return out * fac
        else:
            return self._jmomentdensity(*args, **kwargs)

    def _jmomentdensity(
        self,
        R,
        z,
        n,
        m,
        o,
        nsigma=None,
        mc=True,
        nmc=10000,
        _returnmc=False,
        _vrs=None,
        _vts=None,
        _vzs=None,
        **kwargs,
    ):
        """Non-physical version of jmomentdensity, otherwise the same"""
        xp = get_namespace(R, z)
        if nsigma == None:
            nsigma = _NSIGMA
        if xp is not numpy:  # promote scalar (R,z) so xp.exp etc. run on backend
            R, z = promote_scalars(xp, R, z)
        sigmaR1 = self._sr * xp.exp((self._refr - R) / self._hsr)
        sigmaz1 = self._sz * xp.exp((self._refr - R) / self._hsz)
        thisvc = potential.vcirc(self._pot, R, use_physical=False)
        # Use the asymmetric drift equation to estimate va
        gamma = numpy.sqrt(0.5)
        va = (
            sigmaR1**2.0
            / 2.0
            / thisvc
            * (
                gamma**2.0
                - 1.0  # Assume close to flat rotation curve, sigphi2/sigR2 =~ 0.5
                + R * (1.0 / self._hr + 2.0 / self._hsr)
            )
        )
        if is_backend_array(va):
            va = xp.where(xp.abs(va) > sigmaR1, 0.0, va)  # avoid craziness near center
        elif numpy.fabs(va) > sigmaR1:
            va = 0.0  # To avoid craziness near the center
        if mc:
            mvT = (thisvc - va) / gamma / sigmaR1
            if _vrs is None:
                vrs = numpy.random.normal(size=nmc)
            else:
                vrs = _vrs
            # defer the mvT add so the numpy.random draw order is byte-identical
            add_mvT_to_vts = _vts is None
            if _vts is None:
                vts = numpy.random.normal(size=nmc)
            else:
                vts = _vts
            if _vzs is None:
                vzs = numpy.random.normal(size=nmc)
            else:
                vzs = _vzs
            if xp is not numpy:  # promote the (numpy) draws to combine with backend
                vrs, vts, vzs = promote_scalars(xp, vrs, vts, vzs)
            if add_mvT_to_vts:
                vts = vts + mvT
            Is = _jmomentsurfaceMCIntegrand(
                vzs,
                vrs,
                vts,
                xp.ones(nmc) * R,
                xp.ones(nmc) * z,
                self,
                sigmaR1,
                gamma,
                sigmaz1,
                mvT,
                n,
                m,
                o,
            )
            if _returnmc:
                return (
                    xp.mean(Is) * sigmaR1**2.0 * gamma * sigmaz1,
                    vrs,
                    vts,
                    vzs,
                )
            else:
                return xp.mean(Is) * sigmaR1**2.0 * gamma * sigmaz1
        else:  # pragma: no cover because this is too slow; a warning is shown
            warnings.warn(
                "Calculations using direct numerical integration using tplquad is not recommended and extremely slow; it has also not been carefully tested",
                galpyWarning,
            )
            return (
                integrate.tplquad(
                    _jmomentsurfaceIntegrand,
                    1.0 / gamma * (thisvc - va) / sigmaR1 - nsigma,
                    1.0 / gamma * (thisvc - va) / sigmaR1 + nsigma,
                    lambda x: 0.0,
                    lambda x: nsigma,
                    lambda x, y: 0.0,
                    lambda x, y: nsigma,
                    (R, z, self, sigmaR1, gamma, sigmaz1, n, m, o),
                    **kwargs,
                )[0]
                * sigmaR1**2.0
                * gamma
                * sigmaz1
            )

    @potential_physical_input
    @physical_conversion("numberdensity", pop=True)
    def density(
        self, R, z, nsigma=None, mc=False, nmc=10000, gl=True, ngl=_DEFAULTNGL, **kwargs
    ):
        """
        Calculate the density at R,z by marginalizing over velocity.

        Parameters
        ----------
        R : float or Quantity
            Radius at which to calculate the density.
        z : float or Quantity
            Height at which to calculate the density.
        nsigma : float, optional
            Number of sigma to integrate the velocities over.
        mc : bool, optional
            If True, calculate using Monte Carlo integration.
        nmc : int, optional
            If mc, use nmc samples.
        gl : bool, optional
            If True, calculate using Gauss-Legendre integration.
        ngl : int, optional
            If gl, use ngl-th order Gauss-Legendre integration for each dimension.
        **kwargs : dict, optional
            scipy.integrate.tplquad kwargs epsabs and epsrel.

        Returns
        -------
        float
            Density at (R,z).

        Notes
        -----
        - 2012-07-26 - Written - Bovy (IAS@MPIA)

        """
        return self._vmomentdensity(
            R, z, 0.0, 0.0, 0.0, nsigma=nsigma, mc=mc, nmc=nmc, gl=gl, ngl=ngl, **kwargs
        )

    @potential_physical_input
    @physical_conversion("velocity2", pop=True)
    def sigmaR2(
        self, R, z, nsigma=None, mc=False, nmc=10000, gl=True, ngl=_DEFAULTNGL, **kwargs
    ):
        """
        Calculate sigma_R^2 by marginalizing over velocity.

        Parameters
        ----------
        R : float or Quantity
            Radius at which to calculate this.
        z : float or Quantity
            Height at which to calculate this.
        nsigma : int, optional
            Number of sigma to integrate the velocities over.
        mc : bool, optional
            If True, calculate using Monte Carlo integration.
        nmc : int, optional
            If mc, use nmc samples.
        gl : bool, optional
            If True, calculate using Gauss-Legendre integration.
        ngl : int, optional
            If gl, use ngl-th order Gauss-Legendre integration for each dimension.
        **kwargs : dict, optional
            scipy.integrate.tplquad kwargs epsabs and epsrel.

        Returns
        -------
        float
            sigma_R^2.

        Notes
        -----
        - 2012-07-30 - Written - Bovy (IAS@MPIA)

        """
        if mc:
            surfmass, vrs, vts, vzs = self._vmomentdensity(
                R,
                z,
                0.0,
                0.0,
                0.0,
                nsigma=nsigma,
                mc=mc,
                nmc=nmc,
                _returnmc=True,
                **kwargs,
            )
            return (
                self._vmomentdensity(
                    R,
                    z,
                    2.0,
                    0.0,
                    0.0,
                    nsigma=nsigma,
                    mc=mc,
                    nmc=nmc,
                    _returnmc=False,
                    _vrs=vrs,
                    _vts=vts,
                    _vzs=vzs,
                    **kwargs,
                )
                / surfmass
            )
        elif gl:
            surfmass, glqeval = self._vmomentdensity(
                R, z, 0.0, 0.0, 0.0, gl=gl, ngl=ngl, _returngl=True, **kwargs
            )
            return (
                self._vmomentdensity(
                    R, z, 2.0, 0.0, 0.0, ngl=ngl, gl=gl, _glqeval=glqeval, **kwargs
                )
                / surfmass
            )
        else:  # pragma: no cover because this is too slow; a warning is shown
            return self._vmomentdensity(
                R, z, 2.0, 0.0, 0.0, nsigma=nsigma, mc=mc, nmc=nmc, **kwargs
            ) / self._vmomentdensity(
                R, z, 0.0, 0.0, 0.0, nsigma=nsigma, mc=mc, nmc=nmc, **kwargs
            )

    @potential_physical_input
    @physical_conversion("velocity2", pop=True)
    def sigmaRz(
        self, R, z, nsigma=None, mc=False, nmc=10000, gl=True, ngl=_DEFAULTNGL, **kwargs
    ):
        """
        Calculate sigma_RZ^2 by marginalizing over velocity.

        Parameters
        ----------
        R : float or Quantity
            Radius at which to calculate this.
        z : float or Quantity
            Height at which to calculate this.
        nsigma : int, optional
            Number of sigma to integrate the velocities over.
        mc : bool, optional
            If True, calculate using Monte Carlo integration.
        nmc : int, optional
            If mc, use nmc samples.
        gl : bool, optional
            If True, calculate using Gauss-Legendre integration.
        ngl : int, optional
            If gl, use ngl-th order Gauss-Legendre integration for each dimension.
        **kwargs
            scipy.integrate.tplquad kwargs epsabs and epsrel.

        Returns
        -------
        float
            sigma_Rz^2.

        Notes
        -----
        - 2012-07-30 - Written - Bovy (IAS@MPIA)

        """
        if mc:
            surfmass, vrs, vts, vzs = self._vmomentdensity(
                R,
                z,
                0.0,
                0.0,
                0.0,
                nsigma=nsigma,
                mc=mc,
                nmc=nmc,
                _returnmc=True,
                **kwargs,
            )
            return (
                self._vmomentdensity(
                    R,
                    z,
                    1.0,
                    0.0,
                    1.0,
                    nsigma=nsigma,
                    mc=mc,
                    nmc=nmc,
                    _returnmc=False,
                    _vrs=vrs,
                    _vts=vts,
                    _vzs=vzs,
                    **kwargs,
                )
                / surfmass
            )
        elif gl:
            surfmass, glqeval = self._vmomentdensity(
                R, z, 0.0, 0.0, 0.0, gl=gl, ngl=ngl, _returngl=True, **kwargs
            )
            return (
                self._vmomentdensity(
                    R, z, 1.0, 0.0, 1.0, ngl=ngl, gl=gl, _glqeval=glqeval, **kwargs
                )
                / surfmass
            )
        else:  # pragma: no cover because this is too slow; a warning is shown
            return self._vmomentdensity(
                R, z, 1.0, 0.0, 1.0, nsigma=nsigma, mc=mc, nmc=nmc, **kwargs
            ) / self._vmomentdensity(
                R, z, 0.0, 0.0, 0.0, nsigma=nsigma, mc=mc, nmc=nmc, **kwargs
            )

    @potential_physical_input
    @physical_conversion("angle", pop=True)
    def tilt(
        self, R, z, nsigma=None, mc=False, nmc=10000, gl=True, ngl=_DEFAULTNGL, **kwargs
    ):
        """
        Calculate the tilt of the velocity ellipsoid by marginalizing over velocity.

        Parameters
        ----------
        R : float or Quantity
            Radius at which to calculate this.
        z : float or Quantity
            Height at which to calculate this.
        nsigma : int, optional
            Number of sigma to integrate the velocities over.
        mc : bool, optional
            If True, calculate using Monte Carlo integration.
        nmc : int, optional
            If mc, use nmc samples.
        gl : bool, optional
            If True, calculate using Gauss-Legendre integration.
        ngl : int, optional
            If gl, use ngl-th order Gauss-Legendre integration for each dimension.

        Returns
        -------
        float
            Tilt in radians.

        Notes
        -----
        - 2012-12-23 - Written - Bovy (IAS)
        - 2017-10-28 - Changed return unit to rad - Bovy (UofT)
        """
        if mc:
            surfmass, vrs, vts, vzs = self._vmomentdensity(
                R,
                z,
                0.0,
                0.0,
                0.0,
                nsigma=nsigma,
                mc=mc,
                nmc=nmc,
                _returnmc=True,
                **kwargs,
            )
            tsigmar2 = (
                self._vmomentdensity(
                    R,
                    z,
                    2.0,
                    0.0,
                    0.0,
                    nsigma=nsigma,
                    mc=mc,
                    nmc=nmc,
                    _returnmc=False,
                    _vrs=vrs,
                    _vts=vts,
                    _vzs=vzs,
                    **kwargs,
                )
                / surfmass
            )
            tsigmaz2 = (
                self._vmomentdensity(
                    R,
                    z,
                    0.0,
                    0.0,
                    2.0,
                    nsigma=nsigma,
                    mc=mc,
                    nmc=nmc,
                    _returnmc=False,
                    _vrs=vrs,
                    _vts=vts,
                    _vzs=vzs,
                    **kwargs,
                )
                / surfmass
            )
            tsigmarz = (
                self._vmomentdensity(
                    R,
                    z,
                    1.0,
                    0.0,
                    1.0,
                    nsigma=nsigma,
                    mc=mc,
                    nmc=nmc,
                    _returnmc=False,
                    _vrs=vrs,
                    _vts=vts,
                    _vzs=vzs,
                    **kwargs,
                )
                / surfmass
            )
            xp = get_namespace(tsigmarz, tsigmar2, tsigmaz2)
            return 0.5 * xp.arctan(2.0 * tsigmarz / (tsigmar2 - tsigmaz2))
        elif gl:
            surfmass, glqeval = self._vmomentdensity(
                R, z, 0.0, 0.0, 0.0, gl=gl, ngl=ngl, _returngl=True, **kwargs
            )
            tsigmar2 = (
                self._vmomentdensity(
                    R, z, 2.0, 0.0, 0.0, ngl=ngl, gl=gl, _glqeval=glqeval, **kwargs
                )
                / surfmass
            )
            tsigmaz2 = (
                self._vmomentdensity(
                    R, z, 0.0, 0.0, 2.0, ngl=ngl, gl=gl, _glqeval=glqeval, **kwargs
                )
                / surfmass
            )
            tsigmarz = (
                self._vmomentdensity(
                    R, z, 1.0, 0.0, 1.0, ngl=ngl, gl=gl, _glqeval=glqeval, **kwargs
                )
                / surfmass
            )
            xp = get_namespace(tsigmarz, tsigmar2, tsigmaz2)
            return 0.5 * xp.arctan(2.0 * tsigmarz / (tsigmar2 - tsigmaz2))
        else:
            raise NotImplementedError("Use either mc=True or gl=True")

    @potential_physical_input
    @physical_conversion("velocity2", pop=True)
    def sigmaz2(
        self, R, z, nsigma=None, mc=False, nmc=10000, gl=True, ngl=_DEFAULTNGL, **kwargs
    ):
        """
        Calculate sigma_z^2 by marginalizing over velocity.

        Parameters
        ----------
        R : float or Quantity
            Radius at which to calculate this.
        z : float or Quantity
            Height at which to calculate this.
        nsigma : int, optional
            Number of sigma to integrate the velocities over.
        mc : bool, optional
            If True, calculate using Monte Carlo integration.
        nmc : int, optional
            If mc, use nmc samples.
        gl : bool, optional
            If True, calculate using Gauss-Legendre integration.
        ngl : int, optional
            If gl, use ngl-th order Gauss-Legendre integration for each dimension.
        **kwargs : dict, optional
            scipy.integrate.tplquad kwargs epsabs and epsrel.

        Returns
        -------
        float
            sigma_z^2.

        Notes
        -----
        - 2012-07-30 - Written - Bovy (IAS@MPIA)

        """
        if mc:
            surfmass, vrs, vts, vzs = self._vmomentdensity(
                R,
                z,
                0.0,
                0.0,
                0.0,
                nsigma=nsigma,
                mc=mc,
                nmc=nmc,
                _returnmc=True,
                **kwargs,
            )
            return (
                self._vmomentdensity(
                    R,
                    z,
                    0.0,
                    0.0,
                    2.0,
                    nsigma=nsigma,
                    mc=mc,
                    nmc=nmc,
                    _returnmc=False,
                    _vrs=vrs,
                    _vts=vts,
                    _vzs=vzs,
                    **kwargs,
                )
                / surfmass
            )
        elif gl:
            surfmass, glqeval = self._vmomentdensity(
                R, z, 0.0, 0.0, 0.0, gl=gl, ngl=ngl, _returngl=True, **kwargs
            )
            return (
                self._vmomentdensity(
                    R, z, 0.0, 0.0, 2.0, ngl=ngl, gl=gl, _glqeval=glqeval, **kwargs
                )
                / surfmass
            )
        else:  # pragma: no cover because this is too slow; a warning is shown
            return self._vmomentdensity(
                R, z, 0.0, 0.0, 2.0, nsigma=nsigma, mc=mc, nmc=nmc, **kwargs
            ) / self._vmomentdensity(
                R, z, 0.0, 0.0, 0.0, nsigma=nsigma, mc=mc, nmc=nmc, **kwargs
            )

    @potential_physical_input
    @physical_conversion("velocity", pop=True)
    def meanvT(
        self, R, z, nsigma=None, mc=False, nmc=10000, gl=True, ngl=_DEFAULTNGL, **kwargs
    ):
        """
        Calculate the mean rotational velocity by marginalizing over velocity.

        Parameters
        ----------
        R : float or Quantity
            Radius at which to calculate this.
        z : float or Quantity
            Height at which to calculate this.
        nsigma : float, optional
            Number of sigma to integrate the velocities over.
        mc : bool, optional
            If True, calculate using Monte Carlo integration.
        nmc : int, optional
            If mc, use nmc samples.
        gl : bool, optional
            If True, calculate using Gauss-Legendre integration.
        ngl : int, optional
            If gl, use ngl-th order Gauss-Legendre integration for each dimension.
        **kwargs : dict, optional
            scipy.integrate.tplquad kwargs epsabs and epsrel.

        Returns
        -------
        float
            Mean rotational velocity.

        Notes
        -----
        - 2012-07-30 - Written - Bovy (IAS@MPIA)

        """
        if mc:
            surfmass, vrs, vts, vzs = self._vmomentdensity(
                R,
                z,
                0.0,
                0.0,
                0.0,
                nsigma=nsigma,
                mc=mc,
                nmc=nmc,
                _returnmc=True,
                **kwargs,
            )
            return (
                self._vmomentdensity(
                    R,
                    z,
                    0.0,
                    1.0,
                    0.0,
                    nsigma=nsigma,
                    mc=mc,
                    nmc=nmc,
                    _returnmc=False,
                    _vrs=vrs,
                    _vts=vts,
                    _vzs=vzs,
                    **kwargs,
                )
                / surfmass
            )
        elif gl:
            surfmass, glqeval = self._vmomentdensity(
                R, z, 0.0, 0.0, 0.0, gl=gl, ngl=ngl, _returngl=True, **kwargs
            )
            return (
                self._vmomentdensity(
                    R, z, 0.0, 1.0, 0.0, ngl=ngl, gl=gl, _glqeval=glqeval, **kwargs
                )
                / surfmass
            )
        else:  # pragma: no cover because this is too slow; a warning is shown
            return self._vmomentdensity(
                R, z, 0.0, 1.0, 0.0, nsigma=nsigma, mc=mc, nmc=nmc, **kwargs
            ) / self._vmomentdensity(
                R, z, 0.0, 0.0, 0.0, nsigma=nsigma, mc=mc, nmc=nmc, **kwargs
            )

    @potential_physical_input
    @physical_conversion("velocity", pop=True)
    def meanvR(
        self, R, z, nsigma=None, mc=False, nmc=10000, gl=True, ngl=_DEFAULTNGL, **kwargs
    ):
        """
        Calculate the mean radial velocity by marginalizing over velocity.

        Parameters
        ----------
        R : float or Quantity
            Radius at which to calculate this.
        z : float or Quantity
            Height at which to calculate this.
        nsigma : float, optional
            Number of sigma to integrate the velocities over.
        mc : bool, optional
            If True, calculate using Monte Carlo integration.
        nmc : int, optional
            If mc, use nmc samples.
        gl : bool, optional
            If True, calculate using Gauss-Legendre integration.
        ngl : int, optional
            If gl, use ngl-th order Gauss-Legendre integration for each dimension.
        **kwargs : dict, optional
            scipy.integrate.tplquad kwargs epsabs and epsrel.

        Returns
        -------
        float
            Mean radial velocity.

        Notes
        -----
        - 2012-12-23 - Written - Bovy (IAS)

        """
        if mc:
            surfmass, vrs, vts, vzs = self._vmomentdensity(
                R,
                z,
                0.0,
                0.0,
                0.0,
                nsigma=nsigma,
                mc=mc,
                nmc=nmc,
                _returnmc=True,
                **kwargs,
            )
            return (
                self._vmomentdensity(
                    R,
                    z,
                    1.0,
                    0.0,
                    0.0,
                    nsigma=nsigma,
                    mc=mc,
                    nmc=nmc,
                    _returnmc=False,
                    _vrs=vrs,
                    _vts=vts,
                    _vzs=vzs,
                    **kwargs,
                )
                / surfmass
            )
        elif gl:
            surfmass, glqeval = self._vmomentdensity(
                R, z, 0.0, 0.0, 0.0, gl=gl, ngl=ngl, _returngl=True, **kwargs
            )
            return (
                self._vmomentdensity(
                    R, z, 1.0, 0.0, 0.0, ngl=ngl, gl=gl, _glqeval=glqeval, **kwargs
                )
                / surfmass
            )
        else:  # pragma: no cover because this is too slow; a warning is shown
            return self._vmomentdensity(
                R, z, 1.0, 0.0, 0.0, nsigma=nsigma, mc=mc, nmc=nmc, **kwargs
            ) / self._vmomentdensity(
                R, z, 0.0, 0.0, 0.0, nsigma=nsigma, mc=mc, nmc=nmc, **kwargs
            )

    @potential_physical_input
    @physical_conversion("velocity", pop=True)
    def meanvz(
        self, R, z, nsigma=None, mc=False, nmc=10000, gl=True, ngl=_DEFAULTNGL, **kwargs
    ):
        """
        Calculate the mean vertical velocity by marginalizing over velocity.

        Parameters
        ----------
        R : float or Quantity
            Radius at which to calculate this.
        z : float or Quantity
            Height at which to calculate this.
        nsigma : float, optional
            Number of sigma to integrate the velocities over.
        mc : bool, optional
            If True, calculate using Monte Carlo integration.
        nmc : int, optional
            If mc, use nmc samples.
        gl : bool, optional
            If True, calculate using Gauss-Legendre integration.
        ngl : int, optional
            If gl, use ngl-th order Gauss-Legendre integration for each dimension.
        **kwargs : dict, optional
            scipy.integrate.tplquad kwargs epsabs and epsrel.

        Returns
        -------
        float
            Mean vertical velocity

        Notes
        -----
        - 2012-12-23 - Written - Bovy (IAS)
        """
        if mc:
            surfmass, vrs, vts, vzs = self._vmomentdensity(
                R,
                z,
                0.0,
                0.0,
                0.0,
                nsigma=nsigma,
                mc=mc,
                nmc=nmc,
                _returnmc=True,
                **kwargs,
            )
            return (
                self._vmomentdensity(
                    R,
                    z,
                    0.0,
                    0.0,
                    1.0,
                    nsigma=nsigma,
                    mc=mc,
                    nmc=nmc,
                    _returnmc=False,
                    _vrs=vrs,
                    _vts=vts,
                    _vzs=vzs,
                    **kwargs,
                )
                / surfmass
            )
        elif gl:
            surfmass, glqeval = self._vmomentdensity(
                R, z, 0.0, 0.0, 0.0, gl=gl, ngl=ngl, _returngl=True, **kwargs
            )
            return (
                self._vmomentdensity(
                    R, z, 0.0, 0.0, 1.0, ngl=ngl, gl=gl, _glqeval=glqeval, **kwargs
                )
                / surfmass
            )
        else:  # pragma: no cover because this is too slow; a warning is shown
            return self._vmomentdensity(
                R, z, 0.0, 0.0, 1.0, nsigma=nsigma, mc=mc, nmc=nmc, **kwargs
            ) / self._vmomentdensity(
                R, z, 0.0, 0.0, 0.0, nsigma=nsigma, mc=mc, nmc=nmc, **kwargs
            )

    @potential_physical_input
    @physical_conversion("velocity2", pop=True)
    def sigmaT2(
        self, R, z, nsigma=None, mc=False, nmc=10000, gl=True, ngl=_DEFAULTNGL, **kwargs
    ):
        """
        Calculate sigma_T^2 by marginalizing over velocity.

        Parameters
        ----------
        R : float or Quantity
            Radius at which to calculate this.
        z : float or Quantity
            Height at which to calculate this.
        nsigma : int, optional
            Number of sigma to integrate the velocities over.
        mc : bool, optional
            If True, calculate using Monte Carlo integration.
        nmc : int, optional
            If mc is True, use nmc samples.
        gl : bool, optional
            If True, calculate using Gauss-Legendre integration.
        ngl : int, optional
            If gl is True, use ngl-th order Gauss-Legendre integration for each dimension.
        **kwargs
            scipy.integrate.tplquad kwargs epsabs and epsrel.

        Returns
        -------
        float
            sigma_T^2.

        Notes
        -----
        - 2012-07-30 - Written - Bovy (IAS@MPIA)

        """
        if mc:
            surfmass, vrs, vts, vzs = self._vmomentdensity(
                R,
                z,
                0.0,
                0.0,
                0.0,
                nsigma=nsigma,
                mc=mc,
                nmc=nmc,
                _returnmc=True,
                **kwargs,
            )
            mvt = (
                self._vmomentdensity(
                    R,
                    z,
                    0.0,
                    1.0,
                    0.0,
                    nsigma=nsigma,
                    mc=mc,
                    nmc=nmc,
                    _returnmc=False,
                    _vrs=vrs,
                    _vts=vts,
                    _vzs=vzs,
                    **kwargs,
                )
                / surfmass
            )
            return (
                self._vmomentdensity(
                    R,
                    z,
                    0.0,
                    2.0,
                    0.0,
                    nsigma=nsigma,
                    mc=mc,
                    nmc=nmc,
                    _returnmc=False,
                    _vrs=vrs,
                    _vts=vts,
                    _vzs=vzs,
                    **kwargs,
                )
                / surfmass
                - mvt**2.0
            )
        elif gl:
            surfmass, glqeval = self._vmomentdensity(
                R, z, 0.0, 0.0, 0.0, gl=gl, ngl=ngl, _returngl=True, **kwargs
            )
            mvt = (
                self._vmomentdensity(
                    R, z, 0.0, 1.0, 0.0, ngl=ngl, gl=gl, _glqeval=glqeval, **kwargs
                )
                / surfmass
            )
            return (
                self._vmomentdensity(
                    R, z, 0.0, 2.0, 0.0, ngl=ngl, gl=gl, _glqeval=glqeval, **kwargs
                )
                / surfmass
                - mvt**2.0
            )

        else:  # pragma: no cover because this is too slow; a warning is shown
            surfmass = self._vmomentdensity(
                R, z, 0.0, 0.0, 0.0, nsigma=nsigma, mc=mc, nmc=nmc, **kwargs
            )
            return (
                self._vmomentdensity(
                    R, z, 0.0, 2.0, 0.0, nsigma=nsigma, mc=mc, nmc=nmc, **kwargs
                )
                / surfmass
                - (
                    self._vmomentdensity(
                        R, z, 0.0, 2.0, 0.0, nsigma=nsigma, mc=mc, nmc=nmc, **kwargs
                    )
                    / surfmass
                )
                ** 2.0
            )

    @potential_physical_input
    @physical_conversion("action", pop=True)
    def meanjr(self, R, z, nsigma=None, mc=True, nmc=10000, **kwargs):
        """
        Calculate the mean radial action by marginalizing over velocity

        Parameters
        ----------
        R : float or Quantity
            Radius at which to calculate this
        z : float or Quantity
            Height at which to calculate this
        nsigma : float, optional
            Number of sigma to integrate the velocities over
        mc : bool, optional
            If True, calculate using Monte Carlo integration
        nmc : int, optional
            If mc, use nmc samples
        **kwargs : dict
            scipy.integrate.tplquad kwargs epsabs and epsrel

        Returns
        -------
        float
            Mean jr

        Notes
        -----
        - 2012-08-09 - Written - Bovy (IAS@MPIA)

        """
        if mc:
            surfmass, vrs, vts, vzs = self._vmomentdensity(
                R,
                z,
                0.0,
                0.0,
                0.0,
                nsigma=nsigma,
                mc=mc,
                nmc=nmc,
                _returnmc=True,
                **kwargs,
            )
            return (
                self._jmomentdensity(
                    R,
                    z,
                    1.0,
                    0.0,
                    0.0,
                    nsigma=nsigma,
                    mc=mc,
                    nmc=nmc,
                    _returnmc=False,
                    _vrs=vrs,
                    _vts=vts,
                    _vzs=vzs,
                    **kwargs,
                )
                / surfmass
            )
        else:  # pragma: no cover because this is too slow; a warning is shown
            return self._jmomentdensity(
                R, z, 1.0, 0.0, 0.0, nsigma=nsigma, mc=mc, nmc=nmc, **kwargs
            ) / self._vmomentdensity(
                R, z, 0.0, 0.0, 0.0, nsigma=nsigma, mc=mc, nmc=nmc, **kwargs
            )

    @potential_physical_input
    @physical_conversion("action", pop=True)
    def meanlz(self, R, z, nsigma=None, mc=True, nmc=10000, **kwargs):
        """
        Calculate the mean angular momentum by marginalizing over velocity.

        Parameters
        ----------
        R : float or Quantity
            Radius at which to calculate this.
        z : float or Quantity
            Height at which to calculate this.
        nsigma : float, optional
            Number of sigma to integrate the velocities over.
        mc : bool, optional
            If True, calculate using Monte Carlo integration.
        nmc : int, optional
            If mc, use nmc samples.
        **kwargs
            scipy.integrate.tplquad kwargs epsabs and epsrel.

        Returns
        -------
        float
            Mean angular momentum.

        Notes
        -----
        - 2012-08-09 - Written - Bovy (IAS@MPIA)

        """

        if mc:
            surfmass, vrs, vts, vzs = self._vmomentdensity(
                R,
                z,
                0.0,
                0.0,
                0.0,
                nsigma=nsigma,
                mc=mc,
                nmc=nmc,
                _returnmc=True,
                **kwargs,
            )
            return (
                self._jmomentdensity(
                    R,
                    z,
                    0.0,
                    1.0,
                    0.0,
                    nsigma=nsigma,
                    mc=mc,
                    nmc=nmc,
                    _returnmc=False,
                    _vrs=vrs,
                    _vts=vts,
                    _vzs=vzs,
                    **kwargs,
                )
                / surfmass
            )
        else:  # pragma: no cover because this is too slow; a warning is shown
            return self._jmomentdensity(
                R, z, 0.0, 1.0, 0.0, nsigma=nsigma, mc=mc, nmc=nmc, **kwargs
            ) / self._vmomentdensity(
                R, z, 0.0, 0.0, 0.0, nsigma=nsigma, mc=mc, nmc=nmc, **kwargs
            )

    @potential_physical_input
    @physical_conversion("action", pop=True)
    def meanjz(self, R, z, nsigma=None, mc=True, nmc=10000, **kwargs):
        """
        Calculate the mean vertical action by marginalizing over velocity.

        Parameters
        ----------
        R : float or Quantity
            Radius at which to calculate this.
        z : float or Quantity
            Height at which to calculate this.
        nsigma : float, optional
            Number of sigma to integrate the velocities over.
        mc : bool, optional
            If True, calculate using Monte Carlo integration.
        nmc : int, optional
            If mc, use nmc samples.
        **kwargs : dict
            scipy.integrate.tplquad kwargs epsabs and epsrel.

        Returns
        -------
        float
            Mean jz.

        Notes
        -----
        - 2012-08-09 - Written - Bovy (IAS@MPIA)

        """
        if mc:
            surfmass, vrs, vts, vzs = self._vmomentdensity(
                R,
                z,
                0.0,
                0.0,
                0.0,
                nsigma=nsigma,
                mc=mc,
                nmc=nmc,
                _returnmc=True,
                **kwargs,
            )
            return (
                self._jmomentdensity(
                    R,
                    z,
                    0.0,
                    0.0,
                    1.0,
                    nsigma=nsigma,
                    mc=mc,
                    nmc=nmc,
                    _returnmc=False,
                    _vrs=vrs,
                    _vts=vts,
                    _vzs=vzs,
                    **kwargs,
                )
                / surfmass
            )
        else:  # pragma: no cover because this is too slow; a warning is shown
            return self._jmomentdensity(
                R, z, 0.0, 0.0, 1.0, nsigma=nsigma, mc=mc, nmc=nmc, **kwargs
            ) / self._vmomentdensity(
                R, z, 0.0, 0.0, 0.0, nsigma=nsigma, mc=mc, nmc=nmc, **kwargs
            )

    @potential_physical_input
    def sampleV(self, R, z, n=1, **kwargs):
        """
        Sample a radial, azimuthal, and vertical velocity at R,z

        Parameters
        ----------
        R : float or Quantity
            Galactocentric distance.
        z : float or Quantity
            Height.
        n : int, optional
            Number of distances to sample.

        Returns
        -------
        list
            List of samples.

        Notes
        -----
        - 2012-12-17 - Written - Bovy (IAS@MPIA)
        """
        use_physical = kwargs.pop("use_physical", True)
        vo = kwargs.pop("vo", None)
        if vo is None and hasattr(self, "_voSet") and self._voSet:
            vo = self._vo
        vo = parse_velocity_kms(vo)
        # backend-native inverse-CDF sampler; as_numpy the draws to keep sampleV's
        # numpy-output contract (the CDF build + inversion already ran on the backend)
        out = as_numpy(self._sampleV_icdf(R, z, n, get_namespace()))
        if use_physical and not vo is None:
            if _APY_UNITS:
                return units.Quantity(out * vo, unit=units.km / units.s)
            else:
                return out * vo
        else:
            return out

    def _sampleV_icdf(self, R, z, n, xp, nsigma=5.0, nvT=60, nvR=60, nvz=80):
        """Sample n (vR, vT, vz) at one (R, z) by inverse-CDF (backend-native).

        The quasi-isothermal DF factorises, so p(vR,vT,vz|R,z) is drawn by the
        chain vT -> vR|vT -> vz|vR,vT. ONE 3-D velocity mesh feeds all three: the
        vT marginal (integrate over vR,vz), p(vR,vT) (integrate over vz) and
        p(vz|vR,vT) (the mesh). CDFs are cumulative trapezoids inverted piecewise-
        linearly (no rejection loop), reproducing every marginal AND the tilt.
        Single path (numpy included): replaces the old fmin_powell + numpy.random
        rejection sampler, which was pathologically slow under a forced backend.
        """
        from ..backend.sampling import (
            batched_inverse_cdf_sample,
            linear_inverse_cdf_sample,
        )

        def _seg(p, g, axis):
            d = g[1] - g[0]
            lo = [slice(None)] * p.ndim
            lo[axis] = slice(0, -1)
            hi = [slice(None)] * p.ndim
            hi[axis] = slice(1, None)
            return 0.5 * (p[tuple(lo)] + p[tuple(hi)]) * d

        def _trapz(p, g, axis):
            return xp.sum(_seg(p, g, axis), axis=axis)

        def _cumcdf(p, g, axis):
            c = xp.cumulative_sum(_seg(p, g, axis), axis=axis)
            zsh = list(p.shape)
            zsh[axis] = 1
            c = xp.concatenate([xp.zeros(tuple(zsh), dtype=c.dtype), c], axis=axis)
            last = [slice(None)] * p.ndim
            last[axis] = slice(c.shape[axis] - 1, c.shape[axis])
            return c / c[tuple(last)]

        # local dispersions are plain scalars (numpy.exp, not xp.exp): they set the
        # velocity-grid extents and torch.linspace needs scalar (not tensor) limits
        sigmaR1 = self._sr * numpy.exp((self._refr - R) / self._hsr)
        sigmaz1 = self._sz * numpy.exp((self._refr - R) / self._hsz)
        vTg = xp.linspace(0.0, 1.8, nvT)
        vRg = xp.linspace(-nsigma * sigmaR1, nsigma * sigmaR1, nvR)
        vzg = xp.linspace(-nsigma * sigmaz1, nsigma * sigmaz1, nvz)
        VT, VR, VZ = xp.meshgrid(vTg, vRg, vzg, indexing="ij")
        base = xp.reshape(VR, (-1,)) * 0.0
        mesh = xp.reshape(
            xp.exp(
                self(
                    R + base,
                    xp.reshape(VR, (-1,)),
                    xp.reshape(VT, (-1,)),
                    z + base,
                    xp.reshape(VZ, (-1,)),
                    log=True,
                    use_physical=False,
                )
            ),
            (nvT, nvR, nvz),
        )
        FvT = _cumcdf(_trapz(_trapz(mesh, vzg, 2), vRg, 1), vTg, 0)
        FvR = _cumcdf(_trapz(mesh, vzg, 2), vRg, 1)
        Fvz = _cumcdf(mesh, vzg, 2)
        u = xp.asarray(numpy.random.random((3, n)))
        vT = linear_inverse_cdf_sample(xp, vTg, FvT, u[0])
        iT = xp.clip(xp.searchsorted(vTg, vT) - 1, 0, nvT - 2)
        wT = xp.reshape((vT - vTg[iT]) / (vTg[1] - vTg[0]), (n, 1))
        vR = batched_inverse_cdf_sample(
            xp, vRg, (1.0 - wT) * FvR[iT] + wT * FvR[iT + 1], u[1]
        )
        iR = xp.clip(xp.searchsorted(vRg, vR) - 1, 0, nvR - 2)
        wR = xp.reshape((vR - vRg[iR]) / (vRg[1] - vRg[0]), (n, 1))
        Fvzj = (1.0 - wT) * ((1.0 - wR) * Fvz[iT, iR] + wR * Fvz[iT, iR + 1]) + wT * (
            (1.0 - wR) * Fvz[iT + 1, iR] + wR * Fvz[iT + 1, iR + 1]
        )
        vz = batched_inverse_cdf_sample(xp, vzg, Fvzj, u[2])
        return xp.stack([vR, vT, vz], axis=1)

    @potential_physical_input
    def sampleV_interpolate(
        self,
        R,
        z,
        R_pixel,
        z_pixel,
        num_std=3,
        R_min=None,
        R_max=None,
        z_max=None,
        **kwargs,
    ):
        """
        Sample radial, azimuthal, and vertical velocity at R,z using interpolation.

        Parameters
        ----------
        R : numpy.ndarray or Quantity
            Galactocentric distance.
        z : numpy.ndarray or Quantity
            Height.
        R_pixel : float
            The pixel size for creating the grid for interpolation (in natural units).
        z_pixel : float
            The pixel size for creating the grid for interpolation (in natural units).
        num_std : float, optional
            Number of standard deviation to be considered outliers sampled separately from interpolation.
        R_min : float, optional
            Minimum R value for the grid.
        R_max : float, optional
            Maximum R value for the grid.
        z_max : float, optional
            Maximum z value for the grid.

        Returns
        -------
        numpy.ndarray
            A numpy array containing the sampled velocity, (vR, vT, vz), where each row corresponds to the row of (R,z).

        Notes
        -----
        - 2018-08-10 - Written - Samuel Wong (University of Toronto)
        """
        use_physical = kwargs.pop("use_physical", True)
        vo = kwargs.pop("vo", None)
        if vo is None and hasattr(self, "_voSet") and self._voSet:
            vo = self._vo
        vo = parse_velocity_kms(vo)
        # Initialize output array
        coord_v = numpy.empty((numpy.size(R), 3))
        # Since the sign of z doesn't matter, work with absolute value of z
        z = numpy.abs(z)
        # Grid edges
        if R_min is None:
            R_min = numpy.amax([numpy.mean(R) - num_std * numpy.std(R), numpy.amin(R)])
        if R_max is None:
            R_max = numpy.amin([numpy.mean(R) + num_std * numpy.std(R), numpy.amax(R)])
        if z_max is None:
            z_max = numpy.amin([numpy.mean(z) + num_std * numpy.std(z), numpy.amax(z)])
        z_min = 0.0  # Always start grid at z=0 for stars close to plane
        # Separate the coordinates into outliers and normal points
        # Define outliers as points outside of grid
        mask = numpy.any([R < R_min, R > R_max, z > z_max], axis=0)
        outliers_R = R[mask]
        outliers_z = z[mask]
        normal_R = R[~mask]
        normal_z = z[~mask]
        # Sample the velocity of outliers directly (without interpolation)
        outlier_coord_v = numpy.empty((outliers_R.size, 3))
        for i in range(outliers_R.size):
            outlier_coord_v[i] = self.sampleV(
                outliers_R[i], outliers_z[i], use_physical=False
            )[0]
        # Prepare for optimizing maxVT on a grid
        # Get the new hash of the parameters of grid
        new_hash = hashlib.md5(
            numpy.array([R_min, R_max, z_max, R_pixel, z_pixel])
        ).hexdigest()
        # Reuse old interpolated object if new hash matches the old one
        if new_hash == self._maxVT_hash:
            ip_max_vT = self._maxVT_ip
        # Generate a new interpolation object if different from before
        else:
            R_number = int((R_max - R_min) / R_pixel)
            z_number = int((z_max - z_min) / z_pixel)
            R_linspace = numpy.linspace(R_min, R_max, R_number)
            z_linspace = numpy.linspace(z_min, z_max, z_number)
            Rv, zv = numpy.meshgrid(R_linspace, z_linspace)
            grid = numpy.dstack((Rv, zv))  # This grid stores (R,z) coordinate
            # Grid is a 3 dimensional array since it stores pairs of values, but
            # grid max vT is a 2 dimensional array
            grid_max_vT = numpy.empty((grid.shape[0], grid.shape[1]))
            # Optimize max_vT on the grid
            for i in range(z_number):
                for j in range(R_number):
                    R, z = grid[i][j]
                    grid_max_vT[i][j] = numpy.squeeze(
                        optimize.fmin_powell(
                            (
                                lambda x: (
                                    -as_numpy(
                                        self(
                                            R,
                                            0.0,
                                            x,
                                            z,
                                            0.0,
                                            log=True,
                                            use_physical=False,
                                        )
                                    )
                                )
                            ),
                            1.0,
                        )
                    )
            # Determine degree of interpolation
            ky = numpy.min([R_number - 1, 3])
            kx = numpy.min([z_number - 1, 3])
            # Generate interpolation object
            ip_max_vT = interpolate.RectBivariateSpline(
                z_linspace, R_linspace, grid_max_vT, kx=kx, ky=ky
            )
            # Store interpolation object
            self._maxVT_ip = ip_max_vT
            # Update hash of parameters
            self._maxVT_hash = new_hash
        # Evaluate interpolation object to get maxVT at the normal coordinates
        normal_max_vT = ip_max_vT.ev(normal_z, normal_R)
        # Sample all 3 velocities at a normal point and use interpolated vT
        normal_coord_v = self._sampleV_preoptimized(normal_R, normal_z, normal_max_vT)
        # Combine normal and outlier result, preserving original order
        coord_v[mask] = outlier_coord_v
        coord_v[~mask] = normal_coord_v
        if use_physical and not vo is None:
            if _APY_UNITS:
                return units.Quantity(coord_v * vo, unit=units.km / units.s)
            else:
                return coord_v * vo
        else:
            return coord_v

    def _sampleV_preoptimized(self, R, z, maxVT):
        """
        Sample a radial, azimuthal, and vertical velocity at R,z.

        Parameters
        ----------
        R : float or numpy.ndarray
            Galactocentric distance.
        z : float or numpy.ndarray
            Height.
        maxVT : numpy.ndarray
            An array of pre-optimized maximum vT at corresponding R,z.

        Returns
        -------
        numpy.ndarray
            A numpy array containing the sampled velocity, (vR, vT, vz), where each row correspond to the row of (R,z).

        Notes
        -----
        - 2018-08-10 - Written - Samuel Wong (University of Toronto)

        """
        length = numpy.size(R)
        out = numpy.empty((length, 3))  # Initialize output
        # Determine the maximum of the velocity distribution
        maxVR = numpy.zeros(length)
        maxVz = numpy.zeros(length)
        # as_numpy: fmin_powell's optimum is fed straight into the numpy
        # rejection arithmetic below; under a forced backend self() hands back a
        # backend scalar here too. No-op on numpy.
        logmaxVD = as_numpy(
            self(R, maxVR, maxVT, z, maxVz, log=True, use_physical=False)
        )
        # Now rejection-sample
        # Initialize boolean index of position remaining to be sampled
        remain_indx = numpy.full(length, True)
        while numpy.any(remain_indx):
            nmore = numpy.sum(remain_indx)
            propvR = numpy.random.normal(size=nmore) * 2.0 * self._sr
            propvT = (
                numpy.random.normal(size=nmore) * 2.0 * self._sr + maxVT[remain_indx]
            )
            propvz = numpy.random.normal(size=nmore) * 2.0 * self._sz
            # as_numpy for the same reason as in sampleV above
            VDatprop = (
                as_numpy(
                    self(
                        R[remain_indx],
                        propvR,
                        propvT,
                        z[remain_indx],
                        propvz,
                        log=True,
                        use_physical=False,
                    )
                )
                - logmaxVD[remain_indx]
            )
            VDatprop -= -0.5 * (
                propvR**2.0 / 4.0 / self._sr**2.0
                + propvz**2.0 / 4.0 / self._sz**2.0
                + (propvT - maxVT[remain_indx]) ** 2.0 / 4.0 / self._sr**2.0
            )
            accept_indx = VDatprop > numpy.log(numpy.random.random(size=nmore))
            vR_accept = propvR[accept_indx]
            vT_accept = propvT[accept_indx]
            vz_accept = propvz[accept_indx]
            # Get the indexing of rows of output array that need to be updated
            # with newly accepted velocity
            to_change = numpy.copy(remain_indx)
            to_change[remain_indx] = accept_indx
            out[to_change] = numpy.stack((vR_accept, vT_accept, vz_accept), axis=1)
            # Removing accepted sampled from remain index
            remain_indx[remain_indx] = ~accept_indx
        return out

    @actionAngle_physical_input
    @physical_conversion("phasespacedensityvelocity2", pop=True)
    def pvR(self, vR, R, z, gl=True, ngl=_DEFAULTNGL2, nsigma=4.0, vTmax=1.5):
        """
        Calculate the marginalized vR probability at this location (NOT normalized by the density).

        Parameters
        ----------
        vR : float or Quantity
            Radial velocity.
        R : float or Quantity
            Radius.
        z : float or Quantity
            Height.
        gl : bool, optional
            If True, use Gauss-Legendre integration.
        ngl : int, optional
            If gl, use ngl-th order Gauss-Legendre integration for each dimension.
        nsigma : float, optional
            Number of sigma to integrate the velocities over.
        vTmax : float, optional
            Sets integration limits to [0,vTmax] for integration over vT.

        Returns
        -------
        float
            p(vR,R,z).

        Notes
        -----
        - 2012-12-22 - Written - Bovy (IAS@MPIA)

        """
        xp = get_namespace(vR, R, z)
        if xp is not numpy:
            vR, R, z = promote_scalars(xp, vR, R, z)
        sigmaz1 = self._sz * xp.exp((self._refr - R) / self._hsz)
        if gl:
            if ngl % 2 == 1:
                raise ValueError("ngl must be even")
            # Use Gauss-Legendre integration for all
            if ngl == _DEFAULTNGL:
                glx, glw = self._glxdef, self._glwdef
                glx12, glw12 = self._glxdef12, self._glwdef12
            elif ngl == _DEFAULTNGL2:
                glx, glw = self._glxdef2, self._glwdef2
                glx12, glw12 = self._glxdef, self._glwdef
            else:
                glx, glw = numpy.polynomial.legendre.leggauss(ngl)
                glx12, glw12 = numpy.polynomial.legendre.leggauss(ngl // 2)
            if xp is not numpy:  # promote the GL node/weight tables to the backend
                glx, glw = xp.asarray(glx) * 1.0, xp.asarray(glw) * 1.0
                glx12, glw12 = xp.asarray(glx12) * 1.0, xp.asarray(glw12) * 1.0
            # Evaluate everywhere
            if isinstance(
                self._aA,
                (
                    actionAngle.actionAngleAdiabatic,
                    actionAngle.actionAngleAdiabaticGrid,
                ),
            ):
                vzgl = nsigma * sigmaz1 / 2.0 * (glx + 1.0)
                vzglw = glw
                vzfac = nsigma * sigmaz1  # 2 x integration over [0,nsigma*sigmaz1]
            else:
                vzgl = xp.concatenate(
                    [
                        nsigma * sigmaz1 / 2.0 * (glx12 + 1.0),
                        -nsigma * sigmaz1 / 2.0 * (glx12 + 1.0),
                    ]
                )
                vzglw = xp.concatenate([glw12, glw12])
                vzfac = (
                    0.5 * nsigma * sigmaz1
                )  # integration over [-nsigma*sigmaz1,0] and [0,nsigma*sigmaz1]
            vTgl = vTmax / 2.0 * (glx + 1.0)
            vTfac = 0.5 * vTmax  # integration over [0.,vTmax]
            # Tile everything
            vTgl = xp.tile(vTgl, (ngl, 1)).T
            vzgl = xp.tile(vzgl, (ngl, 1))
            vTglw = xp.tile(glw, (ngl, 1)).T  # also tile weights
            vzglw = xp.tile(vzglw, (ngl, 1))
            # evaluate
            logqeval = xp.reshape(
                self(
                    R + xp.zeros(ngl * ngl),
                    vR + xp.zeros(ngl * ngl),
                    vTgl.flatten(),
                    z + xp.zeros(ngl * ngl),
                    vzgl.flatten(),
                    log=True,
                    use_physical=False,
                ),
                (ngl, ngl),
            )
            return xp.sum(xp.exp(logqeval) * vTglw * vzglw * vzfac) * vTfac

    @actionAngle_physical_input
    @physical_conversion("phasespacedensityvelocity2", pop=True)
    def pvT(self, vT, R, z, gl=True, ngl=_DEFAULTNGL2, nsigma=4.0):
        """
        Calculate the marginalized vT probability at this location (NOT normalized by the density).

        Parameters
        ----------
        vT : float or Quantity
            Azimuthal velocity.
        R : float or Quantity
            Radius.
        z : float or Quantity
            Height.
        gl : bool, optional
            If True, use Gauss-Legendre integration.
        ngl : int, optional
            If gl, use ngl-th order Gauss-Legendre integration for each dimension.
        nsigma : float, optional
            Number of sigma to integrate the velocities over.

        Returns
        -------
        float
            p(vT,R,z).

        Notes
        -----
        - 2012-12-22 - Written - Bovy (IAS@MPIA)
        - 2018-01-12 - Added Gauss-Legendre integration prefactor nsigma^2/4 - Trick (MPA)

        """
        xp = get_namespace(vT, R, z)
        if xp is not numpy:
            vT, R, z = promote_scalars(xp, vT, R, z)
        sigmaR1 = self._sr * xp.exp((self._refr - R) / self._hsr)
        sigmaz1 = self._sz * xp.exp((self._refr - R) / self._hsz)
        if gl:
            if ngl % 2 == 1:
                raise ValueError("ngl must be even")
            # Use Gauss-Legendre integration for all
            if ngl == _DEFAULTNGL:
                glx, glw = self._glxdef, self._glwdef
                glx12, glw12 = self._glxdef12, self._glwdef12
            elif ngl == _DEFAULTNGL2:
                glx, glw = self._glxdef2, self._glwdef2
                glx12, glw12 = self._glxdef, self._glwdef
            else:
                glx, glw = numpy.polynomial.legendre.leggauss(ngl)
                glx12, glw12 = numpy.polynomial.legendre.leggauss(ngl // 2)
            if xp is not numpy:  # promote the GL node/weight tables to the backend
                glx, glw = xp.asarray(glx) * 1.0, xp.asarray(glw) * 1.0
                glx12, glw12 = xp.asarray(glx12) * 1.0, xp.asarray(glw12) * 1.0
            # Evaluate everywhere
            if isinstance(
                self._aA,
                (
                    actionAngle.actionAngleAdiabatic,
                    actionAngle.actionAngleAdiabaticGrid,
                ),
            ):
                vRgl = nsigma * sigmaR1 / 2.0 * (glx + 1.0)
                vzgl = nsigma * sigmaz1 / 2.0 * (glx + 1.0)
                vRglw = glw
                vzglw = glw
                vRfac = nsigma * sigmaR1  # 2 x integration over [0,nsigma*sigmaR1]
                vzfac = nsigma * sigmaz1  # 2 x integration over [0,nsigma*sigmaz1]
            else:
                vRgl = xp.concatenate(
                    [
                        nsigma * sigmaR1 / 2.0 * (glx12 + 1.0),
                        -nsigma * sigmaR1 / 2.0 * (glx12 + 1.0),
                    ]
                )
                vzgl = xp.concatenate(
                    [
                        nsigma * sigmaz1 / 2.0 * (glx12 + 1.0),
                        -nsigma * sigmaz1 / 2.0 * (glx12 + 1.0),
                    ]
                )
                vRglw = xp.concatenate([glw12, glw12])
                vzglw = xp.concatenate([glw12, glw12])
                vRfac = (
                    0.5 * nsigma * sigmaR1
                )  # integration over [-nsigma*sigmaR1,0] and [0,nsigma*sigmaR1]
                vzfac = (
                    0.5 * nsigma * sigmaz1
                )  # integration over [-nsigma*sigmaz1,0] and [0,nsigma*sigmaz1]
            # Tile everything
            vRgl = xp.tile(vRgl, (ngl, 1)).T
            vzgl = xp.tile(vzgl, (ngl, 1))
            vRglw = xp.tile(vRglw, (ngl, 1)).T  # also tile weights
            vzglw = xp.tile(vzglw, (ngl, 1))
            # evaluate
            logqeval = xp.reshape(
                self(
                    R + xp.zeros(ngl * ngl),
                    vRgl.flatten(),
                    vT + xp.zeros(ngl * ngl),
                    z + xp.zeros(ngl * ngl),
                    vzgl.flatten(),
                    log=True,
                    use_physical=False,
                ),
                (ngl, ngl),
            )
            return xp.sum(xp.exp(logqeval) * vRglw * vzglw * vRfac * vzfac)

    @actionAngle_physical_input
    @physical_conversion("phasespacedensityvelocity2", pop=True)
    def pvz(
        self,
        vz,
        R,
        z,
        gl=True,
        ngl=_DEFAULTNGL2,
        nsigma=4.0,
        vTmax=1.5,
        _return_actions=False,
        _jr=None,
        _lz=None,
        _jz=None,
        _return_freqs=False,
        _rg=None,
        _kappa=None,
        _nu=None,
        _Omega=None,
        _sigmaR1=None,
    ):
        """
        Calculate the marginalized vz probability at this location (NOT normalized by the density).

        Parameters
        ----------
        vz : float or Quantity
            Vertical velocity.
        R : float or Quantity
            Radius.
        z : float or Quantity
            Height.
        gl : bool, optional
            If True, use Gauss-Legendre integration.
        ngl : int, optional
            If gl, use ngl-th order Gauss-Legendre integration for each dimension.
        nsigma : float, optional
            Number of sigma to integrate the velocities over.
        vTmax : float, optional
            Sets integration limits to [0,vTmax] for integration over vT.

        Returns
        -------
        float
            p(vz,R,z).

        Notes
        -----
        - 2012-12-22 - Written - Bovy (IAS)
        """
        xp = get_namespace(vz, R, z)
        if xp is not numpy:
            # promote inputs (scalars or numpy arrays) to the backend so the GL
            # grid arithmetic below runs on tensors (numpy path: no-op).
            vz, R, z = promote_scalars(xp, vz, R, z)
        if _sigmaR1 is None:
            sigmaR1 = self._sr * xp.exp((self._refr - R) / self._hsr)
        else:
            sigmaR1 = _sigmaR1
        if gl:
            if ngl % 2 == 1:
                raise ValueError("ngl must be even")
            # Use Gauss-Legendre integration for all
            if ngl == _DEFAULTNGL:
                glx, glw = self._glxdef, self._glwdef
                glx12, glw12 = self._glxdef12, self._glwdef12
            elif ngl == _DEFAULTNGL2:
                glx, glw = self._glxdef2, self._glwdef2
                glx12, glw12 = self._glxdef, self._glwdef
            else:
                glx, glw = numpy.polynomial.legendre.leggauss(ngl)
                glx12, glw12 = numpy.polynomial.legendre.leggauss(ngl // 2)
            if xp is not numpy:  # promote the GL node/weight tables to the backend
                glx, glw = xp.asarray(glx) * 1.0, xp.asarray(glw) * 1.0
                glx12, glw12 = xp.asarray(glx12) * 1.0, xp.asarray(glw12) * 1.0
            # Evaluate everywhere
            if isinstance(
                self._aA,
                (
                    actionAngle.actionAngleAdiabatic,
                    actionAngle.actionAngleAdiabaticGrid,
                ),
            ):
                vRgl = glx + 1.0
                vRglw = glw
                vRfac = nsigma * sigmaR1  # 2 x integration over [0,nsigma*sigmaR1]
            else:
                vRgl = xp.concatenate([glx12 + 1.0, -(glx12 + 1.0)])
                vRglw = xp.concatenate([glw12, glw12])
                vRfac = (
                    0.5 * nsigma * sigmaR1
                )  # integration over [-nsigma*sigmaR1,0] and [0,nsigma*sigmaR1]
            vTgl = vTmax / 2.0 * (glx + 1.0)
            vTfac = 0.5 * vTmax  # integration over [0.,vTmax]
            # Tile everything
            vTgl = xp.tile(vTgl, (ngl, 1)).T
            vRgl = xp.tile(vRgl, (ngl, 1))
            vTglw = xp.tile(glw, (ngl, 1)).T  # also tile weights
            vRglw = xp.tile(vRglw, (ngl, 1))
            # If inputs are arrays, tile (permute_dims not 3-D .T: torch -W errors)
            if getattr(R, "ndim", 0) > 0:
                nR = len(R)
                R = xp.permute_dims(xp.tile(R, (ngl, ngl, 1)), (2, 1, 0)).flatten()
                z = xp.permute_dims(xp.tile(z, (ngl, ngl, 1)), (2, 1, 0)).flatten()
                vz = xp.permute_dims(xp.tile(vz, (ngl, ngl, 1)), (2, 1, 0)).flatten()
                vTgl = xp.tile(vTgl, (nR, 1, 1)).flatten()
                vRgl = xp.tile(vRgl, (nR, 1, 1)).flatten()
                vTglw = xp.tile(vTglw, (nR, 1, 1))
                vRglw = xp.tile(vRglw, (nR, 1, 1))
                scalarOut = False
            else:
                R = R + xp.zeros(ngl * ngl)
                z = z + xp.zeros(ngl * ngl)
                vz = vz + xp.zeros(ngl * ngl)
                nR = 1
                scalarOut = True
                vRgl = vRgl.flatten()
            vRgl = (
                vRgl
                * xp.permute_dims(
                    xp.tile(nsigma * sigmaR1 / 2.0, (ngl, ngl, 1)), (2, 1, 0)
                ).flatten()
            )
            # evaluate
            if _jr is None and _rg is None:
                logqeval, jr, lz, jz, rg, kappa, nu, Omega = self(
                    R,
                    vRgl.flatten(),
                    vTgl.flatten(),
                    z,
                    vz,
                    log=True,
                    _return_actions=True,
                    _return_freqs=True,
                    use_physical=False,
                )
                logqeval = xp.reshape(logqeval, (nR, ngl * ngl))
            elif not _jr is None and not _rg is None:
                logqeval, jr, lz, jz, rg, kappa, nu, Omega = self(
                    (_jr, _lz, _jz),
                    rg=_rg,
                    kappa=_kappa,
                    nu=_nu,
                    Omega=_Omega,
                    log=True,
                    _return_actions=True,
                    _return_freqs=True,
                    use_physical=False,
                )
                logqeval = xp.reshape(logqeval, (nR, ngl * ngl))
            elif not _jr is None and _rg is None:
                logqeval, jr, lz, jz, rg, kappa, nu, Omega = self(
                    (_jr, _lz, _jz),
                    log=True,
                    _return_actions=True,
                    _return_freqs=True,
                    use_physical=False,
                )
                logqeval = xp.reshape(logqeval, (nR, ngl * ngl))
            elif _jr is None and not _rg is None:
                logqeval, jr, lz, jz, rg, kappa, nu, Omega = self(
                    R,
                    vRgl.flatten(),
                    vTgl.flatten(),
                    z,
                    vz,
                    rg=_rg,
                    kappa=_kappa,
                    nu=_nu,
                    Omega=_Omega,
                    log=True,
                    _return_actions=True,
                    _return_freqs=True,
                    use_physical=False,
                )
                logqeval = xp.reshape(logqeval, (nR, ngl * ngl))
            vRglw = xp.reshape(vRglw, (nR, ngl * ngl))
            vTglw = xp.reshape(vTglw, (nR, ngl * ngl))
            if scalarOut:
                result = (
                    xp.sum(xp.exp(logqeval) * vTglw * vRglw, axis=1)[0] * vRfac * vTfac
                )
            else:
                result = (
                    xp.sum(xp.exp(logqeval) * vTglw * vRglw, axis=1) * vRfac * vTfac
                )
            if _return_actions and _return_freqs:
                return (result, jr, lz, jz, rg, kappa, nu, Omega)
            elif _return_freqs:
                return (result, rg, kappa, nu, Omega)
            elif _return_actions:
                return (result, jr, lz, jz)
            else:
                return result

    @actionAngle_physical_input
    @physical_conversion("phasespacedensityvelocity", pop=True)
    def pvRvT(self, vR, vT, R, z, gl=True, ngl=_DEFAULTNGL2, nsigma=4.0):
        """
        Calculate the marginalized (vR,vT) probability at this location (NOT normalized by the density).

        Parameters
        ----------
        vR : float or Quantity
            Radial velocity.
        vT : float or Quantity
            Azimuthal velocity.
        R : float or Quantity
            Radius.
        z : float or Quantity
            Height.
        gl : bool, optional
            If True, use Gauss-Legendre integration.
        ngl : int, optional
            If gl, use ngl-th order Gauss-Legendre integration for each dimension.
        nsigma : float, optional
            Number of sigma to integrate the velocities over.

        Returns
        -------
        float
            p(vR,vT,R,z).

        Notes
        -----
        - 2012-12-22 - Written - Bovy (IAS)
        - 2018-01-12 - Added Gauss-Legendre integration prefactor nsigma/2 - Trick (MPA)
        """
        xp = get_namespace(vR, vT, R, z)
        if xp is not numpy:
            vR, vT, R, z = promote_scalars(xp, vR, vT, R, z)
        sigmaz1 = self._sz * xp.exp((self._refr - R) / self._hsz)
        if gl:
            if ngl % 2 == 1:
                raise ValueError("ngl must be even")
            # Use Gauss-Legendre integration for all
            if ngl == _DEFAULTNGL:
                glx, glw = self._glxdef, self._glwdef
                glx12, glw12 = self._glxdef12, self._glwdef12
            elif ngl == _DEFAULTNGL2:
                glx, glw = self._glxdef2, self._glwdef2
                glx12, glw12 = self._glxdef, self._glwdef
            else:
                glx, glw = numpy.polynomial.legendre.leggauss(ngl)
                glx12, glw12 = numpy.polynomial.legendre.leggauss(ngl // 2)
            if xp is not numpy:  # promote the GL node/weight tables to the backend
                glx, glw = xp.asarray(glx) * 1.0, xp.asarray(glw) * 1.0
                glx12, glw12 = xp.asarray(glx12) * 1.0, xp.asarray(glw12) * 1.0
            # Evaluate everywhere
            if isinstance(
                self._aA,
                (
                    actionAngle.actionAngleAdiabatic,
                    actionAngle.actionAngleAdiabaticGrid,
                ),
            ):
                vzgl = nsigma * sigmaz1 / 2.0 * (glx + 1.0)
                vzglw = glw
                vzfac = nsigma * sigmaz1  # 2 x integration over [0,nsigma*sigmaz1]
            else:
                vzgl = xp.concatenate(
                    [
                        nsigma * sigmaz1 / 2.0 * (glx12 + 1.0),
                        -nsigma * sigmaz1 / 2.0 * (glx12 + 1.0),
                    ]
                )
                vzglw = xp.concatenate([glw12, glw12])
                vzfac = (
                    0.5 * nsigma * sigmaz1
                )  # integration over [-nsigma*sigmaz1,0] and [0,nsigma*sigmaz1]
            # evaluate
            logqeval = self(
                R + xp.zeros(ngl),
                vR + xp.zeros(ngl),
                vT + xp.zeros(ngl),
                z + xp.zeros(ngl),
                vzgl,
                log=True,
                use_physical=False,
            )
            return xp.sum(xp.exp(logqeval) * vzglw * vzfac)

    @actionAngle_physical_input
    @physical_conversion("phasespacedensityvelocity", pop=True)
    def pvTvz(self, vT, vz, R, z, gl=True, ngl=_DEFAULTNGL2, nsigma=4.0):
        """
        Calculate the marginalized (vT,vz) probability at this location (NOT normalized by the density).

        Parameters
        ----------
        vT : float or Quantity
            Azimuthal velocity.
        vz : float or Quantity
            Vertical velocity.
        R : float or Quantity
            Radius.
        z : float or Quantity
            Height.
        gl : bool, optional
            If True, use Gauss-Legendre integration.
        ngl : int, optional
            If gl, use ngl-th order Gauss-Legendre integration for each dimension.
        nsigma : float, optional
            Number of sigma to integrate the velocities over.

        Returns
        -------
        float or Quantity
            p(vT,vz,R,z).

        Notes
        -----
        - 2012-12-22 - Written - Bovy (IAS)
        - 2018-01-12 - Added Gauss-Legendre integration prefactor nsigma/2 - Trick (MPA)

        """
        xp = get_namespace(vT, vz, R, z)
        if xp is not numpy:
            vT, vz, R, z = promote_scalars(xp, vT, vz, R, z)
        sigmaR1 = self._sr * xp.exp((self._refr - R) / self._hsr)
        if gl:
            if ngl % 2 == 1:
                raise ValueError("ngl must be even")
            # Use Gauss-Legendre integration for all
            if ngl == _DEFAULTNGL:
                glx, glw = self._glxdef, self._glwdef
                glx12, glw12 = self._glxdef12, self._glwdef12
            elif ngl == _DEFAULTNGL2:
                glx, glw = self._glxdef2, self._glwdef2
                glx12, glw12 = self._glxdef, self._glwdef
            else:
                glx, glw = numpy.polynomial.legendre.leggauss(ngl)
                glx12, glw12 = numpy.polynomial.legendre.leggauss(ngl // 2)
            if xp is not numpy:  # promote the GL node/weight tables to the backend
                glx, glw = xp.asarray(glx) * 1.0, xp.asarray(glw) * 1.0
                glx12, glw12 = xp.asarray(glx12) * 1.0, xp.asarray(glw12) * 1.0
            # Evaluate everywhere
            if isinstance(
                self._aA,
                (
                    actionAngle.actionAngleAdiabatic,
                    actionAngle.actionAngleAdiabaticGrid,
                ),
            ):
                vRgl = nsigma * sigmaR1 / 2.0 * (glx + 1.0)
                vRglw = glw
                vRfac = nsigma * sigmaR1  # 2 x integration over [0,nsigma*sigmaR1]
            else:
                vRgl = xp.concatenate(
                    [
                        nsigma * sigmaR1 / 2.0 * (glx12 + 1.0),
                        -nsigma * sigmaR1 / 2.0 * (glx12 + 1.0),
                    ]
                )
                vRglw = xp.concatenate([glw12, glw12])
                vRfac = (
                    0.5 * nsigma * sigmaR1
                )  # integration over [-nsigma*sigmaR1,0] and [0,nsigma*sigmaR1]
            # evaluate
            logqeval = self(
                R + xp.zeros(ngl),
                vRgl,
                vT + xp.zeros(ngl),
                z + xp.zeros(ngl),
                vz + xp.zeros(ngl),
                log=True,
                use_physical=False,
            )
            return xp.sum(xp.exp(logqeval) * vRglw * vRfac)

    @actionAngle_physical_input
    @physical_conversion("phasespacedensityvelocity", pop=True)
    def pvRvz(self, vR, vz, R, z, gl=True, ngl=_DEFAULTNGL2, vTmax=1.5):
        """
        Calculate the marginalized (vR,vz) probability at this location (NOT normalized by the density).

        Parameters
        ----------
        vR : float or Quantity
            Radial velocity.
        vz : float or Quantity
            Vertical velocity.
        R : float or Quantity
            Radius.
        z : float or Quantity
            Height.
        gl : bool, optional
            If True, use Gauss-Legendre integration.
        ngl : int, optional
            If gl, use ngl-th order Gauss-Legendre integration for each dimension.
        vTmax : float, optional
            Sets integration limits to [0,vTmax] for integration over vT.

        Returns
        -------
        float or Quantity
            p(vR,vz,R,z).

        Notes
        -----
        - 2013-01-02 - Written - Bovy (IAS)
        - 2018-01-12 - Added Gauss-Legendre integration prefactor vTmax/2 - Trick (MPA)
        """
        xp = get_namespace(vR, vz, R, z)
        if xp is not numpy:
            vR, vz, R, z = promote_scalars(xp, vR, vz, R, z)
        if gl:
            if ngl % 2 == 1:
                raise ValueError("ngl must be even")
            # Use Gauss-Legendre integration for all
            if ngl == _DEFAULTNGL:
                glx, glw = self._glxdef, self._glwdef
                glx12, glw12 = self._glxdef12, self._glwdef12
            elif ngl == _DEFAULTNGL2:
                glx, glw = self._glxdef2, self._glwdef2
                glx12, glw12 = self._glxdef, self._glwdef
            else:
                glx, glw = numpy.polynomial.legendre.leggauss(ngl)
                glx12, glw12 = numpy.polynomial.legendre.leggauss(ngl // 2)
            if xp is not numpy:  # promote the GL node/weight tables to the backend
                glx, glw = xp.asarray(glx) * 1.0, xp.asarray(glw) * 1.0
                glx12, glw12 = xp.asarray(glx12) * 1.0, xp.asarray(glw12) * 1.0
            # Evaluate everywhere
            vTgl = vTmax / 2.0 * (glx + 1.0)
            vTglw = glw
            vTfac = 0.5 * vTmax  # integration over [0.,vTmax]
            # If inputs are arrays, tile
            if getattr(R, "ndim", 0) > 0:
                nR = len(R)
                R = xp.tile(R, (ngl, 1)).T.flatten()
                z = xp.tile(z, (ngl, 1)).T.flatten()
                vR = xp.tile(vR, (ngl, 1)).T.flatten()
                vz = xp.tile(vz, (ngl, 1)).T.flatten()
                vTgl = xp.tile(vTgl, (nR, 1)).flatten()
                vTglw = xp.tile(vTglw, (nR, 1))
                scalarOut = False
            else:
                R = R + xp.zeros(ngl)
                vR = vR + xp.zeros(ngl)
                z = z + xp.zeros(ngl)
                vz = vz + xp.zeros(ngl)
                nR = 1
                scalarOut = True
            # evaluate
            logqeval = xp.reshape(
                self(R, vR, vTgl, z, vz, log=True, use_physical=False), (nR, ngl)
            )
            out = xp.sum(xp.exp(logqeval) * vTglw * vTfac, axis=1)
            if scalarOut:
                return out[0]
            else:
                return out

    def _calc_epifreq(self, r):
        """
        Calculate the epicycle frequency at r.

        Parameters
        ----------
        r : float
            Radius.

        Returns
        -------
        float
            Epicycle frequency.

        Notes
        -----
        - 2012-07-25 - Written - Bovy (IAS@MPIA)
        """
        return potential.epifreq(self._pot, r)

    def _calc_verticalfreq(self, r):
        """
        Calculate the vertical frequency at r.

        Parameters
        ----------
        r : float
            Radius.

        Returns
        -------
        float
            Vertical frequency.

        Notes
        -----
        - 2012-07-25 - Written - Bovy (IAS@MPIA)
        """
        return potential.verticalfreq(self._pot, r)

    def _rg(self, lz):
        """
        Calculate the radius of a circular orbit of Lz.

        Parameters
        ----------
        lz : float
            Angular momentum.

        Returns
        -------
        float
            Radius.

        Notes
        -----
        - 2012-07-25 - Written - Bovy (IAS@MPIA)
        """
        if is_backend_array(lz):  # leaf data-guard: numpy callers stay numpy
            if self._rgInterpBackend is None:  # _precomputerg=False: rl everywhere
                return potential.rl(self._pot, lz)
            # _precomputerg=True: spline everywhere. This mirrors the numpy-array
            # path, whose out-of-range rl branch is dead for a valid Lz grid
            # (indx = (lz>Lzmax)&(lz<Lzmin) is empty when Lzmin<Lzmax), so it too
            # only ever splines an array. AD-safe (no root-find), and it avoids the
            # eager full-array rl solve an xp.where(indx, rl, spline) would run.
            return self._rgInterpBackend(lz)
        if isinstance(lz, numpy.ndarray):
            indx = (lz > self._precomputergLzmax) * (lz < self._precomputergLzmin)
            indxc = True ^ indx
            out = numpy.empty(lz.shape)
            out[indxc] = self._rgInterp(lz[indxc])
            out[indx] = numpy.array(
                [potential.rl(self._pot, lz[indx][ii]) for ii in range(numpy.sum(indx))]
            )
            return out
        else:
            if lz > self._precomputergLzmax or lz < self._precomputergLzmin:
                return potential.rl(self._pot, lz)
            return numpy.atleast_1d(self._rgInterp(lz))


def _vmomentsurfaceIntegrand(
    vz, vR, vT, R, z, df, sigmaR1, gamma, sigmaz1, n, m, o
):  # pragma: no cover because this is too slow; a warning is shown
    """Internal function that is the integrand for the vmomentsurface mass integration"""
    return (
        vR**n
        * vT**m
        * vz**o
        * df(R, vR * sigmaR1, vT * sigmaR1 * gamma, z, vz * sigmaz1, use_physical=False)
    )


def _vmomentsurfaceMCIntegrand(
    vz, vR, vT, R, z, df, sigmaR1, gamma, sigmaz1, mvT, n, m, o
):
    """Internal function that is the integrand for the vmomentsurface mass integration"""
    xp = get_namespace(vz, vR, vT, R, z)
    return (
        vR**n
        * vT**m
        * vz**o
        * df(R, vR * sigmaR1, vT * sigmaR1 * gamma, z, vz * sigmaz1, use_physical=False)
        * xp.exp(vR**2.0 / 2.0 + (vT - mvT) ** 2.0 / 2.0 + vz**2.0 / 2.0)
    )


def _jmomentsurfaceIntegrand(
    vz, vR, vT, R, z, df, sigmaR1, gamma, sigmaz1, n, m, o
):  # pragma: no cover because this is too slow; a warning is shown
    """Internal function that is the integrand for the vmomentsurface mass integration"""
    return df(
        R,
        vR * sigmaR1,
        vT * sigmaR1 * gamma,
        z,
        vz * sigmaz1,
        use_physical=False,
        func=(lambda x, y, z: x**n * y**m * z**o),
    )


def _jmomentsurfaceMCIntegrand(
    vz, vR, vT, R, z, df, sigmaR1, gamma, sigmaz1, mvT, n, m, o
):
    """Internal function that is the integrand for the vmomentsurface mass integration"""
    xp = get_namespace(vz, vR, vT, R, z)
    return df(
        R,
        vR * sigmaR1,
        vT * sigmaR1 * gamma,
        z,
        vz * sigmaz1,
        use_physical=False,
        func=(lambda x, y, z: x**n * y**m * z**o),
    ) * xp.exp(vR**2.0 / 2.0 + (vT - mvT) ** 2.0 / 2.0 + vz**2.0 / 2.0)
