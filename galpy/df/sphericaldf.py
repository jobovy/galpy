# Superclass for spherical distribution functions, contains
#   - sphericaldf: superclass of all spherical DFs
#   - isotropicsphericaldf: superclass of all isotropic spherical DFs
#   - anisotropicsphericaldf: superclass of all anisotropic spherical DFs
#
# To implement a new DF do something like:
#   - Inherit from isotropicsphericaldf for an isotropic DF and implement
#     fE(self,E) which returns the DF as a function of E (see kingdf), then
#     you should be set! You may also have to implement _vmax_at_r(self,pot,r)
#     when the maximum velocity at a given position is less than the escape
#     velocity
#   - Inherit from anisotropicsphericaldf for an anisotropic DF, then you need
#     to implement a bunch of functions:
#       * _call_internal(self,*args,**kwargs): which returns the DF as a
#                                              function of (E,L,Lz)
#       * _sample_eta(self,r,n=1): to sample the velocity angle at r
#       * _p_v_at_r(self,v,r): which returns p(v|r)
#     constantbetadf is an example of this
#
import warnings

import numpy
import scipy.interpolate
from scipy import integrate, special

from ..backend import (
    as_numpy,
    exit_cast,
    get_namespace,
    is_backend_array,
    resolve_namespace,
)
from ..backend.interpolate import Spline1D
from ..backend.quadrature import fixed_quad, nested_quad
from ..orbit import Orbit
from ..potential import (
    CompositePotential,
    KeplerPotential,
    PowerSphericalPotential,
    interpSphericalPotential,
    mass,
)
from ..potential.Potential import (
    _check_potential_list_and_deprecate,
    _evaluatePotentials,
)
from ..potential.SCFPotential import _RToxi, _xiToR
from ..util import _optional_deps, conversion, galpyWarning
from ..util.conversion import physical_conversion, potential_physical_input
from .df import df

# Use _APY_LOADED/_APY_UNITS like this to be able to change them in tests
if _optional_deps._APY_LOADED:
    from astropy import units

# Fixed backend (jax/torch) Gauss-Legendre orders; chosen so the backend path
# matches the adaptive-scipy numpy path to <~1e-9 over the physical range
# (measured in tests/test_backend_sphericaldf.py).
_QUAD_N_VMOM = 100  # velocity-moment integral over v
_QUAD_N_VMOM2D = 60  # (v, eta) tensor product in the anisotropic base
_QUAD_N_DMDE = 100  # dM/dE radius integral


def _handle_rmin(rmin, pot, denspot, scale, ro, df_name):
    """
    Determine the minimum radius for sampling and check if potential diverges.

    For potentials that diverge at r=0 (Phi(0) = -inf), we need a finite rmin
    to define the energy range for sampling.

    Parameters
    ----------
    rmin : float, Quantity, or None
        User-specified minimum radius, or None for auto-detection
    pot : Potential instance or a combined potential formed using addition (pot1+pot2+…)
        The gravitational potential
    denspot : Potential instance or a combined potential formed using addition (pot1+pot2+…)
        The density potential (tracer population)
    scale : float
        Characteristic scale radius
    ro : float
        Distance scale for unit conversion
    df_name : str
        Name of the DF class (for error/warning messages)

    Returns
    -------
    rmin : float
        The rmin value to use (in internal units)
    """
    # Check if potential diverges at r=0
    xp = get_namespace()  # context/forced default only (inputs are scalars)
    if xp is numpy:
        phi_at_zero = _evaluatePotentials(pot, 0.0, 0)
    else:
        # coerce coords: undecorated potential evals reject scalars (torch)
        phi_at_zero = as_numpy(_evaluatePotentials(pot, xp.asarray(0.0), 0))
    is_divergent = not numpy.isfinite(phi_at_zero)

    # If rmin is explicitly specified, use it
    if rmin is not None:
        return conversion.parse_length(rmin, ro=ro)

    # Check all potentials for known problematic types
    for p in denspot:
        # Check for KeplerPotential (point mass - no distributed density)
        if isinstance(p, KeplerPotential):
            raise ValueError(
                f"{df_name} cannot sample from KeplerPotential directly because it "
                "represents a point mass with no distributed density."
            )

        # Check for PowerSphericalPotential
        if isinstance(p, PowerSphericalPotential):
            alpha = p.alpha
            if alpha >= 3.0:
                raise ValueError(
                    f"{df_name} cannot sample from PowerSphericalPotential with "
                    f"alpha={alpha} >= 3."
                )
            elif alpha > 2.0:
                # Divergent potential - auto-set rmin
                auto_rmin = 1e-6 * scale
                warnings.warn(
                    f"PowerSphericalPotential with alpha={alpha} diverges at r=0. "
                    f"Using rmin={auto_rmin:.2e} as minimum radius. "
                    "Set rmin explicitly to suppress this warning.",
                    galpyWarning,
                )
                return auto_rmin

    # Check for other divergent potentials
    if is_divergent:
        auto_rmin = 1e-6 * scale
        warnings.warn(
            f"Potential diverges at r=0 (Phi(0)={phi_at_zero}). "
            f"Using rmin={auto_rmin:.2e} as minimum radius. "
            "Set rmin explicitly to suppress this warning.",
            galpyWarning,
        )
        return auto_rmin

    # Non-divergent potential - use rmin = 0
    return 0.0


def _input_scales(obj, kwargs):
    """The ro and vo to use for parsing Quantity *inputs*: a per-call ro=/vo=
    if one is given, the DF's own otherwise. Same precedence as
    potential_physical_input, which is why the methods using this are decorated
    with pop=False: ro= and vo= have to still be in kwargs when the body runs.
    """
    ro = conversion.parse_length_kpc(kwargs.get("ro", None))
    vo = conversion.parse_velocity_kms(kwargs.get("vo", None))
    return obj._ro if ro is None else ro, obj._vo if vo is None else vo


class sphericaldf(df):
    """Superclass for spherical distribution functions"""

    def __init__(self, pot=None, denspot=None, rmax=None, scale=None, ro=None, vo=None):
        """
        Initializes a spherical DF

        Parameters
        ----------
        pot : Potential instance or a combined potential formed using addition (pot1+pot2+…)
            The potential. Default is None.
        denspot : Potential instance or a combined potential formed using addition (pot1+pot2+…), optional
            The potential that represents the density of the tracers (assumed to be spherical). If None, set equal to pot. Default is None.
        rmax : float or Quantity, optional
            The maximum radius to consider. DF is cut off at E = Phi(rmax). Default is None.
        scale : float or Quantity, optional
            The length-scale parameter to be used internally. Default is None.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2020-07-22 - Written - Lane (UofT)
        """
        df.__init__(self, ro=ro, vo=vo)
        if not conversion.physical_compatible(self, pot):
            raise RuntimeError(
                "Unit-conversion parameters of input potential incompatible with those of the DF instance"
            )
        phys = conversion.get_physical(pot, include_set=True)
        # if pot has physical units, transfer them (if already on, we know
        # they are compatible)
        if phys["roSet"] and phys["voSet"]:
            self.turn_physical_on(ro=phys["ro"], vo=phys["vo"])
        if pot is None:  # pragma: no cover
            raise OSError("pot= must be set")
        self._pot = _check_potential_list_and_deprecate(pot)
        self._denspot = (
            self._pot
            if denspot is None
            else _check_potential_list_and_deprecate(denspot)
        )
        if not conversion.physical_compatible(self._pot, self._denspot):
            raise RuntimeError(
                "Unit-conversion parameters of input potential incompatible with those of the density potential"
            )
        self._rmax = (
            numpy.inf if rmax is None else conversion.parse_length(rmax, ro=self._ro)
        )
        try:
            self._scale = pot._scale
        except AttributeError:
            try:
                self._scale = pot[0]._scale
            except (TypeError, AttributeError):
                self._scale = (
                    conversion.parse_length(scale, ro=self._ro)
                    if scale is not None
                    else 1.0
                )
        # Check that interpolated potential has appropriate grid range for DF
        if isinstance(pot, interpSphericalPotential) and pot._rmax < self._rmax:
            warnings.warn(
                "The interpolated potential's rmax is smaller than the DF's rmax",
                galpyWarning,
            )

    ############################## EVALUATING THE DF###############################
    @physical_conversion("massphasespacedensity", pop=False)
    def __call__(self, *args, **kwargs):
        """
        Evaluate the DF

        Parameters
        ----------
        *args: tuple
            Either:
                a) (E,L,Lz): tuple of E and (optionally) L and (optionally) Lz. Each may be Quantity
                b) R,vR,vT,z,vz,phi: cylindrical coordinates (can be Quantity)
                c) Orbit instance: orbit.Orbit instance and if specific time then orbit.Orbit(t)

        Returns
        -------
        ndarray or Quantity
            Value of DF

        Notes
        -----
        - 2020-07-22 - Written - Lane (UofT)
        - 2024-10-29 - Fixed to return mass/phase-space volume units for physical-unit output - Bovy (UofT)
        - 2026-08-20 - A per-call ro=/vo= is now also used to parse Quantity inputs - Bovy (UofT)
        """
        ro, vo = _input_scales(self, kwargs)
        # Get E,L,Lz
        if len(args) == 1:
            if not isinstance(args[0], Orbit):  # Assume tuple (E,L,Lz)
                E, L, Lz = (args[0] + (None, None))[:3]
            else:  # Orbit
                E = args[0].E(pot=self._pot, use_physical=False)
                Lval = args[0].L(use_physical=False)
                if is_backend_array(Lval):  # forced backend: |L| in-namespace
                    xp = get_namespace(Lval)
                    L = xp.sqrt(xp.sum(Lval**2.0))
                else:
                    L = numpy.sqrt(numpy.sum(Lval**2.0))
                Lz = args[0].Lz(use_physical=False)
            E = conversion.parse_energy(E, vo=vo)
            L = conversion.parse_angmom(L, ro=ro, vo=vo)
            Lz = conversion.parse_angmom(Lz, ro=ro, vo=vo)
            _inp = (E, L, Lz)
            xp = resolve_namespace(E, L, Lz)
            if xp is numpy:
                E = numpy.atleast_1d(E)
                L = numpy.atleast_1d(L)
                Lz = numpy.atleast_1d(Lz)
            else:
                # asarray first (torch.atleast_1d rejects Python scalars); an
                # unspecified L/Lz stays None (only the anisotropic DFs use them)
                E = xp.atleast_1d(xp.asarray(E))
                L = L if L is None else xp.atleast_1d(xp.asarray(L))
                Lz = Lz if Lz is None else xp.atleast_1d(xp.asarray(Lz))
        else:  # Assume R,vR,vT,z,vz,(phi)
            R, vR, vT, z, vz, phi = (args + (None,))[:6]
            R = conversion.parse_length(R, ro=ro)
            vR = conversion.parse_velocity(vR, vo=vo)
            vT = conversion.parse_velocity(vT, vo=vo)
            z = conversion.parse_length(z, ro=ro)
            vz = conversion.parse_velocity(vz, vo=vo)
            _inp = (R, vR, vT, z, vz)
            xp = resolve_namespace(R, vR, vT, z, vz)
            if xp is numpy:
                vtotSq = vR**2.0 + vT**2.0 + vz**2.0
                E = numpy.atleast_1d(
                    0.5 * vtotSq + _evaluatePotentials(self._pot, R, z)
                )
                Lz = numpy.atleast_1d(R * vT)
                r = numpy.sqrt(R**2.0 + z**2.0)
                vrad = (R * vR + z * vz) / r
                L = numpy.atleast_1d(numpy.sqrt(vtotSq - vrad**2.0) * r)
            else:
                # bring possibly-scalar coords into the namespace first (torch)
                R, vR, vT, z, vz = (xp.asarray(c) * 1.0 for c in (R, vR, vT, z, vz))
                vtotSq = vR**2.0 + vT**2.0 + vz**2.0
                E = xp.atleast_1d(0.5 * vtotSq + _evaluatePotentials(self._pot, R, z))
                Lz = xp.atleast_1d(R * vT)
                r = xp.sqrt(R**2.0 + z**2.0)
                vrad = (R * vR + z * vz) / r
                L = xp.atleast_1d(xp.sqrt(vtotSq - vrad**2.0) * r)
        return exit_cast(
            self._call_internal(E, L, Lz).reshape(
                args[0].shape
                if len(args) == 1 and hasattr(args[0], "shape")
                else (
                    args[0][0].shape
                    if len(args) == 1
                    and hasattr(args[0], "__len__")
                    and hasattr(args[0][0], "shape")
                    else (args[0].shape if hasattr(args[0], "shape") else ())
                )
            ),
            *_inp,
        )

    @physical_conversion("massenergydensity", pop=False)
    def dMdE(self, E, **kwargs):
        """
        Compute the differential energy distribution dM/dE: the amount of mass per unit energy

        Parameters
        ----------
        E : float or numpy.ndarray
            Energy; can be a Quantity

        Returns
        -------
        float, numpy.ndarray, or Quantity
            The differential energy distribution

        Notes
        -----
        - 2023-05-23 - Written - Bovy (UofT)
        - 2026-08-20 - A per-call vo= is now also used to parse a Quantity input - Bovy (UofT)

        """
        _, vo = _input_scales(self, kwargs)
        Ei = conversion.parse_energy(E, vo=vo)
        xp = resolve_namespace(Ei)
        if xp is numpy:
            return self._dMdE(numpy.atleast_1d(Ei)).reshape(
                E.shape if isinstance(E, numpy.ndarray) else ()
            )
        return exit_cast(
            self._dMdE(xp.atleast_1d(xp.asarray(Ei))).reshape(
                Ei.shape if hasattr(Ei, "shape") else ()
            ),
            Ei,
        )

    @potential_physical_input
    def vmomentdensity(self, r, n, m, **kwargs):
        """
        Calculate an arbitrary moment of the velocity distribution at r times the density.

        Parameters
        ----------
        r : float
            Spherical radius at which to calculate the moment.
        n : float
            vr^n, where vr = v x cos eta.
        m : float
            vt^m, where vt = v x sin eta.

        Returns
        -------
        float or Quantity
            <vr^n vt^m x density> at r.

        Notes
        -----
        - 2020-09-04 - Written - Bovy (UofT)
        """
        # No-op once the decorator has converted r, but validates the input
        r = conversion.parse_length(r, ro=self._ro)
        use_physical = kwargs.pop("use_physical", True)
        ro = kwargs.pop("ro", None)
        if ro is None and hasattr(self, "_roSet") and self._roSet:
            ro = self._ro
        ro = conversion.parse_length_kpc(ro)
        vo = kwargs.pop("vo", None)
        if vo is None and hasattr(self, "_voSet") and self._voSet:
            vo = self._vo
        vo = conversion.parse_velocity_kms(vo)
        if use_physical and vo is not None and ro is not None:
            fac = conversion.mass_in_msol(vo, ro) * vo ** (n + m) / ro**3
            out = self._vmomentdensity(r, n, m)
            if _optional_deps._APY_UNITS:
                u = units.Msun / units.kpc**3 * (units.km / units.s) ** (n + m)
                # a Quantity is a consumption boundary: astropy can't hold a
                # backend array (#1052), so cast unconditionally
                return units.Quantity(as_numpy(out) * fac, unit=u)
            else:
                return exit_cast(out, r) * fac
        else:
            return exit_cast(self._vmomentdensity(r, n, m), r)

    def _vmomentdensity(self, r, n, m):
        xp = resolve_namespace(r)
        if xp is numpy:
            return (
                2.0
                * numpy.pi
                * integrate.dblquad(
                    lambda eta, v: (
                        v ** (2.0 + m + n)
                        * numpy.sin(eta) ** (1 + m)
                        * numpy.cos(eta) ** n
                        * self(
                            r,
                            v * numpy.cos(eta),
                            v * numpy.sin(eta),
                            0.0,
                            0.0,
                            use_physical=False,
                        )
                    ),
                    0.0,
                    self._vmax_at_r(self._pot, r),
                    lambda x: 0.0,
                    lambda x: numpy.pi,
                )[0]
            )
        # jax/torch: tensor-product GL over (v, eta), differentiable in r; node
        # axes must trail, so broadcast r/vmax/Phi(r) onto two extra axes
        rb = xp.asarray(r) * 1.0  # coerce: torch potentials reject numpy coords
        vmax = self._vmax_at_r(self._pot, rb)
        r_b = rb[..., None, None]
        v_hi = (xp.asarray(vmax) * 1.0)[..., None, None]
        Phir_b = (xp.asarray(_evaluatePotentials(self._pot, rb, 0.0)) * 1.0)[
            ..., None, None
        ]

        def _integrand(v, eta):
            # at (R, vR, vT, z, vz) = (r, v cos eta, v sin eta, 0, 0):
            # E = Phi(r) + v^2/2 and L = Lz = r v sin(eta)
            L = r_b * v * xp.sin(eta)
            return (
                v ** (2.0 + m + n)
                * xp.sin(eta) ** (1 + m)
                * xp.cos(eta) ** n
                * self._call_internal(0.5 * v**2.0 + Phir_b, L, L)
            )

        return (
            2.0
            * numpy.pi
            * nested_quad(
                xp, _integrand, [[0.0, v_hi], [0.0, numpy.pi]], n=_QUAD_N_VMOM2D
            )
        )

    @potential_physical_input
    @physical_conversion("velocity", pop=True)
    def sigmar(self, r):
        """
        Calculate the radial velocity dispersion at radius r.

        Parameters
        ----------
        r : float
            Spherical radius at which to calculate the radial velocity dispersion.

        Returns
        -------
        float or Quantity
            The radial velocity dispersion at radius r.

        Notes
        -----
        - 2020-09-04 - Written - Bovy (UofT)
        """
        # No-op once the decorator has converted r, but validates the input
        r = conversion.parse_length(r, ro=self._ro)
        xp = resolve_namespace(r)  # numpy path: xp.sqrt == numpy.sqrt (byte-identical)
        return exit_cast(
            xp.sqrt(self._vmomentdensity(r, 2, 0) / self._vmomentdensity(r, 0, 0)), r
        )

    @potential_physical_input
    @physical_conversion("velocity", pop=True)
    def sigmat(self, r):
        """
        Calculate the tangential velocity dispersion at radius r.

        Parameters
        ----------
        r : float
            Spherical radius at which to calculate the tangential velocity dispersion.

        Returns
        -------
        float or Quantity
            The tangential velocity dispersion at radius r.

        Notes
        -----
        - 2020-09-04 - Written - Bovy (UofT)

        """
        # No-op once the decorator has converted r, but validates the input
        r = conversion.parse_length(r, ro=self._ro)
        xp = resolve_namespace(r)  # numpy path: xp.sqrt == numpy.sqrt (byte-identical)
        return exit_cast(
            xp.sqrt(self._vmomentdensity(r, 0, 2) / self._vmomentdensity(r, 0, 0)), r
        )

    @potential_physical_input
    def beta(self, r, ro=None, vo=None):
        """
        Calculate the anisotropy at radius r.

        Parameters
        ----------
        r : float
            Spherical radius at which to calculate the anisotropy.
        ro : float or Quantity, optional
            Distance scale used to interpret r when it is a Quantity (default:
            the DF's own ro).
        vo : float or Quantity, optional
            Velocity scale; not used here, accepted so that ro= and vo= can be
            passed together as for the other methods.

        Returns
        -------
        float
            Anisotropy at radius r.

        Notes
        -----
        - 2020-09-04 - Written - Bovy (UofT)

        """
        # No-op once the decorator has converted r, but validates the input
        r = conversion.parse_length(r, ro=self._ro)
        return exit_cast(
            1.0 - self._vmomentdensity(r, 0, 2) / 2.0 / self._vmomentdensity(r, 2, 0), r
        )

    ############################### SAMPLING THE DF################################
    def sample(self, R=None, z=None, phi=None, n=1, return_orbit=True, rmin=0.0):
        """
        Sample the DF

        Parameters
        ----------
        R : float, numpy.ndarray, Quantity, or None, optional
            If set, sample velocities at this radius. If array, sample velocities at these radii, ignoring n.
        z : float, numpy.ndarray, Quantity, or None, optional
            If set, sample velocities at this height. If array, sample velocities at these heights, ignoring n.
        phi : float, numpy.ndarray, Quantity, or None, optional
            If set, sample velocities at this azimuth. If array, sample velocities at these azimuths, ignoring n.
        n : int, optional
            Number of samples to generate. Default is 1.
        return_orbit : bool, optional
            If True, return an orbit.Orbit instance. If False, return a tuple of (R,vR,vT,z,vz,phi). Default is True.
        rmin : float, Quantity, optional
            Minimum radius at which to sample. Default is 0.

        Returns
        -------
        orbit.Orbit instance or tuple
            If return_orbit is True, an orbit.Orbit instance. Otherwise, a tuple of (R,vR,vT,z,vz,phi).

        Notes
        -----
        - When specifying position, it is necessary to specify both R and z; if phi is not set in this case, it is sampled
        - 2020-07-22 - Written - Lane (UofT)
        """
        rmin = conversion.parse_length(rmin, ro=self._ro)
        if hasattr(self, "_rmin_sampling") and rmin != self._rmin_sampling:
            # Build new grids, easiest
            if hasattr(self, "_xi_cmf_interpolator"):
                delattr(self, "_xi_cmf_interpolator")
            if hasattr(self, "_v_vesc_pvr_interpolator"):
                delattr(self, "_v_vesc_pvr_interpolator")
        self._rmin_sampling = conversion.parse_length(rmin, ro=self._ro)
        if R is None or z is None:  # Full 6D samples
            r = self._sample_r(n=n)
            phi, theta = self._sample_position_angles(n=n)
            R = r * numpy.sin(theta)
            z = r * numpy.cos(theta)
        else:  # 3D velocity samples
            R = conversion.parse_length(R, ro=self._ro)
            z = conversion.parse_length(z, ro=self._ro)
            # sampling is numpy-side (stateful numpy RNG): pull backend inputs
            # in; [()] turns a 0-d array into the scalar it wraps
            if is_backend_array(R):
                R = as_numpy(R)[()]
            if is_backend_array(z):
                z = as_numpy(z)[()]
            if isinstance(R, numpy.ndarray):
                assert len(R) == len(z), (
                    """When R= is set to an array, z= needs to be set to """
                    """an equal-length array"""
                )
                n = len(R)
            else:
                R = R * numpy.ones(n)
                z = z * numpy.ones(n)
            r = numpy.sqrt(R**2.0 + z**2.0)
            theta = numpy.arctan2(R, z)
            if phi is None:  # Otherwise assume phi input type matches R,z
                phi, _ = self._sample_position_angles(n=n)
            else:
                phi = conversion.parse_angle(phi)
                if is_backend_array(phi):  # sampling is numpy-side
                    phi = as_numpy(phi)[()]
                phi = (
                    phi * numpy.ones(n)
                    if not hasattr(phi, "__len__") or len(phi) < n
                    else phi
                )
        eta, psi = self._sample_velocity_angles(r, n=n)
        v = self._sample_v(r, eta, n=n)
        vr = v * numpy.cos(eta)
        vtheta = v * numpy.sin(eta) * numpy.cos(psi)
        vT = v * numpy.sin(eta) * numpy.sin(psi)
        vR = vr * numpy.sin(theta) + vtheta * numpy.cos(theta)
        vz = vr * numpy.cos(theta) - vtheta * numpy.sin(theta)
        if return_orbit:
            o = Orbit(vxvv=numpy.array([R, vR, vT, z, vz, phi]).T)
            if self._roSet and self._voSet:
                o.turn_physical_on(ro=self._ro, vo=self._vo)
            return o
        else:
            if _optional_deps._APY_UNITS and self._voSet and self._roSet:
                R = units.Quantity(R) * self._ro * units.kpc
                vR = units.Quantity(vR) * self._vo * units.km / units.s
                vT = units.Quantity(vT) * self._vo * units.km / units.s
                z = units.Quantity(z) * self._ro * units.kpc
                vz = units.Quantity(vz) * self._vo * units.km / units.s
                phi = units.Quantity(phi) * units.rad
            return (R, vR, vT, z, vz, phi)

    def _sample_r(self, n=1):
        """Generate radial position samples from potential
        Note - the function interpolates the normalized CMF onto the variable
        xi defined as:

        .. math:: \\xi = \\frac{r/a-1}{r/a+1}

        so that xi is in the range [-1,1], which corresponds to an r range of
        [0,infinity)"""
        rand_mass_frac = numpy.random.uniform(size=n)
        if hasattr(self, "_icmf"):
            r_samples = self._icmf(rand_mass_frac)
        else:
            if not hasattr(self, "_xi_cmf_interpolator"):
                self._xi_cmf_interpolator = self._make_cmf_interpolator()
            xi_samples = self._xi_cmf_interpolator(rand_mass_frac)
            r_samples = _xiToR(xi_samples, a=self._scale)
        # a forced backend makes the deterministic icdf eval a backend array;
        # samples are numpy-side by design
        return as_numpy(r_samples)

    def _make_cmf_interpolator(self):
        """Create the interpolator object for calculating radii from the CMF
        Note - must use self.xi_to_r() on any output of interpolator
        Note - the function interpolates the normalized CMF onto the variable
        xi defined as:

        .. math:: \\xi = \\frac{r-1}{r+1}

        so that xi is in the range [-1,1], which corresponds to an r range of
        [0,infinity)"""
        xp = get_namespace()  # forced/context default only (inputs are scalars)
        if xp is numpy:
            ximin = _RToxi(self._rmin_sampling, a=self._scale)
            ximax = _RToxi(self._rmax, a=self._scale)
        else:
            # a forced backend makes _RToxi resolve that backend, which rejects
            # plain floats (torch) -- coerce in and pull back to the numpy grid
            ximin = float(
                as_numpy(_RToxi(xp.asarray(self._rmin_sampling) * 1.0, a=self._scale))
            )
            ximax = float(as_numpy(_RToxi(xp.asarray(self._rmax) * 1.0, a=self._scale)))
        xis = numpy.arange(ximin, ximax, 1e-4)
        rs = _xiToR(xis, a=self._scale)
        # try/except necessary when mass doesn't take arrays, also need to
        # switch to a more general mass method at some point... (RuntimeError:
        # a forced-backend integration-based mass can't broadcast the array rs
        # against its quadrature nodes -- fall back to the per-r loop as numpy does)
        try:
            ms = mass(self._denspot, rs, use_physical=False)
        except (ValueError, TypeError, RuntimeError):
            ms = numpy.array(
                [as_numpy(mass(self._denspot, r, use_physical=False)) for r in rs]
            )
            # keep ms on the active backend so the mnorm/rmin arithmetic below
            # stays same-namespace (as_numpy'd for the icdf table at the end)
            if xp is not numpy:
                ms = xp.asarray(ms)
        mnorm = mass(self._denspot, self._rmax, use_physical=False)
        if self._rmin_sampling > 0:
            ms -= mass(self._denspot, self._rmin_sampling, use_physical=False)
            mnorm -= mass(self._denspot, self._rmin_sampling, use_physical=False)
        ms /= mnorm
        # mass() may have evaluated on a (forced) backend; the icdf table is numpy
        ms = as_numpy(ms)
        # Add total mass point to avoid extrapolation beyond rmax
        if numpy.isinf(self._rmax):
            xis = numpy.append(xis, 1)
            ms = numpy.append(ms, 1)
        else:
            # For finite rmax, add the endpoint to ensure r <= rmax
            xis = numpy.append(xis, ximax)
            ms = numpy.append(ms, 1)
        # backend-agnostic inverse-CDF: numpy queries hit the scipy spline
        # (byte-identical); backend queries evaluate the frozen table natively
        return Spline1D(ms, xis, k=1)

    def _sample_position_angles(self, n=1):
        """Generate spherical angle samples"""
        phi_samples = numpy.random.uniform(size=n) * 2 * numpy.pi
        theta_samples = numpy.arccos(1.0 - 2 * numpy.random.uniform(size=n))
        return phi_samples, theta_samples

    def _sample_v(self, r, eta, n=1):
        """Generate velocity samples: typically the total velocity, but not for OM"""
        if not hasattr(self, "_v_vesc_pvr_interpolator"):
            r_a_end = (
                max(numpy.log10(self._rmax / self._scale), 3)
                if numpy.isfinite(self._rmax)
                else 3
            )
            self._v_vesc_pvr_interpolator = self._make_pvr_interpolator(r_a_end=r_a_end)
        # samples are numpy-side by design (_vmax_at_r follows a forced backend)
        return self._v_vesc_pvr_interpolator(
            numpy.log10(r / self._scale), numpy.random.uniform(size=n), grid=False
        ) * as_numpy(self._vmax_at_r(self._pot, r))

    def _sample_velocity_angles(self, r, n=1):
        """Generate samples of angles that set radial vs tangential
        velocities"""
        eta_samples = self._sample_eta(r, n)
        psi_samples = numpy.random.uniform(size=n) * 2 * numpy.pi
        return eta_samples, psi_samples

    def _vmax_at_r(self, pot, r, **kwargs):
        """Function that gives the max velocity in the DF at r;
        typically equal to vesc, but not necessarily for finite systems
        such as King"""
        xp = resolve_namespace(r)
        if xp is numpy:
            return numpy.sqrt(
                2.0
                * (
                    _evaluatePotentials(self._pot, self._rmax + 1e-10, 0)
                    - _evaluatePotentials(self._pot, r, 0.0)
                )
            )
        # coerce coords: undecorated potential evals reject numpy/scalars (torch)
        return xp.sqrt(
            2.0
            * (
                _evaluatePotentials(self._pot, xp.asarray(self._rmax + 1e-10) * 1.0, 0)
                - _evaluatePotentials(self._pot, xp.asarray(r) * 1.0, 0.0)
            )
        )

    def _make_pvr_interpolator(self, r_a_start=-3, r_a_end=3, n_r_a=120, n_v_vesc=100):
        """
        Calculate a grid of the velocity sampling function v^2*f(E) over many
        radii. The radii are fractional with respect to some scale radius
        which characteristically describes the size of the potential,
        and the velocities are fractional with respect to the escape velocity
        at each radius r. This information is saved in a 2D interpolator which
        represents the inverse cumulative distribution at many radii. This
        allows for sampling of v/vesc given an input r/a

        Parameters
        ----------
        r_a_start : float, optional
            Radius grid start location in units of log10(r/a). Default is -3.
        r_a_end : float, optional
            Radius grid end location in units of log10(r/a). Default is 3.
        n_r_a : int, optional
            Number of radius grid points to use. Default is 120.
        n_v_vesc : int, optional
            Number of velocity grid points to use. Default is 100.

        Returns
        -------
        scipy.interpolate.RectBivariateSpline
            Interpolator for v/vesc given an input r/a.

        Notes
        -----
        - 2020-07-24 - Written - Lane (UofT)
        """
        # Check that interpolated potential has appropriate grid range
        if (
            isinstance(self._pot, interpSphericalPotential)
            and self._rmin_sampling < self._pot._rmin
        ):
            warnings.warn(
                "Interpolated potential grid rmin is larger than the rmin to be used for the v_vesc_interpolator grid. This may adversely affect the generated samples. Proceed with care!",
                galpyWarning,
            )
        # Make an array of r/a by v/vesc and then calculate p(v|r)
        r_a_start = numpy.amax(
            [numpy.log10((self._rmin_sampling + 1e-8) / self._scale), r_a_start]
        )
        r_a_end = numpy.amin([numpy.log10((self._rmax - 1e-8) / self._scale), r_a_end])
        r_a_values = 10.0 ** numpy.linspace(r_a_start, r_a_end, n_r_a)
        v_vesc_values = numpy.linspace(0, 1, n_v_vesc)
        r_a_grid, v_vesc_grid = numpy.meshgrid(r_a_values, v_vesc_values)
        vesc_grid = as_numpy(self._vmax_at_r(self._pot, r_a_grid * self._scale))
        r_grid = r_a_grid * self._scale
        vr_grid = v_vesc_grid * vesc_grid
        # Calculate p(v|r) (one vectorized -- possibly forced-backend -- DF eval,
        # pulled numpy-side for the spline construction) and normalize
        pvr_grid = as_numpy(self._p_v_at_r(vr_grid, r_grid))
        pvr_grid_cml = numpy.cumsum(pvr_grid, axis=0)
        pvr_grid_cml_norm = (
            pvr_grid_cml
            / numpy.repeat(
                pvr_grid_cml[-1, :][:, numpy.newaxis], pvr_grid_cml.shape[0], axis=1
            ).T
        )

        # Construct the inverse cumulative distribution on a regular grid
        n_new_pvr = 100  # Must be multiple of r_a_grid.shape[0]
        icdf_pvr_grid_reg = numpy.zeros((n_new_pvr, len(r_a_values)))
        icdf_v_vesc_grid_reg = numpy.zeros((n_new_pvr, len(r_a_values)))
        for i in range(pvr_grid_cml_norm.shape[1]):
            cml_pvr = pvr_grid_cml_norm[:, i]
            if numpy.all(numpy.isnan(cml_pvr)) or numpy.all(cml_pvr == 0):
                # No velocity probability at this radius (e.g., near rmax
                # where vesc ~ 0); set inverse CDF to zero velocity
                icdf_pvr_grid_reg[:, i] = numpy.linspace(0, 1, n_new_pvr)
                icdf_v_vesc_grid_reg[:, i] = 0.0
                continue
            if numpy.any(cml_pvr < 0):
                warnings.warn(
                    "The DF appears to have negative regions; we'll try to ignore these for sampling the DF, but this may adversely affect the generated samples. Proceed with care!",
                    galpyWarning,
                )
            # Negative DF regions make the cumulative velocity distribution dip
            # below zero or decrease (e.g. near a truncation radius where the
            # Eddington-inverted DF goes slightly negative). Clamp it to be
            # non-negative and enforce monotonicity, then interpolate using only
            # its strictly-increasing (unique) points so the inverse-CDF
            # interpolation below is well defined (the normalized cumulative
            # always spans 0 to 1, so at least two distinct points remain).
            cml_pvr = numpy.maximum.accumulate(numpy.clip(cml_pvr, 0.0, None))
            cml_pvr_unique, unique_indx = numpy.unique(cml_pvr, return_index=True)
            cml_pvr_inv_interp = scipy.interpolate.InterpolatedUnivariateSpline(
                cml_pvr_unique, v_vesc_values[unique_indx], k=1
            )
            pvr_samples_reg = numpy.linspace(0, 1, n_new_pvr)
            v_vesc_samples_reg = cml_pvr_inv_interp(pvr_samples_reg)
            icdf_pvr_grid_reg[:, i] = pvr_samples_reg
            icdf_v_vesc_grid_reg[:, i] = v_vesc_samples_reg
        # Create the interpolator
        return scipy.interpolate.RectBivariateSpline(
            numpy.log10(r_a_grid[0, :]),
            icdf_pvr_grid_reg[:, 0],
            icdf_v_vesc_grid_reg.T,
            kx=1,
            ky=1,
        )

    def _setup_rphi_interpolator(self, r_a_min=1e-6, r_a_max=1e6, nra=10001):
        """
        Set up the interpolator for r(phi)

        Parameters
        ----------
        r_a_min : float, optional
            Minimum r/a. Default is 1e-6.
        r_a_max : float, optional
            Maximum r/a. Default is 1e6.
        nra : int, optional
            Number of points to use in the r/a grid. Default is 10001.

        Returns
        -------
        galpy.backend.interpolate.Spline1D
            Interpolator for r(phi) (scipy-backed for numpy queries, natively
            evaluated for backend queries).

        Notes
        -----
        - 2023-02-23 - Written - Lane (UofT)
        """

        # Check if potential at r=0 is finite; if not, start at r_a_min
        xp = get_namespace()  # context/forced default only (the grid is numpy)
        if xp is numpy:
            phi_at_zero = _evaluatePotentials(self._pot, 0.0, 0)
        else:
            # coerce coords: undecorated potential evals reject scalars (torch)
            phi_at_zero = as_numpy(_evaluatePotentials(self._pot, xp.asarray(0.0), 0))
        if numpy.isfinite(phi_at_zero):
            r_a_values = numpy.concatenate(
                (numpy.array([0.0]), numpy.geomspace(r_a_min, r_a_max, nra))
            )
        else:
            r_a_values = numpy.geomspace(r_a_min, r_a_max, nra)
        if xp is numpy:
            phis = numpy.array(
                [_evaluatePotentials(self._pot, r * self._scale, 0) for r in r_a_values]
            )
        else:
            # forced backend: one vectorized eval instead of nra scalar dispatches
            phis = as_numpy(
                _evaluatePotentials(self._pot, xp.asarray(r_a_values) * self._scale, 0)
            )
        # Ensure phi is monotonic (required if coming from interpolated pot)
        if numpy.any(numpy.diff(phis) <= 0):
            phim = numpy.maximum.accumulate(phis)
            indx_rm = numpy.where(numpy.diff(phim) == 0)[0]
            phis = numpy.delete(phim, indx_rm)
            r_a_values = numpy.delete(r_a_values, indx_rm)
        # backend-agnostic r(Phi): numpy queries hit the scipy spline
        # (byte-identical); backend queries evaluate the frozen table natively
        return Spline1D(phis, r_a_values * self._scale, k=3)


class isotropicsphericaldf(sphericaldf):
    """Superclass for isotropic spherical distribution functions"""

    def __init__(self, pot=None, denspot=None, rmax=None, scale=None, ro=None, vo=None):
        """
        Initialize an isotropic distribution function

        Parameters
        ----------
        pot : Potential instance or a combined potential formed using addition (pot1+pot2+…)
            Default: None
        denspot : Potential instance or a combined potential formed using addition (pot1+pot2+…) that represent the density of the tracers (assumed to be spherical; if None, set equal to pot), optional
            Default: None
        rmax : float or Quantity, optional
            Maximum radius to consider; DF is cut off at E = Phi(rmax)
            Default: None
        scale : float, optional
            Scale parameter to be used internally
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2020-09-02 - Written - Bovy (UofT)

        """
        sphericaldf.__init__(
            self, pot=pot, denspot=denspot, rmax=rmax, scale=scale, ro=ro, vo=vo
        )

    def _call_internal(self, *args):
        """
        Calculate the distribution function for an isotropic DF.

        Parameters
        ----------
        *args : tuple of (E,L,Lz) with L and Lz optionalA

        Returns
        -------
        float
            The distribution function evaluated at E.

        Notes
        -----
        - 2020-07 - Written - Lane (UofT)

        """
        return self.fE(args[0])

    def _dMdE(self, E):
        if not hasattr(self, "_rphi"):
            self._rphi = self._setup_rphi_interpolator()
        xp = resolve_namespace(E)
        if xp is numpy:
            fE = numpy.atleast_1d(self.fE(E))
            out = numpy.zeros_like(E)
            out[fE > 0.0] = (
                16.0
                * numpy.pi**2.0
                * numpy.sqrt(2.0)
                * fE[fE > 0.0]
                * numpy.array(
                    [
                        integrate.quad(
                            lambda r: (
                                r**2.0
                                * numpy.sqrt(
                                    tE - _evaluatePotentials(self._pot, r, 0.0)
                                )
                            ),
                            0.0,
                            self._rphi(tE),
                        )[0]
                        for ii, tE in enumerate(E)
                        if fE[ii] > 0.0
                    ]
                )
            )
            # Numerical issues can make the integrand's sqrt argument negative, only
            # happens at dMdE ~ 0, so just set to zero
            out[numpy.isnan(out)] = 0.0
            return out
        # jax/torch: GL after r = rphi(E) - s^2, which cancels the sqrt turning
        # point at r = rphi(E) so fixed-order GL converges fast
        fE = xp.atleast_1d(self.fE(E))
        pos = fE > 0.0
        rphiE = xp.asarray(self._rphi(E)) * 1.0
        # dead-branch guard: out-of-bounds E gets a safe dummy radius, zeroed below
        rphiE = xp.where(pos, rphiE, xp.ones_like(rphiE))
        Eb = (xp.asarray(E) * 1.0)[..., None]

        def _integrand(s):
            r = rphiE[..., None] - s**2.0
            diff = Eb - _evaluatePotentials(self._pot, r, 0.0)
            # guard: numerical noise can push E - Phi below 0 at the turning point
            diffsafe = xp.where(diff > 0.0, diff, xp.ones_like(diff))
            return (
                r**2.0
                * xp.where(diff > 0.0, xp.sqrt(diffsafe), xp.zeros_like(diff))
                * 2.0
                * s
            )

        integral = fixed_quad(xp, _integrand, 0.0, xp.sqrt(rphiE), n=_QUAD_N_DMDE)
        return xp.where(
            pos,
            16.0 * numpy.pi**2.0 * numpy.sqrt(2.0) * fE * integral,
            xp.zeros_like(fE),
        )

    def _vmomentdensity(self, r, n, m):
        if m % 2 == 1 or n % 2 == 1:
            return 0.0
        xp = resolve_namespace(r)
        if xp is numpy:
            return (
                2.0
                * numpy.pi
                * integrate.quad(
                    lambda v: (
                        v ** (2.0 + m + n)
                        * self.fE(_evaluatePotentials(self._pot, r, 0) + 0.5 * v**2.0)
                    ),
                    0.0,
                    self._vmax_at_r(self._pot, r),
                )[0]
                * special.gamma(m // 2 + 1)
                * special.gamma(n // 2 + 0.5)
                / special.gamma(m // 2 + n // 2 + 1.5)
            )
        # jax/torch: fixed-order GL over v, differentiable in r through Phi(r)
        # and the vmax(r) integration limit; the node axis trails
        rb = xp.asarray(r) * 1.0  # coerce: torch potentials reject numpy coords
        Phir_b = (xp.asarray(_evaluatePotentials(self._pot, rb, 0)) * 1.0)[..., None]
        return (
            2.0
            * numpy.pi
            * fixed_quad(
                xp,
                lambda v: v ** (2.0 + m + n) * self.fE(Phir_b + 0.5 * v**2.0),
                0.0,
                self._vmax_at_r(self._pot, rb),
                n=_QUAD_N_VMOM,
            )
            * special.gamma(m // 2 + 1)
            * special.gamma(n // 2 + 0.5)
            / special.gamma(m // 2 + n // 2 + 1.5)
        )

    def _sample_eta(self, r, n=1):
        """Sample the angle eta which defines radial vs tangential velocities"""
        return numpy.arccos(1.0 - 2.0 * numpy.random.uniform(size=n))

    def _p_v_at_r(self, v, r):
        xp = resolve_namespace(v, r)
        if xp is not numpy:
            # coerce: a forced backend sees the numpy sampling grids here and
            # torch potentials reject numpy coords
            v = xp.asarray(v) * 1.0
            r = xp.asarray(r) * 1.0
        if hasattr(self, "_fE_interp"):
            return (
                self._fE_interp(_evaluatePotentials(self._pot, r, 0) + 0.5 * v**2.0)
                * v**2.0
            )
        else:
            return self.fE(_evaluatePotentials(self._pot, r, 0) + 0.5 * v**2.0) * v**2.0


class anisotropicsphericaldf(sphericaldf):
    """Superclass for anisotropic spherical distribution functions"""

    def __init__(self, pot=None, denspot=None, rmax=None, scale=None, ro=None, vo=None):
        """
        Initialize an anisotropic distribution function

        Parameters
        ----------
        pot : Potential instance or a combined potential formed using addition (pot1+pot2+…)
            The potential. Default: None.
        denspot : Potential instance or a combined potential formed using addition (pot1+pot2+…), optional
            The potential representing the density of the tracers (assumed to be spherical). If None, set equal to pot. Default: None.
        rmax : float or Quantity, optional
            Maximum radius to consider. DF is cut off at E = Phi(rmax). Default: None.
        scale : float, optional
            Length-scale parameter to be used internally. Default: None.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2020-07-22 - Written - Lane (UofT)

        """
        sphericaldf.__init__(
            self, pot=pot, denspot=denspot, rmax=rmax, scale=scale, ro=ro, vo=vo
        )

    def _dMdE(self, E):
        if not hasattr(self, "_rphi"):
            self._rphi = self._setup_rphi_interpolator()
        xp = resolve_namespace(E)
        if xp is numpy:

            def Lintegrand(t, L2lim, E):
                return self((E, numpy.sqrt(L2lim - t**2.0)), use_physical=False)

            out = (
                16.0
                * numpy.pi**2.0
                * numpy.array(
                    [
                        integrate.quad(
                            lambda r: (
                                r
                                * integrate.quad(
                                    Lintegrand,
                                    0.0,
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
                                )[0]
                            ),
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
            return out
        # jax/torch: tensor-product GL after r = rphi(E) - s^2 (cancels the outer
        # turning point) and t = Lmax sin(phi) (cancels the inner sqrt endpoint)
        Eb = xp.asarray(E) * 1.0
        rphiE = xp.asarray(self._rphi(E)) * 1.0
        # dead-branch guard: an unphysical (extrapolated) rphi <= 0 contributes 0
        rpos = rphiE > 0.0
        smax = xp.where(rpos, xp.sqrt(xp.where(rpos, rphiE, xp.ones_like(rphiE))), 0.0)
        E_bb = Eb[..., None, None]
        rphi_bb = xp.where(rpos, rphiE, xp.ones_like(rphiE))[..., None, None]

        def _integrand(s, phi):
            r = rphi_bb - s**2.0
            L2lim = 2.0 * r**2.0 * (E_bb - _evaluatePotentials(self._pot, r, 0.0))
            # guard: numerical noise can push E - Phi below 0 at the turning point
            Lmax = xp.where(
                L2lim > 0.0,
                xp.sqrt(xp.where(L2lim > 0.0, L2lim, xp.ones_like(L2lim))),
                xp.zeros_like(L2lim),
            )
            L = Lmax * xp.cos(phi)
            return r * self._call_internal(E_bb, L, None) * L * 2.0 * s

        return (
            16.0
            * numpy.pi**2.0
            * nested_quad(
                xp,
                _integrand,
                [[0.0, smax[..., None, None]], [0.0, numpy.pi / 2.0]],
                n=_QUAD_N_VMOM2D,
            )
        )
