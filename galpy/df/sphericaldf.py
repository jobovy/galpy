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
from scipy import integrate, interpolate, special

from ..orbit import Orbit
from ..potential import interpSphericalPotential, mass
from ..potential.Potential import _evaluatePotentials
from ..potential.SCFPotential import _RToxi, _xiToR
from ..util import _optional_deps, conversion, galpyWarning
from ..util.conversion import physical_conversion
from .df import df

# Use _APY_LOADED/_APY_UNITS like this to be able to change them in tests
if _optional_deps._APY_LOADED:
    from astropy import units


class sphericaldf(df):
    """Superclass for spherical distribution functions"""

    def __init__(self, pot=None, denspot=None, rmax=None, scale=None, ro=None, vo=None):
        """
        Initializes a spherical DF

        Parameters
        ----------
        pot : Potential instance or list thereof
            The potential. Default is None.
        denspot : Potential instance or list thereof, optional
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
        self._pot = pot
        self._denspot = self._pot if denspot is None else denspot
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
    @physical_conversion("phasespacedensity", pop=True)
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
        """
        # Get E,L,Lz
        if len(args) == 1:
            if not isinstance(args[0], Orbit):  # Assume tuple (E,L,Lz)
                E, L, Lz = (args[0] + (None, None))[:3]
            else:  # Orbit
                E = args[0].E(pot=self._pot, use_physical=False)
                L = numpy.sqrt(numpy.sum(args[0].L(use_physical=False) ** 2.0))
                Lz = args[0].Lz(use_physical=False)
            E = numpy.atleast_1d(conversion.parse_energy(E, vo=self._vo))
            L = numpy.atleast_1d(conversion.parse_angmom(L, ro=self._ro, vo=self._vo))
            Lz = numpy.atleast_1d(conversion.parse_angmom(Lz, ro=self._vo, vo=self._vo))
        else:  # Assume R,vR,vT,z,vz,(phi)
            R, vR, vT, z, vz, phi = (args + (None,))[:6]
            R = conversion.parse_length(R, ro=self._ro)
            vR = conversion.parse_velocity(vR, vo=self._vo)
            vT = conversion.parse_velocity(vT, vo=self._vo)
            z = conversion.parse_length(z, ro=self._ro)
            vz = conversion.parse_velocity(vz, vo=self._vo)
            vtotSq = vR**2.0 + vT**2.0 + vz**2.0
            E = numpy.atleast_1d(0.5 * vtotSq + _evaluatePotentials(self._pot, R, z))
            Lz = numpy.atleast_1d(R * vT)
            r = numpy.sqrt(R**2.0 + z**2.0)
            vrad = (R * vR + z * vz) / r
            L = numpy.atleast_1d(numpy.sqrt(vtotSq - vrad**2.0) * r)
        return self._call_internal(E, L, Lz).reshape(
            args[0].shape
            if len(args) == 1 and hasattr(args[0], "shape")
            else (
                args[0][0].shape
                if len(args) == 1
                and hasattr(args[0], "__len__")
                and hasattr(args[0][0], "shape")
                else (args[0].shape if hasattr(args[0], "shape") else ())
            )
        )

    @physical_conversion("energydensity", pop=True)
    def dMdE(self, E):
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

        """
        return self._dMdE(
            numpy.atleast_1d(conversion.parse_energy(E, vo=self._vo))
        ).reshape(E.shape if isinstance(E, numpy.ndarray) else ())

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
        if use_physical and not vo is None and not ro is None:
            fac = vo ** (n + m) / ro**3
            if _optional_deps._APY_UNITS:
                u = 1 / units.kpc**3 * (units.km / units.s) ** (n + m)
            out = self._vmomentdensity(r, n, m)
            if _optional_deps._APY_UNITS:
                return units.Quantity(out * fac, unit=u)
            else:
                return out * fac
        else:
            return self._vmomentdensity(r, n, m)

    def _vmomentdensity(self, r, n, m):
        return (
            2.0
            * numpy.pi
            * integrate.dblquad(
                lambda eta, v: v ** (2.0 + m + n)
                * numpy.sin(eta) ** (1 + m)
                * numpy.cos(eta) ** n
                * self(
                    r,
                    v * numpy.cos(eta),
                    v * numpy.sin(eta),
                    0.0,
                    0.0,
                    use_physical=False,
                ),
                0.0,
                self._vmax_at_r(self._pot, r),
                lambda x: 0.0,
                lambda x: numpy.pi,
            )[0]
        )

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
        r = conversion.parse_length(r, ro=self._ro)
        return numpy.sqrt(self._vmomentdensity(r, 2, 0) / self._vmomentdensity(r, 0, 0))

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
        r = conversion.parse_length(r, ro=self._ro)
        return numpy.sqrt(self._vmomentdensity(r, 0, 2) / self._vmomentdensity(r, 0, 0))

    def beta(self, r):
        """
        Calculate the anisotropy at radius r.

        Parameters
        ----------
        r : float
            Spherical radius at which to calculate the anisotropy.

        Returns
        -------
        float
            Anisotropy at radius r.

        Notes
        -----
        - 2020-09-04 - Written - Bovy (UofT)

        """
        r = conversion.parse_length(r, ro=self._ro)
        return 1.0 - self._vmomentdensity(r, 0, 2) / 2.0 / self._vmomentdensity(r, 2, 0)

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
        return r_samples

    def _make_cmf_interpolator(self):
        """Create the interpolator object for calculating radii from the CMF
        Note - must use self.xi_to_r() on any output of interpolator
        Note - the function interpolates the normalized CMF onto the variable
        xi defined as:

        .. math:: \\xi = \\frac{r-1}{r+1}

        so that xi is in the range [-1,1], which corresponds to an r range of
        [0,infinity)"""
        ximin = _RToxi(self._rmin_sampling, a=self._scale)
        ximax = _RToxi(self._rmax, a=self._scale)
        xis = numpy.arange(ximin, ximax, 1e-4)
        rs = _xiToR(xis, a=self._scale)
        # try/except necessary when mass doesn't take arrays, also need to
        # switch to a more general mass method at some point...
        try:
            ms = mass(self._denspot, rs, use_physical=False)
        except (ValueError, TypeError):
            ms = numpy.array([mass(self._denspot, r, use_physical=False) for r in rs])
        mnorm = mass(self._denspot, self._rmax, use_physical=False)
        if self._rmin_sampling > 0:
            ms -= mass(self._denspot, self._rmin_sampling, use_physical=False)
            mnorm -= mass(self._denspot, self._rmin_sampling, use_physical=False)
        ms /= mnorm
        # Add total mass point
        if numpy.isinf(self._rmax):
            xis = numpy.append(xis, 1)
            ms = numpy.append(ms, 1)
        return scipy.interpolate.InterpolatedUnivariateSpline(ms, xis, k=3)

    def _sample_position_angles(self, n=1):
        """Generate spherical angle samples"""
        phi_samples = numpy.random.uniform(size=n) * 2 * numpy.pi
        theta_samples = numpy.arccos(1.0 - 2 * numpy.random.uniform(size=n))
        return phi_samples, theta_samples

    def _sample_v(self, r, eta, n=1):
        """Generate velocity samples: typically the total velocity, but not for OM"""
        if not hasattr(self, "_v_vesc_pvr_interpolator"):
            self._v_vesc_pvr_interpolator = self._make_pvr_interpolator()
        return self._v_vesc_pvr_interpolator(
            numpy.log10(r / self._scale), numpy.random.uniform(size=n), grid=False
        ) * self._vmax_at_r(self._pot, r)

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
        return numpy.sqrt(
            2.0
            * (
                _evaluatePotentials(self._pot, self._rmax + 1e-10, 0)
                - _evaluatePotentials(self._pot, r, 0.0)
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
        vesc_grid = self._vmax_at_r(self._pot, r_a_grid * self._scale)
        r_grid = r_a_grid * self._scale
        vr_grid = v_vesc_grid * vesc_grid
        # Calculate p(v|r) and normalize
        pvr_grid = self._p_v_at_r(vr_grid, r_grid)
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
            if numpy.any(cml_pvr < 0):
                warnings.warn(
                    "The DF appears to have negative regions; we'll try to ignore these for sampling the DF, but this may adversely affect the generated samples. Proceed with care!",
                    galpyWarning,
                )
            cml_pvr[cml_pvr < 0] = 0.0
            start_indx = numpy.amax(
                numpy.arange(len(cml_pvr))[cml_pvr == numpy.amin(cml_pvr)]
            )
            end_indx = (
                numpy.amin(numpy.arange(len(cml_pvr))[cml_pvr == numpy.amax(cml_pvr)])
                + 1
            )
            cml_pvr_inv_interp = scipy.interpolate.InterpolatedUnivariateSpline(
                cml_pvr[start_indx:end_indx], v_vesc_values[start_indx:end_indx], k=1
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
        scipy.interpolate.InterpolatedUnivariateSpline
            Interpolator for r(phi).

        Notes
        -----
        - 2023-02-23 - Written - Lane (UofT)
        """

        r_a_values = numpy.concatenate(
            (numpy.array([0.0]), numpy.geomspace(r_a_min, r_a_max, nra))
        )
        phis = numpy.array(
            [_evaluatePotentials(self._pot, r * self._scale, 0) for r in r_a_values]
        )
        # Ensure phi is monotonic (required if coming from interpolated pot)
        if numpy.any(numpy.diff(phis) <= 0):
            phim = numpy.maximum.accumulate(phis)
            indx_rm = numpy.where(numpy.diff(phim) == 0)[0]
            phis = numpy.delete(phim, indx_rm)
            r_a_values = numpy.delete(r_a_values, indx_rm)
        return interpolate.InterpolatedUnivariateSpline(
            phis, r_a_values * self._scale, k=3
        )


class isotropicsphericaldf(sphericaldf):
    """Superclass for isotropic spherical distribution functions"""

    def __init__(self, pot=None, denspot=None, rmax=None, scale=None, ro=None, vo=None):
        """
        Initialize an isotropic distribution function

        Parameters
        ----------
        pot : Potential instance or list thereof
            Default: None
        denspot : Potential instance or list thereof that represent the density of the tracers (assumed to be spherical; if None, set equal to pot), optional
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
        fE = self.fE(E)
        out = numpy.zeros_like(E)
        out[fE > 0.0] = (
            16.0
            * numpy.pi**2.0
            * numpy.sqrt(2.0)
            * fE[fE > 0.0]
            * numpy.array(
                [
                    integrate.quad(
                        lambda r: r**2.0
                        * numpy.sqrt(tE - _evaluatePotentials(self._pot, r, 0.0)),
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

    def _vmomentdensity(self, r, n, m):
        if m % 2 == 1 or n % 2 == 1:
            return 0.0
        return (
            2.0
            * numpy.pi
            * integrate.quad(
                lambda v: v ** (2.0 + m + n)
                * self.fE(_evaluatePotentials(self._pot, r, 0) + 0.5 * v**2.0),
                0.0,
                self._vmax_at_r(self._pot, r),
            )[0]
            * special.gamma(m // 2 + 1)
            * special.gamma(n // 2 + 0.5)
            / special.gamma(m // 2 + n // 2 + 1.5)
        )

    def _sample_eta(self, r, n=1):
        """Sample the angle eta which defines radial vs tangential velocities"""
        return numpy.arccos(1.0 - 2.0 * numpy.random.uniform(size=n))

    def _p_v_at_r(self, v, r):
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
        pot : Potential instance or list thereof
            The potential. Default: None.
        denspot : Potential instance or list thereof, optional
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

        def Lintegrand(t, L2lim, E):
            return self((E, numpy.sqrt(L2lim - t**2.0)), use_physical=False)

        out = (
            16.0
            * numpy.pi**2.0
            * numpy.array(
                [
                    integrate.quad(
                        lambda r: r
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
        return out
