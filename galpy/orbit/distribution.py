from __future__ import annotations

import copy as pycopy
import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from scipy.stats import multivariate_normal

from galpy.orbit.Orbits import Orbit
from galpy.util._optional_deps import _APY_LOADED
from galpy.util.coords import rect_to_cyl, rect_to_cyl_vec

if TYPE_CHECKING:
    from astropy.units import UnitBase as UB

if _APY_LOADED:
    import astropy.units as u
    from astropy.uncertainty import Distribution

else:
    def Distribution(x: np.ndarray, /) -> np.ndarray:
        new_dtype = np.dtype({'names': ['samples'],
                              'formats': [(x.dtype, (x.shape[-1],))]})
        samples = x.view(dtype=new_dtype)
        samples.shape = x.shape[:-1]
        return samples


###############################################################################
# OrbitDistribution

@dataclass(frozen=True)
class OrbitDistribution:
    """A distribution of orbits.

    Parameters
    ----------
    orbits : (..., S) Orbit, OrbitDistribution, or array_like
        The orbit distribution. with the last axis interpreted as the sampling
        for the earlier axes. If the orbit distribution is 1D, the sole
        dimension is used as the sampling axis (i.e., it is a scalar orbit with
        S - 1 samples).

        Input options:
            - `~galpy.orbit.Orbit` : see above.
            - `~galpy.orbit.OrbitDistribution` : copies the distribution.
            - other : passed to the `~galpy.orbit.Orbit` constructor.

    *args, **kwargs : Any
        If ``orbit`` is not an `~galpy.orbit.Orbit` (nor
        ~galpy.orbit.OrbitDistribution`), these are passed to the Orbit
        constructor.

    Raises
    ------
    ValueError
        If ``orbit`` is an `~galpy.orbit.Orbit` (nor
       `~galpy.orbit.OrbitDistribution`) and ``args`` or ``kwargs`` are passed.

    Examples
    --------
    For this first example, we'll sample positions from a Plummer sphere model
    of a cluster and then, for each position, draw a distribution of samples
    from a Gaussian distribution with a given covariance matrix to represent a
    Monte Carlo estimate of the true distribution. We can then integrate these
    orbits and get a distribution of the final positions for each sampled star
    in the cluster.

    >>> import numpy as np
    >>> from galpy.potential import PlummerPotential, MWPotential2014
    >>> from galpy.df import isotropicPlummerdf

    We make the cluster sample

    >>> cluster_pot = PlummerPotential(ro=8, vo=220)
    >>> cluster_df = isotropicPlummerdf(cluster_pot)
    >>> cluster_stars = cluster_df.sample(n=100)

    We make the distribution, sampling 1000 orbits for each cluster star. Here
    the covariance matrix is a simple, scaled identity matrix, but it can be any
    valid covariance matrix: for each star or for all the stars (6x6 for 1
    orbit, 12x12 for 2 orbits, etc.).

    >>> od = OrbitDistribution.from_cov_icrs(cluster_stars, cov=0.1, n_samples=1000)
    >>> od.shape, od.n_samples
    (100,), 1000

    We integrate the orbits and get the final positions. The result is a
    `~astropy.uncertainty.Distribution` object, which has convenience methods
    for working with the distribution.

    >>> t = np.linspace(0, 1, 100)
    >>> od.integrate(t, MWPotential2014)

    >>> r = od.r(t[-1])  # r at the final time
    >>> type(r)
    <class 'astropy.uncertainty.core.QuantityDistribution'>
    >>> r.shape, r.n_samples
    (100,), 1000
    >>> r.pdf_mean()
    [3.1766, ...] kpc
    >>> r.pdf_std()
    [0.38180044, ...] kpc

    The constructors ``from_cov_icrs``, ``from_cov_galactocentric``, etc. are
    just convenience functions for creating samples from a given covariance
    matrix. The following example shows how to create a
    `~galpy.orbit.OrbitDistribution` from samples directly.

    We'll use the same cluster as before, but this time we'll draw MC samples
    from a uniform distribution centered on each point.

    >>> import scipy.stats as stats
    >>> c = orbit.SkyCoord().transform_to("icrs")
    >>> primary = np.vstack([c.cartesian.xyz.value,
    ...                      c.cartesian.differentials["s"].d_xyz.value]).T

    >>> deltas = stats.uniform.rvs(
        loc= -np.array([0.1, 0.1, 1, 0.2, 0.2, 5])[None, :, None] / 2,
        scale=np.array([0.1, 0.1, 1, 0.2, 0.2, 5])[None, :, None],
        size=(*primary.shape, 99))
    >>> samples = np.moveaxis(primary[..., None] + deltas, 1, -1)
    >>> coords = np.concatenate([primary[..., None, :], samples], axis=-2)
    >>> od = OrbitDistribution(coords, radec=True,
                  ro=cluster_stars._ro, vo=cluster_stars._vo,
                  zo=cluster_stars._zo, solarmotion=cluster_stars._solarmotion)
    >>> od.shape, od.n_samples
    (100,), 1000

    This can likewise be integrated and all the methods will work like
    `~galpy.orbit.Orbit` objects, but the result will be a
    `~astropy.uncertainty.Distribution`.

    Notes
    -----
    Returned arrays (including `~astropy.units.Quantity`) are wrapped in
    `~astropy.uncertainty.Distribution` if `astropy` is installed. Otherwise,
    the samples are stored in a structured array with a single field named
    ``samples``.

    `~galpy.orbit.OrbitDistribution` is a frozen `~dataclasses.dataclass`.
    For more information on immutability and convenience methods for working
    with dataclasses, see the `dataclasses` documentation.
    """
    distribution: Orbit

    def __init__(self, orbits, *args, **kwargs) -> None:
        # Convert to Orbit
        if isinstance(orbits, OrbitDistribution):
            orbits = pycopy.copy(orbits.distribution)
        # kept separate so only one arg/kwarg check is needed.
        if not isinstance(orbits, Orbit):
            orbits = Orbit(orbits, *args, **kwargs)
        elif args or kwargs:  # orbit, but args/kwargs
            raise ValueError(
                "Cannot pass arguments to OrbitDistribution "
                "if 'orbits' argument is an Orbit or OrbitDistribution.")
        else:  # orbit, no args/kwargs
            orbits = pycopy.copy(orbits)

        # Shape check
        if orbits.shape == () or orbits.shape[-1] == 1:
            raise ValueError("Orbit must be an array of orbits")
        elif len(orbits.shape) == 1:  # 1D -> scalar distribution
            orbits.reshape((1, -1))

        object.__setattr__(self, "distribution", orbits)  # bypass frozen

    @property
    def n_samples(self):
        """The number of samples."""
        return self.distribution.shape[-1]

    # =========================================================================
    # Orbit API

    def __getattr__(self, name):
        # Get the attr / method from the underlying orbit.
        out = getattr(self.distribution, name)

        # attr / property
        if isinstance(out, np.ndarray):
            return Distribution(out)

        # methods
        elif callable(out):

            @functools.wraps(out)  # TODO: rm for speed or keep for IDEs?
            def wrapped(*args, **kwargs):
                _out = out(*args, **kwargs)  # call method

                if isinstance(_out, np.ndarray):
                    return Distribution(_out)
                return _out

            return wrapped

        # other
        return out

    @property
    def shape(self):
        """The shape of the orbit, not including the samples dimension."""
        return self.distribution.shape[:-1]

    @shape.setter
    def shape(self, newshape):
        self.distribution.reshape(newshape + (-1,))

    @property
    def size(self):
        """The size of the orbit, not including the samples dimensions."""
        return np.prod(self.shape)

    def plot(self, *args, **kwargs):
        """Plot the distribution.

        Parameters
        ----------
        *args
            Positional arguments to pass to the orbit's plot method.
        **kwargs
            Keyword arguments to pass to the orbit's plot method.
        """
        # Plot orbit samples
        nkw = kwargs.copy()
        nkw["c"] = "gray"
        # TODO: it would be nice to have the same color as the primary orbit
        # but with alpha set to a lower value.
        nkw["alpha"] = kwargs.get("alpha", 1) / 2
        self.distribution[..., 1:].plot(*args, **nkw)

        # Plot primary orbit
        kwargs["overplot"] = True
        self.distribution[..., 0].plot(*args, **kwargs)

    # =========================================================================
    # More explicit constructors

    @classmethod
    def from_samples(cls, orbit):
        """Create a distribution from an `~galpy.orbit.Orbit` sampling.

        Parameters
        ----------
        orbit : (..., S) Orbit
            The orbit instance. Must be an array of orbits.
            The last axis is the sample axis.

        Returns
        -------
        OrbitDistribution
            The distribution of orbits.

        Raises
        ------
        ValueError
            If the orbit is not an array of orbits.
        """
        return cls(orbit)

    @classmethod
    def from_cov_icrs(cls, orbit, cov, *, cov_units=None, n_samples=1_000):
        """Construct a distribution from the observational covariance matrix.

        Parameters
        ----------
        orbit : Orbit
            The orbit instance. This is the primary orbit of the distribution.
        cov : ndarray
            Columns / rows are [ra, dec, distance, pm_ra_cosdec, pm_dec, vr]

        cov_units : tuple, optional keyword-only
            The units of the covariance matrix, by default (u.deg, u.deg, u.kpc,
            u.mas/u.yr, u.mas/u.yr, u.km/u.s).
        n_samples : int, optional keyword-only
            Number of samples to draw, by default ``1_000``, plus the primary
            orbit (so ``n_samples`` - 1 new samples are drawn.).

        Returnss
        -------
        OrbitDistribution
            The distribution of orbits, with the original orbit included.

        Raises
        ------
        ValueError
            If the orbit is integrated.
        """
        # TODO: orbit cannot be integrated

        # Get the mean and covariance
        primary, _u = _coords_in_system(orbit, "icrs", cov_units)
        cov = _reshape_cov(cov, primary.shape)

        # Draw samples (n_samples, *orbit.shape * 6)
        samples = multivariate_normal(mean=primary.flat, cov=cov).rvs(n_samples - 1)
        # Reshape to (*orbit.shape, n_samples, 6)
        samples = np.moveaxis(samples.reshape((-1,) + primary.shape), 0, -2)
        # Add the original orbit (*orbit.shape, n_samples + 1, 6)
        coords = np.concatenate([primary[..., None, :], samples], axis=-2)

        # Make orbit (*orbit.shape, n_samples)
        o = Orbit(coords, radec=True, ro=orbit._ro, vo=orbit._vo, zo=orbit._zo,
                  solarmotion=orbit._solarmotion)
        if hasattr(orbit, "_name"):
            o.__dict__["_name"] = orbit._name

        # Make Distribution (*orbit.shape)
        return cls(o)

    @classmethod
    def from_cov_galactocentric(cls, orbit, cov, *, cov_units=None, n_samples=1_000):
        """From galactocentric cartesian covariance matrix.

        Parameters
        ----------
        orbit : Orbit
            The orbit instance.
        cov : ndarray
            Columns / rows are [x, y, z, vx, vy, vz]

        cov_units : tuple, optional keyword-only
            The units of the covariance matrix, by default (u.deg, u.deg, u.kpc,
            u.mas/u.yr, u.mas/u.yr, u.km/u.s).
        n_samples : int, optional keyword-only
            Number of samples to draw, by default ``1_000``, plus the primary
            orbit (so ``n_samples`` - 1 new samples are drawn.).

        Returns
        -------
        OrbitDistribution
        """
        # TODO! orbit cannot be integrated

        # Get the mean and covariance
        primary, _u = _coords_in_system(orbit, "galactocentric", cov_units)
        cov = _reshape_cov(cov, primary.shape)

        # Draw samples (n_samples, *orbit.shape * 6)
        samples = multivariate_normal(mean=primary.flat, cov=cov).rvs(n_samples - 1)
        # Reshape to (*orbit.shape, n_samples, 6)
        samples = np.moveaxis(samples.reshape((-1,) + primary.shape), 0, -2)
        # Add the original orbit (*orbit.shape, n_samples + 1, 6)
        coords = np.concatenate([primary[..., None, :], samples], axis=-2)

        # Convert to cylindrical coordinates
        R, phi, z = rect_to_cyl(*(coords[..., i] * _u[i] for i in range(3)))
        vR,vT,vz = rect_to_cyl_vec(*(coords[..., i] * _u[i] for i in range(3, 6)),
                                   R, phi, z, cyl=True)

        o = Orbit([R, vR, vT, z, vz, phi], ro=orbit._ro, vo=orbit._vo, zo=orbit._zo,
                  solarmotion=orbit._solarmotion)
        if hasattr(orbit, "_name"):
            o.__dict__["_name"] = orbit._name

        return cls(o)


###############################################################################
# Constructor helpers

class _SystemInfo(NamedTuple):
    names: tuple[str, str, str, str, str, str]
    default_units: tuple[UB, UB, UB, UB, UB, UB] | None


_system_dict: dict[str, _SystemInfo] = {}
if _APY_LOADED:
    _system_dict["icrs"] = _SystemInfo(
        names=("ra", "dec", "dist", "pmra", "pmdec", "vlos"),
        default_units=(u.deg, u.deg, u.kpc, u.mas/u.yr, u.mas/u.yr, u.km/u.s)
    )
    _system_dict["galactocentric"] = _SystemInfo(
        names=("x", "y", "z", "vx", "vy", "vz"),
        default_units=(u.kpc, u.kpc, u.kpc, u.km/u.s, u.km/u.s, u.km/u.s)
    )
else:
    _system_dict["icrs"] = _SystemInfo(
        names=("ra", "dec", "dist", "pmra", "pmdec", "vlos"),
        default_units=None
    )
    _system_dict["galactocentric"] = _SystemInfo(
        names=("x", "y", "z", "vx", "vy", "vz"),
        default_units=None
    )


def _coords_in_system(orbit, system, units):
    """Get the orbit coordinate in the given system.

    If Astropy is installed, there are more general ways to do this. For
    example, to get the coordinates in the ICRS system and a Cartesian
    representation:

    >>> c = orbit.SkyCoord().transform_to(system)
    >>> vxvv = np.vstack([c.cartesian.xyz.value,
    ...                   c.cartesian.differentials["s"].d_xyz.value]).T

    Parameters
    ----------
    orbit : Orbit
        The orbit instance.
    system : str
        A string in ``_system_dict``.
    units : tuple[Unit, ...] or None
        The units of the system.

    Returns
    -------
    ndarray
        The orbit coordinates.
    units : tuple[Unit, ...] or tuple[1, ...]
        The units of the system. If astropy is not installed, this is a tuple of
        ones.

    Raises
    ------
    ValueError
        If Astropy units are specified but astropy is not installed. I'm not
        sure how this would happen, except if Astropy were installed after galpy
        were imported.
    """
    names, default_units = _system_dict[system]

    if _APY_LOADED:
        if units is None:
            units = default_units
        vxvvs = (getattr(orbit, n)(quantity=True).to_value(unit)
                 for n, unit in zip(names, units))
    elif units is not None:
        raise ValueError("Cannot specify units if astropy is not installed!")
    else:
        vxvvs = (getattr(orbit, n)(quantity=False) for n in names)

    vxvv = np.atleast_2d(np.stack(tuple(vxvvs), axis=-1))  # (N, 6)

    return vxvv, units if _APY_LOADED else (1, 1, 1, 1, 1, 1)


def _reshape_cov(cov, mean_shape):
    if isinstance(cov, float):
        pass
    elif isinstance(cov, np.ndarray):
        # TODO: better shape checks.
        if cov.shape == (6, 6):
            cov = np.kron(np.eye(np.prod(mean_shape[:-1]), dtype=int), cov)

    return cov
