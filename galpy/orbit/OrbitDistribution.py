from __future__ import annotations

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
    orbit : (..., S) Orbit, OrbitDistribution, or array_like
        The distribution, with sampling along the last axis. If 1D, the sole
        dimension is used as the sampling axis (i.e., it is a scalar
        distribution).

    *args, **kwargs : Any
        If ``orbit`` is not an Orbit (or OrbitDistribution), these are passed to
        the Orbit constructor.

    Raises
    ------
    ValueError
        If ``orbit`` is an Orbit (or OrbitDistribution) and ``args`` or
        ``kwargs`` are passed.
    """
    distribution: Orbit

    def __init__(self, orbit, *args, **kwargs) -> None:
        # Convert to Orbit
        if isinstance(orbit, OrbitDistribution):
            orbit = orbit.distribution  # TODO: .copy()
        # kept separate so only one arg/kwarg check is needed.
        if not isinstance(orbit, Orbit):
            orbit = Orbit(orbit, *args, **kwargs)
        elif args or kwargs:  # orbit, but args/kwargs
            raise ValueError("Cannot pass arguments to OrbitDistribution "
                                 "if orbit is an Orbit!")
        # else:  # orbit, no args/kwargs
        #     orbit = orbit.copy()  # copy to avoid modifying original

        # Shape check
        if orbit.shape == ():
            raise ValueError("Orbit must be an array of orbits!")
        elif len(orbit.shape) == 1:  # 1D -> scalar distribution
            orbit.reshape((1, -1))

        object.__setattr__(self, "distribution", orbit)  # bypass frozen

    @property
    def n_samples(self):
        """The number of samples."""
        return self.distribution.shape[-1]


    # =========================================================================
    # Orbit API

    def __getattr__(self, name):
        out = getattr(self.distribution, name)

        # properties
        if isinstance(out, np.ndarray):
            return Distribution(out)

        # methods
        elif callable(out):

            @functools.wraps(out)  # rm for speed?
            def wrapped(*args, **kwargs):
                _out = out(*args, **kwargs)  # call method

                if isinstance(_out, np.ndarray):
                    return Distribution(_out)
                else:
                    # TODO: better plotting
                    return _out

            return wrapped

        # other
        return out

    @property
    def shape(self):
        """The shape of the orbit."""
        return self.distribution.shape[:-1]

    @shape.setter
    def shape(self, newshape):
        self.distribution.reshape(newshape + (-1,))

    @property
    def size(self):
        """The size of the orbit."""
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
        if orbit.shape == ():
            raise ValueError("Orbit must be an array of orbits.")

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
            Number of samples to draw, by default 1_000.

        Returns
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
        mean, _u = _orbit_arr_in_system(orbit, "icrs", cov_units)
        cov = _reshape_cov(cov, mean.shape)

        # Draw samples (n_samples, *orbit.shape * 6)
        samples = multivariate_normal(mean=mean.flat, cov=cov).rvs(n_samples)
        # Reshape to (*orbit.shape, n_samples, 6)
        samples = np.moveaxis(samples.reshape((-1,) + mean.shape), 0, -2)

        # Add the original orbit (*orbit.shape, n_samples + 1, 6)
        coords = np.concatenate([mean[..., None, :], samples], axis=-2)

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
            Number of samples to draw, by default 1_000.

        Returns
        -------
        OrbitDistribution
        """
        # TODO! orbit cannot be integrated

        # Get the mean and covariance
        mean, _u = _orbit_arr_in_system(orbit, "galactocentric", cov_units)
        cov = _reshape_cov(cov, mean.shape)

        # Draw samples (n_samples, *orbit.shape * 6)
        samples_xyz = multivariate_normal(mean=mean.flat, cov=cov).rvs(n_samples)
        # Reshape to (*orbit.shape, n_samples, 6)
        samples_xyz = np.moveaxis(samples_xyz.reshape((-1,) + mean.shape), 0, -2)
        # Add the original orbit (*orbit.shape, n_samples + 1, 6)
        coords_xyz = np.concatenate([mean[..., None, :], samples_xyz], axis=-2)

        # Convert to cylindrical coordinates
        R, phi, z = rect_to_cyl(*(coords_xyz[..., i] * _u[i] for i in range(3)))
        vR,vT,vz = rect_to_cyl_vec(*(coords_xyz[..., i] * _u[i] for i in range(3, 6)),
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


def _orbit_arr_in_system(orbit, system, units):
    """Get the mean of the orbit distribution in the given system.

    Parameters
    ----------
    orbit : Orbit
        The orbit instance.
    system : str
        A string in ``_system_dict``.
    units : tuple[Unit, ...] or None
        The units of the mean.

    Returns
    -------
    ndarray
        The mean of the orbit distribution.
    units : tuple[Unit, ...] or tuple[1, ...]
        The units of the mean. If astropy is not installed, this is a tuple of
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
        means = (getattr(orbit, n)(quantity=True).to_value(unit)
                 for n, unit in zip(names, units))
    elif units is not None:
        raise ValueError("Cannot specify units if astropy is not installed!")
    else:
        means = (getattr(orbit, n)(quantity=False) for n in names)

    mean = np.atleast_2d(np.stack(tuple(means), axis=-1))  # (N, 6)

    return mean, units if _APY_LOADED else (1, 1, 1, 1, 1, 1)


def _reshape_cov(cov, mean_shape):
    if isinstance(cov, float):
        pass
    elif isinstance(cov, np.ndarray):
        # TODO: better shape checks.
        if cov.shape == (6, 6):
            cov = np.kron(np.eye(np.prod(mean_shape[:-1]), dtype=int), cov)

    return cov
