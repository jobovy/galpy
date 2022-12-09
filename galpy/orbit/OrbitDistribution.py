from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.uncertainty import Distribution

from galpy.orbit.Orbits import Orbit


@dataclass(frozen=True)
class OrbitDistribution:
    orbit: Orbit

    def __getattr__(self, name):
        out = getattr(self.orbit, name)

        if isinstance(out, (np.ndarray, u.Quantity)):
            return Distribution(out)

        elif callable(out):

            def wrapped(*args, **kwargs):
                _out = out(*args, **kwargs)
                if isinstance(_out, (np.ndarray, u.Quantity)):
                    return Distribution(_out)
                return _out
            return wrapped
        return out

    @classmethod
    def from_orbit(cls, orbit):
        return cls(orbit)

    @classmethod
    def from_obs_covariance(cls, orbit, covariance, *, units=(u.deg, u.deg, u.kpc, u.mas/u.yr, u.mas/u.yr, u.km/u.s), n_samples=10):
        """From observational covariance matrix.

        Parameters
        ----------
        orbit : Orbit
            The orbit instance.
        covariance : ndarray
            Columns / rows are [ra, dec, distance, pm_ra_cosdec, pm_dec, vr]
        units : tuple, optional
            The units of the covariance matrix, by default (u.deg, u.deg, u.kpc,
            u.mas/u.yr, u.mas/u.yr, u.km/u.s)s
        n_samples : int, optional
            _description_, by default 10

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        if orbit.shape != ():
            raise ValueError("Orbit must be a single orbit!")
        # TODO! orbit cannot be integrated

        from scipy.stats import multivariate_normal

        mean = np.c_[
            tuple(
                getattr(orbit, n)(quantity=True).to_value(unit)
                for n, unit in zip(['ra', 'dec', 'dist', 'pmra', 'pmdec', 'vr'],
                                    units)
            )
        ][0]

        mvn = multivariate_normal(
            mean=mean,
            cov=covariance
        )
        coords = mvn.rvs(n_samples)

        new_orbit = Orbit(coords, radec=True, ro=orbit._ro, vo=orbit._vo, zo=orbit._zo, solarmotion=orbit._solarmotion)
        if hasattr(orbit, "_name"):
            new_orbit.__dict__["_name"] = orbit._name

        return cls(new_orbit)

    # TODO! from galactocentic cartesian covariance matrix
