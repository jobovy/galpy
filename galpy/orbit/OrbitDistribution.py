from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Sequence, TypedDict, TypeVar

import numpy as np
from scipy.stats import multivariate_normal

from galpy.orbit.Orbits import Orbit
from galpy.util._optional_deps import _APY_LOADED
from galpy.util.coords import rect_to_cyl, rect_to_cyl_vec

if TYPE_CHECKING:
    from astropy.units import Quantity
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


Self = TypeVar("Self", bound="OrbitDistribution")
L1 = Literal[1]

class _SystemInfo(TypedDict):
    names: tuple[str, str, str, str, str, str]
    units: tuple[UB, UB, UB, UB, UB, UB] | None


_info: dict[str, _SystemInfo] = {}
if _APY_LOADED:
    _info["icrs"] = _SystemInfo(
        names=("ra", "dec", "dist", "pmra", "pmdec", "vlos"),
        units=(u.deg, u.deg, u.kpc, u.mas/u.yr, u.mas/u.yr, u.km/u.s)
    )
    _info["galactocentric"] = _SystemInfo(
        names=("x", "y", "z", "vx", "vy", "vz"),
        units=(u.kpc, u.kpc, u.kpc, u.km/u.s, u.km/u.s, u.km/u.s)
    )
else:
    _info["icrs"] = _SystemInfo(
        names=("ra", "dec", "dist", "pmra", "pmdec", "vlos"),
        units=None
    )
    _info["galactocentric"] = _SystemInfo(
        names=("x", "y", "z", "vx", "vy", "vz"),
        units=None
    )


def _get_mean(
    orbit: Orbit,
    system: str,
    units: tuple[UB, ...] | None
) -> tuple[np.ndarray, tuple[UB, UB, UB, UB, UB, UB] | tuple[L1, L1, L1, L1, L1, L1]]:
    names, default_units = _info[system]["names"], _info[system]["units"]

    if _APY_LOADED:
        if units is None:
            units = default_units

        mean = np.c_[
            tuple(
                getattr(orbit, n)(quantity=True).to_value(unit)
                for n, unit in zip(names, units)
            )
        ][0]
    elif units is not None:
        raise ValueError("Cannot specify units if astropy is not installed!")
    else:
        mean = np.c_[
            tuple(
                getattr(orbit, n)(quantity=False)
                for n in names
            )
        ][0]

    return mean, units if _APY_LOADED else (1, 1, 1, 1, 1, 1)

@dataclass(frozen=True, slots=True)
class OrbitDistribution:
    """A distribution of orbits.

    Parameters
    ----------
    orbit : Orbit
        The orbit instance.
    """
    orbit: Orbit

    def __getattr__(self, name: str) -> Any:
        out = getattr(self.orbit, name)

        # properties
        if isinstance(out, np.ndarray):
            return Distribution(out)

        # methods
        elif callable(out):

            @functools.wraps(out)  # rm for speed?
            def wrapped(*args: Any, **kwargs: Any) -> Any:
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
    def shape(self) -> tuple[int, ...]:
        """The shape of the distribution."""
        return self.orbit[0].shape

    def plot(self, *args: Any, **kwargs: Any) -> None:
        """Plot the distribution.

        Parameters
        ----------
        *args
            Positional arguments to pass to the orbit's plot method.
        **kwargs
            Keyword arguments to pass to the orbit's plot method.
        """
        # TODO! better plotting
        self.orbit[1:].plot(*args, **kwargs)

        nkw = kwargs.copy()
        nkw["overplot"] = True
        self.orbit[0].plot(*args, **nkw)

    # =========================================================================
    # Constructors

    @classmethod
    def from_orbit(cls: type[Self], orbit: Orbit) -> Self:
        """Create a distribution from an orbit.

        Parameters
        ----------
        orbit : Orbit
            The orbit instance. Must be an array of orbits.

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
    def from_icrs_cov(
        cls: type[Self],
        orbit: Orbit,
        cov: np.ndarray,
        *,
        cov_units: tuple[UB, UB, UB, UB, UB, UB] | None = None,  # noqa: E501
        n_samples: int = 1_000
    ) -> Self:
        """Construct a distribution from the observational covariance matrix.

        Parameters
        ----------
        orbit : Orbit
            The orbit instance. This is the mean of the distribution.
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
        # TODO: relax this and have the cov apply to each orbit
        if orbit.shape != ():
            raise ValueError("Orbit must be a single orbit!")
        # TODO: orbit cannot be integrated

        mean, _u = _get_mean(orbit, "icrs", cov_units)
        coords = multivariate_normal(mean=mean, cov=cov).rvs(n_samples)

        # TODO: add the original orbit as well
        ounc = Orbit(coords, radec=True,
                     ro=orbit._ro, vo=orbit._vo, zo=orbit._zo,
                     solarmotion=orbit._solarmotion)
        if hasattr(orbit, "_name"):
            ounc.__dict__["_name"] = orbit._name

        return cls(ounc)

    @classmethod
    def from_galactocentric_cov(
        cls: type[Self],
        orbit: Orbit,
        cov: np.ndarray,
        *,
        cov_units: tuple[UB, UB, UB, UB, UB, UB] | None = None,  # noqa: E501
        n_samples: int = 1_000
    ) -> Self:
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
        # TODO: relax this and have the cov apply to each orbit
        if orbit.shape != ():
            raise ValueError("Orbit must be a single orbit!")
        # TODO! orbit cannot be integrated

        mean, _u = _get_mean(orbit, "galactocentric", cov_units)
        xyz = multivariate_normal(mean=mean, cov=cov).rvs(n_samples)

        R, phi, z = rect_to_cyl(*(xyz[:, i] * _u[i] for i in range(3)))
        vR,vT,vz = rect_to_cyl_vec(
            *(xyz[:, i] * _u[i] for i in range(3, 6)),
            R, phi, z, cyl=True
        )

        # TODO: add the original orbit as well
        ounc = Orbit(
            [R, vR, vT, z, vz, phi],
            ro=orbit._ro, vo=orbit._vo, zo=orbit._zo,
            solarmotion=orbit._solarmotion
        )
        if hasattr(orbit, "_name"):
            ounc.__dict__["_name"] = orbit._name

        return cls(ounc)

    @classmethod
    def from_fit(
        cls: type[Self],
        init_vxvv: np.ndarray,
        vxvv: np.ndarray,
        vxvv_err: np.ndarray | None = None,
        pot: Potential | None = None,
        radec: bool = False,
        lb: bool = False,
        customsky: bool = False,  # TODO
        lb_to_customsky=None,  # TODO
        pmllpmbb_to_customsky=None,  # TODO
        tintJ: int = 10,
        ntintJ: int = 1000,
        integrate_method: str = 'dopr54_c',
        ro: float | Quantity | None = None,
        vo: float | Quantity | None = None,
        zo: float | Quantity | None = None,
        solarmotion: Sequence[float] | Quantity | None = None,
        disp: bool = False,
    ) -> Self:
        raise NotImplementedError("TODO")
        # TODO: use the vxvv_err to construct the covariance matrix
