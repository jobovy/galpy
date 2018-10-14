from .Orbit import Orbit
from ..util.multi import parallel_map
try:
    from astropy.coordinates import SkyCoord
    _APY_LOADED = True
except ImportError:
    SkyCoord = None
    _APY_LOADED = False


class Orbits:
    """
    Class used to contain multiple Orbit instances.
    """
    def __init__(self, vxvv=None, radec=False, uvw=False, lb=False, ro=None,
                 vo=None, zo=None, solarmotion=None):
        """
        NAME:

            __init__

        PURPOSE:

            Initialize an Orbits instance

        INPUT:

            vxvv - initial conditions; can be either

                a) list of Orbit instances

                b) astropy (>v3.0) SkyCoord including velocities (note that this turns *on* physical output even if ro and vo are not given)

                c) list of initial conditions for individual Orbit instances; elements can be either

                    1) in Galactocentric cylindrical coordinates [R,vR,vT(,z,vz,phi)]; can be Quantities

                    2) [ra,dec,d,mu_ra, mu_dec,vlos] in [deg,deg,kpc,mas/yr,mas/yr,km/s] (all J2000.0; mu_ra = mu_ra * cos dec); can be Quantities; ICRS frame

                    3) [ra,dec,d,U,V,W] in [deg,deg,kpc,km/s,km/s,kms]; can be Quantities; ICRS frame

                    4) [l,b,d,mu_l,mu_b,vlos] in [deg,deg,kpc,mas/yr,mas/yr,km/s) (all J2000.0; mu_l = mu_l * cos b); can be Quantities

                    5) [l,b,d,U,V,W] in [deg,deg,kpc,km/s,km/s,kms]; can be Quantities

                    6) None: assumed to be the Sun (equivalent to ``[0,0,0,0,0,0]`` and ``radec=True``)

                4) and 5) also work when leaving out b and mu_b/W

        OPTIONAL INPUTS:

            radec - if True, input is 2) or 3) above (note that this turns *on* physical output even if ro and vo are not given)

            uvw - if True, velocities are UVW

            lb - if True, input is 4) or 5) above (note that this turns *on* physical output even if ro and vo are not given)

            ro - distance from vantage point to GC (kpc; can be Quantity)

            vo - circular velocity at ro (km/s; can be Quantity)

            zo - offset toward the NGP of the Sun wrt the plane (kpc; can be Quantity; default = 25 pc)

            solarmotion - 'hogg' or 'dehnen', or 'schoenrich', or value in [-U,V,W]; can be Quantity

        OUTPUT:

            instance

        HISTORY:

            XXXX-XX-XX - Written - Mathew Bub (UofT)

        """
        if vxvv is None:
            vxvv = []

        if _APY_LOADED and isinstance(vxvv, SkyCoord):
            self._orbits = [Orbit(vxvv=coord) for coord in vxvv.flatten()]
        else:
            self._orbits = []
            for coord in vxvv:
                if isinstance(coord, Orbit):
                    self._orbits.append(coord)
                else:
                    orbit = Orbit(vxvv=coord, radec=radec, uvw=uvw, lb=lb,
                                  ro=ro, vo=vo, zo=zo, solarmotion=solarmotion)
                    self._orbits.append(orbit)

    def __getattr__(self, name):
        """
        NAME:

            __getattr__

        PURPOSE:

            get or evaluate an attribute for these Orbits

        INPUT:

            name - name of the attribute

        OUTPUT:

            if the attribute is callable, a function to evaluate the attribute for each Orbit; otherwise a list of attributes

        HISTORY:

            XXXX-XX-XX - Written - Mathew Bub (UofT)

        """
        attribute = getattr(Orbit(), name)
        if callable(attribute):
            return lambda *args, **kwargs: [
                getattr(orbit, name)(*args, **kwargs) for orbit in self._orbits
            ]
        else:
            return [getattr(orbit, name) for orbit in self.orbits]

    def integrate(self, t, pot, method='symplec4_c', dt=None, numcores=1):
        """
        NAME:

            integrate

        PURPOSE:

            integrate these Orbits with multiprocessing

        INPUT:

            t - list of times at which to output (0 has to be in this!) (can be Quantity)

            pot - potential instance or list of instances

            method = 'odeint' for scipy's odeint
                     'leapfrog' for a simple leapfrog implementation
                     'leapfrog_c' for a simple leapfrog implementation in C
                     'symplec4_c' for a 4th order symplectic integrator in C
                     'symplec6_c' for a 6th order symplectic integrator in C
                     'rk4_c' for a 4th-order Runge-Kutta integrator in C
                     'rk6_c' for a 6-th order Runge-Kutta integrator in C
                     'dopr54_c' for a Dormand-Prince integrator in C (generally the fastest)

            dt - if set, force the integrator to use this basic stepsize; must be an integer divisor of output stepsize (only works for the C integrators that use a fixed stepsize) (can be Quantity)

            numcores - number of cores to use for multiprocessing; default = 1

        OUTPUT:

            None (get the actual orbit using getOrbit())

        HISTORY:

            XXXX-XX-XX - Written - Mathew Bub (UofT)

        """
        # Must return each Orbit for its values to correctly update
        def integrate(orbit):
            orbit.integrate(t, pot, method=method, dt=dt)
            return orbit

        self._orbits = list(parallel_map(integrate, self._orbits,
                                         numcores=numcores))
