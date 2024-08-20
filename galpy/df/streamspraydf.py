import warnings

import numpy

from ..df.df import df
from ..orbit import Orbit
from ..potential import MovingObjectPotential, evaluateRforces
from ..potential import flatten as flatten_potential
from ..potential import rtide
from ..util import _rotate_to_arbitrary_vector, conversion, coords
from ..util._optional_deps import _APY_LOADED, _APY_UNITS

if _APY_LOADED:
    from astropy import units


class basestreamspraydf(df):
    def __init__(
        self,
        progenitor_mass,
        progenitor=None,
        pot=None,
        rtpot=None,
        tdisrupt=None,
        leading=True,
        center=None,
        centerpot=None,
        progpot=None,
        ro=None,
        vo=None,
    ):
        """
        Initialize a stream spray DF model of a tidal stream

        Parameters
        ----------
        progenitor_mass : float or Quantity
            Mass of the progenitor.
        progenitor : galpy.orbit.Orbit, optional
            Progenitor orbit as Orbit instance (will be re-integrated, so don't bother integrating the orbit before).
        pot : galpy.potential.Potential or list of such instances, optional
            Potential for integrating orbits.
        rtpot : galpy.potential.Potential or list of such instances, optional
            Potential for calculating tidal radius and circular velocity (should generally be the same as pot, but sometimes you need to drop parts of the potential that don't allow the tidal radius / circular velocity to be computed, such as velocity-dependent forces; when using center, rtpot should be the relevant potential in the frame of the center, thus, also being different from pot).
        tdisrupt : float or Quantity, optional
            Time since start of disruption. Default is 5 Gyr.
        leading : bool, optional
            If True, model the leading part of the stream. If False, model the trailing part. Default is True.
        center : galpy.orbit.Orbit, optional
            Orbit instance that represents the center around which the progenitor is orbiting for the purpose of stream formation; allows for a stream to be generated from a progenitor orbiting a moving object, like a satellite galaxy. Integrated internally using centerpot.
        centerpot : galpy.potential.Potential or list of such instances, optional
            Potential for calculating the orbit of the center; this might be different from the potential that the progenitor is integrated in if, for example, dynamical friction is important for the orbit of the center (if it's a satellite).
        progpot : galpy.potential.Potential or list of such instances or None, optional
            Potential for the progenitor. Ignored if None.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2018-07-31 - Written - Bovy (UofT)
        - 2021-05-05 - Added center keyword - Yansong Qian (UofT)
        - 2024-08-11 - Generalized to allow different particle-spray methods - Yingtian Chen (UMich)
        """
        super().__init__(ro=ro, vo=vo)
        self._progenitor_mass = conversion.parse_mass(
            progenitor_mass, ro=self._ro, vo=self._vo
        )
        self._tdisrupt = (
            5.0 / conversion.time_in_Gyr(self._vo, self._ro)
            if tdisrupt is None
            else conversion.parse_time(tdisrupt, ro=self._ro, vo=self._vo)
        )
        if pot is None:  # pragma: no cover
            raise OSError("pot= must be set")
        self._pot = flatten_potential(pot)
        self._rtpot = self._pot if rtpot is None else flatten_potential(rtpot)
        assert conversion.physical_compatible(
            self, self._pot
        ), "Physical conversion for the potential is not consistent with that of the basestreamspraydf object being initialized"
        assert conversion.physical_compatible(
            self, self._rtpot
        ), "Physical conversion for the rt potential is not consistent with that of the basestreamspraydf object being initialized"
        # Set up progenitor orbit
        assert conversion.physical_compatible(
            self, progenitor
        ), "Physical conversion for the progenitor Orbit object is not consistent with that of the basestreamspraydf object being initialized"
        self._orig_progenitor = progenitor  # Store so we can use its ro/vo/etc.
        self._progenitor = progenitor()
        self._progenitor.turn_physical_off()
        self._progenitor_times = numpy.linspace(0.0, -self._tdisrupt, 10001)
        self._progenitor.integrate(self._progenitor_times, self._pot)
        self._leading = leading
        # Set up center orbit if given
        if not center is None:
            self._centerpot = (
                self._pot if centerpot is None else flatten_potential(centerpot)
            )
            assert conversion.physical_compatible(
                self, self._centerpot
            ), "Physical conversion for the center potential is not consistent with that of the basestreamspraydf object being initialized"
            self._center = center()
            self._center.turn_physical_off()
            self._center.integrate(self._progenitor_times, self._centerpot)
        else:
            self._center = None
        if progpot is not None:
            progtrajpot = MovingObjectPotential(
                orbit=self._progenitor,
                pot=progpot,
                ro=self._ro,
                vo=self._vo,
            )
            self._pot = self._pot + progtrajpot

        return None

    def sample(self, n, return_orbit=True, returndt=False, integrate=True):
        """
        Sample from the DF

        Parameters
        ----------
        n : int
            Number of points to return.
        return_orbit : bool, optional
            If True, the output phase-space positions is an orbit.Orbit object. If False, the output is (R,vR,vT,z,vz,phi). Default is True.
        returndt : bool, optional
            If True, also return the time since the star was stripped. Default is False.
        integrate : bool, optional
            If True, integrate the orbits to the present time. If False, return positions at stripping (probably want to combine with returndt=True then to make sense of them!). Default is True.

        Returns
        -------
        Orbit, numpy.ndarray, or tuple
            Orbit instance or (R,vR,vT,z,vz,phi) of points on the stream in 6,N array (set of 6 Quantities when physical output is on); optionally the time is included as well. The ro/vo unit-conversion parameters and the zo/solarmotion parameters as well as whether physical outputs are on, match the settings of the progenitor Orbit given to the class initialization

        Notes
        -----
        - 2018-07-31 - Written - Bovy (UofT)
        - 2022-05-18 - Made output Orbit ro/vo/zo/solarmotion/roSet/voSet match that of the progenitor orbit - Bovy (UofT)
        - 2024-08-11 - Include the progenitor's potential - Yingtian Chen (Umich)
        """
        # First sample times
        dt = numpy.random.uniform(size=n) * self._tdisrupt
        # Build all rotation matrices
        rot, rot_inv = self._setup_rot(dt)
        # Compute progenitor position in the instantaneous frame,
        # relative to the center orbit if necessary
        centerx = self._progenitor.x(-dt)
        centery = self._progenitor.y(-dt)
        centerz = self._progenitor.z(-dt)
        centervx = self._progenitor.vx(-dt)
        centervy = self._progenitor.vy(-dt)
        centervz = self._progenitor.vz(-dt)
        if not self._center is None:
            centerx -= self._center.x(-dt)
            centery -= self._center.y(-dt)
            centerz -= self._center.z(-dt)
            centervx -= self._center.vx(-dt)
            centervy -= self._center.vy(-dt)
            centervz -= self._center.vz(-dt)
        xyzpt = numpy.einsum(
            "ijk,ik->ij", rot, numpy.array([centerx, centery, centerz]).T
        )
        vxyzpt = numpy.einsum(
            "ijk,ik->ij", rot, numpy.array([centervx, centervy, centervz]).T
        )

        # generate the initial conditions
        xst, yst, zst, vxst, vyst, vzst = self.spray_df(xyzpt, vxyzpt, dt)

        xyzs = numpy.einsum("ijk,ik->ij", rot_inv, numpy.array([xst, yst, zst]).T)
        vxyzs = numpy.einsum("ijk,ik->ij", rot_inv, numpy.array([vxst, vyst, vzst]).T)

        absx = xyzs[:, 0]
        absy = xyzs[:, 1]
        absz = xyzs[:, 2]
        absvx = vxyzs[:, 0]
        absvy = vxyzs[:, 1]
        absvz = vxyzs[:, 2]
        if not self._center is None:
            absx += self._center.x(-dt)
            absy += self._center.y(-dt)
            absz += self._center.z(-dt)
            absvx += self._center.vx(-dt)
            absvy += self._center.vy(-dt)
            absvz += self._center.vz(-dt)
        Rs, phis, Zs = coords.rect_to_cyl(absx, absy, absz)
        vRs, vTs, vZs = coords.rect_to_cyl_vec(
            absvx, absvy, absvz, Rs, phis, Zs, cyl=True
        )
        out = numpy.empty((6, n))
        if integrate:
            # Now integrate the orbits
            for ii in range(n):
                o = Orbit([Rs[ii], vRs[ii], vTs[ii], Zs[ii], vZs[ii], phis[ii]])
                o.integrate(numpy.linspace(-dt[ii], 0.0, 10001), self._pot)
                o = o(0.0)
                out[:, ii] = [o.R(), o.vR(), o.vT(), o.z(), o.vz(), o.phi()]
        else:
            out[0] = Rs
            out[1] = vRs
            out[2] = vTs
            out[3] = Zs
            out[4] = vZs
            out[5] = phis
        if return_orbit:
            # Output Orbit ro/vo/zo/solarmotion/roSet/voSet match progenitor
            o = Orbit(
                vxvv=out.T,
                ro=self._orig_progenitor._ro,
                vo=self._orig_progenitor._vo,
                zo=self._orig_progenitor._zo,
                solarmotion=self._orig_progenitor._solarmotion,
            )
            if not self._orig_progenitor._roSet:
                o._roSet = False
            if not self._orig_progenitor._voSet:
                o._voSet = False
            out = o
        elif _APY_UNITS and self._voSet and self._roSet:
            out = (
                out[0] * self._ro * units.kpc,
                out[1] * self._vo * units.km / units.s,
                out[2] * self._vo * units.km / units.s,
                out[3] * self._ro * units.kpc,
                out[4] * self._vo * units.km / units.s,
                out[5] * units.rad,
            )
            dt = dt * conversion.time_in_Gyr(self._vo, self._ro) * units.Gyr
        if returndt:
            return (out, dt)
        else:
            return out

    def _setup_rot(self, dt):
        n = len(dt)
        centerx = self._progenitor.x(-dt)
        centery = self._progenitor.y(-dt)
        centerz = self._progenitor.z(-dt)
        if self._center is None:
            L = self._progenitor.L(-dt)
        # Compute relative angular momentum to the center orbit
        else:
            centerx -= self._center.x(-dt)
            centery -= self._center.y(-dt)
            centerz -= self._center.z(-dt)
            centervx = self._progenitor.vx(-dt) - self._center.vx(-dt)
            centervy = self._progenitor.vy(-dt) - self._center.vy(-dt)
            centervz = self._progenitor.vz(-dt) - self._center.vz(-dt)
            L = numpy.array(
                [
                    centery * centervz - centerz * centervy,
                    centerz * centervx - centerx * centervz,
                    centerx * centervy - centery * centervx,
                ]
            ).T
        Lnorm = L / numpy.tile(numpy.sqrt(numpy.sum(L**2.0, axis=1)), (3, 1)).T
        z_rot = numpy.swapaxes(
            _rotate_to_arbitrary_vector(
                numpy.atleast_2d(Lnorm), [0.0, 0.0, 1], inv=True
            ),
            1,
            2,
        )
        z_rot_inv = numpy.swapaxes(
            _rotate_to_arbitrary_vector(
                numpy.atleast_2d(Lnorm), [0.0, 0.0, 1], inv=False
            ),
            1,
            2,
        )
        xyzt = numpy.einsum(
            "ijk,ik->ij", z_rot, numpy.array([centerx, centery, centerz]).T
        )
        Rt = numpy.sqrt(xyzt[:, 0] ** 2.0 + xyzt[:, 1] ** 2.0)
        cosphi, sinphi = xyzt[:, 0] / Rt, xyzt[:, 1] / Rt
        pa_rot = numpy.array(
            [
                [cosphi, -sinphi, numpy.zeros(n)],
                [sinphi, cosphi, numpy.zeros(n)],
                [numpy.zeros(n), numpy.zeros(n), numpy.ones(n)],
            ]
        ).T
        pa_rot_inv = numpy.array(
            [
                [cosphi, sinphi, numpy.zeros(n)],
                [-sinphi, cosphi, numpy.zeros(n)],
                [numpy.zeros(n), numpy.zeros(n), numpy.ones(n)],
            ]
        ).T
        rot = numpy.einsum("ijk,ikl->ijl", pa_rot, z_rot)
        rot_inv = numpy.einsum("ijk,ikl->ijl", z_rot_inv, pa_rot_inv)
        return (rot, rot_inv)

    def _calc_rtide(self, Rpt, phipt, Zpt, dt):
        try:
            rtides = rtide(
                self._rtpot,
                Rpt,
                Zpt,
                phi=phipt,
                t=-dt,
                M=self._progenitor_mass,
                use_physical=False,
            )
        except (ValueError, TypeError):
            rtides = numpy.array(
                [
                    rtide(
                        self._rtpot,
                        Rpt[ii],
                        Zpt[ii],
                        phi=phipt[ii],
                        t=-dt[ii],
                        M=self._progenitor_mass,
                        use_physical=False,
                    )
                    for ii in range(len(Rpt))
                ]
            )
        return rtides

    def _calc_vc(self, Rpt, phipt, Zpt, dt):
        try:
            vcs = numpy.sqrt(
                -Rpt
                * evaluateRforces(
                    self._rtpot, Rpt, Zpt, phi=phipt, t=-dt, use_physical=False
                )
            )
        except (ValueError, TypeError):
            vcs = numpy.array(
                [
                    numpy.sqrt(
                        -Rpt[ii]
                        * evaluateRforces(
                            self._rtpot,
                            Rpt[ii],
                            Zpt[ii],
                            phi=phipt[ii],
                            t=-dt[ii],
                            use_physical=False,
                        )
                    )
                    for ii in range(len(Rpt))
                ]
            )
        return vcs

    def spray_df(self, xyzpt, vxyzpt, dt):
        """
        Sample the positions and velocities around the progenitor
        Must be implemented in a subclass

        Parameters
        ----------
        xyzpt : array, shape (N,3)
            Positions of progenitor in the progenitor coordinates.
        vxyzpt : array, shape (N,3)
            Velocities of progenitor in the progenitor coordinates.
        dt : array, shape (N,)
            Time of sampling.

        Returns
        -------
        xst, yst, zst : array, shape (N,)
            Positions of points on the stream in the progenitor coordinates.
        vxst, vyst, vzst : array, shape (N,)
            Velocities of points on the stream in the progenitor coordinates.
        """
        warnings.warn("Not implemented!", NotImplementedError, stacklevel=1)
        pass


class chen24spraydf(basestreamspraydf):
    def __init__(
        self,
        progenitor_mass,
        progenitor=None,
        pot=None,
        rtpot=None,
        tdisrupt=None,
        leading=True,
        center=None,
        centerpot=None,
        progpot=None,
        mean=None,
        cov=None,
        ro=None,
        vo=None,
    ):
        """
        Initialize a Chen+24 stream spray DF model of a tidal stream
        https://ui.adsabs.harvard.edu/abs/2024arXiv240801496C/abstract

        Parameters
        ----------
        progenitor_mass : float or Quantity
            Mass of the progenitor.
        progenitor : galpy.orbit.Orbit, optional
            Progenitor orbit as Orbit instance (will be re-integrated, so don't bother integrating the orbit before).
        pot : galpy.potential.Potential or list of such instances, optional
            Potential for integrating orbits.
        rtpot : galpy.potential.Potential or list of such instances, optional
            Potential for calculating tidal radius and circular velocity (should generally be the same as pot, but sometimes you need to drop parts of the potential that don't allow the tidal radius / circular velocity to be computed, such as velocity-dependent forces; when using center, rtpot should be the relevant potential in the frame of the center, thus, also being different from pot).
        tdisrupt : float or Quantity, optional
            Time since start of disruption. Default is 5 Gyr.
        leading : bool, optional
            If True, model the leading part of the stream. If False, model the trailing part. Default is True.
        center : galpy.orbit.Orbit, optional
            Orbit instance that represents the center around which the progenitor is orbiting for the purpose of stream formation; allows for a stream to be generated from a progenitor orbiting a moving object, like a satellite galaxy. Integrated internally using centerpot.
        centerpot : galpy.potential.Potential or list of such instances, optional
            Potential for calculating the orbit of the center; this might be different from the potential that the progenitor is integrated in if, for example, dynamical friction is important for the orbit of the center (if it's a satellite).
        progpot : galpy.potential.Potential or list of such instances or None, optional
            Potential for the progenitor. Ignored if None.
        mean : None or array, shape (6,), optional
            Means of the multivariate Gaussian distribution (angles in radians). If None, use the default values.
        cov : None or array, shape (6,6), optional
            Covariance of the multivariate Gaussian distribution (angles in radians). If None, use the default values.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2024-08-11 - Written - Yingtian Chen (UMich)
        """
        super().__init__(
            progenitor_mass=progenitor_mass,
            progenitor=progenitor,
            pot=pot,
            rtpot=rtpot,
            tdisrupt=tdisrupt,
            leading=leading,
            center=center,
            centerpot=centerpot,
            progpot=progpot,
            ro=ro,
            vo=vo,
        )
        if mean is None:
            self._mean = numpy.array([1.6, -0.523599, 0, 1, 0.349066, 0])
        else:
            self._mean = mean
        if cov is None:
            self._cov = numpy.array(
                [
                    [0.1225, 0, 0, 0, -0.085521, 0],
                    [0, 0.161143, 0, 0, 0, 0],
                    [0, 0, 0.043865, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [-0.085521, 0, 0, 0, 0.121847, 0],
                    [0, 0, 0, 0, 0, 0.147435],
                ]
            )
        else:
            self._cov = cov
        return None

    def spray_df(self, xyzpt, vxyzpt, dt):
        """
        Sample the positions and velocities around the progenitor

        Parameters
        ----------
        xyzpt : array, shape (N,3)
            Positions of progenitor in the progenitor coordinates.
        vxyzpt : array, shape (N,3)
            Velocities of progenitor in the progenitor coordinates.
        dt : array, shape (N,)
            Time of sampling.

        Returns
        -------
        xst, yst, zst : array, shape (N,)
            Positions of points on the stream in the progenitor coordinates.
        vxst, vyst, vzst : array, shape (N,)
            Velocities of points on the stream in the progenitor coordinates.
        """
        Rpt, phipt, Zpt = coords.rect_to_cyl(xyzpt[:, 0], xyzpt[:, 1], xyzpt[:, 2])
        rtides = self._calc_rtide(Rpt, phipt, Zpt, dt)

        # Sample positions and velocities in the instantaneous frame
        posvel = numpy.random.multivariate_normal(self._mean, self._cov, size=len(dt))
        Dr = posvel[:, 0] * rtides
        v_esc = numpy.sqrt(2 * self._progenitor_mass / Dr)
        Dv = posvel[:, 3] * v_esc
        if self._leading:
            Dr *= -1.0
            Dv *= -1.0

        dR, dz, dp = coords.spher_to_cyl(
            r=Dr, theta=0.5 * numpy.pi - posvel[:, 2], phi=posvel[:, 1]
        )
        dx, dy, dz = coords.cyl_to_rect(R=dR, phi=dp, Z=dz)

        dvR, dvz, dvp = coords.spher_to_cyl(
            r=Dv, theta=0.5 * numpy.pi - posvel[:, 5], phi=posvel[:, 4]
        )
        dvx, dvy, dvz = coords.cyl_to_rect(R=dvR, phi=dvp, Z=dvz)

        return (
            xyzpt[:, 0] + dx,
            xyzpt[:, 1] + dy,
            xyzpt[:, 2] + dz,
            vxyzpt[:, 0] + dvx,
            vxyzpt[:, 1] + dvy,
            vxyzpt[:, 2] + dvz,
        )


class fardal15spraydf(basestreamspraydf):
    def __init__(
        self,
        progenitor_mass,
        progenitor=None,
        pot=None,
        rtpot=None,
        tdisrupt=None,
        leading=True,
        center=None,
        centerpot=None,
        progpot=None,
        meankvec=[2.0, 0.0, 0.3, 0.0, 0.0, 0.0],
        sigkvec=[0.4, 0.0, 0.4, 0.5, 0.5, 0.0],
        ro=None,
        vo=None,
    ):
        """
        Initialize a Fardal+15 stream spray DF model of a tidal stream
        https://ui.adsabs.harvard.edu/abs/2014arXiv1410.1861F/abstract

        Parameters
        ----------
        progenitor_mass : float or Quantity
            Mass of the progenitor.
        progenitor : galpy.orbit.Orbit, optional
            Progenitor orbit as Orbit instance (will be re-integrated, so don't bother integrating the orbit before).
        pot : galpy.potential.Potential or list of such instances, optional
            Potential for integrating orbits.
        rtpot : galpy.potential.Potential or list of such instances, optional
            Potential for calculating tidal radius and circular velocity (should generally be the same as pot, but sometimes you need to drop parts of the potential that don't allow the tidal radius / circular velocity to be computed, such as velocity-dependent forces; when using center, rtpot should be the relevant potential in the frame of the center, thus, also being different from pot).
        tdisrupt : float or Quantity, optional
            Time since start of disruption. Default is 5 Gyr.
        leading : bool, optional
            If True, model the leading part of the stream. If False, model the trailing part. Default is True.
        center : galpy.orbit.Orbit, optional
            Orbit instance that represents the center around which the progenitor is orbiting for the purpose of stream formation; allows for a stream to be generated from a progenitor orbiting a moving object, like a satellite galaxy. Integrated internally using centerpot.
        centerpot : galpy.potential.Potential or list of such instances, optional
            Potential for calculating the orbit of the center; this might be different from the potential that the progenitor is integrated in if, for example, dynamical friction is important for the orbit of the center (if it's a satellite).
        progpot : galpy.potential.Potential or list of such instances or None, optional
            Potential for the progenitor. Ignored if None.
        meankvec : list or array, optional
            Mean of the action-angle distribution. Default is [2.0, 0.0, 0.3, 0.0, 0.0, 0.0].
        sigkvec : list or array, optional
            Dispersion of the action-angle distribution. Default is [0.4, 0.0, 0.4, 0.5, 0.5, 0.0].
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2018-07-31 - Written - Bovy (UofT)
        - 2021-05-05 - Added center keyword - Yansong Qian (UofT)
        """
        super().__init__(
            progenitor_mass=progenitor_mass,
            progenitor=progenitor,
            pot=pot,
            rtpot=rtpot,
            tdisrupt=tdisrupt,
            leading=leading,
            center=center,
            centerpot=centerpot,
            progpot=progpot,
            ro=ro,
            vo=vo,
        )
        self._meankvec = numpy.array(meankvec)
        self._sigkvec = numpy.array(sigkvec)
        if leading:
            self._meankvec *= -1.0
        return None

    def spray_df(self, xyzpt, vxyzpt, dt):
        """
        Sample the positions and velocities around the progenitor

        Parameters
        ----------
        xyzpt : array, shape (N,3)
            Positions of progenitor in the progenitor coordinates.
        vxyzpt : array, shape (N,3)
            Velocities of progenitor in the progenitor coordinates.
        dt : array, shape (N,)
            Time of sampling.

        Returns
        -------
        xst, yst, zst : array, shape (N,)
            Positions of points on the stream in the progenitor coordinates.
        vxst, vyst, vzst : array, shape (N,)
            Velocities of points on the stream in the progenitor coordinates.
        """
        Rpt, phipt, Zpt = coords.rect_to_cyl(xyzpt[:, 0], xyzpt[:, 1], xyzpt[:, 2])
        rtides = self._calc_rtide(Rpt, phipt, Zpt, dt)
        vcs = self._calc_vc(Rpt, phipt, Zpt, dt)
        rtides_as_frac = rtides / Rpt

        vRpt, vTpt, vZpt = coords.rect_to_cyl_vec(
            vxyzpt[:, 0], vxyzpt[:, 1], vxyzpt[:, 2], Rpt, phipt, Zpt, cyl=True
        )
        # Sample positions and velocities in the instantaneous frame
        k = self._meankvec + numpy.random.normal(size=(len(dt), 6)) * self._sigkvec

        RpZst = numpy.array(
            [
                Rpt + k[:, 0] * rtides,
                phipt + k[:, 5] * rtides_as_frac,
                k[:, 3] * rtides_as_frac,
            ]
        ).T
        vRTZst = numpy.array(
            [
                vRpt * (1.0 + k[:, 1]),
                vTpt + k[:, 2] * vcs * rtides_as_frac,
                k[:, 4] * vcs * rtides_as_frac,
            ]
        ).T
        # Now rotate these back to the galactocentric frame
        xst, yst, zst = coords.cyl_to_rect(RpZst[:, 0], RpZst[:, 1], RpZst[:, 2])
        vxst, vyst, vzst = coords.cyl_to_rect_vec(
            vRTZst[:, 0], vRTZst[:, 1], vRTZst[:, 2], RpZst[:, 1]
        )

        return xst, yst, zst, vxst, vyst, vzst


class streamspraydf(fardal15spraydf):
    def __init__(self, args, **kwargs):
        """
        For backward compatibility
        """
        super().__init__(args, **kwargs)
        warnings.warn(
            "Class `streamspraydf` will be deprecated in version 1.11. "
            "Please use class `fardal15spraydf` for the Fardal+15 particle spray model.",
            DeprecationWarning,
            stacklevel=1,
        )
        return None
