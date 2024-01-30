import numpy

from ..df.df import df
from ..orbit import Orbit
from ..potential import evaluateRforces
from ..potential import flatten as flatten_potential
from ..potential import rtide
from ..util import _rotate_to_arbitrary_vector, conversion, coords
from ..util._optional_deps import _APY_LOADED, _APY_UNITS

if _APY_LOADED:
    from astropy import units


class streamspraydf(df):
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
        meankvec=[2.0, 0.0, 0.3, 0.0, 0.0, 0.0],
        sigkvec=[0.4, 0.0, 0.4, 0.5, 0.5, 0.0],
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
        df.__init__(self, ro=ro, vo=vo)
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
        ), "Physical conversion for the potential is not consistent with that of the streamspraydf object being initialized"
        assert conversion.physical_compatible(
            self, self._rtpot
        ), "Physical conversion for the rt potential is not consistent with that of the streamspraydf object being initialized"
        # Set up progenitor orbit
        assert conversion.physical_compatible(
            self, progenitor
        ), "Physical conversion for the progenitor Orbit object is not consistent with that of the streamspraydf object being initialized"
        self._orig_progenitor = progenitor  # Store so we can use its ro/vo/etc.
        self._progenitor = progenitor()
        self._progenitor.turn_physical_off()
        self._progenitor_times = numpy.linspace(0.0, -self._tdisrupt, 10001)
        self._progenitor.integrate(self._progenitor_times, self._pot)
        self._meankvec = numpy.array(meankvec)
        self._sigkvec = numpy.array(sigkvec)
        # Set up center orbit if given
        if not center is None:
            self._centerpot = (
                self._pot if centerpot is None else flatten_potential(centerpot)
            )
            assert conversion.physical_compatible(
                self, self._centerpot
            ), "Physical conversion for the center potential is not consistent with that of the streamspraydf object being initialized"
            self._center = center()
            self._center.turn_physical_off()
            self._center.integrate(self._progenitor_times, self._centerpot)
        else:
            self._center = None
        if leading:
            self._meankvec *= -1.0
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
        Rpt, phipt, Zpt = coords.rect_to_cyl(xyzpt[:, 0], xyzpt[:, 1], xyzpt[:, 2])
        vRpt, vTpt, vZpt = coords.rect_to_cyl_vec(
            vxyzpt[:, 0], vxyzpt[:, 1], vxyzpt[:, 2], Rpt, phipt, Zpt, cyl=True
        )
        # Sample positions and velocities in the instantaneous frame
        k = self._meankvec + numpy.random.normal(size=(n, 6)) * self._sigkvec
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
            vcs = numpy.sqrt(
                -Rpt
                * evaluateRforces(
                    self._rtpot, Rpt, Zpt, phi=phipt, t=-dt, use_physical=False
                )
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
        rtides_as_frac = rtides / Rpt
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
