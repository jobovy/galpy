import warnings

import numpy

from ..df.df import df
from ..orbit import Orbit
from ..potential import MovingObjectPotential, evaluateRforces, rtide
from ..potential.Potential import _check_potential_list_and_deprecate
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
        leading=None,
        tail=None,
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
        pot : galpy.potential.Potential or a combined potential formed using addition (pot1+pot2+…), optional
            Potential for integrating orbits.
        rtpot : galpy.potential.Potential or a combined potential formed using addition (pot1+pot2+…), optional
            Potential for calculating tidal radius and circular velocity (should generally be the same as pot, but sometimes you need to drop parts of the potential that don't allow the tidal radius / circular velocity to be computed, such as velocity-dependent forces; when using center, rtpot should be the relevant potential in the frame of the center, thus, also being different from pot).
        tdisrupt : float or Quantity, optional
            Time since start of disruption. Default is 5 Gyr.
        leading : bool, optional
            Deprecated since v1.12. Use ``tail`` instead. If True, model the leading part of the stream. If False, model the trailing part.
        tail : str, optional
            Which tail(s) to model. Can be ``'leading'``, ``'trailing'``, or ``'both'``. Default is ``'leading'``.
        center : galpy.orbit.Orbit, optional
            Orbit instance that represents the center around which the progenitor is orbiting for the purpose of stream formation; allows for a stream to be generated from a progenitor orbiting a moving object, like a satellite galaxy. Integrated internally using centerpot.
        centerpot : galpy.potential.Potential or a combined potential formed using addition (pot1+pot2+…), optional
            Potential for calculating the orbit of the center; this might be different from the potential that the progenitor is integrated in if, for example, dynamical friction is important for the orbit of the center (if it's a satellite).
        progpot : galpy.potential.Potential or a combined potential formed using addition (pot1+pot2+…) or None, optional
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
        # Handle leading= deprecation
        if leading is not None:
            warnings.warn(
                "The leading= keyword is deprecated since v1.12 and will be "
                "removed in v1.14. Use tail= instead: tail='leading' or "
                "tail='trailing'.",
                FutureWarning,
                stacklevel=2,
            )
            if tail is not None:
                raise ValueError(
                    "Cannot specify both leading= and tail=. Use tail= only."
                )
            tail = "leading" if leading else "trailing"
        if tail is None:
            tail = "leading"
        if tail not in ("leading", "trailing", "both"):
            raise ValueError(
                f"tail= must be 'leading', 'trailing', or 'both', got '{tail}'"
            )
        self._tail = tail
        self._leading = tail != "trailing"
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
        self._pot = _check_potential_list_and_deprecate(pot)
        self._rtpot = (
            self._pot if rtpot is None else _check_potential_list_and_deprecate(rtpot)
        )
        assert conversion.physical_compatible(self, self._pot), (
            "Physical conversion for the potential is not consistent with that of the basestreamspraydf object being initialized"
        )
        assert conversion.physical_compatible(self, self._rtpot), (
            "Physical conversion for the rt potential is not consistent with that of the basestreamspraydf object being initialized"
        )
        # Set up progenitor orbit
        assert conversion.physical_compatible(self, progenitor), (
            "Physical conversion for the progenitor Orbit object is not consistent with that of the basestreamspraydf object being initialized"
        )
        self._orig_progenitor = progenitor  # Store so we can use its ro/vo/etc.
        self._progenitor = progenitor()
        self._progenitor.turn_physical_off()
        self._progenitor_times = numpy.linspace(0.0, -self._tdisrupt, 10001)
        self._progenitor.integrate(self._progenitor_times, self._pot)
        # Set up center orbit if given
        if not center is None:
            self._centerpot = (
                self._pot
                if centerpot is None
                else _check_potential_list_and_deprecate(centerpot)
            )
            assert conversion.physical_compatible(self, self._centerpot), (
                "Physical conversion for the center potential is not consistent with that of the basestreamspraydf object being initialized"
            )
            self._center = center()
            self._center.turn_physical_off()
            self._center.integrate(self._progenitor_times, self._centerpot)
        else:
            self._center = None
        if progpot is not None:
            self._orig_pot = self._pot  # save pre-progpot for streamTrack
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
            Number of points to return. When ``tail='both'``, ``n`` is the total number of points, split equally between the leading and trailing tails.
        return_orbit : bool, optional
            If True, the output phase-space positions is an orbit.Orbit object. If False, the output is (R,vR,vT,z,vz,phi). Default is True.
        returndt : bool, optional
            If True, also return the time since the star was stripped. Default is False.
        integrate : bool, optional
            If True, integrate the orbits to the present time. If False, return positions at stripping (probably want to combine with returndt=True then to make sense of them!). Default is True.

        Returns
        -------
        Orbit, numpy.ndarray, or tuple
            Orbit instance or (R,vR,vT,z,vz,phi) of points on the stream in 6,N array (set of 6 Quantities when physical output is on); optionally the time is included as well. When ``tail='both'``, the leading-tail points come first, followed by the trailing-tail points. The ro/vo unit-conversion parameters and the zo/solarmotion parameters as well as whether physical outputs are on, match the settings of the progenitor Orbit given to the class initialization

        Notes
        -----
        - 2018-07-31 - Written - Bovy (UofT)
        - 2022-05-18 - Made output Orbit ro/vo/zo/solarmotion/roSet/voSet match that of the progenitor orbit - Bovy (UofT)
        - 2024-08-11 - Include the progenitor's potential - Yingtian Chen (Umich)
        """
        if self._tail == "both":
            n_leading = n // 2
            n_trailing = n - n_leading
            out_l, dt_l = self._sample_tail(n_leading, integrate, leading=True)
            out_t, dt_t = self._sample_tail(n_trailing, integrate, leading=False)
            out = numpy.hstack([out_l, out_t])
            dt = numpy.concatenate([dt_l, dt_t])
        else:
            out, dt = self._sample_tail(n, integrate, leading=self._tail == "leading")
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

    def streamTrack(
        self,
        n=5000,
        particles=None,
        tail=None,
        track_time_range=None,
        track_n_dense=10001,
        ntp=None,
        ninterp=1001,
        smoothing=None,
        niter=0,
        order=2,
        custom_transform=None,
    ):
        """
        Construct a smooth phase-space track through the stream by sampling
        particles and projecting them onto a finely-integrated progenitor
        orbit.

        The track is parameterized by the progenitor's time coordinate
        ``tp``: ``tp=0`` is the progenitor today, ``tp<0`` are past
        positions (matched by the trailing arm) and ``tp>0`` are future
        positions (matched by the leading arm). Because stream particles
        have small velocity offsets from the progenitor, they lie spatially
        close to a short arc of the progenitor's orbit — the relevant ``tp``
        range is much smaller than ``tdisrupt``.

        Parameters
        ----------
        n : int, optional
            Number of particles to draw (per arm when ``tail='both'``).
            Ignored if ``particles`` is provided. Default is 5000.
        particles : tuple, optional
            Pre-computed ``(xv, dt)`` from ``self.sample(returndt=True,
            return_orbit=False, integrate=True)``. When ``tail='both'``,
            particles must follow the sample ordering (leading first,
            then trailing). Default is None (sample freshly).
        tail : str, optional
            One of ``'leading'``, ``'trailing'``, or ``'both'``. Defaults to
            the value set at initialization.
        track_time_range : float or Quantity, optional
            Half-range (symmetric about tp=0) of the finely-integrated
            progenitor orbit used for closest-point matching. Default is
            data-driven: ``8 * d_max / |v_prog|`` clamped to ``[1,
            tdisrupt]``, where ``d_max`` is the farthest particle's
            distance from the progenitor.
        track_n_dense : int, optional
            Number of time points on the finely-integrated progenitor
            orbit. The actual grid has ``2 * (track_n_dense+1)//2 - 1``
            points (forward + backward with shared t=0). Default 10001.
        ntp : int, optional
            Number of binning nodes. Default ``sqrt(N)`` clipped to
            ``[21, 201]``.
        ninterp : int, optional
            Resolution of the public fine-grid track arrays. Default 1001.
        smoothing : None, float, array-like, or dict, optional
            Smoothing parameter(s). ``None`` (default) uses GCV
            auto-tuning. A float sets a single ``s`` for all coords. An
            array-like of length 6 (mean only) or 27 (mean + covariance)
            sets per-spline ``s`` values — pass a previous call's
            ``track.smoothing_s`` to reproduce the same smoothness
            without re-running GCV. A dict keyed by coordinate name
            sets per-coordinate ``s`` for the 6 mean splines only.
        niter : int, optional
            Iterations beyond the initial fit. Each iteration reassigns
            particles to the closest point on the current track.
        order : int, optional
            1 = mean only, 2 = mean + covariance (default).

        Returns
        -------
        :class:`galpy.df.StreamTrack` or :class:`galpy.df.StreamTrackPair`
            A single-arm track object, or a pair with ``.leading`` and
            ``.trailing`` tracks when ``tail='both'``.

        Notes
        -----
        - 2026-04-14 - Written - Bovy (UofT)
        """
        from ..orbit import Orbit
        from .streamTrack import StreamTrack, StreamTrackPair

        tail = self._tail if tail is None else tail
        if tail not in ("leading", "trailing", "both"):
            raise ValueError(
                f"tail= must be 'leading', 'trailing', or 'both', got '{tail}'"
            )
        if track_time_range is None:
            # Auto: estimate from the stream's spatial extent. Draw a
            # probe sample (or reuse provided particles), measure the
            # farthest particle from the progenitor, convert to an
            # orbital-time scale via the progenitor's present-day speed,
            # and pad by 8x. This scales naturally with stream width
            # (essential for warm / dwarf-galaxy-mass progenitors whose
            # tidal radii and velocity kicks are much larger).
            if particles is not None:
                _xv_probe = particles[0]
            else:
                _xv_probe, _ = self._sample_tail(
                    min(n, 500), True, leading=(tail != "trailing")
                )
            _Rs, _, _, _zs, _, _phis = _xv_probe
            _xs = _Rs * numpy.cos(_phis)
            _ys = _Rs * numpy.sin(_phis)
            _px = float(self._progenitor.x(0.0))
            _py = float(self._progenitor.y(0.0))
            _pz = float(self._progenitor.z(0.0))
            _pv = numpy.sqrt(
                float(self._progenitor.vx(0.0)) ** 2
                + float(self._progenitor.vy(0.0)) ** 2
                + float(self._progenitor.vz(0.0)) ** 2
            )
            _d_max = numpy.sqrt(
                numpy.max((_xs - _px) ** 2 + (_ys - _py) ** 2 + (_zs - _pz) ** 2)
            )
            track_time_range = float(
                numpy.clip(8.0 * _d_max / max(_pv, 1e-6), 1.0, self._tdisrupt)
            )
        else:
            track_time_range = conversion.parse_time(
                track_time_range, ro=self._ro, vo=self._vo
            )

        # Build a finely-sampled progenitor phase-space array spanning
        # [-T, +T] around the present day. Integrate forward and backward
        # separately from the progenitor's present-day state, then combine.
        # Use the BASE potential (without progpot's MovingObjectPotential)
        # because the MovingObjectPotential's internal progenitor was only
        # integrated on [-tdisrupt, 0] and would give wrong or erroring
        # results for positive times.
        _track_pot = self._pot
        if hasattr(self, "_orig_pot"):
            _track_pot = self._orig_pot
        half_dense = (int(track_n_dense) + 1) // 2
        t_back = numpy.linspace(0.0, -track_time_range, half_dense)
        t_fwd = numpy.linspace(0.0, track_time_range, half_dense)
        prog_back = self._orig_progenitor()
        prog_back.turn_physical_off()
        prog_back.integrate(t_back, _track_pot)
        prog_fwd = self._orig_progenitor()
        prog_fwd.turn_physical_off()
        prog_fwd.integrate(t_fwd, _track_pot)
        # Combine, skipping the t=0 duplicate
        track_t_grid = numpy.concatenate([t_back[::-1], t_fwd[1:]])
        track_prog_cart = numpy.column_stack(
            [
                numpy.concatenate([prog_back.x(t_back)[::-1], prog_fwd.x(t_fwd)[1:]]),
                numpy.concatenate([prog_back.y(t_back)[::-1], prog_fwd.y(t_fwd)[1:]]),
                numpy.concatenate([prog_back.z(t_back)[::-1], prog_fwd.z(t_fwd)[1:]]),
                numpy.concatenate([prog_back.vx(t_back)[::-1], prog_fwd.vx(t_fwd)[1:]]),
                numpy.concatenate([prog_back.vy(t_back)[::-1], prog_fwd.vy(t_fwd)[1:]]),
                numpy.concatenate([prog_back.vz(t_back)[::-1], prog_fwd.vz(t_fwd)[1:]]),
            ]
        )

        # Inherited unit metadata from the original progenitor Orbit
        prog_ro = self._orig_progenitor._ro
        prog_vo = self._orig_progenitor._vo
        prog_zo = self._orig_progenitor._zo
        prog_sm = self._orig_progenitor._solarmotion
        prog_roSet = self._orig_progenitor._roSet
        prog_voSet = self._orig_progenitor._voSet

        def _make_track(xv, dt, arm_sign):
            return StreamTrack(
                xv_particles=xv,
                dt_particles=dt,
                track_prog_cart=track_prog_cart,
                track_t_grid=track_t_grid,
                arm_sign=arm_sign,
                ntp=ntp,
                ninterp=ninterp,
                smoothing=smoothing,
                niter=niter,
                order=order,
                custom_transform=custom_transform,
                ro=prog_ro,
                vo=prog_vo,
                zo=prog_zo,
                solarmotion=prog_sm,
                roSet=prog_roSet,
                voSet=prog_voSet,
            )

        if tail == "both":
            if particles is not None:
                xv, dt = particles
                n_lead = len(dt) // 2
                xv_l = xv[:, :n_lead]
                dt_l = dt[:n_lead]
                xv_t = xv[:, n_lead:]
                dt_t = dt[n_lead:]
            else:
                xv_l, dt_l = self._sample_tail(n, True, leading=True)
                xv_t, dt_t = self._sample_tail(n, True, leading=False)
            return StreamTrackPair(
                _make_track(xv_l, dt_l, arm_sign=+1),
                _make_track(xv_t, dt_t, arm_sign=-1),
            )
        else:
            if particles is not None:
                xv, dt = particles
            else:
                xv, dt = self._sample_tail(n, True, leading=(tail == "leading"))
            return _make_track(xv, dt, arm_sign=(+1 if tail == "leading" else -1))

    def _sample_tail(self, n, integrate, leading=True):
        """Sample n points from the specified tail."""
        # First sample times
        dt = numpy.random.uniform(size=n) * self._tdisrupt
        # Build all rotation matrices
        rot, rot_inv = self._setup_rot(dt)
        # Compute progenitor position in the instantaneous frame,
        # relative to the center orbit if necessary
        centerx = numpy.atleast_1d(self._progenitor.x(-dt))
        centery = numpy.atleast_1d(self._progenitor.y(-dt))
        centerz = numpy.atleast_1d(self._progenitor.z(-dt))
        centervx = numpy.atleast_1d(self._progenitor.vx(-dt))
        centervy = numpy.atleast_1d(self._progenitor.vy(-dt))
        centervz = numpy.atleast_1d(self._progenitor.vz(-dt))
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
        xst, yst, zst, vxst, vyst, vzst = self.spray_df(xyzpt, vxyzpt, dt, leading)

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
            # Integrate all sampled particles as a single Orbit instance, with
            # each particle on its own time grid from its stripping time -dt[i]
            # to the present (t=0). The final time step is the present-day state.
            o = Orbit(numpy.array([Rs, vRs, vTs, Zs, vZs, phis]).T)
            ts = numpy.linspace(-dt, numpy.zeros(n), 10001, axis=-1)
            o.integrate(ts, self._pot)
            out[:] = o.orbit[:, -1, :].T
        else:
            out[0] = Rs
            out[1] = vRs
            out[2] = vTs
            out[3] = Zs
            out[4] = vZs
            out[5] = phis
        return out, dt

    def _setup_rot(self, dt):
        n = len(dt)
        centerx = numpy.atleast_1d(self._progenitor.x(-dt))
        centery = numpy.atleast_1d(self._progenitor.y(-dt))
        centerz = numpy.atleast_1d(self._progenitor.z(-dt))
        if self._center is None:
            L = numpy.atleast_2d(self._progenitor.L(-dt))
        # Compute relative angular momentum to the center orbit
        else:
            centerx -= self._center.x(-dt)
            centery -= self._center.y(-dt)
            centerz -= self._center.z(-dt)
            centervx = self._progenitor.vx(-dt) - self._center.vx(-dt)
            centervy = self._progenitor.vy(-dt) - self._center.vy(-dt)
            centervz = self._progenitor.vz(-dt) - self._center.vz(-dt)
            L = numpy.atleast_2d(
                numpy.array(
                    [
                        centery * centervz - centerz * centervy,
                        centerz * centervx - centerx * centervz,
                        centerx * centervy - centery * centervx,
                    ]
                ).T
            )
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

    def spray_df(self, xyzpt, vxyzpt, dt, leading=True):
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
        leading : bool, optional
            If True, generate the leading tail. If False, generate the trailing tail. Default is True.

        Returns
        -------
        xst, yst, zst : array, shape (N,)
            Positions of points on the stream in the progenitor coordinates.
        vxst, vyst, vzst : array, shape (N,)
            Velocities of points on the stream in the progenitor coordinates.
        """
        raise NotImplementedError


class chen24spraydf(basestreamspraydf):
    def __init__(
        self,
        progenitor_mass,
        progenitor=None,
        pot=None,
        rtpot=None,
        tdisrupt=None,
        leading=None,
        tail=None,
        center=None,
        centerpot=None,
        progpot=None,
        mean=None,
        cov=None,
        ro=None,
        vo=None,
    ):
        """
        Initialize a `Chen et al. (2024) <https://ui.adsabs.harvard.edu/abs/2024arXiv240801496C/abstract>`_ stream spray DF model of a tidal stream.


        Parameters
        ----------
        progenitor_mass : float or Quantity
            Mass of the progenitor.
        progenitor : galpy.orbit.Orbit, optional
            Progenitor orbit as Orbit instance (will be re-integrated, so don't bother integrating the orbit before).
        pot : galpy.potential.Potential or a combined potential formed using addition (pot1+pot2+…), optional
            Potential for integrating orbits.
        rtpot : galpy.potential.Potential or a combined potential formed using addition (pot1+pot2+…), optional
            Potential for calculating tidal radius and circular velocity (should generally be the same as pot, but sometimes you need to drop parts of the potential that don't allow the tidal radius / circular velocity to be computed, such as velocity-dependent forces; when using center, rtpot should be the relevant potential in the frame of the center, thus, also being different from pot).
        tdisrupt : float or Quantity, optional
            Time since start of disruption. Default is 5 Gyr.
        leading : bool, optional
            Deprecated since v1.12. Use ``tail`` instead. If True, model the leading part of the stream. If False, model the trailing part.
        tail : str, optional
            Which tail(s) to model. Can be ``'leading'``, ``'trailing'``, or ``'both'``. Default is ``'leading'``.
        center : galpy.orbit.Orbit, optional
            Orbit instance that represents the center around which the progenitor is orbiting for the purpose of stream formation; allows for a stream to be generated from a progenitor orbiting a moving object, like a satellite galaxy. Integrated internally using centerpot.
        centerpot : galpy.potential.Potential or a combined potential formed using addition (pot1+pot2+…), optional
            Potential for calculating the orbit of the center; this might be different from the potential that the progenitor is integrated in if, for example, dynamical friction is important for the orbit of the center (if it's a satellite).
        progpot : galpy.potential.Potential or a combined potential formed using addition (pot1+pot2+…) or None, optional
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
            tail=tail,
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

    def spray_df(self, xyzpt, vxyzpt, dt, leading=True):
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
        leading : bool, optional
            If True, generate the leading tail. If False, generate the trailing tail. Default is True.

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
        if leading:
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
        leading=None,
        tail=None,
        center=None,
        centerpot=None,
        progpot=None,
        meankvec=[2.0, 0.0, 0.3, 0.0, 0.0, 0.0],
        sigkvec=[0.4, 0.0, 0.4, 0.5, 0.5, 0.0],
        ro=None,
        vo=None,
    ):
        """
        Initialize a `Fardal et al. (2015) <https://ui.adsabs.harvard.edu/abs/2014arXiv1410.1861F/abstract>`_ stream spray DF model of a tidal stream.


        Parameters
        ----------
        progenitor_mass : float or Quantity
            Mass of the progenitor.
        progenitor : galpy.orbit.Orbit, optional
            Progenitor orbit as Orbit instance (will be re-integrated, so don't bother integrating the orbit before).
        pot : galpy.potential.Potential or a combined potential formed using addition (pot1+pot2+…), optional
            Potential for integrating orbits.
        rtpot : galpy.potential.Potential or a combined potential formed using addition (pot1+pot2+…), optional
            Potential for calculating tidal radius and circular velocity (should generally be the same as pot, but sometimes you need to drop parts of the potential that don't allow the tidal radius / circular velocity to be computed, such as velocity-dependent forces; when using center, rtpot should be the relevant potential in the frame of the center, thus, also being different from pot).
        tdisrupt : float or Quantity, optional
            Time since start of disruption. Default is 5 Gyr.
        leading : bool, optional
            Deprecated since v1.12. Use ``tail`` instead. If True, model the leading part of the stream. If False, model the trailing part.
        tail : str, optional
            Which tail(s) to model. Can be ``'leading'``, ``'trailing'``, or ``'both'``. Default is ``'leading'``.
        center : galpy.orbit.Orbit, optional
            Orbit instance that represents the center around which the progenitor is orbiting for the purpose of stream formation; allows for a stream to be generated from a progenitor orbiting a moving object, like a satellite galaxy. Integrated internally using centerpot.
        centerpot : galpy.potential.Potential or a combined potential formed using addition (pot1+pot2+…), optional
            Potential for calculating the orbit of the center; this might be different from the potential that the progenitor is integrated in if, for example, dynamical friction is important for the orbit of the center (if it's a satellite).
        progpot : galpy.potential.Potential or a combined potential formed using addition (pot1+pot2+…) or None, optional
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
            tail=tail,
            center=center,
            centerpot=centerpot,
            progpot=progpot,
            ro=ro,
            vo=vo,
        )
        self._meankvec = numpy.array(meankvec)
        self._sigkvec = numpy.array(sigkvec)
        return None

    def spray_df(self, xyzpt, vxyzpt, dt, leading=True):
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
        leading : bool, optional
            If True, generate the leading tail. If False, generate the trailing tail. Default is True.

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
        meankvec = -self._meankvec if leading else self._meankvec
        k = meankvec + numpy.random.normal(size=(len(dt), 6)) * self._sigkvec

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
