import warnings

import numpy
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import find_peaks

from ..backend import (
    as_backend_constant,
    as_numpy,
    get_namespace,
    is_backend_array,
    use,
)
from ..df.df import df
from ..orbit import Orbit
from ..potential import MovingObjectPotential, evaluateRforces, rtide
from ..potential.Potential import _check_potential_list_and_deprecate
from ..util import _rotate_to_arbitrary_vector, conversion, coords
from ..util._optional_deps import _APY_LOADED, _APY_UNITS
from .streamTrack import StreamTrack, StreamTrackPair

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
        stripping_pdf=None,
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
        progenitor_mass : float, Quantity, or callable
            Mass of the progenitor. If a callable, it is a function ``M(t)`` of the progenitor-time coordinate (``t=0`` is now, ``t<0`` is the past, matching the convention used throughout galpy's orbit integration). The callable may take and/or return astropy ``Quantity`` (auto-detected): unitful input is given in Gyr, unitful output should have units of mass.
        progenitor : galpy.orbit.Orbit, optional
            Progenitor orbit as Orbit instance (will be re-integrated, so don't bother integrating the orbit before).
        pot : galpy.potential.Potential or a combined potential formed using addition (pot1+pot2+…), optional
            Potential for integrating orbits.
        rtpot : galpy.potential.Potential or a combined potential formed using addition (pot1+pot2+…), optional
            Potential for calculating tidal radius and circular velocity (should generally be the same as pot, but sometimes you need to drop parts of the potential that don't allow the tidal radius / circular velocity to be computed, such as velocity-dependent forces; when using center, rtpot should be the relevant potential in the frame of the center, thus, also being different from pot).
        tdisrupt : float or Quantity, optional
            Time since start of disruption. Default is 5 Gyr.
        stripping_pdf : callable, optional
            Probability density of stripping over the progenitor time axis ``t in [-tdisrupt, 0]`` (``t=0`` is the present, ``t<0`` the past). Must accept a 1D array (or scalar) and return a 1D array (or scalar) of the same length. Both input and output may be astropy ``Quantity`` (input a time, output a 1/time); detection mirrors :class:`galpy.potential.AnyAxisymmetricRazorThinDiskPotential`. The PDF need not be normalized. Default is None (uniform stripping over ``[-tdisrupt, 0]``).
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
        - 2026-05-11 - Allowed ``progenitor_mass`` to be a callable - Bovy (UofT)
        """
        # If ro/vo are not explicitly given, inherit them from the
        # progenitor's settings so that streamspraydf and progenitor
        # share a single physical-conversion convention. (zo/solarmotion
        # don't enter the spray itself; streamTrack sources them from
        # the progenitor at construction time.)
        if progenitor is not None:
            if ro is None and progenitor._roSet:
                ro = progenitor._ro
            if vo is None and progenitor._voSet:
                vo = progenitor._vo
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
        self._parse_progenitor_mass(progenitor_mass)
        self._tdisrupt = (
            5.0 / conversion.time_in_Gyr(self._vo, self._ro)
            if tdisrupt is None
            else conversion.parse_time(tdisrupt, ro=self._ro, vo=self._vo)
        )
        self._parse_stripping_pdf(stripping_pdf)
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

    def sample(
        self, n, return_orbit=True, returndt=False, integrate=True, tail=None, key=None
    ):
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
        tail : str, optional
            ``'leading'``, ``'trailing'``, or ``'both'`` to override the default set at class initialization. Default is None (use the value of ``tail=`` from ``__init__``). The progenitor is integrated identically for either arm, so any override value works regardless of the initialization choice.
        key : optional
            Backend random key from :func:`galpy.backend.random.key`. Default None uses the global ``numpy.random`` draws (byte-identical to previous behaviour). A jax/torch key makes the stripping-time draws reproducible backend arrays from the key (common-random-numbers), the seam a future differentiable-sampling PR builds on.

        Returns
        -------
        Orbit, numpy.ndarray, or tuple
            Orbit instance or (R,vR,vT,z,vz,phi) of points on the stream in 6,N array (set of 6 Quantities when physical output is on); optionally the time is included as well. When ``tail='both'``, the leading-tail points come first, followed by the trailing-tail points. The ro/vo unit-conversion parameters and the zo/solarmotion parameters as well as whether physical outputs are on, match the settings of the progenitor Orbit given to the class initialization

        Notes
        -----
        - 2018-07-31 - Written - Bovy (UofT)
        - 2022-05-18 - Made output Orbit ro/vo/zo/solarmotion/roSet/voSet match that of the progenitor orbit - Bovy (UofT)
        - 2024-08-11 - Include the progenitor's potential - Yingtian Chen (Umich)
        - 2026-04-28 - Added ``tail`` keyword override to match ``streamTrack`` - Bovy (UofT)
        """
        tail = self._tail if tail is None else tail
        if tail not in ("leading", "trailing", "both"):
            raise ValueError(
                f"tail= must be 'leading', 'trailing', or 'both', got '{tail}'"
            )
        if tail == "both":
            n_leading = n // 2
            n_trailing = n - n_leading
            # Independent sub-keys per arm (numpy: split -> (None, None), so each
            # arm draws sequentially from the global generator: byte-identical).
            from ..backend import random as grandom

            key_l, key_t = grandom.split(key, 2)
            out_l, dt_l = self._sample_tail(
                n_leading, integrate, leading=True, key=key_l
            )
            out_t, dt_t = self._sample_tail(
                n_trailing, integrate, leading=False, key=key_t
            )
            if is_backend_array(out_l):
                out = get_namespace(out_l).hstack([out_l, out_t])
            else:
                out = numpy.hstack([out_l, out_t])
            dt = numpy.concatenate([dt_l, dt_t])
        else:
            out, dt = self._sample_tail(
                n, integrate, leading=tail == "leading", key=key
            )
        if return_orbit:
            # Output Orbit ro/vo/zo/solarmotion/roSet/voSet match progenitor
            o = Orbit(
                vxvv=as_numpy(out).T,
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
            out = as_numpy(out)  # astropy Quantity can't hold a backend array
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
        ntp=None,
        smoothing=None,
        smoothing_factor=1.0,
        niter=0,
        order=2,
        velocity_weight="auto",
        custom_sky_transform=None,
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
            Total number of particles to draw. When ``tail='both'``, ``n``
            is split equally between leading and trailing (matching
            ``self.sample(n, ...)``'s convention). Ignored if
            ``particles`` is provided. Default is 5000.
        particles : array, shape (6, N), optional
            Pre-computed present-day ``(R, vR, vT, z, vz, phi)`` of stream
            particles. Use ``self.sample(returndt=False, return_orbit=False,
            integrate=True)`` to draw, or pass an externally-generated
            sample (e.g. from an N-body run). When ``tail='both'``, the
            array must follow the sample ordering (leading first, then
            trailing) and is split at ``N // 2``. Default is None
            (sample freshly).
        tail : str, optional
            One of ``'leading'``, ``'trailing'``, or ``'both'``. Defaults to
            the value set at initialization.
        track_time_range : float or Quantity, optional
            Half-range (symmetric about tp=0) of the finely-integrated
            progenitor orbit used for closest-point matching. Default is
            data-driven: ``8 * d_max / |v_prog|`` clamped to ``[1,
            tdisrupt]``, where ``d_max`` is the farthest particle's
            distance from the progenitor.
        ntp : int, optional
            Number of binning nodes. Default ``sqrt(N)`` with a floor of
            21 and a ceiling that scales with the arc span (at least 201;
            larger for long streams).
        smoothing : None, float, or array-like, optional
            Smoothing parameter(s). ``None`` (default) uses GCV
            auto-tuning. A float sets a single ``s`` for all coords. An
            array-like of length 6 (mean only) or 27 (mean + covariance)
            sets per-spline ``s`` values — pass a previous call's
            ``track.smoothing_s`` to reproduce the same smoothness
            without re-running GCV.
        smoothing_factor : float, optional
            Multiplier applied to every spline's effective ``s`` after
            GCV (or explicit-``s``) selection. Values > 1 force a smoother
            fit, values < 1 a rougher one. Useful when GCV undersmooths
            in finite samples (a common failure mode of
            ``make_smoothing_spline`` on noisy binned means). Default 1.0.
            For an interactive smoothing sweep, save ``track.particles``
            from the first call and pass it back as ``particles=`` —
            only the cheap re-fit step runs, the orbit-integration sample
            is reused.
        niter : int, optional
            Iterations beyond the initial fit. Each iteration reassigns
            particles to the closest point on the current track.
        order : int, optional
            1 = mean only, 2 = mean + covariance (default).
        velocity_weight : float or ``'auto'``, optional
            Multiplicative weight applied to velocity components when
            computing 6D distances during the closest-point projection.
            Default ``'auto'`` learns the weight from the inner-half
            particle dispersion (``σ_pos / σ_vel``, clipped to
            ``[0.1, 10]``); typically lands at ~2–3 for both clean and
            perturbed streams. Values > 1 make velocity matches more
            important than position matches — useful when the
            progenitor orbit revisits regions of phase space (e.g., in
            strongly-perturbed potentials with a massive LMC). Pass
            ``1.0`` for the legacy unweighted natural-units
            metric.

        Returns
        -------
        :class:`galpy.df.StreamTrack` or :class:`galpy.df.StreamTrackPair`
            A single-arm track object, or a pair with ``.leading`` and
            ``.trailing`` tracks when ``tail='both'``.

        Notes
        -----
        - 2026-04-14 - Written - Bovy (UofT)
        """
        tail = self._tail if tail is None else tail
        if tail not in ("leading", "trailing", "both"):
            raise ValueError(
                f"tail= must be 'leading', 'trailing', or 'both', got '{tail}'"
            )

        # Resolve the particle sample(s) up front. For tail='both' we keep
        # the leading/trailing split — the time-range estimate below pools
        # all of them for a tight bound.
        if tail == "both":
            if particles is not None:
                # A backend (jax/torch) particles array flows through the fit
                # differentiably; a numpy/list input is coerced as before.
                xv_all = (
                    particles
                    if is_backend_array(particles)
                    else numpy.asarray(particles, dtype=float)
                )
                n_lead = xv_all.shape[1] // 2
                xv_lead = xv_all[:, :n_lead]
                xv_trail = xv_all[:, n_lead:]
            else:
                # Match self.sample(n, ...)'s split: half leading, half
                # trailing (with the trailing tail picking up the parity
                # bit when n is odd, like sample()).
                n_lead = n // 2
                n_trail = n - n_lead
                xv_lead, _ = self._sample_tail(n_lead, True, leading=True)
                xv_trail, _ = self._sample_tail(n_trail, True, leading=False)
                xv_all = numpy.column_stack([xv_lead, xv_trail])
        else:
            if particles is not None:
                xv_single = (
                    particles
                    if is_backend_array(particles)
                    else numpy.asarray(particles, dtype=float)
                )
            else:
                xv_single, _ = self._sample_tail(n, True, leading=(tail == "leading"))
            xv_all = xv_single

        if track_time_range is None:
            # Auto: estimate from the stream's spatial extent in the
            # already-sampled particles, measure the farthest from the
            # progenitor, convert to an orbital-time scale via the
            # progenitor's present-day speed, and pad by 8x. Scales
            # naturally with stream width (essential for warm /
            # dwarf-galaxy-mass progenitors whose tidal radii and
            # velocity kicks are much larger).
            # Structural extent estimate (a scalar time-range bound); run on a
            # numpy view so a backend (jax/torch) particles array doesn't turn
            # these numpy reductions into namespace ops.
            _Rs, _, _, _zs, _, _phis = as_numpy(xv_all)
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
        # [-T, +T] around the present day. Integrate forward, then
        # backward — galpy's Orbit.integrate auto-stitches consecutive
        # calls into a single continuous trajectory.
        # Use the base potential (no MovingObjectPotential for the
        # progenitor itself — a body shouldn't generate the field that
        # integrates it).
        _track_pot = getattr(self, "_orig_pot", self._pot)
        # Dense progenitor sampling: hard-coded internal density. 10001
        # points across [-T, +T] is plenty for any plausible track —
        # finer than the spline knot density downstream.
        half_dense = 5001
        t_fwd = numpy.linspace(0.0, track_time_range, half_dense)
        t_back = numpy.linspace(0.0, -track_time_range, half_dense)
        # Stitched grid spans [-T, +T] (skip the t=0 duplicate at the seam).
        track_t_grid = numpy.concatenate([t_back[::-1], t_fwd[1:]])
        prog_ic_backend = getattr(self._orig_progenitor, "_ic_backend", None)
        # A backend (jax/torch) potential PARAMETER makes the force a backend array.
        # Probe at the progenitor's present-day phase-space point -- a valid, non-
        # degenerate (R, phi, z, v) so the probe also works for non-axisymmetric
        # (phi required) and dissipative (v required) track potentials.
        _pp = self._progenitor
        _theta_force = evaluateRforces(
            _track_pot,
            float(_pp.R(0.0)),
            float(_pp.z(0.0)),
            phi=float(_pp.phi(0.0)),
            v=numpy.array([float(_pp.vR(0.0)), float(_pp.vT(0.0)), float(_pp.vz(0.0))]),
        )
        _theta_backend = is_backend_array(_theta_force)

        def _backend_curve(ic, method, xp):
            # Integrate the progenitor fwd+back and stitch into a differentiable
            # backend track_prog_cart (torch has no negative-step slice -> xp.flip).
            o_f = Orbit(ic)
            o_f.turn_physical_off()
            o_f.integrate(t_fwd, _track_pot, method=method)
            o_b = Orbit(ic)
            o_b.turn_physical_off()
            o_b.integrate(t_back, _track_pot, method=method)

            def _cart(o, ts):
                return xp.stack(
                    [o.x(ts), o.y(ts), o.z(ts), o.vx(ts), o.vy(ts), o.vz(ts)], axis=-1
                )

            return xp.concat(
                [xp.flip(_cart(o_b, t_back), axis=0), _cart(o_f, t_fwd)[1:]], axis=0
            )

        if _theta_backend:
            # Backend potential parameter: integrate the progenitor via the
            # in-backend ODE (diffrax/torchdiffeq) so the fitted track carries
            # d(track)/d(theta) -- the C-STM carries no parameter sensitivity.
            # Coerce the progenitor IC onto the potential's backend (a backend IC
            # additionally flows d(track)/d(prog IC)). A backend track_prog_cart
            # takes precedence over prog_orbit in StreamTrack (prog_orbit=None).
            xp = get_namespace(_theta_force)
            if is_backend_array(prog_ic_backend):
                ic = prog_ic_backend
            else:
                p0 = self._orig_progenitor()
                p0.turn_physical_off()
                ic = xp.asarray(
                    numpy.array(
                        [
                            float(p0.R()),
                            float(p0.vR()),
                            float(p0.vT()),
                            float(p0.z()),
                            float(p0.vz()),
                            float(p0.phi()),
                        ]
                    )
                )
            method = "diffrax" if "jax" in xp.__name__ else "torchdiffeq"
            track_prog_cart = _backend_curve(ic, method, xp)
            prog = None
        elif is_backend_array(prog_ic_backend):
            # Backend progenitor IC (numpy-parameter potential): the differentiable
            # C integrator (dop853_c -> C-STM when the potential has the C 3D
            # Hessian, else the in-backend ODE) carries d(track)/d(prog IC). A
            # backend track_prog_cart takes precedence over prog_orbit below.
            track_prog_cart = _backend_curve(
                prog_ic_backend, "dop853_c", get_namespace(prog_ic_backend)
            )
            prog = None
        else:
            prog = self._orig_progenitor()
            prog.turn_physical_off()
            prog.integrate(t_fwd, _track_pot)
            prog.integrate(t_back, _track_pot)
            track_prog_cart = numpy.column_stack(
                [
                    prog.x(track_t_grid),
                    prog.y(track_t_grid),
                    prog.z(track_t_grid),
                    prog.vx(track_t_grid),
                    prog.vy(track_t_grid),
                    prog.vz(track_t_grid),
                ]
            )

        # Inherit unit metadata from the original progenitor Orbit. Pass
        # ``ro``/``vo`` only when the progenitor had them explicitly set —
        # StreamTrack mirrors Orbit's "ro/vo unset means use the config
        # default and keep _roSet=False" pattern, so we propagate the
        # progenitor's ``_roSet``/``_voSet`` state via *not passing* the
        # value rather than via a separate flag.
        prog_ro = self._orig_progenitor._ro if self._orig_progenitor._roSet else None
        prog_vo = self._orig_progenitor._vo if self._orig_progenitor._voSet else None
        prog_zo = self._orig_progenitor._zo
        prog_sm = self._orig_progenitor._solarmotion

        def _make_track(xv, arm_sign):
            return StreamTrack.from_particles(
                xv_particles=xv,
                track_prog_cart=track_prog_cart,
                track_t_grid=track_t_grid,
                arm_sign=arm_sign,
                ntp=ntp,
                smoothing=smoothing,
                smoothing_factor=smoothing_factor,
                niter=niter,
                order=order,
                velocity_weight=velocity_weight,
                prog_orbit=prog,
                custom_sky_transform=custom_sky_transform,
                ro=prog_ro,
                vo=prog_vo,
                zo=prog_zo,
                solarmotion=prog_sm,
            )

        if tail == "both":
            return StreamTrackPair(
                _make_track(xv_lead, arm_sign=+1),
                _make_track(xv_trail, arm_sign=-1),
            )
        return _make_track(xv_single, arm_sign=(+1 if tail == "leading" else -1))

    def _parse_stripping_pdf(self, stripping_pdf):
        if stripping_pdf is None:
            self._stripping_inv_cdf = None
            return
        if not callable(stripping_pdf):
            raise TypeError("stripping_pdf must be callable or None")
        # Detect Quantity input/output the same way
        # AnyAxisymmetricRazorThinDiskPotential does.
        time_in_gyr = conversion.time_in_Gyr(self._vo, self._ro)
        _t_unit_input = False
        _t_unit_output = False
        if _APY_LOADED:
            t_probe = -0.5 * self._tdisrupt
            try:
                stripping_pdf(t_probe)
            except (
                units.UnitConversionError,
                units.UnitTypeError,
                AttributeError,
            ):
                _t_unit_input = True
            if _t_unit_input:
                try:
                    stripping_pdf(t_probe * time_in_gyr * units.Gyr).to(1.0 / units.Gyr)
                except (AttributeError, units.UnitConversionError):
                    pass
                else:
                    _t_unit_output = True
            else:
                try:
                    stripping_pdf(t_probe).to(1.0 / units.Gyr)
                except (AttributeError, units.UnitConversionError):
                    pass
                else:
                    _t_unit_output = True
        if _t_unit_input and _t_unit_output:

            def pdf_internal(t):
                out = stripping_pdf(t * time_in_gyr * units.Gyr)
                return out.to(1.0 / units.Gyr).value * time_in_gyr

        elif _t_unit_input:

            def pdf_internal(t):
                return numpy.asarray(stripping_pdf(t * time_in_gyr * units.Gyr))

        elif _t_unit_output:

            def pdf_internal(t):
                out = stripping_pdf(t)
                return out.to(1.0 / units.Gyr).value * time_in_gyr

        else:

            def pdf_internal(t):
                return numpy.asarray(stripping_pdf(t))

        t_grid = numpy.linspace(-self._tdisrupt, 0.0, 10001)
        pdf_vals = numpy.asarray(pdf_internal(t_grid), dtype=float)
        if numpy.any(pdf_vals < 0):
            raise ValueError("stripping_pdf must be non-negative on [-tdisrupt, 0]")
        cdf_vals = cumulative_trapezoid(pdf_vals, t_grid, initial=0.0)
        if cdf_vals[-1] <= 0:
            raise ValueError("stripping_pdf integrates to zero on [-tdisrupt, 0]")
        cdf_vals /= cdf_vals[-1]
        # Enforce strict monotonicity by accumulating max and dropping ties.
        cdf_vals = numpy.maximum.accumulate(cdf_vals)
        _, unique_idx = numpy.unique(cdf_vals, return_index=True)
        unique_idx = numpy.sort(unique_idx)
        self._stripping_inv_cdf = InterpolatedUnivariateSpline(
            cdf_vals[unique_idx], t_grid[unique_idx], k=1, ext=3
        )

    def _draw_stripping_dt(self, n, key=None):
        """Draw n stripping-time offsets ``dt >= 0``.

        The single random seam of the spray's stripping-time sampling. numpy
        (``key is None``) is byte-identical to the historical
        ``numpy.random.uniform`` draw; a jax/torch ``key`` from
        :func:`galpy.backend.random.key` returns a reproducible backend array
        (the default uniform-stripping path). The ``stripping_pdf`` inverse-CDF
        is a scipy spline, so that path evaluates on the numpy sample for now (a
        backend-native inverse-CDF is a later PR).
        """
        from ..backend import as_numpy
        from ..backend import random as grandom

        if self._stripping_inv_cdf is None:
            return grandom.uniform(key, (n,)) * self._tdisrupt
        u_samples = grandom.uniform(key, (n,))
        return -self._stripping_inv_cdf(as_numpy(u_samples))

    def _sample_tail(self, n, integrate, leading=True, key=None):
        """Sample n points from the specified tail."""
        from ..backend import as_numpy, is_backend_array

        # Stripping times: a backend array when a key is threaded (or under a
        # forced backend). The progenitor/center orbits are numpy-only FOR NOW, so
        # query them at numpy times (dt_np); a later PR makes them backend orbits
        # so d(stream)/d(progenitor IC/FC) flows and the whole path jits / runs on
        # GPU. The einsum frame construction + sample-orbit integration are already
        # on the resolved backend xp.
        dt = self._draw_stripping_dt(n, key=key)
        xp = get_namespace(dt)  # context-resolved backend (numpy under numpy)
        dt_np = as_numpy(dt)
        # Build all rotation matrices
        rot, rot_inv = self._setup_rot(dt)
        # Compute progenitor position in the instantaneous frame,
        # relative to the center orbit if necessary
        centerx = xp.atleast_1d(xp.asarray(self._progenitor.x(-dt_np)))
        centery = xp.atleast_1d(xp.asarray(self._progenitor.y(-dt_np)))
        centerz = xp.atleast_1d(xp.asarray(self._progenitor.z(-dt_np)))
        centervx = xp.atleast_1d(xp.asarray(self._progenitor.vx(-dt_np)))
        centervy = xp.atleast_1d(xp.asarray(self._progenitor.vy(-dt_np)))
        centervz = xp.atleast_1d(xp.asarray(self._progenitor.vz(-dt_np)))
        if not self._center is None:
            centerx = centerx - xp.asarray(self._center.x(-dt_np))
            centery = centery - xp.asarray(self._center.y(-dt_np))
            centerz = centerz - xp.asarray(self._center.z(-dt_np))
            centervx = centervx - xp.asarray(self._center.vx(-dt_np))
            centervy = centervy - xp.asarray(self._center.vy(-dt_np))
            centervz = centervz - xp.asarray(self._center.vz(-dt_np))
        # stack(axis=0).T matches numpy.array([...]).T's F-contiguous layout so
        # einsum rounds byte-identically to the pre-migration numpy path.
        xyzpt = xp.einsum(
            "ijk,ik->ij", rot, xp.stack([centerx, centery, centerz], axis=0).T
        )
        vxyzpt = xp.einsum(
            "ijk,ik->ij", rot, xp.stack([centervx, centervy, centervz], axis=0).T
        )

        # generate the initial conditions
        xst, yst, zst, vxst, vyst, vzst = self.spray_df(xyzpt, vxyzpt, dt, leading)

        xyzs = xp.einsum("ijk,ik->ij", rot_inv, xp.stack([xst, yst, zst], axis=0).T)
        vxyzs = xp.einsum("ijk,ik->ij", rot_inv, xp.stack([vxst, vyst, vzst], axis=0).T)

        absx = xyzs[:, 0]
        absy = xyzs[:, 1]
        absz = xyzs[:, 2]
        absvx = vxyzs[:, 0]
        absvy = vxyzs[:, 1]
        absvz = vxyzs[:, 2]
        if not self._center is None:
            absx = absx + xp.asarray(self._center.x(-dt_np))
            absy = absy + xp.asarray(self._center.y(-dt_np))
            absz = absz + xp.asarray(self._center.z(-dt_np))
            absvx = absvx + xp.asarray(self._center.vx(-dt_np))
            absvy = absvy + xp.asarray(self._center.vy(-dt_np))
            absvz = absvz + xp.asarray(self._center.vz(-dt_np))
        Rs, phis, Zs = coords.rect_to_cyl(absx, absy, absz)
        vRs, vTs, vZs = coords.rect_to_cyl_vec(
            absvx, absvy, absvz, Rs, phis, Zs, cyl=True
        )
        if integrate:
            # Integrate all sampled particles as a single Orbit instance, with
            # each particle on its own time grid from its stripping time -dt[i]
            # to the present (t=0). The final time step is the present-day state.
            ic_arr = xp.stack([Rs, vRs, vTs, Zs, vZs, phis], axis=0).T
            o = Orbit(ic_arr)
            if is_backend_array(ic_arr):
                # Backend ICs -> the ADAPTIVE RK method dop853_c routes to the
                # differentiable C-STM (the default fixed-step symplec4_c is not
                # auto-routed). Only the present-day state is used, and dop853_c
                # takes its own internal substeps, so integrate on a per-orbit
                # 2-point grid [-dt_i, 0] (not the 10001-point fixed-step grid the
                # numpy path needs) -- avoids materialising a (n, 10001, 6, 6) STM.
                ts = xp.stack([-xp.asarray(dt_np), xp.zeros(n)], axis=-1)
                o.integrate(ts, self._pot, method="dop853_c")
            else:
                ts = xp.linspace(-dt_np, xp.zeros(n), 10001, axis=-1)
                o.integrate(ts, self._pot)  # byte-identical numpy default
            out = o.orbit[:, -1, :].T
        else:
            out = xp.stack([Rs, vRs, vTs, Zs, vZs, phis], axis=0)
        return out, dt

    def _setup_rot(self, dt):
        from ..backend import as_numpy

        xp = get_namespace(dt)
        # Progenitor/center orbits are numpy-only FOR NOW -> query at numpy times;
        # keep xp (from dt) so the rotation-matrix arithmetic runs on the backend.
        # A later PR makes the progenitor a backend orbit (progenitor gradients +
        # jit/GPU), at which point these queries take backend times.
        dt_np = as_numpy(dt)
        n = len(dt)
        centerx = xp.atleast_1d(xp.asarray(self._progenitor.x(-dt_np)))
        centery = xp.atleast_1d(xp.asarray(self._progenitor.y(-dt_np)))
        centerz = xp.atleast_1d(xp.asarray(self._progenitor.z(-dt_np)))
        if self._center is None:
            L = xp.atleast_2d(xp.asarray(self._progenitor.L(-dt_np)))
        # Compute relative angular momentum to the center orbit
        else:
            centerx = centerx - xp.asarray(self._center.x(-dt_np))
            centery = centery - xp.asarray(self._center.y(-dt_np))
            centerz = centerz - xp.asarray(self._center.z(-dt_np))
            centervx = xp.asarray(self._progenitor.vx(-dt_np)) - xp.asarray(
                self._center.vx(-dt_np)
            )
            centervy = xp.asarray(self._progenitor.vy(-dt_np)) - xp.asarray(
                self._center.vy(-dt_np)
            )
            centervz = xp.asarray(self._progenitor.vz(-dt_np)) - xp.asarray(
                self._center.vz(-dt_np)
            )
            L = xp.atleast_2d(
                xp.stack(
                    [
                        centery * centervz - centerz * centervy,
                        centerz * centervx - centerx * centervz,
                        centerx * centervy - centery * centervx,
                    ],
                    axis=-1,
                )
            )
        Lnorm = L / xp.sqrt(xp.sum(L**2.0, axis=1))[:, None]
        z_rot = xp.swapaxes(
            _rotate_to_arbitrary_vector(xp.atleast_2d(Lnorm), [0.0, 0.0, 1], inv=True),
            1,
            2,
        )
        z_rot_inv = xp.swapaxes(
            _rotate_to_arbitrary_vector(xp.atleast_2d(Lnorm), [0.0, 0.0, 1], inv=False),
            1,
            2,
        )
        xyzt = xp.einsum(
            "ijk,ik->ij", z_rot, xp.stack([centerx, centery, centerz], axis=0).T
        )
        Rt = xp.sqrt(xyzt[:, 0] ** 2.0 + xyzt[:, 1] ** 2.0)
        cosphi, sinphi = xyzt[:, 0] / Rt, xyzt[:, 1] / Rt
        zero, one = xp.zeros_like(cosphi), xp.ones_like(cosphi)
        # (3,3,n).T -> (n,3,3): each row is the transpose of the numpy block.
        pa_rot = xp.stack(
            [
                xp.stack([cosphi, sinphi, zero], axis=-1),
                xp.stack([-sinphi, cosphi, zero], axis=-1),
                xp.stack([zero, zero, one], axis=-1),
            ],
            axis=1,
        )
        pa_rot_inv = xp.stack(
            [
                xp.stack([cosphi, -sinphi, zero], axis=-1),
                xp.stack([sinphi, cosphi, zero], axis=-1),
                xp.stack([zero, zero, one], axis=-1),
            ],
            axis=1,
        )
        rot = xp.einsum("ijk,ikl->ijl", pa_rot, z_rot)
        rot_inv = xp.einsum("ijk,ikl->ijl", z_rot_inv, pa_rot_inv)
        return (rot, rot_inv)

    def _calc_rtide(self, Rpt, phipt, Zpt, dt):
        xp = get_namespace(dt)
        Ms = self._progenitor_mass_fn(-dt)
        # Anchor Ms on Rpt only when Rpt is a backend array (the spray flow);
        # a direct numpy call keeps Ms numpy (byte-identical).
        M = as_backend_constant(xp, Ms, Rpt) if is_backend_array(Rpt) else Ms
        try:
            rtides = rtide(
                self._rtpot,
                Rpt,
                Zpt,
                phi=phipt,
                t=-dt,
                M=M,
                use_physical=False,
            )
        except (ValueError, TypeError):
            # Per-particle numpy fallback for potentials without array support;
            # coerce the island result back to the active backend.
            with use("numpy", force=True):
                rtides = numpy.array(
                    [
                        rtide(
                            self._rtpot,
                            float(Rpt[ii]),
                            float(Zpt[ii]),
                            phi=float(phipt[ii]),
                            t=-dt[ii],
                            M=float(Ms[ii]),
                            use_physical=False,
                        )
                        for ii in range(len(Rpt))
                    ]
                )
            rtides = as_backend_constant(xp, rtides, Rpt)
        return rtides

    def _calc_vc(self, Rpt, phipt, Zpt, dt):
        xp = get_namespace(dt)
        try:
            vcs = xp.sqrt(
                -Rpt
                * evaluateRforces(
                    self._rtpot, Rpt, Zpt, phi=phipt, t=-dt, use_physical=False
                )
            )
        except (ValueError, TypeError):
            # Per-particle numpy fallback for potentials without array support;
            # coerce the island result back to the active backend.
            with use("numpy", force=True):
                vcs = numpy.array(
                    [
                        numpy.sqrt(
                            -float(Rpt[ii])
                            * evaluateRforces(
                                self._rtpot,
                                float(Rpt[ii]),
                                float(Zpt[ii]),
                                phi=float(phipt[ii]),
                                t=-dt[ii],
                                use_physical=False,
                            )
                        )
                        for ii in range(len(Rpt))
                    ]
                )
            vcs = as_backend_constant(xp, vcs, Rpt)
        return vcs

    def _parse_progenitor_mass(self, progenitor_mass):
        # Sets self._progenitor_mass_fn(t) -> internal-unit mass, where t is
        # the progenitor-time coordinate (t=0 is now, t<0 is the past). Also
        # sets self._progenitor_mass to the present-day value for any external
        # code that reads the attribute. Accepts: scalar float, Quantity, or
        # callable. Callables are auto-detected for Quantity input / output
        # (same four-branch pattern as AnyAxisymmetricRazorThinDiskPotential).
        if not callable(progenitor_mass):
            M0 = conversion.parse_mass(progenitor_mass, ro=self._ro, vo=self._vo)
            self._progenitor_mass_fn = lambda t: (
                M0 * numpy.ones_like(numpy.asarray(t, dtype=float))
            )
            self._progenitor_mass = M0
            return
        _mass_unit_input = False
        _mass_unit_output = False
        if _APY_LOADED:
            try:
                progenitor_mass(0.0)
            except (
                units.UnitConversionError,
                units.UnitTypeError,
                AttributeError,
            ):
                _mass_unit_input = True
            probe_in = 0.0 * units.Gyr if _mass_unit_input else 0.0
            try:
                progenitor_mass(probe_in).to(units.Msun)
            except (AttributeError, units.UnitConversionError):
                pass
            else:
                _mass_unit_output = True
        _time_to_quantity = (
            conversion.time_in_Gyr(self._vo, self._ro) * units.Gyr
            if _APY_LOADED
            else None
        )
        if _mass_unit_input and _mass_unit_output:

            def _mass_fn(t):
                t_q = numpy.asarray(t, dtype=float) * _time_to_quantity
                return conversion.parse_mass(
                    progenitor_mass(t_q), ro=self._ro, vo=self._vo
                )

        elif _mass_unit_input:

            def _mass_fn(t):
                t_q = numpy.asarray(t, dtype=float) * _time_to_quantity
                return numpy.asarray(progenitor_mass(t_q), dtype=float)

        elif _mass_unit_output:

            def _mass_fn(t):
                return conversion.parse_mass(
                    progenitor_mass(numpy.asarray(t, dtype=float)),
                    ro=self._ro,
                    vo=self._vo,
                )

        else:

            def _mass_fn(t):
                return numpy.asarray(
                    progenitor_mass(numpy.asarray(t, dtype=float)),
                    dtype=float,
                )

        self._progenitor_mass_fn = _mass_fn
        self._progenitor_mass = float(self._progenitor_mass_fn(0.0))

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
        stripping_pdf=None,
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
        progenitor_mass : float, Quantity, or callable
            Mass of the progenitor. If a callable, it is a function ``M(t)`` of the progenitor-time coordinate (``t=0`` is now, ``t<0`` is the past); may take and/or return astropy ``Quantity`` (auto-detected). See :class:`basestreamspraydf` for details.
        progenitor : galpy.orbit.Orbit, optional
            Progenitor orbit as Orbit instance (will be re-integrated, so don't bother integrating the orbit before).
        pot : galpy.potential.Potential or a combined potential formed using addition (pot1+pot2+…), optional
            Potential for integrating orbits.
        rtpot : galpy.potential.Potential or a combined potential formed using addition (pot1+pot2+…), optional
            Potential for calculating tidal radius and circular velocity (should generally be the same as pot, but sometimes you need to drop parts of the potential that don't allow the tidal radius / circular velocity to be computed, such as velocity-dependent forces; when using center, rtpot should be the relevant potential in the frame of the center, thus, also being different from pot).
        tdisrupt : float or Quantity, optional
            Time since start of disruption. Default is 5 Gyr.
        stripping_pdf : callable, optional
            Probability density of stripping over the progenitor time axis ``t in [-tdisrupt, 0]``. See :class:`galpy.df.streamspraydf.basestreamspraydf` for the full description. Default is None (uniform stripping).
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
            stripping_pdf=stripping_pdf,
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
        xp = get_namespace(dt)
        Rpt, phipt, Zpt = coords.rect_to_cyl(xyzpt[:, 0], xyzpt[:, 1], xyzpt[:, 2])
        rtides = self._calc_rtide(Rpt, phipt, Zpt, dt)

        # Sample positions and velocities in the instantaneous frame
        # (RNG stays numpy on mean/cov; the draw is coerced to the backend).
        posvel = xp.asarray(
            numpy.random.multivariate_normal(self._mean, self._cov, size=len(dt))
        )
        Dr = posvel[:, 0] * rtides
        Ms = as_backend_constant(xp, self._progenitor_mass_fn(-dt), Dr)
        v_esc = xp.sqrt(2 * Ms / Dr)
        Dv = posvel[:, 3] * v_esc
        if leading:
            Dr = Dr * -1.0
            Dv = Dv * -1.0

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
        stripping_pdf=None,
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
        progenitor_mass : float, Quantity, or callable
            Mass of the progenitor. If a callable, it is a function ``M(t)`` of the progenitor-time coordinate (``t=0`` is now, ``t<0`` is the past); may take and/or return astropy ``Quantity`` (auto-detected). See :class:`basestreamspraydf` for details.
        progenitor : galpy.orbit.Orbit, optional
            Progenitor orbit as Orbit instance (will be re-integrated, so don't bother integrating the orbit before).
        pot : galpy.potential.Potential or a combined potential formed using addition (pot1+pot2+…), optional
            Potential for integrating orbits.
        rtpot : galpy.potential.Potential or a combined potential formed using addition (pot1+pot2+…), optional
            Potential for calculating tidal radius and circular velocity (should generally be the same as pot, but sometimes you need to drop parts of the potential that don't allow the tidal radius / circular velocity to be computed, such as velocity-dependent forces; when using center, rtpot should be the relevant potential in the frame of the center, thus, also being different from pot).
        tdisrupt : float or Quantity, optional
            Time since start of disruption. Default is 5 Gyr.
        stripping_pdf : callable, optional
            Probability density of stripping over the progenitor time axis ``t in [-tdisrupt, 0]``. See :class:`galpy.df.streamspraydf.basestreamspraydf` for the full description. Default is None (uniform stripping).
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
            stripping_pdf=stripping_pdf,
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
        xp = get_namespace(dt)
        Rpt, phipt, Zpt = coords.rect_to_cyl(xyzpt[:, 0], xyzpt[:, 1], xyzpt[:, 2])
        rtides = self._calc_rtide(Rpt, phipt, Zpt, dt)
        vcs = self._calc_vc(Rpt, phipt, Zpt, dt)
        rtides_as_frac = rtides / Rpt

        vRpt, vTpt, vZpt = coords.rect_to_cyl_vec(
            vxyzpt[:, 0], vxyzpt[:, 1], vxyzpt[:, 2], Rpt, phipt, Zpt, cyl=True
        )
        # Sample positions and velocities in the instantaneous frame
        # (RNG stays numpy; the draw and mean/sig constants are coerced).
        meankvec = as_backend_constant(
            xp, -self._meankvec if leading else self._meankvec, Rpt
        )
        sigkvec = as_backend_constant(xp, self._sigkvec, Rpt)
        k = meankvec + xp.asarray(numpy.random.normal(size=(len(dt), 6))) * sigkvec

        RpZst = xp.stack(
            [
                Rpt + k[:, 0] * rtides,
                phipt + k[:, 5] * rtides_as_frac,
                k[:, 3] * rtides_as_frac,
            ],
            axis=-1,
        )
        vRTZst = xp.stack(
            [
                vRpt * (1.0 + k[:, 1]),
                vTpt + k[:, 2] * vcs * rtides_as_frac,
                k[:, 4] * vcs * rtides_as_frac,
            ],
            axis=-1,
        )
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


def pericenter_stripping_pdf(
    progenitor,
    pot,
    tdisrupt,
    sigma,
    ngrid=10001,
    ro=None,
    vo=None,
):
    """
    Build a stripping-time PDF from a progenitor's pericenter passages.

    The returned PDF is an equal-height sum of Gaussians centered on every
    pericenter passage of the progenitor over the interval ``[-tdisrupt, 0]``
    (``t=0`` is the present day, ``t<0`` the past) and truncated to that
    same interval. Useful for ``stripping_pdf=`` of
    :class:`galpy.df.streamspraydf.basestreamspraydf` subclasses, capturing
    the well-known enhancement of tidal stripping at pericenter passages.

    Parameters
    ----------
    progenitor : galpy.orbit.Orbit
        Progenitor orbit. Will be copied and re-integrated internally.
    pot : galpy.potential.Potential or a combined potential
        Potential used to integrate the progenitor.
    tdisrupt : float or Quantity
        Time since start of disruption. Pericenter passages over
        ``[-tdisrupt, 0]`` are located.
    sigma : float or Quantity
        Width (standard deviation) of each Gaussian.
    ngrid : int, optional
        Number of grid points used to integrate the progenitor and locate
        pericenter passages. Default 10001.
    ro : float or Quantity, optional
        Distance scale (defaults to ``progenitor``'s ``ro``).
    vo : float or Quantity, optional
        Velocity scale (defaults to ``progenitor``'s ``vo``).

    Returns
    -------
    pdf : callable
        Function ``pdf(t)`` returning the PDF at ``t``. When ``sigma`` is a
        ``Quantity``, the returned PDF expects ``t`` as a ``Quantity`` and
        returns ``Quantity`` values with units ``1/Gyr``; otherwise it works
        in internal units. The callable carries an attribute
        ``pdf.pericenter_times`` (1D array, internal units) listing the
        pericenter times.

    Raises
    ------
    ValueError
        If ``ro``/``vo`` are explicitly given but disagree with the
        progenitor's, or if no pericenter passages are found on
        ``[-tdisrupt, 0]`` (e.g. a nearly circular orbit) — in the latter
        case, supply a custom ``stripping_pdf`` instead.
    """
    sigma_is_quantity = _APY_LOADED and isinstance(sigma, units.Quantity)
    # Inherit ro/vo from the progenitor if not given; otherwise check
    # consistency (progenitor/pot consistency is enforced by Orbit.integrate).
    if ro is None and progenitor._roSet:
        ro = progenitor._ro
    elif ro is not None and progenitor._roSet:
        ro_internal = conversion.parse_length_kpc(ro)
        if abs(ro_internal - progenitor._ro) / progenitor._ro > 1e-8:
            raise ValueError("ro inconsistent with progenitor's ro; omit ro to inherit")
    if vo is None and progenitor._voSet:
        vo = progenitor._vo
    elif vo is not None and progenitor._voSet:
        vo_internal = conversion.parse_velocity_kms(vo)
        if abs(vo_internal - progenitor._vo) / progenitor._vo > 1e-8:
            raise ValueError("vo inconsistent with progenitor's vo; omit vo to inherit")
    tdisrupt_internal = conversion.parse_time(tdisrupt, ro=ro, vo=vo)
    sigma_internal = conversion.parse_time(sigma, ro=ro, vo=vo)
    # Integrate the progenitor and locate pericenter passages on the grid.
    # The prominence threshold rejects numerical-noise oscillations on
    # near-circular orbits.
    prog_copy = progenitor()
    prog_copy.turn_physical_off()
    ts = numpy.linspace(0.0, -tdisrupt_internal, ngrid)
    prog_copy.integrate(ts, _check_potential_list_and_deprecate(pot))
    r_vals = as_numpy(prog_copy.r(ts))  # scipy find_peaks is numpy-only
    peaks, _ = find_peaks(-r_vals, prominence=1e-6 * float(numpy.mean(r_vals)))
    if peaks.size == 0:
        raise ValueError(
            "No pericenter passages found over [-tdisrupt, 0]. The orbit "
            "may be nearly circular; supply a custom stripping_pdf instead."
        )
    peri_times = ts[peaks]
    # Normalized Gaussian mixture, truncated to [-tdisrupt, 0].
    norm = 1.0 / (peri_times.size * sigma_internal * numpy.sqrt(2.0 * numpy.pi))

    def _pdf_internal(t):
        t_arr = numpy.atleast_1d(numpy.asarray(t, dtype=float))
        dx = (t_arr[:, None] - peri_times[None, :]) / sigma_internal
        out = norm * numpy.sum(numpy.exp(-0.5 * dx * dx), axis=-1)
        out = numpy.where((t_arr >= -tdisrupt_internal) & (t_arr <= 0.0), out, 0.0)
        return float(out[0]) if numpy.ndim(t) == 0 else out

    if not sigma_is_quantity:
        _pdf_internal.pericenter_times = peri_times
        return _pdf_internal

    time_in_gyr = conversion.time_in_Gyr(vo, ro)

    def pdf(t):
        return _pdf_internal(conversion.parse_time(t, ro=ro, vo=vo)) / (
            time_in_gyr * units.Gyr
        )

    pdf.pericenter_times = peri_times
    return pdf
