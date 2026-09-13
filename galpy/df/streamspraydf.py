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
    name_of_namespace,
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
        integrate_kwargs=None,
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
        # Extra options for the in-backend ODE solves (progenitor, center, sampled
        # orbits, track curve). Only reach jax/torch integrators -- the numpy/C paths
        # take none. `adjoint="direct"` is what makes forward-mode and higher-order
        # AD work: the default checkpointed adjoint is a custom_vjp, i.e. reverse-only.
        self._integrate_kwargs = integrate_kwargs
        self._orig_progenitor = progenitor  # Store so we can use its ro/vo/etc.
        self._progenitor = progenitor()
        self._progenitor.turn_physical_off()
        self._progenitor_times = numpy.linspace(0.0, -self._tdisrupt, 10001)
        self._integrate_progenitor()
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
            self._orig_center = center  # kept for its _ic_backend (like the progenitor)
            self._center = center()
            self._center.turn_physical_off()
            self._integrate_center()
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
        key=None,
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
                from ..backend import random as grandom

                key_l, key_t = grandom.split(key)
                xv_lead, _ = self._sample_tail(n_lead, True, leading=True, key=key_l)
                xv_trail, _ = self._sample_tail(n_trail, True, leading=False, key=key_t)
                xv_all = numpy.column_stack([xv_lead, xv_trail])
        else:
            if particles is not None:
                xv_single = (
                    particles
                    if is_backend_array(particles)
                    else numpy.asarray(particles, dtype=float)
                )
            else:
                xv_single, _ = self._sample_tail(
                    n, True, leading=(tail == "leading"), key=key
                )
            xv_all = xv_single

        if track_time_range is None:
            track_time_range = self._auto_track_time_range(xv_all)
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
        # When track_time_range is a TRACED scalar (jit auto-estimate), work in
        # NORMALIZED curve coordinates: the integration times are T*u (traced) but the
        # curve grid / reconstruction axis is the CONCRETE normalized u in [-1, 1], and
        # T is passed to StreamTrack as tp_scale (the physical parameter is u*T). This
        # keeps every interpolation grid concrete under jit while the physical extent
        # stays data-dependent + differentiable. Eager (concrete T) -> physical grid,
        # tp_scale=None (byte-identical).
        try:
            float(track_time_range)
            _tp_scale = None
        except Exception:  # noqa: BLE001 -- traced (jit) extent
            _tp_scale = track_time_range
        if _tp_scale is None:
            # eager: physical integration times + physical reconstruction axis (the
            # numpy path stays byte-identical to numpy.linspace(0, T)).
            t_fwd = numpy.linspace(0.0, track_time_range, half_dense)
            t_back = numpy.linspace(0.0, -track_time_range, half_dense)
            track_t_grid = numpy.concatenate([t_back[::-1], t_fwd[1:]])
        else:
            # jit: physical times T*u (traced) but normalized concrete axis u in [-1,1].
            _u_fwd = numpy.linspace(0.0, 1.0, half_dense)
            _u_back = numpy.linspace(0.0, -1.0, half_dense)
            t_fwd = track_time_range * _u_fwd
            t_back = track_time_range * _u_back
            track_t_grid = numpy.concatenate([_u_back[::-1], _u_fwd[1:]])
        prog_ic_backend = getattr(self._orig_progenitor, "_ic_backend", None)
        # Backend sampling detected at construction (self._bsamp) already resolved the
        # backend namespace + integrator (dop853_c C-STM for a concrete backend IC;
        # the in-backend ODE for a backend theta or under jit) and made
        # self._progenitor a backend orbit -- reuse it instead of a float() probe on
        # self._progenitor, which would break under jit.
        _bsamp = self._backend_sampling()

        def _backend_curve(ic, method, xp):
            # Integrate the progenitor fwd+back and stitch into a differentiable
            # backend track_prog_cart (torch has no negative-step slice -> xp.flip).
            o_f = Orbit(ic)
            o_f.turn_physical_off()
            o_f.integrate(t_fwd, _track_pot, method=method, **self._ikw(method))
            o_b = Orbit(ic)
            o_b.turn_physical_off()
            o_b.integrate(t_back, _track_pot, method=method, **self._ikw(method))

            def _cart(o):
                # Read the integrated states directly (o.orbit at the integration grid)
                # rather than o.x(ts): the interpolator needs a concrete self.t, but
                # under jit the integration times T*u are traced. o.orbit is cylindrical
                # [R,vR,vT,z,vz,phi] -> convert to cartesian (galpy's exact convention).
                ob = o.orbit
                cyl = ob[0] if ob.ndim == 3 else ob  # (nt, 6)
                _R, _vR, _vT, _z, _vz, _phi = (cyl[:, k] for k in range(6))
                _x, _y, _zc = coords.cyl_to_rect(_R, _phi, _z)
                _vx, _vy, _vzc = coords.cyl_to_rect_vec(_vR, _vT, _vz, _phi)
                return xp.stack([_x, _y, _zc, _vx, _vy, _vzc], axis=-1)

            return xp.concat([xp.flip(_cart(o_b), axis=0), _cart(o_f)[1:]], axis=0)

        if _bsamp is not None:
            # Backend sampling: build the differentiable progenitor curve with the
            # resolved namespace + integrator (dop853_c C-STM for a concrete backend
            # IC -- preserves #1102; the in-backend ODE for a backend theta or under
            # jit). Coerce a numpy progenitor IC onto the backend (a backend IC
            # additionally flows d(track)/d(prog IC)). A backend track_prog_cart takes
            # precedence over prog_orbit in StreamTrack (prog=None).
            xp, _, _cmethod = _bsamp
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
            track_prog_cart = _backend_curve(ic, _cmethod, xp)
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
                tp_scale=_tp_scale,
            )

        if tail == "both":
            return StreamTrackPair(
                _make_track(xv_lead, arm_sign=+1),
                _make_track(xv_trail, arm_sign=-1),
            )
        return _make_track(xv_single, arm_sign=(+1 if tail == "leading" else -1))

    def _parse_stripping_pdf(self, stripping_pdf):
        self._stripping_cdf = None  # backend inverse-CDF grid (a differentiable pdf)
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
        _pdf_backend = False
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
                except (AttributeError, TypeError, units.UnitConversionError):
                    pass
                else:
                    _t_unit_output = True
            else:
                try:
                    stripping_pdf(t_probe).to(1.0 / units.Gyr)
                except (AttributeError, TypeError, units.UnitConversionError):
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
            # A differentiable stripping_pdf returns a backend array; keep it on the
            # backend (no numpy coercion) so the inverse-CDF traces the pdf parameters.
            _pdf_backend = is_backend_array(stripping_pdf(-0.5 * self._tdisrupt))
            if _pdf_backend:

                def pdf_internal(t):
                    return stripping_pdf(t)

            else:

                def pdf_internal(t):
                    return numpy.asarray(stripping_pdf(t))

        if _pdf_backend:
            # Backend-native inverse-CDF: a cumulative-trapezoid CDF on the fixed grid,
            # inverted at draw time by searchsorted + linear interp (jit-safe, and
            # differentiable in the pdf parameters via the CDF values). This is the
            # differentiable counterpart of the scipy k=1 spline below.
            xp = get_namespace(pdf_internal(-0.5 * self._tdisrupt))
            t_grid = xp.asarray(numpy.linspace(-self._tdisrupt, 0.0, 10001))
            pdf_vals = pdf_internal(t_grid)
            dtg = t_grid[1:] - t_grid[:-1]
            # clip increments >= 0 so the CDF is monotonic even if the pdf dips slightly
            # negative (the numpy path raises; a jit trace cannot, so clamp instead)
            incr = xp.clip(0.5 * (pdf_vals[1:] + pdf_vals[:-1]) * dtg, 0.0, None)
            cdf_vals = xp.concat(
                [xp.zeros(1, dtype=incr.dtype), xp.cumsum(incr, axis=0)]
            )
            self._stripping_cdf = (cdf_vals / cdf_vals[-1], t_grid)
            self._stripping_inv_cdf = None
            return
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
        (the default uniform-stripping path). A numpy ``stripping_pdf`` inverts a
        scipy spline on the numpy sample; a backend (differentiable) ``stripping_pdf``
        inverts its backend CDF grid natively (searchsorted + linear interp), so the
        drawn ``dt`` traces the pdf parameters.
        """
        from ..backend import as_numpy
        from ..backend import random as grandom

        if self._stripping_cdf is not None:
            cdf_vals, t_grid = self._stripping_cdf
            xp = get_namespace(cdf_vals)
            u = grandom.uniform(key, (n,))
            u = u if is_backend_array(u) else xp.asarray(u)
            # linear inverse-CDF: bracket u in the (monotonic) CDF, interpolate t; the
            # bracket index is frozen (piecewise), the interp weight is differentiable.
            idx = xp.clip(xp.searchsorted(cdf_vals, u), 1, cdf_vals.shape[0] - 1)
            c0, c1 = cdf_vals[idx - 1], cdf_vals[idx]
            t0, t1 = t_grid[idx - 1], t_grid[idx]
            denom = c1 - c0
            # guard the dead branch's div-by-zero (flat CDF ties) -- eager xp.where
            # evaluates both sides (see the xp.where dead-branch idiom)
            w = (u - c0) / xp.where(denom > 0, denom, xp.ones_like(denom))
            return -(t0 + w * (t1 - t0))
        if self._stripping_inv_cdf is None:
            return grandom.uniform(key, (n,)) * self._tdisrupt
        u_samples = grandom.uniform(key, (n,))
        return -self._stripping_inv_cdf(as_numpy(u_samples))

    def _progenitor_now(self):
        """The progenitor's present-day ``(R, vR, vT, z, vz, phi)`` as concrete floats.

        Raises when they cannot be realised as floats (a traced IC) -- callers that
        have a defined answer in that case catch it; callers that do not should fail.
        """
        p = self._progenitor
        return (
            float(p.R(0.0)),
            float(p.vR(0.0)),
            float(p.vT(0.0)),
            float(p.z(0.0)),
            float(p.vz(0.0)),
            float(p.phi(0.0)),
        )

    def _theta_probe_point(self):
        """Concrete ``((R, z), phi, v)`` at which to probe the potential for a backend
        parameter.

        The progenitor's own present-day phase-space point, which is guaranteed to be
        somewhere the potential can be evaluated. A TRACED progenitor IC has no
        concrete coordinates, so fall back to a fixed unit-circular point; that case
        routes in-backend anyway (``ic_concrete`` below), so the probe's answer there
        only has to be well-defined, not particular.
        """
        try:
            R, vR, vT, z, vz, phi = self._progenitor_now()
        except Exception:  # noqa: BLE001 -- traced IC has no concrete coordinates
            return (1.0, 0.0), 0.0, numpy.array([0.0, 1.0, 0.0])
        return (R, z), phi, numpy.array([vR, vT, vz])

    def _integrate_progenitor(self):
        """Integrate the progenitor over ``[0, -tdisrupt]``, choosing the integrator
        by whether the sampling must run on a backend (jax/torch) for differentiable
        + jittable streams: a GENUINE backend potential parameter and/or a backend
        progenitor IC -> the in-backend ODE (diffrax/torchdiffeq), so ``self._progenitor``
        (and everything sampled from it) carries d(stream)/d(theta) and/or
        d(stream)/d(prog IC). Otherwise the numpy/C integrator (byte-identical).

        Detection is jit-safe: a backend IC is spotted by ``is_backend_array`` (works
        on tracers); for a numpy IC, a force probe at the (concrete) IC point under
        FORCED NUMPY isolates a genuine backend parameter (a traced ``amp`` survives
        forced numpy) from a merely forced context. Sets ``self._bsamp`` =
        ``(xp, self._progenitor, method)`` for backend sampling, else ``None``.
        """
        prog_ic = getattr(self._orig_progenitor, "_ic_backend", None)
        ic_backend = is_backend_array(prog_ic)
        xp = get_namespace(prog_ic) if ic_backend else None
        # Probe for a backend potential PARAMETER (theta) ALWAYS -- not only when the
        # IC is numpy. The C-STM carries d/d(IC) but NOT d/d(theta), so a
        # differentiable potential parameter must reach the in-backend ODE. When the
        # IC was a backend array this probe used to be skipped entirely, leaving
        # theta_backend False; the dispatch below then chose dop853_c and jax.grad
        # w.r.t. a potential parameter died in the C parser (as_numpy on the traced
        # parameter -> TracerArrayConversionError). The probe needs CONCRETE
        # coordinates; a traced IC has none, but that case already routes in-backend
        # via ic_concrete below, so a fixed fallback probe point is safe there.
        _pargs, _pphi, _pv = self._theta_probe_point()
        with use("numpy", force=True):
            _tf = evaluateRforces(self._pot, *_pargs, phi=_pphi, v=_pv)
        theta_backend = is_backend_array(_tf)
        if theta_backend and xp is None:
            xp = get_namespace(_tf)
        # A backend progenitor MASS (a differentiable M or M(t)) traces the sampled
        # orbits -- via the tidal radius rtide -- but NOT the mass-independent
        # progenitor curve, so it is its own backend-sampling trigger.
        _mass_probe = self._progenitor_mass_fn(0.0)
        mass_backend = is_backend_array(_mass_probe)
        if mass_backend and xp is None:
            xp = get_namespace(_mass_probe)
        # A backend (differentiable) stripping_pdf traces the sampled orbits' STRIPPING
        # TIMES (its inverse-CDF grid is a backend array) -- again not the progenitor
        # curve -- so it is its own backend-sampling trigger, exactly like the mass.
        stripping_backend = self._stripping_cdf is not None
        if stripping_backend and xp is None:
            xp = get_namespace(self._stripping_cdf[0])
        # numpy/C path (byte-identical): a pure-numpy spdf, OR a backend theta/mass/
        # stripping seen under a merely-FORCED context -- the all-backend suite runs a
        # plain-numpy test under `use(backend, force=True)`, which coerces plain inputs
        # to backend arrays with no differentiable intent (and would crash the sample
        # orbit in the unmigrated MovingObjectPotential under the in-backend ODE). A
        # genuine backend IC survives even a forced context; a theta/mass/stripping under
        # force does not, so `get_namespace(numpy.zeros(1)) is not numpy` isolates it.
        _forced = get_namespace(numpy.zeros(1)) is not numpy
        _backend_trig = theta_backend or mass_backend or stripping_backend
        if not (ic_backend or (_backend_trig and not _forced)):
            self._progenitor.integrate(self._progenitor_times, self._pot)
            self._bsamp = None
            return
        inbackend = "diffrax" if name_of_namespace(xp) == "jax" else "torchdiffeq"
        if ic_backend:
            ic = prog_ic
        else:
            _pp = self._progenitor
            ic = xp.asarray(
                numpy.array(
                    [
                        float(_pp.R(0.0)),
                        float(_pp.vR(0.0)),
                        float(_pp.vT(0.0)),
                        float(_pp.z(0.0)),
                        float(_pp.vz(0.0)),
                        float(_pp.phi(0.0)),
                    ]
                )
            )
        # Choose the integrator. A backend potential PARAMETER (theta) needs the
        # in-backend ODE (the C-STM carries no d/d(theta)). A backend progenitor IC
        # ALONE (numpy potential) keeps the faster dop853_c C-STM when the IC is
        # CONCRETE (eager, preserving the #1102 mechanism); a TRACED IC (under jit)
        # falls to the in-backend ODE (C is not traceable). Concreteness is detected
        # by whether the IC coerces to numpy (the _ic_backend_concrete idiom).
        if theta_backend or ((mass_backend or stripping_backend) and not ic_backend):
            # A backend mass / stripping_pdf alone leaves the progenitor curve unchanged,
            # but its sample orbits trace it (rtide / stripping times) and are queried
            # under jit -> the progenitor must be a backend orbit too, integrate in-backend.
            method = inbackend
        else:
            try:
                as_numpy(ic)  # detaches an eager torch grad-tensor; raises on a tracer
                ic_concrete = True
            except Exception:  # noqa: BLE001 -- traced (jit) backend IC
                ic_concrete = False
            method = "dop853_c" if ic_concrete else inbackend
        # NUMPY times: the grid is a fixed structural axis (not theta-dependent), so
        # keeping self.t concrete lets o.x(t)/L(t) interpolate under jit (the orbit
        # STATE is still a backend array -> differentiable). A backend-array grid
        # would be a tracer under jit and break the interpolator's numpy.asarray(t).
        # Coarser than the numpy 10001: the frame queries interpolate the orbit spline
        # (density-insensitive; 501 pts already match numpy to ~4e-11).
        bgrid = numpy.linspace(0.0, -self._tdisrupt, 2001)
        self._progenitor = Orbit(ic)
        self._progenitor.turn_physical_off()
        self._progenitor.integrate(bgrid, self._pot, method=method, **self._ikw(method))
        self._bsamp = (xp, self._progenitor, method)

    def _integrate_center(self):
        """Integrate the center orbit -- in-backend when the sampling runs on a backend,
        else numpy/C (byte-identical).

        A backend center makes ``self._center.x(qt)`` jit-queryable at a traced ``qt`` and
        carries d(stream)/d(center IC) and d(stream)/d(centerpot theta). The center gets
        its OWN backend-trigger detection because ``self._centerpot`` can hold a different
        backend theta than ``self._pot`` (a satellite whose orbit needs e.g. a
        dynamical-friction potential): a backend center IC (``is_backend_array``, a genuine
        trigger even under a forced context) or a genuine backend ``centerpot`` force (a
        force probe under FORCED NUMPY, excluding a merely-forced context) drives
        differentiable center= sampling even when the progenitor itself is pure numpy. In
        that case the (theta-independent) progenitor is re-integrated in-backend too, so
        both orbits are queryable at the same traced qt. Method: the in-backend ODE for a
        backend centerpot theta / traced IC / any traced-progenitor context; else dop853_c
        (eager, faster).
        """
        _c = self._center
        # The center's own backend triggers (cases the progenitor-side detection misses).
        cic_backend = getattr(self._orig_center, "_ic_backend", None)
        ic_backend = is_backend_array(cic_backend)
        with use("numpy", force=True):
            _cf = evaluateRforces(
                self._centerpot,
                float(_c.R(0.0)),
                float(_c.z(0.0)),
                phi=float(_c.phi(0.0)),
                v=numpy.array(
                    [float(_c.vR(0.0)), float(_c.vT(0.0)), float(_c.vz(0.0))]
                ),  # a dynamical-friction centerpot is velocity-dependent
            )
        centerpot_theta = is_backend_array(_cf)
        _forced = get_namespace(numpy.zeros(1)) is not numpy
        center_trig = ic_backend or (centerpot_theta and not _forced)
        bsamp = getattr(self, "_bsamp", None)
        if bsamp is None and not center_trig:
            # pure numpy/C path -- byte-identical to the pre-backend center integration.
            self._center.integrate(self._progenitor_times, self._centerpot)
            return
        # Resolve the backend namespace: from the progenitor-side _bsamp if present, else
        # from the center trigger (and then promote the numpy progenitor to a backend orbit).
        if bsamp is not None:
            xp, _, prog_method = bsamp
        else:
            xp = get_namespace(cic_backend) if ic_backend else get_namespace(_cf)
        inbackend = "diffrax" if name_of_namespace(xp) == "jax" else "torchdiffeq"
        if bsamp is None:
            self._promote_progenitor_backend(xp, inbackend)
            _, _, prog_method = self._bsamp
        # center IC: a genuine backend IC (d/d center-IC) else coerce the numpy IC.
        if ic_backend:
            ic = cic_backend
        else:
            ic = xp.asarray(
                numpy.array(
                    [
                        float(_c.R(0.0)),
                        float(_c.vR(0.0)),
                        float(_c.vT(0.0)),
                        float(_c.z(0.0)),
                        float(_c.vz(0.0)),
                        float(_c.phi(0.0)),
                    ]
                )
            )
        try:
            as_numpy(ic)  # raises on a tracer
            ic_concrete = True
        except Exception:  # noqa: BLE001 -- traced (jit) backend IC
            ic_concrete = False
        if centerpot_theta or not ic_concrete or prog_method != "dop853_c":
            method = inbackend
        else:
            method = "dop853_c"
        bgrid = numpy.linspace(0.0, -self._tdisrupt, 2001)  # match the progenitor grid
        self._center = Orbit(ic)
        self._center.turn_physical_off()
        self._center.integrate(
            bgrid, self._centerpot, method=method, **self._ikw(method)
        )

    def _promote_progenitor_backend(self, xp, inbackend):
        """Re-integrate a pure-numpy progenitor as a backend orbit (theta/mass-independent
        curve, so the physics is unchanged) so it is queryable at a traced qt. Needed when
        a center-only backend trigger (centerpot theta / center IC) drives differentiable
        center= sampling while ``self._pot`` carries no backend parameter. Sets ``_bsamp``.
        """
        ic = xp.asarray(numpy.array(self._progenitor_now()))
        bgrid = numpy.linspace(0.0, -self._tdisrupt, 2001)
        self._progenitor = Orbit(ic)
        self._progenitor.turn_physical_off()
        self._progenitor.integrate(
            bgrid, self._pot, method=inbackend, **self._ikw(inbackend)
        )
        self._bsamp = (xp, self._progenitor, inbackend)

    def _ikw(self, method):
        """``inbackend_kwargs`` for an in-backend ODE ``method``, else ``{}``.

        The C and numpy integrators take no solver options, so the extra kwargs are
        handed only to the jax/torch paths.
        """
        if not self._integrate_kwargs or method not in ("diffrax", "torchdiffeq"):
            return {}
        return {"inbackend_kwargs": dict(self._integrate_kwargs)}

    def _backend_sampling(self):
        """``(xp, backend_progenitor, sample_method)`` when the sampling runs on a
        backend for differentiable/jittable streams, else ``None`` (pure numpy).
        Set at construction by :meth:`_integrate_progenitor`."""
        return getattr(self, "_bsamp", None)

    def _auto_track_time_range(self, xv_all):
        """Concrete scalar time-range bound for the reconstructed track (``track_t``).

        The accurate particle-extent estimate: 8x the farthest particle's distance
        from the progenitor divided by the progenitor's present-day speed, clipped to
        ``[1, tdisrupt]`` (scales with stream width). Eager returns a concrete float;
        under jit (particles + progenitor TRACED) the SAME estimate is computed with
        backend ops and returned as a TRACED scalar -- the caller then works in
        normalized curve coordinates (concrete grids) with this as the physical scale,
        so nothing needs a concrete extent.
        """
        try:
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
            return float(numpy.clip(8.0 * _d_max / max(_pv, 1e-6), 1.0, self._tdisrupt))
        except Exception:  # noqa: BLE001 -- traced (jit): same estimate, backend ops
            xp = get_namespace(xv_all)
            _R, _, _, _z, _, _phi = xv_all
            _x = _R * xp.cos(_phi)
            _y = _R * xp.sin(_phi)
            _px = self._progenitor.x(0.0)
            _py = self._progenitor.y(0.0)
            _pz = self._progenitor.z(0.0)
            _pv = xp.sqrt(
                self._progenitor.vx(0.0) ** 2
                + self._progenitor.vy(0.0) ** 2
                + self._progenitor.vz(0.0) ** 2
            )
            _d_max = xp.sqrt(
                xp.max((_x - _px) ** 2 + (_y - _py) ** 2 + (_z - _pz) ** 2)
            )
            return xp.clip(8.0 * _d_max / xp.clip(_pv, 1e-6, None), 1.0, self._tdisrupt)

    def _sample_tail(self, n, integrate, leading=True, key=None):
        """Sample n points from the specified tail."""
        from ..backend import as_numpy, is_backend_array
        from ..backend import random as grandom

        # Stripping times are theta-independent random draws. For differentiable
        # sampling (a backend potential parameter and/or backend progenitor IC),
        # `_backend_sampling()` returns the backend namespace, a backend progenitor
        # orbit (whose queries carry the gradient), and the sampled-orbit integrator
        # (in-backend ODE for d/d(theta), C-STM for d/d(prog IC)); dt is coerced to a
        # backend constant so `spray_df`/`_calc_rtide` resolve to the backend. Else
        # the pure-numpy path (a keyed/forced backend still resolves xp from dt).
        # Independent sub-keys for the stripping-time and spray-offset draws so the
        # spray noise is reparameterized (a deterministic function of the key,
        # theta-independent) -- essential under jit, where numpy.random bakes a
        # per-trace constant that makes AD and FD disagree. numpy key (None) ->
        # (None, None): sequential global draws, byte-identical.
        k_dt, k_spray = grandom.split(key)
        dt = self._draw_stripping_dt(n, key=k_dt)
        bsamp = self._backend_sampling()
        if bsamp is not None:
            # backend progenitor: keep dt on the backend (no as_numpy -> jit-safe) and
            # query it at BACKEND times qt so o.x(qt) stays traced/differentiable.
            xp, prog, sample_method = bsamp
            dt = xp.asarray(dt)
            qt = -dt
        else:
            xp = get_namespace(dt)  # context-resolved backend (numpy under numpy)
            prog = self._progenitor
            sample_method = None
            qt = -as_numpy(dt)  # numpy query times (byte-identical)
        # Build all rotation matrices
        rot, rot_inv = self._setup_rot(dt, prog=prog, qt=qt)
        # Compute progenitor position in the instantaneous frame,
        # relative to the center orbit if necessary
        centerx = xp.atleast_1d(xp.asarray(prog.x(qt)))
        centery = xp.atleast_1d(xp.asarray(prog.y(qt)))
        centerz = xp.atleast_1d(xp.asarray(prog.z(qt)))
        centervx = xp.atleast_1d(xp.asarray(prog.vx(qt)))
        centervy = xp.atleast_1d(xp.asarray(prog.vy(qt)))
        centervz = xp.atleast_1d(xp.asarray(prog.vz(qt)))
        if not self._center is None:
            centerx = centerx - xp.asarray(self._center.x(qt))
            centery = centery - xp.asarray(self._center.y(qt))
            centerz = centerz - xp.asarray(self._center.z(qt))
            centervx = centervx - xp.asarray(self._center.vx(qt))
            centervy = centervy - xp.asarray(self._center.vy(qt))
            centervz = centervz - xp.asarray(self._center.vz(qt))
        # stack(axis=0).T matches numpy.array([...]).T's F-contiguous layout so
        # einsum rounds byte-identically to the pre-migration numpy path.
        xyzpt = xp.einsum(
            "ijk,ik->ij", rot, xp.stack([centerx, centery, centerz], axis=0).T
        )
        vxyzpt = xp.einsum(
            "ijk,ik->ij", rot, xp.stack([centervx, centervy, centervz], axis=0).T
        )

        # generate the initial conditions
        xst, yst, zst, vxst, vyst, vzst = self.spray_df(
            xyzpt, vxyzpt, dt, leading, key=k_spray
        )

        xyzs = xp.einsum("ijk,ik->ij", rot_inv, xp.stack([xst, yst, zst], axis=0).T)
        vxyzs = xp.einsum("ijk,ik->ij", rot_inv, xp.stack([vxst, vyst, vzst], axis=0).T)

        absx = xyzs[:, 0]
        absy = xyzs[:, 1]
        absz = xyzs[:, 2]
        absvx = vxyzs[:, 0]
        absvy = vxyzs[:, 1]
        absvz = vxyzs[:, 2]
        if not self._center is None:
            absx = absx + xp.asarray(self._center.x(qt))
            absy = absy + xp.asarray(self._center.y(qt))
            absz = absz + xp.asarray(self._center.z(qt))
            absvx = absvx + xp.asarray(self._center.vx(qt))
            absvy = absvy + xp.asarray(self._center.vy(qt))
            absvz = absvz + xp.asarray(self._center.vz(qt))
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
            if bsamp is not None or is_backend_array(ic_arr):
                # Backend ICs. For differentiable sampling use the resolved
                # integrator (in-backend ODE carries d/d(theta) AND d/d(prog IC),
                # jittable); a merely keyed/forced backend (sample_method None) uses
                # dop853_c -> C-STM. Only the present-day state is used, and each
                # integrator takes its own internal substeps, so integrate on a
                # per-orbit 2-point grid [-dt_i, 0] (not the 10001-point fixed-step
                # grid the numpy path needs).
                ts = xp.stack([-xp.asarray(dt), xp.zeros(n)], axis=-1)
                _m = sample_method or "dop853_c"
                o.integrate(ts, self._pot, method=_m, **self._ikw(_m))
            else:
                ts = xp.linspace(-as_numpy(dt), xp.zeros(n), 10001, axis=-1)
                o.integrate(ts, self._pot)  # byte-identical numpy default
            out = o.orbit[:, -1, :].T
        else:
            out = xp.stack([Rs, vRs, vTs, Zs, vZs, phis], axis=0)
        return out, dt

    def _setup_rot(self, dt, prog, qt):
        # `prog` is a backend progenitor orbit for differentiable sampling (its
        # queries then carry d/d(theta) / d/d(prog IC)); else the numpy progenitor.
        # `qt` = progenitor query times (backend -dt for jit-safety; else numpy).
        xp = get_namespace(dt)
        n = len(dt)
        centerx = xp.atleast_1d(xp.asarray(prog.x(qt)))
        centery = xp.atleast_1d(xp.asarray(prog.y(qt)))
        centerz = xp.atleast_1d(xp.asarray(prog.z(qt)))
        if self._center is None:
            L = xp.atleast_2d(xp.asarray(prog.L(qt)))
        # Compute relative angular momentum to the center orbit
        else:
            centerx = centerx - xp.asarray(self._center.x(qt))
            centery = centery - xp.asarray(self._center.y(qt))
            centerz = centerz - xp.asarray(self._center.z(qt))
            centervx = xp.asarray(prog.vx(qt)) - xp.asarray(self._center.vx(qt))
            centervy = xp.asarray(prog.vy(qt)) - xp.asarray(self._center.vy(qt))
            centervz = xp.asarray(prog.vz(qt)) - xp.asarray(self._center.vz(qt))
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

            def _scalar_mass_fn(t):
                # backend-agnostic ones_like(t): jit-safe for a traced t; numpy
                # byte-identical (M0 * float64 ones of t's shape). Resolve from M0 too
                # so a differentiable backend M broadcasts on a plain-float t (torch).
                xp_t = get_namespace(M0, t)
                return M0 * xp_t.ones_like(xp_t.asarray(t) * 1.0)

            self._progenitor_mass_fn = _scalar_mass_fn
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
            # A differentiable M(t) returns a backend array; detect it once so _mass_fn
            # keeps the backend result (jit-safe, traces the mass) rather than coercing
            # to numpy. A numpy callable takes the byte-identical branch unchanged.
            _mass_backend_out = is_backend_array(progenitor_mass(0.0))

            def _mass_fn(t):
                if _mass_backend_out or is_backend_array(t):
                    return progenitor_mass(t)
                return numpy.asarray(
                    progenitor_mass(numpy.asarray(t, dtype=float)),
                    dtype=float,
                )

        self._progenitor_mass_fn = _mass_fn
        _pm0 = self._progenitor_mass_fn(0.0)
        self._progenitor_mass = _pm0 if is_backend_array(_pm0) else float(_pm0)

    def spray_df(self, xyzpt, vxyzpt, dt, leading=True, key=None):
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
        integrate_kwargs=None,
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
            integrate_kwargs=integrate_kwargs,
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

    def spray_df(self, xyzpt, vxyzpt, dt, leading=True, key=None):
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
        key : optional
            Backend random key for the offset draw. Default None uses ``numpy.random``
            (byte-identical); a jax/torch key makes it a reparameterized function of
            the key (required for a correct gradient under jit).

        Returns
        -------
        xst, yst, zst : array, shape (N,)
            Positions of points on the stream in the progenitor coordinates.
        vxst, vyst, vzst : array, shape (N,)
            Velocities of points on the stream in the progenitor coordinates.
        """
        from ..backend import random as grandom

        xp = get_namespace(dt)
        Rpt, phipt, Zpt = coords.rect_to_cyl(xyzpt[:, 0], xyzpt[:, 1], xyzpt[:, 2])
        rtides = self._calc_rtide(Rpt, phipt, Zpt, dt)

        # Sample positions and velocities in the instantaneous frame. numpy key
        # (None) -> numpy.random (byte-identical); a backend key -> reparameterized.
        posvel = xp.asarray(
            grandom.multivariate_normal(key, self._mean, self._cov, shape=len(dt))
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
        integrate_kwargs=None,
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
            integrate_kwargs=integrate_kwargs,
            ro=ro,
            vo=vo,
        )
        self._meankvec = numpy.array(meankvec)
        self._sigkvec = numpy.array(sigkvec)
        return None

    def spray_df(self, xyzpt, vxyzpt, dt, leading=True, key=None):
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
        key : optional
            Backend random key for the action-angle offset draw. Default None uses
            ``numpy.random`` (byte-identical). A jax/torch key makes the draw a
            reproducible, theta-independent function of the key (reparameterization
            -- required for a correct gradient under jit).

        Returns
        -------
        xst, yst, zst : array, shape (N,)
            Positions of points on the stream in the progenitor coordinates.
        vxst, vyst, vzst : array, shape (N,)
            Velocities of points on the stream in the progenitor coordinates.
        """
        from ..backend import random as grandom

        xp = get_namespace(dt)
        Rpt, phipt, Zpt = coords.rect_to_cyl(xyzpt[:, 0], xyzpt[:, 1], xyzpt[:, 2])
        rtides = self._calc_rtide(Rpt, phipt, Zpt, dt)
        vcs = self._calc_vc(Rpt, phipt, Zpt, dt)
        rtides_as_frac = rtides / Rpt

        vRpt, vTpt, vZpt = coords.rect_to_cyl_vec(
            vxyzpt[:, 0], vxyzpt[:, 1], vxyzpt[:, 2], Rpt, phipt, Zpt, cyl=True
        )
        # Sample the action-angle offsets. numpy key (None) -> numpy.random draw
        # (byte-identical); a backend key -> a reparameterized (theta-independent)
        # backend draw, so the gradient flows through the theta-dependent rtides/vcs
        # and is consistent between AD and FD under jit.
        meankvec = as_backend_constant(
            xp, -self._meankvec if leading else self._meankvec, Rpt
        )
        sigkvec = as_backend_constant(xp, self._sigkvec, Rpt)
        k = meankvec + xp.asarray(grandom.normal(key, (len(dt), 6))) * sigkvec

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
    # A backend sigma is a plain (unitless, internal-unit) array -- keep it on the backend
    # so the returned pdf is differentiable in it; the unit parser is numpy-only.
    sigma_internal = (
        sigma if is_backend_array(sigma) else conversion.parse_time(sigma, ro=ro, vo=vo)
    )
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
    # Backend mode: a backend `sigma` (fit the stripping width) -- or a backend pot / IC, so
    # the pdf composes with a differentiable, jittable backend spdf -- returns a backend
    # Gaussian-mixture pdf. The pericenter TIMES are found concretely (find_peaks is a
    # discrete numpy op) and frozen as a backend constant; the mixture is differentiable in
    # sigma. A pure-numpy call falls through to _pdf_internal below (byte-identical).
    _peri_xp = None
    if not sigma_is_quantity:
        if is_backend_array(sigma):
            _peri_xp = get_namespace(sigma)
        elif is_backend_array(getattr(progenitor, "_ic_backend", None)):
            _peri_xp = get_namespace(progenitor._ic_backend)
        else:
            # a genuine backend pot parameter survives a forced-numpy force eval
            with use("numpy", force=True):
                _tf = evaluateRforces(
                    pot,
                    float(prog_copy.R(0.0)),
                    float(prog_copy.z(0.0)),
                    phi=float(prog_copy.phi(0.0)),
                    v=numpy.array(
                        [
                            float(prog_copy.vR(0.0)),
                            float(prog_copy.vT(0.0)),
                            float(prog_copy.vz(0.0)),
                        ]
                    ),
                )
            if is_backend_array(_tf):
                _peri_xp = get_namespace(_tf)
    if _peri_xp is not None:
        _sqrt_2pi = float(numpy.sqrt(2.0 * numpy.pi))
        peri_b = _peri_xp.asarray(peri_times)
        sigma_b = (
            sigma_internal
            if is_backend_array(sigma_internal)
            else _peri_xp.asarray(sigma_internal)
        )

        def _pdf_backend(t):
            xp = get_namespace(t, sigma_b, peri_b)
            t_arr = xp.reshape(xp.asarray(t) * 1.0, (-1,))
            norm = 1.0 / (peri_b.shape[0] * sigma_b * _sqrt_2pi)
            dx = (t_arr[:, None] - peri_b[None, :]) / sigma_b
            out = norm * xp.sum(xp.exp(-0.5 * dx * dx), axis=-1)
            out = xp.where(
                (t_arr >= -tdisrupt_internal) & (t_arr <= 0.0), out, xp.zeros_like(out)
            )
            return out[0] if getattr(t, "ndim", 0) == 0 else out

        _pdf_backend.pericenter_times = peri_b
        return _pdf_backend
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
