import numpy
from scipy import interpolate

from ..util import conversion, coords
from ..util._optional_deps import _APY_LOADED, _APY_UNITS

if _APY_LOADED:
    from astropy import units


def _particles_to_cartesian(xv_particles):
    """Convert particles from (R,vR,vT,z,vz,phi) to galactocentric
    Cartesian 6-vectors. Returns array of shape (N, 6)."""
    R, vR, vT, z, vz, phi = xv_particles
    x_p, y_p, z_p = coords.cyl_to_rect(R, phi, z)
    vx_p, vy_p, vz_p = coords.cyl_to_rect_vec(vR, vT, vz, phi)
    return numpy.column_stack([x_p, y_p, z_p, vx_p, vy_p, vz_p])


def _bin_by_tp(tp_assign, values, tp_nodes):
    """Bin per-particle ``values`` (N, D) by ``tp_assign`` onto ``tp_nodes``
    (M,). Bins are centered on each node. Returns ``(means (M, D), covs
    (M, D, D), counts (M,))``. Empty / size-1 bins get NaN mean and zero
    cov.
    """
    M = len(tp_nodes)
    dt_node = tp_nodes[1] - tp_nodes[0]
    edges = numpy.concatenate(
        (
            [tp_nodes[0] - 0.5 * dt_node],
            0.5 * (tp_nodes[:-1] + tp_nodes[1:]),
            [tp_nodes[-1] + 0.5 * dt_node],
        )
    )
    idx = numpy.clip(numpy.searchsorted(edges, tp_assign) - 1, 0, M - 1)
    D = values.shape[1]
    means = numpy.full((M, D), numpy.nan)
    covs = numpy.zeros((M, D, D))
    counts = numpy.zeros(M, dtype=int)
    for m in range(M):
        sel = idx == m
        k = int(sel.sum())
        counts[m] = k
        if k < 2:
            continue
        group = values[sel]
        means[m] = group.mean(axis=0)
        covs[m] = numpy.cov(group, rowvar=False)
    return means, covs, counts


def _smooth_series(x, y, sigma, s_user=None):
    """Fit a smoothing spline y(x) with weights derived from sigma.

    Default (``s_user=None``): use ``make_smoothing_spline`` with GCV
    auto-tuning (requires scipy >= 1.10). When ``s_user`` is a float, fall
    back to ``UnivariateSpline`` with that explicit ``s`` value, giving the
    user full control over smoothness.

    Falls back to linear interpolation for fewer than 5 valid bins (the
    minimum required by the GCV routine).
    """
    mask = numpy.isfinite(y) & numpy.isfinite(x)
    n_valid = int(mask.sum())
    order = numpy.argsort(x[mask])
    xv = x[mask][order]
    yv = y[mask][order]
    if n_valid < 5:
        if n_valid < 2:  # pragma: no cover (defensive: <2 valid bins)
            ref = float(yv[0]) if n_valid == 1 else 0.0
            xv = numpy.array([-1.0, 0.0])
            yv = numpy.array([ref, ref])
        return interpolate.interp1d(
            xv,
            yv,
            kind="linear",
            fill_value="extrapolate",
            assume_sorted=True,
        )
    sig_safe = numpy.where(
        numpy.isfinite(sigma) & (sigma > 0),
        sigma,
        numpy.nan,
    )
    with numpy.errstate(invalid="ignore"):
        sig_med = (
            numpy.nanmedian(sig_safe)
            if numpy.any(numpy.isfinite(sig_safe))
            else numpy.nan
        )
    if not numpy.isfinite(sig_med) or sig_med == 0:  # pragma: no cover
        sig_med = 1.0
    sv = numpy.where(numpy.isfinite(sig_safe), sig_safe, sig_med)[mask][order]
    sv = numpy.maximum(sv, 1e-12)
    if s_user is None:
        # GCV auto-tuning: weights are inverse-variances (1/sigma^2)
        return interpolate.make_smoothing_spline(xv, yv, w=1.0 / (sv * sv))
    return interpolate.UnivariateSpline(xv, yv, w=1.0 / sv, s=float(s_user), k=3)


def _closest_point_on_curve(points, curve, curve_t, mask=None):
    """For each point in ``points`` (N, D), find the index of the closest
    entry in ``curve`` (M, D) and return the corresponding time
    ``curve_t[idx]``. Works for any dimensionality D (3 for position-only,
    6 for position+velocity).

    If ``mask`` of shape (N, M) is given, entries where mask is False are
    excluded from the search on a per-point basis.
    """
    diff = points[:, None, :] - curve[None, :, :]
    d2 = numpy.einsum("nmk,nmk->nm", diff, diff)
    if mask is not None:
        d2 = numpy.where(mask, d2, numpy.inf)
    idx = numpy.argmin(d2, axis=1)
    return curve_t[idx]


class StreamTrack:
    """Smooth phase-space track for a single arm of a stream.

    A StreamTrack holds a smooth mean curve (and optional covariance) in
    galactic phase space, parameterized by a progenitor time coordinate
    ``tp``. ``tp=0`` is the progenitor today; for a leading arm ``tp > 0``
    (future positions the progenitor has yet to reach) and for a trailing
    arm ``tp < 0``. The ``tp`` range is determined by the data: it spans
    the percentile-trimmed range of the closest-point assignments, bounded
    by ``track_time_range`` (typically much smaller than ``tdisrupt``).

    Notes
    -----
    This class is intentionally light-weight and generic: while currently
    constructed by ``basestreamspraydf.streamTrack``, it could serve as a
    shared backend for other stream methods' track representations. A
    follow-up PR could refactor parts of ``streamdf``'s track machinery
    to use this class.
    """

    def __init__(
        self,
        xv_particles,
        dt_particles,
        track_prog_cart,
        track_t_grid,
        pot=None,
        arm_sign=1,
        ntp=None,
        ninterp=1001,
        smoothing=None,
        niter=0,
        order=2,
        ro=None,
        vo=None,
        zo=None,
        solarmotion=None,
        roSet=True,
        voSet=True,
    ):
        """
        Parameters
        ----------
        xv_particles : array, shape (6, N)
            Present-day phase space (R, vR, vT, z, vz, phi) of stream particles
            in galpy internal units.
        dt_particles : array, shape (N,)
            Stripping times (positive, galpy internal time units) for each
            particle. Used as a windowing prior when assigning progenitor-time
            coordinates via closest-point projection.
        track_prog_cart : array, shape (M, 6)
            Finely-sampled progenitor phase space (x, y, z, vx, vy, vz) at
            the times given by ``track_t_grid``. The progenitor must cover
            both sides of ``tp=0``: ``tp=0`` is the progenitor today,
            ``tp<0`` is past, ``tp>0`` is future. Because stream particles
            have small velocity offsets from the progenitor, they lie
            spatially close to a short arc of the progenitor orbit and the
            relevant ``tp`` range is much smaller than ``tdisrupt``.
        track_t_grid : array, shape (M,)
            The dense time grid on which ``track_prog_cart`` is evaluated.
            Used for the closest-point projection.
        arm_sign : int, optional
            ``+1`` for leading arm (tp >= 0), ``-1`` for trailing (tp <= 0).
            Controls the sign constraint on the closest-point search window.
        ntp : int, optional
            Number of binning nodes. Default ``sqrt(N)`` clipped to
            ``[21, 201]``.
        ninterp : int, optional
            Resolution of the fine tp grid on which the public track is
            stored.
        smoothing : None, float, or dict, optional
            Smoothing parameter for the mean spline per Cartesian coordinate.
            ``None`` (default) uses GCV auto-tuning via
            ``scipy.interpolate.make_smoothing_spline``. A float sets an
            explicit ``s`` for ``UnivariateSpline``; a dict keyed by
            ``'x','y','z','vx','vy','vz'`` sets per-coordinate ``s`` values.
        niter : int, optional
            Iterations beyond the initial fit. Each iteration reassigns each
            particle to the closest point on the current track and refits.
        order : int, optional
            Moments to keep: 1 = mean only, 2 = mean + covariance.
        ro, vo, zo, solarmotion, roSet, voSet
            Physical-unit configuration (typically inherited from the
            progenitor Orbit).
        """
        self._track_t_grid = numpy.asarray(track_t_grid, dtype=float)
        self._prog_cart = numpy.asarray(track_prog_cart, dtype=float)
        # Interpolating splines for the progenitor phase space; needed to
        # evaluate at arbitrary tp values during the fit and reconstruction.
        self._prog_splines = [
            interpolate.InterpolatedUnivariateSpline(
                self._track_t_grid, self._prog_cart[:, i], k=3
            )
            for i in range(6)
        ]
        self._arm_sign = int(numpy.sign(arm_sign)) or 1
        self._ninterp = int(ninterp)
        self._order = int(order)
        self._ro = ro
        self._vo = vo
        self._zo = zo
        self._solarmotion = solarmotion
        self._roSet = roSet
        self._voSet = voSet
        # Inherit progenitor's physical-output state: if both ro and vo are
        # set on the progenitor, the track starts with physical output on.
        self._physical = bool(roSet and voSet)

        # Particles in galactocentric Cartesian
        self._particles_cart = _particles_to_cartesian(
            numpy.asarray(xv_particles, dtype=float)
        )
        self._dt = numpy.asarray(dt_particles, dtype=float)

        # Normalize smoothing argument: tuple of 6 optional floats aligned
        # with (x, y, z, vx, vy, vz).
        if smoothing is None:
            self._s_user = (None,) * 6
        elif isinstance(smoothing, dict):
            self._s_user = tuple(
                smoothing.get(c, None) for c in ("x", "y", "z", "vx", "vy", "vz")
            )
        else:
            self._s_user = (float(smoothing),) * 6

        # Initial assignment via 6D closest-point on the progenitor orbit,
        # windowed by dt and by arm sign (leading: tp>=0, trailing: tp<=0).
        tp_assign = self._assign_closest_on_progenitor()

        # Exclude particles at the far boundary (mismatched).
        if self._arm_sign > 0:
            far_edge = self._track_t_grid[-1]
        else:
            far_edge = self._track_t_grid[0]
        arc_span = abs(self._track_t_grid[-1] - self._track_t_grid[0])
        interior = numpy.abs(tp_assign - far_edge) > 1e-3 * arc_span
        tp_assign = tp_assign[interior]
        self._particles_cart = self._particles_cart[interior]
        self._dt = self._dt[interior]

        # --- Stream-orbit refinement ---
        # Use the initial progenitor-offset tp_assign to identify the
        # stream tip's mean phase space. Integrate a NEW orbit from the
        # tip back to tp=0 (or the reverse). This "stream orbit" follows
        # the actual stream's mean path and gives smaller, more slowly-
        # varying offsets than the progenitor orbit — especially for warm
        # streams whose mean orbit differs from the progenitor's.
        self._refine_base_orbit(tp_assign, pot)

        # Percentile-trim the tp_grid
        trim_percentile = 99.0
        if self._arm_sign > 0:
            tp_lo = 0.0
            tp_hi = float(numpy.percentile(tp_assign, trim_percentile))
        else:
            tp_hi = 0.0
            tp_lo = float(numpy.percentile(tp_assign, 100.0 - trim_percentile))
        if tp_hi - tp_lo < 1e-12:  # pragma: no cover (defensive)
            tp_lo = float(self._track_t_grid[0])
            tp_hi = float(self._track_t_grid[-1])
        self._tp_grid = numpy.linspace(tp_lo, tp_hi, self._ninterp)

        n_part = self._particles_cart.shape[0]
        if ntp is None:
            ntp = int(max(21, min(201, round(numpy.sqrt(n_part)))))
        self._tp_nodes = numpy.linspace(tp_lo, tp_hi, int(ntp))

        # Re-assign particles to the stream orbit and fit
        tp_assign = self._assign_closest_on_base()
        # Re-trim after re-assignment
        interior2 = numpy.abs(tp_assign - far_edge) > 1e-3 * arc_span
        tp_assign = tp_assign[interior2]
        self._particles_cart = self._particles_cart[interior2]
        self._dt = self._dt[interior2]

        for it in range(niter + 1):
            self._fit(tp_assign)
            if it < niter:
                tp_assign = self._assign_closest_on_track()

    # -----------------------------------------------------------------
    # Stream-orbit refinement
    # -----------------------------------------------------------------
    def _refine_base_orbit(self, tp_assign, pot):
        """Replace the progenitor orbit with a 'stream orbit' integrated
        from the stream tip's mean phase space. For cold streams the
        stream orbit ≈ progenitor orbit; for warm streams it can differ
        significantly and gives smaller smoothing offsets.
        """
        if pot is None:
            return  # can't integrate without a potential; keep progenitor
        from ..orbit import Orbit
        from ..util import coords as gc

        # Identify the tip: particles in the outermost 10% of |tp|
        tp_abs = numpy.abs(tp_assign)
        cutoff = numpy.percentile(tp_abs, 90)
        tip_mask = tp_abs >= cutoff
        if tip_mask.sum() < 5:
            return  # too few particles to define a tip
        tip_mean = self._particles_cart[tip_mask].mean(axis=0)  # (6,)

        # Convert tip from galactocentric Cartesian to cylindrical for Orbit
        R, phi, z = gc.rect_to_cyl(tip_mean[0], tip_mean[1], tip_mean[2])
        vR, vT, vz = gc.rect_to_cyl_vec(
            tip_mean[3], tip_mean[4], tip_mean[5], R, phi, z, cyl=True
        )
        tip_orb = Orbit([R, vR, vT, z, vz, phi])
        tip_orb.turn_physical_off()

        # Integrate from tip back to the progenitor (and a bit beyond)
        # by reversing time. The tp range we need is from the tip's tp
        # back to tp=0. For leading (arm_sign=+1), tip is at large
        # positive tp; integrate backward. For trailing, tip is at large
        # negative tp; integrate forward.
        tp_tip = float(numpy.median(tp_assign[tip_mask]))
        n_dense = len(self._track_t_grid)
        if self._arm_sign > 0:
            # tip at +tp_tip, need orbit from tp_tip down to 0
            t_grid = numpy.linspace(0.0, -tp_tip, n_dense)
            tip_orb.integrate(t_grid, pot)
            # Evaluate: at t=0 we're at the tip; at t=-tp_tip we're at
            # the progenitor end. Map to tp by shifting: tp = tp_tip + t
            stream_t = tp_tip + t_grid  # goes from tp_tip to 0
        else:
            t_grid = numpy.linspace(0.0, -tp_tip, n_dense)
            tip_orb.integrate(t_grid, pot)
            stream_t = tp_tip + t_grid  # goes from tp_tip to 0

        # Build the stream orbit's phase-space array
        stream_cart = numpy.column_stack(
            [
                tip_orb.x(t_grid),
                tip_orb.y(t_grid),
                tip_orb.z(t_grid),
                tip_orb.vx(t_grid),
                tip_orb.vy(t_grid),
                tip_orb.vz(t_grid),
            ]
        )
        # Sort by stream_t (ascending)
        order = numpy.argsort(stream_t)
        self._track_t_grid = stream_t[order]
        self._prog_cart = stream_cart[order]
        self._prog_splines = [
            interpolate.InterpolatedUnivariateSpline(
                self._track_t_grid, self._prog_cart[:, i], k=3
            )
            for i in range(6)
        ]

    def _assign_closest_on_base(self):
        """Like _assign_closest_on_progenitor but uses the (possibly
        refined) base orbit stored in self._prog_cart."""
        t_grid = self._track_t_grid
        sign_mask = (t_grid * self._arm_sign) >= 0
        dt_mask = numpy.abs(t_grid)[None, :] <= self._dt[:, None]
        mask = sign_mask[None, :] & dt_mask
        return _closest_point_on_curve(
            self._particles_cart, self._prog_cart, t_grid, mask=mask
        )

    # -----------------------------------------------------------------
    # Fitting
    # -----------------------------------------------------------------
    def _prog_at(self, tp):
        """Evaluate the track-progenitor 6D phase space at arbitrary tp."""
        tp = numpy.atleast_1d(tp)
        return numpy.column_stack([spl(tp) for spl in self._prog_splines])

    def _fit(self, tp_assign):
        """Bin per-particle offsets-from-progenitor by tp_assign, smooth the
        bin-means (plus optionally bin-covs), and reconstruct the public
        track by adding the smoothed offsets back to the progenitor orbit.
        Smoothing offsets (rather than raw positions) keeps the smoothed
        signal small and well-behaved regardless of orbital phase."""
        # Offsets = particle - progenitor(tp_i). Small by construction for
        # particles close to the orbit.
        prog_at_particles = self._prog_at(tp_assign)
        offsets = self._particles_cart - prog_at_particles  # (N, 6)

        means, covs, counts = _bin_by_tp(tp_assign, offsets, self._tp_nodes)
        with numpy.errstate(invalid="ignore"):
            per_coord_sigma = numpy.sqrt(numpy.diagonal(covs, axis1=1, axis2=2))
            sigma_of_mean = per_coord_sigma / numpy.sqrt(
                numpy.maximum(counts[:, None], 1)
            )
            sigma_of_mean = numpy.where(counts[:, None] > 1, sigma_of_mean, numpy.nan)

        # Smooth per-coord offset. Evaluate on fine tp grid.
        offset_splines = [
            _smooth_series(
                self._tp_nodes,
                means[:, i],
                sigma_of_mean[:, i],
                s_user=self._s_user[i],
            )
            for i in range(6)
        ]
        offset_fine = numpy.column_stack([spl(self._tp_grid) for spl in offset_splines])

        # Add back the progenitor's phase space at each tp_grid point.
        prog_on_grid = self._prog_at(self._tp_grid)
        track_fine = prog_on_grid + offset_fine

        # Covariance: smooth componentwise in the offset frame (≡ Cartesian
        # offset from progenitor). Errors scale as sqrt((C_aa*C_bb+C_ab^2)/k).
        if self._order >= 2:
            cov_fine = numpy.zeros((self._ninterp, 6, 6))
            for a in range(6):
                for b in range(a, 6):
                    vals = covs[:, a, b]
                    diag_a = per_coord_sigma[:, a] ** 2
                    diag_b = per_coord_sigma[:, b] ** 2
                    with numpy.errstate(invalid="ignore"):
                        sigma_c = numpy.sqrt(
                            (diag_a * diag_b + vals**2) / numpy.maximum(counts, 2)
                        )
                    sigma_c = numpy.where(counts > 1, sigma_c, numpy.nan)
                    spl = _smooth_series(self._tp_nodes, vals, sigma_c, s_user=None)
                    val_fine = spl(self._tp_grid)
                    cov_fine[:, a, b] = val_fine
                    cov_fine[:, b, a] = val_fine
            cov_fine = numpy.nan_to_num(cov_fine, nan=0.0, posinf=0.0, neginf=0.0)
            for k in range(self._ninterp):
                evals, evecs = numpy.linalg.eigh(cov_fine[k])
                evals = numpy.clip(evals, 0.0, None)
                cov_fine[k] = (evecs * evals) @ evecs.T
            self._cov_xyz = cov_fine
        else:
            self._cov_xyz = None

        self._track_xyz = track_fine[:, 0:3]
        self._track_vxvyvz = track_fine[:, 3:6]
        self._bin_counts = counts

        # Interpolating splines on the public fine-grid track for evaluation
        self._cart_splines = [
            interpolate.InterpolatedUnivariateSpline(
                self._tp_grid, track_fine[:, i], k=3
            )
            for i in range(6)
        ]

    # -----------------------------------------------------------------
    # Assignment helpers
    # -----------------------------------------------------------------
    def _assign_closest_on_progenitor(self):
        """Assign each particle a tp via 6D closest-point projection
        (position + velocity) onto the progenitor orbit, windowed by arm
        sign and stripping time.

        Using 6D phase space instead of 3D position disambiguates matches
        at orbital self-intersections (same position, different velocity)
        and improves robustness for warm / fat streams.
        """
        t_grid = self._track_t_grid
        sign_mask = (t_grid * self._arm_sign) >= 0
        dt_mask = numpy.abs(t_grid)[None, :] <= self._dt[:, None]
        mask = sign_mask[None, :] & dt_mask
        return _closest_point_on_curve(
            self._particles_cart, self._prog_cart, t_grid, mask=mask
        )

    def _assign_closest_on_track(self):
        """Reassign each particle's tp to the closest point on the current
        smooth track (6D: position + velocity)."""
        track_cart = numpy.column_stack([self._track_xyz, self._track_vxvyvz])
        return _closest_point_on_curve(self._particles_cart, track_cart, self._tp_grid)

    # -----------------------------------------------------------------
    # Public evaluation
    # -----------------------------------------------------------------
    def tp_grid(self):
        """Return the fine tp grid on which the track is stored."""
        return self._tp_grid.copy()

    def _eval_cart(self, tp):
        tp = numpy.clip(numpy.atleast_1d(tp), self._tp_grid[0], self._tp_grid[-1])
        out = numpy.array([spl(tp) for spl in self._cart_splines])  # (6, len)
        return out

    def _maybe_scalar(self, tp, arr):
        if numpy.isscalar(tp) or (hasattr(tp, "ndim") and tp.ndim == 0):
            return arr[0]
        return arr

    def _parse_tp(self, tp):
        return conversion.parse_time(tp, ro=self._ro, vo=self._vo)

    def _scale(self, val, kind):
        """Apply physical-unit scaling to ``val`` for a given coordinate
        ``kind`` (one of 'length', 'velocity', 'angle', 'degree', 'pm',
        'vlos', 'kpc')."""
        if not self._physical:
            return val
        if kind == "length":
            return val * self._ro * (units.kpc if _APY_UNITS else 1)
        if kind == "velocity":
            return val * self._vo * (units.km / units.s if _APY_UNITS else 1)
        if kind == "vlos":
            # vlos from _vrpmllpmbb is already in km/s (helio_xv pre-
            # multiplies by vo); only attach units, don't re-scale.
            return val * (units.km / units.s if _APY_UNITS else 1)
        if kind == "angle":
            return val * (units.rad if _APY_UNITS else 1)
        if kind == "degree":
            return val * (units.deg if _APY_UNITS else 1)
        if kind == "kpc":
            return val * (units.kpc if _APY_UNITS else 1)
        # kind == "pm"
        return val * (units.mas / units.yr if _APY_UNITS else 1)

    def _cart_eval(self, idx, tp):
        # Clip tp to the track's valid range to prevent unbounded cubic-spline
        # extrapolation outside the data support.
        tp_arr = numpy.clip(numpy.atleast_1d(tp), self._tp_grid[0], self._tp_grid[-1])
        val = self._cart_splines[idx](tp_arr)
        return self._maybe_scalar(tp, val)

    def x(self, tp):
        """Galactocentric Cartesian x along the track."""
        tp = self._parse_tp(tp)
        return self._scale(self._cart_eval(0, tp), "length")

    def y(self, tp):
        tp = self._parse_tp(tp)
        return self._scale(self._cart_eval(1, tp), "length")

    def z(self, tp):
        tp = self._parse_tp(tp)
        return self._scale(self._cart_eval(2, tp), "length")

    def vx(self, tp):
        tp = self._parse_tp(tp)
        return self._scale(self._cart_eval(3, tp), "velocity")

    def vy(self, tp):
        tp = self._parse_tp(tp)
        return self._scale(self._cart_eval(4, tp), "velocity")

    def vz(self, tp):
        tp = self._parse_tp(tp)
        return self._scale(self._cart_eval(5, tp), "velocity")

    def _cyl_at(self, tp):
        """Return (R, vR, vT, z, vz, phi) along the track (internal units)."""
        tp = numpy.atleast_1d(tp)
        xyz = self._eval_cart(tp)  # (6, len)
        x, y, zc, vxc, vyc, vzc = xyz
        R, phi, zcyl = coords.rect_to_cyl(x, y, zc)
        vR, vT, vz = coords.rect_to_cyl_vec(vxc, vyc, vzc, R, phi, zcyl, cyl=True)
        return R, vR, vT, zcyl, vz, phi

    def R(self, tp):
        tp = self._parse_tp(tp)
        R, _, _, _, _, _ = self._cyl_at(tp)
        return self._scale(self._maybe_scalar(tp, R), "length")

    def vR(self, tp):
        tp = self._parse_tp(tp)
        _, vR, _, _, _, _ = self._cyl_at(tp)
        return self._scale(self._maybe_scalar(tp, vR), "velocity")

    def vT(self, tp):
        tp = self._parse_tp(tp)
        _, _, vT, _, _, _ = self._cyl_at(tp)
        return self._scale(self._maybe_scalar(tp, vT), "velocity")

    def phi(self, tp):
        tp = self._parse_tp(tp)
        _, _, _, _, _, phi = self._cyl_at(tp)
        return self._scale(self._maybe_scalar(tp, phi), "angle")

    def __call__(self, tp):
        """Return (R, vR, vT, z, vz, phi) stacked along the track at tp."""
        tp = self._parse_tp(tp)
        R, vR, vT, zcyl, vzc, phi = self._cyl_at(tp)
        out = numpy.array([R, vR, vT, zcyl, vzc, phi])
        if numpy.isscalar(tp) or (hasattr(tp, "ndim") and tp.ndim == 0):
            return out[:, 0]
        return out

    # -----------------------------------------------------------------
    # Heliocentric coordinate accessors
    # -----------------------------------------------------------------
    def _helio_xv(self, tp):
        """Compute heliocentric XYZ, vxvyvz at tp using ro/vo/zo/solarmotion."""
        xyzvxyz = self._eval_cart(numpy.atleast_1d(tp))  # (6, len)
        zo_kpc = self._zo if self._zo is not None else 0.0
        xyz_helio = coords.galcenrect_to_XYZ(
            xyzvxyz[0] * self._ro,
            xyzvxyz[1] * self._ro,
            xyzvxyz[2] * self._ro,
            Xsun=self._ro,
            Zsun=zo_kpc,
        )
        vxyz_helio = coords.galcenrect_to_vxvyvz(
            xyzvxyz[3] * self._vo,
            xyzvxyz[4] * self._vo,
            xyzvxyz[5] * self._vo,
            vsun=self._get_vsun_kms(),
            Xsun=self._ro,
            Zsun=zo_kpc,
        )
        return (
            xyz_helio[..., 0],
            xyz_helio[..., 1],
            xyz_helio[..., 2],
            vxyz_helio[..., 0],
            vxyz_helio[..., 1],
            vxyz_helio[..., 2],
        )

    def _get_vsun_kms(self):
        sm = numpy.asarray(self._solarmotion, dtype=float)
        return [sm[0], self._vo + sm[1], sm[2]]

    def _vrpmllpmbb(self, tp):
        X, Y, Z, vX, vY, vZ = self._helio_xv(tp)
        lbd = coords.XYZ_to_lbd(X, Y, Z, degree=True)
        vrpmllpmbb = coords.vxvyvz_to_vrpmllpmbb(
            vX, vY, vZ, lbd[:, 0], lbd[:, 1], lbd[:, 2], degree=True
        )
        return lbd, vrpmllpmbb

    def ra(self, tp):
        tp = self._parse_tp(tp)
        X, Y, Z, _, _, _ = self._helio_xv(tp)
        lbd = coords.XYZ_to_lbd(X, Y, Z, degree=True)
        ra_dec = coords.lb_to_radec(lbd[:, 0], lbd[:, 1], degree=True)
        return self._scale(self._maybe_scalar(tp, ra_dec[:, 0]), "degree")

    def dec(self, tp):
        tp = self._parse_tp(tp)
        X, Y, Z, _, _, _ = self._helio_xv(tp)
        lbd = coords.XYZ_to_lbd(X, Y, Z, degree=True)
        ra_dec = coords.lb_to_radec(lbd[:, 0], lbd[:, 1], degree=True)
        return self._scale(self._maybe_scalar(tp, ra_dec[:, 1]), "degree")

    def ll(self, tp):
        tp = self._parse_tp(tp)
        X, Y, Z, _, _, _ = self._helio_xv(tp)
        lbd = coords.XYZ_to_lbd(X, Y, Z, degree=True)
        return self._scale(self._maybe_scalar(tp, lbd[:, 0]), "degree")

    def bb(self, tp):
        tp = self._parse_tp(tp)
        X, Y, Z, _, _, _ = self._helio_xv(tp)
        lbd = coords.XYZ_to_lbd(X, Y, Z, degree=True)
        return self._scale(self._maybe_scalar(tp, lbd[:, 1]), "degree")

    def dist(self, tp):
        tp = self._parse_tp(tp)
        X, Y, Z, _, _, _ = self._helio_xv(tp)
        lbd = coords.XYZ_to_lbd(X, Y, Z, degree=True)
        return self._scale(self._maybe_scalar(tp, lbd[:, 2]), "kpc")

    def pmra(self, tp):
        tp = self._parse_tp(tp)
        lbd, vrpmllpmbb = self._vrpmllpmbb(tp)
        pmrapmdec = coords.pmllpmbb_to_pmrapmdec(
            vrpmllpmbb[:, 1],
            vrpmllpmbb[:, 2],
            lbd[:, 0],
            lbd[:, 1],
            degree=True,
        )
        return self._scale(self._maybe_scalar(tp, pmrapmdec[:, 0]), "pm")

    def pmdec(self, tp):
        tp = self._parse_tp(tp)
        lbd, vrpmllpmbb = self._vrpmllpmbb(tp)
        pmrapmdec = coords.pmllpmbb_to_pmrapmdec(
            vrpmllpmbb[:, 1],
            vrpmllpmbb[:, 2],
            lbd[:, 0],
            lbd[:, 1],
            degree=True,
        )
        return self._scale(self._maybe_scalar(tp, pmrapmdec[:, 1]), "pm")

    def pmll(self, tp):
        tp = self._parse_tp(tp)
        _, vrpmllpmbb = self._vrpmllpmbb(tp)
        return self._scale(self._maybe_scalar(tp, vrpmllpmbb[:, 1]), "pm")

    def pmbb(self, tp):
        tp = self._parse_tp(tp)
        _, vrpmllpmbb = self._vrpmllpmbb(tp)
        return self._scale(self._maybe_scalar(tp, vrpmllpmbb[:, 2]), "pm")

    def vlos(self, tp):
        tp = self._parse_tp(tp)
        _, vrpmllpmbb = self._vrpmllpmbb(tp)
        return self._scale(self._maybe_scalar(tp, vrpmllpmbb[:, 0]), "vlos")

    # -----------------------------------------------------------------
    # Covariance
    # -----------------------------------------------------------------
    def cov(self, tp):
        """Return the 6x6 covariance matrix of the particle distribution at tp,
        in the galactocentric Cartesian basis (x, y, z, vx, vy, vz)."""
        if self._cov_xyz is None:
            raise RuntimeError(
                "Covariance was not computed (order < 2). Rebuild with order=2."
            )
        tp = self._parse_tp(tp)
        tp_arr = numpy.atleast_1d(tp)
        # Linear interpolation on the stored fine grid; covariance is already
        # stored on a dense 1001-point grid so linear is fine.
        out = numpy.empty((len(tp_arr), 6, 6))
        for a in range(6):
            for b in range(6):
                out[:, a, b] = numpy.interp(
                    tp_arr, self._tp_grid, self._cov_xyz[:, a, b]
                )
        if numpy.isscalar(tp) or (hasattr(tp, "ndim") and tp.ndim == 0):
            return out[0]
        return out

    # -----------------------------------------------------------------
    # Unit toggles
    # -----------------------------------------------------------------
    def turn_physical_on(self, ro=None, vo=None):
        if ro is not None:
            self._ro = conversion.parse_length_kpc(ro)
            self._roSet = True
        if vo is not None:
            self._vo = conversion.parse_velocity_kms(vo)
            self._voSet = True
        self._physical = True

    def turn_physical_off(self):
        self._physical = False

    # -----------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------
    def plot(self, d1="x", d2="y", spread=0, n=None, **kwargs):
        """Plot the smooth track in the (d1, d2) plane.

        Parameters
        ----------
        d1, d2 : str
            Coordinate names. Any of: x, y, z, vx, vy, vz, R, vR, vT, phi,
            ra, dec, dist, ll, bb, pmra, pmdec, pmll, pmbb, vlos.
        spread : float, optional
            If > 0 and d1/d2 are among the Cartesian phase-space axes,
            draw a ±spread·sigma band using the projected covariance.
        n : int, optional
            Number of evaluation points (default: self._ninterp).
        **kwargs
            Passed to matplotlib.pyplot.plot.
        """
        from matplotlib import pyplot

        n_eval = self._ninterp if n is None else int(n)
        tp = numpy.linspace(self._tp_grid[0], self._tp_grid[-1], n_eval)
        v1 = getattr(self, d1)(tp)
        v2 = getattr(self, d2)(tp)
        v1 = numpy.asarray(v1)
        v2 = numpy.asarray(v2)
        line = pyplot.plot(v1, v2, **kwargs)
        if spread > 0 and self._cov_xyz is not None:
            cart_idx = {"x": 0, "y": 1, "z": 2, "vx": 3, "vy": 4, "vz": 5}
            if d2 in cart_idx:
                i2 = cart_idx[d2]
                cov = self.cov(tp)  # (n_eval, 6, 6)
                s2 = numpy.sqrt(numpy.maximum(cov[:, i2, i2], 0.0))
                # Scale covariance sigma the same way _scale handles the
                # coordinate values (check _physical only, consistent with
                # the coordinate accessors).
                if self._physical:
                    if d2 in ("x", "y", "z"):
                        s2 = s2 * self._ro
                    elif d2 in ("vx", "vy", "vz"):
                        s2 = s2 * self._vo
                color = line[0].get_color() if line else None
                pyplot.fill_between(
                    v1,
                    v2 - spread * s2,
                    v2 + spread * s2,
                    alpha=0.2,
                    color=color,
                )
        return line


class StreamTrackPair:
    """Container holding leading and trailing :class:`StreamTrack` instances.

    The individual arms are accessible as ``.leading`` and ``.trailing``.
    """

    def __init__(self, leading, trailing):
        self.leading = leading
        self.trailing = trailing

    def turn_physical_on(self, ro=None, vo=None):
        self.leading.turn_physical_on(ro=ro, vo=vo)
        self.trailing.turn_physical_on(ro=ro, vo=vo)

    def turn_physical_off(self):
        self.leading.turn_physical_off()
        self.trailing.turn_physical_off()

    def plot(self, d1="x", d2="y", spread=0, n=None, **kwargs):
        """Plot both arms on the same axes."""
        return [
            self.leading.plot(d1=d1, d2=d2, spread=spread, n=n, **kwargs),
            self.trailing.plot(d1=d1, d2=d2, spread=spread, n=n, **kwargs),
        ]
