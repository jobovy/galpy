import numpy
from scipy import interpolate

from ..util import conversion, coords
from ..util._optional_deps import _APY_LOADED, _APY_UNITS

if _APY_LOADED:
    from astropy import units


def _jac_galcenrect_to_galcencyl(x, y, z, vx, vy, vz):
    """Closed-form 6x6 Jacobian of galactocentric (R, vR, vT, z, vz, phi)
    w.r.t. (x, y, z, vx, vy, vz), evaluated at the given point."""
    R = numpy.sqrt(x * x + y * y)
    if R == 0.0:  # pragma: no cover (track never sits exactly on z axis)
        R = 1e-30
    cp = x / R
    sp = y / R
    vR_ = cp * vx + sp * vy
    vT_ = -sp * vx + cp * vy
    J = numpy.zeros((6, 6))
    # row 0: R
    J[0, 0] = cp
    J[0, 1] = sp
    # row 1: vR = cos_phi * vx + sin_phi * vy
    J[1, 0] = -sp * vT_ / R
    J[1, 1] = cp * vT_ / R
    J[1, 3] = cp
    J[1, 4] = sp
    # row 2: vT = -sin_phi * vx + cos_phi * vy
    J[2, 0] = sp * vR_ / R
    J[2, 1] = -cp * vR_ / R
    J[2, 3] = -sp
    J[2, 4] = cp
    # row 3: z
    J[3, 2] = 1.0
    # row 4: vz
    J[4, 5] = 1.0
    # row 5: phi = atan2(y, x)
    J[5, 0] = -sp / R
    J[5, 1] = cp / R
    return J


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
    auto-tuning (requires scipy >= 1.10). When ``s_user`` is a float, use
    ``UnivariateSpline`` with that explicit ``s`` value.

    Returns ``(spline, effective_s)`` where ``effective_s`` is the
    weighted residual sum of the fit (= the ``s`` that a
    ``UnivariateSpline`` would need to reproduce the same fit). For
    GCV fits this is computed from the residuals; for explicit-s fits
    it equals the input ``s_user``.

    Falls back to linear interpolation (effective_s=0) for fewer than 5
    valid bins.
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
        return (
            interpolate.interp1d(
                xv,
                yv,
                kind="linear",
                fill_value="extrapolate",
                assume_sorted=True,
            ),
            0.0,
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
        spl = interpolate.make_smoothing_spline(xv, yv, w=1.0 / (sv * sv))
        resid = yv - spl(xv)
        eff_s = float(numpy.sum((resid / sv) ** 2))
        return spl, eff_s
    s_val = float(s_user)
    return interpolate.UnivariateSpline(xv, yv, w=1.0 / sv, s=s_val, k=3), s_val


def _closest_point_on_curve(points, curve, curve_t, mask=None):
    """For each point in ``points`` (N, D), find the index of the closest
    entry in ``curve`` (M, D) and return the corresponding time
    ``curve_t[idx]``. Works for any dimensionality D.

    Uses ``scipy.spatial.cKDTree`` for the unmasked case (sublinear
    per query). When a per-particle ``mask`` of shape (N, M) is given,
    queries K nearest neighbors and picks the closest allowed one,
    growing K until every point has a match.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(curve)
    if mask is None:
        _, idx = tree.query(points, k=1)
        return curve_t[idx]
    N, M = mask.shape
    k = min(max(1, M // 64), M)
    tp_out = numpy.empty(N, dtype=float)
    remaining = numpy.arange(N)
    while remaining.size:
        _, cand = tree.query(points[remaining], k=min(k, M))
        if cand.ndim == 1:
            cand = cand[:, None]
        chosen = numpy.full(remaining.size, -1, dtype=int)
        for j, r in enumerate(remaining):
            allowed = cand[j][mask[r][cand[j]]]
            if allowed.size:
                chosen[j] = allowed[0]
        hit = chosen >= 0
        tp_out[remaining[hit]] = curve_t[chosen[hit]]
        remaining = remaining[~hit]
        if k >= M:
            tp_out[remaining] = 0.0
            break
        k = min(k * 4, M)
    return tp_out


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
        arm_sign=1,
        ntp=None,
        ninterp=1001,
        smoothing=None,
        niter=0,
        order=2,
        custom_transform=None,
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
        smoothing : None, float, array-like, or dict, optional
            Smoothing parameter(s) for the mean-offset splines.
            ``None`` (default) uses GCV auto-tuning via
            ``scipy.interpolate.make_smoothing_spline``. A float sets a
            single explicit ``s`` for all six coordinates. An array-like
            of length 6 sets per-coordinate ``s`` values (order:
            x, y, z, vx, vy, vz). A dict keyed by coordinate name does
            the same. After fitting, the effective per-spline ``s``
            values are available via :attr:`smoothing_s` (length 6 for
            mean, or 6+21=27 when ``order >= 2``), which can be passed
            back as ``smoothing`` in a subsequent call to reproduce the
            same smoothness without re-running GCV.
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
        # Optional rotation from (ra, dec) to a custom (phi1, phi2) frame —
        # enables the .phi1/.phi2/.pmphi1/.pmphi2 accessors. Inverse is
        # the transpose (orthogonal by construction).
        self._custom_transform = (
            None
            if custom_transform is None
            else numpy.asarray(custom_transform, dtype=float)
        )
        self._ro = ro
        self._vo = vo
        self._zo = zo
        self._solarmotion = solarmotion
        self._roSet = roSet
        self._voSet = voSet
        # Inherit progenitor's physical-output state: if both ro and vo are
        # set on the progenitor, the track starts with physical output on.
        self._physical = bool(roSet and voSet)

        # Raw (pre-filter) particles exposed to the user: same (xv, dt)
        # tuple format that ``streamTrack(particles=...)`` accepts — the
        # converse of that input knob, so callers can plot the track over
        # the sample it was built from without resampling.
        self.particles = (
            numpy.asarray(xv_particles, dtype=float).copy(),
            numpy.asarray(dt_particles, dtype=float).copy(),
        )

        # Particles in galactocentric Cartesian
        self._particles_cart = _particles_to_cartesian(
            numpy.asarray(xv_particles, dtype=float)
        )
        self._dt = numpy.asarray(dt_particles, dtype=float)

        # Normalize smoothing argument into per-spline s values.
        # Length 6 = mean only; length 27 = mean (6) + covariance (21).
        _coord_keys = ("x", "y", "z", "vx", "vy", "vz")
        if smoothing is None:
            self._s_user_mean = [None] * 6
            self._s_user_cov = [None] * 21
        elif isinstance(smoothing, dict):
            self._s_user_mean = [smoothing.get(c, None) for c in _coord_keys]
            self._s_user_cov = [None] * 21
        elif hasattr(smoothing, "__len__") and not isinstance(smoothing, str):
            s_arr = list(smoothing)
            self._s_user_mean = [None if v is None else float(v) for v in s_arr[:6]]
            self._s_user_cov = [
                None if v is None else float(v)
                for v in (s_arr[6:27] if len(s_arr) > 6 else [None] * 21)
            ]
        else:
            self._s_user_mean = [float(smoothing)] * 6
            self._s_user_cov = [None] * 21

        # Initial assignment via 6D closest-point on the progenitor orbit,
        # windowed by dt and by arm sign (leading: tp>=0, trailing: tp<=0).
        tp_assign = self._assign_closest_on_progenitor()

        # Exclude particles whose tp landed at the far boundary of the
        # progenitor arc — these are mismatched (their true closest point
        # is outside the searched range) and would bias the track tips.
        if self._arm_sign > 0:
            far_edge = self._track_t_grid[-1]
        else:
            far_edge = self._track_t_grid[0]
        arc_span = abs(self._track_t_grid[-1] - self._track_t_grid[0])
        interior = numpy.abs(tp_assign - far_edge) > 1e-3 * arc_span
        tp_assign = tp_assign[interior]
        self._particles_cart = self._particles_cart[interior]
        self._dt = self._dt[interior]

        # Trim the public tp grid to the percentile range where the binned
        # data supports a fit (outlier particles at large |tp| produce
        # sparse boundary bins that the spline would have to honor).
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

        # Binning nodes. Default: ~sqrt(N), clipped to [21, 201].
        n_part = self._particles_cart.shape[0]
        if ntp is None:
            ntp = int(max(21, min(201, round(numpy.sqrt(n_part)))))
        self._tp_nodes = numpy.linspace(tp_lo, tp_hi, int(ntp))

        for it in range(niter + 1):
            self._fit(tp_assign)
            if it < niter:
                tp_assign = self._assign_closest_on_track()

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

        After fitting, ``self.smoothing_s`` contains the effective s
        values: 6 floats for the mean splines, plus 21 for the
        covariance upper-triangle entries when ``order >= 2``."""
        prog_at_particles = self._prog_at(tp_assign)
        offsets = self._particles_cart - prog_at_particles

        means, covs, counts = _bin_by_tp(tp_assign, offsets, self._tp_nodes)
        with numpy.errstate(invalid="ignore"):
            per_coord_sigma = numpy.sqrt(numpy.diagonal(covs, axis1=1, axis2=2))
            sigma_of_mean = per_coord_sigma / numpy.sqrt(
                numpy.maximum(counts[:, None], 1)
            )
            sigma_of_mean = numpy.where(counts[:, None] > 1, sigma_of_mean, numpy.nan)

        # Mean-offset splines (6 coords)
        eff_s_mean = []
        offset_splines = []
        for i in range(6):
            spl, es = _smooth_series(
                self._tp_nodes,
                means[:, i],
                sigma_of_mean[:, i],
                s_user=self._s_user_mean[i],
            )
            offset_splines.append(spl)
            eff_s_mean.append(es)
        offset_fine = numpy.column_stack([spl(self._tp_grid) for spl in offset_splines])

        prog_on_grid = self._prog_at(self._tp_grid)
        track_fine = prog_on_grid + offset_fine

        # Covariance splines (21 upper-triangle entries)
        eff_s_cov = []
        if self._order >= 2:
            cov_fine = numpy.zeros((self._ninterp, 6, 6))
            cov_idx = 0
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
                    spl, es = _smooth_series(
                        self._tp_nodes,
                        vals,
                        sigma_c,
                        s_user=self._s_user_cov[cov_idx],
                    )
                    eff_s_cov.append(es)
                    cov_idx += 1
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

        self.smoothing_s = eff_s_mean + eff_s_cov

        self._track_xyz = track_fine[:, 0:3]
        self._track_vxvyvz = track_fine[:, 3:6]
        self._bin_counts = counts

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
    # Custom sky frame (requires ``custom_transform`` to be set)
    # -----------------------------------------------------------------
    def _require_custom(self):
        if self._custom_transform is None:
            raise RuntimeError(
                "custom_transform was not set at track construction; "
                "the phi1/phi2/pmphi1/pmphi2 accessors require a rotation "
                "matrix (3x3) from (ra, dec) to the custom sky frame. "
                "See galpy.util.coords.align_to_orbit for a helper that "
                "builds one from a progenitor Orbit."
            )

    def phi1(self, tp):
        """Custom-frame longitude."""
        self._require_custom()
        tp = self._parse_tp(tp)
        ra = numpy.atleast_1d(numpy.asarray(getattr(self.ra(tp), "value", self.ra(tp))))
        dec = numpy.atleast_1d(
            numpy.asarray(getattr(self.dec(tp), "value", self.dec(tp)))
        )
        p12 = coords.radec_to_custom(ra, dec, T=self._custom_transform, degree=True)
        out = p12[:, 0]
        return self._scale(self._maybe_scalar(tp, out), "degree")

    def phi2(self, tp):
        """Custom-frame latitude."""
        self._require_custom()
        tp = self._parse_tp(tp)
        ra = numpy.atleast_1d(numpy.asarray(getattr(self.ra(tp), "value", self.ra(tp))))
        dec = numpy.atleast_1d(
            numpy.asarray(getattr(self.dec(tp), "value", self.dec(tp)))
        )
        p12 = coords.radec_to_custom(ra, dec, T=self._custom_transform, degree=True)
        out = p12[:, 1]
        return self._scale(self._maybe_scalar(tp, out), "degree")

    def pmphi1(self, tp):
        """Proper motion in custom-frame phi1, multiplied by cos(phi2)."""
        self._require_custom()
        tp = self._parse_tp(tp)
        # Need pmra*cos(dec) and pmdec at tp, with (ra, dec)
        ra = numpy.atleast_1d(numpy.asarray(getattr(self.ra(tp), "value", self.ra(tp))))
        dec = numpy.atleast_1d(
            numpy.asarray(getattr(self.dec(tp), "value", self.dec(tp)))
        )
        pmra = numpy.atleast_1d(
            numpy.asarray(getattr(self.pmra(tp), "value", self.pmra(tp)))
        )
        pmdec = numpy.atleast_1d(
            numpy.asarray(getattr(self.pmdec(tp), "value", self.pmdec(tp)))
        )
        pm12 = coords.pmrapmdec_to_custom(
            pmra, pmdec, ra, dec, T=self._custom_transform, degree=True
        )
        return self._scale(self._maybe_scalar(tp, pm12[:, 0]), "pm")

    def pmphi2(self, tp):
        """Proper motion in custom-frame phi2."""
        self._require_custom()
        tp = self._parse_tp(tp)
        ra = numpy.atleast_1d(numpy.asarray(getattr(self.ra(tp), "value", self.ra(tp))))
        dec = numpy.atleast_1d(
            numpy.asarray(getattr(self.dec(tp), "value", self.dec(tp)))
        )
        pmra = numpy.atleast_1d(
            numpy.asarray(getattr(self.pmra(tp), "value", self.pmra(tp)))
        )
        pmdec = numpy.atleast_1d(
            numpy.asarray(getattr(self.pmdec(tp), "value", self.pmdec(tp)))
        )
        pm12 = coords.pmrapmdec_to_custom(
            pmra, pmdec, ra, dec, T=self._custom_transform, degree=True
        )
        return self._scale(self._maybe_scalar(tp, pm12[:, 1]), "pm")

    # -----------------------------------------------------------------
    # Covariance
    # -----------------------------------------------------------------
    _BASIS_COORDS = {
        "galcenrect": ("x", "y", "z", "vx", "vy", "vz"),
        "galcencyl": ("R", "vR", "vT", "z", "vz", "phi"),
        "sky": ("ra", "dec", "dist", "pmra", "pmdec", "vlos"),
        "galsky": ("ll", "bb", "dist", "pmll", "pmbb", "vlos"),
        "customsky": ("phi1", "phi2", "dist", "pmphi1", "pmphi2", "vlos"),
    }

    def cov(self, tp, basis="galcenrect"):
        """Return the 6x6 covariance matrix of the particle distribution at tp.

        Parameters
        ----------
        tp : float or array
            Progenitor time coordinate(s).
        basis : str, optional
            Output basis for the covariance matrix. One of:
            - ``"galcenrect"`` (default): ``(x, y, z, vx, vy, vz)`` in
              galactocentric Cartesian. Stored natively by the fit.
            - ``"galcencyl"``: ``(R, vR, vT, z, vz, phi)`` galactocentric
              cylindrical.
            - ``"galsky"``: ``(ll, bb, dist, pmll, pmbb, vlos)`` heliocentric
              Galactic sky + distance + kinematics.
            - ``"sky"``: ``(ra, dec, dist, pmra, pmdec, vlos)`` heliocentric
              equatorial sky + distance + kinematics.
            - ``"customsky"``: ``(phi1, phi2, dist, pmphi1, pmphi2, vlos)``
              sky rotated by ``custom_transform``. Requires
              ``custom_transform`` to have been set.

        Non-galcenrect bases are computed by analytical Jacobian of the
        coord chain evaluated at the track mean (``C' = J · C · Jᵀ``) — no
        finite differences. Pure rotations (e.g. the tangent rotations
        between sky frames) are inverted via transpose; ``lbd↔XYZ`` is a
        non-rotation transform so its 6x6 block is inverted by
        ``numpy.linalg.inv``.

        Units match the mean-track accessors (kpc, km/s, deg, mas/yr)
        when physical output is on; galpy internal units otherwise."""
        if self._cov_xyz is None:
            raise RuntimeError(
                "Covariance was not computed (order < 2). Rebuild with order=2."
            )
        if basis not in self._BASIS_COORDS:
            raise ValueError(
                f"Unknown basis {basis!r}; expected one of {list(self._BASIS_COORDS)}"
            )
        if basis == "customsky":
            self._require_custom()

        tp = self._parse_tp(tp)
        tp_arr = numpy.atleast_1d(tp)
        out = numpy.empty((len(tp_arr), 6, 6))
        for a in range(6):
            for b in range(6):
                out[:, a, b] = numpy.interp(
                    tp_arr, self._tp_grid, self._cov_xyz[:, a, b]
                )
        if self._physical:
            scale = numpy.array(
                [self._ro, self._ro, self._ro, self._vo, self._vo, self._vo]
            )
            out = out * numpy.outer(scale, scale)

        if basis != "galcenrect":
            for k, tp_k in enumerate(tp_arr):
                J = self._analytical_jacobian(tp_k, basis)
                out[k] = J @ out[k] @ J.T

        if numpy.isscalar(tp) or (hasattr(tp, "ndim") and tp.ndim == 0):
            return out[0]
        return out

    # -----------------------------------------------------------------
    # Analytical-Jacobian evaluator (used by cov(basis=..) and
    # plot(spread=) on non-galcenrect axes).
    # -----------------------------------------------------------------
    def _cart_mean_at(self, tp):
        """Mean (x, y, z, vx, vy, vz) at ``tp`` in the track's physical
        state (kpc, km/s when physical is on, else internal)."""
        return numpy.array(
            [
                float(self.x(tp)),
                float(self.y(tp)),
                float(self.z(tp)),
                float(self.vx(tp)),
                float(self.vy(tp)),
                float(self.vz(tp)),
            ]
        )

    def _analytical_jacobian(self, tp, basis):
        """6x6 analytical Jacobian d(basis)/d(galcenrect) at the track mean.

        Chain:
            galcenrect → galcencyl     (closed form)
            galcenrect → helio_XYZ     (galcenrect_to_XYZ_jac, a rotation)
            helio_XYZ  → galsky        (inv of lbd_to_XYZ_jac via linalg.inv)
            galsky     → sky           (tangent rotation at the sky point)
            sky        → customsky     (tangent rotation via custom_transform)
        """
        mean = self._cart_mean_at(tp)
        x, y, z, vx, vy, vz = mean

        if basis == "galcencyl":
            return _jac_galcenrect_to_galcencyl(x, y, z, vx, vy, vz)

        # Sky bases — chain through heliocentric Galactic Cartesian.
        ro = float(self._ro) if self._ro is not None else 1.0
        zo = float(self._zo) if self._zo is not None else 0.0
        # (1) d(helio_XYZ)/d(galcenrect): 6x6 rotation (block-diagonal).
        J1 = coords.galcenrect_to_XYZ_jac(x, y, z, vx, vy, vz, Xsun=ro, Zsun=zo)
        # Evaluate helio (XYZ, v) at the mean to pass into lbd_to_XYZ_jac.
        XYZ_mean = coords.galcenrect_to_XYZ(x, y, z, Xsun=ro, Zsun=zo)
        vxyz_mean = coords.galcenrect_to_vxvyvz(vx, vy, vz, Xsun=ro, Zsun=zo)
        X, Y, Z = float(XYZ_mean[0]), float(XYZ_mean[1]), float(XYZ_mean[2])
        vX, vY, vZ = float(vxyz_mean[0]), float(vxyz_mean[1]), float(vxyz_mean[2])
        lbd_mean = coords.XYZ_to_lbd(X, Y, Z, degree=False)
        l = float(lbd_mean[0])
        b = float(lbd_mean[1])
        d = float(lbd_mean[2])
        vrpm = coords.vxvyvz_to_vrpmllpmbb(vX, vY, vZ, X, Y, Z, XYZ=True, degree=False)
        vlos = float(vrpm[0])
        pmll = float(vrpm[1])
        pmbb = float(vrpm[2])
        # (2) d(l, b, d, vlos, pmll, pmbb)/d(helio_XYZ) = inv(lbd_to_XYZ_jac).
        #     lbd_to_XYZ_jac has order (l, b, d, vlos, pmll, pmbb); permute
        #     to our galsky order (l, b, d, pmll, pmbb, vlos) afterward.
        #     Internally work in radians for clean trig in (3); convert the
        #     angular output rows to degrees at the end.
        J_lbd = coords.lbd_to_XYZ_jac(l, b, d, vlos, pmll, pmbb, degree=False)
        J_lbd_inv = numpy.linalg.inv(J_lbd)
        perm = numpy.array([0, 1, 2, 4, 5, 3])  # (l,b,d,vlos,pmll,pmbb)→galsky
        P = numpy.eye(6)[perm]
        J_galsky = P @ J_lbd_inv @ J1  # l, b in radians
        if basis == "galsky":
            # Angular rows 0, 1 (l, b) → degrees
            J_galsky = J_galsky.copy()
            J_galsky[0] *= 180.0 / numpy.pi
            J_galsky[1] *= 180.0 / numpy.pi
            return J_galsky

        # (3) galsky → sky: local tangent rotation at (l, b) by angle α.
        #     Formulas for cosα, sinα match ``pmllpmbb_to_pmrapmdec``.
        #     The PM rotation has an extra term since α depends on sky
        #     position — captured by dα/dra and dα/ddec at the mean.
        theta, dec_ngp, ra_ngp = coords.get_epoch_angles(2000.0)
        cosb = numpy.cos(b)
        radec = coords.lb_to_radec(
            numpy.atleast_1d(l), numpy.atleast_1d(b), degree=False
        )
        ra = float(radec[0, 0] if radec.ndim == 2 else radec[0])
        dec_val = float(radec[0, 1] if radec.ndim == 2 else radec[1])
        sindec = numpy.sin(dec_val)
        cosdec = numpy.cos(dec_val)
        sindec_ngp = numpy.sin(dec_ngp)
        cosdec_ngp = numpy.cos(dec_ngp)
        sin_rrngp = numpy.sin(ra - ra_ngp)
        cos_rrngp = numpy.cos(ra - ra_ngp)
        # Un-normalized (σ, κ) then normalize
        κ = sindec_ngp * cosdec - cosdec_ngp * sindec * cos_rrngp
        σ = sin_rrngp * cosdec_ngp
        nrm2 = κ * κ + σ * σ
        nrm = numpy.sqrt(nrm2)
        cosa = κ / nrm
        sina = σ / nrm
        # dα/d(ra, dec) via α = atan2(σ, κ) ⇒ dα = (κ·dσ − σ·dκ)/nrm² with
        # unnormalized σ, κ.
        dsig_dra = cosdec_ngp * cos_rrngp
        dsig_ddec = 0.0
        dkap_dra = cosdec_ngp * sindec * sin_rrngp
        dkap_ddec = -sindec_ngp * sindec - cosdec_ngp * cosdec * cos_rrngp
        dalpha_dra = (κ * dsig_dra - σ * dkap_dra) / nrm2
        dalpha_ddec = (κ * dsig_ddec - σ * dkap_ddec) / nrm2
        # Full (ra, dec, pmra, pmdec) Jacobian wrt (l, b, pmll, pmbb):
        # position: rotation + cos-ratio scaling; PM: rotation + α-depends
        # -on-position correction. Chain through (l, b) → (ra, dec) which
        # itself is the rotation scaled.
        #   d(ra)/d(l) = cosα · cos(b) / cos(dec)
        #   d(ra)/d(b) = -sinα / cos(dec)
        #   d(dec)/d(l) = sinα · cos(b),   d(dec)/d(b) = cosα
        dra_dl = cosa * cosb / cosdec
        dra_db = -sina / cosdec
        ddec_dl = sina * cosb
        ddec_db = cosa
        # pmra = cosα · pmll − sinα · pmbb, similarly pmdec. If we perturb
        # (l, b) while keeping (pmll, pmbb) fixed, α changes via chain:
        #   dα/dl = dα/dra · dra/dl + dα/ddec · ddec/dl
        #   d(pmra)/d(l) = -pmdec · dα/dl,  d(pmdec)/d(l) = +pmra · dα/dl.
        pmra = cosa * pmll - sina * pmbb
        pmdec = sina * pmll + cosa * pmbb
        dalpha_dl = dalpha_dra * dra_dl + dalpha_ddec * ddec_dl
        dalpha_db = dalpha_dra * dra_db + dalpha_ddec * ddec_db
        J_galsky_to_sky = numpy.zeros((6, 6))
        J_galsky_to_sky[0, 0] = dra_dl
        J_galsky_to_sky[0, 1] = dra_db
        J_galsky_to_sky[1, 0] = ddec_dl
        J_galsky_to_sky[1, 1] = ddec_db
        J_galsky_to_sky[2, 2] = 1.0
        J_galsky_to_sky[3, 0] = -pmdec * dalpha_dl
        J_galsky_to_sky[3, 1] = -pmdec * dalpha_db
        J_galsky_to_sky[3, 3] = cosa
        J_galsky_to_sky[3, 4] = -sina
        J_galsky_to_sky[4, 0] = pmra * dalpha_dl
        J_galsky_to_sky[4, 1] = pmra * dalpha_db
        J_galsky_to_sky[4, 3] = sina
        J_galsky_to_sky[4, 4] = cosa
        J_galsky_to_sky[5, 5] = 1.0
        J_sky = J_galsky_to_sky @ J_galsky  # ra, dec in radians
        if basis == "sky":
            J_sky = J_sky.copy()
            J_sky[0] *= 180.0 / numpy.pi
            J_sky[1] *= 180.0 / numpy.pi
            return J_sky

        # (4) sky → customsky: identical form with α' from custom_transform
        #     (its pole in (ra, dec)). Applied to the sky-frame J_sky
        #     which is still in radians for (ra, dec); convert at the end.
        T = self._custom_transform
        ra_pole, dec_pole = coords.custom_to_radec(0.0, numpy.pi / 2.0, T=T)
        sinr_rp = numpy.sin(ra - ra_pole)
        cosr_rp = numpy.cos(ra - ra_pole)
        sindec_p = numpy.sin(dec_pole)
        cosdec_p = numpy.cos(dec_pole)
        κ2 = sindec_p * cosdec - cosdec_p * sindec * cosr_rp
        σ2 = sinr_rp * cosdec_p
        nrm2_2 = κ2 * κ2 + σ2 * σ2
        nrm2 = numpy.sqrt(nrm2_2)
        cosa2 = κ2 / nrm2
        sina2 = σ2 / nrm2
        # (phi1, phi2) evaluated at the mean:
        p12 = coords.radec_to_custom(
            numpy.atleast_1d(ra),
            numpy.atleast_1d(dec_val),
            T=T,
            degree=False,
        )
        cosphi2 = numpy.cos(float(p12[0, 1]))
        dsig2_dra = cosdec_p * cosr_rp
        dkap2_dra = cosdec_p * sindec * sinr_rp
        dkap2_ddec = -sindec_p * sindec - cosdec_p * cosdec * cosr_rp
        dalpha2_dra = (κ2 * dsig2_dra - σ2 * dkap2_dra) / nrm2_2
        dalpha2_ddec = (κ2 * 0.0 - σ2 * dkap2_ddec) / nrm2_2
        # chain dα/d(ra, dec) via d(ra, dec)/d(sky) (ra in the "sky" basis is
        # trivially just ra; so dra_dra=1, etc.)
        # pmrapmdec_to_custom uses the transposed sign convention
        # relative to pmllpmbb_to_pmrapmdec:
        #   pmphi1 =  cos(α')·pmra + sin(α')·pmdec
        #   pmphi2 = -sin(α')·pmra + cos(α')·pmdec
        pmphi1 = cosa2 * pmra + sina2 * pmdec
        pmphi2 = -sina2 * pmra + cosa2 * pmdec
        # Tangent-plane Jacobian. Position block uses the same 2x2
        # rotation with the standard cos-ratio scaling; PM block uses the
        # rotation itself; PM-vs-position has an α'-depends-on-position
        # correction:  ∂pmphi1/∂sky_pos = (-sin α'·pmra + cos α'·pmdec)·dα'
        #                                = +pmphi2 · dα'
        #              ∂pmphi2/∂sky_pos = -(cos α'·pmra + sin α'·pmdec)·dα'
        #                                = -pmphi1 · dα'
        dphi1_dra = cosa2 * cosdec / cosphi2
        dphi1_ddec = sina2 / cosphi2  # + sign with this sign convention
        dphi2_dra = -sina2 * cosdec  # - sign
        dphi2_ddec = cosa2
        J_sky_to_cs = numpy.zeros((6, 6))
        J_sky_to_cs[0, 0] = dphi1_dra
        J_sky_to_cs[0, 1] = dphi1_ddec
        J_sky_to_cs[1, 0] = dphi2_dra
        J_sky_to_cs[1, 1] = dphi2_ddec
        J_sky_to_cs[2, 2] = 1.0
        J_sky_to_cs[3, 0] = pmphi2 * dalpha2_dra
        J_sky_to_cs[3, 1] = pmphi2 * dalpha2_ddec
        J_sky_to_cs[3, 3] = cosa2
        J_sky_to_cs[3, 4] = sina2
        J_sky_to_cs[4, 0] = -pmphi1 * dalpha2_dra
        J_sky_to_cs[4, 1] = -pmphi1 * dalpha2_ddec
        J_sky_to_cs[4, 3] = -sina2
        J_sky_to_cs[4, 4] = cosa2
        J_sky_to_cs[5, 5] = 1.0
        # NOTE: J_sky here had rows 0, 1 converted to degrees if basis=='sky'
        # but we fell through (because basis=='customsky'), so J_sky rows
        # 0, 1 are still in radians. The J_sky_to_cs uses radian-derivatives.
        J_cs = J_sky_to_cs @ J_sky
        J_cs = J_cs.copy()
        J_cs[0] *= 180.0 / numpy.pi
        J_cs[1] *= 180.0 / numpy.pi
        return J_cs

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
    # Coord → (basis, index-within-basis) dispatch for plot(spread=).
    _COORD_BASIS = {
        "x": ("galcenrect", 0),
        "y": ("galcenrect", 1),
        "z": ("galcenrect", 2),
        "vx": ("galcenrect", 3),
        "vy": ("galcenrect", 4),
        "vz": ("galcenrect", 5),
        "R": ("galcencyl", 0),
        "vR": ("galcencyl", 1),
        "vT": ("galcencyl", 2),
        # z, vz, phi share cylindrical positions 3, 4, 5
        "phi": ("galcencyl", 5),
        "ra": ("sky", 0),
        "dec": ("sky", 1),
        "dist": ("sky", 2),
        "pmra": ("sky", 3),
        "pmdec": ("sky", 4),
        "vlos": ("sky", 5),
        "ll": ("galsky", 0),
        "bb": ("galsky", 1),
        "pmll": ("galsky", 3),
        "pmbb": ("galsky", 4),
        "phi1": ("customsky", 0),
        "phi2": ("customsky", 1),
        "pmphi1": ("customsky", 3),
        "pmphi2": ("customsky", 4),
    }

    def plot(self, d1="x", d2="y", spread=0, n=None, **kwargs):
        """Plot the smooth track in the (d1, d2) plane.

        Parameters
        ----------
        d1, d2 : str
            Coordinate names. Any of: x, y, z, vx, vy, vz, R, vR, vT, phi,
            ra, dec, dist, ll, bb, pmra, pmdec, pmll, pmbb, vlos, plus
            phi1, phi2, pmphi1, pmphi2 if ``custom_transform`` is set.
        spread : float, optional
            If > 0, draw a ±spread·sigma band around the track along
            ``d2`` using the projected covariance. Works for any ``d2``
            that has a basis in :attr:`_COORD_BASIS` (i.e. any
            galactocentric Cartesian / cylindrical axis, any sky
            coordinate, or any custom-frame coordinate when a transform
            is set). Silently skipped for axes outside the dispatch
            (e.g. ``ll``, ``bb``, ``pmll``, ``pmbb``).
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
        if spread > 0 and self._cov_xyz is not None and d2 in self._COORD_BASIS:
            basis, idx = self._COORD_BASIS[d2]
            try:
                cov = self.cov(tp, basis=basis)  # (n_eval, 6, 6)
                s2 = numpy.sqrt(numpy.maximum(cov[:, idx, idx], 0.0))
            except RuntimeError:
                # e.g. basis="custom" without custom_transform — skip band
                s2 = None
            if s2 is not None:
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
