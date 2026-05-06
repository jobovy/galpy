import warnings

import numpy
from matplotlib import pyplot
from scipy import interpolate
from scipy.spatial import cKDTree

from ..util import config, conversion, coords, galpyWarning
from ..util._optional_deps import _APY_LOADED, _APY_UNITS
from ..util.conversion import physical_conversion

if _APY_LOADED:
    from astropy import units


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


def _smooth_series(x, y, sigma, s_user=None, smoothing_factor=1.0):
    """Fit a smoothing spline y(x) with weights derived from sigma.

    Default (``s_user=None``): use ``make_smoothing_spline`` with GCV
    auto-tuning (requires scipy >= 1.10). When ``s_user`` is a float, use
    ``UnivariateSpline`` with that explicit ``s`` value.

    ``smoothing_factor`` (default 1.0) multiplies the effective ``s``
    after GCV / user choice — values > 1 produce smoother fits,
    values < 1 rougher. Useful when GCV undersmooths in finite samples
    (a common failure mode of ``make_smoothing_spline`` on noisy binned
    means with mis-estimated per-bin sigmas).

    Returns ``(spline, effective_s)`` where ``effective_s`` is the
    weighted residual sum the returned spline actually achieves —
    the ``s`` that a ``UnivariateSpline`` would need to reproduce
    the same fit. Passing this back as ``s_user`` (with
    ``smoothing_factor=1.0``) reproduces the fit without re-running
    GCV.

    Falls back to linear interpolation (effective_s=0) for fewer than 5
    valid bins.
    """
    mask = numpy.isfinite(y) & numpy.isfinite(x)
    n_valid = int(mask.sum())
    order = numpy.argsort(x[mask])
    xv = x[mask][order]
    yv = y[mask][order]
    if n_valid < 5:
        if n_valid < 2:
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
    if not numpy.isfinite(sig_med) or sig_med == 0:
        sig_med = 1.0
    sv = numpy.where(numpy.isfinite(sig_safe), sig_safe, sig_med)[mask][order]
    sv = numpy.maximum(sv, 1e-12)
    if s_user is None:
        # Normalize y to O(1) before GCV. ``make_smoothing_spline``'s GCV
        # silently collapses to lambda=0 (interpolation) when ``y`` is at
        # very small absolute scale (~1e-4 or smaller), even with the
        # signal-to-noise ratio held fixed. Galpy's covariance series in
        # internal units sit at 1e-5 to 1e-6 and were hitting this regime.
        yscale = float(numpy.nanstd(yv))
        if not numpy.isfinite(yscale) or yscale == 0:
            yscale = 1.0
        spl_n = interpolate.make_smoothing_spline(
            xv, yv / yscale, w=1.0 / ((sv / yscale) ** 2)
        )
        spl = interpolate.BSpline(spl_n.t, spl_n.c * yscale, spl_n.k)
        resid = yv - spl(xv)
        s_gcv = float(numpy.sum((resid / sv) ** 2))
        if smoothing_factor == 1.0:
            return spl, s_gcv
        # Re-fit at s = s_gcv * smoothing_factor; recompute the achieved
        # SSR from the new spline so the returned ``effective_s`` round-
        # trips through ``s_user`` with smoothing_factor=1 unchanged.
        s_target = s_gcv * float(smoothing_factor)
        spl = interpolate.UnivariateSpline(xv, yv, w=1.0 / sv, s=s_target, k=3)
        resid = yv - spl(xv)
        eff_s = float(numpy.sum((resid / sv) ** 2))
        return spl, eff_s
    s_val = float(s_user) * float(smoothing_factor)
    return interpolate.UnivariateSpline(xv, yv, w=1.0 / sv, s=s_val, k=3), s_val


def _closest_point_on_curve(points, curve, curve_t, mask=None, velocity_weight=1.0):
    """For each point in ``points`` (N, D), find the index of the closest
    entry in ``curve`` (M, D) and return the corresponding time
    ``curve_t[idx]``. Works for any dimensionality D.

    Uses ``scipy.spatial.cKDTree`` for the unmasked case (sublinear
    per query). When a per-particle ``mask`` of shape (N, M) is given,
    queries K nearest neighbors and picks the closest allowed one,
    growing K until every point has a match.

    ``velocity_weight`` (default 1.0): when D == 6, scale the velocity
    components (last 3 columns) by this factor before computing distances.
    See :func:`_fit_track_from_particles` for usage and ``'auto'`` mode.
    """
    if velocity_weight != 1.0 and curve.shape[1] == 6:
        sc = numpy.array(
            [1.0, 1.0, 1.0, velocity_weight, velocity_weight, velocity_weight]
        )
        curve = curve * sc
        points = points * sc
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
        # Vectorized: look up the per-particle mask at each candidate index,
        # then take the first allowed candidate. cKDTree returns neighbours
        # in ascending-distance order, so argmax over the bool row gives the
        # index of the closest allowed candidate (or 0 if none — filtered by
        # the any() check below).
        allowed = mask[remaining[:, None], cand]
        hit = allowed.any(axis=1)
        first = allowed.argmax(axis=1)
        chosen = cand[numpy.arange(remaining.size), first]
        tp_out[remaining[hit]] = curve_t[chosen[hit]]
        remaining = remaining[~hit]
        if k >= M:
            tp_out[remaining] = 0.0
            break
        k = min(k * 4, M)
    return tp_out


def _fit_one_pass(
    particles_cart,
    tp_assign,
    tp_nodes,
    tp_grid,
    prog_at_fn,
    order,
    s_user_mean,
    s_user_cov,
    smoothing_factor=1.0,
):
    """One pass of the offset-from-progenitor smoothing fit.

    Bins particle offsets by ``tp_assign`` onto ``tp_nodes``, fits a
    smoothing spline through each binned coordinate (and through each
    upper-triangle covariance entry when ``order >= 2``), and reconstructs
    the dense track on ``tp_grid`` by adding the smoothed offsets back to
    the progenitor curve.

    ``smoothing_factor`` (default 1.0) scales the effective ``s`` of every
    spline (mean and covariance) by the same factor; values > 1 force a
    smoother fit when GCV undersmooths.

    Returns ``(track_xyz, track_vxvyvz, cov_xyz_or_None, smoothing_s)``.
    """
    prog_at_particles = prog_at_fn(tp_assign)
    offsets = particles_cart - prog_at_particles

    means, covs, counts = _bin_by_tp(tp_assign, offsets, tp_nodes)
    with numpy.errstate(invalid="ignore"):
        per_coord_sigma = numpy.sqrt(numpy.diagonal(covs, axis1=1, axis2=2))
        sigma_of_mean = per_coord_sigma / numpy.sqrt(numpy.maximum(counts[:, None], 1))
        sigma_of_mean = numpy.where(counts[:, None] > 1, sigma_of_mean, numpy.nan)

    eff_s_mean = []
    offset_splines = []
    for i in range(6):
        spl, es = _smooth_series(
            tp_nodes,
            means[:, i],
            sigma_of_mean[:, i],
            s_user=s_user_mean[i],
            smoothing_factor=smoothing_factor,
        )
        offset_splines.append(spl)
        eff_s_mean.append(es)
    offset_fine = numpy.column_stack([spl(tp_grid) for spl in offset_splines])

    ninterp = len(tp_grid)
    prog_on_grid = prog_at_fn(tp_grid)
    track_fine = prog_on_grid + offset_fine

    eff_s_cov = []
    cov_xyz = None
    if order >= 2:
        cov_fine = numpy.zeros((ninterp, 6, 6))
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
                    tp_nodes,
                    vals,
                    sigma_c,
                    s_user=s_user_cov[cov_idx],
                    smoothing_factor=smoothing_factor,
                )
                eff_s_cov.append(es)
                cov_idx += 1
                val_fine = spl(tp_grid)
                cov_fine[:, a, b] = val_fine
                cov_fine[:, b, a] = val_fine
        cov_fine = numpy.nan_to_num(cov_fine, nan=0.0, posinf=0.0, neginf=0.0)
        for k in range(ninterp):
            evals, evecs = numpy.linalg.eigh(cov_fine[k])
            evals = numpy.clip(evals, 0.0, None)
            cov_fine[k] = (evecs * evals) @ evecs.T
        cov_xyz = cov_fine

    return track_fine[:, 0:3], track_fine[:, 3:6], cov_xyz, eff_s_mean + eff_s_cov


def _fit_track_from_particles(
    xv_particles,
    dt_particles,
    track_prog_cart,
    track_t_grid,
    arm_sign=1,
    ntp=None,
    ninterp=1001,
    smoothing=None,
    smoothing_factor=1.0,
    niter=0,
    order=2,
    velocity_weight="auto",
    prog_orbit=None,
):
    """Run the full closest-point projection + offset smoothing fit.

    Returns the precomputed-track state needed by :class:`StreamTrack`:
    a dict with keys ``tp_grid``, ``track_xyz``, ``track_vxvyvz``,
    ``cov_xyz`` (None for ``order=1``), ``smoothing_s``, and
    ``particles`` (the raw ``(xv, dt)`` tuple). ``smoothing_factor`` (>0)
    scales every spline's effective ``s`` after GCV / explicit-s choice.

    ``velocity_weight`` (float or ``'auto'``, default ``'auto'``): scale
    velocity components in the 6D distance during closest-point search.
    Values > 1 make velocity matches more important than position
    matches — useful when the progenitor orbit revisits regions of
    phase space (e.g., near apocenter under strong perturbations). When
    ``'auto'`` (the default), runs a probe pass with weight 1, computes
    σ_pos and σ_vel from the inner half of particles (where the
    assignment is unambiguous), and uses ``σ_pos / σ_vel`` (clipped to
    ``[0.1, 10]``) as the weight for the actual fit. The auto value
    typically lands at ~2–3 for both clean and perturbed streams. Pass
    ``1.0`` for the legacy unweighted natural-units metric.
    """
    track_t_grid = numpy.asarray(track_t_grid, dtype=float)
    prog_cart = numpy.asarray(track_prog_cart, dtype=float)
    arm_sign = int(numpy.sign(arm_sign)) or 1
    ninterp = int(ninterp)
    order = int(order)

    if prog_orbit is not None:
        # Reuse the Orbit's internal interpolation directly — the orbit
        # has already been integrated densely on track_t_grid.
        def _prog_at(tp):
            tp = numpy.atleast_1d(tp)
            return numpy.column_stack(
                [
                    prog_orbit.x(tp),
                    prog_orbit.y(tp),
                    prog_orbit.z(tp),
                    prog_orbit.vx(tp),
                    prog_orbit.vy(tp),
                    prog_orbit.vz(tp),
                ]
            )
    else:
        prog_splines = [
            interpolate.InterpolatedUnivariateSpline(track_t_grid, prog_cart[:, i], k=3)
            for i in range(6)
        ]

        def _prog_at(tp):
            tp = numpy.atleast_1d(tp)
            return numpy.column_stack([spl(tp) for spl in prog_splines])

    # Raw (xv, dt) snapshot the user can plot the track over without
    # resampling — the converse of the ``particles=`` knob on
    # ``basestreamspraydf.streamTrack``.
    particles = (
        numpy.asarray(xv_particles, dtype=float).copy(),
        numpy.asarray(dt_particles, dtype=float).copy(),
    )
    particles_cart = coords.galcencyl_to_galcenrect(
        *numpy.asarray(xv_particles, dtype=float)
    )
    dt = numpy.asarray(dt_particles, dtype=float)

    # Normalize smoothing argument into per-spline s values.
    # Length 6 = mean only; length 27 = mean (6) + covariance (21).
    if smoothing is None:
        s_user_mean = [None] * 6
        s_user_cov = [None] * 21
    elif hasattr(smoothing, "__len__") and not isinstance(smoothing, str):
        s_arr = list(smoothing)
        s_user_mean = [None if v is None else float(v) for v in s_arr[:6]]
        s_user_cov = [
            None if v is None else float(v)
            for v in (s_arr[6:27] if len(s_arr) > 6 else [None] * 21)
        ]
    else:
        s_user_mean = [float(smoothing)] * 6
        s_user_cov = [None] * 21

    # Initial assignment: 6D closest point on the progenitor orbit, windowed
    # by arm sign and stripping time.
    sign_mask = (track_t_grid * arm_sign) >= 0
    dt_mask = numpy.abs(track_t_grid)[None, :] <= dt[:, None]
    mask = sign_mask[None, :] & dt_mask

    # Resolve velocity_weight='auto' from a probe pass. Use the inner-half
    # of particles (smaller |tp_assign|) where the assignment is unambiguous,
    # and set weight = σ_pos / σ_vel (clipped to [0.1, 10]). This makes the
    # 6D metric scale-invariant w.r.t. arbitrary ro/vo choices.
    if isinstance(velocity_weight, str):
        if velocity_weight != "auto":
            raise ValueError(
                f"velocity_weight= must be a float or 'auto', got {velocity_weight!r}"
            )
        probe_tp = _closest_point_on_curve(
            particles_cart, prog_cart, track_t_grid, mask=mask
        )
        abs_probe = numpy.abs(probe_tp)
        # With size >= 20 the inner-half (median split) always has at least
        # 10 entries, so we don't need a separate inner_n < 10 fallback.
        if abs_probe.size >= 20 and abs_probe.max() > 0:
            inner = abs_probe <= numpy.median(abs_probe)
            # Closest grid index for each inner particle's tp
            idx_inner = numpy.argmin(
                numpy.abs(track_t_grid[None, :] - probe_tp[inner, None]), axis=1
            )
            prog_at_inner = prog_cart[idx_inner]
            dpos = particles_cart[inner, :3] - prog_at_inner[:, :3]
            dvel = particles_cart[inner, 3:] - prog_at_inner[:, 3:]
            sigma_pos = numpy.sqrt(numpy.mean(numpy.sum(dpos**2, axis=1)))
            sigma_vel = numpy.sqrt(numpy.mean(numpy.sum(dvel**2, axis=1)))
            if sigma_vel > 0:
                # Clip to [0.1, 10] — the upper bound caps pathological
                # high ratios; the lower bound just guards against an
                # ill-conditioned σ_vel estimate. Streams where σ_pos/σ_vel
                # is genuinely below 1 (positions tighter than velocities
                # in natural units) get the value as-is.
                velocity_weight = float(numpy.clip(sigma_pos / sigma_vel, 0.1, 10.0))
            else:
                velocity_weight = 1.0
        else:
            velocity_weight = 1.0

    tp_assign = _closest_point_on_curve(
        particles_cart,
        prog_cart,
        track_t_grid,
        mask=mask,
        velocity_weight=velocity_weight,
    )

    # Drop particles whose closest point hit the far boundary of the searched
    # progenitor arc — those are mismatches that would bias the track tips.
    if arm_sign > 0:
        far_edge = track_t_grid[-1]
    else:
        far_edge = track_t_grid[0]
    arc_span = abs(track_t_grid[-1] - track_t_grid[0])
    interior = numpy.abs(tp_assign - far_edge) > 1e-3 * arc_span
    tp_assign = tp_assign[interior]
    particles_cart = particles_cart[interior]

    # Sanity check: detect a histogram gap in tp_assign — a hallmark of
    # closest-point assignments that landed on a far-away revisit of the
    # progenitor orbit (e.g., particles projected past an apocentric kink in
    # a perturbed potential). Histogram into ~30 bins between
    # min(|tp|) and max(|tp|); look for a contiguous run of empty bins
    # whose total span exceeds 25% of the populated range. If found, warn.
    abs_tp = numpy.abs(tp_assign)
    if abs_tp.size >= 30 and abs_tp.max() > 0:
        nb = 30
        edges = numpy.linspace(abs_tp.min(), abs_tp.max(), nb + 1)
        counts, _ = numpy.histogram(abs_tp, bins=edges)
        # Find the longest run of zero-count bins
        zero = counts == 0
        max_run = 0
        run = 0
        for z in zero:
            if z:
                run += 1
                if run > max_run:
                    max_run = run
            else:
                run = 0
        gap_span = max_run * (edges[1] - edges[0])
        populated_span = abs_tp.max() - abs_tp.min()
        n_far = int((abs_tp > abs_tp.max() - gap_span).sum())
        # Require BOTH a substantial gap AND substantial mass past it.
        # 0.35% would be normal stragglers; 1% indicates a structural issue.
        if (
            populated_span > 0
            and gap_span / populated_span > 0.25
            and n_far / abs_tp.size > 0.01
        ):
            warnings.warn(
                f"streamTrack: tp_assign histogram has a "
                f"{gap_span / populated_span:.0%} gap with "
                f"{n_far / abs_tp.size:.1%} of particles past it; the track "
                "is likely to kink. Try a larger velocity_weight or niter.",
                galpyWarning,
            )

    # Trim the public tp grid to the percentile range where the binned data
    # supports a fit (outliers at extreme |tp| produce sparse boundary bins).
    trim_percentile = 99.0

    def _trim_grid(tp_assign_arr):
        if arm_sign > 0:
            tp_lo_ = 0.0
            tp_hi_ = float(numpy.percentile(tp_assign_arr, trim_percentile))
        else:
            tp_hi_ = 0.0
            tp_lo_ = float(numpy.percentile(tp_assign_arr, 100.0 - trim_percentile))
        if tp_hi_ - tp_lo_ < 1e-12:
            tp_lo_ = float(track_t_grid[0])
            tp_hi_ = float(track_t_grid[-1])
        return tp_lo_, tp_hi_

    tp_lo, tp_hi = _trim_grid(tp_assign)
    tp_grid = numpy.linspace(tp_lo, tp_hi, ninterp)

    # Binning nodes. Default: ~sqrt(N), clipped to [21, max_ntp].
    # The ceiling scales with the arc span so longer streams can afford
    # finer binning — one knot every ~0.1 internal time units (≈3.6 Myr
    # at the galpy reference scale), with a floor of 201.
    n_part = particles_cart.shape[0]
    if ntp is None:
        max_ntp = max(201, int(abs(tp_hi - tp_lo) / 0.1))
        ntp = int(max(21, min(max_ntp, round(numpy.sqrt(n_part)))))
    ntp_int = int(ntp)
    tp_nodes = numpy.linspace(tp_lo, tp_hi, ntp_int)

    track_xyz = None
    track_vxvyvz = None
    cov_xyz = None
    smoothing_s = None
    for it in range(niter + 1):
        track_xyz, track_vxvyvz, cov_xyz, smoothing_s = _fit_one_pass(
            particles_cart,
            tp_assign,
            tp_nodes,
            tp_grid,
            _prog_at,
            order,
            s_user_mean,
            s_user_cov,
            smoothing_factor=smoothing_factor,
        )
        if it < niter:
            track_cart = numpy.column_stack([track_xyz, track_vxvyvz])
            tp_assign = _closest_point_on_curve(
                particles_cart,
                track_cart,
                tp_grid,
                velocity_weight=velocity_weight,
            )
            # Re-trim tp_grid / tp_nodes from the new assignment. This lets
            # iteration usefully shrink the public range when outliers
            # collapse onto the bulk; without this, the smoothed track
            # extrapolates past the new particle support and can produce
            # phantom kinks worse than the unfit baseline.
            tp_lo, tp_hi = _trim_grid(tp_assign)
            tp_grid = numpy.linspace(tp_lo, tp_hi, ninterp)
            tp_nodes = numpy.linspace(tp_lo, tp_hi, ntp_int)

    return {
        "tp_grid": tp_grid,
        "track_xyz": track_xyz,
        "track_vxvyvz": track_vxvyvz,
        "cov_xyz": cov_xyz,
        "smoothing_s": smoothing_s,
        "particles": particles,
    }


class StreamTrack:
    """Smooth phase-space track for a single arm of a stream.

    A StreamTrack holds a smooth mean curve (and optional covariance) in
    galactocentric Cartesian phase space, parameterized by an affine curve
    parameter ``tp``. The class is a pure precomputed-track container —
    accessors evaluate the stored splines and ``cov(basis=...)`` transforms
    the stored covariance into other bases via analytical Jacobians. The
    fitting pipeline (closest-point + offset smoothing) is provided by the
    :meth:`from_particles` classmethod.

    The semantics of ``tp`` are set at construction via ``parameter_kind``:

    * ``"time"`` (default): astropy ``Quantity`` inputs to accessors are
      parsed via ``conversion.parse_time``. Used by the streamspraydf
      pipeline, where ``tp=0`` is the progenitor today, ``tp>0`` is future
      (leading arm), ``tp<0`` is past (trailing arm).
    * ``"angle"``: parsed via ``conversion.parse_angle``. The natural
      choice for streamdf-style tracks indexed by an angle along the
      stream.
    * ``None``: pass-through; useful for any other custom curve parameter.

    Accessors and :meth:`cov` return ``NaN`` for ``tp`` values outside the
    track's valid range (rather than silent cubic-spline extrapolation).
    For an array ``tp``, only the offending entries are NaN.

    Attributes
    ----------
    particles : tuple of (ndarray, ndarray) or absent
        ``(xv, dt)`` of the particles the track was fit to, in the same
        ``(R, vR, vT, z, vz, phi)`` layout that
        :meth:`from_particles` accepts and ``streamspraydf.streamTrack``
        returns from its ``particles=`` knob. Available only on tracks
        built via :meth:`from_particles` (the
        :meth:`streamspraydf.streamTrack
        <galpy.df.streamspraydf.streamTrack>` path); absent on tracks
        constructed directly from precomputed splines via
        :meth:`__init__`. Useful for re-fitting at different smoothing /
        iteration settings without re-sampling the spray DF.
    """

    def __init__(
        self,
        tp_grid,
        track_xyz,
        track_vxvyvz,
        cov_xyz=None,
        custom_transform=None,
        parameter_kind="time",
        ro=None,
        vo=None,
        zo=None,
        solarmotion=None,
    ):
        """Build a StreamTrack from a precomputed smooth track.

        Parameters
        ----------
        tp_grid : array, shape (N,)
            Affine curve parameter samples. Must be strictly monotonic.
        track_xyz : array, shape (N, 3)
            Galactocentric Cartesian positions ``(x, y, z)`` on ``tp_grid``,
            in galpy internal length units.
        track_vxvyvz : array, shape (N, 3)
            Galactocentric Cartesian velocities ``(vx, vy, vz)`` on
            ``tp_grid``, in galpy internal velocity units.
        cov_xyz : array, shape (N, 6, 6), optional
            Galactocentric Cartesian covariance ``(x, y, z, vx, vy, vz)``
            on ``tp_grid`` (internal units). Required for :meth:`cov` and
            for the ``spread`` band in :meth:`plot`. ``None`` (default) ⇒
            no covariance support.
        custom_transform : array, shape (3, 3), optional
            Rotation from equatorial to a custom ``(phi1, phi2)`` sky
            frame. Enables the ``phi1``/``phi2``/``pmphi1``/``pmphi2``
            accessors and ``cov(basis="customsky")``.
        parameter_kind : {"time", "angle", None}, optional
            How astropy ``Quantity`` inputs to accessors are interpreted
            (see class docstring). Default ``"time"``.
        ro : float or Quantity, optional
            Distance scale (kpc). When omitted, falls back to the
            ``normalization.ro`` config value and physical-units output is
            off by default (``_roSet=False``); when set explicitly,
            physical-units output is on by default. Mirrors ``Orbit``.
        vo : float or Quantity, optional
            Circular velocity scale (km/s). Same semantics as ``ro``.
        zo : float or Quantity, optional
            Sun's height above the midplane (kpc). Default None.
        solarmotion : str, numpy.ndarray or Quantity, optional
            ``'hogg'``, ``'dehnen'``, ``'schoenrich'``, or ``[-U, V, W]``
            in km/s. Default None.
        """
        self._tp_grid = numpy.asarray(tp_grid, dtype=float).copy()
        self._track_xyz = numpy.asarray(track_xyz, dtype=float).copy()
        self._track_vxvyvz = numpy.asarray(track_vxvyvz, dtype=float).copy()
        self._cov_xyz = (
            None if cov_xyz is None else numpy.asarray(cov_xyz, dtype=float).copy()
        )
        self._ninterp = len(self._tp_grid)
        self._custom_transform = (
            None
            if custom_transform is None
            else numpy.asarray(custom_transform, dtype=float)
        )
        if parameter_kind not in ("time", "angle", None):
            raise ValueError(
                f"parameter_kind must be 'time', 'angle', or None; "
                f"got {parameter_kind!r}"
            )
        self._parameter_kind = parameter_kind
        # Mirror Orbit's pattern (Orbits.py:600-611): roSet/voSet are
        # derived from whether ``ro``/``vo`` were explicitly passed; an
        # unset scale falls back to the config default with the matching
        # ``_*Set`` flag turned off.
        if ro is None:
            self._ro = config.__config__.getfloat("normalization", "ro")
            self._roSet = False
        else:
            self._ro = conversion.parse_length_kpc(ro)
            self._roSet = True
        if vo is None:
            self._vo = config.__config__.getfloat("normalization", "vo")
            self._voSet = False
        else:
            self._vo = conversion.parse_velocity_kms(vo)
            self._voSet = True
        self._zo = zo
        self._solarmotion = solarmotion

        # Cubic-spline interpolators on the 6D track (mean only — cov() is
        # interpolated linearly entry-by-entry for cheap PSD enforcement).
        track_fine = numpy.column_stack([self._track_xyz, self._track_vxvyvz])
        self._cart_splines = [
            interpolate.InterpolatedUnivariateSpline(
                self._tp_grid, track_fine[:, i], k=3
            )
            for i in range(6)
        ]

    # -----------------------------------------------------------------
    # Particle-fit constructor (the streamspraydf pipeline)
    # -----------------------------------------------------------------
    @classmethod
    def from_particles(
        cls,
        xv_particles,
        dt_particles,
        track_prog_cart,
        track_t_grid,
        arm_sign=1,
        ntp=None,
        ninterp=1001,
        smoothing=None,
        smoothing_factor=1.0,
        niter=0,
        order=2,
        velocity_weight="auto",
        prog_orbit=None,
        custom_transform=None,
        ro=None,
        vo=None,
        zo=None,
        solarmotion=None,
    ):
        """Build a StreamTrack by fitting a smooth curve to stream particles.

        Runs the closest-point projection of each particle onto a
        finely-sampled progenitor orbit, bins the resulting offsets by
        progenitor time, smooths each coordinate (and optionally each
        covariance entry) through GCV or user-specified ``s``, and
        reconstructs the dense track by adding the smoothed offsets back
        to the progenitor orbit. The ``tp`` parameter is interpreted as
        a galpy time, suitable for spraydf models.

        After the fit, ``inst.particles`` holds the raw ``(xv, dt)`` tuple
        the fit saw and ``inst.smoothing_s`` holds the effective per-spline
        ``s`` values (length 6 for ``order=1``, 27 for ``order>=2``) which
        can be passed back as ``smoothing`` in a subsequent call to
        reproduce the same smoothness without re-running GCV.

        Parameters
        ----------
        xv_particles : array, shape (6, N)
            Present-day phase space ``(R, vR, vT, z, vz, phi)`` of stream
            particles in galpy internal units.
        dt_particles : array, shape (N,)
            Stripping times (positive, galpy internal time) for each
            particle. Used as a windowing prior on the closest-point
            search.
        track_prog_cart : array, shape (M, 6)
            Finely-sampled progenitor phase space ``(x, y, z, vx, vy, vz)``
            at the times in ``track_t_grid``. Must cover both sides of
            ``tp=0``.
        track_t_grid : array, shape (M,)
            The dense time grid on which ``track_prog_cart`` is evaluated.
        arm_sign : int, optional
            ``+1`` for leading arm (``tp >= 0``), ``-1`` for trailing
            (``tp <= 0``).
        ntp : int, optional
            Number of binning nodes. Default ``sqrt(N)`` with a floor of
            21 and a ceiling that scales with the arc span (at least 201;
            larger for long streams).
        ninterp : int, optional
            Resolution of the fine ``tp`` grid. Default 1001.
        smoothing : None, float, or array-like, optional
            Smoothing parameter(s); see class docstring. ``None`` (default)
            uses GCV via ``make_smoothing_spline``.
        smoothing_factor : float, optional
            Multiplier applied to every spline's effective ``s`` after
            GCV (or explicit-``s``) selection. Values > 1 force a smoother
            fit, values < 1 a rougher one. Useful when GCV undersmooths
            in finite samples (a common failure mode of
            ``make_smoothing_spline`` on noisy binned means). Default 1.0.
            For an interactive smoothing sweep, save ``track.particles``
            from the first call and pass it back as ``particles=`` to
            subsequent ``streamspraydf.streamTrack`` calls — only the
            cheap re-fit step runs, the orbit-integration sample is
            reused.
        niter : int, optional
            Iterations beyond the initial fit. Each iteration reassigns
            particles to the closest point on the current track.
        order : int, optional
            ``1`` = mean only, ``2`` = mean + covariance.
        velocity_weight : float or ``'auto'``, optional
            Multiplicative weight applied to velocity components when
            computing 6D distances in the closest-point projection.
            Default ``'auto'`` learns the weight from the inner-half
            particle dispersion (``σ_pos / σ_vel``, clipped to
            ``[0.1, 10]``); see ``streamspraydf.streamTrack`` for
            motivation. Pass ``1.0`` for the legacy unweighted metric.
        custom_transform : array, shape (3, 3), optional
            Rotation from equatorial to a custom sky frame. Forwarded to
            the base ``__init__``.
        ro : float or Quantity, optional
            Distance scale (kpc). Default ``None`` — the resulting
            StreamTrack falls back onto the progenitor orbit's ``ro``
            value (and inherits its ``_roSet`` flag). Pass an explicit
            value only if the track should override the progenitor's.
        vo : float or Quantity, optional
            Velocity scale (km/s). Default ``None`` — same fallback to
            the progenitor as for ``ro``.
        zo : float or Quantity, optional
            Sun's height above the midplane (kpc). Default ``None`` —
            falls back to the progenitor orbit's ``zo``.
        solarmotion : str, numpy.ndarray or Quantity, optional
            ``'hogg'``, ``'dehnen'``, ``'schoenrich'``, or ``[-U, V, W]``
            in km/s. Default ``None`` — falls back to the progenitor's.
        """
        fit = _fit_track_from_particles(
            xv_particles,
            dt_particles,
            track_prog_cart,
            track_t_grid,
            arm_sign=arm_sign,
            ntp=ntp,
            ninterp=ninterp,
            smoothing=smoothing,
            smoothing_factor=smoothing_factor,
            niter=niter,
            order=order,
            velocity_weight=velocity_weight,
            prog_orbit=prog_orbit,
        )
        inst = cls(
            tp_grid=fit["tp_grid"],
            track_xyz=fit["track_xyz"],
            track_vxvyvz=fit["track_vxvyvz"],
            cov_xyz=fit["cov_xyz"],
            custom_transform=custom_transform,
            parameter_kind="time",
            ro=ro,
            vo=vo,
            zo=zo,
            solarmotion=solarmotion,
        )
        # Fitter outputs callers may want: the raw (xv, dt) sample the fit
        # saw, and the effective per-spline ``s`` values for reuse.
        inst.particles = fit["particles"]
        inst.smoothing_s = fit["smoothing_s"]
        return inst

    # -----------------------------------------------------------------
    # Public evaluation
    # -----------------------------------------------------------------
    def tp_grid(self):
        """Return the fine tp grid on which the track is stored."""
        return self._tp_grid.copy()

    def _in_range(self, tp_arr):
        """Boolean mask over ``tp_arr`` for entries inside the track's
        valid ``tp`` range. Out-of-range tps get NaN accessor / cov
        outputs rather than silent cubic-spline extrapolation."""
        return (tp_arr >= self._tp_grid[0]) & (tp_arr <= self._tp_grid[-1])

    def _eval_cart(self, tp):
        tp_arr = numpy.atleast_1d(tp)
        in_range = self._in_range(tp_arr)
        out = numpy.array([spl(tp_arr) for spl in self._cart_splines])  # (6, len)
        return numpy.where(in_range[None, :], out, numpy.nan)

    def _maybe_scalar(self, tp, arr):
        if numpy.isscalar(tp) or (hasattr(tp, "ndim") and tp.ndim == 0):
            return arr[0]
        return arr

    def _parse_tp(self, tp):
        if self._parameter_kind == "time":
            return conversion.parse_time(tp, ro=self._ro, vo=self._vo)
        if self._parameter_kind == "angle":
            return conversion.parse_angle(tp)
        # Pass-through for callers that use a custom non-physical parameter.
        return tp

    def _cart_eval(self, idx, tp):
        # Out-of-range tps return NaN (not silent cubic-spline extrapolation).
        # Array tps get NaNs only at the offending entries.
        tp_arr = numpy.atleast_1d(tp)
        in_range = self._in_range(tp_arr)
        val = numpy.where(in_range, self._cart_splines[idx](tp_arr), numpy.nan)
        return self._maybe_scalar(tp, val)

    @physical_conversion("position", pop=True)
    def x(self, tp, **kwargs):
        """Galactocentric Cartesian x along the track.

        Parameters
        ----------
        tp : float, array, or Quantity
            Curve parameter(s); Quantity inputs are parsed per
            ``parameter_kind``. Out-of-range entries return ``NaN``.
        ro : float or Quantity, optional
            Distance scale (kpc); overrides the stored value.
        vo : float or Quantity, optional
            Velocity scale (km/s); overrides the stored value.
        use_physical : bool, optional
            Override the object-wide physical-units default.
        quantity : bool, optional
            Return an astropy ``Quantity`` if True.

        Returns
        -------
        float, numpy.ndarray, or Quantity
            Galactocentric Cartesian x in kpc when physical-units output
            is on, in galpy internal units otherwise.
        """
        tp = self._parse_tp(tp)
        return self._cart_eval(0, tp)

    @physical_conversion("position", pop=True)
    def y(self, tp, **kwargs):
        """Galactocentric Cartesian y along the track.

        Parameters
        ----------
        tp : float, array, or Quantity
            Curve parameter(s); Quantity inputs are parsed per
            ``parameter_kind``. Out-of-range entries return ``NaN``.
        ro : float or Quantity, optional
            Distance scale (kpc); overrides the stored value.
        vo : float or Quantity, optional
            Velocity scale (km/s); overrides the stored value.
        use_physical : bool, optional
            Override the object-wide physical-units default.
        quantity : bool, optional
            Return an astropy ``Quantity`` if True.

        Returns
        -------
        float, numpy.ndarray, or Quantity
            Galactocentric Cartesian y in kpc when physical-units output
            is on, in galpy internal units otherwise.
        """
        tp = self._parse_tp(tp)
        return self._cart_eval(1, tp)

    @physical_conversion("position", pop=True)
    def z(self, tp, **kwargs):
        """Galactocentric Cartesian z along the track.

        Parameters
        ----------
        tp : float, array, or Quantity
            Curve parameter(s); Quantity inputs are parsed per
            ``parameter_kind``. Out-of-range entries return ``NaN``.
        ro : float or Quantity, optional
            Distance scale (kpc); overrides the stored value.
        vo : float or Quantity, optional
            Velocity scale (km/s); overrides the stored value.
        use_physical : bool, optional
            Override the object-wide physical-units default.
        quantity : bool, optional
            Return an astropy ``Quantity`` if True.

        Returns
        -------
        float, numpy.ndarray, or Quantity
            Galactocentric Cartesian z in kpc when physical-units output
            is on, in galpy internal units otherwise.
        """
        tp = self._parse_tp(tp)
        return self._cart_eval(2, tp)

    @physical_conversion("velocity", pop=True)
    def vx(self, tp, **kwargs):
        """Galactocentric Cartesian vx along the track.

        Parameters
        ----------
        tp : float, array, or Quantity
            Curve parameter(s); Quantity inputs are parsed per
            ``parameter_kind``. Out-of-range entries return ``NaN``.
        ro : float or Quantity, optional
            Distance scale (kpc); overrides the stored value.
        vo : float or Quantity, optional
            Velocity scale (km/s); overrides the stored value.
        use_physical : bool, optional
            Override the object-wide physical-units default.
        quantity : bool, optional
            Return an astropy ``Quantity`` if True.

        Returns
        -------
        float, numpy.ndarray, or Quantity
            Galactocentric Cartesian vx in km/s when physical-units output
            is on, in galpy internal units otherwise.
        """
        tp = self._parse_tp(tp)
        return self._cart_eval(3, tp)

    @physical_conversion("velocity", pop=True)
    def vy(self, tp, **kwargs):
        """Galactocentric Cartesian vy along the track.

        Parameters
        ----------
        tp : float, array, or Quantity
            Curve parameter(s); Quantity inputs are parsed per
            ``parameter_kind``. Out-of-range entries return ``NaN``.
        ro : float or Quantity, optional
            Distance scale (kpc); overrides the stored value.
        vo : float or Quantity, optional
            Velocity scale (km/s); overrides the stored value.
        use_physical : bool, optional
            Override the object-wide physical-units default.
        quantity : bool, optional
            Return an astropy ``Quantity`` if True.

        Returns
        -------
        float, numpy.ndarray, or Quantity
            Galactocentric Cartesian vy in km/s when physical-units output
            is on, in galpy internal units otherwise.
        """
        tp = self._parse_tp(tp)
        return self._cart_eval(4, tp)

    @physical_conversion("velocity", pop=True)
    def vz(self, tp, **kwargs):
        """Galactocentric Cartesian vz along the track.

        Parameters
        ----------
        tp : float, array, or Quantity
            Curve parameter(s); Quantity inputs are parsed per
            ``parameter_kind``. Out-of-range entries return ``NaN``.
        ro : float or Quantity, optional
            Distance scale (kpc); overrides the stored value.
        vo : float or Quantity, optional
            Velocity scale (km/s); overrides the stored value.
        use_physical : bool, optional
            Override the object-wide physical-units default.
        quantity : bool, optional
            Return an astropy ``Quantity`` if True.

        Returns
        -------
        float, numpy.ndarray, or Quantity
            Galactocentric Cartesian vz in km/s when physical-units output
            is on, in galpy internal units otherwise.
        """
        tp = self._parse_tp(tp)
        return self._cart_eval(5, tp)

    def _cyl_at(self, tp):
        """Return (R, vR, vT, z, vz, phi) along the track (internal units)."""
        tp = numpy.atleast_1d(tp)
        xyz = self._eval_cart(tp)  # (6, len)
        x, y, zc, vxc, vyc, vzc = xyz
        R, phi, zcyl = coords.rect_to_cyl(x, y, zc)
        vR, vT, vz = coords.rect_to_cyl_vec(vxc, vyc, vzc, R, phi, zcyl, cyl=True)
        return R, vR, vT, zcyl, vz, phi

    @physical_conversion("position", pop=True)
    def R(self, tp, **kwargs):
        """Galactocentric cylindrical radius R along the track.

        Parameters
        ----------
        tp : float, array, or Quantity
            Curve parameter(s); Quantity inputs are parsed per
            ``parameter_kind``. Out-of-range entries return ``NaN``.
        ro : float or Quantity, optional
            Distance scale (kpc); overrides the stored value.
        vo : float or Quantity, optional
            Velocity scale (km/s); overrides the stored value.
        use_physical : bool, optional
            Override the object-wide physical-units default.
        quantity : bool, optional
            Return an astropy ``Quantity`` if True.

        Returns
        -------
        float, numpy.ndarray, or Quantity
            Galactocentric cylindrical radius in kpc when physical-units
            output is on, in galpy internal units otherwise.
        """
        tp = self._parse_tp(tp)
        R, _, _, _, _, _ = self._cyl_at(tp)
        return self._maybe_scalar(tp, R)

    @physical_conversion("velocity", pop=True)
    def vR(self, tp, **kwargs):
        """Galactocentric cylindrical radial velocity vR along the track.

        Parameters
        ----------
        tp : float, array, or Quantity
            Curve parameter(s); Quantity inputs are parsed per
            ``parameter_kind``. Out-of-range entries return ``NaN``.
        ro : float or Quantity, optional
            Distance scale (kpc); overrides the stored value.
        vo : float or Quantity, optional
            Velocity scale (km/s); overrides the stored value.
        use_physical : bool, optional
            Override the object-wide physical-units default.
        quantity : bool, optional
            Return an astropy ``Quantity`` if True.

        Returns
        -------
        float, numpy.ndarray, or Quantity
            Cylindrical radial velocity in km/s when physical-units output
            is on, in galpy internal units otherwise.
        """
        tp = self._parse_tp(tp)
        _, vR, _, _, _, _ = self._cyl_at(tp)
        return self._maybe_scalar(tp, vR)

    @physical_conversion("velocity", pop=True)
    def vT(self, tp, **kwargs):
        """Galactocentric cylindrical tangential velocity vT along the track.

        Parameters
        ----------
        tp : float, array, or Quantity
            Curve parameter(s); Quantity inputs are parsed per
            ``parameter_kind``. Out-of-range entries return ``NaN``.
        ro : float or Quantity, optional
            Distance scale (kpc); overrides the stored value.
        vo : float or Quantity, optional
            Velocity scale (km/s); overrides the stored value.
        use_physical : bool, optional
            Override the object-wide physical-units default.
        quantity : bool, optional
            Return an astropy ``Quantity`` if True.

        Returns
        -------
        float, numpy.ndarray, or Quantity
            Cylindrical tangential velocity in km/s when physical-units
            output is on, in galpy internal units otherwise.
        """
        tp = self._parse_tp(tp)
        _, _, vT, _, _, _ = self._cyl_at(tp)
        return self._maybe_scalar(tp, vT)

    @physical_conversion("angle", pop=True)
    def phi(self, tp, **kwargs):
        """Galactocentric cylindrical azimuth phi along the track.

        Parameters
        ----------
        tp : float, array, or Quantity
            Curve parameter(s); Quantity inputs are parsed per
            ``parameter_kind``. Out-of-range entries return ``NaN``.
        ro : float or Quantity, optional
            Distance scale (kpc); overrides the stored value.
        vo : float or Quantity, optional
            Velocity scale (km/s); overrides the stored value.
        use_physical : bool, optional
            Override the object-wide physical-units default.
        quantity : bool, optional
            Return an astropy ``Quantity`` if True.

        Returns
        -------
        float, numpy.ndarray, or Quantity
            Cylindrical azimuth in radians (galpy convention; ``phi``
            stays in radians whether physical-units output is on or off).
        """
        tp = self._parse_tp(tp)
        _, _, _, _, _, phi = self._cyl_at(tp)
        return self._maybe_scalar(tp, phi)

    def __call__(self, tp):
        """Return ``(R, vR, vT, z, vz, phi)`` stacked along the track at ``tp``.

        Parameters
        ----------
        tp : float, array, or Quantity
            Curve parameter(s); Quantity inputs are parsed per
            ``parameter_kind``. Out-of-range entries return ``NaN``.

        Returns
        -------
        numpy.ndarray
            Shape ``(6,)`` for scalar ``tp``, ``(6, len(tp))`` for array
            ``tp``. Always in galpy internal units (no per-call physical
            override on ``__call__``)."""
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

    @physical_conversion("angle_deg", pop=True)
    def ra(self, tp, **kwargs):
        """Equatorial right ascension along the track.

        Parameters
        ----------
        tp : float, array, or Quantity
            Curve parameter(s); Quantity inputs are parsed per
            ``parameter_kind``. Out-of-range entries return ``NaN``.
        ro : float or Quantity, optional
            Distance scale (kpc); overrides the stored value.
        vo : float or Quantity, optional
            Velocity scale (km/s); overrides the stored value.
        use_physical : bool, optional
            Override the object-wide physical-units default.
        quantity : bool, optional
            Return an astropy ``Quantity`` if True.

        Returns
        -------
        float, numpy.ndarray, or Quantity
            Right ascension in degrees.
        """
        tp = self._parse_tp(tp)
        ra_dec = self._radec_internal(tp)
        return self._maybe_scalar(tp, ra_dec[:, 0])

    @physical_conversion("angle_deg", pop=True)
    def dec(self, tp, **kwargs):
        """Equatorial declination along the track.

        Parameters
        ----------
        tp : float, array, or Quantity
            Curve parameter(s); Quantity inputs are parsed per
            ``parameter_kind``. Out-of-range entries return ``NaN``.
        ro : float or Quantity, optional
            Distance scale (kpc); overrides the stored value.
        vo : float or Quantity, optional
            Velocity scale (km/s); overrides the stored value.
        use_physical : bool, optional
            Override the object-wide physical-units default.
        quantity : bool, optional
            Return an astropy ``Quantity`` if True.

        Returns
        -------
        float, numpy.ndarray, or Quantity
            Declination in degrees.
        """
        tp = self._parse_tp(tp)
        ra_dec = self._radec_internal(tp)
        return self._maybe_scalar(tp, ra_dec[:, 1])

    @physical_conversion("angle_deg", pop=True)
    def ll(self, tp, **kwargs):
        """Galactic longitude along the track.

        Parameters
        ----------
        tp : float, array, or Quantity
            Curve parameter(s); Quantity inputs are parsed per
            ``parameter_kind``. Out-of-range entries return ``NaN``.
        ro : float or Quantity, optional
            Distance scale (kpc); overrides the stored value.
        vo : float or Quantity, optional
            Velocity scale (km/s); overrides the stored value.
        use_physical : bool, optional
            Override the object-wide physical-units default.
        quantity : bool, optional
            Return an astropy ``Quantity`` if True.

        Returns
        -------
        float, numpy.ndarray, or Quantity
            Galactic longitude ``l`` in degrees.
        """
        tp = self._parse_tp(tp)
        X, Y, Z, _, _, _ = self._helio_xv(tp)
        lbd = coords.XYZ_to_lbd(X, Y, Z, degree=True)
        return self._maybe_scalar(tp, lbd[:, 0])

    @physical_conversion("angle_deg", pop=True)
    def bb(self, tp, **kwargs):
        """Galactic latitude along the track.

        Parameters
        ----------
        tp : float, array, or Quantity
            Curve parameter(s); Quantity inputs are parsed per
            ``parameter_kind``. Out-of-range entries return ``NaN``.
        ro : float or Quantity, optional
            Distance scale (kpc); overrides the stored value.
        vo : float or Quantity, optional
            Velocity scale (km/s); overrides the stored value.
        use_physical : bool, optional
            Override the object-wide physical-units default.
        quantity : bool, optional
            Return an astropy ``Quantity`` if True.

        Returns
        -------
        float, numpy.ndarray, or Quantity
            Galactic latitude ``b`` in degrees.
        """
        tp = self._parse_tp(tp)
        X, Y, Z, _, _, _ = self._helio_xv(tp)
        lbd = coords.XYZ_to_lbd(X, Y, Z, degree=True)
        return self._maybe_scalar(tp, lbd[:, 1])

    @physical_conversion("position_kpc", pop=True)
    def dist(self, tp, **kwargs):
        """Heliocentric distance along the track.

        Parameters
        ----------
        tp : float, array, or Quantity
            Curve parameter(s); Quantity inputs are parsed per
            ``parameter_kind``. Out-of-range entries return ``NaN``.
        ro : float or Quantity, optional
            Distance scale (kpc); overrides the stored value.
        vo : float or Quantity, optional
            Velocity scale (km/s); overrides the stored value.
        use_physical : bool, optional
            Override the object-wide physical-units default.
        quantity : bool, optional
            Return an astropy ``Quantity`` if True.

        Returns
        -------
        float, numpy.ndarray, or Quantity
            Heliocentric distance in kpc.
        """
        tp = self._parse_tp(tp)
        X, Y, Z, _, _, _ = self._helio_xv(tp)
        lbd = coords.XYZ_to_lbd(X, Y, Z, degree=True)
        return self._maybe_scalar(tp, lbd[:, 2])

    @physical_conversion("proper-motion_masyr", pop=True)
    def pmra(self, tp, **kwargs):
        """Proper motion in right ascension along the track, multiplied by
        ``cos(dec)``.

        Parameters
        ----------
        tp : float, array, or Quantity
            Curve parameter(s); Quantity inputs are parsed per
            ``parameter_kind``. Out-of-range entries return ``NaN``.
        ro : float or Quantity, optional
            Distance scale (kpc); overrides the stored value.
        vo : float or Quantity, optional
            Velocity scale (km/s); overrides the stored value.
        use_physical : bool, optional
            Override the object-wide physical-units default.
        quantity : bool, optional
            Return an astropy ``Quantity`` if True.

        Returns
        -------
        float, numpy.ndarray, or Quantity
            ``pmra * cos(dec)`` in mas/yr.
        """
        tp = self._parse_tp(tp)
        pmrapmdec = self._pmrapmdec_internal(tp)
        return self._maybe_scalar(tp, pmrapmdec[:, 0])

    @physical_conversion("proper-motion_masyr", pop=True)
    def pmdec(self, tp, **kwargs):
        """Proper motion in declination along the track.

        Parameters
        ----------
        tp : float, array, or Quantity
            Curve parameter(s); Quantity inputs are parsed per
            ``parameter_kind``. Out-of-range entries return ``NaN``.
        ro : float or Quantity, optional
            Distance scale (kpc); overrides the stored value.
        vo : float or Quantity, optional
            Velocity scale (km/s); overrides the stored value.
        use_physical : bool, optional
            Override the object-wide physical-units default.
        quantity : bool, optional
            Return an astropy ``Quantity`` if True.

        Returns
        -------
        float, numpy.ndarray, or Quantity
            ``pmdec`` in mas/yr.
        """
        tp = self._parse_tp(tp)
        pmrapmdec = self._pmrapmdec_internal(tp)
        return self._maybe_scalar(tp, pmrapmdec[:, 1])

    @physical_conversion("proper-motion_masyr", pop=True)
    def pmll(self, tp, **kwargs):
        """Proper motion in Galactic longitude along the track, multiplied
        by ``cos(b)``.

        Parameters
        ----------
        tp : float, array, or Quantity
            Curve parameter(s); Quantity inputs are parsed per
            ``parameter_kind``. Out-of-range entries return ``NaN``.
        ro : float or Quantity, optional
            Distance scale (kpc); overrides the stored value.
        vo : float or Quantity, optional
            Velocity scale (km/s); overrides the stored value.
        use_physical : bool, optional
            Override the object-wide physical-units default.
        quantity : bool, optional
            Return an astropy ``Quantity`` if True.

        Returns
        -------
        float, numpy.ndarray, or Quantity
            ``pmll * cos(b)`` in mas/yr.
        """
        tp = self._parse_tp(tp)
        _, vrpmllpmbb = self._vrpmllpmbb(tp)
        return self._maybe_scalar(tp, vrpmllpmbb[:, 1])

    @physical_conversion("proper-motion_masyr", pop=True)
    def pmbb(self, tp, **kwargs):
        """Proper motion in Galactic latitude along the track.

        Parameters
        ----------
        tp : float, array, or Quantity
            Curve parameter(s); Quantity inputs are parsed per
            ``parameter_kind``. Out-of-range entries return ``NaN``.
        ro : float or Quantity, optional
            Distance scale (kpc); overrides the stored value.
        vo : float or Quantity, optional
            Velocity scale (km/s); overrides the stored value.
        use_physical : bool, optional
            Override the object-wide physical-units default.
        quantity : bool, optional
            Return an astropy ``Quantity`` if True.

        Returns
        -------
        float, numpy.ndarray, or Quantity
            ``pmbb`` in mas/yr.
        """
        tp = self._parse_tp(tp)
        _, vrpmllpmbb = self._vrpmllpmbb(tp)
        return self._maybe_scalar(tp, vrpmllpmbb[:, 2])

    @physical_conversion("velocity_kms", pop=True)
    def vlos(self, tp, **kwargs):
        """Heliocentric line-of-sight (radial) velocity along the track.

        Parameters
        ----------
        tp : float, array, or Quantity
            Curve parameter(s); Quantity inputs are parsed per
            ``parameter_kind``. Out-of-range entries return ``NaN``.
        ro : float or Quantity, optional
            Distance scale (kpc); overrides the stored value.
        vo : float or Quantity, optional
            Velocity scale (km/s); overrides the stored value.
        use_physical : bool, optional
            Override the object-wide physical-units default.
        quantity : bool, optional
            Return an astropy ``Quantity`` if True.

        Returns
        -------
        float, numpy.ndarray, or Quantity
            Line-of-sight velocity in km/s.
        """
        tp = self._parse_tp(tp)
        _, vrpmllpmbb = self._vrpmllpmbb(tp)
        return self._maybe_scalar(tp, vrpmllpmbb[:, 0])

    # -----------------------------------------------------------------
    # Custom sky frame (requires ``custom_transform`` to be set)
    # -----------------------------------------------------------------
    @property
    def custom_transform(self):
        """3x3 rotation matrix from (ra, dec) to the custom sky frame, or
        None. Settable post-construction: assigning a new matrix updates
        the frame used by the ``phi1``/``phi2``/``pmphi1``/``pmphi2``
        accessors and by ``plot``/``cov`` in the ``customsky`` basis."""
        return self._custom_transform

    @custom_transform.setter
    def custom_transform(self, T):
        self._custom_transform = None if T is None else numpy.asarray(T, dtype=float)

    def _require_custom(self):
        if self._custom_transform is None:
            raise RuntimeError(
                "custom_transform was not set at track construction; "
                "the phi1/phi2/pmphi1/pmphi2 accessors require a rotation "
                "matrix (3x3) from (ra, dec) to the custom sky frame. "
                "See galpy.util.coords.align_to_orbit for a helper that "
                "builds one from a progenitor Orbit. You can also assign "
                "to the ``custom_transform`` attribute on an existing "
                "track to set or replace the matrix after construction."
            )

    def _radec_internal(self, tp):
        """ra, dec along the track in degrees (no unit attached)."""
        X, Y, Z, _, _, _ = self._helio_xv(tp)
        lbd = coords.XYZ_to_lbd(X, Y, Z, degree=True)
        return coords.lb_to_radec(lbd[:, 0], lbd[:, 1], degree=True)

    def _pmrapmdec_internal(self, tp):
        """pmra, pmdec along the track in mas/yr (no unit attached)."""
        lbd, vrpmllpmbb = self._vrpmllpmbb(tp)
        return coords.pmllpmbb_to_pmrapmdec(
            vrpmllpmbb[:, 1],
            vrpmllpmbb[:, 2],
            lbd[:, 0],
            lbd[:, 1],
            degree=True,
        )

    @physical_conversion("angle_deg", pop=True)
    def phi1(self, tp, **kwargs):
        """Custom-frame longitude ``phi1`` along the track.

        Requires ``custom_transform`` to have been set at construction.

        Parameters
        ----------
        tp : float, array, or Quantity
            Curve parameter(s); Quantity inputs are parsed per
            ``parameter_kind``. Out-of-range entries return ``NaN``.
        ro : float or Quantity, optional
            Distance scale (kpc); overrides the stored value.
        vo : float or Quantity, optional
            Velocity scale (km/s); overrides the stored value.
        use_physical : bool, optional
            Override the object-wide physical-units default.
        quantity : bool, optional
            Return an astropy ``Quantity`` if True.

        Returns
        -------
        float, numpy.ndarray, or Quantity
            Custom-frame ``phi1`` in degrees.
        """
        self._require_custom()
        tp = self._parse_tp(tp)
        ra_dec = self._radec_internal(tp)
        p12 = coords.radec_to_custom(
            ra_dec[:, 0], ra_dec[:, 1], T=self._custom_transform, degree=True
        )
        return self._maybe_scalar(tp, p12[:, 0])

    @physical_conversion("angle_deg", pop=True)
    def phi2(self, tp, **kwargs):
        """Custom-frame latitude ``phi2`` along the track.

        Requires ``custom_transform`` to have been set at construction.

        Parameters
        ----------
        tp : float, array, or Quantity
            Curve parameter(s); Quantity inputs are parsed per
            ``parameter_kind``. Out-of-range entries return ``NaN``.
        ro : float or Quantity, optional
            Distance scale (kpc); overrides the stored value.
        vo : float or Quantity, optional
            Velocity scale (km/s); overrides the stored value.
        use_physical : bool, optional
            Override the object-wide physical-units default.
        quantity : bool, optional
            Return an astropy ``Quantity`` if True.

        Returns
        -------
        float, numpy.ndarray, or Quantity
            Custom-frame ``phi2`` in degrees.
        """
        self._require_custom()
        tp = self._parse_tp(tp)
        ra_dec = self._radec_internal(tp)
        p12 = coords.radec_to_custom(
            ra_dec[:, 0], ra_dec[:, 1], T=self._custom_transform, degree=True
        )
        return self._maybe_scalar(tp, p12[:, 1])

    @physical_conversion("proper-motion_masyr", pop=True)
    def pmphi1(self, tp, **kwargs):
        """Proper motion in custom-frame ``phi1`` along the track,
        multiplied by ``cos(phi2)``.

        Requires ``custom_transform`` to have been set at construction.

        Parameters
        ----------
        tp : float, array, or Quantity
            Curve parameter(s); Quantity inputs are parsed per
            ``parameter_kind``. Out-of-range entries return ``NaN``.
        ro : float or Quantity, optional
            Distance scale (kpc); overrides the stored value.
        vo : float or Quantity, optional
            Velocity scale (km/s); overrides the stored value.
        use_physical : bool, optional
            Override the object-wide physical-units default.
        quantity : bool, optional
            Return an astropy ``Quantity`` if True.

        Returns
        -------
        float, numpy.ndarray, or Quantity
            ``pmphi1 * cos(phi2)`` in mas/yr.
        """
        self._require_custom()
        tp = self._parse_tp(tp)
        ra_dec = self._radec_internal(tp)
        pmrapmdec = self._pmrapmdec_internal(tp)
        pm12 = coords.pmrapmdec_to_custom(
            pmrapmdec[:, 0],
            pmrapmdec[:, 1],
            ra_dec[:, 0],
            ra_dec[:, 1],
            T=self._custom_transform,
            degree=True,
        )
        return self._maybe_scalar(tp, pm12[:, 0])

    @physical_conversion("proper-motion_masyr", pop=True)
    def pmphi2(self, tp, **kwargs):
        """Proper motion in custom-frame ``phi2`` along the track.

        Requires ``custom_transform`` to have been set at construction.

        Parameters
        ----------
        tp : float, array, or Quantity
            Curve parameter(s); Quantity inputs are parsed per
            ``parameter_kind``. Out-of-range entries return ``NaN``.
        ro : float or Quantity, optional
            Distance scale (kpc); overrides the stored value.
        vo : float or Quantity, optional
            Velocity scale (km/s); overrides the stored value.
        use_physical : bool, optional
            Override the object-wide physical-units default.
        quantity : bool, optional
            Return an astropy ``Quantity`` if True.

        Returns
        -------
        float, numpy.ndarray, or Quantity
            ``pmphi2`` in mas/yr.
        """
        self._require_custom()
        tp = self._parse_tp(tp)
        ra_dec = self._radec_internal(tp)
        pmrapmdec = self._pmrapmdec_internal(tp)
        pm12 = coords.pmrapmdec_to_custom(
            pmrapmdec[:, 0],
            pmrapmdec[:, 1],
            ra_dec[:, 0],
            ra_dec[:, 1],
            T=self._custom_transform,
            degree=True,
        )
        return self._maybe_scalar(tp, pm12[:, 1])

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

    def cov(
        self, tp, basis="galcenrect", ro=None, vo=None, use_physical=None, quantity=None
    ):
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
        ro : float or Quantity, optional
            Distance scale for physical-units output (overrides the stored
            value). Mirrors the per-call override on the mean accessors.
        vo : float or Quantity, optional
            Velocity scale for physical-units output (overrides the stored
            value).
        use_physical : bool, optional
            Override the object-wide default (``_roSet`` and ``_voSet``)
            for whether to scale the covariance to physical units. When
            ``False``, the covariance is returned in galpy internal units
            for galactocentric bases; sky bases retain the prior behavior
            of running the Jacobian with the stored ``ro``/``vo`` defaults
            (output remains unitless in the internal sense, useful only
            as a relative shape).
        quantity : bool, optional
            Not supported: a covariance 6x6 has heterogeneous units across
            entries (e.g. kpc² for position-position, kpc·km/s for
            position-velocity, deg² for sky-sky) and can't be wrapped as
            a single astropy ``Quantity``. Passing ``True`` raises
            ``NotImplementedError``.

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

        if quantity:
            raise NotImplementedError(
                "cov() does not support quantity=True: the 6x6 has "
                "heterogeneous units across rows/columns and can't be "
                "wrapped as a single astropy Quantity."
            )

        # Resolve unit settings (mirrors the @physical_conversion semantics
        # used by the mean accessors, but applied by hand because the
        # decorator scales by a single factor and our entries scale by
        # outer(scale, scale)).
        if use_physical is None:
            use_phys = self._roSet and self._voSet
        else:
            use_phys = bool(use_physical)
        if use_phys:
            ro_use = self._ro if ro is None else conversion.parse_length_kpc(ro)
            vo_use = self._vo if vo is None else conversion.parse_velocity_kms(vo)

        tp = self._parse_tp(tp)
        tp_arr = numpy.atleast_1d(tp)
        in_range = self._in_range(tp_arr)
        out = numpy.empty((len(tp_arr), 6, 6))
        for a in range(6):
            for b in range(6):
                vals = numpy.interp(tp_arr, self._tp_grid, self._cov_xyz[:, a, b])
                out[:, a, b] = numpy.where(in_range, vals, numpy.nan)
        if use_phys:
            scale = numpy.array([ro_use, ro_use, ro_use, vo_use, vo_use, vo_use])
            out = out * numpy.outer(scale, scale)  # NaN · scale = NaN

        if basis != "galcenrect":
            # When use_phys=True we thread the resolved ro/vo through the
            # Jacobian so an override is consistent with the outer-scale.
            # When use_phys=False, we let _analytical_jacobian fall back to
            # its stored-default behavior (self._ro / self._vo) — matching
            # the prior pre-override semantics so callers that relied on
            # cov(basis=...) with _roSet=False keep working.
            jac_ro = ro_use if use_phys else None
            jac_vo = vo_use if use_phys else None
            jac_use_phys = True if use_phys else None
            for k, tp_k in enumerate(tp_arr):
                if not in_range[k]:
                    continue  # out[k] already NaN; skip the Jacobian
                J = self._analytical_jacobian(
                    tp_k,
                    basis,
                    ro=jac_ro,
                    vo=jac_vo,
                    use_physical=jac_use_phys,
                )
                out[k] = J @ out[k] @ J.T

        if numpy.isscalar(tp) or (hasattr(tp, "ndim") and tp.ndim == 0):
            return out[0]
        return out

    # -----------------------------------------------------------------
    # Analytical-Jacobian evaluator (used by cov(basis=..) and
    # plot(spread=) on non-galcenrect axes).
    # -----------------------------------------------------------------
    def _cart_mean_at(self, tp, ro=None, vo=None, use_physical=None):
        """Mean (x, y, z, vx, vy, vz) at ``tp`` in the track's physical
        state (kpc, km/s when physical is on, else internal). The optional
        ``ro``/``vo``/``use_physical`` kwargs are forwarded to the mean
        accessors so callers can request a specific unit choice."""
        kw = {"quantity": False}
        if use_physical is not None:
            kw["use_physical"] = use_physical
        if ro is not None:
            kw["ro"] = ro
        if vo is not None:
            kw["vo"] = vo
        return numpy.array(
            [
                float(self.x(tp, **kw)),
                float(self.y(tp, **kw)),
                float(self.z(tp, **kw)),
                float(self.vx(tp, **kw)),
                float(self.vy(tp, **kw)),
                float(self.vz(tp, **kw)),
            ]
        )

    def _analytical_jacobian(self, tp, basis, ro=None, vo=None, use_physical=None):
        """6x6 analytical Jacobian d(basis)/d(galcenrect) at the track mean.

        Chain (each link is a self-contained Jacobian helper in
        :mod:`galpy.util.coords`):

        * ``galcenrect_to_galcencyl_jac`` — closed-form cylindrical change of
          variable.
        * ``galcenrect_to_XYZ_jac`` — galactocentric → heliocentric (block-
          diagonal rotation, parameterised by Xsun/Zsun).
        * ``XYZ_to_lbd_jac`` — heliocentric Cartesian → Galactic spherical
          (analytical inverse of :func:`coords.lbd_to_XYZ_jac`, no LAPACK).
        * ``galsky_to_sky_jac`` — local tangent rotation at the sky point
          mapping (l, b, pmll, pmbb) ↔ (ra, dec, pmra, pmdec).
        * ``sky_to_customsky_jac`` — same rotation form parameterised by
          ``custom_transform``.

        ``ro``/``vo``/``use_physical`` are forwarded to ``_cart_mean_at``
        so the Jacobian is consistent with the unit choice the caller
        used to scale the input covariance.
        """
        mean = self._cart_mean_at(tp, ro=ro, vo=vo, use_physical=use_physical)
        x, y, z, vx, vy, vz = mean

        if basis == "galcencyl":
            return coords.galcenrect_to_galcencyl_jac(x, y, z, vx, vy, vz)

        # Sky bases — chain through heliocentric Galactic Cartesian. Per-call
        # ``ro`` (already resolved to a float by cov()) when supplied, else
        # the stored solar radius.
        if ro is None:
            ro = float(self._ro) if self._ro is not None else 1.0
        else:
            ro = float(ro)
        zo = float(self._zo) if self._zo is not None else 0.0
        # (1) galcenrect → heliocentric XYZ.
        J1 = coords.galcenrect_to_XYZ_jac(x, y, z, vx, vy, vz, Xsun=ro, Zsun=zo)
        # Evaluate the mean in heliocentric Cartesian for downstream chains.
        XYZ_mean = coords.galcenrect_to_XYZ(x, y, z, Xsun=ro, Zsun=zo)
        vxyz_mean = coords.galcenrect_to_vxvyvz(vx, vy, vz, Xsun=ro, Zsun=zo)
        X, Y, Z = float(XYZ_mean[0]), float(XYZ_mean[1]), float(XYZ_mean[2])
        vX, vY, vZ = float(vxyz_mean[0]), float(vxyz_mean[1]), float(vxyz_mean[2])
        # (2) helio_XYZ → galsky. Note: lbd_to_XYZ_jac uses the order
        # (l, b, d, vlos, pmll, pmbb); XYZ_to_lbd_jac inherits that order,
        # so we permute its output to match our galsky basis ordering
        # (l, b, d, pmll, pmbb, vlos).
        J_xyz_to_lbd = coords.XYZ_to_lbd_jac(X, Y, Z, vX, vY, vZ, degree=False)
        perm = numpy.array([0, 1, 2, 4, 5, 3])
        J_galsky = numpy.eye(6)[perm] @ J_xyz_to_lbd @ J1  # l, b in radians
        if basis == "galsky":
            J_galsky = J_galsky.copy()
            J_galsky[0] *= 1.0 / coords._DEGTORAD
            J_galsky[1] *= 1.0 / coords._DEGTORAD
            return J_galsky

        # (3) galsky → sky. Need (l, b, pmll, pmbb) for the position-vs-PM
        # cross block; recover them from the heliocentric Cartesian state.
        lbd_mean = coords.XYZ_to_lbd(X, Y, Z, degree=False)
        l = float(lbd_mean[0])
        b = float(lbd_mean[1])
        vrpm = coords.vxvyvz_to_vrpmllpmbb(vX, vY, vZ, X, Y, Z, XYZ=True, degree=False)
        pmll = float(vrpm[1])
        pmbb = float(vrpm[2])
        J_galsky_to_sky = coords.galsky_to_sky_jac(l, b, pmll, pmbb, degree=False)
        J_sky = J_galsky_to_sky @ J_galsky  # ra, dec in radians
        if basis == "sky":
            J_sky = J_sky.copy()
            J_sky[0] *= 1.0 / coords._DEGTORAD
            J_sky[1] *= 1.0 / coords._DEGTORAD
            return J_sky

        # (4) sky → customsky. Need (ra, dec, pmra, pmdec) at the mean for
        # the PM-vs-position cross block.
        radec = coords.lb_to_radec(l, b, degree=False)
        ra = float(radec[0])
        dec_val = float(radec[1])
        pmrd = coords.pmllpmbb_to_pmrapmdec(pmll, pmbb, l, b, degree=False)
        pmra = float(pmrd[0])
        pmdec = float(pmrd[1])
        J_sky_to_cs = coords.sky_to_customsky_jac(
            ra, dec_val, pmra, pmdec, T=self._custom_transform, degree=False
        )
        J_cs = J_sky_to_cs @ J_sky  # phi1, phi2 in radians
        J_cs = J_cs.copy()
        J_cs[0] *= 1.0 / coords._DEGTORAD
        J_cs[1] *= 1.0 / coords._DEGTORAD
        return J_cs

    # -----------------------------------------------------------------
    # Unit toggles
    # -----------------------------------------------------------------
    def turn_physical_on(self, ro=None, vo=None):
        """Turn on physical-units output.

        Parameters
        ----------
        ro : float or Quantity, optional
            Distance scale (kpc); if given, sets it on the track.
        vo : float or Quantity, optional
            Velocity scale (km/s); if given, sets it on the track.
        """
        if ro is not None:
            self._ro = conversion.parse_length_kpc(ro)
        if vo is not None:
            self._vo = conversion.parse_velocity_kms(vo)
        self._roSet = True
        self._voSet = True

    def turn_physical_off(self):
        """Turn off physical-units output."""
        self._roSet = False
        self._voSet = False

    # -----------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------
    # Coord → (basis, index-within-basis) dispatch for plot(spread=),
    # derived from _BASIS_COORDS so the two stay in sync. For names that
    # appear in multiple bases (z/vz in galcenrect+galcencyl; dist/vlos in
    # sky/galsky/customsky), the first-listed basis wins — iterating in
    # reverse and letting dict-comprehension overwrites finalize the
    # mapping yields first-occurrence-wins semantics.
    _COORD_BASIS = {
        name: (basis, i)
        for basis, names in reversed(list(_BASIS_COORDS.items()))
        for i, name in enumerate(names)
    }

    def plot(self, d1="x", d2="y", spread=0, **kwargs):
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
            that has a basis in :attr:`_COORD_BASIS`. Silently skipped
            for axes outside the dispatch (e.g. ``ll``, ``bb``).
        ro : float or Quantity, optional
            Distance scale to use for the conversion (overrides
            ``self._ro``).
        vo : float or Quantity, optional
            Velocity scale to use for the conversion (overrides
            ``self._vo``).
        use_physical : bool, optional
            Override the object-wide default (set by ``_roSet``/``_voSet``)
            for using physical units in the plot.
        **kwargs
            Passed to matplotlib.pyplot.plot.

        Notes
        -----
        Uses the same physical-unit logic as ``Orbit.plot``: physical
        output is on iff ``_roSet`` and ``_voSet`` are True (or the user
        passes ``use_physical=True``/explicit ``ro=`` / ``vo=``); axis
        labels are picked accordingly.
        """
        # Forward unit knobs through to the accessors and to cov(); both
        # share the same default-resolution semantics (None → fall back to
        # _roSet/_voSet and stored ro/vo) so we don't re-derive that here.
        use_physical = kwargs.pop("use_physical", None)
        ro = kwargs.pop("ro", None)
        vo = kwargs.pop("vo", None)
        access_kw = {"quantity": False}
        if use_physical is not None:
            access_kw["use_physical"] = use_physical
        if ro is not None:
            access_kw["ro"] = ro
        if vo is not None:
            access_kw["vo"] = vo

        tp = numpy.linspace(self._tp_grid[0], self._tp_grid[-1], self._ninterp)
        v1 = numpy.asarray(getattr(self, d1)(tp, **access_kw))
        v2 = numpy.asarray(getattr(self, d2)(tp, **access_kw))
        line = pyplot.plot(v1, v2, **kwargs)
        if spread > 0 and self._cov_xyz is not None and d2 in self._COORD_BASIS:
            basis, idx = self._COORD_BASIS[d2]
            cov = self.cov(
                tp,
                basis=basis,
                ro=ro,
                vo=vo,
                use_physical=use_physical,
            )
            s2 = numpy.sqrt(numpy.maximum(cov[:, idx, idx], 0.0))
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

    @property
    def particles(self):
        """Concatenated ``(xv, dt)`` from both arms in the leading-first
        order that ``streamspraydf.streamTrack(tail='both', particles=...)``
        expects, so a pair built with ``tail='both'`` can be re-fit at
        different smoothing / iteration settings without re-sampling::

            pair = spdf.streamTrack(n=3000, tail='both')
            pair_smoother = spdf.streamTrack(particles=pair.particles,
                                             tail='both',
                                             smoothing_factor=2.0)

        Available only when the underlying arms were built via
        :meth:`StreamTrack.from_particles` (the ``streamspraydf.streamTrack``
        path); raises ``AttributeError`` otherwise.
        """
        xv_l, dt_l = self.leading.particles
        xv_t, dt_t = self.trailing.particles
        return numpy.hstack([xv_l, xv_t]), numpy.concatenate([dt_l, dt_t])

    @property
    def custom_transform(self):
        """3x3 rotation matrix shared by both arms (returns the leading
        arm's). Settable: assigning broadcasts the new matrix to both
        :class:`StreamTrack` instances."""
        return self.leading.custom_transform

    @custom_transform.setter
    def custom_transform(self, T):
        self.leading.custom_transform = T
        self.trailing.custom_transform = T

    def turn_physical_on(self, ro=None, vo=None):
        self.leading.turn_physical_on(ro=ro, vo=vo)
        self.trailing.turn_physical_on(ro=ro, vo=vo)

    def turn_physical_off(self):
        self.leading.turn_physical_off()
        self.trailing.turn_physical_off()

    def plot(self, d1="x", d2="y", spread=0, **kwargs):
        """Plot both arms on the same axes (mirroring ``StreamTrack.plot``).

        Accepts ``ro``/``vo``/``use_physical`` for per-call unit overrides
        like ``Orbit.plot``."""
        return [
            self.leading.plot(d1=d1, d2=d2, spread=spread, **kwargs),
            self.trailing.plot(d1=d1, d2=d2, spread=spread, **kwargs),
        ]
