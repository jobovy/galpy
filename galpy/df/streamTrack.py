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


def _bin_offsets(tp_assign, offsets, tp_nodes):
    """Bin per-particle offsets (N,6) by tp_assign onto uniform nodes.
    Returns (mean (M,6), cov (M,6,6), counts (M,)) where M = len(tp_nodes).
    Bins are centered on each node; width is the spacing.
    Empty bins are returned as NaN mean and zero cov with count 0.
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
    idx = numpy.searchsorted(edges, tp_assign) - 1
    idx = numpy.clip(idx, 0, M - 1)
    means = numpy.full((M, 6), numpy.nan)
    covs = numpy.zeros((M, 6, 6))
    counts = numpy.zeros(M, dtype=int)
    for m in range(M):
        sel = idx == m
        k = int(sel.sum())
        counts[m] = k
        if k < 2:
            # 0 particles: leave mean=NaN, cov=0. 1 particle: undefined
            # sample covariance, treat the same as empty.
            continue
        group = offsets[sel]
        means[m] = group.mean(axis=0)
        covs[m] = numpy.cov(group, rowvar=False)
    return means, covs, counts


def _smooth_series(x, y, sigma, s_user=None):
    """Fit a weighted smoothing spline y(x) with weights w = 1/sigma.

    When ``s_user`` is None, the default smoothing parameter is the
    chi-square-like target s = (number of valid bins), which produces a
    spline whose weighted residuals are of order unity per point.

    Falls back to an interpolating spline if fewer than 4 valid points.
    """
    mask = numpy.isfinite(y) & numpy.isfinite(x)
    n_valid = int(mask.sum())
    order = numpy.argsort(x[mask])
    xv = x[mask][order]
    yv = y[mask][order]
    if n_valid < 4:
        # Too few valid bins for a cubic spline. Pad so that a linear
        # interpolator is always well-defined: ensure at least two points,
        # duplicating a reference value (zero if empty) when needed.
        if n_valid < 2:
            ref_y = float(yv[0]) if n_valid == 1 else 0.0
            xv = numpy.array([-1.0, 0.0])
            yv = numpy.array([ref_y, ref_y])
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
    if not numpy.isfinite(sig_med) or sig_med == 0:
        sig_med = 1.0
    sv = numpy.where(numpy.isfinite(sig_safe), sig_safe, sig_med)[mask][order]
    sv = numpy.maximum(sv, 1e-12)
    s_eff = float(n_valid) if s_user is None else float(s_user)
    return interpolate.UnivariateSpline(xv, yv, w=1.0 / sv, s=s_eff, k=3)


class StreamTrack:
    """Smooth phase-space track for a single arm of a stream.

    A StreamTrack holds a smooth mean curve (and optional covariance) in
    galactic phase space, parameterized by a progenitor time coordinate
    ``tp`` in ``[-tdisrupt, 0]``. It is constructed from a cloud of stream
    particles (e.g. drawn via ``basestreamspraydf.sample``) plus the
    progenitor's orbit.

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
        progenitor,
        tdisrupt,
        center=None,
        ntp=101,
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
            Stripping times (positive, in galpy internal time units) for each
            particle.
        progenitor : galpy.orbit.Orbit
            Progenitor orbit, already integrated over the range of the stream
            (typically [-tdisrupt, 0]) in unphysical units.
        tdisrupt : float
            Total time since start of disruption (positive, galpy units).
        center : galpy.orbit.Orbit, optional
            Center orbit (for streams around a moving satellite), also
            pre-integrated.
        ntp : int, optional
            Number of nodes in the tp grid used for binning and smoothing.
        ninterp : int, optional
            Resolution of the fine tp grid on which the public track is
            stored.
        smoothing : None, float, or dict, optional
            Smoothing parameter for the mean-offset spline per Cartesian
            coordinate. ``None`` uses an automatic value s = N * <sigma^2>.
            A float sets the same s for all six coordinates. A dict keyed by
            ``'x','y','z','vx','vy','vz'`` sets per-coordinate values.
        niter : int, optional
            Iterations beyond the initial fit. At each iteration, particles
            are reassigned to the closest point on the current track in
            galactocentric Cartesian position, and the offsets and covariance
            are refit.
        order : int, optional
            Moments to keep: 1 = mean only, 2 = mean + covariance. Hooks exist
            for higher orders in future.
        ro, vo, zo, solarmotion, roSet, voSet
            Physical-unit configuration (typically inherited from the
            progenitor Orbit).
        """
        self._progenitor = progenitor
        self._tdisrupt = float(tdisrupt)
        self._center = center
        self._ntp = int(ntp)
        self._ninterp = int(ninterp)
        self._order = int(order)
        self._ro = ro
        self._vo = vo
        self._zo = zo
        self._solarmotion = solarmotion
        self._roSet = roSet
        self._voSet = voSet
        self._physical = False

        # Store particles (copies)
        self._xv = numpy.asarray(xv_particles, dtype=float).copy()
        self._dt = numpy.asarray(dt_particles, dtype=float).copy()

        # Normalize smoothing arg
        self._coord_names = ("x", "y", "z", "vx", "vy", "vz")
        if smoothing is None:
            self._s_user = {c: None for c in self._coord_names}
        elif isinstance(smoothing, dict):
            self._s_user = {c: smoothing.get(c, None) for c in self._coord_names}
        else:
            self._s_user = {c: float(smoothing) for c in self._coord_names}

        # Build the track
        tp_nodes = numpy.linspace(-self._tdisrupt, 0.0, self._ntp)
        # Fine tp grid (public)
        self._tp_grid = numpy.linspace(-self._tdisrupt, 0.0, self._ninterp)

        # Initial tp assignment: stripping time -> -dt
        tp_assign = -self._dt.copy()
        tp_assign = numpy.clip(tp_assign, -self._tdisrupt, 0.0)

        for it in range(niter + 1):
            self._fit(tp_assign, tp_nodes)
            if it < niter:
                tp_assign = self._assign_closest_on_track(self._xv)

        # Save final assignment for diagnostics
        self._tp_assign = tp_assign

    # -----------------------------------------------------------------
    # Fitting
    # -----------------------------------------------------------------
    def _fit(self, tp_assign, tp_nodes):
        particles_cart = _particles_to_cartesian(
            self._xv
        )  # (N, 6) in galactocentric Cartesian
        means, covs, counts = _bin_offsets(tp_assign, particles_cart, tp_nodes)
        # Standard error of the bin mean per coord for auto-s
        with numpy.errstate(invalid="ignore"):
            per_coord_sigma = numpy.sqrt(
                numpy.diagonal(covs, axis1=1, axis2=2)
            )  # (M, 6)
            sigma_of_mean = per_coord_sigma / numpy.sqrt(
                numpy.maximum(counts[:, None], 1)
            )
            sigma_of_mean = numpy.where(counts[:, None] > 1, sigma_of_mean, numpy.nan)

        # Fit smoothing spline per coord directly in galactocentric Cartesian
        coord_splines = []
        for i, name in enumerate(self._coord_names):
            spl = _smooth_series(
                tp_nodes,
                means[:, i],
                sigma_of_mean[:, i],
                s_user=self._s_user[name],
            )
            coord_splines.append(spl)

        # Evaluate on the fine tp grid
        track_xyz = numpy.column_stack(
            [coord_splines[i](self._tp_grid) for i in range(3)]
        )
        track_vxvyvz = numpy.column_stack(
            [coord_splines[i](self._tp_grid) for i in range(3, 6)]
        )

        # Covariance: smooth per-entry directly in galactocentric Cartesian
        if self._order >= 2:
            cov_fine = numpy.zeros((self._ninterp, 6, 6))
            for a in range(6):
                for b in range(a, 6):
                    vals = covs[:, a, b]
                    # For a covariance entry from a sample of size k, the
                    # standard error is approximately
                    #   sqrt((C_aa * C_bb + C_ab^2) / k).
                    diag_a = per_coord_sigma[:, a] ** 2
                    diag_b = per_coord_sigma[:, b] ** 2
                    with numpy.errstate(invalid="ignore"):
                        sigma_c = numpy.sqrt(
                            (diag_a * diag_b + vals**2) / numpy.maximum(counts, 2)
                        )
                    sigma_c = numpy.where(counts > 1, sigma_c, numpy.nan)
                    spl = _smooth_series(
                        tp_nodes,
                        vals,
                        sigma_c,
                        s_user=None,
                    )
                    val_fine = spl(self._tp_grid)
                    cov_fine[:, a, b] = val_fine
                    cov_fine[:, b, a] = val_fine
            # Per-entry smoothing can yield slightly non-PSD matrices;
            # project each back onto the PSD cone by clipping negative
            # eigenvalues to zero. Sanitize NaN/Inf entries first
            # (possible in degenerate low-statistics cases).
            cov_fine = numpy.nan_to_num(cov_fine, nan=0.0, posinf=0.0, neginf=0.0)
            for k in range(self._ninterp):
                evals, evecs = numpy.linalg.eigh(cov_fine[k])
                evals = numpy.clip(evals, 0.0, None)
                cov_fine[k] = (evecs * evals) @ evecs.T
            self._cov_xyz = cov_fine
        else:
            self._cov_xyz = None

        self._track_xyz = track_xyz
        self._track_vxvyvz = track_vxvyvz
        self._coord_splines = coord_splines
        self._bin_counts = counts
        self._bin_means = means
        self._bin_covs = covs

        # Interpolating splines on the public fine-grid track (for evaluation)
        self._cart_splines = [
            interpolate.InterpolatedUnivariateSpline(
                self._tp_grid, track_xyz[:, 0], k=3
            ),
            interpolate.InterpolatedUnivariateSpline(
                self._tp_grid, track_xyz[:, 1], k=3
            ),
            interpolate.InterpolatedUnivariateSpline(
                self._tp_grid, track_xyz[:, 2], k=3
            ),
            interpolate.InterpolatedUnivariateSpline(
                self._tp_grid, track_vxvyvz[:, 0], k=3
            ),
            interpolate.InterpolatedUnivariateSpline(
                self._tp_grid, track_vxvyvz[:, 1], k=3
            ),
            interpolate.InterpolatedUnivariateSpline(
                self._tp_grid, track_vxvyvz[:, 2], k=3
            ),
        ]

    # -----------------------------------------------------------------
    # Assignment helpers
    # -----------------------------------------------------------------
    def _assign_closest_on_track(self, xv_particles):
        """Reassign each particle's tp to the closest point on the current
        track in galactocentric Cartesian position."""
        R, vR, vT, z, vz, phi = xv_particles
        x_p, y_p, z_p = coords.cyl_to_rect(R, phi, z)
        # (ninterp, 3) track positions; (N, 3) particle positions
        dx = x_p[:, None] - self._track_xyz[None, :, 0]
        dy = y_p[:, None] - self._track_xyz[None, :, 1]
        dz = z_p[:, None] - self._track_xyz[None, :, 2]
        d2 = dx * dx + dy * dy + dz * dz
        idx = numpy.argmin(d2, axis=1)
        return self._tp_grid[idx]

    # -----------------------------------------------------------------
    # Public evaluation
    # -----------------------------------------------------------------
    def tp_grid(self):
        """Return the fine tp grid on which the track is stored."""
        return self._tp_grid.copy()

    def _eval_cart(self, tp):
        tp = numpy.atleast_1d(tp)
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
        'vlos')."""
        if not self._physical:
            return val
        if kind == "length":
            return val * self._ro * (units.kpc if _APY_UNITS else 1)
        if kind == "velocity" or kind == "vlos":
            return val * self._vo * (units.km / units.s if _APY_UNITS else 1)
        if kind == "angle":
            return val * (units.rad if _APY_UNITS else 1)
        if kind == "degree":
            return val * (units.deg if _APY_UNITS else 1)
        # kind == "pm"
        return val * (units.mas / units.yr if _APY_UNITS else 1)

    def _cart_eval(self, idx, tp):
        val = self._cart_splines[idx](numpy.atleast_1d(tp))
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
        tp = numpy.atleast_1d(tp)
        xyzvxyz = self._eval_cart(tp)  # (6, len) galactocentric Cartesian
        Xgc, Ygc, Zgc = xyzvxyz[0], xyzvxyz[1], xyzvxyz[2]
        vXgc, vYgc, vZgc = xyzvxyz[3], xyzvxyz[4], xyzvxyz[5]
        # Convert galactocentric to heliocentric (in internal units scaled by ro/vo)
        # Positions: kpc
        Xgc_kpc = Xgc * self._ro
        Ygc_kpc = Ygc * self._ro
        Zgc_kpc = Zgc * self._ro
        zo_kpc = self._zo if self._zo is not None else 0.0
        xyz_helio = coords.galcenrect_to_XYZ(
            Xgc_kpc, Ygc_kpc, Zgc_kpc, Xsun=self._ro, Zsun=zo_kpc
        )
        X = xyz_helio[..., 0]
        Y = xyz_helio[..., 1]
        Z = xyz_helio[..., 2]
        # Velocities: km/s
        vXgc_kms = vXgc * self._vo
        vYgc_kms = vYgc * self._vo
        vZgc_kms = vZgc * self._vo
        vsun = self._get_vsun_kms()
        vxyz_helio = coords.galcenrect_to_vxvyvz(
            vXgc_kms, vYgc_kms, vZgc_kms, vsun=vsun, Xsun=self._ro, Zsun=zo_kpc
        )
        vX = vxyz_helio[..., 0]
        vY = vxyz_helio[..., 1]
        vZ = vxyz_helio[..., 2]
        return X, Y, Z, vX, vY, vZ

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
        val = self._maybe_scalar(tp, lbd[:, 2])
        if self._physical:
            return val * (units.kpc if _APY_UNITS else 1)
        return val

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
        return None

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
            if d1 in cart_idx and d2 in cart_idx:
                i1 = cart_idx[d1]
                i2 = cart_idx[d2]
                cov = self.cov(tp)  # (n_eval, 6, 6)
                s1 = numpy.sqrt(numpy.maximum(cov[:, i1, i1], 0.0))
                s2 = numpy.sqrt(numpy.maximum(cov[:, i2, i2], 0.0))
                # Apply physical scaling if active
                if self._physical and self._roSet and d1 in ("x", "y", "z"):
                    s1 = s1 * self._ro
                if self._physical and self._roSet and d2 in ("x", "y", "z"):
                    s2 = s2 * self._ro
                if self._physical and self._voSet and d1 in ("vx", "vy", "vz"):
                    s1 = s1 * self._vo
                if self._physical and self._voSet and d2 in ("vx", "vy", "vz"):
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
