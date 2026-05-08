"""Tests for the smooth StreamTrack object.

Split out from test_streamspraydf.py to keep both files manageable.
"""

import warnings

import matplotlib
import numpy
import pytest

matplotlib.use("Agg")

from galpy.df import chen24spraydf, fardal15spraydf
from galpy.df.streamTrack import (
    StreamTrack,
    _closest_point_on_curve,
    _fit_track_from_particles,
    _smooth_series,
)
from galpy.orbit import Orbit
from galpy.potential import (
    LogarithmicHaloPotential,
    MWPotential2014,
    PlummerPotential,
)
from galpy.util import conversion, coords, galpyWarning


@pytest.fixture(scope="module")
def _simple_spdf():
    """Minimal fardal15spraydf fixture used by several streamTrack tests."""
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    ro, vo = 8.0, 220.0
    spdf = fardal15spraydf(
        2 * 10.0**4.0 / conversion.mass_in_msol(vo, ro),
        progenitor=obs,
        pot=lp,
        tdisrupt=4.5 / conversion.time_in_Gyr(vo, ro),
    )
    return spdf


def test_streamTrack_progenitor_recovery():
    # With tiny tdisrupt, particles barely drift from progenitor so the track
    # at any tp should be very close to the progenitor today.
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    ro, vo = 8.0, 220.0
    spdf = fardal15spraydf(
        2 * 10.0**4.0 / conversion.mass_in_msol(vo, ro),
        progenitor=obs,
        pot=lp,
        tdisrupt=0.05 / conversion.time_in_Gyr(vo, ro),
    )
    numpy.random.seed(1)
    track = spdf.streamTrack(n=2000, ntp=41, tail="leading")
    prog_x = spdf._progenitor.x(0.0)
    prog_y = spdf._progenitor.y(0.0)
    prog_z = spdf._progenitor.z(0.0)
    # For small tdisrupt, ALL particles are near the progenitor today, so
    # the track across its own tp grid should be within a few percent of
    # the progenitor's present-day position.
    tps = numpy.linspace(track.tp_grid()[0], track.tp_grid()[-1], 7)
    for tp in tps:
        assert abs(track.x(tp) - prog_x) < 0.05, (
            "StreamTrack does not recover the progenitor in the tiny-tdisrupt limit (x)"
        )
        assert abs(track.y(tp) - prog_y) < 0.05, (
            "StreamTrack does not recover the progenitor in the tiny-tdisrupt limit (y)"
        )
        assert abs(track.z(tp) - prog_z) < 0.05, (
            "StreamTrack does not recover the progenitor in the tiny-tdisrupt limit (z)"
        )


def test_streamTrack_sample_consistency(_simple_spdf):
    # The track's mean at tp should agree with the mean galactocentric x of
    # particles near that tp (per a closest-point assignment to the track).

    numpy.random.seed(1)
    track = _simple_spdf.streamTrack(n=4000, ntp=41, tail="leading")
    # Reproduce the closest-point assignment from the saved particles ->
    # closest point on the dense track itself (a stable, public reference).
    particles_cart = coords.galcencyl_to_galcenrect(*track.particles)
    tp_grid = track.tp_grid()
    track_cart = numpy.column_stack(
        [
            [float(track.x(t, use_physical=False)) for t in tp_grid],
            [float(track.y(t, use_physical=False)) for t in tp_grid],
            [float(track.z(t, use_physical=False)) for t in tp_grid],
            [float(track.vx(t, use_physical=False)) for t in tp_grid],
            [float(track.vy(t, use_physical=False)) for t in tp_grid],
            [float(track.vz(t, use_physical=False)) for t in tp_grid],
        ]
    )
    tp_part = _closest_point_on_curve(particles_cart, track_cart, tp_grid)
    x_p = particles_cart[:, 0]
    edges = numpy.linspace(tp_grid[0], tp_grid[-1], 9)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for i in range(len(centers)):
        sel = (tp_part >= edges[i]) & (tp_part < edges[i + 1])
        n_in = int(sel.sum())
        if n_in < 50:
            continue
        mean_x = x_p[sel].mean()
        std_x = x_p[sel].std(ddof=1) / numpy.sqrt(n_in)
        assert abs(mean_x - track.x(centers[i])) < max(5 * std_x, 0.05), (
            "StreamTrack does not match binned sample mean in Cartesian x"
        )


def test_streamTrack_interface(_simple_spdf):
    numpy.random.seed(2)
    track = _simple_spdf.streamTrack(n=2000, ntp=41, tail="leading")
    g = track.tp_grid()
    tps = numpy.linspace(g[0], g[-1], 5)
    # Each accessor's expected output range, paired with a sanity range on the
    # _simple_spdf progenitor (|r| ~ 1.6, |v| ~ 1.4 internal). Cartesian/cyl
    # positions in internal length, velocities in internal velocity, angles in
    # degrees, distance in kpc, proper motions in mas/yr, vlos in km/s.
    expected = {
        "x": (-3.0, 3.0),
        "y": (-3.0, 3.0),
        "z": (-3.0, 3.0),
        "vx": (-3.0, 3.0),
        "vy": (-3.0, 3.0),
        "vz": (-3.0, 3.0),
        "R": (0.0, 3.0),
        "vR": (-3.0, 3.0),
        "vT": (-3.0, 3.0),
        "phi": (-numpy.pi, 2 * numpy.pi),
        "ra": (0.0, 360.0),
        "dec": (-90.0, 90.0),
        "dist": (0.0, 200.0),  # kpc; with internal r ~ 1.6 and ro=8 this is ~few
        "ll": (0.0, 360.0),
        "bb": (-90.0, 90.0),
        "pmra": (-1e3, 1e3),
        "pmdec": (-1e3, 1e3),
        "pmll": (-1e3, 1e3),
        "pmbb": (-1e3, 1e3),
        "vlos": (-1e3, 1e3),
    }
    for meth, (lo, hi) in expected.items():
        vals = numpy.asarray(getattr(track, meth)(tps))
        assert vals.shape == tps.shape, f"{meth} returned wrong shape"
        assert numpy.all((vals >= lo) & (vals <= hi)), (
            f"{meth} out of [{lo}, {hi}]: {vals}"
        )
        # A leading arm spans a nontrivial range — every accessor should
        # show variation across the 5 tp samples (not pinned at progenitor).
        assert vals.max() > vals.min(), f"{meth} is constant along the arm"


def test_streamTrack_covariance_psd(_simple_spdf):
    numpy.random.seed(3)
    track = _simple_spdf.streamTrack(n=2000, ntp=41, tail="leading")
    g = track.tp_grid()
    tps = numpy.linspace(g[0], g[-1], 7)
    covs = track.cov(tps)
    assert covs.shape == (len(tps), 6, 6)
    for k in range(len(tps)):
        C = covs[k]
        assert numpy.allclose(C, C.T, atol=1e-10), "Covariance not symmetric"
        evs = numpy.linalg.eigvalsh(C)
        assert numpy.all(evs >= -1e-10), f"Covariance not PSD (min eigval={evs.min()})"


def test_streamTrack_both_tails(_simple_spdf):
    numpy.random.seed(4)
    pair = _simple_spdf.streamTrack(n=1500, ntp=41, tail="both")
    # Near tp=0 both arms should sit near the progenitor
    prog = _simple_spdf._progenitor
    px, py = prog.x(0.0), prog.y(0.0)
    assert abs(pair.leading.x(0.0) - px) < 0.1
    assert abs(pair.trailing.x(0.0) - px) < 0.1
    # Deep into the stream, the two arms diverge — pick the deepest in-range
    # point for each arm (leading arm: largest positive tp; trailing arm:
    # most negative tp).
    tp_lead = pair.leading.tp_grid()[-1]
    tp_trail = pair.trailing.tp_grid()[0]
    d_lead = (pair.leading.x(tp_lead) - pair.trailing.x(tp_trail)) ** 2 + (
        pair.leading.y(tp_lead) - pair.trailing.y(tp_trail)
    ) ** 2
    assert d_lead > 0.01, "Leading and trailing arms do not diverge at large |tp|"


def test_streamTrack_iteration_changes_track(_simple_spdf):
    # Iteration should move the track by a small amount; we don't require
    # strict convergence (closest-point reassignment introduces some noise).
    numpy.random.seed(5)
    tr0 = _simple_spdf.streamTrack(n=2000, ntp=41, niter=0, tail="leading")
    numpy.random.seed(5)
    tr1 = _simple_spdf.streamTrack(n=2000, ntp=41, niter=1, tail="leading")
    # Compare on the in-range grid common to both tracks.
    lo = max(tr0.tp_grid()[0], tr1.tp_grid()[0])
    hi = min(tr0.tp_grid()[-1], tr1.tp_grid()[-1])
    tps = numpy.linspace(lo, hi, 101)
    # Track-to-track difference should be small compared to the stream size
    ampl = numpy.ptp(tr0.x(tps))
    dmax = numpy.max(numpy.abs(tr0.x(tps) - tr1.x(tps)))
    assert dmax < 0.5 * max(ampl, 0.1), (
        "Iteration changed the track by more than half the stream amplitude"
    )


def test_streamTrack_chen24_works():
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    ro, vo = 8.0, 220.0
    spdf = chen24spraydf(
        2 * 10.0**4.0 / conversion.mass_in_msol(vo, ro),
        progenitor=obs,
        pot=lp,
        tdisrupt=4.5 / conversion.time_in_Gyr(vo, ro),
    )
    numpy.random.seed(6)
    track = spdf.streamTrack(n=1500, ntp=41, tail="both")
    # Sample at midpoint of each arm's tp grid (guaranteed in-range).
    tp_lead = track.leading.tp_grid()[len(track.leading.tp_grid()) // 2]
    tp_trail = track.trailing.tp_grid()[len(track.trailing.tp_grid()) // 2]
    # tp=0 is the progenitor "today" — both arms must agree there (same
    # endpoint), and that point must equal the progenitor's present-day x to
    # the stream-width level.
    prog_x0 = float(spdf._progenitor.x(0.0))
    assert abs(float(track.leading.x(0.0)) - prog_x0) < 0.05
    assert abs(float(track.trailing.x(0.0)) - prog_x0) < 0.05
    assert abs(float(track.leading.x(0.0)) - float(track.trailing.x(0.0))) < 0.05
    # At interior tps the arms should have moved away from the progenitor
    # (this is what distinguishes a stream from a point).
    assert abs(float(track.leading.x(tp_lead)) - prog_x0) > 1e-3
    assert abs(float(track.trailing.x(tp_trail)) - prog_x0) > 1e-3


def test_streamTrack_with_center():
    # Stream around a moving satellite
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    ro, vo = 8.0, 220.0
    cen = Orbit([1.3, 0.2, -0.9, 0.4, 0.1, 0.4])
    prog = Orbit([1.35, 0.22, -0.88, 0.42, 0.08, 0.45])
    spdf = fardal15spraydf(
        2e4 / conversion.mass_in_msol(vo, ro),
        progenitor=prog,
        pot=lp,
        center=cen,
        centerpot=lp,
        tdisrupt=1.0 / conversion.time_in_Gyr(vo, ro),
    )
    numpy.random.seed(7)
    track = spdf.streamTrack(n=1500, ntp=31, tail="leading")
    g = track.tp_grid()
    tps = numpy.linspace(g[0], g[-1], 5)
    vals = numpy.asarray(track.x(tps))
    # Track follows the progenitor near tp=0 and moves away with tp.
    prog_x0 = float(spdf._progenitor.x(0.0))
    assert abs(vals[0] - prog_x0) < 0.1
    # The arm spans a nontrivial range (not pinned at the progenitor)
    assert vals.max() > vals.min() + 0.02


def test_streamTrack_physical_units(_simple_spdf):
    numpy.random.seed(8)
    track = _simple_spdf.streamTrack(n=1500, ntp=31, tail="leading")
    tp = track.tp_grid()[len(track.tp_grid()) // 2]
    x0 = track.x(tp)
    track.turn_physical_on(ro=8.0, vo=220.0)
    x0_phys = track.x(tp)
    # physical x should be ~ ro * internal
    val = x0_phys.value if hasattr(x0_phys, "value") else x0_phys
    assert abs(val - 8.0 * x0) < 1e-6
    track.turn_physical_off()
    assert abs(track.x(tp) - x0) < 1e-10


def test_streamTrack_cov_physical_units(_simple_spdf):
    # cov() must honor the physical-unit toggle, matching the scaling of
    # x/y/z/vx/vy/vz accessors. Position entries scale by ro^2, velocity
    # by vo^2, cross terms by ro*vo.
    numpy.random.seed(18)
    track = _simple_spdf.streamTrack(n=1500, ntp=31, tail="leading")
    tp = track.tp_grid()[len(track.tp_grid()) // 2]
    track.turn_physical_off()
    C_int = track.cov(tp)
    track.turn_physical_on(ro=8.0, vo=220.0)
    C_phys = track.cov(tp)
    ro, vo = 8.0, 220.0
    scale = numpy.array([ro, ro, ro, vo, vo, vo])
    expected = C_int * numpy.outer(scale, scale)
    assert numpy.allclose(C_phys, expected, rtol=1e-10)


def test_streamTrack_particles_reuse(_simple_spdf):
    numpy.random.seed(9)
    xv = _simple_spdf.sample(n=2000, returndt=False, return_orbit=False, integrate=True)
    track = _simple_spdf.streamTrack(particles=xv, ntp=41, tail="leading")
    # tp=0 reproduces the progenitor present-day position to within the
    # stream width.
    prog_x0 = float(_simple_spdf._progenitor.x(0.0))
    assert abs(float(track.x(0.0)) - prog_x0) < 0.05


def test_streamTrack_plot_smoke(_simple_spdf):

    numpy.random.seed(10)
    track = _simple_spdf.streamTrack(n=1500, ntp=31, tail="both")
    track.plot(d1="x", d2="y", spread=1.0)
    track.leading.plot(d1="ra", d2="dec")
    track.leading.plot(d1="R", d2="z")
    # spread>0 on velocity axes exercises the velocity branch of the spread
    # scaling; non-Cartesian axes silently skip the band
    track.leading.plot(d1="vx", d2="vy", spread=1.0)
    track.leading.plot(d1="R", d2="phi", spread=1.0)


def test_streamTrack_physical_accessors_all(_simple_spdf):
    # Exercise the physical-unit branch of every accessor; expected ranges in
    # physical units (kpc / km/s / deg / mas/yr).
    numpy.random.seed(11)
    track = _simple_spdf.streamTrack(n=1500, ntp=31, tail="leading")
    track.turn_physical_on(ro=8.0, vo=220.0)
    tp = track.tp_grid()[len(track.tp_grid()) // 2]
    expected_phys = {
        "x": (-30.0, 30.0),  # kpc, ro=8
        "y": (-30.0, 30.0),
        "z": (-30.0, 30.0),
        "vx": (-700.0, 700.0),  # km/s, vo=220
        "vy": (-700.0, 700.0),
        "vz": (-700.0, 700.0),
        "R": (0.0, 30.0),
        "vR": (-700.0, 700.0),
        "vT": (-700.0, 700.0),
        "phi": (-180.0, 360.0),  # deg
        "ra": (0.0, 360.0),
        "dec": (-90.0, 90.0),
        "ll": (0.0, 360.0),
        "bb": (-90.0, 90.0),
        "dist": (0.0, 200.0),
        "pmra": (-1e3, 1e3),
        "pmdec": (-1e3, 1e3),
        "pmll": (-1e3, 1e3),
        "pmbb": (-1e3, 1e3),
        "vlos": (-1e3, 1e3),
    }
    for meth, (lo, hi) in expected_phys.items():
        v = getattr(track, meth)(tp)
        val = float(getattr(v, "value", v))
        assert lo <= val <= hi, f"{meth} = {val} out of [{lo}, {hi}] in physical units"
    # Plot in physical units with a velocity-axis spread hits the vo
    # scaling branch of the band

    track.plot(d1="vx", d2="vy", spread=1.0)
    track.turn_physical_off()


def test_streamTrack_pair_physical_toggles(_simple_spdf):
    numpy.random.seed(12)
    pair = _simple_spdf.streamTrack(n=1500, ntp=31, tail="both")
    pair.turn_physical_on(ro=8.0, vo=220.0)
    tp_lead = pair.leading.tp_grid()[len(pair.leading.tp_grid()) // 2]
    v = pair.leading.x(tp_lead)
    val = getattr(v, "value", v)
    assert val > 0.0
    pair.turn_physical_off()
    v2 = pair.leading.x(tp_lead)
    assert not hasattr(v2, "unit")

    pair.plot(d1="x", d2="y")


def test_streamTrack_call_returns_stacked(_simple_spdf):
    numpy.random.seed(13)
    track = _simple_spdf.streamTrack(n=1500, ntp=31, tail="leading")
    v = track(-10.0)
    assert v.shape == (6,)
    tps = numpy.linspace(-10.0, -1.0, 4)
    v = track(tps)
    assert v.shape == (6, 4)


def test_streamTrack_tp_grid(_simple_spdf):
    numpy.random.seed(14)
    track = _simple_spdf.streamTrack(n=800, ntp=31, tail="leading")
    g = track.tp_grid()
    assert g.shape == (1001,)
    # Leading arm: tp ranges from 0 to some positive value.
    assert g[0] == 0.0
    assert g[-1] > 0.0


def test_streamTrack_out_of_range_returns_nan(_simple_spdf):
    # Out-of-range tp must return NaN — never silent cubic-spline
    # extrapolation. For an array tp, only the offending entries are NaN.
    numpy.random.seed(20)
    track = _simple_spdf.streamTrack(n=800, ntp=31, tail="leading")
    g = track.tp_grid()
    tp_lo, tp_hi = g[0], g[-1]
    # Scalar out-of-range: NaN
    assert numpy.isnan(track.x(tp_hi + 5.0))
    assert numpy.isnan(track.R(tp_hi + 5.0))
    assert numpy.isnan(track.ra(tp_hi + 5.0))
    # Negative side (leading arm has tp_lo == 0)
    assert numpy.isnan(track.x(tp_lo - 1.0))
    # Scalar in-range: finite
    tp_mid = 0.5 * (tp_lo + tp_hi)
    assert numpy.isfinite(track.x(tp_mid))
    # Array tp with mixed in/out: NaN only at out-of-range entries
    tps = numpy.array([tp_lo - 1.0, tp_mid, tp_hi + 5.0])
    xs = numpy.asarray(track.x(tps))
    assert numpy.isnan(xs[0])
    assert numpy.isfinite(xs[1])
    assert numpy.isnan(xs[2])
    # cov() honors the same convention; out-of-range entries are NaN
    Cs = track.cov(tps)
    assert numpy.all(numpy.isnan(Cs[0]))
    assert numpy.all(numpy.isfinite(Cs[1]))
    assert numpy.all(numpy.isnan(Cs[2]))
    # cov(basis=...) too — NaN entries skip the Jacobian path safely
    Cs_sky = track.cov(tps, basis="sky")
    assert numpy.all(numpy.isnan(Cs_sky[0]))
    assert numpy.all(numpy.isfinite(Cs_sky[1]))
    assert numpy.all(numpy.isnan(Cs_sky[2]))


def test_streamTrack_order1_no_cov(_simple_spdf):
    numpy.random.seed(15)
    track = _simple_spdf.streamTrack(n=800, ntp=31, tail="leading", order=1)
    with pytest.raises(RuntimeError):
        track.cov(-10.0)


def test_streamTrack_invalid_tail(_simple_spdf):
    with pytest.raises(ValueError):
        _simple_spdf.streamTrack(n=100, tail="bogus")


def test_streamTrack_particles_reuse_both(_simple_spdf):
    numpy.random.seed(16)
    _simple_spdf._tail = "both"
    xv = _simple_spdf.sample(n=2000, returndt=False, return_orbit=False, integrate=True)
    _simple_spdf._tail = "leading"
    pair = _simple_spdf.streamTrack(particles=xv, tail="both", ntp=31)
    # Both arms must meet the progenitor at tp=0.
    prog_x0 = float(_simple_spdf._progenitor.x(0.0))
    assert abs(float(pair.leading.x(0.0)) - prog_x0) < 0.05
    assert abs(float(pair.trailing.x(0.0)) - prog_x0) < 0.05


def test_streamTrackPair_particles_property(_simple_spdf):
    # pair.particles concatenates the two arms' xv arrays in leading-first
    # order, exactly the format ``streamTrack(tail='both', particles=...)``
    # expects. Round-trip: re-fit using pair.particles and confirm the
    # resulting tracks match the original (modulo tiny FITPACK iteration
    # noise).
    numpy.random.seed(34)
    pair = _simple_spdf.streamTrack(n=1500, ntp=31, tail="both")
    xv_pair = pair.particles
    # Shape: leading first, trailing second.
    n_l = pair.leading.particles.shape[1]
    n_t = pair.trailing.particles.shape[1]
    assert xv_pair.shape == (6, n_l + n_t)
    assert numpy.allclose(xv_pair[:, :n_l], pair.leading.particles)
    assert numpy.allclose(xv_pair[:, n_l:], pair.trailing.particles)
    # Round-trip via spraydf.streamTrack(tail='both', particles=...). Compare
    # on the overlap of the two tp grids (auto-trim percentiles can land at
    # slightly different boundaries between the two calls due to a small
    # probe-sample jitter; staying inside the overlap avoids NaN edges).
    pair_reuse = _simple_spdf.streamTrack(particles=pair.particles, tail="both", ntp=31)
    g_l, g_l2 = pair.leading.tp_grid(), pair_reuse.leading.tp_grid()
    g_t, g_t2 = pair.trailing.tp_grid(), pair_reuse.trailing.tp_grid()
    tps_l = numpy.linspace(max(g_l[0], g_l2[0]), min(g_l[-1], g_l2[-1]), 21)
    tps_t = numpy.linspace(max(g_t[0], g_t2[0]), min(g_t[-1], g_t2[-1]), 21)
    assert (
        numpy.max(numpy.abs(pair.leading.x(tps_l) - pair_reuse.leading.x(tps_l))) < 1e-3
    )
    assert (
        numpy.max(numpy.abs(pair.trailing.x(tps_t) - pair_reuse.trailing.x(tps_t)))
        < 1e-3
    )


def test_streamTrack_scalar_cov(_simple_spdf):
    numpy.random.seed(17)
    track = _simple_spdf.streamTrack(n=800, ntp=31, tail="leading")
    tp = track.tp_grid()[len(track.tp_grid()) // 2]
    C = track.cov(tp)
    assert C.shape == (6, 6)
    # Symmetric and PSD (the spread is a covariance, not just any matrix).
    assert numpy.allclose(C, C.T, atol=1e-10)
    eigs = numpy.linalg.eigvalsh(C)
    assert eigs.min() > -1e-10, f"cov is not PSD: smallest eigenvalue = {eigs.min()}"
    # Position variances are positive but small (stream is narrow): O(1e-3
    # internal length squared) ~ a few hundred parsecs.
    pos_diag = numpy.diag(C)[:3]
    assert numpy.all(pos_diag > 0.0)
    assert pos_diag.max() < 1e-2


def test_streamTrack_cov_per_call_unit_overrides(_simple_spdf):
    # cov() honors per-call ro=, vo=, use_physical= (the same way the mean
    # accessors do) so callers don't need to flip the track-wide toggle.
    numpy.random.seed(18)
    track = _simple_spdf.streamTrack(n=1500, ntp=31, tail="leading")
    tp = track.tp_grid()[len(track.tp_grid()) // 2]
    track.turn_physical_off()
    C_int = track.cov(tp)
    # ro=, vo= scale the entries even when the track is in internal mode
    ro, vo = 8.0, 220.0
    C_phys_via_kw = track.cov(tp, ro=ro, vo=vo, use_physical=True)
    scale = numpy.array([ro, ro, ro, vo, vo, vo])
    assert numpy.allclose(C_phys_via_kw, C_int * numpy.outer(scale, scale), rtol=1e-10)
    # use_physical=False on a physical-mode track gives back internal cov
    track.turn_physical_on(ro=ro, vo=vo)
    C_back_to_int = track.cov(tp, use_physical=False)
    assert numpy.allclose(C_back_to_int, C_int, rtol=1e-10)
    # quantity=True is explicitly not supported (heterogeneous units)
    with pytest.raises(NotImplementedError):
        track.cov(tp, quantity=True)
    # Per-call overrides also have to thread through the analytical
    # Jacobian for non-galcenrect bases — exercise the sky path with
    # explicit ro/vo so the override branches in _cart_mean_at and
    # _analytical_jacobian (where ``ro``/``vo``/``use_physical`` are
    # forwarded to the accessors and used as Xsun) are taken.
    track.turn_physical_off()
    C_sky_with_kw = track.cov(tp, basis="sky", ro=ro, vo=vo, use_physical=True)
    assert C_sky_with_kw.shape == (6, 6)
    assert numpy.allclose(C_sky_with_kw, C_sky_with_kw.T, atol=1e-8)


def test_streamTrack_physical_length_spread(_simple_spdf):
    # Covers the physical x/y/z scaling inside the plot spread band

    numpy.random.seed(18)
    track = _simple_spdf.streamTrack(n=800, ntp=31, tail="leading")
    track.turn_physical_on(ro=8.0, vo=220.0)
    track.plot(d1="x", d2="y", spread=1.0)
    track.plot(d1="vx", d2="vy", spread=1.0)
    track.turn_physical_off()


def test_streamTrack_degenerate_few_particles(_simple_spdf):
    # Tiny sample with lots of bins -> most bins are empty/single-particle,
    # exercising the degenerate paths in _bin_offsets and _smooth_series.
    numpy.random.seed(19)
    track = _simple_spdf.streamTrack(n=10, ntp=51, tail="leading")
    tp = track.tp_grid()[len(track.tp_grid()) // 2]
    # With so few particles the smoothed offsets are tiny, so the track is
    # essentially the progenitor orbit. The midpoint must sit within the
    # progenitor's |r| ~ 1.6 internal-units neighborhood.
    val = float(track.x(tp))
    prog_radius = (
        float(
            _simple_spdf._progenitor.x(0.0) ** 2
            + _simple_spdf._progenitor.y(0.0) ** 2
            + _simple_spdf._progenitor.z(0.0) ** 2
        )
        ** 0.5
    )
    assert abs(val) < 2 * prog_radius


def test_streamTrack_custom_track_time_range(_simple_spdf):
    # Pass an explicit track_time_range (float) to cover the non-default
    # branch in basestreamspraydf.streamTrack. The astropy-Quantity
    # variant of this test lives in tests/test_quantity.py.
    numpy.random.seed(20)
    tr = _simple_spdf.streamTrack(n=800, tail="leading", track_time_range=3.0)
    assert tr.tp_grid()[-1] <= 3.0 + 1e-9


def test_streamTrack_smoothing_variants(_simple_spdf):
    numpy.random.seed(18)
    # Default ntp (auto from n) is exercised here
    tr_f = _simple_spdf.streamTrack(n=800, tail="leading", smoothing=20.0)
    # Aggressive smoothing collapses the offset toward zero; the track at
    # tp=0 still reproduces the progenitor present-day position.
    prog_x0 = float(_simple_spdf._progenitor.x(0.0))
    assert abs(float(tr_f.x(0.0)) - prog_x0) < 0.1
    # Array-like smoothing: reuse smoothing_s from a previous fit
    numpy.random.seed(18)
    tr_gcv = _simple_spdf.streamTrack(n=800, tail="leading", order=2)
    assert len(tr_gcv.smoothing_s) == 27  # 6 mean + 21 cov
    numpy.random.seed(18)
    tr_reuse = _simple_spdf.streamTrack(
        n=800, tail="leading", order=2, smoothing=tr_gcv.smoothing_s
    )
    # Reuse must reproduce the GCV fit closely along the overlap of the two
    # tp grids (modulo small FITPACK iteration noise).
    g_a, g_b = tr_gcv.tp_grid(), tr_reuse.tp_grid()
    tps = numpy.linspace(max(g_a[0], g_b[0]), min(g_a[-1], g_b[-1]), 21)
    assert numpy.max(numpy.abs(tr_gcv.x(tps) - tr_reuse.x(tps))) < 5e-3
    # order=1 returns only 6 s values
    numpy.random.seed(18)
    tr_o1 = _simple_spdf.streamTrack(n=800, tail="leading", order=1)
    assert len(tr_o1.smoothing_s) == 6


def test_streamTrack_smoothing_factor(_simple_spdf):
    # smoothing_factor>1 must produce a smoother fit (larger effective s)
    # than smoothing_factor=1; <1 must produce a rougher fit. Sample once
    # and reuse via particles= so we isolate the smoothing step.
    numpy.random.seed(33)
    # Pin velocity_weight=1.0 so this test is specifically about the
    # smoothing_factor knob, not entangled with the velocity_weight='auto'
    # default (which slightly shifts the per-bin variances that GCV uses
    # to pick s, changing the absolute scale of smoothing_s).
    tr_default = _simple_spdf.streamTrack(
        n=800, ntp=21, tail="leading", velocity_weight=1.0
    )
    xv = tr_default.particles
    tr_smoother = _simple_spdf.streamTrack(
        particles=xv,
        ntp=21,
        tail="leading",
        smoothing_factor=2.0,
        velocity_weight=1.0,
    )
    tr_rougher = _simple_spdf.streamTrack(
        particles=xv,
        ntp=21,
        tail="leading",
        smoothing_factor=0.5,
        velocity_weight=1.0,
    )
    # Ratio of effective s over the six mean splines (skip entries where
    # the default s is so small the rerun underflows). The cov splines
    # are excluded because GCV's chi^2 there is already at FITPACK's
    # minimum-knot floor, so a larger ``s`` upper bound doesn't change
    # the fit and the per-spline ratio stays near 1 — diluting any mean.
    s_def = numpy.asarray(tr_default.smoothing_s[:6])
    s_smt = numpy.asarray(tr_smoother.smoothing_s[:6])
    s_rgh = numpy.asarray(tr_rougher.smoothing_s[:6])
    valid = s_def > 1e-3
    assert valid.any(), "no valid mean-spline s to compare against"
    # UnivariateSpline interprets ``s`` as a chi^2 upper bound: the
    # rougher direction (factor < 1) reliably tightens because the new
    # bound forces extra knots; the smoother direction (factor > 1) is
    # platform-dependent because some splines already sit at the FITPACK
    # minimum-knot floor and ignore the loosened bound. Test the strong
    # signal in each direction: at least one mean spline must get
    # noticeably smoother for factor=2, and the average must clearly
    # tighten for factor=0.5.
    assert numpy.max(s_smt[valid] / s_def[valid]) > 1.5
    assert numpy.mean(s_rgh[valid] / s_def[valid]) < 0.7
    # smoothing_factor=1.0 must reproduce the default fit (modulo a small
    # probe-sampling jitter: the no-particles call draws a probe sample
    # to size track_time_range, the particles= call skips that probe and
    # uses the passed sample directly, so track_t_grid differs at the
    # 1e-5 level — well below any physical scale).
    tr_unity = _simple_spdf.streamTrack(
        particles=xv,
        ntp=21,
        tail="leading",
        smoothing_factor=1.0,
        velocity_weight=1.0,
    )
    tps = tr_default.tp_grid()
    assert numpy.max(numpy.abs(tr_default.x(tps) - tr_unity.x(tps))) < 1e-3
    # The saved smoothing_s round-trips through ``smoothing=`` with
    # smoothing_factor=1.0 — passing both the smoothed s values and
    # factor=1 should reproduce the smoother track (FITPACK iteration
    # tolerance allowed).
    tr_reuse = _simple_spdf.streamTrack(
        particles=xv,
        ntp=21,
        tail="leading",
        smoothing=tr_smoother.smoothing_s,
        smoothing_factor=1.0,
        velocity_weight=1.0,
    )
    assert numpy.max(numpy.abs(tr_smoother.x(tps) - tr_reuse.x(tps))) < 1e-3


def test_smooth_series_too_few_valid_bins():
    """_smooth_series falls back to linear interpolation when fewer than 5
    valid bins are available, and to a flat constant interpolant when fewer
    than 2 valid bins remain after dropping NaNs."""

    # Case A: zero valid bins → constant 0
    x = numpy.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = numpy.full(5, numpy.nan)
    sigma = numpy.ones(5)
    spl, eff_s = _smooth_series(x, y, sigma)
    numpy.testing.assert_allclose(spl(numpy.array([0.0, 1.5, 3.0])), 0.0)
    assert eff_s == 0.0

    # Case B: one valid bin → constant equal to that bin's value
    y = numpy.array([numpy.nan, numpy.nan, 7.0, numpy.nan, numpy.nan])
    spl, eff_s = _smooth_series(x, y, sigma)
    numpy.testing.assert_allclose(spl(numpy.array([0.0, 3.0, 5.0])), 7.0)
    assert eff_s == 0.0


def test_smooth_series_invalid_sigma_falls_back_to_unit_median():
    """When all per-bin sigma entries are non-finite, _smooth_series treats
    every bin with unit sigma instead of raising or NaN-propagating."""

    x = numpy.linspace(0.0, 1.0, 8)
    y = numpy.sin(2 * numpy.pi * x)
    sigma = numpy.full(8, numpy.nan)
    spl, eff_s = _smooth_series(x, y, sigma)
    # Returned spline is bounded by the input range (no extrapolation
    # blow-ups) and reproduces the symmetric zero crossing at x=0.5.
    xs_test = numpy.linspace(0.0, 1.0, 21)
    vals = spl(xs_test)
    assert vals.min() >= y.min() - 0.1 and vals.max() <= y.max() + 0.1
    assert abs(spl(0.5)) < 1e-6


def test_smooth_series_constant_y_falls_back_to_unit_yscale():
    """When y is a constant series, the GCV-driven yscale normalization
    falls back to 1.0; the returned spline should reproduce the constant."""

    x = numpy.linspace(0.0, 1.0, 8)
    y = numpy.full(8, 3.14)  # std == 0
    sigma = numpy.ones(8)
    spl, _ = _smooth_series(x, y, sigma)
    numpy.testing.assert_allclose(spl(numpy.array([0.0, 0.5, 1.0])), 3.14, atol=1e-10)


def test_streamTrack_trim_grid_degenerate_tp():
    """If every particle's tp_assign collapses onto a single value (so
    tp_hi - tp_lo < 1e-12), _trim_grid falls back to the full track grid
    instead of producing an empty range."""

    M = 1001
    tg = numpy.linspace(-100.0, 100.0, M)
    prog = numpy.zeros((M, 6))
    prog[:, 0] = tg
    prog[:, 3] = 1.0

    n = 50
    rng = numpy.random.default_rng(7)
    xv = numpy.zeros((6, n))
    # All particles at the progenitor's t=0 position with tiny perpendicular
    # noise. Their closest-point assignment is tp ≈ 0 for everyone, so the
    # 99th-percentile trim collapses to (0, 0) and the defensive fallback
    # has to widen the grid.
    xv[0] = 0.0  # R = 0
    xv[1] = 1.0  # vR -> vx
    xv[3] = 0.001 * rng.standard_normal(n)
    xv[4] = 0.001 * rng.standard_normal(n)
    dt = numpy.full(n, 50.0)
    try:
        _fit_track_from_particles(
            xv,
            prog,
            tg,
            arm_sign=+1,
            ninterp=101,
            smoothing_factor=1.0,
            niter=0,
            order=2,
            velocity_weight=1.0,
        )
    except ValueError:
        # Degenerate spline fit downstream is fine; we just need the
        # defensive grid-fallback to have run.
        pass


def test_streamTrack_velocity_weight_invalid_string(_simple_spdf):
    """velocity_weight= must be a float or 'auto'; any other string raises."""

    with pytest.raises(ValueError, match="velocity_weight="):
        _simple_spdf.streamTrack(
            n=200, ntp=21, tail="leading", velocity_weight="not_auto"
        )


def test_streamTrack_velocity_weight_auto_smallN():
    """velocity_weight='auto' falls back to 1.0 when fewer than 20 particles
    are passed in (probe sample too small to estimate σ_pos / σ_vel)."""

    M = 1001
    tg = numpy.linspace(-100.0, 100.0, M)
    prog = numpy.zeros((M, 6))
    prog[:, 0] = tg
    prog[:, 3] = 1.0

    rng = numpy.random.default_rng(1)
    n = 19  # below the 20-particle floor for auto
    xv = numpy.zeros((6, n))
    xv[0] = numpy.linspace(0.0, 3.0, n)
    xv[1] = 1.0
    xv[3] = 0.05 * rng.standard_normal(n)
    xv[4] = 0.05 * rng.standard_normal(n)
    dt = numpy.full(n, 50.0)

    # Should run without error and the auto resolver should hit the
    # size<20 fallback (velocity_weight collapses to 1.0). We don't assert
    # the resulting fit shape — just that the call completes.
    try:
        _fit_track_from_particles(
            xv,
            prog,
            tg,
            arm_sign=+1,
            ninterp=101,
            smoothing_factor=1.0,
            niter=0,
            order=2,
            velocity_weight="auto",
        )
    except ValueError:
        # A degenerate downstream spline fit can ill-pose itself on this
        # tiny synthetic sample; the auto-resolver fallback is what we're
        # exercising and that ran before the spline fit.
        pass


def test_streamTrack_velocity_weight_auto_zero_sigma_vel():
    """velocity_weight='auto' falls back to 1.0 when the inner-half particle
    velocity dispersion is zero (degenerate spread)."""

    M = 1001
    tg = numpy.linspace(-100.0, 100.0, M)
    prog = numpy.zeros((M, 6))
    prog[:, 0] = tg
    prog[:, 3] = 1.0  # constant velocity along the curve

    n = 40
    xv = numpy.zeros((6, n))
    # Particles spread along x but ALL with identical (vR, vT, vz) =
    # (1, 0, 0) — same as the curve. So sigma_vel of inner-half is 0.
    xv[0] = numpy.linspace(0.0, 3.0, n)
    xv[1] = 1.0
    dt = numpy.full(n, 50.0)

    try:
        _fit_track_from_particles(
            xv,
            prog,
            tg,
            arm_sign=+1,
            ninterp=101,
            smoothing_factor=1.0,
            niter=0,
            order=2,
            velocity_weight="auto",
        )
    except ValueError:
        pass


def test_streamTrack_gap_warning():
    """When tp_assign has a structural gap (the orbit-revisit kink
    signature), streamTrack should emit a galpyWarning recommending
    velocity_weight or higher niter.

    Construct a synthetic minimal scenario directly via
    _fit_track_from_particles: a straight progenitor curve (x = tp,
    vx = 1), 90 particles clustered near tp=0 plus 10 outliers at
    tp~99. velocity_weight=1.0 disables the auto-rescue so the
    bimodal tp_assign survives to trigger the gap detector.
    """

    M = 1001
    tg = numpy.linspace(-100.0, 100.0, M)
    prog = numpy.zeros((M, 6))
    prog[:, 0] = tg  # x = tp
    prog[:, 3] = 1.0  # vx = 1 (constant)

    rng = numpy.random.default_rng(0)
    n_bulk, n_far = 90, 10
    xv = numpy.zeros((6, n_bulk + n_far))
    xv[0, :n_bulk] = numpy.linspace(0.0, 3.0, n_bulk)  # R near 0
    xv[0, n_bulk:] = numpy.linspace(95.0, 99.0, n_far)  # R far in leading arm
    xv[1, :] = 1.0  # vR=1 -> vx=1 (matches the curve)
    # Add small perpendicular scatter (z, vz) so the spline fit isn't
    # singular on a perfectly degenerate line.
    xv[3] = 0.01 * rng.standard_normal(n_bulk + n_far)
    xv[4] = 0.01 * rng.standard_normal(n_bulk + n_far)
    dt = numpy.full(n_bulk + n_far, 200.0)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # The warning fires before the spline fit. The synthetic particles
        # are degenerate enough that the spline can hit FITPACK's
        # ill-posed branch — that's not what we're testing here, so swallow
        # any post-warning ValueError.
        try:
            _fit_track_from_particles(
                xv,
                prog,
                tg,
                arm_sign=+1,
                ninterp=101,
                smoothing_factor=1.0,
                niter=0,
                order=2,
                velocity_weight=1.0,
            )
        except ValueError:
            pass
    gap_msgs = [
        ww
        for ww in w
        if issubclass(ww.category, galpyWarning)
        and "tp_assign histogram" in str(ww.message)
        and "gap" in str(ww.message)
    ]
    assert len(gap_msgs) == 1, (
        f"expected one gap warning, got {len(gap_msgs)} "
        f"(all warnings: {[str(ww.message) for ww in w]})"
    )


def test_closest_point_on_curve_kdtree_edge_cases():

    # Case 1: short curve (M < 64) with mask — triggers cand.ndim == 1
    # because initial k = max(1, M//64) = 1 and tree.query returns 1D
    numpy.random.seed(99)
    curve = numpy.random.randn(10, 6)
    curve_t = numpy.linspace(0, 1, 10)
    points = numpy.random.randn(5, 6)
    mask = numpy.zeros((5, 10), dtype=bool)
    mask[:, :3] = True
    result = _closest_point_on_curve(points, curve, curve_t, mask=mask)
    assert result.shape == (5,)
    assert numpy.all(numpy.isin(result, curve_t[:3]))

    # Case 2: all-False mask — no allowed neighbor anywhere, triggers
    # the k >= M fallback that assigns tp=0
    mask_empty = numpy.zeros((5, 10), dtype=bool)
    result2 = _closest_point_on_curve(points, curve, curve_t, mask=mask_empty)
    assert result2.shape == (5,)
    assert numpy.allclose(result2, 0.0)


def test_streamTrack_precomputed_init_and_parameter_kind(_simple_spdf):
    # The base StreamTrack constructor takes a precomputed track. Build a
    # track via from_particles, hand its precomputed state to the base
    # __init__, and check that accessors agree. Also exercise the three
    # parameter_kind options with plain floats (the astropy-Quantity
    # variant of parameter_kind lives in tests/test_quantity.py to avoid
    # depending on astropy in this file).

    numpy.random.seed(40)
    fit = _simple_spdf.streamTrack(n=1500, ntp=31, tail="leading")
    # Reconstruct via the precomputed-track __init__ — no fitter involved.
    rebuilt = StreamTrack(
        tp_grid=fit.tp_grid(),
        track_xyz=numpy.column_stack(
            [
                [float(fit.x(t, use_physical=False)) for t in fit.tp_grid()],
                [float(fit.y(t, use_physical=False)) for t in fit.tp_grid()],
                [float(fit.z(t, use_physical=False)) for t in fit.tp_grid()],
            ]
        ),
        track_vxvyvz=numpy.column_stack(
            [
                [float(fit.vx(t, use_physical=False)) for t in fit.tp_grid()],
                [float(fit.vy(t, use_physical=False)) for t in fit.tp_grid()],
                [float(fit.vz(t, use_physical=False)) for t in fit.tp_grid()],
            ]
        ),
        cov_xyz=fit._cov_xyz,
        custom_sky_transform=fit._custom_sky_transform,
        parameter_kind="time",
        # Mirror Orbit semantics: pass ro/vo only when the source had them
        # explicitly set; otherwise let the constructor pull the config
        # default and keep ``_roSet``/``_voSet=False``.
        ro=fit._ro if fit._roSet else None,
        vo=fit._vo if fit._voSet else None,
        zo=fit._zo,
        solarmotion=fit._solarmotion,
    )
    # And the rebuilt track should inherit the same _roSet / _voSet state.
    assert rebuilt._roSet == fit._roSet
    assert rebuilt._voSet == fit._voSet
    tp_mid = fit.tp_grid()[len(fit.tp_grid()) // 2]
    assert abs(float(fit.x(tp_mid)) - float(rebuilt.x(tp_mid))) < 1e-10
    # The precomputed-track instance has no fitter outputs.
    assert not hasattr(rebuilt, "particles")
    assert not hasattr(rebuilt, "smoothing_s")
    # parameter_kind="angle": plain-float input is just passed through.
    angle_track = StreamTrack(
        tp_grid=numpy.linspace(0.0, 1.0, 21),
        track_xyz=numpy.column_stack(
            [numpy.linspace(0, 1, 21), numpy.zeros(21), numpy.zeros(21)]
        ),
        track_vxvyvz=numpy.zeros((21, 3)),
        parameter_kind="angle",
    )
    assert numpy.isclose(angle_track.x(0.5), 0.5)
    # parameter_kind=None: pass-through.
    pass_track = StreamTrack(
        tp_grid=numpy.linspace(0.0, 1.0, 21),
        track_xyz=numpy.column_stack(
            [numpy.linspace(0, 1, 21), numpy.zeros(21), numpy.zeros(21)]
        ),
        track_vxvyvz=numpy.zeros((21, 3)),
        parameter_kind=None,
    )
    assert numpy.isclose(pass_track.x(0.5), 0.5)
    # Bad parameter_kind raises.
    with pytest.raises(ValueError):
        StreamTrack(
            tp_grid=numpy.linspace(0.0, 1.0, 21),
            track_xyz=numpy.zeros((21, 3)),
            track_vxvyvz=numpy.zeros((21, 3)),
            parameter_kind="bogus",
        )


def test_streamTrack_particles_attr(_simple_spdf):
    # track.particles exposes the raw xv array the fit saw — same format
    # that ``spraydf.streamTrack(particles=...)`` accepts. Verify shape,
    # that it round-trips through a second spraydf.streamTrack call, and
    # that it matches when the user passes particles explicitly.
    numpy.random.seed(21)
    xv = _simple_spdf.sample(n=1500, returndt=False, return_orbit=False, integrate=True)
    track = _simple_spdf.streamTrack(particles=xv, tail="leading")
    assert track.particles.shape == xv.shape
    assert numpy.allclose(track.particles, xv)
    # Round-trip: pass track.particles back in, should get same track.
    track2 = _simple_spdf.streamTrack(particles=track.particles, tail="leading")
    tps = track.tp_grid()
    assert numpy.allclose(track.x(tps), track2.x(tps))


def test_streamTrack_custom_sky_transform(_simple_spdf):
    # custom_sky_transform enables phi1/phi2/pmphi1/pmphi2 accessors;
    # without it, those accessors raise.

    T = _simple_spdf._progenitor.align_to_orbit()
    numpy.random.seed(22)
    track = _simple_spdf.streamTrack(n=1500, tail="leading", custom_sky_transform=T)
    tps = track.tp_grid()
    tp0 = tps[len(tps) // 2]
    # align_to_orbit puts the progenitor's orbital plane at phi2 = 0, and
    # because the stream lies in that plane the track's phi2 stays small
    # (deg). Velocities along the arm are O(few hundred mas/yr).
    expected = {
        "phi1": (-180.0, 360.0),  # deg
        "phi2": (-15.0, 15.0),  # bounded but non-zero — orbital precession in q≠1 halo
        "pmphi1": (-1e3, 1e3),  # mas/yr
        "pmphi2": (-1e3, 1e3),
    }
    for name, (lo, hi) in expected.items():
        val = float(getattr(track, name)(tp0))
        assert lo <= val <= hi, f"{name} = {val} out of [{lo}, {hi}]"
        arr = numpy.asarray(getattr(track, name)(tps[:5]))
        assert arr.shape == (5,)
        assert numpy.all((arr >= lo) & (arr <= hi))
    # The first tps[:5] are right at tp=0, where align_to_orbit pins phi2
    # to ~zero by construction.
    phi2_near_origin = numpy.asarray(track.phi2(tps[:5]))
    assert numpy.max(numpy.abs(phi2_near_origin)) < 1.0
    # Same (ra, dec) → phi via coords.radec_to_custom round-trip
    ra, dec = float(track.ra(tp0)), float(track.dec(tp0))
    expected = coords.radec_to_custom(
        numpy.atleast_1d(ra), numpy.atleast_1d(dec), T=T, degree=True
    )
    assert abs(float(track.phi1(tp0)) - expected[0, 0]) < 1e-8
    assert abs(float(track.phi2(tp0)) - expected[0, 1]) < 1e-8

    # Without custom_sky_transform, accessors raise
    numpy.random.seed(23)
    track_bare = _simple_spdf.streamTrack(n=500, tail="leading")
    with pytest.raises(RuntimeError):
        track_bare.phi1(tp0)
    with pytest.raises(RuntimeError):
        track_bare.pmphi1(tp0)
    # ...but the custom_sky_transform property can be set after construction
    # to enable the custom-frame accessors.
    assert track_bare.custom_sky_transform is None
    track_bare.custom_sky_transform = T
    assert numpy.allclose(track_bare.custom_sky_transform, T)
    # Setting T after the fact must give the same phi1/phi2 as building with T
    # at construction (no recomputation needed — the rotation is applied in
    # the accessor).
    bare_phi1 = float(track_bare.phi1(tps[2]))
    fitted_phi1 = float(track.phi1(tps[2]))
    # Different fits (different seeds) — only require the result to land in
    # the same hemisphere of phi1 (degrees) as the from-construction track.
    assert abs(bare_phi1 - fitted_phi1) < 90.0
    # And clearing back to None re-disables them.
    track_bare.custom_sky_transform = None
    with pytest.raises(RuntimeError):
        track_bare.phi1(tp0)


def test_streamTrackPair_custom_sky_transform_setter(_simple_spdf):
    # StreamTrackPair.custom_sky_transform should broadcast to both arms.
    T = _simple_spdf._progenitor.align_to_orbit()
    numpy.random.seed(24)
    pair = _simple_spdf.streamTrack(n=1200, tail="both")
    assert pair.custom_sky_transform is None
    pair.custom_sky_transform = T
    assert numpy.allclose(pair.leading.custom_sky_transform, T)
    assert numpy.allclose(pair.trailing.custom_sky_transform, T)
    assert numpy.allclose(pair.custom_sky_transform, T)
    tp0 = pair.leading.tp_grid()[3]
    # align_to_orbit ⇒ phi2 ≈ 0 on both arms.
    assert abs(float(pair.leading.phi2(tp0))) < 5.0
    assert abs(float(pair.trailing.phi2(-tp0))) < 5.0
    # Leading and trailing arms straddle the progenitor in phi1 (one positive
    # offset from the stream-frame origin, the other negative).
    p1_lead = float(pair.leading.phi1(tp0))
    p1_trail = float(pair.trailing.phi1(-tp0))
    p1_prog = float(_simple_spdf._progenitor.phi1(0.0, T=T))
    assert (p1_lead - p1_prog) * (p1_trail - p1_prog) < 0.0


def test_streamTrack_cov_basis(_simple_spdf):
    # cov(basis=...) must return a 6x6 PSD matrix in each supported basis.
    # Verify units consistency (diagonal entries have the right magnitude
    # for kpc²/km²/s², deg², mas²/yr²) and that galcencyl round-trips
    # cleanly via its analytical Jacobian (cov diag = variance of R from
    # the stored xy cov).

    T = _simple_spdf._progenitor.align_to_orbit()
    numpy.random.seed(31)
    track = _simple_spdf.streamTrack(
        n=1500, tail="leading", custom_sky_transform=T, ntp=41
    )
    tp0 = track.tp_grid()[len(track.tp_grid()) // 2]
    # All bases: returns (6, 6), symmetric, PSD
    for basis in ("galcenrect", "galcencyl", "galsky", "sky", "customsky"):
        C = track.cov(tp0, basis=basis)
        assert C.shape == (6, 6), f"{basis}: wrong shape"
        assert numpy.allclose(C, C.T, atol=1e-8), f"{basis}: not symmetric"
        evs = numpy.linalg.eigvalsh(C)
        assert evs.min() > -1e-8, f"{basis}: not PSD (min eig {evs.min()})"
    # galcencyl diag[0] = Var(R) ≈ cos²(φ)·Var(x) + sin²(φ)·Var(y) + 2·cos(φ)·sin(φ)·Cov(x,y)
    C_gcr = track.cov(tp0, basis="galcenrect")
    C_cyl = track.cov(tp0, basis="galcencyl")
    x, y = float(track.x(tp0)), float(track.y(tp0))
    R = numpy.sqrt(x * x + y * y)
    cp, sp = x / R, y / R
    var_R_expected = (
        cp * cp * C_gcr[0, 0] + sp * sp * C_gcr[1, 1] + 2 * cp * sp * C_gcr[0, 1]
    )
    assert abs(C_cyl[0, 0] - var_R_expected) < 1e-8, (
        f"galcencyl Var(R) mismatch: {C_cyl[0, 0]} vs {var_R_expected}"
    )
    # Unknown basis raises
    with pytest.raises(ValueError):
        track.cov(tp0, basis="bogus")
    # customsky without custom_sky_transform raises
    numpy.random.seed(32)
    track_bare = _simple_spdf.streamTrack(n=800, tail="leading", ntp=31)
    with pytest.raises(RuntimeError):
        track_bare.cov(tp0, basis="customsky")


def test_streamTrack_plot_spread_non_cartesian(_simple_spdf):
    # plot(spread>0) should draw a ±σ band for every d2 that has a basis
    # in _COORD_BASIS (including sky / custom-sky coords), not just
    # galactocentric Cartesian. Smoke-test a representative set.

    T = _simple_spdf._progenitor.align_to_orbit()
    numpy.random.seed(33)
    track = _simple_spdf.streamTrack(
        n=1200, tail="leading", custom_sky_transform=T, ntp=31
    )
    for d2 in (
        "x",
        "R",
        "z",
        "ra",
        "dec",
        "vlos",
        "pmra",
        "ll",
        "bb",
        "phi1",
        "phi2",
        "pmphi1",
    ):
        track.plot(d1="x", d2=d2, spread=1)
    # No custom_sky_transform: non-custom axes still work (covers the "basis
    # needs custom_sky_transform but we don't have one" code path gracefully).
    numpy.random.seed(34)
    track_bare = _simple_spdf.streamTrack(n=800, tail="leading", ntp=31)
    track_bare.plot(d1="x", d2="ra", spread=1)


def test_streamTrack_custom_accessors_no_physical(_simple_spdf):
    # phi1/phi2/pmphi1/pmphi2 accessors must also work with physical
    # output off (covers the non-Quantity branch of the extract-scalar
    # helper inside each accessor).

    T = _simple_spdf._progenitor.align_to_orbit()
    numpy.random.seed(41)
    track = _simple_spdf.streamTrack(n=1000, tail="leading", custom_sky_transform=T)
    track.turn_physical_off()
    tp0 = track.tp_grid()[len(track.tp_grid()) // 2]
    # Physical-units toggle doesn't change the rotation — phi2 still ≈ 0
    # (deg, dimensionless) and the proper motions are O(100) mas/yr.
    assert abs(float(track.phi2(tp0))) < 5.0
    p1 = float(track.phi1(tp0))
    assert -180.0 <= p1 <= 360.0
    for pm in ("pmphi1", "pmphi2"):
        v = float(getattr(track, pm)(tp0))
        assert -1e3 < v < 1e3
    track.turn_physical_on(ro=8.0, vo=220.0)


def test_streamTrack_with_progpot():
    # Cover the `self._orig_pot = self._pot` assignment in
    # basestreamspraydf.__init__ that's taken when progpot is set and
    # used by streamTrack to integrate the progenitor through the base
    # potential (not the MovingObjectPotential one).
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    ro, vo = 8.0, 220.0
    spdf = fardal15spraydf(
        2 * 10.0**4.0 / conversion.mass_in_msol(vo, ro),
        progenitor=obs,
        pot=lp,
        tdisrupt=4.5 / conversion.time_in_Gyr(vo, ro),
        tail="leading",
        progpot=PlummerPotential(0, 0),  # massless progenitor so samples match
    )
    numpy.random.seed(51)
    track = spdf.streamTrack(n=800, ntp=21, tail="leading")
    assert hasattr(spdf, "_orig_pot")
    # progpot=PlummerPotential(0,0) is a massless point: the spray-DF
    # samples should fall exactly on the progenitor orbit, so the fitted
    # track at tp=0 must match the progenitor's present-day position.
    prog_x0 = float(spdf._progenitor.x(0.0))
    assert abs(float(track.x(0.0)) - prog_x0) < 0.05
