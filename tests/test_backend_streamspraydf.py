###############################################################################
# test_backend_streamspraydf.py: multi-backend tests for the stream particle-
# spray DFs (fardal15spraydf / chen24spraydf).
#
# The sampler core (_sample_tail / _setup_rot / _calc_rtide / _calc_vc / both
# spray_df) is backend-agnostic: it resolves its namespace from the context and
# coerces the (exogenous, numpy) RNG draws onto the active backend, so the
# deterministic transforms run under numpy / jax / torch. The RNG itself stays
# numpy on every backend, so seeding numpy identically before each backend's
# sample makes the draws bit-identical and only the transform arithmetic differs
# in floating point -- hence we compare the actual integrate=False sample arrays
# at a tight rtol (not just summary statistics).
#
# integrate=True is exercised by test_sample_integrate_parity: under a backend the
# per-particle 2D-time-grid Orbit.integrate routes to the differentiable C-STM
# (dop853_c on a per-orbit 2-point [-dt_i, 0] grid), matching the numpy path.
#
# Backends that are not installed self-skip, so this is green on numpy alone.
###############################################################################
import numpy
import pytest
from scipy import interpolate

from galpy.backend import as_numpy, get_namespace, is_backend_array
from galpy.backend import random as grandom
from galpy.backend import use
from galpy.backend.interpolate import cubic_spline_coeffs, eval_cubic
from galpy.backend.linalg import psd_project
from galpy.df import chen24spraydf, fardal15spraydf
from galpy.df.streamTrack import _bin_by_tp, _closest_point_on_curve, _DiffSpline
from galpy.orbit import Orbit
from galpy.potential import LogarithmicHaloPotential
from galpy.util import conversion, coords

# This module manages backends explicitly (parametrizes over them), so it is
# exempt from the global --backend force fixture.
pytestmark = pytest.mark.backend_managed

BACKENDS = ["numpy"]
try:
    import jax

    jax.config.update("jax_enable_x64", True)

    BACKENDS.append("jax")
except ImportError:  # pragma: no cover
    jax = None
try:
    import torch

    torch.set_default_dtype(torch.float64)

    BACKENDS.append("torch")
except ImportError:  # pragma: no cover
    torch = None

AD_BACKENDS = [b for b in BACKENDS if b != "numpy"]

_RO, _VO = 8.0, 220.0
_SEED = 20260707


def _build(cls, **kwargs):
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    mass = 2 * 10.0**4.0 / conversion.mass_in_msol(_VO, _RO)
    td = 4.5 / conversion.time_in_Gyr(_VO, _RO)
    return cls(mass, progenitor=obs, pot=lp, tdisrupt=td, **kwargs)


def _sample(df, backend_name, n, **kwargs):
    numpy.random.seed(_SEED)
    with use(backend_name, force=True):
        return df.sample(n=n, return_orbit=False, integrate=False, **kwargs)


@pytest.mark.parametrize("cls", [fardal15spraydf, chen24spraydf])
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_sample_array_parity(cls, backend_name):
    # Same numpy RNG seed on every backend -> identical draws -> the sampled
    # (R, vR, vT, z, vz, phi) arrays match the numpy path up to the tiny FP
    # differences of the deterministic transforms.
    df = _build(cls, tail="leading")
    ref = _sample(df, "numpy", 300)
    got = _sample(df, backend_name, 300)
    if backend_name != "numpy":
        assert is_backend_array(got), (
            f"{cls.__name__} sample should be a backend array under {backend_name}"
        )
    numpy.testing.assert_allclose(
        as_numpy(got),
        as_numpy(ref),
        rtol=1e-6,
        atol=1e-8,
        err_msg=f"{cls.__name__} sample parity ({backend_name})",
    )


@pytest.mark.parametrize("cls", [fardal15spraydf, chen24spraydf])
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_sample_stats_parity(cls, backend_name):
    # Summary-statistics parity (mean/std per phase-space coordinate).
    df = _build(cls, tail="leading")
    ref = as_numpy(_sample(df, "numpy", 1000))
    got = as_numpy(_sample(df, backend_name, 1000))
    numpy.testing.assert_allclose(
        got.mean(axis=1), ref.mean(axis=1), rtol=1e-6, atol=1e-8
    )
    numpy.testing.assert_allclose(
        got.std(axis=1), ref.std(axis=1), rtol=1e-6, atol=1e-8
    )


@pytest.mark.parametrize("cls", [fardal15spraydf, chen24spraydf])
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_sample_tail_both_parity(cls, backend_name):
    # tail='both' concatenates leading+trailing sub-samples; the concatenation is
    # backend-aware (xp.hstack for backend arrays, numpy.hstack for numpy).
    df = _build(cls, tail="both")
    ref = _sample(df, "numpy", 200, tail="both")
    got = _sample(df, backend_name, 200, tail="both")
    assert as_numpy(got).shape == (6, 200)
    numpy.testing.assert_allclose(as_numpy(got), as_numpy(ref), rtol=1e-6, atol=1e-8)


def _sample_integ(df, backend_name, n, **kwargs):
    numpy.random.seed(_SEED)
    with use(backend_name, force=True):
        return df.sample(n=n, return_orbit=False, integrate=True, **kwargs)


@pytest.mark.parametrize("cls", [fardal15spraydf, chen24spraydf])
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_sample_integrate_parity(cls, backend_name):
    # integrate=True: the per-particle sample orbits are integrated to the present
    # day. Under a backend the per-orbit (N, nt) integration routes to the
    # differentiable C-STM (an RK dxdv-C method, dop853_c) and the result is a
    # backend array matching the numpy path (which uses the fixed-step default
    # symplec4_c) up to the two integrators' agreement (~1e-8).
    df = _build(cls, tail="leading")
    ref = _sample_integ(df, "numpy", 200)
    got = _sample_integ(df, backend_name, 200)
    if backend_name != "numpy":
        assert is_backend_array(got), (
            f"{cls.__name__} integrated sample should be a backend array "
            f"under {backend_name}"
        )
    numpy.testing.assert_allclose(
        as_numpy(got),
        as_numpy(ref),
        rtol=1e-5,
        atol=1e-6,
        err_msg=f"{cls.__name__} integrate=True parity ({backend_name})",
    )


@pytest.mark.parametrize("cls", [fardal15spraydf, chen24spraydf])
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_sample_standalone_key(cls, backend_name):
    # A backend key threaded WITHOUT a forced-backend context (the documented
    # standalone-key API): dt is a backend array, so the whole frame construction
    # (_setup_rot -> _rotate_to_arbitrary_vector) and the sample-orbit integration
    # run on the backend, and integrate=True returns a finite backend array. Guards
    # the regression where _setup_rot / the rotate leaf received the raw backend dt
    # and array_namespace(backend, [0,0,1]) / numpy.any(tensor) crashed.
    df = _build(cls, tail="leading")
    key = grandom.key(_SEED, backend_name)
    numpy.random.seed(_SEED)
    out = df.sample(n=80, return_orbit=False, integrate=True, key=key)
    assert is_backend_array(out), (
        f"{cls.__name__} standalone-{backend_name}-key not backend"
    )
    assert numpy.all(numpy.isfinite(as_numpy(out)))
    assert as_numpy(out).shape == (6, 80)
    # Reproducible given the same key AND the same numpy seed: the key controls the
    # stripping-time draw, while the spray_df offset draws still use the global
    # numpy RNG (a CRN gap for a later PR), so both sources must be reset.
    numpy.random.seed(_SEED)
    out2 = df.sample(n=80, return_orbit=False, integrate=True, key=key)
    numpy.testing.assert_array_equal(as_numpy(out), as_numpy(out2))


# --------------------------------------------------------------------------
# Differentiable stream TRACK from a BACKEND progenitor orbit. A backend
# (jax/torch) progenitor Orbit makes streamTrack integrate the dense progenitor
# curve with the differentiable C-STM (dop853_c) instead of the numpy default and
# stack it into a backend track_prog_cart, so the fitted track carries
# d(track)/d(progenitor). numpy progenitor -> numpy path, byte-identical.
# --------------------------------------------------------------------------
_PROG_IC = [1.0, 0.1, 1.1, 0.05, 0.03, 0.2]


def _spdf_prog(progenitor):
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    mass = 2 * 10.0**4.0 / conversion.mass_in_msol(_VO, _RO)
    td = 4.5 / conversion.time_in_Gyr(_VO, _RO)
    return fardal15spraydf(mass, progenitor=progenitor, pot=lp, tdisrupt=td)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_streamtrack_backend_progenitor_parity(backend_name):
    # A BACKEND progenitor orbit yields a differentiable BACKEND track whose
    # accessors match the numpy-progenitor track at common tp (evaluated via the
    # accessors, since the tp_grid trim can jump a particle under the tiny C-STM-
    # vs-default integrator difference). The progenitor curve integrates via C-STM.
    xp = jax.numpy if backend_name == "jax" else torch
    spdf_np = _spdf_prog(Orbit(_PROG_IC))
    numpy.random.seed(_SEED)
    xv, _ = spdf_np._sample_tail(300, True, leading=True)
    xv = numpy.asarray(xv, dtype=float)
    tr_np = spdf_np.streamTrack(particles=xv, tail="leading", velocity_weight=1.0)
    spdf_b = _spdf_prog(Orbit(xp.asarray(_PROG_IC)))
    tr_b = spdf_b.streamTrack(particles=xv, tail="leading", velocity_weight=1.0)
    assert is_backend_array(tr_b._track_xyz) and is_backend_array(tr_b._track_vxvyvz)
    g_np, g_b = numpy.asarray(tr_np.tp_grid()), numpy.asarray(tr_b.tp_grid())
    lo, hi = max(g_np[0], g_b[0]), min(g_np[-1], g_b[-1])
    tp = numpy.linspace(lo + 1e-6, hi - 1e-6, 100)
    for m in ("x", "y", "z", "vx", "vy", "vz"):
        numpy.testing.assert_allclose(
            as_numpy(getattr(tr_b, m)(tp)),
            numpy.asarray(getattr(tr_np, m)(tp)),
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"streamTrack backend-progenitor {m} parity ({backend_name})",
        )


def _frozen_spray_track(xv, base_leaf, pot=None, T=3.0, hd=801, curve=None):
    """Expose track(leaf, xp) = fit(curve(leaf)) returning the mean(6) AND
    covariance(6x6) track, with the fit STRUCTURE (cKDTree tp_assign, trim grid)
    and the GCV smoothers FROZEN from a base run. ``curve(leaf, xp, t_fwd, t_back)``
    builds the differentiable progenitor track_prog_cart -- default: the C-STM
    dop853_c curve as a function of the progenitor IC; the theta test supplies an
    in-backend-ODE curve as a function of a potential parameter. Frozen structure
    => a numpy FD and the backend autograd apply the identical operators, so
    d(track)/d(leaf) is FD-checkable (an end-to-end FD would move the cKDTree
    assignment + GCV lambda)."""
    t_fwd = numpy.linspace(0.0, T, hd)
    t_back = numpy.linspace(0.0, -T, hd)
    tg = numpy.concatenate([t_back[::-1], t_fwd[1:]])
    pc = coords.galcencyl_to_galcenrect(*xv)

    if curve is None:

        def curve(ic, xp, tf, tb):
            of = Orbit(ic)
            of.turn_physical_off()
            of.integrate(tf, pot, method="dop853_c")
            ob = Orbit(ic)
            ob.turn_physical_off()
            ob.integrate(tb, pot, method="dop853_c")

            def cart(o, ts):
                return xp.stack(
                    [o.x(ts), o.y(ts), o.z(ts), o.vx(ts), o.vy(ts), o.vz(ts)], axis=-1
                )

            return xp.concat([xp.flip(cart(ob, tb), axis=0), cart(of, tf)[1:]], axis=0)

    def _curve(leaf, xp):
        return curve(leaf, xp, t_fwd, t_back)

    def interp(cv, tp, xp):
        if xp is numpy:
            sp = [
                interpolate.InterpolatedUnivariateSpline(tg, cv[:, i], k=3)
                for i in range(6)
            ]
            return numpy.column_stack([s(numpy.atleast_1d(tp)) for s in sp])
        co = [cubic_spline_coeffs(xp, tg, cv[:, i], bc="not-a-knot") for i in range(6)]
        return xp.stack(
            [eval_cubic(xp, tg, co[i], xp.asarray(tp)) for i in range(6)], axis=-1
        )

    # base curve + frozen structure
    base = as_numpy(_curve(base_leaf, numpy))
    sign = numpy.broadcast_to((tg >= 0)[None, :], (pc.shape[0], tg.size))
    ta = _closest_point_on_curve(pc, base, tg, mask=sign, velocity_weight=1.0)
    keep = numpy.abs(ta - tg[-1]) > 1e-3 * abs(tg[-1] - tg[0])
    ta, pc = ta[keep], pc[keep]
    tp_hi = float(numpy.percentile(ta, 99.0))
    tp_grid = numpy.linspace(0.0, tp_hi, 101)
    tp_nodes = numpy.linspace(0.0, tp_hi, 21)
    off0 = pc - interp(base, ta, numpy)
    means0, cov0, cnt0 = _bin_by_tp(ta, off0, tp_nodes)
    with numpy.errstate(invalid="ignore"):
        per0 = numpy.sqrt(numpy.clip(numpy.einsum("mii->mi", cov0), 0.0, None))
        sig = numpy.where(
            cnt0[:, None] > 1,
            per0 / numpy.sqrt(numpy.maximum(cnt0[:, None], 1)),
            numpy.nan,
        )
    cntc = numpy.clip(cnt0.astype(float), 2.0, None)
    gm = [
        _DiffSpline(means0[:, i], tp_nodes, sig[:, i], None, 1.0, "numpy")._build(
            tp_grid
        )(means0[:, i])
        for i in range(6)
    ]
    gc = {}
    for a in range(6):
        for b in range(a, 6):
            v0 = cov0[:, a, b]
            sc = numpy.where(
                cnt0 > 1,
                numpy.sqrt(
                    numpy.clip(
                        (per0[:, a] ** 2 * per0[:, b] ** 2 + v0**2) / cntc, 0.0, None
                    )
                ),
                numpy.nan,
            )
            gc[(a, b)] = _DiffSpline(v0, tp_nodes, sc, None, 1.0, "numpy")._build(
                tp_grid
            )(v0)

    def track(leaf, xp):
        cv = _curve(leaf, xp)
        offsets = xp.asarray(pc) - interp(cv, ta, xp)
        means, covs, _ = _bin_by_tp(ta, offsets, tp_nodes)
        tmean = interp(cv, tp_grid, xp) + xp.stack(
            [xp.asarray(gm[i]) @ xp.nan_to_num(means[:, i]) for i in range(6)], axis=-1
        )
        ent = {}
        for a in range(6):
            for b in range(a, 6):
                ent[(a, b)] = xp.asarray(gc[(a, b)]) @ xp.nan_to_num(covs[:, a, b])
        rows = [
            xp.stack([ent[(min(a, b), max(a, b))] for b in range(6)], axis=-1)
            for a in range(6)
        ]
        cov = psd_project(
            xp.nan_to_num(xp.stack(rows, axis=-2), nan=0.0, posinf=0.0, neginf=0.0)
        )
        return tmean, cov

    return track


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_streamtrack_backend_progenitor_grad_fd(backend_name):
    # STRINGENT grad-vs-FD of d(track)/d(progenitor IC): the C-STM progenitor curve
    # makes the full mean+covariance track differentiable in the progenitor phase
    # space. With the fit structure frozen (see _frozen_spray_track), the gradient
    # of sum(track**2) matches a central finite difference along many random IC
    # directions to ~1e-6 -- exercising every IC component's contribution, not just
    # finite-and-nonzero. Both jax and torch (frozen structure removes the cKDTree
    # from the traced path, so jax.grad works here).
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    spdf = _spdf_prog(Orbit(_PROG_IC))
    numpy.random.seed(_SEED)
    xv, _ = spdf._sample_tail(300, True, leading=True)
    xv = numpy.asarray(xv, dtype=float)
    ic0 = numpy.array(_PROG_IC)
    track = _frozen_spray_track(xv, ic0, lp)

    def loss_np(ic):
        m, c = track(numpy.asarray(ic), numpy)
        return float(
            numpy.sum(numpy.asarray(m) ** 2) + numpy.sum(numpy.asarray(c) ** 2)
        )

    if backend_name == "jax":
        xp = get_namespace(jax.numpy.asarray(ic0))

        def L(ic):
            m, c = track(ic, xp)
            return jax.numpy.sum(m**2) + jax.numpy.sum(c**2)

        g = as_numpy(jax.grad(L)(jax.numpy.asarray(ic0)))
    else:
        ict = torch.tensor(ic0, requires_grad=True)
        m, c = track(ict, get_namespace(ict))
        (torch.sum(m**2) + torch.sum(c**2)).backward()
        g = as_numpy(ict.grad)
    assert numpy.all(numpy.isfinite(g)) and numpy.max(numpy.abs(g)) > 0
    rng = numpy.random.RandomState(_SEED)
    eps = 1e-6
    for _ in range(10):
        v = rng.randn(6)
        v /= numpy.linalg.norm(v)
        fd = (loss_np(ic0 + eps * v) - loss_np(ic0 - eps * v)) / (2 * eps)
        numpy.testing.assert_allclose(
            float(numpy.sum(g * v)),
            fd,
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"d(track)/d(prog IC) grad-vs-FD ({backend_name})",
        )


# --------------------------------------------------------------------------
# Differentiable stream track w.r.t. a POTENTIAL PARAMETER (theta). A backend
# (jax/torch) potential parameter makes spdf.streamTrack integrate the
# progenitor via the in-backend ODE (diffrax/torchdiffeq), so the fitted track
# carries d(track)/d(theta) -- the potential-parameter gradient the C-STM
# cannot give. Sampling is bypassed by passing particles=.
# --------------------------------------------------------------------------
_AMP0, _Q0 = 1.0, 0.9


def _spdf_pot(pot):
    mass = 2 * 10.0**4.0 / conversion.mass_in_msol(_VO, _RO)
    td = 4.5 / conversion.time_in_Gyr(_VO, _RO)
    return fardal15spraydf(mass, progenitor=Orbit(_PROG_IC), pot=pot, tdisrupt=td)


def _theta_curve(leaf, xp, tf, tb):
    # Progenitor curve as a function of theta=[amp, q]: integrate fwd+back and
    # stitch. The backend path uses the in-backend ODE (diffrax/torchdiffeq, the
    # differentiable-in-theta integrator); the numpy base/FD path uses dop853_c
    # (the integrator difference's d/d(theta) is negligible -- see FD tolerance).
    amp, q = leaf[0], leaf[1]
    pot = LogarithmicHaloPotential(amp=amp, q=q)
    if xp is numpy:
        ic, method = list(numpy.asarray(_PROG_IC, dtype=float)), "dop853_c"
    else:
        ic = xp.asarray(_PROG_IC)
        method = "diffrax" if "jax" in xp.__name__ else "torchdiffeq"
    of = Orbit(ic)
    of.turn_physical_off()
    of.integrate(tf, pot, method=method)
    ob = Orbit(ic)
    ob.turn_physical_off()
    ob.integrate(tb, pot, method=method)

    def cart(o, ts):
        return xp.stack(
            [o.x(ts), o.y(ts), o.z(ts), o.vx(ts), o.vy(ts), o.vz(ts)], axis=-1
        )

    return xp.concat([xp.flip(cart(ob, tb), axis=0), cart(of, tf)[1:]], axis=0)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_streamtrack_backend_theta_parity(backend_name):
    # A backend potential PARAMETER routes the progenitor curve through the
    # in-backend ODE -> a differentiable BACKEND track matching the numpy-parameter
    # track at common tp. Sampling is bypassed with particles=.
    xp = jax.numpy if backend_name == "jax" else torch
    spdf_np = _spdf_pot(LogarithmicHaloPotential(amp=_AMP0, q=_Q0))
    numpy.random.seed(_SEED)
    xv, _ = spdf_np._sample_tail(200, True, leading=True)
    xv = numpy.asarray(xv, dtype=float)
    tr_np = spdf_np.streamTrack(particles=xv, tail="leading", velocity_weight=1.0)
    spdf_b = _spdf_pot(LogarithmicHaloPotential(amp=xp.asarray(_AMP0), q=_Q0))
    tr_b = spdf_b.streamTrack(particles=xv, tail="leading", velocity_weight=1.0)
    assert is_backend_array(tr_b._track_xyz)
    g_np, g_b = numpy.asarray(tr_np.tp_grid()), numpy.asarray(tr_b.tp_grid())
    lo, hi = max(g_np[0], g_b[0]), min(g_np[-1], g_b[-1])
    tp = numpy.linspace(lo + 1e-6, hi - 1e-6, 80)
    for m in ("x", "y", "z", "vx", "vy", "vz"):
        numpy.testing.assert_allclose(
            as_numpy(getattr(tr_b, m)(tp)),
            numpy.asarray(getattr(tr_np, m)(tp)),
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"streamTrack backend-theta {m} parity ({backend_name})",
        )


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_streamtrack_backend_theta_grad_fd(backend_name):
    # STRINGENT grad-vs-FD of d(track)/d(potential parameter): the progenitor curve
    # integrates via the in-backend ODE, so the full mean+covariance track is
    # differentiable in theta=[amp, q] (which the C-STM cannot give). Frozen
    # structure -> d(sum(track**2))/d(theta) matches a central FD, both backends.
    spdf_np = _spdf_pot(LogarithmicHaloPotential(amp=_AMP0, q=_Q0))
    numpy.random.seed(_SEED)
    xv, _ = spdf_np._sample_tail(200, True, leading=True)
    xv = numpy.asarray(xv, dtype=float)
    theta0 = numpy.array([_AMP0, _Q0])
    track = _frozen_spray_track(xv, theta0, curve=_theta_curve, T=2.0, hd=401)

    def loss_np(th):
        m, c = track(numpy.asarray(th), numpy)
        return float(
            numpy.sum(numpy.asarray(m) ** 2) + numpy.sum(numpy.asarray(c) ** 2)
        )

    if backend_name == "jax":
        xp = get_namespace(jax.numpy.asarray(theta0))

        def L(th):
            m, c = track(th, xp)
            return jax.numpy.sum(m**2) + jax.numpy.sum(c**2)

        g = as_numpy(jax.grad(L)(jax.numpy.asarray(theta0)))
    else:
        theta_t = torch.tensor(theta0, requires_grad=True)
        m, c = track(theta_t, get_namespace(theta_t))
        (torch.sum(m**2) + torch.sum(c**2)).backward()
        g = as_numpy(theta_t.grad)
    assert numpy.all(numpy.isfinite(g)) and numpy.max(numpy.abs(g)) > 0
    eps = 1e-6
    for k in range(2):  # amp, q
        tp_ = theta0.copy()
        tp_[k] += eps
        tm_ = theta0.copy()
        tm_[k] -= eps
        fd = (loss_np(tp_) - loss_np(tm_)) / (2 * eps)
        numpy.testing.assert_allclose(
            g[k],
            fd,
            rtol=1e-4,
            atol=1e-6,
            err_msg=f"d(track)/d(theta[{k}]) grad-vs-FD ({backend_name})",
        )


def test_streamtrack_nonaxi_probe_numpy():
    # Regression guard: the backend-theta force probe must NOT crash streamTrack
    # for a non-axisymmetric numpy potential -- the probe supplies phi (and v) from
    # the progenitor's present-day phase-space point, so evaluateRforces succeeds.
    spdf = _spdf_pot(LogarithmicHaloPotential(normalize=1.0, b=0.8, q=0.9))
    numpy.random.seed(_SEED)
    xv, _ = spdf._sample_tail(120, True, leading=True)
    xv = numpy.asarray(xv, dtype=float)
    tr = spdf.streamTrack(particles=xv, tail="leading", velocity_weight=1.0)
    assert not is_backend_array(tr._track_xyz)  # numpy potential -> numpy track
    assert numpy.all(numpy.isfinite(numpy.asarray(tr._track_xyz)))


# --------------------------------------------------------------------------
# FULLY JITTABLE differentiable streamspray: jax.jit(sample()) and
# jax.jit(streamTrack()) sample INTERNALLY (no particles=) and reconstruct with the
# jax-native path (static-shape argmin closest-point + GCV P-spline + PSD covariance),
# so the spray sample xv(theta) FLOWS -- the generative d(track)/d(theta). The
# track_time_range is auto-estimated (the TRUE traced particle-extent). Streams are
# sensitive probes, so grad-vs-FD is h-CONVERGED (a single-h central difference is
# nonlinear at h~1e-3; the linear regime is ~1e-5). jax-only (jit).
# --------------------------------------------------------------------------
_JIT_IC = [1.2, 0.15, 0.85, 0.08, 0.05, 0.0]
_JIT_MASS, _JIT_TD, _JIT_N = 3e-5, 2.0, 40


def _jit_grad_hconv(loss, x0):
    # AD gradient + best relative error vs a central FD over a spread of h. The jitted
    # value function is compiled once and reused across the FD points (jit caches).
    jnp = jax.numpy
    g = float(jax.jit(jax.grad(loss))(jnp.asarray(x0)))
    lj = jax.jit(loss)
    best = min(
        abs(
            g
            - (float(lj(jnp.asarray(x0 + h))) - float(lj(jnp.asarray(x0 - h))))
            / (2 * h)
        )
        / max(abs(g), 1e-9)
        for h in (1e-4, 1e-5, 1e-6)
    )
    return g, best


@pytest.mark.skipif(jax is None, reason="jax required for jit tests")
def test_sample_jit_grad_fd():
    # jax.jit(sample()) is differentiable in a backend potential parameter with the
    # spray sample xv(theta) flowing (generative gradient). The k-vector draw is
    # reparameterized via the jax key -> AD and FD see the SAME noise.
    key = grandom.key(_SEED, backend="jax")

    def loss(amp):
        spdf = fardal15spraydf(
            _JIT_MASS,
            progenitor=Orbit(_JIT_IC),
            pot=LogarithmicHaloPotential(amp=amp, q=0.9),
            tdisrupt=_JIT_TD,
        )
        return spdf.sample(_JIT_N, return_orbit=False, tail="leading", key=key)[0].sum()

    g, best = _jit_grad_hconv(loss, 1.1)
    assert numpy.isfinite(g)
    assert best < 1e-3, f"jit sample grad-vs-FD best REL={best:.2e}"


@pytest.mark.skipif(jax is None, reason="jax required for jit tests")
def test_streamtrack_jit_grad_fd():
    # jax.jit(streamTrack()) samples INTERNALLY + reconstructs jax-natively (auto
    # track_time_range + GCV P-spline + PSD covariance), differentiable in a backend
    # potential parameter -- mean track AND covariance flow.
    key = grandom.key(_SEED, backend="jax")

    def loss(amp):
        spdf = fardal15spraydf(
            _JIT_MASS,
            progenitor=Orbit(_JIT_IC),
            pot=LogarithmicHaloPotential(amp=amp, q=0.9),
            tdisrupt=_JIT_TD,
        )
        tr = spdf.streamTrack(
            n=_JIT_N, tail="leading", velocity_weight=1.0, order=2, key=key
        )
        return tr._track_xyz.sum() + tr._cov_xyz.sum()

    g, best = _jit_grad_hconv(loss, 1.1)
    assert numpy.isfinite(g)
    assert best < 1e-3, f"jit streamTrack grad-vs-FD best REL={best:.2e}"

    # covariance is PSD (checked inside jit so cov_xyz stays a traced backend array)
    def min_eig(amp):
        spdf = fardal15spraydf(
            _JIT_MASS,
            progenitor=Orbit(_JIT_IC),
            pot=LogarithmicHaloPotential(amp=amp, q=0.9),
            tdisrupt=_JIT_TD,
        )
        tr = spdf.streamTrack(
            n=_JIT_N, tail="leading", velocity_weight=1.0, order=2, key=key
        )
        return jax.numpy.min(jax.numpy.linalg.eigvalsh(tr._cov_xyz))

    assert float(jax.jit(min_eig)(jax.numpy.asarray(1.1))) > -1e-9


@pytest.mark.skipif(jax is None, reason="jax required for jit tests")
def test_streamtrack_tp_scale_jit():
    # StreamTrack tp_scale mode (how jit streamTrack carries a data-dependent extent):
    # the parameter axis is a CONCRETE normalized grid + a TRACED physical scale
    # (physical tp = u * tp_scale), keeping the cubic-spline geometry concrete under
    # jit while the extent stays differentiable. d(x)/d(scale) flows through both the
    # track values and the tp->u query mapping.
    from galpy.df.streamTrack import StreamTrack

    jnp = jax.numpy
    u = numpy.linspace(0.0, 1.0, 60)

    def qx(scale):
        tx = jnp.stack(
            [jnp.sin(2 * jnp.pi * u) * scale, jnp.cos(2 * jnp.pi * u), u * 0.1], axis=-1
        )
        return StreamTrack(u, tx, tx * 0.0, tp_scale=scale).x(1.5, use_physical=False)

    # physical tp=1.5, scale=3 -> u=0.5 -> sin(2pi*0.5)*3 = 0
    numpy.testing.assert_allclose(float(jax.jit(qx)(jnp.asarray(3.0))), 0.0, atol=1e-10)
    g = float(jax.jit(jax.grad(qx))(jnp.asarray(3.0)))
    fd = (
        float(jax.jit(qx)(jnp.asarray(3.0 + 1e-6)))
        - float(jax.jit(qx)(jnp.asarray(3.0 - 1e-6)))
    ) / 2e-6
    numpy.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-8)
    # tp_grid() returns the physical axis (u * scale)
    cols = jnp.asarray(numpy.column_stack([u, u, u]))
    tr = StreamTrack(u, cols * 3.0, cols, tp_scale=jnp.asarray(3.0))
    numpy.testing.assert_allclose(as_numpy(tr.tp_grid())[-1], 3.0, rtol=1e-9)
