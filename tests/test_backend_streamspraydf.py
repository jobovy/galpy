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


@pytest.mark.skipif(jax is None, reason="jax required for jit tests")
def test_sample_jit_grad_fd_prog_ic():
    # jax.jit over a BACKEND progenitor IC: the traced IC makes the C path
    # inapplicable, so _integrate_progenitor routes the progenitor (and every
    # sampled orbit) to the in-backend ODE (diffrax) -> d(stream)/d(prog IC) flows.
    # Reparameterized noise (jax key) -> AD and FD see the same draw.
    key = grandom.key(_SEED, backend="jax")

    def loss(r0):
        ic = jax.numpy.array([r0, 0.15, 0.85, 0.08, 0.05, 0.0])
        spdf = fardal15spraydf(
            _JIT_MASS,
            progenitor=Orbit(ic),
            pot=LogarithmicHaloPotential(amp=1.0, q=0.9),
            tdisrupt=_JIT_TD,
        )
        return spdf.sample(_JIT_N, return_orbit=False, tail="leading", key=key)[0].sum()

    g, best = _jit_grad_hconv(loss, 1.2)
    assert numpy.isfinite(g)
    assert best < 1e-3, f"jit sample prog-IC grad-vs-FD best REL={best:.2e}"


# --------------------------------------------------------------------------
# Differentiable stream track w.r.t. the PROGENITOR MASS. A backend (jax/torch)
# progenitor mass -- a constant M or a time-varying M(t) -- traces the tidal radius
# rtide -> the spray SCALE, so the sampled particles (and the fitted track) carry
# d(track)/d(mass) even though the progenitor CURVE is mass-independent (it is
# integrated in-backend so the jittable reconstruction stays on the backend). This
# is the mass-loss-history gradient the potential/IC gradients cannot give. Grad-vs-FD
# uses reassignment-INVARIANT functionals (the frozen-assignment AD equals the true
# gradient there; an index-wise loss is dominated by the discrete-assignment jitter).
# --------------------------------------------------------------------------
def _mass_track_xyz(m, key=None):
    spdf = fardal15spraydf(
        m,
        progenitor=Orbit(_JIT_IC),
        pot=LogarithmicHaloPotential(amp=1.0, q=0.9),
        tdisrupt=_JIT_TD,
    )
    return spdf.streamTrack(
        n=_JIT_N,
        tail="leading",
        velocity_weight=1.0,
        order=1,
        key=key,
        track_time_range=2.0,
    )._track_xyz


def _jit_grad_hconv_rel(loss, x0):
    # As _jit_grad_hconv but with RELATIVE FD steps x0*(1+-h): the mass is O(1e-5), so
    # an absolute step would flip its sign (rtide((-M)**(1/3)) -> NaN).
    jnp = jax.numpy
    g = float(jax.jit(jax.grad(loss))(jnp.asarray(x0)))
    lj = jax.jit(loss)
    best = min(
        abs(
            g
            - (
                float(lj(jnp.asarray(x0 * (1 + h))))
                - float(lj(jnp.asarray(x0 * (1 - h))))
            )
            / (2 * x0 * h)
        )
        / max(abs(g), 1e-9)
        for h in (1e-4, 1e-5, 1e-6)
    )
    return g, best


@pytest.mark.skipif(jax is None, reason="jax required for jit tests")
def test_streamtrack_jit_grad_fd_mass():
    # jax.jit(streamTrack()) is differentiable in a CONSTANT backend progenitor mass:
    # M scales rtide -> the spray offsets -> the track. A reassignment-invariant
    # functional (sum over the track) makes the frozen-assignment AD match the true FD.
    key = grandom.key(_SEED, backend="jax")

    def loss(m):
        return _mass_track_xyz(m, key=key).sum()

    g, best = _jit_grad_hconv_rel(loss, _JIT_MASS)
    assert numpy.isfinite(g) and abs(g) > 0
    # the AD gradient is essentially exact; the h-converged FD floor is ~2e-8 here,
    # so 1e-6 is a real regression detector (not a loose finite-difference sanity check)
    assert best < 1e-6, f"jit streamTrack d/dM grad-vs-FD best REL={best:.2e}"


@pytest.mark.skipif(jax is None, reason="jax required for jit tests")
def test_streamtrack_jit_grad_fd_mass_evolving():
    # A time-varying M(t)=M0*exp(rate*t) makes the mass-LOSS HISTORY differentiable:
    # d(track)/d(rate) flows through rtide at each stripping time. Uses a
    # reassignment-invariant smooth functional (mean track radius) where AD == FD.
    jnp = jax.numpy
    key = grandom.key(_SEED, backend="jax")
    M0 = _JIT_MASS

    def loss(rate):
        xyz = _mass_track_xyz(lambda t: M0 * jnp.exp(rate * t), key=key)
        return jnp.mean(jnp.sqrt(jnp.sum(xyz**2, axis=1)))

    g, best = _jit_grad_hconv(loss, 0.05)
    assert numpy.isfinite(g) and abs(g) > 0
    assert best < 1e-6, f"jit streamTrack d/d(rate) grad-vs-FD best REL={best:.2e}"


@pytest.mark.skipif(jax is None, reason="jax required for eager AD")
def test_streamtrack_backend_mass_grad_fd_eager():
    # EAGER (non-jit) d(track)/d(mass): jax.grad traces the particles (mass -> rtide ->
    # spray), so the generative reconstruction stays on the backend and the mass
    # gradient flows without jit. FD reuses the same key (reparameterized noise) with
    # RELATIVE steps (the mass is O(1e-5)); reassignment-invariant sum-of-squares.
    # (torch takes the same namespace-generic path -- validated locally; its eager
    # generative reconstruction is minutes-slow, as for every jit-only generative test.)
    jnp = jax.numpy
    key = grandom.key(_SEED, backend="jax")
    M0 = _JIT_MASS

    def loss(m):
        return jnp.sum(_mass_track_xyz(m, key=key) ** 2)

    # FD is jitted so it takes the SAME jax-native reconstruction as the (traced) AD; a
    # concrete non-jit forward would fall to the numpy reconstruction and not match.
    lj = jax.jit(loss)

    def val(m):
        return float(lj(jnp.asarray(m)))

    g = float(jax.grad(loss)(jnp.asarray(M0)))
    assert numpy.isfinite(g) and abs(g) > 0
    best = min(
        abs(g - (val(M0 * (1 + h)) - val(M0 * (1 - h))) / (2 * M0 * h))
        / max(abs(g), 1e-9)
        for h in (1e-4, 1e-5, 1e-6)
    )
    assert best < 1e-6, f"eager d(track)/dM grad-vs-FD best REL={best:.2e}"


# --------------------------------------------------------------------------
# Differentiable stream track w.r.t. the STRIPPING-TIME DISTRIBUTION. A backend
# (jax/torch) stripping_pdf returns a backend array, so its inverse-CDF is built +
# inverted natively (cumulative-trapezoid CDF + searchsorted linear interp) instead
# of via the scipy spline: the drawn stripping times dt trace the pdf parameters,
# and — like the mass — a backend stripping_pdf is its own backend-sampling trigger.
# This makes the mass-loss-time HISTORY (e.g. a stripping burst) fittable. Distinct
# from a numpy stripping_pdf, which keeps the (byte-identical) scipy path.
# --------------------------------------------------------------------------
def _build_stripping(pdf):
    mass = 2 * 10.0**4.0 / conversion.mass_in_msol(_VO, _RO)
    td = 4.5 / conversion.time_in_Gyr(_VO, _RO)
    return fardal15spraydf(
        mass,
        progenitor=Orbit(_PROG_IC),
        pot=LogarithmicHaloPotential(amp=1.0, q=0.9),
        tdisrupt=td,
        stripping_pdf=pdf,
    )


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_stripping_backend_inv_cdf_matches_scipy(backend_name):
    # The backend-native inverse-CDF (searchsorted + linear interp of a cumulative-
    # trapezoid CDF) must reproduce the numpy scipy k=1 inverse-CDF spline for the SAME
    # pdf: draw at a shared u (same key -> same uniforms) and compare dt to scipy.
    # Fast (no track) -> the torch coverage of the new inverse-CDF numerics.
    xp = jax.numpy if backend_name == "jax" else torch
    td = 4.5 / conversion.time_in_Gyr(_VO, _RO)
    t0, sig = -0.4 * td, 0.5 * td

    def pdf_np(t):
        return numpy.exp(-(((t - t0) / sig) ** 2))

    def pdf_b(t):
        return xp.exp(-(((t - xp.asarray(t0)) / sig) ** 2))

    sp_np = _build_stripping(pdf_np)
    sp_b = _build_stripping(pdf_b)
    assert is_backend_array(sp_b._stripping_cdf[0])
    # a numpy pdf keeps the scipy path untouched (byte-identical)
    assert sp_np._stripping_cdf is None and sp_np._stripping_inv_cdf is not None
    key = grandom.key(_SEED, backend=backend_name)
    u = as_numpy(grandom.uniform(key, (200,)))
    dt_b = as_numpy(sp_b._draw_stripping_dt(200, key=key))
    dt_scipy = -sp_np._stripping_inv_cdf(u)
    numpy.testing.assert_allclose(
        dt_b, dt_scipy, rtol=1e-9, atol=1e-9, err_msg=f"inverse-CDF ({backend_name})"
    )


def _stripping_track_xyz(t0, key):
    def pdf(t):
        return jax.numpy.exp(-(((t - t0) / (0.5 * _JIT_TD)) ** 2))

    spdf = fardal15spraydf(
        _JIT_MASS,
        progenitor=Orbit(_JIT_IC),
        pot=LogarithmicHaloPotential(amp=1.0, q=0.9),
        tdisrupt=_JIT_TD,
        stripping_pdf=pdf,
    )
    return spdf.streamTrack(
        n=_JIT_N,
        tail="leading",
        velocity_weight=1.0,
        order=1,
        key=key,
        track_time_range=2.0,
    )._track_xyz


@pytest.mark.skipif(jax is None, reason="jax required for jit tests")
def test_streamtrack_jit_grad_fd_stripping():
    # jax.jit(streamTrack()) is differentiable in the center t0 of a Gaussian stripping
    # burst: t0 -> the inverse-CDF -> the stripping times -> the sampled particles -> the
    # track. Reassignment-invariant functional (mean track radius) -> grad-vs-FD.
    jnp = jax.numpy
    key = grandom.key(_SEED, backend="jax")

    def loss(t0):
        xyz = _stripping_track_xyz(t0, key)
        return jnp.mean(jnp.sqrt(jnp.sum(xyz**2, axis=1)))

    g, best = _jit_grad_hconv(loss, -0.4 * _JIT_TD)
    assert numpy.isfinite(g) and abs(g) > 0
    # AD is essentially exact; the h-converged FD floor is ~8e-8, so 1e-6 is a real
    # regression detector (the searchsorted inverse-CDF sits a touch above the mass floor)
    assert best < 1e-6, f"jit streamTrack d/d(t0) grad-vs-FD best REL={best:.2e}"


# --------------------------------------------------------------------------
# Differentiable stream track w.r.t. the PERICENTER-STRIPPING width. The built-in
# pericenter_stripping_pdf helper builds a Gaussian mixture centered on the progenitor's
# pericenter passages; a backend `sigma` (or a backend pot/IC) makes it return a BACKEND
# pdf that composes with the backend inverse-CDF above, so the whole stream is jittable and
# differentiable in the stripping WIDTH. The pericenter TIMES are found concretely (a
# find_peaks discrete op) and frozen as a backend constant.
# --------------------------------------------------------------------------
_PERI_TD, _PERI_SIGMA = 12.0, 2.0  # internal units: ~3 pericenter passages of _PROG_IC


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_pericenter_stripping_pdf_backend_parity(backend_name):
    # A backend sigma -> a backend Gaussian-mixture pericenter pdf whose VALUES match the
    # numpy pdf (same concrete pericenter times) and which is a backend array (composes with
    # the backend inverse-CDF). Fast (no track) -> the torch coverage of the helper.
    from galpy.df import pericenter_stripping_pdf

    xp = jax.numpy if backend_name == "jax" else torch
    pot = LogarithmicHaloPotential(amp=1.0, q=0.9)
    pdf_np = pericenter_stripping_pdf(Orbit(_PROG_IC), pot, _PERI_TD, _PERI_SIGMA)
    pdf_b = pericenter_stripping_pdf(
        Orbit(_PROG_IC), pot, _PERI_TD, xp.asarray(_PERI_SIGMA)
    )
    assert not is_backend_array(pdf_np(-0.5 * _PERI_TD))
    tg = numpy.linspace(-_PERI_TD, 0.0, 60)
    vb = pdf_b(xp.asarray(tg))
    assert is_backend_array(vb)
    numpy.testing.assert_allclose(
        as_numpy(vb), pdf_np(tg), rtol=1e-9, atol=1e-12, err_msg=f"({backend_name})"
    )


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
@pytest.mark.parametrize("trigger", ["pot", "ic"])
def test_pericenter_stripping_pdf_backend_triggers(backend_name, trigger):
    # Not only a backend sigma: a backend pot PARAMETER (spotted by the forced-numpy force
    # probe) or a backend progenitor IC also makes the helper return a composable backend
    # pdf, so it works whenever the caller is already in backend-land.
    from galpy.df import pericenter_stripping_pdf

    xp = jax.numpy if backend_name == "jax" else torch
    ic = xp.asarray(_PROG_IC) if trigger == "ic" else _PROG_IC
    amp = xp.asarray(1.0) if trigger == "pot" else 1.0
    pdf = pericenter_stripping_pdf(
        Orbit(ic), LogarithmicHaloPotential(amp=amp, q=0.9), _PERI_TD, _PERI_SIGMA
    )
    v = pdf(xp.asarray(numpy.linspace(-_PERI_TD, 0.0, 40)))
    assert is_backend_array(v)  # numpy sigma, but backend pot/IC -> backend pdf
    assert numpy.all(numpy.isfinite(as_numpy(v)))


@pytest.mark.skipif(jax is None, reason="jax required for jit tests")
def test_streamtrack_jit_grad_fd_pericenter_sigma():
    # d(track)/d(stripping-burst-width sigma) through the backend pericenter pdf + the
    # backend inverse-CDF: the pericenter TIMES are frozen (find_peaks is discrete), the
    # Gaussian width flows. Reassignment-invariant functional -> grad-vs-FD.
    from galpy.df import pericenter_stripping_pdf

    jnp = jax.numpy
    key = grandom.key(_SEED, backend="jax")
    pot = LogarithmicHaloPotential(amp=1.0, q=0.9)

    def loss(sigma):
        pdf = pericenter_stripping_pdf(Orbit(_PROG_IC), pot, _PERI_TD, sigma)
        spdf = fardal15spraydf(
            _JIT_MASS,
            progenitor=Orbit(_PROG_IC),
            pot=pot,
            tdisrupt=_PERI_TD,
            stripping_pdf=pdf,
        )
        xyz = spdf.streamTrack(
            n=_JIT_N,
            tail="leading",
            velocity_weight=1.0,
            order=1,
            key=key,
            track_time_range=2.0,
        )._track_xyz
        return jnp.mean(jnp.sqrt(jnp.sum(xyz**2, axis=1)))

    g, best = _jit_grad_hconv_rel(
        loss, _PERI_SIGMA
    )  # sigma is O(1) -> relative FD steps
    assert numpy.isfinite(g) and abs(g) > 0
    # AD is essentially exact; the h-converged FD floor is ~2e-7 here (Gaussian-mixture
    # inverse-CDF), so 1e-5 is a real regression detector with CI-robust margin.
    assert best < 1e-5, (
        f"jit streamTrack d/d(sigma) pericenter grad-vs-FD best REL={best:.2e}"
    )


@pytest.mark.skipif(jax is None, reason="jax required for backend-theta construction")
def test_backend_sampling_center_not_implemented():
    # Differentiable stream sampling (a backend potential parameter -> _bsamp set)
    # combined with a center orbit is not yet supported and must raise, not silently
    # produce a wrong (numpy-frozen) track.
    with pytest.raises(NotImplementedError):
        fardal15spraydf(
            _JIT_MASS,
            progenitor=Orbit(_JIT_IC),
            pot=LogarithmicHaloPotential(amp=jax.numpy.asarray(1.0), q=0.9),
            tdisrupt=_JIT_TD,
            center=Orbit(_JIT_IC),
        )


@pytest.mark.skipif(jax is None, reason="jax required for tp_scale accessors")
def test_streamtrack_tp_scale_accessors():
    # In tp_scale mode the full-6D eval (_eval_cart, used by R/phi/heliocentric
    # accessors) and the covariance accessor map a PHYSICAL query tp to the concrete
    # normalized axis (tp/tp_scale) before interpolating. Build a backend track with
    # covariance + tp_scale and check both accessors evaluate on the physical axis.
    from galpy.df.streamTrack import StreamTrack

    jnp = jax.numpy
    u = numpy.linspace(0.0, 1.0, 40)
    xyz = jnp.asarray(
        numpy.column_stack(
            [numpy.sin(2 * numpy.pi * u) + 2.0, numpy.cos(2 * numpy.pi * u), u * 0.1]
        )
    )
    vxyz = jnp.asarray(numpy.column_stack([u * 0.0 + 0.1, u * 0.0 + 0.2, u * 0.0]))
    cov = jnp.asarray(numpy.broadcast_to(numpy.eye(6) * 0.01, (40, 6, 6)).copy())
    tr = StreamTrack(jnp.asarray(u), xyz, vxyz, cov_xyz=cov, tp_scale=jnp.asarray(3.0))
    # physical tp=1.5, scale=3 -> u=0.5 -> x=sin(pi)+2=2, y=cos(pi)=-1 -> R=sqrt(5)
    numpy.testing.assert_allclose(
        float(as_numpy(numpy.atleast_1d(tr.R(1.5, use_physical=False))[0])),
        numpy.sqrt(5.0),
        rtol=1e-6,
    )
    cov_b = as_numpy(tr.cov(1.5, use_physical=False))
    assert cov_b.shape == (6, 6) and numpy.all(numpy.isfinite(cov_b))
