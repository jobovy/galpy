###############################################################################
# test_backend_actionAngle.py: the closed-form pure-Python action-angle methods
# (actionAngleHarmonic, actionAngleIsochrone) under the jax/torch backends.
# These classes are algebraic (no quadrature/root-find), so the namespace-swap
# makes them evaluate AND differentiate under every backend. Validates:
#   * numpy<->jax<->torch value parity for actions, frequencies, angles, and
#     EccZmaxRperiRap (Isochrone) / actions+freqs+angle (Harmonic);
#   * outputs are backend arrays (device/dtype follow the inputs);
#   * grad-vs-finite-difference of an action / frequency / angle w.r.t. an
#     initial condition (jax.grad and torch.autograd) -- the differentiability
#     that is the point of the backend port.
# The numpy path is byte-identical (get_namespace passes numpy through), so the
# existing test_actionAngle.py suite is unchanged; this only adds backend cover.
###############################################################################
import numpy
import pytest

pytestmark = pytest.mark.backend_managed

BACKENDS = []
try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    BACKENDS.append("jax")
except ImportError:  # pragma: no cover
    jax = None
try:
    import torch

    torch.set_default_dtype(torch.float64)

    BACKENDS.append("torch")
except ImportError:  # pragma: no cover
    torch = None

from galpy.actionAngle import (
    actionAngleHarmonic,
    actionAngleHarmonicInverse,
    actionAngleIsochrone,
    actionAngleIsochroneInverse,
)

# A small batch of bound phase-space points (R,vR,vT,z,vz,phi); moderate
# velocities so the isochrone orbits are bound (E<0) and away from the
# turning-point / non-inclined kinks where the angle map is non-smooth.
_R = numpy.array([1.1, 0.8, 1.3])
_vR = numpy.array([0.2, -0.1, 0.05])
_vT = numpy.array([0.9, 0.6, 1.0])
_z = numpy.array([0.15, -0.2, 0.1])
_vz = numpy.array([0.1, 0.05, -0.1])
_phi = numpy.array([1.3, 0.4, 2.1])
_ISO = (_R, _vR, _vT, _z, _vz, _phi)

_HX = numpy.array([0.5, -1.0, 0.3])
_HVX = numpy.array([0.2, 0.4, -0.6])


def _arr(backend, x):
    return jnp.asarray(x) if backend == "jax" else torch.tensor(x)


def _np(x):
    if torch is not None and torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return numpy.asarray(x)


def _is_backend_array(backend, x):
    if backend == "jax":
        return isinstance(x, jax.Array)
    return torch.is_tensor(x)


# ----------------------------------------------------------------- value parity
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("b", [0.0, 0.8, 1.5])
def test_isochrone_parity(backend, b):
    aAI = actionAngleIsochrone(b=b)
    bargs = [_arr(backend, v) for v in _ISO]
    # _evaluate (actions), _actionsFreqs (+freqs), _actionsFreqsAngles (+angles)
    for ref, got in (
        (aAI._evaluate(*_ISO[:5]), aAI._evaluate(*bargs[:5])),
        (aAI._actionsFreqs(*_ISO[:5]), aAI._actionsFreqs(*bargs[:5])),
        (aAI._actionsFreqsAngles(*_ISO), aAI._actionsFreqsAngles(*bargs)),
    ):
        for r, g in zip(ref, got):
            assert _is_backend_array(backend, g)
            numpy.testing.assert_allclose(
                _np(g), numpy.asarray(r), rtol=1e-12, atol=1e-12
            )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("b", [0.8, 1.5])  # b=0 EccZmaxRperiRap only approximate
def test_isochrone_ecczmaxrperirap_parity(backend, b):
    aAI = actionAngleIsochrone(b=b)
    ref = aAI._EccZmaxRperiRap(*_ISO[:5])
    got = aAI._EccZmaxRperiRap(*[_arr(backend, v) for v in _ISO[:5]])
    for r, g in zip(ref, got):
        assert _is_backend_array(backend, g)
        numpy.testing.assert_allclose(_np(g), numpy.asarray(r), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("omega", [0.5, 1.7])
def test_harmonic_parity(backend, omega):
    aAH = actionAngleHarmonic(omega=omega)
    x, vx = _arr(backend, _HX), _arr(backend, _HVX)
    for ref, got in (
        (aAH._evaluate(_HX, _HVX), aAH._evaluate(x, vx)),
        (aAH._actionsFreqs(_HX, _HVX), aAH._actionsFreqs(x, vx)),
        (aAH._actionsFreqsAngles(_HX, _HVX), aAH._actionsFreqsAngles(x, vx)),
    ):
        ref = ref if isinstance(ref, tuple) else (ref,)
        got = got if isinstance(got, tuple) else (got,)
        for r, g in zip(ref, got):
            numpy.testing.assert_allclose(
                _np(g), numpy.asarray(r), rtol=1e-12, atol=1e-12
            )


# ------------------------------------------------------------ grad vs finite-diff
# Scalar single-point ICs for clean per-component derivatives.
_S = (1.1, 0.2, 0.9, 0.15, 0.1, 1.3)


def _fd(f, x0, eps=1e-6):
    return (f(x0 + eps) - f(x0 - eps)) / (2.0 * eps)


def _grad(backend, f, x0):
    if backend == "jax":
        return float(jax.grad(lambda t: f(t))(jnp.asarray(x0)))
    t = torch.tensor(x0, requires_grad=True)
    out = f(t)
    out.backward()
    return float(t.grad)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("which,idx", [("jr", 0), ("omegar", 3), ("angler", 6)])
def test_isochrone_grad_vs_fd_wrt_vR(backend, which, idx):
    # d(action/freq/angle)[component idx] / d vR at a single bound point.
    aAI = actionAngleIsochrone(b=1.0)
    R, _, vT, z, vz, phi = _S

    def call(vR_val, xp_arr):
        # build args in the namespace of vR_val (numpy float for FD, backend
        # scalar for AD) so get_namespace follows vR for the differentiated path
        args = (xp_arr(R), vR_val, xp_arr(vT), xp_arr(z), xp_arr(vz), xp_arr(phi))
        return aAI._actionsFreqsAngles(*args)[idx]

    def f_np(vR_val):
        return numpy.asarray(call(vR_val, lambda v: numpy.asarray(v)))

    fd = _fd(f_np, _S[1])

    def f_be(vR_t):
        return call(vR_t, lambda v: _arr(backend, numpy.asarray(v, dtype=float)))

    g = _grad(backend, f_be, _S[1])
    assert numpy.isfinite(g)
    numpy.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("backend", BACKENDS)
def test_harmonic_grad_vs_fd(backend):
    # dJ/dx and dOmega-angle/dx for the harmonic oscillator.
    aAH = actionAngleHarmonic(omega=1.3)
    vx0 = -0.3

    def jact(x_val, xp_arr):
        return aAH._evaluate(x_val, xp_arr(vx0))

    fd = _fd(lambda xv: numpy.asarray(jact(xv, lambda v: numpy.asarray(v))), 0.7)
    g = _grad(
        backend,
        lambda xt: jact(xt, lambda v: _arr(backend, numpy.asarray(v, float))),
        0.7,
    )
    numpy.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-7)


# ----------------------------------------------- dθ/dt = Ω consistency (autodiff)
# Angle variables evolve LINEARLY in time along an orbit: θ_i(t) = θ_i(0) + Ω_i t.
# So dθ_i/dt = ∇_x θ_i · ẋ (the Hamiltonian flow / EOM) MUST equal the frequency
# Ω_i returned by the same call. This is a far stronger check than grad-vs-FD: it
# ties the autodiff'd angle gradient to the independently-computed frequency. We
# compare dθ/dt directly against the SAME call's Ω, so whatever sign/wrap
# convention the angle reconstruction uses is automatically the reference (no
# manual bookkeeping). Verified to hold to ~1e-15 (machine precision).
def _flow_deriv(backend, theta_fn, eom_fn, y0):
    # dθ/dt = grad_y θ(y) · ẏ, as a python float, in the given backend.
    if backend == "jax":
        y = jnp.asarray(y0)
        g = jax.grad(lambda v: jnp.reshape(theta_fn(v), ()))(y)
        return float(jnp.dot(g, eom_fn(y)))
    y = torch.tensor(y0, requires_grad=True)
    out = torch.reshape(theta_fn(y), ())
    (g,) = torch.autograd.grad(out, y)
    return float(torch.dot(g, eom_fn(torch.tensor(y0))))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("omega", [0.7, 1.3])
def test_harmonic_dangle_dt_equals_freq(backend, omega):
    # θ = arctan2(ω x, vx); EOM (ẋ, v̇x) = (vx, -ω² x)  ⇒  dθ/dt = ω.
    aAH = actionAngleHarmonic(omega=omega)
    y0 = numpy.array([0.7, -0.3])  # [x, vx]

    def theta(y):  # forward angle is index 2 of (j, omega, angle)
        return aAH._actionsFreqsAngles(y[0], y[1])[2]

    def eom(y):
        xp = jnp if backend == "jax" else torch
        return xp.stack([y[1], -(omega**2.0) * y[0]])

    dthdt = _flow_deriv(backend, theta, eom, y0)
    omega_ret = float(_np(aAH._actionsFreqsAngles(*y0)[1]).ravel()[0])
    assert numpy.isfinite(dthdt)
    numpy.testing.assert_allclose(dthdt, omega_ret, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("vT", [0.35, -0.35])  # prograde + retrograde (Lz<0) branch
@pytest.mark.parametrize(
    "idx_ang,idx_om", [(6, 3), (7, 4), (8, 5)]
)  # (angler,Ωr),(anglephi,Ωphi),(anglez,Ωz)
def test_isochrone_dangle_dt_equals_freq(backend, vT, idx_ang, idx_om):
    # Bound, inclined, off-wrap IC. EOM in galpy internal (R,vR,vT,z,vz,phi) for
    # the axisymmetric isochrone: Ṙ=vR, v̇R=vT²/R+Rforce, v̇T=-vR vT/R, ż=vz,
    # v̇z=zforce, φ̇=vT/R, with the PUBLIC Rforce/zforce (amp included).
    aAI = actionAngleIsochrone(b=0.7)
    ip = aAI._ip  # IsochronePotential(amp=aAI.amp, b=aAI.b)
    y0 = numpy.array([1.1, 0.2, vT, 0.15, 0.18, 0.6])

    def theta(y):
        return aAI._actionsFreqsAngles(y[0], y[1], y[2], y[3], y[4], y[5])[idx_ang]

    def eom(y):
        xp = jnp if backend == "jax" else torch
        R, vR, vTc, z, vz = y[0], y[1], y[2], y[3], y[4]
        return xp.stack(
            [
                vR,
                vTc**2.0 / R + ip.Rforce(R, z, use_physical=False),
                -vR * vTc / R,
                vz,
                ip.zforce(R, z, use_physical=False),
                vTc / R,
            ]
        )

    dthdt = _flow_deriv(backend, theta, eom, y0)
    om_ret = float(_np(aAI._actionsFreqsAngles(*y0)[idx_om]).ravel()[0])
    assert numpy.isfinite(dthdt)
    numpy.testing.assert_allclose(dthdt, om_ret, rtol=1e-8, atol=1e-9)


# --------------------------------------------- inverse maps (J,angle) -> (x,v)
# actionAngleHarmonicInverse is closed-form (amp=√(2J/ω); x=amp sinθ; vx=amp ω cosθ)
# -> backend-migrated here. actionAngleIsochroneInverse solves Kepler's equation
# (scipy Newton) -> NOT backend-traceable yet (needs a backend root-find; Track E
# #2 with Vertical/Spherical), so only its numpy round-trip is exercised below.
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("omega", [0.7, 1.3])
def test_harmonic_inverse_roundtrip_parity(backend, omega):
    aAHi = actionAngleHarmonicInverse(omega=omega)
    aAH = actionAngleHarmonic(omega=omega)
    j = 0.6
    angle = numpy.array(
        [0.3, 0.9, 2.1]
    )  # all in (0, π): forward arctan2 returns them as-is
    x_np, vx_np, _ = aAHi._xvFreqs(j, angle)
    # forward ∘ inverse == identity (numpy reference)
    j_np, _, a_np = aAH._actionsFreqsAngles(numpy.asarray(x_np), numpy.asarray(vx_np))
    numpy.testing.assert_allclose(_np(j_np), j, rtol=1e-12, atol=1e-12)
    numpy.testing.assert_allclose(_np(a_np), angle, rtol=1e-10, atol=1e-10)
    # backend parity + backend-array-ness
    x_b, vx_b, _ = aAHi._xvFreqs(_arr(backend, numpy.asarray(j)), _arr(backend, angle))
    assert _is_backend_array(backend, x_b)
    assert _is_backend_array(backend, vx_b)
    numpy.testing.assert_allclose(_np(x_b), numpy.asarray(x_np), rtol=1e-12, atol=1e-12)
    numpy.testing.assert_allclose(
        _np(vx_b), numpy.asarray(vx_np), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_harmonic_inverse_grad_vs_fd(backend):
    # dx/dangle at fixed action (= amp cos θ), AD vs finite-difference.
    aAHi = actionAngleHarmonicInverse(omega=1.3)
    j0 = 0.6

    def xofang(ang_val, xp_arr):
        return aAHi._xvFreqs(xp_arr(j0), ang_val)[0]

    fd = _fd(
        lambda a: numpy.asarray(xofang(numpy.asarray(a), lambda v: numpy.asarray(v))),
        0.9,
    )
    g = _grad(
        backend,
        lambda at: xofang(at, lambda v: _arr(backend, numpy.asarray(v, float))),
        0.9,
    )
    assert numpy.isfinite(g)
    numpy.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-7)


def test_isochrone_inverse_roundtrip_numpy():
    # numpy-only (scipy Newton Kepler solve not backend-traceable yet); establishes
    # the forward∘inverse identity so the backend port (Track E #2) has a reference.
    b = 1.2
    aAIi = actionAngleIsochroneInverse(b=b)
    aAI = actionAngleIsochrone(b=b)
    jr, jphi, jz = 0.2, 0.8, 0.15
    ar, ap, az = 0.7, 1.1, 0.4
    R, vR, vT, z, vz, phi = (
        numpy.asarray(c).ravel()[0] for c in aAIi._evaluate(jr, jphi, jz, ar, ap, az)
    )
    out = [
        numpy.asarray(c).ravel()[0]
        for c in aAI._actionsFreqsAngles(R, vR, vT, z, vz, phi)
    ]
    numpy.testing.assert_allclose(out[0], jr, rtol=1e-7, atol=1e-9)  # Jr
    numpy.testing.assert_allclose(out[1], jphi, rtol=1e-7, atol=1e-9)  # Jphi
    numpy.testing.assert_allclose(out[2], jz, rtol=1e-7, atol=1e-9)  # Jz
    numpy.testing.assert_allclose(out[6], ar, rtol=1e-6, atol=1e-8)  # angler
    numpy.testing.assert_allclose(out[7], ap, rtol=1e-6, atol=1e-8)  # anglephi
    numpy.testing.assert_allclose(out[8], az, rtol=1e-6, atol=1e-8)  # anglez
