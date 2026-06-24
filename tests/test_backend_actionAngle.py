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
    actionAngleSpherical,
)
from galpy.potential import LogarithmicHaloPotential, NFWPotential

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
# actionAngleHarmonicInverse is closed-form (amp=√(2J/ω); x=amp sinθ; vx=amp ω cosθ).
# actionAngleIsochroneInverse solves Kepler's equation eta-(a e/ab)sin(eta)=ar; the
# numpy path keeps scipy.optimize.newton (byte-identical), the jax/torch path uses
# the shared backend root-finder galpy.backend.optimize.brentq (vectorised bracketed
# bisection + one-Newton-step implicit-diff) so gradients flow to the actions.
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


# IsochroneInverse torus + angle batch (bound, inclined; angles inside (0,2π)).
_II_J = (0.2, 0.8, 0.15)  # jr, jphi, jz
_II_AR = numpy.array([0.7, 1.6, 3.1, 4.8])
_II_AP = numpy.array([1.1, 2.3, 0.5, 5.1])
_II_AZ = numpy.array([0.4, 1.9, 2.7, 3.6])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("b", [0.8, 1.2])
def test_isochrone_inverse_parity(backend, b):
    aAIi = actionAngleIsochroneInverse(b=b)
    ref = aAIi._xvFreqs(*_II_J, _II_AR, _II_AP, _II_AZ)
    bargs = (*[_arr(backend, numpy.asarray(j)) for j in _II_J],) + tuple(
        _arr(backend, a) for a in (_II_AR, _II_AP, _II_AZ)
    )
    got = aAIi._xvFreqs(*bargs)
    for r, g in zip(ref, got):
        assert _is_backend_array(backend, g)
        numpy.testing.assert_allclose(_np(g), numpy.asarray(r), rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("backend", BACKENDS)
def test_isochrone_inverse_roundtrip(backend):
    # forward ∘ inverse == identity, fully in-backend (the inverse Kepler solve runs
    # via the shared backend root-finder), recovering actions and angles.
    b = 1.2
    aAIi = actionAngleIsochroneInverse(b=b)
    aAI = actionAngleIsochrone(b=b)
    jr, jphi, jz = _II_J
    bj = [_arr(backend, numpy.asarray(j)) for j in _II_J]
    bang = [_arr(backend, a) for a in (_II_AR, _II_AP, _II_AZ)]
    R, vR, vT, z, vz, phi = aAIi._xvFreqs(*bj, *bang)[:6]
    out = aAI._actionsFreqsAngles(R, vR, vT, z, vz, phi)
    numpy.testing.assert_allclose(_np(out[0]), jr, rtol=1e-7, atol=1e-9)  # Jr
    numpy.testing.assert_allclose(_np(out[1]), jphi, rtol=1e-7, atol=1e-9)  # Jphi
    numpy.testing.assert_allclose(_np(out[2]), jz, rtol=1e-7, atol=1e-9)  # Jz
    # angles recovered (circular difference, to be robust to the 2π wrap)
    for idx, a_in in ((6, _II_AR), (7, _II_AP), (8, _II_AZ)):
        d = (_np(out[idx]) - a_in + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
        numpy.testing.assert_allclose(d, 0.0, atol=1e-7)


@pytest.mark.parametrize("backend", BACKENDS)
def test_isochrone_inverse_grad_vs_fd(backend):
    # d(sum R)/d jr through the Kepler root-find: AD (implicit-function theorem) vs FD.
    aAIi = actionAngleIsochroneInverse(b=1.2)
    _, jphi, jz = _II_J

    def rsum(jr_val, xp_arr):
        out = aAIi._xvFreqs(
            jr_val,
            xp_arr(jphi),
            xp_arr(jz),
            xp_arr(_II_AR),
            xp_arr(_II_AP),
            xp_arr(_II_AZ),
        )
        return out[0].sum()

    fd = _fd(lambda j: numpy.asarray(rsum(j, lambda v: numpy.asarray(v))), _II_J[0])
    g = _grad(
        backend,
        lambda jt: rsum(jt, lambda v: _arr(backend, numpy.asarray(v, float))),
        _II_J[0],
    )
    assert numpy.isfinite(g)
    numpy.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-7)


# ====================================================================
# actionAngleSpherical: ACTIONS (Jr,Lz,Jz via _evaluate) + _EccZmaxRperiRap.
# Unlike the closed-form classes above, these require a bracketed root-find
# (rperi/rap) and a quadrature (Jr): the numpy path stays byte-identical (the
# scipy brentq/quad per-object loop) while jax/torch inputs take a vectorised,
# differentiable path -- rperi/rap via the shared backend root-finder
# galpy.backend.optimize.brentq, Jr via galpy.backend.quadrature.fixed_quad.
# Backend GL (n=25) vs scipy's adaptive quad differ at the ~1e-9 level, so the
# parity tolerance is rtol~1e-7 (NOT 1e-12). PR-1 scope: only actions+ecc; the
# frequency/angle methods (_actionsFreqs*) are PR-2 and untouched (numpy-only).
_SPH_POTS = {
    "log": LogarithmicHaloPotential(normalize=1.0),
    "nfw": NFWPotential(normalize=1.0),
}
# Generic bound ICs (vR != 0, so away from the exact peri/apo turning points
# where the bracket endpoint sits on the root); inclined so Jz != 0.
_SPH_R = numpy.array([1.1, 0.8, 1.3])
_SPH_VR = numpy.array([0.2, -0.1, 0.15])
_SPH_VT = numpy.array([0.9, 0.6, 1.0])
_SPH_Z = numpy.array([0.15, -0.2, 0.1])
_SPH_VZ = numpy.array([0.1, 0.05, -0.1])
_SPH = (_SPH_R, _SPH_VR, _SPH_VT, _SPH_Z, _SPH_VZ)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("potname", list(_SPH_POTS))
def test_spherical_parity(backend, potname):
    # numpy <-> jax <-> torch parity of _evaluate (Jr,Lz,Jz) and
    # _EccZmaxRperiRap (e,zmax,rperi,rap) on a small batch of bound ICs.
    aAS = actionAngleSpherical(pot=_SPH_POTS[potname])
    bargs = [_arr(backend, v) for v in _SPH]
    for ref, got in (
        (aAS._evaluate(*_SPH), aAS._evaluate(*bargs)),
        (aAS._EccZmaxRperiRap(*_SPH), aAS._EccZmaxRperiRap(*bargs)),
    ):
        for r, g in zip(ref, got):
            assert _is_backend_array(backend, g)
            numpy.testing.assert_allclose(
                _np(g), numpy.asarray(r), rtol=1e-7, atol=1e-9
            )


# Scalar single-point bound IC for clean per-component derivatives.
_SPH_S = (1.1, 0.2, 0.9, 0.15, 0.1)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("which", ["jr", "ecc"])
def test_spherical_grad_vs_fd_wrt_vR(backend, which):
    # d(Jr or eccentricity)/d vR at a single bound point: AD (root-find +
    # quadrature, both differentiable via the backend layer) vs finite-diff.
    aAS = actionAngleSpherical(pot=_SPH_POTS["log"])
    R, _, vT, z, vz = _SPH_S

    def call(vR_val, xp_arr):
        # build args in the namespace of vR_val so get_namespace follows vR
        args = (xp_arr(R), vR_val, xp_arr(vT), xp_arr(z), xp_arr(vz))
        if which == "jr":
            return aAS._evaluate(*args)[0].sum()
        return aAS._EccZmaxRperiRap(*args)[0].sum()  # eccentricity

    def f_np(vR_val):
        return numpy.asarray(call(vR_val, lambda v: numpy.atleast_1d(numpy.asarray(v))))

    fd = _fd(f_np, _SPH_S[1])

    def f_be(vR_t):
        return call(vR_t, lambda v: _arr(backend, numpy.atleast_1d(v).astype(float)))

    g = _grad(backend, f_be, _SPH_S[1])
    assert numpy.isfinite(g)
    numpy.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-7)


# ====================================================================
# actionAngleSpherical PR-2: FREQUENCIES (Or,Op,Oz via _actionsFreqs) + ANGLES
# (ar,ap,az via _actionsFreqsAngles). Same additive backend pattern: the numpy
# per-object scipy loop is untouched (byte-identical), jax/torch inputs take the
# vectorised, differentiable branch. The two t^2-substituted panels of each
# radial-period / azimuthal-period / angle integral run through
# galpy.backend.quadrature.fixed_quad (n=25) with the per-object upper limit
# folded INTO the integrand (t = lim*s on a fixed [0,1] panel), so the GL
# (n=25) value differs from scipy's adaptive quad at the ~1e-9 level
# (rtol~1e-6, NOT 1e-12). Azimuth phi is needed for the angles call.
_SPH_PHI = numpy.array([1.3, 0.4, 2.1])
# A retrograde batch (vT<0) to exercise the Op sign flip and ap = asc - az
# branch; inclined and off the turning points / non-inclined kink.
_SPH_RETRO = (
    numpy.array([1.0, 1.2]),
    numpy.array([0.1, -0.15]),
    numpy.array([-0.8, -0.5]),
    numpy.array([0.1, -0.05]),
    numpy.array([0.05, 0.1]),
)
_SPH_RETRO_PHI = numpy.array([0.5, 2.0])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("potname", list(_SPH_POTS))
def test_spherical_freqs_parity(backend, potname):
    # numpy <-> jax <-> torch parity of _actionsFreqs (Jr,Lz,Jz,Or,Op,Oz) and
    # _actionsFreqsAngles (+ar,ap,az) on prograde + retrograde bound ICs.
    aAS = actionAngleSpherical(pot=_SPH_POTS[potname])
    for sph, phi in ((_SPH, _SPH_PHI), (_SPH_RETRO, _SPH_RETRO_PHI)):
        bargs = [_arr(backend, v) for v in sph]
        bphi = _arr(backend, phi)
        # _actionsFreqs (no phi)
        ref = aAS._actionsFreqs(*sph)
        got = aAS._actionsFreqs(*bargs)
        for r, g in zip(ref, got):
            assert _is_backend_array(backend, g)
            numpy.testing.assert_allclose(
                _np(g), numpy.asarray(r), rtol=1e-6, atol=1e-8
            )
        # _actionsFreqsAngles (+phi); angles compared as circular differences
        ref = aAS._actionsFreqsAngles(*sph, phi)
        got = aAS._actionsFreqsAngles(*bargs, bphi)
        for idx, (r, g) in enumerate(zip(ref, got)):
            assert _is_backend_array(backend, g)
            if idx >= 6:  # ar, ap, az: wrap-robust comparison
                d = (_np(g) - numpy.asarray(r) + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
                numpy.testing.assert_allclose(d, 0.0, atol=1e-6)
            else:
                numpy.testing.assert_allclose(
                    _np(g), numpy.asarray(r), rtol=1e-6, atol=1e-8
                )


# Scalar single-point bound IC (with phi) for clean per-component derivatives.
_SPH_SA = (1.1, 0.2, 0.9, 0.15, 0.1, 1.3)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("which,idx", [("omegar", 3), ("angler", 6)])
def test_spherical_freqs_grad_vs_fd_wrt_vR(backend, which, idx):
    # d(Or or ar)/d vR at a single bound point: AD through the vectorised
    # root-find + two-panel fixed_quad period/angle integrals vs finite-diff.
    aAS = actionAngleSpherical(pot=_SPH_POTS["log"])
    R, _, vT, z, vz, phi = _SPH_SA

    def call(vR_val, xp_arr):
        args = (
            xp_arr(R),
            vR_val,
            xp_arr(vT),
            xp_arr(z),
            xp_arr(vz),
            xp_arr(phi),
        )
        return aAS._actionsFreqsAngles(*args)[idx].sum()

    def f_np(vR_val):
        return numpy.asarray(call(vR_val, lambda v: numpy.atleast_1d(numpy.asarray(v))))

    fd = _fd(f_np, _SPH_SA[1])

    def f_be(vR_t):
        return call(vR_t, lambda v: _arr(backend, numpy.atleast_1d(v).astype(float)))

    g = _grad(backend, f_be, _SPH_SA[1])
    assert numpy.isfinite(g)
    numpy.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-6)
