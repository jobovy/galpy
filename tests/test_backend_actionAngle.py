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
    actionAngleAdiabatic,
    actionAngleAdiabaticGrid,
    actionAngleHarmonic,
    actionAngleHarmonicInverse,
    actionAngleIsochrone,
    actionAngleIsochroneApprox,
    actionAngleIsochroneInverse,
    actionAngleSpherical,
    actionAngleStaeckel,
    actionAngleStaeckelGrid,
    actionAngleVertical,
    estimateBIsochrone,
    estimateDeltaStaeckel,
)
from galpy.potential import (
    HernquistPotential,
    IsochronePotential,
    IsothermalDiskPotential,
    KGPotential,
    LogarithmicHaloPotential,
    MiyamotoNagaiPotential,
    MWPotential2014,
    NFWPotential,
    toPlanarPotential,
    toVerticalPotential,
    vcirc,
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


def _hess(backend, f, x0):
    # second derivative of a scalar->scalar f at x0 (jax.hessian / torch double-grad)
    if backend == "jax":
        return float(jax.hessian(lambda t: f(t))(jnp.asarray(x0)))
    t = torch.tensor(x0, requires_grad=True)
    (g,) = torch.autograd.grad(f(t), t, create_graph=True)
    (h,) = torch.autograd.grad(g, t)
    return float(h)


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
# Backend GL (n=50) vs scipy's adaptive quad differ at the ~1e-9 level, so the
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
# galpy.backend.quadrature.fixed_quad (n=50) with the per-object upper limit
# folded INTO the integrand (t = lim*s on a fixed [0,1] panel), so the GL
# (n=50) value differs from scipy's adaptive quad at the ~1e-9 level
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


# ====================================================================
# actionAngleSpherical EDGE-CASE parity. The spherical action/freq/angle map
# has data-dependent branches that only fire at special parts of phase space:
#   * at peri / at apo (vr=0): the radial-period bracket endpoint sits ON the
#     turning-point root, so the panel limits collapse to 0;
#   * exactly circular (Jr<1e-9): the epifreq/omegac fast branch;
#   * z=0 with vz!=0 (plane-crossing but inclined): sin(psi)=0 but finite;
#   * z=0 AND vz=0 (non-inclined, planar): numpy's z/r/sin(i) is non-finite, so
#     psi=phi and the longitude of the ascending node is 0 by convention -- the
#     branch the backend must reproduce EXACTLY (no xfall-through to psi=0);
#   * near-radial (L/Lcirc~3e-3, deeply plunging) and near-circular (vT just off
#     vcirc): the quadrature stress cases.
# Every case must match numpy on ALL of (Jr,Lz,Jz,Or,Op,Oz,ar,ap,az,e,zmax,
# rperi,rap), prograde AND retrograde, across Log/NFW/Hernquist/Isochrone. The
# only non-byte-identity is the inherent GL(n=50)-vs-adaptive-quad floor (~1e-6
# on the continuous freq/angle integrals; actions/ecc/peri/apo are ~1e-9). At
# pathological L/Lcirc<~1e-4 scipy's OWN adaptive quad fails to ~1e-5, so that
# extreme is excluded as an irreducible quadrature-method difference (not a
# logic/convention divergence).
_EDGE_POTS = {
    "log": LogarithmicHaloPotential(normalize=1.0),
    "nfw": NFWPotential(normalize=1.0),
    "hernquist": HernquistPotential(normalize=1.0),
    "isochrone": IsochronePotential(normalize=1.0),
}
_EDGE_CASES = [
    "apocenter",
    "pericenter",
    "circular",
    "zcross_incl",
    "planar",
    "near_radial",
    "very_ecc",
    "near_circular",
]


def _edge_ic(pot, case, prograde):
    """(R,vR,vT,z,vz,phi) for a spherical action-angle edge case."""
    sgn = 1.0 if prograde else -1.0
    phi = 0.7
    R = 1.0
    vc = vcirc(pot, R, use_physical=False)
    if case == "apocenter":  # vr=0, sub-circular vt, inclined -> r is apo
        R, z = 0.9, 0.4
        r = numpy.sqrt(R**2 + z**2)
        vt = 0.6 * vcirc(pot, r, use_physical=False)
        return (R, 0.0, sgn * vt * r / R, z, 0.0, phi)
    if case == "pericenter":  # vr=0, super-circular vt, inclined -> r is peri
        R, z = 0.9, 0.4
        r = numpy.sqrt(R**2 + z**2)
        vt = 1.4 * vcirc(pot, r, use_physical=False)
        return (R, 0.0, sgn * vt * r / R, z, 0.0, phi)
    if case == "circular":  # z=0,vz=0,vR=0,vT=vcirc -> Jr=0, non-inclined
        return (1.1, 0.0, sgn * vcirc(pot, 1.1, use_physical=False), 0.0, 0.0, phi)
    if case == "zcross_incl":  # z=0 but vz!=0: plane-crossing, inclined
        return (R, 0.15, sgn * 0.8 * vc, 0.0, 0.3, phi)
    if case == "planar":  # z=0 AND vz=0: non-inclined (the psi=phi branch)
        return (R, 0.2, sgn * 0.7 * vc, 0.0, 0.0, phi)
    if case == "near_radial":  # L/Lcirc~3e-3: deeply plunging (physical)
        return (R, 0.6 * vc, sgn * 3e-3 * vc, 0.0, 0.05, phi)
    if case == "very_ecc":  # bound, high ecc, inclined
        return (R, 0.6 * vc, sgn * 0.18 * vc, 0.0, 0.05, phi)
    if case == "near_circular":  # vT just off vcirc -> small Jr, non-inclined
        return (R, 0.0, sgn * 1.002 * vc, 0.0, 0.0, phi)
    raise ValueError(case)  # pragma: no cover


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("potname", list(_EDGE_POTS))
@pytest.mark.parametrize("case", _EDGE_CASES)
@pytest.mark.parametrize("prograde", [True, False])
def test_spherical_edge_case_parity(backend, potname, case, prograde):
    # numpy <-> backend parity of _actionsFreqsAngles AND _EccZmaxRperiRap at
    # the spherical edge cases (peri/apo, circular, z-crossing, non-inclined,
    # near-radial, near-circular), prograde + retrograde. The non-inclined cases
    # exercise the psi=phi / Omega=0 conventions the backend must match exactly.
    import warnings

    pot = _EDGE_POTS[potname]
    aAS = actionAngleSpherical(pot=pot)
    R, vR, vT, z, vz, phi = _edge_ic(pot, case, prograde)
    npargs = tuple(
        numpy.atleast_1d(numpy.asarray(float(v))) for v in (R, vR, vT, z, vz)
    )
    npphi = numpy.atleast_1d(numpy.asarray(float(phi)))
    bargs = [_arr(backend, numpy.asarray(float(v))[None]) for v in (R, vR, vT, z, vz)]
    bphi = _arr(backend, numpy.asarray(float(phi))[None])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref_afa = aAS._actionsFreqsAngles(*npargs, npphi)
        got_afa = aAS._actionsFreqsAngles(*bargs, bphi)
        ref_ecc = aAS._EccZmaxRperiRap(*npargs)
        got_ecc = aAS._EccZmaxRperiRap(*bargs)
    # (Jr,Lz,Jz,Or,Op,Oz,ar,ap,az): values to ~1e-6 (GL-vs-adaptive floor);
    # angles (idx>=6) compared as wrap-robust circular differences.
    for idx, (r, g) in enumerate(zip(ref_afa, got_afa)):
        assert _is_backend_array(backend, g)
        if idx >= 6:
            d = (_np(g) - numpy.asarray(r) + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
            numpy.testing.assert_allclose(
                d, 0.0, atol=2e-6, err_msg=f"{case}/{potname}/angle{idx}"
            )
        else:
            numpy.testing.assert_allclose(
                _np(g),
                numpy.asarray(r),
                rtol=2e-6,
                atol=2e-6,
                err_msg=f"{case}/{potname}/afa{idx}",
            )
    # (e,zmax,rperi,rap): root-find + action quadrature, ~1e-9.
    for idx, (r, g) in enumerate(zip(ref_ecc, got_ecc)):
        assert _is_backend_array(backend, g)
        numpy.testing.assert_allclose(
            _np(g),
            numpy.asarray(r),
            rtol=1e-7,
            atol=1e-9,
            err_msg=f"{case}/{potname}/ecc{idx}",
        )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("idx,name", [(7, "ap"), (8, "az")])
def test_spherical_noninclined_angle_grad_vs_fd(backend, idx, name):
    # d(ap or az)/d vR for a NON-INCLINED (z=0, vz=0) orbit. numpy's psi and
    # longitude-of-ascending-node use z/r/sin(i) and z/R/tan(i), both 0/0 (sin
    # i==0) -> non-finite -> psi=phi / Omega=0. The backend reproduces those
    # conventions via xp.where on a finiteness mask computed from the TRUE
    # (un-safed) division, so NO NaN reaches the where value/grad path: the
    # gradient stays finite and matches finite-difference (perturbing vR keeps
    # the orbit in-plane, so the angle is smooth there). Guards against the
    # xp.where dead-branch NaN-poisoning that a naive psi=phi guard reintroduces.
    aAS = actionAngleSpherical(pot=_EDGE_POTS["log"])
    pot = _EDGE_POTS["log"]
    vc = vcirc(pot, 1.0, use_physical=False)
    R, vT, z, vz, phi = 1.0, 0.7 * vc, 0.0, 0.0, 0.7  # non-inclined, az mid-range

    def call(vR_val, xp_arr):
        args = (xp_arr(R), vR_val, xp_arr(vT), xp_arr(z), xp_arr(vz), xp_arr(phi))
        return aAS._actionsFreqsAngles(*args)[idx].sum()

    def f_np(vR_val):
        return numpy.asarray(call(vR_val, lambda v: numpy.atleast_1d(numpy.asarray(v))))

    fd = _fd(f_np, 0.2)

    def f_be(vR_t):
        return call(vR_t, lambda v: _arr(backend, numpy.atleast_1d(v).astype(float)))

    g = _grad(backend, f_be, 0.2)
    assert numpy.isfinite(g), f"{name} grad is NaN at non-inclined point ({backend})"
    numpy.testing.assert_allclose(g, fd, rtol=1e-4, atol=1e-6)


# ====================================================================
# actionAngleVertical: 1D vertical action-angle (J, Omega, angle). The numpy
# per-object scipy loop (brentq for xmax + adaptive quad) is untouched (byte-
# identical); jax/torch inputs take a vectorised, differentiable branch -- xmax
# via the shared backend.optimize.brentq, the J/Omega/angle integrals via
# backend.quadrature.fixed_quad with the x = xmax - t^2 turning-point
# substitution. The 1D analog of actionAngleSpherical's radial Jr/Or/ar.
_VERT_POTS = {
    "isodisk": IsothermalDiskPotential(amp=1.0, sigma=0.5),
    "kg": KGPotential(),
    "vertMN": toVerticalPotential(MiyamotoNagaiPotential(normalize=1.0), 1.1),
}
# Generic + every edge: midplane (x=0), turning point (vx=0), near-turn, large
# amplitude, and all four (x,vx)-sign quadrants for the angle assembly.
_VERT_X = numpy.array([0.1, -0.2, 0.3, 0.0, 0.3, 0.05, 0.15, -0.15, -0.15])
_VERT_VX = numpy.array([0.2, 0.15, -0.1, 0.2, 0.0, 1.5, -0.3, -0.3, 0.3])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("potname", list(_VERT_POTS))
def test_vertical_parity(backend, potname):
    # numpy <-> jax <-> torch parity of _evaluate (J), _actionsFreqs (J,Omega),
    # and _actionsFreqsAngles (J,Omega,angle) over generic + edge-case ICs.
    aAV = actionAngleVertical(pot=_VERT_POTS[potname])
    bx, bvx = _arr(backend, _VERT_X), _arr(backend, _VERT_VX)
    for ref, got in (
        (aAV._evaluate(_VERT_X, _VERT_VX), aAV._evaluate(bx, bvx)),
        (aAV._actionsFreqs(_VERT_X, _VERT_VX), aAV._actionsFreqs(bx, bvx)),
        (aAV._actionsFreqsAngles(_VERT_X, _VERT_VX), aAV._actionsFreqsAngles(bx, bvx)),
    ):
        ref = ref if isinstance(ref, tuple) else (ref,)
        got = got if isinstance(got, tuple) else (got,)
        for idx, (r, g) in enumerate(zip(ref, got)):
            assert _is_backend_array(backend, g)
            if len(ref) == 3 and idx == 2:  # angle: wrap-robust
                d = (_np(g) - numpy.asarray(r) + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
                numpy.testing.assert_allclose(d, 0.0, atol=1e-6)
            else:
                numpy.testing.assert_allclose(
                    _np(g), numpy.asarray(r), rtol=1e-6, atol=1e-8
                )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("which,idx", [("J", 0), ("Omega", 1), ("angle", 2)])
def test_vertical_grad_vs_fd_wrt_vx(backend, which, idx):
    # d(J/Omega/angle)/d vx at a single bound point: AD (vectorised root-find +
    # fixed_quad, both differentiable via the backend layer) vs finite-diff.
    aAV = actionAngleVertical(pot=_VERT_POTS["isodisk"])
    x0, vx0 = 0.15, 0.2

    def call(vx_val, xp_arr):
        return aAV._actionsFreqsAngles(xp_arr(x0), vx_val)[idx].sum()

    def f_np(vx_val):
        # x and vx both floats -> the numpy path wraps them to 1-element arrays
        return numpy.asarray(call(vx_val, lambda v: v))

    fd = _fd(f_np, vx0)

    def f_be(vx_t):
        return call(vx_t, lambda v: _arr(backend, numpy.atleast_1d(v).astype(float)))

    g = _grad(backend, f_be, vx0)
    assert numpy.isfinite(g)
    numpy.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-6)


# ----------------------------------------------------- Staeckel actions (PR-1)
# Vectorised C-consistent actions (jr,Lz,jz) under the backends. The numpy and
# jax/torch paths run the SAME unified vectorised code (turning points via the
# shared bisect_root, J integrals via fixed_quad at the C GL order); validate
# numpy<->backend value parity and consistency with the C path (c=True).
# NOTE: accurate action GRADIENTS need the substitution-regularised derivative
# integrands (the dJ/dE,dLz,dI3 Leibniz rules) and land with the frequencies in
# the next Staeckel PR -- d(sqrt S)/dparam ~ 1/sqrt(S) is singular at the turning
# points, so AD through the plain-GL value is only GL-order accurate. Hence this
# module asserts VALUE parity only for Staeckel.
_STK = (
    numpy.array([0.9, 1.1, 0.7]),
    numpy.array([0.1, -0.2, 0.05]),
    numpy.array([1.05, 0.9, 1.2]),
    numpy.array([0.2, -0.1, 0.3]),
    numpy.array([0.08, 0.2, -0.1]),
)


@pytest.mark.parametrize("backend", BACKENDS)
def test_staeckel_actions_parity(backend):
    aA = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=False)
    ref = aA(*_STK)
    got = aA(*[_arr(backend, v) for v in _STK])
    for r, g in zip(ref, got):
        assert _is_backend_array(backend, g)
        numpy.testing.assert_allclose(_np(g), numpy.asarray(r), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_staeckel_actions_vs_c(backend):
    # the unified vectorised (numpy + backend) GL actions match the C path
    aF = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=False)
    aC = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=True)
    jr_c, lz_c, jz_c = aC(*_STK)
    jr_b, lz_b, jz_b = aF(*[_arr(backend, v) for v in _STK])
    numpy.testing.assert_allclose(_np(jr_b), numpy.asarray(jr_c), rtol=1e-8, atol=1e-9)
    numpy.testing.assert_allclose(_np(jz_b), numpy.asarray(jz_c), rtol=1e-8, atol=1e-9)


@pytest.mark.skipif("jax" not in BACKENDS, reason="jax not installed")
def test_staeckel_jit_grad_rolls_direct_bisection():
    # Staeckel's turning points call the module-level ``bisect_root`` DIRECTLY
    # (not via ``brentq``). Under a jax trace (jit/grad) the bracket endpoints are
    # tracers, so ``bisect_root`` dispatches to the rolled ``lax.fori_loop`` kernel
    # -- the ~100-step bisection does NOT unroll ~100 copies of the Staeckel
    # integrand into the user's jaxpr. Covers the ``under_jax_trace(a, b)`` branch
    # of ``galpy.backend.optimize.bisect_root`` reached only by these direct
    # callers (the brentq-based AA methods route through a different kernel).
    aA = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=False)
    R = jnp.asarray(_STK[0])
    rest = tuple(jnp.asarray(v) for v in _STK[1:])
    jr_e, _, jz_e = aA(R, *rest)  # eager (Python-loop) reference
    jr_j, _, jz_j = jax.jit(lambda r: aA(r, *rest))(R)  # traced (fori_loop) value
    numpy.testing.assert_allclose(_np(jr_j), _np(jr_e), rtol=1e-8, atol=1e-10)
    numpy.testing.assert_allclose(_np(jz_j), _np(jz_e), rtol=1e-8, atol=1e-10)
    # the jaxpr is ROLLED: a loop primitive, not ~100 unrolled bisection steps.
    txt = str(jax.make_jaxpr(lambda r: aA(r, *rest)[0])(R))
    assert ("while" in txt) or ("scan" in txt)
    # grad flows through the direct-bisection turning points: finite dJr/dR.
    g = jax.grad(lambda r: jnp.sum(aA(r, *rest)[0]))(R)
    assert numpy.all(numpy.isfinite(_np(g)))


@pytest.mark.parametrize("backend", BACKENDS)
def test_staeckel_unbound_backend_no_raise(backend):
    # An unbound orbit raises UnboundError on the numpy path (eager), but must NOT
    # raise under a backend: the vectorised turning-point search cannot branch on
    # the traced `unbound` mask under jit, so it falls through to a (garbage)
    # backend array instead -- which is exactly what keeps actionsFreqsAngles
    # jax.jit-traceable (jit traces+matches eager to ~9e-10). Exercises both sides
    # of the `not is_backend_array(R)` guard in _staeckel_prep.
    from galpy.actionAngle import UnboundError

    aA = actionAngleStaeckel(pot=MWPotential2014, delta=0.5, c=False)
    ub = (0.9, 10.0, -20.0, 0.1, 10.0)
    with pytest.raises(UnboundError):
        aA(*ub)  # numpy path: eager raise
    out = aA(*[_arr(backend, numpy.atleast_1d(v).astype(float)) for v in ub])
    assert _is_backend_array(backend, out[0])  # backend: no raise (garbage value ok)


@pytest.mark.parametrize("backend", BACKENDS)
def test_staeckel_actionsfreqs_parity(backend):
    # vectorised c=False (jr,Lz,jz,Or,Op,Oz) numpy<->backend parity + vs c=True
    aF = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=False)
    ref = aF.actionsFreqs(*_STK)
    got = aF.actionsFreqs(*[_arr(backend, v) for v in _STK])
    for r, g in zip(ref, got):
        assert _is_backend_array(backend, g)
        numpy.testing.assert_allclose(_np(g), numpy.asarray(r), rtol=1e-9, atol=1e-10)
    aC = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=True)
    refc = aC.actionsFreqs(*_STK)
    for g, c in zip(got, refc):  # freqs match the C path to the GL floor
        numpy.testing.assert_allclose(_np(g), numpy.asarray(c), rtol=1e-7, atol=1e-9)


# A substantial grid of bound orbits (R x vR x vT x z x vz; 768 points) spanning
# eccentric, retrograde (vT<0), high-z, and near-radial regimes -- exercises the
# vectorised turning points + actions/freqs/EccZmaxRperiRap far more broadly than
# a handful of points. (All bound for MWPotential2014 + delta=0.45; verified.)
def _staeckel_grid():
    Rg = numpy.array([0.7, 0.9, 1.1, 1.3])
    vRg = numpy.array([-0.3, -0.1, 0.1, 0.3])
    vTg = numpy.array([-0.8, 0.6, 0.9, 1.2])
    zg = numpy.array([-0.3, -0.1, 0.1, 0.3])
    vzg = numpy.array([-0.2, 0.0, 0.15])
    G = numpy.meshgrid(Rg, vRg, vTg, zg, vzg, indexing="ij")
    return tuple(g.ravel() for g in G)


_STK_GRID = _staeckel_grid()


def test_staeckel_grid_vs_c():
    # vectorised numpy c=False == the C path (c=True) across the whole grid, for
    # actions, frequencies, and EccZmaxRperiRap (the GL-order floor is ~3e-9 on Oz
    # at the most extreme grid points; jr/jz/ecc are machine precision).
    aF = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=False)
    aC = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=True)
    jr_f, lz_f, jz_f, Or_f, Op_f, Oz_f = aF.actionsFreqs(*_STK_GRID)
    jr_c, lz_c, jz_c, Or_c, Op_c, Oz_c = aC.actionsFreqs(*_STK_GRID)
    numpy.testing.assert_allclose(jr_f, jr_c, rtol=1e-7, atol=1e-9)
    numpy.testing.assert_allclose(jz_f, jz_c, rtol=1e-7, atol=1e-9)
    for o_f, o_c in ((Or_f, Or_c), (Op_f, Op_c), (Oz_f, Oz_c)):
        numpy.testing.assert_allclose(o_f, o_c, rtol=1e-6, atol=1e-7)
    ef, zmf, rpf, raf = aF.EccZmaxRperiRap(*_STK_GRID)
    ec, zmc, rpc, rac = aC.EccZmaxRperiRap(*_STK_GRID)
    for a, b in ((ef, ec), (zmf, zmc), (rpf, rpc), (raf, rac)):
        numpy.testing.assert_allclose(a, b, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("backend", BACKENDS)
def test_staeckel_grid_parity(backend):
    # numpy <-> jax/torch parity across the whole grid for actions, freqs, ecc
    # (one vectorised call processes all 768 orbits at once).
    aF = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=False)
    bargs = [_arr(backend, v) for v in _STK_GRID]
    for ref, got in (
        (aF(*_STK_GRID), aF(*bargs)),
        (aF.actionsFreqs(*_STK_GRID), aF.actionsFreqs(*bargs)),
        (aF.EccZmaxRperiRap(*_STK_GRID), aF.EccZmaxRperiRap(*bargs)),
    ):
        for r, g in zip(ref, got):
            assert _is_backend_array(backend, g)
            numpy.testing.assert_allclose(
                _np(g), numpy.asarray(r), rtol=1e-8, atol=1e-9
            )


# Turning-point orbits: vR=vz=0, so the orbit sits exactly AT a radial turning
# point (ux==umin). The OTHER turning point (umax) must be bracketed from
# strictly INSIDE the allowed region (ux+eps, where the J_R integrand>0); a naive
# [ux,hi] bracket has the integrand <0 at both ends and misses a narrow interior
# umax root, returning hi -> corrupting umax, J_R and the eccentricity. The
# generic grid above never sets vR=0, so it cannot catch this -- hence a dedicated
# grid. z!=0 so J_z>0 and the frequencies stay finite & well-posed.
def _staeckel_turningpoint_grid():
    Rg = numpy.array([0.7, 0.9, 1.1, 1.3])
    vTg = numpy.array([0.6, 0.8, 1.0, 1.1])
    zg = numpy.array([0.05, 0.15, 0.3])
    G = numpy.meshgrid(Rg, vTg, zg, indexing="ij")
    R, vT, z = (g.ravel() for g in G)
    return (R, numpy.zeros_like(R), vT, z, numpy.zeros_like(R))


_STK_TURN = _staeckel_turningpoint_grid()


def test_staeckel_turningpoint_vs_c():
    # Regression for the radial-turning-point umin/umax bracketing bug: actions
    # and EccZmaxRperiRap of vR=vz=0 orbits must match the C path to machine
    # precision (a wrong umax silently zeroed J_R and inflated the eccentricity).
    aF = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=False)
    aC = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=True)
    jr_f, lz_f, jz_f = aF(*_STK_TURN)
    jr_c, lz_c, jz_c = aC(*_STK_TURN)
    numpy.testing.assert_allclose(jr_f, jr_c, rtol=1e-7, atol=1e-10)
    numpy.testing.assert_allclose(jz_f, jz_c, rtol=1e-7, atol=1e-10)
    for a, b in zip(aF.EccZmaxRperiRap(*_STK_TURN), aC.EccZmaxRperiRap(*_STK_TURN)):
        numpy.testing.assert_allclose(a, b, rtol=1e-7, atol=1e-10)


@pytest.mark.parametrize("backend", BACKENDS)
def test_staeckel_turningpoint_parity(backend):
    # numpy <-> jax/torch parity of the turning-point actions + ecc.
    aF = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=False)
    bargs = [_arr(backend, v) for v in _STK_TURN]
    ref = tuple(aF(*_STK_TURN)) + tuple(aF.EccZmaxRperiRap(*_STK_TURN))
    got = tuple(aF(*bargs)) + tuple(aF.EccZmaxRperiRap(*bargs))
    for r, g in zip(ref, got):
        assert _is_backend_array(backend, g)
        numpy.testing.assert_allclose(_np(g), numpy.asarray(r), rtol=1e-9, atol=1e-10)


# Exactly-planar orbits (z=vz=0, so J_z=0): the vertical turning point snaps to
# v=pi/2, the J_z derivative panels collapse to zero width, and the frequency
# determinant det(A)=0. The C path then returns Omegar,Omegaphi=NaN and
# Omegaz=Inf (IEEE 0/0 and x/0); the vectorised path must reproduce this EXACTLY
# and deterministically across numpy/jax/torch (the vmin->pi/2 snap makes det(A)
# identically zero rather than a tiny round-off value that would yield finite
# garbage). Omegaz=Inf (not NaN) is load-bearing: it keeps the J_z<1e-3
# frequency substitution from firing, so Omegar stays NaN for the genuinely
# eccentric radial motion instead of being wrongly overwritten with epifreq.
def _staeckel_planar_grid():
    Rg = numpy.array([0.7, 0.9, 1.1, 1.3])
    vRg = numpy.array([-0.2, 0.1, 0.25])
    vTg = numpy.array([0.6, 0.9, 1.1])
    G = numpy.meshgrid(Rg, vRg, vTg, indexing="ij")
    R, vR, vT = (g.ravel() for g in G)
    return (R, vR, vT, numpy.zeros_like(R), numpy.zeros_like(R))


_STK_PLANAR = _staeckel_planar_grid()


def _kind(x):  # 0=NaN, 1=Inf, 2=finite -- the degeneracy class of a value
    return numpy.where(numpy.isnan(x), 0, numpy.where(numpy.isinf(x), 1, 2))


def test_staeckel_planar_freqs_degenerate_vs_c():
    # actions stay finite & correct; the degenerate frequencies match the C path
    # class-for-class (NaN/Inf), and the finite ones agree numerically.
    aF = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=False)
    aC = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=True)
    jr_f, lz_f, jz_f, Or_f, Op_f, Oz_f = aF.actionsFreqs(*_STK_PLANAR)
    jr_c, lz_c, jz_c, Or_c, Op_c, Oz_c = aC.actionsFreqs(*_STK_PLANAR)
    numpy.testing.assert_allclose(jr_f, jr_c, rtol=1e-7, atol=1e-9)
    assert numpy.all(jz_f == 0.0)
    for o_f, o_c in ((Or_f, Or_c), (Op_f, Op_c), (Oz_f, Oz_c)):
        numpy.testing.assert_array_equal(_kind(o_f), _kind(o_c))  # same NaN/Inf/finite
        fin = numpy.isfinite(o_f) & numpy.isfinite(o_c)
        if numpy.any(fin):
            numpy.testing.assert_allclose(o_f[fin], o_c[fin], rtol=1e-6, atol=1e-7)


# Near-axis / purely-radial orbits: small or zero Lz (vT~0). The J_R turning point
# umin sits close to (or exactly AT, for Lz=0) the symmetry axis u=0; the lower
# bracket must descend far enough to straddle it (and detect the axis-reaching
# Lz=0 case -> umin=0). A too-shallow down-expansion collapsed umin to ux,
# silently zeroing J_R and corrupting ecc/freqs; this grid regresses that.
def _staeckel_nearaxis_grid():
    Rg = numpy.array([0.8, 1.0, 1.3])
    vRg = numpy.array([0.2, 0.45])
    vTg = numpy.array([0.0, 1e-4, 3e-4])  # Lz = 0, ~1e-4, ~3e-4
    zg = numpy.array([0.1, 0.28])
    G = numpy.meshgrid(Rg, vRg, vTg, zg, indexing="ij")
    R, vR, vT, z = (g.ravel() for g in G)
    return (R, vR, vT, z, 0.1 * numpy.ones_like(R))


_STK_NEARAXIS = _staeckel_nearaxis_grid()


def test_staeckel_nearaxis_vs_c():
    # Regression for the umin lower-bracket collapse at small/zero Lz: actions and
    # EccZmaxRperiRap of near-axis / purely-radial (vT=0) orbits must match C.
    aF = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=False)
    aC = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=True)
    jr_f, lz_f, jz_f = aF(*_STK_NEARAXIS)
    jr_c, lz_c, jz_c = aC(*_STK_NEARAXIS)
    numpy.testing.assert_allclose(jr_f, jr_c, rtol=1e-7, atol=1e-10)
    numpy.testing.assert_allclose(jz_f, jz_c, rtol=1e-7, atol=1e-10)
    for a, b in zip(
        aF.EccZmaxRperiRap(*_STK_NEARAXIS), aC.EccZmaxRperiRap(*_STK_NEARAXIS)
    ):
        numpy.testing.assert_allclose(a, b, rtol=1e-7, atol=1e-10)


@pytest.mark.parametrize("backend", BACKENDS)
def test_staeckel_nearaxis_parity(backend):
    # numpy <-> jax/torch parity of the near-axis actions + ecc.
    aF = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=False)
    bargs = [_arr(backend, v) for v in _STK_NEARAXIS]
    ref = tuple(aF(*_STK_NEARAXIS)) + tuple(aF.EccZmaxRperiRap(*_STK_NEARAXIS))
    got = tuple(aF(*bargs)) + tuple(aF.EccZmaxRperiRap(*bargs))
    for r, g in zip(ref, got):
        assert _is_backend_array(backend, g)
        numpy.testing.assert_allclose(_np(g), numpy.asarray(r), rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("backend", BACKENDS)
def test_staeckel_planar_freqs_degenerate_parity(backend):
    # the NaN/Inf degeneracy is identical across numpy and the backends (so the
    # vmin->pi/2 snap is genuinely backend-deterministic, not round-off-dependent).
    aF = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=False)
    ref = aF.actionsFreqs(*_STK_PLANAR)
    got = aF.actionsFreqs(*[_arr(backend, v) for v in _STK_PLANAR])
    for r, g in zip(ref, got):
        assert _is_backend_array(backend, g)
        numpy.testing.assert_array_equal(_kind(numpy.asarray(r)), _kind(_np(g)))
        fin = numpy.isfinite(numpy.asarray(r)) & numpy.isfinite(_np(g))
        if numpy.any(fin):
            numpy.testing.assert_allclose(
                _np(g)[fin], numpy.asarray(r)[fin], rtol=1e-8, atol=1e-9
            )


# Angles: the vectorized actionsFreqsAngles quadrant tree (4 leaves in u, 8 in v)
# over the whole 768-orbit grid, with per-orbit azimuths so anglephi exercises the
# +phi fold. The grid spans both pux/pvx signs, vx </> pi/2, and both panels, so
# all 12 leaves are hit (verified: each leaf agrees with C to <1e-8).
_STK_GRID_PHI = numpy.linspace(0.3, 5.9, _STK_GRID[0].size)


def _wrapdiff(a, b):  # smallest |a-b| modulo 2pi, elementwise
    d = numpy.abs((numpy.asarray(a) - numpy.asarray(b)) % (2.0 * numpy.pi))
    return numpy.minimum(d, 2.0 * numpy.pi - d)


def test_staeckel_grid_angles_vs_c():
    # the vectorized c=False angles match the C calcAnglesStaeckel across the grid.
    aF = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=False)
    aC = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=True)
    rF = aF.actionsFreqsAngles(*_STK_GRID, _STK_GRID_PHI)
    rC = aC.actionsFreqsAngles(*_STK_GRID, _STK_GRID_PHI)
    for i in (6, 7, 8):  # angler, anglephi, anglez
        f, c = numpy.asarray(rF[i]), numpy.asarray(rC[i])
        fin = numpy.isfinite(f) & numpy.isfinite(c)
        assert numpy.all(fin)
        assert numpy.max(_wrapdiff(f[fin], c[fin])) < 1e-6


@pytest.mark.parametrize("backend", BACKENDS)
def test_staeckel_grid_angles_parity(backend):
    # numpy <-> jax/torch parity of the full actionsFreqsAngles across the grid.
    aF = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=False)
    bargs = [_arr(backend, v) for v in (_STK_GRID + (_STK_GRID_PHI,))]
    ref = aF.actionsFreqsAngles(*_STK_GRID, _STK_GRID_PHI)
    got = aF.actionsFreqsAngles(*bargs)
    for i, (r, g) in enumerate(zip(ref, got)):
        assert _is_backend_array(backend, g)
        if i >= 6:  # angles: wrap-aware
            assert numpy.max(_wrapdiff(_np(g), numpy.asarray(r))) < 1e-8
        else:
            numpy.testing.assert_allclose(
                _np(g), numpy.asarray(r), rtol=1e-8, atol=1e-9
            )


@pytest.mark.parametrize("backend", BACKENDS)
def test_staeckel_angles_numpy_phi(backend):
    # backend R/v but a PLAIN-NUMPY phi: actionsFreqsAngles coerces the azimuth
    # into R's namespace so anglephi += phi works under jax/torch. Exercises the
    # mixed-namespace phi-coercion that both the all-backend and all-numpy grids
    # skip (they pass phi in the same namespace as R).
    aF = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=False)
    bargs = [_arr(backend, v) for v in _STK_GRID]
    got = aF.actionsFreqsAngles(*bargs, _STK_GRID_PHI)  # phi stays numpy
    ref = aF.actionsFreqsAngles(*_STK_GRID, _STK_GRID_PHI)
    for i in (6, 7, 8):  # angler, anglephi, anglez
        assert _is_backend_array(backend, got[i])
        assert numpy.max(_wrapdiff(_np(got[i]), numpy.asarray(ref[i]))) < 1e-8


def test_actionAngleVerticalInverse_rejects_backend():
    # NOT yet backend-migrated (under development): both a forced/active backend
    # context (the all-backend harness's use(force=True), which would break the
    # scipy grid SETUP) and backend-array inputs raise NotImplementedError. numpy
    # is unaffected.
    from galpy import backend as gb
    from galpy.actionAngle import actionAngleVerticalInverse
    from galpy.actionAngle.actionAngleVerticalInverse import _reject_backend
    from galpy.potential import IsothermalDiskPotential

    _reject_backend(1.0, numpy.array([0.1]))  # numpy: no raise
    isopot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    for bk in BACKENDS:
        with pytest.raises(NotImplementedError):  # backend-array input
            _reject_backend(_arr(bk, numpy.array([0.1])))
        with gb.use(bk, force=True):  # forced/active backend context
            with pytest.raises(NotImplementedError):
                _reject_backend()
            with pytest.raises(NotImplementedError):  # construction = setup chokepoint
                actionAngleVerticalInverse(pot=isopot, Es=[0.3])
    return None


def test_actionAngleTorus_rejects_backend():
    # actionAngleTorus wraps the external Torus C++ library and will NOT be made
    # backend-compatible (out of scope); same guard, permanent message.
    from galpy import backend as gb
    from galpy.actionAngle import actionAngleTorus
    from galpy.actionAngle.actionAngleTorus import _reject_backend
    from galpy.potential import MWPotential2014

    _reject_backend(1.0)  # numpy: no raise
    for bk in BACKENDS:
        with pytest.raises(NotImplementedError):
            _reject_backend(_arr(bk, numpy.array([0.1])))
        with gb.use(bk, force=True):
            with pytest.raises(NotImplementedError):
                _reject_backend()
            with pytest.raises(NotImplementedError):
                actionAngleTorus(pot=MWPotential2014)
    return None


# ====================================================================
# actionAngleIsochroneApprox: the Bovy (2014) isochrone-approximation actions,
# frequencies, and angles. Unlike every class above (closed-form / quadrature on
# the INPUT phase-space point), this method INTEGRATES an orbit and angle-averages
# the isochrone actions over it. The backend path therefore reuses TWO already-
# migrated pieces: (1) Orbit(stacked-backend-IC).integrate(...) routes to the
# in-backend differentiable integrator (Track D: C-STM for dopr54_c, else
# diffrax/torchdiffeq), and (2) self._aAI._actionsFreqsAngles runs the isochrone
# action/angle map on the integrated backend trajectory. The numpy per-object loop
# is byte-identical (the diff is purely additive is_backend_array early-returns);
# jax/torch ICs take the vectorised _parse_args_backend / *_backend branches.
# Validates: numpy<->jax<->torch value parity for _evaluate (actions),
# _actionsFreqs (+freqs), _actionsFreqsAngles (+angles), on a small bound 3D
# MWPotential2014 grid with an EXPLICIT b (so the test does NOT depend on
# estimateBIsochrone, migrated separately); that outputs are backend arrays; and
# grad-vs-finite-difference of an action w.r.t. an IC under jax -- differentiating
# THROUGH the orbit integration (orbit-STM), which is the point of the port.
#
# tintJ/ntintJ are reduced from the defaults (100/10000) to keep each integration
# fast; the angle-fit still covers the full radial/vertical angle range for these
# bound MWPotential2014 orbits. b is EXPLICIT (0.8 / 1.2); estimateBIsochrone is
# never called. Parity tolerance ~1e-8: the in-backend integrator's trajectory
# differs from the numpy C integrator at the ~1e-8 step level for fixed-step
# methods (dop853_c is ~1e-12), and the angle-averaged actions inherit that.
_IA_R = numpy.array([1.1, 0.8, 1.2])
_IA_VR = numpy.array([0.1, -0.1, 0.05])
_IA_VT = numpy.array([0.9, 1.0, -0.7])  # incl. a retrograde orbit (negFreqIndx)
_IA_Z = numpy.array([0.05, -0.1, 0.1])
_IA_VZ = numpy.array([0.1, 0.05, -0.08])
_IA_PHI = numpy.array([1.3, 0.4, 2.0])
_IA = (_IA_R, _IA_VR, _IA_VT, _IA_Z, _IA_VZ, _IA_PHI)


def _aAIA(b):
    # Explicit b (no estimateBIsochrone); short integration for test speed.
    return actionAngleIsochroneApprox(pot=MWPotential2014, b=b, tintJ=20.0, ntintJ=2000)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("b", [0.8, 1.2])
def test_isochroneapprox_parity(backend, b):
    # numpy <-> jax <-> torch parity of _evaluate (jr,lz,jz), _actionsFreqs
    # (+Or,Op,Oz), and _actionsFreqsAngles (+ar,ap,az) on a bound 3D grid (one
    # retrograde orbit). Actions/freqs to ~1e-8 (the in-backend-vs-C integrator
    # floor); angles compared as wrap-robust circular differences.
    aAIA = _aAIA(b)
    bargs = [_arr(backend, v) for v in _IA]
    # _evaluate: (jr, lz, jz)
    ref = aAIA._evaluate(*_IA)
    got = aAIA._evaluate(*bargs)
    for r, g in zip(ref, got):
        assert _is_backend_array(backend, g)
        numpy.testing.assert_allclose(_np(g), numpy.asarray(r), rtol=1e-7, atol=1e-8)
    # _actionsFreqs: (jr,lz,jz,Or,Op,Oz)
    ref = aAIA._actionsFreqs(*_IA)
    got = aAIA._actionsFreqs(*bargs)
    for r, g in zip(ref, got):
        assert _is_backend_array(backend, g)
        numpy.testing.assert_allclose(_np(g), numpy.asarray(r), rtol=1e-7, atol=1e-8)
    # _actionsFreqsAngles: (...,ar,ap,az)
    ref = aAIA._actionsFreqsAngles(*_IA)
    got = aAIA._actionsFreqsAngles(*bargs)
    for idx, (r, g) in enumerate(zip(ref, got)):
        assert _is_backend_array(backend, g)
        if idx >= 6:  # angles: wrap-robust
            d = (_np(g) - numpy.asarray(r) + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
            numpy.testing.assert_allclose(d, 0.0, atol=1e-6)
        else:
            numpy.testing.assert_allclose(
                _np(g), numpy.asarray(r), rtol=1e-7, atol=1e-8
            )


@pytest.mark.parametrize("backend", BACKENDS)
def test_isochroneapprox_scalar_parity(backend):
    # a single (scalar-array) IC: _parse_args promotes it to a 1-element batch, so
    # the backend path's atleast_1d / one batched (1,6) Orbit solve handles the
    # batch-of-one. Exercises the scalar-vs-batch handling separately from the grid.
    aAIA = _aAIA(0.8)
    s = (1.1, 0.2, 0.9, 0.15, 0.1, 1.3)
    ref = aAIA._evaluate(*[numpy.atleast_1d(numpy.asarray(v)) for v in s])
    got = aAIA._evaluate(
        *[_arr(backend, numpy.atleast_1d(numpy.asarray(v))) for v in s]
    )
    for r, g in zip(ref, got):
        assert _is_backend_array(backend, g)
        numpy.testing.assert_allclose(_np(g), numpy.asarray(r), rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("backend", BACKENDS)
def test_isochroneapprox_cumul_parity(backend):
    # cumul=True returns the CUMULATIVE angle-averaged actions (convergence trace),
    # shape (no, nt-1) rather than (no,). numpy <-> backend parity of the trace.
    aAIA = _aAIA(0.8)
    bargs = [_arr(backend, v) for v in _IA]
    ref = aAIA._evaluate(*_IA, cumul=True)
    got = aAIA._evaluate(*bargs, cumul=True)
    for r, g in zip(ref, got):
        assert _is_backend_array(backend, g)
        numpy.testing.assert_allclose(_np(g), numpy.asarray(r), rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("backend", BACKENDS)
def test_isochroneapprox_planar_parity(backend):
    # 4-argument (planar) call signature R,vR,vT,phi (z=vz=0): the backend
    # _parse_args branch builds the z=vz=0 ICs. Jz is ~0; numpy<->backend parity.
    aAIA = _aAIA(0.8)
    planar = (_IA_R, _IA_VR, _IA_VT, _IA_PHI)
    ref = aAIA._evaluate(*planar)
    got = aAIA._evaluate(*[_arr(backend, v) for v in planar])
    for r, g in zip(ref, got):
        assert _is_backend_array(backend, g)
        numpy.testing.assert_allclose(_np(g), numpy.asarray(r), rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("which,idx", [("jr", 0), ("jz", 2)])
def test_isochroneapprox_grad_vs_fd_wrt_vR(backend, which, idx):
    # d(action[idx]) / d vR at a single bound point, AD vs finite-difference --
    # differentiating THROUGH the orbit integration (the orbit-STM / in-backend
    # ODE) and the angle-averaging. This is the whole point of the port: gradients
    # of the isochrone-approximation actions w.r.t. the initial condition.
    aAIA = _aAIA(0.8)
    R, _, vT, z, vz, phi = (1.1, 0.2, 0.9, 0.15, 0.1, 1.3)

    def call(vR_arr, xp_arr):
        args = (xp_arr(R), vR_arr, xp_arr(vT), xp_arr(z), xp_arr(vz), xp_arr(phi))
        return aAIA._evaluate(*args)[idx][0]  # scalar action

    def f_np(vR_val):
        return numpy.asarray(
            call(
                numpy.atleast_1d(numpy.asarray(vR_val, dtype=float)),
                lambda v: numpy.atleast_1d(numpy.asarray(v, dtype=float)),
            )
        )

    fd = _fd(f_np, 0.2)

    def f_be(vR_t):
        return call(
            (jnp.atleast_1d(vR_t) if backend == "jax" else torch.atleast_1d(vR_t)),
            lambda v: _arr(backend, numpy.atleast_1d(numpy.asarray(v, dtype=float))),
        )

    g = _grad(backend, f_be, 0.2)
    assert numpy.isfinite(g)
    numpy.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-7)


# Non-axisymmetric IsochroneApprox: the same orbit-integrate-and-angle-average
# machinery, but for a potential with _isNonAxi(pot)=True. Two extra backend
# branches fire that the axisymmetric tests never reach: (1) Lz is itself
# angle-averaged (danglephi-weighted), not L_z(t0); and (2) the angle-fit design
# matrix uses the 3-D (n_R, n_phi, n_Z) Fourier grid (with the half-space mask)
# and an n_phi*anglephi term inside sin(n.angle). A mildly triaxial
# LogarithmicHaloPotential (b=0.9 in y, q=0.8 in z) keeps the orbits bound and
# the isochrone approximation applicable; the retrograde orbit additionally
# exercises the negFreqIndx (2pi-anglephi / -Omegaphi) branch under non-axi.
_NA_POT = LogarithmicHaloPotential(normalize=1.0, b=0.9, q=0.8)
_NA_R = numpy.array([1.1, 0.9])
_NA_VR = numpy.array([0.1, -0.05])
_NA_VT = numpy.array([0.9, -0.8])  # second orbit retrograde -> negFreqIndx
_NA_Z = numpy.array([0.05, 0.1])
_NA_VZ = numpy.array([0.1, -0.07])
_NA_PHI = numpy.array([1.3, 0.5])
_NA = (_NA_R, _NA_VR, _NA_VT, _NA_Z, _NA_VZ, _NA_PHI)


@pytest.mark.parametrize("backend", BACKENDS)
def test_isochroneapprox_nonaxi_parity(backend):
    # numpy <-> jax <-> torch parity for a NON-axisymmetric potential. Exercises
    # the angle-averaged Lz (_evaluate index 1) and the 3-D-grid angle-fit
    # (anglephi, index 7) backend branches, plus the retrograde negFreqIndx
    # branch under non-axi. b is EXPLICIT (no estimateBIsochrone).
    aAIA = actionAngleIsochroneApprox(pot=_NA_POT, b=0.8, tintJ=20.0, ntintJ=2000)
    bargs = [_arr(backend, v) for v in _NA]
    # _evaluate: (jr, lz, jz) -- lz is danglephi-weighted angle average here
    ref = aAIA._evaluate(*_NA)
    got = aAIA._evaluate(*bargs)
    for r, g in zip(ref, got):
        assert _is_backend_array(backend, g)
        numpy.testing.assert_allclose(_np(g), numpy.asarray(r), rtol=1e-7, atol=1e-8)
    # _actionsFreqsAngles: (jr,lz,jz,Or,Op,Oz,ar,aphi,az) -- non-axi angle-fit
    ref = aAIA._actionsFreqsAngles(*_NA)
    got = aAIA._actionsFreqsAngles(*bargs)
    for idx, (r, g) in enumerate(zip(ref, got)):
        assert _is_backend_array(backend, g)
        if idx >= 6:  # angles (incl. anglephi at idx 7): wrap-robust
            d = (_np(g) - numpy.asarray(r) + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
            numpy.testing.assert_allclose(d, 0.0, atol=1e-6)
        else:
            numpy.testing.assert_allclose(
                _np(g), numpy.asarray(r), rtol=1e-7, atol=1e-8
            )
    # the retrograde (second) orbit must give a negative azimuthal frequency
    assert float(_np(got[4]).ravel()[1]) < 0.0


# ------- IsochroneApprox via the in-backend diffrax/torchdiffeq integrator (#102)
# integrate_method='diffrax'/'torchdiffeq' routes the orbit integration through the
# in-backend differentiable ODE solver instead of the default first-order C-STM
# ('dopr54_c'). That is what makes the actions GPU-integrable AND twice
# differentiable: the C-STM is a jax.pure_callback / torch.autograd.Function that
# supports only first derivatives, while the in-backend solver is pure-backend. For
# jax, second derivatives additionally require the 'direct' diffrax adjoint (the
# default reverse-mode RecursiveCheckpointAdjoint cannot be differentiated twice --
# no forward-mode, and its backward has a dynamic while_loop); torchdiffeq's plain
# odeint retains the graph and double-backprops as-is. b is EXPLICIT throughout so
# estimateBIsochrone (whose backend root-find is first-order only) is never on the
# differentiated path. tintJ/ntintJ are small for test speed.
_IA_HESS_IC = (1.1, 0.2, 0.9, 0.15, 0.1, 1.3)


def _aAIA_small(method=None, **kw):
    m = {} if method is None else {"integrate_method": method}
    return actionAngleIsochroneApprox(
        pot=MWPotential2014, b=0.8, tintJ=8.0, ntintJ=400, **m, **kw
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_isochroneapprox_inbackend_method_parity(backend):
    # integrate_method='diffrax'/'torchdiffeq' gives the SAME actions/freqs as the
    # numpy C path (same b/tintJ/ntintJ) -- the in-backend ODE integrator matches the
    # C integrator to the integrator floor. Confirms routing the AA orbit through the
    # in-backend solver does not change the answer.
    method = "diffrax" if backend == "jax" else "torchdiffeq"
    ref = _aAIA_small()._actionsFreqs(*_IA)  # numpy / C reference
    got = _aAIA_small(method)._actionsFreqs(*[_arr(backend, v) for v in _IA])
    for r, g in zip(ref, got):
        assert _is_backend_array(backend, g)
        numpy.testing.assert_allclose(_np(g), numpy.asarray(r), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("backend", BACKENDS)
def test_isochroneapprox_inbackend_hessian(backend):
    # SECOND derivative d2 jr / d vR2 through the orbit integration + angle-average,
    # AD vs finite-difference-of-the-gradient. jax needs integrate_kwargs with the
    # 'direct' adjoint (+ a modest max_steps that the DirectAdjoint scan uses);
    # torchdiffeq double-backprops with no extra option. This is the capability the
    # first-order C-STM path cannot provide.
    if backend == "jax":
        aA = _aAIA_small(
            "diffrax", integrate_kwargs={"adjoint": "direct", "max_steps": 4096}
        )
    else:
        aA = _aAIA_small("torchdiffeq")
    R, _, vT, z, vz, phi = _IA_HESS_IC

    def f_be(vR_t):
        vR = jnp.atleast_1d(vR_t) if backend == "jax" else torch.atleast_1d(vR_t)
        args = (
            _arr(backend, [R]),
            vR,
            _arr(backend, [vT]),
            _arr(backend, [z]),
            _arr(backend, [vz]),
            _arr(backend, [phi]),
        )
        return aA._evaluate(*args)[0][0]  # scalar jr

    # finite-difference of the AD gradient is the 2nd-derivative reference
    fd2 = (_grad(backend, f_be, 0.2 + 1e-3) - _grad(backend, f_be, 0.2 - 1e-3)) / 2e-3
    h = _hess(backend, f_be, 0.2)
    assert numpy.isfinite(h)
    numpy.testing.assert_allclose(h, fd2, rtol=1e-3, atol=1e-5)


# --------------------------------------------------------- delta/b estimators
# estimateDeltaStaeckel / estimateBIsochrone: standalone helpers that consume a
# potential's 2nd-derivative / rotation-curve evaluators and root-find for the
# focal length delta / isochrone scale b. The numpy path is byte-identical (the
# existing test_actionAngle.py suite is unchanged); these add backend value
# parity, the backend-array return type, and AD-vs-finite-difference of delta/b
# w.r.t. R -- the differentiability that is the point of the port.
_EST_R = numpy.array([1.1, 0.8, 1.3, 0.9])
_EST_Z = numpy.array([0.15, 0.2, 0.1, 0.25])


@pytest.mark.parametrize("backend", BACKENDS)
def test_estimateDeltaStaeckel_parity(backend):
    Rb, zb = _arr(backend, _EST_R), _arr(backend, _EST_Z)
    # array, median over the grid (single value)
    ref_med = numpy.asarray(estimateDeltaStaeckel(MWPotential2014, _EST_R, _EST_Z))
    got_med = estimateDeltaStaeckel(MWPotential2014, Rb, zb)
    assert _is_backend_array(backend, got_med)
    numpy.testing.assert_allclose(_np(got_med), ref_med, rtol=1e-12, atol=1e-12)
    # array, no_median: per-point deltas
    ref_all = numpy.asarray(
        estimateDeltaStaeckel(MWPotential2014, _EST_R, _EST_Z, no_median=True)
    )
    got_all = estimateDeltaStaeckel(MWPotential2014, Rb, zb, no_median=True)
    assert _is_backend_array(backend, got_all)
    numpy.testing.assert_allclose(_np(got_all), ref_all, rtol=1e-12, atol=1e-12)
    # scalar, incl. a z==0 (plane) point that hits the 1e-4 fallback
    for Rs, zs in ((1.1, 0.15), (1.2, 0.0)):
        ref = numpy.asarray(estimateDeltaStaeckel(MWPotential2014, Rs, zs))
        got = estimateDeltaStaeckel(
            MWPotential2014, _arr(backend, Rs), _arr(backend, zs)
        )
        assert _is_backend_array(backend, got)
        numpy.testing.assert_allclose(_np(got), ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_estimateBIsochrone_parity(backend):
    Rb, zb = _arr(backend, _EST_R), _arr(backend, _EST_Z)
    # array -> (bmin, bmedian, bmax)
    ref = numpy.asarray(estimateBIsochrone(MWPotential2014, _EST_R, _EST_Z))
    got = estimateBIsochrone(MWPotential2014, Rb, zb)
    assert _is_backend_array(backend, got)
    numpy.testing.assert_allclose(_np(got), ref, rtol=1e-11, atol=1e-11)
    # scalar -> single b
    ref_s = float(estimateBIsochrone(MWPotential2014, 1.1, 0.15))
    got_s = estimateBIsochrone(MWPotential2014, _arr(backend, 1.1), _arr(backend, 0.15))
    assert _is_backend_array(backend, got_s)
    numpy.testing.assert_allclose(_np(got_s), ref_s, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("backend", BACKENDS)
def test_estimateDeltaStaeckel_grad_vs_fd_wrt_R(backend):
    # d(delta)/dR at a fixed bound (R,z) point: AD vs central finite-difference.
    z0 = 0.15

    def f(backend, Rval):
        return estimateDeltaStaeckel(MWPotential2014, Rval, _arr(backend, z0))

    R0 = 1.1
    g = _grad(backend, lambda t: f(backend, t), R0)
    fd = _fd(lambda Rv: float(_np(f(backend, _arr(backend, Rv)))), R0)
    assert numpy.isfinite(g)
    numpy.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("backend", BACKENDS)
def test_estimateBIsochrone_grad_vs_fd_wrt_R(backend):
    # db/dR at a fixed (R,z) point: AD (through the bisection+Newton root) vs FD.
    z0 = 0.15

    def f(backend, Rval):
        return estimateBIsochrone(MWPotential2014, Rval, _arr(backend, z0))

    R0 = 1.1
    g = _grad(backend, lambda t: f(backend, t), R0)
    fd = _fd(lambda Rv: float(_np(f(backend, _arr(backend, Rv)))), R0)
    assert numpy.isfinite(g)
    numpy.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-7)


# ====================================================================
# actionAngleAdiabatic: actions (jr,Lz,jz), frequencies (Or,Op,Oz), angles
# (ar,aphi,az), and EccZmaxRperiRap in the adiabatic approximation. The numpy
# per-object loop (a per-R actionAngleVertical for the vertical action + the
# spherical actionAngleSpherical for the radial part) is untouched (byte-
# identical); jax/torch inputs take a vectorised, differentiable branch that
# processes ALL N objects at once: the RADIAL part via the already-backend-
# migrated actionAngleSpherical self._aAS (gamma + the per-element _Jz array
# handled internally), the VERTICAL part via actionAngleVertical's backend
# Gauss-Legendre / root-find machinery over a per-R effective vertical potential
# Phi(R_i, z) - Phi(R_i, 0). Backend GL (n=50) vs scipy adaptive quad differ at
# the ~1e-9 level on the continuous freq/angle integrals (rtol~1e-6); the
# actions/peri/apo are ~1e-9. The planar (z=vz=0) and 2D-potential edges are
# covered (Jz=0, Oz=verticalfreq(R), az=0).
_ADB_R = numpy.array([1.1, 0.8, 1.3])
_ADB_VR = numpy.array([0.2, -0.1, 0.05])
_ADB_VT = numpy.array([0.9, 0.6, 1.0])
_ADB_Z = numpy.array([0.15, -0.2, 0.1])
_ADB_VZ = numpy.array([0.1, 0.05, -0.1])
_ADB = (_ADB_R, _ADB_VR, _ADB_VT, _ADB_Z, _ADB_VZ)
_ADB_PHI = numpy.array([1.3, 0.4, 2.1])


@pytest.mark.parametrize("backend", BACKENDS)
def test_adiabatic_actions_parity(backend):
    # numpy <-> jax <-> torch parity of _evaluate (jr,Lz,jz) on a bound 3D grid.
    aA = actionAngleAdiabatic(pot=MWPotential2014, c=False)
    bargs = [_arr(backend, v) for v in _ADB]
    ref = aA._evaluate(*_ADB)
    got = aA._evaluate(*bargs)
    for r, g in zip(ref, got):
        assert _is_backend_array(backend, g)
        numpy.testing.assert_allclose(_np(g), numpy.asarray(r), rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("backend", BACKENDS)
def test_adiabatic_actionsfreqs_parity(backend):
    # numpy <-> backend parity of _actionsFreqs (jr,Lz,jz,Or,Op,Oz).
    aA = actionAngleAdiabatic(pot=MWPotential2014, c=False)
    bargs = [_arr(backend, v) for v in _ADB]
    ref = aA._actionsFreqs(*_ADB)
    got = aA._actionsFreqs(*bargs)
    for r, g in zip(ref, got):
        assert _is_backend_array(backend, g)
        numpy.testing.assert_allclose(_np(g), numpy.asarray(r), rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("backend", BACKENDS)
def test_adiabatic_actionsfreqsangles_parity(backend):
    # numpy <-> backend parity of _actionsFreqsAngles (jr,Lz,jz,Or,Op,Oz,ar,aphi,az);
    # angles compared as wrap-robust circular differences.
    aA = actionAngleAdiabatic(pot=MWPotential2014, c=False)
    bargs = [_arr(backend, v) for v in _ADB]
    bphi = _arr(backend, _ADB_PHI)
    ref = aA._actionsFreqsAngles(*_ADB, _ADB_PHI)
    got = aA._actionsFreqsAngles(*bargs, bphi)
    for idx, (r, g) in enumerate(zip(ref, got)):
        assert _is_backend_array(backend, g)
        if idx >= 6:  # ar, aphi, az
            d = (_np(g) - numpy.asarray(r) + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
            numpy.testing.assert_allclose(d, 0.0, atol=1e-6)
        else:
            numpy.testing.assert_allclose(
                _np(g), numpy.asarray(r), rtol=1e-6, atol=1e-8
            )


@pytest.mark.parametrize("backend", BACKENDS)
def test_adiabatic_ecczmaxrperirap_parity(backend):
    # numpy <-> backend parity of _EccZmaxRperiRap (e,zmax,rperi,rap).
    aA = actionAngleAdiabatic(pot=MWPotential2014, c=False)
    bargs = [_arr(backend, v) for v in _ADB]
    ref = aA._EccZmaxRperiRap(*_ADB)
    got = aA._EccZmaxRperiRap(*bargs)
    for r, g in zip(ref, got):
        assert _is_backend_array(backend, g)
        numpy.testing.assert_allclose(_np(g), numpy.asarray(r), rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("backend", BACKENDS)
def test_adiabatic_actions_vs_c(backend):
    # the vectorised backend actions are consistent with the C path (c=True) to
    # the inherent adiabatic Python-vs-C floor (the SAME floor the numpy c=False
    # path has -- the backend adds no error of its own).
    aF = actionAngleAdiabatic(pot=MWPotential2014, c=False)
    aC = actionAngleAdiabatic(pot=MWPotential2014, c=True)
    jr_c, lz_c, jz_c = aC._evaluate(*_ADB)
    jr_n, lz_n, jz_n = aF._evaluate(*_ADB)
    # numpy c=False vs c=True sets the floor; the backend must match numpy c=False
    # at machine precision, so its distance to c=True equals numpy's.
    floor = max(
        numpy.max(numpy.abs(numpy.asarray(a) - numpy.asarray(b)))
        for a, b in ((jr_n, jr_c), (lz_n, lz_c), (jz_n, jz_c))
    )
    jr_b, lz_b, jz_b = aF._evaluate(*[_arr(backend, v) for v in _ADB])
    for g, c in ((jr_b, jr_c), (lz_b, lz_c), (jz_b, jz_c)):
        numpy.testing.assert_allclose(
            _np(g), numpy.asarray(c), rtol=1e-5, atol=floor + 1e-9
        )


# Scalar single-point bound IC for clean per-component derivatives.
_ADB_S = (1.1, 0.2, 0.9, 0.15, 0.1)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "which,deriv_idx,wrt", [("jz", 2, 4), ("jr", 0, 1)]
)  # d jz / d vz ; d jr / d vR
def test_adiabatic_grad_vs_fd(backend, which, deriv_idx, wrt):
    # d(action[deriv_idx]) / d (coord wrt) at a single bound point: AD through the
    # vertical root-find + GL action quadrature AND the spherical radial path
    # (gamma!=0 _Jz threaded as an array) vs finite-difference.
    aA = actionAngleAdiabatic(pot=MWPotential2014, c=False)
    base = list(_ADB_S)

    def call(p_val, xp_arr):
        args = [xp_arr(v) for v in base]
        args[wrt] = p_val
        return aA._evaluate(*args)[deriv_idx].sum()

    def f_np(p_val):
        return numpy.asarray(
            call(
                numpy.atleast_1d(numpy.asarray(p_val)),
                lambda v: numpy.atleast_1d(numpy.asarray(v)),
            )
        )

    fd = _fd(f_np, base[wrt])

    def f_be(p_t):
        return call(p_t, lambda v: _arr(backend, numpy.atleast_1d(v).astype(float)))

    if backend == "jax":
        g = float(
            jax.grad(lambda t: f_be(t).reshape(()))(
                jnp.atleast_1d(jnp.asarray(base[wrt]))
            ).reshape(())
        )
    else:
        t = torch.tensor(
            numpy.atleast_1d(numpy.asarray(base[wrt], dtype=float)), requires_grad=True
        )
        out = f_be(t)
        out.backward()
        g = float(t.grad.sum())
    assert numpy.isfinite(g)
    numpy.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("backend", BACKENDS)
def test_adiabatic_planar_edge_parity(backend):
    # Planar orbits (z=vz=0) mixed with off-plane ones: the backend must set
    # Jz=0, az=0, and Oz=verticalfreq(R) EXACTLY at the planar elements (the
    # xp.where mask), matching the numpy 2D-in-plane branch.
    aA = actionAngleAdiabatic(pot=MWPotential2014, c=False)
    R = numpy.array([1.1, 0.8, 1.3])
    vR = numpy.array([0.2, -0.1, 0.05])
    vT = numpy.array([0.9, 0.6, 1.0])
    z = numpy.array([0.0, -0.2, 0.0])  # elements 0,2 planar
    vz = numpy.array([0.0, 0.05, 0.0])
    phi = numpy.array([1.3, 0.4, 2.1])
    args = (R, vR, vT, z, vz)
    bargs = [_arr(backend, v) for v in args]
    import warnings

    with warnings.catch_warnings():
        # numpy reference's longitude-of-ascending-node uses z/R/tan(i) = 0/0
        # (non-finite, masked) for the non-inclined planar elements.
        warnings.simplefilter("ignore")
        ref = aA._actionsFreqsAngles(*args, phi)
        got = aA._actionsFreqsAngles(*bargs, _arr(backend, phi))
    for idx, (r, g) in enumerate(zip(ref, got)):
        assert _is_backend_array(backend, g)
        if idx >= 6:
            d = (_np(g) - numpy.asarray(r) + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
            numpy.testing.assert_allclose(d, 0.0, atol=1e-6)
        else:
            numpy.testing.assert_allclose(
                _np(g), numpy.asarray(r), rtol=1e-6, atol=1e-8
            )
    # planar elements: Jz==0, az==0, Oz==verticalfreq(R)
    from galpy.potential import verticalfreq

    assert _np(got[2])[0] == 0.0 and _np(got[2])[2] == 0.0  # Jz
    assert abs(_np(got[8])[0]) < 1e-12 and abs(_np(got[8])[2]) < 1e-12  # az
    for ii in (0, 2):
        numpy.testing.assert_allclose(
            _np(got[5])[ii], verticalfreq(MWPotential2014, R[ii]), rtol=1e-12
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_adiabatic_2dpot_edge_parity(backend):
    # A 2D (planar) potential forces gamma=0 and Jz=0: only _evaluate and
    # _EccZmaxRperiRap are well-posed (a planarPotential has no verticalfreq, so
    # _actionsFreqs is unsupported on BOTH the numpy and backend paths). Check
    # the supported methods match numpy and that Jz/zmax are identically 0.
    plog = toPlanarPotential(LogarithmicHaloPotential(normalize=1.0))
    aA = actionAngleAdiabatic(pot=plog, c=False)
    assert aA._gamma == 0.0
    R = numpy.array([1.1, 0.8, 1.3])
    vR = numpy.array([0.2, -0.1, 0.05])
    vT = numpy.array([0.9, 0.6, 1.0])
    z = numpy.zeros(3)
    vz = numpy.zeros(3)
    args = (R, vR, vT, z, vz)
    bargs = [_arr(backend, v) for v in args]
    for meth in ("_evaluate", "_EccZmaxRperiRap"):
        ref = getattr(aA, meth)(*args)
        got = getattr(aA, meth)(*bargs)
        for r, g in zip(ref, got):
            assert _is_backend_array(backend, g)
            numpy.testing.assert_allclose(
                _np(g), numpy.asarray(r), rtol=1e-8, atol=1e-9
            )
    # Jz (index 2 of _evaluate) and zmax (index 1 of ecc) are identically 0.
    assert numpy.all(_np(aA._evaluate(*bargs)[2]) == 0.0)
    assert numpy.all(_np(aA._EccZmaxRperiRap(*bargs)[1]) == 0.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_adiabatic_gamma0_ecczmaxrperirap_parity(backend):
    # gamma=0 on a 3D potential: _EccZmaxRperiRap takes the Jz=0 branch (zmax
    # still from the vertical potential), so radial peri/apo ignore the adiabatic
    # Jz shift. numpy <-> backend parity (GL-vs-adaptive-quad floor ~1e-6).
    aA = actionAngleAdiabatic(pot=MWPotential2014, c=False, gamma=0.0)
    assert aA._gamma == 0.0
    bargs = [_arr(backend, v) for v in _ADB]
    ref = aA._EccZmaxRperiRap(*_ADB)
    got = aA._EccZmaxRperiRap(*bargs)
    for r, g in zip(ref, got):
        assert _is_backend_array(backend, g)
        numpy.testing.assert_allclose(_np(g), numpy.asarray(r), rtol=1e-6, atol=1e-8)


# ====================================================================
# actionAngleStaeckelGrid / actionAngleAdiabaticGrid: GRID-INTERPOLATED actions.
# The grid SETUP (scipy spline/RectBivariate fits + ndimage.spline_filter) stays
# numpy-only and BYTE-IDENTICAL; only the EVALUATION (interpolation lookups +
# the surrounding coordinate/energy/u0/psi math) gets an additive backend branch.
# At __init__ the SAME fitted scipy objects / filtered grids are wrapped with the
# galpy.backend.interpolate helpers (Spline1D / Spline2D / MapCoordinates -- the
# latter cubic map_coordinates off a setup-time prefilter), so the numpy path is
# a literal scipy passthrough and jax/torch inputs evaluate natively and
# differentiably. ICs are chosen well inside the grid (the off-grid fallback to
# self._aA is numpy-only), so the backend path is the pure on-grid interpolation.
# Parity is at machine precision (~1e-13) because the wrappers reuse scipy's own
# coefficients; grad-vs-FD validates the differentiability that is the point.
_GRID_R = numpy.array([1.1, 0.8, 1.3])
_GRID_VR = numpy.array([0.1, -0.2, 0.05])
_GRID_VT = numpy.array([0.9, 0.6, 1.0])
_GRID_Z = numpy.array([0.1, -0.15, 0.08])
_GRID_VZ = numpy.array([0.08, 0.1, -0.05])
_GRID = (_GRID_R, _GRID_VR, _GRID_VT, _GRID_Z, _GRID_VZ)

# Built once (grid construction is the expensive part); small grids for speed.
_aASG = actionAngleStaeckelGrid(
    pot=MWPotential2014, delta=0.45, nE=15, npsi=15, nLz=18, interpecc=True
)
_aAAG = actionAngleAdiabaticGrid(pot=MWPotential2014, nR=12, nEz=12, nEr=20, nLz=20)


@pytest.mark.parametrize("backend", BACKENDS)
def test_staeckelgrid_parity(backend):
    # numpy <-> jax <-> torch parity of the grid-interpolated (Jr,Lz,Jz) and
    # (ecc,zmax,rperi,rap). The backend wrappers reuse scipy's fitted
    # coefficients, so parity is at machine precision (rtol~1e-10).
    bargs = [_arr(backend, v) for v in _GRID]
    for ref, got in (
        (_aASG(*_GRID), _aASG(*bargs)),
        (_aASG.EccZmaxRperiRap(*_GRID), _aASG.EccZmaxRperiRap(*bargs)),
    ):
        for r, g in zip(ref, got):
            assert _is_backend_array(backend, g)
            numpy.testing.assert_allclose(
                _np(g), numpy.asarray(r), rtol=1e-10, atol=1e-12
            )


@pytest.mark.parametrize("backend", BACKENDS)
def test_adiabaticgrid_parity(backend):
    # numpy <-> jax <-> torch parity of the grid-interpolated (Jr,Lz,Jz).
    bargs = [_arr(backend, v) for v in _GRID]
    ref, got = _aAAG(*_GRID), _aAAG(*bargs)
    for r, g in zip(ref, got):
        assert _is_backend_array(backend, g)
        numpy.testing.assert_allclose(_np(g), numpy.asarray(r), rtol=1e-10, atol=1e-12)


def test_staeckelgrid_numpy_byte_identity():
    # A numpy-array input takes the untouched scipy interpolation path; the
    # backend dispatch is is_backend_array(R) == False, so the result is the
    # EXACT pre-migration output. We verify the numpy-array path equals the
    # scalar-loop path bit-for-bit (the scalar branch recurses to the array
    # branch, so this pins the array path to per-point evaluation).
    act = _aASG(*_GRID)
    ecc = _aASG.EccZmaxRperiRap(*_GRID)
    for ii in range(len(_GRID_R)):
        sact = _aASG(*[v[ii] for v in _GRID])
        secc = _aASG.EccZmaxRperiRap(*[v[ii] for v in _GRID])
        for comp, s in zip(act, sact):
            numpy.testing.assert_array_equal(comp[ii], s)
        for comp, s in zip(ecc, secc):
            numpy.testing.assert_array_equal(comp[ii], s)


def test_adiabaticgrid_numpy_byte_identity():
    # As above for the adiabatic grid: numpy-array path == scalar-loop path,
    # confirming the numpy output is unchanged by the additive backend branch.
    act = _aAAG(*_GRID)
    for ii in range(len(_GRID_R)):
        sact = _aAAG(*[v[ii] for v in _GRID])
        for comp, s in zip(act, sact):
            numpy.testing.assert_array_equal(comp[ii], s)


# Single-point bound IC well inside both grids, away from turning points / edges.
_GRID_S = (1.1, 0.15, 0.9, 0.12, 0.13)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "aa,idx", [("staeckel", 2), ("adiabatic", 2)]
)  # Jz vs vz through both grids
def test_grid_jz_grad_vs_fd_wrt_vz(backend, aa, idx):
    # d Jz / d vz at a single on-grid point: AD through the cubic-spline
    # interpolation (map_coordinates for Staeckel, RectBivariate for adiabatic)
    # vs central finite difference. The interpolation lookups carry the gradient.
    grid = _aASG if aa == "staeckel" else _aAAG
    R, vR, vT, z, _ = _GRID_S
    vz0 = _GRID_S[4]

    def call(vz_arr, xp_arr):
        args = (xp_arr(R), xp_arr(vR), xp_arr(vT), xp_arr(z), vz_arr)
        return grid(*args)[idx].sum()

    def f_np(vz_val):
        # FD reference: vz is a plain numpy 1-element array (numpy array path).
        return numpy.asarray(
            call(numpy.atleast_1d(vz_val), lambda v: numpy.atleast_1d(numpy.asarray(v)))
        )

    fd = _fd(f_np, vz0)

    def f_be(vz_t):
        # AD: vz_t is the differentiated rank-0 backend scalar (kept raw so grad
        # sees a scalar); the backend grid path is fully vectorised on rank-0.
        return call(vz_t, lambda v: _arr(backend, numpy.atleast_1d(v).astype(float)))

    g = _grad(backend, f_be, vz0)
    assert numpy.isfinite(g)
    numpy.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("backend", BACKENDS)
def test_staeckelgrid_vatu0_backend(backend):
    # vatu0(retv2=False) backend branch: the grid eval path only calls
    # vatu0(retv2=True), so its masked-sqrt xp.where branch is otherwise untested.
    u0 = numpy.array([0.6, 0.8])
    E = numpy.array([-0.9, -0.8])
    Lz = numpy.array([0.3, 0.4])
    R = _aASG._delta * numpy.sinh(u0)
    vn = _aASG.vatu0(E, Lz, u0, R)  # numpy
    vb = _aASG.vatu0(
        *[_arr(backend, a) for a in (E, Lz, u0, R)]
    )  # backend -> xp.where branch
    assert _is_backend_array(backend, vb)
    numpy.testing.assert_allclose(_np(vb), vn, rtol=1e-10, atol=1e-12)
