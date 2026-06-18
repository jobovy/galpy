###############################################################################
# test_backend_orbit_integrate.py: the in-backend differentiable ODE integrator
# wired into Orbit.integrate via the new method names 'diffrax' (jax) and
# 'torchdiffeq' (torch).
#
# Proves, for the WIRED path (Orbit.integrate(..., method='diffrax'/'torchdiffeq')):
#   1. the stored trajectory (o.getOrbit()) matches galpy's C integrator
#      (modulo phi 2pi),
#   2. autodiff through Orbit.integrate gives correct gradients of the final
#      state w.r.t. the initial conditions AND potential parameters (vs FD),
#   3. the torch.autograd gradient matches the jax.grad one (cross-backend),
#   4. the routing rule (method <-> IC namespace), single-orbit / 3D-only, and
#      numpy-IC restrictions raise clear errors,
#   5. the numpy path is unaffected (a numpy orbit still uses the C integrators).
#
# Self-skips unless the runtime ODE extra (diffrax / torchdiffeq) is installed.
###############################################################################
import numpy
import pytest

from galpy.orbit import Orbit
from galpy.potential import (
    IsochronePotential,
    IsothermalDiskPotential,
    MiyamotoNagaiPotential,
    PlummerPotential,
)

pytestmark = pytest.mark.backend_managed

HAVE_JAX = False
HAVE_TORCH = False
try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import diffrax  # noqa: F401
    import jax.numpy as jnp

    HAVE_JAX = True
except ImportError:  # pragma: no cover
    pass
try:
    import torch

    torch.set_default_dtype(torch.float64)
    import torchdiffeq  # noqa: F401

    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    pass

_IC = [1.0, 0.1, 0.9, 0.2, 0.05, 0.3]  # R, vR, vT, z, vz, phi
_TS = numpy.linspace(0.0, 6.0, 120)
_POTS = [
    ("Plummer", PlummerPotential(amp=1.0, b=0.6)),
    ("Isochrone", IsochronePotential(amp=1.0, b=0.8)),
]


def _wrap_phi(a):
    a = numpy.array(a, dtype=float)
    a[..., 5] = (a[..., 5] + numpy.pi) % (2 * numpy.pi) - numpy.pi
    return a


def _c_reference(pot):
    # The trusted C trajectory in Orbit order [R,vR,vT,z,vz,phi] at the _TS grid.
    o = Orbit(_IC)
    o.integrate(_TS, pot, method="dop853_c")
    return numpy.array(
        [o.R(_TS), o.vR(_TS), o.vT(_TS), o.z(_TS), o.vz(_TS), o.phi(_TS)]
    ).T


# All phase-space dimensions the in-backend path supports: (id, IC, potential,
# accessors in Orbit order, index of the phi column to 2pi-wrap or None). 2D/3D
# orbits run in a 3D potential (planar = 3D with z=vz=0); 1D needs a
# linearPotential. phasedim 6 is exercised by test_integrate_diffrax_matches_c.
_PLUMMER = PlummerPotential(amp=1.0, b=0.6)
_DISK1D = IsothermalDiskPotential(amp=1.0, sigma=0.5)
_PHASEDIM_CASES = [
    ("pd5", [1.0, 0.1, 0.9, 0.2, 0.05], _PLUMMER, ["R", "vR", "vT", "z", "vz"], None),
    ("pd4", [1.0, 0.1, 0.9, 0.3], _PLUMMER, ["R", "vR", "vT", "phi"], 3),
    ("pd3", [1.0, 0.1, 0.9], _PLUMMER, ["R", "vR", "vT"], None),
    ("pd2", [0.1, 0.05], _DISK1D, ["x", "vx"], None),
]


def _wrap_col(a, col):
    a = numpy.array(a, dtype=float)
    if col is not None:
        a[..., col] = (a[..., col] + numpy.pi) % (2 * numpy.pi) - numpy.pi
    return a


def _c_reference_general(ic, pot, accessors):
    # Trusted C trajectory for an arbitrary-phasedim IC, in Orbit order.
    o = Orbit(list(ic))
    o.integrate(_TS, pot, method="dop853_c")
    return numpy.array([getattr(o, a)(_TS) for a in accessors]).T


# ------------------------------------------------------------- forward parity vs C
@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
@pytest.mark.parametrize("name,pot", _POTS, ids=[p[0] for p in _POTS])
def test_integrate_diffrax_matches_c(name, pot):
    o = Orbit(jnp.asarray(_IC))
    o.integrate(jnp.asarray(_TS), pot, method="diffrax")
    got = numpy.asarray(o.getOrbit())
    assert got.shape == (len(_TS), 6)
    numpy.testing.assert_allclose(
        _wrap_phi(got), _wrap_phi(_c_reference(pot)), rtol=1e-8, atol=1e-9
    )


@pytest.mark.skipif(not HAVE_TORCH, reason="torch/torchdiffeq not installed")
@pytest.mark.parametrize("name,pot", _POTS, ids=[p[0] for p in _POTS])
def test_integrate_torchdiffeq_matches_c(name, pot):
    o = Orbit(torch.as_tensor(_IC))
    o.integrate(torch.as_tensor(_TS), pot, method="torchdiffeq")
    got = o.getOrbit().detach().cpu().numpy()
    assert got.shape == (len(_TS), 6)
    numpy.testing.assert_allclose(
        _wrap_phi(got), _wrap_phi(_c_reference(pot)), rtol=1e-8, atol=1e-9
    )


# --------------------------------------------------------- gradients (jax) vs FD
@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
def test_integrate_diffrax_grad_ic_vs_fd():
    pot = PlummerPotential(amp=1.0, b=0.6)

    def final_R(vR0):
        o = Orbit(jnp.array([1.0, vR0, 0.9, 0.2, 0.05, 0.3]))
        o.integrate(jnp.asarray(_TS), pot, method="diffrax")
        return o.getOrbit()[-1, 0]

    g = float(jax.grad(final_R)(0.1))
    eps = 1e-6
    fd = float((final_R(0.1 + eps) - final_R(0.1 - eps)) / (2 * eps))
    numpy.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-7)


@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
def test_integrate_diffrax_grad_param_vs_fd():
    # gradient of the final R w.r.t. a potential parameter (Plummer scale b)
    def final_R(b):
        pot = PlummerPotential(amp=1.0, b=b)
        o = Orbit(jnp.asarray(_IC))
        o.integrate(jnp.asarray(_TS), pot, method="diffrax")
        return o.getOrbit()[-1, 0]

    g = float(jax.grad(final_R)(0.6))
    eps = 1e-6
    fd = float((final_R(0.6 + eps) - final_R(0.6 - eps)) / (2 * eps))
    numpy.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-6)


# ----------------------------------------------- torch gradient matches jax (FD-free)
@pytest.mark.skipif(not (HAVE_JAX and HAVE_TORCH), reason="needs both jax and torch")
def test_integrate_torchdiffeq_grad_matches_diffrax():
    # torchdiffeq adaptive-step FD is noisy; cross-validate torch grad vs jax grad
    pot = PlummerPotential(amp=1.0, b=0.6)

    def final_R_jax(vR0):
        o = Orbit(jnp.array([1.0, vR0, 0.9, 0.2, 0.05, 0.3]))
        o.integrate(jnp.asarray(_TS), pot, method="diffrax")
        return o.getOrbit()[-1, 0]

    g_jax = float(jax.grad(final_R_jax)(0.1))
    v = torch.tensor([1.0, 0.1, 0.9, 0.2, 0.05, 0.3], requires_grad=True)
    o = Orbit(v)
    o.integrate(torch.as_tensor(_TS), pot, method="torchdiffeq")
    o.getOrbit()[-1, 0].backward()
    numpy.testing.assert_allclose(float(v.grad[1]), g_jax, rtol=1e-6, atol=1e-8)


# --------------------------------------------------- batch via jax.vmap (single-orbit)
@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
def test_integrate_diffrax_vmap_batch():
    pot = PlummerPotential(amp=1.0, b=0.6)

    def final_R(ic):
        o = Orbit(ic)
        o.integrate(jnp.asarray(_TS), pot, method="diffrax")
        return o.getOrbit()[-1, 0]

    ics = jnp.asarray(
        numpy.stack([_IC, numpy.array(_IC) * 1.01, numpy.array(_IC) * 0.99])
    )
    batched = jax.vmap(final_R)(ics)
    looped = numpy.array([float(final_R(ic)) for ic in ics])
    # adaptive-solver step sequences can differ slightly between vmapped and
    # looped runs (and across CPU/GPU/diffrax versions); a real vmap bug would
    # differ by O(1e-2) given the 1% ICs, so this still pins vmap correctness.
    numpy.testing.assert_allclose(numpy.asarray(batched), looped, rtol=1e-7, atol=1e-9)


# --------------------------------------------------------------- error / routing rules
@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
def test_integrate_numpy_ic_diffrax_raises():
    # a numpy-IC Orbit cannot use the in-backend methods (no differentiable IC)
    o = Orbit(_IC)
    with pytest.raises(ValueError):
        o.integrate(_TS, PlummerPotential(amp=1.0, b=0.6), method="diffrax")


@pytest.mark.skipif(not (HAVE_JAX and HAVE_TORCH), reason="needs both jax and torch")
def test_integrate_method_namespace_mismatch_raises():
    pot = PlummerPotential(amp=1.0, b=0.6)
    # torch IC asked to use diffrax (jax) -> error
    with pytest.raises(ValueError):
        Orbit(torch.as_tensor(_IC)).integrate(
            torch.as_tensor(_TS), pot, method="diffrax"
        )
    # jax IC asked to use torchdiffeq (torch) -> error
    with pytest.raises(ValueError):
        Orbit(jnp.asarray(_IC)).integrate(jnp.asarray(_TS), pot, method="torchdiffeq")


@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
def test_integrate_multiorbit_inbackend_raises():
    # the in-backend path is single-orbit only for now
    o = Orbit(jnp.asarray(numpy.stack([_IC, _IC])))
    with pytest.raises(ValueError):
        o.integrate(
            jnp.asarray(_TS), PlummerPotential(amp=1.0, b=0.6), method="diffrax"
        )


# ----------------------------------------- all phase-space dimensions (not just 6)
@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
@pytest.mark.parametrize(
    "ic,pot,accessors,phicol",
    [c[1:] for c in _PHASEDIM_CASES],
    ids=[c[0] for c in _PHASEDIM_CASES],
)
def test_integrate_diffrax_phasedim_matches_c(ic, pot, accessors, phicol):
    # every supported phasedim (2/3/4/5) integrates and matches the C integrator;
    # planar (3/4) runs as 3D with z=vz=0, 1D (2) uses a linearPotential
    o = Orbit(jnp.asarray(ic))
    o.integrate(jnp.asarray(_TS), pot, method="diffrax")
    got = numpy.asarray(o.getOrbit())
    assert got.shape == (len(_TS), len(ic))
    ref = _c_reference_general(ic, pot, accessors)
    numpy.testing.assert_allclose(
        _wrap_col(got, phicol), _wrap_col(ref, phicol), rtol=1e-8, atol=1e-9
    )


@pytest.mark.skipif(not HAVE_TORCH, reason="torch/torchdiffeq not installed")
@pytest.mark.parametrize(
    "ic,pot,accessors,phicol",
    [c[1:] for c in _PHASEDIM_CASES],
    ids=[c[0] for c in _PHASEDIM_CASES],
)
def test_integrate_torchdiffeq_phasedim_matches_c(ic, pot, accessors, phicol):
    o = Orbit(torch.as_tensor(ic))
    o.integrate(torch.as_tensor(_TS), pot, method="torchdiffeq")
    got = o.getOrbit().detach().cpu().numpy()
    assert got.shape == (len(_TS), len(ic))
    ref = _c_reference_general(ic, pot, accessors)
    numpy.testing.assert_allclose(
        _wrap_col(got, phicol), _wrap_col(ref, phicol), rtol=1e-8, atol=1e-9
    )


@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
def test_integrate_diffrax_grad_ic_planar_and_1d():
    # gradients flow through the planar (phasedim 4) and 1D (phasedim 2) paths
    def planar_final_R(vR0):
        o = Orbit(jnp.array([1.0, vR0, 0.9, 0.3]))
        o.integrate(jnp.asarray(_TS), _PLUMMER, method="diffrax")
        return o.getOrbit()[-1, 0]

    def linear_final_x(vx0):
        o = Orbit(jnp.array([0.1, vx0]))
        o.integrate(jnp.asarray(_TS), _DISK1D, method="diffrax")
        return o.getOrbit()[-1, 0]

    for fn, x0, eps in ((planar_final_R, 0.1, 1e-6), (linear_final_x, 0.05, 1e-4)):
        g = float(jax.grad(fn)(x0))
        fd = float((fn(x0 + eps) - fn(x0 - eps)) / (2 * eps))
        numpy.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-7)


# ------------------------------------------- backend IC + plain numpy output times
@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
def test_integrate_diffrax_numpy_times():
    # a backend IC with a plain numpy times array (not differentiating w.r.t. time)
    # is fine: the times are moved onto the IC's backend
    pot = PlummerPotential(amp=1.0, b=0.6)
    o = Orbit(jnp.asarray(_IC))
    o.integrate(_TS, pot, method="diffrax")  # _TS is a numpy array
    got = numpy.asarray(o.getOrbit())
    numpy.testing.assert_allclose(
        _wrap_phi(got), _wrap_phi(_c_reference(pot)), rtol=1e-8, atol=1e-9
    )


# ------------------- concrete backend IC + numpy/C method works (forced-backend suite)
@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
def test_integrate_concrete_backend_ic_numpy_method_works():
    # a CONCRETE (eager) jax-IC Orbit keeps its real values in self.vxvv, so a
    # numpy/C integrator runs normally and matches the numpy-IC result -- this lets
    # the existing suite be driven under a forced jax/torch backend.
    pot = PlummerPotential(amp=1.0, b=0.6)
    o = Orbit(jnp.asarray(_IC))
    o.integrate(_TS, pot, method="dop853_c")
    ref = Orbit(list(_IC))
    ref.integrate(_TS, pot, method="dop853_c")
    numpy.testing.assert_allclose(o.R(_TS), ref.R(_TS), rtol=1e-12, atol=1e-12)


# ------------------- traced backend IC + numpy/C method must raise (no real values)
@pytest.mark.skipif(not HAVE_TORCH, reason="torch/torchdiffeq not installed")
def test_integrate_gradtracking_backend_ic_numpy_method_raises_torch():
    # a grad-tracking torch IC has no concrete values (self.vxvv is a placeholder),
    # so a numpy/C method must raise rather than integrate degenerate values
    o = Orbit(torch.tensor(_IC, requires_grad=True))
    with pytest.raises(ValueError):
        o.integrate(_TS, PlummerPotential(amp=1.0, b=0.6), method="leapfrog_c")


@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
def test_integrate_traced_backend_ic_numpy_method_raises_jax():
    # under jax.grad the IC is a tracer -> placeholder self.vxvv -> a numpy/C method
    # must raise (a differentiable method must be used instead)
    pot = PlummerPotential(amp=1.0, b=0.6)

    def run(vR0):
        o = Orbit(jnp.array([1.0, vR0, 0.9, 0.2, 0.05, 0.3]))
        o.integrate(_TS, pot, method="dop853_c")
        return o.getOrbit()[-1, 0]

    with pytest.raises(ValueError):
        jax.grad(run)(0.1)


# ----------------------------------------- accessors on a backend orbit raise clearly
@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
def test_integrate_backend_orbit_accessor_raises():
    o = Orbit(jnp.asarray(_IC))
    o.integrate(jnp.asarray(_TS), PlummerPotential(amp=1.0, b=0.6), method="diffrax")
    with pytest.raises(NotImplementedError):
        o.R(_TS)
    with pytest.raises(NotImplementedError):
        o.E(pot=PlummerPotential(amp=1.0, b=0.6))


# ------------------------------------------- differentiate w.r.t. integration times
@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
def test_integrate_diffrax_grad_time_and_jit():
    pot = PlummerPotential(amp=1.0, b=0.6)

    def final_R(tmax):
        # times built inside the trace -> a backend (traced) ts must not be forced
        # through numpy
        o = Orbit(jnp.asarray(_IC))
        o.integrate(jnp.linspace(0.0, tmax, 60), pot, method="diffrax")
        return o.getOrbit()[-1, 0]

    g = float(jax.grad(final_R)(6.0))
    eps = 1e-5
    fd = float((final_R(6.0 + eps) - final_R(6.0 - eps)) / (2 * eps))
    numpy.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-6)
    # and the same path survives jax.jit (in-trace time grid)
    rjit = float(jax.jit(final_R)(6.0))
    numpy.testing.assert_allclose(rjit, float(final_R(6.0)), rtol=1e-8, atol=1e-10)


# --------------------------------------------------- torch potential-param gradient
@pytest.mark.skipif(not (HAVE_JAX and HAVE_TORCH), reason="needs both jax and torch")
def test_integrate_torchdiffeq_grad_param_matches_diffrax():
    # gradient of final R w.r.t. Plummer scale b, torch.autograd vs jax.grad
    def final_R_jax(b):
        pot = PlummerPotential(amp=1.0, b=b)
        o = Orbit(jnp.asarray(_IC))
        o.integrate(jnp.asarray(_TS), pot, method="diffrax")
        return o.getOrbit()[-1, 0]

    g_jax = float(jax.grad(final_R_jax)(0.6))
    b = torch.tensor(0.6, requires_grad=True)
    o = Orbit(torch.as_tensor(_IC))
    o.integrate(
        torch.as_tensor(_TS), PlummerPotential(amp=1.0, b=b), method="torchdiffeq"
    )
    o.getOrbit()[-1, 0].backward()
    numpy.testing.assert_allclose(float(b.grad), g_jax, rtol=1e-6, atol=1e-8)


# --------------------------------------------------------------- Quantity-time input
@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
def test_integrate_diffrax_quantity_time():
    # Quantity output times are parsed to natural units, like the numpy path
    units = pytest.importorskip("astropy.units")  # backend CI job runs astropy-free

    pot = PlummerPotential(amp=1.0, b=0.6)
    o = Orbit(jnp.asarray(_IC))
    o.integrate(_TS * units.Gyr, pot, method="diffrax")
    got = numpy.asarray(o.getOrbit())
    oc = Orbit(_IC)
    oc.integrate(_TS * units.Gyr, pot, method="dop853_c")
    ref = numpy.array(
        [
            oc.R(_TS * units.Gyr),
            oc.vR(_TS * units.Gyr),
            oc.vT(_TS * units.Gyr),
            oc.z(_TS * units.Gyr),
            oc.vz(_TS * units.Gyr),
            oc.phi(_TS * units.Gyr),
        ]
    ).T
    # _TS Gyr parses to ~28x more natural time units (1 natural time ~ 0.036 Gyr),
    # so this is a much longer integration than the other parity tests; the looser
    # tolerance reflects the larger accumulated Dopri8-vs-dop853 difference (the
    # point of this test is the Quantity parsing, not tight parity).
    numpy.testing.assert_allclose(_wrap_phi(got), _wrap_phi(ref), rtol=1e-6, atol=1e-7)


# ------------------------------------------------------------- numpy path unaffected
def test_integrate_numpy_path_unchanged():
    # a numpy Orbit with a standard method is untouched by the new dispatch
    pot = MiyamotoNagaiPotential(normalize=1.0)
    o = Orbit(_IC)
    o.integrate(_TS, pot, method="dop853_c")
    got = o.getOrbit()
    assert isinstance(got, numpy.ndarray)
    assert got.shape == (len(_TS), 6)
    # reference value from the C integrator (independent of this PR's dispatch)
    o2 = Orbit(_IC)
    o2.integrate(_TS, pot, method="dop853")
    numpy.testing.assert_allclose(o.R(_TS), o2.R(_TS), rtol=1e-6, atol=1e-7)


# ------------------------------------------- inbackend_ode defensive guards (no jax/torch needed)
def test_inbackend_to_eom_unsupported_phasedim_raises():
    # _to_eom validates the phase-space dimension; the public Orbit path only ever
    # passes phasedim 2-6, so this guards a direct misuse (and covers the branch).
    from galpy.backend._reference.inbackend_ode import _to_eom

    with pytest.raises(ValueError, match="unsupported phase-space dimension"):
        _to_eom(numpy, numpy.zeros(7))


def test_inbackend_integrate_orbit_numpy_input_raises():
    # integrate_orbit needs a jax/torch array (it picks the solver from the input
    # namespace); a numpy input is rejected with a clear message.
    from galpy.backend._reference.inbackend_ode import integrate_orbit

    with pytest.raises(NotImplementedError, match="requires a jax or torch"):
        integrate_orbit(
            PlummerPotential(amp=1.0, b=0.6),
            numpy.array(_IC),
            numpy.linspace(0.0, 1.0, 5),
        )
