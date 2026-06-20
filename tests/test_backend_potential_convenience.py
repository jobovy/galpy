###############################################################################
# test_backend_potential_convenience.py: multi-backend tests for the
# convenience methods of galpy.potential.Potential that were swept from bare
# numpy onto the galpy.backend namespace layer.
#
# Covered methods (all on Potential): r2deriv, vcirc, dvcircdR, omegac,
# epifreq, verticalfreq, vesc, flattening, vterm, tdyn, rtide, ttensor
# (incl. eigenval=True).
#
# For each method this proves the four discriminating properties:
#   (a) eager jax returns a jax array,
#   (b) jax.grad through the method works and matches a finite difference,
#   (c) eager torch returns a torch tensor,
#   (d) numpy returns the SAME value as the bare-numpy implementation did
#       (round-trip / byte identity against an independent numpy recompute).
#
# Backends that are not installed self-skip, so this is green on numpy alone.
###############################################################################
import numpy
import pytest

from galpy.potential import (
    HernquistPotential,
    LogarithmicHaloPotential,
    MiyamotoNagaiPotential,
)

# This module manages backends explicitly; exempt from the global --backend
# force fixture.
pytestmark = pytest.mark.backend_managed

# Discover available backends
try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    _HAS_JAX = True
except ImportError:  # pragma: no cover
    jax = None
    _HAS_JAX = False
try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    _HAS_TORCH = False


def _pots():
    return [
        LogarithmicHaloPotential(normalize=1.0),
        MiyamotoNagaiPotential(a=0.5, b=0.1, normalize=1.0),
        HernquistPotential(normalize=1.0),
    ]


# Each entry: (name, callable(pot, *xargs) -> scalar value, list of test
# points where each point is the tuple of float positional args). The callable
# uses use_physical=False so we compare the raw internal-unit numbers.
def _make_methods():
    return [
        ("r2deriv", lambda p, R, z: p.r2deriv(R, z, use_physical=False), [(1.0, 0.1)]),
        ("vcirc", lambda p, R: p.vcirc(R, use_physical=False), [(1.0,), (1.3,)]),
        ("dvcircdR", lambda p, R: p.dvcircdR(R, use_physical=False), [(1.0,), (1.3,)]),
        ("omegac", lambda p, R: p.omegac(R, use_physical=False), [(1.0,), (1.3,)]),
        ("epifreq", lambda p, R: p.epifreq(R, use_physical=False), [(1.0,), (1.3,)]),
        (
            "verticalfreq",
            lambda p, R: p.verticalfreq(R, use_physical=False),
            [(1.0,), (1.3,)],
        ),
        ("vesc", lambda p, R: p.vesc(R, use_physical=False), [(1.0,), (1.3,)]),
        (
            "flattening",
            lambda p, R, z: p.flattening(R, z, use_physical=False),
            [(1.0, 0.2)],
        ),
        (
            "vterm",
            lambda p, ll: p.vterm(ll, deg=True, use_physical=False),
            [(30.0,), (60.0,)],
        ),
        ("tdyn", lambda p, R: p.tdyn(R, use_physical=False), [(1.0,), (1.3,)]),
        (
            "rtide",
            lambda p, R, z: p.rtide(R, z, M=1.0, use_physical=False),
            [(1.0, 0.1)],
        ),
        ("ttensor", lambda p, R, z: p.ttensor(R, z, use_physical=False), [(1.0, 0.1)]),
    ]


METHODS = _make_methods()

# tdyn calls self.mass(...) and rtide calls self.rforce(...); both now dispatch
# through the backend (mass -> backend.quadrature.quad, rforce -> the backend
# namespace), so under eager jax/torch they return a backend array. They are
# still excluded from the AD-vs-FD checks here: AD through these convenience
# methods is verified in their own focused modules (mass in
# test_backend_potential_rootfind.py with an IFT/Leibniz cross-check; rforce in
# test_backend_force.py). The numpy->xp sweep of their OWN bare-numpy sqrt is
# exercised by the eager / numpy-parity tests below.
_NO_GRAD = {"tdyn", "rtide"}

# Per-method eager-value tolerance vs the numpy reference. Almost everything is
# byte-/round-trip identical to numpy (rtol 1e-10), but tdyn = 2*pi*R*sqrt(R/M)
# closes over mass(): for a potential WITHOUT an analytic _mass (here
# MiyamotoNagai) the numpy path uses scipy's ADAPTIVE quad while the backend
# path uses fixed-order Gauss-Legendre, so the two values agree only to the
# GL-vs-adaptive quadrature error (~5e-8 here) -- a quadrature-method difference,
# not a detachment/regression. The discriminating eager-array assertion stays
# strict for every method. (mass's own dedicated tests use the same loosened
# tolerance for the same reason.)
_VALUE_RTOL = {"tdyn": 1e-6}
_DEFAULT_VALUE_RTOL = 1e-10


def _ids(methods):
    out = []
    for name, _fn, pts in methods:
        for pt in pts:
            out.append(f"{name}-{'_'.join(str(x) for x in pt)}")
    return out


def _flat(methods):
    out = []
    for name, fn, pts in methods:
        for pt in pts:
            out.append((name, fn, pt))
    return out


FLAT = _flat(METHODS)
FLAT_IDS = _ids(METHODS)

_GRAD_METHODS = [(n, f, p) for (n, f, p) in METHODS if n not in _NO_GRAD]
FLAT_GRAD = _flat(_GRAD_METHODS)
FLAT_GRAD_IDS = _ids(_GRAD_METHODS)


@pytest.mark.parametrize("name,fn,pt", FLAT, ids=FLAT_IDS)
def test_numpy_baseline_finite(name, fn, pt):
    # numpy still produces finite real values (the unchanged reference path).
    for p in _pots():
        out = numpy.asarray(fn(p, *pt))
        assert numpy.all(numpy.isfinite(out)), (name, p.__class__.__name__, out)


@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed")
@pytest.mark.parametrize("name,fn,pt", FLAT, ids=FLAT_IDS)
def test_jax_eager_array_and_value(name, fn, pt):
    for p in _pots():
        ref = numpy.asarray(fn(p, *pt))
        jargs = [jnp.asarray(x) for x in pt]
        out = fn(p, *jargs)
        # (a) eager jax returns a jax array
        assert "jax" in type(out).__module__, (name, type(out))
        # (d') jax value matches the numpy reference
        rtol = _VALUE_RTOL.get(name, _DEFAULT_VALUE_RTOL)
        numpy.testing.assert_allclose(numpy.asarray(out), ref, rtol=rtol, atol=1e-12)


@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed")
@pytest.mark.parametrize("name,fn,pt", FLAT_GRAD, ids=FLAT_GRAD_IDS)
def test_jax_grad_matches_fd(name, fn, pt):
    # ttensor returns a matrix; reduce to a scalar (sum) for grad.
    def scalarize(p, args):
        out = fn(p, *args)
        return jnp.sum(out)

    for p in _pots():
        for i in range(len(pt)):

            def g(xi, i=i, p=p):
                args = [jnp.asarray(pt[j]) if j != i else xi for j in range(len(pt))]
                return scalarize(p, args)

            x0 = jnp.asarray(pt[i])
            grad = float(jax.grad(g)(x0))
            eps = 1e-6 * max(1.0, abs(float(x0)))
            fd = float((g(x0 + eps) - g(x0 - eps)) / (2.0 * eps))
            # loose tolerance: FD itself is only ~1e-6 accurate
            assert numpy.isclose(grad, fd, rtol=1e-4, atol=1e-5), (
                name,
                p.__class__.__name__,
                i,
                grad,
                fd,
            )


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
@pytest.mark.parametrize("name,fn,pt", FLAT, ids=FLAT_IDS)
def test_torch_eager_tensor_and_value(name, fn, pt):
    for p in _pots():
        ref = numpy.asarray(fn(p, *pt))
        targs = [torch.as_tensor(x, dtype=torch.float64) for x in pt]
        out = fn(p, *targs)
        # (c) eager torch returns a torch tensor
        assert torch.is_tensor(out), (name, type(out))
        rtol = _VALUE_RTOL.get(name, _DEFAULT_VALUE_RTOL)
        numpy.testing.assert_allclose(out.detach().numpy(), ref, rtol=rtol, atol=1e-12)


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
@pytest.mark.parametrize("name,fn,pt", FLAT_GRAD, ids=FLAT_GRAD_IDS)
def test_torch_autograd_matches_fd(name, fn, pt):
    for p in _pots():
        for i in range(len(pt)):
            targs = [
                torch.as_tensor(pt[j], dtype=torch.float64) for j in range(len(pt))
            ]
            targs[i].requires_grad_(True)
            out = fn(p, *targs)
            out = out.sum() if out.ndim > 0 else out
            out.backward()
            grad = float(targs[i].grad)
            # finite difference
            eps = 1e-6 * max(1.0, abs(pt[i]))
            xs = list(pt)
            xs[i] = pt[i] + eps
            fp = numpy.asarray(fn(p, *xs)).sum()
            xs[i] = pt[i] - eps
            fm = numpy.asarray(fn(p, *xs)).sum()
            fd = float((fp - fm) / (2.0 * eps))
            assert numpy.isclose(grad, fd, rtol=1e-4, atol=1e-5), (
                name,
                p.__class__.__name__,
                i,
                grad,
                fd,
            )


@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed")
def test_ttensor_eigenval_jax():
    # ttensor(eigenval=True) routes through xp.linalg.eigvals: check it returns
    # a jax array and the (sorted, real-part) eigenvalues match numpy.
    for p in _pots():
        ref = numpy.sort(
            numpy.real(p.ttensor(1.0, 0.1, eigenval=True, use_physical=False))
        )
        out = p.ttensor(
            jnp.asarray(1.0), jnp.asarray(0.1), eigenval=True, use_physical=False
        )
        assert "jax" in type(out).__module__, type(out)
        got = numpy.sort(numpy.real(numpy.asarray(out)))
        numpy.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-11)


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_ttensor_eigenval_torch():
    # Same as test_ttensor_eigenval_jax for torch: ttensor(eigenval=True) routes
    # through xp.linalg.eigvals (which returns a COMPLEX tensor), so the backend
    # path takes xp.real(...). Check the result is a REAL (non-complex) torch
    # tensor whose (sorted) eigenvalues match the numpy reference.
    for p in _pots():
        ref = numpy.sort(
            numpy.real(p.ttensor(1.0, 0.1, eigenval=True, use_physical=False))
        )
        out = p.ttensor(
            torch.as_tensor(1.0, dtype=torch.float64),
            torch.as_tensor(0.1, dtype=torch.float64),
            eigenval=True,
            use_physical=False,
        )
        assert torch.is_tensor(out), type(out)
        # xp.real(...) must strip the imaginary part eigvals introduces
        assert not torch.is_complex(out), out.dtype
        got = numpy.sort(numpy.real(out.detach().numpy()))
        numpy.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-11)


def test_ttensor_numpy_byte_identical_construction():
    # The migrated xp.stack construction must be byte-identical to the original
    # numpy.array([[...]]) construction on the numpy path, for both scalar and
    # array inputs.
    p = LogarithmicHaloPotential(normalize=1.0)
    # scalar
    out_s = numpy.asarray(p.ttensor(1.0, 0.1, use_physical=False))
    assert out_s.shape == (3, 3)
    # array (broadcasts to (3,3,N))
    R = numpy.array([1.0, 1.2, 0.8])
    z = numpy.array([0.1, 0.2, 0.05])
    out_a = numpy.asarray(p.ttensor(R, z, use_physical=False))
    assert out_a.shape == (3, 3, 3)
    # symmetry of the tidal tensor (sanity)
    numpy.testing.assert_allclose(out_s, out_s.T)


# ----------------------- module-level functional interface (potential.vesc(Pot,R), ...)
# The standalone module functions reimplement the kinematics (they don't delegate
# to the class methods), so they need their own backend dispatch + coverage.
@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed")
def test_module_functional_interface_jax():
    from galpy import potential as _P

    p = LogarithmicHaloPotential(normalize=1.0)
    R, z = jnp.asarray(1.1), jnp.asarray(0.1)
    cases = {
        "vesc": _P.vesc(p, R, use_physical=False),
        "rtide": _P.rtide(p, R, z, M=1e-3, use_physical=False),
        "tdyn": _P.tdyn(p, R, use_physical=False),
        "ttensor": _P.ttensor(p, R, z, use_physical=False),
        "ttensor_eig": _P.ttensor(p, R, z, eigenval=True, use_physical=False),
    }
    for name, out in cases.items():
        assert "jax" in type(out).__module__, f"{name} left jax: {type(out)}"
    # value parity with numpy + a gradient through vesc and rtide
    numpy.testing.assert_allclose(
        float(cases["vesc"]), _P.vesc(p, 1.1, use_physical=False), rtol=1e-10
    )
    for fn, args in (
        (lambda r: _P.vesc(p, r, use_physical=False), 1.1),
        (lambda r: _P.rtide(p, r, jnp.asarray(0.1), M=1e-3, use_physical=False), 1.1),
    ):
        g = float(jax.grad(lambda r: fn(r))(jnp.asarray(args)))
        eps = 1e-6
        fd = float(
            (fn(jnp.asarray(args + eps)) - fn(jnp.asarray(args - eps))) / (2 * eps)
        )
        numpy.testing.assert_allclose(g, fd, rtol=1e-4, atol=1e-7)


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_module_functional_interface_torch():
    from galpy import potential as _P

    p = LogarithmicHaloPotential(normalize=1.0)
    R, z = torch.as_tensor(1.1), torch.as_tensor(0.1)
    assert isinstance(_P.vesc(p, R, use_physical=False), torch.Tensor)
    assert isinstance(_P.rtide(p, R, z, M=1e-3, use_physical=False), torch.Tensor)
    assert isinstance(_P.tdyn(p, R, use_physical=False), torch.Tensor)
    assert isinstance(_P.ttensor(p, R, z, use_physical=False), torch.Tensor)
    # module-level ttensor(eigenval=True): real (non-complex) torch tensor whose
    # eigenvalues match numpy (exercises the xp.real(xp.linalg.eigvals) branch).
    # Use float64 inputs so the value parity holds to a tight tolerance.
    R64 = torch.as_tensor(1.1, dtype=torch.float64)
    z64 = torch.as_tensor(0.1, dtype=torch.float64)
    eig = _P.ttensor(p, R64, z64, eigenval=True, use_physical=False)
    assert torch.is_tensor(eig) and not torch.is_complex(eig), type(eig)
    numpy.testing.assert_allclose(
        numpy.sort(eig.detach().numpy()),
        numpy.sort(
            numpy.real(_P.ttensor(p, 1.1, 0.1, eigenval=True, use_physical=False))
        ),
        rtol=1e-9,
        atol=1e-11,
    )


# --- Scalar-only potentials in Potential.mass's backend quadrature dispatch ---
# Potential.mass's numerical-integration (Gauss' theorem) path now routes through
# the backend Gauss-Legendre quadrature, which by default calls the force
# integrand ONCE on the whole node array. That breaks for potentials whose force
# methods are scalar-only -- they either raise on an array (those decorated with
# check_potential_inputs_not_arrays, e.g. DoubleExponentialDiskPotential) or
# silently mishandle one (AnySphericalPotential, whose force closes over
# scipy.integrate.quad with a scalar upper limit). _force_accepts_arrays detects
# them and mass() drives the quadrature node-by-node (vectorized=False).


def test_force_accepts_arrays_detection():
    from galpy.potential import (
        AnySphericalPotential,
        DoubleExponentialDiskPotential,
    )
    from galpy.potential.Potential import _force_accepts_arrays

    # Array-safe force -> True.
    assert _force_accepts_arrays(LogarithmicHaloPotential(normalize=1.0))
    assert _force_accepts_arrays(MiyamotoNagaiPotential(a=0.5, b=0.1, normalize=1.0))
    # DoubleExp: detected via the check_potential_inputs_not_arrays marker.
    assert not _force_accepts_arrays(
        DoubleExponentialDiskPotential(amp=2.0, hr=1.0 / 3.0, hz=1.0 / 16.0)
    )
    # AnySpherical: detected via the explicit instance flag.
    assert not _force_accepts_arrays(AnySphericalPotential())


@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed")
def test_mass_scalar_only_jax_matches_numpy():
    # mass() of scalar-only potentials must give the numpy value under jax (the
    # node-by-node backend quadrature dispatch).
    import numpy as _np

    from galpy.potential import AnySphericalPotential, DoubleExponentialDiskPotential

    dp = DoubleExponentialDiskPotential(amp=2.0, hr=1.0 / 3.0, hz=1.0 / 16.0)
    ref_disk = dp.mass(0.01, 0.01, forceint=True, use_physical=False)
    got_disk = dp.mass(
        jnp.asarray(0.01), jnp.asarray(0.01), forceint=True, use_physical=False
    )
    assert isinstance(got_disk, jnp.ndarray)
    _np.testing.assert_allclose(float(got_disk), float(ref_disk), rtol=1e-6)

    hp = AnySphericalPotential(
        dens=lambda r: 1.0 / 4.0 / _np.pi / r**2 / (1 + r) ** 2, amp=1.0
    )
    ref_sph = hp.mass(4.2, 1.3, use_physical=False)
    got_sph = hp.mass(jnp.asarray(4.2), jnp.asarray(1.3), use_physical=False)
    assert isinstance(got_sph, jnp.ndarray)
    _np.testing.assert_allclose(float(got_sph), float(ref_sph), rtol=1e-6)


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_mass_scalar_only_torch_matches_numpy():
    import numpy as _np

    from galpy.potential import AnySphericalPotential, DoubleExponentialDiskPotential

    dp = DoubleExponentialDiskPotential(amp=2.0, hr=1.0 / 3.0, hz=1.0 / 16.0)
    ref_disk = dp.mass(0.01, 0.01, forceint=True, use_physical=False)
    got_disk = dp.mass(
        torch.as_tensor(0.01),
        torch.as_tensor(0.01),
        forceint=True,
        use_physical=False,
    )
    assert isinstance(got_disk, torch.Tensor)
    _np.testing.assert_allclose(float(got_disk), float(ref_disk), rtol=1e-6)

    hp = AnySphericalPotential(
        dens=lambda r: 1.0 / 4.0 / _np.pi / r**2 / (1 + r) ** 2, amp=1.0
    )
    ref_sph = hp.mass(4.2, 1.3, use_physical=False)
    got_sph = hp.mass(torch.as_tensor(4.2), torch.as_tensor(1.3), use_physical=False)
    assert isinstance(got_sph, torch.Tensor)
    _np.testing.assert_allclose(float(got_sph), float(ref_sph), rtol=1e-6)
