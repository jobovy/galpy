###############################################################################
# test_backend_halobar.py: per-family backend tests for the halo + bar +
# non-axisymmetric analytic potentials migrated in P2.4.
#
# For each migrated potential and each migrated compute method, this asserts
#   1. numpy / jax / torch produce identical values (rtol=1e-12, atol=1e-14),
#   2. autodiff (jax.grad / torch.autograd) of _evaluate matches central finite
#      differences (rtol 1e-5).
#
# Backends that are not installed self-skip, so this is green on numpy alone.
# Structure mirrors tests/test_backend_pilot.py.
###############################################################################
import numpy
import pytest

from galpy.potential import (
    CosmphiDiskPotential,
    DehnenBarPotential,
    EllipticalDiskPotential,
    HenonHeilesPotential,
    LogarithmicHaloPotential,
    LopsidedDiskPotential,
    SoftenedNeedleBarPotential,
    SpiralArmsPotential,
    SteadyLogSpiralPotential,
    TransientLogSpiralPotential,
)

# This module manages backends explicitly (parametrizes over them), so it is
# exempt from the global --backend force fixture.
pytestmark = pytest.mark.backend_managed

# Discover available backends
BACKENDS = ["numpy"]
try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    BACKENDS.append("jax")
except ImportError:  # pragma: no cover
    jax = None
try:
    import torch

    BACKENDS.append("torch")
except ImportError:  # pragma: no cover
    torch = None

AD_BACKENDS = [b for b in BACKENDS if b != "numpy"]

# Methods grouped by signature.
# 3D potentials: methods take (R, z, phi, t).
THREED_METHODS = [
    "_evaluate",
    "_Rforce",
    "_zforce",
    "_phitorque",
    "_R2deriv",
    "_z2deriv",
    "_Rzderiv",
    "_phi2deriv",
    "_Rphideriv",
    "_phizderiv",
    "_dens",
]
# planar potentials: methods take (R, phi, t).
PLANAR_METHODS = [
    "_evaluate",
    "_Rforce",
    "_phitorque",
    "_R2deriv",
    "_phi2deriv",
    "_Rphideriv",
]


# Each entry: (instance, ndim, [methods]). ndim==3 -> (R,z,phi); ndim==2 -> (R,phi)
def _make_pots():
    pots = []
    # --- 3D potentials -------------------------------------------------------
    pots.append(
        (
            LogarithmicHaloPotential(amp=1.3, q=0.8, core=0.1),
            3,
            [m for m in THREED_METHODS],
        )
    )
    # triaxial (nonaxi) LogarithmicHalo exercises the phi-dependent branches
    pots.append(
        (
            LogarithmicHaloPotential(amp=1.1, q=0.9, b=0.7, core=0.1),
            3,
            [m for m in THREED_METHODS],
        )
    )
    # DehnenBar (fully grown so smooth==1); all 2nd derivs supported
    pots.append(
        (
            DehnenBarPotential(
                amp=1.0, omegab=1.8, rb=0.6, Af=0.03, tform=-100.0, tsteady=1.0
            ),
            3,
            [
                "_evaluate",
                "_Rforce",
                "_zforce",
                "_phitorque",
                "_R2deriv",
                "_z2deriv",
                "_Rzderiv",
                "_phi2deriv",
                "_Rphideriv",
                "_phizderiv",
            ],
        )
    )
    # SoftenedNeedleBar's force methods use a per-call md5 hash cache that only
    # supports scalar inputs (true on every backend, both before and after the
    # migration); _evaluate/_dens are fully vectorized.
    pots.append(
        (
            SoftenedNeedleBarPotential(
                amp=1.2, a=4.0, b=0.5, c=1.0, pa=0.3, omegab=1.8
            ),
            3,
            ["_evaluate", "_dens"],
            ["_Rforce", "_zforce", "_phitorque"],
        )
    )
    pots.append(
        (
            SpiralArmsPotential(
                N=3,
                alpha=0.15,
                Cs=[8.0 / (3.0 * numpy.pi), 0.5, 8.0 / (15.0 * numpy.pi)],
                omega=0.3,
                Rs=0.4,
                H=0.2,
            ),
            3,
            [m for m in THREED_METHODS],
        )
    )
    # --- planar potentials ---------------------------------------------------
    pots.append((HenonHeilesPotential(amp=1.2), 2, [m for m in PLANAR_METHODS]))
    pots.append(
        (
            CosmphiDiskPotential(amp=1.1, phib=0.3, p=1.0, phio=0.02, m=4, r1=1.0),
            2,
            [m for m in PLANAR_METHODS],
        )
    )
    # CosmphiDisk with break radius rb (exercises the inside-rb where-branch)
    pots.append(
        (
            CosmphiDiskPotential(
                amp=1.0, phib=0.3, p=1.0, phio=0.02, m=4, r1=1.0, rb=1.2
            ),
            2,
            [m for m in PLANAR_METHODS],
        )
    )
    pots.append(
        (
            LopsidedDiskPotential(amp=1.0, phib=0.3, p=1.0, phio=0.02),
            2,
            [m for m in PLANAR_METHODS],
        )
    )
    pots.append(
        (
            EllipticalDiskPotential(amp=1.0, phib=0.3, p=1.0, twophio=0.02),
            2,
            [m for m in PLANAR_METHODS],
        )
    )
    pots.append(
        (
            SteadyLogSpiralPotential(amp=1.0, omegas=0.65, A=-0.035, alpha=-7.0, m=2),
            2,
            ["_evaluate", "_Rforce", "_phitorque"],
        )
    )
    pots.append(
        (
            TransientLogSpiralPotential(
                amp=1.0, omegas=0.65, A=-0.035, alpha=-7.0, m=2, sigma=1.0, to=0.0
            ),
            2,
            ["_evaluate", "_Rforce", "_phitorque"],
        )
    )
    # Normalize every entry to (instance, ndim, array_methods, scalar_only_methods)
    return [e if len(e) == 4 else (*e, []) for e in pots]


POTS = _make_pots()


def _id(entry):
    p, ndim = entry[0], entry[1]
    suffix = ""
    if isinstance(p, LogarithmicHaloPotential) and p.isNonAxi:
        suffix = "_triax"
    if isinstance(p, CosmphiDiskPotential) and not isinstance(p, LopsidedDiskPotential):
        suffix = "_rb" if p._rb > 0.0 else ""
    return type(p).__name__ + suffix


POT_IDS = [_id(e) for e in POTS]

# Evaluation grid (kept away from R=0 / z=0 singularities of the bar potentials).
_RS = numpy.array([0.6, 1.0, 1.7])
_ZS = numpy.array([0.1, 0.2, 0.35])
_PHIS = numpy.array([0.3, 0.9, 1.6])
_T = 0.0


def _asarray(backend_name, x, requires_grad=False):
    if backend_name == "numpy":
        return numpy.asarray(x, dtype=float)
    if backend_name == "jax":
        return jnp.asarray(x, dtype=jnp.float64)
    if backend_name == "torch":
        return torch.tensor(x, dtype=torch.float64, requires_grad=requires_grad)


def _tonumpy(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().numpy()
    return numpy.asarray(x)


def _call(method, backend_name, ndim):
    R = _asarray(backend_name, _RS)
    phi = _asarray(backend_name, _PHIS)
    if ndim == 3:
        z = _asarray(backend_name, _ZS)
        return method(R, z, phi, _T)
    return method(R, phi, _T)


def _call_numpy(method, ndim):
    if ndim == 3:
        return numpy.asarray(method(_RS, _ZS, _PHIS, _T))
    return numpy.asarray(method(_RS, _PHIS, _T))


def _call_scalar(method, backend_name, ndim, R0, z0, phi0):
    R = _asarray(backend_name, R0)
    phi = _asarray(backend_name, phi0)
    if ndim == 3:
        z = _asarray(backend_name, z0)
        return method(R, z, phi, _T)
    return method(R, phi, _T)


@pytest.mark.parametrize("entry", POTS, ids=POT_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_value_parity(backend_name, entry):
    pot, ndim, methods, scalar_only = entry
    for mname in methods:
        method = getattr(pot, mname)
        ref = _call_numpy(method, ndim)
        got = _tonumpy(_call(method, backend_name, ndim))
        numpy.testing.assert_allclose(
            got,
            ref,
            rtol=1e-12,
            atol=1e-14,
            err_msg=f"{type(pot).__name__}.{mname} parity ({backend_name})",
        )
    # methods that only support scalar inputs: probe element-wise on the grid
    for mname in scalar_only:
        method = getattr(pot, mname)
        for R0, z0, phi0 in zip(_RS, _ZS, _PHIS):
            ref = numpy.asarray(
                method(R0, z0, phi0, _T) if ndim == 3 else method(R0, phi0, _T)
            )
            got = _tonumpy(_call_scalar(method, backend_name, ndim, R0, z0, phi0))
            numpy.testing.assert_allclose(
                got,
                ref,
                rtol=1e-12,
                atol=1e-14,
                err_msg=f"{type(pot).__name__}.{mname} scalar parity ({backend_name})",
            )


@pytest.mark.parametrize("entry", POTS, ids=POT_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_grad_evaluate_vs_finite_difference(backend_name, entry):
    pot, ndim = entry[0], entry[1]
    R0, z0, phi0 = 1.3, 0.4, 0.7
    eps = 1e-6

    def phi_np(R):
        if ndim == 3:
            return float(pot._evaluate(numpy.asarray(R), numpy.asarray(z0), phi0, _T))
        return float(pot._evaluate(numpy.asarray(R), phi0, _T))

    fd = (phi_np(R0 + eps) - phi_np(R0 - eps)) / (2 * eps)
    if backend_name == "jax":
        if ndim == 3:
            f = lambda R: pot._evaluate(R, jnp.asarray(z0), jnp.asarray(phi0), _T)
        else:
            f = lambda R: pot._evaluate(R, jnp.asarray(phi0), _T)
        ad = float(jax.grad(f)(jnp.asarray(R0)))
    else:
        R = torch.tensor(R0, dtype=torch.float64, requires_grad=True)
        if ndim == 3:
            y = pot._evaluate(
                R, torch.tensor(z0, dtype=torch.float64), torch.tensor(phi0), _T
            )
        else:
            y = pot._evaluate(R, torch.tensor(phi0, dtype=torch.float64), _T)
        y.backward()
        ad = float(R.grad)
    numpy.testing.assert_allclose(
        ad, fd, rtol=1e-5, err_msg=f"{type(pot).__name__} grad ({backend_name})"
    )
