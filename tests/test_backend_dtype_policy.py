###############################################################################
# test_backend_dtype_policy.py: shared regression gate for the backend dtype
# policy across the migrated-potential registry.
#
# Policy: compute methods follow the dtype of their coordinate inputs --
# float32 in -> float32 out, float64 in -> float64 out. Most migrated
# potentials satisfy this naturally (their constants are Python floats, which
# are dtype-weak under torch/jax). The table-backed potentials (SCFPotential,
# DoubleExponentialDiskPotential, interpSphericalPotential/KingPotential,
# MultipoleExpansionPotential and the DiskSCF/DiskMultipole wrappers around
# them) deliberately keep float64 interiors -- expansion-coefficient tables,
# Ogata quadrature nodes/weights, spline coefficients; precision is the point
# -- and cast the result to the input dtype at method exit
# (galpy.backend.match_input_dtype). This module asserts
#   1. torch float32 inputs -> float32 output and float64 -> float64 for the
#      gate methods (_evaluate and _Rforce where migrated) of EVERY registry
#      entry,
#   2. the float64-QUALITY pin for the four table-backed potentials: the
#      torch-float32-input result equals the float64 result cast to float32
#      (an implementation that anchored the tables to float32 -- degrading
#      the interior -- would fail this),
#   3. the numpy semantics of the exit cast: float64 numpy inputs return the
#      raw result object unchanged (bit-identical no-op), float32 numpy
#      arrays now return float32 (the documented semantic change),
#   4. autodiff flows through the exit cast (jax.grad / torch.autograd at
#      float32 reproduce -_Rforce).
#
# Backends that are not installed self-skip, so this is green on numpy alone
# (the match_input_dtype no-op/cast branches are covered by the numpy tests).
###############################################################################
import numpy
import pytest

from galpy.backend import match_input_dtype
from galpy.potential import (
    BurkertPotential,
    CosmphiDiskPotential,
    DehnenBarPotential,
    DehnenCoreSphericalPotential,
    DehnenSphericalPotential,
    DiskMultipoleExpansionPotential,
    DiskSCFPotential,
    DoubleExponentialDiskPotential,
    EinastoPotential,
    EllipticalDiskPotential,
    FlattenedPowerPotential,
    HenonHeilesPotential,
    HernquistPotential,
    HomogeneousSpherePotential,
    IsochronePotential,
    IsothermalDiskPotential,
    JaffePotential,
    KeplerPotential,
    KGPotential,
    KingPotential,
    KuzminDiskPotential,
    LogarithmicHaloPotential,
    LopsidedDiskPotential,
    MiyamotoNagaiPotential,
    MultipoleExpansionPotential,
    NFWPotential,
    PerfectEllipsoidPotential,
    PlummerPotential,
    PowerSphericalPotential,
    PowerSphericalPotentialwCutoff,
    PowerTriaxialPotential,
    RazorThinExponentialDiskPotential,
    SCFPotential,
    SoftenedNeedleBarPotential,
    SphericalShellPotential,
    SpiralArmsPotential,
    SteadyLogSpiralPotential,
    TransientLogSpiralPotential,
    TriaxialGaussianPotential,
    TriaxialHernquistPotential,
    TriaxialJaffePotential,
    TriaxialNFWPotential,
    TwoPowerSphericalPotential,
    TwoPowerTriaxialPotential,
    interpSphericalPotential,
)

# This module manages backends explicitly, so it is exempt from the global
# --backend force fixture.
pytestmark = pytest.mark.backend_managed

# Discover available backends
try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    jax = None
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

# Known in-flight failures: gate failures for these ids skip (visibly) with a
# pointer to the branch carrying the fix instead of failing the suite.
# Currently empty: the former entries -- TwoPowerSpherical-generic (scipy
# hyp2f1 router bypass, fixed by the special-router migration + the router
# fallbacks' input-dtype exit cast) and PerfectEllipsoid/TriaxialNFW
# (all-tensor TypeError, fixed by the EllipsoidalPotential scalar-phi
# anchoring) -- have landed on feat/backends.
KNOWN_INFLIGHT = {}

_HERN = HernquistPotential(amp=1.3, a=0.8)


def _scf_coeffs():
    Acos = numpy.zeros((3, 3, 3))
    Asin = numpy.zeros((3, 3, 3))
    Acos[0, 0, 0] = 1.0
    Acos[1, 1, 0] = 0.05
    Acos[2, 2, 2] = 0.02
    Asin[1, 1, 1] = 0.03
    return Acos, Asin


def _make_registry():
    """One instance per migrated potential family member.

    Each entry: (id, instance, kind, gate methods); kind selects the call
    signature: '3d' -> (R, z, phi, t), 'planar' -> (R, phi, t), 'linear' ->
    (x, t). The gate methods default to (_evaluate, _Rforce) and are reduced
    where only other methods are migrated (noted per entry).
    """
    Acos, Asin = _scf_coeffs()
    entries = [
        # --- spherical family ------------------------------------------------
        ("Plummer", PlummerPotential(amp=1.3, b=0.7)),
        ("Isochrone", IsochronePotential(amp=2.0, b=1.1)),
        ("DehnenSpherical", DehnenSphericalPotential(amp=1.3, a=1.1, alpha=1.5)),
        ("DehnenCoreSpherical", DehnenCoreSphericalPotential(amp=1.3, a=1.1)),
        ("Hernquist", HernquistPotential(amp=1.3, a=1.1)),
        ("Jaffe", JaffePotential(amp=1.3, a=1.1)),
        ("NFW", NFWPotential(amp=1.3, a=1.1)),
        ("PowerSpherical", PowerSphericalPotential(amp=1.3, alpha=1.5)),
        # special-function-backed (gamma/gammainc router)
        (
            "PowerSphericalwCutoff",
            PowerSphericalPotentialwCutoff(amp=1.3, alpha=1.0, rc=1.2),
        ),
        ("Kepler", KeplerPotential(amp=1.3)),
        ("HomogeneousSphere", HomogeneousSpherePotential(amp=1.3, R=1.1)),
        ("Burkert", BurkertPotential(amp=1.3, a=1.1)),
        # special-function-backed (gamma/gammaincc router)
        ("Einasto", EinastoPotential(amp=1.3, h=1.1, n=1.5)),
        ("SphericalShell", SphericalShellPotential(amp=1.3, a=0.7)),
        (
            "TwoPowerSpherical-generic",
            TwoPowerSphericalPotential(amp=1.3, a=1.1, alpha=1.5, beta=3.5),
        ),
        # --- disk family -----------------------------------------------------
        ("MiyamotoNagai", MiyamotoNagaiPotential(amp=1.3, a=0.8, b=0.3)),
        ("KuzminDisk", KuzminDiskPotential(amp=1.2, a=0.7)),
        ("FlattenedPower", FlattenedPowerPotential(amp=1.1, alpha=0.5, q=0.9)),
        # only _surfdens is migrated (the rest use scipy.special directly)
        (
            "RazorThinExpDisk",
            RazorThinExponentialDiskPotential(amp=1.0, hr=0.4),
            "3d",
            ("_surfdens",),
        ),
        # table-backed (Ogata quadrature nodes/weights)
        (
            "DoubleExponentialDisk",
            DoubleExponentialDiskPotential(amp=1.3, hr=0.4, hz=0.1, de_n=1000),
        ),
        (
            "IsothermalDisk",
            IsothermalDiskPotential(amp=1.0, sigma=0.2),
            "linear",
            ("_evaluate", "_force"),
        ),
        (
            "KG",
            KGPotential(amp=1.0, K=1.15, F=0.03, D=1.8),
            "linear",
            ("_evaluate", "_force"),
        ),
        # --- ellipsoidal family ----------------------------------------------
        ("PerfectEllipsoid", PerfectEllipsoidPotential(amp=1.3, a=1.5, b=0.9, c=0.7)),
        (
            "TriaxialGaussian",
            TriaxialGaussianPotential(amp=1.3, sigma=1.5, b=0.9, c=0.7),
        ),
        ("PowerTriaxial", PowerTriaxialPotential(amp=1.3, alpha=1.2, b=0.9, c=0.7)),
        (
            "TriaxialHernquist",
            TriaxialHernquistPotential(amp=1.3, a=1.5, b=0.9, c=0.7),
        ),
        ("TriaxialJaffe", TriaxialJaffePotential(amp=1.3, a=1.5, b=0.9, c=0.7)),
        ("TriaxialNFW", TriaxialNFWPotential(amp=1.3, a=1.5, b=0.9, c=0.7)),
        # _evaluate deferred (hyp2f1); the forces are migrated
        (
            "TwoPowerTriaxial-generic",
            TwoPowerTriaxialPotential(
                amp=1.3, a=1.5, alpha=1.5, beta=3.5, b=0.9, c=0.7
            ),
            "3d",
            ("_Rforce",),
        ),
        # --- halo / bar / non-axisymmetric family -----------------------------
        ("LogarithmicHalo", LogarithmicHaloPotential(amp=1.3, q=0.8, core=0.1)),
        (
            "LogarithmicHalo-triaxial",
            LogarithmicHaloPotential(amp=1.1, q=0.9, b=0.7, core=0.1),
        ),
        (
            "DehnenBar",
            DehnenBarPotential(
                amp=1.0, omegab=1.8, rb=0.6, Af=0.03, tform=-100.0, tsteady=1.0
            ),
        ),
        (
            "SoftenedNeedleBar",
            SoftenedNeedleBarPotential(
                amp=1.2, a=4.0, b=0.5, c=1.0, pa=0.3, omegab=1.8
            ),
        ),
        ("SpiralArms", SpiralArmsPotential(N=2, alpha=0.2, Rs=0.3, H=0.125, omega=0.4)),
        ("HenonHeiles", HenonHeilesPotential(amp=1.2), "planar"),
        (
            "CosmphiDisk",
            CosmphiDiskPotential(amp=1.1, phib=0.3, p=1.0, phio=0.02, m=4, r1=1.0),
            "planar",
        ),
        (
            "LopsidedDisk",
            LopsidedDiskPotential(amp=1.0, phib=0.3, p=1.0, phio=0.02),
            "planar",
        ),
        (
            "EllipticalDisk",
            EllipticalDiskPotential(amp=1.0, phib=0.3, p=1.0, twophio=0.02),
            "planar",
        ),
        (
            "SteadyLogSpiral",
            SteadyLogSpiralPotential(amp=1.0, omegas=0.65, A=-0.035, alpha=-7.0, m=2),
            "planar",
            ("_evaluate", "_Rforce"),
        ),
        (
            "TransientLogSpiral",
            TransientLogSpiralPotential(
                amp=1.0, omegas=0.65, A=-0.035, alpha=-7.0, m=2, sigma=1.0, to=0.0
            ),
            "planar",
            ("_evaluate", "_Rforce"),
        ),
        # --- table-backed (float64 interior + exit cast) ----------------------
        ("SCF", SCFPotential(amp=1.3, Acos=Acos, Asin=Asin, a=1.2)),
        (
            "interpSpherical",
            interpSphericalPotential(
                rforce=HernquistPotential(amp=2.0, a=1.3),
                rgrid=numpy.geomspace(0.01, 1.5, 151),
            ),
        ),
        ("King", KingPotential(W0=3.0, M=2.3, rt=1.4)),
        (
            "Multipole",
            MultipoleExpansionPotential.from_density(
                lambda R, z, phi: (
                    (1.0 + 0.5 * numpy.cos(phi)) * _HERN(numpy.sqrt(R**2 + z**2), 0.0)
                ),
                L=4,
                rgrid=numpy.geomspace(1e-3, 10.0, 101),
            ),
        ),
        # wrappers around the table-backed expansions (self._me inherits the cast)
        (
            "DiskSCF",
            DiskSCFPotential(
                dens=lambda R, z: (
                    13.5 * numpy.exp(-3.0 * R) * numpy.exp(-27.0 * numpy.fabs(z))
                ),
                a=1.0,
                N=4,
                L=4,
            ),
        ),
        (
            "DiskMultipole",
            DiskMultipoleExpansionPotential(
                dens=lambda R, z: (
                    13.5 * numpy.exp(-3.0 * R) * numpy.exp(-27.0 * numpy.fabs(z))
                ),
                L=6,
            ),
        ),
    ]
    # Normalize each entry to (id, pot, kind, methods)
    return [
        (
            e[0],
            e[1],
            e[2] if len(e) > 2 else "3d",
            e[3] if len(e) > 3 else ("_evaluate", "_Rforce"),
        )
        for e in entries
    ]


REGISTRY = _make_registry()

# The four table-backed potentials (+ their inheritors) for the float64-quality
# pin: 3d-signature gate methods, scalar-capable.
_TABLE_IDS = ["SCF", "DoubleExponentialDisk", "interpSpherical", "King", "Multipole"]
_TABLE_ENTRIES = [e for e in REGISTRY if e[0] in _TABLE_IDS]

# Evaluation points: (R, z, phi, t); inside the interp spline domain and the
# r > rmax Kepler-extrapolation region (R=2.5 row).
_POINTS = [(0.7, 0.2, 0.3, 0.0), (1.1, 0.3, 0.4, 0.0), (2.5, 0.5, 1.1, 0.0)]


def _gate_args(kind, dtype, point=(1.1, 0.3, 0.4, 0.1)):
    R, z, phi, t = point
    if kind == "3d":
        vals = (R, z, phi, t)
    elif kind == "planar":
        vals = (R, phi, t)
    else:  # linear
        vals = (0.8, t)
    return [torch.tensor(v, dtype=dtype) for v in vals]


def _gate_params():
    for entry_id, pot, kind, methods in REGISTRY:
        for method in methods:
            yield pytest.param(pot, kind, method, entry_id, id=f"{entry_id}-{method}")


@pytest.mark.skipif(torch is None, reason="torch not installed")
@pytest.mark.parametrize(
    "dtype",
    ["float32", "float64"] if torch is None else [torch.float32, torch.float64],
    ids=["float32", "float64"],
)
@pytest.mark.parametrize("pot,kind,method,entry_id", list(_gate_params()))
def test_torch_dtype_follows_input(pot, kind, method, entry_id, dtype):
    # torch float32 inputs -> float32 output; float64 -> float64, for every
    # migrated potential's gate methods.
    args = _gate_args(kind, dtype)
    try:
        out = getattr(pot, method)(*args)
    except (TypeError, ValueError, RuntimeError) as exc:
        if entry_id in KNOWN_INFLIGHT:
            pytest.skip(
                f"{entry_id}.{method}: known in-flight backend failure "
                f"(fix on {KNOWN_INFLIGHT[entry_id]}): {exc}"
            )
        raise
    assert isinstance(out, torch.Tensor), (
        f"{entry_id}.{method}: torch input gave {type(out).__name__} output"
    )
    if out.dtype != dtype and entry_id in KNOWN_INFLIGHT:
        pytest.skip(
            f"{entry_id}.{method}: known in-flight dtype failure "
            f"({dtype} in, {out.dtype} out; fix on {KNOWN_INFLIGHT[entry_id]})"
        )
    assert out.dtype == dtype, (
        f"{entry_id}.{method}: {dtype} input gave {out.dtype} output"
    )


def _table_params():
    for entry_id, pot, kind, methods in _TABLE_ENTRIES:
        for method in methods:
            yield pytest.param(pot, method, entry_id, id=f"{entry_id}-{method}")


@pytest.mark.skipif(torch is None, reason="torch not installed")
@pytest.mark.parametrize("pot,method,entry_id", list(_table_params()))
def test_table_backed_f32_is_f64_quality(pot, method, entry_id):
    # float64-QUALITY pin: the torch-float32-input result must equal the
    # float64 result cast to float32 at float32 accuracy (rtol ~1e-6 with a
    # small headroom for cancellation-prone points). An anchor-style
    # implementation -- casting the tables (Ogata nodes/weights, expansion
    # coefficients, spline coefficients) to float32 and computing the interior
    # in float32 -- degrades the result by orders of magnitude more and fails.
    f = getattr(pot, method)
    for point in _POINTS:
        out32 = f(*[torch.tensor(v, dtype=torch.float32) for v in point])
        out64 = f(*[torch.tensor(v, dtype=torch.float64) for v in point])
        expected = out64.to(torch.float32)
        numpy.testing.assert_allclose(
            float(out32),
            float(expected),
            rtol=2e-6,
            atol=1e-9,
            err_msg=f"{entry_id}.{method} at {point}: float32 result is not "
            "float64-quality (was the interior degraded to float32?)",
        )


@pytest.mark.parametrize("pot,method,entry_id", list(_table_params()))
def test_numpy_exit_cast_semantics(pot, method, entry_id):
    # numpy semantics of the exit cast: float64 numpy inputs are a strict
    # no-op (and stay float64); float32 numpy arrays now come back float32
    # (the documented semantic change for these table-backed potentials);
    # python-scalar inputs are returned as before (no dtype to match).
    point = (1.1, 0.3, 0.4, 0.0)
    f = getattr(pot, method)
    out64 = f(*[numpy.float64(v) for v in point])
    assert numpy.asarray(out64).dtype == numpy.float64
    out32 = f(*[numpy.float32(v) for v in point])
    assert numpy.asarray(out32).dtype == numpy.float32, (
        f"{entry_id}.{method}: numpy float32 input gave "
        f"{numpy.asarray(out32).dtype} output"
    )
    numpy.testing.assert_allclose(float(out32), float(out64), rtol=2e-6, atol=1e-9)
    out_scalar = f(*point)
    numpy.testing.assert_array_equal(numpy.asarray(out_scalar), numpy.asarray(out64))


def test_match_input_dtype_helper_branches():
    # Direct unit checks of the helper's no-op guarantees:
    out = numpy.array([1.0, 2.0])
    # float64 coords: the SAME object comes back (bit-identical numpy path)
    assert match_input_dtype(out, numpy.array([0.5]), numpy.array(0.3)) is out
    # python-scalar coords carry no dtype: no-op
    assert match_input_dtype(out, 1.0, 2, None) is out
    # a plain Python float result (f64 by construction, e.g. the scalar _dens
    # path of MultipoleExpansion) is cast only when ALL coords carry a
    # narrower floating dtype...
    scast = match_input_dtype(2.5, numpy.array([0.5], dtype=numpy.float32))
    assert isinstance(scast, numpy.float32) and scast == numpy.float32(2.5)
    # ...and keeps the plain-float type bit-identically for f64, plain-scalar,
    # and mixed-dtype coords
    sout = 2.5
    assert match_input_dtype(sout, numpy.array([0.5])) is sout
    assert match_input_dtype(sout, 1.0, 0.3) is sout
    assert (
        match_input_dtype(
            sout,
            numpy.array([0.5], dtype=numpy.float32),
            numpy.array([0.5], dtype=numpy.float64),
        )
        is sout
    )
    iout = numpy.array([1, 2])
    assert match_input_dtype(iout, numpy.array([0.5], dtype=numpy.float32)) is iout
    # mixed float32/float64 coords resolve by result_type -> float64: no-op
    assert (
        match_input_dtype(
            out,
            numpy.array([0.5], dtype=numpy.float32),
            numpy.array([0.5], dtype=numpy.float64),
        )
        is out
    )
    # all-float32 coords: cast to float32
    cast = match_input_dtype(out, numpy.array([0.5], dtype=numpy.float32))
    assert cast.dtype == numpy.float32
    # integer coords carry no floating dtype: no-op
    assert match_input_dtype(out, numpy.array([1, 2])) is out


@pytest.mark.skipif(jax is None, reason="jax not installed")
def test_jax_grad_flows_through_exit_cast():
    # jax.grad at float32 must flow through the exit cast and reproduce
    # -_Rforce (float32 accuracy).
    Acos, Asin = _scf_coeffs()
    pot = SCFPotential(amp=1.3, Acos=Acos, Asin=Asin, a=1.2)
    R0, z0, phi0 = 1.1, 0.3, 0.4
    grad = jax.grad(
        lambda R: pot._evaluate(
            R, jnp.asarray(z0, dtype=jnp.float32), jnp.asarray(phi0, dtype=jnp.float32)
        )
    )(jnp.asarray(R0, dtype=jnp.float32))
    assert grad.dtype == jnp.float32
    ref = -float(pot._Rforce(R0, z0, phi0))
    numpy.testing.assert_allclose(float(grad), ref, rtol=1e-4)


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_special_fallback_f32_exit_cast():
    # The galpy.backend.special router's fallbacks (hyp2f1/hyp1f1/k0/k1/kn/
    # ellipk/ellipe/gamma) keep float64 quadrature tables and cast to the
    # input dtype at exit: float32 in -> float32 out at float64 quality.
    import galpy.backend.special as gsp

    cases = [
        ("hyp2f1", lambda d: gsp.hyp2f1(1.5, 0.5, 2.5, torch.tensor(-0.7, dtype=d))),
        ("hyp1f1", lambda d: gsp.hyp1f1(0.5, 1.5, torch.tensor(-0.7, dtype=d))),
        ("k0", lambda d: gsp.k0(torch.tensor(0.7, dtype=d))),
        ("k1", lambda d: gsp.k1(torch.tensor(0.7, dtype=d))),
        ("kn", lambda d: gsp.kn(2, torch.tensor(0.7, dtype=d))),
        ("ellipk", lambda d: gsp.ellipk(torch.tensor(0.3, dtype=d))),
        ("ellipe", lambda d: gsp.ellipe(torch.tensor(0.3, dtype=d))),
        ("gamma", lambda d: gsp.gamma(torch.tensor(1.5, dtype=d))),
    ]
    for name, call in cases:
        out32 = call(torch.float32)
        out64 = call(torch.float64)
        assert out32.dtype == torch.float32, f"{name}: float32 in gave {out32.dtype}"
        assert out64.dtype == torch.float64, f"{name}: float64 in gave {out64.dtype}"
        numpy.testing.assert_allclose(
            float(out32),
            float(out64.to(torch.float32)),
            rtol=1e-6,
            err_msg=f"{name}: float32 fallback result is not float64-quality",
        )
    # autodiff flows through the exit cast
    zt = torch.tensor(-0.7, dtype=torch.float32, requires_grad=True)
    gsp.hyp2f1(1.5, 0.5, 2.5, zt).backward()
    assert zt.grad is not None and torch.isfinite(zt.grad)


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_torch_grad_flows_through_exit_cast():
    # torch.autograd at float32 must flow through the exit cast and reproduce
    # -_Rforce (float32 accuracy).
    pot = interpSphericalPotential(
        rforce=HernquistPotential(amp=2.0, a=1.3),
        rgrid=numpy.geomspace(0.01, 1.5, 151),
    )
    R0, z0 = 1.1, 0.3
    Rt = torch.tensor(R0, dtype=torch.float32, requires_grad=True)
    pot._evaluate(Rt, torch.tensor(z0, dtype=torch.float32)).backward()
    assert Rt.grad.dtype == torch.float32
    ref = -float(pot._Rforce(numpy.array(R0), numpy.array(z0)))
    numpy.testing.assert_allclose(float(Rt.grad), ref, rtol=1e-4)
