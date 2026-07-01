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


###############################################################################
# Differentiability of *every* migrated method (not just _evaluate), wrt both R
# and phi, against central finite differences. This guards the safe-denominator
# rewrites of the where-branches and the phi-derivatives against NaN poisoning
# of reverse-mode gradients (both where-branches are evaluated under a trace).
###############################################################################
def _ad_grad(method, backend_name, ndim, var, R0, z0, phi0):
    """Reverse-mode d(method)/d(var) at a scalar point, var in {'R','phi'}."""
    if backend_name == "jax":

        def f(x):
            R = x if var == "R" else jnp.asarray(R0)
            phi = x if var == "phi" else jnp.asarray(phi0)
            if ndim == 3:
                return method(R, jnp.asarray(z0), phi, _T)
            return method(R, phi, _T)

        return float(jax.grad(f)(jnp.asarray(R0 if var == "R" else phi0)))
    # torch
    x = torch.tensor(
        R0 if var == "R" else phi0, dtype=torch.float64, requires_grad=True
    )
    R = x if var == "R" else torch.tensor(R0, dtype=torch.float64)
    phi = x if var == "phi" else torch.tensor(phi0, dtype=torch.float64)
    if ndim == 3:
        y = method(R, torch.tensor(z0, dtype=torch.float64), phi, _T)
    else:
        y = method(R, phi, _T)
    # Output may not depend on the differentiation variable (e.g. an axisymmetric
    # potential wrt phi): then there is no graph edge and the gradient is 0.
    if not (isinstance(y, torch.Tensor) and y.requires_grad):
        return 0.0
    y.backward()
    return 0.0 if x.grad is None else float(x.grad)


def _is_traceable_output(method, backend_name, ndim, R0, z0, phi0):
    """True unless the method returns a plain Python constant (no autodiff graph).

    Axisymmetric LogarithmicHalo phitorque / phi-derivatives return 0 / 0.0;
    differentiating those is a no-op (gradient is exactly 0), so we skip them.
    """
    if ndim == 3:
        out = method(numpy.asarray(R0), numpy.asarray(z0), numpy.asarray(phi0), _T)
    else:
        out = method(numpy.asarray(R0), numpy.asarray(phi0), _T)
    return hasattr(out, "ndim")


def _fd_grad(method, ndim, var, R0, z0, phi0, eps=1e-6):
    def at(R, phi):
        if ndim == 3:
            return float(method(numpy.asarray(R), numpy.asarray(z0), phi, _T))
        return float(method(numpy.asarray(R), phi, _T))

    if var == "R":
        return (at(R0 + eps, phi0) - at(R0 - eps, phi0)) / (2 * eps)
    return (at(R0, phi0 + eps) - at(R0, phi0 - eps)) / (2 * eps)


@pytest.mark.parametrize("entry", POTS, ids=POT_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
@pytest.mark.parametrize("var", ["R", "phi"])
def test_grad_all_methods_vs_finite_difference(backend_name, entry, var):
    pot, ndim, methods, scalar_only = entry
    R0, z0, phi0 = 1.3, 0.4, 0.7
    # differentiate every vectorized method (scalar-only md5-cached force methods
    # of SoftenedNeedleBar read per-instance state and are excluded by design).
    for mname in methods:
        method = getattr(pot, mname)
        # Some methods are identically zero (e.g. axisymmetric phitorque /
        # phi-derivatives) and return a plain Python constant with no graph to
        # differentiate; their FD is 0 and the analytic gradient is 0, so skip.
        if not _is_traceable_output(method, backend_name, ndim, R0, z0, phi0):
            continue
        fd = _fd_grad(method, ndim, var, R0, z0, phi0)
        ad = _ad_grad(method, backend_name, ndim, var, R0, z0, phi0)
        assert not numpy.isnan(ad), (
            f"{type(pot).__name__}.{mname} d/d{var} is NaN ({backend_name})"
        )
        # FD of a method that is identically zero (e.g. axisymmetric phitorque)
        # has no meaningful relative tolerance; compare on an absolute floor too.
        numpy.testing.assert_allclose(
            ad,
            fd,
            rtol=1e-4,
            atol=1e-7,
            err_msg=f"{type(pot).__name__}.{mname} d/d{var} ({backend_name})",
        )


###############################################################################
# Singular / break-point coverage of the rewritten where-branches: evaluate the
# potentials *at and around* the points where the safe-denominator substitution
# kicks in (CosmphiDisk's break radius R==rb and the inside R<rb branch;
# DehnenBar's r==rb seam and small-r inner branch), checking both numpy/jax/torch
# value parity and that reverse-mode gradients there are finite (not NaN).
###############################################################################
# (instance, ndim, points[(R,z,phi,on_seam)], methods). Points straddle the seam;
# ``on_seam`` marks the exact break radius where 2nd derivatives are discontinuous
# (there a central FD straddles two branches, so we check only finiteness there,
# not FD agreement). Off-seam points get the full grad-vs-FD check.
_SINGULAR_CASES = [
    (
        CosmphiDiskPotential(amp=1.0, phib=0.3, p=1.0, phio=0.02, m=4, r1=1.0, rb=1.2),
        2,
        [(0.5, None, 0.7, False), (1.2, None, 0.7, True), (1.9, None, 0.7, False)],
        PLANAR_METHODS,
    ),
    (
        # negative power-law index + a different break radius: exercises the
        # 1/Rsafe**p safe-denominator guard for p<0 in the inside-rb branch.
        CosmphiDiskPotential(amp=1.0, phib=0.3, p=-1.5, phio=0.02, m=2, r1=1.0, rb=0.9),
        2,
        [(0.4, None, 0.7, False), (0.9, None, 0.7, True), (1.5, None, 0.7, False)],
        PLANAR_METHODS,
    ),
    (
        DehnenBarPotential(
            amp=1.0, omegab=1.8, rb=0.6, Af=0.03, tform=-100.0, tsteady=1.0
        ),
        3,
        # r = sqrt(R^2+z^2): just inside, on, and outside rb=0.6
        [
            (0.3, 0.2, 0.7, False),
            (0.5, float(numpy.sqrt(0.6**2 - 0.5**2)), 0.7, True),
            (1.0, 0.4, 0.7, False),
        ],
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
    ),
]
_SINGULAR_IDS = ["CosmphiDisk_rb", "CosmphiDisk_rb_negp", "DehnenBar_rb"]


@pytest.mark.parametrize("case", _SINGULAR_CASES, ids=_SINGULAR_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_singular_branch_value_parity(backend_name, case):
    pot, ndim, points, methods = case
    for mname in methods:
        method = getattr(pot, mname)
        for R0, z0, phi0, _on_seam in points:
            ref = numpy.asarray(
                method(R0, z0, phi0, _T) if ndim == 3 else method(R0, phi0, _T)
            )
            got = _tonumpy(_call_scalar(method, backend_name, ndim, R0, z0, phi0))
            numpy.testing.assert_allclose(
                got,
                ref,
                rtol=1e-11,
                atol=1e-13,
                err_msg=f"{type(pot).__name__}.{mname} seam parity "
                f"@R={R0} ({backend_name})",
            )


@pytest.mark.parametrize("case", _SINGULAR_CASES, ids=_SINGULAR_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
@pytest.mark.parametrize("var", ["R", "phi"])
def test_singular_branch_grad_finite(backend_name, case, var):
    # The dead where-branch must not poison the gradient with NaNs at/near the
    # seam, and the live-branch gradient must match finite differences.
    pot, ndim, points, methods = case
    for mname in methods:
        method = getattr(pot, mname)
        for R0, z0, phi0, on_seam in points:
            if not _is_traceable_output(method, backend_name, ndim, R0, z0, phi0):
                continue
            ad = _ad_grad(method, backend_name, ndim, var, R0, z0, phi0)
            assert not numpy.isnan(ad), (
                f"{type(pot).__name__}.{mname} d/d{var} NaN @R={R0} ({backend_name})"
            )
            # At the exact break radius only the potential is C1 (its gradient,
            # ==-force, is continuous across the seam); the forces/2nd-derivatives
            # are not C1 there, so a central FD straddling the two branches is
            # meaningless and we require only the (live-branch) gradient to be
            # finite, checked above.
            if on_seam and mname != "_evaluate":
                continue
            fd = _fd_grad(method, ndim, var, R0, z0, phi0)
            numpy.testing.assert_allclose(
                ad,
                fd,
                rtol=1e-4,
                atol=1e-7,
                err_msg=f"{type(pot).__name__}.{mname} d/d{var} "
                f"@R={R0} ({backend_name})",
            )


# Explicitly time-dependent potentials, set up so the evaluation time falls in
# the smoothing GROWTH region (DehnenBar/Steady/Elliptical) or the active part
# of the Gaussian envelope (Transient) / a rotated phase (SoftenedNeedleBar).
# The other tests pin the bar fully grown (smooth==1) at t=0, so they never
# exercise the xp.where growth branch / xp.exp envelope, nor a traced t. These
# paths must (a) match numpy bit-for-bit under jax/torch and (b) be
# differentiable wrt t -- the case that breaks a bare numpy.exp / numpy-branch
# smoothing once t is a tracer (e.g. the in-backend diffrax/torchdiffeq
# integrator, or autodiff wrt time).
def _time_cases():
    cases = []
    # DehnenBar: pick t at the midpoint of (tform, tsteady) -> growth region.
    db = DehnenBarPotential(
        amp=1.0, omegab=1.0, rb=0.5, Af=0.03, tform=-2.0, tsteady=-1.0
    )
    cases.append((db, 3, 0.5 * (db._tform + db._tsteady)))
    sl = SteadyLogSpiralPotential(
        amp=1.0, omegas=1.0, A=-0.035, alpha=-7.0, m=2, tform=-2.0, tsteady=2.0
    )
    cases.append((sl, 2, 0.5 * (sl._tform + sl._tsteady)))
    ed = EllipticalDiskPotential(
        amp=1.0, phib=0.3, p=1.0, twophio=0.02, tform=-2.0, tsteady=2.0
    )
    cases.append((ed, 2, 0.5 * (ed._tform + ed._tsteady)))
    # Transient envelope is centered on to=0.5; evaluate slightly off-center so
    # d/dt of the Gaussian is nonzero.
    tl = TransientLogSpiralPotential(amp=1.0, omegas=1.0, to=0.5, sigma=1.0)
    cases.append((tl, 2, 0.8))
    # SoftenedNeedleBar: rotated (omegab*t != 0) so the de-rotation cos/sin matter.
    snb = SoftenedNeedleBarPotential(amp=1.0, omegab=1.0, pa=0.3)
    cases.append((snb, 3, 0.7))
    return cases


_TIME_CASES = _time_cases()
_TIME_IDS = [type(c[0]).__name__ for c in _TIME_CASES]


@pytest.mark.parametrize("case", _TIME_CASES, ids=_TIME_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_time_dependence_value_parity(backend_name, case):
    pot, ndim, tg = case
    R0, z0, phi0 = 1.3, 0.4, 0.7
    ref = float(
        pot._evaluate(R0, z0, phi0, tg) if ndim == 3 else pot._evaluate(R0, phi0, tg)
    )
    Rb = _asarray(backend_name, R0)
    phib = _asarray(backend_name, phi0)
    tgb = _asarray(backend_name, tg)
    if ndim == 3:
        got = _tonumpy(pot._evaluate(Rb, _asarray(backend_name, z0), phib, tgb))
    else:
        got = _tonumpy(pot._evaluate(Rb, phib, tgb))
    numpy.testing.assert_allclose(
        got,
        ref,
        rtol=1e-12,
        atol=1e-14,
        err_msg=f"{type(pot).__name__} time-dependent value parity ({backend_name})",
    )


@pytest.mark.parametrize("case", _TIME_CASES, ids=_TIME_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_grad_wrt_time_vs_finite_difference(backend_name, case):
    pot, ndim, tg = case
    R0, z0, phi0 = 1.3, 0.4, 0.7
    eps = 1e-6

    def phi_np(t):
        return float(
            pot._evaluate(R0, z0, phi0, t) if ndim == 3 else pot._evaluate(R0, phi0, t)
        )

    fd = (phi_np(tg + eps) - phi_np(tg - eps)) / (2 * eps)
    if backend_name == "jax":
        if ndim == 3:
            f = lambda t: pot._evaluate(
                jnp.asarray(R0), jnp.asarray(z0), jnp.asarray(phi0), t
            )
        else:
            f = lambda t: pot._evaluate(jnp.asarray(R0), jnp.asarray(phi0), t)
        ad = float(jax.grad(f)(jnp.asarray(tg)))
    else:
        t = torch.tensor(tg, dtype=torch.float64, requires_grad=True)
        if ndim == 3:
            y = pot._evaluate(
                torch.tensor(R0, dtype=torch.float64),
                torch.tensor(z0, dtype=torch.float64),
                torch.tensor(phi0, dtype=torch.float64),
                t,
            )
        else:
            y = pot._evaluate(
                torch.tensor(R0, dtype=torch.float64),
                torch.tensor(phi0, dtype=torch.float64),
                t,
            )
        y.backward()
        ad = float(t.grad)
    numpy.testing.assert_allclose(
        ad,
        fd,
        rtol=1e-5,
        atol=1e-12,
        err_msg=f"{type(pot).__name__} d/dt grad ({backend_name})",
    )


# CosmphiDisk/LopsidedDisk with a break radius rb: the potential is a piecewise
# power law (R**p outside rb, ~R**-p inside). The other tests only evaluate at
# R>rb (outside); the INSIDE regime (R<rb) and the R=0 axis exercise the other
# branch of the eager xp.where -- including a positive AND a negative break-power
# p, where the dead (selected-away) branch carries the R**(-|p|) singularity.
# This guards the safe-radius handling of both branches (R_in/R_out): values must
# match numpy bit-for-bit under jax/torch, R=0 must not raise, and AD must match
# -force on the inside.
_BREAK_CASES = [
    # (potential, has finite value at R=0?)  p>0: -inf at R=0 (genuine cusp);
    # p<0: finite at R=0.
    # (LopsidedDisk has no rb parameter -> no break radius -> no inside branch.)
    (
        CosmphiDiskPotential(amp=1.0, phib=0.3, p=1.0, phio=0.02, m=4, r1=1.0, rb=1.2),
        False,
    ),
    (
        CosmphiDiskPotential(amp=1.0, phib=0.3, p=-0.7, phio=0.02, m=3, r1=1.0, rb=1.2),
        True,
    ),
    (
        CosmphiDiskPotential(amp=1.0, phib=0.3, p=2.5, phio=0.02, m=2, r1=1.0, rb=0.9),
        False,
    ),
]
_BREAK_IDS = [f"{type(c[0]).__name__}_p{c[0]._p}" for c in _BREAK_CASES]


@pytest.mark.parametrize("case", _BREAK_CASES, ids=_BREAK_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_break_radius_inside_value_parity(backend_name, case):
    pot, _ = case
    phi0 = 0.4
    # R well inside rb, just inside, and at the seam; all use the inside branch.
    for R0 in [0.2, 0.7, pot._rb - 1e-6]:
        for mname in ["_evaluate", "_Rforce", "_phitorque", "_R2deriv", "_Rphideriv"]:
            ref = float(getattr(pot, mname)(R0, phi0))
            got = _tonumpy(
                getattr(pot, mname)(
                    _asarray(backend_name, R0), _asarray(backend_name, phi0)
                )
            )
            numpy.testing.assert_allclose(
                got,
                ref,
                rtol=1e-12,
                atol=1e-14,
                err_msg=f"{type(pot).__name__} inside-rb {mname} parity ({backend_name}) at R={R0}",
            )


@pytest.mark.parametrize("case", _BREAK_CASES, ids=_BREAK_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_break_radius_axis_no_crash(backend_name, case):
    # R=0 (the axis) is an inside element: the eager xp.where evaluates the dead
    # OUTSIDE branch R**p, which for p<0 is singular -- it must be safe-guarded so
    # this neither raises (scalar 0**-p) nor returns NaN. For p>0 the inside value
    # genuinely diverges (-inf); for p<0 it is finite, and must match numpy.
    pot, finite = case
    phi0 = 0.4
    ref = float(pot._evaluate(0.0, phi0))  # numpy scalar: must not raise
    got = _tonumpy(
        pot._evaluate(_asarray(backend_name, 0.0), _asarray(backend_name, phi0))
    )
    if finite:
        assert numpy.isfinite(ref), "p<0 inside value at R=0 should be finite"
        numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)
    else:
        assert numpy.isinf(ref) and not numpy.isnan(float(got)), (
            "p>0 inside value at R=0 is a genuine -inf singularity"
        )


@pytest.mark.parametrize("case", _BREAK_CASES, ids=_BREAK_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_break_radius_inside_grad_identity(backend_name, case):
    # AD(_evaluate)/dR == -_Rforce on the INSIDE branch (R<rb), and no NaN
    # poisoning from the dead outside branch at small (but nonzero) R.
    pot, _ = case
    phi0 = 0.4
    for R0 in [0.05, 0.3, 0.7]:
        ref = -float(pot._Rforce(R0, phi0))
        if backend_name == "jax":
            ad = float(
                jax.grad(lambda R: pot._evaluate(R, jnp.asarray(phi0)))(jnp.asarray(R0))
            )
        else:
            Rt = torch.tensor(R0, dtype=torch.float64, requires_grad=True)
            pot._evaluate(Rt, torch.tensor(phi0, dtype=torch.float64)).backward()
            ad = float(Rt.grad)
        assert not numpy.isnan(ad), f"NaN AD grad inside rb ({backend_name}) at R={R0}"
        numpy.testing.assert_allclose(
            ad,
            ref,
            rtol=1e-7,
            atol=1e-10,
            err_msg=f"{type(pot).__name__} inside-rb AD==-Rforce ({backend_name}) at R={R0}",
        )


###############################################################################
# SpiralArms dtype anchoring: the stored constant arrays (Cs, HNn) in
# _nvectors must follow the input dtype. Under torch, the stored float64
# numpy constants used to become strong float64 tensors that (a) promoted
# float32 inputs to float64 in every compute method and (b), for the default
# integer Cs=[1], routed `Cs * self._rho0` (int64 tensor x python float)
# through torch's default dtype (float32), costing ~4e-8 relative error in
# _dens even at float64.
###############################################################################
_SPIRAL_DTYPE_CASES = [
    (
        "default_Cs",
        SpiralArmsPotential(N=2, alpha=0.2, Rs=0.3, H=0.125, omega=0.4),
    ),
    (
        "float_Cs",
        SpiralArmsPotential(
            N=2,
            alpha=0.2,
            Rs=0.3,
            H=0.125,
            omega=0.4,
            Cs=[8.0 / (3.0 * numpy.pi), 0.5, 8.0 / (15.0 * numpy.pi)],
        ),
    ),
]
_SPIRAL_DTYPE_IDS = [c[0] for c in _SPIRAL_DTYPE_CASES]


@pytest.mark.skipif(torch is None, reason="torch not installed")
@pytest.mark.parametrize("case", _SPIRAL_DTYPE_CASES, ids=_SPIRAL_DTYPE_IDS)
@pytest.mark.parametrize("mname", ["_evaluate", "_Rforce", "_dens"])
def test_spiralarms_torch_float32_preserved(case, mname):
    # float32 inputs must produce float32 outputs: the stored constants must
    # not promote the computation to float64
    label, pot = case
    method = getattr(pot, mname)
    args = [torch.tensor(v, dtype=torch.float32) for v in (_RS, _ZS, _PHIS)]
    out = method(*args, _T)
    assert out.dtype == torch.float32, (
        f"SpiralArms ({label}) {mname}: float32 in gave {out.dtype} out"
    )


@pytest.mark.skipif(torch is None, reason="torch not installed")
@pytest.mark.parametrize("case", _SPIRAL_DTYPE_CASES, ids=_SPIRAL_DTYPE_IDS)
def test_spiralarms_torch_float64_dens_parity(case):
    # Regression pin for the 4e-8 relative error: with the default integer
    # Cs=[1], `Cs * self._rho0` used to round-trip through torch's default
    # dtype (float32) even for float64 inputs.
    label, pot = case
    ref = numpy.asarray(pot._dens(_RS, _ZS, _PHIS, _T))
    args = [torch.tensor(v, dtype=torch.float64) for v in (_RS, _ZS, _PHIS)]
    got = pot._dens(*args, _T).numpy()
    numpy.testing.assert_allclose(
        got,
        ref,
        rtol=1e-14,
        atol=0.0,
        err_msg=f"SpiralArms ({label}) _dens float64 torch/numpy parity",
    )
