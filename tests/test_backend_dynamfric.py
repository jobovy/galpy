###############################################################################
# test_backend_dynamfric.py: multi-backend (jax/torch) coverage for the
# dynamical-friction forces whose OWN _Rforce/_zforce/_phitorque compute path
# was migrated to the galpy.backend namespace layer:
#
#   * ChandrasekharDynamicalFrictionForce -- classical Chandrasekhar friction.
#     The scipy sigma_r(r) spline is now a backend-agnostic Spline1D (numpy
#     hits scipy byte-identically, jax/torch evaluate the frozen ppoly), the
#     scipy.special.erf + numpy.exp/log are on galpy.backend.special / xp, and
#     the r<minr / rhm-vs-GM-over-v^2 python branches are xp.where.
#   * FDMDynamicalFrictionForce -- fuzzy-dark-matter friction, sharing the
#     Chandrasekhar internals. Its three kr-regimes (zero-velocity Cin/sici,
#     dispersion log, and the intermediate linear interp) plus the C<C_cdm
#     classical cutoff are nested xp.where / xp.minimum.
#
# Before this migration, evaluateRforces on a jax array COERCED back to numpy
# (hashlib.md5(numpy.array([...tracers...]))) and jax.grad/jit died with a
# TracerArrayConversionError. This module proves, per backend:
#   1. eager jax returns a jax array, eager torch a torch tensor,
#   2. the value matches the numpy path (which is byte-identical -- see
#      test_dynamfric / test_FDMdynamfric, unchanged),
#   3. jax.jit / jax.jacfwd over evaluateRforces (with v=) return finite (the
#      exact gap that defined this migration),
#   4. the force gradient w.r.t. R h-converges to a central finite difference
#      of the numpy path (grad-vs-FD, not a finite-and-nonzero check).
#
# The velocity-dependent forces require v=; a background density potential
# (NFW / default LogarithmicHalo) supplies rho, both already backend-native.
# Run under -W error::DeprecationWarning to catch the numpy-2.0 __array_wrap__
# coercion trap that a lingering scipy/numpy op on a backend array would raise.
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

from backend_jit_helpers import assert_jit_matches_eager

from galpy.backend import as_numpy
from galpy.potential import (
    ChandrasekharDynamicalFrictionForce,
    FDMDynamicalFrictionForce,
    NFWPotential,
    evaluatephitorques,
    evaluateRforces,
    evaluatezforces,
)

_NFW = NFWPotential(normalize=1.0, a=1.5)
_SIGMAR = lambda r: 1.0 / numpy.sqrt(2.0)  # noqa: E731 (analytic, pickle-safe stand-in)

_R0, _Z0, _PHI0, _T0 = 1.3, 0.4, 0.5, 0.0
_V0 = [0.15, 0.25, 0.08]  # cylindrical velocity (vR, vT, vz)


def _mhbar_per_m():
    # backend-independent conversion so an m can be chosen to land kr in a
    # specific FDM regime for _R0,_Z0,_V0.
    ref = FDMDynamicalFrictionForce(
        GMs=0.05, rhm=0.1, dens=_NFW, m=1e-99, sigmar=_SIGMAR
    )
    return ref._mhbar / 1e-99


def _m_for_kr(target):
    r = numpy.sqrt(_R0**2 + _Z0**2)
    vs = numpy.sqrt(sum(x**2 for x in _V0))
    return target / (vs * r) / _mhbar_per_m()


# (label, force). Covers: classical friction with a finite half-mass radius,
# the rhm=0 black-hole default (r/gamma/rhm dead branch = division by zero,
# guarded in Python on the static attr), the GMvs<rhm Coulomb-log branch, the
# constant-Coulomb-log / constant-FDM-factor shortcuts (const_lnLambda /
# const_FDMfactor -- the backend const branch that skips the r/v computation),
# and the three FDM kr-regimes (zero-velocity Cin, intermediate interp, dispersion).
_CASES = [
    ("CDF", ChandrasekharDynamicalFrictionForce(GMs=0.05, rhm=0.1, dens=_NFW)),
    (
        "CDF-blackhole",
        ChandrasekharDynamicalFrictionForce(GMs=0.05, rhm=0.0, dens=_NFW),
    ),
    (
        "CDF-rhmbranch",
        ChandrasekharDynamicalFrictionForce(GMs=0.001, rhm=2.0, dens=_NFW),
    ),
    (
        "CDF-constlnLambda",
        ChandrasekharDynamicalFrictionForce(
            GMs=0.05, rhm=0.1, dens=_NFW, const_lnLambda=3.0
        ),
    ),
    (
        "FDM-constfactor",
        FDMDynamicalFrictionForce(
            GMs=0.05,
            rhm=0.1,
            dens=_NFW,
            m=_m_for_kr(0.5),
            sigmar=_SIGMAR,
            const_FDMfactor=2.0,
        ),
    ),
    (
        "FDM-zero",
        FDMDynamicalFrictionForce(
            GMs=0.05, rhm=0.1, dens=_NFW, m=_m_for_kr(0.1), sigmar=_SIGMAR
        ),
    ),
    (
        "FDM-intermediate",
        FDMDynamicalFrictionForce(
            GMs=0.05, rhm=0.1, dens=_NFW, m=_m_for_kr(0.5), sigmar=_SIGMAR
        ),
    ),
    (
        "FDM-dispersion",
        FDMDynamicalFrictionForce(
            GMs=0.05, rhm=0.1, dens=_NFW, m=_m_for_kr(5.0), sigmar=_SIGMAR
        ),
    ),
]
_CASE_IDS = [c[0] for c in _CASES]


def _arr(backend, x):
    if backend == "jax":
        return jnp.asarray(x, dtype=jnp.float64)
    return torch.tensor(x, dtype=torch.float64)


def _module_of(x):
    return type(x).__module__


def _np_forces(obj, R):
    vn = numpy.array(_V0, dtype=float)
    return numpy.array(
        [
            float(
                evaluateRforces(obj, R, _Z0, phi=_PHI0, t=_T0, v=vn, use_physical=False)
            ),
            float(
                evaluatezforces(obj, R, _Z0, phi=_PHI0, t=_T0, v=vn, use_physical=False)
            ),
            float(
                evaluatephitorques(
                    obj, R, _Z0, phi=_PHI0, t=_T0, v=vn, use_physical=False
                )
            ),
        ]
    )


def _backend_forces(backend, obj, R):
    Rb, zb = _arr(backend, R), _arr(backend, _Z0)
    pb, tb = _arr(backend, _PHI0), _arr(backend, _T0)
    vb = _arr(backend, _V0)
    fr = evaluateRforces(obj, Rb, zb, phi=pb, t=tb, v=vb, use_physical=False)
    fz = evaluatezforces(obj, Rb, zb, phi=pb, t=tb, v=vb, use_physical=False)
    fp = evaluatephitorques(obj, Rb, zb, phi=pb, t=tb, v=vb, use_physical=False)
    return fr, fz, fp


@pytest.mark.filterwarnings("error::DeprecationWarning")
@pytest.mark.filterwarnings("error::FutureWarning")
@pytest.mark.parametrize("label,obj", _CASES, ids=_CASE_IDS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_dynamfric_value_and_backend_array(backend, label, obj):
    # eager backend eval must (a) return a native backend array (a lingering
    # scipy/numpy op would silently DETACH to numpy) and (b) match the numpy
    # value the class produces (numpy path is byte-identical -- pinned in
    # test_dynamfric/test_FDMdynamfric).
    ref = _np_forces(obj, _R0)
    fr, fz, fp = _backend_forces(backend, obj, _R0)
    for f in (fr, fz, fp):
        assert backend in _module_of(f), (
            f"{label}: force left the {backend} namespace ({_module_of(f)})"
        )
    got = numpy.array([float(as_numpy(fr)), float(as_numpy(fz)), float(as_numpy(fp))])
    numpy.testing.assert_allclose(got, ref, rtol=1e-11, atol=1e-13, err_msg=label)


@pytest.mark.filterwarnings("error::DeprecationWarning")
@pytest.mark.filterwarnings("error::FutureWarning")
@pytest.mark.parametrize("label,obj", _CASES, ids=_CASE_IDS)
def test_dynamfric_jax_jit_jacfwd_finite(label, obj):
    # The exact gap: jax.jit / jax.jacfwd over evaluateRforces (velocity-dep,
    # so v= is passed) must trace to a finite result. Pre-migration this died
    # with a TracerArrayConversionError from hashlib.md5(numpy.array([tracers])).
    if jax is None:  # pragma: no cover
        pytest.skip("jax not installed")

    def fR(R):
        return evaluateRforces(
            obj,
            R,
            jnp.asarray(_Z0),
            phi=jnp.asarray(_PHI0),
            t=jnp.asarray(_T0),
            v=jnp.asarray(_V0),
            use_physical=False,
        )

    # jit value must equal the eager numpy Rforce, supplied via ref=; the helper
    # additionally rejects a trace that folded R0 away into a constant.
    assert_jit_matches_eager(
        fR,
        jnp.asarray(_R0),
        rtol=1e-10,
        atol=1e-13,
        ref=_np_forces(obj, _R0)[0],
        err_msg=label,
    )
    jac = float(jax.jacfwd(fR)(jnp.asarray(_R0)))
    assert numpy.isfinite(jac), f"{label}: jacfwd not finite"


@pytest.mark.filterwarnings("error::DeprecationWarning")
@pytest.mark.filterwarnings("error::FutureWarning")
@pytest.mark.parametrize("label,obj", _CASES, ids=_CASE_IDS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_dynamfric_grad_vs_finite_difference(backend, label, obj):
    # d(Rforce)/dR from AD must h-CONVERGE to a central finite difference of the
    # numpy path (checked at two h; the tighter h must match to ~1e-5), not just
    # be finite-and-nonzero.
    if backend == "jax":
        ad = float(
            jax.grad(
                lambda R: evaluateRforces(
                    obj,
                    R,
                    jnp.asarray(_Z0),
                    phi=jnp.asarray(_PHI0),
                    t=jnp.asarray(_T0),
                    v=jnp.asarray(_V0),
                    use_physical=False,
                )
            )(jnp.asarray(_R0))
        )
    else:
        R = torch.tensor(_R0, dtype=torch.float64, requires_grad=True)
        out = evaluateRforces(
            obj,
            R,
            torch.tensor(_Z0),
            phi=torch.tensor(_PHI0),
            t=torch.tensor(_T0),
            v=torch.tensor(_V0),
            use_physical=False,
        )
        (g,) = torch.autograd.grad(out, R)
        ad = float(g)
    assert numpy.isfinite(ad), f"{label}: grad not finite"

    def fd(h):
        return (_np_forces(obj, _R0 + h)[0] - _np_forces(obj, _R0 - h)[0]) / (2 * h)

    fd_coarse, fd_fine = fd(1e-4), fd(1e-6)
    # central FD converges O(h^2): the finer step must be at least as close.
    assert abs(ad - fd_fine) <= abs(ad - fd_coarse) + 1e-9, (
        f"{label}: grad does not h-converge (ad={ad}, fd(1e-4)={fd_coarse}, "
        f"fd(1e-6)={fd_fine})"
    )
    numpy.testing.assert_allclose(
        ad, fd_fine, rtol=1e-5, atol=1e-9, err_msg=f"{backend} {label}"
    )
