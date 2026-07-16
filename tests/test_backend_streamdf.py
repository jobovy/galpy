###############################################################################
# test_backend_streamdf.py: multi-backend tests for streamdf's DF-evaluation
# methods (Phase A.1: the closed-form distributions + frequency moments that
# operate on an ASSEMBLED track).
#
# The stream track itself is assembled with numpy (Phase B); these methods
# evaluate the stream DF at given parallel-angle / frequency offsets using the
# precomputed track scalars (self._meandO, self._sortedSigOEig, ...). Migrated to
# the galpy.backend namespace layer: a jax/torch input routes to native
# erf/exp/sqrt (so d(DF)/d(offset) flows and jits), the numpy path keeps
# scipy.special (byte-identical). pOparapar / ptdAngle's numpy in-place masked
# write becomes xp.where (jit/grad-safe; the t=0 dO->inf dead branch is guarded).
#
# Proves per method: (a) backend value parity vs the numpy path (which is
# byte-identical -- the else-branch is the verbatim original), and (b) grad-vs-FD
# h-converges (stringent, not finite-and-nonzero). Backends not installed self-skip.
###############################################################################
import numpy
import pytest

pytestmark = pytest.mark.backend_managed

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

    torch.set_default_dtype(torch.float64)
    BACKENDS.append("torch")
except ImportError:  # pragma: no cover
    torch = None

AD_BACKENDS = [b for b in BACKENDS if b != "numpy"]

from galpy.actionAngle import actionAngleIsochroneApprox
from galpy.orbit import Orbit
from galpy.potential import LogarithmicHaloPotential
from galpy.util import conversion


@pytest.fixture(scope="module")
def sdf():
    # The canonical Bovy (2014) GD-1-like stream (as in test_streamdf.py). Assembling
    # the track is slow, so build once per module.
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    aAI = actionAngleIsochroneApprox(pot=lp, b=0.8)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    return streamdf_ctor(lp, aAI, obs)


def streamdf_ctor(lp, aAI, obs):
    from galpy.df import streamdf

    return streamdf(
        0.365 / 220.0,
        progenitor=obs,
        pot=lp,
        aA=aAI,
        leading=True,
        nTrackChunks=11,
        tdisrupt=4.5 / conversion.time_in_Gyr(220.0, 8.0),
    )


def _arr(backend_name, x):
    if backend_name == "jax":
        return jnp.asarray(x, dtype=jnp.float64)
    return torch.tensor(x, dtype=torch.float64)


# scalar-valued (or first-element) DF evaluations at a parallel angle, and a
# stripping-time p(t|a) that is array-valued.
_DANGLE = 0.5
_METHODS = [
    ("density_par", lambda s, d: s._density_par(d), False),
    ("meanOmega1D", lambda s, d: s.meanOmega(d, oned=True, use_physical=False), False),
    ("sigOmega", lambda s, d: s.sigOmega(d, use_physical=False), False),
]


@pytest.mark.parametrize("name,fn,_arrarg", _METHODS, ids=[m[0] for m in _METHODS])
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_df_eval_value_parity(sdf, backend_name, name, fn, _arrarg):
    ref = float(fn(sdf, _DANGLE))
    got = float(fn(sdf, _arr(backend_name, _DANGLE)))
    numpy.testing.assert_allclose(
        got, ref, rtol=1e-11, atol=1e-13, err_msg=f"{name} {backend_name}"
    )


@pytest.mark.parametrize("name,fn,_arrarg", _METHODS, ids=[m[0] for m in _METHODS])
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_df_eval_grad_vs_fd(sdf, backend_name, name, fn, _arrarg):
    # d(DF)/d(dangle) AD must h-converge to a central FD of the numpy path.
    if backend_name == "jax":
        ad = float(jax.grad(lambda d: fn(sdf, d))(jnp.asarray(_DANGLE)))
    else:
        dt = torch.tensor(_DANGLE, dtype=torch.float64, requires_grad=True)
        fn(sdf, dt).backward()
        ad = float(dt.grad)
    assert numpy.isfinite(ad) and abs(ad) > 0
    best = min(
        abs(ad - (float(fn(sdf, _DANGLE + h)) - float(fn(sdf, _DANGLE - h))) / (2 * h))
        for h in (1e-4, 1e-5, 1e-6)
    )
    assert best < 1e-5 * abs(ad) + 1e-7, (
        f"{name} {backend_name} grad-vs-FD best={best:.2e}"
    )


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_pOparapar_where_and_grad(sdf, backend_name):
    # pOparapar's masked Gaussian: backend value parity + d/d(Opar) h-converges.
    Op0, ap0 = float(sdf._meandO) * 1.05, 0.3
    ref = float(numpy.atleast_1d(sdf.pOparapar(Op0, ap0))[0])
    got = float(
        numpy.atleast_1d(
            sdf.pOparapar(_arr(backend_name, Op0), _arr(backend_name, ap0))
        )[0]
    )
    numpy.testing.assert_allclose(got, ref, rtol=1e-11, atol=1e-13)
    if backend_name == "jax":
        ad = float(
            jax.grad(lambda O: sdf.pOparapar(O, jnp.asarray(ap0)).sum())(
                jnp.asarray(Op0)
            )
        )
    else:
        opar_t = torch.tensor(Op0, dtype=torch.float64, requires_grad=True)
        sdf.pOparapar(opar_t, torch.tensor(ap0, dtype=torch.float64)).sum().backward()
        ad = float(opar_t.grad)
    h = 1e-5
    fd = (
        float(sdf.pOparapar(Op0 + h, ap0).sum())
        - float(sdf.pOparapar(Op0 - h, ap0).sum())
    ) / (2 * h)
    assert abs(ad - fd) < 1e-4 * abs(fd) + 1e-6, (
        f"pOparapar grad {backend_name}: {ad} vs {fd}"
    )


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_ptdAngle_where_and_grad(sdf, backend_name):
    # ptdAngle's masked p(t|a): backend value parity (incl. the t<=0 / t>=tdisrupt zero
    # region) + d/d(t) h-converges, with the dO=dangle/t dead branch guarded.
    ts = numpy.array([0.5, 1.5, 2.5]) * sdf._tdisrupt / 3.0
    ref = numpy.asarray(sdf.ptdAngle(ts, _DANGLE))
    got = numpy.asarray(
        sdf.ptdAngle(_arr(backend_name, ts), _arr(backend_name, _DANGLE))
    )
    numpy.testing.assert_allclose(got, ref, rtol=1e-11, atol=1e-13)
    t0 = float(ts[1])
    if backend_name == "jax":
        ad = float(
            jax.grad(lambda t: sdf.ptdAngle(t, jnp.asarray(_DANGLE)))(jnp.asarray(t0))
        )
    else:
        tt = torch.tensor(t0, dtype=torch.float64, requires_grad=True)
        sdf.ptdAngle(tt, torch.tensor(_DANGLE, dtype=torch.float64)).backward()
        ad = float(tt.grad)
    h = 1e-5
    fd = (
        float(sdf.ptdAngle(t0 + h, _DANGLE)) - float(sdf.ptdAngle(t0 - h, _DANGLE))
    ) / (2 * h)
    assert abs(ad - fd) < 1e-4 * abs(fd) + 1e-6, (
        f"ptdAngle grad {backend_name}: {ad} vs {fd}"
    )
