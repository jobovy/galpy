###############################################################################
# test_backend_qdf.py: Track F Pdf.4 PR-4a -- backend (jax/torch) coverage for
# the quasiisothermaldf __call__ + _rg core. The numpy path is byte-identical
# (test_qdf unchanged); this exercises the resolved-namespace dispatch:
#   (a) value parity numpy<->jax<->torch of __call__ (log + non-log), including
#       the lz<0 / NaN sentinel branches, and
#   (b) grad-vs-FD of __call__ w.r.t. (R,vR,vT,z,vz), flowing the #131 Staeckel
#       c=True action gradients through the pure-arithmetic DF core.
# The Staeckel action gradient is first-order-accurate (#1050), so grad-vs-FD is
# checked at rtol 8e-3 (>2x margin over the observed ~3.4e-3 floor).
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

from galpy.actionAngle import actionAngleStaeckel
from galpy.backend import as_numpy, is_backend_array
from galpy.df import quasiisothermaldf
from galpy.potential import MWPotential

# Staeckel-based qdf fixture, mirroring test_qdf.py's Staeckel qdf setup.
_aAS = actionAngleStaeckel(pot=MWPotential, c=True, delta=0.5)
_qdf = quasiisothermaldf(
    1.0 / 4.0, 0.2, 0.1, 1.0, 1.0, pot=MWPotential, aA=_aAS, cutcounter=True
)

# Interior prograde orbits (bound, lz>0); includes vR=0/vz=0/z=0 edge orbits.
_ORBITS = numpy.array(
    [
        [0.9, 0.1, 0.9, 0.05, 0.02],
        [1.1, -0.05, 0.8, 0.1, -0.03],
        [0.7, 0.2, 1.0, -0.08, 0.05],
        [1.0, 0.0, 0.95, 0.0, 0.0],
    ]
)


def _arr(backend, x):
    return jnp.asarray(x) if backend == "jax" else torch.tensor(numpy.asarray(x, float))


def _cols(backend, orbits):
    return [_arr(backend, orbits[:, i]) for i in range(5)]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("log", [False, True])
def test_call_value_parity(backend, log):
    # __call__ (R,vR,vT,z,vz) value byte-identity numpy<->backend, both modes.
    cn = [_ORBITS[:, i] for i in range(5)]
    ref = as_numpy(_qdf(*cn, log=log))
    got = _qdf(*_cols(backend, _ORBITS), log=log)
    assert is_backend_array(got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-12, atol=1e-300)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("log", [False, True])
def test_call_sentinel_lzneg(backend, log):
    # cutcounter lz<0 sentinel branch: retrograde orbit -> -finfo.max (log) / 0.
    retro = numpy.array([[0.9, 0.1, -1.5, 0.05, 0.02]])
    cn = [retro[:, i] for i in range(5)]
    ref = as_numpy(_qdf(*cn, log=log))
    got = as_numpy(_qdf(*_cols(backend, retro), log=log))
    # exact match of the sentinel/zero (dtype-generic finfo max == float64 max)
    assert numpy.array_equal(got, ref)
    if log:
        assert ref[0] == -numpy.finfo(numpy.float64).max
    else:
        assert ref[0] == 0.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_call_sentinel_nan(backend):
    # NaN sentinel branch (log mode): a func returning a negative value makes
    # log(func) NaN -> masked to -finfo.max on both numpy and backend paths.
    negfunc = lambda jr, lz, jz: jr - 100.0
    cn = [_ORBITS[:, i] for i in range(5)]
    ref = as_numpy(_qdf(*cn, log=True, func=negfunc))
    got = as_numpy(_qdf(*_cols(backend, _ORBITS), log=True, func=negfunc))
    assert numpy.all(ref == -numpy.finfo(numpy.float64).max)  # NaN tripped
    assert numpy.array_equal(got, ref)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("log", [False, True])
def test_call_func_parity(backend, log):
    # func-path value parity (moment weight lz**2 * jr) through xp.log/funcFactor.
    f = lambda jr, lz, jz: lz**2 * jr + 1e-3
    cn = [_ORBITS[:, i] for i in range(5)]
    ref = as_numpy(_qdf(*cn, log=log, func=f))
    got = _qdf(*_cols(backend, _ORBITS), log=log, func=f)
    assert is_backend_array(got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-12, atol=1e-300)


@pytest.mark.parametrize("backend", BACKENDS)
def test_rg_backend_parity(backend):
    # _rg backend Spline1D eval matches the numpy scipy spline to ~1e-12 in the
    # physical (in-precompute-range) regime; output is a backend array.
    lzn = numpy.array([0.3, 0.5, 0.81, 1.0, 1.4])
    ref = as_numpy(_qdf._rg(lzn))
    got = _qdf._rg(_arr(backend, lzn))
    assert is_backend_array(got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("log", [False, True])
def test_call_grad_vs_fd(backend, log):
    # jax.grad / torch.autograd of __call__ w.r.t. (R,vR,vT,z,vz) vs central FD.
    # Exercises the #131 Staeckel c=True action grads through jr/lz/jz; the
    # action grad is first-order-accurate (#1050) -> rtol 8e-3.
    def npval(p):
        a = [numpy.atleast_1d(x) for x in p]
        return float(as_numpy(_qdf(*a, log=log)).reshape(-1)[0])

    def fd(p, h=1e-6):
        g = numpy.zeros(5)
        for i in range(5):
            pp = p.copy()
            pp[i] += h
            pm = p.copy()
            pm[i] -= h
            g[i] = (npval(pp) - npval(pm)) / (2.0 * h)
        return g

    for ic in _ORBITS:
        ic = numpy.asarray(ic, float)
        gfd = fd(ic)
        if backend == "jax":

            def f(v):
                a = [v[i].reshape(1) for i in range(5)]
                return _qdf(*a, log=log).reshape(())

            g = as_numpy(jax.grad(f)(jnp.asarray(ic)))
        else:
            vt = torch.tensor(ic, requires_grad=True)
            a = [vt[i].reshape(1) for i in range(5)]
            _qdf(*a, log=log).reshape(()).backward()
            g = vt.grad.numpy()
        numpy.testing.assert_allclose(g, gfd, rtol=8e-3, atol=1e-3)
