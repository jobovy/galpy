###############################################################################
# test_backend_dehnenbar.py: backend tests for DehnenBarPotential.
#
# The rotating Dehnen bar has a time-smoothing factor _smooth(t) (xp.where over
# t < tform / t < tsteady) and a cos(phi - omegab t) rotation. Under a FORCED
# backend the pure-Python integrator hands the force numpy coordinates + a scalar
# Python-float t; get_namespace then resolves to the backend, so xp.sqrt(numpy)
# and array_api_compat torch.where(python-bool) raised. coerce_coords at each leaf
# (before _smooth) brings R,z,phi,t onto the backend so the whole force chain runs
# and differentiates, while the numpy path stays byte-identical.
###############################################################################
import numpy
import pytest

from galpy.backend import as_numpy, use
from galpy.potential import DehnenBarPotential, evaluateRforces, evaluatezforces

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

_DB = DehnenBarPotential()  # default rotating bar
# mid-growth t so _smooth's interior (growth) branch is exercised, not just 0/1
_TG = 0.5 * (_DB._tform + _DB._tsteady)
_METHODS = [
    "_evaluate",
    "_Rforce",
    "_zforce",
    "_phitorque",
    "_R2deriv",
    "_z2deriv",
    "_phi2deriv",
    "_Rzderiv",
    "_Rphideriv",
]
_R0, _Z0, _PHI0 = 1.1, 0.15, 0.35


def _asarray(backend_name, x):
    if backend_name == "numpy":
        return numpy.asarray(x, dtype=float)
    if backend_name == "jax":
        return jnp.asarray(x, dtype=jnp.float64)
    return torch.tensor(x, dtype=torch.float64)


@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_value_parity(backend_name, method):
    ref = float(
        getattr(_DB, method)(
            numpy.asarray(_R0), numpy.asarray(_Z0), numpy.asarray(_PHI0), _TG
        )
    )
    got = float(
        as_numpy(
            getattr(_DB, method)(
                _asarray(backend_name, _R0),
                _asarray(backend_name, _Z0),
                _asarray(backend_name, _PHI0),
                _asarray(backend_name, _TG),
            )
        )
    )
    rtol, atol = (0.0, 0.0) if backend_name == "numpy" else (1e-12, 1e-14)
    numpy.testing.assert_allclose(
        got, ref, rtol=rtol, atol=atol, err_msg=f"DehnenBar.{method} ({backend_name})"
    )


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_numpy_coords_scalar_t_under_forced_backend(backend_name):
    # The orbit-integrator scenario: the pure-Python integrator hands the force
    # numpy-scalar coordinates + a scalar Python-float t while a FORCED backend is
    # active. get_namespace resolves to the backend and xp.sqrt(numpy) / _smooth's
    # torch.where(python-bool) raised; coerce_coords fixes it. Checked at a t inside
    # the growth window (so _smooth's interior branch runs) against numpy.
    methods = ("_evaluate", "_Rforce", "_zforce", "_phitorque")
    ref = numpy.array([float(getattr(_DB, m)(_R0, _Z0, _PHI0, _TG)) for m in methods])
    with use(backend_name, force=True):
        got = numpy.array(
            [float(as_numpy(getattr(_DB, m)(_R0, _Z0, _PHI0, _TG))) for m in methods]
        )
    numpy.testing.assert_allclose(
        got,
        ref,
        rtol=1e-11,
        atol=1e-13,
        err_msg=f"DehnenBar forced scalar-t ({backend_name})",
    )


@pytest.mark.parametrize("force", ["R", "z"])
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_force_grad_vs_finite_difference(backend_name, force):
    # Stringent grad-vs-FD through the smoothing + rotation + softened-needle force.
    fn = {"R": evaluateRforces, "z": evaluatezforces}[force]
    R0 = 1.3

    def num_force(R):
        return float(
            fn(
                _DB,
                numpy.asarray(R),
                numpy.asarray(_Z0),
                phi=numpy.asarray(_PHI0),
                t=_TG,
            )
        )

    if backend_name == "jax":
        ad = float(
            jax.jacfwd(
                lambda R: fn(
                    _DB, R, jnp.asarray(_Z0), phi=jnp.asarray(_PHI0), t=jnp.asarray(_TG)
                )
            )(jnp.asarray(R0))
        )
    else:
        Rt = torch.tensor(R0, requires_grad=True)
        fn(
            _DB, Rt, torch.tensor(_Z0), phi=torch.tensor(_PHI0), t=torch.tensor(_TG)
        ).backward()
        ad = float(Rt.grad)
    prev = None
    for h in (1e-3, 1e-4, 1e-5):
        fd = (num_force(R0 + h) - num_force(R0 - h)) / (2 * h)
        rel = abs(ad - fd) / (abs(fd) + 1e-30)
        if prev is not None:
            assert rel <= prev + 1e-12
        prev = rel
    assert rel < 1e-6, (
        f"DehnenBar {force}-force grad-vs-FD rel={rel:.2e} ({backend_name})"
    )


# Shape parameters (gap audit R-SHAPE): beta's sqrt on the backend and the
# omegab = 0 period selected under a trace, so d/dbeta, d/drolr, d/domegab
# exist in data and forced mode, eagerly, under jit/vmap and on torch. t sits
# in the growth phase, where the period tb enters through tform/tsteady.
from backend_param_grad import MODES, assert_param_grad  # noqa: E402

_BAR_COORDS = [0.9, 0.2, 0.4, 0.3]


def _bar_chi(**kw):
    return DehnenBarPotential(amp=1.3, barphi=0.4, tform=-0.1, tsteady=2.0, **kw)


def _bar_omegab(**kw):
    return DehnenBarPotential(barphi=0.4, rb=0.8, Af=0.01, tform=-0.1, **kw)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "build,params,name,method",
    [
        (
            _bar_chi,
            {"chi": 0.8, "rolr": 0.9, "beta": 0.1, "alpha": 0.01},
            "beta",
            "Phi",
        ),
        (
            _bar_chi,
            {"chi": 0.8, "rolr": 0.9, "beta": 0.1, "alpha": 0.01},
            "beta",
            "Rforce",
        ),
        (
            _bar_chi,
            {"chi": 0.8, "rolr": 0.9, "beta": -0.2, "alpha": 0.01},
            "rolr",
            "phitorque",
        ),
        (_bar_omegab, {"omegab": 1.85}, "omegab", "Phi"),
        (_bar_omegab, {"omegab": 1.85}, "omegab", "zforce"),
    ],
    ids=["beta-Phi", "beta-Rforce", "rolr-phitorque", "omegab-Phi", "omegab-zforce"],
)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_dehnenbar_shape_parameter_gradient(
    backend_name, mode, build, params, name, method
):
    assert_param_grad(backend_name, mode, build, params, name, method, _BAR_COORDS)


def test_dehnenbar_traced_zero_pattern_speed_keeps_unit_period():
    if jax is None:  # pragma: no cover
        pytest.skip("jax not installed")
    tb = jax.jit(lambda om: _bar_omegab(omegab=om)._tb)
    assert float(tb(jnp.asarray(0.0))) == _bar_omegab(omegab=0.0)._tb == 1.0
    assert float(tb(jnp.asarray(1.85))) == _bar_omegab(omegab=1.85)._tb
