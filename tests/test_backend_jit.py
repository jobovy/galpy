###############################################################################
# test_backend_jit.py: the boundary-jit dimension.
#
# galpy is meant to be jit/compile-COMPATIBLE (never self-jitting, see the
# no-internal-jit rule). This module asserts that by tracing PUBLIC entry points
# whole -- jax.jit / torch.compile over `pot.Rforce(R, z)` etc. -- rather than
# jitting an internal seam. Everything the boundary decorators do (unit parsing,
# coordinate coercion) traces away, so what is measured is the real user-facing
# contract: "can I jit a galpy call?".
#
# Coverage is the whole zoo: every concrete Potential subclass that constructs
# with no required arguments. Anything that cannot be traced must be listed in
# _NOT_TRACEABLE with its error and reason, so the gaps are auditable and the
# list is a burndown list rather than silent breakage.
###############################################################################
import os

import numpy
import pytest

pytestmark = pytest.mark.backend_managed

# torch.compile writes its generated kernels to one inductor cache directory.
# Concurrent xdist workers collide there and it surfaces as an InductorError
# wrapping an ImportError on a half-written .so. Must be set before torch loads.
_worker = os.environ.get("PYTEST_XDIST_WORKER")
if _worker:  # pragma: no cover - only set under xdist
    os.environ.setdefault(
        "TORCHINDUCTOR_CACHE_DIR", f"/tmp/torchinductor_galpy_{_worker}"
    )

try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    jax = None
try:
    import torch

    torch.set_default_dtype(torch.float64)
except ImportError:  # pragma: no cover
    torch = None

if torch is not None:
    # Every case compiles the SAME lambda code object with a different closure,
    # so dynamo counts them as recompiles of one frame. Past its cache-size limit
    # (default 8) it stops tracing that frame and silently falls back to eager --
    # and an untraceable potential then "passes". That made a serial run green
    # and an xdist run red for identical code, and it hid real gaps. Raise the
    # limits past the case count so every case is really traced.
    torch._dynamo.config.cache_size_limit = 4096
    torch._dynamo.config.accumulated_cache_size_limit = 8192

if torch is not None:  # pragma: no cover - depends on the interpreter version
    # torch.compile trails the Python release (dynamo refuses to run on a Python
    # it does not support yet), so probe the capability rather than assuming that
    # `import torch` implies it. CI runs this shard on 3.10 through 3.14.
    try:
        torch.compile(lambda x: x + 1.0, fullgraph=True)(torch.tensor(1.0))
        _TORCH_COMPILES = True
    except Exception:
        _TORCH_COMPILES = False
else:  # pragma: no cover
    _TORCH_COMPILES = False

import galpy.potential as gp
from galpy.potential import Potential

_R0, _Z0 = 1.1, 0.2

# Public entry points traced whole. `dens` is included because it exercises a
# different code path (Poisson/second derivatives) than the forces.
_ENTRY = {
    "__call__": lambda p, R, z: p(R, z),
    "Rforce": lambda p, R, z: p.Rforce(R, z),
    "zforce": lambda p, R, z: p.zforce(R, z),
    "dens": lambda p, R, z: p.dens(R, z),
}

# (potential, entry, backend) that cannot be traced yet, with the failure mode.
# Shrink this list; do not grow it without a stated reason.
_NOT_TRACEABLE = {
    # Data-dependent Python branch on a traced value (TracerBoolConversionError).
    ("RazorThinExponentialDiskPotential", "__call__", "jax"),
    ("RazorThinExponentialDiskPotential", "Rforce", "jax"),
    ("RazorThinExponentialDiskPotential", "zforce", "jax"),
    # numpy conversion of a traced array (TracerArrayConversionError).
    ("TwoPowerTriaxialPotential", "__call__", "jax"),
    # The lazy backend table build (_backend_static_data -> scipy PPoly /
    # scipy.special.binom) runs inside the FIRST traced call and dynamo cannot
    # trace scipy. Calling the potential once eagerly first is a user-side
    # workaround: it then compiles and matches eager exactly.
    ("DiskMultipoleExpansionPotential", "__call__", "torch"),
    ("DiskMultipoleExpansionPotential", "Rforce", "torch"),
    ("DiskMultipoleExpansionPotential", "zforce", "torch"),
    ("DiskMultipoleExpansionPotential", "dens", "torch"),
    # Writes into a numpy scalar (`TypeError: 'numpy.float64' object does not
    # support item assignment`): the coefficient path indexes an array that is
    # 0-d once the coordinates are tensors.
    ("MultipoleExpansionPotential", "__call__", "torch"),
    ("MultipoleExpansionPotential", "Rforce", "torch"),
    ("MultipoleExpansionPotential", "zforce", "torch"),
    ("MultipoleExpansionPotential", "dens", "torch"),
    # Compiles, but returns inf where eager returns -0.1979: the compiled graph
    # evaluates the DEAD side of an xp.where (the alpha/beta special-case
    # branch), which is singular at these parameters. Same hazard as the
    # AD-NaN-poisoning one, surfacing through inductor instead of a gradient.
    ("TwoPowerSphericalPotential", "__call__", "torch"),
    # torch dynamo cannot trace this yet (InternalTorchDynamoError); it takes a
    # user-supplied Python callable for the surface density.
    ("AnyAxisymmetricRazorThinDiskPotential", "__call__", "torch"),
    ("AnyAxisymmetricRazorThinDiskPotential", "Rforce", "torch"),
    ("AnyAxisymmetricRazorThinDiskPotential", "zforce", "torch"),
    ("AnyAxisymmetricRazorThinDiskPotential", "dens", "torch"),
}


def _default_constructible():
    """Every concrete Potential subclass that builds with no required args."""
    out = {}
    for name in sorted(dir(gp)):
        obj = getattr(gp, name)
        if (
            not isinstance(obj, type)
            or not issubclass(obj, Potential)
            or obj is Potential
            or name.startswith("_")
        ):
            continue
        try:
            out[name] = obj()
        except Exception:  # pragma: no cover - needs required args
            continue
    return out


_POTS = _default_constructible()
_CASES = [(n, e) for n in _POTS for e in _ENTRY]


# Absolute floor for the comparison. Some entry points are mathematically ZERO
# and are computed by cancellation -- DehnenBarPotential.dens (a pure-potential
# perturbation) lands on ~7e-20, KuzminDiskPotential.dens (razor-thin, so no
# volume density off the plane) on ~7e-18 -- and XLA/dynamo reassociate those
# differently than eager does, which is a 30-70% *relative* swing on a number
# that is zero. Everything physical agrees to machine precision (forces to
# ~1e-16 relative), so rtol governs real values and atol absorbs the zeros.
_ATOL = 1e-12


def _reference(pot, entry):
    """Eager numpy value, or None when the entry point does not apply."""
    try:
        ref = float(_ENTRY[entry](pot, _R0, _Z0))
    except Exception:
        return None
    return ref if numpy.isfinite(ref) else None


@pytest.mark.skipif("jax is None")
@pytest.mark.parametrize("name,entry", _CASES)
def test_jax_jit_traces_public_entry_point(name, entry):
    pot = _POTS[name]
    ref = _reference(pot, entry)
    if ref is None:
        pytest.skip(f"{name}.{entry} not applicable")
    fn = _ENTRY[entry]
    expected_fail = (name, entry, "jax") in _NOT_TRACEABLE
    try:
        got = float(
            jax.jit(lambda R, z: fn(pot, R, z))(jnp.asarray(_R0), jnp.asarray(_Z0))
        )
    except Exception as exc:
        if expected_fail:
            pytest.xfail(f"{name}.{entry} not jax-traceable: {type(exc).__name__}")
        raise
    assert not expected_fail, (
        f"{name}.{entry} now traces under jax.jit -- remove it from _NOT_TRACEABLE"
    )
    numpy.testing.assert_allclose(got, ref, rtol=1e-6, atol=_ATOL)


@pytest.mark.skipif("not _TORCH_COMPILES")
@pytest.mark.parametrize("name,entry", _CASES)
def test_torch_compile_traces_public_entry_point(name, entry):
    pot = _POTS[name]
    ref = _reference(pot, entry)
    if ref is None:
        pytest.skip(f"{name}.{entry} not applicable")
    fn = _ENTRY[entry]
    expected_fail = (name, entry, "torch") in _NOT_TRACEABLE
    try:
        compiled = torch.compile(
            lambda R, z: fn(pot, R, z), fullgraph=False, dynamic=False
        )
        got = float(compiled(torch.tensor(_R0), torch.tensor(_Z0)))
    except Exception as exc:
        if expected_fail:
            pytest.xfail(f"{name}.{entry} not torch-compilable: {type(exc).__name__}")
        raise
    assert not expected_fail, (
        f"{name}.{entry} now compiles under torch -- remove it from _NOT_TRACEABLE"
    )
    numpy.testing.assert_allclose(got, ref, rtol=1e-6, atol=_ATOL)


def test_zoo_is_actually_covered():
    # Guard against the discovery silently finding nothing (a rename would make
    # both parametrised tests vacuous).
    assert len(_POTS) > 30, f"only {len(_POTS)} potentials discovered"
