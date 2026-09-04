###############################################################################
# test_backend_potential_jit.py: the boundary-jit dimension, for POTENTIALS.
#
# galpy is meant to be jit/compile-COMPATIBLE (never self-jitting, see the
# no-internal-jit rule). This module asserts that by tracing PUBLIC entry points
# whole -- jax.jit / torch.compile over `pot.Rforce(R, z)` etc. -- rather than
# jitting an internal seam. Everything the boundary decorators do (unit parsing,
# coordinate coercion) traces away, so what is measured is the real user-facing
# contract: "can I jit a galpy call?".
#
# Coverage is the whole potential zoo: every concrete Potential subclass that
# constructs with no required arguments. Anything that cannot be traced must be
# listed in _NOT_TRACEABLE with its error and reason, so the gaps are auditable
# and the list is a burndown list rather than silent breakage.
#
# POTENTIALS ONLY, hence the name: this is the fast always-on check for the one
# object type. Making the REST of galpy jittable -- orbits, action-angle, DFs,
# streams -- is not this file's job and must not be bolted onto it; that is the
# suite-wide trace mode (`pytest <any file> --backend jax --jit`), which runs
# the existing tests traced and keeps its own "<backend>-jit" burndown lists.
# Expect siblings here only if some other object type needs a comparable
# fast always-on check of its own (test_backend_orbit_jit.py, ...).
###############################################################################
import numpy
import pytest

pytestmark = pytest.mark.backend_managed

# NB: the private TORCHINDUCTOR_CACHE_DIR that makes every compile below a COLD
# one is set in conftest.py, which pytest imports before this module.

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

from conftest import torch_compiles

_TORCH_COMPILES = torch_compiles()

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
# EMPTY: every default-constructible potential in the zoo now traces under both
# jax.jit and torch.compile, on all four entry points. Keep it that way -- a new
# entry here is a regression, not a TODO, unless it comes with a stated reason.
_NOT_TRACEABLE = set()


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


def _fresh(name):
    """A newly constructed instance, with no lazily-built state carried over."""
    return type(_POTS[name])()


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
    pot = _fresh(name)
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
    if expected_fail:
        # A listed gap is either untraceable (raised above) or traces to the
        # WRONG value -- TwoPowerSpherical compiles and returns inf. Check the
        # value before declaring it fixed, or a wrong-value gap looks like an
        # XPASS and gets deleted from the list while still being broken.
        if not numpy.allclose(got, ref, rtol=1e-6, atol=_ATOL):
            pytest.xfail(f"{name}.{entry} traces but does not match eager")
        pytest.fail(
            f"{name}.{entry} now traces under jax.jit AND matches eager"
            " -- remove it from _NOT_TRACEABLE"
        )
    numpy.testing.assert_allclose(got, ref, rtol=1e-6, atol=_ATOL)


# What these cases verify is that a freshly built potential's public entry point
# TRACES under torch dynamo (and, with fullgraph=False, falls back cleanly) --
# that is galpy's contract. torch.compile defaults to the inductor backend, whose
# per-potential codegen dominates this shard (Isochrone alone is ~47 s; the file
# was ~38 min), and whether inductor's *generated kernel* is correct is torch's
# concern, not galpy's. So the sweep runs under the cheap "eager" backend, which
# still exercises tracing and the fall-back. _INDUCTOR_NAMES opts individual
# potentials back onto real inductor -- empty by default; populate it to
# spot-check inductor codegen for a specific potential under investigation.
_INDUCTOR_NAMES = frozenset()


@pytest.mark.skipif("not _TORCH_COMPILES")
@pytest.mark.parametrize("name,entry", _CASES)
def test_torch_compile_traces_public_entry_point(name, entry):
    # A FRESH instance, not the shared one: several potentials build their
    # backend tables lazily on the first backend call, so a shared instance
    # makes the verdict depend on whether another case for the same potential
    # already warmed it -- which under xdist depends on worker assignment. The
    # contract worth measuring is "can I compile a potential I just built".
    pot = _fresh(name)
    ref = _reference(pot, entry)
    if ref is None:
        pytest.skip(f"{name}.{entry} not applicable")
    fn = _ENTRY[entry]
    expected_fail = (name, entry, "torch") in _NOT_TRACEABLE
    try:
        compiled = torch.compile(
            lambda R, z: fn(pot, R, z),
            fullgraph=False,
            dynamic=False,
            backend="inductor" if name in _INDUCTOR_NAMES else "eager",
        )
        got = float(compiled(torch.tensor(_R0), torch.tensor(_Z0)))
    except Exception as exc:
        if expected_fail:
            pytest.xfail(f"{name}.{entry} not torch-compilable: {type(exc).__name__}")
        raise
    if expected_fail:
        # A listed gap is either untraceable (raised above) or traces to the
        # WRONG value -- TwoPowerSpherical compiles and returns inf. Check the
        # value before declaring it fixed, or a wrong-value gap looks like an
        # XPASS and gets deleted from the list while still being broken.
        if not numpy.allclose(got, ref, rtol=1e-6, atol=_ATOL):
            pytest.xfail(f"{name}.{entry} compiles but does not match eager")
        pytest.fail(
            f"{name}.{entry} now compiles under torch AND matches eager"
            " -- remove it from _NOT_TRACEABLE"
        )
    numpy.testing.assert_allclose(got, ref, rtol=1e-6, atol=_ATOL)


def test_zoo_is_actually_covered():
    # Guard against the discovery silently finding nothing (a rename would make
    # both parametrised tests vacuous).
    assert len(_POTS) > 30, f"only {len(_POTS)} potentials discovered"
