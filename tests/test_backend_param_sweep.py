###############################################################################
# test_backend_param_sweep.py: autodiff w.r.t. EVERY numeric constructor
# parameter (and one coordinate) of every public potential, df and actionAngle
# class, vs a Richardson-extrapolated central finite difference.
#
# Specs live in backend_param_specs.py; each builds its object INSIDE the
# differentiated function. Cases are (spec, key, backend, mode, eager|jit):
#
#   data         backend-array coordinates; only the differentiated key is a
#                backend array (the other parameters are Python floats)
#   forced       Python-float coordinates under use(backend, force=True)
#   floatcoords  Python-float coordinates, backend key, no force
#
# jit (jax.jit of jax.jacrev) only for jax. The reference is a numpy FD cached
# per (spec, key); sampled outputs use an FD on the same backend ("self"), and
# numerical DFs the numpy FD of their closed-form twin ("ref:<spec>").
#
# Known failures are strict xfails listed one per line in
# backend_param_known_gaps.txt, so a fix XPASSes until its line is deleted.
# Slow cases are skipped via backend_slow_skip.txt (both read here, since this
# file manages its own backends).
###############################################################################
import contextlib
import inspect
import math
import os

import conftest
import numpy
import pytest
from backend_param_specs import SPECS, SPECS_BY_NAME

pytestmark = pytest.mark.backend_managed

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

_HERE = os.path.dirname(os.path.abspath(__file__))
_GAPS_PATH = os.path.join(_HERE, "backend_param_known_gaps.txt")
_NODE_PREFIX = "tests/test_backend_param_sweep.py::test_param_grad"
MODES = ("data", "forced", "floatcoords")
RTOL, ATOL = 1e-6, 1e-10

# Public classes without a spec of their own: abstract bases, pynbody-only
# snapshots and the out-of-scope Torus wrapper.
NO_SPEC = {
    "Force": "abstract base",
    "planarForce": "abstract base",
    "Potential": "abstract base",
    "planarPotential": "abstract base",
    "NumericalPotentialDerivativesMixin": "mixin",
    "SnapshotRZPotential": "needs pynbody (not in the test_backend shards)",
    "InterpSnapshotRZPotential": "needs pynbody (not in the test_backend shards)",
    "sphericaldf": "abstract base",
    "actionAngle": "abstract base",
    "actionAngleInverse": "abstract base",
    "actionAngleTorus": "external C++ Torus library: out of backend scope",
}


def _keys(spec):
    return [(k, True) for k in spec["params"]] + [
        (k, False) for k in spec.get("diff_coords", list(spec["coords"])[:1])
    ]


def _case_id(name, key, bk, mode, jit):
    return f"{name}-{key}-{bk}-{mode}-{jit}"


def _all_cases():
    for spec in SPECS:
        for key, _ in _keys(spec):
            for bk in ("jax", "torch"):
                for mode in MODES:
                    if mode == "floatcoords" and not spec["coords"]:
                        continue  # no coordinates: identical to data
                    for jit in ("eager", "jit") if bk == "jax" else ("eager",):
                        yield (spec["name"], key, bk, mode, jit)


def _read_known_gaps(path=_GAPS_PATH):
    """{(spec, key, backend, mode, jit): reason} from the known-gaps file."""
    gaps = {}
    with open(path) as fh:
        for raw in fh:
            line, _, reason = raw.partition("#")
            if not line.strip():
                continue
            entry = tuple(line.split())
            assert len(entry) == 5, f"malformed known-gaps line: {raw!r}"
            assert entry not in gaps, f"duplicate known-gaps line: {raw!r}"
            gaps[entry] = reason.strip()
    return gaps


def _slow_skips():
    path = os.path.join(_HERE, conftest._SLOW_SKIP_FILENAME)
    return {
        bk: conftest._load_backend_nodeids(path, bk, inherit_eager=False)
        for bk in ("jax", "torch")
    }


def _params():
    gaps = _read_known_gaps()
    slow = _slow_skips()
    out = []
    for case in _all_cases():
        cid = _case_id(*case)
        marks = []
        if f"{_NODE_PREFIX}[{cid}]" in slow[case[2]]:
            marks.append(pytest.mark.skip(reason="backend-slow-skip"))
        elif case in gaps:
            marks.append(pytest.mark.xfail(strict=True, reason=gaps[case]))
        out.append(pytest.param(*case, id=cid, marks=marks))
    return out


@pytest.fixture(scope="session")
def fd_cache():
    return {}


@pytest.fixture
def _clear_jax_caches(bk):
    # thousands of distinct compiled functions otherwise accumulate (>8 GB)
    yield
    if bk == "jax":
        jax.clear_caches()


def _flt(v):
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    return float(numpy.asarray(v).reshape(-1)[0])


def _richardson(f, x0, s, hs=(2e-3, 1e-3)):
    """Richardson central FD of every output of f at x0; {out: (value, error)}."""

    def cd(h):
        a, b = f(x0 + h), f(x0 - h)
        return {k: (a[k] - b[k]) / (2 * h) for k in a}

    def rich(h):
        d1, d2 = cd(h), cd(h / 2)
        return {k: (4 * d2[k] - d1[k]) / 3 for k in d1}

    r1, r2 = rich(hs[0] * s), rich(hs[1] * s)
    return {k: (r2[k], abs(r2[k] - r1[k])) for k in r2}


def _numpy_fd(spec, key, isparam):
    P0, C0 = dict(spec["params"]), dict(spec["coords"])

    def f(x):
        P, C = dict(P0), dict(C0)
        (P if isparam else C)[key] = x
        out = spec["outputs"](spec["build"](P), C)
        return {k: _flt(v) for k, v in out.items()}

    x0 = (P0 if isparam else C0)[key]
    return _richardson(f, x0, max(abs(x0), 0.1) * spec.get("fdscale", 1.0))


def _to_backend(bk, v):
    if bk == "jax":
        return jnp.asarray(v, dtype=jnp.float64)
    return torch.tensor(float(v), dtype=torch.float64)


def _inputs(spec, key, isparam, x, bk, mode):
    """(P, C) for one evaluation: x is the (backend) value of the key."""
    P = dict(spec["params"])
    C = {
        k: (_to_backend(bk, v) if mode == "data" else v)
        for k, v in spec["coords"].items()
    }
    (P if isparam else C)[key] = x
    return P, C


def _context(bk, mode):
    if mode != "forced":
        return contextlib.nullcontext()
    from galpy.backend import use

    return use(bk, force=True)


def _evaluate(spec, P, C, bk, mode):
    with _context(bk, mode):
        return spec["outputs"](spec["build"](P), C)


def _self_fd(spec, key, isparam, bk, mode):
    P0, C0 = spec["params"], spec["coords"]

    def f(x):
        P, C = _inputs(spec, key, isparam, _to_backend(bk, x), bk, mode)
        return {k: _flt(v) for k, v in _evaluate(spec, P, C, bk, mode).items()}

    x0 = (P0 if isparam else C0)[key]
    s = max(abs(x0), 0.1) * spec.get("fdscale", 1.0)
    return _richardson(f, x0, s, spec.get("self_fd_h", (2e-3, 1e-3)))


def _reference(spec, key, isparam, bk, mode, fd_cache):
    fd = spec.get("fd", "numpy")
    if fd == "self":  # the sampling key's stream can differ between modes
        ck = (spec["name"], key, bk, mode)
        if ck not in fd_cache:
            fd_cache[ck] = _self_fd(spec, key, isparam, bk, mode)
        return fd_cache[ck]
    if fd.startswith("ref:"):
        return _reference(SPECS_BY_NAME[fd[4:]], key, isparam, bk, mode, fd_cache)
    ck = (spec["name"], key, "numpy")
    if ck not in fd_cache:
        fd_cache[ck] = _numpy_fd(spec, key, isparam)
    return fd_cache[ck]


def _jax_grads(spec, key, isparam, mode, jit):
    not_tensor = set()

    def f(x):
        P, C = _inputs(spec, key, isparam, x, "jax", mode)
        out = _evaluate(spec, P, C, "jax", mode)
        for k, v in out.items():
            if not isinstance(v, jax.Array):
                not_tensor.add(k)
        return {k: jnp.reshape(jnp.asarray(v), (-1,))[0] for k, v in out.items()}

    x0 = _to_backend("jax", (spec["params"] if isparam else spec["coords"])[key])
    g = jax.jacrev(f)
    g = jax.jit(g)(x0) if jit == "jit" else g(x0)
    return {k: _flt(v) for k, v in g.items()}, not_tensor


def _torch_grads(spec, key, isparam, mode):
    x = torch.tensor(
        float((spec["params"] if isparam else spec["coords"])[key]),
        dtype=torch.float64,
        requires_grad=True,
    )
    P, C = _inputs(spec, key, isparam, x, "torch", mode)
    out = _evaluate(spec, P, C, "torch", mode)
    grads, not_tensor = {}, set()
    for k, v in out.items():
        if not isinstance(v, torch.Tensor):
            not_tensor.add(k)
            continue
        v = v.reshape(-1)[0]
        if v.grad_fn is None:
            grads[k] = None  # disconnected
            continue
        (g,) = torch.autograd.grad(v, x, retain_graph=True, allow_unused=True)
        grads[k] = None if g is None else float(g)
    return grads, not_tensor


def _classify(g, ref, err, not_tensor, rtol=RTOL):
    zero_ref = abs(ref) <= max(10 * err, ATOL)
    if not_tensor:
        return "ok" if zero_ref else "NOT_TENSOR"
    if g is None:
        return "ok" if zero_ref else "DISC"
    if not math.isfinite(g):
        return "NONFINITE"
    if abs(g - ref) <= max(rtol * abs(ref), 10 * err, ATOL):
        return "ok"
    if g == 0.0:
        return f"ZERO(fd={ref:.6g})"
    if err > 1e-3 * max(abs(ref), 1e-8):
        return f"FD_UNCONVERGED(ad={g:.6g},fd={ref:.6g}+-{err:.2g})"
    rel = abs(g - ref) / max(abs(ref), 1e-12)
    return f"WRONG(ad={g:.10g},fd={ref:.10g},rel={rel:.2g})"


@pytest.mark.parametrize("spec_name,key,bk,mode,jit", _params())
def test_param_grad(spec_name, key, bk, mode, jit, fd_cache, _clear_jax_caches):
    if (bk == "jax" and jax is None) or (bk == "torch" and torch is None):
        pytest.skip(f"{bk} not installed")  # pragma: no cover
    spec = SPECS_BY_NAME[spec_name]
    isparam = key in spec["params"]
    ref = _reference(spec, key, isparam, bk, mode, fd_cache)
    if bk == "jax":
        grads, not_tensor = _jax_grads(spec, key, isparam, mode, jit)
    else:
        grads, not_tensor = _torch_grads(spec, key, isparam, mode)
    bad = {}
    for out, (r, err) in ref.items():
        if out not in grads and out not in not_tensor:
            continue  # a "ref:" twin's extra output
        status = _classify(
            grads.get(out), r, err, out in not_tensor, spec.get("rtol", RTOL)
        )
        if status != "ok":
            bad[out] = status
    assert not bad, f"d/d{key}: {bad}"


def _covered(spec):
    if "covers" in spec:
        return set(spec["covers"])
    return {spec["name"].split("[")[0].split("(")[0]}


def test_every_public_class_has_a_spec():
    import galpy.actionAngle
    import galpy.df
    import galpy.potential

    covered = set().union(*(_covered(s) for s in SPECS))
    missing = []
    for mod in (galpy.potential, galpy.df, galpy.actionAngle):
        for name in dir(mod):
            obj = getattr(mod, name)
            if (
                name.startswith("_")
                or not inspect.isclass(obj)
                or not obj.__module__.startswith("galpy")
                or issubclass(obj, BaseException)
            ):
                continue
            if name not in covered and name not in NO_SPEC:
                missing.append(f"{mod.__name__}.{name}")
    assert not missing, f"public classes without a param-sweep spec: {missing}"
    assert not (set(NO_SPEC) & covered), "NO_SPEC lists a class that has a spec"


def test_spec_names_and_keys_are_unique_tokens():
    assert len(SPECS_BY_NAME) == len(SPECS), "duplicate spec names"
    for spec in SPECS:
        keys = [k for k, _ in _keys(spec)]
        assert len(set(keys)) == len(keys), f"{spec['name']}: param/coord clash"
        assert spec["name"].split() == [spec["name"]], spec["name"]
        assert "#" not in spec["name"] and "-" not in spec["name"], spec["name"]


def test_known_gaps_file_is_sorted_and_live():
    """One entry per line, sorted, and each names an existing case."""
    with open(_GAPS_PATH) as fh:
        entries = [
            " ".join(raw.partition("#")[0].split())
            for raw in fh
            if raw.partition("#")[0].strip()
        ]
    assert entries == sorted(entries), "backend_param_known_gaps.txt is not sorted"
    cases = set(_all_cases())
    stale = [g for g in _read_known_gaps() if g not in cases]
    assert not stale, f"known gaps for cases that do not exist: {stale}"


def test_slow_skips_are_live():
    ids = {f"{_NODE_PREFIX}[{_case_id(*c)}]" for c in _all_cases()}
    stale = [
        n
        for entries in _slow_skips().values()
        for n in entries
        if n.startswith(_NODE_PREFIX) and n not in ids
    ]
    assert not stale, f"slow-skips for cases that do not exist: {stale}"
