###############################################################################
# test_backend_conventions.py: enforce the backend namespace-swap conventions on
# migrated potentials, so a migrated potential can never regress to bare numpy in
# its compute methods. The checked set is derived from the potentials themselves,
# so a newly migrated one is covered the day it lands.
###############################################################################
import ast
import inspect
import pathlib

import numpy
import pytest

from galpy import potential

# Static source analysis — independent of the active backend.
pytestmark = pytest.mark.backend_managed

# Non-numpy namespaces available, for the coerce_coords coverage test below.
_NS = {"numpy": numpy}
try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    _NS["jax"] = jnp
except ImportError:  # pragma: no cover
    pass
try:
    import torch

    torch.set_default_dtype(torch.float64)
    _NS["torch"] = torch
except ImportError:  # pragma: no cover
    pass

# Private compute methods that must be backend-agnostic: no bare ``numpy.<fn>``
# on a path a backend array can reach. Use ``xp = get_namespace(...)`` then
# ``xp.<fn>``; bare numpy is fine behind a namespace guard (``if xp is numpy:``)
# and for scalar constants (see SAFE_NUMPY_ATTRS).
COMPUTE_METHODS = {
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
    "_surfdens",
    "_revaluate",
    "_rforce",
    "_r2deriv",
    "_rdens",
}

# ``numpy`` attributes that are plain Python scalars rather than array
# operations: ``numpy.pi`` is bit-for-bit ``math.pi`` and ``numpy.newaxis`` IS
# ``None``, so they behave identically under every backend and need no ``xp.``
# form. Nothing callable may be added here -- see the test below.
SAFE_NUMPY_ATTRS = frozenset({"pi", "e", "euler_gamma", "inf", "nan", "newaxis"})


def _numpy_guard(test):
    """Polarity of a namespace guard.

    ``True``  -- the test holding means we are on the numpy path,
    ``False`` -- it means we are on a backend path,
    ``None``  -- the test says nothing about the namespace.
    """
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        inner = _numpy_guard(test.operand)
        return None if inner is None else not inner
    if (
        isinstance(test, ast.Compare)
        and len(test.ops) == 1
        and isinstance(test.left, ast.Name)
        and test.left.id == "xp"
        and isinstance(test.comparators[0], ast.Name)
        and test.comparators[0].id == "numpy"
    ):
        if isinstance(test.ops[0], ast.Is):
            return True
        if isinstance(test.ops[0], ast.IsNot):
            return False
    if (
        isinstance(test, ast.Call)
        and isinstance(test.func, ast.Name)
        and test.func.id == "is_backend_array"
    ):
        return False
    return None


def _always_exits(body):
    """True if control can not fall out of the bottom of ``body``."""
    if not body:
        return False
    last = body[-1]
    if isinstance(last, (ast.Return, ast.Raise)):
        return True
    if isinstance(last, ast.If) and last.orelse:
        return _always_exits(last.body) and _always_exits(last.orelse)
    return False


class _BareNumpyVisitor:
    """Collect bare ``numpy.<fn>`` uses in compute methods that a backend array
    can actually reach.

    A migrated potential is allowed a numpy branch alongside its backend branch
    (that dual path is the documented convention, not a numpy island), so uses
    inside a region a namespace guard proves is numpy-only are not violations.
    Tracking that is what lets dual-path modules be checked at all -- and they
    are the ones where the check earns its keep, since the two branches sit
    side by side and it is easy to edit the wrong one.
    """

    def __init__(self):
        self.violations = []

    def visit(self, tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in COMPUTE_METHODS:
                self._block(node.body, node.name, False)
        return self

    def _block(self, stmts, where, numpy_only):
        for stmt in stmts:
            if isinstance(stmt, ast.If):
                polarity = _numpy_guard(stmt.test)
                self._expr(stmt.test, where, numpy_only)
                self._block(stmt.body, where, numpy_only or polarity is True)
                self._block(stmt.orelse, where, numpy_only or polarity is False)
                # `if <backend>: return ...` with no else: the rest of this
                # block is only reachable on the numpy path
                if polarity is False and not stmt.orelse and _always_exits(stmt.body):
                    numpy_only = True
                continue
            for field, value in ast.iter_fields(stmt):
                items = value if isinstance(value, list) else [value]
                if field in ("body", "orelse", "finalbody"):
                    # nested block (loop, with, try, closure): same path
                    self._block(
                        [s for s in items if isinstance(s, ast.AST)], where, numpy_only
                    )
                else:
                    for item in items:
                        if isinstance(item, ast.AST):
                            self._expr(item, where, numpy_only)

    def _expr(self, node, where, numpy_only):
        if isinstance(node, ast.IfExp):  # x if <guard> else y
            polarity = _numpy_guard(node.test)
            self._expr(node.test, where, numpy_only)
            self._expr(node.body, where, numpy_only or polarity is True)
            self._expr(node.orelse, where, numpy_only or polarity is False)
            return
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "numpy"
        ):
            if not numpy_only and node.attr not in SAFE_NUMPY_ATTRS:
                self.violations.append((where, f"numpy.{node.attr}", node.lineno))
            return
        for child in ast.iter_child_nodes(node):
            self._expr(child, where, numpy_only)


def _declares_backend_compatible(path):
    """True if a class in this module sets ``self._backend_compatible = True``."""
    for node in ast.walk(ast.parse(path.read_text())):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and node.value.value is True
            and any(
                isinstance(t, ast.Attribute) and t.attr == "_backend_compatible"
                for t in node.targets
            )
        ):
            return True
    return False


# Derived, not hand-maintained: every module declaring a backend-compatible
# potential is checked, so a newly migrated potential is covered the day it
# lands. The allowlist this replaces had drifted to 15 of 48 modules.
CHECKED_MODULES = sorted(
    {
        p
        for p in pathlib.Path(potential.__file__).parent.glob("*.py")
        if _declares_backend_compatible(p)
    }
    | {
        # these two inherit the flag from the potential they wrap, so the scan
        # above cannot see it; they were on the original allowlist
        pathlib.Path(inspect.getmodule(getattr(potential, _name)).__file__)
        for _name in (
            "SolidBodyRotationWrapperPotential",
            "TimeDependentAmplitudeWrapperPotential",
        )
    }
)


@pytest.mark.parametrize("module_path", CHECKED_MODULES, ids=lambda p: p.stem)
def test_no_bare_numpy_in_compute_methods(module_path):
    violations = (
        _BareNumpyVisitor().visit(ast.parse(module_path.read_text())).violations
    )
    assert not violations, (
        f"{module_path.name}: bare numpy.* reachable from a backend path in "
        f"compute methods. Use xp = get_namespace(...) then xp.<fn>, or put the "
        f"use behind `if xp is numpy:`. Sites: {violations}"
    )


def test_checked_module_registry_is_populated():
    """CHECKED_MODULES is derived by an AST scan, so a scan that silently stops
    matching would empty the parametrization above and make it a vacuous pass.
    """
    assert len(CHECKED_MODULES) >= 40, (
        f"registry collapsed to {len(CHECKED_MODULES)} modules: "
        f"{sorted(p.stem for p in CHECKED_MODULES)}"
    )
    # anchored on the classes, not on file names, so moving a class between
    # modules cannot quietly drop it from the checked set
    checked = set(CHECKED_MODULES)
    for anchor in (
        "PlummerPotential",
        "NFWPotential",
        "MiyamotoNagaiPotential",
        "RazorThinExponentialDiskPotential",
        "SolidBodyRotationWrapperPotential",
    ):
        home = pathlib.Path(inspect.getmodule(getattr(potential, anchor)).__file__)
        assert home in checked, f"{anchor} ({home.name}) dropped out of the checked set"


def test_safe_numpy_attrs_holds_only_scalar_constants():
    """The allowlist exists for constants like numpy.pi; letting an array
    operation in would blind the checker to exactly what it looks for.
    """
    for attr in SAFE_NUMPY_ATTRS:
        value = getattr(numpy, attr)
        assert value is None or isinstance(value, float), (
            f"numpy.{attr} is {type(value).__name__}, not a scalar constant"
        )


def _scan(body):
    return (
        _BareNumpyVisitor()
        .visit(ast.parse(f"def _evaluate(self, R):\n{body}"))
        .violations
    )


# For each namespace-guard form: (numpy use on the numpy side -> exempt,
# the same guard with the numpy use moved to the BACKEND side -> must be
# caught). The second half is what stops the exemptions from quietly
# swallowing the regressions they exist to permit past.
_GUARD_CASES = {
    "if_xp_is_numpy": (
        "    if xp is numpy:\n        return numpy.exp(R)\n    return xp.exp(R)\n",
        "    if xp is numpy:\n        return xp.exp(R)\n    return numpy.exp(R)\n",
    ),
    "early_return_backend": (
        "    if xp is not numpy:\n        return xp.exp(R)\n    return numpy.exp(R)\n",
        "    if xp is not numpy:\n        return numpy.exp(R)\n    return xp.exp(R)\n",
    ),
    "else_of_is_backend_array": (
        "    if is_backend_array(R):\n        return xp.exp(R)\n    else:\n        return numpy.exp(R)\n",
        "    if is_backend_array(R):\n        return numpy.exp(R)\n    else:\n        return xp.exp(R)\n",
    ),
    "not_is_backend_array": (
        "    if not is_backend_array(R):\n        return numpy.exp(R)\n    return xp.exp(R)\n",
        "    if not is_backend_array(R):\n        return xp.exp(R)\n    return numpy.exp(R)\n",
    ),
    "ternary": (
        "    return numpy.exp(R) if xp is numpy else xp.exp(R)\n",
        "    return xp.exp(R) if xp is numpy else numpy.exp(R)\n",
    ),
    "nested_in_loop": (
        "    if xp is numpy:\n        for i in range(3):\n            R = numpy.exp(R)\n    return R\n",
        "    for i in range(3):\n        R = numpy.exp(R)\n    return R\n",
    ),
}


@pytest.mark.parametrize("form", sorted(_GUARD_CASES))
def test_numpy_guarded_use_is_not_a_violation(form):
    assert not _scan(_GUARD_CASES[form][0])


@pytest.mark.parametrize("form", sorted(_GUARD_CASES))
def test_bare_numpy_on_the_backend_side_is_caught(form):
    violations = _scan(_GUARD_CASES[form][1])
    assert [(where, name) for where, name, _ in violations] == [
        ("_evaluate", "numpy.exp")
    ]


def test_scalar_constants_are_allowed_on_a_backend_path():
    assert not _scan("    return numpy.pi * R + numpy.newaxis\n")


def test_unguarded_array_function_is_a_violation():
    assert [v[1] for v in _scan("    return numpy.exp(R)\n")] == ["numpy.exp"]


def test_only_compute_methods_are_checked():
    """__init__ machinery is numpy-by-design; the check must not spread to it."""
    assert (
        not _BareNumpyVisitor()
        .visit(ast.parse("def __init__(self, R):\n    self._glx = numpy.exp(R)\n"))
        .violations
    )


###############################################################################
# _backend_compatible flag (set in each migrated potential's __init__, like hasC)
# and the _check_backend_compatible gate used by potential_physical_input to
# coerce coordinate inputs only for backend-aware targets. A migrated leaf reads
# True; a list is compatible iff every member is; a wrapper iff it and its wrapped
# potential are. The flag set was derived empirically (each runs its compute
# methods under forced jax+torch, scalar and array, returning a backend array).

# Sample of migrated potentials that construct with normalize=1.0 (the full set
# is exercised by the all-backend suite).
_MIGRATED_SAMPLE = [
    "MiyamotoNagaiPotential",
    "NFWPotential",
    "PlummerPotential",
    "IsochronePotential",
    "PowerSphericalPotential",
    "KeplerPotential",
    "LogarithmicHaloPotential",
    "BurkertPotential",
    "EinastoPotential",
    "HernquistPotential",
    "KuzminDiskPotential",
    "DoubleExponentialDiskPotential",
    "PowerTriaxialPotential",
    "MN3ExponentialDiskPotential",
    "RingPotential",
    "FerrersPotential",
    "KuzminKutuzovStaeckelPotential",
    "PseudoIsothermalPotential",
    "RazorThinExponentialDiskPotential",
    "AnyAxisymmetricRazorThinDiskPotential",
    "AnySphericalPotential",
]


# Potentials whose CONSTRUCTOR does real numerical work (a root solve, a special
# function) on its scalar parameters. Under ``use(backend, force=True)`` that work
# must run ON the backend: the params arrive as plain Python floats, so without a
# coerce_coords at the top of __init__ the setup silently falls through to scipy
# and the potential is built on numpy inside a forced-backend run. That is
# invisible to a test that calls the setup helper with a hand-passed value --
# it has to go through the CONSTRUCTOR and inspect what the constructor stored.
# (name, kwargs, attributes that must end up as backend arrays)
_CONSTRUCTS_ON_BACKEND = [
    ("EinastoPotential", {"amp": 1.0, "n": 4.0, "rs": 1.0}, ("n", "amp", "h")),
    (
        "ExpTruncNFWPotential",
        {"amp": 1.0, "a": 2.0, "rc": 1.5},
        ("a", "rc", "_alpha", "_E1_alpha", "_Ftot"),
    ),
    # gamma-of-a-param normalizations: the param is NOT stored coerced (the
    # Python-level branches and the numpy.fabs(b-1) check below need raw
    # floats), so what has to land on the backend is the computed constant.
    (
        "FerrersPotential",
        {"amp": 1.0, "n": 2, "a": 1.0, "b": 0.7, "c": 0.5},
        ("_rhoc_M",),
    ),
    (
        "TwoPowerTriaxialPotential",
        {"amp": 1.0, "a": 1.0, "alpha": 1.0, "beta": 4.0},
        ("psi_inf",),
    ),
]

# The spherical power-law DFs build their fE normalization the same way, from
# gamma() of a derived exponent. They take a potential rather than plain kwargs,
# so they get their own registry: (name, df kwargs, attributes).
_DF_CONSTRUCTS_ON_BACKEND = [
    ("isotropicPowerLawdf", {}, ("_fEnorm",)),
    ("constantbetaPowerLawdf", {"beta": 0.3}, ("_fEnorm",)),
    ("osipkovmerrittPowerLawdf", {"ra": 1.5}, ("_A1", "_A2")),
]


def _plaw_pot():
    return potential.PowerSphericalPotential(amp=1.0, alpha=2.5, normalize=False)


@pytest.mark.parametrize("clsname,kwargs,attrs", _DF_CONSTRUCTS_ON_BACKEND)
@pytest.mark.parametrize("backend", [b for b in _NS if b != "numpy"])
def test_df_constructor_builds_on_the_forced_backend(clsname, kwargs, attrs, backend):
    """A forced backend must build the DF's normalization ON that backend."""
    from galpy import backend as _backend
    from galpy import df as _df
    from galpy.backend import is_backend_array

    cls = getattr(_df, clsname)
    with _backend.use(backend, force=True):
        d = cls(pot=_plaw_pot(), **kwargs)
        stored = {a: getattr(d, a) for a in attrs}
    numpy_attrs = [a for a, v in stored.items() if not is_backend_array(v)]
    assert not numpy_attrs, (
        f"{clsname} under forced {backend}: {numpy_attrs} stayed numpy, so the "
        f"normalization was computed on scipy inside a forced-backend run"
    )


@pytest.mark.parametrize("clsname,kwargs,attrs", _DF_CONSTRUCTS_ON_BACKEND)
def test_df_constructor_coercion_is_a_no_op_without_a_force(clsname, kwargs, attrs):
    """No force -> strict pass-through, the numpy path is untouched."""
    from galpy import df as _df
    from galpy.backend import is_backend_array

    d = getattr(_df, clsname)(pot=_plaw_pot(), **kwargs)
    leaked = [a for a in attrs if is_backend_array(getattr(d, a))]
    assert not leaked, f"{clsname} unforced: {leaked} became backend arrays"


# (name, kwargs, param differentiated, attribute read back, backends where the
# gradient is blocked UPSTREAM with the reason). Being a backend array is not
# enough on its own -- see the test below.
_CONSTRUCTOR_GRADS = [
    (
        "EinastoPotential",
        {"amp": 1.0, "rs": 1.0},
        "n",
        4.0,
        "amp",
        # torch implements no derivative for igammac w.r.t. its order, so no
        # gradient can flow through gammaincc at all on torch.
        {"torch"},
    ),
    ("ExpTruncNFWPotential", {"amp": 1.0, "rc": 1.5}, "a", 2.0, "_Ftot", set()),
    (
        "FerrersPotential",
        {"amp": 1.0, "a": 1.0, "b": 0.7, "c": 0.5},
        "n",
        2.0,
        "_rhoc_M",
        set(),
    ),
    (
        "TwoPowerTriaxialPotential",
        {"amp": 1.0, "a": 1.0, "beta": 4.0},
        "alpha",
        1.0,
        "psi_inf",
        set(),
    ),
]


@pytest.mark.parametrize("clsname,kwargs,pname,pval,attr,blocked", _CONSTRUCTOR_GRADS)
@pytest.mark.parametrize("backend", [b for b in _NS if b != "numpy"])
def test_constructor_output_is_differentiable_in_its_param(
    clsname, kwargs, pname, pval, attr, blocked, backend
):
    """Being a backend array is NOT enough -- the graph has to be intact.

    ``scipy.special.gamma(tensor)`` RETURNS a tensor (it round-trips through
    ``__array__``), so a param can look perfectly coerced while the autograd
    graph is silently cut; on a grad-requiring tensor the same call raises
    outright. A setup path built on scipy would therefore sail through the
    is_backend_array check above while delivering no derivative at all. Compare
    against a central difference computed on the NUMPY path, so this is an
    independent reference rather than the backend checking itself.
    """
    if backend in blocked:
        pytest.skip(f"{clsname} d/d{pname} is blocked upstream on {backend}")
    cls = getattr(potential, clsname)

    def build(v):
        return getattr(cls(**{**kwargs, pname: v}), attr)

    h = 1e-6
    fd = (float(build(pval + h)) - float(build(pval - h))) / (2.0 * h)
    if backend == "jax":
        import jax

        got = float(jax.grad(lambda v: build(v))(_NS["jax"].asarray(pval)))
    else:
        import torch

        t = torch.tensor(pval, dtype=torch.float64, requires_grad=True)
        out = build(t)
        assert out.grad_fn is not None, (
            f"{clsname}.{attr} is a backend array but DETACHED from {pname} -- "
            f"the setup path round-tripped through numpy/scipy"
        )
        out.backward()
        got = float(t.grad)
    assert abs(got - fd) < 1e-6 * abs(fd), (
        f"{clsname} d({attr})/d{pname} on {backend}: {got!r} vs finite "
        f"difference {fd!r} (rel {abs(got - fd) / abs(fd):.3e})"
    )


@pytest.mark.parametrize("clsname,kwargs,attrs", _CONSTRUCTS_ON_BACKEND)
@pytest.mark.parametrize("backend", [b for b in _NS if b != "numpy"])
def test_constructor_builds_on_the_forced_backend(clsname, kwargs, attrs, backend):
    """A forced backend must build the potential ON that backend, not on scipy."""
    from galpy import backend as _backend
    from galpy.backend import is_backend_array

    cls = getattr(potential, clsname)
    with _backend.use(backend, force=True):
        pot = cls(**kwargs)
        stored = {a: getattr(pot, a) for a in attrs}
    numpy_attrs = [a for a, v in stored.items() if not is_backend_array(v)]
    assert not numpy_attrs, (
        f"{clsname} under forced {backend}: {numpy_attrs} stayed numpy, so "
        f"construction fell through to scipy inside a forced-backend run "
        f"({ {a: type(stored[a]).__name__ for a in numpy_attrs} })"
    )


@pytest.mark.parametrize("clsname,kwargs,attrs", _CONSTRUCTS_ON_BACKEND)
def test_constructor_coercion_is_a_no_op_without_a_force(clsname, kwargs, attrs):
    """No force -> coerce_coords is a strict pass-through, numpy path untouched."""
    from galpy.backend import is_backend_array

    pot = getattr(potential, clsname)(**kwargs)
    leaked = [a for a in attrs if is_backend_array(getattr(pot, a))]
    assert not leaked, f"{clsname} unforced: {leaked} became backend arrays"


@pytest.mark.parametrize("clsname", _MIGRATED_SAMPLE)
def test_backend_compatible_true(clsname):
    from galpy.potential import _check_backend_compatible as cbc

    assert cbc(getattr(potential, clsname)(normalize=1.0)) is True


def test_backend_compatible_false():
    from galpy.potential import _check_backend_compatible as cbc

    # Synthetic "unmigrated" leaf: a migrated potential with the flag forced
    # off. A real-potential negative sample rots the moment its members land
    # their backend PRs (AnySpherical/AnyAxisym did), so force the flag instead.
    p = potential.MiyamotoNagaiPotential(normalize=1.0)
    p._backend_compatible = False
    assert cbc(p) is False


def test_check_backend_compatible_semantics():
    from galpy.potential import _check_backend_compatible as cbc

    mn = potential.MiyamotoNagaiPotential(normalize=1.0)
    # Synthetic "unmigrated" leaves: the flag forced off (robust to migration,
    # see test_backend_compatible_false).
    unmig = potential.MiyamotoNagaiPotential(normalize=1.0)
    unmig._backend_compatible = False
    unmig2 = potential.MiyamotoNagaiPotential(normalize=1.0)
    unmig2._backend_compatible = False
    # combined potential: all members must be compatible
    assert cbc([mn, potential.NFWPotential(normalize=1.0)]) is True
    assert cbc([mn, unmig]) is False
    # composite potential (pot1+pot2): delegates to its components, so it is
    # compatible iff every component is (mirrors the list branch)
    assert cbc(mn + potential.NFWPotential(normalize=1.0)) is True
    assert cbc(mn + unmig) is False
    # wrapper: own flag AND wrapped potential
    assert cbc(potential.OblateStaeckelWrapperPotential(pot=mn)) is True
    assert (
        cbc(potential.KuzminLikeWrapperPotential(amp=1.0, pot=unmig2, a=1.0, b=0.2))
        is False
    )
    # migrated amplitude wrappers: compatible iff the wrapped potential is too
    assert (
        cbc(potential.DehnenSmoothWrapperPotential(pot=mn, tform=-1.0, tsteady=1.0))
        is True
    )
    # a still-unmigrated amplitude wrapper backs out (own flag defaults to False)
    assert (
        cbc(potential.TimeDependentAmplitudeWrapperPotential(pot=mn, A=lambda t: 1.0))
        is False
    )
    # opts back out despite its interpSphericalPotential base
    ac = potential.AdiabaticContractionWrapperPotential(
        pot=mn, baryonpot=potential.NFWPotential(amp=0.2)
    )
    assert cbc(ac) is False
    # a non-potential first arg (e.g. a df instance) is never compatible
    assert cbc(object()) is False


@pytest.mark.parametrize("backend", list(_NS))
def test_coerce_coords_branches(backend):
    # Exercises every branch of coerce_coords: numpy pass-through, None,
    # float-dtype preservation, and python/int -> backend float64.
    from galpy.backend import coerce_coords

    xp = _NS[backend]
    f32 = numpy.float32 if backend != "torch" else None  # torch handles below
    R = numpy.array([1.0, 2.0])  # float64 array -> dtype preserved
    out = coerce_coords(xp, R, None, 1.0, 2)  # array, None, py-float, py-int
    if backend == "numpy":
        # strict pass-through: object-identical, byte-identical numpy path
        assert out == (R, None, 1.0, 2)
        return
    R_o, none_o, f_o, i_o = out
    assert none_o is None
    # py-float and py-int are lifted to the backend's float64
    for v in (R_o, f_o, i_o):
        assert "float64" in str(getattr(v, "dtype", ""))
    # a float32 input keeps its dtype (exit-cast policy still applies)
    if f32 is not None:
        (R32_o,) = coerce_coords(xp, R.astype(f32))
        assert "float32" in str(R32_o.dtype)
    else:  # torch float32 tensor
        (R32_o,) = coerce_coords(xp, torch.tensor([1.0, 2.0], dtype=torch.float32))
        assert "float32" in str(R32_o.dtype)


@pytest.mark.parametrize("backend", [b for b in _NS if b != "numpy"])
def test_scalar_only_gate_spares_unmigrated_potential(backend):
    # The check_potential_inputs_not_arrays decorator coerces R, z, phi onto the
    # active backend ONLY for _backend_compatible potentials. An UNMIGRATED
    # scalar-only potential must keep its plain python-float inputs even under a
    # forced backend, or its bare scipy.integrate.quad / numpy internals crash
    # ("sqrt(): argument must be Tensor, not float" / "'<' not supported between
    # numpy.ndarray and Tensor"). Regression guard for the decorator's
    # _backend_compatible gate. AnyAxisymmetricRazorThinDiskPotential is now
    # migrated (its force is backend-native), so use a forced-flag synthetic that
    # pins _backend_compatible=False and keeps the scipy scalar path (mirrors the
    # negative-example-fragility pattern in the auto-memory).
    import galpy.backend

    class _UnmigratedDisk(potential.AnyAxisymmetricRazorThinDiskPotential):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._backend_compatible = False  # pin: decorator must NOT coerce

    pot = _UnmigratedDisk(normalize=1.0)
    assert potential._check_backend_compatible(pot) is False
    ref = float(pot._evaluate(0.9, 0.1, 0.0, 0.0))
    with galpy.backend.use(backend, force=True):
        # flag False -> R, z stay python floats -> is_backend_array False ->
        # the scipy scalar path runs and returns a plain float, no crash.
        got = float(pot._evaluate(0.9, 0.1, 0.0, 0.0))
    numpy.testing.assert_allclose(got, ref, rtol=1e-10, atol=0.0)


###############################################################################
# Decorator ORDER: the jit boundary must sit inside the units layer.
#
# Decorators apply bottom-up, so a method written
#
#     @backend_input("R", "z")      <- jit boundary
#     @physical_conversion("force")  <- builds the Quantity
#
# attaches output units INSIDE the traced region, and Quantity.__new__ calls
# __array__, which a tracer refuses (TracerArrayConversionError). Putting
# physical_conversion OUTSIDE means the trace returns a concrete array and the
# units are attached afterwards, which is both traceable and what the eager
# path already does. This was 75 sites across 7 files, every one in the wrong
# order, costing 33 jax --jit failures that had been mis-diagnosed as "astropy
# and tracing are incompatible by design".
###############################################################################

_UNITS_DECORATORS = (
    "physical_conversion",
    "physical_conversion_tuple",
    "physical_conversion_actionAngle",
    "physical_conversion_actionAngleInverse",
)


def _decorator_name(dec):
    """Bare name of a decorator node, whether or not it is called."""
    node = dec.func if isinstance(dec, ast.Call) else dec
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _order_violations(path):
    """(qualname, units_idx, boundary_idx) where the boundary is OUTSIDE units."""
    out = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.FunctionDef):
            continue
        names = [_decorator_name(d) for d in node.decorator_list]
        if "backend_input" not in names:
            continue
        units = [ii for ii, n in enumerate(names) if n in _UNITS_DECORATORS]
        if not units:
            continue
        boundary = names.index("backend_input")
        # decorator_list is source order, i.e. OUTERMOST first: a lower index is
        # further out. The units decorator must be outside the boundary.
        if min(units) > boundary:
            out.append((node.name, min(units), boundary))
    return out


_UNITS_BOUNDARY_FILES = sorted(
    p
    for p in pathlib.Path(potential.__file__).parent.parent.rglob("*.py")
    if "backend_input" in p.read_text() and "physical_conversion" in p.read_text()
)


def test_units_boundary_files_are_found():
    """Guard the glob: an empty file list would make the check below vacuous."""
    assert len(_UNITS_BOUNDARY_FILES) >= 5, (
        f"only {len(_UNITS_BOUNDARY_FILES)} files carry both decorators; the "
        "search is probably broken"
    )


@pytest.mark.parametrize("module_path", _UNITS_BOUNDARY_FILES, ids=lambda p: p.stem)
def test_units_decorator_is_outside_the_jit_boundary(module_path):
    violations = _order_violations(module_path)
    assert not violations, (
        f"{module_path.name}: @backend_input is OUTSIDE the units decorator on "
        f"{[v[0] for v in violations]}. The Quantity would then be built inside "
        "the jit trace and raise TracerArrayConversionError; put "
        "@physical_conversion above @backend_input."
    )


def test_no_module_hand_rolls_namespace_detection():
    """Backend identity is asked via name_of_namespace, not by inspecting __name__.

    galpy.backend.name_of_namespace maps a resolved namespace to "numpy"/"jax"/
    "torch". Modules that re-derive that from ``xp.__name__`` are re-implementing
    it, and the re-implementation is only accidentally correct: ``"torch" in
    xp.__name__`` works, ``xp.__name__.startswith("torch")`` does NOT, because the
    torch namespace is ``array_api_compat.torch``. That exact slip disabled torch
    tracing everywhere in galpy/backend/_jit.py and no test noticed -- an untraced
    call returns the same values a traced one does, so only a coverage delta
    caught it.

    Asking through the helper makes the invariant structural instead of a
    convention every author has to remember.
    """
    import ast
    import pathlib

    import galpy

    root = pathlib.Path(galpy.__file__).parent
    # _namespaces.py DEFINES the mapping, so it is the one place that may look at
    # __name__ directly.
    allowed = {"_namespaces.py"}
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if path.name in allowed:
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover - galpy always parses
            continue
        for node in ast.walk(tree):
            # `<something>.__name__` compared against, or searched for, a backend
            # name -- in either argument order.
            if not isinstance(node, ast.Compare):
                continue
            src = ast.unparse(node)
            if "__name__" not in src:
                continue
            if any(f'"{b}"' in src or f"'{b}'" in src for b in ("jax", "torch")):
                offenders.append(f"{path.relative_to(root)}:{node.lineno}: {src}")
    assert not offenders, (
        "modules deriving the backend name from __name__ instead of calling "
        "galpy.backend.name_of_namespace:\n  " + "\n  ".join(offenders)
    )
