###############################################################################
# test_backend_input.py: the @backend_input boundary decorator
# (galpy.backend._input).
#
# @backend_input coerces a potential/df evaluator's DECLARED coordinate inputs
# onto the active backend so torch's strict scalar handling accepts them. It is
# the backend counterpart to the legacy units-only potential_physical_input,
# stacked just inside it on the public entry points. Contract:
#   * numpy path (xp is numpy) -> object-identical pass-through (byte-identical);
#   * forced non-numpy backend on a migrated target -> the DECLARED coordinates
#     are coerced whether passed positionally or by keyword, and a coordinate
#     left at a non-None signature default is injected coerced;
#   * everything NOT declared (dR/dphi derivative orders, forceint, M, nsigma,
#     option strings, grid objects) is passed through untouched -- these entry
#     points mix coordinates with control parameters, and coercing a control
#     parameter is either silently wrong or an outright error;
#   * a declared name that is not a parameter raises at DECORATION time, so a
#     typo cannot silently skip a coordinate.
###############################################################################
import numpy
import pytest

from galpy import backend
from galpy.backend import backend_input, is_backend_array

# This module manages backends explicitly, so it is exempt from the global force.
pytestmark = pytest.mark.backend_managed

_NS = {"numpy": numpy}
try:
    import jax

    jax.config.update("jax_enable_x64", True)

    _NS["jax"] = jax
except ImportError:  # pragma: no cover
    pass
try:
    import torch

    _NS["torch"] = torch
except ImportError:  # pragma: no cover
    pass

AD_BACKENDS = [b for b in _NS if b != "numpy"]


def _pot():
    from galpy.potential import MiyamotoNagaiPotential

    return MiyamotoNagaiPotential(normalize=1.0, a=0.5, b=0.05)


def test_numpy_path_is_byte_identical_passthrough():
    # numpy backend: the decorator's non-numpy branch is skipped entirely; the
    # evaluator runs exactly as it would undecorated (byte-identical numpy path).
    mp = _pot()
    assert not is_backend_array(mp.Rforce(1.0, 0.1))
    assert mp.Rforce(1.0, 0.1) == mp.Rforce(1.0, 0.1)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_positional_coords_coerced_under_forced_backend(backend_name):
    # numpy/python positional coords fed under a forced backend must be coerced
    # so the migrated evaluator returns a backend array matching numpy. This also
    # exercises the signature-default injection: Rforce(R, z) leaves phi=0.0 and
    # t=0.0 at their defaults, which the decorator coerces before the call (torch
    # would reject a raw python 0.0).
    mp = _pot()
    ref = mp.Rforce(1.1, 0.2)
    with backend.use(backend_name, force=True):
        got = mp.Rforce(1.1, 0.2)
    assert is_backend_array(got), f"{backend_name}: coords not coerced"
    numpy.testing.assert_allclose(float(got), float(ref), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_coordinate_kwargs_coerced_under_forced_backend(backend_name):
    # The phi/t/R/z/x/v kwarg loop: coords passed by keyword must be coerced too.
    mp = _pot()
    ref = mp.Rforce(R=1.1, z=0.2, phi=0.3)
    with backend.use(backend_name, force=True):
        got = mp.Rforce(R=1.1, z=0.2, phi=0.3)
    assert is_backend_array(got), f"{backend_name}: coord kwargs not coerced"
    numpy.testing.assert_allclose(float(got), float(ref), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_explicitly_passed_phi_t_coerced_under_forced_backend(backend_name):
    # phi/t supplied positionally (so they do NOT hit the signature-default
    # branch) are coerced by the positional loop; result still a backend array.
    mp = _pot()
    ref = mp.Rforce(1.1, 0.2, 0.3, 0.0)
    with backend.use(backend_name, force=True):
        got = mp.Rforce(1.1, 0.2, 0.3, 0.0)
    assert is_backend_array(got)
    numpy.testing.assert_allclose(float(got), float(ref), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_undeclared_control_params_are_not_coerced(backend_name):
    # The point of declaring coordinates explicitly: these entry points mix
    # coordinates with control parameters, and only the coordinates may cross
    # onto the backend. A derivative order turned into a float tensor, or an
    # option string fed to asarray, is silently wrong / an outright error.
    seen = {}

    @backend_input("R", "z")
    def probe(pot, R, z, dR=0, forceint=False, method="rk4"):
        seen.update(R=R, z=z, dR=dR, forceint=forceint, method=method)

    with backend.use(backend_name, force=True):
        probe(_pot(), 1.1, 0.2, 3, True, "rk4_c")  # control params POSITIONAL
    assert is_backend_array(seen["R"]) and is_backend_array(seen["z"])
    assert seen["dR"] == 3 and isinstance(seen["dR"], int)
    assert seen["forceint"] is True
    assert seen["method"] == "rk4_c"


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_control_params_untouched_through_a_real_entry_point(backend_name):
    # Integration: Potential.mass declares ("R", "z", "t") and leaves the
    # trailing forceint flag alone, so passing it positionally still selects the
    # analytic branch (a coerced flag would be an array, not a bool).
    mp = _pot()
    ref = mp.mass(1.1, 0.2, 0.0, False)
    with backend.use(backend_name, force=True):
        got = mp.mass(1.1, 0.2, 0.0, False)
    assert is_backend_array(got)
    numpy.testing.assert_allclose(float(got), float(ref), rtol=1e-12, atol=1e-14)


def test_unknown_declared_coordinate_raises_at_decoration():
    # A typo'd coordinate name must fail loudly at import, not silently skip
    # coercing that coordinate.
    with pytest.raises(ValueError, match="not parameters"):

        @backend_input("R", "notaparameter")
        def _bad(pot, R, z):  # pragma: no cover - never called
            return R


def test_bare_usage_without_coordinates_raises():
    # @backend_input must be called with names; bare usage would coerce nothing.
    with pytest.raises(TypeError, match="coordinate names"):

        @backend_input
        def _bad(pot, R, z):  # pragma: no cover - never called
            return R


@pytest.mark.parametrize("backend_name", AD_BACKENDS + ["numpy"])
def test_sequence_valued_coordinate_as_kwarg(backend_name):
    # The dissipative forces take v=[vR,vT,vz] as a LIST. get_namespace
    # dispatches on arrays and rejects a bare list, so the decorator must probe
    # the components -- otherwise passing v by keyword raises
    # "list is not a supported array type" even on the numpy path.
    from galpy.potential import ChandrasekharDynamicalFrictionForce

    cdf = ChandrasekharDynamicalFrictionForce(GMs=0.01, rhm=0.1)
    ref = cdf.Rforce(1.1, 0.2, phi=0.0, t=0.0, v=[0.1, 1.0, 0.05])
    if backend_name == "numpy":
        assert not is_backend_array(ref)
        return
    with backend.use(backend_name, force=True):
        got = cdf.Rforce(1.1, 0.2, phi=0.0, t=0.0, v=[0.1, 1.0, 0.05])
    numpy.testing.assert_allclose(float(got), float(ref), rtol=1e-10, atol=1e-14)


def _asarray(backend_name, x):
    if backend_name == "jax":
        import jax.numpy

        return jax.numpy.asarray(x, dtype=jax.numpy.float64)
    import torch

    return torch.as_tensor(x, dtype=torch.float64)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_mixed_backend_and_numpy_coordinates(backend_name):
    # Data-driven dispatch (NO forced backend -- forcing short-circuits the
    # namespace probe): Orbits.E calls evaluatePotentials with backend R/z but a
    # NUMPY t=. Probing that mix raises "Multiple namespaces for array inputs",
    # so the namespace must come from the backend coordinates while the numpy
    # ones are coerced across.
    mp = _pot()
    R = _asarray(backend_name, [1.1, 1.2])
    z = _asarray(backend_name, [0.2, 0.1])
    got = mp.Rforce(R, z, t=numpy.zeros(2))
    assert is_backend_array(got), (
        f"{backend_name}: mixed coords did not stay on backend"
    )
    ref = [float(mp.Rforce(1.1, 0.2)), float(mp.Rforce(1.2, 0.1))]
    numpy.testing.assert_allclose(numpy.asarray(got, dtype=float), ref, rtol=1e-12)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_sequence_coordinate_is_coerced_element_wise(backend_name):
    # A sequence coordinate keeps its container and is coerced ELEMENT-WISE.
    # Stacking it into a single array with one asarray detaches grad-carrying
    # elements from the autograd graph -- that is what broke the
    # NonInertialFrameForce grad-vs-finite-difference tests.
    seen = {}

    @backend_input("R", "v")
    def probe(pot, R, v=None):
        seen.update(R=R, v=v)

    with backend.use(backend_name, force=True):
        probe(_pot(), 1.1, v=[0.1, 1.0, 0.05])
    assert is_backend_array(seen["R"])
    assert isinstance(seen["v"], list), "sequence coordinate must stay a sequence"
    assert len(seen["v"]) == 3
    assert all(is_backend_array(c) for c in seen["v"])


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_quantity_coordinate_passes_through(backend_name):
    # @backend_input is only correct INSIDE something that has already stripped
    # units. Some entry points strip them in the body instead -- sphericaldf.sigmar
    # does `r = conversion.parse_length(r, ro=self._ro)` -- so the boundary can be
    # handed a Quantity. asarray() of a Quantity yields garbage that the later
    # parse turns into NaN, so it must pass through untouched. Uses a stub rather
    # than astropy: the coverage shard runs without it.
    class _Quantityish:
        unit = "kpc"
        value = 1.1

    q = _Quantityish()
    seen = {}

    @backend_input("R", "z")
    def probe(pot, R, z):
        seen.update(R=R, z=z)

    with backend.use(backend_name, force=True):
        probe(_pot(), q, 0.2)
    assert seen["R"] is q, "a Quantity-like coordinate must not be coerced"
    assert is_backend_array(seen["z"]), "plain coordinates alongside it still coerce"


# Coordinate parameter names used across galpy's entry points. A decorated entry
# point must declare every one of these that it takes -- an undeclared
# coordinate is silently NOT coerced, which the decoration-time check (which only
# sees the names that WERE declared) structurally cannot catch.
_COORD_NAMES = frozenset(("R", "z", "phi", "t", "x", "r", "v", "vR", "vT", "vz"))


def _decorated_entry_points():
    """Every @backend_input site in the installed galpy, via a static AST walk."""
    import ast
    import pathlib

    import galpy

    for path in sorted(pathlib.Path(galpy.__file__).parent.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover - galpy always parses
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for dec in node.decorator_list:
                if (
                    isinstance(dec, ast.Call)
                    and isinstance(dec.func, ast.Name)
                    and dec.func.id == "backend_input"
                ):
                    declared = [
                        a.value for a in dec.args if isinstance(a, ast.Constant)
                    ]
                    named = [a.arg for a in node.args.posonlyargs + node.args.args][1:]
                    yield path.name, node.name, declared, named, node.args.kwarg


def test_every_declared_coordinate_exists_in_the_signature():
    # Typos: a declared name that is not a parameter would coerce nothing. Only
    # legal when the entry point takes **kwargs (Force.rforce forwards phi/t).
    bad = [
        f"{f}::{fn} declares {sorted(set(dec) - set(named))}"
        for f, fn, dec, named, kwarg in _decorated_entry_points()
        if not set(dec) <= set(named) and kwarg is None
    ]
    assert not bad, "declared coordinates that are not parameters:\n  " + "\n  ".join(
        bad
    )


def test_no_coordinate_parameter_is_left_undeclared():
    # Wrong application: a coordinate the entry point takes but does not declare
    # is silently left on numpy while its siblings move to the backend.
    bad = [
        f"{f}::{fn} takes {sorted(set(named) & _COORD_NAMES - set(dec))} but declares {dec}"
        for f, fn, dec, named, _ in _decorated_entry_points()
        if (set(named) & _COORD_NAMES) - set(dec)
    ]
    assert not bad, "undeclared coordinate parameters:\n  " + "\n  ".join(bad)


def test_decorated_entry_points_are_registered():
    # Guard against the audit silently finding nothing (a refactor renaming the
    # decorator would make both checks above vacuously pass).
    assert len(list(_decorated_entry_points())) > 50


# Whole modules whose public entry points deliberately do NOT coerce yet, with
# the blocker for each. These are the burndown list: decorate them as the blocker
# clears -- sphericaldf/kingdf came off it once sphericaldf opted in
# (``_backend_compatible``) and its radius entry points declared their coordinate. (Under the old coerce_backend gate these modules could carry the
# decorator harmlessly because _check_backend_compatible returned False for every
# df, so it never fired; with the gate gone, decorating them would route real
# calls onto an incomplete backend path.)
_UNDECORATED_MODULES = {
    # evolveddiskdf raises NotImplementedError for grid deriv= on jax/torch;
    # quasiisothermaldf's sampling and adiabatic action-Jacobian path take 0-d
    # backend arrays through a C callback that expects sized numpy input;
    # sphericaldf.sigmar returns NaN for a coerced radius (kingdf).
    "diskdf.py": "backend evaluation path incomplete",
    "evolveddiskdf.py": "grid deriv= unsupported on jax/torch",
    "quasiisothermaldf.py": "sampling + adiabatic action-Jacobian C callback",
    "streamdf.py": "not yet migrated to the coercion boundary",
    "streamgapdf.py": "not yet migrated to the coercion boundary",
    "streamspraydf.py": "not yet migrated to the coercion boundary",
}

# Individual public coordinate-taking entry points that deliberately do NOT
# coerce, with the reason each one is exempt. Everything else that takes a
# coordinate must carry @backend_input -- these two lists are what keep the
# footprint auditable instead of historical, so adding to them should be a
# deliberate, reviewed act.
_UNDECORATED_BY_DESIGN = {
    # Numpy-only computation ON the coordinates: they consume a potential's
    # evaluator output and do their own numpy/scipy work, so coercing their
    # inputs would only force a round trip (the old coerce_backend=False opt-outs).
    ("jeans.py", "sigmar"),
    ("jeans.py", "sigmalos"),
    ("actionAngleStaeckel.py", "estimateDeltaStaeckel"),
    ("actionAngleIsochroneApprox.py", "estimateBIsochrone"),
    # scipy-interpolation island; migrating it is tracked separately.
    ("interpRZPotential.py", "vcirc"),
    ("interpRZPotential.py", "dvcircdR"),
    ("interpRZPotential.py", "epifreq"),
    ("interpRZPotential.py", "verticalfreq"),
    # scipy root-finders / quadrature over E, Lz, OmegaP, l: the coordinate-named
    # parameter is an evaluation time, and the solve itself is numpy.
    ("Potential.py", "rl"),
    ("Potential.py", "rE"),
    ("Potential.py", "LcE"),
    ("Potential.py", "lindbladR"),
    ("Potential.py", "vterm"),
    ("Potential.py", "rhalf"),
    ("Potential.py", "mvir"),
    ("Potential.py", "zvc"),
    ("Potential.py", "zvc_range"),
    ("planarPotential.py", "lindbladR"),
}


def test_coordinate_entry_points_are_decorated_or_listed():
    """Every public coordinate-taking entry point is decorated or exempt-by-name.

    Keeps the decorated footprint principled rather than historical: a new public
    method that takes R/z/phi/t must either coerce or be added to
    _UNDECORATED_BY_DESIGN with a reason.
    """
    import ast
    import pathlib

    import galpy

    stray = []
    for path in sorted(pathlib.Path(galpy.__file__).parent.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover - galpy always parses
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
                continue
            names = {
                (
                    d.func.id
                    if isinstance(d, ast.Call) and isinstance(d.func, ast.Name)
                    else getattr(d, "id", None)
                )
                for d in node.decorator_list
            }
            if "physical_conversion" not in names or "backend_input" in names:
                continue
            named = [a.arg for a in node.args.posonlyargs + node.args.args][1:]
            if (
                set(named) & _COORD_NAMES
                and (path.name, node.name) not in _UNDECORATED_BY_DESIGN
                and path.name not in _UNDECORATED_MODULES
            ):
                stray.append(f"{path.name}::{node.name}({', '.join(named)})")
    assert not stray, (
        "public coordinate-taking entry points that neither coerce nor are listed "
        "in _UNDECORATED_BY_DESIGN / _UNDECORATED_MODULES:\n  "
        + "\n  ".join(sorted(set(stray)))
    )
