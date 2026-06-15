###############################################################################
# test_backend_conventions.py: enforce the backend namespace-swap conventions on
# migrated potentials, so a migrated potential can never regress to bare numpy in
# its compute methods. The allowlist grows PR by PR as potentials are migrated.
###############################################################################
import ast
import inspect

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

# Private compute methods that must be backend-agnostic (no bare ``numpy.<fn>``;
# use ``xp = get_namespace(...)`` then ``xp.<fn>``, or ``math.pi`` for constants).
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

# Potentials migrated to the backend convention (extend as PRs land).
MIGRATED = [
    "PlummerPotential",
    "IsochronePotential",
    # P2.6 wrappers (their __init__ machinery is numpy-by-design; only the
    # private compute methods below are checked)
    "DehnenSmoothWrapperPotential",
    "GaussianAmplitudeWrapperPotential",
    "TimeDependentAmplitudeWrapperPotential",
    "SolidBodyRotationWrapperPotential",
    "CorotatingRotationWrapperPotential",
    "RotateAndTiltWrapperPotential",
    "KuzminLikeWrapperPotential",
    "OblateStaeckelWrapperPotential",
    "CylindricallySeparablePotentialWrapper",
]


class _BareNumpyVisitor(ast.NodeVisitor):
    def __init__(self):
        self.violations = []

    def visit_FunctionDef(self, node):
        if node.name in COMPUTE_METHODS:
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Attribute)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id == "numpy"
                ):
                    self.violations.append((node.name, f"numpy.{sub.attr}", sub.lineno))
        self.generic_visit(node)


@pytest.mark.parametrize("clsname", MIGRATED)
def test_no_bare_numpy_in_compute_methods(clsname):
    cls = getattr(potential, clsname)
    module = inspect.getmodule(cls)
    tree = ast.parse(inspect.getsource(module))
    visitor = _BareNumpyVisitor()
    visitor.visit(tree)
    assert not visitor.violations, (
        f"{clsname}: bare numpy.* in compute methods (use xp=get_namespace(...) "
        f"or math.pi): {visitor.violations}"
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
]
_UNMIGRATED_SAMPLE = [
    "FerrersPotential",
    "KuzminKutuzovStaeckelPotential",
    "PseudoIsothermalPotential",
]


@pytest.mark.parametrize("clsname", _MIGRATED_SAMPLE)
def test_backend_compatible_true(clsname):
    from galpy.potential import _check_backend_compatible as cbc

    assert cbc(getattr(potential, clsname)(normalize=1.0)) is True


@pytest.mark.parametrize("clsname", _UNMIGRATED_SAMPLE)
def test_backend_compatible_false(clsname):
    from galpy.potential import _check_backend_compatible as cbc

    assert cbc(getattr(potential, clsname)(normalize=1.0)) is False


def test_check_backend_compatible_semantics():
    from galpy.potential import _check_backend_compatible as cbc

    mn = potential.MiyamotoNagaiPotential(normalize=1.0)
    fe = potential.FerrersPotential(normalize=1.0)
    pit = potential.PseudoIsothermalPotential(normalize=1.0)
    # combined potential: all members must be compatible
    assert cbc([mn, potential.NFWPotential(normalize=1.0)]) is True
    assert cbc([mn, fe]) is False
    # wrapper: own flag AND wrapped potential
    assert cbc(potential.OblateStaeckelWrapperPotential(pot=mn)) is True
    assert (
        cbc(potential.KuzminLikeWrapperPotential(amp=1.0, pot=pit, a=1.0, b=0.2))
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
    from galpy.backend._namespaces import coerce_coords

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
