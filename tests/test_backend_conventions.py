###############################################################################
# test_backend_conventions.py: enforce the backend namespace-swap conventions on
# migrated potentials, so a migrated potential can never regress to bare numpy in
# its compute methods. The allowlist grows PR by PR as potentials are migrated.
###############################################################################
import ast
import inspect

import pytest

from galpy import potential

# Static source analysis — independent of the active backend.
pytestmark = pytest.mark.backend_managed

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
