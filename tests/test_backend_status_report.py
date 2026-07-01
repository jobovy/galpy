"""Guard tests for the all-backend status report (tests/backend_status_report.py).

The report's ``SHARDS`` list MUST stay in sync with the ``backend-suite`` matrix
in .github/workflows/backend-tests.yml: every shard's uploaded junit artifact has
to map to exactly one table row, otherwise the row renders ``(no result)`` and the
shard's results silently vanish from the summary (this is what happened to the
``potential``/``scf``/``multipole`` rows after test_potential.py was pytest-split).
"""

import importlib.util
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKFLOW = os.path.join(_HERE, os.pardir, ".github", "workflows", "backend-tests.yml")


def _load_report():
    # Load by path so the test is independent of the pytest import mode / sys.path.
    spec = importlib.util.spec_from_file_location(
        "backend_status_report", os.path.join(_HERE, "backend_status_report.py")
    )
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so the module's @dataclass can resolve its own module.
    sys.modules["backend_status_report"] = mod
    spec.loader.exec_module(mod)
    return mod


def _matrix_test_files():
    yaml = pytest.importorskip("yaml")
    if not os.path.exists(_WORKFLOW):
        pytest.skip("backend-tests.yml not found (not a source checkout)")
    with open(_WORKFLOW) as fh:
        wf = yaml.safe_load(fh)
    for job in wf.get("jobs", {}).values():
        matrix = job.get("strategy", {}).get("matrix", {})
        if "TEST_FILES" in matrix:
            return list(matrix["TEST_FILES"])
    pytest.skip("no backend-suite TEST_FILES matrix found")


def test_report_shards_match_workflow_matrix():
    """Every backend-suite matrix shard maps to exactly one report row and every
    report row is exercised by >=1 matrix shard -- guards SHARDS<->matrix drift."""
    report = _load_report()
    matrix = _matrix_test_files()
    row_sids = [report.shard_id(tf) for tf, _ in report.SHARDS]

    # Report rows must have distinct ids (no two rows share an artifact id).
    assert len(row_sids) == len(set(row_sids)), (
        f"duplicate report shard ids: {row_sids}"
    )

    # Every matrix shard's artifact maps to exactly one report row.
    for tf in matrix:
        sid = report.shard_id(tf)
        artifact = f"backend-junit-jax-{sid}.xml"
        hits = [rs for rs in row_sids if report.match_cell(artifact, "jax", rs)]
        assert len(hits) == 1, (
            f"matrix shard {tf!r} (artifact id {sid!r}) maps to {len(hits)} report "
            f"rows {hits}; update SHARDS in tests/backend_status_report.py to match "
            f"the backend-suite matrix"
        )

    # Every report row is exercised by at least one matrix shard (no stale rows).
    matrix_artifacts = [f"backend-junit-jax-{report.shard_id(tf)}.xml" for tf in matrix]
    for (_tf, label), rs in zip(report.SHARDS, row_sids):
        assert any(report.match_cell(a, "jax", rs) for a in matrix_artifacts), (
            f"report row {label!r} (id {rs!r}) matches no backend-suite matrix shard "
            f"-- stale SHARDS entry"
        )
