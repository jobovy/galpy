#!/usr/bin/env python
"""Render an all-backend (jax / torch) per-file test-status markdown table.

This drives the `report` job of .github/workflows/backend-tests.yml. Each
backend-suite shard (BACKEND x TEST_FILES) uploads its pytest ``--junitxml``
output as an artifact; the report job downloads them all into one directory and
runs this script over that directory to produce ``status.md``.

Usage
-----
    python tests/backend_status_report.py <junit-dir> [out.md]

``<junit-dir>`` is searched recursively for ``*.xml`` files (download-artifact@v4
unpacks each artifact into its own sub-directory). ``out.md`` defaults to
``status.md`` in the current directory.

Artifact / file naming contract
--------------------------------
The workflow points each shard's junit at a file whose name encodes the backend
and a *sanitized shard id* derived from the matrix ``TEST_FILES`` string via
``shard_id()`` below, e.g.::

    backend-junit-jax-test_actionAngle.xml
    backend-junit-torch-test_orbit_subset_main.xml

This script independently computes the SAME sanitized ids from its own canonical
copy of the 13 shards (``SHARDS`` below -- kept in sync with the workflow
matrix), so it can map each XML back to a (backend, shard) cell and, crucially,
show a shard that produced NO xml as "-- (no result)" instead of crashing.

JUnit parsing
-------------
pytest's junitxml encodes outcomes as:
  * pass          -> <testcase> with no child <failure>/<error>/<skipped>
  * xfail         -> <testcase><skipped message="...xfail...">
  * plain skip    -> <testcase><skipped message="..."> (no "xfail")
  * fail          -> <testcase><failure>
  * error         -> <testcase><error>
  * strict XPASS  -> <testcase><failure message="...XPASS(strict)...">  (a now
                     -passing ledgered test; must be removed from the ledger ->
                     counted separately as "XPASS(fix-me)", NOT as a plain fail)
"""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass

BACKENDS = ["jax", "torch"]

# Canonical list of the 13 TEST_FILES shards, in matrix order, each with a
# short human-readable label for the table rows. MUST stay in sync with the
# backend-suite matrix in .github/workflows/backend-tests.yml -- the raw string
# is what the workflow feeds to shard_id() to name the junit file, and the label
# is what we print in the row header.
SHARDS = [
    ("tests/test_actionAngle.py", "actionAngle"),
    ("tests/test_sphericaldf.py", "sphericaldf"),
    (
        "tests/test_actionAngleTorus.py tests/test_conversion.py "
        "tests/test_galpypaper.py tests/test_import.py "
        "tests/test_interp_potential.py tests/test_kuzminkutuzov.py  "
        "tests/test_util.py",
        "actionAngleTorus + conversion + misc",
    ),
    (
        "tests/test_SpiralArmsPotential.py tests/test_potential.py "
        "tests/test_scf.py tests/test_MultipoleExpansionPotential.py "
        "tests/test_snapshotpotential.py",
        "potential + scf + multipole",
    ),
    ("tests/test_quantity.py tests/test_coords.py", "quantity + coords"),
    (
        "tests/test_orbit.py -k 'test_energy_jacobi_conservation or from_name'",
        "orbit (energy/Jacobi + from_name)",
    ),
    (
        "tests/test_orbit.py tests/test_orbits.py "
        "-k 'not test_energy_jacobi_conservation'",
        "orbit + orbits (main)",
    ),
    ("tests/test_evolveddiskdf.py", "evolveddiskdf"),
    (
        "tests/test_jeans.py tests/test_dynamfric.py tests/test_FDMdynamfric.py",
        "jeans + dynamfric",
    ),
    (
        "tests/test_qdf.py tests/test_pv2qdf.py "
        "tests/test_streamgapdf_impulse.py tests/test_noninertial.py",
        "qdf + pv2qdf + streamgapdf_impulse + noninertial",
    ),
    ("tests/test_streamgapdf.py", "streamgapdf"),
    ("tests/test_diskdf.py", "diskdf"),
    (
        "tests/test_streamdf.py tests/test_streamspraydf.py tests/test_streamTrack.py",
        "streamdf + streamspraydf + streamTrack",
    ),
]


def shard_id(test_files: str) -> str:
    """Sanitize a TEST_FILES matrix string into a stable artifact-safe id.

    Deterministic and collision-free across the 13 shards: take the basenames of
    the .py files in order (dropping the ``tests/`` prefix and ``.py`` suffix),
    join the first few with ``_``, and append a short tag for any ``-k`` subset
    so the two test_orbit.py shards get distinct ids. The result contains only
    [A-Za-z0-9_], safe for an actions/upload-artifact@v4 name and a filename.
    """
    toks = test_files.split()
    pyfiles = [t for t in toks if t.endswith(".py")]
    stems = [os.path.basename(p)[: -len(".py")] for p in pyfiles]
    base = "_".join(stems[:3]) if stems else "shard"
    # Disambiguate the two test_orbit.py shards via their -k expression.
    if "-k" in toks:
        if "from_name" in test_files:
            base += "_energy_fromname"
        elif "not test_energy_jacobi_conservation" in test_files:
            base += "_main"
    # Keep it short-ish but unique.
    out = "".join(c if (c.isalnum() or c == "_") else "_" for c in base)
    return out


@dataclass
class Counts:
    passed: int = 0
    xfailed: int = 0
    xpassed: int = 0  # strict XPASS -> "fix-me" (ledger must shrink)
    failed: int = 0
    errored: int = 0
    skipped: int = 0
    found: bool = False  # was an xml present/parseable for this cell?
    parse_error: str = ""

    @property
    def total(self) -> int:
        return (
            self.passed
            + self.xfailed
            + self.xpassed
            + self.failed
            + self.errored
            + self.skipped
        )

    @property
    def is_bad(self) -> bool:
        return bool(self.failed or self.errored or self.xpassed)


def parse_junit(path: str) -> Counts:
    """Parse one junit xml file into a Counts. Robust to malformed/empty files."""
    c = Counts(found=True)
    try:
        root = ET.parse(path).getroot()
    except Exception as e:  # truncated/empty xml from a crashed shard
        c.found = False
        c.parse_error = str(e)
        return c
    for tc in root.iter("testcase"):
        fail = tc.find("failure")
        err = tc.find("error")
        skip = tc.find("skipped")
        if fail is not None:
            msg = (fail.get("message") or "") + (fail.text or "")
            # strict XPASS: a ledgered test that now passes (must be de-ledgered)
            if "xpass" in msg.lower():
                c.xpassed += 1
            else:
                c.failed += 1
            continue
        if err is not None:
            c.errored += 1
            continue
        if skip is not None:
            msg = (skip.get("message") or "") + (skip.text or "")
            if "xfail" in msg.lower():
                c.xfailed += 1
            else:
                c.skipped += 1
            continue
        c.passed += 1
    return c


def find_xml_files(junit_dir: str) -> list[str]:
    out = []
    for dirpath, _dirnames, filenames in os.walk(junit_dir):
        for fn in filenames:
            if fn.endswith(".xml"):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


def match_cell(filename: str, backend: str, sid: str) -> bool:
    """Does this junit filename belong to (backend, shard-id)?

    The workflow names files ``backend-junit-<backend>-<sid>.xml``; download-
    artifact@v4 may nest them under a per-artifact directory. Match on the
    basename containing both the backend token and the sid token, anchored so
    e.g. sid 'test_orbit_test_orbits_main' is not matched by the energy shard.
    """
    base = os.path.basename(filename)
    if base.endswith(".xml"):
        base = base[: -len(".xml")]
    return base == f"backend-junit-{backend}-{sid}"


def ledger_size(ledger_path: str) -> tuple[int, dict[str, int]]:
    """Total ledger entries and per-backend counts (best-effort)."""
    per = {b: 0 for b in BACKENDS}
    total = 0
    if not os.path.exists(ledger_path):
        return 0, per
    with open(ledger_path) as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            total += 1
            be = parts[0].strip()
            if be in per:
                per[be] += 1
    return total, per


def glyph(c: Counts) -> str:
    if not c.found:
        return "—"
    if c.failed or c.errored:
        return "❌"
    if c.xpassed:
        return "⚠️"
    return "✅"


def cell_text(c: Counts) -> str:
    if not c.found:
        return "— (no result)"
    parts = [f"{c.passed} pass", f"{c.xfailed} xfail"]
    if c.failed or c.errored:
        parts.append(f"{c.failed + c.errored} FAIL/ERR")
    if c.xpassed:
        parts.append(f"{c.xpassed} XPASS(fix-me)")
    body = " · ".join(parts)
    return f"{glyph(c)} {body}"


def render(junit_dir: str, ledger_path: str, sha: str) -> str:
    xmls = find_xml_files(junit_dir)
    # cells[(shard_index, backend)] -> Counts
    cells: dict[tuple[int, str], Counts] = {}
    for idx, (test_files, _label) in enumerate(SHARDS):
        sid = shard_id(test_files)
        for backend in BACKENDS:
            matched = [f for f in xmls if match_cell(f, backend, sid)]
            if not matched:
                cells[(idx, backend)] = Counts(found=False)
            else:
                # If a shard somehow uploaded multiple xmls, merge them.
                merged = Counts(found=True)
                for f in matched:
                    c = parse_junit(f)
                    if not c.found:
                        # parse failure on one file; keep it as not-found unless
                        # another file for the cell parsed.
                        continue
                    merged.passed += c.passed
                    merged.xfailed += c.xfailed
                    merged.xpassed += c.xpassed
                    merged.failed += c.failed
                    merged.errored += c.errored
                    merged.skipped += c.skipped
                if merged.total == 0 and all(not parse_junit(f).found for f in matched):
                    merged.found = False
                cells[(idx, backend)] = merged

    # Overall per-backend totals (the burndown headline).
    totals = {b: Counts(found=True) for b in BACKENDS}
    for (idx, backend), c in cells.items():
        if not c.found:
            continue
        t = totals[backend]
        t.passed += c.passed
        t.xfailed += c.xfailed
        t.xpassed += c.xpassed
        t.failed += c.failed
        t.errored += c.errored
        t.skipped += c.skipped

    led_total, led_per = ledger_size(ledger_path)

    lines: list[str] = []
    lines.append("## All-backend test status (jax / torch)")
    lines.append("")
    if sha:
        lines.append(f"Commit `{sha}`")
        lines.append("")
    lines.append(
        "Green is achieved via the checked-in xfail-ledger "
        "(`tests/backend_xfail.txt`), so the metric to watch is the "
        "**shrinking xfail count** (burndown), not a raw pass count. A "
        "`XPASS(fix-me)` means a ledgered test now passes and must be removed "
        "from the ledger; a `FAIL/ERR` is an un-ledgered regression."
    )
    lines.append("")

    # Headline totals line.
    head = []
    for b in BACKENDS:
        t = totals[b]
        seg = f"**{b}**: {t.passed} passed · {t.xfailed} xfail"
        if t.xpassed:
            seg += f" · {t.xpassed} XPASS(fix-me)"
        if t.failed or t.errored:
            seg += f" · {t.failed + t.errored} FAIL/ERR"
        head.append(seg)
    lines.append("**Overall:** " + "  |  ".join(head))
    lines.append("")
    led_detail = ", ".join(f"{b}={led_per[b]}" for b in BACKENDS)
    lines.append(f"**Ledger size:** {led_total} entries ({led_detail}).")
    lines.append("")

    # Per-file x backend table.
    lines.append("| Test shard | jax | torch |")
    lines.append("| --- | --- | --- |")
    for idx, (_test_files, label) in enumerate(SHARDS):
        row = [label]
        for backend in BACKENDS:
            row.append(cell_text(cells[(idx, backend)]))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Per-shard raw counts (collapsible detail).
    lines.append("<details><summary>Per-shard counts</summary>")
    lines.append("")
    lines.append("| Test shard | backend | pass | xfail | XPASS | fail | error |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for idx, (_test_files, label) in enumerate(SHARDS):
        for backend in BACKENDS:
            c = cells[(idx, backend)]
            if not c.found:
                lines.append(f"| {label} | {backend} | — | — | — | — | — |")
            else:
                lines.append(
                    f"| {label} | {backend} | {c.passed} | {c.xfailed} | "
                    f"{c.xpassed} | {c.failed} | {c.errored} |"
                )
    lines.append("")
    lines.append("</details>")
    lines.append("")

    return "\n".join(lines)


def main(argv: list[str]) -> int:
    # Helper mode for the workflow: print the sanitized shard id for the
    # TEST_FILES string in the env var TEST_FILES (avoids fragile shell quoting
    # of the embedded -k '...' expression). Single source of truth for the id.
    if len(argv) >= 2 and argv[1] == "--id":
        sys.stdout.write(shard_id(os.environ.get("TEST_FILES", "")))
        return 0
    if len(argv) < 2:
        sys.stderr.write(
            "usage: backend_status_report.py <junit-dir> [out.md]\n"
            "       backend_status_report.py --id  "
            "(prints shard_id of $TEST_FILES)\n"
        )
        return 2
    junit_dir = argv[1]
    out_path = argv[2] if len(argv) > 2 else "status.md"
    ledger_path = os.environ.get(
        "GALPY_BACKEND_LEDGER",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend_xfail.txt"),
    )
    sha = os.environ.get("GITHUB_SHA", "")
    md = render(junit_dir, ledger_path, sha)
    with open(out_path, "w") as fh:
        fh.write(md)
    sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
