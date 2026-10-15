# Tests for the backend-list machinery in conftest.py itself (xfail ledger,
# slow-skip, permanent skip). The "-jit inherits eager" rule below is otherwise
# reachable only from a traced (--jit) dispatch, which does not run on every
# push, so it would land unexercised.
import os

import conftest
import pytest


def _ledger(tmp_path, body):
    path = tmp_path / "ledger.txt"
    path.write_text(body)
    return str(path)


def test_backend_nodeids_selects_one_backend(tmp_path):
    """The base case: a plain backend name reads only its own lines."""
    path = _ledger(
        tmp_path,
        "# a comment line\n"
        "\n"
        "jax tests/test_a.py::test_one\n"
        "jax tests/test_a.py::test_two  # trailing comment\n"
        "torch tests/test_a.py::test_three\n",
    )
    assert conftest._load_backend_nodeids(path, "jax") == {
        "tests/test_a.py::test_one",
        "tests/test_a.py::test_two",
    }
    assert conftest._load_backend_nodeids(path, "torch") == {
        "tests/test_a.py::test_three"
    }


def test_jit_inherits_eager(tmp_path):
    """A traced backend reads its own lines UNION the eager backend's lines."""
    path = _ledger(
        tmp_path,
        "jax tests/test_a.py::test_eager_only\n"
        "jax-jit tests/test_a.py::test_traced_only\n"
        "torch tests/test_a.py::test_other_backend\n"
        "torch-jit tests/test_a.py::test_other_backend_traced\n",
    )
    assert conftest._load_backend_nodeids(path, "jax-jit") == {
        "tests/test_a.py::test_eager_only",
        "tests/test_a.py::test_traced_only",
    }
    # Inheritance is one-directional and per-backend: the eager list never gains
    # traced entries, and jax-jit never picks up anything torch.
    assert conftest._load_backend_nodeids(path, "jax") == {
        "tests/test_a.py::test_eager_only"
    }
    assert conftest._load_backend_nodeids(path, "torch-jit") == {
        "tests/test_a.py::test_other_backend",
        "tests/test_a.py::test_other_backend_traced",
    }


def test_jit_inheritance_is_per_list_not_per_backend(tmp_path, monkeypatch):
    """FAILURE lists inherit eager entries; the SLOWNESS list does not.

    The rule used to be "every list inherits", justified by "a test too slow to
    run eager is slower still traced (it must compile first)". That is false,
    and measurably so: the dominant eager cost in this suite is per-call
    dispatch, which is exactly what tracing removes. test_streamdf.py is
    slow-skipped under jax because it takes >90 min eager (the shard cancels);
    traced it is 13 min for the whole file, with a 0.11 s median test.

    So inheritance is a property of WHICH LIST is being read:

      xfail ledger    inherit -- tracing cannot repair a numerical gap, a
                                 `_reject_backend` guard, or an out-of-scope family
      permanent skip  inherit -- a test excluded for hitting the network is
                                 excluded however it is run
      slow-skip       DO NOT  -- tracing changes runtime in BOTH directions, so a
                                 traced run must defer only what is measured slow
                                 WHEN TRACED
    """
    path = _ledger(tmp_path, "jax tests/test_a.py::test_eager_only\n")
    for attr in ("_ledger_path", "_slow_skip_path", "_backend_skip_path"):
        monkeypatch.setattr(conftest, attr, lambda path=path: path)
    for loader in (conftest._load_ledger, conftest._load_backend_skip):
        assert loader("jax-jit") == {"tests/test_a.py::test_eager_only"}, loader
    # The whole point of the split: an eager slow-skip entry does NOT silence
    # the traced run, so tests that tracing makes fast start running again.
    assert conftest._load_slow_skip("jax-jit") == set()
    # ... while a traced-only entry still applies, and the eager list is
    # unaffected in both directions.
    jit_path = _ledger(tmp_path, "jax-jit tests/test_a.py::test_traced_only\n")
    monkeypatch.setattr(conftest, "_slow_skip_path", lambda: jit_path)
    assert conftest._load_slow_skip("jax-jit") == {"tests/test_a.py::test_traced_only"}
    assert conftest._load_slow_skip("jax") == set()


def test_regen_of_traced_run_drops_inherited_eager_entries(tmp_path, monkeypatch):
    """Regen for "<backend>-jit" writes only what fails ONLY when traced.

    Regen xfails nothing, so the traced run also reports every eager failure;
    writing those back out would duplicate the whole eager list.
    """
    path = _ledger(
        tmp_path,
        "jax tests/test_a.py::test_eager\n"
        "jax tests/test_b.py::test_parametrized\n",  # base entry, covers all params
    )
    monkeypatch.setattr(conftest, "_ledger_path", lambda: path)
    observed = [
        "tests/test_a.py::test_eager",  # inherited -> dropped
        "tests/test_b.py::test_parametrized[Case1]",  # covered by base -> dropped
        "tests/test_c.py::test_traced_only",  # genuinely new -> kept
    ]
    assert conftest._regen_entries("jax-jit", observed) == [
        "tests/test_c.py::test_traced_only"
    ]
    # An eager regen is untouched -- it has nothing to inherit from.
    assert conftest._regen_entries("jax", observed) == sorted(observed)


def test_missing_file_is_empty(tmp_path):
    """A backend with no list at all is not an error."""
    assert conftest._load_backend_nodeids(str(tmp_path / "nope.txt"), "jax") == set()


# --- per-process TORCHINDUCTOR_CACHE_DIR cleanup ----------------------------
def _conftest_child(tmp_path, body):
    """Run ``body`` in a fresh interpreter that imports conftest with
    TMPDIR=tmp_path; returns the completed process."""
    import os
    import subprocess
    import sys

    env = dict(os.environ, TMPDIR=str(tmp_path))
    code = (
        "import sys; sys.path.insert(0, %r); import conftest\n"
        % os.path.dirname(conftest.__file__)
        + body
    )
    return subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )


def _inductor_dirs(tmp_path):
    return sorted(
        p.name for p in tmp_path.iterdir() if p.name.startswith("torchinductor_galpy_")
    )


def test_inductor_cache_is_pid_tagged_and_private():
    name = os.path.basename(conftest._inductor_cache)
    assert name.startswith(f"torchinductor_galpy_{os.getpid()}_")
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == conftest._inductor_cache


def test_inductor_cache_removed_at_normal_exit(tmp_path):
    proc = _conftest_child(tmp_path, "print(conftest._inductor_cache)")
    assert proc.returncode == 0, proc.stderr
    assert _inductor_dirs(tmp_path) == []


def test_inductor_cache_removed_at_force_exit(tmp_path):
    # _backend_force_exit's os._exit skips atexit: it removes the dir itself.
    proc = _conftest_child(tmp_path, "conftest._backend_force_exit(3)")
    assert proc.returncode == 3, proc.stderr
    assert _inductor_dirs(tmp_path) == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX signals/PIDs")
def test_inductor_cache_of_killed_process_swept(tmp_path):
    # SIGKILL skips every cleanup; the next session sweeps the dead PID's dir.
    import signal

    proc = _conftest_child(
        tmp_path, "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"
    )
    assert proc.returncode == -signal.SIGKILL
    leaked = _inductor_dirs(tmp_path)
    assert len(leaked) == 1
    live = tmp_path / f"torchinductor_galpy_{os.getpid()}_live"
    untagged = tmp_path / "torchinductor_galpy_abc123"
    other = tmp_path / "unrelated_1_dir"
    for d in (live, untagged, other):
        d.mkdir()
    conftest._sweep_dead_inductor_caches(str(tmp_path))
    assert _inductor_dirs(tmp_path) == sorted([live.name, untagged.name])
    assert other.exists()
