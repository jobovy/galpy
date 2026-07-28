# Tests for the backend-list machinery in conftest.py itself (xfail ledger,
# slow-skip, permanent skip). The "-jit inherits eager" rule below is otherwise
# reachable only from a traced (--jit) dispatch, which does not run on every
# push, so it would land unexercised.
import conftest


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


def test_jit_inheritance_applies_to_every_list(tmp_path, monkeypatch):
    """Ledger, slow-skip and permanent-skip all inherit -- one shared rule.

    A test too slow to run eager is slower still traced (it must compile first),
    and a test excluded for hitting the network is excluded however it is run.
    """
    path = _ledger(tmp_path, "jax tests/test_a.py::test_eager_only\n")
    for attr in ("_ledger_path", "_slow_skip_path", "_backend_skip_path"):
        monkeypatch.setattr(conftest, attr, lambda path=path: path)
    for loader in (
        conftest._load_ledger,
        conftest._load_slow_skip,
        conftest._load_backend_skip,
    ):
        assert loader("jax-jit") == {"tests/test_a.py::test_eager_only"}, loader


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
