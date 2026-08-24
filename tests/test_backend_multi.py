###############################################################################
# test_backend_multi.py: fork-safety of galpy.util.multi.parallel_map under a
# multi-threaded array backend.
#
# parallel_map forks (spawn cannot pickle the mapped closures, #457). torch's
# intra-op thread pool does not survive fork: the child inherits pool state it
# cannot use and wedges on its first parallel region, so the parent hangs in
# proc.join() forever. galpy caps each forked child at one torch thread, which
# both avoids the deadlock and is the correct split -- one process per core
# already saturates the machine.
#
# The deadlock is a HANG, not an exception, so the tests here are written to
# fail fast and deterministically: they assert the mechanism (the child's thread
# count) rather than waiting on a wedged join. The one test that does exercise a
# real parallel region in the child is the direct regression for the hang.
###############################################################################
import numpy
import pytest

pytestmark = [
    pytest.mark.backend_managed,
    # Python warns at every fork from a multi-threaded process, and the
    # tests/test_backend*.py shard runs under -W error::DeprecationWarning
    # (build.yml). Forking from a multi-threaded process is precisely what is
    # under test here, so the warning is expected rather than a defect.
    pytest.mark.filterwarnings("ignore:This process:DeprecationWarning"),
]


def _child_torch_threads(i):
    import torch

    return torch.get_num_threads()


def _child_torch_matmul(x):
    # A genuine torch parallel region in the child -- this is what wedges an
    # unrestricted forked child.
    import torch

    m = torch.ones((256, 256), dtype=torch.float64)
    return float((m @ m)[0, 0]) + float(x)  # == 256 + x, exactly


def test_forked_child_is_capped_at_one_torch_thread():
    torch = pytest.importorskip("torch")
    from galpy.util.multi import parallel_map

    parent_before = torch.get_num_threads()
    got = list(parallel_map(_child_torch_threads, numpy.arange(4), numcores=2))
    assert got == [1, 1, 1, 1], (
        f"forked children must run torch single-threaded, got {got}"
    )
    assert torch.get_num_threads() == parent_before, (
        "restricting the child must not change the parent's thread count"
    )


def test_parallel_map_survives_torch_work_in_the_child():
    # Direct regression for the hang: warm the pool in the parent, then make
    # every child enter a torch parallel region. Before the fix this never
    # returned.
    torch = pytest.importorskip("torch")
    from galpy.util.multi import parallel_map

    warm = torch.ones((512, 512), dtype=torch.float64)
    assert float((warm @ warm)[0, 0]) == 512.0

    seq = numpy.arange(6.0)
    got = numpy.array(list(parallel_map(_child_torch_matmul, seq, numcores=3)))
    assert numpy.array_equal(got, 256.0 + seq), f"parallel_map returned {got}"


def _square(x):
    return float(x) ** 2.0


def test_parallel_map_values_are_exact():
    # The capping is a no-op for the numpy path: same values, exactly.
    from galpy.util.multi import parallel_map

    seq = numpy.arange(8.0)
    got = numpy.array(list(parallel_map(_square, seq, numcores=3)))
    assert numpy.array_equal(got, seq**2.0), f"parallel_map returned {got}"


def _child_pid(i):
    import os

    return os.getpid()


def test_parallel_map_does_not_fork_under_jax():
    # jax has no equivalent of torch's thread cap: a forked child inherits jax
    # state it cannot use and the parent hangs in proc.join(). The only cure is
    # not forking, so under jax parallel_map must run in-process. Asserted via
    # the child pid rather than by waiting on a join that would never return.
    pytest.importorskip("jax")
    import os

    from galpy.backend import use
    from galpy.util.multi import parallel_map

    with use("jax", force=True):
        pids = list(parallel_map(_child_pid, numpy.arange(4), numcores=2))
        seq = numpy.arange(8.0)
        got = numpy.array(list(parallel_map(_square, seq, numcores=3)))
    assert pids == [os.getpid()] * 4, f"parallel_map forked under jax: {pids}"
    assert numpy.array_equal(got, seq**2.0), f"parallel_map returned {got}"


def test_parallel_map_still_forks_off_jax():
    # Negative control for the test above: a guard that serialized everything
    # would satisfy it while silently costing every numpy user their cores.
    import os

    from galpy.util.multi import parallel_map

    pids = set(parallel_map(_child_pid, numpy.arange(4), numcores=2))
    assert pids != {os.getpid()}, "numpy parallel_map must still fork"
