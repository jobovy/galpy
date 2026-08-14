###############################################################################
# test_backend_jit_helpers.py: the shared jit assertions in
# tests/backend_jit_helpers.py.
#
# 22 test modules route their jit-vs-eager checks through
# assert_jit_matches_eager, and its whole reason to exist is the jaxpr
# dependence check: a bare assert_allclose(jitted, eager) passes vacuously when
# a trace folds its arguments away and returns a constant. Nothing tested the
# helper itself, so if jit_output_depends_on_inputs ever started returning True
# unconditionally -- a jax release reshaping jaxpr.invars/eqns is the obvious
# way -- all 22 adopters would silently degrade to plain value comparisons and
# the suite would stay green. These tests pin the DISCRIMINATION, not just the
# happy path: every one of them asserts the helper REJECTS something.
###############################################################################
import jax
import jax.numpy as jnp
import numpy
import pytest
from backend_jit_helpers import assert_jit_matches_eager, jit_output_depends_on_inputs


def test_depends_on_inputs_true_when_output_uses_the_argument():
    assert jit_output_depends_on_inputs(lambda x: x * 2.0, jnp.asarray(1.5))


def test_depends_on_inputs_false_when_the_trace_folds_the_argument_away():
    # the discriminating case: closes over a constant, ignores x entirely
    assert not jit_output_depends_on_inputs(
        lambda x: jnp.asarray(3.0), jnp.asarray(1.5)
    )


def test_depends_on_inputs_true_when_only_one_of_several_arguments_is_used():
    # the check is "consumes ANY declared argument", so a partially-used
    # signature passes; documented here so the looseness is deliberate rather
    # than an accident of the set intersection.
    assert jit_output_depends_on_inputs(
        lambda x, y: y * 2.0, jnp.asarray(1.5), jnp.asarray(2.5)
    )


def test_assert_jit_matches_eager_rejects_a_constant_trace():
    with pytest.raises(AssertionError, match="ignores its arguments"):
        assert_jit_matches_eager(lambda x: jnp.asarray(3.0), jnp.asarray(1.5))


def test_assert_jit_matches_eager_accepts_a_real_function_and_returns_its_value():
    got = assert_jit_matches_eager(lambda x: x**2, jnp.asarray(3.0))
    numpy.testing.assert_allclose(got, 9.0, rtol=0.0, atol=0.0)


def test_ref_is_actually_compared_against_and_not_ignored():
    # a wrong ref must fail, otherwise ref= would be decorative and the
    # anyaxisymdisk / dynamfric sites (which pass their numpy value that way)
    # would be asserting nothing at all.
    with pytest.raises(AssertionError):
        assert_jit_matches_eager(lambda x: x**2, jnp.asarray(3.0), ref=9.5)
    assert_jit_matches_eager(lambda x: x**2, jnp.asarray(3.0), ref=9.0, atol=0.0)


def test_tolerances_are_honoured_in_both_directions():
    # 3.0**2 vs a ref off by 1e-9: passes at atol 1e-8, fails at 1e-12. Pins
    # that the tolerance arguments reach assert_allclose rather than being
    # swallowed by the signature.
    f, x = lambda x: x**2, jnp.asarray(3.0)
    assert_jit_matches_eager(f, x, ref=9.0 + 1e-9, rtol=0.0, atol=1e-8)
    with pytest.raises(AssertionError):
        assert_jit_matches_eager(f, x, ref=9.0 + 1e-9, rtol=0.0, atol=1e-12)


def test_err_msg_reaches_both_the_dependence_and_the_value_failure():
    with pytest.raises(AssertionError, match="folded-away marker"):
        assert_jit_matches_eager(
            lambda x: jnp.asarray(3.0), jnp.asarray(1.5), err_msg="folded-away marker"
        )
    with pytest.raises(AssertionError, match="mismatch marker"):
        assert_jit_matches_eager(
            lambda x: x**2, jnp.asarray(3.0), ref=9.5, err_msg="mismatch marker"
        )


def test_array_valued_output_compares_elementwise():
    # a per-element check: an all-but-one match must still fail, so the
    # comparison is not silently reduced to a scalar (e.g. via .all() outside
    # the tolerance) the way a hand-rolled `all(fabs(d)) < tol` would be.
    f = lambda v: v * 2.0
    v = jnp.asarray([1.0, 2.0, 3.0])
    assert_jit_matches_eager(f, v, rtol=0.0, atol=0.0)
    with pytest.raises(AssertionError):
        assert_jit_matches_eager(f, v, ref=numpy.array([2.0, 4.0, 6.5]), atol=1e-12)


def test_the_dependence_check_survives_a_traced_constant_multiply():
    # x * 0.0 still CONSUMES x in the jaxpr even though the value is constant.
    # Recording the boundary: the check is structural (does the trace read the
    # argument), not numerical (does the output vary with it).
    assert jit_output_depends_on_inputs(lambda x: x * 0.0, jnp.asarray(1.5))
