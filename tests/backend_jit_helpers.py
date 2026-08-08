"""Shared assertions for the jax.jit tests in the ``test_backend_*`` modules.

Every one of those modules had grown its own copy of

    ref = numpy.asarray(fn(*args))
    got = numpy.asarray(jax.jit(fn)(*args))
    numpy.testing.assert_allclose(got, ref, rtol=..., atol=...)

which checks that jitting does not change the value but says nothing about
whether the jit actually traced anything. A trace that collapses to a constant
produces a jaxpr with no dependence on the arguments and still compares equal,
so the value check alone cannot see it (the same blind spot a numpy fallback
has under a forced backend). ``assert_jit_matches_eager`` does both: it
compares against eager and it inspects the jaxpr to confirm the output really
is a function of the traced arguments.
"""

import numpy

from galpy.backend import as_numpy

try:
    import jax
except ImportError:  # pragma: no cover - modules using this skip without jax
    jax = None


def jit_output_depends_on_inputs(fn, *args):
    """Whether ``fn``'s jaxpr consumes any of its traced arguments.

    ``False`` means the trace folded the arguments away entirely, so the
    compiled function returns a constant regardless of what it is called with.
    """
    jaxpr = jax.make_jaxpr(fn)(*args)
    invars = {str(v) for v in jaxpr.jaxpr.invars}
    consumed = {str(v) for eqn in jaxpr.eqns for v in eqn.invars}
    return bool(invars & consumed)


def assert_jit_matches_eager(fn, *args, rtol=1e-12, atol=1e-14, err_msg="", ref=None):
    """Assert ``jax.jit(fn)(*args)`` reproduces eager ``fn(*args)``.

    ``ref`` supplies an independently computed reference when the eager call
    should differ from ``fn(*args)`` -- typically to also exercise plain-float
    inputs against traced array ones.

    A quantity that genuinely does not vary with its arguments (a uniform
    density over its flat region, say) will trip the dependence assertion
    below; compare such a case directly rather than weakening the check here,
    since the check is what makes this stronger than a bare value comparison.
    """
    expected = as_numpy(fn(*args)) if ref is None else numpy.asarray(ref)
    assert jit_output_depends_on_inputs(fn, *args), (
        f"jit trace ignores its arguments{': ' + err_msg if err_msg else ''} "
        "-- the compiled function is a constant, so the value comparison "
        "below would pass vacuously"
    )
    got = as_numpy(jax.jit(fn)(*args))
    numpy.testing.assert_allclose(got, expected, rtol=rtol, atol=atol, err_msg=err_msg)
    return got


def count_boundary_crossings(fn, backend):
    """Run ``fn()`` under a forced backend; count non-numpy ``coerce_coords`` calls.

    The ``@backend_input`` boundary costs ~178 us per crossing under a forced
    backend. That is harmless at a few calls and ruinous inside a loop -- and it
    is invisible to ordinary assertions, because the values stay correct and only
    the time changes (galpy #1261/#1268, and the adapter forwards fixed after).

    Callers pass inputs that are ALREADY backend arrays, so a correctly-written
    internal forward reaches the undecorated inner evaluator and crosses **zero**
    times; any nonzero count means it went back through a decorated public entry.

    Pair every such assertion with a negative control that is *expected* to cross
    (e.g. a deliberately unmigrated second-derivative forward). Without one, a
    counter that silently stopped working makes the whole check pass vacuously.
    """
    import galpy.backend._input as _bi
    from galpy.backend import use

    real = _bi.coerce_coords
    n = [0]

    def spy(xp, *coords):
        if xp is not numpy:
            n[0] += 1
        return real(xp, *coords)

    _bi.coerce_coords = spy
    try:
        with use(backend, force=True):
            fn()
    finally:
        _bi.coerce_coords = real
    return n[0]
