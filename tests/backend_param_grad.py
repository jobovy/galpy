"""Shared grad-vs-finite-difference check for potential constructor parameters.

``assert_param_grad(build, params, name, method, coords)`` differentiates
``method`` of ``build(**params)`` w.r.t. ``params[name]``, with the potential
built INSIDE the differentiated function, and compares against a Richardson-
extrapolated central difference of the numpy path:

* jax: eager ``jax.grad``, ``jax.jit(jax.grad)`` and ``jax.vmap(jax.grad)`` over
  two parameter values; torch: ``autograd.grad`` (and a graph must exist);
* in ``data`` mode (backend-array coordinates) and ``forced`` mode (plain-float
  coordinates under ``use(backend, force=True)``).

The tolerance is ``rtol`` or 10x the FD's own convergence estimate (the
difference of two Richardson levels), whichever is larger.
"""

import contextlib

import numpy

from galpy import backend

try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    jax = None
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

AD_BACKENDS = (["jax"] if jax is not None else []) + (
    ["torch"] if torch is not None else []
)
MODES = ["data", "forced"]


def _call(pot, method, coords):
    f = pot if method == "Phi" else getattr(pot, method)
    return f(*coords)


def fd_reference(build, params, name, method, coords, h=None):
    """(Richardson central difference, its convergence estimate) on numpy"""
    x0 = params[name]
    h = 1e-3 * max(abs(x0), 0.1) if h is None else h

    def f(x):
        return float(_call(build(**{**params, name: x}), method, coords))

    def cd(h):
        return (f(x0 + h) - f(x0 - h)) / (2.0 * h)

    def rich(h):
        return (4.0 * cd(h / 2.0) - cd(h)) / 3.0

    r1, r2 = rich(2.0 * h), rich(h)
    return r2, abs(r2 - r1)


def _ad(backend_name, mode, build, params, name, method, coords, x0):
    fixed = {k: v for k, v in params.items() if k != name}
    if backend_name == "jax":
        xp_coords = coords if mode == "forced" else [jnp.asarray(c) for c in coords]

        def f(x):
            with (
                backend.use("jax", force=True)
                if mode == "forced"
                else contextlib.nullcontext()
            ):
                return _call(build(**fixed, **{name: x}), method, xp_coords)

        g = jax.grad(f)
        x = jnp.asarray(x0)
        out = [float(g(x)), float(jax.jit(g)(x))]
        out.append(float(jax.vmap(g)(jnp.asarray([x0, x0]))[1]))
        return out
    xp_coords = (
        coords
        if mode == "forced"
        else [torch.tensor(c, dtype=torch.float64) for c in coords]
    )
    x = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
    with (
        backend.use("torch", force=True)
        if mode == "forced"
        else contextlib.nullcontext()
    ):
        out = _call(build(**fixed, **{name: x}), method, xp_coords)
    assert isinstance(out, torch.Tensor) and out.grad_fn is not None, (
        f"torch {method}: no graph to {name}"
    )
    (g,) = torch.autograd.grad(out, x)
    return [float(g)]


def assert_param_grad(
    backend_name, mode, build, params, name, method, coords, rtol=1e-9, fd=None
):
    if backend_name == "torch":
        torch.set_default_dtype(torch.float64)
    ref, err = fd_reference(build, params, name, method, coords) if fd is None else fd
    tol = max(rtol * abs(ref), 10.0 * err, 1e-14)
    got = _ad(backend_name, mode, build, params, name, method, coords, params[name])
    for g in got:
        assert abs(g - ref) <= tol, (
            f"{backend_name}/{mode} d{method}/d{name}: {got} vs FD {ref} "
            f"(tol {tol:.1e})"
        )
    return got
