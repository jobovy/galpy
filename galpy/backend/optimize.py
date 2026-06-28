###############################################################################
#   galpy.backend.optimize: a backend-agnostic 1-D bracketed root finder.
#
#   ``brentq(func, a, b, args=(), xtol=...)`` mirrors the part of
#   ``scipy.optimize.brentq`` that galpy uses. The dispatch follows the data:
#
#     * numpy / Python-scalar bracket -> delegate to ``scipy.optimize.brentq``,
#       so the numpy path is BYTE-IDENTICAL to today (galpy already imports it
#       everywhere it root-finds).
#     * a jax / torch bracket -> a vectorisable, jit/grad-COMPATIBLE bracketed
#       root find: bisection (sign-change preserving, so it can never leave the
#       bracket) for a fixed schedule of halvings, then a single Newton step to
#       polish the value AND carry the parameter gradient. There is NO internal
#       jit -- galpy is jit-COMPATIBLE; users wrap their own galpy-using code.
#
#   DIFFERENTIABILITY (the point of the backend path). The root x* of
#   f(x; theta) = 0 depends on the parameters theta that f closes over. The
#   implicit function theorem gives the exact sensitivity
#
#       dx*/dtheta = -(df/dtheta) / (df/dx)        (evaluated at x*),
#
#   but bisection's branchy comparisons carry no usable gradient. We recover the
#   exact implicit-diff gradient with the standard ONE-NEWTON-STEP
#   REPARAMETERISATION: with x0 = stop_gradient(x*_bisection),
#
#       x* = x0 - f(x0; theta) / (df/dx)(x0; theta).
#
#   The forward value is unchanged (f(x0) ~ 0 at the bisection root, so the
#   correction is ~0), while differentiating the closed form -- x0 is a constant
#   w.r.t. theta and f(x0) ~ 0 -- yields
#
#       dx*/dtheta = -(df/dtheta)(x0) / (df/dx)(x0) + O(f(x0)),
#
#   i.e. exactly the implicit-function-theorem sensitivity. (df/dx)(x0) is
#   obtained by the backend's own autodiff -- jax.jvp for jax, a tiny
#   autograd-graph derivative for torch -- so no hand-coded df/dx is needed and
#   the gradient flows to every theta f closes over. This is mathematically the
#   same as a jax.custom_jvp / torch.autograd.Function implementing the implicit
#   theorem, but written so it composes with both backends' AD transparently.
#
#   SCOPE: this is a FIRST-ORDER construction. The forward value sits at the
#   bisection root and x0 is stop_gradient/detached, so the FIRST derivative
#   dx*/dtheta is exact (implicit-function theorem) and composes with jit/vmap,
#   but SECOND and higher derivatives are NOT recovered (jax returns 0; torch
#   reports the tensor as unused). galpy's root-finding call sites need only the
#   first-order sensitivity; a Hessian through x* would require a true
#   custom_jvp differentiating the implicit relation a second time.
###############################################################################
from ._namespaces import is_backend_array
from ._resolver import get_namespace

# Default bracketing tolerance (matches scipy.optimize.brentq's xtol default, so
# the backend path converges to the same neighbourhood as the numpy path).
_XTOL = 2e-12
_MAXITER = 100


def brentq(func, a, b, args=(), xtol=_XTOL, rtol=None, maxiter=_MAXITER):
    """Find a root of ``func`` in the bracket ``[a, b]`` (backend-agnostic).

    A drop-in for the subset of ``scipy.optimize.brentq`` galpy uses:
    ``brentq(func, a, b, args=(), xtol=...)``.

    Parameters
    ----------
    func : callable
        ``func(x, *args) -> scalar/array``. Must change sign on ``[a, b]``.
        For the backend (jax/torch) path it must be written in the array
        namespace (so it is differentiable); for the numpy path it is whatever
        ``scipy.optimize.brentq`` accepts.
    a, b : scalar or backend array
        The bracket endpoints, with ``func(a)`` and ``func(b)`` of opposite
        sign. Their type selects the backend: numpy/Python scalars route to
        scipy (byte-identical); a jax/torch array routes to the in-backend
        bisection + Newton-polish, and gradients flow w.r.t. any parameters
        ``func`` closes over.
    args : tuple, optional
        Extra arguments forwarded to ``func`` as ``func(x, *args)``.
    xtol : float, optional
        Absolute bracket tolerance (default ``2e-12``, scipy's default). On the
        backend path it sets the number of bisection halvings (enough to shrink
        the initial bracket below ``xtol``) before the Newton polish.
    rtol : float, optional
        Accepted for signature compatibility with ``scipy.optimize.brentq``;
        forwarded to scipy on the numpy path, ignored on the backend path
        (which converges in absolute bracket width then Newton-polishes).
    maxiter : int, optional
        Maximum number of bisection iterations on the backend path / forwarded
        to scipy on the numpy path (default 100).

    Returns
    -------
    scalar or backend array
        The root. numpy -> a Python float (scipy). jax/torch -> a backend array
        differentiable w.r.t. ``func``'s parameters via the implicit function
        theorem.
    """
    # Follow the data: the bracket endpoints (and any array in args) decide.
    if not (
        is_backend_array(a) or is_backend_array(b) or any(map(is_backend_array, args))
    ):
        from scipy import optimize as _sopt

        kw = {} if rtol is None else {"rtol": rtol}
        return _sopt.brentq(func, a, b, args=args, xtol=xtol, maxiter=maxiter, **kw)
    xp = get_namespace(a, b, *args)
    name = getattr(xp, "__name__", "")
    f = (lambda x: func(x, *args)) if args else func
    if "jax" in name:
        from ._jax.optimize import brentq_backend as _bk
    else:  # array_api_compat.torch
        from ._torch.optimize import brentq_backend as _bk
    return _bk(f, a, b, xp, xtol=xtol, maxiter=maxiter)


def n_bisect_steps(a, b, xtol, maxiter):
    """Number of bisection halvings to shrink ``[a, b]`` below ``xtol``.

    A fixed schedule (no data-dependent early-out) keeps the backend bisection
    jit-/vmap-compatible: ``ceil(log2(width / xtol))`` halvings bring the bracket
    half-width below ``xtol``, capped at ``maxiter``. ``a, b`` may be arrays; the
    widest bracket sets the count so every element is converged.

    The count is a *structural* (Python int) quantity: a jit/grad TRACER carries
    no concrete width, so probing it would raise (the bracket value is abstract
    when differentiating w.r.t. an endpoint). In that case -- and on any read
    failure -- fall back to the full ``maxiter`` schedule, which is correct
    (more halvings only tighten the bracket) and fully jit-compatible.
    """
    import math

    import numpy

    try:
        width = float(numpy.max(numpy.abs(numpy.asarray(b) - numpy.asarray(a))))
    except Exception:
        # Traced (abstract) bracket under jit/grad: no concrete width available.
        return min(maxiter, _MAXITER)
    if not width > 0.0 or not xtol > 0.0:  # pragma: no cover - degenerate bracket
        return min(maxiter, _MAXITER)
    n = int(math.ceil(math.log2(width / xtol))) + 1
    return max(1, min(n, maxiter))


def bisect_step(lo, hi, slo, f, xp):
    """One sign-preserving bisection halving of ``[lo, hi]`` (branch-free).

    Returns the updated ``(lo, hi)``. Root in ``[lo, mid]`` iff ``f(mid)`` keeps
    the sign of ``f(lo)`` on ``[mid, hi]`` (``sign(f(mid)) == slo`` -> the sign
    change is in ``[mid, hi]``: move ``lo`` up). The shared per-step kernel so the
    Python-loop (eager/torch) and the jax ``lax.fori_loop`` body run identical
    arithmetic.
    """
    mid = 0.5 * (lo + hi)
    same = xp.sign(f(mid)) == slo
    return xp.where(same, mid, lo), xp.where(same, hi, mid)


def bisect_root(f, a, b, xp, *, xtol, maxiter):
    """Vectorised, sign-preserving bisection root of ``f`` on ``[a, b]`` in ``xp``.

    Runs a FIXED schedule of ``n_bisect_steps`` halvings (no data-dependent
    branch, so it vmaps/jits cleanly) and returns the final bracket midpoint.
    Branch-free per step (``xp.where`` on sign agreement), so it vectorises over
    array brackets and can never leave the bracket. The returned value carries no
    meaningful gradient (the comparisons are piecewise-constant); the caller
    reparameterises it through one Newton step for the implicit-function
    gradient. Shared by both backend paths so the logic lives in one place.

    This is the eager Python-loop form (torch, and jax outside a trace); the jax
    path swaps in a ``lax.fori_loop`` over the same ``bisect_step`` only when
    tracing (see ``_jax.optimize``), keeping the jaxpr small under the user's jit.

    ``f`` must change sign on ``[a, b]`` (``f(a)``, ``f(b)`` opposite sign), as
    for ``scipy.optimize.brentq``; same-sign brackets are a caller error and
    return a midpoint without a guarantee.
    """
    # Anchor on the inputs (device/dtype); * 1.0 promotes integer brackets.
    lo = xp.asarray(a) * 1.0
    hi = xp.asarray(b) * 1.0
    slo = xp.sign(f(lo))  # sign of f at the low endpoint
    n = n_bisect_steps(a, b, xtol, maxiter)
    for _ in range(n):
        lo, hi = bisect_step(lo, hi, slo, f, xp)
    return 0.5 * (lo + hi)
