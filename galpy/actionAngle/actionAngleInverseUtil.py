###############################################################################
# actionAngleInverseUtil.py: what the momentum-matched inverse action-angle
#   transformations (actionAngleSphericalInverse, actionAngleStaeckelInverse)
#   share: the spectral tools on the regular offset anomaly grid (Fourier
#   coefficients, sine projections and series, the Galerkin solve of the
#   anomaly map's slopes), the margin of the bound-lift guard, the progress
#   bar, and the closed forms of the isochrone auxiliary's tori (energy and
#   radial action, orbital parameters, radius, momentum, flux and cumulative
#   action, with their derivatives), as a mixin class
###############################################################################
import numpy

from ..backend import asarray_on_device, device_of, is_backend_array
from ..backend._namespaces import namespace_from_arrays
from ..backend.optimize import iterate_to_fixed_point
from ..util._optional_deps import _TQDM_LOADED

if _TQDM_LOADED:
    import tqdm

# The lift of every torus onto the auxiliary must stay bound in it, with
# this fraction of its own auxiliary torus's binding energy in hand. The
# value is not delicate: a resolved map puts the lifted curve on its torus
# to round-off (the ratio of the highest lifted energy to the torus's is
# 1.000000 on every node of a typical grid), and an under-resolved one
# overshoots by orders of magnitude (a torus with apocentre 670 times its
# pericentre gives ratios of -121, -25 and -1.5 at 32, 64 and 128
# harmonics and 0.990 at 256), so any margin between a few per cent and
# most of the binding energy sorts the same cases the same way.
_BOUND_MARGIN = 0.05


def _offset_spec_coeffs(f):
    """True Fourier coefficients c_k of a real periodic function sampled on
    the regular offset grid tau_j = 2 pi (j + 1/2)/N, such that
    f(tau) = Re c_0 + sum_{k=1}^{N/2-1} 2 Re[c_k e^{i k tau}] +
    Re[c_{N/2} e^{i N tau / 2}]; the offset grid requires the phase
    correction e^{-i k pi / N} relative to the raw rfft"""
    N = len(f)
    k = numpy.arange(N // 2 + 1)
    return numpy.fft.rfft(f) / N * numpy.exp(-1j * k * numpy.pi / N)


def _spec_eval(c, tau, deriv=False):
    """Evaluate the Fourier series with coefficients c (from
    _offset_spec_coeffs) or its derivative at arbitrary tau"""
    k = numpy.arange(len(c))
    w = numpy.ones(len(c))
    w[1:-1] = 2.0
    cc = c * (1j * k) if deriv else c
    ph = numpy.exp(1j * numpy.atleast_1d(tau)[:, None] * k[None, :])
    return numpy.real(ph @ (w * cc))


def _sine_project(g, ms):
    """The sine coefficients 2 mean_k g_k sin(m tau_k) of a function sampled
    on the regular offset grid tau_k = 2 pi (k + 1/2)/N, for the harmonics
    ms, by one FFT (the grid's offset is the phase e^{i m pi/N}); g may have
    several columns"""
    N = len(g)
    ph = numpy.exp(1j * ms * numpy.pi / N).reshape((len(ms),) + (1,) * (g.ndim - 1))
    return 2.0 / N * numpy.imag(ph * numpy.conj(numpy.fft.fft(g, axis=0)[ms]))


# points per chunk of an array evaluation: a call over many distinct tori is
# split into chunks, because the per-point cost of the kernel grows with the
# size of its arrays once they leave the cache. The arrays hold one row of
# harmonics per point, so the chunk holds a fixed number of elements (points
# times harmonics) rather than of points: at most _EVAL_CHUNK points, fewer
# for maps with many harmonics, but no fewer than _EVAL_CHUNK_MIN, below
# which the fixed cost of a call dominates
_EVAL_CHUNK = 16384
_EVAL_CHUNK_ELEMENTS = 200000
_EVAL_CHUNK_MIN = 1024


def _chunked(fn, n, *arrays, nharm=1):
    """fn(*arrays) in chunks along the first axis of each of the n-long
    arrays, the outputs (a tuple of n-long arrays) concatenated; the chunk
    has at most _EVAL_CHUNK points and about _EVAL_CHUNK_ELEMENTS / nharm
    points for a kernel with nharm harmonics per point"""
    chunk = min(
        _EVAL_CHUNK, max(_EVAL_CHUNK_ELEMENTS // max(nharm, 1), _EVAL_CHUNK_MIN)
    )
    if n <= chunk:
        return fn(*arrays)
    parts = [fn(*(a[k : k + chunk] for a in arrays)) for k in range(0, n, chunk)]
    return tuple(numpy.concatenate([p[i] for p in parts]) for i in range(len(parts[0])))


def _harmonics(tau, ms, pointwise=False):
    """sin(m tau) and cos(m tau) for the harmonics ms = ms[0] (1, ..., M),
    from the sine and cosine of ms[0] tau alone: the angle-addition formulae
    double the number of harmonics in hand at every step, so that a point
    costs two trigonometric evaluations instead of 2 M and the round-off
    grows only with the logarithm of M. Returned as (M, npoints) arrays, or
    as (npoints, M) arrays when pointwise, the layout in which coefficients
    that differ from point to point, (npoints, M), contract with them as
    one contiguous product per point"""
    M = len(ms)
    S, C = (
        numpy.empty((len(tau), M) if pointwise else (M, len(tau))) for _ in range(2)
    )
    # the harmonics along the first axis, whichever the layout
    s, c = (S.T, C.T) if pointwise else (S, C)
    s[0], c[0] = numpy.sin(ms[0] * tau), numpy.cos(ms[0] * tau)
    m = 1
    while m < M:
        # harmonics m+1 ... 2m from 1 ... m and m
        n = min(m, M - m)
        s[m : m + n] = s[:n] * c[m - 1] + c[:n] * s[m - 1]
        c[m : m + n] = c[:n] * c[m - 1] - s[:n] * s[m - 1]
        m += n
    return S, C


def _harmonics_xp(xp, tau, ms, pointwise=False):
    """_harmonics in namespace xp: on jax/torch by direct sines and cosines,
    which cost no more there than the angle-addition steps (those add
    dispatches eagerly and compile time under jax.jit)"""
    if xp is numpy:
        return _harmonics(tau, ms, pointwise=pointwise)
    mt = tau[:, None] * ms[None, :] if pointwise else ms[:, None] * tau[None, :]
    return xp.sin(mt), xp.cos(mt)


def _sdot(smt, D):
    """The sine (or cosine) matrix sin(m tau_k) of shape (n, M) times the
    coefficients D: one coefficient vector for all anomalies (shape (1, M),
    a matrix product) or one per anomaly (shape (n, M))"""
    D = numpy.asarray(D)
    if D.shape[0] == 1:
        return smt @ D[0]
    return numpy.einsum("nm,nm->n", smt, D)


def _fit_isochrone_vc2(rf, lnvc2, maxiter=100):
    """Least-squares fit of an isochrone's (GM, b) to a rotation curve, given
    as ln v_c^2 at the radii rf: minimizes sum_i rho_i^2 with rho_i = ln GM
    + 2 ln r_i - ln s_i - 2 ln(b + s_i) - lnvc2_i, s_i = sqrt(b^2 + r_i^2).
    ln GM enters linearly, so at given b its optimum is the mean of the
    other terms (variable projection), which leaves a one-dimensional
    problem in beta = ln b solved by Newton's method with closed-form
    derivatives (Gauss-Newton where the Hessian is not positive), the step
    clipped at 1/2 and beta kept in [ln 1e-8, ln(1e3 r_max)], from b_0 =
    sqrt(r_min r_max), until the step is below 1e-14 or beta sits at a
    bound with the step pointing outward; converges in a handful of
    iterations and is exact for an isochrone target. A rotation curve sampled
    at a single radius to round-off (a family of one circular torus) is
    fitted by any b, so the start b_0 = that radius is kept and GM is the
    isochrone's through its v_c there. Returns (GM, b): floats for numpy
    inputs, 0-d arrays for backend ones, which a traced
    construction thus fits on the device (each step frozen once converged,
    a lax.while_loop under a jax trace); stop_gradient those, the auxiliary
    being frozen."""
    xp = namespace_from_arrays((rf, lnvc2)) or numpy
    if xp is numpy:
        rf = numpy.asarray(rf, dtype="float")
    lnr2 = 2.0 * xp.log(rf) - lnvc2
    lo, hi = numpy.log(1e-8), xp.log(1e3 * rf[-1])

    def _h(beta):
        b = xp.exp(beta)
        s = xp.sqrt(b**2 + rf**2)
        return b, s, -xp.log(s) - 2.0 * xp.log(b + s) + lnr2

    def _step(x):
        # x = (beta, done)
        beta, done = x[0], x[1] > 0.0
        b, s, h = _h(beta)
        rho = h - xp.mean(h)
        t = b / s
        q = -t * (t + 2.0)
        dq = -2.0 * t * (1.0 + t) * (1.0 - t**2)
        qc, dqc = q - xp.mean(q), dq - xp.mean(dq)
        g = xp.sum(rho * qc)
        Hgn = xp.sum(qc**2)
        H = Hgn + xp.sum(rho * dqc)
        # Gauss-Newton where the Hessian is not positive
        H = xp.where(H > 0.0, H, Hgn)
        d = xp.clip(-g / H, -0.5, 0.5)
        # at a bound, with the step pointing outward, beta stays
        stop = done | ((beta <= lo) & (d < 0.0)) | ((beta >= hi) & (d > 0.0))
        beta = xp.where(stop, beta, xp.clip(beta + d, lo, hi))
        stop = stop | (xp.abs(d) < 1e-14)
        return xp.stack([beta, xp.where(stop, 1.0 + 0.0 * beta, 0.0 * beta)])

    beta = 0.5 * (xp.log(rf[0]) + xp.log(rf[-1]))
    # a single radius to round-off (a lone circular torus's range): the
    # curvature of the rotation curve, which pins b, enters at the square of
    # the relative range and is lost below its square root, so every b fits
    # and the start is kept (an iteration on round-off would wander)
    single = rf[-1] - rf[0] <= 1e-8 * rf[-1]
    x0 = xp.stack([beta, xp.where(single, 1.0 + 0.0 * beta, 0.0 * beta)])
    beta = iterate_to_fixed_point(_step, x0, maxiter)[0]
    b, _, h = _h(beta)
    GM = xp.exp(-xp.mean(h))
    return (float(GM), float(b)) if xp is numpy else (GM, b)


def _safeguarded_newton(f, lo, hi, rtol, maxiter=100, x0=None, args=()):
    """Roots of the monotonically increasing functions f(x, *args) = (value,
    derivative), one per element of the brackets [lo, hi] (args are arrays
    with one entry per element), by Newton steps from x0 (the brackets'
    midpoints by default) kept inside the brackets (a step that leaves its
    bracket is replaced by the bracket's midpoint), the brackets shrinking
    around the roots as the iteration proceeds, until every step is below
    rtol (1 + |x|), the residual vanishes or the bracket has closed;
    vectorized over the elements, each of which leaves the iteration, and
    is no longer evaluated, once it has converged"""
    lo, hi = numpy.array(lo, dtype="float"), numpy.array(hi, dtype="float")
    x = (
        0.5 * (lo + hi)
        if x0 is None
        else numpy.clip(numpy.array(x0, dtype="float"), lo, hi)
    )
    # the iterates, brackets and arguments flattened, and the elements that
    # are still active
    shape = x.shape
    x, lo, hi = x.ravel(), lo.ravel(), hi.ravel()
    args = [numpy.ravel(a) for a in args]
    active = numpy.arange(x.size)
    for _ in range(maxiter):
        xa = x[active]
        val, der = f(xa, *(a[active] for a in args))
        with numpy.errstate(divide="ignore", invalid="ignore"):
            step = val / der
        below = val < 0.0
        loa = numpy.where(below, xa, lo[active])
        hia = numpy.where(below, hi[active], xa)
        new = xa - step
        bad = ~numpy.isfinite(new) | (new <= loa) | (new >= hia)
        new = numpy.where(bad, 0.5 * (loa + hia), new)
        tol = rtol * (1.0 + numpy.fabs(xa))
        conv = (val == 0.0) | (numpy.fabs(new - xa) <= tol) | (hia - loa <= tol)
        x[active], lo[active], hi[active] = numpy.where(conv, xa, new), loa, hia
        active = active[~conv]
        if active.size == 0:
            break
    return x.reshape(shape)


def _sine_series(N, Dm, ms):
    """The sine series sum_m D_m sin(m tau_k) over the harmonics ms and its
    derivative sum_m m D_m cos(m tau_k) on the regular offset grid tau_k =
    2 pi (k + 1/2)/N, by one inverse FFT each"""
    c = numpy.zeros(N, dtype="complex")
    c[ms] = Dm * numpy.exp(1j * ms * numpy.pi / N)
    s = numpy.imag(numpy.fft.ifft(c)) * N
    c[ms] *= ms
    return s, numpy.real(numpy.fft.ifft(c)) * N


def _sine_galerkin(w, rhs, ms):
    """Solve w(tau) sum_m x_m sin(m tau) = rhs(tau) over the harmonics ms
    for the x_m in the Galerkin sense on the regular offset grid: project
    both sides onto sin(m' tau), which gives the system sum_m G_{m'm} x_m =
    r_{m'} with G_{m'm} = sum_k w_k sin(m tau_k) sin(m' tau_k) = [C_{|m-m'|}
    - C_{m+m'}]/2, C_j = sum_k w_k cos(j tau_k), and r_{m'} = sum_k rhs_k
    sin(m' tau_k), all of which one FFT each supplies; rhs may have several
    columns. A vanishing weight (the flux at the turning points) enters the
    matrix instead of dividing anything."""
    N = len(w)
    M = ms[-1]
    j = numpy.arange(2 * M + 1)
    C = numpy.real(numpy.exp(-1j * j * numpy.pi / N) * numpy.fft.fft(w)[: 2 * M + 1])
    G = 0.5 * (C[numpy.abs(ms[:, None] - ms[None, :])] - C[ms[:, None] + ms[None, :]])
    Fr = numpy.fft.fft(rhs, axis=0)[ms]
    r = numpy.imag(
        numpy.exp(1j * ms * numpy.pi / N).reshape((len(ms),) + (1,) * (rhs.ndim - 1))
        * numpy.conj(Fr)
    )
    return numpy.linalg.solve(G, r)


# ---------- the spectral tools in jax/torch, batched over tori: the samples
# (or harmonics) along the last axis and any leading axes over tori, where
# the numpy functions above take one torus with its samples along the first
def _ns(*args):
    """The namespace of the backend arrays among args, else numpy"""
    bk = tuple(a for a in args if is_backend_array(a))
    return namespace_from_arrays(bk) if bk else numpy


def _const(xp, a, ref):
    """The numpy constant a on the backend and device of the array ref"""
    return asarray_on_device(xp, a, device_of(ref))


def _offset_spec_coeffs_xp(xp, f):
    """_offset_spec_coeffs along the last axis"""
    N = f.shape[-1]
    k = numpy.arange(N // 2 + 1)
    return (
        xp.fft.rfft(f, axis=-1) / N * _const(xp, numpy.exp(-1j * k * numpy.pi / N), f)
    )


def _spec_eval_xp(xp, c, tau):
    """_spec_eval at the anomalies tau (shape (m,)) of the series with
    coefficients c (shape (..., K)): returns (..., m)"""
    k = numpy.arange(c.shape[-1])
    w = numpy.ones(c.shape[-1])
    w[1:-1] = 2.0
    ph = xp.exp(1j * tau[:, None] * _const(xp, k * 1.0, c)[None, :])
    return xp.real((c * _const(xp, w, c)) @ xp.swapaxes(ph, -1, -2))


def _antiderivative_xp(xp, f):
    """The periodic antiderivative of f minus its mean along the last axis,
    with zero mean, by FFT"""
    N = f.shape[-1]
    k = numpy.fft.fftfreq(N, d=1.0 / N)
    inv = numpy.zeros(N, dtype="complex")
    inv[1:] = 1.0 / (1j * k[1:])
    fh = xp.fft.fft(f - xp.mean(f, axis=-1, keepdims=True), axis=-1)
    return xp.real(xp.fft.ifft(fh * _const(xp, inv, f), axis=-1))


def _sine_project_xp(xp, g, ms):
    """_sine_project of samples along the last axis: (..., N) -> (..., len(ms))"""
    N = g.shape[-1]
    Fm = xp.take(xp.fft.fft(g, axis=-1), _const(xp, ms, g), axis=-1)
    ph = _const(xp, numpy.exp(1j * ms * numpy.pi / N), g)
    return 2.0 / N * xp.imag(ph * xp.conj(Fm))


def _sine_series_xp(xp, N, Dm, ms):
    """_sine_series of coefficients of shape (..., len(ms)): the series and its
    derivative on the offset grid, each of shape (..., N)"""
    # the harmonics scattered into a length-N spectrum by a one-hot matrix
    E = numpy.zeros((len(ms), N), dtype="complex")
    E[numpy.arange(len(ms)), ms] = numpy.exp(1j * ms * numpy.pi / N)
    E = _const(xp, E, Dm)
    s = xp.imag(xp.fft.ifft((Dm + 0j) @ E, axis=-1)) * N
    ds = xp.real(xp.fft.ifft((Dm * _const(xp, ms * 1.0, Dm) + 0j) @ E, axis=-1)) * N
    return s, ds


def _sine_galerkin_xp(xp, w, rhs, ms, nkeep=None):
    """_sine_galerkin batched over tori: w (..., N) and rhs (..., N, ncol)
    give (..., len(ms), ncol); with nkeep (...,), each torus's problem in its
    harmonics m <= nkeep only, zero beyond (the static-shape form of
    _sine_galerkin with ms[:nkeep]: identity rows and columns beyond, with
    zero right-hand side, so that the leading block is that system)"""
    N = w.shape[-1]
    M = ms[-1]
    j = numpy.arange(2 * M + 1)
    C = xp.real(
        _const(xp, numpy.exp(-1j * j * numpy.pi / N), w)
        * xp.fft.fft(w, axis=-1)[..., : 2 * M + 1]
    )
    nm = len(ms)
    Gd = xp.take(
        C, _const(xp, numpy.abs(ms[:, None] - ms[None, :]).ravel(), w), axis=-1
    )
    Gs = xp.take(C, _const(xp, (ms[:, None] + ms[None, :]).ravel(), w), axis=-1)
    G = xp.reshape(0.5 * (Gd - Gs), tuple(C.shape[:-1]) + (nm, nm))
    Fr = xp.take(xp.fft.fft(rhs, axis=-2), _const(xp, ms, w), axis=-2)
    ph = _const(xp, numpy.exp(1j * ms * numpy.pi / N), w)
    b = xp.imag(ph[:, None] * xp.conj(Fr))
    if nkeep is not None:
        keep = _const(xp, ms * 1.0, w) <= nkeep[..., None]
        G = xp.where(
            keep[..., :, None] & keep[..., None, :], G, _const(xp, numpy.eye(nm), w)
        )
        b = xp.where(keep[..., None], b, 0.0 * b)
    return xp.linalg.solve(G, b)


def _bk_matching(xp, resid, slope, resid_sg, slope_sg, eta0, mt, maxiter=200):
    """The pointwise momentum matching of _momentum_matched_lift in jax/torch:
    Newton on resid(eta) = 0 with clipped steps and the floored slope, from
    eta0 (the anomaly itself: both cumulative actions are monotone, so the
    root is the same as from numpy's interpolated start), without a
    derivative until every sample has converged (at most maxiter steps, as
    numpy), then one step that carries the implicit-function derivative of
    the matched anomaly (resid_sg and slope_sg: the same on derivative-free
    copies of the tables, for the iteration)"""
    tol = 1e-13 * xp.clip(mt, 1e-10, None)[:, None]

    def step(eta):
        f, fp = resid_sg(eta), slope_sg(eta)
        return xp.where(xp.abs(f) < tol, eta, eta + xp.clip(-f / fp, -0.5, 0.5))

    eta = iterate_to_fixed_point(step, eta0, maxiter)
    return eta - resid(eta) / slope(eta)


def _progress(iterator, desc, show):
    """The iterator, wrapped in a tqdm progress bar when one is wanted"""
    if show:
        return tqdm.tqdm(iterator, desc=desc, leave=False)
    return iterator


def _spaced_grid(lo, hi, n, spacing):
    """n grid nodes between lo and hi (both positive) spaced uniformly in the
    variable named by spacing: "linear" (the variable itself), "sqrt" (its
    square root, finer at the small end) or "log" (its logarithm). For
    backend lo and hi (a construction differentiable in them), the nodes as
    numpy.linspace computes them (start + step k, the end set exactly) in
    that variable, numpy.geomspace's ends exact too"""
    if spacing not in ("linear", "sqrt", "log"):
        raise ValueError(
            f"spacing {spacing!r} not understood; use 'linear', 'sqrt' or 'log'"
        )
    xp = namespace_from_arrays((lo, hi)) or numpy
    if xp is numpy:
        if spacing == "linear":
            return numpy.linspace(lo, hi, n)
        if spacing == "sqrt":
            return numpy.linspace(numpy.sqrt(lo), numpy.sqrt(hi), n) ** 2.0
        return numpy.geomspace(lo, hi, n)
    cat = getattr(xp, "concat", None) or xp.concatenate
    k = asarray_on_device(xp, numpy.arange(n - 1, dtype="float"), device_of(lo, hi))

    def _lin(a, b):
        return cat([k * ((b - a) / (n - 1)) + a, xp.reshape(b, (1,))])

    if spacing == "linear":
        return _lin(lo, hi)
    if spacing == "sqrt":
        return _lin(xp.sqrt(lo), xp.sqrt(hi)) ** 2.0
    g = xp.exp(_lin(xp.log(lo), xp.log(hi)))
    return cat([xp.reshape(lo, (1,)), g[1:-1], xp.reshape(hi, (1,))])
