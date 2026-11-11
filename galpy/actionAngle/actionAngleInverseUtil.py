###############################################################################
# actionAngleInverseUtil.py: what the momentum-matched inverse action-angle
#   transformations (actionAngleSphericalInverse, actionAngleStaeckelInverse)
#   share: the spectral tools on the regular offset anomaly grid (Fourier
#   coefficients, sine projections and series, the Galerkin solve of the
#   anomaly map's slopes), the margin of the bound-lift guard, the progress
#   bar, and the closed forms of the isochrone auxiliary's tori (energy and
#   radial action, orbital parameters, radius, momentum, flux and cumulative
#   action, with their derivatives) live on actionAngleIsochroneInverse
###############################################################################
import numpy

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
# split into chunks of this size, because the per-point cost of the kernel
# grows with the size of its arrays once they leave the cache (four times
# from 1e4 to 1e6 points)
_EVAL_CHUNK = 16384


def _chunked(fn, n, *arrays):
    """fn(*arrays) in chunks of _EVAL_CHUNK along the first axis of each of
    the n-long arrays, the outputs (a tuple of n-long arrays) concatenated"""
    if n <= _EVAL_CHUNK:
        return fn(*arrays)
    parts = [
        fn(*(a[k : k + _EVAL_CHUNK] for a in arrays)) for k in range(0, n, _EVAL_CHUNK)
    ]
    return tuple(numpy.concatenate([p[i] for p in parts]) for i in range(len(parts[0])))


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
    isochrone's through its v_c there. Returns (GM, b)."""
    rf = numpy.asarray(rf, dtype="float")
    lnr2 = 2.0 * numpy.log(rf) - lnvc2
    lo, hi = numpy.log(1e-8), numpy.log(1e3 * rf[-1])
    beta = 0.5 * (numpy.log(rf[0]) + numpy.log(rf[-1]))
    # a single radius to round-off (a lone circular torus's range): the
    # curvature of the rotation curve, which pins b, enters at the square of
    # the relative range and is lost below its square root, so every b fits
    # and the start is kept (an iteration on round-off would wander)
    single = rf[-1] - rf[0] <= 1e-8 * rf[-1]
    for _ in range(0 if single else maxiter):
        b = numpy.exp(beta)
        s = numpy.sqrt(b**2 + rf**2)
        h = -numpy.log(s) - 2.0 * numpy.log(b + s) + lnr2
        rho = h - numpy.mean(h)
        t = b / s
        q = -t * (t + 2.0)
        dq = -2.0 * t * (1.0 + t) * (1.0 - t**2)
        qc, dqc = q - numpy.mean(q), dq - numpy.mean(dq)
        g = numpy.sum(rho * qc)
        Hgn = numpy.sum(qc**2)
        H = Hgn + numpy.sum(rho * dqc)
        H = H if H > 0.0 else Hgn  # Gauss-Newton where the Hessian is not positive
        d = numpy.clip(-g / H, -0.5, 0.5)
        if (beta <= lo and d < 0.0) or (beta >= hi and d > 0.0):
            break  # at a bound, with the step pointing outward
        beta = numpy.clip(beta + d, lo, hi)
        if abs(d) < 1e-14:
            break
    b = numpy.exp(beta)
    s = numpy.sqrt(b**2 + rf**2)
    h = -numpy.log(s) - 2.0 * numpy.log(b + s) + lnr2
    return float(numpy.exp(-numpy.mean(h))), float(b)


def _safeguarded_newton(f, lo, hi, rtol, maxiter=100, x0=None):
    """Roots of the monotonically increasing functions f(x) = (value,
    derivative), one per element of the brackets [lo, hi], by Newton steps
    from x0 (the brackets' midpoints by default) kept inside the brackets
    (a step that leaves its bracket is replaced by the bracket's midpoint),
    the brackets shrinking around the roots as the iteration proceeds,
    until every step is below rtol (1 + |x|), the residual vanishes or the
    bracket has closed; vectorized over the elements"""
    lo, hi = numpy.array(lo, dtype="float"), numpy.array(hi, dtype="float")
    x = (
        0.5 * (lo + hi)
        if x0 is None
        else numpy.clip(numpy.array(x0, dtype="float"), lo, hi)
    )
    conv = numpy.zeros(x.shape, dtype="bool")
    for _ in range(maxiter):
        val, der = f(x)
        with numpy.errstate(divide="ignore", invalid="ignore"):
            step = val / der
        below = val < 0.0
        lo = numpy.where(below, x, lo)
        hi = numpy.where(below, hi, x)
        new = x - step
        bad = ~numpy.isfinite(new) | (new <= lo) | (new >= hi)
        new = numpy.where(bad, 0.5 * (lo + hi), new)
        tol = rtol * (1.0 + numpy.fabs(x))
        conv |= (val == 0.0) | (numpy.fabs(new - x) <= tol) | (hi - lo <= tol)
        x = numpy.where(conv, x, new)
        if numpy.all(conv):
            break
    return x


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


def _progress(iterator, desc, show):
    """The iterator, wrapped in a tqdm progress bar when one is wanted"""
    if show:
        return tqdm.tqdm(iterator, desc=desc, leave=False)
    return iterator


def _spaced_grid(lo, hi, n, spacing):
    """n grid nodes between lo and hi (both positive) spaced uniformly in the
    variable named by spacing: "linear" (the variable itself), "sqrt" (its
    square root, finer at the small end) or "log" (its logarithm)"""
    if spacing == "linear":
        return numpy.linspace(lo, hi, n)
    if spacing == "sqrt":
        return numpy.linspace(numpy.sqrt(lo), numpy.sqrt(hi), n) ** 2.0
    if spacing == "log":
        return numpy.geomspace(lo, hi, n)
    raise ValueError(
        f"spacing {spacing!r} not understood; use 'linear', 'sqrt' or 'log'"
    )
