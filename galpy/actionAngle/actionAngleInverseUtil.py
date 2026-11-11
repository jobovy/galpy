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
    coefficients D, of shape (n, M) or broadcastable to it: one coefficient
    vector per anomaly, or one for all of them"""
    return numpy.einsum("nm,nm->n", smt, numpy.broadcast_to(D, smt.shape))


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
