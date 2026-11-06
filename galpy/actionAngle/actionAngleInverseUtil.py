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


# A mixin: the inverse classes list it after actionAngleInverse, whose
# metaclass looks up the public cousins of _evaluate and friends on the
# first base
class _IsochroneAuxiliary:
    """The closed forms of the isochrone auxiliary's tori, for an inverse
    transformation that lifts its tori onto a frozen isochrone of mass _GM and
    scale _b (both set by the subclass's _setup_auxiliary): energy and radial
    action, the orbital parameters (a, e) and their derivatives along the
    actions, and the radius, momentum, flux and cumulative radial action in
    the eccentric anomaly with their partials with respect to (a, e)"""

    def _auxiliary_E(self, Jr, L):
        """Energy of the auxiliary torus with actions (J_r, L)"""
        CA = 0.5 * (L + numpy.sqrt(L**2 + 4.0 * self._GM * self._b))
        return -(self._GM**2) / (2.0 * (Jr + CA) ** 2)

    def _auxiliary_Jr(self, E, L):
        """Radial action of the auxiliary torus with energy E and angular
        momentum L (the inverse of _auxiliary_E)"""
        CA = 0.5 * (L + numpy.sqrt(L**2 + 4.0 * self._GM * self._b))
        return self._GM / numpy.sqrt(-2.0 * E) - CA

    def _auxiliary_orbital_params(self, Jr, L):
        """Closed-form (a, e) of the auxiliary torus with actions (J_r, L):
        a + b = (J_r + C_A)^2 / GM, and e^2 = 1 - L^2 (a + b) / (GM a^2)
        evaluated as the product J_r (J_r + s) [GM a + L (J_r + C_A)] /
        (GM a)^2 with s = sqrt(L^2 + 4 GM b), the same quantity with its
        cancellation at small J_r removed, so that e ~ sqrt(J_r) down to
        round-off and vanishes exactly at the circular orbit"""
        GM, b = self._GM, self._b
        s = numpy.sqrt(L**2 + 4.0 * GM * b)
        CA = 0.5 * (L + s)
        a = (Jr + CA) ** 2 / GM - b
        e = numpy.sqrt(Jr * (Jr + s) * (GM * a + L * (Jr + CA))) / (GM * a)
        return a, e

    def _auxiliary_orbital_param_chains(self, Jr, L, dJr, dL):
        """(a, e) of the auxiliary torus and their derivatives along a
        chain with dJr = dJ_r/dalpha and dL = dL/dalpha, all closed forms"""
        GM, b = self._GM, self._b
        sq = numpy.sqrt(L**2 + 4.0 * GM * b)
        CA = 0.5 * (L + sq)
        EA = -(GM**2) / (2.0 * (Jr + CA) ** 2)
        dEA = GM**2 / (Jr + CA) ** 3 * (dJr + 0.5 * (1.0 + L / sq) * dL)
        a, e = self._auxiliary_orbital_params(Jr, L)
        da = GM / (2.0 * EA**2) * dEA
        de = (
            2.0 * L * dL / (2.0 * EA * a**2)
            - L**2 * (dEA * a + 2.0 * EA * da) / (2.0 * EA**2 * a**3)
        ) / (2.0 * e)
        return a, e, da, de

    def _auxiliary_profile(self, a, e, eta):
        """The auxiliary torus's radius, radial momentum, and dr^A/deta with
        its vanishing factor sin(eta) removed, at eccentric anomaly eta:
        sqrt(b^2 + r^A^2) = b + a (1 - e cos eta), so that with
        y = 1 - e cos eta and beta = b/a, r^A = a sqrt(y (y + 2 beta)),
        p^A = sqrt(GM/(a+b)) a e sin(eta) / r^A and dr^A/deta = g^A sin(eta)
        with g^A = a e (y + beta) / sqrt(y (y + 2 beta)), all closed forms"""
        beta = self._b / a
        y = 1.0 - e * numpy.cos(eta)
        sq = numpy.sqrt(y * (y + 2.0 * beta))
        rA = a * sq
        pA = numpy.sqrt(self._GM / (a + self._b)) * a * e * numpy.sin(eta) / rA
        gA = a * e * (y + beta) / sq
        return rA, pA, gA

    def _auxiliary_radius_partials(self, a, e, eta):
        """Partials of the auxiliary radius with respect to its torus
        parameters (a, e) at fixed eccentric anomaly, from r^A^2 =
        a^2 y (y + 2 b/a) with y = 1 - e cos eta"""
        y = 1.0 - e * numpy.cos(eta)
        sA = self._b + a * y  # = sqrt(b^2 + r^A^2)
        rA = numpy.sqrt(sA**2 - self._b**2)
        return y * sA / rA, -a * sA * numpy.cos(eta) / rA

    def _auxiliary_flux_derivs(self, a, e, eta):
        """The auxiliary's radial action flux f^A = p^A dr^A/deta at
        eccentric anomaly eta, and its partials with respect to the torus
        parameters (a, e) at fixed eta, all closed forms: with beta = b/a,
        y = 1 - e cos(eta), c = sqrt(GM/(a+b)) and g(y) = (y + beta) /
        [y (y + 2 beta)], f^A = c a e^2 sin^2(eta) g"""
        b = self._b
        beta = b / a
        y = 1.0 - e * numpy.cos(eta)
        c = numpy.sqrt(self._GM / (a + b))
        s2 = numpy.sin(eta) ** 2
        g = (y + beta) / (y * (y + 2.0 * beta))
        dg_dy = -(y**2 + 2.0 * beta * y + 2.0 * beta**2) / (y * (y + 2.0 * beta)) ** 2
        dg_da = (b / a**2) / (y + 2.0 * beta) ** 2
        fA = c * a * e**2 * s2 * g
        dfA_da = e**2 * s2 * (c * (a + 2.0 * b) / (2.0 * (a + b)) * g + c * a * dg_da)
        dfA_de = c * a * s2 * (2.0 * e * g - e**2 * dg_dy * numpy.cos(eta))
        return fA, dfA_da, dfA_de

    def _auxiliary_cumulative(self, a, e, eta):
        """The auxiliary's cumulative radial action from pericentre,
        A^A(eta) = int_0^eta f^A, and its partials with respect to the torus
        parameters (a, e) at fixed eta, all closed forms: with beta = b/a and
        c = sqrt(GM/(a+b)), the flux's factor (y + beta)/[y (y + 2 beta)] is
        [1/y + 1/(y + 2 beta)]/2, and each term integrates to
        c a [e sin(eta)/2 + A eta/2 - sqrt(A^2 - e^2) Theta_A(eta)] with A = 1
        and A = 1 + 2 beta, where Theta_A = arctan(sqrt((A+e)/(A-e))
        tan(eta/2)), continued through its branches by the winding number of
        eta -- the two arctangents of the isochrone's angle relation. The
        secular rate over a libration is the auxiliary's radial action.

        The sum is O(eta^3) at pericentre while its terms are O(eta), and
        their cancellation would leave the round-off of the largest term,
        (1 + beta) eta, on a cumulative action e^2 times smaller; the anomaly
        that the matching's Newton finds on this function inherits that
        error divided by the flux, and the map's derivative amplifies it by
        the number of harmonics. So the linear parts are gathered
        analytically: Theta_A = eta/2 + phi_A with phi_A = arctan(k_A t) -
        arctan(t) = arctan[(k_A - 1) t/(1 + k_A t^2)] (t = tan(eta/2), no
        branch to continue), the coefficient of eta that remains, lambda =
        1 + beta - sum_A sqrt(A^2 - e^2)/2 = e^2/2 sum_A 1/(A + sqrt(A^2 -
        e^2)), and k_A - 1 = 2 e/[(A - e)(k_A + 1)] are formed without
        cancellation, and every term is O(e)."""
        b = self._b
        c = numpy.sqrt(self._GM / (a + b))
        beta = b / a
        eta = numpy.asarray(eta, dtype="float")
        t = numpy.tan(0.5 * eta)
        dca_da = c * (a + 2.0 * b) / (2.0 * (a + b))
        dbeta_da = -b / a**2
        bracket = e * numpy.sin(eta)
        dbr_de = numpy.sin(eta)
        dbr_da = numpy.zeros_like(eta)
        lam, dlam_de, dlam_da = 0.0, 0.0, 0.0
        for A, dA_da in ((1.0, 0.0), (1.0 + 2.0 * beta, 2.0 * dbeta_da)):
            k = numpy.sqrt((A + e) / (A - e))
            sq = numpy.sqrt(A**2 - e**2)
            km1 = 2.0 * e / ((A - e) * (k + 1.0))
            phi = numpy.arctan(km1 * t / (1.0 + k * t**2))
            dphi_dk = t / (1.0 + k**2 * t**2)
            dk_de = A / (k * (A - e) ** 2)
            dk_dA = -e / (k * (A - e) ** 2)
            lam = lam + 0.5 * e**2 / (A + sq)
            dlam_de = dlam_de + 0.5 * e / sq
            dlam_da = dlam_da - 0.5 * e**2 * dA_da / (sq * (A + sq))
            bracket = bracket - sq * phi
            dbr_de = dbr_de + (e / sq) * phi - sq * dphi_dk * dk_de
            dbr_da = dbr_da - (A / sq) * dA_da * phi - sq * dphi_dk * dk_dA * dA_da
        bracket = bracket + lam * eta
        dbr_de = dbr_de + dlam_de * eta
        dbr_da = dbr_da + dlam_da * eta
        return c * a * bracket, dca_da * bracket + c * a * dbr_da, c * a * dbr_de
