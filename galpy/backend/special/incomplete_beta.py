###############################################################################
#   galpy.backend.special.incomplete_beta: the jax/torch incomplete beta
#   B_z(p, q) of galpy.util.special.incomplete_beta (p > 0, q > -1), with
#   static series coefficients so it is differentiable and jit-compatible.
###############################################################################
import functools
import math

import numpy
from scipy import special

from ...util.special import (
    _IBETA_QSMALL,
    incomplete_beta_at_split,
    incomplete_beta_k_series,
    incomplete_beta_split,
)
from .. import asarray_on_device, branch_where, device_of

# Backend (jax/torch) versions of galpy.util.special's incomplete beta: p and q
# are fixed (TwoPowerSphericalPotential: at construction), so every series has static coefficients, summed as
# sum_n sign_n exp(log|c_n| + n log v) (no overflow of the Pochhammer ratios at
# large beta) up to a static length set by the largest argument it sees.
_SERIES_TOL = 1e-17
_SERIES_NMAX = 20000


@functools.lru_cache(maxsize=256)  # alpha, beta vary freely (fits): bounded
def incomplete_beta_series_coeffs(kind, p, q, vmax):
    """(sign, log|c_n|, n) of sum_n c_n v^n, truncated where its tail at
    v = vmax is below _SERIES_TOL of the sum: kind 'lo' is 2F1(1, p+q; p+1;
    v), 'hi' 2F1(1, p+q; q+1; v), 'k' the incomplete_beta_k_series sum (without its
    (1-v)^p)."""
    b, c = p + q, (p + 1.0 if kind == "lo" else q + 1.0)
    sgn, loga, ns = [], [], []
    sg, la, D, tot = 1.0, 0.0, 0.0, 0.0
    lv = math.log(vmax)
    for n in range(_SERIES_NMAX):
        if kind == "k":
            if n > 0:  # (p)_n/n! D_n, D_n as in incomplete_beta_k_series
                la += math.log((p + n - 1.0) / n)
                D = D * (1.0 + q / (p + n - 1.0)) / (1.0 + q / n) + (1.0 - p) / (
                    (p + n - 1.0) * (n + q)
                )
                e = D
                cn_sgn, cn_la = (
                    (math.copysign(1.0, e), la + math.log(abs(e))) if e else (0.0, 0.0)
                )
            else:
                cn_sgn, cn_la = 0.0, 0.0
        else:
            if n > 0:  # (b)_n/(c)_n
                if b + n - 1.0 == 0.0:
                    break  # the series terminates
                sg *= math.copysign(1.0, b + n - 1.0)
                la += math.log(abs(b + n - 1.0) / (c + n - 1.0))
            cn_sgn, cn_la = sg, la
        if cn_sgn:
            sgn.append(cn_sgn)
            loga.append(cn_la)
            ns.append(float(n))
            term = math.exp(cn_la + n * lv)
            tot += cn_sgn * term
            if n > 5 and ns[-2] == n - 1.0:
                rat = max(term / math.exp(loga[-2] + (n - 1) * lv), vmax)
                if rat < 1.0 and term * rat / (1.0 - rat) < _SERIES_TOL * abs(tot):
                    break
    return numpy.array(sgn), numpy.array(loga), numpy.array(ns)


def incomplete_beta_series_xp(xp, coeffs, v):
    dev = device_of(v)
    sgn, loga, ns = (asarray_on_device(xp, c, dev, dtype=v.dtype) for c in coeffs)
    return xp.sum(sgn * xp.exp(loga + ns * xp.log(v)[..., None]), axis=-1)


def incomplete_beta_xp(xp, p, q, z, s):
    """incomplete_beta for backend z, s (the same split and pieces; masked so that
    either branch is finite everywhere)."""
    c = incomplete_beta_split(p, q)
    s2 = 1.0 - c
    lo = z <= c
    one = xp.ones_like(z * 1.0)

    def below():
        zl = xp.where(lo, z, 0.5 * c * one)
        sl = xp.where(lo, s, (1.0 - 0.5 * c) * one)
        return (
            zl**p
            * sl**q
            / p
            * incomplete_beta_series_xp(
                xp, incomplete_beta_series_coeffs("lo", p, q, c), zl
            )
        )

    def above():
        s1 = xp.where(lo, 0.5 * s2 * one, s)
        z1 = xp.where(lo, (1.0 - 0.5 * s2) * one, z)  # 1 - s1, exactly
        return incomplete_beta_hi_xp(xp, p, q, s1, z1, c)

    return branch_where(xp, lo, below, above)


def incomplete_beta_hi_xp(xp, p, q, s1, z1, c):
    """incomplete_beta_hi for backend s1 > 1 - c (z1 = 1 - s1)"""
    s2 = 1.0 - c
    ibc = incomplete_beta_at_split(p, q, c)
    if abs(q) < _IBETA_QSMALL:
        return _incomplete_beta_reflected_smallq_xp(xp, p, q, s1, z1, s2, ibc)
    if abs(q + 1.0) < _IBETA_QSMALL:  # as in incomplete_beta_hi
        B2 = float(s2**q * (1.0 - s2) ** p / q)
        return (
            ibc
            + B2
            - s1**q * z1**p / q
            + (p + q)
            / q
            * _incomplete_beta_reflected_smallq_xp(xp, p, q + 1.0, s1, z1, s2, 0.0)
        )
    B2 = float(s2**q * (1.0 - s2) ** p / q * special.hyp2f1(1.0, p + q, q + 1.0, s2))
    return (
        ibc
        + B2
        - s1**q
        * z1**p
        / q
        * incomplete_beta_series_xp(
            xp, incomplete_beta_series_coeffs("hi", p, q, s2), s1
        )
    )


def _incomplete_beta_reflected_smallq_xp(xp, p, q, s1, z1, s2, base):
    """_incomplete_beta_reflected_smallq for backend s1 (z1 = 1 - s1)"""
    K2 = float(incomplete_beta_k_series(p, q, s2))
    K1 = z1**p * incomplete_beta_series_xp(
        xp, incomplete_beta_series_coeffs("k", p, q, s2), s1
    )
    lg = xp.log(s2 / s1)
    if q == 0.0:
        first = lg
    else:
        # (s2^q - s1^q)/q; the expm1 form only where it cannot overflow
        qlg = q * lg
        ok = xp.abs(qlg) < 1.0
        first = xp.where(
            ok,
            s1**q * xp.expm1(xp.where(ok, qlg, 0.0 * qlg)) / q,
            (s2**q - s1**q) / q,
        )
    return base + first * (1.0 + q * K2) + s1**q * (K2 - K1)
