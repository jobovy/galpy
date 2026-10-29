###############################################################################
# test_potential_numpy_byte_identity.py: the numpy path of potentials whose
# formulas were restructured for the backends evaluates exactly (bitwise) the
# pre-backend formulas, i.e. in the same floating-point operation order.
# Each test writes the pre-backend formula out in full and compares with ==.
###############################################################################
import numpy
import pytest

from galpy.potential import (
    BurkertPotential,
    NFWPotential,
    PerfectEllipsoidPotential,
    TriaxialNFWPotential,
)
from galpy.potential.BurkertPotential import _BURKERT_NTERMS
from galpy.potential.TwoPowerSphericalPotential import _NFW_S

# the numpy path's own contract: exempt from --backend
pytestmark = pytest.mark.backend_managed

_R = numpy.geomspace(0.003, 20.0, 60)
_z = numpy.linspace(-2.0, 2.0, 7)
_RR, _ZZ = numpy.meshgrid(_R, _z, indexing="ij")


def _nfw_h_ref(x):
    # the pre-backend h(x): 2 (t^2/(1+t) + S(t)), S by Horner then * t^2 * t
    t = x / (2.0 + x)
    t2 = t * t
    S = _NFW_S[-1]
    for c in _NFW_S[-2::-1]:
        S = S * t2 + c
    S = S * t2 * t
    return 2.0 * (t * t / (1.0 + t) + S)


def _nfw_hk_ref(x):
    t = x / (2.0 + x)
    t2 = t * t
    S = _NFW_S[-1]
    for c in _NFW_S[-2::-1]:
        S = S * t2 + c
    S = S * t2 * t
    u = t / (1.0 + t)
    return 2.0 * (t * u + S), -4.0 * (t * u * u + S)


def test_nfw_small_radius_bitwise():
    a = 1.7
    p = NFWPotential(amp=2.0, a=a)
    r = numpy.sqrt(_RR**2 + _ZZ**2)
    small = r < 0.25 * a
    # the potential: -log1p(r/a)/r (not -(1/r) log1p(r/a))
    assert numpy.array_equal(
        p._evaluate(_RR, _ZZ)[small], (-numpy.log1p(r / a) / r)[small]
    )
    # the radial force: R * (-h(x) / r^3)
    assert numpy.array_equal(
        p._Rforce(_RR, _ZZ)[small], (_RR * (-_nfw_h_ref(r / a) / (r * r * r)))[small]
    )
    # the second derivatives: (k R^2 + h z^2) / r^5 and (k - h) R z / r^5
    h, k = _nfw_hk_ref(r / a)
    r5 = r * r * r * r * r
    assert numpy.array_equal(
        p._R2deriv(_RR, _ZZ)[small], ((k * _RR * _RR + h * _ZZ * _ZZ) / r5)[small]
    )
    assert numpy.array_equal(
        p._Rzderiv(_RR, _ZZ)[small], ((k - h) * _RR * _ZZ / r5)[small]
    )
    # the enclosed mass: h(x) itself
    assert numpy.array_equal(
        p._mass(_R)[_R < 0.25 * a], _nfw_h_ref(_R / a)[_R < 0.25 * a]
    )


def test_burkert_small_radius_bitwise():
    a = 1.3
    p = BurkertPotential(amp=1.5, a=a)
    r = numpy.sqrt(_RR**2 + _ZZ**2)
    small = r < 0.25 * a
    # the force series, Horner in y = x^4 with x-dependent coefficients
    x = r / a
    y = x**4.0
    series = -4.0 / (4 * _BURKERT_NTERMS - 1) + x / _BURKERT_NTERMS
    for j in range(_BURKERT_NTERMS - 2, -1, -1):
        series = series * y + (-4.0 / (4 * j + 3) + x / (j + 1))
    rforce = numpy.pi * r * series
    assert numpy.array_equal(p._rforce(r)[small], rforce[small])
    assert numpy.array_equal(p._Rforce(_RR, _ZZ)[small], (rforce * _RR / r)[small])
    # Phi'' - Phi'/r series for the mixed derivative
    out = 4.0 * (_BURKERT_NTERMS - 1) / (4.0 * _BURKERT_NTERMS - 1.0) - x * (
        4.0 * _BURKERT_NTERMS - 3.0
    ) / (4.0 * _BURKERT_NTERMS)
    for j in range(_BURKERT_NTERMS - 2, -1, -1):
        out = out * y + (
            4.0 * j / (4.0 * j + 3.0) - x * (4.0 * j + 1.0) / (4.0 * j + 4.0)
        )
    assert numpy.array_equal(
        p._Rzderiv(_RR, _ZZ)[small],
        (_RR * _ZZ / r**2.0 * (4.0 * numpy.pi * out))[small],
    )


def _second_derivs_ref(p, x, y, z):
    # the pre-backend Gauss-Legendre loop: `xx += a + b`, i.e. xx + (a + b)
    b2, c2 = p._b2, p._c2
    xx = xy = xz = yy = yz = zz = 0.0
    x2, y2, z2 = x**2, y**2, z**2
    for k in range(len(p._glx)):
        s = p._glx[k]
        w = p._glw[k]
        t = 1.0 / s**2 - 1.0
        inv1t = 1.0 / (1.0 + t)
        invb2t = 1.0 / (b2 + t)
        invc2t = 1.0 / (c2 + t)
        m = numpy.sqrt(x2 * inv1t + y2 * invb2t + z2 * invc2t)
        denom = numpy.sqrt((1.0 + (b2 - 1.0) * s**2) * (1.0 + (c2 - 1.0) * s**2))
        w_over_denom = w / denom
        dens_val = p._mdens(m)
        dderiv_over_m = p._mdens_deriv(m) / m
        xi = x * inv1t
        yi = y * invb2t
        zi = z * invc2t
        dd_xi = w_over_denom * dderiv_over_m * xi
        dd_yi = w_over_denom * dderiv_over_m * yi
        dd_zi = w_over_denom * dderiv_over_m * zi
        xx += dd_xi * xi + w_over_denom * dens_val * inv1t
        xy += dd_xi * yi
        xz += dd_xi * zi
        yy += dd_yi * yi + w_over_denom * dens_val * invb2t
        yz += dd_yi * zi
        zz += dd_zi * zi + w_over_denom * dens_val * invc2t
    prefac = 4.0 * numpy.pi * p._b * p._c
    return tuple(prefac * v for v in (xx, xy, xz, yy, yz, zz))


def test_ellipsoidal_second_derivatives_bitwise():
    for p in (
        TriaxialNFWPotential(amp=1.2, a=1.8, b=0.8, c=0.6),
        PerfectEllipsoidPotential(amp=0.7, a=1.1, b=0.9, c=0.5),
    ):
        x, y, z = _RR * numpy.cos(0.7), _RR * numpy.sin(0.7), _ZZ
        xx, xy, xz, yy, yz, zz = _second_derivs_ref(p, x, y, z)
        c, s = numpy.cos(0.7), numpy.sin(0.7)
        assert numpy.array_equal(
            p._R2deriv(_RR, _ZZ, phi=0.7), c**2.0 * xx + s**2.0 * yy + 2.0 * c * s * xy
        )
        assert numpy.array_equal(p._z2deriv(_RR, _ZZ, phi=0.7), zz)
