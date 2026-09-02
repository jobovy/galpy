import numpy
import pytest

from galpy.actionAngle import actionAngleStaeckel
from galpy.actionAngle.actionAngleStaeckelInverse import actionAngleStaeckelInverse
from galpy.actionAngle.actionAngleTorusStaeckel import actionAngleTorusStaeckel
from galpy.orbit import Orbit
from galpy.potential import (
    KuzminKutuzovStaeckelPotential,
    OblateStaeckelWrapperPotential,
    evaluatePotentials,
)

# A KuzminKutuzov potential is exactly Staeckel, so the auxiliary already
# reproduces it and only the family's own interpolation error remains for the
# Fourier layer to remove -- small enough that tiny families run in seconds,
# which is what the tests below use.
_KKP = KuzminKutuzovStaeckelPotential(amp=4.0, ac=5.0, Delta=1.3)


def _kk_torus():
    aAS = actionAngleStaeckel(pot=_KKP, delta=1.3, c=True, order=100)
    return tuple(
        float(numpy.atleast_1d(x)[0]) for x in aAS(1.1, 0.25, 1.1, 0.15, 0.15, 0.0)
    )


def test_actionAngleTorusStaeckel_guards():
    # the constructor's required arguments and the alias-free lattice bound
    with pytest.raises(OSError, match="pot="):
        actionAngleTorusStaeckel()
    with pytest.raises(OSError, match="delta="):
        actionAngleTorusStaeckel(pot=_KKP)
    with pytest.raises(ValueError, match="alias-free"):
        actionAngleTorusStaeckel(
            pot=_KKP,
            delta=1.3,
            ngrid=8,
            maxn=4,
            setup_interp=True,
            Rmin=0.7,
            Rmax=1.6,
            Rinf=8.0,
            nLz=4,
            nE=4,
            nI3=4,
        )
    return None


def test_actionAngleTorusStaeckel_family_uses_true_potential():
    # given an already-built family on a WRAPPED potential, the mapper must
    # flatten the TRUE (raw) Hamiltonian, not the Staeckel model's -- the
    # model is trivially flat on its own tori, so getting this wrong is a
    # silent no-op
    swp = OblateStaeckelWrapperPotential(pot=_KKP, delta=1.3)
    fam = actionAngleStaeckelInverse(
        pot=swp,
        setup_interp=True,
        Rmin=0.7,
        Rmax=1.6,
        Rinf=8.0,
        nLz=4,
        nE=4,
        nI3=4,
    )
    tm = actionAngleTorusStaeckel(family=fam, ngrid=8, maxn=3)
    assert tm._pot is _KKP, (
        "the mapper's true potential is the wrapper, not the raw potential"
    )
    return None


def test_actionAngleTorusStaeckel_flatten_and_symmetry():
    # the fitter reduces the Hamiltonian's variation on the torus, and for a
    # time-reversal-symmetric potential the sine coefficients vanish (the
    # generating function is a pure cosine series)
    jr, lz, jz = _kk_torus()
    tm = actionAngleTorusStaeckel(
        pot=_KKP,
        delta=1.3,
        ngrid=8,
        maxn=3,
        polish=1,
        starfrac=(0.1, 0.008, 0.1),
        Rmin=0.7,
        Rmax=1.6,
        Rinf=8.0,
        nLz=4,
        nE=4,
        nI3=4,
    )
    f = tm._flatten(jr, lz, jz)
    assert f["flat"] < 0.5 * f["flat0"], (
        "the fitter did not flatten the Hamiltonian: {:g} -> {:g}".format(
            f["flat0"], f["flat"]
        )
    )
    assert numpy.fabs(f["B"]).max() < 1e-10, (
        "the sine coefficients did not vanish for a symmetric potential"
    )
    assert f["skipped"] == 0.0, "an off-resonance torus skipped modes"
    # the polish keeps the BEST pass, so more iterations never worsen the
    # result even though the approximate Gauss-Newton step can drift past
    # the floor: many-pass flat must not exceed few-pass flat
    tm_few = actionAngleTorusStaeckel(
        pot=_KKP,
        delta=1.3,
        ngrid=8,
        maxn=3,
        polish=1,
        starfrac=(0.1, 0.008, 0.1),
        Rmin=0.7,
        Rmax=1.6,
        Rinf=8.0,
        nLz=4,
        nE=4,
        nI3=4,
    )
    tm_many = actionAngleTorusStaeckel(
        pot=_KKP,
        delta=1.3,
        ngrid=8,
        maxn=3,
        polish=6,
        starfrac=(0.1, 0.008, 0.1),
        Rmin=0.7,
        Rmax=1.6,
        Rinf=8.0,
        nLz=4,
        nE=4,
        nI3=4,
    )
    assert (
        tm_many._flatten(jr, lz, jz)["flat"]
        <= tm_few._flatten(jr, lz, jz)["flat"] + 1e-12
    ), "more polish iterations worsened the flatness (best-pass not kept)"
    return None


def test_actionAngleTorusStaeckel_call_and_orbit():
    # the map returns finite (x, v) and reproduces a short integrated orbit:
    # map at theta0 + Omega t must track the true trajectory
    jr, lz, jz = _kk_torus()
    tm = actionAngleTorusStaeckel(
        pot=_KKP,
        delta=1.3,
        ngrid=12,
        maxn=4,
        polish=2,
        starfrac=(0.1, 0.008, 0.1),
        Rmin=0.7,
        Rmax=1.6,
        Rinf=8.0,
        nLz=5,
        nE=5,
        nI3=5,
    )
    th0 = numpy.array([0.7, 0.3, 1.9])
    Om = tm.Freqs(jr, lz, jz)
    assert all(numpy.isfinite(Om)) and Om[0] > 0.0 and Om[2] > 0.0, (
        f"the frequencies are not positive and finite: {Om}"
    )
    o0 = tm(jr, lz, jz, *(numpy.array([t]) for t in th0))
    assert all(numpy.isfinite(numpy.atleast_1d(q)[0]) for q in o0), (
        "the map returned a non-finite phase-space point"
    )
    orb = Orbit([float(numpy.atleast_1d(q)[0]) for q in o0])
    ts = numpy.linspace(0.0, 6.0, 61)
    orb.integrate(ts, _KKP, method="dop853_c")
    dx = []
    for t in (2.0, 4.0, 6.0):
        pt = tm(
            jr,
            lz,
            jz,
            numpy.array([th0[0] + Om[0] * t]),
            numpy.array([th0[1] + Om[1] * t]),
            numpy.array([th0[2] + Om[2] * t]),
        )
        dx.append(numpy.hypot(pt[0][0] - orb.R(t), pt[3][0] - orb.z(t)))
    # a coarse family/lattice on KK: the interpolation floor limits this to
    # ~1e-3, which is still far below the O(1) scale of the orbit
    assert max(dx) < 5e-3, "the torus map does not track the integrated orbit: %s" % dx
    # array angles evaluate in one call and match the scalar path
    thr = numpy.array([0.2, 1.3, 2.9])
    many = tm(jr, lz, jz, thr, numpy.zeros(3), thr)
    one = tm(
        jr, lz, jz, numpy.array([thr[1]]), numpy.array([0.0]), numpy.array([thr[1]])
    )
    assert numpy.fabs(many[0][1] - one[0][0]) < 1e-10, (
        "array and scalar angle evaluation disagree"
    )
    # generous polish on a well-modeled torus converges before it is spent:
    # the fit stops once a full step no longer improves the flatness by 10%
    conv = tm._flatten(jr, lz, jz)
    assert conv["flat"] < conv["flat0"], "the converging fit did not improve"
    return None


def test_actionAngleTorusStaeckel_u0_and_xvFreqs_and_guard():
    # u0= passes through to the family build; xvFreqs returns the map plus
    # the frequencies; and an edge (J_R = 0 or J_z = 0) torus is refused
    jr, lz, jz = _kk_torus()
    tm = actionAngleTorusStaeckel(
        pot=_KKP,
        delta=1.3,
        u0=1.1,
        ngrid=8,
        maxn=3,
        polish=1,
        starfrac=(0.1, 0.008, 0.1),
        Rmin=0.7,
        Rmax=1.6,
        Rinf=8.0,
        nLz=4,
        nE=4,
        nI3=4,
    )
    assert numpy.fabs(tm._fam._staeckelwrap._u0 - 1.1) < 1e-12, (
        "u0= did not reach the family build"
    )
    out = tm.xvFreqs(
        jr, lz, jz, numpy.array([0.4]), numpy.array([0.2]), numpy.array([1.0])
    )
    assert len(out) == 9 and all(numpy.isfinite(numpy.atleast_1d(q)[0]) for q in out), (
        "xvFreqs did not return six coordinates and three frequencies"
    )
    with pytest.raises(ValueError, match="interior torus"):
        tm.Freqs(0.0, lz, jz)
    with pytest.raises(ValueError, match="interior torus"):
        tm(jr, lz, 0.0, numpy.array([0.1]), numpy.array([0.1]), numpy.array([0.1]))
    # u0='fit' builds the adaptive (varying reference curve) family
    tmf = actionAngleTorusStaeckel(
        pot=_KKP,
        delta=1.3,
        u0="fit",
        ngrid=8,
        maxn=3,
        polish=0,
        starfrac=(0.1, 0.008, 0.1),
        Rmin=0.7,
        Rmax=1.6,
        Rinf=8.0,
        nLz=4,
        nE=4,
        nI3=4,
    )
    assert tmf._fam._u0_func is not None, "u0='fit' did not build an adaptive family"
    return None


def test_actionAngleTorusStaeckel_resonance_skip():
    # a deliberately loose resonance tolerance forces low-order modes below
    # the divisor threshold to be skipped, which accumulates skipped power
    # and raises the diagnostic warning (the small-divisor policy)
    jr, lz, jz = _kk_torus()
    tm = actionAngleTorusStaeckel(
        pot=_KKP,
        delta=1.3,
        ngrid=8,
        maxn=3,
        polish=0,
        resonance_tol=0.5,
        starfrac=(0.1, 0.008, 0.1),
        Rmin=0.7,
        Rmax=1.6,
        Rinf=8.0,
        nLz=4,
        nE=4,
        nI3=4,
    )
    with pytest.warns(Warning, match="near-resonant"):
        f = tm._flatten(jr, lz, jz)
    assert f["skipped"] > 0.0, "no power was skipped at a large resonance_tol"
    return None


def test_actionAngleTorusStaeckel_eccentric_trust_region():
    # a DELIBERATELY POOR chart (KK modeled at delta=0.7 instead of its true
    # 1.3) makes the residual large -- the regime where the perturbative
    # solve overshoots -- so the trust region must clip the correction to
    # keep J^S inside the physical domain (J_R + dJ_R, J_z + dJ_z > 0)
    jr, lz, jz = (
        float(numpy.atleast_1d(x)[0])
        for x in actionAngleStaeckel(pot=_KKP, delta=1.3, c=True, order=100)(
            1.1, 0.30, 1.05, 0.10, 0.12, 0.0
        )
    )
    tm = actionAngleTorusStaeckel(
        pot=_KKP,
        delta=0.7,
        ngrid=10,
        maxn=3,
        polish=3,
        starfrac=(0.1, 0.008, 0.1),
        Rmin=0.6,
        Rmax=1.8,
        Rinf=8.0,
        nLz=4,
        nE=4,
        nI3=4,
    )
    f = tm._flatten(jr, lz, jz)
    assert f["nclip"] > 0, "the trust region never engaged on a poor chart"
    return None


def test_actionAngleTorusStaeckel_canonical_model():
    # the map is exactly symplectic for the model it evaluates: with the
    # coefficients and their derivatives both coming from the stored
    # quadratic model, dx/dtheta . dp/dJ - dx/dJ . dp/dtheta = I (the
    # standard symplectic-defect check on the composite map)
    jr, lz, jz = _kk_torus()
    tm = actionAngleTorusStaeckel(
        pot=_KKP,
        delta=1.3,
        ngrid=10,
        maxn=3,
        polish=1,
        starfrac=(0.1, 0.008, 0.1),
        Rmin=0.7,
        Rmax=1.6,
        Rinf=8.0,
        nLz=5,
        nE=5,
        nI3=5,
    )
    # prime the torus cache so the finite differences below reuse one model
    tm.Freqs(jr, lz, jz)
    th = numpy.array([0.6, 0.0, 1.7])

    def xp(dr, dL, dz, dthr, dthz):
        o = tm(
            jr + dr,
            lz + dL,
            jz + dz,
            numpy.array([th[0] + dthr]),
            numpy.array([th[1]]),
            numpy.array([th[2] + dthz]),
        )
        R, vR, vT, z, vz = (float(numpy.atleast_1d(q)[0]) for q in o[:5])
        # planar (R, z, p_R, p_z) block; p_R = vR, p_z = vz
        return numpy.array([R, z, vR, vz])

    h = 1e-6
    # Jacobian of (R, z, pR, pz) wrt (theta_r, theta_z, J_r, J_z)
    cols = []
    for arg in ("thr", "thz", "jr", "jz"):
        kw = dict(dr=0.0, dL=0.0, dz=0.0, dthr=0.0, dthz=0.0)
        key = {"thr": "dthr", "thz": "dthz", "jr": "dr", "jz": "dz"}[arg]
        kw[key] = h
        p1 = xp(**kw)
        kw[key] = -h
        m1 = xp(**kw)
        cols.append((p1 - m1) / (2.0 * h))
    Jm = numpy.array(cols).T  # (4 coords) x (theta_r, theta_z, J_r, J_z)
    # symplectic form: {R, pR} = {z, pz} = 1, others 0, in (theta, J) coords
    # M^T Omega M = Omega  <=>  the Poisson brackets of the coords are canonical
    dR, dz, dpR, dpz = Jm
    # {R, pR} + {z, pz} evaluated as dq/dtheta . dp/dJ - dq/dJ . dp/dtheta
    br = (
        dR[0] * dpR[2]
        + dR[1] * dpR[3]
        - dR[2] * dpR[0]
        - dR[3] * dpR[1]
        + dz[0] * dpz[2]
        + dz[1] * dpz[3]
        - dz[2] * dpz[0]
        - dz[3] * dpz[1]
    )
    assert numpy.fabs(br - 2.0) < 1e-3, (
        "the composite map is not symplectic: sum of brackets %g != 2" % br
    )
    return None


def test_actionAngleTorusStaeckel_family_freq_grid():
    # the family E(J) frequency route: one flatten per node of a local
    # action grid and the gradient of a least-squares polynomial. On the
    # exactly-Staeckel KuzminKutuzov potential it must agree with the
    # forward frequencies; the degree rule, the cache, the node-failure
    # guard, and the interior-torus guard are exercised white-box
    jr, lz, jz = _kk_torus()
    tm = actionAngleTorusStaeckel(
        pot=_KKP,
        delta=1.3,
        ngrid=8,
        maxn=3,
        polish=1,
        starfrac=(0.1, 0.008, 0.1),
        Rmin=0.7,
        Rmax=1.6,
        Rinf=8.0,
        nLz=5,
        nE=5,
        nI3=5,
    )
    # a compact grid keeps the test fast; the production one is just more
    # nodes of the same construction
    tm._freqgrid_frel = (0.85, 1.0, 1.18)
    tm._freqgrid_frelL = (0.995, 1.0, 1.005)
    tm._freqgrid_minnodes = 20
    aAS = actionAngleStaeckel(pot=_KKP, delta=1.3, c=True, order=100)
    OmT = aAS.actionsFreqs(1.1, 0.25, 1.1, 0.15, 0.15, 0.0)[3:]
    OmT = [float(numpy.atleast_1d(q)[0]) for q in OmT]
    Om = tm.Freqs(jr, lz, jz)
    for k in range(3):
        assert numpy.fabs(Om[k] - OmT[k]) / numpy.fabs(OmT[k]) < 3e-3, (
            "family E(J) frequency %i disagrees with the forward Staeckel "
            "frequency: %g vs %g" % (k, Om[k], OmT[k])
        )
    # the fit is cached per torus
    fg = tm._fit_freq_grid(jr, lz, jz)
    assert fg is tm._fit_freq_grid(jr, lz, jz), (
        "the frequency grid was refit instead of cached"
    )
    # the degree rule follows the center flatness against the threshold:
    # drive it both ways and check the two fits agree on this clean torus
    tm._freqgrid_cubic_thresh = 1.0
    tm._torus_cache.clear()
    Om = tm.Freqs(jr, lz, jz)
    assert tm._fit_freq_grid(jr, lz, jz)["deg"] == 3, (
        "a permissive threshold did not select the cubic fit"
    )
    tm._freqgrid_cubic_thresh = 0.0
    tm._torus_cache.clear()
    Om2 = tm.Freqs(jr, lz, jz)
    assert tm._fit_freq_grid(jr, lz, jz)["deg"] == 2, (
        "a zero threshold did not select the quadratic fit"
    )
    for k in range(3):
        assert numpy.fabs(Om2[k] - Om[k]) / numpy.fabs(Om[k]) < 1e-2, (
            "quadratic and cubic family frequencies disagree on a clean torus"
        )
    # method='star' remains available and agrees on this benign torus
    OmS = tm.Freqs(jr, lz, jz, method="star")
    for k in range(3):
        assert numpy.fabs(OmS[k] - Om[k]) / numpy.fabs(Om[k]) < 1e-2, (
            "J-star and family frequencies disagree on a benign torus"
        )
    # xvFreqs passes the method through
    out = tm.xvFreqs(
        jr,
        lz,
        jz,
        numpy.array([0.4]),
        numpy.array([0.2]),
        numpy.array([1.0]),
        method="star",
    )
    assert len(out) == 9, "xvFreqs(method='star') did not return 9 outputs"
    # every node outside the family's grid: the informative failure
    tm._freqgrid_frelL = (100.0, 200.0, 300.0)
    tm._torus_cache.clear()
    with pytest.raises(RuntimeError, match="nodes could be"):
        tm.Freqs(jr, lz, jz)
    # the interior-torus guard on the family route
    with pytest.raises(ValueError, match="interior torus"):
        tm.Freqs(0.0, lz, jz)
    return None
